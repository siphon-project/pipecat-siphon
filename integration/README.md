# Four-way integration harness

A real call, carrying real audio, through real components, with the assertions on the audio
rather than on the connection.

```
  SIPp  ──SIP/SDP──▶  siphon-sip  ──JSON control──▶  siphon-rtp
   │                  (B2BUA)      (via the tap)     (media engine)
   │                                                      │
   └───────────────────── RTP (G.711 A-law) ──────────────┤
                                                          │ media WebSocket
                                                          ▼
                                             pipecat bot + SiphonFrameSerializer
```

Everything in that picture is the real thing:

| Component | What runs |
|---|---|
| Caller | `sipp` 3.7 playing a generated RTP capture with `play_pcap_audio` |
| B2BUA | siphon-sip, built from the checkout beside this repository, answering with `rtpengine.answer_local` |
| Media engine | siphon-rtp, built from the checkout beside this repository, UDP datapath |
| Bot | a pipecat `Pipeline` behind `SiphonFrameSerializer` from `src/`, on pipecat's own WebSocket server transport |
| Capture oracle | `tshark`, reading the RTP back out of a capture taken on the caller's own interface |

There is one extra hop that is not part of the product: a **control tap** between the proxy and
the engine. It forwards every byte in both directions unchanged and decodes a copy into
`artifacts/control-transcript.jsonl`, because neither siphon-sip nor siphon-rtp logs the JSON
bodies they exchange and "what did the proxy actually ask for" is the first question a red run
raises. It answers nothing and rewrites nothing; if it dies the run fails.

## Run it

```bash
cd integration
./run.sh
```

That builds what needs building, brings the stack up, runs both scenarios, tears it down, and
prints pass/fail per scenario. Everything a failure needs is left in `artifacts/`.

```bash
./run.sh --scenario roundtrip     # one scenario, while iterating
./run.sh --no-build               # reuse the images exactly as they are
./run.sh --keep-up                # leave the stack running to poke at it
./run.sh --no-clean               # keep the previous run's artifacts
```

Requirements: Docker with the compose plugin, and checkouts of siphon-sip and siphon-rtp beside
this repository. Point `SIPHON_SIP_PATH` / `SIPHON_RTP_PATH` elsewhere if yours are not there.
Nothing else is needed on the host: SIPp, tshark and the analyser all live in the UAC image.

The first build compiles both Rust projects and takes a while. After that the layer cache makes
it seconds. `CARGO_BUILD_JOBS` defaults to 4 rather than to whatever cargo picks, because a
fresh checkout building jemalloc-sys at full parallelism is a known way to lose a build.

## Scenario 1 — audio round trip

The caller plays 4 seconds of a two-tone fixture (500 Hz + 1300 Hz, G.711 A-law, 20 ms
packets). The bot echoes every uplink frame back down with a **2400 Hz marker tone** mixed in.
The harness then asserts on the RTP that reached the caller:

| Assertion | What it rules out |
|---|---|
| The 200 OK's `c=` line is the engine's relay address | An answer anchored on the engine's loopback default, where the RTP would go nowhere |
| The `start` envelope announced L16 / 8000 Hz / mono / 20 ms / duplex, and every uplink frame is exactly 320 bytes | A bridge whose frame geometry does not match the negotiated ptime, which plays back at the wrong speed rather than failing |
| ~200 downlink RTP packets, one payload type, one SSRC, no sequence gaps, ~50 packets/second | Audio that arrives in bursts, from two sources, or with holes |
| **500 Hz and 1300 Hz are present in the decoded downlink, measured by Goertzel against an empty 3300 Hz control bin** | A bridge that is up and passing zeros, comfort noise, or somebody else's audio. This is the assertion that a connectivity check cannot make |
| **2400 Hz is present too** | The audio having been looped back inside the engine rather than travelling through the bot. The caller never emits 2400 Hz, and a single-leg `answer_local` has no second party and no relay path, so there is nowhere else it could come from |
| `siphon_rtp_sessions == 0`, `deletes_total >= 1`, `control_errors_total == 0` after the call | A leaked session, a delete the proxy never sent, and (the valuable one) a command the engine rejected while the call flow completed 200/ACK/BYE looking perfect |

The spectral check is a ratio against a bin the fixture never excites, not an absolute level.
"There is energy at 500 Hz" is satisfied by noise; "there is energy at 500 Hz and 13 dB more of
it than at 3300 Hz" is not.

Measured on the development machine: 500 Hz at **18 600x** the control bin, 1300 Hz at
**18 700x**, the bot's marker at **4 800x**, against a required 20x. 200 uplink frames in, 200
downlink packets out, zero sequence gaps.

## Scenario 2 — turn taking and barge-in

The engine's VAD and barge-in are switched on **in the media profile** (`ws_vad`,
`ws_barge_in`, `ws_vad_threshold`, `ws_vad_hangover_ms`, see `proxy/siphon.yaml`), not left to
the engine's defaults, so the assertion windows are derived from numbers the harness states.

The bot starts talking the moment the stream opens and queues twenty seconds of speech. The
caller's fixture is 2 s of silence, a 1.5 s speech burst, then 1.5 s of silence.

| Assertion | Measured |
|---|---|
| `speech_started` fires on the frame the fixture's speech begins on (the engine's VAD has no onset gate, so the edge should be immediate) | **0 frames / 0 ms** late |
| `speech_stopped` fires at `offset + hangover/ptime + 1` frames, i.e. 320 ms after the last loud frame at a 300 ms hangover | exactly the predicted frame, **320 ms** after the burst |
| The interruption travels the whole chain: serializer → pipecat turn processor → bot and output transport → `clear` → the engine's `mark` named `cleared` | acknowledged **1 frame** after `speech_started` |
| **The audio returning to the caller stops within 250 ms of the caller starting to talk** | **11–19 ms** across runs |
| Nothing audible comes back after the budget expires, over a ≥ 1 s observation window | 0 packets, ~2.9 s window |
| The bot had a real backlog to throw away (frames generated minus frames actually delivered as RTP ≥ 250) | **902 frames ≈ 18 s** discarded |
| The bot was continuously audible for ≥ 1 s beforehand | ~1.94 s |
| Engine teardown, as scenario 1 | clean |

The backlog check is what stops the barge-in assertion being vacuous. "The audio stopped" is
also true of a bot that ran out of things to say; "the audio stopped 19 ms in while eighteen
seconds of it were still queued" is not.

The two flush paths are both real and both exercised. The engine flushes its own playout queue
on the VAD edge without a round trip (`ws_barge_in`), which is what the 19 ms comes from, and
the serializer's `InterruptionFrame` → `clear` round trip is asserted separately through the
engine's `mark` reply. Neither one alone would prove the other works.

## Why the timings are trustworthy

Every turn-edge measurement is expressed in **uplink frames**, not seconds. One uplink binary
frame is one ptime of caller audio, so frame *n* is *n × 20 ms* into the stream regardless of
how loaded the machine was. The bot records the index and the energy of every frame it
receives; the analyser finds the fixture's speech burst by where the energy steps up and
measures the VAD edges relative to that. Nothing in the edge assertions reads a clock.

The barge-in latency is the one measurement in wall-clock time, and both ends of it come from
**the same capture**, so it is a difference between two timestamps taken by one observer, with
no cross-container clock comparison.

Nothing in the harness sleeps as a form of synchronisation. Every wait goes through
`tools/wait_for.py`, which polls an observable condition (an HTTP 200, an accepted TCP
connection, a capture file opening, a record appearing in the bot's trace, a metric reaching a
value) and gives up after a stated maximum with a message saying what it was waiting for and
what it last saw.

## What is in `artifacts/` after a run

| File | |
|---|---|
| `<scenario>.pcap` | the generated fixture, as SIPp played it |
| `<scenario>-capture.pcap` | everything the caller's interface saw, signalling and media |
| `<scenario>-bot.jsonl` | the bot's trace: every uplink frame with its energy, every control event, indexed by frame |
| `<scenario>-analysis.json` | every measurement the analyser took, pass or fail |
| `<scenario>-sipp-{messages,errors,screen}.log` | SIPp's own record of the dialogue |
| `control-transcript.jsonl` | every control command and reply between the proxy and the engine, decoded |
| `engine.log`, `proxy.log`, `bot-*.log`, `control-tap.log` | container logs, collected on the way out even when the run fails |

Re-run just the analysis over artifacts you already have, without touching the stack:

```bash
docker compose run --rm --entrypoint python3 uac -m tools.analyse roundtrip \
    --artifacts /harness/artifacts
```

## Should CI gate on this?

**Not yet. Run it manually or nightly.** It is deliberately not wired into a workflow.

In its favour: six consecutive full runs (twelve scenario executions) passed with no flakes, a
warm-cache run takes about a minute for both scenarios, and the measurements are tight and
repeatable (the VAD edges landed on exactly the predicted frame every time; the barge-in
latency varied between 11 and 19 ms against a 250 ms budget).

Against it, and the reason it stays manual:

* **It builds two Rust projects from sibling checkouts.** A GitHub runner has neither, so
  gating on it means pinning published images instead, at which point the harness stops
  testing the engine you are working on, which is most of the point.
* **The cold build is long.** Warm it is fast; cold it is two Rust compilations.
* **It needs `NET_ADMIN`/`NET_RAW`** for the capture, which not every runner allows.

If it is ever gated, gate it on a self-hosted runner with the sibling checkouts present, as a
nightly job rather than per-PR, and keep the artifact upload, because the numbers are only
useful if you can read them after a red run.

## Type checking and linting

`ruff check`, `ruff format --check` and `mypy --strict` cover `integration/` exactly as they
cover the package. There is **one exclusion**, declared in `pyproject.toml`:
`integration/proxy/harness_script.py` is skipped by mypy because it is a siphon-sip *call
script*: it runs inside that daemon's embedded interpreter, against a `siphon` module that
exists only in that process, and its handler signatures are fixed by that API. Everything else
here is checked strictly.

## Things worth knowing before you change it

* **The engine dials out; the bot is the server.** The dial happens inside the answer command,
  and a refused connection makes the engine tear the half-built call down and fail the command.
  So the bots have to be listening before the call is placed, which is why `prepare.sh` waits
  on their sockets rather than on a timer.
* **One WebSocket binary message becomes exactly one RTP packet**, whatever its length, and the
  RTP timestamp still advances by one ptime. Pipecat's default `audio_out_10ms_chunks=4` would
  put 40 ms of audio in every packet and play the bot back at double speed; the bot sets it to
  2. This is the single easiest way to break the harness invisibly.
* **The wire rate is not selectable.** It is the negotiated codec's native decoder rate, so a
  G.711 leg gives 8 kHz. The bot's pipeline runs at 8 kHz too, which means no resampler runs
  anywhere, because pipecat's SOXR resampler short-circuits when the rates match. That is
  deliberate:
  at VHQ it buffers heavily (the first chunks of a conversion return nothing and output then
  arrives in bursts), which would blur the barge-in timing. If you change either rate, the
  barge-in budget needs revisiting.
* **`answer_local` reads the media profile's `answer:` block**, not `offer:`. The YAML anchors
  in `proxy/siphon.yaml` make both sides the same object so a flag cannot be set on the half
  that is never read.
* **Unknown keys in siphon-sip's YAML are ignored silently.** A typo in a profile flag does not
  fail the boot; it just does nothing. This is why both scenarios assert on behaviour the flags
  produce rather than trusting that they arrived: scenario 2 could not see a turn edge without
  `ws_vad`, and scenario 1 could not echo anything back if `ws_barge_in` were on.
* **Echo cancellation and noise suppression are off** in both profiles. AEC would take the
  bot's echo as the far-end reference and subtract the caller's own tone out of the uplink,
  which is exactly the signal scenario 1 measures.
* **tshark drops privileges before creating its output file** and so cannot write into the
  bind-mounted artifacts directory. The capture goes to container-local storage and is copied
  out when it stops, including on failure.
* **A takeover call has no B leg** and never will. Do not add an assertion that expects one.
  Relatedly, `proxy.log` carries a `B2BUA: no winner set for BYE` warning on every call. That is
  siphon-sip noting there is no B leg to pick a winner from on a single-leg answer, and it is
  expected here.
