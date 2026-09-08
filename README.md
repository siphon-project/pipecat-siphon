# pipecat-siphon

A [pipecat](https://github.com/pipecat-ai/pipecat) frame serializer for the **siphon-rtp media
WebSocket protocol**.

siphon-rtp is a media engine that can bridge a call leg's audio to an external WebSocket server:
it decodes RTP to linear PCM, streams it up, and encodes PCM coming back down into RTP toward the
caller. Your bot never touches RTP, jitter buffers, or codecs. This package is the small adapter
that lets a pipecat pipeline read and write that wire.

siphon-rtp keeps its own native wire on purpose. Every telephony vendor has a different one, and
serializers exist precisely so an engine does not have to pretend to be somebody else. There is no
Twilio/Telnyx emulation mode here, and there will not be one. The adapter lives on this side, and
it is small: siphon-rtp's binary-L16-plus-JSON-envelope wire is *simpler* than the base64-in-JSON
formats pipecat already ships support for.

Pipecat's contributing guide directs new service and transport integrations to
community-maintained external packages rather than pull requests into the core repository, so this
is the sanctioned shape for an integration like this one.

## Install

```bash
pip install pipecat-siphon
```

It depends on `pipecat-ai>=1.8.0,<2` and nothing else. Python 3.11 or newer, matching pipecat.

## The shape of the integration

**The engine dials out.** siphon-rtp is the WebSocket *client*; your bot is the *server*. So the
example under `examples/` stands up a server and waits, rather than connecting anywhere.

```python
from pipecat.pipeline.pipeline import Pipeline
from pipecat.transports.websocket.server import (
    SingleClientWebsocketServerParams,
    SingleClientWebsocketServerTransport,
)
from pipecat_siphon import SiphonFrameSerializer

transport = SingleClientWebsocketServerTransport(
    host="127.0.0.1",
    port=9001,
    params=SingleClientWebsocketServerParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        serializer=SiphonFrameSerializer(),
    ),
)

pipeline = Pipeline([transport.input(), your_processors, transport.output()])
```

That transport holds one connection for the life of the process, which is the right shape for
`examples/echo_bot.py` and the wrong one for a phone number: the engine dials a fresh WebSocket per
call, so a single-client server is a hard ceiling of one call at a time. For anything that answers
a real number, serve a WebSocket per call and build the pipeline inside the handler --
`examples/agent_bot.py` does exactly that with pipecat's `FastAPIWebsocketTransport`. The
serializer is per call either way; it carries the call's own `start` parameters.

Then point the engine at it, in the `profile` of a native-JSON `offer`:

```json
{
  "id": 1,
  "command": "offer",
  "call_id": "call-7@198.51.100.2",
  "from_tag": "caller",
  "sdp": "v=0\r\n...",
  "profile": { "ws_uri": "ws://127.0.0.1:9001/stream", "ws_vad": true, "ws_barge_in": true }
}
```

`ws_uri` is *takeover*: the WebSocket server is the far side of the call. `ws_tee` (or the
`attach_ws_tee` verb) is a *tee*: the call keeps relaying between two real parties and a copy of
the decoded audio is streamed out, send-only. Both use the same wire, and this serializer handles
both; on a tee it refuses to write audio back, because the engine would not inject it anyway.

## Which callers can reach a bot

A takeover makes the engine the caller's *only* peer, so the engine has to terminate whatever the
caller negotiated and say so in the answer it writes back. `answer_local` writes that answer itself;
`offer`/`answer` rewrites it from the B leg's SDP, and a takeover call has no B leg. So the verb
decides what the caller may be (siphon-rtp 0.3.0):

| Caller's `m=audio` | `offer` + `answer` | `answer_local` |
|---|---|---|
| `RTP/AVP` (plaintext) | supported | supported |
| `RTP/AVP` + ICE | refused (`ws-takeover-ice-offerer`) | supported with `--ice-full`, else refused (`ws-takeover-ice-unsupported`) |
| `RTP/SAVP` + `a=crypto` (SDES-SRTP) | refused (`ws-takeover-secure-offerer`) | supported |
| `UDP/TLS/RTP/SAVPF` (DTLS-SRTP) | refused (`ws-takeover-secure-offerer`) | supported |

On a supported secure takeover the engine mints its **own** keying rather than echoing the caller's,
decrypts the caller's ingress before it becomes the audio your bot hears, and encrypts the downlink
on the way back out. It is fail-closed: nothing leaves in the clear toward a peer that negotiated
encryption. Every refusal comes back at offer time with a stable token at the front of the reason,
so a controller can branch on it without parsing prose. Before 0.3.0 those same offers returned `ok`
and produced a call that answered and bridged nowhere, which is worth knowing if you are moving a
bot off an older engine.

None of this reaches the WebSocket. The bot sees the same `start` envelope and the same L16 frames
whether the caller is a plaintext SIP phone or a WebRTC endpoint.

## Attaching a bot to a call that is already up

`ws_uri` puts a bot on the call at negotiation time and keeps it there until the call ends. From
siphon-rtp **0.4.0** a controller can also do it mid-call, with two verbs:

* **`attach_ws_bridge`** (`call_id`, `from_tag`, `ws_uri`) on a call that has no bridge takes a live
  two-party relay over: leg A's audio goes to your bot and the A↔B path is unwired, so the other
  party hears nothing until it is detached. On a call that already has one it **re-points** it at a
  different server without renegotiating — codec, wire rate, VAD, echo cancellation, source gate,
  SRTP keying and ICE selection all carry across, so there is no re-INVITE and nothing reverts to a
  default.
* **`detach_ws_bridge`** puts the two parties back together, reinstalling the exact forward rules the
  takeover displaced.

For a pipecat deployment that is the difference between "the bot answers the call" and "the bot joins
a call in progress" — an agent taking over from a human, a supervisor handing a caller to a bot, or a
bot swapped out mid-conversation. From this side of the socket nothing changes: your server gets the
same `start` envelope and the same audio, and this serializer does not care which verb put it there.

The same release added **`ws_bridge_started` / `ws_bridge_ended`** events. The end reason
(`detached`, `server_closed`, `server_stopped`, `call_ended`, `transport_error`) is how a controller
finds out the bot's socket died — worth wiring up, because a takeover bridge is the caller's *only*
far side, so a bot that goes away is a live call with nobody on it.

## The wire

Two frame kinds share one socket:

* **binary** frames carry raw audio in the negotiated format, one frame per ptime, no base64;
* **text** frames carry a `{"type": "<snake_case tag>", "data": {...}}` control envelope whose
  `data` object uses camelCase field names.

The first text frame is always `start`:

```json
{
  "type": "start",
  "data": {
    "streamId": "ws-call-7@198.51.100.2",
    "callId": "call-7@198.51.100.2",
    "direction": "duplex",
    "media": {
      "encoding": "L16", "sampleRate": 8000, "channels": 1,
      "bitDepth": 16, "endianness": "little", "ptime": 20
    }
  }
}
```

## Sample rates: the wire rate is negotiated, and authoritative

`media.sampleRate` in the `start` envelope is **the wire rate**, in both directions, for the life of
the stream, and it is authoritative. Never assume 8000. Frame against the number `start` gives you,
and expect it to differ from what your pipeline runs at.

From siphon-rtp **0.3.0** that rate is selectable independently of the call's codec, so an 8 kHz
G.711 caller can stream 16 kHz L16 to a bot that wants wideband input:

| Knob | Where it goes |
|---|---|
| `ws_sample_rate` | the takeover profile, alongside `ws_uri`. Applied in both directions: the engine resamples the leg's uplink into it, and resamples the bot's downlink back into the codec's rate before re-encoding. |
| `ws_tee_sample_rate` | the tee profile, alongside `ws_tee`. Every tapped leg is resampled into it, so a stereo tee across two different codecs still produces one coherent stream. |
| `sample_rate` | the `attach_ws_tee` verb, for a tee attached after the call is up. |

Valid rates are multiples of 1000 from 8000 to 48000. Anything else fails the offer, answer or
attach with a typed error before the call is dialled — the engine never silently clamps to a rate
you did not ask for. Leave the knob out and the wire follows the leg's codec rate (8000 for G.711,
16000 for G.722/AMR-WB) with no conversion built at all, exactly as before 0.3.0.

**Set it to whatever your pipeline runs at.** The conversion then happens in the engine instead of
here, which is the better place for it: siphon-rtp resamples frame by frame at the RTP boundary with
nothing held back, while pipecat's SOXR stream resampler at VHQ swallows the first few chunks and
then delivers in bursts. Same audio, steadier timing, and one less stage between the caller and the
model.

On 0.2.1 and earlier the knobs do not exist. An older engine *ignores* the field rather than
rejecting it, so you get the codec rate back with nothing to point at — check the engine version,
or just read the rate out of `start`, which is right on every build.

This serializer handles the mismatch for you. `setup(StartFrame)` records the pipeline's
`audio_in_sample_rate`; the wire rate arrives later, in `start` (the engine cannot dial you before
you are listening, so the ordering is always pipeline-first). Every frame is then resampled
between the two with pipecat's own `create_stream_resampler`, in both directions, and a
`media_renegotiate` mid-stream moves the wire rate without restarting anything. When the rates
already match, nothing is resampled and the bytes pass through untouched.

If you want to fix the wire rate before `start` arrives, set `wire_sample_rate` in the params;
`start` still wins the moment it lands.

## Stereo and track separation

A tee with `channels: 2` interleaves the two legs: **channel 0 is the caller, channel 1 is the
callee**. The `stereo_input` param decides what the pipeline sees:

| `stereo_input` | Result |
|---|---|
| `"caller"` (default) | Mono, channel 0 only. Keeps the far party out of your ASR. |
| `"callee"` | Mono, channel 1 only. |
| `"mix"` | Mono, the two legs summed at half amplitude each. |
| `"interleaved"` | A 2-channel `InputAudioRawFrame`, each channel resampled on its own. |

Outbound, if the wire is stereo the bot's mono audio is written to both channels: a bot has one
voice, and silencing a channel would be a stranger default than duplicating it.

## Message matrix

### Engine → pipecat (`deserialize`)

| Wire message | Pipecat frame | Notes |
|---|---|---|
| binary audio | `InputAudioRawFrame` | Resampled to the pipeline rate; L16/PCMU/PCMA; endianness honoured; stereo split per `stereo_input`. |
| `start` | *(none)* | Captures `streamId`, `callId`, `direction`, `tracks` and the media format. Readable afterwards on `.stream_id` / `.media_format` / `.wire_sample_rate` / `.direction` / `.tracks`. |
| `media_renegotiate` | *(none)* | Replaces the wire format mid-stream. |
| `speech_started` | `VADUserStartedSpeakingFrame` | Default (`speech_frames="vad"`). Pipecat's default `VADUserTurnStartStrategy` turns it into a user turn start *and* an interruption, so the engine's VAD (`ws_vad`) replaces a local one with no extra wiring. With `speech_frames="user"` it becomes `UserStartedSpeakingFrame` instead, for `ExternalUserTurnStartStrategy`. |
| `speech_stopped` | `VADUserStoppedSpeakingFrame` | The turn endpoint, past the VAD hangover. `UserStoppedSpeakingFrame` in `"user"` mode. |
| `dtmf` | `InputDTMFFrame` | `0`–`9`, `*`, `#`. See the unmapped list below for `A`–`D`. |
| `mark` | `BotStoppedSpeakingFrame` | A playout boundary was rendered, or skipped on a `clear` (the engine answers a `clear` with a mark named `cleared`). Disable with `emit_bot_speaking_frames=False`. |
| `stop` | `EndWorkerFrame` | The pipeline worker converts it into a pipeline-wide `EndFrame`, so the transports shut down in order. The engine's `reason` is carried through. |
| `error` (`fatal: true`) | `CancelWorkerFrame` | Logged at `error`, then the pipeline is cancelled. The engine closes the socket after a fatal error, so this has to be the frame that ends things. |
| `error` (`fatal: false`) | `ErrorFrame` | Logged at `error` and surfaced; the stream continues. |
| `event` | *(none)* | Logged at `debug`. Opaque passthrough, so there is no frame it maps to. |
| `play_start`, `play_stop`, `clear` | *(none)* | Server-to-engine verbs. If one arrives *from* the engine the peer is not siphon-rtp, so it is logged and ignored rather than acted on. |
| malformed / truncated | *(none)* | Logged at `warning` and dropped. The socket is untrusted input; `deserialize` never raises. |

### Pipecat → engine (`serialize`)

| Pipecat frame | Wire message | Notes |
|---|---|---|
| `OutputAudioRawFrame` (any `AudioRawFrame`) | binary audio | Resampled to the wire rate, folded to mono or duplicated to stereo as the wire requires, byte-swapped for a big-endian wire, encoded to PCMU/PCMA if that is what was negotiated. Dropped on a send-only (tee) stream. |
| `InterruptionFrame` | `clear` | Barge-in: flush anything the engine still has queued for playout. `reason` from `clear_reason`, default `barge_in`. |
| `OutputTransportMessageFrame`, `OutputTransportMessageUrgentFrame` | `event` | A `{"name": ..., "payload": ...}` message keeps its name; anything else is wrapped under `event_name`. RTVI messages are filtered out by the base class unless you set `ignore_rtvi_messages=False`. |
| `EndFrame`, `CancelFrame` | `stop` | Carries the frame's own `reason` when it has one, else `stop_reason`. |
| everything else | *(none)* | Not sent. |

### Deliberately unmapped

* **`A`–`D` DTMF digits.** RFC 4733 defines events 12–15 as the A–D digits and the engine will
  report them, but pipecat's `KeypadEntry` only models `0`–`9`, `*` and `#`. Nothing downstream
  could consume an invented frame, so those digits are logged at `debug` and dropped.
* **`play_start` / `play_stop`.** Downlink audio needs no announcement on this wire: binary frames
  are enough, and the engine's v1 takeover bridge rejects an inline (base64) `play_start` with an
  `error` anyway. There is no pipecat frame that means "start a named playback segment", so
  nothing is emitted and nothing is consumed.
* **`event` inbound.** An opaque application payload has no pipecat frame it corresponds to; it is
  logged rather than guessed at.
* **`FrameSerializer.type` / `FrameSerializerType`.** Older pipecat releases had a `type` property
  on serializers. It does not exist in pipecat 1.8 — the transport decides binary versus text from
  whether `serialize` returned `bytes` or `str` — so this package does not define one.

## Turn taking and barge-in

Engine profile flags move the turn work into the media engine, which is usually where you want it
for telephony:

| Flag | What it does |
|---|---|
| `ws_vad` | Emit `speech_started` / `speech_stopped` on the caller's speech edges. |
| `ws_barge_in` | Flush queued playout the instant the caller talks over the bot, with no server round-trip. Implies `ws_vad`. |
| `ws_vad_engine` | `"energy"` (default) or `"neural"` — which detector runs. New in 0.3.0. |
| `ws_vad_min_speech_ms` | A **leading** run: the uplink must read as speech continuously for this long before the start edge (and barge-in) fires. New in 0.3.0. |
| `ws_vad_threshold`, `ws_vad_hangover_ms` | Tune the energy detector specifically: the mean-square energy that counts as speech, and how long speech is held after it drops before `speech_stopped`. |
| `echo_cancellation` | Cancel the bot's own audio coming back through the caller's handset. |

With `ws_vad` on and the default `speech_frames="vad"`, the engine's VAD drives pipecat's turn
machinery directly and you do not need a local VAD in the pipeline at all — no
`SileroVADAnalyzer` on the transport, no second copy of the audio being classified.

Which detector: the energy gate answers "is something loud here", so as a *turn* detector it fires
on mains hum, breathing and fan noise. `"neural"` runs a Silero v5 forward pass inside the engine
and answers "is what is here speech" instead, for about 27 µs per 20 ms frame on an 8 kHz leg, with
a 32 ms detection floor from the network's own window. For a conversational bot, `"neural"` plus
`ws_vad_min_speech_ms` somewhere in 60–120 ms is the pairing that stops a cough, a door or one burst
of echo cutting the bot off mid-sentence. A detector the engine cannot build for the leg fails the
offer rather than quietly falling back to the one you were avoiding.

`echo_cancellation` is not optional once barge-in is on. Echo of speech is speech to any detector,
the neural one included, so on a handsfree or loudspeaker endpoint the bot's own voice returning up
the caller's uplink reads as the caller interrupting.

**If you are pinned to siphon-rtp 0.3.0, upgrade.** That release counted the energy detector's
trailing hangover in milliseconds rather than in ptime frames, so `speech_stopped` — and the
`VADUserStoppedSpeakingFrame` this serializer emits from it — landed twenty times too late at a
20 ms ptime, i.e. never inside a normal turn, while `speech_started` and barge-in kept working and
made turn taking look healthy. Fixed in 0.3.1.

Put together, the profile for a conversational bot on 0.3.0 — wideband wire on a narrowband call,
engine-side turn taking, no local VAD:

```json
"profile": {
  "ws_uri": "ws://198.51.100.10:9001/stream",
  "ws_sample_rate": 16000,
  "ws_vad": true,
  "ws_barge_in": true,
  "ws_vad_engine": "neural",
  "ws_vad_min_speech_ms": 100,
  "echo_cancellation": true
}
```

If you would rather drive the turn explicitly, set `speech_frames="user"` and wire
`ExternalUserTurnStartStrategy` / `ExternalUserTurnStopStrategy`. That strategy is constructed with
interruptions disabled, so `auto_interrupt=True` (the default) also emits an
`InterruptionWorkerFrame` after the `UserStartedSpeakingFrame`. Pipecat's serializer API hands the
transport one frame per WebSocket message, so the second frame is queued and drains on the next
message — at most one ptime later, since the engine's takeover ticker never stops sending.

Barge-in in the other direction is automatic: any `InterruptionFrame` in the pipeline becomes a
`clear` on the wire.

## Params

All of these live on `SiphonFrameSerializer.InputParams`, which extends pipecat's own
`FrameSerializer.InputParams` (so `ignore_rtvi_messages` and `resampler_clear_after_secs` are there
too).

| Param | Default | Meaning |
|---|---|---|
| `sample_rate` | `None` | Override the pipeline input rate instead of taking `StartFrame.audio_in_sample_rate`. |
| `wire_sample_rate` | `None` | Wire rate assumed before `start` arrives. `start` always wins afterwards. |
| `wire_ptime` | `None` | Packetization time assumed before `start`. |
| `speech_frames` | `"vad"` | `"vad"`, `"user"` or `"none"`. See the matrix above. |
| `auto_interrupt` | `True` | With `speech_frames="user"`, also emit an `InterruptionWorkerFrame` on `speech_started`. |
| `emit_bot_speaking_frames` | `True` | Emit `BotStoppedSpeakingFrame` on `mark`. |
| `stereo_input` | `"caller"` | `"caller"`, `"callee"`, `"mix"` or `"interleaved"`. |
| `clear_reason` | `"barge_in"` | `reason` on the `clear` an interruption produces. |
| `stop_reason` | `"pipeline_ended"` | `reason` on the `stop` an `EndFrame`/`CancelFrame` produces when the frame has none. |
| `event_name` | `"pipecat"` | `name` for an `event` built from an unnamed transport message. |

## Example

`examples/echo_bot.py` is the minimum that proves the path: a WebSocket server, this serializer,
and a processor that echoes the caller's audio back into the call. Run it and point an engine at
it:

```bash
python examples/echo_bot.py --host 127.0.0.1 --port 9001
```

Then offer a call with `"profile": {"ws_uri": "ws://127.0.0.1:9001/stream"}` and you hear yourself
back, one jitter-buffer frame plus one playout frame later.

`examples/agent_bot.py` is that same path with the echo processor replaced by an STT → LLM → TTS
chain, so a caller talks to a model instead of to themselves. It needs three API keys in the
environment and `pip install "pipecat-ai[anthropic,deepgram,cartesia]"`; the vendors are one line
each to swap, and the file explains which knobs are there for latency rather than for taste. The
turn taking is the engine's, not the pipeline's — there is no VAD analyzer in it at all.

Given `--control-url`, it also connects to siphon's control plane and gets an `end_call` tool, so
the model can hang up when the conversation is done rather than emitting a magic phrase for
something else to notice. The example's docstring covers the platform side: the media profile the
engine's built-in `voice_ai` does not fully supply, and the correlation detail that decides
whether the control channel ever finds its call — the engine expands `{call_id}` in a `ws_uri` to
the **SIP** Call-ID, so the two channels join on `sip_call_id`, not on the control frame's own
`call_id`.

`examples/probe.py` speaks the engine's half of the media wire — connect, `start`, stream silence
— and reports how much audio comes back and how long the first frame took. No phone, no SIP
stack, no engine. It is the fastest way to tell "the AI is broken" from "the media path is
broken", because every failure above the media path looks identical from outside: a socket that
connects, a bot that reports healthy, and silence.

```bash
python examples/probe.py ws://127.0.0.1:9001/stream
```

## Quickstart: a real phone number

[`examples/quickstart/`](examples/quickstart/) is the whole thing end to end: siphon-sip registers
to a SIP provider, inbound calls on that registration are handed to the bot, and the bot can hang
up. Nothing in it is provider-specific — registration is RFC 3261, and every credential comes from
the environment.

It runs as three containers, so a clone, a `.env` and one command is the whole setup:

```bash
cd examples/quickstart
cp .env.example .env      # three API keys, the SIP account, the address the provider can reach
docker compose up --build
```

There is a by-hand path for the three processes too, and a probe that tells you whether the bot
answers with audio before you spend a phone call finding out. See
[`examples/quickstart/README.md`](examples/quickstart/README.md).

## Integration harness

`integration/` holds a four-way harness that runs the whole chain with nothing mocked: SIPp
places a call through siphon-sip, siphon-rtp bridges the leg to a pipecat bot using this
serializer, and the assertions are on the audio that comes back rather than on the socket being
open. Three scenarios: a known signal survives the round trip, checked spectrally against a tshark
capture; the engine's VAD edges land where the fixture puts them and barge-in flushes the bot's
queued speech; and the same call again with `ws_sample_rate: 16000`, where the wire runs at 16 kHz
over an 8 kHz G.711 leg and the audio has to come back at the pitch it went out at.

```bash
cd integration && ./run.sh
```

It needs Docker and nothing else: siphon-sip and siphon-rtp are pulled as their published
release images, so this runs on a fresh clone. Point `SIPHON_RTP_PATH` / `SIPHON_SIP_PATH` at a
checkout to build either from source instead. Not part of CI yet; see
[`integration/README.md`](integration/README.md).

## Compatibility

| | Verified against |
|---|---|
| pipecat | `pipecat-ai` 1.8.1 |
| siphon-rtp | 0.4.x, end to end through the harness under `integration/`. The bridge protocol lives in `crates/siphon-rtp-media/src/bridge/protocol.rs`, and that file has not been touched since before 0.2.0 — its diff across every release since is empty — so the same bytes work against 0.2.x, 0.3.x and 0.4.x alike. What those releases add is surface a *controller* uses: the wire-rate and detector knobs (0.3.0), callers a takeover can terminate (0.3.0), and the attach/detach bridge lifecycle (0.4.0). None of it changes the wire this package speaks. |
| siphon-sip | 1.7.0+ for the harness, which is the release that carries `ws_sample_rate` and the other 0.3.0 profile flags through to the engine. Not a dependency of this package — the serializer never sees the signalling side. |
| Python | 3.13 (declared support 3.11+, matching pipecat's floor) |

The control-frame fixtures under `tests/wire_fixtures.py` are byte-exact strings produced by
compiling the engine's own Rust definition against `serde_json` — not by this package's encoder.
A shared bug on both sides of an encode/decode pair sails straight through a round-trip test; it
does not survive a comparison against bytes another implementation produced.

Note on version skew: an unrecognised control `type` is dropped with a warning rather than
raising, so a bot pinned to this release keeps working against a newer engine that adds a message.

## Development

```bash
uv venv --python 3.13 .venv
uv pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
```

## License

MIT. See [LICENSE](LICENSE).
