# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-09-05

Nothing in the package itself changed: `src/pipecat_siphon` is byte-identical. Everything below is
the examples, which is where the defects were — four of them silent, each one presenting as a bot
that connects, reports healthy and never speaks.

### Fixed

- **The greeting ran against an empty context**, so every call opened with an HTTP 400
  (`messages: at least one message is required`) and the bot stayed mute until the caller said
  something. It reads as a dead line, which sends people to the media path. The context is now
  seeded with a bracketed stage direction. Seeding also *clears* the previous call's transcript,
  which matters now that the server outlives a call.
- **`fallbacks` and `betas` were passed through the Anthropic service's `extra`** and could never
  work. Pipecat does call the beta endpoint, but overwrites `betas` with its own interleaved
  thinking flag *after* merging `extra`, so `fallbacks` arrived without the beta that would make
  it legal and every turn 400'd, greeting included. Both keys are dropped, with the constraint
  named in a comment; fixing it properly means merging `betas` upstream in pipecat.
- **`--fast` could not have worked** for the same reason — fast mode needs its beta flag on the
  request, and that flag was being overwritten — so the option is gone rather than left in
  `--help` promising something it never delivered.
- **Output frames ignored the negotiated ptime.** Pipecat's transport defaults to four 10 ms
  chunks, so both examples sent 40 ms per frame against a 20 ms wire. An engine that stamps each
  received frame as one ptime then advances the RTP timestamp at half the rate of the audio and
  every packet overlaps its predecessor: impeccable packet by packet, silent on the handset.
  Recent engine builds drain by samples and rescue it, but one frame per ptime is what the
  protocol asks for and it is the lower-latency shape.
- **The pipeline worker cancelled itself after five idle minutes**, taking the runner and the
  listening socket with it, so every call after the first five-minute gap was refused at the TCP
  connect. These are servers the engine dials into; idling between calls is the normal state.
  `idle_timeout_secs=None` in both examples.
- **`output_config.effort` was a free-standing constant beside `MODEL`.** It is valid on the
  pinned model and rejected outright by others, so switching models for latency produced a 400 on
  every turn. Effort is now bound to the model, and an unlisted model sends none.
- The serializer's wire rate is stated up front rather than left to be discovered from `start`,
  so no resampler is built for the handful of frames before it arrives and then discarded.

### Added

- **`examples/quickstart/`** — the whole path end to end: siphon-sip registers to a SIP provider,
  inbound calls on that registration are handed to the bot, and the bot can hang up. Config,
  routing script and a README of what goes wrong. Provider-neutral; credentials come from the
  environment.
- **A control plane in `examples/agent_bot.py`**, given `--control-url`: the model gets an
  `end_call` tool that says goodbye, waits for the words to actually finish playing, and then
  drops the line. The tool takes no arguments on purpose — it acts on the call the bot is already
  on, and an argument would be a model-chosen string reaching a verb that can hang up a stranger.
  The control connection retries with backoff, so starting the bot before the engine is a routine
  race rather than an exit.
- **`examples/probe.py`** — opens the media WebSocket, sends `start`, streams silence and reports
  how much audio comes back and how soon. No phone, no SIP stack, no engine; it separates "the AI
  is broken" from "the media path is broken" in seconds.
- **The platform side is documented**, in the example's docstring and the quickstart config. The
  engine's built-in `voice_ai` profile sets neither `received_from` (without which a NATed caller
  is gated out entirely, on a call whose signalling is clean throughout) nor `ws_vad_engine` /
  `ws_vad_min_speech_ms` (without which barge-in fires on a cough or on one burst of the bot's own
  echo).
- Tests for the example's own logic — the farewell wait and the media/control correlation — with
  the control call faked, so no Rust toolchain enters the CI path.

## [0.2.0] - 2026-09-04

The first release published to PyPI. `0.1.0` was the initial code drop and never left the
repository, so `pip install pipecat-siphon` starts here.

### Changed

- Moved to `pipecat-ai` 1.8, and the floor with it (`>=1.8.0,<2`). 1.8.0 moved the pipeline's
  sample rates out of `StartFrame` and into the `FrameProcessorSetup` a transport hands every
  processor, deprecating the old reads; `SiphonFrameSerializer.setup()` now takes that object,
  which is what the transport was already passing. The non-fatal `error` mapping also stops
  passing `ErrorFrame(fatal=False)`, another 1.8 deprecation and the default anyway -- a fatal
  engine error still becomes a `CancelWorkerFrame`, which is what pipecat now recommends instead
  of a fatal frame. The wire, the frame mapping and every parameter are unchanged.

- Documentation aligned to the siphon-rtp 0.3.x/0.4.x line. The bridge control envelope is
  untouched across all of it -- `crates/siphon-rtp-media/src/bridge/protocol.rs` has not changed
  since before 0.2.0 -- so the serializer, the protocol module and the engine-emitted wire fixtures
  are unaffected and there is no code change here. What did change is the surface a *controller*
  drives, and the README now says so: the selectable wire rate (`ws_sample_rate` /
  `ws_tee_sample_rate`) is released rather than unreleased, with the advice to set it to the
  pipeline's own rate so the one conversion in the path is the engine's; the turn-taking flags gain
  `ws_vad_engine` (`energy` / `neural`) and `ws_vad_min_speech_ms`; a new section covers which
  callers a takeover can terminate now that `answer_local` handles SDES-SRTP, DTLS-SRTP and full
  ICE, with the stable refusal tokens for the shapes it cannot; and another covers the 0.4.0
  takeover-bridge lifecycle, where `attach_ws_bridge` / `detach_ws_bridge` can put a bot on a call
  that is already up, move it to a different bot, or hand the two parties back to each other, and
  `ws_bridge_ended` finally tells a controller when the bot's socket died.

### Added

- `examples/agent_bot.py`, a conversational demo bot: the caller's speech goes to a streaming
  recognizer, the transcript to Claude, and the reply to a streaming voice, with the turn taking
  left to the engine (`ws_vad` / `ws_barge_in`) rather than to a VAD analyzer in the pipeline. It
  runs the pipeline at the negotiated wire rate, so neither side of the socket resamples.

- The harness runs on a fresh clone. siphon-sip and siphon-rtp default to their published
  release images rather than to builds of checkouts beside this repository, so `./run.sh` needs
  Docker and nothing else; only the bot and the UAC are built, from this repository. Naming a
  checkout (`SIPHON_RTP_PATH` / `SIPHON_SIP_PATH`) still builds that component from source, which
  is what you want when the engine change is yours, and `SIPHON_RTP_IMAGE` / `SIPHON_SIP_IMAGE`
  pin a different release without one.

- A four-way integration harness under `integration/`: SIPp to siphon-sip to siphon-rtp to a
  pipecat bot using this serializer, in compose, with no mocks in the path. Three scenarios, all
  asserting on audio rather than on connectivity -- a Goertzel check on the RTP that returns to
  the caller for the round trip, VAD edge timing plus barge-in flush latency for turn taking, and
  a wideband run that negotiates a 16 kHz wire over an 8 kHz G.711 leg with `ws_sample_rate` and
  checks the audio comes back at the pitch it went out at, which is the case this serializer's
  rate reconciliation exists for. Not wired into CI; see `integration/README.md` for why.

## [0.1.0] - 2026-08-20

First release.

### Added

- `SiphonFrameSerializer`, a `pipecat.serializers.base_serializer.FrameSerializer` for the
  siphon-rtp media WebSocket protocol: binary L16/PCMU/PCMA audio frames plus the
  `{"type", "data"}` JSON control envelope, in both directions.
- `pipecat_siphon.protocol`, a dependency-free transcription of the engine-side message
  definitions, byte-exact against the engine's own `serde` output (field order, camelCase keys,
  omit-when-absent optionals).
- Negotiated-wire-rate handling: the `start` envelope's `sampleRate` is authoritative and is
  reconciled against the pipeline's rates with pipecat's own streaming resampler, in both
  directions, including a `media_renegotiate` mid-stream.
- Stereo tee support: channel 0 caller, channel 1 callee, selectable per `stereo_input`
  (`caller`, `callee`, `mix`, `interleaved`).
- Engine-side VAD turn boundaries (`speech_started` / `speech_stopped`) mapped to either the VAD
  frames pipecat's default turn-start strategy consumes or explicit user-turn frames, plus an
  optional interruption.
- DTMF, playout marks, graceful `stop`, and fatal/non-fatal `error` handling.
- A runnable parrot-bot example under `examples/`.

### Notes

- Verified against `pipecat-ai` 1.7.0 on Python 3.11 and 3.13.
- `FrameSerializer` in pipecat 1.7 has no `type` property and no `FrameSerializerType`; the
  transport decides binary versus text from the return type of `serialize`.
