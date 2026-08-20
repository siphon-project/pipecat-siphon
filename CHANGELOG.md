# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

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
