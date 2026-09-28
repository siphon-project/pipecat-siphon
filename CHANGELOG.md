# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2026-09-28

Nothing in the package itself changed: `src/pipecat_siphon` is byte-identical to 0.2.1. What is new
is the examples, CI, and the README that PyPI shows.

### Added

- **The agent example reads its persona from a file, `BOT_SYSTEM_PROMPT_FILE`, and ships one.** A
  persona worth writing runs to paragraphs, which one `.env` line carries badly. The file is read on
  every call, and the quickstart mounts `examples/prompts/` over the image's copy, so a persona can
  be tuned between calls without a restart. Setting both it and `BOT_SYSTEM_PROMPT` stops the bot at
  startup rather than letting one silently win, as does a path that cannot be read. The shipped
  `escalation_demo.txt` is a shop's customer line that offers a fix once and then transfers a caller
  who insists on the manager, which exercises barge-in and `transfer_call` in one call.

### Changed

- **The agent example moves to pipecat 1.12 and answers a turn it could not make out.** Before 1.12
  a caller turn that ended with nothing recognised in it simply did not run the model, so the bot
  said nothing — and silence after somebody speaks reads as a dropped call rather than as a missed
  word. 1.12 runs the model on one of those, and the example keeps that on and replaces the prompt:
  what ships is three sentences written for a chat window that ask the model to re-ask its own
  question, and this is a voice route, so it is one spoken sentence instead. A caller who simply
  goes quiet is still left alone, for the same reason `idle_timeout_secs` is unset — a pause is
  someone thinking. Recovery is capped at one in a row, which is the guard that matters on a leg
  whose echo the engine cannot cancel: the bot's own voice can open a turn whose transcript the echo
  guard then drops, and that is indistinguishable from a caller who was not heard.
- **The example asks for pipecat 1.12; the package still installs against 1.8.** The serializer does
  not use anything newer, so the wheel's own requirement is unchanged and its tests now run against
  that floor in their own CI job rather than the claim going unexercised. `[dev]` asks for 1.12
  because it is the example's tests that need it.

### Added

- **A provider per role in the agent example, instead of one backend for all three.** The bot does
  three jobs, and `BOT_LLM_PROVIDER`, `BOT_STT_PROVIDER` and `BOT_TTS_PROVIDER` each choose who does
  one of them: `anthropic`, `google` or `openai`, or `local` for a model on the host. `BOT_BACKEND`
  stays as a preset that sets all three (`cloud` is Claude, Deepgram and Cartesia; `local` keeps
  everything on the host), so an existing deployment is unaffected. One string for all three roles
  would have needed a name per combination, and the useful ones are mixed: a hosted model with a
  local recognizer, or a local model with a hosted voice. Which credentials are required follows
  from the selection, and the bot still names what is missing at startup rather than at the first
  call. Note that Gemini and Google's speech APIs are different products with different credentials
  — `GOOGLE_API_KEY` against `GOOGLE_APPLICATION_CREDENTIALS` — and that an image carries only the
  libraries it was built with, so a mix outside the two presets needs `BOT_EXTRAS` at build time.
- **A local backend for the agent example.** `BOT_BACKEND=local` runs the three models on the host
  instead of the vendors: an OpenAI-compatible LLM server (llama.cpp's `llama-server`, a new `llm`
  service under the quickstart's `local` compose profile), faster-whisper and Kokoro. The turn
  strategies, the echo guard and the tools are the same objects on both paths. Whisper and Kokoro
  each load their model in their constructor and the example builds its services per call, so as
  shipped every call would pay a model load before the greeting; the local path shares one loaded
  model per process and loads it at startup instead. The vendor keys are no longer required by
  compose: the bot checks what its selected backend needs when it starts and names what is missing.
- **Pinned sampling for the agent example.** `BOT_LLM_TEMPERATURE` sets the model's temperature on
  both backends and `BOT_LLM_SEED` its seed on the local one; the cloud model takes no seed. Both
  are unset by default, which leaves each model server's own defaults. Pin them when calls are
  compared with each other: a scenario repeated a hundred times measures the agent only if the
  model samples the same way every time.
- **An observer hook in the agent example.** `run(arguments, observer_factory=...)` builds an
  observer for each call from that call's serializer, which is what learns the SIP Call-ID, and
  attaches it to the call's pipeline. The example's own entrypoint passes none.

### Fixed

- **OpenAI's voice would have played a third slow and a fifth flat.** Its speech endpoint emits PCM
  at one rate and takes no rate parameter, so asking it for the pipeline's 16 kHz does not change
  the bytes — it only labels 24 kHz audio as 16 kHz, and pipecat warns and carries on. The service
  is now built at the rate it actually emits and the output transport resamples on the way to the
  wire, which is the one place in the process that converts.
- **A probe against a cold bot reported no audio on a working bot.** The window was three seconds
  and the first greeting of a process takes longer than that: the recognizer and synthesizer are
  still opening their sockets and the model has nothing cached. Measured at 3.2 s on the default
  providers, against a 3.0 s window. The default is eight seconds now. A false negative in the tool
  whose whole job is telling a broken model apart from a broken media path is worse than a slow
  check, because it sends you to the wrong half.
- **A pinned seed reached a model whose API has none.** It was gated on the backend being the local
  one, which was true while only that path spoke the OpenAI API. It is now a property of the
  provider, so Gemini and OpenAI get the seed and Anthropic, which has no such parameter, does not.
- **The model could wait five seconds after the caller had finished speaking.** Two processors
  decided the caller's turn from the same frames: a turn processor, and behind it the aggregator
  that releases the context to the model. When the recognizer announces a transcription latency
  (pipecat assumes one second for a service that sets none) and the word count opens the turn on
  the transcript, the processor cancels its pending end-of-turn timer before it announces the
  start, and that cancellation yields. The aggregator handles the transcript in the gap, ahead of
  the turn it belongs to, and then cannot end that turn: it waited out pipecat's five-second
  fallback. Measured on a call, the processor ended the turn 200 ms after the caller said goodbye
  and the model was asked 5.0 s after that. Having the aggregator follow the processor's
  decisions instead of repeating them stalls the same way. The turn processor is gone, and the
  aggregator runs the strategies itself.
- **The echo guard dropped the second half of a sentence.** A caller who said "my name is", paused,
  and said the name lost the name. Every caller turn interrupts, the interruption sends the engine
  a `clear`, and the engine acknowledges each one with a mark whether or not anything was playing.
  The serializer reports that mark as the bot stopping, so the example's speech monitor restarted
  its echo tail while the bot had been silent for seconds, and the guard discarded whatever the
  caller said in the next 0.8 s. The monitor now takes a stop as an edge only after the bot has
  actually spoken.
- **The bot said it was hanging up or transferring, and then did not.** The prompt and the tool
  descriptions asked for the words first and the tool after them. Models served locally took that
  as two turns: they spoke the line and stopped. Replayed from a call, a 4B and a 30B model both
  made 0 of 5 hangups and 0 of 5 transfers. The prompt now asks for the tool first and the words
  after it returns, which pipecat already supports by running the model again on the tool's
  result; the same replay made 10 of 10 of each, with the goodbye and the handover line spoken by
  that second reply. The farewell wait gives that reply longer to start, because it now includes
  a second pass through the model.
- **A transfer that completed was logged as failed.** The verdict was read from the status code
  alone, and a completed transfer can arrive with reason OK and no code. The verdict's kind now
  decides, and the code is reported alongside it when there is one.
- **The bot could hang up just before its goodbye played.** The farewell wait counted any speech
  earlier in the call as the farewell having started, so with the tool called before the goodbye
  it took the silence in between for the end of it. Measured on a call: the line went 40 ms before
  the goodbye would have played. Only speech that starts after the tool fires counts now.

### Changed

- **The agent example waits for the rest of a sentence the caller stopped in the middle of, on
  both backends.** A caller who said "my name is", paused and then said the name was answered
  after the first half: the turn ended on a fixed 0.2 s of silence, and the name arrived while the
  bot was talking, where the echo guard dropped it. A turn that would end on an English transcript
  stopping on a word a sentence cannot end on ("is", "the", "to", "and", "my" and a few more) is
  now held open for up to 1.5 s, and speech resuming in that time joins the same turn; anything
  else ends exactly as before. The end of the turn is still decided by pipecat's speech-timeout
  strategy, which the hold wraps rather than replaces, so a transcript the recognizer is still
  delivering is still waited for. Whether the word list applies is decided per transcript: by the
  language the recognizer reports for it, or by the configured language when it reports none, so a
  recognizer set to `multi` that reports nothing is never held. pipecat's bundled smart-turn model
  was tried first and misjudged phone-band speech both ways: "my name is" passed through an 8 kHz
  channel read as finished, and a recorded "goodbye" as unfinished.

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
