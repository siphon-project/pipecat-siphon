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

It depends on `pipecat-ai>=1.7.0,<2` and nothing else. Python 3.11 or newer, matching pipecat.

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

`media.sampleRate` in the `start` envelope is **the negotiated wire rate**, in both directions, for
the life of the stream. It is *not* the call's codec rate. A controller selects it with the
`ws_sample_rate` (takeover) or `ws_tee_sample_rate` / `sample_rate` (tee) knobs, anywhere in the
8000–48000 band in multiples of 1000, and the engine resamples on its side. So an 8 kHz G.711 call
can speak 16 kHz L16 to a model that wants wideband input.

Never assume 8000. Frame against the number `start` gives you, and expect it to differ from what
your pipeline runs at.

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
  on serializers. It does not exist in pipecat 1.7 — the transport decides binary versus text from
  whether `serialize` returned `bytes` or `str` — so this package does not define one.

## Turn taking and barge-in

Three engine profile flags move the turn work into the media engine, which is usually where you
want it for telephony: `ws_vad` (emit `speech_started` / `speech_stopped`), `ws_barge_in` (flush
queued playout the instant the caller talks over the bot, with no server round-trip), and
`echo_cancellation` (stop the bot hearing itself through the caller's handset).

With `ws_vad` on and the default `speech_frames="vad"`, the engine's VAD drives pipecat's turn
machinery directly and you do not need a local VAD in the pipeline at all.

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
back, one jitter-buffer frame plus one playout frame later. Swap the echo processor for an
STT → LLM → TTS chain and it is a bot.

## Compatibility

| | Verified against |
|---|---|
| pipecat | `pipecat-ai` 1.7.0 |
| siphon-rtp bridge protocol | `crates/siphon-rtp-media/src/bridge/protocol.rs` as of the selectable-wire-rate change (`ws_sample_rate` / `ws_tee_sample_rate`) |
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
