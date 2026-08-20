"""Byte-exact control-frame fixtures.

Every literal below was emitted by the engine's own Rust definition
(``crates/siphon-rtp-media/src/bridge/protocol.rs`` compiled against ``serde_json``), not by
this package's encoder. That independence is the point: a shared encode/decode bug in
``pipecat_siphon.protocol`` would sail straight through a round-trip test, but it cannot
survive a comparison against bytes a different implementation produced.
"""

from __future__ import annotations

from typing import Final

START_8K: Final = (
    '{"type":"start","data":{"streamId":"ws-call-7@198.51.100.2",'
    '"callId":"call-7@198.51.100.2","direction":"duplex",'
    '"media":{"encoding":"L16","sampleRate":8000,"channels":1,"bitDepth":16,'
    '"endianness":"little","ptime":20}}}'
)

START_16K: Final = (
    '{"type":"start","data":{"streamId":"ws-call-9@198.51.100.2",'
    '"callId":"call-9@198.51.100.2","direction":"duplex",'
    '"media":{"encoding":"L16","sampleRate":16000,"channels":1,"bitDepth":16,'
    '"endianness":"little","ptime":20}}}'
)

START_TEE_STEREO: Final = (
    '{"type":"start","data":{"streamId":"ws-tee-call-7@198.51.100.2",'
    '"callId":"call-7@198.51.100.2","direction":"send",'
    '"media":{"encoding":"L16","sampleRate":16000,"channels":2,"bitDepth":16,'
    '"endianness":"little","ptime":20},"tracks":["inbound","outbound"],'
    '"metadata":{"tenant":"documentation"}}}'
)

MEDIA_RENEGOTIATE: Final = (
    '{"type":"media_renegotiate","data":{"streamId":"ws-call-7@198.51.100.2",'
    '"media":{"encoding":"L16","sampleRate":24000,"channels":1,"bitDepth":16,'
    '"endianness":"little","ptime":20}}}'
)

PLAY_START_INLINE: Final = (
    '{"type":"play_start","data":{"streamId":"ws-call-7@198.51.100.2","playId":"p1",'
    '"source":"inline","audioDataType":"raw","encoding":"L16","sampleRate":8000,'
    '"audioData":"AAAA","interruptible":true,"markName":"p1_end"}}'
)

PLAY_START_BINARY: Final = (
    '{"type":"play_start","data":{"streamId":"ws-call-7@198.51.100.2","playId":"p2",'
    '"source":"binary","interruptible":false}}'
)

PLAY_STOP: Final = '{"type":"play_stop","data":{"streamId":"ws-call-7@198.51.100.2","playId":"p2"}}'

CLEAR_BARGE_IN: Final = (
    '{"type":"clear","data":{"streamId":"ws-call-7@198.51.100.2","reason":"barge_in"}}'
)

CLEAR_PLAY: Final = '{"type":"clear","data":{"streamId":"ws-call-7@198.51.100.2","playId":"p1"}}'

MARK: Final = (
    '{"type":"mark","data":{"streamId":"ws-call-7@198.51.100.2","playId":"p1","name":"p1_end"}}'
)

MARK_CLEARED: Final = (
    '{"type":"mark","data":{"streamId":"ws-call-7@198.51.100.2","name":"cleared"}}'
)

DTMF: Final = (
    '{"type":"dtmf","data":{"streamId":"ws-call-7@198.51.100.2","digit":"5",'
    '"track":"inbound","durationMs":160}}'
)

SPEECH_STARTED: Final = '{"type":"speech_started","data":{"streamId":"ws-call-7@198.51.100.2"}}'

SPEECH_STOPPED: Final = '{"type":"speech_stopped","data":{"streamId":"ws-call-7@198.51.100.2"}}'

STOP: Final = '{"type":"stop","data":{"streamId":"ws-call-7@198.51.100.2","reason":"call_ended"}}'

ERROR_FATAL: Final = (
    '{"type":"error","data":{"streamId":"ws-call-7@198.51.100.2",'
    '"code":"unsupported_encoding","message":"inline play_start is not supported",'
    '"fatal":true}}'
)

EVENT: Final = (
    '{"type":"event","data":{"streamId":"ws-call-7@198.51.100.2","name":"transcription",'
    '"payload":{"final":true,"text":"hello"}}}'
)

STREAM_ID: Final = "ws-call-7@198.51.100.2"
CALL_ID: Final = "call-7@198.51.100.2"

ALL_MESSAGES: Final = {
    "start": START_8K,
    "start_16k": START_16K,
    "start_tee_stereo": START_TEE_STEREO,
    "media_renegotiate": MEDIA_RENEGOTIATE,
    "play_start_inline": PLAY_START_INLINE,
    "play_start_binary": PLAY_START_BINARY,
    "play_stop": PLAY_STOP,
    "clear_barge_in": CLEAR_BARGE_IN,
    "clear_play": CLEAR_PLAY,
    "mark": MARK,
    "mark_cleared": MARK_CLEARED,
    "dtmf": DTMF,
    "speech_started": SPEECH_STARTED,
    "speech_stopped": SPEECH_STOPPED,
    "stop": STOP,
    "error_fatal": ERROR_FATAL,
    "event": EVENT,
}
