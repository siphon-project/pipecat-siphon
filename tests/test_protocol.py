"""Wire-level tests for the control envelope.

Encoding is checked against fixtures the engine itself produced (see ``wire_fixtures``), so a
matching bug on both sides of this package's own encode/decode pair cannot hide here.
"""

from __future__ import annotations

import pytest

import wire_fixtures as wire
from pipecat_siphon.protocol import (
    ClearData,
    Direction,
    DtmfData,
    Encoding,
    Endianness,
    ErrorData,
    EventData,
    MarkData,
    MediaFormat,
    PlaySource,
    PlayStartData,
    PlayStopData,
    RenegotiateData,
    SiphonProtocolError,
    SpeechStartedData,
    SpeechStoppedData,
    StartData,
    StopData,
    encode_control,
    parse_control,
)


@pytest.mark.parametrize(("name", "text"), sorted(wire.ALL_MESSAGES.items()))
def test_every_engine_fixture_parses_and_re_encodes_byte_identically(name: str, text: str) -> None:
    message = parse_control(text)
    assert encode_control(message) == text, name


def test_start_at_eight_kilohertz_carries_the_telephony_default_format() -> None:
    message = parse_control(wire.START_8K)
    assert isinstance(message, StartData)
    assert message.stream_id == wire.STREAM_ID
    assert message.call_id == wire.CALL_ID
    assert message.direction is Direction.DUPLEX
    assert message.media == MediaFormat.telephony_default()
    assert message.media.sample_rate == 8000
    assert message.media.frame_bytes == 320
    assert message.tracks == ()
    assert message.metadata is None


def test_start_at_a_selected_sixteen_kilohertz_wire_rate() -> None:
    message = parse_control(wire.START_16K)
    assert isinstance(message, StartData)
    # The controller selected the wire rate independently of the call's codec rate.
    assert message.media.sample_rate == 16000
    assert message.media.frame_bytes == 640


def test_start_for_a_send_only_stereo_tee_carries_tracks_and_metadata() -> None:
    message = parse_control(wire.START_TEE_STEREO)
    assert isinstance(message, StartData)
    assert message.direction is Direction.SEND
    assert message.media.channels == 2
    assert message.media.sample_rate == 16000
    assert message.media.frame_bytes == 1280
    assert message.tracks == ("inbound", "outbound")
    assert message.metadata == {"tenant": "documentation"}


def test_media_renegotiate_replaces_the_format() -> None:
    message = parse_control(wire.MEDIA_RENEGOTIATE)
    assert isinstance(message, RenegotiateData)
    assert message.media.sample_rate == 24000


def test_play_start_inline_carries_the_base64_container_fields() -> None:
    message = parse_control(wire.PLAY_START_INLINE)
    assert isinstance(message, PlayStartData)
    assert message.source is PlaySource.INLINE
    assert message.audio_data_type == "raw"
    assert message.encoding is Encoding.L16
    assert message.sample_rate == 8000
    assert message.audio_data == "AAAA"
    assert message.interruptible is True
    assert message.mark_name == "p1_end"


def test_play_start_omitting_interruptible_defaults_to_true() -> None:
    text = '{"type":"play_start","data":{"streamId":"s","playId":"p","source":"binary"}}'
    message = parse_control(text)
    assert isinstance(message, PlayStartData)
    assert message.interruptible is True
    assert message.source is PlaySource.BINARY


def test_play_stop_and_clear_and_mark_round_trip() -> None:
    assert parse_control(wire.PLAY_STOP) == PlayStopData(stream_id=wire.STREAM_ID, play_id="p2")
    assert parse_control(wire.CLEAR_BARGE_IN) == ClearData(
        stream_id=wire.STREAM_ID, reason="barge_in"
    )
    assert parse_control(wire.CLEAR_PLAY) == ClearData(stream_id=wire.STREAM_ID, play_id="p1")
    assert parse_control(wire.MARK) == MarkData(
        stream_id=wire.STREAM_ID, name="p1_end", play_id="p1"
    )
    assert parse_control(wire.MARK_CLEARED) == MarkData(stream_id=wire.STREAM_ID, name="cleared")


def test_dtmf_carries_digit_track_and_duration() -> None:
    assert parse_control(wire.DTMF) == DtmfData(
        stream_id=wire.STREAM_ID, digit="5", track="inbound", duration_ms=160
    )


def test_speech_started_and_stopped_are_distinct_types() -> None:
    started = parse_control(wire.SPEECH_STARTED)
    stopped = parse_control(wire.SPEECH_STOPPED)
    assert isinstance(started, SpeechStartedData)
    assert isinstance(stopped, SpeechStoppedData)
    # The two edges carry the same single field, so only the type tells them apart. If they ever
    # collapsed into one class a turn start would be indistinguishable from a turn endpoint.
    assert started.TAG == "speech_started"
    assert stopped.TAG == "speech_stopped"
    assert started.stream_id == stopped.stream_id


def test_stop_and_error_and_event_round_trip() -> None:
    assert parse_control(wire.STOP) == StopData(stream_id=wire.STREAM_ID, reason="call_ended")
    assert parse_control(wire.ERROR_FATAL) == ErrorData(
        stream_id=wire.STREAM_ID,
        code="unsupported_encoding",
        message="inline play_start is not supported",
        fatal=True,
    )
    assert parse_control(wire.EVENT) == EventData(
        stream_id=wire.STREAM_ID,
        name="transcription",
        payload={"final": True, "text": "hello"},
    )


def test_frame_bytes_matches_the_engine_formula() -> None:
    assert MediaFormat.telephony_default().frame_bytes == 320
    assert MediaFormat(sample_rate=16000).frame_bytes == 640
    assert MediaFormat(sample_rate=16000, channels=2).frame_bytes == 1280
    assert MediaFormat(encoding=Encoding.PCMU, bit_depth=8).frame_bytes == 160


def test_a_big_endian_format_survives_a_round_trip() -> None:
    media = MediaFormat(endianness=Endianness.BIG)
    text = encode_control(
        StartData(
            stream_id="s", call_id="c", direction=Direction.RECV, media=media, tracks=("inbound",)
        )
    )
    assert '"endianness":"big"' in text
    assert '"direction":"recv"' in text
    parsed = parse_control(text)
    assert isinstance(parsed, StartData)
    assert parsed.media.endianness is Endianness.BIG


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not json at all",
        "{",
        '{"type":"start","data":{"streamId":"s"',
        "[]",
        '"start"',
        "null",
        "123",
        '{"data":{}}',
        '{"type":42,"data":{}}',
        '{"type":"start"}',
        '{"type":"start","data":null}',
        '{"type":"start","data":[]}',
        '{"type":"nonesuch","data":{}}',
        '{"type":"start","data":{"streamId":"s"}}',
        '{"type":"start","data":{"streamId":"s","callId":"c","direction":"sideways",'
        '"media":{"encoding":"L16","sampleRate":8000,"channels":1,"bitDepth":16,'
        '"endianness":"little","ptime":20}}}',
        '{"type":"start","data":{"streamId":"s","callId":"c","direction":"duplex","media":7}}',
        '{"type":"start","data":{"streamId":"s","callId":"c","direction":"duplex",'
        '"media":{"encoding":"OPUS","sampleRate":8000,"channels":1,"bitDepth":16,'
        '"endianness":"little","ptime":20}}}',
        '{"type":"start","data":{"streamId":"s","callId":"c","direction":"duplex",'
        '"media":{"encoding":"L16","sampleRate":"8000","channels":1,"bitDepth":16,'
        '"endianness":"little","ptime":20}}}',
        '{"type":"start","data":{"streamId":"s","callId":"c","direction":"duplex",'
        '"media":{"encoding":"L16","sampleRate":8000,"channels":1,"bitDepth":16,'
        '"endianness":"little","ptime":20},"tracks":"inbound"}}',
        '{"type":"dtmf","data":{"streamId":"s","digit":"5","track":"inbound"}}',
        '{"type":"dtmf","data":{"streamId":"s","digit":5,"track":"inbound","durationMs":160}}',
        '{"type":"error","data":{"streamId":"s","code":"c","message":"m","fatal":"yes"}}',
        '{"type":"event","data":{"streamId":"s","name":"n"}}',
        '{"type":"stop","data":{"streamId":"s"}}',
    ],
)
def test_malformed_control_frames_raise_the_one_protocol_error(text: str) -> None:
    with pytest.raises(SiphonProtocolError):
        parse_control(text)


def test_truncated_utf8_bytes_raise_the_one_protocol_error() -> None:
    with pytest.raises(SiphonProtocolError):
        parse_control(b'{"type":"start","data":{"streamId":"\xff\xfe"}}')


def test_a_boolean_is_not_accepted_where_an_integer_is_required() -> None:
    text = (
        '{"type":"start","data":{"streamId":"s","callId":"c","direction":"duplex",'
        '"media":{"encoding":"L16","sampleRate":true,"channels":1,"bitDepth":16,'
        '"endianness":"little","ptime":20}}}'
    )
    with pytest.raises(SiphonProtocolError):
        parse_control(text)
