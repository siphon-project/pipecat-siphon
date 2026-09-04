"""Serializer tests: the engine's wire in, pipecat frames out, and back again."""

from __future__ import annotations

import json
import math
import struct

import pytest
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    CancelWorkerFrame,
    EndFrame,
    EndWorkerFrame,
    ErrorFrame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    InterruptionWorkerFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    TextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager

import wire_fixtures as wire
from pipecat_siphon import Direction, Encoding, Endianness, SiphonFrameSerializer
from pipecat_siphon.protocol import MediaFormat, StartData, encode_control

PIPELINE_IN_RATE = 16000
PIPELINE_OUT_RATE = 24000


def tone(samples: int, *, sample_rate: int, hertz: float = 440.0, channels: int = 1) -> bytes:
    """Build a deterministic little-endian 16-bit tone, interleaved across ``channels``."""
    frames = []
    for index in range(samples):
        value = int(12000 * math.sin(2 * math.pi * hertz * index / sample_rate))
        frames.extend([value] * channels)
    return struct.pack(f"<{len(frames)}h", *frames)


def callee_tone(samples: int, *, sample_rate: int) -> bytes:
    """Build the callee half of :func:`stereo_tone`: 880 Hz at half the caller's amplitude."""
    values = [
        int(6000 * math.sin(2 * math.pi * 880.0 * index / sample_rate)) for index in range(samples)
    ]
    return struct.pack(f"<{len(values)}h", *values)


def stereo_tone(samples: int, *, sample_rate: int) -> bytes:
    """Build a stereo tone whose two channels are plainly different (caller 440, callee 880)."""
    caller = struct.unpack(f"<{samples}h", tone(samples, sample_rate=sample_rate))
    callee = struct.unpack(f"<{samples}h", callee_tone(samples, sample_rate=sample_rate))
    interleaved = [sample for pair in zip(caller, callee, strict=True) for sample in pair]
    return struct.pack(f"<{len(interleaved)}h", *interleaved)


def pipeline_setup(
    audio_in_sample_rate: int = PIPELINE_IN_RATE,
    audio_out_sample_rate: int = PIPELINE_OUT_RATE,
) -> FrameProcessorSetup:
    """Build the setup object a transport hands the serializer.

    Pipecat 1.8 moved the pipeline's sample rates out of ``StartFrame`` and into this object, so
    this is what the transport really passes. It is a real one rather than a stand-in: the clock,
    the task manager and the worker are inert here (the serializer reads only the two rates), but
    constructing them is what makes the test fail if the argument type moves again.
    """
    return FrameProcessorSetup(
        # pipecat's clock constructor is untyped; the call is fine, mypy just cannot see it.
        clock=SystemClock(),  # type: ignore[no-untyped-call]
        task_manager=TaskManager(),
        pipeline_worker=PipelineWorker(Pipeline([])),
        audio_in_sample_rate=audio_in_sample_rate,
        audio_out_sample_rate=audio_out_sample_rate,
    )


async def make_serializer(
    start_text: str = wire.START_8K,
    *,
    params: SiphonFrameSerializer.InputParams | None = None,
    audio_in_sample_rate: int = PIPELINE_IN_RATE,
    audio_out_sample_rate: int = PIPELINE_OUT_RATE,
) -> SiphonFrameSerializer:
    """Build a serializer that has been set up by the pipeline and seen the engine's ``start``."""
    serializer = SiphonFrameSerializer(params=params)
    await serializer.setup(pipeline_setup(audio_in_sample_rate, audio_out_sample_rate))
    assert await serializer.deserialize(start_text) is None
    return serializer


#
# Handshake
#


async def test_start_handshake_at_eight_kilohertz_captures_the_negotiated_format() -> None:
    serializer = await make_serializer(wire.START_8K)
    assert serializer.stream_id == wire.STREAM_ID
    assert serializer.call_id == wire.CALL_ID
    assert serializer.direction is Direction.DUPLEX
    assert serializer.wire_sample_rate == 8000
    assert serializer.media_format.frame_bytes == 320


async def test_start_handshake_at_a_selected_sixteen_kilohertz_wire_rate() -> None:
    serializer = await make_serializer(wire.START_16K)
    assert serializer.wire_sample_rate == 16000
    assert serializer.media_format.frame_bytes == 640


async def test_the_wire_rate_from_start_overrides_the_configured_assumption() -> None:
    params = SiphonFrameSerializer.InputParams(wire_sample_rate=8000)
    serializer = SiphonFrameSerializer(params=params)
    assert serializer.wire_sample_rate == 8000
    await serializer.setup(pipeline_setup())
    await serializer.deserialize(wire.START_16K)
    assert serializer.wire_sample_rate == 16000


async def test_media_renegotiate_moves_the_wire_rate_mid_stream() -> None:
    serializer = await make_serializer(wire.START_8K)
    assert await serializer.deserialize(wire.MEDIA_RENEGOTIATE) is None
    assert serializer.wire_sample_rate == 24000


async def test_a_tee_start_reports_send_only_stereo_with_track_labels() -> None:
    serializer = await make_serializer(wire.START_TEE_STEREO)
    assert serializer.direction is Direction.SEND
    assert serializer.tracks == ("inbound", "outbound")
    assert serializer.media_format.channels == 2


#
# Uplink audio (engine -> pipecat)
#


async def test_binary_uplink_at_the_pipeline_rate_needs_no_resampling() -> None:
    serializer = await make_serializer(wire.START_16K, audio_in_sample_rate=16000)
    payload = tone(320, sample_rate=16000)  # 20 ms at 16 kHz
    frame = await serializer.deserialize(payload)
    assert isinstance(frame, InputAudioRawFrame)
    assert frame.sample_rate == 16000
    assert frame.num_channels == 1
    assert frame.audio == payload


async def drain_uplink(
    serializer: SiphonFrameSerializer, payload: bytes, *, frames: int, expect_rate: int
) -> int:
    """Feed ``frames`` binary frames and return how many samples came back out.

    Pipecat's streaming SOXR resampler buffers: the first chunks of a conversion produce nothing
    and the output arrives in bursts once the filter has filled. That is the same behaviour every
    in-tree telephony serializer sees, so the assertion is on the total, not on each frame.
    """
    total = 0
    produced = 0
    for _ in range(frames):
        frame = await serializer.deserialize(payload)
        if frame is None:
            continue
        assert isinstance(frame, InputAudioRawFrame)
        assert frame.sample_rate == expect_rate
        assert frame.num_channels == 1
        produced += 1
        total += len(frame.audio) // 2
    assert produced, "the resampler never produced a frame"
    return total


async def test_binary_uplink_upsamples_an_eight_kilohertz_wire_for_a_wideband_pipeline() -> None:
    serializer = await make_serializer(wire.START_8K, audio_in_sample_rate=16000)
    payload = tone(160, sample_rate=8000)  # 20 ms at 8 kHz = 320 bytes
    assert len(payload) == serializer.media_format.frame_bytes

    total = await drain_uplink(serializer, payload, frames=50, expect_rate=16000)
    # Fifty 20 ms frames upsampled 8 k -> 16 k is 16000 samples, less what is still buffered.
    assert 14000 <= total <= 16000


async def test_binary_uplink_downsamples_a_wideband_wire_for_a_narrowband_pipeline() -> None:
    serializer = await make_serializer(wire.START_16K, audio_in_sample_rate=8000)
    payload = tone(320, sample_rate=16000)

    total = await drain_uplink(serializer, payload, frames=50, expect_rate=8000)
    assert 7000 <= total <= 8000


async def test_a_stereo_uplink_keeps_only_the_caller_by_default() -> None:
    serializer = await make_serializer(wire.START_TEE_STEREO, audio_in_sample_rate=16000)
    payload = stereo_tone(320, sample_rate=16000)
    assert len(payload) == serializer.media_format.frame_bytes

    frame = await serializer.deserialize(payload)
    assert isinstance(frame, InputAudioRawFrame)
    assert frame.num_channels == 1
    assert frame.audio == tone(320, sample_rate=16000)


async def test_a_stereo_uplink_can_select_the_callee_channel() -> None:
    params = SiphonFrameSerializer.InputParams(stereo_input="callee")
    serializer = await make_serializer(
        wire.START_TEE_STEREO, params=params, audio_in_sample_rate=16000
    )
    frame = await serializer.deserialize(stereo_tone(320, sample_rate=16000))
    assert isinstance(frame, InputAudioRawFrame)
    assert frame.num_channels == 1
    # Channel 1 is the callee: the 880 Hz half-amplitude leg, not the caller's 440 Hz.
    assert frame.audio == callee_tone(320, sample_rate=16000)
    assert frame.audio != tone(320, sample_rate=16000)


async def test_a_stereo_uplink_can_stay_interleaved() -> None:
    params = SiphonFrameSerializer.InputParams(stereo_input="interleaved")
    serializer = await make_serializer(
        wire.START_TEE_STEREO, params=params, audio_in_sample_rate=16000
    )
    payload = stereo_tone(320, sample_rate=16000)
    frame = await serializer.deserialize(payload)
    assert isinstance(frame, InputAudioRawFrame)
    assert frame.num_channels == 2
    assert frame.audio == payload


async def test_a_stereo_uplink_can_be_mixed_to_mono() -> None:
    params = SiphonFrameSerializer.InputParams(stereo_input="mix")
    serializer = await make_serializer(
        wire.START_TEE_STEREO, params=params, audio_in_sample_rate=16000
    )
    frame = await serializer.deserialize(stereo_tone(320, sample_rate=16000))
    assert isinstance(frame, InputAudioRawFrame)
    assert frame.num_channels == 1
    assert len(frame.audio) == 640


async def test_a_big_endian_wire_is_byte_swapped_on_the_way_in() -> None:
    media = MediaFormat(endianness=Endianness.BIG)
    start = encode_control(
        StartData(stream_id="s", call_id="c", direction=Direction.DUPLEX, media=media)
    )
    serializer = await make_serializer(start, audio_in_sample_rate=8000)
    host_order = tone(160, sample_rate=8000)
    big_endian = struct.pack(">160h", *struct.unpack("<160h", host_order))

    frame = await serializer.deserialize(big_endian)
    assert isinstance(frame, InputAudioRawFrame)
    assert frame.audio == host_order


async def test_a_mu_law_wire_is_decoded_to_linear_pcm() -> None:
    media = MediaFormat(encoding=Encoding.PCMU, bit_depth=8)
    start = encode_control(
        StartData(stream_id="s", call_id="c", direction=Direction.DUPLEX, media=media)
    )
    serializer = await make_serializer(start, audio_in_sample_rate=8000)
    frame = await serializer.deserialize(bytes(range(160)))
    assert isinstance(frame, InputAudioRawFrame)
    assert frame.sample_rate == 8000
    assert len(frame.audio) == 320


#
# Downlink audio (pipecat -> engine)
#


async def test_downlink_audio_is_resampled_into_the_negotiated_wire_rate() -> None:
    serializer = await make_serializer(wire.START_8K)
    audio = tone(320, sample_rate=16000)

    total = 0
    produced = 0
    for _ in range(50):
        payload = await serializer.serialize(
            OutputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1)
        )
        if payload is None:
            continue
        assert isinstance(payload, bytes)
        produced += 1
        total += len(payload) // 2
    assert produced, "the resampler never produced a frame"
    # Fifty 20 ms frames downsampled 16 k -> 8 k is 8000 samples, less what is still buffered.
    assert 7000 <= total <= 8000


async def test_downlink_audio_passes_through_untouched_when_the_rates_already_match() -> None:
    serializer = await make_serializer(wire.START_16K)
    audio = tone(320, sample_rate=16000)
    payload = await serializer.serialize(
        OutputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1)
    )
    assert payload == audio


async def test_downlink_audio_is_byte_swapped_for_a_big_endian_wire() -> None:
    media = MediaFormat(endianness=Endianness.BIG)
    start = encode_control(
        StartData(stream_id="s", call_id="c", direction=Direction.DUPLEX, media=media)
    )
    serializer = await make_serializer(start)
    audio = tone(160, sample_rate=8000)
    payload = await serializer.serialize(
        OutputAudioRawFrame(audio=audio, sample_rate=8000, num_channels=1)
    )
    assert isinstance(payload, bytes)
    assert payload == struct.pack(">160h", *struct.unpack("<160h", audio))


async def test_downlink_audio_is_duplicated_across_a_stereo_wire() -> None:
    media = MediaFormat(channels=2)
    start = encode_control(
        StartData(stream_id="s", call_id="c", direction=Direction.DUPLEX, media=media)
    )
    serializer = await make_serializer(start)
    audio = tone(160, sample_rate=8000)
    payload = await serializer.serialize(
        OutputAudioRawFrame(audio=audio, sample_rate=8000, num_channels=1)
    )
    assert isinstance(payload, bytes)
    assert len(payload) == 2 * len(audio)
    assert struct.unpack("<320h", payload)[0:2] == struct.unpack("<160h", audio)[0:1] * 2


async def test_a_stereo_pipeline_frame_is_folded_to_mono_for_a_mono_wire() -> None:
    serializer = await make_serializer(wire.START_8K)
    payload = await serializer.serialize(
        OutputAudioRawFrame(
            audio=stereo_tone(160, sample_rate=8000), sample_rate=8000, num_channels=2
        )
    )
    assert isinstance(payload, bytes)
    assert len(payload) == 320


async def test_a_send_only_stream_never_gets_audio_written_back_into_it() -> None:
    serializer = await make_serializer(wire.START_TEE_STEREO)
    payload = await serializer.serialize(
        OutputAudioRawFrame(audio=tone(320, sample_rate=16000), sample_rate=16000, num_channels=1)
    )
    assert payload is None


async def test_an_empty_audio_frame_serializes_to_nothing() -> None:
    serializer = await make_serializer(wire.START_8K)
    payload = await serializer.serialize(
        OutputAudioRawFrame(audio=b"", sample_rate=8000, num_channels=1)
    )
    assert payload is None


#
# Control, pipecat -> engine
#


async def test_an_interruption_becomes_a_clear_addressed_to_the_stream() -> None:
    serializer = await make_serializer(wire.START_8K)
    payload = await serializer.serialize(InterruptionFrame())
    assert payload == wire.CLEAR_BARGE_IN


async def test_an_end_frame_becomes_a_stop_carrying_its_reason() -> None:
    serializer = await make_serializer(wire.START_8K)
    payload = await serializer.serialize(EndFrame(reason="call_ended"))
    assert payload == wire.STOP


async def test_a_cancel_frame_becomes_a_stop_with_the_configured_default_reason() -> None:
    serializer = await make_serializer(wire.START_8K)
    payload = await serializer.serialize(CancelFrame())
    assert isinstance(payload, str)
    assert json.loads(payload) == {
        "type": "stop",
        "data": {"streamId": wire.STREAM_ID, "reason": "pipeline_ended"},
    }


async def test_a_named_transport_message_becomes_an_event_with_that_name() -> None:
    serializer = await make_serializer(wire.START_8K)
    payload = await serializer.serialize(
        OutputTransportMessageFrame(
            message={"name": "transcription", "payload": {"text": "hello", "final": True}}
        )
    )
    assert isinstance(payload, str)
    assert json.loads(payload) == {
        "type": "event",
        "data": {
            "streamId": wire.STREAM_ID,
            "name": "transcription",
            "payload": {"text": "hello", "final": True},
        },
    }


async def test_an_unnamed_transport_message_is_wrapped_under_the_default_event_name() -> None:
    serializer = await make_serializer(wire.START_8K)
    payload = await serializer.serialize(OutputTransportMessageFrame(message={"foo": "bar"}))
    assert isinstance(payload, str)
    assert json.loads(payload)["data"]["name"] == "pipecat"
    assert json.loads(payload)["data"]["payload"] == {"foo": "bar"}


async def test_an_rtvi_transport_message_is_not_forwarded_to_the_engine() -> None:
    serializer = await make_serializer(wire.START_8K)
    payload = await serializer.serialize(
        OutputTransportMessageFrame(message={"label": "rtvi-ai", "type": "anything"})
    )
    assert payload is None


async def test_an_unmapped_frame_serializes_to_nothing() -> None:
    serializer = await make_serializer(wire.START_8K)
    assert await serializer.serialize(TextFrame(text="hello")) is None


async def test_control_frames_are_dropped_while_the_stream_id_is_still_unknown() -> None:
    serializer = SiphonFrameSerializer()
    await serializer.setup(pipeline_setup())
    assert await serializer.serialize(InterruptionFrame()) is None


async def test_a_stream_id_supplied_up_front_addresses_control_frames_before_start() -> None:
    serializer = SiphonFrameSerializer(stream_id=wire.STREAM_ID)
    await serializer.setup(pipeline_setup())
    assert await serializer.serialize(InterruptionFrame()) == wire.CLEAR_BARGE_IN


#
# Control, engine -> pipecat
#


async def test_speech_edges_default_to_the_vad_frames_pipecat_turn_taking_consumes() -> None:
    serializer = await make_serializer(wire.START_8K)
    started = await serializer.deserialize(wire.SPEECH_STARTED)
    stopped = await serializer.deserialize(wire.SPEECH_STOPPED)
    assert isinstance(started, VADUserStartedSpeakingFrame)
    assert isinstance(stopped, VADUserStoppedSpeakingFrame)


async def test_speech_edges_can_be_emitted_as_user_turn_frames_with_an_interruption() -> None:
    params = SiphonFrameSerializer.InputParams(speech_frames="user", auto_interrupt=True)
    serializer = await make_serializer(wire.START_8K, params=params)

    started = await serializer.deserialize(wire.SPEECH_STARTED)
    assert isinstance(started, UserStartedSpeakingFrame)
    # deserialize() can only hand back one frame per message, so the interruption is queued and
    # drains on the next message: at most one ptime later, since the engine keeps ticking.
    queued = await serializer.deserialize(wire.MARK)
    assert isinstance(queued, InterruptionWorkerFrame)
    # Nothing is lost while the queue drains: the mark that carried the drain is next in line.
    carried = await serializer.deserialize(wire.SPEECH_STOPPED)
    assert isinstance(carried, BotStoppedSpeakingFrame)
    stopped = await serializer.deserialize(wire.MARK)
    assert isinstance(stopped, UserStoppedSpeakingFrame)


async def test_speech_edges_without_auto_interrupt_emit_only_the_turn_frame() -> None:
    params = SiphonFrameSerializer.InputParams(speech_frames="user", auto_interrupt=False)
    serializer = await make_serializer(wire.START_8K, params=params)
    assert isinstance(await serializer.deserialize(wire.SPEECH_STARTED), UserStartedSpeakingFrame)
    assert await serializer.deserialize(wire.SPEECH_STOPPED) is not None


async def test_speech_edges_can_be_ignored_entirely() -> None:
    params = SiphonFrameSerializer.InputParams(speech_frames="none")
    serializer = await make_serializer(wire.START_8K, params=params)
    assert await serializer.deserialize(wire.SPEECH_STARTED) is None
    assert await serializer.deserialize(wire.SPEECH_STOPPED) is None


async def test_a_dtmf_digit_becomes_an_input_dtmf_frame() -> None:
    serializer = await make_serializer(wire.START_8K)
    frame = await serializer.deserialize(wire.DTMF)
    assert isinstance(frame, InputDTMFFrame)
    assert frame.button.value == "5"


@pytest.mark.parametrize("digit", ["*", "#", "0", "9"])
async def test_every_keypad_digit_pipecat_models_maps_through(digit: str) -> None:
    serializer = await make_serializer(wire.START_8K)
    text = json.dumps(
        {
            "type": "dtmf",
            "data": {
                "streamId": wire.STREAM_ID,
                "digit": digit,
                "track": "inbound",
                "durationMs": 160,
            },
        }
    )
    frame = await serializer.deserialize(text)
    assert isinstance(frame, InputDTMFFrame)
    assert frame.button.value == digit


@pytest.mark.parametrize("digit", ["A", "B", "C", "D"])
async def test_the_abcd_digits_pipecat_does_not_model_are_dropped(digit: str) -> None:
    serializer = await make_serializer(wire.START_8K)
    text = json.dumps(
        {
            "type": "dtmf",
            "data": {
                "streamId": wire.STREAM_ID,
                "digit": digit,
                "track": "inbound",
                "durationMs": 160,
            },
        }
    )
    assert await serializer.deserialize(text) is None


async def test_a_mark_reports_the_playout_boundary_as_bot_stopped_speaking() -> None:
    serializer = await make_serializer(wire.START_8K)
    assert isinstance(await serializer.deserialize(wire.MARK), BotStoppedSpeakingFrame)
    assert isinstance(await serializer.deserialize(wire.MARK_CLEARED), BotStoppedSpeakingFrame)


async def test_marks_can_be_left_unmapped() -> None:
    params = SiphonFrameSerializer.InputParams(emit_bot_speaking_frames=False)
    serializer = await make_serializer(wire.START_8K, params=params)
    assert await serializer.deserialize(wire.MARK) is None


async def test_stop_ends_the_pipeline_gracefully() -> None:
    serializer = await make_serializer(wire.START_8K)
    frame = await serializer.deserialize(wire.STOP)
    assert isinstance(frame, EndWorkerFrame)
    assert frame.reason == "call_ended"


async def test_a_fatal_error_cancels_the_pipeline_and_carries_the_reason() -> None:
    serializer = await make_serializer(wire.START_8K)
    frame = await serializer.deserialize(wire.ERROR_FATAL)
    assert isinstance(frame, CancelWorkerFrame)
    assert frame.reason == "unsupported_encoding: inline play_start is not supported"


async def test_a_non_fatal_error_surfaces_as_an_error_frame() -> None:
    serializer = await make_serializer(wire.START_8K)
    text = json.dumps(
        {
            "type": "error",
            "data": {
                "streamId": wire.STREAM_ID,
                "code": "queue_overflow",
                "message": "downlink queue full",
                "fatal": False,
            },
        }
    )
    frame = await serializer.deserialize(text)
    assert isinstance(frame, ErrorFrame)
    assert frame.fatal is False
    assert frame.error == "queue_overflow: downlink queue full"


async def test_an_engine_event_is_logged_and_not_pushed_as_a_frame() -> None:
    serializer = await make_serializer(wire.START_8K)
    assert await serializer.deserialize(wire.EVENT) is None


async def test_server_to_engine_verbs_arriving_from_the_engine_are_ignored() -> None:
    serializer = await make_serializer(wire.START_8K)
    assert await serializer.deserialize(wire.PLAY_START_BINARY) is None
    assert await serializer.deserialize(wire.PLAY_STOP) is None
    assert await serializer.deserialize(wire.CLEAR_BARGE_IN) is None


#
# Untrusted input
#


@pytest.mark.parametrize(
    "text",
    [
        "",
        "{",
        "not json",
        "[]",
        '{"type":"nonesuch","data":{}}',
        '{"type":"start","data":{}}',
        '{"type":"dtmf","data":{"streamId":"s","digit":"5","track":"inbound"}}',
    ],
)
async def test_malformed_control_frames_are_dropped_rather_than_raised(text: str) -> None:
    serializer = await make_serializer(wire.START_8K)
    assert await serializer.deserialize(text) is None


async def test_an_empty_binary_frame_is_dropped() -> None:
    serializer = await make_serializer(wire.START_8K)
    assert await serializer.deserialize(b"") is None


async def test_a_binary_frame_with_a_dangling_byte_is_dropped() -> None:
    serializer = await make_serializer(wire.START_8K, audio_in_sample_rate=8000)
    assert await serializer.deserialize(tone(160, sample_rate=8000) + b"\x01") is None


async def test_a_stereo_frame_with_a_dangling_sample_is_dropped() -> None:
    serializer = await make_serializer(wire.START_TEE_STEREO, audio_in_sample_rate=16000)
    assert await serializer.deserialize(stereo_tone(320, sample_rate=16000)[:-2]) is None


async def test_a_short_binary_frame_is_still_decoded() -> None:
    # The engine frames on ptime, but nothing in the protocol forbids a shorter binary frame.
    serializer = await make_serializer(wire.START_8K, audio_in_sample_rate=8000)
    frame = await serializer.deserialize(tone(8, sample_rate=8000))
    assert isinstance(frame, InputAudioRawFrame)
    assert len(frame.audio) == 16


async def test_binary_audio_arriving_before_start_uses_the_configured_assumption() -> None:
    params = SiphonFrameSerializer.InputParams(wire_sample_rate=8000)
    serializer = SiphonFrameSerializer(params=params)
    await serializer.setup(pipeline_setup(audio_in_sample_rate=8000))
    frame = await serializer.deserialize(tone(160, sample_rate=8000))
    assert isinstance(frame, InputAudioRawFrame)
    assert frame.sample_rate == 8000
