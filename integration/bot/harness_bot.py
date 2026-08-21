"""The pipecat bot the engine dials into, in both harness scenarios.

This is a real pipecat pipeline behind :class:`~pipecat_siphon.SiphonFrameSerializer`, not a
WebSocket script pretending to be one: the transport, the serializer, the resamplers, the turn
processor and the interruption machinery are all pipecat's own. The only harness-specific parts
are the two processors that decide what the bot says, and a recorder that writes down what it
saw so the analyser can assert on it afterwards.

Two modes, one per scenario:

``echo``
    Scenario 1. Every uplink frame goes straight back down with a marker tone mixed in. The
    marker is absent from the caller's fixture, so its presence in the RTP that reaches the
    caller is what distinguishes "the bot echoed me" from "something inside the engine looped
    my audio back".

``speaker``
    Scenario 2. The bot starts talking as soon as the stream opens and has twenty seconds of
    speech queued. The caller then talks over it. Everything still queued has to be dropped, in
    the pipeline *and* in the engine, which is what the barge-in assertion measures.

The trace is written as JSON Lines and flushed on every record, so a bot that dies mid-call
still leaves behind everything it saw up to that point.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, TextIO

from loguru import logger
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.websocket.server import (
    SingleClientWebsocketServerParams,
    SingleClientWebsocketServerTransport,
)
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from pipecat_siphon import SiphonFrameSerializer
from tools import signals

BOT_SPEECH_FRAMES = 1000
"""Twenty seconds of bot speech, queued in one go the moment the stream opens.

The call is five seconds long, so the bot always has far more left to say than it will ever get
to deliver. That is the point: "the downlink stops on barge-in" is only worth asserting when
there was a backlog for the barge-in to throw away."""


class TraceRecorder(FrameProcessor):
    """Write down what arrived on the wire, indexed by uplink frame.

    The index is a logical sample clock: one uplink binary frame is one ptime, so frame *n*
    happened at *n x ptime* milliseconds of caller audio. Every timing assertion in the analyser
    is expressed in those units, which is what keeps the results independent of how loaded the
    machine was. The wall clock is recorded too, but only ever printed, never asserted on.
    """

    def __init__(self, trace: TextIO, **kwargs: Any) -> None:
        """Initialize the recorder.

        Args:
            trace: Open text file the JSON Lines records are written to.
            **kwargs: Passed to :class:`~pipecat.processors.frame_processor.FrameProcessor`.

        """
        super().__init__(**kwargs)
        self._trace = trace
        self._uplink_frames = 0
        self._uplink_bytes = 0
        self._started = time.monotonic()

    @property
    def uplink_frames(self) -> int:
        """Return how many uplink audio frames have arrived so far."""
        return self._uplink_frames

    @property
    def uplink_bytes(self) -> int:
        """Return how many uplink audio bytes have arrived so far."""
        return self._uplink_bytes

    def record(self, event: str, **fields: Any) -> None:
        """Append one record to the trace and flush it."""
        record: dict[str, Any] = {
            "event": event,
            "uplink_frame": self._uplink_frames,
            "uplink_ms": self._uplink_frames * signals.PTIME_MS,
            "monotonic": round(time.monotonic() - self._started, 4),
        }
        record.update(fields)
        self._trace.write(json.dumps(record) + "\n")
        self._trace.flush()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Record the frame, then pass it on untouched."""
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            # Recorded with the index it *has*, then counted. The engine sends a tick's turn
            # signals before that tick's binary frame, so a speech_started recorded at index n
            # belongs to the same tick as uplink frame n -- no off-by-one to reason about in
            # the analyser.
            self._uplink_bytes += len(frame.audio)
            # Per-frame energy is what lets the analyser locate the fixture inside the uplink
            # without trusting any clock: the speech burst is wherever the energy steps up.
            self.record(
                "uplink_audio",
                bytes=len(frame.audio),
                sample_rate=frame.sample_rate,
                channels=frame.num_channels,
                rms=round(signals.rms(frame.audio), 2),
            )
            self._uplink_frames += 1
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self.record("speech_started")
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self.record("speech_stopped")
        elif isinstance(frame, UserStartedSpeakingFrame):
            self.record("user_started_speaking")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self.record("user_stopped_speaking")
        elif isinstance(frame, InterruptionFrame):
            self.record("interruption")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self.record("mark")
        elif isinstance(frame, StartFrame):
            self.record("pipeline_started")
        elif isinstance(frame, (EndFrame, CancelFrame)):
            self.record("pipeline_ending", frame=type(frame).__name__)

        await self.push_frame(frame, direction)


class EchoProcessor(FrameProcessor):
    """Echo the caller back with a marker tone mixed in (scenario 1)."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the processor.

        Args:
            **kwargs: Passed to :class:`~pipecat.processors.frame_processor.FrameProcessor`.

        """
        super().__init__(**kwargs)
        self._downlink_frames = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Turn each uplink frame into a downlink frame; pass everything else through."""
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            marker = signals.bot_marker_frame(self._downlink_frames)
            mixed = _mix(frame.audio, marker)
            self._downlink_frames += 1
            await self.push_frame(
                OutputAudioRawFrame(
                    audio=mixed,
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                )
            )
            return

        await self.push_frame(frame, direction)


class SpeakerProcessor(FrameProcessor):
    """Talk continuously until the caller interrupts (scenario 2)."""

    def __init__(self, recorder: TraceRecorder, **kwargs: Any) -> None:
        """Initialize the processor.

        Args:
            recorder: Recorder the speech-generation counters are reported to.
            **kwargs: Passed to :class:`~pipecat.processors.frame_processor.FrameProcessor`.

        """
        super().__init__(**kwargs)
        self._recorder = recorder
        self._generated = 0
        self._speaking = False
        self._interrupted = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Start talking on the first uplink frame, stop on the first interruption."""
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame) and not self._interrupted:
            # Pipecat drops the audio it still holds on its own; this stops us adding more.
            self._interrupted = True
            self._recorder.record("bot_stopped_generating", generated=self._generated)

        if isinstance(frame, InputAudioRawFrame) and not self._speaking:
            self._speaking = True
            await self._speak()
            return

        await self.push_frame(frame, direction)

    async def _speak(self) -> None:
        """Queue the whole utterance at once and let the transport drip it out in real time."""
        self._recorder.record("bot_started_speaking", frames=BOT_SPEECH_FRAMES)
        for index in range(BOT_SPEECH_FRAMES):
            if self._interrupted:
                break
            await self.push_frame(
                OutputAudioRawFrame(
                    audio=signals.bot_speech_frame(index),
                    sample_rate=signals.SAMPLE_RATE,
                    num_channels=1,
                )
            )
            self._generated = index + 1
            # Yield so an interruption arriving mid-burst is processed rather than starved
            # behind a thousand synchronous pushes.
            await asyncio.sleep(0)


def _mix(first: bytes, second: bytes) -> bytes:
    """Sum two little-endian 16-bit PCM buffers, saturating, over their common length."""
    count = min(len(first), len(second)) // 2
    out = bytearray()
    for index in range(count):
        offset = index * 2
        left = int.from_bytes(first[offset : offset + 2], "little", signed=True)
        right = int.from_bytes(second[offset : offset + 2], "little", signed=True)
        total = max(-32768, min(32767, left + right))
        out += total.to_bytes(2, "little", signed=True)
    return bytes(out)


def build_worker(mode: str, host: str, port: int, trace: TextIO) -> PipelineWorker:
    """Assemble the transport, the serializer and the pipeline for one mode."""
    serializer = SiphonFrameSerializer(
        params=SiphonFrameSerializer.InputParams(
            # The engine's own VAD drives turn taking. "vad" is the mapping that feeds pipecat's
            # VADUserTurnStartStrategy, so the engine replaces a local VAD with no extra wiring.
            speech_frames="vad",
            stereo_input="caller",
        ),
    )
    transport = SingleClientWebsocketServerTransport(
        host=host,
        port=port,
        params=SingleClientWebsocketServerParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=signals.SAMPLE_RATE,
            audio_out_sample_rate=signals.SAMPLE_RATE,
            # Load-bearing. One WebSocket binary message becomes exactly one RTP packet in the
            # engine, whatever its length, and the RTP timestamp still advances by one ptime.
            # Pipecat's default of four 10 ms chunks would put 40 ms of audio in every packet
            # and play the bot back at double speed.
            audio_out_10ms_chunks=signals.PTIME_MS // 10,
            serializer=serializer,
        ),
    )

    recorder = TraceRecorder(trace)
    speaker: FrameProcessor = EchoProcessor() if mode == "echo" else SpeakerProcessor(recorder)
    # Explicit strategies. Pipecat's defaults would pull in the neural smart-turn analyser,
    # which is a model download and a wall-clock decision — neither belongs in a fixture-driven
    # test. The start strategy is the one that matters here: it turns the engine's
    # speech_started into the interruption that flushes the downlink.
    turn_processor = UserTurnProcessor(
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy()],
            stop=[SpeechTimeoutUserTurnStopStrategy(wait_for_transcript=False)],
        ),
    )

    @transport.event_handler("on_websocket_ready")
    async def on_websocket_ready(_transport: object) -> None:
        recorder.record("listening", host=host, port=port, mode=mode)
        logger.info(f"harness bot listening on ws://{host}:{port}/ in {mode} mode")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport: object, _client: object) -> None:
        recorder.record("engine_connected")
        logger.info("engine connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport: object, _client: object) -> None:
        recorder.record(
            "engine_disconnected",
            uplink_bytes=recorder.uplink_bytes,
            stream_id=serializer.stream_id,
            call_id=serializer.call_id,
            wire_sample_rate=serializer.wire_sample_rate,
            wire_ptime=serializer.media_format.ptime,
            wire_encoding=serializer.media_format.encoding.value,
            wire_channels=serializer.media_format.channels,
            direction=serializer.direction.value,
        )
        logger.info("engine disconnected")

    return PipelineWorker(
        Pipeline([transport.input(), recorder, turn_processor, speaker, transport.output()]),
        params=PipelineParams(
            audio_in_sample_rate=signals.SAMPLE_RATE,
            audio_out_sample_rate=signals.SAMPLE_RATE,
        ),
    )


async def main(mode: str, host: str, port: int, trace_path: Path) -> None:
    """Run the bot until the process is stopped."""
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as trace:
        runner = WorkerRunner()
        await runner.add_workers(build_worker(mode, host, port, trace))
        await runner.run()


def parse_arguments() -> argparse.Namespace:
    """Parse the mode, the bind address and where the trace goes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("echo", "speaker"), required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--trace", required=True, help="JSON Lines trace file to write")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    try:
        asyncio.run(main(arguments.mode, arguments.host, arguments.port, Path(arguments.trace)))
    except KeyboardInterrupt:
        sys.exit(0)
