"""End-to-end smoke test over a real loopback WebSocket.

A real ``SingleClientWebsocketServerTransport`` is stood up on 127.0.0.1 with the serializer
attached, and the test plays the part siphon-rtp plays: it dials in as the WebSocket *client*,
sends a recorded exchange (``start``, binary audio, ``speech_started``, ``dtmf``, ``stop``), and
asserts both the frames that reached the pipeline and the bytes that came back out.

No external network: the server binds an ephemeral loopback port.
"""

from __future__ import annotations

import asyncio
import json
import math
import socket
import struct
from typing import TypeVar

import pytest
import websockets
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.websocket.server import (
    SingleClientWebsocketServerParams,
    SingleClientWebsocketServerTransport,
)
from pipecat.workers.runner import WorkerRunner

import wire_fixtures as wire
from pipecat_siphon import SiphonFrameSerializer

WIRE_RATE = 8000
PTIME_SAMPLES = 160  # 20 ms at 8 kHz

_FrameT = TypeVar("_FrameT", bound=Frame)


def tone(samples: int, *, sample_rate: int = WIRE_RATE) -> bytes:
    """Build a deterministic little-endian 16-bit 440 Hz tone."""
    values = [
        int(12000 * math.sin(2 * math.pi * 440.0 * index / sample_rate)) for index in range(samples)
    ]
    return struct.pack(f"<{len(values)}h", *values)


def free_loopback_port() -> int:
    """Reserve and release an ephemeral loopback port for the server to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class RecordingEchoProcessor(FrameProcessor):
    """Record every frame that reaches it, and echo the caller's audio back into the call."""

    def __init__(self) -> None:
        """Start with an empty recording."""
        super().__init__()
        self.seen: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Record the frame, then forward it (echoing audio as bot output)."""
        await super().process_frame(frame, direction)
        self.seen.append(frame)
        if isinstance(frame, InputAudioRawFrame):
            await self.push_frame(
                OutputAudioRawFrame(
                    audio=frame.audio,
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                )
            )
            return
        await self.push_frame(frame, direction)

    def frames_of(self, frame_type: type[_FrameT]) -> list[_FrameT]:
        """Return every recorded frame of the given type."""
        return [frame for frame in self.seen if isinstance(frame, frame_type)]


async def test_a_recorded_engine_exchange_drives_a_real_pipeline() -> None:
    port = free_loopback_port()
    ready = asyncio.Event()
    recorder = RecordingEchoProcessor()

    transport = SingleClientWebsocketServerTransport(
        host="127.0.0.1",
        port=port,
        params=SingleClientWebsocketServerParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=WIRE_RATE,
            audio_out_sample_rate=WIRE_RATE,
            audio_out_end_silence_secs=0,
            add_wav_header=False,
            serializer=SiphonFrameSerializer(),
        ),
    )

    @transport.event_handler("on_websocket_ready")
    async def on_websocket_ready(_transport: object) -> None:
        ready.set()

    worker = PipelineWorker(
        Pipeline([transport.input(), recorder, transport.output()]),
        params=PipelineParams(
            audio_in_sample_rate=WIRE_RATE,
            audio_out_sample_rate=WIRE_RATE,
        ),
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    runner_task = asyncio.create_task(runner.run())

    received_binary: list[bytes] = []
    received_text: list[dict[str, object]] = []

    try:
        await asyncio.wait_for(ready.wait(), timeout=20)

        async with websockets.connect(f"ws://127.0.0.1:{port}/stream") as engine:
            # The engine's first text frame announces the leg and the negotiated wire format.
            await engine.send(wire.START_8K)

            # A second of caller audio, one 20 ms binary frame per ptime.
            for _ in range(50):
                await engine.send(tone(PTIME_SAMPLES))
                await asyncio.sleep(0.001)

            # A local-VAD turn boundary and a keypress on the same socket.
            await engine.send(wire.SPEECH_STARTED)
            await engine.send(wire.DTMF)

            # Collect whatever the bot wrote back before closing the stream.
            deadline = asyncio.get_running_loop().time() + 5
            while asyncio.get_running_loop().time() < deadline:
                remaining = deadline - asyncio.get_running_loop().time()
                try:
                    message = await asyncio.wait_for(engine.recv(), timeout=remaining)
                except (TimeoutError, websockets.ConnectionClosed):
                    break
                if isinstance(message, bytes):
                    received_binary.append(message)
                    if len(received_binary) >= 20:
                        break
                else:
                    received_text.append(json.loads(message))

            await engine.send(wire.STOP)
            await asyncio.sleep(0.5)

        await asyncio.wait_for(runner_task, timeout=20)
    finally:
        if not runner_task.done():
            await worker.cancel()
            runner_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await runner_task

    audio_in = recorder.frames_of(InputAudioRawFrame)
    assert audio_in, "no caller audio reached the pipeline"
    assert all(frame.sample_rate == WIRE_RATE for frame in audio_in)

    assert recorder.frames_of(VADUserStartedSpeakingFrame), "speech_started never reached the bot"

    dtmf = recorder.frames_of(InputDTMFFrame)
    assert dtmf, "dtmf never reached the bot"
    assert dtmf[0].button.value == "5"

    assert received_binary, "the bot's echoed audio never reached the engine"
    assert all(len(payload) % 2 == 0 for payload in received_binary)


async def test_an_interruption_reaches_the_engine_as_a_clear_over_the_socket() -> None:
    port = free_loopback_port()
    ready = asyncio.Event()

    transport = SingleClientWebsocketServerTransport(
        host="127.0.0.1",
        port=port,
        params=SingleClientWebsocketServerParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=WIRE_RATE,
            audio_out_sample_rate=WIRE_RATE,
            audio_out_end_silence_secs=0,
            serializer=SiphonFrameSerializer(),
        ),
    )

    @transport.event_handler("on_websocket_ready")
    async def on_websocket_ready(_transport: object) -> None:
        ready.set()

    worker = PipelineWorker(
        Pipeline([transport.input(), transport.output()]),
        params=PipelineParams(
            audio_in_sample_rate=WIRE_RATE,
            audio_out_sample_rate=WIRE_RATE,
        ),
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    runner_task = asyncio.create_task(runner.run())

    clears: list[dict[str, object]] = []
    try:
        await asyncio.wait_for(ready.wait(), timeout=20)
        async with websockets.connect(f"ws://127.0.0.1:{port}/stream") as engine:
            await engine.send(wire.START_8K)
            await asyncio.sleep(0.2)

            await worker.queue_frame(InterruptionFrame())

            deadline = asyncio.get_running_loop().time() + 5
            while asyncio.get_running_loop().time() < deadline:
                remaining = deadline - asyncio.get_running_loop().time()
                try:
                    message = await asyncio.wait_for(engine.recv(), timeout=remaining)
                except (TimeoutError, websockets.ConnectionClosed):
                    break
                if isinstance(message, str):
                    parsed = json.loads(message)
                    if parsed.get("type") == "clear":
                        clears.append(parsed)
                        break

            await engine.send(wire.STOP)
            await asyncio.sleep(0.5)

        await asyncio.wait_for(runner_task, timeout=20)
    finally:
        if not runner_task.done():
            await worker.cancel()
            runner_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await runner_task

    assert clears, "the interruption never reached the engine as a clear"
    assert clears[0]["data"] == {"streamId": wire.STREAM_ID, "reason": "barge_in"}
