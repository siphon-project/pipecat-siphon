"""Minimal siphon-rtp bot: a WebSocket server that parrots the caller back into the call.

siphon-rtp dials out, so this is a *server*. Start it, then offer a call to the engine with
``"profile": {"ws_uri": "ws://127.0.0.1:9001/stream"}`` and you hear yourself back, roughly one
jitter-buffer frame plus one playout frame later.

Swap :class:`ParrotProcessor` for an STT -> LLM -> TTS chain and this is a voice agent. The
serializer, the transport and the wiring below do not change.

Run it::

    python examples/echo_bot.py --host 127.0.0.1 --port 9001
"""

from __future__ import annotations

import argparse
import asyncio

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    OutputAudioRawFrame,
    UserStartedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.websocket.server import (
    SingleClientWebsocketServerParams,
    SingleClientWebsocketServerTransport,
)
from pipecat.workers.runner import WorkerRunner

from pipecat_siphon import SiphonFrameSerializer

# The wire rate the engine will announce in `start`, which is what this pipeline is built to run
# at so nothing has to be resampled. Leave the offer profile alone and the engine uses the leg's own
# codec rate (8000 for G.711); from siphon-rtp 0.3.0 a controller picks it independently of the
# codec with `ws_sample_rate`, in which case set both to the same number:
#
#     "profile": {"ws_uri": "ws://127.0.0.1:9001/stream", "ws_sample_rate": 16000}
#
# The serializer reads the real rate out of `start` either way and resamples if they disagree; this
# constant only decides what the pipeline itself runs at.
WIRE_SAMPLE_RATE = 8000


class ParrotProcessor(FrameProcessor):
    """Send the caller's audio straight back into the call, and log the interesting edges."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Echo input audio as output audio; pass everything else through untouched."""
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            await self.push_frame(
                OutputAudioRawFrame(
                    audio=frame.audio,
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                )
            )
            return

        if isinstance(frame, UserStartedSpeakingFrame):
            logger.info("caller started speaking")
        elif isinstance(frame, InputDTMFFrame):
            logger.info(f"caller pressed {frame.button.value}")

        await self.push_frame(frame, direction)


def build_transport(host: str, port: int) -> SingleClientWebsocketServerTransport:
    """Build the WebSocket server transport the engine will dial into."""
    return SingleClientWebsocketServerTransport(
        host=host,
        port=port,
        params=SingleClientWebsocketServerParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=WIRE_SAMPLE_RATE,
            audio_out_sample_rate=WIRE_SAMPLE_RATE,
            serializer=SiphonFrameSerializer(
                params=SiphonFrameSerializer.InputParams(
                    # The engine's own VAD (profile flag `ws_vad`) drives turn taking, so the
                    # pipeline needs no local VAD. Its edges arrive as the VAD frames pipecat's
                    # default turn-start strategy already consumes.
                    speech_frames="vad",
                    # A tee (`ws_tee`) is stereo: channel 0 caller, channel 1 callee.
                    stereo_input="caller",
                ),
            ),
        ),
    )


async def main(host: str, port: int) -> None:
    """Run the parrot bot until interrupted."""
    transport = build_transport(host, port)

    @transport.event_handler("on_websocket_ready")
    async def on_websocket_ready(_transport: object) -> None:
        logger.info(f"listening on ws://{host}:{port}/ -- point the engine's ws_uri here")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport: object, _client: object) -> None:
        logger.info("engine connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport: object, _client: object) -> None:
        logger.info("engine disconnected")

    worker = PipelineWorker(
        Pipeline([transport.input(), ParrotProcessor(), transport.output()]),
        params=PipelineParams(
            audio_in_sample_rate=WIRE_SAMPLE_RATE,
            audio_out_sample_rate=WIRE_SAMPLE_RATE,
        ),
    )

    runner = WorkerRunner()
    await runner.add_workers(worker)
    await runner.run()


def parse_arguments() -> argparse.Namespace:
    """Parse the host/port the WebSocket server binds to."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="address to bind (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=9001, help="port to bind (default 9001)")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    asyncio.run(main(arguments.host, arguments.port))
