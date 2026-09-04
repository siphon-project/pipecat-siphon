"""Conversational demo bot: a phone caller talks to Claude, over the siphon-rtp media WebSocket.

The parrot in `echo_bot.py` proves the media path. This one proves the product: speech in,
speech out, with the turn taking done by the media engine rather than in the pipeline.

    caller -> siphon-sip -> siphon-rtp --(media WebSocket)--> this bot
                                                              STT -> Claude -> TTS

Three vendors, three keys, all read from the environment:

    export ANTHROPIC_API_KEY=...        # console.anthropic.com -- an API key, not a subscription
    export DEEPGRAM_API_KEY=...         # streaming speech to text
    export CARTESIA_API_KEY=...         # streaming text to speech
    export CARTESIA_VOICE_ID=...        # a voice from the Cartesia dashboard

    pip install "pipecat-ai[anthropic,deepgram,cartesia]"
    python examples/agent_bot.py --host 0.0.0.0 --port 9001

Then point the engine at it, in the `profile` of a native-JSON `offer` (or `answer_local`)::

    {
      "ws_uri": "ws://198.51.100.10:9001/stream",
      "ws_sample_rate": 16000,
      "ws_vad": true,
      "ws_barge_in": true,
      "ws_vad_engine": "neural",
      "ws_vad_min_speech_ms": 100,
      "echo_cancellation": true
    }

`ws_sample_rate` matters twice over: 16 kHz is what the recognizer wants from an 8 kHz G.711 call,
and setting the pipeline to the same number means neither side builds a resampler.

Swapping a vendor is a one-line change -- pipecat ships around seventy service integrations, and
nothing below depends on which three you pick.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any

from loguru import logger
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
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

PIPELINE_SAMPLE_RATE = 16000
"""What the pipeline runs at. Match it to `ws_sample_rate` on the offer profile and nothing in
this process resamples: the engine converts once, at the RTP boundary, where it is frame-exact."""

MODEL = "claude-opus-5"
"""pipecat defaults its Anthropic service to an older model, so this is set explicitly."""

MAXIMUM_RESPONSE_TOKENS = 400
"""A spoken answer is short. This is a ceiling against a runaway monologue on the phone, not a
target -- 400 tokens is a couple of minutes of speech, far past anything the prompt asks for."""

TURN_END_SILENCE_SECONDS = 0.2
"""How long pipecat waits after the engine reports end-of-speech before running the model.

The engine has already applied its own VAD hangover (`ws_vad_hangover_ms`, ~200 ms by default),
so pipecat's own 0.6 s default would be a second silence budget stacked on the first one. This
still waits for the recognizer's final transcript, which is the part that must not be cut."""

SYSTEM_PROMPT = """You are a voice assistant on a telephone call. You are talking, not writing.

Keep answers to one or two sentences unless you are explicitly asked for detail. Ask one question
at a time. Never use markdown, bullet points, headings, emoji or any other formatting: everything
you write is read aloud verbatim, so write numbers, dates and units as a person would say them.

The caller can interrupt you at any time and you will be cut off mid-sentence when they do. That
is normal. When it happens, answer what they just said rather than finishing your last thought.

Open the call by greeting the caller and asking how you can help."""


def _required_environment_variable(name: str, purpose: str) -> str:
    """Read an API key from the environment, or exit saying exactly which one is missing."""
    value = os.environ.get(name)
    if not value:
        sys.exit(f"{name} is not set. It is the {purpose}; export it and run again.")
    return value


def build_transport(host: str, port: int) -> SingleClientWebsocketServerTransport:
    """Build the WebSocket server transport the engine dials into."""
    return SingleClientWebsocketServerTransport(
        host=host,
        port=port,
        params=SingleClientWebsocketServerParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=PIPELINE_SAMPLE_RATE,
            audio_out_sample_rate=PIPELINE_SAMPLE_RATE,
            serializer=SiphonFrameSerializer(
                params=SiphonFrameSerializer.InputParams(
                    # The engine's VAD drives the turn, so there is no VAD analyzer in this
                    # pipeline at all -- its edges arrive as the frames pipecat's own turn-start
                    # strategy already consumes.
                    speech_frames="vad",
                    stereo_input="caller",
                ),
            ),
        ),
    )


def build_llm_service(fast: bool) -> AnthropicLLMService:
    """Build the Claude service, tuned for a conversation happening in real time."""
    extra: dict[str, Any] = {
        # Effort is the thinking-depth dial. A phone turn is answered in one or two sentences and
        # the caller is listening to silence while the model works, so this is the end of the
        # range that belongs on a voice route.
        "output_config": {"effort": "low"},
        # A policy refusal mid-call would otherwise be dead air. This re-runs the same request on
        # a fallback model inside the same call, billed at the fallback's own rates.
        "fallbacks": "default",
        "betas": ["server-side-fallback-2026-07-01"],
    }
    if fast:
        # Research preview, Opus 5 and 4.8 only: the same model at up to 2.5x the output rate,
        # at premium pricing. On a call, output rate is time-to-first-word.
        extra["speed"] = "fast"
        extra["betas"] = [*extra["betas"], "fast-mode-2026-02-01"]

    return AnthropicLLMService(
        api_key=_required_environment_variable("ANTHROPIC_API_KEY", "Claude API key"),
        settings=AnthropicLLMService.Settings(
            model=MODEL,
            system_instruction=SYSTEM_PROMPT,
            max_tokens=MAXIMUM_RESPONSE_TOKENS,
            # The system prompt is resent on every turn of the call; caching it is free money.
            enable_prompt_caching=True,
            extra=extra,
        ),
    )


def build_speech_to_text() -> DeepgramSTTService:
    """Build the recognizer, at the pipeline's rate."""
    return DeepgramSTTService(
        api_key=_required_environment_variable("DEEPGRAM_API_KEY", "speech-to-text key"),
        # Left on pipecat's own default model rather than pinned here: a model string in an
        # example is a thing to maintain, and the default is the current streaming one.
        sample_rate=PIPELINE_SAMPLE_RATE,
    )


def build_text_to_speech() -> CartesiaTTSService:
    """Build the voice, at the pipeline's rate."""
    return CartesiaTTSService(
        api_key=_required_environment_variable("CARTESIA_API_KEY", "text-to-speech key"),
        settings=CartesiaTTSService.Settings(
            voice=_required_environment_variable("CARTESIA_VOICE_ID", "voice to speak with"),
        ),
        sample_rate=PIPELINE_SAMPLE_RATE,
    )


def build_worker(host: str, port: int, fast: bool) -> PipelineWorker:
    """Assemble the whole call: transport, recognizer, model, voice, and the turn machinery."""
    transport = build_transport(host, port)
    speech_to_text = build_speech_to_text()
    llm = build_llm_service(fast)
    text_to_speech = build_text_to_speech()

    # The conversation itself. The system prompt lives on the service (and is cached there), so
    # the context starts empty and fills with the call.
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(context)

    # Explicit strategies. Pipecat's defaults would reach for its neural smart-turn analyzer,
    # which is a model download and a second opinion about something the engine has already
    # decided. Start on the engine's speech edge, stop on its endpoint plus the transcript.
    turn_processor = UserTurnProcessor(
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy()],
            stop=[
                SpeechTimeoutUserTurnStopStrategy(
                    user_speech_timeout=TURN_END_SILENCE_SECONDS,
                    wait_for_transcript=True,
                )
            ],
        ),
    )

    worker = PipelineWorker(
        Pipeline(
            [
                transport.input(),
                speech_to_text,
                turn_processor,
                aggregators.user(),
                llm,
                text_to_speech,
                transport.output(),
                aggregators.assistant(),
            ]
        ),
        params=PipelineParams(
            audio_in_sample_rate=PIPELINE_SAMPLE_RATE,
            audio_out_sample_rate=PIPELINE_SAMPLE_RATE,
        ),
    )

    @transport.event_handler("on_websocket_ready")
    async def on_websocket_ready(_transport: object) -> None:
        logger.info(f"listening on ws://{host}:{port}/ -- point the engine's ws_uri here")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport: object, _client: object) -> None:
        logger.info("engine connected, greeting the caller")
        # The bot speaks first. Running the model against an empty context makes it follow the
        # system prompt's last line, so the greeting is generated rather than canned.
        await worker.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport: object, _client: object) -> None:
        logger.info("engine disconnected, call over")

    return worker


async def main(host: str, port: int, fast: bool) -> None:
    """Answer calls until the process is stopped."""
    worker = build_worker(host, port, fast)
    runner = WorkerRunner()
    await runner.add_workers(worker)
    await runner.run()


def parse_arguments() -> argparse.Namespace:
    """Parse the bind address and the latency options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="address to bind (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=9001, help="port to bind (default 9001)")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="run the model in fast mode: same model, up to 2.5x the output rate, premium price",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    try:
        asyncio.run(main(arguments.host, arguments.port, arguments.fast))
    except KeyboardInterrupt:
        sys.exit(0)
