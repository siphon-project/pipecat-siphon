"""The bot's pipeline told as events: what it heard, decided and said, one line each."""

from __future__ import annotations

from typing import Any

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
    FunctionCallInProgressFrame,
    InputAudioRawFrame,
    LLMContextFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from tools.conversation.bot_events import parse_bot_log
from tools.conversation.call_events import CallEventObserver

CALL = "conversation-goodbye-1-1@172.28.8.40"


class _Pipeline:
    """An observer, the processors frames travel between, and the lines it wrote."""

    def __init__(self, call_id: str | None = CALL) -> None:
        self.call_id = call_id
        self.now = 1000.0
        self.lines: list[str] = []
        self.observer = CallEventObserver(
            call_id=lambda: self.call_id, write=self.lines.append, clock=lambda: self.now
        )
        self.recognizer = FrameProcessor()
        self.echo_guard = FrameProcessor()
        self.turn_logic = LLMContextAggregatorPair(LLMContext()).user()
        self.model = FrameProcessor()

    async def push(
        self,
        frame: Frame,
        source: FrameProcessor | None = None,
        destination: FrameProcessor | None = None,
    ) -> None:
        await self.observer.on_push_frame(
            FramePushed(
                source=source or self.recognizer,
                destination=destination or self.model,
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )

    def events(self) -> list[tuple[str, dict[str, Any]]]:
        log = parse_bot_log("\n".join(self.lines))
        assert log.unreadable == 0
        return [(event.event, dict(event.fields)) for event in log.events]


def _transcript(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text, "caller", time_now_iso8601())


class TestTranscripts:
    async def test_a_transcript_is_heard_once_and_accepted_once_it_reaches_the_turn_logic(
        self,
    ) -> None:
        pipeline = _Pipeline()
        kept = _transcript("My name is Karen.")
        dropped = _transcript("Hello, how can I help?")

        await pipeline.push(kept, pipeline.recognizer, pipeline.echo_guard)
        await pipeline.push(kept, pipeline.echo_guard, pipeline.turn_logic)
        await pipeline.push(dropped, pipeline.recognizer, pipeline.echo_guard)

        assert pipeline.events() == [
            ("transcript_heard", {"text": "My name is Karen.", "language": None}),
            ("transcript_accepted", {"text": "My name is Karen.", "language": None}),
            ("transcript_heard", {"text": "Hello, how can I help?", "language": None}),
        ]

    async def test_the_language_the_recognizer_reported_is_kept(self) -> None:
        """It decides whether a turn that stops mid-sentence is held, so a failure needs it."""
        pipeline = _Pipeline()
        frame = TranscriptionFrame("My name is", "caller", time_now_iso8601(), Language.EN)

        await pipeline.push(frame, pipeline.recognizer, pipeline.echo_guard)

        assert pipeline.events() == [("transcript_heard", {"text": "My name is", "language": "en"})]


class TestOneEventPerFrame:
    async def test_a_frame_seen_on_every_hop_is_one_event(self) -> None:
        pipeline = _Pipeline()
        frame = BotStartedSpeakingFrame()

        for _ in range(3):
            await pipeline.push(frame)

        assert pipeline.events() == [("bot_started_speaking", {})]

    async def test_a_broadcast_is_one_event(self) -> None:
        """A broadcast is two frames, one each way, and still one turn ending."""
        pipeline = _Pipeline()
        downstream = UserStoppedSpeakingFrame()
        upstream = UserStoppedSpeakingFrame()
        downstream.broadcast_sibling_id = upstream.id
        upstream.broadcast_sibling_id = downstream.id

        await pipeline.push(downstream)
        await pipeline.push(upstream)

        assert pipeline.events() == [("user_turn_stopped", {})]

    async def test_audio_is_not_an_event(self) -> None:
        pipeline = _Pipeline()

        await pipeline.push(
            InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1)
        )

        assert pipeline.lines == []


class TestWhatTheBotDid:
    async def test_speech_edges_turns_and_speaking(self) -> None:
        pipeline = _Pipeline()

        for frame in (
            VADUserStartedSpeakingFrame(),
            UserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
        ):
            await pipeline.push(frame)

        assert [event for event, _ in pipeline.events()] == [
            "engine_speech_started",
            "user_turn_started",
            "engine_speech_stopped",
            "user_turn_stopped",
            "bot_started_speaking",
            "bot_stopped_speaking",
        ]

    async def test_a_model_run_carries_the_callers_latest_words(self) -> None:
        pipeline = _Pipeline()
        context = LLMContext(
            messages=[
                {"role": "user", "content": "[The caller has just come on the line.]"},
                {"role": "assistant", "content": "Hello, how can I help?"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "My name is"},
                        {"type": "text", "text": "Karen."},
                    ],
                },
            ]
        )

        await pipeline.push(LLMContextFrame(context=context))

        assert pipeline.events() == [("llm_run", {"user_text": "My name is Karen."})]

    async def test_tools_words_and_errors(self) -> None:
        pipeline = _Pipeline()

        await pipeline.push(
            FunctionCallInProgressFrame(
                function_name="transfer_call", tool_call_id="call-1", arguments={}
            )
        )
        await pipeline.push(TTSTextFrame(text="Putting you through now.", aggregated_by="sentence"))
        await pipeline.push(ErrorFrame(error="the recognizer refused the connection"))

        assert pipeline.events() == [
            ("tool_called", {"name": "transfer_call"}),
            ("bot_text", {"text": "Putting you through now."}),
            ("pipeline_error", {"message": "the recognizer refused the connection"}),
        ]


class TestTheCall:
    async def test_every_event_carries_the_call_and_when_it_happened(self) -> None:
        pipeline = _Pipeline()
        pipeline.now = 1234.5

        await pipeline.push(BotStartedSpeakingFrame())

        (event,) = parse_bot_log("\n".join(pipeline.lines)).events
        assert event.call_id == CALL
        assert event.wall_time == 1234.5

    async def test_events_before_the_call_is_known_are_written_once_it_is(self) -> None:
        """The greeting is asked for as the socket opens, before the call's `start` is read."""
        pipeline = _Pipeline(call_id=None)
        pipeline.now = 10.0
        await pipeline.push(VADUserStartedSpeakingFrame())
        assert pipeline.lines == []

        pipeline.call_id = CALL
        pipeline.now = 11.0
        await pipeline.push(BotStartedSpeakingFrame())

        events = parse_bot_log("\n".join(pipeline.lines)).events
        assert [(event.event, event.call_id, event.wall_time) for event in events] == [
            ("engine_speech_started", CALL, 10.0),
            ("bot_started_speaking", CALL, 11.0),
        ]
