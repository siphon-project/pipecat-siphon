"""Tell the harness what the bot heard, decided and said on a call, one JSON line per event.

A call is judged from two sides. The wire shows what the caller got; these events show why: which
transcripts the echo guard let through, where the caller's turn ended, what the model was asked,
which tools it called and what the bot said. `bot_events.parse_bot_log` reads the lines back.

An observer rather than a processor, so the pipeline under test is the one the example ships. Each
frame is written once, however many processors it passes through and whichever way it is broadcast,
because the checks count events: two `user_turn_stopped` for one turn would read as the caller
being cut off.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Mapping

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
    FunctionCallInProgressFrame,
    LLMContextFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator

from tools.conversation.bot_events import EVENT_MARKER

_EDGES: Mapping[type[Frame], str] = {
    VADUserStartedSpeakingFrame: "engine_speech_started",
    VADUserStoppedSpeakingFrame: "engine_speech_stopped",
    UserStartedSpeakingFrame: "user_turn_started",
    UserStoppedSpeakingFrame: "user_turn_stopped",
    BotStartedSpeakingFrame: "bot_started_speaking",
    BotStoppedSpeakingFrame: "bot_stopped_speaking",
}
"""Frames that are an event by their type alone, looked up by exact type."""


def _write_line(line: str) -> None:
    """Write one event line to standard output, where the container log keeps it in order."""
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


class CallEventObserver(BaseObserver):
    """Write an event line for each frame of a call that says what the bot did."""

    def __init__(
        self,
        *,
        call_id: Callable[[], str | None],
        write: Callable[[str], None] = _write_line,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Label events with `call_id()`, the SIP Call-ID once known, and time them by `clock`.

        Wall-clock time, because the harness lines the events up with a packet capture taken in
        another container on the same host.
        """
        super().__init__()
        self._call_id = call_id
        self._write = write
        self._clock = clock
        self._seen: set[int] = set()
        self._accepted: set[int] = set()
        self._pending: list[tuple[str, float, dict[str, object]]] = []

    async def on_push_frame(self, data: FramePushed) -> None:
        """Turn a frame passing between two processors into an event, the first time it passes."""
        frame = data.frame
        edge = _EDGES.get(type(frame))
        if edge is not None:
            if self._first_sighting(frame):
                self._emit(edge, {})
            return

        if isinstance(frame, TranscriptionFrame):
            fields: dict[str, object] = {
                "text": frame.text,
                "language": str(frame.language) if frame.language else None,
            }
            if self._first_sighting(frame):
                self._emit("transcript_heard", fields)
            # Whatever reaches the turn logic got past the echo guard.
            if isinstance(data.destination, LLMUserAggregator) and frame.id not in self._accepted:
                self._accepted.add(frame.id)
                self._emit("transcript_accepted", fields)
        elif isinstance(frame, LLMContextFrame):
            if self._first_sighting(frame):
                self._emit("llm_run", {"user_text": _latest_user_text(frame)})
        elif isinstance(frame, FunctionCallInProgressFrame):
            if self._first_sighting(frame):
                self._emit("tool_called", {"name": frame.function_name})
        elif isinstance(frame, TTSTextFrame):
            if self._first_sighting(frame):
                self._emit("bot_text", {"text": frame.text})
        elif isinstance(frame, ErrorFrame) and self._first_sighting(frame):
            self._emit("pipeline_error", {"message": str(frame.error)})

    def _first_sighting(self, frame: Frame) -> bool:
        """Whether this frame, or the other half of its broadcast, has not been written yet."""
        if frame.id in self._seen or frame.broadcast_sibling_id in self._seen:
            return False
        self._seen.add(frame.id)
        return True

    def _emit(self, event: str, fields: dict[str, object]) -> None:
        """Write the event, holding it until the call it belongs to is known."""
        self._pending.append((event, self._clock(), fields))
        call_id = self._call_id()
        if call_id is None:
            return
        for held_event, wall_time, held_fields in self._pending:
            record = {"event": held_event, "call_id": call_id, "wall_time": wall_time}
            self._write(EVENT_MARKER + json.dumps({**record, **held_fields}, ensure_ascii=False))
        self._pending.clear()


def _latest_user_text(frame: LLMContextFrame) -> str:
    """Return the words of the last user message in the context the model is run on."""
    for message in reversed(frame.context.get_messages()):
        if isinstance(message, Mapping) and message.get("role") == "user":
            return _text_of(message.get("content"))
    return ""


def _text_of(content: object) -> str:
    """Return a message's text, whether it is a string or a list of content parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part["text"]
            for part in content
            if isinstance(part, Mapping)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        )
    return ""
