"""When the scripted caller speaks: once the bot has spoken and gone quiet, and never into it."""

from __future__ import annotations

from tools import signals
from tools.conversation.downlink_activity import ActivityThresholds, DownlinkActivity
from tools.conversation.script_planner import (
    CallTiming,
    LineAudio,
    ScriptPlanner,
    TimelineEvent,
)

FRAME = 0.02
FRAME_BYTES = 160
THRESHOLDS = ActivityThresholds(
    loud_rms=400.0, quiet_rms=250.0, frames_to_open=3, seconds_to_close=0.3
)
TIMING = CallTiming(
    greeting_timeout_seconds=5.0,
    reply_timeout_seconds=4.0,
    quiet_before_speaking_seconds=1.0,
    minimum_bot_speech_seconds=0.3,
)
SPEECH = 3000.0
QUIET = 6.0
SILENCE = signals.pcm_to_alaw(b"\x00\x00" * FRAME_BYTES)


def _audio(fill: int, frames: int) -> bytes:
    """Line audio made of `frames` frames of one recognisable byte."""
    return bytes([fill]) * (FRAME_BYTES * frames)


class _Call:
    """A planner driven tick by tick, with the bot's level chosen for each stretch of the call."""

    def __init__(self, lines: list[LineAudio]) -> None:
        self.activity = DownlinkActivity(THRESHOLDS)
        self.planner = ScriptPlanner(lines, TIMING, self.activity, start_time=0.0)
        self.tick = 0
        self.sent: list[tuple[float, bytes, bool]] = []

    @property
    def now(self) -> float:
        return self.tick * FRAME

    def run(self, seconds: float, bot_level: float) -> None:
        for _ in range(round(seconds / FRAME)):
            self.activity.observe(self.now, bot_level)
            frame = self.planner.next_frame(self.now)
            self.sent.append((self.now, frame.payload, frame.marker))
            self.tick += 1

    def spoken(self) -> list[tuple[float, bytes, bool]]:
        return [sent for sent in self.sent if sent[1] != SILENCE]

    def events(self, kind: str) -> list[TimelineEvent]:
        return [event for event in self.planner.timeline if event.kind == kind]


def _one_line(*, pause_ms: int = 0) -> LineAudio:
    if pause_ms:
        return LineAudio(
            line_id="ask",
            wait_for="greeting",
            segments=(_audio(0x11, 2), _audio(0x22, 2)),
            pauses_ms=(pause_ms, 0),
        )
    return LineAudio(
        line_id="ask", wait_for="greeting", segments=(_audio(0x11, 2),), pauses_ms=(0,)
    )


class TestScriptPlanner:
    def test_nothing_is_said_before_the_greeting(self) -> None:
        call = _Call([_one_line()])
        call.run(3.0, QUIET)

        assert call.spoken() == []
        assert call.events("line_started") == []

    def test_the_first_line_starts_once_the_greeting_has_ended(self) -> None:
        call = _Call([_one_line()])
        call.run(1.0, SPEECH)
        call.run(2.0, QUIET)

        (started,) = call.events("line_started")
        first_time, first_payload, first_marker = call.spoken()[0]
        assert abs(started.time - 2.0) < 1e-9
        assert abs(first_time - 2.0) < 1e-9
        assert (first_payload, first_marker) == (_audio(0x11, 1), True)

    def test_segments_are_sent_with_the_scripted_pause_between_them(self) -> None:
        call = _Call([_one_line(pause_ms=60)])
        call.run(1.0, SPEECH)
        call.run(3.0, QUIET)

        start = next(index for index, sent in enumerate(call.sent) if sent[1] != SILENCE)
        window = call.sent[start : start + 7]
        assert [payload for _, payload, _ in window] == [
            _audio(0x11, 1),
            _audio(0x11, 1),
            SILENCE,
            SILENCE,
            SILENCE,
            _audio(0x22, 1),
            _audio(0x22, 1),
        ]
        assert [marker for _, _, marker in window] == [
            True,
            False,
            False,
            False,
            False,
            True,
            False,
        ]
        assert len(call.events("segment_started")) == 2
        assert len(call.events("line_ended")) == 1

    def test_the_wait_for_a_reply_ignores_speech_before_the_line(self) -> None:
        second = LineAudio(
            line_id="goodbye", wait_for="reply", segments=(_audio(0x33, 1),), pauses_ms=(0,)
        )
        call = _Call([_one_line(), second])
        call.run(1.0, SPEECH)
        call.run(1.2, QUIET)  # the first line plays in here
        call.run(5.0, QUIET)  # the bot never answers it

        assert [event.line_id for event in call.events("line_started")] == ["ask"]
        (timed_out,) = call.events("wait_timed_out")
        assert timed_out.line_id == "goodbye"
        assert _audio(0x33, 1) not in [payload for _, payload, _ in call.sent]

    def test_a_reply_lets_the_next_line_start(self) -> None:
        second = LineAudio(
            line_id="goodbye", wait_for="reply", segments=(_audio(0x33, 1),), pauses_ms=(0,)
        )
        call = _Call([_one_line(), second])
        call.run(1.0, SPEECH)
        call.run(1.2, QUIET)
        call.run(0.5, SPEECH)
        call.run(1.5, QUIET)

        assert [event.line_id for event in call.events("line_started")] == ["ask", "goodbye"]
        assert call.planner.finished

    def test_talking_into_the_bot_is_recorded(self) -> None:
        call = _Call([_one_line(pause_ms=600)])
        call.run(1.0, SPEECH)
        call.run(1.04, QUIET)  # the line starts, its first segment plays
        call.run(0.3, SPEECH)  # the bot starts again during the pause

        (overlap,) = call.events("overlap")
        assert overlap.line_id == "ask"

    def test_only_silence_after_the_last_line(self) -> None:
        call = _Call([_one_line()])
        call.run(1.0, SPEECH)
        call.run(1.1, QUIET)
        assert call.planner.finished

        sent_before = len(call.sent)
        call.run(2.0, SPEECH)
        assert all(payload == SILENCE for _, payload, _ in call.sent[sent_before:])

    def test_line_audio_must_be_whole_frames(self) -> None:
        bad = LineAudio(
            line_id="ask", wait_for="greeting", segments=(b"\x00" * 161,), pauses_ms=(0,)
        )

        try:
            ScriptPlanner([bad], TIMING, DownlinkActivity(THRESHOLDS), start_time=0.0)
        except ValueError as error:
            assert "ask" in str(error)
        else:
            raise AssertionError("audio that is not whole frames was accepted")
