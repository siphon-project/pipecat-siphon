"""What the scripted caller sends on each 20 ms tick: silence, or the next frame of a line.

The caller behaves like a person who waits their turn. A line starts only once the bot has said
something since the caller last spoke (the greeting, for the first line) and has then been quiet
for a while; until then, and between and after lines, the caller sends silence rather than nothing,
because the engine only ends a caller turn while RTP keeps arriving.

Everything that decides timing lives here, with the clock passed in, so it is tested tick by tick
without sockets. The driver around it only moves packets.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from tools import signals
from tools.conversation.downlink_activity import TIME_TOLERANCE_SECONDS, DownlinkActivity
from tools.conversation.scenario import FRAME_MILLISECONDS, WaitFor

FRAME_BYTES = 160
"""One 20 ms frame of 8 kHz G.711: one byte per sample."""

FRAME_SECONDS = FRAME_MILLISECONDS / 1000

EventKind = Literal[
    "line_started",
    "segment_started",
    "segment_ended",
    "line_ended",
    "wait_timed_out",
    "overlap",
]

_State = Literal["waiting", "speaking", "pausing", "finished", "stalled"]


@dataclass(frozen=True)
class CallTiming:
    """How long the caller waits, and for what, before each line."""

    greeting_timeout_seconds: float
    """How long the first line waits for the bot's greeting to be said and finished."""
    reply_timeout_seconds: float
    """How long every later line waits for the bot's reply to the line before it."""
    quiet_before_speaking_seconds: float
    """How long the bot must have been quiet before the caller speaks."""
    minimum_bot_speech_seconds: float
    """How much the bot must have said since the caller last spoke. Less is not a reply."""


@dataclass(frozen=True)
class LineAudio:
    """One scripted line, encoded: G.711 A-law per segment and the pause after each."""

    line_id: str
    wait_for: WaitFor
    segments: tuple[bytes, ...]
    pauses_ms: tuple[int, ...]


@dataclass(frozen=True)
class PlannedFrame:
    """The payload to send this tick, and whether it opens a talkspurt."""

    payload: bytes
    marker: bool


@dataclass(frozen=True)
class TimelineEvent:
    """Something the caller did or saw, for the call's timeline."""

    time: float
    kind: EventKind
    line_id: str


class ScriptPlanner:
    """Decides, tick by tick, what the caller sends."""

    def __init__(
        self,
        lines: Sequence[LineAudio],
        timing: CallTiming,
        activity: DownlinkActivity,
        *,
        start_time: float,
    ) -> None:
        """Plan `lines` from `start_time`, judging the bot's speech through `activity`.

        Raises:
            ValueError: When a line's audio is not whole frames or its pauses do not fit it.

        """
        for line in lines:
            _check_line(line)
        self._lines = list(lines)
        self._timing = timing
        self._activity = activity
        self._silence = signals.pcm_to_alaw(b"\x00\x00" * FRAME_BYTES)
        self.timeline: list[TimelineEvent] = []

        self._state: _State = "waiting" if self._lines else "finished"
        self._line_index = 0
        self._wait_started = start_time
        self._segment_index = 0
        self._frame_index = 0
        self._pause_frames_left = 0
        self._overlapped = False
        activity.mark(start_time)

    @property
    def finished(self) -> bool:
        """Whether every line has been said."""
        return self._state == "finished"

    def next_frame(self, now: float) -> PlannedFrame:
        """Return what to send for the tick starting at `now`."""
        if self._state == "waiting":
            self._start_line_if_ready(now)
        if self._state == "speaking":
            return self._speak(now)
        if self._state == "pausing":
            return self._pause(now)
        return PlannedFrame(payload=self._silence, marker=False)

    def _start_line_if_ready(self, now: float) -> None:
        line = self._lines[self._line_index]
        if self._activity.ready_to_speak(
            now,
            minimum_speech_seconds=self._timing.minimum_bot_speech_seconds,
            quiet_seconds=self._timing.quiet_before_speaking_seconds,
        ):
            self._record(now, "line_started")
            self._state = "speaking"
            self._segment_index = 0
            self._frame_index = 0
            self._overlapped = False
            return
        timeout = (
            self._timing.greeting_timeout_seconds
            if line.wait_for == "greeting"
            else self._timing.reply_timeout_seconds
        )
        if now - self._wait_started >= timeout - TIME_TOLERANCE_SECONDS:
            self._record(now, "wait_timed_out")
            self._state = "stalled"

    def _speak(self, now: float) -> PlannedFrame:
        self._note_overlap(now)
        segment = self._lines[self._line_index].segments[self._segment_index]
        marker = self._frame_index == 0
        if marker:
            self._record(now, "segment_started")
        offset = self._frame_index * FRAME_BYTES
        payload = segment[offset : offset + FRAME_BYTES]
        self._frame_index += 1
        if self._frame_index * FRAME_BYTES >= len(segment):
            self._end_segment(now + FRAME_SECONDS)
        return PlannedFrame(payload=payload, marker=marker)

    def _pause(self, now: float) -> PlannedFrame:
        self._note_overlap(now)
        self._pause_frames_left -= 1
        if self._pause_frames_left <= 0:
            self._segment_index += 1
            self._frame_index = 0
            self._state = "speaking"
        return PlannedFrame(payload=self._silence, marker=False)

    def _end_segment(self, end: float) -> None:
        line = self._lines[self._line_index]
        self._record(end, "segment_ended")
        if self._segment_index + 1 < len(line.segments):
            self._pause_frames_left = line.pauses_ms[self._segment_index] // FRAME_MILLISECONDS
            if self._pause_frames_left > 0:
                self._state = "pausing"
            else:
                self._segment_index += 1
                self._frame_index = 0
            return
        self._record(end, "line_ended")
        # Only speech after this line is a reply to it.
        self._activity.mark(end)
        self._line_index += 1
        if self._line_index >= len(self._lines):
            self._state = "finished"
        else:
            self._state = "waiting"
            self._wait_started = end

    def _note_overlap(self, now: float) -> None:
        if not self._overlapped and self._activity.quiet_seconds(now) <= 0.0:
            self._record(now, "overlap")
            self._overlapped = True

    def _record(self, time: float, kind: EventKind) -> None:
        line_id = self._lines[self._line_index].line_id
        self.timeline.append(TimelineEvent(time=time, kind=kind, line_id=line_id))


def _check_line(line: LineAudio) -> None:
    if not line.segments:
        raise ValueError(f"line {line.line_id!r} has no audio")
    if len(line.pauses_ms) != len(line.segments):
        raise ValueError(f"line {line.line_id!r} needs one pause per segment")
    for index, segment in enumerate(line.segments):
        if not segment or len(segment) % FRAME_BYTES:
            raise ValueError(
                f"line {line.line_id!r} segment {index} is {len(segment)} bytes, not whole "
                f"{FRAME_BYTES}-byte frames"
            )
    if line.pauses_ms[-1]:
        raise ValueError(f"line {line.line_id!r} pauses after its last segment")
    if any(pause < 0 or pause % FRAME_MILLISECONDS for pause in line.pauses_ms):
        raise ValueError(
            f"line {line.line_id!r} has a pause that is not whole {FRAME_MILLISECONDS} ms frames"
        )
