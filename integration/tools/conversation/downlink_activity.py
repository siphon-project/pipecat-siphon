"""Whether the bot is speaking on the downlink, decided one 20 ms frame at a time.

The caller must not talk over the bot: the agent's echo guard drops what it hears while the bot
holds the line, so a line played into the bot's reply is a line the bot never gets. The caller
therefore waits until the bot has spoken and then fallen quiet. "Quiet" cannot mean "no packets":
the engine fills an idle leg with comfort noise, so the decision is made on each frame's level.

Hysteresis keeps a syllable boundary from ending a reply and a click from starting one. A span of
speech opens after a run of frames at or above `loud_rms`, is kept alive by any frame at or above
`quiet_rms` (the soft tail of a word is still the bot), and closes after `seconds_to_close` below
it. The same class decides live in the caller and offline over a capture, so one implementation is
what both the timing of a call and its evaluation rest on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TIME_TOLERANCE_SECONDS = 1e-6
"""Slack on every duration comparison. Frame times are sums of 20 ms steps in floating point, so a
window that is exactly full can measure a hair short, and the verdict would flip on rounding."""


@dataclass(frozen=True)
class ActivityThresholds:
    """The levels and durations that separate speech from the line's background."""

    loud_rms: float
    """A frame at or above this RMS is speech, and a run of them opens a span."""
    quiet_rms: float
    """A frame below this RMS is background. Between the two, a frame keeps the current state."""
    frames_to_open: int
    """How many consecutive loud frames open a span. Fewer is a click."""
    seconds_to_close: float
    """How long below `quiet_rms` closes a span. Shorter is a gap between words."""


@dataclass(frozen=True)
class Span:
    """One stretch of bot speech: its first loud frame's start to its last active frame's end."""

    start: float
    end: float


class DownlinkActivity:
    """Speech spans on the downlink, and whether the caller may speak now."""

    def __init__(self, thresholds: ActivityThresholds, *, frame_seconds: float = 0.02) -> None:
        """Track speech with `thresholds`, for frames `frame_seconds` long."""
        self._thresholds = thresholds
        self._frame_seconds = frame_seconds
        self._closed: list[Span] = []
        self._open_start: float | None = None
        self._active_end: float | None = None
        self._candidate_start: float | None = None
        self._candidate_frames = 0
        self._candidate_spoken = 0.0
        self._mark = -math.inf
        self._spoken_since_mark = 0.0

    def observe(self, time: float, level: float) -> None:
        """Take one frame that starts at `time` with RMS `level`."""
        thresholds = self._thresholds
        end = time + self._frame_seconds
        loud = level >= thresholds.loud_rms
        spoken = self._frame_seconds if loud and time >= self._mark else 0.0

        if self._open_start is not None and self._active_end is not None:
            if level >= thresholds.quiet_rms:
                self._active_end = end
                self._spoken_since_mark += spoken
            elif end - self._active_end >= thresholds.seconds_to_close - TIME_TOLERANCE_SECONDS:
                self._closed.append(Span(start=self._open_start, end=self._active_end))
                self._open_start = None
            return

        if not loud:
            self._candidate_start = None
            self._candidate_frames = 0
            self._candidate_spoken = 0.0
            return
        if self._candidate_start is None:
            self._candidate_start = time
        self._candidate_frames += 1
        self._candidate_spoken += spoken
        if self._candidate_frames >= thresholds.frames_to_open:
            self._open_start = self._candidate_start
            self._active_end = end
            self._spoken_since_mark += self._candidate_spoken
            self._candidate_start = None
            self._candidate_frames = 0
            self._candidate_spoken = 0.0

    def mark(self, time: float) -> None:
        """Count speech only from `time` on: the reply to a line, not what came before it."""
        self._mark = time
        self._spoken_since_mark = 0.0

    def spoken_seconds_since_mark(self) -> float:
        """Return how much speech has been heard since the last mark."""
        return self._spoken_since_mark

    def quiet_seconds(self, now: float) -> float:
        """Return how long the bot has been quiet at `now`, infinite if it has never spoken."""
        if self._active_end is None:
            return math.inf
        return now - self._active_end

    def ready_to_speak(
        self, now: float, *, minimum_speech_seconds: float, quiet_seconds: float
    ) -> bool:
        """Return whether the caller may speak at `now`.

        That is once the bot has said at least `minimum_speech_seconds` since the mark and has then
        been quiet for `quiet_seconds`.
        """
        return (
            self._spoken_since_mark >= minimum_speech_seconds - TIME_TOLERANCE_SECONDS
            and self.quiet_seconds(now) >= quiet_seconds - TIME_TOLERANCE_SECONDS
        )

    def spans(self) -> list[Span]:
        """Return every span so far, the one still open included, up to its last active frame."""
        spans = list(self._closed)
        if self._open_start is not None and self._active_end is not None:
            spans.append(Span(start=self._open_start, end=self._active_end))
        return spans
