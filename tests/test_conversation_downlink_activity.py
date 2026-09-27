"""Telling the bot's speech from the line's comfort noise, one 20 ms frame at a time."""

from __future__ import annotations

import pytest

from tools.conversation.downlink_activity import ActivityThresholds, DownlinkActivity

FRAME = 0.02
THRESHOLDS = ActivityThresholds(
    loud_rms=400.0, quiet_rms=250.0, frames_to_open=3, seconds_to_close=0.3
)
SPEECH = 3000.0
# What the engine sends while nobody talks: noise at -75 dBov, about 6 on a 16-bit scale.
COMFORT_NOISE = 6.0
BETWEEN_THRESHOLDS = 300.0


def _feed(activity: DownlinkActivity, start: float, seconds: float, level: float) -> float:
    """Observe `seconds` of frames at `level` from `start`; return the time after the last one."""
    count = round(seconds / FRAME)
    for index in range(count):
        activity.observe(start + index * FRAME, level)
    return start + count * FRAME


def _edges(activity: DownlinkActivity) -> list[float]:
    """Flatten the spans to start, end, start, end, so they compare with `pytest.approx`."""
    return [edge for span in activity.spans() for edge in (span.start, span.end)]


class TestDownlinkActivity:
    def test_comfort_noise_is_not_speech(self) -> None:
        activity = DownlinkActivity(THRESHOLDS)
        end = _feed(activity, 0.0, 5.0, COMFORT_NOISE)

        assert activity.spans() == []
        assert not activity.ready_to_speak(end, minimum_speech_seconds=0.3, quiet_seconds=1.0)

    def test_a_click_is_not_speech(self) -> None:
        activity = DownlinkActivity(THRESHOLDS)
        time = _feed(activity, 0.0, 0.04, SPEECH)
        _feed(activity, time, 1.0, COMFORT_NOISE)

        assert activity.spans() == []

    def test_ready_exactly_when_the_quiet_window_has_passed(self) -> None:
        activity = DownlinkActivity(THRESHOLDS)
        time = _feed(activity, 0.0, 0.5, SPEECH)
        time = _feed(activity, time, 1.0, COMFORT_NOISE)

        assert activity.quiet_seconds(time) == pytest.approx(1.0)
        assert activity.ready_to_speak(time, minimum_speech_seconds=0.3, quiet_seconds=1.0)
        assert not activity.ready_to_speak(
            time - 0.01, minimum_speech_seconds=0.3, quiet_seconds=1.0
        )

    def test_speech_resuming_restarts_the_quiet_window(self) -> None:
        activity = DownlinkActivity(THRESHOLDS)
        time = _feed(activity, 0.0, 0.5, SPEECH)
        time = _feed(activity, time, 0.8, COMFORT_NOISE)
        time = _feed(activity, time, 0.4, SPEECH)
        time = _feed(activity, time, 0.8, COMFORT_NOISE)

        assert not activity.ready_to_speak(time, minimum_speech_seconds=0.3, quiet_seconds=1.0)

    def test_only_speech_after_the_mark_counts(self) -> None:
        """The reply to a line has to come after the line, not be the greeting before it."""
        activity = DownlinkActivity(THRESHOLDS)
        time = _feed(activity, 0.0, 0.5, SPEECH)
        activity.mark(time)
        time = _feed(activity, time, 1.5, COMFORT_NOISE)

        assert activity.spoken_seconds_since_mark() == 0.0
        assert not activity.ready_to_speak(time, minimum_speech_seconds=0.3, quiet_seconds=1.0)

        time = _feed(activity, time, 0.4, SPEECH)
        time = _feed(activity, time, 1.0, COMFORT_NOISE)

        assert activity.spoken_seconds_since_mark() == pytest.approx(0.4)
        assert activity.ready_to_speak(time, minimum_speech_seconds=0.3, quiet_seconds=1.0)

    def test_a_soft_tail_between_the_thresholds_is_still_the_bot(self) -> None:
        activity = DownlinkActivity(THRESHOLDS)
        time = _feed(activity, 0.0, 0.3, SPEECH)
        time = _feed(activity, time, 1.0, BETWEEN_THRESHOLDS)

        assert not activity.ready_to_speak(time, minimum_speech_seconds=0.2, quiet_seconds=1.0)

        time = _feed(activity, time, 1.0, COMFORT_NOISE)

        assert activity.ready_to_speak(time, minimum_speech_seconds=0.2, quiet_seconds=1.0)
        assert _edges(activity) == pytest.approx([0.0, 1.3])

    def test_a_dip_shorter_than_the_close_is_one_span(self) -> None:
        activity = DownlinkActivity(THRESHOLDS)
        time = _feed(activity, 0.0, 0.5, SPEECH)
        time = _feed(activity, time, 0.2, COMFORT_NOISE)
        time = _feed(activity, time, 0.5, SPEECH)
        time = _feed(activity, time, 1.0, COMFORT_NOISE)
        time = _feed(activity, time, 0.5, SPEECH)
        _feed(activity, time, 1.0, COMFORT_NOISE)

        assert _edges(activity) == pytest.approx([0.0, 1.2, 2.2, 2.7])

    def test_an_open_span_is_reported_up_to_its_last_frame(self) -> None:
        activity = DownlinkActivity(THRESHOLDS)
        _feed(activity, 0.0, 0.5, SPEECH)

        assert _edges(activity) == pytest.approx([0.0, 0.5])
