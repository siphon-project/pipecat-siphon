"""Caller audio shaped like a telephone line: band-limited, levelled, without long pauses."""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence

from tools.conversation.telephone_band import (
    compress_pauses,
    high_pass,
    low_pass,
    normalize_speech,
    pad_to_frames,
    to_pcm16,
    trim_silence,
)

RATE = 8000


def _sine(frequency: float, seconds: float, amplitude: float = 0.5) -> list[float]:
    count = round(seconds * RATE)
    return [amplitude * math.sin(2 * math.pi * frequency * index / RATE) for index in range(count)]


def _rms(samples: Sequence[float]) -> float:
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def _steady(samples: Sequence[float]) -> Sequence[float]:
    """Skip the filter's first 100 ms, where it is still settling."""
    return samples[RATE // 10 :]


class TestBand:
    def test_the_high_pass_takes_out_rumble_and_keeps_speech(self) -> None:
        rumble = _sine(100, 1.0)
        speech = _sine(1000, 1.0)

        assert _rms(_steady(high_pass(rumble, 300, RATE))) < 0.2 * _rms(_steady(rumble))
        assert _rms(_steady(high_pass(speech, 300, RATE))) > 0.9 * _rms(_steady(speech))

    def test_the_low_pass_takes_out_what_the_line_cannot_carry(self) -> None:
        hiss = _sine(3900, 1.0)
        speech = _sine(1000, 1.0)

        assert _rms(_steady(low_pass(hiss, 3400, RATE))) < 0.5 * _rms(_steady(hiss))
        assert _rms(_steady(low_pass(speech, 3400, RATE))) > 0.9 * _rms(_steady(speech))


class TestShaping:
    def test_silence_at_either_end_is_trimmed_to_a_short_margin(self) -> None:
        samples = [0.0] * 4000 + _sine(1000, 0.5) + [0.0] * 4000

        trimmed = trim_silence(samples, sample_rate=RATE)

        # 30 ms of margin kept on each side, give or take the 10 ms analysis window.
        assert abs(len(trimmed) - (4000 + 2 * 240)) <= 80

    def test_a_long_pause_inside_speech_is_shortened(self) -> None:
        """A synthesis pause at a comma would split one scripted line into two caller turns."""
        samples = _sine(1000, 0.5) + [0.0] * 3200 + _sine(1000, 0.5)

        compressed = compress_pauses(samples, sample_rate=RATE)

        assert abs(len(compressed) - (8000 + 960)) <= 80

    def test_a_short_pause_is_left_alone(self) -> None:
        samples = _sine(1000, 0.5) + [0.0] * 1200 + _sine(1000, 0.5)

        assert len(compress_pauses(samples, sample_rate=RATE)) == len(samples)

    def test_speech_is_levelled_to_the_target(self) -> None:
        levelled = normalize_speech(
            _sine(1000, 1.0, amplitude=0.05), target_dbov=-20.0, peak_dbov=-1.0, sample_rate=RATE
        )

        assert abs(20 * math.log10(_rms(levelled)) - (-20.0)) < 0.5
        assert max(abs(sample) for sample in levelled) <= 10 ** (-1.0 / 20) + 1e-9

    def test_a_level_that_would_clip_is_held_at_the_peak_ceiling(self) -> None:
        levelled = normalize_speech(
            _sine(1000, 1.0, amplitude=0.1), target_dbov=-2.0, peak_dbov=-1.0, sample_rate=RATE
        )

        assert max(abs(sample) for sample in levelled) <= 10 ** (-1.0 / 20) + 1e-9


class TestEncoding:
    def test_samples_clip_and_pack_as_little_endian_16_bit(self) -> None:
        assert to_pcm16([0.0, 1.5, -1.5, 0.5]) == struct.pack("<4h", 0, 32767, -32768, 16384)

    def test_audio_is_padded_to_whole_frames(self) -> None:
        padded = pad_to_frames(b"\x01\x00" * 170, frame_samples=160)

        assert len(padded) == 320 * 2
        assert padded.endswith(b"\x00\x00" * 150)
