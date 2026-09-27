"""Shape synthesized caller speech into what a telephone line would carry.

A text-to-speech voice is wideband, evenly levelled and pauses at every comma. A caller on a phone
is none of those, and the difference matters to what is under test: the agent's recognizer and
turn logic run on an 8 kHz G.711 leg. So each caller line is band-limited to the 300-3400 Hz a
telephone channel passes, levelled like speech rather than like a studio, and has its internal
synthesis pauses shortened: left long, a pause at "please?" would end the engine's stretch of
speech and split one scripted line into two caller turns, and the only pause a scenario should
contain is the one it scripts.

Plain Python on purpose. This runs once, when the fixtures are generated, over a few seconds of
audio per line; a dependency for it would outweigh the work.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from typing import Literal

SILENCE_DBFS = -45.0
"""Below this, an analysis window is silence."""

WINDOW_SECONDS = 0.01
"""The analysis window for silence and level."""

_BUTTERWORTH_Q = 1 / math.sqrt(2)


def high_pass(samples: Sequence[float], cutoff_hz: float, sample_rate: int) -> list[float]:
    """Filter out what lies below `cutoff_hz` with a second-order Butterworth high-pass."""
    return _biquad(samples, _coefficients("high", cutoff_hz, sample_rate))


def low_pass(samples: Sequence[float], cutoff_hz: float, sample_rate: int) -> list[float]:
    """Filter out what lies above `cutoff_hz` with a second-order Butterworth low-pass."""
    return _biquad(samples, _coefficients("low", cutoff_hz, sample_rate))


def trim_silence(
    samples: Sequence[float],
    *,
    sample_rate: int,
    threshold_dbfs: float = SILENCE_DBFS,
    window_seconds: float = WINDOW_SECONDS,
    keep_seconds: float = 0.03,
) -> list[float]:
    """Cut the silence before the first and after the last speech, keeping `keep_seconds`."""
    window = _window_samples(window_seconds, sample_rate)
    flags = _active_windows(samples, threshold_dbfs, window)
    if not any(flags):
        return []
    first = flags.index(True)
    last = len(flags) - 1 - flags[::-1].index(True)
    keep = round(keep_seconds * sample_rate)
    start = max(0, first * window - keep)
    end = min(len(samples), (last + 1) * window + keep)
    return list(samples[start:end])


def compress_pauses(
    samples: Sequence[float],
    *,
    sample_rate: int,
    longer_than_seconds: float = 0.18,
    to_seconds: float = 0.12,
    threshold_dbfs: float = SILENCE_DBFS,
    window_seconds: float = WINDOW_SECONDS,
) -> list[float]:
    """Shorten every silence between two stretches of speech that is longer than a word gap.

    The start and the end of the pause are kept, so the speech either side decays and rises the
    way it did; only the middle is taken out.
    """
    window = _window_samples(window_seconds, sample_rate)
    flags = _active_windows(samples, threshold_dbfs, window)
    longer_than = round(longer_than_seconds * sample_rate)
    keep = round(to_seconds * sample_rate)

    shaped: list[float] = []
    index = 0
    speech_seen = False
    while index < len(flags):
        if flags[index]:
            shaped.extend(samples[index * window : (index + 1) * window])
            speech_seen = True
            index += 1
            continue
        run_end = index
        while run_end < len(flags) and not flags[run_end]:
            run_end += 1
        start = index * window
        end = min(len(samples), run_end * window)
        between_speech = speech_seen and run_end < len(flags)
        if between_speech and end - start > longer_than:
            head = keep // 2
            shaped.extend(samples[start : start + head])
            shaped.extend(samples[end - (keep - head) : end])
        else:
            shaped.extend(samples[start:end])
        index = run_end
    return shaped


def normalize_speech(
    samples: Sequence[float],
    *,
    sample_rate: int,
    target_dbov: float = -20.0,
    peak_dbov: float = -1.0,
    threshold_dbfs: float = SILENCE_DBFS,
    window_seconds: float = WINDOW_SECONDS,
) -> list[float]:
    """Scale so the speech, not the silence around it, sits at `target_dbov`.

    The gain is capped so no sample goes past `peak_dbov`: a clipped line would put distortion into
    the recognizer that no real caller's phone would.
    """
    window = _window_samples(window_seconds, sample_rate)
    flags = _active_windows(samples, threshold_dbfs, window)
    active = [
        sample
        for index, flag in enumerate(flags)
        if flag
        for sample in samples[index * window : (index + 1) * window]
    ]
    if not active:
        return list(samples)
    level = math.sqrt(sum(sample * sample for sample in active) / len(active))
    gain = 10 ** (target_dbov / 20) / level
    peak = max(abs(sample) for sample in samples)
    ceiling = 10 ** (peak_dbov / 20)
    if peak * gain > ceiling:
        gain = ceiling / peak
    return [sample * gain for sample in samples]


def to_pcm16(samples: Sequence[float]) -> bytes:
    """Encode samples in [-1, 1] as little-endian 16-bit PCM, clipping anything outside."""
    values = [max(-32768, min(32767, round(sample * 32768))) for sample in samples]
    return struct.pack(f"<{len(values)}h", *values)


def pad_to_frames(pcm: bytes, *, frame_samples: int) -> bytes:
    """Pad 16-bit PCM with silence to a whole number of `frame_samples`-sample frames.

    Raises:
        ValueError: When `pcm` is not whole 16-bit samples.

    """
    if len(pcm) % 2:
        raise ValueError(f"{len(pcm)} bytes is not whole 16-bit samples")
    remainder = (len(pcm) // 2) % frame_samples
    if not remainder:
        return pcm
    return pcm + b"\x00\x00" * (frame_samples - remainder)


def _coefficients(
    kind: Literal["low", "high"], cutoff_hz: float, sample_rate: int
) -> tuple[float, float, float, float, float]:
    """RBJ audio-EQ-cookbook biquad, normalised by a0."""
    omega = 2 * math.pi * cutoff_hz / sample_rate
    alpha = math.sin(omega) / (2 * _BUTTERWORTH_Q)
    cos_omega = math.cos(omega)
    if kind == "low":
        b0 = (1 - cos_omega) / 2
        b1 = 1 - cos_omega
    else:
        b0 = (1 + cos_omega) / 2
        b1 = -(1 + cos_omega)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * cos_omega
    a2 = 1 - alpha
    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


def _biquad(
    samples: Sequence[float], coefficients: tuple[float, float, float, float, float]
) -> list[float]:
    b0, b1, b2, a1, a2 = coefficients
    input_1 = input_2 = output_1 = output_2 = 0.0
    filtered: list[float] = []
    for sample in samples:
        output = b0 * sample + b1 * input_1 + b2 * input_2 - a1 * output_1 - a2 * output_2
        input_2, input_1 = input_1, sample
        output_2, output_1 = output_1, output
        filtered.append(output)
    return filtered


def _window_samples(window_seconds: float, sample_rate: int) -> int:
    return max(1, round(window_seconds * sample_rate))


def _active_windows(samples: Sequence[float], threshold_dbfs: float, window: int) -> list[bool]:
    threshold = 10 ** (threshold_dbfs / 20)
    flags: list[bool] = []
    for start in range(0, len(samples), window):
        chunk = samples[start : start + window]
        level = math.sqrt(sum(sample * sample for sample in chunk) / len(chunk))
        flags.append(level >= threshold)
    return flags
