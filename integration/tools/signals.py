"""Deterministic audio fixtures and the spectral tools that assert on them.

Everything here is a closed-form function of a sample index. There is no RNG, no seed and no
wall clock, so the same fixture bytes come out on every machine and every run, and the analyser
can recompute the reference signal instead of shipping a golden blob.

The one dependency is :mod:`audioop` (``audioop-lts`` on Python 3.13+), which pipecat already
pulls in. G.711 encoding is *not* hand-written here on purpose: a codec written next to the test
that consumes it shares the test's mistakes.
"""

from __future__ import annotations

import audioop
import math
from dataclasses import dataclass

__all__ = [
    "BOT_MARKER_HZ",
    "BOT_SPEECH_HZ",
    "CALLER_TONES_HZ",
    "CONTROL_HZ",
    "FRAME_SAMPLES",
    "PTIME_MS",
    "SAMPLE_RATE",
    "SPEECH_HZ",
    "Fixture",
    "alaw_to_pcm",
    "bot_marker_frame",
    "bot_speech_frame",
    "goertzel_power",
    "pcm_to_alaw",
    "rms",
    "roundtrip_fixture",
    "silence_frame",
    "speech_frame",
    "tone_frame",
    "turntaking_fixture",
]

SAMPLE_RATE = 8000
"""Wire rate for both fixtures. G.711 is 8 kHz, and the bridge is configured to match it so no
resampler runs anywhere in the path (see the README's note on the SOXR buffering hazard)."""

PTIME_MS = 20
"""Packetization interval. One RTP packet, one WebSocket binary frame, one logical clock tick."""

FRAME_SAMPLES = SAMPLE_RATE // 1000 * PTIME_MS
"""160 samples per frame."""

SAMPLE_WIDTH = 2
"""Bytes per linear PCM sample."""

CALLER_TONES_HZ = (500.0, 1300.0)
"""The two tones the round-trip fixture carries. Neither is a DTMF row/column frequency, so the
engine's RFC 4733 detector has nothing to latch onto."""

CALLER_TONE_AMPLITUDE = 6000
"""Per-tone amplitude. Two of them sum to 12000 peak, comfortably inside int16."""

BOT_MARKER_HZ = 2400.0
"""The tone the round-trip bot mixes into its echo. It is absent from the caller's fixture, so
finding it in the audio that comes back proves the samples went *through the bot* rather than
around it somewhere inside the engine."""

BOT_MARKER_AMPLITUDE = 3000

CONTROL_HZ = 3300.0
"""A bin the fixture never excites. Comparing the tone bins against this one turns "there is
energy at 500 Hz" into "there is energy at 500 Hz *and nowhere else*", which is the difference
between measuring the fixture and measuring noise."""

SPEECH_HZ = 620.0
"""Fundamental of the turn-taking fixture's speech surrogate."""

SPEECH_AMPLITUDE = 7000

SPEECH_HARMONICS = (1.0, 0.6, 0.35, 0.2)
"""Relative amplitudes of the fundamental and its first three harmonics. A voiced vowel is a
harmonic stack, not a sine, and an energy VAD is not the only thing that might sit on this path."""

SPEECH_SYLLABLE_HZ = 5.0
"""Amplitude-modulation rate, roughly a syllable per 200 ms."""

SPEECH_MODULATION_DEPTH = 0.4
"""Kept below 0.5 so the envelope never approaches zero: a dip that crossed the VAD's silence
threshold would split one turn into two and the edge assertions would be measuring the wrong
thing."""

BOT_SPEECH_HZ = 1000.0
"""The tone the turn-taking bot plays as its "voice". Distinct from every caller component, so
the analyser can tell bot audio from a caller echo in the returning RTP."""

BOT_SPEECH_AMPLITUDE = 8000


@dataclass(frozen=True, slots=True)
class Fixture:
    """One generated caller-side audio fixture.

    Attributes:
        name: Fixture identifier, also the scenario name.
        frames: Linear 16-bit little-endian PCM, one entry per ``PTIME_MS``.
        speech_frames: Half-open ``(first, last_exclusive)`` frame indices of the speech burst,
            or ``None`` for a fixture with no speech/silence structure. These are the indices
            the VAD edge assertions are measured against.

    """

    name: str
    frames: tuple[bytes, ...]
    speech_frames: tuple[int, int] | None = None

    @property
    def duration_ms(self) -> int:
        """Return the fixture's total duration in milliseconds."""
        return len(self.frames) * PTIME_MS


def _pack(samples: list[int]) -> bytes:
    """Clamp to int16 and pack little-endian."""
    return b"".join(
        int(max(-32768, min(32767, value))).to_bytes(2, "little", signed=True) for value in samples
    )


def tone_frame(frame_index: int, tones_hz: tuple[float, ...], amplitude: int) -> bytes:
    """Return one frame of a sum of continuous tones.

    Phase is derived from the absolute sample index rather than carried in a state variable, so
    frames are independent and the waveform is continuous across frame boundaries.
    """
    start = frame_index * FRAME_SAMPLES
    samples = []
    for offset in range(FRAME_SAMPLES):
        index = start + offset
        value = 0.0
        for frequency in tones_hz:
            value += amplitude * math.sin(2.0 * math.pi * frequency * index / SAMPLE_RATE)
        samples.append(round(value))
    return _pack(samples)


def silence_frame() -> bytes:
    """Return one frame of digital silence."""
    return bytes(FRAME_SAMPLES * SAMPLE_WIDTH)


def speech_frame(frame_index: int, burst_start_frame: int) -> bytes:
    """Return one frame of the speech surrogate: a modulated harmonic stack.

    Args:
        frame_index: Absolute frame index in the fixture, which fixes the carrier phase.
        burst_start_frame: Frame the burst began on, which fixes the envelope phase so the burst
            always opens at the top of a syllable rather than wherever the carrier happens to be.

    """
    start = frame_index * FRAME_SAMPLES
    envelope_start = (frame_index - burst_start_frame) * FRAME_SAMPLES
    samples = []
    for offset in range(FRAME_SAMPLES):
        index = start + offset
        envelope = 1.0 - SPEECH_MODULATION_DEPTH * (
            0.5
            - 0.5
            * math.cos(2.0 * math.pi * SPEECH_SYLLABLE_HZ * (envelope_start + offset) / SAMPLE_RATE)
        )
        value = 0.0
        for harmonic, weight in enumerate(SPEECH_HARMONICS, start=1):
            value += weight * math.sin(2.0 * math.pi * SPEECH_HZ * harmonic * index / SAMPLE_RATE)
        samples.append(round(SPEECH_AMPLITUDE * envelope * value / sum(SPEECH_HARMONICS)))
    return _pack(samples)


def bot_marker_frame(frame_index: int) -> bytes:
    """Return one frame of the round-trip bot's marker tone."""
    return tone_frame(frame_index, (BOT_MARKER_HZ,), BOT_MARKER_AMPLITUDE)


def bot_speech_frame(frame_index: int) -> bytes:
    """Return one frame of the turn-taking bot's "voice"."""
    return tone_frame(frame_index, (BOT_SPEECH_HZ,), BOT_SPEECH_AMPLITUDE)


ROUNDTRIP_FRAMES = 200
"""4.0 s of continuous two-tone. Long enough that the spectral estimate has thousands of cycles
of each tone to average over, short enough that a stuck run fails fast."""

TURNTAKING_LEAD_FRAMES = 100
"""2.0 s of caller silence at the head. The bot talks over this, so there is a downlink in
flight to flush by the time the caller barges in."""

TURNTAKING_SPEECH_FRAMES = 75
"""1.5 s of caller speech: the barge-in."""

TURNTAKING_TAIL_FRAMES = 75
"""1.5 s of caller silence after the burst, so ``speech_stopped`` has room to land past the
hangover and still be inside the call."""


def roundtrip_fixture() -> Fixture:
    """Return the scenario 1 fixture: a continuous two-tone the bot will echo back."""
    frames = tuple(
        tone_frame(index, CALLER_TONES_HZ, CALLER_TONE_AMPLITUDE)
        for index in range(ROUNDTRIP_FRAMES)
    )
    return Fixture(name="roundtrip", frames=frames)


def turntaking_fixture() -> Fixture:
    """Return the scenario 2 fixture: silence, one speech burst, silence."""
    burst_start = TURNTAKING_LEAD_FRAMES
    burst_end = burst_start + TURNTAKING_SPEECH_FRAMES
    total = burst_end + TURNTAKING_TAIL_FRAMES
    frames = []
    for index in range(total):
        if burst_start <= index < burst_end:
            frames.append(speech_frame(index, burst_start))
        else:
            frames.append(silence_frame())
    return Fixture(name="turntaking", frames=tuple(frames), speech_frames=(burst_start, burst_end))


def pcm_to_alaw(pcm: bytes) -> bytes:
    """Encode little-endian 16-bit PCM to G.711 A-law."""
    return audioop.lin2alaw(pcm, SAMPLE_WIDTH)


def alaw_to_pcm(payload: bytes) -> bytes:
    """Decode G.711 A-law to little-endian 16-bit PCM."""
    return audioop.alaw2lin(payload, SAMPLE_WIDTH)


def rms(pcm: bytes) -> float:
    """Return the root-mean-square level of little-endian 16-bit PCM."""
    count = len(pcm) // SAMPLE_WIDTH
    if count == 0:
        return 0.0
    total = 0
    for index in range(0, count * SAMPLE_WIDTH, SAMPLE_WIDTH):
        sample = int.from_bytes(pcm[index : index + SAMPLE_WIDTH], "little", signed=True)
        total += sample * sample
    return math.sqrt(total / count)


def goertzel_power(pcm: bytes, frequency: float, sample_rate: int = SAMPLE_RATE) -> float:
    """Return the power at one frequency, normalised so it is comparable across buffer lengths.

    Goertzel is a single-bin DFT: one second-order recurrence over the samples, no FFT and no
    numpy. The frequency does not have to fall on a DFT bin centre, which is what lets the
    fixture pick round numbers instead of multiples of ``sample_rate / N``.
    """
    count = len(pcm) // SAMPLE_WIDTH
    if count == 0:
        return 0.0
    omega = 2.0 * math.pi * frequency / sample_rate
    coefficient = 2.0 * math.cos(omega)
    first = 0.0
    second = 0.0
    for index in range(0, count * SAMPLE_WIDTH, SAMPLE_WIDTH):
        sample = int.from_bytes(pcm[index : index + SAMPLE_WIDTH], "little", signed=True)
        current = sample + coefficient * first - second
        second = first
        first = current
    power = first * first + second * second - coefficient * first * second
    # Divide by N^2 so the result is an amplitude-squared estimate rather than growing with the
    # length of the buffer, which would make thresholds depend on how long the call ran.
    return power / (count * count)
