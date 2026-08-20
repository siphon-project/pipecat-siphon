"""Generate the caller-side audio fixtures the SIPp scenarios play into the call.

Run it with the directory to write into::

    python3 -m tools.make_fixtures /harness/artifacts

It writes one pcap per scenario plus a ``fixtures.json`` recording what went into them, so a
run's artifacts say what was played without anyone having to re-run the generator. The analyser
does not read it: it imports :mod:`tools.signals` and recomputes the reference itself.

``--self-test`` checks the generator against the analyser before either is trusted: it Goertzels
the fixture through a G.711 A-law encode/decode round trip and requires the tones to stand out
from the control bin. That catches a fixture that is silent, clipped, or aliased *before* a red
run blames the engine for it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools import signals
from tools.rtp_pcap import write_rtp_pcap

MINIMUM_TONE_MARGIN = 100.0
"""How far above the empty control bin a fixture tone has to sit in the self-test. 100x in power
is 20 dB; A-law's own quantisation noise at these levels is far below that."""


def _write(fixture: signals.Fixture, directory: Path) -> dict[str, object]:
    """Write one fixture's pcap and return its manifest entry."""
    path = directory / f"{fixture.name}.pcap"
    packets = write_rtp_pcap(
        str(path),
        (signals.pcm_to_alaw(frame) for frame in fixture.frames),
        ptime_ms=signals.PTIME_MS,
        samples_per_frame=signals.FRAME_SAMPLES,
    )
    return {
        "name": fixture.name,
        "pcap": path.name,
        "packets": packets,
        "ptime_ms": signals.PTIME_MS,
        "sample_rate": signals.SAMPLE_RATE,
        "duration_ms": fixture.duration_ms,
        "speech_frames": list(fixture.speech_frames) if fixture.speech_frames else None,
    }


def _self_test() -> int:
    """Prove the fixtures carry the signal the analyser will look for."""
    failures: list[str] = []

    roundtrip = signals.roundtrip_fixture()
    coded = signals.alaw_to_pcm(signals.pcm_to_alaw(b"".join(roundtrip.frames)))
    control = signals.goertzel_power(coded, signals.CONTROL_HZ)
    for frequency in signals.CALLER_TONES_HZ:
        power = signals.goertzel_power(coded, frequency)
        ratio = power / control if control else float("inf")
        print(f"roundtrip {frequency:.0f} Hz: power {power:.1f}, {ratio:.0f}x the control bin")
        if ratio < MINIMUM_TONE_MARGIN:
            failures.append(f"{frequency:.0f} Hz is only {ratio:.1f}x the control bin")

    marker = b"".join(signals.bot_marker_frame(index) for index in range(signals.ROUNDTRIP_FRAMES))
    marker_coded = signals.alaw_to_pcm(signals.pcm_to_alaw(marker))
    marker_power = signals.goertzel_power(marker_coded, signals.BOT_MARKER_HZ)
    marker_control = signals.goertzel_power(marker_coded, signals.CONTROL_HZ)
    marker_ratio = marker_power / marker_control if marker_control else float("inf")
    print(
        f"bot marker {signals.BOT_MARKER_HZ:.0f} Hz: power {marker_power:.1f}, "
        f"{marker_ratio:.0f}x the control bin"
    )
    if marker_ratio < MINIMUM_TONE_MARGIN:
        failures.append(f"bot marker is only {marker_ratio:.1f}x the control bin")

    # The caller's own fixture must be *quiet* where the bot's marker sits, or finding the marker
    # in the returning audio would prove nothing about the bot.
    leak = signals.goertzel_power(coded, signals.BOT_MARKER_HZ)
    leak_ratio = marker_power / leak if leak else float("inf")
    print(f"caller leakage into the marker bin: power {leak:.1f}, marker is {leak_ratio:.0f}x it")
    if leak_ratio < MINIMUM_TONE_MARGIN:
        failures.append(f"the caller fixture leaks into the marker bin ({leak_ratio:.1f}x)")

    turntaking = signals.turntaking_fixture()
    assert turntaking.speech_frames is not None
    burst_start, burst_end = turntaking.speech_frames
    silent = signals.rms(turntaking.frames[0])
    voiced = min(signals.rms(turntaking.frames[index]) for index in range(burst_start, burst_end))
    print(f"turntaking: silence rms {silent:.1f}, quietest speech frame rms {voiced:.1f}")
    if silent != 0.0:
        failures.append(f"the silent lead is not digital silence (rms {silent})")
    if voiced < 1000.0:
        failures.append(f"the quietest speech frame is only rms {voiced:.1f}")

    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        return 1
    print("PASS: fixtures carry the signal the analyser looks for")
    return 0


def main() -> int:
    """Write the fixtures, or run the self-test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", help="directory to write the fixtures into")
    parser.add_argument(
        "--self-test", action="store_true", help="check the fixtures spectrally and exit"
    )
    arguments = parser.parse_args()

    if arguments.self_test:
        return _self_test()

    if not arguments.directory:
        parser.error("a directory is required unless --self-test is given")
    directory = Path(arguments.directory)
    directory.mkdir(parents=True, exist_ok=True)

    manifest = [
        _write(signals.roundtrip_fixture(), directory),
        _write(signals.turntaking_fixture(), directory),
    ]
    (directory / "fixtures.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for entry in manifest:
        print(f"wrote {entry['pcap']}: {entry['packets']} packets, {entry['duration_ms']} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
