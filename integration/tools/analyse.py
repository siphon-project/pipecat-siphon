"""Assert on what actually happened: the capture, the bot's trace and the engine's counters.

Three oracles, none of which can see what the others see:

* **the capture**, read back by tshark, which is what the caller really received;
* **the bot's trace**, indexed by uplink frame, which is the engine's turn detection as the
  serializer delivered it;
* **the engine's Prometheus counters**, which are the only place a leaked session or a command
  the engine rejected shows up.

Run one scenario at a time::

    python3 -m tools.analyse roundtrip  --artifacts /harness/artifacts
    python3 -m tools.analyse turntaking --artifacts /harness/artifacts
    python3 -m tools.analyse wideband   --artifacts /harness/artifacts

Every check prints its measured value whether it passes or fails, and the whole measurement set
is written to ``<scenario>-analysis.json``. A red run that only says "assertion failed" costs
more time than it saves.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools import signals
from tools.capture import RtpPacket, TsharkError, read_rtp

CALLER_MEDIA_PORT = 6000
"""The port SIPp binds its media socket to (``-mp``). Packets arriving here are the downlink:
what the caller actually got back."""

PCMA_PAYLOAD_TYPE = 8

DOWNLINK_WARMUP_SECONDS = 0.5
"""Skipped at the head of the downlink before the spectrum is measured. The bridge has a jitter
buffer to prime and the first frames the bot echoes are the caller's own lead-in, so the opening
of the stream is not representative of the steady state the check is about."""

DOWNLINK_TAIL_SECONDS = 0.1
"""Skipped at the tail, where the call is being torn down under the last few packets."""

MINIMUM_TONE_RATIO = 20.0
"""How far above the empty control bin a tone has to sit in the returning audio to count as
present. The fixture itself clears 20 000x through a bare codec round trip (``make_fixtures
--self-test``); relaying it through a jitter buffer, a bot and a re-encode costs some of that,
so the bar here is deliberately looser than the generator's. 20x is 13 dB -- far more than
concealment or quantisation noise can fake, far less than a real tone gives."""

SILENCE_RMS = 300.0
"""Below this a decoded frame counts as silence. The fixture's speech sits above 1700 and its
silence decodes to a handful of LSBs, so anywhere in between works; this is nearer the floor so
a partially-concealed frame does not read as speech."""

BARGE_IN_BUDGET_MS = 250.0
"""How long the caller may still hear the bot after starting to talk over it.

The engine flushes its own playout queue on the VAD edge (``ws_barge_in``) and drops the frame
it had already taken for that tick, so the engine's own contribution is under one ptime. The
rest of the budget is the round trip out to the bot and back: the serializer turns the turn
edge into an interruption, pipecat drops the audio it still holds, and a ``clear`` comes back.
Anything beyond a quarter of a second would be audible as the bot talking over the caller."""

BARGE_IN_SPEAKING_FLOOR_MS = 1000.0
"""How long the bot must have been continuously audible before the barge-in. Without this the
flush assertion would also pass on a bot that never said anything."""


@dataclass
class Report:
    """Accumulated measurements and failures for one scenario."""

    scenario: str
    measurements: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def record(self, name: str, value: Any) -> None:
        """Store a measurement and print it."""
        self.measurements[name] = value
        print(f"  {name}: {value}")

    def check(self, condition: bool, message: str) -> bool:
        """Record a failure when ``condition`` is false. Returns the condition."""
        if not condition:
            self.failures.append(message)
        return condition

    def check_between(self, name: str, value: float, low: float, high: float) -> None:
        """Record ``value`` and require ``low <= value <= high``."""
        self.record(name, round(value, 4) if isinstance(value, float) else value)
        self.check(low <= value <= high, f"{name}={value} is outside [{low}, {high}]")


def _load_trace(path: Path) -> list[dict[str, Any]]:
    """Read the bot's JSON Lines trace."""
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _first(records: list[dict[str, Any]], event: str) -> dict[str, Any] | None:
    """Return the first record with the given event name."""
    return next((record for record in records if record["event"] == event), None)


def read_metrics(url: str, timeout: float = 5.0) -> dict[str, float]:
    """Return the engine's Prometheus metrics as a name -> summed-value mapping."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8", "replace")
    values: dict[str, float] = {}
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(-?[\d.eE+]+)$", line)
        if match:
            values[match.group(1)] = values.get(match.group(1), 0.0) + float(match.group(2))
    return values


def check_engine_state(report: Report, metrics_url: str) -> None:
    """Assert the engine tore the call down and never rejected anything the proxy sent."""
    print("engine")
    try:
        metrics = read_metrics(metrics_url)
    except (urllib.error.URLError, OSError) as error:
        report.failures.append(f"could not read engine metrics from {metrics_url}: {error}")
        return

    sessions = metrics.get("siphon_rtp_sessions", -1.0)
    deletes = metrics.get("siphon_rtp_deletes_total", 0.0)
    errors = metrics.get("siphon_rtp_control_errors_total", 0.0)
    report.record("engine_sessions", sessions)
    report.record("engine_deletes_total", deletes)
    report.record("engine_control_errors_total", errors)

    report.check(
        sessions == 0.0,
        f"siphon_rtp_sessions={sessions:g} after the call: engine state was left behind",
    )
    report.check(
        deletes >= 1.0,
        f"siphon_rtp_deletes_total={deletes:g}: the call was never deleted",
    )
    # The single highest-value engine assertion. A flow can complete 200/ACK/BYE looking perfect
    # while the engine rejected a command the proxy sent and carried on regardless.
    report.check(
        errors == 0.0,
        f"siphon_rtp_control_errors_total={errors:g}: the engine rejected a command",
    )


def check_wire_format(
    report: Report, trace: list[dict[str, Any]], wire_sample_rate: int = signals.SAMPLE_RATE
) -> None:
    """Assert the bridge announced and used the format this scenario negotiated.

    ``wire_sample_rate`` is the rate the *WebSocket* carries, which is the leg's codec rate unless
    the profile asked for another one with ``ws_sample_rate``. Everything here is read out of the
    bot's trace, so it is what the serializer actually delivered rather than what the engine says
    it sent.
    """
    print("websocket")
    disconnected = _first(trace, "engine_disconnected")
    if not report.check(
        disconnected is not None, "the bot never saw the engine disconnect: the call did not end"
    ):
        return
    assert disconnected is not None

    report.record("stream_id", disconnected.get("stream_id"))
    report.record("call_id", disconnected.get("call_id"))
    for name, expected in (
        ("wire_sample_rate", wire_sample_rate),
        ("wire_ptime", signals.PTIME_MS),
        ("wire_encoding", "L16"),
        ("wire_channels", 1),
        ("direction", "duplex"),
        # Equal to the wire rate on purpose: with the two the same, the serializer builds no
        # resampler in either direction, so a frame that arrives late or short is the engine's
        # doing and not pipecat's SOXR buffering.
        ("pipeline_sample_rate", wire_sample_rate),
    ):
        actual = disconnected.get(name)
        report.record(name, actual)
        report.check(actual == expected, f"start announced {name}={actual!r}, wanted {expected!r}")

    report.check(
        bool(disconnected.get("stream_id")),
        "the start envelope carried no streamId, so the bridge never announced itself",
    )

    audio = [record for record in trace if record["event"] == "uplink_audio"]
    report.record("uplink_frames", len(audio))
    sizes = {record["bytes"] for record in audio}
    report.record("uplink_frame_bytes", sorted(sizes))
    expected_bytes = signals.frame_samples(wire_sample_rate) * 2
    report.check(
        sizes in ({expected_bytes}, set()),
        f"uplink frames were {sorted(sizes)} bytes, wanted only {expected_bytes}",
    )
    # The frame geometry above is a byte count; this is the rate the frames claim to be in. A
    # bridge that framed 20 ms at one rate and labelled it another would pass one and fail the
    # other.
    rates = {record["sample_rate"] for record in audio}
    report.record("uplink_frame_sample_rates", sorted(rates))
    report.check(
        rates in ({wire_sample_rate}, set()),
        f"uplink frames were labelled {sorted(rates)} Hz, wanted only {wire_sample_rate}",
    )


def split_rtp(report: Report, packets: list[RtpPacket]) -> tuple[list[RtpPacket], list[RtpPacket]]:
    """Split the capture into what SIPp sent and what came back to it."""
    uplink = [packet for packet in packets if packet.source_port == CALLER_MEDIA_PORT]
    downlink = [packet for packet in packets if packet.destination_port == CALLER_MEDIA_PORT]
    report.record("rtp_packets_total", len(packets))
    report.record("rtp_uplink_packets", len(uplink))
    report.record("rtp_downlink_packets", len(downlink))
    return uplink, downlink


def check_downlink_stream(report: Report, downlink: list[RtpPacket], expected: int) -> None:
    """Assert the returning stream is one well-formed RTP stream of the right length."""
    if not report.check(bool(downlink), "no RTP came back to the caller at all"):
        return

    payload_types = {packet.payload_type for packet in downlink}
    ssrcs = {packet.ssrc for packet in downlink}
    report.record("downlink_payload_types", sorted(payload_types))
    report.record("downlink_ssrc_count", len(ssrcs))
    report.check(
        payload_types == {PCMA_PAYLOAD_TYPE},
        f"the downlink carried payload types {sorted(payload_types)}, wanted only "
        f"{PCMA_PAYLOAD_TYPE} (the codec the answer negotiated)",
    )
    report.check(len(ssrcs) == 1, f"the downlink used {len(ssrcs)} SSRCs, wanted exactly one")

    sequences = [packet.sequence for packet in downlink]
    gaps = sum(
        1
        for previous, current in itertools.pairwise(sequences)
        if (current - previous) % 65536 != 1
    )
    report.record("downlink_sequence_gaps", gaps)
    report.check(gaps == 0, f"{gaps} gaps in the downlink RTP sequence numbers")

    span = downlink[-1].time - downlink[0].time
    report.record("downlink_span_seconds", round(span, 3))
    # One packet per ptime is the contract. Allow a packet either side of the arithmetic.
    packets_per_second = 1000.0 / signals.PTIME_MS
    report.check_between(
        "downlink_packets_per_second",
        len(downlink) / span if span else 0.0,
        packets_per_second - 2,
        packets_per_second + 2,
    )
    report.check_between("downlink_packets", len(downlink), expected * 0.75, expected * 1.15)


def decoded_audio(packets: list[RtpPacket]) -> bytes:
    """Decode a run of A-law RTP payloads into one linear PCM buffer."""
    return signals.alaw_to_pcm(b"".join(packet.payload for packet in packets))


def analyse_echo_scenario(
    report: Report,
    artifacts: Path,
    metrics_url: str,
    scenario: str,
    wire_sample_rate: int = signals.SAMPLE_RATE,
) -> None:
    """Assert the caller's tones came back, through the bot, at the pitch they went out at.

    Shared by scenario 1 and scenario 3: the caller, the fixture and every assertion are the same,
    and the only variable is the rate the WebSocket carries. That is deliberate -- two runs that
    differ in one negotiated number are comparable, two harnesses are not.
    """
    trace = _load_trace(artifacts / f"{scenario}-bot.jsonl")
    check_wire_format(report, trace, wire_sample_rate)

    print("capture")
    packets = read_rtp(str(artifacts / f"{scenario}-capture.pcap"))
    _, downlink = split_rtp(report, packets)
    check_downlink_stream(report, downlink, expected=signals.ROUNDTRIP_FRAMES)
    if not downlink:
        return

    start = downlink[0].time + DOWNLINK_WARMUP_SECONDS
    end = downlink[-1].time - DOWNLINK_TAIL_SECONDS
    window = [packet for packet in downlink if start <= packet.time <= end]
    report.record("spectrum_packets", len(window))
    if not report.check(
        len(window) >= 50, f"only {len(window)} packets in the measurement window, wanted 50+"
    ):
        return

    print("spectrum")
    pcm = decoded_audio(window)
    control = signals.goertzel_power(pcm, signals.CONTROL_HZ)
    report.record("control_bin_power", round(control, 3))

    bins = {f"caller_{int(frequency)}hz": frequency for frequency in signals.CALLER_TONES_HZ}
    bins[f"bot_marker_{int(signals.BOT_MARKER_HZ)}hz"] = signals.BOT_MARKER_HZ
    for name, frequency in bins.items():
        power = signals.goertzel_power(pcm, frequency)
        ratio = power / control if control else float("inf")
        report.record(f"{name}_power", round(power, 1))
        report.record(f"{name}_ratio", round(ratio, 1))
        report.check(
            ratio >= MINIMUM_TONE_RATIO,
            f"{name} is only {ratio:.1f}x the empty control bin, wanted "
            f"{MINIMUM_TONE_RATIO:.0f}x -- that frequency is not in the returning audio",
        )

    # The marker is the part that proves the audio went *through the bot*. The caller never
    # emitted it, and a single-leg answer has no second party and no relay path back, so there
    # is nowhere else in the topology it could have come from.

    if wire_sample_rate != signals.SAMPLE_RATE:
        # A downlink rendered at the wrong rate is the silent failure of a negotiated wire rate:
        # the samples are right, the audio is present, and it plays at the wrong speed and pitch.
        # The bot builds its marker at the wire rate, so if the engine encoded that PCM into the
        # 8 kHz codec sample-for-sample the tone would arrive an octave down, at
        # BOT_MARKER_HALF_RATE_HZ. Asserting the marker bin is full does not catch that on its
        # own; asserting the ghost bin is empty does.
        ghost = signals.goertzel_power(pcm, signals.BOT_MARKER_HALF_RATE_HZ)
        marker = signals.goertzel_power(pcm, signals.BOT_MARKER_HZ)
        ratio = marker / ghost if ghost else float("inf")
        ghost_name = f"bot_marker_half_rate_{int(signals.BOT_MARKER_HALF_RATE_HZ)}hz_power"
        report.record(ghost_name, round(ghost, 1))
        report.record("bot_marker_rate_discrimination", round(ratio, 1))
        report.check(
            ratio >= MINIMUM_TONE_RATIO,
            f"the marker is only {ratio:.1f}x its half-rate ghost at "
            f"{signals.BOT_MARKER_HALF_RATE_HZ:.0f} Hz, wanted {MINIMUM_TONE_RATIO:.0f}x -- the "
            "downlink is being rendered at the wrong rate",
        )

    check_engine_state(report, metrics_url)


def analyse_roundtrip(report: Report, artifacts: Path, metrics_url: str) -> None:
    """Scenario 1: the wire follows the codec, so nothing is resampled anywhere."""
    analyse_echo_scenario(report, artifacts, metrics_url, "roundtrip", signals.SAMPLE_RATE)


def analyse_wideband(report: Report, artifacts: Path, metrics_url: str) -> None:
    """Scenario 3: the same call, with the wire negotiated up to 16 kHz by ``ws_sample_rate``.

    The caller is an unchanged 8 kHz G.711 phone, so the engine resamples on both halves: leg to
    wire on the uplink, wire back to the leg's codec on the downlink. Everything scenario 1
    asserts still has to hold, plus the pitch check that says the downlink came back at the rate
    it was rendered at.
    """
    analyse_echo_scenario(report, artifacts, metrics_url, "wideband", signals.WIDEBAND_WIRE_RATE)


def _speech_window(trace: list[dict[str, Any]]) -> tuple[int, int] | None:
    """Return the first and last uplink frame indices whose energy reads as speech."""
    loud = [
        record["uplink_frame"]
        for record in trace
        if record["event"] == "uplink_audio" and record["rms"] >= SILENCE_RMS
    ]
    if not loud:
        return None
    return loud[0], loud[-1]


def analyse_turntaking(report: Report, artifacts: Path, metrics_url: str) -> None:
    """Scenario 2: the turn edges landed where the fixture put them, and barge-in flushed."""
    trace = _load_trace(artifacts / "turntaking-bot.jsonl")
    check_wire_format(report, trace)

    print("turn edges")
    window = _speech_window(trace)
    if not report.check(
        window is not None, "no frame in the uplink read as speech: the fixture never arrived"
    ):
        return
    assert window is not None
    onset, offset = window
    report.record("fixture_speech_onset_frame", onset)
    report.record("fixture_speech_offset_frame", offset)
    report.check_between(
        "fixture_speech_frames",
        offset - onset + 1,
        signals.TURNTAKING_SPEECH_FRAMES - 3,
        signals.TURNTAKING_SPEECH_FRAMES + 3,
    )

    started = _first(trace, "speech_started")
    stopped = _first(trace, "speech_stopped")
    if not report.check(started is not None, "the engine never reported speech_started"):
        return
    if not report.check(stopped is not None, "the engine never reported speech_stopped"):
        return
    assert started is not None
    assert stopped is not None

    # The engine's VAD has no onset gate: one frame at or above the threshold flips the state
    # and the turn signal goes out on that same tick, ahead of the tick's audio. So the edge
    # should land on the very frame the fixture's speech starts on.
    started_delta = started["uplink_frame"] - onset
    report.check_between("speech_started_delta_frames", started_delta, 0, 2)
    report.record("speech_started_delta_ms", started_delta * signals.PTIME_MS)

    # The hangover holds the turn open for `hangover_ms / ptime` frames after the last loud one,
    # and the frame after that is the first to read as silence -- which is the tick the engine
    # reports the turn ended on.
    hangover_frames = HANGOVER_MS // signals.PTIME_MS
    expected_stop = offset + hangover_frames + 1
    stopped_delta = stopped["uplink_frame"] - expected_stop
    report.record("speech_stopped_frame", stopped["uplink_frame"])
    report.record("expected_speech_stopped_frame", expected_stop)
    report.check_between("speech_stopped_error_frames", stopped_delta, -2, 3)
    report.record(
        "speech_stopped_after_offset_ms",
        (stopped["uplink_frame"] - offset) * signals.PTIME_MS,
    )

    # The turn edge has to travel the whole way, not just reach the engine's own barge-in: the
    # serializer turns it into pipecat's interruption, the interruption reaches the bot and the
    # output transport, the serializer sends `clear`, and the engine answers with a mark named
    # "cleared" -- which is the `mark` record here. Requiring that round trip is what separates
    # "the engine flushed locally" from "barge-in works end to end".
    print("interruption chain")
    interrupted = _first(trace, "bot_stopped_generating")
    interruption = _first(trace, "interruption")
    mark = _first(trace, "mark")
    report.check(
        interrupted is not None,
        "the bot was never interrupted: the turn edge did not reach the pipeline",
    )
    report.check(
        interruption is not None,
        "no interruption reached the output transport, so no clear was ever sent",
    )
    if not report.check(
        mark is not None,
        "the engine never answered with a mark, so the clear never arrived",
    ):
        return
    assert mark is not None
    report.record("clear_acknowledged_frame", mark["uplink_frame"])
    report.check_between(
        "clear_acknowledged_delta_frames", mark["uplink_frame"] - started["uplink_frame"], 0, 5
    )

    print("capture")
    packets = read_rtp(str(artifacts / "turntaking-capture.pcap"))
    uplink, downlink = split_rtp(report, packets)
    if not report.check(bool(uplink) and bool(downlink), "the capture is missing a direction"):
        return

    report.check(
        {packet.payload_type for packet in downlink} == {PCMA_PAYLOAD_TYPE},
        "the downlink carried a payload type the answer did not negotiate",
    )

    print("barge-in")
    caller_onset = next(
        (
            packet.time
            for packet in uplink
            if signals.rms(signals.alaw_to_pcm(packet.payload)) >= SILENCE_RMS
        ),
        None,
    )
    if not report.check(
        caller_onset is not None, "the caller's speech is not in the capture at all"
    ):
        return
    assert caller_onset is not None
    report.record("caller_onset_seconds", round(caller_onset, 3))

    loud_downlink = [
        packet
        for packet in downlink
        if signals.rms(signals.alaw_to_pcm(packet.payload)) >= SILENCE_RMS
    ]
    if not report.check(bool(loud_downlink), "the bot was never audible in the capture"):
        return

    speaking_ms = (loud_downlink[-1].time - loud_downlink[0].time) * 1000.0
    report.record("bot_audible_ms_before_barge_in", round(speaking_ms, 1))
    report.check(
        speaking_ms >= BARGE_IN_SPEAKING_FLOOR_MS,
        f"the bot was only audible for {speaking_ms:.0f} ms, so the flush assertion would pass "
        "on a bot that barely spoke",
    )

    last_loud = loud_downlink[-1].time
    flush_ms = (last_loud - caller_onset) * 1000.0
    report.record("barge_in_flush_ms", round(flush_ms, 1))
    report.check(
        flush_ms <= BARGE_IN_BUDGET_MS,
        f"the caller still heard the bot {flush_ms:.0f} ms after starting to talk, budget is "
        f"{BARGE_IN_BUDGET_MS:.0f} ms -- the queued downlink was played out instead of flushed",
    )

    # And it has to *stay* flushed for the rest of the call. Stated as "nothing audible after the
    # budget expires" rather than "quiet after the last loud packet", which would be true by
    # construction and therefore prove nothing.
    deadline = caller_onset + BARGE_IN_BUDGET_MS / 1000.0
    late = [packet for packet in loud_downlink if packet.time > deadline]
    call_end = uplink[-1].time
    observation_ms = (call_end - last_loud) * 1000.0
    report.record("audible_downlink_after_budget", len(late))
    report.record("quiet_observation_ms", round(observation_ms, 1))
    report.record("downlink_packets_after_flush", sum(1 for p in downlink if p.time > last_loud))
    report.check(
        observation_ms >= 1000.0,
        f"only {observation_ms:.0f} ms of call left after the flush; too short to show the "
        "downlink stayed quiet rather than briefly dipping",
    )
    report.check(
        not late,
        f"{len(late)} audible downlink packets after the barge-in budget: the bot resumed, or "
        "the flush only dropped part of what was queued",
    )

    # Twenty seconds of speech went into the pipeline and under two came out as RTP. The
    # difference is what the barge-in threw away -- and the reason the flush assertion above is
    # measuring a flush rather than a bot that simply ran out of things to say.
    if interrupted is not None:
        generated = int(interrupted["generated"])
        undelivered = generated - len(downlink)
        report.record("bot_frames_generated", generated)
        report.record("bot_frames_undelivered", undelivered)
        report.check(
            undelivered >= 250,
            f"the bot only had {undelivered} frames left undelivered when it was cut off "
            "(5 s minimum), so there was no meaningful backlog for the flush to discard",
        )

    check_engine_state(report, metrics_url)


HANGOVER_MS = 300
"""Mirrors ``ws_vad_hangover_ms`` in the proxy's media profile. Duplicated on purpose: reading
the expected value out of the configuration under test would make the assertion vacuous."""


def main() -> int:
    """Analyse one scenario and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=("roundtrip", "turntaking", "wideband"))
    parser.add_argument("--artifacts", required=True, help="directory the run wrote into")
    parser.add_argument("--metrics-url", default="http://172.28.7.10:9091/metrics")
    arguments = parser.parse_args()

    artifacts = Path(arguments.artifacts)
    report = Report(scenario=arguments.scenario)
    print(f"== {arguments.scenario} ==")
    try:
        if arguments.scenario == "roundtrip":
            analyse_roundtrip(report, artifacts, arguments.metrics_url)
        elif arguments.scenario == "wideband":
            analyse_wideband(report, artifacts, arguments.metrics_url)
        else:
            analyse_turntaking(report, artifacts, arguments.metrics_url)
    except (TsharkError, FileNotFoundError, json.JSONDecodeError) as error:
        report.failures.append(f"{type(error).__name__}: {error}")

    output = {
        "scenario": report.scenario,
        "pass": not report.failures,
        "failures": report.failures,
        "measurements": report.measurements,
    }
    (artifacts / f"{arguments.scenario}-analysis.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )

    if report.failures:
        print()
        for failure in report.failures:
            print(f"FAIL: {failure}")
        print(f"\n{arguments.scenario}: FAILED ({len(report.failures)} check(s))")
        return 1
    print(f"\n{arguments.scenario}: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
