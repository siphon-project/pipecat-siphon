"""The harness's own two files: what a backend costs in time, and what counts as speech.

A scenario says what the caller does and what the bot must do about it, and those are the same on
every backend. Everything that differs between running the vendors over the internet and running a
model on a GPU in this room is timing: a cloud recognizer answers in a third of a second and a local
one announces a whole second, so a reply timeout that is right for one is either flaky or slow for
the other. Keeping that in `backends/*.yaml` is what lets the four scenarios be literally the same
four everywhere.

`harness.yaml` holds what is true of the harness itself rather than of any one backend: the
addresses on its isolated network, and the levels that separate the bot's speech from the comfort
noise the engine fills an idle leg with. Those thresholds are the ones a call's whole timing rests
on, so they are checked here rather than trusted: a `loud_rms` under `quiet_rms` would never open a
span, so the caller would wait for a reply that, as far as it could tell, never came.

Both files are read through the strict reader in `yaml_fields`, so a typo names the file and the key
instead of silently selecting a default and testing something else.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from tools.conversation.downlink_activity import ActivityThresholds
from tools.conversation.expectations import SignalRules
from tools.conversation.scenario import FRAME_MILLISECONDS
from tools.conversation.script_planner import CallTiming
from tools.conversation.yaml_fields import ConfigError, Fields


@dataclass(frozen=True)
class Backend:
    """One place the models can run, and what a call there costs in time."""

    name: str
    """The backend's own name, which is also the column it reports under."""
    description: str
    """What this backend is, for the report. Empty when the file says nothing."""
    compose_files: tuple[str, ...]
    """The compose files to bring up, in order, base first. At least one."""
    environment: Mapping[str, str]
    """What the bot container is given, including any pinned sampling folded in."""
    timing: CallTiming
    """How long the caller waits, and for what, before each line."""
    outcome_timeout_seconds: float
    """How long to wait after the last line for the BYE or REFER the scenario expects.

    Read from the same `timing:` block as the rest, because it is the same kind of number and
    belongs next to them, but it is the caller's wait *after* the script rather than during it, so
    `CallTiming` (which the planner consumes per line) has no field for it.
    """
    warm_up_calls: int
    """Calls to place and discard before the run counts, so a cold model is not scenario one."""


@dataclass(frozen=True)
class NetworkAddresses:
    """Where each container sits on the harness network, and where the caller's media lands."""

    caller_address: str
    proxy_address: str
    engine_address: str
    caller_media_port: int
    engine_metrics_url: str


@dataclass(frozen=True)
class HarnessConfig:
    """What is true of the harness itself, on any backend."""

    network: NetworkAddresses
    pause_bounds_ms: tuple[int, int]
    """The shortest and longest pause a scenario may put between two segments."""
    activity: ActivityThresholds
    """What separates the bot's speech from the line it is heard over."""
    signal: SignalRules
    """What "the goodbye played before the hangup" means, in seconds."""
    caller_voice: str
    """The synthesized voice the caller speaks with. Not the bot's."""
    transfer_target: str
    """Where a transfer is expected to be sent, matched against the REFER."""


def load_backend(path: Path) -> Backend:
    """Read and check the backend file at `path`.

    Raises:
        ConfigError: For anything missing, unknown, of the wrong shape or inconsistent.

    """
    with path.open(encoding="utf-8") as handle:
        document: object = yaml.safe_load(handle)
    return parse_backend(document, source=path.name)


def parse_backend(document: object, *, source: str) -> Backend:
    """Check a backend document read from `source`."""
    root = Fields(document, source=source, path="")
    name = root.text("name")
    description = root.optional_text("description") or ""

    compose_files = root.texts("compose_files")
    if not compose_files:
        # Nothing to bring up is not a backend. `texts` reads an absent key as empty, so the
        # required-ness has to be said here rather than left to the reader.
        raise root.error("at least one compose file is needed", "compose_files")

    environment = dict(root.text_mapping("environment"))

    timing_fields = root.mapping("timing")
    timing = CallTiming(
        greeting_timeout_seconds=timing_fields.positive_number("greeting_timeout_seconds"),
        reply_timeout_seconds=timing_fields.positive_number("reply_timeout_seconds"),
        quiet_before_speaking_seconds=timing_fields.positive_number(
            "quiet_before_speaking_seconds"
        ),
        minimum_bot_speech_seconds=timing_fields.positive_number("minimum_bot_speech_seconds"),
    )
    outcome_timeout_seconds = timing_fields.positive_number("outcome_timeout_seconds")
    timing_fields.finish()

    warm_up_calls = root.optional_integer("warm_up_calls") or 0
    if warm_up_calls < 0:
        raise root.error("must not be negative", "warm_up_calls")

    _fold_in_sampling(root, environment)
    root.finish()

    return Backend(
        name=name,
        description=description,
        compose_files=compose_files,
        environment=environment,
        timing=timing,
        outcome_timeout_seconds=outcome_timeout_seconds,
        warm_up_calls=warm_up_calls,
    )


def _fold_in_sampling(root: Fields, environment: dict[str, str]) -> None:
    """Turn a pinned `sampling:` block into the bot's own environment variables.

    Every call of a repeated run has to sample the same way or a difference between two calls says
    nothing. The bot reads its sampling from `BOT_LLM_TEMPERATURE` and `BOT_LLM_SEED`, so pinning it
    per backend means writing those, and a backend that pins nothing must leave them *absent*
    rather than empty: the bot treats unset as "the server's own default", and a temperature of 0
    is a real value that has to survive the round trip as `"0.0"`.
    """
    sampling_fields = root.optional_mapping("sampling")
    if sampling_fields is None:
        return
    temperature = sampling_fields.optional_number("temperature")
    seed = sampling_fields.optional_integer("seed")
    sampling_fields.finish()
    if temperature is not None:
        environment["BOT_LLM_TEMPERATURE"] = str(temperature)
    if seed is not None:
        environment["BOT_LLM_SEED"] = str(seed)


def load_harness(path: Path) -> HarnessConfig:
    """Read and check the harness file at `path`.

    Raises:
        ConfigError: For anything missing, unknown, of the wrong shape or inconsistent.

    """
    with path.open(encoding="utf-8") as handle:
        document: object = yaml.safe_load(handle)
    return parse_harness(document, source=path.name)


def parse_harness(document: object, *, source: str) -> HarnessConfig:
    """Check a harness document read from `source`."""
    root = Fields(document, source=source, path="")

    network_fields = root.mapping("network")
    network = NetworkAddresses(
        caller_address=network_fields.text("caller_address"),
        proxy_address=network_fields.text("proxy_address"),
        engine_address=network_fields.text("engine_address"),
        caller_media_port=network_fields.positive_integer("caller_media_port"),
        engine_metrics_url=network_fields.text("engine_metrics_url"),
    )
    network_fields.finish()

    pause_bounds_ms = _checked_pause_bounds(root)

    activity_fields = root.mapping("activity")
    activity = ActivityThresholds(
        loud_rms=activity_fields.positive_number("loud_rms"),
        quiet_rms=activity_fields.positive_number("quiet_rms"),
        frames_to_open=activity_fields.positive_integer("frames_to_open"),
        seconds_to_close=activity_fields.positive_number("seconds_to_close"),
    )
    activity_fields.finish()
    if activity.loud_rms < activity.quiet_rms:
        # Below `quiet_rms` is background and a run at or above `loud_rms` is speech. Inverted, no
        # frame could ever open a span, so the caller would wait out every reply it was given.
        raise activity_fields.error(
            f"must be at least quiet_rms ({activity.quiet_rms:g})", "loud_rms"
        )

    signal_fields = root.mapping("signal")
    signal = SignalRules(
        minimum_farewell_seconds=signal_fields.positive_number("minimum_farewell_seconds"),
        silent_before_signal_seconds=signal_fields.positive_number("silent_before_signal_seconds"),
        maximum_signal_delay_seconds=signal_fields.positive_number("maximum_signal_delay_seconds"),
    )
    signal_fields.finish()

    caller_voice = root.text("caller_voice")
    transfer_target = root.text("transfer_target")
    root.finish()

    return HarnessConfig(
        network=network,
        pause_bounds_ms=pause_bounds_ms,
        activity=activity,
        signal=signal,
        caller_voice=caller_voice,
        transfer_target=transfer_target,
    )


def _checked_pause_bounds(root: Fields) -> tuple[int, int]:
    """Read the pause range a scenario's pauses must fall inside.

    A pause is sent as whole 20 ms frames, so a bound off that grid is a bound no scenario can
    actually meet -- it would refuse a 600 ms pause for being under a 610 ms floor that nothing can
    reach either.
    """
    shortest, longest = root.integer_pair("pause_bounds_ms")
    if shortest <= 0:
        raise root.error("must be greater than 0", "pause_bounds_ms")
    if shortest > longest:
        raise root.error(
            f"runs backwards: {shortest} ms is longer than {longest} ms", "pause_bounds_ms"
        )
    off_grid = [bound for bound in (shortest, longest) if bound % FRAME_MILLISECONDS]
    if off_grid:
        raise root.error(
            f"must be multiples of {FRAME_MILLISECONDS} ms, the caller's frame: "
            f"{', '.join(f'{bound} ms' for bound in off_grid)}",
            "pause_bounds_ms",
        )
    return shortest, longest


__all__ = [
    "Backend",
    "ConfigError",
    "HarnessConfig",
    "NetworkAddresses",
    "load_backend",
    "load_harness",
    "parse_backend",
    "parse_harness",
]
