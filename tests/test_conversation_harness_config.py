"""The harness's own configuration: what a backend costs in time, and what counts as speech."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.conversation.harness_config import load_backend, load_harness
from tools.conversation.yaml_fields import ConfigError

BACKEND = """
name: cloud
description: The vendors, over the internet.
compose_files: ["compose.yaml", "compose.cloud.yaml"]
environment:
  BOT_BACKEND: cloud
  BOT_STT_LANGUAGE: multi
timing:
  greeting_timeout_seconds: 20
  reply_timeout_seconds: 25
  outcome_timeout_seconds: 30
  quiet_before_speaking_seconds: 0.8
  minimum_bot_speech_seconds: 0.3
warm_up_calls: 1
sampling:
  temperature: 0.0
  seed: 7
"""

HARNESS = """
network:
  caller_address: 172.28.8.40
  proxy_address: 172.28.8.20
  engine_address: 172.28.8.10
  caller_media_port: 7078
  engine_metrics_url: http://172.28.8.10:9091/metrics
pause_bounds_ms: [600, 1200]
activity:
  loud_rms: 300
  quiet_rms: 60
  frames_to_open: 3
  seconds_to_close: 0.3
signal:
  minimum_farewell_seconds: 0.4
  silent_before_signal_seconds: 0.2
  maximum_signal_delay_seconds: 3.0
caller_voice: am_michael
transfer_target: sip:manager@harness.example
"""


def _written(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestBackend:
    def test_a_backend_reads_into_what_a_run_needs(self, tmp_path: Path) -> None:
        backend = load_backend(_written(tmp_path, "cloud.yaml", BACKEND))

        assert backend.name == "cloud"
        assert backend.compose_files == ("compose.yaml", "compose.cloud.yaml")
        assert backend.environment["BOT_STT_LANGUAGE"] == "multi"
        assert backend.timing.greeting_timeout_seconds == 20.0
        assert backend.timing.quiet_before_speaking_seconds == 0.8
        assert backend.outcome_timeout_seconds == 30.0
        assert backend.warm_up_calls == 1

    def test_pinned_sampling_reaches_the_bot_as_its_own_environment(self, tmp_path: Path) -> None:
        """Every call of a repeated run has to sample the same way, so it is pinned per backend."""
        backend = load_backend(_written(tmp_path, "cloud.yaml", BACKEND))

        assert backend.environment["BOT_LLM_TEMPERATURE"] == "0.0"
        assert backend.environment["BOT_LLM_SEED"] == "7"

    def test_sampling_may_be_left_to_the_model_server(self, tmp_path: Path) -> None:
        text = BACKEND.replace("sampling:\n  temperature: 0.0\n  seed: 7\n", "")

        backend = load_backend(_written(tmp_path, "cloud.yaml", text))

        assert "BOT_LLM_TEMPERATURE" not in backend.environment
        assert "BOT_LLM_SEED" not in backend.environment

    def test_a_key_nothing_reads_names_the_file(self, tmp_path: Path) -> None:
        text = BACKEND + "warmup_calls: 2\n"

        with pytest.raises(ConfigError, match=r"^cloud\.yaml: unknown key 'warmup_calls'"):
            load_backend(_written(tmp_path, "cloud.yaml", text))

    def test_a_backend_with_no_compose_file_is_refused(self, tmp_path: Path) -> None:
        text = BACKEND.replace('compose_files: ["compose.yaml", "compose.cloud.yaml"]', "")

        with pytest.raises(ConfigError, match="compose_files"):
            load_backend(_written(tmp_path, "cloud.yaml", text))


class TestHarness:
    def test_the_harness_file_reads_into_thresholds_and_addresses(self, tmp_path: Path) -> None:
        harness = load_harness(_written(tmp_path, "harness.yaml", HARNESS))

        assert harness.network.caller_address == "172.28.8.40"
        assert harness.network.caller_media_port == 7078
        assert harness.pause_bounds_ms == (600, 1200)
        assert harness.activity.loud_rms == 300.0
        assert harness.activity.frames_to_open == 3
        assert harness.signal.minimum_farewell_seconds == 0.4
        assert harness.caller_voice == "am_michael"
        assert harness.transfer_target == "sip:manager@harness.example"

    def test_speech_must_be_louder_than_the_line_it_is_heard_over(self, tmp_path: Path) -> None:
        """Below the quiet threshold is background, so a louder one would never open a span."""
        text = HARNESS.replace("loud_rms: 300", "loud_rms: 50")

        with pytest.raises(ConfigError, match="loud_rms"):
            load_harness(_written(tmp_path, "harness.yaml", text))

    def test_a_pause_range_that_runs_backwards_is_refused(self, tmp_path: Path) -> None:
        text = HARNESS.replace("pause_bounds_ms: [600, 1200]", "pause_bounds_ms: [1200, 600]")

        with pytest.raises(ConfigError, match="pause_bounds_ms"):
            load_harness(_written(tmp_path, "harness.yaml", text))

    def test_a_pause_bound_off_the_frame_grid_is_refused(self, tmp_path: Path) -> None:
        """A scenario's pause is whole frames, so a bound it cannot reach is one nothing meets."""
        text = HARNESS.replace("pause_bounds_ms: [600, 1200]", "pause_bounds_ms: [610, 1200]")

        with pytest.raises(ConfigError, match="pause_bounds_ms"):
            load_harness(_written(tmp_path, "harness.yaml", text))
