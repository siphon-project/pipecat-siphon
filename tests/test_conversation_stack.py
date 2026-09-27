"""The harness stack puts a scripted call on the media path a deployment's own call takes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

REPOSITORY = Path(__file__).resolve().parent.parent
QUICKSTART_CONFIG = REPOSITORY / "examples" / "quickstart" / "siphon.yaml"
HARNESS_CONFIG = REPOSITORY / "integration" / "conversation" / "proxy" / "siphon.yaml"
HARNESS_ROUTE = REPOSITORY / "integration" / "conversation" / "proxy" / "route.py"


def _agent_profile(path: Path) -> Any:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return document["media"]["profiles"]["agent"]


class TestAgentProfile:
    def test_the_harness_proxy_carries_the_quickstart_profile_value_for_value(self) -> None:
        """A call that passes on some other profile says nothing about a deployed bot."""
        assert _agent_profile(HARNESS_CONFIG) == _agent_profile(QUICKSTART_CONFIG)

    def test_calls_are_handed_over_on_that_profile(self) -> None:
        source = HARNESS_ROUTE.read_text(encoding="utf-8")

        assert 'profile="agent"' in source
        assert "answer=True" in source
