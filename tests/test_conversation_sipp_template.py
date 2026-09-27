"""The SIPp scenario the caller runs, rendered for each call from the committed template."""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

from tools.conversation.sipp_template import ere_escape, render_scenario

TEMPLATE = (
    Path(__file__).resolve().parent.parent
    / "integration"
    / "conversation"
    / "sipp"
    / "conversation_uac.xml.template"
)


def _render(*, refer_to: str | None = "sip:manager@harness.example") -> str:
    return render_scenario(
        TEMPLATE.read_text(encoding="utf-8"),
        outcome_timeout_ms=90000,
        refer_to=refer_to,
        engine_address="172.28.8.10",
        caller_media_port=7078,
    )


class TestTheCommittedTemplate:
    def test_it_renders_to_well_formed_xml(self) -> None:
        rendered = _render()

        assert ElementTree.fromstring(rendered).tag == "scenario"
        assert "%{" not in rendered

    def test_the_wait_for_the_outcome_uses_the_backend_timeout(self) -> None:
        root = ElementTree.fromstring(_render())

        (wait,) = [
            element
            for element in root.iter("recv")
            if element.get("request") == "BYE" and element.get("timeout")
        ]
        assert wait.get("timeout") == "90000"

    def test_the_refer_must_name_the_exact_target(self) -> None:
        assert r"Refer-To:[^\r\n]*sip:manager@harness\.example" in _render()

    def test_without_a_transfer_any_refer_is_left_for_the_evaluator(self) -> None:
        """A REFER the scenario forbids is a wrong decision to report, not a SIPp failure."""
        assert r"Refer-To:[^\r\n]*sip:[^>;\r\n]*" in _render(refer_to=None)

    def test_the_answer_must_come_from_the_engine(self) -> None:
        assert r"c=IN IP4 172\.28\.8\.10" in _render()

    def test_the_offer_carries_the_callers_media_port(self) -> None:
        assert "m=audio 7078 RTP/AVP 8 101" in _render()

    def test_sipp_keywords_are_left_for_sipp(self) -> None:
        rendered = _render()

        assert "[call_id]" in rendered
        assert "[$refer_to]" in rendered


class TestRendering:
    def test_extended_regular_expression_metacharacters_are_escaped(self) -> None:
        assert ere_escape("a.b*c+d?e(f)g[h]i{j}k|l^m$n\\o") == (
            r"a\.b\*c\+d\?e\(f\)g\[h\]i\{j\}k\|l\^m\$n\\o"
        )

    def test_ordinary_characters_are_not_escaped(self) -> None:
        """POSIX leaves a backslash before an ordinary character undefined, unlike Python's re."""
        assert ere_escape("sip:manager@harness-01") == "sip:manager@harness-01"

    def test_an_unfilled_placeholder_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown_value"):
            render_scenario(
                "<scenario>%{unknown_value}</scenario>",
                outcome_timeout_ms=1,
                refer_to=None,
                engine_address="172.28.8.10",
                caller_media_port=7078,
            )

    def test_output_that_is_not_xml_is_refused(self) -> None:
        with pytest.raises(ValueError, match="XML"):
            render_scenario(
                "<scenario>",
                outcome_timeout_ms=1,
                refer_to=None,
                engine_address="172.28.8.10",
                caller_media_port=7078,
            )
