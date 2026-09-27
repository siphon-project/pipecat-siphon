"""Strict reading of the harness's YAML: every key read, every value checked, every error placed."""

from __future__ import annotations

import pytest

from tools.conversation.yaml_fields import ConfigError, Fields


def _fields(value: object, *, path: str = "") -> Fields:
    return Fields(value, source="backend.yaml", path=path)


class TestNumbers:
    def test_whole_and_decimal_numbers_both_read(self) -> None:
        fields = _fields({"whole": 2, "decimal": 0.25})

        assert fields.positive_number("whole") == 2.0
        assert fields.positive_number("decimal") == 0.25

    @pytest.mark.parametrize("value", [0, -1.5, True, "1.0", None])
    def test_anything_but_a_positive_number_is_refused_where_it_stands(self, value: object) -> None:
        fields = _fields({"greeting_timeout_seconds": value}, path="timing")

        with pytest.raises(
            ConfigError, match=r"^backend\.yaml: timing\.greeting_timeout_seconds: "
        ):
            fields.positive_number("greeting_timeout_seconds")

    def test_an_optional_number_may_be_absent_or_zero(self) -> None:
        fields = _fields({"temperature": 0})

        assert fields.optional_number("temperature") == 0.0
        assert fields.optional_number("top_p") is None

    @pytest.mark.parametrize("value", [-0.1, False, "0"])
    def test_an_optional_number_is_still_checked_when_present(self, value: object) -> None:
        with pytest.raises(ConfigError, match="temperature"):
            _fields({"temperature": value}).optional_number("temperature")

    def test_a_pair_of_whole_numbers(self) -> None:
        assert _fields({"bounds": [600, 1200]}).integer_pair("bounds") == (600, 1200)

    @pytest.mark.parametrize("value", [[600], [600, 1200, 1800], ["600", 1200], [600, True], 600])
    def test_anything_but_two_whole_numbers_is_refused(self, value: object) -> None:
        with pytest.raises(ConfigError, match="bounds"):
            _fields({"bounds": value}).integer_pair("bounds")


class TestStrictness:
    def test_a_key_nothing_read_is_refused(self) -> None:
        fields = _fields({"name": "cloud", "nmae": "cloud"})
        fields.text("name")

        with pytest.raises(ConfigError, match="unknown key 'nmae'"):
            fields.finish()

    def test_errors_are_raised_as_the_readers_own_type(self) -> None:
        """A scenario's errors are ScenarioErrors, so its callers can tell them apart."""

        class ScenarioProblem(ConfigError):
            pass

        with pytest.raises(ScenarioProblem, match="expected a mapping"):
            Fields([], source="goodbye.yaml", path="", error_type=ScenarioProblem)


class TestPassThroughMapping:
    def test_text_and_numbers_read_as_text(self) -> None:
        """A container's environment is text, and a number written in YAML is still a setting."""
        fields = _fields({"environment": {"BOT_BACKEND": "local", "BOT_ECHO_GUARD_TAIL": 0.6}})

        assert fields.text_mapping("environment") == {
            "BOT_BACKEND": "local",
            "BOT_ECHO_GUARD_TAIL": "0.6",
        }

    def test_an_absent_mapping_is_empty(self) -> None:
        assert _fields({}).text_mapping("environment") == {}

    @pytest.mark.parametrize("value", [{"BOT_BACKEND": ["local"]}, {"BOT_BACKEND": True}, ["a"]])
    def test_anything_that_is_not_a_setting_is_refused(self, value: object) -> None:
        with pytest.raises(ConfigError, match="environment"):
            _fields({"environment": value}).text_mapping("environment")
