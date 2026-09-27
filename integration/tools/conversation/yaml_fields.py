"""Read a YAML document strictly: every key read, every value checked, every error placed.

The harness is configured in YAML -- the scenarios, the backends, its own thresholds -- and a typo
in any of them has to stop the run with the file and the key named. Quietly falling back to a
default would run a test of something else and report it as this one. So a mapping is read key by
key, each value is checked as it is read, and `finish` refuses whatever nothing read.
"""

from __future__ import annotations


class ConfigError(ValueError):
    """A harness file with something missing, something unknown, or a value of the wrong shape."""


class Fields:
    """One mapping from a document, read key by key so that no key goes unread."""

    def __init__(
        self,
        value: object,
        *,
        source: str,
        path: str,
        error_type: type[ConfigError] = ConfigError,
    ) -> None:
        """Read `value`, found at `path` in the file `source`, raising errors as `error_type`.

        Raises:
            ConfigError: As `error_type`, when `value` is not a mapping with text keys.

        """
        self._source = source
        self._path = path
        self._error_type = error_type
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise self.error("expected a mapping")
        self._values: dict[str, object] = value
        self._read: set[str] = set()

    def error(self, problem: str, key: str | None = None) -> ConfigError:
        """Build an error located at this mapping, or at `key` inside it."""
        where = self._path if key is None else self._join(key)
        location = f"{where}: " if where else ""
        return self._error_type(f"{self._source}: {location}{problem}")

    def text(self, key: str) -> str:
        """Read required, non-empty text."""
        value = self._take(key)
        if value is None:
            raise self.error("is required", key)
        return self._checked_text(key, value)

    def optional_text(self, key: str) -> str | None:
        """Read non-empty text, or None when the key is absent."""
        value = self._take(key)
        return None if value is None else self._checked_text(key, value)

    def optional_integer(self, key: str) -> int | None:
        """Read a whole number, or None when the key is absent."""
        value = self._take(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise self.error("expected a whole number", key)
        return value

    def positive_integer(self, key: str) -> int:
        """Read a required whole number of at least one."""
        value = self.optional_integer(key)
        if value is None:
            raise self.error("is required", key)
        if value < 1:
            raise self.error("must be at least 1", key)
        return value

    def positive_number(self, key: str) -> float:
        """Read a required number greater than zero, whole or decimal."""
        value = self._take(key)
        if value is None:
            raise self.error("is required", key)
        number = self._checked_number(key, value)
        if number <= 0:
            raise self.error("must be greater than 0", key)
        return number

    def optional_number(self, key: str) -> float | None:
        """Read a number of at least zero, or None when the key is absent."""
        value = self._take(key)
        if value is None:
            return None
        number = self._checked_number(key, value)
        if number < 0:
            raise self.error("must not be negative", key)
        return number

    def integer_pair(self, key: str) -> tuple[int, int]:
        """Read a required list of exactly two whole numbers."""
        value = self._take(key)
        if not isinstance(value, list) or len(value) != 2:
            raise self.error("expected two whole numbers", key)
        first, second = value
        if any(isinstance(item, bool) or not isinstance(item, int) for item in (first, second)):
            raise self.error("expected two whole numbers", key)
        return int(first), int(second)

    def texts(self, key: str) -> tuple[str, ...]:
        """Read a list of non-empty text, empty when the key is absent."""
        value = self._take(key)
        if value is None:
            return ()
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise self.error("expected a list of text", key)
        return tuple(value)

    def choice(self, key: str, choices: tuple[str, ...], *, default: str | None = None) -> str:
        """Read one of `choices`: `default` when absent, and required when there is no default."""
        value = self._take(key)
        if value is None:
            if default is None:
                raise self.error("is required", key)
            return default
        if not isinstance(value, str) or value not in choices:
            raise self.error(f"{value!r} is not one of {', '.join(choices)}", key)
        return value

    def mapping(self, key: str) -> Fields:
        """Read a required nested mapping."""
        nested = self.optional_mapping(key)
        if nested is None:
            raise self.error("is required", key)
        return nested

    def optional_mapping(self, key: str) -> Fields | None:
        """Read a nested mapping, or None when the key is absent."""
        value = self._take(key)
        if value is None:
            return None
        return Fields(value, source=self._source, path=self._join(key), error_type=self._error_type)

    def mappings(self, key: str) -> list[Fields]:
        """Read a list of nested mappings, empty when the key is absent."""
        value = self._take(key)
        if value is None:
            return []
        if not isinstance(value, list):
            raise self.error("expected a list", key)
        return [
            Fields(
                item,
                source=self._source,
                path=f"{self._join(key)}[{index}]",
                error_type=self._error_type,
            )
            for index, item in enumerate(value)
        ]

    def text_mapping(self, key: str) -> dict[str, str]:
        """Read a mapping of text to text, empty when the key is absent. Numbers read as text.

        For settings passed straight through to something else -- a container's environment -- where
        the keys are not this harness's to know, so `finish` cannot check them.
        """
        value = self._take(key)
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise self.error("expected a mapping of text to text", key)
        mapping: dict[str, str] = {}
        for name, item in value.items():
            if not isinstance(name, str):
                raise self.error("expected a mapping of text to text", key)
            if isinstance(item, bool) or not isinstance(item, str | int | float):
                raise self.error(f"{name!r} is not text or a number", key)
            mapping[name] = item if isinstance(item, str) else str(item)
        return mapping

    def finish(self) -> None:
        """Refuse any key that nothing read."""
        for key in self._values:
            if key not in self._read:
                raise self.error(f"unknown key {key!r}")

    def _take(self, key: str) -> object:
        self._read.add(key)
        return self._values.get(key)

    def _checked_text(self, key: str, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise self.error("expected non-empty text", key)
        return value

    def _checked_number(self, key: str, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise self.error("expected a number", key)
        return float(value)

    def _join(self, key: str) -> str:
        return f"{self._path}.{key}" if self._path else key
