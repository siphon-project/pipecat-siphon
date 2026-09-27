"""Many calls, one verdict: pass counts per backend and scenario, what failed, and how fast.

A run the stack or the harness spoiled is a verdict on neither the agent nor its pass rate, so it
is counted apart as inconclusive: a run whose only failures are `infrastructure` or `harness`. It
does not dilute the pass fraction, but it still fails the exit code, because a report built on
runs that did not happen is not a report that everything works.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from tools.conversation.expectations import Category, CheckResult

INCONCLUSIVE_CATEGORIES: frozenset[Category] = frozenset({"infrastructure", "harness"})

Label = Literal["PASS", "PARTIAL", "FAIL", "INCONCLUSIVE"]


@dataclass(frozen=True)
class RunVerdict:
    """One call's checks, and what it measured."""

    backend: str
    scenario: str
    repetition: int
    results: tuple[CheckResult, ...]
    measurements: Mapping[str, float]

    @property
    def passed(self) -> bool:
        """Whether every check passed."""
        return all(result.passed for result in self.results)

    @property
    def failure_categories(self) -> frozenset[Category]:
        """The categories of the checks that failed."""
        return frozenset(result.category for result in self.results if not result.passed)

    @property
    def inconclusive(self) -> bool:
        """Whether the run failed only on the stack or the harness."""
        failed = self.failure_categories
        return bool(failed) and failed <= INCONCLUSIVE_CATEGORIES


@dataclass(frozen=True)
class Cell:
    """Every run of one scenario on one backend."""

    backend: str
    scenario: str
    passed: int
    counted: int
    """Runs that are a verdict on the agent, passed or not; inconclusive runs are left out."""
    inconclusive: int
    failure_categories: Mapping[str, int]
    """How many counted runs failed with each category."""

    @property
    def label(self) -> Label:
        """PASS, PARTIAL or FAIL over the counted runs; INCONCLUSIVE when there are none."""
        if self.counted == 0:
            return "INCONCLUSIVE"
        if self.passed == self.counted:
            return "PASS"
        if self.passed == 0:
            return "FAIL"
        return "PARTIAL"


@dataclass(frozen=True)
class Stats:
    """A measurement's spread over a backend's runs."""

    count: int
    median: float
    p95: float


@dataclass(frozen=True)
class Summary:
    """The whole run: every cell, and latency per backend."""

    backends: tuple[str, ...]
    scenarios: tuple[str, ...]
    cells: tuple[Cell, ...]
    latency: Mapping[str, Mapping[str, Stats]]

    @property
    def exit_code(self) -> int:
        """0 when every cell passed with no inconclusive runs, 1 otherwise."""
        if self.cells and all(
            cell.label == "PASS" and cell.inconclusive == 0 for cell in self.cells
        ):
            return 0
        return 1

    def cell(self, backend: str, scenario: str) -> Cell | None:
        """Return the cell for `scenario` on `backend`, or None if it never ran."""
        return next(
            (cell for cell in self.cells if cell.backend == backend and cell.scenario == scenario),
            None,
        )


def summarize(verdicts: Sequence[RunVerdict]) -> Summary:
    """Aggregate run verdicts into cells and latency, in the order they first appear."""
    backends = tuple(dict.fromkeys(verdict.backend for verdict in verdicts))
    scenarios = tuple(dict.fromkeys(verdict.scenario for verdict in verdicts))

    cells: list[Cell] = []
    for scenario in scenarios:
        for backend in backends:
            runs = [
                verdict
                for verdict in verdicts
                if verdict.backend == backend and verdict.scenario == scenario
            ]
            if not runs:
                continue
            counted = [run for run in runs if not run.inconclusive]
            categories = Counter(
                category for run in counted for category in sorted(run.failure_categories)
            )
            cells.append(
                Cell(
                    backend=backend,
                    scenario=scenario,
                    passed=sum(1 for run in counted if run.passed),
                    counted=len(counted),
                    inconclusive=len(runs) - len(counted),
                    failure_categories={
                        str(category): count for category, count in categories.items()
                    },
                )
            )

    latency: dict[str, dict[str, Stats]] = {}
    for backend in backends:
        values: dict[str, list[float]] = {}
        for verdict in verdicts:
            if verdict.backend != backend:
                continue
            for name, value in verdict.measurements.items():
                values.setdefault(name, []).append(value)
        latency[backend] = {name: _stats(samples) for name, samples in values.items()}

    return Summary(backends=backends, scenarios=scenarios, cells=tuple(cells), latency=latency)


def render_text(summary: Summary) -> str:
    """Render the summary as a table with one row per scenario and one column per backend."""
    header = ["scenario", *summary.backends]
    rows = [header]
    for scenario in summary.scenarios:
        row = [scenario]
        for backend in summary.backends:
            cell = summary.cell(backend, scenario)
            row.append("-" if cell is None else _cell_text(cell))
        rows.append(row)
    widths = [max(len(row[column]) for row in rows) for column in range(len(header))]
    lines = [
        "  ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)).rstrip()
        for row in rows
    ]
    for backend in summary.backends:
        for name, stats in sorted(summary.latency.get(backend, {}).items()):
            lines.append(
                f"{backend} {name}: median {stats.median:.2f}s, p95 {stats.p95:.2f}s "
                f"over {stats.count} run(s)"
            )
    return "\n".join(lines) + "\n"


def _cell_text(cell: Cell) -> str:
    text = f"{cell.passed}/{cell.counted} {cell.label}"
    if cell.failure_categories:
        text += " " + ", ".join(
            f"{category} x{count}" for category, count in sorted(cell.failure_categories.items())
        )
    if cell.inconclusive:
        text += f" (+{cell.inconclusive} inconclusive)"
    return text


def _stats(samples: list[float]) -> Stats:
    ordered = sorted(samples)
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return Stats(count=len(ordered), median=statistics.median(ordered), p95=ordered[rank - 1])
