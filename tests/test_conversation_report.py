"""Many calls, one verdict: pass counts per backend and scenario, and what the failures were."""

from __future__ import annotations

from tools.conversation.expectations import Category, CheckResult
from tools.conversation.report import RunVerdict, render_text, summarize


def _check(passed: bool, category: Category = "decision") -> CheckResult:
    return CheckResult(name="outcome", passed=passed, category=category, message="")


def _run(
    backend: str,
    scenario: str,
    repetition: int,
    *failures: Category,
    reply_seconds: float | None = None,
) -> RunVerdict:
    results = [_check(True)] + [_check(False, category) for category in failures]
    measurements = {} if reply_seconds is None else {"reply_seconds": reply_seconds}
    return RunVerdict(
        backend=backend,
        scenario=scenario,
        repetition=repetition,
        results=tuple(results),
        measurements=measurements,
    )


class TestCells:
    def test_every_run_passing_is_a_pass(self) -> None:
        summary = summarize([_run("cloud", "goodbye", index) for index in range(1, 6)])

        (cell,) = summary.cells
        assert (cell.passed, cell.counted, cell.inconclusive, cell.label) == (5, 5, 0, "PASS")
        assert summary.exit_code == 0

    def test_one_failure_is_partial_and_names_its_category(self) -> None:
        runs = [_run("cloud", "goodbye", index) for index in range(1, 5)]
        runs.append(_run("cloud", "goodbye", 5, "decision"))

        summary = summarize(runs)

        (cell,) = summary.cells
        assert (cell.passed, cell.counted, cell.label) == (4, 5, "PARTIAL")
        assert cell.failure_categories == {"decision": 1}
        assert summary.exit_code == 1

    def test_no_run_passing_is_a_fail(self) -> None:
        summary = summarize([_run("cloud", "goodbye", index, "media") for index in range(1, 4)])

        assert summary.cells[0].label == "FAIL"

    def test_a_run_the_stack_or_the_harness_spoiled_is_inconclusive(self) -> None:
        """Not a verdict on the agent either way, so it neither passes nor fails the cell."""
        runs = [
            _run("cloud", "goodbye", 1),
            _run("cloud", "goodbye", 2, "infrastructure"),
            _run("cloud", "goodbye", 3, "harness", "infrastructure"),
        ]

        summary = summarize(runs)

        (cell,) = summary.cells
        assert (cell.passed, cell.counted, cell.inconclusive, cell.label) == (1, 1, 2, "PASS")
        assert summary.exit_code == 1

    def test_an_agent_failure_beside_a_stack_failure_still_counts(self) -> None:
        summary = summarize([_run("cloud", "goodbye", 1, "infrastructure", "decision")])

        (cell,) = summary.cells
        assert (cell.passed, cell.counted, cell.inconclusive, cell.label) == (0, 1, 0, "FAIL")

    def test_a_cell_with_only_inconclusive_runs_is_inconclusive(self) -> None:
        summary = summarize([_run("cloud", "goodbye", 1, "harness")])

        assert summary.cells[0].label == "INCONCLUSIVE"


class TestLatency:
    def test_median_and_95th_percentile_per_backend(self) -> None:
        runs = [
            _run("cloud", "goodbye", index, reply_seconds=seconds)
            for index, seconds in enumerate([1.0, 2.0, 3.0, 4.0, 10.0], start=1)
        ]
        runs.append(_run("local-cuda", "goodbye", 1, reply_seconds=0.5))

        summary = summarize(runs)

        cloud = summary.latency["cloud"]["reply_seconds"]
        assert (cloud.count, cloud.median, cloud.p95) == (5, 3.0, 10.0)
        assert summary.latency["local-cuda"]["reply_seconds"].median == 0.5


class TestText:
    def test_rows_follow_scenarios_and_columns_follow_backends_in_first_seen_order(self) -> None:
        runs = [
            _run("local-cuda", "goodbye", 1),
            _run("cloud", "goodbye", 1, "decision"),
            _run("cloud", "transfer_named_manager", 1),
        ]

        lines = render_text(summarize(runs)).splitlines()

        header = lines[0].split()
        assert header == ["scenario", "local-cuda", "cloud"]
        goodbye = next(line for line in lines if line.startswith("goodbye"))
        assert "1/1 PASS" in goodbye
        assert "0/1 FAIL" in goodbye
        assert "decision x1" in goodbye
        transfer = next(line for line in lines if line.startswith("transfer_named_manager"))
        assert " - " in f" {transfer} "
