from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_tier2_rolling_pass_through.py"
    spec = importlib.util.spec_from_file_location("run_tier2_rolling_pass_through", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pearson_returns_none_for_constant_series() -> None:
    runner = _load_runner()

    assert runner._pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None


def test_correlation_row_marks_secondary_descriptive_boundary() -> None:
    runner = _load_runner()
    rows = []
    quarters = [
        f"{year}Q{quarter}"
        for year in range(2010, 2022)
        for quarter in range(1, 5)
    ]
    for idx, quarter in enumerate(quarters):
        rows.append(
            {
                "quarter": quarter,
                runner.PRIMARY_TREATMENT_ID: "" if idx < 8 else str(idx + 1),
                runner.OUTCOMES[0]: "" if idx < 8 else str((idx + 1) * 2),
            }
        )

    row = runner._correlation_row(
        rows=rows,
        outcome_id=runner.OUTCOMES[0],
        window_start="2010Q1",
        window_end="2021Q4",
    )

    assert row is not None
    assert row["diagnostic_type"] == "rolling_pearson_correlation"
    assert row["diagnostic_role"] == "secondary_descriptive_stability_evidence"
    assert "not the canonical pass-through magnitude" in row["claim_boundary"]
    assert row["canonical_interpretation"].startswith("Use beta-per-dollar")
    assert row["correlation"] == 1.0
    assert row["window_start_quarter"] == "2010Q1"
    assert row["window_end_quarter"] == "2021Q4"
    assert row["effective_sample_start"] == "2012Q1"
    assert row["effective_sample_end"] == "2021Q4"


def test_correlation_row_rejects_unobserved_nominal_endpoint() -> None:
    runner = _load_runner()
    rows = [
        {
            "quarter": f"{year}Q{quarter}",
            runner.PRIMARY_TREATMENT_ID: str(idx + 1),
            runner.OUTCOMES[0]: "" if idx == 47 else str((idx + 1) * 2),
        }
        for idx, (year, quarter) in enumerate(
            (year, quarter)
            for year in range(2010, 2022)
            for quarter in range(1, 5)
        )
    ]

    assert (
        runner._correlation_row(
            rows=rows,
            outcome_id=runner.OUTCOMES[0],
            window_start="2010Q1",
            window_end="2021Q4",
        )
        is None
    )


def test_estimate_window_records_truthful_control_complete_bounds(monkeypatch) -> None:
    runner = _load_runner()
    rows = []
    for idx, (year, quarter) in enumerate(
        (year, quarter)
        for year in range(2010, 2022)
        for quarter in range(1, 5)
    ):
        rows.append(
            {
                "quarter": f"{year}Q{quarter}",
                runner.PRIMARY_TREATMENT_ID: str(idx + 1),
                runner.OUTCOMES[0]: str((idx + 1) * 2),
                "pinned_control": "" if idx < 8 else str(idx + 3),
            }
        )

    fit = SimpleNamespace(beta=[0.0, 0.5], ses=[0.0, 0.1], rsquared=0.8)
    monkeypatch.setattr(
        runner,
        "_fit_lp",
        lambda *args, **kwargs: (fit, 40, ["pinned_control"], []),
    )

    row = runner._estimate_window(
        rows=rows,
        controls=["pinned_control"],
        outcome_id=runner.OUTCOMES[0],
        window_start="2010Q1",
        window_end="2021Q4",
    )

    assert row is not None
    assert row["window_start_quarter"] == "2010Q1"
    assert row["window_end_quarter"] == "2021Q4"
    assert row["effective_sample_start"] == "2012Q1"
    assert row["effective_sample_end"] == "2021Q4"
    assert row["effective_sample_label"] == "2012Q1_to_2021Q4"


def test_estimate_window_rejects_unobserved_nominal_endpoint(monkeypatch) -> None:
    runner = _load_runner()
    rows = [
        {
            "quarter": f"{year}Q{quarter}",
            runner.PRIMARY_TREATMENT_ID: str(idx + 1),
            runner.OUTCOMES[0]: "" if idx == 47 else str((idx + 1) * 2),
        }
        for idx, (year, quarter) in enumerate(
            (year, quarter)
            for year in range(2010, 2022)
            for quarter in range(1, 5)
        )
    ]

    def fail_if_called(*args, **kwargs):
        raise AssertionError("endpoint gate must run before fitting")

    monkeypatch.setattr(runner, "_fit_lp", fail_if_called)
    assert (
        runner._estimate_window(
            rows=rows,
            controls=[],
            outcome_id=runner.OUTCOMES[0],
            window_start="2010Q1",
            window_end="2021Q4",
        )
        is None
    )


def test_build_rows_never_emits_after_last_joint_observation(monkeypatch) -> None:
    runner = _load_runner()
    quarters = [
        f"{year}Q{quarter}"
        for year in range(2010, 2023)
        for quarter in range(1, 5)
    ][:50]
    rows = []
    for idx, quarter in enumerate(quarters):
        observed = idx < 48
        rows.append(
            {
                "quarter": quarter,
                runner.PRIMARY_TREATMENT_ID: str(idx + 1) if observed else "",
                runner.OUTCOMES[0]: str((idx + 1) * 2) if observed else "",
                runner.OUTCOMES[1]: str((idx + 1) * -1) if observed else "",
            }
        )

    monkeypatch.setattr(
        runner,
        "_build_inputs",
        lambda: (None, rows, [], 0, 0, [], []),
    )
    calls = []

    def fake_estimate_window(**kwargs):
        calls.append(
            (
                kwargs["window_start"],
                kwargs["window_end"],
                len(kwargs["rows"]),
                kwargs["outcome_id"],
            )
        )
        return {"normalized_beta": 1.0, "outcome": kwargs["outcome_id"]}

    def fake_correlation_row(**kwargs):
        return {"correlation": 1.0, "outcome": kwargs["outcome_id"]}

    monkeypatch.setattr(runner, "_estimate_window", fake_estimate_window)
    monkeypatch.setattr(runner, "_correlation_row", fake_correlation_row)

    estimates, correlations, metadata = runner.build_rows()

    assert len(estimates) == len(runner.OUTCOMES)
    assert len(correlations) == len(runner.OUTCOMES)
    assert {call[0] for call in calls} == {"2010Q1"}
    assert {call[1] for call in calls} == {"2021Q4"}
    assert {call[2] for call in calls} == {48}
    assert metadata["last_joint_observed_quarter"] == {
        outcome: "2021Q4"
        for outcome in runner.OUTCOMES
    }


def test_manifest_records_contract_window_and_effective_bounds(tmp_path: Path, monkeypatch) -> None:
    runner = _load_runner()
    manifest_path = tmp_path / "rolling_summary.json"
    monkeypatch.setattr(runner, "MANIFEST_OUTPUT", manifest_path)
    outcome_id = runner.OUTCOMES[0]
    regression_rows = [
        {
            "outcome": outcome_id,
            "effective_sample_start": "2002Q1",
            "effective_sample_end": "2025Q4",
        }
    ]
    correlation_rows = [
        {
            "outcome": outcome_id,
            "effective_sample_start": "2002Q1",
            "effective_sample_end": "2025Q4",
        }
    ]

    runner._write_manifest(
        regression_rows=regression_rows,
        correlation_rows=correlation_rows,
        control_ids=list(runner.CANONICAL_CONTROL_IDS),
        last_joint_observed_quarter={
            outcome: "2025Q4"
            for outcome in runner.OUTCOMES
        },
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["treatment_id"] == runner.PRIMARY_TREATMENT_ID
    assert payload["outcome_ids"] == runner.OUTCOMES
    assert payload["control_ids"] == list(runner.CANONICAL_CONTROL_IDS)
    assert payload["window_quarters"] == 48
    assert payload["window"]["nominal_quarters"] == 48
    assert payload["effective_sample_start"] == "2002Q1"
    assert payload["effective_sample_end"] == "2025Q4"
    assert payload["last_joint_observed_quarter"][outcome_id] == "2025Q4"
