from __future__ import annotations

import importlib.util
from pathlib import Path


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
    for idx in range(runner.MIN_OBSERVATIONS):
        rows.append(
            {
                runner.PRIMARY_TREATMENT_ID: str(idx + 1),
                "matched_total_deposits": str((idx + 1) * 2),
            }
        )

    row = runner._correlation_row(
        rows=rows,
        outcome_id="matched_total_deposits",
        window_start="2010Q1",
        window_end="2021Q4",
    )

    assert row is not None
    assert row["diagnostic_type"] == "rolling_pearson_correlation"
    assert row["diagnostic_role"] == "secondary_descriptive_stability_evidence"
    assert "not the canonical pass-through magnitude" in row["claim_boundary"]
    assert row["canonical_interpretation"].startswith("Use beta-per-dollar")
    assert row["correlation"] == 1.0
