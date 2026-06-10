from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_tier2_pass_through_regime_persistence.py"
    spec = importlib.util.spec_from_file_location("run_tier2_pass_through_regime_persistence", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_numeric_rows_requires_common_outcomes_and_controls() -> None:
    runner = _load_runner()
    rows = [
        {
            "quarter": "2020Q1",
            runner.PRIMARY_TREATMENT_ID: "1",
            "matched_total_deposits": "2",
            runner.PRIMARY_RESIDUAL_ID: "1",
            "control": "3",
        },
        {
            "quarter": "2020Q2",
            runner.PRIMARY_TREATMENT_ID: "2",
            "matched_total_deposits": "",
            runner.PRIMARY_RESIDUAL_ID: "1",
            "control": "3",
        },
    ]

    numeric = runner._numeric_rows(rows, outcome_ids=["matched_total_deposits", runner.PRIMARY_RESIDUAL_ID], control_ids=["control"])

    assert len(numeric) == 1
    assert numeric[0]["quarter"] == "2020Q1"
    assert numeric[0][runner.PRIMARY_TREATMENT_ID] == 1.0


def test_residualize_removes_control_projection() -> None:
    runner = _load_runner()

    residuals = runner._residualize([2.0, 4.0, 6.0, 8.0], [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])

    assert max(abs(value) for value in residuals) < 1e-10


def test_full_design_leverage_has_expected_trace() -> None:
    runner = _load_runner()
    x_rows = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]

    leverages = runner._full_design_leverage(x_rows)

    assert len(leverages) == 4
    assert abs(sum(leverages) - 2.0) < 1e-10
    assert all(0.0 <= value <= 1.0 for value in leverages)
