from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_tier2_pass_through_regime_validation.py"
    spec = importlib.util.spec_from_file_location("run_tier2_pass_through_regime_validation", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quantile_interpolates_between_ordered_values() -> None:
    runner = _load_runner()

    assert runner._quantile([10.0, 0.0, 20.0, 30.0], 0.25) == 7.5
    assert runner._quantile([10.0, 0.0, 20.0, 30.0], 0.75) == 22.5


def test_slope_no_intercept_requires_support() -> None:
    runner = _load_runner()

    assert runner._slope_no_intercept([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == 2.0
    assert runner._slope_no_intercept([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) is None
    assert runner._slope_no_intercept([1.0, 2.0], [2.0, 4.0]) is None


def test_candidate_abs_thresholds_use_absolute_quantiles() -> None:
    runner = _load_runner()
    rows = [
        {
            "quarter": f"2020Q{idx + 1}",
            runner.PRIMARY_TREATMENT_ID: str(value),
            "tga_balance_qoq": str(value),
            "reserve_balances_qoq": str(value),
            "on_rrp_balance_qoq": str(value),
            "FEDFUNDS": str(idx),
            "reserve_balances": str(idx + 1),
        }
        for idx, value in enumerate([-100.0, -50.0, 10.0, 20.0])
    ]

    specs = {row["candidate_trigger_id"]: row for row in runner._candidate_specs(rows)}

    assert specs["high_tdc_abs_q75"]["threshold"] == 62.5
    assert specs["reserve_change_abs_q75"]["threshold"] == 62.5
    assert specs["on_rrp_flow_abs_q75"]["threshold"] == 62.5


def test_lp_complete_case_quarters_exclude_missing_controls_and_horizon() -> None:
    runner = _load_runner()
    rows = [
        {"quarter": "2000Q1", "treatment": "1", "outcome": "10", "control": "5"},
        {"quarter": "2000Q2", "treatment": "2", "outcome": "11", "control": ""},
        {"quarter": "2000Q3", "treatment": "3", "outcome": "12", "control": "7"},
    ]

    assert runner._lp_complete_case_quarters(
        rows,
        treatment_id="treatment",
        outcome_id="outcome",
        horizon=0,
        control_ids=["control"],
    ) == ["2000Q1", "2000Q3"]
    assert runner._lp_complete_case_quarters(
        rows,
        treatment_id="treatment",
        outcome_id="outcome",
        horizon=1,
        control_ids=["control"],
    ) == ["2000Q1"]


def test_contract_blocks_runtime_selection_even_when_scenario_allowed() -> None:
    runner = _load_runner()
    estimates = [
        {
            "regime_id": "normal_forward",
            "horizon": 0,
            "status": "estimated",
            "n": 40,
            "point_estimate": 0.3,
            "lower95": 0.1,
            "upper95": 0.5,
            "estimator_id": "test_estimator",
            "data_coverage_start": "1945Q4",
            "data_coverage_end": "2026Q2",
            "estimation_sample_start": "2002Q1",
            "estimation_sample_end": "2025Q4",
            "n_complete_cases": 40,
            "sample_start": "2002Q1",
            "sample_end": "2025Q4",
            "sample_window": "2002Q1_to_2025Q4",
        },
        {
            "regime_id": "pooled_full_sample",
            "horizon": 0,
            "status": "estimated",
            "n": 100,
            "point_estimate": 0.6,
            "lower95": 0.3,
            "upper95": 0.9,
            "estimator_id": "test_pooled",
        },
    ]
    validations = [
        {
            "regime_id": "normal_forward",
            "validation_case": "oos_classifier_vs_pooled_baseline",
            "validation_status": "review_only_oos_improves_not_runtime_grade",
        }
    ]

    rows = {row["regime_id"]: row for row in runner._contract_rows(estimates, validations)}

    assert rows["normal_forward"]["recommended_ratewall_use"] == "assumption_mode_scenario_allowed"
    assert rows["normal_forward"]["scenario_default_allowed"] == "true"
    assert rows["normal_forward"]["runtime_selector_allowed"] == "false"
    assert rows["normal_forward"]["data_coverage_start"] == "1945Q4"
    assert rows["normal_forward"]["estimation_sample_start"] == "2002Q1"
    assert rows["normal_forward"]["sample_window"] == "2002Q1_to_2025Q4"
    assert rows["pooled_full_sample"]["recommended_ratewall_use"] == "review_only"


def test_totresns_decision_uses_materiality_threshold() -> None:
    runner = _load_runner()
    estimates = [
        {
            "regime_id": "normal_forward",
            "horizon": 0,
            "robustness_check": "no_contemporaneous_totresns",
            "controls_variant": "with_contemporaneous_totresns",
            "point_estimate": 0.342,
        },
        {
            "regime_id": "normal_forward",
            "horizon": 0,
            "robustness_check": "no_contemporaneous_totresns",
            "controls_variant": "no_contemporaneous_totresns",
            "point_estimate": 0.40,
        },
        {
            "regime_id": "pooled_full_sample",
            "horizon": 0,
            "robustness_check": "no_contemporaneous_totresns",
            "controls_variant": "no_contemporaneous_totresns",
            "point_estimate": 0.55,
        },
    ]

    decision = runner._totresns_decision(estimates)

    assert decision["status"] == "freeze_ok"
    assert round(decision["delta"], 3) == 0.058
    assert "within the 0.15 materiality rule" in decision["message"]
