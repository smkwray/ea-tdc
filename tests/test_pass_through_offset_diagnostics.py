from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_tier2_pass_through_offset_diagnostics.py"
    spec = importlib.util.spec_from_file_location("run_tier2_pass_through_offset_diagnostics", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quarter_key_sorts_calendar_order() -> None:
    runner = _load_runner()

    quarters = ["2020Q4", "2019Q4", "2020Q1", "2021Q1"]

    assert sorted(quarters, key=runner._quarter_key) == ["2019Q4", "2020Q1", "2020Q4", "2021Q1"]


def test_shifted_pairs_positive_shift_uses_future_outcome() -> None:
    runner = _load_runner()
    rows = [
        {"quarter": "2020Q1", "tdc": "1", "outcome": "10"},
        {"quarter": "2020Q2", "tdc": "2", "outcome": "20"},
        {"quarter": "2020Q3", "tdc": "3", "outcome": "30"},
    ]

    treatments, outcomes = runner._shifted_pairs(
        rows,
        treatment_id="tdc",
        outcome_id="outcome",
        outcome_shift_quarters=1,
    )

    assert treatments == [1.0, 2.0]
    assert outcomes == [20.0, 30.0]


def test_identity_rows_check_deposit_equals_one_plus_residual() -> None:
    runner = _load_runner()

    rows = [
        {
            "window_start_quarter": "2010Q1",
            "window_end_quarter": "2021Q4",
            "outcome": "matched_total_deposits",
            "normalized_beta": "0.62",
        },
        {
            "window_start_quarter": "2010Q1",
            "window_end_quarter": "2021Q4",
            "outcome": runner.PRIMARY_RESIDUAL_ID,
            "normalized_beta": "-0.38",
        },
    ]

    identity = runner._identity_rows_from_rolling(rows)

    assert len(identity) == 1
    assert abs(identity[0]["deposit_minus_residual_minus_one"]) < 1e-12


def test_canonical_controls_inserts_splice_control_when_available() -> None:
    runner = _load_runner()

    rows = [{"quarter": "2020Q1", runner.CANONICAL_SPLICE_CONTROL: "1"}]
    controls = ["GDP", "TOTRESNS", "dflmx_k100_f1"]

    updated = runner._canonical_controls(rows, controls)

    assert updated == ["GDP", "TOTRESNS", runner.CANONICAL_SPLICE_CONTROL, "dflmx_k100_f1"]
    assert controls == ["GDP", "TOTRESNS", "dflmx_k100_f1"]


def test_level_summary_converts_periods_without_external_data() -> None:
    runner = _load_runner()
    rows = [
        {"quarter": "2019Q4", runner.PRIMARY_TREATMENT_ID: "1000"},
        {"quarter": "2020Q1", runner.PRIMARY_TREATMENT_ID: "3000"},
        {"quarter": "2020Q2", runner.PRIMARY_TREATMENT_ID: "5000"},
        {"quarter": "2022Q1", runner.PRIMARY_TREATMENT_ID: "-1000"},
    ]

    summaries = {row["period"]: row for row in runner._level_summary(rows)}

    assert summaries["full_available"]["n"] == 4
    assert summaries["pre_2020"]["mean_mil"] == 1000.0
    assert summaries["covid_2020_2021"]["max_quarter"] == "2020Q2"
    assert summaries["post_2022"]["min_mil"] == -1000.0


def test_rolling_beta_correlates_rank_window_features() -> None:
    runner = _load_runner()
    feature_rows = []
    for idx in range(12):
        beta = float(idx)
        share = 0.0 if idx < 6 else 1.0
        feature_rows.append(
            {
                "feature_group": "window",
                "feature_id": "share_2020_2021",
                "feature_label": "Share of quarters in 2020Q1-2021Q4",
                "feature_stat": "window_value",
                "window_start_quarter": f"200{idx // 4}Q{idx % 4 + 1}",
                "window_end_quarter": f"201{idx // 4}Q{idx % 4 + 1}",
                "deposit_beta_per_dollar_tdc": beta,
                "feature_value": share,
                "diagnostic_role": "regime_composition",
            }
        )
        feature_rows.append(
            {
                "feature_group": "test",
                "feature_id": "candidate_linear",
                "feature_label": "Candidate linear feature",
                "feature_stat": "window_value",
                "window_start_quarter": f"200{idx // 4}Q{idx % 4 + 1}",
                "window_end_quarter": f"201{idx // 4}Q{idx % 4 + 1}",
                "deposit_beta_per_dollar_tdc": beta,
                "feature_value": beta * 2.0,
                "diagnostic_role": "test_candidate",
            }
        )
        feature_rows.append(
            {
                "feature_group": "tdc_scale",
                "feature_id": "tdc_mean_abs_mil",
                "feature_label": "tdc_mean_abs_mil",
                "feature_stat": "window_value",
                "window_start_quarter": f"200{idx // 4}Q{idx % 4 + 1}",
                "window_end_quarter": f"201{idx // 4}Q{idx % 4 + 1}",
                "deposit_beta_per_dollar_tdc": beta,
                "feature_value": -beta,
                "diagnostic_role": "window_tdc_scale",
            }
        )

    correlates = runner._rolling_beta_correlates(feature_rows)

    assert len(correlates) == 3
    assert correlates[0]["abs_correlation"] == 1.0
    assert {row["feature_id"] for row in correlates} == {"share_2020_2021", "candidate_linear", "tdc_mean_abs_mil"}
    assert all(
        row["correlation_residualized_on_share_2020_2021"] != ""
        for row in correlates
        if row["feature_id"] != "share_2020_2021"
    )
