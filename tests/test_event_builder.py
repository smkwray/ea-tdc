from __future__ import annotations

import csv
import json
from pathlib import Path

from ea_tdc.designs.events import build_event_design
from ea_tdc.paths import ensure_repo_dirs, project_paths


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_event_design_builder_writes_release_horizon_bundle(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)

    _write_text(
        paths.config / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: qra_event_rates_63bd",
                "    estimator: event_lp",
                "    freq: irregular_event",
                "    treatment_id: qra_release_63bd",
                "    outcomes: [threefytp10, dgs2, repo_spread]",
                "    horizons_bd: [1, 5]",
                "    cutoff_rule: event_close_with_embargo",
                "    output_family: supporting_reduced_form",
            ]
        ),
    )
    _write_text(
        paths.bundles / "qrawatch" / "event_bundle.csv",
        "\n".join(
            [
                "event_id,quarter,event_label,event_date,event_type,source_repo,treatment_id,treatment_value,treatment_units,cutoff_timestamp,embargo_rule,horizon_unit,usable_for_headline,usable_for_headline_reason,quality_tier,claim_scope",
                "qra_2024_01,2024Q1,2024 Jan QRA,2024-01-03,qra_release,qrawatch,canonical_shock_bn,10,usd_billions,2024-01-03T08:30:00-05:00,event_close_with_embargo,business_day,true,usable,Tier A,headline",
                "qra_2024_02,2024Q1,2024 Feb QRA,2024-01-10,qra_release,qrawatch,canonical_shock_bn,,usd_billions,2024-01-10T08:30:00-05:00,event_close_with_embargo,business_day,true,missing_shock,Tier C,descriptive_only",
            ]
        ),
    )
    _write_text(
        paths.config / "debt_limit_intervals.csv",
        "\n".join(
            [
                "quarter,start_date,end_date,source_note",
                "2024Q1,2024-01-01,2024-01-15,test fixture",
            ]
        ),
    )

    for series_id, rows in {
        "THREEFYTP10": [
            ("2023-12-27", "1.50"),
            ("2024-01-02", "1.60"),
            ("2024-01-04", "1.66"),
            ("2024-01-10", "1.70"),
        ],
        "DGS2": [
            ("2023-12-27", "3.80"),
            ("2024-01-02", "4.00"),
            ("2024-01-04", "4.20"),
            ("2024-01-10", "4.50"),
        ],
        "TGCRRATE": [
            ("2023-12-27", "5.25"),
            ("2024-01-02", "5.30"),
            ("2024-01-04", "5.35"),
            ("2024-01-10", "5.40"),
        ],
        "RRPONTSYAWARD": [
            ("2023-12-27", "5.00"),
            ("2024-01-02", "5.00"),
            ("2024-01-04", "5.02"),
            ("2024-01-10", "5.00"),
        ],
        "DFF": [
            ("2023-12-27", "5.30"),
            ("2024-01-02", "5.33"),
            ("2024-01-04", "5.38"),
            ("2024-01-10", "5.42"),
        ],
        "SOFR": [
            ("2023-12-27", "5.28"),
            ("2024-01-02", "5.31"),
            ("2024-01-04", "5.35"),
            ("2024-01-10", "5.39"),
        ],
    }.items():
        _write_text(
            paths.raw_fred / f"{series_id}.csv",
            "\n".join(["date,value", *[f"{observed_date},{value}" for observed_date, value in rows]]),
        )

    result = build_event_design(paths, job_id="qra_event_rates_63bd")

    assert result.rows_written == 2
    assert result.usable_rows == 1

    with result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["debt_limit_dummy"] == "1"
    assert rows[0]["window_start_date"] == "2024-01-02"
    assert rows[0]["start_date_threefytp10"] == "2024-01-02"
    assert rows[0]["delta_threefytp10_h1bd"] == "6.0"
    assert rows[0]["delta_threefytp10_h5bd"] == "10.0"
    assert rows[0]["start_date_dgs2"] == "2024-01-02"
    assert rows[0]["start_date_repo_spread"] == "2024-01-02"
    assert rows[0]["start_date_dff"] == "2024-01-02"
    assert rows[0]["delta_dgs2_h1bd"] == "20.0"
    assert rows[0]["delta_dgs2_h5bd"] == "50.0"
    assert rows[0]["delta_repo_spread_h1bd"] == "3.0"
    assert rows[0]["delta_repo_spread_h5bd"] == "10.0"
    assert rows[0]["delta_dff_release_plus_1bd"] == "0.05"
    assert rows[0]["delta_dff_release_plus_5bd"] == "0.09"
    assert rows[0]["delta_dgs2_release_minus_5bd_to_minus_1bd"] == "20.0"
    assert rows[0]["delta_dff_release_minus_5bd_to_minus_1bd"] == "0.03"
    assert rows[0]["end_date_repo_spread_h5bd"] == "2024-01-10"
    assert rows[1]["delta_dgs2_h1bd"] == ""

    design_manifest = json.loads(result.design_manifest_path.read_text(encoding="utf-8"))
    assert design_manifest["status"] == "ready_for_estimation"
    assert design_manifest["missing_required_series"] == []
    assert design_manifest["control_ids"] == [
        "debt_limit_dummy",
        "delta_dff_release_plus_1bd",
        "delta_dff_release_plus_5bd",
        "delta_sofr_release_plus_1bd",
        "delta_sofr_release_plus_5bd",
    ]
    assert design_manifest["outcome_ids"] == ["threefytp10", "dgs2", "repo_spread"]
    assert design_manifest["exclusion_windows"] == [
        "release_minus_21bd_to_minus_1bd",
        "release_minus_5bd_to_minus_1bd",
    ]
    assert design_manifest["scaling_rule"] == "rate_and_level_changes_in_catalog_units"
    assert design_manifest["calendar"] == "us_market_holiday_business_day"
    assert design_manifest["sample_policy"] == "headline_strict"
    assert design_manifest["event_sample_counts"]["headline_eligible_rows"] == 1
    assert design_manifest["event_sample_counts"]["reviewed_nonmissing_rows"] == 0
    assert design_manifest["event_sample_counts"]["requested_sample_rows"] == 1
    assert design_manifest["usable_rows"] == 1

    sample_manifest = json.loads(result.sample_manifest_path.read_text(encoding="utf-8"))
    assert sample_manifest["sample_policy"] == "headline_strict"
    assert sample_manifest["counts"]["requested_sample_rows"] == 1
    assert sample_manifest["rows"][1]["observations_remaining"] == 1
    assert sample_manifest["rows"][2]["observations_remaining"] == 1
    assert sample_manifest["rows"][3]["observations_remaining"] == 1


def test_event_design_builder_supports_risk_catalog_and_pct_returns(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)

    _write_text(
        paths.config / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: qra_event_risk_21bd",
                "    estimator: event_lp",
                "    freq: irregular_event",
                "    treatment_id: qra_release_63bd",
                "    outcomes: [sp500_return, vix_change, tga_balance_change, reserve_balances_change, rrp_balance_change, fed_balance_sheet_change]",
                "    horizons_bd: [1, 5]",
                "    cutoff_rule: event_close_with_embargo",
                "    output_family: supporting_reduced_form",
            ]
        ),
    )
    _write_text(
        paths.bundles / "qrawatch" / "event_bundle.csv",
        "\n".join(
            [
                "event_id,quarter,event_label,event_date,event_type,source_repo,treatment_id,treatment_value,treatment_units,cutoff_timestamp,embargo_rule,horizon_unit,usable_for_headline,usable_for_headline_reason,quality_tier,claim_scope",
                "qra_2024_01,2024Q1,2024 Jan QRA,2024-01-03,qra_release,qrawatch,canonical_shock_bn,10,usd_billions,2024-01-03T08:30:00-05:00,event_close_with_embargo,business_day,true,usable,Tier A,headline",
            ]
        ),
    )
    _write_text(
        paths.config / "debt_limit_intervals.csv",
        "\n".join(
            [
                "quarter,start_date,end_date,source_note",
                "2024Q1,2024-01-01,2024-01-15,test fixture",
            ]
        ),
    )

    for series_id, rows, target_dir in [
        ("SP500", [("2024-01-02", "100"), ("2024-01-04", "110"), ("2024-01-10", "121")], paths.raw_fred),
        ("VIXCLS", [("2024-01-02", "20"), ("2024-01-04", "23"), ("2024-01-10", "25")], paths.raw_fred),
        ("DFF", [("2024-01-02", "5.33"), ("2024-01-04", "5.38"), ("2024-01-10", "5.42")], paths.raw_fred),
        ("SOFR", [("2024-01-02", "5.31"), ("2024-01-04", "5.34"), ("2024-01-10", "5.40")], paths.raw_fred),
        ("DGS2", [("2024-01-02", "4.00"), ("2024-01-04", "4.20"), ("2024-01-10", "4.50")], paths.raw_fred),
        ("DGS10", [("2024-01-02", "3.90"), ("2024-01-04", "4.00"), ("2024-01-10", "4.15")], paths.raw_fred),
        ("DGS3MO", [("2024-01-02", "5.10"), ("2024-01-04", "5.05"), ("2024-01-10", "5.00")], paths.raw_fred),
        ("TGCRRATE", [("2024-01-02", "5.30"), ("2024-01-04", "5.35"), ("2024-01-10", "5.40")], paths.raw_fred),
        ("RRPONTSYAWARD", [("2024-01-02", "5.00"), ("2024-01-04", "5.02"), ("2024-01-10", "5.00")], paths.raw_fred),
        ("THREEFYTP10", [("2024-01-02", "1.60"), ("2024-01-04", "1.66"), ("2024-01-10", "1.70")], paths.raw_fred),
    ]:
        _write_text(
            target_dir / f"{series_id}.csv",
            "\n".join(["date,value", *[f"{observed_date},{value}" for observed_date, value in rows]]),
        )

    for series_id, rows in {
        "WDTGAL": [("2024-01-02", "700"), ("2024-01-04", "720"), ("2024-01-10", "760")],
        "WRESBAL": [("2024-01-02", "3200"), ("2024-01-04", "3215"), ("2024-01-10", "3250")],
        "RRPONTSYD": [("2024-01-02", "900"), ("2024-01-04", "870"), ("2024-01-10", "820")],
        "WALCL": [("2024-01-02", "7800"), ("2024-01-04", "7810"), ("2024-01-10", "7835")],
    }.items():
        _write_text(
            paths.seed / "interpol" / "raw" / f"FRED_{series_id}_{series_id}.csv",
            "\n".join(["date,value", *[f"{observed_date},{value}" for observed_date, value in rows]]),
        )

    result = build_event_design(paths, job_id="qra_event_risk_21bd")

    with result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["delta_sp500_return_h1bd"] == "10.0"
    assert rows[0]["delta_sp500_return_h5bd"] == "21.0"
    assert rows[0]["delta_vix_change_h1bd"] == "3.0"
    assert rows[0]["delta_tga_balance_change_h1bd"] == "20.0"
    assert rows[0]["delta_reserve_balances_change_h5bd"] == "50.0"
    assert rows[0]["delta_rrp_balance_change_h5bd"] == "-80.0"
    assert rows[0]["delta_fed_balance_sheet_change_h5bd"] == "35.0"

    design_manifest = json.loads(result.design_manifest_path.read_text(encoding="utf-8"))
    assert design_manifest["status"] == "ready_for_estimation"
    assert design_manifest["scaling_rule"] == "mixed_catalog_units_with_pct_returns"
    assert design_manifest["missing_required_series"] == []
    assert design_manifest["sample_policy"] == "headline_strict"
    assert design_manifest["control_ids"] == [
        "debt_limit_dummy",
        "delta_dff_release_plus_1bd",
        "delta_dff_release_plus_5bd",
        "delta_sofr_release_plus_1bd",
        "delta_sofr_release_plus_5bd",
        "delta_threefytp10_release_plus_1bd",
        "delta_threefytp10_release_plus_5bd",
        "delta_dgs2_release_plus_1bd",
        "delta_dgs2_release_plus_5bd",
        "delta_dgs10_release_plus_1bd",
        "delta_dgs10_release_plus_5bd",
        "delta_term_spread_10y_3m_release_plus_1bd",
        "delta_term_spread_10y_3m_release_plus_5bd",
        "delta_repo_spread_release_plus_1bd",
        "delta_repo_spread_release_plus_5bd",
    ]


def test_event_design_builder_supports_reviewed_nonmissing_sample_policy(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)

    _write_text(
        paths.config / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: qra_event_rates_reviewed_descriptive",
                "    estimator: event_lp",
                "    freq: irregular_event",
                "    treatment_id: qra_release_63bd",
                "    sample_policy: reviewed_nonmissing",
                "    controls_explicit: [dff]",
                "    outcomes: [dgs2]",
                "    horizons_bd: [1]",
                "    cutoff_rule: event_close_with_embargo",
                "    output_family: supporting_descriptive",
            ]
        ),
    )
    _write_text(
        paths.bundles / "qrawatch" / "event_bundle.csv",
        "\n".join(
            [
                "event_id,quarter,event_label,event_date,event_type,source_repo,treatment_id,treatment_value,treatment_units,cutoff_timestamp,embargo_rule,horizon_unit,usable_for_headline,usable_for_headline_reason,usable_for_descriptive_headline,descriptive_headline_reason,quality_tier,claim_scope,shock_review_status,shock_missing_flag,small_denominator_flag",
                "qra_2024_01,2024Q1,2024 Jan QRA,2024-01-03,qra_release,qrawatch,canonical_shock_bn,10,usd_billions,2024-01-03T08:30:00-05:00,event_close_with_embargo,business_day,true,usable,true,usable,Tier A,descriptive_only,reviewed,false,false",
                "qra_2024_02,2024Q1,2024 Feb QRA,2024-02-07,qra_release,qrawatch,canonical_shock_bn,0.0,usd_billions,2024-02-07T08:30:00-05:00,event_close_with_embargo,business_day,false,small_denominator,false,small_denominator,Tier B,descriptive_only,reviewed,false,true",
            ]
        ),
    )
    _write_text(paths.config / "debt_limit_intervals.csv", "quarter,start_date,end_date,source_note\n")
    _write_text(
        paths.raw_fred / "DGS2.csv",
        "\n".join(
            [
                "date,value",
                "2024-01-02,4.00",
                "2024-01-04,4.10",
                "2024-02-06,4.20",
                "2024-02-08,4.18",
            ]
        ),
    )
    _write_text(
        paths.raw_fred / "DFF.csv",
        "\n".join(
            [
                "date,value",
                "2024-01-02,5.33",
                "2024-01-04,5.35",
                "2024-02-06,5.32",
                "2024-02-08,5.31",
            ]
        ),
    )
    _write_text(
        paths.raw_fred / "SOFR.csv",
        "\n".join(
            [
                "date,value",
                "2024-01-02,5.31",
                "2024-01-04,5.34",
                "2024-02-06,5.30",
                "2024-02-08,5.29",
            ]
        ),
    )

    result = build_event_design(paths, job_id="qra_event_rates_reviewed_descriptive")

    with result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["include_in_sample"] == "true"
    assert rows[0]["sample_bucket"] == "headline_strict"
    assert rows[1]["include_in_sample"] == "true"
    assert rows[1]["sample_bucket"] == "reviewed_zero_shock_small_denom"

    design_manifest = json.loads(result.design_manifest_path.read_text(encoding="utf-8"))
    assert design_manifest["sample_policy"] == "reviewed_nonmissing"
    assert design_manifest["control_selection_policy"] == "explicit"
    assert design_manifest["control_ids"] == ["debt_limit_dummy", "delta_dff_release_plus_1bd"]
    assert design_manifest["event_sample_counts"]["headline_eligible_rows"] == 1
    assert design_manifest["event_sample_counts"]["reviewed_nonmissing_rows"] == 2
    assert design_manifest["event_sample_counts"]["reviewed_zero_shock_rows"] == 1
    assert design_manifest["event_sample_counts"]["requested_sample_rows"] == 2
    assert design_manifest["usable_rows"] == 2

    sample_manifest = json.loads(result.sample_manifest_path.read_text(encoding="utf-8"))
    assert sample_manifest["sample_policy"] == "reviewed_nonmissing"
    assert sample_manifest["rows"][2]["observations_remaining"] == 2


def test_event_design_builder_uses_market_holiday_calendar(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)

    _write_text(
        paths.config / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: qra_event_rates_63bd",
                "    estimator: event_lp",
                "    freq: irregular_event",
                "    treatment_id: qra_release_63bd",
                "    outcomes: [dgs2]",
                "    horizons_bd: [1]",
                "    cutoff_rule: event_close_with_embargo",
                "    output_family: supporting_reduced_form",
            ]
        ),
    )
    _write_text(
        paths.bundles / "qrawatch" / "event_bundle.csv",
        "\n".join(
            [
                "event_id,quarter,event_label,event_date,event_type,source_repo,treatment_id,treatment_value,treatment_units,cutoff_timestamp,embargo_rule,horizon_unit,usable_for_headline,usable_for_headline_reason,quality_tier,claim_scope",
                "qra_2024_03,2024Q1,2024 Mar QRA,2024-03-28,qra_release,qrawatch,canonical_shock_bn,10,usd_billions,2024-03-28T08:30:00-05:00,event_close_with_embargo,business_day,true,usable,Tier A,headline",
            ]
        ),
    )
    _write_text(paths.config / "debt_limit_intervals.csv", "quarter,start_date,end_date,source_note\n")
    _write_text(
        paths.raw_fred / "DGS2.csv",
        "\n".join(
            [
                "date,value",
                "2024-03-27,4.00",
                "2024-04-01,4.10",
            ]
        ),
    )
    _write_text(
        paths.raw_fred / "DFF.csv",
        "\n".join(
            [
                "date,value",
                "2024-03-27,5.33",
                "2024-04-01,5.35",
            ]
        ),
    )

    result = build_event_design(paths, job_id="qra_event_rates_63bd")

    with result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["window_start_date"] == "2024-03-27"
    assert rows[0]["end_date_dgs2_h1bd"] == "2024-04-01"
    assert rows[0]["delta_dgs2_h1bd"] == "10.0"
