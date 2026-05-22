from __future__ import annotations

import csv
import json
from pathlib import Path

from ea_tdc.designs.quarterly import _overlay_missing_quarters, _set_row_linear_combo, build_quarterly_design
from ea_tdc.paths import ensure_repo_dirs, project_paths


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_quarterly_design_builder_writes_manifests(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: baseline_tdc_lp_deposits",
                "    estimator: lp",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: headline_identified",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,100,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,120,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "qrawatch" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "net_bills_bn,net_bills_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,10,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "bill_share,bill_share,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,share,0.1,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "M2SL.csv",
        "\n".join(
            [
                "date,value",
                "2024-01-31,1000",
                "2024-02-29,1005",
                "2024-03-31,1010",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "BOGZ1FL764100005Q.csv",
        "\n".join(
            [
                "date,value",
                "2023-10-01,900",
                "2024-01-01,1030",
            ]
        ),
    )

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = build_quarterly_design(paths, job_id="baseline_tdc_lp_deposits")

    assert result.bundle_path.exists()
    assert result.design_manifest_path.exists()
    assert result.sample_manifest_path.exists()
    assert result.rows_written == 2
    assert result.usable_rows == 1

    with result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["tdc_bank_only_qoq"] == "100"
    assert rows[1]["qra_ati_baseline_bn"] == "5"
    assert rows[1]["M2SL"] == "1010"
    assert rows[1]["matched_total_deposits"] == "130.0"
    assert rows[1]["other_component_qoq"] == "30.0"

    design_manifest = json.loads(result.design_manifest_path.read_text(encoding="utf-8"))
    assert design_manifest["job_id"] == "baseline_tdc_lp_deposits"
    assert design_manifest["missing_required_series"] == []

    sample_manifest = json.loads(result.sample_manifest_path.read_text(encoding="utf-8"))
    assert sample_manifest["rows"][-1]["observations_remaining"] == 1
    assert sample_manifest["rows"][-1]["reason"] == "required outcomes available within usable treatment-anchor sample"


def test_quarterly_design_builder_imports_canonical_tier2_mmf_rrp_rows(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: tdc_tier2_mmf_rrp_canonical_full_panel",
                "    estimator: lp",
                "    treatment_id: tdc_tier2_mmf_rrp_prop_bank_only_qoq",
                "    outcomes:",
                "      - matched_total_deposits",
                "      - other_component_tier2_mmf_rrp_prop_bank_only_qoq",
                "      - domestic_nonbank_deposits_qoq",
                "      - domestic_nonbank_other_component_tier2_mmf_rrp_prop_bank_only_qoq",
                "      - m2",
                "    horizons: [0]",
                "    controls_block: headline_core",
                "    output_family: supporting_descriptive",
            ]
        ),
    )
    header = (
        "series_id,series_label,source_family,source_repo,source_table,freq,period_end,"
        "release_date,available_at,vintage_policy,units,value,transform_default,"
        "seasonal_adjustment_flag,interpolated_flag,component_group,role,notes"
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                header,
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,100,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_tier2_mmf_rrp_prop_bank_only_ru_flow,tdc_tier2_mmf_rrp_prop_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,10000,none,unknown,false,estimate_variant,treatment,fixture",
                "tdc_tier2_mmf_rrp_lb_bank_only_ru_flow,tdc_tier2_mmf_rrp_lb_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,8000,none,unknown,false,estimate_variant,treatment,fixture",
                "tdc_tier2_mmf_rrp_ub_bank_only_ru_flow,tdc_tier2_mmf_rrp_ub_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,12000,none,unknown,false,estimate_variant,treatment,fixture",
                "tdc_tier2_mmf_rrp_prop_depository_institution_np_cu_ru_flow,tdc_tier2_mmf_rrp_prop_depository_institution_np_cu_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,9000,none,unknown,false,estimate_variant,treatment,fixture",
                "tdc_tier2_treasury_interest_robust_mmf_rrp_prop_bank_only_ru_flow,tdc_tier2_treasury_interest_robust_mmf_rrp_prop_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,7000,none,unknown,false,estimate_variant,treatment,fixture",
                "tdc_tier2_canonical_depository_institution_mmf_rrp_prop_ru_flow,tdc_tier2_canonical_depository_institution_mmf_rrp_prop_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,6500,none,unknown,false,estimate_variant,treatment,fixture",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,120,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tdcpass" / "standardized_series.csv",
        "\n".join(
            [
                header,
                "tdcpass_domestic_nonbank_deposits_qoq,domestic_nonbank_deposits_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-03-31,2024-06-29,2024-06-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,40,none,unknown,false,published_panel,mechanism,fixture",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "BOGZ1FL764100005Q.csv",
        "\n".join(["date,value", "2023-10-01,900", "2024-01-01,1030"]),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "M2SL.csv",
        "\n".join(["date,value", "2024-01-31,1000", "2024-02-29,1005", "2024-03-31,1010"]),
    )

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = build_quarterly_design(paths, job_id="tdc_tier2_mmf_rrp_canonical_full_panel")

    with result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = next(item for item in rows if item["quarter"] == "2024Q1")
    assert row["tdc_tier2_mmf_rrp_prop_bank_only_qoq"] == "10000"
    assert row["other_component_tier2_mmf_rrp_prop_bank_only_qoq"] == "-9870.0"
    assert row["domestic_nonbank_other_component_tier2_mmf_rrp_prop_bank_only_qoq"] == "30000.0"
    assert row["tdc_tier2_mmf_rrp_prop_bank_only_qoq__source_repo"] == "tdcest"
    assert row["tdc_tier2_canonical_di_mmf_rrp_prop_qoq"] == "6500"

    sample_manifest = json.loads(result.sample_manifest_path.read_text(encoding="utf-8"))
    assert sample_manifest["rows"][1]["observations_remaining"] == 1


def test_quarterly_design_builder_supports_corrected_treatment_residuals_and_identity_gaps(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: custom_tier3_identity_job",
                "    estimator: lp",
                "    treatment_id: tdc_tier3_fiscal_corrected_bank_only_ru_flow",
                "    outcomes:",
                "      - other_component_tier3_bank_only_qoq",
                "      - accounting_identity_total_qoq",
                "      - accounting_identity_gap_tier3_bank_only_qoq",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: supporting_descriptive",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_tier3_fiscal_corrected_bank_only_ru_flow,tdc_tier3_fiscal_corrected_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,95,none,unknown,false,estimate_variant,treatment,fixture",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,120,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "accounting" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "accounting_deposit_substitution_qoq,accounting_deposit_substitution_qoq,repo_seed_bundle,accounting,standardized,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,2,none,unknown,false,component,outcome,fixture",
                "accounting_bank_balance_sheet_qoq,accounting_bank_balance_sheet_qoq,repo_seed_bundle,accounting,standardized,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,3,none,unknown,false,component,outcome,fixture",
                "accounting_public_liquidity_qoq,accounting_public_liquidity_qoq,repo_seed_bundle,accounting,standardized,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,4,none,unknown,false,component,outcome,fixture",
                "accounting_external_flow_qoq,accounting_external_flow_qoq,repo_seed_bundle,accounting,standardized,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,1,none,unknown,false,component,outcome,fixture",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "BOGZ1FL764100005Q.csv",
        "\n".join(
            [
                "date,value",
                "2023-10-01,900",
                "2024-01-01,1010",
            ]
        ),
    )

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = build_quarterly_design(paths, job_id="custom_tier3_identity_job")

    with result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = rows[1]
    assert row["other_component_tier3_bank_only_qoq"] == "15.0"
    assert row["accounting_identity_total_qoq"] == "10.0"
    assert row["accounting_identity_gap_tier3_bank_only_qoq"] == "5.0"

    design_manifest = json.loads(result.design_manifest_path.read_text(encoding="utf-8"))
    assert design_manifest["treatment_id"] == "tdc_tier3_fiscal_corrected_bank_only_ru_flow"
    assert design_manifest["missing_required_series"] == []
    assert design_manifest["status"] == "ready_for_estimation"

    sample_manifest = json.loads(result.sample_manifest_path.read_text(encoding="utf-8"))
    assert sample_manifest["rows"][-1]["observations_remaining"] == 1
    assert sample_manifest["rows"][-1]["reason"] == "required outcomes available within usable treatment sample"


def test_quarterly_design_builder_supports_corrected_tier3_identity_pct_gdp_outputs(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: custom_tier3_identity_pct_gdp_job",
                "    estimator: lp",
                "    treatment_id: tdc_tier3_fiscal_corrected_bank_only_ru_flow",
                "    outcomes:",
                "      - other_component_tier3_bank_only_qoq_pct_gdp",
                "      - accounting_identity_total_qoq_pct_gdp",
                "      - accounting_identity_gap_tier3_bank_only_qoq_pct_gdp",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: supporting_descriptive",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_tier3_fiscal_corrected_bank_only_ru_flow,tdc_tier3_fiscal_corrected_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,85,none,unknown,false,measurement_variant,treatment,fixture",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,120,none,unknown,false,reference,control,fixture",
                "accounting_deposit_substitution_qoq,accounting_deposit_substitution_qoq,repo_seed_bundle,accounting,identity,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,4,none,unknown,false,identity_component,outcome,fixture",
                "accounting_bank_balance_sheet_qoq,accounting_bank_balance_sheet_qoq,repo_seed_bundle,accounting,identity,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,3,none,unknown,false,identity_component,outcome,fixture",
                "accounting_public_liquidity_qoq,accounting_public_liquidity_qoq,repo_seed_bundle,accounting,identity,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,2,none,unknown,false,identity_component,outcome,fixture",
                "accounting_external_flow_qoq,accounting_external_flow_qoq,repo_seed_bundle,accounting,identity,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,1,none,unknown,false,identity_component,outcome,fixture",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "BOGZ1FL764100005Q.csv",
        "\n".join(
            [
                "date,value",
                "2023-10-01,900",
                "2024-01-01,1000",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "GDP.csv",
        "\n".join(
            [
                "date,value",
                "2024-01-01,10000",
            ]
        ),
    )

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = build_quarterly_design(paths, job_id="custom_tier3_identity_pct_gdp_job")

    with result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = rows[1]
    assert row["other_component_tier3_bank_only_qoq_pct_gdp"] == "0.15"
    assert row["accounting_identity_total_qoq_pct_gdp"] == "0.1"
    assert row["accounting_identity_gap_tier3_bank_only_qoq_pct_gdp"] == "0.05"

    design_manifest = json.loads(result.design_manifest_path.read_text(encoding="utf-8"))
    assert design_manifest["missing_required_series"] == []
    assert design_manifest["status"] == "ready_for_estimation"


def test_quarterly_design_builder_maps_component_treatment_ids_to_upstream_series(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: custom_component_job",
                "    estimator: lp",
                "    treatment_id: tdc_positive_remit_component_qoq",
                "    outcomes: [matched_total_deposits]",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: supporting_descriptive",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "fed_remit_positive,fed_remit_positive,repo_seed_bundle,tdcest,components,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,5,none,unknown,false,component,treatment,fixture",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "BOGZ1FL764100005Q.csv",
        "\n".join(
            [
                "date,value",
                "2023-10-01,900",
                "2024-01-01,1010",
            ]
        ),
    )

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = build_quarterly_design(paths, job_id="custom_component_job")

    design_manifest = json.loads(result.design_manifest_path.read_text(encoding="utf-8"))
    assert design_manifest["treatment_id"] == "fed_remit_positive"
    assert design_manifest["status"] == "ready_for_estimation"
    assert result.usable_rows == 1


def test_quarterly_design_builder_supports_funding_and_credit_outcomes(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: baseline_tdc_lp_funding",
                "    estimator: lp",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: headline_identified",
                "  - job_id: baseline_tdc_lp_credit_spreads",
                "    estimator: lp",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: headline_identified",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,100,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,120,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "qrawatch" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "net_bills_bn,net_bills_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,10,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "bill_share,bill_share,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,share,0.1,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "WRESBAL.csv",
        "\n".join(
            [
                "date,value",
                "2024-01-03,3200",
                "2024-03-27,3300",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "TGCRRATE.csv",
        "\n".join(
            [
                "date,value",
                "2024-01-02,5.30",
                "2024-02-01,5.32",
                "2024-03-01,5.34",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "RRPONTSYAWARD.csv",
        "\n".join(
            [
                "date,value",
                "2024-01-02,5.20",
                "2024-02-01,5.21",
                "2024-03-01,5.22",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "FEDFUNDS.csv",
        "\n".join(
            [
                "date,value",
                "2024-01-01,5.33",
                "2024-02-01,5.33",
                "2024-03-01,5.33",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "SOFR.csv",
        "\n".join(
            [
                "date,value",
                "2024-01-02,5.31",
                "2024-02-01,5.32",
                "2024-03-01,5.33",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "BAA.csv",
        "\n".join(
            [
                "date,value",
                "2024-01-01,6.00",
                "2024-02-01,6.10",
                "2024-03-01,6.20",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "AAA.csv",
        "\n".join(
            [
                "date,value",
                "2024-01-01,5.00",
                "2024-02-01,5.10",
                "2024-03-01,5.20",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "BAMLC0A0CM.csv",
        "\n".join(
            [
                "date,value",
                "2024-01-01,1.10",
                "2024-02-01,1.20",
                "2024-03-01,1.30",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "BAMLH0A0HYM2.csv",
        "\n".join(
            [
                "date,value",
                "2024-01-01,3.10",
                "2024-02-01,3.20",
                "2024-03-01,3.30",
            ]
        ),
    )

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)

    funding_result = build_quarterly_design(paths, job_id="baseline_tdc_lp_funding")
    credit_result = build_quarterly_design(paths, job_id="baseline_tdc_lp_credit_spreads")

    with funding_result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        funding_rows = list(csv.DictReader(handle))
    funding_row = funding_rows[0]
    assert funding_row["reserve_balances"] == "3300"
    assert funding_row["repo_spread"] == "11.0"
    assert funding_row["FEDFUNDS"] == "5.33"
    assert funding_row["SOFR"] == "5.32"

    funding_manifest = json.loads(funding_result.design_manifest_path.read_text(encoding="utf-8"))
    assert funding_manifest["missing_required_series"] == []

    with credit_result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        credit_rows = list(csv.DictReader(handle))
    credit_row = credit_rows[0]
    assert credit_row["baa_aaa"] == "100.0"
    assert credit_row["investment_grade_oas"] == "1.2"
    assert credit_row["high_yield_oas"] == "3.2"

    credit_manifest = json.loads(credit_result.design_manifest_path.read_text(encoding="utf-8"))
    assert credit_manifest["missing_required_series"] == []


def test_quarterly_design_builder_supports_inflation_fx_private_assets_and_liquidity_decomp(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: baseline_tdc_lp_inflation",
                "    estimator: lp",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: supporting_descriptive",
                "  - job_id: baseline_tdc_lp_fx",
                "    estimator: lp",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: supporting_descriptive",
                "  - job_id: baseline_tdc_lp_private_assets",
                "    estimator: lp",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: supporting_descriptive",
                "  - job_id: baseline_tdc_lp_liquidity_decomposition",
                "    estimator: lp",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: supporting_descriptive",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2023-12-31,2024-03-30,2024-03-30,seed_bundle_snapshot_conservative_90d_lag,usd_millions,80,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,100,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,120,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "qrawatch" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2023-12-31,2023-11-01,2023-11-01T08:30:00-05:00,official_qra_release_timestamp,usd_billions,4,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "net_bills_bn,net_bills_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,10,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "bill_share,bill_share,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,share,0.1,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
            ]
        ),
    )
    for name, lines in {
        "CPIAUCSL.csv": ["date,value", "2023-10-01,100", "2023-11-01,101", "2023-12-01,102", "2024-01-01,103", "2024-02-01,104", "2024-03-01,105"],
        "CPILFESL.csv": ["date,value", "2023-10-01,200", "2023-11-01,201", "2023-12-01,202", "2024-01-01,204", "2024-02-01,205", "2024-03-01,206"],
        "PCEPILFE.csv": ["date,value", "2023-10-01,50", "2023-11-01,50.5", "2023-12-01,51", "2024-01-01,52", "2024-02-01,52.5", "2024-03-01,53"],
        "DTWEXBGS.csv": ["date,value", "2023-10-02,100", "2023-11-01,101", "2023-12-01,102", "2024-01-02,103", "2024-02-01,104", "2024-03-01,105"],
        "TOTBKCR.csv": ["date,value", "2023-12-01,500", "2024-03-01,530"],
        "BUSLOANS.csv": ["date,value", "2023-12-01,120", "2024-03-01,129"],
        "TOTCI.csv": ["date,value", "2023-12-27,98", "2024-03-27,109"],
        "BOGZ1FL704041005Q.csv": ["date,value", "2023-10-01,40", "2024-01-01,52"],
        "LTDACBM027NBOG.csv": ["date,value", "2023-12-01,200", "2024-03-01,218"],
        "RMFSL.csv": ["date,value", "2023-12-01,300", "2024-03-01,306"],
        "WIMFSL.csv": ["date,value", "2023-12-27,500", "2024-03-27,490"],
        "OSEACBW027SBOG.csv": ["date,value", "2023-12-01,50", "2024-03-01,55"],
        "CLSACBW027SBOG.csv": ["date,value", "2023-12-01,70", "2024-03-01,78"],
        "RELACBW027SBOG.csv": ["date,value", "2023-12-01,80", "2024-03-01,92"],
        "BOGZ1FL264035005Q.csv": ["date,value", "2023-10-01,900", "2024-01-01,915"],
        "WRESBAL.csv": ["date,value", "2023-10-04,1000", "2024-01-03,1035"],
        "WDFOL.csv": ["date,value", "2023-10-04,40", "2024-01-03,45"],
        "WALCL.csv": ["date,value", "2023-10-04,7900", "2024-01-03,7925"],
        "TREAST.csv": ["date,value", "2023-10-04,4800", "2024-01-03,4810"],
    }.items():
        _write_text(tmp_path / "data" / "raw" / "fred" / name, "\n".join(lines))

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)

    inflation_result = build_quarterly_design(paths, job_id="baseline_tdc_lp_inflation")
    fx_result = build_quarterly_design(paths, job_id="baseline_tdc_lp_fx")
    private_assets_result = build_quarterly_design(paths, job_id="baseline_tdc_lp_private_assets")
    liquidity_result = build_quarterly_design(paths, job_id="baseline_tdc_lp_liquidity_decomposition")

    with inflation_result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        inflation_rows = list(csv.DictReader(handle))
    inflation_row = inflation_rows[1]
    assert inflation_row["headline_cpi_inflation_qoq_ann"] != ""
    assert inflation_row["core_cpi_inflation_qoq_ann"] != ""
    assert inflation_row["core_pce_inflation_qoq_ann"] != ""

    with fx_result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        fx_rows = list(csv.DictReader(handle))
    assert fx_rows[1]["broad_dollar_change"] == "3.0"

    with private_assets_result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        private_rows = list(csv.DictReader(handle))
    private_row = private_rows[1]
    assert private_row["bank_credit_qoq"] == "30.0"
    assert private_row["bank_business_loans_qoq"] == "9.0"
    assert private_row["bank_non_treasury_securities_qoq"] == "5.0"
    assert private_row["bank_consumer_loans_qoq"] == "8.0"
    assert private_row["bank_real_estate_loans_qoq"] == "12.0"
    assert private_row["row_loans_assets_qoq"] == "15.0"

    with liquidity_result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        liquidity_rows = list(csv.DictReader(handle))
    liquidity_row = liquidity_rows[1]
    assert liquidity_row["reserve_balances_qoq"] == "35.0"
    assert liquidity_row["foreign_official_deposits_qoq"] == "5.0"
    assert liquidity_row["total_reserve_balances_plus_foreign_official_qoq"] == "40.0"
    assert liquidity_row["fed_total_assets_qoq"] == "25.0"
    assert liquidity_row["fed_treasury_holdings_qoq"] == "10.0"
    assert liquidity_row["reserve_balances_net_fed_assets_qoq"] == "10.0"
    assert liquidity_row["reserve_balances_net_fed_treasury_qoq"] == "25.0"
    assert liquidity_row["total_reserves_plus_foreign_official_net_fed_treasury_qoq"] == "30.0"


def test_quarterly_design_builder_supports_deposit_source_decomposition_job(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: baseline_tdc_lp_deposit_sources",
                "    estimator: lp",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: supporting_descriptive",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2023-12-31,2024-03-30,2024-03-30,seed_bundle_snapshot_conservative_90d_lag,usd_millions,80,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,100,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,120,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    for name, lines in {
        "BOGZ1FL764100005Q.csv": ["date,value", "2023-10-01,900", "2024-01-01,1030"],
        "TOTBKCR.csv": ["date,value", "2023-12-01,500", "2024-03-01,530"],
        "BUSLOANS.csv": ["date,value", "2023-12-01,120", "2024-03-01,129"],
        "TOTCI.csv": ["date,value", "2023-12-27,98", "2024-03-27,109"],
        "BOGZ1FL704041005Q.csv": ["date,value", "2023-10-01,40", "2024-01-01,52"],
        "LTDACBM027NBOG.csv": ["date,value", "2023-12-01,200", "2024-03-01,218"],
        "RMFSL.csv": ["date,value", "2023-12-01,300", "2024-03-01,306"],
        "WIMFSL.csv": ["date,value", "2023-12-27,500", "2024-03-27,490"],
        "OSEACBW027SBOG.csv": ["date,value", "2023-12-01,50", "2024-03-01,55"],
        "CLSACBW027SBOG.csv": ["date,value", "2023-12-01,70", "2024-03-01,78"],
        "RELACBW027SBOG.csv": ["date,value", "2023-12-01,80", "2024-03-01,92"],
        "BOGZ1FL264035005Q.csv": ["date,value", "2023-10-01,900", "2024-01-01,915"],
        "ROWCBAQ027S.csv": ["date,value", "2023-10-01,20", "2024-01-01,35"],
        "ROWCEAQ027S.csv": ["date,value", "2023-10-01,10", "2024-01-01,14"],
        "ROWGSEQ027S.csv": ["date,value", "2023-10-01,5", "2024-01-01,7"],
        "ROWNBLQ027S.csv": ["date,value", "2023-10-01,8", "2024-01-01,9"],
        "EXPGS.csv": ["date,value", "2023-10-01,2500", "2024-01-01,2540"],
        "IMPGS.csv": ["date,value", "2023-10-01,3100", "2024-01-01,3145"],
        "BOPBCA.csv": ["date,value", "2023-10-01,-100", "2024-01-01,-140"],
        "WDTGAL.csv": ["date,value", "2023-10-02,700", "2024-01-02,760"],
        "RRPONTSYD.csv": ["date,value", "2023-10-02,80", "2024-01-02,10"],
    }.items():
        _write_text(tmp_path / "data" / "raw" / "fred" / name, "\n".join(lines))

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = build_quarterly_design(paths, job_id="baseline_tdc_lp_deposit_sources")

    with result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = rows[1]
    assert row["matched_total_deposits"] == "130.0"
    assert row["other_component_qoq"] == "30.0"
    assert row["large_time_deposits_qoq"] == "18.0"
    assert row["retail_mmf_assets_qoq"] == "6.0"
    assert row["institutional_mmf_assets_qoq"] == "-10.0"
    assert row["bank_credit_qoq"] == "30.0"
    assert row["bank_business_loans_qoq"] == "9.0"
    assert row["bank_ci_loans_h8_qoq"] == "11.0"
    assert row["bank_short_term_loans_z1_qoq"] == "12.0"
    assert row["bank_non_treasury_securities_qoq"] == "5.0"
    assert row["bank_consumer_loans_qoq"] == "8.0"
    assert row["bank_real_estate_loans_qoq"] == "12.0"
    assert row["row_loans_assets_qoq"] == "15.0"
    assert row["row_corp_bonds_flow"] == "35"
    assert row["row_private_flow_block"] == "65.0"
    assert row["exports_qoq"] == "40.0"
    assert row["imports_qoq"] == "45.0"
    assert row["net_exports_qoq"] == "-5.0"
    assert row["current_account_balance"] == "-140"
    assert row["tga_balance_qoq"] == "60.0"
    assert row["on_rrp_balance_qoq"] == "-70.0"

    design_manifest = json.loads(result.design_manifest_path.read_text(encoding="utf-8"))
    assert design_manifest["missing_required_series"] == []


def test_quarterly_design_builder_supports_pct_gdp_deposit_source_job(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: baseline_tdc_lp_deposit_sources_pct_gdp",
                "    estimator: lp",
                "    outcomes:",
                "      - matched_total_deposits_pct_gdp",
                "      - other_component_qoq_pct_gdp",
                "      - large_time_deposits_qoq_pct_gdp",
                "      - retail_mmf_assets_qoq_pct_gdp",
                "      - institutional_mmf_assets_qoq_pct_gdp",
                "      - bank_credit_qoq_pct_gdp",
                "      - bank_business_loans_qoq_pct_gdp",
                "      - bank_ci_loans_h8_qoq_pct_gdp",
                "      - bank_short_term_loans_z1_qoq_pct_gdp",
                "      - bank_non_treasury_securities_qoq_pct_gdp",
                "      - bank_consumer_loans_qoq_pct_gdp",
                "      - bank_real_estate_loans_qoq_pct_gdp",
                "      - row_loans_assets_qoq_pct_gdp",
                "      - row_corp_bonds_flow_pct_gdp",
                "      - row_private_flow_block_pct_gdp",
                "      - exports_qoq_pct_gdp",
                "      - imports_qoq_pct_gdp",
                "      - net_exports_qoq_pct_gdp",
                "      - current_account_balance_pct_gdp",
                "      - tga_balance_qoq_pct_gdp",
                "      - on_rrp_balance_qoq_pct_gdp",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: supporting_descriptive",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2023-12-31,2024-03-30,2024-03-30,seed_bundle_snapshot_conservative_90d_lag,usd_millions,80,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,100,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,120,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    for name, lines in {
        "GDP.csv": ["date,value", "2023-12-31,20000", "2024-03-31,21000"],
        "BOGZ1FL764100005Q.csv": ["date,value", "2023-10-01,900", "2024-01-01,1030"],
        "TOTBKCR.csv": ["date,value", "2023-12-01,500", "2024-03-01,530"],
        "BUSLOANS.csv": ["date,value", "2023-12-01,120", "2024-03-01,129"],
        "TOTCI.csv": ["date,value", "2023-12-27,98", "2024-03-27,109"],
        "BOGZ1FL704041005Q.csv": ["date,value", "2023-10-01,40", "2024-01-01,52"],
        "LTDACBM027NBOG.csv": ["date,value", "2023-12-01,200", "2024-03-01,218"],
        "RMFSL.csv": ["date,value", "2023-12-01,300", "2024-03-01,306"],
        "WIMFSL.csv": ["date,value", "2023-12-27,500", "2024-03-27,490"],
        "OSEACBW027SBOG.csv": ["date,value", "2023-12-01,50", "2024-03-01,55"],
        "CLSACBW027SBOG.csv": ["date,value", "2023-12-01,70", "2024-03-01,78"],
        "RELACBW027SBOG.csv": ["date,value", "2023-12-01,80", "2024-03-01,92"],
        "BOGZ1FL264035005Q.csv": ["date,value", "2023-10-01,900", "2024-01-01,915"],
        "ROWCBAQ027S.csv": ["date,value", "2023-10-01,20", "2024-01-01,35"],
        "ROWCEAQ027S.csv": ["date,value", "2023-10-01,10", "2024-01-01,14"],
        "ROWGSEQ027S.csv": ["date,value", "2023-10-01,5", "2024-01-01,7"],
        "ROWNBLQ027S.csv": ["date,value", "2023-10-01,8", "2024-01-01,9"],
        "EXPGS.csv": ["date,value", "2023-10-01,2500", "2024-01-01,2540"],
        "IMPGS.csv": ["date,value", "2023-10-01,3100", "2024-01-01,3145"],
        "BOPBCA.csv": ["date,value", "2023-10-01,-100", "2024-01-01,-140"],
        "WDTGAL.csv": ["date,value", "2023-10-02,700", "2024-01-02,760"],
        "RRPONTSYD.csv": ["date,value", "2023-10-02,80", "2024-01-02,10"],
    }.items():
        _write_text(tmp_path / "data" / "raw" / "fred" / name, "\n".join(lines))

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = build_quarterly_design(paths, job_id="baseline_tdc_lp_deposit_sources_pct_gdp")

    with result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = rows[1]
    assert row["matched_total_deposits_pct_gdp"] == "0.619047619"
    assert row["other_component_qoq_pct_gdp"] == "0.1428571429"
    assert row["large_time_deposits_qoq_pct_gdp"] == "0.0857142857"
    assert row["retail_mmf_assets_qoq_pct_gdp"] == "0.0285714286"
    assert row["institutional_mmf_assets_qoq_pct_gdp"] == "-0.0476190476"
    assert row["bank_credit_qoq_pct_gdp"] == "0.1428571429"
    assert row["bank_business_loans_qoq_pct_gdp"] == "0.0428571429"
    assert row["bank_ci_loans_h8_qoq_pct_gdp"] == "0.0523809524"
    assert row["bank_short_term_loans_z1_qoq_pct_gdp"] == "0.0571428571"
    assert row["bank_non_treasury_securities_qoq_pct_gdp"] == "0.0238095238"
    assert row["bank_consumer_loans_qoq_pct_gdp"] == "0.0380952381"
    assert row["bank_real_estate_loans_qoq_pct_gdp"] == "0.0571428571"
    assert row["row_loans_assets_qoq_pct_gdp"] == "0.0714285714"
    assert row["row_corp_bonds_flow_pct_gdp"] == "0.1666666667"
    assert row["row_private_flow_block_pct_gdp"] == "0.3095238095"
    assert row["exports_qoq_pct_gdp"] == "0.1904761905"
    assert row["imports_qoq_pct_gdp"] == "0.2142857143"
    assert row["net_exports_qoq_pct_gdp"] == "-0.0238095238"
    assert row["current_account_balance_pct_gdp"] == "-0.6666666667"
    assert row["tga_balance_qoq_pct_gdp"] == "0.2857142857"
    assert row["on_rrp_balance_qoq_pct_gdp"] == "-0.3333333333"

    design_manifest = json.loads(result.design_manifest_path.read_text(encoding="utf-8"))
    assert design_manifest["missing_required_series"] == []
    assert design_manifest["scaling_rule"] == "qoq_change_as_pct_of_nominal_gdp"


def test_quarterly_design_builder_supports_deposit_source_block_job(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: baseline_tdc_lp_deposit_source_blocks",
                "    estimator: lp",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: supporting_descriptive",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2023-12-31,2024-03-30,2024-03-30,seed_bundle_snapshot_conservative_90d_lag,usd_millions,80,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,100,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,120,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    for name, lines in {
        "BOGZ1FL764100005Q.csv": ["date,value", "2023-10-01,900", "2024-01-01,1030"],
        "TOTCI.csv": ["date,value", "2023-12-27,98", "2024-03-27,109"],
        "BOGZ1FL704041005Q.csv": ["date,value", "2023-10-01,40", "2024-01-01,52"],
        "LTDACBM027NBOG.csv": ["date,value", "2023-12-01,200", "2024-03-01,218"],
        "RMFSL.csv": ["date,value", "2023-12-01,300", "2024-03-01,306"],
        "WIMFSL.csv": ["date,value", "2023-12-27,500", "2024-03-27,490"],
        "OSEACBW027SBOG.csv": ["date,value", "2023-12-01,50", "2024-03-01,55"],
        "ROWCBAQ027S.csv": ["date,value", "2023-10-01,20", "2024-01-01,35"],
        "ROWCEAQ027S.csv": ["date,value", "2023-10-01,10", "2024-01-01,14"],
        "ROWGSEQ027S.csv": ["date,value", "2023-10-01,5", "2024-01-01,7"],
        "ROWNBLQ027S.csv": ["date,value", "2023-10-01,8", "2024-01-01,9"],
        "EXPGS.csv": ["date,value", "2023-10-01,2500", "2024-01-01,2540"],
        "IMPGS.csv": ["date,value", "2023-10-01,3100", "2024-01-01,3145"],
        "WDTGAL.csv": ["date,value", "2023-10-02,700", "2024-01-02,760"],
        "RRPONTSYD.csv": ["date,value", "2023-10-02,80", "2024-01-02,10"],
    }.items():
        _write_text(tmp_path / "data" / "raw" / "fred" / name, "\n".join(lines))

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = build_quarterly_design(paths, job_id="baseline_tdc_lp_deposit_source_blocks")

    with result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = rows[1]
    assert row["other_component_qoq"] == "30.0"
    assert row["deposit_substitution_block_qoq"] == "22.0"
    assert row["bank_balance_sheet_proxy_block_qoq"] == "17.0"
    assert row["public_liquidity_proxy_block_qoq"] == "10.0"
    assert row["external_flow_proxy_block_qoq"] == "60.0"
    assert row["proxy_accounting_total_qoq"] == "109.0"
    assert row["proxy_unexplained_gap_qoq"] == "-79.0"


def test_quarterly_design_builder_supports_pct_gdp_deposit_source_block_job(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: baseline_tdc_lp_deposit_source_blocks_pct_gdp",
                "    estimator: lp",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: supporting_descriptive",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2023-12-31,2024-03-30,2024-03-30,seed_bundle_snapshot_conservative_90d_lag,usd_millions,80,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,100,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,120,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    for name, lines in {
        "GDP.csv": ["date,value", "2023-12-31,20000", "2024-03-31,21000"],
        "BOGZ1FL764100005Q.csv": ["date,value", "2023-10-01,900", "2024-01-01,1030"],
        "TOTCI.csv": ["date,value", "2023-12-27,98", "2024-03-27,109"],
        "BOGZ1FL704041005Q.csv": ["date,value", "2023-10-01,40", "2024-01-01,52"],
        "LTDACBM027NBOG.csv": ["date,value", "2023-12-01,200", "2024-03-01,218"],
        "RMFSL.csv": ["date,value", "2023-12-01,300", "2024-03-01,306"],
        "WIMFSL.csv": ["date,value", "2023-12-27,500", "2024-03-27,490"],
        "OSEACBW027SBOG.csv": ["date,value", "2023-12-01,50", "2024-03-01,55"],
        "ROWCBAQ027S.csv": ["date,value", "2023-10-01,20", "2024-01-01,35"],
        "ROWCEAQ027S.csv": ["date,value", "2023-10-01,10", "2024-01-01,14"],
        "ROWGSEQ027S.csv": ["date,value", "2023-10-01,5", "2024-01-01,7"],
        "ROWNBLQ027S.csv": ["date,value", "2023-10-01,8", "2024-01-01,9"],
        "EXPGS.csv": ["date,value", "2023-10-01,2500", "2024-01-01,2540"],
        "IMPGS.csv": ["date,value", "2023-10-01,3100", "2024-01-01,3145"],
        "WDTGAL.csv": ["date,value", "2023-10-02,700", "2024-01-02,760"],
        "RRPONTSYD.csv": ["date,value", "2023-10-02,80", "2024-01-02,10"],
    }.items():
        _write_text(tmp_path / "data" / "raw" / "fred" / name, "\n".join(lines))

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = build_quarterly_design(paths, job_id="baseline_tdc_lp_deposit_source_blocks_pct_gdp")

    with result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = rows[1]
    assert row["other_component_qoq_pct_gdp"] == "0.1428571429"
    assert row["deposit_substitution_block_qoq_pct_gdp"] == "0.1047619048"
    assert row["bank_balance_sheet_proxy_block_qoq_pct_gdp"] == "0.080952381"
    assert row["public_liquidity_proxy_block_qoq_pct_gdp"] == "0.0476190476"
    assert row["external_flow_proxy_block_qoq_pct_gdp"] == "0.2857142857"
    assert row["proxy_accounting_total_qoq_pct_gdp"] == "0.519047619"
    assert row["proxy_unexplained_gap_qoq_pct_gdp"] == "-0.3761904762"

    design_manifest = json.loads(result.design_manifest_path.read_text(encoding="utf-8"))
    assert design_manifest["scaling_rule"] == "qoq_change_as_pct_of_nominal_gdp"


def test_quarterly_design_builder_supports_accounting_identity_job(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: baseline_tdc_lp_deposit_source_identity",
                "    estimator: lp",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: supporting_descriptive",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2023-12-31,2024-03-30,2024-03-30,seed_bundle_snapshot_conservative_90d_lag,usd_millions,80,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,100,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,120,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "BOGZ1FL764100005Q.csv",
        "\n".join(["date,value", "2023-10-01,900", "2024-01-01,1030"]),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "accounting" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "accounting_deposit_substitution_qoq,Accounting deposit substitution,repo_seed_bundle,accounting,quarterly_identity_flows,quarterly,2024-03-31,2024-06-29,2024-06-29,repo_local_accounting_input,usd_billions,10,none,unknown,false,identity_accounting,mechanism,fixture",
                "accounting_bank_balance_sheet_qoq,Accounting bank balance-sheet channel,repo_seed_bundle,accounting,quarterly_identity_flows,quarterly,2024-03-31,2024-06-29,2024-06-29,repo_local_accounting_input,usd_billions,-5,none,unknown,false,identity_accounting,mechanism,fixture",
                "accounting_public_liquidity_qoq,Accounting public-liquidity channel,repo_seed_bundle,accounting,quarterly_identity_flows,quarterly,2024-03-31,2024-06-29,2024-06-29,repo_local_accounting_input,usd_billions,3,none,unknown,false,identity_accounting,mechanism,fixture",
                "accounting_external_flow_qoq,Accounting external-flow channel,repo_seed_bundle,accounting,quarterly_identity_flows,quarterly,2024-03-31,2024-06-29,2024-06-29,repo_local_accounting_input,usd_billions,2,none,unknown,false,identity_accounting,mechanism,fixture",
            ]
        ),
    )

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = build_quarterly_design(paths, job_id="baseline_tdc_lp_deposit_source_identity")

    with result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = rows[1]
    assert row["other_component_qoq"] == "30.0"
    assert row["accounting_deposit_substitution_qoq"] == "10"
    assert row["accounting_bank_balance_sheet_qoq"] == "-5"
    assert row["accounting_public_liquidity_qoq"] == "3"
    assert row["accounting_external_flow_qoq"] == "2"
    assert row["accounting_identity_total_qoq"] == "10.0"
    assert row["accounting_identity_gap_qoq"] == "20.0"


def test_quarterly_design_builder_supports_pct_gdp_accounting_identity_job(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: baseline_tdc_lp_deposit_source_identity_pct_gdp",
                "    estimator: lp",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: supporting_descriptive",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2023-12-31,2024-03-30,2024-03-30,seed_bundle_snapshot_conservative_90d_lag,usd_millions,80,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,100,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,120,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    _write_text(tmp_path / "data" / "raw" / "fred" / "GDP.csv", "\n".join(["date,value", "2023-12-31,20000", "2024-03-31,21000"]))
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "BOGZ1FL764100005Q.csv",
        "\n".join(["date,value", "2023-10-01,900", "2024-01-01,1030"]),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "accounting" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "accounting_deposit_substitution_qoq,Accounting deposit substitution,repo_seed_bundle,accounting,quarterly_identity_flows,quarterly,2024-03-31,2024-06-29,2024-06-29,repo_local_accounting_input,usd_billions,10,none,unknown,false,identity_accounting,mechanism,fixture",
                "accounting_bank_balance_sheet_qoq,Accounting bank balance-sheet channel,repo_seed_bundle,accounting,quarterly_identity_flows,quarterly,2024-03-31,2024-06-29,2024-06-29,repo_local_accounting_input,usd_billions,-5,none,unknown,false,identity_accounting,mechanism,fixture",
                "accounting_public_liquidity_qoq,Accounting public-liquidity channel,repo_seed_bundle,accounting,quarterly_identity_flows,quarterly,2024-03-31,2024-06-29,2024-06-29,repo_local_accounting_input,usd_billions,3,none,unknown,false,identity_accounting,mechanism,fixture",
                "accounting_external_flow_qoq,Accounting external-flow channel,repo_seed_bundle,accounting,quarterly_identity_flows,quarterly,2024-03-31,2024-06-29,2024-06-29,repo_local_accounting_input,usd_billions,2,none,unknown,false,identity_accounting,mechanism,fixture",
            ]
        ),
    )

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = build_quarterly_design(paths, job_id="baseline_tdc_lp_deposit_source_identity_pct_gdp")

    with result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = rows[1]
    assert row["other_component_qoq_pct_gdp"] == "0.1428571429"
    assert row["accounting_deposit_substitution_qoq_pct_gdp"] == "0.0476190476"
    assert row["accounting_bank_balance_sheet_qoq_pct_gdp"] == "-0.0238095238"
    assert row["accounting_public_liquidity_qoq_pct_gdp"] == "0.0142857143"
    assert row["accounting_external_flow_qoq_pct_gdp"] == "0.0095238095"
    assert row["accounting_identity_total_qoq_pct_gdp"] == "0.0476190476"
    assert row["accounting_identity_gap_qoq_pct_gdp"] == "0.0952380952"


def test_overlay_missing_quarters_prefers_primary_values_and_fills_gaps() -> None:
    primary = {"2021Q1": "1.0", "2021Q2": ""}
    fallback = {"2021Q2": "2.0", "2021Q3": "3.0"}

    result = _overlay_missing_quarters(primary, fallback)

    assert result == {"2021Q1": "1.0", "2021Q2": "2.0", "2021Q3": "3.0"}


def test_set_row_linear_combo_treats_missing_on_rrp_as_zero_for_public_liquidity() -> None:
    row = {
        "tga_balance_qoq": "10.0",
        "tga_balance_qoq__available_at": "2024-06-29",
        "on_rrp_balance_qoq": "",
        "on_rrp_balance_qoq__available_at": "",
    }

    _set_row_linear_combo(
        row,
        output_name="public_liquidity_proxy_block_qoq",
        components=(("tga_balance_qoq", -1.0), ("on_rrp_balance_qoq", -1.0)),
    )

    assert row["public_liquidity_proxy_block_qoq"] == "-10.0"


def test_quarterly_design_builder_supports_low_reserve_state_job(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: tdc_state_dep_low_reserves",
                "    estimator: lp",
                "    state_id: coord_low_reserve_state_l1",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: supporting_reduced_form",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,100,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,120,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "qrawatch" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "net_bills_bn,net_bills_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,10,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "bill_share,bill_share,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,share,0.1,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "M2SL.csv",
        "\n".join(
            [
                "date,value",
                "2024-01-31,1000",
                "2024-02-29,1005",
                "2024-03-31,1010",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "BOGZ1FL764100005Q.csv",
        "\n".join(
            [
                "date,value",
                "2023-10-01,900",
                "2024-01-01,1030",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "TGCRRATE.csv",
        "\n".join(
            [
                "date,value",
                "2024-01-02,5.30",
                "2024-02-01,5.32",
                "2024-03-01,5.34",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "RRPONTSYAWARD.csv",
        "\n".join(
            [
                "date,value",
                "2024-01-02,5.20",
                "2024-02-01,5.21",
                "2024-03-01,5.22",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "WRESBAL.csv",
        "\n".join(
            [
                "date,value",
                "2023-10-04,100",
                "2024-01-03,250",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "RRPONTSYD.csv",
        "\n".join(
            [
                "date,value",
                "2023-10-02,5",
                "2024-01-02,50",
            ]
        ),
    )

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = build_quarterly_design(paths, job_id="tdc_state_dep_low_reserves")

    with result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["coord_low_reserve_state_l1"] == "1"
    assert rows[1]["matched_total_deposits"] == "130.0"
    assert rows[1]["repo_spread"] == "11.0"

    design_manifest = json.loads(result.design_manifest_path.read_text(encoding="utf-8"))
    assert design_manifest["missing_required_series"] == []
    assert design_manifest["missing_state_ids"] == []
    assert "coord_low_reserve_state_l1" in design_manifest["state_ids"]

    sample_manifest = json.loads(result.sample_manifest_path.read_text(encoding="utf-8"))
    assert sample_manifest["rows"][-1]["observations_remaining"] == 1


def test_quarterly_design_builder_supports_on_rrp_drain_state_job(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: tdc_state_dep_on_rrp_drain",
                "    estimator: lp",
                "    state_id: coord_on_rrp_drain_state_l1",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: supporting_reduced_form",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2023-12-31,2024-03-30,2024-03-30,seed_bundle_snapshot_conservative_90d_lag,usd_millions,80,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,100,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,120,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "qrawatch" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "net_bills_bn,net_bills_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,10,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "bill_share,bill_share,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,share,0.1,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "M2SL.csv",
        "\n".join(
            [
                "date,value",
                "2024-01-31,1000",
                "2024-02-29,1005",
                "2024-03-31,1010",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "BOGZ1FL764100005Q.csv",
        "\n".join(
            [
                "date,value",
                "2023-10-01,900",
                "2024-01-01,1030",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "TGCRRATE.csv",
        "\n".join(
            [
                "date,value",
                "2024-01-02,5.30",
                "2024-02-01,5.32",
                "2024-03-01,5.34",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "RRPONTSYAWARD.csv",
        "\n".join(
            [
                "date,value",
                "2024-01-02,5.20",
                "2024-02-01,5.21",
                "2024-03-01,5.22",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "WRESBAL.csv",
        "\n".join(
            [
                "date,value",
                "2023-10-04,100",
                "2024-01-03,120",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "RRPONTSYD.csv",
        "\n".join(
            [
                "date,value",
                "2023-10-02,80",
                "2024-01-02,10",
            ]
        ),
    )

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = build_quarterly_design(paths, job_id="tdc_state_dep_on_rrp_drain")

    with result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["coord_on_rrp_drain_state_l1"] == "-1.0"
    assert rows[1]["coord_on_rrp_share_q"] == "0.076923"
    assert rows[1]["matched_total_deposits"] == "130.0"
    assert rows[1]["repo_spread"] == "11.0"

    design_manifest = json.loads(result.design_manifest_path.read_text(encoding="utf-8"))
    assert design_manifest["missing_required_series"] == []
    assert design_manifest["missing_state_ids"] == []
    assert "coord_on_rrp_drain_state_l1" in design_manifest["state_ids"]


def test_quarterly_design_builder_supports_lp_iv_qra_ru_gap_job(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: tdc_lpiv_deposits_qra_ru_gap",
                "    estimator: lp_iv",
                "    instruments: [iv_qra_x_ru_gap]",
                "    controls:",
                "      include_main_qra_shock: true",
                "      include_main_state: true",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: headline_identified",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2023-12-31,2024-03-30,2024-03-30,seed_bundle_snapshot_conservative_90d_lag,usd_millions,80,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,100,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,120,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "qrawatch" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "net_bills_bn,net_bills_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,10,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "bill_share,bill_share,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,share,0.1,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tsyparty" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tsyparty_ru_gap_l1,tsyparty_ru_gap_l1,repo_publish,tsyparty,counterparty_flows,quarterly,2024-03-31,2024-03-30,2024-03-30,tsyparty_publish_snapshot_conservative_lagged_state,share_gap,0.2,none,unknown,false,absorption_state,state,lagged_row_minus_bank_absorption_share_from_tsyparty_counterparty_flows",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "M2SL.csv",
        "\n".join(
            [
                "date,value",
                "2024-01-31,1000",
                "2024-02-29,1005",
                "2024-03-31,1010",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "BOGZ1FL764100005Q.csv",
        "\n".join(
            [
                "date,value",
                "2023-10-01,900",
                "2024-01-01,1030",
            ]
        ),
    )

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = build_quarterly_design(paths, job_id="tdc_lpiv_deposits_qra_ru_gap")

    with result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["qra_maturity_tilt_flow"] == "5"
    assert rows[1]["iv_qra_x_ru_gap"] == "1.0"
    assert rows[1]["tsyparty_ru_gap_l1"] == "0.2"

    design_manifest = json.loads(result.design_manifest_path.read_text(encoding="utf-8"))
    assert design_manifest["instrument_ids"] == ["iv_qra_x_ru_gap"]
    assert design_manifest["outcome_ids"] == ["matched_total_deposits", "m2", "other_component_qoq"]
    assert design_manifest["missing_instrument_ids"] == []
    assert "qra_maturity_tilt_flow" in design_manifest["control_ids"]
    assert "tsyparty_ru_gap_l1" in design_manifest["control_ids"]
    assert "tsyparty_ru_gap_l1" in design_manifest["state_ids"]
    assert design_manifest["status"] == "ready_for_estimation"

    sample_manifest = json.loads(result.sample_manifest_path.read_text(encoding="utf-8"))
    assert sample_manifest["rows"][-1]["observations_remaining"] == 1
    assert sample_manifest["rows"][-1]["reason"] == "required outcomes, states, and instruments available within usable treatment-anchor sample"

    diagnostics_manifest = json.loads(result.diagnostics_manifest_path.read_text(encoding="utf-8"))
    assert diagnostics_manifest["rows_analyzed"] == 1
    assert diagnostics_manifest["instrument_diagnostics"][0]["instrument_id"] == "iv_qra_x_ru_gap"


def test_quarterly_design_builder_supports_lp_iv_bank_absorption_job(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: tdc_lpiv_deposits_qra_bank_absorption",
                "    estimator: lp_iv",
                "    instruments: [iv_qra_x_bank_absorption]",
                "    controls:",
                "      include_main_qra_shock: true",
                "      include_main_state: true",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: headline_identified",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2023-12-31,2024-03-30,2024-03-30,seed_bundle_snapshot_conservative_90d_lag,usd_millions,80,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,100,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,120,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "qrawatch" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "net_bills_bn,net_bills_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,10,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "bill_share,bill_share,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,share,0.1,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tsyparty" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tsyparty_bank_absorption_share_l1,tsyparty_bank_absorption_share_l1,repo_publish,tsyparty,counterparty_flows,quarterly,2024-03-31,2024-03-30,2024-03-30,tsyparty_publish_snapshot_conservative_lagged_state,share,0.4,none,unknown,false,absorption_state,state,lagged_bank_absorption_share_from_tsyparty_counterparty_flows",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "M2SL.csv",
        "\n".join(
            [
                "date,value",
                "2024-01-31,1000",
                "2024-02-29,1005",
                "2024-03-31,1010",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "BOGZ1FL764100005Q.csv",
        "\n".join(
            [
                "date,value",
                "2023-10-01,900",
                "2024-01-01,1030",
            ]
        ),
    )

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = build_quarterly_design(paths, job_id="tdc_lpiv_deposits_qra_bank_absorption")

    with result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["iv_qra_x_bank_absorption"] == "2.0"
    assert rows[1]["tsyparty_bank_absorption_share_l1"] == "0.4"

    design_manifest = json.loads(result.design_manifest_path.read_text(encoding="utf-8"))
    assert design_manifest["instrument_ids"] == ["iv_qra_x_bank_absorption"]
    assert design_manifest["missing_instrument_ids"] == []
    assert "tsyparty_bank_absorption_share_l1" in design_manifest["control_ids"]
    assert "tsyparty_bank_absorption_share_l1" in design_manifest["state_ids"]
    assert design_manifest["diagnostics_manifest_path"].endswith("__iv_diagnostics.json")

    diagnostics_manifest = json.loads(result.diagnostics_manifest_path.read_text(encoding="utf-8"))
    assert diagnostics_manifest["instrument_diagnostics"][0]["instrument_id"] == "iv_qra_x_bank_absorption"


def test_quarterly_design_builder_supports_tsyparty_behavior_state_job(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: tdc_state_dep_bank_foreign_private_corr",
                "    estimator: lp",
                "    state_id: tsyparty_bank_foreign_private_corr_l1",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: supporting_reduced_form",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2023-12-31,2024-03-30,2024-03-30,seed_bundle_snapshot_conservative_90d_lag,usd_millions,80,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,100,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,120,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "qrawatch" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "net_bills_bn,net_bills_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,10,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "bill_share,bill_share,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,share,0.1,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tsyparty" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tsyparty_bank_foreign_private_corr_l1,tsyparty_bank_foreign_private_corr_l1,repo_publish,tsyparty,rolling_correlations,quarterly,2024-03-31,2024-03-30,2024-03-30,tsyparty_publish_snapshot_conservative_lagged_state,correlation,0.31,none,unknown,false,behavior_state,state,lagged_rolling_partial_pearson_between_banks_and_foreigners_private_from_tsyparty_similarity_enriched",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "M2SL.csv",
        "\n".join(["date,value", "2024-01-31,1000", "2024-02-29,1005", "2024-03-31,1010"]),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "BOGZ1FL764100005Q.csv",
        "\n".join(["date,value", "2023-10-01,900", "2024-01-01,1030"]),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "TGCRRATE.csv",
        "\n".join(["date,value", "2024-01-02,5.30", "2024-02-01,5.32", "2024-03-01,5.34"]),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "RRPONTSYAWARD.csv",
        "\n".join(["date,value", "2024-01-02,5.20", "2024-02-01,5.21", "2024-03-01,5.22"]),
    )

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = build_quarterly_design(paths, job_id="tdc_state_dep_bank_foreign_private_corr")

    with result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["tsyparty_bank_foreign_private_corr_l1"] == "0.31"
    assert rows[1]["repo_spread"] == "11.0"

    design_manifest = json.loads(result.design_manifest_path.read_text(encoding="utf-8"))
    assert design_manifest["missing_state_ids"] == []
    assert "tsyparty_bank_foreign_private_corr_l1" in design_manifest["state_ids"]


def test_quarterly_design_builder_supports_tsyparty_behavior_iv_job(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: tdc_lpiv_deposits_qra_bank_foreign_private_corr",
                "    estimator: lp_iv",
                "    instruments: [iv_qra_x_bank_foreign_private_corr]",
                "    controls:",
                "      include_main_qra_shock: true",
                "      include_main_state: true",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: supporting_descriptive",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2023-12-31,2024-03-30,2024-03-30,seed_bundle_snapshot_conservative_90d_lag,usd_millions,80,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,100,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,120,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "qrawatch" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "net_bills_bn,net_bills_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,10,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "bill_share,bill_share,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,share,0.1,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tsyparty" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tsyparty_bank_foreign_private_corr_l1,tsyparty_bank_foreign_private_corr_l1,repo_publish,tsyparty,rolling_correlations,quarterly,2024-03-31,2024-03-30,2024-03-30,tsyparty_publish_snapshot_conservative_lagged_state,correlation,0.31,none,unknown,false,behavior_state,state,lagged_rolling_partial_pearson_between_banks_and_foreigners_private_from_tsyparty_similarity_enriched",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "M2SL.csv",
        "\n".join(["date,value", "2024-01-31,1000", "2024-02-29,1005", "2024-03-31,1010"]),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "BOGZ1FL764100005Q.csv",
        "\n".join(["date,value", "2023-10-01,900", "2024-01-01,1030"]),
    )

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = build_quarterly_design(paths, job_id="tdc_lpiv_deposits_qra_bank_foreign_private_corr")

    with result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["iv_qra_x_bank_foreign_private_corr"] == "1.55"
    assert rows[1]["tsyparty_bank_foreign_private_corr_l1"] == "0.31"

    design_manifest = json.loads(result.design_manifest_path.read_text(encoding="utf-8"))
    assert design_manifest["instrument_ids"] == ["iv_qra_x_bank_foreign_private_corr"]
    assert "tsyparty_bank_foreign_private_corr_l1" in design_manifest["control_ids"]
    assert "tsyparty_bank_foreign_private_corr_l1" in design_manifest["state_ids"]

    diagnostics_manifest = json.loads(result.diagnostics_manifest_path.read_text(encoding="utf-8"))
    assert diagnostics_manifest["instrument_diagnostics"][0]["instrument_id"] == "iv_qra_x_bank_foreign_private_corr"


def test_quarterly_design_builder_supports_wamest_state_job(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: tdc_state_dep_bank_short_share",
                "    estimator: lp",
                "    state_id: wamest_bank_reserve_short_share_l1",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: supporting_reduced_form",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2023-12-31,2024-03-30,2024-03-30,seed_bundle_snapshot_conservative_90d_lag,usd_millions,80,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,100,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,120,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "qrawatch" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "net_bills_bn,net_bills_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,10,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "bill_share,bill_share,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,share,0.1,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "wamest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "wamest_bank_reserve_short_share_l1,wamest_bank_reserve_short_share_l1,repo_publish,wamest,sector_effective_maturity_full,quarterly,2024-03-31,2024-03-30,2024-03-30,wamest_publish_snapshot_conservative_lagged_state,share,0.241,none,unknown,false,bank_capacity_state,state,lagged_short_share_le_1y_from_wamest_bank_reserve_access_core",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "M2SL.csv",
        "\n".join(["date,value", "2024-01-31,1000", "2024-02-29,1005", "2024-03-31,1010"]),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "BOGZ1FL764100005Q.csv",
        "\n".join(["date,value", "2023-10-01,900", "2024-01-01,1030"]),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "TGCRRATE.csv",
        "\n".join(["date,value", "2024-01-02,5.30", "2024-02-01,5.32", "2024-03-01,5.34"]),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "RRPONTSYAWARD.csv",
        "\n".join(["date,value", "2024-01-02,5.20", "2024-02-01,5.21", "2024-03-01,5.22"]),
    )

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = build_quarterly_design(paths, job_id="tdc_state_dep_bank_short_share")

    with result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["wamest_bank_reserve_short_share_l1"] == "0.241"
    assert rows[1]["repo_spread"] == "11.0"

    design_manifest = json.loads(result.design_manifest_path.read_text(encoding="utf-8"))
    assert design_manifest["missing_state_ids"] == []
    assert "wamest_bank_reserve_short_share_l1" in design_manifest["state_ids"]


def test_quarterly_design_builder_supports_wamest_iv_job(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: tdc_lpiv_deposits_qra_bank_short_share",
                "    estimator: lp_iv",
                "    instruments: [iv_qra_x_bank_short_share]",
                "    controls:",
                "      include_main_qra_shock: true",
                "      include_main_state: true",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: supporting_descriptive",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2023-12-31,2024-03-30,2024-03-30,seed_bundle_snapshot_conservative_90d_lag,usd_millions,80,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,100,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,120,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "qrawatch" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "net_bills_bn,net_bills_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,10,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "bill_share,bill_share,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,share,0.1,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "wamest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "wamest_bank_reserve_short_share_l1,wamest_bank_reserve_short_share_l1,repo_publish,wamest,sector_effective_maturity_full,quarterly,2024-03-31,2024-03-30,2024-03-30,wamest_publish_snapshot_conservative_lagged_state,share,0.241,none,unknown,false,bank_capacity_state,state,lagged_short_share_le_1y_from_wamest_bank_reserve_access_core",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "M2SL.csv",
        "\n".join(["date,value", "2024-01-31,1000", "2024-02-29,1005", "2024-03-31,1010"]),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "BOGZ1FL764100005Q.csv",
        "\n".join(["date,value", "2023-10-01,900", "2024-01-01,1030"]),
    )

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = build_quarterly_design(paths, job_id="tdc_lpiv_deposits_qra_bank_short_share")

    with result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["iv_qra_x_bank_short_share"] == "1.205"
    assert rows[1]["wamest_bank_reserve_short_share_l1"] == "0.241"

    design_manifest = json.loads(result.design_manifest_path.read_text(encoding="utf-8"))
    assert design_manifest["instrument_ids"] == ["iv_qra_x_bank_short_share"]
    assert "wamest_bank_reserve_short_share_l1" in design_manifest["control_ids"]
    assert "wamest_bank_reserve_short_share_l1" in design_manifest["state_ids"]

    diagnostics_manifest = json.loads(result.diagnostics_manifest_path.read_text(encoding="utf-8"))
    assert diagnostics_manifest["instrument_diagnostics"][0]["instrument_id"] == "iv_qra_x_bank_short_share"


def test_quarterly_design_builder_supports_slrwatch_state_job(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: tdc_state_dep_slr_bank_leverage_pressure",
                "    estimator: lp",
                "    state_id: slrwatch_bank_leverage_pressure_l1",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: supporting_reduced_form",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2023-12-31,2024-03-30,2024-03-30,seed_bundle_snapshot_conservative_90d_lag,usd_millions,80,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,100,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,120,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "qrawatch" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "net_bills_bn,net_bills_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,10,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "bill_share,bill_share,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,share,0.1,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "slrwatch" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "slrwatch_bank_leverage_pressure_l1,slrwatch_bank_leverage_pressure_l1,repo_publish,slrwatch,constraint_decomposition_prepared_panel,quarterly,2024-03-31,2024-03-30,2024-03-30,slrwatch_publish_snapshot_conservative_lagged_state,score,0.55,none,unknown,false,slr_constraint_state,state,lagged_mean_leverage_pressure_score_from_slrwatch_insured_bank_constraint_panel",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "M2SL.csv",
        "\n".join(["date,value", "2024-01-31,1000", "2024-02-29,1005", "2024-03-31,1010"]),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "BOGZ1FL764100005Q.csv",
        "\n".join(["date,value", "2023-10-01,900", "2024-01-01,1030"]),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "TGCRRATE.csv",
        "\n".join(["date,value", "2024-01-02,5.30", "2024-02-01,5.32", "2024-03-01,5.34"]),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "RRPONTSYAWARD.csv",
        "\n".join(["date,value", "2024-01-02,5.20", "2024-02-01,5.21", "2024-03-01,5.22"]),
    )

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = build_quarterly_design(paths, job_id="tdc_state_dep_slr_bank_leverage_pressure")

    with result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["slrwatch_bank_leverage_pressure_l1"] == "0.55"
    assert rows[1]["repo_spread"] == "11.0"

    design_manifest = json.loads(result.design_manifest_path.read_text(encoding="utf-8"))
    assert design_manifest["missing_state_ids"] == []
    assert "slrwatch_bank_leverage_pressure_l1" in design_manifest["state_ids"]


def test_quarterly_design_builder_supports_slrwatch_iv_job(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: tdc_lpiv_deposits_qra_slr_bank_leverage_pressure",
                "    estimator: lp_iv",
                "    instruments: [iv_qra_x_slr_bank_leverage_pressure]",
                "    controls:",
                "      include_main_qra_shock: true",
                "      include_main_state: true",
                "    horizons: [0, 1, 2, 4]",
                "    output_family: supporting_descriptive",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2023-12-31,2024-03-30,2024-03-30,seed_bundle_snapshot_conservative_90d_lag,usd_millions,80,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,100,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,120,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "qrawatch" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "net_bills_bn,net_bills_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,usd_billions,10,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "bill_share,bill_share,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07T08:30:00-05:00,official_qra_release_timestamp,share,0.1,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "bundles" / "slrwatch" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "slrwatch_bank_leverage_pressure_l1,slrwatch_bank_leverage_pressure_l1,repo_publish,slrwatch,constraint_decomposition_prepared_panel,quarterly,2024-03-31,2024-03-30,2024-03-30,slrwatch_publish_snapshot_conservative_lagged_state,score,0.55,none,unknown,false,slr_constraint_state,state,lagged_mean_leverage_pressure_score_from_slrwatch_insured_bank_constraint_panel",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "M2SL.csv",
        "\n".join(["date,value", "2024-01-31,1000", "2024-02-29,1005", "2024-03-31,1010"]),
    )
    _write_text(
        tmp_path / "data" / "raw" / "fred" / "BOGZ1FL764100005Q.csv",
        "\n".join(["date,value", "2023-10-01,900", "2024-01-01,1030"]),
    )

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = build_quarterly_design(paths, job_id="tdc_lpiv_deposits_qra_slr_bank_leverage_pressure")

    with result.bundle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["iv_qra_x_slr_bank_leverage_pressure"] == "2.75"
    assert rows[1]["slrwatch_bank_leverage_pressure_l1"] == "0.55"

    design_manifest = json.loads(result.design_manifest_path.read_text(encoding="utf-8"))
    assert design_manifest["instrument_ids"] == ["iv_qra_x_slr_bank_leverage_pressure"]
    assert "slrwatch_bank_leverage_pressure_l1" in design_manifest["control_ids"]
    assert "slrwatch_bank_leverage_pressure_l1" in design_manifest["state_ids"]

    diagnostics_manifest = json.loads(result.diagnostics_manifest_path.read_text(encoding="utf-8"))
    assert diagnostics_manifest["instrument_diagnostics"][0]["instrument_id"] == "iv_qra_x_slr_bank_leverage_pressure"
