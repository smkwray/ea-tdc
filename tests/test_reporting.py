from __future__ import annotations

import csv
import json
from pathlib import Path

from ea_tdc.paths import ensure_repo_dirs, project_paths
from ea_tdc.reporting import (
    _classify_release_contract_row,
    build_component_sidecar_artifact_pack,
    build_component_sidecar_screening,
    build_event_sidecar_artifact_pack,
    build_event_sidecar_screening,
    build_robustness_snapshot,
    build_release_artifact_contract,
    build_release_contract,
    build_release_scorecard,
    build_release_snapshot,
    build_stage_completion_closeout,
)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_release_snapshot_builds_all_jobs_and_writes_summary(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)

    _write_text(
        paths.config / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: baseline_tdc_lp_deposits",
                "    estimator: lp",
                "    freq: quarterly",
                "    outcomes: [matched_total_deposits, other_component_qoq, m2]",
                "    horizons: [0, 1]",
                "    output_family: headline_identified",
                "  - job_id: qra_event_rates_63bd",
                "    estimator: event_lp",
                "    freq: irregular_event",
                "    treatment_id: qra_release_63bd",
                "    outcomes: [threefytp10]",
                "    horizons_bd: [1]",
                "    cutoff_rule: event_close_with_embargo",
                "    output_family: supporting_reduced_form",
            ]
        ),
    )
    _write_text(
        paths.bundles / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,100,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,120,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    _write_text(
        paths.bundles / "qrawatch" / "standardized_series.csv",
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
        paths.bundles / "qrawatch" / "event_bundle.csv",
        "\n".join(
            [
                "event_id,quarter,event_label,event_date,event_type,source_repo,treatment_id,treatment_value,treatment_units,cutoff_timestamp,embargo_rule,horizon_unit,usable_for_headline,usable_for_headline_reason,quality_tier,claim_scope",
                "qra_2024_01,2024Q1,2024 Jan QRA,2024-01-03,qra_release,qrawatch,canonical_shock_bn,10,usd_billions,2024-01-03T08:30:00-05:00,event_close_with_embargo,business_day,true,usable,Tier A,headline",
            ]
        ),
    )
    _write_text(paths.config / "debt_limit_intervals.csv", "quarter,start_date,end_date,source_note\n")
    _write_text(paths.raw_fred / "M2SL.csv", "date,value\n2024-03-31,1010\n")
    _write_text(paths.raw_fred / "BOGZ1FL764100005Q.csv", "date,value\n2023-10-01,900\n2024-01-01,1030\n")
    _write_text(paths.raw_fred / "THREEFYTP10.csv", "date,value\n2024-01-02,1.60\n2024-01-04,1.66\n")
    _write_text(paths.raw_fred / "DFF.csv", "date,value\n2024-01-02,5.33\n2024-01-04,5.38\n")

    result = build_release_snapshot(paths)

    assert result.jobs_built == 2
    assert result.ready_jobs == 2
    assert result.partial_jobs == 0

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["jobs_built"] == 2
    assert {row["job_id"] for row in summary["rows"]} == {"baseline_tdc_lp_deposits", "qra_event_rates_63bd"}
    event_row = next(row for row in summary["rows"] if row["job_id"] == "qra_event_rates_63bd")
    assert event_row["sample_policy"] == "headline_strict"
    assert event_row["requested_sample_rows"] == 1
    assert event_row["headline_eligible_rows"] == 1

    with result.summary_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"ready_for_estimation"}


def test_release_scorecard_combines_readiness_and_estimation(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)

    _write_text(
        paths.config / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: custom_lp_job",
                "    estimator: lp",
                "    freq: quarterly",
                "    treatment_id: tdc_bank_only_shock",
                "    outcomes: [matched_total_deposits]",
                "    horizons: [0]",
                "    controls_explicit: [gdp_deflator]",
                "    output_family: headline_identified",
            ]
        ),
    )
    _write_text(
        paths.bundles / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,1,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-06-30,2024-09-28,2024-09-28,seed_bundle_snapshot_conservative_90d_lag,usd_millions,2,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-09-30,2024-12-29,2024-12-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,3,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-12-31,2025-03-31,2025-03-31,seed_bundle_snapshot_conservative_90d_lag,usd_millions,4,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,1,none,unknown,false,reference,control,reference",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-06-30,2024-09-28,2024-09-28,seed_bundle_snapshot_conservative_90d_lag,index,2,none,unknown,false,reference,control,reference",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-09-30,2024-12-29,2024-12-29,seed_bundle_snapshot_conservative_90d_lag,index,1,none,unknown,false,reference,control,reference",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-12-31,2025-03-31,2025-03-31,seed_bundle_snapshot_conservative_90d_lag,index,2,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    _write_text(
        paths.bundles / "qrawatch" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-06-30,2024-05-07,2024-05-07,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-09-30,2024-08-07,2024-08-07,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-12-31,2024-11-07,2024-11-07,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
            ]
        ),
    )
    _write_text(paths.raw_fred / "BOGZ1FL764100005Q.csv", "date,value\n2023-10-01,0\n2024-01-01,1\n2024-04-01,3\n2024-07-01,6\n2024-10-01,10\n")
    _write_text(paths.raw_fred / "GDP.csv", "date,value\n2024-03-31,1\n2024-06-30,1\n2024-09-30,1\n2024-12-31,1\n")
    _write_text(paths.raw_fred / "FEDFUNDS.csv", "date,value\n2024-01-01,1\n2024-04-01,1\n2024-07-01,1\n2024-10-01,1\n")
    _write_text(paths.raw_fred / "TOTRESNS.csv", "date,value\n2024-03-31,1\n2024-06-30,1\n2024-09-30,1\n2024-12-31,1\n")

    result = build_release_scorecard(paths)

    assert result.public_jobs == 1
    assert result.committed_public_jobs == 1
    assert result.ready_jobs == 1
    assert result.estimated_jobs == 1
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["committed_public_jobs"] == 1
    assert summary["deferred_public_jobs"] == 0
    assert summary["estimated_jobs"] == 1
    assert summary["jobs_with_warnings"] == 0
    assert summary["rows"][0]["job_id"] == "custom_lp_job"
    assert summary["rows"][0]["estimation_status"] == "estimated"
    assert summary["rows"][0]["covariance_estimators_used"] == "newey_west"
    assert summary["rows"][0]["warning_rows"] == 0
    assert summary["rows"][0]["publication_risk_flags"] == ""
    assert summary["rows"][0]["sample_policy"] == ""


def test_classify_release_contract_row_tiers() -> None:
    assert _classify_release_contract_row(
        is_public_job=True,
        release1_scope="committed",
        output_family="headline_identified",
        sample_policy="",
        readiness_status="ready_for_estimation",
        estimated_rows_written=10,
        publication_risk_flags=[],
    ) == ("release1_main_candidate", "main_text", "clean_headline_identified")
    assert _classify_release_contract_row(
        is_public_job=True,
        release1_scope="committed",
        output_family="supporting_reduced_form",
        sample_policy="",
        readiness_status="ready_for_estimation",
        estimated_rows_written=8,
        publication_risk_flags=[],
    ) == ("release1_appendix_candidate", "appendix", "clean_supporting_reduced_form")
    assert _classify_release_contract_row(
        is_public_job=False,
        release1_scope="committed",
        output_family="supporting_descriptive",
        sample_policy="reviewed_nonmissing",
        readiness_status="ready_for_estimation",
        estimated_rows_written=18,
        publication_risk_flags=[],
    ) == ("exploratory_sidecar", "sidecar", "descriptive_or_exploratory_sample")
    assert _classify_release_contract_row(
        is_public_job=True,
        release1_scope="committed",
        output_family="headline_identified",
        sample_policy="",
        readiness_status="ready_for_estimation",
        estimated_rows_written=10,
        publication_risk_flags=["weak_instrument"],
    ) == ("blocked", "blocked", "weak_instrument")
    assert _classify_release_contract_row(
        is_public_job=True,
        release1_scope="deferred",
        output_family="headline_identified",
        sample_policy="",
        readiness_status="ready_for_estimation",
        estimated_rows_written=10,
        publication_risk_flags=["weak_instrument"],
    ) == ("deferred_development", "deferred", "deferred_weak_instrument")


def test_release_contract_separates_main_appendix_and_sidecar(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)

    _write_text(
        paths.config / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: custom_lp_job",
                "    estimator: lp",
                "    freq: quarterly",
                "    treatment_id: tdc_bank_only_shock",
                "    outcomes: [matched_total_deposits]",
                "    horizons: [0]",
                "    controls_explicit: [gdp_deflator]",
                "    output_family: headline_identified",
                "  - job_id: custom_appendix_job",
                "    estimator: lp",
                "    freq: quarterly",
                "    treatment_id: tdc_bank_only_shock",
                "    outcomes: [matched_total_deposits]",
                "    horizons: [0]",
                "    controls_explicit: [gdp_deflator]",
                "    output_family: supporting_reduced_form",
                "  - job_id: custom_event_descriptive",
                "    estimator: event_lp",
                "    freq: irregular_event",
                "    treatment_id: qra_release_63bd",
                "    sample_policy: reviewed_nonmissing",
                "    controls_explicit: [dff]",
                "    outcomes: [threefytp10]",
                "    horizons_bd: [1]",
                "    cutoff_rule: event_close_with_embargo",
                "    output_family: supporting_descriptive",
                "    track_in_release_snapshot: false",
                "  - job_id: custom_deferred_iv",
                "    estimator: lp",
                "    freq: quarterly",
                "    treatment_id: tdc_bank_only_shock",
                "    outcomes: [matched_total_deposits]",
                "    horizons: [0]",
                "    controls_explicit: [gdp_deflator]",
                "    output_family: headline_identified",
                "    release1_scope: deferred",
            ]
        ),
    )
    _write_text(
        paths.bundles / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,1,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-06-30,2024-09-28,2024-09-28,seed_bundle_snapshot_conservative_90d_lag,usd_millions,2,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-09-30,2024-12-29,2024-12-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,3,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-12-31,2025-03-31,2025-03-31,seed_bundle_snapshot_conservative_90d_lag,usd_millions,4,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,1,none,unknown,false,reference,control,reference",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-06-30,2024-09-28,2024-09-28,seed_bundle_snapshot_conservative_90d_lag,index,2,none,unknown,false,reference,control,reference",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-09-30,2024-12-29,2024-12-29,seed_bundle_snapshot_conservative_90d_lag,index,1,none,unknown,false,reference,control,reference",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-12-31,2025-03-31,2025-03-31,seed_bundle_snapshot_conservative_90d_lag,index,2,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    _write_text(
        paths.bundles / "qrawatch" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-06-30,2024-05-07,2024-05-07,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-09-30,2024-08-07,2024-08-07,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-12-31,2024-11-07,2024-11-07,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
            ]
        ),
    )
    _write_text(
        paths.bundles / "qrawatch" / "event_bundle.csv",
        "\n".join(
                [
                    "event_id,quarter,event_label,event_date,event_type,source_repo,treatment_id,treatment_value,treatment_units,cutoff_timestamp,embargo_rule,horizon_unit,usable_for_headline,usable_for_headline_reason,usable_for_descriptive_headline,descriptive_headline_reason,quality_tier,claim_scope,shock_review_status,shock_missing_flag,small_denominator_flag",
                    "qra_2024_01,2024Q1,2024 Jan QRA,2024-01-03,qra_release,qrawatch,canonical_shock_bn,10,usd_billions,2024-01-03T08:30:00-05:00,event_close_with_embargo,business_day,true,usable,true,usable,Tier A,headline,reviewed,false,false",
                    "qra_2024_02,2024Q2,2024 May QRA,2024-05-01,qra_release,qrawatch,canonical_shock_bn,11,usd_billions,2024-05-01T08:30:00-04:00,event_close_with_embargo,business_day,true,usable,true,usable,Tier A,headline,reviewed,false,false",
                    "qra_2024_03,2024Q3,2024 Aug QRA,2024-07-31,qra_release,qrawatch,canonical_shock_bn,9,usd_billions,2024-07-31T08:30:00-04:00,event_close_with_embargo,business_day,true,usable,true,usable,Tier A,headline,reviewed,false,false",
                    "qra_2024_04,2024Q4,2024 Nov QRA,2024-10-30,qra_release,qrawatch,canonical_shock_bn,12,usd_billions,2024-10-30T08:30:00-04:00,event_close_with_embargo,business_day,true,usable,true,usable,Tier A,headline,reviewed,false,false",
                    "qra_2025_01,2025Q1,2025 Feb QRA,2025-01-29,qra_release,qrawatch,canonical_shock_bn,13,usd_billions,2025-01-29T08:30:00-05:00,event_close_with_embargo,business_day,false,not_headline,true,reviewed_only,Tier B,reviewed,reviewed,false,false",
                    "qra_2025_02,2025Q2,2025 Apr QRA,2025-04-30,qra_release,qrawatch,canonical_shock_bn,8,usd_billions,2025-04-30T08:30:00-04:00,event_close_with_embargo,business_day,false,not_headline,true,reviewed_only,Tier B,reviewed,reviewed,false,false",
                    "qra_2025_03,2025Q3,2025 Jul QRA,2025-07-30,qra_release,qrawatch,canonical_shock_bn,7,usd_billions,2025-07-30T08:30:00-04:00,event_close_with_embargo,business_day,false,not_headline,true,reviewed_only,Tier B,reviewed,reviewed,false,false",
                    "qra_2025_04,2025Q4,2025 Oct QRA,2025-10-29,qra_release,qrawatch,canonical_shock_bn,6,usd_billions,2025-10-29T08:30:00-04:00,event_close_with_embargo,business_day,false,not_headline,true,reviewed_only,Tier B,reviewed,reviewed,false,false",
                ]
            ),
        )
    _write_text(paths.config / "debt_limit_intervals.csv", "quarter,start_date,end_date,source_note\n")
    _write_text(paths.raw_fred / "BOGZ1FL764100005Q.csv", "date,value\n2023-10-01,0\n2024-01-01,1\n2024-04-01,3\n2024-07-01,6\n2024-10-01,10\n")
    _write_text(paths.raw_fred / "GDP.csv", "date,value\n2024-03-31,1\n2024-06-30,1\n2024-09-30,1\n2024-12-31,1\n")
    _write_text(paths.raw_fred / "FEDFUNDS.csv", "date,value\n2024-01-01,1\n2024-04-01,1\n2024-07-01,1\n2024-10-01,1\n")
    _write_text(paths.raw_fred / "TOTRESNS.csv", "date,value\n2024-03-31,1\n2024-06-30,1\n2024-09-30,1\n2024-12-31,1\n")
    _write_text(paths.raw_fred / "THREEFYTP10.csv", "date,value\n2024-01-02,1.60\n2024-01-04,1.66\n2024-05-01,1.50\n2024-05-02,1.55\n2024-07-30,1.40\n2024-07-31,1.44\n2024-10-29,1.35\n2024-10-30,1.38\n2025-01-28,1.33\n2025-01-29,1.37\n2025-04-29,1.31\n2025-04-30,1.34\n2025-07-29,1.29\n2025-07-30,1.31\n2025-10-28,1.28\n2025-10-29,1.30\n")
    _write_text(paths.raw_fred / "DFF.csv", "date,value\n2024-01-02,5.33\n2024-01-04,5.38\n2024-05-01,5.31\n2024-05-02,5.35\n2024-07-30,5.30\n2024-07-31,5.34\n2024-10-29,5.28\n2024-10-30,5.30\n2025-01-28,5.27\n2025-01-29,5.29\n2025-04-29,5.25\n2025-04-30,5.27\n2025-07-29,5.22\n2025-07-30,5.24\n2025-10-28,5.20\n2025-10-29,5.21\n")

    result = build_release_contract(paths)

    assert result.active_jobs == 4
    assert result.main_candidates == 1
    assert result.appendix_candidates == 1
    assert result.exploratory_sidecar_jobs == 1
    assert result.deferred_jobs == 1
    assert result.blocked_jobs == 0
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["main_candidates"] == 1
    assert summary["appendix_candidates"] == 1
    assert summary["exploratory_sidecar_jobs"] == 1
    assert summary["deferred_jobs"] == 1
    rows = {row["job_id"]: row for row in summary["rows"]}
    assert rows["custom_lp_job"]["contract_tier"] == "release1_main_candidate"
    assert rows["custom_appendix_job"]["contract_tier"] == "release1_appendix_candidate"
    assert rows["custom_event_descriptive"]["contract_tier"] == "exploratory_sidecar"
    assert rows["custom_event_descriptive"]["release_channel"] == "sidecar"
    assert rows["custom_deferred_iv"]["contract_tier"] == "deferred_development"
    assert rows["custom_deferred_iv"]["release_channel"] == "deferred"


def test_event_sidecar_screening_builds_summary(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    models_dir = paths.output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    _write_text(
        models_dir / "qra_event_rates_63bd__event_lp_estimates.csv",
        "\n".join(
            [
                "job_id,outcome,horizon,beta,se,lower95,upper95,z_score,p_value_normal,n,treatment_id,control_ids_used,response_type,inference_method,covariance_estimator,covariance_lags,rsquared,warning_flags,dropped_control_ids",
                "qra_event_rates_63bd,dgs10,63,2.25,0.5,1.2,3.3,4.5,0.0001,14,treatment_value,controls,direct_event_delta_horizon,event_ols_hc1_scaffold,hc1,0,0.4,,",
                "qra_event_rates_63bd,repo_spread,21,0.03,0.01,0.01,0.05,2.1,0.04,14,treatment_value,controls,direct_event_delta_horizon,event_ols_hc1_scaffold,hc1,0,0.2,,",
                "qra_event_rates_63bd,dgs2,1,0.01,0.02,-0.03,0.05,0.5,0.60,14,treatment_value,controls,direct_event_delta_horizon,event_ols_hc1_scaffold,hc1,0,0.1,,",
            ]
        ),
    )
    _write_text(
        models_dir / "qra_event_risk_21bd__event_lp_estimates.csv",
        "\n".join(
            [
                "job_id,outcome,horizon,beta,se,lower95,upper95,z_score,p_value_normal,n,treatment_id,control_ids_used,response_type,inference_method,covariance_estimator,covariance_lags,rsquared,warning_flags,dropped_control_ids",
                "qra_event_risk_21bd,sp500_return,1,-0.044,0.01,-0.06,-0.02,-3.6,0.0003,14,treatment_value,controls,direct_event_delta_horizon,event_ols_hc1_scaffold,hc1,0,0.6,,",
                "qra_event_risk_21bd,tga_balance_change,21,3514,1000,1500,5500,2.7,0.0064,14,treatment_value,controls,direct_event_delta_horizon,event_ols_hc1_scaffold,hc1,0,0.3,,",
                "qra_event_risk_21bd,rrp_balance_change,1,-1.0,1.0,-3.0,1.0,-1.0,0.30,14,treatment_value,controls,direct_event_delta_horizon,event_ols_hc1_scaffold,hc1,0,0.1,,",
            ]
        ),
    )

    result = build_event_sidecar_screening(paths)

    assert result.jobs_summarized == 2
    assert result.signal_count == 4
    with result.summary_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["lane"] for row in rows} == {"rates", "risk_plumbing"}
    summary = result.summary_path.read_text(encoding="utf-8")
    assert "Rates benchmark" in summary
    assert "Risk and plumbing benchmark" in summary
    assert "`dgs10` at `h=63`" in summary
    assert "`sp500_return` at `h=1`" in summary


def test_component_sidecar_screening_builds_summary(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    models_dir = paths.output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    _write_text(
        models_dir / "tdc_component_lp_ru_acquisition__lp_estimates.csv",
        "\n".join(
            [
                "job_id,outcome,horizon,beta,se,lower95,upper95,z_score,p_value_normal,n,treatment_id,control_ids_used,response_type,inference_method,covariance_estimator,covariance_lags,rsquared,warning_flags,dropped_control_ids,state_id,state_profile,state_reference_value,state_interaction_beta,state_interaction_se",
                "tdc_component_lp_ru_acquisition,matched_total_deposits,0,0.80,0.15,0.50,1.10,5.3,0.00001,183,ru_bank_only_tsy_tx,controls,direct_at_h,ols_newey_west_scaffold,newey_west,1,0.5,,,,,,",
                "tdc_component_lp_ru_acquisition,repo_spread,0,0.00001,0.00002,-0.00002,0.00004,0.5,0.60,31,ru_bank_only_tsy_tx,controls,direct_at_h,ols_newey_west_scaffold,newey_west,1,0.2,,,,,,",
            ]
        ),
    )
    _write_text(
        models_dir / "tdc_component_lp_positive_remit_liquidity_decomposition__lp_estimates.csv",
        "\n".join(
            [
                "job_id,outcome,horizon,beta,se,lower95,upper95,z_score,p_value_normal,n,treatment_id,control_ids_used,response_type,inference_method,covariance_estimator,covariance_lags,rsquared,warning_flags,dropped_control_ids,state_id,state_profile,state_reference_value,state_interaction_beta,state_interaction_se",
                "tdc_component_lp_positive_remit_liquidity_decomposition,reserve_balances_net_fed_assets_qoq,0,-12.4,4.0,-20.0,-4.8,-3.1,0.0011,92,fed_remit_positive,controls,direct_at_h,ols_newey_west_scaffold,newey_west,1,0.2,,,,,,",
            ]
        ),
    )
    _write_text(
        models_dir / "tdc_component_state_dep_treasury_cash_drain_on_rrp_drain__lp_estimates.csv",
        "\n".join(
            [
                "job_id,outcome,horizon,beta,se,lower95,upper95,z_score,p_value_normal,n,treatment_id,control_ids_used,response_type,inference_method,covariance_estimator,covariance_lags,rsquared,warning_flags,dropped_control_ids,state_id,state_profile,state_reference_value,state_interaction_beta,state_interaction_se",
                "tdc_component_state_dep_treasury_cash_drain_on_rrp_drain,matched_total_deposits,4,-0.01,0.08,-0.16,0.15,-0.08,0.94,68,minus_treasury_operating_cash_tx,controls,direct_at_h,ols_newey_west_state_interaction_scaffold,newey_west,4,0.3,,,coord_on_rrp_drain_state_l1,low_state,-0.2,-0.164,0.062",
                "tdc_component_state_dep_treasury_cash_drain_on_rrp_drain,matched_total_deposits,4,-0.15,0.09,-0.33,0.03,-1.60,0.089,68,minus_treasury_operating_cash_tx,controls,direct_at_h,ols_newey_west_state_interaction_scaffold,newey_west,4,0.3,,,coord_on_rrp_drain_state_l1,high_state,0.61,-0.164,0.062",
            ]
        ),
    )

    result = build_component_sidecar_screening(paths)

    assert result.jobs_summarized == 3
    assert result.signal_count == 3
    with result.summary_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert {row["lane"] for row in rows} == {"component_reduced_form", "liquidity_decomposition", "state_probe"}
    summary = result.summary_path.read_text(encoding="utf-8")
    assert "RU acquisition reduced form" in summary
    assert "Positive remittance liquidity decomposition" in summary
    assert "Treasury cash under ON RRP drain" in summary
    assert "`matched_total_deposits` at `h=0`" in summary
    assert "`reserve_balances_net_fed_assets_qoq` at `h=0`" in summary
    assert "`matched_total_deposits` at `h=4`: `interaction beta ≈ -0.164`" in summary
    assert "`low-state beta ≈ -0.01`, `p ≈ 0.94`" in summary
    assert "`high-state beta ≈ -0.15`, `p ≈ 0.089`" in summary


def test_component_sidecar_artifact_pack_builds_exports(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    models_dir = paths.output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    _write_text(
        models_dir / "tdc_component_lp_ru_acquisition__lp_estimates.csv",
        "\n".join(
            [
                "job_id,outcome,horizon,beta,se,lower95,upper95,z_score,p_value_normal,n,treatment_id,control_ids_used,response_type,inference_method,covariance_estimator,covariance_lags,rsquared,warning_flags,dropped_control_ids,state_id,state_profile,state_reference_value,state_interaction_beta,state_interaction_se",
                "tdc_component_lp_ru_acquisition,matched_total_deposits,0,0.80,0.15,0.50,1.10,5.3,0.00001,183,ru_bank_only_tsy_tx,controls,direct_at_h,ols_newey_west_scaffold,newey_west,1,0.5,,,,,,",
            ]
        ),
    )
    _write_text(
        models_dir / "tdc_component_lp_positive_remit_liquidity_decomposition__lp_estimates.csv",
        "\n".join(
            [
                "job_id,outcome,horizon,beta,se,lower95,upper95,z_score,p_value_normal,n,treatment_id,control_ids_used,response_type,inference_method,covariance_estimator,covariance_lags,rsquared,warning_flags,dropped_control_ids,state_id,state_profile,state_reference_value,state_interaction_beta,state_interaction_se",
                "tdc_component_lp_positive_remit_liquidity_decomposition,reserve_balances_net_fed_assets_qoq,0,-12.4,4.0,-20.0,-4.8,-3.1,0.0011,92,fed_remit_positive,controls,direct_at_h,ols_newey_west_scaffold,newey_west,1,0.2,,,,,,",
            ]
        ),
    )
    _write_text(
        models_dir / "tdc_component_state_dep_treasury_cash_drain_on_rrp_drain__lp_estimates.csv",
        "\n".join(
            [
                "job_id,outcome,horizon,beta,se,lower95,upper95,z_score,p_value_normal,n,treatment_id,control_ids_used,response_type,inference_method,covariance_estimator,covariance_lags,rsquared,warning_flags,dropped_control_ids,state_id,state_profile,state_reference_value,state_interaction_beta,state_interaction_se",
                "tdc_component_state_dep_treasury_cash_drain_on_rrp_drain,matched_total_deposits,4,-0.01,0.08,-0.16,0.15,-0.08,0.94,68,minus_treasury_operating_cash_tx,controls,direct_at_h,ols_newey_west_state_interaction_scaffold,newey_west,4,0.3,,,coord_on_rrp_drain_state_l1,low_state,-0.2,-0.164,0.062",
                "tdc_component_state_dep_treasury_cash_drain_on_rrp_drain,matched_total_deposits,4,-0.15,0.09,-0.33,0.03,-1.60,0.089,68,minus_treasury_operating_cash_tx,controls,direct_at_h,ols_newey_west_state_interaction_scaffold,newey_west,4,0.3,,,coord_on_rrp_drain_state_l1,high_state,0.61,-0.164,0.062",
            ]
        ),
    )

    result = build_component_sidecar_artifact_pack(paths)

    assert result.signal_count == 3
    assert result.reduced_form_csv_path.exists()
    assert result.liquidity_csv_path.exists()
    assert result.state_probe_csv_path.exists()
    summary = result.summary_path.read_text(encoding="utf-8")
    assert "Component Sidecar Artifact Pack" in summary
    assert "component_sidecar_reduced_form_table.csv" in summary
    assert "component_sidecar_liquidity_table.csv" in summary
    assert "component_sidecar_state_probe_table.csv" in summary
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["signal_count"] == 3


def test_event_sidecar_artifact_pack_and_completion_closeout(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    models_dir = paths.output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    _write_text(
        models_dir / "qra_event_rates_63bd__event_lp_estimates.csv",
        "\n".join(
            [
                "job_id,outcome,horizon,beta,se,lower95,upper95,z_score,p_value_normal,n,treatment_id,control_ids_used,response_type,inference_method,covariance_estimator,covariance_lags,rsquared,warning_flags,dropped_control_ids",
                "qra_event_rates_63bd,dgs10,63,2.25,0.5,1.2,3.3,4.5,0.0001,14,treatment_value,controls,direct_event_delta_horizon,event_ols_hc1_scaffold,hc1,0,0.4,,",
                "qra_event_rates_63bd,repo_spread,21,0.03,0.01,0.01,0.05,2.1,0.04,14,treatment_value,controls,direct_event_delta_horizon,event_ols_hc1_scaffold,hc1,0,0.2,,",
            ]
        ),
    )
    _write_text(
        models_dir / "qra_event_risk_21bd__event_lp_estimates.csv",
        "\n".join(
            [
                "job_id,outcome,horizon,beta,se,lower95,upper95,z_score,p_value_normal,n,treatment_id,control_ids_used,response_type,inference_method,covariance_estimator,covariance_lags,rsquared,warning_flags,dropped_control_ids",
                "qra_event_risk_21bd,sp500_return,1,-0.044,0.01,-0.06,-0.02,-3.6,0.0003,14,treatment_value,controls,direct_event_delta_horizon,event_ols_hc1_scaffold,hc1,0,0.6,,",
                "qra_event_risk_21bd,tga_balance_change,21,3514,1000,1500,5500,2.7,0.0064,14,treatment_value,controls,direct_event_delta_horizon,event_ols_hc1_scaffold,hc1,0,0.3,,",
            ]
        ),
    )

    artifact_pack = build_event_sidecar_artifact_pack(paths)
    assert artifact_pack.signal_count == 4
    assert artifact_pack.rates_csv_path.exists()
    assert artifact_pack.plumbing_csv_path.exists()
    assert artifact_pack.manifest_path.exists()
    artifact_summary = artifact_pack.summary_path.read_text(encoding="utf-8")
    assert "Event Sidecar Artifact Pack" in artifact_summary
    assert "event_sidecar_rates_table.csv" in artifact_summary

    completion = build_stage_completion_closeout(paths)
    assert completion.summary_path.exists()
    assert completion.manifest_path.exists()
    closeout = completion.summary_path.read_text(encoding="utf-8")
    assert "Stage Completion Closeout" in closeout
    assert "event sidecar artifact pack" in closeout.lower()


def test_release_artifact_contract_builds_committed_figure_and_table_plan(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)

    _write_text(
        paths.config / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: custom_lp_job",
                "    estimator: lp",
                "    freq: quarterly",
                "    treatment_id: tdc_bank_only_shock",
                "    outcomes: [matched_total_deposits]",
                "    horizons: [0, 1]",
                "    controls_explicit: [gdp_deflator]",
                "    output_family: headline_identified",
                "  - job_id: custom_appendix_job",
                "    estimator: lp",
                "    freq: quarterly",
                "    treatment_id: tdc_bank_only_shock",
                "    outcomes: [matched_total_deposits]",
                "    horizons: [0]",
                "    controls_explicit: [gdp_deflator]",
                "    output_family: supporting_reduced_form",
                "  - job_id: custom_deferred_iv",
                "    estimator: lp",
                "    freq: quarterly",
                "    treatment_id: tdc_bank_only_shock",
                "    outcomes: [matched_total_deposits]",
                "    horizons: [0]",
                "    controls_explicit: [gdp_deflator]",
                "    output_family: headline_identified",
                "    release1_scope: deferred",
                "  - job_id: custom_event_descriptive",
                "    estimator: event_lp",
                "    freq: irregular_event",
                "    treatment_id: qra_release_63bd",
                "    sample_policy: reviewed_nonmissing",
                "    controls_explicit: [dff]",
                "    outcomes: [threefytp10]",
                "    horizons_bd: [1]",
                "    cutoff_rule: event_close_with_embargo",
                "    output_family: supporting_descriptive",
                "    track_in_release_snapshot: false",
            ]
        ),
    )
    _write_text(
        paths.bundles / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,1,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-06-30,2024-09-28,2024-09-28,seed_bundle_snapshot_conservative_90d_lag,usd_millions,2,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-09-30,2024-12-29,2024-12-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,3,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-12-31,2025-03-31,2025-03-31,seed_bundle_snapshot_conservative_90d_lag,usd_millions,4,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,1,none,unknown,false,reference,control,reference",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-06-30,2024-09-28,2024-09-28,seed_bundle_snapshot_conservative_90d_lag,index,2,none,unknown,false,reference,control,reference",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-09-30,2024-12-29,2024-12-29,seed_bundle_snapshot_conservative_90d_lag,index,1,none,unknown,false,reference,control,reference",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-12-31,2025-03-31,2025-03-31,seed_bundle_snapshot_conservative_90d_lag,index,2,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    _write_text(
        paths.bundles / "qrawatch" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-03-31,2024-02-07,2024-02-07,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-06-30,2024-05-07,2024-05-07,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-09-30,2024-08-07,2024-08-07,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
                "ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,2024-12-31,2024-11-07,2024-11-07,official_qra_release_timestamp,usd_billions,5,none,unknown,false,debt_management_treatment,treatment,exact_official_numeric",
            ]
        ),
    )
    _write_text(
        paths.bundles / "qrawatch" / "event_bundle.csv",
        "\n".join(
            [
                "event_id,quarter,event_label,event_date,event_type,source_repo,treatment_id,treatment_value,treatment_units,cutoff_timestamp,embargo_rule,horizon_unit,usable_for_headline,usable_for_headline_reason,usable_for_descriptive_headline,descriptive_headline_reason,quality_tier,claim_scope,shock_review_status,shock_missing_flag,small_denominator_flag",
                "qra_2024_01,2024Q1,2024 Jan QRA,2024-01-03,qra_release,qrawatch,canonical_shock_bn,10,usd_billions,2024-01-03T08:30:00-05:00,event_close_with_embargo,business_day,true,usable,true,usable,Tier A,headline,reviewed,false,false",
            ]
        ),
    )
    _write_text(paths.config / "debt_limit_intervals.csv", "quarter,start_date,end_date,source_note\n")
    _write_text(paths.raw_fred / "BOGZ1FL764100005Q.csv", "date,value\n2023-10-01,0\n2024-01-01,1\n2024-04-01,3\n2024-07-01,6\n2024-10-01,10\n")
    _write_text(paths.raw_fred / "GDP.csv", "date,value\n2024-03-31,1\n2024-06-30,1\n2024-09-30,1\n2024-12-31,1\n")
    _write_text(paths.raw_fred / "FEDFUNDS.csv", "date,value\n2024-01-01,1\n2024-04-01,1\n2024-07-01,1\n2024-10-01,1\n")
    _write_text(paths.raw_fred / "TOTRESNS.csv", "date,value\n2024-03-31,1\n2024-06-30,1\n2024-09-30,1\n2024-12-31,1\n")
    _write_text(paths.raw_fred / "THREEFYTP10.csv", "date,value\n2024-01-02,1.60\n2024-01-04,1.66\n")
    _write_text(paths.raw_fred / "DFF.csv", "date,value\n2024-01-02,5.33\n2024-01-04,5.38\n")

    result = build_release_artifact_contract(paths)

    assert result.committed_jobs == 2
    assert result.main_text_artifacts == 2
    assert result.appendix_artifacts == 1
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["committed_jobs"] == 2
    assert summary["main_text_artifacts"] == 2
    assert summary["appendix_artifacts"] == 1
    rows = {row["artifact_id"]: row for row in summary["rows"]}
    assert rows["main_figure_1"]["job_id"] == "custom_lp_job"
    assert rows["main_figure_1"]["display_spec"] == "impulse_response_grid"
    assert rows["main_table_1"]["release_channel"] == "main_text"
    assert rows["appendix_table_1"]["job_id"] == "custom_appendix_job"
    assert rows["appendix_table_1"]["display_spec"] == "supporting_table"


def test_build_robustness_snapshot_summarizes_control_universe_and_ladder(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)

    _write_text(
        paths.config / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: custom_lp_job",
                "    estimator: lp",
                "    freq: quarterly",
                "    treatment_id: tdc_bank_only_shock",
                "    outcomes: [matched_total_deposits, other_component_qoq]",
                "    horizons: [0, 1]",
                "    controls_explicit: [gdp_deflator]",
                "    output_family: headline_identified",
            ]
        ),
    )
    _write_text(
        paths.bundles / "tdcest" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2023-12-31,2024-03-30,2024-03-30,seed_bundle_snapshot_conservative_90d_lag,usd_millions,10,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,12,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-06-30,2024-09-28,2024-09-28,seed_bundle_snapshot_conservative_90d_lag,usd_millions,15,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-09-30,2024-12-29,2024-12-29,seed_bundle_snapshot_conservative_90d_lag,usd_millions,18,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,2024-12-31,2025-03-31,2025-03-31,seed_bundle_snapshot_conservative_90d_lag,usd_millions,20,none,unknown,false,canonical_headline,treatment,primary_treatment",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2023-12-31,2024-03-30,2024-03-30,seed_bundle_snapshot_conservative_90d_lag,index,1,none,unknown,false,reference,control,reference",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-03-31,2024-06-29,2024-06-29,seed_bundle_snapshot_conservative_90d_lag,index,2,none,unknown,false,reference,control,reference",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-06-30,2024-09-28,2024-09-28,seed_bundle_snapshot_conservative_90d_lag,index,3,none,unknown,false,reference,control,reference",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-09-30,2024-12-29,2024-12-29,seed_bundle_snapshot_conservative_90d_lag,index,4,none,unknown,false,reference,control,reference",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-12-31,2025-03-31,2025-03-31,seed_bundle_snapshot_conservative_90d_lag,index,5,none,unknown,false,reference,control,reference",
            ]
        ),
    )
    _write_text(paths.raw_fred / "BOGZ1FL764100005Q.csv", "date,value\n2023-07-01,900\n2023-10-01,950\n2024-01-01,1000\n2024-04-01,1060\n2024-07-01,1120\n2024-10-01,1180\n")
    _write_text(paths.raw_fred / "GDP.csv", "date,value\n2023-12-31,1\n2024-03-31,1.1\n2024-06-30,1.2\n2024-09-30,1.3\n2024-12-31,1.4\n")
    _write_text(paths.raw_fred / "FEDFUNDS.csv", "date,value\n2023-10-01,1\n2024-01-01,1.1\n2024-04-01,1.2\n2024-07-01,1.3\n2024-10-01,1.4\n")
    _write_text(paths.raw_fred / "TOTRESNS.csv", "date,value\n2023-12-31,1\n2024-03-31,1.1\n2024-06-30,1.2\n2024-09-30,1.3\n2024-12-31,1.4\n")
    _write_text(paths.seed / "interpol" / "raw" / "daily_liquidity.csv", "date,value\n2023-09-29,1\n2023-12-29,2\n2024-03-29,3\n2024-06-28,4\n2024-09-30,5\n2024-12-31,6\n")
    _write_text(paths.seed / "interpol" / "raw" / "monthly_unrate.csv", "date,value\n2023-09-30,3.8\n2023-10-31,3.8\n2023-11-30,3.7\n2023-12-31,3.7\n2024-01-31,3.8\n2024-02-29,3.9\n2024-03-31,3.9\n2024-04-30,4.0\n2024-05-31,4.0\n2024-06-30,4.1\n2024-07-31,4.1\n2024-08-31,4.2\n2024-09-30,4.2\n2024-10-31,4.1\n2024-11-30,4.1\n2024-12-31,4.0\n")
    _write_text(paths.seed / "interpol" / "raw" / "quarterly_psave.csv", "date,value\n2023-09-30,4.0\n2023-12-31,4.1\n2024-03-31,4.3\n2024-06-30,4.2\n2024-09-30,4.1\n2024-12-31,4.0\n")

    result = build_robustness_snapshot(paths, job_ids=["custom_lp_job"])

    assert result.jobs_summarized == 1
    assert result.feature_count > 0
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["control_universe"]["feature_count"] > 0
    assert summary["control_universe"]["series_count"] == 3
    assert summary["rows"][0]["job_id"] == "custom_lp_job"
    assert summary["rows"][0]["recommended_k"] in {0, 100, 200, 300}
    assert summary["rows"][0]["ladder_rows"]
    assert summary["rows"][0]["ml_public_branch"] in {"none", "dml", "forest", "tmle"}
    assert summary["rows"][0]["ml_public_branch_label"]
    assert summary["rows"][0]["negative_controls"]["signal"] in {"quiet", "mixed", "cautionary"}
