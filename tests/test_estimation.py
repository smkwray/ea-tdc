from __future__ import annotations

import csv
import json
from pathlib import Path

from ea_tdc.designs.quarterly import build_quarterly_design
from ea_tdc.estimation import build_estimation_snapshot, estimate_job, estimate_quarterly_job
from ea_tdc.paths import ensure_repo_dirs, project_paths


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_estimate_quarterly_job_writes_reference_comparison(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    _write_text(
        paths.config / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: tdcpass_identity_baseline_deposits",
                "    estimator: lp",
                "    freq: quarterly",
                "    treatment_id: tdcpass_tdc_residual_z",
                "    outcomes: [tdcpass_tdc_bank_only_qoq, tdcpass_total_deposits_bank_qoq]",
                "    controls_explicit: [tdcpass_lag_tdc_bank_only_qoq]",
                "    horizons: [0, 1]",
                "    response_type: cumulative_sum_h0_to_h",
                "    output_family: supporting_reduced_form",
                "    published_reference_artifact: data/bundles/tdcpass/published_identity_baseline.csv",
                "    published_reference_outcome_prefix: tdcpass_",
                "    track_in_release_snapshot: false",
                "    track_in_estimation_snapshot: false",
            ]
        ),
    )
    _write_text(
        paths.bundles / "tdcpass" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdcpass_tdc_bank_only_qoq,tdc_bank_only_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-03-31,2024-06-29,2024-06-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,1,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_total_deposits_bank_qoq,total_deposits_bank_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-03-31,2024-06-29,2024-06-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,2,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_lag_tdc_bank_only_qoq,lag_tdc_bank_only_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-03-31,2024-06-29,2024-06-29,tdcpass_publish_snapshot_conservative_90d_lag,percent,2,none,unknown,false,published_panel,reference,fixture",
                "tdcpass_tdc_residual_z,tdc_residual_z,repo_seed_bundle,tdcpass,unexpected_tdc,quarterly,2024-03-31,2024-06-29,2024-06-29,tdcpass_publish_snapshot_conservative_90d_lag,ratio,1,none,unknown,false,published_shock,treatment,fixture",
                "tdcpass_tdc_bank_only_qoq,tdc_bank_only_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-06-30,2024-09-28,2024-09-28,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,2,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_total_deposits_bank_qoq,total_deposits_bank_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-06-30,2024-09-28,2024-09-28,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,4,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_lag_tdc_bank_only_qoq,lag_tdc_bank_only_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-06-30,2024-09-28,2024-09-28,tdcpass_publish_snapshot_conservative_90d_lag,percent,1,none,unknown,false,published_panel,reference,fixture",
                "tdcpass_tdc_residual_z,tdc_residual_z,repo_seed_bundle,tdcpass,unexpected_tdc,quarterly,2024-06-30,2024-09-28,2024-09-28,tdcpass_publish_snapshot_conservative_90d_lag,ratio,2,none,unknown,false,published_shock,treatment,fixture",
                "tdcpass_tdc_bank_only_qoq,tdc_bank_only_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-09-30,2024-12-29,2024-12-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,3,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_total_deposits_bank_qoq,total_deposits_bank_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-09-30,2024-12-29,2024-12-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,6,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_lag_tdc_bank_only_qoq,lag_tdc_bank_only_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-09-30,2024-12-29,2024-12-29,tdcpass_publish_snapshot_conservative_90d_lag,percent,3,none,unknown,false,published_panel,reference,fixture",
                "tdcpass_tdc_residual_z,tdc_residual_z,repo_seed_bundle,tdcpass,unexpected_tdc,quarterly,2024-09-30,2024-12-29,2024-12-29,tdcpass_publish_snapshot_conservative_90d_lag,ratio,3,none,unknown,false,published_shock,treatment,fixture",
                "tdcpass_tdc_bank_only_qoq,tdc_bank_only_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-12-31,2025-03-31,2025-03-31,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,4,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_total_deposits_bank_qoq,total_deposits_bank_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-12-31,2025-03-31,2025-03-31,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,8,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_lag_tdc_bank_only_qoq,lag_tdc_bank_only_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-12-31,2025-03-31,2025-03-31,tdcpass_publish_snapshot_conservative_90d_lag,percent,1,none,unknown,false,published_panel,reference,fixture",
                "tdcpass_tdc_residual_z,tdc_residual_z,repo_seed_bundle,tdcpass,unexpected_tdc,quarterly,2024-12-31,2025-03-31,2025-03-31,tdcpass_publish_snapshot_conservative_90d_lag,ratio,4,none,unknown,false,published_shock,treatment,fixture",
                "tdcpass_tdc_bank_only_qoq,tdc_bank_only_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2025-03-31,2025-06-29,2025-06-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,5,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_total_deposits_bank_qoq,total_deposits_bank_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2025-03-31,2025-06-29,2025-06-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,10,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_lag_tdc_bank_only_qoq,lag_tdc_bank_only_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2025-03-31,2025-06-29,2025-06-29,tdcpass_publish_snapshot_conservative_90d_lag,percent,4,none,unknown,false,published_panel,reference,fixture",
                "tdcpass_tdc_residual_z,tdc_residual_z,repo_seed_bundle,tdcpass,unexpected_tdc,quarterly,2025-03-31,2025-06-29,2025-06-29,tdcpass_publish_snapshot_conservative_90d_lag,ratio,5,none,unknown,false,published_shock,treatment,fixture",
            ]
        ),
    )
    _write_text(
        paths.bundles / "tdcpass" / "published_identity_baseline.csv",
        "\n".join(
            [
                "outcome,horizon,beta",
                "tdc_bank_only_qoq,0,0.9",
                "tdc_bank_only_qoq,1,2.0",
                "total_deposits_bank_qoq,0,1.8",
                "total_deposits_bank_qoq,1,4.0",
            ]
        ),
    )

    build_quarterly_design(paths, job_id="tdcpass_identity_baseline_deposits")
    result = estimate_quarterly_job(paths, job_id="tdcpass_identity_baseline_deposits")

    with result.estimates_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0]["response_type"] == "cumulative_sum_h0_to_h"
    assert {row["inference_method"] for row in rows} == {"ols_newey_west_scaffold"}
    assert {row["covariance_estimator"] for row in rows} == {"newey_west"}

    comparison = json.loads(result.comparison_path.read_text(encoding="utf-8"))
    assert comparison["matched_rows"] == 4
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["covariance_estimators_used"] == ["newey_west"]
    assert summary["warning_rows"] == 0


def test_estimate_job_quarterly_lp_drops_collinear_controls(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    _write_text(
        paths.config / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: custom_quarterly_collinear",
                "    estimator: lp",
                "    freq: quarterly",
                "    treatment_id: tdc_bank_only_qoq",
                "    outcomes: [matched_total_deposits]",
                "    controls_explicit: [ctrl, ctrl_duplicate]",
                "    horizons: [0]",
                "    response_type: direct_at_h",
                "    output_family: supporting_reduced_form",
            ]
        ),
    )
    bundle_path = paths.bundles / "designs" / "custom_quarterly_collinear__quarterly_bundle.csv"
    _write_text(
        bundle_path,
        "\n".join(
            [
                "quarter,tdc_bank_only_qoq,ctrl,ctrl_duplicate,matched_total_deposits",
                "2023Q1,1.0,1.0,1.0,2.0",
                "2023Q2,2.0,0.0,0.0,3.1",
                "2023Q3,3.0,1.0,1.0,4.1",
                "2023Q4,4.0,0.0,0.0,5.2",
                "2024Q1,5.0,1.0,1.0,6.0",
                "2024Q2,6.0,0.0,0.0,7.3",
            ]
        ),
    )
    _write_text(
        paths.manifests / "custom_quarterly_collinear__design_manifest.json",
        json.dumps(
            {
                "job_id": "custom_quarterly_collinear",
                "status": "ready_for_estimation",
                "bundle_path": str(bundle_path),
                "treatment_id": "tdc_bank_only_qoq",
                "instrument_ids": [],
                "control_ids": ["ctrl", "ctrl_duplicate"],
                "outcome_ids": ["matched_total_deposits"],
                "horizon_grid": [0],
                "response_type": "direct_at_h",
            }
        ),
    )

    result = estimate_job(paths, job_id="custom_quarterly_collinear")

    with result.estimates_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["control_ids_used"] == "ctrl"
    assert rows[0]["dropped_control_ids"] == "ctrl_duplicate"
    assert rows[0]["inference_method"] == "ols_newey_west_scaffold_adaptive_controls"
    assert "adaptive_controls" in rows[0]["warning_flags"]


def test_estimate_quarterly_job_supports_tdcpass_strict_source_side_job(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    _write_text(
        paths.config / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: tdcpass_strict_source_side_nontdc",
                "    estimator: lp",
                "    freq: quarterly",
                "    treatment_id: tdcpass_tdc_residual_z",
                "    outcomes: [tdcpass_other_component_qoq, tdcpass_strict_loan_core_min_qoq, tdcpass_strict_non_treasury_securities_qoq, tdcpass_strict_identifiable_total_qoq, tdcpass_strict_identifiable_gap_qoq]",
                "    controls_explicit: [tdcpass_lag_tdc_bank_only_qoq]",
                "    horizons: [0, 1]",
                "    response_type: cumulative_sum_h0_to_h",
                "    output_family: supporting_reduced_form",
                "    track_in_release_snapshot: false",
                "    track_in_estimation_snapshot: false",
            ]
        ),
    )
    _write_text(
        paths.bundles / "tdcpass" / "standardized_series.csv",
        "\n".join(
            [
                "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes",
                "tdcpass_other_component_qoq,other_component_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-03-31,2024-06-29,2024-06-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,2,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_strict_loan_core_min_qoq,strict_loan_core_min_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-03-31,2024-06-29,2024-06-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,1,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_strict_non_treasury_securities_qoq,strict_non_treasury_securities_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-03-31,2024-06-29,2024-06-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,0.2,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_strict_identifiable_total_qoq,strict_identifiable_total_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-03-31,2024-06-29,2024-06-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,1.2,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_strict_identifiable_gap_qoq,strict_identifiable_gap_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-03-31,2024-06-29,2024-06-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,0.8,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_lag_tdc_bank_only_qoq,lag_tdc_bank_only_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-03-31,2024-06-29,2024-06-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,1,none,unknown,false,published_panel,reference,fixture",
                "tdcpass_tdc_residual_z,tdc_residual_z,repo_seed_bundle,tdcpass,unexpected_tdc,quarterly,2024-03-31,2024-06-29,2024-06-29,tdcpass_publish_snapshot_conservative_90d_lag,ratio,1,none,unknown,false,published_shock,treatment,fixture",
                "tdcpass_other_component_qoq,other_component_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-06-30,2024-09-28,2024-09-28,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,4,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_strict_loan_core_min_qoq,strict_loan_core_min_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-06-30,2024-09-28,2024-09-28,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,2,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_strict_non_treasury_securities_qoq,strict_non_treasury_securities_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-06-30,2024-09-28,2024-09-28,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,0.4,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_strict_identifiable_total_qoq,strict_identifiable_total_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-06-30,2024-09-28,2024-09-28,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,2.4,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_strict_identifiable_gap_qoq,strict_identifiable_gap_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-06-30,2024-09-28,2024-09-28,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,1.6,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_lag_tdc_bank_only_qoq,lag_tdc_bank_only_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-06-30,2024-09-28,2024-09-28,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,4,none,unknown,false,published_panel,reference,fixture",
                "tdcpass_tdc_residual_z,tdc_residual_z,repo_seed_bundle,tdcpass,unexpected_tdc,quarterly,2024-06-30,2024-09-28,2024-09-28,tdcpass_publish_snapshot_conservative_90d_lag,ratio,2,none,unknown,false,published_shock,treatment,fixture",
                "tdcpass_other_component_qoq,other_component_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-09-30,2024-12-29,2024-12-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,6,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_strict_loan_core_min_qoq,strict_loan_core_min_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-09-30,2024-12-29,2024-12-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,3,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_strict_non_treasury_securities_qoq,strict_non_treasury_securities_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-09-30,2024-12-29,2024-12-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,0.6,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_strict_identifiable_total_qoq,strict_identifiable_total_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-09-30,2024-12-29,2024-12-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,3.6,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_strict_identifiable_gap_qoq,strict_identifiable_gap_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-09-30,2024-12-29,2024-12-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,2.4,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_lag_tdc_bank_only_qoq,lag_tdc_bank_only_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-09-30,2024-12-29,2024-12-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,2,none,unknown,false,published_panel,reference,fixture",
                "tdcpass_tdc_residual_z,tdc_residual_z,repo_seed_bundle,tdcpass,unexpected_tdc,quarterly,2024-09-30,2024-12-29,2024-12-29,tdcpass_publish_snapshot_conservative_90d_lag,ratio,3,none,unknown,false,published_shock,treatment,fixture",
                "tdcpass_other_component_qoq,other_component_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-12-31,2025-03-31,2025-03-31,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,8,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_strict_loan_core_min_qoq,strict_loan_core_min_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-12-31,2025-03-31,2025-03-31,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,4,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_strict_non_treasury_securities_qoq,strict_non_treasury_securities_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-12-31,2025-03-31,2025-03-31,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,0.8,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_strict_identifiable_total_qoq,strict_identifiable_total_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-12-31,2025-03-31,2025-03-31,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,4.8,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_strict_identifiable_gap_qoq,strict_identifiable_gap_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-12-31,2025-03-31,2025-03-31,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,3.2,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_lag_tdc_bank_only_qoq,lag_tdc_bank_only_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2024-12-31,2025-03-31,2025-03-31,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,5,none,unknown,false,published_panel,reference,fixture",
                "tdcpass_tdc_residual_z,tdc_residual_z,repo_seed_bundle,tdcpass,unexpected_tdc,quarterly,2024-12-31,2025-03-31,2025-03-31,tdcpass_publish_snapshot_conservative_90d_lag,ratio,4,none,unknown,false,published_shock,treatment,fixture",
                "tdcpass_other_component_qoq,other_component_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2025-03-31,2025-06-29,2025-06-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,10,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_strict_loan_core_min_qoq,strict_loan_core_min_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2025-03-31,2025-06-29,2025-06-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,5,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_strict_non_treasury_securities_qoq,strict_non_treasury_securities_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2025-03-31,2025-06-29,2025-06-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,1.0,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_strict_identifiable_total_qoq,strict_identifiable_total_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2025-03-31,2025-06-29,2025-06-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,6.0,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_strict_identifiable_gap_qoq,strict_identifiable_gap_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2025-03-31,2025-06-29,2025-06-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,4.0,none,unknown,false,published_panel,mechanism,fixture",
                "tdcpass_lag_tdc_bank_only_qoq,lag_tdc_bank_only_qoq,repo_seed_bundle,tdcpass,quarterly_panel,quarterly,2025-03-31,2025-06-29,2025-06-29,tdcpass_publish_snapshot_conservative_90d_lag,usd_billions,3,none,unknown,false,published_panel,reference,fixture",
                "tdcpass_tdc_residual_z,tdc_residual_z,repo_seed_bundle,tdcpass,unexpected_tdc,quarterly,2025-03-31,2025-06-29,2025-06-29,tdcpass_publish_snapshot_conservative_90d_lag,ratio,5,none,unknown,false,published_shock,treatment,fixture",
            ]
        ),
    )

    build_quarterly_design(paths, job_id="tdcpass_strict_source_side_nontdc")
    result = estimate_quarterly_job(paths, job_id="tdcpass_strict_source_side_nontdc")

    with result.estimates_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert {row["response_type"] for row in rows} == {"cumulative_sum_h0_to_h"}
    assert {row["covariance_estimator"] for row in rows} == {"newey_west"}


def test_build_estimation_snapshot_runs_ready_lp_jobs(tmp_path: Path) -> None:
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
                "    output_family: headline_identified",
                "  - job_id: skip_iv_job",
                "    estimator: lp_iv",
                "    freq: quarterly",
                "    treatment_id: tdc_bank_only_shock",
                "    outcomes: [matched_total_deposits]",
                "    horizons: [0]",
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
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-06-30,2024-09-28,2024-09-28,seed_bundle_snapshot_conservative_90d_lag,index,1,none,unknown,false,reference,control,reference",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-09-30,2024-12-29,2024-12-29,seed_bundle_snapshot_conservative_90d_lag,index,1,none,unknown,false,reference,control,reference",
                "gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,2024-12-31,2025-03-31,2025-03-31,seed_bundle_snapshot_conservative_90d_lag,index,1,none,unknown,false,reference,control,reference",
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

    build_quarterly_design(paths, job_id="custom_lp_job")
    snapshot = build_estimation_snapshot(paths)
    summary = json.loads(snapshot.summary_path.read_text(encoding="utf-8"))
    assert summary["jobs_estimated"] == 1
    assert summary["rows"][0]["sample_policy"] == ""


def test_estimate_job_supports_lp_iv_scaffold(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    _write_text(
        paths.config / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: custom_iv_job",
                "    estimator: lp_iv",
                "    freq: quarterly",
                "    treatment_id: tdc_bank_only_shock",
                "    instruments: [iv_custom]",
                "    outcomes: [matched_total_deposits]",
                "    horizons: [0, 1]",
                "    output_family: headline_identified",
            ]
        ),
    )
    bundle_path = paths.bundles / "designs" / "custom_iv_job__quarterly_bundle.csv"
    _write_text(
        bundle_path,
        "\n".join(
            [
                "quarter,tdc_bank_only_qoq,iv_custom,ctrl,matched_total_deposits",
                "2024Q1,1.0,0.8,1.0,2.1",
                "2024Q2,2.0,1.9,0.9,4.2",
                "2024Q3,3.0,2.7,1.1,6.0",
                "2024Q4,4.0,3.8,1.0,8.1",
                "2025Q1,5.0,4.9,1.2,10.2",
                "2025Q2,6.0,6.1,1.1,12.1",
            ]
        ),
    )
    _write_text(
        paths.manifests / "custom_iv_job__design_manifest.json",
        json.dumps(
            {
                "job_id": "custom_iv_job",
                "status": "ready_for_estimation",
                "bundle_path": str(bundle_path),
                "treatment_id": "tdc_bank_only_qoq",
                "instrument_ids": ["iv_custom"],
                "control_ids": ["ctrl"],
                "outcome_ids": ["matched_total_deposits"],
                "horizon_grid": [0, 1],
            }
        ),
    )

    result = estimate_job(paths, job_id="custom_iv_job")

    assert result.estimator == "lp_iv"
    with result.estimates_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["inference_method"] for row in rows} == {"two_stage_least_squares_hc1_scaffold"}
    assert rows[0]["instrument_ids"] == "iv_custom"
    assert rows[0]["covariance_estimator"] == "hc1"
    assert float(rows[0]["first_stage_f_excluded"]) > 10.0
    assert rows[0]["weak_instrument_flag"] == "false"


def test_estimate_job_supports_state_interaction_lp(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    _write_text(
        paths.config / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: custom_state_job",
                "    estimator: lp",
                "    freq: quarterly",
                "    treatment_id: tdc_bank_only_qoq",
                "    state_id: coord_low_reserve_state_l1",
                "    outcomes: [matched_total_deposits]",
                "    horizons: [0]",
                "    output_family: supporting_reduced_form",
            ]
        ),
    )
    bundle_path = paths.bundles / "designs" / "custom_state_job__quarterly_bundle.csv"
    _write_text(
        bundle_path,
        "\n".join(
            [
                "quarter,tdc_bank_only_qoq,coord_low_reserve_state_l1,matched_total_deposits",
                "2023Q1,1.0,0.0,11.0",
                "2023Q2,2.0,0.0,12.0",
                "2023Q3,3.0,0.0,13.0",
                "2023Q4,4.0,0.0,14.0",
                "2024Q1,5.0,1.0,25.0",
                "2024Q2,6.0,1.0,28.0",
                "2024Q3,7.0,1.0,31.0",
                "2024Q4,8.0,1.0,34.0",
            ]
        ),
    )
    _write_text(
        paths.manifests / "custom_state_job__design_manifest.json",
        json.dumps(
            {
                "job_id": "custom_state_job",
                "status": "ready_for_estimation",
                "bundle_path": str(bundle_path),
                "treatment_id": "tdc_bank_only_qoq",
                "instrument_ids": [],
                "control_ids": [],
                "outcome_ids": ["matched_total_deposits"],
                "horizon_grid": [0],
                "state_ids": ["coord_low_reserve_state_l1"],
            }
        ),
    )

    result = estimate_job(paths, job_id="custom_state_job")

    with result.estimates_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    by_profile = {row["state_profile"]: row for row in rows}
    assert set(by_profile) == {"low_state", "high_state"}
    assert rows[0]["state_id"] == "coord_low_reserve_state_l1"
    assert {row["inference_method"] for row in rows} == {"ols_newey_west_state_interaction_scaffold"}
    assert abs(float(by_profile["low_state"]["beta"]) - 1.0) < 1e-6
    assert abs(float(by_profile["high_state"]["beta"]) - 3.0) < 1e-6
    assert abs(float(by_profile["low_state"]["state_reference_value"]) - 0.0) < 1e-9
    assert abs(float(by_profile["high_state"]["state_reference_value"]) - 1.0) < 1e-9
    assert abs(float(by_profile["low_state"]["state_interaction_beta"]) - 2.0) < 1e-6

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["state_id"] == "coord_low_reserve_state_l1"
    assert summary["min_observations"] == 8
    assert summary["max_observations"] == 8


def test_estimate_job_supports_event_lp_adaptive_controls(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    _write_text(
        paths.config / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: custom_event_job",
                "    estimator: event_lp",
                "    freq: irregular_event",
                "    treatment_id: qra_release_63bd",
                "    outcomes: [threefytp10]",
                "    horizons_bd: [1, 5]",
                "    output_family: supporting_reduced_form",
            ]
        ),
    )
    bundle_path = paths.bundles / "designs" / "custom_event_job__event_panel.csv"
    _write_text(
        bundle_path,
        "\n".join(
            [
                "event_id,event_date,usable_for_headline,include_in_sample,treatment_value,debt_limit_dummy,delta_dff_release_plus_1bd,delta_threefytp10_h1bd,delta_threefytp10_h5bd",
                "evt1,2024-01-03,true,true,1.0,0,0.10,2.0,3.0",
                "evt2,2024-02-07,true,true,2.0,1,0.20,4.0,6.1",
                "evt3,2024-05-01,true,true,3.0,0,0.30,6.1,9.0",
                "evt4,2024-07-31,true,true,4.0,1,0.40,8.0,12.2",
            ]
        ),
    )
    _write_text(
        paths.manifests / "custom_event_job__design_manifest.json",
        json.dumps(
            {
                "job_id": "custom_event_job",
                "status": "ready_for_estimation",
                "bundle_path": str(bundle_path),
                "treatment_id": "qra_release_63bd",
                "instrument_ids": [],
                "control_ids": ["debt_limit_dummy", "delta_dff_release_plus_1bd"],
                "outcome_ids": ["threefytp10"],
                "horizon_grid": [1, 5],
                "sample_policy": "headline_strict",
                "event_sample_counts": {"requested_sample_rows": 4, "headline_eligible_rows": 4},
            }
        ),
    )

    result = estimate_job(paths, job_id="custom_event_job")

    assert result.estimator == "event_lp"
    with result.estimates_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["control_ids_used"] for row in rows} == {"debt_limit_dummy"}
    assert {row["covariance_estimator"] for row in rows} == {"hc1"}
    assert all("small_sample_event" in row["warning_flags"] for row in rows)
    by_horizon = {row["horizon"]: row for row in rows}
    assert "adaptive_controls" in by_horizon["1"]["inference_method"]
    assert by_horizon["5"]["inference_method"] == "event_ols_hc1_scaffold"
    assert by_horizon["1"]["dropped_control_ids"] == "delta_dff_release_plus_1bd"
    assert by_horizon["5"]["dropped_control_ids"] == ""

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["sample_policy"] == "headline_strict"
    assert summary["event_sample_counts"]["requested_sample_rows"] == 4


def test_estimate_job_event_lp_drops_collinear_controls(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    _write_text(
        paths.config / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: custom_event_collinear_job",
                "    estimator: event_lp",
                "    freq: irregular_event",
                "    treatment_id: qra_release_63bd",
                "    outcomes: [threefytp10]",
                "    horizons_bd: [1]",
                "    output_family: supporting_descriptive",
            ]
        ),
    )
    bundle_path = paths.bundles / "designs" / "custom_event_collinear_job__event_panel.csv"
    _write_text(
        bundle_path,
        "\n".join(
            [
                "event_id,event_date,include_in_sample,treatment_value,debt_limit_dummy,delta_dff_release_plus_1bd,delta_sofr_release_plus_1bd,delta_threefytp10_h1bd",
                "evt1,2024-01-03,true,0.0,0,0.10,0.10,1.0",
                "evt2,2024-02-07,true,1.0,1,0.20,0.20,2.2",
                "evt3,2024-05-01,true,0.0,0,0.30,0.30,1.5",
                "evt4,2024-07-31,true,1.0,1,0.40,0.40,2.8",
                "evt5,2024-11-06,true,0.0,0,0.50,0.50,1.7",
                "evt6,2025-02-05,true,1.0,1,0.60,0.60,3.0",
            ]
        ),
    )
    _write_text(
        paths.manifests / "custom_event_collinear_job__design_manifest.json",
        json.dumps(
            {
                "job_id": "custom_event_collinear_job",
                "status": "ready_for_estimation",
                "bundle_path": str(bundle_path),
                "treatment_id": "qra_release_63bd",
                "instrument_ids": [],
                "control_ids": ["debt_limit_dummy", "delta_dff_release_plus_1bd", "delta_sofr_release_plus_1bd"],
                "outcome_ids": ["threefytp10"],
                "horizon_grid": [1],
                "sample_policy": "reviewed_nonmissing",
                "event_sample_counts": {"requested_sample_rows": 6, "headline_eligible_rows": 4},
            }
        ),
    )

    result = estimate_job(paths, job_id="custom_event_collinear_job")

    with result.estimates_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert "adaptive_controls" in rows[0]["warning_flags"]
    assert "collinear_controls" in rows[0]["warning_flags"]
    assert rows[0]["control_ids_used"] == ""
    assert rows[0]["dropped_control_ids"] == "debt_limit_dummy,delta_dff_release_plus_1bd,delta_sofr_release_plus_1bd"


def test_estimate_job_event_lp_matches_controls_to_horizon(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    _write_text(
        paths.config / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: custom_event_horizon_controls",
                "    estimator: event_lp",
                "    freq: irregular_event",
                "    treatment_id: qra_release_63bd",
                "    outcomes: [threefytp10]",
                "    horizons_bd: [1, 5]",
                "    output_family: supporting_descriptive",
            ]
        ),
    )
    bundle_path = paths.bundles / "designs" / "custom_event_horizon_controls__event_panel.csv"
    _write_text(
        bundle_path,
        "\n".join(
            [
                "event_id,event_date,include_in_sample,treatment_value,debt_limit_dummy,delta_dff_release_plus_1bd,delta_dff_release_plus_5bd,delta_threefytp10_h1bd,delta_threefytp10_h5bd",
                "evt1,2024-01-03,true,1.0,0,0.13,1.05,2.0,3.1",
                "evt2,2024-02-07,true,2.0,1,0.19,1.27,4.1,6.0",
                "evt3,2024-05-01,true,3.0,0,0.18,1.31,6.0,9.2",
                "evt4,2024-07-31,true,4.0,1,0.29,1.36,8.2,12.0",
                "evt5,2024-11-06,true,5.0,0,0.37,1.58,10.1,15.1",
                "evt6,2025-02-05,true,6.0,1,0.35,1.63,12.0,18.2",
            ]
        ),
    )
    _write_text(
        paths.manifests / "custom_event_horizon_controls__design_manifest.json",
        json.dumps(
            {
                "job_id": "custom_event_horizon_controls",
                "status": "ready_for_estimation",
                "bundle_path": str(bundle_path),
                "treatment_id": "qra_release_63bd",
                "instrument_ids": [],
                "control_ids": ["debt_limit_dummy", "delta_dff_release_plus_1bd", "delta_dff_release_plus_5bd"],
                "outcome_ids": ["threefytp10"],
                "horizon_grid": [1, 5],
                "sample_policy": "reviewed_nonmissing",
                "event_sample_counts": {"requested_sample_rows": 6, "headline_eligible_rows": 4},
            }
        ),
    )

    result = estimate_job(paths, job_id="custom_event_horizon_controls")

    with result.estimates_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    by_horizon = {row["horizon"]: row for row in rows}
    assert by_horizon["1"]["control_ids_used"] == "debt_limit_dummy,delta_dff_release_plus_1bd"
    assert by_horizon["1"]["dropped_control_ids"] == ""
    assert by_horizon["5"]["control_ids_used"] == "debt_limit_dummy,delta_dff_release_plus_5bd"
    assert by_horizon["5"]["dropped_control_ids"] == ""
