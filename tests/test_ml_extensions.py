from __future__ import annotations

import csv
import json
from pathlib import Path

from ea_tdc.ml_extensions import (
    build_negative_control_mining,
    build_quarterly_dml,
    build_quarterly_forest,
    build_quarterly_tmle,
)
from ea_tdc.paths import ensure_repo_dirs, project_paths
from ea_tdc.robustness import build_quarterly_robustness


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _setup_quarterly_lp_fixture(tmp_path: Path) -> tuple[Path, str]:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    job_id = "baseline_tdc_lp_deposits"
    _write_text(
        paths.config / "dass_job_blueprint.yaml",
        "\n".join(
            [
                "jobs:",
                "  - job_id: baseline_tdc_lp_deposits",
                "    estimator: lp",
                "    freq: quarterly",
                "    treatment_id: tdc_bank_only_shock",
                "    outcomes: [matched_total_deposits, other_component_qoq]",
                "    horizons: [0, 1, 2]",
                "    output_family: headline_identified",
            ]
        ),
    )

    bundle_lines = [
        "quarter,tdc_bank_only_qoq,matched_total_deposits,other_component_qoq,GDP,gdp_deflator,FEDFUNDS,TOTRESNS,coord_low_reserve_state_l1,nc_credit_proxy,nc_price_proxy"
    ]
    design_quarters: list[str] = []
    for idx in range(1, 49):
        year = 2010 + ((idx - 1) // 4)
        quarter_num = ((idx - 1) % 4) + 1
        quarter = f"{year}Q{quarter_num}"
        design_quarters.append(quarter)
        treatment = float(idx) / 2.0
        matched = 1.5 * treatment + 0.1 * idx + (0.4 if idx % 2 else -0.3)
        other = -0.8 * treatment + 0.05 * idx + (0.3 if idx % 3 else -0.2)
        gdp = 100.0 + idx
        gdp_deflator = 95.0 + idx / 10.0
        fedfunds = 0.5 + idx / 100.0
        reserves = 40.0 + idx * 0.75
        low_reserve = 1 if idx % 5 in (0, 1) else 0
        nc_credit = 2.0 + ((idx % 7) - 3) * 0.23 + (0.17 if idx % 2 else -0.11)
        nc_price = 10.0 + ((idx * idx) % 11) * 0.19 + (0.13 if idx % 5 else -0.21)
        bundle_lines.append(
            ",".join(
                [
                    quarter,
                    f"{treatment:.4f}",
                    f"{matched:.4f}",
                    f"{other:.4f}",
                    f"{gdp:.4f}",
                    f"{gdp_deflator:.4f}",
                    f"{fedfunds:.4f}",
                    f"{reserves:.4f}",
                    str(low_reserve),
                    f"{nc_credit:.4f}",
                    f"{nc_price:.4f}",
                ]
            )
        )
    bundle_path = paths.bundles / "designs" / f"{job_id}__quarterly_bundle.csv"
    _write_text(bundle_path, "\n".join(bundle_lines))

    _write_text(
        paths.manifests / f"{job_id}__design_manifest.json",
        json.dumps(
            {
                "job_id": job_id,
                "status": "ready_for_estimation",
                "bundle_path": str(bundle_path),
                "treatment_id": "tdc_bank_only_qoq",
                "instrument_ids": ["qra_ati_baseline_bn", "qra_net_bills_bn", "qra_bill_share"],
                "outcome_ids": ["matched_total_deposits", "other_component_qoq"],
                "control_ids": ["GDP", "gdp_deflator", "FEDFUNDS", "TOTRESNS"],
                "horizon_grid": [0, 1, 2],
            }
        ),
    )

    tdcest_lines = [
        "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes"
    ]
    for idx, quarter in enumerate(design_quarters, start=1):
        year = int(quarter[:4])
        q = int(quarter[-1])
        month = q * 3
        day = 31 if month in {3, 12} else 30
        period_end = f"{year}-{month:02d}-{day:02d}"
        tdcest_lines.append(
            f"tdc_base_broad_depository_np_cu_ru_flow,alt,repo_seed_bundle,tdcest,estimates,quarterly,{period_end},{period_end},{period_end},seed,usd,{idx * 1.15:.4f},none,false,false,canonical_headline,treatment,fixture"
        )
        tdcest_lines.append(
            f"tdc_no_remit_bank_only,alt,repo_seed_bundle,tdcest,estimates,quarterly,{period_end},{period_end},{period_end},seed,usd,{idx * 0.92:.4f},none,false,false,canonical_headline,treatment,fixture"
        )
        tdcest_lines.append(
            f"tdc_domestic_bank_only_ru_flow,alt,repo_seed_bundle,tdcest,estimates,quarterly,{period_end},{period_end},{period_end},seed,usd,{idx * 0.88:.4f},none,false,false,canonical_headline,treatment,fixture"
        )
        tdcest_lines.append(
            f"tdc_bank_only_extended_1990,alt,repo_seed_bundle,tdcest,estimates,quarterly,{period_end},{period_end},{period_end},seed,usd,{idx * 1.05:.4f},none,false,false,canonical_headline,treatment,fixture"
        )
    _write_text(paths.bundles / "tdcest" / "standardized_series.csv", "\n".join(tdcest_lines))

    daily_lines = ["date,value"]
    for idx in range(1, 220):
        month = ((idx - 1) % 12) + 1
        day = ((idx - 1) % 27) + 1
        daily_lines.append(f"2010-{month:02d}-{day:02d},{2.0 + idx / 250.0:.4f}")
    _write_text(paths.seed / "interpol" / "raw" / "FRED_DGS10_rate.csv", "\n".join(daily_lines))

    monthly_lines = ["date,value"]
    for idx in range(1, 160):
        month = ((idx - 1) % 12) + 1
        monthly_lines.append(f"{2009 + ((idx - 1) // 12)}-{month:02d}-28,{4.0 + idx / 180.0:.4f}")
    _write_text(paths.seed / "interpol" / "raw" / "FRED_UNRATE_unrate.csv", "\n".join(monthly_lines))

    quarterly_lines = ["date,value"]
    for idx in range(1, 80):
        month = (((idx - 1) % 4) + 1) * 3
        quarterly_lines.append(f"{2008 + ((idx - 1) // 4)}-{month:02d}-30,{200.0 + idx:.4f}")
    _write_text(paths.seed / "interpol" / "raw" / "FRED_GDP_gdp.csv", "\n".join(quarterly_lines))
    return tmp_path, job_id


def test_build_quarterly_dml_writes_estimates_and_summary(tmp_path: Path) -> None:
    repo_root, job_id = _setup_quarterly_lp_fixture(tmp_path)
    paths = project_paths(repo_root)

    build_quarterly_robustness(paths, job_id=job_id, k_grid=[100, 200], factor_count=2)
    result = build_quarterly_dml(paths, job_id=job_id, fold_count=3, ridge_alpha=0.5)

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["job_id"] == job_id
    assert summary["rows_written"] > 0
    assert summary["recommended_k"] in {100, 200}
    assert sum(1 for control_id in summary["control_ids"] if str(control_id).startswith("dflmx_k")) == 2

    with result.estimates_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(row["inference_method"] == "dml_crossfit_ridge_newey_west" for row in rows)
    assert any(row["outcome"] == "matched_total_deposits" for row in rows)


def test_build_negative_control_mining_writes_placebo_reports(tmp_path: Path) -> None:
    repo_root, job_id = _setup_quarterly_lp_fixture(tmp_path)
    paths = project_paths(repo_root)

    build_quarterly_robustness(paths, job_id=job_id, k_grid=[100, 200], factor_count=2)
    result = build_negative_control_mining(paths, job_id=job_id, top_n=5)

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["job_id"] == job_id
    assert summary["lead_placebos"]
    assert summary["recommended_branch_candidate_count"] >= 0
    assert summary["baseline_branch_candidate_count"] >= 0
    assert "top_clean_candidates" in summary
    assert "most_responsive_candidates" in summary
    assert all("lead" in item for item in summary["lead_placebos"])

    with result.summary_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)
    assert {"candidate_outcome", "control_branch"}.issubset(fieldnames)
    assert rows == [] or {"candidate_outcome", "control_branch"}.issubset(rows[0].keys())


def test_build_quarterly_forest_writes_estimates_and_summary(tmp_path: Path) -> None:
    repo_root, job_id = _setup_quarterly_lp_fixture(tmp_path)
    paths = project_paths(repo_root)

    build_quarterly_robustness(paths, job_id=job_id, k_grid=[100, 200], factor_count=2)
    result = build_quarterly_forest(
        paths,
        job_id=job_id,
        fold_count=3,
        tree_count=12,
        max_depth=2,
        min_leaf=4,
        feature_fraction=0.6,
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["job_id"] == job_id
    assert summary["rows_written"] > 0
    assert summary["tree_count"] == 12

    with result.estimates_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(row["inference_method"] == "forest_crossfit_newey_west" for row in rows)


def test_build_quarterly_tmle_writes_estimates_and_summary(tmp_path: Path) -> None:
    repo_root, job_id = _setup_quarterly_lp_fixture(tmp_path)
    paths = project_paths(repo_root)

    build_quarterly_robustness(paths, job_id=job_id, k_grid=[100, 200], factor_count=2)
    result = build_quarterly_tmle(paths, job_id=job_id, fold_count=3, ridge_alpha=0.75)

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["job_id"] == job_id
    assert summary["rows_written"] > 0
    assert summary["ridge_alpha"] == 0.75
    assert summary["avg_tmle_theta_init"] is not None
    assert summary["avg_tmle_theta_init_std"] is not None
    assert summary["avg_tmle_raw_epsilon"] is not None
    assert summary["avg_tmle_low_density_share"] is not None
    assert summary["avg_tmle_epsilon_theta_ratio"] is not None
    assert "valid_row_count" in summary
    assert "invalid_row_count" in summary
    assert "tmle_warning_counts" in summary
    assert 0.0 <= float(summary["epsilon_clip_rate"]) <= 1.0

    with result.estimates_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(row["inference_method"] == "tmle_crossfit_ridge_newey_west" for row in rows)
    assert all("tmle_theta_init" in row for row in rows)
    assert all("tmle_low_density_share" in row for row in rows)
    assert all("tmle_valid" in row for row in rows)
