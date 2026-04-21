from __future__ import annotations

import csv
import json
from pathlib import Path

from ea_tdc.paths import ensure_repo_dirs, project_paths
from ea_tdc.robustness import _select_recommended_k, build_control_universe, build_quarterly_robustness


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_control_universe_creates_mixed_frequency_lagged_panel(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    _write_text(
        paths.bundles / "designs" / "baseline_tdc_lp_deposits__quarterly_bundle.csv",
        "\n".join(
            [
                "quarter",
                "2020Q1",
                "2020Q2",
                "2020Q3",
                "2020Q4",
            ]
        ),
    )
    _write_text(
        paths.seed / "interpol" / "raw" / "FRED_DGS10_rate.csv",
        "\n".join(
            ["date,value"]
            + [f"2019-10-{day:02d},{1 + day / 100:.2f}" for day in range(1, 21)]
            + ["2019-12-31,1.3", "2020-03-31,1.4"]
        ),
    )
    _write_text(
        paths.seed / "interpol" / "raw" / "FRED_UNRATE_unrate.csv",
        "\n".join(
            [
                "date,value",
                "2019-09-30,3.5",
                "2019-10-31,3.6",
                "2019-11-30,3.7",
                "2019-12-31,3.8",
                "2020-01-31,3.9",
            ]
        ),
    )
    _write_text(
        paths.seed / "interpol" / "raw" / "FRED_GDP_gdp.csv",
        "\n".join(
            [
                "date,value",
                "2019-09-30,100",
                "2019-12-31,101",
                "2020-03-31,102",
            ]
        ),
    )

    result = build_control_universe(paths)
    assert result.feature_count == 110

    with result.panel_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["d__dgs10_rate__lag001"] == "1.3"
    assert rows[0]["m__unrate_unrate__lag001"] == "3.8"
    assert rows[1]["q__gdp_gdp__lag001"] == "101.0"


def test_select_recommended_k_prefers_smallest_near_best_factor_screen() -> None:
    recommended_k, reason = _select_recommended_k(
        [
            {"k_screened": 100, "rows_written": 10, "warning_rows": 0, "weak_instrument_rows": 0, "avg_abs_beta": 0.245351},
            {"k_screened": 200, "rows_written": 10, "warning_rows": 0, "weak_instrument_rows": 0, "avg_abs_beta": 0.264495},
            {"k_screened": 300, "rows_written": 10, "warning_rows": 0, "weak_instrument_rows": 0, "avg_abs_beta": 0.278542},
        ]
    )

    assert recommended_k == 200
    assert "smallest factor-screened branch" in reason


def test_build_quarterly_robustness_writes_ladder_regime_and_treatment_reports(tmp_path: Path) -> None:
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
                "    treatment_id: tdc_bank_only_shock",
                "    outcomes: [matched_total_deposits, other_component_qoq]",
                "    horizons: [0, 1]",
                "    output_family: headline_identified",
            ]
        ),
    )

    bundle_lines = ["quarter,tdc_bank_only_qoq,matched_total_deposits,other_component_qoq,GDP,gdp_deflator,FEDFUNDS,TOTRESNS,coord_low_reserve_state_l1"]
    design_rows = []
    for idx in range(1, 33):
        year = 2015 + ((idx - 1) // 4)
        quarter_num = ((idx - 1) % 4) + 1
        quarter = f"{year}Q{quarter_num}"
        treatment = float(idx)
        matched = 2.0 * treatment + (0.5 if idx % 2 else -0.25)
        other = -1.0 * treatment + (0.25 if idx % 3 else -0.5)
        gdp = 100.0 + idx
        bundle_lines.append(
            f"{quarter},{treatment},{matched},{other},{gdp},{100 + idx / 10:.3f},{1 + idx / 100:.3f},{50 + idx:.3f},{1 if idx % 2 else 0}"
        )
        design_rows.append(quarter)
    _write_text(paths.bundles / "designs" / "baseline_tdc_lp_deposits__quarterly_bundle.csv", "\n".join(bundle_lines))
    _write_text(
        paths.manifests / "baseline_tdc_lp_deposits__design_manifest.json",
        json.dumps(
            {
                "job_id": "baseline_tdc_lp_deposits",
                "status": "ready_for_estimation",
                "bundle_path": str(paths.bundles / "designs" / "baseline_tdc_lp_deposits__quarterly_bundle.csv"),
                "treatment_id": "tdc_bank_only_qoq",
                "instrument_ids": ["qra_ati_baseline_bn", "qra_net_bills_bn", "qra_bill_share"],
                "outcome_ids": ["matched_total_deposits", "other_component_qoq"],
                "control_ids": ["GDP", "gdp_deflator", "FEDFUNDS", "TOTRESNS"],
                "horizon_grid": [0, 1],
            }
        ),
    )

    tdcest_lines = [
        "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes"
    ]
    for idx, quarter in enumerate(design_rows, start=1):
        year = int(quarter[:4])
        q = int(quarter[-1])
        month = q * 3
        day = 31 if month in {3, 12} else 30
        period_end = f"{year}-{month:02d}-{day:02d}"
        tdcest_lines.append(
            f"tdc_base_broad_depository_np_cu_ru_flow,alt,repo_seed_bundle,tdcest,estimates,quarterly,{period_end},{period_end},{period_end},seed,usd,{idx * 1.1},none,false,false,canonical_headline,treatment,fixture"
        )
        tdcest_lines.append(
            f"tdc_tier2_interest_corrected_bank_only_ru_flow,alt,repo_seed_bundle,tdcest,estimates,quarterly,{period_end},{period_end},{period_end},seed,usd,{idx * 1.02},none,false,false,estimate_variant,treatment,fixture"
        )
        tdcest_lines.append(
            f"tdc_tier3_fiscal_corrected_bank_only_ru_flow,alt,repo_seed_bundle,tdcest,estimates,quarterly,{period_end},{period_end},{period_end},seed,usd,{idx * 0.98},none,false,false,estimate_variant,treatment,fixture"
        )
        tdcest_lines.append(
            f"tdc_no_remit_bank_only,alt,repo_seed_bundle,tdcest,estimates,quarterly,{period_end},{period_end},{period_end},seed,usd,{idx * 0.9},none,false,false,canonical_headline,treatment,fixture"
        )
    _write_text(paths.bundles / "tdcest" / "standardized_series.csv", "\n".join(tdcest_lines))

    _write_text(
        paths.seed / "interpol" / "raw" / "FRED_DGS10_rate.csv",
        "\n".join(
            ["date,value"]
            + [f"2014-01-{(idx % 28) + 1:02d},{2 + idx / 100:.4f}" for idx in range(1, 120)]
            + [f"2015-{((idx - 1) % 12) + 1:02d}-{((idx - 1) % 27) + 1:02d},{3 + idx / 200:.4f}" for idx in range(1, 120)]
        ),
    )
    _write_text(
        paths.seed / "interpol" / "raw" / "FRED_UNRATE_unrate.csv",
        "\n".join(
            ["date,value"]
            + [f"2014-{((idx - 1) % 12) + 1:02d}-28,{4 + idx / 100:.4f}" for idx in range(1, 80)]
        ),
    )
    _write_text(
        paths.seed / "interpol" / "raw" / "FRED_GDP_gdp.csv",
        "\n".join(
            ["date,value"]
            + [f"{2013 + ((idx - 1) // 4)}-{(((idx - 1) % 4) + 1) * 3:02d}-30,{200 + idx:.4f}" for idx in range(1, 40)]
        ),
    )

    result = build_quarterly_robustness(paths, job_id="baseline_tdc_lp_deposits", k_grid=[100, 200], factor_count=2)

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["recommended_k"] in {100, 200}
    assert summary["recommended_factor_count"] == 2
    assert "smallest factor-screened branch" in summary["recommended_k_reason"]
    assert summary["screened_feature_count"] > 0

    with result.ladder_path.open("r", encoding="utf-8", newline="") as handle:
        ladder_rows = list(csv.DictReader(handle))
    assert any(row["run_type"] == "baseline_core" for row in ladder_rows)
    assert any(row["k_screened"] == "100" for row in ladder_rows)
    assert any(row["k_screened"] == "200" for row in ladder_rows)

    with result.regime_path.open("r", encoding="utf-8", newline="") as handle:
        regime_rows = list(csv.DictReader(handle))
    assert any(row["regime_id"] == "coord_low_reserve_state_l1" for row in regime_rows)

    with result.treatment_path.open("r", encoding="utf-8", newline="") as handle:
        treatment_rows = list(csv.DictReader(handle))
    assert any(row["treatment_variant"] == "tdc_base_broad_depository_np_cu_ru_flow" for row in treatment_rows)
    assert any(row["treatment_variant"] == "tdc_tier3_fiscal_corrected_bank_only_ru_flow" for row in treatment_rows)
