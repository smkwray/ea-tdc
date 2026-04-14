from __future__ import annotations

import json
from pathlib import Path

from ea_tdc.iv_lab import build_iv_lab
from ea_tdc.paths import ensure_repo_dirs, project_paths


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_iv_lab_ranks_current_candidate_when_strongest(tmp_path: Path) -> None:
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
                "    instruments: [iv_qra_x_ru_gap]",
                "    controls_explicit: [gdp_deflator]",
                "    outcomes: [matched_total_deposits]",
                "    horizons: [0]",
                "    output_family: headline_identified",
            ]
        ),
    )

    tdc_rows = [
        "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes"
    ]
    qra_rows = [
        "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes"
    ]
    tsy_rows = [
        "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes"
    ]
    coord_rows = [
        "series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes"
    ]
    fred_levels = ["date,value"]
    for quarter_end, idx in [
        ("2023-03-31", 1),
        ("2023-06-30", 2),
        ("2023-09-30", 3),
        ("2023-12-31", 4),
        ("2024-03-31", 5),
        ("2024-06-30", 6),
        ("2024-09-30", 7),
        ("2024-12-31", 8),
    ]:
        release_date = quarter_end
        qra = float(idx)
        ru_gap = float((idx % 3) + (idx / 2.0) + 1.0)
        bank_absorption = 0.1 * idx
        row_absorption = 0.05 * idx
        low_reserve = 1.0 if idx % 2 == 0 else 0.0
        treatment = qra * ru_gap
        deposit_level = float(idx * idx + 100.0)
        gdp_deflator = 1.0 if idx % 2 else 2.0
        tdc_rows.append(
            f"tdc_base_bank_only_ru_flow,tdc_base_bank_only_ru_flow,repo_seed_bundle,tdcest,estimates,quarterly,{quarter_end},{release_date},{release_date},seed_bundle_snapshot_conservative_90d_lag,usd_millions,{treatment},none,unknown,false,canonical_headline,treatment,primary_treatment"
        )
        tdc_rows.append(
            f"gdp_deflator,gdp_deflator,repo_seed_bundle,tdcest,references,quarterly,{quarter_end},{release_date},{release_date},seed_bundle_snapshot_conservative_90d_lag,index,{gdp_deflator},none,unknown,false,reference,control,reference"
        )
        qra_rows.append(
            f"ati_baseline_bn,ati_baseline_bn,repo_publish,qrawatch,ati_quarter_table,quarterly,{quarter_end},{release_date},{release_date},official_qra_release_timestamp,usd_billions,{qra},none,unknown,false,debt_management_treatment,treatment,exact_official_numeric"
        )
        tsy_rows.append(
            f"tsyparty_ru_gap_l1,tsyparty_ru_gap_l1,repo_publish,tsyparty,quarterly_panel,quarterly,{quarter_end},{release_date},{release_date},published_panel_snapshot,share,{ru_gap},none,unknown,false,states,state,fixture"
        )
        tsy_rows.append(
            f"tsyparty_bank_absorption_share_l1,tsyparty_bank_absorption_share_l1,repo_publish,tsyparty,quarterly_panel,quarterly,{quarter_end},{release_date},{release_date},published_panel_snapshot,share,{bank_absorption},none,unknown,false,states,state,fixture"
        )
        tsy_rows.append(
            f"tsyparty_row_absorption_share_l1,tsyparty_row_absorption_share_l1,repo_publish,tsyparty,quarterly_panel,quarterly,{quarter_end},{release_date},{release_date},published_panel_snapshot,share,{row_absorption},none,unknown,false,states,state,fixture"
        )
        coord_rows.append(
            f"coord_low_reserve_state_l1,coord_low_reserve_state_l1,repo_publish,coordwatch,quarterly_panel,quarterly,{quarter_end},{release_date},{release_date},published_panel_snapshot,binary,{low_reserve},none,unknown,false,states,state,fixture"
        )
        fred_date = quarter_end.replace("-03-31", "-01-01").replace("-06-30", "-04-01").replace("-09-30", "-07-01").replace("-12-31", "-10-01")
        fred_levels.append(f"{fred_date},{deposit_level}")

    _write_text(paths.bundles / "tdcest" / "standardized_series.csv", "\n".join(tdc_rows))
    _write_text(paths.bundles / "qrawatch" / "standardized_series.csv", "\n".join(qra_rows))
    _write_text(paths.bundles / "tsyparty" / "standardized_series.csv", "\n".join(tsy_rows))
    _write_text(paths.bundles / "coordwatch" / "standardized_series.csv", "\n".join(coord_rows))
    _write_text(paths.raw_fred / "BOGZ1FL764100005Q.csv", "\n".join(fred_levels))

    result = build_iv_lab(paths, job_id="custom_iv_job")

    assert result.jobs_scanned == 1
    assert result.total_candidates >= 2
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["jobs_scanned"] == 1
    rows = summary["jobs"][0]["candidate_rows"]
    assert rows[0]["candidate_id"] == "iv_qra_maturity_tilt_flow_x_tsyparty_ru_gap_l1"
    assert rows[0]["is_current_instrument"] is True
    assert rows[0]["recommendation"] == "current_viable"
    assert rows[1]["candidate_id"] == "iv_qra_ati_baseline_bn_x_tsyparty_ru_gap_l1"
    assert rows[1]["is_current_instrument"] is False
