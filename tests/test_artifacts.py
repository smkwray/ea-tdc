from __future__ import annotations

import json
from pathlib import Path

from ea_tdc.artifacts import build_release_artifacts
from ea_tdc.paths import ensure_repo_dirs, project_paths


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_release_artifacts_renders_svg_and_tables(tmp_path: Path) -> None:
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
    _write_text(
        paths.output / "models" / "custom_lp_job__robustness_k200_estimates.csv",
        "\n".join(
            [
                "job_id,outcome,horizon,beta,se,lower95,upper95,p_value_normal,n",
                "custom_lp_job,matched_total_deposits,0,0.5,0.1,0.3,0.7,0.002,40",
                "custom_lp_job,matched_total_deposits,1,0.4,0.1,0.2,0.6,0.010,39",
            ]
        ),
    )
    _write_text(
        paths.manifests / "custom_lp_job__robustness_summary.json",
        json.dumps({"recommended_k": 200}, indent=2),
    )

    result = build_release_artifacts(paths)

    assert result.artifacts_built == 3
    assert result.figure_artifacts == 1
    assert result.table_artifacts == 2
    assert result.gallery_path.exists()

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["artifacts_built"] == 3
    rows = {row["artifact_id"]: row for row in summary["rows"]}
    figure_path = Path(rows["main_figure_1"]["primary_path"])
    figure_html_path = Path(rows["main_figure_1"]["html_path"])
    main_table_path = Path(rows["main_table_1"]["primary_path"])
    main_table_html_path = Path(rows["main_table_1"]["html_path"])
    appendix_csv_path = Path(rows["appendix_table_1"]["secondary_path"])

    assert figure_path.exists()
    assert "<svg" in figure_path.read_text(encoding="utf-8")
    assert "Custom Lp Job" in figure_path.read_text(encoding="utf-8")
    assert "Treatment:" in figure_path.read_text(encoding="utf-8")
    assert "GDP deflator" in figure_path.read_text(encoding="utf-8")
    assert figure_html_path.exists()
    assert "<img src=" in figure_html_path.read_text(encoding="utf-8")
    assert "Local projection" in figure_html_path.read_text(encoding="utf-8")
    assert "Main Figure 1" in figure_html_path.read_text(encoding="utf-8")
    assert "K=200 screened branch" in figure_html_path.read_text(encoding="utf-8")
    assert main_table_path.exists()
    main_table_text = main_table_path.read_text(encoding="utf-8")
    assert "| matched_total_deposits |" not in main_table_text
    assert "| matched total deposits |" in main_table_text.lower()
    assert "| outcome | horizon | beta | se | lower 95% | upper 95% | p-value | observations | significance |" in main_table_text.lower()
    assert "## Notes" in main_table_text
    assert "Coefficient table for" in main_table_text
    assert "in response to" in main_table_text
    assert main_table_html_path.exists()
    assert "<table>" in main_table_html_path.read_text(encoding="utf-8")
    assert "Main Table 1" in main_table_html_path.read_text(encoding="utf-8")
    assert "<th>significance</th>" in main_table_html_path.read_text(encoding="utf-8").lower()
    assert appendix_csv_path.exists()
    assert appendix_csv_path.read_text(encoding="utf-8").splitlines()[0].startswith("outcome,horizon,beta,se")
    assert rows["main_figure_1"]["source_estimates_path"].endswith("custom_lp_job__robustness_k200_estimates.csv")
    assert "Main Text" in result.gallery_path.read_text(encoding="utf-8")
    assert "Committed Release 1 artifacts" in result.gallery_path.read_text(encoding="utf-8")
    assert "Rendered artifacts" in result.gallery_path.read_text(encoding="utf-8")
    assert "Main Figure 1" in result.gallery_path.read_text(encoding="utf-8")
