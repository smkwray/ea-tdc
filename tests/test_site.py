from __future__ import annotations

import json
from pathlib import Path

from ea_tdc.paths import ensure_repo_dirs, project_paths
from ea_tdc.site import build_site


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_site_exports_docs_bundle(tmp_path: Path) -> None:
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
    _write_text(paths.raw_fred / "BOGZ1FL764100005Q.csv", "date,value\n2023-10-01,0\n2024-01-01,1\n2024-04-01,3\n2024-07-01,6\n2024-10-01,10\n")
    _write_text(paths.raw_fred / "GDP.csv", "date,value\n2024-03-31,1\n2024-06-30,1\n2024-09-30,1\n2024-12-31,1\n")
    _write_text(paths.raw_fred / "FEDFUNDS.csv", "date,value\n2024-01-01,1\n2024-04-01,1\n2024-07-01,1\n2024-10-01,1\n")
    _write_text(paths.raw_fred / "TOTRESNS.csv", "date,value\n2024-03-31,1\n2024-06-30,1\n2024-09-30,1\n2024-12-31,1\n")
    _write_text(paths.reports / "macro_prices_secret_screening.md", "internal only")

    result = build_site(paths)

    assert result.index_path.exists()
    assert result.sidecar_index_path.exists()
    assert (paths.root / "docs" / ".nojekyll").exists()
    assert (paths.root / "docs" / "assets" / "css" / "style.css").exists()
    assert (paths.root / "docs" / "assets" / "js" / "theme.js").exists()
    assert (paths.root / "docs" / "assets" / "js" / "main.js").exists()
    assert (paths.root / "docs" / "assets" / "data" / "site_data.json").exists()
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["copied_artifacts"] > 0
    assert summary["copied_reports"] > 0
    assert summary["copied_models"] > 0
    assert summary["sidecar_index_path"] == "docs/sidecar-results/index.html"
    assert summary["site_data_path"] == "docs/assets/data/site_data.json"
    assert summary["artifact_gallery_path"] == "docs/artifacts/index.html"

    index_text = result.index_path.read_text(encoding="utf-8")
    assert 'data-theme="light"' in index_text
    assert 'id="theme-toggle"' in index_text
    assert 'href="assets/css/style.css?v=' in index_text
    assert 'src="assets/js/theme.js?v=' in index_text
    assert 'src="assets/js/main.js?v=' in index_text
    assert 'cdn.plot.ly/plotly-2.35.2.min.js' in index_text
    assert "Treasury contribution to deposits: estimates and transmission." in index_text
    assert "Research questions." in index_text
    assert "View equations" in index_text
    assert "Results that still need stronger identification." in index_text
    assert "Federal Reserve net transactions in marketable Treasury securities." in index_text
    assert "Treasury Deposit Contribution." in index_text
    assert "Treasury Deposit Contribution: the contribution of Treasury-related cash flows and Treasury-security transactions to changes in domestic nonbank deposits." in index_text
    assert "Fed-net reserve responses." not in index_text
    assert "Raw reserves are not treated as a headline result here" not in index_text
    assert "What the release claims." in index_text
    assert "Deposits are the clearest headline response." in index_text
    assert "Strict independent non-TDC evidence is narrower and source-side." in index_text
    assert "The valid independent non-TDC comparison is the narrower `tdcpass` source-side lane, not residual closure." in index_text
    assert "Inflation, FX, and private balance sheets." in index_text
    assert "Which indicators dominate the screened control set?" not in index_text
    assert "Did the IV search find stronger instruments?" not in index_text
    assert "Change in Treasury operating cash transactions; higher Treasury cash drains deposits before they reach domestic nonbank deposits." in index_text
    assert 'href="#additional-evidence"' in index_text
    assert 'data-page="home"' in index_text
    sidecar_text = result.sidecar_index_path.read_text(encoding="utf-8")
    assert "This material now lives on the main page." in sidecar_text
    assert "Open additional evidence" in sidecar_text
    assert "The public site now uses a single continuous narrative." in sidecar_text
    assert 'href="../assets/css/style.css?v=' in sidecar_text
    assert 'src="../assets/js/theme.js?v=' in sidecar_text
    assert 'src="../assets/js/main.js?v=' in sidecar_text
    assert 'data-page="sidecar"' in sidecar_text
    gallery_text = (paths.root / "docs" / "artifacts" / "index.html").read_text(encoding="utf-8")
    assert "All figures and tables in one place." in gallery_text
    assert 'data-page="gallery"' in gallery_text
    artifact_dirs = sorted(
        path for path in (paths.root / "docs" / "artifacts").iterdir() if path.is_dir()
    )
    assert artifact_dirs
    artifact_text = (artifact_dirs[0] / "index.html").read_text(encoding="utf-8")
    assert "Selected figure or table." in artifact_text
    assert 'assets/js/theme.js?v=' in artifact_text
    site_data = json.loads((paths.root / "docs" / "assets" / "data" / "site_data.json").read_text(encoding="utf-8"))
    assert site_data["artifact_gallery"]
    assert "robustness" in site_data
    assert "treatment_comparisons" in site_data["sidecar"]
    assert isinstance(site_data["sidecar"]["treatment_comparisons"], list)
    assert "factor_summaries" not in site_data["sidecar"]
    assert site_data["jobs"]["custom_lp_job"]["branch_label"] == "Baseline controls"
    assert site_data["jobs"]["custom_lp_job"]["branch_note"] == "Selected public version uses the baseline quarterly control set."
    assert "jobs" in site_data["robustness"]
    assert "deposit_accounting" not in site_data["home"]
    assert "independent_evidence" in site_data["home"]
    main_js = (paths.root / "docs" / "assets" / "js" / "main.js").read_text(encoding="utf-8")
    assert r"\\[" in main_js
    assert "Lpiv" not in json.dumps(site_data)
    home_html = (paths.root / "docs" / "index.html").read_text(encoding="utf-8")
    assert 'id="component-evidence"' in home_html
    assert 'id="independent-evidence-section"' in home_html
    assert "Open component summary" in home_html
    assert "Final interpretation" in home_html
    if site_data["robustness"]["jobs"]:
        assert site_data["robustness"]["jobs"][0]["ml_public_branch"] in {"none", "dml", "forest", "tmle"}
        assert site_data["robustness"]["jobs"][0]["ml_public_branch_label"]
        assert "screens K=" in site_data["robustness"]["jobs"][0]["interpretation"]
        link_labels = [link["label"] for link in site_data["robustness"]["jobs"][0]["links"]]
        assert "Experimental TMLE" not in link_labels
        assert "TMLE estimates" not in link_labels

    copied_preview = paths.root / "docs" / "site_assets" / "artifacts" / "main_text" / "main_figure_1" / "main_figure_1.html"
    assert copied_preview.exists()
    assert (paths.root / "docs" / "site_assets" / "reports" / "component_sidecar_screening.md").exists()
    assert (paths.root / "docs" / "site_assets" / "reports" / "final_interpretation_closeout.md").exists()
    assert not (paths.root / "docs" / "site_assets" / "reports" / "macro_prices_secret_screening.md").exists()
    assert not (paths.root / "docs" / "site_assets" / "reports" / "release_snapshot.json").exists()
    assert not (paths.root / "docs" / "site_assets" / "reports" / "site_build.json").exists()
    public_bundle_text = (paths.root / "docs" / "site_assets" / "reports" / "release_scorecard.json").read_text(encoding="utf-8")
    assert "/Users/" not in public_bundle_text
    assert str(paths.root) not in public_bundle_text
    scorecard_text = (paths.root / "output" / "reports" / "release_scorecard.json").read_text(encoding="utf-8")
    assert str(paths.root) not in scorecard_text
    estimation_snapshot_text = (paths.root / "output" / "reports" / "estimation_snapshot.csv").read_text(encoding="utf-8")
    assert str(paths.root) not in estimation_snapshot_text
    site_build_text = (paths.root / "output" / "reports" / "site_build.json").read_text(encoding="utf-8")
    assert "/Users/" not in site_build_text
    assert str(paths.root) not in site_build_text
