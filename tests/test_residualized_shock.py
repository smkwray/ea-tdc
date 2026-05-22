from __future__ import annotations

import csv
import json
from pathlib import Path

from ea_tdc.paths import ensure_repo_dirs, project_paths
from ea_tdc.residualized_shock import build_quarterly_fwl_audit


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_quarterly_fwl_audit_matches_controlled_lp(tmp_path: Path) -> None:
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
                "    outcomes: [matched_total_deposits]",
                "    horizons: [0]",
                "    output_family: headline_identified",
            ]
        ),
    )
    bundle_path = paths.bundles / "designs" / "baseline_tdc_lp_deposits__quarterly_bundle.csv"
    lines = ["quarter,tdc_bank_only_qoq,matched_total_deposits,GDP,gdp_deflator,FEDFUNDS,TOTRESNS"]
    for idx in range(1, 45):
        quarter = f"{2010 + ((idx - 1) // 4)}Q{((idx - 1) % 4) + 1}"
        treatment = 1.3 * idx + (idx % 5)
        gdp = 100 + idx + (idx % 3) * 0.7
        price = 95 + idx / 4 + (idx % 5) * 0.2
        fedfunds = 1 + idx / 100 + (idx % 7) * 0.01
        reserves = 50 + idx / 2 + (idx % 4) * 1.1
        outcome = 0.7 * treatment + 0.15 * gdp - 0.03 * reserves + (0.2 if idx % 2 else -0.1)
        lines.append(f"{quarter},{treatment},{outcome},{gdp},{price},{fedfunds},{reserves}")
    _write_text(bundle_path, "\n".join(lines))
    _write_text(
        paths.manifests / "baseline_tdc_lp_deposits__design_manifest.json",
        json.dumps(
            {
                "job_id": "baseline_tdc_lp_deposits",
                "status": "ready_for_estimation",
                "bundle_path": str(bundle_path),
                "treatment_id": "tdc_bank_only_qoq",
                "outcome_ids": ["matched_total_deposits"],
                "control_ids": ["GDP", "gdp_deflator", "FEDFUNDS", "TOTRESNS"],
                "horizon_grid": [0],
            }
        ),
    )
    for filename, scale in [
        ("FRED_GDP_gdp.csv", 1.0),
        ("FRED_UNRATE_unrate.csv", 0.1),
        ("FRED_INDPRO_indpro.csv", 0.5),
    ]:
        raw_lines = ["date,value"]
        for idx in range(1, 70):
            year = 2009 + ((idx - 1) // 4)
            month = (((idx - 1) % 4) + 1) * 3
            raw_lines.append(f"{year}-{month:02d}-28,{(scale * idx) + ((idx % 6) * 0.13):.4f}")
        _write_text(paths.seed / "interpol" / "raw" / filename, "\n".join(raw_lines))

    result = build_quarterly_fwl_audit(
        paths,
        job_id="baseline_tdc_lp_deposits",
        k_screened=3,
        factor_count=1,
    )

    rows = list(csv.DictReader(result.estimates_path.open("r", encoding="utf-8", newline="")))
    assert len(rows) == 1
    assert float(rows[0]["beta_abs_diff_vs_controlled_lp"]) < 1e-6

    diagnostics = list(csv.DictReader(result.diagnostics_path.open("r", encoding="utf-8", newline="")))
    assert len(diagnostics) == 1
    assert float(diagnostics[0]["tdc_residual_max_abs_control_corr"]) < 1e-6
