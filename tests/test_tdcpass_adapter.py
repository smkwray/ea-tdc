from __future__ import annotations

import csv
from pathlib import Path

from ea_tdc.adapters.tdcpass import adapt_tdcpass
from ea_tdc.paths import ensure_repo_dirs, project_paths


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_tdcpass_adapter_writes_standardized_bundle(tmp_path: Path) -> None:
    source_root = tmp_path / "tdcpass_seed"
    _write_text(
        source_root / "data/derived/quarterly_panel.csv",
        "\n".join(
            [
                "quarter,tdc_bank_only_qoq,total_deposits_bank_qoq,other_component_qoq,lag_tdc_bank_only_qoq,lag_fedfunds,lag_unemployment,lag_inflation",
                "2024Q1,100,40,-60,90,5.0,4.0,2.0",
                "2024Q2,110,45,-65,100,5.1,4.1,2.1",
            ]
        ),
    )
    _write_text(
        source_root / "output/shocks/unexpected_tdc.csv",
        "\n".join(
            [
                "quarter,tdc_fitted,tdc_residual,tdc_residual_z",
                "2024Q1,80,20,1.5",
                "2024Q2,85,25,1.7",
            ]
        ),
    )
    _write_text(
        source_root / "output/models/lp_irf_identity_baseline.csv",
        "\n".join(
            [
                "outcome,horizon,beta",
                "tdc_bank_only_qoq,0,100",
            ]
        ),
    )
    _write_text(source_root / "output/models/result_readiness_summary.json", '{"status":"provisional"}')

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = adapt_tdcpass(paths, publish_dir=str(source_root))

    assert result.standardized_path.exists()
    assert result.published_reference_path.exists()
    with result.standardized_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["series_id"] == "tdcpass_tdc_bank_only_qoq" for row in rows)
    assert any(row["series_id"] == "tdcpass_tdc_residual_z" and row["role"] == "treatment" for row in rows)
