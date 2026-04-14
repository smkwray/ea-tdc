from __future__ import annotations

import csv
import json
from pathlib import Path

from ea_tdc.adapters.tdcest import adapt_tdcest
from ea_tdc.paths import ensure_repo_dirs, project_paths


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_tdcest_adapter_writes_standardized_bundle_and_manifest(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "runtime.yaml",
        "\n".join(
            [
                "project:",
                "  name: ea-tdc",
                "  package_name: ea_tdc",
                "  active_release: release_1",
                "paths:",
                "  data_root: data",
                "  output_root: output",
                "fetch:",
                "  fred_series_manifest: config/fred_manifest_seed.csv",
                "  fred_api_key_env: FRED_API_KEY",
                "  default_start_date: 1980-01-01",
                "  allow_graph_csv_fallback: true",
                "remote:",
                "  ssh_host: shanewray@100.71.19.72",
                "  run_heavy_jobs_remotely: true",
                "  path_parity_note: mirrored",
            ]
        ),
    )
    _write_text(
        tmp_path / "config" / "fred_manifest_seed.csv",
        "\n".join(
            [
                "series_id,domain,role,priority,default_transform,notes",
                "GDP,real_activity,control,headline,logdiff,GDP level",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "seed" / "tdcest" / "bundle.json",
        json.dumps(
            {
                "bundle_format": "tdc_site_bundle_v2",
                "generated_at_utc": "2026-04-11T00:00:00+00:00",
                "summary": {"preferred_method": "tdc_base_bank_only_ru_flow"},
                "metadata": {},
                "dates": ["2024-03-31", "2024-06-30"],
                "estimates": {
                    "columns": ["tdc_base_bank_only_ru_flow"],
                    "tdc_base_bank_only_ru_flow": [1.5, 2.5],
                },
                "components": {
                    "columns": ["fed_tsy_tx"],
                    "fed_tsy_tx": [0.5, 0.75],
                },
                "references": {
                    "columns": ["gdp_deflator"],
                    "gdp_deflator": [120.0, 121.0],
                },
            }
        ),
    )

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = adapt_tdcest(paths)

    assert result.rows_written == 6
    assert result.standardized_path.exists()
    assert result.manifest_path.exists()

    with result.standardized_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert any(row["series_id"] == "tdc_base_bank_only_ru_flow" and row["role"] == "treatment" for row in rows)
    assert any(row["series_id"] == "gdp_deflator" and row["role"] == "control" for row in rows)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_repo"] == "tdcest"
    assert manifest["rows_written"] == 6
    assert manifest["bundle_hash"] == result.bundle_hash
