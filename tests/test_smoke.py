from __future__ import annotations

import json
from pathlib import Path

from ea_tdc.smoke import run_smoke


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_smoke_writes_bootstrap_manifest(tmp_path: Path) -> None:
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
                "  default_end_date:",
                "  allow_graph_csv_fallback: true",
                "remote:",
                "  ssh_host: shanewray@100.71.19.72",
                "  run_heavy_jobs_remotely: true",
                "  path_parity_note: mirrored",
            ]
        ),
    )
    _write_text(
        tmp_path / "config" / "source_registry.template.yaml",
        "\n".join(
            [
                "sources:",
                "  fred:",
                "    kind: direct_download",
                "    target_dir: data/raw/fred",
                "    manifest: config/fred_manifest_seed.csv",
                "    adapter: fred_csv_folder",
                "    required: true",
                "    notes: demo",
                "  tdcest_seed:",
                "    kind: optional_seed_file",
                "    path_env: EA_TDC_SEED_TDCEST_BUNDLE",
                "    copy_to: data/seed/tdcest",
                "    adapter: tdcest_bundle",
                "    required: false",
                "    notes: demo",
            ]
        ),
    )
    _write_text(
        tmp_path / "config" / "fred_manifest_seed.csv",
        "\n".join(
            [
                "series_id,domain,role,priority,default_transform,notes",
                "GDP,real_activity,control,headline,logdiff,GDP level",
                "M2SL,deposits_money,outcome,headline,logdiff,M2",
            ]
        ),
    )

    manifest = run_smoke(tmp_path)
    manifest_path = tmp_path / "output" / "manifests" / "bootstrap_manifest_bundle.json"

    assert manifest["fetch"]["fred_series_count"] == 2
    assert manifest_path.exists()

    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stored["project"]["package_name"] == "ea_tdc"
