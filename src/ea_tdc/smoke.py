from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .config import load_runtime_config, load_source_registry
from .paths import ensure_repo_dirs, project_paths
from .utils import utc_now_iso, write_json


def _count_manifest_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def run_smoke(repo_root: Path | str | None = None) -> dict[str, Any]:
    runtime = load_runtime_config(repo_root)
    paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
    ensure_repo_dirs(paths)
    sources = load_source_registry(repo_root)

    source_status: list[dict[str, Any]] = []
    for source in sources:
        resolved_seed_path = source.resolved_seed_path()
        source_status.append(
            {
                "name": source.name,
                "kind": source.kind,
                "adapter": source.adapter,
                "required": source.required,
                "path_env": source.path_env,
                "env_is_set": bool(resolved_seed_path),
                "seed_exists": bool(resolved_seed_path and resolved_seed_path.exists()),
                "target_dir": str(source.target_dir) if source.target_dir else None,
                "copy_to": str(source.copy_to) if source.copy_to else None,
            }
        )

    manifest = {
        "kind": "bootstrap_manifest_bundle",
        "generated_at_utc": utc_now_iso(),
        "project": {
            "name": runtime.project_name,
            "package_name": runtime.package_name,
            "active_release": runtime.active_release,
        },
        "paths": {
            "root": str(paths.root),
            "data": str(paths.data),
            "raw_fred": str(paths.raw_fred),
            "raw_treasury": str(paths.raw_treasury),
            "seed": str(paths.seed),
            "bundles": str(paths.bundles),
            "output": str(paths.output),
            "manifests": str(paths.manifests),
        },
        "fetch": {
            "fred_series_manifest": str(runtime.fred_series_manifest),
            "fred_series_count": _count_manifest_rows(runtime.fred_series_manifest),
            "fred_api_key_env": runtime.fred_api_key_env,
            "default_start_date": runtime.default_start_date,
            "default_end_date": runtime.default_end_date,
            "allow_graph_csv_fallback": runtime.allow_graph_csv_fallback,
        },
        "remote": {
            "ssh_host": runtime.remote.ssh_host,
            "run_heavy_jobs_remotely": runtime.remote.run_heavy_jobs_remotely,
            "path_parity_note": runtime.remote.path_parity_note,
        },
        "sources": source_status,
        "design_manifests": [],
        "sample_manifests": [],
        "results": [],
    }
    write_json(paths.manifests / "bootstrap_manifest_bundle.json", manifest)
    return manifest
