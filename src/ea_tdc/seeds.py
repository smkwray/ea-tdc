from __future__ import annotations

import shutil
from pathlib import Path

from .config import SourceSpec
from .paths import ProjectPaths
from .utils import utc_now_iso, write_json


def copy_seed_source(source: SourceSpec, paths: ProjectPaths) -> dict[str, object]:
    origin = source.resolved_seed_path()
    if origin is None:
        raise RuntimeError(f"Environment variable {source.path_env} is not set")
    if not origin.exists():
        raise FileNotFoundError(f"Seed path does not exist: {origin}")
    if source.copy_to is None:
        raise RuntimeError(f"Source {source.name} does not define a copy target")

    destination = source.copy_to
    destination.parent.mkdir(parents=True, exist_ok=True)

    if origin.is_dir():
        shutil.copytree(origin, destination, dirs_exist_ok=True)
        copy_kind = "directory"
    else:
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, destination / origin.name)
        copy_kind = "file"

    manifest = {
        "kind": "seed_copy_manifest",
        "source_name": source.name,
        "source_kind": source.kind,
        "copied_at_utc": utc_now_iso(),
        "origin": str(origin),
        "destination": str(destination),
        "copy_kind": copy_kind,
    }
    write_json(paths.manifests / f"seed_copy__{source.name}.json", manifest)
    return manifest
