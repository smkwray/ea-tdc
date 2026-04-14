from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .paths import REPO_ROOT


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict-like YAML at {path}, got {type(data)!r}")
    return data


def _resolve_repo_path(value: str | None, repo_root: Path) -> Path | None:
    if not value:
        return None
    expanded = os.path.expandvars(value)
    candidate = Path(expanded)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def _coerce_optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


@dataclass(frozen=True)
class RemoteConfig:
    ssh_host: str
    run_heavy_jobs_remotely: bool
    path_parity_note: str


@dataclass(frozen=True)
class RuntimeConfig:
    project_name: str
    package_name: str
    active_release: str
    data_root: str
    output_root: str
    fred_series_manifest: Path
    treasury_dataset_manifest: Path
    fred_api_key_env: str
    default_start_date: str | None
    default_end_date: str | None
    allow_graph_csv_fallback: bool
    remote: RemoteConfig


@dataclass(frozen=True)
class SourceSpec:
    name: str
    kind: str
    adapter: str
    required: bool
    notes: str
    target_dir: Path | None = None
    manifest: Path | None = None
    path_env: str | None = None
    copy_to: Path | None = None

    def resolved_seed_path(self) -> Path | None:
        if not self.path_env:
            return None
        value = os.environ.get(self.path_env)
        if not value:
            return None
        return Path(os.path.expandvars(value)).expanduser()


def load_runtime_config(repo_root: Path | str | None = None) -> RuntimeConfig:
    root = Path(repo_root or REPO_ROOT).resolve()
    payload = _load_yaml(root / "config" / "runtime.yaml")
    project = payload.get("project", {})
    paths = payload.get("paths", {})
    fetch = payload.get("fetch", {})
    remote = payload.get("remote", {})

    return RuntimeConfig(
        project_name=str(project.get("name", "ea-tdc")),
        package_name=str(project.get("package_name", "ea_tdc")),
        active_release=str(project.get("active_release", "release_1")),
        data_root=str(paths.get("data_root", "data")),
        output_root=str(paths.get("output_root", "output")),
        fred_series_manifest=_resolve_repo_path(str(fetch.get("fred_series_manifest", "config/fred_manifest_seed.csv")), root)
        or (root / "config" / "fred_manifest_seed.csv"),
        treasury_dataset_manifest=_resolve_repo_path(str(fetch.get("treasury_dataset_manifest", "config/treasury_manifest.yaml")), root)
        or (root / "config" / "treasury_manifest.yaml"),
        fred_api_key_env=str(fetch.get("fred_api_key_env", "FRED_API_KEY")),
        default_start_date=_coerce_optional_text(fetch.get("default_start_date")),
        default_end_date=_coerce_optional_text(fetch.get("default_end_date")),
        allow_graph_csv_fallback=bool(fetch.get("allow_graph_csv_fallback", True)),
        remote=RemoteConfig(
            ssh_host=str(remote.get("ssh_host") or ""),
            run_heavy_jobs_remotely=bool(remote.get("run_heavy_jobs_remotely", True)),
            path_parity_note=str(remote.get("path_parity_note") or ""),
        ),
    )


def load_source_registry(repo_root: Path | str | None = None) -> list[SourceSpec]:
    root = Path(repo_root or REPO_ROOT).resolve()
    payload = _load_yaml(root / "config" / "source_registry.template.yaml")
    registry = payload.get("sources", {})
    if not isinstance(registry, dict):
        raise TypeError("Expected 'sources' to be a mapping")

    specs: list[SourceSpec] = []
    for name, item in registry.items():
        if not isinstance(item, dict):
            raise TypeError(f"Expected source '{name}' to be a mapping")
        specs.append(
            SourceSpec(
                name=name,
                kind=str(item.get("kind", "")),
                adapter=str(item.get("adapter", "")),
                required=bool(item.get("required", False)),
                notes=str(item.get("notes", "")),
                target_dir=_resolve_repo_path(item.get("target_dir"), root),
                manifest=_resolve_repo_path(item.get("manifest"), root),
                path_env=item.get("path_env"),
                copy_to=_resolve_repo_path(item.get("copy_to"), root),
            )
        )
    return specs
