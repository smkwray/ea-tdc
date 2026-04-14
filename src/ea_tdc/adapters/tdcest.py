from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ea_tdc.paths import ProjectPaths
from ea_tdc.utils import utc_now_iso, write_json


REQUIRED_BUNDLE_KEYS = {
    "bundle_format",
    "generated_at_utc",
    "summary",
    "metadata",
    "dates",
    "estimates",
    "components",
    "references",
}


@dataclass(frozen=True)
class AdapterResult:
    standardized_path: Path
    manifest_path: Path
    rows_written: int
    bundle_hash: str


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict-like JSON payload at {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _conservative_quarterly_available_at(period_end: str) -> str:
    parsed = datetime.strptime(period_end, "%Y-%m-%d").date()
    return (parsed + timedelta(days=90)).isoformat()


def resolve_seed_bundle(paths: ProjectPaths, explicit_path: str | None = None) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_absolute():
            path = (paths.root / path).resolve()
        return path
    return paths.seed / "tdcest" / "bundle.json"


def read_raw(bundle_path: Path) -> dict[str, Any]:
    payload = _read_json(bundle_path)
    missing = REQUIRED_BUNDLE_KEYS.difference(payload)
    if missing:
        raise KeyError(f"Missing required keys in tdcest bundle: {sorted(missing)}")
    if payload.get("bundle_format") != "tdc_site_bundle_v2":
        raise ValueError(f"Unexpected tdcest bundle format: {payload.get('bundle_format')!r}")
    return payload


def _role_for_series(source_table: str, series_id: str) -> tuple[str, str, str]:
    if source_table == "estimates":
        if series_id == "tdc_base_bank_only_ru_flow":
            return "treatment", "primary_treatment", "canonical_headline"
        if series_id.startswith("tdc_"):
            return "treatment", "treatment_variant", "estimate_variant"
        return "mechanism", "estimate_support", "estimate_support"
    if source_table == "components":
        if series_id in {"tdc_base_bank_only_ru_flow", "tdc_base_broad_depository_np_cu_ru_flow"}:
            return "treatment", "component_bridge", "component_aggregate"
        return "mechanism", "component", "component"
    if source_table == "references":
        return "control", "reference", "reference"
    return "mechanism", "unknown", "unknown"


def normalize_schema(payload: dict[str, Any]) -> list[dict[str, str]]:
    dates = payload["dates"]
    rows: list[dict[str, str]] = []

    for source_table in ("estimates", "components", "references"):
        table = payload[source_table]
        columns = [col for col in table.get("columns", []) if col in table]
        for series_id in columns:
            values = table.get(series_id, [])
            role, family, component_group = _role_for_series(source_table, series_id)
            for date_value, value in zip(dates, values):
                available_at = _conservative_quarterly_available_at(str(date_value))
                rows.append(
                    {
                        "series_id": series_id,
                        "series_label": series_id,
                        "source_family": "repo_seed_bundle",
                        "source_repo": "tdcest",
                        "source_table": source_table,
                        "freq": "quarterly",
                        "period_end": str(date_value),
                        "release_date": available_at,
                        "available_at": available_at,
                        "vintage_policy": "seed_bundle_snapshot_conservative_90d_lag",
                        "units": "usd_millions" if series_id != "gdp_deflator" else "index",
                        "value": "" if value is None else str(value),
                        "transform_default": "none",
                        "seasonal_adjustment_flag": "unknown",
                        "interpolated_flag": "false",
                        "component_group": component_group,
                        "role": role,
                        "notes": f"{family}|availability_proxy=period_end_plus_90d",
                    }
                )
    return rows


def validate_contract(rows: list[dict[str, str]]) -> None:
    required = {
        "series_id",
        "series_label",
        "source_family",
        "source_repo",
        "source_table",
        "freq",
        "period_end",
        "release_date",
        "available_at",
        "vintage_policy",
        "units",
        "value",
        "transform_default",
        "seasonal_adjustment_flag",
        "interpolated_flag",
        "component_group",
        "role",
        "notes",
    }
    if not rows:
        raise ValueError("No standardized rows were produced from the tdcest bundle")
    missing = required.difference(rows[0])
    if missing:
        raise KeyError(f"Standardized rows are missing required fields: {sorted(missing)}")
    if not any(row["series_id"] == "tdc_base_bank_only_ru_flow" and row["role"] == "treatment" for row in rows):
        raise ValueError("Canonical treatment tdc_base_bank_only_ru_flow was not found in standardized rows")


def write_standard_bundle(paths: ProjectPaths, rows: list[dict[str, str]]) -> Path:
    bundle_dir = paths.bundles / "tdcest"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    target = bundle_dir / "standardized_series.csv"

    fieldnames = list(rows[0].keys())
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target


def write_source_manifest(
    paths: ProjectPaths,
    *,
    source_path: Path,
    standardized_path: Path,
    rows_written: int,
    bundle_hash: str,
) -> Path:
    manifest = {
        "kind": "source_manifest",
        "source_repo": "tdcest",
        "adapter": "tdcest_bundle",
        "generated_at_utc": utc_now_iso(),
        "source_path": str(source_path),
        "standardized_path": str(standardized_path),
        "rows_written": rows_written,
        "bundle_hash": bundle_hash,
    }
    target = paths.manifests / "tdcest_source_manifest.json"
    write_json(target, manifest)
    return target


def adapt_tdcest(paths: ProjectPaths, *, bundle_path: str | None = None) -> AdapterResult:
    source_path = resolve_seed_bundle(paths, bundle_path)
    payload = read_raw(source_path)
    rows = normalize_schema(payload)
    validate_contract(rows)
    standardized_path = write_standard_bundle(paths, rows)
    bundle_hash = _sha256_file(source_path)
    manifest_path = write_source_manifest(
        paths,
        source_path=source_path,
        standardized_path=standardized_path,
        rows_written=len(rows),
        bundle_hash=bundle_hash,
    )
    return AdapterResult(
        standardized_path=standardized_path,
        manifest_path=manifest_path,
        rows_written=len(rows),
        bundle_hash=bundle_hash,
    )
