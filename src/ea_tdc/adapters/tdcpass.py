from __future__ import annotations

import csv
import hashlib
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ea_tdc.paths import ProjectPaths
from ea_tdc.utils import utc_now_iso, write_json


PANEL_PATH = Path("data/derived/quarterly_panel.csv")
SHOCK_PATH = Path("output/shocks/unexpected_tdc.csv")
REFERENCE_IRF_PATH = Path("output/models/lp_irf_identity_baseline.csv")
READINESS_PATH = Path("output/models/result_readiness_summary.json")


@dataclass(frozen=True)
class AdapterResult:
    standardized_path: Path
    manifest_path: Path
    published_reference_path: Path
    rows_written: int
    bundle_hash: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _quarter_end_date(quarter: str) -> str:
    year_text, quarter_text = quarter.split("Q", 1)
    year = int(year_text)
    quarter_num = int(quarter_text)
    month = quarter_num * 3
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    return (next_month.date() - timedelta(days=1)).isoformat()


def _conservative_available_at(quarter: str) -> str:
    period_end = datetime.fromisoformat(_quarter_end_date(quarter)).date()
    return (period_end + timedelta(days=90)).isoformat()


def _unit_for_series(series_id: str) -> str:
    if series_id.endswith("_qoq") or series_id.endswith("_residual") or series_id.endswith("_fitted"):
        return "usd_billions"
    if series_id.endswith("_share") or series_id.endswith("_z"):
        return "ratio"
    if series_id in {"fedfunds", "unemployment", "inflation"} or series_id.startswith("lag_"):
        return "percent"
    return "unitless"


def resolve_seed_dir(paths: ProjectPaths, explicit_dir: str | None = None) -> Path:
    if explicit_dir:
        path = Path(explicit_dir).expanduser()
        if not path.is_absolute():
            path = (paths.root / path).resolve()
        return path
    return paths.seed / "tdcpass"


def _validate_source_dir(root: Path) -> tuple[Path, Path, Path]:
    panel_path = root / PANEL_PATH
    shock_path = root / SHOCK_PATH
    reference_path = root / REFERENCE_IRF_PATH
    missing = [path for path in [panel_path, shock_path, reference_path] if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required tdcpass artifacts: {missing_text}")
    return panel_path, shock_path, reference_path


def normalize_schema(panel_rows: list[dict[str, str]], shock_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    panel_columns = [key for key in panel_rows[0].keys() if key != "quarter"] if panel_rows else []
    shock_by_quarter = {str(row.get("quarter", "")).strip(): row for row in shock_rows}

    for row in panel_rows:
        quarter = str(row.get("quarter", "")).strip()
        if not quarter:
            continue
        period_end = _quarter_end_date(quarter)
        available_at = _conservative_available_at(quarter)
        for column in panel_columns:
            value = str(row.get(column, "")).strip()
            rows.append(
                {
                    "series_id": f"tdcpass_{column}",
                    "series_label": column,
                    "source_family": "repo_seed_bundle",
                    "source_repo": "tdcpass",
                    "source_table": "quarterly_panel",
                    "freq": "quarterly",
                    "period_end": period_end,
                    "release_date": available_at,
                    "available_at": available_at,
                    "vintage_policy": "tdcpass_publish_snapshot_conservative_90d_lag",
                    "units": _unit_for_series(column),
                    "value": value,
                    "transform_default": "none",
                    "seasonal_adjustment_flag": "unknown",
                    "interpolated_flag": "false",
                    "component_group": "published_panel",
                    "role": "reference" if column.startswith("lag_") else "mechanism",
                    "notes": "tdcpass_quarterly_panel",
                }
            )
        shock_row = shock_by_quarter.get(quarter)
        if shock_row is None:
            continue
        for column in ["tdc_fitted", "tdc_residual", "tdc_residual_z"]:
            value = str(shock_row.get(column, "")).strip()
            rows.append(
                {
                    "series_id": f"tdcpass_{column}",
                    "series_label": column,
                    "source_family": "repo_seed_bundle",
                    "source_repo": "tdcpass",
                    "source_table": "unexpected_tdc",
                    "freq": "quarterly",
                    "period_end": period_end,
                    "release_date": available_at,
                    "available_at": available_at,
                    "vintage_policy": "tdcpass_publish_snapshot_conservative_90d_lag",
                    "units": _unit_for_series(column),
                    "value": value,
                    "transform_default": "none",
                    "seasonal_adjustment_flag": "unknown",
                    "interpolated_flag": "false",
                    "component_group": "published_shock",
                    "role": "treatment" if column == "tdc_residual_z" else "reference",
                    "notes": "tdcpass_unexpected_tdc",
                }
            )
    return rows


def validate_contract(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No standardized rows were produced from tdcpass artifacts")
    required_ids = {
        "tdcpass_tdc_bank_only_qoq",
        "tdcpass_total_deposits_bank_qoq",
        "tdcpass_other_component_qoq",
        "tdcpass_tdc_residual_z",
    }
    seen = {row["series_id"] for row in rows}
    missing = required_ids.difference(seen)
    if missing:
        raise KeyError(f"Missing required tdcpass standardized series: {sorted(missing)}")


def _bundle_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(_sha256_file(path).encode("utf-8"))
    return digest.hexdigest()


def write_standard_bundle(paths: ProjectPaths, rows: list[dict[str, str]]) -> Path:
    bundle_dir = paths.bundles / "tdcpass"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    target = bundle_dir / "standardized_series.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return target


def write_published_reference(paths: ProjectPaths, source_path: Path) -> Path:
    target = paths.bundles / "tdcpass" / "published_identity_baseline.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target)
    return target


def write_source_manifest(
    paths: ProjectPaths,
    *,
    source_root: Path,
    standardized_path: Path,
    published_reference_path: Path,
    readiness_summary_path: Path | None,
    rows_written: int,
    bundle_hash: str,
) -> Path:
    readiness_status = ""
    if readiness_summary_path is not None and readiness_summary_path.exists():
        readiness_payload = readiness_summary_path.read_text(encoding="utf-8")
        readiness_status = readiness_payload
    manifest = {
        "kind": "source_manifest",
        "source_repo": "tdcpass",
        "adapter": "tdcpass_outputs",
        "generated_at_utc": utc_now_iso(),
        "source_root": str(source_root),
        "standardized_path": str(standardized_path),
        "published_reference_path": str(published_reference_path),
        "rows_written": rows_written,
        "bundle_hash": bundle_hash,
        "readiness_summary_path": str(readiness_summary_path) if readiness_summary_path else "",
        "readiness_summary_present": bool(readiness_status),
    }
    target = paths.manifests / "tdcpass_source_manifest.json"
    write_json(target, manifest)
    return target


def adapt_tdcpass(paths: ProjectPaths, *, publish_dir: str | None = None) -> AdapterResult:
    source_root = resolve_seed_dir(paths, publish_dir)
    panel_path, shock_path, reference_path = _validate_source_dir(source_root)
    readiness_path = source_root / READINESS_PATH
    panel_rows = _read_csv(panel_path)
    shock_rows = _read_csv(shock_path)
    rows = normalize_schema(panel_rows, shock_rows)
    validate_contract(rows)
    standardized_path = write_standard_bundle(paths, rows)
    published_reference_path = write_published_reference(paths, reference_path)
    bundle_hash = _bundle_hash([panel_path, shock_path, reference_path])
    manifest_path = write_source_manifest(
        paths,
        source_root=source_root,
        standardized_path=standardized_path,
        published_reference_path=published_reference_path,
        readiness_summary_path=readiness_path if readiness_path.exists() else None,
        rows_written=len(rows),
        bundle_hash=bundle_hash,
    )
    return AdapterResult(
        standardized_path=standardized_path,
        manifest_path=manifest_path,
        published_reference_path=published_reference_path,
        rows_written=len(rows),
        bundle_hash=bundle_hash,
    )
