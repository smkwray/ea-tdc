from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from ea_tdc.paths import ProjectPaths
from ea_tdc.utils import utc_now_iso, write_json


SECTOR_PANEL_PATH = Path("data/processed/sector_effective_maturity_full.csv")
RUN_MANIFEST_PATH = Path("outputs/full_coverage_release/run_manifest.json")
SUMMARY_PATH = Path("outputs/full_coverage_release/full_coverage_summary.json")
LAG_DAYS = 90

SECTOR_SERIES = {
    "bank_reserve_access_core": {
        "bill_share": ("wamest_bank_reserve_bill_share_l1", "share", "bank_capacity_state"),
        "short_share_le_1y": ("wamest_bank_reserve_short_share_l1", "share", "bank_capacity_state"),
        "zero_coupon_equivalent_years": ("wamest_bank_reserve_wam_years_l1", "years", "bank_capacity_state"),
    },
    "bank_broad_private_depositories_marketable_proxy": {
        "bill_share": ("wamest_bank_broad_bill_share_l1", "share", "bank_capacity_state"),
        "short_share_le_1y": ("wamest_bank_broad_short_share_l1", "share", "bank_capacity_state"),
        "zero_coupon_equivalent_years": ("wamest_bank_broad_wam_years_l1", "years", "bank_capacity_state"),
    },
    "foreigners_total": {
        "short_share_le_1y": ("wamest_foreigners_short_share_l1", "share", "holder_maturity_state"),
        "zero_coupon_equivalent_years": ("wamest_foreigners_wam_years_l1", "years", "holder_maturity_state"),
    },
    "domestic_nonbank_residual_broad": {
        "short_share_le_1y": ("wamest_domestic_nonbank_short_share_l1", "share", "holder_maturity_state"),
        "zero_coupon_equivalent_years": ("wamest_domestic_nonbank_wam_years_l1", "years", "holder_maturity_state"),
    },
}


@dataclass(frozen=True)
class AdapterResult:
    standardized_path: Path
    manifest_path: Path
    rows_written: int
    bundle_hash: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(_sha256_file(path).encode("utf-8"))
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _quarter_from_period_end(period_end: str) -> str:
    date_value = datetime.fromisoformat(str(period_end)[:10]).date()
    quarter_num = ((date_value.month - 1) // 3) + 1
    return f"{date_value.year}Q{quarter_num}"


def _quarter_sort_key(quarter: str) -> tuple[int, int]:
    year_text, quarter_text = quarter.split("Q", 1)
    return int(year_text), int(quarter_text)


def _quarter_to_period_end(quarter: str) -> str:
    year, quarter_num = _quarter_sort_key(quarter)
    month_day = {
        1: (3, 31),
        2: (6, 30),
        3: (9, 30),
        4: (12, 31),
    }
    month, day = month_day[quarter_num]
    return date(year, month, day).isoformat()


def _previous_quarter(quarter: str) -> str | None:
    year, quarter_num = _quarter_sort_key(quarter)
    if quarter_num == 1:
        return f"{year - 1}Q4" if year > 1 else None
    return f"{year}Q{quarter_num - 1}"


def _conservative_available_at(quarter: str, *, lag_days: int = LAG_DAYS) -> str:
    period_end = datetime.strptime(_quarter_to_period_end(quarter), "%Y-%m-%d").date()
    return (period_end + timedelta(days=lag_days)).isoformat()


def resolve_seed_dir(paths: ProjectPaths, explicit_dir: str | None = None) -> Path:
    if explicit_dir:
        path = Path(explicit_dir).expanduser()
        if not path.is_absolute():
            path = (paths.root / path).resolve()
        return path
    return paths.seed / "wamest"


def _validate_source_dir(root: Path) -> tuple[Path, Path | None, Path | None]:
    panel_path = root / SECTOR_PANEL_PATH
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing required wamest artifact: {panel_path}")
    run_manifest_path = root / RUN_MANIFEST_PATH
    summary_path = root / SUMMARY_PATH
    return panel_path, (run_manifest_path if run_manifest_path.exists() else None), (summary_path if summary_path.exists() else None)


def _series_row(
    *,
    series_id: str,
    quarter: str,
    source_table: str,
    units: str,
    value: str,
    notes: str,
) -> dict[str, str]:
    previous = _previous_quarter(quarter)
    available_at = _conservative_available_at(previous, lag_days=LAG_DAYS) if previous else ""
    return {
        "series_id": series_id,
        "series_label": series_id,
        "source_family": "repo_publish",
        "source_repo": "wamest",
        "source_table": source_table,
        "freq": "quarterly",
        "period_end": _quarter_to_period_end(quarter),
        "release_date": available_at[:10],
        "available_at": available_at,
        "vintage_policy": "wamest_publish_snapshot_conservative_lagged_state",
        "units": units,
        "value": value,
        "transform_default": "none",
        "seasonal_adjustment_flag": "unknown",
        "interpolated_flag": "false",
        "component_group": "maturity_state",
        "role": "state",
        "notes": notes,
    }


def normalize_schema(sector_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    current_values: dict[tuple[str, str], str] = {}
    for row in sector_rows:
        sector_key = str(row.get("sector_key", "")).strip()
        if sector_key not in SECTOR_SERIES:
            continue
        in_publication_range = str(row.get("in_publication_range", "")).strip().lower()
        if in_publication_range and in_publication_range not in {"true", "1", "yes"}:
            continue
        quarter = _quarter_from_period_end(str(row.get("date", "")).strip())
        for field, (series_id, _units, _group) in SECTOR_SERIES[sector_key].items():
            value = str(row.get(field, "")).strip()
            if value:
                current_values[(series_id, quarter)] = value

    standardized_rows: list[dict[str, str]] = []
    for sector_key, field_map in SECTOR_SERIES.items():
        for field, (series_id, units, component_group) in field_map.items():
            quarter_values = {
                quarter: value
                for (current_series_id, quarter), value in current_values.items()
                if current_series_id == series_id
            }
            for quarter in sorted(quarter_values, key=_quarter_sort_key):
                previous = _previous_quarter(quarter)
                if previous is None or previous not in quarter_values:
                    continue
                standardized_rows.append(
                    {
                        **_series_row(
                            series_id=series_id,
                            quarter=quarter,
                            source_table="sector_effective_maturity_full",
                            units=units,
                            value=quarter_values[previous],
                            notes=f"lagged_{field}_from_wamest_{sector_key}",
                        ),
                        "component_group": component_group,
                    }
                )
    return standardized_rows


def validate_contract(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No standardized rows were produced from wamest artifacts")
    required_ids = {
        "wamest_bank_reserve_short_share_l1",
        "wamest_bank_reserve_wam_years_l1",
        "wamest_foreigners_wam_years_l1",
    }
    seen = {row["series_id"] for row in rows}
    missing = required_ids.difference(seen)
    if missing:
        raise KeyError(f"Missing required wamest standardized series: {sorted(missing)}")


def write_standard_bundle(paths: ProjectPaths, rows: list[dict[str, str]]) -> Path:
    bundle_dir = paths.bundles / "wamest"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    target = bundle_dir / "standardized_series.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return target


def write_source_manifest(
    paths: ProjectPaths,
    *,
    source_root: Path,
    standardized_path: Path,
    run_manifest_path: Path | None,
    summary_path: Path | None,
    rows_written: int,
    bundle_hash: str,
    sectors_covered: list[str],
) -> Path:
    run_manifest = _read_json(run_manifest_path) if run_manifest_path else {}
    full_summary = _read_json(summary_path) if summary_path else {}
    manifest = {
        "kind": "source_manifest",
        "source_repo": "wamest",
        "adapter": "wamest_outputs",
        "generated_at_utc": utc_now_iso(),
        "source_root": str(source_root),
        "standardized_path": str(standardized_path),
        "rows_written": rows_written,
        "bundle_hash": bundle_hash,
        "sectors_covered": sectors_covered,
        "resolved_latest_snapshot_date": str(run_manifest.get("resolved_latest_snapshot_date", "")),
        "run_timestamp_utc": str(run_manifest.get("run_timestamp_utc", "")),
        "schema_version": str(run_manifest.get("schema_version", "")),
        "latest_snapshot_summary_present": bool(full_summary.get("latest_snapshot_summary")),
    }
    target = paths.manifests / "wamest_source_manifest.json"
    write_json(target, manifest)
    return target


def adapt_wamest(paths: ProjectPaths, *, publish_dir: str | None = None) -> AdapterResult:
    source_root = resolve_seed_dir(paths, publish_dir)
    panel_path, run_manifest_path, summary_path = _validate_source_dir(source_root)
    rows = normalize_schema(_read_csv(panel_path))
    validate_contract(rows)
    standardized_path = write_standard_bundle(paths, rows)
    hash_paths = [panel_path]
    if run_manifest_path:
        hash_paths.append(run_manifest_path)
    if summary_path:
        hash_paths.append(summary_path)
    bundle_hash = _bundle_hash(hash_paths)
    manifest_path = write_source_manifest(
        paths,
        source_root=source_root,
        standardized_path=standardized_path,
        run_manifest_path=run_manifest_path,
        summary_path=summary_path,
        rows_written=len(rows),
        bundle_hash=bundle_hash,
        sectors_covered=sorted(SECTOR_SERIES.keys()),
    )
    return AdapterResult(
        standardized_path=standardized_path,
        manifest_path=manifest_path,
        rows_written=len(rows),
        bundle_hash=bundle_hash,
    )
