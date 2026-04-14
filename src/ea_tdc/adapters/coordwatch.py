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


REQUIRED_FILES = (
    "quarterly_panel.json",
    "quarterly_descriptive.json",
    "summary.json",
)


@dataclass(frozen=True)
class AdapterResult:
    standardized_path: Path
    manifest_path: Path
    rows_written: int
    bundle_hash: str


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(_sha256_file(path).encode("utf-8"))
    return digest.hexdigest()


def _quarter_to_period_end(quarter: str) -> str:
    if len(quarter) != 6 or quarter[4] != "Q":
        return quarter
    year = int(quarter[:4])
    quarter_num = int(quarter[5])
    month_day = {
        1: (3, 31),
        2: (6, 30),
        3: (9, 30),
        4: (12, 31),
    }
    month, day = month_day[quarter_num]
    return date(year, month, day).isoformat()


def _quarter_sort_key(quarter: str) -> tuple[int, int]:
    year_text, quarter_text = quarter.split("Q", 1)
    return int(year_text), int(quarter_text)


def _previous_quarter(quarter: str) -> str | None:
    year, quarter_num = _quarter_sort_key(quarter)
    if quarter_num == 1:
        return f"{year - 1}Q4" if year > 1 else None
    return f"{year}Q{quarter_num - 1}"


def _conservative_quarterly_available_at(quarter: str, *, lag_days: int = 14) -> str:
    period_end = datetime.strptime(_quarter_to_period_end(quarter), "%Y-%m-%d").date()
    return (period_end + timedelta(days=lag_days)).isoformat()


def _stable_float_text(value: float, *, digits: int = 10) -> str:
    return str(round(float(value), digits))


def resolve_publish_dir(paths: ProjectPaths, explicit_dir: str | None = None) -> Path:
    if explicit_dir:
        path = Path(explicit_dir).expanduser()
        if not path.is_absolute():
            path = (paths.root / path).resolve()
        return path
    return paths.seed / "coordwatch"


def read_raw(publish_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FILES if not (publish_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required coordwatch publish files: {missing}")

    quarterly_panel = _read_json(publish_dir / "quarterly_panel.json")
    quarterly_descriptive = _read_json(publish_dir / "quarterly_descriptive.json")
    summary = _read_json(publish_dir / "summary.json")
    if not isinstance(quarterly_panel, list):
        raise TypeError("coordwatch quarterly_panel.json must be a list")
    if not isinstance(quarterly_descriptive, list):
        raise TypeError("coordwatch quarterly_descriptive.json must be a list")
    if not isinstance(summary, dict):
        raise TypeError("coordwatch summary.json must be a mapping")
    return {
        "quarterly_panel": quarterly_panel,
        "quarterly_descriptive": quarterly_descriptive,
        "summary": summary,
    }


def _normalize_bool_text(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return "true"
    if text in {"0", "false", "f", "no", "n"}:
        return "false"
    return ""


def _series_row(
    *,
    series_id: str,
    quarter: str,
    source_table: str,
    units: str,
    value: Any,
    available_at: str,
    role: str,
    component_group: str,
    notes: str,
) -> dict[str, str]:
    return {
        "series_id": series_id,
        "series_label": series_id,
        "source_family": "repo_publish",
        "source_repo": "coordwatch",
        "source_table": source_table,
        "freq": "quarterly",
        "period_end": _quarter_to_period_end(quarter),
        "release_date": available_at[:10],
        "available_at": available_at,
        "vintage_policy": "coordwatch_publish_snapshot_conservative_quarter_lag",
        "units": units,
        "value": "" if value is None else str(value),
        "transform_default": "none",
        "seasonal_adjustment_flag": "unknown",
        "interpolated_flag": "false",
        "component_group": component_group,
        "role": role,
        "notes": notes,
    }


def _derive_on_rrp_drain_state(rows: list[dict[str, Any]]) -> dict[str, str]:
    ordered: list[tuple[str, float]] = []
    for row in sorted(rows, key=lambda item: _quarter_sort_key(str(item.get("quarter", "")))):
        quarter = str(row.get("quarter", "")).strip()
        raw_value = row.get("on_rrp_share")
        if not quarter or raw_value in (None, ""):
            continue
        try:
            ordered.append((quarter, float(raw_value)))
        except (TypeError, ValueError):
            continue
    if not ordered:
        return {}

    values = [value for _, value in ordered]
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    std_value = variance ** 0.5

    current: dict[str, str] = {}
    for quarter, value in ordered:
        score = (mean_value - value) / std_value if std_value else 0.0
        current[quarter] = _stable_float_text(score, digits=4)

    lagged: dict[str, str] = {}
    for quarter, _ in ordered:
        previous = _previous_quarter(quarter)
        if previous and previous in current:
            lagged[quarter] = current[previous]
    return lagged


def normalize_schema(payload: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    descriptive_by_quarter = {
        str(row.get("quarter", "")).strip(): row
        for row in payload["quarterly_descriptive"]
        if str(row.get("quarter", "")).strip()
    }
    on_rrp_drain_state = _derive_on_rrp_drain_state(payload["quarterly_descriptive"])

    for row in payload["quarterly_panel"]:
        quarter = str(row.get("quarter", "")).strip()
        if not quarter:
            continue
        previous = _previous_quarter(quarter)
        lagged_available_at = (
            _conservative_quarterly_available_at(previous, lag_days=14)
            if previous
            else ""
        )
        rows.append(
            _series_row(
                series_id="coord_low_reserve_state_l1",
                quarter=quarter,
                source_table="quarterly_panel",
                units="binary_or_index",
                value=row.get("low_liquidity_prev"),
                available_at=lagged_available_at,
                role="state",
                component_group="liquidity_state",
                notes="published_low_liquidity_prev_from_coordwatch_quarterly_panel",
            )
        )
        rows.append(
            _series_row(
                series_id="coord_liquidity_tightness_q_z_l1",
                quarter=quarter,
                source_table="quarterly_panel",
                units="zscore",
                value=row.get("liquidity_tightness_q_z_prev"),
                available_at=lagged_available_at,
                role="state",
                component_group="liquidity_state",
                notes="published_liquidity_tightness_q_z_prev_from_coordwatch_quarterly_panel",
            )
        )
        rows.append(
            _series_row(
                series_id="coord_system_liquidity_q_bn",
                quarter=quarter,
                source_table="quarterly_panel",
                units="usd_billions",
                value=row.get("system_liquidity_q_bn"),
                available_at=_conservative_quarterly_available_at(quarter, lag_days=14),
                role="mechanism",
                component_group="liquidity_level",
                notes="published_system_liquidity_q_bn_from_coordwatch_quarterly_panel",
            )
        )

    for quarter, row in descriptive_by_quarter.items():
        current_available_at = _conservative_quarterly_available_at(quarter, lag_days=14)
        for series_id, field, units, role, component_group in (
            ("coord_on_rrp_share_q", "on_rrp_share", "share", "mechanism", "on_rrp_buffer"),
            ("coord_reserves_bn_q", "reserves_bn_q", "usd_billions", "mechanism", "reserve_level"),
            ("coord_on_rrp_bn_q", "on_rrp_bn_q", "usd_billions", "mechanism", "on_rrp_buffer"),
            ("coord_repo_spread_bp_q", "repo_spread_bp_q", "basis_points", "outcome", "funding_outcome"),
            ("coord_dealer_inventory_bn_q", "dealer_inventory_bn_q", "usd_billions", "mechanism", "dealer_balance_sheet"),
            ("coord_net_private_duration_dv01_q", "net_private_duration_dv01", "dv01_usd_millions", "mechanism", "duration_supply"),
        ):
            rows.append(
                _series_row(
                    series_id=series_id,
                    quarter=quarter,
                    source_table="quarterly_descriptive",
                    units=units,
                    value=row.get(field),
                    available_at=current_available_at,
                    role=role,
                    component_group=component_group,
                    notes=f"published_{field}_from_coordwatch_quarterly_descriptive",
                )
            )
        rows.append(
            _series_row(
                series_id="coord_on_rrp_drain_state_l1",
                quarter=quarter,
                source_table="quarterly_descriptive",
                units="zscore",
                value=on_rrp_drain_state.get(quarter, ""),
                available_at=(
                    _conservative_quarterly_available_at(_previous_quarter(quarter), lag_days=14)
                    if _previous_quarter(quarter)
                    else ""
                ),
                role="state",
                component_group="liquidity_state",
                notes="derived_inverse_zscore_of_lagged_on_rrp_share_from_coordwatch_quarterly_descriptive",
            )
        )
        rows.append(
            _series_row(
                series_id="coord_debt_limit_flag_q",
                quarter=quarter,
                source_table="quarterly_descriptive",
                units="binary",
                value=_normalize_bool_text(row.get("debt_limit_flag")),
                available_at=current_available_at,
                role="control",
                component_group="debt_limit_context",
                notes="published_debt_limit_flag_from_coordwatch_quarterly_descriptive",
            )
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
        raise ValueError("No standardized rows were produced from coordwatch")
    missing = required.difference(rows[0])
    if missing:
        raise KeyError(f"Standardized rows are missing required fields: {sorted(missing)}")
    for series_id in (
        "coord_low_reserve_state_l1",
        "coord_liquidity_tightness_q_z_l1",
        "coord_on_rrp_drain_state_l1",
    ):
        if not any(row["series_id"] == series_id for row in rows):
            raise ValueError(f"Missing required coordwatch standardized series: {series_id}")


def write_standard_bundle(paths: ProjectPaths, rows: list[dict[str, str]]) -> Path:
    bundle_dir = paths.bundles / "coordwatch"
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
    summary: dict[str, Any],
) -> Path:
    manifest = {
        "kind": "source_manifest",
        "source_repo": "coordwatch",
        "adapter": "coordwatch_outputs",
        "generated_at_utc": utc_now_iso(),
        "source_path": str(source_path),
        "standardized_path": str(standardized_path),
        "rows_written": rows_written,
        "bundle_hash": bundle_hash,
        "upstream_generated_at_utc": summary.get("generated_at_utc"),
        "quarter_rows": summary.get("quarter_rows"),
        "weekly_rows": summary.get("weekly_rows"),
    }
    target = paths.manifests / "coordwatch_source_manifest.json"
    write_json(target, manifest)
    return target


def adapt_coordwatch(paths: ProjectPaths, *, publish_dir: str | None = None) -> AdapterResult:
    source_path = resolve_publish_dir(paths, publish_dir)
    payload = read_raw(source_path)
    rows = normalize_schema(payload)
    validate_contract(rows)
    standardized_path = write_standard_bundle(paths, rows)
    bundle_hash = _combined_hash([source_path / name for name in REQUIRED_FILES])
    manifest_path = write_source_manifest(
        paths,
        source_path=source_path,
        standardized_path=standardized_path,
        rows_written=len(rows),
        bundle_hash=bundle_hash,
        summary=payload["summary"],
    )
    return AdapterResult(
        standardized_path=standardized_path,
        manifest_path=manifest_path,
        rows_written=len(rows),
        bundle_hash=bundle_hash,
    )
