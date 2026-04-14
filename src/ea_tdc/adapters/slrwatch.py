from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from ea_tdc.paths import ProjectPaths
from ea_tdc.utils import utc_now_iso, write_json


CONSTRAINT_PANEL_PATH = Path("output/reports/constraint_decomposition/prepared_panel.csv")
REGIME_PANEL_PATH = Path("output/reports/policy_regime_panel/regime_quarter_panel.csv")
LAG_DAYS = 90

PRESSURE_SOURCE_MAP = {
    "insured_bank": "bank",
    "parent_or_ihc": "parent",
}

PRESSURE_METRICS = {
    "leverage_pressure_score": ("leverage_pressure", "score"),
    "duration_pressure_score": ("duration_pressure", "score"),
    "funding_pressure_score": ("funding_pressure", "score"),
}

DOMINANT_CONSTRAINT_SERIES = {
    "leverage": ("bank_leverage_dominant_share", "share"),
    "duration_loss": ("bank_duration_loss_dominant_share", "share"),
    "funding": ("bank_funding_dominant_share", "share"),
}

HEADROOM_SERIES = {
    "bank_headroom_pp_mean": ("bank_headroom_pp", "share_points"),
    "parent_headroom_pp_mean": ("parent_headroom_pp", "share_points"),
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


def _combined_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(_sha256_file(path).encode("utf-8"))
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _quarter_from_date(text: str) -> str:
    date_value = datetime.fromisoformat(str(text)[:10]).date()
    quarter_num = ((date_value.month - 1) // 3) + 1
    return f"{date_value.year}Q{quarter_num}"


def _previous_quarter(quarter: str) -> str | None:
    year, quarter_num = _quarter_sort_key(quarter)
    if quarter_num == 1:
        return f"{year - 1}Q4" if year > 1 else None
    return f"{year}Q{quarter_num - 1}"


def _conservative_available_at(quarter: str, *, lag_days: int = LAG_DAYS) -> str:
    period_end = datetime.strptime(_quarter_to_period_end(quarter), "%Y-%m-%d").date()
    return (period_end + timedelta(days=lag_days)).isoformat()


def _coerce_float(value: str | None) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _stable_float_text(value: float, *, digits: int = 6) -> str:
    return str(round(float(value), digits))


def resolve_publish_dir(paths: ProjectPaths, explicit_dir: str | None = None) -> Path:
    if explicit_dir:
        path = Path(explicit_dir).expanduser()
        if not path.is_absolute():
            path = (paths.root / path).resolve()
        return path
    return paths.seed / "slrwatch"


def read_raw(publish_dir: Path) -> dict[str, object]:
    constraint_panel_path = publish_dir / CONSTRAINT_PANEL_PATH
    regime_panel_path = publish_dir / REGIME_PANEL_PATH
    missing = [str(path) for path in [constraint_panel_path, regime_panel_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required slrwatch report artifacts: {missing}")
    return {
        "source_root": publish_dir,
        "constraint_panel_path": constraint_panel_path,
        "regime_panel_path": regime_panel_path,
        "constraint_rows": _read_csv(constraint_panel_path),
        "regime_rows": _read_csv(regime_panel_path),
    }


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
        "source_repo": "slrwatch",
        "source_table": source_table,
        "freq": "quarterly",
        "period_end": _quarter_to_period_end(quarter),
        "release_date": available_at[:10],
        "available_at": available_at,
        "vintage_policy": "slrwatch_publish_snapshot_conservative_lagged_state",
        "units": units,
        "value": value,
        "transform_default": "none",
        "seasonal_adjustment_flag": "unknown",
        "interpolated_flag": "false",
        "component_group": "slr_constraint_state",
        "role": "state",
        "notes": notes,
    }


def _lagged_rows_from_current(
    *,
    series_id: str,
    source_table: str,
    units: str,
    current_values: dict[str, float],
    notes: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for quarter in sorted(current_values, key=_quarter_sort_key):
        previous = _previous_quarter(quarter)
        if previous is None or previous not in current_values:
            continue
        rows.append(
            _series_row(
                series_id=series_id,
                quarter=quarter,
                source_table=source_table,
                units=units,
                value=_stable_float_text(current_values[previous]),
                notes=notes,
            )
        )
    return rows


def normalize_schema(payload: dict[str, object]) -> list[dict[str, str]]:
    constraint_rows = list(payload["constraint_rows"])
    regime_rows = list(payload["regime_rows"])

    metric_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    dominant_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for row in constraint_rows:
        quarter_raw = str(row.get("quarter_end", "")).strip()
        source_raw = str(row.get("entity_source", "")).strip()
        source_prefix = PRESSURE_SOURCE_MAP.get(source_raw)
        if not quarter_raw or not source_prefix:
            continue
        quarter = _quarter_from_date(quarter_raw)
        for metric_name in PRESSURE_METRICS:
            value = _coerce_float(row.get(metric_name))
            if value is not None:
                metric_values[(source_prefix, metric_name, quarter)].append(value)
        dominant_constraint = str(row.get("dominant_constraint", "")).strip()
        if source_prefix == "bank" and dominant_constraint in DOMINANT_CONSTRAINT_SERIES:
            dominant_counts[quarter][dominant_constraint] += 1

    current_series: dict[tuple[str, str], dict[str, float]] = {}
    for (source_prefix, metric_name, quarter), values in metric_values.items():
        current_series.setdefault((source_prefix, metric_name), {})[quarter] = sum(values) / len(values)

    for quarter, counts in dominant_counts.items():
        total = sum(counts.values())
        if total <= 0:
            continue
        for constraint_name in DOMINANT_CONSTRAINT_SERIES:
            current_series.setdefault(("bank", constraint_name), {})[quarter] = counts[constraint_name] / total

    for row in regime_rows:
        quarter_raw = str(row.get("quarter_end", "")).strip()
        if not quarter_raw:
            continue
        quarter = _quarter_from_date(quarter_raw)
        for column_name, (series_suffix, _units) in HEADROOM_SERIES.items():
            value = _coerce_float(row.get(column_name))
            if value is None:
                continue
            source_prefix = "bank" if column_name.startswith("bank_") else "parent"
            current_series.setdefault((source_prefix, column_name), {})[quarter] = value

    standardized_rows: list[dict[str, str]] = []
    for source_prefix, source_label in (("bank", "insured_bank"), ("parent", "parent_or_ihc")):
        for metric_name, (series_suffix, units) in PRESSURE_METRICS.items():
            current_values = current_series.get((source_prefix, metric_name), {})
            standardized_rows.extend(
                _lagged_rows_from_current(
                    series_id=f"slrwatch_{source_prefix}_{series_suffix}_l1",
                    source_table="constraint_decomposition_prepared_panel",
                    units=units,
                    current_values=current_values,
                    notes=f"lagged_mean_{metric_name}_from_slrwatch_{source_label}_constraint_panel",
                )
            )

    for constraint_name, (series_suffix, units) in DOMINANT_CONSTRAINT_SERIES.items():
        standardized_rows.extend(
            _lagged_rows_from_current(
                series_id=f"slrwatch_{series_suffix}_l1",
                source_table="constraint_decomposition_prepared_panel",
                units=units,
                current_values=current_series.get(("bank", constraint_name), {}),
                notes=f"lagged_share_of_bank_obs_with_{constraint_name}_dominant_constraint_from_slrwatch",
            )
        )

    for source_prefix, source_label in (("bank", "insured_bank"), ("parent", "parent_or_ihc")):
        column_name = "bank_headroom_pp_mean" if source_prefix == "bank" else "parent_headroom_pp_mean"
        series_suffix, units = HEADROOM_SERIES[column_name]
        standardized_rows.extend(
            _lagged_rows_from_current(
                series_id=f"slrwatch_{series_suffix}_l1",
                source_table="policy_regime_panel",
                units=units,
                current_values=current_series.get((source_prefix, column_name), {}),
                notes=f"lagged_mean_headroom_pp_from_slrwatch_{source_label}_policy_regime_panel",
            )
        )
    return standardized_rows


def validate_contract(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No standardized rows were produced from slrwatch artifacts")
    required_ids = {
        "slrwatch_bank_leverage_pressure_l1",
        "slrwatch_bank_duration_pressure_l1",
        "slrwatch_bank_headroom_pp_l1",
        "slrwatch_parent_leverage_pressure_l1",
    }
    seen = {row["series_id"] for row in rows}
    missing = required_ids.difference(seen)
    if missing:
        raise KeyError(f"Missing required slrwatch standardized series: {sorted(missing)}")


def write_standard_bundle(paths: ProjectPaths, rows: list[dict[str, str]]) -> Path:
    bundle_dir = paths.bundles / "slrwatch"
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
    constraint_panel_path: Path,
    regime_panel_path: Path,
    rows_written: int,
    bundle_hash: str,
    constraint_rows: list[dict[str, str]],
    regime_rows: list[dict[str, str]],
) -> Path:
    quarters = sorted(
        {
            _quarter_from_date(str(row.get("quarter_end", "")).strip())
            for row in [*constraint_rows, *regime_rows]
            if str(row.get("quarter_end", "")).strip()
        },
        key=_quarter_sort_key,
    )
    manifest = {
        "kind": "source_manifest",
        "source_repo": "slrwatch",
        "adapter": "slrwatch_reports",
        "generated_at_utc": utc_now_iso(),
        "source_root": str(source_root),
        "constraint_panel_path": str(constraint_panel_path),
        "regime_panel_path": str(regime_panel_path),
        "standardized_path": str(standardized_path),
        "rows_written": rows_written,
        "bundle_hash": bundle_hash,
        "quarters_covered": len(quarters),
        "first_quarter": quarters[0] if quarters else "",
        "last_quarter": quarters[-1] if quarters else "",
        "constraint_observations": len(constraint_rows),
        "regime_quarters": len(regime_rows),
    }
    target = paths.manifests / "slrwatch_source_manifest.json"
    write_json(target, manifest)
    return target


def adapt_slrwatch(paths: ProjectPaths, *, publish_dir: str | None = None) -> AdapterResult:
    payload = read_raw(resolve_publish_dir(paths, publish_dir))
    rows = normalize_schema(payload)
    validate_contract(rows)
    standardized_path = write_standard_bundle(paths, rows)
    constraint_panel_path = Path(payload["constraint_panel_path"])
    regime_panel_path = Path(payload["regime_panel_path"])
    bundle_hash = _combined_hash([constraint_panel_path, regime_panel_path])
    manifest_path = write_source_manifest(
        paths,
        source_root=Path(payload["source_root"]),
        standardized_path=standardized_path,
        constraint_panel_path=constraint_panel_path,
        regime_panel_path=regime_panel_path,
        rows_written=len(rows),
        bundle_hash=bundle_hash,
        constraint_rows=list(payload["constraint_rows"]),
        regime_rows=list(payload["regime_rows"]),
    )
    return AdapterResult(
        standardized_path=standardized_path,
        manifest_path=manifest_path,
        rows_written=len(rows),
        bundle_hash=bundle_hash,
    )
