from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from ea_tdc.paths import ProjectPaths
from ea_tdc.utils import utc_now_iso, write_json


REQUIRED_INFERENCE_FILES = (
    "counterparty_flows.csv",
    "manifest.json",
)
REQUIRED_SIMILARITY_FILES = (
    "rolling_correlations.csv",
    "manifest.json",
)
LAG_DAYS = 90
BUYER_BANKS = "banks"
BUYER_ROW_PROXY = "foreigners_official"
RESIDUAL_BUCKET = "_residual"
BEHAVIOR_CORRELATION_SPECS = {
    "tsyparty_bank_foreign_official_corr_l1": (
        "banks_vs_foreigners_official",
        "correlation",
        "lagged_rolling_partial_pearson_between_banks_and_foreigners_official_from_tsyparty_similarity_enriched",
    ),
    "tsyparty_bank_foreign_private_corr_l1": (
        "banks_vs_foreigners_private",
        "correlation",
        "lagged_rolling_partial_pearson_between_banks_and_foreigners_private_from_tsyparty_similarity_enriched",
    ),
    "tsyparty_bank_mmf_corr_l1": (
        "banks_vs_money_market_funds",
        "correlation",
        "lagged_rolling_partial_pearson_between_banks_and_money_market_funds_from_tsyparty_similarity_enriched",
    ),
}


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


def _quarter_sort_key(quarter: str) -> tuple[int, int]:
    year_text, quarter_text = quarter.split("Q", 1)
    return int(year_text), int(quarter_text)


def _previous_quarter(quarter: str) -> str | None:
    year, quarter_num = _quarter_sort_key(quarter)
    if quarter_num == 1:
        return f"{year - 1}Q4" if year > 1 else None
    return f"{year}Q{quarter_num - 1}"


def _parse_quarter(text: str) -> str:
    date_value = datetime.fromisoformat(str(text)[:10]).date()
    quarter = ((date_value.month - 1) // 3) + 1
    return f"{date_value.year}Q{quarter}"


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


def _conservative_available_at(quarter: str, *, lag_days: int = LAG_DAYS) -> str:
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
    return paths.seed / "tsyparty"


def _resolve_existing_dir(candidates: list[Path], required_files: tuple[str, ...]) -> Path | None:
    for candidate in candidates:
        if all((candidate / name).exists() for name in required_files):
            return candidate
    return None


def _inference_dir(publish_dir: Path) -> Path:
    candidates = [
        publish_dir,
        publish_dir / "outputs" / "inference",
        publish_dir / "site" / "data" / "outputs" / "inference",
    ]
    resolved = _resolve_existing_dir(candidates, REQUIRED_INFERENCE_FILES)
    if resolved is not None:
        return resolved
    return publish_dir


def _similarity_dir(publish_dir: Path) -> Path | None:
    candidates = [
        publish_dir / "outputs" / "similarity_enriched",
        publish_dir / "site" / "data" / "outputs" / "similarity_enriched",
        publish_dir / "outputs" / "similarity",
        publish_dir / "site" / "data" / "outputs" / "similarity",
    ]
    return _resolve_existing_dir(candidates, REQUIRED_SIMILARITY_FILES)


def read_raw(publish_dir: Path) -> dict[str, Any]:
    inference_dir = _inference_dir(publish_dir)
    missing = [name for name in REQUIRED_INFERENCE_FILES if not (inference_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required tsyparty publish files: {missing}")

    with (inference_dir / "counterparty_flows.csv").open("r", encoding="utf-8", newline="") as handle:
        counterparty_flows = list(csv.DictReader(handle))
    manifest = _read_json(inference_dir / "manifest.json")
    if not isinstance(manifest, dict):
        raise TypeError("tsyparty manifest.json must be a mapping")
    similarity_dir = _similarity_dir(publish_dir)
    rolling_correlations: list[dict[str, str]] = []
    similarity_manifest: dict[str, Any] | None = None
    if similarity_dir is not None:
        with (similarity_dir / "rolling_correlations.csv").open("r", encoding="utf-8", newline="") as handle:
            rolling_correlations = list(csv.DictReader(handle))
        similarity_manifest = _read_json(similarity_dir / "manifest.json")
        if not isinstance(similarity_manifest, dict):
            raise TypeError("tsyparty similarity manifest.json must be a mapping")
    return {
        "source_dir": inference_dir,
        "similarity_dir": similarity_dir,
        "counterparty_flows": counterparty_flows,
        "rolling_correlations": rolling_correlations,
        "manifest": manifest,
        "similarity_manifest": similarity_manifest,
    }


def _series_row(
    *,
    series_id: str,
    quarter: str,
    value: str,
    units: str,
    notes: str,
    source_table: str = "counterparty_flows",
    component_group: str = "absorption_state",
) -> dict[str, str]:
    available_at = _conservative_available_at(_previous_quarter(quarter), lag_days=LAG_DAYS) if _previous_quarter(quarter) else ""
    return {
        "series_id": series_id,
        "series_label": series_id,
        "source_family": "repo_publish",
        "source_repo": "tsyparty",
        "source_table": source_table,
        "freq": "quarterly",
        "period_end": _quarter_to_period_end(quarter),
        "release_date": available_at[:10],
        "available_at": available_at,
        "vintage_policy": "tsyparty_publish_snapshot_conservative_lagged_state",
        "units": units,
        "value": value,
        "transform_default": "none",
        "seasonal_adjustment_flag": "unknown",
        "interpolated_flag": "false",
        "component_group": component_group,
        "role": "state",
        "notes": notes,
    }


def normalize_schema(payload: dict[str, Any]) -> list[dict[str, str]]:
    buyer_amounts: dict[tuple[str, str], float] = defaultdict(float)
    totals_by_quarter: dict[str, float] = defaultdict(float)
    for row in payload["counterparty_flows"]:
        buyer = str(row.get("buyer", "")).strip()
        if not buyer or buyer == RESIDUAL_BUCKET:
            continue
        quarter = _parse_quarter(str(row.get("date", "")).strip())
        try:
            amount = float(str(row.get("amount", "")).strip())
        except ValueError:
            continue
        buyer_amounts[(quarter, buyer)] += amount
        totals_by_quarter[quarter] += amount

    current_bank_share: dict[str, float] = {}
    current_row_share: dict[str, float] = {}
    current_ru_gap: dict[str, float] = {}
    for quarter in sorted(totals_by_quarter, key=_quarter_sort_key):
        total_amount = totals_by_quarter[quarter]
        if total_amount <= 0:
            continue
        bank_share = buyer_amounts.get((quarter, BUYER_BANKS), 0.0) / total_amount
        row_share = buyer_amounts.get((quarter, BUYER_ROW_PROXY), 0.0) / total_amount
        current_bank_share[quarter] = bank_share
        current_row_share[quarter] = row_share
        current_ru_gap[quarter] = row_share - bank_share

    rows: list[dict[str, str]] = []
    for quarter in sorted(current_bank_share, key=_quarter_sort_key):
        previous = _previous_quarter(quarter)
        if previous is None:
            continue
        if previous not in current_bank_share:
            continue
        rows.append(
            _series_row(
                series_id="tsyparty_bank_absorption_share_l1",
                quarter=quarter,
                value=_stable_float_text(current_bank_share[previous], digits=6),
                units="share",
                notes="lagged_bank_buyer_share_from_tsyparty_counterparty_flows",
            )
        )
        rows.append(
            _series_row(
                series_id="tsyparty_row_absorption_share_l1",
                quarter=quarter,
                value=_stable_float_text(current_row_share[previous], digits=6),
                units="share",
                notes=(
                    "lagged_row_proxy_share_from_tsyparty_counterparty_flows;"
                    "current_public_publish_uses_foreigners_official_as_row_proxy"
                ),
            )
        )
        rows.append(
            _series_row(
                series_id="tsyparty_ru_gap_l1",
                quarter=quarter,
                value=_stable_float_text(current_ru_gap[previous], digits=6),
                units="share_gap",
                notes=(
                    "lagged_row_minus_bank_absorption_share_from_tsyparty_counterparty_flows;"
                    "row_proxy=foreigners_official"
                ),
            )
        )
    behavior_current: dict[str, dict[str, float]] = {series_id: {} for series_id in BEHAVIOR_CORRELATION_SPECS}
    private_minus_official_current: dict[str, float] = {}
    for correlation_row in payload.get("rolling_correlations", []):
        quarter = _parse_quarter(str(correlation_row.get("date", "")).strip())
        for series_id, (column_name, _, _) in BEHAVIOR_CORRELATION_SPECS.items():
            try:
                behavior_current[series_id][quarter] = float(str(correlation_row.get(column_name, "")).strip())
            except ValueError:
                continue
        try:
            private_value = float(str(correlation_row.get("banks_vs_foreigners_private", "")).strip())
            official_value = float(str(correlation_row.get("banks_vs_foreigners_official", "")).strip())
        except ValueError:
            continue
        private_minus_official_current[quarter] = private_value - official_value

    for series_id, (_, units, notes) in BEHAVIOR_CORRELATION_SPECS.items():
        current_values = behavior_current[series_id]
        for quarter in sorted(current_values, key=_quarter_sort_key):
            previous = _previous_quarter(quarter)
            if previous is None or previous not in current_values:
                continue
            rows.append(
                _series_row(
                    series_id=series_id,
                    quarter=quarter,
                    value=_stable_float_text(current_values[previous], digits=6),
                    units=units,
                    notes=notes,
                    source_table="rolling_correlations",
                    component_group="behavior_state",
                )
            )
    for quarter in sorted(private_minus_official_current, key=_quarter_sort_key):
        previous = _previous_quarter(quarter)
        if previous is None or previous not in private_minus_official_current:
            continue
        rows.append(
            _series_row(
                series_id="tsyparty_private_minus_official_corr_l1",
                quarter=quarter,
                value=_stable_float_text(private_minus_official_current[previous], digits=6),
                units="correlation_gap",
                notes=(
                    "lagged_difference_between_bank_foreign_private_and_bank_foreigners_official"
                    "_rolling_partial_pearson_from_tsyparty_similarity_enriched"
                ),
                source_table="rolling_correlations",
                component_group="behavior_state",
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
        raise ValueError("No standardized rows were produced from tsyparty")
    missing = required.difference(rows[0])
    if missing:
        raise KeyError(f"Standardized rows are missing required fields: {sorted(missing)}")
    for series_id in (
        "tsyparty_bank_absorption_share_l1",
        "tsyparty_row_absorption_share_l1",
        "tsyparty_ru_gap_l1",
    ):
        if not any(row["series_id"] == series_id for row in rows):
            raise ValueError(f"Missing required tsyparty standardized series: {series_id}")


def write_standard_bundle(paths: ProjectPaths, rows: list[dict[str, str]]) -> Path:
    bundle_dir = paths.bundles / "tsyparty"
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
    source_path: Path,
    standardized_path: Path,
    rows_written: int,
    bundle_hash: str,
    manifest_payload: dict[str, Any],
    similarity_manifest_payload: dict[str, Any] | None,
    similarity_source_path: Path | None,
) -> Path:
    manifest = {
        "kind": "source_manifest",
        "source_repo": "tsyparty",
        "adapter": "tsyparty_outputs",
        "generated_at_utc": utc_now_iso(),
        "source_path": str(source_path),
        "standardized_path": str(standardized_path),
        "rows_written": rows_written,
        "bundle_hash": bundle_hash,
        "upstream_build_timestamp": manifest_payload.get("build_timestamp"),
        "quarters_processed": manifest_payload.get("quarters_processed"),
        "quarters_skipped": manifest_payload.get("quarters_skipped"),
        "claims_label": manifest_payload.get("claims_label"),
        "similarity_source_path": str(similarity_source_path) if similarity_source_path else "",
        "similarity_build_timestamp": similarity_manifest_payload.get("build_timestamp") if similarity_manifest_payload else None,
        "similarity_targets_found": similarity_manifest_payload.get("targets_found") if similarity_manifest_payload else [],
        "similarity_pipeline": similarity_manifest_payload.get("pipeline") if similarity_manifest_payload else None,
    }
    target = paths.manifests / "tsyparty_source_manifest.json"
    write_json(target, manifest)
    return target


def adapt_tsyparty(paths: ProjectPaths, *, publish_dir: str | None = None) -> AdapterResult:
    payload = read_raw(resolve_publish_dir(paths, publish_dir))
    rows = normalize_schema(payload)
    validate_contract(rows)
    standardized_path = write_standard_bundle(paths, rows)
    source_dir = Path(payload["source_dir"])
    bundle_hash_paths = [source_dir / name for name in REQUIRED_INFERENCE_FILES]
    similarity_dir = payload.get("similarity_dir")
    if isinstance(similarity_dir, Path):
        bundle_hash_paths.extend(similarity_dir / name for name in REQUIRED_SIMILARITY_FILES)
    bundle_hash = _combined_hash(bundle_hash_paths)
    manifest_path = write_source_manifest(
        paths,
        source_path=source_dir,
        standardized_path=standardized_path,
        rows_written=len(rows),
        bundle_hash=bundle_hash,
        manifest_payload=payload["manifest"],
        similarity_manifest_payload=payload.get("similarity_manifest"),
        similarity_source_path=similarity_dir if isinstance(similarity_dir, Path) else None,
    )
    return AdapterResult(
        standardized_path=standardized_path,
        manifest_path=manifest_path,
        rows_written=len(rows),
        bundle_hash=bundle_hash,
    )
