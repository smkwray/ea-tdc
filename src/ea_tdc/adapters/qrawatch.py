from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from ea_tdc.paths import ProjectPaths
from ea_tdc.utils import utc_now_iso, write_json


REQUIRED_FILES = (
    "ati_quarter_table.csv",
    "qra_event_shock_summary.csv",
    "qra_event_registry_v2.csv",
    "duration_supply_summary.csv",
)


@dataclass(frozen=True)
class AdapterResult:
    standardized_path: Path
    event_bundle_path: Path
    manifest_path: Path
    series_rows_written: int
    event_rows_written: int
    bundle_hash: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
    qnum = int(quarter[5])
    month_day = {
        1: (3, 31),
        2: (6, 30),
        3: (9, 30),
        4: (12, 31),
    }
    month, day = month_day[qnum]
    return date(year, month, day).isoformat()


def resolve_publish_dir(paths: ProjectPaths, explicit_dir: str | None = None) -> Path:
    if explicit_dir:
        path = Path(explicit_dir).expanduser()
        if not path.is_absolute():
            path = (paths.root / path).resolve()
        return path
    return paths.seed / "qrawatch"


def read_raw(publish_dir: Path) -> dict[str, list[dict[str, str]]]:
    missing = [name for name in REQUIRED_FILES if not (publish_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required qrawatch publish files: {missing}")
    return {name: _read_csv(publish_dir / name) for name in REQUIRED_FILES}


def _first_nonempty(row: dict[str, str], fields: list[str]) -> str:
    for field in fields:
        value = str(row.get(field, "")).strip()
        if value:
            return value
    return ""


def _normalize_bool_text(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return "true"
    if normalized in {"0", "false", "f", "no", "n"}:
        return "false"
    return ""


def _series_role(source_table: str, series_id: str) -> tuple[str, str]:
    if source_table == "ati_quarter_table":
        if series_id in {"net_bills_bn", "bill_share", "ati_baseline_bn"}:
            return "treatment", "debt_management_treatment"
        if series_id.startswith("missing_coupons_"):
            return "mechanism", "coupon_gap"
        return "control", "financing_context"
    if source_table == "duration_supply_summary":
        if series_id in {"headline_public_duration_supply", "provisional_public_duration_supply"}:
            return "treatment", "duration_supply"
        return "mechanism", "duration_support"
    return "mechanism", "unknown"


def normalize_series_schema(payload: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    registry_by_quarter = {}
    for row in payload["qra_event_registry_v2.csv"]:
        quarter = str(row.get("quarter", "")).strip()
        release_timestamp = str(row.get("release_timestamp_et", "")).strip()
        if quarter and release_timestamp and quarter not in registry_by_quarter:
            registry_by_quarter[quarter] = release_timestamp

    ati_fields = [
        "financing_need_bn",
        "net_bills_bn",
        "bill_share",
        "missing_coupons_15_bn",
        "missing_coupons_18_bn",
        "missing_coupons_20_bn",
        "ati_baseline_bn",
    ]
    for row in payload["ati_quarter_table.csv"]:
        quarter = str(row.get("quarter", "")).strip()
        period_end = _quarter_to_period_end(quarter)
        release_timestamp = registry_by_quarter.get(quarter, "")
        release_date = release_timestamp[:10] if release_timestamp else period_end
        available_at = release_timestamp or release_date
        vintage_policy = (
            "official_qra_release_timestamp"
            if release_timestamp
            else "publish_snapshot_quarter_end_fallback"
        )
        source_quality = str(row.get("source_quality", "")).strip()
        for field in ati_fields:
            role, component_group = _series_role("ati_quarter_table", field)
            rows.append(
                {
                    "series_id": field,
                    "series_label": field,
                    "source_family": "repo_publish",
                    "source_repo": "qrawatch",
                    "source_table": "ati_quarter_table",
                    "freq": "quarterly",
                    "period_end": period_end,
                    "release_date": release_date,
                    "available_at": available_at,
                    "vintage_policy": vintage_policy,
                    "units": "share" if field == "bill_share" else "usd_billions",
                    "value": str(row.get(field, "")).strip(),
                    "transform_default": "none",
                    "seasonal_adjustment_flag": "unknown",
                    "interpolated_flag": "false",
                    "component_group": component_group,
                    "role": role,
                    "notes": (source_quality or str(row.get("public_role", "")).strip()),
                }
            )

    duration_fields = [
        "coupon_like_total",
        "headline_public_duration_supply",
        "provisional_public_duration_supply",
        "qt_proxy",
        "buybacks_accepted",
    ]
    for row in payload["duration_supply_summary.csv"]:
        period_end = str(row.get("date", "")).strip()
        source_quality = _first_nonempty(row, ["headline_source_quality", "fallback_source_quality"])
        for field in duration_fields:
            role, component_group = _series_role("duration_supply_summary", field)
            rows.append(
                {
                    "series_id": field,
                    "series_label": field,
                    "source_family": "repo_publish",
                    "source_repo": "qrawatch",
                    "source_table": "duration_supply_summary",
                    "freq": "weekly",
                    "period_end": period_end,
                    "release_date": period_end,
                    "available_at": period_end,
                    "vintage_policy": "publish_snapshot",
                    "units": "usd_notional",
                    "value": str(row.get(field, "")).strip(),
                    "transform_default": "none",
                    "seasonal_adjustment_flag": "unknown",
                    "interpolated_flag": str(row.get("qt_proxy_is_zero_filled", "")).strip().lower(),
                    "component_group": component_group,
                    "role": role,
                    "notes": source_quality,
                }
            )

    return rows


def normalize_event_bundle(payload: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    registry_by_event = {
        str(row.get("event_id", "")).strip(): row
        for row in payload["qra_event_registry_v2.csv"]
        if str(row.get("event_id", "")).strip()
    }

    rows: list[dict[str, str]] = []
    for row in payload["qra_event_shock_summary.csv"]:
        event_id = str(row.get("event_id", "")).strip()
        registry = registry_by_event.get(event_id, {})
        event_date = _first_nonempty(row, ["event_date_aligned", "event_date_requested"])
        treatment_value = _first_nonempty(
            row,
            [
                "shock_bn",
                "schedule_diff_10y_eq_bn",
                "schedule_diff_dynamic_10y_eq_bn",
                "schedule_diff_dv01_usd",
                "gross_notional_delta_bn",
            ],
        )
        cutoff_timestamp = _first_nonempty(
            registry,
            ["release_timestamp_et"],
        ) or event_date
        notes = "|".join(
            [
                _first_nonempty(row, ["claim_scope"]),
                _first_nonempty(row, ["headline_bucket"]),
                _first_nonempty(row, ["shock_review_status"]),
                _first_nonempty(row, ["usable_for_headline_reason", "descriptive_headline_reason"]),
            ]
        ).strip("|")
        quality_tier = _first_nonempty(registry, ["quality_tier", "review_maturity"])
        release_component_count = _first_nonempty(registry, ["release_component_count"])
        causal_component_count = _first_nonempty(registry, ["causal_eligible_component_count"])

        rows.append(
            {
                "event_id": event_id,
                "quarter": _first_nonempty(row, ["quarter"]) or _first_nonempty(registry, ["quarter"]),
                "event_label": _first_nonempty(row, ["event_label"]),
                "event_date": event_date,
                "event_date_type": _first_nonempty(row, ["event_date_type"]),
                "event_type": "qra_release",
                "source_repo": "qrawatch",
                "treatment_id": _first_nonempty(row, ["treatment_variant"]) or "canonical_shock_bn",
                "treatment_value": treatment_value,
                "treatment_units": "usd_billions",
                "cutoff_timestamp": cutoff_timestamp,
                "embargo_rule": "event_close_with_embargo",
                "horizon_unit": "business_day",
                "release_timestamp_kind": _first_nonempty(registry, ["release_timestamp_kind"]),
                "release_bundle_type": _first_nonempty(registry, ["release_bundle_type"]),
                "timing_quality": _first_nonempty(row, ["timing_quality"]) or _first_nonempty(registry, ["timing_quality"]),
                "overlap_severity": _first_nonempty(row, ["overlap_severity"]) or _first_nonempty(registry, ["overlap_severity"]),
                "quality_tier": quality_tier,
                "headline_bucket": _first_nonempty(row, ["headline_bucket"]),
                "usable_for_headline": _normalize_bool_text(row.get("usable_for_headline", "")),
                "usable_for_headline_reason": _first_nonempty(row, ["usable_for_headline_reason"]),
                "usable_for_descriptive_headline": _normalize_bool_text(row.get("usable_for_descriptive_headline", "")),
                "descriptive_headline_reason": _first_nonempty(row, ["descriptive_headline_reason"]),
                "claim_scope": _first_nonempty(row, ["claim_scope"]),
                "shock_review_status": _first_nonempty(row, ["shock_review_status"]),
                "shock_missing_flag": _normalize_bool_text(row.get("shock_missing_flag", "")),
                "small_denominator_flag": _normalize_bool_text(row.get("small_denominator_flag", "")),
                "timestamp_precision": _first_nonempty(registry, ["timestamp_precision"]),
                "separability_status": _first_nonempty(registry, ["separability_status"]),
                "expectation_status": _first_nonempty(registry, ["expectation_status"]),
                "contamination_status": _first_nonempty(registry, ["contamination_status"]),
                "eligibility_blockers": _first_nonempty(registry, ["eligibility_blockers"]),
                "release_component_count": release_component_count,
                "causal_eligible_component_count": causal_component_count,
                "policy_statement_url": _first_nonempty(row, ["policy_statement_url"]) or _first_nonempty(registry, ["policy_statement_url"]),
                "financing_estimates_url": _first_nonempty(row, ["financing_estimates_url"]) or _first_nonempty(registry, ["financing_estimates_url"]),
                "notes": notes,
            }
        )
    return rows


def validate_contract(series_rows: list[dict[str, str]], event_rows: list[dict[str, str]]) -> None:
    if not series_rows:
        raise ValueError("No standardized qrawatch series rows were produced")
    if not event_rows:
        raise ValueError("No qrawatch event rows were produced")
    if not any(row["series_id"] == "ati_baseline_bn" and row["role"] == "treatment" for row in series_rows):
        raise ValueError("Missing quarterly ATI treatment rows")
    if not any(row["series_id"] == "headline_public_duration_supply" and row["role"] == "treatment" for row in series_rows):
        raise ValueError("Missing duration supply treatment rows")
    if not any(row["event_type"] == "qra_release" for row in event_rows):
        raise ValueError("Missing qra_release event rows")


def write_standard_bundle(paths: ProjectPaths, series_rows: list[dict[str, str]], event_rows: list[dict[str, str]]) -> tuple[Path, Path]:
    bundle_dir = paths.bundles / "qrawatch"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    standardized_path = bundle_dir / "standardized_series.csv"
    event_bundle_path = bundle_dir / "event_bundle.csv"

    with standardized_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(series_rows[0].keys()))
        writer.writeheader()
        writer.writerows(series_rows)

    with event_bundle_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(event_rows[0].keys()))
        writer.writeheader()
        writer.writerows(event_rows)

    return standardized_path, event_bundle_path


def write_source_manifest(
    paths: ProjectPaths,
    *,
    publish_dir: Path,
    standardized_path: Path,
    event_bundle_path: Path,
    series_rows_written: int,
    event_rows_written: int,
    bundle_hash: str,
) -> Path:
    with event_bundle_path.open("r", encoding="utf-8", newline="") as handle:
        event_rows = list(csv.DictReader(handle))
    usable_for_headline = sum(1 for row in event_rows if row.get("usable_for_headline") == "true")
    nonmissing_treatment = sum(1 for row in event_rows if str(row.get("treatment_value", "")).strip())
    manifest = {
        "kind": "source_manifest",
        "source_repo": "qrawatch",
        "adapter": "qrawatch_publish",
        "generated_at_utc": utc_now_iso(),
        "publish_dir": str(publish_dir),
        "standardized_path": str(standardized_path),
        "event_bundle_path": str(event_bundle_path),
        "series_rows_written": series_rows_written,
        "event_rows_written": event_rows_written,
        "event_rows_with_treatment": nonmissing_treatment,
        "event_rows_usable_for_headline": usable_for_headline,
        "bundle_hash": bundle_hash,
        "source_files": list(REQUIRED_FILES),
    }
    target = paths.manifests / "qrawatch_source_manifest.json"
    write_json(target, manifest)
    return target


def adapt_qrawatch(paths: ProjectPaths, *, publish_dir: str | None = None) -> AdapterResult:
    resolved_publish_dir = resolve_publish_dir(paths, publish_dir)
    payload = read_raw(resolved_publish_dir)
    series_rows = normalize_series_schema(payload)
    event_rows = normalize_event_bundle(payload)
    validate_contract(series_rows, event_rows)
    standardized_path, event_bundle_path = write_standard_bundle(paths, series_rows, event_rows)
    input_paths = [resolved_publish_dir / name for name in REQUIRED_FILES]
    bundle_hash = _combined_hash(input_paths)
    manifest_path = write_source_manifest(
        paths,
        publish_dir=resolved_publish_dir,
        standardized_path=standardized_path,
        event_bundle_path=event_bundle_path,
        series_rows_written=len(series_rows),
        event_rows_written=len(event_rows),
        bundle_hash=bundle_hash,
    )
    return AdapterResult(
        standardized_path=standardized_path,
        event_bundle_path=event_bundle_path,
        manifest_path=manifest_path,
        series_rows_written=len(series_rows),
        event_rows_written=len(event_rows),
        bundle_hash=bundle_hash,
    )
