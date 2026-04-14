from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from ea_tdc.calendar import add_us_market_business_days, previous_us_market_business_day
from ea_tdc.paths import ProjectPaths
from ea_tdc.utils import utc_now_iso, write_json


PLACEBO_WINDOWS_BD = [(-21, -1), (-5, -1)]
DFF_SERIES_ID = "DFF"
DEFAULT_EVENT_SAMPLE_POLICY = "headline_strict"


EVENT_OUTCOME_SPECS = {
    "threefytp10": {"kind": "direct", "series": "THREEFYTP10", "transform": "difference", "scale": 100.0},
    "dgs2": {"kind": "direct", "series": "DGS2", "transform": "difference", "scale": 100.0},
    "dgs10": {"kind": "direct", "series": "DGS10", "transform": "difference", "scale": 100.0},
    "term_spread_10y_3m": {
        "kind": "spread",
        "left": "DGS10",
        "right": "DGS3MO",
        "transform": "difference",
        "scale": 100.0,
    },
    "repo_spread": {
        "kind": "spread",
        "left": "TGCRRATE",
        "right": "RRPONTSYAWARD",
        "transform": "difference",
        "scale": 100.0,
    },
    "sp500_return": {"kind": "direct", "series": "SP500", "transform": "pct_change", "scale": 100.0},
    "vix_change": {"kind": "direct", "series": "VIXCLS", "transform": "difference", "scale": 1.0},
    "tga_balance_change": {"kind": "direct", "series": "WDTGAL", "transform": "difference", "scale": 1.0},
    "reserve_balances_change": {"kind": "direct", "series": "WRESBAL", "transform": "difference", "scale": 1.0},
    "rrp_balance_change": {"kind": "direct", "series": "RRPONTSYD", "transform": "difference", "scale": 1.0},
    "fed_balance_sheet_change": {"kind": "direct", "series": "WALCL", "transform": "difference", "scale": 1.0},
}

EVENT_CONTROL_SPECS = {
    "dff": {"kind": "direct", "series": DFF_SERIES_ID, "transform": "difference", "scale": 1.0},
    "sofr": {"kind": "direct", "series": "SOFR", "transform": "difference", "scale": 1.0},
    "threefytp10": EVENT_OUTCOME_SPECS["threefytp10"],
    "dgs2": EVENT_OUTCOME_SPECS["dgs2"],
    "dgs10": EVENT_OUTCOME_SPECS["dgs10"],
    "term_spread_10y_3m": EVENT_OUTCOME_SPECS["term_spread_10y_3m"],
    "repo_spread": EVENT_OUTCOME_SPECS["repo_spread"],
    "tga_balance_change": EVENT_OUTCOME_SPECS["tga_balance_change"],
    "reserve_balances_change": EVENT_OUTCOME_SPECS["reserve_balances_change"],
}
EVENT_CONTROL_CANDIDATES = [
    "dff",
    "sofr",
    "threefytp10",
    "dgs2",
    "dgs10",
    "term_spread_10y_3m",
    "repo_spread",
    "tga_balance_change",
    "reserve_balances_change",
]
DEFAULT_EVENT_CONTROL_SELECTION_POLICY = "catalog_auto"


@dataclass(frozen=True)
class EventDesignBuildResult:
    bundle_path: Path
    design_manifest_path: Path
    sample_manifest_path: Path
    rows_written: int
    usable_rows: int


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_jobs(config_path: Path) -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    jobs = payload.get("jobs", [])
    if not isinstance(jobs, list):
        raise TypeError("Expected 'jobs' list in dass blueprint")
    return {
        str(item["job_id"]): item
        for item in jobs
        if isinstance(item, dict) and item.get("job_id")
    }


def _parse_date(value: str) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    return datetime.fromisoformat(text[:10]).date()


def _stable_float_text(value: float, *, digits: int = 10) -> str:
    return str(round(float(value), digits))


def _load_daily_series(raw_dir: Path, series_id: str) -> list[tuple[date, float]]:
    return _load_daily_series_from_path(raw_dir / f"{series_id}.csv")


def _load_daily_series_from_path(path: Path) -> list[tuple[date, float]]:
    if not path.exists():
        return []
    rows: list[tuple[date, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            date_value = _parse_date(str(row.get("date", "")))
            raw_value = str(row.get("value", "")).strip()
            if date_value is None or not raw_value or raw_value == ".":
                continue
            try:
                rows.append((date_value, float(raw_value)))
            except ValueError:
                continue
    rows.sort(key=lambda item: item[0])
    return rows


def _load_daily_series_candidates(paths: ProjectPaths, series_id: str) -> list[tuple[date, float]]:
    candidates = [
        paths.raw_fred / f"{series_id}.csv",
        paths.seed / "qrawatch_raw_fred" / f"{series_id}.csv",
    ]
    interpol_raw = paths.seed / "interpol" / "raw"
    if interpol_raw.exists():
        candidates.extend(sorted(interpol_raw.glob(f"FRED_{series_id}_*.csv")))
    for candidate in candidates:
        rows = _load_daily_series_from_path(candidate)
        if rows:
            return rows
    return []


def _first_available_on_or_after(observations: list[tuple[date, float]], target: date) -> tuple[date, float] | None:
    for observed_date, value in observations:
        if observed_date >= target:
            return observed_date, value
    return None


def _build_outcome_levels(paths: ProjectPaths, outcome_id: str) -> list[tuple[date, float]]:
    spec = EVENT_OUTCOME_SPECS[outcome_id]
    if spec["kind"] == "direct":
        return _load_daily_series_candidates(paths, str(spec["series"]))

    left = dict(_load_daily_series_candidates(paths, str(spec["left"])))
    right = dict(_load_daily_series_candidates(paths, str(spec["right"])))
    common_dates = sorted(set(left).intersection(right))
    return [(observed_date, left[observed_date] - right[observed_date]) for observed_date in common_dates]


def _compute_event_delta(
    *,
    spec: dict[str, Any],
    start_value: float,
    end_value: float,
) -> float | None:
    transform = str(spec.get("transform", "difference")).strip()
    scale = float(spec.get("scale", 1.0))
    if transform == "difference":
        return (end_value - start_value) * scale
    if transform == "pct_change":
        if start_value == 0:
            return None
        return ((end_value / start_value) - 1.0) * scale
    raise ValueError(f"Unsupported event transform: {transform}")


def _event_scaling_rule(outcome_ids: list[str]) -> str:
    transforms = {str(EVENT_OUTCOME_SPECS[item].get("transform", "difference")) for item in outcome_ids if item in EVENT_OUTCOME_SPECS}
    if transforms == {"difference"}:
        return "rate_and_level_changes_in_catalog_units"
    return "mixed_catalog_units_with_pct_returns"


def _control_release_plus_column(control_id: str, horizon: int) -> str:
    if control_id == "dff":
        return f"delta_dff_{_release_plus_label(horizon)}"
    return f"delta_{control_id}_{_release_plus_label(horizon)}"


def _control_end_date_column(control_id: str, horizon: int) -> str:
    if control_id == "dff":
        return f"end_date_dff_{_release_plus_label(horizon)}"
    return f"end_date_{control_id}_{_release_plus_label(horizon)}"


def _control_placebo_delta_column(control_id: str, start_bd: int, end_bd: int) -> str:
    if control_id == "dff":
        return f"delta_dff_{_release_minus_label(start_bd, end_bd)}"
    return f"delta_{control_id}_{_release_minus_label(start_bd, end_bd)}"


def _control_placebo_start_date_column(control_id: str, start_bd: int, end_bd: int) -> str:
    if control_id == "dff":
        return f"placebo_start_date_dff_{_release_minus_label(start_bd, end_bd)}"
    return f"placebo_start_date_{control_id}_{_release_minus_label(start_bd, end_bd)}"


def _event_is_usable(row: dict[str, str]) -> bool:
    treatment_value = str(row.get("treatment_value", "")).strip()
    usable_for_headline = str(row.get("usable_for_headline", "")).strip().lower() == "true"
    return bool(treatment_value) and usable_for_headline


def _event_is_reviewed_nonmissing(row: dict[str, str]) -> bool:
    treatment_value = str(row.get("treatment_value", "")).strip()
    reviewed = str(row.get("shock_review_status", "")).strip().lower() == "reviewed"
    shock_nonmissing = str(row.get("shock_missing_flag", "")).strip().lower() == "false"
    return bool(treatment_value) and reviewed and shock_nonmissing


def _event_sample_bucket(row: dict[str, str]) -> str:
    if _event_is_usable(row):
        return "headline_strict"
    if _event_is_reviewed_nonmissing(row):
        if str(row.get("small_denominator_flag", "")).strip().lower() == "true":
            return "reviewed_zero_shock_small_denom"
        return "reviewed_nonheadline"
    return "out_of_sample"


def _event_in_requested_sample(row: dict[str, str], sample_policy: str) -> bool:
    if sample_policy == "headline_strict":
        return _event_is_usable(row)
    if sample_policy == "reviewed_nonmissing":
        return _event_is_reviewed_nonmissing(row)
    raise ValueError(f"Unsupported event sample_policy: {sample_policy}")


def _resolve_event_control_ids(job: dict[str, Any], outcome_ids: list[str]) -> tuple[list[str], str]:
    explicit_controls = [str(item).strip() for item in job.get("controls_explicit", []) if str(item).strip()]
    if explicit_controls:
        invalid_controls = [item for item in explicit_controls if item not in EVENT_CONTROL_SPECS]
        if invalid_controls:
            raise ValueError(f"Unsupported explicit event controls: {', '.join(invalid_controls)}")
        return [item for item in explicit_controls if item not in outcome_ids], "explicit"
    return [item for item in EVENT_CONTROL_CANDIDATES if item not in outcome_ids], DEFAULT_EVENT_CONTROL_SELECTION_POLICY


def _release_plus_label(horizon: int) -> str:
    return f"release_plus_{int(horizon)}bd"


def _release_minus_label(start_bd: int, end_bd: int) -> str:
    return f"release_minus_{abs(int(start_bd))}bd_to_minus_{abs(int(end_bd))}bd"


def _load_debt_limit_intervals(config_dir: Path) -> list[tuple[date, date | None]]:
    path = config_dir / "debt_limit_intervals.csv"
    if not path.exists():
        return []
    intervals: list[tuple[date, date | None]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            start_date = _parse_date(str(row.get("start_date", "")))
            end_date = _parse_date(str(row.get("end_date", "")))
            if start_date is None:
                continue
            intervals.append((start_date, end_date))
    return intervals


def _date_in_intervals(value: date, intervals: list[tuple[date, date | None]]) -> bool:
    for start_date, end_date in intervals:
        if value < start_date:
            continue
        if end_date is None or value <= end_date:
            return True
    return False


def build_event_design(paths: ProjectPaths, *, job_id: str) -> EventDesignBuildResult:
    jobs = _load_jobs(paths.config / "dass_job_blueprint.yaml")
    if job_id not in jobs:
        raise KeyError(f"Unknown event job_id: {job_id}")
    job = jobs[job_id]
    if str(job.get("estimator", "")).strip() != "event_lp":
        raise ValueError(f"Job '{job_id}' is not an event_lp design")

    event_bundle_path = paths.bundles / "qrawatch" / "event_bundle.csv"
    if not event_bundle_path.exists():
        raise FileNotFoundError(f"Missing qrawatch event bundle: {event_bundle_path}")
    event_rows = _read_csv(event_bundle_path)

    outcome_ids = [str(item) for item in job.get("outcomes", [])]
    horizons = [int(item) for item in job.get("horizons_bd", [])]
    debt_limit_intervals = _load_debt_limit_intervals(paths.config)
    sample_policy = str(job.get("sample_policy", DEFAULT_EVENT_SAMPLE_POLICY)).strip() or DEFAULT_EVENT_SAMPLE_POLICY
    selected_control_ids, control_selection_policy = _resolve_event_control_ids(job, outcome_ids)

    missing_series: list[str] = []
    outcome_levels: dict[str, list[tuple[date, float]]] = {}
    for outcome_id in outcome_ids:
        if outcome_id not in EVENT_OUTCOME_SPECS:
            missing_series.append(outcome_id)
            continue
        levels = _build_outcome_levels(paths, outcome_id)
        if not levels:
            missing_series.append(outcome_id)
            continue
        outcome_levels[outcome_id] = levels

    control_levels: dict[str, list[tuple[date, float]]] = {}
    for control_id in selected_control_ids:
        spec = EVENT_CONTROL_SPECS[control_id]
        if str(spec.get("kind", "")).strip() == "direct":
            levels = _load_daily_series(paths.raw_fred, str(spec.get("series", "")))
        else:
            levels = _build_outcome_levels(paths, control_id)
        if levels:
            control_levels[control_id] = levels

    bundle_rows: list[dict[str, str]] = []
    usable_rows = 0
    for row in sorted(event_rows, key=lambda item: (str(item.get("event_date", "")), str(item.get("event_id", "")))):
        event_date = _parse_date(str(row.get("event_date", "")))
        if event_date is None:
            continue
        start_target = previous_us_market_business_day(event_date)
        bundle_row = {
            "event_id": str(row.get("event_id", "")).strip(),
            "quarter": str(row.get("quarter", "")).strip(),
            "event_label": str(row.get("event_label", "")).strip(),
            "event_date": event_date.isoformat(),
            "cutoff_timestamp": str(row.get("cutoff_timestamp", "")).strip(),
            "treatment_id": str(job.get("treatment_id", "")).strip(),
            "source_treatment_id": str(row.get("treatment_id", "")).strip(),
            "treatment_value": str(row.get("treatment_value", "")).strip(),
            "treatment_units": str(row.get("treatment_units", "")).strip() or "usd_billions",
            "usable_for_headline": str(row.get("usable_for_headline", "")).strip(),
            "usable_for_headline_reason": str(row.get("usable_for_headline_reason", "")).strip(),
            "usable_for_descriptive_headline": str(row.get("usable_for_descriptive_headline", "")).strip(),
            "descriptive_headline_reason": str(row.get("descriptive_headline_reason", "")).strip(),
            "quality_tier": str(row.get("quality_tier", "")).strip(),
            "claim_scope": str(row.get("claim_scope", "")).strip(),
            "shock_review_status": str(row.get("shock_review_status", "")).strip(),
            "shock_missing_flag": str(row.get("shock_missing_flag", "")).strip(),
            "small_denominator_flag": str(row.get("small_denominator_flag", "")).strip(),
            "debt_limit_dummy": "1" if _date_in_intervals(event_date, debt_limit_intervals) else "0",
            "window_start_target": start_target.isoformat(),
            "window_start_date": "",
        }
        bundle_row["sample_policy"] = sample_policy
        bundle_row["sample_bucket"] = _event_sample_bucket(row)
        bundle_row["include_in_sample"] = "true" if _event_in_requested_sample(row, sample_policy) else "false"

        row_complete = _event_in_requested_sample(row, sample_policy) and not missing_series
        generic_window_start_set = False
        for outcome_id in outcome_ids:
            levels = outcome_levels.get(outcome_id, [])
            start_observation = _first_available_on_or_after(levels, start_target) if levels else None
            start_date_column = f"start_date_{outcome_id}"
            if start_observation is not None:
                bundle_row[start_date_column] = start_observation[0].isoformat()
                if not generic_window_start_set:
                    bundle_row["window_start_date"] = start_observation[0].isoformat()
                    generic_window_start_set = True
            else:
                bundle_row[start_date_column] = ""
            for horizon in horizons:
                delta_column = f"delta_{outcome_id}_h{horizon}bd"
                end_date_column = f"end_date_{outcome_id}_h{horizon}bd"
                if start_observation is None:
                    bundle_row[delta_column] = ""
                    bundle_row[end_date_column] = ""
                    row_complete = False
                else:
                    spec = EVENT_OUTCOME_SPECS[outcome_id]
                    end_target = add_us_market_business_days(event_date, horizon)
                    end_observation = _first_available_on_or_after(levels, end_target)
                    if end_observation is None or end_observation[0] <= start_observation[0]:
                        bundle_row[delta_column] = ""
                        bundle_row[end_date_column] = ""
                        row_complete = False
                    else:
                        delta_value = _compute_event_delta(
                            spec=spec,
                            start_value=start_observation[1],
                            end_value=end_observation[1],
                        )
                        if delta_value is None:
                            bundle_row[delta_column] = ""
                            bundle_row[end_date_column] = ""
                            row_complete = False
                        else:
                            bundle_row[delta_column] = _stable_float_text(delta_value)
                            bundle_row[end_date_column] = end_observation[0].isoformat()
            for start_bd, end_bd in PLACEBO_WINDOWS_BD:
                placebo_delta_column = f"delta_{outcome_id}_{_release_minus_label(start_bd, end_bd)}"
                placebo_start_date_column = (
                    f"placebo_start_date_{outcome_id}_{_release_minus_label(start_bd, end_bd)}"
                )
                placebo_target = add_us_market_business_days(event_date, start_bd)
                placebo_start_observation = _first_available_on_or_after(levels, placebo_target) if levels else None
                if placebo_start_observation is None or start_observation is None:
                    bundle_row[placebo_delta_column] = ""
                    bundle_row[placebo_start_date_column] = ""
                elif placebo_start_observation[0] >= start_observation[0]:
                    bundle_row[placebo_delta_column] = ""
                    bundle_row[placebo_start_date_column] = ""
                else:
                    bundle_row[placebo_start_date_column] = placebo_start_observation[0].isoformat()
                    delta_value = _compute_event_delta(
                        spec=EVENT_OUTCOME_SPECS[outcome_id],
                        start_value=placebo_start_observation[1],
                        end_value=start_observation[1],
                    )
                    bundle_row[placebo_delta_column] = _stable_float_text(delta_value) if delta_value is not None else ""
        for control_id, levels in control_levels.items():
            start_observation = _first_available_on_or_after(levels, start_target)
            bundle_row[f"start_date_{control_id}"] = start_observation[0].isoformat() if start_observation is not None else ""
            for horizon in horizons:
                control_column = _control_release_plus_column(control_id, horizon)
                control_end_date_column = _control_end_date_column(control_id, horizon)
                if start_observation is None:
                    bundle_row[control_column] = ""
                    bundle_row[control_end_date_column] = ""
                    row_complete = False
                    continue
                end_target = add_us_market_business_days(event_date, horizon)
                end_observation = _first_available_on_or_after(levels, end_target)
                if end_observation is None or end_observation[0] <= start_observation[0]:
                    bundle_row[control_column] = ""
                    bundle_row[control_end_date_column] = ""
                    row_complete = False
                    continue
                delta_value = _compute_event_delta(
                    spec=EVENT_CONTROL_SPECS[control_id],
                    start_value=start_observation[1],
                    end_value=end_observation[1],
                )
                bundle_row[control_column] = _stable_float_text(delta_value) if delta_value is not None else ""
                bundle_row[control_end_date_column] = end_observation[0].isoformat() if delta_value is not None else ""
                if delta_value is None:
                    row_complete = False
            for start_bd, end_bd in PLACEBO_WINDOWS_BD:
                placebo_delta_column = _control_placebo_delta_column(control_id, start_bd, end_bd)
                placebo_start_date_column = _control_placebo_start_date_column(control_id, start_bd, end_bd)
                placebo_target = add_us_market_business_days(event_date, start_bd)
                placebo_start_observation = _first_available_on_or_after(levels, placebo_target)
                if placebo_start_observation is None or start_observation is None:
                    bundle_row[placebo_delta_column] = ""
                    bundle_row[placebo_start_date_column] = ""
                    continue
                if placebo_start_observation[0] >= start_observation[0]:
                    bundle_row[placebo_delta_column] = ""
                    bundle_row[placebo_start_date_column] = ""
                    continue
                delta_value = _compute_event_delta(
                    spec=EVENT_CONTROL_SPECS[control_id],
                    start_value=placebo_start_observation[1],
                    end_value=start_observation[1],
                )
                bundle_row[placebo_delta_column] = _stable_float_text(delta_value) if delta_value is not None else ""
                bundle_row[placebo_start_date_column] = placebo_start_observation[0].isoformat()
        if row_complete:
            usable_rows += 1
        bundle_rows.append(bundle_row)

    bundle_dir = paths.bundles / "designs"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = bundle_dir / f"{job_id}__event_panel.csv"
    with bundle_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in bundle_rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bundle_rows)

    all_rows = len(bundle_rows)
    rows_with_treatment = sum(1 for row in bundle_rows if str(row.get("treatment_value", "")).strip())
    headline_eligible_rows = sum(1 for row in bundle_rows if _event_is_usable(row))
    reviewed_nonmissing_rows = sum(1 for row in bundle_rows if _event_is_reviewed_nonmissing(row))
    reviewed_zero_shock_rows = sum(
        1
        for row in bundle_rows
        if _event_is_reviewed_nonmissing(row)
        and str(row.get("small_denominator_flag", "")).strip().lower() == "true"
    )
    requested_sample_rows = sum(1 for row in bundle_rows if str(row.get("include_in_sample", "")).strip().lower() == "true")
    observed_dates = [_parse_date(str(row.get("event_date", ""))) for row in bundle_rows]
    event_dates = [value for value in observed_dates if value is not None]

    design_manifest = {
        "job_id": job_id,
        "sample_start": min(event_dates).isoformat() if event_dates else "",
        "sample_end": max(event_dates).isoformat() if event_dates else "",
        "treatment_id": str(job.get("treatment_id", "")).strip(),
        "instrument_ids": [],
        "outcome_ids": outcome_ids,
        "control_ids": [
            "debt_limit_dummy",
            *[
                _control_release_plus_column(control_id, horizon)
                for control_id in control_levels
                for horizon in horizons
            ],
        ],
        "state_ids": [],
        "cutoff_rule": str(job.get("cutoff_rule", "")).strip(),
        "horizon_grid": horizons,
        "exclusion_windows": [_release_minus_label(start_bd, end_bd) for start_bd, end_bd in PLACEBO_WINDOWS_BD],
        "scaling_rule": _event_scaling_rule(outcome_ids),
        "shock_definition": "qrawatch_release_shock_with_pre_release_marker_window",
        "multiple_testing_family": str(job.get("output_family", "")).strip(),
        "generated_at": utc_now_iso(),
        "bundle_path": str(bundle_path),
        "calendar": "us_market_holiday_business_day",
        "sample_policy": sample_policy,
        "control_selection_policy": control_selection_policy,
        "event_sample_counts": {
            "all_events": all_rows,
            "rows_with_treatment": rows_with_treatment,
            "headline_eligible_rows": headline_eligible_rows,
            "reviewed_nonmissing_rows": reviewed_nonmissing_rows,
            "reviewed_zero_shock_rows": reviewed_zero_shock_rows,
            "requested_sample_rows": requested_sample_rows,
            "usable_rows": usable_rows,
        },
        "missing_required_series": missing_series,
        "status": "ready_for_estimation" if usable_rows > 0 and not missing_series else "partial_ready",
        "usable_rows": usable_rows,
    }
    design_manifest_path = paths.manifests / f"{job_id}__design_manifest.json"
    write_json(design_manifest_path, design_manifest)

    sample_manifest = {
        "generated_at": utc_now_iso(),
        "job_id": job_id,
        "sample_policy": sample_policy,
        "counts": {
            "all_events": all_rows,
            "rows_with_treatment": rows_with_treatment,
            "headline_eligible_rows": headline_eligible_rows,
            "reviewed_nonmissing_rows": reviewed_nonmissing_rows,
            "reviewed_zero_shock_rows": reviewed_zero_shock_rows,
            "requested_sample_rows": requested_sample_rows,
            "usable_rows": usable_rows,
        },
        "rows": [
            {
                "job_id": job_id,
                "step_order": 1,
                "step_label": "all_events_seen",
                "observations_remaining": all_rows,
                "reason": "normalized qrawatch event bundle coverage",
            },
            {
                "job_id": job_id,
                "step_order": 2,
                "step_label": "has_treatment_value",
                "observations_remaining": rows_with_treatment,
                "reason": "nonmissing event-level treatment values present",
            },
            {
                "job_id": job_id,
                "step_order": 3,
                "step_label": "requested_estimation_sample",
                "observations_remaining": requested_sample_rows,
                "reason": (
                    "reviewed qrawatch headline-eligible releases with treatment values"
                    if sample_policy == "headline_strict"
                    else "reviewed qrawatch releases with nonmissing shocks, including small-denominator zero-shock events"
                ),
            },
            {
                "job_id": job_id,
                "step_order": 4,
                "step_label": "required_outcomes_present",
                "observations_remaining": usable_rows,
                "reason": (
                    "all requested event outcomes present across configured business-day horizons"
                    if not missing_series
                    else "currently limited by downloaded or derived event outcome series"
                ),
            },
        ],
    }
    sample_manifest_path = paths.manifests / f"{job_id}__sample_manifest.json"
    write_json(sample_manifest_path, sample_manifest)

    return EventDesignBuildResult(
        bundle_path=bundle_path,
        design_manifest_path=design_manifest_path,
        sample_manifest_path=sample_manifest_path,
        rows_written=len(bundle_rows),
        usable_rows=usable_rows,
    )
