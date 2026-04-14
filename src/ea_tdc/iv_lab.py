from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from ea_tdc.designs.quarterly import INSTRUMENT_COMPONENTS, _load_jobs, build_quarterly_design
from ea_tdc.estimation import _build_quarterly_target, _coerce_float, _first_stage_diagnostics
from ea_tdc.paths import ProjectPaths
from ea_tdc.utils import utc_now_iso, write_json


IV_SCAN_SHOCK_IDS = [
    "qra_maturity_tilt_flow",
    "qra_ati_baseline_bn",
    "qra_net_bills_bn",
    "qra_bill_share",
]

IV_SCAN_STATE_IDS = [
    "tsyparty_ru_gap_l1",
    "tsyparty_bank_absorption_share_l1",
    "tsyparty_row_absorption_share_l1",
    "tsyparty_bank_foreign_official_corr_l1",
    "tsyparty_bank_foreign_private_corr_l1",
    "tsyparty_bank_mmf_corr_l1",
    "tsyparty_private_minus_official_corr_l1",
    "wamest_bank_reserve_short_share_l1",
    "wamest_bank_reserve_wam_years_l1",
    "wamest_bank_broad_short_share_l1",
    "wamest_bank_broad_wam_years_l1",
    "wamest_foreigners_short_share_l1",
    "wamest_foreigners_wam_years_l1",
    "slrwatch_bank_leverage_pressure_l1",
    "slrwatch_bank_duration_pressure_l1",
    "slrwatch_bank_funding_pressure_l1",
    "slrwatch_bank_headroom_pp_l1",
    "slrwatch_bank_duration_loss_dominant_share_l1",
    "slrwatch_bank_leverage_dominant_share_l1",
    "coord_low_reserve_state_l1",
    "coord_on_rrp_drain_state_l1",
    "coord_liquidity_tightness_q_z_l1",
    "coord_on_rrp_share_q",
]


@dataclass(frozen=True)
class IVLabResult:
    summary_path: Path
    summary_csv_path: Path
    jobs_scanned: int
    total_candidates: int


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _stable_float_text(value: float | None) -> str:
    if value is None:
        return ""
    return str(round(float(value), 6))


def _unique_preserve_order(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        result.append(text)
        seen.add(text)
    return result


def _candidate_id(shock_id: str, state_id: str) -> str:
    return f"iv_{shock_id}_x_{state_id}"


def _configured_candidate_ids(configured_instruments: list[str]) -> set[str]:
    result: set[str] = set()
    for instrument_id in configured_instruments:
        component = INSTRUMENT_COMPONENTS.get(instrument_id, {})
        shock_id = str(component.get("shock", "")).strip()
        state_id = str(component.get("state", "")).strip()
        if shock_id and state_id:
            result.add(_candidate_id(shock_id, state_id))
    return result


def _base_control_ids(configured_instruments: list[str], configured_controls: list[str]) -> list[str]:
    component_ids: list[str] = []
    for instrument_id in configured_instruments:
        component = INSTRUMENT_COMPONENTS.get(instrument_id, {})
        for key in ("shock", "state"):
            value = str(component.get(key, "")).strip()
            if value:
                component_ids.append(value)
    return [control_id for control_id in configured_controls if control_id not in set(component_ids)]


def _instrument_available(bundle_rows: list[dict[str, str]], shock_id: str, state_id: str) -> bool:
    for row in bundle_rows:
        if _coerce_float(row.get(shock_id, "")) is None:
            continue
        if _coerce_float(row.get(state_id, "")) is None:
            continue
        return True
    return False


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(median(values))


def _build_candidate_summary(
    *,
    bundle_rows: list[dict[str, str]],
    job_id: str,
    treatment_id: str,
    outcome_ids: list[str],
    horizons: list[int],
    response_type: str,
    configured_controls: list[str],
    configured_instruments: list[str],
    shock_id: str,
    state_id: str,
) -> dict[str, Any] | None:
    if not _instrument_available(bundle_rows, shock_id, state_id):
        return None
    candidate_id = _candidate_id(shock_id, state_id)
    current_candidate_ids = _configured_candidate_ids(configured_instruments)
    control_ids = _unique_preserve_order([*_base_control_ids(configured_instruments, configured_controls), shock_id, state_id])
    row_results: list[dict[str, Any]] = []
    singular_rows = 0
    for outcome_id in outcome_ids:
        for horizon in horizons:
            y_values: list[float] = []
            x_rows: list[list[float]] = []
            z_rows: list[list[float]] = []
            for idx, row in enumerate(bundle_rows):
                treatment_value = _coerce_float(row.get(treatment_id, ""))
                shock_value = _coerce_float(row.get(shock_id, ""))
                state_value = _coerce_float(row.get(state_id, ""))
                if treatment_value is None or shock_value is None or state_value is None:
                    continue
                controls: list[float] = []
                controls_ok = True
                for control_id in control_ids:
                    control_value = _coerce_float(row.get(control_id, ""))
                    if control_value is None:
                        controls_ok = False
                        break
                    controls.append(control_value)
                if not controls_ok:
                    continue
                target_value = _build_quarterly_target(
                    bundle_rows,
                    start_idx=idx,
                    outcome_id=outcome_id,
                    horizon=horizon,
                    response_type=response_type,
                )
                if target_value is None:
                    continue
                instrument_value = shock_value * state_value
                y_values.append(target_value)
                x_rows.append([1.0, treatment_value, *controls])
                z_rows.append([1.0, instrument_value, *controls])
            min_rows = max(len(control_ids) + 3, 4)
            if len(y_values) < min_rows:
                continue
            try:
                first_stage = _first_stage_diagnostics(
                    [row[1] for row in x_rows],
                    x_rows,
                    z_rows,
                    instrument_count=1,
                )
            except ValueError:
                singular_rows += 1
                continue
            row_results.append(
                {
                    "outcome": outcome_id,
                    "horizon": horizon,
                    "n": len(y_values),
                    "first_stage_f_excluded": first_stage.excluded_instrument_f,
                    "first_stage_partial_r2": first_stage.partial_r2,
                    "first_stage_r2": first_stage.rsquared,
                    "weak_instrument_flag": first_stage.weak_instrument_flag,
                }
            )
    if not row_results:
        return None
    f_values = [float(item["first_stage_f_excluded"]) for item in row_results if item["first_stage_f_excluded"] is not None]
    partial_r2_values = [float(item["first_stage_partial_r2"]) for item in row_results if item["first_stage_partial_r2"] is not None]
    weak_rows = sum(1 for item in row_results if bool(item["weak_instrument_flag"]))
    summary = {
        "job_id": job_id,
        "candidate_id": candidate_id,
        "shock_id": shock_id,
        "state_id": state_id,
        "control_ids": control_ids,
        "is_current_instrument": candidate_id in current_candidate_ids,
        "rows_estimated": len(row_results),
        "singular_rows": singular_rows,
        "weak_rows": weak_rows,
        "weak_row_share": weak_rows / len(row_results),
        "min_observations": min(int(item["n"]) for item in row_results),
        "max_observations": max(int(item["n"]) for item in row_results),
        "min_first_stage_f": min(f_values) if f_values else None,
        "median_first_stage_f": _median_or_none(f_values),
        "max_first_stage_f": max(f_values) if f_values else None,
        "min_partial_r2": min(partial_r2_values) if partial_r2_values else None,
        "median_partial_r2": _median_or_none(partial_r2_values),
        "max_partial_r2": max(partial_r2_values) if partial_r2_values else None,
        "row_results": row_results,
    }
    return summary


def _score_candidate(summary: dict[str, Any]) -> tuple[float, float, float, int]:
    weak_row_share_value = summary.get("weak_row_share", 1.0)
    median_f_value = summary.get("median_first_stage_f", -1.0)
    min_f_value = summary.get("min_first_stage_f", -1.0)
    weak_row_share = 1.0 if weak_row_share_value is None else float(weak_row_share_value)
    median_f = -1.0 if median_f_value is None else float(median_f_value)
    min_f = -1.0 if min_f_value is None else float(min_f_value)
    rows_estimated = int(summary.get("rows_estimated", 0) or 0)
    return (-weak_row_share, median_f, min_f, rows_estimated)


def _recommendation(summary: dict[str, Any], current_summary: dict[str, Any] | None) -> str:
    weak_row_share_value = summary.get("weak_row_share", 1.0)
    min_f_value = summary.get("min_first_stage_f", -1.0)
    median_f_value = summary.get("median_first_stage_f", -1.0)
    weak_row_share = 1.0 if weak_row_share_value is None else float(weak_row_share_value)
    min_f = -1.0 if min_f_value is None else float(min_f_value)
    median_f = -1.0 if median_f_value is None else float(median_f_value)
    if bool(summary.get("is_current_instrument")):
        return "current_blocked" if weak_row_share > 0.0 else "current_viable"
    if weak_row_share == 0.0 and min_f >= 10.0:
        return "promising_upgrade"
    if current_summary is not None:
        current_weak_value = current_summary.get("weak_row_share", 1.0)
        current_median_value = current_summary.get("median_first_stage_f", -1.0)
        current_weak = 1.0 if current_weak_value is None else float(current_weak_value)
        current_median = -1.0 if current_median_value is None else float(current_median_value)
        if weak_row_share < current_weak or (weak_row_share == current_weak and median_f > current_median):
            return "better_than_current_but_still_weak"
    return "weak_or_insufficient"


def build_iv_lab(paths: ProjectPaths, *, job_id: str | None = None) -> IVLabResult:
    jobs = _load_jobs(paths.config / "dass_job_blueprint.yaml")
    job_ids = [job_id] if job_id else [current_job_id for current_job_id, job in jobs.items() if str(job.get("estimator", "")).strip() == "lp_iv"]
    summaries: list[dict[str, Any]] = []
    job_rows: list[dict[str, Any]] = []
    jobs_scanned = 0
    for current_job_id in job_ids:
        if current_job_id not in jobs:
            raise KeyError(f"Unknown job_id: {current_job_id}")
        job = jobs[current_job_id]
        if str(job.get("estimator", "")).strip() != "lp_iv":
            raise ValueError(f"Job '{current_job_id}' is not an lp_iv estimator")
        built = build_quarterly_design(paths, job_id=current_job_id)
        design_manifest_path = built.design_manifest_path
        design_manifest = json.loads(design_manifest_path.read_text(encoding="utf-8"))
        bundle_rows = _read_csv(built.bundle_path)
        treatment_id = str(design_manifest.get("treatment_id", "")).strip()
        outcome_ids = [str(item) for item in design_manifest.get("outcome_ids", [])]
        horizons = [int(item) for item in design_manifest.get("horizon_grid", [])]
        response_type = str(job.get("response_type", "direct_at_h")).strip()
        configured_controls = [str(item) for item in design_manifest.get("control_ids", [])]
        configured_instruments = [str(item) for item in job.get("instruments", [])]
        candidate_summaries: list[dict[str, Any]] = []
        for shock_id in IV_SCAN_SHOCK_IDS:
            for state_id in IV_SCAN_STATE_IDS:
                summary = _build_candidate_summary(
                    bundle_rows=bundle_rows,
                    job_id=current_job_id,
                    treatment_id=treatment_id,
                    outcome_ids=outcome_ids,
                    horizons=horizons,
                    response_type=response_type,
                    configured_controls=configured_controls,
                    configured_instruments=configured_instruments,
                    shock_id=shock_id,
                    state_id=state_id,
                )
                if summary is not None:
                    candidate_summaries.append(summary)
        candidate_summaries.sort(key=_score_candidate, reverse=True)
        current_summary = next((item for item in candidate_summaries if bool(item.get("is_current_instrument"))), None)
        for rank, summary in enumerate(candidate_summaries, start=1):
            summary["rank"] = rank
            summary["recommendation"] = _recommendation(summary, current_summary)
            summary["beats_current_candidate"] = bool(
                current_summary
                and summary["candidate_id"] != current_summary["candidate_id"]
                and _score_candidate(summary) > _score_candidate(current_summary)
            )
            job_rows.append(
                {
                    "job_id": current_job_id,
                    "rank": rank,
                    "candidate_id": summary["candidate_id"],
                    "shock_id": summary["shock_id"],
                    "state_id": summary["state_id"],
                    "is_current_instrument": bool(summary["is_current_instrument"]),
                    "rows_estimated": int(summary["rows_estimated"]),
                    "singular_rows": int(summary["singular_rows"]),
                    "weak_rows": int(summary["weak_rows"]),
                    "weak_row_share": _stable_float_text(float(summary["weak_row_share"])),
                    "min_observations": int(summary["min_observations"]),
                    "max_observations": int(summary["max_observations"]),
                    "min_first_stage_f": _stable_float_text(summary["min_first_stage_f"]),
                    "median_first_stage_f": _stable_float_text(summary["median_first_stage_f"]),
                    "max_first_stage_f": _stable_float_text(summary["max_first_stage_f"]),
                    "min_partial_r2": _stable_float_text(summary["min_partial_r2"]),
                    "median_partial_r2": _stable_float_text(summary["median_partial_r2"]),
                    "max_partial_r2": _stable_float_text(summary["max_partial_r2"]),
                    "recommendation": str(summary["recommendation"]),
                    "beats_current_candidate": bool(summary["beats_current_candidate"]),
                    "control_ids": ",".join(str(item) for item in summary["control_ids"]),
                }
            )
        summaries.append(
            {
                "job_id": current_job_id,
                "current_instruments": configured_instruments,
                "treatment_id": treatment_id,
                "outcome_ids": outcome_ids,
                "horizons": horizons,
                "response_type": response_type,
                "candidate_rows": candidate_summaries,
                "top_candidates": candidate_summaries[:5],
                "current_candidate": current_summary,
            }
        )
        jobs_scanned += 1

    summary_payload = {
        "generated_at": utc_now_iso(),
        "jobs_scanned": jobs_scanned,
        "total_candidates": len(job_rows),
        "jobs": summaries,
    }
    summary_path = paths.reports / "iv_lab.json"
    summary_csv_path = paths.reports / "iv_lab.csv"
    write_json(summary_path, summary_payload)
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(job_rows[0].keys()) if job_rows else [
            "job_id",
            "rank",
            "candidate_id",
            "shock_id",
            "state_id",
            "is_current_instrument",
            "rows_estimated",
            "singular_rows",
            "weak_rows",
            "weak_row_share",
            "min_observations",
            "max_observations",
            "min_first_stage_f",
            "median_first_stage_f",
            "max_first_stage_f",
            "min_partial_r2",
            "median_partial_r2",
            "max_partial_r2",
            "recommendation",
            "beats_current_candidate",
            "control_ids",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(job_rows)
    return IVLabResult(
        summary_path=summary_path,
        summary_csv_path=summary_csv_path,
        jobs_scanned=jobs_scanned,
        total_candidates=len(job_rows),
    )
