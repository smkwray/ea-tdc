"""Frozen quarterly residual-covariance and calendar-deletion diagnostics.

No control selection, factor training, source retrieval, or leg-slope fitting
occurs here. A failed leg-design preflight leaves coefficients unestimated.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from ea_tdc.covariance import COVARIANCE_OPERATOR_POLICY, canonical_covariance
from ea_tdc.estimation import RegressionFit, _invert, _matmul, _ols, _transpose
from ea_tdc.open01 import _fit_projection, _quarter_ordinal, _residualize
from ea_tdc.open_contract import (
    CANONICAL_CONTROL_IDS,
    CANONICAL_OUTCOME_ID,
    CANONICAL_TREATMENT_ID,
)

LEGS = ("R", "J", "O")
FROZEN_QUARTERS = tuple(f"{2002 + i // 4}Q{i % 4 + 1}" for i in range(96))
ORIGINAL_ROLLING_ENDPOINTS = FROZEN_QUARTERS[39:]
DELETED_QUARTERS = tuple(f"2020Q{i}" for i in range(1, 5)) + ("2021Q1",)
CHECKPOINTS = ("2020Q1", "2020Q4", "2022Q3", "2025Q4")
DISCLOSURES = {
    "perimeter": "The outcome is US-chartered institutions' total currency and deposits to all holders; the treatment bank perimeter also includes foreign banking offices and affiliated-area banks. The measured residual is not exact same-perimeter N.",
    "controls": "The dflmx_k100 controls were selected and scored on the full panel including 2020-21. Quarterly features labeled lag001 anchor at t-2. Inference is conditional on these frozen generated controls, not selection-adjusted.",
    "claim": "Descriptive conditional associations; not funding shares, causal mechanisms, landing, or retention.",
}


@dataclass(frozen=True)
class CalendarHACFit(RegressionFit):
    """Canonical covariance plus the unmodified calendar sandwich and its checks."""

    raw_covariance: list[list[float]]
    covariance_diagnostics: dict[str, Any]


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing or malformed {label}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite {label}")
    return number


def numerical_tolerance(*values: float) -> float:
    """Fixed absolute floor plus scale-aware numerical arithmetic allowance."""
    return 1e-9 + 1e-10 * max((abs(v) for v in values), default=0.0)


def validate_panel(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if tuple(str(row.get("quarter", "")) for row in rows) != FROZEN_QUARTERS:
        raise ValueError("Frozen panel must have exactly 96 ordered unique 2002Q1-2025Q4 quarters")
    required = (CANONICAL_TREATMENT_ID, CANONICAL_OUTCOME_ID, *CANONICAL_CONTROL_IDS)
    return [{"quarter": row["quarter"], **{key: _finite(row.get(key), key) for key in required}} for row in rows]


def join_legs(rows: Sequence[Mapping[str, Any]], legs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    panel = validate_panel(rows)
    if tuple(str(row.get("quarter", "")) for row in legs) != FROZEN_QUARTERS:
        raise ValueError("Treatment legs must exactly match frozen ordered quarters")
    for row, source in zip(panel, legs, strict=True):
        row.update({key: _finite(source.get(key), key) for key in ("C", *LEGS, "row_tsy_tx")})
        c = row[CANONICAL_TREATMENT_ID]
        if abs(row["C"] - c) > numerical_tolerance(row["C"], c):
            raise ValueError(f"Treatment differs from frozen panel at {row['quarter']}")
        total = sum(row[key] for key in LEGS)
        if abs(row["C"] - total) > numerical_tolerance(row["C"], total):
            raise ValueError(f"C=R+J+O fails at {row['quarter']}")
    return panel


def _scaled_rank(x: np.ndarray) -> int:
    scales = np.linalg.norm(x, axis=0)
    return int(np.linalg.matrix_rank(x / np.where(scales > 0, scales, 1.0)))


def calendar_hac(y: Sequence[float], x: Sequence[Sequence[float]], ordinals: Sequence[int], *, lags: int = 1) -> CalendarHACFit:
    """Observed-row OLS with Bartlett score products at calendar lag distances."""
    y_array, x_array = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    if x_array.ndim != 2 or len(y_array) != len(x_array) or len(ordinals) != len(y_array):
        raise ValueError("Calendar HAC inputs are not aligned")
    if not np.isfinite(x_array).all() or not np.isfinite(y_array).all():
        raise ValueError("Calendar HAC requires complete finite observed rows")
    if any(not isinstance(q, (int, np.integer)) for q in ordinals) or any(b <= a for a, b in pairwise(ordinals)):
        raise ValueError("Quarter ordinals must be unique and strictly increasing")
    rank = _scaled_rank(x_array)
    if rank != x_array.shape[1] or len(y_array) <= rank:
        raise ValueError("Calendar HAC requires a full-rank observed design with residual degrees of freedom")
    if lags < 0:
        raise ValueError("HAC lags cannot be negative")
    # Preserve OPEN-01 coefficient/residual arithmetic exactly.
    fit = _ols(y_array.tolist(), x_array.tolist(), covariance_estimator="classical")
    scores = x_array * np.asarray(fit.residuals)[:, None]
    meat = scores.T @ scores
    score_by_quarter = dict(zip(ordinals, scores, strict=True))
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        for q, score in score_by_quarter.items():
            previous = score_by_quarter.get(q - lag)
            if previous is not None:
                cross = np.outer(score, previous)
                meat += weight * (cross + cross.T)
    bread = np.asarray(_invert(_matmul(_transpose(x_array.tolist()), x_array.tolist())))
    raw_covariance = bread @ (meat * (len(y_array) / (len(y_array) - rank))) @ bread
    scales = np.linalg.norm(x_array, axis=0)
    outcome_scale = max(float(np.linalg.norm(y_array)), 1.0)
    operator = canonical_covariance(raw_covariance, scales, outcome_scale)
    covariance = operator["covariance"]
    diagnostics = operator["diagnostics"] | {
        "scales": scales.tolist(), "outcome_scale": outcome_scale,
        "policy": dict(COVARIANCE_OPERATOR_POLICY),
    }
    return CalendarHACFit(**(vars(fit) | {
        "covariance": covariance, "ses": np.sqrt(np.diag(covariance)).tolist(),
        "covariance_estimator": "calendar_newey_west", "covariance_lags": lags,
        "raw_covariance": operator["raw_covariance"], "covariance_diagnostics": diagnostics,
    }))


def pandemic_path(rows: Sequence[Mapping[str, Any]], *, nominal_endpoints: Sequence[str] | None = None) -> list[dict[str, Any]]:
    panel = validate_panel(rows)
    result = []
    # Preserve the archived endpoint inventory, including partially observed
    # initial windows. Nominal boundaries precede both availability and deletion.
    endpoints = tuple(nominal_endpoints) if nominal_endpoints is not None else ORIGINAL_ROLLING_ENDPOINTS
    if tuple(sorted(set(endpoints), key=_quarter_ordinal)) != endpoints or any(q not in FROZEN_QUARTERS for q in endpoints):
        raise ValueError("Nominal endpoints must be unique ordered frozen quarters")
    for endpoint in endpoints:
        start = _quarter_ordinal(endpoint) - 47
        nominal_start = f"{start // 4}Q{start % 4 + 1}"
        window = [r for r in panel if nominal_start <= r["quarter"] <= endpoint]
        observed = [r for r in window if r["quarter"] not in DELETED_QUARTERS]
        unchanged = _fit_projection(observed, treatment_id=CANONICAL_TREATMENT_ID, outcome_id=CANONICAL_OUTCOME_ID, control_ids=CANONICAL_CONTROL_IDS)
        x = [[1.0, r[CANONICAL_TREATMENT_ID], *[r[c] for c in unchanged.control_ids_used]] for r in observed]
        fit = calendar_hac([r[CANONICAL_OUTCOME_ID] for r in observed], x, [_quarter_ordinal(r["quarter"]) for r in observed])
        beta, se = fit.beta[1], fit.ses[1]
        result.append({
            "nominal_start": nominal_start, "nominal_end": endpoint,
            "first_observed": observed[0]["quarter"], "last_observed": observed[-1]["quarter"],
            "nominal_n": 48, "n_obs": len(observed), "rank": len(x[0]),
            "controls_used": ",".join(unchanged.control_ids_used),
            "controls_rejected": ",".join(unchanged.control_ids_rejected),
            "beta": beta, "se": se, "lower95": beta - 1.96 * se, "upper95": beta + 1.96 * se,
            "prespecified_endpoint": window[-1]["quarter"] in CHECKPOINTS,
            "description": "pandemic-row deletion conditional on frozen full-panel-generated controls",
            "covariance_lags": 1, "finite_sample_scale": len(observed) / (len(observed) - len(x[0])),
            "covariance_evidence": {"raw_covariance": fit.raw_covariance,
                                    "covariance": fit.covariance,
                                    "diagnostics": fit.covariance_diagnostics},
            **DISCLOSURES,
        })
    return result


def covariance_contributions(rows: Sequence[Mapping[str, Any]], frozen_beta: float) -> list[dict[str, Any]]:
    controls = [[row[c] for c in CANONICAL_CONTROL_IDS] for row in rows]
    c = np.asarray(_residualize([r["C"] for r in rows], controls))
    n = np.asarray(_residualize([r[CANONICAL_OUTCOME_ID] - r["C"] for r in rows], controls))
    denominator = float(c @ c)
    if denominator <= 0:
        raise ValueError("Treatment residual sum of squares is not positive")
    result = []
    for classification in ("corrected", "raw_purchases_sensitivity"):
        source = {leg: [r[leg] for r in rows] for leg in LEGS}
        if classification != "corrected":
            source["R"] = [r["row_tsy_tx"] for r in rows]
            source["J"] = [r["C"] - r["row_tsy_tx"] - r["O"] for r in rows]
        residual = {leg: np.asarray(_residualize(values, controls)) for leg, values in source.items()}
        residual["R+J"] = residual["R"] + residual["J"]
        nums = {leg: float(value @ n) for leg, value in residual.items()}
        total = sum(nums[leg] / denominator for leg in LEGS)
        target = frozen_beta - 1
        gap = total - target
        tolerance = numerical_tolerance(total, target)
        if abs(gap) > tolerance:
            raise ValueError(f"Covariance contribution identity fails: {gap} > {tolerance}")
        for leg, numerator in nums.items():
            result.append({"classification": classification, "leg": leg,
                "numerator": numerator, "common_denominator": denominator,
                "contribution": numerator / denominator, "contribution_sum": total,
                "frozen_beta_minus_one": target, "adding_up_gap": gap,
                "adding_up_tolerance": tolerance, "identity_status": "pass",
                "label": "leg covariance contributions to the measured residual", **DISCLOSURES})
    combined = [r["contribution"] for r in result if r["leg"] == "R+J"]
    if abs(combined[0] - combined[1]) > numerical_tolerance(*combined):
        raise ValueError("R+J classification invariance fails")
    return result


def leg_preflight(rows: Sequence[Mapping[str, Any]], resolution: Mapping[str, Any]) -> dict[str, Any]:
    """Design-only gate: deliberately does not compute unrestricted leg slopes."""
    controls = [[r[c] for c in CANONICAL_CONTROL_IDS] for r in rows]
    z = np.asarray([[1.0, *r] for r in controls])
    raw = np.asarray([[r[leg] for leg in LEGS] for r in rows])
    full = np.column_stack((z, raw))
    rank_z, rank_full = _scaled_rank(z), _scaled_rank(full)
    reasons = []
    if rank_z != z.shape[1]:
        reasons.append("control_matrix_not_full_rank")
    if rank_full != full.shape[1]:
        reasons.append("three_leg_design_not_full_rank")
    diagnostics = []
    condition, singular = None, []
    if rank_z == z.shape[1]:
        residual = np.column_stack([_residualize(raw[:, i].tolist(), controls) for i in range(3)])
        scales = np.std(residual, axis=0, ddof=1)
        if np.all(scales > 0):
            standardized = residual / scales
            s = np.linalg.svd(standardized, compute_uv=False)
            singular = s.tolist()
            if s[-1] > np.finfo(float).eps * s[0]:
                condition = float(s[0] / s[-1])
            if condition is None or condition > 30:
                reasons.append("condition_number_exceeds_30")
        else:
            reasons.append("zero_residual_variation")
        for i, leg in enumerate(LEGS):
            raw_sd, residual_sd = float(np.std(raw[:, i], ddof=1)), float(scales[i])
            others = np.delete(residual, i, axis=1)
            fitted = others @ np.linalg.lstsq(others, residual[:, i], rcond=None)[0]
            total = float(residual[:, i] @ residual[:, i])
            rss = float(np.sum((residual[:, i] - fitted) ** 2))
            vif = total / rss if rss > np.finfo(float).eps * total else None
            if vif is None or vif > 10:
                reasons.append(f"{leg}_vif_exceeds_10")
            record = resolution.get(leg, {})
            delta = record.get("delta_usd_million")
            supported = record.get("status") == "established" and bool(record.get("lineage"))
            if delta is not None:
                delta = _finite(delta, f"{leg} resolution")
            if not supported or delta is None or delta < 0:
                reasons.append(f"{leg}_source_resolution_unestablished")
                resolution_status = "failed_unestablished"
            elif residual_sd <= 10 * delta:
                reasons.append(f"{leg}_residual_sd_not_above_10_delta")
                resolution_status = "failed_below_resolution"
            else:
                resolution_status = "pass"
            diagnostics.append({"leg": leg, "raw_sd": raw_sd, "residual_sd": residual_sd,
                "residual_raw_sd_ratio": residual_sd / raw_sd if raw_sd else None,
                "distinct_value_count": len(set(raw[:, i])), "vif": vif,
                "delta_usd_million": delta, "source_resolution": dict(record),
                "resolution_gate": resolution_status})
    return {"status": "failed" if reasons else "passed", "reason_codes": reasons,
        "n_obs": len(rows), "controls_requested": list(CANONICAL_CONTROL_IDS),
        "controls_dropped": [], "control_rank": rank_z, "control_columns": z.shape[1],
        "three_leg_rank": rank_full, "three_leg_columns": full.shape[1],
        "standardized_singular_values": singular, "condition_number": condition,
        "condition_threshold": 30, "vif_threshold": 10, "resolution_multiplier": 10,
        "leg_diagnostics": diagnostics, "leg_estimates_computed": False,
        "inference_status": "withheld_preflight_failed" if reasons else "requires_gated_estimation",
        **DISCLOSURES}
