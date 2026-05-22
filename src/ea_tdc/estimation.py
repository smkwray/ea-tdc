from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from math import erf, sqrt
from pathlib import Path
from statistics import fmean
from typing import Any

from ea_tdc.designs.quarterly import _load_jobs, _quarter_sort_key
from ea_tdc.paths import ProjectPaths
from ea_tdc.utils import utc_now_iso, write_json


@dataclass(frozen=True)
class JobEstimationResult:
    estimates_path: Path
    summary_path: Path
    comparison_path: Path | None
    rows_written: int
    estimator: str


@dataclass(frozen=True)
class EstimationSnapshotResult:
    summary_path: Path
    summary_csv_path: Path
    jobs_estimated: int


@dataclass(frozen=True)
class RegressionFit:
    beta: list[float]
    ses: list[float]
    covariance: list[list[float]]
    fitted: list[float]
    residuals: list[float]
    rsquared: float | None
    covariance_estimator: str
    covariance_lags: int


@dataclass(frozen=True)
class FirstStageDiagnostics:
    rsquared: float | None
    excluded_instrument_f: float | None
    partial_r2: float | None
    weak_instrument_flag: bool


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _coerce_float(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(column) for column in zip(*matrix)]


def _matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [sum(left[row_idx][k] * right[k][col_idx] for k in range(len(right))) for col_idx in range(len(right[0]))]
        for row_idx in range(len(left))
    ]


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(row[col_idx] * vector[col_idx] for col_idx in range(len(vector))) for row in matrix]


def _identity(size: int) -> list[list[float]]:
    return [[1.0 if row_idx == col_idx else 0.0 for col_idx in range(size)] for row_idx in range(size)]


def _zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def _outer(left: list[float], right: list[float]) -> list[list[float]]:
    return [[left[row_idx] * right[col_idx] for col_idx in range(len(right))] for row_idx in range(len(left))]


def _matrix_add_in_place(target: list[list[float]], addition: list[list[float]], *, scale: float = 1.0) -> None:
    for row_idx in range(len(target)):
        for col_idx in range(len(target[row_idx])):
            target[row_idx][col_idx] += addition[row_idx][col_idx] * scale


def _invert(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    augmented = [row[:] + ident_row[:] for row, ident_row in zip(matrix, _identity(size))]
    for pivot_idx in range(size):
        pivot_row = max(range(pivot_idx, size), key=lambda idx: abs(augmented[idx][pivot_idx]))
        pivot_value = augmented[pivot_row][pivot_idx]
        if abs(pivot_value) < 1e-12:
            raise ValueError("Singular matrix")
        if pivot_row != pivot_idx:
            augmented[pivot_idx], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_idx]
        pivot_value = augmented[pivot_idx][pivot_idx]
        augmented[pivot_idx] = [value / pivot_value for value in augmented[pivot_idx]]
        for row_idx in range(size):
            if row_idx == pivot_idx:
                continue
            factor = augmented[row_idx][pivot_idx]
            augmented[row_idx] = [
                current - factor * pivot
                for current, pivot in zip(augmented[row_idx], augmented[pivot_idx])
            ]
    return [row[size:] for row in augmented]


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _compute_rsquared(y_values: list[float], fitted: list[float]) -> float | None:
    mean_value = fmean(y_values)
    total_ss = sum((value - mean_value) ** 2 for value in y_values)
    if total_ss <= 0:
        return None
    residual_ss = sum((actual - estimate) ** 2 for actual, estimate in zip(y_values, fitted))
    return max(0.0, 1.0 - (residual_ss / total_ss))


def _ols(
    y_values: list[float],
    x_rows: list[list[float]],
    *,
    covariance_estimator: str = "classical",
    covariance_lags: int = 0,
) -> RegressionFit:
    if not x_rows:
        raise ValueError("No estimation rows")
    xt = _transpose(x_rows)
    xtx = _matmul(xt, x_rows)
    xtx_inv = _invert(xtx)
    xty = _matvec(xt, y_values)
    beta = _matvec(xtx_inv, xty)
    fitted = [sum(coeff * value for coeff, value in zip(beta, row)) for row in x_rows]
    residuals = [actual - fit for actual, fit in zip(y_values, fitted)]
    n_obs = len(y_values)
    n_params = len(beta)
    dof = max(n_obs - n_params, 1)
    rss = sum(residual ** 2 for residual in residuals)

    if covariance_estimator == "classical":
        sigma2 = rss / dof
        covariance = [[sigma2 * xtx_inv[row_idx][col_idx] for col_idx in range(n_params)] for row_idx in range(n_params)]
    else:
        scores = [[residual * value for value in row] for residual, row in zip(residuals, x_rows)]
        meat = _zeros(n_params, n_params)
        for score in scores:
            _matrix_add_in_place(meat, _outer(score, score))
        if covariance_estimator == "newey_west":
            lag_count = min(max(covariance_lags, 0), n_obs - 1)
            for lag in range(1, lag_count + 1):
                weight = 1.0 - (lag / (lag_count + 1))
                for idx in range(lag, n_obs):
                    _matrix_add_in_place(meat, _outer(scores[idx], scores[idx - lag]), scale=weight)
                    _matrix_add_in_place(meat, _outer(scores[idx - lag], scores[idx]), scale=weight)
        elif covariance_estimator != "hc1":
            raise ValueError(f"Unsupported covariance estimator: {covariance_estimator}")
        if n_obs > n_params:
            _matrix_add_in_place(meat, meat, scale=((n_obs / (n_obs - n_params)) - 1.0))
        covariance = _matmul(_matmul(xtx_inv, meat), xtx_inv)

    ses = [covariance[idx][idx] ** 0.5 if covariance[idx][idx] >= 0 else 0.0 for idx in range(n_params)]
    return RegressionFit(
        beta=beta,
        ses=ses,
        covariance=covariance,
        fitted=fitted,
        residuals=residuals,
        rsquared=_compute_rsquared(y_values, fitted),
        covariance_estimator=covariance_estimator,
        covariance_lags=max(covariance_lags, 0) if covariance_estimator == "newey_west" else 0,
    )


def _first_stage_diagnostics(
    treatment_values: list[float],
    x_rows: list[list[float]],
    z_rows: list[list[float]],
    *,
    instrument_count: int,
) -> FirstStageDiagnostics:
    unrestricted = _ols(treatment_values, z_rows, covariance_estimator="classical")
    restricted_rows = [[row[0], *row[2:]] for row in x_rows]
    restricted = _ols(treatment_values, restricted_rows, covariance_estimator="classical")
    residual_ss_u = sum(value ** 2 for value in unrestricted.residuals)
    residual_ss_r = sum(value ** 2 for value in restricted.residuals)
    n_obs = len(treatment_values)
    n_params_unrestricted = len(z_rows[0]) if z_rows else 0
    denominator_df = n_obs - n_params_unrestricted
    excluded_f = None
    partial_r2 = None
    if instrument_count > 0 and denominator_df > 0 and residual_ss_r > 0 and residual_ss_r >= residual_ss_u:
        explained_ss = residual_ss_r - residual_ss_u
        if explained_ss >= 0:
            partial_r2 = max(0.0, explained_ss / residual_ss_r)
            denominator = residual_ss_u / denominator_df if residual_ss_u > 0 else 0.0
            if denominator > 0:
                excluded_f = (explained_ss / instrument_count) / denominator
    return FirstStageDiagnostics(
        rsquared=unrestricted.rsquared,
        excluded_instrument_f=excluded_f,
        partial_r2=partial_r2,
        weak_instrument_flag=excluded_f is not None and excluded_f < 10.0,
    )


def _two_stage_least_squares(
    y_values: list[float],
    x_rows: list[list[float]],
    z_rows: list[list[float]],
    *,
    instrument_count: int,
) -> tuple[RegressionFit, FirstStageDiagnostics]:
    if not x_rows or not z_rows:
        raise ValueError("No estimation rows")
    zt = _transpose(z_rows)
    ztz = _matmul(zt, z_rows)
    ztz_inv = _invert(ztz)
    projection = _matmul(_matmul(z_rows, ztz_inv), zt)
    xhat_rows = _matmul(projection, x_rows)
    x_pz_x = _matmul(_transpose(x_rows), xhat_rows)
    x_pz_y = _matvec(_transpose(xhat_rows), y_values)
    x_pz_x_inv = _invert(x_pz_x)
    beta = _matvec(x_pz_x_inv, x_pz_y)
    fitted = [sum(coeff * value for coeff, value in zip(beta, row)) for row in x_rows]
    residuals = [actual - fit for actual, fit in zip(y_values, fitted)]
    n_obs = len(y_values)
    n_params = len(beta)
    meat = _zeros(n_params, n_params)
    for residual, xhat_row in zip(residuals, xhat_rows):
        score = [residual * value for value in xhat_row]
        _matrix_add_in_place(meat, _outer(score, score))
    if n_obs > n_params:
        _matrix_add_in_place(meat, meat, scale=((n_obs / (n_obs - n_params)) - 1.0))
    covariance = _matmul(_matmul(x_pz_x_inv, meat), x_pz_x_inv)
    ses = [covariance[idx][idx] ** 0.5 if covariance[idx][idx] >= 0 else 0.0 for idx in range(n_params)]
    fit = RegressionFit(
        beta=beta,
        ses=ses,
        covariance=covariance,
        fitted=fitted,
        residuals=residuals,
        rsquared=_compute_rsquared(y_values, fitted),
        covariance_estimator="hc1",
        covariance_lags=0,
    )
    treatment_values = [row[1] for row in x_rows]
    diagnostics = _first_stage_diagnostics(
        treatment_values,
        x_rows,
        z_rows,
        instrument_count=instrument_count,
    )
    return fit, diagnostics


def _warning_flags_text(flags: list[str]) -> str:
    return ";".join(flag for flag in flags if flag)


def _estimate_row_payload(
    *,
    job_id: str,
    outcome_id: str,
    horizon: int,
    treatment_id: str,
    control_ids_used: list[str],
    response_type: str,
    inference_method: str,
    fit: RegressionFit,
    beta_index: int = 1,
    beta_override: float | None = None,
    se_override: float | None = None,
    instrument_ids: list[str] | None = None,
    first_stage: FirstStageDiagnostics | None = None,
    warning_flags: list[str] | None = None,
    dropped_control_ids: list[str] | None = None,
    state_id: str | None = None,
    state_profile: str | None = None,
    state_reference_value: float | None = None,
    state_interaction_beta: float | None = None,
    state_interaction_se: float | None = None,
) -> dict[str, str]:
    beta = fit.beta[beta_index] if beta_override is None else beta_override
    se = fit.ses[beta_index] if se_override is None else se_override
    z_score = None if se <= 0 else beta / se
    p_value = None if z_score is None else max(0.0, min(1.0, 2.0 * (1.0 - _normal_cdf(abs(z_score)))))
    payload = {
        "job_id": job_id,
        "outcome": outcome_id,
        "horizon": str(horizon),
        "beta": str(beta),
        "se": str(se),
        "lower95": str(beta - 1.96 * se),
        "upper95": str(beta + 1.96 * se),
        "z_score": "" if z_score is None else str(z_score),
        "p_value_normal": "" if p_value is None else str(p_value),
        "n": str(len(fit.fitted)),
        "treatment_id": treatment_id,
        "control_ids_used": ",".join(control_ids_used),
        "response_type": response_type,
        "inference_method": inference_method,
        "covariance_estimator": fit.covariance_estimator,
        "covariance_lags": str(fit.covariance_lags),
        "rsquared": "" if fit.rsquared is None else str(round(fit.rsquared, 6)),
        "warning_flags": _warning_flags_text(warning_flags or []),
        "dropped_control_ids": ",".join(dropped_control_ids or []),
    }
    if state_id is not None:
        payload["state_id"] = state_id
        payload["state_profile"] = state_profile or ""
        payload["state_reference_value"] = "" if state_reference_value is None else str(state_reference_value)
        payload["state_interaction_beta"] = "" if state_interaction_beta is None else str(state_interaction_beta)
        payload["state_interaction_se"] = "" if state_interaction_se is None else str(state_interaction_se)
    if instrument_ids is not None:
        payload["instrument_ids"] = ",".join(instrument_ids)
    if first_stage is not None:
        payload["first_stage_r2"] = "" if first_stage.rsquared is None else str(round(first_stage.rsquared, 6))
        payload["first_stage_partial_r2"] = (
            "" if first_stage.partial_r2 is None else str(round(first_stage.partial_r2, 6))
        )
        payload["first_stage_f_excluded"] = (
            "" if first_stage.excluded_instrument_f is None else str(round(first_stage.excluded_instrument_f, 6))
        )
        payload["weak_instrument_flag"] = "true" if first_stage.weak_instrument_flag else "false"
    return payload


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("No values supplied")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, quantile)) * (len(ordered) - 1)
    lower_idx = int(position)
    upper_idx = min(lower_idx + 1, len(ordered) - 1)
    weight = position - lower_idx
    return ordered[lower_idx] * (1.0 - weight) + ordered[upper_idx] * weight


def _state_reference_profiles(state_values: list[float]) -> list[tuple[str, float]]:
    unique_values = sorted({round(value, 10) for value in state_values})
    if unique_values and all(value in {0.0, 1.0} for value in unique_values):
        profiles: list[tuple[str, float]] = []
        if 0.0 in unique_values:
            profiles.append(("low_state", 0.0))
        if 1.0 in unique_values:
            profiles.append(("high_state", 1.0))
        return profiles
    low_value = _quantile(state_values, 0.25)
    high_value = _quantile(state_values, 0.75)
    if abs(high_value - low_value) < 1e-9:
        low_value = min(state_values)
        high_value = max(state_values)
    if abs(high_value - low_value) < 1e-9:
        return [("state_reference", low_value)]
    return [("low_state", low_value), ("high_state", high_value)]


def _build_quarterly_target(
    rows: list[dict[str, str]],
    *,
    start_idx: int,
    outcome_id: str,
    horizon: int,
    response_type: str,
) -> float | None:
    values: list[float] = []
    if response_type == "cumulative_sum_h0_to_h":
        end_idx = start_idx + horizon
        if end_idx >= len(rows):
            return None
        indices = range(start_idx, end_idx + 1)
    else:
        target_idx = start_idx + horizon
        if target_idx >= len(rows):
            return None
        indices = [target_idx]
    for idx in indices:
        value = _coerce_float(rows[idx].get(outcome_id, ""))
        if value is None:
            return None
        values.append(value)
    return sum(values)


def _reference_outcome_name(outcome_id: str, prefix: str) -> str:
    if prefix and outcome_id.startswith(prefix):
        return outcome_id[len(prefix) :]
    return outcome_id


def _estimate_rows_lp(
    *,
    bundle_rows: list[dict[str, str]],
    treatment_id: str,
    control_ids: list[str],
    outcome_ids: list[str],
    horizons: list[int],
    response_type: str,
    job_id: str,
) -> list[dict[str, str]]:
    result_rows: list[dict[str, str]] = []
    for outcome_id in outcome_ids:
        for horizon in horizons:
            candidate_controls = control_ids[:]
            best_fit: tuple[list[float], list[list[float]], list[str], list[str]] | None = None
            fit_warning_flags: list[str] = []
            while True:
                y_values: list[float] = []
                x_rows: list[list[float]] = []
                for idx, row in enumerate(bundle_rows):
                    treatment_value = _coerce_float(row.get(treatment_id, ""))
                    if treatment_value is None:
                        continue
                    controls: list[float] = []
                    controls_ok = True
                    for control_id in candidate_controls:
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
                    y_values.append(target_value)
                    x_rows.append([1.0, treatment_value, *controls])
                if len(y_values) > len(candidate_controls) + 2:
                    try:
                        _ols(
                            y_values,
                            x_rows,
                            covariance_estimator="newey_west",
                            covariance_lags=max(horizon, 1),
                        )
                    except ValueError:
                        if not candidate_controls:
                            break
                        candidate_controls = candidate_controls[:-1]
                        if "collinear_controls" not in fit_warning_flags:
                            fit_warning_flags.append("collinear_controls")
                        continue
                    dropped_controls = [control_id for control_id in control_ids if control_id not in candidate_controls]
                    best_fit = (y_values, x_rows, candidate_controls[:], dropped_controls)
                    break
                if not candidate_controls:
                    break
                candidate_controls = candidate_controls[:-1]

            if best_fit is None:
                continue

            y_values, x_rows, controls_used, dropped_controls = best_fit
            fit = _ols(
                y_values,
                x_rows,
                covariance_estimator="newey_west",
                covariance_lags=max(horizon, 1),
            )
            warning_flags = fit_warning_flags[:]
            if dropped_controls:
                warning_flags.append("adaptive_controls")
            result_rows.append(
                _estimate_row_payload(
                    job_id=job_id,
                    outcome_id=outcome_id,
                    horizon=horizon,
                    treatment_id=treatment_id,
                    control_ids_used=controls_used,
                    response_type=response_type,
                    inference_method=(
                        "ols_newey_west_scaffold"
                        if not dropped_controls
                        else "ols_newey_west_scaffold_adaptive_controls"
                    ),
                    fit=fit,
                    warning_flags=warning_flags,
                    dropped_control_ids=dropped_controls,
                )
            )
    return result_rows


def _estimate_rows_lp_state(
    *,
    bundle_rows: list[dict[str, str]],
    treatment_id: str,
    state_id: str,
    control_ids: list[str],
    outcome_ids: list[str],
    horizons: list[int],
    response_type: str,
    job_id: str,
) -> list[dict[str, str]]:
    result_rows: list[dict[str, str]] = []
    for outcome_id in outcome_ids:
        for horizon in horizons:
            eligible_rows: list[tuple[float, float, list[float], float]] = []
            for idx, row in enumerate(bundle_rows):
                treatment_value = _coerce_float(row.get(treatment_id, ""))
                state_value = _coerce_float(row.get(state_id, ""))
                if treatment_value is None or state_value is None:
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
                eligible_rows.append((treatment_value, state_value, controls, target_value))
            min_rows = len(control_ids) + 5
            if len(eligible_rows) < min_rows:
                continue
            state_values = [item[1] for item in eligible_rows]
            if max(state_values) - min(state_values) < 1e-9:
                continue
            state_mean = fmean(state_values)
            y_values: list[float] = []
            x_rows: list[list[float]] = []
            for treatment_value, state_value, controls, target_value in eligible_rows:
                centered_state = state_value - state_mean
                y_values.append(target_value)
                x_rows.append([1.0, treatment_value, centered_state, treatment_value * centered_state, *controls])
            try:
                fit = _ols(
                    y_values,
                    x_rows,
                    covariance_estimator="newey_west",
                    covariance_lags=max(horizon, 1),
                )
            except ValueError:
                continue
            profiles = _state_reference_profiles(state_values)
            interaction_beta = fit.beta[3]
            interaction_se = fit.ses[3]
            covariance = fit.covariance
            for profile_label, reference_value in profiles:
                centered_reference = reference_value - state_mean
                marginal_beta = fit.beta[1] + interaction_beta * centered_reference
                marginal_variance = (
                    covariance[1][1]
                    + (centered_reference ** 2) * covariance[3][3]
                    + (2.0 * centered_reference * covariance[1][3])
                )
                marginal_se = marginal_variance ** 0.5 if marginal_variance >= 0 else 0.0
                result_rows.append(
                    _estimate_row_payload(
                        job_id=job_id,
                        outcome_id=outcome_id,
                        horizon=horizon,
                        treatment_id=treatment_id,
                        control_ids_used=control_ids,
                        response_type=response_type,
                        inference_method="ols_newey_west_state_interaction_scaffold",
                        fit=fit,
                        beta_override=marginal_beta,
                        se_override=marginal_se,
                        state_id=state_id,
                        state_profile=profile_label,
                        state_reference_value=reference_value,
                        state_interaction_beta=interaction_beta,
                        state_interaction_se=interaction_se,
                    )
                )
    return result_rows


def _estimate_rows_lp_iv(
    *,
    bundle_rows: list[dict[str, str]],
    treatment_id: str,
    control_ids: list[str],
    outcome_ids: list[str],
    horizons: list[int],
    response_type: str,
    job_id: str,
    instrument_ids: list[str],
) -> list[dict[str, str]]:
    result_rows: list[dict[str, str]] = []
    for outcome_id in outcome_ids:
        for horizon in horizons:
            y_values: list[float] = []
            x_rows: list[list[float]] = []
            z_rows: list[list[float]] = []
            for idx, row in enumerate(bundle_rows):
                treatment_value = _coerce_float(row.get(treatment_id, ""))
                if treatment_value is None:
                    continue
                instruments: list[float] = []
                instruments_ok = True
                for instrument_id in instrument_ids:
                    instrument_value = _coerce_float(row.get(instrument_id, ""))
                    if instrument_value is None:
                        instruments_ok = False
                        break
                    instruments.append(instrument_value)
                if not instruments_ok:
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
                y_values.append(target_value)
                x_rows.append([1.0, treatment_value, *controls])
                z_rows.append([1.0, *instruments, *controls])
            min_rows = max(len(control_ids) + len(instrument_ids) + 2, len(control_ids) + 3)
            if len(y_values) < min_rows:
                continue
            fit, first_stage = _two_stage_least_squares(
                y_values,
                x_rows,
                z_rows,
                instrument_count=len(instrument_ids),
            )
            warning_flags = ["weak_instrument"] if first_stage.weak_instrument_flag else []
            result_rows.append(
                _estimate_row_payload(
                    job_id=job_id,
                    outcome_id=outcome_id,
                    horizon=horizon,
                    treatment_id=treatment_id,
                    instrument_ids=instrument_ids,
                    control_ids_used=control_ids,
                    response_type=response_type,
                    inference_method="two_stage_least_squares_hc1_scaffold",
                    fit=fit,
                    first_stage=first_stage,
                    warning_flags=warning_flags,
                )
            )
    return result_rows


def _resolve_event_treatment_column(bundle_rows: list[dict[str, str]], treatment_id: str) -> str:
    if any(_coerce_float(row.get(treatment_id, "")) is not None for row in bundle_rows):
        return treatment_id
    return "treatment_value"


def _event_row_in_sample(row: dict[str, str]) -> bool:
    include_in_sample = str(row.get("include_in_sample", "")).strip().lower()
    if include_in_sample in {"true", "false"}:
        return include_in_sample == "true"
    return str(row.get("usable_for_headline", "")).strip().lower() == "true"


def _event_controls_for_horizon(control_ids: list[str], horizon: int) -> list[str]:
    suffix = f"_release_plus_{int(horizon)}bd"
    matched: list[str] = []
    for control_id in control_ids:
        if "_release_plus_" not in control_id:
            matched.append(control_id)
            continue
        if control_id.endswith(suffix):
            matched.append(control_id)
    return matched


def _estimate_rows_event_lp(
    *,
    bundle_rows: list[dict[str, str]],
    treatment_id: str,
    control_ids: list[str],
    outcome_ids: list[str],
    horizons: list[int],
    job_id: str,
) -> list[dict[str, str]]:
    result_rows: list[dict[str, str]] = []
    event_treatment_column = _resolve_event_treatment_column(bundle_rows, treatment_id)
    usable_rows = [row for row in bundle_rows if _event_row_in_sample(row)]

    for outcome_id in outcome_ids:
        for horizon in horizons:
            outcome_column = f"delta_{outcome_id}_h{horizon}bd"
            eligible_controls = _event_controls_for_horizon(control_ids, horizon)
            candidate_controls = eligible_controls[:]
            best_fit: tuple[list[float], list[list[float]], list[str]] | None = None
            fit_warning_flags: list[str] = []

            while True:
                y_values: list[float] = []
                x_rows: list[list[float]] = []
                for row in usable_rows:
                    treatment_value = _coerce_float(row.get(event_treatment_column, ""))
                    outcome_value = _coerce_float(row.get(outcome_column, ""))
                    if treatment_value is None or outcome_value is None:
                        continue
                    controls: list[float] = []
                    controls_ok = True
                    for control_id in candidate_controls:
                        control_value = _coerce_float(row.get(control_id, ""))
                        if control_value is None:
                            controls_ok = False
                            break
                        controls.append(control_value)
                    if not controls_ok:
                        continue
                    y_values.append(outcome_value)
                    x_rows.append([1.0, treatment_value, *controls])

                if len(y_values) > len(candidate_controls) + 2:
                    try:
                        _ols(y_values, x_rows, covariance_estimator="hc1")
                    except ValueError:
                        if not candidate_controls:
                            break
                        candidate_controls = candidate_controls[:-1]
                        if "collinear_controls" not in fit_warning_flags:
                            fit_warning_flags.append("collinear_controls")
                        continue
                    best_fit = (y_values, x_rows, candidate_controls[:])
                    break
                if not candidate_controls:
                    break
                candidate_controls = candidate_controls[:-1]

            if best_fit is None:
                continue

            y_values, x_rows, controls_used = best_fit
            fit = _ols(y_values, x_rows, covariance_estimator="hc1")
            warning_flags: list[str] = fit_warning_flags[:]
            dropped_controls = [control_id for control_id in eligible_controls if control_id not in controls_used]
            if dropped_controls:
                warning_flags.append("adaptive_controls")
            if len(y_values) < 8:
                warning_flags.append("small_sample_event")
            result_rows.append(
                _estimate_row_payload(
                    job_id=job_id,
                    outcome_id=outcome_id,
                    horizon=horizon,
                    treatment_id=event_treatment_column,
                    control_ids_used=controls_used,
                    response_type="direct_event_delta_horizon",
                    inference_method=(
                        "event_ols_hc1_scaffold"
                        if len(controls_used) == len(eligible_controls)
                        else "event_ols_hc1_scaffold_adaptive_controls"
                    ),
                    fit=fit,
                    warning_flags=warning_flags,
                    dropped_control_ids=dropped_controls,
                )
            )
    return result_rows


def _estimate_rows(
    *,
    estimator: str,
    bundle_rows: list[dict[str, str]],
    treatment_id: str,
    control_ids: list[str],
    outcome_ids: list[str],
    horizons: list[int],
    response_type: str,
    job_id: str,
    instrument_ids: list[str],
    state_id: str,
) -> list[dict[str, str]]:
    if estimator == "lp":
        if state_id:
            return _estimate_rows_lp_state(
                bundle_rows=bundle_rows,
                treatment_id=treatment_id,
                state_id=state_id,
                control_ids=control_ids,
                outcome_ids=outcome_ids,
                horizons=horizons,
                response_type=response_type,
                job_id=job_id,
            )
        return _estimate_rows_lp(
            bundle_rows=bundle_rows,
            treatment_id=treatment_id,
            control_ids=control_ids,
            outcome_ids=outcome_ids,
            horizons=horizons,
            response_type=response_type,
            job_id=job_id,
        )
    if estimator == "lp_iv":
        return _estimate_rows_lp_iv(
            bundle_rows=bundle_rows,
            treatment_id=treatment_id,
            control_ids=control_ids,
            outcome_ids=outcome_ids,
            horizons=horizons,
            response_type=response_type,
            job_id=job_id,
            instrument_ids=instrument_ids,
        )
    if estimator == "event_lp":
        return _estimate_rows_event_lp(
            bundle_rows=bundle_rows,
            treatment_id=treatment_id,
            control_ids=control_ids,
            outcome_ids=outcome_ids,
            horizons=horizons,
            job_id=job_id,
        )
    raise ValueError(f"Unsupported estimator: {estimator}")


def _estimates_filename(job_id: str, estimator: str) -> str:
    if estimator == "event_lp":
        return f"{job_id}__event_lp_estimates.csv"
    if estimator == "lp_iv":
        return f"{job_id}__lp_iv_estimates.csv"
    return f"{job_id}__lp_estimates.csv"


def _write_estimates_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else [
            "job_id",
            "outcome",
            "horizon",
            "beta",
            "se",
            "lower95",
            "upper95",
            "z_score",
            "p_value_normal",
            "n",
            "treatment_id",
            "control_ids_used",
            "response_type",
            "inference_method",
            "covariance_estimator",
            "covariance_lags",
            "rsquared",
            "warning_flags",
            "dropped_control_ids",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _maybe_write_reference_comparison(
    *,
    paths: ProjectPaths,
    job: dict[str, Any],
    job_id: str,
    result_rows: list[dict[str, str]],
) -> Path | None:
    published_reference = str(job.get("published_reference_artifact", "")).strip()
    if not published_reference:
        return None
    reference_path = (paths.root / published_reference).resolve()
    if not reference_path.exists():
        return None
    reference_rows = _read_csv(reference_path)
    prefix = str(job.get("published_reference_outcome_prefix", "")).strip()
    reference_lookup = {
        (str(row.get("outcome", "")).strip(), str(row.get("horizon", "")).strip()): row
        for row in reference_rows
    }
    matched: list[dict[str, Any]] = []
    beta_gaps: list[float] = []
    for row in result_rows:
        key = (_reference_outcome_name(str(row["outcome"]), prefix), str(row["horizon"]))
        reference = reference_lookup.get(key)
        if reference is None:
            continue
        reference_beta = _coerce_float(str(reference.get("beta", "")))
        estimated_beta = _coerce_float(str(row.get("beta", "")))
        if reference_beta is None or estimated_beta is None:
            continue
        gap = estimated_beta - reference_beta
        beta_gaps.append(abs(gap))
        matched.append(
            {
                "outcome": row["outcome"],
                "horizon": row["horizon"],
                "estimated_beta": estimated_beta,
                "reference_beta": reference_beta,
                "beta_gap": gap,
            }
        )
    comparison_payload = {
        "job_id": job_id,
        "generated_at": utc_now_iso(),
        "reference_path": str(reference_path),
        "matched_rows": len(matched),
        "mean_abs_beta_gap": (sum(beta_gaps) / len(beta_gaps)) if beta_gaps else None,
        "max_abs_beta_gap": max(beta_gaps) if beta_gaps else None,
        "rows": matched,
    }
    comparison_path = paths.reports / f"{job_id}__reference_comparison.json"
    write_json(comparison_path, comparison_payload)
    return comparison_path


def estimate_job(paths: ProjectPaths, *, job_id: str) -> JobEstimationResult:
    jobs = _load_jobs(paths.config / "dass_job_blueprint.yaml")
    if job_id not in jobs:
        raise KeyError(f"Unknown job_id: {job_id}")
    job = jobs[job_id]
    estimator = str(job.get("estimator", "")).strip()
    if estimator not in {"lp", "lp_iv", "event_lp"}:
        raise ValueError(f"Job '{job_id}' does not have a supported estimation scaffold")

    design_manifest_path = paths.manifests / f"{job_id}__design_manifest.json"
    if not design_manifest_path.exists():
        raise FileNotFoundError(f"Missing design manifest for job '{job_id}'")
    design_manifest = json.loads(design_manifest_path.read_text(encoding="utf-8"))
    if str(design_manifest.get("status", "")).strip() != "ready_for_estimation":
        raise ValueError(f"Design manifest for job '{job_id}' is not ready_for_estimation")

    bundle_path = Path(str(design_manifest.get("bundle_path", "")))
    bundle_rows = _read_csv(bundle_path)
    if estimator == "event_lp":
        bundle_rows.sort(key=lambda row: str(row.get("event_date", "")))
        horizons = [int(item) for item in design_manifest.get("horizon_grid", [])]
        response_type = "direct_event_delta_horizon"
    else:
        bundle_rows.sort(key=lambda row: _quarter_sort_key(str(row.get("quarter", ""))))
        horizons = [int(item) for item in design_manifest.get("horizon_grid", [])]
        response_type = str(job.get("response_type", "direct_at_h")).strip()

    treatment_id = str(design_manifest.get("treatment_id", "")).strip()
    control_ids = [str(item) for item in design_manifest.get("control_ids", [])]
    outcome_ids = [str(item) for item in design_manifest.get("outcome_ids", [])]
    instrument_ids = [str(item) for item in design_manifest.get("instrument_ids", [])]
    state_id = str(job.get("state_id", "")).strip()

    result_rows = _estimate_rows(
        estimator=estimator,
        bundle_rows=bundle_rows,
        treatment_id=treatment_id,
        control_ids=control_ids,
        outcome_ids=outcome_ids,
        horizons=horizons,
        response_type=response_type,
        job_id=job_id,
        instrument_ids=instrument_ids,
        state_id=state_id,
    )

    models_dir = paths.output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    estimates_path = models_dir / _estimates_filename(job_id, estimator)
    _write_estimates_csv(estimates_path, result_rows)

    comparison_path = _maybe_write_reference_comparison(
        paths=paths,
        job=job,
        job_id=job_id,
        result_rows=result_rows,
    )

    warning_rows = sum(1 for row in result_rows if str(row.get("warning_flags", "")).strip())
    weak_instrument_rows = sum(1 for row in result_rows if str(row.get("weak_instrument_flag", "")).strip() == "true")
    adaptive_control_rows = sum(
        1 for row in result_rows if "adaptive_controls" in str(row.get("warning_flags", "")).split(";")
    )
    small_sample_rows = sum(
        1 for row in result_rows if "small_sample_event" in str(row.get("warning_flags", "")).split(";")
    )
    covariance_estimators_used = sorted(
        {str(row.get("covariance_estimator", "")).strip() for row in result_rows if str(row.get("covariance_estimator", "")).strip()}
    )
    ns = [_coerce_float(str(row.get("n", ""))) for row in result_rows]
    n_values = [int(value) for value in ns if value is not None]

    summary = {
        "job_id": job_id,
        "estimator": estimator,
        "generated_at": utc_now_iso(),
        "estimates_path": str(estimates_path),
        "comparison_path": str(comparison_path) if comparison_path else "",
        "rows_written": len(result_rows),
        "treatment_id": treatment_id,
        "instrument_ids": instrument_ids,
        "state_id": state_id,
        "control_ids": control_ids,
        "outcome_ids": outcome_ids,
        "horizon_grid": horizons,
        "response_type": response_type,
        "covariance_estimators_used": covariance_estimators_used,
        "warning_rows": warning_rows,
        "weak_instrument_rows": weak_instrument_rows,
        "adaptive_control_rows": adaptive_control_rows,
        "small_sample_rows": small_sample_rows,
        "min_observations": min(n_values) if n_values else 0,
        "max_observations": max(n_values) if n_values else 0,
        "sample_policy": str(design_manifest.get("sample_policy", "")).strip(),
        "event_sample_counts": design_manifest.get("event_sample_counts", {}),
    }
    summary_path = paths.manifests / f"{job_id}__estimation_summary.json"
    write_json(summary_path, summary)

    return JobEstimationResult(
        estimates_path=estimates_path,
        summary_path=summary_path,
        comparison_path=comparison_path,
        rows_written=len(result_rows),
        estimator=estimator,
    )


def estimate_quarterly_job(paths: ProjectPaths, *, job_id: str) -> JobEstimationResult:
    jobs = _load_jobs(paths.config / "dass_job_blueprint.yaml")
    if job_id not in jobs:
        raise KeyError(f"Unknown job_id: {job_id}")
    if str(jobs[job_id].get("estimator", "")).strip() != "lp":
        raise ValueError(f"Job '{job_id}' is not an lp estimator")
    return estimate_job(paths, job_id=job_id)


def build_estimation_snapshot(paths: ProjectPaths) -> EstimationSnapshotResult:
    jobs = _load_jobs(paths.config / "dass_job_blueprint.yaml")
    rows: list[dict[str, str]] = []
    for job_id, job in jobs.items():
        estimator = str(job.get("estimator", "")).strip()
        if estimator not in {"lp", "lp_iv", "event_lp"}:
            continue
        if not bool(job.get("track_in_estimation_snapshot", True)):
            continue
        design_manifest_path = paths.manifests / f"{job_id}__design_manifest.json"
        if not design_manifest_path.exists():
            continue
        design_manifest = json.loads(design_manifest_path.read_text(encoding="utf-8"))
        if str(design_manifest.get("status", "")).strip() != "ready_for_estimation":
            continue
        estimated = estimate_job(paths, job_id=job_id)
        summary = json.loads(estimated.summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "job_id": job_id,
                "estimator": estimator,
                "rows_written": str(summary.get("rows_written", 0)),
                "estimates_path": str(summary.get("estimates_path", "")),
                "comparison_path": str(summary.get("comparison_path", "")),
                "sample_policy": str(summary.get("sample_policy", "")),
                "requested_sample_rows": str((summary.get("event_sample_counts", {}) or {}).get("requested_sample_rows", "")),
                "headline_eligible_rows": str((summary.get("event_sample_counts", {}) or {}).get("headline_eligible_rows", "")),
            }
        )

    summary_path = paths.reports / "estimation_snapshot.json"
    summary_csv_path = paths.reports / "estimation_snapshot.csv"
    write_json(
        summary_path,
        {
            "generated_at": utc_now_iso(),
            "jobs_estimated": len(rows),
            "rows": rows,
        },
    )
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else [
            "job_id",
            "estimator",
            "rows_written",
            "estimates_path",
            "comparison_path",
            "sample_policy",
            "requested_sample_rows",
            "headline_eligible_rows",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return EstimationSnapshotResult(
        summary_path=summary_path,
        summary_csv_path=summary_csv_path,
        jobs_estimated=len(rows),
    )
