from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any

from ea_tdc.estimation import (
    _compute_rsquared,
    _invert,
    _matmul,
    _matvec,
    _ols,
    _transpose,
)


@dataclass(frozen=True)
class CrossfitNuisanceResult:
    y_hat: list[float]
    t_hat: list[float]
    y_r2: float | None
    t_r2: float | None


@dataclass(frozen=True)
class ContinuousTreatmentTMLEConfig:
    fold_count: int = 3
    ridge_alpha: float = 1.0
    h_clip: float = 4.0
    epsilon_scale_multiple: float = 2.0
    density_floor_quantile: float = 0.05
    min_density_floor: float = 1e-6
    max_low_density_share: float = 0.35
    max_epsilon_theta_ratio: float = 3.0
    min_outcome_r2: float = 0.0
    covariance_lags: int = 1


@dataclass(frozen=True)
class ContinuousTreatmentTMLEFit:
    fit: Any
    y_hat: list[float]
    t_hat: list[float]
    y_r2: float | None
    t_r2: float | None
    theta_init: float
    theta_init_std: float
    epsilon: float
    raw_epsilon: float
    epsilon_cap: float
    clever_clip_share: float
    low_density_share: float
    density_floor: float
    treatment_scale: float
    outcome_scale: float
    epsilon_theta_ratio: float
    valid: bool
    warning_flags: list[str]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def variance(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return sum((value - avg) ** 2 for value in values) / len(values)


def stddev(values: list[float]) -> float:
    sample_variance = variance(values)
    return sample_variance ** 0.5 if sample_variance > 0 else 0.0


def _ridge_fit(
    x_rows: list[list[float]],
    y_values: list[float],
    *,
    alpha: float,
) -> dict[str, Any]:
    if not x_rows:
        raise ValueError("No rows for ridge fit")
    feature_count = len(x_rows[0])
    means = [fmean(row[idx] for row in x_rows) for idx in range(feature_count)]
    stds: list[float] = []
    for idx in range(feature_count):
        feature_variance = sum((row[idx] - means[idx]) ** 2 for row in x_rows) / max(len(x_rows) - 1, 1)
        stds.append(feature_variance ** 0.5 if feature_variance > 1e-12 else 1.0)
    centered_x = [
        [(row[idx] - means[idx]) / stds[idx] for idx in range(feature_count)]
        for row in x_rows
    ]
    y_mean = fmean(y_values)
    centered_y = [value - y_mean for value in y_values]
    xt = _transpose(centered_x)
    xtx = _matmul(xt, centered_x)
    for idx in range(feature_count):
        xtx[idx][idx] += alpha
    beta = _matvec(_invert(xtx), _matvec(xt, centered_y))
    return {
        "beta": beta,
        "means": means,
        "stds": stds,
        "y_mean": y_mean,
    }


def _ridge_predict(model: dict[str, Any], x_rows: list[list[float]]) -> list[float]:
    beta = list(model["beta"])
    means = list(model["means"])
    stds = list(model["stds"])
    y_mean = float(model["y_mean"])
    predictions: list[float] = []
    for row in x_rows:
        centered = [(row[idx] - means[idx]) / stds[idx] for idx in range(len(beta))]
        predictions.append(y_mean + sum(centered[idx] * beta[idx] for idx in range(len(beta))))
    return predictions


def _contiguous_folds(count: int, fold_count: int) -> list[list[int]]:
    fold_count = max(2, min(fold_count, count))
    base = count // fold_count
    remainder = count % fold_count
    folds: list[list[int]] = []
    start = 0
    for fold_idx in range(fold_count):
        width = base + (1 if fold_idx < remainder else 0)
        end = start + width
        if end > start:
            folds.append(list(range(start, end)))
        start = end
    return folds


def crossfit_ridge_predictions(
    *,
    x_rows: list[list[float]],
    y_values: list[float],
    treatment_values: list[float],
    fold_count: int,
    ridge_alpha: float,
) -> CrossfitNuisanceResult:
    count = len(y_values)
    folds = _contiguous_folds(count, min(fold_count, max(count // 6, 2)))
    y_hat = [0.0 for _ in range(count)]
    t_hat = [0.0 for _ in range(count)]
    for fold in folds:
        fold_set = set(fold)
        train_idx = [idx for idx in range(count) if idx not in fold_set]
        if len(train_idx) <= len(x_rows[0]) + 1:
            raise ValueError("Insufficient rows for cross-fit training")
        x_train = [x_rows[idx] for idx in train_idx]
        y_train = [y_values[idx] for idx in train_idx]
        t_train = [treatment_values[idx] for idx in train_idx]
        x_test = [x_rows[idx] for idx in fold]
        y_model = _ridge_fit(x_train, y_train, alpha=ridge_alpha)
        t_model = _ridge_fit(x_train, t_train, alpha=ridge_alpha)
        y_pred = _ridge_predict(y_model, x_test)
        t_pred = _ridge_predict(t_model, x_test)
        for pos, idx in enumerate(fold):
            y_hat[idx] = y_pred[pos]
            t_hat[idx] = t_pred[pos]
    return CrossfitNuisanceResult(
        y_hat=y_hat,
        t_hat=t_hat,
        y_r2=_compute_rsquared(y_values, y_hat),
        t_r2=_compute_rsquared(treatment_values, t_hat),
    )


def _normal_pdf(value: float, sigma: float) -> float:
    if sigma <= 1e-12:
        return 0.0
    coefficient = 1.0 / (sigma * (2.0 * 3.141592653589793) ** 0.5)
    exponent = -0.5 * (value / sigma) ** 2
    return coefficient * (2.718281828459045 ** exponent)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    q = clamp(q, 0.0, 1.0)
    pos = q * (len(ordered) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def fit_continuous_treatment_tmle(
    *,
    x_rows: list[list[float]],
    y_values: list[float],
    treatment_values: list[float],
    config: ContinuousTreatmentTMLEConfig,
) -> ContinuousTreatmentTMLEFit:
    nuisance = crossfit_ridge_predictions(
        x_rows=x_rows,
        y_values=y_values,
        treatment_values=treatment_values,
        fold_count=config.fold_count,
        ridge_alpha=config.ridge_alpha,
    )
    y_resid = [y_values[idx] - nuisance.y_hat[idx] for idx in range(len(y_values))]
    t_resid = [treatment_values[idx] - nuisance.t_hat[idx] for idx in range(len(treatment_values))]
    initial_fit = _ols(
        y_resid,
        [[1.0, value] for value in t_resid],
        covariance_estimator="newey_west",
        covariance_lags=max(config.covariance_lags, 1),
    )
    theta_init = float(initial_fit.beta[1])
    treatment_scale = stddev(t_resid)
    if treatment_scale <= 1e-9:
        raise ValueError("Treatment residual scale is too small for TMLE")

    density_values = [_normal_pdf(value, treatment_scale) for value in t_resid]
    density_floor = max(_quantile(density_values, config.density_floor_quantile), config.min_density_floor)
    low_density_share = sum(1 for value in density_values if value <= density_floor + 1e-12) / len(density_values)

    score_base = [value / max(treatment_scale**2, 1e-9) for value in t_resid]
    overlap_stabilizer = [min(1.0, density / density_floor) for density in density_values]
    clever_raw = [score_base[idx] * overlap_stabilizer[idx] for idx in range(len(score_base))]
    clever_mean = mean(clever_raw)
    clever_scale = stddev(clever_raw)
    if clever_scale <= 1e-9:
        raise ValueError("TMLE clever covariate collapsed to zero scale")
    clever_standardized = [(value - clever_mean) / clever_scale for value in clever_raw]
    clever_covariate = [clamp(value, -config.h_clip, config.h_clip) for value in clever_standardized]
    clever_clip_share = (
        sum(1 for raw, clipped in zip(clever_standardized, clever_covariate) if abs(raw - clipped) > 1e-9)
        / len(clever_covariate)
    )

    outcome_scale = max(stddev(y_resid), 1.0)
    theta_init_std = theta_init * treatment_scale / outcome_scale
    targeting_residual = [y_resid[idx] - theta_init * t_resid[idx] for idx in range(len(y_values))]
    targeting_standardized = [value / outcome_scale for value in targeting_residual]
    fluctuation_fit = _ols(
        targeting_standardized,
        [[value] for value in clever_covariate],
        covariance_estimator="hc1",
    )
    raw_epsilon = float(fluctuation_fit.beta[0])
    epsilon_cap = max(0.1, config.epsilon_scale_multiple * max(abs(theta_init_std), 0.05))
    epsilon = clamp(raw_epsilon, -epsilon_cap, epsilon_cap)

    targeted_y_resid = [
        y_resid[idx] - (epsilon * outcome_scale * clever_covariate[idx])
        for idx in range(len(y_values))
    ]
    fit = _ols(
        targeted_y_resid,
        [[1.0, value] for value in t_resid],
        covariance_estimator="newey_west",
        covariance_lags=max(config.covariance_lags, 1),
    )
    epsilon_theta_ratio = abs(epsilon) / max(abs(theta_init_std), 1e-6)
    warning_flags: list[str] = []
    if (nuisance.y_r2 or 0.0) <= config.min_outcome_r2:
        warning_flags.append("tmle_weak_outcome_nuisance")
    if low_density_share > config.max_low_density_share:
        warning_flags.append("tmle_low_overlap")
    if epsilon_theta_ratio > config.max_epsilon_theta_ratio:
        warning_flags.append("tmle_large_targeting_step")
    if abs(raw_epsilon - epsilon) > 1e-9:
        warning_flags.append("tmle_epsilon_capped")

    return ContinuousTreatmentTMLEFit(
        fit=fit,
        y_hat=nuisance.y_hat,
        t_hat=nuisance.t_hat,
        y_r2=nuisance.y_r2,
        t_r2=nuisance.t_r2,
        theta_init=theta_init,
        theta_init_std=theta_init_std,
        epsilon=epsilon,
        raw_epsilon=raw_epsilon,
        epsilon_cap=epsilon_cap,
        clever_clip_share=clever_clip_share,
        low_density_share=low_density_share,
        density_floor=density_floor,
        treatment_scale=treatment_scale,
        outcome_scale=outcome_scale,
        epsilon_theta_ratio=epsilon_theta_ratio,
        valid=not warning_flags,
        warning_flags=warning_flags,
    )
