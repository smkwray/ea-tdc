from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from ea_tdc.continuous_treatment import (
    ContinuousTreatmentTMLEConfig,
    crossfit_ridge_predictions as _ct_crossfit_ridge_predictions,
    fit_continuous_treatment_tmle as _fit_continuous_treatment_tmle,
)
from ea_tdc.designs.quarterly import _quarter_sort_key
from ea_tdc.estimation import (
    _build_quarterly_target,
    _coerce_float,
    _compute_rsquared,
    _estimate_row_payload,
    _invert,
    _matmul,
    _matvec,
    _ols,
    _transpose,
    _write_estimates_csv,
)
from ea_tdc.paths import ProjectPaths
from ea_tdc.robustness import (
    DEFAULT_CONTROL_POLICY_MODE,
    _apply_control_policy,
    _base_controls_from_design,
    _extract_factor_controls,
    _load_alt_treatments,
    _merge_control_rows,
    _read_csv,
    _screen_features,
    build_control_universe,
    build_quarterly_robustness,
)
from ea_tdc.utils import utc_now_iso, write_json


DEFAULT_DML_FOLD_COUNT = 3
DEFAULT_RIDGE_ALPHA = 1.0
DEFAULT_NEGATIVE_CONTROL_TOP_N = 12
DEFAULT_NEGATIVE_CONTROL_LEADS = [1, 2, 4]
DEFAULT_NEGATIVE_CONTROL_HORIZONS = [0, 1]
DEFAULT_FOREST_TREE_COUNT = 40
DEFAULT_FOREST_MAX_DEPTH = 3
DEFAULT_FOREST_MIN_LEAF = 8
DEFAULT_FOREST_FEATURE_FRACTION = 0.5
DEFAULT_TMLE_H_CLIP = 4.0
DEFAULT_TMLE_EPSILON_SCALE_MULTIPLE = 3.0
DEFAULT_NEGATIVE_CONTROL_MIN_ROWS = 48
DEFAULT_NEGATIVE_CONTROL_MIN_SHARE = 0.25
NEGATIVE_CONTROL_EXCLUDE_TOKENS = (
    "__",
    "tdc_",
    "qra_",
    "iv_",
    "coord_",
    "tsyparty_",
    "wamest_",
    "slrwatch_",
    "_l1",
    "matched_total_deposits",
    "other_component_qoq",
    "reserve_balances",
)


@dataclass(frozen=True)
class QuarterlyDMLResult:
    estimates_path: Path
    summary_path: Path
    rows_written: int


@dataclass(frozen=True)
class NegativeControlMiningResult:
    summary_path: Path
    summary_csv_path: Path
    rows_written: int


@dataclass(frozen=True)
class QuarterlyForestResult:
    estimates_path: Path
    summary_path: Path
    rows_written: int


@dataclass(frozen=True)
class QuarterlyTMLEResult:
    estimates_path: Path
    summary_path: Path
    rows_written: int


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


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
        variance = sum((row[idx] - means[idx]) ** 2 for row in x_rows) / max(len(x_rows) - 1, 1)
        stds.append(variance ** 0.5 if variance > 1e-12 else 1.0)
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


def _crossfit_residuals(
    *,
    x_rows: list[list[float]],
    y_values: list[float],
    treatment_values: list[float],
    fold_count: int,
    ridge_alpha: float,
) -> tuple[list[float], list[float], float | None, float | None]:
    count = len(y_values)
    y_hat, t_hat, y_r2, t_r2 = _crossfit_ridge_predictions(
        x_rows=x_rows,
        y_values=y_values,
        treatment_values=treatment_values,
        fold_count=fold_count,
        ridge_alpha=ridge_alpha,
    )
    y_resid = [y_values[idx] - y_hat[idx] for idx in range(count)]
    t_resid = [treatment_values[idx] - t_hat[idx] for idx in range(count)]
    return (y_resid, t_resid, y_r2, t_r2)


def _crossfit_ridge_predictions(
    *,
    x_rows: list[list[float]],
    y_values: list[float],
    treatment_values: list[float],
    fold_count: int,
    ridge_alpha: float,
) -> tuple[list[float], list[float], float | None, float | None]:
    count = len(y_values)
    folds = _contiguous_folds(count, min(fold_count, max(count // 6, 2)))
    y_hat = [0.0 for _ in range(count)]
    t_hat = [0.0 for _ in range(count)]
    for fold in folds:
        train_idx = [idx for idx in range(count) if idx not in set(fold)]
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
    return (
        y_hat,
        t_hat,
        _compute_rsquared(y_values, y_hat),
        _compute_rsquared(treatment_values, t_hat),
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = _mean(values)
    return sum((value - avg) ** 2 for value in values) / len(values)


def _stddev(values: list[float]) -> float:
    variance = _variance(values)
    return variance ** 0.5 if variance > 0 else 0.0


def _sse(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = _mean(values)
    return sum((value - avg) ** 2 for value in values)


def _feature_candidates(values: list[float]) -> list[float]:
    ordered = sorted(values)
    if len(ordered) < 4:
        return []
    positions = sorted(set([
        len(ordered) // 4,
        len(ordered) // 2,
        (3 * len(ordered)) // 4,
    ]))
    thresholds: list[float] = []
    for pos in positions:
        if pos <= 0 or pos >= len(ordered):
            continue
        left = ordered[pos - 1]
        right = ordered[pos]
        threshold = (left + right) / 2.0
        if thresholds and abs(thresholds[-1] - threshold) < 1e-12:
            continue
        thresholds.append(threshold)
    return thresholds


def _build_forest_tree(
    x_rows: list[list[float]],
    y_values: list[float],
    *,
    rng: random.Random,
    max_depth: int,
    min_leaf: int,
    feature_fraction: float,
    depth: int = 0,
) -> dict[str, Any]:
    prediction = _mean(y_values)
    if depth >= max_depth or len(y_values) <= max(min_leaf * 2, 6):
        return {"type": "leaf", "value": prediction}

    feature_count = len(x_rows[0])
    sample_size = max(1, min(feature_count, int(round(feature_count * feature_fraction))))
    feature_ids = rng.sample(list(range(feature_count)), sample_size)
    parent_sse = _sse(y_values)
    best_split: dict[str, Any] | None = None

    for feature_idx in feature_ids:
        column = [row[feature_idx] for row in x_rows]
        for threshold in _feature_candidates(column):
            left_pairs = [(row, y) for row, y in zip(x_rows, y_values) if row[feature_idx] <= threshold]
            right_pairs = [(row, y) for row, y in zip(x_rows, y_values) if row[feature_idx] > threshold]
            if len(left_pairs) < min_leaf or len(right_pairs) < min_leaf:
                continue
            left_y = [item[1] for item in left_pairs]
            right_y = [item[1] for item in right_pairs]
            gain = parent_sse - (_sse(left_y) + _sse(right_y))
            if gain <= 1e-9:
                continue
            if best_split is None or gain > float(best_split["gain"]):
                best_split = {
                    "gain": gain,
                    "feature_idx": feature_idx,
                    "threshold": threshold,
                    "left_x": [item[0] for item in left_pairs],
                    "left_y": left_y,
                    "right_x": [item[0] for item in right_pairs],
                    "right_y": right_y,
                }

    if best_split is None:
        return {"type": "leaf", "value": prediction}

    return {
        "type": "node",
        "feature_idx": int(best_split["feature_idx"]),
        "threshold": float(best_split["threshold"]),
        "left": _build_forest_tree(
            list(best_split["left_x"]),
            list(best_split["left_y"]),
            rng=rng,
            max_depth=max_depth,
            min_leaf=min_leaf,
            feature_fraction=feature_fraction,
            depth=depth + 1,
        ),
        "right": _build_forest_tree(
            list(best_split["right_x"]),
            list(best_split["right_y"]),
            rng=rng,
            max_depth=max_depth,
            min_leaf=min_leaf,
            feature_fraction=feature_fraction,
            depth=depth + 1,
        ),
        "value": prediction,
    }


def _predict_tree(tree: dict[str, Any], row: list[float]) -> float:
    node = tree
    while str(node.get("type", "")) != "leaf":
        feature_idx = int(node["feature_idx"])
        if row[feature_idx] <= float(node["threshold"]):
            node = dict(node["left"])
        else:
            node = dict(node["right"])
    return float(node["value"])


def _fit_forest(
    x_rows: list[list[float]],
    y_values: list[float],
    *,
    tree_count: int,
    max_depth: int,
    min_leaf: int,
    feature_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    trees: list[dict[str, Any]] = []
    row_count = len(x_rows)
    for tree_idx in range(tree_count):
        sample_indices = [rng.randrange(row_count) for _ in range(row_count)]
        sample_x = [x_rows[idx] for idx in sample_indices]
        sample_y = [y_values[idx] for idx in sample_indices]
        tree_rng = random.Random(seed + tree_idx + 1)
        trees.append(
            _build_forest_tree(
                sample_x,
                sample_y,
                rng=tree_rng,
                max_depth=max_depth,
                min_leaf=min_leaf,
                feature_fraction=feature_fraction,
            )
        )
    return trees


def _forest_predict(forest: list[dict[str, Any]], x_rows: list[list[float]]) -> list[float]:
    if not forest:
        return [0.0 for _ in x_rows]
    return [
        sum(_predict_tree(tree, row) for tree in forest) / len(forest)
        for row in x_rows
    ]


def _crossfit_forest_residuals(
    *,
    x_rows: list[list[float]],
    y_values: list[float],
    treatment_values: list[float],
    fold_count: int,
    tree_count: int,
    max_depth: int,
    min_leaf: int,
    feature_fraction: float,
) -> tuple[list[float], list[float], float | None, float | None]:
    count = len(y_values)
    folds = _contiguous_folds(count, min(fold_count, max(count // 6, 2)))
    y_hat = [0.0 for _ in range(count)]
    t_hat = [0.0 for _ in range(count)]
    for fold_idx, fold in enumerate(folds):
        fold_set = set(fold)
        train_idx = [idx for idx in range(count) if idx not in fold_set]
        if len(train_idx) <= max(min_leaf * 2, 12):
            raise ValueError("Insufficient rows for cross-fit forest training")
        x_train = [x_rows[idx] for idx in train_idx]
        y_train = [y_values[idx] for idx in train_idx]
        t_train = [treatment_values[idx] for idx in train_idx]
        x_test = [x_rows[idx] for idx in fold]
        y_forest = _fit_forest(
            x_train,
            y_train,
            tree_count=tree_count,
            max_depth=max_depth,
            min_leaf=min_leaf,
            feature_fraction=feature_fraction,
            seed=1000 + fold_idx,
        )
        t_forest = _fit_forest(
            x_train,
            t_train,
            tree_count=tree_count,
            max_depth=max_depth,
            min_leaf=min_leaf,
            feature_fraction=feature_fraction,
            seed=2000 + fold_idx,
        )
        y_pred = _forest_predict(y_forest, x_test)
        t_pred = _forest_predict(t_forest, x_test)
        for pos, idx in enumerate(fold):
            y_hat[idx] = y_pred[pos]
            t_hat[idx] = t_pred[pos]
    y_resid = [y_values[idx] - y_hat[idx] for idx in range(count)]
    t_resid = [treatment_values[idx] - t_hat[idx] for idx in range(count)]
    return (
        y_resid,
        t_resid,
        _compute_rsquared(y_values, y_hat),
        _compute_rsquared(treatment_values, t_hat),
    )


def _load_recommended_factor_rows(
    paths: ProjectPaths,
    *,
    job_id: str,
    design_manifest: dict[str, Any],
    control_policy_mode: str = DEFAULT_CONTROL_POLICY_MODE,
) -> tuple[list[dict[str, str]], list[str], int]:
    summary_path = paths.manifests / f"{job_id}__robustness_summary.json"
    if not summary_path.exists():
        build_quarterly_robustness(paths, job_id=job_id, control_policy_mode=control_policy_mode)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    bundle_path = Path(str(design_manifest.get("bundle_path", "")))
    bundle_rows = _read_csv(bundle_path)
    bundle_rows.sort(key=lambda row: _quarter_sort_key(str(row.get("quarter", ""))))
    control_universe = build_control_universe(
        paths,
        quarter_grid=[str(row.get("quarter", "")).strip() for row in bundle_rows if str(row.get("quarter", "")).strip()],
    )
    merged_rows, universe_feature_ids = _merge_control_rows(bundle_rows, control_universe.panel_path)
    alt_treatment_map = _load_alt_treatments(paths)
    for row in merged_rows:
        quarter = str(row.get("quarter", "")).strip()
        for treatment_id in [
            "tdc_base_broad_depository_np_cu_ru_flow",
            "tdc_no_remit_bank_only",
            "tdc_domestic_bank_only_ru_flow",
            "tdc_bank_only_extended_1990",
        ]:
            row.setdefault(treatment_id, alt_treatment_map.get((treatment_id, quarter), ""))
    treatment_id = str(design_manifest.get("treatment_id", "")).strip()
    outcome_ids = [str(item) for item in design_manifest.get("outcome_ids", [])]
    eligible_feature_ids, _ = _apply_control_policy(
        candidate_ids=universe_feature_ids,
        treatment_id=treatment_id,
        outcome_ids=outcome_ids,
        mode=str(summary.get("control_policy_mode", DEFAULT_CONTROL_POLICY_MODE) or DEFAULT_CONTROL_POLICY_MODE),
    )
    screened = _screen_features(
        rows=merged_rows,
        candidate_ids=eligible_feature_ids,
        treatment_id=treatment_id,
        outcome_ids=outcome_ids,
        min_coverage=0.4,
    )
    recommended_k = int(summary.get("recommended_k", 0) or 0)
    if recommended_k <= 0:
        return merged_rows, _base_controls_from_design(design_manifest), 0
    recommended_factor_count = int(summary.get("recommended_factor_count", 4) or 4)
    selected = [item["feature_id"] for item in screened[:recommended_k]]
    factor_ids, factor_rows, _, _ = _extract_factor_controls(
        rows=merged_rows,
        feature_ids=selected,
        prefix=f"dflmx_k{recommended_k}",
        n_factors=recommended_factor_count,
    )
    return factor_rows if factor_rows else merged_rows, [*_base_controls_from_design(design_manifest), *factor_ids], recommended_k


def build_quarterly_dml(
    paths: ProjectPaths,
    *,
    job_id: str,
    fold_count: int = DEFAULT_DML_FOLD_COUNT,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    control_policy_mode: str = DEFAULT_CONTROL_POLICY_MODE,
) -> QuarterlyDMLResult:
    from ea_tdc.designs.quarterly import _load_jobs

    job_map = _load_jobs(paths.config / "dass_job_blueprint.yaml")
    if job_id not in job_map:
        raise KeyError(f"Unknown job_id: {job_id}")
    job = job_map[job_id]
    if str(job.get("estimator", "")).strip() != "lp":
        raise ValueError("Quarterly DML currently supports lp jobs only")

    design_manifest_path = paths.manifests / f"{job_id}__design_manifest.json"
    if not design_manifest_path.exists():
        raise FileNotFoundError(f"Missing design manifest for job '{job_id}'")
    design_manifest = json.loads(design_manifest_path.read_text(encoding="utf-8"))
    if str(design_manifest.get("status", "")).strip() != "ready_for_estimation":
        raise ValueError(f"Design manifest for job '{job_id}' is not ready_for_estimation")

    factor_rows, control_ids, recommended_k = _load_recommended_factor_rows(
        paths,
        job_id=job_id,
        design_manifest=design_manifest,
        control_policy_mode=control_policy_mode,
    )
    treatment_id = str(design_manifest.get("treatment_id", "")).strip()
    outcome_ids = [str(item) for item in design_manifest.get("outcome_ids", [])]
    horizons = [int(item) for item in design_manifest.get("horizon_grid", [])]
    response_type = str(job.get("response_type", "direct_at_h")).strip()

    result_rows: list[dict[str, str]] = []
    nuisance_rows: list[dict[str, Any]] = []
    for outcome_id in outcome_ids:
        for horizon in horizons:
            y_values: list[float] = []
            treatment_values: list[float] = []
            x_rows: list[list[float]] = []
            for idx, row in enumerate(factor_rows):
                treatment_value = _coerce_float(row.get(treatment_id, ""))
                if treatment_value is None:
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
                    factor_rows,
                    start_idx=idx,
                    outcome_id=outcome_id,
                    horizon=horizon,
                    response_type=response_type,
                )
                if target_value is None:
                    continue
                y_values.append(target_value)
                treatment_values.append(treatment_value)
                x_rows.append(controls)
            if len(y_values) <= len(control_ids) + 4:
                continue
            nuisance = _ct_crossfit_ridge_predictions(
                x_rows=x_rows,
                y_values=y_values,
                treatment_values=treatment_values,
                fold_count=fold_count,
                ridge_alpha=ridge_alpha,
            )
            y_resid = [y_values[idx] - nuisance.y_hat[idx] for idx in range(len(y_values))]
            t_resid = [treatment_values[idx] - nuisance.t_hat[idx] for idx in range(len(treatment_values))]
            fit = _ols(
                y_resid,
                [[1.0, value] for value in t_resid],
                covariance_estimator="newey_west",
                covariance_lags=max(horizon, 1),
            )
            payload = _estimate_row_payload(
                job_id=job_id,
                outcome_id=outcome_id,
                horizon=horizon,
                treatment_id=treatment_id,
                control_ids_used=control_ids,
                response_type=response_type,
                inference_method="dml_crossfit_ridge_newey_west",
                fit=fit,
            )
            payload["dml_fold_count"] = str(fold_count)
            payload["ridge_alpha"] = str(ridge_alpha)
            payload["nuisance_r2_outcome"] = "" if nuisance.y_r2 is None else str(round(nuisance.y_r2, 6))
            payload["nuisance_r2_treatment"] = "" if nuisance.t_r2 is None else str(round(nuisance.t_r2, 6))
            result_rows.append(payload)
            nuisance_rows.append(
                {
                    "job_id": job_id,
                    "outcome": outcome_id,
                    "horizon": horizon,
                    "nuisance_r2_outcome": nuisance.y_r2,
                    "nuisance_r2_treatment": nuisance.t_r2,
                    "n": len(y_values),
                }
            )

    estimates_path = paths.output / "models" / f"{job_id}__dml_estimates.csv"
    _write_estimates_csv(estimates_path, result_rows)
    summary_path = paths.manifests / f"{job_id}__dml_summary.json"
    write_json(
        summary_path,
        {
            "job_id": job_id,
            "generated_at": utc_now_iso(),
            "estimates_path": str(estimates_path),
            "rows_written": len(result_rows),
            "recommended_k": recommended_k,
            "fold_count": fold_count,
            "ridge_alpha": ridge_alpha,
            "control_ids": control_ids,
            "avg_nuisance_r2_outcome": (
                round(sum(item["nuisance_r2_outcome"] for item in nuisance_rows if item["nuisance_r2_outcome"] is not None) / max(sum(1 for item in nuisance_rows if item["nuisance_r2_outcome"] is not None), 1), 6)
                if nuisance_rows else None
            ),
            "avg_nuisance_r2_treatment": (
                round(sum(item["nuisance_r2_treatment"] for item in nuisance_rows if item["nuisance_r2_treatment"] is not None) / max(sum(1 for item in nuisance_rows if item["nuisance_r2_treatment"] is not None), 1), 6)
                if nuisance_rows else None
            ),
            "notes": "Continuous-treatment quarterly DML with cross-fitted ridge nuisance models over the screened-factor control branch.",
        },
    )
    return QuarterlyDMLResult(
        estimates_path=estimates_path,
        summary_path=summary_path,
        rows_written=len(result_rows),
    )


def build_quarterly_forest(
    paths: ProjectPaths,
    *,
    job_id: str,
    fold_count: int = DEFAULT_DML_FOLD_COUNT,
    tree_count: int = DEFAULT_FOREST_TREE_COUNT,
    max_depth: int = DEFAULT_FOREST_MAX_DEPTH,
    min_leaf: int = DEFAULT_FOREST_MIN_LEAF,
    feature_fraction: float = DEFAULT_FOREST_FEATURE_FRACTION,
    control_policy_mode: str = DEFAULT_CONTROL_POLICY_MODE,
) -> QuarterlyForestResult:
    from ea_tdc.designs.quarterly import _load_jobs

    job_map = _load_jobs(paths.config / "dass_job_blueprint.yaml")
    if job_id not in job_map:
        raise KeyError(f"Unknown job_id: {job_id}")
    job = job_map[job_id]
    if str(job.get("estimator", "")).strip() != "lp":
        raise ValueError("Quarterly forest robustness currently supports lp jobs only")

    design_manifest_path = paths.manifests / f"{job_id}__design_manifest.json"
    if not design_manifest_path.exists():
        raise FileNotFoundError(f"Missing design manifest for job '{job_id}'")
    design_manifest = json.loads(design_manifest_path.read_text(encoding="utf-8"))
    if str(design_manifest.get("status", "")).strip() != "ready_for_estimation":
        raise ValueError(f"Design manifest for job '{job_id}' is not ready_for_estimation")

    factor_rows, control_ids, recommended_k = _load_recommended_factor_rows(
        paths,
        job_id=job_id,
        design_manifest=design_manifest,
        control_policy_mode=control_policy_mode,
    )
    treatment_id = str(design_manifest.get("treatment_id", "")).strip()
    outcome_ids = [str(item) for item in design_manifest.get("outcome_ids", [])]
    horizons = [int(item) for item in design_manifest.get("horizon_grid", [])]
    response_type = str(job.get("response_type", "direct_at_h")).strip()

    result_rows: list[dict[str, str]] = []
    nuisance_rows: list[dict[str, Any]] = []
    for outcome_id in outcome_ids:
        for horizon in horizons:
            y_values: list[float] = []
            treatment_values: list[float] = []
            x_rows: list[list[float]] = []
            for idx, row in enumerate(factor_rows):
                treatment_value = _coerce_float(row.get(treatment_id, ""))
                if treatment_value is None:
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
                    factor_rows,
                    start_idx=idx,
                    outcome_id=outcome_id,
                    horizon=horizon,
                    response_type=response_type,
                )
                if target_value is None:
                    continue
                y_values.append(target_value)
                treatment_values.append(treatment_value)
                x_rows.append(controls)
            if len(y_values) <= max(len(control_ids) + 6, min_leaf * 3):
                continue
            y_resid, t_resid, y_r2, t_r2 = _crossfit_forest_residuals(
                x_rows=x_rows,
                y_values=y_values,
                treatment_values=treatment_values,
                fold_count=fold_count,
                tree_count=tree_count,
                max_depth=max_depth,
                min_leaf=min_leaf,
                feature_fraction=feature_fraction,
            )
            fit = _ols(
                y_resid,
                [[1.0, value] for value in t_resid],
                covariance_estimator="newey_west",
                covariance_lags=max(horizon, 1),
            )
            payload = _estimate_row_payload(
                job_id=job_id,
                outcome_id=outcome_id,
                horizon=horizon,
                treatment_id=treatment_id,
                control_ids_used=control_ids,
                response_type=response_type,
                inference_method="forest_crossfit_newey_west",
                fit=fit,
            )
            payload["forest_tree_count"] = str(tree_count)
            payload["forest_max_depth"] = str(max_depth)
            payload["forest_min_leaf"] = str(min_leaf)
            payload["forest_feature_fraction"] = str(round(feature_fraction, 4))
            payload["nuisance_r2_outcome"] = "" if y_r2 is None else str(round(y_r2, 6))
            payload["nuisance_r2_treatment"] = "" if t_r2 is None else str(round(t_r2, 6))
            result_rows.append(payload)
            nuisance_rows.append(
                {
                    "job_id": job_id,
                    "outcome": outcome_id,
                    "horizon": horizon,
                    "nuisance_r2_outcome": y_r2,
                    "nuisance_r2_treatment": t_r2,
                    "n": len(y_values),
                }
            )

    estimates_path = paths.output / "models" / f"{job_id}__forest_estimates.csv"
    _write_estimates_csv(estimates_path, result_rows)
    summary_path = paths.manifests / f"{job_id}__forest_summary.json"
    write_json(
        summary_path,
        {
            "job_id": job_id,
            "generated_at": utc_now_iso(),
            "estimates_path": str(estimates_path),
            "rows_written": len(result_rows),
            "recommended_k": recommended_k,
            "fold_count": fold_count,
            "tree_count": tree_count,
            "max_depth": max_depth,
            "min_leaf": min_leaf,
            "feature_fraction": feature_fraction,
            "control_ids": control_ids,
            "avg_nuisance_r2_outcome": (
                round(sum(item["nuisance_r2_outcome"] for item in nuisance_rows if item["nuisance_r2_outcome"] is not None) / max(sum(1 for item in nuisance_rows if item["nuisance_r2_outcome"] is not None), 1), 6)
                if nuisance_rows else None
            ),
            "avg_nuisance_r2_treatment": (
                round(sum(item["nuisance_r2_treatment"] for item in nuisance_rows if item["nuisance_r2_treatment"] is not None) / max(sum(1 for item in nuisance_rows if item["nuisance_r2_treatment"] is not None), 1), 6)
                if nuisance_rows else None
            ),
            "notes": "Continuous-treatment quarterly orthogonal learner with cross-fitted bagged shallow-tree nuisance models over the screened-factor control branch.",
        },
    )
    return QuarterlyForestResult(
        estimates_path=estimates_path,
        summary_path=summary_path,
        rows_written=len(result_rows),
    )


def build_quarterly_tmle(
    paths: ProjectPaths,
    *,
    job_id: str,
    fold_count: int = DEFAULT_DML_FOLD_COUNT,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    h_clip: float = DEFAULT_TMLE_H_CLIP,
    epsilon_scale_multiple: float = DEFAULT_TMLE_EPSILON_SCALE_MULTIPLE,
    control_policy_mode: str = DEFAULT_CONTROL_POLICY_MODE,
) -> QuarterlyTMLEResult:
    from ea_tdc.designs.quarterly import _load_jobs

    job_map = _load_jobs(paths.config / "dass_job_blueprint.yaml")
    if job_id not in job_map:
        raise KeyError(f"Unknown job_id: {job_id}")
    job = job_map[job_id]
    if str(job.get("estimator", "")).strip() != "lp":
        raise ValueError("Quarterly TMLE currently supports lp jobs only")

    design_manifest_path = paths.manifests / f"{job_id}__design_manifest.json"
    if not design_manifest_path.exists():
        raise FileNotFoundError(f"Missing design manifest for job '{job_id}'")
    design_manifest = json.loads(design_manifest_path.read_text(encoding="utf-8"))
    if str(design_manifest.get("status", "")).strip() != "ready_for_estimation":
        raise ValueError(f"Design manifest for job '{job_id}' is not ready_for_estimation")

    factor_rows, control_ids, recommended_k = _load_recommended_factor_rows(
        paths,
        job_id=job_id,
        design_manifest=design_manifest,
        control_policy_mode=control_policy_mode,
    )
    treatment_id = str(design_manifest.get("treatment_id", "")).strip()
    outcome_ids = [str(item) for item in design_manifest.get("outcome_ids", [])]
    horizons = [int(item) for item in design_manifest.get("horizon_grid", [])]
    response_type = str(job.get("response_type", "direct_at_h")).strip()

    result_rows: list[dict[str, str]] = []
    nuisance_rows: list[dict[str, Any]] = []
    for outcome_id in outcome_ids:
        for horizon in horizons:
            y_values: list[float] = []
            treatment_values: list[float] = []
            x_rows: list[list[float]] = []
            for idx, row in enumerate(factor_rows):
                treatment_value = _coerce_float(row.get(treatment_id, ""))
                if treatment_value is None:
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
                    factor_rows,
                    start_idx=idx,
                    outcome_id=outcome_id,
                    horizon=horizon,
                    response_type=response_type,
                )
                if target_value is None:
                    continue
                y_values.append(target_value)
                treatment_values.append(treatment_value)
                x_rows.append(controls)
            if len(y_values) <= len(control_ids) + 4:
                continue
            tmle_fit = _fit_continuous_treatment_tmle(
                x_rows=x_rows,
                y_values=y_values,
                treatment_values=treatment_values,
                config=ContinuousTreatmentTMLEConfig(
                    fold_count=fold_count,
                    ridge_alpha=ridge_alpha,
                    h_clip=h_clip,
                    epsilon_scale_multiple=epsilon_scale_multiple,
                    covariance_lags=max(horizon, 1),
                ),
            )
            payload = _estimate_row_payload(
                job_id=job_id,
                outcome_id=outcome_id,
                horizon=horizon,
                treatment_id=treatment_id,
                control_ids_used=control_ids,
                response_type=response_type,
                inference_method="tmle_crossfit_ridge_newey_west",
                fit=tmle_fit.fit,
            )
            payload["tmle_fold_count"] = str(fold_count)
            payload["ridge_alpha"] = str(ridge_alpha)
            payload["tmle_theta_init"] = str(round(tmle_fit.theta_init, 8))
            payload["tmle_theta_init_std"] = str(round(tmle_fit.theta_init_std, 8))
            payload["tmle_epsilon"] = str(round(tmle_fit.epsilon, 8))
            payload["tmle_raw_epsilon"] = str(round(tmle_fit.raw_epsilon, 8))
            payload["tmle_epsilon_cap"] = str(round(tmle_fit.epsilon_cap, 8))
            payload["tmle_h_clip"] = str(round(h_clip, 4))
            payload["tmle_h_clip_share"] = str(round(tmle_fit.clever_clip_share, 6))
            payload["tmle_low_density_share"] = str(round(tmle_fit.low_density_share, 6))
            payload["tmle_density_floor"] = str(round(tmle_fit.density_floor, 8))
            payload["tmle_treatment_scale"] = str(round(tmle_fit.treatment_scale, 8))
            payload["tmle_outcome_scale"] = str(round(tmle_fit.outcome_scale, 8))
            payload["tmle_epsilon_theta_ratio"] = str(round(tmle_fit.epsilon_theta_ratio, 6))
            payload["tmle_valid"] = "true" if tmle_fit.valid else "false"
            payload["tmle_epsilon_was_clipped"] = "true" if abs(tmle_fit.raw_epsilon - tmle_fit.epsilon) > 1e-9 else "false"
            payload["nuisance_r2_outcome"] = "" if tmle_fit.y_r2 is None else str(round(tmle_fit.y_r2, 6))
            payload["nuisance_r2_treatment"] = "" if tmle_fit.t_r2 is None else str(round(tmle_fit.t_r2, 6))
            payload["warning_flags"] = ",".join(
                item for item in [payload.get("warning_flags", ""), *tmle_fit.warning_flags] if item
            )
            result_rows.append(payload)
            nuisance_rows.append(
                {
                    "job_id": job_id,
                    "outcome": outcome_id,
                    "horizon": horizon,
                    "theta_init": tmle_fit.theta_init,
                    "theta_init_std": tmle_fit.theta_init_std,
                    "epsilon": tmle_fit.epsilon,
                    "raw_epsilon": tmle_fit.raw_epsilon,
                    "epsilon_cap": tmle_fit.epsilon_cap,
                    "clip_share": tmle_fit.clever_clip_share,
                    "low_density_share": tmle_fit.low_density_share,
                    "density_floor": tmle_fit.density_floor,
                    "epsilon_theta_ratio": tmle_fit.epsilon_theta_ratio,
                    "valid": tmle_fit.valid,
                    "warning_flags": list(tmle_fit.warning_flags),
                    "nuisance_r2_outcome": tmle_fit.y_r2,
                    "nuisance_r2_treatment": tmle_fit.t_r2,
                    "n": len(y_values),
                }
            )

    estimates_path = paths.output / "models" / f"{job_id}__tmle_estimates.csv"
    _write_estimates_csv(estimates_path, result_rows)
    summary_path = paths.manifests / f"{job_id}__tmle_summary.json"
    write_json(
        summary_path,
        {
            "job_id": job_id,
            "generated_at": utc_now_iso(),
            "estimates_path": str(estimates_path),
            "rows_written": len(result_rows),
            "recommended_k": recommended_k,
            "fold_count": fold_count,
            "ridge_alpha": ridge_alpha,
            "h_clip": h_clip,
            "epsilon_scale_multiple": epsilon_scale_multiple,
            "control_ids": control_ids,
            "avg_tmle_theta_init": (
                round(sum(item["theta_init"] for item in nuisance_rows) / len(nuisance_rows), 8)
                if nuisance_rows else None
            ),
            "avg_tmle_theta_init_std": (
                round(sum(item["theta_init_std"] for item in nuisance_rows) / len(nuisance_rows), 8)
                if nuisance_rows else None
            ),
            "avg_tmle_epsilon": (
                round(sum(item["epsilon"] for item in nuisance_rows) / len(nuisance_rows), 8)
                if nuisance_rows else None
            ),
            "avg_tmle_raw_epsilon": (
                round(sum(item["raw_epsilon"] for item in nuisance_rows) / len(nuisance_rows), 8)
                if nuisance_rows else None
            ),
            "avg_tmle_epsilon_cap": (
                round(sum(item["epsilon_cap"] for item in nuisance_rows) / len(nuisance_rows), 8)
                if nuisance_rows else None
            ),
            "avg_tmle_h_clip_share": (
                round(sum(item["clip_share"] for item in nuisance_rows) / len(nuisance_rows), 6)
                if nuisance_rows else None
            ),
            "avg_tmle_low_density_share": (
                round(sum(item["low_density_share"] for item in nuisance_rows) / len(nuisance_rows), 6)
                if nuisance_rows else None
            ),
            "avg_tmle_density_floor": (
                round(sum(item["density_floor"] for item in nuisance_rows) / len(nuisance_rows), 8)
                if nuisance_rows else None
            ),
            "avg_tmle_epsilon_theta_ratio": (
                round(sum(item["epsilon_theta_ratio"] for item in nuisance_rows) / len(nuisance_rows), 6)
                if nuisance_rows else None
            ),
            "valid_row_count": sum(1 for item in nuisance_rows if item["valid"]),
            "invalid_row_count": sum(1 for item in nuisance_rows if not item["valid"]),
            "tmle_warning_counts": {
                warning: sum(1 for item in nuisance_rows if warning in item["warning_flags"])
                for warning in sorted({flag for item in nuisance_rows for flag in item["warning_flags"]})
            },
            "epsilon_clip_rate": (
                round(
                    sum(1 for item in nuisance_rows if abs(item["raw_epsilon"] - item["epsilon"]) > 1e-9)
                    / len(nuisance_rows),
                    6,
                )
                if nuisance_rows else None
            ),
            "avg_nuisance_r2_outcome": (
                round(sum(item["nuisance_r2_outcome"] for item in nuisance_rows if item["nuisance_r2_outcome"] is not None) / max(sum(1 for item in nuisance_rows if item["nuisance_r2_outcome"] is not None), 1), 6)
                if nuisance_rows else None
            ),
            "avg_nuisance_r2_treatment": (
                round(sum(item["nuisance_r2_treatment"] for item in nuisance_rows if item["nuisance_r2_treatment"] is not None) / max(sum(1 for item in nuisance_rows if item["nuisance_r2_treatment"] is not None), 1), 6)
                if nuisance_rows else None
            ),
            "notes": "Continuous-treatment quarterly TMLE-style update over the screened-factor branch with density-stabilized clever covariates, overlap diagnostics, and hard targeting-step validity gates.",
        },
    )
    return QuarterlyTMLEResult(
        estimates_path=estimates_path,
        summary_path=summary_path,
        rows_written=len(result_rows),
    )


def _candidate_negative_control_columns(
    rows: list[dict[str, str]],
    *,
    treatment_id: str,
    outcome_ids: list[str],
    control_ids: list[str],
    instrument_ids: list[str],
) -> list[str]:
    if not rows:
        return []
    excluded = set([treatment_id, *outcome_ids, *control_ids, *instrument_ids, "quarter", "cutoff_timestamp"])
    candidates: list[str] = []
    for column in rows[0].keys():
        if column in excluded:
            continue
        if any(token in column for token in NEGATIVE_CONTROL_EXCLUDE_TOKENS):
            continue
        if any(column.endswith(suffix) for suffix in ("_available_at", "_source_repo")):
            continue
        observed = sum(1 for row in rows if _coerce_float(row.get(column, "")) is not None)
        if observed < max(DEFAULT_NEGATIVE_CONTROL_MIN_ROWS, int(len(rows) * DEFAULT_NEGATIVE_CONTROL_MIN_SHARE)):
            continue
        candidates.append(column)
    return candidates


def _lead_treatment_rows(
    rows: list[dict[str, str]],
    treatment_id: str,
    lead: int,
) -> list[dict[str, str]]:
    shifted = [dict(row) for row in rows]
    for idx, row in enumerate(shifted):
        source_idx = idx + lead
        row[f"lead_placebo_{lead}"] = rows[source_idx].get(treatment_id, "") if source_idx < len(rows) else ""
    return shifted


def build_negative_control_mining(
    paths: ProjectPaths,
    *,
    job_id: str,
    top_n: int = DEFAULT_NEGATIVE_CONTROL_TOP_N,
    control_policy_mode: str = DEFAULT_CONTROL_POLICY_MODE,
) -> NegativeControlMiningResult:
    from ea_tdc.designs.quarterly import _load_jobs

    job_map = _load_jobs(paths.config / "dass_job_blueprint.yaml")
    if job_id not in job_map:
        raise KeyError(f"Unknown job_id: {job_id}")
    job = job_map[job_id]
    if str(job.get("estimator", "")).strip() != "lp":
        raise ValueError("Negative-control mining currently supports quarterly lp jobs only")

    design_manifest = json.loads((paths.manifests / f"{job_id}__design_manifest.json").read_text(encoding="utf-8"))
    rows, control_ids, recommended_k = _load_recommended_factor_rows(
        paths,
        job_id=job_id,
        design_manifest=design_manifest,
        control_policy_mode=control_policy_mode,
    )
    treatment_id = str(design_manifest.get("treatment_id", "")).strip()
    outcome_ids = [str(item) for item in design_manifest.get("outcome_ids", [])]
    instrument_ids = [str(item) for item in design_manifest.get("instrument_ids", [])]
    response_type = str(job.get("response_type", "direct_at_h")).strip()

    candidate_columns = _candidate_negative_control_columns(
        rows,
        treatment_id=treatment_id,
        outcome_ids=outcome_ids,
        control_ids=control_ids,
        instrument_ids=instrument_ids,
    )

    def _mine_candidate_rows(active_control_ids: list[str]) -> list[dict[str, Any]]:
        mined_rows: list[dict[str, Any]] = []
        for candidate in candidate_columns:
            stats: list[tuple[int, float, float]] = []
            for horizon in DEFAULT_NEGATIVE_CONTROL_HORIZONS:
                y_values: list[float] = []
                x_rows: list[list[float]] = []
                for idx, row in enumerate(rows):
                    treatment_value = _coerce_float(row.get(treatment_id, ""))
                    if treatment_value is None:
                        continue
                    target_value = _build_quarterly_target(
                        rows,
                        start_idx=idx,
                        outcome_id=candidate,
                        horizon=horizon,
                        response_type=response_type,
                    )
                    if target_value is None:
                        continue
                    controls: list[float] = []
                    controls_ok = True
                    for control_id in active_control_ids:
                        control_value = _coerce_float(row.get(control_id, ""))
                        if control_value is None:
                            controls_ok = False
                            break
                        controls.append(control_value)
                    if not controls_ok:
                        continue
                    y_values.append(target_value)
                    x_rows.append([1.0, treatment_value, *controls])
                if len(y_values) <= len(active_control_ids) + 2:
                    continue
                try:
                    fit = _ols(
                        y_values,
                        x_rows,
                        covariance_estimator="newey_west",
                        covariance_lags=max(horizon, 1),
                    )
                except ValueError:
                    continue
                se = fit.ses[1]
                if se <= 0:
                    continue
                z_score = abs(fit.beta[1] / se)
                p_value = _coerce_float(
                    _estimate_row_payload(
                        job_id=job_id,
                        outcome_id=candidate,
                        horizon=horizon,
                        treatment_id=treatment_id,
                        control_ids_used=active_control_ids,
                        response_type=response_type,
                        inference_method="ols_newey_west_scaffold",
                        fit=fit,
                    ).get("p_value_normal", "")
                )
                if p_value is None:
                    continue
                stats.append((horizon, z_score, p_value))
            if stats:
                mined_rows.append(
                    {
                        "job_id": job_id,
                        "candidate_outcome": candidate,
                        "avg_abs_z": round(sum(item[1] for item in stats) / len(stats), 6),
                        "min_p_value": round(min(item[2] for item in stats), 6),
                        "tested_horizons": ",".join(str(item[0]) for item in stats),
                        "control_branch": "recommended_factors" if active_control_ids == control_ids else "baseline_core",
                    }
                )
        mined_rows.sort(key=lambda item: (item["avg_abs_z"], item["min_p_value"], item["candidate_outcome"]))
        return mined_rows

    candidate_rows = _mine_candidate_rows(control_ids)
    baseline_control_ids = _base_controls_from_design(design_manifest)
    baseline_candidate_rows = []
    if baseline_control_ids != control_ids:
        baseline_candidate_rows = _mine_candidate_rows(baseline_control_ids)
    if candidate_rows and baseline_candidate_rows:
        candidate_rows = (
            candidate_rows
            if (
                len(candidate_rows) > len(baseline_candidate_rows)
                or (
                    len(candidate_rows) == len(baseline_candidate_rows)
                    and float(candidate_rows[0]["avg_abs_z"]) <= float(baseline_candidate_rows[0]["avg_abs_z"])
                )
            )
            else baseline_candidate_rows
        )
    elif not candidate_rows:
        candidate_rows = baseline_candidate_rows

    placebo_rows: list[dict[str, Any]] = []
    for lead in DEFAULT_NEGATIVE_CONTROL_LEADS:
        shifted_rows = _lead_treatment_rows(rows, treatment_id, lead)
        estimates = []
        for outcome_id in outcome_ids:
            for horizon in DEFAULT_NEGATIVE_CONTROL_HORIZONS:
                y_values: list[float] = []
                x_rows: list[list[float]] = []
                for idx, row in enumerate(shifted_rows):
                    placebo_treatment = _coerce_float(row.get(f"lead_placebo_{lead}", ""))
                    if placebo_treatment is None:
                        continue
                    target_value = _build_quarterly_target(
                        shifted_rows,
                        start_idx=idx,
                        outcome_id=outcome_id,
                        horizon=horizon,
                        response_type=response_type,
                    )
                    if target_value is None:
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
                    y_values.append(target_value)
                    x_rows.append([1.0, placebo_treatment, *controls])
                if len(y_values) <= len(control_ids) + 2:
                    continue
                try:
                    fit = _ols(
                        y_values,
                        x_rows,
                        covariance_estimator="newey_west",
                        covariance_lags=max(horizon, 1),
                    )
                except ValueError:
                    continue
                payload = _estimate_row_payload(
                    job_id=job_id,
                    outcome_id=outcome_id,
                    horizon=horizon,
                    treatment_id=f"lead_placebo_{lead}",
                    control_ids_used=control_ids,
                    response_type=response_type,
                    inference_method="negative_control_lead_placebo",
                    fit=fit,
                )
                payload["lead"] = str(lead)
                estimates.append(payload)
        abs_z = []
        for row in estimates:
            z_val = _coerce_float(row.get("z_score", ""))
            if z_val is not None:
                abs_z.append(abs(z_val))
        placebo_rows.append(
            {
                "job_id": job_id,
                "lead": lead,
                "rows_written": len(estimates),
                "avg_abs_z": round(sum(abs_z) / len(abs_z), 6) if abs_z else None,
                "max_abs_z": round(max(abs_z), 6) if abs_z else None,
            }
        )

    summary_path = paths.reports / f"{job_id}__negative_control_mining.json"
    summary_csv_path = paths.reports / f"{job_id}__negative_control_mining.csv"
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["job_id", "candidate_outcome", "avg_abs_z", "min_p_value", "tested_horizons", "control_branch"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidate_rows[:top_n])
    write_json(
        summary_path,
        {
            "job_id": job_id,
            "generated_at": utc_now_iso(),
            "recommended_k": recommended_k,
            "candidate_control_branch": candidate_rows[0]["control_branch"] if candidate_rows else None,
            "recommended_branch_candidate_count": len(_mine_candidate_rows(control_ids)),
            "baseline_branch_candidate_count": len(baseline_candidate_rows),
            "top_clean_candidates": candidate_rows[:top_n],
            "most_responsive_candidates": list(reversed(sorted(candidate_rows, key=lambda item: (item["avg_abs_z"], -item["min_p_value"]))))[:top_n],
            "lead_placebos": placebo_rows,
            "notes": "Negative-control mining here means lead-placebo checks plus candidate placebo-outcome screening over numeric bundle columns that are not treatment, instrument, or primary outcome fields. Candidate outcomes need enough coverage to support quarterly horizons, and the miner now compares the screened-factor branch against the baseline core controls rather than returning an empty panel immediately.",
        },
    )
    return NegativeControlMiningResult(
        summary_path=summary_path,
        summary_csv_path=summary_csv_path,
        rows_written=len(candidate_rows),
    )
