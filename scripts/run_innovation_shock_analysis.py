from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from math import sqrt
from statistics import fmean, median
from typing import Any

from ea_tdc.designs.quarterly import _quarter_sort_key
from ea_tdc.estimation import (
    _build_quarterly_target,
    _coerce_float,
    _estimate_row_payload,
    _ols,
    _write_estimates_csv,
)
from ea_tdc.paths import project_paths
from ea_tdc.residualized_shock import _load_factor_branch
from ea_tdc.robustness import DEFAULT_CONTROL_POLICY_MODE, MIN_COVERAGE
from ea_tdc.utils import utc_now_iso, write_json


JOB_ID = "tdc_tier2_mmf_rrp_canonical_full_panel"
TREATMENT_ID = "tdc_tier2_canonical_di_mmf_rrp_prop_qoq"
SHOCK_PREFIX = "tdc_tier2_canonical_innovation"
HORIZONS = [0, 1, 2, 4, 8]

REQUESTED_OUTCOMES = [
    "matched_total_deposits",
    "domestic_nonbank_deposits_qoq",
    "tdcpass_strict_loan_consumer_credit_qoq",
    "tdcpass_strict_loan_mortgages_qoq",
    "tdcpass_strict_loan_core_min_qoq",
    "domestic_nonbank_other_component_tier2_canonical_no_toc_di_mmf_rrp_prop_qoq",
    "reserve_balances_net_fed_treasury_qoq",
    "dgs1",
    "dgs2",
    "row_private_flow_block",
]

OUTCOME_LABELS = {
    "matched_total_deposits": "Deposits total",
    "domestic_nonbank_deposits_qoq": "Deposits domestic nonbank",
    "tdcpass_strict_loan_consumer_credit_qoq": "Consumer credit",
    "tdcpass_strict_loan_mortgages_qoq": "Mortgages",
    "tdcpass_strict_loan_core_min_qoq": "Loan core",
    "domestic_nonbank_other_component_tier2_canonical_no_toc_di_mmf_rrp_prop_qoq": "Residual DN, no TOC",
    "reserve_balances_net_fed_treasury_qoq": "Reserves less Fed TS",
    "dgs1": "1y Treasury",
    "dgs2": "2y Treasury",
    "row_private_flow_block": "ROW private flow",
}


@dataclass(frozen=True)
class ShockSpec:
    run_id: str
    shock_id: str
    first_stage_predictors: list[str]
    lp_controls: list[str]
    first_stage_note: str


RIDGE_ALPHA = 10.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stable(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.12g}"


def add_quarterly_average_from_raw(
    rows: list[dict[str, str]],
    *,
    raw_path: Path,
    output_id: str,
) -> None:
    values_by_quarter: dict[str, list[float]] = {}
    for raw in read_csv(raw_path):
        raw_value = _coerce_float(raw.get("value", ""))
        if raw_value is None:
            continue
        try:
            raw_date = date.fromisoformat(str(raw.get("date", ""))[:10])
        except ValueError:
            continue
        quarter = f"{raw_date.year}Q{((raw_date.month - 1) // 3) + 1}"
        values_by_quarter.setdefault(quarter, []).append(raw_value)
    average_by_quarter = {
        quarter: fmean(values)
        for quarter, values in values_by_quarter.items()
        if values
    }
    for row in rows:
        quarter = str(row.get("quarter", "")).strip()
        row[output_id] = stable(average_by_quarter.get(quarter))


def add_lags(rows: list[dict[str, str]], columns: list[str], *, lag: int = 1) -> list[str]:
    lag_ids: list[str] = []
    for column in columns:
        lag_id = f"{column}_lag{lag}"
        lag_ids.append(lag_id)
        for idx, row in enumerate(rows):
            source_idx = idx - lag
            row[lag_id] = rows[source_idx].get(column, "") if source_idx >= 0 else ""
    return lag_ids


def impute_columns(rows: list[dict[str, str]], columns: list[str], *, prefix: str = "imp") -> list[str]:
    imputed_ids: list[str] = []
    for column in columns:
        imputed_id = f"{prefix}_{column}"
        values = [_coerce_float(row.get(column, "")) for row in rows]
        observed = [value for value in values if value is not None]
        fill = median(observed) if observed else 0.0
        imputed_ids.append(imputed_id)
        for row, value in zip(rows, values):
            row[imputed_id] = stable(fill if value is None else value)
    return imputed_ids


def matrix_rank(matrix: list[list[float]], *, tol: float = 1e-9) -> int:
    if not matrix:
        return 0
    work = [row[:] for row in matrix]
    row_count = len(work)
    col_count = len(work[0])
    rank = 0
    for col in range(col_count):
        pivot = max(range(rank, row_count), key=lambda idx: abs(work[idx][col]))
        pivot_value = work[pivot][col]
        if abs(pivot_value) <= tol:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        for idx in range(rank + 1, row_count):
            factor = work[idx][col] / pivot_value
            if abs(factor) <= tol:
                continue
            for jdx in range(col, col_count):
                work[idx][jdx] -= factor * work[rank][jdx]
        rank += 1
        if rank == row_count:
            break
    return rank


def solve_linear_system(matrix: list[list[float]], values: list[float]) -> list[float]:
    size = len(values)
    work = [row[:] + [value] for row, value in zip(matrix, values)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda idx: abs(work[idx][col]))
        if abs(work[pivot][col]) <= 1e-12:
            raise ValueError("singular ridge system")
        work[col], work[pivot] = work[pivot], work[col]
        pivot_value = work[col][col]
        for jdx in range(col, size + 1):
            work[col][jdx] /= pivot_value
        for idx in range(size):
            if idx == col:
                continue
            factor = work[idx][col]
            if abs(factor) <= 1e-12:
                continue
            for jdx in range(col, size + 1):
                work[idx][jdx] -= factor * work[col][jdx]
    return [work[idx][size] for idx in range(size)]


def fit_standardized_ridge(
    y_values: list[float],
    x_rows: list[list[float]],
    *,
    alpha: float = RIDGE_ALPHA,
) -> dict[str, Any]:
    if not y_values or not x_rows:
        raise ValueError("empty ridge sample")
    predictor_count = len(x_rows[0])
    means: list[float] = []
    scales: list[float] = []
    standardized_rows: list[list[float]] = []
    for col in range(predictor_count):
        values = [row[col] for row in x_rows]
        mean = fmean(values)
        variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
        scale = sqrt(variance)
        if scale <= 1e-12:
            scale = 1.0
        means.append(mean)
        scales.append(scale)
    for row in x_rows:
        standardized_rows.append([(value - mean) / scale for value, mean, scale in zip(row, means, scales)])
    y_mean = fmean(y_values)
    y_centered = [value - y_mean for value in y_values]
    xtx = [[0.0 for _ in range(predictor_count)] for _ in range(predictor_count)]
    xty = [0.0 for _ in range(predictor_count)]
    for y, row in zip(y_centered, standardized_rows):
        for i, value_i in enumerate(row):
            xty[i] += value_i * y
            for j, value_j in enumerate(row):
                xtx[i][j] += value_i * value_j
    for idx in range(predictor_count):
        xtx[idx][idx] += alpha
    coefficients = solve_linear_system(xtx, xty)
    return {
        "alpha": alpha,
        "means": means,
        "scales": scales,
        "y_mean": y_mean,
        "coefficients": coefficients,
    }


def predict_standardized_ridge(fit: dict[str, Any], x_row: list[float]) -> float:
    total = float(fit["y_mean"])
    for value, mean, scale, coefficient in zip(
        x_row,
        fit["means"],
        fit["scales"],
        fit["coefficients"],
    ):
        total += coefficient * ((value - mean) / scale)
    return total


def select_cycle_controls(rows: list[dict[str, str]], *, max_controls: int = 24) -> list[str]:
    preferred_terms = [
        "unrate",
        "payems",
        "indpro",
        "umcsent",
        "psavert",
        "houst",
        "permit",
        "mortgage",
        "baa",
        "baaff",
        "delinq",
        "drt",
        "recession",
        "recprob",
        "sahm",
        "cpi",
        "pce",
        "dgs10",
        "dgs2",
    ]
    columns = [column for column in rows[0].keys() if "__lag001" in column]
    selected: list[str] = []
    seen_bases: set[str] = set()
    for term in preferred_terms:
        for column in columns:
            lowered = column.lower()
            if term not in lowered:
                continue
            base = column.split("__", 2)[1] if "__" in column else column
            if base in seen_bases:
                continue
            if any(_coerce_float(row.get(column, "")) is not None for row in rows):
                selected.append(column)
                seen_bases.add(base)
                break
        if len(selected) >= max_controls:
            break
    return selected


def fit_crossfit_shock(
    rows: list[dict[str, str]],
    *,
    treatment_id: str,
    predictors: list[str],
    shock_id: str,
    folds: int = 5,
) -> dict[str, Any]:
    eligible_indices = [
        idx
        for idx, row in enumerate(rows)
        if _coerce_float(row.get(treatment_id, "")) is not None
        and all(_coerce_float(row.get(predictor, "")) is not None for predictor in predictors)
    ]
    predictions: dict[int, float] = {}
    dropped_by_fold: list[list[str]] = []
    for fold in range(folds):
        test_indices = [
            idx
            for order, idx in enumerate(eligible_indices)
            if order % folds == fold
        ]
        train_indices = [idx for idx in eligible_indices if idx not in set(test_indices)]
        y_train = [_coerce_float(rows[idx].get(treatment_id, "")) or 0.0 for idx in train_indices]
        x_train = [
            [(_coerce_float(rows[idx].get(predictor, "")) or 0.0) for predictor in predictors]
            for idx in train_indices
        ]
        fit = fit_standardized_ridge(y_train, x_train)
        dropped_by_fold.append([])
        for idx in test_indices:
            x = [(_coerce_float(rows[idx].get(predictor, "")) or 0.0) for predictor in predictors]
            predictions[idx] = predict_standardized_ridge(fit, x)

    actuals: list[float] = []
    residuals: list[float] = []
    for idx, row in enumerate(rows):
        actual = _coerce_float(row.get(treatment_id, ""))
        prediction = predictions.get(idx)
        if actual is None or prediction is None:
            row[shock_id] = ""
            row[f"{shock_id}_predicted"] = ""
            continue
        residual = actual - prediction
        row[shock_id] = stable(residual)
        row[f"{shock_id}_predicted"] = stable(prediction)
        actuals.append(actual)
        residuals.append(residual)
    mean_actual = fmean(actuals) if actuals else 0.0
    total_ss = sum((actual - mean_actual) ** 2 for actual in actuals)
    residual_ss = sum(value**2 for value in residuals)
    shock_mean = fmean(residuals) if residuals else None
    shock_std = (sum((value - (shock_mean or 0.0)) ** 2 for value in residuals) / max(len(residuals) - 1, 1)) ** 0.5 if residuals else None
    return {
        "shock_id": shock_id,
        "n": len(residuals),
        "predictor_count": len(predictors),
        "crossfit_folds": folds,
        "crossfit_r2": None if total_ss <= 0 else 1.0 - residual_ss / total_ss,
        "shock_mean": shock_mean,
        "shock_std": shock_std,
        "shock_abs_max": max((abs(value) for value in residuals), default=None),
        "dropped_predictors_by_fold": dropped_by_fold,
    }


def estimate_lp(
    rows: list[dict[str, str]],
    *,
    run_id: str,
    treatment_id: str,
    control_ids: list[str],
    outcome_ids: list[str],
) -> list[dict[str, str]]:
    estimates: list[dict[str, str]] = []
    for outcome_id in outcome_ids:
        for horizon in HORIZONS:
            sample_rows: list[tuple[float, float, list[float]]] = []
            for idx, row in enumerate(rows):
                treatment_value = _coerce_float(row.get(treatment_id, ""))
                if treatment_value is None:
                    continue
                controls: list[float] = []
                controls_ok = True
                for control_id in control_ids:
                    value = _coerce_float(row.get(control_id, ""))
                    if value is None:
                        controls_ok = False
                        break
                    controls.append(value)
                if not controls_ok:
                    continue
                target = _build_quarterly_target(
                    rows,
                    start_idx=idx,
                    outcome_id=outcome_id,
                    horizon=horizon,
                    response_type="direct_at_h",
                )
                if target is None:
                    continue
                sample_rows.append((target, treatment_value, controls))
            if len(sample_rows) <= 2:
                continue
            active_controls: list[str] = []
            active_positions: list[int] = []
            current_x = [[1.0, treatment] for _, treatment, _ in sample_rows]
            current_rank = matrix_rank(current_x)
            for position, control_id in enumerate(control_ids):
                candidate_x = [row + [controls[position]] for row, (_, _, controls) in zip(current_x, sample_rows)]
                candidate_rank = matrix_rank(candidate_x)
                if candidate_rank > current_rank and len(sample_rows) > len(active_controls) + 3:
                    active_controls.append(control_id)
                    active_positions.append(position)
                    current_x = candidate_x
                    current_rank = candidate_rank
            dropped_controls = [
                control_id
                for position, control_id in enumerate(control_ids)
                if position not in set(active_positions)
            ]
            y_values = [target for target, _, _ in sample_rows]
            x_rows = [
                [1.0, treatment_value, *[controls[position] for position in active_positions]]
                for _, treatment_value, controls in sample_rows
            ]
            try:
                fit = _ols(
                    y_values,
                    x_rows,
                    covariance_estimator="newey_west",
                    covariance_lags=max(horizon, 1),
                )
            except ValueError:
                continue
            if not y_values:
                continue
            estimates.append(
                _estimate_row_payload(
                    job_id=run_id,
                    outcome_id=outcome_id,
                    horizon=horizon,
                    treatment_id=treatment_id,
                    control_ids_used=active_controls,
                    response_type="direct_at_h",
                    inference_method="forecast_residual_innovation_lp_newey_west",
                    fit=fit,
                    dropped_control_ids=dropped_controls,
                )
            )
    return estimates


def display_multiplier_map(path: Path) -> dict[str, tuple[float, str]]:
    mapping: dict[str, tuple[float, str]] = {}
    if not path.exists():
        return mapping
    for row in read_csv(path):
        if str(row.get("run", "")) != "k100" or str(row.get("horizon", "")) != "0":
            continue
        beta = _coerce_float(row.get("beta", ""))
        effect = _coerce_float(row.get("deck_effect", ""))
        if beta is None or effect is None or abs(beta) <= 1e-15:
            continue
        mapping[str(row.get("outcome", ""))] = (effect / beta, str(row.get("deck_unit", "")))
    mapping.setdefault("dgs1", (10_000_000.0, "bp per +$100B"))
    return mapping


def deck_rows(
    estimates: list[dict[str, str]],
    *,
    run_id: str,
    multiplier_map: dict[str, tuple[float, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in estimates:
        outcome = str(row.get("outcome", ""))
        beta = _coerce_float(row.get("beta", ""))
        se = _coerce_float(row.get("se", ""))
        lower = _coerce_float(row.get("lower95", ""))
        upper = _coerce_float(row.get("upper95", ""))
        multiplier, unit = multiplier_map.get(outcome, (100.0, "$B per +$100B"))
        rows.append(
            {
                "run": run_id,
                "label": OUTCOME_LABELS.get(outcome, outcome),
                "outcome": outcome,
                "horizon": row.get("horizon", ""),
                "beta": row.get("beta", ""),
                "se": row.get("se", ""),
                "p": row.get("p_value_normal", ""),
                "n": row.get("n", ""),
                "deck_unit": unit,
                "deck_effect": "" if beta is None else stable(beta * multiplier),
                "deck_lower95": "" if lower is None else stable(lower * multiplier),
                "deck_upper95": "" if upper is None else stable(upper * multiplier),
                "controls": len(str(row.get("control_ids_used", "")).split(",")) if row.get("control_ids_used") else 0,
            }
        )
    return rows


def main() -> None:
    paths = project_paths(Path.cwd())
    manifest_path = paths.manifests / f"{JOB_ID}__design_manifest.json"
    design_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    factor_rows, factor_ids, screened_count, actual_factor_count = _load_factor_branch(
        paths,
        job_id=JOB_ID,
        design_manifest=design_manifest,
        k_screened=100,
        factor_count=4,
        control_policy_mode=DEFAULT_CONTROL_POLICY_MODE,
        min_coverage=MIN_COVERAGE,
    )
    factor_rows.sort(key=lambda row: _quarter_sort_key(str(row.get("quarter", ""))))
    add_quarterly_average_from_raw(
        factor_rows,
        raw_path=paths.seed / "interpol" / "raw" / "FRED_DGS1_DGS1.csv",
        output_id="dgs1",
    )

    base_controls = [str(item) for item in design_manifest.get("control_ids", [])]
    lag_source_cols = [TREATMENT_ID, *base_controls, *factor_ids]
    lagged_cols = add_lags(factor_rows, lag_source_cols)
    imputed_lagged_cols = impute_columns(factor_rows, lagged_cols)
    cycle_cols = select_cycle_controls(factor_rows)
    imputed_cycle_cols = impute_columns(factor_rows, cycle_cols)

    lagged_treatment = f"imp_{TREATMENT_ID}_lag1"
    lagged_core = [f"imp_{control}_lag1" for control in base_controls]
    lagged_factors = [f"imp_{factor_id}_lag1" for factor_id in factor_ids]
    factor_predictors = [lagged_treatment, *lagged_core, *lagged_factors]
    cycle_predictors = [*factor_predictors, *imputed_cycle_cols]

    specs = [
        ShockSpec(
            run_id="innovation_factor_xfit",
            shock_id=f"{SHOCK_PREFIX}_factor_xfit",
            first_stage_predictors=factor_predictors,
            lp_controls=factor_predictors,
            first_stage_note="5-fold forecast residual using lagged TDC, lagged core controls, and lagged K=100 factors.",
        ),
        ShockSpec(
            run_id="innovation_factor_xfit_cycle_controls",
            shock_id=f"{SHOCK_PREFIX}_factor_xfit",
            first_stage_predictors=factor_predictors,
            lp_controls=[*factor_predictors, *imputed_cycle_cols],
            first_stage_note="Same factor innovation shock; outcome LP also controls for lagged cycle/risk variables.",
        ),
        ShockSpec(
            run_id="innovation_cycle_risk_xfit",
            shock_id=f"{SHOCK_PREFIX}_cycle_risk_xfit",
            first_stage_predictors=cycle_predictors,
            lp_controls=cycle_predictors,
            first_stage_note="5-fold forecast residual using lagged TDC, lagged core/factor controls, and lagged cycle/risk variables.",
        ),
    ]

    shock_diagnostics: list[dict[str, Any]] = []
    for spec in {item.shock_id: item for item in specs}.values():
        diagnostics = fit_crossfit_shock(
            factor_rows,
            treatment_id=TREATMENT_ID,
            predictors=spec.first_stage_predictors,
            shock_id=spec.shock_id,
        )
        diagnostics["run_id"] = spec.run_id
        diagnostics["first_stage_note"] = spec.first_stage_note
        diagnostics["screened_feature_count"] = screened_count
        diagnostics["factor_count"] = actual_factor_count
        shock_diagnostics.append(diagnostics)

    all_estimates: list[dict[str, str]] = []
    all_deck_rows: list[dict[str, Any]] = []
    multiplier_map = display_multiplier_map(paths.reports / f"{JOB_ID}__deck_replacement_readout.csv")
    for spec in specs:
        estimates = estimate_lp(
            factor_rows,
            run_id=spec.run_id,
            treatment_id=spec.shock_id,
            control_ids=spec.lp_controls,
            outcome_ids=REQUESTED_OUTCOMES,
        )
        all_estimates.extend(estimates)
        all_deck_rows.extend(deck_rows(estimates, run_id=spec.run_id, multiplier_map=multiplier_map))

    estimates_path = paths.output / "models" / f"{JOB_ID}__innovation_shock_estimates.csv"
    deck_path = paths.reports / f"{JOB_ID}__innovation_shock_deck_readout.csv"
    shock_path = paths.reports / f"{JOB_ID}__innovation_shock_diagnostics.csv"
    summary_path = paths.manifests / f"{JOB_ID}__innovation_shock_summary.json"
    _write_estimates_csv(estimates_path, all_estimates)
    write_csv(
        deck_path,
        all_deck_rows,
        [
            "run",
            "label",
            "outcome",
            "horizon",
            "beta",
            "se",
            "p",
            "n",
            "deck_unit",
            "deck_effect",
            "deck_lower95",
            "deck_upper95",
            "controls",
        ],
    )
    write_csv(
        shock_path,
        shock_diagnostics,
        [
            "run_id",
            "shock_id",
            "n",
            "predictor_count",
            "crossfit_folds",
            "crossfit_r2",
            "shock_mean",
            "shock_std",
            "shock_abs_max",
            "screened_feature_count",
            "factor_count",
            "first_stage_note",
            "dropped_predictors_by_fold",
        ],
    )
    write_json(
        summary_path,
        {
            "generated_at": utc_now_iso(),
            "job_id": JOB_ID,
            "treatment_id": TREATMENT_ID,
            "estimates_path": str(estimates_path),
            "deck_readout_path": str(deck_path),
            "shock_diagnostics_path": str(shock_path),
            "runs": [spec.run_id for spec in specs],
            "outcomes": REQUESTED_OUTCOMES,
            "notes": "Lagged-information innovation analysis. Effects in deck readout are scaled to the same per +$100B display units used by the canonical slide readout.",
        },
    )
    print(json.dumps({"deck_readout_path": str(deck_path), "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
