from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ea_tdc.designs.quarterly import _load_jobs, _quarter_sort_key
from ea_tdc.estimation import (
    _build_quarterly_target,
    _coerce_float,
    _compute_rsquared,
    _estimate_row_payload,
    _ols,
    _write_estimates_csv,
)
from ea_tdc.paths import ProjectPaths
from ea_tdc.robustness import (
    DEFAULT_CONTROL_POLICY_MODE,
    MIN_COVERAGE,
    _apply_control_policy,
    _base_controls_from_design,
    _extract_factor_controls,
    _load_alt_treatments,
    _merge_control_rows,
    _pearson,
    _read_csv,
    _screen_features,
    build_control_universe,
)
from ea_tdc.utils import utc_now_iso, write_json


@dataclass(frozen=True)
class QuarterlyFWLResult:
    estimates_path: Path
    diagnostics_path: Path
    summary_path: Path
    rows_written: int


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fit_values(y_values: list[float], x_rows: list[list[float]]) -> list[float]:
    fit = _ols(y_values, x_rows)
    return fit.fitted


def _residualize(values: list[float], controls: list[list[float]]) -> tuple[list[float], float | None]:
    x_rows = [[1.0, *row] for row in controls]
    fitted = _fit_values(values, x_rows)
    residuals = [actual - fitted_value for actual, fitted_value in zip(values, fitted)]
    return residuals, _compute_rsquared(values, fitted)


def _max_abs_control_corr(residuals: list[float], controls: list[list[float]], control_ids: list[str]) -> tuple[float, str]:
    max_corr = 0.0
    max_control = ""
    for idx, control_id in enumerate(control_ids):
        corr = abs(_pearson(residuals, [row[idx] for row in controls]) or 0.0)
        if corr > max_corr:
            max_corr = corr
            max_control = control_id
    return max_corr, max_control


def _load_factor_branch(
    paths: ProjectPaths,
    *,
    job_id: str,
    design_manifest: dict[str, Any],
    k_screened: int,
    factor_count: int,
    control_policy_mode: str,
    min_coverage: float,
) -> tuple[list[dict[str, str]], list[str], int, int]:
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
        for (alt_treatment_id, alt_quarter), value in alt_treatment_map.items():
            if alt_quarter == quarter:
                row.setdefault(alt_treatment_id, value)

    treatment_id = str(design_manifest.get("treatment_id", "")).strip()
    outcome_ids = [str(item) for item in design_manifest.get("outcome_ids", [])]
    eligible_feature_ids, _ = _apply_control_policy(
        candidate_ids=universe_feature_ids,
        treatment_id=treatment_id,
        outcome_ids=outcome_ids,
        mode=control_policy_mode,
    )
    screened = _screen_features(
        rows=merged_rows,
        candidate_ids=eligible_feature_ids,
        treatment_id=treatment_id,
        outcome_ids=outcome_ids,
        min_coverage=min_coverage,
    )
    selected = [item["feature_id"] for item in screened[:k_screened]]
    factor_ids, factor_rows, _, _ = _extract_factor_controls(
        rows=merged_rows,
        feature_ids=selected,
        prefix=f"dflmx_k{k_screened}",
        n_factors=factor_count,
    )
    return factor_rows if factor_rows else merged_rows, [*_base_controls_from_design(design_manifest), *factor_ids], len(screened), len(factor_ids)


def build_quarterly_fwl_audit(
    paths: ProjectPaths,
    *,
    job_id: str,
    k_screened: int = 100,
    factor_count: int = 4,
    control_policy_mode: str = DEFAULT_CONTROL_POLICY_MODE,
    min_coverage: float = MIN_COVERAGE,
) -> QuarterlyFWLResult:
    job_map = _load_jobs(paths.config / "dass_job_blueprint.yaml")
    if job_id not in job_map:
        raise KeyError(f"Unknown job_id: {job_id}")
    job = job_map[job_id]
    if str(job.get("estimator", "")).strip() != "lp":
        raise ValueError("Quarterly FWL audit currently supports lp jobs only")

    design_manifest_path = paths.manifests / f"{job_id}__design_manifest.json"
    if not design_manifest_path.exists():
        raise FileNotFoundError(f"Missing design manifest for job '{job_id}'")
    design_manifest = json.loads(design_manifest_path.read_text(encoding="utf-8"))
    if str(design_manifest.get("status", "")).strip() != "ready_for_estimation":
        raise ValueError(f"Design manifest for job '{job_id}' is not ready_for_estimation")

    factor_rows, control_ids, screened_count, actual_factor_count = _load_factor_branch(
        paths,
        job_id=job_id,
        design_manifest=design_manifest,
        k_screened=k_screened,
        factor_count=factor_count,
        control_policy_mode=control_policy_mode,
        min_coverage=min_coverage,
    )
    treatment_id = str(design_manifest.get("treatment_id", "")).strip()
    outcome_ids = [str(item) for item in design_manifest.get("outcome_ids", [])]
    horizons = [int(item) for item in design_manifest.get("horizon_grid", [])]
    response_type = str(job.get("response_type", "direct_at_h")).strip()

    result_rows: list[dict[str, str]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for outcome_id in outcome_ids:
        for horizon in horizons:
            y_values: list[float] = []
            treatment_values: list[float] = []
            controls_matrix: list[list[float]] = []
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
                controls_matrix.append(controls)

            if len(y_values) <= len(control_ids) + 2:
                continue

            full_fit = _ols(
                y_values,
                [[1.0, treatment_value, *controls] for treatment_value, controls in zip(treatment_values, controls_matrix)],
                covariance_estimator="newey_west",
                covariance_lags=max(horizon, 1),
            )
            y_resid, y_control_r2 = _residualize(y_values, controls_matrix)
            t_resid, t_control_r2 = _residualize(treatment_values, controls_matrix)
            fwl_fit = _ols(
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
                inference_method="fwl_residualized_tdc_newey_west",
                fit=fwl_fit,
            )
            controlled_beta = full_fit.beta[1]
            controlled_se = full_fit.ses[1]
            payload["controlled_lp_beta"] = str(controlled_beta)
            payload["controlled_lp_se"] = str(controlled_se)
            payload["beta_abs_diff_vs_controlled_lp"] = str(abs(float(payload["beta"]) - controlled_beta))
            payload["se_abs_diff_vs_controlled_lp"] = str(abs(float(payload["se"]) - controlled_se))
            result_rows.append(payload)

            max_corr, max_control = _max_abs_control_corr(t_resid, controls_matrix, control_ids)
            residual_mean = sum(t_resid) / len(t_resid)
            residual_ss = sum((value - residual_mean) ** 2 for value in t_resid)
            residual_std = (residual_ss / max(len(t_resid) - 1, 1)) ** 0.5
            diagnostic_rows.append(
                {
                    "job_id": job_id,
                    "outcome": outcome_id,
                    "horizon": horizon,
                    "n": len(y_values),
                    "treatment_id": treatment_id,
                    "k_screened": k_screened,
                    "factor_count": actual_factor_count,
                    "control_count": len(control_ids),
                    "treatment_control_r2": "" if t_control_r2 is None else round(t_control_r2, 8),
                    "outcome_control_r2": "" if y_control_r2 is None else round(y_control_r2, 8),
                    "tdc_residual_mean": residual_mean,
                    "tdc_residual_std": residual_std,
                    "tdc_residual_max_abs_control_corr": max_corr,
                    "tdc_residual_max_abs_control_corr_id": max_control,
                    "controlled_lp_beta": controlled_beta,
                    "fwl_beta": float(payload["beta"]),
                    "beta_abs_diff_vs_controlled_lp": float(payload["beta_abs_diff_vs_controlled_lp"]),
                }
            )

    estimates_path = paths.output / "models" / f"{job_id}__fwl_k{k_screened}_estimates.csv"
    diagnostics_path = paths.reports / f"{job_id}__fwl_k{k_screened}_diagnostics.csv"
    summary_path = paths.manifests / f"{job_id}__fwl_k{k_screened}_summary.json"
    estimates_path.parent.mkdir(parents=True, exist_ok=True)
    _write_estimates_csv(estimates_path, result_rows)
    _write_csv(
        diagnostics_path,
        diagnostic_rows,
        fieldnames=[
            "job_id",
            "outcome",
            "horizon",
            "n",
            "treatment_id",
            "k_screened",
            "factor_count",
            "control_count",
            "treatment_control_r2",
            "outcome_control_r2",
            "tdc_residual_mean",
            "tdc_residual_std",
            "tdc_residual_max_abs_control_corr",
            "tdc_residual_max_abs_control_corr_id",
            "controlled_lp_beta",
            "fwl_beta",
            "beta_abs_diff_vs_controlled_lp",
        ],
    )
    max_beta_diff = max(
        (float(row["beta_abs_diff_vs_controlled_lp"]) for row in result_rows),
        default=None,
    )
    max_control_corr = max(
        (float(row["tdc_residual_max_abs_control_corr"]) for row in diagnostic_rows),
        default=None,
    )
    write_json(
        summary_path,
        {
            "job_id": job_id,
            "generated_at": utc_now_iso(),
            "estimates_path": str(estimates_path),
            "diagnostics_path": str(diagnostics_path),
            "rows_written": len(result_rows),
            "k_screened": k_screened,
            "screened_feature_count": screened_count,
            "factor_count": actual_factor_count,
            "control_policy_mode": control_policy_mode,
            "control_ids": control_ids,
            "max_beta_abs_diff_vs_controlled_lp": max_beta_diff,
            "max_tdc_residual_abs_control_corr": max_control_corr,
            "notes": "FWL audit for the factor-augmented LP: residualize both TDC and each horizon target on the same controls, then regress residualized target on residualized TDC. Coefficients should match the controlled LP on the same sample.",
        },
    )
    return QuarterlyFWLResult(
        estimates_path=estimates_path,
        diagnostics_path=diagnostics_path,
        summary_path=summary_path,
        rows_written=len(result_rows),
    )
