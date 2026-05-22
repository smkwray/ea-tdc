from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ea_tdc.designs.quarterly import build_quarterly_design
from ea_tdc.estimation import _build_quarterly_target, _coerce_float, _estimate_row_payload, _ols
from ea_tdc.paths import project_paths
from ea_tdc.residualized_shock import _load_factor_branch
from ea_tdc.utils import utc_now_iso, write_json
from run_pinned_factor_residual_bridge import (
    ANCHOR_JOB_ID,
    CONTROL_POLICY_MODE,
    FACTOR_COUNT,
    K_SCREENED,
    MERGE_JOBS,
    _load_manifest,
    _merge_by_quarter,
)
from run_tier2_state_dependent_credit_causality import (
    BRIDGE_TREATMENTS,
    LEADS,
    _active_controls,
    _decorate_estimate,
    _format_value,
    _add_future_treatment,
)


PRIMARY_TREATMENT_LABEL = "regression_mmf_rrp_bank_long"
TREATMENT_LABELS = [
    PRIMARY_TREATMENT_LABEL,
]
HORIZONS = [0, 4]
FOCUS_OUTCOMES = [
    "matched_total_deposits",
    "tdcpass_strict_loan_core_min_qoq",
    "tdcpass_strict_loan_mortgages_qoq",
    "tdcpass_strict_loan_consumer_credit_qoq",
    "bank_credit_qoq",
]
CREDIT_PLACEBO_OUTCOMES = [
    "tdcpass_strict_loan_core_min_qoq",
    "tdcpass_strict_loan_mortgages_qoq",
    "tdcpass_strict_loan_consumer_credit_qoq",
]
LAG_PERIODS = [1, 2, 4]
MAX_FACTOR_TAIL_CONTROLS = 12
CREDIT_LAG_SOURCES = [
    "tdcpass_strict_loan_core_min_qoq",
    "tdcpass_strict_loan_mortgages_qoq",
    "tdcpass_strict_loan_consumer_credit_qoq",
    "bank_credit_qoq",
]
RATE_RISK_LAG_SOURCES = [
    "dgs2",
    "dgs10",
    "mortgage_30y_dgs10_spread",
    "baa_aaa",
    "BAMLH0A0HYM2",
]

SELECTED_CREDIT_LAG_PATTERNS = {
    "tdcpass_strict_loan_consumer_credit_qoq__lag_4",
    "tdcpass_strict_loan_core_min_qoq__lag_4",
    "bank_credit_qoq__lag_4",
    "tdcpass_strict_loan_core_min_qoq__lag_2",
}
SELECTED_RATE_RISK_LAG_PATTERNS = {
    "dgs10__lag_2",
    "dgs10__lag_1",
    "dgs2__lag_4",
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _normal_p(z_score: float) -> float:
    return math.erfc(abs(z_score) / math.sqrt(2.0))


def _add_lags(rows: list[dict[str, str]], columns: list[str], lags: list[int]) -> list[str]:
    lagged_columns: list[str] = []
    for column in columns:
        if not any(column in row for row in rows):
            continue
        for lag in lags:
            lagged_column = f"{column}__lag_{lag}"
            lagged_columns.append(lagged_column)
            for idx, row in enumerate(rows):
                source_idx = idx - lag
                row[lagged_column] = rows[source_idx].get(column, "") if source_idx >= 0 else ""
    return lagged_columns


def _rank_incremental_controls(
    rows: list[dict[str, str]],
    *,
    treatment_id: str,
    base_controls: list[str],
    candidates: list[str],
    max_added: int = 12,
) -> list[dict[str, Any]]:
    base_sample = _predictability_sample(rows, treatment_id=treatment_id, control_ids=base_controls)
    base_r2 = _fit_predictability(base_sample, base_controls).get("rsquared")
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        controls = [*base_controls, candidate]
        sample = _predictability_sample(rows, treatment_id=treatment_id, control_ids=controls)
        if len(sample) <= len(controls) + 2:
            continue
        fit = _fit_predictability(sample, controls)
        output.append(
            {
                "candidate_control": candidate,
                "n": fit.get("n", ""),
                "base_r2": base_r2 if base_r2 is not None else "",
                "candidate_r2": fit.get("rsquared", ""),
                "incremental_r2": "" if base_r2 is None or fit.get("rsquared") == "" else float(fit["rsquared"]) - float(base_r2),
                "candidate_beta": fit.get(f"beta__{candidate}", ""),
                "candidate_p_value": fit.get(f"p_value__{candidate}", ""),
            }
        )
    return sorted(output, key=lambda row: abs(float(row["incremental_r2"] or 0.0)), reverse=True)[:max_added]


def _predictability_sample(
    rows: list[dict[str, str]],
    *,
    treatment_id: str,
    control_ids: list[str],
) -> list[tuple[float, list[float]]]:
    sample: list[tuple[float, list[float]]] = []
    for row in rows:
        treatment_value = _coerce_float(row.get(treatment_id, ""))
        if treatment_value is None:
            continue
        controls: list[float] = []
        ok = True
        for control_id in control_ids:
            value = _coerce_float(row.get(control_id, ""))
            if value is None:
                ok = False
                break
            controls.append(value)
        if ok:
            sample.append((treatment_value, controls))
    return sample


def _fit_predictability(sample: list[tuple[float, list[float]]], control_ids: list[str]) -> dict[str, Any]:
    y_values = [item[0] for item in sample]
    selected_controls: list[str] = []
    dropped_controls: list[str] = []
    for control_id in control_ids:
        trial_controls = [*selected_controls, control_id]
        positions = [control_ids.index(item) for item in trial_controls]
        trial_x_rows = [[1.0, *[item[1][position] for position in positions]] for item in sample]
        if len(trial_x_rows) <= len(trial_controls) + 4:
            dropped_controls.append(control_id)
            continue
        if _matrix_rank(trial_x_rows) < len(trial_controls) + 1:
            dropped_controls.append(control_id)
            continue
        try:
            _ols(y_values, trial_x_rows, covariance_estimator="newey_west", covariance_lags=1)
        except ValueError:
            dropped_controls.append(control_id)
            continue
        selected_controls.append(control_id)
    if not selected_controls:
        return {"n": len(sample), "rsquared": "", "warning": "insufficient_rank"}
    positions = [control_ids.index(control_id) for control_id in selected_controls]
    x_rows = [[1.0, *[item[1][position] for position in positions]] for item in sample]
    try:
        fit = _ols(y_values, x_rows, covariance_estimator="newey_west", covariance_lags=1)
    except ValueError:
        return {"n": len(sample), "rsquared": "", "warning": "singular"}
    payload: dict[str, Any] = {
        "n": len(sample),
        "rsquared": "" if fit.rsquared is None else fit.rsquared,
        "controls_used": ",".join(selected_controls),
        "dropped_control_ids": ",".join(dropped_controls),
    }
    for idx, control_id in enumerate(selected_controls, start=1):
        beta = fit.beta[idx]
        se = fit.ses[idx]
        payload[f"beta__{control_id}"] = beta
        payload[f"se__{control_id}"] = se
        payload[f"p_value__{control_id}"] = "" if se <= 0 else _normal_p(beta / se)
    return payload


def _predictability_rows(
    rows: list[dict[str, str]],
    *,
    treatment_label: str,
    treatment_id: str,
    scenarios: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    increment_rows: list[dict[str, Any]] = []
    base_r2: float | None = None
    for scenario, controls in scenarios.items():
        sample = _predictability_sample(rows, treatment_id=treatment_id, control_ids=controls)
        fit = _fit_predictability(sample, controls)
        rsquared = fit.get("rsquared", "")
        if scenario == "baseline_controls" and rsquared != "":
            base_r2 = float(rsquared)
        summary_rows.append(
            {
                "treatment_label": treatment_label,
                "treatment_id": treatment_id,
                "scenario": scenario,
                "n": fit.get("n", ""),
                "control_count": len([item for item in str(fit.get("controls_used", "")).split(",") if item]),
                "requested_control_count": len(controls),
                "rsquared": rsquared,
                "delta_r2_vs_baseline": "" if base_r2 is None or rsquared == "" else float(rsquared) - base_r2,
                "dropped_control_ids": fit.get("dropped_control_ids", ""),
            }
        )
        if scenario != "baseline_controls":
            candidates = [control for control in controls if control not in scenarios["baseline_controls"]]
            for row in _rank_incremental_controls(
                rows,
                treatment_id=treatment_id,
                base_controls=scenarios["baseline_controls"],
                candidates=candidates,
            ):
                row["treatment_label"] = treatment_label
                row["treatment_id"] = treatment_id
                row["scenario"] = scenario
                increment_rows.append(row)
    return summary_rows, increment_rows


def _lead_count(rows: list[dict[str, Any]], *, scenario: str, treatment_label: str) -> int:
    return sum(
        1
        for row in rows
        if row.get("diagnostic_scenario") == scenario
        and row.get("surface") == "lead_placebo"
        and row.get("treatment_label") == treatment_label
        and row.get("outcome") in CREDIT_PLACEBO_OUTCOMES
        and str(row.get("p_value_normal", "")).strip()
        and float(row["p_value_normal"]) < 0.1
    )


def _estimate_rows_for_scenarios(
    rows: list[dict[str, str]],
    *,
    treatment_specs: dict[str, dict[str, Any]],
    control_scenarios: dict[str, dict[str, list[str]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for treatment_label, treatment_spec in treatment_specs.items():
        for scenario, controls_by_treatment in control_scenarios.items():
            controls = controls_by_treatment[treatment_label]
            estimates = _estimate_rank_aware_surface(
                rows=rows,
                treatment_label=treatment_label,
                treatment_spec=treatment_spec,
                control_ids=controls,
                outcome_ids=FOCUS_OUTCOMES,
                horizons=HORIZONS,
                surface="pooled_lag_diagnostic",
                sample_label="full_available",
            )
            for estimate in estimates:
                estimate["diagnostic_scenario"] = scenario
            output.extend(estimates)

            lead_estimates = _lead_placebo_rank_aware_estimates(
                rows=rows,
                treatment_label=treatment_label,
                treatment_spec=treatment_spec,
                control_ids=controls,
                outcome_ids=[*CREDIT_PLACEBO_OUTCOMES, "bank_credit_qoq"],
            )
            for estimate in lead_estimates:
                estimate["diagnostic_scenario"] = scenario
            output.extend(lead_estimates)
    return output


def _matrix_rank(matrix: list[list[float]], *, tol: float = 1e-10) -> int:
    if not matrix:
        return 0
    work = [row[:] for row in matrix]
    row_count = len(work)
    col_count = len(work[0])
    rank = 0
    for col_idx in range(col_count):
        pivot_idx = max(range(rank, row_count), key=lambda idx: abs(work[idx][col_idx]))
        pivot = work[pivot_idx][col_idx]
        if abs(pivot) <= tol:
            continue
        work[rank], work[pivot_idx] = work[pivot_idx], work[rank]
        pivot = work[rank][col_idx]
        work[rank] = [value / pivot for value in work[rank]]
        for row_idx in range(row_count):
            if row_idx == rank:
                continue
            factor = work[row_idx][col_idx]
            if abs(factor) <= tol:
                continue
            work[row_idx] = [
                current - factor * pivot_value
                for current, pivot_value in zip(work[row_idx], work[rank])
            ]
        rank += 1
        if rank == row_count:
            break
    return rank


def _lp_sample(
    rows: list[dict[str, str]],
    *,
    treatment_id: str,
    control_ids: list[str],
    outcome_id: str,
    horizon: int,
) -> tuple[list[float], list[list[float]]]:
    y_values: list[float] = []
    x_rows: list[list[float]] = []
    for idx, row in enumerate(rows):
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
            rows,
            start_idx=idx,
            outcome_id=outcome_id,
            horizon=horizon,
            response_type="direct_at_h",
        )
        if target_value is None:
            continue
        y_values.append(target_value)
        x_rows.append([1.0, treatment_value, *controls])
    return y_values, x_rows


def _rank_aware_lp_estimates(
    *,
    rows: list[dict[str, str]],
    treatment_id: str,
    control_ids: list[str],
    outcome_ids: list[str],
    horizons: list[int],
    job_id: str,
) -> list[dict[str, str]]:
    result_rows: list[dict[str, str]] = []
    for outcome_id in outcome_ids:
        for horizon in horizons:
            selected_controls: list[str] = []
            rejected_controls: list[str] = []
            for control_id in control_ids:
                trial_controls = [*selected_controls, control_id]
                y_values, x_rows = _lp_sample(
                    rows,
                    treatment_id=treatment_id,
                    control_ids=trial_controls,
                    outcome_id=outcome_id,
                    horizon=horizon,
                )
                required_rank = len(trial_controls) + 2
                if len(y_values) <= required_rank + 3 or _matrix_rank(x_rows) < required_rank:
                    rejected_controls.append(control_id)
                    continue
                try:
                    _ols(
                        y_values,
                        x_rows,
                        covariance_estimator="newey_west",
                        covariance_lags=max(horizon, 1),
                    )
                except ValueError:
                    rejected_controls.append(control_id)
                    continue
                selected_controls.append(control_id)

            y_values, x_rows = _lp_sample(
                rows,
                treatment_id=treatment_id,
                control_ids=selected_controls,
                outcome_id=outcome_id,
                horizon=horizon,
            )
            required_rank = len(selected_controls) + 2
            if len(y_values) <= required_rank + 3 or _matrix_rank(x_rows) < required_rank:
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
            warning_flags = ["rank_aware_controls"]
            if rejected_controls:
                warning_flags.append("rank_rejected_controls")
            result_rows.append(
                _estimate_row_payload(
                    job_id=job_id,
                    outcome_id=outcome_id,
                    horizon=horizon,
                    treatment_id=treatment_id,
                    control_ids_used=selected_controls,
                    response_type="direct_at_h",
                    inference_method=(
                        "ols_newey_west_scaffold_rank_aware"
                        if not rejected_controls
                        else "ols_newey_west_scaffold_rank_aware_controls"
                    ),
                    fit=fit,
                    warning_flags=warning_flags,
                    dropped_control_ids=rejected_controls,
                )
            )
    return result_rows


def _estimate_rank_aware_surface(
    *,
    rows: list[dict[str, str]],
    treatment_label: str,
    treatment_spec: dict[str, Any],
    control_ids: list[str],
    outcome_ids: list[str],
    horizons: list[int],
    surface: str,
    sample_label: str = "",
) -> list[dict[str, Any]]:
    estimates = _rank_aware_lp_estimates(
        rows=rows,
        treatment_id=str(treatment_spec["treatment_id"]),
        control_ids=control_ids,
        outcome_ids=outcome_ids,
        horizons=horizons,
        job_id=f"tier2_credit_lead_diagnostic_{surface}_{treatment_label}_{sample_label or 'full'}",
    )
    return [
        _decorate_estimate(
            row,
            surface=surface,
            treatment_label=treatment_label,
            sample_label=sample_label,
        )
        for row in estimates
    ]


def _lead_placebo_rank_aware_estimates(
    *,
    rows: list[dict[str, str]],
    treatment_label: str,
    treatment_spec: dict[str, Any],
    control_ids: list[str],
    outcome_ids: list[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    treatment_id = str(treatment_spec["treatment_id"])
    for lead in LEADS:
        lead_rows, lead_column = _add_future_treatment(rows, treatment_id, lead)
        estimates = _estimate_rank_aware_surface(
            rows=lead_rows,
            treatment_label=treatment_label,
            treatment_spec={"treatment_id": lead_column},
            control_ids=control_ids,
            outcome_ids=outcome_ids,
            horizons=[0],
            surface="lead_placebo",
            sample_label=f"lead_{lead}",
        )
        for estimate in estimates:
            estimate["lead_quarters"] = lead
            estimate["future_treatment_id"] = lead_column
            estimate["actual_treatment_id"] = treatment_id
        output.extend(estimates)
    return output


def _insert_lags_before_factor_tail(base_controls: list[str], lag_controls: list[str]) -> list[str]:
    prefix: list[str] = []
    factor_tail: list[str] = []
    for control_id in base_controls:
        if control_id.startswith("dflmx_") or control_id.startswith("imp_dflmx_"):
            factor_tail.append(control_id)
        else:
            prefix.append(control_id)
    return [*prefix, *lag_controls, *factor_tail]


def _cap_factor_tail(controls: list[str], max_factor_tail: int = MAX_FACTOR_TAIL_CONTROLS) -> list[str]:
    prefix: list[str] = []
    factor_tail: list[str] = []
    for control_id in controls:
        if control_id.startswith("dflmx_") or control_id.startswith("imp_dflmx_"):
            factor_tail.append(control_id)
        else:
            prefix.append(control_id)
    return [*prefix, *factor_tail[:max_factor_tail]]


def _main_effect_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if row.get("surface") != "pooled_lag_diagnostic":
            continue
        if int(row.get("horizon", 0)) not in HORIZONS:
            continue
        if row.get("outcome") not in FOCUS_OUTCOMES:
            continue
        output.append(
            {
                "diagnostic_scenario": row.get("diagnostic_scenario", ""),
                "treatment_label": row.get("treatment_label", ""),
                "outcome": row.get("outcome", ""),
                "horizon": row.get("horizon", ""),
                "normalized_beta": row.get("normalized_beta", ""),
                "effect_per_100b_tdc": row.get("effect_per_100b_tdc", ""),
                "effect_per_100b_unit": row.get("effect_per_100b_unit", ""),
                "p_value": row.get("p_value_normal", ""),
                "n": row.get("n", ""),
                "control_ids_used": row.get("control_ids_used", ""),
                "dropped_control_ids": row.get("dropped_control_ids", ""),
                "warning_flags": row.get("warning_flags", ""),
            }
        )
    return output


def _lead_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scenario in sorted({str(row.get("diagnostic_scenario", "")) for row in rows if row.get("surface") == "lead_placebo"}):
        if not scenario:
            continue
        for treatment_label in TREATMENT_LABELS:
            lead_rows = [
                row
                for row in rows
                if row.get("surface") == "lead_placebo"
                and row.get("diagnostic_scenario") == scenario
                and row.get("treatment_label") == treatment_label
                and row.get("outcome") in CREDIT_PLACEBO_OUTCOMES
            ]
            sig_rows = [
                row
                for row in lead_rows
                if str(row.get("p_value_normal", "")).strip()
                and float(row["p_value_normal"]) < 0.1
            ]
            max_abs_effect = max((abs(float(row.get("effect_per_100b_tdc", 0.0))) for row in lead_rows), default=0.0)
            output.append(
                {
                    "diagnostic_scenario": scenario,
                    "treatment_label": treatment_label,
                    "lead_rows": len(lead_rows),
                    "significant_credit_leads_p_lt_0_10": len(sig_rows),
                    "max_abs_effect_per_100b_tdc": max_abs_effect,
                    "significant_lead_details": "; ".join(
                        f"L{row.get('lead_quarters')} {row.get('outcome')}={_format_value(row.get('effect_per_100b_tdc'))} p={_format_value(row.get('p_value_normal'))}"
                        for row in sig_rows
                    ),
                }
            )
    return output


def _write_markdown(
    path: Path,
    *,
    predictability_rows: list[dict[str, Any]],
    lead_summary_rows: list[dict[str, Any]],
    main_effect_rows: list[dict[str, Any]],
    increment_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Tier 2 Credit Lead Diagnostics",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        "This diagnostic asks whether active credit lead placebos are mostly a pre-trend / predictability problem. It compares baseline controls against selected lagged credit controls and selected lagged credit plus rate-risk controls.",
        "",
        f"Controls are admitted with a rank-aware greedy check. The factor tail is capped at {MAX_FACTOR_TAIL_CONTROLS} controls so the diagnostic tests the targeted lag block instead of spending rank on a high-dimensional factor screen.",
        "",
        "## Lead-Placebo Counts",
        "",
        "| scenario | treatment | significant credit leads | max abs lead effect |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in lead_summary_rows:
        lines.append(
            f"| {row['diagnostic_scenario']} | {row['treatment_label']} | {row['significant_credit_leads_p_lt_0_10']} | {_format_value(row['max_abs_effect_per_100b_tdc'])} |"
        )

    lines.extend(["", "## Main H=0 Effects", "", "| scenario | treatment | outcome | effect per +$100B | p | n |", "| --- | --- | --- | ---: | ---: | ---: |"])
    for row in main_effect_rows:
        if row["treatment_label"] != PRIMARY_TREATMENT_LABEL:
            continue
        if int(row["horizon"]) != 0:
            continue
        if row["outcome"] not in {"matched_total_deposits", *CREDIT_PLACEBO_OUTCOMES, "bank_credit_qoq"}:
            continue
        lines.append(
            f"| {row['diagnostic_scenario']} | {row['treatment_label']} | {row['outcome']} | {_format_value(row['effect_per_100b_tdc'])} | {_format_value(row['p_value'])} | {row['n']} |"
        )

    lines.extend(["", "## Treatment Predictability", "", "| scenario | n | controls used | requested | R2 | delta R2 |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in predictability_rows:
        if row["treatment_label"] != PRIMARY_TREATMENT_LABEL:
            continue
        lines.append(
            f"| {row['scenario']} | {row['n']} | {row['control_count']} | {row['requested_control_count']} | {_format_value(row['rsquared'])} | {_format_value(row['delta_r2_vs_baseline'])} |"
        )

    lines.extend(["", "## Strongest Incremental Lag Predictors", "", "| scenario | lag control | incremental R2 | beta | p |", "| --- | --- | ---: | ---: | ---: |"])
    for row in increment_rows[:20]:
        if row["treatment_label"] != PRIMARY_TREATMENT_LABEL:
            continue
        lines.append(
            f"| {row['scenario']} | {row['candidate_control']} | {_format_value(row['incremental_r2'])} | {_format_value(row['candidate_beta'])} | {_format_value(row['candidate_p_value'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "- If lagged credit/rate controls materially reduce significant leads while preserving main credit effects, treat the original lead issue as partly pre-trend control insufficiency.",
            "- If leads persist after these controls, keep the causal claim guarded and prioritize a stronger predetermined shock or external instrument.",
            "- If main credit effects vanish under lag controls, frame the original credit pattern as credit-cycle confounding rather than crowding out.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    paths = project_paths(ROOT)
    for job_id in MERGE_JOBS:
        build_quarterly_design(paths, job_id=job_id)

    anchor_manifest = _load_manifest(paths, ANCHOR_JOB_ID)
    factor_rows, control_ids, screened_count, factor_count = _load_factor_branch(
        paths,
        job_id=ANCHOR_JOB_ID,
        design_manifest=anchor_manifest,
        k_screened=K_SCREENED,
        factor_count=FACTOR_COUNT,
        control_policy_mode=CONTROL_POLICY_MODE,
        min_coverage=0.4,
    )
    bundle_paths = [Path(str(_load_manifest(paths, job_id).get("bundle_path", ""))) for job_id in MERGE_JOBS]
    rows = _merge_by_quarter(factor_rows, bundle_paths)
    credit_lags = _add_lags(rows, CREDIT_LAG_SOURCES, LAG_PERIODS)
    rate_risk_lags = _add_lags(rows, RATE_RISK_LAG_SOURCES, LAG_PERIODS)
    selected_credit_lags = [column for column in credit_lags if column in SELECTED_CREDIT_LAG_PATTERNS]
    selected_rate_risk_lags = [column for column in rate_risk_lags if column in SELECTED_RATE_RISK_LAG_PATTERNS]

    innovation_control_map: dict[str, list[str]] = {}
    innovation_diagnostics: list[dict[str, Any]] = []
    treatment_specs = {PRIMARY_TREATMENT_LABEL: BRIDGE_TREATMENTS[PRIMARY_TREATMENT_LABEL]}

    base_controls_by_treatment: dict[str, list[str]] = {}
    for treatment_label, treatment_spec in treatment_specs.items():
        if treatment_label in innovation_control_map:
            base_controls_by_treatment[treatment_label] = _cap_factor_tail(innovation_control_map[treatment_label])
        else:
            base_controls_by_treatment[treatment_label] = _cap_factor_tail(_active_controls(control_ids, treatment_spec))

    control_scenarios = {
        "baseline_controls": {
            label: controls[:] for label, controls in base_controls_by_treatment.items()
        },
        "selected_credit_lags": {
            label: _insert_lags_before_factor_tail(controls, selected_credit_lags)
            for label, controls in base_controls_by_treatment.items()
        },
        "selected_credit_rate_risk_lags": {
            label: _insert_lags_before_factor_tail(controls, [*selected_credit_lags, *selected_rate_risk_lags])
            for label, controls in base_controls_by_treatment.items()
        },
    }

    estimates = _estimate_rows_for_scenarios(
        rows,
        treatment_specs=treatment_specs,
        control_scenarios=control_scenarios,
    )
    estimates_path = paths.output / "models" / "tier2_credit_lead_diagnostic_estimates.csv"
    _write_csv(estimates_path, estimates)

    main_rows = _main_effect_rows(estimates)
    main_path = paths.reports / "tier2_credit_lead_diagnostic_main_effects.csv"
    _write_csv(main_path, main_rows)

    lead_rows = _lead_summary_rows(estimates)
    lead_path = paths.reports / "tier2_credit_lead_diagnostic_lead_summary.csv"
    _write_csv(lead_path, lead_rows)

    predict_rows: list[dict[str, Any]] = []
    increment_rows: list[dict[str, Any]] = []
    for treatment_label, treatment_spec in treatment_specs.items():
        scenario_controls = {
            scenario: controls_by_treatment[treatment_label]
            for scenario, controls_by_treatment in control_scenarios.items()
        }
        summary_rows, control_rows = _predictability_rows(
            rows,
            treatment_label=treatment_label,
            treatment_id=str(treatment_spec["treatment_id"]),
            scenarios=scenario_controls,
        )
        predict_rows.extend(summary_rows)
        increment_rows.extend(control_rows)
    predict_path = paths.reports / "tier2_credit_lead_diagnostic_tdc_predictability.csv"
    _write_csv(predict_path, predict_rows)
    increment_path = paths.reports / "tier2_credit_lead_diagnostic_incremental_lag_predictors.csv"
    _write_csv(increment_path, increment_rows)

    markdown_path = paths.reports / "tier2_credit_lead_diagnostic.md"
    _write_markdown(
        markdown_path,
        predictability_rows=predict_rows,
        lead_summary_rows=lead_rows,
        main_effect_rows=main_rows,
        increment_rows=increment_rows,
    )

    summary_path = paths.manifests / "tier2_credit_lead_diagnostic_summary.json"
    write_json(
        summary_path,
        {
            "generated_at": utc_now_iso(),
            "anchor_job_id": ANCHOR_JOB_ID,
            "k_screened": K_SCREENED,
            "factor_count": factor_count,
            "screened_count": screened_count,
            "control_policy_mode": CONTROL_POLICY_MODE,
            "treatment_labels": TREATMENT_LABELS,
            "innovation_diagnostics": innovation_diagnostics,
            "lag_periods": LAG_PERIODS,
            "credit_lag_sources": CREDIT_LAG_SOURCES,
            "rate_risk_lag_sources": RATE_RISK_LAG_SOURCES,
            "selected_credit_lags": selected_credit_lags,
            "selected_rate_risk_lags": selected_rate_risk_lags,
            "max_factor_tail_controls": MAX_FACTOR_TAIL_CONTROLS,
            "estimates_path": str(estimates_path),
            "main_effects_path": str(main_path),
            "lead_summary_path": str(lead_path),
            "predictability_path": str(predict_path),
            "incremental_lag_predictors_path": str(increment_path),
            "markdown_path": str(markdown_path),
            "rows_written": len(estimates),
            "lead_summary": lead_rows,
        },
    )
    print(
        json.dumps(
            {
                "markdown_path": str(markdown_path),
                "summary_path": str(summary_path),
                "rows_written": len(estimates),
                "lead_summary_rows": len(lead_rows),
                "main_effect_rows": len(main_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
