"""Regime-persistence diagnostics for Tier 2 deposit pass-through.

These diagnostics implement the first regime-persistence recommendation tranche:

* re-estimate rolling windows that contain pandemic quarters after dropping
  2020/2021 blocks; and
* compute Frisch-Waugh-Lovell quarter-level contributions to the selected-lag
  full-sample deposit and residual slopes.

The output is a persistence and influence screen, not a structural mechanism
estimate.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ea_tdc.estimation import _invert, _matmul, _matvec, _ols, _transpose  # noqa: E402
from ea_tdc.open_contract import (  # noqa: E402
    CANONICAL_OUTCOME_ID,
    CANONICAL_RESIDUAL_ID,
    CANONICAL_TREATMENT_ID,
    ROLLING_WINDOW_QUARTERS,
)
from run_submission_appendix_diagnostics import (  # noqa: E402
    _build_inputs,
    _effect_per_100b,
    _fit_lp,
    _normal_p_two_sided,
    _normalization,
    _scenario_controls,
)
from run_tier2_pass_through_offset_diagnostics import (  # noqa: E402
    _between,
    _canonical_controls,
    _quarter_key,
    _safe_float,
)


PRIMARY_TREATMENT_ID = CANONICAL_TREATMENT_ID
PRIMARY_RESIDUAL_ID = CANONICAL_RESIDUAL_ID
JOB_ID = "tier2_pass_through_regime_persistence"
WINDOW_QUARTERS = ROLLING_WINDOW_QUARTERS
MIN_OBSERVATIONS = 24
OUTCOMES = [CANONICAL_OUTCOME_ID, PRIMARY_RESIDUAL_ID]

ROLLING_MINUS_OUTPUT = ROOT / "output/reports/tier2_pass_through_rolling_minus_pandemic_betas.csv"
INFLUENCE_OUTPUT = ROOT / "output/reports/tier2_pass_through_influence_quarters.csv"
RATEWALL_SUMMARY_OUTPUT = ROOT / "output/models/tdc_deposit_pass_through_pandemic_exclusion_diagnostics.csv"
REPORT_OUTPUT = ROOT / "output/reports/tier2_pass_through_regime_persistence.md"
MANIFEST_OUTPUT = ROOT / "output/manifests/tier2_pass_through_regime_persistence_summary.json"


DROP_RULES: list[tuple[str, Callable[[str], bool], str]] = [
    ("none", lambda _quarter: False, "No pandemic-quarter drop."),
    ("drop_2020_2021", lambda quarter: _between(quarter, "2020Q1", "2021Q4"), "Drop 2020Q1-2021Q4."),
    ("drop_2020", lambda quarter: _between(quarter, "2020Q1", "2020Q4"), "Drop 2020Q1-2020Q4."),
    ("drop_2021", lambda quarter: _between(quarter, "2021Q1", "2021Q4"), "Drop 2021Q1-2021Q4."),
    ("drop_2020h1", lambda quarter: _between(quarter, "2020Q1", "2020Q2"), "Drop 2020Q1-2020Q2."),
    ("drop_2021q1", lambda quarter: quarter == "2021Q1", "Drop 2021Q1."),
]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _numeric_rows(
    rows: list[dict[str, str]],
    *,
    outcome_ids: list[str],
    control_ids: list[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    required = [PRIMARY_TREATMENT_ID, *outcome_ids, *control_ids]
    for row in rows:
        values: dict[str, float] = {}
        missing = False
        for column in required:
            value = _safe_float(row.get(column, ""))
            if value is None:
                missing = True
                break
            values[column] = value
        if missing:
            continue
        output.append({"quarter": str(row.get("quarter", "")), **values})
    return output


def _estimate_common_pair(
    rows: list[dict[str, str]],
    *,
    controls: list[str],
) -> dict[str, Any]:
    try:
        _fit, _n, used, rejected = _fit_lp(
            rows,
            treatment_id=PRIMARY_TREATMENT_ID,
            outcome_id="matched_total_deposits",
            horizon=0,
            control_ids=controls,
            covariance_lags=1,
        )
    except ValueError as exc:
        return {"status": "not_estimable", "error": str(exc)}
    numeric = _numeric_rows(rows, outcome_ids=OUTCOMES, control_ids=used)
    if len(numeric) < MIN_OBSERVATIONS:
        return {"status": "insufficient_observations", "n": len(numeric), "controls_used": ",".join(used)}
    x_rows = [[1.0, row[PRIMARY_TREATMENT_ID], *[row[control_id] for control_id in used]] for row in numeric]
    output: dict[str, Any] = {
        "status": "estimated",
        "n": len(numeric),
        "controls_used": ",".join(used),
        "controls_rejected": ",".join(rejected),
    }
    for outcome_id in OUTCOMES:
        y_values = [row[outcome_id] for row in numeric]
        try:
            fit = _ols(y_values, x_rows, covariance_estimator="newey_west", covariance_lags=1)
        except ValueError as exc:
            return {"status": "not_estimable", "error": str(exc), "controls_used": ",".join(used)}
        beta = fit.beta[1]
        se = fit.ses[1]
        z_score = beta / se if se > 0 else None
        _, multiplier = _normalization(outcome_id)
        effect_unit, effect_value = _effect_per_100b(outcome_id, beta)
        prefix = "deposit" if outcome_id == "matched_total_deposits" else "residual"
        output.update(
            {
                f"{prefix}_beta": beta * multiplier,
                f"{prefix}_se": se * multiplier,
                f"{prefix}_p": "" if z_score is None else _normal_p_two_sided(z_score),
                f"{prefix}_effect_per_100b_tdc": effect_value,
                f"{prefix}_effect_per_100b_unit": effect_unit,
                f"{prefix}_rsquared": fit.rsquared,
            }
        )
    if "deposit_beta" in output and "residual_beta" in output:
        output["deposit_minus_residual_minus_one"] = output["deposit_beta"] - output["residual_beta"] - 1.0
    treatment_values = [row[PRIMARY_TREATMENT_ID] for row in numeric]
    output["tdc_mean_abs_mil"] = sum(abs(value) for value in treatment_values) / len(treatment_values)
    output["tdc_max_abs_mil"] = max(abs(value) for value in treatment_values)
    return output


def _rolling_minus_pandemic_rows(rows: list[dict[str, str]], controls: list[str]) -> list[dict[str, Any]]:
    rows_by_quarter = {str(row.get("quarter", "")): row for row in rows if str(row.get("quarter", ""))}
    quarters = sorted(rows_by_quarter, key=_quarter_key)
    output: list[dict[str, Any]] = []
    for end_index in range(WINDOW_QUARTERS - 1, len(quarters)):
        window_quarters = quarters[end_index - WINDOW_QUARTERS + 1 : end_index + 1]
        if not any(_between(quarter, "2020Q1", "2021Q4") for quarter in window_quarters):
            continue
        window_rows = [rows_by_quarter[quarter] for quarter in window_quarters]
        for drop_rule, predicate, description in DROP_RULES:
            subset = [row for row in window_rows if not predicate(str(row.get("quarter", "")))]
            estimate = _estimate_common_pair(subset, controls=controls)
            output.append(
                {
                    "job_id": JOB_ID,
                    "window_start_quarter": window_quarters[0],
                    "window_end_quarter": window_quarters[-1],
                    "drop_rule": drop_rule,
                    "drop_description": description,
                    "dropped_quarters": ",".join([quarter for quarter in window_quarters if predicate(quarter)]),
                    **estimate,
                    "claim_boundary": "Rolling-minus-pandemic persistence diagnostic only; not a structural pass-through estimate.",
                }
            )
    return output


def _residualize(values: list[float], controls: list[list[float]]) -> list[float]:
    if not values:
        return []
    fit = _ols(values, controls, covariance_estimator="classical")
    return fit.residuals


def _full_design_leverage(x_rows: list[list[float]]) -> list[float]:
    xtx_inv = _invert(_matmul(_transpose(x_rows), x_rows))
    leverages: list[float] = []
    for row in x_rows:
        leverages.append(sum(row[col] * value for col, value in enumerate(_matvec(xtx_inv, row))))
    return leverages


def _influence_rows_for_outcome(
    rows: list[dict[str, str]],
    *,
    controls: list[str],
    outcome_id: str,
) -> list[dict[str, Any]]:
    try:
        _fit, _n, used, rejected = _fit_lp(
            rows,
            treatment_id=PRIMARY_TREATMENT_ID,
            outcome_id=outcome_id,
            horizon=0,
            control_ids=controls,
            covariance_lags=1,
        )
    except ValueError:
        return []
    numeric = _numeric_rows(rows, outcome_ids=[outcome_id], control_ids=used)
    if len(numeric) < MIN_OBSERVATIONS:
        return []
    control_rows = [[1.0, *[row[control_id] for control_id in used]] for row in numeric]
    treatment_values = [row[PRIMARY_TREATMENT_ID] for row in numeric]
    outcome_values = [row[outcome_id] for row in numeric]
    residual_treatment = _residualize(treatment_values, control_rows)
    residual_outcome = _residualize(outcome_values, control_rows)
    denominator = sum(value * value for value in residual_treatment)
    if denominator <= 0:
        return []
    slope = sum(xval * yval for xval, yval in zip(residual_treatment, residual_outcome)) / denominator
    x_rows = [[1.0, row[PRIMARY_TREATMENT_ID], *[row[control_id] for control_id in used]] for row in numeric]
    full_fit = _ols(outcome_values, x_rows, covariance_estimator="newey_west", covariance_lags=1)
    leverages = _full_design_leverage(x_rows)
    n_obs = len(numeric)
    n_params = len(x_rows[0])
    mse = sum(residual * residual for residual in full_fit.residuals) / max(n_obs - n_params, 1)
    _, multiplier = _normalization(outcome_id)
    output: list[dict[str, Any]] = []
    for row, resid_x, resid_y, full_resid, leverage in zip(
        numeric,
        residual_treatment,
        residual_outcome,
        full_fit.residuals,
        leverages,
    ):
        contribution = resid_x * resid_y / denominator
        leave_denominator = denominator - resid_x * resid_x
        leave_numerator = slope * denominator - resid_x * resid_y
        leave_one_beta = leave_numerator / leave_denominator if leave_denominator > 0 else ""
        dfbeta = "" if leave_one_beta == "" else leave_one_beta - slope
        cooks_distance = ""
        if mse > 0 and leverage < 1:
            cooks_distance = ((full_resid * full_resid) / (n_params * mse)) * (leverage / ((1.0 - leverage) ** 2))
        output.append(
            {
                "job_id": JOB_ID,
                "outcome": outcome_id,
                "quarter": row["quarter"],
                "n": n_obs,
                "beta": slope * multiplier,
                "beta_full_ols": full_fit.beta[1] * multiplier,
                "residualized_tdc": resid_x,
                "residualized_outcome": resid_y,
                "slope_contribution": contribution * multiplier,
                "leave_one_beta": "" if leave_one_beta == "" else leave_one_beta * multiplier,
                "dfbeta": "" if dfbeta == "" else dfbeta * multiplier,
                "leverage": leverage,
                "cooks_distance": cooks_distance,
                "tdc_value_mil": row[PRIMARY_TREATMENT_ID],
                "outcome_value": row[outcome_id],
                "controls_used": ",".join(used),
                "controls_rejected": ",".join(rejected),
                "claim_boundary": "FWL influence diagnostic; high influence is evidence about leverage, not a causal mechanism.",
            }
        )
    return output


def _influence_rows(rows: list[dict[str, str]], controls: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for outcome_id in OUTCOMES:
        output.extend(_influence_rows_for_outcome(rows, controls=controls, outcome_id=outcome_id))
    return sorted(
        output,
        key=lambda row: abs(float(row["dfbeta"])) if row.get("dfbeta") not in ("", None) else -1.0,
        reverse=True,
    )


def _format_number(value: Any, *, digits: int = 3) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):.{digits}f}"


def _write_report(
    *,
    rolling_rows: list[dict[str, Any]],
    influence_rows: list[dict[str, Any]],
) -> None:
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tier 2 Pass-Through Regime Persistence",
        "",
        "This diagnostic screens whether the elevated rolling pass-through is pandemic-window composition, persistent post-2020 behavior, or quarter-level leverage. It is descriptive and does not identify a structural mechanism.",
        "",
        "## Latest Rolling Window With Pandemic Drops",
        "",
        "| Drop rule | n | Deposit beta | Residual beta | Identity error | TDC mean abs $B |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    latest_window = max(
        (row["window_end_quarter"] for row in rolling_rows if row.get("status") == "estimated"),
        key=_quarter_key,
        default="",
    )
    for row in rolling_rows:
        if row.get("window_end_quarter") != latest_window:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("drop_rule", "")),
                    str(row.get("n", "")),
                    _format_number(row.get("deposit_beta")),
                    _format_number(row.get("residual_beta")),
                    _format_number(row.get("deposit_minus_residual_minus_one"), digits=6),
                    _format_number(float(row.get("tdc_mean_abs_mil", 0.0)) / 1000.0),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Top Quarter Influence Rows",
            "",
            "| Outcome | Quarter | DFBETA | Contribution | Leverage | Cook's distance |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in influence_rows[:16]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('outcome', '')}`",
                    str(row.get("quarter", "")),
                    _format_number(row.get("dfbeta")),
                    _format_number(row.get("slope_contribution")),
                    _format_number(row.get("leverage")),
                    _format_number(row.get("cooks_distance")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- If dropping 2020-2021 materially lowers rolling-window betas, the elevated rolling result is partly window composition.",
            "- If high DFBETA quarters concentrate in 2020-2021 or debt-limit/TGA blocks, the persistence claim should be weaker.",
            "- If non-pandemic windows remain high after these drops, the next step is state-interaction testing rather than more rolling correlations.",
            "",
        ]
    )
    REPORT_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def _ratewall_summary_rows(
    *,
    rolling_rows: list[dict[str, Any]],
    influence_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    latest_window = max(
        (row["window_end_quarter"] for row in rolling_rows if row.get("status") == "estimated"),
        key=_quarter_key,
        default="",
    )
    selected_drop_rules = {
        "none": "latest_rolling_h0_persistence_diagnostic",
        "drop_2020_2021": "pandemic_exclusion_2020q1_2021q4_artifact_diagnostic",
        "drop_2020": "pandemic_exclusion_drop_2020_artifact_diagnostic",
        "drop_2021": "pandemic_exclusion_drop_2021_artifact_diagnostic",
    }
    rows: list[dict[str, Any]] = []
    for row in rolling_rows:
        if (
            row.get("window_end_quarter") != latest_window
            or row.get("drop_rule") not in selected_drop_rules
            or row.get("status") != "estimated"
        ):
            continue
        rows.append(
            {
                "source_diagnostic_row_id": (
                    "tdc_deposit_pass_through_pandemic_exclusion::"
                    f"{row.get('drop_rule', '')}::{row.get('window_start_quarter', '')}_to_"
                    f"{row.get('window_end_quarter', '')}"
                ),
                "job_id": JOB_ID,
                "diagnostic_family": "tdc_deposit_pass_through_pandemic_exclusion",
                "source_row_role": selected_drop_rules[str(row.get("drop_rule", ""))],
                "outcome": "matched_total_deposits",
                "horizon": "0",
                "window_start_quarter": row.get("window_start_quarter", ""),
                "window_end_quarter": row.get("window_end_quarter", ""),
                "window_quarters": WINDOW_QUARTERS,
                "drop_rule": row.get("drop_rule", ""),
                "drop_description": row.get("drop_description", ""),
                "dropped_quarters": row.get("dropped_quarters", ""),
                "sample_label": (
                    f"{row.get('window_start_quarter', '')}_to_"
                    f"{row.get('window_end_quarter', '')}::{row.get('drop_rule', '')}"
                ),
                "n": row.get("n", ""),
                "treatment_id": PRIMARY_TREATMENT_ID,
                "control_policy_mode": "selected_credit_rate_risk_lags_balanced",
                "method_label": "rolling_minus_pandemic_selected_credit_rate_lags",
                "covariance_estimator": "newey_west",
                "covariance_lags": "1",
                "rsquared": row.get("deposit_rsquared", ""),
                "normalized_unit": "dollars_per_dollar_tdc",
                "normalized_beta": row.get("deposit_beta", ""),
                "normalized_se": row.get("deposit_se", ""),
                "normalized_lower95": "",
                "normalized_upper95": "",
                "residual_beta": row.get("residual_beta", ""),
                "residual_se": row.get("residual_se", ""),
                "deposit_minus_residual_minus_one": row.get(
                    "deposit_minus_residual_minus_one", ""
                ),
                "tdc_mean_abs_mil": row.get("tdc_mean_abs_mil", ""),
                "tdc_max_abs_mil": row.get("tdc_max_abs_mil", ""),
                "state_dependence_status": (
                    "pandemic_window_composition_diagnostic_not_structural_regime_model"
                ),
                "source_artifact_backing_status": "pass_ea_tdc_generated_artifact",
                "ratewall_admission_status": (
                    "blocked_diagnostic_not_dynamic_default_or_runtime_input"
                ),
                "scenario_default_allowed": "false",
                "dynamic_path_reference_allowed": "false",
                "claim_boundary": (
                    "pandemic_exclusion_pass_through_diagnostic_not_canonical_or_denominator_input"
                ),
            }
        )
    top_influence = [
        row
        for row in influence_rows
        if row.get("outcome") == "matched_total_deposits"
        and _between(str(row.get("quarter", "")), "2020Q1", "2021Q4")
    ][:8]
    for row in top_influence:
        rows.append(
            {
                "source_diagnostic_row_id": (
                    "tdc_deposit_pass_through_pandemic_influence::"
                    f"{row.get('quarter', '')}"
                ),
                "job_id": JOB_ID,
                "diagnostic_family": "tdc_deposit_pass_through_pandemic_influence",
                "source_row_role": "pandemic_quarter_influence_artifact_diagnostic",
                "outcome": row.get("outcome", ""),
                "horizon": "0",
                "window_start_quarter": "",
                "window_end_quarter": "",
                "window_quarters": "",
                "drop_rule": "not_applicable",
                "drop_description": "FWL influence row for a pandemic-period quarter.",
                "dropped_quarters": "",
                "sample_label": row.get("quarter", ""),
                "n": row.get("n", ""),
                "treatment_id": PRIMARY_TREATMENT_ID,
                "control_policy_mode": "selected_credit_rate_risk_lags_balanced",
                "method_label": "full_sample_fwl_quarter_influence",
                "covariance_estimator": "not_applicable",
                "covariance_lags": "",
                "rsquared": "",
                "normalized_unit": "dollars_per_dollar_tdc",
                "normalized_beta": row.get("beta", ""),
                "normalized_se": "",
                "normalized_lower95": "",
                "normalized_upper95": "",
                "residual_beta": "",
                "residual_se": "",
                "deposit_minus_residual_minus_one": "",
                "tdc_mean_abs_mil": "",
                "tdc_max_abs_mil": row.get("tdc_value_mil", ""),
                "state_dependence_status": (
                    "pandemic_quarter_leverage_diagnostic_not_structural_regime_model"
                ),
                "source_artifact_backing_status": "pass_ea_tdc_generated_artifact",
                "ratewall_admission_status": (
                    "blocked_diagnostic_not_dynamic_default_or_runtime_input"
                ),
                "scenario_default_allowed": "false",
                "dynamic_path_reference_allowed": "false",
                "claim_boundary": (
                    "pandemic_influence_diagnostic_not_canonical_or_denominator_input"
                ),
            }
        )
    return rows


def build_outputs() -> dict[str, list[dict[str, Any]]]:
    _paths, rows, control_ids, _factor_count, _screened_count, selected_credit_lags, selected_rate_lags = _build_inputs()
    controls = _canonical_controls(
        rows,
        _scenario_controls(control_ids, selected_credit_lags, selected_rate_lags)["selected_credit_rate_risk_lags"],
    )
    rolling_rows = _rolling_minus_pandemic_rows(rows, controls)
    influence_rows = _influence_rows(rows, controls)
    return {"rolling_minus_pandemic": rolling_rows, "influence": influence_rows}


def _write_manifest(outputs: dict[str, list[dict[str, Any]]]) -> None:
    MANIFEST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": JOB_ID,
        "outputs": {
            "rolling_minus_pandemic": str(ROLLING_MINUS_OUTPUT.relative_to(ROOT)),
            "influence_quarters": str(INFLUENCE_OUTPUT.relative_to(ROOT)),
            "ratewall_summary": str(RATEWALL_SUMMARY_OUTPUT.relative_to(ROOT)),
            "report": str(REPORT_OUTPUT.relative_to(ROOT)),
        },
        "row_counts": {key: len(value) for key, value in outputs.items()},
        "claim_boundary": "Regime-persistence diagnostics are descriptive screens, not structural estimates.",
    }
    MANIFEST_OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    outputs = build_outputs()
    _write_csv(ROLLING_MINUS_OUTPUT, outputs["rolling_minus_pandemic"])
    _write_csv(INFLUENCE_OUTPUT, outputs["influence"])
    summary_rows = _ratewall_summary_rows(
        rolling_rows=outputs["rolling_minus_pandemic"],
        influence_rows=outputs["influence"],
    )
    _write_csv(RATEWALL_SUMMARY_OUTPUT, summary_rows)
    _write_report(rolling_rows=outputs["rolling_minus_pandemic"], influence_rows=outputs["influence"])
    _write_manifest(outputs)
    print(f"wrote {len(outputs['rolling_minus_pandemic'])} rolling-minus-pandemic rows to {ROLLING_MINUS_OUTPUT}")
    print(f"wrote {len(outputs['influence'])} influence rows to {INFLUENCE_OUTPUT}")
    print(f"wrote {len(summary_rows)} RateWall summary rows to {RATEWALL_SUMMARY_OUTPUT}")
    print(f"wrote report to {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
