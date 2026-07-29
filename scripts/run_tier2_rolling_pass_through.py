"""Rolling selected-credit/rate-lag TDC pass-through diagnostics.

This script reuses the same input construction, selected lag controls, rank-aware
control admission, and Newey-West OLS helper as
``run_submission_appendix_diagnostics.py``. It writes both rolling regression
estimates and a scale-free rolling correlation diagnostic. The correlation output
is descriptive stability evidence only; it is not a replacement for beta-per-$1
or LP/regression-style pass-through estimates.
"""
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

from ea_tdc.open_contract import (  # noqa: E402
    CANONICAL_CONTROL_IDS,
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
)


JOB_ID = "tier2_rolling_selected_credit_rate_pass_through"
PRIMARY_TREATMENT_ID = CANONICAL_TREATMENT_ID
PRIMARY_RESIDUAL_ID = CANONICAL_RESIDUAL_ID
OUTCOMES = [CANONICAL_OUTCOME_ID, PRIMARY_RESIDUAL_ID]
WINDOW_QUARTERS = ROLLING_WINDOW_QUARTERS
MIN_OBSERVATIONS = 40
REGRESSION_OUTPUT = ROOT / "output/models/tier2_rolling_selected_credit_rate_pass_through_estimates.csv"
CORRELATION_OUTPUT = ROOT / "output/reports/tier2_rolling_selected_credit_rate_pass_through_correlations.csv"
REPORT_OUTPUT = ROOT / "output/reports/tier2_rolling_selected_credit_rate_pass_through.md"
MANIFEST_OUTPUT = ROOT / "output/manifests/tier2_rolling_selected_credit_rate_pass_through_summary.json"


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


def _numeric_count(
    rows: list[dict[str, str]],
    *,
    treatment_id: str,
    outcome_id: str,
) -> int:
    return len(
        _effective_sample_quarters(
            rows,
            treatment_id=treatment_id,
            outcome_id=outcome_id,
        )
    )


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _effective_sample_quarters(
    rows: list[dict[str, str]],
    *,
    treatment_id: str,
    outcome_id: str,
    control_ids: list[str] | tuple[str, ...] = (),
) -> list[str]:
    quarters: list[str] = []
    for row in rows:
        quarter = str(row.get("quarter", "")).strip()
        if not quarter:
            continue
        required_ids = (treatment_id, outcome_id, *control_ids)
        if all(_finite_float(row.get(series_id, "")) is not None for series_id in required_ids):
            quarters.append(quarter)
    return sorted(quarters)


def _last_joint_observed_quarter(
    rows: list[dict[str, str]],
    *,
    treatment_id: str,
    outcome_id: str,
) -> str | None:
    quarters = _effective_sample_quarters(
        rows,
        treatment_id=treatment_id,
        outcome_id=outcome_id,
    )
    return quarters[-1] if quarters else None


def _endpoint_is_jointly_observed(
    rows: list[dict[str, str]],
    *,
    treatment_id: str,
    outcome_id: str,
    window_end: str,
) -> bool:
    for row in rows:
        if str(row.get("quarter", "")).strip() != window_end:
            continue
        return (
            _finite_float(row.get(treatment_id, "")) is not None
            and _finite_float(row.get(outcome_id, "")) is not None
        )
    return False


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    left_ss = sum(value * value for value in left_centered)
    right_ss = sum(value * value for value in right_centered)
    if left_ss <= 0 or right_ss <= 0:
        return None
    covariance = sum(lval * rval for lval, rval in zip(left_centered, right_centered))
    return covariance / math.sqrt(left_ss * right_ss)


def _paired_values(
    rows: list[dict[str, str]],
    *,
    treatment_id: str,
    outcome_id: str,
) -> tuple[list[float], list[float]]:
    treatment_values: list[float] = []
    outcome_values: list[float] = []
    for row in rows:
        treatment = _finite_float(row.get(treatment_id, ""))
        outcome = _finite_float(row.get(outcome_id, ""))
        if treatment is None or outcome is None:
            continue
        treatment_values.append(treatment)
        outcome_values.append(outcome)
    return treatment_values, outcome_values


def _correlation_row(
    *,
    rows: list[dict[str, str]],
    outcome_id: str,
    window_start: str,
    window_end: str,
) -> dict[str, Any] | None:
    last_joint_observed = _last_joint_observed_quarter(
        rows,
        treatment_id=PRIMARY_TREATMENT_ID,
        outcome_id=outcome_id,
    )
    if (
        last_joint_observed is None
        or window_end > last_joint_observed
        or not _endpoint_is_jointly_observed(
            rows,
            treatment_id=PRIMARY_TREATMENT_ID,
            outcome_id=outcome_id,
            window_end=window_end,
        )
    ):
        return None
    treatment_values, outcome_values = _paired_values(
        rows,
        treatment_id=PRIMARY_TREATMENT_ID,
        outcome_id=outcome_id,
    )
    if len(treatment_values) < MIN_OBSERVATIONS:
        return None
    correlation = _pearson(treatment_values, outcome_values)
    if correlation is None:
        return None
    effective_quarters = _effective_sample_quarters(
        rows,
        treatment_id=PRIMARY_TREATMENT_ID,
        outcome_id=outcome_id,
    )
    return {
        "job_id": JOB_ID,
        "window_start_quarter": window_start,
        "window_end_quarter": window_end,
        "effective_sample_start": effective_quarters[0],
        "effective_sample_end": effective_quarters[-1],
        "window_quarters": WINDOW_QUARTERS,
        "outcome": outcome_id,
        "treatment_id": PRIMARY_TREATMENT_ID,
        "correlation": correlation,
        "n": len(treatment_values),
        "diagnostic_type": "rolling_pearson_correlation",
        "diagnostic_role": "secondary_descriptive_stability_evidence",
        "claim_boundary": (
            "Scale-free co-movement diagnostic only; not a causal estimate and "
            "not the canonical pass-through magnitude."
        ),
        "canonical_interpretation": (
            "Use beta-per-dollar or selected-lag LP/regression pass-through estimates "
            "where available."
        ),
        "pinned_anchor_job_id": "tdc_tier2_mmf_rrp_canonical_full_panel",
        "pinned_control_policy_mode": "balanced",
    }


def _estimate_window(
    *,
    rows: list[dict[str, str]],
    controls: list[str],
    outcome_id: str,
    window_start: str,
    window_end: str,
) -> dict[str, Any] | None:
    last_joint_observed = _last_joint_observed_quarter(
        rows,
        treatment_id=PRIMARY_TREATMENT_ID,
        outcome_id=outcome_id,
    )
    if (
        last_joint_observed is None
        or window_end > last_joint_observed
        or not _endpoint_is_jointly_observed(
            rows,
            treatment_id=PRIMARY_TREATMENT_ID,
            outcome_id=outcome_id,
            window_end=window_end,
        )
    ):
        return None
    if _numeric_count(rows, treatment_id=PRIMARY_TREATMENT_ID, outcome_id=outcome_id) < MIN_OBSERVATIONS:
        return None
    try:
        fit, n, used, rejected = _fit_lp(
            rows,
            treatment_id=PRIMARY_TREATMENT_ID,
            outcome_id=outcome_id,
            horizon=0,
            control_ids=controls,
            covariance_lags=1,
        )
    except ValueError:
        return None
    if n < MIN_OBSERVATIONS:
        return None
    effective_quarters = _effective_sample_quarters(
        rows,
        treatment_id=PRIMARY_TREATMENT_ID,
        outcome_id=outcome_id,
        control_ids=used,
    )
    if len(effective_quarters) != n:
        raise ValueError(
            "Rolling effective-sample reconstruction disagrees with the fitted sample: "
            f"outcome={outcome_id!r}, reconstructed={len(effective_quarters)}, fitted={n}"
        )
    beta = fit.beta[1]
    se = fit.ses[1]
    z = beta / se if se > 0 else None
    p_two = "" if z is None else _normal_p_two_sided(z)
    _, multiplier = _normalization(outcome_id)
    effect_unit, effect_value = _effect_per_100b(outcome_id, beta)
    normalized_beta = beta * multiplier
    normalized_se = se * multiplier
    return {
        "job_id": JOB_ID,
        "window_start_quarter": window_start,
        "window_end_quarter": window_end,
        "effective_sample_start": effective_quarters[0],
        "effective_sample_end": effective_quarters[-1],
        "window_quarters": WINDOW_QUARTERS,
        "outcome": outcome_id,
        "horizon": 0,
        "beta": beta,
        "se": se,
        "lower95": beta - 1.96 * se,
        "upper95": beta + 1.96 * se,
        "z_score": "" if z is None else z,
        "p_value_normal": p_two,
        "n": n,
        "treatment_id": PRIMARY_TREATMENT_ID,
        "control_ids_used": ",".join(used),
        "dropped_control_ids": ",".join(rejected),
        "response_type": "direct_at_h",
        "inference_method": "rolling_selected_credit_rate_lags_rank_aware",
        "covariance_estimator": "newey_west",
        "covariance_lags": 1,
        "rsquared": fit.rsquared,
        "sample_label": f"{window_start}_to_{window_end}",
        "effective_sample_label": f"{effective_quarters[0]}_to_{effective_quarters[-1]}",
        "normalized_unit": "dollars_per_dollar_tdc",
        "normalized_beta": normalized_beta,
        "normalized_se": normalized_se,
        "normalized_lower95": (beta - 1.96 * se) * multiplier,
        "normalized_upper95": (beta + 1.96 * se) * multiplier,
        "effect_per_100b_unit": effect_unit,
        "effect_per_100b_tdc": effect_value,
        "effect_per_100b_se": normalized_se * 100.0,
        "effect_per_100b_lower95": effect_value - 1.96 * normalized_se * 100.0,
        "effect_per_100b_upper95": effect_value + 1.96 * normalized_se * 100.0,
        "pinned_anchor_job_id": "tdc_tier2_mmf_rrp_canonical_full_panel",
        "pinned_control_policy_mode": "balanced",
    }


def build_rows() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    _paths, rows, _control_ids, _factor_count, _screened_count, _selected_credit_lags, _selected_rate_lags = _build_inputs()
    controls = list(CANONICAL_CONTROL_IDS)
    rows_by_quarter = {
        str(row.get("quarter", "")): row
        for row in rows
        if str(row.get("quarter", ""))
    }
    quarters = sorted(rows_by_quarter)
    last_joint_observed_quarter = {
        outcome_id: _last_joint_observed_quarter(
            rows,
            treatment_id=PRIMARY_TREATMENT_ID,
            outcome_id=outcome_id,
        )
        for outcome_id in OUTCOMES
    }
    estimates: list[dict[str, Any]] = []
    correlations: list[dict[str, Any]] = []
    for end_index in range(WINDOW_QUARTERS - 1, len(quarters)):
        window_quarters = quarters[end_index - WINDOW_QUARTERS + 1 : end_index + 1]
        window_rows = [rows_by_quarter[quarter] for quarter in window_quarters]
        window_start = window_quarters[0]
        window_end = window_quarters[-1]
        for outcome_id in OUTCOMES:
            last_joint = last_joint_observed_quarter[outcome_id]
            if last_joint is None or window_end > last_joint:
                continue
            estimate = _estimate_window(
                rows=window_rows,
                controls=controls,
                outcome_id=outcome_id,
                window_start=window_start,
                window_end=window_end,
            )
            if estimate is not None and math.isfinite(float(estimate["normalized_beta"])):
                estimates.append(estimate)
            correlation = _correlation_row(
                rows=window_rows,
                outcome_id=outcome_id,
                window_start=window_start,
                window_end=window_end,
            )
            if correlation is not None and math.isfinite(float(correlation["correlation"])):
                correlations.append(correlation)
    return estimates, correlations, {
        "control_ids": controls,
        "last_joint_observed_quarter": last_joint_observed_quarter,
    }


def _format_number(value: Any, *, digits: int = 3) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.{digits}f}"


def _latest_by_outcome(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        outcome = str(row.get("outcome", ""))
        if not outcome:
            continue
        if outcome not in latest or str(row.get("window_end_quarter", "")) > str(latest[outcome].get("window_end_quarter", "")):
            latest[outcome] = row
    return latest


def _write_report(
    *,
    regression_rows: list[dict[str, Any]],
    correlation_rows: list[dict[str, Any]],
) -> None:
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    latest_regression = _latest_by_outcome(regression_rows)
    latest_correlation = _latest_by_outcome(correlation_rows)
    lines = [
        "# Rolling Selected-Lag Pass-Through Diagnostic",
        "",
        "This diagnostic is descriptive stability evidence for an assumptive project. "
        "The rolling correlation rows are scale-free co-movement checks, not causal "
        "or canonical pass-through estimates.",
        "",
        "Canonical interpretation remains the beta-per-$1 or selected-lag LP/regression "
        "pass-through estimate where available. Rolling correlations should only be used "
        "as secondary context on whether the deposit association is stable across windows.",
        "",
        f"- Window length: {WINDOW_QUARTERS} quarters.",
        f"- Minimum observations per window: {MIN_OBSERVATIONS}.",
        f"- Treatment: `{PRIMARY_TREATMENT_ID}`.",
        "",
        "## Latest Window",
        "",
        "| Outcome | Window | Rolling beta per $1 TDC | Effect per +$100B TDC | Rolling correlation | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for outcome_id in OUTCOMES:
        regression = latest_regression.get(outcome_id, {})
        correlation = latest_correlation.get(outcome_id, {})
        window = ""
        if regression:
            window = f"{regression.get('window_start_quarter', '')} to {regression.get('window_end_quarter', '')}"
        elif correlation:
            window = f"{correlation.get('window_start_quarter', '')} to {correlation.get('window_end_quarter', '')}"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{outcome_id}`",
                    window,
                    _format_number(regression.get("normalized_beta")),
                    _format_number(regression.get("effect_per_100b_tdc"), digits=2),
                    _format_number(correlation.get("correlation")),
                    str(regression.get("n", correlation.get("n", ""))),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- Treat sign and broad stability as descriptive evidence only.",
            "- Do not read the rolling correlation as a pass-through share, price effect, or identification result.",
            "- Use the selected-lag regression/LP coefficients for pass-through magnitudes.",
            "",
        ]
    )
    REPORT_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(
    *,
    regression_rows: list[dict[str, Any]],
    correlation_rows: list[dict[str, Any]],
    control_ids: list[str],
    last_joint_observed_quarter: dict[str, str | None],
) -> None:
    MANIFEST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    all_rows = [*regression_rows, *correlation_rows]
    effective_starts = [
        str(row["effective_sample_start"])
        for row in all_rows
        if row.get("effective_sample_start")
    ]
    effective_ends = [
        str(row["effective_sample_end"])
        for row in all_rows
        if row.get("effective_sample_end")
    ]
    effective_sample_bounds: dict[str, dict[str, dict[str, str | None]]] = {}
    for outcome_id in OUTCOMES:
        effective_sample_bounds[outcome_id] = {}
        for output_name, rows in (
            ("regression", regression_rows),
            ("correlation", correlation_rows),
        ):
            outcome_rows = [row for row in rows if row.get("outcome") == outcome_id]
            starts = [
                str(row["effective_sample_start"])
                for row in outcome_rows
                if row.get("effective_sample_start")
            ]
            ends = [
                str(row["effective_sample_end"])
                for row in outcome_rows
                if row.get("effective_sample_end")
            ]
            effective_sample_bounds[outcome_id][output_name] = {
                "start": min(starts) if starts else None,
                "end": max(ends) if ends else None,
            }
    payload = {
        "job_id": JOB_ID,
        "treatment_id": PRIMARY_TREATMENT_ID,
        "outcome_ids": OUTCOMES,
        "control_ids": control_ids,
        "regression_rows": len(regression_rows),
        "correlation_rows": len(correlation_rows),
        "window_quarters": WINDOW_QUARTERS,
        "min_observations": MIN_OBSERVATIONS,
        "effective_sample_start": min(effective_starts) if effective_starts else None,
        "effective_sample_end": max(effective_ends) if effective_ends else None,
        "effective_sample_bounds": effective_sample_bounds,
        "last_joint_observed_quarter": last_joint_observed_quarter,
        "window": {
            "nominal_quarters": WINDOW_QUARTERS,
            "minimum_observations": MIN_OBSERVATIONS,
            "endpoint_rule": (
                "Emit an outcome window only when canonical treatment and outcome are "
                "both observed at the nominal window end."
            ),
        },
        "outputs": {
            "regression_estimates": str(REGRESSION_OUTPUT.relative_to(ROOT)),
            "correlations": str(CORRELATION_OUTPUT.relative_to(ROOT)),
            "report": str(REPORT_OUTPUT.relative_to(ROOT)),
        },
        "claim_boundary": (
            "Rolling correlations are descriptive stability diagnostics only; canonical "
            "pass-through remains beta-per-dollar or LP/regression estimates."
        ),
    }
    MANIFEST_OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    regression_rows, correlation_rows, metadata = build_rows()
    _write_csv(REGRESSION_OUTPUT, regression_rows)
    _write_csv(CORRELATION_OUTPUT, correlation_rows)
    _write_report(regression_rows=regression_rows, correlation_rows=correlation_rows)
    _write_manifest(
        regression_rows=regression_rows,
        correlation_rows=correlation_rows,
        control_ids=metadata["control_ids"],
        last_joint_observed_quarter=metadata["last_joint_observed_quarter"],
    )
    print(f"wrote {len(regression_rows)} rows to {REGRESSION_OUTPUT}")
    print(f"wrote {len(correlation_rows)} rows to {CORRELATION_OUTPUT}")
    print(f"wrote report to {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
