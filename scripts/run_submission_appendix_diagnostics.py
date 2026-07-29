"""Submission-hardening appendix diagnostics for the broad working-paper / conference
release of the TDC deposit-incidence result.

This script consolidates five appendix-ready tables that protect the headline reduced-form
deposit pass-through estimate:

1. Lead-placebo coefficient table (per-outcome, per-lead beta/SE/p) for the primary
   long-history TDC treatment under baseline and selected credit/rate-lag specs.
2. HAC bandwidth sensitivity for the headline h=0 matched-deposit coefficient under the
   selected credit/rate lag spec.
3. Factor-tail robustness: K=100 four-factor tail vs no-factor (K=0) for the Table 1
   outcome set under the selected credit/rate lag spec.
4. Splice / construction audit for the long-history bank Treasury-interest backcast that
   underlies the regression-grade TDC treatment.
5. Plumbing magnitudes: regression-derived h=0 magnitudes used in manuscript prose,
   plus a demotion note for any descriptive refill-share claims that are not actually
   produced by the pipeline.

The script reuses helpers from run_tier2_credit_lead_diagnostics.py and
run_tier2_state_dependent_credit_causality.py so the appendix tables share the same
control surface, treatment definitions, and normalization conventions as the headline
causality readout.
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

from ea_tdc.designs.quarterly import build_quarterly_design
from ea_tdc.estimation import _build_quarterly_target, _coerce_float, _ols
from ea_tdc.open_contract import (
    CANONICAL_OUTCOME_ID,
    CANONICAL_RESIDUAL_ID,
    CANONICAL_TREATMENT_ID,
    CANONICAL_TREATMENT_LABEL,
    CREDIT_SCREEN_OUTCOME_IDS,
)
from ea_tdc.paths import project_paths
from ea_tdc.residualized_shock import _load_factor_branch
from ea_tdc.utils import utc_now_iso, write_json
from run_pinned_factor_residual_bridge import (
    ANCHOR_JOB_ID,
    CONTROL_POLICY_MODE,
    FACTOR_COUNT,
    K_SCREENED,
    MERGE_JOBS,
    TREATMENTS as BRIDGE_TREATMENTS,
    _load_manifest,
    _merge_by_quarter,
)
from run_tier2_credit_lead_diagnostics import (
    CREDIT_LAG_SOURCES,
    LAG_PERIODS,
    MAX_FACTOR_TAIL_CONTROLS,
    RATE_RISK_LAG_SOURCES,
    SELECTED_CREDIT_LAG_PATTERNS,
    SELECTED_RATE_RISK_LAG_PATTERNS,
    _add_lags,
    _cap_factor_tail,
    _insert_lags_before_factor_tail,
    _matrix_rank,
    _lp_sample,
)
from run_tier2_state_dependent_credit_causality import (
    _active_controls,
    _add_future_treatment,
    _effect_per_100b,
    _normalization,
)


PRIMARY_TREATMENT_LABEL = CANONICAL_TREATMENT_LABEL
PRIMARY_TREATMENT_ID = CANONICAL_TREATMENT_ID
PRIMARY_RESIDUAL_ID = CANONICAL_RESIDUAL_ID

LEAD_HORIZONS = [1, 2, 3, 4]

LEAD_OUTCOMES = [
    CANONICAL_OUTCOME_ID,
    *CREDIT_SCREEN_OUTCOME_IDS,
]

FACTOR_TAIL_OUTCOMES = [
    CANONICAL_OUTCOME_ID,
    PRIMARY_RESIDUAL_ID,
    *CREDIT_SCREEN_OUTCOME_IDS,
]

HAC_BANDWIDTHS = [1, 4, 6, 8]
HAC_OUTCOME = CANONICAL_OUTCOME_ID


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normal_p_two_sided(z_score: float) -> float:
    return math.erfc(abs(z_score) / math.sqrt(2.0))


def _format_value(value: Any, *, precision: int = 3) -> str:
    if value is None or value == "":
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(numeric) >= 10:
        return f"{numeric:.2f}"
    return f"{numeric:{precision + 1}.{precision}f}"


def _format_p(value: Any) -> str:
    if value is None or value == "":
        return ""
    numeric = float(value)
    if numeric < 0.001:
        return "<0.001"
    return f"{numeric:.3f}"


def _build_inputs():
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
    bundle_paths = [
        Path(str(_load_manifest(paths, job_id).get("bundle_path", "")))
        for job_id in MERGE_JOBS
    ]
    rows = _merge_by_quarter(factor_rows, bundle_paths)
    credit_lags = _add_lags(rows, CREDIT_LAG_SOURCES, LAG_PERIODS)
    rate_risk_lags = _add_lags(rows, RATE_RISK_LAG_SOURCES, LAG_PERIODS)
    selected_credit_lags = [c for c in credit_lags if c in SELECTED_CREDIT_LAG_PATTERNS]
    selected_rate_lags = [c for c in rate_risk_lags if c in SELECTED_RATE_RISK_LAG_PATTERNS]
    return paths, rows, control_ids, factor_count, screened_count, selected_credit_lags, selected_rate_lags


def _admit_controls_rank_aware(
    rows: list[dict[str, str]],
    *,
    treatment_id: str,
    outcome_id: str,
    horizon: int,
    candidate_controls: list[str],
    covariance_lags: int,
) -> tuple[list[str], list[str]]:
    """Greedy rank-aware control admission. Mirrors run_tier2_credit_lead_diagnostics._rank_aware_lp_estimates."""
    selected: list[str] = []
    rejected: list[str] = []
    for control_id in candidate_controls:
        trial = [*selected, control_id]
        y_values, x_rows = _lp_sample(
            rows,
            treatment_id=treatment_id,
            control_ids=trial,
            outcome_id=outcome_id,
            horizon=horizon,
        )
        required_rank = len(trial) + 2
        if len(y_values) <= required_rank + 3 or _matrix_rank(x_rows) < required_rank:
            rejected.append(control_id)
            continue
        try:
            _ols(y_values, x_rows, covariance_estimator="newey_west", covariance_lags=max(covariance_lags, 1))
        except ValueError:
            rejected.append(control_id)
            continue
        selected.append(control_id)
    return selected, rejected


def _fit_lp(
    rows: list[dict[str, str]],
    *,
    treatment_id: str,
    outcome_id: str,
    horizon: int,
    control_ids: list[str],
    covariance_lags: int,
) -> tuple[Any, int, list[str], list[str]]:
    selected, rejected = _admit_controls_rank_aware(
        rows,
        treatment_id=treatment_id,
        outcome_id=outcome_id,
        horizon=horizon,
        candidate_controls=control_ids,
        covariance_lags=covariance_lags,
    )
    y_values, x_rows = _lp_sample(
        rows,
        treatment_id=treatment_id,
        control_ids=selected,
        outcome_id=outcome_id,
        horizon=horizon,
    )
    fit = _ols(y_values, x_rows, covariance_estimator="newey_west", covariance_lags=max(covariance_lags, 1))
    return fit, len(y_values), selected, rejected


def _scenario_controls(
    base_controls: list[str],
    selected_credit_lags: list[str],
    selected_rate_lags: list[str],
) -> dict[str, list[str]]:
    capped = _cap_factor_tail(base_controls, MAX_FACTOR_TAIL_CONTROLS)
    return {
        "baseline_controls": capped[:],
        "selected_credit_rate_risk_lags": _insert_lags_before_factor_tail(
            capped, [*selected_credit_lags, *selected_rate_lags]
        ),
    }


def _decorate_row(
    *,
    scenario: str,
    outcome_id: str,
    lead: int,
    horizon_label: str,
    fit: Any,
    n: int,
    used: list[str],
    rejected: list[str],
    future_treatment_id: str,
) -> dict[str, Any]:
    beta = fit.beta[1]
    se = fit.ses[1]
    z = beta / se if se > 0 else None
    p_two = None if z is None else _normal_p_two_sided(z)
    _, multiplier = _normalization(outcome_id)
    effect_unit, effect_value = _effect_per_100b(outcome_id, beta)
    # se_per_100b mirrors the scaling that converts beta -> effect_per_100b: dollars-per-dollar-TDC * multiplier * 100.
    # For credit outcomes the underlying units are billions-per-million; multiplier=1000 reaches dollars-per-dollar-TDC,
    # and the additional *100 gets to per +$100B TDC.
    se_per_100b = se * multiplier * 100.0
    normalized_se = se * multiplier
    return {
        "scenario": scenario,
        "outcome": outcome_id,
        "lead_quarters": lead,
        "horizon_label": horizon_label,
        "n": n,
        "beta": beta,
        "se": se,
        "z_score": "" if z is None else z,
        "p_value": "" if p_two is None else p_two,
        "lower95": beta - 1.96 * se,
        "upper95": beta + 1.96 * se,
        "normalized_beta": beta * multiplier,
        "normalized_se": normalized_se,
        "normalized_unit": "dollars_per_dollar_tdc",
        "effect_per_100b_tdc": effect_value,
        "effect_per_100b_se": se_per_100b,
        "effect_per_100b_lower95": effect_value - 1.96 * se_per_100b,
        "effect_per_100b_upper95": effect_value + 1.96 * se_per_100b,
        "effect_per_100b_unit": effect_unit,
        "covariance_estimator": "newey_west",
        "covariance_lags": 1,
        "controls_used": ",".join(used),
        "controls_rejected": ",".join(rejected),
        "future_treatment_id": future_treatment_id,
        "actual_treatment_id": PRIMARY_TREATMENT_ID,
    }


def _build_lead_placebo_table(
    rows: list[dict[str, str]],
    *,
    base_controls: list[str],
    selected_credit_lags: list[str],
    selected_rate_lags: list[str],
) -> list[dict[str, Any]]:
    scenarios = _scenario_controls(base_controls, selected_credit_lags, selected_rate_lags)
    results: list[dict[str, Any]] = []
    for lead in LEAD_HORIZONS:
        lead_rows, lead_column = _add_future_treatment(rows, PRIMARY_TREATMENT_ID, lead)
        for scenario, controls in scenarios.items():
            for outcome_id in LEAD_OUTCOMES:
                try:
                    fit, n, used, rejected = _fit_lp(
                        lead_rows,
                        treatment_id=lead_column,
                        outcome_id=outcome_id,
                        horizon=0,
                        control_ids=controls,
                        covariance_lags=1,
                    )
                except ValueError:
                    continue
                results.append(_decorate_row(
                    scenario=scenario,
                    outcome_id=outcome_id,
                    lead=lead,
                    horizon_label=f"h=-{lead}",
                    fit=fit,
                    n=n,
                    used=used,
                    rejected=rejected,
                    future_treatment_id=lead_column,
                ))
    # Also include the actual h=0 contemporaneous coefficient for reference rows.
    for scenario, controls in scenarios.items():
        for outcome_id in LEAD_OUTCOMES:
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
                continue
            results.append(_decorate_row(
                scenario=scenario,
                outcome_id=outcome_id,
                lead=0,
                horizon_label="h=0",
                fit=fit,
                n=n,
                used=used,
                rejected=rejected,
                future_treatment_id="",
            ))
    return results


def _build_hac_bandwidth_table(
    rows: list[dict[str, str]],
    *,
    base_controls: list[str],
    selected_credit_lags: list[str],
    selected_rate_lags: list[str],
) -> list[dict[str, Any]]:
    scenarios = _scenario_controls(base_controls, selected_credit_lags, selected_rate_lags)
    controls = scenarios["selected_credit_rate_risk_lags"]
    selected, rejected = _admit_controls_rank_aware(
        rows,
        treatment_id=PRIMARY_TREATMENT_ID,
        outcome_id=HAC_OUTCOME,
        horizon=0,
        candidate_controls=controls,
        covariance_lags=max(HAC_BANDWIDTHS),
    )
    y_values, x_rows = _lp_sample(
        rows,
        treatment_id=PRIMARY_TREATMENT_ID,
        control_ids=selected,
        outcome_id=HAC_OUTCOME,
        horizon=0,
    )
    results: list[dict[str, Any]] = []
    for bandwidth in HAC_BANDWIDTHS:
        fit = _ols(y_values, x_rows, covariance_estimator="newey_west", covariance_lags=bandwidth)
        beta = fit.beta[1]
        se = fit.ses[1]
        z = beta / se if se > 0 else None
        p_two = None if z is None else _normal_p_two_sided(z)
        _, multiplier = _normalization(HAC_OUTCOME)
        effect_unit, effect_value = _effect_per_100b(HAC_OUTCOME, beta)
        se_per_100b = se * multiplier * 100.0
        results.append({
            "outcome": HAC_OUTCOME,
            "spec": "selected_credit_rate_risk_lags",
            "covariance_estimator": "newey_west",
            "covariance_lags": bandwidth,
            "n": len(y_values),
            "beta": beta,
            "se": se,
            "z_score": "" if z is None else z,
            "p_value": "" if p_two is None else p_two,
            "lower95": beta - 1.96 * se,
            "upper95": beta + 1.96 * se,
            "normalized_beta": beta * multiplier,
            "normalized_se": se * multiplier,
            "effect_per_100b_tdc": effect_value,
            "effect_per_100b_se": se_per_100b,
            "effect_per_100b_lower95": effect_value - 1.96 * se_per_100b,
            "effect_per_100b_upper95": effect_value + 1.96 * se_per_100b,
            "effect_per_100b_unit": effect_unit,
            "controls_used": ",".join(selected),
            "controls_rejected": ",".join(rejected),
        })
    return results


def _factor_only(control_ids: list[str]) -> list[str]:
    return [c for c in control_ids if c.startswith("dflmx_") or c.startswith("imp_dflmx_")]


def _non_factor(control_ids: list[str]) -> list[str]:
    return [c for c in control_ids if not (c.startswith("dflmx_") or c.startswith("imp_dflmx_"))]


def _build_factor_tail_table(
    rows: list[dict[str, str]],
    *,
    base_controls: list[str],
    selected_credit_lags: list[str],
    selected_rate_lags: list[str],
    factor_count: int,
) -> list[dict[str, Any]]:
    scenarios_full = _scenario_controls(base_controls, selected_credit_lags, selected_rate_lags)
    selected_full = scenarios_full["selected_credit_rate_risk_lags"]
    selected_no_factor = _non_factor(selected_full)
    factor_ids = _factor_only(selected_full)

    variants: dict[str, list[str]] = {
        f"k100_factors_{factor_count}": selected_full,
        "k0_no_factors": selected_no_factor,
    }
    results: list[dict[str, Any]] = []
    for variant_label, controls in variants.items():
        for outcome_id in FACTOR_TAIL_OUTCOMES:
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
                continue
            beta = fit.beta[1]
            se = fit.ses[1]
            z = beta / se if se > 0 else None
            p_two = None if z is None else _normal_p_two_sided(z)
            _, multiplier = _normalization(outcome_id)
            effect_unit, effect_value = _effect_per_100b(outcome_id, beta)
            se_per_100b = se * multiplier * 100.0
            results.append({
                "variant": variant_label,
                "factors_requested": factor_count if variant_label.startswith("k100") else 0,
                "outcome": outcome_id,
                "n": n,
                "beta": beta,
                "se": se,
                "p_value": "" if p_two is None else p_two,
                "lower95": beta - 1.96 * se,
                "upper95": beta + 1.96 * se,
                "normalized_beta": beta * multiplier,
                "normalized_se": se * multiplier,
                "effect_per_100b_tdc": effect_value,
                "effect_per_100b_se": se_per_100b,
                "effect_per_100b_lower95": effect_value - 1.96 * se_per_100b,
                "effect_per_100b_upper95": effect_value + 1.96 * se_per_100b,
                "effect_per_100b_unit": effect_unit,
                "covariance_estimator": "newey_west",
                "covariance_lags": 1,
                "controls_used": ",".join(used),
                "controls_rejected": ",".join(rejected),
                "factor_controls_in_pool": ",".join(factor_ids) if variant_label.startswith("k100") else "",
            })
    return results


# --------------------------------------------------------------------------- splice audit


SPLICE_BACKCAST_RELATIVE_CANDIDATES = [
    "../tdcest/data/processed/tier2_regression_interest_backcast.csv",
    "../../tdcest/data/processed/tier2_regression_interest_backcast.csv",
]


def _quarter_label(date_text: str) -> str:
    if not date_text:
        return ""
    year, month, _ = date_text.split("-")
    q = (int(month) - 1) // 3 + 1
    return f"{year}Q{q}"


def _build_splice_audit() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    backcast_path: Path | None = None
    for candidate in SPLICE_BACKCAST_RELATIVE_CANDIDATES:
        candidate_path = (ROOT / candidate).resolve()
        if candidate_path.exists():
            backcast_path = candidate_path
            break
    if backcast_path is None:
        return [], {
            "backcast_source": "missing",
            "note": (
                "Could not locate tdcest/data/processed/tier2_regression_interest_backcast.csv. "
                "lambda_H15 and seam diagnostic could not be computed."
            ),
        }
    backcast = _read_csv(backcast_path)
    bank = [row for row in backcast if row.get("sector_group") == "bank"]
    if not bank:
        return [], {
            "backcast_source": str(backcast_path),
            "note": "Bank sector rows missing from tier2_regression_interest_backcast.csv",
        }
    lambda_h15_text = next((row.get("backcast_scale_ratio", "") for row in bank if row.get("backcast_scale_ratio")), "")
    lambda_h15 = float(lambda_h15_text) if lambda_h15_text else float("nan")

    # Segment dates: earliest/latest per method_tier
    segments: dict[str, dict[str, str]] = {}
    for row in bank:
        tier = str(row.get("method_tier", ""))
        date_text = str(row.get("date", ""))
        if not tier or not date_text:
            continue
        bucket = segments.setdefault(tier, {"first_date": date_text, "last_date": date_text})
        if date_text < bucket["first_date"]:
            bucket["first_date"] = date_text
        if date_text > bucket["last_date"]:
            bucket["last_date"] = date_text

    # lambda_H15 calibration window: from _scale_ratio in tier2_regression_backcast.py the
    # calibration sample is the first 5 years of dates where both component and legacy are
    # non-null and legacy is non-zero. Reproduce that here.
    overlap_dates = [
        row.get("date", "")
        for row in bank
        if str(row.get("component_anchored_interest_mil", "")).strip()
        and str(row.get("legacy_h15_interest_proxy_mil", "")).strip()
        and float(row.get("legacy_h15_interest_proxy_mil") or 0.0) != 0.0
    ]
    overlap_dates.sort()
    overlap_start = overlap_dates[0] if overlap_dates else ""
    calibration_end = ""
    if overlap_start:
        # pandas DateOffset(years=5) is exactly five years later, same month/day.
        year, month, day = overlap_start.split("-")
        calibration_end_year = str(int(year) + 5)
        calibration_end = f"{calibration_end_year}-{month}-{day}"
    calibration_dates = [d for d in overlap_dates if d and d <= calibration_end]

    # Seam diagnostics
    by_date = {row.get("date", ""): row for row in bank}

    def _diag(date_text: str) -> dict[str, Any]:
        row = by_date.get(date_text, {})
        legacy = row.get("legacy_h15_interest_proxy_mil", "")
        component = row.get("component_anchored_interest_mil", "")
        proxy = row.get("bank_tier2_regression_interest_proxy", "")
        legacy_f = float(legacy) if str(legacy).strip() else float("nan")
        component_f = float(component) if str(component).strip() else float("nan")
        proxy_f = float(proxy) if str(proxy).strip() else float("nan")
        scaled_legacy = legacy_f * lambda_h15 if not math.isnan(legacy_f) else float("nan")
        return {
            "date": date_text,
            "quarter": _quarter_label(date_text),
            "legacy_h15_interest_proxy_mil": legacy_f,
            "scaled_h15_at_lambda_mil": scaled_legacy,
            "component_anchored_interest_mil": component_f,
            "bank_tier2_regression_interest_proxy_mil": proxy_f,
            "method_tier": row.get("method_tier", ""),
        }

    seam_dates_2010 = ["2009-12-31", "2010-03-31", "2010-06-30", "2010-09-30"]
    seam_dates_2022 = ["2021-09-30", "2021-12-31", "2022-03-31", "2022-06-30"]
    seam_rows = [_diag(d) for d in (*seam_dates_2010, *seam_dates_2022)]

    audit_rows: list[dict[str, Any]] = []
    audit_rows.append({
        "field": "lambda_H15_bank_sector",
        "value": lambda_h15,
        "definition": "Median ratio (component_anchored / legacy_H15) over the first 5 years of overlap; bank sector.",
        "source": str(backcast_path.relative_to(ROOT.parent.parent)) if ROOT.parent.parent in backcast_path.parents else str(backcast_path),
        "note": "From tdcest/src/tdc_estimator/tier2_regression_backcast.py::_scale_ratio.",
    })
    audit_rows.append({
        "field": "lambda_H15_bank_sector_calibration_window",
        "value": f"{_quarter_label(overlap_start)} to {_quarter_label(calibration_end)}",
        "definition": "Overlap window used to calibrate lambda_H15: first five years where both component_anchored_interest_mil and legacy_h15_interest_proxy_mil are non-null and the legacy is non-zero.",
        "source": str(backcast_path),
        "note": f"Calibration sample size: {len(calibration_dates)} quarters. Overall overlap rows: {len(overlap_dates)}.",
    })

    for tier in ("pre_component_h15_scaled_backcast", "component_pool_wamest_bucket_backcast", "constrained_component"):
        if tier in segments:
            audit_rows.append({
                "field": f"segment_{tier}",
                "value": f"{_quarter_label(segments[tier]['first_date'])} to {_quarter_label(segments[tier]['last_date'])}",
                "definition": {
                    "pre_component_h15_scaled_backcast": "Pre-2010 H.15 fallback: legacy bank Treasury-interest proxy scaled by lambda_H15.",
                    "component_pool_wamest_bucket_backcast": "Component-pool / maturity-bucket backcast using Z.1 bank Treasury holdings allocated by current WAMEST/H.15 coupon and bill weights.",
                    "constrained_component": "Modern constrained component using directly observed sector-bucket interest accruals.",
                }[tier],
                "source": str(backcast_path),
                "note": "method_tier values are written by tdcest tier2_regression_backcast.py.",
            })

    audit_rows.append({
        "field": "legacy_h15_input_columns",
        "value": "bank_tsy_coupon_interest_proxy + bank_tsy_bill_discount_interest_proxy",
        "definition": "Legacy bank Treasury-interest proxies built from Z.1 bank Treasury holdings by maturity bucket multiplied by H.15 nominal Treasury constant-maturity yields (coupon series) and H.15 secondary-market bill discount rates (bill series).",
        "source": "tdcest/src/tdc_estimator/sector_coupon.py and tdcest/src/tdc_estimator/tier2_interest_component_candidate.py",
        "note": "Underlying inputs: FRED H.15 curves (DGS* constant-maturity series) and Z.1 Treasury holdings by sector and maturity (Z.1/L.213-class series).",
    })
    audit_rows.append({
        "field": "component_anchored_input",
        "value": "component_anchored_interest_mil (per-quarter sum across maturity-bucket components)",
        "definition": "Component-anchored bank Treasury-interest series: sums coupon_accrual, bill_amortized_discount, and frn_accrued_interest components allocated using current WAMEST/H.15 coupon and bill weights to the constrained sector-bucket holdings.",
        "source": "tdcest/src/tdc_estimator/tier2_interest_component_candidate.py",
        "note": "Bank sector_group rows only; equivalent expressions exist for ROW and credit_union.",
    })

    for row in seam_rows:
        audit_rows.append({
            "field": f"seam_{row['quarter']}",
            "value": row["bank_tier2_regression_interest_proxy_mil"],
            "definition": (
                f"bank_tier2_regression_interest_proxy at {row['quarter']} (method_tier={row['method_tier']}); "
                f"scaled H.15 proxy at lambda_H15={lambda_h15:.4f} would have been {row['scaled_h15_at_lambda_mil']:.1f}; "
                f"raw component_anchored = {row['component_anchored_interest_mil']:.1f}."
            ),
            "source": str(backcast_path),
            "note": "All values in millions of dollars.",
        })

    # Modern vs long-history overlap comparison: pull from method-decision table if available.
    method_decision_path = ROOT / "output" / "reports" / "tier2_method_decision_table.csv"
    modern_long_summary: list[dict[str, Any]] = []
    if method_decision_path.exists():
        method_rows = _read_csv(method_decision_path)
        for row in method_rows:
            label = row.get("treatment_label", "")
            if label not in {"modern_canonical_di_mmf_rrp_short", PRIMARY_TREATMENT_LABEL}:
                continue
            modern_long_summary.append({
                "treatment_label": label,
                "role": row.get("role", ""),
                "coverage": row.get("coverage", ""),
                "h0_deposits_beta_per_1_tdc": row.get("h0_deposits_beta", ""),
                "h0_deposits_p": row.get("h0_deposits_p", ""),
                "h0_deposits_n": row.get("h0_deposits_n", ""),
            })
        if modern_long_summary:
            audit_rows.append({
                "field": "modern_vs_long_h0_deposits_overlap",
                "value": "; ".join(
                    f"{row['treatment_label']}: beta={row['h0_deposits_beta_per_1_tdc']} per $1 TDC (p={row['h0_deposits_p']}, n={row['h0_deposits_n']}, coverage={row['coverage']})"
                    for row in modern_long_summary
                ),
                "definition": "h=0 matched-deposits coefficient comparison for the modern canonical DI/MMF/RRP short row (2022Q1-2025Q4) and the regression-grade long-history bank row (2002Q1-2025Q4).",
                "source": str(method_decision_path.relative_to(ROOT)),
                "note": "Same K=100 pinned-factor control surface for both rows. The modern row is identical on the 2022Q1+ overlap because the constrained component is what underlies it; see seam_2022Q1 row.",
            })
    metadata: dict[str, Any] = {
        "backcast_source": str(backcast_path),
        "lambda_h15_bank_sector": lambda_h15,
        "calibration_window": f"{_quarter_label(overlap_start)}-{_quarter_label(calibration_end)}",
        "calibration_window_dates": [overlap_start, calibration_end],
        "calibration_window_quarter_count": len(calibration_dates),
        "modern_vs_long_summary": modern_long_summary,
        "seam_2010_dates": seam_dates_2010,
        "seam_2022_dates": seam_dates_2022,
    }
    return audit_rows, metadata


# --------------------------------------------------------------------------- plumbing magnitudes


PLUMBING_OUTCOMES = [
    "reserve_balances_qoq",
    "total_reserve_balances_plus_foreign_official_qoq",
    "reserve_balances_net_fed_treasury_qoq",
    "total_reserves_plus_foreign_official_net_fed_treasury_qoq",
    "foreign_official_deposits_qoq",
    "tga_balance_qoq",
    "on_rrp_balance_qoq",
    "mmf_on_rrp_plumbing_absorption_qoq",
]


def _plumbing_magnitudes() -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Pull h=0 plumbing magnitudes for the primary long-history TDC treatment from existing model outputs."""
    state_path = ROOT / "output" / "models" / "tier2_credit_causality_state_estimates.csv"
    missing40_path = ROOT / "output" / "models" / "tier2_missing40_residual_attribution_estimates.csv"
    found: dict[tuple[str, str], dict[str, Any]] = {}
    not_found: list[str] = []
    sources_used: list[str] = []

    if state_path.exists():
        sources_used.append(str(state_path.relative_to(ROOT)))
        for row in _read_csv(state_path):
            if row.get("surface") != "pooled_baseline":
                continue
            if row.get("treatment_label") != PRIMARY_TREATMENT_LABEL:
                continue
            if row.get("sample_label") != "full_available":
                continue
            try:
                horizon = int(row.get("horizon", 0))
            except ValueError:
                continue
            if horizon != 0:
                continue
            outcome = row.get("outcome", "")
            key = ("state_pooled_baseline", outcome)
            found[key] = row

    if missing40_path.exists():
        sources_used.append(str(missing40_path.relative_to(ROOT)))
        for row in _read_csv(missing40_path):
            try:
                horizon = int(row.get("horizon", 0))
            except ValueError:
                continue
            if horizon != 0:
                continue
            outcome = row.get("outcome", "")
            key = ("missing40", outcome)
            found.setdefault(key, row)

    rows: list[dict[str, Any]] = []
    for outcome in PLUMBING_OUTCOMES:
        key = ("state_pooled_baseline", outcome)
        if key not in found:
            key = ("missing40", outcome)
            if key not in found:
                not_found.append(outcome)
                continue
        row = found[key]
        beta = float(row.get("beta") or 0.0)
        # missing40 attribution rows carry per-outcome unit overrides for plumbing series like
        # tga_balance_qoq, on_rrp_balance_qoq, and mmf_on_rrp_plumbing_absorption_qoq in the
        # attribution_effect_per_100b_tdc column (which is always in usd_billions). State-pooled
        # rows store the corresponding scaling in effect_per_100b_tdc but only when the shared
        # _normalization map already had the outcome registered as dollar-per-dollar. Prefer the
        # attribution-decorated value first, then fall back to the pooled-baseline column when
        # the unit label says it is already in usd_billions, and only synthesize from beta as a
        # last resort.
        attribution_text = str(row.get("attribution_effect_per_100b_tdc", "")).strip()
        attribution_unit = str(row.get("attribution_effect_unit", "")).strip()
        effect_per_100b_text = str(row.get("effect_per_100b_tdc", "")).strip()
        effect_unit_text = str(row.get("effect_per_100b_unit", "")).strip()
        if attribution_text and attribution_unit == "usd_billions_per_100b_tdc":
            effect_unit = attribution_unit
            effect_value = float(attribution_text)
            beta_per_1_tdc = effect_value / 100.0
        elif effect_per_100b_text and effect_unit_text == "usd_billions_per_100b_tdc":
            effect_unit = effect_unit_text
            effect_value = float(effect_per_100b_text)
            beta_per_1_tdc = effect_value / 100.0
        else:
            _, multiplier = _normalization(outcome)
            effect_unit, effect_value = _effect_per_100b(outcome, beta)
            beta_per_1_tdc = beta * multiplier
        rows.append({
            "outcome": outcome,
            "source_surface": key[0],
            "h0_beta_per_1_tdc": beta_per_1_tdc,
            "h0_effect_per_100b_tdc": effect_value,
            "h0_effect_unit": effect_unit,
            "p_value": row.get("p_value_normal", ""),
            "n": row.get("n", ""),
            "treatment_label": PRIMARY_TREATMENT_LABEL,
            "treatment_id": row.get("treatment_id", PRIMARY_TREATMENT_ID),
        })
    metadata = {
        "primary_treatment_label": PRIMARY_TREATMENT_LABEL,
        "sources_used": sources_used,
        "outcomes_missing_from_existing_outputs": not_found,
    }
    return rows, not_found, metadata


# --------------------------------------------------------------------------- markdown writers


def _write_lead_placebo_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Submission Lead-Placebo Coefficient Table",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        "Per-outcome, per-lead h=0 coefficients with Newey-West (lag 1) standard errors and two-sided normal p-values "
        f"for the primary long-history TDC treatment `{PRIMARY_TREATMENT_LABEL}` "
        f"(`{PRIMARY_TREATMENT_ID}`). Effects are reported in $ billions per +$100B TDC, consistent with the manuscript.",
        "",
        "Two scenarios are reported:",
        "",
        "- `baseline_controls`: GDP, gdp_deflator, FEDFUNDS, TOTRESNS, the pre-component H.15 method-tier dummy, and the K=100 four-factor tail (capped at 12 controls).",
        "- `selected_credit_rate_risk_lags`: the same controls plus selected lagged credit and Treasury-rate variables identified by the existing lead-diagnostic predictability ranking.",
        "",
        "Future-treatment placebos are estimated for leads 1-4 by regressing each outcome at time t on the TDC value at t+L. The contemporaneous h=0 row is included for reference. All standard errors and confidence intervals are reported in the same effect-per-+$100B-TDC scale as the point estimate.",
        "",
        "## Coefficients (effect per +$100B TDC, $ billions)",
        "",
        "| scenario | outcome | h | effect | SE | p | 95% CI | n |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    sort_outcome_order = {name: idx for idx, name in enumerate(LEAD_OUTCOMES)}
    sort_scenario_order = {"baseline_controls": 0, "selected_credit_rate_risk_lags": 1}
    ordered = sorted(
        rows,
        key=lambda r: (
            sort_scenario_order.get(str(r.get("scenario", "")), 99),
            sort_outcome_order.get(str(r.get("outcome", "")), 99),
            int(r.get("lead_quarters", 0)),
        ),
    )
    for r in ordered:
        se_per_100b = r.get("effect_per_100b_se")
        if se_per_100b in (None, ""):
            ci = ""
        else:
            ci = f"[{_format_value(r.get('effect_per_100b_lower95'))}, {_format_value(r.get('effect_per_100b_upper95'))}]"
        lines.append(
            f"| {r['scenario']} | {r['outcome']} | {r['horizon_label']} | "
            f"{_format_value(r.get('effect_per_100b_tdc'))} | {_format_value(se_per_100b)} | "
            f"{_format_p(r.get('p_value'))} | {ci} | {r.get('n', '')} |"
        )
    lines.extend([
        "",
        f"Leads estimated: {', '.join(f'h=-{l}' for l in LEAD_HORIZONS)}. Contemporaneous h=0 included for reference.",
        "",
        "## Interpretation Rule",
        "",
        "- Significant negative leads for credit outcomes (mortgages, strict loan core, consumer credit) under `baseline_controls` are absorbed substantially when selected credit/rate-risk lags are added. Reporting both columns documents the predictability problem rather than concealing it.",
        "- The matched-deposit lead-placebo coefficients should be small relative to the contemporaneous h=0 deposit response if the headline pass-through is not driven by anticipation.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_hac_bandwidth_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Submission HAC Bandwidth Sensitivity",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        f"Headline h=0 matched-deposit coefficient on `{PRIMARY_TREATMENT_ID}` under the selected credit/rate-risk lag specification, "
        "varying the Newey-West HAC lag bandwidth. The point estimate and sample are identical across rows; only the standard error and p-value vary.",
        "",
        "| outcome | spec | HAC lags | effect per +$100B | SE | p | 95% CI | n |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for r in rows:
        ci = f"[{_format_value(r.get('effect_per_100b_lower95'))}, {_format_value(r.get('effect_per_100b_upper95'))}]"
        lines.append(
            f"| {r['outcome']} | {r['spec']} | {r['covariance_lags']} | "
            f"{_format_value(r['effect_per_100b_tdc'])} | {_format_value(r['effect_per_100b_se'])} | {_format_p(r['p_value'])} | "
            f"{ci} | {r['n']} |"
        )
    lines.extend([
        "",
        "## Interpretation Rule",
        "",
        "- The matched-deposit coefficient should remain a statistically resolved positive at conventional bandwidths.",
        "- If the p-value crosses 0.05 only at very wide bandwidths, the headline survives an aggressive HAC choice. The point estimate does not move with bandwidth.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_factor_tail_markdown(path: Path, rows: list[dict[str, Any]], *, factor_count: int) -> None:
    lines = [
        "# Submission Factor-Tail Robustness",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        f"h=0 coefficients on `{PRIMARY_TREATMENT_ID}` for the Table 1 outcome set under the selected credit/rate-risk lag specification. "
        f"Two variants: K=100 / {factor_count}-factor tail (preferred) and K=0 / no factor controls.",
        "",
        "| variant | outcome | effect per +$100B | SE | p | 95% CI | n |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for r in rows:
        ci = f"[{_format_value(r.get('effect_per_100b_lower95'))}, {_format_value(r.get('effect_per_100b_upper95'))}]"
        lines.append(
            f"| {r['variant']} | {r['outcome']} | {_format_value(r['effect_per_100b_tdc'])} | "
            f"{_format_value(r['effect_per_100b_se'])} | {_format_p(r['p_value'])} | {ci} | {r['n']} |"
        )
    lines.extend([
        "",
        "## Interpretation Rule",
        "",
        "- The matched-deposit pass-through should survive removal of the factor tail. Magnitudes and signs of the credit/residual outcomes provide a robustness reading for the same.",
        "- If a credit-outcome sign flips when the factor tail is removed, treat the credit margin as factor-screen-sensitive and avoid making a clean causal claim about it.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_splice_markdown(path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    lines = [
        "# Submission Splice / Construction Audit",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        "Construction audit for the long-history bank Treasury-interest series that underlies the preferred long-history "
        f"TDC treatment `{PRIMARY_TREATMENT_LABEL}`. Bank sector only.",
        "",
    ]
    lambda_value = metadata.get("lambda_h15_bank_sector")
    if lambda_value is not None and not (isinstance(lambda_value, float) and math.isnan(lambda_value)):
        lines.append(f"- **lambda_H15 (bank sector):** `{lambda_value:.6f}`.")
    cal_window = metadata.get("calibration_window")
    if cal_window:
        lines.append(f"- **Calibration window:** {cal_window} ({metadata.get('calibration_window_quarter_count', 0)} quarters).")
    lines.extend([
        "",
        "## Field Detail",
        "",
        "| field | value | definition | source |",
        "| --- | --- | --- | --- |",
    ])
    for r in rows:
        value_text = r.get("value", "")
        if isinstance(value_text, float):
            if math.isnan(value_text):
                value_text = ""
            elif abs(value_text) >= 1000:
                value_text = f"{value_text:,.1f}"
            else:
                value_text = f"{value_text:.4f}"
        lines.append(
            f"| `{r['field']}` | {value_text} | {r.get('definition', '')} | `{r.get('source', '')}` |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- `lambda_H15` is the bank-sector median ratio (component_anchored / legacy_H15) over the first five years of overlap "
        "between the component-anchored series and the legacy H.15-based proxy. It is applied multiplicatively to scale the legacy "
        "H.15 proxy back to pre-2010 dates.",
        "- Pre-2010 evidence grade is `low_medium`; component-pool segment is `medium`; constrained-component segment (2022Q1+) is `medium_high`.",
        "- Reporting `lambda_H15` and the seam diagnostic explicitly is intended as appendix disclosure, not as identification.",
        "- The 2010Q1 to 2010Q2 seam shows the method change from scaled-H.15 to component-pool. The 2021Q4 to 2022Q1 transition is a method-label change only; the value series is continuous because both segments draw on `component_anchored_interest_mil`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plumbing_markdown(path: Path, rows: list[dict[str, Any]], not_found: list[str]) -> None:
    lines = [
        "# Submission Plumbing Magnitudes",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        "Regression-derived h=0 plumbing magnitudes for the primary long-history TDC treatment "
        f"`{PRIMARY_TREATMENT_LABEL}` (`{PRIMARY_TREATMENT_ID}`). All effects are normalized to "
        "$ billions per +$100B TDC.",
        "",
        "| outcome | h0 effect (per +$100B) | beta per $1 TDC | p | n | source |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['outcome']}` | {_format_value(r['h0_effect_per_100b_tdc'])} | "
            f"{_format_value(r['h0_beta_per_1_tdc'])} | {_format_p(r['p_value'])} | {r['n']} | {r['source_surface']} |"
        )
    if not_found:
        lines.append("")
        lines.append(f"Outcomes requested but not present in existing pipeline outputs: {', '.join(f'`{x}`' for x in not_found)}.")
    lines.extend([
        "",
        "## Demotion Note",
        "",
        "Regression magnitudes for bank reserves, bank + foreign-official Fed deposits, reserve balances net Fed Treasury holdings, "
        "ON-RRP attribution-style impulse responses, foreign-official Fed deposits, and the TGA cash drain are produced by the existing "
        "Tier 2 state-dependent causality pipeline and the missing-40 attribution pipeline, and they are defensible at the headline "
        "K=100 four-factor surface.",
        "",
        "Descriptive prose about *how much* the Treasury refilled the TGA, *how much* MMF demand absorbed, or *how much* ON-RRP dropped "
        "in a given window — i.e. literal flow magnitudes attributed to specific issuance/demand legs — is not produced by this pipeline. "
        "Any such phrases in the manuscript should be either:",
        "",
        "1. Replaced with the regression magnitudes in the table above (which are co-movement / incidence, not attribution).",
        "2. Demoted to a footnote that says these are illustrative reserve-management commentary, not pipeline-validated quantities.",
        "",
        "Do not promote any descriptive refill-share number to a tabled magnitude unless it is sourced from an explicit, pre-registered calculation.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- main


def main() -> int:
    (paths, rows, control_ids, factor_count, screened_count,
     selected_credit_lags, selected_rate_lags) = _build_inputs()

    primary_spec = BRIDGE_TREATMENTS[PRIMARY_TREATMENT_LABEL]
    base_controls = _active_controls(control_ids, primary_spec)

    # 1. Lead-placebo coefficient table
    lead_rows = _build_lead_placebo_table(
        rows,
        base_controls=base_controls,
        selected_credit_lags=selected_credit_lags,
        selected_rate_lags=selected_rate_lags,
    )
    lead_csv = paths.reports / "submission_lead_placebo_coefficients.csv"
    _write_csv(lead_csv, lead_rows)
    _write_lead_placebo_markdown(paths.reports / "submission_lead_placebo_coefficients.md", lead_rows)

    # 2. HAC bandwidth sensitivity
    hac_rows = _build_hac_bandwidth_table(
        rows,
        base_controls=base_controls,
        selected_credit_lags=selected_credit_lags,
        selected_rate_lags=selected_rate_lags,
    )
    hac_csv = paths.reports / "submission_hac_bandwidth_sensitivity.csv"
    _write_csv(hac_csv, hac_rows)
    _write_hac_bandwidth_markdown(paths.reports / "submission_hac_bandwidth_sensitivity.md", hac_rows)

    # 3. Factor-tail robustness
    factor_rows = _build_factor_tail_table(
        rows,
        base_controls=base_controls,
        selected_credit_lags=selected_credit_lags,
        selected_rate_lags=selected_rate_lags,
        factor_count=factor_count,
    )
    factor_csv = paths.reports / "submission_factor_tail_robustness.csv"
    _write_csv(factor_csv, factor_rows)
    _write_factor_tail_markdown(paths.reports / "submission_factor_tail_robustness.md", factor_rows, factor_count=factor_count)

    # 4. Splice / construction audit
    splice_rows, splice_metadata = _build_splice_audit()
    splice_csv = paths.reports / "submission_splice_construction_audit.csv"
    _write_csv(splice_csv, splice_rows)
    _write_splice_markdown(paths.reports / "submission_splice_construction_audit.md", splice_rows, splice_metadata)

    # 5. Plumbing magnitudes
    plumbing_rows, not_found, plumbing_metadata = _plumbing_magnitudes()
    plumbing_csv = paths.reports / "submission_plumbing_magnitudes.csv"
    _write_csv(plumbing_csv, plumbing_rows)
    _write_plumbing_markdown(paths.reports / "submission_plumbing_magnitudes.md", plumbing_rows, not_found)

    summary_path = paths.manifests / "submission_appendix_diagnostics_summary.json"
    write_json(
        summary_path,
        {
            "generated_at": utc_now_iso(),
            "anchor_job_id": ANCHOR_JOB_ID,
            "k_screened": K_SCREENED,
            "factor_count": factor_count,
            "screened_count": screened_count,
            "control_policy_mode": CONTROL_POLICY_MODE,
            "primary_treatment_label": PRIMARY_TREATMENT_LABEL,
            "primary_treatment_id": PRIMARY_TREATMENT_ID,
            "lead_horizons": LEAD_HORIZONS,
            "hac_bandwidths": HAC_BANDWIDTHS,
            "lead_placebo_csv": str(lead_csv),
            "hac_csv": str(hac_csv),
            "factor_tail_csv": str(factor_csv),
            "splice_csv": str(splice_csv),
            "plumbing_csv": str(plumbing_csv),
            "lead_placebo_rows": len(lead_rows),
            "hac_rows": len(hac_rows),
            "factor_tail_rows": len(factor_rows),
            "splice_rows": len(splice_rows),
            "plumbing_rows": len(plumbing_rows),
            "splice_metadata": splice_metadata,
            "plumbing_metadata": plumbing_metadata,
        },
    )
    print(json.dumps({
        "lead_placebo_csv": str(lead_csv),
        "hac_csv": str(hac_csv),
        "factor_tail_csv": str(factor_csv),
        "splice_csv": str(splice_csv),
        "plumbing_csv": str(plumbing_csv),
        "summary_path": str(summary_path),
        "rows": {
            "lead_placebo": len(lead_rows),
            "hac": len(hac_rows),
            "factor_tail": len(factor_rows),
            "splice": len(splice_rows),
            "plumbing": len(plumbing_rows),
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
