"""Offset and regime diagnostics for Tier 2 deposit pass-through.

The output is deliberately descriptive. It checks whether low or changing
deposit pass-through is paired with non-TDC deposit residuals, selected credit
aggregates, bank asset rows, or liquidity-plumbing rows. It does not identify a
structural offset channel.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_submission_appendix_diagnostics import (  # noqa: E402
    PRIMARY_RESIDUAL_ID,
    PRIMARY_TREATMENT_ID,
    _build_inputs,
    _effect_per_100b,
    _fit_lp,
    _normal_p_two_sided,
    _normalization,
    _scenario_controls,
)

JOB_ID = "tier2_pass_through_offset_diagnostics"
MIN_BETA_OBSERVATIONS = 24
CANONICAL_SPLICE_CONTROL = "tier2_regression_bank_row_tier_pre_component_h15_scaled"

EPISODE_OUTPUT = ROOT / "output/reports/tier2_pass_through_offset_episode_betas.csv"
LEVEL_OUTPUT = ROOT / "output/reports/tier2_pass_through_offset_level_summary.csv"
CORRELATION_OUTPUT = ROOT / "output/reports/tier2_pass_through_offset_correlations.csv"
IDENTITY_OUTPUT = ROOT / "output/reports/tier2_pass_through_offset_identity_windows.csv"
LEAD_LAG_OUTPUT = ROOT / "output/reports/tier2_pass_through_offset_lead_lag_correlations.csv"
JACKKNIFE_OUTPUT = ROOT / "output/reports/tier2_pass_through_offset_2020_2021_jackknife.csv"
ROLLING_FEATURE_OUTPUT = ROOT / "output/reports/tier2_pass_through_offset_rolling_beta_features.csv"
ROLLING_CORRELATE_OUTPUT = ROOT / "output/reports/tier2_pass_through_offset_rolling_beta_correlates.csv"
REPORT_OUTPUT = ROOT / "output/reports/tier2_pass_through_offset_diagnostics.md"
MANIFEST_OUTPUT = ROOT / "output/manifests/tier2_pass_through_offset_diagnostics_summary.json"
ROLLING_REGRESSION_OUTPUT = ROOT / "output/models/tier2_rolling_selected_credit_rate_pass_through_estimates.csv"

OUTCOMES = [
    {"group": "deposit_identity", "id": "matched_total_deposits", "label": "Matched deposits"},
    {"group": "deposit_identity", "id": PRIMARY_RESIDUAL_ID, "label": "Same-treatment other component"},
    {"group": "credit", "id": "tdcpass_strict_loan_core_min_qoq", "label": "Strict loan core"},
    {"group": "credit", "id": "tdcpass_strict_loan_mortgages_qoq", "label": "Strict mortgages"},
    {"group": "credit", "id": "tdcpass_strict_loan_consumer_credit_qoq", "label": "Strict consumer credit"},
    {"group": "credit", "id": "bank_credit_qoq", "label": "Bank credit"},
    {"group": "bank_asset", "id": "bank_treasury_securities_qoq", "label": "Bank Treasury securities"},
    {"group": "bank_asset", "id": "bank_treasury_securities_transactions_qoq", "label": "Bank Treasury transactions"},
    {"group": "bank_asset", "id": "bank_treasury_agency_securities_qoq", "label": "Bank Treasury and agency securities"},
    {"group": "bank_asset", "id": "bank_non_treasury_securities_qoq", "label": "Bank non-Treasury securities"},
    {"group": "plumbing", "id": "reserve_balances_qoq", "label": "Reserve balances"},
    {"group": "plumbing", "id": "tga_balance_qoq", "label": "TGA balance"},
    {"group": "plumbing", "id": "on_rrp_balance_qoq", "label": "ON RRP balance"},
    {"group": "plumbing", "id": "mmf_on_rrp_plumbing_absorption_qoq", "label": "MMF/ON RRP absorption"},
    {"group": "plumbing", "id": "reserve_balances_net_fed_treasury_qoq", "label": "Reserves net Fed Treasury"},
]
FITTED_OUTCOME_IDS = {
    "matched_total_deposits",
    PRIMARY_RESIDUAL_ID,
    "tdcpass_strict_loan_core_min_qoq",
    "tdcpass_strict_loan_mortgages_qoq",
    "tdcpass_strict_loan_consumer_credit_qoq",
    "bank_credit_qoq",
    "bank_treasury_securities_qoq",
}
ROLLING_FEATURE_SPECS = [
    *OUTCOMES,
    {"group": "rate", "id": "FEDFUNDS", "label": "Federal funds rate"},
    {"group": "rate", "id": "dgs2", "label": "2-year Treasury yield"},
    {"group": "rate", "id": "dgs10", "label": "10-year Treasury yield"},
    {"group": "rate", "id": "dgs10_2y_spread", "label": "10-year minus 2-year spread"},
    {"group": "rate", "id": "baa_aaa", "label": "BAA minus AAA spread"},
    {"group": "rate", "id": "BAMLH0A0HYM2", "label": "High-yield spread"},
    {"group": "bank_asset", "id": "bank_consumer_loans_qoq", "label": "Bank consumer loans"},
    {"group": "bank_asset", "id": "bank_business_loans_qoq", "label": "Bank business loans"},
    {"group": "bank_asset", "id": "bank_real_estate_loans_qoq", "label": "Bank real-estate loans"},
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _quarter_key(quarter: str) -> tuple[int, int]:
    year_text, q_text = quarter.split("Q", 1)
    return int(year_text), int(q_text)


def _between(quarter: str, start: str, end: str) -> bool:
    key = _quarter_key(quarter)
    return _quarter_key(start) <= key <= _quarter_key(end)


def _paired_values(
    rows: list[dict[str, str]],
    *,
    treatment_id: str,
    outcome_id: str,
) -> tuple[list[float], list[float]]:
    treatments: list[float] = []
    outcomes: list[float] = []
    for row in rows:
        treatment = _safe_float(row.get(treatment_id, ""))
        outcome = _safe_float(row.get(outcome_id, ""))
        if treatment is None or outcome is None:
            continue
        treatments.append(treatment)
        outcomes.append(outcome)
    return treatments, outcomes


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
    return sum(lval * rval for lval, rval in zip(left_centered, right_centered)) / math.sqrt(left_ss * right_ss)


def _periods() -> list[tuple[str, Callable[[str], bool], str]]:
    return [
        ("full_available", lambda q: True, "All available quarters."),
        ("pre_2020", lambda q: _quarter_key(q) < _quarter_key("2020Q1"), "Quarters before 2020Q1."),
        ("covid_2020_2021", lambda q: _between(q, "2020Q1", "2021Q4"), "2020Q1 through 2021Q4."),
        ("post_2022", lambda q: _quarter_key(q) >= _quarter_key("2022Q1"), "2022Q1 onward."),
        (
            "exclude_2020_2021",
            lambda q: not _between(q, "2020Q1", "2021Q4"),
            "Full sample excluding 2020Q1 through 2021Q4.",
        ),
        (
            "exclude_transition_2019_2021",
            lambda q: not _between(q, "2019Q1", "2021Q4"),
            "Full sample excluding 2019Q1 through 2021Q4.",
        ),
        ("exclude_gfc", lambda q: not _between(q, "2007Q4", "2009Q2"), "Full sample excluding 2007Q4 through 2009Q2."),
        (
            "exclude_gfc_covid_transition",
            lambda q: not _between(q, "2007Q4", "2009Q2") and not _between(q, "2019Q1", "2021Q4"),
            "Full sample excluding 2007Q4-2009Q2 and 2019Q1-2021Q4.",
        ),
    ]


def _subset_rows(rows: list[dict[str, str]], predicate: Callable[[str], bool]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("quarter") and predicate(str(row["quarter"]))]


def _level_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for period, predicate, description in _periods()[:4]:
        subset = _subset_rows(rows, predicate)
        values = [
            (str(row.get("quarter", "")), value)
            for row in subset
            if (value := _safe_float(row.get(PRIMARY_TREATMENT_ID, ""))) is not None
        ]
        if not values:
            continue
        numeric = [value for _, value in values]
        sorted_numeric = sorted(numeric)
        mid = len(sorted_numeric) // 2
        median = sorted_numeric[mid] if len(sorted_numeric) % 2 else (sorted_numeric[mid - 1] + sorted_numeric[mid]) / 2.0
        max_q, max_value = max(values, key=lambda item: item[1])
        min_q, min_value = min(values, key=lambda item: item[1])
        mean = sum(numeric) / len(numeric)
        variance = sum((value - mean) ** 2 for value in numeric) / (len(numeric) - 1) if len(numeric) > 1 else 0.0
        output.append(
            {
                "job_id": JOB_ID,
                "period": period,
                "description": description,
                "n": len(numeric),
                "mean_mil": mean,
                "median_mil": median,
                "mean_abs_mil": sum(abs(value) for value in numeric) / len(numeric),
                "sd_mil": math.sqrt(variance),
                "min_mil": min_value,
                "min_quarter": min_q,
                "max_mil": max_value,
                "max_quarter": max_q,
                "positive_quarters": sum(1 for value in numeric if value > 0),
            }
        )
    return output


def _sample_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _sample_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _correlation_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for period, predicate, description in _periods()[:4]:
        subset = _subset_rows(rows, predicate)
        for spec in OUTCOMES:
            treatments, outcomes = _paired_values(subset, treatment_id=PRIMARY_TREATMENT_ID, outcome_id=spec["id"])
            correlation = _pearson(treatments, outcomes)
            output.append(
                {
                    "job_id": JOB_ID,
                    "period": period,
                    "description": description,
                    "outcome_group": spec["group"],
                    "outcome": spec["id"],
                    "outcome_label": spec["label"],
                    "correlation": "" if correlation is None else correlation,
                    "n": len(treatments),
                    "diagnostic_type": "period_pearson_correlation",
                    "diagnostic_role": "descriptive_comovement_only",
                    "claim_boundary": "Not a pass-through magnitude and not a causal estimate.",
                }
            )
    return output


def _estimate_episode(
    *,
    rows: list[dict[str, str]],
    controls: list[str],
    period: str,
    description: str,
    outcome_spec: dict[str, str],
) -> dict[str, Any]:
    treatments, _outcomes = _paired_values(rows, treatment_id=PRIMARY_TREATMENT_ID, outcome_id=outcome_spec["id"])
    base = {
        "job_id": JOB_ID,
        "period": period,
        "description": description,
        "outcome_group": outcome_spec["group"],
        "outcome": outcome_spec["id"],
        "outcome_label": outcome_spec["label"],
        "treatment_id": PRIMARY_TREATMENT_ID,
        "n_pairs": len(treatments),
        "status": "estimated",
    }
    if len(treatments) < MIN_BETA_OBSERVATIONS:
        base["status"] = "insufficient_observations"
        base["claim_boundary"] = "Too few paired observations for the selected-lag regression diagnostic."
        return base
    try:
        fit, n, used, rejected = _fit_lp(
            rows,
            treatment_id=PRIMARY_TREATMENT_ID,
            outcome_id=outcome_spec["id"],
            horizon=0,
            control_ids=controls,
            covariance_lags=1,
        )
    except ValueError as exc:
        base["status"] = "not_estimable"
        base["claim_boundary"] = f"Selected-lag regression was not estimable: {exc}"
        return base
    beta = fit.beta[1]
    se = fit.ses[1]
    z = beta / se if se > 0 else None
    p_value = "" if z is None else _normal_p_two_sided(z)
    _, multiplier = _normalization(outcome_spec["id"])
    effect_unit, effect_value = _effect_per_100b(outcome_spec["id"], beta)
    normalized_se = se * multiplier
    base.update(
        {
            "n": n,
            "beta": beta,
            "se": se,
            "z_score": "" if z is None else z,
            "p_value_normal": p_value,
            "lower95": beta - 1.96 * se,
            "upper95": beta + 1.96 * se,
            "normalized_unit": "dollars_per_dollar_tdc",
            "normalized_beta": beta * multiplier,
            "normalized_se": normalized_se,
            "effect_per_100b_unit": effect_unit,
            "effect_per_100b_tdc": effect_value,
            "effect_per_100b_se": normalized_se * 100.0,
            "effect_per_100b_lower95": effect_value - 1.96 * normalized_se * 100.0,
            "effect_per_100b_upper95": effect_value + 1.96 * normalized_se * 100.0,
            "rsquared": fit.rsquared,
            "controls_used": ",".join(used),
            "controls_rejected": ",".join(rejected),
            "inference_method": "selected_credit_rate_lags_rank_aware",
            "covariance_estimator": "newey_west",
            "covariance_lags": 1,
            "claim_boundary": "Descriptive episode stability check; not a structural channel estimate.",
        }
    )
    return base


def _episode_rows(rows: list[dict[str, str]], controls: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    fitted_outcomes = [spec for spec in OUTCOMES if spec["id"] in FITTED_OUTCOME_IDS]
    for period, predicate, description in _periods():
        subset = _subset_rows(rows, predicate)
        for spec in fitted_outcomes:
            output.append(_estimate_episode(rows=subset, controls=controls, period=period, description=description, outcome_spec=spec))
    return output


def _canonical_controls(rows: list[dict[str, str]], controls: list[str]) -> list[str]:
    if CANONICAL_SPLICE_CONTROL in controls:
        return controls
    if not any(row.get(CANONICAL_SPLICE_CONTROL, "").strip() for row in rows):
        return controls
    updated = controls[:]
    try:
        insert_at = updated.index("TOTRESNS") + 1
    except ValueError:
        insert_at = 0
    updated.insert(insert_at, CANONICAL_SPLICE_CONTROL)
    return updated


def _shifted_pairs(
    rows: list[dict[str, str]],
    *,
    treatment_id: str,
    outcome_id: str,
    outcome_shift_quarters: int,
) -> tuple[list[float], list[float]]:
    ordered = sorted([row for row in rows if row.get("quarter")], key=lambda row: _quarter_key(str(row["quarter"])))
    treatment_values: list[float] = []
    outcome_values: list[float] = []
    for index, row in enumerate(ordered):
        shifted_index = index + outcome_shift_quarters
        if shifted_index < 0 or shifted_index >= len(ordered):
            continue
        treatment = _safe_float(row.get(treatment_id, ""))
        outcome = _safe_float(ordered[shifted_index].get(outcome_id, ""))
        if treatment is None or outcome is None:
            continue
        treatment_values.append(treatment)
        outcome_values.append(outcome)
    return treatment_values, outcome_values


def _lead_lag_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for spec in OUTCOMES:
        for shift in range(-4, 5):
            treatments, outcomes = _shifted_pairs(
                rows,
                treatment_id=PRIMARY_TREATMENT_ID,
                outcome_id=spec["id"],
                outcome_shift_quarters=shift,
            )
            correlation = _pearson(treatments, outcomes)
            output.append(
                {
                    "job_id": JOB_ID,
                    "outcome_group": spec["group"],
                    "outcome": spec["id"],
                    "outcome_label": spec["label"],
                    "outcome_shift_quarters": shift,
                    "timing_note": "positive shift means the outcome follows TDC; negative shift means the outcome leads TDC",
                    "correlation": "" if correlation is None else correlation,
                    "n": len(treatments),
                    "diagnostic_type": "lead_lag_pearson_correlation",
                    "diagnostic_role": "descriptive_timing_check_only",
                }
            )
    return output


def _identity_rows_from_rolling(regression_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_window: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in regression_rows:
        key = (str(row.get("window_start_quarter", "")), str(row.get("window_end_quarter", "")))
        by_window.setdefault(key, {})[str(row.get("outcome", ""))] = row
    output: list[dict[str, Any]] = []
    for (start, end), bucket in sorted(by_window.items()):
        deposit = bucket.get("matched_total_deposits")
        residual = bucket.get(PRIMARY_RESIDUAL_ID)
        if deposit is None or residual is None:
            continue
        deposit_beta = _safe_float(deposit.get("normalized_beta"))
        residual_beta = _safe_float(residual.get("normalized_beta"))
        if deposit_beta is None or residual_beta is None:
            continue
        output.append(
            {
                "job_id": JOB_ID,
                "window_start_quarter": start,
                "window_end_quarter": end,
                "deposit_beta_per_dollar_tdc": deposit_beta,
                "residual_beta_per_dollar_tdc": residual_beta,
                "deposit_minus_residual_minus_one": deposit_beta - residual_beta - 1.0,
                "identity_note": "For matched deposits = TDC + same-treatment other component, deposit beta should equal 1 + residual beta.",
            }
        )
    return output


def _jackknife_rows(rows: list[dict[str, str]], controls: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    drop_quarters = [f"{year}Q{quarter}" for year in (2020, 2021) for quarter in range(1, 5)]
    for drop_quarter in drop_quarters:
        subset = [row for row in rows if row.get("quarter") != drop_quarter]
        for spec in OUTCOMES[:2]:
            row = _estimate_episode(
                rows=subset,
                controls=controls,
                period=f"drop_{drop_quarter}",
                description=f"Full sample excluding {drop_quarter}.",
                outcome_spec=spec,
            )
            row["dropped_quarter"] = drop_quarter
            output.append(row)
    return output


def _rolling_deposit_beta_rows(regression_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in regression_rows:
        if row.get("outcome") != "matched_total_deposits":
            continue
        beta = _safe_float(row.get("normalized_beta"))
        if beta is None:
            continue
        output.append(
            {
                "window_start_quarter": str(row.get("window_start_quarter", "")),
                "window_end_quarter": str(row.get("window_end_quarter", "")),
                "deposit_beta_per_dollar_tdc": beta,
                "n": row.get("n", ""),
            }
        )
    return sorted(output, key=lambda row: (_quarter_key(row["window_start_quarter"]), _quarter_key(row["window_end_quarter"])))


def _window_share(rows: list[dict[str, str]], predicate: Callable[[str], bool]) -> float | None:
    quarters = [str(row.get("quarter", "")) for row in rows if row.get("quarter")]
    if not quarters:
        return None
    return sum(1 for quarter in quarters if predicate(quarter)) / len(quarters)


def _rolling_window_features(
    rows: list[dict[str, str]],
    rolling_beta_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for beta_row in rolling_beta_rows:
        start = str(beta_row["window_start_quarter"])
        end = str(beta_row["window_end_quarter"])
        subset = _subset_rows(rows, lambda quarter, start=start, end=end: _between(quarter, start, end))
        beta = float(beta_row["deposit_beta_per_dollar_tdc"])
        end_year, end_quarter = _quarter_key(end)
        time_index = end_year * 4 + end_quarter
        window_features = [
            ("window", "window_end_time_index", "Window end time index", time_index, len(subset), "mechanical_time_trend"),
            (
                "window",
                "share_2020_2021",
                "Share of quarters in 2020Q1-2021Q4",
                _window_share(subset, lambda quarter: _between(quarter, "2020Q1", "2021Q4")),
                len(subset),
                "regime_composition",
            ),
            (
                "window",
                "share_post_2020",
                "Share of quarters after 2020Q1",
                _window_share(subset, lambda quarter: _quarter_key(quarter) >= _quarter_key("2020Q1")),
                len(subset),
                "regime_composition",
            ),
            (
                "window",
                "share_post_2022",
                "Share of quarters after 2022Q1",
                _window_share(subset, lambda quarter: _quarter_key(quarter) >= _quarter_key("2022Q1")),
                len(subset),
                "regime_composition",
            ),
        ]
        treatment_values = [
            value
            for row in subset
            if (value := _safe_float(row.get(PRIMARY_TREATMENT_ID, ""))) is not None
        ]
        for feature_name, value in (
            ("tdc_mean_mil", _sample_mean(treatment_values)),
            ("tdc_mean_abs_mil", _sample_mean([abs(value) for value in treatment_values])),
            ("tdc_sd_mil", _sample_sd(treatment_values)),
            ("tdc_max_abs_mil", max([abs(value) for value in treatment_values], default=None)),
            ("tdc_positive_share", _sample_mean([1.0 if value > 0 else 0.0 for value in treatment_values])),
        ):
            window_features.append(("tdc_scale", feature_name, feature_name, value, len(treatment_values), "window_tdc_scale"))
        for group, feature_name, label, value, n, role in window_features:
            output.append(
                {
                    "job_id": JOB_ID,
                    "window_start_quarter": start,
                    "window_end_quarter": end,
                    "deposit_beta_per_dollar_tdc": beta,
                    "feature_group": group,
                    "feature_id": feature_name,
                    "feature_label": label,
                    "feature_stat": "window_value",
                    "feature_value": "" if value is None else value,
                    "feature_n": n,
                    "diagnostic_role": role,
                }
            )
        for spec in ROLLING_FEATURE_SPECS:
            values = [
                value
                for row in subset
                if (value := _safe_float(row.get(spec["id"], ""))) is not None
            ]
            treatments, outcomes = _paired_values(subset, treatment_id=PRIMARY_TREATMENT_ID, outcome_id=spec["id"])
            corr = _pearson(treatments, outcomes)
            for stat, value, n in (
                ("window_mean", _sample_mean(values), len(values)),
                ("window_sd", _sample_sd(values), len(values)),
                ("window_abs_mean", _sample_mean([abs(value) for value in values]), len(values)),
                ("tdc_feature_correlation", corr, len(treatments)),
            ):
                output.append(
                    {
                        "job_id": JOB_ID,
                        "window_start_quarter": start,
                        "window_end_quarter": end,
                        "deposit_beta_per_dollar_tdc": beta,
                        "feature_group": spec["group"],
                        "feature_id": spec["id"],
                        "feature_label": spec["label"],
                        "feature_stat": stat,
                        "feature_value": "" if value is None else value,
                        "feature_n": n,
                        "diagnostic_role": "rolling_beta_correlate_candidate",
                    }
                )
    return output


def _rolling_beta_correlates(feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    share_2020_by_window: dict[tuple[str, str], float] = {}
    for row in feature_rows:
        if row.get("feature_id") != "share_2020_2021" or row.get("feature_stat") != "window_value":
            continue
        value = _safe_float(row.get("feature_value"))
        if value is None:
            continue
        share_2020_by_window[(str(row.get("window_start_quarter", "")), str(row.get("window_end_quarter", "")))] = value
    buckets: dict[tuple[str, str, str, str], list[tuple[float, float, float | None]]] = {}
    exemplar: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in feature_rows:
        beta = _safe_float(row.get("deposit_beta_per_dollar_tdc"))
        value = _safe_float(row.get("feature_value"))
        if beta is None or value is None:
            continue
        key = (
            str(row.get("feature_group", "")),
            str(row.get("feature_id", "")),
            str(row.get("feature_label", "")),
            str(row.get("feature_stat", "")),
        )
        window = (str(row.get("window_start_quarter", "")), str(row.get("window_end_quarter", "")))
        buckets.setdefault(key, []).append((beta, value, share_2020_by_window.get(window)))
        exemplar.setdefault(key, row)
    output: list[dict[str, Any]] = []
    for key, pairs in buckets.items():
        if len(pairs) < 10:
            continue
        beta_values = [pair[0] for pair in pairs]
        feature_values = [pair[1] for pair in pairs]
        corr = _pearson(beta_values, feature_values)
        if corr is None:
            continue
        partial_pairs = [pair for pair in pairs if pair[2] is not None]
        partial_corr = None
        if len(partial_pairs) >= 10:
            partial_beta = _residualize_on_one([pair[0] for pair in partial_pairs], [float(pair[2]) for pair in partial_pairs])
            partial_feature = _residualize_on_one([pair[1] for pair in partial_pairs], [float(pair[2]) for pair in partial_pairs])
            partial_corr = _pearson(partial_beta, partial_feature)
        example = exemplar[key]
        output.append(
            {
                "job_id": JOB_ID,
                "feature_group": key[0],
                "feature_id": key[1],
                "feature_label": key[2],
                "feature_stat": key[3],
                "correlation_with_rolling_deposit_beta": corr,
                "abs_correlation": abs(corr),
                "correlation_residualized_on_share_2020_2021": "" if partial_corr is None else partial_corr,
                "abs_residualized_correlation": "" if partial_corr is None else abs(partial_corr),
                "n_windows": len(pairs),
                "mean_feature_value": _sample_mean(feature_values),
                "sd_feature_value": _sample_sd(feature_values),
                "diagnostic_role": example.get("diagnostic_role", ""),
                "claim_boundary": (
                    "Exploratory correlation across overlapping rolling windows. "
                    "This is not an independent-observation test or a causal channel estimate."
                ),
            }
        )
    return sorted(output, key=lambda row: float(row["abs_correlation"]), reverse=True)


def _residualize_on_one(values: list[float], control_values: list[float]) -> list[float]:
    if len(values) != len(control_values) or not values:
        return []
    control_mean = sum(control_values) / len(control_values)
    value_mean = sum(values) / len(values)
    control_ss = sum((value - control_mean) ** 2 for value in control_values)
    slope = 0.0
    if control_ss > 0:
        slope = sum((control - control_mean) * (value - value_mean) for value, control in zip(values, control_values)) / control_ss
    intercept = value_mean - slope * control_mean
    return [value - intercept - slope * control for value, control in zip(values, control_values)]


def _format_number(value: Any, *, digits: int = 2) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.{digits}f}"


def _first_row(rows: list[dict[str, Any]], **match: str) -> dict[str, Any]:
    for row in rows:
        if all(str(row.get(key, "")) == value for key, value in match.items()):
            return row
    return {}


def _write_report(
    *,
    level_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    correlation_rows: list[dict[str, Any]],
    identity_rows: list[dict[str, Any]],
    rolling_correlate_rows: list[dict[str, Any]],
) -> None:
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tier 2 Pass-Through Offset Diagnostics",
        "",
        "These diagnostics are descriptive stability checks for the selected-lag Tier 2 pass-through result. "
        "They should not be read as a causal decomposition or as the canonical pass-through estimate.",
        "",
        "Canonical interpretation remains the beta-per-$1 TDC selected-lag LP/regression pass-through result where available. "
        "Pearson correlations, lead-lag correlations, and episode splits are secondary context on offset patterns and timing.",
        "",
        "## TDC Level by Period",
        "",
        "| Period | n | Mean $B | Median $B | Mean abs $B | Max quarter | Max $B |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in level_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["period"]),
                    str(row["n"]),
                    _format_number(float(row["mean_mil"]) / 1000.0),
                    _format_number(float(row["median_mil"]) / 1000.0),
                    _format_number(float(row["mean_abs_mil"]) / 1000.0),
                    str(row["max_quarter"]),
                    _format_number(float(row["max_mil"]) / 1000.0),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Selected Episode Betas", "", "| Period | Outcome | Status | Effect per +$100B TDC | p | n |", "|---|---|---:|---:|---:|---:|"])
    for period in ("full_available", "pre_2020", "exclude_2020_2021", "exclude_transition_2019_2021"):
        for outcome in ("matched_total_deposits", PRIMARY_RESIDUAL_ID, "tdcpass_strict_loan_consumer_credit_qoq", "bank_treasury_securities_qoq"):
            row = _first_row(episode_rows, period=period, outcome=outcome)
            if not row:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        period,
                        f"`{outcome}`",
                        str(row.get("status", "")),
                        _format_number(row.get("effect_per_100b_tdc")),
                        _format_number(row.get("p_value_normal"), digits=3),
                        str(row.get("n", row.get("n_pairs", ""))),
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Pre-2020 Correlations", "", "| Outcome | Correlation | n |", "|---|---:|---:|"])
    for outcome in ("matched_total_deposits", PRIMARY_RESIDUAL_ID, "tdcpass_strict_loan_core_min_qoq", "tdcpass_strict_loan_mortgages_qoq", "tdcpass_strict_loan_consumer_credit_qoq", "bank_treasury_securities_qoq"):
        row = _first_row(correlation_rows, period="pre_2020", outcome=outcome)
        if row:
            lines.append(f"| `{outcome}` | {_format_number(row.get('correlation'), digits=3)} | {row.get('n', '')} |")
    latest_identity = identity_rows[-1] if identity_rows else {}
    lines.extend(
        [
            "",
            "## Identity Check",
            "",
            "For the same matched-deposit identity, the rolling deposit beta should equal one plus the same-treatment residual beta. "
            "The `deposit_minus_residual_minus_one` column is the numerical check on that accounting relationship.",
        ]
    )
    if latest_identity:
        lines.append(
            f"- Latest rolling-window identity error: {_format_number(latest_identity.get('deposit_minus_residual_minus_one'), digits=6)} "
            f"for {latest_identity.get('window_start_quarter')} to {latest_identity.get('window_end_quarter')}."
        )
    lines.extend(
        [
            "",
            "## Rolling Beta Correlates",
            "",
            "The table ranks exploratory correlations between the rolling deposit beta and rolling-window feature values. "
            "The windows overlap, so these rows are a screening device rather than an independent-observation test.",
            "",
            "| Feature | Statistic | Correlation with rolling beta | n windows |",
            "|---|---:|---:|---:|",
        ]
    )
    reported = 0
    for row in rolling_correlate_rows:
        if row.get("feature_id") == "window_end_time_index":
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{row.get('feature_group')}: `{row.get('feature_id')}`",
                    str(row.get("feature_stat", "")),
                    _format_number(row.get("correlation_with_rolling_deposit_beta"), digits=3),
                    str(row.get("n_windows", "")),
                ]
            )
            + " |"
        )
        reported += 1
        if reported >= 12:
            break
    residualized_rows = [
        row
        for row in rolling_correlate_rows
        if row.get("correlation_residualized_on_share_2020_2021") not in ("", None)
        and row.get("feature_id") != "share_2020_2021"
    ]
    residualized_rows.sort(key=lambda row: float(row["abs_residualized_correlation"]), reverse=True)
    lines.extend(
        [
            "",
            "### After Residualizing on 2020-2021 Window Share",
            "",
            "| Feature | Statistic | Residualized correlation | Zero-order correlation | n windows |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in residualized_rows[:12]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{row.get('feature_group')}: `{row.get('feature_id')}`",
                    str(row.get("feature_stat", "")),
                    _format_number(row.get("correlation_residualized_on_share_2020_2021"), digits=3),
                    _format_number(row.get("correlation_with_rolling_deposit_beta"), digits=3),
                    str(row.get("n_windows", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Bounded Interpretation",
            "",
            "- A low deposit beta means the companion non-TDC deposit component is negative in the same regression/accounting frame; it does not name a causal offset channel.",
            "- Pre-2020 offset evidence should be described as residual, credit-cycle, asset-side, and plumbing co-movement until a cleaner source of identifying timing is available.",
            "- The 2020-2021 jump is partly a data-regime fact because TDC quarterly flows are much larger in that period; the diagnostic does not by itself separate structural pass-through from policy/fiscal/liquidity regime changes.",
            "",
        ]
    )
    REPORT_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def build_outputs() -> dict[str, list[dict[str, Any]]]:
    _paths, rows, control_ids, _factor_count, _screened_count, selected_credit_lags, selected_rate_lags = _build_inputs()
    controls = _canonical_controls(
        rows,
        _scenario_controls(control_ids, selected_credit_lags, selected_rate_lags)["selected_credit_rate_risk_lags"],
    )
    rolling_rows = _read_csv(ROLLING_REGRESSION_OUTPUT)
    rolling_beta_rows = _rolling_deposit_beta_rows(rolling_rows)
    rolling_features = _rolling_window_features(rows, rolling_beta_rows)
    return {
        "levels": _level_summary(rows),
        "episode_betas": _episode_rows(rows, controls),
        "correlations": _correlation_rows(rows),
        "identity_windows": _identity_rows_from_rolling(rolling_rows),
        "lead_lag_correlations": _lead_lag_rows(rows),
        "jackknife": _jackknife_rows(rows, controls),
        "rolling_beta_features": rolling_features,
        "rolling_beta_correlates": _rolling_beta_correlates(rolling_features),
    }


def _write_manifest(outputs: dict[str, list[dict[str, Any]]]) -> None:
    MANIFEST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": JOB_ID,
        "outputs": {
            "episode_betas": str(EPISODE_OUTPUT.relative_to(ROOT)),
            "level_summary": str(LEVEL_OUTPUT.relative_to(ROOT)),
            "correlations": str(CORRELATION_OUTPUT.relative_to(ROOT)),
            "identity_windows": str(IDENTITY_OUTPUT.relative_to(ROOT)),
            "lead_lag_correlations": str(LEAD_LAG_OUTPUT.relative_to(ROOT)),
            "jackknife": str(JACKKNIFE_OUTPUT.relative_to(ROOT)),
            "rolling_beta_features": str(ROLLING_FEATURE_OUTPUT.relative_to(ROOT)),
            "rolling_beta_correlates": str(ROLLING_CORRELATE_OUTPUT.relative_to(ROOT)),
            "report": str(REPORT_OUTPUT.relative_to(ROOT)),
        },
        "row_counts": {key: len(value) for key, value in outputs.items()},
        "claim_boundary": (
            "Offset diagnostics are descriptive stability and timing checks only; canonical pass-through remains "
            "the selected-lag beta-per-dollar LP/regression estimate."
        ),
    }
    MANIFEST_OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    outputs = build_outputs()
    _write_csv(LEVEL_OUTPUT, outputs["levels"])
    _write_csv(EPISODE_OUTPUT, outputs["episode_betas"])
    _write_csv(CORRELATION_OUTPUT, outputs["correlations"])
    _write_csv(IDENTITY_OUTPUT, outputs["identity_windows"])
    _write_csv(LEAD_LAG_OUTPUT, outputs["lead_lag_correlations"])
    _write_csv(JACKKNIFE_OUTPUT, outputs["jackknife"])
    _write_csv(ROLLING_FEATURE_OUTPUT, outputs["rolling_beta_features"])
    _write_csv(ROLLING_CORRELATE_OUTPUT, outputs["rolling_beta_correlates"])
    _write_report(
        level_rows=outputs["levels"],
        episode_rows=outputs["episode_betas"],
        correlation_rows=outputs["correlations"],
        identity_rows=outputs["identity_windows"],
        rolling_correlate_rows=outputs["rolling_beta_correlates"],
    )
    _write_manifest(outputs)
    print(f"wrote {len(outputs['episode_betas'])} episode rows to {EPISODE_OUTPUT}")
    print(f"wrote {len(outputs['correlations'])} correlation rows to {CORRELATION_OUTPUT}")
    print(f"wrote {len(outputs['rolling_beta_correlates'])} rolling-beta correlate rows to {ROLLING_CORRELATE_OUTPUT}")
    print(f"wrote report to {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
