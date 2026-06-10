"""RateWall-facing regime-validation package for TDC deposit pass-through.

This package is intentionally conservative. It promotes source-bound pass-
through values for RateWall Assumption Mode scenarios where the evidence is
traceable, and it blocks runtime selector use unless validation is strong enough.
Current evidence is not promotion-grade for runtime selection.
"""
from __future__ import annotations

import csv
import hashlib
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

from ea_tdc.estimation import _build_quarterly_target, _coerce_float  # noqa: E402
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
from run_tier2_pass_through_offset_diagnostics import (  # noqa: E402
    _between,
    _canonical_controls,
    _pearson,
    _quarter_key,
    _safe_float,
    _subset_rows,
)


JOB_ID = "ea_tdc_pass_through_regime_validation"
TABLE_DIR = ROOT / "outputs/tables"
REPORT_DIR = ROOT / "outputs/reports"
MANIFEST_DIR = ROOT / "outputs/manifests"

CLASSIFIER_OUTPUT = TABLE_DIR / "ea_tdc_pass_through_regime_classifier_candidates.csv"
ESTIMATES_OUTPUT = TABLE_DIR / "ea_tdc_pass_through_regime_estimates.csv"
VALIDATION_OUTPUT = TABLE_DIR / "ea_tdc_pass_through_regime_validation.csv"
CONTRACT_OUTPUT = TABLE_DIR / "ea_tdc_pass_through_ratewall_import_contract.csv"
MEMO_OUTPUT = REPORT_DIR / "ea_tdc_pass_through_regime_validation_memo.md"
MANIFEST_OUTPUT = MANIFEST_DIR / "ea_tdc_pass_through_regime_validation_summary.json"

DESIGN_BUNDLE = ROOT / "data/bundles/designs/tdc_tier2_mmf_rrp_canonical_full_panel__quarterly_bundle.csv"
ROLLING_ESTIMATES = ROOT / "output/models/tier2_rolling_selected_credit_rate_pass_through_estimates.csv"
ROLLING_MINUS = ROOT / "output/reports/tier2_pass_through_rolling_minus_pandemic_betas.csv"
INFLUENCE = ROOT / "output/reports/tier2_pass_through_influence_quarters.csv"
OFFSET_EPISODES = ROOT / "output/reports/tier2_pass_through_offset_episode_betas.csv"
SUBMISSION_HAC = ROOT / "output/reports/submission_hac_bandwidth_sensitivity.csv"

MIN_SCENARIO_N = 24
MIN_RUNTIME_VALIDATION_N = 40
TOTRESNS_MATERIALITY_THRESHOLD = 0.15
HORIZONS = [0, 1]


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hash(path: Path) -> str:
    return _sha256_file(path) if path.exists() else ""


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[int(index)]
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def _finite_values(rows: list[dict[str, str]], column: str) -> list[float]:
    return [value for row in rows if (value := _safe_float(row.get(column, ""))) is not None]


def _sample_window(rows: list[dict[str, str]]) -> tuple[str, str]:
    quarters = sorted([str(row.get("quarter", "")) for row in rows if row.get("quarter")], key=_quarter_key)
    return (quarters[0], quarters[-1]) if quarters else ("", "")


def _lp_complete_case_quarters(
    rows: list[dict[str, str]],
    *,
    treatment_id: str,
    outcome_id: str,
    horizon: int,
    control_ids: list[str],
) -> list[str]:
    quarters: list[str] = []
    for idx, row in enumerate(rows):
        if _coerce_float(row.get(treatment_id, "")) is None:
            continue
        controls_ok = all(_coerce_float(row.get(control_id, "")) is not None for control_id in control_ids)
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
        quarter = str(row.get("quarter", ""))
        if quarter:
            quarters.append(quarter)
    return quarters


def _candidate_specs(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    source_hash = _artifact_hash(DESIGN_BUNDLE)

    def q(column: str, p: float) -> float:
        return _quantile(_finite_values(rows, column), p)

    def q_abs(column: str, p: float) -> float:
        return _quantile([abs(value) for value in _finite_values(rows, column)], p)

    specs = [
        {
            "candidate_trigger_id": "normal_forward_nonpandemic_baseline",
            "regime_id": "normal_forward",
            "predictor": "quarter",
            "predictor_source": _relative(DESIGN_BUNDLE),
            "transform": "exclude_2020Q1_2021Q4",
            "lag": 0,
            "threshold": "not_in_2020Q1_2021Q4",
            "sign": "baseline",
            "rationale": "Forward scenario baseline excludes pandemic fiscal-transfer block.",
        },
        {
            "candidate_trigger_id": "latest_rolling_persistence_window",
            "regime_id": "latest_rolling_persistence",
            "predictor": "rolling_window_end",
            "predictor_source": _relative(ROLLING_ESTIMATES),
            "transform": "latest_48q_selected_lag_window",
            "lag": 0,
            "threshold": "latest_available_window",
            "sign": "diagnostic_persistence",
            "rationale": "Latest rolling beta remains elevated but is partly pandemic-window composition.",
        },
        {
            "candidate_trigger_id": "pandemic_fiscal_transfer_block",
            "regime_id": "pandemic_fiscal_transfer_block",
            "predictor": "quarter",
            "predictor_source": _relative(DESIGN_BUNDLE),
            "transform": "calendar_block",
            "lag": 0,
            "threshold": "2020Q1_to_2021Q4",
            "sign": "positive_pass_through_expected",
            "rationale": "Pandemic fiscal-transfer/TGA/reserve block has high-leverage observations.",
        },
        {
            "candidate_trigger_id": "post_pandemic_nonblock",
            "regime_id": "post_pandemic_nonblock",
            "predictor": "quarter",
            "predictor_source": _relative(DESIGN_BUNDLE),
            "transform": "calendar_block",
            "lag": 0,
            "threshold": "quarter>=2022Q1",
            "sign": "review_persistence",
            "rationale": "Tests whether elevated beta persists after the pandemic block.",
        },
        {
            "candidate_trigger_id": "high_tdc_abs_q75",
            "regime_id": "high_tdc_scale",
            "predictor": PRIMARY_TREATMENT_ID,
            "predictor_source": _relative(DESIGN_BUNDLE),
            "transform": "abs_value_q75",
            "lag": 0,
            "threshold": q_abs(PRIMARY_TREATMENT_ID, 0.75),
            "sign": "positive_pass_through_expected",
            "rationale": "Rolling beta correlates strongly with mean absolute TDC scale.",
        },
        {
            "candidate_trigger_id": "tga_drawdown_q25",
            "regime_id": "tga_drawdown_liquidity_event",
            "predictor": "tga_balance_qoq",
            "predictor_source": _relative(DESIGN_BUNDLE),
            "transform": "low_tail_q25",
            "lag": 0,
            "threshold": q("tga_balance_qoq", 0.25),
            "sign": "positive_pass_through_expected_for_drawdown",
            "rationale": "TGA drawdowns can move Treasury cash into private deposits/reserves.",
        },
        {
            "candidate_trigger_id": "tga_rebuild_q75",
            "regime_id": "tga_rebuild_liquidity_event",
            "predictor": "tga_balance_qoq",
            "predictor_source": _relative(DESIGN_BUNDLE),
            "transform": "high_tail_q75",
            "lag": 0,
            "threshold": q("tga_balance_qoq", 0.75),
            "sign": "negative_or_ambiguous_pass_through_expected",
            "rationale": "TGA rebuilds may drain private cash and lower pass-through.",
        },
        {
            "candidate_trigger_id": "reserve_change_abs_q75",
            "regime_id": "high_liquidity_event",
            "predictor": "reserve_balances_qoq",
            "predictor_source": _relative(DESIGN_BUNDLE),
            "transform": "abs_value_q75",
            "lag": 0,
            "threshold": q_abs("reserve_balances_qoq", 0.75),
            "sign": "positive_when_fiscal_tga_aligned",
            "rationale": "Reserve movement is a liquidity-state diagnostic, not sufficient alone.",
        },
        {
            "candidate_trigger_id": "on_rrp_flow_abs_q75",
            "regime_id": "on_rrp_mmf_absorption_state",
            "predictor": "on_rrp_balance_qoq",
            "predictor_source": _relative(DESIGN_BUNDLE),
            "transform": "abs_value_q75",
            "lag": 0,
            "threshold": q_abs("on_rrp_balance_qoq", 0.75),
            "sign": "ambiguous_buffer_or_leakage",
            "rationale": "ON RRP can buffer liquidity shocks or divert cash away from deposits.",
        },
        {
            "candidate_trigger_id": "low_short_rate_q25",
            "regime_id": "low_rate_liquidity_state",
            "predictor": "FEDFUNDS",
            "predictor_source": _relative(DESIGN_BUNDLE),
            "transform": "low_tail_q25",
            "lag": 0,
            "threshold": q("FEDFUNDS", 0.25),
            "sign": "positive_conditional_on_fiscal_liquidity",
            "rationale": "Post-2008 placebo implies low rates alone are not enough.",
        },
        {
            "candidate_trigger_id": "reserve_scarcity_low_reserve_q25",
            "regime_id": "reserve_scarcity_or_low_liquidity",
            "predictor": "reserve_balances",
            "predictor_source": _relative(DESIGN_BUNDLE),
            "transform": "low_tail_q25",
            "lag": 0,
            "threshold": q("reserve_balances", 0.25),
            "sign": "negative_or_blocked_if_thin",
            "rationale": "Low reserve state is included only if sample support is adequate.",
        },
    ]
    start, end = _sample_window(rows)
    for row in specs:
        source_path = ROOT / str(row["predictor_source"])
        row.update(
            {
                "data_coverage_start": start,
                "data_coverage_end": end,
                "sample_window": f"{start}_to_{end}",
                "source_artifact_sha256": _artifact_hash(source_path) or source_hash,
                "provenance_note": "thresholds computed from current EA-TDC quarterly design bundle; estimation rows report complete-case samples separately",
            }
        )
    return specs


def _predicate_for_trigger(spec: dict[str, Any]) -> Callable[[dict[str, str]], bool]:
    trigger = spec["candidate_trigger_id"]
    threshold = _safe_float(spec.get("threshold"))
    if trigger == "normal_forward_nonpandemic_baseline":
        return lambda row: not _between(str(row.get("quarter", "")), "2020Q1", "2021Q4")
    if trigger == "latest_rolling_persistence_window":
        return lambda row: _between(str(row.get("quarter", "")), "2014Q3", "2026Q2")
    if trigger == "pandemic_fiscal_transfer_block":
        return lambda row: _between(str(row.get("quarter", "")), "2020Q1", "2021Q4")
    if trigger == "post_pandemic_nonblock":
        return lambda row: _quarter_key(str(row.get("quarter", "0000Q1"))) >= _quarter_key("2022Q1")
    if trigger == "high_tdc_abs_q75":
        return lambda row: (value := _safe_float(row.get(PRIMARY_TREATMENT_ID, ""))) is not None and threshold is not None and abs(value) >= threshold
    if trigger == "tga_drawdown_q25":
        return lambda row: (value := _safe_float(row.get("tga_balance_qoq", ""))) is not None and threshold is not None and value <= threshold
    if trigger == "tga_rebuild_q75":
        return lambda row: (value := _safe_float(row.get("tga_balance_qoq", ""))) is not None and threshold is not None and value >= threshold
    if trigger == "reserve_change_abs_q75":
        return lambda row: (value := _safe_float(row.get("reserve_balances_qoq", ""))) is not None and threshold is not None and abs(value) >= threshold
    if trigger == "on_rrp_flow_abs_q75":
        return lambda row: (value := _safe_float(row.get("on_rrp_balance_qoq", ""))) is not None and threshold is not None and abs(value) >= threshold
    if trigger == "low_short_rate_q25":
        return lambda row: (value := _safe_float(row.get("FEDFUNDS", ""))) is not None and threshold is not None and value <= threshold
    if trigger == "reserve_scarcity_low_reserve_q25":
        return lambda row: (value := _safe_float(row.get("reserve_balances", ""))) is not None and threshold is not None and value <= threshold
    return lambda _row: False


def _estimate_subset(
    rows: list[dict[str, str]],
    *,
    controls: list[str],
    regime_id: str,
    estimator_id: str,
    horizon: int,
    excluded_blocks: str,
    source_artifact_path: Path,
) -> dict[str, Any]:
    data_start, data_end = _sample_window(rows)
    base = {
        "job_id": JOB_ID,
        "regime_id": regime_id,
        "estimator_id": estimator_id,
        "horizon": horizon,
        "dependent_variable": "matched_total_deposits",
        "tdc_regressor_definition": PRIMARY_TREATMENT_ID,
        "data_coverage_start": data_start,
        "data_coverage_end": data_end,
        "estimation_sample_start": "",
        "estimation_sample_end": "",
        "n_complete_cases": "",
        "sample_start": "",
        "sample_end": "",
        "sample_window": "",
        "excluded_blocks": excluded_blocks,
        "controls": ",".join(controls),
        "source_artifact_path": _relative(source_artifact_path),
        "source_artifact_sha256": _artifact_hash(source_artifact_path),
    }
    if len(rows) < MIN_SCENARIO_N:
        base.update({"status": "insufficient_observations", "n": len(rows)})
        return base
    try:
        fit, n, used, rejected = _fit_lp(
            rows,
            treatment_id=PRIMARY_TREATMENT_ID,
            outcome_id="matched_total_deposits",
            horizon=horizon,
            control_ids=controls,
            covariance_lags=1,
        )
    except ValueError as exc:
        base.update({"status": "not_estimable", "n": len(rows), "exact_blocker": str(exc)})
        return base
    complete_case_quarters = _lp_complete_case_quarters(
        rows,
        treatment_id=PRIMARY_TREATMENT_ID,
        outcome_id="matched_total_deposits",
        horizon=horizon,
        control_ids=used,
    )
    estimation_start = complete_case_quarters[0] if complete_case_quarters else ""
    estimation_end = complete_case_quarters[-1] if complete_case_quarters else ""
    beta = fit.beta[1]
    se = fit.ses[1]
    z_score = beta / se if se > 0 else None
    _, multiplier = _normalization("matched_total_deposits")
    point = beta * multiplier
    se_norm = se * multiplier
    base.update(
        {
            "status": "estimated",
            "point_estimate": point,
            "standard_error": se_norm,
            "lower95": point - 1.96 * se_norm,
            "upper95": point + 1.96 * se_norm,
            "p_value_normal": "" if z_score is None else _normal_p_two_sided(z_score),
            "n": n,
            "n_complete_cases": len(complete_case_quarters),
            "estimation_sample_start": estimation_start,
            "estimation_sample_end": estimation_end,
            "sample_start": estimation_start,
            "sample_end": estimation_end,
            "sample_window": f"{estimation_start}_to_{estimation_end}" if estimation_start and estimation_end else "",
            "controls_used": ",".join(used),
            "controls_rejected": ",".join(rejected),
            "covariance_estimator": "newey_west",
            "covariance_lags": 1,
            "rsquared": fit.rsquared,
            "normalized_unit": "dollars_per_dollar_tdc",
            "source_row_key": f"{regime_id}::{estimator_id}::h{horizon}::{estimation_start}_to_{estimation_end}",
        }
    )
    return base


def _build_estimates(rows: list[dict[str, str]], controls: list[str], candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    estimates: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        estimates.append(
            _estimate_subset(
                rows,
                controls=controls,
                regime_id="pooled_full_sample",
                estimator_id="selected_credit_rate_lags_pooled",
                horizon=horizon,
                excluded_blocks="none",
                source_artifact_path=DESIGN_BUNDLE,
            )
        )
        nonpandemic = _subset_rows(rows, lambda quarter: not _between(quarter, "2020Q1", "2021Q4"))
        estimates.append(
            _estimate_subset(
                nonpandemic,
                controls=controls,
                regime_id="normal_forward",
                estimator_id="selected_credit_rate_lags_exclude_2020_2021",
                horizon=horizon,
                excluded_blocks="2020Q1-2021Q4",
                source_artifact_path=DESIGN_BUNDLE,
            )
        )
    for candidate in candidate_rows:
        predicate = _predicate_for_trigger(candidate)
        subset = [row for row in rows if predicate(row)]
        estimates.append(
            _estimate_subset(
                subset,
                controls=controls,
                regime_id=str(candidate["regime_id"]),
                estimator_id=f"sample_split_{candidate['candidate_trigger_id']}",
                horizon=0,
                excluded_blocks="trigger_complement",
                source_artifact_path=DESIGN_BUNDLE,
            )
        )
    estimates.extend(_estimates_from_existing_artifacts())
    return estimates


def _totresns_robustness_rows(rows: list[dict[str, str]], controls: list[str]) -> list[dict[str, Any]]:
    rows_by_regime = {
        "pooled_full_sample": (rows, "none"),
        "normal_forward": (_subset_rows(rows, lambda quarter: not _between(quarter, "2020Q1", "2021Q4")), "2020Q1-2021Q4"),
    }
    variants = [
        ("with_contemporaneous_totresns", controls),
        ("no_contemporaneous_totresns", [control for control in controls if control != "TOTRESNS"]),
    ]
    output: list[dict[str, Any]] = []
    for regime_id, (regime_rows, excluded_blocks) in rows_by_regime.items():
        for variant_id, variant_controls in variants:
            estimate = _estimate_subset(
                regime_rows,
                controls=variant_controls,
                regime_id=regime_id,
                estimator_id=f"totresns_robustness_{variant_id}",
                horizon=0,
                excluded_blocks=excluded_blocks,
                source_artifact_path=DESIGN_BUNDLE,
            )
            estimate.update(
                {
                    "robustness_check": "no_contemporaneous_totresns",
                    "controls_variant": variant_id,
                    "decision_rule_materiality_threshold": TOTRESNS_MATERIALITY_THRESHOLD,
                }
            )
            output.append(estimate)
    return output


def _totresns_decision(estimates: list[dict[str, Any]]) -> dict[str, Any]:
    rows = {
        (str(row.get("regime_id", "")), str(row.get("controls_variant", ""))): row
        for row in estimates
        if row.get("robustness_check") == "no_contemporaneous_totresns"
        and str(row.get("horizon", "")) in {"0", "0.0"}
    }
    normal_with = _safe_float(rows.get(("normal_forward", "with_contemporaneous_totresns"), {}).get("point_estimate"))
    normal_without = _safe_float(rows.get(("normal_forward", "no_contemporaneous_totresns"), {}).get("point_estimate"))
    pooled_without = _safe_float(rows.get(("pooled_full_sample", "no_contemporaneous_totresns"), {}).get("point_estimate"))
    if normal_with is None or normal_without is None:
        return {
            "status": "missing",
            "delta": "",
            "message": "No-TOTRESNS robustness decision unavailable because the normal-forward comparison is missing.",
        }
    delta = normal_without - normal_with
    ordering_intact = pooled_without is None or pooled_without >= normal_without
    if abs(delta) <= TOTRESNS_MATERIALITY_THRESHOLD and ordering_intact:
        status = "freeze_ok"
        message = (
            f"No-TOTRESNS robustness leaves the normal-forward coefficient at {_fmt(normal_without)} versus "
            f"{_fmt(normal_with)} with contemporaneous TOTRESNS (delta {_fmt(delta)}), within the "
            f"{TOTRESNS_MATERIALITY_THRESHOLD:.2f} materiality rule; regime ordering remains intact, so publish-and-freeze is supported."
        )
    elif delta > TOTRESNS_MATERIALITY_THRESHOLD:
        status = "materially_higher"
        message = (
            f"No-TOTRESNS robustness raises the normal-forward coefficient to {_fmt(normal_without)} versus "
            f"{_fmt(normal_with)} with contemporaneous TOTRESNS (delta {_fmt(delta)}), exceeding the "
            f"{TOTRESNS_MATERIALITY_THRESHOLD:.2f} rule; with-reserves rows should be labeled lower/direct-effect estimates and the RateWall envelope widened."
        )
    else:
        status = "collapse_or_ordering_break"
        message = (
            f"No-TOTRESNS robustness changes the normal-forward coefficient to {_fmt(normal_without)} versus "
            f"{_fmt(normal_with)} with contemporaneous TOTRESNS (delta {_fmt(delta)}), failing the materiality/order rule; run H.4.1 reserve-accounting decomposition before freeze."
        )
    return {"status": status, "delta": delta, "message": message}


def _estimates_from_existing_artifacts() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rolling = _read_csv(ROLLING_ESTIMATES)
    latest = None
    for row in rolling:
        if row.get("outcome") == "matched_total_deposits" and (
            latest is None or _quarter_key(str(row.get("window_end_quarter", "0000Q1"))) > _quarter_key(str(latest.get("window_end_quarter", "0000Q1")))
        ):
            latest = row
    if latest:
        rows.append(
            {
                "job_id": JOB_ID,
                "regime_id": "latest_rolling_persistence",
                "estimator_id": "rolling_48q_selected_credit_rate_lags",
                "horizon": 0,
                "dependent_variable": "matched_total_deposits",
                "tdc_regressor_definition": PRIMARY_TREATMENT_ID,
                "point_estimate": latest.get("normalized_beta", ""),
                "standard_error": latest.get("normalized_se", ""),
                "lower95": latest.get("normalized_lower95", ""),
                "upper95": latest.get("normalized_upper95", ""),
                "n": latest.get("n", ""),
                "data_coverage_start": latest.get("window_start_quarter", ""),
                "data_coverage_end": latest.get("window_end_quarter", ""),
                "estimation_sample_start": latest.get("window_start_quarter", ""),
                "estimation_sample_end": latest.get("window_end_quarter", ""),
                "n_complete_cases": latest.get("n", ""),
                "sample_start": latest.get("window_start_quarter", ""),
                "sample_end": latest.get("window_end_quarter", ""),
                "sample_window": latest.get("sample_label", ""),
                "excluded_blocks": "none",
                "controls": latest.get("control_ids_used", ""),
                "controls_used": latest.get("control_ids_used", ""),
                "controls_rejected": latest.get("dropped_control_ids", ""),
                "covariance_estimator": latest.get("covariance_estimator", ""),
                "covariance_lags": latest.get("covariance_lags", ""),
                "rsquared": latest.get("rsquared", ""),
                "normalized_unit": latest.get("normalized_unit", ""),
                "source_artifact_path": _relative(ROLLING_ESTIMATES),
                "source_artifact_sha256": _artifact_hash(ROLLING_ESTIMATES),
                "source_row_key": f"rolling::{latest.get('outcome')}::{latest.get('window_start_quarter')}_{latest.get('window_end_quarter')}",
                "status": "estimated",
            }
        )
    persistence = _read_csv(ROLLING_MINUS)
    for drop_rule, regime_id in [
        ("drop_2020", "pandemic_exclusion_drop_2020"),
        ("drop_2020_2021", "pandemic_exclusion_drop_2020q1_2021q4"),
        ("drop_2021", "pandemic_exclusion_drop_2021"),
    ]:
        candidates = [row for row in persistence if row.get("drop_rule") == drop_rule and row.get("status") == "estimated"]
        if not candidates:
            continue
        row = max(candidates, key=lambda item: _quarter_key(str(item.get("window_end_quarter", "0000Q1"))))
        se = _safe_float(row.get("deposit_se"))
        point = _safe_float(row.get("deposit_beta"))
        rows.append(
            {
                "job_id": JOB_ID,
                "regime_id": regime_id,
                "estimator_id": f"rolling_48q_{drop_rule}",
                "horizon": 0,
                "dependent_variable": "matched_total_deposits",
                "tdc_regressor_definition": PRIMARY_TREATMENT_ID,
                "point_estimate": "" if point is None else point,
                "standard_error": "" if se is None else se,
                "lower95": "" if point is None or se is None else point - 1.96 * se,
                "upper95": "" if point is None or se is None else point + 1.96 * se,
                "n": row.get("n", ""),
                "data_coverage_start": row.get("window_start_quarter", ""),
                "data_coverage_end": row.get("window_end_quarter", ""),
                "estimation_sample_start": row.get("window_start_quarter", ""),
                "estimation_sample_end": row.get("window_end_quarter", ""),
                "n_complete_cases": row.get("n", ""),
                "sample_start": row.get("window_start_quarter", ""),
                "sample_end": row.get("window_end_quarter", ""),
                "sample_window": f"{row.get('window_start_quarter', '')}_to_{row.get('window_end_quarter', '')}",
                "excluded_blocks": row.get("dropped_quarters", ""),
                "controls": row.get("controls_used", ""),
                "controls_used": row.get("controls_used", ""),
                "controls_rejected": row.get("controls_rejected", ""),
                "covariance_estimator": "newey_west",
                "covariance_lags": 1,
                "rsquared": row.get("deposit_rsquared", ""),
                "normalized_unit": "dollars_per_dollar_tdc",
                "source_artifact_path": _relative(ROLLING_MINUS),
                "source_artifact_sha256": _artifact_hash(ROLLING_MINUS),
                "source_row_key": f"{regime_id}::{row.get('window_start_quarter')}_{row.get('window_end_quarter')}::{drop_rule}",
                "status": "estimated",
            }
        )
    return rows


def _rmse(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else float("nan")


def _simple_oos_validation(rows: list[dict[str, str]], candidate_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    sorted_rows = sorted([row for row in rows if row.get("quarter")], key=lambda row: _quarter_key(str(row["quarter"])))
    validations: dict[str, dict[str, Any]] = {}
    for candidate in candidate_rows:
        regime_id = str(candidate["regime_id"])
        predicate = _predicate_for_trigger(candidate)
        pooled_errors: list[float] = []
        regime_errors: list[float] = []
        false_positive = 0
        false_negative = 0
        evaluated = 0
        for index, row in enumerate(sorted_rows):
            if index < 40:
                continue
            y = _safe_float(row.get("matched_total_deposits", ""))
            x = _safe_float(row.get(PRIMARY_TREATMENT_ID, ""))
            if y is None or x is None:
                continue
            train = sorted_rows[:index]
            pooled_x, pooled_y = _paired(train, PRIMARY_TREATMENT_ID, "matched_total_deposits")
            pooled_beta = _slope_no_intercept(pooled_x, pooled_y)
            triggered = predicate(row)
            state_train = [train_row for train_row in train if predicate(train_row) == triggered]
            state_x, state_y = _paired(state_train, PRIMARY_TREATMENT_ID, "matched_total_deposits")
            state_beta = _slope_no_intercept(state_x, state_y) if len(state_x) >= 12 else pooled_beta
            if pooled_beta is None or state_beta is None:
                continue
            pooled_errors.append(y - pooled_beta * x)
            regime_errors.append(y - state_beta * x)
            evaluated += 1
            realized_high = abs(y) >= _quantile([abs(value) for value in pooled_y], 0.75) if len(pooled_y) >= 20 else False
            if triggered and not realized_high:
                false_positive += 1
            if (not triggered) and realized_high:
                false_negative += 1
        pooled_rmse = _rmse(pooled_errors)
        regime_rmse = _rmse(regime_errors)
        validations[regime_id] = {
            "oos_n": evaluated,
            "pooled_baseline_rmse": pooled_rmse,
            "regime_classifier_rmse": regime_rmse,
            "rmse_improvement": "" if not math.isfinite(pooled_rmse) or pooled_rmse == 0 else (pooled_rmse - regime_rmse) / pooled_rmse,
            "false_positive_count": false_positive,
            "false_negative_count": false_negative,
        }
    return validations


def _paired(rows: list[dict[str, str]], x_col: str, y_col: str) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []
    for row in rows:
        x = _safe_float(row.get(x_col, ""))
        y = _safe_float(row.get(y_col, ""))
        if x is None or y is None:
            continue
        x_values.append(x)
        y_values.append(y)
    return x_values, y_values


def _slope_no_intercept(x_values: list[float], y_values: list[float]) -> float | None:
    denominator = sum(value * value for value in x_values)
    if denominator <= 0 or len(x_values) < 3:
        return None
    return sum(x * y for x, y in zip(x_values, y_values)) / denominator


def _validation_rows(
    estimates: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    oos = _simple_oos_validation(rows, candidate_rows)
    influence = _read_csv(INFLUENCE)
    rolling_minus = _read_csv(ROLLING_MINUS)
    validation: list[dict[str, Any]] = []
    by_regime = _best_h0_estimates_by_regime(estimates)
    for regime_id, estimate in sorted(by_regime.items()):
        n = int(float(estimate.get("n", 0) or 0))
        oos_row = oos.get(regime_id, {})
        validation.append(
            {
                "job_id": JOB_ID,
                "regime_id": regime_id,
                "validation_case": "in_sample_fit",
                "validation_status": "pass_minimum_n" if n >= MIN_SCENARIO_N else "blocked_insufficient_n",
                "in_sample_fit": estimate.get("rsquared", ""),
                "out_of_sample_or_rolling_validation": "",
                "leave_one_quarter_out": "",
                "leave_2020_out": _latest_drop_beta(rolling_minus, "drop_2020"),
                "leave_2021_out": _latest_drop_beta(rolling_minus, "drop_2021"),
                "leave_2020q1_2021q4_out": _latest_drop_beta(rolling_minus, "drop_2020_2021"),
                "influence_diagnostics": _top_influence_summary(influence),
                "false_positive_check": "",
                "false_negative_check": "",
                "pooled_baseline_comparison": by_regime.get("pooled_full_sample", {}).get("point_estimate", ""),
                "rolling_baseline_comparison": by_regime.get("latest_rolling_persistence", {}).get("point_estimate", ""),
                "exact_blocker": "" if n >= MIN_SCENARIO_N else "blocked_insufficient_observations",
            }
        )
        validation.append(
            {
                "job_id": JOB_ID,
                "regime_id": regime_id,
                "validation_case": "oos_classifier_vs_pooled_baseline",
                "validation_status": _oos_status(oos_row),
                "in_sample_fit": "",
                "out_of_sample_or_rolling_validation": oos_row.get("regime_classifier_rmse", ""),
                "pooled_baseline_rmse": oos_row.get("pooled_baseline_rmse", ""),
                "regime_classifier_rmse": oos_row.get("regime_classifier_rmse", ""),
                "rmse_improvement": oos_row.get("rmse_improvement", ""),
                "leave_one_quarter_out": "available_in_influence_table",
                "leave_2020_out": _latest_drop_beta(rolling_minus, "drop_2020"),
                "leave_2021_out": _latest_drop_beta(rolling_minus, "drop_2021"),
                "leave_2020q1_2021q4_out": _latest_drop_beta(rolling_minus, "drop_2020_2021"),
                "influence_diagnostics": _top_influence_summary(influence),
                "false_positive_check": oos_row.get("false_positive_count", ""),
                "false_negative_check": oos_row.get("false_negative_count", ""),
                "pooled_baseline_comparison": by_regime.get("pooled_full_sample", {}).get("point_estimate", ""),
                "rolling_baseline_comparison": by_regime.get("latest_rolling_persistence", {}).get("point_estimate", ""),
                "exact_blocker": "blocked_not_promotion_grade_oos_false_positive_validation",
            }
        )
    return validation


def _oos_status(row: dict[str, Any]) -> str:
    improvement = row.get("rmse_improvement")
    n = int(row.get("oos_n", 0) or 0)
    if n < MIN_RUNTIME_VALIDATION_N:
        return "blocked_insufficient_oos_observations"
    try:
        numeric = float(improvement)
    except (TypeError, ValueError):
        return "blocked_missing_oos_improvement"
    return "review_only_oos_improves_not_runtime_grade" if numeric > 0 else "blocked_no_oos_improvement"


def _latest_drop_beta(rows: list[dict[str, str]], drop_rule: str) -> str:
    candidates = [row for row in rows if row.get("drop_rule") == drop_rule and row.get("status") == "estimated"]
    if not candidates:
        return ""
    latest = max(candidates, key=lambda row: _quarter_key(str(row.get("window_end_quarter", "0000Q1"))))
    return latest.get("deposit_beta", "")


def _top_influence_summary(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    top = rows[:6]
    return ";".join(f"{row.get('outcome')}:{row.get('quarter')}:dfbeta={row.get('dfbeta')}" for row in top)


def _contract_rows(estimates: list[dict[str, Any]], validations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validation_by_regime: dict[str, list[dict[str, Any]]] = {}
    for row in validations:
        validation_by_regime.setdefault(str(row.get("regime_id")), []).append(row)
    output: list[dict[str, Any]] = []
    best_estimates = _best_h0_estimates_by_regime(estimates)
    for regime_id, estimate in sorted(best_estimates.items()):
        status = str(estimate.get("status", ""))
        n = int(float(estimate.get("n", 0) or 0))
        stability_ok = status == "estimated" and n >= MIN_SCENARIO_N and _ci_width_ok(estimate)
        scenario_allowed = stability_ok and regime_id not in {"pooled_full_sample"}
        recommended = "assumption_mode_scenario_allowed" if scenario_allowed else "review_only"
        trigger_status = "review_only_source_bound_trigger" if status == "estimated" else "blocked_trigger_not_admitted"
        oos_status = _validation_status(validation_by_regime.get(regime_id, []), "oos_classifier_vs_pooled_baseline")
        stability = "scenario_stable_enough" if stability_ok else "review_only_wide_or_missing_ci"
        exact_blocker = "runtime selector blocked: out-of-sample and false-positive controls are not promotion-grade"
        scenario_blocker = _scenario_blocker(estimate, regime_id, stability_ok)
        output.append(
            {
                "regime_id": regime_id,
                "allowed_use": "tdc_deposit_pass_through_assumption_mode",
                "regime_label": _regime_label(regime_id),
                "pass_through_point": estimate.get("point_estimate", ""),
                "pass_through_lower95": estimate.get("lower95", ""),
                "pass_through_upper95": estimate.get("upper95", ""),
                "estimator_id": estimate.get("estimator_id", ""),
                "source_artifact_path": estimate.get("source_artifact_path", ""),
                "source_artifact_sha256": estimate.get("source_artifact_sha256", ""),
                "source_row_key": estimate.get("source_row_key", ""),
                "data_coverage_start": estimate.get("data_coverage_start", ""),
                "data_coverage_end": estimate.get("data_coverage_end", ""),
                "estimation_sample_start": estimate.get("estimation_sample_start", estimate.get("sample_start", "")),
                "estimation_sample_end": estimate.get("estimation_sample_end", estimate.get("sample_end", "")),
                "n_complete_cases": estimate.get("n_complete_cases", estimate.get("n", "")),
                "sample_start": estimate.get("sample_start", ""),
                "sample_end": estimate.get("sample_end", ""),
                "sample_window": estimate.get("sample_window", ""),
                "trigger_rule_id": _trigger_for_regime(regime_id),
                "trigger_rule_text": _trigger_text(regime_id),
                "trigger_validation_status": trigger_status,
                "out_of_sample_validation_status": oos_status or "blocked_missing_oos_validation",
                "false_positive_control_status": "blocked_not_promotion_grade",
                "stability_status": stability,
                "recommended_ratewall_use": recommended,
                "scenario_default_allowed": "true" if regime_id == "normal_forward" and scenario_allowed else "false",
                "runtime_selector_allowed": "false",
                "exact_blocker": "" if scenario_allowed else scenario_blocker,
                "next_ratewall_action": (
                    "import_as_assumption_mode_scenario; keep runtime selector disabled"
                    if scenario_allowed
                    else "review_only; do_not_import_as_runtime_selector"
                ),
                "forbidden_claims": "no_denominator_calibration;no_holder_allocation;no_incidence;no_welfare;no_pricing;no_causal_financialization",
                "claim_boundary": "source-bound pass-through scenario evidence only; not causal deposit creation and not runtime regime selection",
                "runtime_selector_blocker": exact_blocker,
            }
        )
    return output


def _best_h0_estimates_by_regime(estimates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best_estimates: dict[str, dict[str, Any]] = {}
    for estimate in estimates:
        if str(estimate.get("horizon", "")) != "0" and estimate.get("horizon") != 0:
            continue
        regime_id = str(estimate.get("regime_id", ""))
        current = best_estimates.get(regime_id)
        if current is None or _estimate_contract_priority(estimate) < _estimate_contract_priority(current):
            best_estimates[regime_id] = estimate
    return best_estimates


def _estimate_contract_priority(row: dict[str, Any]) -> int:
    estimator = str(row.get("estimator_id", ""))
    if estimator.startswith("rolling_48q"):
        return 0
    if estimator == "selected_credit_rate_lags_exclude_2020_2021":
        return 1
    if estimator == "selected_credit_rate_lags_pooled":
        return 2
    if estimator.startswith("sample_split"):
        return 3
    return 4


def _scenario_blocker(row: dict[str, Any], regime_id: str, stability_ok: bool) -> str:
    status = str(row.get("status", ""))
    n = int(float(row.get("n", 0) or 0))
    if regime_id == "pooled_full_sample":
        return "blocked_pooled_full_sample_is_reference_not_scenario_trigger"
    if status != "estimated":
        return "blocked_not_source_backed_or_not_estimable"
    if n < MIN_SCENARIO_N:
        return "blocked_insufficient_n"
    if not stability_ok:
        return "blocked_wide_or_missing_confidence_interval"
    return "blocked_not_admitted"


def _validation_status(rows: list[dict[str, Any]], validation_case: str) -> str:
    for row in rows:
        if row.get("validation_case") == validation_case:
            return str(row.get("validation_status", ""))
    return ""


def _ci_width_ok(row: dict[str, Any]) -> bool:
    low = _safe_float(row.get("lower95"))
    high = _safe_float(row.get("upper95"))
    return low is not None and high is not None and (high - low) <= 1.5


def _regime_label(regime_id: str) -> str:
    return {
        "normal_forward": "Normal forward / non-pandemic baseline",
        "latest_rolling_persistence": "Latest rolling persistence",
        "pandemic_exclusion_drop_2020": "Latest rolling excluding 2020",
        "pandemic_exclusion_drop_2020q1_2021q4": "Latest rolling excluding 2020Q1-2021Q4",
        "pandemic_exclusion_drop_2021": "Latest rolling excluding 2021",
        "pandemic_fiscal_transfer_block": "Pandemic fiscal-transfer block",
        "post_pandemic_nonblock": "Post-pandemic nonblock",
        "high_tdc_scale": "High TDC scale",
        "high_liquidity_event": "High liquidity event",
        "tga_drawdown_liquidity_event": "TGA drawdown liquidity event",
        "tga_rebuild_liquidity_event": "TGA rebuild liquidity event",
        "on_rrp_mmf_absorption_state": "ON RRP / MMF absorption state",
        "low_rate_liquidity_state": "Low short-rate liquidity state",
        "reserve_scarcity_or_low_liquidity": "Reserve-scarcity / low-liquidity state",
    }.get(regime_id, regime_id.replace("_", " ").title())


def _trigger_for_regime(regime_id: str) -> str:
    mapping = {
        "normal_forward": "normal_forward_nonpandemic_baseline",
        "latest_rolling_persistence": "latest_rolling_persistence_window",
        "pandemic_fiscal_transfer_block": "pandemic_fiscal_transfer_block",
        "post_pandemic_nonblock": "post_pandemic_nonblock",
        "high_tdc_scale": "high_tdc_abs_q75",
        "high_liquidity_event": "reserve_change_abs_q75",
        "tga_drawdown_liquidity_event": "tga_drawdown_q25",
        "tga_rebuild_liquidity_event": "tga_rebuild_q75",
        "on_rrp_mmf_absorption_state": "on_rrp_flow_abs_q75",
        "low_rate_liquidity_state": "low_short_rate_q25",
        "reserve_scarcity_or_low_liquidity": "reserve_scarcity_low_reserve_q25",
    }
    return mapping.get(regime_id, regime_id)


def _trigger_text(regime_id: str) -> str:
    return {
        "normal_forward": "quarters outside 2020Q1-2021Q4; scenario baseline only",
        "latest_rolling_persistence": "latest available 48-quarter rolling selected-lag beta",
        "pandemic_exclusion_drop_2020": "latest 48-quarter rolling beta excluding 2020",
        "pandemic_exclusion_drop_2020q1_2021q4": "latest 48-quarter rolling beta excluding 2020Q1-2021Q4",
        "pandemic_exclusion_drop_2021": "latest 48-quarter rolling beta excluding 2021",
        "pandemic_fiscal_transfer_block": "quarter in 2020Q1-2021Q4",
        "post_pandemic_nonblock": "quarter >= 2022Q1",
        "high_tdc_scale": "abs(TDC) above current-sample q75",
        "high_liquidity_event": "abs(reserve balance qoq change) above current-sample q75",
        "tga_drawdown_liquidity_event": "TGA balance qoq change below current-sample q25",
        "tga_rebuild_liquidity_event": "TGA balance qoq change above current-sample q75",
        "on_rrp_mmf_absorption_state": "abs(ON RRP qoq change) above current-sample q75",
        "low_rate_liquidity_state": "fed funds below current-sample q25",
        "reserve_scarcity_or_low_liquidity": "reserve balances below current-sample q25",
    }.get(regime_id, "review-only trigger candidate")


def _write_memo(
    *,
    estimates: list[dict[str, Any]],
    validations: list[dict[str, Any]],
    contract_rows: list[dict[str, Any]],
) -> None:
    MEMO_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    contract_allowed = [row for row in contract_rows if row.get("recommended_ratewall_use") == "assumption_mode_scenario_allowed"]
    runtime_allowed = [row for row in contract_rows if row.get("runtime_selector_allowed") == "true"]
    latest_drop = {row["regime_id"]: row for row in estimates if str(row.get("horizon")) == "0"}
    lines = [
        "# EA-TDC Pass-Through Regime Validation Memo",
        "",
        "This package converts EA-TDC TDC-deposit pass-through diagnostics into source-bound RateWall Assumption Mode inputs. It does not claim causal deposit creation, denominator calibration, holder allocation, incidence, welfare, pricing, or runtime regime selection.",
        "",
        "## What Changed Relative To Pooled And Rolling Estimates",
        "",
        "- The pooled selected-lag full-sample h0 estimate remains the high historical reference.",
        "- Latest rolling persistence remains elevated, but pandemic-exclusion diagnostics lower it materially.",
        "- Trigger candidates are now explicit, source-hashed, and fail closed unless sample and validation checks support scenario use.",
        "",
        "## 2020 Versus 2021 Heterogeneity",
        "",
        f"- Latest rolling drop-2020 point: {_fmt(latest_drop.get('pandemic_exclusion_drop_2020', {}).get('point_estimate'))}.",
        f"- Latest rolling drop-2021 point: {_fmt(latest_drop.get('pandemic_exclusion_drop_2021', {}).get('point_estimate'))}.",
        "- The signs of the drop tests are heterogeneous: dropping 2020 lowers the latest rolling beta sharply, while dropping 2021 raises it. Treat the pandemic block as heterogeneous, not as a single permanent structural break.",
        "",
        "## Post-2020 Persistence",
        "",
        f"- Latest rolling excluding 2020Q1-2021Q4: {_fmt(latest_drop.get('pandemic_exclusion_drop_2020q1_2021q4', {}).get('point_estimate'))}.",
        "- The beta does not collapse to zero, so the evidence supports some post-pandemic persistence; however, validation is not strong enough for automatic runtime selection.",
        "",
        "## RateWall Use",
        "",
        f"- Assumption Mode scenario rows allowed: {len(contract_allowed)}.",
        f"- Runtime selector rows allowed: {len(runtime_allowed)}.",
        "- Current recommendation: import source-backed scenario rows for Assumption Mode review, keep runtime_selector_allowed=false.",
        "- Sample windows in estimates and the RateWall contract are complete cases after transformations, lags, controls, and factor availability; raw data coverage is reported separately.",
        "",
        "## No-TOTRESNS Robustness",
        "",
        f"- {_totresns_decision(estimates)['message']}",
        "",
        "## Runtime Selector Status",
        "",
        "No trigger rule is runtime-selector validated. Out-of-sample and false-positive checks are screening diagnostics only, not promotion-grade validation.",
        "",
    ]
    MEMO_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any) -> str:
    if value in ("", None):
        return "missing"
    return f"{float(value):.3f}"


def build_outputs() -> dict[str, list[dict[str, Any]]]:
    _paths, rows, control_ids, _factor_count, _screened_count, selected_credit_lags, selected_rate_lags = _build_inputs()
    controls = _canonical_controls(
        rows,
        _scenario_controls(control_ids, selected_credit_lags, selected_rate_lags)["selected_credit_rate_risk_lags"],
    )
    classifier = _candidate_specs(rows)
    estimates = _build_estimates(rows, controls, classifier)
    estimates.extend(_totresns_robustness_rows(rows, controls))
    validation = _validation_rows(estimates, classifier, rows)
    contract = _contract_rows(estimates, validation)
    return {
        "classifier_candidates": classifier,
        "regime_estimates": estimates,
        "regime_validation": validation,
        "ratewall_import_contract": contract,
    }


def _write_manifest(outputs: dict[str, list[dict[str, Any]]]) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "classifier_candidates": CLASSIFIER_OUTPUT,
        "regime_estimates": ESTIMATES_OUTPUT,
        "regime_validation": VALIDATION_OUTPUT,
        "ratewall_import_contract": CONTRACT_OUTPUT,
        "memo": MEMO_OUTPUT,
    }
    payload = {
        "job_id": JOB_ID,
        "outputs": {key: _relative(path) for key, path in output_paths.items()},
        "output_sha256": {key: _artifact_hash(path) for key, path in output_paths.items()},
        "row_counts": {key: len(value) for key, value in outputs.items()},
        "runtime_selector_allowed": False,
        "claim_boundary": "RateWall source-bound Assumption Mode inputs only; runtime selector remains blocked.",
    }
    MANIFEST_OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    outputs = build_outputs()
    _write_csv(CLASSIFIER_OUTPUT, outputs["classifier_candidates"])
    _write_csv(ESTIMATES_OUTPUT, outputs["regime_estimates"])
    _write_csv(VALIDATION_OUTPUT, outputs["regime_validation"])
    _write_csv(CONTRACT_OUTPUT, outputs["ratewall_import_contract"])
    _write_memo(
        estimates=outputs["regime_estimates"],
        validations=outputs["regime_validation"],
        contract_rows=outputs["ratewall_import_contract"],
    )
    _write_manifest(outputs)
    print(f"wrote {len(outputs['classifier_candidates'])} classifier rows to {CLASSIFIER_OUTPUT}")
    print(f"wrote {len(outputs['regime_estimates'])} estimate rows to {ESTIMATES_OUTPUT}")
    print(f"wrote {len(outputs['regime_validation'])} validation rows to {VALIDATION_OUTPUT}")
    print(f"wrote {len(outputs['ratewall_import_contract'])} contract rows to {CONTRACT_OUTPUT}")


if __name__ == "__main__":
    main()
