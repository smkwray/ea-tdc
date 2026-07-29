from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ea_tdc.estimation import RegressionFit, _ols
from ea_tdc.open_contract import (
    CANONICAL_CONTROL_IDS,
    CANONICAL_OUTCOME_ID,
    CANONICAL_RESIDUAL_ID,
    CANONICAL_TREATMENT_ID,
    CANONICAL_TREATMENT_SOURCE_SERIES,
    CREDIT_SCREEN_OUTCOME_IDS,
    EXPECTED_METHOD_TIER_COUNTS,
    METHOD_TIER_SERIES_ID,
    MMF_TREATMENT_IDS,
    OPEN01_DESIGN_JOB_IDS,
    OPEN_CONTRACT,
    OUTCOME_UNIT_MULTIPLIERS,
    ROLLING_WINDOW_QUARTERS,
)
from ea_tdc.utils import utc_now_iso


OPEN_ID = "OPEN-01"
IDENTITY_TOLERANCE = 1e-9
BREAK_QUARTER = "2020Q1"
INFLUENCE_BLOCK_QUARTERS = 4
CREDIT_WINDOW_QUARTERS = (40, ROLLING_WINDOW_QUARTERS, 60)
CREDIT_ADJUSTMENTS = (
    "raw",
    "share_2020_2021_adjusted",
    "linear_time_adjusted",
)
MATERIALITY_BANDS: Mapping[str, Any] = {
    "metric": "absolute_beta_change",
    "unit": "dollars_per_dollar_tdc",
    "stable_lte": 0.15,
    "review_lte": 0.30,
    "unstable_gt": 0.30,
}

_QUARTER_PATTERN = re.compile(r"^(?P<year>[0-9]{4})Q(?P<quarter>[1-4])$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Open01AcceptanceResult:
    contract_rows: list[dict[str, Any]]
    headline_rows: list[dict[str, Any]]
    stability_rows: list[dict[str, Any]]
    credit_screen_rows: list[dict[str, Any]]
    producer_status: str
    scientific_status: str
    issues: tuple[str, ...]
    coverage_counts: Mapping[str, int]


@dataclass(frozen=True)
class ProjectionEstimate:
    fit: RegressionFit
    n: int
    sample_start: str
    sample_end: str
    sample_hash: str
    beta: float
    se: float
    lower95: float
    upper95: float
    p_value: float
    control_ids_used: tuple[str, ...]
    control_ids_rejected: tuple[str, ...]


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _unique(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def _as_float(value: Any) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _quarter_ordinal(quarter: str) -> int:
    match = _QUARTER_PATTERN.fullmatch(str(quarter).strip())
    if match is None:
        raise ValueError(f"Invalid quarter label: {quarter!r}")
    return int(match.group("year")) * 4 + int(match.group("quarter")) - 1


def _quarter_sort_key(row: Mapping[str, Any]) -> int:
    return _quarter_ordinal(str(row.get("quarter", "")))


def _sample_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    quarters = "\n".join(str(row["quarter"]) for row in rows)
    return hashlib.sha256(quarters.encode("utf-8")).hexdigest()


def _normal_p(beta: float, se: float) -> float:
    if not math.isfinite(se) or se <= 0:
        raise ValueError("Inference requires a finite positive standard error")
    return math.erfc(abs(beta / se) / math.sqrt(2.0))


def _materiality_band(value: float) -> str:
    absolute = abs(value)
    if absolute <= float(MATERIALITY_BANDS["stable_lte"]):
        return "stable"
    if absolute <= float(MATERIALITY_BANDS["review_lte"]):
        return "review"
    return "unstable"


def _worst_materiality(values: Iterable[str]) -> str:
    order = {"stable": 0, "review": 1, "unstable": 2}
    present = list(values)
    if not present:
        return "not_computed"
    return max(present, key=lambda value: order.get(value, 3))


def _holm_adjust(rows: list[dict[str, Any]], *, family_key: str) -> None:
    families: dict[str, list[tuple[int, float]]] = {}
    for index, row in enumerate(rows):
        p_value = _as_float(row.get("p_value_raw"))
        if p_value is None:
            continue
        families.setdefault(str(row.get(family_key, "")), []).append((index, p_value))
    for members in families.values():
        ordered = sorted(members, key=lambda item: item[1])
        previous = 0.0
        count = len(ordered)
        for rank, (index, p_value) in enumerate(ordered):
            adjusted = min(1.0, max(previous, (count - rank) * p_value))
            rows[index]["p_value_holm"] = adjusted
            previous = adjusted


def _numeric_sample(
    rows: Sequence[Mapping[str, Any]],
    required_columns: Sequence[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        values: dict[str, float] = {}
        for column in required_columns:
            value = _as_float(row.get(column))
            if value is None:
                break
            values[column] = value
        else:
            output.append(
                {
                    "quarter": str(row["quarter"]),
                    METHOD_TIER_SERIES_ID: str(row.get(METHOD_TIER_SERIES_ID, "")).strip(),
                    **values,
                }
            )
    return output


def _consecutive(rows: Sequence[Mapping[str, Any]]) -> bool:
    ordinals = [_quarter_ordinal(str(row["quarter"])) for row in rows]
    return all(right == left + 1 for left, right in zip(ordinals, ordinals[1:]))


def _fit_projection(
    rows: Sequence[Mapping[str, Any]],
    *,
    treatment_id: str,
    outcome_id: str,
    control_ids: Sequence[str],
    covariance_lags: int = 1,
) -> ProjectionEstimate:
    required = _unique([treatment_id, outcome_id, *control_ids])
    numeric = _numeric_sample(rows, required)
    selected_controls: list[str] = []
    rejected_controls: list[str] = []
    for control_id in control_ids:
        values = [float(row[control_id]) for row in numeric]
        if not values or math.isclose(
            min(values),
            max(values),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            rejected_controls.append(control_id)
        else:
            selected_controls.append(control_id)
    parameter_count = len(selected_controls) + 2
    if len(numeric) <= parameter_count + 3:
        raise ValueError(
            f"Insufficient complete observations for {outcome_id} on {treatment_id}: "
            f"n={len(numeric)}, parameters={parameter_count}"
        )
    x_rows = [
        [
            1.0,
            row[treatment_id],
            *[row[control] for control in selected_controls],
        ]
        for row in numeric
    ]
    y_values = [row[outcome_id] for row in numeric]
    fit = _ols(
        y_values,
        x_rows,
        covariance_estimator="newey_west",
        covariance_lags=max(1, covariance_lags),
    )
    multiplier = float(OUTCOME_UNIT_MULTIPLIERS[outcome_id])
    beta = fit.beta[1] * multiplier
    se = fit.ses[1] * multiplier
    return ProjectionEstimate(
        fit=fit,
        n=len(numeric),
        sample_start=str(numeric[0]["quarter"]),
        sample_end=str(numeric[-1]["quarter"]),
        sample_hash=_sample_hash(numeric),
        beta=beta,
        se=se,
        lower95=beta - 1.96 * se,
        upper95=beta + 1.96 * se,
        p_value=_normal_p(beta, se),
        control_ids_used=tuple(selected_controls),
        control_ids_rejected=tuple(rejected_controls),
    )


def _fit_break(
    rows: Sequence[Mapping[str, Any]],
    *,
    break_quarter: str,
) -> dict[str, Any]:
    required = [
        CANONICAL_TREATMENT_ID,
        CANONICAL_OUTCOME_ID,
        *CANONICAL_CONTROL_IDS,
    ]
    numeric = _numeric_sample(rows, required)
    break_ordinal = _quarter_ordinal(break_quarter)
    pre_count = sum(
        _quarter_ordinal(str(row["quarter"])) < break_ordinal for row in numeric
    )
    post_count = len(numeric) - pre_count
    minimum_segment = max(12, len(CANONICAL_CONTROL_IDS) + 4)
    if min(pre_count, post_count) < minimum_segment:
        raise ValueError(
            f"Break {break_quarter} lacks a fixed minimum segment: "
            f"pre={pre_count}, post={post_count}, required={minimum_segment}"
        )
    post_values = [
        1.0
        if _quarter_ordinal(str(row["quarter"])) >= break_ordinal
        else 0.0
        for row in numeric
    ]
    post_is_existing_control = any(
        all(
            math.isclose(
                float(row[control_id]),
                post,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for row, post in zip(numeric, post_values, strict=True)
        )
        or all(
            math.isclose(
                float(row[control_id]),
                1.0 - post,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for row, post in zip(numeric, post_values, strict=True)
        )
        for control_id in CANONICAL_CONTROL_IDS
    )
    x_rows: list[list[float]] = []
    for row, post in zip(numeric, post_values, strict=True):
        treatment = row[CANONICAL_TREATMENT_ID]
        x_rows.append(
            [
                1.0,
                treatment,
                *[row[control] for control in CANONICAL_CONTROL_IDS],
                *([] if post_is_existing_control else [post]),
                treatment * post,
            ]
        )
    fit = _ols(
        [row[CANONICAL_OUTCOME_ID] for row in numeric],
        x_rows,
        covariance_estimator="newey_west",
        covariance_lags=1,
    )
    change = fit.beta[-1] * float(OUTCOME_UNIT_MULTIPLIERS[CANONICAL_OUTCOME_ID])
    change_se = fit.ses[-1] * float(
        OUTCOME_UNIT_MULTIPLIERS[CANONICAL_OUTCOME_ID]
    )
    return {
        "break_quarter": break_quarter,
        "n": len(numeric),
        "pre_n": pre_count,
        "post_n": post_count,
        "pre_beta": fit.beta[1],
        "post_beta": fit.beta[1] + fit.beta[-1],
        "beta_change": change,
        "se": change_se,
        "lower95": change - 1.96 * change_se,
        "upper95": change + 1.96 * change_se,
        "p_value_raw": _normal_p(change, change_se),
        "inference_method": "treatment_by_post_interaction_normal_hac",
        "covariance_estimator": "newey_west",
        "covariance_lags": 1,
        "materiality_band": _materiality_band(change),
        "sample_start": str(numeric[0]["quarter"]),
        "sample_end": str(numeric[-1]["quarter"]),
        "sample_hash": _sample_hash(numeric),
    }


def _fit_tier_interaction(
    rows: Sequence[Mapping[str, Any]],
    *,
    tier_id: str,
) -> dict[str, Any]:
    required = [
        CANONICAL_TREATMENT_ID,
        CANONICAL_OUTCOME_ID,
        *CANONICAL_CONTROL_IDS,
    ]
    numeric = _numeric_sample(rows, required)
    numeric = [
        row for row in numeric if str(row.get(METHOD_TIER_SERIES_ID, "")).strip()
    ]
    tier_count = sum(
        str(row[METHOD_TIER_SERIES_ID]).strip() == tier_id for row in numeric
    )
    complement_count = len(numeric) - tier_count
    if min(tier_count, complement_count) < 8:
        raise ValueError(
            f"Tier interaction {tier_id} lacks support: "
            f"tier={tier_count}, complement={complement_count}"
        )
    indicator_values = [
        1.0
        if str(row[METHOD_TIER_SERIES_ID]).strip() == tier_id
        else 0.0
        for row in numeric
    ]
    indicator_is_existing_control = any(
        all(
            math.isclose(
                float(row[control_id]),
                indicator,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for row, indicator in zip(numeric, indicator_values, strict=True)
        )
        for control_id in CANONICAL_CONTROL_IDS
    )
    x_rows: list[list[float]] = []
    for row, indicator in zip(numeric, indicator_values, strict=True):
        treatment = row[CANONICAL_TREATMENT_ID]
        x_rows.append(
            [
                1.0,
                treatment,
                *[row[control] for control in CANONICAL_CONTROL_IDS],
                *([] if indicator_is_existing_control else [indicator]),
                treatment * indicator,
            ]
        )
    fit = _ols(
        [row[CANONICAL_OUTCOME_ID] for row in numeric],
        x_rows,
        covariance_estimator="newey_west",
        covariance_lags=1,
    )
    multiplier = float(OUTCOME_UNIT_MULTIPLIERS[CANONICAL_OUTCOME_ID])
    change = fit.beta[-1] * multiplier
    se = fit.ses[-1] * multiplier
    return {
        "n": len(numeric),
        "tier_n": tier_count,
        "complement_n": complement_count,
        "complement_beta": fit.beta[1] * multiplier,
        "tier_beta": (fit.beta[1] + fit.beta[-1]) * multiplier,
        "beta_change": change,
        "se": se,
        "lower95": change - 1.96 * se,
        "upper95": change + 1.96 * se,
        "p_value_raw": _normal_p(change, se),
        "inference_method": "one_tier_vs_complement_interaction_normal_hac",
        "covariance_estimator": "newey_west",
        "covariance_lags": 1,
        "materiality_band": _materiality_band(change),
        "sample_start": str(numeric[0]["quarter"]),
        "sample_end": str(numeric[-1]["quarter"]),
        "sample_hash": _sample_hash(numeric),
    }


def _residualize(values: Sequence[float], controls: Sequence[Sequence[float]]) -> list[float]:
    if not values:
        raise ValueError("Cannot residualize an empty series")
    x_rows = [[1.0, *row] for row in controls]
    return _ols(
        list(values),
        x_rows,
        covariance_estimator="classical",
    ).residuals


def _standardize(values: Sequence[float]) -> list[float]:
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    scale = math.sqrt(sum(value * value for value in centered) / len(centered))
    if scale <= 0:
        raise ValueError("Association input is constant")
    return [value / scale for value in centered]


def _overlap_association(
    beta_values: Sequence[float],
    feature_values: Sequence[float],
    adjustment_values: Sequence[float] | None,
    *,
    window_quarters: int,
) -> dict[str, Any]:
    if len(beta_values) != len(feature_values) or len(beta_values) < 10:
        raise ValueError(
            f"Association requires at least 10 paired windows; n={len(beta_values)}"
        )
    if adjustment_values is None:
        beta_work = list(beta_values)
        feature_work = list(feature_values)
    else:
        if len(adjustment_values) != len(beta_values):
            raise ValueError("Adjustment series is not aligned with rolling windows")
        controls = [[value] for value in adjustment_values]
        beta_work = _residualize(beta_values, controls)
        feature_work = _residualize(feature_values, controls)
    beta_standard = _standardize(beta_work)
    feature_standard = _standardize(feature_work)
    covariance_lags = min(window_quarters - 1, len(beta_values) - 2)
    fit = _ols(
        beta_standard,
        [[1.0, value] for value in feature_standard],
        covariance_estimator="newey_west",
        covariance_lags=covariance_lags,
    )
    correlation = fit.beta[1]
    se = fit.ses[1]
    return {
        "correlation": correlation,
        "overlap_hac_se": se,
        "lower95": max(-1.0, correlation - 1.96 * se),
        "upper95": min(1.0, correlation + 1.96 * se),
        "lower95_unbounded": correlation - 1.96 * se,
        "upper95_unbounded": correlation + 1.96 * se,
        "p_value_raw": _normal_p(correlation, se),
        "covariance_estimator": "newey_west",
        "covariance_lags": covariance_lags,
    }


def _sign(value: float, *, tolerance: float = 1e-12) -> str:
    if value > tolerance:
        return "positive"
    if value < -tolerance:
        return "negative"
    return "zero"


def _valid_sha256(value: Any) -> bool:
    return _SHA256_PATTERN.fullmatch(str(value).strip().lower()) is not None


def build_treatment_outcome_contract(
    *,
    adapter_manifest: Mapping[str, Any],
    design_bundle_hashes: Mapping[str, str],
) -> list[dict[str, Any]]:
    seed_hash = str(adapter_manifest.get("bundle_hash", "")).strip().lower()
    combined_hash = str(
        adapter_manifest.get("combined_input_hash", "")
    ).strip().lower()
    input_hashes = adapter_manifest.get("input_hashes", {})
    if not _valid_sha256(seed_hash):
        raise ValueError("TDCest manifest lacks a valid seed-bundle SHA-256")
    if not _valid_sha256(combined_hash):
        raise ValueError("TDCest manifest lacks a valid combined-input SHA-256")
    if not isinstance(input_hashes, Mapping) or not input_hashes:
        raise ValueError("TDCest manifest lacks its component input hashes")
    if any(not _valid_sha256(value) for value in input_hashes.values()):
        raise ValueError("TDCest manifest contains an invalid component input hash")
    expected_jobs = set(OPEN01_DESIGN_JOB_IDS)
    if set(design_bundle_hashes) != expected_jobs:
        raise ValueError(
            "Design-bundle hash keys do not match the four canonical OPEN-01 jobs"
        )
    if any(not _valid_sha256(value) for value in design_bundle_hashes.values()):
        raise ValueError("At least one OPEN-01 design-bundle hash is invalid")

    roles = [
        (CANONICAL_OUTCOME_ID, "headline_deposit_outcome"),
        (CANONICAL_RESIDUAL_ID, "same_treatment_identity_residual"),
        *[
            (outcome_id, "predeclared_credit_screen")
            for outcome_id in CREDIT_SCREEN_OUTCOME_IDS
        ],
    ]
    rows: list[dict[str, Any]] = []
    for outcome_id, role in roles:
        multiplier = float(OUTCOME_UNIT_MULTIPLIERS[outcome_id])
        outcome_units = (
            OPEN_CONTRACT.deposit_outcome_units
            if multiplier == 1.0
            else OPEN_CONTRACT.credit_outcome_units
        )
        rows.append(
            {
                "open_id": OPEN_ID,
                "contract_status": "frozen",
                "treatment_label": OPEN_CONTRACT.treatment_label,
                "treatment_id": CANONICAL_TREATMENT_ID,
                "treatment_source_series": CANONICAL_TREATMENT_SOURCE_SERIES,
                "outcome_id": outcome_id,
                "outcome_role": role,
                "treatment_units": OPEN_CONTRACT.treatment_units,
                "outcome_units": outcome_units,
                "normalized_estimand_units": "dollars_per_dollar_tdc",
                "outcome_unit_multiplier": multiplier,
                "sign_convention": OPEN_CONTRACT.sign_convention,
                "clock": OPEN_CONTRACT.clock,
                "treatment_perimeter": OPEN_CONTRACT.treatment_perimeter,
                "outcome_perimeter": OPEN_CONTRACT.outcome_perimeter,
                "construction_tier_policy": OPEN_CONTRACT.construction_tier_policy,
                "method_tier_series_id": METHOD_TIER_SERIES_ID,
                "method_tier_expected_counts_json": _json(
                    dict(EXPECTED_METHOD_TIER_COUNTS)
                ),
                "mmf_treatment_ids_json": _json(list(MMF_TREATMENT_IDS)),
                "canonical_control_ids_json": _json(list(CANONICAL_CONTROL_IDS)),
                "embedded_bank_treasury_component_id": (
                    OPEN_CONTRACT.embedded_bank_treasury_component_id
                ),
                "tdcest_seed_bundle_sha256": seed_hash,
                "tdcest_combined_input_sha256": combined_hash,
                "tdcest_input_hashes_json": _json(dict(input_hashes)),
                "design_bundle_hashes_json": _json(dict(design_bundle_hashes)),
                "claim_boundary": OPEN_CONTRACT.claim_boundary,
            }
        )
    return rows


def _failure_result(
    *,
    contract_rows: Sequence[Mapping[str, Any]],
    issues: Sequence[str],
    coverage_counts: Mapping[str, int],
) -> Open01AcceptanceResult:
    issue_json = _json(list(issues))
    failure = {
        "open_id": OPEN_ID,
        "status": "failed",
        "producer_status": "fail",
        "issue_count": len(issues),
        "issues_json": issue_json,
        "coverage_counts_json": _json(dict(coverage_counts)),
    }
    return Open01AcceptanceResult(
        contract_rows=[{**row, **failure} for row in contract_rows]
        or [dict(failure)],
        headline_rows=[dict(failure)],
        stability_rows=[
            {
                **failure,
                "test_type": "producer_validation",
                "materiality_bands_json": _json(MATERIALITY_BANDS),
            }
        ],
        credit_screen_rows=[dict(failure)],
        producer_status="fail",
        scientific_status="not_computed",
        issues=tuple(issues),
        coverage_counts=dict(coverage_counts),
    )


def _influence_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    full_beta: float,
    block_quarters: int,
) -> dict[str, Any]:
    estimates: list[dict[str, Any]] = []
    if block_quarters == 1:
        blocks = [(index, index + 1) for index in range(len(rows))]
    else:
        blocks = [
            (index, index + block_quarters)
            for index in range(0, len(rows) - block_quarters + 1)
        ]
    for start, end in blocks:
        subset = [*rows[:start], *rows[end:]]
        estimate = _fit_projection(
            subset,
            treatment_id=CANONICAL_TREATMENT_ID,
            outcome_id=CANONICAL_OUTCOME_ID,
            control_ids=CANONICAL_CONTROL_IDS,
        )
        omitted = [str(row["quarter"]) for row in rows[start:end]]
        estimates.append(
            {
                "omitted": omitted,
                "beta": estimate.beta,
                "delta": estimate.beta - full_beta,
            }
        )
    worst = max(estimates, key=lambda row: abs(float(row["delta"])))
    max_delta = abs(float(worst["delta"]))
    return {
        "omitted_unit_quarters": block_quarters,
        "checks_computed": len(estimates),
        "worst_omitted": ",".join(worst["omitted"]),
        "worst_beta": worst["beta"],
        "max_abs_beta_change": max_delta,
        "minimum_beta": min(float(row["beta"]) for row in estimates),
        "maximum_beta": max(float(row["beta"]) for row in estimates),
        "materiality_band": _materiality_band(max_delta),
        "inference_method": "fixed_control_refit_newey_west",
        "covariance_estimator": "newey_west",
        "covariance_lags": 1,
        "details_json": _json(estimates),
    }


def _credit_screen(
    rows: Sequence[Mapping[str, Any]],
    *,
    issues: list[str],
) -> list[dict[str, Any]]:
    required = [
        CANONICAL_TREATMENT_ID,
        CANONICAL_OUTCOME_ID,
        *CANONICAL_CONTROL_IDS,
        *CREDIT_SCREEN_OUTCOME_IDS,
    ]
    numeric = _numeric_sample(rows, _unique(required))
    if len(numeric) < max(CREDIT_WINDOW_QUARTERS):
        issues.append(
            "credit_screen_insufficient_common_sample:"
            f"{len(numeric)}<{max(CREDIT_WINDOW_QUARTERS)}"
        )
        return []
    if not _consecutive(numeric):
        issues.append("credit_screen_common_sample_has_calendar_gaps")
        return []

    rolling: dict[int, list[dict[str, Any]]] = {}
    for window_quarters in CREDIT_WINDOW_QUARTERS:
        window_rows: list[dict[str, Any]] = []
        for end in range(window_quarters - 1, len(numeric)):
            subset = numeric[end - window_quarters + 1 : end + 1]
            try:
                estimate = _fit_projection(
                    subset,
                    treatment_id=CANONICAL_TREATMENT_ID,
                    outcome_id=CANONICAL_OUTCOME_ID,
                    control_ids=CANONICAL_CONTROL_IDS,
                )
            except ValueError as exc:
                issues.append(
                    f"credit_rolling_fit_failed:{window_quarters}:"
                    f"{subset[0]['quarter']}:{subset[-1]['quarter']}:{exc}"
                )
                return []
            share_2020_2021 = sum(
                _quarter_ordinal("2020Q1")
                <= _quarter_ordinal(str(row["quarter"]))
                <= _quarter_ordinal("2021Q4")
                for row in subset
            ) / window_quarters
            payload: dict[str, Any] = {
                "window_start": str(subset[0]["quarter"]),
                "window_end": str(subset[-1]["quarter"]),
                "window_end_index": _quarter_ordinal(str(subset[-1]["quarter"])),
                "share_2020_2021": share_2020_2021,
                "deposit_beta": estimate.beta,
                "deposit_beta_n": estimate.n,
                "deposit_beta_sample_hash": estimate.sample_hash,
                "control_ids_used": ",".join(estimate.control_ids_used),
                "control_ids_rejected": ",".join(
                    estimate.control_ids_rejected
                ),
            }
            for outcome_id in CREDIT_SCREEN_OUTCOME_IDS:
                multiplier = float(OUTCOME_UNIT_MULTIPLIERS[outcome_id])
                payload[outcome_id] = (
                    sum(float(row[outcome_id]) for row in subset)
                    / window_quarters
                    * multiplier
                )
            window_rows.append(payload)
        rolling[window_quarters] = window_rows

    output: list[dict[str, Any]] = []
    for window_quarters, window_rows in rolling.items():
        beta_values = [float(row["deposit_beta"]) for row in window_rows]
        for outcome_id in CREDIT_SCREEN_OUTCOME_IDS:
            feature_values = [float(row[outcome_id]) for row in window_rows]
            for adjustment in CREDIT_ADJUSTMENTS:
                if adjustment == "raw":
                    adjustment_values = None
                elif adjustment == "share_2020_2021_adjusted":
                    adjustment_values = [
                        float(row["share_2020_2021"]) for row in window_rows
                    ]
                else:
                    adjustment_values = [
                        float(row["window_end_index"]) for row in window_rows
                    ]
                try:
                    association = _overlap_association(
                        beta_values,
                        feature_values,
                        adjustment_values,
                        window_quarters=window_quarters,
                    )
                except ValueError as exc:
                    issues.append(
                        f"credit_association_failed:{outcome_id}:"
                        f"{window_quarters}:{adjustment}:{exc}"
                    )
                    continue
                output.append(
                    {
                        "open_id": OPEN_ID,
                        "status": "computed",
                        "screen_family": "predeclared_credit_admission",
                        "credit_outcome_id": outcome_id,
                        "treatment_id": CANONICAL_TREATMENT_ID,
                        "rolling_outcome_id": CANONICAL_OUTCOME_ID,
                        "control_ids": ",".join(CANONICAL_CONTROL_IDS),
                        "window_quarters": window_quarters,
                        "feature_stat": "window_mean",
                        "adjustment": adjustment,
                        "n_windows": len(window_rows),
                        "first_window_start": window_rows[0]["window_start"],
                        "first_window_end": window_rows[0]["window_end"],
                        "last_window_start": window_rows[-1]["window_start"],
                        "last_window_end": window_rows[-1]["window_end"],
                        "last_observed_treatment_outcome_quarter": numeric[-1][
                            "quarter"
                        ],
                        "rolling_effective_n": window_quarters,
                        "rolling_control_patterns_json": _json(
                            sorted(
                                {
                                    (
                                        str(row["control_ids_used"]),
                                        str(row["control_ids_rejected"]),
                                    )
                                    for row in window_rows
                                }
                            )
                        ),
                        "association_units": "standardized_rolling_window_correlation",
                        "claim_boundary": (
                            "Descriptive association across overlapping rolling "
                            "windows; not an independent-observation or causal test."
                        ),
                        "multiple_testing_family": (
                            f"credit_{window_quarters}_{adjustment}"
                        ),
                        **association,
                    }
                )

    _holm_adjust(output, family_key="multiple_testing_family")
    by_outcome_adjustment: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    by_outcome: dict[str, list[dict[str, Any]]] = {}
    for row in output:
        outcome_id = str(row["credit_outcome_id"])
        adjustment = str(row["adjustment"])
        window = int(row["window_quarters"])
        by_outcome_adjustment.setdefault((outcome_id, adjustment), {})[window] = row
        by_outcome.setdefault(outcome_id, []).append(row)

    for (outcome_id, adjustment), window_rows in by_outcome_adjustment.items():
        signs = {
            window: _sign(float(window_rows[window]["correlation"]))
            for window in CREDIT_WINDOW_QUARTERS
            if window in window_rows
        }
        stable = (
            set(signs) == set(CREDIT_WINDOW_QUARTERS)
            and len(set(signs.values())) == 1
            and "zero" not in signs.values()
        )
        for row in window_rows.values():
            row["sign_40"] = signs.get(40, "missing")
            row["sign_48"] = signs.get(ROLLING_WINDOW_QUARTERS, "missing")
            row["sign_60"] = signs.get(60, "missing")
            row["sign_stable_40_48_60"] = stable

    for outcome_id, outcome_rows in by_outcome.items():
        signs = [_sign(float(row["correlation"])) for row in outcome_rows]
        same_sign = len(signs) == 9 and len(set(signs)) == 1 and "zero" not in signs
        intervals_exclude_zero = len(outcome_rows) == 9 and all(
            float(row["lower95_unbounded"]) > 0
            or float(row["upper95_unbounded"]) < 0
            for row in outcome_rows
        )
        holm_significant = len(outcome_rows) == 9 and all(
            _as_float(row.get("p_value_holm")) is not None
            and float(row["p_value_holm"]) <= 0.05
            for row in outcome_rows
        )
        admitted = same_sign and intervals_exclude_zero and holm_significant
        for row in outcome_rows:
            row["all_adjustment_window_signs_stable"] = same_sign
            row["all_overlap_intervals_exclude_zero"] = intervals_exclude_zero
            row["all_holm_p_lte_0_05"] = holm_significant
            row["admission_status"] = (
                "main_text_eligible" if admitted else "appendix_only"
            )
    return output


def _credit_screen_cartesian(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    window_contract_exact = tuple(CREDIT_WINDOW_QUARTERS) == (40, 48, 60)
    adjustment_contract_exact = tuple(CREDIT_ADJUSTMENTS) == (
        "raw",
        "share_2020_2021_adjusted",
        "linear_time_adjusted",
    )
    expected_cells = {
        (outcome_id, str(window_quarters), adjustment)
        for outcome_id in CREDIT_SCREEN_OUTCOME_IDS
        for window_quarters in CREDIT_WINDOW_QUARTERS
        for adjustment in CREDIT_ADJUSTMENTS
    }
    observed_cells = [
        (
            str(row.get("credit_outcome_id", "")).strip(),
            str(row.get("window_quarters", "")).strip(),
            str(row.get("adjustment", "")).strip(),
        )
        for row in rows
    ]
    cell_counts: dict[tuple[str, str, str], int] = {}
    for cell in observed_cells:
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
    observed_set = set(observed_cells)
    missing = sorted(expected_cells.difference(observed_set))
    unexpected = sorted(observed_set.difference(expected_cells))
    duplicates = sorted(
        cell for cell, count in cell_counts.items() if count > 1
    )
    return {
        "passed": (
            window_contract_exact
            and adjustment_contract_exact
            and len(observed_cells) == len(expected_cells)
            and not missing
            and not unexpected
            and not duplicates
        ),
        "expected_outcome_ids": list(CREDIT_SCREEN_OUTCOME_IDS),
        "expected_window_quarters": [40, 48, 60],
        "expected_adjustments": [
            "raw",
            "share_2020_2021_adjusted",
            "linear_time_adjusted",
        ],
        "expected_rows": len(expected_cells),
        "observed_rows": len(observed_cells),
        "missing_cells": [list(cell) for cell in missing],
        "unexpected_cells": [list(cell) for cell in unexpected],
        "duplicate_cells": [
            {"cell": list(cell), "count": cell_counts[cell]}
            for cell in duplicates
        ],
    }


def build_open01_acceptance(
    rows: Sequence[Mapping[str, Any]],
    *,
    contract_rows: Sequence[Mapping[str, Any]],
    identity_tolerance: float = IDENTITY_TOLERANCE,
) -> Open01AcceptanceResult:
    issues: list[str] = []
    coverage_counts: dict[str, int] = {}
    if not contract_rows:
        issues.append("missing_treatment_outcome_contract")
    if not rows:
        issues.append("empty_quarterly_input")
        return _failure_result(
            contract_rows=contract_rows,
            issues=issues,
            coverage_counts=coverage_counts,
        )

    try:
        ordered = sorted(rows, key=_quarter_sort_key)
    except ValueError as exc:
        issues.append(f"invalid_quarter:{exc}")
        return _failure_result(
            contract_rows=contract_rows,
            issues=issues,
            coverage_counts=coverage_counts,
        )
    quarters = [str(row.get("quarter", "")) for row in ordered]
    if len(set(quarters)) != len(quarters):
        issues.append("duplicate_quarters")

    required_numeric = _unique(
        [
            CANONICAL_TREATMENT_ID,
            CANONICAL_OUTCOME_ID,
            CANONICAL_RESIDUAL_ID,
            *CANONICAL_CONTROL_IDS,
            *MMF_TREATMENT_IDS,
            *CREDIT_SCREEN_OUTCOME_IDS,
        ]
    )
    for column in required_numeric:
        coverage_counts[column] = sum(
            _as_float(row.get(column)) is not None for row in ordered
        )
        if coverage_counts[column] == 0:
            issues.append(f"required_numeric_series_absent:{column}")

    canonical_rows = [
        row
        for row in ordered
        if _as_float(row.get(CANONICAL_TREATMENT_ID)) is not None
    ]
    tier_counts = {
        tier: sum(
            str(row.get(METHOD_TIER_SERIES_ID, "")).strip() == tier
            for row in canonical_rows
        )
        for tier in EXPECTED_METHOD_TIER_COUNTS
    }
    unknown_tier_count = sum(
        str(row.get(METHOD_TIER_SERIES_ID, "")).strip()
        not in EXPECTED_METHOD_TIER_COUNTS
        for row in canonical_rows
    )
    coverage_counts[f"{METHOD_TIER_SERIES_ID}:unknown_or_missing"] = (
        unknown_tier_count
    )
    for tier_id, expected in EXPECTED_METHOD_TIER_COUNTS.items():
        actual = tier_counts[tier_id]
        coverage_counts[f"{METHOD_TIER_SERIES_ID}:{tier_id}"] = actual
        if actual != expected:
            issues.append(
                f"method_tier_count_mismatch:{tier_id}:{actual}!={expected}"
            )
    if unknown_tier_count:
        issues.append(f"unknown_or_missing_method_tier:{unknown_tier_count}")

    if issues:
        return _failure_result(
            contract_rows=contract_rows,
            issues=_unique(issues),
            coverage_counts=coverage_counts,
        )

    headline_required = [
        CANONICAL_TREATMENT_ID,
        CANONICAL_OUTCOME_ID,
        CANONICAL_RESIDUAL_ID,
        *CANONICAL_CONTROL_IDS,
    ]
    headline_sample = _numeric_sample(ordered, headline_required)
    coverage_counts["headline_common_sample"] = len(headline_sample)
    if not _consecutive(headline_sample):
        issues.append("headline_common_sample_has_calendar_gaps")

    headline_rows: list[dict[str, Any]] = []
    headline_estimates: dict[str, ProjectionEstimate] = {}
    for outcome_id, role in (
        (CANONICAL_OUTCOME_ID, "headline_conditional_projection"),
        (CANONICAL_RESIDUAL_ID, "accounting_identity_diagnostic"),
    ):
        try:
            estimate = _fit_projection(
                headline_sample,
                treatment_id=CANONICAL_TREATMENT_ID,
                outcome_id=outcome_id,
                control_ids=CANONICAL_CONTROL_IDS,
            )
        except ValueError as exc:
            issues.append(f"headline_not_estimable:{outcome_id}:{exc}")
            continue
        headline_estimates[outcome_id] = estimate
        headline_rows.append(
            {
                "open_id": OPEN_ID,
                "status": "estimated",
                "estimand_role": role,
                "treatment_id": CANONICAL_TREATMENT_ID,
                "outcome_id": outcome_id,
                "control_ids": ",".join(CANONICAL_CONTROL_IDS),
                "control_ids_used": ",".join(estimate.control_ids_used),
                "control_ids_rejected": ",".join(
                    estimate.control_ids_rejected
                ),
                "beta": estimate.beta,
                "se": estimate.se,
                "lower95": estimate.lower95,
                "upper95": estimate.upper95,
                "p_value_raw": estimate.p_value,
                "multiple_testing_family": "headline_same_quarter",
                "inference_method": "same_quarter_ols_normal_hac",
                "covariance_estimator": "newey_west",
                "covariance_lags": 1,
                "n": estimate.n,
                "sample_start": estimate.sample_start,
                "sample_end": estimate.sample_end,
                "sample_hash": estimate.sample_hash,
                "units": "dollars_per_dollar_tdc",
                "clock": OPEN_CONTRACT.clock,
                "claim_status": (
                    "conditional_projection_not_causal"
                    if outcome_id == CANONICAL_OUTCOME_ID
                    else "accounting_complement_not_independent_mechanism"
                ),
            }
        )
    _holm_adjust(headline_rows, family_key="multiple_testing_family")

    identity_gap: float | None = None
    identity_pass = False
    if set(headline_estimates) == {CANONICAL_OUTCOME_ID, CANONICAL_RESIDUAL_ID}:
        deposit_beta = headline_estimates[CANONICAL_OUTCOME_ID].beta
        residual_beta = headline_estimates[CANONICAL_RESIDUAL_ID].beta
        identity_gap = deposit_beta - 1.0 - residual_beta
        identity_pass = abs(identity_gap) <= identity_tolerance
        if not identity_pass:
            issues.append(
                f"deposit_residual_identity_failed:{identity_gap}:"
                f"tolerance={identity_tolerance}"
            )
        for row in headline_rows:
            row["identity_gap"] = identity_gap
            row["identity_tolerance"] = identity_tolerance
            row["identity_status"] = "pass" if identity_pass else "fail"

    stability_rows: list[dict[str, Any]] = [
        {
            "open_id": OPEN_ID,
            "status": "computed",
            "test_type": "fixed_sample_contract",
            "test_id": "headline_common_sample",
            "n": len(headline_sample),
            "sample_start": headline_sample[0]["quarter"] if headline_sample else "",
            "sample_end": headline_sample[-1]["quarter"] if headline_sample else "",
            "sample_hash": _sample_hash(headline_sample) if headline_sample else "",
            "control_ids": ",".join(CANONICAL_CONTROL_IDS),
            "coverage_counts_json": _json(coverage_counts),
            "method_tier_counts_json": _json(tier_counts),
            "materiality_bands_json": _json(MATERIALITY_BANDS),
            "scientific_status": "fixed",
        },
        {
            "open_id": OPEN_ID,
            "status": "computed" if identity_gap is not None else "failed",
            "test_type": "accounting_identity",
            "test_id": "deposit_beta_equals_one_plus_residual_beta",
            "statistic": identity_gap if identity_gap is not None else "",
            "threshold": identity_tolerance,
            "materiality_bands_json": _json(MATERIALITY_BANDS),
            "scientific_status": "stable" if identity_pass else "unstable",
        },
    ]

    declared_break: dict[str, Any] | None = None
    try:
        declared_break = _fit_break(
            headline_sample,
            break_quarter=BREAK_QUARTER,
        )
        declared_break["p_value_holm"] = declared_break["p_value_raw"]
        stability_rows.append(
            {
                "open_id": OPEN_ID,
                "status": "computed",
                "test_type": "declared_break",
                "test_id": f"beta_break_at_{BREAK_QUARTER}",
                "multiple_testing_family": "declared_break",
                "materiality_bands_json": _json(MATERIALITY_BANDS),
                "scientific_status": declared_break["materiality_band"],
                **declared_break,
            }
        )
    except ValueError as exc:
        issues.append(f"declared_break_failed:{exc}")

    minimum_segment = max(12, len(CANONICAL_CONTROL_IDS) + 4)
    unknown_break_rows: list[dict[str, Any]] = []
    candidate_quarters = [
        str(headline_sample[index]["quarter"])
        for index in range(
            minimum_segment,
            len(headline_sample) - minimum_segment + 1,
        )
    ]
    for break_quarter in candidate_quarters:
        try:
            estimate = _fit_break(
                headline_sample,
                break_quarter=break_quarter,
            )
        except ValueError as exc:
            issues.append(f"unknown_break_candidate_failed:{break_quarter}:{exc}")
            continue
        unknown_break_rows.append(
            {
                "open_id": OPEN_ID,
                "status": "computed",
                "test_type": "unknown_break_candidate",
                "test_id": f"candidate_{break_quarter}",
                "multiple_testing_family": "unknown_break_scan",
                "materiality_bands_json": _json(MATERIALITY_BANDS),
                "scientific_status": estimate["materiality_band"],
                **estimate,
            }
        )
    _holm_adjust(unknown_break_rows, family_key="multiple_testing_family")
    if unknown_break_rows:
        scan_extremum = min(
            unknown_break_rows,
            key=lambda row: float(row["p_value_raw"]),
        )
        for row in unknown_break_rows:
            row["is_scan_extremum"] = row is scan_extremum
        stability_rows.extend(unknown_break_rows)
        stability_rows.append(
            {
                "open_id": OPEN_ID,
                "status": "computed",
                "test_type": "unknown_break_scan_summary",
                "test_id": "minimum_raw_p_candidate",
                "selected_break_quarter": scan_extremum["break_quarter"],
                "beta_change": scan_extremum["beta_change"],
                "p_value_raw": scan_extremum["p_value_raw"],
                "p_value_holm": scan_extremum.get("p_value_holm", ""),
                "candidates_tested": len(unknown_break_rows),
                "inference_method": (
                    "HAC treatment-by-post interaction scan with "
                    "Holm adjustment across candidate breaks"
                ),
                "materiality_bands_json": _json(MATERIALITY_BANDS),
                "scientific_status": scan_extremum["materiality_band"],
            }
        )
    else:
        issues.append("unknown_break_scan_produced_no_candidates")

    full_beta = (
        headline_estimates[CANONICAL_OUTCOME_ID].beta
        if CANONICAL_OUTCOME_ID in headline_estimates
        else None
    )
    influence_bands: list[str] = []
    if full_beta is not None:
        for block_quarters, test_type in (
            (1, "leave_quarter_out_influence"),
            (INFLUENCE_BLOCK_QUARTERS, "leave_block_out_influence"),
        ):
            try:
                summary = _influence_summary(
                    headline_sample,
                    full_beta=full_beta,
                    block_quarters=block_quarters,
                )
            except ValueError as exc:
                issues.append(f"{test_type}_failed:{exc}")
                continue
            influence_bands.append(str(summary["materiality_band"]))
            stability_rows.append(
                {
                    "open_id": OPEN_ID,
                    "status": "computed",
                    "test_type": test_type,
                    "test_id": (
                        "all_single_quarters"
                        if block_quarters == 1
                        else f"all_rolling_{block_quarters}q_blocks"
                    ),
                    "full_beta": full_beta,
                    "materiality_bands_json": _json(MATERIALITY_BANDS),
                    "scientific_status": summary["materiality_band"],
                    **summary,
                }
            )

    tier_rows: list[dict[str, Any]] = []
    for tier_id, expected_count in EXPECTED_METHOD_TIER_COUNTS.items():
        try:
            estimate = _fit_tier_interaction(
                headline_sample,
                tier_id=tier_id,
            )
        except ValueError as exc:
            issues.append(f"method_tier_sensitivity_failed:{tier_id}:{exc}")
            continue
        tier_rows.append(
            {
                "open_id": OPEN_ID,
                "status": "computed",
                "test_type": "construction_tier_sensitivity",
                "test_id": tier_id,
                "method_tier_id": tier_id,
                "expected_tier_count": expected_count,
                "observed_tier_count": tier_counts[tier_id],
                "multiple_testing_family": "construction_tier_interactions",
                "materiality_bands_json": _json(MATERIALITY_BANDS),
                "scientific_status": estimate["materiality_band"],
                **estimate,
            }
        )
    _holm_adjust(tier_rows, family_key="multiple_testing_family")
    stability_rows.extend(tier_rows)

    mmf_required = [
        *MMF_TREATMENT_IDS,
        CANONICAL_OUTCOME_ID,
        *CANONICAL_CONTROL_IDS,
    ]
    headline_quarters = {str(row["quarter"]) for row in headline_sample}
    mmf_sample = _numeric_sample(
        [
            row
            for row in ordered
            if str(row.get("quarter", "")) in headline_quarters
        ],
        _unique(mmf_required),
    )
    mmf_rows: list[dict[str, Any]] = []
    mmf_betas: dict[str, float] = {}
    for bound, treatment_id in zip(
        ("lower", "proportional", "upper"),
        MMF_TREATMENT_IDS,
        strict=True,
    ):
        try:
            estimate = _fit_projection(
                mmf_sample,
                treatment_id=treatment_id,
                outcome_id=CANONICAL_OUTCOME_ID,
                control_ids=CANONICAL_CONTROL_IDS,
            )
        except ValueError as exc:
            issues.append(f"mmf_sensitivity_failed:{bound}:{exc}")
            continue
        mmf_betas[bound] = estimate.beta
        mmf_rows.append(
            {
                "open_id": OPEN_ID,
                "status": "computed",
                "test_type": "mmf_bound_sensitivity",
                "test_id": bound,
                "mmf_bound": bound,
                "treatment_id": treatment_id,
                "beta": estimate.beta,
                "se": estimate.se,
                "lower95": estimate.lower95,
                "upper95": estimate.upper95,
                "p_value_raw": estimate.p_value,
                "multiple_testing_family": "mmf_bound_estimates",
                "inference_method": "fixed_common_sample_ols_normal_hac",
                "covariance_estimator": "newey_west",
                "covariance_lags": 1,
                "n": estimate.n,
                "sample_start": estimate.sample_start,
                "sample_end": estimate.sample_end,
                "sample_hash": estimate.sample_hash,
                "materiality_bands_json": _json(MATERIALITY_BANDS),
            }
        )
    if "proportional" in mmf_betas:
        proportional = mmf_betas["proportional"]
        for row in mmf_rows:
            delta = float(row["beta"]) - proportional
            row["beta_change"] = delta
            row["materiality_band"] = _materiality_band(delta)
            row["scientific_status"] = row["materiality_band"]
    _holm_adjust(mmf_rows, family_key="multiple_testing_family")
    stability_rows.extend(mmf_rows)
    mmf_band = "not_computed"
    if len(mmf_betas) == 3:
        spread = max(mmf_betas.values()) - min(mmf_betas.values())
        mmf_band = _materiality_band(spread)
        stability_rows.append(
            {
                "open_id": OPEN_ID,
                "status": "computed",
                "test_type": "mmf_bound_sensitivity_summary",
                "test_id": "lower_proportional_upper_range",
                "minimum_beta": min(mmf_betas.values()),
                "maximum_beta": max(mmf_betas.values()),
                "max_abs_beta_change": spread,
                "materiality_bands_json": _json(MATERIALITY_BANDS),
                "scientific_status": mmf_band,
            }
        )

    credit_rows = _credit_screen(ordered, issues=issues)
    expected_credit_rows = (
        len(CREDIT_SCREEN_OUTCOME_IDS)
        * len(CREDIT_WINDOW_QUARTERS)
        * len(CREDIT_ADJUSTMENTS)
    )
    coverage_counts["credit_screen_rows"] = len(credit_rows)
    if len(credit_rows) != expected_credit_rows:
        issues.append(
            f"credit_screen_row_count:{len(credit_rows)}!={expected_credit_rows}"
        )
    credit_cartesian = _credit_screen_cartesian(credit_rows)
    if not credit_cartesian["passed"]:
        issues.append(
            "credit_screen_cartesian:"
            f"missing={len(credit_cartesian['missing_cells'])}:"
            f"unexpected={len(credit_cartesian['unexpected_cells'])}:"
            f"duplicates={len(credit_cartesian['duplicate_cells'])}"
        )

    relevant_bands = [
        "stable" if identity_pass else "unstable",
        *influence_bands,
        *[str(row["scientific_status"]) for row in tier_rows],
        mmf_band,
    ]
    if declared_break is not None:
        relevant_bands.append(str(declared_break["materiality_band"]))
    if unknown_break_rows:
        relevant_bands.append(
            str(
                min(
                    unknown_break_rows,
                    key=lambda row: float(row["p_value_raw"]),
                )["materiality_band"]
            )
        )
    scientific_status = _worst_materiality(
        band for band in relevant_bands if band != "not_computed"
    )
    issues = _unique(issues)
    producer_status = "pass" if not issues else "fail"
    stability_rows.append(
        {
            "open_id": OPEN_ID,
            "status": "computed" if producer_status == "pass" else "failed",
            "test_type": "overall_gate",
            "test_id": "open01_stability_execution_and_science",
            "producer_status": producer_status,
            "scientific_status": scientific_status,
            "issue_count": len(issues),
            "issues_json": _json(issues),
            "materiality_bands_json": _json(MATERIALITY_BANDS),
            "claim_boundary": (
                "Producer pass means every predeclared check was computed. "
                "Scientific status separately governs promotion strength."
            ),
        }
    )

    annotations = {
        "producer_status": producer_status,
        "scientific_status": scientific_status,
        "issue_count": len(issues),
        "issues_json": _json(issues),
        "coverage_counts_json": _json(coverage_counts),
    }
    output_contract_rows = [{**row, **annotations} for row in contract_rows]
    for table in (headline_rows, stability_rows, credit_rows):
        for row in table:
            row.setdefault("producer_status", producer_status)
            row.setdefault("overall_scientific_status", scientific_status)
            row.setdefault("issue_count", len(issues))
            row.setdefault("issues_json", _json(issues))

    if not headline_rows:
        headline_rows = [{**annotations, "open_id": OPEN_ID, "status": "failed"}]
    if not credit_rows:
        credit_rows = [{**annotations, "open_id": OPEN_ID, "status": "failed"}]
    return Open01AcceptanceResult(
        contract_rows=output_contract_rows,
        headline_rows=headline_rows,
        stability_rows=stability_rows,
        credit_screen_rows=credit_rows,
        producer_status=producer_status,
        scientific_status=scientific_status,
        issues=tuple(issues),
        coverage_counts=coverage_counts,
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty acceptance table: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(
    passed: bool,
    *,
    details: Any,
) -> dict[str, Any]:
    return {"passed": bool(passed), "details": details}


def _acceptance_checks(
    result: Open01AcceptanceResult,
    *,
    output_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    headline = [
        row for row in result.headline_rows if row.get("status") == "estimated"
    ]
    headline_samples = {
        (
            row.get("n"),
            row.get("sample_start"),
            row.get("sample_end"),
            row.get("sample_hash"),
            row.get("control_ids"),
            row.get("control_ids_used"),
            row.get("control_ids_rejected"),
        )
        for row in headline
    }
    stability_types = {
        str(row.get("test_type", "")) for row in result.stability_rows
    }
    p_rows = [
        row
        for row in [
            *result.headline_rows,
            *result.stability_rows,
            *result.credit_screen_rows,
        ]
        if _as_float(row.get("p_value_raw")) is not None
    ]
    credit_rows = [
        row
        for row in result.credit_screen_rows
        if row.get("status") == "computed"
    ]
    expected_credit_rows = (
        len(CREDIT_SCREEN_OUTCOME_IDS)
        * len(CREDIT_WINDOW_QUARTERS)
        * len(CREDIT_ADJUSTMENTS)
    )
    observed_credit_ids = {
        str(row.get("credit_outcome_id", "")) for row in credit_rows
    }
    credit_cartesian = _credit_screen_cartesian(credit_rows)
    rolling_bounds_pass = all(
        _quarter_ordinal(str(row["last_window_end"]))
        <= _quarter_ordinal(
            str(row["last_observed_treatment_outcome_quarter"])
        )
        for row in credit_rows
    )
    tier_count_details = {
        tier: {
            "expected": expected,
            "observed": result.coverage_counts.get(
                f"{METHOD_TIER_SERIES_ID}:{tier}"
            ),
        }
        for tier, expected in EXPECTED_METHOD_TIER_COUNTS.items()
    }
    tier_counts_pass = all(
        detail["observed"] == detail["expected"]
        for detail in tier_count_details.values()
    )
    required_stability_types = {
        "fixed_sample_contract",
        "accounting_identity",
        "declared_break",
        "unknown_break_scan_summary",
        "leave_quarter_out_influence",
        "leave_block_out_influence",
        "construction_tier_sensitivity",
        "mmf_bound_sensitivity",
        "mmf_bound_sensitivity_summary",
        "overall_gate",
    }
    output_hashes_pass = set(output_records) == {
        "contract",
        "headline",
        "stability",
        "credit_screen",
    } and all(
        _valid_sha256(record.get("sha256"))
        and int(record.get("rows", 0)) > 0
        for record in output_records.values()
    )
    return {
        "producer_validation": _check(
            result.producer_status == "pass" and not result.issues,
            details={"issues": list(result.issues)},
        ),
        "contract_frozen": _check(
            bool(result.contract_rows)
            and all(
                row.get("contract_status") == "frozen"
                for row in result.contract_rows
            ),
            details={"rows": len(result.contract_rows)},
        ),
        "method_tier_counts": _check(
            tier_counts_pass,
            details=tier_count_details,
        ),
        "headline_common_sample": _check(
            len(headline) == 2 and len(headline_samples) == 1,
            details={
                "estimated_rows": len(headline),
                "distinct_sample_control_contracts": len(headline_samples),
            },
        ),
        "deposit_residual_identity": _check(
            len(headline) == 2
            and all(row.get("identity_status") == "pass" for row in headline),
            details={
                "tolerance": IDENTITY_TOLERANCE,
                "gaps": [row.get("identity_gap") for row in headline],
            },
        ),
        "stability_suite_complete": _check(
            required_stability_types.issubset(stability_types),
            details={
                "required": sorted(required_stability_types),
                "observed": sorted(stability_types),
            },
        ),
        "materiality_stability_gate": _check(
            result.scientific_status in {"stable", "review", "unstable"},
            details={
                "scientific_status": result.scientific_status,
                "bands": dict(MATERIALITY_BANDS),
            },
        ),
        "holm_adjusted_inference": _check(
            bool(p_rows)
            and all(_as_float(row.get("p_value_holm")) is not None for row in p_rows),
            details={"inference_rows": len(p_rows)},
        ),
        "credit_family_exact": _check(
            observed_credit_ids == set(CREDIT_SCREEN_OUTCOME_IDS)
            and len(CREDIT_SCREEN_OUTCOME_IDS) == 5,
            details={
                "expected": list(CREDIT_SCREEN_OUTCOME_IDS),
                "observed": sorted(observed_credit_ids),
            },
        ),
        "credit_screen_complete": _check(
            credit_cartesian["passed"],
            details=credit_cartesian,
        ),
        "credit_overlap_aware_uncertainty": _check(
            len(credit_rows) == expected_credit_rows
            and all(
                row.get("covariance_estimator") == "newey_west"
                and int(row.get("covariance_lags", 0)) > 0
                and _as_float(row.get("overlap_hac_se")) is not None
                for row in credit_rows
            ),
            details={
                "rows_with_overlap_hac": sum(
                    row.get("covariance_estimator") == "newey_west"
                    and int(row.get("covariance_lags", 0)) > 0
                    for row in credit_rows
                )
            },
        ),
        "credit_window_sign_checks": _check(
            len(credit_rows) == expected_credit_rows
            and all(
                all(
                    str(row.get(field, "")) in {"positive", "negative", "zero"}
                    for field in ("sign_40", "sign_48", "sign_60")
                )
                and isinstance(row.get("sign_stable_40_48_60"), bool)
                for row in credit_rows
            ),
            details={
                "window_lengths": list(CREDIT_WINDOW_QUARTERS),
                "adjustments": list(CREDIT_ADJUSTMENTS),
            },
        ),
        "rolling_window_bounds": _check(
            bool(credit_rows) and rolling_bounds_pass,
            details={
                "last_observed_quarter": (
                    credit_rows[0].get(
                        "last_observed_treatment_outcome_quarter"
                    )
                    if credit_rows
                    else None
                )
            },
        ),
        "retained_output_hashes": _check(
            output_hashes_pass,
            details=dict(output_records),
        ),
    }


def write_open01_outputs(
    result: Open01AcceptanceResult,
    *,
    root: Path,
    producer_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    paths = {
        "contract": root / "outputs/tables/tdc_treatment_outcome_contract.csv",
        "headline": root / "outputs/tables/tdc_same_quarter_headline.csv",
        "stability": root
        / "output/reports/tier2_pass_through_stability_gate.csv",
        "credit_screen": root
        / "output/reports/tier2_pass_through_offset_rolling_beta_correlates.csv",
    }
    _write_csv(paths["contract"], result.contract_rows)
    _write_csv(paths["headline"], result.headline_rows)
    _write_csv(paths["stability"], result.stability_rows)
    _write_csv(paths["credit_screen"], result.credit_screen_rows)
    row_counts = {
        "contract": len(result.contract_rows),
        "headline": len(result.headline_rows),
        "stability": len(result.stability_rows),
        "credit_screen": len(result.credit_screen_rows),
    }
    output_records = {
        key: {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256_file(path),
            "rows": row_counts[key],
        }
        for key, path in paths.items()
    }
    checks = _acceptance_checks(
        result,
        output_records=output_records,
    )
    status = (
        "passed"
        if checks and all(check["passed"] for check in checks.values())
        else "failed"
    )
    headline = [
        row for row in result.headline_rows if row.get("status") == "estimated"
    ]
    manifest_payload = {
        "open_id": OPEN_ID,
        "generated_at_utc": utc_now_iso(),
        "status": status,
        "producer_status": result.producer_status,
        "scientific_status": result.scientific_status,
        "acceptance_checks": checks,
        "outputs": output_records,
        "manifest_path": (
            "output/manifests/open01_acceptance_summary.json"
        ),
        "contract": {
            "treatment_id": CANONICAL_TREATMENT_ID,
            "treatment_source_series": CANONICAL_TREATMENT_SOURCE_SERIES,
            "outcome_id": CANONICAL_OUTCOME_ID,
            "residual_id": CANONICAL_RESIDUAL_ID,
            "control_ids": list(CANONICAL_CONTROL_IDS),
            "credit_screen_outcome_ids": list(CREDIT_SCREEN_OUTCOME_IDS),
            "mmf_treatment_ids": list(MMF_TREATMENT_IDS),
            "method_tier_series_id": METHOD_TIER_SERIES_ID,
            "expected_method_tier_counts": dict(EXPECTED_METHOD_TIER_COUNTS),
        },
        "units": {
            "treatment": OPEN_CONTRACT.treatment_units,
            "deposit_outcome": OPEN_CONTRACT.deposit_outcome_units,
            "credit_outcomes": OPEN_CONTRACT.credit_outcome_units,
            "estimand": "dollars_per_dollar_tdc",
        },
        "sample": {
            "start": headline[0].get("sample_start") if headline else None,
            "end": headline[0].get("sample_end") if headline else None,
            "n": headline[0].get("n") if headline else 0,
            "quarter_hash": headline[0].get("sample_hash") if headline else None,
        },
        "producer_inputs": dict(producer_inputs or {}),
        "issues": list(result.issues),
        "credit_admission": {
            outcome_id: sorted(
                {
                    str(row.get("admission_status", ""))
                    for row in result.credit_screen_rows
                    if row.get("credit_outcome_id") == outcome_id
                }
            )
            for outcome_id in CREDIT_SCREEN_OUTCOME_IDS
        },
    }
    manifest_path = root / "output/manifests/open01_acceptance_summary.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["manifest"] = manifest_path
    return paths


__all__ = [
    "BREAK_QUARTER",
    "CREDIT_ADJUSTMENTS",
    "CREDIT_WINDOW_QUARTERS",
    "IDENTITY_TOLERANCE",
    "MATERIALITY_BANDS",
    "Open01AcceptanceResult",
    "build_open01_acceptance",
    "build_treatment_outcome_contract",
    "write_open01_outputs",
]
