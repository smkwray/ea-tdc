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
from ea_tdc.estimation import _estimate_rows
from ea_tdc.open_contract import (
    CANONICAL_CONTROL_IDS,
    CANONICAL_OUTCOME_ID,
    CANONICAL_RESIDUAL_ID,
    CANONICAL_TREATMENT_LABEL,
    CREDIT_SCREEN_OUTCOME_IDS,
    OUTCOME_UNIT_MULTIPLIERS,
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
    METHOD_TIER_CONTROLS,
    TREATMENTS as BRIDGE_TREATMENTS,
    _load_manifest,
    _merge_by_quarter,
)
from run_innovation_shock_analysis import (
    add_lags,
    fit_crossfit_shock,
    impute_columns,
    select_cycle_controls,
)


HORIZONS = [0, 1, 2, 4, 8]
FOCUS_HORIZONS = {0, 4}
LEADS = [1, 2, 4]

TREATMENT_LABELS = [
    CANONICAL_TREATMENT_LABEL,
    "regression_mmf_rrp_di_long",
    "modern_canonical_di_mmf_rrp_short",
    "available_plumbing_bridge",
]

PRIMARY_TREATMENT_LABEL = CANONICAL_TREATMENT_LABEL
SELECTED_LAG_SENSITIVITY_LABEL = f"{CANONICAL_TREATMENT_LABEL}_selected_credit_rate_lags"

INNOVATION_TREATMENT_LABELS = [
    "canon_long_innovation_factor_xfit",
    "canon_long_innovation_cycle_risk_xfit",
]

INNOVATION_SOURCE_TREATMENT_LABEL = CANONICAL_TREATMENT_LABEL
INNOVATION_SHOCK_PREFIX = "tdc_tier2_main_long_history_innovation"

SELECTED_LAG_PERIODS = [1, 2, 4]
SELECTED_LAG_SOURCES = [
    "tdcpass_strict_loan_core_min_qoq",
    "tdcpass_strict_loan_mortgages_qoq",
    "tdcpass_strict_loan_consumer_credit_qoq",
    "bank_credit_qoq",
    "dgs2",
    "dgs10",
]
SELECTED_LAG_CONTROLS = [
    control_id for control_id in CANONICAL_CONTROL_IDS if "__lag_" in control_id
]

OUTCOMES = [
    CANONICAL_OUTCOME_ID,
    "domestic_nonbank_deposits_qoq",
    "other_component_tier2_canonical_di_mmf_rrp_prop_qoq",
    CANONICAL_RESIDUAL_ID,
    "other_component_tier2_regression_mmf_rrp_prop_di_np_cu_qoq",
    "other_component_tier2_mmf_rrp_plumbing_adjusted_qoq",
    *CREDIT_SCREEN_OUTCOME_IDS,
    "bank_consumer_loans_qoq",
    "bank_real_estate_loans_qoq",
    "bank_non_treasury_securities_qoq",
    "bank_treasury_agency_securities_qoq",
    "bank_treasury_securities_qoq",
    "bank_treasury_securities_transactions_qoq",
    "reserve_balances_qoq",
    "foreign_official_deposits_qoq",
    "total_reserve_balances_plus_foreign_official_qoq",
    "reserve_balances_net_fed_treasury_qoq",
    "total_reserves_plus_foreign_official_net_fed_treasury_qoq",
    "mortgage_30y",
    "mortgage_30y_dgs10_spread",
    "dgs2",
    "dgs10",
    "dgs10_2y_spread",
    "dgs10_3mo_spread",
    "baa_aaa",
    "BAMLC0A0CM",
    "BAMLH0A0HYM2",
    "repo_spread",
]

PATTERN_OUTCOMES = [
    "matched_total_deposits",
    "tdcpass_strict_loan_core_min_qoq",
    "tdcpass_strict_loan_consumer_credit_qoq",
    "tdcpass_strict_loan_mortgages_qoq",
    "bank_credit_qoq",
    "bank_real_estate_loans_qoq",
    "bank_consumer_loans_qoq",
    "bank_non_treasury_securities_qoq",
    "bank_treasury_agency_securities_qoq",
    "bank_treasury_securities_qoq",
    "bank_treasury_securities_transactions_qoq",
    "reserve_balances_qoq",
    "foreign_official_deposits_qoq",
    "total_reserve_balances_plus_foreign_official_qoq",
    "reserve_balances_net_fed_treasury_qoq",
    "total_reserves_plus_foreign_official_net_fed_treasury_qoq",
    "mortgage_30y_dgs10_spread",
    "dgs2",
    "dgs10",
]

SAMPLE_WINDOWS = [
    {
        "sample_label": "full_available",
        "description": "All available quarters after treatment/control/outcome availability filters.",
    },
    {
        "sample_label": "exclude_gfc",
        "exclude_windows": [("2007Q4", "2009Q2")],
        "description": "Excludes GFC/NBER recession quarters, 2007Q4-2009Q2.",
    },
    {
        "sample_label": "exclude_covid",
        "exclude_windows": [("2020Q1", "2020Q2")],
        "description": "Excludes acute COVID quarters, 2020Q1-2020Q2.",
    },
    {
        "sample_label": "exclude_transition_2019_2021",
        "exclude_windows": [("2019Q1", "2021Q4")],
        "description": "Excludes the 2019Q1-2021Q4 stress/transition interval.",
    },
    {
        "sample_label": "exclude_gfc_covid",
        "exclude_windows": [("2007Q4", "2009Q2"), ("2020Q1", "2020Q2")],
        "description": "Excludes GFC and acute COVID quarters.",
    },
    {
        "sample_label": "exclude_gfc_covid_transition",
        "exclude_windows": [("2007Q4", "2009Q2"), ("2020Q1", "2020Q2"), ("2019Q1", "2021Q4")],
        "description": "Excludes GFC, acute COVID, and 2019Q1-2021Q4 transition/stress quarters.",
    },
]

EXISTING_STATE_CANDIDATES = [
    ("coord_low_reserve_state_l1", "Low-reserve state", "existing"),
    ("coord_on_rrp_drain_state_l1", "ON RRP drain state", "existing"),
    ("slrwatch_bank_leverage_pressure_l1", "Bank leverage-pressure state", "existing"),
    ("tsyparty_bank_foreign_private_corr_l1", "Bank/foreign-private correlation state", "existing"),
]

GENERATED_STRESS_SOURCES = [
    ("baa_aaa", "high_baa_aaa_state_l1", "High BAA-AAA spread state, lagged"),
    ("BAMLH0A0HYM2", "high_hy_oas_state_l1", "High high-yield OAS state, lagged"),
    ("mortgage_30y_dgs10_spread", "high_mortgage_spread_state_l1", "High mortgage-Treasury spread state, lagged"),
]

GENERATED_WINDOW_STATES = [
    ("gfc_state", "GFC recession window, 2007Q4-2009Q2", "2007Q4", "2009Q2"),
    ("covid_state", "Acute COVID window, 2020Q1-2020Q2", "2020Q1", "2020Q2"),
    ("transition_2019_2021_state", "2019Q1-2021Q4 transition/stress window", "2019Q1", "2021Q4"),
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    return float(text)


def _normal_p(z_score: float) -> float:
    return math.erfc(abs(z_score) / math.sqrt(2.0))


def _quarter_key(quarter: str) -> tuple[int, int]:
    text = str(quarter).strip()
    return int(text[:4]), int(text[-1])


def _in_quarter_range(quarter: str, start: str, end: str) -> bool:
    key = _quarter_key(quarter)
    return _quarter_key(start) <= key <= _quarter_key(end)


def _excluded(quarter: str, windows: list[tuple[str, str]]) -> bool:
    return any(_in_quarter_range(quarter, start, end) for start, end in windows)


def _rows_for_window(rows: list[dict[str, str]], sample: dict[str, Any]) -> list[dict[str, str]]:
    windows = [(str(start), str(end)) for start, end in sample.get("exclude_windows", [])]
    if not windows:
        return [dict(row) for row in rows]
    return [dict(row) for row in rows if not _excluded(str(row.get("quarter", "")), windows)]


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("No values supplied")
    position = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _available_columns(rows: list[dict[str, str]]) -> set[str]:
    columns: set[str] = set()
    for row in rows:
        columns.update(row)
    return columns


def _available_outcomes(rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    columns = _available_columns(rows)
    available = [outcome for outcome in OUTCOMES if outcome in columns]
    missing = [outcome for outcome in OUTCOMES if outcome not in columns]
    return available, missing


def _active_controls(control_ids: list[str], treatment_spec: dict[str, Any]) -> list[str]:
    if not treatment_spec.get("use_method_tier_controls"):
        return control_ids[:]
    return [*control_ids[:4], *METHOD_TIER_CONTROLS, *control_ids[4:]]


def _add_selected_lags(rows: list[dict[str, str]]) -> list[str]:
    lagged_columns: list[str] = []
    available_columns = _available_columns(rows)
    for column in SELECTED_LAG_SOURCES:
        if column not in available_columns:
            continue
        for lag in SELECTED_LAG_PERIODS:
            lagged_column = f"{column}__lag_{lag}"
            lagged_columns.append(lagged_column)
            for idx, row in enumerate(rows):
                source_idx = idx - lag
                row[lagged_column] = rows[source_idx].get(column, "") if source_idx >= 0 else ""
    return lagged_columns


def _insert_controls_before_factor_tail(base_controls: list[str], inserted_controls: list[str]) -> list[str]:
    prefix: list[str] = []
    factor_tail: list[str] = []
    for control_id in base_controls:
        if control_id.startswith("dflmx_") or control_id.startswith("imp_dflmx_"):
            factor_tail.append(control_id)
        else:
            prefix.append(control_id)
    selected_insertions = [control_id for control_id in inserted_controls if control_id not in prefix]
    return [*prefix, *selected_insertions, *factor_tail]


def _augment_states(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    augmented = [dict(row) for row in rows]
    columns = _available_columns(augmented)
    state_rows: list[dict[str, str]] = []

    for state_id, label, source in EXISTING_STATE_CANDIDATES:
        if state_id in columns:
            state_rows.append({"state_id": state_id, "state_label": label, "state_source": source, "available": "true"})

    for state_id, label, start, end in GENERATED_WINDOW_STATES:
        for row in augmented:
            quarter = str(row.get("quarter", "")).strip()
            row[state_id] = "1" if quarter and _in_quarter_range(quarter, start, end) else "0"
        state_rows.append(
            {
                "state_id": state_id,
                "state_label": label,
                "state_source": "generated_window",
                "available": "true",
            }
        )

    for source_id, state_id, label in GENERATED_STRESS_SOURCES:
        if source_id not in columns:
            state_rows.append(
                {
                    "state_id": state_id,
                    "state_label": label,
                    "state_source": f"generated_lagged_q75:{source_id}",
                    "available": "false",
                }
            )
            continue
        values = [_float(row.get(source_id, "")) for row in augmented]
        numeric_values = [value for value in values if value is not None]
        if len(numeric_values) < 8:
            state_rows.append(
                {
                    "state_id": state_id,
                    "state_label": label,
                    "state_source": f"generated_lagged_q75:{source_id}",
                    "available": "false",
                }
            )
            continue
        threshold = _quantile(numeric_values, 0.75)
        previous_value: float | None = None
        for row in augmented:
            current_value = _float(row.get(source_id, ""))
            row[state_id] = "" if previous_value is None else ("1" if previous_value >= threshold else "0")
            previous_value = current_value
        state_rows.append(
            {
                "state_id": state_id,
                "state_label": label,
                "state_source": f"generated_lagged_q75:{source_id}",
                "available": "true",
                "threshold": threshold,
            }
        )

    active_state_ids = [row["state_id"] for row in state_rows if row["available"] == "true"]
    return augmented, state_rows, active_state_ids


def _normalization(outcome_id: str) -> tuple[str, float]:
    canonical_multiplier = OUTCOME_UNIT_MULTIPLIERS.get(outcome_id)
    if canonical_multiplier is not None:
        return "dollars_per_dollar_tdc", canonical_multiplier
    dollar_per_dollar_outcomes = {
        "domestic_nonbank_deposits_qoq",
        "other_component_tier2_canonical_di_mmf_rrp_prop_qoq",
        "other_component_tier2_regression_mmf_rrp_prop_di_np_cu_qoq",
        "other_component_tier2_mmf_rrp_plumbing_adjusted_qoq",
        "reserve_balances_qoq",
        "foreign_official_deposits_qoq",
        "total_reserve_balances_plus_foreign_official_qoq",
        "reserve_balances_net_fed_treasury_qoq",
        "total_reserves_plus_foreign_official_net_fed_treasury_qoq",
        "bank_treasury_securities_qoq",
        "bank_treasury_securities_transactions_qoq",
    }
    quantity_outcomes = {
        "bank_consumer_loans_qoq",
        "bank_real_estate_loans_qoq",
        "bank_non_treasury_securities_qoq",
        "bank_treasury_agency_securities_qoq",
    }
    if outcome_id in dollar_per_dollar_outcomes:
        return "dollars_per_dollar_tdc", 1.0
    if outcome_id in quantity_outcomes:
        return "dollars_per_dollar_tdc", 1000.0
    return "raw_outcome_units_per_dollar_tdc", 1.0


def _effect_per_100b(outcome_id: str, beta: float) -> tuple[str, float]:
    canonical_multiplier = OUTCOME_UNIT_MULTIPLIERS.get(outcome_id)
    if canonical_multiplier is not None:
        return "usd_billions_per_100b_tdc", beta * canonical_multiplier * 100.0
    dollar_per_dollar_outcomes = {
        "domestic_nonbank_deposits_qoq",
        "other_component_tier2_canonical_di_mmf_rrp_prop_qoq",
        "other_component_tier2_regression_mmf_rrp_prop_di_np_cu_qoq",
        "other_component_tier2_mmf_rrp_plumbing_adjusted_qoq",
        "reserve_balances_qoq",
        "foreign_official_deposits_qoq",
        "total_reserve_balances_plus_foreign_official_qoq",
        "reserve_balances_net_fed_treasury_qoq",
        "total_reserves_plus_foreign_official_net_fed_treasury_qoq",
        "bank_treasury_securities_qoq",
        "bank_treasury_securities_transactions_qoq",
    }
    quantity_outcomes = {
        "bank_consumer_loans_qoq",
        "bank_real_estate_loans_qoq",
        "bank_non_treasury_securities_qoq",
        "bank_treasury_agency_securities_qoq",
    }
    rate_outcomes = {
        "mortgage_30y",
        "mortgage_30y_dgs10_spread",
        "dgs2",
        "dgs10",
        "dgs10_2y_spread",
        "dgs10_3mo_spread",
        "baa_aaa",
        "BAMLC0A0CM",
        "BAMLH0A0HYM2",
        "repo_spread",
    }
    if outcome_id in dollar_per_dollar_outcomes:
        return "usd_billions_per_100b_tdc", beta * 100.0
    if outcome_id in quantity_outcomes:
        return "usd_billions_per_100b_tdc", beta * 100000.0
    if outcome_id in rate_outcomes:
        return "basis_points_per_100b_tdc", beta * 100000.0 * 100.0
    return "outcome_units_per_100b_tdc", beta * 100000.0


def _decorate_estimate(row: dict[str, Any], *, surface: str, treatment_label: str, sample_label: str = "", state_id: str = "") -> dict[str, Any]:
    outcome = str(row.get("outcome", ""))
    beta = float(row.get("beta", 0.0))
    unit, multiplier = _normalization(outcome)
    effect_unit, effect_value = _effect_per_100b(outcome, beta)
    decorated = dict(row)
    decorated["surface"] = surface
    decorated["treatment_label"] = treatment_label
    decorated["sample_label"] = sample_label
    if state_id:
        decorated["state_id"] = state_id
    decorated["normalized_unit"] = unit
    decorated["normalized_beta"] = beta * multiplier
    decorated["effect_per_100b_unit"] = effect_unit
    decorated["effect_per_100b_tdc"] = effect_value
    decorated["pinned_anchor_job_id"] = ANCHOR_JOB_ID
    decorated["pinned_k_screened"] = K_SCREENED
    decorated["pinned_control_policy_mode"] = CONTROL_POLICY_MODE
    return decorated


def _estimate_surface(
    *,
    rows: list[dict[str, str]],
    treatment_label: str,
    treatment_spec: dict[str, Any],
    control_ids: list[str],
    outcome_ids: list[str],
    horizons: list[int],
    surface: str,
    sample_label: str = "",
    state_id: str = "",
) -> list[dict[str, Any]]:
    estimates = _estimate_rows(
        estimator="lp",
        bundle_rows=rows,
        treatment_id=str(treatment_spec["treatment_id"]),
        control_ids=_active_controls(control_ids, treatment_spec),
        outcome_ids=outcome_ids,
        horizons=horizons,
        response_type="direct_at_h",
        job_id=f"tier2_credit_causality_{surface}_{treatment_label}_{sample_label or state_id or 'full'}",
        instrument_ids=[],
        state_id=state_id,
    )
    return [
        _decorate_estimate(
            row,
            surface=surface,
            treatment_label=treatment_label,
            sample_label=sample_label,
            state_id=state_id,
        )
        for row in estimates
    ]


def _build_innovation_specs(
    rows: list[dict[str, str]],
    *,
    design_manifest: dict[str, Any],
    control_ids: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], list[dict[str, Any]]]:
    source_spec = BRIDGE_TREATMENTS[INNOVATION_SOURCE_TREATMENT_LABEL]
    source_treatment_id = str(source_spec["treatment_id"])
    source_residual_id = str(source_spec["residual_id"])
    base_controls = [str(item) for item in design_manifest.get("control_ids", [])]
    factor_ids = [control_id for control_id in control_ids if control_id not in base_controls]
    method_controls = [
        control_id
        for control_id in METHOD_TIER_CONTROLS
        if any(str(row.get(control_id, "")).strip() for row in rows)
    ]
    lag_source_cols = [source_treatment_id, *base_controls, *method_controls, *factor_ids]
    lagged_cols = add_lags(rows, lag_source_cols)
    imputed_lagged_cols = impute_columns(rows, lagged_cols)
    cycle_cols = select_cycle_controls(rows)
    imputed_cycle_cols = impute_columns(rows, cycle_cols)

    lagged_treatment = f"imp_{source_treatment_id}_lag1"
    lagged_core = [f"imp_{control}_lag1" for control in [*base_controls, *method_controls]]
    lagged_factors = [f"imp_{factor_id}_lag1" for factor_id in factor_ids]
    factor_predictors = [lagged_treatment, *lagged_core, *lagged_factors]
    cycle_predictors = [*factor_predictors, *imputed_cycle_cols]

    specs = {
        "canon_long_innovation_factor_xfit": {
            "treatment_id": f"{INNOVATION_SHOCK_PREFIX}_factor_xfit",
            "first_stage_predictors": factor_predictors,
            "lp_controls": factor_predictors,
            "description": "5-fold forecast residual using lagged long-history canonical TDC, method-tier controls, base controls, and K=100 factors.",
        },
        "canon_long_innovation_cycle_risk_xfit": {
            "treatment_id": f"{INNOVATION_SHOCK_PREFIX}_cycle_risk_xfit",
            "first_stage_predictors": cycle_predictors,
            "lp_controls": cycle_predictors,
            "description": "5-fold forecast residual using lagged long-history canonical TDC, method-tier/base/factor controls, and lagged cycle/risk variables.",
        },
    }
    source_quarters = [
        str(row.get("quarter", ""))
        for row in rows
        if str(row.get(source_treatment_id, "")).strip()
    ]
    diagnostics: list[dict[str, Any]] = []
    for label, spec in specs.items():
        diagnostic = fit_crossfit_shock(
            rows,
            treatment_id=source_treatment_id,
            predictors=list(spec["first_stage_predictors"]),
            shock_id=str(spec["treatment_id"]),
        )
        diagnostic["treatment_label"] = label
        diagnostic["source_treatment_label"] = INNOVATION_SOURCE_TREATMENT_LABEL
        diagnostic["source_treatment_id"] = source_treatment_id
        diagnostic["source_quarter_start"] = source_quarters[0] if source_quarters else ""
        diagnostic["source_quarter_end"] = source_quarters[-1] if source_quarters else ""
        diagnostic["description"] = spec["description"]
        diagnostics.append(diagnostic)
    treatment_specs = {
        label: {
            "treatment_id": str(spec["treatment_id"]),
            "use_method_tier_controls": False,
            "residual_id": source_residual_id,
        }
        for label, spec in specs.items()
    }
    control_map = {label: list(spec["lp_controls"]) for label, spec in specs.items()}
    return treatment_specs, control_map, diagnostics


def _add_future_treatment(rows: list[dict[str, str]], treatment_id: str, lead: int) -> tuple[list[dict[str, str]], str]:
    column = f"{treatment_id}__lead_{lead}"
    augmented = [dict(row) for row in rows]
    for idx, row in enumerate(augmented):
        future_idx = idx + lead
        row[column] = rows[future_idx].get(treatment_id, "") if future_idx < len(rows) else ""
    return augmented, column


def _lead_placebo_estimates(
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
        lead_spec = dict(treatment_spec)
        lead_spec["treatment_id"] = lead_column
        estimates = _estimate_surface(
            rows=lead_rows,
            treatment_label=treatment_label,
            treatment_spec=lead_spec,
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


def _summarize_sample_splits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    focus = [
        "matched_total_deposits",
        "tdcpass_strict_loan_core_min_qoq",
        "tdcpass_strict_loan_consumer_credit_qoq",
        "tdcpass_strict_loan_mortgages_qoq",
        "bank_credit_qoq",
        "bank_non_treasury_securities_qoq",
        "bank_treasury_securities_qoq",
        "bank_treasury_securities_transactions_qoq",
        "bank_treasury_agency_securities_qoq",
        "reserve_balances_qoq",
        "total_reserve_balances_plus_foreign_official_qoq",
        "reserve_balances_net_fed_treasury_qoq",
        "total_reserves_plus_foreign_official_net_fed_treasury_qoq",
    ]
    return [
        {
            "treatment_label": row["treatment_label"],
            "sample_label": row["sample_label"],
            "outcome": row["outcome"],
            "horizon": row["horizon"],
            "beta": row["beta"],
            "normalized_beta": row["normalized_beta"],
            "normalized_unit": row["normalized_unit"],
            "effect_per_100b_tdc": row["effect_per_100b_tdc"],
            "effect_per_100b_unit": row["effect_per_100b_unit"],
            "p_value": row.get("p_value_normal", ""),
            "n": row.get("n", ""),
        }
        for row in rows
        if row.get("surface") in {"pooled_baseline", "sample_split"}
        and str(row.get("outcome")) in focus
        and int(row.get("horizon", 0)) in FOCUS_HORIZONS
    ]


def _summarize_state_effects(rows: list[dict[str, Any]], state_metadata: list[dict[str, str]]) -> list[dict[str, Any]]:
    labels = {row["state_id"]: row["state_label"] for row in state_metadata}
    state_rows = [
        row
        for row in rows
        if row.get("surface") == "state_dependent"
        and int(row.get("horizon", 0)) in FOCUS_HORIZONS
        and str(row.get("outcome")) in {
            "matched_total_deposits",
            "tdcpass_strict_loan_core_min_qoq",
            "tdcpass_strict_loan_consumer_credit_qoq",
            "tdcpass_strict_loan_mortgages_qoq",
            "bank_credit_qoq",
            "bank_non_treasury_securities_qoq",
            "bank_treasury_securities_qoq",
            "bank_treasury_securities_transactions_qoq",
            "bank_treasury_agency_securities_qoq",
            "reserve_balances_qoq",
            "total_reserve_balances_plus_foreign_official_qoq",
            "reserve_balances_net_fed_treasury_qoq",
            "total_reserves_plus_foreign_official_net_fed_treasury_qoq",
        }
    ]
    output: list[dict[str, Any]] = []
    for row in state_rows:
        interaction_beta = _float(row.get("state_interaction_beta", ""))
        interaction_se = _float(row.get("state_interaction_se", ""))
        interaction_p = ""
        if interaction_beta is not None and interaction_se and interaction_se > 0:
            interaction_p = _normal_p(interaction_beta / interaction_se)
        output.append(
            {
                "treatment_label": row["treatment_label"],
                "state_id": row.get("state_id", ""),
                "state_label": labels.get(str(row.get("state_id", "")), ""),
                "state_profile": row.get("state_profile", ""),
                "state_reference_value": row.get("state_reference_value", ""),
                "outcome": row["outcome"],
                "horizon": row["horizon"],
                "profile_beta": row["beta"],
                "profile_normalized_beta": row["normalized_beta"],
                "profile_effect_per_100b_tdc": row["effect_per_100b_tdc"],
                "profile_effect_per_100b_unit": row["effect_per_100b_unit"],
                "profile_p_value": row.get("p_value_normal", ""),
                "interaction_beta": row.get("state_interaction_beta", ""),
                "interaction_se": row.get("state_interaction_se", ""),
                "interaction_p_value": interaction_p,
                "n": row.get("n", ""),
            }
        )
    return output


def _summarize_leads(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "treatment_label": row["treatment_label"],
            "lead_quarters": row.get("lead_quarters", ""),
            "outcome": row["outcome"],
            "beta": row["beta"],
            "normalized_beta": row["normalized_beta"],
            "normalized_unit": row["normalized_unit"],
            "effect_per_100b_tdc": row["effect_per_100b_tdc"],
            "effect_per_100b_unit": row["effect_per_100b_unit"],
            "p_value": row.get("p_value_normal", ""),
            "n": row.get("n", ""),
        }
        for row in rows
        if row.get("surface") == "lead_placebo"
        and str(row.get("outcome")) in {
            "tdcpass_strict_loan_core_min_qoq",
            "tdcpass_strict_loan_consumer_credit_qoq",
            "tdcpass_strict_loan_mortgages_qoq",
            "bank_credit_qoq",
        }
    ]


def _pattern_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if row.get("surface") not in {"pooled_baseline", "sample_split"}:
            continue
        if row.get("sample_label") not in {"", "full_available", "exclude_gfc_covid_transition"}:
            continue
        if int(row.get("horizon", 0)) not in FOCUS_HORIZONS:
            continue
        if str(row.get("outcome")) not in PATTERN_OUTCOMES:
            continue
        output.append(
            {
                "treatment_label": row["treatment_label"],
                "sample_label": row.get("sample_label") or "full_available",
                "outcome": row["outcome"],
                "horizon": row["horizon"],
                "effect_per_100b_tdc": row["effect_per_100b_tdc"],
                "effect_per_100b_unit": row["effect_per_100b_unit"],
                "p_value": row.get("p_value_normal", ""),
                "n": row.get("n", ""),
            }
        )
    return output


def _key_rows(rows: list[dict[str, Any]], *, treatment_label: str, sample_label: str, horizon: int) -> dict[str, dict[str, Any]]:
    keyed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("treatment_label") != treatment_label:
            continue
        row_sample = row.get("sample_label") or "full_available"
        if row_sample != sample_label:
            continue
        if int(row.get("horizon", 0)) != horizon:
            continue
        keyed[str(row.get("outcome", ""))] = row
    return keyed


def _readout_rows(sample_rows: list[dict[str, Any]], state_rows: list[dict[str, Any]], lead_rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = PRIMARY_TREATMENT_LABEL
    full_h0 = _key_rows(sample_rows, treatment_label=primary, sample_label="full_available", horizon=0)
    normal_h0 = _key_rows(sample_rows, treatment_label=primary, sample_label="exclude_gfc_covid_transition", horizon=0)
    core = "tdcpass_strict_loan_core_min_qoq"
    consumer = "tdcpass_strict_loan_consumer_credit_qoq"
    mortgage = "tdcpass_strict_loan_mortgages_qoq"

    full_credit = sum(
        abs(float(full_h0[outcome]["normalized_beta"]))
        for outcome in (consumer, mortgage)
        if outcome in full_h0 and str(full_h0[outcome]["normalized_beta"]).strip()
    )
    normal_credit = sum(
        abs(float(normal_h0[outcome]["normalized_beta"]))
        for outcome in (consumer, mortgage)
        if outcome in normal_h0 and str(normal_h0[outcome]["normalized_beta"]).strip()
    )
    full_deposits = float(full_h0["matched_total_deposits"]["normalized_beta"]) if "matched_total_deposits" in full_h0 else None
    normal_deposits = float(normal_h0["matched_total_deposits"]["normalized_beta"]) if "matched_total_deposits" in normal_h0 else None

    significant_leads = [
        row
        for row in lead_rows
        if row["treatment_label"] == primary
        and row["outcome"] in {core, consumer, mortgage}
        and str(row.get("p_value", "")).strip()
        and float(row["p_value"]) < 0.1
    ]
    significant_interactions = [
        row
        for row in state_rows
        if row["treatment_label"] == primary
        and row["outcome"] in {core, consumer, mortgage}
        and str(row.get("interaction_p_value", "")).strip()
        and float(row["interaction_p_value"]) < 0.1
    ]
    return {
        "primary_treatment": primary,
        "full_h0_deposits": full_deposits,
        "normal_h0_deposits": normal_deposits,
        "full_h0_consumer_plus_mortgage_abs": full_credit,
        "normal_h0_consumer_plus_mortgage_abs": normal_credit,
        "significant_credit_leads_count": len(significant_leads),
        "significant_credit_state_interactions_count": len(significant_interactions),
        "significant_credit_leads": significant_leads[:10],
        "significant_credit_state_interactions": significant_interactions[:10],
    }


def _format_value(value: Any) -> str:
    if value is None or value == "":
        return ""
    numeric = float(value)
    if abs(numeric) >= 10:
        return f"{numeric:.2f}"
    return f"{numeric:.3f}"


def _write_markdown(
    path: Path,
    *,
    summary: dict[str, Any],
    sample_rows: list[dict[str, Any]],
    state_rows: list[dict[str, Any]],
    lead_rows: list[dict[str, Any]],
    missing_outcomes: list[str],
    state_metadata: list[dict[str, str]],
    innovation_diagnostics: list[dict[str, Any]],
) -> None:
    read = "mixed_or_unresolved"
    full_credit = summary.get("full_h0_consumer_plus_mortgage_abs") or 0.0
    normal_credit = summary.get("normal_h0_consumer_plus_mortgage_abs") or 0.0
    if full_credit and normal_credit < 0.5 * full_credit:
        read = "downturn_policy_response_confounding"
    elif summary.get("significant_credit_leads_count", 0):
        read = "reverse_causality_or_common_downturn_risk"
    elif normal_credit >= 0.75 * full_credit and not summary.get("significant_credit_leads_count", 0):
        read = "crowding_out_plausible_but_not_proven"
    if summary.get("significant_credit_state_interactions_count", 0):
        read = f"{read}_with_state_dependence"

    lines = [
        "# Tier 2 Credit Causality Readout",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        f"Primary read: `{read}`.",
        "",
        "The goal is to separate generic credit crowding out from downturn/policy-response and rate-regime confounding. Values for private-credit quantities are normalized to dollars per $1 TDC, and pattern-table effects are also reported per +$100B TDC.",
        "",
        "## Primary H=0 Check",
        "",
        f"- Full-sample deposits: `{_format_value(summary.get('full_h0_deposits'))}` per $1 TDC.",
        f"- Full-sample consumer + mortgage absolute credit offset: `{_format_value(summary.get('full_h0_consumer_plus_mortgage_abs'))}` per $1 TDC.",
        f"- Normal-window deposits after excluding GFC, acute COVID, and 2019Q1-2021Q4: `{_format_value(summary.get('normal_h0_deposits'))}` per $1 TDC.",
        f"- Normal-window consumer + mortgage absolute credit offset: `{_format_value(summary.get('normal_h0_consumer_plus_mortgage_abs'))}` per $1 TDC.",
        f"- Significant credit lead placebos at p<0.10: `{summary.get('significant_credit_leads_count', 0)}`.",
        f"- Significant credit state interactions at p<0.10: `{summary.get('significant_credit_state_interactions_count', 0)}`.",
        "",
        "## Sample Split Rows",
        "",
        "| sample | outcome | h | normalized beta | p | n |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in sample_rows:
        if row["treatment_label"] != PRIMARY_TREATMENT_LABEL:
            continue
        if row["outcome"] not in {"matched_total_deposits", "tdcpass_strict_loan_consumer_credit_qoq", "tdcpass_strict_loan_mortgages_qoq", "tdcpass_strict_loan_core_min_qoq"}:
            continue
        if int(row["horizon"]) != 0:
            continue
        lines.append(
            f"| {row['sample_label']} | {row['outcome']} | {row['horizon']} | {_format_value(row['normalized_beta'])} | {_format_value(row.get('p_value'))} | {row.get('n', '')} |"
        )

    lines.extend(
        [
            "",
            "## Selected Credit/Rate Lag Sensitivity",
            "",
            "This sensitivity keeps the primary long-history TDC treatment but adds selected lagged credit and Treasury-rate controls before the factor tail.",
            "",
            "| run | outcome | h0 normalized beta | p | n | significant credit leads |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label in [PRIMARY_TREATMENT_LABEL, SELECTED_LAG_SENSITIVITY_LABEL]:
        lead_count = sum(
            1
            for row in lead_rows
            if row["treatment_label"] == label
            and row["outcome"] in {
                "tdcpass_strict_loan_core_min_qoq",
                "tdcpass_strict_loan_consumer_credit_qoq",
                "tdcpass_strict_loan_mortgages_qoq",
            }
            and str(row.get("p_value", "")).strip()
            and float(row["p_value"]) < 0.1
        )
        for outcome in [
            "matched_total_deposits",
            "tdcpass_strict_loan_core_min_qoq",
            "tdcpass_strict_loan_mortgages_qoq",
            "tdcpass_strict_loan_consumer_credit_qoq",
            "bank_credit_qoq",
        ]:
            matches = [
                row
                for row in sample_rows
                if row["treatment_label"] == label
                and row["sample_label"] == "full_available"
                and row["outcome"] == outcome
                and int(row["horizon"]) == 0
            ]
            if not matches:
                continue
            row = matches[0]
            lines.append(
                f"| {label} | {outcome} | {_format_value(row['normalized_beta'])} | {_format_value(row.get('p_value'))} | {row.get('n', '')} | {lead_count} |"
            )

    lines.extend(
        [
            "",
            "## Innovation-Shock Comparison",
            "",
            "Innovation shocks are built from the long-history canonical regression/MMF/RRP Tier 2 row. The 2022+ direct canonical row remains a short modern-overlap sensitivity, not the primary innovation source.",
            "",
        ]
    )
    if innovation_diagnostics:
        lines.extend(["| run | source range | n | first-stage R2 | shock sd |", "| --- | --- | ---: | ---: | ---: |"])
        for row in innovation_diagnostics:
            lines.append(
                f"| {row.get('treatment_label', '')} | {row.get('source_quarter_start', '')}-{row.get('source_quarter_end', '')} | {row.get('n', '')} | {_format_value(row.get('crossfit_r2'))} | {_format_value(row.get('shock_std'))} |"
            )
        lines.append("")
    lines.extend(["| run | outcome | h0 normalized beta | p | n | significant credit leads |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for label in INNOVATION_TREATMENT_LABELS:
        lead_count = sum(
            1
            for row in lead_rows
            if row["treatment_label"] == label
            and row["outcome"] in {
                "tdcpass_strict_loan_core_min_qoq",
                "tdcpass_strict_loan_consumer_credit_qoq",
                "tdcpass_strict_loan_mortgages_qoq",
            }
            and str(row.get("p_value", "")).strip()
            and float(row["p_value"]) < 0.1
        )
        for outcome in [
            "matched_total_deposits",
            "tdcpass_strict_loan_core_min_qoq",
            "tdcpass_strict_loan_mortgages_qoq",
            "tdcpass_strict_loan_consumer_credit_qoq",
        ]:
            matches = [
                row
                for row in sample_rows
                if row["treatment_label"] == label
                and row["sample_label"] == "full_available"
                and row["outcome"] == outcome
                and int(row["horizon"]) == 0
            ]
            if not matches:
                continue
            row = matches[0]
            lines.append(
                f"| {label} | {outcome} | {_format_value(row['normalized_beta'])} | {_format_value(row.get('p_value'))} | {row.get('n', '')} | {lead_count} |"
            )

    lines.extend(["", "## State Interactions", "", "| state | profile | outcome | h | profile beta | interaction p | n |", "| --- | --- | --- | --- | ---: | ---: | ---: |"])
    shown = 0
    for row in state_rows:
        if row["treatment_label"] != PRIMARY_TREATMENT_LABEL:
            continue
        if row["outcome"] not in {"tdcpass_strict_loan_consumer_credit_qoq", "tdcpass_strict_loan_mortgages_qoq", "tdcpass_strict_loan_core_min_qoq", "matched_total_deposits"}:
            continue
        if int(row["horizon"]) != 0:
            continue
        if shown >= 24:
            break
        lines.append(
            f"| {row['state_id']} | {row['state_profile']} | {row['outcome']} | {row['horizon']} | {_format_value(row['profile_normalized_beta'])} | {_format_value(row.get('interaction_p_value'))} | {row.get('n', '')} |"
        )
        shown += 1

    lines.extend(["", "## Lead Placebos", "", "| lead quarters | outcome | normalized beta | p | n |", "| ---: | --- | ---: | ---: | ---: |"])
    for row in lead_rows:
        if row["treatment_label"] != PRIMARY_TREATMENT_LABEL:
            continue
        if row["outcome"] not in {"tdcpass_strict_loan_consumer_credit_qoq", "tdcpass_strict_loan_mortgages_qoq", "tdcpass_strict_loan_core_min_qoq"}:
            continue
        lines.append(
            f"| {row['lead_quarters']} | {row['outcome']} | {_format_value(row['normalized_beta'])} | {_format_value(row.get('p_value'))} | {row.get('n', '')} |"
        )

    lines.extend(["", "## Available States", "", "| state | source | available |", "| --- | --- | --- |"])
    for row in state_metadata:
        lines.append(f"| {row['state_id']} | {row.get('state_source', '')} | {row.get('available', '')} |")
    if missing_outcomes:
        lines.extend(["", "## Missing Outcomes", "", ", ".join(f"`{outcome}`" for outcome in missing_outcomes)])
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "- If credit effects disappear in normal windows or lead placebos are active, describe the result as downturn/policy-response or reverse-causality risk rather than crowding out.",
            "- If normal-window private-credit declines persist, leads are quiet, and securities/reserve patterns line up, crowding out becomes plausible but still not proven without a stronger shock or IV branch.",
            "- If state interactions dominate, keep the credit claim state-contingent and avoid a generic private-credit statement.",
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
    merged_rows = _merge_by_quarter(factor_rows, bundle_paths)
    available_selected_lag_columns = _add_selected_lags(merged_rows)
    selected_lag_controls = [
        control_id for control_id in SELECTED_LAG_CONTROLS if control_id in available_selected_lag_columns
    ]
    primary_spec = BRIDGE_TREATMENTS[PRIMARY_TREATMENT_LABEL]
    selected_lag_treatment_spec = dict(primary_spec)
    selected_lag_treatment_spec["use_method_tier_controls"] = False
    selected_lag_active_controls = _insert_controls_before_factor_tail(
        _active_controls(control_ids, primary_spec),
        selected_lag_controls,
    )
    innovation_specs, innovation_control_map, innovation_diagnostics = _build_innovation_specs(
        merged_rows,
        design_manifest=anchor_manifest,
        control_ids=control_ids,
    )
    merged_rows, state_metadata, active_state_ids = _augment_states(merged_rows)
    outcome_ids, missing_outcomes = _available_outcomes(merged_rows)

    all_estimates: list[dict[str, Any]] = []
    treatment_specs = {label: BRIDGE_TREATMENTS[label] for label in TREATMENT_LABELS}
    treatment_specs.update(innovation_specs)
    for treatment_label, treatment_spec in treatment_specs.items():
        active_control_ids = innovation_control_map.get(treatment_label, control_ids)
        all_estimates.extend(
            _estimate_surface(
                rows=merged_rows,
                treatment_label=treatment_label,
                treatment_spec=treatment_spec,
                control_ids=active_control_ids,
                outcome_ids=outcome_ids,
                horizons=HORIZONS,
                surface="pooled_baseline",
                sample_label="full_available",
            )
        )
        for sample in SAMPLE_WINDOWS[1:]:
            sample_rows = _rows_for_window(merged_rows, sample)
            all_estimates.extend(
                _estimate_surface(
                    rows=sample_rows,
                    treatment_label=treatment_label,
                    treatment_spec=treatment_spec,
                    control_ids=active_control_ids,
                    outcome_ids=outcome_ids,
                    horizons=HORIZONS,
                    surface="sample_split",
                    sample_label=str(sample["sample_label"]),
                )
            )
        for state_id in active_state_ids:
            all_estimates.extend(
                _estimate_surface(
                    rows=merged_rows,
                    treatment_label=treatment_label,
                    treatment_spec=treatment_spec,
                    control_ids=active_control_ids,
                    outcome_ids=outcome_ids,
                    horizons=HORIZONS,
                    surface="state_dependent",
                    state_id=state_id,
                )
            )
        all_estimates.extend(
            _lead_placebo_estimates(
                rows=merged_rows,
                treatment_label=treatment_label,
                treatment_spec=treatment_spec,
                control_ids=active_control_ids,
                outcome_ids=[
                    outcome
                    for outcome in outcome_ids
                    if outcome
                    in {
                        "tdcpass_strict_loan_core_min_qoq",
                        "tdcpass_strict_loan_mortgages_qoq",
                        "tdcpass_strict_loan_consumer_credit_qoq",
                        "bank_credit_qoq",
                    }
                ],
            )
        )

    all_estimates.extend(
        _estimate_surface(
            rows=merged_rows,
            treatment_label=SELECTED_LAG_SENSITIVITY_LABEL,
            treatment_spec=selected_lag_treatment_spec,
            control_ids=selected_lag_active_controls,
            outcome_ids=outcome_ids,
            horizons=HORIZONS,
            surface="pooled_baseline",
            sample_label="full_available",
        )
    )
    for sample in SAMPLE_WINDOWS[1:]:
        sample_rows_for_sensitivity = _rows_for_window(merged_rows, sample)
        all_estimates.extend(
            _estimate_surface(
                rows=sample_rows_for_sensitivity,
                treatment_label=SELECTED_LAG_SENSITIVITY_LABEL,
                treatment_spec=selected_lag_treatment_spec,
                control_ids=selected_lag_active_controls,
                outcome_ids=outcome_ids,
                horizons=HORIZONS,
                surface="sample_split",
                sample_label=str(sample["sample_label"]),
            )
        )
    all_estimates.extend(
        _lead_placebo_estimates(
            rows=merged_rows,
            treatment_label=SELECTED_LAG_SENSITIVITY_LABEL,
            treatment_spec=selected_lag_treatment_spec,
            control_ids=selected_lag_active_controls,
            outcome_ids=[
                outcome
                for outcome in outcome_ids
                if outcome
                in {
                    "tdcpass_strict_loan_core_min_qoq",
                    "tdcpass_strict_loan_mortgages_qoq",
                    "tdcpass_strict_loan_consumer_credit_qoq",
                    "bank_credit_qoq",
                }
            ],
        )
    )

    estimates_path = paths.output / "models" / "tier2_credit_causality_state_estimates.csv"
    _write_csv(estimates_path, all_estimates)

    sample_rows = _summarize_sample_splits(all_estimates)
    sample_path = paths.reports / "tier2_credit_causality_sample_splits.csv"
    _write_csv(sample_path, sample_rows)

    state_rows = _summarize_state_effects(all_estimates, state_metadata)
    state_path = paths.reports / "tier2_credit_causality_state_effects.csv"
    _write_csv(state_path, state_rows)

    lead_rows = _summarize_leads(all_estimates)
    lead_path = paths.reports / "tier2_credit_causality_lead_placebos.csv"
    _write_csv(lead_path, lead_rows)

    pattern_rows = _pattern_table(all_estimates)
    pattern_path = paths.reports / "tier2_credit_causality_pattern_table.csv"
    _write_csv(pattern_path, pattern_rows)

    summary = _readout_rows(sample_rows, state_rows, lead_rows)
    readout_path = paths.reports / "tier2_credit_causality_readout.md"
    _write_markdown(
        readout_path,
        summary=summary,
        sample_rows=sample_rows,
        state_rows=state_rows,
        lead_rows=lead_rows,
        missing_outcomes=missing_outcomes,
        state_metadata=state_metadata,
        innovation_diagnostics=innovation_diagnostics,
    )

    summary_path = paths.manifests / "tier2_credit_causality_summary.json"
    write_json(
        summary_path,
        {
            "generated_at": utc_now_iso(),
            "anchor_job_id": ANCHOR_JOB_ID,
            "k_screened": K_SCREENED,
            "factor_count": factor_count,
            "screened_count": screened_count,
            "control_policy_mode": CONTROL_POLICY_MODE,
            "control_ids": control_ids,
            "treatment_labels": TREATMENT_LABELS,
            "innovation_treatment_labels": INNOVATION_TREATMENT_LABELS,
            "selected_lag_sensitivity_label": SELECTED_LAG_SENSITIVITY_LABEL,
            "selected_lag_controls": selected_lag_controls,
            "innovation_diagnostics": innovation_diagnostics,
            "outcome_ids": outcome_ids,
            "missing_outcomes": missing_outcomes,
            "state_metadata": state_metadata,
            "active_state_ids": active_state_ids,
            "sample_windows": SAMPLE_WINDOWS,
            "estimates_path": str(estimates_path),
            "sample_splits_path": str(sample_path),
            "state_effects_path": str(state_path),
            "lead_placebos_path": str(lead_path),
            "pattern_table_path": str(pattern_path),
            "readout_path": str(readout_path),
            "rows_written": len(all_estimates),
            "primary_readout": summary,
        },
    )
    print(
        json.dumps(
            {
                "estimates_path": str(estimates_path),
                "sample_splits_path": str(sample_path),
                "state_effects_path": str(state_path),
                "lead_placebos_path": str(lead_path),
                "pattern_table_path": str(pattern_path),
                "readout_path": str(readout_path),
                "summary_path": str(summary_path),
                "rows_written": len(all_estimates),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
