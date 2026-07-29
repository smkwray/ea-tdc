from __future__ import annotations

import csv
import json
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
from ea_tdc.estimation import _coerce_float
from ea_tdc.open_contract import CANONICAL_RESIDUAL_ID, CANONICAL_TREATMENT_ID
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
from run_tier2_state_dependent_credit_causality import (
    PRIMARY_TREATMENT_LABEL,
    SELECTED_LAG_CONTROLS,
    SELECTED_LAG_PERIODS,
    SELECTED_LAG_SENSITIVITY_LABEL,
    SELECTED_LAG_SOURCES,
    _active_controls,
    _add_future_treatment,
    _add_selected_lags,
    _augment_states,
    _estimate_surface,
    _format_value,
    _in_quarter_range,
    _insert_controls_before_factor_tail,
)


HORIZONS = [0, 1, 2, 4]
RESIDUAL_OUTCOME = CANONICAL_RESIDUAL_ID
TREATMENT_ID = CANONICAL_TREATMENT_ID
TREATMENT_LABEL = SELECTED_LAG_SENSITIVITY_LABEL
LEADS = [1, 2, 4]

# scale_to_millions converts a one-unit outcome change into USD millions.
# Most tdcest/tdcpass deposit decompositions are already carried in millions;
# many FRED/H.8/Z.1 proxy blocks are in billions.
CANDIDATES = [
    # Residual / deposit perimeter
    {"group": "residual_perimeter", "id": "matched_total_deposits", "label": "Matched total deposits", "scale_to_millions": 1.0},
    {"group": "residual_perimeter", "id": RESIDUAL_OUTCOME, "label": "Other component, same long-history Tier 2 treatment", "scale_to_millions": 1.0},
    {"group": "residual_perimeter", "id": "other_component_tier2_mmf_rrp_plumbing_adjusted_qoq", "label": "Other component, plumbing-adjusted TDC", "scale_to_millions": 1.0},
    {"group": "residual_perimeter", "id": "domestic_nonbank_deposits_qoq", "label": "Domestic nonbank deposits", "scale_to_millions": 1.0},
    {"group": "residual_perimeter", "id": "domestic_nonbank_other_component_qoq", "label": "Domestic nonbank other component", "scale_to_millions": 1.0},
    {"group": "residual_perimeter", "id": "domestic_nonbank_other_component_no_row_qoq", "label": "Domestic nonbank residual excluding ROW", "scale_to_millions": 1.0},
    {"group": "residual_perimeter", "id": "domestic_nonbank_other_component_no_toc_no_row_qoq", "label": "Domestic nonbank residual excluding TOC and ROW", "scale_to_millions": 1.0},
    {"group": "residual_perimeter", "id": "m2_qoq", "label": "M2 level change, script-derived", "scale_to_millions": 1000.0},
    # Liability substitution
    {"group": "liability_substitution", "id": "large_time_deposits_qoq", "label": "Large time deposits", "scale_to_millions": 1000.0},
    {"group": "liability_substitution", "id": "retail_mmf_assets_qoq", "label": "Retail MMF assets", "scale_to_millions": 1000.0},
    {"group": "liability_substitution", "id": "institutional_mmf_assets_qoq", "label": "Institutional MMF assets", "effect_unit": "source_units_per_100b_tdc"},
    {"group": "liability_substitution", "id": "deposit_substitution_block_qoq", "label": "Deposit-substitution proxy block", "effect_unit": "source_units_per_100b_tdc"},
    {"group": "liability_substitution", "id": "accounting_deposit_substitution_qoq", "label": "Accounting deposit-substitution block", "effect_unit": "source_units_per_100b_tdc"},
    # Liquidity plumbing
    {"group": "liquidity_plumbing", "id": "tga_balance_qoq", "label": "TGA balance change", "scale_to_millions": 1.0},
    {"group": "liquidity_plumbing", "id": "on_rrp_balance_qoq", "label": "ON RRP balance change", "scale_to_millions": 1000.0},
    {"group": "liquidity_plumbing", "id": "mmf_on_rrp_plumbing_absorption_qoq", "label": "MMF/ON RRP plumbing absorption", "scale_to_millions": 1.0},
    {"group": "liquidity_plumbing", "id": "reserve_balances_qoq", "label": "Reserve balances", "scale_to_millions": 1.0},
    {"group": "liquidity_plumbing", "id": "foreign_official_deposits_qoq", "label": "Foreign official Fed deposits", "scale_to_millions": 1.0},
    {"group": "liquidity_plumbing", "id": "total_reserve_balances_plus_foreign_official_qoq", "label": "Bank + foreign official Fed deposits", "scale_to_millions": 1.0},
    {"group": "liquidity_plumbing", "id": "reserve_balances_net_fed_treasury_qoq", "label": "Reserve balances net Fed Treasury holdings", "scale_to_millions": 1.0},
    {"group": "liquidity_plumbing", "id": "total_reserves_plus_foreign_official_net_fed_treasury_qoq", "label": "Broad Fed deposits net Fed Treasury holdings", "scale_to_millions": 1.0},
    {"group": "liquidity_plumbing", "id": "public_liquidity_proxy_block_qoq", "label": "Public-liquidity proxy block", "effect_unit": "source_units_per_100b_tdc"},
    {"group": "liquidity_plumbing", "id": "accounting_public_liquidity_qoq", "label": "Accounting public-liquidity block", "effect_unit": "source_units_per_100b_tdc"},
    {"group": "rate_plumbing", "id": "repo_spread", "label": "Repo spread", "effect_unit": "basis_points_per_100b_tdc"},
    # Bank asset side
    {"group": "bank_asset_side", "id": "tdcpass_strict_loan_core_min_qoq", "label": "Strict loan core minimum", "scale_to_millions": 1000.0},
    {"group": "bank_asset_side", "id": "tdcpass_strict_loan_mortgages_qoq", "label": "Strict mortgage credit", "scale_to_millions": 1000.0},
    {"group": "bank_asset_side", "id": "tdcpass_strict_loan_consumer_credit_qoq", "label": "Strict consumer credit", "scale_to_millions": 1000.0},
    {"group": "bank_asset_side", "id": "bank_credit_qoq", "label": "Bank credit", "scale_to_millions": 1000.0},
    {"group": "bank_asset_side", "id": "bank_consumer_loans_qoq", "label": "Bank consumer loans", "scale_to_millions": 1000.0},
    {"group": "bank_asset_side", "id": "bank_business_loans_qoq", "label": "Bank business loans", "scale_to_millions": 1000.0},
    {"group": "bank_asset_side", "id": "bank_real_estate_loans_qoq", "label": "Bank real-estate loans", "scale_to_millions": 1000.0},
    {"group": "bank_asset_side", "id": "bank_treasury_securities_qoq", "label": "Bank Treasury securities", "scale_to_millions": 1.0},
    {"group": "bank_asset_side", "id": "bank_treasury_securities_transactions_qoq", "label": "Bank Treasury-security transactions", "scale_to_millions": 1.0},
    {"group": "bank_asset_side", "id": "bank_treasury_agency_securities_qoq", "label": "Bank Treasury-and-agency securities", "scale_to_millions": 1000.0},
    {"group": "bank_asset_side", "id": "bank_non_treasury_securities_qoq", "label": "Bank non-Treasury securities", "scale_to_millions": 1000.0},
    {"group": "bank_asset_side", "id": "bank_balance_sheet_proxy_block_qoq", "label": "Bank balance-sheet proxy block", "effect_unit": "source_units_per_100b_tdc"},
    {"group": "accounting_blocks", "id": "accounting_bank_balance_sheet_qoq", "label": "Accounting bank balance-sheet block", "effect_unit": "source_units_per_100b_tdc"},
    {"group": "accounting_blocks", "id": "accounting_external_flow_qoq", "label": "Accounting external-flow block", "effect_unit": "source_units_per_100b_tdc"},
    {"group": "accounting_blocks", "id": "accounting_identity_total_qoq", "label": "Accounting identity total", "effect_unit": "source_units_per_100b_tdc"},
    {"group": "accounting_blocks", "id": "accounting_identity_gap_qoq", "label": "Accounting identity gap, base residual", "effect_unit": "source_units_per_100b_tdc"},
    {"group": "accounting_blocks", "id": "accounting_identity_gap_tier2_bank_only_qoq", "label": "Accounting identity gap, Tier 2 bank-only", "effect_unit": "source_units_per_100b_tdc"},
]

SAMPLE_WINDOWS = [
    {"sample_label": "full_available"},
    {"sample_label": "exclude_gfc", "exclude_windows": [("2007Q4", "2009Q2")]},
    {"sample_label": "exclude_covid", "exclude_windows": [("2020Q1", "2020Q2")]},
    {"sample_label": "exclude_transition_2019_2021", "exclude_windows": [("2019Q1", "2021Q4")]},
    {"sample_label": "exclude_gfc_covid_transition", "exclude_windows": [("2007Q4", "2009Q2"), ("2020Q1", "2020Q2"), ("2019Q1", "2021Q4")]},
    {"sample_label": "period_2002_2010", "include_windows": [("2002Q1", "2010Q4")]},
    {"sample_label": "period_2011_2021", "include_windows": [("2011Q1", "2021Q4")]},
    {"sample_label": "period_2022_2025", "include_windows": [("2022Q1", "2025Q4")]},
]


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


def _available_columns(rows: list[dict[str, str]]) -> set[str]:
    columns: set[str] = set()
    for row in rows:
        columns.update(row)
    return columns


def _add_m2_qoq(rows: list[dict[str, str]]) -> None:
    previous: float | None = None
    previous_available = ""
    for row in rows:
        current = _coerce_float(row.get("M2SL", ""))
        if current is None or previous is None:
            row["m2_qoq"] = ""
            row["m2_qoq__available_at"] = ""
            row["m2_qoq__source_repo"] = "script_derived_fred"
        else:
            row["m2_qoq"] = str(current - previous)
            row["m2_qoq__available_at"] = previous_available
            row["m2_qoq__source_repo"] = "script_derived_fred"
        previous = current
        previous_available = row.get("M2SL__available_at", "") or row.get("quarter", "")


def _rows_for_sample(rows: list[dict[str, str]], sample: dict[str, Any]) -> list[dict[str, str]]:
    include_windows = [(str(start), str(end)) for start, end in sample.get("include_windows", [])]
    exclude_windows = [(str(start), str(end)) for start, end in sample.get("exclude_windows", [])]
    output: list[dict[str, str]] = []
    for row in rows:
        quarter = str(row.get("quarter", ""))
        if include_windows and not any(_in_quarter_range(quarter, start, end) for start, end in include_windows):
            continue
        if exclude_windows and any(_in_quarter_range(quarter, start, end) for start, end in exclude_windows):
            continue
        output.append(dict(row))
    return output


def _effect(row: dict[str, Any], meta: dict[str, Any]) -> tuple[str, float | str]:
    beta_text = str(row.get("beta", "")).strip()
    if not beta_text:
        return "", ""
    beta = float(beta_text)
    if meta.get("effect_unit") == "basis_points_per_100b_tdc":
        return "basis_points_per_100b_tdc", beta * 100000.0 * 100.0
    if meta.get("effect_unit") == "source_units_per_100b_tdc":
        return "source_units_per_100b_tdc", beta * 100000.0
    scale = float(meta.get("scale_to_millions", 1.0))
    return "usd_billions_per_100b_tdc", beta * 100.0 * scale


def _candidate_map() -> dict[str, dict[str, Any]]:
    return {str(item["id"]): item for item in CANDIDATES}


def _decorate_rows(rows: list[dict[str, Any]], candidate_meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        outcome = str(row.get("outcome", ""))
        meta = candidate_meta.get(outcome, {"group": "", "label": outcome, "scale_to_millions": 1.0})
        unit, effect_value = _effect(row, meta)
        decorated = dict(row)
        decorated["candidate_group"] = meta.get("group", "")
        decorated["candidate_label"] = meta.get("label", outcome)
        decorated["attribution_effect_unit"] = unit
        decorated["attribution_effect_per_100b_tdc"] = effect_value
        output.append(decorated)
    return output


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_group": row.get("candidate_group", ""),
            "candidate_label": row.get("candidate_label", row.get("outcome", "")),
            "outcome": row.get("outcome", ""),
            "horizon": row.get("horizon", ""),
            "effect_per_100b_tdc": row.get("attribution_effect_per_100b_tdc", ""),
            "effect_unit": row.get("attribution_effect_unit", ""),
            "p_value": row.get("p_value_normal", ""),
            "n": row.get("n", ""),
            "control_ids_used": row.get("control_ids_used", ""),
            "dropped_control_ids": row.get("dropped_control_ids", ""),
        }
        for row in rows
    ]


def _lead_placebo_rows(
    rows: list[dict[str, str]],
    *,
    treatment_spec: dict[str, Any],
    control_ids: list[str],
    candidate_meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for lead in LEADS:
        lead_rows, lead_column = _add_future_treatment(rows, str(treatment_spec["treatment_id"]), lead)
        lead_spec = dict(treatment_spec)
        lead_spec["treatment_id"] = lead_column
        estimates = _estimate_surface(
            rows=lead_rows,
            treatment_label=TREATMENT_LABEL,
            treatment_spec=lead_spec,
            control_ids=control_ids,
            outcome_ids=[RESIDUAL_OUTCOME],
            horizons=[0],
            surface="residual_lead_placebo",
            sample_label=f"lead_{lead}",
        )
        for estimate in _decorate_rows(estimates, candidate_meta):
            estimate["lead_quarters"] = lead
            estimate["actual_treatment_id"] = treatment_spec["treatment_id"]
            estimate["future_treatment_id"] = lead_column
            output.append(estimate)
    return output


def _write_markdown(
    path: Path,
    *,
    attribution_rows: list[dict[str, Any]],
    residual_window_rows: list[dict[str, Any]],
    residual_state_rows: list[dict[str, Any]],
    residual_lead_rows: list[dict[str, Any]],
    missing_candidates: list[str],
) -> None:
    h0 = [row for row in attribution_rows if str(row.get("horizon")) == "0"]
    by_outcome = {str(row.get("outcome")): row for row in h0}
    residual = by_outcome.get(RESIDUAL_OUTCOME, {})
    deposits = by_outcome.get("matched_total_deposits", {})
    top_negative = sorted(
        [
            row
            for row in h0
            if row.get("attribution_effect_unit") == "usd_billions_per_100b_tdc"
            and str(row.get("attribution_effect_per_100b_tdc", "")).strip()
        ],
        key=lambda row: float(row["attribution_effect_per_100b_tdc"]),
    )[:8]
    top_positive = sorted(
        [
            row
            for row in h0
            if row.get("attribution_effect_unit") == "usd_billions_per_100b_tdc"
            and str(row.get("attribution_effect_per_100b_tdc", "")).strip()
        ],
        key=lambda row: float(row["attribution_effect_per_100b_tdc"]),
        reverse=True,
    )[:8]
    sig_leads = [
        row
        for row in residual_lead_rows
        if str(row.get("p_value_normal", "")).strip()
        and float(row["p_value_normal"]) < 0.1
    ]
    lines = [
        "# Tier 2 Missing-40 Residual Attribution",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        "This diagnostic uses the selected credit/rate lag specification from the main causality readout. It asks what named balance-sheet, liability, liquidity, and accounting candidates move with the negative non-TDC deposit component. The rows are not forced to sum to the residual.",
        "",
        "## Headline",
        "",
        f"- Matched deposits h=0: `{_format_value(deposits.get('attribution_effect_per_100b_tdc'))}` $B per +$100B TDC.",
        f"- Same-treatment other component h=0: `{_format_value(residual.get('attribution_effect_per_100b_tdc'))}` $B per +$100B TDC.",
        f"- Residual lead placebos at p<0.10: `{len(sig_leads)}` of `{len(residual_lead_rows)}`.",
        "",
        "## H=0 Attribution Candidates",
        "",
        "| group | candidate | effect per +$100B | p | n |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in h0:
        if row.get("attribution_effect_unit") != "usd_billions_per_100b_tdc":
            continue
        lines.append(
            f"| {row.get('candidate_group', '')} | {row.get('candidate_label', row.get('outcome', ''))} | {_format_value(row.get('attribution_effect_per_100b_tdc'))} | {_format_value(row.get('p_value_normal'))} | {row.get('n', '')} |"
        )
    lines.extend(["", "## H=0 Composite / Source-Unit Candidates", "", "These rows are useful for direction and relative movement, but their source units are mixed enough that they should not be read as clean dollar magnitudes.", "", "| group | candidate | source-unit effect per +$100B | p | n |", "| --- | --- | ---: | ---: | ---: |"])
    for row in h0:
        if row.get("attribution_effect_unit") != "source_units_per_100b_tdc":
            continue
        lines.append(
            f"| {row.get('candidate_group', '')} | {row.get('candidate_label', row.get('outcome', ''))} | {_format_value(row.get('attribution_effect_per_100b_tdc'))} | {_format_value(row.get('p_value_normal'))} | {row.get('n', '')} |"
        )
    lines.extend(["", "## H=0 Rate Candidates", "", "| group | candidate | bp per +$100B | p | n |", "| --- | --- | ---: | ---: | ---: |"])
    for row in h0:
        if row.get("attribution_effect_unit") != "basis_points_per_100b_tdc":
            continue
        lines.append(
            f"| {row.get('candidate_group', '')} | {row.get('candidate_label', row.get('outcome', ''))} | {_format_value(row.get('attribution_effect_per_100b_tdc'))} | {_format_value(row.get('p_value_normal'))} | {row.get('n', '')} |"
        )
    lines.extend(["", "## Largest Negative H=0 Rows", "", "| candidate | effect per +$100B | p |", "| --- | ---: | ---: |"])
    for row in top_negative:
        lines.append(
            f"| {row.get('candidate_label', row.get('outcome', ''))} | {_format_value(row.get('attribution_effect_per_100b_tdc'))} | {_format_value(row.get('p_value_normal'))} |"
        )
    lines.extend(["", "## Largest Positive H=0 Rows", "", "| candidate | effect per +$100B | p |", "| --- | ---: | ---: |"])
    for row in top_positive:
        lines.append(
            f"| {row.get('candidate_label', row.get('outcome', ''))} | {_format_value(row.get('attribution_effect_per_100b_tdc'))} | {_format_value(row.get('p_value_normal'))} |"
        )
    lines.extend(["", "## Residual Windows", "", "| sample | h | effect per +$100B | p | n |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in residual_window_rows:
        lines.append(
            f"| {row.get('sample_label', '')} | {row.get('horizon', '')} | {_format_value(row.get('attribution_effect_per_100b_tdc'))} | {_format_value(row.get('p_value_normal'))} | {row.get('n', '')} |"
        )
    lines.extend(["", "## Residual State Interactions", "", "| state | profile | h | profile effect per +$100B | interaction beta | interaction se | n |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in residual_state_rows:
        lines.append(
            f"| {row.get('state_id', '')} | {row.get('state_profile', '')} | {row.get('horizon', '')} | {_format_value(row.get('attribution_effect_per_100b_tdc'))} | {_format_value(row.get('state_interaction_beta'))} | {_format_value(row.get('state_interaction_se'))} | {row.get('n', '')} |"
        )
    lines.extend(["", "## Residual Leads", "", "| lead | effect per +$100B | p | n |", "| ---: | ---: | ---: | ---: |"])
    for row in residual_lead_rows:
        lines.append(
            f"| {row.get('lead_quarters', '')} | {_format_value(row.get('attribution_effect_per_100b_tdc'))} | {_format_value(row.get('p_value_normal'))} | {row.get('n', '')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Treat the missing-40 residual as an offsetting non-TDC deposit component, not as a named structural channel.",
            "- If named liability/liquidity/accounting candidates are large but unstable across windows or states, prefer a plumbing/perimeter interpretation.",
            "- If credit and securities rows remain too small or wrong-signed, do not use them as the main explanation for the residual.",
        ]
    )
    if missing_candidates:
        lines.extend(["", "## Missing Candidates", "", ", ".join(f"`{item}`" for item in missing_candidates)])
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
    _add_m2_qoq(rows)
    available_selected_lag_columns = _add_selected_lags(rows)
    selected_lag_controls = [
        control_id for control_id in SELECTED_LAG_CONTROLS if control_id in available_selected_lag_columns
    ]
    primary_spec = BRIDGE_TREATMENTS[PRIMARY_TREATMENT_LABEL]
    treatment_spec = dict(primary_spec)
    treatment_spec["use_method_tier_controls"] = False
    active_controls = _insert_controls_before_factor_tail(
        _active_controls(control_ids, primary_spec),
        selected_lag_controls,
    )
    rows, state_metadata, active_state_ids = _augment_states(rows)
    columns = _available_columns(rows)
    candidate_meta = _candidate_map()
    candidate_ids = [str(item["id"]) for item in CANDIDATES if str(item["id"]) in columns]
    missing_candidates = [str(item["id"]) for item in CANDIDATES if str(item["id"]) not in columns]

    attribution_estimates = _estimate_surface(
        rows=rows,
        treatment_label=TREATMENT_LABEL,
        treatment_spec=treatment_spec,
        control_ids=active_controls,
        outcome_ids=candidate_ids,
        horizons=HORIZONS,
        surface="missing40_attribution",
        sample_label="full_available",
    )
    attribution_rows = _decorate_rows(attribution_estimates, candidate_meta)

    residual_window_rows: list[dict[str, Any]] = []
    for sample in SAMPLE_WINDOWS:
        sample_rows = _rows_for_sample(rows, sample)
        estimates = _estimate_surface(
            rows=sample_rows,
            treatment_label=TREATMENT_LABEL,
            treatment_spec=treatment_spec,
            control_ids=active_controls,
            outcome_ids=[RESIDUAL_OUTCOME],
            horizons=HORIZONS,
            surface="missing40_residual_window",
            sample_label=str(sample["sample_label"]),
        )
        residual_window_rows.extend(_decorate_rows(estimates, candidate_meta))

    residual_state_rows: list[dict[str, Any]] = []
    for state_id in active_state_ids:
        estimates = _estimate_surface(
            rows=rows,
            treatment_label=TREATMENT_LABEL,
            treatment_spec=treatment_spec,
            control_ids=active_controls,
            outcome_ids=[RESIDUAL_OUTCOME],
            horizons=[0, 4],
            surface="missing40_residual_state",
            state_id=state_id,
        )
        residual_state_rows.extend(_decorate_rows(estimates, candidate_meta))

    residual_lead_rows = _lead_placebo_rows(
        rows,
        treatment_spec=treatment_spec,
        control_ids=active_controls,
        candidate_meta=candidate_meta,
    )

    estimates_path = paths.output / "models" / "tier2_missing40_residual_attribution_estimates.csv"
    _write_csv(estimates_path, [*attribution_rows, *residual_window_rows, *residual_state_rows, *residual_lead_rows])

    attribution_path = paths.reports / "tier2_missing40_residual_attribution_table.csv"
    _write_csv(attribution_path, _summary_rows(attribution_rows))
    window_path = paths.reports / "tier2_missing40_residual_windows.csv"
    _write_csv(window_path, _summary_rows(residual_window_rows))
    state_path = paths.reports / "tier2_missing40_residual_states.csv"
    _write_csv(state_path, _summary_rows(residual_state_rows))
    lead_path = paths.reports / "tier2_missing40_residual_leads.csv"
    _write_csv(lead_path, _summary_rows(residual_lead_rows))

    markdown_path = paths.reports / "tier2_missing40_residual_attribution.md"
    _write_markdown(
        markdown_path,
        attribution_rows=attribution_rows,
        residual_window_rows=residual_window_rows,
        residual_state_rows=residual_state_rows,
        residual_lead_rows=residual_lead_rows,
        missing_candidates=missing_candidates,
    )

    summary_path = paths.manifests / "tier2_missing40_residual_attribution_summary.json"
    write_json(
        summary_path,
        {
            "generated_at": utc_now_iso(),
            "anchor_job_id": ANCHOR_JOB_ID,
            "k_screened": K_SCREENED,
            "factor_count": factor_count,
            "screened_count": screened_count,
            "control_policy_mode": CONTROL_POLICY_MODE,
            "treatment_label": TREATMENT_LABEL,
            "treatment_id": TREATMENT_ID,
            "residual_outcome": RESIDUAL_OUTCOME,
            "horizons": HORIZONS,
            "selected_lag_sources": SELECTED_LAG_SOURCES,
            "selected_lag_periods": SELECTED_LAG_PERIODS,
            "selected_lag_controls": selected_lag_controls,
            "active_state_ids": active_state_ids,
            "state_metadata": state_metadata,
            "candidate_ids": candidate_ids,
            "missing_candidates": missing_candidates,
            "estimates_path": str(estimates_path),
            "attribution_path": str(attribution_path),
            "window_path": str(window_path),
            "state_path": str(state_path),
            "lead_path": str(lead_path),
            "markdown_path": str(markdown_path),
            "rows_written": len(attribution_rows) + len(residual_window_rows) + len(residual_state_rows) + len(residual_lead_rows),
        },
    )
    print(
        json.dumps(
            {
                "markdown_path": str(markdown_path),
                "summary_path": str(summary_path),
                "rows_written": len(attribution_rows) + len(residual_window_rows) + len(residual_state_rows) + len(residual_lead_rows),
                "candidate_count": len(candidate_ids),
                "missing_candidate_count": len(missing_candidates),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
