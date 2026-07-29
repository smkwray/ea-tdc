from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ea_tdc.designs.quarterly import build_quarterly_design
from ea_tdc.estimation import _estimate_rows, _write_estimates_csv
from ea_tdc.open_contract import CANONICAL_RESIDUAL_ID
from ea_tdc.paths import project_paths
from ea_tdc.residualized_shock import _load_factor_branch
from ea_tdc.utils import utc_now_iso, write_json
from run_tier2_canonical_component_credit_attribution import (
    ANCHOR_JOB_ID,
    CONTROL_POLICY_MODE,
    FACTOR_COUNT,
    K_SCREENED,
    _augment_rows,
    _load_component_panel,
    _load_manifest,
)


HORIZONS = [0, 4]

OUTCOME_SPECS = {
    "matched_total_deposits": {
        "family": "deposits",
        "label": "Total deposits",
        "unit_type": "dollar_per_dollar",
    },
    "domestic_nonbank_deposits_qoq": {
        "family": "deposits",
        "label": "Domestic nonbank deposits",
        "unit_type": "dollar_per_dollar",
    },
    CANONICAL_RESIDUAL_ID: {
        "family": "deposit_residual",
        "label": "Other deposit component, long-history canon",
        "unit_type": "dollar_per_dollar",
    },
    "tdcpass_strict_loan_core_min_qoq": {
        "family": "credit",
        "label": "Strict loan-core proxy",
        "unit_type": "quantity_mil_to_dollar",
    },
    "tdcpass_strict_loan_mortgages_qoq": {
        "family": "credit",
        "label": "Mortgage-created deposits proxy",
        "unit_type": "quantity_mil_to_dollar",
    },
    "tdcpass_strict_loan_consumer_credit_qoq": {
        "family": "credit",
        "label": "Consumer-credit-created deposits proxy",
        "unit_type": "quantity_mil_to_dollar",
    },
    "bank_credit_qoq": {
        "family": "bank_credit",
        "label": "Bank credit",
        "unit_type": "quantity_mil_to_dollar",
    },
    "bank_consumer_loans_qoq": {
        "family": "bank_credit",
        "label": "Bank consumer loans",
        "unit_type": "quantity_mil_to_dollar",
    },
    "bank_business_loans_qoq": {
        "family": "bank_credit",
        "label": "Bank business loans",
        "unit_type": "quantity_mil_to_dollar",
    },
    "bank_real_estate_loans_qoq": {
        "family": "bank_credit",
        "label": "Bank real-estate loans",
        "unit_type": "quantity_mil_to_dollar",
    },
    "bank_non_treasury_securities_qoq": {
        "family": "bank_assets",
        "label": "Bank non-Treasury securities",
        "unit_type": "quantity_mil_to_dollar",
    },
    "bank_treasury_securities_qoq": {
        "family": "bank_assets",
        "label": "Bank Treasury securities",
        "unit_type": "dollar_per_dollar",
    },
    "bank_treasury_securities_transactions_qoq": {
        "family": "bank_assets",
        "label": "Bank Treasury securities transactions",
        "unit_type": "dollar_per_dollar",
    },
    "bank_treasury_agency_securities_qoq": {
        "family": "bank_assets",
        "label": "Bank Treasury and agency securities",
        "unit_type": "quantity_mil_to_dollar",
    },
    "reserve_balances_qoq": {
        "family": "reserves",
        "label": "Reserve balances",
        "unit_type": "dollar_per_dollar",
    },
    "foreign_official_deposits_qoq": {
        "family": "reserves",
        "label": "Foreign official Fed deposits",
        "unit_type": "dollar_per_dollar",
    },
    "total_reserve_balances_plus_foreign_official_qoq": {
        "family": "reserves",
        "label": "Bank + foreign official Fed deposits",
        "unit_type": "dollar_per_dollar",
    },
    "reserve_balances_net_fed_treasury_qoq": {
        "family": "reserves",
        "label": "Reserves net Fed Treasury holdings",
        "unit_type": "dollar_per_dollar",
    },
    "total_reserves_plus_foreign_official_net_fed_treasury_qoq": {
        "family": "reserves",
        "label": "Bank + foreign official Fed deposits net Fed Treasury holdings",
        "unit_type": "dollar_per_dollar",
    },
    "dgs2": {
        "family": "rates",
        "label": "2-year Treasury yield",
        "unit_type": "rate_pct_to_bp",
    },
    "dgs10": {
        "family": "rates",
        "label": "10-year Treasury yield",
        "unit_type": "rate_pct_to_bp",
    },
    "dgs10_2y_spread": {
        "family": "rates",
        "label": "10y-2y spread",
        "unit_type": "rate_pct_to_bp",
    },
    "dgs10_3mo_spread": {
        "family": "rates",
        "label": "10y-3m spread",
        "unit_type": "rate_pct_to_bp",
    },
    "mortgage_30y": {
        "family": "rates",
        "label": "30-year mortgage rate",
        "unit_type": "rate_pct_to_bp",
    },
    "mortgage_30y_dgs10_spread": {
        "family": "rates",
        "label": "Mortgage-Treasury spread",
        "unit_type": "rate_pct_to_bp",
    },
    "repo_spread": {
        "family": "money_markets",
        "label": "Repo spread",
        "unit_type": "rate_pct_to_bp",
    },
}

FOCUS_OUTCOMES = [
    "matched_total_deposits",
    "reserve_balances_qoq",
    "total_reserve_balances_plus_foreign_official_qoq",
    "reserve_balances_net_fed_treasury_qoq",
    "total_reserves_plus_foreign_official_net_fed_treasury_qoq",
    "tdcpass_strict_loan_core_min_qoq",
    "tdcpass_strict_loan_mortgages_qoq",
    "tdcpass_strict_loan_consumer_credit_qoq",
    "bank_treasury_securities_qoq",
    "bank_treasury_securities_transactions_qoq",
    "bank_treasury_agency_securities_qoq",
    "dgs2",
    "dgs10",
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


def _format_value(value: Any) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return value
        if value.strip().isdigit():
            return value
    else:
        number = float(value)
    if abs(number) >= 100:
        return f"{number:.1f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    if abs(number) >= 1:
        return f"{number:.3f}"
    return f"{number:.4f}"


def _normalization_multiplier(unit_type: str) -> float:
    if unit_type == "quantity_mil_to_dollar":
        return 1000.0
    return 1.0


def _typical_effect(beta: float, source_sd_mil: float, unit_type: str) -> tuple[str, float]:
    if unit_type in {"dollar_per_dollar", "quantity_mil_to_dollar"}:
        normalized_beta = beta * _normalization_multiplier(unit_type)
        return "usd_billions_per_1sd_component", normalized_beta * source_sd_mil / 1000.0
    if unit_type == "rate_pct_to_bp":
        return "basis_points_per_1sd_component", beta * source_sd_mil * 100.0
    return "outcome_units_per_1sd_component", beta * source_sd_mil


def _normalized_beta(beta: float, unit_type: str) -> tuple[str, float]:
    if unit_type in {"dollar_per_dollar", "quantity_mil_to_dollar"}:
        return "dollars_per_dollar_component", beta * _normalization_multiplier(unit_type)
    if unit_type == "rate_pct_to_bp":
        return "rate_units_per_component_mil", beta
    return "raw_outcome_units_per_component_mil", beta


def _ranking_rows(
    estimates: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for estimate in estimates:
        outcome = str(estimate.get("outcome", ""))
        spec = OUTCOME_SPECS.get(outcome)
        if spec is None:
            continue
        component_id = str(estimate.get("component_id", estimate.get("treatment_id", "")))
        component_meta = metadata.get(component_id, {})
        beta = float(estimate["beta"])
        source_sd_mil = float(component_meta.get("source_sd_mil", 0.0) or 0.0)
        normalized_unit, normalized_value = _normalized_beta(beta, str(spec["unit_type"]))
        typical_unit, typical_value = _typical_effect(beta, source_sd_mil, str(spec["unit_type"]))
        rows.append(
            {
                "component_id": component_id,
                "component_label": component_meta.get("label", component_id),
                "component_group": component_meta.get("group", ""),
                "outcome": outcome,
                "outcome_label": spec["label"],
                "outcome_family": spec["family"],
                "horizon": int(estimate.get("horizon", 0)),
                "beta": beta,
                "normalized_beta": normalized_value,
                "normalized_unit": normalized_unit,
                "typical_effect": typical_value,
                "typical_effect_unit": typical_unit,
                "abs_typical_effect": abs(typical_value),
                "p_value": estimate.get("p_value_normal", ""),
                "n": estimate.get("n", ""),
                "source_sd_mil": source_sd_mil,
                "source_start_quarter": component_meta.get("source_start_quarter", ""),
                "source_end_quarter": component_meta.get("source_end_quarter", ""),
                "source_nonmissing_quarters": component_meta.get("source_nonmissing_quarters", ""),
                "dropped_control_ids": estimate.get("dropped_control_ids", ""),
            }
        )

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["outcome"]), int(row["horizon"]))].append(row)
    for group_rows in grouped.values():
        group_rows.sort(key=lambda row: float(row["abs_typical_effect"]), reverse=True)
        for rank, row in enumerate(group_rows, start=1):
            row["rank_typical_within_outcome_horizon"] = rank
    rows.sort(key=lambda row: (str(row["outcome_family"]), str(row["outcome"]), int(row["horizon"]), int(row["rank_typical_within_outcome_horizon"])))
    return rows


def _group_signal_rows(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in ranked:
        grouped[(str(row["outcome"]), int(row["horizon"]), str(row["component_group"]))].append(row)
    rows: list[dict[str, Any]] = []
    for (outcome, horizon, component_group), group_rows in grouped.items():
        signal_rows = [
            row
            for row in group_rows
            if str(row.get("p_value", "")).strip()
            and float(row["p_value"]) <= 0.10
        ]
        selected_rows = signal_rows or group_rows
        selected_rows.sort(key=lambda row: float(row["abs_typical_effect"]), reverse=True)
        top = selected_rows[0]
        rows.append(
            {
                "outcome": outcome,
                "outcome_label": top["outcome_label"],
                "outcome_family": top["outcome_family"],
                "horizon": horizon,
                "component_group": component_group,
                "top_component_id": top["component_id"],
                "top_component_label": top["component_label"],
                "top_typical_effect": top["typical_effect"],
                "top_typical_effect_unit": top["typical_effect_unit"],
                "top_p_value": top["p_value"],
                "top_n": top["n"],
                "top_source_start_quarter": top["source_start_quarter"],
                "top_source_nonmissing_quarters": top["source_nonmissing_quarters"],
                "selection_rule": "largest_abs_typical_p_le_0.10" if signal_rows else "largest_abs_typical_no_p_filter",
            }
        )
    rows.sort(key=lambda row: (str(row["outcome_family"]), str(row["outcome"]), int(row["horizon"]), -abs(float(row["top_typical_effect"]))))
    return rows


def _interpretation_rows(group_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    focus_groups = {
        "base_flow",
        "plumbing_adjustment",
        "sector_interest_correction",
        "detailed_interest_correction",
        "detailed_component_total",
    }
    rows: list[dict[str, Any]] = []
    for outcome in FOCUS_OUTCOMES:
        for horizon in HORIZONS:
            candidates = [
                row
                for row in group_rows
                if row["outcome"] == outcome
                and int(row["horizon"]) == horizon
                and row["component_group"] in focus_groups
            ]
            if not candidates:
                continue
            candidates.sort(key=lambda row: abs(float(row["top_typical_effect"])), reverse=True)
            top = candidates[:4]
            leading_groups = "; ".join(
                f"{row['component_group']}: {row['top_component_label']} ({_format_value(row['top_typical_effect'])} {row['top_typical_effect_unit']})"
                for row in top
            )
            rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": top[0]["outcome_label"],
                    "outcome_family": top[0]["outcome_family"],
                    "horizon": horizon,
                    "leading_component_groups": leading_groups,
                    "interpretation_hint": _interpretation_hint(outcome, top),
                }
            )
    return rows


def _interpretation_hint(outcome: str, top_rows: list[dict[str, Any]]) -> str:
    groups = [str(row["component_group"]) for row in top_rows[:3]]
    labels = " ".join(str(row["top_component_label"]).lower() for row in top_rows[:3])
    if any(group in {"sector_interest_correction", "detailed_interest_correction", "detailed_component_total"} for group in groups):
        if outcome in {"tdcpass_strict_loan_core_min_qoq", "tdcpass_strict_loan_mortgages_qoq", "tdcpass_strict_loan_consumer_credit_qoq", "dgs2", "dgs10", "mortgage_30y"}:
            return "Loads on interest-correction/accrual components; treat as rate-regime or portfolio-channel evidence before calling it mechanical crowding out."
    if "base_flow" in groups and outcome in {
        "matched_total_deposits",
        "domestic_nonbank_deposits_qoq",
        "reserve_balances_qoq",
        "foreign_official_deposits_qoq",
        "total_reserve_balances_plus_foreign_official_qoq",
        "reserve_balances_net_fed_treasury_qoq",
        "total_reserves_plus_foreign_official_net_fed_treasury_qoq",
    }:
        return "Loads on base-flow/liquidity components; this is closer to the core Treasury-deposit/liquidity channel."
    if "mmf" in labels or "rrp" in labels:
        return "Loads on MMF/RRP adjustment; interpret through money-market plumbing and reserve redistribution."
    return "Mixed loading; use as a diagnostic ranking rather than a causal component decomposition."


def _md_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        cells: list[str] = []
        for column in columns:
            cells.append(_format_value(row.get(column, "")))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _write_markdown(
    path: Path,
    ranked: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    interpretation: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Tier 2 Component Outcome Decomposition",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        "This diagnostic asks which signed Tier 2 components carry the outcome patterns for deposits, credit, reserves, rates, and bank balance-sheet outcomes. Components are not mutually exclusive in every block, so rankings should be read as diagnostic loadings rather than an additive causal decomposition.",
        "",
        "Ranking uses the estimated effect of a one-standard-deviation component move. Quantity outcomes are reported in USD billions per one-standard-deviation component move; rate outcomes are reported in basis points.",
        "",
        "## Interpretation Map",
        "",
    ]
    lines.extend(
        _md_table(
            interpretation,
            ["outcome_label", "horizon", "leading_component_groups", "interpretation_hint"],
        )
    )
    lines.extend(["", "## H=0 Top Components", ""])
    for outcome in FOCUS_OUTCOMES:
        focus = [
            row
            for row in ranked
            if row["outcome"] == outcome and int(row["horizon"]) == 0
        ][:8]
        if not focus:
            continue
        lines.extend(
            [
                f"### {OUTCOME_SPECS[outcome]['label']}",
                "",
                *_md_table(
                    focus,
                    [
                        "rank_typical_within_outcome_horizon",
                        "component_label",
                        "component_group",
                        "typical_effect",
                        "typical_effect_unit",
                        "p_value",
                        "n",
                        "source_start_quarter",
                    ],
                ),
                "",
            ]
        )
    lines.extend(["## Group Signal Table", ""])
    compact_groups = [
        row
        for row in group_rows
        if row["outcome"] in FOCUS_OUTCOMES and int(row["horizon"]) == 0
    ][:80]
    lines.extend(
        _md_table(
            compact_groups,
            [
                "outcome_label",
                "component_group",
                "top_component_label",
                "top_typical_effect",
                "top_typical_effect_unit",
                "top_p_value",
                "top_n",
                "selection_rule",
            ],
        )
    )
    lines.extend(["", "## Component Coverage", ""])
    lines.extend(
        _md_table(
            sorted(metadata_rows, key=lambda row: (str(row["group"]), str(row["component_id"]))),
            [
                "component_id",
                "label",
                "group",
                "source_start_quarter",
                "source_end_quarter",
                "source_nonmissing_quarters",
                "source_sd_mil",
            ],
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    paths = project_paths(ROOT)
    build_quarterly_design(paths, job_id=ANCHOR_JOB_ID)
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
    component_panel, metadata = _load_component_panel()
    augmented_rows = _augment_rows(factor_rows, component_panel)

    augmented_bundle_path = paths.bundles / "designs" / "tier2_component_outcome_decomposition__quarterly_bundle.csv"
    _write_csv(augmented_bundle_path, augmented_rows)

    all_estimates: list[dict[str, Any]] = []
    component_ids = sorted(metadata)
    outcome_ids = list(OUTCOME_SPECS)
    for component_id in component_ids:
        estimates = _estimate_rows(
            estimator="lp",
            bundle_rows=augmented_rows,
            treatment_id=component_id,
            control_ids=control_ids,
            outcome_ids=outcome_ids,
            horizons=HORIZONS,
            response_type="direct_at_h",
            job_id=f"tier2_component_outcome_decomposition__{component_id}",
            instrument_ids=[],
            state_id="",
        )
        for row in estimates:
            row["component_id"] = component_id
            row["component_label"] = metadata[component_id]["label"]
            row["component_group"] = metadata[component_id]["group"]
            row["pinned_anchor_job_id"] = ANCHOR_JOB_ID
            row["pinned_k_screened"] = K_SCREENED
            row["pinned_control_policy_mode"] = CONTROL_POLICY_MODE
        all_estimates.extend(estimates)

    ranked = _ranking_rows(all_estimates, metadata)
    group_rows = _group_signal_rows(ranked)
    interpretation = _interpretation_rows(group_rows)
    metadata_rows = list(metadata.values())

    estimates_path = paths.output / "models" / "tier2_component_outcome_decomposition_estimates.csv"
    ranking_path = paths.reports / "tier2_component_outcome_decomposition_rankings.csv"
    group_path = paths.reports / "tier2_component_outcome_decomposition_group_signals.csv"
    interpretation_path = paths.reports / "tier2_component_outcome_decomposition_interpretation.csv"
    metadata_path = paths.reports / "tier2_component_outcome_decomposition_components.csv"
    md_path = paths.reports / "tier2_component_outcome_decomposition.md"
    summary_path = paths.manifests / "tier2_component_outcome_decomposition_summary.json"

    _write_estimates_csv(estimates_path, all_estimates)
    _write_csv(ranking_path, ranked)
    _write_csv(group_path, group_rows)
    _write_csv(interpretation_path, interpretation)
    _write_csv(metadata_path, metadata_rows)
    _write_markdown(md_path, ranked, group_rows, interpretation, metadata_rows)
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
            "augmented_bundle_path": str(augmented_bundle_path),
            "estimates_path": str(estimates_path),
            "ranking_path": str(ranking_path),
            "group_signal_path": str(group_path),
            "interpretation_path": str(interpretation_path),
            "metadata_path": str(metadata_path),
            "markdown_path": str(md_path),
            "components_estimated": len(component_ids),
            "outcomes_estimated": len(outcome_ids),
            "estimate_rows_written": len(all_estimates),
            "ranking_rows_written": len(ranked),
            "group_signal_rows_written": len(group_rows),
            "interpretation_rows_written": len(interpretation),
            "horizons": HORIZONS,
        },
    )
    print(
        json.dumps(
            {
                "markdown_path": str(md_path),
                "summary_path": str(summary_path),
                "components_estimated": len(component_ids),
                "outcomes_estimated": len(outcome_ids),
                "estimate_rows_written": len(all_estimates),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
