from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ea_tdc.open_contract import (  # noqa: E402
    CANONICAL_RESIDUAL_ID,
    CANONICAL_TREATMENT_LABEL,
)


ESTIMATES_PATH = ROOT / "output" / "models" / "tier2_pinned_factor_bridge_estimates.csv"
READOUT_CSV = ROOT / "output" / "reports" / "tier2_regression_promotion_readout.csv"
READOUT_MD = ROOT / "output" / "reports" / "tier2_regression_promotion_readout.md"
READOUT_JSON = ROOT / "output" / "manifests" / "tier2_regression_promotion_readout_summary.json"

TREATMENTS: dict[str, dict[str, str]] = {
    "modern_canonical_di_mmf_rrp_short": {
        "role": "modern measurement default",
        "coverage": "2022Q1-2025Q4",
        "recommendation": "Use for current-period accounting and charts; too short for standalone long-history inference.",
        "residual_id": "other_component_tier2_canonical_di_mmf_rrp_prop_qoq",
    },
    CANONICAL_TREATMENT_LABEL: {
        "role": "preferred long-history regression default",
        "coverage": "2002Q1-2025Q4",
        "recommendation": "Promote for long-history regressions with method-tier controls and sample-split disclosure.",
        "residual_id": CANONICAL_RESIDUAL_ID,
    },
    "regression_mmf_rrp_di_long": {
        "role": "matched-perimeter long-history companion",
        "coverage": "2002Q1-2025Q4",
        "recommendation": "Use beside the bank-only row when comparing to the DI modern default.",
        "residual_id": "other_component_tier2_regression_mmf_rrp_prop_di_np_cu_qoq",
    },
    "regression_no_mmf_bank_long": {
        "role": "no-MMF/RRP sensitivity",
        "coverage": "2002Q1-2025Q4",
        "recommendation": "Keep as a sensitivity that isolates the MMF/RRP add-back.",
        "residual_id": "other_component_tier2_regression_bank_only_qoq",
    },
    "legacy_h15_bank_sensitivity": {
        "role": "legacy WAMEST/H15 sensitivity",
        "coverage": "2002Q1-2025Q4",
        "recommendation": "Demote to appendix/sensitivity; it is not the preferred holder-interest object.",
        "residual_id": "other_component_tier2_legacy_h15_bank_only_qoq",
    },
    "available_plumbing_bridge": {
        "role": "bridge-only convenience proxy",
        "coverage": "2013Q4-2025Q4 approximately",
        "recommendation": "Use only as a bridge check; it is not the constrained component canonical row.",
        "residual_id": "other_component_tier2_mmf_rrp_plumbing_adjusted_qoq",
    },
}

OUTCOMES: list[dict[str, Any]] = [
    {"label": "Total deposits", "id": "matched_total_deposits", "multiplier": 1.0},
    {"label": "Treatment residual", "id": "__treatment_residual__", "multiplier": 1.0},
    {"label": "TGA balance", "id": "tga_balance_qoq", "multiplier": 1.0},
    {"label": "Reserves", "id": "reserve_balances_qoq", "multiplier": 1.0},
    {
        "label": "Reserves net Fed Treasury",
        "id": "reserve_balances_net_fed_treasury_qoq",
        "multiplier": 1.0,
    },
    {"label": "Loan-created deposits core", "id": "tdcpass_strict_loan_core_min_qoq", "multiplier": 1000.0},
    {"label": "Consumer credit created deposits", "id": "tdcpass_strict_loan_consumer_credit_qoq", "multiplier": 1000.0},
    {"label": "Mortgage created deposits", "id": "tdcpass_strict_loan_mortgages_qoq", "multiplier": 1000.0},
    {"label": "Bank consumer loans", "id": "bank_consumer_loans_qoq", "multiplier": 1000.0},
    {"label": "Bank real-estate loans", "id": "bank_real_estate_loans_qoq", "multiplier": 1000.0},
    {"label": "Large time deposits", "id": "large_time_deposits_qoq", "multiplier": 1000.0},
]

HORIZONS = [0, 1, 2, 4, 8]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _f(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def _estimate_index(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], dict[str, str]]:
    indexed: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        label = str(row.get("treatment_label", ""))
        outcome = str(row.get("outcome", ""))
        horizon = int(row.get("horizon", "0") or 0)
        indexed[(label, outcome, horizon)] = row
    return indexed


def build_readout() -> list[dict[str, Any]]:
    rows = _read_csv(ESTIMATES_PATH)
    indexed = _estimate_index(rows)
    readout: list[dict[str, Any]] = []
    for treatment_label, spec in TREATMENTS.items():
        for outcome_spec in OUTCOMES:
            outcome_id = str(outcome_spec["id"])
            source_id = spec["residual_id"] if outcome_id == "__treatment_residual__" else outcome_id
            for horizon in HORIZONS:
                est = indexed.get((treatment_label, source_id, horizon))
                if not est:
                    continue
                multiplier = float(outcome_spec["multiplier"])
                beta = _f(est.get("beta"))
                se = _f(est.get("se"))
                lower95 = _f(est.get("lower95"))
                upper95 = _f(est.get("upper95"))
                readout.append(
                    {
                        "treatment_label": treatment_label,
                        "role": spec["role"],
                        "coverage": spec["coverage"],
                        "outcome_label": outcome_spec["label"],
                        "outcome_id": source_id,
                        "horizon": horizon,
                        "beta_per_1_tdc": "" if beta is None else beta * multiplier,
                        "se_per_1_tdc": "" if se is None else se * multiplier,
                        "lower95_per_1_tdc": "" if lower95 is None else lower95 * multiplier,
                        "upper95_per_1_tdc": "" if upper95 is None else upper95 * multiplier,
                        "p_value": est.get("p_value_normal", ""),
                        "n": est.get("n", ""),
                        "recommendation": spec["recommendation"],
                        "normalization_note": (
                            "Loan/security/deposit quantity rows are multiplied by 1000 to convert billions-per-million into dollars per $1 TDC."
                            if multiplier != 1.0
                            else "Already in dollars per $1 TDC scale."
                        ),
                    }
                )
    return readout


def write_markdown(readout: list[dict[str, Any]]) -> None:
    by_key = {
        (row["treatment_label"], row["outcome_label"], int(row["horizon"])): row
        for row in readout
    }
    lines = [
        "# Tier 2 Regression Promotion Readout",
        "",
        "## Decision",
        "",
        "Promote `tdc_tier2_regression_mmf_rrp_prop_bank_only_ru_flow` as the preferred long-history regression row. Keep `tdc_tier2_canonical_depository_institution_mmf_rrp_prop_ru_flow` as the modern/current-period measurement default. The DI regression+MMF/RRP row is the matched-perimeter companion, not the main long-history default.",
        "",
        "This closes the feasible method recommendations under the current source stack: explicit regression+MMF/RRP rows exist, method-tier controls survive into EA-TDC, the old H15 comparator is separated, and the 2019-2021 transition audit no longer shows a unit-break collapse after the `tdcest` source-constraint fix.",
        "",
        "## H=0 Comparison",
        "",
        "| treatment | role | deposits | p | n | residual | TGA | loan core | consumer credit | mortgages | recommendation |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for label, spec in TREATMENTS.items():
        def h0(outcome: str) -> dict[str, Any] | None:
            return by_key.get((label, outcome, 0))

        deposits = h0("Total deposits")
        residual = h0("Treatment residual")
        tga = h0("TGA balance")
        loan_core = h0("Loan-created deposits core")
        consumer = h0("Consumer credit created deposits")
        mortgages = h0("Mortgage created deposits")
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    spec["role"],
                    _fmt(deposits and deposits["beta_per_1_tdc"]),
                    _fmt(deposits and deposits["p_value"]),
                    str(deposits and deposits["n"] or ""),
                    _fmt(residual and residual["beta_per_1_tdc"]),
                    _fmt(tga and tga["beta_per_1_tdc"]),
                    _fmt(loan_core and loan_core["beta_per_1_tdc"]),
                    _fmt(consumer and consumer["beta_per_1_tdc"]),
                    _fmt(mortgages and mortgages["beta_per_1_tdc"]),
                    spec["recommendation"],
                ]
            )
            + " |"
        )

    preferred = CANONICAL_TREATMENT_LABEL
    lines.extend(
        [
            "",
            "## Preferred Long-History Row By Horizon",
            "",
            "Values are direct-at-h responses per $1 TDC. Loan rows are normalized from billions-per-million source units.",
            "",
            "| outcome | h0 | h1 | h2 | h4 | h8 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for outcome_spec in OUTCOMES:
        label = str(outcome_spec["label"])
        values = []
        for horizon in HORIZONS:
            row = by_key.get((preferred, label, horizon))
            values.append(_fmt(row and row["beta_per_1_tdc"]))
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Required Caveats",
            "",
            "- The modern measurement default remains the short constrained DI/MMF/RRP row, not the long-history proxy.",
            "- The promoted long-history row is a tiered proxy: 2002Q1-2010Q1 is scaled H15 fallback, 2010Q2-2021Q4 is component-pool/WAMEST-bucket backcast, and 2022Q1-2025Q4 is constrained component.",
            "- Report sample splits from `tier2_method_decision_and_splits.md`, especially the weaker pre-2013 and 2010-2021 windows.",
            "- Keep old WAMEST/H15 and no-MMF/RRP rows as sensitivities; do not present them as co-equal defaults.",
            "- Keep the exact Fed bill/FRN extension nondefault until the paper explicitly expands the Tier 1 interest target.",
            "",
            "## Outputs",
            "",
            f"- `{READOUT_CSV.relative_to(ROOT)}`",
            f"- `{READOUT_MD.relative_to(ROOT)}`",
            f"- `{READOUT_JSON.relative_to(ROOT)}`",
        ]
    )
    READOUT_MD.parent.mkdir(parents=True, exist_ok=True)
    READOUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not ESTIMATES_PATH.exists():
        raise FileNotFoundError(f"Missing estimates file: {ESTIMATES_PATH}")
    readout = build_readout()
    _write_csv(READOUT_CSV, readout)
    write_markdown(readout)
    READOUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    READOUT_JSON.write_text(
        json.dumps(
            {
                "estimates_path": str(ESTIMATES_PATH),
                "readout_csv": str(READOUT_CSV),
                "readout_md": str(READOUT_MD),
                "rows_written": len(readout),
                "decision": "promote_regression_mmf_rrp_prop_bank_only_for_long_history_with_caveats",
                "modern_measurement_default": "tdc_tier2_canonical_depository_institution_mmf_rrp_prop_ru_flow",
                "long_history_regression_default": "tdc_tier2_regression_mmf_rrp_prop_bank_only_ru_flow",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"readout_csv": str(READOUT_CSV), "readout_md": str(READOUT_MD), "rows_written": len(readout)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
