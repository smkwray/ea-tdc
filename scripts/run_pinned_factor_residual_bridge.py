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

from ea_tdc.designs.quarterly import build_quarterly_design
from ea_tdc.estimation import _estimate_rows, _write_estimates_csv
from ea_tdc.paths import project_paths
from ea_tdc.residualized_shock import _load_factor_branch
from ea_tdc.utils import utc_now_iso, write_json


ANCHOR_JOB_ID = "tdc_tier2_mmf_rrp_canonical_full_panel"
K_SCREENED = 100
FACTOR_COUNT = 4
CONTROL_POLICY_MODE = "balanced"

MERGE_JOBS = [
    ANCHOR_JOB_ID,
    "tdc_tier2_regression_deposit_anatomy",
    "tdc_tier2_regression_credit_anatomy",
    "tdc_tier2_regression_plumbing_rates",
]

TREATMENTS = {
    "modern_canonical_di_mmf_rrp_short": {
        "treatment_id": "tdc_tier2_canonical_di_mmf_rrp_prop_qoq",
        "residual_id": "other_component_tier2_canonical_di_mmf_rrp_prop_qoq",
        "use_method_tier_controls": False,
    },
    "regression_mmf_rrp_bank_long": {
        "treatment_id": "tdc_tier2_regression_mmf_rrp_prop_bank_only_qoq",
        "residual_id": "other_component_tier2_regression_mmf_rrp_prop_bank_only_qoq",
        "use_method_tier_controls": True,
    },
    "regression_mmf_rrp_di_long": {
        "treatment_id": "tdc_tier2_regression_mmf_rrp_prop_di_np_cu_qoq",
        "residual_id": "other_component_tier2_regression_mmf_rrp_prop_di_np_cu_qoq",
        "use_method_tier_controls": True,
    },
    "regression_no_mmf_bank_long": {
        "treatment_id": "tdc_tier2_regression_bank_only_qoq",
        "residual_id": "other_component_tier2_regression_bank_only_qoq",
        "use_method_tier_controls": True,
    },
    "legacy_h15_bank_sensitivity": {
        "treatment_id": "tdc_tier2_legacy_h15_bank_only_qoq",
        "residual_id": "other_component_tier2_legacy_h15_bank_only_qoq",
        "use_method_tier_controls": True,
    },
    "available_plumbing_bridge": {
        "treatment_id": "tdc_tier2_mmf_rrp_plumbing_adjusted_qoq",
        "residual_id": "other_component_tier2_mmf_rrp_plumbing_adjusted_qoq",
        "use_method_tier_controls": False,
    },
}

METHOD_TIER_CONTROLS = [
    "tier2_regression_bank_row_tier_pre_component_h15_scaled",
]

OUTCOMES = [
    "matched_total_deposits",
    "domestic_nonbank_deposits_qoq",
    "other_component_qoq",
    "other_component_tier2_bank_only_qoq",
    "other_component_tier2_legacy_h15_bank_only_qoq",
    "other_component_tier2_canonical_di_mmf_rrp_prop_qoq",
    "other_component_tier2_mmf_rrp_plumbing_adjusted_qoq",
    "other_component_tier2_regression_bank_only_qoq",
    "other_component_tier2_regression_mmf_rrp_prop_bank_only_qoq",
    "other_component_tier2_regression_mmf_rrp_prop_di_np_cu_qoq",
    "domestic_nonbank_other_component_qoq",
    "domestic_nonbank_other_component_no_row_qoq",
    "domestic_nonbank_other_component_no_toc_no_row_qoq",
    "tdcpass_strict_loan_core_min_qoq",
    "tdcpass_strict_loan_mortgages_qoq",
    "tdcpass_strict_loan_consumer_credit_qoq",
    "bank_consumer_loans_qoq",
    "bank_business_loans_qoq",
    "bank_real_estate_loans_qoq",
    "bank_non_treasury_securities_qoq",
    "large_time_deposits_qoq",
    "M2SL",
    "tga_balance_qoq",
    "reserve_balances_qoq",
    "reserve_balances_net_fed_treasury_qoq",
]

HORIZONS = [0, 1, 2, 4, 8]

BRIDGE_COMPONENTS = [
    ("private_loan_created_deposits_minimum_core", "tdcpass_strict_loan_core_min_qoq", 1000.0),
    ("mortgage_created_deposits", "tdcpass_strict_loan_mortgages_qoq", 1000.0),
    ("consumer_credit_created_deposits", "tdcpass_strict_loan_consumer_credit_qoq", 1000.0),
    ("bank_consumer_loans", "bank_consumer_loans_qoq", 1000.0),
    ("bank_business_loans", "bank_business_loans_qoq", 1000.0),
    ("bank_real_estate_loans", "bank_real_estate_loans_qoq", 1000.0),
    ("bank_non_treasury_securities", "bank_non_treasury_securities_qoq", 1000.0),
    ("large_time_deposits", "large_time_deposits_qoq", 1000.0),
    ("treasury_general_account_cash_drain", "tga_balance_qoq", 1.0),
    ("domestic_nonbank_residual", "domestic_nonbank_other_component_qoq", 1.0),
    ("domestic_nonbank_residual_ex_row", "domestic_nonbank_other_component_no_row_qoq", 1.0),
    ("domestic_nonbank_residual_ex_toc_and_row", "domestic_nonbank_other_component_no_toc_no_row_qoq", 1.0),
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


def _load_manifest(paths, job_id: str) -> dict[str, Any]:
    manifest_path = paths.manifests / f"{job_id}__design_manifest.json"
    if not manifest_path.exists():
        build_quarterly_design(paths, job_id=job_id)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _merge_by_quarter(base_rows: list[dict[str, str]], bundle_paths: list[Path]) -> list[dict[str, str]]:
    by_quarter = {str(row.get("quarter", "")).strip(): dict(row) for row in base_rows}
    for bundle_path in bundle_paths:
        if not bundle_path.exists():
            continue
        for row in _read_csv(bundle_path):
            quarter = str(row.get("quarter", "")).strip()
            if not quarter:
                continue
            merged = by_quarter.setdefault(quarter, {"quarter": quarter})
            for key, value in row.items():
                if key == "quarter":
                    continue
                if str(value).strip() and not str(merged.get(key, "")).strip():
                    merged[key] = value
    return [by_quarter[key] for key in sorted(by_quarter)]


def _row_key(row: dict[str, str]) -> tuple[str, int]:
    return str(row.get("outcome", row.get("outcome_id", ""))), int(row.get("horizon", "0") or 0)


def _normalization_multiplier(outcome_id: str) -> float:
    for _, component_id, multiplier in BRIDGE_COMPONENTS:
        if component_id == outcome_id:
            return multiplier
    return 1.0


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
    bundle_paths = [
        Path(str(_load_manifest(paths, job_id).get("bundle_path", "")))
        for job_id in MERGE_JOBS
    ]
    merged_rows = _merge_by_quarter(factor_rows, bundle_paths)

    all_estimates: list[dict[str, Any]] = []
    for label, treatment_spec in TREATMENTS.items():
        treatment_id = str(treatment_spec["treatment_id"])
        active_controls = (
            [*control_ids[:4], *METHOD_TIER_CONTROLS, *control_ids[4:]]
            if treatment_spec.get("use_method_tier_controls")
            else control_ids[:]
        )
        estimates = _estimate_rows(
            estimator="lp",
            bundle_rows=merged_rows,
            treatment_id=treatment_id,
            control_ids=active_controls,
            outcome_ids=OUTCOMES,
            horizons=HORIZONS,
            response_type="direct_at_h",
            job_id=f"pinned_factor_bridge_{label}",
            instrument_ids=[],
            state_id="",
        )
        for row in estimates:
            row["treatment_label"] = label
            row["pinned_anchor_job_id"] = ANCHOR_JOB_ID
            row["pinned_k_screened"] = K_SCREENED
            row["pinned_control_policy_mode"] = CONTROL_POLICY_MODE
        all_estimates.extend(estimates)

    estimates_path = paths.output / "models" / "tier2_pinned_factor_bridge_estimates.csv"
    _write_estimates_csv(estimates_path, all_estimates)

    h0_by_treatment: dict[str, dict[str, dict[str, Any]]] = {}
    for row in all_estimates:
        if int(row.get("horizon", 0)) != 0:
            continue
        label = str(row.get("treatment_label", ""))
        outcome_id, _ = _row_key(row)
        h0_by_treatment.setdefault(label, {})[outcome_id] = row

    bridge_rows: list[dict[str, Any]] = []
    for label, treatment_spec in TREATMENTS.items():
        residual_row = h0_by_treatment.get(label, {}).get(str(treatment_spec["residual_id"]))
        if residual_row is None:
            residual_row = h0_by_treatment.get(label, {}).get("other_component_qoq")
        residual_coef = float(residual_row["beta"]) if residual_row else 0.0
        for component_label, outcome_id, multiplier in BRIDGE_COMPONENTS:
            row = h0_by_treatment.get(label, {}).get(outcome_id)
            if not row:
                continue
            raw_beta = float(row["beta"])
            normalized = raw_beta * multiplier
            share = abs(normalized / residual_coef) if residual_coef and normalized < 0 else ""
            bridge_rows.append(
                {
                    "treatment_label": label,
                    "component": component_label,
                    "outcome_id": outcome_id,
                    "raw_beta": raw_beta,
                    "normalized_dollars_per_dollar_tdc": normalized,
                    "p_value": row.get("p_value_normal", ""),
                    "n": row.get("n", ""),
                    "residual_reference_beta": residual_coef,
                    "absolute_share_of_negative_residual": share,
                    "notes": "Loan/security/deposit quantity rows are multiplied by 1000 to convert billions-per-million into dollars-per-dollar. Residual/plumbing ratio rows are already in dollars-per-dollar scale.",
                }
            )
    bridge_rows.sort(
        key=lambda row: (
            str(row["treatment_label"]),
            float(row["normalized_dollars_per_dollar_tdc"]),
        )
    )
    bridge_path = paths.reports / "tier2_pinned_factor_residual_bridge.csv"
    _write_csv(bridge_path, bridge_rows)

    markdown_lines = [
        "# Tier 2 Pinned-Factor Residual Bridge",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        f"Common control surface: `{ANCHOR_JOB_ID}`, K=`{K_SCREENED}`, factors=`{factor_count}`, policy=`{CONTROL_POLICY_MODE}`.",
        f"Method-tier controls for long-history rows: `{', '.join(METHOD_TIER_CONTROLS)}`; collinear trailing controls are dropped adaptively.",
        f"Screened candidate count: `{screened_count}`.",
        "",
        "## H=0 Residual Bridge",
        "",
        "Values are normalized to dollars per $1 TDC where possible. Negative rows are candidate contributors to the negative non-TDC residual.",
        "",
    ]
    for label in TREATMENTS:
        markdown_lines.append(f"### {label}")
        for row in [item for item in bridge_rows if item["treatment_label"] == label]:
            value = float(row["normalized_dollars_per_dollar_tdc"])
            if value >= 0:
                continue
            share = row["absolute_share_of_negative_residual"]
            share_text = f", about {float(share):.1%} of residual magnitude" if share != "" else ""
            markdown_lines.append(
                f"- {row['component']}: `{value:.3f}` per $1 TDC (p=`{float(row['p_value']):.3g}`{share_text})."
            )
        markdown_lines.append("")
    markdown_lines.extend(
        [
            "## Outputs",
            "",
            f"- `{estimates_path.relative_to(paths.root)}`",
            f"- `{bridge_path.relative_to(paths.root)}`",
        ]
    )
    md_path = paths.reports / "tier2_pinned_factor_residual_bridge.md"
    md_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    summary_path = paths.manifests / "tier2_pinned_factor_residual_bridge_summary.json"
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
            "method_tier_control_ids_available": METHOD_TIER_CONTROLS,
            "method_tier_control_ids_requested_for_long_history": METHOD_TIER_CONTROLS,
            "estimates_path": str(estimates_path),
            "bridge_path": str(bridge_path),
            "markdown_path": str(md_path),
            "rows_written": len(all_estimates),
            "bridge_rows_written": len(bridge_rows),
        },
    )
    print(json.dumps({"estimates_path": str(estimates_path), "bridge_path": str(bridge_path), "markdown_path": str(md_path), "summary_path": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
