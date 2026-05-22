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

from ea_tdc.estimation import _estimate_rows, _write_estimates_csv
from ea_tdc.paths import project_paths
from ea_tdc.utils import utc_now_iso, write_json
from run_pinned_factor_residual_bridge import (
    ANCHOR_JOB_ID,
    CONTROL_POLICY_MODE,
    FACTOR_COUNT,
    K_SCREENED,
    MERGE_JOBS,
    METHOD_TIER_CONTROLS,
    TREATMENTS,
    _load_manifest,
    _merge_by_quarter,
)
from ea_tdc.designs.quarterly import build_quarterly_design
from ea_tdc.residualized_shock import _load_factor_branch


OUTCOMES = [
    "matched_total_deposits",
    "tga_balance_qoq",
    "other_component_tier2_canonical_di_mmf_rrp_prop_qoq",
    "other_component_tier2_mmf_rrp_plumbing_adjusted_qoq",
    "other_component_tier2_regression_bank_only_qoq",
    "other_component_tier2_regression_mmf_rrp_prop_bank_only_qoq",
    "other_component_tier2_regression_mmf_rrp_prop_di_np_cu_qoq",
    "other_component_tier2_legacy_h15_bank_only_qoq",
    "domestic_nonbank_other_component_qoq",
    "domestic_nonbank_other_component_no_row_qoq",
    "tdcpass_strict_loan_core_min_qoq",
    "tdcpass_strict_loan_consumer_credit_qoq",
    "tdcpass_strict_loan_mortgages_qoq",
    "reserve_balances_qoq",
    "reserve_balances_net_fed_treasury_qoq",
]

SAMPLE_WINDOWS = [
    {
        "sample_label": "full_available",
        "description": "All available quarters after each treatment/control availability filter.",
    },
    {
        "sample_label": "pre_component_2002q1_2010q1",
        "start_quarter": "2002Q1",
        "end_quarter": "2010Q1",
        "description": "Scaled H15/WAMEST fallback segment.",
    },
    {
        "sample_label": "component_pool_2010q2_2021q4",
        "start_quarter": "2010Q2",
        "end_quarter": "2021Q4",
        "description": "Component-pool/WAMEST-bucket backcast segment.",
    },
    {
        "sample_label": "onrrp_full_quarter_2013q4_plus",
        "start_quarter": "2013Q4",
        "description": "Full-quarter ON RRP era; MMF/RRP plumbing is structurally available.",
    },
    {
        "sample_label": "constrained_2022q1_plus",
        "start_quarter": "2022Q1",
        "description": "Constrained component measurement era and true modern canonical overlap.",
    },
    {
        "sample_label": "exclude_2019q1_2021q4",
        "exclude_start_quarter": "2019Q1",
        "exclude_end_quarter": "2021Q4",
        "description": "Full panel excluding the suspicious 2019-2021 component-pool transition interval.",
    },
]

METHOD_DECISIONS = {
    "modern_canonical_di_mmf_rrp_short": {
        "role": "modern measurement default",
        "coverage": "2022Q1-2025Q4",
        "recommendation": "Use for current-period accounting and charts; too short for standalone long-history inference.",
    },
    "regression_mmf_rrp_bank_long": {
        "role": "preferred long-history regression candidate",
        "coverage": "2002Q1-2025Q4",
        "recommendation": "Use for long-history regressions with method-tier controls and sample-split disclosure.",
    },
    "regression_mmf_rrp_di_long": {
        "role": "matched-perimeter long-history companion",
        "coverage": "2002Q1-2025Q4",
        "recommendation": "Use to compare against the DI modern default; bank-only remains cleaner for the long sample.",
    },
    "regression_no_mmf_bank_long": {
        "role": "no-MMF/RRP sensitivity",
        "coverage": "2002Q1-2025Q4",
        "recommendation": "Keep as sensitivity showing the contribution of the MMF/RRP plumbing add-back.",
    },
    "legacy_h15_bank_sensitivity": {
        "role": "legacy WAMEST/H15 sensitivity",
        "coverage": "2002Q1-2025Q4",
        "recommendation": "Demote to appendix/sensitivity; it is not the preferred holder-interest object.",
    },
    "available_plumbing_bridge": {
        "role": "bridge-only convenience proxy",
        "coverage": "2013Q4-2025Q4 approximately",
        "recommendation": "Use only as a bridge check; it is not the constrained component canonical row.",
    },
}


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


def _quarter_key(quarter: str) -> tuple[int, int]:
    text = str(quarter).strip()
    return int(text[:4]), int(text[-1])


def _in_window(row: dict[str, str], window: dict[str, str]) -> bool:
    quarter = str(row.get("quarter", "")).strip()
    if not quarter:
        return False
    key = _quarter_key(quarter)
    start = str(window.get("start_quarter", "")).strip()
    end = str(window.get("end_quarter", "")).strip()
    exclude_start = str(window.get("exclude_start_quarter", "")).strip()
    exclude_end = str(window.get("exclude_end_quarter", "")).strip()
    if start and key < _quarter_key(start):
        return False
    if end and key > _quarter_key(end):
        return False
    if exclude_start and exclude_end and _quarter_key(exclude_start) <= key <= _quarter_key(exclude_end):
        return False
    return True


def _active_controls(control_ids: list[str], treatment_spec: dict[str, Any]) -> list[str]:
    if not treatment_spec.get("use_method_tier_controls"):
        return control_ids[:]
    return [*control_ids[:4], *METHOD_TIER_CONTROLS, *control_ids[4:]]


def _estimate_window(
    *,
    rows: list[dict[str, str]],
    control_ids: list[str],
    treatment_label: str,
    treatment_spec: dict[str, Any],
    sample_label: str,
) -> list[dict[str, Any]]:
    estimates = _estimate_rows(
        estimator="lp",
        bundle_rows=rows,
        treatment_id=str(treatment_spec["treatment_id"]),
        control_ids=_active_controls(control_ids, treatment_spec),
        outcome_ids=OUTCOMES,
        horizons=[0],
        response_type="direct_at_h",
        job_id=f"tier2_method_split_{treatment_label}_{sample_label}",
        instrument_ids=[],
        state_id="",
    )
    for estimate in estimates:
        estimate["treatment_label"] = treatment_label
        estimate["sample_label"] = sample_label
        estimate["pinned_anchor_job_id"] = ANCHOR_JOB_ID
        estimate["pinned_k_screened"] = K_SCREENED
        estimate["pinned_control_policy_mode"] = CONTROL_POLICY_MODE
    return estimates


def _estimate_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    keyed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if int(row.get("horizon", 0)) != 0:
            continue
        keyed[
            (
                str(row.get("treatment_label", "")),
                str(row.get("sample_label", "")),
                str(row.get("outcome", row.get("outcome_id", ""))),
            )
        ] = row
    return keyed


def _float(row: dict[str, Any] | None, key: str) -> float | str:
    if row is None:
        return ""
    value = str(row.get(key, "")).strip()
    if not value:
        return ""
    return float(value)


def _decision_rows(estimates: dict[tuple[str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, treatment_spec in TREATMENTS.items():
        decision = METHOD_DECISIONS[label]
        deposit = estimates.get((label, "full_available", "matched_total_deposits"))
        tga = estimates.get((label, "full_available", "tga_balance_qoq"))
        core = estimates.get((label, "full_available", "tdcpass_strict_loan_core_min_qoq"))
        consumer = estimates.get((label, "full_available", "tdcpass_strict_loan_consumer_credit_qoq"))
        mortgage = estimates.get((label, "full_available", "tdcpass_strict_loan_mortgages_qoq"))
        rows.append(
            {
                "treatment_label": label,
                "role": decision["role"],
                "treatment_id": treatment_spec["treatment_id"],
                "coverage": decision["coverage"],
                "uses_method_tier_controls": bool(treatment_spec.get("use_method_tier_controls")),
                "h0_deposits_beta": _float(deposit, "beta"),
                "h0_deposits_p": _float(deposit, "p_value_normal"),
                "h0_deposits_n": deposit.get("n", "") if deposit else "",
                "h0_tga_beta": _float(tga, "beta"),
                "h0_loan_core_beta_per_1_tdc": _float(core, "beta") * 1000.0 if core else "",
                "h0_consumer_credit_beta_per_1_tdc": _float(consumer, "beta") * 1000.0 if consumer else "",
                "h0_mortgage_beta_per_1_tdc": _float(mortgage, "beta") * 1000.0 if mortgage else "",
                "recommendation": decision["recommendation"],
            }
        )
    return rows


def _split_rows(estimates: dict[tuple[str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in TREATMENTS:
        for window in SAMPLE_WINDOWS:
            sample = str(window["sample_label"])
            deposit = estimates.get((label, sample, "matched_total_deposits"))
            if not deposit:
                continue
            residual_id = str(TREATMENTS[label]["residual_id"])
            residual = estimates.get((label, sample, residual_id))
            tga = estimates.get((label, sample, "tga_balance_qoq"))
            core = estimates.get((label, sample, "tdcpass_strict_loan_core_min_qoq"))
            consumer = estimates.get((label, sample, "tdcpass_strict_loan_consumer_credit_qoq"))
            mortgage = estimates.get((label, sample, "tdcpass_strict_loan_mortgages_qoq"))
            rows.append(
                {
                    "treatment_label": label,
                    "sample_label": sample,
                    "sample_description": window["description"],
                    "h0_deposits_beta": _float(deposit, "beta"),
                    "h0_deposits_p": _float(deposit, "p_value_normal"),
                    "h0_deposits_n": deposit.get("n", ""),
                    "h0_residual_beta": _float(residual, "beta"),
                    "h0_tga_beta": _float(tga, "beta"),
                    "h0_loan_core_beta_per_1_tdc": _float(core, "beta") * 1000.0 if core else "",
                    "h0_consumer_credit_beta_per_1_tdc": _float(consumer, "beta") * 1000.0 if consumer else "",
                    "h0_mortgage_beta_per_1_tdc": _float(mortgage, "beta") * 1000.0 if mortgage else "",
                    "control_ids_used": deposit.get("control_ids_used", ""),
                    "warning_flags": deposit.get("warning_flags", ""),
                    "dropped_control_ids": deposit.get("dropped_control_ids", ""),
                }
            )
    return rows


def _md_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                if "p" in column:
                    cells.append(f"{value:.3g}")
                else:
                    cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


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

    all_estimates: list[dict[str, Any]] = []
    for window in SAMPLE_WINDOWS:
        sample_rows = [row for row in merged_rows if _in_window(row, window)]
        for label, spec in TREATMENTS.items():
            all_estimates.extend(
                _estimate_window(
                    rows=sample_rows,
                    control_ids=control_ids,
                    treatment_label=label,
                    treatment_spec=spec,
                    sample_label=str(window["sample_label"]),
                )
            )

    estimates_path = paths.output / "models" / "tier2_method_decision_sample_split_estimates.csv"
    _write_estimates_csv(estimates_path, all_estimates)

    keyed = _estimate_by_key(all_estimates)
    decision_rows = _decision_rows(keyed)
    split_rows = _split_rows(keyed)

    decision_path = paths.reports / "tier2_method_decision_table.csv"
    split_path = paths.reports / "tier2_sample_split_diagnostics.csv"
    _write_csv(decision_path, decision_rows)
    _write_csv(split_path, split_rows)

    md_lines = [
        "# Tier 2 Method Decision And Sample Splits",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        f"Common control surface: `{ANCHOR_JOB_ID}`, K=`{K_SCREENED}`, factors=`{factor_count}`, policy=`{CONTROL_POLICY_MODE}`.",
        f"Method-tier controls for long-history rows: `{', '.join(METHOD_TIER_CONTROLS)}`; collinear trailing controls are dropped adaptively.",
        f"Screened candidate count: `{screened_count}`.",
        "",
        "## Method Decision Table",
        "",
        *_md_table(
            decision_rows,
            [
                "treatment_label",
                "role",
                "coverage",
                "h0_deposits_beta",
                "h0_deposits_p",
                "h0_deposits_n",
                "h0_tga_beta",
                "h0_loan_core_beta_per_1_tdc",
                "recommendation",
            ],
        ),
        "",
        "## Sample Split Diagnostics",
        "",
        "These are h=0 direct responses. Loan rows are normalized to dollars per $1 TDC by multiplying the FRED-style billions-per-million estimates by 1000.",
        "",
    ]
    for label in TREATMENTS:
        label_rows = [row for row in split_rows if row["treatment_label"] == label]
        if not label_rows:
            continue
        md_lines.append(f"### {label}")
        md_lines.extend(
            _md_table(
                label_rows,
                [
                    "sample_label",
                    "h0_deposits_beta",
                    "h0_deposits_p",
                    "h0_deposits_n",
                    "h0_residual_beta",
                    "h0_tga_beta",
                    "h0_loan_core_beta_per_1_tdc",
                    "warning_flags",
                    "dropped_control_ids",
                ],
            )
        )
        md_lines.append("")
    md_lines.extend(
        [
            "## Outputs",
            "",
            f"- `{estimates_path.relative_to(paths.root)}`",
            f"- `{decision_path.relative_to(paths.root)}`",
            f"- `{split_path.relative_to(paths.root)}`",
        ]
    )
    md_path = paths.reports / "tier2_method_decision_and_splits.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    summary_path = paths.manifests / "tier2_method_decision_and_splits_summary.json"
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
            "method_tier_control_ids_requested_for_long_history": METHOD_TIER_CONTROLS,
            "sample_windows": SAMPLE_WINDOWS,
            "estimates_path": str(estimates_path),
            "decision_path": str(decision_path),
            "split_path": str(split_path),
            "markdown_path": str(md_path),
            "rows_written": len(all_estimates),
            "decision_rows_written": len(decision_rows),
            "split_rows_written": len(split_rows),
        },
    )
    print(
        json.dumps(
            {
                "estimates_path": str(estimates_path),
                "decision_path": str(decision_path),
                "split_path": str(split_path),
                "markdown_path": str(md_path),
                "summary_path": str(summary_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
