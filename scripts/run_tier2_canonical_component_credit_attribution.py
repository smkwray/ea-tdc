from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
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

TDC_EST_PROCESSED = ROOT.parent / "tdcest" / "data" / "processed"
COMPONENT_CANDIDATE = TDC_EST_PROCESSED / "tier2_interest_component_candidate.csv"
REGRESSION_SERIES = TDC_EST_PROCESSED / "tdc_tier2_regression_series.csv"

OUTCOMES = [
    "matched_total_deposits",
    "domestic_nonbank_deposits_qoq",
    "other_component_tier2_canonical_di_mmf_rrp_prop_qoq",
    "tdcpass_strict_loan_core_min_qoq",
    "tdcpass_strict_loan_mortgages_qoq",
    "tdcpass_strict_loan_consumer_credit_qoq",
    "bank_credit_qoq",
    "bank_consumer_loans_qoq",
    "bank_real_estate_loans_qoq",
    "mortgage_30y",
    "mortgage_30y_dgs10_spread",
]

FOCUS_OUTCOMES = [
    "tdcpass_strict_loan_consumer_credit_qoq",
    "tdcpass_strict_loan_mortgages_qoq",
    "tdcpass_strict_loan_core_min_qoq",
    "matched_total_deposits",
    "other_component_tier2_canonical_di_mmf_rrp_prop_qoq",
]

HORIZONS = [0, 1, 2, 4, 8]

REGRESSION_COMPONENTS = {
    "tier2_component_base_bank_only_qoq": {
        "source_column": "tdc_base_bank_only_ru_flow",
        "label": "Base bank-only RU flow",
        "group": "base_flow",
        "sign": 1.0,
    },
    "tier2_component_base_broad_depository_np_cu_qoq": {
        "source_column": "tdc_base_broad_depository_np_cu_ru_flow",
        "label": "Base broad depository RU flow",
        "group": "base_flow",
        "sign": 1.0,
    },
    "tier2_component_bank_interest_correction_qoq": {
        "source_column": "bank_tier2_regression_interest_proxy",
        "label": "Bank Treasury-interest correction",
        "group": "sector_interest_correction",
        "sign": -1.0,
    },
    "tier2_component_row_interest_correction_qoq": {
        "source_column": "row_tier2_regression_interest_proxy",
        "label": "ROW Treasury-interest correction",
        "group": "sector_interest_correction",
        "sign": -1.0,
    },
    "tier2_component_credit_union_interest_correction_qoq": {
        "source_column": "credit_union_tier2_regression_interest_proxy",
        "label": "Credit-union Treasury-interest correction",
        "group": "sector_interest_correction",
        "sign": -1.0,
    },
    "tier2_component_di_interest_correction_qoq": {
        "source_column": "di_tier2_regression_interest_proxy",
        "label": "Total DI Treasury-interest correction",
        "group": "sector_interest_correction",
        "sign": -1.0,
    },
    "tier2_component_mmf_rrp_prop_adjustment_qoq": {
        "source_column": "mmf_rrp_adjustment_prop",
        "label": "MMF/RRP proportional adjustment",
        "group": "plumbing_adjustment",
        "sign": 1.0,
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


def _float(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    return float(text)


def _quarter_from_date(date_text: str) -> str:
    year = int(date_text[:4])
    month = int(date_text[5:7])
    return f"{year}Q{((month - 1) // 3) + 1}"


def _load_manifest(paths, job_id: str) -> dict[str, Any]:
    manifest_path = paths.manifests / f"{job_id}__design_manifest.json"
    if not manifest_path.exists():
        build_quarterly_design(paths, job_id=job_id)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _component_id(*parts: str) -> str:
    return "tier2_component_" + "_".join(parts) + "_tdc_contribution_qoq"


def _add_value(row: dict[str, str], key: str, value: float) -> None:
    current = _float(row.get(key, "")) or 0.0
    row[key] = str(current + value)


def _register_component(
    metadata: dict[str, dict[str, Any]],
    component_id: str,
    *,
    label: str,
    group: str,
    source: str,
    signed_as: str,
) -> None:
    metadata.setdefault(
        component_id,
        {
            "component_id": component_id,
            "label": label,
            "group": group,
            "source": source,
            "signed_as": signed_as,
            "source_start_quarter": "",
            "source_end_quarter": "",
            "source_nonmissing_quarters": 0,
        },
    )


def _note_component_observation(metadata: dict[str, dict[str, Any]], component_id: str, quarter: str) -> None:
    spec = metadata[component_id]
    spec["source_nonmissing_quarters"] = int(spec["source_nonmissing_quarters"]) + 1
    if not spec["source_start_quarter"] or quarter < spec["source_start_quarter"]:
        spec["source_start_quarter"] = quarter
    if not spec["source_end_quarter"] or quarter > spec["source_end_quarter"]:
        spec["source_end_quarter"] = quarter


def _finalize_metadata(
    panel: dict[str, dict[str, str]],
    metadata: dict[str, dict[str, Any]],
) -> None:
    for component_id, spec in metadata.items():
        values: list[float] = []
        quarters: list[str] = []
        for quarter in sorted(panel):
            value = _float(panel[quarter].get(component_id, ""))
            if value is None:
                continue
            quarters.append(quarter)
            values.append(value)
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
        spec["source_start_quarter"] = quarters[0]
        spec["source_end_quarter"] = quarters[-1]
        spec["source_nonmissing_quarters"] = len(values)
        spec["source_mean_mil"] = mean
        spec["source_sd_mil"] = variance ** 0.5
        spec["source_min_mil"] = min(values)
        spec["source_max_mil"] = max(values)


def _load_component_panel() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    if not COMPONENT_CANDIDATE.exists():
        raise FileNotFoundError(f"Missing upstream component candidate file: {COMPONENT_CANDIDATE}")
    if not REGRESSION_SERIES.exists():
        raise FileNotFoundError(f"Missing upstream Tier 2 regression series file: {REGRESSION_SERIES}")

    panel: dict[str, dict[str, str]] = defaultdict(dict)
    metadata: dict[str, dict[str, Any]] = {}

    for row in _read_csv(REGRESSION_SERIES):
        quarter = _quarter_from_date(str(row["date"]))
        for component_id, spec in REGRESSION_COMPONENTS.items():
            value = _float(row.get(str(spec["source_column"]), ""))
            if value is None:
                continue
            signed_value = float(spec["sign"]) * value
            panel[quarter][component_id] = str(signed_value)
            _register_component(
                metadata,
                component_id,
                label=str(spec["label"]),
                group=str(spec["group"]),
                source="tdcest:data/processed/tdc_tier2_regression_series.csv",
                signed_as="TDC contribution; interest corrections are multiplied by -1, base/plumbing by +1",
            )
            _note_component_observation(metadata, component_id, quarter)

    detail_total = _component_id("all", "interest_correction")
    _register_component(
        metadata,
        detail_total,
        label="Detailed interest-correction total",
        group="detailed_interest_correction",
        source="tdcest:data/processed/tier2_interest_component_candidate.csv",
        signed_as="TDC contribution; candidate interest components are multiplied by -1",
    )
    for row in _read_csv(COMPONENT_CANDIDATE):
        quarter = _quarter_from_date(str(row["date"]))
        sector = str(row["sector_group"]).strip()
        component_key = str(row["component_key"]).strip()
        value = _float(row.get("component_anchored_interest_mil", ""))
        if value is None:
            continue
        signed_value = -value
        sector_component = _component_id(sector, component_key)
        sector_total = _component_id(sector, "interest_correction")
        component_total = _component_id("all", component_key)

        for component_id, label, group in [
            (sector_component, f"{sector} {component_key}", "detailed_sector_component"),
            (sector_total, f"{sector} detailed interest-correction total", "detailed_sector_total"),
            (component_total, f"All-sector {component_key}", "detailed_component_total"),
            (detail_total, "Detailed interest-correction total", "detailed_interest_correction"),
        ]:
            _register_component(
                metadata,
                component_id,
                label=label,
                group=group,
                source="tdcest:data/processed/tier2_interest_component_candidate.csv",
                signed_as="TDC contribution; candidate interest components are multiplied by -1",
            )
            _add_value(panel[quarter], component_id, signed_value)
            _note_component_observation(metadata, component_id, quarter)

    finalized_panel = dict(panel)
    _finalize_metadata(finalized_panel, metadata)
    return finalized_panel, metadata


def _augment_rows(
    rows: list[dict[str, str]],
    component_panel: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    augmented: list[dict[str, str]] = []
    for row in rows:
        merged = dict(row)
        quarter = str(row.get("quarter", "")).strip()
        if quarter in component_panel:
            merged.update(component_panel[quarter])
        augmented.append(merged)
    return augmented


def _normalization_multiplier(outcome_id: str) -> float:
    if outcome_id.startswith("tdcpass_strict_loan_"):
        return 1000.0
    if outcome_id in {
        "bank_credit_qoq",
        "bank_consumer_loans_qoq",
        "bank_real_estate_loans_qoq",
    }:
        return 1000.0
    return 1.0


def _estimate_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("component_id", row.get("treatment_id", ""))),
        str(row.get("outcome", row.get("outcome_id", ""))),
        int(row.get("horizon", 0)),
    )


def _ranking_rows(
    estimates: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for estimate in estimates:
        outcome_id = str(estimate.get("outcome", estimate.get("outcome_id", "")))
        horizon = int(estimate.get("horizon", 0))
        component_id = str(estimate.get("component_id", estimate.get("treatment_id", "")))
        beta = float(estimate["beta"])
        normalized_beta = beta * _normalization_multiplier(outcome_id)
        spec = metadata.get(component_id, {})
        source_sd_mil = float(spec.get("source_sd_mil", 0.0) or 0.0)
        typical_effect_usd_bn = normalized_beta * source_sd_mil / 1000.0
        rows.append(
            {
                "component_id": component_id,
                "component_label": spec.get("label", component_id),
                "component_group": spec.get("group", ""),
                "outcome": outcome_id,
                "horizon": horizon,
                "beta": beta,
                "normalized_beta_dollars_per_dollar_component": normalized_beta,
                "abs_normalized_beta": abs(normalized_beta),
                "source_sd_mil": source_sd_mil,
                "typical_effect_per_1sd_component_usd_bn": typical_effect_usd_bn,
                "abs_typical_effect_per_1sd_component_usd_bn": abs(typical_effect_usd_bn),
                "se": estimate.get("se", ""),
                "p_value": estimate.get("p_value_normal", ""),
                "n": estimate.get("n", ""),
                "controls_used_count": len(str(estimate.get("control_ids_used", "")).split(";"))
                if str(estimate.get("control_ids_used", "")).strip()
                else 0,
                "dropped_control_ids": estimate.get("dropped_control_ids", ""),
                "source_start_quarter": spec.get("source_start_quarter", ""),
                "source_end_quarter": spec.get("source_end_quarter", ""),
                "source_nonmissing_quarters": spec.get("source_nonmissing_quarters", ""),
                "signed_as": spec.get("signed_as", ""),
            }
        )

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["outcome"]), int(row["horizon"]))].append(row)
    for group_rows in grouped.values():
        group_rows.sort(key=lambda row: float(row["abs_normalized_beta"]), reverse=True)
        for rank, row in enumerate(group_rows, start=1):
            row["rank_abs_within_outcome_horizon"] = rank
        group_rows.sort(key=lambda row: float(row["abs_typical_effect_per_1sd_component_usd_bn"]), reverse=True)
        for rank, row in enumerate(group_rows, start=1):
            row["rank_typical_within_outcome_horizon"] = rank
    rows.sort(key=lambda row: (str(row["outcome"]), int(row["horizon"]), int(row["rank_typical_within_outcome_horizon"])))
    return rows


def _md_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        cells: list[str] = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                if abs(value) >= 10:
                    cells.append(f"{value:.2f}")
                else:
                    cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _write_markdown(path: Path, ranking_rows: list[dict[str, Any]], metadata_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Tier 2 Canonical Component Credit Attribution",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        "Component treatments are signed as contributions to TDC: base/plumbing pieces enter positive, while Treasury-interest correction pieces enter negative because they reduce the Tier 2 TDC measure.",
        "Loan and bank-balance-sheet quantity outcomes are normalized to dollars per $1 component by multiplying the raw coefficient by 1000. Rankings use the estimated effect of a one-standard-deviation move in each component, which avoids over-ranking tiny components on per-dollar coefficients alone.",
        "",
        "## H=0 Focus Rankings",
        "",
    ]
    for outcome in FOCUS_OUTCOMES:
        focus = [
            row
            for row in ranking_rows
            if row["outcome"] == outcome and int(row["horizon"]) == 0
        ]
        focus.sort(key=lambda row: int(row["rank_typical_within_outcome_horizon"]))
        focus = focus[:8]
        if not focus:
            continue
        lines.extend(
            [
                f"### {outcome}",
                "",
                *_md_table(
                    focus,
                    [
                        "rank_typical_within_outcome_horizon",
                        "component_label",
                        "component_group",
                        "typical_effect_per_1sd_component_usd_bn",
                        "normalized_beta_dollars_per_dollar_component",
                        "p_value",
                        "n",
                        "source_start_quarter",
                    ],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Component Coverage",
            "",
            *_md_table(
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
            ),
            "",
        ]
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

    augmented_bundle_path = paths.bundles / "designs" / "tier2_canonical_component_credit_attribution__quarterly_bundle.csv"
    _write_csv(augmented_bundle_path, augmented_rows)

    all_estimates: list[dict[str, Any]] = []
    component_ids = sorted(metadata)
    for component_id in component_ids:
        estimates = _estimate_rows(
            estimator="lp",
            bundle_rows=augmented_rows,
            treatment_id=component_id,
            control_ids=control_ids,
            outcome_ids=OUTCOMES,
            horizons=HORIZONS,
            response_type="direct_at_h",
            job_id=f"tier2_component_credit_attribution__{component_id}",
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

    estimates_path = paths.output / "models" / "tier2_canonical_component_credit_attribution_estimates.csv"
    _write_estimates_csv(estimates_path, all_estimates)

    ranked = _ranking_rows(all_estimates, metadata)
    ranking_path = paths.reports / "tier2_canonical_component_credit_attribution.csv"
    _write_csv(ranking_path, ranked)

    metadata_rows = list(metadata.values())
    metadata_path = paths.reports / "tier2_canonical_component_credit_components.csv"
    _write_csv(metadata_path, metadata_rows)

    md_path = paths.reports / "tier2_canonical_component_credit_attribution.md"
    _write_markdown(md_path, ranked, metadata_rows)

    summary_path = paths.manifests / "tier2_canonical_component_credit_attribution_summary.json"
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
            "component_candidate_path": str(COMPONENT_CANDIDATE),
            "regression_series_path": str(REGRESSION_SERIES),
            "augmented_bundle_path": str(augmented_bundle_path),
            "estimates_path": str(estimates_path),
            "ranking_path": str(ranking_path),
            "metadata_path": str(metadata_path),
            "markdown_path": str(md_path),
            "components_estimated": len(component_ids),
            "estimate_rows_written": len(all_estimates),
            "ranking_rows_written": len(ranked),
            "focus_outcomes": FOCUS_OUTCOMES,
            "horizons": HORIZONS,
        },
    )
    print(
        json.dumps(
            {
                "augmented_bundle_path": str(augmented_bundle_path),
                "estimates_path": str(estimates_path),
                "ranking_path": str(ranking_path),
                "metadata_path": str(metadata_path),
                "markdown_path": str(md_path),
                "summary_path": str(summary_path),
                "components_estimated": len(component_ids),
                "estimate_rows_written": len(all_estimates),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
