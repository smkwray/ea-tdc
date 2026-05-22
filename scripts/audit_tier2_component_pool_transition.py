from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TDC_EST = ROOT.parent / "tdcest"

REGRESSION_SERIES = TDC_EST / "data" / "processed" / "tdc_tier2_regression_series.csv"
COMPONENT_CANDIDATE = TDC_EST / "data" / "processed" / "tier2_interest_component_candidate.csv"
SOURCE_CONSTRAINTS = TDC_EST / "data" / "processed" / "tier2_interest_source_constraints.csv"

START_DATE = "2018-12-31"
END_DATE = "2022-03-31"
DETAIL_DATES = {"2019-12-31", "2020-03-31", "2022-03-31"}


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


def _float(value: str) -> float:
    text = str(value or "").strip()
    return float(text) if text else 0.0


def _date_in_window(date: str) -> bool:
    return START_DATE <= date <= END_DATE


def _aggregate_candidate() -> dict[str, dict[str, float]]:
    aggregates: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in _read_csv(COMPONENT_CANDIDATE):
        date = str(row.get("date", "")).strip()
        if not _date_in_window(date):
            continue
        sector = str(row.get("sector_group", "")).strip()
        component = str(row.get("component_key", "")).strip()
        prefix = f"candidate_{sector}"
        aggregates[date][f"{prefix}_anchored_interest_mil"] += _float(row.get("component_anchored_interest_mil", ""))
        aggregates[date][f"{prefix}_current_raw_proxy_mil"] += _float(row.get("current_raw_proxy_mil", ""))
        aggregates[date][f"{prefix}_{component}_anchored_interest_mil"] += _float(row.get("component_anchored_interest_mil", ""))
    for date, values in aggregates.items():
        values["candidate_bank_row_anchored_interest_mil"] = (
            values.get("candidate_bank_anchored_interest_mil", 0.0)
            + values.get("candidate_row_anchored_interest_mil", 0.0)
        )
        values["candidate_di_anchored_interest_mil"] = (
            values.get("candidate_bank_row_anchored_interest_mil", 0.0)
            + values.get("candidate_credit_union_anchored_interest_mil", 0.0)
        )
    return aggregates


def _constraint_summary() -> dict[str, dict[str, str]]:
    by_date: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in _read_csv(SOURCE_CONSTRAINTS):
        date = str(row.get("date", "")).strip()
        if not _date_in_window(date):
            continue
        family = str(row.get("source_family", "")).strip()
        status = str(row.get("constraint_status", "")).strip()
        basis = str(row.get("constraint_basis", "")).strip()
        sector = str(row.get("sector_key", "")).strip()
        label = f"{sector}:{family}:{status}:{basis}"
        if label not in by_date[date]["source_constraints"]:
            by_date[date]["source_constraints"].append(label)
    return {
        date: {key: "; ".join(values) for key, values in fields.items()}
        for date, fields in by_date.items()
    }


def _transition_rows() -> list[dict[str, Any]]:
    candidate = _aggregate_candidate()
    constraints = _constraint_summary()
    rows: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for row in _read_csv(REGRESSION_SERIES):
        date = str(row.get("date", "")).strip()
        if not _date_in_window(date):
            continue
        record: dict[str, Any] = {
            "date": date,
            "bank_method_tier": row.get("bank_method_tier", ""),
            "row_method_tier": row.get("row_method_tier", ""),
            "credit_union_method_tier": row.get("credit_union_method_tier", ""),
            "bank_interest_mil": _float(row.get("bank_tier2_regression_interest_proxy", "")),
            "row_interest_mil": _float(row.get("row_tier2_regression_interest_proxy", "")),
            "credit_union_interest_mil": _float(row.get("credit_union_tier2_regression_interest_proxy", "")),
            "bank_row_interest_mil": _float(row.get("bank_row_tier2_regression_interest_proxy", "")),
            "di_interest_mil": _float(row.get("di_tier2_regression_interest_proxy", "")),
            "mmf_rrp_adjustment_prop_mil": _float(row.get("mmf_rrp_adjustment_prop", "")),
            "tdc_regression_mmf_rrp_bank_mil": _float(row.get("tdc_tier2_regression_mmf_rrp_prop_bank_only_ru_flow", "")),
            "tdc_regression_mmf_rrp_di_mil": _float(row.get("tdc_tier2_regression_mmf_rrp_prop_depository_institution_np_cu_ru_flow", "")),
            **candidate.get(date, {}),
            **constraints.get(date, {}),
        }
        for field in [
            "bank_interest_mil",
            "row_interest_mil",
            "credit_union_interest_mil",
            "bank_row_interest_mil",
            "di_interest_mil",
            "candidate_bank_row_anchored_interest_mil",
            "candidate_di_anchored_interest_mil",
        ]:
            prior = float(previous.get(field, 0.0)) if previous else 0.0
            current = float(record.get(field, 0.0))
            record[f"{field}_qoq_change_mil"] = current - prior if previous else ""
        candidate_bank_row = float(record.get("candidate_bank_row_anchored_interest_mil", 0.0))
        regression_bank_row = float(record.get("bank_row_interest_mil", 0.0))
        record["candidate_to_regression_bank_row_ratio"] = (
            candidate_bank_row / regression_bank_row if regression_bank_row else ""
        )
        rows.append(record)
        previous = record
    return rows


def _event_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "bank_interest_mil_qoq_change_mil",
        "row_interest_mil_qoq_change_mil",
        "credit_union_interest_mil_qoq_change_mil",
        "bank_row_interest_mil_qoq_change_mil",
        "di_interest_mil_qoq_change_mil",
    ]
    events: list[dict[str, Any]] = []
    for row in rows:
        for field in fields:
            value = row.get(field, "")
            if value == "":
                continue
            events.append(
                {
                    "date": row["date"],
                    "series": field.removesuffix("_qoq_change_mil"),
                    "qoq_change_mil": value,
                    "abs_qoq_change_mil": abs(float(value)),
                    "bank_method_tier": row.get("bank_method_tier", ""),
                    "row_method_tier": row.get("row_method_tier", ""),
                    "credit_union_method_tier": row.get("credit_union_method_tier", ""),
                    "source_constraints": row.get("source_constraints", ""),
                }
            )
    events.sort(key=lambda item: float(item["abs_qoq_change_mil"]), reverse=True)
    return events


def _component_detail_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(COMPONENT_CANDIDATE):
        date = str(row.get("date", "")).strip()
        if date not in DETAIL_DATES:
            continue
        sector = str(row.get("sector_group", "")).strip()
        if sector not in {"bank", "row", "credit_union"}:
            continue
        rows.append(
            {
                "date": date,
                "sector_group": sector,
                "component_key": row.get("component_key", ""),
                "allocator_basis": row.get("allocator_basis", ""),
                "selected_raw_weight_mil": _float(row.get("selected_raw_weight_mil", "")),
                "denominator_raw_weight_mil": _float(row.get("denominator_raw_weight_mil", "")),
                "component_anchored_interest_mil": _float(row.get("component_anchored_interest_mil", "")),
                "current_raw_proxy_mil": _float(row.get("current_raw_proxy_mil", "")),
                "candidate_default_status": row.get("candidate_default_status", ""),
            }
        )
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
                cells.append(f"{value:.1f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main() -> int:
    rows = _transition_rows()
    events = _event_rows(rows)
    component_details = _component_detail_rows()
    reports = ROOT / "output" / "reports"
    manifests = ROOT / "output" / "manifests"
    audit_path = reports / "tier2_component_pool_transition_audit.csv"
    events_path = reports / "tier2_component_pool_transition_events.csv"
    details_path = reports / "tier2_component_pool_transition_component_details.csv"
    _write_csv(audit_path, rows)
    _write_csv(events_path, events)
    _write_csv(details_path, component_details)

    top_events = events[:12]
    md_lines = [
        "# Tier 2 Component-Pool Transition Audit",
        "",
        f"Window: `{START_DATE}` through `{END_DATE}`.",
        "",
        "This checks the exact interval implicated by the sample splits. Values are millions of dollars.",
        "",
        "## Quarterly Series",
        "",
        *_md_table(
            rows,
            [
                "date",
                "bank_method_tier",
                "bank_interest_mil",
                "row_interest_mil",
                "credit_union_interest_mil",
                "bank_row_interest_mil",
                "di_interest_mil",
                "mmf_rrp_adjustment_prop_mil",
                "candidate_to_regression_bank_row_ratio",
            ],
        ),
        "",
        "## Largest Quarter-To-Quarter Moves",
        "",
        *_md_table(
            top_events,
            [
                "date",
                "series",
                "qoq_change_mil",
                "abs_qoq_change_mil",
                "bank_method_tier",
                "row_method_tier",
                "credit_union_method_tier",
            ],
        ),
        "",
        "## Component Detail For Key Dates",
        "",
        *_md_table(
            component_details,
            [
                "date",
                "sector_group",
                "component_key",
                "selected_raw_weight_mil",
                "denominator_raw_weight_mil",
                "component_anchored_interest_mil",
                "current_raw_proxy_mil",
            ],
        ),
        "",
        "## Read",
        "",
        "- The regression-interest series stays in `component_pool_wamest_bucket_backcast` through 2021Q4, then switches to constrained components in 2022Q1.",
        "- The former 2020Q1 ROW collapse was traced to a source-constraint unit bug upstream: `level_mil` was divided by 1000 a second time when replacing WAMEST weights.",
        "- After the fix, 2020-2021 still show a meaningful ROW/bank-row decline, but it is no longer a scale-break collapse. Treat the long-history row as materially improved, while still reporting sample splits.",
        "",
        "## Outputs",
        "",
        f"- `{audit_path.relative_to(ROOT)}`",
        f"- `{events_path.relative_to(ROOT)}`",
        f"- `{details_path.relative_to(ROOT)}`",
    ]
    md_path = reports / "tier2_component_pool_transition_audit.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    summary_path = manifests / "tier2_component_pool_transition_audit_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "window_start": START_DATE,
                "window_end": END_DATE,
                "audit_path": str(audit_path),
                "events_path": str(events_path),
                "details_path": str(details_path),
                "markdown_path": str(md_path),
                "rows_written": len(rows),
                "events_written": len(events),
                "component_detail_rows_written": len(component_details),
                "top_events": top_events,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "audit_path": str(audit_path),
                "events_path": str(events_path),
                "details_path": str(details_path),
                "markdown_path": str(md_path),
                "summary_path": str(summary_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
