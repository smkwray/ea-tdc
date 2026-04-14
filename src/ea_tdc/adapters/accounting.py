from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from ea_tdc.paths import ProjectPaths
from ea_tdc.utils import utc_now_iso, write_json


SEED_FILE = Path("quarterly_identity_flows.csv")
STANDARD_FIELDS = [
    "series_id",
    "series_label",
    "source_family",
    "source_repo",
    "source_table",
    "freq",
    "period_end",
    "release_date",
    "available_at",
    "vintage_policy",
    "units",
    "value",
    "transform_default",
    "seasonal_adjustment_flag",
    "interpolated_flag",
    "component_group",
    "role",
    "notes",
]
SERIES_LABELS = {
    "accounting_deposit_substitution_qoq": "Accounting deposit substitution",
    "accounting_bank_balance_sheet_qoq": "Accounting bank balance-sheet channel",
    "accounting_public_liquidity_qoq": "Accounting public-liquidity channel",
    "accounting_external_flow_qoq": "Accounting external-flow channel",
}


@dataclass(frozen=True)
class AdapterResult:
    standardized_path: Path
    manifest_path: Path
    rows_written: int
    seed_path: Path
    bundle_hash: str


@dataclass(frozen=True)
class DraftSeedResult:
    seed_path: Path
    reference_path: Path
    manifest_path: Path
    rows_written: int


@dataclass(frozen=True)
class ReviewResult:
    review_csv_path: Path
    review_md_path: Path
    summary_md_path: Path
    rewrite_csv_path: Path
    rewrite_md_path: Path
    manifest_path: Path
    rows_written: int
    high_priority_rows: int


@dataclass(frozen=True)
class RewriteApplyResult:
    seed_path: Path
    manifest_path: Path
    rows_updated: int


@dataclass(frozen=True)
class SeedFillResult:
    seed_path: Path
    manifest_path: Path
    rows_updated: int
    deposit_substitution_fills: int
    public_liquidity_fills: int


@dataclass(frozen=True)
class AlignmentResult:
    csv_path: Path
    md_path: Path
    manifest_path: Path
    rows_written: int


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_estimate_lookup(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    rows = _read_csv(path)
    return {
        (str(row.get("outcome", "")).strip(), str(row.get("horizon", "")).strip()): row
        for row in rows
        if str(row.get("outcome", "")).strip() and str(row.get("horizon", "")).strip()
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _quarter_end_date(quarter: str) -> str:
    year_text, quarter_text = quarter.split("Q", 1)
    year = int(year_text)
    quarter_num = int(quarter_text)
    month = quarter_num * 3
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    return (next_month.date() - timedelta(days=1)).isoformat()


def _default_available_at(quarter: str) -> str:
    return (datetime.fromisoformat(_quarter_end_date(quarter)) + timedelta(days=90)).date().isoformat()


def _quarter_sort_key(quarter: str) -> tuple[int, int]:
    year_text, quarter_text = quarter.split("Q", 1)
    return int(year_text), int(quarter_text)


def _coerce_float(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _stable_float_text(value: float, *, digits: int = 10) -> str:
    return str(round(float(value), digits))


def resolve_seed_path(paths: ProjectPaths, explicit_path: str | None = None) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_absolute():
            path = (paths.root / path).resolve()
        return path
    return paths.seed / "accounting" / SEED_FILE


def normalize_schema(seed_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    field_map = {
        "deposit_substitution_qoq": "accounting_deposit_substitution_qoq",
        "bank_balance_sheet_qoq": "accounting_bank_balance_sheet_qoq",
        "public_liquidity_qoq": "accounting_public_liquidity_qoq",
        "external_flow_qoq": "accounting_external_flow_qoq",
    }
    for seed_row in seed_rows:
        quarter = str(seed_row.get("quarter", "")).strip()
        if not quarter:
            continue
        period_end = _quarter_end_date(quarter)
        available_at = str(seed_row.get("available_at", "")).strip() or _default_available_at(quarter)
        notes = str(seed_row.get("notes", "")).strip()
        units = str(seed_row.get("units", "")).strip() or "usd_billions"
        for seed_name, series_id in field_map.items():
            value = str(seed_row.get(seed_name, "")).strip()
            if not value:
                continue
            rows.append(
                {
                    "series_id": series_id,
                    "series_label": SERIES_LABELS[series_id],
                    "source_family": "repo_seed_bundle",
                    "source_repo": "accounting",
                    "source_table": "quarterly_identity_flows",
                    "freq": "quarterly",
                    "period_end": period_end,
                    "release_date": available_at,
                    "available_at": available_at,
                    "vintage_policy": "repo_local_accounting_input",
                    "units": units,
                    "value": value,
                    "transform_default": "none",
                    "seasonal_adjustment_flag": "unknown",
                    "interpolated_flag": "false",
                    "component_group": "identity_accounting",
                    "role": "mechanism",
                    "notes": notes,
                }
            )
    return rows


def write_standard_bundle(paths: ProjectPaths, rows: list[dict[str, str]]) -> Path:
    bundle_dir = paths.bundles / "accounting"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    target = bundle_dir / "standardized_series.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STANDARD_FIELDS)
        writer.writeheader()
        if rows:
            writer.writerows(rows)
    return target


def write_source_manifest(
    paths: ProjectPaths,
    *,
    seed_path: Path,
    standardized_path: Path,
    rows_written: int,
    bundle_hash: str,
) -> Path:
    manifest = {
        "kind": "source_manifest",
        "source_repo": "accounting",
        "adapter": "accounting_identity_inputs",
        "generated_at_utc": utc_now_iso(),
        "seed_path": _display_path(seed_path, paths.root),
        "standardized_path": _display_path(standardized_path, paths.root),
        "rows_written": rows_written,
        "bundle_hash": bundle_hash,
    }
    target = paths.manifests / "accounting_source_manifest.json"
    write_json(target, manifest)
    return target


def _read_bundle_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return _read_csv(path)


def _read_quarterly_gdp(paths: ProjectPaths) -> dict[str, float]:
    path = paths.raw_fred / "GDP.csv"
    if not path.exists():
        return {}
    quarter_last: dict[str, tuple[str, float]] = {}
    for row in _read_csv(path):
        date_text = str(row.get("date", "")).strip()
        value = _coerce_float(row.get("value", ""))
        if not date_text or value is None:
            continue
        quarter = f"{date_text[:4]}Q{((int(date_text[5:7]) - 1) // 3) + 1}"
        current = quarter_last.get(quarter)
        if current is None or date_text > current[0]:
            quarter_last[quarter] = (date_text, value)
    return {quarter: value for quarter, (_, value) in quarter_last.items()}


def _seed_has_user_rows(path: Path) -> bool:
    if not path.exists():
        return False
    rows = _read_csv(path)
    return any(
        any(str(row.get(field, "")).strip() for field in [
            "deposit_substitution_qoq",
            "bank_balance_sheet_qoq",
            "public_liquidity_qoq",
            "external_flow_qoq",
            "notes",
        ])
        for row in rows
    )


def write_draft_seed_from_proxy_blocks(
    paths: ProjectPaths,
    *,
    seed_path: str | None = None,
    overwrite: bool = False,
) -> DraftSeedResult:
    raw_bundle_path = paths.bundles / "designs" / "baseline_tdc_lp_deposit_source_blocks__quarterly_bundle.csv"
    scaled_bundle_path = paths.bundles / "designs" / "baseline_tdc_lp_deposit_source_blocks_pct_gdp__quarterly_bundle.csv"
    raw_rows = _read_bundle_rows(raw_bundle_path)
    scaled_rows = {row.get("quarter", ""): row for row in _read_bundle_rows(scaled_bundle_path)}
    resolved_seed = resolve_seed_path(paths, seed_path)
    if _seed_has_user_rows(resolved_seed) and not overwrite:
        raise FileExistsError(f"Accounting seed already has user-entered rows: {resolved_seed}")

    seed_columns = [
        "quarter",
        "deposit_substitution_qoq",
        "bank_balance_sheet_qoq",
        "public_liquidity_qoq",
        "external_flow_qoq",
        "available_at",
        "units",
        "notes",
    ]
    reference_columns = [
        "quarter",
        "other_component_qoq",
        "deposit_substitution_block_qoq",
        "bank_balance_sheet_proxy_block_qoq",
        "public_liquidity_proxy_block_qoq",
        "external_flow_proxy_block_qoq",
        "proxy_accounting_total_qoq",
        "proxy_unexplained_gap_qoq",
        "other_component_qoq_pct_gdp",
        "deposit_substitution_block_qoq_pct_gdp",
        "bank_balance_sheet_proxy_block_qoq_pct_gdp",
        "public_liquidity_proxy_block_qoq_pct_gdp",
        "external_flow_proxy_block_qoq_pct_gdp",
        "proxy_accounting_total_qoq_pct_gdp",
        "proxy_unexplained_gap_qoq_pct_gdp",
    ]
    seed_rows: list[dict[str, str]] = []
    reference_rows: list[dict[str, str]] = []
    for raw_row in raw_rows:
        quarter = str(raw_row.get("quarter", "")).strip()
        if not quarter:
            continue
        other_component = str(raw_row.get("other_component_qoq", "")).strip()
        if not other_component:
            continue
        scaled_row = scaled_rows.get(quarter, {})
        seed_rows.append(
            {
                "quarter": quarter,
                "deposit_substitution_qoq": str(raw_row.get("deposit_substitution_block_qoq", "")).strip(),
                "bank_balance_sheet_qoq": str(raw_row.get("bank_balance_sheet_proxy_block_qoq", "")).strip(),
                "public_liquidity_qoq": str(raw_row.get("public_liquidity_proxy_block_qoq", "")).strip(),
                "external_flow_qoq": str(raw_row.get("external_flow_proxy_block_qoq", "")).strip(),
                "available_at": str(raw_row.get("cutoff_timestamp", "")).strip()[:10],
                "units": "usd_billions",
                "notes": "draft_prefill_from_proxy_blocks; edit_before_use",
            }
        )
        reference_rows.append(
            {
                "quarter": quarter,
                "other_component_qoq": other_component,
                "deposit_substitution_block_qoq": str(raw_row.get("deposit_substitution_block_qoq", "")).strip(),
                "bank_balance_sheet_proxy_block_qoq": str(raw_row.get("bank_balance_sheet_proxy_block_qoq", "")).strip(),
                "public_liquidity_proxy_block_qoq": str(raw_row.get("public_liquidity_proxy_block_qoq", "")).strip(),
                "external_flow_proxy_block_qoq": str(raw_row.get("external_flow_proxy_block_qoq", "")).strip(),
                "proxy_accounting_total_qoq": str(raw_row.get("proxy_accounting_total_qoq", "")).strip(),
                "proxy_unexplained_gap_qoq": str(raw_row.get("proxy_unexplained_gap_qoq", "")).strip(),
                "other_component_qoq_pct_gdp": str(scaled_row.get("other_component_qoq_pct_gdp", "")).strip(),
                "deposit_substitution_block_qoq_pct_gdp": str(scaled_row.get("deposit_substitution_block_qoq_pct_gdp", "")).strip(),
                "bank_balance_sheet_proxy_block_qoq_pct_gdp": str(scaled_row.get("bank_balance_sheet_proxy_block_qoq_pct_gdp", "")).strip(),
                "public_liquidity_proxy_block_qoq_pct_gdp": str(scaled_row.get("public_liquidity_proxy_block_qoq_pct_gdp", "")).strip(),
                "external_flow_proxy_block_qoq_pct_gdp": str(scaled_row.get("external_flow_proxy_block_qoq_pct_gdp", "")).strip(),
                "proxy_accounting_total_qoq_pct_gdp": str(scaled_row.get("proxy_accounting_total_qoq_pct_gdp", "")).strip(),
                "proxy_unexplained_gap_qoq_pct_gdp": str(scaled_row.get("proxy_unexplained_gap_qoq_pct_gdp", "")).strip(),
            }
        )

    resolved_seed.parent.mkdir(parents=True, exist_ok=True)
    with resolved_seed.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=seed_columns)
        writer.writeheader()
        writer.writerows(seed_rows)

    reference_path = paths.reports / "accounting_identity_proxy_reference.csv"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    with reference_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=reference_columns)
        writer.writeheader()
        writer.writerows(reference_rows)

    manifest_path = paths.manifests / "accounting_seed_draft_manifest.json"
    write_json(
        manifest_path,
        {
            "kind": "accounting_seed_draft_manifest",
            "generated_at_utc": utc_now_iso(),
            "seed_path": _display_path(resolved_seed, paths.root),
            "reference_path": _display_path(reference_path, paths.root),
            "proxy_raw_bundle_path": _display_path(raw_bundle_path, paths.root),
            "proxy_pct_gdp_bundle_path": _display_path(scaled_bundle_path, paths.root),
            "rows_written": len(seed_rows),
            "notes": "Draft accounting seed prefilled from proxy block bundles; edit before running adapt-accounting.",
        },
    )
    return DraftSeedResult(
        seed_path=resolved_seed,
        reference_path=reference_path,
        manifest_path=manifest_path,
        rows_written=len(seed_rows),
    )


def build_seed_review(
    paths: ProjectPaths,
    *,
    seed_path: str | None = None,
) -> ReviewResult:
    resolved_seed = resolve_seed_path(paths, seed_path)
    reference_path = paths.reports / "accounting_identity_proxy_reference.csv"
    seed_rows = {str(row.get("quarter", "")).strip(): row for row in _read_csv(resolved_seed)}
    reference_rows = {str(row.get("quarter", "")).strip(): row for row in _read_csv(reference_path)} if reference_path.exists() else {}
    gdp_by_quarter = _read_quarterly_gdp(paths)

    review_columns = [
        "quarter",
        "priority",
        "component_completeness",
        "missing_components",
        "sign_conflict",
        "gap_ratio",
        "other_component_qoq",
        "accounting_identity_total_qoq",
        "accounting_identity_gap_qoq",
        "other_component_qoq_pct_gdp",
        "accounting_identity_total_qoq_pct_gdp",
        "accounting_identity_gap_qoq_pct_gdp",
        "deposit_substitution_qoq",
        "bank_balance_sheet_qoq",
        "public_liquidity_qoq",
        "external_flow_qoq",
        "available_at",
        "notes",
    ]
    review_rows: list[dict[str, str]] = []
    for quarter in sorted(set(seed_rows).union(reference_rows), key=_quarter_sort_key):
        seed_row = seed_rows.get(quarter, {})
        reference_row = reference_rows.get(quarter, {})
        deposit_substitution = _coerce_float(seed_row.get("deposit_substitution_qoq", ""))
        bank_balance_sheet = _coerce_float(seed_row.get("bank_balance_sheet_qoq", ""))
        public_liquidity = _coerce_float(seed_row.get("public_liquidity_qoq", ""))
        external_flow = _coerce_float(seed_row.get("external_flow_qoq", ""))
        components = [deposit_substitution, bank_balance_sheet, public_liquidity, external_flow]
        missing_components = [
            label
            for label, value in [
                ("deposit_substitution_qoq", deposit_substitution),
                ("bank_balance_sheet_qoq", bank_balance_sheet),
                ("public_liquidity_qoq", public_liquidity),
                ("external_flow_qoq", external_flow),
            ]
            if value is None
        ]
        identity_total = None if any(value is None for value in components) else sum(value for value in components if value is not None)
        other_component = _coerce_float(reference_row.get("other_component_qoq", ""))
        identity_gap = None if identity_total is None or other_component is None else other_component - identity_total
        sign_conflict = bool(
            identity_total is not None
            and other_component is not None
            and identity_total != 0
            and other_component != 0
            and (identity_total > 0) != (other_component > 0)
        )
        gap_ratio = None
        if identity_gap is not None and other_component not in (None, 0.0):
            gap_ratio = abs(identity_gap) / abs(other_component)
        gdp_value = gdp_by_quarter.get(quarter)
        other_component_pct_gdp = None if other_component is None or not gdp_value else (100.0 * other_component) / gdp_value
        identity_total_pct_gdp = None if identity_total is None or not gdp_value else (100.0 * identity_total) / gdp_value
        identity_gap_pct_gdp = None if identity_gap is None or not gdp_value else (100.0 * identity_gap) / gdp_value
        if sign_conflict or (gap_ratio is not None and gap_ratio > 1.0):
            priority = "high"
        elif gap_ratio is not None and gap_ratio > 0.5:
            priority = "medium"
        else:
            priority = "low"
        review_rows.append(
            {
                "quarter": quarter,
                "priority": priority,
                "component_completeness": "complete" if not missing_components else "incomplete",
                "missing_components": ";".join(missing_components),
                "sign_conflict": "true" if sign_conflict else "false",
                "gap_ratio": _stable_float_text(gap_ratio, digits=6) if gap_ratio is not None else "",
                "other_component_qoq": seed_row.get("other_component_qoq", "") or reference_row.get("other_component_qoq", ""),
                "accounting_identity_total_qoq": _stable_float_text(identity_total) if identity_total is not None else "",
                "accounting_identity_gap_qoq": _stable_float_text(identity_gap) if identity_gap is not None else "",
                "other_component_qoq_pct_gdp": _stable_float_text(other_component_pct_gdp) if other_component_pct_gdp is not None else "",
                "accounting_identity_total_qoq_pct_gdp": _stable_float_text(identity_total_pct_gdp) if identity_total_pct_gdp is not None else "",
                "accounting_identity_gap_qoq_pct_gdp": _stable_float_text(identity_gap_pct_gdp) if identity_gap_pct_gdp is not None else "",
                "deposit_substitution_qoq": seed_row.get("deposit_substitution_qoq", ""),
                "bank_balance_sheet_qoq": seed_row.get("bank_balance_sheet_qoq", ""),
                "public_liquidity_qoq": seed_row.get("public_liquidity_qoq", ""),
                "external_flow_qoq": seed_row.get("external_flow_qoq", ""),
                "available_at": seed_row.get("available_at", ""),
                "notes": seed_row.get("notes", ""),
            }
        )

    review_csv_path = paths.reports / "accounting_identity_seed_review.csv"
    review_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with review_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_columns)
        writer.writeheader()
        writer.writerows(review_rows)

    ranked_rows = sorted(
        review_rows,
        key=lambda row: (
            0 if row["priority"] == "high" else 1 if row["priority"] == "medium" else 2,
            -abs(float(row["accounting_identity_gap_qoq_pct_gdp"])) if row["accounting_identity_gap_qoq_pct_gdp"] else 0.0,
            row["quarter"],
        ),
    )
    high_priority_rows = sum(1 for row in review_rows if row["priority"] == "high")
    medium_priority_rows = sum(1 for row in review_rows if row["priority"] == "medium")
    incomplete_rows = sum(1 for row in review_rows if row["component_completeness"] == "incomplete")
    top_rows = ranked_rows[:12]
    component_fields = [
        "deposit_substitution_qoq",
        "bank_balance_sheet_qoq",
        "public_liquidity_qoq",
        "external_flow_qoq",
    ]
    component_labels = {
        "deposit_substitution_qoq": "Deposit substitution",
        "bank_balance_sheet_qoq": "Bank balance sheet",
        "public_liquidity_qoq": "Public liquidity",
        "external_flow_qoq": "External flow",
    }
    high_rows = [row for row in review_rows if row["priority"] == "high"]
    leader_counts: dict[str, int] = {field: 0 for field in component_fields}
    for row in high_rows:
        values = {
            field: _coerce_float(row.get(field, ""))
            for field in component_fields
        }
        available = {field: value for field, value in values.items() if value is not None}
        if not available:
            continue
        leader = max(available, key=lambda field: abs(available[field]))
        leader_counts[leader] += 1
    era_buckets = [
        ("2003-2007", lambda year: 2003 <= year <= 2007),
        ("2008-2010", lambda year: 2008 <= year <= 2010),
        ("2011-2019", lambda year: 2011 <= year <= 2019),
        ("2020-2021", lambda year: 2020 <= year <= 2021),
        ("2022+", lambda year: year >= 2022),
    ]
    era_counts: list[tuple[str, int]] = []
    for label, predicate in era_buckets:
        era_counts.append((label, sum(1 for row in high_rows if predicate(int(row["quarter"][:4])))))
    header = [
        "# Accounting Seed Review",
        "",
        f"- Seed path: `{_display_path(resolved_seed, paths.root)}`",
        f"- Reference path: `{_display_path(reference_path, paths.root)}`",
        f"- Rows reviewed: `{len(review_rows)}`",
        f"- High-priority rows: `{high_priority_rows}`",
        f"- Medium-priority rows: `{medium_priority_rows}`",
        f"- Incomplete rows: `{incomplete_rows}`",
        "",
        "Top rows to rewrite first:",
        "",
        "| Quarter | Priority | Completeness | Missing components | Sign conflict | Gap ratio | Other component | Identity total | Identity gap | Notes |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    body = [
        (
            f"| {row['quarter']} | {row['priority']} | {row['component_completeness']} | {row['missing_components'] or ''} | {row['sign_conflict']} | {row['gap_ratio'] or ''} | "
            f"{row['other_component_qoq'] or ''} | {row['accounting_identity_total_qoq'] or ''} | "
            f"{row['accounting_identity_gap_qoq'] or ''} | {row['notes'] or ''} |"
        )
        for row in top_rows
    ]
    review_md_path = paths.reports / "accounting_identity_seed_review.md"
    review_md_path.write_text("\n".join([*header, *body, ""]), encoding="utf-8")

    summary_lines = [
        "# Accounting Seed Review Summary",
        "",
        f"- Rows reviewed: `{len(review_rows)}`",
        f"- High-priority rows: `{high_priority_rows}`",
        f"- Medium-priority rows: `{medium_priority_rows}`",
        f"- Incomplete rows: `{incomplete_rows}`",
        "",
        "## Main pattern",
        "",
        "The current draft problems are not evenly spread across channels. High-priority rows identify complete-quarter contradictions, while incomplete rows are quarters excluded from direct identity scoring because at least one accounting channel is blank.",
        "",
        "### High-priority leader counts",
        "",
        "| Channel | High-priority rows led |",
        "| --- | ---: |",
    ]
    summary_lines.extend(
        f"| {component_labels[field]} | {leader_counts[field]} |"
        for field in component_fields
    )
    summary_lines.extend(
        [
            "",
            "### High-priority rows by era",
            "",
            "| Era | High-priority rows |",
            "| --- | ---: |",
        ]
    )
    summary_lines.extend(f"| {label} | {count} |" for label, count in era_counts)
    summary_lines.extend(
        [
            "",
            "## Rewrite order",
            "",
            "1. Rewrite the high-priority rows first, starting with quarters where the identity total has the wrong sign or a very large gap ratio.",
            "2. In those rows, rewrite `external_flow_qoq` first unless another channel clearly dominates; it is the leading component in most high-priority rows.",
            "3. Use the CSV review table for the full queue and the Markdown review table for the top quarters.",
            "",
            "## Worst gap-ratio rows",
            "",
            "| Quarter | Gap ratio | Other component | Identity total | Identity gap |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    summary_lines.extend(
        f"| {row['quarter']} | {row['gap_ratio'] or ''} | {row['other_component_qoq'] or ''} | {row['accounting_identity_total_qoq'] or ''} | {row['accounting_identity_gap_qoq'] or ''} |"
        for row in sorted(
            [row for row in high_rows if row["gap_ratio"]],
            key=lambda row: float(row["gap_ratio"]),
            reverse=True,
        )[:10]
    )
    summary_md_path = paths.reports / "accounting_identity_seed_summary.md"
    summary_md_path.write_text("\n".join([*summary_lines, ""]), encoding="utf-8")

    rewrite_columns = [
        "quarter",
        "priority",
        "other_component_qoq",
        "fixed_non_external_total_qoq",
        "current_external_flow_qoq",
        "implied_external_flow_qoq",
        "external_flow_delta_qoq",
        "other_component_qoq_pct_gdp",
        "fixed_non_external_total_qoq_pct_gdp",
        "current_external_flow_qoq_pct_gdp",
        "implied_external_flow_qoq_pct_gdp",
        "external_flow_delta_qoq_pct_gdp",
        "notes",
    ]
    rewrite_rows: list[dict[str, str]] = []
    for row in review_rows:
        deposit_substitution = _coerce_float(row.get("deposit_substitution_qoq", ""))
        bank_balance_sheet = _coerce_float(row.get("bank_balance_sheet_qoq", ""))
        public_liquidity = _coerce_float(row.get("public_liquidity_qoq", ""))
        current_external = _coerce_float(row.get("external_flow_qoq", ""))
        other_component = _coerce_float(row.get("other_component_qoq", ""))
        if other_component is None:
            continue
        non_external_components = [deposit_substitution, bank_balance_sheet, public_liquidity]
        fixed_non_external_total = None if any(value is None for value in non_external_components) else sum(
            value for value in non_external_components if value is not None
        )
        implied_external = None if fixed_non_external_total is None else other_component - fixed_non_external_total
        external_delta = None if implied_external is None or current_external is None else implied_external - current_external
        quarter = row["quarter"]
        gdp_value = gdp_by_quarter.get(quarter)
        fixed_non_external_total_pct_gdp = None if fixed_non_external_total is None or not gdp_value else (100.0 * fixed_non_external_total) / gdp_value
        current_external_pct_gdp = None if current_external is None or not gdp_value else (100.0 * current_external) / gdp_value
        implied_external_pct_gdp = None if implied_external is None or not gdp_value else (100.0 * implied_external) / gdp_value
        external_delta_pct_gdp = None if external_delta is None or not gdp_value else (100.0 * external_delta) / gdp_value
        rewrite_rows.append(
            {
                "quarter": quarter,
                "priority": row["priority"],
                "other_component_qoq": row["other_component_qoq"],
                "fixed_non_external_total_qoq": _stable_float_text(fixed_non_external_total) if fixed_non_external_total is not None else "",
                "current_external_flow_qoq": row["external_flow_qoq"],
                "implied_external_flow_qoq": _stable_float_text(implied_external) if implied_external is not None else "",
                "external_flow_delta_qoq": _stable_float_text(external_delta) if external_delta is not None else "",
                "other_component_qoq_pct_gdp": row["other_component_qoq_pct_gdp"],
                "fixed_non_external_total_qoq_pct_gdp": _stable_float_text(fixed_non_external_total_pct_gdp) if fixed_non_external_total_pct_gdp is not None else "",
                "current_external_flow_qoq_pct_gdp": _stable_float_text(current_external_pct_gdp) if current_external_pct_gdp is not None else "",
                "implied_external_flow_qoq_pct_gdp": _stable_float_text(implied_external_pct_gdp) if implied_external_pct_gdp is not None else "",
                "external_flow_delta_qoq_pct_gdp": _stable_float_text(external_delta_pct_gdp) if external_delta_pct_gdp is not None else "",
                "notes": row["notes"],
            }
        )

    rewrite_csv_path = paths.reports / "accounting_identity_external_flow_rewrite.csv"
    with rewrite_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rewrite_columns)
        writer.writeheader()
        writer.writerows(rewrite_rows)
    top_rewrite_rows = sorted(
        [row for row in rewrite_rows if row["priority"] == "high" and row["external_flow_delta_qoq"]],
        key=lambda row: abs(float(row["external_flow_delta_qoq"])),
        reverse=True,
    )[:12]
    rewrite_md_path = paths.reports / "accounting_identity_external_flow_rewrite.md"
    rewrite_md_path.write_text(
        "\n".join(
            [
                "# External-Flow Rewrite Worksheet",
                "",
                "These rows hold deposit substitution, bank balance sheet, and public liquidity fixed, then compute the `external_flow_qoq` that would exactly reconcile `other_component_qoq` quarter by quarter.",
                "",
                "| Quarter | Priority | Other component | Fixed non-external total | Current external | Implied external | External delta |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
                *[
                    (
                        f"| {row['quarter']} | {row['priority']} | {row['other_component_qoq'] or ''} | "
                        f"{row['fixed_non_external_total_qoq'] or ''} | {row['current_external_flow_qoq'] or ''} | "
                        f"{row['implied_external_flow_qoq'] or ''} | {row['external_flow_delta_qoq'] or ''} |"
                    )
                    for row in top_rewrite_rows
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest_path = paths.manifests / "accounting_identity_seed_review_manifest.json"
    write_json(
        manifest_path,
        {
            "kind": "accounting_identity_seed_review_manifest",
            "generated_at_utc": utc_now_iso(),
            "seed_path": _display_path(resolved_seed, paths.root),
            "reference_path": _display_path(reference_path, paths.root),
            "review_csv_path": _display_path(review_csv_path, paths.root),
            "review_md_path": _display_path(review_md_path, paths.root),
            "summary_md_path": _display_path(summary_md_path, paths.root),
            "rewrite_csv_path": _display_path(rewrite_csv_path, paths.root),
            "rewrite_md_path": _display_path(rewrite_md_path, paths.root),
            "rows_written": len(review_rows),
            "high_priority_rows": high_priority_rows,
            "medium_priority_rows": medium_priority_rows,
        },
    )
    return ReviewResult(
        review_csv_path=review_csv_path,
        review_md_path=review_md_path,
        summary_md_path=summary_md_path,
        rewrite_csv_path=rewrite_csv_path,
        rewrite_md_path=rewrite_md_path,
        manifest_path=manifest_path,
        rows_written=len(review_rows),
        high_priority_rows=high_priority_rows,
    )


def apply_external_flow_rewrite(
    paths: ProjectPaths,
    *,
    seed_path: str | None = None,
    rewrite_csv_path: str | None = None,
    min_priority: str = "high",
) -> RewriteApplyResult:
    priority_order = {"high": 0, "medium": 1, "low": 2}
    threshold = priority_order[min_priority]
    resolved_seed = resolve_seed_path(paths, seed_path)
    resolved_rewrite = Path(rewrite_csv_path).expanduser() if rewrite_csv_path else (paths.reports / "accounting_identity_external_flow_rewrite.csv")
    if not resolved_rewrite.is_absolute():
        resolved_rewrite = (paths.root / resolved_rewrite).resolve()

    rewrite_rows = {
        str(row.get("quarter", "")).strip(): row
        for row in _read_csv(resolved_rewrite)
        if str(row.get("quarter", "")).strip()
    }
    seed_rows = _read_csv(resolved_seed)
    fieldnames = list(seed_rows[0].keys()) if seed_rows else [
        "quarter",
        "deposit_substitution_qoq",
        "bank_balance_sheet_qoq",
        "public_liquidity_qoq",
        "external_flow_qoq",
        "available_at",
        "units",
        "notes",
    ]
    rows_updated = 0
    for row in seed_rows:
        quarter = str(row.get("quarter", "")).strip()
        rewrite_row = rewrite_rows.get(quarter)
        if rewrite_row is None:
            continue
        priority = str(rewrite_row.get("priority", "")).strip()
        if priority not in priority_order or priority_order[priority] > threshold:
            continue
        implied_external = str(rewrite_row.get("implied_external_flow_qoq", "")).strip()
        if not implied_external:
            continue
        row["external_flow_qoq"] = implied_external
        notes = str(row.get("notes", "")).strip()
        marker = f"external_flow_rewritten_from_identity_{priority}"
        if marker not in notes:
            row["notes"] = f"{notes}; {marker}".strip("; ")
        rows_updated += 1

    with resolved_seed.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(seed_rows)

    manifest_path = paths.manifests / "accounting_external_flow_rewrite_apply_manifest.json"
    write_json(
        manifest_path,
        {
            "kind": "accounting_external_flow_rewrite_apply_manifest",
            "generated_at_utc": utc_now_iso(),
            "seed_path": _display_path(resolved_seed, paths.root),
            "rewrite_csv_path": _display_path(resolved_rewrite, paths.root),
            "min_priority": min_priority,
            "rows_updated": rows_updated,
        },
    )
    return RewriteApplyResult(
        seed_path=resolved_seed,
        manifest_path=manifest_path,
        rows_updated=rows_updated,
    )


def fill_missing_seed_channels_from_proxy_blocks(
    paths: ProjectPaths,
    *,
    seed_path: str | None = None,
) -> SeedFillResult:
    resolved_seed = resolve_seed_path(paths, seed_path)
    raw_bundle_path = paths.bundles / "designs" / "baseline_tdc_lp_deposit_source_blocks__quarterly_bundle.csv"
    proxy_rows = {str(row.get("quarter", "")).strip(): row for row in _read_bundle_rows(raw_bundle_path)}
    seed_rows = _read_csv(resolved_seed)
    fieldnames = list(seed_rows[0].keys()) if seed_rows else [
        "quarter",
        "deposit_substitution_qoq",
        "bank_balance_sheet_qoq",
        "public_liquidity_qoq",
        "external_flow_qoq",
        "available_at",
        "units",
        "notes",
    ]

    rows_updated = 0
    deposit_substitution_fills = 0
    public_liquidity_fills = 0
    for row in seed_rows:
        quarter = str(row.get("quarter", "")).strip()
        proxy_row = proxy_rows.get(quarter)
        if proxy_row is None:
            continue
        row_updated = False
        if not str(row.get("deposit_substitution_qoq", "")).strip():
            proxy_value = str(proxy_row.get("deposit_substitution_block_qoq", "")).strip()
            if proxy_value:
                row["deposit_substitution_qoq"] = proxy_value
                notes = str(row.get("notes", "")).strip()
                marker = "deposit_substitution_filled_from_proxy_blocks"
                if marker not in notes:
                    row["notes"] = f"{notes}; {marker}".strip("; ")
                deposit_substitution_fills += 1
                row_updated = True
        if not str(row.get("public_liquidity_qoq", "")).strip():
            proxy_value = str(proxy_row.get("public_liquidity_proxy_block_qoq", "")).strip()
            if proxy_value:
                row["public_liquidity_qoq"] = proxy_value
                notes = str(row.get("notes", "")).strip()
                marker = "public_liquidity_filled_from_proxy_blocks"
                if marker not in notes:
                    row["notes"] = f"{notes}; {marker}".strip("; ")
                public_liquidity_fills += 1
                row_updated = True
        if row_updated:
            rows_updated += 1

    with resolved_seed.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(seed_rows)

    manifest_path = paths.manifests / "accounting_seed_fill_manifest.json"
    write_json(
        manifest_path,
        {
            "kind": "accounting_seed_fill_manifest",
            "generated_at_utc": utc_now_iso(),
            "seed_path": _display_path(resolved_seed, paths.root),
            "proxy_bundle_path": _display_path(raw_bundle_path, paths.root),
            "rows_updated": rows_updated,
            "deposit_substitution_fills": deposit_substitution_fills,
            "public_liquidity_fills": public_liquidity_fills,
        },
    )
    return SeedFillResult(
        seed_path=resolved_seed,
        manifest_path=manifest_path,
        rows_updated=rows_updated,
        deposit_substitution_fills=deposit_substitution_fills,
        public_liquidity_fills=public_liquidity_fills,
    )


def build_accounting_identity_alignment(paths: ProjectPaths) -> AlignmentResult:
    public_estimates_path = paths.output / "models" / "baseline_tdc_lp_deposits__robustness_k200_estimates.csv"
    identity_estimates_path = paths.output / "models" / "baseline_tdc_lp_deposit_source_identity__lp_estimates.csv"
    identity_pct_gdp_path = paths.output / "models" / "baseline_tdc_lp_deposit_source_identity_pct_gdp__lp_estimates.csv"
    public_lookup = _read_estimate_lookup(public_estimates_path)
    identity_lookup = _read_estimate_lookup(identity_estimates_path)
    identity_pct_gdp_lookup = _read_estimate_lookup(identity_pct_gdp_path)

    horizons = sorted(
        {
            int(horizon)
            for outcome, horizon in public_lookup
            if outcome == "other_component_qoq"
        }
        & {
            int(horizon)
            for outcome, horizon in identity_lookup
            if outcome in {"accounting_identity_total_qoq", "accounting_identity_gap_qoq"}
        }
        & {
            int(horizon)
            for outcome, horizon in identity_pct_gdp_lookup
            if outcome == "accounting_identity_gap_qoq_pct_gdp"
        }
    )

    fieldnames = [
        "horizon",
        "residual_beta",
        "accounting_total_beta",
        "identity_gap_beta",
        "gap_pct_gdp_beta",
        "public_minus_accounting_total_beta",
        "identity_gap_share_of_residual",
        "public_residual_p",
        "accounting_total_p",
        "identity_gap_p",
        "gap_pct_gdp_p",
    ]
    rows: list[dict[str, str]] = []
    for horizon in horizons:
        key = ("other_component_qoq", str(horizon))
        public_row = public_lookup.get(key)
        total_row = identity_lookup.get(("accounting_identity_total_qoq", str(horizon)))
        gap_row = identity_lookup.get(("accounting_identity_gap_qoq", str(horizon)))
        gap_pct_gdp_row = identity_pct_gdp_lookup.get(("accounting_identity_gap_qoq_pct_gdp", str(horizon)))
        if public_row is None or total_row is None or gap_row is None or gap_pct_gdp_row is None:
            continue
        residual_beta = _coerce_float(public_row.get("beta", ""))
        total_beta = _coerce_float(total_row.get("beta", ""))
        gap_beta = _coerce_float(gap_row.get("beta", ""))
        gap_pct_gdp_beta = _coerce_float(gap_pct_gdp_row.get("beta", ""))
        if residual_beta is None or total_beta is None or gap_beta is None or gap_pct_gdp_beta is None:
            continue
        public_minus_total = residual_beta - total_beta
        gap_share = abs(gap_beta) / abs(residual_beta) if residual_beta not in (0.0, None) else None
        rows.append(
            {
                "horizon": str(horizon),
                "residual_beta": _stable_float_text(residual_beta, digits=12),
                "accounting_total_beta": _stable_float_text(total_beta, digits=12),
                "identity_gap_beta": _stable_float_text(gap_beta, digits=12),
                "gap_pct_gdp_beta": _stable_float_text(gap_pct_gdp_beta, digits=12),
                "public_minus_accounting_total_beta": _stable_float_text(public_minus_total, digits=12),
                "identity_gap_share_of_residual": _stable_float_text(gap_share, digits=6) if gap_share is not None else "",
                "public_residual_p": str(public_row.get("p_value", "")).strip(),
                "accounting_total_p": str(total_row.get("p_value", "")).strip(),
                "identity_gap_p": str(gap_row.get("p_value", "")).strip(),
                "gap_pct_gdp_p": str(gap_pct_gdp_row.get("p_value", "")).strip(),
            }
        )

    csv_path = paths.reports / "accounting_identity_alignment.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    table_lines = [
        "| Horizon | Public residual | Identity total | Direct identity gap | Arithmetic residual minus total | Gap share of residual |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        table_lines.append(
            f"| {row['horizon']} | {float(row['residual_beta']):.3f} | {float(row['accounting_total_beta']):.3f} | "
            f"{float(row['identity_gap_beta']):.3f} | {float(row['public_minus_accounting_total_beta']):.3f} | "
            f"{float(row['identity_gap_share_of_residual']):.3f} |"
        )

    md_path = paths.reports / "accounting_identity_alignment.md"
    md_path.write_text(
        "\n".join(
            [
                "# Accounting Identity Alignment",
                "",
                "This note compares two different objects:",
                "- the public `other_component_qoq` residual from the selected screened deposits branch;",
                "- the hidden accounting-identity outcomes from the rebuilt identity job.",
                "",
                "Important distinction:",
                "- `identity_gap_beta` is the coefficient on the direct hidden outcome `accounting_identity_gap_qoq`.",
                "- `arithmetic residual minus total` is the simple difference between the public residual IRF and the hidden accounting-total IRF.",
                "- Those two numbers need not match because they come from different estimation jobs with different samples and control stacks.",
                "",
                "What this can support:",
                "- weak-to-moderate internal coherence evidence that the TDC measure is not grossly misaligned with the theory;",
                "- a mechanism-side closure check for the hidden accounting design.",
                "",
                "What this cannot support:",
                "- independent validation of the TDC treatment;",
                "- a claim that the public residual and hidden accounting total are pointwise identical at every horizon.",
                "",
                *table_lines,
                "",
                "Readout:",
                f"- Impact (`h=0`): public residual = {float(rows[0]['residual_beta']):.3f}, identity total = {float(rows[0]['accounting_total_beta']):.3f}, direct identity gap = {float(rows[0]['identity_gap_beta']):.3f}." if rows else "- No aligned horizons found.",
                "- Best interpretation: the direct identity-gap outcome is now small relative to the residual, which is useful internal evidence against gross misspecification, but not a standalone validation design.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest_path = paths.manifests / "accounting_identity_alignment_manifest.json"
    write_json(
        manifest_path,
        {
            "kind": "accounting_identity_alignment_manifest",
            "generated_at_utc": utc_now_iso(),
            "public_estimates_path": _display_path(public_estimates_path, paths.root),
            "identity_estimates_path": _display_path(identity_estimates_path, paths.root),
            "identity_pct_gdp_estimates_path": _display_path(identity_pct_gdp_path, paths.root),
            "csv_path": _display_path(csv_path, paths.root),
            "md_path": _display_path(md_path, paths.root),
            "rows_written": len(rows),
        },
    )
    return AlignmentResult(
        csv_path=csv_path,
        md_path=md_path,
        manifest_path=manifest_path,
        rows_written=len(rows),
    )


def adapt_accounting(paths: ProjectPaths, *, seed_path: str | None = None) -> AdapterResult:
    resolved_seed = resolve_seed_path(paths, seed_path)
    if not resolved_seed.exists():
        raise FileNotFoundError(f"Missing accounting seed file: {resolved_seed}")
    seed_rows = _read_csv(resolved_seed)
    rows = normalize_schema(seed_rows)
    standardized_path = write_standard_bundle(paths, rows)
    bundle_hash = _sha256_file(resolved_seed)
    manifest_path = write_source_manifest(
        paths,
        seed_path=resolved_seed,
        standardized_path=standardized_path,
        rows_written=len(rows),
        bundle_hash=bundle_hash,
    )
    return AdapterResult(
        standardized_path=standardized_path,
        manifest_path=manifest_path,
        rows_written=len(rows),
        seed_path=resolved_seed,
        bundle_hash=bundle_hash,
    )
