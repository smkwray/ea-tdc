from __future__ import annotations

import csv
import html
import json
import shutil
from datetime import date
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap
from typing import Any

from ea_tdc.open_contract import (
    CANONICAL_CONTROL_IDS,
    CANONICAL_OUTCOME_ID,
    CANONICAL_RESIDUAL_ID,
    CANONICAL_TREATMENT_ID,
    CANONICAL_TREATMENT_LABEL,
    CREDIT_SCREEN_OUTCOME_IDS,
)
from ea_tdc.paths import ProjectPaths
from ea_tdc.reporting import build_release_artifact_contract
from ea_tdc.utils import utc_now_iso, write_json

LABEL_OVERRIDES = {
    "matched_total_deposits": "Matched total deposits",
    "domestic_nonbank_deposits_qoq": "Domestic nonbank deposits",
    "domestic_nonbank_other_component_qoq": "Domestic nonbank residual",
    "domestic_nonbank_other_component_core_deposit_proximate_qoq": "Domestic nonbank residual, TOC/ROW-excluded core",
    "other_component_qoq": "Other component (q/q)",
    CANONICAL_RESIDUAL_ID: "Other component, same Tier 2 treatment",
    "m2": "M2",
    "GDP": "Real GDP",
    "gdp_deflator": "GDP deflator",
    "FEDFUNDS": "Effective federal funds rate",
    "TOTRESNS": "Total reserves",
    "DFF": "Daily federal funds rate",
    "DGS2": "2Y Treasury yield",
    "DGS10": "10Y Treasury yield",
    "dgs2": "2Y Treasury yield",
    "dgs10": "10Y Treasury yield",
    "THREEFYTP10": "10Y term premium",
    "WDTGAL": "Treasury General Account",
    "WRESBAL": "Reserve balances",
    "repo_spread": "Repo spread",
    "reserve_balances": "Raw reserve balances",
    "fed_funds": "Fed funds",
    "sofr": "SOFR",
    "large_time_deposits_qoq": "Large time deposits",
    "retail_mmf_assets_qoq": "Retail MMFs",
    "institutional_mmf_assets_qoq": "Institutional MMFs",
    "baa_aaa": "BAA-AAA spread",
    "investment_grade_oas": "Investment-grade OAS",
    "high_yield_oas": "High-yield OAS",
    "bank_non_treasury_securities_qoq": "Bank non-Treasury securities",
    "bank_treasury_securities_qoq": "Bank Treasury securities",
    "bank_treasury_securities_transactions_qoq": "Bank Treasury securities transactions",
    "bank_treasury_agency_securities_qoq": "Bank Treasury and agency securities",
    "bank_credit_qoq": "Bank credit",
    "tdcpass_strict_loan_core_min_qoq": "Strict loan core",
    "tdcpass_strict_loan_mortgages_qoq": "Mortgages",
    "tdcpass_strict_loan_consumer_credit_qoq": "Consumer credit",
    "bank_business_loans_qoq": "Business loans",
    "bank_ci_loans_h8_qoq": "C&I loans (H.8)",
    "bank_short_term_loans_z1_qoq": "Short-term bank loans (Z.1)",
    "bank_consumer_loans_qoq": "Consumer loans",
    "bank_real_estate_loans_qoq": "Real-estate loans",
    "row_loans_assets_qoq": "ROW loans/assets",
    "row_corp_bonds_flow": "ROW corporate-bond flow",
    "row_private_flow_block": "ROW private-flow block",
    "exports_qoq": "Exports",
    "imports_qoq": "Imports",
    "net_exports_qoq": "Net exports",
    "current_account_balance": "Current account balance",
    "tga_balance_qoq": "TGA balance",
    "on_rrp_balance_qoq": "ON RRP balance",
    "deposit_substitution_block_qoq": "Deposit-substitution block",
    "bank_balance_sheet_proxy_block_qoq": "Bank balance-sheet block",
    "public_liquidity_proxy_block_qoq": "Public-liquidity block",
    "external_flow_proxy_block_qoq": "External-flow block",
    "proxy_accounting_total_qoq": "Proxy-accounting total",
    "proxy_unexplained_gap_qoq": "Proxy unexplained gap",
    "accounting_deposit_substitution_qoq": "Accounting deposit substitution",
    "accounting_bank_balance_sheet_qoq": "Accounting bank balance-sheet channel",
    "accounting_public_liquidity_qoq": "Accounting public-liquidity channel",
    "accounting_external_flow_qoq": "Accounting external-flow channel",
    "accounting_identity_total_qoq": "Accounting identity total",
    "accounting_identity_gap_qoq": "Accounting identity gap",
    "matched_total_deposits_pct_gdp": "Matched total deposits (% GDP)",
    "other_component_qoq_pct_gdp": "Other component (% GDP)",
    "large_time_deposits_qoq_pct_gdp": "Large time deposits (% GDP)",
    "retail_mmf_assets_qoq_pct_gdp": "Retail MMFs (% GDP)",
    "institutional_mmf_assets_qoq_pct_gdp": "Institutional MMFs (% GDP)",
    "bank_credit_qoq_pct_gdp": "Bank credit (% GDP)",
    "bank_business_loans_qoq_pct_gdp": "Business loans (% GDP)",
    "bank_ci_loans_h8_qoq_pct_gdp": "C&I loans (H.8, % GDP)",
    "bank_short_term_loans_z1_qoq_pct_gdp": "Short-term bank loans (Z.1, % GDP)",
    "bank_non_treasury_securities_qoq_pct_gdp": "Bank non-Treasury securities (% GDP)",
    "bank_treasury_securities_qoq_pct_gdp": "Bank Treasury securities (% GDP)",
    "bank_treasury_securities_transactions_qoq_pct_gdp": "Bank Treasury securities transactions (% GDP)",
    "bank_treasury_agency_securities_qoq_pct_gdp": "Bank Treasury and agency securities (% GDP)",
    "bank_consumer_loans_qoq_pct_gdp": "Consumer loans (% GDP)",
    "bank_real_estate_loans_qoq_pct_gdp": "Real-estate loans (% GDP)",
    "row_loans_assets_qoq_pct_gdp": "ROW loans/assets (% GDP)",
    "row_corp_bonds_flow_pct_gdp": "ROW corporate-bond flow (% GDP)",
    "row_private_flow_block_pct_gdp": "ROW private-flow block (% GDP)",
    "exports_qoq_pct_gdp": "Exports (% GDP)",
    "imports_qoq_pct_gdp": "Imports (% GDP)",
    "net_exports_qoq_pct_gdp": "Net exports (% GDP)",
    "current_account_balance_pct_gdp": "Current account balance (% GDP)",
    "tga_balance_qoq_pct_gdp": "TGA balance (% GDP)",
    "on_rrp_balance_qoq_pct_gdp": "ON RRP balance (% GDP)",
    "deposit_substitution_block_qoq_pct_gdp": "Deposit-substitution block (% GDP)",
    "bank_balance_sheet_proxy_block_qoq_pct_gdp": "Bank balance-sheet block (% GDP)",
    "public_liquidity_proxy_block_qoq_pct_gdp": "Public-liquidity block (% GDP)",
    "external_flow_proxy_block_qoq_pct_gdp": "External-flow block (% GDP)",
    "proxy_accounting_total_qoq_pct_gdp": "Proxy-accounting total (% GDP)",
    "proxy_unexplained_gap_qoq_pct_gdp": "Proxy unexplained gap (% GDP)",
    "accounting_deposit_substitution_qoq_pct_gdp": "Accounting deposit substitution (% GDP)",
    "accounting_bank_balance_sheet_qoq_pct_gdp": "Accounting bank balance-sheet channel (% GDP)",
    "accounting_public_liquidity_qoq_pct_gdp": "Accounting public-liquidity channel (% GDP)",
    "accounting_external_flow_qoq_pct_gdp": "Accounting external-flow channel (% GDP)",
    "accounting_identity_total_qoq_pct_gdp": "Accounting identity total (% GDP)",
    "accounting_identity_gap_qoq_pct_gdp": "Accounting identity gap (% GDP)",
    "tdc_bank_only_qoq": "Baseline TDC estimate",
    CANONICAL_TREATMENT_ID: "Long-history Tier 2 TDC",
    "qra_ati_baseline_bn": "QRA baseline tilt",
    "direct_at_h": "Direct response at horizon h",
    "tier2_regression_bank_row_tier_pre_component_h15_scaled": "Tier 2 method-tier control",
    "tdcpass_strict_loan_core_min_qoq__lag_2": "Strict loan-core lag 2",
    "tdcpass_strict_loan_core_min_qoq__lag_4": "Strict loan-core lag 4",
    "tdcpass_strict_loan_consumer_credit_qoq__lag_4": "Consumer-credit lag 4",
    "bank_credit_qoq__lag_4": "Bank-credit lag 4",
    "dgs2__lag_4": "2Y Treasury-yield lag 4",
    "dgs10__lag_1": "10Y Treasury-yield lag 1",
    "dgs10__lag_2": "10Y Treasury-yield lag 2",
    "newey_west": "Newey-West HAC",
    "ols_newey_west_scaffold": "OLS with Newey-West HAC",
    "ols_hc1_scaffold": "OLS with HC1 robust errors",
    "iv_2sls_hc1_scaffold": "2SLS with HC1 robust errors",
    "event_window_delta": "Event-window change",
    "headline_identified": "Committed headline design",
    "supporting_reduced_form": "Appendix supporting design",
    "p_value_normal": "P-value",
    "lower95": "Lower 95%",
    "upper95": "Upper 95%",
    "rsquared": "R-squared",
    "n": "Observations",
}

TOKEN_TITLE_OVERRIDES = {
    "tdc": "TDC",
    "slr": "SLR",
    "qra": "QRA",
    "lpiv": "LP-IV",
    "fx": "FX",
    "rrp": "RRP",
    "on": "ON",
    "fed": "Fed",
    "row": "ROW",
    "iv": "IV",
    "tmle": "TMLE",
    "dml": "DML",
    "sofr": "SOFR",
    "tga": "TGA",
    "m2": "M2",
}

ESTIMATOR_LABELS = {
    "lp": "Local projection",
    "lp_iv": "IV local projection",
    "event_lp": "Event study",
}

JOB_TITLE_OVERRIDES = {
    "baseline_tdc_lp_deposits": "Deposit responses to the baseline TDC estimate",
    "paper_tier2_selected_credit_rate_lags": "Long-history Tier 2 selected-lag pass-through",
    "baseline_tdc_lp_funding": "Funding and rate responses to the baseline TDC estimate",
    "baseline_tdc_lp_credit_spreads": "Credit-spread responses to the baseline TDC estimate",
    "tdc_state_dep_low_reserves": "State dependence under low-reserve conditions",
    "tdc_state_dep_on_rrp_drain": "State dependence during ON RRP drain episodes",
    "tdc_state_dep_bank_short_share": "State dependence with high bank short-share exposure",
    "tdc_state_dep_slr_bank_leverage_pressure": "State dependence under SLR leverage pressure",
    "tdc_state_dep_bank_foreign_private_corr": "State dependence with high bank-foreign-private correlation",
}

PAPER_TIER2_JOB_ID = "paper_tier2_selected_credit_rate_lags"
PAPER_TIER2_SOURCE_ESTIMATES = "tier2_credit_causality_state_estimates.csv"
PAPER_TIER2_ESTIMATES = f"{PAPER_TIER2_JOB_ID}_estimates.csv"
PAPER_TIER2_TREATMENT_LABEL = f"{CANONICAL_TREATMENT_LABEL}_selected_credit_rate_lags"
PAPER_TIER2_SAMPLE_LABEL = "full_available"
PAPER_TIER2_SURFACE = "pooled_baseline"
PAPER_TIER2_TREATMENT_ID = CANONICAL_TREATMENT_ID
PAPER_TIER2_CONTROLS = list(CANONICAL_CONTROL_IDS)
PAPER_TIER2_MAIN_FIGURE_OUTCOMES = [
    CANONICAL_OUTCOME_ID,
    CANONICAL_RESIDUAL_ID,
]
PAPER_TIER2_MAIN_TABLE_OUTCOMES = [
    CANONICAL_OUTCOME_ID,
    CANONICAL_RESIDUAL_ID,
    *CREDIT_SCREEN_OUTCOME_IDS,
]
PAPER_TIER2_HORIZONS = ["0", "1", "2", "4", "8"]


@dataclass(frozen=True)
class ReleaseArtifactBuildResult:
    summary_path: Path
    summary_csv_path: Path
    artifacts_built: int
    figure_artifacts: int
    table_artifacts: int
    gallery_path: Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_float(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _split_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def _humanize(text: str) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return ""
    if normalized in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[normalized]
    parts = [part for part in normalized.replace("_", " ").split() if part]
    rendered: list[str] = []
    for part in parts:
        lower = part.lower()
        if lower in TOKEN_TITLE_OVERRIDES:
            rendered.append(TOKEN_TITLE_OVERRIDES[lower])
        elif part.isupper():
            rendered.append(part)
        else:
            rendered.append(part.capitalize())
    return " ".join(rendered)


def _job_display_name(job_id: str) -> str:
    normalized = str(job_id or "").strip()
    if not normalized:
        return ""
    return JOB_TITLE_OVERRIDES.get(normalized, _humanize(normalized).title())


def _wrap_lines(text: str, width: int) -> list[str]:
    normalized = str(text or "").strip()
    if not normalized:
        return []
    return wrap(normalized, width=width, break_long_words=False, break_on_hyphens=False) or [normalized]


def _artifact_slot_label(artifact_id: str) -> str:
    parts = str(artifact_id or "").strip().split("_")
    if len(parts) < 3:
        return _humanize(str(artifact_id or "")).title()
    channel, kind, ordinal = parts[0], parts[1], parts[2]
    channel_label = "Main" if channel == "main" else "Appendix"
    kind_label = "Figure" if kind == "figure" else "Table"
    return f"{channel_label} {kind_label} {ordinal}"


def _format_number(value: float | None, *, digits: int = 3, scientific_floor: float = 1e-4) -> str:
    if value is None:
        return ""
    if abs(value) < scientific_floor and value != 0:
        return f"{value:.2e}"
    rendered = f"{value:.{digits}f}"
    return "0.000" if rendered in {"-0.000", "0.000"} and digits == 3 else rendered


def _format_p_value(value: str) -> str:
    numeric = _coerce_float(value)
    if numeric is None:
        return value
    if numeric < 0.001:
        return "<0.001"
    return _format_number(numeric, digits=3, scientific_floor=1e-4)


def _significance_stars(value: str) -> str:
    numeric = _coerce_float(value)
    if numeric is None:
        return ""
    if numeric < 0.01:
        return "***"
    if numeric < 0.05:
        return "**"
    if numeric < 0.1:
        return "*"
    return ""


def _robustness_selected_estimates_path(paths: ProjectPaths, job_id: str) -> Path | None:
    summary_path = paths.manifests / f"{job_id}__robustness_summary.json"
    if not summary_path.exists():
        return None
    summary = _read_json(summary_path)
    recommended_k = int(summary.get("recommended_k", 0) or 0)
    if recommended_k <= 0:
        return None
    candidate = paths.output / "models" / f"{job_id}__robustness_k{recommended_k}_estimates.csv"
    if candidate.exists():
        return candidate
    return None


def _paper_tier2_source_path(paths: ProjectPaths) -> Path:
    return paths.output / "models" / PAPER_TIER2_SOURCE_ESTIMATES


def _scale_row_to_effect_per_100b(row: dict[str, str]) -> dict[str, str]:
    beta = _coerce_float(str(row.get("beta", "")))
    effect = _coerce_float(str(row.get("effect_per_100b_tdc", "")))
    scale = effect / beta if beta not in {None, 0.0} and effect is not None else None
    converted = row.copy()
    if effect is not None:
        converted["beta"] = str(effect)
    if scale is not None:
        for column in ("se", "lower95", "upper95"):
            value = _coerce_float(str(row.get(column, "")))
            if value is not None:
                converted[column] = str(value * scale)
    converted["response_scale"] = "usd_billions_per_100b_tdc"
    converted["source_beta"] = str(row.get("beta", ""))
    converted["source_normalized_beta"] = str(row.get("normalized_beta", ""))
    converted["source_effect_per_100b_tdc"] = str(row.get("effect_per_100b_tdc", ""))
    converted["job_id"] = PAPER_TIER2_JOB_ID
    return converted


def _build_paper_tier2_estimates(paths: ProjectPaths) -> Path | None:
    source_path = _paper_tier2_source_path(paths)
    if not source_path.exists():
        return None
    source_rows = _read_csv(source_path)
    wanted_outcomes = set(PAPER_TIER2_MAIN_TABLE_OUTCOMES)
    selected_rows = [
        _scale_row_to_effect_per_100b(row)
        for row in source_rows
        if str(row.get("surface", "")).strip() == PAPER_TIER2_SURFACE
        and str(row.get("treatment_label", "")).strip() == PAPER_TIER2_TREATMENT_LABEL
        and str(row.get("sample_label", "")).strip() == PAPER_TIER2_SAMPLE_LABEL
        and str(row.get("outcome", "")).strip() in wanted_outcomes
        and str(row.get("horizon", "")).strip() in set(PAPER_TIER2_HORIZONS)
    ]
    if not selected_rows:
        return None
    outcome_rank = {outcome: index for index, outcome in enumerate(PAPER_TIER2_MAIN_TABLE_OUTCOMES)}
    selected_rows.sort(key=lambda row: (outcome_rank.get(str(row.get("outcome", "")), 999), int(str(row.get("horizon", "0")))))
    estimates_path = paths.output / "models" / PAPER_TIER2_ESTIMATES
    fieldnames = list(source_rows[0].keys())
    for extra in ("response_scale", "source_beta", "source_normalized_beta", "source_effect_per_100b_tdc"):
        if extra not in fieldnames:
            fieldnames.append(extra)
    estimates_path.parent.mkdir(parents=True, exist_ok=True)
    with estimates_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)
    return estimates_path


def _paper_tier2_is_available(paths: ProjectPaths) -> bool:
    estimates_path = _build_paper_tier2_estimates(paths)
    if estimates_path is None:
        return False
    rows = _read_csv(estimates_path)
    outcomes = {str(row.get("outcome", "")).strip() for row in rows}
    horizons = {str(row.get("horizon", "")).strip() for row in rows}
    return set(PAPER_TIER2_MAIN_TABLE_OUTCOMES).issubset(outcomes) and "0" in horizons


def _paper_tier2_context() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {
            "job_id": PAPER_TIER2_JOB_ID,
            "treatment_id": PAPER_TIER2_TREATMENT_ID,
            "outcome_ids": PAPER_TIER2_MAIN_TABLE_OUTCOMES,
            "horizon_grid": [int(item) for item in PAPER_TIER2_HORIZONS],
            "sample_start": "2002Q1",
            "sample_end": "2025Q4",
            "response_type": "direct_at_h",
        },
        {
            "job_id": PAPER_TIER2_JOB_ID,
            "estimates_path": str(Path("output") / "models" / PAPER_TIER2_ESTIMATES),
            "response_type": "direct_at_h",
            "control_ids": PAPER_TIER2_CONTROLS,
            "covariance_estimators_used": ["newey_west"],
            "min_observations": 88,
            "max_observations": 96,
            "warning_rows": 0,
        },
    )


def _paper_tier2_contract_rows(contract_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    appendix_rows = [row for row in contract_rows if str(row.get("release_channel", "")).strip() == "appendix"]
    output_rows: list[dict[str, Any]] = [
        {
            "artifact_id": "main_figure_1",
            "artifact_kind": "figure",
            "release_channel": "main_text",
            "job_id": PAPER_TIER2_JOB_ID,
            "estimator": "lp",
            "output_family": "headline_identified",
            "display_spec": "impulse_response_grid",
            "outcome_ids": ",".join(PAPER_TIER2_MAIN_FIGURE_OUTCOMES),
            "horizons": ",".join(PAPER_TIER2_HORIZONS),
            "contract_source": "paper_tier2_selected_lag_override",
            "status": "ready",
            "title_override": "Long-history Tier 2 selected-lag pass-through",
            "extra_notes": (
                f"Run/specification: {PAPER_TIER2_TREATMENT_LABEL}. "
                "This paper-facing branch adds selected lagged credit and Treasury-rate controls before the K=100 factor tail.|"
                "Claim boundary: deposit pass-through is the headline; broad mortgage/core-loan crowding-out is weak after selected lags."
            ),
        },
        {
            "artifact_id": "main_table_1",
            "artifact_kind": "table",
            "release_channel": "main_text",
            "job_id": PAPER_TIER2_JOB_ID,
            "estimator": "lp",
            "output_family": "headline_identified",
            "display_spec": "coefficient_table",
            "outcome_ids": ",".join(PAPER_TIER2_MAIN_TABLE_OUTCOMES),
            "horizons": "0",
            "contract_source": "paper_tier2_selected_lag_override",
            "status": "ready",
            "title_override": "Long-history Tier 2 selected-lag h=0 coefficient table",
            "extra_notes": (
                f"Run/specification: {PAPER_TIER2_TREATMENT_LABEL}. "
                "Rows report h=0 effects in $B per +$100B TDC.|"
                "Claim boundary: consumer credit is a guarded candidate margin, not a broad crowding-out headline."
            ),
        },
    ]
    old_main_rows = [
        row
        for row in contract_rows
        if str(row.get("release_channel", "")).strip() == "main_text"
        and str(row.get("job_id", "")).strip() == "baseline_tdc_lp_deposits"
    ]
    legacy_index = 1
    legacy_table_index = 1
    for row in old_main_rows:
        legacy_row = row.copy()
        if str(row.get("artifact_kind", "")).strip() == "figure":
            legacy_row["artifact_id"] = f"appendix_figure_{legacy_index}"
            legacy_index += 1
        else:
            legacy_row["artifact_id"] = f"appendix_table_{legacy_table_index}"
            legacy_table_index += 1
        legacy_row["release_channel"] = "appendix"
        legacy_row["contract_source"] = "legacy_baseline_k200_sensitivity"
        legacy_row["title_override"] = "Older baseline K=200 sensitivity"
        legacy_row["extra_notes"] = (
            "Relabeled legacy surface: baseline_tdc_lp_deposits with the K=200 screened branch. "
            "This is preserved as appendix/sensitivity output and is no longer the paper-facing main surface."
        )
        output_rows.append(legacy_row)
    for row in appendix_rows:
        appendix_row = row.copy()
        if str(appendix_row.get("artifact_kind", "")).strip() == "table":
            appendix_row["artifact_id"] = f"appendix_table_{legacy_table_index}"
            legacy_table_index += 1
        output_rows.append(appendix_row)
    return output_rows


def _dedupe_text(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = str(item or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _current_quarter_label(today: date | None = None) -> str:
    current = today or date.today()
    quarter = ((current.month - 1) // 3) + 1
    return f"{current.year}Q{quarter}"


def _selected_control_ids(
    *,
    paths: ProjectPaths,
    job_id: str,
    estimation_summary: dict[str, Any],
    source_estimates_path: Path,
) -> list[str]:
    controls = [str(item) for item in estimation_summary.get("control_ids", []) if str(item).strip()]
    estimates_name = source_estimates_path.name
    if "__robustness_k" not in estimates_name:
        return _dedupe_text(controls)
    robustness_summary_path = paths.manifests / f"{job_id}__robustness_summary.json"
    if not robustness_summary_path.exists():
        return _dedupe_text(controls)
    robustness_summary = _read_json(robustness_summary_path)
    base_controls = [str(item) for item in robustness_summary.get("base_controls", []) if str(item).strip()]
    factor_controls = [str(item) for item in robustness_summary.get("recommended_factor_ids", []) if str(item).strip()]
    combined = base_controls + factor_controls
    return _dedupe_text(combined or controls)


def _observed_outcome_ids(rows: list[dict[str, str]], preferred_order: list[str]) -> list[str]:
    observed = {
        str(row.get("outcome", "")).strip()
        for row in rows
        if str(row.get("outcome", "")).strip()
    }
    ordered = [outcome for outcome in preferred_order if outcome in observed]
    extras = sorted(observed.difference(ordered))
    return ordered + extras


def _observed_horizons(rows: list[dict[str, str]], preferred_order: list[str]) -> list[str]:
    observed = {
        str(row.get("horizon", "")).strip()
        for row in rows
        if str(row.get("horizon", "")).strip()
    }
    ordered = [horizon for horizon in preferred_order if horizon in observed]
    extras = sorted(observed.difference(ordered), key=lambda item: int(item))
    return ordered + extras


def _artifact_notes(
    *,
    paths: ProjectPaths,
    design_manifest: dict[str, Any],
    estimation_summary: dict[str, Any],
    artifact_row: dict[str, str],
    source_estimates_path: Path,
) -> list[str]:
    job_id = str(artifact_row.get("job_id", "")).strip()
    sample_start = str(design_manifest.get("sample_start", "")).strip()
    sample_end = str(design_manifest.get("sample_end", "")).strip()
    control_ids = _selected_control_ids(
        paths=paths,
        job_id=job_id,
        estimation_summary=estimation_summary,
        source_estimates_path=source_estimates_path,
    )
    scale_note = "Scale: coefficients are responses of quarterly outcomes to the GDP-scaled TDC flow."
    if job_id == PAPER_TIER2_JOB_ID:
        scale_note = "Scale: coefficients are $B responses per +$100B TDC."
    notes = [
        f"Treatment: {_humanize(str(design_manifest.get('treatment_id', '')).strip())}",
        f"Response: {_humanize(str(estimation_summary.get('response_type', '')).strip() or str(design_manifest.get('response_type', '')).strip())}",
        f"Controls: {', '.join(_humanize(str(item)) for item in control_ids) or 'none'}",
        f"Covariance: {', '.join(_humanize(str(item)) for item in estimation_summary.get('covariance_estimators_used', []) if str(item).strip()) or 'unknown'}",
        f"Sample span: {sample_start} to {sample_end}",
        f"Observations: {estimation_summary.get('min_observations', 0)} to {estimation_summary.get('max_observations', 0)}",
        scale_note,
    ]
    warning_rows = int(estimation_summary.get("warning_rows", 0) or 0)
    if warning_rows > 0:
        notes.append(f"Warnings: {warning_rows} estimated rows flagged")
    estimates_name = source_estimates_path.name
    if "__robustness_k" in estimates_name:
        selected_k = estimates_name.split("__robustness_k", 1)[1].split("_", 1)[0]
        notes.append(f"Displayed branch: K={selected_k} screened branch")
    elif "__robustness_baseline" in estimates_name:
        notes.append("Displayed branch: baseline macro block")
    notes.extend([item.strip() for item in str(artifact_row.get("extra_notes", "")).split("|") if item.strip()])
    if sample_end and sample_end == _current_quarter_label():
        notes.append(
            f"Sample endpoint note: {sample_end} is the latest labeled quarter as of {date.today().isoformat()} and may reflect an in-progress quarter endpoint."
        )
    notes.append(f"Artifact source: {_job_display_name(str(artifact_row.get('job_id', '')).strip())}")
    return [note for note in notes if note.split(": ", 1)[-1]]


def _artifact_title(artifact_row: dict[str, str]) -> str:
    explicit_title = str(artifact_row.get("title_override", "")).strip()
    if explicit_title:
        return explicit_title
    job_title = _job_display_name(str(artifact_row.get("job_id", "")).strip())
    artifact_kind = str(artifact_row.get("artifact_kind", "")).strip()
    release_channel = str(artifact_row.get("release_channel", "")).strip()
    if artifact_kind == "figure":
        return job_title
    if release_channel == "appendix":
        return f"{job_title} estimates"
    return f"{job_title} coefficient table"


def _artifact_download_links(
    *,
    preview_name: str,
    primary_name: str,
    secondary_name: str,
    manifest_name: str,
) -> list[tuple[str, str]]:
    links = [
        ("Preview", preview_name),
        ("Primary file", primary_name),
        ("Manifest", manifest_name),
    ]
    if secondary_name:
        links.append(("Data export", secondary_name))
    return links


def _artifact_subtitle(
    *,
    artifact_row: dict[str, str],
    design_manifest: dict[str, Any],
    estimation_summary: dict[str, Any],
    selected_rows: list[dict[str, str]],
) -> str:
    estimator = ESTIMATOR_LABELS.get(str(artifact_row.get("estimator", "")).strip(), str(artifact_row.get("estimator", "")).strip())
    family_label = _humanize(str(artifact_row.get("output_family", "")).strip())
    sample_start = str(design_manifest.get("sample_start", "")).strip()
    sample_end = str(design_manifest.get("sample_end", "")).strip()
    preferred_outcomes = _split_csv(str(artifact_row.get("outcome_ids", "")))
    outcome_count = len(_observed_outcome_ids(selected_rows, preferred_outcomes))
    n_values = [int(str(row.get("n", "0") or "0")) for row in selected_rows if str(row.get("n", "")).strip()]
    if n_values:
        observation_range = f"{min(n_values)}-{max(n_values)} obs"
    else:
        observation_range = f"{estimation_summary.get('min_observations', 0)}-{estimation_summary.get('max_observations', 0)} obs"
    pieces = [estimator]
    if family_label:
        pieces.append(family_label)
    if sample_start or sample_end:
        pieces.append(f"{sample_start} to {sample_end}".strip())
    pieces.append(f"{outcome_count} outcomes")
    pieces.append(observation_range)
    return " | ".join(piece for piece in pieces if piece)


def _artifact_caption(
    *,
    artifact_row: dict[str, str],
    design_manifest: dict[str, Any],
    selected_rows: list[dict[str, str]],
) -> str:
    preferred_outcomes = _split_csv(str(artifact_row.get("outcome_ids", "")))
    preferred_horizons = _split_csv(str(artifact_row.get("horizons", "")))
    outcomes = [_humanize(item) for item in _observed_outcome_ids(selected_rows, preferred_outcomes)]
    horizons = _observed_horizons(selected_rows, preferred_horizons)
    treatment = _humanize(str(design_manifest.get("treatment_id", "")).strip())
    horizon_text = ", ".join(horizons)
    if str(artifact_row.get("artifact_kind", "")).strip() == "figure":
        return (
            f"Impulse responses for {', '.join(outcomes)} to {treatment}. "
            f"Points are local-projection estimates and whiskers show 95% confidence intervals over horizons {horizon_text}."
        )
    if str(artifact_row.get("job_id", "")).strip() == PAPER_TIER2_JOB_ID:
        return (
            f"H=0 coefficient table for {', '.join(outcomes)} in response to {treatment}. "
            "Entries are $B per +$100B TDC, with p-values and observations."
        )
    return (
        f"Coefficient table for {', '.join(outcomes)} in response to {treatment}. "
        f"Reported horizons are {horizon_text} with the corresponding standard errors and confidence intervals."
    )


def _render_svg_figure(
    title: str,
    subtitle: str,
    caption: str,
    rows: list[dict[str, str]],
    notes: list[str],
    outcomes_order: list[str],
) -> str:
    outcomes = [outcome for outcome in outcomes_order if any(str(row.get("outcome", "")).strip() == outcome for row in rows)]
    panel_rows = max(1, (len(outcomes) + 1) // 2)
    title_lines = _wrap_lines(title, 62)
    subtitle_lines = _wrap_lines(subtitle, 100)
    caption_lines = _wrap_lines(caption, 112)
    note_lines = [
        wrapped_line
        for note in notes
        for wrapped_line in _wrap_lines(note, 115)
    ]
    width = 1020
    panel_width = 440
    panel_height = 220
    gutter_x = 40
    gutter_y = 40
    header_height = 36 + (len(title_lines) * 28) + (len(subtitle_lines) * 16) + (len(caption_lines) * 16) + 18
    footer_height = 34 + (len(note_lines) * 16)
    height = header_height + panel_rows * (panel_height + gutter_y) + footer_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf6"/>',
        '<rect x="36" y="28" width="72" height="4" rx="2" fill="#204b57"/>',
    ]
    cursor_y = 54
    for line in title_lines:
        parts.append(f'<text x="36" y="{cursor_y}" font-size="24" font-family="Georgia, serif" fill="#1d1d1b">{html.escape(line)}</text>')
        cursor_y += 28
    for line in subtitle_lines:
        parts.append(f'<text x="36" y="{cursor_y}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="#555">{html.escape(line)}</text>')
        cursor_y += 16
    for line in caption_lines:
        parts.append(f'<text x="36" y="{cursor_y}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="#555">{html.escape(line)}</text>')
        cursor_y += 16
    for outcome_index, outcome in enumerate(outcomes):
        outcome_rows = [
            row
            for row in rows
            if str(row.get("outcome", "")).strip() == outcome
            and _coerce_float(str(row.get("beta", ""))) is not None
            and _coerce_float(str(row.get("lower95", ""))) is not None
            and _coerce_float(str(row.get("upper95", ""))) is not None
        ]
        if not outcome_rows:
            continue
        outcome_rows.sort(key=lambda row: int(str(row.get("horizon", "0"))))
        ci_min = min(_coerce_float(str(row["lower95"])) or 0.0 for row in outcome_rows)
        ci_max = max(_coerce_float(str(row["upper95"])) or 0.0 for row in outcome_rows)
        if ci_max <= ci_min:
            ci_min -= 1.0
            ci_max += 1.0
        padding = max((ci_max - ci_min) * 0.1, 1e-9)
        y_min = ci_min - padding
        y_max = ci_max + padding
        panel_col = outcome_index % 2
        panel_row = outcome_index // 2
        panel_x = 36 + panel_col * (panel_width + gutter_x)
        panel_y = header_height + panel_row * (panel_height + gutter_y)
        plot_left = panel_x + 52
        plot_right = panel_x + panel_width - 16
        plot_top = panel_y + 24
        plot_bottom = panel_y + panel_height - 36
        plot_width = plot_right - plot_left
        plot_height = plot_bottom - plot_top
        horizons = [int(str(row.get("horizon", "0"))) for row in outcome_rows]
        horizon_min = min(horizons)
        horizon_max = max(horizons)
        horizon_span = max(horizon_max - horizon_min, 1)

        def x_pos(horizon: int) -> float:
            return plot_left + ((horizon - horizon_min) / horizon_span) * plot_width

        def y_pos(value: float) -> float:
            return plot_bottom - ((value - y_min) / (y_max - y_min)) * plot_height

        zero_y = y_pos(0.0) if y_min <= 0.0 <= y_max else None
        line_points = " ".join(
            f"{x_pos(int(str(row['horizon']))):.2f},{y_pos(_coerce_float(str(row['beta'])) or 0.0):.2f}"
            for row in outcome_rows
        )
        parts.extend(
            [
                f'<rect x="{panel_x}" y="{panel_y}" width="{panel_width}" height="{panel_height}" rx="10" fill="#fffef9" stroke="#d9d2c3"/>',
                f'<rect x="{panel_x}" y="{panel_y}" width="{panel_width}" height="10" rx="10" fill="#ece6d8" stroke="none"/>',
                f'<text x="{panel_x + 16}" y="{panel_y + 18}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#222">{html.escape(_humanize(outcome))}</text>',
                f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#888" stroke-width="1"/>',
                f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#888" stroke-width="1"/>',
            ]
        )
        if zero_y is not None:
            parts.append(
                f'<line x1="{plot_left}" y1="{zero_y:.2f}" x2="{plot_right}" y2="{zero_y:.2f}" stroke="#c9c1b0" stroke-dasharray="4 4" stroke-width="1"/>'
            )
        tick_values = [y_min, 0.0 if y_min <= 0.0 <= y_max else (y_min + y_max) / 2.0, y_max]
        for tick_value in tick_values:
            tick_y = y_pos(tick_value)
            parts.append(
                f'<text x="{panel_x + 8}" y="{tick_y + 4:.2f}" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="#666">{html.escape(_format_number(tick_value, digits=3, scientific_floor=1e-3))}</text>'
            )
        for row in outcome_rows:
            horizon = int(str(row["horizon"]))
            beta = _coerce_float(str(row["beta"])) or 0.0
            lower = _coerce_float(str(row["lower95"])) or 0.0
            upper = _coerce_float(str(row["upper95"])) or 0.0
            p_value = _coerce_float(str(row.get("p_value_normal", "")))
            point_x = x_pos(horizon)
            parts.append(
                f'<line x1="{point_x:.2f}" y1="{y_pos(lower):.2f}" x2="{point_x:.2f}" y2="{y_pos(upper):.2f}" stroke="#5c6f7b" stroke-width="2"/>'
            )
            point_fill = "#204b57" if p_value is not None and p_value < 0.1 else "#7f8d93"
            parts.append(f'<circle cx="{point_x:.2f}" cy="{y_pos(beta):.2f}" r="4" fill="{point_fill}"/>')
            parts.append(
                f'<text x="{point_x:.2f}" y="{plot_bottom + 18:.2f}" text-anchor="middle" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="#666">{horizon}</text>'
            )
        parts.append(f'<polyline points="{line_points}" fill="none" stroke="#204b57" stroke-width="2.5"/>')
        parts.append(f'<text x="{(plot_left + plot_right) / 2:.2f}" y="{plot_bottom + 32:.2f}" text-anchor="middle" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="#666">Quarters after TDC change</text>')
    notes_y = header_height + panel_rows * (panel_height + gutter_y)
    parts.append(f'<text x="36" y="{notes_y:.2f}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="#444">Notes</text>')
    for note_index, note in enumerate(note_lines, start=1):
        parts.append(
            f'<text x="36" y="{notes_y + 18 + (note_index * 14):.2f}" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="#555">{html.escape(note)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _select_table_columns(rows: list[dict[str, str]], display_spec: str) -> list[str]:
    preferred = ["outcome", "horizon", "beta", "se", "lower95", "upper95", "p_value_normal", "n"]
    if rows and all(str(row.get("job_id", "")).strip() == PAPER_TIER2_JOB_ID for row in rows):
        preferred = ["outcome", "horizon", "beta", "p_value_normal", "n", "significance"]
        available = set(rows[0].keys())
        return [column for column in preferred if column in available or column == "significance"]
    if display_spec == "supporting_table":
        preferred.append("rsquared")
    else:
        preferred.append("significance")
    available = set(rows[0].keys()) if rows else set()
    return [column for column in preferred if column in available or column == "significance"]


def _format_table_value(column: str, value: str) -> str:
    numeric_columns = {"beta", "se", "lower95", "upper95", "p_value_normal", "rsquared"}
    if column == "significance":
        return value
    if column in {"outcome", "treatment_id", "response_type"}:
        return _humanize(value)
    if column not in numeric_columns:
        return value
    numeric_value = _coerce_float(value)
    if numeric_value is None:
        return value
    if column == "p_value_normal":
        return _format_number(numeric_value, digits=4, scientific_floor=1e-4)
    return _format_number(numeric_value, digits=3, scientific_floor=1e-4)


def _render_table_markdown(title: str, subtitle: str, caption: str, rows: list[dict[str, str]], columns: list[str], notes: list[str]) -> str:
    lines = [f"# {title}", "", subtitle, "", caption, "", f"Generated at: {utc_now_iso()}", ""]
    if notes:
        lines.append("## Notes")
        lines.extend([f"- {note}" for note in notes])
        lines.append("")
    table_rows = []
    for row in rows:
        table_row = row.copy()
        table_row["significance"] = _significance_stars(str(row.get("p_value_normal", "")))
        table_rows.append(table_row)
    if not table_rows:
        lines.append("No rows available.")
        return "\n".join(lines) + "\n"
    lines.append("| " + " | ".join(_humanize(column) for column in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in table_rows:
        values = [_format_table_value(column, str(row.get(column, "")).replace("\n", " ")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def _render_table_html(
    title: str,
    slot_label: str,
    subtitle: str,
    caption: str,
    rows: list[dict[str, str]],
    columns: list[str],
    notes: list[str],
    links: list[tuple[str, str]],
) -> str:
    note_items = "".join(f"<li>{html.escape(note)}</li>" for note in notes)
    header_cells = "".join(f"<th>{html.escape(_humanize(column))}</th>" for column in columns)
    link_items = "".join(
        f'<a href="{html.escape(href)}">{html.escape(label)}</a>'
        for label, href in links
    )
    body_rows = []
    for row in rows:
        table_row = row.copy()
        table_row["significance"] = _significance_stars(str(row.get("p_value_normal", "")))
        cells = "".join(
            f"<td>{html.escape(_format_table_value(column, str(table_row.get(column, '')).replace(chr(10), ' ')))}</td>"
            for column in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")
    body_html = "".join(body_rows)
    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en"><head><meta charset="utf-8">',
            f"<title>{html.escape(title)}</title>",
            "<style>body{font-family:Georgia,serif;background:#fbfaf6;color:#1d1d1b;margin:0;line-height:1.45;}main{max-width:1160px;margin:0 auto;padding:36px 32px 52px;}table{border-collapse:collapse;width:100%;font-family:Helvetica,Arial,sans-serif;font-size:13px;background:#fffef9;}th,td{border:1px solid #d9d2c3;padding:8px 10px;text-align:left;}th{background:#f2eee4;}td:nth-child(n+2){text-align:right;}td:last-child,th:last-child{text-align:center;}ul{font-family:Helvetica,Arial,sans-serif;color:#444;}h1{margin:6px 0 10px;} .stamp,.subtitle,.caption,.slot,.nav a{font-family:Helvetica,Arial,sans-serif;color:#666;font-size:12px;} .slot{letter-spacing:0.08em;text-transform:uppercase;} .caption{max-width:900px;margin-bottom:18px;} .panel{background:#fffef9;border:1px solid #ddd3c2;border-radius:16px;padding:18px 20px;box-shadow:0 12px 32px rgba(29,29,27,0.04);} .nav{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0 18px;} .nav a{color:#204b57;text-decoration:none;} .nav a:hover{text-decoration:underline;}</style></head><body>",
            "<main>",
            f'<div class="slot">{html.escape(slot_label)}</div>',
            f"<h1>{html.escape(title)}</h1>",
            f'<p class="subtitle">{html.escape(subtitle)}</p>',
            f'<p class="caption">{html.escape(caption)}</p>',
            f'<div class="stamp">Generated at: {html.escape(utc_now_iso())}</div>',
            f'<div class="nav">{link_items}</div>',
            '<section class="panel">',
            "<h2>Notes</h2>",
            f"<ul>{note_items}</ul>",
            f"<table><thead><tr>{header_cells}</tr></thead><tbody>{body_html}</tbody></table>",
            "</section>",
            "</main></body></html>",
        ]
    )


def _render_figure_html(
    title: str,
    slot_label: str,
    subtitle: str,
    caption: str,
    svg_path: Path,
    notes: list[str],
    links: list[tuple[str, str]],
) -> str:
    note_items = "".join(f"<li>{html.escape(note)}</li>" for note in notes)
    link_items = "".join(
        f'<a href="{html.escape(href)}">{html.escape(label)}</a>'
        for label, href in links
    )
    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en"><head><meta charset="utf-8">',
            f"<title>{html.escape(title)}</title>",
            "<style>body{font-family:Georgia,serif;background:#fbfaf6;color:#1d1d1b;margin:0;line-height:1.45;}main{max-width:1160px;margin:0 auto;padding:36px 32px 52px;}img{max-width:100%;border:1px solid #d9d2c3;background:#fffef9;box-shadow:0 10px 30px rgba(29,29,27,0.06);}ul{font-family:Helvetica,Arial,sans-serif;color:#444;} .stamp,.subtitle,.caption,.slot,.nav a,.legend{font-family:Helvetica,Arial,sans-serif;color:#666;font-size:12px;} .slot{letter-spacing:0.08em;text-transform:uppercase;} .caption{max-width:900px;margin-bottom:18px;} .layout{display:grid;grid-template-columns:minmax(0,1.75fr) minmax(280px,0.9fr);gap:24px;align-items:start;} .panel{background:#fffef9;border:1px solid #ddd3c2;border-radius:16px;padding:18px 20px;box-shadow:0 12px 32px rgba(29,29,27,0.04);} .nav{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0 18px;} .nav a{color:#204b57;text-decoration:none;} .nav a:hover{text-decoration:underline;} @media (max-width:900px){.layout{grid-template-columns:1fr;}}</style></head><body>",
            "<main>",
            f'<div class="slot">{html.escape(slot_label)}</div>',
            f"<h1>{html.escape(title)}</h1>",
            f'<p class="subtitle">{html.escape(subtitle)}</p>',
            f'<p class="caption">{html.escape(caption)}</p>',
            f'<div class="stamp">Generated at: {html.escape(utc_now_iso())}</div>',
            f'<div class="nav">{link_items}</div>',
            '<section class="layout">',
            f'<div class="panel"><p class="legend">Filled dark points indicate estimates with p &lt; 0.10.</p><p><img src="{html.escape(svg_path.name)}" alt="{html.escape(title)}"></p></div>',
            f'<aside class="panel"><h2>Notes</h2><ul>{note_items}</ul></aside>',
            "</section>",
            "</main></body></html>",
        ]
    )


def _write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _estimates_path_for_job(paths: ProjectPaths, job_id: str) -> Path:
    if job_id == PAPER_TIER2_JOB_ID:
        estimates_path = _build_paper_tier2_estimates(paths)
        if estimates_path is None:
            raise FileNotFoundError(f"Could not build {PAPER_TIER2_JOB_ID} from {_paper_tier2_source_path(paths)}")
        return estimates_path
    selected = _robustness_selected_estimates_path(paths, job_id)
    if selected is not None:
        return selected
    summary_path = paths.manifests / f"{job_id}__estimation_summary.json"
    summary = _read_json(summary_path)
    return Path(str(summary["estimates_path"]))


def _artifact_context(paths: ProjectPaths, job_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if job_id == PAPER_TIER2_JOB_ID:
        return _paper_tier2_context()
    design_manifest = _read_json(paths.manifests / f"{job_id}__design_manifest.json")
    estimation_summary = _read_json(paths.manifests / f"{job_id}__estimation_summary.json")
    return design_manifest, estimation_summary


def build_release_artifacts(paths: ProjectPaths) -> ReleaseArtifactBuildResult:
    artifact_contract = build_release_artifact_contract(paths)
    contract_rows = _read_json(artifact_contract.summary_path).get("rows", [])
    if _paper_tier2_is_available(paths):
        contract_rows = _paper_tier2_contract_rows(contract_rows)
    artifacts_root = paths.output / "artifacts"
    if artifacts_root.exists():
        shutil.rmtree(artifacts_root)
    rows: list[dict[str, str]] = []
    figure_artifacts = 0
    table_artifacts = 0
    gallery_items: list[dict[str, str]] = []

    for artifact_row in contract_rows:
        artifact_id = str(artifact_row.get("artifact_id", "")).strip()
        job_id = str(artifact_row.get("job_id", "")).strip()
        artifact_kind = str(artifact_row.get("artifact_kind", "")).strip()
        release_channel = str(artifact_row.get("release_channel", "")).strip()
        display_spec = str(artifact_row.get("display_spec", "")).strip()
        title = _artifact_title(artifact_row)
        slot_label = _artifact_slot_label(artifact_id)
        artifact_dir = artifacts_root / release_channel / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        estimates_path = _estimates_path_for_job(paths, job_id)
        design_manifest, estimation_summary = _artifact_context(paths, job_id)
        estimate_rows = _read_csv(estimates_path)
        outcomes_order = _split_csv(str(artifact_row.get("outcome_ids", "")))
        selected_horizons = set(_split_csv(str(artifact_row.get("horizons", ""))))
        selected_rows = [
            row
            for row in estimate_rows
            if str(row.get("outcome", "")).strip() in set(outcomes_order)
            and str(row.get("horizon", "")).strip() in selected_horizons
        ]
        outcome_rank = {outcome: index for index, outcome in enumerate(outcomes_order)}
        selected_rows.sort(key=lambda row: (outcome_rank.get(str(row.get("outcome", "")), 999), int(str(row.get("horizon", "0")))))
        observed_outcomes = _observed_outcome_ids(selected_rows, outcomes_order)
        observed_horizons = _observed_horizons(selected_rows, _split_csv(str(artifact_row.get("horizons", ""))))
        notes = _artifact_notes(
            paths=paths,
            design_manifest=design_manifest,
            estimation_summary=estimation_summary,
            artifact_row=artifact_row,
            source_estimates_path=estimates_path,
        )
        subtitle = _artifact_subtitle(
            artifact_row=artifact_row,
            design_manifest=design_manifest,
            estimation_summary=estimation_summary,
            selected_rows=selected_rows,
        )
        caption = _artifact_caption(
            artifact_row=artifact_row,
            design_manifest=design_manifest,
            selected_rows=selected_rows,
        )
        manifest_path = artifact_dir / "artifact_manifest.json"
        primary_path: Path
        html_path: Path
        secondary_path: Path | None = None

        if artifact_kind == "figure":
            primary_path = artifact_dir / f"{artifact_id}.svg"
            primary_path.write_text(_render_svg_figure(title, subtitle, caption, selected_rows, notes, outcomes_order), encoding="utf-8")
            html_path = artifact_dir / f"{artifact_id}.html"
            html_path.write_text(
                _render_figure_html(
                    title,
                    slot_label,
                    subtitle,
                    caption,
                    primary_path,
                    notes,
                    _artifact_download_links(
                        preview_name=html_path.name,
                        primary_name=primary_path.name,
                        secondary_name="",
                        manifest_name="artifact_manifest.json",
                    ),
                ),
                encoding="utf-8",
            )
            figure_artifacts += 1
        else:
            columns = _select_table_columns(selected_rows, display_spec)
            csv_path = artifact_dir / f"{artifact_id}.csv"
            md_path = artifact_dir / f"{artifact_id}.md"
            _write_csv(csv_path, selected_rows, columns)
            md_path.write_text(_render_table_markdown(title, subtitle, caption, selected_rows, columns, notes), encoding="utf-8")
            html_path = artifact_dir / f"{artifact_id}.html"
            html_path.write_text(
                _render_table_html(
                    title,
                    slot_label,
                    subtitle,
                    caption,
                    selected_rows,
                    columns,
                    notes,
                    _artifact_download_links(
                        preview_name=html_path.name,
                        primary_name=md_path.name,
                        secondary_name=csv_path.name,
                        manifest_name="artifact_manifest.json",
                    ),
                ),
                encoding="utf-8",
            )
            primary_path = md_path
            secondary_path = csv_path
            table_artifacts += 1

        manifest_payload = {
            "artifact_id": artifact_id,
            "artifact_kind": artifact_kind,
            "release_channel": release_channel,
            "job_id": job_id,
            "display_spec": display_spec,
            "title": title,
            "slot_label": slot_label,
            "subtitle": subtitle,
            "caption": caption,
            "outcome_ids": observed_outcomes,
            "horizons": observed_horizons,
            "source_estimates_path": str(estimates_path),
            "primary_path": str(primary_path),
            "html_path": str(html_path),
            "secondary_path": str(secondary_path) if secondary_path else "",
            "notes": notes,
            "generated_at": utc_now_iso(),
        }
        write_json(manifest_path, manifest_payload)
        gallery_items.append(
            {
                "artifact_id": artifact_id,
                "artifact_kind": artifact_kind,
                "release_channel": release_channel,
                "slot_label": slot_label,
                "job_id": job_id,
                "title": title,
                "subtitle": subtitle,
                "caption": caption,
                "outcome_ids": ",".join(observed_outcomes),
                "horizons": ",".join(observed_horizons),
                "preview_href": str(html_path.relative_to(artifacts_root)),
                "primary_href": str(primary_path.relative_to(artifacts_root)),
                "secondary_href": str(secondary_path.relative_to(artifacts_root)) if secondary_path else "",
                "manifest_href": str(manifest_path.relative_to(artifacts_root)),
            }
        )
        rows.append(
            {
                "artifact_id": artifact_id,
                "artifact_kind": artifact_kind,
                "release_channel": release_channel,
                "job_id": job_id,
                "display_spec": display_spec,
                "title": title,
                "slot_label": slot_label,
                "subtitle": subtitle,
                "caption": caption,
                "outcome_ids": ",".join(observed_outcomes),
                "horizons": ",".join(observed_horizons),
                "primary_path": str(primary_path),
                "html_path": str(html_path),
                "secondary_path": str(secondary_path) if secondary_path else "",
                "manifest_path": str(manifest_path),
                "source_estimates_path": str(estimates_path),
                "status": "rendered",
            }
        )

    summary = {
        "generated_at": utc_now_iso(),
        "artifacts_built": len(rows),
        "figure_artifacts": figure_artifacts,
        "table_artifacts": table_artifacts,
        "rows": rows,
    }
    summary_path = paths.reports / "release_artifact_build.json"
    summary_csv_path = paths.reports / "release_artifact_build.csv"
    gallery_path = artifacts_root / "index.html"
    write_json(summary_path, summary)
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else [
            "artifact_id",
            "artifact_kind",
            "release_channel",
            "job_id",
            "display_spec",
            "primary_path",
            "html_path",
            "secondary_path",
            "manifest_path",
            "source_estimates_path",
            "status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    def render_gallery_cards(release_channel: str) -> list[str]:
        items = [item for item in gallery_items if item["release_channel"] == release_channel]
        cards: list[str] = []
        for item in items:
            links = [
                f'<a href="{html.escape(item["preview_href"])}">Preview</a>',
                f'<a href="{html.escape(item["primary_href"])}">Primary file</a>',
                f'<a href="{html.escape(item["manifest_href"])}">Manifest</a>',
            ]
            if item["secondary_href"]:
                links.append(f'<a href="{html.escape(item["secondary_href"])}">Data export</a>')
            cards.append(
                "\n".join(
                    [
                        '<article class="artifact-card">',
                        f'<div class="artifact-meta">{html.escape(item["slot_label"])} · {html.escape(_humanize(item["artifact_kind"]))}</div>',
                        f'<h3>{html.escape(item["title"])}</h3>',
                        f'<p class="subtitle">{html.escape(item["subtitle"])}</p>',
                        f'<p class="caption">{html.escape(item["caption"])}</p>',
                        f'<div class="artifact-links">{" ".join(links)}</div>',
                        "</article>",
                    ]
                )
            )
        return cards

    gallery_path.parent.mkdir(parents=True, exist_ok=True)
    gallery_path.write_text(
        "\n".join(
            [
                "<!DOCTYPE html>",
                '<html lang="en"><head><meta charset="utf-8"><title>Release Artifact Gallery</title>',
                "<style>body{font-family:Georgia,serif;background:#f7f4ec;color:#1d1d1b;margin:0;}main{max-width:1200px;margin:0 auto;padding:40px 32px 56px;}h1{font-size:34px;margin:0 0 8px;}h2{margin:36px 0 16px;font-size:22px;}p,div{line-height:1.5;}a{color:#204b57;text-decoration:none;font-family:Helvetica,Arial,sans-serif;font-size:12px;}a:hover{text-decoration:underline;} .lede,.stamp,.subtitle,.caption,.artifact-meta{font-family:Helvetica,Arial,sans-serif;color:#5a5a55;font-size:12px;} .summary-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:24px 0 36px;} .summary-card{background:#fffef9;border:1px solid #ddd3c2;border-radius:14px;padding:16px;} .summary-value{font-size:28px;color:#1d1d1b;font-family:Georgia,serif;} .artifact-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;} .artifact-card{background:#fffef9;border:1px solid #ddd3c2;border-radius:16px;padding:18px;box-shadow:0 12px 32px rgba(29,29,27,0.04);} .artifact-card h3{margin:8px 0 10px;font-size:20px;} .caption{margin-bottom:14px;} .artifact-links{display:flex;flex-wrap:wrap;gap:12px;} .artifact-meta{letter-spacing:0.04em;text-transform:uppercase;}</style></head><body>",
                "<main>",
                "<h1>Release Artifact Gallery</h1>",
                '<p class="lede">Committed Release 1 artifacts rendered from the current contract, estimates, and design manifests.</p>',
                f'<div class="stamp">Generated at: {html.escape(utc_now_iso())}</div>',
                '<section class="summary-grid">',
                f'<div class="summary-card"><div class="summary-value">{figure_artifacts}</div><div class="artifact-meta">Main figures</div></div>',
                f'<div class="summary-card"><div class="summary-value">{table_artifacts}</div><div class="artifact-meta">Tables</div></div>',
                f'<div class="summary-card"><div class="summary-value">{len(rows)}</div><div class="artifact-meta">Rendered artifacts</div></div>',
                "</section>",
                "<h2>Main Text</h2>",
                '<section class="artifact-grid">',
                *render_gallery_cards("main_text"),
                "</section>",
                "<h2>Appendix</h2>",
                '<section class="artifact-grid">',
                *render_gallery_cards("appendix"),
                "</section>",
                "</main></body></html>",
            ]
        ),
        encoding="utf-8",
    )
    return ReleaseArtifactBuildResult(
        summary_path=summary_path,
        summary_csv_path=summary_csv_path,
        artifacts_built=len(rows),
        figure_artifacts=figure_artifacts,
        table_artifacts=table_artifacts,
        gallery_path=gallery_path,
    )
