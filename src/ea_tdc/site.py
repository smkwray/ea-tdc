from __future__ import annotations

import csv
import html
import json
import shutil
from urllib.parse import quote
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any

from ea_tdc.artifacts import build_release_artifacts
from ea_tdc.paths import ProjectPaths
from ea_tdc.reporting import build_component_sidecar_screening, build_robustness_snapshot, build_stage_completion_closeout
from ea_tdc.sanitize import sanitize_output_paths
from ea_tdc.utils import utc_now_iso, write_json


@dataclass(frozen=True)
class SiteBuildResult:
    index_path: Path
    sidecar_index_path: Path
    summary_path: Path
    copied_artifacts: int
    copied_reports: int
    copied_models: int


MAIN_JOB_IDS = [
    "baseline_tdc_lp_deposits",
]

SIDECAR_JOB_IDS = [
    "baseline_tdc_lp_credit_spreads",
    "baseline_tdc_lp_inflation",
    "baseline_tdc_lp_fx",
    "baseline_tdc_lp_private_assets",
]

ROBUSTNESS_JOB_IDS = [
    "baseline_tdc_lp_deposits",
    "baseline_tdc_lp_credit_spreads",
    "baseline_tdc_lp_inflation",
    "baseline_tdc_lp_fx",
    "baseline_tdc_lp_private_assets",
    "baseline_tdc_lp_liquidity_decomposition",
]

JOB_META: dict[str, dict[str, str]] = {
    "baseline_tdc_lp_deposits": {
        "title": "Deposits reallocate first",
        "subtitle": "Matched deposits rise on impact while another deposit component leans against them.",
        "summary": "The cleanest current transmission result is deposit-side. The baseline TDC estimate raises matched deposits early, but another deposit component moves the other way, which looks more like balance-sheet reallocation than uniform expansion.",
        "kicker": "Deposits",
    },
    "baseline_tdc_lp_funding": {
        "title": "Funding responses are clearer in reserves than in spreads",
        "subtitle": "Money-market spread responses are weak in the current quarterly sample.",
        "summary": "The funding block is not used as a headline result on the site because the raw-reserve response is contaminated by Federal Reserve balance-sheet movements and repo-spread responses are weak.",
        "kicker": "Funding",
    },
    "baseline_tdc_lp_credit_spreads": {
        "title": "Credit spreads remain secondary evidence",
        "subtitle": "The selected public branch does not support an early credit-spread headline.",
        "summary": "Credit measures remain a cross-check, not a lead result. The current public branch does not show a robust early spread response, so this stays in additional evidence.",
        "kicker": "Credit",
    },
    "baseline_tdc_lp_inflation": {
        "title": "Inflation remains a secondary branch",
        "subtitle": "Any inflation signal is later and less secure than the deposit result.",
        "summary": "Inflation is not part of the headline claim stack. At most, the current branch suggests a later and mixed pattern, with somewhat more movement in core measures than in headline CPI.",
        "kicker": "Inflation",
    },
    "baseline_tdc_lp_fx": {
        "title": "FX evidence stays modest",
        "subtitle": "The broad-dollar branch is directional, not decisive.",
        "summary": "The broad-dollar branch points toward near-term softening, but it is still a one-series supporting check rather than a headline result.",
        "kicker": "FX",
    },
    "baseline_tdc_lp_private_assets": {
        "title": "Private-asset responses look mixed",
        "subtitle": "Category shifts matter more than a single expansion story.",
        "summary": "The private-asset block is a composition check, not a settled mechanism result. It looks across non-Treasury bank securities and lending categories to see where any crowd-out might appear.",
        "kicker": "Private assets",
    },
    "baseline_tdc_lp_liquidity_decomposition": {
        "title": "Raw reserves and Fed-net reserves tell different stories",
        "subtitle": "Netting out Federal Reserve balance-sheet growth weakens the reserve response materially.",
        "summary": "This branch is the clean answer to the reserve question. Raw reserves rise on impact, but once Fed asset growth is netted out the short-run response is much weaker and can reverse after impact.",
        "kicker": "Liquidity",
    },
    "qra_event_rates_63bd": {
        "title": "QRA dates move Treasury rates and repo pricing",
        "subtitle": "The public event branch now uses the broader reviewed release sample.",
        "summary": "This event branch traces how reviewed QRA releases move Treasury rates, the term spread, and repo pricing across 1, 5, 21, and 63 business days.",
        "kicker": "Event rates",
    },
    "qra_event_risk_21bd": {
        "title": "QRA dates move market risk measures",
        "subtitle": "The public event branch now uses the broader reviewed release sample.",
        "summary": "This event branch tracks equity returns, volatility, and liquidity-related balances around reviewed QRA releases over short business-day windows.",
        "kicker": "Event risk",
    },
    "tdc_state_dep_low_reserves": {
        "title": "State dependence under low-reserve conditions",
        "subtitle": "This is the only public heterogeneity check still left on the site.",
        "summary": "The low-reserve branch is still only supporting evidence. It is kept as a suggestive heterogeneity check, not as a headline mechanism result.",
        "kicker": "State dependence",
    },
    "tdc_state_dep_on_rrp_drain": {
        "title": "State dependence during ON RRP drain episodes",
        "subtitle": "The balance-sheet channel may shift as ON RRP balances run down.",
        "summary": "This state branch asks whether the transmission pattern changes when the ON RRP facility is already draining.",
        "kicker": "State dependence",
    },
    "tdc_state_dep_bank_short_share": {
        "title": "State dependence under high bank short-share exposure",
        "subtitle": "Short-duration bank positioning may change deposit transmission.",
        "summary": "This state branch asks whether Treasury Deposit Contribution matters differently when banks are tilted toward short-maturity holdings.",
        "kicker": "State dependence",
    },
    "tdc_state_dep_bank_foreign_private_corr": {
        "title": "State dependence under tighter bank and foreign-private co-movement",
        "subtitle": "Behavioral co-movement can proxy for absorption conditions.",
        "summary": "This state branch asks whether deposit transmission changes when banks and foreign private investors move together more tightly in Treasury absorption.",
        "kicker": "State dependence",
    },
    "tdc_state_dep_slr_bank_leverage_pressure": {
        "title": "State dependence under SLR leverage pressure",
        "subtitle": "Leverage constraints can change how Treasury absorption feeds into deposits.",
        "summary": "This state branch asks whether stronger SLR pressure changes the balance-sheet transmission of Treasury Deposit Contribution.",
        "kicker": "State dependence",
    },
}

ROBUSTNESS_META: dict[str, dict[str, str]] = {
    "baseline_tdc_lp_deposits": {
        "title": "Deposits remain the cleanest robust result",
        "summary": "The deposit response survives the larger control universe and the broad-depository treatment variant. Excluding 2008-2009 or 2020 weakens precision in places but does not flip the headline deposit result.",
    },
    "baseline_tdc_lp_credit_spreads": {
        "title": "Credit spreads remain a weak public branch",
        "summary": "The larger control ladder still leaves credit spreads too small and unstable for a lead public claim.",
    },
    "baseline_tdc_lp_inflation": {
        "title": "Inflation remains a lagged branch",
        "summary": "The high-dimensional ladder does not turn inflation into an immediate response. The later core-inflation pattern remains the relevant part of the result.",
    },
    "baseline_tdc_lp_fx": {
        "title": "FX remains modest and directional",
        "summary": "The broad-dollar response stays modest under the larger control set. The most that can be said publicly is directional: the dollar tends to soften rather than strengthen in the near term.",
    },
    "baseline_tdc_lp_private_assets": {
        "title": "Private assets still look like recomposition, not one-way crowding out",
        "summary": "The larger control set still points to composition effects across bank private-asset categories rather than one simple expansion or contraction of all private non-Treasury assets.",
    },
    "baseline_tdc_lp_liquidity_decomposition": {
        "title": "Fed-net reserves are the relevant robustness result",
        "summary": "Once the Federal Reserve balance-sheet contribution is removed, the reserve story changes materially. That survives the larger control set and remains the right interpretation of the liquidity branch.",
    },
}

OUTCOME_LABELS = {
    "matched_total_deposits": "Matched total deposits",
    "other_component_qoq": "Other deposit component",
    "tdcpass_other_component_qoq": "Residual non-TDC component (`tdcpass`)",
    "tdcpass_strict_loan_core_min_qoq": "Strict loan-core minimum (`tdcpass`)",
    "tdcpass_strict_non_treasury_securities_qoq": "Strict non-Treasury securities (`tdcpass`)",
    "tdcpass_strict_identifiable_total_qoq": "Strict identifiable total (`tdcpass`)",
    "tdcpass_strict_identifiable_gap_qoq": "Residual minus strict total (`tdcpass`)",
    "m2": "M2",
    "reserve_balances": "Raw reserve balances",
    "repo_spread": "Repo spread",
    "fed_funds": "Fed funds",
    "sofr": "SOFR",
    "baa_aaa": "BAA - AAA spread",
    "investment_grade_oas": "Investment-grade OAS",
    "high_yield_oas": "High-yield OAS",
    "headline_cpi_inflation_qoq_ann": "Headline CPI",
    "core_cpi_inflation_qoq_ann": "Core CPI",
    "core_pce_inflation_qoq_ann": "Core PCE",
    "broad_dollar_change": "Broad dollar",
    "large_time_deposits_qoq": "Large time deposits",
    "retail_mmf_assets_qoq": "Retail MMFs",
    "institutional_mmf_assets_qoq": "Institutional MMFs",
    "bank_credit_qoq": "Bank credit",
    "bank_business_loans_qoq": "Business loans",
    "bank_ci_loans_h8_qoq": "C&I loans (H.8)",
    "bank_short_term_loans_z1_qoq": "Short-term bank loans (Z.1)",
    "bank_non_treasury_securities_qoq": "Bank non-Treasury securities",
    "bank_consumer_loans_qoq": "Consumer loans",
    "bank_real_estate_loans_qoq": "Real-estate loans",
    "row_loans_assets_qoq": "ROW non-Treasury assets",
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
    "matched_total_deposits_pct_gdp": "Matched deposits (% GDP)",
    "other_component_qoq_pct_gdp": "Other component (% GDP)",
    "large_time_deposits_qoq_pct_gdp": "Large time deposits (% GDP)",
    "retail_mmf_assets_qoq_pct_gdp": "Retail MMFs (% GDP)",
    "institutional_mmf_assets_qoq_pct_gdp": "Institutional MMFs (% GDP)",
    "bank_credit_qoq_pct_gdp": "Bank credit (% GDP)",
    "bank_business_loans_qoq_pct_gdp": "Business loans (% GDP)",
    "bank_ci_loans_h8_qoq_pct_gdp": "C&I loans (H.8, % GDP)",
    "bank_short_term_loans_z1_qoq_pct_gdp": "Short-term bank loans (Z.1, % GDP)",
    "bank_non_treasury_securities_qoq_pct_gdp": "Bank non-Treasury securities (% GDP)",
    "bank_consumer_loans_qoq_pct_gdp": "Consumer loans (% GDP)",
    "bank_real_estate_loans_qoq_pct_gdp": "Real-estate loans (% GDP)",
    "row_loans_assets_qoq_pct_gdp": "ROW non-Treasury assets (% GDP)",
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
    "reserve_balances_qoq": "Raw reserve balances",
    "reserve_balances_net_fed_assets_qoq": "Reserves net of Fed total assets",
    "reserve_balances_net_fed_treasury_qoq": "Reserves net of Fed Treasury holdings",
    "fed_total_assets_qoq": "Fed total assets",
    "fed_treasury_holdings_qoq": "Fed Treasury holdings",
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
    "dml": "DML",
    "tmle": "TMLE",
    "sofr": "SOFR",
    "tga": "TGA",
    "m2": "M2",
}

ABBREVIATION_GLOSSARY = [
    ("TDC", "Treasury Deposit Contribution."),
    ("DU", "Domestic nonbank deposit-using sector."),
    ("Fed", "Federal Reserve."),
    ("RU", "Reserve-side sector."),
    ("TOC", "Treasury operating cash."),
    ("ROW", "Rest of world."),
    ("TGA", "Treasury General Account."),
    ("ON RRP", "Overnight reverse repurchase facility."),
    ("QRA", "Quarterly Refunding Announcement."),
]

TREATMENT_LABELS = {
    "tdc_bank_only_qoq": "Baseline TDC estimate",
    "tdc_bank_only_shock": "Baseline TDC estimate",
    "tdc_base_bank_only_ru_flow": "Baseline TDC estimate",
    "tdc_base_broad_depository_np_cu_ru_flow": "Broad-depository variant",
    "tdc_tier2_interest_corrected_bank_only_ru_flow": "Interest-corrected bank-only variant",
    "tdc_tier3_fiscal_corrected_bank_only_ru_flow": "Fiscal-corrected bank-only variant",
    "tdc_tier2_interest_corrected_broad_depository_np_cu_ru_flow": "Interest-corrected broad-depository variant",
    "tdc_tier3_fiscal_corrected_broad_depository_np_cu_ru_flow": "Fiscal-corrected broad-depository variant",
    "tdc_no_remit_bank_only": "No-remittance variant",
    "tdc_domestic_bank_only_ru_flow": "Domestic-bank-only variant",
    "tdc_bank_only_extended_1990": "Extended-bank variant",
}

PUBLIC_TREATMENT_VARIANTS = [
    "tdc_base_broad_depository_np_cu_ru_flow",
]

TDC_EQUATIONS = [
    {
        "kicker": "Headline estimator",
        "title": "Baseline TDC estimate used in the quarterly results",
        "body": "This is the implemented quarterly approximation used in the main EA-TDC results. It includes Federal Reserve, bank-sector, and rest-of-world Treasury transactions, subtracts Treasury operating cash, and adds positive Federal Reserve remittances.",
        "latex": r"\widehat{\Delta D}^{mkt,bank}_{TDC,t} = \left(\Delta TS^{tx}_{Fed,t} + \Delta TS^{tx}_{Banks,t} + \Delta TS^{tx}_{ROW,t}\right) - \Delta Cash^{tx}_{Treasury,t} + Remit^{+}_{Fed,t}",
        "definitions": [
            (r"\widehat{\Delta D}^{mkt,bank}_{TDC,t}", "Estimated quarterly approximation to Treasury Deposit Contribution used in period t."),
            (r"\Delta TS^{tx}_{Fed,t}", "Federal Reserve net transactions in marketable Treasury securities."),
            (r"\Delta TS^{tx}_{Banks,t}", "Bank-sector net transactions in marketable Treasury securities."),
            (r"\Delta TS^{tx}_{ROW,t}", "Rest-of-world net transactions in marketable Treasury securities."),
            (r"\Delta Cash^{tx}_{Treasury,t}", "Change in Treasury operating cash transactions; higher Treasury cash drains deposits before they reach domestic nonbank deposits."),
            (r"Remit^{+}_{Fed,t}", "Positive Federal Reserve remittances to Treasury, summed within the period."),
        ],
    },
    {
        "kicker": "Theory identity 1",
        "title": "DU-facing definition",
        "body": "The theory object is Treasury Deposit Contribution to domestic nonbank deposits: net Treasury payments to DUs, plus Treasury debt service to DUs, plus net Treasury-security sales from DUs to RUs.",
        "latex": r"\Delta D^{TDC}_{DU} = \left(G^{ND}_{DU} - R^T_{DU}\right) + DS^T_{DU} + \left(Q^T_{DU\to RU} - Q^T_{RU\to DU}\right)",
        "definitions": [
            (r"\Delta D^{TDC}_{DU}", "Treasury Deposit Contribution to domestic nonbank deposits."),
            (r"G^{ND}_{DU}", "Non-debt-service Treasury spending paid into DU deposits."),
            (r"R^T_{DU}", "Taxes and other non-Treasury-security receipts paid from DU deposits."),
            (r"DS^T_{DU}", "Treasury-security debt service paid into DU deposits."),
            (r"Q^T_{DU\to RU}", "Cash settlement value of Treasury securities sold by DUs to RUs."),
            (r"Q^T_{RU\to DU}", "Cash settlement value of Treasury securities sold by RUs to DUs."),
        ],
    },
    {
        "kicker": "Theory identity 2",
        "title": "Treasury-cash constraint",
        "body": "The same theory object can be re-expressed through reserve-side Treasury-security settlement, RU-facing Treasury cash flows, actual positive Fed remittances, and Treasury operating cash.",
        "latex": r"\Delta D^{TDC}_{DU} = \left(Q^T_{DU\to RU} - Q^T_{RU\to DU}\right) + \left(I^T + R^T_{RU} + \Pi^F_T - G^{ND}_{RU} - DS^T_{RU}\right) - \Delta TOC",
        "definitions": [
            (r"I^T", "Treasury-security issuance proceeds received by Treasury at cash settlement value."),
            (r"R^T_{RU}", "Taxes and other non-Treasury-security receipts paid by RU sectors."),
            (r"\Pi^F_T", "Actual Federal Reserve remittances transferred to Treasury during the period."),
            (r"G^{ND}_{RU}", "Non-debt-service Treasury spending to RU sectors."),
            (r"DS^T_{RU}", "Treasury-security debt service paid to RU sectors."),
            (r"\Delta TOC", "Change in Treasury operating cash, including the TGA, TT&L, and related operating balances."),
        ],
    },
    {
        "kicker": "Theory identity 3",
        "title": "Residual deposit decomposition",
        "body": "This is a diagnostic cross-check rather than the headline estimator: start from deposit change, subtract the major non-Treasury deposit drivers, and treat the remainder as TDC.",
        "latex": r"\Delta D^{TDC}_{DU} = (\Delta M - \Delta C - \Delta X) - (\Delta L^B_{DU} + \Delta A^{B,NT}_{DU}) - \Delta A^{CB,NT}_{DU} - \Delta F^{NT}_{DU} - \varepsilon",
        "definitions": [
            (r"\Delta M", "Change in the chosen money aggregate that includes DU deposits."),
            (r"\Delta C", "Change in currency held outside DU deposits."),
            (r"\Delta X", "Other non-deposit money components included in the selected aggregate."),
            (r"\Delta L^B_{DU}", "Net bank lending to DUs."),
            (r"\Delta A^{B,NT}_{DU}", "Domestic-bank non-Treasury asset flows to DUs."),
            (r"\Delta A^{CB,NT}_{DU}", "Central-bank non-Treasury asset flows to DUs."),
            (r"\Delta F^{NT}_{DU}", "Foreign non-Treasury flows to DUs."),
            (r"\varepsilon", "Timing, reclassification, and other residual measurement error."),
        ],
    },
]

INSIGHTS_HOME = [
    {
        "kicker": "Question 1",
        "title": "Where does Treasury financing show up first?",
        "body": "The baseline reduced-form work asks whether Treasury Deposit Contribution produces an early, robust deposit response in the selected public quarterly specification.",
    },
    {
        "kicker": "Question 2",
        "title": "Does the deposit result survive broader holder definitions?",
        "body": "The main perimeter check asks whether the deposit result survives when the TDC construction broadens from the baseline bank-focused measure to a broader depository-holder definition.",
    },
    {
        "kicker": "Question 3",
        "title": "What survives under strict independent non-TDC measurement?",
        "body": "The boundary check uses the narrower `tdcpass` strict source-side lane rather than residual closure, so independent evidence is limited to directly identifiable non-Treasury bank-asset support.",
    },
]

INSIGHTS_SIDECAR = [
    {
        "kicker": "Inflation",
        "title": "Inflation moves later than deposits and reserves",
        "body": "Inflation is included as a secondary check on whether the baseline TDC flow reaches prices after it passes through balance sheets and liquidity.",
    },
    {
        "kicker": "FX",
        "title": "FX tracks the external pricing margin",
        "body": "The dollar branch asks whether the baseline TDC flow also shows up in exchange-rate pricing as well as domestic balance sheets.",
    },
    {
        "kicker": "Private assets",
        "title": "Private balance sheets remain secondary context",
        "body": "The private-asset branch stays below the headline claim, but it helps show where composition effects matter more than a single expansion story.",
    },
]

TREATMENT_COMPARISON_JOBS = {
    "baseline_tdc_lp_deposits": {
        "title": "Deposit results survive the main TDC variants",
        "subtitle": "The broad-depository variant stays close to the selected public estimate.",
        "summary": "This comparison shows whether the deposit result depends on one narrow TDC construction. The broad-depository variant remains close to the selected public branch, so the deposit result does not appear to be a banks-only artifact.",
        "outcomes": ["matched_total_deposits", "other_component_qoq"],
    },
}

INDEPENDENT_NON_TDC_JOB_ID = "tdcpass_strict_source_side_nontdc"
INDEPENDENT_NON_TDC_OUTCOME_KEYS = [
    "tdcpass_other_component_qoq",
    "tdcpass_strict_loan_core_min_qoq",
    "tdcpass_strict_non_treasury_securities_qoq",
    "tdcpass_strict_identifiable_total_qoq",
    "tdcpass_strict_identifiable_gap_qoq",
]


def _eq_term_html(
    base: str,
    *,
    sup: str | None = None,
    sub: str | None = None,
    delta: bool = False,
    hat: bool = False,
) -> str:
    base_html = f'<span class="eq-var">{html.escape(base)}</span>'
    if hat:
        base_html = f'<span class="eq-hat">{base_html}</span>'
    delta_html = '<span class="eq-op">&Delta;</span>' if delta else ""
    sup_html = f'<sup class="eq-sup">{html.escape(sup)}</sup>' if sup else ""
    sub_html = f'<sub class="eq-sub">{html.escape(sub)}</sub>' if sub else ""
    return f"{delta_html}{base_html}{sup_html}{sub_html}"


def _eq_def(latex: str, meaning: str, symbol_html: str) -> tuple[str, str, str]:
    return (latex, meaning, symbol_html)


RESIDUAL_SYMBOL_HTML = _eq_term_html("D", sup="non-TDC", sub="t", delta=True)
MATCHED_SYMBOL_HTML = _eq_term_html("D", sup="matched", sub="t", delta=True)
TDC_SYMBOL_HTML = _eq_term_html("D", sup="TDC", sub="t", delta=True, hat=True)
RECON_SYMBOL_HTML = _eq_term_html("D", sup="recon", sub="t", delta=True)
GAP_SYMBOL_HTML = _eq_term_html("D", sup="gap", sub="t", delta=True)
SUB_SYMBOL_HTML = _eq_term_html("D", sup="sub", sub="t", delta=True)
BANK_SYMBOL_HTML = _eq_term_html("D", sup="bank", sub="t", delta=True)
PUBLIC_SYMBOL_HTML = _eq_term_html("D", sup="public", sub="t", delta=True)
EXT_SYMBOL_HTML = _eq_term_html("D", sup="ext", sub="t", delta=True)
LTD_SYMBOL_HTML = _eq_term_html("LTD", sub="t", delta=True)
RETAIL_MMF_SYMBOL_HTML = _eq_term_html("MMF", sup="retail", sub="t", delta=True)
INST_MMF_SYMBOL_HTML = _eq_term_html("MMF", sup="inst", sub="t", delta=True)
BANK_SECURITIES_SYMBOL_HTML = _eq_term_html("S", sup="bank, non-Tsy", sub="t", delta=True)
BANK_LOANS_SYMBOL_HTML = _eq_term_html("L", sup="bank, short", sub="t", delta=True)
TGA_SYMBOL_HTML = _eq_term_html("TGA", sub="t", delta=True)
ON_RRP_SYMBOL_HTML = _eq_term_html("ONRRP", sub="t", delta=True)
ROW_PRIVATE_SYMBOL_HTML = _eq_term_html("F", sup="ROW, private", sub="t", delta=True)
NET_EXPORTS_SYMBOL_HTML = _eq_term_html("NX", sub="t", delta=True)

DEPOSIT_ACCOUNTING_BUCKETS = [
    {
        "title": "Deposit substitution",
        "summary": "This bucket asks whether money moves between deposits and deposit-like instruments rather than disappearing outright.",
        "equation": r"\Delta D^{\text{sub}}_t = \Delta LTD_t - \Delta MMF^{\text{retail}}_t - \Delta MMF^{\text{inst}}_t",
        "equation_html": f"{SUB_SYMBOL_HTML} = {LTD_SYMBOL_HTML} - {RETAIL_MMF_SYMBOL_HTML} - {INST_MMF_SYMBOL_HTML}",
        "definitions": [
            _eq_def(r"\Delta D^{\text{sub}}_t", "Interpretive proxy bucket `deposit_substitution_block_qoq`, constructed as `large_time_deposits_qoq - retail_mmf_assets_qoq - institutional_mmf_assets_qoq`.", SUB_SYMBOL_HTML),
            _eq_def(r"\Delta LTD_t", "Large time deposits: outcome `large_time_deposits_qoq`, FRED `LTDACBM027NBOG`, quarter-over-quarter change.", LTD_SYMBOL_HTML),
            _eq_def(r"\Delta MMF^{\text{retail}}_t", "Retail money market fund assets: outcome `retail_mmf_assets_qoq`, FRED `RMFSL`, quarter-over-quarter change.", RETAIL_MMF_SYMBOL_HTML),
            _eq_def(r"\Delta MMF^{\text{inst}}_t", "Institutional money market fund assets: outcome `institutional_mmf_assets_qoq`, FRED `WIMFSL` with Z.1 fallback `BOGZ1FL883034010Q`, quarter-over-quarter change.", INST_MMF_SYMBOL_HTML),
        ],
        "series": [
            "Large time deposits (`large_time_deposits_qoq`; FRED `LTDACBM027NBOG`)",
            "Retail money market fund assets (`retail_mmf_assets_qoq`; FRED `RMFSL`)",
            "Institutional money market fund assets (`institutional_mmf_assets_qoq`; FRED `WIMFSL`, fallback `BOGZ1FL883034010Q`)",
        ],
    },
    {
        "title": "Bank balance-sheet adjustment",
        "summary": "This bucket tracks the other bank asset moves that can create or absorb deposits outside the direct Treasury channel.",
        "equation": r"\Delta D^{\text{bank}}_t = \Delta S^{\text{bank, non-Tsy}}_t + \Delta L^{\text{bank, short}}_t",
        "equation_html": f"{BANK_SYMBOL_HTML} = {BANK_SECURITIES_SYMBOL_HTML} + {BANK_LOANS_SYMBOL_HTML}",
        "definitions": [
            _eq_def(r"\Delta D^{\text{bank}}_t", "Interpretive proxy bucket `bank_balance_sheet_proxy_block_qoq`, constructed as `bank_non_treasury_securities_qoq + bank_short_term_loans_z1_qoq`.", BANK_SYMBOL_HTML),
            _eq_def(r"\Delta S^{\text{bank, non-Tsy}}_t", "Banks' non-Treasury securities: outcome `bank_non_treasury_securities_qoq`, FRED `OSEACBW027SBOG`, quarter-over-quarter change.", BANK_SECURITIES_SYMBOL_HTML),
            _eq_def(r"\Delta L^{\text{bank, short}}_t", "Short-term bank loans: outcome `bank_short_term_loans_z1_qoq`, FRED `BOGZ1FL704041005Q`, quarter-over-quarter change.", BANK_LOANS_SYMBOL_HTML),
        ],
        "series": [
            "Bank non-Treasury securities (`bank_non_treasury_securities_qoq`; FRED `OSEACBW027SBOG`)",
            "Short-term bank loans (`bank_short_term_loans_z1_qoq`; FRED `BOGZ1FL704041005Q`)",
        ],
    },
    {
        "title": "Public-liquidity plumbing",
        "summary": "This proxy bucket tracks public cash balances that pull money into or out of the private deposit base. It is a diagnostic liquidity-plumbing surface, not the full historical Treasury operating cash leg.",
        "equation": r"\Delta D^{\text{public}}_t = - \Delta TGA_t - \Delta ONRRP_t",
        "equation_html": f"{PUBLIC_SYMBOL_HTML} = - {TGA_SYMBOL_HTML} - {ON_RRP_SYMBOL_HTML}",
        "definitions": [
            _eq_def(r"\Delta D^{\text{public}}_t", "Interpretive proxy bucket `public_liquidity_proxy_block_qoq`, constructed as `- tga_balance_qoq - on_rrp_balance_qoq`. This is narrower than the full Treasury operating cash concept used in the theory identities.", PUBLIC_SYMBOL_HTML),
            _eq_def(r"\Delta TGA_t", "Treasury General Account balance: outcome `tga_balance_qoq`, FRED `WDTGAL`, quarter-over-quarter change. Useful as a diagnostic cash proxy, but not the full TOC term when TT&L and related balances matter.", TGA_SYMBOL_HTML),
            _eq_def(r"\Delta ONRRP_t", "ON RRP balance: outcome `on_rrp_balance_qoq`, FRED `RRPONTSYD`, quarter-over-quarter change.", ON_RRP_SYMBOL_HTML),
        ],
        "series": [
            "Treasury General Account balance (`tga_balance_qoq`; FRED `WDTGAL`)",
            "ON RRP balance (`on_rrp_balance_qoq`; FRED `RRPONTSYD`)",
        ],
    },
    {
        "title": "External flow",
        "summary": "This bucket tracks cross-border and external-balance channels that can add to or subtract from deposits.",
        "equation": r"\Delta D^{\text{ext}}_t = \Delta F^{\text{ROW, private}}_t + \Delta NX_t",
        "equation_html": f"{EXT_SYMBOL_HTML} = {ROW_PRIVATE_SYMBOL_HTML} + {NET_EXPORTS_SYMBOL_HTML}",
        "definitions": [
            _eq_def(r"\Delta D^{\text{ext}}_t", "Interpretive proxy bucket `external_flow_proxy_block_qoq`, constructed as `row_private_flow_block + net_exports_qoq`.", EXT_SYMBOL_HTML),
            _eq_def(r"\Delta F^{\text{ROW, private}}_t", "Rest-of-world private-flow block: outcome `row_private_flow_block`, constructed as `row_corp_bonds_flow + row_corp_equities_flow + row_agency_flow + row_nonfin_business_loans_flow`.", ROW_PRIVATE_SYMBOL_HTML),
            _eq_def(r"\Delta NX_t", "Net exports: outcome `net_exports_qoq`, constructed as `exports_qoq - imports_qoq`, using FRED `EXPGS` and `IMPGS`.", NET_EXPORTS_SYMBOL_HTML),
        ],
        "series": [
            "ROW private-flow block (`row_private_flow_block` = `row_corp_bonds_flow + row_corp_equities_flow + row_agency_flow + row_nonfin_business_loans_flow`)",
            "Net exports (`net_exports_qoq` = `exports_qoq - imports_qoq`; FRED `EXPGS` minus `IMPGS`)",
        ],
    },
]

HIDDEN_SITE_JOB_IDS = {
    "baseline_tdc_lp_funding",
    "tdc_state_dep_on_rrp_drain",
    "tdc_state_dep_bank_foreign_private_corr",
    "tdc_state_dep_bank_short_share",
    "tdc_state_dep_slr_bank_leverage_pressure",
}

CSS_TEXT = dedent(
    """
    :root {
      --bg-primary: #fafaf7;
      --bg-secondary: #f0efe9;
      --bg-surface: #ffffff;
      --bg-raised: #ffffff;
      --bg-nav: rgba(250, 250, 247, 0.88);
      --text-primary: #17171a;
      --text-secondary: #4e5460;
      --text-tertiary: #777f8c;
      --text-inverse: #fafaf7;
      --accent: #19385f;
      --accent-light: #2b6cb0;
      --accent-muted: #e7eef7;
      --accent-hover: #355f97;
      --border: #dde4ee;
      --border-strong: #c7d0db;
      --chart-1: #2b6cb0;
      --chart-2: #d69e2e;
      --chart-3: #9b2c2c;
      --chart-4: #276749;
      --chart-5: #805ad5;
      --chart-6: #dd6b20;
      --chart-grid: #e2e8f0;
      --chart-axis: #9aa5b3;
      --chart-label: #4e5460;
      --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.03);
      --shadow-md: 0 10px 30px rgba(18, 28, 45, 0.05);
      --radius-sm: 8px;
      --radius-md: 14px;
      --radius-lg: 24px;
      --max-width: 1200px;
      --nav-height: 64px;
      --transition: 220ms ease;
      color-scheme: light;
    }

    [data-theme="dark"] {
      --bg-primary: #0d1117;
      --bg-secondary: #161b22;
      --bg-surface: #1c2128;
      --bg-raised: #242a33;
      --bg-nav: rgba(13, 17, 23, 0.9);
      --text-primary: #e6edf3;
      --text-secondary: #96a0ab;
      --text-tertiary: #6f7882;
      --text-inverse: #0d1117;
      --accent: #79c0ff;
      --accent-light: #58a6ff;
      --accent-muted: #16253a;
      --accent-hover: #a5d6ff;
      --border: #30363d;
      --border-strong: #434a54;
      --chart-1: #58a6ff;
      --chart-2: #d29922;
      --chart-3: #f85149;
      --chart-4: #3fb950;
      --chart-5: #c27abf;
      --chart-6: #e0a050;
      --chart-grid: #30363d;
      --chart-axis: #4a525d;
      --chart-label: #96a0ab;
      --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.2);
      --shadow-md: 0 10px 32px rgba(0, 0, 0, 0.22);
      color-scheme: dark;
    }

    *, *::before, *::after {
      box-sizing: border-box;
    }

    html {
      scroll-behavior: smooth;
      scroll-padding-top: calc(var(--nav-height) + 18px);
      -webkit-text-size-adjust: 100%;
    }

    body {
      margin: 0;
      font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      line-height: 1.65;
      color: var(--text-primary);
      background:
        radial-gradient(circle at top right, rgba(43, 108, 176, 0.06), transparent 24%),
        linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 22%, var(--bg-primary) 100%);
      transition: background-color var(--transition), color var(--transition);
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }

    a {
      color: var(--accent-light);
      text-decoration: none;
      transition: color var(--transition);
    }

    a:hover {
      color: var(--accent-hover);
      text-decoration: underline;
    }

    h1, h2, h3, h4 {
      margin: 0;
      font-family: "Source Serif 4", Georgia, serif;
      line-height: 1.1;
      letter-spacing: -0.015em;
      color: var(--text-primary);
    }

    p {
      margin: 0;
      color: var(--text-secondary);
    }

    code {
      font-family: "JetBrains Mono", monospace;
    }

    .container {
      max-width: var(--max-width);
      margin: 0 auto;
      padding: 0 24px;
    }

    .nav {
      position: sticky;
      top: 0;
      z-index: 100;
      background: var(--bg-nav);
      backdrop-filter: blur(14px);
      border-bottom: 1px solid rgba(127, 127, 127, 0.08);
    }

    .nav-inner {
      min-height: var(--nav-height);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
    }

    .brand {
      display: flex;
      flex-direction: column;
      gap: 2px;
      min-width: 0;
    }

    .brand-mark {
      font-size: 0.78rem;
      font-weight: 800;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--accent-light);
    }

    .brand-copy {
      font-size: 0.88rem;
      color: var(--text-tertiary);
    }

    .nav-links {
      display: flex;
      align-items: center;
      gap: 14px;
      flex-wrap: wrap;
    }

    .nav-links a {
      font-size: 0.88rem;
      color: var(--text-secondary);
    }

    .theme-toggle {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 42px;
      height: 42px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--bg-surface);
      color: var(--text-primary);
      cursor: pointer;
      box-shadow: var(--shadow-sm);
    }

    .page {
      padding-bottom: 96px;
    }

    .hero {
      min-height: calc(100svh - var(--nav-height));
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(300px, 0.95fr);
      gap: 30px;
      align-items: center;
      padding: 28px 0 52px;
    }

    .hero-copy {
      max-width: 36rem;
      display: grid;
      gap: 18px;
    }

    .hero-copy h1 {
      font-size: clamp(2.05rem, 4vw, 3.35rem);
      max-width: 12ch;
    }

    .hero-copy p {
      font-size: 1rem;
    }

    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 0.76rem;
      font-weight: 800;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--accent-light);
    }

    .button-row {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }

    .button,
    .link-chip {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 40px;
      padding: 0 16px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--bg-surface);
      color: var(--text-primary);
      font-size: 0.84rem;
      font-weight: 600;
      box-shadow: var(--shadow-sm);
      text-decoration: none;
    }

    .button.primary {
      background: var(--accent);
      color: var(--text-inverse);
      border-color: var(--accent);
    }

    .hero-panel {
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      background:
        radial-gradient(circle at top right, rgba(43, 108, 176, 0.12), transparent 35%),
        linear-gradient(145deg, rgba(255, 255, 255, 0.72), rgba(255, 255, 255, 0.24));
      padding: 26px;
      box-shadow: var(--shadow-md);
    }

    [data-theme="dark"] .hero-panel {
      background:
        radial-gradient(circle at top right, rgba(88, 166, 255, 0.16), transparent 35%),
        linear-gradient(145deg, rgba(28, 33, 40, 0.92), rgba(36, 42, 51, 0.78));
    }

    .metric-grid {
      display: grid;
      gap: 14px;
    }

    .metric-card,
    .insight-card,
    .section-block,
    .artifact-card,
    .deferred-card,
    .mini-chart-card,
    .equation-card,
    .table-shell {
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      background: var(--bg-surface);
      box-shadow: var(--shadow-sm);
    }

    .metric-card {
      padding: 18px 20px;
      display: grid;
      gap: 8px;
    }

    .metric-card span {
      font-size: 0.76rem;
      font-weight: 800;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--text-tertiary);
    }

    .metric-card strong {
      font-size: 1.95rem;
      color: var(--accent-light);
      line-height: 1;
    }

    .section {
      padding: 68px 0;
      border-top: 1px solid rgba(127, 127, 127, 0.08);
    }

    .section-header {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(300px, 0.72fr);
      gap: 26px;
      align-items: start;
      margin-bottom: 26px;
    }

    .section-header h2 {
      margin-top: 12px;
      font-size: clamp(1.8rem, 3.2vw, 2.9rem);
    }

    .insight-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
    }

    .insight-card {
      padding: 18px;
      display: grid;
      gap: 12px;
    }

    .definition-shell {
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      background: var(--bg-surface);
      box-shadow: var(--shadow-md);
      overflow: hidden;
    }

    .definition-shell summary {
      list-style: none;
      cursor: pointer;
      padding: 22px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }

    .definition-shell summary::-webkit-details-marker {
      display: none;
    }

    .definition-shell[open] summary {
      border-bottom: 1px solid var(--border);
    }

    .definition-body {
      padding: 22px 24px 26px;
      display: grid;
      gap: 20px;
    }

    .equation-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }

    .equation-card {
      padding: 18px;
      display: grid;
      gap: 12px;
    }

    .equation-card .math,
    .insight-card .math {
      padding: 14px;
      border-radius: 12px;
      background: rgba(127, 127, 127, 0.05);
      overflow-x: auto;
    }

    .equation-rendered {
      font-family: "Source Serif 4", Georgia, serif;
      font-size: 1.28rem;
      line-height: 1.45;
      white-space: nowrap;
      color: var(--text-primary);
    }

    .equation-rendered .eq-var {
      font-style: italic;
    }

    .equation-rendered .eq-op {
      margin-right: 0.04em;
    }

    .equation-rendered .eq-hat {
      position: relative;
      display: inline-block;
      padding-top: 0.16em;
    }

    .equation-rendered .eq-hat::before {
      content: "";
      position: absolute;
      left: 0.08em;
      right: 0.08em;
      top: 0.02em;
      border-top: 1.5px solid currentColor;
      transform: skewX(-18deg);
      transform-origin: center;
    }

    .equation-rendered .eq-sup,
    .equation-rendered .eq-sub {
      font-size: 0.7em;
      font-style: normal;
      letter-spacing: 0.01em;
    }

    .equation-symbol {
      display: inline-flex;
      align-items: baseline;
      gap: 0;
      font-family: "Source Serif 4", Georgia, serif;
      font-size: 1rem;
      line-height: 1.2;
      color: var(--text-primary);
      white-space: nowrap;
    }

    .equation-symbol .eq-var {
      font-style: italic;
    }

    .equation-symbol .eq-op {
      margin-right: 0.04em;
    }

    .equation-symbol .eq-hat {
      position: relative;
      display: inline-block;
      padding-top: 0.16em;
    }

    .equation-symbol .eq-hat::before {
      content: "";
      position: absolute;
      left: 0.08em;
      right: 0.08em;
      top: 0.02em;
      border-top: 1.5px solid currentColor;
      transform: skewX(-18deg);
      transform-origin: center;
    }

    .equation-symbol .eq-sup,
    .equation-symbol .eq-sub {
      font-size: 0.72em;
      font-style: normal;
      letter-spacing: 0.01em;
    }

    .equation-definitions {
      display: grid;
      gap: 10px;
      margin: 0;
    }

    .equation-definitions .definition-row {
      display: grid;
      gap: 4px;
      padding-top: 10px;
      border-top: 1px solid var(--border);
    }

    .equation-definitions dt {
      margin: 0;
      margin: 0;
    }

    .equation-definitions dd {
      margin: 0;
      color: var(--text-secondary);
      font-size: 0.96rem;
      line-height: 1.55;
    }

    .symbol-code {
      display: inline-block;
      max-width: 100%;
      font-family: "JetBrains Mono", monospace;
      font-size: 0.9rem;
      color: var(--text-primary);
      white-space: normal;
      overflow-wrap: anywhere;
      word-break: break-word;
    }

    .math-inline {
      white-space: nowrap;
    }

    .notation-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }

    .notation-chip {
      padding: 14px 16px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: rgba(127, 127, 127, 0.04);
    }

    .notation-chip strong {
      display: block;
      margin-bottom: 6px;
      font-size: 0.92rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }

    .notation-chip p {
      margin: 0;
      color: var(--text-secondary);
      font-size: 0.94rem;
      line-height: 1.5;
    }

    .job-list,
    .artifact-grid,
    .deferred-grid,
    .robustness-grid {
      display: grid;
      gap: 20px;
    }

    .job-block,
    .artifact-card,
    .deferred-card,
    .robustness-card {
      padding: 22px;
    }

    .job-block {
      border-top: 1px solid var(--border);
      padding-top: 22px;
    }

    .job-top {
      display: grid;
      grid-template-columns: minmax(0, 0.9fr) minmax(240px, 0.7fr);
      gap: 18px;
      align-items: start;
      margin-bottom: 18px;
    }

    .job-top h3 {
      font-size: 1.65rem;
      margin-bottom: 8px;
    }

    .meta-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }

    .meta-row span {
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      padding: 0 12px;
      border-radius: 999px;
      background: var(--accent-muted);
      color: var(--accent);
      font-size: 0.76rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }

    .branch-note {
      margin-top: 12px;
      max-width: 42rem;
      font-size: 0.84rem;
      color: var(--text-secondary);
    }

    .mini-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }

    .mini-chart-card {
      padding: 14px 14px 8px;
      display: grid;
      gap: 10px;
      min-height: 300px;
    }

    .mini-chart-card h4 {
      font-size: 1rem;
    }

    .mini-chart {
      min-height: 220px;
    }

    .mini-note {
      font-size: 0.82rem;
      color: var(--text-tertiary);
    }

    .artifact-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .artifact-card,
    .deferred-card {
      display: grid;
      gap: 12px;
    }

    .slot-label {
      font-size: 0.76rem;
      font-weight: 800;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--accent-light);
    }

    .artifact-card h3,
    .deferred-card h3,
    .robustness-card h3 {
      font-size: 1.1rem;
    }

    .robustness-overview {
      display: grid;
      grid-template-columns: minmax(0, 0.9fr) minmax(260px, 0.7fr);
      gap: 18px;
      align-items: start;
      padding: 22px;
    }

    .robustness-overview h3 {
      font-size: 1.5rem;
      margin-bottom: 8px;
    }

    .robustness-stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }

    .robustness-stat {
      padding: 14px 16px;
      border-radius: 12px;
      background: rgba(127, 127, 127, 0.05);
      border: 1px solid var(--border);
    }

    .robustness-stat span {
      display: block;
      margin-bottom: 6px;
      font-size: 0.74rem;
      font-weight: 800;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--text-tertiary);
    }

    .robustness-stat strong {
      font-size: 1.35rem;
      color: var(--accent-light);
    }

    .robustness-card {
      display: grid;
      gap: 16px;
    }

    .robustness-chart {
      min-height: 260px;
    }

    .robustness-links {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .table-shell {
      padding: 18px;
      overflow-x: auto;
    }

    .result-table {
      width: 100%;
      border-collapse: collapse;
      min-width: 720px;
    }

    .result-table th,
    .result-table td {
      padding: 12px 10px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      font-size: 0.88rem;
    }

    .result-table th {
      font-size: 0.75rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--text-tertiary);
      font-weight: 800;
    }

    .artifact-layout {
      display: grid;
      gap: 20px;
    }

    .artifact-layout .section-block {
      padding: 22px;
    }

    .artifact-downloads {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 14px;
    }

    .gallery-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }

    .site-footer {
      padding: 72px 0 30px;
      border-top: 1px solid rgba(127, 127, 127, 0.08);
    }

    .footer-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(300px, 0.7fr);
      gap: 24px;
    }

    .footer-note {
      margin-top: 14px;
      font-size: 0.82rem;
      color: var(--text-tertiary);
    }

    .reveal {
      opacity: 0;
      transform: translateY(14px);
      transition: opacity 0.45s ease, transform 0.45s ease;
    }

    .reveal.visible {
      opacity: 1;
      transform: translateY(0);
    }

    @media (max-width: 980px) {
      .hero,
      .section-header,
      .job-top,
      .footer-grid,
      .robustness-overview {
        grid-template-columns: 1fr;
      }

      .hero {
        min-height: auto;
      }

      .insight-grid,
      .equation-grid,
      .notation-grid,
      .mini-grid,
      .artifact-grid,
      .gallery-grid,
      .robustness-stats {
        grid-template-columns: 1fr;
      }
    }

    @media (max-width: 720px) {
      .container {
        padding: 0 16px;
      }

      .nav-links {
        gap: 10px;
      }

      .hero-copy h1 {
        font-size: clamp(2rem, 10vw, 3rem);
      }

      .metric-grid {
        grid-template-columns: 1fr;
      }
    }
    """
).strip()

THEME_JS_TEXT = dedent(
    """
    (function () {
      const STORAGE_KEY = "ea-tdc-theme";
      const media = window.matchMedia("(prefers-color-scheme: dark)");
      const SUN = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>';
      const MOON = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';

      function readSavedTheme() {
        try {
          return localStorage.getItem(STORAGE_KEY);
        } catch {
          return null;
        }
      }

      function writeSavedTheme(theme) {
        try {
          localStorage.setItem(STORAGE_KEY, theme);
        } catch {
          // Ignore storage failures.
        }
      }

      function currentSystemTheme() {
        return media.matches ? "dark" : "light";
      }

      function effectiveTheme() {
        return readSavedTheme() || currentSystemTheme();
      }

      function updateThemeColor(theme) {
        const metas = document.querySelectorAll('meta[name="theme-color"]');
        metas.forEach((meta) => {
          meta.setAttribute("content", theme === "dark" ? "#0d1117" : "#fafbfd");
        });
      }

      function syncToggle(theme) {
        const button = document.getElementById("theme-toggle");
        if (!button) {
          return;
        }
        button.innerHTML = theme === "dark" ? SUN : MOON;
        button.setAttribute("aria-pressed", String(theme === "dark"));
        button.setAttribute("aria-label", `Switch to ${theme === "dark" ? "light" : "dark"} mode`);
      }

      function applyTheme(theme, { persist = false } = {}) {
        document.documentElement.dataset.theme = theme;
        document.documentElement.style.colorScheme = theme === "dark" ? "only dark" : "only light";
        updateThemeColor(theme);
        syncToggle(theme);
        if (persist) {
          writeSavedTheme(theme);
        }
        window.dispatchEvent(new CustomEvent("ea-tdc-themechange", { detail: { theme } }));
      }

      function initToggle() {
        const button = document.getElementById("theme-toggle");
        if (!button || button.dataset.themeBound === "true") {
          syncToggle(effectiveTheme());
          return;
        }
        button.dataset.themeBound = "true";
        syncToggle(effectiveTheme());
        button.addEventListener("click", () => {
          const nextTheme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
          applyTheme(nextTheme, { persist: true });
        });
      }

      window.eaTdcTheme = {
        applyTheme,
        effectiveTheme,
        initToggle,
      };

      applyTheme(effectiveTheme());

      document.addEventListener("DOMContentLoaded", () => {
        initToggle();
      });

      media.addEventListener("change", () => {
        if (!readSavedTheme()) {
          applyTheme(currentSystemTheme());
        }
      });
    })();
    """
).strip()

JS_TEXT = dedent(
    """
    (function () {
      'use strict';

      var DATA_FILE = 'assets/data/site_data.json';
      var ROOT_PREFIX = '';
      var PAGE = 'home';
      var ARTIFACT_ID = '';
      var SITE_VERSION = '';
      var SITE_DATA = null;

      function initPageConfig() {
        var body = document.body || document.getElementsByTagName('body')[0];
        if (!body) return;
        ROOT_PREFIX = body.getAttribute('data-root-prefix') || '';
        PAGE = body.getAttribute('data-page') || 'home';
        ARTIFACT_ID = body.getAttribute('data-artifact-id') || '';
        SITE_VERSION = body.getAttribute('data-site-version') || '';
      }

      function resolvePath(path) {
        return ROOT_PREFIX + path;
      }

      function versionedPath(path) {
        if (!SITE_VERSION) return path;
        return path + (path.indexOf('?') === -1 ? '?v=' : '&v=') + encodeURIComponent(SITE_VERSION);
      }

      function endsWith(text, suffix) {
        return String(text).slice(-suffix.length) === suffix;
      }

      function isDownloadable(path) {
        var suffixes = ['.csv', '.json', '.svg', '.md'];
        for (var i = 0; i < suffixes.length; i++) {
          if (endsWith(path, suffixes[i])) return true;
        }
        return false;
      }

      function escapeHtml(value) {
        return String(value == null ? '' : value)
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;')
          .replace(/'/g, '&#39;');
      }

      function renderDefinitionList(definitions) {
        if (!definitions || !definitions.length) return '';
        var rows = [];
        for (var i = 0; i < definitions.length; i++) {
          var item = definitions[i] || [];
          var symbolHtml = item.length > 2 ? String(item[2] || '') : '';
          var symbolMarkup = symbolHtml
            ? '<span class="equation-symbol">' + symbolHtml + '</span>'
            : '<span class="math-inline">\\\\(' + escapeHtml(item[0] || '') + '\\\\)</span>';
          rows.push(
            '<div class="definition-row"><dt>' +
            symbolMarkup +
            '</dt><dd>' +
            escapeHtml(item[1] || '') +
            '</dd></div>'
          );
        }
        return '<dl class="equation-definitions">' + rows.join('') + '</dl>';
      }

      function renderEquationMarkup(latex, htmlMarkup) {
        if (htmlMarkup) {
          return '<div class="math"><div class="equation-rendered">' + htmlMarkup + '</div></div>';
        }
        return '<div class="math">\\\\[' + escapeHtml(latex || '') + '\\\\]</div>';
      }

      function linkMarkup(link, className) {
        var cls = className || 'link-chip';
        var download = isDownloadable(link.href) ? ' download' : '';
        return '<a class="' + cls + '" href="' + escapeHtml(resolvePath(link.href)) + '"' + download + '>' + escapeHtml(link.label) + '</a>';
      }

      function getThemePalette() {
        var dark = (document.documentElement.getAttribute('data-theme') || 'light') === 'dark';
        if (dark) {
          return {
            paper: '#1c2128',
            plot: '#1c2128',
            text: '#e6edf3',
            grid: '#30363d',
            axis: '#4a525d',
            hoverBg: '#242a33',
            hoverFont: '#ffffff',
            colors: ['#58a6ff', '#d29922', '#f85149', '#3fb950', '#c27abf', '#e0a050']
          };
        }
        return {
          paper: '#ffffff',
          plot: '#ffffff',
          text: '#17171a',
          grid: '#e2e8f0',
          axis: '#9aa5b3',
          hoverBg: '#19385f',
          hoverFont: '#ffffff',
          colors: ['#2b6cb0', '#d69e2e', '#9b2c2c', '#276749', '#805ad5', '#dd6b20']
        };
      }

      function rgba(hex, alpha) {
        var value = String(hex).replace('#', '');
        var bigint = parseInt(value, 16);
        var r = (bigint >> 16) & 255;
        var g = (bigint >> 8) & 255;
        var b = bigint & 255;
        return 'rgba(' + r + ', ' + g + ', ' + b + ', ' + alpha + ')';
      }

      function requestText(path, callback) {
        var xhr = new XMLHttpRequest();
        xhr.open('GET', resolvePath(path), true);
        xhr.onreadystatechange = function () {
          if (xhr.readyState !== 4) return;
          if (xhr.status >= 200 && xhr.status < 300) {
            callback(null, xhr.responseText);
          } else {
            callback(new Error('Failed to load ' + path + ': ' + xhr.status));
          }
        };
        xhr.send();
      }

      function requestJson(path, callback) {
        requestText(path, function (error, text) {
          if (error) {
            callback(error);
            return;
          }
          try {
            callback(null, JSON.parse(text));
          } catch (parseError) {
            callback(parseError);
          }
        });
      }

      function parseCsv(text) {
        var lines = String(text || '').trim().split(/\\r?\\n/);
        if (!lines.length || !lines[0]) return [];
        var headers = lines[0].split(',');
        var rows = [];
        for (var i = 1; i < lines.length; i++) {
          if (!lines[i]) continue;
          var values = lines[i].split(',');
          var row = {};
          for (var j = 0; j < headers.length; j++) {
            row[headers[j]] = values[j] !== undefined ? values[j] : '';
          }
          rows.push(row);
        }
        return rows;
      }

      function loadSiteData(callback) {
        requestJson(versionedPath(DATA_FILE), function (error, data) {
          if (error) {
            callback(error);
            return;
          }
          SITE_DATA = data;
          callback(null, data);
        });
      }

      function typesetMath(targets) {
        if (window.MathJax && window.MathJax.typesetPromise) {
          return window.MathJax.typesetPromise(targets || []);
        }
        return Promise.resolve();
      }

      function scheduleTypeset(targets, attempt) {
        var tries = attempt || 0;
        if (window.MathJax && window.MathJax.typesetPromise) {
          typesetMath(targets);
          return;
        }
        if (tries >= 40) {
          return;
        }
        window.setTimeout(function () {
          scheduleTypeset(targets, tries + 1);
        }, 250);
      }

      function revealAll() {
        if (!window.IntersectionObserver) return;
        var observer = new IntersectionObserver(function (entries) {
          entries.forEach(function (entry) {
            if (entry.isIntersecting) {
              entry.target.classList.add('visible');
              observer.unobserve(entry.target);
            }
          });
        }, { threshold: 0.08 });
        var nodes = document.querySelectorAll('.reveal');
        for (var i = 0; i < nodes.length; i++) {
          observer.observe(nodes[i]);
        }
      }

      function renderMetrics() {
        var target = document.getElementById('metric-grid');
        if (!target || !SITE_DATA) return;
        var html = '';
        var metrics = SITE_DATA.metrics || [];
        for (var i = 0; i < metrics.length; i++) {
          html += '<article class="metric-card reveal"><span>' + escapeHtml(metrics[i].label) + '</span><strong>' + escapeHtml(metrics[i].value) + '</strong><p>' + escapeHtml(metrics[i].note) + '</p></article>';
        }
        target.innerHTML = html;
      }

      function renderInsightCards(cards, targetId) {
        var target = document.getElementById(targetId || 'insight-grid');
        if (!target) return;
        var html = '';
        for (var i = 0; i < cards.length; i++) {
          html += '<article class="insight-card reveal"><div class="eyebrow">' + escapeHtml(cards[i].kicker) + '</div><h3>' + escapeHtml(cards[i].title) + '</h3><p>' + escapeHtml(cards[i].body) + '</p></article>';
        }
        target.innerHTML = html;
      }

      function buildLinkRow(links, className) {
        var html = '';
        for (var i = 0; i < links.length; i++) {
          html += linkMarkup(links[i], className);
        }
        return html;
      }

      function renderJobList(jobIds, targetId) {
        var target = document.getElementById(targetId || 'job-list');
        if (!target || !SITE_DATA) return;
        var html = '';
        for (var i = 0; i < jobIds.length; i++) {
          var job = SITE_DATA.jobs[jobIds[i]];
          if (!job) continue;
          var branchNote = job.branch_note ? '<p class="branch-note">' + escapeHtml(job.branch_note) + '</p>' : '';
          html += '<section class="section-block job-block reveal" id="' + escapeHtml(job.anchor) + '">';
          html += '<div class="job-top"><div><div class="eyebrow">' + escapeHtml(job.kicker) + '</div><h3>' + escapeHtml(job.title) + '</h3><p>' + escapeHtml(job.subtitle) + '</p>';
          html += '<div class="meta-row"><span>' + escapeHtml(job.estimator_label) + '</span><span>' + escapeHtml(job.observation_label) + '</span><span>' + escapeHtml(job.treatment_label) + '</span><span>' + escapeHtml(job.branch_label) + '</span></div>' + branchNote + '</div>';
          html += '<div><p>' + escapeHtml(job.summary) + '</p><div class="button-row" style="margin-top:14px;">' + buildLinkRow(job.links) + '</div></div></div>';
          html += '<div class="mini-grid">';
          for (var j = 0; j < job.outcomes.length; j++) {
            var outcome = job.outcomes[j];
            html += '<article class="mini-chart-card"><div><h4>' + escapeHtml(outcome.label) + '</h4><p class="mini-note">Quarter-by-quarter response under the selected public control branch.</p></div><div class="mini-chart" id="' + escapeHtml(outcome.chart_dom_id) + '"></div></article>';
          }
          html += '</div></section>';
        }
        target.innerHTML = html;
      }

      function renderDepositAccounting() {
        var target = document.getElementById('independent-evidence');
        if (!target || !SITE_DATA || !SITE_DATA.home || !SITE_DATA.home.independent_evidence) return;
        var block = SITE_DATA.home.independent_evidence;
        var html = '';
          html += '<section class="section-block job-block reveal" id="independent-evidence-block">';
          html += '<div class="job-top"><div><div class="eyebrow">Independent non-TDC evidence</div><h3>' + escapeHtml(block.title) + '</h3><p>' + escapeHtml(block.subtitle) + '</p></div>';
          html += '<div><p>' + escapeHtml(block.summary) + '</p><p class="branch-note">' + escapeHtml(block.impact_summary || '') + '</p><div class="button-row" style="margin-top:14px;">' + buildLinkRow(block.links || []) + '</div></div></div>';
          html += '<div class="mini-grid">';
          for (var i = 0; i < block.outcomes.length; i++) {
            var outcome = block.outcomes[i];
            html += '<article class="mini-chart-card"><div><h4>' + escapeHtml(outcome.label) + '</h4><p class="mini-note">Quarter-by-quarter response under the imported `tdcpass` strict-source comparison.</p></div><div class="mini-chart" id="' + escapeHtml(outcome.chart_dom_id) + '"></div></article>';
          }
          html += '</div>';
          html += '<div class="insight-grid" style="margin-top:18px;">';
          for (var j = 0; j < (block.note_lines || []).length; j++) {
            html += '<article class="insight-card"><div class="slot-label">Boundary</div><p>' + escapeHtml(block.note_lines[j]) + '</p></article>';
          }
          html += '</div>';
        html += '</section>';
        target.innerHTML = html;
      }

      function renderTreatmentComparisons() {
        var target = document.getElementById('treatment-comparisons');
        if (!target || !SITE_DATA || !SITE_DATA.sidecar || !SITE_DATA.sidecar.treatment_comparisons) return;
        var blocks = SITE_DATA.sidecar.treatment_comparisons;
        var html = '';
        for (var i = 0; i < blocks.length; i++) {
          var block = blocks[i];
          html += '<section class="section-block job-block reveal">';
          html += '<div class="job-top"><div><div class="eyebrow">Treatment variants</div><h3>' + escapeHtml(block.title) + '</h3><p>' + escapeHtml(block.subtitle) + '</p></div>';
          html += '<div><p>' + escapeHtml(block.summary) + '</p><div class="button-row" style="margin-top:14px;">' + buildLinkRow(block.links) + '</div></div></div>';
          html += '<div class="mini-grid">';
          for (var j = 0; j < block.outcomes.length; j++) {
            var outcome = block.outcomes[j];
            html += '<article class="mini-chart-card"><div><h4>' + escapeHtml(outcome.label) + '</h4><p class="mini-note">Each line shows the same quarterly response under a different TDC construction.</p></div><div class="mini-chart" id="' + escapeHtml(outcome.chart_dom_id) + '"></div></article>';
          }
          html += '</div></section>';
        }
        target.innerHTML = html;
      }

      function renderIvLabSummary() {
        var target = document.getElementById('iv-lab-summary');
        if (!target || !SITE_DATA || !SITE_DATA.sidecar || !SITE_DATA.sidecar.iv_lab) return;
        var block = SITE_DATA.sidecar.iv_lab;
        var html = '<section class="section-block reveal"><div class="job-top"><div><div class="eyebrow">IV search</div><h3>IV mining results.</h3><p>The current search scans available shock × state candidates and compares the configured instrument with the best alternative found in the current data.</p></div><div><div class="meta-row"><span>' + escapeHtml(String(block.jobs_scanned)) + ' IV jobs</span><span>' + escapeHtml(String(block.total_candidates)) + ' candidates</span></div><div class="button-row" style="margin-top:14px;">' + buildLinkRow(block.links) + '</div></div></div>';
        html += '<div class="gallery-grid">';
        for (var i = 0; i < block.jobs.length; i++) {
          var job = block.jobs[i];
          html += '<article class="artifact-card"><div class="slot-label">IV search</div><h3>' + escapeHtml(job.job_id) + '</h3><p>Current median F: ' + escapeHtml(job.current_median_f == null ? 'n/a' : Number(job.current_median_f).toFixed(2)) + ' • Best median F: ' + escapeHtml(job.best_median_f == null ? 'n/a' : Number(job.best_median_f).toFixed(2)) + '</p><p>Current: ' + escapeHtml(job.current_recommendation || 'n/a') + '<br>Best alternative: ' + escapeHtml(job.best_recommendation || 'n/a') + '</p></article>';
        }
        html += '</div></section>';
        target.innerHTML = html;
      }

      function renderRobustnessSummary() {
        var overviewTarget = document.getElementById('robustness-overview');
        var gridTarget = document.getElementById('robustness-grid');
        if (!overviewTarget || !gridTarget || !SITE_DATA || !SITE_DATA.robustness) return;
        var robustness = SITE_DATA.robustness;
        var overview = robustness.overview || {};
        overviewTarget.innerHTML =
          '<section class="section-block robustness-overview reveal">' +
          '<div><div class="eyebrow">Robustness</div><h3>' + escapeHtml(overview.title || 'Larger control set') + '</h3><p>' + escapeHtml(overview.summary || '') + '</p><div class="button-row" style="margin-top:14px;">' + buildLinkRow(overview.links || []) + '</div></div>' +
          '<div class="robustness-stats">' +
          '<article class="robustness-stat"><span>Raw series</span><strong>' + escapeHtml(String(overview.series_count || '0')) + '</strong></article>' +
          '<article class="robustness-stat"><span>Lagged indicators</span><strong>' + escapeHtml(String(overview.feature_count || '0')) + '</strong></article>' +
          '<article class="robustness-stat"><span>Daily lags</span><strong>' + escapeHtml(String(overview.daily_lags || '0')) + '</strong></article>' +
          '<article class="robustness-stat"><span>Common K</span><strong>' + escapeHtml(String(overview.recommended_k_mode || 'n/a')) + '</strong></article>' +
          '</div></section>';

        var html = '';
        for (var i = 0; i < robustness.jobs.length; i++) {
          var job = robustness.jobs[i];
          html += '<article class="robustness-card section-block reveal">';
          html += '<div class="slot-label">Robustness</div><h3>' + escapeHtml(job.title) + '</h3><p>' + escapeHtml(job.summary) + '</p>';
          html += '<div class="meta-row"><span>Cross-check: ' + escapeHtml(String(job.ml_public_branch_label || 'DML')) + '</span><span>K=' + escapeHtml(String(job.recommended_k || 'n/a')) + '</span><span>' + escapeHtml(String(job.screened_feature_count || '0')) + ' lagged indicators</span><span>' + escapeHtml(String(job.treatment_variant_count || '0')) + ' treatment variants</span></div>';
          html += '<div class="mini-grid">';
          html += '<article class="mini-chart-card"><div><h4>Control ladder</h4><p class="mini-note">Average absolute response magnitude under the baseline and expanded control sets.</p></div><div class="robustness-chart" id="' + escapeHtml(job.chart_dom_id) + '"></div></article>';
          html += '<article class="mini-chart-card"><div><h4>Interpretation</h4><p class="mini-note">' + escapeHtml(job.interpretation) + '</p><p class="mini-note">' + escapeHtml(job.ml_public_branch_reason || '') + '</p></div><div class="robustness-links">' + buildLinkRow(job.links || []) + '</div></article>';
          html += '</div></article>';
        }
        gridTarget.innerHTML = html;
      }

      function renderArtifacts(mainRows, appendixRows) {
        var mainTarget = document.getElementById('artifact-grid-main');
        var appendixTarget = document.getElementById('artifact-grid-appendix');
        if (!mainTarget || !appendixTarget || !SITE_DATA) return;

        function cardMarkup(rows) {
          var html = '';
          for (var i = 0; i < rows.length; i++) {
            html += '<article class="artifact-card reveal"><div class="slot-label">' + escapeHtml(rows[i].slot_label) + '</div><h3>' + escapeHtml(rows[i].title) + '</h3><p>' + escapeHtml(rows[i].subtitle) + '</p><div class="button-row">' + buildLinkRow(rows[i].links) + '</div></article>';
          }
          return html;
        }

        mainTarget.innerHTML = cardMarkup(mainRows);
        appendixTarget.innerHTML = cardMarkup(appendixRows);
      }

      function renderDeferred(rows) {
        var target = document.getElementById('deferred-grid');
        if (!target) return;
        var html = '';
        for (var i = 0; i < rows.length; i++) {
          var meta = '';
          for (var j = 0; j < rows[i].meta.length; j++) {
            meta += '<span>' + escapeHtml(rows[i].meta[j]) + '</span>';
          }
          html += '<article class="deferred-card reveal"><div class="slot-label">' + escapeHtml(rows[i].tag) + '</div><h3>' + escapeHtml(rows[i].title) + '</h3><p>' + escapeHtml(rows[i].reason) + '</p><div class="meta-row">' + meta + '</div></article>';
        }
        target.innerHTML = html;
      }

      function renderGallery(rows) {
        var target = document.getElementById('artifact-gallery-grid');
        if (!target) return;
        var html = '';
        for (var i = 0; i < rows.length; i++) {
          html += '<article class="artifact-card reveal"><div class="slot-label">' + escapeHtml(rows[i].slot_label) + '</div><h3>' + escapeHtml(rows[i].title) + '</h3><p>' + escapeHtml(rows[i].subtitle) + '</p><div class="button-row">' + buildLinkRow(rows[i].links) + '</div></article>';
        }
        target.innerHTML = html;
      }

      function renderArtifactTableRows(rows) {
        var html = '';
        for (var i = 0; i < rows.length; i++) {
          html += '<tr><td>' + escapeHtml(rows[i].outcome_label) + '</td><td>' + escapeHtml(rows[i].horizon) + '</td><td>' + escapeHtml(rows[i].beta) + '</td><td>' + escapeHtml(rows[i].se) + '</td><td>' + escapeHtml(rows[i].lower95) + '</td><td>' + escapeHtml(rows[i].upper95) + '</td><td>' + escapeHtml(rows[i].p_value_normal) + '</td><td>' + escapeHtml(rows[i].n) + '</td></tr>';
        }
        return html;
      }

      function renderArtifactDetail(artifact, callback) {
        var summaryTarget = document.getElementById('artifact-summary');
        var bodyTarget = document.getElementById('artifact-body');
        if (!summaryTarget || !bodyTarget || !artifact || !SITE_DATA) {
          if (callback) callback();
          return;
        }
        summaryTarget.innerHTML = '<div class="eyebrow">' + escapeHtml(artifact.slot_label) + '</div><h1>' + escapeHtml(artifact.title) + '</h1><p>' + escapeHtml(artifact.subtitle) + '</p><div class="artifact-downloads">' + buildLinkRow(artifact.links, 'button') + '</div>';

        if (artifact.kind === 'figure') {
          var job = SITE_DATA.jobs[artifact.job_id];
          var outcomes = [];
          for (var i = 0; i < job.outcomes.length; i++) {
            if ((artifact.outcome_ids || []).indexOf(job.outcomes[i].key) !== -1) outcomes.push(job.outcomes[i]);
          }
          var figureHtml = '<section class="section-block reveal"><p>' + escapeHtml(artifact.caption) + '</p><div class="mini-grid" style="margin-top:20px;">';
          for (var j = 0; j < outcomes.length; j++) {
            figureHtml += '<article class="mini-chart-card"><div><h4>' + escapeHtml(outcomes[j].label) + '</h4><p class="mini-note">Quarter-by-quarter response of this outcome.</p></div><div class="mini-chart" id="' + escapeHtml(artifact.artifact_id + '-' + outcomes[j].key) + '"></div></article>';
          }
          figureHtml += '</div></section>';
          bodyTarget.innerHTML = figureHtml;
          if (callback) callback();
          return;
        }

        var jobRows = [];
        var jobData = SITE_DATA.jobs[artifact.job_id];
        var k;
        if (artifact.table_rows && artifact.table_rows.length) {
          jobRows = artifact.table_rows;
          bodyTarget.innerHTML = '<section class="table-shell reveal"><p style="margin-bottom:14px;">' + escapeHtml(artifact.caption) + '</p><table class="result-table"><thead><tr><th>Outcome</th><th>H</th><th>Beta</th><th>SE</th><th>95% lower</th><th>95% upper</th><th>p-value</th><th>N</th></tr></thead><tbody>' + renderArtifactTableRows(jobRows) + '</tbody></table></section>';
          if (callback) callback();
          return;
        }

        if (artifact.outcome_ids && artifact.outcome_ids.length && artifact.horizons && artifact.horizons.length) {
          for (k = 0; k < jobData.table_rows.length; k++) {
            if (artifact.outcome_ids.indexOf(jobData.table_rows[k].outcome) !== -1 && artifact.horizons.indexOf(jobData.table_rows[k].horizon) !== -1) {
              jobRows.push(jobData.table_rows[k]);
            }
          }
          bodyTarget.innerHTML = '<section class="table-shell reveal"><p style="margin-bottom:14px;">' + escapeHtml(artifact.caption) + '</p><table class="result-table"><thead><tr><th>Outcome</th><th>H</th><th>Beta</th><th>SE</th><th>95% lower</th><th>95% upper</th><th>p-value</th><th>N</th></tr></thead><tbody>' + renderArtifactTableRows(jobRows) + '</tbody></table></section>';
          if (callback) callback();
          return;
        }

        var csvLink = null;
        for (k = 0; k < artifact.links.length; k++) {
          if (artifact.links[k].label === 'CSV') csvLink = artifact.links[k];
        }
        if (!csvLink) {
          if (callback) callback();
          return;
        }

        requestText(csvLink.href, function (error, text) {
          var rows = [];
          if (!error) {
            var csvRows = parseCsv(text);
            for (var m = 0; m < csvRows.length; m++) {
              rows.push({
                outcome_label: String(csvRows[m].outcome || '').split('_').join(' '),
                horizon: csvRows[m].horizon,
                beta: csvRows[m].beta,
                se: csvRows[m].se,
                lower95: csvRows[m].lower95,
                upper95: csvRows[m].upper95,
                p_value_normal: csvRows[m].p_value_normal,
                n: csvRows[m].n
              });
            }
          }
          bodyTarget.innerHTML = '<section class="table-shell reveal"><p style="margin-bottom:14px;">' + escapeHtml(artifact.caption) + '</p><table class="result-table"><thead><tr><th>Outcome</th><th>H</th><th>Beta</th><th>SE</th><th>95% lower</th><th>95% upper</th><th>p-value</th><th>N</th></tr></thead><tbody>' + renderArtifactTableRows(rows) + '</tbody></table></section>';
          if (callback) callback();
        });
      }

      function renderChart(domId, outcome, colorIndex) {
        var container = document.getElementById(domId);
        if (!container || !outcome || typeof Plotly === 'undefined') return;
        var palette = getThemePalette();
        var color = palette.colors[colorIndex % palette.colors.length];
        var x = [];
        var beta = [];
        var lower = [];
        var upper = [];
        var customdata = [];
        for (var i = 0; i < outcome.points.length; i++) {
          x.push(outcome.points[i].horizon);
          beta.push(outcome.points[i].beta);
          lower.push(outcome.points[i].lower95);
          upper.push(outcome.points[i].upper95);
          customdata.push([outcome.points[i].p_value, outcome.points[i].lower95, outcome.points[i].upper95]);
        }
        var traces = [
          { x: x, y: lower, mode: 'lines', line: { width: 0 }, hoverinfo: 'skip', showlegend: false },
          { x: x, y: upper, mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: rgba(color, 0.14), hoverinfo: 'skip', showlegend: false },
          {
            x: x,
            y: beta,
            mode: 'lines+markers',
            line: { color: color, width: 2.3 },
            marker: { color: color, size: 6, line: { color: color, width: 1.4 } },
            customdata: customdata,
            hovertemplate: 'h=%{x}<br>%{y:.4f}<br>p=%{customdata[0]:.3f}<br>95% CI [%{customdata[1]:.4f}, %{customdata[2]:.4f}]<extra></extra>'
          }
        ];
        var layout = {
          autosize: true,
          margin: { t: 8, r: 10, b: 42, l: 50 },
          paper_bgcolor: palette.paper,
          plot_bgcolor: palette.plot,
          font: { color: palette.text, family: 'Inter, sans-serif', size: 13 },
          hovermode: 'x unified',
          hoverlabel: { bgcolor: palette.hoverBg, font: { color: palette.hoverFont } },
          xaxis: {
            title: { text: 'Horizon', font: { size: 11 } },
            tickmode: 'linear',
            dtick: 1,
            gridcolor: palette.grid,
            linecolor: palette.axis,
            zeroline: false
          },
          yaxis: {
            title: { text: 'Response', font: { size: 11 } },
            gridcolor: palette.grid,
            zeroline: true,
            zerolinecolor: palette.axis,
            zerolinewidth: 1.1
          },
          showlegend: false
        };
        Plotly.react(container, traces, layout, { displayModeBar: false, responsive: true });
      }

      function renderComparisonChart(domId, outcome) {
        var container = document.getElementById(domId);
        if (!container || !outcome || typeof Plotly === 'undefined') return;
        var palette = getThemePalette();
        var traces = [];
        for (var i = 0; i < outcome.lines.length; i++) {
          var line = outcome.lines[i];
          var color = palette.colors[i % palette.colors.length];
          var x = [];
          var beta = [];
          var lower = [];
          var upper = [];
          var customdata = [];
          for (var j = 0; j < line.points.length; j++) {
            x.push(line.points[j].horizon);
            beta.push(line.points[j].beta);
            lower.push(line.points[j].lower95);
            upper.push(line.points[j].upper95);
            customdata.push([line.points[j].p_value, line.points[j].lower95, line.points[j].upper95]);
          }
          traces.push({
            x: x,
            y: lower,
            mode: 'lines',
            line: { width: 0 },
            hoverinfo: 'skip',
            showlegend: false
          });
          traces.push({
            x: x,
            y: upper,
            mode: 'lines',
            line: { width: 0 },
            fill: 'tonexty',
            fillcolor: rgba(color, i === 0 ? 0.16 : 0.08),
            hoverinfo: 'skip',
            showlegend: false
          });
          traces.push({
            x: x,
            y: beta,
            mode: 'lines+markers',
            name: line.label,
            line: { color: color, width: i === 0 ? 2.8 : 2 },
            marker: { color: color, size: i === 0 ? 7 : 5 },
            customdata: customdata,
            hovertemplate: '%{fullData.name}<br>h=%{x}<br>%{y:.4f}<br>p=%{customdata[0]:.3f}<br>95% CI [%{customdata[1]:.4f}, %{customdata[2]:.4f}]<extra></extra>'
          });
        }
        var layout = {
          autosize: true,
          margin: { t: 8, r: 10, b: 42, l: 50 },
          paper_bgcolor: palette.paper,
          plot_bgcolor: palette.plot,
          font: { color: palette.text, family: 'Inter, sans-serif', size: 13 },
          hovermode: 'x unified',
          hoverlabel: { bgcolor: palette.hoverBg, font: { color: palette.hoverFont } },
          legend: { orientation: 'h', y: -0.28, x: 0, font: { size: 11 } },
          xaxis: {
            title: { text: 'Horizon', font: { size: 11 } },
            tickmode: 'linear',
            dtick: 1,
            gridcolor: palette.grid,
            linecolor: palette.axis,
            zeroline: false
          },
          yaxis: {
            title: { text: 'Response', font: { size: 11 } },
            gridcolor: palette.grid,
            zeroline: true,
            zerolinecolor: palette.axis,
            zerolinewidth: 1.1
          },
          showlegend: true
        };
        Plotly.react(container, traces, layout, { displayModeBar: false, responsive: true });
      }

      function renderRobustnessChart(domId, job, colorIndex) {
        var container = document.getElementById(domId);
        if (!container || !job || typeof Plotly === 'undefined') return;
        var palette = getThemePalette();
        var color = palette.colors[colorIndex % palette.colors.length];
        var x = [];
        var y = [];
        var text = [];
        var markerColors = [];
        for (var i = 0; i < job.ladder.length; i++) {
          x.push(job.ladder[i].label);
          y.push(job.ladder[i].avg_abs_beta);
          text.push('rows=' + job.ladder[i].rows_written);
          markerColors.push(i === job.recommended_index ? color : rgba(color, 0.45));
        }
        var traces = [
          {
            x: x,
            y: y,
            type: 'bar',
            marker: { color: markerColors },
            text: text,
            hovertemplate: '%{x}<br>|beta|=%{y:.4f}<br>%{text}<extra></extra>'
          }
        ];
        var layout = {
          autosize: true,
          margin: { t: 8, r: 10, b: 52, l: 58 },
          paper_bgcolor: palette.paper,
          plot_bgcolor: palette.plot,
          font: { color: palette.text, family: 'Inter, sans-serif', size: 13 },
          hoverlabel: { bgcolor: palette.hoverBg, font: { color: palette.hoverFont } },
          xaxis: {
            tickangle: -20,
            gridcolor: palette.grid,
            linecolor: palette.axis
          },
          yaxis: {
            title: { text: 'Average |response|' },
            gridcolor: palette.grid,
            zeroline: true,
            zerolinecolor: palette.axis,
            zerolinewidth: 1.1
          },
          showlegend: false
        };
        Plotly.react(container, traces, layout, { displayModeBar: false, responsive: true });
      }

      function renderAllCharts() {
        var i;
        var j;
        if (!SITE_DATA) return;
        if (PAGE === 'home') {
          for (i = 0; i < SITE_DATA.home.main_job_ids.length; i++) {
            var homeJob = SITE_DATA.jobs[SITE_DATA.home.main_job_ids[i]];
            var homeOutcomes = (homeJob && homeJob.outcomes) || [];
            for (j = 0; j < homeOutcomes.length; j++) {
              renderChart(homeOutcomes[j].chart_dom_id, homeOutcomes[j], j);
            }
          }
          var independentEvidence = SITE_DATA.home.independent_evidence;
          if (independentEvidence && independentEvidence.outcomes) {
            for (i = 0; i < independentEvidence.outcomes.length; i++) {
              renderChart(independentEvidence.outcomes[i].chart_dom_id, independentEvidence.outcomes[i], i);
            }
          }
          for (i = 0; i < SITE_DATA.sidecar.job_ids.length; i++) {
            var evidenceJob = SITE_DATA.jobs[SITE_DATA.sidecar.job_ids[i]];
            var evidenceOutcomes = (evidenceJob && evidenceJob.outcomes) || [];
            for (j = 0; j < evidenceOutcomes.length; j++) {
              renderChart(evidenceOutcomes[j].chart_dom_id, evidenceOutcomes[j], j);
            }
          }
          var treatmentComparisonsHome = SITE_DATA.sidecar.treatment_comparisons || [];
          for (i = 0; i < treatmentComparisonsHome.length; i++) {
            for (j = 0; j < treatmentComparisonsHome[i].outcomes.length; j++) {
              renderComparisonChart(treatmentComparisonsHome[i].outcomes[j].chart_dom_id, treatmentComparisonsHome[i].outcomes[j]);
            }
          }
          if (SITE_DATA.robustness && SITE_DATA.robustness.jobs) {
            for (i = 0; i < SITE_DATA.robustness.jobs.length; i++) {
              renderRobustnessChart(SITE_DATA.robustness.jobs[i].chart_dom_id, SITE_DATA.robustness.jobs[i], i);
            }
          }
        } else if (PAGE === 'sidecar') {
          for (i = 0; i < SITE_DATA.sidecar.job_ids.length; i++) {
            var sideJob = SITE_DATA.jobs[SITE_DATA.sidecar.job_ids[i]];
            var sideOutcomes = (sideJob && sideJob.outcomes) || [];
            for (j = 0; j < sideOutcomes.length; j++) {
              renderChart(sideOutcomes[j].chart_dom_id, sideOutcomes[j], j);
            }
          }
          var treatmentComparisons = SITE_DATA.sidecar.treatment_comparisons || [];
          for (i = 0; i < treatmentComparisons.length; i++) {
            for (j = 0; j < treatmentComparisons[i].outcomes.length; j++) {
              renderComparisonChart(treatmentComparisons[i].outcomes[j].chart_dom_id, treatmentComparisons[i].outcomes[j]);
            }
          }
          if (SITE_DATA.robustness && SITE_DATA.robustness.jobs) {
            for (i = 0; i < SITE_DATA.robustness.jobs.length; i++) {
              renderRobustnessChart(SITE_DATA.robustness.jobs[i].chart_dom_id, SITE_DATA.robustness.jobs[i], i);
            }
          }
        } else if (PAGE === 'artifact') {
          var artifact = SITE_DATA.artifacts[ARTIFACT_ID];
          if (!artifact || artifact.kind !== 'figure') return;
          var artifactJob = SITE_DATA.jobs[artifact.job_id];
          for (i = 0; i < artifactJob.outcomes.length; i++) {
            if ((artifact.outcome_ids || []).indexOf(artifactJob.outcomes[i].key) !== -1) {
              renderChart(artifact.artifact_id + '-' + artifactJob.outcomes[i].key, artifactJob.outcomes[i], i);
            }
          }
        }
      }

      function renderPage(callback) {
        if (!SITE_DATA) {
          if (callback) callback();
          return;
        }
        if (PAGE === 'home') {
          renderMetrics();
          renderInsightCards(SITE_DATA.home.insights, 'questions-grid');
          renderJobList(SITE_DATA.home.main_job_ids, 'headline-job-list');
          renderDepositAccounting();
          renderInsightCards(SITE_DATA.sidecar.insights, 'additional-evidence-grid');
          renderJobList(SITE_DATA.sidecar.job_ids, 'job-list-sidecar');
          renderTreatmentComparisons();
          renderIvLabSummary();
          renderRobustnessSummary();
          renderArtifacts(SITE_DATA.home.main_artifacts, SITE_DATA.home.appendix_artifacts);
          renderDeferred(SITE_DATA.deferred_jobs);
          if (callback) callback();
        } else if (PAGE === 'sidecar') {
          renderMetrics();
          renderInsightCards(SITE_DATA.sidecar.insights, 'additional-evidence-grid');
          renderJobList(SITE_DATA.sidecar.job_ids, 'job-list-sidecar');
          renderTreatmentComparisons();
          renderIvLabSummary();
          renderRobustnessSummary();
          if (callback) callback();
        } else if (PAGE === 'gallery') {
          renderGallery(SITE_DATA.artifact_gallery);
          if (callback) callback();
        } else if (PAGE === 'artifact') {
          renderArtifactDetail(SITE_DATA.artifacts[ARTIFACT_ID], callback);
        } else if (callback) {
          callback();
        }
      }

      function afterRender() {
        revealAll();
        renderAllCharts();
        scheduleTypeset();
      }

      document.addEventListener('DOMContentLoaded', function () {
        initPageConfig();
        if (window.eaTdcTheme && window.eaTdcTheme.initToggle) {
          window.eaTdcTheme.initToggle();
        }
        loadSiteData(function (error) {
          if (error) {
            console.error(error);
            return;
          }
          renderPage(function () {
            afterRender();
          });
        });
        window.addEventListener('resize', function () { renderAllCharts(); });
        window.addEventListener('ea-tdc-themechange', function () { renderAllCharts(); });
        document.addEventListener('toggle', function (event) {
          var target = event.target;
          if (target && target.tagName === 'DETAILS' && target.open) {
            scheduleTypeset([target]);
          }
        });
      });
    })();
    """
).strip()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _copy_tree(source: Path, target: Path) -> int:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    return sum(1 for path in target.rglob("*") if path.is_file())


def _copy_selected_files(paths: list[Path], target: Path) -> int:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    seen_names: set[str] = set()
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        if path.name in seen_names:
            continue
        shutil.copy2(path, target / path.name)
        seen_names.add(path.name)
        copied += 1
    return copied


_PUBLIC_REPORT_ALLOWLIST = {
    "component_sidecar_artifact_pack.md",
    "component_sidecar_liquidity_table.csv",
    "component_sidecar_reduced_form_table.csv",
    "component_sidecar_screening.csv",
    "component_sidecar_screening.md",
    "component_sidecar_state_probe_table.csv",
    "final_interpretation_closeout.md",
    "release_contract.json",
    "release_scorecard.json",
    "robustness_snapshot.json",
}


def _prune_public_reports(target: Path) -> None:
    for path in target.iterdir():
        if not path.is_file():
            continue
        if path.name not in _PUBLIC_REPORT_ALLOWLIST:
            path.unlink()


def _sanitize_public_tree(target: Path, repo_root: Path) -> None:
    root_text = str(repo_root.resolve())
    root_prefix = f"{root_text}/"
    text_suffixes = {".json", ".md", ".txt", ".html", ".csv", ".svg", ".js", ".css"}
    for path in target.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        sanitized = content.replace(root_prefix, "")
        if sanitized != content:
            path.write_text(sanitized, encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _coerce_float(value: str | float | int | None) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _fmt(value: str | float | int | None, digits: int = 4) -> str:
    number = _coerce_float(value)
    if number is None:
        return str(value or "")
    return f"{number:.{digits}f}"


def _humanize(text: str) -> str:
    normalized = str(text or "").replace("qoq ann", "QoQ ann").strip()
    if not normalized:
        return ""
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


def _symbol_display(symbol: str) -> str:
    display = str(symbol)
    replacements = {
        r"\Delta ": "Δ",
        r"\Delta": "Δ",
        r"\widehat{": "",
        r"\max": "max",
        r"\left(": "(",
        r"\right)": ")",
        r"\left": "",
        r"\right": "",
    }
    for old, new in replacements.items():
        display = display.replace(old, new)
    display = display.replace("{", "").replace("}", "")
    return display


def _job_meta(job_id: str) -> dict[str, str]:
    default_title = _humanize(job_id)
    return {
        "title": JOB_META.get(job_id, {}).get("title", default_title),
        "subtitle": JOB_META.get(job_id, {}).get("subtitle", f"Dynamic estimates for {default_title.lower()}."),
        "summary": JOB_META.get(job_id, {}).get("summary", f"This branch shows the current estimated response pattern for {default_title.lower()}."),
        "kicker": JOB_META.get(job_id, {}).get("kicker", "Result"),
    }


def _outcome_label(outcome: str) -> str:
    return OUTCOME_LABELS.get(outcome, _humanize(outcome))


FEATURE_LABEL_OVERRIDES = {
    "bogz1fl702050005q_usb_ffrrp": "Money market fund repo assets",
    "tgdef_net_gov_sav": "Net government saving",
    "bogz1lm403061105q_gses_ts": "GSE Treasury securities",
    "bogz1lm403061105q_gse_tsy": "GSE Treasury holdings",
    "bogz1fl704135005q_usb_loan_l": "Bank loans",
    "bogz1fl704141005q_usb_stloan_l": "Bank short-term loans",
    "dgs3_dgs3": "3-year Treasury yield",
}

FREQUENCY_LABELS = {
    "d": "Daily",
    "m": "Monthly",
    "q": "Quarterly",
}


def _feature_label(feature_id: str) -> str:
    text = str(feature_id or "").strip()
    if not text:
        return ""
    parts = text.split("__")
    if len(parts) < 2:
        return _humanize(text)
    freq = FREQUENCY_LABELS.get(parts[0], _humanize(parts[0]))
    series_key = parts[1]
    lag_label = ""
    if len(parts) >= 3 and parts[2].startswith("lag"):
        lag_digits = parts[2][3:].lstrip("0") or "0"
        lag_label = f", lag {lag_digits}"
    series_label = FEATURE_LABEL_OVERRIDES.get(series_key, _humanize(series_key))
    return f"{freq}: {series_label}{lag_label}"


def _factor_label(factor_id: str) -> str:
    text = str(factor_id or "").strip()
    if not text:
        return ""
    parts = text.split("_")
    if len(parts) >= 3 and parts[0] == "dflmx" and parts[1].startswith("k") and parts[2].startswith("f"):
        k_value = parts[1][1:]
        factor_value = parts[2][1:]
        return f"Factor {factor_value} (K={k_value})"
    return _humanize(text)


def _copied_model_href(model_path: Path) -> str:
    return f"site_assets/models/{model_path.name}"


def _resolve_repo_path(path_like: str | Path, paths: ProjectPaths) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return paths.root / path


def _copied_artifact_href(path: Path, paths: ProjectPaths) -> str:
    resolved = _resolve_repo_path(path, paths)
    relative = resolved.relative_to(paths.output / "artifacts").as_posix()
    return f"site_assets/artifacts/{relative}"


def _copied_report_href(path_text: str, paths: ProjectPaths) -> str:
    path = _resolve_repo_path(path_text, paths)
    if not path_text or not path.exists():
        return ""
    if path.name not in _PUBLIC_REPORT_ALLOWLIST:
        return ""
    return f"site_assets/reports/{path.name}"


def _site_model_paths(site_data: dict[str, Any], paths: ProjectPaths) -> list[Path]:
    model_paths: list[Path] = []
    seen: set[Path] = set()

    def add_from_href(href: str) -> None:
        text = str(href or "").strip()
        if not text.startswith("site_assets/models/"):
            return
        filename = text.split("/")[-1]
        candidate = paths.output / "models" / filename
        if candidate not in seen:
            seen.add(candidate)
            model_paths.append(candidate)

    jobs = site_data.get("jobs", {}) or {}
    for job in jobs.values():
        for link in job.get("links", []) or []:
            add_from_href(str((link or {}).get("href", "")))

    independent_evidence = site_data.get("home", {}).get("independent_evidence", {}) or {}
    for link in independent_evidence.get("links", []) or []:
        add_from_href(str((link or {}).get("href", "")))

    return model_paths


def _robustness_summary_for_job(paths: ProjectPaths, job_id: str) -> dict[str, Any]:
    summary_path = paths.manifests / f"{job_id}__robustness_summary.json"
    if not summary_path.exists():
        return {}
    return _read_json(summary_path)


def _selected_result_branch(paths: ProjectPaths, job_id: str, baseline_path: Path) -> dict[str, Any]:
    summary = _robustness_summary_for_job(paths, job_id)
    recommended_k = int(summary.get("recommended_k", 0) or 0)
    if recommended_k > 0:
        recommended_path = paths.output / "models" / f"{job_id}__robustness_k{recommended_k}_estimates.csv"
        if recommended_path.exists():
            return {
                "selected_path": recommended_path,
                "baseline_path": baseline_path,
                "branch_label": "Expanded control set",
                "branch_summary": (
                    f"Selected public version uses the expanded control set built from a screened pool of K={recommended_k} lagged indicators."
                ),
                "recommended_k": recommended_k,
                "robustness_summary": summary,
            }
    return {
        "selected_path": baseline_path,
        "baseline_path": baseline_path,
        "branch_label": "Baseline controls",
        "branch_summary": "Selected public version uses the baseline quarterly control set.",
        "recommended_k": 0,
        "robustness_summary": summary,
    }


def _find_estimates_path(job_id: str, scorecard_rows: dict[str, dict[str, Any]], paths: ProjectPaths) -> Path:
    candidate = str(scorecard_rows.get(job_id, {}).get("estimates_path", "")).strip()
    if candidate:
        return _resolve_repo_path(candidate, paths)
    return paths.output / "models" / f"{job_id}__lp_estimates.csv"


def _job_outcomes(rows: list[dict[str, str]], job_id: str) -> list[dict[str, Any]]:
    outcomes = []
    for outcome in sorted({row["outcome"] for row in rows}):
        points = []
        for row in sorted((item for item in rows if item["outcome"] == outcome), key=lambda item: int(item["horizon"])):
            beta = _coerce_float(row.get("beta"))
            lower = _coerce_float(row.get("lower95"))
            upper = _coerce_float(row.get("upper95"))
            p_value = _coerce_float(row.get("p_value_normal"))
            if beta is None or lower is None or upper is None:
                continue
            points.append(
                {
                    "horizon": int(row["horizon"]),
                    "beta": beta,
                    "lower95": lower,
                    "upper95": upper,
                    "p_value": p_value if p_value is not None else 1.0,
                }
            )
        if points:
            safe_outcome = outcome.replace("_", "-")
            outcomes.append(
                {
                    "key": outcome,
                    "label": _outcome_label(outcome),
                    "chart_dom_id": f"chart-{job_id.replace('_', '-')}-{safe_outcome}",
                    "points": points,
                }
            )
    return outcomes


def _select_outcomes(job: dict[str, Any], keys: list[str]) -> list[dict[str, Any]]:
    outcomes = list(job.get("outcomes", []))
    lookup = {str(item.get("key", "")): item for item in outcomes}
    selected = [lookup[key] for key in keys if key in lookup]
    if selected:
        return selected
    return outcomes


def _variant_lines(
    *,
    job_id: str,
    selected_job: dict[str, Any],
    selected_keys: list[str],
    treatment_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    selected_lookup = {str(item.get("key", "")): item for item in selected_job.get("outcomes", [])}
    for key in selected_keys:
        selected = selected_lookup.get(key)
        if not selected:
            continue
        lines.append(
            {
                "id": f"{job_id}-{key}-selected",
                "label": "Selected public estimate",
                "points": list(selected.get("points", [])),
            }
        )
    variant_ids = []
    for row in treatment_rows:
        variant_id = str(row.get("treatment_variant", "")).strip()
        if variant_id and variant_id in PUBLIC_TREATMENT_VARIANTS and variant_id not in variant_ids:
            variant_ids.append(variant_id)
    for variant_id in variant_ids:
        for key in selected_keys:
            points = []
            matching_rows = [
                row
                for row in treatment_rows
                if str(row.get("treatment_variant", "")).strip() == variant_id
                and str(row.get("outcome", "")).strip() == key
            ]
            for row in sorted(matching_rows, key=lambda item: int(item["horizon"])):
                beta = _coerce_float(row.get("beta"))
                lower = _coerce_float(row.get("lower95"))
                upper = _coerce_float(row.get("upper95"))
                p_value = _coerce_float(row.get("p_value_normal"))
                if beta is None or lower is None or upper is None:
                    continue
                points.append(
                    {
                        "horizon": int(row["horizon"]),
                        "beta": beta,
                        "lower95": lower,
                        "upper95": upper,
                        "p_value": p_value if p_value is not None else 1.0,
                    }
                )
            if points:
                lines.append(
                    {
                        "id": f"{job_id}-{key}-{variant_id}",
                        "label": TREATMENT_LABELS.get(variant_id, variant_id.replace("_", " ")),
                        "outcome_key": key,
                        "points": points,
                    }
                )
    return lines


def _treatment_comparison_payload(
    *,
    paths: ProjectPaths,
    jobs_by_id: dict[str, Any],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for job_id, meta in TREATMENT_COMPARISON_JOBS.items():
        selected_job = jobs_by_id.get(job_id)
        if not selected_job:
            continue
        report_path = paths.reports / f"{job_id}__treatment_sensitivity_estimates.csv"
        if not report_path.exists():
            continue
        treatment_rows = _read_csv(report_path)
        if not treatment_rows:
            continue
        outcomes = []
        for outcome_key in meta["outcomes"]:
            chart_dom_id = f"treatment-{job_id.replace('_', '-')}-{outcome_key.replace('_', '-')}"
            lines = [
                line
                for line in _variant_lines(
                    job_id=job_id,
                    selected_job=selected_job,
                    selected_keys=[outcome_key],
                    treatment_rows=treatment_rows,
                )
                if line.get("points")
            ]
            if not lines:
                continue
            outcomes.append(
                {
                    "key": outcome_key,
                    "label": _outcome_label(outcome_key),
                    "chart_dom_id": chart_dom_id,
                    "lines": lines,
                }
            )
        if not outcomes:
            continue
        payload.append(
            {
                "job_id": job_id,
                "title": meta["title"],
                "subtitle": meta["subtitle"],
                "summary": meta["summary"],
                "links": [
                    *(
                        [{"label": "Treatment variants CSV", "href": _copied_report_href(str(report_path), paths)}]
                        if _copied_report_href(str(report_path), paths)
                        else []
                    ),
                    *list(selected_job.get("links", [])),
                ],
                "outcomes": outcomes,
            }
        )
    return payload


def _outcome_payload_map(rows: list[dict[str, str]], job_id: str) -> dict[str, dict[str, Any]]:
    return {item["key"]: item for item in _job_outcomes(rows, job_id)}


def _estimate_beta(rows: list[dict[str, str]], outcome: str, horizon: int) -> float | None:
    for row in rows:
        if str(row.get("outcome", "")).strip() != outcome:
            continue
        if int(row.get("horizon", 0) or 0) != horizon:
            continue
        return _coerce_float(row.get("beta"))
    return None


def _independent_nontdc_payload(paths: ProjectPaths) -> dict[str, Any] | None:
    model_path = paths.output / "models" / f"{INDEPENDENT_NON_TDC_JOB_ID}__lp_estimates.csv"
    if not model_path.exists():
        return None
    rows = _read_csv(model_path)
    if not rows:
        return None
    outcome_map = _outcome_payload_map(rows, INDEPENDENT_NON_TDC_JOB_ID)
    outcomes = [outcome_map[key] for key in INDEPENDENT_NON_TDC_OUTCOME_KEYS if key in outcome_map]
    if not outcomes:
        return None
    residual_h0 = _estimate_beta(rows, "tdcpass_other_component_qoq", 0)
    core_h0 = _estimate_beta(rows, "tdcpass_strict_loan_core_min_qoq", 0)
    securities_h0 = _estimate_beta(rows, "tdcpass_strict_non_treasury_securities_qoq", 0)
    total_h0 = _estimate_beta(rows, "tdcpass_strict_identifiable_total_qoq", 0)
    gap_h0 = _estimate_beta(rows, "tdcpass_strict_identifiable_gap_qoq", 0)
    parts: list[str] = []
    if residual_h0 is not None:
        parts.append(f"the residual comparison surface moves by {residual_h0:.2f}")
    if core_h0 is not None:
        parts.append(f"the strict loan-core minimum moves by {core_h0:.2f}")
    if securities_h0 is not None:
        parts.append(f"the strict non-Treasury securities add-on moves by {securities_h0:.2f}")
    if total_h0 is not None:
        parts.append(f"the strict identifiable total moves by {total_h0:.2f}")
    if gap_h0 is not None:
        parts.append(f"the remaining residual gap is {gap_h0:.2f}")
    impact_summary = "On impact, " + ", ".join(parts) + "." if parts else ""
    links = [{"label": "Strict-source comparison CSV", "href": _copied_model_href(model_path)}]
    return {
        "title": "Strict independent non-TDC evidence is narrower and source-side",
        "subtitle": "EA-TDC no longer treats residual closure as an independent measurement lane; the valid comparison is the narrower `tdcpass` strict source-side block.",
        "summary": (
            "This block imports the stricter `tdcpass` comparison surface built from direct non-Treasury bank-asset transactions. "
            "It is intentionally narrower than the full non-TDC residual, so it should be read as independent source-side support rather than as a closure-by-construction total."
        ),
        "impact_summary": impact_summary,
        "note_lines": [
            "Independent evidence here means direct source-side non-Treasury bank-asset support, not residual closure.",
            "The strict loan-core minimum is the narrow direct core; the strict identifiable total adds a separate direct securities leg.",
            "The remaining gap stays nonzero by design because the strict lane does not try to close the whole non-TDC residual.",
        ],
        "outcomes": outcomes,
        "links": links,
    }


def _factor_summary_payload(robustness_snapshot: dict[str, Any], paths: ProjectPaths) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    rows = {str(item.get("job_id", "")).strip(): item for item in robustness_snapshot.get("rows", [])}
    for job_id in FACTOR_SUMMARY_JOB_IDS:
        row = rows.get(job_id)
        if not row:
            continue
        top_loadings = []
        for item in list(row.get("top_loadings", []))[:6]:
            feature_id = str(item.get("feature_id", "")).strip()
            if not feature_id:
                continue
            factor_id = str(item.get("factor_id", "")).strip()
            loading_value = _coerce_float(item.get("loading_abs"))
            if loading_value is None:
                loading_value = _coerce_float(item.get("loading"))
            top_loadings.append(
                {
                    "factor_id": factor_id,
                    "factor_label": _factor_label(factor_id),
                    "feature_id": feature_id,
                    "feature_label": _feature_label(feature_id),
                    "loading": loading_value,
                }
            )
        payload.append(
            {
                "job_id": job_id,
                "title": _job_meta(job_id)["title"],
                "summary": ROBUSTNESS_META.get(job_id, {}).get("summary", ""),
                "top_loadings": top_loadings,
                "links": [
                    {
                        "label": "Top loadings CSV",
                        "href": _copied_report_href(str((row.get("links", {}) or {}).get("factor_loadings_path", "")), paths),
                    },
                    {
                        "label": "Factor metadata",
                        "href": _copied_report_href(str((row.get("links", {}) or {}).get("factor_meta_path", "")), paths),
                    },
                    {
                        "label": "Control screen",
                        "href": _copied_report_href(str((row.get("links", {}) or {}).get("control_screen_path", "")), paths),
                    },
                ],
            }
        )
        payload[-1]["links"] = [link for link in payload[-1]["links"] if link["href"]]
    return payload


def _iv_lab_payload(paths: ProjectPaths) -> dict[str, Any] | None:
    summary_path = paths.reports / "iv_lab.json"
    if not summary_path.exists():
        return None
    summary = _read_json(summary_path)
    jobs = []
    for job in summary.get("jobs", []):
        current_candidate = job.get("current_candidate") or {}
        top_candidates = list(job.get("top_candidates", []))
        best_candidate = top_candidates[0] if top_candidates else {}
        jobs.append(
            {
                "job_id": str(job.get("job_id", "")).strip(),
                "title": _job_meta(str(job.get("job_id", "")).strip())["title"],
                "current_candidate_id": str(current_candidate.get("candidate_id", "")).strip(),
                "current_median_f": _coerce_float(current_candidate.get("median_first_stage_f")),
                "current_weak_share": _coerce_float(current_candidate.get("weak_row_share")),
                "current_recommendation": str(current_candidate.get("recommendation", "")).strip(),
                "best_candidate_id": str(best_candidate.get("candidate_id", "")).strip(),
                "best_median_f": _coerce_float(best_candidate.get("median_first_stage_f")),
                "best_weak_share": _coerce_float(best_candidate.get("weak_row_share")),
                "best_recommendation": str(best_candidate.get("recommendation", "")).strip(),
            }
        )
    return {
        "jobs_scanned": int(summary.get("jobs_scanned", 0) or 0),
        "total_candidates": int(summary.get("total_candidates", 0) or 0),
        "jobs": jobs,
        "links": [
            {"label": "IV lab JSON", "href": _copied_report_href(str(summary_path), paths)},
            {"label": "IV lab CSV", "href": _copied_report_href(str(paths.reports / 'iv_lab.csv'), paths)},
        ],
    }


def _table_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rendered = []
    for row in rows:
        rendered.append(
            {
                "outcome": row["outcome"],
                "outcome_label": _outcome_label(row["outcome"]),
                "horizon": row["horizon"],
                "beta": _fmt(row.get("beta")),
                "se": _fmt(row.get("se")),
                "lower95": _fmt(row.get("lower95")),
                "upper95": _fmt(row.get("upper95")),
                "p_value_normal": _fmt(row.get("p_value_normal")),
                "n": str(row.get("n", "")),
            }
        )
    return rendered


def _observation_label(row: dict[str, Any]) -> str:
    minimum = str(row.get("min_observations", "")).strip()
    maximum = str(row.get("max_observations", "")).strip()
    final = str(row.get("final_sample_observations", "")).strip()
    if minimum and maximum:
      return f"{minimum}-{maximum} obs"
    if maximum:
      return f"{maximum} obs"
    if final:
      return f"{final} obs"
    return "Sample varies by horizon"


def _estimator_label(row: dict[str, Any]) -> str:
    estimator = str(row.get("estimator", "")).replace("_", " ").strip()
    covariance = str(row.get("covariance_estimators_used", "")).replace("_", " ").strip()
    if estimator and covariance:
        return f"{estimator} | {covariance}"
    return estimator or covariance or "estimated"


def _treatment_label(row: dict[str, Any], rows: list[dict[str, str]]) -> str:
    treatment = str(row.get("treatment_id", "")).strip()
    if treatment:
        return TREATMENT_LABELS.get(treatment, treatment.replace("_", " "))
    if rows:
        first_treatment = str(rows[0].get("treatment_id", "")).strip()
        return TREATMENT_LABELS.get(first_treatment, first_treatment.replace("_", " "))
    return "treatment branch"


def _artifact_links(row: dict[str, Any], paths: ProjectPaths) -> list[dict[str, str]]:
    artifact_id = str(row.get("artifact_id", ""))
    links = [{"label": "Open", "href": f"artifacts/{artifact_id}/index.html"}]
    if row.get("secondary_path"):
        links.append({"label": "CSV", "href": _copied_artifact_href(Path(row["secondary_path"]), paths)})
    if row.get("primary_path"):
        links.append({"label": "Raw", "href": _copied_artifact_href(Path(row["primary_path"]), paths)})
    return links


def _robustness_payload(
    robustness_snapshot: dict[str, Any],
    paths: ProjectPaths,
) -> dict[str, Any]:
    control_universe = robustness_snapshot.get("control_universe", {}) or {}
    jobs = []
    recommended_k_values: list[int] = []
    for item in robustness_snapshot.get("rows", []):
        job_id = str(item.get("job_id", "")).strip()
        if job_id not in ROBUSTNESS_JOB_IDS:
            continue
        meta = ROBUSTNESS_META.get(job_id, {})
        ladder = [
            {
                "label": str(row.get("label", "")).strip(),
                "avg_abs_beta": float(row.get("avg_abs_beta", 0.0) or 0.0),
                "rows_written": int(row.get("rows_written", 0) or 0),
            }
            for row in item.get("ladder_rows", [])
            if str(row.get("label", "")).strip()
        ]
        recommended_k = int(item.get("recommended_k", 0) or 0)
        if recommended_k > 0:
            recommended_k_values.append(recommended_k)
        recommended_index = 0
        for idx, row in enumerate(item.get("ladder_rows", [])):
            if int(row.get("k_screened", 0) or 0) == recommended_k:
                recommended_index = idx
                break
        jobs.append(
            {
                "job_id": job_id,
                "title": meta.get("title", _job_meta(job_id)["title"]),
                "summary": meta.get("summary", _job_meta(job_id)["summary"]),
                "interpretation": (
                    f"The selected public version screens K={recommended_k} lagged indicators from the larger control set. "
                    f"{str(item.get('recommended_k_reason', '')).strip()}"
                ),
                "ml_public_branch": str(item.get("ml_public_branch", "")).strip(),
                "ml_public_branch_label": str(item.get("ml_public_branch_label", "")).strip(),
                "ml_public_branch_reason": str(item.get("ml_public_branch_reason", "")).strip(),
                "recommended_k": recommended_k,
                "recommended_factor_count": int(item.get("recommended_factor_count", 0) or 0),
                "screened_feature_count": int(item.get("screened_feature_count", 0) or 0),
                "treatment_variant_count": int(item.get("treatment_variant_count", 0) or 0),
                "regime_filter_count": int(item.get("regime_filter_count", 0) or 0),
                "dml_rows_written": int(((item.get("dml", {}) or {}).get("rows_written", 0)) or 0),
                "tmle_rows_written": int(((item.get("tmle", {}) or {}).get("rows_written", 0)) or 0),
                "forest_rows_written": int(((item.get("forest", {}) or {}).get("rows_written", 0)) or 0),
                "negative_control_max_lead_placebo_abs_z": float(((item.get("negative_controls", {}) or {}).get("max_lead_placebo_abs_z", 0.0)) or 0.0),
                "negative_control_signal": str(((item.get("negative_controls", {}) or {}).get("signal", "")) or ""),
                "negative_control_signal_label": str(((item.get("negative_controls", {}) or {}).get("signal_label", "")) or ""),
                "chart_dom_id": f"robustness-{job_id.replace('_', '-')}",
                "recommended_index": recommended_index,
                "ladder": ladder,
                "links": [
                    {"label": "Ladder CSV", "href": _copied_report_href(str((item.get("links", {}) or {}).get("ladder_path", "")), paths)},
                    {"label": "Treatment CSV", "href": _copied_report_href(str((item.get("links", {}) or {}).get("treatment_path", "")), paths)},
                    {"label": "Regime CSV", "href": _copied_report_href(str((item.get("links", {}) or {}).get("regime_path", "")), paths)},
                    {"label": "Quoted DML summary", "href": _copied_report_href(str((item.get("links", {}) or {}).get("dml_summary_path", "")), paths)},
                    {"label": "Quoted DML estimates", "href": _copied_report_href(str((item.get("links", {}) or {}).get("dml_estimates_path", "")), paths)},
                    {"label": "Forest cross-check", "href": _copied_report_href(str((item.get("links", {}) or {}).get("forest_summary_path", "")), paths)},
                    {"label": "Forest estimates", "href": _copied_report_href(str((item.get("links", {}) or {}).get("forest_estimates_path", "")), paths)},
                ],
            }
        )
        jobs[-1]["links"] = [link for link in jobs[-1]["links"] if link["href"]]
    jobs.sort(key=lambda item: ROBUSTNESS_JOB_IDS.index(item["job_id"]))
    recommended_k_mode = 0
    if recommended_k_values:
        recommended_k_mode = max(sorted(set(recommended_k_values)), key=recommended_k_values.count)
    overview_links = [
        {"label": "Universe JSON", "href": "site_assets/reports/robustness_snapshot.json"},
        {"label": "Universe columns", "href": _copied_report_href(str(control_universe.get("columns_path", "")), paths)},
        {"label": "Panel CSV", "href": _copied_report_href(str(control_universe.get("panel_path", "")), paths)},
    ]
    overview_links = [link for link in overview_links if link["href"]]
    return {
        "overview": {
            "title": "Larger control set",
            "summary": "These checks rerun the quarterly results with a much broader set of lagged indicators, then compare that version with the baseline controls and the main treatment variants.",
            "series_count": int(control_universe.get("series_count", 0) or 0),
            "feature_count": int(control_universe.get("feature_count", 0) or 0),
            "daily_lags": int(((control_universe.get("lag_structure", {}) or {}).get("daily_lags", 0)) or 0),
            "recommended_k_mode": recommended_k_mode,
            "links": overview_links,
        },
        "jobs": jobs,
    }


def _artifact_payload(row: dict[str, Any], paths: ProjectPaths) -> dict[str, Any]:
    outcome_ids = [item.strip() for item in str(row.get("outcome_ids", "")).split(",") if item.strip()]
    horizons = [item.strip() for item in str(row.get("horizons", "")).split(",") if item.strip()]
    title = str(row.get("title", "")).replace("Main impulse response:", "").replace("Main coefficient table:", "").strip()
    if title:
        title = title[0].upper() + title[1:]
    if not title:
        title = str(row.get("title", ""))
    table_rows: list[dict[str, str]] = []
    secondary_path = str(row.get("secondary_path", "")).strip()
    if str(row.get("artifact_kind", "")) == "table" and secondary_path:
        csv_path = _resolve_repo_path(secondary_path, paths)
        if csv_path.exists():
            table_rows = _table_rows(_read_csv(csv_path))
    return {
        "artifact_id": str(row.get("artifact_id", "")),
        "kind": str(row.get("artifact_kind", "")),
        "job_id": str(row.get("job_id", "")),
        "slot_label": str(row.get("slot_label", "")),
        "title": title,
        "subtitle": str(row.get("subtitle", "")),
        "caption": str(row.get("caption", "")),
        "outcome_ids": outcome_ids,
        "horizons": horizons,
        "table_rows": table_rows,
        "links": _artifact_links(row, paths),
        "release_channel": str(row.get("release_channel", "")),
    }


def _site_data_payload(
    *,
    paths: ProjectPaths,
    scorecard: dict[str, Any],
    contract: dict[str, Any],
    artifact_build: dict[str, Any],
    robustness_snapshot: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    scorecard_rows = {str(row.get("job_id", "")): row for row in scorecard.get("rows", [])}
    jobs_by_id: dict[str, Any] = {}
    candidate_job_ids = set(MAIN_JOB_IDS + SIDECAR_JOB_IDS)
    candidate_job_ids.update(str(row.get("job_id", "")) for row in artifact_build.get("rows", []))
    for job_id in sorted(candidate_job_ids):
        baseline_model_path = _find_estimates_path(job_id, scorecard_rows, paths)
        branch = _selected_result_branch(paths, job_id, baseline_model_path)
        model_path = Path(branch["selected_path"])
        if not model_path.exists():
            continue
        rows = _read_csv(model_path)
        if not rows:
            continue
        meta = _job_meta(job_id)
        scorecard_row = scorecard_rows.get(job_id, {})
        robustness_summary = branch.get("robustness_summary", {}) or {}
        links = [{"label": "Selected CSV", "href": _copied_model_href(model_path)}]
        if Path(branch["baseline_path"]) != model_path and Path(branch["baseline_path"]).exists():
            links.append({"label": "Baseline CSV", "href": _copied_model_href(Path(branch["baseline_path"]))})
        ladder_href = _copied_report_href(str(robustness_summary.get("ladder_path", "")), paths)
        if ladder_href:
            links.append({"label": "Control ladder", "href": ladder_href})
        treatment_href = _copied_report_href(str(robustness_summary.get("treatment_path", "")), paths)
        if treatment_href:
            links.append({"label": "Treatment variants", "href": treatment_href})
        jobs_by_id[job_id] = {
            "job_id": job_id,
            "anchor": job_id.replace("_", "-"),
            "kicker": meta["kicker"],
            "title": meta["title"],
            "subtitle": meta["subtitle"],
            "summary": meta["summary"],
            "estimator_label": _estimator_label(scorecard_row),
            "observation_label": _observation_label(scorecard_row),
            "treatment_label": _treatment_label(scorecard_row, rows),
            "branch_label": str(branch.get("branch_label", "")).strip(),
            "branch_note": str(branch.get("branch_summary", "")).strip(),
            "links": links,
            "outcomes": _job_outcomes(rows, job_id),
            "table_rows": _table_rows(rows),
        }

    contract_rows = contract.get("rows", [])
    deferred_rows = [
        row for row in contract_rows if str(row.get("contract_tier", "")).strip() == "deferred_development"
    ]
    contract_rows_by_artifact = {
        str(row.get("artifact_id", "")): row
        for row in contract_rows
        if str(row.get("artifact_id", "")).strip()
    }
    artifact_rows = [
        _artifact_payload({**contract_rows_by_artifact.get(str(row.get("artifact_id", "")), {}), **row}, paths)
        for row in artifact_build.get("rows", [])
    ]
    artifact_rows = [row for row in artifact_rows if row.get("job_id") not in HIDDEN_SITE_JOB_IDS]
    artifact_lookup = {row["artifact_id"]: row for row in artifact_rows}
    main_artifacts = [row for row in artifact_rows if row["release_channel"] == "main_text"]
    appendix_artifacts = [row for row in artifact_rows if row["release_channel"] == "appendix"]
    independent_evidence_payload = _independent_nontdc_payload(paths)
    treatment_comparisons = _treatment_comparison_payload(paths=paths, jobs_by_id=jobs_by_id)
    iv_lab = _iv_lab_payload(paths)
    return {
        "generated_at": generated_at,
        "metrics": [
            {
                "label": "Baseline jobs",
                "value": str(scorecard.get("committed_public_jobs", 0)),
                "note": "Core public result blocks in the current release.",
            },
            {
                "label": "Appendix jobs",
                "value": str(
                    sum(
                        1
                        for row in contract_rows
                        if str(row.get("contract_tier", "")).strip() == "release1_appendix_candidate"
                    )
                ),
                "note": "Supporting public jobs that stay below the headline claim.",
            },
            {
                "label": "Estimated jobs",
                "value": str(scorecard.get("estimated_jobs", 0)),
                "note": "Tracked jobs with nonempty estimate outputs.",
            },
        ],
        "jobs": jobs_by_id,
        "home": {
            "insights": INSIGHTS_HOME,
            "main_job_ids": [job_id for job_id in MAIN_JOB_IDS if job_id in jobs_by_id],
            "main_artifacts": main_artifacts,
            "appendix_artifacts": appendix_artifacts,
            "independent_evidence": independent_evidence_payload,
        },
        "sidecar": {
            "insights": INSIGHTS_SIDECAR,
            "job_ids": [job_id for job_id in SIDECAR_JOB_IDS if job_id in jobs_by_id],
            "treatment_comparisons": treatment_comparisons,
            "iv_lab": iv_lab,
        },
        "deferred_jobs": [
            {
                "tag": "Needs more evidence",
                "title": _job_meta(str(row.get("job_id", "")))["title"],
                "reason": str(row.get("contract_reason", "")).replace("_", " "),
                "meta": [
                    item
                    for item in [
                        str(row.get("estimator", "")).replace("_", " ").strip(),
                        f"{row.get('final_sample_observations', '')} obs" if str(row.get("final_sample_observations", "")).strip() else "",
                        str(row.get("sample_policy", "")).replace("_", " ").strip(),
                    ]
                    if item
                ],
            }
            for row in deferred_rows
        ],
        "robustness": _robustness_payload(robustness_snapshot, paths),
        "artifacts": artifact_lookup,
        "artifact_gallery": main_artifacts + appendix_artifacts,
    }


def _head(relative_assets: str, title: str, include_math: bool = False, asset_version: str = "") -> str:
    asset_version = quote(asset_version, safe="")
    asset_suffix = f"?v={asset_version}" if asset_version else ""
    math_script = '<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>' if include_math else ""
    favicon_href = (
        "data:image/svg+xml,"
        "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E"
        "%3Crect width='64' height='64' rx='14' fill='%230d1117'/%3E"
        "%3Ctext x='50%25' y='54%25' dominant-baseline='middle' text-anchor='middle' "
        "font-family='Arial' font-size='28' fill='white'%3EEA%3C/text%3E%3C/svg%3E"
    )
    return "\n".join(
        [
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            '<meta name="color-scheme" content="light dark">',
            '<meta name="theme-color" content="#fafbfd" media="(prefers-color-scheme: light)">',
            '<meta name="theme-color" content="#0d1117" media="(prefers-color-scheme: dark)">',
            f"<title>{html.escape(title)}</title>",
            f'<link rel="icon" href="{favicon_href}">',
            '<link rel="preconnect" href="https://fonts.googleapis.com">',
            '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
            '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">',
            f'<script src="{html.escape(relative_assets)}js/theme.js{asset_suffix}"></script>',
            f'<link rel="stylesheet" href="{html.escape(relative_assets)}css/style.css{asset_suffix}">',
            '<script defer src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>',
            math_script,
            f'<script defer src="{html.escape(relative_assets)}js/main.js{asset_suffix}"></script>',
            "</head>",
        ]
    )


def _nav(page: str) -> str:
    if page == "home":
        links = [
            ("#questions", "Questions"),
            ("#definitions", "TDC definition"),
            ("#headline-results", "Results"),
            ("#independent-evidence-section", "Independent evidence"),
            ("#additional-evidence", "Additional evidence"),
            ("#component-evidence", "Component evidence"),
            ("#robustness", "Robustness"),
            ("#release-package", "Artifacts"),
        ]
    elif page == "sidecar":
        links = [
            ("../index.html#questions", "Questions"),
            ("../index.html#definitions", "TDC definition"),
            ("../index.html#headline-results", "Results"),
            ("../index.html#independent-evidence-section", "Independent evidence"),
            ("../index.html#additional-evidence", "Additional evidence"),
            ("../index.html#component-evidence", "Component evidence"),
            ("../index.html#robustness", "Robustness"),
            ("../index.html#release-package", "Artifacts"),
        ]
    elif page == "gallery":
        links = [
            ("../index.html#questions", "Questions"),
            ("../index.html#definitions", "TDC definition"),
            ("../index.html#headline-results", "Results"),
            ("../index.html#independent-evidence-section", "Independent evidence"),
            ("../index.html#additional-evidence", "Additional evidence"),
            ("../index.html#component-evidence", "Component evidence"),
            ("#top", "Artifacts"),
        ]
    else:
        links = [
            ("../../index.html#questions", "Questions"),
            ("../../index.html#definitions", "TDC definition"),
            ("../../index.html#headline-results", "Results"),
            ("../../index.html#independent-evidence-section", "Independent evidence"),
            ("../../index.html#additional-evidence", "Additional evidence"),
            ("../../index.html#component-evidence", "Component evidence"),
            ("../index.html", "Artifact gallery"),
        ]
    return "\n".join(
        [
            '<header class="nav">',
            '<div class="container nav-inner">',
            '<div class="brand">',
            '<span class="brand-mark">EA-TDC</span>',
            '<span class="brand-copy">Definitions, estimates, and evidence</span>',
            "</div>",
            '<nav class="nav-links" aria-label="Primary">',
            *[f'<a href="{html.escape(href)}">{html.escape(label)}</a>' for href, label in links],
            '<button id="theme-toggle" class="theme-toggle" type="button" aria-label="Toggle theme"></button>',
            "</nav>",
            "</div>",
            "</header>",
        ]
    )


def _equation_markup() -> str:
    cards = []
    for item in TDC_EQUATIONS:
        definitions = item.get("definitions", [])
        definitions_markup = ""
        if definitions:
            definition_rows = []
            for symbol, meaning in definitions:
                definition_rows.append(
                    "\n".join(
                        [
                            '<div class="definition-row">',
                            f'<dt><code class="symbol-code" title="{html.escape(meaning)}">{html.escape(_symbol_display(symbol))}</code></dt>',
                            f'<dd>{html.escape(meaning)}</dd>',
                            "</div>",
                        ]
                    )
                )
            definitions_markup = "\n".join(
                [
                    '<dl class="equation-definitions">',
                    *definition_rows,
                    "</dl>",
                ]
            )
        cards.append(
            "\n".join(
                [
                    '<article class="equation-card">',
                    f'<div class="eyebrow">{html.escape(item["kicker"])}</div>',
                    f'<h3>{html.escape(item["title"])}</h3>',
                    f'<p>{html.escape(item["body"])}</p>',
                    f'<div class="math">\\[{item["latex"]}\\]</div>',
                    definitions_markup,
                    "</article>",
                ]
            )
        )
    notation_cards = []
    for term, meaning in ABBREVIATION_GLOSSARY:
        notation_cards.append(
            "\n".join(
                [
                    '<article class="notation-chip">',
                    f'<strong><abbr title="{html.escape(meaning)}">{html.escape(term)}</abbr></strong>',
                    f'<p>{html.escape(meaning)}</p>',
                    "</article>",
                ]
            )
        )
    return "\n".join(
        [
            '<details class="definition-shell reveal" id="definitions">',
            '<summary><div><div class="eyebrow">TDC definition</div><h2>Equations and notation.</h2></div><p>View equations</p></summary>',
            '<div class="definition-body">',
            '<p>The baseline treatment is the marketable-Treasury, transaction-based quarterly approximation to Treasury Deposit Contribution. In the upstream estimator its short name includes "bank" because it excludes credit unions, not because it excludes the rest of world. The implemented baseline formula below still includes Federal Reserve, bank-sector, and rest-of-world Treasury transactions.</p>',
            '<div class="notation-grid">',
            *notation_cards,
            "</div>",
            '<div class="equation-grid">',
            *cards,
            "</div>",
            '<p>TDC means Treasury Deposit Contribution: the contribution of Treasury-related cash flows and Treasury-security transactions to changes in domestic nonbank deposits. The quarterly headline on this site is an implemented approximation to that theory object, not a full direct DU ledger.</p>',
            "</div>",
            "</details>",
        ]
    )


def _home_html(generated_at: str) -> str:
    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en" data-theme="light">',
            _head("assets/", "EA-TDC", include_math=True, asset_version=generated_at),
            f'<body data-page="home" data-root-prefix="" data-site-version="{html.escape(generated_at)}">',
            _nav("home"),
            '<main class="page">',
            '<section class="hero container">',
            '<div class="hero-copy reveal">',
            '<div class="eyebrow">EA-TDC</div>',
            '<h1>Treasury contribution to deposits: estimates and transmission.</h1>',
            '<p>This work asks whether the baseline estimate of Treasury Deposit Contribution produces an early deposit response, how that result survives broader holder definitions, and what remains under a narrower independent source-side comparison imported from `tdcpass`.</p>',
            '<div class="button-row">',
            '<a class="button primary" href="#headline-results">Read the results</a>',
            '<a class="button" href="#definitions">See the TDC definition</a>',
            '<a class="button" href="artifacts/index.html">Open figures and tables</a>',
            "</div>",
            "</div>",
            '<div id="metric-grid" class="metric-grid"></div>',
            "</section>",
            '<section class="section container" id="claim-hierarchy">',
            '<div class="section-header reveal">',
            '<div><div class="eyebrow">Claim hierarchy</div><h2>What the release claims.</h2></div>',
            '<p>The public package is intentionally narrow: baseline deposits are the headline, strict independent non-TDC evidence comes from the narrower `tdcpass` source-side lane, component regressions are explanatory sidecars, and corrected Tier 2 / Tier 3 treatments stay sensitivity-only.</p>',
            "</div>",
            '<div class="insight-grid"><article class="insight-card reveal"><div class="eyebrow">Headline</div><h3>Baseline deposit response.</h3><p>The selected quarterly baseline delivers the main public result.</p></article><article class="insight-card reveal"><div class="eyebrow">Boundary</div><h3>Strict source-side comparison.</h3><p>The valid independent non-TDC comparison is the narrower `tdcpass` source-side lane, not residual closure.</p></article><article class="insight-card reveal"><div class="eyebrow">Explanation</div><h3>Component sidecars.</h3><p>RU acquisition, Treasury cash, and remittances help interpret the deposit result.</p></article><article class="insight-card reveal"><div class="eyebrow">Sensitivity</div><h3>Corrected variants.</h3><p>Tier 2 and Tier 3 treatments remain public sensitivity branches only.</p></article></div>',
            "</section>",
            '<section class="section container" id="questions">',
            '<div class="section-header reveal">',
            '<div><div class="eyebrow">Questions</div><h2>Research questions.</h2></div>',
            '<p>The central questions are whether Treasury Deposit Contribution produces an early deposit response, whether that result survives broader holder definitions, and what remains under the narrower strict independent non-TDC evidence surface imported from `tdcpass`.</p>',
            "</div>",
            '<div id="questions-grid" class="insight-grid"></div>',
            "</section>",
            '<section class="section container">',
            _equation_markup(),
            "</section>",
            '<section class="section container" id="headline-results">',
            '<div class="section-header reveal">',
            '<div><div class="eyebrow">Results</div><h2>Deposits are the clearest headline response.</h2></div>',
            '<p>This figure traces the main deposit response to the baseline TDC estimate using the selected public version of the quarterly model.</p>',
            "</div>",
            '<div id="headline-job-list" class="job-list"></div>',
            "</section>",
            '<section class="section container" id="independent-evidence-section">',
            '<div class="section-header reveal">',
            '<div><div class="eyebrow">Independent evidence</div><h2>Strict independent non-TDC evidence is narrower and source-side.</h2></div>',
            '<p>EA-TDC does not use residual/accounting closure as an independent non-TDC measure. The relevant comparison here is the narrower `tdcpass` source-side lane built from direct non-Treasury bank-asset transactions.</p>',
            "</div>",
            '<div id="independent-evidence" class="job-list"></div>',
            "</section>",
            '<section class="section container" id="additional-evidence">',
            '<div class="section-header reveal">',
            '<div><div class="eyebrow">Additional evidence</div><h2>Inflation, FX, and private balance sheets.</h2></div>',
            '<p>These branches extend the same baseline TDC estimate into prices, exchange rates, and private non-Treasury balance sheets without changing the main public claim hierarchy.</p>',
            "</div>",
            '<div id="additional-evidence-grid" class="insight-grid"></div>',
            '<div style="height:18px"></div>',
            '<div id="job-list-sidecar" class="job-list"></div>',
            "</section>",
            '<section class="section container" id="component-evidence">',
            '<div class="section-header reveal">',
            '<div><div class="eyebrow">Component evidence</div><h2>Which Treasury leg is actually moving deposits and liquidity?</h2></div>',
            '<p>This summary collects the live component regressions for RU acquisition, Treasury operating cash, and positive remittances, along with the narrow state-probe results that still look worth watching.</p>',
            "</div>",
            '<div class="section-block reveal">',
            '<div class="button-row"><a class="button primary" href="site_assets/reports/component_sidecar_screening.md">Open component summary</a><a class="button" href="site_assets/reports/final_interpretation_closeout.md">Final interpretation</a><a class="button" href="site_assets/reports/component_sidecar_screening.csv" download>Download component CSV</a><a class="button" href="#robustness">Open robustness</a></div>',
            '<p class="section-note">Current read: RU acquisition is the strongest long-sample component, Treasury cash is mainly a liquidity-accounting channel, and remittances matter for deposits plus Fed-assets-relative liquidity. The only retained state probe with a clear interaction signal is Treasury cash during ON RRP drain.</p>',
            "</div>",
            "</section>",
            '<section class="section container" id="treatment-variants">',
            '<div class="section-header reveal">',
            '<div><div class="eyebrow">Treatment variants</div><h2>Does the result survive a broader deposit-holder definition?</h2></div>',
            '<p>This comparison asks whether the public deposit result looks similar when the construction broadens from the baseline bank-focused perimeter to a broader depository and credit-union-inclusive perimeter.</p>',
            "</div>",
            '<div id="treatment-comparisons" class="job-list"></div>',
            "</section>",
            '<section class="section container" id="robustness">',
            '<div class="section-header reveal">',
            '<div><div class="eyebrow">Robustness</div><h2>Larger control sets and sensitivity checks.</h2></div>',
            '<p>This section summarizes how the quarterly results look under the larger control set, treatment variants, and other supporting checks.</p>',
            "</div>",
            '<div id="robustness-overview"></div>',
            '<div style="height:18px"></div>',
            '<div id="robustness-grid" class="robustness-grid"></div>',
            "</section>",
            '<section class="section container" id="release-package">',
            '<div class="section-header reveal">',
            '<div><div class="eyebrow">Figures and tables</div><h2>Interactive views and downloads.</h2></div>',
            '<p>The same results are available as figures, tables, and downloadable files.</p>',
            "</div>",
            '<div id="artifact-grid-main" class="artifact-grid"></div>',
            '<div style="height:18px"></div>',
            '<div id="artifact-grid-appendix" class="artifact-grid"></div>',
            "</section>",
            '<section class="section container" id="deferred-work">',
            '<div class="section-header reveal">',
            '<div><div class="eyebrow">Open questions</div><h2>Results that still need stronger identification.</h2></div>',
            '<p>These branches remain visible because the questions matter, but they are not used as headline claims while the weak-IV problem remains unresolved.</p>',
            "</div>",
            '<div id="deferred-grid" class="deferred-grid"></div>',
            "</section>",
            '<footer class="site-footer container">',
            '<div class="footer-grid reveal">',
            '<div><div class="eyebrow">About EA-TDC</div><h2>Treasury contribution to deposits: definitions, estimates, and evidence.</h2><p>Definitions, estimates, figures, and downloadable result files are collected here together.</p></div>',
            f'<div><div class="eyebrow">Downloads</div><div class="button-row"><a class="link-chip" href="site_assets/reports/release_scorecard.json" download>Scorecard JSON</a><a class="link-chip" href="site_assets/reports/release_contract.json" download>Contract JSON</a><a class="link-chip" href="#robustness">Robustness</a></div><p class="footer-note">Generated {html.escape(generated_at)}</p></div>',
            "</div>",
            "</footer>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _sidecar_html(generated_at: str) -> str:
    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en" data-theme="light">',
            _head("../assets/", "EA-TDC Additional Evidence", asset_version=generated_at),
            f'<body data-page="sidecar" data-root-prefix="../" data-site-version="{html.escape(generated_at)}">',
            _nav("sidecar"),
            '<main class="page">',
            '<section class="hero container">',
            '<div class="hero-copy reveal">',
            '<div class="eyebrow">Additional evidence</div>',
            '<h1>This material now lives on the main page.</h1>',
            '<p>The additional evidence and robustness checks are now integrated into the main site so the project reads as one continuous research product.</p>',
            '<div class="button-row">',
            '<a class="button primary" href="../index.html#additional-evidence">Open additional evidence</a>',
            '<a class="button" href="../index.html#component-evidence">Open component evidence</a>',
            '<a class="button" href="../index.html#robustness">Open robustness</a>',
            '<a class="button" href="../artifacts/index.html">Figures and tables</a>',
            "</div>",
            "</div>",
            "</section>",
            '<footer class="site-footer container">',
            '<div class="footer-grid reveal">',
            '<div><div class="eyebrow">One-page layout</div><h2>The public site now uses a single continuous narrative.</h2><p>Use the main page sections for additional evidence and robustness rather than a separate secondary page.</p></div>',
            f'<div><div class="eyebrow">Downloads</div><div class="button-row"><a class="link-chip" href="../index.html#additional-evidence">Additional evidence</a><a class="link-chip" href="../index.html#component-evidence">Component evidence</a><a class="link-chip" href="../index.html#robustness">Robustness</a><a class="link-chip" href="../site_assets/reports/component_sidecar_screening.md">Component summary</a><a class="link-chip" href="../site_assets/reports/robustness_snapshot.json" download>Robustness JSON</a></div><p class="footer-note">Generated {html.escape(generated_at)}</p></div>',
            "</div>",
            "</footer>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _artifact_gallery_html(generated_at: str) -> str:
    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en" data-theme="light">',
            _head("../assets/", "EA-TDC Artifact Pages", asset_version=generated_at),
            f'<body data-page="gallery" data-root-prefix="../" data-site-version="{html.escape(generated_at)}">',
            _nav("gallery"),
            '<main class="page">',
            '<section class="hero container" id="top">',
            '<div class="hero-copy reveal">',
            '<div class="eyebrow">Figures and tables</div>',
            '<h1>All figures and tables in one place.</h1>',
            '<p>Open any result below for a focused view, or use the download links when you want the underlying files directly.</p>',
            '<div class="button-row"><a class="button primary" href="../index.html#release-package">Back to figures and tables</a><a class="button" href="../index.html#additional-evidence">Additional evidence</a></div>',
            "</div>",
            '<div id="metric-grid" class="metric-grid"></div>',
            "</section>",
            '<section class="section container">',
            '<div class="section-header reveal">',
            '<div><div class="eyebrow">Gallery</div><h2>Browse figures and tables.</h2></div>',
            '<p>Each entry opens to a focused result view with the same styling and navigation as the rest of the release.</p>',
            "</div>",
            '<div id="artifact-gallery-grid" class="gallery-grid"></div>',
            "</section>",
            '<footer class="site-footer container"><div class="footer-grid reveal"><div><div class="eyebrow">Gallery</div><h2>Figures, tables, and downloads.</h2><p>Figures, tables, and downloadable files are grouped together here.</p></div><div><p class="footer-note">Generated '
            + html.escape(generated_at)
            + "</p></div></div></footer>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _artifact_page_html(artifact: dict[str, Any], generated_at: str) -> str:
    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en" data-theme="light">',
            _head("../../assets/", f'EA-TDC • {artifact["slot_label"]}', asset_version=generated_at),
            f'<body data-page="artifact" data-artifact-id="{html.escape(artifact["artifact_id"])}" data-root-prefix="../../" data-site-version="{html.escape(generated_at)}">',
            _nav("artifact"),
            '<main class="page">',
            '<section class="hero container">',
            '<div class="hero-copy reveal" id="artifact-summary"></div>',
            '<div id="metric-grid" class="metric-grid"></div>',
            "</section>",
            '<section class="section container"><div id="artifact-body" class="artifact-layout"></div></section>',
            '<footer class="site-footer container"><div class="footer-grid reveal"><div><div class="eyebrow">Result detail</div><h2>Selected figure or table.</h2><p>This view shows the selected figure or table and links to the underlying files.</p></div><div><p class="footer-note">Generated '
            + html.escape(generated_at)
            + "</p></div></div></footer>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def build_site(paths: ProjectPaths) -> SiteBuildResult:
    artifact_result = build_release_artifacts(paths)
    robustness_result = build_robustness_snapshot(paths, job_ids=ROBUSTNESS_JOB_IDS)
    build_component_sidecar_screening(paths)
    build_stage_completion_closeout(paths)
    sanitize_output_paths(paths)
    generated_at = utc_now_iso()
    scorecard = _read_json(paths.reports / "release_scorecard.json")
    contract = _read_json(paths.reports / "release_contract.json")
    artifact_build = _read_json(artifact_result.summary_path)
    robustness_snapshot = _read_json(robustness_result.summary_path)

    site_root = paths.root / "docs"
    sidecar_root = site_root / "sidecar-results"
    assets_root = site_root / "assets"
    artifacts_root = site_root / "artifacts"
    site_assets_root = site_root / "site_assets"
    site_artifacts = site_assets_root / "artifacts"
    site_reports = site_assets_root / "reports"
    site_models = site_assets_root / "models"

    site_root.mkdir(parents=True, exist_ok=True)
    sidecar_root.mkdir(parents=True, exist_ok=True)
    if artifacts_root.exists():
        shutil.rmtree(artifacts_root)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    _write_text(site_root / ".nojekyll", "")

    site_data = _site_data_payload(
        paths=paths,
        scorecard=scorecard,
        contract=contract,
        artifact_build=artifact_build,
        robustness_snapshot=robustness_snapshot,
        generated_at=generated_at,
    )
    model_paths = _site_model_paths(site_data, paths)

    copied_artifacts = _copy_tree(paths.output / "artifacts", site_artifacts)
    _copy_tree(paths.reports, site_reports)
    copied_models = _copy_selected_files(model_paths, site_models)
    _prune_public_reports(site_reports)
    _sanitize_public_tree(site_artifacts, paths.root)
    _sanitize_public_tree(site_reports, paths.root)
    _sanitize_public_tree(site_models, paths.root)
    copied_reports = sum(1 for path in site_reports.rglob("*") if path.is_file())

    _write_text(assets_root / "css" / "style.css", CSS_TEXT)
    _write_text(assets_root / "js" / "theme.js", THEME_JS_TEXT)
    _write_text(assets_root / "js" / "main.js", JS_TEXT)
    write_json(assets_root / "data" / "site_data.json", site_data)

    index_path = site_root / "index.html"
    sidecar_index_path = sidecar_root / "index.html"
    _write_text(index_path, _home_html(generated_at))
    _write_text(sidecar_index_path, _sidecar_html(generated_at))
    _write_text(artifacts_root / "index.html", _artifact_gallery_html(generated_at))

    for artifact in site_data["artifact_gallery"]:
        artifact_dir = artifacts_root / artifact["artifact_id"]
        _write_text(artifact_dir / "index.html", _artifact_page_html(artifact, generated_at))

    summary = {
        "generated_at": generated_at,
        "index_path": index_path.relative_to(paths.root).as_posix(),
        "sidecar_index_path": sidecar_index_path.relative_to(paths.root).as_posix(),
        "artifact_gallery_path": (artifacts_root / "index.html").relative_to(paths.root).as_posix(),
        "site_data_path": (assets_root / "data" / "site_data.json").relative_to(paths.root).as_posix(),
        "css_path": (assets_root / "css" / "style.css").relative_to(paths.root).as_posix(),
        "theme_js_path": (assets_root / "js" / "theme.js").relative_to(paths.root).as_posix(),
        "js_path": (assets_root / "js" / "main.js").relative_to(paths.root).as_posix(),
        "copied_artifacts": copied_artifacts,
        "copied_reports": copied_reports,
        "copied_models": copied_models,
    }
    summary_path = paths.reports / "site_build.json"
    write_json(summary_path, summary)
    return SiteBuildResult(
        index_path=index_path,
        sidecar_index_path=sidecar_index_path,
        summary_path=summary_path,
        copied_artifacts=copied_artifacts,
        copied_reports=copied_reports,
        copied_models=copied_models,
    )
