from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml

from ea_tdc.paths import ProjectPaths
from ea_tdc.utils import utc_now_iso, write_json


BASELINE_SERIES_MAP = {
    "tdc_bank_only_qoq": ("tdcest", "tdc_base_bank_only_ru_flow"),
    "tdc_tier2_interest_corrected_bank_only_ru_flow": (
        "tdcest",
        "tdc_tier2_interest_corrected_bank_only_ru_flow",
    ),
    "tdc_tier2_legacy_h15_bank_only_qoq": (
        "tdcest",
        "tdc_tier2_h15_intensity_corrected_bank_only_ru_flow",
    ),
    "tdc_tier2_legacy_h15_bill_robust_bank_only_qoq": (
        "tdcest",
        "tdc_tier2_h15_treasury_interest_robust_bank_only_ru_flow",
    ),
    "tdc_tier2_regression_bank_only_qoq": (
        "tdcest",
        "tdc_tier2_regression_bank_only_ru_flow",
    ),
    "tdc_tier2_regression_mmf_rrp_lb_bank_only_qoq": (
        "tdcest",
        "tdc_tier2_regression_mmf_rrp_lb_bank_only_ru_flow",
    ),
    "tdc_tier2_regression_mmf_rrp_prop_bank_only_qoq": (
        "tdcest",
        "tdc_tier2_regression_mmf_rrp_prop_bank_only_ru_flow",
    ),
    "tdc_tier2_regression_mmf_rrp_ub_bank_only_qoq": (
        "tdcest",
        "tdc_tier2_regression_mmf_rrp_ub_bank_only_ru_flow",
    ),
    "tdc_tier2_regression_di_np_cu_qoq": (
        "tdcest",
        "tdc_tier2_regression_depository_institution_np_cu_ru_flow",
    ),
    "tdc_tier2_regression_mmf_rrp_lb_di_np_cu_qoq": (
        "tdcest",
        "tdc_tier2_regression_mmf_rrp_lb_depository_institution_np_cu_ru_flow",
    ),
    "tdc_tier2_regression_mmf_rrp_prop_di_np_cu_qoq": (
        "tdcest",
        "tdc_tier2_regression_mmf_rrp_prop_depository_institution_np_cu_ru_flow",
    ),
    "tdc_tier2_regression_mmf_rrp_ub_di_np_cu_qoq": (
        "tdcest",
        "tdc_tier2_regression_mmf_rrp_ub_depository_institution_np_cu_ru_flow",
    ),
    "tier2_regression_bank_row_tier_pre_component_h15_scaled": (
        "tdcest",
        "tier2_regression_bank_row_method_tier__is_pre_component_h15_scaled_backcast",
    ),
    "tier2_regression_bank_row_tier_component_pool": (
        "tdcest",
        "tier2_regression_bank_row_method_tier__is_component_pool_wamest_bucket_backcast",
    ),
    "tier2_regression_bank_row_tier_constrained": (
        "tdcest",
        "tier2_regression_bank_row_method_tier__is_constrained_component",
    ),
    "tier2_regression_di_tier_pre_component_h15_scaled": (
        "tdcest",
        "tier2_regression_di_method_tier__is_pre_component_h15_scaled_backcast",
    ),
    "tier2_regression_di_tier_component_pool": (
        "tdcest",
        "tier2_regression_di_method_tier__is_component_pool_wamest_bucket_backcast",
    ),
    "tier2_regression_di_tier_constrained": (
        "tdcest",
        "tier2_regression_di_method_tier__is_constrained_component",
    ),
    "tdc_tier2_di_np_cu_qoq": ("tdcest", "tdc_tier2_interest_corrected_depository_institution_np_cu_ru_flow"),
    "tdc_tier2_mmf_rrp_prop_bank_only_qoq": (
        "tdcest",
        "tdc_tier2_mmf_rrp_prop_bank_only_ru_flow",
    ),
    "tdc_tier2_mmf_rrp_lb_bank_only_qoq": (
        "tdcest",
        "tdc_tier2_mmf_rrp_lb_bank_only_ru_flow",
    ),
    "tdc_tier2_mmf_rrp_ub_bank_only_qoq": (
        "tdcest",
        "tdc_tier2_mmf_rrp_ub_bank_only_ru_flow",
    ),
    "tdc_tier2_mmf_rrp_prop_di_np_cu_qoq": (
        "tdcest",
        "tdc_tier2_mmf_rrp_prop_depository_institution_np_cu_ru_flow",
    ),
    "tdc_tier2_treasury_interest_robust_bank_only_qoq": (
        "tdcest",
        "tdc_tier2_treasury_interest_robust_bank_only_ru_flow",
    ),
    "tdc_tier2_treasury_interest_robust_mmf_rrp_prop_bank_only_qoq": (
        "tdcest",
        "tdc_tier2_treasury_interest_robust_mmf_rrp_prop_bank_only_ru_flow",
    ),
    "tdc_tier2_treasury_interest_robust_di_np_cu_qoq": (
        "tdcest",
        "tdc_tier2_treasury_interest_robust_depository_institution_np_cu_ru_flow",
    ),
    "tdc_tier2_treasury_interest_robust_mmf_rrp_prop_di_np_cu_qoq": (
        "tdcest",
        "tdc_tier2_treasury_interest_robust_mmf_rrp_prop_depository_institution_np_cu_ru_flow",
    ),
    "tdc_tier2_canonical_di_mmf_rrp_prop_qoq": (
        "tdcest",
        "tdc_tier2_canonical_depository_institution_mmf_rrp_prop_ru_flow",
    ),
    "mmf_rrp_adjustment_prop_qoq": ("tdcest", "mmf_rrp_adjustment_prop"),
    "tdc_du_fiscal_flow_first_pass_narrow": ("tdcest", "tdc_du_fiscal_flow_first_pass_narrow"),
    "tdc_du_fiscal_flow_first_pass_broad": ("tdcest", "tdc_du_fiscal_flow_first_pass_broad"),
    "tdc_du_residual_proxy_full_cu_ru": ("tdcest", "tdc_du_residual_proxy_full_cu_ru"),
    "du_noninterest_outlay_proxy": ("tdcest", "du_noninterest_outlay_proxy"),
    "du_receipt_proxy": ("tdcest", "du_receipt_proxy"),
    "mmf_treasury_bills_reallocation_qoq": ("tdcpass", "tdcpass_mmf_treasury_bills_reallocation_qoq"),
    "tdc_domestic_bank_only_qoq": ("tdcpass", "tdcpass_tdc_domestic_bank_only_qoq"),
    "tdc_no_toc_bank_only_qoq": ("tdcpass", "tdcpass_tdc_no_toc_bank_only_qoq"),
    "tdc_no_toc_no_row_bank_only_qoq": ("tdcpass", "tdcpass_tdc_no_toc_no_row_bank_only_qoq"),
    "tdc_core_deposit_proximate_bank_only_qoq": ("tdcpass", "tdcpass_tdc_core_deposit_proximate_bank_only_qoq"),
    "tdc_us_chartered_bank_only_qoq": ("tdcpass", "tdcpass_tdc_us_chartered_bank_only_qoq"),
    "tdc_no_foreign_bank_sectors_qoq": ("tdcpass", "tdcpass_tdc_no_foreign_bank_sectors_qoq"),
    "domestic_nonbank_deposits_qoq": ("tdcpass", "tdcpass_domestic_nonbank_deposits_qoq"),
    "domestic_nonbank_other_component_qoq": ("tdcpass", "tdcpass_domestic_nonbank_other_component_qoq"),
    "domestic_nonbank_other_component_core_deposit_proximate_qoq": (
        "tdcpass",
        "tdcpass_domestic_nonbank_other_component_core_deposit_proximate_qoq",
    ),
    "qra_ati_baseline_bn": ("qrawatch", "ati_baseline_bn"),
    "qra_net_bills_bn": ("qrawatch", "net_bills_bn"),
    "qra_bill_share": ("qrawatch", "bill_share"),
    "qra_duration_supply_weekly": ("qrawatch", "headline_public_duration_supply"),
    "gdp_deflator": ("tdcest", "gdp_deflator"),
    "tsyparty_bank_absorption_share_l1": ("tsyparty", "tsyparty_bank_absorption_share_l1"),
    "tsyparty_row_absorption_share_l1": ("tsyparty", "tsyparty_row_absorption_share_l1"),
    "tsyparty_ru_gap_l1": ("tsyparty", "tsyparty_ru_gap_l1"),
    "tsyparty_bank_foreign_official_corr_l1": ("tsyparty", "tsyparty_bank_foreign_official_corr_l1"),
    "tsyparty_bank_foreign_private_corr_l1": ("tsyparty", "tsyparty_bank_foreign_private_corr_l1"),
    "tsyparty_bank_mmf_corr_l1": ("tsyparty", "tsyparty_bank_mmf_corr_l1"),
    "tsyparty_private_minus_official_corr_l1": ("tsyparty", "tsyparty_private_minus_official_corr_l1"),
    "wamest_bank_reserve_bill_share_l1": ("wamest", "wamest_bank_reserve_bill_share_l1"),
    "wamest_bank_reserve_short_share_l1": ("wamest", "wamest_bank_reserve_short_share_l1"),
    "wamest_bank_reserve_wam_years_l1": ("wamest", "wamest_bank_reserve_wam_years_l1"),
    "wamest_bank_broad_bill_share_l1": ("wamest", "wamest_bank_broad_bill_share_l1"),
    "wamest_bank_broad_short_share_l1": ("wamest", "wamest_bank_broad_short_share_l1"),
    "wamest_bank_broad_wam_years_l1": ("wamest", "wamest_bank_broad_wam_years_l1"),
    "wamest_foreigners_short_share_l1": ("wamest", "wamest_foreigners_short_share_l1"),
    "wamest_foreigners_wam_years_l1": ("wamest", "wamest_foreigners_wam_years_l1"),
    "wamest_domestic_nonbank_short_share_l1": ("wamest", "wamest_domestic_nonbank_short_share_l1"),
    "wamest_domestic_nonbank_wam_years_l1": ("wamest", "wamest_domestic_nonbank_wam_years_l1"),
    "slrwatch_bank_leverage_pressure_l1": ("slrwatch", "slrwatch_bank_leverage_pressure_l1"),
    "slrwatch_bank_duration_pressure_l1": ("slrwatch", "slrwatch_bank_duration_pressure_l1"),
    "slrwatch_bank_funding_pressure_l1": ("slrwatch", "slrwatch_bank_funding_pressure_l1"),
    "slrwatch_bank_headroom_pp_l1": ("slrwatch", "slrwatch_bank_headroom_pp_l1"),
    "slrwatch_bank_duration_loss_dominant_share_l1": ("slrwatch", "slrwatch_bank_duration_loss_dominant_share_l1"),
    "slrwatch_bank_leverage_dominant_share_l1": ("slrwatch", "slrwatch_bank_leverage_dominant_share_l1"),
    "slrwatch_bank_funding_dominant_share_l1": ("slrwatch", "slrwatch_bank_funding_dominant_share_l1"),
    "slrwatch_parent_leverage_pressure_l1": ("slrwatch", "slrwatch_parent_leverage_pressure_l1"),
    "slrwatch_parent_duration_pressure_l1": ("slrwatch", "slrwatch_parent_duration_pressure_l1"),
    "slrwatch_parent_funding_pressure_l1": ("slrwatch", "slrwatch_parent_funding_pressure_l1"),
    "slrwatch_parent_headroom_pp_l1": ("slrwatch", "slrwatch_parent_headroom_pp_l1"),
    "accounting_deposit_substitution_qoq": ("accounting", "accounting_deposit_substitution_qoq"),
    "accounting_bank_balance_sheet_qoq": ("accounting", "accounting_bank_balance_sheet_qoq"),
    "accounting_public_liquidity_qoq": ("accounting", "accounting_public_liquidity_qoq"),
    "accounting_external_flow_qoq": ("accounting", "accounting_external_flow_qoq"),
}

BASELINE_SERIES_SCALE = {
    # tdcest deposit/TDC flows are carried in millions in the EA-TDC design.
    # These tdcpass holder-side diagnostics arrive in billions, so scale them
    # before estimating pass-through shares against the EA-TDC treatment.
    "domestic_nonbank_deposits_qoq": 1000.0,
    "domestic_nonbank_other_component_qoq": 1000.0,
    "domestic_nonbank_other_component_core_deposit_proximate_qoq": 1000.0,
    "mmf_treasury_bills_reallocation_qoq": 1000.0,
    "tdc_domestic_bank_only_qoq": 1000.0,
    "tdc_no_toc_bank_only_qoq": 1000.0,
    "tdc_no_toc_no_row_bank_only_qoq": 1000.0,
    "tdc_core_deposit_proximate_bank_only_qoq": 1000.0,
    "tdc_us_chartered_bank_only_qoq": 1000.0,
    "tdc_no_foreign_bank_sectors_qoq": 1000.0,
}

TREATMENT_COLUMN_MAP = {
    "tdc_bank_only_shock": "tdc_bank_only_qoq",
    "tdc_ru_acquisition_component_qoq": "ru_bank_only_tsy_tx",
    "tdc_treasury_cash_drain_component_qoq": "minus_treasury_operating_cash_tx",
    "tdc_positive_remit_component_qoq": "fed_remit_positive",
    "tdc_du_fiscal_flow_first_pass_narrow": "tdc_du_fiscal_flow_first_pass_narrow",
    "tdc_du_fiscal_flow_first_pass_broad": "tdc_du_fiscal_flow_first_pass_broad",
    "tdc_du_net_noninterest_fiscal_payment_proxy": "du_net_noninterest_fiscal_payment_proxy",
}

ALTERNATIVE_OTHER_COMPONENT_SERIES = {
    "other_component_tier1_mmf_rrp_plumbing_adjusted_qoq": "tdc_tier1_mmf_rrp_plumbing_adjusted_qoq",
    "other_component_tier2_bank_only_qoq": "tdc_tier2_interest_corrected_bank_only_ru_flow",
    "other_component_tier2_legacy_h15_bank_only_qoq": "tdc_tier2_legacy_h15_bank_only_qoq",
    "other_component_tier2_regression_bank_only_qoq": "tdc_tier2_regression_bank_only_qoq",
    "other_component_tier2_regression_di_np_cu_qoq": "tdc_tier2_regression_di_np_cu_qoq",
    "other_component_tier2_regression_mmf_rrp_prop_bank_only_qoq": (
        "tdc_tier2_regression_mmf_rrp_prop_bank_only_qoq"
    ),
    "other_component_tier2_regression_mmf_rrp_prop_di_np_cu_qoq": (
        "tdc_tier2_regression_mmf_rrp_prop_di_np_cu_qoq"
    ),
    "other_component_tier2_mmf_rrp_prop_bank_only_qoq": "tdc_tier2_mmf_rrp_prop_bank_only_qoq",
    "other_component_tier2_mmf_rrp_lb_bank_only_qoq": "tdc_tier2_mmf_rrp_lb_bank_only_qoq",
    "other_component_tier2_mmf_rrp_ub_bank_only_qoq": "tdc_tier2_mmf_rrp_ub_bank_only_qoq",
    "other_component_tier2_mmf_rrp_prop_di_np_cu_qoq": "tdc_tier2_mmf_rrp_prop_di_np_cu_qoq",
    "other_component_tier2_treasury_interest_robust_mmf_rrp_prop_bank_only_qoq": (
        "tdc_tier2_treasury_interest_robust_mmf_rrp_prop_bank_only_qoq"
    ),
    "other_component_tier2_canonical_di_mmf_rrp_prop_qoq": "tdc_tier2_canonical_di_mmf_rrp_prop_qoq",
    "other_component_tier2_canonical_no_toc_di_mmf_rrp_prop_qoq": (
        "tdc_tier2_canonical_no_toc_di_mmf_rrp_prop_qoq"
    ),
    "other_component_tier2_mmf_rrp_plumbing_adjusted_qoq": "tdc_tier2_mmf_rrp_plumbing_adjusted_qoq",
    "other_component_du_residual_mmf_rrp_plumbing_adjusted_qoq": "tdc_du_residual_mmf_rrp_plumbing_adjusted_qoq",
    "other_component_tier3_bank_only_qoq": "tdc_tier3_fiscal_corrected_bank_only_ru_flow",
    "other_component_tier2_broad_depository_qoq": "tdc_tier2_interest_corrected_broad_depository_np_cu_ru_flow",
    "other_component_tier3_broad_depository_qoq": "tdc_tier3_fiscal_corrected_broad_depository_np_cu_ru_flow",
}

BASELINE_REQUIRED_OUTPUTS = {
    "baseline_tdc_lp_deposits": ["matched_total_deposits", "other_component_qoq", "m2"],
    "baseline_tdc_lp_domestic_nonbank_deposits": [
        "domestic_nonbank_deposits_qoq",
        "domestic_nonbank_other_component_tier1_qoq",
        "domestic_nonbank_other_component_qoq",
        "domestic_nonbank_other_component_core_deposit_proximate_qoq",
    ],
    "baseline_tdc_lp_domestic_nonbank_deposits_tier2_bank_only": [
        "domestic_nonbank_deposits_qoq",
        "domestic_nonbank_other_component_tier2_bank_only_qoq",
        "domestic_nonbank_other_component_qoq",
        "domestic_nonbank_other_component_core_deposit_proximate_qoq",
    ],
    "tdc_tier2_mmf_rrp_canonical_full_panel": [
        "matched_total_deposits",
        "other_component_tier2_mmf_rrp_prop_bank_only_qoq",
        "other_component_tier2_mmf_rrp_lb_bank_only_qoq",
        "other_component_tier2_mmf_rrp_ub_bank_only_qoq",
        "other_component_tier2_mmf_rrp_prop_di_np_cu_qoq",
        "other_component_tier2_treasury_interest_robust_mmf_rrp_prop_bank_only_qoq",
        "other_component_tier2_canonical_di_mmf_rrp_prop_qoq",
        "other_component_tier2_canonical_no_toc_di_mmf_rrp_prop_qoq",
        "domestic_nonbank_deposits_qoq",
        "domestic_nonbank_other_component_tier2_mmf_rrp_prop_bank_only_qoq",
        "domestic_nonbank_other_component_tier2_mmf_rrp_lb_bank_only_qoq",
        "domestic_nonbank_other_component_tier2_mmf_rrp_ub_bank_only_qoq",
        "domestic_nonbank_other_component_tier2_mmf_rrp_prop_di_np_cu_qoq",
        "domestic_nonbank_other_component_tier2_treasury_interest_robust_mmf_rrp_prop_bank_only_qoq",
        "domestic_nonbank_other_component_tier2_canonical_di_mmf_rrp_prop_qoq",
        "domestic_nonbank_other_component_tier2_canonical_no_toc_di_mmf_rrp_prop_qoq",
        "m2",
        "tdcpass_strict_loan_mortgages_qoq",
        "tdcpass_strict_loan_consumer_credit_qoq",
        "tdcpass_strict_loan_core_min_qoq",
        "reserve_balances_qoq",
        "foreign_official_deposits_qoq",
        "total_reserve_balances_plus_foreign_official_qoq",
        "reserve_balances_net_fed_treasury_qoq",
        "total_reserves_plus_foreign_official_net_fed_treasury_qoq",
        "reserve_balances_net_fed_assets_qoq",
        "repo_spread",
        "fed_funds",
        "sofr",
        "dgs3mo",
        "dgs2",
        "dgs5",
        "dgs10",
        "dgs30",
        "dgs10_2y_spread",
        "dgs10_3mo_spread",
        "term_premium_10y",
        "mortgage_30y",
        "mortgage_30y_dgs10_spread",
        "baa_aaa",
        "investment_grade_oas",
        "bbb_oas",
        "high_yield_oas",
        "headline_cpi_inflation_qoq_ann",
        "core_cpi_inflation_qoq_ann",
        "headline_pce_inflation_qoq_ann",
        "core_pce_inflation_qoq_ann",
        "broad_dollar_change",
        "bank_credit_qoq",
        "bank_business_loans_qoq",
        "bank_ci_loans_h8_qoq",
        "bank_short_term_loans_z1_qoq",
        "bank_non_treasury_securities_qoq",
        "bank_treasury_securities_qoq",
        "bank_treasury_securities_transactions_qoq",
        "bank_treasury_agency_securities_qoq",
        "bank_consumer_loans_qoq",
        "bank_real_estate_loans_qoq",
        "row_loans_assets_qoq",
        "row_corp_bonds_flow",
        "row_private_flow_block",
        "current_account_balance",
        "tga_balance_qoq",
        "on_rrp_balance_qoq",
    ],
    "tdc_tier2_regression_full_panel": [
        "matched_total_deposits",
        "other_component_tier2_regression_bank_only_qoq",
        "other_component_tier2_regression_di_np_cu_qoq",
        "tdcpass_strict_loan_mortgages_qoq",
        "tdcpass_strict_loan_consumer_credit_qoq",
        "bank_consumer_loans_qoq",
        "bank_real_estate_loans_qoq",
        "mortgage_30y",
        "mortgage_30y_dgs10_spread",
        "m2",
    ],
    "tdc_du_residual_lp_domestic_nonbank_deposits": [
        "domestic_nonbank_deposits_qoq",
        "domestic_nonbank_other_component_du_residual_qoq",
    ],
    "baseline_tdc_lp_money": ["m2"],
    "baseline_tdc_lp_funding": ["reserve_balances", "repo_spread", "fed_funds", "sofr"],
    "baseline_tdc_lp_credit_spreads": ["baa_aaa", "investment_grade_oas", "high_yield_oas"],
    "baseline_tdc_lp_inflation": [
        "headline_cpi_inflation_qoq_ann",
        "core_cpi_inflation_qoq_ann",
        "core_pce_inflation_qoq_ann",
    ],
    "baseline_tdc_lp_fx": ["broad_dollar_change"],
    "baseline_tdc_lp_private_assets": [
        "bank_credit_qoq",
        "bank_business_loans_qoq",
        "bank_non_treasury_securities_qoq",
        "bank_treasury_securities_qoq",
        "bank_treasury_securities_transactions_qoq",
        "bank_treasury_agency_securities_qoq",
        "bank_consumer_loans_qoq",
        "bank_real_estate_loans_qoq",
        "row_loans_assets_qoq",
    ],
    "baseline_tdc_lp_liquidity_decomposition": [
        "reserve_balances_qoq",
        "foreign_official_deposits_qoq",
        "total_reserve_balances_plus_foreign_official_qoq",
        "fed_total_assets_qoq",
        "fed_treasury_holdings_qoq",
        "reserve_balances_net_fed_assets_qoq",
        "reserve_balances_net_fed_treasury_qoq",
        "total_reserves_plus_foreign_official_net_fed_treasury_qoq",
    ],
    "tdc_mmf_rrp_plumbing_adjusted_tier1_full_panel": [
        "matched_total_deposits",
        "other_component_tier1_mmf_rrp_plumbing_adjusted_qoq",
        "domestic_nonbank_deposits_qoq",
        "domestic_nonbank_other_component_tier1_mmf_rrp_plumbing_adjusted_qoq",
        "m2",
        "tdcpass_strict_loan_mortgages_qoq",
        "tdcpass_strict_loan_consumer_credit_qoq",
        "tdcpass_strict_loan_core_min_qoq",
        "reserve_balances_qoq",
        "foreign_official_deposits_qoq",
        "total_reserve_balances_plus_foreign_official_qoq",
        "reserve_balances_net_fed_treasury_qoq",
        "total_reserves_plus_foreign_official_net_fed_treasury_qoq",
        "reserve_balances_net_fed_assets_qoq",
        "on_rrp_balance_qoq",
        "tga_balance_qoq",
        "repo_spread",
        "fed_funds",
        "sofr",
        "dgs3mo",
        "dgs2",
        "dgs5",
        "dgs10",
        "dgs30",
        "dgs10_2y_spread",
        "dgs10_3mo_spread",
        "term_premium_10y",
        "mortgage_30y",
        "mortgage_30y_dgs10_spread",
        "baa_aaa",
        "investment_grade_oas",
        "bbb_oas",
        "high_yield_oas",
        "headline_cpi_inflation_qoq_ann",
        "core_cpi_inflation_qoq_ann",
        "headline_pce_inflation_qoq_ann",
        "core_pce_inflation_qoq_ann",
        "broad_dollar_change",
        "bank_credit_qoq",
        "bank_business_loans_qoq",
        "bank_ci_loans_h8_qoq",
        "bank_short_term_loans_z1_qoq",
        "bank_non_treasury_securities_qoq",
        "bank_treasury_securities_qoq",
        "bank_treasury_securities_transactions_qoq",
        "bank_treasury_agency_securities_qoq",
        "bank_consumer_loans_qoq",
        "bank_real_estate_loans_qoq",
        "row_loans_assets_qoq",
        "row_corp_bonds_flow",
        "row_private_flow_block",
        "current_account_balance",
    ],
    "tdc_mmf_rrp_plumbing_adjusted_tier2_full_panel": [
        "matched_total_deposits",
        "other_component_tier2_mmf_rrp_plumbing_adjusted_qoq",
        "domestic_nonbank_deposits_qoq",
        "domestic_nonbank_other_component_tier2_mmf_rrp_plumbing_adjusted_qoq",
        "m2",
        "tdcpass_strict_loan_mortgages_qoq",
        "tdcpass_strict_loan_consumer_credit_qoq",
        "tdcpass_strict_loan_core_min_qoq",
        "reserve_balances_qoq",
        "reserve_balances_net_fed_treasury_qoq",
        "reserve_balances_net_fed_assets_qoq",
        "on_rrp_balance_qoq",
        "tga_balance_qoq",
        "repo_spread",
        "fed_funds",
        "sofr",
        "dgs3mo",
        "dgs2",
        "dgs5",
        "dgs10",
        "dgs30",
        "dgs10_2y_spread",
        "dgs10_3mo_spread",
        "term_premium_10y",
        "mortgage_30y",
        "mortgage_30y_dgs10_spread",
        "baa_aaa",
        "investment_grade_oas",
        "bbb_oas",
        "high_yield_oas",
        "headline_cpi_inflation_qoq_ann",
        "core_cpi_inflation_qoq_ann",
        "headline_pce_inflation_qoq_ann",
        "core_pce_inflation_qoq_ann",
        "broad_dollar_change",
        "bank_credit_qoq",
        "bank_business_loans_qoq",
        "bank_ci_loans_h8_qoq",
        "bank_short_term_loans_z1_qoq",
        "bank_non_treasury_securities_qoq",
        "bank_treasury_securities_qoq",
        "bank_treasury_securities_transactions_qoq",
        "bank_consumer_loans_qoq",
        "bank_real_estate_loans_qoq",
        "row_loans_assets_qoq",
        "row_corp_bonds_flow",
        "row_private_flow_block",
        "current_account_balance",
    ],
    "tdc_mmf_rrp_plumbing_adjusted_domestic_residual_ladder": [
        "domestic_nonbank_deposits_qoq",
        "domestic_nonbank_other_component_qoq",
        "domestic_nonbank_other_component_core_deposit_proximate_qoq",
        "domestic_nonbank_other_component_no_row_qoq",
        "domestic_nonbank_other_component_no_toc_qoq",
        "domestic_nonbank_other_component_no_toc_no_row_qoq",
        "domestic_nonbank_other_component_tier1_mmf_rrp_plumbing_adjusted_qoq",
        "domestic_nonbank_other_component_no_row_mmf_rrp_plumbing_adjusted_qoq",
        "domestic_nonbank_other_component_no_toc_mmf_rrp_plumbing_adjusted_qoq",
        "domestic_nonbank_other_component_no_toc_no_row_mmf_rrp_plumbing_adjusted_qoq",
        "domestic_nonbank_other_component_core_mmf_rrp_plumbing_adjusted_qoq",
        "domestic_nonbank_other_component_us_chartered_mmf_rrp_plumbing_adjusted_qoq",
        "domestic_nonbank_other_component_no_foreign_bank_sectors_mmf_rrp_plumbing_adjusted_qoq",
        "mmf_on_rrp_plumbing_absorption_qoq",
        "tdc_no_toc_bank_only_qoq",
        "tdc_no_toc_no_row_bank_only_qoq",
        "tdc_core_deposit_proximate_bank_only_qoq",
        "tdc_toc_row_support_bundle_qoq",
        "row_private_flow_block",
        "tga_balance_qoq",
        "on_rrp_balance_qoq",
    ],
    "baseline_tdc_lp_deposit_sources_pct_gdp": [
        "matched_total_deposits_pct_gdp",
        "other_component_qoq_pct_gdp",
        "large_time_deposits_qoq_pct_gdp",
        "retail_mmf_assets_qoq_pct_gdp",
        "institutional_mmf_assets_qoq_pct_gdp",
        "bank_credit_qoq_pct_gdp",
        "bank_business_loans_qoq_pct_gdp",
        "bank_ci_loans_h8_qoq_pct_gdp",
        "bank_short_term_loans_z1_qoq_pct_gdp",
        "bank_non_treasury_securities_qoq_pct_gdp",
        "bank_consumer_loans_qoq_pct_gdp",
        "bank_real_estate_loans_qoq_pct_gdp",
        "row_loans_assets_qoq_pct_gdp",
        "row_corp_bonds_flow_pct_gdp",
        "row_private_flow_block_pct_gdp",
        "exports_qoq_pct_gdp",
        "imports_qoq_pct_gdp",
        "net_exports_qoq_pct_gdp",
        "current_account_balance_pct_gdp",
        "tga_balance_qoq_pct_gdp",
        "on_rrp_balance_qoq_pct_gdp",
    ],
    "baseline_tdc_lp_deposit_source_blocks": [
        "other_component_qoq",
        "deposit_substitution_block_qoq",
        "bank_balance_sheet_proxy_block_qoq",
        "public_liquidity_proxy_block_qoq",
        "external_flow_proxy_block_qoq",
        "proxy_accounting_total_qoq",
        "proxy_unexplained_gap_qoq",
    ],
    "baseline_tdc_lp_deposit_source_blocks_pct_gdp": [
        "other_component_qoq_pct_gdp",
        "deposit_substitution_block_qoq_pct_gdp",
        "bank_balance_sheet_proxy_block_qoq_pct_gdp",
        "public_liquidity_proxy_block_qoq_pct_gdp",
        "external_flow_proxy_block_qoq_pct_gdp",
        "proxy_accounting_total_qoq_pct_gdp",
        "proxy_unexplained_gap_qoq_pct_gdp",
    ],
    "baseline_tdc_lp_deposit_source_identity": [
        "other_component_qoq",
        "accounting_deposit_substitution_qoq",
        "accounting_bank_balance_sheet_qoq",
        "accounting_public_liquidity_qoq",
        "accounting_external_flow_qoq",
        "accounting_identity_total_qoq",
        "accounting_identity_gap_qoq",
    ],
    "baseline_tdc_lp_deposit_source_identity_pct_gdp": [
        "other_component_qoq_pct_gdp",
        "accounting_deposit_substitution_qoq_pct_gdp",
        "accounting_bank_balance_sheet_qoq_pct_gdp",
        "accounting_public_liquidity_qoq_pct_gdp",
        "accounting_external_flow_qoq_pct_gdp",
        "accounting_identity_total_qoq_pct_gdp",
        "accounting_identity_gap_qoq_pct_gdp",
    ],
    "tdc_lpiv_deposits_qra_ru_gap": ["matched_total_deposits", "m2", "other_component_qoq"],
    "tdc_du_fiscal_flow_broad_lp_domestic_nonbank_deposits": [
        "domestic_nonbank_deposits_qoq",
        "domestic_nonbank_other_component_qoq",
        "domestic_nonbank_other_component_core_deposit_proximate_qoq",
    ],
    "tdc_du_fiscal_flow_narrow_lp_domestic_nonbank_deposits": [
        "domestic_nonbank_deposits_qoq",
        "domestic_nonbank_other_component_qoq",
        "domestic_nonbank_other_component_core_deposit_proximate_qoq",
    ],
    "tdc_du_net_fiscal_payment_lp_domestic_nonbank_deposits": [
        "domestic_nonbank_deposits_qoq",
        "domestic_nonbank_other_component_qoq",
        "domestic_nonbank_other_component_core_deposit_proximate_qoq",
    ],
    "tdc_lpiv_deposits_qra_bank_absorption": ["matched_total_deposits", "m2", "other_component_qoq"],
    "tdc_lpiv_deposits_qra_bank_foreign_private_corr": ["matched_total_deposits", "m2", "other_component_qoq"],
    "tdc_lpiv_deposits_qra_bank_short_share": ["matched_total_deposits", "m2", "other_component_qoq"],
    "tdc_lpiv_deposits_qra_slr_bank_leverage_pressure": ["matched_total_deposits", "m2", "other_component_qoq"],
    "tdc_state_dep_low_reserves": ["matched_total_deposits", "m2", "repo_spread"],
    "tdc_state_dep_on_rrp_drain": ["matched_total_deposits", "m2", "repo_spread"],
    "tdc_state_dep_bank_foreign_private_corr": ["matched_total_deposits", "m2", "repo_spread"],
    "tdc_state_dep_bank_short_share": ["matched_total_deposits", "m2", "repo_spread"],
    "tdc_state_dep_slr_bank_leverage_pressure": ["matched_total_deposits", "m2", "repo_spread"],
}

OUTCOME_ALIASES = {
    "m2": ["m2", "M2SL"],
    "matched_total_deposits": ["matched_total_deposits", "total_deposits_bank_qoq"],
    "domestic_nonbank_deposits_qoq": ["domestic_nonbank_deposits_qoq"],
    "domestic_nonbank_other_component_qoq": ["domestic_nonbank_other_component_qoq"],
    "domestic_nonbank_other_component_tier1_qoq": ["domestic_nonbank_other_component_tier1_qoq"],
    "domestic_nonbank_other_component_tier2_bank_only_qoq": [
        "domestic_nonbank_other_component_tier2_bank_only_qoq"
    ],
    "domestic_nonbank_other_component_tier2_mmf_rrp_prop_bank_only_qoq": [
        "domestic_nonbank_other_component_tier2_mmf_rrp_prop_bank_only_qoq"
    ],
    "domestic_nonbank_other_component_tier2_mmf_rrp_lb_bank_only_qoq": [
        "domestic_nonbank_other_component_tier2_mmf_rrp_lb_bank_only_qoq"
    ],
    "domestic_nonbank_other_component_tier2_mmf_rrp_ub_bank_only_qoq": [
        "domestic_nonbank_other_component_tier2_mmf_rrp_ub_bank_only_qoq"
    ],
    "domestic_nonbank_other_component_tier2_mmf_rrp_prop_di_np_cu_qoq": [
        "domestic_nonbank_other_component_tier2_mmf_rrp_prop_di_np_cu_qoq"
    ],
    "domestic_nonbank_other_component_tier2_treasury_interest_robust_mmf_rrp_prop_bank_only_qoq": [
        "domestic_nonbank_other_component_tier2_treasury_interest_robust_mmf_rrp_prop_bank_only_qoq"
    ],
    "domestic_nonbank_other_component_tier2_canonical_no_toc_di_mmf_rrp_prop_qoq": [
        "domestic_nonbank_other_component_tier2_canonical_no_toc_di_mmf_rrp_prop_qoq"
    ],
    "domestic_nonbank_other_component_du_residual_qoq": [
        "domestic_nonbank_other_component_du_residual_qoq"
    ],
    "domestic_nonbank_other_component_core_deposit_proximate_qoq": [
        "domestic_nonbank_other_component_core_deposit_proximate_qoq"
    ],
    "domestic_nonbank_other_component_no_row_qoq": [
        "domestic_nonbank_other_component_no_row_qoq"
    ],
    "domestic_nonbank_other_component_no_toc_qoq": [
        "domestic_nonbank_other_component_no_toc_qoq"
    ],
    "domestic_nonbank_other_component_no_toc_no_row_qoq": [
        "domestic_nonbank_other_component_no_toc_no_row_qoq"
    ],
    "domestic_nonbank_other_component_tier1_mmf_rrp_plumbing_adjusted_qoq": [
        "domestic_nonbank_other_component_tier1_mmf_rrp_plumbing_adjusted_qoq"
    ],
    "domestic_nonbank_other_component_tier2_mmf_rrp_plumbing_adjusted_qoq": [
        "domestic_nonbank_other_component_tier2_mmf_rrp_plumbing_adjusted_qoq"
    ],
    "domestic_nonbank_other_component_du_residual_mmf_rrp_plumbing_adjusted_qoq": [
        "domestic_nonbank_other_component_du_residual_mmf_rrp_plumbing_adjusted_qoq"
    ],
    "domestic_nonbank_other_component_no_row_mmf_rrp_plumbing_adjusted_qoq": [
        "domestic_nonbank_other_component_no_row_mmf_rrp_plumbing_adjusted_qoq"
    ],
    "domestic_nonbank_other_component_no_toc_mmf_rrp_plumbing_adjusted_qoq": [
        "domestic_nonbank_other_component_no_toc_mmf_rrp_plumbing_adjusted_qoq"
    ],
    "domestic_nonbank_other_component_no_toc_no_row_mmf_rrp_plumbing_adjusted_qoq": [
        "domestic_nonbank_other_component_no_toc_no_row_mmf_rrp_plumbing_adjusted_qoq"
    ],
    "domestic_nonbank_other_component_core_mmf_rrp_plumbing_adjusted_qoq": [
        "domestic_nonbank_other_component_core_mmf_rrp_plumbing_adjusted_qoq"
    ],
    "domestic_nonbank_other_component_us_chartered_mmf_rrp_plumbing_adjusted_qoq": [
        "domestic_nonbank_other_component_us_chartered_mmf_rrp_plumbing_adjusted_qoq"
    ],
    "domestic_nonbank_other_component_no_foreign_bank_sectors_mmf_rrp_plumbing_adjusted_qoq": [
        "domestic_nonbank_other_component_no_foreign_bank_sectors_mmf_rrp_plumbing_adjusted_qoq"
    ],
    "large_time_deposits_qoq": ["large_time_deposits_qoq"],
    "retail_mmf_assets_qoq": ["retail_mmf_assets_qoq"],
    "institutional_mmf_assets_qoq": ["institutional_mmf_assets_qoq"],
    "reserve_balances": ["reserve_balances", "WRESBAL", "TOTRESNS"],
    "reserve_balances_qoq": ["reserve_balances_qoq"],
    "foreign_official_deposits_qoq": ["foreign_official_deposits_qoq"],
    "total_reserve_balances_plus_foreign_official_qoq": [
        "total_reserve_balances_plus_foreign_official_qoq"
    ],
    "fed_total_assets_qoq": ["fed_total_assets_qoq"],
    "fed_treasury_holdings_qoq": ["fed_treasury_holdings_qoq"],
    "reserve_balances_net_fed_assets_qoq": ["reserve_balances_net_fed_assets_qoq"],
    "reserve_balances_net_fed_treasury_qoq": ["reserve_balances_net_fed_treasury_qoq"],
    "total_reserves_plus_foreign_official_net_fed_treasury_qoq": [
        "total_reserves_plus_foreign_official_net_fed_treasury_qoq"
    ],
    "repo_spread": ["repo_spread"],
    "fed_funds": ["fed_funds", "FEDFUNDS"],
    "sofr": ["sofr", "SOFR"],
    "dgs3mo": ["dgs3mo", "DGS3MO"],
    "dgs2": ["dgs2", "DGS2"],
    "dgs5": ["dgs5", "DGS5"],
    "dgs10": ["dgs10", "DGS10"],
    "dgs30": ["dgs30", "DGS30"],
    "term_premium_10y": ["term_premium_10y", "THREEFYTP10"],
    "mortgage_30y": ["mortgage_30y", "MORTGAGE30US"],
    "mortgage_30y_dgs10_spread": ["mortgage_30y_dgs10_spread"],
    "dgs10_2y_spread": ["dgs10_2y_spread"],
    "dgs10_3mo_spread": ["dgs10_3mo_spread"],
    "baa_aaa": ["baa_aaa"],
    "investment_grade_oas": ["investment_grade_oas", "BAMLC0A0CM"],
    "bbb_oas": ["bbb_oas", "BAMLC0A4CBBB"],
    "high_yield_oas": ["high_yield_oas", "BAMLH0A0HYM2"],
    "headline_cpi_inflation_qoq_ann": ["headline_cpi_inflation_qoq_ann"],
    "core_cpi_inflation_qoq_ann": ["core_cpi_inflation_qoq_ann"],
    "headline_pce_inflation_qoq_ann": ["headline_pce_inflation_qoq_ann"],
    "core_pce_inflation_qoq_ann": ["core_pce_inflation_qoq_ann"],
    "broad_dollar_change": ["broad_dollar_change"],
    "bank_credit_qoq": ["bank_credit_qoq"],
    "bank_business_loans_qoq": ["bank_business_loans_qoq"],
    "bank_ci_loans_h8_qoq": ["bank_ci_loans_h8_qoq"],
    "bank_short_term_loans_z1_qoq": ["bank_short_term_loans_z1_qoq"],
    "bank_non_treasury_securities_qoq": ["bank_non_treasury_securities_qoq"],
    "bank_treasury_securities_qoq": ["bank_treasury_securities_qoq"],
    "bank_treasury_securities_transactions_qoq": ["bank_treasury_securities_transactions_qoq"],
    "bank_treasury_agency_securities_qoq": ["bank_treasury_agency_securities_qoq"],
    "bank_consumer_loans_qoq": ["bank_consumer_loans_qoq"],
    "bank_real_estate_loans_qoq": ["bank_real_estate_loans_qoq"],
    "row_loans_assets_qoq": ["row_loans_assets_qoq"],
    "row_corp_bonds_flow": ["row_corp_bonds_flow"],
    "row_private_flow_block": ["row_private_flow_block"],
    "tdc_no_toc_bank_only_qoq": ["tdc_no_toc_bank_only_qoq"],
    "tdc_no_toc_no_row_bank_only_qoq": ["tdc_no_toc_no_row_bank_only_qoq"],
    "tdc_core_deposit_proximate_bank_only_qoq": ["tdc_core_deposit_proximate_bank_only_qoq"],
    "tdc_toc_row_support_bundle_qoq": ["tdc_toc_row_support_bundle_qoq"],
    "current_account_balance": ["current_account_balance"],
    "tga_balance_qoq": ["tga_balance_qoq"],
    "on_rrp_balance_qoq": ["on_rrp_balance_qoq"],
    "mmf_on_rrp_plumbing_absorption_qoq": ["mmf_on_rrp_plumbing_absorption_qoq"],
    "deposit_substitution_block_qoq": ["deposit_substitution_block_qoq"],
    "bank_balance_sheet_proxy_block_qoq": ["bank_balance_sheet_proxy_block_qoq"],
    "public_liquidity_proxy_block_qoq": ["public_liquidity_proxy_block_qoq"],
    "external_flow_proxy_block_qoq": ["external_flow_proxy_block_qoq"],
    "proxy_accounting_total_qoq": ["proxy_accounting_total_qoq"],
    "proxy_unexplained_gap_qoq": ["proxy_unexplained_gap_qoq"],
    "accounting_deposit_substitution_qoq": ["accounting_deposit_substitution_qoq"],
    "accounting_bank_balance_sheet_qoq": ["accounting_bank_balance_sheet_qoq"],
    "accounting_public_liquidity_qoq": ["accounting_public_liquidity_qoq"],
    "accounting_external_flow_qoq": ["accounting_external_flow_qoq"],
    "accounting_identity_total_qoq": ["accounting_identity_total_qoq"],
    "accounting_identity_gap_qoq": ["accounting_identity_gap_qoq"],
    "other_component_tier2_bank_only_qoq": ["other_component_tier2_bank_only_qoq"],
    "other_component_tier2_regression_bank_only_qoq": ["other_component_tier2_regression_bank_only_qoq"],
    "other_component_tier2_regression_di_np_cu_qoq": ["other_component_tier2_regression_di_np_cu_qoq"],
    "other_component_tier2_mmf_rrp_prop_bank_only_qoq": [
        "other_component_tier2_mmf_rrp_prop_bank_only_qoq"
    ],
    "other_component_tier2_mmf_rrp_lb_bank_only_qoq": [
        "other_component_tier2_mmf_rrp_lb_bank_only_qoq"
    ],
    "other_component_tier2_mmf_rrp_ub_bank_only_qoq": [
        "other_component_tier2_mmf_rrp_ub_bank_only_qoq"
    ],
    "other_component_tier2_mmf_rrp_prop_di_np_cu_qoq": [
        "other_component_tier2_mmf_rrp_prop_di_np_cu_qoq"
    ],
    "other_component_tier2_treasury_interest_robust_mmf_rrp_prop_bank_only_qoq": [
        "other_component_tier2_treasury_interest_robust_mmf_rrp_prop_bank_only_qoq"
    ],
    "other_component_tier2_canonical_di_mmf_rrp_prop_qoq": [
        "other_component_tier2_canonical_di_mmf_rrp_prop_qoq"
    ],
    "other_component_tier2_canonical_no_toc_di_mmf_rrp_prop_qoq": [
        "other_component_tier2_canonical_no_toc_di_mmf_rrp_prop_qoq"
    ],
    "other_component_tier1_mmf_rrp_plumbing_adjusted_qoq": [
        "other_component_tier1_mmf_rrp_plumbing_adjusted_qoq"
    ],
    "other_component_tier2_mmf_rrp_plumbing_adjusted_qoq": [
        "other_component_tier2_mmf_rrp_plumbing_adjusted_qoq"
    ],
    "other_component_du_residual_mmf_rrp_plumbing_adjusted_qoq": [
        "other_component_du_residual_mmf_rrp_plumbing_adjusted_qoq"
    ],
    "other_component_tier3_bank_only_qoq": ["other_component_tier3_bank_only_qoq"],
    "other_component_tier2_broad_depository_qoq": ["other_component_tier2_broad_depository_qoq"],
    "other_component_tier3_broad_depository_qoq": ["other_component_tier3_broad_depository_qoq"],
    "accounting_identity_gap_tier2_bank_only_qoq": ["accounting_identity_gap_tier2_bank_only_qoq"],
    "accounting_identity_gap_tier3_bank_only_qoq": ["accounting_identity_gap_tier3_bank_only_qoq"],
    "accounting_identity_gap_tier2_broad_depository_qoq": ["accounting_identity_gap_tier2_broad_depository_qoq"],
    "accounting_identity_gap_tier3_broad_depository_qoq": ["accounting_identity_gap_tier3_broad_depository_qoq"],
    "matched_total_deposits_pct_gdp": ["matched_total_deposits_pct_gdp"],
    "other_component_qoq_pct_gdp": ["other_component_qoq_pct_gdp"],
    "other_component_tier2_bank_only_qoq_pct_gdp": ["other_component_tier2_bank_only_qoq_pct_gdp"],
    "other_component_tier3_bank_only_qoq_pct_gdp": ["other_component_tier3_bank_only_qoq_pct_gdp"],
    "other_component_tier2_broad_depository_qoq_pct_gdp": ["other_component_tier2_broad_depository_qoq_pct_gdp"],
    "other_component_tier3_broad_depository_qoq_pct_gdp": ["other_component_tier3_broad_depository_qoq_pct_gdp"],
    "large_time_deposits_qoq_pct_gdp": ["large_time_deposits_qoq_pct_gdp"],
    "retail_mmf_assets_qoq_pct_gdp": ["retail_mmf_assets_qoq_pct_gdp"],
    "institutional_mmf_assets_qoq_pct_gdp": ["institutional_mmf_assets_qoq_pct_gdp"],
    "bank_credit_qoq_pct_gdp": ["bank_credit_qoq_pct_gdp"],
    "bank_business_loans_qoq_pct_gdp": ["bank_business_loans_qoq_pct_gdp"],
    "bank_ci_loans_h8_qoq_pct_gdp": ["bank_ci_loans_h8_qoq_pct_gdp"],
    "bank_short_term_loans_z1_qoq_pct_gdp": ["bank_short_term_loans_z1_qoq_pct_gdp"],
    "bank_non_treasury_securities_qoq_pct_gdp": ["bank_non_treasury_securities_qoq_pct_gdp"],
    "bank_consumer_loans_qoq_pct_gdp": ["bank_consumer_loans_qoq_pct_gdp"],
    "bank_real_estate_loans_qoq_pct_gdp": ["bank_real_estate_loans_qoq_pct_gdp"],
    "row_loans_assets_qoq_pct_gdp": ["row_loans_assets_qoq_pct_gdp"],
    "row_corp_bonds_flow_pct_gdp": ["row_corp_bonds_flow_pct_gdp"],
    "row_private_flow_block_pct_gdp": ["row_private_flow_block_pct_gdp"],
    "exports_qoq_pct_gdp": ["exports_qoq_pct_gdp"],
    "imports_qoq_pct_gdp": ["imports_qoq_pct_gdp"],
    "net_exports_qoq_pct_gdp": ["net_exports_qoq_pct_gdp"],
    "current_account_balance_pct_gdp": ["current_account_balance_pct_gdp"],
    "tga_balance_qoq_pct_gdp": ["tga_balance_qoq_pct_gdp"],
    "on_rrp_balance_qoq_pct_gdp": ["on_rrp_balance_qoq_pct_gdp"],
    "deposit_substitution_block_qoq_pct_gdp": ["deposit_substitution_block_qoq_pct_gdp"],
    "bank_balance_sheet_proxy_block_qoq_pct_gdp": ["bank_balance_sheet_proxy_block_qoq_pct_gdp"],
    "public_liquidity_proxy_block_qoq_pct_gdp": ["public_liquidity_proxy_block_qoq_pct_gdp"],
    "external_flow_proxy_block_qoq_pct_gdp": ["external_flow_proxy_block_qoq_pct_gdp"],
    "proxy_accounting_total_qoq_pct_gdp": ["proxy_accounting_total_qoq_pct_gdp"],
    "proxy_unexplained_gap_qoq_pct_gdp": ["proxy_unexplained_gap_qoq_pct_gdp"],
    "accounting_deposit_substitution_qoq_pct_gdp": ["accounting_deposit_substitution_qoq_pct_gdp"],
    "accounting_bank_balance_sheet_qoq_pct_gdp": ["accounting_bank_balance_sheet_qoq_pct_gdp"],
    "accounting_public_liquidity_qoq_pct_gdp": ["accounting_public_liquidity_qoq_pct_gdp"],
    "accounting_external_flow_qoq_pct_gdp": ["accounting_external_flow_qoq_pct_gdp"],
    "accounting_identity_total_qoq_pct_gdp": ["accounting_identity_total_qoq_pct_gdp"],
    "accounting_identity_gap_qoq_pct_gdp": ["accounting_identity_gap_qoq_pct_gdp"],
    "accounting_identity_gap_tier2_bank_only_qoq_pct_gdp": ["accounting_identity_gap_tier2_bank_only_qoq_pct_gdp"],
    "accounting_identity_gap_tier3_bank_only_qoq_pct_gdp": ["accounting_identity_gap_tier3_bank_only_qoq_pct_gdp"],
    "accounting_identity_gap_tier2_broad_depository_qoq_pct_gdp": ["accounting_identity_gap_tier2_broad_depository_qoq_pct_gdp"],
    "accounting_identity_gap_tier3_broad_depository_qoq_pct_gdp": ["accounting_identity_gap_tier3_broad_depository_qoq_pct_gdp"],
}

PCT_GDP_SOURCE_OUTCOME_MAP = {
    "matched_total_deposits_pct_gdp": "matched_total_deposits",
    "other_component_qoq_pct_gdp": "other_component_qoq",
    "other_component_tier2_bank_only_qoq_pct_gdp": "other_component_tier2_bank_only_qoq",
    "other_component_tier3_bank_only_qoq_pct_gdp": "other_component_tier3_bank_only_qoq",
    "other_component_tier2_broad_depository_qoq_pct_gdp": "other_component_tier2_broad_depository_qoq",
    "other_component_tier3_broad_depository_qoq_pct_gdp": "other_component_tier3_broad_depository_qoq",
    "large_time_deposits_qoq_pct_gdp": "large_time_deposits_qoq",
    "retail_mmf_assets_qoq_pct_gdp": "retail_mmf_assets_qoq",
    "institutional_mmf_assets_qoq_pct_gdp": "institutional_mmf_assets_qoq",
    "bank_credit_qoq_pct_gdp": "bank_credit_qoq",
    "bank_business_loans_qoq_pct_gdp": "bank_business_loans_qoq",
    "bank_ci_loans_h8_qoq_pct_gdp": "bank_ci_loans_h8_qoq",
    "bank_short_term_loans_z1_qoq_pct_gdp": "bank_short_term_loans_z1_qoq",
    "bank_non_treasury_securities_qoq_pct_gdp": "bank_non_treasury_securities_qoq",
    "bank_consumer_loans_qoq_pct_gdp": "bank_consumer_loans_qoq",
    "bank_real_estate_loans_qoq_pct_gdp": "bank_real_estate_loans_qoq",
    "row_loans_assets_qoq_pct_gdp": "row_loans_assets_qoq",
    "row_corp_bonds_flow_pct_gdp": "row_corp_bonds_flow",
    "row_private_flow_block_pct_gdp": "row_private_flow_block",
    "exports_qoq_pct_gdp": "exports_qoq",
    "imports_qoq_pct_gdp": "imports_qoq",
    "net_exports_qoq_pct_gdp": "net_exports_qoq",
    "current_account_balance_pct_gdp": "current_account_balance",
    "tga_balance_qoq_pct_gdp": "tga_balance_qoq",
    "on_rrp_balance_qoq_pct_gdp": "on_rrp_balance_qoq",
    "deposit_substitution_block_qoq_pct_gdp": "deposit_substitution_block_qoq",
    "bank_balance_sheet_proxy_block_qoq_pct_gdp": "bank_balance_sheet_proxy_block_qoq",
    "public_liquidity_proxy_block_qoq_pct_gdp": "public_liquidity_proxy_block_qoq",
    "external_flow_proxy_block_qoq_pct_gdp": "external_flow_proxy_block_qoq",
    "proxy_accounting_total_qoq_pct_gdp": "proxy_accounting_total_qoq",
    "proxy_unexplained_gap_qoq_pct_gdp": "proxy_unexplained_gap_qoq",
    "accounting_deposit_substitution_qoq_pct_gdp": "accounting_deposit_substitution_qoq",
    "accounting_bank_balance_sheet_qoq_pct_gdp": "accounting_bank_balance_sheet_qoq",
    "accounting_public_liquidity_qoq_pct_gdp": "accounting_public_liquidity_qoq",
    "accounting_external_flow_qoq_pct_gdp": "accounting_external_flow_qoq",
    "accounting_identity_total_qoq_pct_gdp": "accounting_identity_total_qoq",
    "accounting_identity_gap_qoq_pct_gdp": "accounting_identity_gap_qoq",
    "accounting_identity_gap_tier2_bank_only_qoq_pct_gdp": "accounting_identity_gap_tier2_bank_only_qoq",
    "accounting_identity_gap_tier3_bank_only_qoq_pct_gdp": "accounting_identity_gap_tier3_bank_only_qoq",
    "accounting_identity_gap_tier2_broad_depository_qoq_pct_gdp": "accounting_identity_gap_tier2_broad_depository_qoq",
    "accounting_identity_gap_tier3_broad_depository_qoq_pct_gdp": "accounting_identity_gap_tier3_broad_depository_qoq",
}

ROW_LINEAR_COMBO_SERIES = {
    "deposit_substitution_block_qoq": (
        ("large_time_deposits_qoq", 1.0),
        ("retail_mmf_assets_qoq", -1.0),
        ("institutional_mmf_assets_qoq", -1.0),
    ),
    "bank_balance_sheet_proxy_block_qoq": (
        ("bank_non_treasury_securities_qoq", 1.0),
        ("bank_short_term_loans_z1_qoq", 1.0),
    ),
    "public_liquidity_proxy_block_qoq": (
        ("tga_balance_qoq", -1.0),
        ("on_rrp_balance_qoq", -1.0),
    ),
    "external_flow_proxy_block_qoq": (
        ("row_private_flow_block", 1.0),
        ("net_exports_qoq", 1.0),
    ),
    "proxy_accounting_total_qoq": (
        ("deposit_substitution_block_qoq", 1.0),
        ("bank_balance_sheet_proxy_block_qoq", 1.0),
        ("public_liquidity_proxy_block_qoq", 1.0),
        ("external_flow_proxy_block_qoq", 1.0),
    ),
    "proxy_unexplained_gap_qoq": (
        ("other_component_qoq", 1.0),
        ("proxy_accounting_total_qoq", -1.0),
    ),
    "accounting_identity_total_qoq": (
        ("accounting_deposit_substitution_qoq", 1.0),
        ("accounting_bank_balance_sheet_qoq", 1.0),
        ("accounting_public_liquidity_qoq", 1.0),
        ("accounting_external_flow_qoq", 1.0),
    ),
    "accounting_identity_gap_qoq": (
        ("other_component_qoq", 1.0),
        ("accounting_identity_total_qoq", -1.0),
    ),
    "accounting_identity_gap_tier2_bank_only_qoq": (
        ("other_component_tier2_bank_only_qoq", 1.0),
        ("accounting_identity_total_qoq", -1.0),
    ),
    "accounting_identity_gap_tier3_bank_only_qoq": (
        ("other_component_tier3_bank_only_qoq", 1.0),
        ("accounting_identity_total_qoq", -1.0),
    ),
    "accounting_identity_gap_tier2_broad_depository_qoq": (
        ("other_component_tier2_broad_depository_qoq", 1.0),
        ("accounting_identity_total_qoq", -1.0),
    ),
    "accounting_identity_gap_tier3_broad_depository_qoq": (
        ("other_component_tier3_broad_depository_qoq", 1.0),
        ("accounting_identity_total_qoq", -1.0),
    ),
    "tdc_tier2_canonical_no_toc_di_mmf_rrp_prop_qoq": (
        ("tdc_tier2_canonical_di_mmf_rrp_prop_qoq", 1.0),
        ("tdc_no_toc_bank_only_qoq", 1.0),
        ("tdc_bank_only_qoq", -1.0),
    ),
    "tdc_toc_row_support_bundle_qoq": (
        ("tdc_bank_only_qoq", 1.0),
        ("tdc_core_deposit_proximate_bank_only_qoq", -1.0),
    ),
    "du_net_noninterest_fiscal_payment_proxy": (
        ("du_noninterest_outlay_proxy", 1.0),
        ("du_receipt_proxy", -1.0),
    ),
}

INSTRUMENT_COMPONENTS = {
    "iv_qra_x_ru_gap": {
        "shock": "qra_maturity_tilt_flow",
        "state": "tsyparty_ru_gap_l1",
    },
    "iv_qra_x_bank_absorption": {
        "shock": "qra_maturity_tilt_flow",
        "state": "tsyparty_bank_absorption_share_l1",
    },
    "iv_qra_x_bank_foreign_private_corr": {
        "shock": "qra_maturity_tilt_flow",
        "state": "tsyparty_bank_foreign_private_corr_l1",
    },
    "iv_qra_x_bank_short_share": {
        "shock": "qra_maturity_tilt_flow",
        "state": "wamest_bank_reserve_short_share_l1",
    },
    "iv_qra_x_slr_bank_leverage_pressure": {
        "shock": "qra_maturity_tilt_flow",
        "state": "slrwatch_bank_leverage_pressure_l1",
    },
}

RAW_FRED_LEVEL_CHANGE_SERIES = {
    "matched_total_deposits": "BOGZ1FL764100005Q",
    "large_time_deposits_qoq": "LTDACBM027NBOG",
    "retail_mmf_assets_qoq": "RMFSL",
    "institutional_mmf_assets_qoq": "WIMFSL",
    "institutional_mmf_assets_z1_qoq": "BOGZ1FL883034010Q",
    "reserve_balances_qoq": "WRESBAL",
    "foreign_official_deposits_qoq": "WDFOL",
    "fed_total_assets_qoq": "WALCL",
    "fed_treasury_holdings_qoq": "TREAST",
    "exports_qoq": "EXPGS",
    "imports_qoq": "IMPGS",
    "tga_balance_qoq": "WDTGAL",
    "on_rrp_balance_qoq": "RRPONTSYD",
    "bank_credit_qoq": "TOTBKCR",
    "bank_business_loans_qoq": "BUSLOANS",
    "bank_ci_loans_h8_qoq": "TOTCI",
    "bank_short_term_loans_z1_qoq": "BOGZ1FL704041005Q",
    "bank_non_treasury_securities_qoq": "OSEACBW027SBOG",
    "bank_treasury_securities_qoq": "BOGZ1FL763061100Q",
    "bank_treasury_agency_securities_qoq": "TASACBW027SBOG",
    "bank_consumer_loans_qoq": "CLSACBW027SBOG",
    "bank_real_estate_loans_qoq": "RELACBW027SBOG",
    "row_loans_assets_qoq": "BOGZ1FL264035005Q",
}

RAW_FRED_QOQ_FALLBACK_SERIES = {
    "institutional_mmf_assets_qoq": "institutional_mmf_assets_z1_qoq",
}

RAW_FRED_DIRECT_SERIES = {
    "bank_treasury_securities_transactions_qoq": "BOGZ1FU763061100Q",
    "current_account_balance": "BOPBCA",
    "row_corp_bonds_flow": "ROWCBAQ027S",
    "row_corp_equities_flow": "ROWCEAQ027S",
    "row_agency_flow": "ROWGSEQ027S",
    "row_nonfin_business_loans_flow": "ROWNBLQ027S",
    "reserve_balances": "WRESBAL",
    "dgs3mo": "DGS3MO",
    "dgs2": "DGS2",
    "dgs5": "DGS5",
    "dgs10": "DGS10",
    "dgs30": "DGS30",
    "term_premium_10y": "THREEFYTP10",
    "mortgage_30y": "MORTGAGE30US",
    "investment_grade_oas": "BAMLC0A0CM",
    "bbb_oas": "BAMLC0A4CBBB",
    "high_yield_oas": "BAMLH0A0HYM2",
}

RAW_FRED_LOG_DIFF_ANNUALIZED_SERIES = {
    "headline_cpi_inflation_qoq_ann": "CPIAUCSL",
    "core_cpi_inflation_qoq_ann": "CPILFESL",
    "headline_pce_inflation_qoq_ann": "PCEPI",
    "core_pce_inflation_qoq_ann": "PCEPILFE",
}

RAW_FRED_LEVEL_CHANGE_FROM_AGGREGATES = {
    "broad_dollar_change": "DTWEXBGS",
}

RAW_FRED_DIFFERENCE_SERIES = {
    "reserve_balances_net_fed_assets_qoq": ("reserve_balances_qoq", "fed_total_assets_qoq"),
    "reserve_balances_net_fed_treasury_qoq": ("reserve_balances_qoq", "fed_treasury_holdings_qoq"),
    "total_reserves_plus_foreign_official_net_fed_treasury_qoq": (
        "total_reserve_balances_plus_foreign_official_qoq",
        "fed_treasury_holdings_qoq",
    ),
    "net_exports_qoq": ("exports_qoq", "imports_qoq"),
}

RAW_FRED_QOQ_SUM_SERIES = {
    "total_reserve_balances_plus_foreign_official_qoq": (
        "reserve_balances_qoq",
        "foreign_official_deposits_qoq",
    ),
}

RAW_FRED_SUM_SERIES = {
    "row_private_flow_block": (
        "row_corp_bonds_flow",
        "row_corp_equities_flow",
        "row_agency_flow",
        "row_nonfin_business_loans_flow",
    ),
}

RAW_FRED_SPREAD_SERIES = {
    "repo_spread": ("TGCRRATE", "RRPONTSYAWARD", 100.0),
    "baa_aaa": ("BAA", "AAA", 100.0),
    "mortgage_30y_dgs10_spread": ("MORTGAGE30US", "DGS10", 100.0),
    "dgs10_2y_spread": ("DGS10", "DGS2", 100.0),
    "dgs10_3mo_spread": ("DGS10", "DGS3MO", 100.0),
}

RAW_FRED_LAG_DAYS = {
    "matched_total_deposits": 90,
    "large_time_deposits_qoq": 30,
    "retail_mmf_assets_qoq": 30,
    "institutional_mmf_assets_qoq": 14,
    "reserve_balances": 14,
    "repo_spread": 7,
    "baa_aaa": 30,
    "investment_grade_oas": 7,
    "high_yield_oas": 7,
    "headline_cpi_inflation_qoq_ann": 30,
    "core_cpi_inflation_qoq_ann": 30,
    "core_pce_inflation_qoq_ann": 30,
    "broad_dollar_change": 7,
    "fed_total_assets_qoq": 14,
    "fed_treasury_holdings_qoq": 14,
    "foreign_official_deposits_qoq": 14,
    "total_reserve_balances_plus_foreign_official_qoq": 14,
    "current_account_balance": 90,
    "reserve_balances_qoq": 14,
    "reserve_balances_net_fed_assets_qoq": 14,
    "reserve_balances_net_fed_treasury_qoq": 14,
    "total_reserves_plus_foreign_official_net_fed_treasury_qoq": 14,
    "exports_qoq": 30,
    "imports_qoq": 30,
    "net_exports_qoq": 30,
    "tga_balance_qoq": 14,
    "on_rrp_balance_qoq": 14,
    "bank_credit_qoq": 30,
    "bank_business_loans_qoq": 30,
    "bank_ci_loans_h8_qoq": 14,
    "bank_short_term_loans_z1_qoq": 90,
    "bank_non_treasury_securities_qoq": 30,
    "bank_treasury_securities_qoq": 90,
    "bank_treasury_securities_transactions_qoq": 90,
    "bank_treasury_agency_securities_qoq": 30,
    "bank_consumer_loans_qoq": 30,
    "bank_real_estate_loans_qoq": 30,
    "row_loans_assets_qoq": 90,
    "row_corp_bonds_flow": 90,
    "row_private_flow_block": 90,
    "coord_low_reserve_state_l1": 14,
    "coord_liquidity_tightness_q_z_l1": 14,
    "coord_on_rrp_drain_state_l1": 14,
}

FRED_QUARTERLY_AGGREGATION = {
    "FEDFUNDS": "mean",
    "SOFR": "mean",
    "TGCRRATE": "mean",
    "RRPONTSYAWARD": "mean",
    "RRPONTSYD": "mean",
    "IORB": "mean",
    "BAA": "mean",
    "AAA": "mean",
    "BAMLC0A0CM": "mean",
    "BAMLC0A4CBBB": "mean",
    "BAMLH0A0HYM2": "mean",
    "DGS3MO": "mean",
    "DGS2": "mean",
    "DGS5": "mean",
    "DGS10": "mean",
    "DGS30": "mean",
    "THREEFYTP10": "mean",
    "MORTGAGE30US": "mean",
    "CPIAUCSL": "mean",
    "CPILFESL": "mean",
    "PCEPI": "mean",
    "PCEPILFE": "mean",
    "DTWEXBGS": "mean",
}


@dataclass(frozen=True)
class DesignBuildResult:
    bundle_path: Path
    design_manifest_path: Path
    sample_manifest_path: Path
    diagnostics_manifest_path: Path | None
    rows_written: int
    usable_rows: int


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_jobs(config_path: Path) -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    jobs = payload.get("jobs", [])
    if not isinstance(jobs, list):
        raise TypeError("Expected 'jobs' list in dass blueprint")
    result: dict[str, dict[str, Any]] = {}
    for item in jobs:
        if isinstance(item, dict) and item.get("job_id"):
            result[str(item["job_id"])] = item
    return result


def _parse_available_at(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if "T" in text:
        text = text.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    return datetime.fromisoformat(f"{text}T00:00:00")


def _quarter_from_date(text: str) -> str:
    date_value = datetime.fromisoformat(str(text)[:10]).date()
    quarter = ((date_value.month - 1) // 3) + 1
    return f"{date_value.year}Q{quarter}"


def _quarter_sort_key(quarter: str) -> tuple[int, int]:
    year_text, quarter_text = quarter.split("Q", 1)
    return int(year_text), int(quarter_text)


def _quarter_end_date(quarter: str) -> date:
    year, quarter_num = _quarter_sort_key(quarter)
    month = quarter_num * 3
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    return next_month - timedelta(days=1)


def _conservative_quarterly_available_at(quarter: str, *, lag_days: int = 90) -> str:
    return (_quarter_end_date(quarter) + timedelta(days=lag_days)).isoformat()


def _stable_float_text(value: float, *, digits: int = 10) -> str:
    return str(round(float(value), digits))


def _coerce_float(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _max_available_iso(*values: str) -> str:
    parsed = [item for item in (_parse_available_at(value) for value in values) if item is not None]
    if not parsed:
        return ""
    return max(parsed).isoformat()


def _previous_quarter(quarter: str) -> str | None:
    year, quarter_num = _quarter_sort_key(quarter)
    if quarter_num == 1:
        return f"{year - 1}Q4" if year > 1 else None
    return f"{year}Q{quarter_num - 1}"


def _summarize_float_series(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    mean_value = fmean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "mean": round(mean_value, 6),
        "std": round(variance ** 0.5, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = fmean(left)
    right_mean = fmean(right)
    left_var = sum((value - left_mean) ** 2 for value in left)
    right_var = sum((value - right_mean) ** 2 for value in right)
    if left_var <= 0 or right_var <= 0:
        return None
    covariance = sum((left_value - left_mean) * (right_value - right_mean) for left_value, right_value in zip(left, right))
    return round(covariance / ((left_var * right_var) ** 0.5), 6)


def _build_iv_diagnostics(
    *,
    bundle_rows: list[dict[str, str]],
    job_id: str,
    treatment_id: str,
    required_outputs: list[str],
    required_state_ids: list[str],
    configured_instruments: list[str],
) -> dict[str, Any]:
    eligible_rows: list[dict[str, str]] = []

    def _row_has_outcome(row: dict[str, str], outcome: str) -> bool:
        aliases = OUTCOME_ALIASES.get(outcome, [outcome])
        return any(str(row.get(alias, "")).strip() != "" for alias in aliases)

    for row in bundle_rows:
        if not str(row.get(treatment_id, "")).strip():
            continue
        if any(not _row_has_outcome(row, outcome) for outcome in required_outputs):
            continue
        if any(not str(row.get(state_id, "")).strip() for state_id in required_state_ids):
            continue
        if any(not str(row.get(instrument_id, "")).strip() for instrument_id in configured_instruments):
            continue
        eligible_rows.append(row)

    treatment_values = [
        value
        for row in eligible_rows
        if (value := _coerce_float(row.get(treatment_id, ""))) is not None
    ]
    diagnostics: dict[str, Any] = {
        "job_id": job_id,
        "rows_analyzed": len(eligible_rows),
        "treatment_id": treatment_id,
        "treatment_summary": _summarize_float_series(treatment_values),
        "instrument_diagnostics": [],
    }

    for instrument_id in configured_instruments:
        component = INSTRUMENT_COMPONENTS.get(instrument_id, {})
        shock_id = str(component.get("shock", "")).strip()
        state_id = str(component.get("state", "")).strip()
        triples: list[tuple[float, float, float | None, float | None]] = []
        for row in eligible_rows:
            instrument_value = _coerce_float(row.get(instrument_id, ""))
            treatment_value = _coerce_float(row.get(treatment_id, ""))
            if instrument_value is None or treatment_value is None:
                continue
            shock_value = _coerce_float(row.get(shock_id, "")) if shock_id else None
            state_value = _coerce_float(row.get(state_id, "")) if state_id else None
            triples.append((instrument_value, treatment_value, shock_value, state_value))

        instrument_values = [item[0] for item in triples]
        treatment_pair_values = [item[1] for item in triples]
        shock_values = [item[2] for item in triples if item[2] is not None]
        state_values = [item[3] for item in triples if item[3] is not None]
        shock_pairs = [(item[0], item[2]) for item in triples if item[2] is not None]
        state_pairs = [(item[0], item[3]) for item in triples if item[3] is not None]
        diagnostics["instrument_diagnostics"].append(
            {
                "instrument_id": instrument_id,
                "shock_id": shock_id,
                "state_id": state_id,
                "summary": _summarize_float_series(instrument_values),
                "shock_summary": _summarize_float_series(shock_values),
                "state_summary": _summarize_float_series(state_values),
                "corr_instrument_treatment": _correlation(instrument_values, treatment_pair_values),
                "corr_instrument_shock": _correlation(
                    [item[0] for item in shock_pairs],
                    [item[1] for item in shock_pairs],
                ),
                "corr_instrument_state": _correlation(
                    [item[0] for item in state_pairs],
                    [item[1] for item in state_pairs],
                ),
            }
        )
    return diagnostics


def _aggregate_quarterly_fred(raw_dir: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    if not raw_dir.exists():
        return result
    for path in sorted(raw_dir.glob("*.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        method = FRED_QUARTERLY_AGGREGATION.get(path.stem, "last")
        quarter_map: dict[str, tuple[str, str]] = {}
        quarter_values: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            date_text = str(row.get("date", "")).strip()
            value = str(row.get("value", "")).strip()
            if not date_text or not value or value == ".":
                continue
            quarter = _quarter_from_date(date_text)
            if method == "mean":
                try:
                    quarter_values[quarter].append(float(value))
                except ValueError:
                    continue
                continue
            current = quarter_map.get(quarter)
            if current is None or date_text > current[0]:
                quarter_map[quarter] = (date_text, value)
        if method == "mean":
            result[path.stem] = {
                quarter: _stable_float_text(sum(values) / len(values))
                for quarter, values in quarter_values.items()
                if values
            }
        else:
            result[path.stem] = {quarter: value for quarter, (_, value) in quarter_map.items()}
    return result


def _compute_qoq_level_changes(levels_by_quarter: dict[str, str]) -> dict[str, str]:
    sorted_quarters = sorted(levels_by_quarter.keys(), key=_quarter_sort_key)
    output: dict[str, str] = {}
    previous_level: float | None = None
    for quarter in sorted_quarters:
        raw_value = str(levels_by_quarter.get(quarter, "")).strip()
        if not raw_value:
            previous_level = None
            continue
        try:
            current_level = float(raw_value)
        except ValueError:
            previous_level = None
            continue
        if previous_level is not None:
            output[quarter] = _stable_float_text(current_level - previous_level)
        previous_level = current_level
    return output


def _compute_qoq_logdiff_annualized(levels_by_quarter: dict[str, str]) -> dict[str, str]:
    sorted_quarters = sorted(levels_by_quarter.keys(), key=_quarter_sort_key)
    output: dict[str, str] = {}
    previous_level: float | None = None
    for quarter in sorted_quarters:
        raw_value = str(levels_by_quarter.get(quarter, "")).strip()
        if not raw_value:
            previous_level = None
            continue
        try:
            current_level = float(raw_value)
        except ValueError:
            previous_level = None
            continue
        if previous_level is not None and current_level > 0 and previous_level > 0:
            output[quarter] = _stable_float_text((math.log(current_level) - math.log(previous_level)) * 400.0)
        previous_level = current_level
    return output


def _compute_scaled_spread(
    left_by_quarter: dict[str, str],
    right_by_quarter: dict[str, str],
    *,
    scale: float,
) -> dict[str, str]:
    output: dict[str, str] = {}
    common_quarters = set(left_by_quarter).intersection(right_by_quarter)
    for quarter in sorted(common_quarters, key=_quarter_sort_key):
        left_value = str(left_by_quarter.get(quarter, "")).strip()
        right_value = str(right_by_quarter.get(quarter, "")).strip()
        if not left_value or not right_value:
            continue
        try:
            output[quarter] = _stable_float_text((float(left_value) - float(right_value)) * scale)
        except ValueError:
            continue
    return output


def _compute_difference_series(
    left_by_quarter: dict[str, str],
    right_by_quarter: dict[str, str],
) -> dict[str, str]:
    output: dict[str, str] = {}
    for quarter in sorted(set(left_by_quarter).intersection(right_by_quarter), key=_quarter_sort_key):
        left_value = str(left_by_quarter.get(quarter, "")).strip()
        right_value = str(right_by_quarter.get(quarter, "")).strip()
        if not left_value or not right_value:
            continue
        try:
            output[quarter] = _stable_float_text(float(left_value) - float(right_value))
        except ValueError:
            continue
    return output


def _compute_sum_series(components: list[dict[str, str]]) -> dict[str, str]:
    quarter_pool: set[str] = set()
    for component in components:
        quarter_pool.update(component.keys())
    output: dict[str, str] = {}
    for quarter in sorted(quarter_pool, key=_quarter_sort_key):
        values: list[float] = []
        for component in components:
            raw_value = str(component.get(quarter, "")).strip()
            if not raw_value:
                values = []
                break
            try:
                values.append(float(raw_value))
            except ValueError:
                values = []
                break
        if values:
            output[quarter] = _stable_float_text(sum(values))
    return output


def _overlay_missing_quarters(primary: dict[str, str], fallback: dict[str, str]) -> dict[str, str]:
    output = dict(primary)
    for quarter, value in fallback.items():
        if quarter not in output or not str(output.get(quarter, "")).strip():
            output[quarter] = value
    return output


def _set_row_linear_combo(
    row: dict[str, str],
    *,
    output_name: str,
    components: tuple[tuple[str, float], ...],
) -> None:
    values: list[float] = []
    available_values: list[str] = []
    for component_name, weight in components:
        component_value = _coerce_float(row.get(component_name, ""))
        if component_value is None:
            if output_name == "public_liquidity_proxy_block_qoq" and component_name == "on_rrp_balance_qoq":
                component_value = 0.0
            else:
                row[output_name] = ""
                row[f"{output_name}__available_at"] = ""
                row[f"{output_name}__source_repo"] = "derived"
                return
        values.append(weight * component_value)
        available_values.append(row.get(f"{component_name}__available_at", ""))
    row[output_name] = _stable_float_text(sum(values))
    row[f"{output_name}__available_at"] = _max_available_iso(*available_values)
    row[f"{output_name}__source_repo"] = "derived"


def _set_simple_derived_series(
    row: dict[str, str],
    *,
    output_name: str,
    value: float | None,
    available_at: str,
    source_repo: str = "derived",
) -> None:
    row[output_name] = "" if value is None else _stable_float_text(value)
    row[f"{output_name}__available_at"] = available_at if value is not None else ""
    row[f"{output_name}__source_repo"] = source_repo


def _set_mmf_rrp_plumbing_adjustments(row: dict[str, str]) -> None:
    plumbing_absorption = _coerce_float(row.get("mmf_rrp_adjustment_prop_qoq", ""))
    plumbing_available_at = row.get("mmf_rrp_adjustment_prop_qoq__available_at", "")

    if plumbing_absorption is None:
        mmf_reallocation = _coerce_float(row.get("mmf_treasury_bills_reallocation_qoq", ""))
        on_rrp_balance_qoq = _coerce_float(row.get("on_rrp_balance_qoq", ""))
        if mmf_reallocation is None or on_rrp_balance_qoq is None:
            plumbing_absorption = None
        else:
            # Legacy fallback for older bundles: aggregate bills/RRP min rule.
            # Current bundles should use the fund-month proportional adjustment
            # from tdcest via mmf_rrp_adjustment_prop_qoq.
            mmf_bill_absorption = max(0.0, -mmf_reallocation)
            on_rrp_runoff = max(0.0, -on_rrp_balance_qoq * 1000.0)
            plumbing_absorption = min(mmf_bill_absorption, on_rrp_runoff)
        plumbing_available_at = _max_available_iso(
            row.get("mmf_treasury_bills_reallocation_qoq__available_at", ""),
            row.get("on_rrp_balance_qoq__available_at", ""),
        )
    _set_simple_derived_series(
        row,
        output_name="mmf_on_rrp_plumbing_absorption_qoq",
        value=plumbing_absorption,
        available_at=plumbing_available_at,
    )

    for output_name, base_tdc_name in (
        ("tdc_tier1_mmf_rrp_plumbing_adjusted_qoq", "tdc_bank_only_qoq"),
        ("tdc_tier2_mmf_rrp_plumbing_adjusted_qoq", "tdc_tier2_interest_corrected_bank_only_ru_flow"),
        ("tdc_du_residual_mmf_rrp_plumbing_adjusted_qoq", "tdc_du_residual_proxy_full_cu_ru"),
        ("tdc_no_row_mmf_rrp_plumbing_adjusted_qoq", "tdc_domestic_bank_only_qoq"),
        ("tdc_no_toc_mmf_rrp_plumbing_adjusted_qoq", "tdc_no_toc_bank_only_qoq"),
        ("tdc_no_toc_no_row_mmf_rrp_plumbing_adjusted_qoq", "tdc_no_toc_no_row_bank_only_qoq"),
        ("tdc_core_mmf_rrp_plumbing_adjusted_qoq", "tdc_core_deposit_proximate_bank_only_qoq"),
        ("tdc_us_chartered_mmf_rrp_plumbing_adjusted_qoq", "tdc_us_chartered_bank_only_qoq"),
        ("tdc_no_foreign_bank_sectors_mmf_rrp_plumbing_adjusted_qoq", "tdc_no_foreign_bank_sectors_qoq"),
    ):
        base_tdc = _coerce_float(row.get(base_tdc_name, ""))
        if base_tdc is None or plumbing_absorption is None:
            adjusted_tdc = None
            adjusted_available_at = ""
        else:
            adjusted_tdc = base_tdc + plumbing_absorption
            adjusted_available_at = _max_available_iso(
                row.get(f"{base_tdc_name}__available_at", ""),
                plumbing_available_at,
            )
        _set_simple_derived_series(
            row,
            output_name=output_name,
            value=adjusted_tdc,
            available_at=adjusted_available_at,
        )


def _set_domestic_nonbank_adjusted_residuals(row: dict[str, str]) -> None:
    domestic_deposits = _coerce_float(row.get("domestic_nonbank_deposits_qoq", ""))
    for output_name, treatment_name in (
        (
            "domestic_nonbank_other_component_tier1_qoq",
            "tdc_bank_only_qoq",
        ),
        (
            "domestic_nonbank_other_component_tier2_bank_only_qoq",
            "tdc_tier2_interest_corrected_bank_only_ru_flow",
        ),
        (
            "domestic_nonbank_other_component_tier2_mmf_rrp_prop_bank_only_qoq",
            "tdc_tier2_mmf_rrp_prop_bank_only_qoq",
        ),
        (
            "domestic_nonbank_other_component_tier2_mmf_rrp_lb_bank_only_qoq",
            "tdc_tier2_mmf_rrp_lb_bank_only_qoq",
        ),
        (
            "domestic_nonbank_other_component_tier2_mmf_rrp_ub_bank_only_qoq",
            "tdc_tier2_mmf_rrp_ub_bank_only_qoq",
        ),
        (
            "domestic_nonbank_other_component_tier2_mmf_rrp_prop_di_np_cu_qoq",
            "tdc_tier2_mmf_rrp_prop_di_np_cu_qoq",
        ),
        (
            "domestic_nonbank_other_component_tier2_treasury_interest_robust_mmf_rrp_prop_bank_only_qoq",
            "tdc_tier2_treasury_interest_robust_mmf_rrp_prop_bank_only_qoq",
        ),
        (
            "domestic_nonbank_other_component_tier2_canonical_di_mmf_rrp_prop_qoq",
            "tdc_tier2_canonical_di_mmf_rrp_prop_qoq",
        ),
        (
            "domestic_nonbank_other_component_tier2_canonical_no_toc_di_mmf_rrp_prop_qoq",
            "tdc_tier2_canonical_no_toc_di_mmf_rrp_prop_qoq",
        ),
        (
            "domestic_nonbank_other_component_du_residual_qoq",
            "tdc_du_residual_proxy_full_cu_ru",
        ),
        (
            "domestic_nonbank_other_component_no_row_qoq",
            "tdc_domestic_bank_only_qoq",
        ),
        (
            "domestic_nonbank_other_component_no_toc_qoq",
            "tdc_no_toc_bank_only_qoq",
        ),
        (
            "domestic_nonbank_other_component_no_toc_no_row_qoq",
            "tdc_no_toc_no_row_bank_only_qoq",
        ),
        (
            "domestic_nonbank_other_component_tier1_mmf_rrp_plumbing_adjusted_qoq",
            "tdc_tier1_mmf_rrp_plumbing_adjusted_qoq",
        ),
        (
            "domestic_nonbank_other_component_tier2_mmf_rrp_plumbing_adjusted_qoq",
            "tdc_tier2_mmf_rrp_plumbing_adjusted_qoq",
        ),
        (
            "domestic_nonbank_other_component_du_residual_mmf_rrp_plumbing_adjusted_qoq",
            "tdc_du_residual_mmf_rrp_plumbing_adjusted_qoq",
        ),
        (
            "domestic_nonbank_other_component_no_row_mmf_rrp_plumbing_adjusted_qoq",
            "tdc_no_row_mmf_rrp_plumbing_adjusted_qoq",
        ),
        (
            "domestic_nonbank_other_component_no_toc_mmf_rrp_plumbing_adjusted_qoq",
            "tdc_no_toc_mmf_rrp_plumbing_adjusted_qoq",
        ),
        (
            "domestic_nonbank_other_component_no_toc_no_row_mmf_rrp_plumbing_adjusted_qoq",
            "tdc_no_toc_no_row_mmf_rrp_plumbing_adjusted_qoq",
        ),
        (
            "domestic_nonbank_other_component_core_mmf_rrp_plumbing_adjusted_qoq",
            "tdc_core_mmf_rrp_plumbing_adjusted_qoq",
        ),
        (
            "domestic_nonbank_other_component_us_chartered_mmf_rrp_plumbing_adjusted_qoq",
            "tdc_us_chartered_mmf_rrp_plumbing_adjusted_qoq",
        ),
        (
            "domestic_nonbank_other_component_no_foreign_bank_sectors_mmf_rrp_plumbing_adjusted_qoq",
            "tdc_no_foreign_bank_sectors_mmf_rrp_plumbing_adjusted_qoq",
        ),
    ):
        treatment_value = _coerce_float(row.get(treatment_name, ""))
        if domestic_deposits is None or treatment_value is None:
            residual = None
            residual_available_at = ""
        else:
            residual = domestic_deposits - treatment_value
            residual_available_at = _max_available_iso(
                row.get("domestic_nonbank_deposits_qoq__available_at", ""),
                row.get(f"{treatment_name}__available_at", ""),
            )
        _set_simple_derived_series(
            row,
            output_name=output_name,
            value=residual,
            available_at=residual_available_at,
        )


def _derive_quarterly_low_liquidity_state(
    *,
    reserves_by_quarter: dict[str, str],
    on_rrp_by_quarter: dict[str, str],
    quantile: float = 0.35,
) -> tuple[dict[str, str], dict[str, str]]:
    system_liquidity: dict[str, float] = {}
    for quarter in sorted(set(reserves_by_quarter).intersection(on_rrp_by_quarter), key=_quarter_sort_key):
        reserve_value = str(reserves_by_quarter.get(quarter, "")).strip()
        on_rrp_value = str(on_rrp_by_quarter.get(quarter, "")).strip()
        if not reserve_value or not on_rrp_value:
            continue
        try:
            system_liquidity[quarter] = float(reserve_value) + float(on_rrp_value)
        except ValueError:
            continue

    if not system_liquidity:
        return {}, {}

    ordered = sorted(system_liquidity.items(), key=lambda item: _quarter_sort_key(item[0]))
    values = [value for _, value in ordered]
    threshold_index = max(min(int((len(values) - 1) * quantile), len(values) - 1), 0)
    threshold = sorted(values)[threshold_index]
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    std_value = variance ** 0.5

    low_liquidity_current: dict[str, str] = {}
    tightness_current: dict[str, str] = {}
    for quarter, value in ordered:
        low_liquidity_current[quarter] = "1" if value <= threshold else "0"
        tightness_current[quarter] = _stable_float_text((-(value - mean_value) / std_value) if std_value else 0.0, digits=4)

    low_liquidity_lagged: dict[str, str] = {}
    tightness_lagged: dict[str, str] = {}
    for quarter, _ in ordered:
        previous = _previous_quarter(quarter)
        if previous is None:
            continue
        if previous in low_liquidity_current:
            low_liquidity_lagged[quarter] = low_liquidity_current[previous]
        if previous in tightness_current:
            tightness_lagged[quarter] = tightness_current[previous]
    return low_liquidity_lagged, tightness_lagged


def _derive_quarterly_on_rrp_drain_state(
    *,
    reserves_by_quarter: dict[str, str],
    on_rrp_by_quarter: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    share_current: dict[str, float] = {}
    for quarter in sorted(set(reserves_by_quarter).intersection(on_rrp_by_quarter), key=_quarter_sort_key):
        reserve_value = str(reserves_by_quarter.get(quarter, "")).strip()
        on_rrp_value = str(on_rrp_by_quarter.get(quarter, "")).strip()
        if not reserve_value or not on_rrp_value:
            continue
        try:
            reserve_level = float(reserve_value)
            on_rrp_level = float(on_rrp_value)
        except ValueError:
            continue
        denominator = reserve_level + on_rrp_level
        if denominator <= 0:
            continue
        share_current[quarter] = on_rrp_level / denominator

    if not share_current:
        return {}, {}

    values = list(share_current.values())
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    std_value = variance ** 0.5

    drain_current: dict[str, str] = {}
    share_text: dict[str, str] = {}
    for quarter, share_value in sorted(share_current.items(), key=lambda item: _quarter_sort_key(item[0])):
        share_text[quarter] = _stable_float_text(share_value, digits=6)
        drain_current[quarter] = _stable_float_text(
            (mean_value - share_value) / std_value if std_value else 0.0,
            digits=4,
        )

    drain_lagged: dict[str, str] = {}
    for quarter in sorted(share_current, key=_quarter_sort_key):
        previous = _previous_quarter(quarter)
        if previous is not None and previous in drain_current:
            drain_lagged[quarter] = drain_current[previous]
    return drain_lagged, share_text


def _load_standardized_series(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return _read_csv(path)


def _resolve_treatment_column(job: dict[str, Any]) -> str:
    treatment_id = str(job.get("treatment_id", "")).strip()
    if not treatment_id:
        return "tdc_bank_only_qoq"
    return TREATMENT_COLUMN_MAP.get(treatment_id, treatment_id)


def build_quarterly_design(paths: ProjectPaths, *, job_id: str) -> DesignBuildResult:
    jobs = _load_jobs(paths.config / "dass_job_blueprint.yaml")
    if job_id not in jobs:
        raise KeyError(f"Unknown job_id: {job_id}")
    job = jobs[job_id]

    tdcest_rows = _load_standardized_series(paths.bundles / "tdcest" / "standardized_series.csv")
    qrawatch_rows = _load_standardized_series(paths.bundles / "qrawatch" / "standardized_series.csv")
    coordwatch_rows = _load_standardized_series(paths.bundles / "coordwatch" / "standardized_series.csv")
    tsyparty_rows = _load_standardized_series(paths.bundles / "tsyparty" / "standardized_series.csv")
    tdcpass_rows = _load_standardized_series(paths.bundles / "tdcpass" / "standardized_series.csv")
    wamest_rows = _load_standardized_series(paths.bundles / "wamest" / "standardized_series.csv")
    slrwatch_rows = _load_standardized_series(paths.bundles / "slrwatch" / "standardized_series.csv")
    accounting_rows = _load_standardized_series(paths.bundles / "accounting" / "standardized_series.csv")
    fred_quarterly = _aggregate_quarterly_fred(paths.raw_fred)
    derived_fred_qoq = {
        output_name: _compute_qoq_level_changes(fred_quarterly.get(series_id, {}))
        for output_name, series_id in RAW_FRED_LEVEL_CHANGE_SERIES.items()
    }
    for output_name, fallback_name in RAW_FRED_QOQ_FALLBACK_SERIES.items():
        derived_fred_qoq[output_name] = _overlay_missing_quarters(
            derived_fred_qoq.get(output_name, {}),
            derived_fred_qoq.get(fallback_name, {}),
        )
    derived_fred_logdiff_ann = {
        output_name: _compute_qoq_logdiff_annualized(fred_quarterly.get(series_id, {}))
        for output_name, series_id in RAW_FRED_LOG_DIFF_ANNUALIZED_SERIES.items()
    }
    direct_fred_outputs = {
        output_name: fred_quarterly.get(series_id, {})
        for output_name, series_id in RAW_FRED_DIRECT_SERIES.items()
    }
    derived_fred_aggregate_changes = {
        output_name: _compute_qoq_level_changes(fred_quarterly.get(series_id, {}))
        for output_name, series_id in RAW_FRED_LEVEL_CHANGE_FROM_AGGREGATES.items()
    }
    derived_fred_spreads = {
        output_name: _compute_scaled_spread(
            fred_quarterly.get(left_series, {}),
            fred_quarterly.get(right_series, {}),
            scale=scale,
        )
        for output_name, (left_series, right_series, scale) in RAW_FRED_SPREAD_SERIES.items()
    }
    derived_fred_qoq_sums = {
        output_name: _compute_sum_series([derived_fred_qoq.get(component, {}) for component in component_ids])
        for output_name, component_ids in RAW_FRED_QOQ_SUM_SERIES.items()
    }
    derived_fred_qoq.update(derived_fred_qoq_sums)
    derived_fred_differences = {
        output_name: _compute_difference_series(
            derived_fred_qoq.get(left_series, {}),
            derived_fred_qoq.get(right_series, {}),
        )
        for output_name, (left_series, right_series) in RAW_FRED_DIFFERENCE_SERIES.items()
    }
    derived_fred_sums = {
        output_name: _compute_sum_series([direct_fred_outputs.get(component, {}) for component in component_ids])
        for output_name, component_ids in RAW_FRED_SUM_SERIES.items()
    }
    derived_states = {}
    derived_state_scores = {}
    low_liquidity_state, liquidity_tightness_state = _derive_quarterly_low_liquidity_state(
        reserves_by_quarter=fred_quarterly.get("WRESBAL", {}),
        on_rrp_by_quarter=fred_quarterly.get("RRPONTSYD", {}),
    )
    on_rrp_drain_state, on_rrp_share = _derive_quarterly_on_rrp_drain_state(
        reserves_by_quarter=fred_quarterly.get("WRESBAL", {}),
        on_rrp_by_quarter=fred_quarterly.get("RRPONTSYD", {}),
    )
    derived_states["coord_low_reserve_state_l1"] = low_liquidity_state
    derived_states["coord_on_rrp_drain_state_l1"] = on_rrp_drain_state
    derived_state_scores["coord_liquidity_tightness_q_z_l1"] = liquidity_tightness_state
    derived_state_scores["coord_on_rrp_share_q"] = on_rrp_share

    by_series_quarter: dict[tuple[str, str], dict[str, str]] = {}
    for row in [*tdcest_rows, *qrawatch_rows, *coordwatch_rows, *tsyparty_rows, *tdcpass_rows, *wamest_rows, *slrwatch_rows, *accounting_rows]:
        series_id = row["series_id"]
        quarter = _quarter_from_date(row["period_end"])
        key = (series_id, quarter)
        existing = by_series_quarter.get(key)
        if existing is None:
            by_series_quarter[key] = row
            continue
        existing_time = _parse_available_at(existing.get("available_at", ""))
        current_time = _parse_available_at(row.get("available_at", ""))
        if current_time and (existing_time is None or current_time < existing_time):
            by_series_quarter[key] = row

    quarter_pool: set[str] = set()
    for _, quarter in by_series_quarter.keys():
        quarter_pool.add(quarter)
    for series_quarters in fred_quarterly.values():
        quarter_pool.update(series_quarters.keys())

    required_outputs = BASELINE_REQUIRED_OUTPUTS.get(job_id, [str(item) for item in job.get("outcomes", [])])
    configured_instruments = [str(item) for item in job.get("instruments", [])]
    required_state_ids = [str(job["state_id"])] if job.get("state_id") else []
    if job.get("controls", {}).get("include_main_state"):
        for instrument_id in configured_instruments:
            state_id = INSTRUMENT_COMPONENTS.get(instrument_id, {}).get("state")
            if state_id and state_id not in required_state_ids:
                required_state_ids.append(str(state_id))
    explicit_controls = [str(item) for item in job.get("controls_explicit", [])]
    treatment_column = _resolve_treatment_column(job)
    requested_series = {
        treatment_column,
        *required_outputs,
        *required_state_ids,
        *configured_instruments,
        *explicit_controls,
    }
    for instrument_id in configured_instruments:
        component = INSTRUMENT_COMPONENTS.get(instrument_id, {})
        for key in ("shock", "state"):
            value = str(component.get(key, "")).strip()
            if value:
                requested_series.add(value)

    bundle_rows: list[dict[str, str]] = []
    for quarter in sorted(quarter_pool):
        row: dict[str, str] = {"quarter": quarter}
        earliest_cutoff: datetime | None = None

        for output_name, (source_repo, series_id) in BASELINE_SERIES_MAP.items():
            standardized = by_series_quarter.get((series_id, quarter))
            if standardized:
                value = standardized.get("value", "")
                scale = BASELINE_SERIES_SCALE.get(output_name)
                if value and scale is not None:
                    value = _stable_float_text(float(value) * scale)
                row[output_name] = value
                row[f"{output_name}__available_at"] = standardized.get("available_at", "")
                row[f"{output_name}__source_repo"] = source_repo
                available = _parse_available_at(standardized.get("available_at", ""))
                if available and (earliest_cutoff is None or available < earliest_cutoff):
                    earliest_cutoff = available
            else:
                row[output_name] = ""
                row[f"{output_name}__available_at"] = ""
                row[f"{output_name}__source_repo"] = source_repo

        row["qra_maturity_tilt_flow"] = row.get("qra_ati_baseline_bn", "")
        row["qra_maturity_tilt_flow__available_at"] = row.get("qra_ati_baseline_bn__available_at", "")
        row["qra_maturity_tilt_flow__source_repo"] = (
            "qrawatch_proxy" if row.get("qra_maturity_tilt_flow", "") else ""
        )

        for fred_series in [
            "GDP",
            "M2SL",
            "TOTRESNS",
            "FEDFUNDS",
            "SOFR",
            "BAA",
            "AAA",
            "BAMLC0A0CM",
            "BAMLH0A0HYM2",
            "WRESBAL",
            "WDFOL",
            "WALCL",
            "TREAST",
            "TGCRRATE",
            "RRPONTSYAWARD",
            "RRPONTSYD",
            "CPIAUCSL",
            "CPILFESL",
            "PCEPILFE",
            "DTWEXBGS",
            "TOTBKCR",
            "TASACBW027SBOG",
            "CLSACBW027SBOG",
            "RELACBW027SBOG",
            "BOGZ1FL264035005Q",
        ]:
            value = fred_quarterly.get(fred_series, {}).get(quarter, "")
            row[fred_series] = value

        for output_name, qoq_values in derived_fred_qoq.items():
            value = qoq_values.get(quarter, "")
            row[output_name] = value
            row[f"{output_name}__available_at"] = _conservative_quarterly_available_at(
                quarter,
                lag_days=RAW_FRED_LAG_DAYS.get(output_name, 90),
            ) if value else ""
            row[f"{output_name}__source_repo"] = "fred"

        for output_name, logdiff_values in derived_fred_logdiff_ann.items():
            value = logdiff_values.get(quarter, "")
            row[output_name] = value
            row[f"{output_name}__available_at"] = _conservative_quarterly_available_at(
                quarter,
                lag_days=RAW_FRED_LAG_DAYS.get(output_name, 30),
            ) if value else ""
            row[f"{output_name}__source_repo"] = "fred"

        for output_name, change_values in derived_fred_aggregate_changes.items():
            value = change_values.get(quarter, "")
            row[output_name] = value
            row[f"{output_name}__available_at"] = _conservative_quarterly_available_at(
                quarter,
                lag_days=RAW_FRED_LAG_DAYS.get(output_name, 30),
            ) if value else ""
            row[f"{output_name}__source_repo"] = "fred"

        for output_name, direct_values in direct_fred_outputs.items():
            value = direct_values.get(quarter, "")
            row[output_name] = value
            row[f"{output_name}__available_at"] = _conservative_quarterly_available_at(
                quarter,
                lag_days=RAW_FRED_LAG_DAYS.get(output_name, 30),
            ) if value else ""
            row[f"{output_name}__source_repo"] = "fred"

        for output_name, spread_values in derived_fred_spreads.items():
            value = spread_values.get(quarter, "")
            row[output_name] = value
            row[f"{output_name}__available_at"] = _conservative_quarterly_available_at(
                quarter,
                lag_days=RAW_FRED_LAG_DAYS.get(output_name, 30),
            ) if value else ""
            row[f"{output_name}__source_repo"] = "fred"

        for output_name, difference_values in derived_fred_differences.items():
            value = difference_values.get(quarter, "")
            row[output_name] = value
            row[f"{output_name}__available_at"] = _conservative_quarterly_available_at(
                quarter,
                lag_days=RAW_FRED_LAG_DAYS.get(output_name, 30),
            ) if value else ""
            row[f"{output_name}__source_repo"] = "derived"

        for output_name, sum_values in derived_fred_sums.items():
            value = sum_values.get(quarter, "")
            row[output_name] = value
            row[f"{output_name}__available_at"] = _conservative_quarterly_available_at(
                quarter,
                lag_days=RAW_FRED_LAG_DAYS.get(output_name, 90),
            ) if value else ""
            row[f"{output_name}__source_repo"] = "derived"

        for output_name, state_values in derived_states.items():
            imported = by_series_quarter.get((output_name, quarter))
            value = imported.get("value", "") if imported else state_values.get(quarter, "")
            row[output_name] = value
            if imported:
                row[f"{output_name}__available_at"] = imported.get("available_at", "")
                row[f"{output_name}__source_repo"] = imported.get("source_repo", "coordwatch")
            else:
                previous = _previous_quarter(quarter)
                row[f"{output_name}__available_at"] = (
                    _conservative_quarterly_available_at(
                        previous,
                        lag_days=RAW_FRED_LAG_DAYS.get(output_name, 14),
                    )
                    if value and previous
                    else ""
                )
                row[f"{output_name}__source_repo"] = "derived"

        for output_name, state_values in derived_state_scores.items():
            imported = by_series_quarter.get((output_name, quarter))
            value = imported.get("value", "") if imported else state_values.get(quarter, "")
            row[output_name] = value
            if imported:
                row[f"{output_name}__available_at"] = imported.get("available_at", "")
                row[f"{output_name}__source_repo"] = imported.get("source_repo", "coordwatch")
            else:
                previous = _previous_quarter(quarter)
                row[f"{output_name}__available_at"] = (
                    _conservative_quarterly_available_at(
                        previous,
                        lag_days=RAW_FRED_LAG_DAYS.get(output_name, 14),
                    )
                    if value and previous
                    else ""
                )
                row[f"{output_name}__source_repo"] = "derived"

        matched_total = row.get("matched_total_deposits", "")
        tdc_bank_only = row.get("tdc_bank_only_qoq", "")
        if matched_total and tdc_bank_only:
            row["other_component_qoq"] = _stable_float_text(float(matched_total) - float(tdc_bank_only))
            matched_available = _parse_available_at(row.get("matched_total_deposits__available_at", ""))
            tdc_available = _parse_available_at(row.get("tdc_bank_only_qoq__available_at", ""))
            derived_available = max(
                [item for item in [matched_available, tdc_available] if item is not None],
                default=None,
            )
            row["other_component_qoq__available_at"] = derived_available.isoformat() if derived_available else ""
            row["other_component_qoq__source_repo"] = "derived"
        else:
            row["other_component_qoq"] = ""
            row["other_component_qoq__available_at"] = ""
            row["other_component_qoq__source_repo"] = "derived"

        _set_mmf_rrp_plumbing_adjustments(row)

        for output_name, components in ROW_LINEAR_COMBO_SERIES.items():
            _set_row_linear_combo(
                row,
                output_name=output_name,
                components=components,
            )

        for instrument_id, component in INSTRUMENT_COMPONENTS.items():
            shock_id = str(component.get("shock", "")).strip()
            state_id = str(component.get("state", "")).strip()
            shock_value = row.get(shock_id, "")
            state_value = row.get(state_id, "")
            if shock_value and state_value:
                row[instrument_id] = _stable_float_text(float(shock_value) * float(state_value))
                shock_available = _parse_available_at(row.get(f"{shock_id}__available_at", ""))
                state_available = _parse_available_at(row.get(f"{state_id}__available_at", ""))
                derived_available = max(
                    [item for item in [shock_available, state_available] if item is not None],
                    default=None,
                )
                row[f"{instrument_id}__available_at"] = derived_available.isoformat() if derived_available else ""
                row[f"{instrument_id}__source_repo"] = "derived"
            else:
                row[instrument_id] = ""
                row[f"{instrument_id}__available_at"] = ""
                row[f"{instrument_id}__source_repo"] = "derived"

        for series_id in sorted(requested_series):
            if row.get(series_id, ""):
                continue
            standardized = by_series_quarter.get((series_id, quarter))
            if standardized is None:
                continue
            row[series_id] = standardized.get("value", "")
            row[f"{series_id}__available_at"] = standardized.get("available_at", "")
            row[f"{series_id}__source_repo"] = standardized.get("source_repo", "")
            available = _parse_available_at(standardized.get("available_at", ""))
            if available and (earliest_cutoff is None or available < earliest_cutoff):
                earliest_cutoff = available

        matched_total = row.get("matched_total_deposits", "")
        for output_name, treatment_series in ALTERNATIVE_OTHER_COMPONENT_SERIES.items():
            alternative_tdc = row.get(treatment_series, "")
            if matched_total and alternative_tdc:
                row[output_name] = _stable_float_text(float(matched_total) - float(alternative_tdc))
                matched_available = _parse_available_at(row.get("matched_total_deposits__available_at", ""))
                treatment_available = _parse_available_at(row.get(f"{treatment_series}__available_at", ""))
                derived_available = max(
                    [item for item in [matched_available, treatment_available] if item is not None],
                    default=None,
                )
                row[f"{output_name}__available_at"] = derived_available.isoformat() if derived_available else ""
                row[f"{output_name}__source_repo"] = "derived"
            else:
                row[output_name] = ""
                row[f"{output_name}__available_at"] = ""
                row[f"{output_name}__source_repo"] = "derived"

        _set_domestic_nonbank_adjusted_residuals(row)

        for output_name, components in ROW_LINEAR_COMBO_SERIES.items():
            if output_name.startswith("accounting_identity_gap_tier"):
                _set_row_linear_combo(
                    row,
                    output_name=output_name,
                    components=components,
                )

        gdp_value = _coerce_float(row.get("GDP", ""))
        gdp_available_at = _conservative_quarterly_available_at(quarter, lag_days=30) if gdp_value else ""
        for output_name, base_output in PCT_GDP_SOURCE_OUTCOME_MAP.items():
            base_value = _coerce_float(row.get(base_output, ""))
            if base_value is None or gdp_value is None or gdp_value == 0:
                row[output_name] = ""
                row[f"{output_name}__available_at"] = ""
                row[f"{output_name}__source_repo"] = "derived"
                continue
            row[output_name] = _stable_float_text((100.0 * base_value) / gdp_value)
            row[f"{output_name}__available_at"] = _max_available_iso(
                row.get(f"{base_output}__available_at", ""),
                gdp_available_at,
            )
            row[f"{output_name}__source_repo"] = "derived"

        row["cutoff_timestamp"] = earliest_cutoff.isoformat() if earliest_cutoff else ""
        bundle_rows.append(row)

    bundle_dir = paths.bundles / "designs"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = bundle_dir / f"{job_id}__quarterly_bundle.csv"
    with bundle_path.open("w", encoding="utf-8", newline="") as handle:
        if bundle_rows:
            fieldnames = ["quarter"]
            seen = {"quarter"}
            for row in bundle_rows:
                for key in row.keys():
                    if key not in seen:
                        fieldnames.append(key)
                        seen.add(key)
        else:
            fieldnames = ["quarter"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bundle_rows)

    def _row_has_outcome(row: dict[str, str], outcome: str) -> bool:
        aliases = OUTCOME_ALIASES.get(outcome, [outcome])
        return any(str(row.get(alias, "")).strip() != "" for alias in aliases)

    def _row_has_state(row: dict[str, str], state_id: str) -> bool:
        return str(row.get(state_id, "")).strip() != ""

    def _row_has_instrument(row: dict[str, str], instrument_id: str) -> bool:
        return str(row.get(instrument_id, "")).strip() != ""

    missing_series = [series for series in required_outputs if all(not _row_has_outcome(row, series) for row in bundle_rows)]
    missing_state_ids = [state_id for state_id in required_state_ids if all(not _row_has_state(row, state_id) for row in bundle_rows)]
    missing_instrument_ids = [
        instrument_id for instrument_id in configured_instruments if all(not _row_has_instrument(row, instrument_id) for row in bundle_rows)
    ]
    if treatment_column == "tdc_bank_only_qoq":
        usable_rows = sum(
            1
            for row in bundle_rows
            if row.get("tdc_bank_only_qoq", "") and row.get("qra_ati_baseline_bn", "")
        )
    else:
        usable_rows = sum(1 for row in bundle_rows if str(row.get(treatment_column, "")).strip())

    design_instrument_ids = configured_instruments or ["qra_ati_baseline_bn", "qra_net_bills_bn", "qra_bill_share"]
    if explicit_controls:
        control_ids = explicit_controls[:]
    else:
        control_ids = ["GDP", "gdp_deflator", "FEDFUNDS", "TOTRESNS"]
        if job.get("controls", {}).get("include_main_qra_shock") and "qra_maturity_tilt_flow" not in control_ids:
            control_ids.append("qra_maturity_tilt_flow")
        if job.get("controls", {}).get("include_main_state"):
            for instrument_id in configured_instruments:
                state_id = INSTRUMENT_COMPONENTS.get(instrument_id, {}).get("state")
                if state_id and state_id not in control_ids:
                    control_ids.append(str(state_id))

    design_manifest = {
        "job_id": job_id,
        "generated_at": utc_now_iso(),
        "sample_start": bundle_rows[0]["quarter"] if bundle_rows else None,
        "sample_end": bundle_rows[-1]["quarter"] if bundle_rows else None,
        "treatment_id": treatment_column,
        "instrument_ids": design_instrument_ids,
        "outcome_ids": required_outputs,
        "control_ids": control_ids,
        "state_ids": [item for item in ["qra_duration_supply_weekly", *required_state_ids] if item],
        "cutoff_rule": "conservative_source_available_at",
        "horizon_grid": job.get("horizons", []),
        "exclusion_windows": [],
        "scaling_rule": (
            "qoq_change_as_pct_of_nominal_gdp"
            if required_outputs and all(outcome.endswith("_pct_gdp") for outcome in required_outputs)
            else "raw_units_scaffold_only"
        ),
        "shock_definition": "canonical_tdc_level_change_placeholder",
        "multiple_testing_family": job.get("output_family"),
        "bundle_path": str(bundle_path),
        "missing_required_series": missing_series,
        "missing_state_ids": missing_state_ids,
        "missing_instrument_ids": missing_instrument_ids,
        "diagnostics_manifest_path": "",
        "usable_rows": usable_rows,
        "status": "partial_ready" if missing_series or missing_state_ids or missing_instrument_ids else "ready_for_estimation",
    }
    design_manifest_path = paths.manifests / f"{job_id}__design_manifest.json"

    required_outcome_rows = sum(
        1
        for row in bundle_rows
        if str(row.get(treatment_column, "")).strip()
        and (row.get("qra_ati_baseline_bn", "") if treatment_column == "tdc_bank_only_qoq" else True)
        and all(_row_has_outcome(row, outcome) for outcome in required_outputs)
        and all(_row_has_state(row, state_id) for state_id in required_state_ids)
        and all(_row_has_instrument(row, instrument_id) for instrument_id in configured_instruments)
    )

    sample_manifest_rows = [
        {"job_id": job_id, "step_order": 1, "step_label": "all_quarters_seen", "observations_remaining": len(bundle_rows), "reason": "union_of_available_standardized_quarters"},
        {"job_id": job_id, "step_order": 2, "step_label": "has_treatment", "observations_remaining": sum(1 for row in bundle_rows if str(row.get(treatment_column, "")).strip()), "reason": f"configured treatment column {treatment_column} present"},
    ]
    if treatment_column == "tdc_bank_only_qoq":
        sample_manifest_rows.append(
            {
                "job_id": job_id,
                "step_order": 3,
                "step_label": "has_qra_anchor",
                "observations_remaining": usable_rows,
                "reason": "qrawatch ATI baseline present alongside treatment",
            }
        )
        final_step_order = 4
        final_reason_suffix = "within usable treatment-anchor sample"
    else:
        sample_manifest_rows.append(
            {
                "job_id": job_id,
                "step_order": 3,
                "step_label": "usable_treatment_sample",
                "observations_remaining": usable_rows,
                "reason": "nonmissing configured treatment present",
            }
        )
        final_step_order = 4
        final_reason_suffix = "within usable treatment sample"
    sample_manifest_rows.append(
        {
            "job_id": job_id,
            "step_order": final_step_order,
            "step_label": "required_outcomes_present",
            "observations_remaining": required_outcome_rows if required_outputs else usable_rows,
            "reason": (
                (
                    (
                        f"required outcomes, states, and instruments available {final_reason_suffix}"
                        if configured_instruments
                        else f"required outcomes and states available {final_reason_suffix}"
                    )
                    if required_state_ids
                    else (
                        f"required outcomes and instruments available {final_reason_suffix}"
                        if configured_instruments
                        else f"required outcomes available {final_reason_suffix}"
                    )
                )
                if not missing_series and not missing_state_ids and not missing_instrument_ids
                else "currently limited by downloaded, normalized, or state inputs"
            ),
        }
    )
    sample_manifest_path = paths.manifests / f"{job_id}__sample_manifest.json"
    write_json(sample_manifest_path, {"job_id": job_id, "generated_at": utc_now_iso(), "rows": sample_manifest_rows})

    diagnostics_manifest_path: Path | None = None
    if str(job.get("estimator", "")).strip() == "lp_iv" and configured_instruments:
        diagnostics_manifest_path = paths.manifests / f"{job_id}__iv_diagnostics.json"
        write_json(
            diagnostics_manifest_path,
            _build_iv_diagnostics(
                bundle_rows=bundle_rows,
                job_id=job_id,
                treatment_id=treatment_column,
                required_outputs=required_outputs,
                required_state_ids=required_state_ids,
                configured_instruments=configured_instruments,
            ),
        )
        design_manifest["diagnostics_manifest_path"] = str(diagnostics_manifest_path)

    write_json(design_manifest_path, design_manifest)

    return DesignBuildResult(
        bundle_path=bundle_path,
        design_manifest_path=design_manifest_path,
        sample_manifest_path=sample_manifest_path,
        diagnostics_manifest_path=diagnostics_manifest_path,
        rows_written=len(bundle_rows),
        usable_rows=usable_rows,
    )
