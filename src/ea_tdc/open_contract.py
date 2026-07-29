from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class MethodTierContract:
    tier_id: str
    indicator_series_id: str
    expected_count: int


@dataclass(frozen=True)
class OpenContract:
    treatment_label: str
    canonical_treatment_id: str
    canonical_treatment_source_series: str
    canonical_residual_id: str
    canonical_outcome_id: str
    canonical_control_ids: tuple[str, ...]
    rolling_window_quarters: int
    open01_design_job_ids: tuple[str, ...]
    mmf_lower_treatment_id: str
    mmf_proportional_treatment_id: str
    mmf_upper_treatment_id: str
    method_tier_series_id: str
    method_tiers: tuple[MethodTierContract, ...]
    credit_screen_outcome_ids: tuple[str, ...]
    outcome_unit_multipliers: tuple[tuple[str, float], ...]
    treatment_units: str
    deposit_outcome_units: str
    credit_outcome_units: str
    sign_convention: str
    clock: str
    treatment_perimeter: str
    outcome_perimeter: str
    construction_tier_policy: str
    embedded_bank_treasury_component_id: str
    claim_boundary: str

    @property
    def mmf_treatment_ids(self) -> tuple[str, str, str]:
        return (
            self.mmf_lower_treatment_id,
            self.mmf_proportional_treatment_id,
            self.mmf_upper_treatment_id,
        )


OPEN_CONTRACT = OpenContract(
    treatment_label="regression_mmf_rrp_bank_long",
    canonical_treatment_id="tdc_tier2_regression_mmf_rrp_prop_bank_only_qoq",
    canonical_treatment_source_series="tdc_tier2_regression_mmf_rrp_prop_bank_only_ru_flow",
    canonical_residual_id="other_component_tier2_regression_mmf_rrp_prop_bank_only_qoq",
    canonical_outcome_id="matched_total_deposits",
    canonical_control_ids=(
        "GDP",
        "gdp_deflator",
        "FEDFUNDS",
        "TOTRESNS",
        "tier2_regression_bank_row_tier_pre_component_h15_scaled",
        "tdcpass_strict_loan_core_min_qoq__lag_2",
        "tdcpass_strict_loan_core_min_qoq__lag_4",
        "tdcpass_strict_loan_consumer_credit_qoq__lag_4",
        "bank_credit_qoq__lag_4",
        "dgs2__lag_4",
        "dgs10__lag_1",
        "dgs10__lag_2",
        "dflmx_k100_f1",
        "dflmx_k100_f2",
        "dflmx_k100_f3",
        "dflmx_k100_f4",
    ),
    rolling_window_quarters=48,
    open01_design_job_ids=(
        "tdc_tier2_mmf_rrp_canonical_full_panel",
        "tdc_tier2_regression_deposit_anatomy",
        "tdc_tier2_regression_credit_anatomy",
        "tdc_tier2_regression_plumbing_rates",
    ),
    mmf_lower_treatment_id="tdc_tier2_regression_mmf_rrp_lb_bank_only_qoq",
    mmf_proportional_treatment_id="tdc_tier2_regression_mmf_rrp_prop_bank_only_qoq",
    mmf_upper_treatment_id="tdc_tier2_regression_mmf_rrp_ub_bank_only_qoq",
    method_tier_series_id="tier2_regression_bank_row_method_tier",
    method_tiers=(
        MethodTierContract(
            tier_id="pre_component_h15_scaled_backcast",
            indicator_series_id=(
                "tier2_regression_bank_row_method_tier"
                "__is_pre_component_h15_scaled_backcast"
            ),
            expected_count=33,
        ),
        MethodTierContract(
            tier_id="component_pool_wamest_bucket_backcast",
            indicator_series_id=(
                "tier2_regression_bank_row_method_tier"
                "__is_component_pool_wamest_bucket_backcast"
            ),
            expected_count=47,
        ),
        MethodTierContract(
            tier_id="constrained_component",
            indicator_series_id=(
                "tier2_regression_bank_row_method_tier"
                "__is_constrained_component"
            ),
            expected_count=16,
        ),
    ),
    credit_screen_outcome_ids=(
        "tdcpass_strict_loan_core_min_qoq",
        "tdcpass_strict_loan_mortgages_qoq",
        "tdcpass_strict_loan_consumer_credit_qoq",
        "bank_credit_qoq",
        "bank_business_loans_qoq",
    ),
    outcome_unit_multipliers=(
        ("matched_total_deposits", 1.0),
        ("other_component_tier2_regression_mmf_rrp_prop_bank_only_qoq", 1.0),
        ("tdcpass_strict_loan_core_min_qoq", 1000.0),
        ("tdcpass_strict_loan_mortgages_qoq", 1000.0),
        ("tdcpass_strict_loan_consumer_credit_qoq", 1000.0),
        ("bank_credit_qoq", 1000.0),
        ("bank_business_loans_qoq", 1000.0),
    ),
    treatment_units="usd_millions_per_quarter",
    deposit_outcome_units="usd_millions_per_quarter",
    credit_outcome_units="usd_billions_per_quarter",
    sign_convention="positive_means_deposit_positive_tdc",
    clock="quarterly_flow_at_period_end; h0_is_same_quarter_contemporaneous_projection",
    treatment_perimeter=(
        "bank-deposit-scope Tier 2 regression TDC with the proportional MMF/RRP "
        "source-of-funds adjustment"
    ),
    outcome_perimeter=(
        "matched bank deposits and the same-treatment residual; credit outcomes are "
        "predeclared admission screens"
    ),
    construction_tier_policy=(
        "exactly one chronological method tier per canonical observation; expected "
        "counts are 33 pre-component H15, 47 component-pool/WAMEST, and 16 "
        "constrained-component quarters; fail closed on missing, unknown, overlap, "
        "or count drift"
    ),
    embedded_bank_treasury_component_id="bank_depository_tsy_tx",
    claim_boundary=(
        "Conditional same-quarter and rolling pass-through projection, not causal "
        "deposit creation, landing, or retention; credit claims require every "
        "predeclared stability screen."
    ),
)


def get_open_contract() -> OpenContract:
    return OPEN_CONTRACT


CANONICAL_TREATMENT_LABEL = OPEN_CONTRACT.treatment_label
CANONICAL_TREATMENT_ID = OPEN_CONTRACT.canonical_treatment_id
CANONICAL_TREATMENT_SOURCE_SERIES = OPEN_CONTRACT.canonical_treatment_source_series
CANONICAL_RESIDUAL_ID = OPEN_CONTRACT.canonical_residual_id
CANONICAL_OUTCOME_ID = OPEN_CONTRACT.canonical_outcome_id
CANONICAL_CONTROL_IDS = OPEN_CONTRACT.canonical_control_ids
ROLLING_WINDOW_QUARTERS = OPEN_CONTRACT.rolling_window_quarters
OPEN01_DESIGN_JOB_IDS = OPEN_CONTRACT.open01_design_job_ids
MMF_TREATMENT_IDS = OPEN_CONTRACT.mmf_treatment_ids
METHOD_TIER_SERIES_ID = OPEN_CONTRACT.method_tier_series_id
CREDIT_SCREEN_OUTCOME_IDS = OPEN_CONTRACT.credit_screen_outcome_ids
EMBEDDED_BANK_TREASURY_COMPONENT_ID = OPEN_CONTRACT.embedded_bank_treasury_component_id
EXPECTED_METHOD_TIER_COUNTS: Mapping[str, int] = MappingProxyType(
    {tier.tier_id: tier.expected_count for tier in OPEN_CONTRACT.method_tiers}
)
OUTCOME_UNIT_MULTIPLIERS: Mapping[str, float] = MappingProxyType(
    dict(OPEN_CONTRACT.outcome_unit_multipliers)
)


__all__ = [
    "CANONICAL_CONTROL_IDS",
    "CANONICAL_OUTCOME_ID",
    "CANONICAL_RESIDUAL_ID",
    "CANONICAL_TREATMENT_ID",
    "CANONICAL_TREATMENT_LABEL",
    "CANONICAL_TREATMENT_SOURCE_SERIES",
    "CREDIT_SCREEN_OUTCOME_IDS",
    "EMBEDDED_BANK_TREASURY_COMPONENT_ID",
    "EXPECTED_METHOD_TIER_COUNTS",
    "METHOD_TIER_SERIES_ID",
    "MMF_TREATMENT_IDS",
    "OPEN01_DESIGN_JOB_IDS",
    "OPEN_CONTRACT",
    "OUTCOME_UNIT_MULTIPLIERS",
    "ROLLING_WINDOW_QUARTERS",
    "MethodTierContract",
    "OpenContract",
    "get_open_contract",
]
