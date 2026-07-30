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


TreasuryLineage = tuple[int, int, int]


@dataclass(frozen=True)
class Open02SeriesContract:
    key: str
    fred_id: str
    board_series_id: str
    official_title: str
    sector: str
    side: str
    positive_sign: str
    role: str
    treasury_lineage: TreasuryLineage = (0, 0, 0)
    board_locator_exception: bool = False
    frequency: str = "quarterly"
    units: str = "millions_of_us_dollars"
    financial_accounts_type: str = "FU_transaction"
    seasonal_adjustment: str = "not_seasonally_adjusted"
    primary_transform: str = "none"


@dataclass(frozen=True)
class Open02FormulaContract:
    formula_id: str
    output_id: str
    terms: tuple[tuple[str, float], ...]
    tolerance_kind: str
    tolerance: float
    tolerance_scale_formula: str = "1"


@dataclass(frozen=True)
class Open02SystemContract:
    system_id: str
    outcome_ids: tuple[str, ...]
    regressor_ids: tuple[str, ...]
    coefficient_ids: tuple[str, ...]
    agency_component_outcome_ids: tuple[str, ...]
    design_parameter_count: int
    formula: str


@dataclass(frozen=True)
class Open02CoefficientSelectionContract:
    coefficient_id: str
    outcome_id: str
    regressor_id: str
    equation_index: int
    coefficient_index: int
    null_value: float


@dataclass(frozen=True)
class Open02WaldContract:
    hypothesis_id: str
    system_id: str
    coefficient_ids: tuple[str, ...]
    selections: tuple[Open02CoefficientSelectionContract, ...]
    degrees_of_freedom: int


@dataclass(frozen=True)
class Open02SampleContract:
    start_quarter: str
    end_quarter: str
    observations: int
    quarter_hash: str
    accepted_tdcest_bundle_generated_at: str
    observation_vintage_cutoff: str
    vintage_policy: str
    complete_case_policy: str
    required_common_hashes: tuple[str, str, str]
    rank_policy: str
    identity_tolerance_usd_millions: float
    coefficient_adding_up_relative_tolerance: float


@dataclass(frozen=True)
class Open02CovarianceContract:
    coefficient_estimator: str
    estimator: str
    kernel: str
    lag_quarters: int
    prewhitening: str
    finite_sample_correction: str
    test_sidedness: str
    score_definition: str
    sandwich_bread: str


@dataclass(frozen=True)
class Open02HolmContract:
    hypothesis_ids: tuple[str, ...]
    familywise_alpha: float
    all_hypotheses_required_for_promotion: bool
    agency_component_p_value_role: str


@dataclass(frozen=True)
class Open02InfluenceGroupContract:
    group_id: str
    hypothesis_id: str
    coefficient_ids: tuple[str, ...]


@dataclass(frozen=True)
class Open02InfluenceContract:
    groups: tuple[Open02InfluenceGroupContract, ...]
    quarter_deletion_size: int
    quarter_deletion_fits: int
    block_deletion_size: int
    block_deletion_fits: int
    quarter_deletion_policy: str
    block_deletion_policy: str
    refit_policy: str
    relative_l2_formula: str
    relative_l2_denominator_floor: float
    maximum_quarter_influence: float
    maximum_block_influence: float
    sign_stability_raw_p_threshold: float
    sign_stability_policy: str


@dataclass(frozen=True)
class Open02LineageContract:
    raw_treasury_ids: tuple[str, str, str]
    lineage_by_id: tuple[tuple[str, TreasuryLineage], ...]
    no_double_side_policy: str


@dataclass(frozen=True)
class Open02ControlContract:
    control_id: str
    source_series_id: str | None
    lag_quarters: int
    indicator_quarter: int | None
    source_policy: str
    generation_policy: str
    treasury_lineage: TreasuryLineage


@dataclass(frozen=True)
class Open02ValidityGateContract:
    gate_id: str
    reason_code: str
    requirement: str


@dataclass(frozen=True)
class Open02ValidityDisposition:
    valid_result: bool
    main_text_eligible: bool
    appendix_only: bool


@dataclass(frozen=True)
class Open02OfficialSourceContract:
    release_date: str
    archive_url: str
    archive_sha256: str
    unit_label: str
    csv_member_sha256: tuple[tuple[str, str], ...]
    dictionary_member_sha256: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Open02OutputContract:
    report_path: str
    receipt_path: str
    required_receipt_fields: tuple[str, ...]


@dataclass(frozen=True)
class Open02Contract:
    open01_contract: OpenContract
    series: tuple[Open02SeriesContract, ...]
    formulas: tuple[Open02FormulaContract, ...]
    systems: tuple[Open02SystemContract, ...]
    sample: Open02SampleContract
    source: Open02OfficialSourceContract
    control_ids: tuple[str, ...]
    controls: tuple[Open02ControlContract, ...]
    rejected_control_ids: tuple[str, ...]
    rejected_proxy_ids: tuple[str, ...]
    covariance: Open02CovarianceContract
    wald_hypotheses: tuple[Open02WaldContract, ...]
    holm: Open02HolmContract
    influence: Open02InfluenceContract
    lineage: Open02LineageContract
    validity_gates: tuple[Open02ValidityGateContract, ...]
    output: Open02OutputContract
    perimeter_sectors: tuple[str, str, str]
    excluded_sector: str
    initial_status: str
    initial_main_text_eligible: bool
    initial_appendix_only: str
    invalid_result_disposition: Open02ValidityDisposition
    valid_nonpromoted_disposition: Open02ValidityDisposition
    promoted_result_disposition: Open02ValidityDisposition
    promotion_reason_codes: tuple[str, ...]
    agency_composition_boundary: str
    us_chartered_sensitivity_claim_boundary: str
    treatment_naming_boundary: str
    balance_sheet_boundary: str
    claim_boundary: str

    @property
    def canonical_treatment_id(self) -> str:
        return self.open01_contract.canonical_treatment_id

    @property
    def canonical_treatment_source_series(self) -> str:
        return self.open01_contract.canonical_treatment_source_series

    @property
    def embedded_bank_treasury_component_id(self) -> str:
        return self.open01_contract.embedded_bank_treasury_component_id

    @property
    def official_source(self) -> Open02OfficialSourceContract:
        return self.source

    @property
    def control_specs(self) -> tuple[Open02ControlContract, ...]:
        return self.controls

    @property
    def control_contracts(self) -> tuple[Open02ControlContract, ...]:
        return self.controls

    @property
    def influence_groups(self) -> tuple[Open02InfluenceGroupContract, ...]:
        return self.influence.groups


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


OPEN02_LEAVE_OUT_ID = "open02_tdc_ex_bank_treasury"
OPEN02_BANK_TREASURY_ID = "open02_bank_treasury_acquisition"
OPEN02_BANK_AGENCY_ID = "open02_bank_agency_gse_acquisition"
OPEN02_BANK_LOANS_ID = "open02_bank_loan_acquisition"
OPEN02_BANK_DEPOSITS_ID = "open02_bank_customer_deposit_increase"

OPEN02_SERIES = (
    Open02SeriesContract(
        key="tsy_us",
        fred_id="BOGZ1FU763061100Q",
        board_series_id="FU763061100.Q",
        official_title=(
            "U.S.-Chartered Depository Institutions; Treasury Securities; "
            "Asset, Transactions"
        ),
        sector="us_chartered_depository_institutions",
        side="asset",
        positive_sign="net_acquisition",
        role="treasury_component",
        treasury_lineage=(1, 0, 0),
    ),
    Open02SeriesContract(
        key="tsy_fbo",
        fred_id="BOGZ1FU753061103Q",
        board_series_id="FU753061103.Q",
        official_title=(
            "Foreign Banking Offices in the U.S.; Treasury Securities; "
            "Asset, Transactions"
        ),
        sector="foreign_banking_offices_in_us",
        side="asset",
        positive_sign="net_acquisition",
        role="treasury_component",
        treasury_lineage=(0, 1, 0),
    ),
    Open02SeriesContract(
        key="tsy_aff",
        fred_id="BOGZ1FU743061103Q",
        board_series_id="FU743061103.Q",
        official_title=(
            "Banks in U.S.-Affiliated Areas; Treasury Securities; Asset, Transactions"
        ),
        sector="banks_in_us_affiliated_areas",
        side="asset",
        positive_sign="net_acquisition",
        role="treasury_component",
        treasury_lineage=(0, 0, 1),
    ),
    Open02SeriesContract(
        key="agency_us_total",
        fred_id="BOGZ1FU763061705Q",
        board_series_id="FU763061705.Q",
        official_title=(
            "U.S.-Chartered Depository Institutions; Agency- And GSE-Backed "
            "Securities; Asset, Transactions"
        ),
        sector="us_chartered_depository_institutions",
        side="asset",
        positive_sign="net_acquisition",
        role="agency_sector_total",
    ),
    Open02SeriesContract(
        key="agency_fbo_total",
        fred_id="BOGZ1FU753061703Q",
        board_series_id="FU753061703.Q",
        official_title=(
            "Foreign Banking Offices in the U.S.; Agency- And GSE-Backed "
            "Securities; Asset, Transactions"
        ),
        sector="foreign_banking_offices_in_us",
        side="asset",
        positive_sign="net_acquisition",
        role="agency_sector_total",
    ),
    Open02SeriesContract(
        key="agency_aff_total",
        fred_id="BOGZ1FU743061703Q",
        board_series_id="FU743061703.Q",
        official_title=(
            "Banks in U.S.-Affiliated Areas; Agency- And GSE-Backed Securities; "
            "Asset, Transactions"
        ),
        sector="banks_in_us_affiliated_areas",
        side="asset",
        positive_sign="net_acquisition",
        role="agency_sector_total",
    ),
    Open02SeriesContract(
        key="agency_us_res_pass",
        fred_id="BOGZ1FU763061805Q",
        board_series_id="FU763061805.Q",
        official_title=(
            "U.S.-Chartered Depository Institutions; Total Agency Issued Residential "
            "Mortgage Pass-Through Securities; Asset, Transactions"
        ),
        sector="us_chartered_depository_institutions",
        side="asset",
        positive_sign="net_acquisition",
        role="agency_us_component",
    ),
    Open02SeriesContract(
        key="agency_us_com_pass",
        fred_id="BOGZ1FU763061303Q",
        board_series_id="FU763061503.Q",
        official_title=(
            "U.S.-Chartered Depository Institutions; Agency Issued Commercial "
            "Mortgage Pass-Through Securities; Asset, Transactions"
        ),
        sector="us_chartered_depository_institutions",
        side="asset",
        positive_sign="net_acquisition",
        role="agency_us_component",
        board_locator_exception=True,
    ),
    Open02SeriesContract(
        key="agency_us_res_cmo",
        fred_id="BOGZ1FU763061603Q",
        board_series_id="FU763061603.Q",
        official_title=(
            "U.S.-Chartered Depository Institutions; Agency Issued Residential CMOs "
            "and Other Structured MBS; Asset, Transactions"
        ),
        sector="us_chartered_depository_institutions",
        side="asset",
        positive_sign="net_acquisition",
        role="agency_us_component",
    ),
    Open02SeriesContract(
        key="agency_us_com_cmo",
        fred_id="BOGZ1FU763061403Q",
        board_series_id="FU763061403.Q",
        official_title=(
            "U.S.-Chartered Depository Institutions; Agency Issued Commercial CMOs "
            "and Other Structured MBS; Asset, Transactions"
        ),
        sector="us_chartered_depository_institutions",
        side="asset",
        positive_sign="net_acquisition",
        role="agency_us_component",
    ),
    Open02SeriesContract(
        key="agency_us_other",
        fred_id="BOGZ1FU763061795Q",
        board_series_id="FU763061795.Q",
        official_title=(
            "U.S.-Chartered Depository Institutions; Agency- And GSE-Backed "
            "Securities, Excluding MBS and CMOs; Asset, Transactions"
        ),
        sector="us_chartered_depository_institutions",
        side="asset",
        positive_sign="net_acquisition",
        role="agency_us_component",
    ),
    Open02SeriesContract(
        key="loans_us",
        fred_id="BOGZ1FU764023005Q",
        board_series_id="FU764023005.Q",
        official_title=(
            "U.S.-Chartered Depository Institutions; Loans; Asset, Transactions"
        ),
        sector="us_chartered_depository_institutions",
        side="asset",
        positive_sign="net_acquisition_or_origination",
        role="loan_component",
    ),
    Open02SeriesContract(
        key="loans_fbo",
        fred_id="BOGZ1FU754023005Q",
        board_series_id="FU754023005.Q",
        official_title=(
            "Foreign Banking Offices in the U.S.; Loans; Asset, Transactions"
        ),
        sector="foreign_banking_offices_in_us",
        side="asset",
        positive_sign="net_acquisition_or_origination",
        role="loan_component",
    ),
    Open02SeriesContract(
        key="loans_aff",
        fred_id="BOGZ1FU744023003Q",
        board_series_id="FU744023003.Q",
        official_title=(
            "Banks in U.S.-Affiliated Areas; Loans; Asset, Transactions"
        ),
        sector="banks_in_us_affiliated_areas",
        side="asset",
        positive_sign="net_acquisition_or_origination",
        role="loan_component",
    ),
    Open02SeriesContract(
        key="dep_us_check",
        fred_id="BOGZ1FU763127005Q",
        board_series_id="FU763127005.Q",
        official_title=(
            "U.S.-Chartered Depository Institutions; Checkable Deposits; "
            "Liability, Transactions"
        ),
        sector="us_chartered_depository_institutions",
        side="liability",
        positive_sign="net_liability_increase",
        role="customer_deposit_component",
    ),
    Open02SeriesContract(
        key="dep_us_time",
        fred_id="BOGZ1FU763130005Q",
        board_series_id="FU763130005.Q",
        official_title=(
            "U.S.-Chartered Depository Institutions, Including IBFs; Total Time and "
            "Savings Deposits; Liability, Transactions"
        ),
        sector="us_chartered_depository_institutions",
        side="liability",
        positive_sign="net_liability_increase",
        role="customer_deposit_component",
    ),
    Open02SeriesContract(
        key="dep_fbo_check",
        fred_id="BOGZ1FU753127005Q",
        board_series_id="FU753127005.Q",
        official_title=(
            "Foreign Banking Offices in the U.S.; Checkable Deposits; "
            "Liability, Transactions"
        ),
        sector="foreign_banking_offices_in_us",
        side="liability",
        positive_sign="net_liability_increase",
        role="customer_deposit_component",
    ),
    Open02SeriesContract(
        key="dep_fbo_time",
        fred_id="BOGZ1FU753130005Q",
        board_series_id="FU753130005.Q",
        official_title=(
            "Foreign Banking Offices in the U.S., Including IBFs; Total Time and "
            "Savings Deposits; Liability, Transactions"
        ),
        sector="foreign_banking_offices_in_us",
        side="liability",
        positive_sign="net_liability_increase",
        role="customer_deposit_component",
    ),
    Open02SeriesContract(
        key="dep_aff_check",
        fred_id="BOGZ1FU743127003Q",
        board_series_id="FU743127003.Q",
        official_title=(
            "Banks in U.S.-Affiliated Areas; Checkable Deposits; Liability, "
            "Transactions"
        ),
        sector="banks_in_us_affiliated_areas",
        side="liability",
        positive_sign="net_liability_increase",
        role="customer_deposit_component",
    ),
    Open02SeriesContract(
        key="dep_aff_time",
        fred_id="BOGZ1FU743130003Q",
        board_series_id="FU743130003.Q",
        official_title=(
            "Banks in U.S.-Affiliated Areas; Total Time and Savings Deposits; "
            "Liability, Transactions"
        ),
        sector="banks_in_us_affiliated_areas",
        side="liability",
        positive_sign="net_liability_increase",
        role="customer_deposit_component",
    ),
)

OPEN02_CONTROL_IDS = (
    "GDP",
    "gdp_deflator",
    "FEDFUNDS",
    "TOTRESNS",
    "tier2_regression_bank_row_tier_pre_component_h15_scaled",
    "dgs2__lag_4",
    "dgs10__lag_1",
    "dgs10__lag_2",
    "quarter_is_q2",
    "quarter_is_q3",
    "quarter_is_q4",
)

OPEN02_CONTROLS = (
    *(
        Open02ControlContract(
            control_id=control_id,
            source_series_id=control_id,
            lag_quarters=0,
            indicator_quarter=None,
            source_policy="accepted_open01_design_rows",
            generation_policy="reuse_exact_value_no_refetch_no_retransform",
            treasury_lineage=(0, 0, 0),
        )
        for control_id in OPEN02_CONTROL_IDS[:5]
    ),
    Open02ControlContract(
        control_id="dgs2__lag_4",
        source_series_id="dgs2",
        lag_quarters=4,
        indicator_quarter=None,
        source_policy="accepted_open01_design_history",
        generation_policy="exact_four_quarter_row_lag_no_refetch",
        treasury_lineage=(0, 0, 0),
    ),
    Open02ControlContract(
        control_id="dgs10__lag_1",
        source_series_id="dgs10",
        lag_quarters=1,
        indicator_quarter=None,
        source_policy="accepted_open01_design_history",
        generation_policy="exact_one_quarter_row_lag_no_refetch",
        treasury_lineage=(0, 0, 0),
    ),
    Open02ControlContract(
        control_id="dgs10__lag_2",
        source_series_id="dgs10",
        lag_quarters=2,
        indicator_quarter=None,
        source_policy="accepted_open01_design_history",
        generation_policy="exact_two_quarter_row_lag_no_refetch",
        treasury_lineage=(0, 0, 0),
    ),
    Open02ControlContract(
        control_id="quarter_is_q2",
        source_series_id=None,
        lag_quarters=0,
        indicator_quarter=2,
        source_policy="deterministic_from_ordered_quarter",
        generation_policy="indicator_quarter_equals_q2_with_q1_omitted",
        treasury_lineage=(0, 0, 0),
    ),
    Open02ControlContract(
        control_id="quarter_is_q3",
        source_series_id=None,
        lag_quarters=0,
        indicator_quarter=3,
        source_policy="deterministic_from_ordered_quarter",
        generation_policy="indicator_quarter_equals_q3_with_q1_omitted",
        treasury_lineage=(0, 0, 0),
    ),
    Open02ControlContract(
        control_id="quarter_is_q4",
        source_series_id=None,
        lag_quarters=0,
        indicator_quarter=4,
        source_policy="deterministic_from_ordered_quarter",
        generation_policy="indicator_quarter_equals_q4_with_q1_omitted",
        treasury_lineage=(0, 0, 0),
    ),
)

OPEN02_AGENCY_COMPONENT_IDS = (
    "agency_us_res_pass",
    "agency_us_com_pass",
    "agency_us_res_cmo",
    "agency_us_com_cmo",
    "agency_us_other",
    "agency_fbo_total",
    "agency_aff_total",
)

OPEN02_FORMULAS = (
    Open02FormulaContract(
        formula_id="bank_treasury_three_sector_sum",
        output_id=OPEN02_BANK_TREASURY_ID,
        terms=(("tsy_us", 1.0), ("tsy_fbo", 1.0), ("tsy_aff", 1.0)),
        tolerance_kind="absolute_usd_millions",
        tolerance=1e-6,
    ),
    Open02FormulaContract(
        formula_id="accepted_bank_treasury_component_reconciliation",
        output_id=OPEN_CONTRACT.embedded_bank_treasury_component_id,
        terms=((OPEN02_BANK_TREASURY_ID, 1.0),),
        tolerance_kind="absolute_usd_millions",
        tolerance=1e-6,
    ),
    Open02FormulaContract(
        formula_id="leave_out_definition",
        output_id=OPEN02_LEAVE_OUT_ID,
        terms=(
            (OPEN_CONTRACT.canonical_treatment_source_series, 1.0),
            (OPEN02_BANK_TREASURY_ID, -1.0),
        ),
        tolerance_kind="absolute_usd_millions",
        tolerance=1e-6,
    ),
    Open02FormulaContract(
        formula_id="canonical_reconstruction",
        output_id=OPEN_CONTRACT.canonical_treatment_source_series,
        terms=((OPEN02_LEAVE_OUT_ID, 1.0), (OPEN02_BANK_TREASURY_ID, 1.0)),
        tolerance_kind="absolute_usd_millions",
        tolerance=1e-6,
    ),
    Open02FormulaContract(
        formula_id="agency_us_five_component_identity",
        output_id="agency_us_total",
        terms=(
            ("agency_us_res_pass", 1.0),
            ("agency_us_com_pass", 1.0),
            ("agency_us_res_cmo", 1.0),
            ("agency_us_com_cmo", 1.0),
            ("agency_us_other", 1.0),
        ),
        tolerance_kind="absolute_usd_millions",
        tolerance=1e-6,
    ),
    Open02FormulaContract(
        formula_id="agency_three_sector_identity",
        output_id=OPEN02_BANK_AGENCY_ID,
        terms=(
            ("agency_us_total", 1.0),
            ("agency_fbo_total", 1.0),
            ("agency_aff_total", 1.0),
        ),
        tolerance_kind="absolute_usd_millions",
        tolerance=1e-6,
    ),
    Open02FormulaContract(
        formula_id="agency_seven_component_identity",
        output_id=OPEN02_BANK_AGENCY_ID,
        terms=tuple((series_id, 1.0) for series_id in OPEN02_AGENCY_COMPONENT_IDS),
        tolerance_kind="absolute_usd_millions",
        tolerance=1e-6,
    ),
    Open02FormulaContract(
        formula_id="loans_three_sector_identity",
        output_id=OPEN02_BANK_LOANS_ID,
        terms=(("loans_us", 1.0), ("loans_fbo", 1.0), ("loans_aff", 1.0)),
        tolerance_kind="absolute_usd_millions",
        tolerance=1e-6,
    ),
    Open02FormulaContract(
        formula_id="customer_deposit_identity",
        output_id=OPEN02_BANK_DEPOSITS_ID,
        terms=(
            ("dep_us_check", 1.0),
            ("dep_us_time", 1.0),
            ("dep_fbo_check", 1.0),
            ("dep_fbo_time", 1.0),
            ("dep_aff_check", 1.0),
            ("dep_aff_time", 1.0),
        ),
        tolerance_kind="absolute_usd_millions",
        tolerance=1e-6,
    ),
    Open02FormulaContract(
        formula_id="source_agency_coefficient_adding_up",
        output_id="beta_A",
        terms=tuple((f"beta_{series_id}", 1.0) for series_id in OPEN02_AGENCY_COMPONENT_IDS),
        tolerance_kind="relative_max_one_or_coefficients",
        tolerance=1e-10,
        tolerance_scale_formula=(
            "max(1, abs(aggregate_coefficient), "
            "abs(sum_component_coefficients))"
        ),
    ),
    Open02FormulaContract(
        formula_id="within_agency_coefficient_adding_up",
        output_id="theta_A",
        terms=tuple((f"theta_{series_id}", 1.0) for series_id in OPEN02_AGENCY_COMPONENT_IDS),
        tolerance_kind="relative_max_one_or_coefficients",
        tolerance=1e-10,
        tolerance_scale_formula=(
            "max(1, abs(aggregate_coefficient), "
            "abs(sum_component_coefficients))"
        ),
    ),
)

OPEN02_SYSTEMS = (
    Open02SystemContract(
        system_id="source_side_response",
        outcome_ids=(
            OPEN02_BANK_TREASURY_ID,
            OPEN02_BANK_AGENCY_ID,
            OPEN02_BANK_LOANS_ID,
            OPEN02_BANK_DEPOSITS_ID,
        ),
        regressor_ids=(OPEN02_LEAVE_OUT_ID,),
        coefficient_ids=("beta_B", "beta_A", "beta_L", "beta_D"),
        agency_component_outcome_ids=OPEN02_AGENCY_COMPONENT_IDS,
        design_parameter_count=13,
        formula="Y_j = intercept + beta_j * X + frozen_controls",
    ),
    Open02SystemContract(
        system_id="within_bank_conditional_co_movement",
        outcome_ids=(
            OPEN02_BANK_AGENCY_ID,
            OPEN02_BANK_LOANS_ID,
            OPEN02_BANK_DEPOSITS_ID,
        ),
        regressor_ids=(OPEN02_BANK_TREASURY_ID, OPEN02_LEAVE_OUT_ID),
        coefficient_ids=(
            "theta_A",
            "theta_L",
            "theta_D",
            "lambda_A",
            "lambda_L",
            "lambda_D",
        ),
        agency_component_outcome_ids=OPEN02_AGENCY_COMPONENT_IDS,
        design_parameter_count=14,
        formula=(
            "Y_j = intercept + theta_j * B + lambda_j * X + frozen_controls"
        ),
    ),
)

OPEN02_WALD_HYPOTHESES = (
    Open02WaldContract(
        hypothesis_id="H_T",
        system_id="source_side_response",
        coefficient_ids=("beta_B",),
        selections=(
            Open02CoefficientSelectionContract(
                coefficient_id="beta_B",
                outcome_id=OPEN02_BANK_TREASURY_ID,
                regressor_id=OPEN02_LEAVE_OUT_ID,
                equation_index=0,
                coefficient_index=1,
                null_value=0.0,
            ),
        ),
        degrees_of_freedom=1,
    ),
    Open02WaldContract(
        hypothesis_id="H_P",
        system_id="source_side_response",
        coefficient_ids=("beta_A", "beta_L", "beta_D"),
        selections=(
            Open02CoefficientSelectionContract(
                coefficient_id="beta_A",
                outcome_id=OPEN02_BANK_AGENCY_ID,
                regressor_id=OPEN02_LEAVE_OUT_ID,
                equation_index=1,
                coefficient_index=1,
                null_value=0.0,
            ),
            Open02CoefficientSelectionContract(
                coefficient_id="beta_L",
                outcome_id=OPEN02_BANK_LOANS_ID,
                regressor_id=OPEN02_LEAVE_OUT_ID,
                equation_index=2,
                coefficient_index=1,
                null_value=0.0,
            ),
            Open02CoefficientSelectionContract(
                coefficient_id="beta_D",
                outcome_id=OPEN02_BANK_DEPOSITS_ID,
                regressor_id=OPEN02_LEAVE_OUT_ID,
                equation_index=3,
                coefficient_index=1,
                null_value=0.0,
            ),
        ),
        degrees_of_freedom=3,
    ),
    Open02WaldContract(
        hypothesis_id="H_W",
        system_id="within_bank_conditional_co_movement",
        coefficient_ids=("theta_A", "theta_L", "theta_D"),
        selections=(
            Open02CoefficientSelectionContract(
                coefficient_id="theta_A",
                outcome_id=OPEN02_BANK_AGENCY_ID,
                regressor_id=OPEN02_BANK_TREASURY_ID,
                equation_index=0,
                coefficient_index=1,
                null_value=0.0,
            ),
            Open02CoefficientSelectionContract(
                coefficient_id="theta_L",
                outcome_id=OPEN02_BANK_LOANS_ID,
                regressor_id=OPEN02_BANK_TREASURY_ID,
                equation_index=1,
                coefficient_index=1,
                null_value=0.0,
            ),
            Open02CoefficientSelectionContract(
                coefficient_id="theta_D",
                outcome_id=OPEN02_BANK_DEPOSITS_ID,
                regressor_id=OPEN02_BANK_TREASURY_ID,
                equation_index=2,
                coefficient_index=1,
                null_value=0.0,
            ),
        ),
        degrees_of_freedom=3,
    ),
)

OPEN02_REJECTED_CONTROL_IDS = (
    "tdcpass_strict_loan_core_min_qoq__lag_2",
    "tdcpass_strict_loan_core_min_qoq__lag_4",
    "tdcpass_strict_loan_consumer_credit_qoq__lag_4",
    "bank_credit_qoq__lag_4",
    "dflmx_k100_f1",
    "dflmx_k100_f2",
    "dflmx_k100_f3",
    "dflmx_k100_f4",
)

OPEN02_REJECTED_PROXY_IDS = (
    "BOGZ1FL764100005Q",
    "matched_total_deposits",
    "BOGZ1FL763061100Q",
    "BOGZ1FL704041005Q",
    "TOTBKCR",
    "BUSLOANS",
    "TOTCI",
    "OSEACBW027SBOG",
    "TASACBW027SBOG",
    "CLSACBW027SBOG",
    "RELACBW027SBOG",
    "BOGZ1FU703061705Q",
    "BOGZ1FU763061503Q",
)

OPEN02_RAW_TREASURY_IDS = (
    "BOGZ1FU763061100Q",
    "BOGZ1FU753061103Q",
    "BOGZ1FU743061103Q",
)

OPEN02_LINEAGE_BY_ID = (
    (OPEN_CONTRACT.canonical_treatment_id, (1, 1, 1)),
    (OPEN_CONTRACT.canonical_treatment_source_series, (1, 1, 1)),
    (OPEN_CONTRACT.embedded_bank_treasury_component_id, (1, 1, 1)),
    (OPEN02_BANK_TREASURY_ID, (1, 1, 1)),
    (OPEN02_LEAVE_OUT_ID, (0, 0, 0)),
    (OPEN02_BANK_AGENCY_ID, (0, 0, 0)),
    (OPEN02_BANK_LOANS_ID, (0, 0, 0)),
    (OPEN02_BANK_DEPOSITS_ID, (0, 0, 0)),
    *((control_id, (0, 0, 0)) for control_id in OPEN02_CONTROL_IDS),
)

OPEN02_VALIDITY_GATES = (
    Open02ValidityGateContract(
        gate_id="metadata",
        reason_code="metadata_gate_failed",
        requirement=(
            "exact 20-series allowlist, titles, sectors, sides, units, FU status, "
            "and NSA status"
        ),
    ),
    Open02ValidityGateContract(
        gate_id="vintage",
        reason_code="vintage_gate_failed",
        requirement="one pinned official observation vintage for all 20 series",
    ),
    Open02ValidityGateContract(
        gate_id="coverage",
        reason_code="coverage_gate_failed",
        requirement="exactly 96 complete consecutive observations, 2002Q1-2025Q4",
    ),
    Open02ValidityGateContract(
        gate_id="treasury_component",
        reason_code="treasury_component_gate_failed",
        requirement="B equals the three raw Treasury transactions within 1e-6",
    ),
    Open02ValidityGateContract(
        gate_id="accepted_component_reconciliation",
        reason_code="accepted_component_reconciliation_failed",
        requirement="B equals accepted bank_depository_tsy_tx within 1e-6",
    ),
    Open02ValidityGateContract(
        gate_id="leave_out_reconstruction",
        reason_code="leave_out_reconstruction_failed",
        requirement="canonical equals X plus B within 1e-6",
    ),
    Open02ValidityGateContract(
        gate_id="us_agency_identity",
        reason_code="us_agency_identity_failed",
        requirement="U.S. agency total equals five published components within 1e-6",
    ),
    Open02ValidityGateContract(
        gate_id="three_sector_agency_identity",
        reason_code="three_sector_agency_identity_failed",
        requirement=(
            "A equals five U.S. components plus FBO and affiliated totals within 1e-6"
        ),
    ),
    Open02ValidityGateContract(
        gate_id="coefficient_adding_up",
        reason_code="coefficient_adding_up_failed",
        requirement=(
            "aggregate and component coefficients satisfy the frozen relative "
            "tolerance"
        ),
    ),
    Open02ValidityGateContract(
        gate_id="common_sample_design",
        reason_code="common_sample_design_failed",
        requirement=(
            "exact row, column, and design hashes match across relevant equations"
        ),
    ),
    Open02ValidityGateContract(
        gate_id="rank",
        reason_code="rank_gate_failed",
        requirement="every design matrix is full column rank",
    ),
    Open02ValidityGateContract(
        gate_id="lineage",
        reason_code="lineage_gate_failed",
        requirement="no raw Treasury transaction has nonzero lineage on both sides",
    ),
)

OPEN02_OFFICIAL_SOURCE = Open02OfficialSourceContract(
    release_date="2026-03-19",
    archive_url=(
        "https://www.federalreserve.gov/releases/z1/20260319/"
        "z1_csv_files.zip"
    ),
    archive_sha256=(
        "4a758a65a5190987a53e24039d91cc2b09ed55e57a2560bc640fdfe191ceee35"
    ),
    unit_label="Millions of dollars; transactions, not seasonally adjusted",
    csv_member_sha256=(
        (
            "csv/fu111.csv",
            "2dc83502d138e5253117a784c28b8fbeeba0e1460db2439ad243c116c1de9a11",
        ),
        (
            "csv/fu112.csv",
            "b016e3d742f4dffd61b12947a2e64605c3338c71507297d925f61f4980f7bae7",
        ),
        (
            "csv/fu113.csv",
            "e25f4a6843c428bd120fc416974578765f14253a76d0fb07c79369c46c7df952",
        ),
    ),
    dictionary_member_sha256=(
        (
            "data_dictionary/fu111.txt",
            "df992f6d6868a8665023f558018a2baf69b24c32933c34221d31acae8c82f1f7",
        ),
        (
            "data_dictionary/fu112.txt",
            "f91d4217bf99636658068456ccbb7d9c7c9fabcf268d886238fdd182ad4bd818",
        ),
        (
            "data_dictionary/fu113.txt",
            "2e51ba3a7b9f24a236ba4ba8dee57da04c8d61deb86f657d7e02f2a8f3873155",
        ),
    ),
)

OPEN02_OUTPUT = Open02OutputContract(
    report_path="output/reports/tier2_bank_portfolio_response_system.csv",
    receipt_path="output/manifests/open02_producer_run_receipt.json",
    required_receipt_fields=(
        "producer_commit",
        "run_id",
        "argv",
        "source.input_hashes",
        "accepted_open01_inputs.input_hashes",
        "outputs.response_system.sha256",
        "units",
        "sample",
        "acceptance.gates",
        "acceptance.identity_errors",
        "acceptance.row_hash",
        "acceptance.column_hashes",
        "acceptance.design_hashes",
        "acceptance.wald_statistics",
        "acceptance.raw_p_values",
        "acceptance.holm_adjusted_p_values",
        "acceptance.influence_maxima",
        "acceptance.valid_result",
        "acceptance.main_text_eligible",
        "acceptance.appendix_only",
        "acceptance.reason_codes",
    ),
)

OPEN02_CONTRACT = Open02Contract(
    open01_contract=OPEN_CONTRACT,
    series=OPEN02_SERIES,
    formulas=OPEN02_FORMULAS,
    systems=OPEN02_SYSTEMS,
    sample=Open02SampleContract(
        start_quarter="2002Q1",
        end_quarter="2025Q4",
        observations=96,
        quarter_hash="f0de664ba1588848933205da2fd01df64a864ed5696c307771a7eebf88d56713",
        accepted_tdcest_bundle_generated_at="2026-05-22T16:56:46Z",
        observation_vintage_cutoff="2026-05-22T16:56:46Z",
        vintage_policy=(
            "the completed Federal Reserve 2026-03-19 Z.1 archive for all 20 "
            "response series, with exact accepted TDCest bank-Treasury equality "
            "authoritative; no current-release or cross-vintage substitution"
        ),
        complete_case_policy=(
            "exactly the ordered 96 consecutive quarters for canonical, accepted "
            "Treasury component, all 20 raw series, and all 11 controls; fail closed "
            "on missing, duplicate, nonconsecutive, or rehashed rows inside the "
            "frozen window; later accepted-design rows are out of sample"
        ),
        required_common_hashes=("row_hash", "column_hash", "design_hash"),
        rank_policy="every_relevant_design_matrix_must_be_full_column_rank",
        identity_tolerance_usd_millions=1e-6,
        coefficient_adding_up_relative_tolerance=1e-10,
    ),
    source=OPEN02_OFFICIAL_SOURCE,
    control_ids=OPEN02_CONTROL_IDS,
    controls=OPEN02_CONTROLS,
    rejected_control_ids=OPEN02_REJECTED_CONTROL_IDS,
    rejected_proxy_ids=OPEN02_REJECTED_PROXY_IDS,
    covariance=Open02CovarianceContract(
        coefficient_estimator="equation_by_equation_ols",
        estimator="stacked_system_newey_west_hac",
        kernel="bartlett",
        lag_quarters=4,
        prewhitening="none",
        finite_sample_correction="T/(T-K)",
        test_sidedness="two_sided",
        score_definition="equation_major_stack_of_x_t_times_equation_residual",
        sandwich_bread="block_diagonal_copies_of_inverse_X_transpose_X",
    ),
    wald_hypotheses=OPEN02_WALD_HYPOTHESES,
    holm=Open02HolmContract(
        hypothesis_ids=("H_T", "H_P", "H_W"),
        familywise_alpha=0.05,
        all_hypotheses_required_for_promotion=True,
        agency_component_p_value_role="descriptive_only_not_a_promotion_family",
    ),
    influence=Open02InfluenceContract(
        groups=(
            Open02InfluenceGroupContract(
                group_id="g_T",
                hypothesis_id="H_T",
                coefficient_ids=("beta_B",),
            ),
            Open02InfluenceGroupContract(
                group_id="g_P",
                hypothesis_id="H_P",
                coefficient_ids=("beta_A", "beta_L", "beta_D"),
            ),
            Open02InfluenceGroupContract(
                group_id="g_W",
                hypothesis_id="H_W",
                coefficient_ids=("theta_A", "theta_L", "theta_D"),
            ),
        ),
        quarter_deletion_size=1,
        quarter_deletion_fits=96,
        block_deletion_size=4,
        block_deletion_fits=93,
        quarter_deletion_policy="delete_each_single_quarter_once",
        block_deletion_policy="delete_each_overlapping_contiguous_four_quarter_block",
        refit_policy="refit_entire_relevant_system_for_every_deletion",
        relative_l2_formula=(
            "l2_norm(refit_coefficients-full_coefficients)"
            "/max(l2_norm(full_coefficients),1e-12)"
        ),
        relative_l2_denominator_floor=1e-12,
        maximum_quarter_influence=0.25,
        maximum_block_influence=0.50,
        sign_stability_raw_p_threshold=0.05,
        sign_stability_policy=(
            "no deletion may reverse beta_B when significant, or any individually "
            "significant coefficient in g_P or g_W; Holm significance need not persist"
        ),
    ),
    lineage=Open02LineageContract(
        raw_treasury_ids=OPEN02_RAW_TREASURY_IDS,
        lineage_by_id=OPEN02_LINEAGE_BY_ID,
        no_double_side_policy=(
            "raw Treasury series with nonzero lineage on the left and right of every "
            "equation must be disjoint; canonical TDC is prohibited on the right"
        ),
    ),
    validity_gates=OPEN02_VALIDITY_GATES,
    output=OPEN02_OUTPUT,
    perimeter_sectors=(
        "us_chartered_depository_institutions",
        "foreign_banking_offices_in_us",
        "banks_in_us_affiliated_areas",
    ),
    excluded_sector="credit_unions",
    initial_status="valid_contract_but_not_yet_estimated",
    initial_main_text_eligible=False,
    initial_appendix_only="not_yet_determined",
    invalid_result_disposition=Open02ValidityDisposition(
        valid_result=False,
        main_text_eligible=False,
        appendix_only=False,
    ),
    valid_nonpromoted_disposition=Open02ValidityDisposition(
        valid_result=True,
        main_text_eligible=False,
        appendix_only=True,
    ),
    promoted_result_disposition=Open02ValidityDisposition(
        valid_result=True,
        main_text_eligible=True,
        appendix_only=False,
    ),
    promotion_reason_codes=(
        "source_anchor_holm_gt_0_05",
        "portfolio_joint_holm_gt_0_05",
        "within_joint_holm_gt_0_05",
        "leave_quarter_influence_gt_0_25",
        "leave_block_influence_gt_0_50",
        "sign_flip_under_influence",
    ),
    agency_composition_boundary=(
        "MBS composition is U.S.-chartered-only; the seven-part additive "
        "decomposition is not a uniform three-sector MBS taxonomy"
    ),
    us_chartered_sensitivity_claim_boundary=(
        "Association between U.S.-chartered portfolio transactions and an "
        "ex-bank-Treasury treatment that removes Treasury acquisitions of all three "
        "TDCest bank sectors."
    ),
    treatment_naming_boundary=(
        "The leave-out treatment is the accepted canonical construction net of one "
        "identified bank Treasury component and must not be called nonbank TDC."
    ),
    balance_sheet_boundary=(
        "The response system is not a closed balance sheet: do not force coefficients "
        "to add to one or infer that Treasury purchases were funded specifically by "
        "deposits, loan contraction, or agency sales."
    ),
    claim_boundary=(
        "In the fixed quarterly sample and conditional on the predeclared controls, "
        "the leave-out TDC measure, aggregate bank Treasury acquisitions, and other "
        "aggregate bank portfolio transactions exhibit the reported joint "
        "associations. This is not a causal portfolio-allocation equation, Treasury "
        "settlement landing estimate, retained-deposit share, financing-share "
        "decomposition, or independent mechanism evidence. The leave-out treatment "
        "must not be called nonbank TDC. This is not a closed balance sheet; "
        "coefficients must not be forced to add to one and do not identify which "
        "portfolio category funded Treasury purchases."
    ),
)

OPEN02_SERIES_BY_KEY: Mapping[str, Open02SeriesContract] = MappingProxyType(
    {series.key: series for series in OPEN02_SERIES}
)
OPEN02_SERIES_BY_FRED_ID: Mapping[str, Open02SeriesContract] = MappingProxyType(
    {series.fred_id: series for series in OPEN02_SERIES}
)


def get_open_contract() -> OpenContract:
    return OPEN_CONTRACT


def get_open02_contract() -> Open02Contract:
    return OPEN02_CONTRACT


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
    "OPEN02_AGENCY_COMPONENT_IDS",
    "OPEN02_BANK_AGENCY_ID",
    "OPEN02_BANK_DEPOSITS_ID",
    "OPEN02_BANK_LOANS_ID",
    "OPEN02_BANK_TREASURY_ID",
    "OPEN02_CONTRACT",
    "OPEN02_CONTROL_IDS",
    "OPEN02_CONTROLS",
    "OPEN02_FORMULAS",
    "OPEN02_LEAVE_OUT_ID",
    "OPEN02_LINEAGE_BY_ID",
    "OPEN02_RAW_TREASURY_IDS",
    "OPEN02_REJECTED_CONTROL_IDS",
    "OPEN02_REJECTED_PROXY_IDS",
    "OPEN02_SERIES",
    "OPEN02_SERIES_BY_FRED_ID",
    "OPEN02_SERIES_BY_KEY",
    "OPEN02_SYSTEMS",
    "OPEN02_OFFICIAL_SOURCE",
    "OPEN02_OUTPUT",
    "OPEN02_VALIDITY_GATES",
    "OPEN02_WALD_HYPOTHESES",
    "OPEN_CONTRACT",
    "OUTCOME_UNIT_MULTIPLIERS",
    "ROLLING_WINDOW_QUARTERS",
    "MethodTierContract",
    "Open02CoefficientSelectionContract",
    "Open02Contract",
    "Open02ControlContract",
    "Open02CovarianceContract",
    "Open02FormulaContract",
    "Open02HolmContract",
    "Open02InfluenceContract",
    "Open02InfluenceGroupContract",
    "Open02LineageContract",
    "Open02OfficialSourceContract",
    "Open02OutputContract",
    "Open02SampleContract",
    "Open02SeriesContract",
    "Open02SystemContract",
    "Open02ValidityDisposition",
    "Open02ValidityGateContract",
    "Open02WaldContract",
    "OpenContract",
    "TreasuryLineage",
    "get_open02_contract",
    "get_open_contract",
]
