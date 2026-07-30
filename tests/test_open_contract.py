from __future__ import annotations

from pathlib import Path

import yaml

from ea_tdc.artifacts import (
    PAPER_TIER2_CONTROLS,
    PAPER_TIER2_MAIN_FIGURE_OUTCOMES,
    PAPER_TIER2_MAIN_TABLE_OUTCOMES,
    PAPER_TIER2_TREATMENT_ID,
)
from ea_tdc.designs.quarterly import BASELINE_SERIES_MAP, _load_jobs
from ea_tdc.open_contract import (
    CANONICAL_CONTROL_IDS,
    CANONICAL_OUTCOME_ID,
    CANONICAL_RESIDUAL_ID,
    CANONICAL_TREATMENT_ID,
    CANONICAL_TREATMENT_SOURCE_SERIES,
    CREDIT_SCREEN_OUTCOME_IDS,
    EMBEDDED_BANK_TREASURY_COMPONENT_ID,
    EXPECTED_METHOD_TIER_COUNTS,
    METHOD_TIER_SERIES_ID,
    MMF_TREATMENT_IDS,
    OPEN01_DESIGN_JOB_IDS,
    OPEN02_AGENCY_COMPONENT_IDS,
    OPEN02_BANK_AGENCY_ID,
    OPEN02_BANK_DEPOSITS_ID,
    OPEN02_BANK_LOANS_ID,
    OPEN02_BANK_TREASURY_ID,
    OPEN02_CONTROL_IDS,
    OPEN02_CONTROLS,
    OPEN02_LEAVE_OUT_ID,
    OPEN02_OFFICIAL_SOURCE,
    OPEN02_OUTPUT,
    OPEN02_REJECTED_CONTROL_IDS,
    OPEN02_REJECTED_PROXY_IDS,
    OPEN02_SERIES_BY_FRED_ID,
    OPEN02_SERIES_BY_KEY,
    OPEN02_VALIDITY_GATES,
    OUTCOME_UNIT_MULTIPLIERS,
    ROLLING_WINDOW_QUARTERS,
    get_open02_contract,
    get_open_contract,
)


def test_open_contract_freezes_literal_treatment_outcome_and_stability_surface() -> None:
    contract = get_open_contract()

    assert CANONICAL_TREATMENT_ID == "tdc_tier2_regression_mmf_rrp_prop_bank_only_qoq"
    assert CANONICAL_TREATMENT_SOURCE_SERIES == (
        "tdc_tier2_regression_mmf_rrp_prop_bank_only_ru_flow"
    )
    assert CANONICAL_RESIDUAL_ID == (
        "other_component_tier2_regression_mmf_rrp_prop_bank_only_qoq"
    )
    assert CANONICAL_OUTCOME_ID == "matched_total_deposits"
    assert MMF_TREATMENT_IDS == (
        "tdc_tier2_regression_mmf_rrp_lb_bank_only_qoq",
        CANONICAL_TREATMENT_ID,
        "tdc_tier2_regression_mmf_rrp_ub_bank_only_qoq",
    )
    assert CANONICAL_CONTROL_IDS == (
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
    )
    assert ROLLING_WINDOW_QUARTERS == 48
    assert METHOD_TIER_SERIES_ID == "tier2_regression_bank_row_method_tier"
    assert dict(EXPECTED_METHOD_TIER_COUNTS) == {
        "pre_component_h15_scaled_backcast": 33,
        "component_pool_wamest_bucket_backcast": 47,
        "constrained_component": 16,
    }
    assert sum(EXPECTED_METHOD_TIER_COUNTS.values()) == 96
    assert CREDIT_SCREEN_OUTCOME_IDS == (
        "tdcpass_strict_loan_core_min_qoq",
        "tdcpass_strict_loan_mortgages_qoq",
        "tdcpass_strict_loan_consumer_credit_qoq",
        "bank_credit_qoq",
        "bank_business_loans_qoq",
    )
    assert dict(OUTCOME_UNIT_MULTIPLIERS) == {
        CANONICAL_OUTCOME_ID: 1.0,
        CANONICAL_RESIDUAL_ID: 1.0,
        **{outcome_id: 1000.0 for outcome_id in CREDIT_SCREEN_OUTCOME_IDS},
    }
    assert EMBEDDED_BANK_TREASURY_COMPONENT_ID == "bank_depository_tsy_tx"
    assert contract.mmf_proportional_treatment_id == CANONICAL_TREATMENT_ID
    assert contract.treatment_units == "usd_millions_per_quarter"
    assert contract.deposit_outcome_units == "usd_millions_per_quarter"
    assert contract.credit_outcome_units == "usd_billions_per_quarter"
    assert contract.sign_convention == "positive_means_deposit_positive_tdc"
    assert "h0_is_same_quarter" in contract.clock
    assert "fail closed" in contract.construction_tier_policy
    assert "not causal" in contract.claim_boundary


def test_open01_blueprint_jobs_share_canonical_treatment_without_release_activation() -> None:
    project_root = Path(__file__).resolve().parents[1]
    jobs = _load_jobs(project_root / "config" / "dass_job_blueprint.yaml")

    assert OPEN01_DESIGN_JOB_IDS == (
        "tdc_tier2_mmf_rrp_canonical_full_panel",
        "tdc_tier2_regression_deposit_anatomy",
        "tdc_tier2_regression_credit_anatomy",
        "tdc_tier2_regression_plumbing_rates",
    )
    for job_id in OPEN01_DESIGN_JOB_IDS:
        assert jobs[job_id]["treatment_id"] == CANONICAL_TREATMENT_ID
        assert jobs[job_id]["track_in_release_snapshot"] is False
        assert jobs[job_id]["track_in_estimation_snapshot"] is False

    assert BASELINE_SERIES_MAP[CANONICAL_TREATMENT_ID] == (
        "tdcest",
        CANONICAL_TREATMENT_SOURCE_SERIES,
    )
    assert BASELINE_SERIES_MAP[METHOD_TIER_SERIES_ID] == (
        "tdcest",
        METHOD_TIER_SERIES_ID,
    )
    assert all(treatment_id in BASELINE_SERIES_MAP for treatment_id in MMF_TREATMENT_IDS)


def test_paper_artifact_constants_consume_open_contract() -> None:
    assert PAPER_TIER2_TREATMENT_ID == CANONICAL_TREATMENT_ID
    assert tuple(PAPER_TIER2_CONTROLS) == CANONICAL_CONTROL_IDS
    assert PAPER_TIER2_MAIN_FIGURE_OUTCOMES == [
        CANONICAL_OUTCOME_ID,
        CANONICAL_RESIDUAL_ID,
    ]
    assert PAPER_TIER2_MAIN_TABLE_OUTCOMES == [
        CANONICAL_OUTCOME_ID,
        CANONICAL_RESIDUAL_ID,
        *CREDIT_SCREEN_OUTCOME_IDS,
    ]


def test_public_treatment_registry_is_an_export_of_the_runtime_contract() -> None:
    project_root = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load(
        (project_root / "docs" / "04_treatment_registry.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert payload["registry_role"] == "published_export"
    assert payload["runtime_authority"] == "src/ea_tdc/open_contract.py"
    exported = {
        row["treatment_id"]: row for row in payload["treatments"]
    }[CANONICAL_TREATMENT_ID]
    assert exported["upstream_object"] == CANONICAL_TREATMENT_SOURCE_SERIES
    assert exported["units"] == get_open_contract().treatment_units
    assert exported["sign_convention"] == get_open_contract().sign_convention
    assert exported["family"] == "canonical_open01_treatment"


def test_open02_contract_freezes_exact_twenty_series_allowlist() -> None:
    contract = get_open02_contract()
    expected = {
        "tsy_us": (
            "BOGZ1FU763061100Q",
            "FU763061100.Q",
            "us_chartered_depository_institutions",
            "asset",
            "U.S.-Chartered Depository Institutions; Treasury Securities; Asset, Transactions",
        ),
        "tsy_fbo": (
            "BOGZ1FU753061103Q",
            "FU753061103.Q",
            "foreign_banking_offices_in_us",
            "asset",
            "Foreign Banking Offices in the U.S.; Treasury Securities; Asset, Transactions",
        ),
        "tsy_aff": (
            "BOGZ1FU743061103Q",
            "FU743061103.Q",
            "banks_in_us_affiliated_areas",
            "asset",
            "Banks in U.S.-Affiliated Areas; Treasury Securities; Asset, Transactions",
        ),
        "agency_us_total": (
            "BOGZ1FU763061705Q",
            "FU763061705.Q",
            "us_chartered_depository_institutions",
            "asset",
            "U.S.-Chartered Depository Institutions; Agency- And GSE-Backed Securities; Asset, Transactions",
        ),
        "agency_fbo_total": (
            "BOGZ1FU753061703Q",
            "FU753061703.Q",
            "foreign_banking_offices_in_us",
            "asset",
            "Foreign Banking Offices in the U.S.; Agency- And GSE-Backed Securities; Asset, Transactions",
        ),
        "agency_aff_total": (
            "BOGZ1FU743061703Q",
            "FU743061703.Q",
            "banks_in_us_affiliated_areas",
            "asset",
            "Banks in U.S.-Affiliated Areas; Agency- And GSE-Backed Securities; Asset, Transactions",
        ),
        "agency_us_res_pass": (
            "BOGZ1FU763061805Q",
            "FU763061805.Q",
            "us_chartered_depository_institutions",
            "asset",
            "U.S.-Chartered Depository Institutions; Total Agency Issued Residential Mortgage Pass-Through Securities; Asset, Transactions",
        ),
        "agency_us_com_pass": (
            "BOGZ1FU763061303Q",
            "FU763061503.Q",
            "us_chartered_depository_institutions",
            "asset",
            "U.S.-Chartered Depository Institutions; Agency Issued Commercial Mortgage Pass-Through Securities; Asset, Transactions",
        ),
        "agency_us_res_cmo": (
            "BOGZ1FU763061603Q",
            "FU763061603.Q",
            "us_chartered_depository_institutions",
            "asset",
            "U.S.-Chartered Depository Institutions; Agency Issued Residential CMOs and Other Structured MBS; Asset, Transactions",
        ),
        "agency_us_com_cmo": (
            "BOGZ1FU763061403Q",
            "FU763061403.Q",
            "us_chartered_depository_institutions",
            "asset",
            "U.S.-Chartered Depository Institutions; Agency Issued Commercial CMOs and Other Structured MBS; Asset, Transactions",
        ),
        "agency_us_other": (
            "BOGZ1FU763061795Q",
            "FU763061795.Q",
            "us_chartered_depository_institutions",
            "asset",
            "U.S.-Chartered Depository Institutions; Agency- And GSE-Backed Securities, Excluding MBS and CMOs; Asset, Transactions",
        ),
        "loans_us": (
            "BOGZ1FU764023005Q",
            "FU764023005.Q",
            "us_chartered_depository_institutions",
            "asset",
            "U.S.-Chartered Depository Institutions; Loans; Asset, Transactions",
        ),
        "loans_fbo": (
            "BOGZ1FU754023005Q",
            "FU754023005.Q",
            "foreign_banking_offices_in_us",
            "asset",
            "Foreign Banking Offices in the U.S.; Loans; Asset, Transactions",
        ),
        "loans_aff": (
            "BOGZ1FU744023003Q",
            "FU744023003.Q",
            "banks_in_us_affiliated_areas",
            "asset",
            "Banks in U.S.-Affiliated Areas; Loans; Asset, Transactions",
        ),
        "dep_us_check": (
            "BOGZ1FU763127005Q",
            "FU763127005.Q",
            "us_chartered_depository_institutions",
            "liability",
            "U.S.-Chartered Depository Institutions; Checkable Deposits; Liability, Transactions",
        ),
        "dep_us_time": (
            "BOGZ1FU763130005Q",
            "FU763130005.Q",
            "us_chartered_depository_institutions",
            "liability",
            "U.S.-Chartered Depository Institutions, Including IBFs; Total Time and Savings Deposits; Liability, Transactions",
        ),
        "dep_fbo_check": (
            "BOGZ1FU753127005Q",
            "FU753127005.Q",
            "foreign_banking_offices_in_us",
            "liability",
            "Foreign Banking Offices in the U.S.; Checkable Deposits; Liability, Transactions",
        ),
        "dep_fbo_time": (
            "BOGZ1FU753130005Q",
            "FU753130005.Q",
            "foreign_banking_offices_in_us",
            "liability",
            "Foreign Banking Offices in the U.S., Including IBFs; Total Time and Savings Deposits; Liability, Transactions",
        ),
        "dep_aff_check": (
            "BOGZ1FU743127003Q",
            "FU743127003.Q",
            "banks_in_us_affiliated_areas",
            "liability",
            "Banks in U.S.-Affiliated Areas; Checkable Deposits; Liability, Transactions",
        ),
        "dep_aff_time": (
            "BOGZ1FU743130003Q",
            "FU743130003.Q",
            "banks_in_us_affiliated_areas",
            "liability",
            "Banks in U.S.-Affiliated Areas; Total Time and Savings Deposits; Liability, Transactions",
        ),
    }

    assert len(contract.series) == 20
    assert set(OPEN02_SERIES_BY_KEY) == set(expected)
    assert len(OPEN02_SERIES_BY_FRED_ID) == 20
    for key, (fred_id, board_id, sector, side, title) in expected.items():
        series = OPEN02_SERIES_BY_KEY[key]
        assert (
            series.fred_id,
            series.board_series_id,
            series.sector,
            series.side,
            series.official_title,
        ) == (fred_id, board_id, sector, side, title)
        assert series.frequency == "quarterly"
        assert series.units == "millions_of_us_dollars"
        assert series.financial_accounts_type == "FU_transaction"
        assert series.seasonal_adjustment == "not_seasonally_adjusted"
        assert series.primary_transform == "none"

    exceptions = [series for series in contract.series if series.board_locator_exception]
    assert [(series.fred_id, series.board_series_id) for series in exceptions] == [
        ("BOGZ1FU763061303Q", "FU763061503.Q")
    ]
    assert "BOGZ1FU763061503Q" not in OPEN02_SERIES_BY_FRED_ID
    treasury_keys = {"tsy_us", "tsy_fbo", "tsy_aff"}
    loan_keys = {"loans_us", "loans_fbo", "loans_aff"}
    deposit_keys = {
        "dep_us_check",
        "dep_us_time",
        "dep_fbo_check",
        "dep_fbo_time",
        "dep_aff_check",
        "dep_aff_time",
    }
    assert {
        series.key
        for series in contract.series
        if series.role == "treasury_component"
    } == treasury_keys
    assert {
        series.key for series in contract.series if series.role == "loan_component"
    } == loan_keys
    assert {
        series.key
        for series in contract.series
        if series.role == "customer_deposit_component"
    } == deposit_keys
    assert all(
        OPEN02_SERIES_BY_KEY[key].positive_sign == "net_acquisition"
        for key in treasury_keys
    )
    assert all(
        OPEN02_SERIES_BY_KEY[key].positive_sign
        == "net_acquisition_or_origination"
        for key in loan_keys
    )
    assert all(
        OPEN02_SERIES_BY_KEY[key].positive_sign == "net_liability_increase"
        for key in deposit_keys
    )
    assert {
        key: OPEN02_SERIES_BY_KEY[key].treasury_lineage for key in treasury_keys
    } == {
        "tsy_us": (1, 0, 0),
        "tsy_fbo": (0, 1, 0),
        "tsy_aff": (0, 0, 1),
    }
    assert all(
        series.treasury_lineage == (0, 0, 0)
        for series in contract.series
        if series.key not in treasury_keys
    )


def test_open02_contract_is_unique_and_points_to_open01_authority() -> None:
    contract = get_open02_contract()

    assert contract.open01_contract is get_open_contract()
    assert contract.canonical_treatment_id == CANONICAL_TREATMENT_ID
    assert contract.canonical_treatment_source_series == CANONICAL_TREATMENT_SOURCE_SERIES
    assert (
        contract.embedded_bank_treasury_component_id
        == EMBEDDED_BANK_TREASURY_COMPONENT_ID
    )
    assert len({series.key for series in contract.series}) == 20
    assert len({series.fred_id for series in contract.series}) == 20
    assert len({series.board_series_id for series in contract.series}) == 20
    assert contract.control_specs is contract.controls
    assert contract.control_contracts is contract.controls
    assert contract.influence_groups is contract.influence.groups
    assert contract.official_source is contract.source
    assert contract.perimeter_sectors == (
        "us_chartered_depository_institutions",
        "foreign_banking_offices_in_us",
        "banks_in_us_affiliated_areas",
    )
    assert contract.excluded_sector == "credit_unions"
    assert contract.initial_status == "valid_contract_but_not_yet_estimated"
    assert contract.initial_main_text_eligible is False
    assert contract.initial_appendix_only == "not_yet_determined"


def test_open02_contract_freezes_formulas_systems_and_lineage() -> None:
    contract = get_open02_contract()
    formulas = {formula.formula_id: formula for formula in contract.formulas}
    assert tuple(formulas) == (
        "bank_treasury_three_sector_sum",
        "accepted_bank_treasury_component_reconciliation",
        "leave_out_definition",
        "canonical_reconstruction",
        "agency_us_five_component_identity",
        "agency_three_sector_identity",
        "agency_seven_component_identity",
        "loans_three_sector_identity",
        "customer_deposit_identity",
        "source_agency_coefficient_adding_up",
        "within_agency_coefficient_adding_up",
    )

    assert formulas["bank_treasury_three_sector_sum"].terms == (
        ("tsy_us", 1.0),
        ("tsy_fbo", 1.0),
        ("tsy_aff", 1.0),
    )
    assert formulas["accepted_bank_treasury_component_reconciliation"].output_id == (
        EMBEDDED_BANK_TREASURY_COMPONENT_ID
    )
    assert formulas["leave_out_definition"].terms == (
        (CANONICAL_TREATMENT_SOURCE_SERIES, 1.0),
        (OPEN02_BANK_TREASURY_ID, -1.0),
    )
    assert formulas["canonical_reconstruction"].terms == (
        (OPEN02_LEAVE_OUT_ID, 1.0),
        (OPEN02_BANK_TREASURY_ID, 1.0),
    )
    assert formulas["agency_us_five_component_identity"].terms == (
        ("agency_us_res_pass", 1.0),
        ("agency_us_com_pass", 1.0),
        ("agency_us_res_cmo", 1.0),
        ("agency_us_com_cmo", 1.0),
        ("agency_us_other", 1.0),
    )
    assert formulas["agency_seven_component_identity"].terms == tuple(
        (series_id, 1.0) for series_id in OPEN02_AGENCY_COMPONENT_IDS
    )
    assert formulas["loans_three_sector_identity"].terms == (
        ("loans_us", 1.0),
        ("loans_fbo", 1.0),
        ("loans_aff", 1.0),
    )
    assert formulas["customer_deposit_identity"].terms == (
        ("dep_us_check", 1.0),
        ("dep_us_time", 1.0),
        ("dep_fbo_check", 1.0),
        ("dep_fbo_time", 1.0),
        ("dep_aff_check", 1.0),
        ("dep_aff_time", 1.0),
    )
    assert all(
        formula.tolerance == 1e-6
        for formula in contract.formulas
        if formula.tolerance_kind == "absolute_usd_millions"
    )
    assert all(
        formula.tolerance == 1e-10
        for formula in contract.formulas
        if formula.tolerance_kind == "relative_max_one_or_coefficients"
    )
    assert {
        formulas[formula_id].tolerance_scale_formula
        for formula_id in (
            "source_agency_coefficient_adding_up",
            "within_agency_coefficient_adding_up",
        )
    } == {
        "max(1, abs(aggregate_coefficient), abs(sum_component_coefficients))"
    }

    systems = {system.system_id: system for system in contract.systems}
    assert tuple(systems) == (
        "source_side_response",
        "within_bank_conditional_co_movement",
    )
    assert systems["source_side_response"].outcome_ids == (
        OPEN02_BANK_TREASURY_ID,
        OPEN02_BANK_AGENCY_ID,
        OPEN02_BANK_LOANS_ID,
        OPEN02_BANK_DEPOSITS_ID,
    )
    assert systems["source_side_response"].regressor_ids == (OPEN02_LEAVE_OUT_ID,)
    assert systems["source_side_response"].design_parameter_count == 13
    assert systems["within_bank_conditional_co_movement"].outcome_ids == (
        OPEN02_BANK_AGENCY_ID,
        OPEN02_BANK_LOANS_ID,
        OPEN02_BANK_DEPOSITS_ID,
    )
    assert systems["within_bank_conditional_co_movement"].regressor_ids == (
        OPEN02_BANK_TREASURY_ID,
        OPEN02_LEAVE_OUT_ID,
    )
    assert systems["within_bank_conditional_co_movement"].design_parameter_count == 14
    assert all(
        system.agency_component_outcome_ids == OPEN02_AGENCY_COMPONENT_IDS
        for system in systems.values()
    )
    for hypothesis in contract.wald_hypotheses:
        system = systems[hypothesis.system_id]
        assert tuple(
            selection.coefficient_id for selection in hypothesis.selections
        ) == hypothesis.coefficient_ids
        assert hypothesis.degrees_of_freedom == len(hypothesis.selections)
        for selection in hypothesis.selections:
            assert system.outcome_ids[selection.equation_index] == selection.outcome_id
            assert (
                system.regressor_ids[selection.coefficient_index - 1]
                == selection.regressor_id
            )
            assert selection.null_value == 0.0
    assert {
        hypothesis.hypothesis_id: (
            hypothesis.system_id,
            tuple(
                (
                    selection.outcome_id,
                    selection.regressor_id,
                    selection.equation_index,
                    selection.coefficient_index,
                )
                for selection in hypothesis.selections
            ),
        )
        for hypothesis in contract.wald_hypotheses
    } == {
        "H_T": (
            "source_side_response",
            ((OPEN02_BANK_TREASURY_ID, OPEN02_LEAVE_OUT_ID, 0, 1),),
        ),
        "H_P": (
            "source_side_response",
            (
                (OPEN02_BANK_AGENCY_ID, OPEN02_LEAVE_OUT_ID, 1, 1),
                (OPEN02_BANK_LOANS_ID, OPEN02_LEAVE_OUT_ID, 2, 1),
                (OPEN02_BANK_DEPOSITS_ID, OPEN02_LEAVE_OUT_ID, 3, 1),
            ),
        ),
        "H_W": (
            "within_bank_conditional_co_movement",
            (
                (OPEN02_BANK_AGENCY_ID, OPEN02_BANK_TREASURY_ID, 0, 1),
                (OPEN02_BANK_LOANS_ID, OPEN02_BANK_TREASURY_ID, 1, 1),
                (OPEN02_BANK_DEPOSITS_ID, OPEN02_BANK_TREASURY_ID, 2, 1),
            ),
        ),
    }

    lineage = dict(contract.lineage.lineage_by_id)
    assert contract.lineage.raw_treasury_ids == (
        "BOGZ1FU763061100Q",
        "BOGZ1FU753061103Q",
        "BOGZ1FU743061103Q",
    )
    assert lineage[CANONICAL_TREATMENT_ID] == (1, 1, 1)
    assert lineage[OPEN02_BANK_TREASURY_ID] == (1, 1, 1)
    assert lineage[OPEN02_LEAVE_OUT_ID] == (0, 0, 0)
    assert all(lineage[control_id] == (0, 0, 0) for control_id in OPEN02_CONTROL_IDS)


def test_open02_contract_freezes_sample_inference_and_promotion_gates() -> None:
    contract = get_open02_contract()

    assert OPEN02_CONTROL_IDS == (
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
    assert (
        contract.sample.start_quarter,
        contract.sample.end_quarter,
        contract.sample.observations,
        contract.sample.quarter_hash,
    ) == (
        "2002Q1",
        "2025Q4",
        96,
        "f0de664ba1588848933205da2fd01df64a864ed5696c307771a7eebf88d56713",
    )
    assert contract.sample.accepted_tdcest_bundle_generated_at == (
        "2026-05-22T16:56:46Z"
    )
    assert contract.sample.observation_vintage_cutoff == "2026-05-22T16:56:46Z"
    assert contract.sample.identity_tolerance_usd_millions == 1e-6
    assert contract.sample.coefficient_adding_up_relative_tolerance == 1e-10
    assert contract.sample.required_common_hashes == (
        "row_hash",
        "column_hash",
        "design_hash",
    )
    assert contract.sample.rank_policy == (
        "every_relevant_design_matrix_must_be_full_column_rank"
    )
    assert contract.covariance.coefficient_estimator == "equation_by_equation_ols"
    assert contract.covariance.estimator == "stacked_system_newey_west_hac"
    assert contract.covariance.kernel == "bartlett"
    assert contract.covariance.lag_quarters == 4
    assert contract.covariance.prewhitening == "none"
    assert contract.covariance.finite_sample_correction == "T/(T-K)"
    assert contract.covariance.test_sidedness == "two_sided"
    assert contract.covariance.score_definition == (
        "equation_major_stack_of_x_t_times_equation_residual"
    )
    assert contract.covariance.sandwich_bread == (
        "block_diagonal_copies_of_inverse_X_transpose_X"
    )
    assert [
        (test.hypothesis_id, test.coefficient_ids, test.degrees_of_freedom)
        for test in contract.wald_hypotheses
    ] == [
        ("H_T", ("beta_B",), 1),
        ("H_P", ("beta_A", "beta_L", "beta_D"), 3),
        ("H_W", ("theta_A", "theta_L", "theta_D"), 3),
    ]
    assert contract.holm.hypothesis_ids == ("H_T", "H_P", "H_W")
    assert contract.holm.familywise_alpha == 0.05
    assert contract.holm.all_hypotheses_required_for_promotion is True
    assert contract.influence.quarter_deletion_fits == 96
    assert contract.influence.block_deletion_size == 4
    assert contract.influence.block_deletion_fits == 93
    assert contract.influence.maximum_quarter_influence == 0.25
    assert contract.influence.maximum_block_influence == 0.50
    assert contract.influence.relative_l2_denominator_floor == 1e-12
    assert contract.influence.sign_stability_raw_p_threshold == 0.05
    assert [
        (group.group_id, group.hypothesis_id, group.coefficient_ids)
        for group in contract.influence_groups
    ] == [
        ("g_T", "H_T", ("beta_B",)),
        ("g_P", "H_P", ("beta_A", "beta_L", "beta_D")),
        ("g_W", "H_W", ("theta_A", "theta_L", "theta_D")),
    ]
    assert contract.influence.quarter_deletion_policy == (
        "delete_each_single_quarter_once"
    )
    assert contract.influence.block_deletion_policy == (
        "delete_each_overlapping_contiguous_four_quarter_block"
    )
    assert contract.influence.refit_policy == (
        "refit_entire_relevant_system_for_every_deletion"
    )
    assert contract.influence.relative_l2_formula == (
        "l2_norm(refit_coefficients-full_coefficients)"
        "/max(l2_norm(full_coefficients),1e-12)"
    )


def test_open02_contract_freezes_control_source_gate_and_output_obligations() -> None:
    contract = get_open02_contract()

    assert contract.controls == OPEN02_CONTROLS
    assert tuple(spec.control_id for spec in contract.controls) == OPEN02_CONTROL_IDS
    assert len({spec.control_id for spec in contract.controls}) == 11
    assert all(spec.treasury_lineage == (0, 0, 0) for spec in contract.controls)
    assert {
        spec.control_id: (
            spec.source_series_id,
            spec.lag_quarters,
            spec.indicator_quarter,
            spec.source_policy,
            spec.generation_policy,
        )
        for spec in contract.controls
    } == {
        **{
            control_id: (
                control_id,
                0,
                None,
                "accepted_open01_design_rows",
                "reuse_exact_value_no_refetch_no_retransform",
            )
            for control_id in OPEN02_CONTROL_IDS[:5]
        },
        "dgs2__lag_4": (
            "dgs2",
            4,
            None,
            "accepted_open01_design_history",
            "exact_four_quarter_row_lag_no_refetch",
        ),
        "dgs10__lag_1": (
            "dgs10",
            1,
            None,
            "accepted_open01_design_history",
            "exact_one_quarter_row_lag_no_refetch",
        ),
        "dgs10__lag_2": (
            "dgs10",
            2,
            None,
            "accepted_open01_design_history",
            "exact_two_quarter_row_lag_no_refetch",
        ),
        **{
            f"quarter_is_q{quarter}": (
                None,
                0,
                quarter,
                "deterministic_from_ordered_quarter",
                f"indicator_quarter_equals_q{quarter}_with_q1_omitted",
            )
            for quarter in (2, 3, 4)
        },
    }

    assert contract.source is OPEN02_OFFICIAL_SOURCE
    assert contract.source.release_date == "2026-03-19"
    assert contract.source.archive_url == (
        "https://www.federalreserve.gov/releases/z1/20260319/"
        "z1_csv_files.zip"
    )
    assert contract.source.archive_sha256 == (
        "4a758a65a5190987a53e24039d91cc2b09ed55e57a2560bc640fdfe191ceee35"
    )
    assert contract.source.unit_label == (
        "Millions of dollars; transactions, not seasonally adjusted"
    )
    assert dict(contract.source.csv_member_sha256) == {
        "csv/fu111.csv": (
            "2dc83502d138e5253117a784c28b8fbeeba0e1460db2439ad243c116c1de9a11"
        ),
        "csv/fu112.csv": (
            "b016e3d742f4dffd61b12947a2e64605c3338c71507297d925f61f4980f7bae7"
        ),
        "csv/fu113.csv": (
            "e25f4a6843c428bd120fc416974578765f14253a76d0fb07c79369c46c7df952"
        ),
    }
    assert dict(contract.source.dictionary_member_sha256) == {
        "data_dictionary/fu111.txt": (
            "df992f6d6868a8665023f558018a2baf69b24c32933c34221d31acae8c82f1f7"
        ),
        "data_dictionary/fu112.txt": (
            "f91d4217bf99636658068456ccbb7d9c7c9fabcf268d886238fdd182ad4bd818"
        ),
        "data_dictionary/fu113.txt": (
            "2e51ba3a7b9f24a236ba4ba8dee57da04c8d61deb86f657d7e02f2a8f3873155"
        ),
    }
    all_source_hashes = (
        contract.source.archive_sha256,
        *(value for _, value in contract.source.csv_member_sha256),
        *(value for _, value in contract.source.dictionary_member_sha256),
    )
    assert all(
        len(value) == 64 and value == value.lower() and int(value, 16) >= 0
        for value in all_source_hashes
    )

    assert contract.validity_gates == OPEN02_VALIDITY_GATES
    assert [
        (gate.gate_id, gate.reason_code) for gate in contract.validity_gates
    ] == [
        ("metadata", "metadata_gate_failed"),
        ("vintage", "vintage_gate_failed"),
        ("coverage", "coverage_gate_failed"),
        ("treasury_component", "treasury_component_gate_failed"),
        (
            "accepted_component_reconciliation",
            "accepted_component_reconciliation_failed",
        ),
        ("leave_out_reconstruction", "leave_out_reconstruction_failed"),
        ("us_agency_identity", "us_agency_identity_failed"),
        ("three_sector_agency_identity", "three_sector_agency_identity_failed"),
        ("coefficient_adding_up", "coefficient_adding_up_failed"),
        ("common_sample_design", "common_sample_design_failed"),
        ("rank", "rank_gate_failed"),
        ("lineage", "lineage_gate_failed"),
    ]
    assert len({gate.gate_id for gate in contract.validity_gates}) == 12
    assert len({gate.reason_code for gate in contract.validity_gates}) == 12
    assert all(gate.requirement for gate in contract.validity_gates)

    assert contract.output is OPEN02_OUTPUT
    assert contract.output.report_path == (
        "output/reports/tier2_bank_portfolio_response_system.csv"
    )
    assert contract.output.receipt_path == (
        "output/manifests/open02_producer_run_receipt.json"
    )
    assert {
        "producer_commit",
        "run_id",
        "source.input_hashes",
        "outputs.response_system.sha256",
        "units",
        "sample",
        "acceptance.gates",
        "acceptance.identity_errors",
        "acceptance.row_hash",
        "acceptance.column_hashes",
        "acceptance.design_hashes",
        "acceptance.wald_statistics",
        "acceptance.holm_adjusted_p_values",
        "acceptance.influence_maxima",
        "acceptance.valid_result",
        "acceptance.main_text_eligible",
        "acceptance.appendix_only",
        "acceptance.reason_codes",
    }.issubset(contract.output.required_receipt_fields)


def test_open02_contract_rejects_proxies_and_unsafe_controls() -> None:
    contract = get_open02_contract()
    allowed_ids = set(OPEN02_SERIES_BY_FRED_ID)

    assert OPEN02_REJECTED_CONTROL_IDS == (
        "tdcpass_strict_loan_core_min_qoq__lag_2",
        "tdcpass_strict_loan_core_min_qoq__lag_4",
        "tdcpass_strict_loan_consumer_credit_qoq__lag_4",
        "bank_credit_qoq__lag_4",
        "dflmx_k100_f1",
        "dflmx_k100_f2",
        "dflmx_k100_f3",
        "dflmx_k100_f4",
    )
    assert set(OPEN02_CONTROL_IDS).isdisjoint(OPEN02_REJECTED_CONTROL_IDS)
    assert set(OPEN02_CONTROL_IDS).isdisjoint(OPEN02_REJECTED_PROXY_IDS)
    assert allowed_ids.isdisjoint(OPEN02_REJECTED_PROXY_IDS)
    assert {
        "BOGZ1FL764100005Q",
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
    }.issubset(contract.rejected_proxy_ids)
    assert all(series.fred_id.startswith("BOGZ1FU") for series in contract.series)
    assert OPEN02_SERIES_BY_KEY["tsy_us"].fred_id in allowed_ids
    assert "BOGZ1FU763061100Q" not in contract.rejected_proxy_ids
    assert (
        contract.invalid_result_disposition.valid_result,
        contract.invalid_result_disposition.main_text_eligible,
        contract.invalid_result_disposition.appendix_only,
    ) == (False, False, False)
    assert (
        contract.valid_nonpromoted_disposition.valid_result,
        contract.valid_nonpromoted_disposition.main_text_eligible,
        contract.valid_nonpromoted_disposition.appendix_only,
    ) == (True, False, True)
    assert (
        contract.promoted_result_disposition.valid_result,
        contract.promoted_result_disposition.main_text_eligible,
        contract.promoted_result_disposition.appendix_only,
    ) == (True, True, False)
    assert contract.promotion_reason_codes == (
        "source_anchor_holm_gt_0_05",
        "portfolio_joint_holm_gt_0_05",
        "within_joint_holm_gt_0_05",
        "leave_quarter_influence_gt_0_25",
        "leave_block_influence_gt_0_50",
        "sign_flip_under_influence",
    )
    assert contract.us_chartered_sensitivity_claim_boundary == (
        "Association between U.S.-chartered portfolio transactions and an "
        "ex-bank-Treasury treatment that removes Treasury acquisitions of all three "
        "TDCest bank sectors."
    )
    assert "not a uniform three-sector MBS taxonomy" in (
        contract.agency_composition_boundary
    )
    assert "must not be called nonbank TDC" in contract.treatment_naming_boundary
    assert "do not force coefficients to add to one" in contract.balance_sheet_boundary
    assert "funded specifically by deposits" in contract.balance_sheet_boundary
    for required_boundary in (
        "not a causal portfolio-allocation equation",
        "Treasury settlement landing estimate",
        "retained-deposit share",
        "financing-share decomposition",
        "independent mechanism evidence",
        "must not be called nonbank TDC",
        "not a closed balance sheet",
        "must not be forced to add to one",
        "do not identify which portfolio category funded Treasury purchases",
    ):
        assert required_boundary in contract.claim_boundary
