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
    OUTCOME_UNIT_MULTIPLIERS,
    ROLLING_WINDOW_QUARTERS,
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
