from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from ea_tdc.open01 import (
    BREAK_QUARTER,
    CREDIT_ADJUSTMENTS,
    CREDIT_WINDOW_QUARTERS,
    _fit_break,
    _fit_projection,
    _fit_tier_interaction,
    build_open01_acceptance,
    build_treatment_outcome_contract,
    write_open01_outputs,
)
from ea_tdc.open_contract import (
    CANONICAL_CONTROL_IDS,
    CANONICAL_OUTCOME_ID,
    CANONICAL_RESIDUAL_ID,
    CANONICAL_TREATMENT_ID,
    CREDIT_SCREEN_OUTCOME_IDS,
    EXPECTED_METHOD_TIER_COUNTS,
    METHOD_TIER_SERIES_ID,
    MMF_TREATMENT_IDS,
    OPEN01_DESIGN_JOB_IDS,
)


def _quarter(start_year: int, index: int) -> str:
    ordinal = start_year * 4 + index
    return f"{ordinal // 4}Q{ordinal % 4 + 1}"


def _fixture_rows() -> list[dict[str, object]]:
    tier_ids: list[str] = []
    for tier_id, count in EXPECTED_METHOD_TIER_COUNTS.items():
        tier_ids.extend([tier_id] * count)
    assert len(tier_ids) == 96

    rows: list[dict[str, object]] = []
    for index, tier_id in enumerate(tier_ids):
        controls = {
            control_id: (
                math.sin((index + 1) * (control_index + 1) * 0.137)
                + math.cos((index + 2) * (control_index + 2) * 0.071)
                + 0.0001 * (control_index + 1) * (index + 1) ** 2
            )
            for control_index, control_id in enumerate(CANONICAL_CONTROL_IDS)
        }
        treatment = (
            25.0 * math.sin(index * 0.29)
            + 13.0 * math.cos(index * 0.11)
            + 0.4 * index
        )
        residual = (
            -0.45 * treatment
            + sum(
                (control_index + 1) * 0.004 * controls[control_id]
                for control_index, control_id in enumerate(
                    CANONICAL_CONTROL_IDS
                )
            )
            + 2.0 * math.sin(index * 0.53)
            + 1.5 * math.cos(index * 0.47)
        )
        row: dict[str, object] = {
            "quarter": _quarter(2002, index),
            METHOD_TIER_SERIES_ID: tier_id,
            CANONICAL_TREATMENT_ID: treatment,
            CANONICAL_RESIDUAL_ID: residual,
            CANONICAL_OUTCOME_ID: treatment + residual,
            MMF_TREATMENT_IDS[0]: 0.95 * treatment
            + 0.3 * math.sin(index * 0.41),
            MMF_TREATMENT_IDS[1]: treatment,
            MMF_TREATMENT_IDS[2]: 1.05 * treatment
            + 0.3 * math.cos(index * 0.37),
            **controls,
        }
        for outcome_index, outcome_id in enumerate(
            CREDIT_SCREEN_OUTCOME_IDS
        ):
            pandemic = (
                1.0
                if _quarter(2002, index)
                in {
                    "2020Q1",
                    "2020Q2",
                    "2020Q3",
                    "2020Q4",
                    "2021Q1",
                    "2021Q2",
                    "2021Q3",
                    "2021Q4",
                }
                else 0.0
            )
            row[outcome_id] = (
                (0.002 + outcome_index * 0.0004) * treatment
                + 0.06
                * math.sin(
                    index * (0.19 + outcome_index * 0.017)
                    + outcome_index
                )
                + 0.002 * (outcome_index + 1) * index
                + 0.03 * (outcome_index + 1) * pandemic
            )
        rows.append(row)
    return rows


def _contract_rows() -> list[dict[str, object]]:
    adapter_manifest = {
        "bundle_hash": "a" * 64,
        "combined_input_hash": "b" * 64,
        "input_hashes": {
            "seed_bundle": "a" * 64,
            "regression_series": "c" * 64,
            "processed_estimates": "d" * 64,
        },
    }
    design_hashes = {
        job_id: hashlib.sha256(job_id.encode("utf-8")).hexdigest()
        for job_id in OPEN01_DESIGN_JOB_IDS
    }
    return build_treatment_outcome_contract(
        adapter_manifest=adapter_manifest,
        design_bundle_hashes=design_hashes,
    )


@pytest.fixture(scope="module")
def accepted_result():
    return build_open01_acceptance(
        _fixture_rows(),
        contract_rows=_contract_rows(),
    )


def test_contract_serializes_the_single_registry_and_exact_hashes() -> None:
    rows = _contract_rows()

    assert {row["outcome_id"] for row in rows} == {
        CANONICAL_OUTCOME_ID,
        CANONICAL_RESIDUAL_ID,
        *CREDIT_SCREEN_OUTCOME_IDS,
    }
    assert {row["treatment_id"] for row in rows} == {
        CANONICAL_TREATMENT_ID
    }
    assert all(row["contract_status"] == "frozen" for row in rows)
    assert all(row["tdcest_seed_bundle_sha256"] == "a" * 64 for row in rows)
    assert all(row["tdcest_combined_input_sha256"] == "b" * 64 for row in rows)
    assert all(row["embedded_bank_treasury_component_id"] for row in rows)
    assert all(row["clock"] and row["treatment_perimeter"] for row in rows)


def test_headline_identity_and_stability_suite_are_fixed_and_complete(
    accepted_result,
) -> None:
    assert accepted_result.producer_status == "pass", accepted_result.issues
    headline = accepted_result.headline_rows
    assert len(headline) == 2
    assert len(
        {
            (
                row["n"],
                row["sample_start"],
                row["sample_end"],
                row["sample_hash"],
                row["control_ids"],
            )
            for row in headline
        }
    ) == 1
    assert all(row["identity_status"] == "pass" for row in headline)
    assert all(abs(float(row["identity_gap"])) <= 1e-9 for row in headline)
    assert len(
        {
            (row["control_ids_used"], row["control_ids_rejected"])
            for row in headline
        }
    ) == 1

    stability_types = {
        row["test_type"] for row in accepted_result.stability_rows
    }
    assert {
        "fixed_sample_contract",
        "declared_break",
        "unknown_break_candidate",
        "unknown_break_scan_summary",
        "leave_quarter_out_influence",
        "leave_block_out_influence",
        "construction_tier_sensitivity",
        "mmf_bound_sensitivity",
        "mmf_bound_sensitivity_summary",
        "overall_gate",
    }.issubset(stability_types)
    declared = next(
        row
        for row in accepted_result.stability_rows
        if row.get("test_type") == "declared_break"
    )
    assert declared["break_quarter"] == BREAK_QUARTER
    assert declared["p_value_holm"] == declared["p_value_raw"]
    assert "stable_lte" in json.loads(declared["materiality_bands_json"])
    inferential = [
        row
        for row in accepted_result.stability_rows
        if row.get("p_value_raw") not in (None, "")
    ]
    assert inferential
    assert all(row.get("p_value_holm") not in (None, "") for row in inferential)


def test_credit_screen_is_exact_overlap_aware_and_window_stable(
    accepted_result,
) -> None:
    rows = accepted_result.credit_screen_rows
    assert len(rows) == (
        len(CREDIT_SCREEN_OUTCOME_IDS)
        * len(CREDIT_WINDOW_QUARTERS)
        * len(CREDIT_ADJUSTMENTS)
    )
    assert {row["credit_outcome_id"] for row in rows} == set(
        CREDIT_SCREEN_OUTCOME_IDS
    )
    assert {int(row["window_quarters"]) for row in rows} == set(
        CREDIT_WINDOW_QUARTERS
    )
    assert {row["adjustment"] for row in rows} == set(CREDIT_ADJUSTMENTS)
    expected_cells = {
        (outcome_id, window_quarters, adjustment)
        for outcome_id in CREDIT_SCREEN_OUTCOME_IDS
        for window_quarters in (40, 48, 60)
        for adjustment in (
            "raw",
            "share_2020_2021_adjusted",
            "linear_time_adjusted",
        )
    }
    observed_cells = [
        (
            row["credit_outcome_id"],
            int(row["window_quarters"]),
            row["adjustment"],
        )
        for row in rows
    ]
    assert set(observed_cells) == expected_cells
    assert len(observed_cells) == len(set(observed_cells))
    assert all(row["covariance_estimator"] == "newey_west" for row in rows)
    assert all(int(row["covariance_lags"]) > 0 for row in rows)
    assert all(row.get("p_value_holm") not in (None, "") for row in rows)
    assert all(row["rolling_control_patterns_json"] for row in rows)
    assert all(
        isinstance(row["sign_stable_40_48_60"], bool) for row in rows
    )
    assert all(
        row["last_window_end"]
        <= row["last_observed_treatment_outcome_quarter"]
        for row in rows
    )
    assert {
        row["admission_status"] for row in rows
    }.issubset({"main_text_eligible", "appendix_only"})


def test_pre_component_tier_interaction_reuses_existing_indicator_control() -> None:
    rows = _fixture_rows()
    tier_id = next(iter(EXPECTED_METHOD_TIER_COUNTS))
    indicator_control = CANONICAL_CONTROL_IDS[4]
    for row in rows:
        row[indicator_control] = (
            1.0 if row[METHOD_TIER_SERIES_ID] == tier_id else 0.0
        )

    estimate = _fit_tier_interaction(rows, tier_id=tier_id)

    assert estimate["tier_n"] == EXPECTED_METHOD_TIER_COUNTS[tier_id]
    assert estimate["complement_n"] == len(rows) - estimate["tier_n"]
    assert math.isfinite(float(estimate["beta_change"]))


def test_unknown_break_at_tier_boundary_reuses_complementary_control() -> None:
    rows = _fixture_rows()
    pre_tier_id = next(iter(EXPECTED_METHOD_TIER_COUNTS))
    indicator_control = CANONICAL_CONTROL_IDS[4]
    for row in rows:
        row[indicator_control] = (
            1.0 if row[METHOD_TIER_SERIES_ID] == pre_tier_id else 0.0
        )
    first_post_tier = rows[EXPECTED_METHOD_TIER_COUNTS[pre_tier_id]][
        "quarter"
    ]

    estimate = _fit_break(rows, break_quarter=str(first_post_tier))

    assert estimate["pre_n"] == EXPECTED_METHOD_TIER_COUNTS[pre_tier_id]
    assert math.isfinite(float(estimate["beta_change"]))


def test_rolling_projection_records_structurally_inactive_tier_control() -> None:
    rows = _fixture_rows()[-48:]
    indicator_control = CANONICAL_CONTROL_IDS[4]
    for row in rows:
        row[indicator_control] = 0.0

    estimate = _fit_projection(
        rows,
        treatment_id=CANONICAL_TREATMENT_ID,
        outcome_id=CANONICAL_OUTCOME_ID,
        control_ids=CANONICAL_CONTROL_IDS,
    )

    assert indicator_control not in estimate.control_ids_used
    assert indicator_control in estimate.control_ids_rejected
    assert estimate.n == 48


def test_writer_uses_literal_paths_and_pass_manifest(
    tmp_path: Path,
    accepted_result,
) -> None:
    paths = write_open01_outputs(
        accepted_result,
        root=tmp_path,
        producer_inputs={
            "tdcest": {
                "seed_bundle_sha256": "a" * 64,
                "combined_input_sha256": "b" * 64,
            },
            "design_bundles": {
                job_id: hashlib.sha256(job_id.encode("utf-8")).hexdigest()
                for job_id in OPEN01_DESIGN_JOB_IDS
            },
        },
    )

    assert paths["contract"] == (
        tmp_path / "outputs/tables/tdc_treatment_outcome_contract.csv"
    )
    assert paths["headline"] == (
        tmp_path / "outputs/tables/tdc_same_quarter_headline.csv"
    )
    assert paths["stability"] == (
        tmp_path / "output/reports/tier2_pass_through_stability_gate.csv"
    )
    assert paths["credit_screen"] == (
        tmp_path
        / "output/reports/tier2_pass_through_offset_rolling_beta_correlates.csv"
    )
    assert paths["manifest"] == (
        tmp_path / "output/manifests/open01_acceptance_summary.json"
    )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["status"] == "passed"
    assert manifest["scientific_status"] in {
        "stable",
        "review",
        "unstable",
    }
    assert manifest["acceptance_checks"]
    assert all(
        check["passed"]
        for check in manifest["acceptance_checks"].values()
    )
    assert set(manifest["outputs"]) == {
        "contract",
        "headline",
        "stability",
        "credit_screen",
    }
    assert all(
        len(record["sha256"]) == 64
        for record in manifest["outputs"].values()
    )


@pytest.mark.parametrize("mutation", ["duplicate", "unexpected_adjustment"])
def test_writer_rejects_non_cartesian_credit_screen(
    tmp_path: Path,
    accepted_result,
    mutation: str,
) -> None:
    credit_rows = [dict(row) for row in accepted_result.credit_screen_rows]
    if mutation == "duplicate":
        credit_rows[-1] = dict(credit_rows[0])
    else:
        credit_rows[-1]["adjustment"] = "undeclared_adjustment"
    corrupted = replace(
        accepted_result,
        credit_screen_rows=credit_rows,
    )

    paths = write_open01_outputs(
        corrupted,
        root=tmp_path,
    )

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    completeness = manifest["acceptance_checks"]["credit_screen_complete"]
    assert manifest["status"] == "failed"
    assert completeness["passed"] is False
    assert completeness["details"]["observed_rows"] == 45
    assert len(completeness["details"]["missing_cells"]) == 1
    if mutation == "duplicate":
        assert len(completeness["details"]["duplicate_cells"]) == 1
        assert completeness["details"]["unexpected_cells"] == []
    else:
        assert completeness["details"]["duplicate_cells"] == []
        assert len(completeness["details"]["unexpected_cells"]) == 1


def test_missing_credit_series_fails_closed_with_counts() -> None:
    missing_id = CREDIT_SCREEN_OUTCOME_IDS[-1]
    rows = [
        {key: value for key, value in row.items() if key != missing_id}
        for row in _fixture_rows()
    ]

    result = build_open01_acceptance(
        rows,
        contract_rows=_contract_rows(),
    )

    assert result.producer_status == "fail"
    assert result.coverage_counts[missing_id] == 0
    assert any(
        issue == f"required_numeric_series_absent:{missing_id}"
        for issue in result.issues
    )
    assert result.credit_screen_rows[0]["status"] == "failed"
