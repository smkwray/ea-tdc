from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import asdict, replace

import pytest

from ea_tdc.open02 import (
    Open02ValidationError,
    evaluate_open02_eligibility,
    run_open02_pipeline,
)
from ea_tdc.open_contract import (
    OPEN02_CONTRACT,
    OPEN02_LEAVE_OUT_ID,
)


def _quarter(year: int, quarter: int) -> str:
    return f"{year:04d}Q{quarter}"


def _period_end(year: int, quarter: int) -> str:
    return {
        1: f"{year:04d}-03-31",
        2: f"{year:04d}-06-30",
        3: f"{year:04d}-09-30",
        4: f"{year:04d}-12-31",
    }[quarter]


def _valid_fixture() -> tuple[dict, list[dict], list[dict]]:
    rng = random.Random(20260730)
    history_quarters = [
        _quarter(year, quarter)
        for year in range(2001, 2026)
        for quarter in range(1, 5)
    ]
    target_quarters = history_quarters[4:]
    design_rows: list[dict] = []
    history_values: dict[str, dict[str, float]] = {}
    for index, quarter in enumerate(history_quarters):
        year = int(quarter[:4])
        quarter_number = int(quarter[-1])
        values = {
            "GDP": rng.uniform(-2.0, 2.0),
            "gdp_deflator": rng.uniform(-2.0, 2.0),
            "FEDFUNDS": rng.uniform(-2.0, 2.0),
            "TOTRESNS": rng.uniform(-2.0, 2.0),
            "tier2_regression_bank_row_tier_pre_component_h15_scaled": (
                rng.uniform(-2.0, 2.0)
            ),
            "dgs2": rng.uniform(-2.0, 2.0),
            "dgs10": rng.uniform(-2.0, 2.0),
            "X": rng.uniform(-20.0, 20.0),
            "b_noise": rng.uniform(-5.0, 5.0),
            "a_noise": rng.uniform(-2.0, 2.0),
            "l_noise": rng.uniform(-2.0, 2.0),
            "d_noise": rng.uniform(-2.0, 2.0),
        }
        history_values[quarter] = values
        design_rows.append(
            {
                "quarter": quarter,
                "period_end": _period_end(year, quarter_number),
                "GDP": values["GDP"],
                "gdp_deflator": values["gdp_deflator"],
                "FEDFUNDS": values["FEDFUNDS"],
                "TOTRESNS": values["TOTRESNS"],
                "tier2_regression_bank_row_tier_pre_component_h15_scaled": (
                    values[
                        "tier2_regression_bank_row_tier_pre_component_h15_scaled"
                    ]
                ),
                "dgs2": values["dgs2"],
                "dgs10": values["dgs10"],
                # Deliberately wrong precomputed values: the pipeline must derive these.
                "dgs2__lag_4": 999999.0,
                "dgs10__lag_1": 999999.0,
                "dgs10__lag_2": 999999.0,
                "quarter_is_q2": 999999.0,
                "quarter_is_q3": 999999.0,
                "quarter_is_q4": 999999.0,
            }
        )

    raw_values = {
        series.key: {} for series in OPEN02_CONTRACT.series
    }
    canonical_values: dict[str, float] = {}
    bank_values: dict[str, float] = {}
    agency_fractions = (0.15, 0.14, 0.13, 0.12, 0.11, 0.18, 0.17)
    agency_components = (
        "agency_us_res_pass",
        "agency_us_com_pass",
        "agency_us_res_cmo",
        "agency_us_com_cmo",
        "agency_us_other",
        "agency_fbo_total",
        "agency_aff_total",
    )
    for target_index, quarter in enumerate(target_quarters):
        history_index = target_index + 4
        values = history_values[quarter]
        controls = (
            values["GDP"],
            values["gdp_deflator"],
            values["FEDFUNDS"],
            values["TOTRESNS"],
            values["tier2_regression_bank_row_tier_pre_component_h15_scaled"],
        )
        x_value = values["X"]
        bank = (
            2.4 * x_value
            + 0.35 * controls[0]
            - 0.20 * controls[1]
            + values["b_noise"]
        )
        agency = (
            1.25 * x_value
            + 1.70 * bank
            + 0.2 * controls[2]
            + values["a_noise"]
        )
        loans = (
            -0.95 * x_value
            + 1.15 * bank
            + 0.2 * controls[3]
            + values["l_noise"]
        )
        deposits = (
            0.80 * x_value
            - 1.30 * bank
            + 0.2 * controls[4]
            + values["d_noise"]
        )
        canonical = x_value + bank
        canonical_values[quarter] = canonical
        bank_values[quarter] = bank
        design_rows[history_index][
            OPEN02_CONTRACT.open01_contract.canonical_treatment_id
        ] = canonical

        raw_values["tsy_us"][quarter] = 0.60 * bank
        raw_values["tsy_fbo"][quarter] = 0.25 * bank
        raw_values["tsy_aff"][quarter] = 0.15 * bank
        for component, fraction in zip(
            agency_components,
            agency_fractions,
            strict=True,
        ):
            raw_values[component][quarter] = fraction * agency
        raw_values["agency_us_total"][quarter] = sum(
            raw_values[component][quarter]
            for component in agency_components[:5]
        )
        raw_values["loans_us"][quarter] = 0.65 * loans
        raw_values["loans_fbo"][quarter] = 0.20 * loans
        raw_values["loans_aff"][quarter] = 0.15 * loans
        raw_values["dep_us_check"][quarter] = 0.28 * deposits
        raw_values["dep_us_time"][quarter] = 0.24 * deposits
        raw_values["dep_fbo_check"][quarter] = 0.16 * deposits
        raw_values["dep_fbo_time"][quarter] = 0.13 * deposits
        raw_values["dep_aff_check"][quarter] = 0.10 * deposits
        raw_values["dep_aff_time"][quarter] = 0.09 * deposits

    vintage = "2026-03-19"
    archive_member = OPEN02_CONTRACT.source.csv_member_sha256[0][0]
    dictionary_member = (
        OPEN02_CONTRACT.source.dictionary_member_sha256[0][0]
    )
    source_series = [
        {
            **asdict(series),
            "observation_vintage": vintage,
            "source_metadata": {
                "key": series.key,
                "fred_id": series.fred_id,
                "board_series_id": series.board_series_id,
                "archive_member": archive_member,
                "dictionary_member": dictionary_member,
                "official_description": (
                    series.official_title.removesuffix(", Transactions")
                ),
                "table_line": "Line 1",
                "table": "fixture",
                "unit_label": OPEN02_CONTRACT.source.unit_label,
                "side": series.side,
                "units": series.units,
                "seasonal_adjustment": series.seasonal_adjustment,
            },
            "observations": [
                {
                    "quarter": quarter,
                    "value": raw_values[series.key][quarter],
                }
                for quarter in target_quarters
            ],
        }
        for series in OPEN02_CONTRACT.series
    ]
    wide_rows = [
        {
            "quarter": quarter,
            **{
                series.key: raw_values[series.key][quarter]
                for series in OPEN02_CONTRACT.series
            },
        }
        for quarter in target_quarters
    ]
    rows_hash = hashlib.sha256(
        json.dumps(
            wide_rows,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    source_bundle = {
        "schema_version": 1,
        "kind": "open02_financial_accounts_input",
        "accepted_tdcest_bundle_generated_at": (
            OPEN02_CONTRACT.sample.accepted_tdcest_bundle_generated_at
        ),
        "observation_vintage": vintage,
        "observation_vintage_cutoff": (
            OPEN02_CONTRACT.sample.observation_vintage_cutoff
        ),
        "official_release": {
            "kind": "open02_board_z1_archive",
            "source_url": OPEN02_CONTRACT.source.archive_url,
            "release_date": OPEN02_CONTRACT.source.release_date,
            "observation_vintage_cutoff": (
                OPEN02_CONTRACT.sample.observation_vintage_cutoff
            ),
            "archive_sha256": OPEN02_CONTRACT.source.archive_sha256,
            "csv_member_sha256": dict(
                OPEN02_CONTRACT.source.csv_member_sha256
            ),
            "dictionary_member_sha256": dict(
                OPEN02_CONTRACT.source.dictionary_member_sha256
            ),
            "rows_sha256": rows_hash,
            "sample_start": OPEN02_CONTRACT.sample.start_quarter,
            "sample_end": OPEN02_CONTRACT.sample.end_quarter,
            "observations": OPEN02_CONTRACT.sample.observations,
            "series_count": len(OPEN02_CONTRACT.series),
        },
        "series": source_series,
    }
    standardized_rows = [
        {
            "series_id": series_id,
            "freq": "quarterly",
            "period_end": _period_end(int(quarter[:4]), int(quarter[-1])),
            "units": "usd_millions",
            "value": values[quarter],
        }
        for series_id, values in (
            (
                OPEN02_CONTRACT.canonical_treatment_source_series,
                canonical_values,
            ),
            (
                OPEN02_CONTRACT.embedded_bank_treasury_component_id,
                bank_values,
            ),
        )
        for quarter in target_quarters
    ]
    return source_bundle, design_rows, standardized_rows


def _assert_reason(
    expected: str,
    source_bundle: dict,
    design_rows: list[dict],
    standardized_rows: list[dict],
    *,
    contract=OPEN02_CONTRACT,
) -> None:
    with pytest.raises(Open02ValidationError) as caught:
        run_open02_pipeline(
            source_bundle,
            design_rows,
            standardized_rows,
            contract=contract,
        )
    assert caught.value.reason_code == expected
    assert isinstance(caught.value.details, dict)


def _refresh_source_rows_hash(source_bundle: dict) -> None:
    by_key = {
        series["key"]: {
            row["quarter"]: row["value"]
            for row in series["observations"]
        }
        for series in source_bundle["series"]
    }
    rows = [
        {
            "quarter": quarter,
            **{
                series.key: by_key[series.key][quarter]
                for series in OPEN02_CONTRACT.series
            },
        }
        for quarter in (
            row["quarter"]
            for row in source_bundle["series"][0]["observations"]
        )
    ]
    source_bundle["official_release"]["rows_sha256"] = hashlib.sha256(
        json.dumps(
            rows,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def test_open02_pipeline_valid_synthetic_96_quarter_bundle() -> None:
    source_bundle, design_rows, standardized_rows = _valid_fixture()
    for quarter, period_end in (
        ("2026Q1", "2026-03-31"),
        ("2026Q2", "2026-06-30"),
    ):
        future = copy.deepcopy(design_rows[-1])
        future["quarter"] = quarter
        future["period_end"] = period_end
        design_rows.append(future)

    result = run_open02_pipeline(
        source_bundle,
        design_rows,
        standardized_rows,
    )

    assert len(result.panel_rows) == 96
    assert len(result.estimate_rows) == 31
    assert len(result.wald_rows) == 3
    assert len(result.influence_rows) == 3 * (96 + 93)
    assert len(result.influence_summaries) == 3
    assert result.acceptance["valid_result"] is True
    assert result.acceptance["all_deterministic_gates_passed"] is True
    assert len(result.acceptance["deterministic_gates"]) == 12
    assert all(
        row["passed"] for row in result.acceptance["deterministic_gates"]
    )
    assert result.acceptance["main_text_eligible"] is True
    assert result.acceptance["appendix_only"] is False
    assert result.acceptance["reason_codes"] == ()
    adding_up = result.acceptance["coefficient_adding_up_evidence"]
    assert adding_up["deletion_fits_checked"] == 96 + 93
    assert (
        adding_up[
            "maximum_deletion_source_agency_beta_relative_error"
        ]
        <= OPEN02_CONTRACT.sample.coefficient_adding_up_relative_tolerance
    )
    assert (
        adding_up[
            "maximum_deletion_within_agency_theta_relative_error"
        ]
        <= OPEN02_CONTRACT.sample.coefficient_adding_up_relative_tolerance
    )

    first = result.panel_rows[0]
    assert first["quarter"] == "2002Q1"
    assert first["dgs2__lag_4"] == design_rows[0]["dgs2"]
    assert first["dgs10__lag_1"] == design_rows[3]["dgs10"]
    assert first["dgs10__lag_2"] == design_rows[2]["dgs10"]
    assert first["quarter_is_q2"] == 0.0
    q2 = result.panel_rows[1]
    assert q2["quarter_is_q2"] == 1.0
    assert q2["quarter_is_q3"] == 0.0
    assert q2["quarter_is_q4"] == 0.0
    assert first["C"] == pytest.approx(first["X"] + first["B"])


@pytest.mark.parametrize("mutation", ("missing", "duplicate"))
def test_open02_pipeline_rejects_missing_or_duplicate_observations(
    mutation: str,
) -> None:
    source_bundle, design_rows, standardized_rows = _valid_fixture()
    observations = source_bundle["series"][0]["observations"]
    if mutation == "missing":
        observations.pop()
    else:
        observations[-1] = copy.deepcopy(observations[0])

    _assert_reason(
        "coverage_gate_failed",
        source_bundle,
        design_rows,
        standardized_rows,
    )


def test_open02_pipeline_rejects_metadata_and_mixed_vintage() -> None:
    source_bundle, design_rows, standardized_rows = _valid_fixture()
    metadata_bad = copy.deepcopy(source_bundle)
    metadata_bad["series"][0]["official_title"] += " drift"
    _assert_reason(
        "metadata_gate_failed",
        metadata_bad,
        design_rows,
        standardized_rows,
    )

    vintage_bad = copy.deepcopy(source_bundle)
    vintage_bad["series"][1]["observation_vintage"] = (
        "2026-03-18T00:00:00Z"
    )
    _assert_reason(
        "vintage_gate_failed",
        vintage_bad,
        design_rows,
        standardized_rows,
    )


def test_open02_pipeline_rejects_identity_and_accepted_component_drift() -> None:
    source_bundle, design_rows, standardized_rows = _valid_fixture()
    identity_bad = copy.deepcopy(source_bundle)
    agency_parent = next(
        series
        for series in identity_bad["series"]
        if series["key"] == "agency_us_total"
    )
    agency_parent["observations"][0]["value"] += 1.0
    _refresh_source_rows_hash(identity_bad)
    _assert_reason(
        "us_agency_identity_failed",
        identity_bad,
        design_rows,
        standardized_rows,
    )

    accepted_bad = copy.deepcopy(standardized_rows)
    bank_id = OPEN02_CONTRACT.embedded_bank_treasury_component_id
    next(row for row in accepted_bad if row["series_id"] == bank_id)[
        "value"
    ] += 1.0
    _assert_reason(
        "accepted_component_reconciliation_failed",
        source_bundle,
        design_rows,
        accepted_bad,
    )


def test_open02_pipeline_rejects_sample_hash_and_no_double_side_lineage() -> None:
    source_bundle, design_rows, standardized_rows = _valid_fixture()
    bad_sample = replace(
        OPEN02_CONTRACT,
        sample=replace(
            OPEN02_CONTRACT.sample,
            quarter_hash="0" * 64,
        ),
    )
    _assert_reason(
        "coverage_gate_failed",
        source_bundle,
        design_rows,
        standardized_rows,
        contract=bad_sample,
    )

    bad_lineage_items = tuple(
        (
            identifier,
            (1, 1, 1) if identifier == OPEN02_LEAVE_OUT_ID else lineage,
        )
        for identifier, lineage in OPEN02_CONTRACT.lineage.lineage_by_id
    )
    bad_lineage = replace(
        OPEN02_CONTRACT,
        lineage=replace(
            OPEN02_CONTRACT.lineage,
            lineage_by_id=bad_lineage_items,
        ),
    )
    _assert_reason(
        "lineage_gate_failed",
        source_bundle,
        design_rows,
        standardized_rows,
        contract=bad_lineage,
    )


def test_open02_pipeline_rejects_rank_deficient_common_design() -> None:
    source_bundle, design_rows, standardized_rows = _valid_fixture()
    for row in design_rows:
        row["TOTRESNS"] = row["GDP"]

    _assert_reason(
        "rank_gate_failed",
        source_bundle,
        design_rows,
        standardized_rows,
    )


def test_open02_statistical_eligibility_uses_exact_holm_and_influence_reasons() -> None:
    wald_rows = [
        {"hypothesis_id": hypothesis_id, "holm_adjusted_p_value": 0.01}
        for hypothesis_id in ("H_T", "H_P", "H_W")
    ]
    summaries = [
        {"group_id": group_id, "passed": True, "reason_codes": ()}
        for group_id in ("g_T", "g_P", "g_W")
    ]

    accepted = evaluate_open02_eligibility(wald_rows, summaries)
    assert accepted == {
        "valid_result": True,
        "main_text_eligible": True,
        "appendix_only": False,
        "reason_codes": (),
        "holm_familywise_alpha": 0.05,
    }

    wald_rows[1]["holm_adjusted_p_value"] = 0.20
    summaries[2]["passed"] = False
    summaries[2]["reason_codes"] = ("leave_block_influence_gt_0_50",)
    nonpromoted = evaluate_open02_eligibility(wald_rows, summaries)
    assert nonpromoted["valid_result"] is True
    assert nonpromoted["main_text_eligible"] is False
    assert nonpromoted["appendix_only"] is True
    assert nonpromoted["reason_codes"] == (
        "portfolio_joint_holm_gt_0_05",
        "leave_block_influence_gt_0_50",
    )
