from __future__ import annotations

import csv
from pathlib import Path

from ea_tdc.adapters.accounting import (
    adapt_accounting,
    apply_external_flow_rewrite,
    build_accounting_identity_alignment,
    build_seed_review,
    fill_missing_seed_channels_from_proxy_blocks,
    write_draft_seed_from_proxy_blocks,
)
from ea_tdc.paths import ensure_repo_dirs, project_paths


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_accounting_adapter_writes_standardized_bundle(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed" / "accounting" / "quarterly_identity_flows.csv"
    _write_text(
        seed_path,
        "\n".join(
            [
                "quarter,deposit_substitution_qoq,bank_balance_sheet_qoq,public_liquidity_qoq,external_flow_qoq,available_at,units,notes",
                "2024Q1,1.0,-2.0,3.0,4.0,2024-06-29,usd_billions,fixture",
                "2024Q2,1.5,-1.5,2.5,3.5,,usd_billions,fixture",
            ]
        ),
    )

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = adapt_accounting(paths, seed_path=str(seed_path))

    assert result.standardized_path.exists()
    with result.standardized_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert any(row["series_id"] == "accounting_deposit_substitution_qoq" and row["value"] == "1.0" for row in rows)
    assert any(row["series_id"] == "accounting_external_flow_qoq" and row["period_end"] == "2024-06-30" for row in rows)
    manifest = result.manifest_path.read_text(encoding="utf-8")
    assert '"seed_path": "seed/accounting/quarterly_identity_flows.csv"' in manifest
    assert '"standardized_path": "data/bundles/accounting/standardized_series.csv"' in manifest


def test_accounting_draft_seed_prefills_from_proxy_bundles(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    designs_dir = paths.bundles / "designs"
    designs_dir.mkdir(parents=True, exist_ok=True)
    raw_bundle = designs_dir / "baseline_tdc_lp_deposit_source_blocks__quarterly_bundle.csv"
    raw_bundle.write_text(
        "\n".join(
            [
                "quarter,other_component_qoq,deposit_substitution_block_qoq,bank_balance_sheet_proxy_block_qoq,public_liquidity_proxy_block_qoq,external_flow_proxy_block_qoq,proxy_accounting_total_qoq,proxy_unexplained_gap_qoq,cutoff_timestamp",
                "2024Q1,-5.0,1.0,-2.0,3.0,4.0,6.0,-11.0,2024-06-29T00:00:00",
            ]
        ),
        encoding="utf-8",
    )
    scaled_bundle = designs_dir / "baseline_tdc_lp_deposit_source_blocks_pct_gdp__quarterly_bundle.csv"
    scaled_bundle.write_text(
        "\n".join(
            [
                "quarter,other_component_qoq_pct_gdp,deposit_substitution_block_qoq_pct_gdp,bank_balance_sheet_proxy_block_qoq_pct_gdp,public_liquidity_proxy_block_qoq_pct_gdp,external_flow_proxy_block_qoq_pct_gdp,proxy_accounting_total_qoq_pct_gdp,proxy_unexplained_gap_qoq_pct_gdp",
                "2024Q1,-0.5,0.1,-0.2,0.3,0.4,0.6,-1.1",
            ]
        ),
        encoding="utf-8",
    )

    result = write_draft_seed_from_proxy_blocks(paths)

    with result.seed_path.open("r", encoding="utf-8", newline="") as handle:
        seed_rows = list(csv.DictReader(handle))
    assert seed_rows == [
        {
            "quarter": "2024Q1",
            "deposit_substitution_qoq": "1.0",
            "bank_balance_sheet_qoq": "-2.0",
            "public_liquidity_qoq": "3.0",
            "external_flow_qoq": "4.0",
            "available_at": "2024-06-29",
            "units": "usd_billions",
            "notes": "draft_prefill_from_proxy_blocks; edit_before_use",
        }
    ]
    with result.reference_path.open("r", encoding="utf-8", newline="") as handle:
        reference_rows = list(csv.DictReader(handle))
    assert reference_rows[0]["other_component_qoq_pct_gdp"] == "-0.5"
    manifest = result.manifest_path.read_text(encoding="utf-8")
    assert '"seed_path": "data/seed/accounting/quarterly_identity_flows.csv"' in manifest
    assert '"reference_path": "output/reports/accounting_identity_proxy_reference.csv"' in manifest


def test_accounting_seed_review_flags_sign_conflict_rows(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    _write_text(
        paths.seed / "accounting" / "quarterly_identity_flows.csv",
        "\n".join(
            [
                "quarter,deposit_substitution_qoq,bank_balance_sheet_qoq,public_liquidity_qoq,external_flow_qoq,available_at,units,notes",
                "2024Q1,1.0,2.0,0.0,1.0,2024-06-29,usd_billions,draft_prefill_from_proxy_blocks; edit_before_use",
            ]
        ),
    )
    _write_text(
        paths.reports / "accounting_identity_proxy_reference.csv",
        "\n".join(
            [
                "quarter,other_component_qoq,deposit_substitution_block_qoq,bank_balance_sheet_proxy_block_qoq,public_liquidity_proxy_block_qoq,external_flow_proxy_block_qoq,proxy_accounting_total_qoq,proxy_unexplained_gap_qoq,other_component_qoq_pct_gdp,deposit_substitution_block_qoq_pct_gdp,bank_balance_sheet_proxy_block_qoq_pct_gdp,public_liquidity_proxy_block_qoq_pct_gdp,external_flow_proxy_block_qoq_pct_gdp,proxy_accounting_total_qoq_pct_gdp,proxy_unexplained_gap_qoq_pct_gdp",
                "2024Q1,-5.0,1.0,2.0,0.0,1.0,4.0,-9.0,-0.5,0.1,0.2,0.0,0.1,0.4,-0.9",
            ]
        ),
    )
    _write_text(paths.raw_fred / "GDP.csv", "\n".join(["date,value", "2024-03-31,1000"]))

    result = build_seed_review(paths)

    with result.review_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["priority"] == "high"
    assert rows[0]["component_completeness"] == "complete"
    assert rows[0]["missing_components"] == ""
    assert rows[0]["sign_conflict"] == "true"
    assert rows[0]["accounting_identity_total_qoq"] == "4.0"
    assert rows[0]["accounting_identity_gap_qoq"] == "-9.0"
    assert rows[0]["accounting_identity_gap_qoq_pct_gdp"] == "-0.9"
    review_md = result.review_md_path.read_text(encoding="utf-8")
    assert "| 2024Q1 | high | complete |  | true |" in review_md
    summary_md = result.summary_md_path.read_text(encoding="utf-8")
    assert "| External flow |" in summary_md
    assert "- Incomplete rows: `0`" in summary_md
    with result.rewrite_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rewrite_rows = list(csv.DictReader(handle))
    assert rewrite_rows[0]["implied_external_flow_qoq"] == "-8.0"
    assert rewrite_rows[0]["external_flow_delta_qoq"] == "-9.0"
    rewrite_md = result.rewrite_md_path.read_text(encoding="utf-8")
    assert "| 2024Q1 | high | -5.0 | 3.0 | 1.0 | -8.0 | -9.0 |" in rewrite_md


def test_accounting_seed_review_marks_incomplete_rows(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    _write_text(
        paths.seed / "accounting" / "quarterly_identity_flows.csv",
        "\n".join(
            [
                "quarter,deposit_substitution_qoq,bank_balance_sheet_qoq,public_liquidity_qoq,external_flow_qoq,available_at,units,notes",
                "2024Q1,1.0,2.0,,1.0,2024-06-29,usd_billions,draft_prefill_from_proxy_blocks; edit_before_use",
            ]
        ),
    )
    _write_text(
        paths.reports / "accounting_identity_proxy_reference.csv",
        "\n".join(
            [
                "quarter,other_component_qoq,deposit_substitution_block_qoq,bank_balance_sheet_proxy_block_qoq,public_liquidity_proxy_block_qoq,external_flow_proxy_block_qoq,proxy_accounting_total_qoq,proxy_unexplained_gap_qoq,other_component_qoq_pct_gdp,deposit_substitution_block_qoq_pct_gdp,bank_balance_sheet_proxy_block_qoq_pct_gdp,public_liquidity_proxy_block_qoq_pct_gdp,external_flow_proxy_block_qoq_pct_gdp,proxy_accounting_total_qoq_pct_gdp,proxy_unexplained_gap_qoq_pct_gdp",
                "2024Q1,-5.0,1.0,2.0,0.0,1.0,4.0,-9.0,-0.5,0.1,0.2,0.0,0.1,0.4,-0.9",
            ]
        ),
    )
    _write_text(paths.raw_fred / "GDP.csv", "\n".join(["date,value", "2024-03-31,1000"]))

    result = build_seed_review(paths)

    with result.review_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["component_completeness"] == "incomplete"
    assert rows[0]["missing_components"] == "public_liquidity_qoq"
    assert rows[0]["accounting_identity_total_qoq"] == ""
    summary_md = result.summary_md_path.read_text(encoding="utf-8")
    assert "- Incomplete rows: `1`" in summary_md


def test_apply_external_flow_rewrite_updates_seed_rows(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    _write_text(
        paths.seed / "accounting" / "quarterly_identity_flows.csv",
        "\n".join(
            [
                "quarter,deposit_substitution_qoq,bank_balance_sheet_qoq,public_liquidity_qoq,external_flow_qoq,available_at,units,notes",
                "2024Q1,1.0,2.0,0.0,1.0,2024-06-29,usd_billions,draft",
                "2024Q2,1.0,2.0,0.0,1.0,2024-09-29,usd_billions,draft",
            ]
        ),
    )
    _write_text(
        paths.reports / "accounting_identity_external_flow_rewrite.csv",
        "\n".join(
            [
                "quarter,priority,other_component_qoq,fixed_non_external_total_qoq,current_external_flow_qoq,implied_external_flow_qoq,external_flow_delta_qoq,other_component_qoq_pct_gdp,fixed_non_external_total_qoq_pct_gdp,current_external_flow_qoq_pct_gdp,implied_external_flow_qoq_pct_gdp,external_flow_delta_qoq_pct_gdp,notes",
                "2024Q1,high,-5.0,3.0,1.0,-8.0,-9.0,-0.5,0.3,0.1,-0.8,-0.9,draft",
                "2024Q2,medium,-5.0,3.0,1.0,-7.0,-8.0,-0.5,0.3,0.1,-0.7,-0.8,draft",
            ]
        ),
    )

    result = apply_external_flow_rewrite(paths, min_priority="high")

    with (paths.seed / "accounting" / "quarterly_identity_flows.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["external_flow_qoq"] == "-8.0"
    assert "external_flow_rewritten_from_identity_high" in rows[0]["notes"]
    assert rows[1]["external_flow_qoq"] == "1.0"
    assert result.rows_updated == 1


def test_fill_missing_seed_channels_from_proxy_blocks_updates_only_blank_fields(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    _write_text(
        paths.seed / "accounting" / "quarterly_identity_flows.csv",
        "\n".join(
            [
                "quarter,deposit_substitution_qoq,bank_balance_sheet_qoq,public_liquidity_qoq,external_flow_qoq,available_at,units,notes",
                "2024Q1,,2.0,,1.0,2024-06-29,usd_billions,draft",
                "2024Q2,5.0,2.0,3.0,1.0,2024-09-29,usd_billions,draft",
            ]
        ),
    )
    _write_text(
        paths.bundles / "designs" / "baseline_tdc_lp_deposit_source_blocks__quarterly_bundle.csv",
        "\n".join(
            [
                "quarter,deposit_substitution_block_qoq,public_liquidity_proxy_block_qoq",
                "2024Q1,7.0,-4.0",
                "2024Q2,8.0,-5.0",
            ]
        ),
    )

    result = fill_missing_seed_channels_from_proxy_blocks(paths)

    with (paths.seed / "accounting" / "quarterly_identity_flows.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["deposit_substitution_qoq"] == "7.0"
    assert rows[0]["public_liquidity_qoq"] == "-4.0"
    assert "deposit_substitution_filled_from_proxy_blocks" in rows[0]["notes"]
    assert "public_liquidity_filled_from_proxy_blocks" in rows[0]["notes"]
    assert rows[1]["deposit_substitution_qoq"] == "5.0"
    assert rows[1]["public_liquidity_qoq"] == "3.0"
    assert result.rows_updated == 1
    assert result.deposit_substitution_fills == 1
    assert result.public_liquidity_fills == 1


def test_build_accounting_identity_alignment_writes_repeatable_note(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    _write_text(
        paths.output / "models" / "baseline_tdc_lp_deposits__robustness_k200_estimates.csv",
        "\n".join(
            [
                "outcome,horizon,beta,p_value",
                "other_component_qoq,0,-0.5,0.01",
                "other_component_qoq,1,0.2,0.20",
            ]
        ),
    )
    _write_text(
        paths.output / "models" / "baseline_tdc_lp_deposit_source_identity__lp_estimates.csv",
        "\n".join(
            [
                "outcome,horizon,beta,p_value",
                "accounting_identity_total_qoq,0,-0.45,0.02",
                "accounting_identity_gap_qoq,0,-0.05,0.40",
                "accounting_identity_total_qoq,1,-0.1,0.30",
                "accounting_identity_gap_qoq,1,-0.02,0.80",
            ]
        ),
    )
    _write_text(
        paths.output / "models" / "baseline_tdc_lp_deposit_source_identity_pct_gdp__lp_estimates.csv",
        "\n".join(
            [
                "outcome,horizon,beta,p_value",
                "accounting_identity_gap_qoq_pct_gdp,0,-0.001,0.50",
                "accounting_identity_gap_qoq_pct_gdp,1,-0.0005,0.60",
            ]
        ),
    )

    result = build_accounting_identity_alignment(paths)

    with result.csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["public_minus_accounting_total_beta"] == "-0.05"
    assert rows[0]["identity_gap_share_of_residual"] == "0.1"
    note = result.md_path.read_text(encoding="utf-8")
    assert "Those two numbers need not match" in note
    assert "| 0 | -0.500 | -0.450 | -0.050 | -0.050 | 0.100 |" in note
    manifest = result.manifest_path.read_text(encoding="utf-8")
    assert '"kind": "accounting_identity_alignment_manifest"' in manifest
