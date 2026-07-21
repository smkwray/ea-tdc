from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from ea_tdc.adapters.tdcest import adapt_tdcest
from ea_tdc.paths import ensure_repo_dirs, project_paths


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_tdcest_adapter_writes_standardized_bundle_and_manifest(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "runtime.yaml",
        "\n".join(
            [
                "project:",
                "  name: ea-tdc",
                "  package_name: ea_tdc",
                "  active_release: release_1",
                "paths:",
                "  data_root: data",
                "  output_root: output",
                "fetch:",
                "  fred_series_manifest: config/fred_manifest_seed.csv",
                "  fred_api_key_env: FRED_API_KEY",
                "  default_start_date: 1980-01-01",
                "  allow_graph_csv_fallback: true",
                "remote:",
                "  ssh_host: shanewray@100.71.19.72",
                "  run_heavy_jobs_remotely: true",
                "  path_parity_note: mirrored",
            ]
        ),
    )
    _write_text(
        tmp_path / "config" / "fred_manifest_seed.csv",
        "\n".join(
            [
                "series_id,domain,role,priority,default_transform,notes",
                "GDP,real_activity,control,headline,logdiff,GDP level",
            ]
        ),
    )
    _write_text(
        tmp_path / "data" / "seed" / "tdcest" / "bundle.json",
        json.dumps(
            {
                "bundle_format": "tdc_site_bundle_v2",
                "generated_at_utc": "2026-04-11T00:00:00+00:00",
                "summary": {"preferred_method": "tdc_base_bank_only_ru_flow"},
                "metadata": {},
                "dates": ["2024-03-31", "2024-06-30"],
                "estimates": {
                    "columns": ["tdc_base_bank_only_ru_flow"],
                    "tdc_base_bank_only_ru_flow": [1.5, 2.5],
                },
                "components": {
                    "columns": ["fed_tsy_tx"],
                    "fed_tsy_tx": [0.5, 0.75],
                },
                "references": {
                    "columns": ["gdp_deflator"],
                    "gdp_deflator": [120.0, 121.0],
                },
            }
        ),
    )

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = adapt_tdcest(paths)

    assert result.rows_written == 6
    assert result.standardized_path.exists()
    assert result.manifest_path.exists()

    with result.standardized_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert any(row["series_id"] == "tdc_base_bank_only_ru_flow" and row["role"] == "treatment" for row in rows)
    assert any(row["series_id"] == "gdp_deflator" and row["role"] == "control" for row in rows)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_repo"] == "tdcest"
    assert manifest["rows_written"] == 6
    assert manifest["bundle_hash"] == result.bundle_hash


def test_tdcest_adapter_supplements_processed_estimates(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "runtime.yaml",
        "\n".join(
            [
                "project:",
                "  name: ea-tdc",
                "  package_name: ea_tdc",
                "  active_release: release_1",
                "paths:",
                "  data_root: data",
                "  output_root: output",
                "fetch:",
                "  fred_series_manifest: config/fred_manifest_seed.csv",
                "  fred_api_key_env: FRED_API_KEY",
                "  default_start_date: 1980-01-01",
                "  allow_graph_csv_fallback: true",
                "remote:",
                "  ssh_host: shanewray@100.71.19.72",
                "  run_heavy_jobs_remotely: true",
                "  path_parity_note: mirrored",
            ]
        ),
    )
    _write_text(
        tmp_path / "config" / "fred_manifest_seed.csv",
        "series_id,domain,role,priority,default_transform,notes\nGDP,real_activity,control,headline,logdiff,GDP level",
    )
    _write_text(
        tmp_path / "site" / "data" / "bundle.json",
        json.dumps(
            {
                "bundle_format": "tdc_site_bundle_v4",
                "generated_at_utc": "2026-04-11T00:00:00+00:00",
                "summary": {"preferred_method": "tdc_base_bank_only_ru_flow"},
                "metadata": {},
                "dates": ["2024-03-31"],
                "estimates": {
                    "columns": ["tdc_base_bank_only_ru_flow"],
                    "tdc_base_bank_only_ru_flow": [1.5],
                },
                "components": {"columns": []},
                "references": {"columns": []},
            }
        ),
    )
    _write_text(
        tmp_path / "data" / "processed" / "tdc_estimates.csv",
        "\n".join(
            [
                "date,tdc_tier2_canonical_depository_institution_mmf_rrp_prop_ru_flow,tdc_tier2_h15_treasury_interest_robust_mmf_rrp_prop_bank_only_ru_flow",
                "2024-03-31,123.0,456.0",
            ]
        ),
    )

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = adapt_tdcest(paths, bundle_path=str(tmp_path / "site" / "data" / "bundle.json"))

    with result.standardized_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_series = {row["series_id"]: row for row in rows}
    canonical = by_series["tdc_tier2_canonical_depository_institution_mmf_rrp_prop_ru_flow"]
    robust_alias = by_series["tdc_tier2_treasury_interest_robust_mmf_rrp_prop_bank_only_ru_flow"]

    assert canonical["value"] == "123.0"
    assert canonical["source_table"] == "tdc_estimates"
    assert robust_alias["value"] == "456.0"
    assert "source_column=tdc_tier2_h15_treasury_interest_robust_mmf_rrp_prop_bank_only_ru_flow" in robust_alias["notes"]


def test_tdcest_adapter_supplements_regression_mmf_rrp_rows_and_tier_indicators(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "runtime.yaml",
        "\n".join(
            [
                "project:",
                "  name: ea-tdc",
                "  package_name: ea_tdc",
                "  active_release: release_1",
                "paths:",
                "  data_root: data",
                "  output_root: output",
                "fetch:",
                "  fred_series_manifest: config/fred_manifest_seed.csv",
                "  fred_api_key_env: FRED_API_KEY",
                "  default_start_date: 1980-01-01",
                "  allow_graph_csv_fallback: true",
                "remote:",
                "  ssh_host: shanewray@100.71.19.72",
                "  run_heavy_jobs_remotely: true",
                "  path_parity_note: mirrored",
            ]
        ),
    )
    _write_text(
        tmp_path / "config" / "fred_manifest_seed.csv",
        "series_id,domain,role,priority,default_transform,notes\nGDP,real_activity,control,headline,logdiff,GDP level",
    )
    _write_text(
        tmp_path / "site" / "data" / "bundle.json",
        json.dumps(
            {
                "bundle_format": "tdc_site_bundle_v4",
                "generated_at_utc": "2026-04-11T00:00:00+00:00",
                "summary": {"preferred_method": "tdc_base_bank_only_ru_flow"},
                "metadata": {},
                "dates": ["2024-03-31"],
                "estimates": {
                    "columns": ["tdc_base_bank_only_ru_flow"],
                    "tdc_base_bank_only_ru_flow": [1.5],
                },
                "components": {"columns": []},
                "references": {"columns": []},
            }
        ),
    )
    _write_text(
        tmp_path / "data" / "processed" / "tdc_tier2_regression_series.csv",
        "\n".join(
            [
                (
                    "date,tdc_tier2_regression_bank_only_ru_flow,"
                    "tdc_tier2_regression_mmf_rrp_prop_bank_only_ru_flow,"
                    "tdc_tier2_regression_mmf_rrp_prop_broad_depository_np_cu_ru_flow,"
                    "tier2_regression_bank_row_method_tier"
                ),
                "2024-03-31,100.0,110.0,120.0,constrained_component",
            ]
        ),
    )

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = adapt_tdcest(paths, bundle_path=str(tmp_path / "site" / "data" / "bundle.json"))

    with result.standardized_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_series = {row["series_id"]: row for row in rows}
    assert by_series["tdc_tier2_regression_mmf_rrp_prop_bank_only_ru_flow"]["value"] == "110.0"
    assert (
        by_series["tdc_tier2_regression_mmf_rrp_prop_broad_depository_np_cu_ru_flow"]["value"]
        == "120.0"
    )
    tier = by_series["tier2_regression_bank_row_method_tier"]
    assert tier["value"] == "constrained_component"
    assert tier["units"] == "category"
    dummy = by_series["tier2_regression_bank_row_method_tier__is_constrained_component"]
    assert dummy["value"] == "1"
    assert dummy["role"] == "control"


def test_tdcest_adapter_replaces_seed_estimates_and_records_all_input_hashes(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "config" / "runtime.yaml",
        "\n".join(
            [
                "project:",
                "  name: ea-tdc",
                "  package_name: ea_tdc",
                "paths:",
                "  data_root: data",
                "  output_root: output",
            ]
        ),
    )
    _write_text(
        tmp_path / "site" / "data" / "bundle.json",
        json.dumps(
            {
                "bundle_format": "tdc_site_bundle_v4",
                "generated_at_utc": "2026-04-11T00:00:00+00:00",
                "summary": {"preferred_method": "tdc_base_bank_only_ru_flow"},
                "metadata": {},
                "dates": ["2024-03-31"],
                "estimates": {
                    "columns": [
                        "tdc_base_bank_only_ru_flow",
                        "tdc_tier2_interest_corrected_bank_only_ru_flow",
                        "tdc_tier3_fiscal_corrected_bank_only_ru_flow",
                        "tdc_tier2_canonical_depository_institution_mmf_rrp_prop_ru_flow",
                        "tdc_unenumerated_current_variant",
                        "tdc_du_selected_domestic_nonfinancial_proxy",
                    ],
                    "tdc_base_bank_only_ru_flow": [1.0],
                    "tdc_tier2_interest_corrected_bank_only_ru_flow": [2.0],
                    "tdc_tier3_fiscal_corrected_bank_only_ru_flow": [3.0],
                    "tdc_tier2_canonical_depository_institution_mmf_rrp_prop_ru_flow": [4.0],
                    "tdc_unenumerated_current_variant": [5.0],
                    "tdc_du_selected_domestic_nonfinancial_proxy": [6.0],
                },
                "components": {"columns": []},
                "references": {"columns": []},
            }
        ),
    )
    _write_text(
        tmp_path / "data" / "processed" / "tdc_estimates.csv",
        "\n".join(
            [
                (
                    "date,tdc_base_bank_only_ru_flow,"
                    "tdc_tier2_interest_corrected_bank_only_ru_flow,"
                    "tdc_tier3_fiscal_corrected_bank_only_ru_flow,"
                    "tdc_tier2_canonical_depository_institution_mmf_rrp_prop_ru_flow,"
                    "tdc_unenumerated_current_variant,"
                    "tdc_tier2_h15_treasury_interest_robust_bank_only_ru_flow,"
                    "tdc_tier2_h15_treasury_interest_robust_depository_institution_np_cu_ru_flow,"
                    "tdc_tier2_h15_treasury_interest_robust_mmf_rrp_prop_bank_only_ru_flow,"
                    "tdc_tier2_h15_treasury_interest_robust_mmf_rrp_prop_depository_institution_np_cu_ru_flow"
                ),
                "2024-03-31,101,102,103,104,105,201,202,203,204",
            ]
        ),
    )
    regression_path = tmp_path / "data" / "processed" / "tdc_tier2_regression_series.csv"
    _write_text(regression_path, "date,tdc_tier2_regression_bank_only_ru_flow\n2024-03-31,300\n")

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = adapt_tdcest(paths, bundle_path=str(tmp_path / "site" / "data" / "bundle.json"))
    with result.standardized_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_series = {row["series_id"]: row for row in rows}

    for series_id, expected in {
        "tdc_base_bank_only_ru_flow": "101",
        "tdc_tier2_interest_corrected_bank_only_ru_flow": "102",
        "tdc_tier3_fiscal_corrected_bank_only_ru_flow": "103",
        "tdc_tier2_canonical_depository_institution_mmf_rrp_prop_ru_flow": "104",
        "tdc_unenumerated_current_variant": "105",
    }.items():
        assert by_series[series_id]["value"] == expected
        assert by_series[series_id]["source_table"] == "tdc_estimates"

    for series_id, expected_source, expected_value in (
        (
            "tdc_tier2_treasury_interest_robust_bank_only_ru_flow",
            "tdc_tier2_h15_treasury_interest_robust_bank_only_ru_flow",
            "201",
        ),
        (
            "tdc_tier2_treasury_interest_robust_depository_institution_np_cu_ru_flow",
            "tdc_tier2_h15_treasury_interest_robust_depository_institution_np_cu_ru_flow",
            "202",
        ),
        (
            "tdc_tier2_treasury_interest_robust_mmf_rrp_prop_bank_only_ru_flow",
            "tdc_tier2_h15_treasury_interest_robust_mmf_rrp_prop_bank_only_ru_flow",
            "203",
        ),
        (
            "tdc_tier2_treasury_interest_robust_mmf_rrp_prop_depository_institution_np_cu_ru_flow",
            "tdc_tier2_h15_treasury_interest_robust_mmf_rrp_prop_depository_institution_np_cu_ru_flow",
            "204",
        ),
    ):
        assert by_series[series_id]["value"] == expected_value
        assert f"source_column={expected_source}" in by_series[series_id]["notes"]

    assert "tdc_du_selected_domestic_nonfinancial_proxy" not in by_series

    first_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert set(first_manifest["input_hashes"]) == {
        "seed_bundle",
        "processed_estimates",
        "regression_series",
    }
    assert all(len(value) == 64 for value in first_manifest["input_hashes"].values())
    assert first_manifest["input_hashes"]["regression_series"] == hashlib.sha256(
        regression_path.read_bytes()
    ).hexdigest()

    _write_text(regression_path, "date,tdc_tier2_regression_bank_only_ru_flow\n2024-03-31,301\n")
    rerun = adapt_tdcest(paths, bundle_path=str(tmp_path / "site" / "data" / "bundle.json"))
    second_manifest = json.loads(rerun.manifest_path.read_text(encoding="utf-8"))
    assert second_manifest["combined_input_hash"] != first_manifest["combined_input_hash"]
