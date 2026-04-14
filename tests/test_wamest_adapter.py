from __future__ import annotations

import csv
import json
from pathlib import Path

from ea_tdc.adapters.wamest import adapt_wamest
from ea_tdc.paths import ensure_repo_dirs, project_paths


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_wamest_adapter_normalizes_lagged_maturity_states(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)

    publish_dir = tmp_path / "wamest_publish"
    _write_text(
        publish_dir / "data" / "processed" / "sector_effective_maturity_full.csv",
        "\n".join(
            [
                "date,sector_key,in_publication_range,bill_share,short_share_le_1y,zero_coupon_equivalent_years",
                "2023-12-31,bank_reserve_access_core,True,0.136,0.241,4.05",
                "2024-03-31,bank_reserve_access_core,True,0.135,0.240,4.07",
                "2023-12-31,bank_broad_private_depositories_marketable_proxy,True,0.128,0.232,3.95",
                "2024-03-31,bank_broad_private_depositories_marketable_proxy,True,0.127,0.231,3.96",
                "2023-12-31,foreigners_total,True,0.185,0.604,4.70",
                "2024-03-31,foreigners_total,True,0.184,0.603,4.73",
                "2023-12-31,domestic_nonbank_residual_broad,True,0.333,0.417,6.59",
                "2024-03-31,domestic_nonbank_residual_broad,True,0.333,0.417,6.59",
            ]
        ),
    )
    _write_json(
        publish_dir / "outputs" / "full_coverage_release" / "run_manifest.json",
        {
            "resolved_latest_snapshot_date": "2024-03-31",
            "run_timestamp_utc": "2026-04-11T15:00:00+00:00",
            "schema_version": "v0.2-full-coverage",
        },
    )
    _write_json(
        publish_dir / "outputs" / "full_coverage_release" / "full_coverage_summary.json",
        {"latest_snapshot_summary": {"quarter": "2024-03-31"}},
    )

    result = adapt_wamest(paths, publish_dir=str(publish_dir))

    assert result.rows_written == 10
    with result.standardized_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["series_id"]: row for row in rows}
    assert by_id["wamest_bank_reserve_short_share_l1"]["value"] == "0.241"
    assert by_id["wamest_bank_reserve_wam_years_l1"]["value"] == "4.05"
    assert by_id["wamest_foreigners_wam_years_l1"]["value"] == "4.70"
    assert by_id["wamest_domestic_nonbank_short_share_l1"]["period_end"] == "2024-03-31"

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_repo"] == "wamest"
    assert manifest["resolved_latest_snapshot_date"] == "2024-03-31"
