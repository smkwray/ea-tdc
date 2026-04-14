from __future__ import annotations

import csv
import json
from pathlib import Path

from ea_tdc.adapters.tsyparty import adapt_tsyparty
from ea_tdc.paths import ensure_repo_dirs, project_paths


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_tsyparty_adapter_normalizes_lagged_absorption_states(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)

    publish_dir = tmp_path / "tsyparty_publish" / "outputs" / "inference"
    _write_text(
        publish_dir / "counterparty_flows.csv",
        "\n".join(
            [
                "date,seller,buyer,amount,method,converged",
                "2024-03-31,dealers,banks,40,dense,True",
                "2024-03-31,dealers,foreigners_official,60,dense,True",
                "2024-03-31,dealers,_residual,10,dense,True",
                "2024-06-30,dealers,banks,20,dense,True",
                "2024-06-30,dealers,foreigners_official,10,dense,True",
                "2024-06-30,dealers,insurers,70,dense,True",
            ]
        ),
    )
    _write_json(
        publish_dir / "manifest.json",
        {
            "build_timestamp": "2026-04-11T12:00:00+00:00",
            "quarters_processed": 2,
            "quarters_skipped": 0,
            "claims_label": "likely_net_counterparties",
        },
    )
    similarity_dir = tmp_path / "tsyparty_publish" / "outputs" / "similarity_enriched"
    _write_text(
        similarity_dir / "rolling_correlations.csv",
        "\n".join(
            [
                "date,banks_vs_foreigners_official,banks_vs_foreigners_private,banks_vs_money_market_funds,foreigners_official_vs_foreigners_private,foreigners_official_vs_money_market_funds,foreigners_private_vs_money_market_funds",
                "2024-03-31,0.1,0.3,-0.2,0.9,0.0,0.1",
                "2024-06-30,0.2,0.5,-0.1,0.8,0.1,0.2",
            ]
        ),
    )
    _write_json(
        similarity_dir / "manifest.json",
        {
            "build_timestamp": "2026-04-11T12:30:00+00:00",
            "pipeline": "similarity",
            "targets_found": ["banks", "foreigners_official", "foreigners_private", "money_market_funds"],
        },
    )

    result = adapt_tsyparty(paths, publish_dir=str(tmp_path / "tsyparty_publish"))

    assert result.rows_written == 7
    with result.standardized_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["series_id"]: row for row in rows}
    assert by_id["tsyparty_bank_absorption_share_l1"]["value"] == "0.4"
    assert by_id["tsyparty_row_absorption_share_l1"]["value"] == "0.6"
    assert by_id["tsyparty_ru_gap_l1"]["value"] == "0.2"
    assert by_id["tsyparty_ru_gap_l1"]["period_end"] == "2024-06-30"
    assert by_id["tsyparty_bank_foreign_official_corr_l1"]["value"] == "0.1"
    assert by_id["tsyparty_bank_foreign_private_corr_l1"]["value"] == "0.3"
    assert by_id["tsyparty_bank_mmf_corr_l1"]["value"] == "-0.2"
    assert by_id["tsyparty_private_minus_official_corr_l1"]["value"] == "0.2"
    assert by_id["tsyparty_bank_foreign_private_corr_l1"]["source_table"] == "rolling_correlations"

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_repo"] == "tsyparty"
    assert manifest["quarters_processed"] == 2
    assert manifest["similarity_pipeline"] == "similarity"
