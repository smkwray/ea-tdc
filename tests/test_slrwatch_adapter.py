from __future__ import annotations

import csv
import json
from pathlib import Path

from ea_tdc.adapters.slrwatch import adapt_slrwatch
from ea_tdc.paths import ensure_repo_dirs, project_paths


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_slrwatch_adapter_normalizes_lagged_pressure_states(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)

    publish_dir = tmp_path / "slrwatch_publish"
    _write_text(
        publish_dir / "output" / "reports" / "constraint_decomposition" / "prepared_panel.csv",
        "\n".join(
            [
                "quarter_end,entity_source,entity_id,entity_name,leverage_pressure_score,duration_pressure_score,funding_pressure_score,dominant_constraint",
                "2023-12-31,insured_bank,b1,Bank 1,0.60,0.20,0.40,leverage",
                "2023-12-31,insured_bank,b2,Bank 2,0.40,0.30,0.50,duration_loss",
                "2024-03-31,insured_bank,b1,Bank 1,0.65,0.25,0.35,leverage",
                "2024-03-31,insured_bank,b2,Bank 2,0.55,0.35,0.45,funding",
                "2023-12-31,parent_or_ihc,p1,Parent 1,0.30,0.50,0.40,leverage",
                "2024-03-31,parent_or_ihc,p1,Parent 1,0.45,0.55,0.35,duration_loss",
            ]
        ),
    )
    _write_text(
        publish_dir / "output" / "reports" / "policy_regime_panel" / "regime_quarter_panel.csv",
        "\n".join(
            [
                "quarter_end,bank_headroom_pp_mean,parent_headroom_pp_mean,policy_regime",
                "2023-12-31,0.031,3.25,post_exclusion_normalization",
                "2024-03-31,0.028,3.10,duration_loss_window",
            ]
        ),
    )

    result = adapt_slrwatch(paths, publish_dir=str(publish_dir))

    assert result.rows_written == 11
    with result.standardized_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["series_id"]: row for row in rows}
    assert by_id["slrwatch_bank_leverage_pressure_l1"]["value"] == "0.5"
    assert by_id["slrwatch_bank_duration_pressure_l1"]["value"] == "0.25"
    assert by_id["slrwatch_bank_leverage_dominant_share_l1"]["value"] == "0.5"
    assert by_id["slrwatch_bank_duration_loss_dominant_share_l1"]["value"] == "0.5"
    assert by_id["slrwatch_parent_leverage_pressure_l1"]["value"] == "0.3"
    assert by_id["slrwatch_bank_headroom_pp_l1"]["period_end"] == "2024-03-31"

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_repo"] == "slrwatch"
    assert manifest["quarters_covered"] == 2
    assert manifest["constraint_observations"] == 6
