from __future__ import annotations

import csv
import json
from pathlib import Path

from ea_tdc.adapters.coordwatch import adapt_coordwatch
from ea_tdc.paths import ensure_repo_dirs, project_paths


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_coordwatch_adapter_normalizes_published_liquidity_states(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)

    publish_dir = tmp_path / "coordwatch_publish"
    _write_json(
        publish_dir / "quarterly_panel.json",
        [
            {
                "quarter": "2024Q1",
                "low_liquidity_prev": 1,
                "liquidity_tightness_q_z_prev": 1.25,
                "system_liquidity_q_bn": 100.0,
            },
            {
                "quarter": "2024Q2",
                "low_liquidity_prev": 0,
                "liquidity_tightness_q_z_prev": -0.5,
                "system_liquidity_q_bn": 120.0,
            },
        ],
    )
    _write_json(
        publish_dir / "quarterly_descriptive.json",
        [
            {
                "quarter": "2024Q1",
                "on_rrp_share": 0.40,
                "reserves_bn_q": 800.0,
                "on_rrp_bn_q": 500.0,
                "repo_spread_bp_q": 10.0,
                "dealer_inventory_bn_q": 100.0,
                "net_private_duration_dv01": 50.0,
                "debt_limit_flag": 0,
            },
            {
                "quarter": "2024Q2",
                "on_rrp_share": 0.20,
                "reserves_bn_q": 900.0,
                "on_rrp_bn_q": 250.0,
                "repo_spread_bp_q": 12.0,
                "dealer_inventory_bn_q": 110.0,
                "net_private_duration_dv01": 55.0,
                "debt_limit_flag": 1,
            },
        ],
    )
    _write_json(
        publish_dir / "summary.json",
        {
            "generated_at_utc": "2026-04-11T12:00:00+00:00",
            "quarter_rows": 2,
            "weekly_rows": 10,
        },
    )

    result = adapt_coordwatch(paths, publish_dir=str(publish_dir))

    assert result.rows_written > 0
    with result.standardized_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_key = {(row["series_id"], row["period_end"]): row for row in rows}
    assert by_key[("coord_low_reserve_state_l1", "2024-03-31")]["value"] == "1"
    assert by_key[("coord_liquidity_tightness_q_z_l1", "2024-06-30")]["value"] == "-0.5"
    assert by_key[("coord_on_rrp_share_q", "2024-03-31")]["value"] == "0.4"
    assert by_key[("coord_on_rrp_drain_state_l1", "2024-06-30")]["value"] == "-1.0"
    assert by_key[("coord_debt_limit_flag_q", "2024-06-30")]["value"] == "true"

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_repo"] == "coordwatch"
    assert manifest["quarter_rows"] == 2
