#!/usr/bin/env python3
"""Refit the primary reserve projection in every 48-quarter window of the declared sample.

Same outcome, treatment, controls and primary calendar HAC interval as the
pooled reserve estimate (run_reserve_projection.py) on the same hash-pinned
inputs. Windows never leave the declared sample. Adjacent windows share all but
one quarter, so the path describes sample dependence, not independent
estimates; interval coverage is unverified.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_open02_producer import _project_path, _sha256_file, _verify_producer_commit  # noqa: E402

from ea_tdc import reserve_projection as rp  # noqa: E402

WINDOW = 48
TABLE = "reserve_rolling_v1.csv"


def rolling(panel: rp.Panel, design: rp.Design, window: int = WINDOW) -> list[dict[str, Any]]:
    if not 2 <= window <= len(design.sample):
        raise ValueError("Rolling window must fit inside the declared sample")
    lag = design.hac_lags[0]
    rows = []
    for end in range(window, len(design.sample) + 1):
        quarters = design.sample[end - window:end]
        cell = rp.make_cell(panel, design, design.primary_outcome, design.primary_variant,
                            f"{quarters[0]}_{quarters[-1]}", design.primary_spec, quarters)
        if cell.dropped:
            raise ValueError(f"Window ending {quarters[-1]} drops {cell.dropped}")
        fit = rp.fit_cell(panel, design, cell)
        beta, hac = float(fit["beta"]), fit["hac"][lag]
        rows.append({"window_start_quarter": quarters[0], "window_end_quarter": quarters[-1],
                     "n": len(cell.positions), "outcome": cell.outcome, "treatment": cell.variant,
                     "spec": cell.spec, "beta": beta, f"se_hac{lag}": hac["se"],
                     "lower95": beta - rp.Z95 * hac["se"], "upper95": beta + rp.Z95 * hac["se"],
                     "p_beta_zero": hac["p_beta_zero"], "p_beta_one": hac["p_beta_one"],
                     "partial_variation_share": fit["partial_variation_share"],
                     "uncertainty_status": "coverage_unverified"})
    return rows


def run(args: argparse.Namespace) -> Path:
    commit = _verify_producer_commit(ROOT, args.producer_commit)
    design = rp.load_design(ROOT, args.config)
    panel = rp.load_panel(ROOT, design)
    rows = rolling(panel, design)
    output = _project_path(ROOT, args.output_dir)
    if output.exists():
        raise ValueError("Output directory already exists; preserve or adjudicate it before rerunning")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".reserve-rolling-", dir=output.parent))
    try:
        rp.write_table(stage / TABLE, rows)
        receipt = {
            "schema_version": "ea_tdc_reserve_rolling_receipt_v1",
            "producer_commit": commit,
            "dependency_lock_sha256": _sha256_file(ROOT / "uv.lock"),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "machine": platform.machine(), "system": platform.system()},
            "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": {"path": args.config, "sha256": design.sha256},
            "vintage": design.raw.get("vintage", "admitted"),
            "inputs": {key: {**record, "upstream_producer_commit": design.raw["upstream"]["producer_commit"]}
                       for key, record in design.raw["inputs"].items()},
            "input_cross_checks": panel.checks,
            "window_quarters": WINDOW,
            "primary": {"outcome": design.primary_outcome, "treatment": design.primary_variant,
                        "spec": design.primary_spec, "hac_lags": design.hac_lags[0],
                        "hac": design.raw["hac"]["kernel"]},
            "uncertainty_definition": design.raw["uncertainty_definitions"]["hac_interval"],
            "outputs": {TABLE: {"sha256": _sha256_file(stage / TABLE), "bytes": (stage / TABLE).stat().st_size,
                                "rows": len(rows)}},
            "scientific_status": "descriptive rolling conditional associations; overlapping windows; pointwise intervals with unverified coverage",
        }
        (stage / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if _verify_producer_commit(ROOT, commit) != commit:
            raise ValueError("Producer changed during the run")
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return output / "receipt.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--config", default="config/reserve_projection_2026q2.json")
    parser.add_argument("--output-dir", default="output/reports/reserve_rolling_2026q2")
    receipt = run(parser.parse_args())
    print(json.dumps({"receipt": str(receipt.relative_to(ROOT)), "sha256": _sha256_file(receipt)}))


if __name__ == "__main__":
    main()
