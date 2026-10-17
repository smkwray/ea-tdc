#!/usr/bin/env python3
"""Produce kernel-weighted local deposit and reserve projections from a clean committed checkout.

Reads the sealed OPEN-01 deposit panel and the pinned reserve design inputs,
verifies the frozen deposit sources, proves that uniform kernels reproduce every
admitted rolling row and both full-sample estimates, then writes the declared
local paths, the coverage calibration and a receipt binding every hash.
"""
from __future__ import annotations

import argparse
import csv
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from run_open02_producer import _project_path, _sha256_file, _verify_producer_commit  # noqa: E402

from ea_tdc import local_projection as lp  # noqa: E402
from ea_tdc import reserve_projection as rp  # noqa: E402

CONFIG = "config/local_projection_2026q2.json"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def deposit_parity_inputs(root: Path, spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rolling_spec, full_spec = spec["parity"]["rolling"], spec["parity"]["full_sample"]
    rolling = [{"end": r["window_end_quarter"], "n": int(r["n"]), "beta": float(r["beta"]), "se": float(r["se"]),
                "controls": r["control_ids_used"].split(",")}
               for r in _rows(lp.pinned(root, rolling_spec)) if r["outcome"] == rolling_spec["outcome_filter"]]
    if len(rolling) != rolling_spec["rows"]:
        raise ValueError("Deposit rolling table row count differs from the declared parity set")
    (full,) = [r for r in _rows(lp.pinned(root, full_spec)) if r["estimand_role"] == full_spec["estimand_role"]]
    return rolling, {"beta": float(full["beta"]), "se": float(full["se"])}


def reserve_parity_inputs(root: Path, spec: dict[str, Any], design: rp.Design,
                          source: rp.Panel) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rolling_spec = spec["parity"]["rolling"]
    lag = design.hac_lags[0]
    rolling = [{"end": r["window_end_quarter"], "n": int(r["n"]), "beta": float(r["beta"]), "se": float(r[f"se_hac{lag}"])}
               for r in _rows(lp.pinned(root, rolling_spec))]
    if len(rolling) != rolling_spec["rows"]:
        raise ValueError("Reserve rolling table row count differs from the declared parity set")
    cell = rp.make_cell(source, design, design.primary_outcome, design.primary_variant, "pooled",
                        design.primary_spec, design.sample)
    fit = rp.fit_cell(source, design, cell)
    return rolling, {"beta": float(fit["beta"]), "se": float(fit["hac"][lag]["se"])}


def produce(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    lp.validate(config)
    estimator, panels = config["estimator"], config["panels"]
    numerical = estimator["numerical"]
    deposit = lp.load_deposit(root, panels["deposit"])
    reserve, design, source = lp.load_reserve(root, panels["reserve"])
    frozen = rp.verify_frozen_sources(root, design)
    tolerance = config["parity_tolerance"]
    parity = {"deposit": lp.parity(deposit, *deposit_parity_inputs(root, panels["deposit"]),
                                   panels["deposit"]["parity"]["rolling"]["window_quarters"], tolerance, numerical),
              "reserve": lp.parity(reserve, *reserve_parity_inputs(root, panels["reserve"], design, source),
                                   panels["reserve"]["parity"]["rolling"]["window_quarters"], tolerance, numerical)}
    primary = int(estimator["primary_half_width"])
    widths = [(primary, "primary"), *((int(h), "sensitivity") for h in estimator["sensitivity_half_widths"])]
    estimates, calibration, summary = [], [], {}
    for index, panel in enumerate((deposit, reserve)):
        rows = [row for h, role in widths for row in lp.local_rows(panel, h, role, config["influence"]["flags"], numerical)]
        targets = [r["quarter"] for r in rows if r["role"] == "primary" and r["status"] == "reported"]
        nuisance = panel.controls[:, panel.control_ids.index(panels[panel.name]["nuisance_drift_control"])]
        panel_calibration = lp.calibrate(panel, index, primary, targets, config["calibration"], nuisance, numerical)
        status = lp.coverage_status(panel_calibration)
        summary[panel.name] = {"targets": len(rows) // len(widths),
                               **{f"{role}_h{h}_{s}": sum(r["half_width"] == h and r["status"] == s for r in rows)
                                  for h, role in widths for s in sorted({r["status"] for r in rows})},
                               **{c: sum(v["coverage_status"] == c for v in status.values()) for c in lp.COVERAGE}}
        for row in rows:
            if row["status"] != "reported":
                row["coverage_status"] = ""
            elif row["role"] == "primary":
                row |= status[row["quarter"]]
            else:
                row["coverage_status"] = "coverage_unverified"
        estimates += rows
        calibration += panel_calibration
    return {"estimates": estimates, "calibration": calibration, "parity": parity, "frozen": frozen, "summary": summary,
            "reserve_design_sha256": design.sha256, "reserve_checks": source.checks}


def run(args: argparse.Namespace) -> Path:
    commit = _verify_producer_commit(ROOT, args.producer_commit)
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    lp.validate(config)
    output = _project_path(ROOT, args.output_dir or config["output_dir"])
    if output.exists():
        raise ValueError("Output directory already exists; preserve or adjudicate it before rerunning")
    result = produce(ROOT, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".local-projection-", dir=output.parent))
    try:
        names = config["outputs"]
        rp.write_table(stage / names["estimates"], result["estimates"])
        rp.write_table(stage / names["calibration"], result["calibration"])
        deposit = config["panels"]["deposit"]
        receipt = {
            "schema_version": "ea_tdc_local_projection_receipt_v1",
            "producer_commit": commit,
            "dependency_lock_sha256": _sha256_file(ROOT / "uv.lock"),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "machine": platform.machine(), "system": platform.system()},
            "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": {"path": args.config, "sha256": _sha256_file(config_path)},
            "authority": config["authority"],
            "inputs": {"deposit_panel": deposit["panel"], "deposit_panel_receipt": deposit["panel_receipt"],
                       "sealed_refresh_receipt": deposit["sealed_refresh_receipt"],
                       "deposit_rolling": deposit["parity"]["rolling"], "deposit_headline": deposit["parity"]["full_sample"],
                       "reserve_design": {"path": config["panels"]["reserve"]["design_config"],
                                          "sha256": result["reserve_design_sha256"]},
                       "reserve_rolling": config["panels"]["reserve"]["parity"]["rolling"]},
            "reserve_input_cross_checks": result["reserve_checks"],
            "frozen_deposit_sources": {"status": "byte_identical", "files": result["frozen"]},
            "parity": result["parity"],
            "summary": result["summary"],
            "estimator": config["estimator"],
            "influence": config["influence"],
            "calibration": config["calibration"],
            "outputs": {name: {"sha256": _sha256_file(stage / name), "bytes": (stage / name).stat().st_size}
                        for name in names.values()},
            "scientific_status": "descriptive retrospective kernel-weighted conditional projections; pointwise intervals; coverage checked only for the declared DGPs at the primary bandwidth",
            "scope": config["scope"],
        }
        (stage / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False, default=float) + "\n",
                                            encoding="utf-8")
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
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--output-dir", default=None)
    receipt = run(parser.parse_args())
    print(json.dumps({"receipt": str(receipt.relative_to(ROOT)), "sha256": _sha256_file(receipt)}))


if __name__ == "__main__":
    main()
