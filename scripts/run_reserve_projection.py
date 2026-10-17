#!/usr/bin/env python3
"""Produce the reserve-side projection tables from a clean committed checkout.

Reads only the hash-pinned TDCest reserve-factor objects declared in the
selected design (default config/reserve_projection.json; a refresh vintage has
its own config and TDCest refresh receipt), verifies that the frozen deposit
producers are byte-identical, and writes versioned tables plus a receipt binding
inputs, producer commit, runtime and output hashes.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_open02_producer import _project_path, _sha256_file, _verify_producer_commit  # noqa: E402

from ea_tdc import reserve_projection as rp  # noqa: E402

DEFAULT_OUTPUT = "output/reports/reserve_projection_v1"

TABLES = {
    "estimates": "reserve_projection_estimates_v1.csv",
    "contributions": "reserve_projection_contributions_v1.csv",
    "grouped_contributions": "reserve_projection_grouped_contributions_v1.csv",
    "leg_factor_matrix": "reserve_projection_leg_factor_matrix_v1.csv",
    "influence": "reserve_projection_influence_v1.csv",
    "era_contrasts": "reserve_projection_era_contrasts_v1.csv",
    "calibration": "reserve_projection_calibration_v1.csv",
    "transition_quarters": "reserve_projection_transition_quarters_v1.csv",
    "thin_era_table_y_h": "thin/reserve_era_table_y_h_v1.csv",
    "thin_era_table_y_z": "thin/reserve_era_table_y_z_v1.csv",
    "thin_grouped_contributions_y_h": "thin/reserve_grouped_contributions_y_h_v1.csv",
    "thin_grouped_contributions_y_z": "thin/reserve_grouped_contributions_y_z_v1.csv",
}
# Written only when the design declares balance coefficients.
BALANCE_TABLES = {
    "balance_coefficients": "reserve_projection_balance_coefficients_v1.csv",
    "thin_balance_coefficients": "thin/reserve_balance_coefficients_v1.csv",
}


def _headline(design: rp.Design, result: dict) -> dict:
    lag = design.hac_lags[0]
    estimates = [
        {k: r[k] for k in ("outcome", "treatment", "sample", "spec", "n", "beta", f"se_hac{lag}",
                           "bootstrap_lower", "bootstrap_upper", "partial_variation_share",
                           "bootstrap_uncertainty_status", "hac_uncertainty_status", "coverage_minimum",
                           "coverage_minimum_mc_se", "coverage_evidence", "calibration_scope")}
        for r in result["tables"]["estimates"] if r["treatment"] == design.primary_variant]
    joint = [r for r in result["tables"]["era_contrasts"]
             if r["row_type"] == "joint" and r["treatment"] == design.primary_variant]
    balances = [r for r in result["tables"].get("balance_coefficients", [])
                if r["treatment"] == design.primary_variant and r["spec"] == design.primary_spec]
    return {"primary": {"outcome": design.primary_outcome, "treatment": design.primary_variant,
                        "spec": design.primary_spec, "hac_lags": lag},
            "estimates": json.loads(json.dumps(estimates, default=float)),
            "era_joint_tests": json.loads(json.dumps(joint, default=float)),
            "balance_coefficients": json.loads(json.dumps(balances, default=float))}


def _verify_source(root: Path) -> None:
    """The receipted commit must own the estimator that actually ran."""
    try:
        Path(rp.__file__).resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Imported estimator is outside the receipted checkout; run with this checkout's src first on the path") from exc


def run(args: argparse.Namespace) -> Path:
    commit = _verify_producer_commit(ROOT, args.producer_commit)
    _verify_source(ROOT)
    design = rp.load_design(ROOT, args.config)
    frozen = rp.verify_frozen_sources(ROOT, design)
    panel = rp.load_panel(ROOT, design)
    result = rp.produce(panel, design)
    tables = TABLES | (BALANCE_TABLES if design.balances else {})
    output = _project_path(ROOT, args.output_dir or design.raw.get("output_dir", DEFAULT_OUTPUT))
    if output.exists():
        raise ValueError("Output directory already exists; preserve or adjudicate it before rerunning")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".reserve-projection-", dir=output.parent))
    try:
        (stage / "thin").mkdir()
        for key, name in tables.items():
            rp.write_table(stage / name, result["tables"][key])
        outputs = {name: {"sha256": _sha256_file(stage / name), "bytes": (stage / name).stat().st_size}
                   for name in tables.values()}
        inputs = {key: {**record, "upstream_producer_commit": design.raw["upstream"]["producer_commit"]}
                  for key, record in design.raw["inputs"].items()}
        receipt = {
            "schema_version": "ea_tdc_reserve_projection_receipt_v1",
            "producer_commit": commit,
            "dependency_lock_sha256": _sha256_file(ROOT / "uv.lock"),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "machine": platform.machine(), "system": platform.system()},
            "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": {"path": args.config, "sha256": design.sha256},
            "vintage": design.raw.get("vintage", "admitted"),
            "inputs": inputs,
            "input_cross_checks": panel.checks,
            "frozen_deposit_sources": {"status": "byte_identical", "files": frozen},
            "numerical_checks": result["gaps"] | {"cross_check_tolerance": rp.CROSS_CHECK_TOLERANCE},
            "bootstrap_representation": result["bootstrap_representation"],
            "uncertainty_definitions": design.raw["uncertainty_definitions"],
            "calibration_results": result["tables"]["calibration"],
            "design": {key: design.raw[key] for key in (
                "scope", "units", "sample", "eras", "transition_quarters", "era_declaration", "outcomes",
                "treatments", "legs", "specifications", "hac", "bootstrap", "calibration", "influence",
                "component_groups", "balance_coefficients") if key in design.raw},
            "headline": _headline(design, result),
            "outputs": outputs,
            "scientific_status": "descriptive conditional associations; era differences remain descriptive only; joint-test null-size checks establish no equality, power or interval coverage",
            "disclosures": design.raw["disclosures"],
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
    parser.add_argument("--config", default=rp.CONFIG_LOCATOR, help="Design locator; refresh vintages use their own config")
    parser.add_argument("--output-dir", default=None, help="Defaults to the design's output_dir, else the admitted v1 directory")
    receipt = run(parser.parse_args())
    print(json.dumps({"receipt": str(receipt.relative_to(ROOT)), "sha256": _sha256_file(receipt)}))


if __name__ == "__main__":
    main()
