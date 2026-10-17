#!/usr/bin/env python3
"""Build versioned descriptive tables from a clean committed producer."""
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
from run_open02_producer import _project_path, _verify_producer_commit  # noqa: E402
from ea_tdc import matched_clock_reserve as mc  # noqa: E402
from ea_tdc import reserve_projection as rp  # noqa: E402


DEFAULT_OUTPUT = "output/reports/matched_clock_reserve_v1"


def verify_admitted(root: Path, config: dict) -> dict:
    """Both contents and the full file inventory must match the pinned reserve-projection snapshot."""
    records = config["admitted_outputs"]
    receipts = [Path(k).parent for k in records if Path(k).name == "receipt.json"]
    if len(receipts) != 1:
        raise ValueError("Pinned reserve-projection outputs must contain exactly one receipt")
    directory = root / receipts[0]
    actual = {p.relative_to(root).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != set(records):
        raise ValueError("Admitted output inventory changed")
    for locator, record in records.items():
        mc.verify(root / locator, record)
    return records


def load(root: Path, upstream: Path | None, authorities: dict, locator: str = mc.CONFIG) -> tuple[dict, dict, dict]:
    config_path = root / locator
    config = json.loads(config_path.read_text())
    if config["schema_version"] != "ea_tdc_matched_clock_reserve_v1" or config["controls"] != ["quarter_of_year"]:
        raise ValueError("Matched-clock design differs")
    mc.verify(root / config["admitted_config"]["path"], config["admitted_config"])
    for record in config["admitted_inputs"].values():
        rp._pinned(root, record)
    for record in config["validation_inputs"].values():
        rp._pinned(root, record)
    for key, record in config["authorities"].items():
        mc.verify(authorities[key], record)
    if config.get("vintage"):
        # A refresh vintage reads byte-exact bundle copies bound by the TDCest refresh receipt.
        refresh = rp.verify_refresh_receipt(root, config["inputs"], config["upstream_producer_commit"], config["vintage"])
        input_root = root
    else:
        if upstream is None:
            raise ValueError("The admitted design reads a TDCest checkout; pass --tdcest-root")
        _verify_producer_commit(upstream, config["upstream_commit"])
        refresh, input_root = None, upstream
    files = {key: rp._pinned(input_root, record) for key, record in config["inputs"].items()}
    weekly_receipt = json.loads(files["weekly_receipt"].read_text())
    if weekly_receipt["producer_commit"] != config["upstream_producer_commit"] or weekly_receipt["units"] != config["units"]:
        raise ValueError("Weekly receipt producer or units differ")
    for key in ("weekly", "quarterly", "identity"):
        if weekly_receipt["outputs"][key]["sha256"] != config["inputs"][key]["sha256"]:
            raise ValueError("Weekly receipt does not bind the consumed object")
    design = rp.load_design(root, config["admitted_config"]["path"])
    if design.raw.get("vintage") != config.get("vintage"):
        raise ValueError("Matched-clock and reserve-projection vintages differ")
    admitted = rp.load_panel(root, design)
    panel = mc.build_panel(config, admitted, rp._read_csv(files["quarterly"]), rp._read_csv(files["weekly"]), rp._read_csv(files["identity"]))
    return config, panel, {"weekly_producer": weekly_receipt["producer_commit"], "admitted_primary_variant": design.primary_variant,
                           "config_sha256": rp._sha256(config_path), "refresh_receipt": refresh}


def run(args: argparse.Namespace) -> Path:
    commit = _verify_producer_commit(ROOT, args.producer_commit)
    Path(mc.__file__).resolve().relative_to(ROOT.resolve())
    authorities = {"adjudication": args.adjudication, "design_memo": args.design_memo}
    config, panel, checks = load(ROOT, args.tdcest_root, authorities, args.config)
    preserved = verify_admitted(ROOT, config)
    result = mc.produce(config, panel)
    declared = config.get("output_dir", DEFAULT_OUTPUT)
    output = _project_path(ROOT, args.output_dir or declared)
    if output.exists():
        raise ValueError("Output directory already exists; preserve it before rerunning")
    if output != ROOT / declared:
        raise ValueError("Only the declared matched-clock output directory is writable")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".matched-clock-", dir=output.parent))
    try:
        for name, rows in result.items():
            rp.write_table(stage / f"{name}_v1.csv", rows)
        outputs = {p.name: {"sha256": rp._sha256(p), "bytes": p.stat().st_size} for p in sorted(stage.iterdir())}
        receipt = {"schema_version": "ea_tdc_matched_clock_reserve_receipt_v1", "producer_commit": commit,
                   "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "runtime": {"python": platform.python_version(), "numpy": np.__version__, "system": platform.system()},
                   "config": {"path": args.config, "sha256": checks["config_sha256"]},
                   "vintage": config.get("vintage", "admitted"), "refresh_receipt": checks["refresh_receipt"],
                   "dependency_lock": {"path": "uv.lock", "sha256": rp._sha256(ROOT / "uv.lock")},
                   "inputs": config["inputs"], "admitted_inputs": config["admitted_inputs"],
                   "validation_inputs": config["validation_inputs"],
                   "admitted_config": config["admitted_config"],
                   "source_roots": {"ea_tdc": str(ROOT),
                                    "tdcest": str(args.tdcest_root.resolve()) if args.tdcest_root else "bundle_copies_bound_by_refresh_receipt"},
                   "design_authorities": {key: {**record, "path": str(path.resolve()), "config_locator": record["path"]}
                                          for key, record, path in (
                                              ("adjudication", config["authorities"]["adjudication"], args.adjudication),
                                              ("design_memo", config["authorities"]["design_memo"], args.design_memo))},
                   "upstream_checkout_commit": config.get("upstream_commit"), "upstream_producer_commit": checks["weekly_producer"],
                   "admitted_primary_variant": checks["admitted_primary_variant"], "admitted_outputs_unchanged": preserved,
                   "sample": config["sample"], "quarter_mask": list(panel["quarters"]),
                   "excluded_quarters": panel["excluded"], "complete_quarters_outside_sample": panel["outside_sample"],
                   "units": config["units"], "controls": config["controls"],
                   "variants": config["variants"], "samples": config["samples"], "components": config["components"],
                   "inference": config["inference"], "outputs": outputs,
                   "checks": {"common_outcome": "verified", "weekly_sums_and_telescoping": "verified",
                              "complete_quarters_only": "verified", "quarterly_identities": "verified",
                              "matched_clock_bridge": "exactly_zero",
                              "maximum_contribution_sum_error": max(abs(r["contribution_sum"] - r["beta_minus_one"]) for r in result["projections"] if r["beta"] is not None)},
                   "interpretation": "A matched exposure clock removes the Treasury-deposit clock bridge, not unknown secondary prices, coupon entitlement, route ambiguity or leverage. Exact same-clock TRC has accounting coefficient one; these are measured-exposure conditional projections. Quarterly aggregation changes the projection functional. Variance shares are Cov(component,total difference)/Var(total difference); own-variance shares are also provided and need not sum to one. Covariances use n normalization; partial means residualized on intercept and quarter of year. Singleton transitions have no identified conditional slope and are reported as raw accounting rows."}
        (stage / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        # Verify consumed pins and protected artifacts again before installing results.
        load(ROOT, args.tdcest_root, authorities, args.config)
        verify_admitted(ROOT, config)
        _verify_producer_commit(ROOT, commit)
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return output / "receipt.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--config", default=mc.CONFIG, help="Design locator; refresh vintages use their own config")
    parser.add_argument("--tdcest-root", type=Path, help="TDCest checkout; required only by the admitted design")
    parser.add_argument("--adjudication", required=True, type=Path)
    parser.add_argument("--design-memo", required=True, type=Path)
    parser.add_argument("--output-dir", default=None, help="Must equal the design's declared output directory")
    receipt = run(parser.parse_args())
    print(json.dumps({"receipt": str(receipt.relative_to(ROOT)), "sha256": rp._sha256(receipt)}))


if __name__ == "__main__":
    main()
