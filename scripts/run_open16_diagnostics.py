"""Consume frozen quarterly controls and receipted treatment legs without refitting."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ea_tdc.open01 import _fit_projection  # noqa: E402
from ea_tdc.open16 import (  # noqa: E402
    CANONICAL_CONTROL_IDS, CANONICAL_OUTCOME_ID, CANONICAL_TREATMENT_ID,
    DISCLOSURES, FROZEN_QUARTERS, covariance_contributions, join_legs,
    leg_preflight, pandemic_path, validate_panel,
)
from run_open02_producer import (  # noqa: E402
    _file_record, _load_accepted_inputs, _project_path,
    _read_csv_rows, _sha256_file, _verify_producer_commit, OPEN01_RECEIPT_LOCATOR,
)


def validate_frozen_controls(root: Path, rows: list[dict], receipt: dict) -> tuple[list[dict], float, dict]:
    """Bind inputs and recover archived estimates, without running factor code."""
    panel = validate_panel(rows)
    design, _, evidence = _load_accepted_inputs(root)
    by_quarter = {r["quarter"]: r for r in design}
    for row in panel:
        accepted = by_quarter[row["quarter"]]
        for key in (CANONICAL_TREATMENT_ID, CANONICAL_OUTCOME_ID, *CANONICAL_CONTROL_IDS):
            if key in accepted and accepted[key] != "" and float(accepted[key]) != row[key]:
                raise ValueError(f"Frozen base value changed: {row['quarter']} {key}")
    fixed = receipt["retained_outputs"]["fixed"]
    for key in ("same_quarter_headline", "rolling_estimates"):
        record = fixed[key]
        _file_record(root, record["path"], expected_sha256=record["sha256"])
    headlines = _read_csv_rows(root / fixed["same_quarter_headline"]["path"])
    headline = [r for r in headlines if r["outcome_id"] == CANONICAL_OUTCOME_ID and r["treatment_id"] == CANONICAL_TREATMENT_ID]
    if len(headline) != 1:
        raise ValueError("Frozen headline is not unique")
    estimate = _fit_projection(panel, treatment_id=CANONICAL_TREATMENT_ID, outcome_id=CANONICAL_OUTCOME_ID, control_ids=CANONICAL_CONTROL_IDS)
    frozen_beta = float(headline[0]["beta"])
    # Equivalence is a necessary check; upstream panel provenance is also required.
    tolerance = 1e-7
    gaps = {"headline_beta": abs(estimate.beta - frozen_beta), "headline_se": abs(estimate.se - float(headline[0]["se"]))}
    rolling = _read_csv_rows(root / fixed["rolling_estimates"]["path"])
    compared = 0
    endpoints = []
    for stored in rolling:
        if stored["outcome"] != CANONICAL_OUTCOME_ID or stored["horizon"] != "0":
            continue
        subset = [r for r in panel if stored["window_start_quarter"] <= r["quarter"] <= stored["window_end_quarter"]]
        if len(subset) != int(stored["n"]):
            raise ValueError("Frozen rolling observations differ from retained output")
        endpoints.append(stored["window_end_quarter"])
        fit = _fit_projection(subset, treatment_id=CANONICAL_TREATMENT_ID, outcome_id=CANONICAL_OUTCOME_ID, control_ids=CANONICAL_CONTROL_IDS)
        for name in ("beta", "se"):
            gaps[f"rolling_{stored['window_end_quarter']}_{name}"] = abs(getattr(fit, name) - float(stored[name]))
        compared += 1
    if not endpoints or len(set(endpoints)) != compared or endpoints != sorted(endpoints) or max(gaps.values()) > tolerance:
        raise ValueError(f"Stored controls fail frozen estimate equivalence: rolling={compared}, max_gap={max(gaps.values())}")
    return panel, frozen_beta, {"accepted_inputs": evidence, "estimate_equivalence_tolerance": tolerance, "estimate_gaps": gaps, "rolling_windows_compared": compared, "original_rolling_endpoints": endpoints}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Refusing an empty diagnostic table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    commit = _verify_producer_commit(ROOT, args.producer_commit)
    panel_path = _project_path(ROOT, args.frozen_panel)
    panel_record = _file_record(ROOT, args.frozen_panel, expected_sha256=args.frozen_panel_sha256)
    provenance = json.loads(_project_path(ROOT, args.frozen_panel_receipt).read_text())
    old_receipt_path = ROOT / OPEN01_RECEIPT_LOCATOR
    if provenance.get("panel_sha256") != panel_record["sha256"] or provenance.get("open01_receipt_sha256") != _sha256_file(old_receipt_path):
        raise ValueError("Frozen panel provenance must bind the panel bytes and accepted OPEN-01 receipt")
    if provenance.get("factor_values_origin") != "retained_accepted_realizations":
        raise ValueError("Frozen controls require retained accepted realizations, not rebuilt factors")
    old_receipt = json.loads(old_receipt_path.read_text())
    rows, frozen_beta, checks = validate_frozen_controls(ROOT, _read_csv_rows(panel_path), old_receipt)
    legs_path = _project_path(ROOT, args.legs)
    leg_receipt_path = _project_path(ROOT, args.legs_receipt)
    leg_receipt_record = _file_record(ROOT, args.legs_receipt, expected_sha256=args.legs_receipt_sha256)
    leg_receipt = json.loads(leg_receipt_path.read_text())
    # The exact upstream receipt itself is pinned; its output hash must also bind CSV bytes.
    leg_hash = _sha256_file(legs_path)
    if leg_hash != args.legs_sha256:
        raise ValueError("Treatment-leg CSV hash differs from expected upstream bytes")
    if (leg_receipt.get("schema_version") != "regression_legs_v1"
            or leg_receipt.get("output", {}).get("sha256") != leg_hash
            or leg_receipt.get("units") != "USD million"
            or leg_receipt.get("sample") != {"start": "2002Q1", "end": "2025Q4", "n": 96}):
        raise ValueError("Treatment-leg receipt schema, units, sample, or output hash is invalid")
    rows = join_legs(rows, _read_csv_rows(legs_path))
    preflight = leg_preflight(rows, leg_receipt["source_resolution"])
    if preflight["status"] == "passed":
        raise ValueError("Leg preflight passed: gated estimation and seven-test inference must be implemented before accepting this disposition")
    covariance = covariance_contributions(rows, frozen_beta)
    rolling = pandemic_path(rows, nominal_endpoints=checks["original_rolling_endpoints"])
    output = _project_path(ROOT, args.output_dir)
    if output.exists():
        raise ValueError("Output directory already exists; preserve or adjudicate it before rerunning")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".open16-", dir=output.parent))
    try:
        _write_csv(stage / "leg_covariance_contributions.csv", covariance)
        _write_csv(stage / "pandemic_calendar_deletion.csv", rolling)
        (stage / "leg_preflight.json").write_text(json.dumps(preflight, indent=2, allow_nan=False) + "\n")
        inputs = {"frozen_panel": panel_record, "frozen_panel_receipt": _file_record(ROOT, args.frozen_panel_receipt),
                  "legs": _file_record(ROOT, args.legs), "legs_receipt": leg_receipt_record}
        outputs = {p.name: {"sha256": _sha256_file(p), "bytes": p.stat().st_size} for p in sorted(stage.iterdir())}
        receipt = {"schema_version": "ea_tdc_frozen_diagnostics_v1", "producer_commit": commit,
                   "dependency_lock_sha256": _sha256_file(ROOT / "uv.lock"),
                   "status": "completed_with_failed_leg_preflight", "sample": list(FROZEN_QUARTERS),
                   "inputs": inputs, "outputs": outputs, "frozen_controls": checks,
                   "leg_estimates_computed": False, "leg_preflight_reason_codes": preflight["reason_codes"],
                   "scientific_status": "appendix_only", **DISCLOSURES}
        (stage / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        if _verify_producer_commit(ROOT, commit) != commit:
            raise ValueError("Producer changed during diagnostics")
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return output / "receipt.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("producer-commit", "frozen-panel", "frozen-panel-sha256", "frozen-panel-receipt", "legs", "legs-sha256", "legs-receipt", "legs-receipt-sha256"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--output-dir", default="output/reports/open16")
    receipt = run(parser.parse_args())
    print(json.dumps({"receipt": str(receipt.relative_to(ROOT)), "sha256": _sha256_file(receipt)}))


if __name__ == "__main__":
    main()
