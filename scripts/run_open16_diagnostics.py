"""Consume frozen quarterly controls and receipted treatment legs without refitting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_open02_producer import (  # noqa: E402
    EXPECTED_OPEN01_PRODUCER_COMMIT,
    OPEN01_RECEIPT_LOCATOR,
    _file_record,
    _load_accepted_inputs,
    _project_path,
    _read_csv_rows,
    _sha256_file,
    _verify_producer_commit,
)

from ea_tdc.open01 import _fit_projection, _quarter_ordinal  # noqa: E402
from ea_tdc.open16 import (  # noqa: E402
    CANONICAL_CONTROL_IDS,
    CANONICAL_OUTCOME_ID,
    CANONICAL_TREATMENT_ID,
    DELETED_QUARTERS,
    DISCLOSURES,
    FROZEN_QUARTERS,
    ORIGINAL_ROLLING_ENDPOINTS,
    calendar_hac,
    covariance_contributions,
    join_legs,
    leg_preflight,
    pandemic_path,
    validate_panel,
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
    rolling = [r for r in rolling if r.get("outcome") == CANONICAL_OUTCOME_ID and r.get("horizon") == "0"]
    endpoints = [r.get("window_end_quarter") for r in rolling]
    if tuple(endpoints) != ORIGINAL_ROLLING_ENDPOINTS:
        raise ValueError("Frozen rolling inventory must equal the exact 57 original endpoints")
    compared = 0
    for index, stored in enumerate(rolling):
        start_ordinal = _quarter_ordinal(ORIGINAL_ROLLING_ENDPOINTS[index]) - 47
        nominal_start = f"{start_ordinal // 4}Q{start_ordinal % 4 + 1}"
        expected = {
            "window_quarters": "48", "window_start_quarter": nominal_start,
            "window_end_quarter": ORIGINAL_ROLLING_ENDPOINTS[index],
            "n": str(min(40 + index, 48)), "treatment_id": CANONICAL_TREATMENT_ID,
            "outcome": CANONICAL_OUTCOME_ID, "horizon": "0",
            "covariance_estimator": "newey_west", "covariance_lags": "1",
            "effective_sample_start": max(nominal_start, FROZEN_QUARTERS[0]),
            "effective_sample_end": ORIGINAL_ROLLING_ENDPOINTS[index],
        }
        if any(stored.get(key) != value for key, value in expected.items()):
            raise ValueError("Frozen rolling metadata differs from the exact window contract")
        subset = [r for r in panel if stored["window_start_quarter"] <= r["quarter"] <= stored["window_end_quarter"]]
        if len(subset) != int(stored["n"]):
            raise ValueError("Frozen rolling observations differ from retained output")
        fit = _fit_projection(subset, treatment_id=CANONICAL_TREATMENT_ID, outcome_id=CANONICAL_OUTCOME_ID, control_ids=CANONICAL_CONTROL_IDS)
        if (stored.get("control_ids_used") != ",".join(fit.control_ids_used)
                or stored.get("dropped_control_ids") != ",".join(fit.control_ids_rejected)):
            raise ValueError("Frozen rolling control partition differs")
        for name in ("beta", "se"):
            gaps[f"rolling_{stored['window_end_quarter']}_{name}"] = abs(getattr(fit, name) - float(stored[name]))
        compared += 1
    if any(not math.isfinite(gap) for gap in gaps.values()) or max(gaps.values()) > tolerance:
        raise ValueError(f"Stored controls fail frozen estimate equivalence: rolling={compared}, max_gap={max(gaps.values())}")
    return panel, frozen_beta, {"accepted_inputs": evidence, "estimate_equivalence_tolerance": tolerance, "estimate_gaps": gaps, "rolling_windows_compared": compared, "original_rolling_endpoints": endpoints}


AUTHORITY_LOCATOR = "config/open16_authority.json"
FACTOR_IDS = tuple(f"dflmx_k100_f{i}" for i in range(1, 5))
FACTOR_POLICY = {"k_screened": 100, "n_factors": 4, "control_policy": "balanced",
                 "min_coverage": 0.4, "selection_sample": "full_panel_including_pandemic",
                 "coordinate_identity": "exact_named_ordered_ten_decimal_scores"}


def _json_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _authority_file(root: Path, record: dict) -> dict:
    """Every admitted object has an independent SHA-256 and byte-length pin."""
    if (not isinstance(record, dict) or set(record) != {"path", "sha256", "bytes"}
            or not isinstance(record["sha256"], str) or len(record["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in record["sha256"])
            or type(record["bytes"]) is not int or record["bytes"] <= 0):
        raise ValueError("Authority requires a complete path/SHA-256/bytes record")
    actual = _file_record(root, record["path"], expected_sha256=record["sha256"])
    if _project_path(root, record["path"]).stat().st_size != record["bytes"]:
        raise ValueError("Authority object byte length differs")
    return actual


def load_authority(root: Path, commit: str) -> dict:
    """A clean committed trust root, never caller hashes or self-attested origin."""
    path = root / AUTHORITY_LOCATOR
    committed = subprocess.run(["git", "show", f"{commit}:{AUTHORITY_LOCATOR}"], cwd=root,
                               check=True, capture_output=True).stdout
    if path.read_bytes() != committed:
        raise ValueError("OPEN-16 authority differs from committed trust root")
    gate = json.loads(committed)
    if gate.get("schema_version") != "ea_tdc_open16_authority_v1" or gate.get("status") != "approved":
        raise ValueError("OPEN-16 scientific authority unavailable; factor restoration remains held")
    pins = gate.get("approved_inputs")
    if not isinstance(pins, dict):
        raise TypeError("Approved authority has no input inventory")
    _authority_file(root, gate.get("approval_receipt"))
    approval = json.loads(_project_path(root, gate["approval_receipt"]["path"]).read_text())
    if (approval.get("decision") != "approved" or approval.get("scope") != "exact_factor_coordinates"
            or approval.get("approved_inputs_sha256") != _json_digest(pins)):
        raise ValueError("Independent approval does not bind these exact inputs")
    if pins.get("factor_origin") not in {"retained_original_coordinates", "deterministically_restored_accepted_graph"}:
        raise ValueError("Factor origin is not an approved coordinate recovery")
    if pins.get("accepted_source_commit") != EXPECTED_OPEN01_PRODUCER_COMMIT:
        raise ValueError("Authority does not name the accepted source commit")
    tree = subprocess.run(["git", "rev-parse", f"{EXPECTED_OPEN01_PRODUCER_COMMIT}^{{tree}}"],
                          cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    if pins.get("accepted_source_tree") != tree or pins.get("factor_policy") != FACTOR_POLICY:
        raise ValueError("Accepted source tree or frozen factor policy differs")
    for key in ("panel", "provenance_receipt", "open01_receipt", "full_factor_scores",
                "sample_factor_scores", "ordered_screening", "ordered_top100", "equivalence",
                "legs", "legs_receipt"):
        _authority_file(root, pins.get(key))
    if pins["open01_receipt"]["path"] != OPEN01_RECEIPT_LOCATOR:
        raise ValueError("Authority must bind the accepted OPEN-01 receipt")
    graph = pins.get("graph_cut", {})
    if graph.get("kind") not in {"complete_historical_raw_universe", "historically_bound_control_universe", "original_factor_coordinates"}:
        raise ValueError("Historical graph cut is unavailable")
    files = graph.get("files", [])
    if not files or [r.get("path") for r in files] != sorted({r.get("path") for r in files}):
        raise ValueError("Graph cut needs a complete unique ordered inventory")
    for record in files:
        _authority_file(root, record)
    _authority_file(root, graph.get("completeness_evidence"))
    runtime = pins.get("runtime", {})
    if runtime.get("kind") not in {"historical_runtime_identity", "numerical_invariance_certificate"}:
        raise ValueError("Historical runtime or numerical invariance evidence is unavailable")
    _authority_file(root, runtime.get("evidence"))
    provenance = json.loads(_project_path(root, pins["provenance_receipt"]["path"]).read_text())
    # The independent approval pins this receipt; the receipt cannot approve itself.
    expected_provenance = {k: v for k, v in pins.items() if k != "provenance_receipt"}
    if provenance != expected_provenance:
        raise ValueError("Provenance receipt differs from independently approved input inventory")
    receipt = json.loads(_project_path(root, pins["legs_receipt"]["path"]).read_text())
    producer = pins.get("upstream_producer_commit", "")
    if (len(producer) != 40 or any(c not in "0123456789abcdef" for c in producer)
            or receipt.get("producer_commit") != producer
            or receipt.get("schema_version") != "regression_legs_v1"
            or receipt.get("output", {}).get("sha256") != pins["legs"]["sha256"]
            or receipt.get("units") != "USD million"
            or receipt.get("sample") != {"start": "2002Q1", "end": "2025Q4", "n": 96}):
        raise ValueError("Approved treatment-leg receipt identity, units, sample or producer differs")
    return pins


def validate_coordinate_evidence(root: Path, rows: list[dict], pins: dict) -> None:
    """Check approved coordinate and independent full-vector reference objects."""
    full = _read_csv_rows(_project_path(root, pins["full_factor_scores"]["path"]))
    sample = _read_csv_rows(_project_path(root, pins["sample_factor_scores"]["path"]))
    full_quarters = [r["quarter"] for r in full]
    if (len(full) != 320 or full_quarters != pins.get("full_factor_quarters")
            or any(_quarter_ordinal(b) - _quarter_ordinal(a) != 1 for a, b in pairwise(full_quarters))
            or tuple(r["quarter"] for r in sample) != FROZEN_QUARTERS):
        raise ValueError("Factor scores require the approved 320-quarter grid and exact 96-quarter sample")
    for record in full + sample:
        if tuple(record) != ("quarter", *FACTOR_IDS):
            raise ValueError("Factor coordinate names or ordering differ")
        for key in FACTOR_IDS:
            value = float(record[key])
            if not math.isfinite(value) or record[key] != f"{value:.10f}":
                raise ValueError("Factor scores must preserve finite ten-decimal coordinate strings")
    if [r for r in full if r["quarter"] in FROZEN_QUARTERS] != sample:
        raise ValueError("Full and frozen-sample factor coordinates differ")
    if any(float(score[key]) != row[key] for score, row in zip(sample, rows, strict=True) for key in FACTOR_IDS):
        raise ValueError("Panel differs from approved factor coordinates")
    reference = json.loads(_project_path(root, pins["equivalence"]["path"]).read_text())
    if (reference.get("panel_sha256") != pins["panel"]["sha256"]
            or reference.get("original_rolling_endpoints") != list(ORIGINAL_ROLLING_ENDPOINTS)
            or len(reference.get("windows", [])) != 57):
        raise ValueError("Equivalence reference must bind the panel and exact 57 windows")
    for endpoint, stored in zip(ORIGINAL_ROLLING_ENDPOINTS, reference["windows"], strict=True):
        start = _quarter_ordinal(endpoint) - 47
        nominal = [f"{i // 4}Q{i % 4 + 1}" for i in range(start, start + 48)]
        if stored.get("nominal_quarters") != nominal:
            raise ValueError("Equivalence nominal quarter inventory differs")
        for lane in ("no_deletion", "calendar_deletion"):
            observed = [r for r in rows if r["quarter"] in nominal and (lane == "no_deletion" or r["quarter"] not in DELETED_QUARTERS)]
            fit = _fit_projection(observed, treatment_id=CANONICAL_TREATMENT_ID, outcome_id=CANONICAL_OUTCOME_ID, control_ids=CANONICAL_CONTROL_IDS)
            expected = stored.get(lane, {})
            if (expected.get("observed_quarters") != [r["quarter"] for r in observed]
                    or expected.get("controls_used") != list(fit.control_ids_used)
                    or expected.get("controls_rejected") != list(fit.control_ids_rejected)
                    or expected.get("covariance_lags") != 1):
                raise ValueError("Equivalence observed inventory or control/HAC metadata differs")
            actual = fit.fit
            if lane == "calendar_deletion":
                x = [[1.0, r[CANONICAL_TREATMENT_ID], *[r[c] for c in fit.control_ids_used]] for r in observed]
                actual = calendar_hac([r[CANONICAL_OUTCOME_ID] for r in observed], x, [_quarter_ordinal(r["quarter"]) for r in observed])
            for key, values in (("coefficients", actual.beta), ("fitted", actual.fitted),
                                ("residuals", actual.residuals), ("covariance", actual.covariance),
                                ("treatment_beta_se", [actual.beta[1], actual.ses[1]])):
                supplied = np.asarray(expected.get(key), dtype=float)
                target = np.asarray(values)
                if supplied.shape != target.shape or not np.isfinite(supplied).all() or not np.allclose(supplied, target, rtol=0, atol=1e-7):
                    raise ValueError(f"Full-vector equivalence differs: {endpoint} {lane} {key}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Refusing an empty diagnostic table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    commit = _verify_producer_commit(ROOT, args.producer_commit)
    pins = load_authority(ROOT, commit)
    panel_record = pins["panel"]
    panel_path = _project_path(ROOT, panel_record["path"])
    old_receipt = json.loads(_project_path(ROOT, pins["open01_receipt"]["path"]).read_text())
    rows, frozen_beta, checks = validate_frozen_controls(ROOT, _read_csv_rows(panel_path), old_receipt)
    validate_coordinate_evidence(ROOT, rows, pins)
    legs_path = _project_path(ROOT, pins["legs"]["path"])
    leg_receipt = json.loads(_project_path(ROOT, pins["legs_receipt"]["path"]).read_text())
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
        inputs = {"authority": _file_record(ROOT, AUTHORITY_LOCATOR), "approved_inputs": pins}
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
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--output-dir", default="output/reports/open16")
    receipt = run(parser.parse_args())
    print(json.dumps({"receipt": str(receipt.relative_to(ROOT)), "sha256": _sha256_file(receipt)}))


if __name__ == "__main__":
    main()
