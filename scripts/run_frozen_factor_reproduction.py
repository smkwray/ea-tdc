"""Record a new fixed-method reproduction from a frozen control-universe graph cut."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "config/frozen_factor_reproduction.json"
ORIGIN = "new_frozen_input_reproduction"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"Missing or duplicated CSV header: {path.name}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path.name}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Refusing an empty reproduction table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def record(root: Path, path: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}


def copy_pinned(root: Path, destination: Path, expected: dict) -> dict:
    relative = Path(expected["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Frozen input locator must be a canonical relative path")
    source = (root / relative).resolve()
    source.relative_to(root.resolve())
    if relative.is_absolute() or sha256(source) != expected["sha256"]:
        raise ValueError(f"Frozen input hash/path mismatch: {relative}")
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha256(target) != expected["sha256"]:
        raise ValueError(f"Frozen input changed during copy: {relative}")
    return record(destination, target)


def validate_universe(anchor: list[dict], universe: list[dict], columns: list[dict]) -> list[str]:
    grid = [r["quarter"] for r in anchor]
    if len(grid) != 320 or len(set(grid)) != 320 or grid != sorted(grid):
        raise ValueError("Expected the exact ordered 320-row design grid")
    features = list(universe[0])[1:]
    if next(iter(universe[0])) != "quarter" or [r["quarter"] for r in universe] != grid:
        raise ValueError("Control universe and design grids differ")
    if features != [r["feature_id"] for r in columns] or len(features) != len(set(features)):
        raise ValueError("Ordered columns must equal the control-universe CSV header")
    for row in universe:
        for key in features:
            if row[key] != "" and not math.isfinite(float(row[key])):
                raise ValueError("Control universe contains nonfinite values")
    return features


def compute_snapshot(stage: Path) -> None:
    """Called in a fresh interpreter with only the archived method on its path."""
    method = stage / "method"
    sys.path[:0] = [str(method / "src"), str(method / "scripts")]
    from run_pinned_factor_residual_bridge import (
        ANCHOR_JOB_ID,
        CONTROL_POLICY_MODE,
        FACTOR_COUNT,
        K_SCREENED,
        MERGE_JOBS,
        _merge_by_quarter,
    )
    from run_tier2_credit_lead_diagnostics import (
        CREDIT_LAG_SOURCES,
        LAG_PERIODS,
        RATE_RISK_LAG_SOURCES,
        _add_lags,
    )

    from ea_tdc.open01 import _fit_projection, _quarter_ordinal
    from ea_tdc.open_contract import (
        CANONICAL_CONTROL_IDS,
        CANONICAL_OUTCOME_ID,
        CANONICAL_TREATMENT_ID,
    )
    from ea_tdc.paths import project_paths
    from ea_tdc.robustness import (
        _apply_control_policy,
        _extract_factor_controls,
        _load_alt_treatments,
        _merge_control_rows,
        _screen_features,
    )

    inputs, outputs = stage / "inputs", stage / "results"
    config = json.loads((inputs / CONFIG).read_text())
    accepted = json.loads((inputs / config["accepted_receipt"]["path"]).read_text())
    if (config["origin"] != ORIGIN or list(CANONICAL_CONTROL_IDS) != accepted["contract"]["control_ids"]
            or CANONICAL_TREATMENT_ID != accepted["contract"]["treatment_id"]
            or list(MERGE_JOBS) != accepted["contract"]["design_job_ids"]
            or (K_SCREENED, FACTOR_COUNT, CONTROL_POLICY_MODE) != (100, 4, "balanced")):
        raise ValueError("Frozen method contract differs")
    design = accepted["designs"][ANCHOR_JOB_ID]
    anchor = read_csv(inputs / design["bundle"]["path"])
    manifest = json.loads((inputs / design["design_manifest"]["path"]).read_text())
    universe_path = inputs / config["graph_cut"]["panel"]["path"]
    features = validate_universe(anchor, read_csv(universe_path), read_csv(inputs / config["graph_cut"]["columns"]["path"]))
    merged, feature_ids = _merge_control_rows(anchor, universe_path)
    assert feature_ids == features
    alternatives = _load_alt_treatments(project_paths(inputs))
    for row in merged:
        for (treatment, quarter), value in alternatives.items():
            if quarter == row["quarter"]:
                row.setdefault(treatment, value)
    eligible, exclusions = _apply_control_policy(candidate_ids=features, treatment_id=manifest["treatment_id"],
                                                outcome_ids=manifest["outcome_ids"], mode=CONTROL_POLICY_MODE)
    screened = _screen_features(rows=merged, candidate_ids=eligible, treatment_id=manifest["treatment_id"],
                               outcome_ids=manifest["outcome_ids"], min_coverage=0.4)
    if len(screened) < K_SCREENED:
        raise ValueError("Fewer than 100 eligible screened features")
    selected = [r["feature_id"] for r in screened[:K_SCREENED]]
    factor_ids, factor_rows, factors, loadings = _extract_factor_controls(rows=merged, feature_ids=selected,
                                                                        prefix="dflmx_k100", n_factors=FACTOR_COUNT)
    if factor_ids != [f"dflmx_k100_f{i}" for i in range(1, 5)]:
        raise ValueError("Fixed extraction did not return four ordered factors")
    write_csv(outputs / "ordered_screening.csv", screened)
    write_json(outputs / "ordered_eligible_features.json", {"features": eligible, "policy_exclusions": exclusions})
    write_json(outputs / "ordered_top100.json", selected)
    sort_fields = ("screen_score", "coverage_share", "abs_corr_treatment", "abs_corr_outcome_max")
    cutoff_key = tuple(screened[99][key] for key in sort_fields)
    write_json(outputs / "cutoff_ties.json", {"sort_fields": sort_fields, "cutoff_sort_key": cutoff_key,
        "rank100": screened[99], "rank101": screened[100] if len(screened) > 100 else None,
        "same_sort_key_features_in_order": [r["feature_id"] for r in screened if tuple(r[key] for key in sort_fields) == cutoff_key],
        "tie_policy": "accepted stable descending sort preserves incoming feature order"})
    full_scores = [{"quarter": row["quarter"], **{key: row[key] for key in factor_ids}} for row in factor_rows]
    write_csv(outputs / "full_factor_scores.csv", full_scores)
    write_json(outputs / "factor_metadata.json", {"factors": factors, "top_loadings": loadings})
    rows = _merge_by_quarter(factor_rows, [inputs / accepted["designs"][job]["bundle"]["path"] for job in MERGE_JOBS])
    _add_lags(rows, CREDIT_LAG_SOURCES, LAG_PERIODS)
    _add_lags(rows, RATE_RISK_LAG_SOURCES, LAG_PERIODS)
    quarters = tuple(f"{2002 + i // 4}Q{i % 4 + 1}" for i in range(96))
    frozen = [r for r in rows if r["quarter"] in quarters]
    if tuple(r["quarter"] for r in frozen) != quarters:
        raise ValueError("Canonical 96-row sample changed")
    keys = (CANONICAL_TREATMENT_ID, CANONICAL_OUTCOME_ID, *CANONICAL_CONTROL_IDS)
    panel = [{"quarter": r["quarter"], **{key: float(r[key]) for key in keys}} for r in frozen]
    if any(not math.isfinite(r[key]) for r in panel for key in keys):
        raise ValueError("Canonical projection has missing or nonfinite inputs")
    write_csv(outputs / "frozen_projection_panel.csv", panel)
    write_csv(outputs / "sample_factor_scores.csv", [r for r in full_scores if r["quarter"] in quarters])
    stored_headlines = read_csv(inputs / accepted["retained_outputs"]["fixed"]["same_quarter_headline"]["path"])
    stored_headline = [r for r in stored_headlines if r["outcome_id"] == CANONICAL_OUTCOME_ID and r["treatment_id"] == CANONICAL_TREATMENT_ID]
    stored_rolling = [r for r in read_csv(inputs / accepted["retained_outputs"]["fixed"]["rolling_estimates"]["path"]) if r["outcome"] == CANONICAL_OUTCOME_ID and r["horizon"] == "0"]
    if len(stored_headline) != 1 or tuple(r["window_end_quarter"] for r in stored_rolling) != quarters[39:]:
        raise ValueError("Archived comparison inventory differs from the fixed contract")
    comparisons, vectors = [], []
    windows = [("headline", panel, stored_headline[0])]
    for stored in stored_rolling:
        endpoint = stored["window_end_quarter"]
        start = _quarter_ordinal(endpoint) - 47
        nominal_start = f"{start // 4}Q{start % 4 + 1}"
        windows.append((endpoint, [r for r in panel if nominal_start <= r["quarter"] <= endpoint], stored))
    for label, sample, stored in windows:
        estimate = _fit_projection(sample, treatment_id=CANONICAL_TREATMENT_ID, outcome_id=CANONICAL_OUTCOME_ID,
                                   control_ids=CANONICAL_CONTROL_IDS, covariance_lags=1)
        comparisons.append({"window": label, "n": estimate.n, "beta": estimate.beta, "se": estimate.se,
                            "archived_beta": float(stored["beta"]), "archived_se": float(stored["se"]),
                            "beta_difference": estimate.beta - float(stored["beta"]), "se_difference": estimate.se - float(stored["se"])})
        vectors.append({"window": label, "quarters": [r["quarter"] for r in sample],
                        "controls_used": estimate.control_ids_used, "controls_rejected": estimate.control_ids_rejected,
                        "coefficients": estimate.fit.beta, "fitted": estimate.fit.fitted,
                        "residuals": estimate.fit.residuals, "covariance": estimate.fit.covariance})
    write_csv(outputs / "archived_estimate_comparisons.csv", comparisons)
    write_json(outputs / "projection_vectors.json", vectors)
    write_json(outputs / "summary.json", {"origin": ORIGIN, "scientific_status": config["claim_status"],
        "full_rows": len(rows), "frozen_rows": len(panel), "universe_features": len(features),
        "eligible_features": len(eligible), "screened_features": len(screened), "selected_features": len(selected),
        "factors": factor_ids, "headline": comparisons[0], "rolling_windows": len(windows) - 1,
        "max_abs_rolling_beta_difference": max(abs(r["beta_difference"]) for r in comparisons[1:]),
        "max_abs_rolling_se_difference": max(abs(r["se_difference"]) for r in comparisons[1:]),
        "estimate_matching_was_not_a_selection_rule": True, "original_coordinate_identity_established": False,
        "historical_authority_modified": False, "open16_leg_slopes_computed": False})
    write_json(stage / "runtime.json", {"python": sys.version, "executable": sys.executable,
        "executable_sha256": sha256(Path(sys.executable).resolve()), "platform": platform.platform(),
        "machine": platform.machine(), "python_build": platform.python_build(), "compiler": platform.python_compiler(),
        "math_runtime": {"module_origin": math.__spec__.origin,
                         "binary_sha256": sha256(Path(getattr(math, "__file__", sys.executable)).resolve()),
                         "binding": "extension_binary" if hasattr(math, "__file__") else "builtin_interpreter_binary"},
        "packages": sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()),
        "runtime_claim": "current reproduction runtime; not historical runtime authentication"})


def run(root: Path, producer_commit: str, output_dir: str) -> Path:
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from run_open02_producer import (
        _load_accepted_inputs,
        _project_path,
        _verify_producer_commit,
    )

    commit = _verify_producer_commit(root, producer_commit)
    config = json.loads((root / CONFIG).read_text())
    if config.get("origin") != ORIGIN or config.get("schema_version") != "ea_tdc_frozen_factor_reproduction_v1":
        raise ValueError("Only a separately labeled new frozen-input reproduction is supported")
    _load_accepted_inputs(root)  # Verify known accepted design/standardized-series/receipt pins.
    accepted = json.loads((root / config["accepted_receipt"]["path"]).read_text())
    if accepted["producer_commit"] != config["method_commit"]:
        raise ValueError("Accepted method commit differs")
    output = _project_path(root, output_dir)
    if output.exists():
        raise ValueError("Reproduction destination already exists; no overwrite permitted")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".factor-reproduction-", dir=output.parent))
    try:
        expected = [config["accepted_receipt"], *config["graph_cut"].values(),
                    {"path": CONFIG, "sha256": sha256(root / CONFIG)},
                    {"path": "uv.lock", "sha256": sha256(root / "uv.lock")},
                    {"path": "data/bundles/tdcest/standardized_series.csv", "sha256": "1a8d1fabd8b05aa1e223381d2fe0eb2cca89c4f6eb27b608aae7156d247e7223"}]
        for design in accepted["designs"].values():
            expected.extend(design[key] for key in ("bundle", "design_manifest", "sample_manifest"))
        expected.extend(accepted["retained_outputs"]["fixed"][key] for key in ("same_quarter_headline", "rolling_estimates"))
        frozen_records = [copy_pinned(root, stage / "inputs", r) for r in expected]
        raw_records = [record(root, p) for p in sorted((root / "data/seed/interpol/raw").glob("*.csv"))]
        write_json(stage / "input_receipt.json", {"origin": ORIGIN, "producer_commit": commit, "method_commit": config["method_commit"],
            "method_tree": subprocess.check_output(["git", "rev-parse", config["method_commit"] + "^{tree}"], cwd=root, text=True).strip(),
            "frozen_input_graph": frozen_records, "raw_inventory_provenance_only_not_consumed": raw_records,
            "raw_inventory_is_not_historical_authentication": True, "frozen_at_utc": datetime.now(UTC).isoformat()})
        archive = stage / "accepted_method.tar"
        with archive.open("wb") as handle:
            subprocess.run(["git", "archive", config["method_commit"], "src", "scripts"], cwd=root, stdout=handle, check=True)
        method = stage / "method"
        with tarfile.open(archive) as tar:
            tar.extractall(method, filter="data")
        driver = stage / "reproduction_driver.py"
        shutil.copyfile(Path(__file__), driver)
        code = "import runpy,sys; from pathlib import Path; runpy.run_path(sys.argv[1])['compute_snapshot'](Path(sys.argv[2]))"
        command = [sys.executable, "-B", "-c", code, str(driver), str(stage)]
        result = subprocess.run(command, cwd=stage, capture_output=True, text=True, check=False)
        (stage / "execution.log").write_text(result.stdout + result.stderr)
        write_json(stage / "execution.json", {"argv": command, "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(f"Reproduction failed; evidence retained at {stage}")
        for item in frozen_records:
            if sha256(root / item["path"]) != item["sha256"] or sha256(stage / "inputs" / item["path"]) != item["sha256"]:
                raise ValueError("Frozen source changed during reproduction")
        _verify_producer_commit(root, commit)
        shutil.rmtree(method)  # The byte-identical archive retains method source compactly.
        outputs = [record(stage, p) for p in sorted(stage.rglob("*")) if p.is_file() and "inputs" not in p.relative_to(stage).parts]
        write_json(stage / "receipt.json", {"origin": ORIGIN, "producer_commit": commit, "status": "completed_pending_scientific_review",
            "input_receipt_sha256": sha256(stage / "input_receipt.json"), "retained_outputs": outputs,
            "historical_authority_status": "unchanged_unavailable", "original_outputs_unchanged": True,
            "completed_at_utc": datetime.now(UTC).isoformat()})
        os.rename(stage, output)
    except BaseException as exc:
        if stage.exists():
            write_json(stage / "failure.json", {"status": "failed_not_admitted", "error": str(exc), "producer_commit": commit})
        raise
    return output / "receipt.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(run(ROOT, args.producer_commit, args.output_dir))


if __name__ == "__main__":
    main()
