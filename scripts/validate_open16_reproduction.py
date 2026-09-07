"""Validate the retained fresh reproduction for two bounded OPEN-16 diagnostics."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from run_frozen_factor_reproduction import (
    copy_pinned,
    read_csv,
    record,
    sha256,
    write_csv,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-7
GATES = ("raw_semantic_equality", "screen_top100", "rank_partition", "projector_equivalence", "economics_equivalence")
CONFIG = "config/open16_reproduction_authority.json"
PRODUCTION_SOURCES = (
    "src/ea_tdc/estimation.py", "src/ea_tdc/open01.py", "src/ea_tdc/open16.py",
    "src/ea_tdc/open_contract.py", "scripts/run_open02_producer.py",
    "scripts/run_open16_diagnostics.py", "scripts/validate_open16_reproduction.py",
    "scripts/run_frozen_factor_reproduction.py",
)


def production_manifest(root: Path) -> list[dict]:
    return [record(root, root / name) for name in PRODUCTION_SOURCES]


def verify_production_manifest(root: Path, supplied: object) -> None:
    if supplied != production_manifest(root):
        raise ValueError("Production source differs from validated manifest")


def verify_accepted_estimator(root: Path, archive_path: Path) -> None:
    with tarfile.open(archive_path) as archive:
        for name in ("src/ea_tdc/estimation.py", "src/ea_tdc/open01.py"):
            member = archive.extractfile(name)
            if member is None or member.read() != (root / name).read_bytes():
                raise ValueError("Current estimator differs from accepted archived method")



def verify_package(package: Path, expected_sha256: str) -> dict:
    if sha256(package / "receipt.json") != expected_sha256:
        raise ValueError("Reproduction receipt differs from pinned authority")
    receipt = json.loads((package / "receipt.json").read_text())
    if receipt["origin"] != "new_frozen_input_reproduction":
        raise ValueError("Expected a fresh reproduction, not historical-coordinate authority")
    inputs = json.loads((package / "input_receipt.json").read_text())
    if sha256(package / "input_receipt.json") != receipt["input_receipt_sha256"]:
        raise ValueError("Reproduction input receipt changed")
    for parent, records in ((package, receipt["retained_outputs"]), (package / "inputs", inputs["frozen_input_graph"])):
        for item in records:
            path = parent / item["path"]
            path.resolve().relative_to(parent.resolve())
            if sha256(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
                raise ValueError(f"Changed reproduction object: {item['path']}")
    return inputs


def compare_semantic_universes(expected: list[dict], actual: list[dict]) -> dict:
    if [r["quarter"] for r in expected] != [r["quarter"] for r in actual] or set(expected[0]) != set(actual[0]):
        raise ValueError("Rebuilt universe changed named row/feature inventory")
    differences = sum(a[key] != b[key] for a, b in zip(expected, actual, strict=True) for key in a)
    if differences:
        raise ValueError(f"Rebuilt universe changed {differences} named cells")
    return {"rows": len(expected), "features": len(expected[0]) - 1, "cell_differences": differences,
            "serialized_column_order_equal": list(expected[0]) == list(actual[0])}


def raw_worker(stage: Path, package: Path) -> None:
    sys.path[:0] = [str(stage / "method/src"), str(stage / "method/scripts")]
    from ea_tdc.paths import project_paths
    from ea_tdc.robustness import (
        _apply_control_policy,
        _load_alt_treatments,
        _merge_control_rows,
        _screen_features,
        build_control_universe,
    )

    inputs = package / "inputs"
    config = json.loads((inputs / "config/frozen_factor_reproduction.json").read_text())
    accepted = json.loads((inputs / config["accepted_receipt"]["path"]).read_text())
    design = accepted["designs"][accepted["contract"]["design_job_ids"][0]]
    anchor = read_csv(inputs / design["bundle"]["path"])
    manifest = json.loads((inputs / design["design_manifest"]["path"]).read_text())
    original_path = inputs / config["graph_cut"]["panel"]["path"]
    original = read_csv(original_path)
    rebuilt = build_control_universe(project_paths(stage / "raw_rebuild"), quarter_grid=[r["quarter"] for r in anchor])
    semantic = compare_semantic_universes(original, read_csv(rebuilt.panel_path))
    expected_columns = read_csv(inputs / config["graph_cut"]["columns"]["path"])
    rebuilt_columns = read_csv(rebuilt.columns_path)
    july_sources = list(dict.fromkeys(r["source_file"] for r in expected_columns))
    native_sources = list(dict.fromkeys(r["source_file"] for r in rebuilt_columns))
    semantic.update({"retained_order_matches_casefolded_filenames": july_sources == sorted(july_sources, key=str.casefold),
                     "rebuilt_order_matches_native_filenames": native_sources == sorted(native_sources),
                     "retained_source_order": july_sources, "rebuilt_source_order": native_sources,
                     "ordering_interpretation": "Current case-folded versus case-sensitive ordering explains serialization differences; not historical runtime proof"})
    write_json(stage / "universe_comparison.json", semantic)
    # Preserve the retained July order when applying the fixed screen.
    merged, features = _merge_control_rows(anchor, original_path)
    if features != [r["feature_id"] for r in expected_columns]:
        raise ValueError("Retained column metadata/header order differs")
    alternatives = _load_alt_treatments(project_paths(inputs))
    for row in merged:
        for (series, quarter), value in alternatives.items():
            if quarter == row["quarter"]:
                row.setdefault(series, value)
    eligible, _ = _apply_control_policy(candidate_ids=features, treatment_id=manifest["treatment_id"], outcome_ids=manifest["outcome_ids"], mode="balanced")
    screen = _screen_features(rows=merged, candidate_ids=eligible, treatment_id=manifest["treatment_id"], outcome_ids=manifest["outcome_ids"], min_coverage=0.4)
    prior = read_csv(package / "results/ordered_screening.csv")
    selected = [r["feature_id"] for r in screen[:100]]
    if len(screen) != 6436 or [{k: str(v) for k, v in r.items()} for r in screen] != prior:
        raise ValueError("Fixed ordered screen differs from the retained 6,436-feature screen")
    if selected != json.loads((package / "results/ordered_top100.json").read_text()):
        raise ValueError("Recovered ordered top 100 changed")
    separation = screen[99]["screen_score"] - screen[100]["screen_score"]
    if separation <= 0:
        raise ValueError("K=100 cutoff is not strictly separated")
    write_csv(stage / "rebuilt_ordered_screen.csv", screen)
    write_json(stage / "screen_comparison.json", {"screened_features": len(screen), "ordered_top100_equal": True,
        "cutoff_separation": separation, "factor_extractions_performed": 0})


def runtime_identity() -> dict:
    import numpy as np
    import numpy._core._multiarray_umath as core
    return {"python": sys.version, "executable_sha256": sha256(Path(sys.executable).resolve()),
            "numpy_version": np.__version__, "numpy_core_sha256": sha256(Path(core.__file__))}


def rank_projector(matrix: list[list[float]]) -> tuple[int, list[list[float]]]:
    import numpy as np
    x = np.asarray(matrix, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite conditioning matrix")
    scales = np.linalg.norm(x, axis=0)
    normalized = x / np.where(scales > 0, scales, 1)
    rank = int(np.linalg.matrix_rank(normalized))
    if rank != x.shape[1]:
        raise ValueError("Conditioning matrix is not full rank")
    q, _ = np.linalg.qr(normalized, mode="reduced")
    return rank, (q @ q.T).tolist()


def numeric_worker(stage: Path, package: Path, destination: Path) -> None:
    sys.path[:0] = [str(stage / "method/src"), str(stage / "method/scripts")]
    from ea_tdc.open01 import _fit_projection, _quarter_ordinal
    from ea_tdc.open_contract import (
        CANONICAL_CONTROL_IDS,
        CANONICAL_OUTCOME_ID,
        CANONICAL_TREATMENT_ID,
    )

    panel = read_csv(package / "results/frozen_projection_panel.csv")
    factors = [f"dflmx_k100_f{i}" for i in range(1, 5)]
    full = read_csv(package / "results/full_factor_scores.csv")
    factor_rank, _ = rank_projector([[float(r[k]) for k in factors] for r in full])
    sample_factor_rank, _ = rank_projector([[float(r[k]) for k in factors] for r in panel])
    if (factor_rank, sample_factor_rank) != (4, 4):
        raise ValueError("Four-factor rank changed")
    reference = json.loads((package / "results/projection_vectors.json").read_text())
    comparisons = read_csv(package / "results/archived_estimate_comparisons.csv")
    quarters = tuple(f"{2002 + i // 4}Q{i % 4 + 1}" for i in range(96))
    labels = ["headline", *quarters[39:]]
    if tuple(r["quarter"] for r in panel) != quarters or [r["window"] for r in reference] != labels:
        raise ValueError("Fixed panel or 57-window reference inventory differs")
    windows = []
    reference_maxima = dict.fromkeys(("coefficients", "fitted", "residuals", "covariance", "archived_beta_se"), 0.0)
    for label, retained, comparison in zip(labels, reference, comparisons, strict=True):
        start = _quarter_ordinal(label) - 47 if label != "headline" else None
        observed = [r for r in panel if start is None or start <= _quarter_ordinal(r["quarter"]) <= _quarter_ordinal(label)]
        fit = _fit_projection(observed, treatment_id=CANONICAL_TREATMENT_ID, outcome_id=CANONICAL_OUTCOME_ID, control_ids=CANONICAL_CONTROL_IDS, covariance_lags=1)
        if ([r["quarter"] for r in observed] != retained["quarters"]
                or list(fit.control_ids_used) != retained["controls_used"] or list(fit.control_ids_rejected) != retained["controls_rejected"]):
            raise ValueError("Rank-aware control partition or row inventory changed")
        z = [[1.0, *[float(r[k]) for k in fit.control_ids_used]] for r in observed]
        rank, projector = rank_projector(z)
        rank_projector([[*row, float(r[CANONICAL_TREATMENT_ID])] for row, r in zip(z, observed, strict=True)])
        current = {"window": label, "quarters": retained["quarters"], "controls_used": list(fit.control_ids_used),
                   "controls_rejected": list(fit.control_ids_rejected), "conditioning_rank": rank, "projector": projector,
                   "coefficients": fit.fit.beta, "fitted": fit.fit.fitted, "residuals": fit.fit.residuals,
                   "covariance": fit.fit.covariance, "beta_se": [fit.beta, fit.se]}
        for key in ("coefficients", "fitted", "residuals", "covariance"):
            reference_maxima[key] = max(reference_maxima[key], max_difference(current[key], retained[key]))
        reference_maxima["archived_beta_se"] = max(reference_maxima["archived_beta_se"], max_difference(current["beta_se"], [float(comparison["archived_beta"]), float(comparison["archived_se"])]))
        windows.append(current)
    write_json(destination, {"runtime": runtime_identity(), "factor_rank": factor_rank,
        "sample_factor_rank": sample_factor_rank, "windows": windows, "max_abs_reference_differences": reference_maxima,
        "reference_scope": "780d new reproduction vectors; archived OPEN01 supplies beta/SE only"})


def max_difference(left: object, right: object) -> float:
    import numpy as np
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Numerical comparison shape or finiteness failure")
    gap = float(np.max(np.abs(a - b)))
    if gap > TOLERANCE:
        raise ValueError(f"Numerical equivalence failed: {gap} > {TOLERANCE}")
    return gap


def compare_environments(first: dict, second: dict) -> dict:
    if (len(first.get("windows", [])) != 58 or len(second.get("windows", [])) != 58
            or any(r.get(k) != 4 for r in (first, second) for k in ("factor_rank", "sample_factor_rank"))):
        raise ValueError("Comparison requires four-factor ranks and all 58 reference fits")
    if first["runtime"] == second["runtime"]:
        raise ValueError("Two distinct pinned numerical environments are required")
    maxima = {k: 0.0 for k in ("projector", "coefficients", "fitted", "residuals", "covariance", "beta_se")}
    for a, b in zip(first["windows"], second["windows"], strict=True):
        for key in ("window", "quarters", "controls_used", "controls_rejected", "conditioning_rank"):
            if a[key] != b[key]:
                raise ValueError(f"Cross-environment partition/rank mismatch: {key}")
        for key, previous in maxima.items():
            maxima[key] = max(previous, max_difference(a[key], b[key]))
    return maxima


def run(root: Path, commit: str, other_python: str, output_dir: str) -> Path:
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from run_open02_producer import _project_path, _verify_producer_commit

    _verify_producer_commit(root, commit)
    config = json.loads((root / CONFIG).read_text())
    package = _project_path(root, config["reproduction_receipt"]["path"]).parent
    inputs = verify_package(package, config["reproduction_receipt"]["sha256"])
    verify_accepted_estimator(root, package / "accepted_method.tar")
    source_manifest = production_manifest(root)
    output = _project_path(root, output_dir)
    if output.exists():
        raise ValueError("Validation destination exists; no overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".open16-validation-", dir=output.parent))
    try:
        raw = inputs["raw_inventory_provenance_only_not_consumed"]
        current = [record(root, p) for p in sorted((root / "data/seed/interpol/raw").glob("*.csv"))]
        if current != raw or len(raw) != 395:
            raise ValueError("Current raw inventory differs from the frozen reproduction inventory")
        for item in raw:
            copy_pinned(root, stage / "raw_rebuild", item)
        with tarfile.open(package / "accepted_method.tar") as archive:
            archive.extractall(stage / "method", filter="data")
        driver = stage / "validation_driver.py"
        shutil.copyfile(Path(__file__), driver)
        shutil.copyfile(root / "scripts/run_frozen_factor_reproduction.py", stage / "run_frozen_factor_reproduction.py")
        commands = []
        for name, python, function, extra in (("raw", sys.executable, "raw_worker", []),
                ("primary", sys.executable, "numeric_worker", [str(stage / "primary.json")]),
                ("comparison", other_python, "numeric_worker", [str(stage / "comparison.json")])):
            code = "import runpy,sys; from pathlib import Path; runpy.run_path(sys.argv[1])[sys.argv[2]](*map(Path,sys.argv[3:]))"
            command = [python, "-B", "-c", code, str(driver), function, str(stage), str(package), *extra]
            result = subprocess.run(command, cwd=stage, capture_output=True, text=True, check=False)
            (stage / f"{name}.log").write_text(result.stdout + result.stderr)
            commands.append({"name": name, "argv": command, "exit_code": result.returncode})
            write_json(stage / "commands.json", commands)
            if result.returncode:
                raise RuntimeError(f"{name} validation failed; see {stage / (name + '.log')}")
        first, second = [json.loads((stage / f"{name}.json").read_text()) for name in ("primary", "comparison")]
        if [first["runtime"], second["runtime"]] != config["environments"]:
            raise ValueError("Numerical environments differ from committed pins")
        maxima = compare_environments(first, second)
        write_json(stage / "numerical_comparison.json", {"tolerance": TOLERANCE, "max_abs_differences": maxima})
        # Freeze the raw bytes compactly after the sole accepted-helper rebuild.
        with tarfile.open(stage / "raw_snapshot.tar.gz", "w:gz") as archive:
            for item in raw:
                archive.add(stage / "raw_rebuild" / item["path"], arcname=item["path"])
        shutil.rmtree(stage / "raw_rebuild/data")
        shutil.rmtree(stage / "method")
        verify_package(package, config["reproduction_receipt"]["sha256"])
        _verify_producer_commit(root, commit)
        outputs = [record(stage, p) for p in sorted(stage.rglob("*")) if p.is_file()]
        write_json(stage / "receipt.json", {"authority_class": "fresh_frozen_conditioning_space", "status": "passed",
            "producer_commit": commit, "production_source_manifest": source_manifest, "reproduction_receipt_sha256": config["reproduction_receipt"]["sha256"],
            "gates": dict.fromkeys(GATES, True), "tolerance": TOLERANCE, "environments": config["environments"],
            "raw_factor_extractions_performed": 0, "outputs": outputs,
            "scope": config["scope"], "historical_coordinate_authenticity": "unavailable"})
        os.rename(stage, output)
    except BaseException as exc:
        write_json(stage / "failure.json", {"status": "failed_no_diagnostic_admission", "error": str(exc), "producer_commit": commit})
        raise
    return output / "receipt.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--comparison-python", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(run(ROOT, args.producer_commit, args.comparison_python, args.output_dir))
