"""Rebuild and execute the fail-closed OPEN-01 acceptance package.

This entry point is intentionally the only production path for the focused
contract, headline, stability, and credit-screen tables. The rebuild is a
remote/heavy lane; fixture tests exercise the same package implementation
without touching live inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from ea_tdc.adapters.tdcest import adapt_tdcest  # noqa: E402
from ea_tdc.open01 import (  # noqa: E402
    build_open01_acceptance,
    build_treatment_outcome_contract,
    write_open01_outputs,
)
from ea_tdc.open_contract import (  # noqa: E402
    OPEN01_DESIGN_JOB_IDS,
)
from ea_tdc.paths import project_paths  # noqa: E402
from run_pinned_factor_residual_bridge import MERGE_JOBS  # noqa: E402
from run_submission_appendix_diagnostics import _build_inputs  # noqa: E402


ACCEPTANCE_MANIFEST = (
    ROOT / "output/manifests/open01_acceptance_summary.json"
)
DEFAULT_TDCEST_BUNDLE = ROOT.parent / "tdcest/site/data/bundle.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _design_bundle_hashes(root: Path) -> dict[str, str]:
    bundle_dir = root / "data/bundles/designs"
    hashes: dict[str, str] = {}
    for job_id in OPEN01_DESIGN_JOB_IDS:
        path = bundle_dir / f"{job_id}__quarterly_bundle.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"OPEN-01 design bundle was not rebuilt: {path}"
            )
        hashes[job_id] = _sha256_file(path)
    return hashes


def run(*, tdcest_bundle: Path = DEFAULT_TDCEST_BUNDLE) -> dict[str, Any]:
    if tuple(MERGE_JOBS) != tuple(OPEN01_DESIGN_JOB_IDS):
        raise ValueError(
            "The live four-bundle merge path drifted from OPEN01_DESIGN_JOB_IDS"
        )

    paths = project_paths(ROOT)
    if not tdcest_bundle.is_file():
        raise FileNotFoundError(
            f"TDCest seed bundle does not exist: {tdcest_bundle}"
        )
    adapter_result = adapt_tdcest(
        paths,
        bundle_path=str(tdcest_bundle.resolve()),
    )
    adapter_manifest = _read_json(adapter_result.manifest_path)

    _built_paths, rows, *_unused = _build_inputs()
    design_hashes = _design_bundle_hashes(ROOT)
    contract_rows = build_treatment_outcome_contract(
        adapter_manifest=adapter_manifest,
        design_bundle_hashes=design_hashes,
    )
    result = build_open01_acceptance(
        rows,
        contract_rows=contract_rows,
    )
    producer_inputs = {
        "tdcest": {
            "seed_bundle_sha256": adapter_manifest.get("bundle_hash", ""),
            "combined_input_sha256": adapter_manifest.get(
                "combined_input_hash", ""
            ),
            "component_input_hashes": adapter_manifest.get(
                "input_hashes", {}
            ),
            "rows_written": adapter_manifest.get("rows_written", 0),
        },
        "design_bundles": design_hashes,
    }
    write_open01_outputs(
        result,
        root=ROOT,
        producer_inputs=producer_inputs,
    )
    manifest = _read_json(ACCEPTANCE_MANIFEST)
    checks = manifest.get("acceptance_checks", {})
    if (
        manifest.get("status") != "passed"
        or not isinstance(checks, dict)
        or not checks
        or not all(
            isinstance(check, dict) and check.get("passed") is True
            for check in checks.values()
        )
    ):
        failed = [
            check_id
            for check_id, check in checks.items()
            if not isinstance(check, dict) or check.get("passed") is not True
        ]
        raise RuntimeError(
            "OPEN-01 acceptance failed closed; failed checks: "
            + ",".join(failed)
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild and execute the OPEN-01 acceptance package."
    )
    parser.add_argument(
        "--tdcest-bundle",
        type=Path,
        default=DEFAULT_TDCEST_BUNDLE,
        help=(
            "TDCest bundle.json input; defaults to the sibling producer's "
            "published site bundle."
        ),
    )
    args = parser.parse_args()
    manifest = run(tdcest_bundle=args.tdcest_bundle)
    print(
        "OPEN-01 acceptance passed: "
        f"{len(manifest['acceptance_checks'])} checks; "
        f"scientific_status={manifest['scientific_status']}"
    )


if __name__ == "__main__":
    main()
