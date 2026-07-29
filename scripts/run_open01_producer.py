"""Run the complete OPEN-01 producer chain and write its passing receipt.

This runner is intended to be invoked through the project's remote execution
lane. It does not recover from partial work: every subprocess, evidence file,
hash, and acceptance check must pass before a receipt is written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ea_tdc.open_contract import (  # noqa: E402
    CANONICAL_CONTROL_IDS,
    CANONICAL_OUTCOME_ID,
    CANONICAL_RESIDUAL_ID,
    CANONICAL_TREATMENT_ID,
    CANONICAL_TREATMENT_SOURCE_SERIES,
    CREDIT_SCREEN_OUTCOME_IDS,
    OPEN01_DESIGN_JOB_IDS,
    OPEN_CONTRACT,
    ROLLING_WINDOW_QUARTERS,
)
from ea_tdc.open01 import (  # noqa: E402
    CREDIT_ADJUSTMENTS,
    CREDIT_CALIBRATION_METHODS,
    CREDIT_WINDOW_QUARTERS,
)


RECEIPT_LOCATOR = "output/manifests/open01_producer_run_receipt.json"
SOURCE_MANIFEST_LOCATOR = "output/manifests/tdcest_source_manifest.json"
STANDARDIZED_LOCATOR = "data/bundles/tdcest/standardized_series.csv"
ACCEPTANCE_MANIFEST_LOCATOR = "output/manifests/open01_acceptance_summary.json"
SUBMISSION_MANIFEST_LOCATOR = (
    "output/manifests/submission_appendix_diagnostics_summary.json"
)
ROLLING_MANIFEST_LOCATOR = (
    "output/manifests/"
    "tier2_rolling_selected_credit_rate_pass_through_summary.json"
)
ROLLING_ESTIMATES_LOCATOR = (
    "output/models/tier2_rolling_selected_credit_rate_pass_through_estimates.csv"
)
OFFSET_MANIFEST_LOCATOR = (
    "output/manifests/tier2_pass_through_offset_diagnostics_summary.json"
)
FORMAL_CREDIT_SCREEN_LOCATOR = (
    "output/reports/tier2_pass_through_offset_rolling_beta_correlates.csv"
)
PERSISTENCE_MANIFEST_LOCATOR = (
    "output/manifests/tier2_pass_through_regime_persistence_summary.json"
)
REGIME_VALIDATION_MANIFEST_LOCATOR = (
    "outputs/manifests/ea_tdc_pass_through_regime_validation_summary.json"
)

RETAINED_OUTPUT_LOCATORS: Mapping[str, str] = {
    "treatment_outcome_contract": "outputs/tables/tdc_treatment_outcome_contract.csv",
    "same_quarter_headline": "outputs/tables/tdc_same_quarter_headline.csv",
    "rolling_estimates": ROLLING_ESTIMATES_LOCATOR,
    "rolling_summary": ROLLING_MANIFEST_LOCATOR,
    "formal_offset_correlates_credit_screen": FORMAL_CREDIT_SCREEN_LOCATOR,
    "offset_summary": OFFSET_MANIFEST_LOCATOR,
    "regime_persistence_summary": PERSISTENCE_MANIFEST_LOCATOR,
    "regime_validation_summary": REGIME_VALIDATION_MANIFEST_LOCATOR,
    "stability_gate": "output/reports/tier2_pass_through_stability_gate.csv",
    "submission_appendix_summary": SUBMISSION_MANIFEST_LOCATOR,
}

STAGE_SUMMARY_MANIFESTS: Mapping[str, tuple[str, ...]] = {
    "adapt_tdcest": (SOURCE_MANIFEST_LOCATOR,),
    "submission_appendix_factor_preparation": (SUBMISSION_MANIFEST_LOCATOR,),
    "rolling": (ROLLING_MANIFEST_LOCATOR,),
    "offset": (OFFSET_MANIFEST_LOCATOR,),
    "regime_persistence": (PERSISTENCE_MANIFEST_LOCATOR,),
    "regime_validation": (REGIME_VALIDATION_MANIFEST_LOCATOR,),
}

SUBMISSION_OUTPUT_FIELDS = (
    "lead_placebo_csv",
    "hac_csv",
    "factor_tail_csv",
    "splice_csv",
    "plumbing_csv",
)

PIPELINE_SCRIPT_LOCATORS = (
    "scripts/run_submission_appendix_diagnostics.py",
    "scripts/run_tier2_rolling_pass_through.py",
    "scripts/run_tier2_pass_through_offset_diagnostics.py",
    "scripts/run_tier2_pass_through_regime_persistence.py",
    "scripts/run_tier2_pass_through_regime_validation.py",
    "scripts/run_open01_acceptance.py",
)

RunCommand = Callable[[list[str], Path], subprocess.CompletedProcess[Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}")
    return payload


def _project_path(root: Path, locator: str) -> Path:
    candidate = Path(locator)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"OPEN-01 artifact escapes the project root: {locator}") from exc
    return resolved


def _relative_locator(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _file_record(
    root: Path,
    locator: str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    path = _project_path(root, locator)
    if not path.is_file():
        raise FileNotFoundError(f"Missing required OPEN-01 artifact: {locator}")
    actual_sha256 = _sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"Hash mismatch for {locator}: expected {expected_sha256}, got {actual_sha256}"
        )
    return {
        "path": _relative_locator(root, path),
        "sha256": actual_sha256,
        "bytes": path.stat().st_size,
    }


def _validate_sha256(value: Any, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError(f"{label} is not a full SHA-256 digest")
    return text


def _resolve_tdcest_bundle(root: Path, locator: str) -> Path:
    candidate = Path(locator).expanduser()
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing required TDCest bundle: {path}")
    return path


def _git_text(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _verify_producer_commit(root: Path, producer_commit: str) -> str:
    supplied = producer_commit.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", supplied):
        raise ValueError("--producer-commit must be a full 40-character Git commit")
    head = _git_text(root, "rev-parse", "HEAD").lower()
    if supplied != head:
        raise ValueError(
            f"Producer commit {supplied} does not equal checkout HEAD {head}"
        )
    checkout_changes = _git_text(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    if checkout_changes:
        raise ValueError(
            "Checkout has tracked or untracked nonignored files; commit the "
            "producer state before running OPEN-01"
        )
    return head


def _validate_run_id(run_id: str) -> str:
    value = run_id.strip()
    if not value or len(value) > 160:
        raise ValueError("--run-id must contain between 1 and 160 characters")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", value):
        raise ValueError(
            "--run-id may contain only letters, digits, period, underscore, colon, and hyphen"
        )
    return value


def _pipeline_commands(
    *,
    python_executable: str,
    tdcest_bundle: str,
) -> list[tuple[str, list[str]]]:
    python = str(python_executable)
    commands: list[tuple[str, list[str]]] = [
        (
            "adapt_tdcest",
            [
                python,
                "-B",
                "-m",
                "ea_tdc",
                "adapt-tdcest",
                "--repo-root",
                ".",
                "--bundle-path",
                tdcest_bundle,
            ],
        )
    ]
    commands.extend(
        (
            f"design:{job_id}",
            [
                python,
                "-B",
                "-m",
                "ea_tdc",
                "build-quarterly-design",
                job_id,
                "--repo-root",
                ".",
            ],
        )
        for job_id in OPEN01_DESIGN_JOB_IDS
    )
    commands.extend(
        (
            stage,
            [python, "-B", script],
        )
        for stage, script in (
            (
                "submission_appendix_factor_preparation",
                "scripts/run_submission_appendix_diagnostics.py",
            ),
            ("rolling", "scripts/run_tier2_rolling_pass_through.py"),
            ("offset", "scripts/run_tier2_pass_through_offset_diagnostics.py"),
            ("regime_persistence", "scripts/run_tier2_pass_through_regime_persistence.py"),
            ("regime_validation", "scripts/run_tier2_pass_through_regime_validation.py"),
        )
    )
    commands.append(
        (
            "acceptance",
            [
                python,
                "-B",
                "scripts/run_open01_acceptance.py",
                "--tdcest-bundle",
                tdcest_bundle,
            ],
        )
    )
    return commands


def _default_run_command(
    argv: list[str],
    cwd: Path,
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(argv, cwd=cwd, check=True)


def _design_manifest_locators(job_id: str) -> tuple[str, str]:
    return (
        f"output/manifests/{job_id}__design_manifest.json",
        f"output/manifests/{job_id}__sample_manifest.json",
    )


def _stage_manifest_locators(stage: str) -> tuple[str, ...]:
    if stage.startswith("design:"):
        job_id = stage.removeprefix("design:")
        if job_id not in OPEN01_DESIGN_JOB_IDS:
            raise ValueError(f"Unknown OPEN-01 design stage: {stage}")
        return _design_manifest_locators(job_id)
    if stage == "acceptance":
        final_inputs = [
            SOURCE_MANIFEST_LOCATOR,
            *(
                locator
                for job_id in OPEN01_DESIGN_JOB_IDS
                for locator in _design_manifest_locators(job_id)
            ),
            ACCEPTANCE_MANIFEST_LOCATOR,
        ]
        return tuple(final_inputs)
    try:
        return STAGE_SUMMARY_MANIFESTS[stage]
    except KeyError as exc:
        raise ValueError(f"OPEN-01 stage has no freshness contract: {stage}") from exc


def _remove_stage_manifests(
    *,
    root: Path,
    stage: str,
) -> dict[str, dict[str, Any] | None]:
    previous: dict[str, dict[str, Any] | None] = {}
    for locator in _stage_manifest_locators(stage):
        path = _project_path(root, locator)
        if path.exists() and not path.is_file():
            raise ValueError(f"Producer manifest locator is not a file: {locator}")
        previous[locator] = _file_record(root, locator) if path.is_file() else None
        if path.is_file():
            path.unlink()
    return previous


def _require_recreated_stage_manifests(
    *,
    root: Path,
    stage: str,
    previous: Mapping[str, dict[str, Any] | None],
) -> dict[str, dict[str, Any]]:
    recreated: dict[str, dict[str, Any]] = {}
    for locator in _stage_manifest_locators(stage):
        record = _file_record(root, locator)
        record["previous_sha256"] = (
            previous[locator]["sha256"] if previous[locator] is not None else None
        )
        record["recreated_after_pre_removal"] = True
        recreated[locator] = record
    return recreated


def _run_commands(
    commands: list[tuple[str, list[str]]],
    *,
    root: Path,
    run_command: RunCommand,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for stage, argv in commands:
        previous_manifests = _remove_stage_manifests(root=root, stage=stage)
        started_at = _utc_now()
        completed = run_command(argv, root)
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, argv)
        refreshed_manifests = _require_recreated_stage_manifests(
            root=root,
            stage=stage,
            previous=previous_manifests,
        )
        records.append(
            {
                "stage": stage,
                "argv": argv,
                "started_at_utc": started_at,
                "completed_at_utc": _utc_now(),
                "returncode": completed.returncode,
                "refreshed_manifests": refreshed_manifests,
            }
        )
    return records


def _acceptance_check_passed(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if not isinstance(value, dict) or not value:
        return False
    recognized = False
    passed = True
    if "passed" in value:
        recognized = True
        passed = passed and value["passed"] is True
    if "status" in value:
        recognized = True
        status = str(value["status"]).strip().lower()
        passed = passed and status in {"pass", "passed"}
    return recognized and passed


def _acceptance_output_locator(key: str, value: Any) -> tuple[str, str | None]:
    if isinstance(value, str):
        locator = value.strip()
        expected_sha256 = None
    elif isinstance(value, dict):
        locator = str(
            value.get("path")
            or value.get("locator")
            or value.get("file")
            or ""
        ).strip()
        expected_raw = value.get("sha256")
        expected_sha256 = (
            _validate_sha256(expected_raw, label=f"acceptance output {key} sha256")
            if expected_raw is not None
            else None
        )
    else:
        raise TypeError(f"Unknown acceptance output record for {key!r}")
    if not locator:
        raise ValueError(f"Acceptance output {key!r} has no locator")
    return locator, expected_sha256


def _validate_acceptance(
    root: Path,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if payload.get("status") != "passed":
        raise ValueError("OPEN-01 acceptance status is not 'passed'")
    checks = payload.get("acceptance_checks")
    if not isinstance(checks, dict) or not checks:
        raise ValueError("OPEN-01 acceptance_checks must be a non-empty mapping")
    failed = [
        str(check_id)
        for check_id, value in checks.items()
        if not _acceptance_check_passed(value)
    ]
    if failed:
        raise ValueError(
            "OPEN-01 acceptance has non-passing or unknown checks: "
            + ", ".join(sorted(failed))
        )
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("OPEN-01 acceptance outputs must be a non-empty mapping")
    output_records: dict[str, dict[str, Any]] = {}
    for key, value in outputs.items():
        locator, expected_sha256 = _acceptance_output_locator(str(key), value)
        output_records[str(key)] = _file_record(
            root,
            locator,
            expected_sha256=expected_sha256,
        )
    return checks, output_records


def _output_declarations(
    value: Any,
    *,
    declaration: str,
) -> list[dict[str, str | None]]:
    if isinstance(value, str):
        locator = value.strip()
        if not locator:
            raise ValueError(f"Empty output locator at {declaration}")
        return [
            {
                "declaration": declaration,
                "locator": locator,
                "expected_sha256": None,
            }
        ]
    if isinstance(value, list):
        if not value:
            raise ValueError(f"Empty output locator list at {declaration}")
        output: list[dict[str, str | None]] = []
        for index, item in enumerate(value):
            output.extend(
                _output_declarations(
                    item,
                    declaration=f"{declaration}[{index}]",
                )
            )
        return output
    if not isinstance(value, dict) or not value:
        raise TypeError(f"Unknown output declaration at {declaration}")

    locator_keys = [
        key
        for key in ("path", "locator", "file")
        if str(value.get(key, "")).strip()
    ]
    if locator_keys:
        locators = {str(value[key]).strip() for key in locator_keys}
        if len(locators) != 1:
            raise ValueError(f"Conflicting output locators at {declaration}")
        expected_raw = value.get("sha256")
        return [
            {
                "declaration": declaration,
                "locator": locators.pop(),
                "expected_sha256": (
                    _validate_sha256(
                        expected_raw,
                        label=f"{declaration} sha256",
                    )
                    if expected_raw is not None
                    else None
                ),
            }
        ]

    output = []
    for key, item in value.items():
        output.extend(
            _output_declarations(
                item,
                declaration=f"{declaration}.{key}",
            )
        )
    return output


def _manifest_output_declarations(
    *,
    manifest_name: str,
    payload: Mapping[str, Any],
) -> list[dict[str, str | None]]:
    declarations: list[dict[str, str | None]] = []
    outputs = payload.get("outputs")
    if outputs is not None:
        if not isinstance(outputs, dict) or not outputs:
            raise ValueError(f"{manifest_name} outputs must be a non-empty mapping")
        by_output_key: dict[str, list[dict[str, str | None]]] = {}
        for key, value in outputs.items():
            records = _output_declarations(
                value,
                declaration=f"{manifest_name}.outputs.{key}",
            )
            by_output_key[str(key)] = records
            declarations.extend(records)

        parallel_hashes = payload.get("output_sha256")
        if parallel_hashes is not None:
            if not isinstance(parallel_hashes, dict):
                raise TypeError(f"{manifest_name} output_sha256 must be a mapping")
            if set(parallel_hashes) != set(outputs):
                raise ValueError(
                    f"{manifest_name} output_sha256 keys do not match outputs"
                )
            for key, expected_raw in parallel_hashes.items():
                records = by_output_key[str(key)]
                if len(records) != 1:
                    raise ValueError(
                        f"{manifest_name}.outputs.{key} must resolve to one locator "
                        "when output_sha256 is declared"
                    )
                expected = _validate_sha256(
                    expected_raw,
                    label=f"{manifest_name}.output_sha256.{key}",
                )
                in_record = records[0]["expected_sha256"]
                if in_record is not None and in_record != expected:
                    raise ValueError(
                        f"Conflicting declared hashes for "
                        f"{manifest_name}.outputs.{key}"
                    )
                records[0]["expected_sha256"] = expected

    if manifest_name == "submission_appendix":
        for field in SUBMISSION_OUTPUT_FIELDS:
            if field not in payload:
                raise ValueError(
                    f"Submission appendix manifest omits required {field}"
                )
            declarations.extend(
                _output_declarations(
                    payload[field],
                    declaration=f"{manifest_name}.{field}",
                )
            )
    if not declarations:
        raise ValueError(f"{manifest_name} declares no retained outputs")
    return declarations


def _collect_declared_outputs(
    *,
    root: Path,
    manifests: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    retained: dict[str, dict[str, Any]] = {}
    for manifest_name, manifest_locator in manifests.items():
        payload = _load_json_object(_project_path(root, manifest_locator))
        declarations = _manifest_output_declarations(
            manifest_name=manifest_name,
            payload=payload,
        )
        for declaration in declarations:
            locator = str(declaration["locator"])
            expected = declaration["expected_sha256"]
            record = _file_record(
                root,
                locator,
                expected_sha256=expected,
            )
            normalized = record["path"]
            existing = retained.get(normalized)
            if existing is None:
                retained[normalized] = {
                    **record,
                    "declared_by": [declaration["declaration"]],
                    "declared_sha256": expected,
                }
                continue
            if existing["sha256"] != record["sha256"]:
                raise ValueError(
                    f"Duplicate output locator changed while collecting: {normalized}"
                )
            prior_expected = existing["declared_sha256"]
            if (
                prior_expected is not None
                and expected is not None
                and prior_expected != expected
            ):
                raise ValueError(
                    f"Duplicate output locator has conflicting hashes: {normalized}"
                )
            if prior_expected is None and expected is not None:
                existing["declared_sha256"] = expected
            existing["declared_by"].append(declaration["declaration"])
    return dict(sorted(retained.items()))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required OPEN-01 CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Required OPEN-01 CSV is empty: {path}")
    return rows


def _quarter_ordinal(value: Any, *, label: str) -> int:
    text = str(value or "").strip()
    match = re.fullmatch(r"(\d{4})Q([1-4])", text)
    if match is None:
        raise ValueError(f"{label} is not a valid quarter: {text!r}")
    return int(match.group(1)) * 4 + int(match.group(2)) - 1


def _integer(value: Any, *, label: str) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not an integer: {value!r}") from exc
    return parsed


def _finite_number(value: Any, *, label: str) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} is not finite: {value!r}")
    return parsed


def _identifier_tuple(value: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        identifiers = tuple(str(item).strip() for item in value)
    elif isinstance(value, str):
        identifiers = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        raise TypeError(f"{label} must be a sequence or comma-separated string")
    if not identifiers or any(not item for item in identifiers):
        raise ValueError(f"{label} contains no usable identifiers")
    return identifiers


def _optional_identifier_tuple(value: Any) -> tuple[str, ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    return tuple(item.strip() for item in text.split(",") if item.strip())


def _required_mapping(
    payload: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label}.{key} must be a non-empty mapping")
    return value


def _validate_acceptance_metadata(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = _required_mapping(payload, "contract", label="acceptance")
    units = _required_mapping(payload, "units", label="acceptance")
    sample = _required_mapping(payload, "sample", label="acceptance")
    expected_contract = {
        "treatment_id": CANONICAL_TREATMENT_ID,
        "outcome_id": CANONICAL_OUTCOME_ID,
        "residual_id": CANONICAL_RESIDUAL_ID,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise ValueError(
                f"Acceptance contract {key} does not match open_contract"
            )
    if _identifier_tuple(
        contract.get("control_ids"),
        label="acceptance.contract.control_ids",
    ) != tuple(CANONICAL_CONTROL_IDS):
        raise ValueError("Acceptance contract controls do not match open_contract")
    expected_units = {
        "treatment": OPEN_CONTRACT.treatment_units,
        "deposit_outcome": OPEN_CONTRACT.deposit_outcome_units,
        "credit_outcomes": OPEN_CONTRACT.credit_outcome_units,
    }
    for key, expected in expected_units.items():
        if units.get(key) != expected:
            raise ValueError(f"Acceptance unit {key} does not match open_contract")
    _quarter_ordinal(sample.get("start"), label="acceptance.sample.start")
    _quarter_ordinal(sample.get("end"), label="acceptance.sample.end")
    if _integer(sample.get("n"), label="acceptance.sample.n") <= 0:
        raise ValueError("Acceptance sample n must be positive")
    return contract, units, sample


def _cross_surface_contract_gate(
    *,
    root: Path,
    acceptance_outputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_outcomes = (CANONICAL_OUTCOME_ID, CANONICAL_RESIDUAL_ID)
    expected_controls = tuple(CANONICAL_CONTROL_IDS)
    rolling_manifest = _load_json_object(
        _project_path(root, ROLLING_MANIFEST_LOCATOR)
    )
    manifest_errors: list[str] = []
    if rolling_manifest.get("treatment_id") != CANONICAL_TREATMENT_ID:
        manifest_errors.append("treatment_id")
    if _identifier_tuple(
        rolling_manifest.get("outcome_ids"),
        label="rolling_manifest.outcome_ids",
    ) != expected_outcomes:
        manifest_errors.append("outcome_ids")
    if _identifier_tuple(
        rolling_manifest.get("control_ids"),
        label="rolling_manifest.control_ids",
    ) != expected_controls:
        manifest_errors.append("control_ids")
    if _integer(
        rolling_manifest.get("window_quarters"),
        label="rolling_manifest.window_quarters",
    ) != ROLLING_WINDOW_QUARTERS:
        manifest_errors.append("window_quarters")
    window_contract = _required_mapping(
        rolling_manifest,
        "window",
        label="rolling_manifest",
    )
    if _integer(
        window_contract.get("nominal_quarters"),
        label="rolling_manifest.window.nominal_quarters",
    ) != ROLLING_WINDOW_QUARTERS:
        manifest_errors.append("window.nominal_quarters")
    if manifest_errors:
        raise ValueError(
            "Rolling manifest drifted from open_contract: "
            + ", ".join(manifest_errors)
        )

    last_joint = _required_mapping(
        rolling_manifest,
        "last_joint_observed_quarter",
        label="rolling_manifest",
    )
    if set(last_joint) != set(expected_outcomes):
        raise ValueError("Rolling manifest last-joint outcomes are not exact")
    regression_rows = _read_csv_rows(
        _project_path(root, ROLLING_ESTIMATES_LOCATOR)
    )
    if _integer(
        rolling_manifest.get("regression_rows"),
        label="rolling_manifest.regression_rows",
    ) != len(regression_rows):
        raise ValueError("Rolling manifest regression row count does not match CSV")

    rolling_bounds: dict[str, dict[str, str]] = {}
    observed_rolling_outcomes: set[str] = set()
    rolling_deposit_control_patterns: set[
        tuple[tuple[str, ...], tuple[str, ...]]
    ] = set()
    for index, row in enumerate(regression_rows, start=2):
        prefix = f"rolling estimate row {index}"
        outcome = str(row.get("outcome", "")).strip()
        observed_rolling_outcomes.add(outcome)
        if outcome not in expected_outcomes:
            raise ValueError(f"{prefix} has an undeclared outcome")
        if row.get("treatment_id") != CANONICAL_TREATMENT_ID:
            raise ValueError(f"{prefix} does not use canonical treatment")
        if _integer(row.get("window_quarters"), label=f"{prefix} window") != (
            ROLLING_WINDOW_QUARTERS
        ):
            raise ValueError(f"{prefix} does not use the registry window")
        used_controls = _identifier_tuple(
            row.get("control_ids_used"),
            label=f"{prefix} control_ids_used",
        )
        rejected_controls = _optional_identifier_tuple(
            row.get("dropped_control_ids")
        )
        if (
            len(set(used_controls)) != len(used_controls)
            or len(set(rejected_controls)) != len(rejected_controls)
            or set(used_controls).intersection(rejected_controls)
            or set(used_controls).union(rejected_controls)
            != set(expected_controls)
        ):
            raise ValueError(
                f"{prefix} does not serialize an exact canonical control partition"
            )
        if any(
            expected_controls.index(left) >= expected_controls.index(right)
            for left, right in zip(used_controls, used_controls[1:])
        ) or any(
            expected_controls.index(left) >= expected_controls.index(right)
            for left, right in zip(rejected_controls, rejected_controls[1:])
        ):
            raise ValueError(f"{prefix} control partition is not registry ordered")
        pattern = (used_controls, rejected_controls)
        if outcome == CANONICAL_OUTCOME_ID:
            rolling_deposit_control_patterns.add(pattern)

        nominal_start = _quarter_ordinal(
            row.get("window_start_quarter"),
            label=f"{prefix} nominal start",
        )
        nominal_end = _quarter_ordinal(
            row.get("window_end_quarter"),
            label=f"{prefix} nominal end",
        )
        effective_start = _quarter_ordinal(
            row.get("effective_sample_start"),
            label=f"{prefix} effective start",
        )
        effective_end = _quarter_ordinal(
            row.get("effective_sample_end"),
            label=f"{prefix} effective end",
        )
        joint_end = _quarter_ordinal(
            last_joint[outcome],
            label=f"rolling_manifest.last_joint_observed_quarter.{outcome}",
        )
        if nominal_end - nominal_start + 1 != ROLLING_WINDOW_QUARTERS:
            raise ValueError(f"{prefix} nominal bounds are not 48 quarters")
        if not (
            nominal_start
            <= effective_start
            <= effective_end
            <= nominal_end
            <= joint_end
        ):
            raise ValueError(f"{prefix} violates effective-end bounds")
        bounds = rolling_bounds.setdefault(
            outcome,
            {
                "start": str(row["effective_sample_start"]),
                "end": str(row["effective_sample_end"]),
            },
        )
        if effective_start < _quarter_ordinal(
            bounds["start"],
            label=f"{prefix} accumulated start",
        ):
            bounds["start"] = str(row["effective_sample_start"])
        if effective_end > _quarter_ordinal(
            bounds["end"],
            label=f"{prefix} accumulated end",
        ):
            bounds["end"] = str(row["effective_sample_end"])
    if observed_rolling_outcomes != set(expected_outcomes):
        raise ValueError("Rolling estimates do not cover the exact outcome contract")
    if not rolling_deposit_control_patterns:
        raise ValueError("Rolling deposit estimates have no realized control pattern")

    if rolling_manifest.get("effective_sample_start") != min(
        record["start"] for record in rolling_bounds.values()
    ):
        raise ValueError("Rolling manifest effective_sample_start is inconsistent")
    if rolling_manifest.get("effective_sample_end") != max(
        record["end"] for record in rolling_bounds.values()
    ):
        raise ValueError("Rolling manifest effective_sample_end is inconsistent")
    nested_bounds = _required_mapping(
        rolling_manifest,
        "effective_sample_bounds",
        label="rolling_manifest",
    )
    for outcome, expected in rolling_bounds.items():
        outcome_bounds = _required_mapping(
            nested_bounds,
            outcome,
            label="rolling_manifest.effective_sample_bounds",
        )
        regression_bounds = _required_mapping(
            outcome_bounds,
            "regression",
            label=f"rolling_manifest.effective_sample_bounds.{outcome}",
        )
        if regression_bounds != expected:
            raise ValueError(
                f"Rolling manifest regression bounds disagree for {outcome}"
            )

    acceptance_credit = acceptance_outputs.get("credit_screen")
    if (
        not isinstance(acceptance_credit, Mapping)
        or acceptance_credit.get("path") != FORMAL_CREDIT_SCREEN_LOCATOR
    ):
        raise ValueError(
            "Acceptance credit_screen does not name the formal offset-correlates surface"
        )
    credit_rows = _read_csv_rows(
        _project_path(root, FORMAL_CREDIT_SCREEN_LOCATOR)
    )
    if (
        ROLLING_WINDOW_QUARTERS != 48
        or tuple(CREDIT_WINDOW_QUARTERS) != (40, 48, 60)
    ):
        raise ValueError("OPEN-01 credit-screen windows drifted from 40/48/60")
    if tuple(CREDIT_ADJUSTMENTS) != (
        "raw",
        "share_2020_2021_adjusted",
        "linear_time_adjusted",
    ):
        raise ValueError("OPEN-01 credit-screen adjustment family drifted")
    sensitivity_windows = set(CREDIT_WINDOW_QUARTERS).difference(
        {ROLLING_WINDOW_QUARTERS}
    )
    expected_windows = set(CREDIT_WINDOW_QUARTERS)
    expected_credit_cells = {
        (outcome_id, window, adjustment)
        for outcome_id in CREDIT_SCREEN_OUTCOME_IDS
        for window in CREDIT_WINDOW_QUARTERS
        for adjustment in CREDIT_ADJUSTMENTS
    }
    credit_windows_by_outcome: dict[str, set[int]] = {}
    adjustments: set[str] = set()
    calibration_statuses: set[str] = set()
    admission_statuses: set[str] = set()
    credit_hac_by_window: dict[int, dict[str, Any]] = {}
    observed_credit_cells: list[tuple[str, int, str]] = []
    formal_48q_control_patterns: set[
        tuple[tuple[str, ...], tuple[str, ...]]
    ] = set()
    for index, row in enumerate(credit_rows, start=2):
        prefix = f"formal credit-screen row {index}"
        outcome = str(row.get("credit_outcome_id", "")).strip()
        if outcome not in CREDIT_SCREEN_OUTCOME_IDS:
            raise ValueError(f"{prefix} has an undeclared credit outcome")
        if row.get("status") != "computed":
            raise ValueError(f"{prefix} is not computed")
        if row.get("treatment_id") != CANONICAL_TREATMENT_ID:
            raise ValueError(f"{prefix} does not use canonical treatment")
        if row.get("rolling_outcome_id") != CANONICAL_OUTCOME_ID:
            raise ValueError(f"{prefix} does not use matched-deposit rolling outcome")
        if _identifier_tuple(
            row.get("control_ids"),
            label=f"{prefix} control_ids",
        ) != expected_controls:
            raise ValueError(
                f"{prefix} does not declare exact canonical control candidates"
            )
        window = _integer(row.get("window_quarters"), label=f"{prefix} window")
        if window not in expected_windows:
            raise ValueError(f"{prefix} has an undeclared window sensitivity")
        if _integer(
            row.get("rolling_window_observations"),
            label=f"{prefix} rolling_window_observations",
        ) != window:
            raise ValueError(
                f"{prefix} rolling-window observations do not match its window"
            )
        association_observations = _integer(
            row.get("association_observations"),
            label=f"{prefix} association_observations",
        )
        if association_observations < 10:
            raise ValueError(
                f"{prefix} has fewer than 10 association observations"
            )
        if association_observations != _integer(
            row.get("n_windows"),
            label=f"{prefix} n_windows",
        ):
            raise ValueError(
                f"{prefix} association observations disagree with n_windows"
            )
        expected_hac_lags = min(window - 1, association_observations - 2)
        association_hac_lags = _integer(
            row.get("association_hac_lags"),
            label=f"{prefix} association_hac_lags",
        )
        if (
            association_hac_lags != expected_hac_lags
            or _integer(
                row.get("covariance_lags"),
                label=f"{prefix} covariance_lags",
            )
            != expected_hac_lags
        ):
            raise ValueError(f"{prefix} has untruthful overlap-HAC lag metadata")
        bandwidth_ratio = _finite_number(
            row.get("association_hac_bandwidth_ratio"),
            label=f"{prefix} association_hac_bandwidth_ratio",
        )
        if not math.isclose(
            bandwidth_ratio,
            expected_hac_lags / association_observations,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{prefix} has untruthful HAC bandwidth ratio")
        calibration_status = str(
            row.get("inference_calibration_status", "")
        ).strip()
        calibration_method = str(row.get("calibration_method", "")).strip()
        if calibration_status == "calibrated":
            if calibration_method not in CREDIT_CALIBRATION_METHODS:
                raise ValueError(f"{prefix} has an unrecognized calibration method")
        elif calibration_status == (
            "uncalibrated_fixed_bandwidth_normal_reference"
        ):
            if (
                calibration_method
                or str(row.get("calibrated_p_value", "")).strip()
                or str(row.get("calibrated_lower95", "")).strip()
                or str(row.get("calibrated_upper95", "")).strip()
                or str(row.get("outcome_iut_p_value_raw", "")).strip()
                or str(row.get("outcome_iut_p_value_holm", "")).strip()
                or str(row.get("outcome_iut_family_complete", "")).strip().lower()
                not in {"false", "0"}
                or row.get("admission_status") != "appendix_only"
                or row.get("admission_reason")
                != "uncalibrated_component_inference"
            ):
                raise ValueError(
                    f"{prefix} lets uncalibrated normal-HAC inference escape "
                    "the appendix-only gate"
                )
        else:
            raise ValueError(f"{prefix} has an unknown calibration status")
        calibration_statuses.add(calibration_status)
        admission_statuses.add(str(row.get("admission_status", "")).strip())
        hac_record = {
            "association_observations": association_observations,
            "association_hac_lags": association_hac_lags,
            "association_hac_bandwidth_ratio": bandwidth_ratio,
        }
        prior_hac_record = credit_hac_by_window.get(window)
        if prior_hac_record is not None and prior_hac_record != hac_record:
            raise ValueError(
                f"{prefix} disagrees with its window's HAC metadata"
            )
        credit_hac_by_window[window] = hac_record
        last_end = _quarter_ordinal(
            row.get("last_window_end"),
            label=f"{prefix} last_window_end",
        )
        last_observed = _quarter_ordinal(
            row.get("last_observed_treatment_outcome_quarter"),
            label=f"{prefix} last observed quarter",
        )
        if last_end > last_observed:
            raise ValueError(f"{prefix} exceeds the last jointly observed quarter")
        for sign_field in ("sign_40", "sign_48", "sign_60"):
            if row.get(sign_field) not in {"positive", "negative", "zero"}:
                raise ValueError(f"{prefix} lacks declared {sign_field}")
        adjustment = str(row.get("adjustment", "")).strip()
        family = str(row.get("multiple_testing_family", ""))
        if family != f"credit_{window}_{adjustment}":
            raise ValueError(f"{prefix} has a mismatched testing family")
        try:
            serialized_patterns = json.loads(
                str(row.get("rolling_control_patterns_json", ""))
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{prefix} has invalid rolling_control_patterns_json"
            ) from exc
        if not isinstance(serialized_patterns, list) or not serialized_patterns:
            raise ValueError(f"{prefix} has no realized control patterns")
        row_patterns: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        for pattern_index, serialized in enumerate(serialized_patterns):
            if not isinstance(serialized, list) or len(serialized) != 2:
                raise ValueError(
                    f"{prefix} control pattern {pattern_index} is malformed"
                )
            used = _optional_identifier_tuple(serialized[0])
            rejected = _optional_identifier_tuple(serialized[1])
            if (
                not used
                or len(set(used)) != len(used)
                or len(set(rejected)) != len(rejected)
                or set(used).intersection(rejected)
                or set(used).union(rejected) != set(expected_controls)
            ):
                raise ValueError(
                    f"{prefix} control pattern {pattern_index} is not an exact "
                    "canonical partition"
                )
            row_patterns.add((used, rejected))
        if window == ROLLING_WINDOW_QUARTERS:
            if row_patterns != rolling_deposit_control_patterns:
                raise ValueError(
                    f"{prefix} 48q realized controls disagree with rolling estimates"
                )
            formal_48q_control_patterns.update(row_patterns)
        credit_windows_by_outcome.setdefault(outcome, set()).add(window)
        observed_credit_cells.append((outcome, window, adjustment))
        adjustments.add(adjustment)
    if set(credit_windows_by_outcome) != set(CREDIT_SCREEN_OUTCOME_IDS):
        raise ValueError("Formal credit screen does not cover the exact outcome family")
    if any(windows != expected_windows for windows in credit_windows_by_outcome.values()):
        raise ValueError("Formal credit screen has incomplete window sensitivities")
    cell_counts: dict[tuple[str, int, str], int] = {}
    for cell in observed_credit_cells:
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
    observed_credit_set = set(observed_credit_cells)
    duplicate_cells = {
        cell for cell, count in cell_counts.items() if count > 1
    }
    if (
        len(observed_credit_cells) != len(expected_credit_cells)
        or observed_credit_set != expected_credit_cells
        or duplicate_cells
    ):
        raise ValueError(
            "Formal credit screen is not the exact registered "
            "outcome × window × adjustment Cartesian product"
        )
    if formal_48q_control_patterns != rolling_deposit_control_patterns:
        raise ValueError("Formal 48q control patterns do not match rolling estimates")

    return {
        "passed": True,
        "details": {
            "treatment_id": CANONICAL_TREATMENT_ID,
            "rolling_outcome_ids": list(expected_outcomes),
            "control_ids": list(expected_controls),
            "rolling_window_quarters": ROLLING_WINDOW_QUARTERS,
            "rolling_estimate_rows": len(regression_rows),
            "rolling_effective_bounds": rolling_bounds,
            "rolling_realized_control_patterns": [
                {
                    "used": list(used),
                    "rejected": list(rejected),
                }
                for used, rejected in sorted(rolling_deposit_control_patterns)
            ],
            "formal_credit_screen_path": FORMAL_CREDIT_SCREEN_LOCATOR,
            "formal_credit_screen_rows": len(credit_rows),
            "credit_outcome_ids": list(CREDIT_SCREEN_OUTCOME_IDS),
            "canonical_credit_window_quarters": ROLLING_WINDOW_QUARTERS,
            "sign_sensitivity_window_quarters": sorted(sensitivity_windows),
            "adjustments": sorted(adjustments),
            "credit_hac_by_window": {
                str(window): record
                for window, record in sorted(credit_hac_by_window.items())
            },
            "inference_calibration_statuses": sorted(
                calibration_statuses
            ),
            "admission_statuses": sorted(admission_statuses),
        },
    }


def _source_evidence(
    *,
    root: Path,
    tdcest_bundle_path: Path,
    tdcest_bundle_argument: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = _project_path(root, SOURCE_MANIFEST_LOCATOR)
    manifest = _load_json_object(manifest_path)
    input_hashes = manifest.get("input_hashes")
    if not isinstance(input_hashes, dict) or not input_hashes:
        raise ValueError("TDCest source manifest has no input_hashes")
    normalized_input_hashes = {
        str(key): _validate_sha256(value, label=f"TDCest input hash {key}")
        for key, value in input_hashes.items()
    }
    combined_input_hash = _validate_sha256(
        manifest.get("combined_input_hash"),
        label="TDCest combined_input_hash",
    )
    recomputed_combined = hashlib.sha256(
        json.dumps(
            normalized_input_hashes,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if combined_input_hash != recomputed_combined:
        raise ValueError(
            "TDCest combined_input_hash does not match its recorded input_hashes"
        )
    bundle_hash = _sha256_file(tdcest_bundle_path)
    recorded_bundle_hash = _validate_sha256(
        manifest.get("bundle_hash"),
        label="TDCest bundle_hash",
    )
    if recorded_bundle_hash != bundle_hash:
        raise ValueError("TDCest bundle_hash does not match --tdcest-bundle")
    if normalized_input_hashes.get("seed_bundle") != bundle_hash:
        raise ValueError("TDCest seed_bundle input hash does not match --tdcest-bundle")

    standardized_record = _file_record(root, STANDARDIZED_LOCATOR)
    sample_periods: list[str] = []
    canonical_rows = 0
    with _project_path(root, STANDARDIZED_LOCATOR).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            if row.get("series_id") != CANONICAL_TREATMENT_SOURCE_SERIES:
                continue
            value = str(row.get("value", "")).strip()
            period_end = str(row.get("period_end", "")).strip()
            try:
                numeric = float(value)
            except ValueError:
                continue
            if not period_end or not math.isfinite(numeric):
                continue
            canonical_rows += 1
            sample_periods.append(period_end)
    if not sample_periods:
        raise ValueError("Canonical treatment has no finite standardized observations")

    source_record = {
        "manifest": _file_record(root, SOURCE_MANIFEST_LOCATOR),
        "bundle_hash": bundle_hash,
        "input_hashes": normalized_input_hashes,
        "combined_input_hash": combined_input_hash,
        "rows_written": manifest.get("rows_written"),
        "source_bundle_argument": tdcest_bundle_argument,
        "resolved_source_bundle": str(tdcest_bundle_path),
        "standardized": standardized_record,
    }
    sample = {
        "canonical_treatment_source_series": CANONICAL_TREATMENT_SOURCE_SERIES,
        "canonical_observations": canonical_rows,
        "canonical_sample_start": min(sample_periods),
        "canonical_sample_end": max(sample_periods),
    }
    return source_record, sample


def _design_evidence(root: Path) -> dict[str, dict[str, Any]]:
    if len(OPEN01_DESIGN_JOB_IDS) != 4 or len(set(OPEN01_DESIGN_JOB_IDS)) != 4:
        raise ValueError("OPEN01_DESIGN_JOB_IDS must contain exactly four unique jobs")
    evidence: dict[str, dict[str, Any]] = {}
    for job_id in OPEN01_DESIGN_JOB_IDS:
        bundle_locator = f"data/bundles/designs/{job_id}__quarterly_bundle.csv"
        manifest_locator = f"output/manifests/{job_id}__design_manifest.json"
        sample_manifest_locator = f"output/manifests/{job_id}__sample_manifest.json"
        manifest = _load_json_object(_project_path(root, manifest_locator))
        if manifest.get("job_id") != job_id:
            raise ValueError(f"Design manifest job mismatch for {job_id}")
        if manifest.get("status") != "ready_for_estimation":
            raise ValueError(f"Design {job_id} is not ready_for_estimation")
        if manifest.get("treatment_id") != CANONICAL_TREATMENT_ID:
            raise ValueError(
                f"Design {job_id} does not use canonical treatment "
                f"{CANONICAL_TREATMENT_ID}"
            )
        evidence[job_id] = {
            "bundle": _file_record(root, bundle_locator),
            "design_manifest": _file_record(root, manifest_locator),
            "sample_manifest": _file_record(root, sample_manifest_locator),
            "sample_start": manifest.get("sample_start"),
            "sample_end": manifest.get("sample_end"),
            "usable_rows": manifest.get("usable_rows"),
            "treatment_id": manifest.get("treatment_id"),
            "outcome_ids": manifest.get("outcome_ids"),
            "control_ids": manifest.get("control_ids"),
        }
    return evidence


def _producer_manifest_evidence(
    *,
    root: Path,
    command_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected = {
        SOURCE_MANIFEST_LOCATOR,
        SUBMISSION_MANIFEST_LOCATOR,
        ROLLING_MANIFEST_LOCATOR,
        OFFSET_MANIFEST_LOCATOR,
        PERSISTENCE_MANIFEST_LOCATOR,
        REGIME_VALIDATION_MANIFEST_LOCATOR,
        ACCEPTANCE_MANIFEST_LOCATOR,
        *(
            locator
            for job_id in OPEN01_DESIGN_JOB_IDS
            for locator in _design_manifest_locators(job_id)
        ),
    }
    final_producer: dict[str, str] = {}
    for command in command_records:
        stage = str(command["stage"])
        refreshed = command.get("refreshed_manifests")
        if not isinstance(refreshed, dict) or not refreshed:
            raise ValueError(f"Stage {stage} has no refreshed manifest evidence")
        for locator, record in refreshed.items():
            if (
                not isinstance(record, dict)
                or record.get("recreated_after_pre_removal") is not True
            ):
                raise ValueError(
                    f"Stage {stage} lacks fail-closed freshness for {locator}"
                )
            final_producer[str(locator)] = stage
    if set(final_producer) != expected:
        missing = sorted(expected.difference(final_producer))
        extra = sorted(set(final_producer).difference(expected))
        raise ValueError(
            "Producer manifest freshness coverage mismatch: "
            f"missing={missing}, extra={extra}"
        )
    return {
        locator: {
            **_file_record(root, locator),
            "final_writer_stage": final_producer[locator],
            "recreated_after_pre_removal": True,
        }
        for locator in sorted(expected)
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    temporary_path.replace(path)


def _preflight(root: Path, tdcest_bundle: str) -> Path:
    if len(OPEN01_DESIGN_JOB_IDS) != 4 or len(set(OPEN01_DESIGN_JOB_IDS)) != 4:
        raise ValueError("OPEN01_DESIGN_JOB_IDS must contain exactly four unique jobs")
    for locator in PIPELINE_SCRIPT_LOCATORS:
        if not _project_path(root, locator).is_file():
            raise FileNotFoundError(f"Missing OPEN-01 producer stage: {locator}")
    return _resolve_tdcest_bundle(root, tdcest_bundle)


def run_open01(
    *,
    producer_commit: str,
    run_id: str,
    tdcest_bundle: str = "../tdcest/site/data/bundle.json",
    root: Path = ROOT,
    python_executable: str = sys.executable,
    run_command: RunCommand = _default_run_command,
    invocation_argv: list[str] | None = None,
) -> Path:
    root = root.resolve()
    verified_commit = _verify_producer_commit(root, producer_commit)
    verified_run_id = _validate_run_id(run_id)
    tdcest_bundle_path = _preflight(root, tdcest_bundle)
    receipt_path = _project_path(root, RECEIPT_LOCATOR)
    if receipt_path.exists():
        receipt_path.unlink()

    commands = _pipeline_commands(
        python_executable=python_executable,
        tdcest_bundle=tdcest_bundle,
    )
    recorded_argv = (
        list(invocation_argv)
        if invocation_argv is not None
        else [
            str(python_executable),
            "scripts/run_open01_producer.py",
            "--producer-commit",
            verified_commit,
            "--run-id",
            verified_run_id,
            "--tdcest-bundle",
            tdcest_bundle,
        ]
    )
    if not recorded_argv or any(not isinstance(item, str) for item in recorded_argv):
        raise ValueError("Producer argv must be a non-empty list of strings")
    run_started_at = _utc_now()
    command_records = _run_commands(
        commands,
        root=root,
        run_command=run_command,
    )
    if _git_text(root, "rev-parse", "HEAD").lower() != verified_commit:
        raise ValueError("Checkout HEAD changed during the OPEN-01 producer run")

    acceptance_path = _project_path(root, ACCEPTANCE_MANIFEST_LOCATOR)
    acceptance_payload = _load_json_object(acceptance_path)
    acceptance_checks, acceptance_outputs = _validate_acceptance(
        root,
        acceptance_payload,
    )
    acceptance_contract, acceptance_units, acceptance_sample = (
        _validate_acceptance_metadata(acceptance_payload)
    )
    receipt_contract_check = _cross_surface_contract_gate(
        root=root,
        acceptance_outputs=acceptance_outputs,
    )

    source_evidence, standardized_sample = _source_evidence(
        root=root,
        tdcest_bundle_path=tdcest_bundle_path,
        tdcest_bundle_argument=tdcest_bundle,
    )
    design_evidence = _design_evidence(root)
    fixed_outputs = {
        key: _file_record(root, locator)
        for key, locator in RETAINED_OUTPUT_LOCATORS.items()
    }
    declared_outputs = _collect_declared_outputs(
        root=root,
        manifests={
            "submission_appendix": SUBMISSION_MANIFEST_LOCATOR,
            "rolling": ROLLING_MANIFEST_LOCATOR,
            "offset": OFFSET_MANIFEST_LOCATOR,
            "regime_persistence": PERSISTENCE_MANIFEST_LOCATOR,
            "regime_validation": REGIME_VALIDATION_MANIFEST_LOCATOR,
            "acceptance": ACCEPTANCE_MANIFEST_LOCATOR,
        },
    )
    for key, record in fixed_outputs.items():
        declared = declared_outputs.get(record["path"])
        if declared is not None and declared["sha256"] != record["sha256"]:
            raise ValueError(
                f"Fixed and manifest-declared hashes disagree for {key}"
            )
    producer_manifests = _producer_manifest_evidence(
        root=root,
        command_records=command_records,
    )
    receipt = {
        "schema_version": 1,
        "open_id": "OPEN-01",
        "status": "passed",
        "run_id": verified_run_id,
        "producer_commit": verified_commit,
        "argv": recorded_argv,
        "started_at_utc": run_started_at,
        "completed_at_utc": _utc_now(),
        "commands": command_records,
        "producer_manifests": producer_manifests,
        "tdcest": source_evidence,
        "designs": design_evidence,
        "retained_outputs": {
            "fixed": fixed_outputs,
            "manifest_declared": declared_outputs,
        },
        "units": {
            "treatment": OPEN_CONTRACT.treatment_units,
            "deposit_outcome": OPEN_CONTRACT.deposit_outcome_units,
            "credit_outcome": OPEN_CONTRACT.credit_outcome_units,
            "sign_convention": OPEN_CONTRACT.sign_convention,
            "clock": OPEN_CONTRACT.clock,
        },
        "sample": {
            **standardized_sample,
            "designs": {
                job_id: {
                    "sample_start": record["sample_start"],
                    "sample_end": record["sample_end"],
                    "usable_rows": record["usable_rows"],
                }
                for job_id, record in design_evidence.items()
            },
            "rolling_window_quarters": ROLLING_WINDOW_QUARTERS,
            "acceptance": acceptance_sample,
        },
        "contract": {
            "treatment_id": CANONICAL_TREATMENT_ID,
            "outcome_ids": [CANONICAL_OUTCOME_ID, CANONICAL_RESIDUAL_ID],
            "control_ids": list(CANONICAL_CONTROL_IDS),
            "design_job_ids": list(OPEN01_DESIGN_JOB_IDS),
        },
        "acceptance": {
            "status": acceptance_payload["status"],
            "manifest": _file_record(root, ACCEPTANCE_MANIFEST_LOCATOR),
            "acceptance_checks": acceptance_checks,
            "receipt_checks": {
                "rolling_offset_credit_contract": receipt_contract_check,
            },
            "outputs": acceptance_outputs,
            "contract": acceptance_contract,
            "units": acceptance_units,
            "sample": acceptance_sample,
            "producer_inputs": acceptance_payload.get("producer_inputs"),
            "producer_status": acceptance_payload.get("producer_status"),
            "scientific_status": acceptance_payload.get("scientific_status"),
            "issues": acceptance_payload.get("issues"),
            "credit_admission": acceptance_payload.get("credit_admission"),
        },
    }
    _atomic_write_json(receipt_path, receipt)
    return receipt_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fail-closed OPEN-01 producer and write its passing receipt."
    )
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--tdcest-bundle",
        default="../tdcest/site/data/bundle.json",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    receipt = run_open01(
        producer_commit=args.producer_commit,
        run_id=args.run_id,
        tdcest_bundle=args.tdcest_bundle,
        invocation_argv=list(sys.argv),
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "receipt": _relative_locator(ROOT, receipt),
                "sha256": _sha256_file(receipt),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
