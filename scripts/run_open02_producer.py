"""Run the frozen OPEN-02 bank-portfolio response-system producer.

The official Financial Accounts archive is acquired through an explicit mode
and retained as one hash-pinned input bundle.  The response-system mode performs
no network access.  It validates that bundle and the accepted OPEN-01 inputs,
executes the live OPEN-02 pipeline, writes inspectable numerical outputs, and
writes its receipt last so a failed run cannot leave a passing receipt.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ea_tdc.download import (  # noqa: E402
    OPEN02_BOARD_ARCHIVE_RELEASE_DATE,
    OPEN02_BOARD_ARCHIVE_SHA256,
    OPEN02_BOARD_ARCHIVE_URL,
    OPEN02_BOARD_CSV_MEMBER_SHA256,
    OPEN02_BOARD_DICTIONARY_MEMBER_SHA256,
    fetch_open02_board_archive,
)
from ea_tdc.open02 import Open02ValidationError, run_open02_pipeline  # noqa: E402
from ea_tdc.open_contract import OPEN02_CONTRACT, OPEN_CONTRACT  # noqa: E402


SOURCE_BUNDLE_LOCATOR = (
    "data/bundles/open02/financial_accounts_2026-03-19.json"
)
SOURCE_MANIFEST_LOCATOR = "output/manifests/open02_source_manifest.json"
OPEN01_RECEIPT_LOCATOR = "output/manifests/open01_producer_run_receipt.json"
ACCEPTED_DESIGN_LOCATOR = (
    "data/bundles/designs/"
    "tdc_tier2_mmf_rrp_canonical_full_panel__quarterly_bundle.csv"
)
ACCEPTED_STANDARDIZED_LOCATOR = "data/bundles/tdcest/standardized_series.csv"

PANEL_LOCATOR = "data/bundles/open02/tier2_bank_portfolio_panel.csv"
RESPONSE_SYSTEM_LOCATOR = (
    "output/reports/tier2_bank_portfolio_response_system.csv"
)
INFLUENCE_LOCATOR = "output/reports/tier2_bank_portfolio_influence.csv"
ACCEPTANCE_LOCATOR = "output/manifests/open02_acceptance_summary.json"
RECEIPT_LOCATOR = "output/manifests/open02_producer_run_receipt.json"

EXPECTED_OPEN01_PRODUCER_COMMIT = (
    "2e35658c055f7686f621618af14f9d5a9d8b35c3"
)
EXPECTED_ACCEPTED_INPUT_SHA256: Mapping[str, str] = {
    OPEN01_RECEIPT_LOCATOR: (
        "1ee790a0ddaddd8dd06da0236eabd48c7e035354105cf9595c62342470dd303d"
    ),
    ACCEPTED_DESIGN_LOCATOR: (
        "9f7f31c9e139d3cc6dd265c772e3210dad6f1d3294b9fe705d7fa77db83af390"
    ),
    ACCEPTED_STANDARDIZED_LOCATOR: (
        "1a8d1fabd8b05aa1e223381d2fe0eb2cca89c4f6eb27b608aae7156d247e7223"
    ),
}

OUTPUT_LOCATORS = (
    PANEL_LOCATOR,
    RESPONSE_SYSTEM_LOCATOR,
    INFLUENCE_LOCATOR,
    ACCEPTANCE_LOCATOR,
    RECEIPT_LOCATOR,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: Any, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError(f"{label} is not a full SHA-256 digest")
    return text


def _validate_utc_timestamp(value: Any, *, label: str) -> datetime:
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{label} is not an explicit UTC timestamp")
    return parsed


def _project_path(root: Path, locator: str) -> Path:
    candidate = Path(locator)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"OPEN-02 artifact escapes the project root: {locator}"
        ) from exc
    return resolved


def _file_record(
    root: Path,
    locator: str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    path = _project_path(root, locator)
    if not path.is_file():
        raise FileNotFoundError(f"Missing required OPEN-02 artifact: {locator}")
    actual_sha256 = _sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"Hash mismatch for {locator}: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    return {
        "path": path.relative_to(root.resolve()).as_posix(),
        "sha256": actual_sha256,
        "bytes": path.stat().st_size,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}")
    return payload


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Required CSV has no rows: {path}")
    return rows


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
        raise ValueError("--producer-commit must be a full 40-character commit")
    head = _git_text(root, "rev-parse", "HEAD").lower()
    if supplied != head:
        raise ValueError(
            f"Producer commit {supplied} does not equal checkout HEAD {head}"
        )
    changes = _git_text(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    if changes:
        raise ValueError(
            "Checkout has tracked or untracked nonignored files; commit the "
            "producer state before running OPEN-02"
        )
    return head


def _verify_committed_source_producer(
    root: Path,
    producer_commit: str,
) -> str:
    supplied = producer_commit.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", supplied):
        raise ValueError(
            "OPEN-02 source manifest producer must be a full 40-character commit"
        )
    try:
        resolved = _git_text(
            root,
            "rev-parse",
            "--verify",
            f"{supplied}^{{commit}}",
        ).lower()
        _git_text(
            root,
            "merge-base",
            "--is-ancestor",
            supplied,
            "HEAD",
        )
        for locator in (
            "scripts/run_open02_producer.py",
            "src/ea_tdc/open02.py",
            "src/ea_tdc/open_contract.py",
        ):
            _git_text(
                root,
                "cat-file",
                "-e",
                f"{supplied}:{locator}",
            )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "OPEN-02 source manifest producer is not a committed ancestor "
            "containing the required producer files"
        ) from exc
    if resolved != supplied:
        raise ValueError(
            "OPEN-02 source manifest producer does not resolve to its declared "
            "commit"
        )
    return supplied


def _validate_run_id(run_id: str) -> str:
    value = run_id.strip()
    if not value or len(value) > 160:
        raise ValueError("--run-id must contain between 1 and 160 characters")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", value):
        raise ValueError(
            "--run-id may contain only letters, digits, period, underscore, "
            "colon, and hyphen"
        )
    return value


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, _json_bytes(payload))


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty OPEN-02 CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: _csv_cell(row.get(key)) for key in fieldnames}
            )
        temporary = Path(handle.name)
    temporary.replace(path)


def _verify_open01_receipt(payload: Mapping[str, Any]) -> None:
    if payload.get("open_id") != "OPEN-01" or payload.get("status") != "passed":
        raise ValueError("Accepted OPEN-01 producer receipt is not passing")
    if payload.get("producer_commit") != EXPECTED_OPEN01_PRODUCER_COMMIT:
        raise ValueError("Accepted OPEN-01 producer commit changed")
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("status") != "passed":
        raise ValueError("Accepted OPEN-01 acceptance record is not passing")
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("Accepted OPEN-01 receipt has no contract")
    if contract.get("treatment_id") != OPEN_CONTRACT.canonical_treatment_id:
        raise ValueError("OPEN-01 receipt changed the canonical treatment")
    sample = payload.get("sample")
    if not isinstance(sample, dict):
        raise ValueError("Accepted OPEN-01 receipt has no sample evidence")
    accepted_sample = acceptance.get("sample")
    if not isinstance(accepted_sample, dict):
        raise ValueError("Accepted OPEN-01 receipt has no accepted sample")
    if (
        accepted_sample.get("start") != OPEN02_CONTRACT.sample.start_quarter
        or accepted_sample.get("end") != OPEN02_CONTRACT.sample.end_quarter
        or accepted_sample.get("n") != OPEN02_CONTRACT.sample.observations
    ):
        raise ValueError("OPEN-01 receipt does not carry the frozen OPEN-02 sample")


def _load_source_bundle(
    *,
    root: Path,
    bundle_locator: str,
    manifest_locator: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle_record = _file_record(root, bundle_locator)
    manifest_path = _project_path(root, manifest_locator)
    manifest = _load_json_object(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("open_id") != "OPEN-02"
        or manifest.get("status") != "passed"
        or manifest.get("stage") != "source_acquisition"
    ):
        raise Open02ValidationError(
            "metadata_gate_failed",
            "OPEN-02 source manifest is not passing",
        )
    if manifest.get("observation_vintage") != OPEN02_BOARD_ARCHIVE_RELEASE_DATE:
        raise Open02ValidationError(
            "vintage_gate_failed",
            "OPEN-02 source manifest is not the pinned March vintage",
        )
    if (
        manifest.get("observation_vintage_cutoff")
        != OPEN02_CONTRACT.sample.observation_vintage_cutoff
    ):
        raise Open02ValidationError(
            "vintage_gate_failed",
            "OPEN-02 source manifest changed the observation-vintage cutoff",
        )
    acquisition_commit = str(manifest.get("producer_commit", "")).strip()
    acquisition_commit = _verify_committed_source_producer(
        root,
        acquisition_commit,
    )
    if not str(manifest.get("run_id", "")).strip():
        raise Open02ValidationError(
            "metadata_gate_failed",
            "OPEN-02 source manifest has no exact acquisition run",
        )
    acquisition_argv = manifest.get("argv")
    if (
        not isinstance(acquisition_argv, list)
        or not acquisition_argv
        or any(not isinstance(item, str) or not item for item in acquisition_argv)
    ):
        raise Open02ValidationError(
            "metadata_gate_failed",
            "OPEN-02 source manifest has no exact acquisition argv",
        )
    try:
        started_at = _validate_utc_timestamp(
            manifest.get("started_at_utc"),
            label="OPEN-02 source started_at_utc",
        )
        completed_at = _validate_utc_timestamp(
            manifest.get("completed_at_utc"),
            label="OPEN-02 source completed_at_utc",
        )
    except ValueError as exc:
        raise Open02ValidationError(
            "metadata_gate_failed",
            str(exc),
        ) from exc
    if completed_at < started_at:
        raise Open02ValidationError(
            "metadata_gate_failed",
            "OPEN-02 source acquisition completed before it started",
        )
    official_release = manifest.get("official_release")
    expected_release = {
        "source_url": OPEN02_BOARD_ARCHIVE_URL,
        "release_date": OPEN02_BOARD_ARCHIVE_RELEASE_DATE,
        "archive_sha256": OPEN02_BOARD_ARCHIVE_SHA256,
        "csv_member_sha256": dict(OPEN02_BOARD_CSV_MEMBER_SHA256),
        "dictionary_member_sha256": dict(
            OPEN02_BOARD_DICTIONARY_MEMBER_SHA256
        ),
    }
    if not isinstance(official_release, dict):
        raise Open02ValidationError(
            "metadata_gate_failed",
            "OPEN-02 source manifest has no official release record",
        )
    for field, expected in expected_release.items():
        if official_release.get(field) != expected:
            raise Open02ValidationError(
                "vintage_gate_failed",
                f"OPEN-02 source manifest changed official release field {field}",
                details={"field": field},
            )
    _validate_sha256(
        official_release.get("normalized_rows_sha256"),
        label="OPEN-02 normalized official rows hash",
    )
    metadata_gate = manifest.get("metadata_gate")
    coverage_gate = manifest.get("coverage_gate")
    if not isinstance(metadata_gate, dict) or metadata_gate.get("passed") is not True:
        raise Open02ValidationError(
            "metadata_gate_failed",
            "OPEN-02 source metadata gate is not passing",
        )
    if (
        metadata_gate.get("series_ids")
        != [
            series.board_series_id for series in OPEN02_CONTRACT.series
        ]
        or metadata_gate.get(
            "verified_from_bundled_data_dictionaries"
        )
        is not True
    ):
        raise Open02ValidationError(
            "metadata_gate_failed",
            "OPEN-02 source manifest omits exact dictionary-backed series evidence",
        )
    if not isinstance(coverage_gate, dict) or coverage_gate.get("passed") is not True:
        raise Open02ValidationError(
            "coverage_gate_failed",
            "OPEN-02 source coverage gate is not passing",
        )
    if (
        metadata_gate.get("series_count") != len(OPEN02_CONTRACT.series)
        or coverage_gate.get("series_count") != len(OPEN02_CONTRACT.series)
        or coverage_gate.get("observations")
        != OPEN02_CONTRACT.sample.observations
        or coverage_gate.get("sample_start")
        != OPEN02_CONTRACT.sample.start_quarter
        or coverage_gate.get("sample_end") != OPEN02_CONTRACT.sample.end_quarter
    ):
        raise Open02ValidationError(
            "coverage_gate_failed",
            "OPEN-02 source manifest gate evidence drifted",
        )
    declared_bundle = manifest.get("source_bundle")
    if not isinstance(declared_bundle, dict):
        raise ValueError("OPEN-02 source manifest has no source_bundle record")
    if declared_bundle.get("path") != bundle_record["path"]:
        raise ValueError("OPEN-02 source manifest points to another bundle")
    if declared_bundle.get("bytes") != bundle_record["bytes"]:
        raise ValueError("OPEN-02 source manifest changed the bundle byte count")
    if (
        _validate_sha256(
            declared_bundle.get("sha256"),
            label="OPEN-02 source bundle hash",
        )
        != bundle_record["sha256"]
    ):
        raise ValueError("OPEN-02 source bundle hash does not match its manifest")
    source_bundle = _load_json_object(_project_path(root, bundle_locator))
    if (
        source_bundle.get("schema_version") != 1
        or source_bundle.get("kind") != "open02_financial_accounts_input"
    ):
        raise Open02ValidationError(
            "metadata_gate_failed",
            "OPEN-02 source bundle changed its schema or kind",
        )
    if (
        source_bundle.get("observation_vintage")
        != OPEN02_BOARD_ARCHIVE_RELEASE_DATE
        or source_bundle.get("accepted_tdcest_bundle_generated_at")
        != OPEN02_CONTRACT.sample.accepted_tdcest_bundle_generated_at
    ):
        raise Open02ValidationError(
            "vintage_gate_failed",
            "OPEN-02 source bundle changed its pinned vintage",
        )
    bundle_release = source_bundle.get("official_release")
    if not isinstance(bundle_release, dict):
        raise ValueError("OPEN-02 source bundle has no official release record")
    for bundle_field, manifest_field in (
        ("source_url", "source_url"),
        ("release_date", "release_date"),
        ("archive_sha256", "archive_sha256"),
        ("csv_member_sha256", "csv_member_sha256"),
        ("dictionary_member_sha256", "dictionary_member_sha256"),
        ("rows_sha256", "normalized_rows_sha256"),
    ):
        if bundle_release.get(bundle_field) != official_release.get(
            manifest_field
        ):
            raise Open02ValidationError(
                "vintage_gate_failed",
                "OPEN-02 source bundle and manifest disagree on "
                f"{bundle_field}",
                details={"field": bundle_field},
            )
    input_hashes = {
        "official_archive": official_release["archive_sha256"],
        **{
            f"official:{member_name}": sha256
            for member_name, sha256 in sorted(
                official_release["csv_member_sha256"].items()
            )
        },
        **{
            f"official:{member_name}": sha256
            for member_name, sha256 in sorted(
                official_release["dictionary_member_sha256"].items()
            )
        },
        "normalized_official_rows": official_release[
            "normalized_rows_sha256"
        ],
        "source_bundle": bundle_record["sha256"],
        "source_manifest": _file_record(root, manifest_locator)["sha256"],
    }
    combined_input_hash = hashlib.sha256(
        json.dumps(
            input_hashes,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return source_bundle, {
        "bundle": bundle_record,
        "manifest": _file_record(root, manifest_locator),
        "input_hashes": input_hashes,
        "combined_sha256": combined_input_hash,
        "observation_vintage": manifest.get("observation_vintage"),
        "official_release": official_release,
        "acquisition": {
            "producer_commit": acquisition_commit,
            "run_id": manifest.get("run_id"),
            "completed_at_utc": manifest.get("completed_at_utc"),
        },
        "metadata_gate": metadata_gate,
        "coverage_gate": coverage_gate,
    }


def _load_accepted_inputs(
    root: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    records = {
        locator: _file_record(
            root,
            locator,
            expected_sha256=expected_sha256,
        )
        for locator, expected_sha256 in EXPECTED_ACCEPTED_INPUT_SHA256.items()
    }
    receipt = _load_json_object(_project_path(root, OPEN01_RECEIPT_LOCATOR))
    _verify_open01_receipt(receipt)
    design_rows = _read_csv_rows(_project_path(root, ACCEPTED_DESIGN_LOCATOR))
    standardized_rows = _read_csv_rows(
        _project_path(root, ACCEPTED_STANDARDIZED_LOCATOR)
    )
    combined_hash = hashlib.sha256(
        json.dumps(
            {
                locator: record["sha256"]
                for locator, record in sorted(records.items())
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return design_rows, standardized_rows, {
        "files": records,
        "input_hashes": {
            locator: record["sha256"]
            for locator, record in sorted(records.items())
        },
        "combined_sha256": combined_hash,
        "open01_producer_commit": receipt.get("producer_commit"),
        "open01_run_id": receipt.get("run_id"),
        "open01_scientific_status": (
            receipt.get("acceptance", {}).get("scientific_status")
            if isinstance(receipt.get("acceptance"), dict)
            else None
        ),
    }


def _remove_previous_outputs(root: Path) -> None:
    for locator in OUTPUT_LOCATORS:
        path = _project_path(root, locator)
        if path.exists() and not path.is_file():
            raise ValueError(f"OPEN-02 output locator is not a file: {locator}")
        if path.is_file():
            path.unlink()


def _expected_sample_quarters() -> tuple[str, ...]:
    return tuple(
        f"{ordinal // 4}Q{ordinal % 4 + 1}"
        for ordinal in range(
            int(OPEN02_CONTRACT.sample.start_quarter[:4]) * 4
            + int(OPEN02_CONTRACT.sample.start_quarter[-1])
            - 1,
            int(OPEN02_CONTRACT.sample.end_quarter[:4]) * 4
            + int(OPEN02_CONTRACT.sample.end_quarter[-1]),
        )
    )


def _pipeline_source_bundle(
    parsed_archive: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = parsed_archive.get("metadata")
    rows = parsed_archive.get("rows")
    if not isinstance(metadata, dict) or not isinstance(rows, list):
        raise TypeError("Parsed OPEN-02 Board archive is malformed")
    expected_source_fields = {
        "source_url": OPEN02_BOARD_ARCHIVE_URL,
        "release_date": OPEN02_BOARD_ARCHIVE_RELEASE_DATE,
        "archive_sha256": OPEN02_BOARD_ARCHIVE_SHA256,
        "csv_member_sha256": dict(OPEN02_BOARD_CSV_MEMBER_SHA256),
        "dictionary_member_sha256": dict(
            OPEN02_BOARD_DICTIONARY_MEMBER_SHA256
        ),
    }
    for field, expected in expected_source_fields.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"Parsed OPEN-02 Board archive changed source field {field}"
            )
    _validate_sha256(
        metadata.get("rows_sha256"),
        label="parsed OPEN-02 normalized rows hash",
    )
    if metadata.get("series_count") != len(OPEN02_CONTRACT.series):
        raise ValueError("Parsed OPEN-02 Board archive has the wrong series count")
    if metadata.get("observations") != OPEN02_CONTRACT.sample.observations:
        raise ValueError("Parsed OPEN-02 Board archive has the wrong row count")
    metadata_by_key: dict[str, Mapping[str, Any]] = {}
    metadata_rows = metadata.get("series")
    if not isinstance(metadata_rows, list):
        raise TypeError("Parsed OPEN-02 Board archive has no series metadata")
    for record in metadata_rows:
        if not isinstance(record, dict):
            raise TypeError("Parsed OPEN-02 series metadata is malformed")
        key = str(record.get("key", ""))
        if not key or key in metadata_by_key:
            raise ValueError("Parsed OPEN-02 series metadata has duplicate keys")
        metadata_by_key[key] = record

    expected_quarters = _expected_sample_quarters()
    observed_quarters = tuple(
        str(row.get("quarter", "")) if isinstance(row, dict) else ""
        for row in rows
    )
    if observed_quarters != expected_quarters:
        raise ValueError("Parsed OPEN-02 Board rows violate the frozen sample")

    series_bundle: list[dict[str, Any]] = []
    for series in OPEN02_CONTRACT.series:
        if series.key not in metadata_by_key:
            raise ValueError(f"Parsed archive omits metadata for {series.key}")
        source_metadata = dict(metadata_by_key[series.key])
        observations = []
        for row in rows:
            if not isinstance(row, dict) or series.key not in row:
                raise ValueError(f"Parsed archive omits {series.key} observations")
            observations.append(
                {
                    "quarter": str(row["quarter"]),
                    "value": row[series.key],
                }
            )
        series_bundle.append(
            {
                **asdict(series),
                "observation_vintage": "2026-03-19",
                "source_metadata": source_metadata,
                "observations": observations,
            }
        )
    return {
        "schema_version": 1,
        "kind": "open02_financial_accounts_input",
        "accepted_tdcest_bundle_generated_at": (
            OPEN02_CONTRACT.sample.accepted_tdcest_bundle_generated_at
        ),
        "observation_vintage": "2026-03-19",
        "observation_vintage_cutoff": (
            OPEN02_CONTRACT.sample.observation_vintage_cutoff
        ),
        "official_release": {
            key: value
            for key, value in metadata.items()
            if key != "series"
        },
        "series": series_bundle,
    }


def acquire_open02_source(
    *,
    producer_commit: str,
    run_id: str,
    root: Path = ROOT,
    source_bundle_locator: str = SOURCE_BUNDLE_LOCATOR,
    source_manifest_locator: str = SOURCE_MANIFEST_LOCATOR,
    invocation_argv: list[str] | None = None,
) -> Path:
    """Acquire and materialize the one frozen official OPEN-02 source bundle."""

    root = root.resolve()
    verified_commit = _verify_producer_commit(root, producer_commit)
    verified_run_id = _validate_run_id(run_id)
    bundle_path = _project_path(root, source_bundle_locator)
    manifest_path = _project_path(root, source_manifest_locator)
    for path in (manifest_path, bundle_path):
        if path.exists() and not path.is_file():
            raise ValueError(f"OPEN-02 source locator is not a file: {path}")

    started_at = _utc_now()
    parsed_archive = fetch_open02_board_archive()
    bundle = _pipeline_source_bundle(parsed_archive)
    bundle_bytes = _json_bytes(bundle)
    bundle_record = {
        "path": bundle_path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "bytes": len(bundle_bytes),
    }
    metadata = parsed_archive["metadata"]
    recorded_argv = (
        list(invocation_argv)
        if invocation_argv is not None
        else [
            str(sys.executable),
            "scripts/run_open02_producer.py",
            "--producer-commit",
            verified_commit,
            "--run-id",
            verified_run_id,
            "--acquire-source",
        ]
    )
    if not recorded_argv or any(not isinstance(item, str) for item in recorded_argv):
        raise ValueError("Acquisition argv must be a non-empty list of strings")
    if _git_text(root, "rev-parse", "HEAD").lower() != verified_commit:
        raise ValueError("Checkout HEAD changed during OPEN-02 source acquisition")

    manifest = {
        "schema_version": 1,
        "open_id": "OPEN-02",
        "status": "passed",
        "stage": "source_acquisition",
        "producer_commit": verified_commit,
        "run_id": verified_run_id,
        "argv": recorded_argv,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "observation_vintage": "2026-03-19",
        "observation_vintage_cutoff": (
            OPEN02_CONTRACT.sample.observation_vintage_cutoff
        ),
        "official_release": {
            "source_url": metadata.get("source_url"),
            "release_date": metadata.get("release_date"),
            "archive_sha256": metadata.get("archive_sha256"),
            "csv_member_sha256": metadata.get("csv_member_sha256"),
            "dictionary_member_sha256": (
                metadata.get("dictionary_member_sha256")
            ),
            "normalized_rows_sha256": metadata.get("rows_sha256"),
        },
        "source_bundle": bundle_record,
        "metadata_gate": {
            "passed": True,
            "series_count": metadata.get("series_count"),
            "series_ids": [
                series.board_series_id for series in OPEN02_CONTRACT.series
            ],
            "verified_from_bundled_data_dictionaries": True,
        },
        "coverage_gate": {
            "passed": True,
            "sample_start": metadata.get("sample_start"),
            "sample_end": metadata.get("sample_end"),
            "observations": metadata.get("observations"),
            "series_count": metadata.get("series_count"),
        },
    }
    manifest_bytes = _json_bytes(manifest)
    prior_bytes = {
        path: path.read_bytes() if path.is_file() else None
        for path in (bundle_path, manifest_path)
    }
    try:
        _atomic_write_bytes(bundle_path, bundle_bytes)
        _atomic_write_bytes(manifest_path, manifest_bytes)
    except BaseException:
        for path, previous in prior_bytes.items():
            if previous is None:
                if path.is_file():
                    path.unlink()
            else:
                _atomic_write_bytes(path, previous)
        raise
    return manifest_path


def _write_pipeline_outputs(root: Path, result: Any) -> dict[str, Any]:
    panel_rows = tuple(result.panel_rows)
    estimate_rows = tuple(result.estimate_rows)
    wald_rows = tuple(result.wald_rows)
    influence_rows = tuple(result.influence_rows)
    influence_summaries = tuple(result.influence_summaries)
    acceptance = dict(result.acceptance)

    expected_quarters = _expected_sample_quarters()
    panel_quarters = tuple(str(row.get("quarter", "")) for row in panel_rows)
    if (
        len(panel_rows) != OPEN02_CONTRACT.sample.observations
        or panel_quarters != expected_quarters
    ):
        raise ValueError(
            "OPEN-02 pipeline output does not contain the exact frozen "
            "96-quarter panel"
        )
    source_system, within_system = OPEN02_CONTRACT.systems
    source_component_ids = tuple(
        f"beta_{outcome_id}"
        for outcome_id in source_system.agency_component_outcome_ids
    )
    within_equation_count = len(within_system.outcome_ids)
    within_component_ids = tuple(
        f"{prefix}_{outcome_id}"
        for prefix in (
            within_system.coefficient_ids[0].split("_", 1)[0],
            within_system.coefficient_ids[within_equation_count].split("_", 1)[0],
        )
        for outcome_id in within_system.agency_component_outcome_ids
    )
    expected_coefficient_ids = {
        *source_system.coefficient_ids,
        *source_component_ids,
        *within_system.coefficient_ids,
        *within_component_ids,
    }
    observed_coefficient_ids = [
        str(row.get("coefficient_id", "")) for row in estimate_rows
    ]
    if (
        len(observed_coefficient_ids) != len(expected_coefficient_ids)
        or len(set(observed_coefficient_ids)) != len(observed_coefficient_ids)
        or set(observed_coefficient_ids) != expected_coefficient_ids
    ):
        raise ValueError(
            "OPEN-02 pipeline output does not contain the exact coefficient "
            "registry"
        )

    response_rows = [
        {"record_type": "coefficient", **dict(row)}
        for row in estimate_rows
    ]
    response_rows.extend(
        {"record_type": "wald", **dict(row)}
        for row in wald_rows
    )
    design_evidence = acceptance.get("design_evidence")
    identity_evidence = acceptance.get("identity_evidence")
    deterministic_gates = acceptance.get("deterministic_gates")
    if not isinstance(design_evidence, dict):
        raise ValueError("OPEN-02 pipeline omitted design evidence")
    if not isinstance(identity_evidence, dict):
        raise ValueError("OPEN-02 pipeline omitted identity evidence")
    if not isinstance(deterministic_gates, (list, tuple)):
        raise ValueError("OPEN-02 pipeline omitted deterministic gates")
    gate_by_id = {
        str(row.get("gate_id")): row
        for row in deterministic_gates
        if isinstance(row, Mapping)
    }
    expected_gate_ids = tuple(
        gate.gate_id for gate in OPEN02_CONTRACT.validity_gates
    )
    if (
        tuple(gate_by_id) != expected_gate_ids
        or len(gate_by_id) != len(deterministic_gates)
        or any(row.get("passed") is not True for row in gate_by_id.values())
    ):
        raise ValueError("OPEN-02 pipeline did not pass the exact 12 validity gates")
    source_design = design_evidence.get("source")
    within_design = design_evidence.get("within")
    if not isinstance(source_design, dict) or not isinstance(within_design, dict):
        raise ValueError("OPEN-02 pipeline omitted system design hashes")
    row_hash = _validate_sha256(
        design_evidence.get("row_hash"),
        label="OPEN-02 result row hash",
    )
    column_hashes = {
        "source": _validate_sha256(
            source_design.get("column_hash"),
            label="OPEN-02 source column hash",
        ),
        "within": _validate_sha256(
            within_design.get("column_hash"),
            label="OPEN-02 within column hash",
        ),
    }
    design_hashes = {
        "source": _validate_sha256(
            source_design.get("design_hash"),
            label="OPEN-02 source design hash",
        ),
        "within": _validate_sha256(
            within_design.get("design_hash"),
            label="OPEN-02 within design hash",
        ),
    }
    expected_identity_ids = {
        "treasury_component",
        "accepted_component_reconciliation",
        "leave_out_reconstruction",
        "us_agency_identity",
        "three_sector_agency_identity",
    }
    if set(identity_evidence) != expected_identity_ids:
        raise ValueError(
            "OPEN-02 pipeline returned the wrong identity-evidence registry"
        )
    for identity_id, value in identity_evidence.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"OPEN-02 identity evidence is not numeric: {identity_id}"
            ) from exc
        if not isfinite(numeric) or numeric < 0.0:
            raise ValueError(
                f"OPEN-02 identity evidence is invalid: {identity_id}"
            )
        if numeric > OPEN02_CONTRACT.sample.identity_tolerance_usd_millions:
            raise ValueError(
                f"OPEN-02 identity evidence exceeds tolerance: {identity_id}"
            )
    wald_by_id = {
        str(row["hypothesis_id"]): row
        for row in wald_rows
    }
    if (
        len(wald_rows) != len(OPEN02_CONTRACT.holm.hypothesis_ids)
        or len(wald_by_id) != len(wald_rows)
        or set(wald_by_id) != set(OPEN02_CONTRACT.holm.hypothesis_ids)
    ):
        raise ValueError("OPEN-02 pipeline returned the wrong Wald family")
    computed_reason_codes: set[str] = set()
    wald_reason_by_id = dict(
        zip(
            OPEN02_CONTRACT.holm.hypothesis_ids,
            OPEN02_CONTRACT.promotion_reason_codes[:3],
            strict=True,
        )
    )
    for hypothesis_id, row in wald_by_id.items():
        statistic = float(row.get("statistic"))
        raw_p_value = float(row.get("raw_p_value"))
        adjusted_p_value = float(row.get("holm_adjusted_p_value"))
        if (
            not isfinite(statistic)
            or statistic < 0.0
            or not isfinite(raw_p_value)
            or not 0.0 <= raw_p_value <= 1.0
            or not isfinite(adjusted_p_value)
            or not 0.0 <= adjusted_p_value <= 1.0
        ):
            raise ValueError(
                f"OPEN-02 pipeline returned invalid Wald evidence for {hypothesis_id}"
            )
        rejected = (
            adjusted_p_value <= OPEN02_CONTRACT.holm.familywise_alpha
        )
        if row.get("passed") is not rejected:
            raise ValueError(
                f"OPEN-02 Wald pass flag disagrees with Holm p-value for {hypothesis_id}"
            )
        if not rejected:
            computed_reason_codes.add(wald_reason_by_id[hypothesis_id])
    influence_by_group = {
        str(row["group_id"]): row
        for row in influence_summaries
    }
    expected_groups = {
        group.group_id for group in OPEN02_CONTRACT.influence.groups
    }
    if (
        len(influence_summaries) != len(expected_groups)
        or len(influence_by_group) != len(influence_summaries)
        or set(influence_by_group) != expected_groups
    ):
        raise ValueError("OPEN-02 pipeline returned the wrong influence groups")
    expected_influence_rows = len(expected_groups) * (
        OPEN02_CONTRACT.influence.quarter_deletion_fits
        + OPEN02_CONTRACT.influence.block_deletion_fits
    )
    if len(influence_rows) != expected_influence_rows:
        raise ValueError(
            "OPEN-02 pipeline returned the wrong number of influence refits"
        )
    expected_leave_one = tuple((quarter,) for quarter in expected_quarters)
    block_size = OPEN02_CONTRACT.influence.block_deletion_size
    expected_leave_block = tuple(
        expected_quarters[start : start + block_size]
        for start in range(OPEN02_CONTRACT.influence.block_deletion_fits)
    )
    for group_id, row in influence_by_group.items():
        if (
            row.get("leave_one_fits")
            != OPEN02_CONTRACT.influence.quarter_deletion_fits
            or row.get("leave_block_fits")
            != OPEN02_CONTRACT.influence.block_deletion_fits
        ):
            raise ValueError(
                f"OPEN-02 pipeline returned incomplete influence counts for {group_id}"
            )
        group_rows = [
            candidate
            for candidate in influence_rows
            if candidate.get("group_id") == group_id
        ]
        leave_one_omissions = tuple(
            tuple(candidate.get("omitted_quarters", ()))
            for candidate in group_rows
            if candidate.get("deletion_kind") == "leave_one"
        )
        leave_block_omissions = tuple(
            tuple(candidate.get("omitted_quarters", ()))
            for candidate in group_rows
            if candidate.get("deletion_kind") == "leave_block"
        )
        if (
            leave_one_omissions != expected_leave_one
            or leave_block_omissions != expected_leave_block
        ):
            raise ValueError(
                f"OPEN-02 pipeline returned incomplete deletion windows for {group_id}"
            )
        for field in (
            "maximum_leave_one_influence",
            "maximum_leave_block_influence",
        ):
            value = float(row.get(field))
            if not isfinite(value) or value < 0.0:
                raise ValueError(
                    f"OPEN-02 pipeline returned invalid {field} for {group_id}"
                )
        maximum_one = float(row["maximum_leave_one_influence"])
        maximum_block = float(row["maximum_leave_block_influence"])
        sign_flip = row.get("sign_flip_detected")
        if not isinstance(sign_flip, bool):
            raise ValueError(
                f"OPEN-02 pipeline omitted the influence sign flag for {group_id}"
            )
        local_reasons: list[str] = []
        if maximum_one > OPEN02_CONTRACT.influence.maximum_quarter_influence:
            local_reasons.append("leave_quarter_influence_gt_0_25")
        if maximum_block > OPEN02_CONTRACT.influence.maximum_block_influence:
            local_reasons.append("leave_block_influence_gt_0_50")
        if sign_flip:
            local_reasons.append("sign_flip_under_influence")
        if (
            row.get("passed") is not (not local_reasons)
            or tuple(row.get("reason_codes", ())) != tuple(local_reasons)
        ):
            raise ValueError(
                f"OPEN-02 influence disposition is inconsistent for {group_id}"
            )
        computed_reason_codes.update(local_reasons)
    valid_result = acceptance.get("valid_result")
    main_text_eligible = acceptance.get("main_text_eligible")
    appendix_only = acceptance.get("appendix_only")
    disposition = (
        valid_result,
        main_text_eligible,
        appendix_only,
    )
    allowed_dispositions = {
        (
            item.valid_result,
            item.main_text_eligible,
            item.appendix_only,
        )
        for item in (
            OPEN02_CONTRACT.invalid_result_disposition,
            OPEN02_CONTRACT.valid_nonpromoted_disposition,
            OPEN02_CONTRACT.promoted_result_disposition,
        )
    }
    if disposition not in allowed_dispositions or valid_result is not True:
        raise ValueError("OPEN-02 pipeline returned an invalid result disposition")
    reason_codes = acceptance.get("reason_codes")
    if not isinstance(reason_codes, (list, tuple)):
        raise ValueError("OPEN-02 pipeline omitted eligibility reason codes")
    ordered_reason_codes = tuple(
        reason
        for reason in OPEN02_CONTRACT.promotion_reason_codes
        if reason in computed_reason_codes
    )
    if (
        tuple(reason_codes) != ordered_reason_codes
        or main_text_eligible is not (not ordered_reason_codes)
        or appendix_only is not bool(ordered_reason_codes)
    ):
        raise ValueError("OPEN-02 pipeline returned inconsistent eligibility reasons")
    acceptance_payload = {
        "schema_version": 1,
        "open_id": "OPEN-02",
        "status": "passed",
        "gates": list(deterministic_gates),
        "identity_errors": identity_evidence,
        "row_hash": row_hash,
        "column_hashes": column_hashes,
        "design_hashes": design_hashes,
        "wald_statistics": {
            hypothesis_id: row.get("statistic")
            for hypothesis_id, row in wald_by_id.items()
        },
        "raw_p_values": {
            hypothesis_id: row.get("raw_p_value")
            for hypothesis_id, row in wald_by_id.items()
        },
        "holm_adjusted_p_values": {
            hypothesis_id: row.get("holm_adjusted_p_value")
            for hypothesis_id, row in wald_by_id.items()
        },
        "influence_maxima": {
            group_id: {
                "leave_one": row.get("maximum_leave_one_influence"),
                "leave_block": row.get("maximum_leave_block_influence"),
            }
            for group_id, row in influence_by_group.items()
        },
        "wald_tests": list(wald_rows),
        "influence_summaries": list(influence_summaries),
        "valid_result": valid_result,
        "main_text_eligible": main_text_eligible,
        "appendix_only": appendix_only,
        "reason_codes": list(reason_codes),
        "pipeline_evidence": acceptance,
    }
    _atomic_write_csv(_project_path(root, PANEL_LOCATOR), panel_rows)
    _atomic_write_csv(
        _project_path(root, RESPONSE_SYSTEM_LOCATOR),
        response_rows,
    )
    _atomic_write_csv(
        _project_path(root, INFLUENCE_LOCATOR),
        influence_rows,
    )
    _atomic_write_json(_project_path(root, ACCEPTANCE_LOCATOR), acceptance_payload)
    return {
        "panel_rows": len(panel_rows),
        "estimate_rows": len(estimate_rows),
        "wald_rows": len(wald_rows),
        "influence_rows": len(influence_rows),
        "artifacts": {
            "panel": _file_record(root, PANEL_LOCATOR),
            "response_system": _file_record(root, RESPONSE_SYSTEM_LOCATOR),
            "influence": _file_record(root, INFLUENCE_LOCATOR),
            "acceptance": _file_record(root, ACCEPTANCE_LOCATOR),
        },
        "acceptance": acceptance_payload,
    }


def _require_receipt_contract(receipt: Mapping[str, Any]) -> None:
    if OPEN02_CONTRACT.output.report_path != RESPONSE_SYSTEM_LOCATOR:
        raise ValueError("OPEN-02 response-system locator drifted from the contract")
    if OPEN02_CONTRACT.output.receipt_path != RECEIPT_LOCATOR:
        raise ValueError("OPEN-02 receipt locator drifted from the contract")
    missing: list[str] = []
    for field_path in OPEN02_CONTRACT.output.required_receipt_fields:
        value: Any = receipt
        for part in field_path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                missing.append(field_path)
                break
            value = value[part]
    if missing:
        raise ValueError(
            "OPEN-02 receipt omits required contract fields: "
            + ", ".join(missing)
        )


def _receipt_common_payload(
    *,
    verified_commit: str,
    verified_run_id: str,
    recorded_argv: Sequence[str],
    started_at: str,
    source_evidence: Mapping[str, Any],
    accepted_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "open_id": "OPEN-02",
        "run_id": verified_run_id,
        "producer_commit": verified_commit,
        "argv": list(recorded_argv),
        "started_at_utc": started_at,
        "source": dict(source_evidence),
        "accepted_open01_inputs": dict(accepted_inputs),
        "units": {
            "treatment_and_responses": "usd_millions_per_quarter",
            "coefficients": "usd_millions_per_usd_million",
            "sign": (
                "positive assets are net acquisition; positive liabilities are "
                "net incurrence; canonical treatment is deposit-positive"
            ),
        },
        "sample": {
            "start": OPEN02_CONTRACT.sample.start_quarter,
            "end": OPEN02_CONTRACT.sample.end_quarter,
            "n": OPEN02_CONTRACT.sample.observations,
            "quarter_hash": OPEN02_CONTRACT.sample.quarter_hash,
        },
        "contract": {
            "canonical_treatment_id": OPEN02_CONTRACT.canonical_treatment_id,
            "canonical_treatment_source_series": (
                OPEN02_CONTRACT.canonical_treatment_source_series
            ),
            "embedded_bank_treasury_component_id": (
                OPEN02_CONTRACT.embedded_bank_treasury_component_id
            ),
            "series_count": len(OPEN02_CONTRACT.series),
            "control_ids": list(OPEN02_CONTRACT.control_ids),
            "wald_hypothesis_ids": list(
                OPEN02_CONTRACT.holm.hypothesis_ids
            ),
            "covariance": {
                "estimator": OPEN02_CONTRACT.covariance.estimator,
                "kernel": OPEN02_CONTRACT.covariance.kernel,
                "lag_quarters": OPEN02_CONTRACT.covariance.lag_quarters,
                "prewhitening": OPEN02_CONTRACT.covariance.prewhitening,
                "finite_sample_correction": (
                    OPEN02_CONTRACT.covariance.finite_sample_correction
                ),
            },
        },
    }


def _write_invalid_result(
    *,
    root: Path,
    error: Open02ValidationError,
    receipt_common: Mapping[str, Any],
) -> Path:
    gate_rows = [
        {
            "gate_id": gate.gate_id,
            "reason_code": gate.reason_code,
            "passed": (
                False if gate.reason_code == error.reason_code else None
            ),
        }
        for gate in OPEN02_CONTRACT.validity_gates
    ]
    acceptance = {
        "schema_version": 1,
        "open_id": "OPEN-02",
        "status": "failed",
        "gates": gate_rows,
        "identity_errors": {},
        "row_hash": None,
        "column_hashes": {},
        "design_hashes": {},
        "wald_statistics": {},
        "raw_p_values": {},
        "holm_adjusted_p_values": {},
        "influence_maxima": {},
        "valid_result": (
            OPEN02_CONTRACT.invalid_result_disposition.valid_result
        ),
        "main_text_eligible": (
            OPEN02_CONTRACT.invalid_result_disposition.main_text_eligible
        ),
        "appendix_only": (
            OPEN02_CONTRACT.invalid_result_disposition.appendix_only
        ),
        "reason_codes": [error.reason_code],
        "failure": {
            "reason_code": error.reason_code,
            "message": str(error),
            "details": error.details,
        },
    }
    acceptance_path = _project_path(root, ACCEPTANCE_LOCATOR)
    _atomic_write_json(acceptance_path, acceptance)
    unwritten_outputs = {
        key: {
            "path": locator,
            "status": "not_written_invalid_result",
            "sha256": None,
            "bytes": 0,
        }
        for key, locator in (
            ("panel", PANEL_LOCATOR),
            ("response_system", RESPONSE_SYSTEM_LOCATOR),
            ("influence", INFLUENCE_LOCATOR),
        )
    }
    receipt = {
        **dict(receipt_common),
        "status": "failed",
        "scientific_status": "invalid_result",
        "completed_at_utc": _utc_now(),
        "outputs": {
            **unwritten_outputs,
            "acceptance": _file_record(root, ACCEPTANCE_LOCATOR),
        },
        "row_counts": {
            "panel_rows": 0,
            "estimate_rows": 0,
            "wald_rows": 0,
            "influence_rows": 0,
        },
        "acceptance": acceptance,
    }
    _require_receipt_contract(receipt)
    receipt_path = _project_path(root, RECEIPT_LOCATOR)
    _atomic_write_json(receipt_path, receipt)
    return receipt_path


def _input_validation_error(
    error: Exception,
    *,
    stage: str,
    fallback_reason_code: str,
) -> Open02ValidationError:
    reason_code = (
        error.reason_code
        if isinstance(error, Open02ValidationError)
        else fallback_reason_code
    )
    details: dict[str, Any] = {
        "stage": stage,
        "error_type": type(error).__name__,
        "cause": str(error),
    }
    if isinstance(error, Open02ValidationError):
        details["gate_details"] = error.details
    return Open02ValidationError(
        reason_code,
        f"OPEN-02 {stage} failed: {error}",
        details=details,
    )


def run_open02(
    *,
    producer_commit: str,
    run_id: str,
    root: Path = ROOT,
    source_bundle_locator: str = SOURCE_BUNDLE_LOCATOR,
    source_manifest_locator: str = SOURCE_MANIFEST_LOCATOR,
    invocation_argv: list[str] | None = None,
) -> Path:
    root = root.resolve()
    verified_commit = _verify_producer_commit(root, producer_commit)
    verified_run_id = _validate_run_id(run_id)
    _remove_previous_outputs(root)
    started_at = _utc_now()
    recorded_argv = (
        list(invocation_argv)
        if invocation_argv is not None
        else [
            str(sys.executable),
            "scripts/run_open02_producer.py",
            "--producer-commit",
            verified_commit,
            "--run-id",
            verified_run_id,
        ]
    )
    if not recorded_argv or any(not isinstance(item, str) for item in recorded_argv):
        raise ValueError("Producer argv must be a non-empty list of strings")
    source_evidence: dict[str, Any] = {
        "status": "not_loaded",
        "input_hashes": {},
    }
    accepted_inputs: dict[str, Any] = {
        "status": "not_loaded",
        "input_hashes": {},
    }
    try:
        source_bundle, source_evidence = _load_source_bundle(
            root=root,
            bundle_locator=source_bundle_locator,
            manifest_locator=source_manifest_locator,
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        failure = _input_validation_error(
            exc,
            stage="source_input_validation",
            fallback_reason_code="metadata_gate_failed",
        )
        receipt_common = _receipt_common_payload(
            verified_commit=verified_commit,
            verified_run_id=verified_run_id,
            recorded_argv=recorded_argv,
            started_at=started_at,
            source_evidence=source_evidence,
            accepted_inputs=accepted_inputs,
        )
        _write_invalid_result(
            root=root,
            error=failure,
            receipt_common=receipt_common,
        )
        raise failure from exc
    try:
        design_rows, standardized_rows, accepted_inputs = (
            _load_accepted_inputs(root)
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        failure = _input_validation_error(
            exc,
            stage="accepted_open01_input_validation",
            fallback_reason_code="common_sample_design_failed",
        )
        receipt_common = _receipt_common_payload(
            verified_commit=verified_commit,
            verified_run_id=verified_run_id,
            recorded_argv=recorded_argv,
            started_at=started_at,
            source_evidence=source_evidence,
            accepted_inputs=accepted_inputs,
        )
        _write_invalid_result(
            root=root,
            error=failure,
            receipt_common=receipt_common,
        )
        raise failure from exc
    receipt_common = _receipt_common_payload(
        verified_commit=verified_commit,
        verified_run_id=verified_run_id,
        recorded_argv=recorded_argv,
        started_at=started_at,
        source_evidence=source_evidence,
        accepted_inputs=accepted_inputs,
    )
    try:
        result = run_open02_pipeline(
            source_bundle,
            design_rows,
            standardized_rows,
            contract=OPEN02_CONTRACT,
        )
        output = _write_pipeline_outputs(root, result)
    except Open02ValidationError as exc:
        if _git_text(root, "rev-parse", "HEAD").lower() != verified_commit:
            raise ValueError(
                "Checkout HEAD changed during the OPEN-02 producer run"
            ) from exc
        _write_invalid_result(
            root=root,
            error=exc,
            receipt_common=receipt_common,
        )
        raise
    except ValueError as exc:
        failure = _input_validation_error(
            exc,
            stage="pipeline_output_validation",
            fallback_reason_code="common_sample_design_failed",
        )
        if _git_text(root, "rev-parse", "HEAD").lower() != verified_commit:
            raise ValueError(
                "Checkout HEAD changed during the OPEN-02 producer run"
            ) from exc
        _write_invalid_result(
            root=root,
            error=failure,
            receipt_common=receipt_common,
        )
        raise failure from exc
    if _git_text(root, "rev-parse", "HEAD").lower() != verified_commit:
        raise ValueError("Checkout HEAD changed during the OPEN-02 producer run")

    acceptance = output["acceptance"]
    eligibility = acceptance
    receipt = {
        **receipt_common,
        "status": "passed",
        "scientific_status": (
            "main_text_eligible"
            if eligibility["main_text_eligible"] is True
            else "appendix_only"
        ),
        "completed_at_utc": _utc_now(),
        "outputs": output["artifacts"],
        "row_counts": {
            key: output[key]
            for key in (
                "panel_rows",
                "estimate_rows",
                "wald_rows",
                "influence_rows",
            )
        },
        "acceptance": acceptance,
    }
    _require_receipt_contract(receipt)
    receipt_path = _project_path(root, RECEIPT_LOCATOR)
    _atomic_write_json(receipt_path, receipt)
    return receipt_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fail-closed OPEN-02 producer and write its receipt."
    )
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--source-bundle",
        default=SOURCE_BUNDLE_LOCATOR,
    )
    parser.add_argument(
        "--source-manifest",
        default=SOURCE_MANIFEST_LOCATOR,
    )
    parser.add_argument(
        "--acquire-source",
        action="store_true",
        help=(
            "Fetch and materialize the pinned official source bundle, then stop "
            "without running regressions."
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.acquire_source:
        receipt = acquire_open02_source(
            producer_commit=args.producer_commit,
            run_id=args.run_id,
            source_bundle_locator=args.source_bundle,
            source_manifest_locator=args.source_manifest,
            invocation_argv=list(sys.argv),
        )
        stage = "source_acquisition"
    else:
        receipt = run_open02(
            producer_commit=args.producer_commit,
            run_id=args.run_id,
            source_bundle_locator=args.source_bundle,
            source_manifest_locator=args.source_manifest,
            invocation_argv=list(sys.argv),
        )
        stage = "response_system"
    print(
        json.dumps(
            {
                "status": "passed",
                "stage": stage,
                "receipt": receipt.relative_to(ROOT).as_posix(),
                "sha256": _sha256_file(receipt),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
