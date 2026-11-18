from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _load_runner():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_open02_producer.py"
    )
    spec = importlib.util.spec_from_file_location("run_open02_producer", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_text(root: Path, locator: str, content: str) -> Path:
    path = root / locator
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_json(root: Path, locator: str, payload: dict[str, Any]) -> Path:
    return _write_text(
        root,
        locator,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _write_csv(
    root: Path,
    locator: str,
    rows: list[dict[str, Any]],
) -> Path:
    path = root / locator
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_inputs(runner, root: Path, monkeypatch) -> dict[str, Any]:
    source_bundle = {
        "schema_version": 1,
        "kind": "open02_financial_accounts_input",
        "accepted_tdcest_bundle_generated_at": "2026-05-22T16:56:46Z",
        "observation_vintage": "2026-03-19",
        "official_release": {
            "source_url": runner.OPEN02_BOARD_ARCHIVE_URL,
            "release_date": runner.OPEN02_BOARD_ARCHIVE_RELEASE_DATE,
            "archive_sha256": runner.OPEN02_BOARD_ARCHIVE_SHA256,
            "csv_member_sha256": dict(
                runner.OPEN02_BOARD_CSV_MEMBER_SHA256
            ),
            "dictionary_member_sha256": dict(
                runner.OPEN02_BOARD_DICTIONARY_MEMBER_SHA256
            ),
            "rows_sha256": "a" * 64,
        },
        "series": [{"key": "fixture", "observations": []}],
    }
    bundle_path = _write_json(
        root,
        runner.SOURCE_BUNDLE_LOCATOR,
        source_bundle,
    )
    _write_json(
        root,
        runner.SOURCE_MANIFEST_LOCATOR,
        {
            "schema_version": 1,
            "open_id": "OPEN-02",
            "status": "passed",
            "stage": "source_acquisition",
            "producer_commit": "1" * 40,
            "run_id": "source-fixture",
            "argv": ["scripts/run_open02_producer.py", "--acquire-source"],
            "started_at_utc": "2026-07-30T00:00:00+00:00",
            "completed_at_utc": "2026-07-30T00:00:00+00:00",
            "observation_vintage": "2026-03-19",
            "observation_vintage_cutoff": (
                runner.OPEN02_CONTRACT.sample.observation_vintage_cutoff
            ),
            "official_release": {
                "source_url": runner.OPEN02_BOARD_ARCHIVE_URL,
                "release_date": runner.OPEN02_BOARD_ARCHIVE_RELEASE_DATE,
                "archive_sha256": runner.OPEN02_BOARD_ARCHIVE_SHA256,
                "csv_member_sha256": dict(
                    runner.OPEN02_BOARD_CSV_MEMBER_SHA256
                ),
                "dictionary_member_sha256": dict(
                    runner.OPEN02_BOARD_DICTIONARY_MEMBER_SHA256
                ),
                "normalized_rows_sha256": "a" * 64,
            },
            "source_bundle": {
                "path": runner.SOURCE_BUNDLE_LOCATOR,
                "sha256": _sha256(bundle_path),
                "bytes": bundle_path.stat().st_size,
            },
            "metadata_gate": {
                "passed": True,
                "series_count": 20,
                "series_ids": [
                    series.board_series_id
                    for series in runner.OPEN02_CONTRACT.series
                ],
                "verified_from_bundled_data_dictionaries": True,
            },
            "coverage_gate": {
                "passed": True,
                "series_count": 20,
                "observations": 96,
                "sample_start": "2002Q1",
                "sample_end": "2025Q4",
            },
        },
    )
    receipt_path = _write_json(
        root,
        runner.OPEN01_RECEIPT_LOCATOR,
        {
            "open_id": "OPEN-01",
            "status": "passed",
            "producer_commit": runner.EXPECTED_OPEN01_PRODUCER_COMMIT,
            "run_id": "open01-fixture",
            "contract": {
                "treatment_id": (
                    runner.OPEN_CONTRACT.canonical_treatment_id
                )
            },
            "sample": {},
            "acceptance": {
                "status": "passed",
                "scientific_status": "unstable",
                "sample": {
                    "start": runner.OPEN02_CONTRACT.sample.start_quarter,
                    "end": runner.OPEN02_CONTRACT.sample.end_quarter,
                    "n": runner.OPEN02_CONTRACT.sample.observations,
                },
            },
        },
    )
    design_path = _write_csv(
        root,
        runner.ACCEPTED_DESIGN_LOCATOR,
        [{"quarter": "2002Q1", "fixture": "1"}],
    )
    standardized_path = _write_csv(
        root,
        runner.ACCEPTED_STANDARDIZED_LOCATOR,
        [{"series_id": "fixture", "period_end": "2002-03-31", "value": "1"}],
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_ACCEPTED_INPUT_SHA256",
        {
            runner.OPEN01_RECEIPT_LOCATOR: _sha256(receipt_path),
            runner.ACCEPTED_DESIGN_LOCATOR: _sha256(design_path),
            runner.ACCEPTED_STANDARDIZED_LOCATOR: _sha256(standardized_path),
        },
    )
    monkeypatch.setattr(
        runner,
        "_verify_committed_source_producer",
        lambda root, producer_commit: producer_commit,
    )
    return source_bundle


def _passing_result(runner) -> SimpleNamespace:
    quarters = runner._expected_sample_quarters()
    source_system, within_system = runner.OPEN02_CONTRACT.systems
    within_equation_count = len(within_system.outcome_ids)
    coefficient_ids = [
        *source_system.coefficient_ids,
        *(
            f"beta_{outcome_id}"
            for outcome_id in source_system.agency_component_outcome_ids
        ),
        *within_system.coefficient_ids,
        *(
            f"{prefix}_{outcome_id}"
            for prefix in (
                within_system.coefficient_ids[0].split("_", 1)[0],
                within_system.coefficient_ids[within_equation_count].split(
                    "_",
                    1,
                )[0],
            )
            for outcome_id in within_system.agency_component_outcome_ids
        ),
    ]
    influence_rows = []
    influence_summaries = []
    for group in runner.OPEN02_CONTRACT.influence.groups:
        for quarter in quarters:
            influence_rows.append(
                {
                    "group_id": group.group_id,
                    "deletion_kind": "leave_one",
                    "omitted_quarters": (quarter,),
                    "relative_l2_influence": 0.1,
                    "sign_flip_for_significant_coefficient": False,
                }
            )
        for start in range(
            runner.OPEN02_CONTRACT.influence.block_deletion_fits
        ):
            influence_rows.append(
                {
                    "group_id": group.group_id,
                    "deletion_kind": "leave_block",
                    "omitted_quarters": quarters[start : start + 4],
                    "relative_l2_influence": 0.2,
                    "sign_flip_for_significant_coefficient": False,
                }
            )
        influence_summaries.append(
            {
                "group_id": group.group_id,
                "leave_one_fits": 96,
                "leave_block_fits": 93,
                "maximum_leave_one_influence": 0.1,
                "maximum_leave_block_influence": 0.2,
                "leave_one_passed": True,
                "leave_block_passed": True,
                "sign_flip_detected": False,
                "passed": True,
                "reason_codes": (),
            }
        )
    return SimpleNamespace(
        panel_rows=tuple(
            {
                "quarter": quarter,
                "canonical": 3.0,
                "leave_out": 2.0,
                "bank_treasury": 1.0,
            }
            for quarter in quarters
        ),
        estimate_rows=tuple(
            {
                "system_id": "source_side_response",
                "outcome_id": "bank_treasury",
                "coefficient_id": coefficient_id,
                "estimate": 0.25,
                "standard_error": 0.10,
                "raw_p_value": 0.01,
            }
            for coefficient_id in coefficient_ids
        ),
        wald_rows=(
            {
                "hypothesis_id": "H_T",
                "statistic": 6.25,
                "degrees_of_freedom": 1,
                "raw_p_value": 0.01,
                "holm_adjusted_p_value": 0.03,
                "passed": True,
            },
            {
                "hypothesis_id": "H_P",
                "statistic": 9.0,
                "degrees_of_freedom": 3,
                "raw_p_value": 0.01,
                "holm_adjusted_p_value": 0.03,
                "passed": True,
            },
            {
                "hypothesis_id": "H_W",
                "statistic": 8.0,
                "degrees_of_freedom": 3,
                "raw_p_value": 0.02,
                "holm_adjusted_p_value": 0.03,
                "passed": True,
            },
        ),
        influence_rows=tuple(influence_rows),
        influence_summaries=tuple(influence_summaries),
        acceptance={
            "deterministic_gates": tuple(
                {
                    "gate_id": gate.gate_id,
                    "reason_code": gate.reason_code,
                    "passed": True,
                }
                for gate in runner.OPEN02_CONTRACT.validity_gates
            ),
            "identity_evidence": {
                "treasury_component": 0.0,
                "accepted_component_reconciliation": 0.0,
                "leave_out_reconstruction": 0.0,
                "us_agency_identity": 0.0,
                "three_sector_agency_identity": 0.0,
            },
            "design_evidence": {
                "row_hash": "a" * 64,
                "source": {
                    "column_hash": "b" * 64,
                    "design_hash": "c" * 64,
                },
                "within": {
                    "column_hash": "d" * 64,
                    "design_hash": "e" * 64,
                },
            },
            "valid_result": True,
            "main_text_eligible": True,
            "appendix_only": False,
            "reason_codes": [],
        },
    )


def _parsed_archive(runner) -> dict[str, Any]:
    quarters = [
        f"{ordinal // 4}Q{ordinal % 4 + 1}"
        for ordinal in range(2002 * 4, 2026 * 4)
    ]
    rows = []
    for row_index, quarter in enumerate(quarters):
        rows.append(
            {
                "quarter": quarter,
                **{
                    series.key: float(row_index + series_index)
                    for series_index, series in enumerate(
                        runner.OPEN02_CONTRACT.series
                    )
                },
            }
        )
    return {
        "metadata": {
            "kind": "open02_board_z1_archive",
            "source_url": runner.OPEN02_BOARD_ARCHIVE_URL,
            "release_date": runner.OPEN02_BOARD_ARCHIVE_RELEASE_DATE,
            "archive_sha256": runner.OPEN02_BOARD_ARCHIVE_SHA256,
            "csv_member_sha256": dict(
                runner.OPEN02_BOARD_CSV_MEMBER_SHA256
            ),
            "dictionary_member_sha256": dict(
                runner.OPEN02_BOARD_DICTIONARY_MEMBER_SHA256
            ),
            "rows_sha256": "d" * 64,
            "sample_start": "2002Q1",
            "sample_end": "2025Q4",
            "observations": 96,
            "series_count": len(runner.OPEN02_CONTRACT.series),
            "series": [
                {
                    "key": series.key,
                    "fred_id": series.fred_id,
                    "board_series_id": series.board_series_id,
                    "archive_member": "csv/fu111.csv",
                    "dictionary_member": "data_dictionary/fu111.txt",
                    "official_description": (
                        series.official_title.removesuffix(", Transactions")
                    ),
                    "table_line": "Line 1",
                    "table": "fixture",
                    "unit_label": (
                        "Millions of dollars; transactions, not seasonally adjusted"
                    ),
                    "side": series.side,
                    "units": series.units,
                    "seasonal_adjustment": series.seasonal_adjustment,
                }
                for series in runner.OPEN02_CONTRACT.series
            ],
        },
        "rows": rows,
    }


def test_open02_source_acquisition_materializes_one_frozen_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    commit = "1" * 40
    parsed = _parsed_archive(runner)
    monkeypatch.setattr(
        runner,
        "_verify_producer_commit",
        lambda root, producer_commit: commit,
    )
    monkeypatch.setattr(runner, "_git_text", lambda root, *args: commit)
    monkeypatch.setattr(
        runner,
        "fetch_open02_board_archive",
        lambda: parsed,
    )
    manifest_path = runner.acquire_open02_source(
        producer_commit=commit,
        run_id="open02-source-fixture",
        root=tmp_path,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle_path = tmp_path / runner.SOURCE_BUNDLE_LOCATOR
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "passed"
    assert manifest["producer_commit"] == commit
    assert manifest["observation_vintage"] == "2026-03-19"
    assert manifest["source_bundle"]["sha256"] == _sha256(bundle_path)
    assert manifest["metadata_gate"]["series_count"] == 20
    assert manifest["coverage_gate"]["observations"] == 96
    assert bundle["accepted_tdcest_bundle_generated_at"] == (
        "2026-05-22T16:56:46Z"
    )
    assert bundle["observation_vintage"] == "2026-03-19"
    assert len(bundle["series"]) == 20
    assert all(len(series["observations"]) == 96 for series in bundle["series"])


def test_source_acquisition_preserves_prior_pair_when_fetch_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    commit = "1" * 40
    bundle_path = _write_text(
        tmp_path,
        runner.SOURCE_BUNDLE_LOCATOR,
        "prior source bundle\n",
    )
    manifest_path = _write_text(
        tmp_path,
        runner.SOURCE_MANIFEST_LOCATOR,
        "prior source manifest\n",
    )
    prior_bundle = bundle_path.read_bytes()
    prior_manifest = manifest_path.read_bytes()
    monkeypatch.setattr(
        runner,
        "_verify_producer_commit",
        lambda root, producer_commit: commit,
    )

    def fail_fetch():
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(runner, "fetch_open02_board_archive", fail_fetch)
    with pytest.raises(RuntimeError, match="network unavailable"):
        runner.acquire_open02_source(
            producer_commit=commit,
            run_id="open02-source-failure",
            root=tmp_path,
        )

    assert bundle_path.read_bytes() == prior_bundle
    assert manifest_path.read_bytes() == prior_manifest


def test_source_acquisition_restores_prior_pair_when_manifest_write_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    commit = "1" * 40
    bundle_path = _write_text(
        tmp_path,
        runner.SOURCE_BUNDLE_LOCATOR,
        "prior source bundle\n",
    )
    manifest_path = _write_text(
        tmp_path,
        runner.SOURCE_MANIFEST_LOCATOR,
        "prior source manifest\n",
    )
    prior_bundle = bundle_path.read_bytes()
    prior_manifest = manifest_path.read_bytes()
    monkeypatch.setattr(
        runner,
        "_verify_producer_commit",
        lambda root, producer_commit: commit,
    )
    monkeypatch.setattr(runner, "_git_text", lambda root, *args: commit)
    monkeypatch.setattr(
        runner,
        "fetch_open02_board_archive",
        lambda: _parsed_archive(runner),
    )
    atomic_write = runner._atomic_write_bytes
    calls = 0

    def fail_second_write(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("manifest replace failed")
        atomic_write(path, payload)

    monkeypatch.setattr(runner, "_atomic_write_bytes", fail_second_write)
    with pytest.raises(OSError, match="manifest replace failed"):
        runner.acquire_open02_source(
            producer_commit=commit,
            run_id="open02-source-write-failure",
            root=tmp_path,
        )

    assert bundle_path.read_bytes() == prior_bundle
    assert manifest_path.read_bytes() == prior_manifest


def test_open02_producer_writes_outputs_and_receipt_last(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    source_bundle = _prepare_inputs(runner, tmp_path, monkeypatch)
    commit = "2" * 40
    monkeypatch.setattr(
        runner,
        "_verify_producer_commit",
        lambda root, producer_commit: commit,
    )
    monkeypatch.setattr(runner, "_git_text", lambda root, *args: commit)
    calls: list[tuple[Any, ...]] = []

    def fake_pipeline(bundle, design, standardized, *, contract):
        calls.append((bundle, design, standardized, contract))
        return _passing_result(runner)

    monkeypatch.setattr(runner, "run_open02_pipeline", fake_pipeline)
    invocation = [
        "scripts/run_open02_producer.py",
        "--producer-commit",
        commit,
        "--run-id",
        "open02-fixture",
    ]
    receipt_path = runner.run_open02(
        producer_commit=commit,
        run_id="open02-fixture",
        root=tmp_path,
        invocation_argv=invocation,
    )

    assert len(calls) == 1
    assert calls[0][0] == source_bundle
    assert calls[0][3] is runner.OPEN02_CONTRACT
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["open_id"] == "OPEN-02"
    assert receipt["status"] == "passed"
    assert receipt["scientific_status"] == "main_text_eligible"
    assert receipt["producer_commit"] == commit
    assert receipt["argv"] == invocation
    assert receipt["row_counts"] == {
        "panel_rows": 96,
        "estimate_rows": 31,
        "wald_rows": 3,
        "influence_rows": 567,
    }
    assert receipt["source"]["bundle"]["sha256"]
    assert receipt["accepted_open01_inputs"]["combined_sha256"]
    assert set(receipt["outputs"]) == {
        "panel",
        "response_system",
        "influence",
        "acceptance",
    }
    for record in receipt["outputs"].values():
        path = tmp_path / record["path"]
        assert path.is_file()
        assert _sha256(path) == record["sha256"]
    with (tmp_path / runner.RESPONSE_SYSTEM_LOCATOR).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        response_rows = list(csv.DictReader(handle))
    record_types = [row["record_type"] for row in response_rows]
    assert record_types.count("coefficient") == 31
    assert record_types.count("wald") == 3


def test_open02_writer_rejects_incomplete_sample_and_influence(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    incomplete_panel = _passing_result(runner)
    incomplete_panel.panel_rows = incomplete_panel.panel_rows[:-1]
    with pytest.raises(ValueError, match="exact frozen 96-quarter panel"):
        runner._write_pipeline_outputs(tmp_path, incomplete_panel)
    assert not (tmp_path / runner.RESPONSE_SYSTEM_LOCATOR).exists()

    incomplete_influence = _passing_result(runner)
    incomplete_influence.influence_rows = (
        incomplete_influence.influence_rows[:-1]
    )
    with pytest.raises(ValueError, match="number of influence refits"):
        runner._write_pipeline_outputs(tmp_path, incomplete_influence)
    assert not (tmp_path / runner.RESPONSE_SYSTEM_LOCATOR).exists()


def test_open02_writer_rejects_false_gate_and_duplicate_evidence(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    identity_failure = _passing_result(runner)
    identity_failure.acceptance["identity_evidence"][
        "treasury_component"
    ] = 1.0
    with pytest.raises(ValueError, match="identity evidence exceeds tolerance"):
        runner._write_pipeline_outputs(tmp_path, identity_failure)

    influence_failure = _passing_result(runner)
    influence_failure.influence_summaries[0][
        "maximum_leave_one_influence"
    ] = 0.30
    with pytest.raises(ValueError, match="influence disposition is inconsistent"):
        runner._write_pipeline_outputs(tmp_path, influence_failure)

    duplicate_wald = _passing_result(runner)
    duplicate_wald.wald_rows = (
        *duplicate_wald.wald_rows,
        duplicate_wald.wald_rows[0],
    )
    with pytest.raises(ValueError, match="wrong Wald family"):
        runner._write_pipeline_outputs(tmp_path, duplicate_wald)


def test_open02_producer_records_invalid_disposition_after_gate_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    _prepare_inputs(runner, tmp_path, monkeypatch)
    stale = _write_json(
        tmp_path,
        runner.RECEIPT_LOCATOR,
        {"status": "passed", "run_id": "stale"},
    )
    commit = "3" * 40
    monkeypatch.setattr(
        runner,
        "_verify_producer_commit",
        lambda root, producer_commit: commit,
    )
    monkeypatch.setattr(runner, "_git_text", lambda root, *args: commit)

    def fail_pipeline(*args, **kwargs):
        raise runner.Open02ValidationError(
            "coverage_gate_failed",
            "deterministic gate failed",
            details={"observations": 95},
        )

    monkeypatch.setattr(runner, "run_open02_pipeline", fail_pipeline)
    with pytest.raises(
        runner.Open02ValidationError,
        match="deterministic gate failed",
    ):
        runner.run_open02(
            producer_commit=commit,
            run_id="open02-failure",
            root=tmp_path,
        )
    acceptance = json.loads(
        (tmp_path / runner.ACCEPTANCE_LOCATOR).read_text(encoding="utf-8")
    )
    receipt = json.loads(stale.read_text(encoding="utf-8"))
    assert acceptance["status"] == "failed"
    assert acceptance["valid_result"] is False
    assert acceptance["main_text_eligible"] is False
    assert acceptance["appendix_only"] is False
    assert acceptance["reason_codes"] == ["coverage_gate_failed"]
    assert acceptance["failure"]["details"] == {"observations": 95}
    assert receipt["status"] == "failed"
    assert receipt["scientific_status"] == "invalid_result"
    assert receipt["run_id"] == "open02-failure"
    assert receipt["outputs"]["response_system"]["sha256"] is None
    assert not (tmp_path / runner.RESPONSE_SYSTEM_LOCATOR).exists()


def test_source_bundle_hash_mismatch_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    _prepare_inputs(runner, tmp_path, monkeypatch)
    bundle_path = tmp_path / runner.SOURCE_BUNDLE_LOCATOR
    bundle_path.write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bundle (byte count|hash)"):
        runner._load_source_bundle(
            root=tmp_path,
            bundle_locator=runner.SOURCE_BUNDLE_LOCATOR,
            manifest_locator=runner.SOURCE_MANIFEST_LOCATOR,
        )
    commit = "6" * 40
    monkeypatch.setattr(
        runner,
        "_verify_producer_commit",
        lambda root, producer_commit: commit,
    )
    with pytest.raises(
        runner.Open02ValidationError,
        match="source_input_validation failed",
    ):
        runner.run_open02(
            producer_commit=commit,
            run_id="open02-source-invalid",
            root=tmp_path,
        )
    receipt = json.loads(
        (tmp_path / runner.RECEIPT_LOCATOR).read_text(encoding="utf-8")
    )
    assert receipt["status"] == "failed"
    assert receipt["acceptance"]["valid_result"] is False
    assert receipt["acceptance"]["main_text_eligible"] is False
    assert receipt["acceptance"]["appendix_only"] is False
    assert receipt["acceptance"]["reason_codes"] == ["metadata_gate_failed"]


def test_accepted_input_hash_failure_records_invalid_disposition(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    _prepare_inputs(runner, tmp_path, monkeypatch)
    design_path = tmp_path / runner.ACCEPTED_DESIGN_LOCATOR
    design_path.write_text("tampered\n", encoding="utf-8")
    commit = "7" * 40
    monkeypatch.setattr(
        runner,
        "_verify_producer_commit",
        lambda root, producer_commit: commit,
    )

    with pytest.raises(
        runner.Open02ValidationError,
        match="accepted_open01_input_validation failed",
    ):
        runner.run_open02(
            producer_commit=commit,
            run_id="open02-accepted-input-invalid",
            root=tmp_path,
        )

    receipt = json.loads(
        (tmp_path / runner.RECEIPT_LOCATOR).read_text(encoding="utf-8")
    )
    assert receipt["status"] == "failed"
    assert receipt["source"]["input_hashes"]
    assert receipt["acceptance"]["valid_result"] is False
    assert receipt["acceptance"]["main_text_eligible"] is False
    assert receipt["acceptance"]["appendix_only"] is False
    assert receipt["acceptance"]["reason_codes"] == [
        "common_sample_design_failed"
    ]


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    (
        ("vintage", "vintage_gate_failed"),
        ("coverage", "coverage_gate_failed"),
        ("argv", "metadata_gate_failed"),
    ),
)
def test_source_manifest_failures_keep_exact_gate_reason(
    mutation: str,
    reason_code: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    _prepare_inputs(runner, tmp_path, monkeypatch)
    manifest_path = tmp_path / runner.SOURCE_MANIFEST_LOCATOR
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "vintage":
        manifest["observation_vintage"] = "2026-03-18"
    elif mutation == "coverage":
        manifest["coverage_gate"]["observations"] = 95
    else:
        manifest["argv"] = []
    _write_json(tmp_path, runner.SOURCE_MANIFEST_LOCATOR, manifest)
    commit = "8" * 40
    monkeypatch.setattr(
        runner,
        "_verify_producer_commit",
        lambda root, producer_commit: commit,
    )

    with pytest.raises(runner.Open02ValidationError) as caught:
        runner.run_open02(
            producer_commit=commit,
            run_id=f"open02-source-{mutation}-invalid",
            root=tmp_path,
        )
    assert caught.value.reason_code == reason_code
    receipt = json.loads(
        (tmp_path / runner.RECEIPT_LOCATOR).read_text(encoding="utf-8")
    )
    assert receipt["acceptance"]["reason_codes"] == [reason_code]


def test_producer_commit_verification_rejects_dirty_checkout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    commit = "4" * 40

    def git_text(root: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return commit
        if args == (
            "status",
            "--porcelain",
            "--untracked-files=all",
        ):
            return "?? scripts/uncommitted_open02.py"
        raise AssertionError(args)

    monkeypatch.setattr(runner, "_git_text", git_text)
    with pytest.raises(ValueError, match="untracked nonignored"):
        runner._verify_producer_commit(tmp_path, commit)


def test_source_manifest_commit_must_be_a_committed_ancestor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    commit = "5" * 40

    def git_text(root: Path, *args: str) -> str:
        if args == (
            "rev-parse",
            "--verify",
            f"{commit}^{{commit}}",
        ):
            raise runner.subprocess.CalledProcessError(128, ["git", *args])
        raise AssertionError(args)

    monkeypatch.setattr(runner, "_git_text", git_text)
    with pytest.raises(ValueError, match="not a committed ancestor"):
        runner._verify_committed_source_producer(tmp_path, commit)


def test_source_manifest_commit_must_contain_the_producer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    commit = "9" * 40

    def git_text(root: Path, *args: str) -> str:
        if args == (
            "rev-parse",
            "--verify",
            f"{commit}^{{commit}}",
        ):
            return commit
        if args == ("merge-base", "--is-ancestor", commit, "HEAD"):
            return ""
        if args[:2] == ("cat-file", "-e"):
            raise runner.subprocess.CalledProcessError(128, ["git", *args])
        raise AssertionError(args)

    monkeypatch.setattr(runner, "_git_text", git_text)
    with pytest.raises(ValueError, match="required producer files"):
        runner._verify_committed_source_producer(tmp_path, commit)


# ----------------------------------------------- corrected-input variant (v1 TDC correction)

VARIANT_COMMIT = "c" * 40
VARIANT_GENERATED = "2026-11-18T02:21:54Z"
VARIANT_CONFIG = "config/open02_variant.json"
ADMITTED_OPEN01 = "2e35658c055f7686f621618af14f9d5a9d8b35c3"


def _load_variant_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_open02_variant.py"
    spec = importlib.util.spec_from_file_location("run_open02_variant", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _open01_receipt(producer, bundle: Path, standardized_sha: str, **changes) -> dict:
    return {"open_id": "OPEN-01", "status": "passed", "producer_commit": VARIANT_COMMIT, "run_id": "open01-corrected",
            "tdcest": {"bundle_hash": _sha256(bundle), "resolved_source_bundle": str(bundle),
                       "standardized": {"sha256": standardized_sha}},
            "contract": {"treatment_id": producer.OPEN_CONTRACT.canonical_treatment_id}, "sample": {},
            "acceptance": {"status": "passed", "scientific_status": "unstable",
                           "sample": {"start": "2002Q1", "end": "2025Q4", "n": 96}}, **changes}


def _variant_tree(wrapper, root: Path, monkeypatch, *, receipt_changes: dict | None = None,
                  bundle_generated: str = "2026-11-18T02:21:54+00:00", **changes) -> Path:
    """An admitted source bundle, a corrected OPEN-01 run and its TDCest bundle, and a variant config."""
    from ea_tdc import open02

    producer = wrapper.producer
    parsed = _parsed_archive(producer)
    parsed["metadata"]["observation_vintage_cutoff"] = producer.OPEN02_CONTRACT.sample.observation_vintage_cutoff
    parsed["metadata"]["rows_sha256"] = open02._stable_hash(parsed["rows"])
    admitted = root / producer.SOURCE_BUNDLE_LOCATOR
    admitted.parent.mkdir(parents=True, exist_ok=True)
    admitted.write_bytes(producer._json_bytes(producer._pipeline_source_bundle(parsed)))
    pin = {"path": producer.SOURCE_BUNDLE_LOCATOR, "sha256": _sha256(admitted), "bytes": admitted.stat().st_size}
    monkeypatch.setattr(wrapper, "ADMITTED_SOURCE_BUNDLE", pin)
    tdcest_bundle = _write_json(root, "tdcest/site/data/bundle.json", {"generated_at_utc": bundle_generated})
    design = _write_csv(root, producer.ACCEPTED_DESIGN_LOCATOR, [{"quarter": "2002Q1", "fixture": "2"}])
    standardized = _write_csv(root, producer.ACCEPTED_STANDARDIZED_LOCATOR,
                              [{"series_id": "fixture", "period_end": "2002-03-31", "value": "2"}])
    receipt_path = _write_json(root, producer.OPEN01_RECEIPT_LOCATOR, _open01_receipt(
        producer, tdcest_bundle, _sha256(standardized), **(receipt_changes or {})))
    config = {
        "schema_version": wrapper.SCHEMA,
        "variant": "corrected_fixture",
        "tdcest_bundle": {"sha256": _sha256(tdcest_bundle), "generated_at": VARIANT_GENERATED},
        "observation_vintage_cutoff": VARIANT_GENERATED,
        "open01_producer_commit": VARIANT_COMMIT,
        "accepted_inputs": {producer.OPEN01_RECEIPT_LOCATOR: _sha256(receipt_path),
                            producer.ACCEPTED_DESIGN_LOCATOR: _sha256(design),
                            producer.ACCEPTED_STANDARDIZED_LOCATOR: _sha256(standardized)},
        "source": {"bundle": "data/bundles/open02/variant_source.json",
                   "manifest": "output/manifests/open02_variant_source_manifest.json",
                   "derived_from": {"bundle": dict(pin)}},
        **changes,
    }
    _write_json(root, VARIANT_CONFIG, config)
    monkeypatch.setattr(producer, "_verify_producer_commit", lambda root, producer_commit: "e" * 40)
    monkeypatch.setattr(producer, "_git_text", lambda root, *args: "e" * 40)
    monkeypatch.setattr(producer, "_verify_committed_source_producer", lambda root, producer_commit: producer_commit)

    def no_fetch():
        raise AssertionError("a corrected-input variant must not refetch the Board archive")

    monkeypatch.setattr(producer, "fetch_open02_board_archive", no_fetch)
    return admitted


def _strip_stamps(wrapper, bundle: dict) -> dict:
    bundle = json.loads(json.dumps(bundle))
    for field in wrapper.STAMP_FIELDS:
        bundle.pop(field)
    bundle["official_release"].pop("observation_vintage_cutoff")
    return bundle


def _derived(wrapper, root: Path, monkeypatch, **kwargs):
    _variant_tree(wrapper, root, monkeypatch, **kwargs)
    variant = wrapper.load_variant(root, VARIANT_CONFIG)
    manifest = wrapper.derive_source(producer_commit="e" * 40, run_id="derive-fixture", variant=variant, root=root)
    return variant, manifest


def test_committed_v1_variant_pins_the_admitted_source_and_a_later_tdcest_pairing() -> None:
    wrapper = _load_variant_runner()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "config/open02_v1_tfx.json").read_text())
    assert config["source"]["derived_from"]["bundle"] == wrapper.ADMITTED_SOURCE_BUNDLE
    assert config["observation_vintage_cutoff"] == config["tdcest_bundle"]["generated_at"] > "2026-05-22T16:56:46Z"
    assert set(config["accepted_inputs"]) == set(wrapper.producer.EXPECTED_ACCEPTED_INPUT_SHA256)
    assert config["open01_producer_commit"] != wrapper.producer.EXPECTED_OPEN01_PRODUCER_COMMIT


def test_variant_changes_only_the_tdcest_pairing(tmp_path: Path, monkeypatch) -> None:
    wrapper = _load_variant_runner()
    _variant_tree(wrapper, tmp_path, monkeypatch)
    variant = wrapper.load_variant(tmp_path, VARIANT_CONFIG)
    admitted = wrapper.producer.OPEN02_CONTRACT
    assert (variant.contract.sample.accepted_tdcest_bundle_generated_at,
            variant.contract.sample.observation_vintage_cutoff) == (VARIANT_GENERATED, VARIANT_GENERATED)
    # Exactly the two stamps differ; the sample window, hash, counts, series and systems do not.
    assert wrapper.replace(variant.contract.sample,
                           accepted_tdcest_bundle_generated_at=admitted.sample.accepted_tdcest_bundle_generated_at,
                           observation_vintage_cutoff=admitted.sample.observation_vintage_cutoff) == admitted.sample
    assert wrapper.replace(variant.contract, sample=admitted.sample) == admitted
    assert wrapper.producer.EXPECTED_OPEN01_PRODUCER_COMMIT == ADMITTED_OPEN01
    assert variant.config["sha256"] == _sha256(tmp_path / VARIANT_CONFIG)


@pytest.mark.parametrize("change, message", [
    ({"source": {"bundle": "data/bundles/open02/financial_accounts_2026-03-19.json", "manifest": "x.json",
                 "derived_from": {}}}, "would overwrite the admitted source"),
    ({"source": {"bundle": "output/reports/tier2_bank_portfolio_response_system.csv", "manifest": "x.json",
                 "derived_from": {}}}, "or an output"),
    ({"observation_vintage_cutoff": "2026-05-22T16:56:46Z"}, "precedes its TDCest bundle"),
    ({"tdcest_bundle": {"sha256": "b" * 64, "generated_at": "2026-03-01T00:00:00Z"},
      "observation_vintage_cutoff": "2026-03-18T00:00:00Z"}, "precedes its TDCest bundle or the Board release"),
    ({"accepted_inputs": {}}, "malformed"),
    ({"open01_producer_commit": "2e35658"}, "full 40-character commit"),
])
def test_malformed_variant_fails_closed(tmp_path: Path, monkeypatch, change, message) -> None:
    wrapper = _load_variant_runner()
    _variant_tree(wrapper, tmp_path, monkeypatch, **change)
    if "source" in change:
        config = json.loads((tmp_path / VARIANT_CONFIG).read_text())
        config["source"]["derived_from"] = {"bundle": dict(wrapper.ADMITTED_SOURCE_BUNDLE)}
        _write_json(tmp_path, VARIANT_CONFIG, config)
    with pytest.raises(ValueError, match=message):
        wrapper.load_variant(tmp_path, VARIANT_CONFIG)


def test_variant_must_derive_from_the_pinned_admitted_bundle(tmp_path: Path, monkeypatch) -> None:
    wrapper = _load_variant_runner()
    _variant_tree(wrapper, tmp_path, monkeypatch)
    config = json.loads((tmp_path / VARIANT_CONFIG).read_text())
    config["source"]["derived_from"]["bundle"]["sha256"] = "a" * 64
    _write_json(tmp_path, VARIANT_CONFIG, config)
    with pytest.raises(ValueError, match="derives from another bundle"):
        wrapper.load_variant(tmp_path, VARIANT_CONFIG)


def test_variant_configuration_must_be_committed(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    wrapper = _load_variant_runner()
    real_git_text = wrapper.producer._git_text
    _variant_tree(wrapper, tmp_path, monkeypatch)
    monkeypatch.setattr(wrapper.producer, "_git_text", real_git_text)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    with pytest.raises(ValueError, match="not a committed file"):
        wrapper.load_variant(tmp_path, VARIANT_CONFIG)
    subprocess.run(["git", "-C", str(tmp_path), "add", VARIANT_CONFIG], check=True)
    assert wrapper.load_variant(tmp_path, VARIANT_CONFIG).name == "corrected_fixture"


def test_variant_source_is_derived_from_the_admitted_bundle_without_refetching(tmp_path: Path, monkeypatch) -> None:
    from ea_tdc import open02

    wrapper = _load_variant_runner()
    variant, manifest_path = _derived(wrapper, tmp_path, monkeypatch)
    manifest = json.loads(manifest_path.read_text())
    admitted = tmp_path / wrapper.producer.SOURCE_BUNDLE_LOCATOR
    variant_path = tmp_path / variant.source_bundle_locator
    bundle = json.loads(variant_path.read_text())
    assert _strip_stamps(wrapper, json.loads(admitted.read_text())) == _strip_stamps(wrapper, bundle)
    assert bundle["accepted_tdcest_bundle_generated_at"] == VARIANT_GENERATED
    assert manifest["derived_from"] == {"bundle": dict(variant.derived_from)}
    assert manifest["source_bundle"]["sha256"] == _sha256(variant_path)
    # The real OPEN-02 normalization accepts the variant bundle under the variant contract only.
    quarters = open02._expected_quarters(variant.contract)
    open02._normalize_source_bundle(bundle, contract=variant.contract, quarters=quarters)
    with pytest.raises(open02.Open02ValidationError, match="does not pin the accepted TDCest generation timestamp"):
        open02._normalize_source_bundle(bundle, contract=wrapper.producer.OPEN02_CONTRACT, quarters=quarters)
    with pytest.raises(ValueError, match="no overwrite"):
        wrapper.derive_source(producer_commit="e" * 40, run_id="again", variant=variant, root=tmp_path)
    with pytest.raises(wrapper.producer.Open02ValidationError):
        wrapper.producer._load_source_bundle(root=tmp_path, bundle_locator=variant.source_bundle_locator,
                                             manifest_locator=variant.source_manifest_locator)


def test_variant_command_line_derives_then_runs_and_restores_the_producer(tmp_path: Path, monkeypatch) -> None:
    wrapper = _load_variant_runner()
    _variant_tree(wrapper, tmp_path, monkeypatch)
    monkeypatch.setattr(wrapper, "ROOT", tmp_path)
    calls = []

    def fake_pipeline(bundle, design, standardized, *, contract):
        calls.append((bundle["accepted_tdcest_bundle_generated_at"], contract.sample.observation_vintage_cutoff))
        return _passing_result(wrapper.producer)

    monkeypatch.setattr(wrapper.producer, "run_open02_pipeline", fake_pipeline)
    common = ["--inputs", VARIANT_CONFIG, "--producer-commit", "e" * 40]
    wrapper.main([*common, "--run-id", "derive-cli", "--derive-source"])
    wrapper.main([*common, "--run-id", "open02-cli"])
    assert calls == [(VARIANT_GENERATED, VARIANT_GENERATED)]
    assert wrapper.producer.OPEN02_CONTRACT is wrapper.OPEN02_CONTRACT
    assert wrapper.producer.EXPECTED_OPEN01_PRODUCER_COMMIT == ADMITTED_OPEN01
    assert wrapper.producer.EXPECTED_ACCEPTED_INPUT_SHA256[wrapper.producer.OPEN01_RECEIPT_LOCATOR] == (
        "1ee790a0ddaddd8dd06da0236eabd48c7e035354105cf9595c62342470dd303d")
    receipt = json.loads((tmp_path / wrapper.RECEIPT_LOCATOR).read_text())
    producer_receipt = json.loads((tmp_path / wrapper.producer.RECEIPT_LOCATOR).read_text())
    assert producer_receipt["status"] == "passed" and producer_receipt["argv"][-2:] == ["--run-id", "open02-cli"]
    assert producer_receipt["source"]["bundle"]["path"] == "data/bundles/open02/variant_source.json"
    assert receipt["producer_receipt"]["sha256"] == _sha256(tmp_path / wrapper.producer.RECEIPT_LOCATOR)
    assert (receipt["variant"], receipt["open01_producer_commit"], receipt["observation_vintage_cutoff"]) == (
        "corrected_fixture", VARIANT_COMMIT, VARIANT_GENERATED)
    assert not (tmp_path / wrapper.producer.SOURCE_MANIFEST_LOCATOR).exists()


def test_variant_rebinding_does_not_nest(tmp_path: Path, monkeypatch) -> None:
    wrapper = _load_variant_runner()
    _variant_tree(wrapper, tmp_path, monkeypatch)
    variant = wrapper.load_variant(tmp_path, VARIANT_CONFIG)
    with wrapper._rebound(variant):
        assert wrapper.producer.OPEN02_CONTRACT is variant.contract
        with pytest.raises(RuntimeError, match="refusing to nest"):
            with wrapper._rebound(variant):
                pass
    assert wrapper.producer.OPEN02_CONTRACT is wrapper.OPEN02_CONTRACT


@pytest.mark.parametrize("case", ["financial_accounts_edited", "derivation_unrecorded", "other_tdcest_bundle",
                                  "bundle_generated_elsewhen", "admitted_open01_run", "outputs_present"])
def test_variant_run_fails_closed(tmp_path: Path, monkeypatch, case) -> None:
    wrapper = _load_variant_runner()
    receipt_changes = {"admitted_open01_run": {"producer_commit": ADMITTED_OPEN01}}.get(case)
    generated = "2026-11-17T00:00:00+00:00" if case == "bundle_generated_elsewhen" else "2026-11-18T02:21:54+00:00"
    variant, manifest_path = _derived(wrapper, tmp_path, monkeypatch, receipt_changes=receipt_changes,
                                      bundle_generated=generated)
    monkeypatch.setattr(wrapper.producer, "run_open02_pipeline", lambda *a, **k: _passing_result(wrapper.producer))
    error, message = ValueError, None
    if case == "financial_accounts_edited":
        path = tmp_path / variant.source_bundle_locator
        bundle = json.loads(path.read_text())
        bundle["series"][0]["observations"][40]["value"] += 1.0
        path.write_bytes(wrapper.producer._json_bytes(bundle))
        manifest = json.loads(manifest_path.read_text())
        manifest["source_bundle"].update(sha256=_sha256(path), bytes=path.stat().st_size)
        manifest_path.write_text(json.dumps(manifest))
        message = "admitted Financial Accounts content"
    elif case == "derivation_unrecorded":
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("derived_from")
        manifest_path.write_text(json.dumps(manifest))
        message = "does not record the declared derivation"
    elif case == "other_tdcest_bundle":
        receipt_path = tmp_path / wrapper.producer.OPEN01_RECEIPT_LOCATOR
        receipt = json.loads(receipt_path.read_text())
        receipt["tdcest"]["standardized"]["sha256"] = "a" * 64
        receipt_path.write_text(json.dumps(receipt))
        message = "another standardized series"
    elif case == "bundle_generated_elsewhen":
        message = "generated at another time"
    elif case == "admitted_open01_run":
        error, message = wrapper.producer.Open02ValidationError, "accepted_open01_input_validation failed"
    else:
        _write_text(tmp_path, wrapper.producer.RECEIPT_LOCATOR, "admitted receipt\n")
        message = "outputs already exist"
    with pytest.raises(error, match=message):
        wrapper.run_variant(producer_commit="e" * 40, run_id="open02-variant", variant=variant, root=tmp_path)
    assert not (tmp_path / wrapper.RECEIPT_LOCATOR).exists()
    assert wrapper.producer.OPEN02_CONTRACT is wrapper.OPEN02_CONTRACT
    if case == "outputs_present":
        assert (tmp_path / wrapper.producer.RECEIPT_LOCATOR).read_text() == "admitted receipt\n"
