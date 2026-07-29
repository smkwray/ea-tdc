from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest


def _load_runner():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_open01_producer.py"
    )
    spec = importlib.util.spec_from_file_location("run_open01_producer", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_text(root: Path, locator: str, content: str = "artifact\n") -> Path:
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
    assert rows
    path = root / locator
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_preflight(runner, root: Path) -> Path:
    for locator in runner.PIPELINE_SCRIPT_LOCATORS:
        _write_text(root, locator, "# fixture stage\n")
    return _write_text(root, "tdcest/bundle.json", '{"fixture": true}\n')


def _fixture_materializer(runner, root: Path, bundle_path: Path):
    controls = tuple(runner.CANONICAL_CONTROL_IDS)
    rejected = ("tier2_regression_bank_row_tier_pre_component_h15_scaled",)
    used = tuple(control for control in controls if control not in rejected)
    used_csv = ",".join(used)
    rejected_csv = ",".join(rejected)

    def write_source() -> None:
        standardized = _write_csv(
            root,
            runner.STANDARDIZED_LOCATOR,
            [
                {
                    "series_id": runner.CANONICAL_TREATMENT_SOURCE_SERIES,
                    "period_end": "2002-03-31",
                    "value": "1.0",
                },
                {
                    "series_id": runner.CANONICAL_TREATMENT_SOURCE_SERIES,
                    "period_end": "2025-12-31",
                    "value": "2.0",
                },
            ],
        )
        assert standardized.is_file()
        input_hashes = {
            "seed_bundle": _sha256(bundle_path),
            "processed_estimates": hashlib.sha256(b"processed").hexdigest(),
            "regression_series": hashlib.sha256(b"regression").hexdigest(),
        }
        combined = hashlib.sha256(
            json.dumps(
                input_hashes,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        _write_json(
            root,
            runner.SOURCE_MANIFEST_LOCATOR,
            {
                "kind": "source_manifest",
                "source_repo": "tdcest",
                "adapter": "tdcest_bundle",
                "bundle_hash": _sha256(bundle_path),
                "input_hashes": input_hashes,
                "combined_input_hash": combined,
                "rows_written": 2,
            },
        )

    def write_design(job_id: str) -> None:
        _write_csv(
            root,
            f"data/bundles/designs/{job_id}__quarterly_bundle.csv",
            [{"quarter": "2020Q1", runner.CANONICAL_TREATMENT_ID: "1.0"}],
        )
        _write_json(
            root,
            f"output/manifests/{job_id}__design_manifest.json",
            {
                "job_id": job_id,
                "status": "ready_for_estimation",
                "treatment_id": runner.CANONICAL_TREATMENT_ID,
                "outcome_ids": [
                    runner.CANONICAL_OUTCOME_ID,
                    runner.CANONICAL_RESIDUAL_ID,
                ],
                "control_ids": list(controls),
                "sample_start": "2002Q1",
                "sample_end": "2025Q4",
                "usable_rows": 96,
            },
        )
        _write_json(
            root,
            f"output/manifests/{job_id}__sample_manifest.json",
            {
                "job_id": job_id,
                "sample_start": "2002Q1",
                "sample_end": "2025Q4",
                "usable_rows": 96,
            },
        )

    def write_submission() -> None:
        fields = {
            "lead_placebo_csv": (
                "output/reports/submission_lead_placebo_coefficients.csv"
            ),
            "hac_csv": "output/reports/submission_hac_bandwidth_sensitivity.csv",
            "factor_tail_csv": (
                "output/reports/submission_factor_tail_robustness.csv"
            ),
            "splice_csv": "output/reports/submission_splice_construction_audit.csv",
            "plumbing_csv": "output/reports/submission_plumbing_magnitudes.csv",
        }
        for locator in fields.values():
            _write_csv(root, locator, [{"status": "computed"}])
        _write_json(
            root,
            runner.SUBMISSION_MANIFEST_LOCATOR,
            {
                "generated_at": "2026-07-29T00:00:00+00:00",
                **fields,
            },
        )

    def write_rolling() -> None:
        regression_rows = [
            {
                "window_start_quarter": "2010Q1",
                "window_end_quarter": "2021Q4",
                "effective_sample_start": "2010Q1",
                "effective_sample_end": "2021Q4",
                "window_quarters": runner.ROLLING_WINDOW_QUARTERS,
                "outcome": outcome,
                "treatment_id": runner.CANONICAL_TREATMENT_ID,
                "control_ids_used": used_csv,
                "dropped_control_ids": rejected_csv,
            }
            for outcome in (
                runner.CANONICAL_OUTCOME_ID,
                runner.CANONICAL_RESIDUAL_ID,
            )
        ]
        _write_csv(root, runner.ROLLING_ESTIMATES_LOCATOR, regression_rows)
        correlation_locator = (
            "output/reports/"
            "tier2_rolling_selected_credit_rate_pass_through_correlations.csv"
        )
        report_locator = (
            "output/reports/"
            "tier2_rolling_selected_credit_rate_pass_through.md"
        )
        _write_csv(root, correlation_locator, [{"status": "computed"}])
        _write_text(root, report_locator, "# Rolling fixture\n")
        bounds = {
            outcome: {
                "regression": {"start": "2010Q1", "end": "2021Q4"},
                "correlation": {"start": "2010Q1", "end": "2021Q4"},
            }
            for outcome in (
                runner.CANONICAL_OUTCOME_ID,
                runner.CANONICAL_RESIDUAL_ID,
            )
        }
        _write_json(
            root,
            runner.ROLLING_MANIFEST_LOCATOR,
            {
                "job_id": "tier2_rolling_selected_credit_rate_pass_through",
                "treatment_id": runner.CANONICAL_TREATMENT_ID,
                "outcome_ids": [
                    runner.CANONICAL_OUTCOME_ID,
                    runner.CANONICAL_RESIDUAL_ID,
                ],
                "control_ids": list(controls),
                "window_quarters": runner.ROLLING_WINDOW_QUARTERS,
                "window": {
                    "nominal_quarters": runner.ROLLING_WINDOW_QUARTERS,
                    "minimum_observations": 40,
                },
                "regression_rows": len(regression_rows),
                "correlation_rows": 1,
                "effective_sample_start": "2010Q1",
                "effective_sample_end": "2021Q4",
                "effective_sample_bounds": bounds,
                "last_joint_observed_quarter": {
                    runner.CANONICAL_OUTCOME_ID: "2021Q4",
                    runner.CANONICAL_RESIDUAL_ID: "2021Q4",
                },
                "outputs": {
                    "regression_estimates": runner.ROLLING_ESTIMATES_LOCATOR,
                    "correlations": correlation_locator,
                    "report": report_locator,
                },
            },
        )

    def write_offset() -> None:
        locators = {
            "episode_betas": (
                "output/reports/tier2_pass_through_offset_episode_betas.csv"
            ),
            "level_summary": (
                "output/reports/tier2_pass_through_offset_level_summary.csv"
            ),
            "correlations": (
                "output/reports/tier2_pass_through_offset_correlations.csv"
            ),
            "identity_windows": (
                "output/reports/tier2_pass_through_offset_identity_windows.csv"
            ),
            "lead_lag_correlations": (
                "output/reports/"
                "tier2_pass_through_offset_lead_lag_correlations.csv"
            ),
            "jackknife": (
                "output/reports/tier2_pass_through_offset_2020_2021_jackknife.csv"
            ),
            "rolling_beta_features": (
                "output/reports/"
                "tier2_pass_through_offset_rolling_beta_features.csv"
            ),
            "rolling_beta_correlates": runner.FORMAL_CREDIT_SCREEN_LOCATOR,
            "report": "output/reports/tier2_pass_through_offset_diagnostics.md",
        }
        for key, locator in locators.items():
            if key == "report":
                _write_text(root, locator, "# Offset fixture\n")
            else:
                _write_csv(root, locator, [{"status": "computed"}])
        _write_json(
            root,
            runner.OFFSET_MANIFEST_LOCATOR,
            {"job_id": "offset", "outputs": locators},
        )

    def write_persistence() -> None:
        locators = {
            "rolling_minus_pandemic": (
                "output/reports/"
                "tier2_pass_through_rolling_minus_pandemic_betas.csv"
            ),
            "influence_quarters": (
                "output/reports/tier2_pass_through_influence_quarters.csv"
            ),
            "ratewall_summary": (
                "output/models/"
                "tdc_deposit_pass_through_pandemic_exclusion_diagnostics.csv"
            ),
            "report": (
                "output/reports/tier2_pass_through_regime_persistence.md"
            ),
        }
        for key, locator in locators.items():
            if key == "report":
                _write_text(root, locator, "# Persistence fixture\n")
            else:
                _write_csv(root, locator, [{"status": "computed"}])
        _write_json(
            root,
            runner.PERSISTENCE_MANIFEST_LOCATOR,
            {"job_id": "persistence", "outputs": locators},
        )

    def write_regime_validation() -> None:
        locators = {
            "classifier_candidates": (
                "outputs/tables/"
                "ea_tdc_pass_through_regime_classifier_candidates.csv"
            ),
            "regime_estimates": (
                "outputs/tables/ea_tdc_pass_through_regime_estimates.csv"
            ),
            "regime_validation": (
                "outputs/tables/ea_tdc_pass_through_regime_validation.csv"
            ),
            "ratewall_import_contract": (
                "outputs/tables/ea_tdc_pass_through_ratewall_import_contract.csv"
            ),
            "memo": (
                "outputs/reports/ea_tdc_pass_through_regime_validation_memo.md"
            ),
        }
        hashes: dict[str, str] = {}
        for key, locator in locators.items():
            if key == "memo":
                path = _write_text(root, locator, "# Regime fixture\n")
            else:
                path = _write_csv(root, locator, [{"status": "computed"}])
            hashes[key] = _sha256(path)
        _write_json(
            root,
            runner.REGIME_VALIDATION_MANIFEST_LOCATOR,
            {
                "job_id": "regime_validation",
                "outputs": locators,
                "output_sha256": hashes,
            },
        )

    def write_acceptance() -> None:
        write_source()
        for job_id in runner.OPEN01_DESIGN_JOB_IDS:
            write_design(job_id)
        contract_path = _write_csv(
            root,
            "outputs/tables/tdc_treatment_outcome_contract.csv",
            [{"contract_status": "frozen"}],
        )
        headline_path = _write_csv(
            root,
            "outputs/tables/tdc_same_quarter_headline.csv",
            [{"status": "estimated"}],
        )
        stability_path = _write_csv(
            root,
            "output/reports/tier2_pass_through_stability_gate.csv",
            [{"status": "pass"}],
        )
        pattern_json = json.dumps([[used_csv, rejected_csv]])
        credit_rows = []
        for outcome in runner.CREDIT_SCREEN_OUTCOME_IDS:
            for window in (40, runner.ROLLING_WINDOW_QUARTERS, 60):
                for adjustment in runner.CREDIT_ADJUSTMENTS:
                    association_observations = 96 - window + 1
                    association_hac_lags = min(
                        window - 1,
                        association_observations - 2,
                    )
                    credit_rows.append(
                        {
                            "status": "computed",
                            "credit_outcome_id": outcome,
                            "treatment_id": runner.CANONICAL_TREATMENT_ID,
                            "rolling_outcome_id": runner.CANONICAL_OUTCOME_ID,
                            "control_ids": ",".join(controls),
                            "window_quarters": window,
                            "rolling_window_observations": window,
                            "n_windows": association_observations,
                            "association_observations": (
                                association_observations
                            ),
                            "association_hac_lags": association_hac_lags,
                            "covariance_lags": association_hac_lags,
                            "association_hac_bandwidth_ratio": (
                                association_hac_lags
                                / association_observations
                            ),
                            "inference_calibration_status": (
                                "uncalibrated_fixed_bandwidth_normal_reference"
                            ),
                            "calibration_method": "",
                            "calibrated_p_value": "",
                            "calibrated_lower95": "",
                            "calibrated_upper95": "",
                            "outcome_iut_p_value_raw": "",
                            "outcome_iut_p_value_holm": "",
                            "outcome_iut_family_complete": False,
                            "admission_status": "appendix_only",
                            "admission_reason": (
                                "uncalibrated_component_inference"
                            ),
                            "last_window_end": "2021Q4",
                            "last_observed_treatment_outcome_quarter": "2021Q4",
                            "sign_40": "positive",
                            "sign_48": "positive",
                            "sign_60": "positive",
                            "adjustment": adjustment,
                            "multiple_testing_family": (
                                f"credit_{window}_{adjustment}"
                            ),
                            "rolling_control_patterns_json": pattern_json,
                        }
                    )
        credit_path = _write_csv(
            root,
            runner.FORMAL_CREDIT_SCREEN_LOCATOR,
            credit_rows,
        )
        outputs = {
            "contract": {
                "path": "outputs/tables/tdc_treatment_outcome_contract.csv",
                "sha256": _sha256(contract_path),
                "rows": 1,
            },
            "headline": {
                "path": "outputs/tables/tdc_same_quarter_headline.csv",
                "sha256": _sha256(headline_path),
                "rows": 1,
            },
            "stability": {
                "path": "output/reports/tier2_pass_through_stability_gate.csv",
                "sha256": _sha256(stability_path),
                "rows": 1,
            },
            "credit_screen": {
                "path": runner.FORMAL_CREDIT_SCREEN_LOCATOR,
                "sha256": _sha256(credit_path),
                "rows": len(credit_rows),
            },
        }
        _write_json(
            root,
            runner.ACCEPTANCE_MANIFEST_LOCATOR,
            {
                "open_id": "OPEN-01",
                "status": "passed",
                "producer_status": "pass",
                "scientific_status": "stable",
                "acceptance_checks": {
                    "fixture_dict": {"passed": True},
                    "fixture_bool": True,
                },
                "outputs": outputs,
                "contract": {
                    "treatment_id": runner.CANONICAL_TREATMENT_ID,
                    "outcome_id": runner.CANONICAL_OUTCOME_ID,
                    "residual_id": runner.CANONICAL_RESIDUAL_ID,
                    "control_ids": list(controls),
                },
                "units": {
                    "treatment": runner.OPEN_CONTRACT.treatment_units,
                    "deposit_outcome": runner.OPEN_CONTRACT.deposit_outcome_units,
                    "credit_outcomes": runner.OPEN_CONTRACT.credit_outcome_units,
                    "estimand": "dollars_per_dollar_tdc",
                },
                "sample": {
                    "start": "2002Q1",
                    "end": "2025Q4",
                    "n": 96,
                },
                "producer_inputs": {"fixture": True},
                "issues": [],
            },
        )

    writers = {
        "adapt_tdcest": write_source,
        "submission_appendix_factor_preparation": write_submission,
        "rolling": write_rolling,
        "offset": write_offset,
        "regime_persistence": write_persistence,
        "regime_validation": write_regime_validation,
        "acceptance": write_acceptance,
    }

    def materialize(stage: str) -> None:
        if stage.startswith("design:"):
            write_design(stage.removeprefix("design:"))
        else:
            writers[stage]()

    return materialize


def test_open01_producer_runs_exact_chain_and_writes_complete_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    bundle_path = _prepare_preflight(runner, tmp_path)
    materialize = _fixture_materializer(runner, tmp_path, bundle_path)
    commit = "a" * 40
    monkeypatch.setattr(
        runner,
        "_verify_producer_commit",
        lambda root, producer_commit: commit,
    )
    monkeypatch.setattr(runner, "_git_text", lambda root, *args: commit)
    expected_commands = runner._pipeline_commands(
        python_executable="python",
        tdcest_bundle="tdcest/bundle.json",
    )
    calls: list[list[str]] = []

    def fake_run(argv: list[str], cwd: Path):
        stage, expected_argv = expected_commands[len(calls)]
        assert cwd == tmp_path.resolve()
        assert argv == expected_argv
        calls.append(list(argv))
        materialize(stage)
        return subprocess.CompletedProcess(argv, 0)

    invocation = [
        "scripts/run_open01_producer.py",
        "--producer-commit",
        commit,
        "--run-id",
        "open01-fixture",
        "--tdcest-bundle",
        "tdcest/bundle.json",
    ]
    receipt_path = runner.run_open01(
        producer_commit=commit,
        run_id="open01-fixture",
        tdcest_bundle="tdcest/bundle.json",
        root=tmp_path,
        python_executable="python",
        run_command=fake_run,
        invocation_argv=invocation,
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "passed"
    assert receipt["run_id"] == "open01-fixture"
    assert receipt["producer_commit"] == commit
    assert receipt["argv"] == invocation
    assert [record["stage"] for record in receipt["commands"]] == [
        "adapt_tdcest",
        *(f"design:{job_id}" for job_id in runner.OPEN01_DESIGN_JOB_IDS),
        "submission_appendix_factor_preparation",
        "rolling",
        "offset",
        "regime_persistence",
        "regime_validation",
        "acceptance",
    ]
    assert len(calls) == 11
    assert calls[-1][-2:] == ["--tdcest-bundle", "tdcest/bundle.json"]
    assert all(
        record["refreshed_manifests"]
        for record in receipt["commands"]
    )
    assert len(receipt["producer_manifests"]) == 15
    assert (
        receipt["producer_manifests"][runner.SOURCE_MANIFEST_LOCATOR][
            "final_writer_stage"
        ]
        == "acceptance"
    )

    assert set(receipt["designs"]) == set(runner.OPEN01_DESIGN_JOB_IDS)
    assert all(
        design["bundle"]["sha256"]
        and design["design_manifest"]["sha256"]
        and design["sample_manifest"]["sha256"]
        for design in receipt["designs"].values()
    )
    source = receipt["tdcest"]
    assert source["input_hashes"]["seed_bundle"] == _sha256(bundle_path)
    assert source["combined_input_hash"]
    assert source["standardized"]["sha256"] == _sha256(
        tmp_path / runner.STANDARDIZED_LOCATOR
    )

    declared = receipt["retained_outputs"]["manifest_declared"]
    assert runner.FORMAL_CREDIT_SCREEN_LOCATOR in declared
    assert len(
        declared[runner.FORMAL_CREDIT_SCREEN_LOCATOR]["declared_by"]
    ) == 2
    assert (
        receipt["acceptance"]["outputs"]["credit_screen"]["sha256"]
        == _sha256(tmp_path / runner.FORMAL_CREDIT_SCREEN_LOCATOR)
    )
    assert receipt["acceptance"]["contract"]["control_ids"] == list(
        runner.CANONICAL_CONTROL_IDS
    )
    assert receipt["acceptance"]["sample"]["n"] == 96
    cross_check = receipt["acceptance"]["receipt_checks"][
        "rolling_offset_credit_contract"
    ]
    assert cross_check["passed"] is True
    assert cross_check["details"]["canonical_credit_window_quarters"] == 48
    assert cross_check["details"]["sign_sensitivity_window_quarters"] == [40, 60]
    assert cross_check["details"]["rolling_realized_control_patterns"] == [
        {
            "used": [
                control
                for control in runner.CANONICAL_CONTROL_IDS
                if control
                != "tier2_regression_bank_row_tier_pre_component_h15_scaled"
            ],
            "rejected": [
                "tier2_regression_bank_row_tier_pre_component_h15_scaled"
            ],
        }
    ]


@pytest.mark.parametrize("mutation", ["duplicate", "unexpected_adjustment"])
def test_cross_surface_gate_rejects_non_cartesian_credit_screen(
    tmp_path: Path,
    mutation: str,
) -> None:
    runner = _load_runner()
    bundle_path = _prepare_preflight(runner, tmp_path)
    materialize = _fixture_materializer(runner, tmp_path, bundle_path)
    materialize("rolling")
    materialize("acceptance")
    credit_path = tmp_path / runner.FORMAL_CREDIT_SCREEN_LOCATOR
    with credit_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 45
    if mutation == "duplicate":
        rows[-1] = dict(rows[0])
    else:
        rows[-1]["adjustment"] = "undeclared_adjustment"
        rows[-1]["multiple_testing_family"] = (
            f"credit_{rows[-1]['window_quarters']}_undeclared_adjustment"
        )
    _write_csv(
        tmp_path,
        runner.FORMAL_CREDIT_SCREEN_LOCATOR,
        rows,
    )

    with pytest.raises(ValueError, match="Cartesian product"):
        runner._cross_surface_contract_gate(
            root=tmp_path,
            acceptance_outputs={
                "credit_screen": {
                    "path": runner.FORMAL_CREDIT_SCREEN_LOCATOR,
                }
            },
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("hac_lag", "overlap-HAC lag metadata"),
        ("uncalibrated_main_text", "appendix-only gate"),
    ],
)
def test_cross_surface_gate_rejects_untruthful_credit_inference(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    runner = _load_runner()
    bundle_path = _prepare_preflight(runner, tmp_path)
    materialize = _fixture_materializer(runner, tmp_path, bundle_path)
    materialize("rolling")
    materialize("acceptance")
    credit_path = tmp_path / runner.FORMAL_CREDIT_SCREEN_LOCATOR
    with credit_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if mutation == "hac_lag":
        rows[0]["association_hac_lags"] = "1"
    else:
        rows[0]["admission_status"] = "main_text_eligible"
    _write_csv(
        tmp_path,
        runner.FORMAL_CREDIT_SCREEN_LOCATOR,
        rows,
    )

    with pytest.raises(ValueError, match=message):
        runner._cross_surface_contract_gate(
            root=tmp_path,
            acceptance_outputs={
                "credit_screen": {
                    "path": runner.FORMAL_CREDIT_SCREEN_LOCATOR,
                }
            },
        )


def test_open01_producer_removes_stale_receipt_and_fails_on_subprocess(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    _prepare_preflight(runner, tmp_path)
    stale = _write_json(
        tmp_path,
        runner.RECEIPT_LOCATOR,
        {"status": "passed", "run_id": "stale"},
    )
    commit = "b" * 40
    monkeypatch.setattr(
        runner,
        "_verify_producer_commit",
        lambda root, producer_commit: commit,
    )

    def fail(argv: list[str], cwd: Path):
        raise subprocess.CalledProcessError(7, argv)

    with pytest.raises(subprocess.CalledProcessError):
        runner.run_open01(
            producer_commit=commit,
            run_id="open01-fails",
            tdcest_bundle="tdcest/bundle.json",
            root=tmp_path,
            python_executable="python",
            run_command=fail,
        )
    assert not stale.exists()


@pytest.mark.parametrize(
    "checks",
    [
        {},
        {"unknown": {"details": "no explicit status"}},
        {"failed_bool": False},
        {"failed_dict": {"passed": False}},
        {"unknown_status": {"status": "green"}},
    ],
)
def test_acceptance_rejects_empty_unknown_or_failed_checks(
    tmp_path: Path,
    checks: dict[str, Any],
) -> None:
    runner = _load_runner()
    output = _write_text(tmp_path, "output/result.txt")
    payload = {
        "status": "passed",
        "acceptance_checks": checks,
        "outputs": {
            "result": {
                "path": "output/result.txt",
                "sha256": _sha256(output),
            }
        },
    }
    with pytest.raises(ValueError):
        runner._validate_acceptance(tmp_path, payload)


def test_stage_must_recreate_its_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    _prepare_preflight(runner, tmp_path)
    commit = "c" * 40
    monkeypatch.setattr(
        runner,
        "_verify_producer_commit",
        lambda root, producer_commit: commit,
    )

    def no_output(argv: list[str], cwd: Path):
        return subprocess.CompletedProcess(argv, 0)

    with pytest.raises(FileNotFoundError, match="tdcest_source_manifest"):
        runner.run_open01(
            producer_commit=commit,
            run_id="open01-no-manifest",
            tdcest_bundle="tdcest/bundle.json",
            root=tmp_path,
            python_executable="python",
            run_command=no_output,
        )
    assert not (tmp_path / runner.RECEIPT_LOCATOR).exists()


def test_producer_commit_verification_rejects_untracked_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    commit = "d" * 40
    calls: list[tuple[str, ...]] = []

    def git_text(root: Path, *args: str) -> str:
        calls.append(args)
        if args == ("rev-parse", "HEAD"):
            return commit
        if args == (
            "status",
            "--porcelain",
            "--untracked-files=all",
        ):
            return "?? scripts/untracked_producer.py"
        raise AssertionError(args)

    monkeypatch.setattr(runner, "_git_text", git_text)
    with pytest.raises(ValueError, match="untracked nonignored"):
        runner._verify_producer_commit(tmp_path, commit)
    assert (
        "status",
        "--porcelain",
        "--untracked-files=all",
    ) in calls


def test_producer_commit_verification_accepts_only_clean_checkout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    commit = "e" * 40

    def git_text(root: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return commit
        if args == (
            "status",
            "--porcelain",
            "--untracked-files=all",
        ):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(runner, "_git_text", git_text)
    assert runner._verify_producer_commit(tmp_path, commit) == commit
