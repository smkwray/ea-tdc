from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def validator(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("open16_admission_test", scripts / "validate_open16_reproduction.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_semantic_comparison_preserves_named_values_despite_column_order(validator):
    first = [{"quarter": "1945Q4", "a": "1.000", "b": ""}]
    second = [{"quarter": "1945Q4", "b": "", "a": "1.000"}]
    assert validator.compare_semantic_universes(first, second) == {
        "rows": 1, "features": 2, "cell_differences": 0, "serialized_column_order_equal": False}
    second[0]["a"] = "1.0"
    with pytest.raises(ValueError, match="named cells"):
        validator.compare_semantic_universes(first, second)


def test_semantic_comparison_rejects_inventory_change(validator):
    with pytest.raises(ValueError, match="inventory"):
        validator.compare_semantic_universes([{"quarter": "1945Q4", "a": "1"}], [{"quarter": "1946Q4", "a": "1"}])


def test_conditioning_projector_is_rotation_invariant_with_full_rank(validator):
    rng = np.random.default_rng(64)
    x = np.column_stack((np.ones(96), rng.normal(size=(96, 4))))
    transform = np.eye(5)
    transform[1:, 1:] = [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 2, 1], [0, 0, 0, .5]]
    rank, projector = validator.rank_projector(x.tolist())
    transformed_rank, other = validator.rank_projector((x @ transform).tolist())
    assert rank == transformed_rank == 5
    assert validator.max_difference(projector, other) < 1e-12
    with pytest.raises(ValueError, match="full rank"):
        validator.rank_projector(np.column_stack((x, x[:, 1])).tolist())


def reports():
    windows = [{"window": str(i), "quarters": ["2002Q1"], "controls_used": ["z"], "controls_rejected": [],
                "conditioning_rank": 2, "projector": [[1.0]], "coefficients": [1.0], "fitted": [1.0],
                "residuals": [0.0], "covariance": [[.2]], "beta_se": [1.0, .4]} for i in range(58)]
    first = {"runtime": {"executable_sha256": "a"}, "factor_rank": 4, "sample_factor_rank": 4, "windows": windows}
    second = copy.deepcopy(first)
    second["runtime"] = {"executable_sha256": "b"}
    return first, second


@pytest.mark.parametrize("mutation", ["projector", "fitted", "controls_used", "conditioning_rank", "missing_window", "factor_rank", "same_environment"])
def test_pinned_environment_gate_rejects_drift_and_missing_evidence(validator, mutation):
    first, second = reports()
    if mutation == "missing_window": second["windows"].pop()
    elif mutation == "factor_rank": second["factor_rank"] = 3
    elif mutation == "same_environment": second["runtime"] = first["runtime"]
    elif mutation == "controls_used": second["windows"][0][mutation] = ["other"]
    elif mutation == "conditioning_rank": second["windows"][0][mutation] = 1
    elif mutation == "projector": second["windows"][0][mutation][0][0] += 1e-5
    else: second["windows"][0][mutation][0] += 1e-5
    with pytest.raises(ValueError): validator.compare_environments(first, second)


def test_pinned_environment_gate_keeps_exact_absolute_tolerance(validator):
    first, second = reports()
    second["windows"][0]["fitted"][0] += 1e-8
    assert validator.compare_environments(first, second)["fitted"] == pytest.approx(1e-8)
    with pytest.raises(ValueError): validator.max_difference([1e12], [1e12 + .001])


@pytest.mark.parametrize("expanded_scope", [False, True])
def test_fresh_authority_fails_closed_before_any_output(validator, tmp_path, monkeypatch, expanded_scope):
    import run_open16_diagnostics as runner

    gate = json.loads((Path(__file__).resolve().parents[1] / "config/open16_reproduction_authority.json").read_text())
    gate["status"] = "conditional_pending_validation"
    if expanded_scope:
        gate["status"] = "approved_bounded_scope"
        gate["scope"].append("three_leg_slopes")
    path = tmp_path / "config/open16_reproduction_authority.json"
    path.parent.mkdir(); path.write_text(json.dumps(gate))
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=path.read_bytes()))
    with pytest.raises(ValueError, match="unavailable or scope expanded"):
        runner.load_fresh_authority(tmp_path, "c" * 40)
    assert list(tmp_path.iterdir()) == [path.parent]


def test_fresh_authority_rejects_unqualified_current_runtime_before_inputs(validator, tmp_path, monkeypatch):
    import run_open16_diagnostics as runner
    import validate_open16_reproduction as admission

    gate = json.loads((Path(__file__).resolve().parents[1] / "config/open16_reproduction_authority.json").read_text())
    gate["status"] = "approved_bounded_scope"
    path = tmp_path / "config/open16_reproduction_authority.json"
    path.parent.mkdir(); path.write_text(json.dumps(gate))
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=path.read_bytes()))
    monkeypatch.setattr(admission, "runtime_identity", lambda: {"unqualified": True})
    with pytest.raises(ValueError, match="runtime lacks pinned"):
        runner.load_fresh_authority(tmp_path, "c" * 40)
    assert list(tmp_path.iterdir()) == [path.parent]


def test_manifested_source_drift_rejects_before_reproduction_input(validator, tmp_path, monkeypatch):
    import run_open16_diagnostics as runner
    import validate_open16_reproduction as admission

    gate = json.loads((Path(__file__).resolve().parents[1] / "config/open16_reproduction_authority.json").read_text())
    gate["status"] = "approved_bounded_scope"
    gate["validation_producer_commit"] = "c" * 40
    gate["validation_receipt"] = {"path": "proof.json", "sha256": "a" * 64, "bytes": 1}
    for name in admission.PRODUCTION_SOURCES:
        p = tmp_path / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("validated source")
    manifest = admission.production_manifest(tmp_path)
    (tmp_path / admission.PRODUCTION_SOURCES[0]).write_text("changed source")
    proof = {"status": "passed", "gates": dict.fromkeys(admission.GATES, True),
             "authority_class": gate["authority_class"], "scope": gate["scope"],
             "numerical_policy": gate["numerical_policy"], "structural_evidence": gate["structural_evidence"],
             "prior_failed_validation_records": gate["prior_failed_validation_records"],
             "environments": gate["environments"], "producer_commit": gate["validation_producer_commit"],
             "reproduction_receipt_sha256": gate["reproduction_receipt"]["sha256"],
             "production_source_manifest": manifest}
    (tmp_path / "proof.json").write_text(json.dumps(proof))
    path = tmp_path / "config/open16_reproduction_authority.json"
    path.parent.mkdir(exist_ok=True); path.write_text(json.dumps(gate))
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=path.read_bytes()))
    monkeypatch.setattr(admission, "runtime_identity", lambda: gate["environments"][0])
    operations = []
    def authority_file(root, item):
        operations.append(item["path"])
        assert item == gate["validation_receipt"]
    monkeypatch.setattr(runner, "_authority_file", authority_file)
    monkeypatch.setattr(admission, "verify_package", lambda *a: pytest.fail("Reproduction input read before source rejection"))
    with pytest.raises(ValueError, match="Production source differs"):
        runner.load_fresh_authority(tmp_path, "c" * 40)
    assert operations == ["proof.json"]
    assert not (tmp_path / "output").exists()


@pytest.fixture
def policy():
    return json.loads((Path(__file__).resolve().parents[1] / "config/open16_reproduction_authority.json").read_text())["numerical_policy"]


def synthetic_design():
    rng = np.random.default_rng(871)
    x = np.column_stack((np.ones(96), rng.normal(size=(96, 4))))
    y = x @ np.array([2., .5, -.2, .3, .4]) + rng.normal(size=96)
    return x, y


def test_independent_scaled_svd_matches_accepted_newey_west(validator, policy):
    from ea_tdc.estimation import _ols
    x, y = synthetic_design()
    geometry = validator.scaled_geometry(x, y, policy)
    reference = validator.svd_reference(x, y, geometry)
    fit = _ols(y.tolist(), x.tolist(), covariance_estimator="newey_west", covariance_lags=1)
    accepted = {"coefficients": fit.beta, "fitted": fit.fitted, "residuals": fit.residuals,
                "covariance": fit.covariance, "beta_se": [fit.beta[1], fit.ses[1]]}
    gaps = validator.compare_scaled(accepted, reference, geometry["scales"], geometry["outcome_scale"], policy)
    assert max(gaps.values()) < 1e-12


def test_v2_full_comparison_is_invariant_to_control_units(validator, policy):
    x, y = synthetic_design()
    geometry = validator.scaled_geometry(x, y, policy)
    reference = validator.svd_reference(x, y, geometry)
    perturbed = copy.deepcopy(reference)
    perturbed["coefficients"][2] += 1e-10
    perturbed["covariance"][2][2] += 1e-12
    reference.pop("raw_covariance"); perturbed.pop("raw_covariance")
    before = validator.compare_scaled(reference, perturbed, geometry["scales"], geometry["outcome_scale"], policy)
    conversion = np.array([1., 1., 1024., 1. / 1024., 1.])
    for values in (reference, perturbed):
        values["coefficients"] = (np.asarray(values["coefficients"]) / conversion).tolist()
        values["covariance"] = (np.asarray(values["covariance"]) / np.outer(conversion, conversion)).tolist()
    after = validator.compare_scaled(reference, perturbed, geometry["scales"] * conversion, geometry["outcome_scale"], policy)
    assert after == pytest.approx(before, rel=2e-6, abs=1e-16)
    _, old = validator.rank_projector(x.tolist())
    _, new = validator.rank_projector((x * conversion).tolist())
    assert np.linalg.norm(np.asarray(old) - new, 2) < policy["projector_tolerance"]


def test_v2_condition_guard_does_not_expand_tolerance(validator, policy):
    x, y = synthetic_design()
    x[:, 4] = x[:, 3] + 1e-5 * x[:, 4]
    with pytest.raises(ValueError, match="condition guard"):
        validator.scaled_geometry(x, y, policy)


@pytest.mark.parametrize("fault", ["beta", "nonfinite", "asymmetric", "negative_variance", "vector"])
def test_v2_rejects_substantive_or_invalid_numerical_values(validator, policy, fault):
    x, y = synthetic_design()
    geometry = validator.scaled_geometry(x, y, policy)
    reference = validator.svd_reference(x, y, geometry)
    changed = copy.deepcopy(reference)
    changed.pop("raw_covariance")
    if fault == "beta": changed["beta_se"][0] += 1e-5
    elif fault == "nonfinite": changed["fitted"][0] = float("nan")
    elif fault == "asymmetric": changed["covariance"][1][2] += .01
    elif fault == "negative_variance": changed["covariance"][2][2] = -1.
    else: changed["fitted"][0] += .01
    with pytest.raises(ValueError):
        validator.compare_scaled(reference, changed, geometry["scales"], geometry["outcome_scale"], policy)


@pytest.mark.parametrize("fault", ["conditioning_projector", "full_projector", "policy", "rank", "missing_window"])
def test_v2_pair_gate_rejects_projector_policy_or_discrete_drift(validator, policy, fault):
    first, second = reports()
    for report in (first, second):
        report["numerical_policy"] = policy
        for window in report["windows"]:
            window.update(full_projector=[[1.]], scales=[1., 1.], outcome_scale=1., design_rank=2, condition=1.,
                          input_inventory_sha256="a" * 64, coefficients=[0., 1.], covariance=[[.2, 0.], [0., .2]])
            window["independent_svd"] = {k: window[k] for k in ("coefficients", "fitted", "residuals", "covariance", "beta_se")}
    if fault == "policy": second["numerical_policy"] = {**policy, "projector_tolerance": 1.}
    elif fault == "rank": second["windows"][0]["design_rank"] = 3
    elif fault == "missing_window": second["windows"].pop()
    else:
        key = "projector" if fault == "conditioning_projector" else "full_projector"
        second["windows"][0][key] = [[1. + 2e-10]]
    with pytest.raises(ValueError):
        validator.compare_policy_environments(first, second, policy)


def operator_reports(policy):
    first, second = reports()
    for report in (first, second):
        report["numerical_policy"] = policy
        for window in report["windows"]:
            window.update(full_projector=[[1.]], scales=[1., 1.], outcome_scale=1., design_rank=2, condition=1.,
                          input_inventory_sha256="a" * 64, coefficients=[0., 1.], covariance=[[.2, 0.], [0., .2]])
            window["independent_svd"] = {k: window[k] for k in ("coefficients", "fitted", "residuals", "covariance", "beta_se")}
    return first, second


def test_one_ulp_normalization_drift_preserves_input_identity(validator, policy):
    first, second = operator_reports(policy)
    second["windows"][0]["scales"][0] = float(np.nextafter(1., 2.))
    second["windows"][0]["outcome_scale"] = float(np.nextafter(1., 2.))
    assert max(validator.compare_policy_environments(first, second, policy)["maxima"].values()) == 0


@pytest.mark.parametrize("field", ["quarter", "value"])
def test_changed_input_row_or_value_fails_exact_inventory_gate(validator, policy, field):
    first, second = operator_reports(policy)
    original = [{"quarter": "2002Q1", "value": "1.000"}]
    changed = [{**original[0], field: "2002Q2" if field == "quarter" else "1.001"}]
    first["windows"][0]["input_inventory_sha256"] = validator.input_inventory_hash(original, ["quarter", "value"])
    second["windows"][0]["input_inventory_sha256"] = validator.input_inventory_hash(changed, ["quarter", "value"])
    with pytest.raises(ValueError, match="discrete design identity"):
        validator.compare_policy_environments(first, second, policy)


def test_covariance_operator_preserves_raw_and_exact_diagonal():
    from ea_tdc.covariance import canonical_covariance
    raw = np.array([[2., .2 + 1e-10], [.2, 1.]])
    original = raw.copy()
    evidence = canonical_covariance(raw, [2., 3.], 4.)
    assert np.array_equal(raw, original)
    assert np.array_equal(evidence["raw_covariance"], original)
    assert np.array_equal(np.diag(evidence["covariance"]), np.diag(original))
    assert np.array_equal(evidence["covariance"], np.asarray(evidence["covariance"]).T)
    assert 0 < evidence["diagnostics"]["representation_relative_error"] < 1e-7
    assert evidence["diagnostics"]["minimum_scaled_eigenvalue"] > 0


@pytest.mark.parametrize("raw", [[[1., 2.], [2., 1.]], [[1., 0.], [0., -1e-30]],
                                 [[1., 0.], [0., 0.]], [[1., .1], [0., 1.]]])
def test_covariance_operator_rejects_indefiniteness_negative_zero_or_skew(raw):
    from ea_tdc.covariance import canonical_covariance
    with pytest.raises(ValueError):
        canonical_covariance(raw, [1., 1.], 1.)


def test_covariance_operator_rejects_nonfinite_derived_norm(monkeypatch):
    from ea_tdc.covariance import canonical_covariance
    monkeypatch.setattr(np.linalg, "norm", lambda *a, **kw: float("inf"))
    with pytest.raises(ValueError, match="Nonfinite covariance norm"):
        canonical_covariance([[1., 0.], [0., 1.]], [1., 1.], 1.)


def test_authority_file_accepts_exact_empty_file(validator, tmp_path):
    import run_open16_diagnostics as runner

    path = tmp_path / "silent.log"
    path.write_bytes(b"")
    item = validator.record(tmp_path, path)
    assert item["bytes"] == 0
    assert runner._authority_file(tmp_path, item) == item


@pytest.mark.parametrize("fault", ["negative", "wrong_size", "wrong_hash", "missing", "escape"])
def test_authority_file_rejects_invalid_empty_file_record(validator, tmp_path, fault):
    import run_open16_diagnostics as runner

    path = tmp_path / "silent.log"
    path.write_bytes(b"")
    item = validator.record(tmp_path, path)
    if fault == "negative": item["bytes"] = -1
    elif fault == "wrong_size": item["bytes"] = 1
    elif fault == "wrong_hash": item["sha256"] = "0" * 64
    elif fault == "missing": path.unlink()
    else: item["path"] = "../silent.log"
    with pytest.raises(FileNotFoundError if fault == "missing" else ValueError):
        runner._authority_file(tmp_path, item)


def test_complete_validation_receipt_loads_under_proposed_authority(validator, tmp_path, monkeypatch):
    import run_open16_diagnostics as runner
    import validate_open16_reproduction as admission

    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        return admission.record(tmp_path, path)

    gate = json.loads((Path(__file__).resolve().parents[1] / admission.CONFIG).read_text())
    gate.update(status="approved_bounded_scope", validation_producer_commit="c" * 40,
                environments=[admission.runtime_identity()])
    for name in admission.PRODUCTION_SOURCES:
        write(name, b"synthetic validated source\n")
    package = tmp_path / "reproduction"
    panel = write("reproduction/results/frozen_projection_panel.csv", b"quarter,value\n2002Q1,1\n")
    original = write("reproduction/inputs/" + runner.OPEN01_RECEIPT_LOCATOR, {"synthetic": True})
    input_receipt = write("reproduction/input_receipt.json", {
        "frozen_input_graph": [{**original, "path": runner.OPEN01_RECEIPT_LOCATOR}]})
    gate["reproduction_receipt"] = write("reproduction/receipt.json", {
        "origin": "new_frozen_input_reproduction", "input_receipt_sha256": input_receipt["sha256"],
        "retained_outputs": [{**panel, "path": "results/frozen_projection_panel.csv"}]})
    gate["structural_evidence"] = {"records": [write("structural.json", {"passed": True})]}
    gate["prior_failed_validation_records"] = [write("prior_failure.json", {"status": "failed"})]
    gate["legs"] = write("legs.csv", b"quarter,C,R,J,O\n2002Q1,3,1,1,1\n")
    gate["legs_receipt"] = write("legs_receipt.json", {
        "producer_commit": gate["upstream_producer_commit"], "output": {"sha256": gate["legs"]["sha256"]},
        "schema_version": "regression_legs_v1", "units": "USD million",
        "sample": {"start": "2002Q1", "end": "2025Q4", "n": 96}})
    for name in ("primary.log", "comparison.log"):
        write("proof/" + name, b"")
    write("proof/numerical_comparison.json", {"passed": True})
    proof_dir = tmp_path / "proof"
    proof = {"status": "passed", "gates": dict.fromkeys(admission.GATES, True),
             "producer_commit": gate["validation_producer_commit"],
             "production_source_manifest": admission.production_manifest(tmp_path),
             "reproduction_receipt_sha256": gate["reproduction_receipt"]["sha256"],
             "outputs": [admission.record(proof_dir, p) for p in sorted(proof_dir.iterdir())]}
    proof.update({key: gate[key] for key in ("authority_class", "scope", "numerical_policy",
                  "structural_evidence", "prior_failed_validation_records", "environments")})
    gate["validation_receipt"] = write("proof/receipt.json", proof)
    write(admission.CONFIG, gate)
    proposed = (tmp_path / admission.CONFIG).read_bytes()

    def proposed_git_show(argv, **kwargs):
        assert argv == ["git", "show", "proposed:" + admission.CONFIG]
        return SimpleNamespace(stdout=proposed)

    # Only the proposed Git trust-root read is simulated; every source, receipt,
    # input and output uses the real file/hash/path/length validators.
    monkeypatch.setattr(runner.subprocess, "run", proposed_git_show)
    loaded = runner.load_fresh_authority(tmp_path, "proposed")
    assert loaded["validation_receipt"] == gate["validation_receipt"]
    assert loaded["panel"]["path"] == str((package / "results/frozen_projection_panel.csv").relative_to(tmp_path))
    assert (proof_dir / "primary.log").stat().st_size == (proof_dir / "comparison.log").stat().st_size == 0
