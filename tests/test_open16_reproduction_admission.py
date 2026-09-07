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
             "authority_class": gate["authority_class"], "scope": gate["scope"], "tolerance": 1e-7,
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
