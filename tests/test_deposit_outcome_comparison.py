"""A deposit-outcome comparison is valid only when the outcome is the sole change."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

from ea_tdc.open_contract import CANONICAL_CONTROL_IDS, CANONICAL_OUTCOME_ID, CANONICAL_TREATMENT_ID

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def compare():
    spec = importlib.util.spec_from_file_location("compare_deposit_outcome_variant",
                                                  ROOT / "scripts" / "compare_deposit_outcome_variant.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run(base: Path, compare, outcome: float, control: float = 1.0, score: float = 0.5, treatment: float = 10.0) -> Path:
    columns = {CANONICAL_TREATMENT_ID: treatment, CANONICAL_OUTCOME_ID: outcome, **{c: control for c in CANONICAL_CONTROL_IDS}}
    _write(base / compare.PANEL_LOCATOR, [{"quarter": q, **columns} for q in ("2026Q1", "2026Q2")])
    _write(base / compare.FACTOR_SCORES_LOCATOR, [{"quarter": "2026Q1", "dflmx_k100_f1": score}])
    return base


def test_only_the_deposit_outcome_may_differ_between_the_compared_runs(compare, tmp_path):
    baseline = _run(tmp_path / "a", compare, outcome=5.0)
    assert compare.identical_design(baseline, _run(tmp_path / "b", compare, outcome=7.0))["columns_differing"] == [CANONICAL_OUTCOME_ID]
    with pytest.raises(ValueError, match="more than the deposit outcome"):
        compare.identical_design(baseline, _run(tmp_path / "c", compare, outcome=7.0, control=2.0))
    with pytest.raises(ValueError, match="frozen factors were not reused"):
        compare.identical_design(baseline, _run(tmp_path / "d", compare, outcome=7.0, score=0.6))


def test_the_treatment_may_differ_only_when_the_variant_declares_one(compare, tmp_path):
    baseline = _run(tmp_path / "a", compare, outcome=5.0)
    moved = _run(tmp_path / "b", compare, outcome=7.0, treatment=10.5)
    with pytest.raises(ValueError, match="more than the deposit outcome"):
        compare.identical_design(baseline, moved)
    design = compare.identical_design(baseline, moved, treatment_declared=True)
    assert design["columns_differing"] == sorted([CANONICAL_OUTCOME_ID, CANONICAL_TREATMENT_ID])
    assert design["controls_identical"] and not design["treatment_identical"]
    # A declared treatment never licenses a change in the frozen controls or factors.
    with pytest.raises(ValueError, match="more than the deposit outcome"):
        compare.identical_design(baseline, _run(tmp_path / "c", compare, outcome=7.0, control=2.0, treatment=10.5),
                                 treatment_declared=True)
    with pytest.raises(ValueError, match="frozen factors were not reused"):
        compare.identical_design(baseline, _run(tmp_path / "d", compare, outcome=7.0, score=0.6, treatment=10.5),
                                 treatment_declared=True)


def test_conditioning_audit_reports_factor_and_bandwidth_changes_against_the_headline_fit(compare, tmp_path):
    def fit(beta, lags):
        return {"outcome": CANONICAL_OUTCOME_ID, "covariance_lags": lags, "n": "98", "beta": beta, "se": "0.2",
                "p_value": "0.04", "lower95": "0.1", "upper95": "0.9"}
    _write(tmp_path / compare.FACTOR_TAIL, [{"variant": "k100_factors_4", "factors_requested": "4", **fit("0.5", "1")},
                                             {"variant": "k0_no_factors", "factors_requested": "0", **fit("0.75", "1")},
                                             {"variant": "k100_factors_4", "factors_requested": "4",
                                              **fit("9.0", "1"), "outcome": "other"}])
    _write(tmp_path / compare.HAC, [fit("0.5", lag) for lag in ("1", "4", "6", "8")])
    audit = compare.conditioning_audit(tmp_path)
    assert [(r["variant"], r["newey_west_lag"]) for r in audit] == [
        ("k100_factors_4", "1"), ("k0_no_factors", "1"), ("k100_factors_4", "4"), ("k100_factors_4", "6"), ("k100_factors_4", "8")]
    assert [r["absolute_beta_change_from_k100_lag1"] for r in audit] == ["0.0", "0.25", "0.0", "0.0", "0.0"]


def test_a_seal_may_bind_an_earlier_configuration_that_lacked_only_the_reuse_pin(compare, monkeypatch):
    # The corrected vintage was sealed before its configuration gained the reuse pin; any other
    # difference would mean the baseline outputs came from a different specification.
    import hashlib
    import json
    import types

    from ea_tdc.vintage import load_vintage

    vintage = load_vintage(ROOT, "config/vintage_2026q2_tfx.json")

    def seal(**changes):
        earlier = {k: v for k, v in vintage.payload.items() if k != "reuse_acquisition"} | changes
        shown = json.dumps(earlier).encode()
        monkeypatch.setattr(compare.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=shown))
        return {"producer_commit": "a" * 40, "vintage": {**vintage.record(), "sha256": hashlib.sha256(shown).hexdigest()}}

    compare.sealed_configuration(vintage, seal())
    with pytest.raises(ValueError, match="sealed under another configuration"):
        compare.sealed_configuration(vintage, seal(observation_cutoff="2026-03-31"))
    tampered = seal()
    tampered["vintage"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sealed under another configuration"):
        compare.sealed_configuration(vintage, tampered)
