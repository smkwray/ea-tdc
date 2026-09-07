from __future__ import annotations

import copy

import numpy as np
import pytest

from ea_tdc.estimation import _ols
from ea_tdc.open01 import _fit_projection
from ea_tdc.open16 import (
    CANONICAL_CONTROL_IDS, CANONICAL_OUTCOME_ID, CANONICAL_TREATMENT_ID,
    DELETED_QUARTERS, FROZEN_QUARTERS, calendar_hac, covariance_contributions,
    join_legs, leg_preflight, pandemic_path, validate_panel,
)


@pytest.fixture
def sample():
    rng = np.random.default_rng(182)
    z = rng.normal(size=(96, 16))
    legs = rng.normal(size=(96, 3)) * [100, 30, 50]
    c = legs.sum(axis=1)
    y = .516 * c + z @ np.arange(16) + rng.normal(size=96) * 20
    rows, sources = [], []
    for i, q in enumerate(FROZEN_QUARTERS):
        rows.append({"quarter": q, CANONICAL_TREATMENT_ID: c[i], CANONICAL_OUTCOME_ID: y[i],
                     **dict(zip(CANONICAL_CONTROL_IDS, z[i], strict=True))})
        sources.append({"quarter": q, "C": c[i], "R": legs[i, 0], "J": legs[i, 1], "O": legs[i, 2], "row_tsy_tx": legs[i, 0] + 12 * np.sin(i)})
    return rows, sources


def test_calendar_hac_matches_open01_coefficient_residual_covariance_and_se(sample):
    rows, _ = sample
    ref = _fit_projection(rows, treatment_id=CANONICAL_TREATMENT_ID, outcome_id=CANONICAL_OUTCOME_ID, control_ids=CANONICAL_CONTROL_IDS)
    x = [[1, r[CANONICAL_TREATMENT_ID], *[r[c] for c in CANONICAL_CONTROL_IDS]] for r in rows]
    actual = calendar_hac([r[CANONICAL_OUTCOME_ID] for r in rows], x, list(range(96)))
    assert actual.beta == ref.fit.beta
    assert actual.residuals == ref.fit.residuals
    np.testing.assert_allclose(actual.covariance, ref.fit.covariance, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual.ses, ref.fit.ses, rtol=1e-10, atol=1e-10)


def test_gap_does_not_pair_2019q4_with_2021q2():
    # Observed quarters 2019Q3, 2019Q4, 2021Q2, 2021Q3, 2021Q4.
    ordinals = [8078, 8079, 8085, 8086, 8087]
    x = [[1, v] for v in [0, 1, 4, 7, 9]]
    y = [3, 10, -2, 12, 3]
    actual = calendar_hac(y, x, ordinals)
    fit = _ols(y, x, covariance_estimator="classical")
    scores = np.asarray(x) * np.asarray(fit.residuals)[:, None]
    meat = scores.T @ scores
    # Independent explicit permitted calendar pairs; no pair (index 1,index 2).
    for a, b in [(0, 1), (2, 3), (3, 4)]:
        cross = np.outer(scores[a], scores[b])
        meat += .5 * (cross + cross.T)
    bread = np.linalg.inv(np.asarray(x).T @ np.asarray(x))
    expected = bread @ meat @ bread * 5 / 3
    np.testing.assert_allclose(actual.covariance, expected, rtol=1e-12, atol=1e-12)
    compressed = _ols(y, x, covariance_estimator="newey_west", covariance_lags=1)
    assert not np.allclose(actual.covariance, compressed.covariance)


@pytest.mark.parametrize("bad", [[1, 1, 2], [2, 1, 3], [1, 2]])
def test_calendar_rejects_duplicate_unordered_or_unaligned_quarters(bad):
    with pytest.raises(ValueError):
        calendar_hac([1, 4, 2], [[1, 0], [1, 1], [1, 3]], bad)


def test_calendar_rejects_rank_deficient_design():
    with pytest.raises(ValueError, match="full-rank"):
        calendar_hac([1, 2, 3], [[1, 1]] * 3, [1, 2, 3])


def test_pandemic_endpoints_no_refill_and_constant_control_policy(sample):
    rows, _ = sample
    # The frozen OPEN01 rule rejects controls constant inside a window.
    for i, row in enumerate(rows):
        row[CANONICAL_CONTROL_IDS[4]] = float(i < 33)
    result = pandemic_path(rows)
    assert len(result) == 49
    for q, n, last in [("2020Q1", 47, "2019Q4"), ("2020Q4", 44, "2019Q4"), ("2022Q3", 43, "2022Q3"), ("2025Q4", 43, "2025Q4")]:
        found = next(r for r in result if r["nominal_end"] == q)
        assert found["n_obs"] == n
        assert found["last_observed"] == last
        assert found["prespecified_endpoint"]
        expected_rank = 18 if q in ("2020Q1", "2020Q4") else 17
        assert found["rank"] == expected_rank
        assert found["controls_rejected"] == (CANONICAL_CONTROL_IDS[4] if expected_rank == 17 else "")
        assert found["finite_sample_scale"] == n / (n - expected_rank)
    assert len(DELETED_QUARTERS) == 5


def test_covariance_identity_and_classification_invariance(sample):
    rows, sources = sample
    panel = join_legs(rows, sources)
    b = _fit_projection(rows, treatment_id=CANONICAL_TREATMENT_ID, outcome_id=CANONICAL_OUTCOME_ID, control_ids=CANONICAL_CONTROL_IDS).beta
    result = covariance_contributions(panel, b)
    assert len(result) == 8
    for r in result:
        assert r["contribution_sum"] == pytest.approx(b - 1, abs=1e-12)
        assert abs(r["adding_up_gap"]) < r["adding_up_tolerance"]
    sums = [r["contribution"] for r in result if r["leg"] == "R+J"]
    assert sums[0] == pytest.approx(sums[1], abs=1e-12)
    rvals = [r["contribution"] for r in result if r["leg"] == "R"]
    assert abs(rvals[0] - rvals[1]) > 1e-5
    with pytest.raises(ValueError, match="identity fails"):
        covariance_contributions(panel, b + .1)


@pytest.mark.parametrize("mutation", ["drop", "duplicate", "missing_control", "nonfinite"])
def test_frozen_panel_rejects_drift(sample, mutation):
    rows = copy.deepcopy(sample[0])
    if mutation == "drop": rows.pop()
    elif mutation == "duplicate": rows[-1]["quarter"] = rows[0]["quarter"]
    elif mutation == "missing_control": rows[0].pop(CANONICAL_CONTROL_IDS[0])
    else: rows[0][CANONICAL_OUTCOME_ID] = float("inf")
    with pytest.raises(ValueError): validate_panel(rows)


@pytest.mark.parametrize("key", ["C", "R"])
def test_leg_join_rejects_treatment_or_adding_up_drift(sample, key):
    rows, sources = sample
    sources[0][key] += 1
    with pytest.raises(ValueError): join_legs(rows, sources)


def test_leg_join_does_not_drop_structural_raw_metadata_missingness(sample):
    rows, sources = sample
    for r in sources: r["mmf_rrp_adjustment_prop"] = ""
    assert len(join_legs(rows, sources)) == 96


def test_resolution_failure_is_design_only_and_has_no_leg_estimates(sample):
    rows = join_legs(*sample)
    # Residualization is allowed. The outcome may not enter the leg preflight.
    for r in rows: r[CANONICAL_OUTCOME_ID] = object()
    resolution = {leg: {"delta_usd_million": None, "status": "unestablished"} for leg in ("R", "J", "O")}
    result = leg_preflight(rows, resolution)
    assert result["status"] == "failed"
    assert result["reason_codes"] == [f"{leg}_source_resolution_unestablished" for leg in ("R", "J", "O")]
    assert not result["leg_estimates_computed"]
    assert result["control_rank"] == 17 and result["three_leg_rank"] == 20
    assert len(result["leg_diagnostics"]) == 3
    assert "beta" not in str(result)


def test_resolution_bound_needs_lineage_and_bites(sample):
    panel = join_legs(*sample)
    resolution = {leg: {"delta_usd_million": 0, "status": "established"} for leg in ("R", "J", "O")}
    assert leg_preflight(panel, resolution)["status"] == "failed"
    for r in resolution.values(): r["lineage"] = "fixture_exact_integer_source"
    assert leg_preflight(panel, resolution)["status"] == "passed"
    resolution["R"]["delta_usd_million"] = 1e6
    assert "R_residual_sd_not_above_10_delta" in leg_preflight(panel, resolution)["reason_codes"]


def test_three_leg_rank_and_condition_fail_without_control_drops(sample):
    panel = join_legs(*sample)
    for row in panel: row["J"] = row["R"]
    result = leg_preflight(panel, {})
    assert "three_leg_design_not_full_rank" in result["reason_codes"]
    assert "condition_number_exceeds_30" in result["reason_codes"]
    assert result["controls_dropped"] == []


def test_producer_frozen_gate_rejects_stale_factor_realizations(sample, tmp_path, monkeypatch):
    import csv
    import importlib.util
    import hashlib
    from pathlib import Path
    import sys

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("open16_runner_test", scripts / "run_open16_diagnostics.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    rows, _ = sample
    monkeypatch.setattr(runner, "_load_accepted_inputs", lambda root: (rows, [], {}))
    def table(name, records):
        p = tmp_path / name
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
        return {"path": name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    reference = _fit_projection(rows, treatment_id=CANONICAL_TREATMENT_ID, outcome_id=CANONICAL_OUTCOME_ID, control_ids=CANONICAL_CONTROL_IDS)
    h = table("headline.csv", [{"treatment_id": CANONICAL_TREATMENT_ID, "outcome_id": CANONICAL_OUTCOME_ID, "beta": reference.beta, "se": reference.se}])
    rolling = []
    for end in range(39, 96):
        window = rows[max(0, end - 47):end + 1]
        fit = _fit_projection(window, treatment_id=CANONICAL_TREATMENT_ID, outcome_id=CANONICAL_OUTCOME_ID, control_ids=CANONICAL_CONTROL_IDS)
        rolling.append({"outcome": CANONICAL_OUTCOME_ID, "horizon": "0", "n": str(len(window)), "window_start_quarter": window[0]["quarter"], "window_end_quarter": window[-1]["quarter"], "beta": fit.beta, "se": fit.se})
    r = table("rolling.csv", rolling)
    receipt = {"retained_outputs": {"fixed": {"same_quarter_headline": h, "rolling_estimates": r}}}
    _, _, gate = runner.validate_frozen_controls(tmp_path, rows, receipt)
    assert gate["rolling_windows_compared"] == 57
    assert gate["original_rolling_endpoints"] == list(FROZEN_QUARTERS[39:])
    path = pandemic_path(rows, nominal_endpoints=gate["original_rolling_endpoints"])
    assert [r["nominal_end"] for r in path] == gate["original_rolling_endpoints"]
    assert [r["n_obs"] for r in path[:8]] == list(range(40, 48))
    assert path[0]["nominal_start"] == "2000Q1"
    assert path[0]["first_observed"] == "2002Q1"
    stale = copy.deepcopy(rows)
    for i, row in enumerate(stale): row["dflmx_k100_f1"] += 0.7 * np.sin(i)
    # Base design never stored factor columns; equality must be checked against outputs.
    base = [{k: v for k, v in row.items() if not k.startswith("dflmx_")} for row in rows]
    monkeypatch.setattr(runner, "_load_accepted_inputs", lambda root: (base, [], {}))
    with pytest.raises(ValueError, match="frozen estimate equivalence"):
        runner.validate_frozen_controls(tmp_path, stale, receipt)
    assert not (tmp_path / "receipt.json").exists()
