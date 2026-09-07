from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from ea_tdc import open16
from ea_tdc.covariance import COVARIANCE_OPERATOR_POLICY, canonical_covariance
from ea_tdc.estimation import _ols
from ea_tdc.open01 import _fit_projection
from ea_tdc.open16 import (
    CANONICAL_CONTROL_IDS,
    CANONICAL_OUTCOME_ID,
    CANONICAL_TREATMENT_ID,
    DELETED_QUARTERS,
    FROZEN_QUARTERS,
    calendar_hac,
    covariance_contributions,
    join_legs,
    leg_preflight,
    pandemic_path,
    validate_panel,
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
    np.testing.assert_allclose(actual.raw_covariance, expected, rtol=1e-12, atol=1e-12)
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


@pytest.mark.parametrize("raw,message", [
    ([[1, 2], [2, 1]], "materially indefinite"),
    ([[-1e-20, 0], [0, 1]], "Negative reported covariance diagonal"),
    ([[1, 0], [0, 0]], "Treatment variance must be strictly positive"),
])
def test_calendar_consumer_rejects_invalid_raw_sandwich(monkeypatch, raw, message):
    calls = []

    def injected_sandwich(actual_raw, scales, outcome_scale):
        calls.append(np.asarray(actual_raw).copy())
        return canonical_covariance(raw, scales, outcome_scale)

    monkeypatch.setattr(open16, "canonical_covariance", injected_sandwich)
    with pytest.raises(ValueError, match=message):
        calendar_hac([3, 10, -2, 12, 3], [[1, v] for v in [0, 1, 4, 7, 9]], range(5))
    assert len(calls) == 1
    assert calls[0].shape == (2, 2)


def test_calendar_consumer_preserves_raw_and_reports_unclipped_canonical(monkeypatch):
    raw = np.array([[1, 1e-10], [0, 2]], dtype=float)
    original = raw.copy()

    def injected_sandwich(actual_raw, scales, outcome_scale):
        return canonical_covariance(raw, scales, outcome_scale)

    monkeypatch.setattr(open16, "canonical_covariance", injected_sandwich)
    fit = calendar_hac([3, 10, -2, 12, 3], [[1, v] for v in [0, 1, 4, 7, 9]], range(5))
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(fit.raw_covariance, original)
    np.testing.assert_array_equal(np.diag(fit.covariance), np.diag(original))
    np.testing.assert_array_equal(fit.ses, np.sqrt(np.diag(original)))
    np.testing.assert_allclose(fit.covariance, (original + original.T) / 2, rtol=1e-15)
    assert fit.covariance_diagnostics["raw_skew_relative_norm"] > 0
    assert fit.covariance_diagnostics["minimum_scaled_eigenvalue"] > 0
    assert fit.covariance_diagnostics["policy"] == COVARIANCE_OPERATOR_POLICY
    assert len(fit.covariance_diagnostics["scales"]) == 2
    assert fit.covariance_diagnostics["outcome_scale"] > 0


def test_pandemic_endpoints_no_refill_and_constant_control_policy(sample):
    rows, _ = sample
    # The frozen OPEN01 rule rejects controls constant inside a window.
    for i, row in enumerate(rows):
        row[CANONICAL_CONTROL_IDS[4]] = float(i < 33)
    result = pandemic_path(rows)
    assert len(result) == 57
    assert [r["n_obs"] for r in result[:8]] == list(range(40, 48))
    assert [r["nominal_start"] for r in result[:8]] == [f"{2000 + i // 4}Q{i % 4 + 1}" for i in range(8)]
    for q, n, last in [("2020Q1", 47, "2019Q4"), ("2020Q4", 44, "2019Q4"), ("2022Q3", 43, "2022Q3"), ("2025Q4", 43, "2025Q4")]:
        found = next(r for r in result if r["nominal_end"] == q)
        assert found["n_obs"] == n
        assert found["last_observed"] == last
        assert found["prespecified_endpoint"]
        expected_rank = 18 if q in ("2020Q1", "2020Q4") else 17
        assert found["rank"] == expected_rank
        assert found["controls_rejected"] == (CANONICAL_CONTROL_IDS[4] if expected_rank == 17 else "")
        assert found["finite_sample_scale"] == n / (n - expected_rank)
        evidence = found["covariance_evidence"]
        assert np.asarray(evidence["raw_covariance"]).shape == (expected_rank, expected_rank)
        np.testing.assert_array_equal(np.diag(evidence["covariance"]), np.diag(evidence["raw_covariance"]))
        assert found["se"] == np.sqrt(evidence["covariance"][1][1])
        assert len(evidence["diagnostics"]["scales"]) == expected_rank
        assert evidence["diagnostics"]["policy"] == COVARIANCE_OPERATOR_POLICY
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


@pytest.fixture
def frozen_gate(sample, tmp_path, monkeypatch):
    import csv
    import hashlib
    import importlib.util
    from pathlib import Path

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
        start = 2002 * 4 + end - 47
        rolling.append({"outcome": CANONICAL_OUTCOME_ID, "treatment_id": CANONICAL_TREATMENT_ID,
                        "horizon": "0", "n": str(len(window)), "window_quarters": "48",
                        "window_start_quarter": f"{start // 4}Q{start % 4 + 1}",
                        "window_end_quarter": window[-1]["quarter"],
                        "effective_sample_start": window[0]["quarter"], "effective_sample_end": window[-1]["quarter"],
                        "covariance_estimator": "newey_west", "covariance_lags": "1",
                        "control_ids_used": ",".join(fit.control_ids_used), "dropped_control_ids": ",".join(fit.control_ids_rejected),
                        "beta": fit.beta, "se": fit.se})
    r = table("rolling.csv", rolling)
    receipt = {"retained_outputs": {"fixed": {"same_quarter_headline": h, "rolling_estimates": r}}}
    return runner, rows, receipt, rolling, table


def test_calendar_sidecar_retains_raw_and_operator_evidence(frozen_gate, tmp_path):
    runner, _, _, _, _ = frozen_gate
    operator = canonical_covariance([[1, 1e-10], [0, 2]], [2, 3], 5)
    diagnostics = operator["diagnostics"] | {"scales": [2, 3], "outcome_scale": 5}
    original = [{"nominal_end": "2025Q4", "beta": .5, "covariance_evidence": {
        "raw_covariance": operator["raw_covariance"], "covariance": operator["covariance"],
        "diagnostics": diagnostics,
    }}]
    compact = copy.deepcopy(original)
    summary = runner._write_calendar_evidence(tmp_path, compact)
    assert "covariance_evidence" in original[0]
    assert compact == [{"nominal_end": "2025Q4", "beta": .5}]
    retained = json.loads((tmp_path / "calendar_covariance_evidence.json").read_text())
    assert retained == [{"nominal_end": "2025Q4", **original[0]["covariance_evidence"]}]
    assert summary == [{"nominal_end": "2025Q4", **diagnostics}]


def test_producer_frozen_gate_rejects_stale_factor_realizations(frozen_gate, tmp_path, monkeypatch):
    runner, rows, receipt, _, _ = frozen_gate
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


@pytest.mark.parametrize("mutation", ["missing_first", "duplicate", "window_start_quarter", "n", "covariance_lags", "window_quarters", "treatment_id", "control_ids_used"])
def test_frozen_gate_rejects_changed_rolling_contract(frozen_gate, tmp_path, mutation):
    runner, rows, receipt, rolling, table = frozen_gate
    if mutation == "missing_first":
        rolling.pop(0)
    elif mutation == "duplicate":
        rolling[1] = rolling[0].copy()
    else:
        rolling[0][mutation] = "changed"
    # Refresh the self-hash so this tests semantic identity, not byte corruption.
    receipt["retained_outputs"]["fixed"]["rolling_estimates"] = table("rolling.csv", rolling)
    with pytest.raises(ValueError, match="rolling"):
        runner.validate_frozen_controls(tmp_path, rows, receipt)


@pytest.fixture
def authority(tmp_path, monkeypatch):
    import importlib.util
    import json
    from pathlib import Path
    from types import SimpleNamespace

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("open16_authority_test", scripts / "run_open16_diagnostics.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True))
        return {"path": name, "sha256": runner._sha256_file(path), "bytes": path.stat().st_size}

    pins = {"factor_origin": "retained_original_coordinates",
            "accepted_source_commit": runner.EXPECTED_OPEN01_PRODUCER_COMMIT,
            "accepted_source_tree": "a" * 40, "factor_policy": runner.FACTOR_POLICY,
            "upstream_producer_commit": "b" * 40}
    for key in ("panel", "full_factor_scores", "sample_factor_scores", "ordered_screening", "ordered_top100", "equivalence", "legs"):
        pins[key] = write(key + ".json", {"synthetic": key})
    pins["open01_receipt"] = write(runner.OPEN01_RECEIPT_LOCATOR, {"synthetic": "accepted"})
    pins["graph_cut"] = {"kind": "complete_historical_raw_universe", "files": [write("raw.csv", {"synthetic": "raw"})],
                         "completeness_evidence": write("completeness.json", {"synthetic": "complete"})}
    pins["runtime"] = {"kind": "historical_runtime_identity", "evidence": write("runtime.json", {"synthetic": "runtime"})}
    pins["legs_receipt"] = write("legs_receipt.json", {"producer_commit": "b" * 40,
        "schema_version": "regression_legs_v1", "output": {"sha256": pins["legs"]["sha256"]},
        "units": "USD million", "sample": {"start": "2002Q1", "end": "2025Q4", "n": 96}})
    gate = {"schema_version": "ea_tdc_open16_authority_v1", "status": "approved", "approved_inputs": pins}

    def approve():
        # Synthetic approval only. Production config remains unavailable.
        pins["provenance_receipt"] = write("provenance.json", {k: v for k, v in pins.items() if k != "provenance_receipt"})
        gate["approval_receipt"] = write("approval.json", {"decision": "approved", "scope": "exact_factor_coordinates",
                                                        "approved_inputs_sha256": runner._json_digest(pins)})
        write(runner.AUTHORITY_LOCATOR, gate)
    approve()
    committed = (tmp_path / runner.AUTHORITY_LOCATOR).read_bytes()

    def git(args, **kwargs):
        if args[1] == "show":
            return SimpleNamespace(stdout=committed)
        return SimpleNamespace(stdout="a" * 40 + "\n")
    monkeypatch.setattr(runner.subprocess, "run", git)

    def commit():
        nonlocal committed
        write(runner.AUTHORITY_LOCATOR, gate)
        committed = (tmp_path / runner.AUTHORITY_LOCATOR).read_bytes()
    return runner, pins, gate, write, approve, commit


@pytest.mark.parametrize("origin", ["retained_original_coordinates", "deterministically_restored_accepted_graph"])
def test_provenance_accepts_only_independently_approved_coordinate_origins(authority, tmp_path, origin):
    runner, pins, _, _, approve, commit = authority
    pins["factor_origin"] = origin
    approve(); commit()
    assert runner.load_authority(tmp_path, "c" * 40)["factor_origin"] == origin


@pytest.mark.parametrize("mutation", ["panel_self_receipt", "origin", "graph", "accepted_commit", "accepted_tree", "runtime", "factor_scores", "screening", "legs", "leg_receipt", "upstream_producer", "equivalence", "approval", "uncommitted_gate"])
def test_provenance_rejects_independent_mutations(authority, tmp_path, mutation):
    runner, pins, gate, write, _, commit = authority
    if mutation == "panel_self_receipt":
        write(pins["panel"]["path"], {"new": "panel"})
        write(pins["provenance_receipt"]["path"], {"panel_sha256": runner._sha256_file(tmp_path / pins["panel"]["path"])})
    elif mutation in {"origin", "accepted_commit", "accepted_tree", "upstream_producer"}:
        key = {"origin": "factor_origin", "accepted_commit": "accepted_source_commit", "accepted_tree": "accepted_source_tree", "upstream_producer": "upstream_producer_commit"}[mutation]
        pins[key] = "changed"
        # Even a newly pinned approval cannot expand the hard accepted contract.
        gate["approval_receipt"] = write("approval.json", {"decision": "approved", "scope": "exact_factor_coordinates", "approved_inputs_sha256": runner._json_digest(pins)})
        commit()
    elif mutation == "uncommitted_gate":
        gate["status"] = "unavailable"
        write(runner.AUTHORITY_LOCATOR, gate)
    else:
        record = {"graph": pins["graph_cut"]["files"][0], "runtime": pins["runtime"]["evidence"],
                  "factor_scores": pins["full_factor_scores"], "screening": pins["ordered_screening"],
                  "legs": pins["legs"], "leg_receipt": pins["legs_receipt"], "equivalence": pins["equivalence"],
                  "approval": gate["approval_receipt"]}[mutation]
        write(record["path"], {"changed": mutation})
    with pytest.raises(ValueError):
        runner.load_authority(tmp_path, "c" * 40)


@pytest.mark.parametrize("origin", ["reselected", "refreshed_inputs", "estimate_matched_only", "equivalent_factor_space", "retained_accepted_realizations"])
def test_restoration_rejects_unapproved_recovery_claims_even_with_integrity(authority, tmp_path, origin):
    runner, pins, _, _, approve, commit = authority
    pins["factor_origin"] = origin
    approve(); commit()
    with pytest.raises(ValueError, match="origin"):
        runner.load_authority(tmp_path, "c" * 40)


def test_provenance_unavailable_stops_before_calculation_or_destination(authority, tmp_path, monkeypatch):
    from types import SimpleNamespace
    runner, _, gate, _, _, commit = authority
    gate.update(status="unavailable", approved_inputs=None, approval_receipt=None)
    commit()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "_verify_producer_commit", lambda *args: "c" * 40)
    monkeypatch.setattr(runner, "validate_frozen_controls", lambda *args: pytest.fail("unavailable authority reached calculation"))
    with pytest.raises(ValueError, match="authority unavailable"):
        runner.run(SimpleNamespace(producer_commit="c" * 40, output_dir="absent/outputs"))
    assert not (tmp_path / "absent").exists()


@pytest.fixture
def coordinate_evidence(authority, sample, tmp_path):
    import csv

    from ea_tdc.open01 import _quarter_ordinal
    runner, pins, _, write, _, _ = authority
    rows = copy.deepcopy(sample[0])
    for row in rows:
        for key in runner.FACTOR_IDS:
            row[key] = float(f"{row[key]:.10f}")
    # Accepted full-grid shape has an initial gap; only the 96-row sample is contiguous.
    full_quarters = ["1945Q4"] + [f"{i // 4}Q{i % 4 + 1}" for i in range(1946 * 4 + 3, 2026 * 4 + 2)]
    by_quarter = {r["quarter"]: r for r in rows}
    full = [{"quarter": q, **{k: f"{by_quarter.get(q, {}).get(k, 0):.10f}" for k in runner.FACTOR_IDS}} for q in full_quarters]
    selected = [r for r in full if r["quarter"] in FROZEN_QUARTERS]
    for key, records in (("full_factor_scores", full), ("sample_factor_scores", selected)):
        path = tmp_path / pins[key]["path"]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
        pins[key].update(sha256=runner._sha256_file(path), bytes=path.stat().st_size)
    pins["full_factor_quarters"] = full_quarters
    reference = {"panel_sha256": pins["panel"]["sha256"], "original_rolling_endpoints": list(runner.ORIGINAL_ROLLING_ENDPOINTS), "windows": []}
    for endpoint in runner.ORIGINAL_ROLLING_ENDPOINTS:
        end = _quarter_ordinal(endpoint)
        nominal = [f"{i // 4}Q{i % 4 + 1}" for i in range(end - 47, end + 1)]
        window = {"nominal_quarters": nominal}
        for lane in ("no_deletion", "calendar_deletion"):
            observed = [r for r in rows if r["quarter"] in nominal and (lane == "no_deletion" or r["quarter"] not in DELETED_QUARTERS)]
            estimate = _fit_projection(observed, treatment_id=CANONICAL_TREATMENT_ID, outcome_id=CANONICAL_OUTCOME_ID, control_ids=CANONICAL_CONTROL_IDS)
            fit = estimate.fit
            covariance = np.asarray(fit.covariance)
            if lane == "calendar_deletion":
                # Independent full-grid zero-score reference, not calendar_hac.
                x = np.asarray([[1, r[CANONICAL_TREATMENT_ID], *[r[k] for k in estimate.control_ids_used]] for r in observed])
                observed_scores = dict(zip((r["quarter"] for r in observed), x * np.asarray(fit.residuals)[:, None], strict=True))
                scores = np.asarray([observed_scores.get(q, np.zeros(x.shape[1])) for q in nominal])
                meat = scores.T @ scores + .5 * (scores[1:].T @ scores[:-1] + scores[:-1].T @ scores[1:])
                bread = np.linalg.inv(x.T @ x)
                covariance = bread @ meat @ bread * len(x) / (len(x) - x.shape[1])
            window[lane] = {"observed_quarters": [r["quarter"] for r in observed],
                            "controls_used": list(estimate.control_ids_used), "controls_rejected": list(estimate.control_ids_rejected),
                            "covariance_lags": 1, "coefficients": fit.beta, "fitted": fit.fitted,
                            "residuals": fit.residuals, "covariance": covariance.tolist(),
                            "treatment_beta_se": [fit.beta[1], float(np.sqrt(covariance[1, 1]))]}
        reference["windows"].append(window)
    pins["equivalence"] = write(pins["equivalence"]["path"], reference)
    return runner, pins, rows, reference, write


def test_restoration_accepts_approved_irregular_grid_and_full_vector_calendar_reference(coordinate_evidence, tmp_path):
    runner, pins, rows, _, _ = coordinate_evidence
    runner.validate_coordinate_evidence(tmp_path, rows, pins)


@pytest.mark.parametrize("mutation", ["missing_first", "nominal_inventory", "observed_inventory", "control_partition", "coefficient_vector", "residual_vector", "full_covariance", "factor_coordinate"])
def test_restoration_rejects_changed_equivalence_despite_estimate_match(coordinate_evidence, tmp_path, mutation):
    runner, pins, rows, reference, write = coordinate_evidence
    if mutation == "missing_first":
        reference["windows"].pop(0)
    elif mutation == "nominal_inventory":
        reference["windows"][0]["nominal_quarters"][0] = "2001Q1"
    elif mutation == "factor_coordinate":
        rows[0][runner.FACTOR_IDS[0]] *= -1
    else:
        lane = reference["windows"][0]["calendar_deletion"]
        if mutation == "observed_inventory": lane["observed_quarters"].pop(0)
        elif mutation == "control_partition": lane["controls_used"].reverse()
        elif mutation == "coefficient_vector": lane["coefficients"][-1] += .1
        elif mutation == "residual_vector": lane["residuals"][0] += .1
        else: lane["covariance"][-1][-1] += .1
    # Deliberately retain treatment beta/SE; those scalars cannot identify factors.
    pins["equivalence"] = write(pins["equivalence"]["path"], reference)
    with pytest.raises(ValueError):
        runner.validate_coordinate_evidence(tmp_path, rows, pins)


@pytest.mark.parametrize("mutation", ["reordered", "duplicate"])
def test_restoration_rejects_nonunique_or_reordered_approved_grid(coordinate_evidence, tmp_path, mutation):
    import csv

    runner, pins, rows, _, _ = coordinate_evidence
    path = tmp_path / pins["full_factor_scores"]["path"]
    with path.open() as handle:
        scores = list(csv.DictReader(handle))
    if mutation == "reordered":
        scores[0], scores[1] = scores[1], scores[0]
    else:
        scores[0] = scores[1].copy()
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scores[0]))
        writer.writeheader(); writer.writerows(scores)
    # An internally matching pin cannot turn a duplicate or reversed grid into the
    # exact ordered historical coordinate object.
    pins["full_factor_quarters"] = [r["quarter"] for r in scores]
    pins["full_factor_scores"].update(sha256=runner._sha256_file(path), bytes=path.stat().st_size)
    with pytest.raises(ValueError, match="320-quarter grid"):
        runner.validate_coordinate_evidence(tmp_path, rows, pins)
