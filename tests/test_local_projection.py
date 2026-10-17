from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ea_tdc import local_projection as lp
from ea_tdc import reserve_projection as rp
from ea_tdc.estimation import _ols
from ea_tdc.open16 import calendar_hac

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/local_projection_2026q2.json").read_text())
NUM = CONFIG["estimator"]["numerical"]


def synthetic(n: int = 60, lags: int = 1, seed: int = 3, drop: tuple[int, ...] = (), rows: bool = False) -> lp.Panel:
    """Outcome = 0.6 * treatment + controls + noise on a quarterly calendar, optionally with calendar gaps."""
    rng = np.random.default_rng(seed)
    quarters = [q for i, q in enumerate(rp.quarter_range("2002Q1", rp.quarter_label(lp._quarter_ordinal("2002Q1") + n - 1)))
                if i not in drop]
    m = len(quarters)
    x = rng.normal(0, 100, m)
    z = rng.normal(0, 1, (m, 3))
    y = 0.6 * x + z @ np.array([5.0, -3.0, 2.0]) + rng.normal(0, 40, m)
    ids = ("c1", "c2", "c3")
    raw = tuple({"quarter": q, "t": repr(float(x[i])), "y": repr(float(y[i])), **{c: repr(float(z[i, j])) for j, c in enumerate(ids)}}
                for i, q in enumerate(quarters)) if rows else ()
    return lp.Panel("synthetic", tuple(quarters), np.asarray([lp._quarter_ordinal(q) for q in quarters]), y, x, z, ids,
                    lags, "t", "y", raw)


def design(panel: lp.Panel, mask: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(int(mask.sum())), panel.x[mask], panel.controls[mask]])


@pytest.mark.parametrize("lags", [1, 4])
def test_uniform_window_is_the_admitted_ols_and_calendar_hac(lags):
    """A uniform window must be exactly the admitted rolling estimator, or the paths are not comparable."""
    panel = synthetic(lags=lags)
    weights = lp.trailing_weights(panel.ordinals, int(panel.ordinals[47]), 48)
    beta, se = lp.apply(lp.operator(panel, weights, NUM), panel.ordinals, panel.y, lags)
    mask = weights > 0
    fit = calendar_hac(panel.y[mask], design(panel, mask), [int(o) for o in panel.ordinals[mask]], lags=lags)
    assert beta == pytest.approx(fit.beta[1], rel=1e-12) and se == pytest.approx(fit.ses[1], rel=1e-10)
    if lags == 1:
        frozen = _ols(panel.y[mask].tolist(), design(panel, mask).tolist(), covariance_estimator="newey_west", covariance_lags=1)
        assert beta == pytest.approx(frozen.beta[1], rel=1e-10) and se == pytest.approx(frozen.ses[1], rel=1e-10)


def test_kernel_slope_and_sandwich_are_the_weighted_regression_on_weighted_scores():
    """The local slope is WLS and its variance uses scores w*x*e with the effective-sample scale."""
    panel = synthetic(lags=4)
    weights = lp.kernel_weights(panel.ordinals, int(panel.ordinals[30]), 24)
    op = lp.operator(panel, weights, NUM)
    beta, se = lp.apply(op, panel.ordinals, panel.y, 4)
    mask, w = weights > 0, weights[weights > 0]
    sw = np.sqrt(w)
    x, y = design(panel, mask) * sw[:, None], panel.y[mask] * sw
    assert beta == pytest.approx(np.linalg.lstsq(x, y, rcond=None)[0][1], rel=1e-10)
    scaled = calendar_hac(y, x, [int(o) for o in panel.ordinals[mask]], lags=4)
    n, k = x.shape
    n_eff = w.sum() ** 2 / (w ** 2).sum()
    assert se ** 2 == pytest.approx(scaled.ses[1] ** 2 * (n_eff / (n_eff - k)) / (n / (n - k)), rel=1e-9)
    assert op.n_eff == pytest.approx(n_eff) and op.n_eff < n


def test_weights_scale_free_and_estimates_local_in_calendar_time():
    """Rescaling weights changes nothing; a shock 24+ quarters away cannot move an estimate."""
    panel = synthetic(lags=4)
    target = int(panel.ordinals[20])
    weights = lp.kernel_weights(panel.ordinals, target, 24)
    base = lp.apply(lp.operator(panel, weights, NUM), panel.ordinals, panel.y, 4)
    assert lp.apply(lp.operator(panel, 7.3 * weights, NUM), panel.ordinals, panel.y, 4) == pytest.approx(base, rel=1e-12)
    far = np.flatnonzero(np.abs(panel.ordinals - target) >= 24)
    shocked = panel.y.copy()
    shocked[far] += 1e9
    assert lp.apply(lp.operator(panel, weights, NUM), panel.ordinals, shocked, 4) == pytest.approx(base, rel=1e-12)


def test_calendar_gaps_set_weights_and_hac_lags_by_quarter_distance():
    panel = synthetic(lags=4, drop=(25, 26))
    target = int(panel.ordinals[24])
    weights = lp.kernel_weights(panel.ordinals, target, 24)
    assert weights[25] == pytest.approx(lp.epanechnikov(np.array([3.0]), 24)[0])
    op = lp.operator(panel, np.ones(len(panel.quarters)), NUM)
    _, se = lp.apply(op, panel.ordinals, panel.y, 4)
    fit = calendar_hac(panel.y, design(panel, np.ones(len(panel.y), bool)), [int(o) for o in panel.ordinals], lags=4)
    assert se == pytest.approx(fit.ses[1], rel=1e-10)


def test_screen_withholds_thin_estimates_but_keeps_every_row():
    panel = synthetic(lags=1)
    flags = CONFIG["influence"]["flags"]
    rows = lp.local_rows(panel, 8, "sensitivity", flags, NUM)
    assert len(rows) == len(panel.quarters)
    thin = [r for r in rows if r["n_eff"] < r["screen_min_n_eff"]]
    assert thin and all(r["status"] == "effective_sample_below_screen" and r["beta"] is None for r in thin)
    wide = lp.local_rows(panel, 40, "primary", flags, NUM)
    reported = [r for r in wide if r["status"] == "reported"]
    assert reported and all(r["lower95"] < r["beta"] < r["upper95"] for r in reported)
    assert wide[0]["boundary"] and wide[0]["centroid_shift_quarters"] > 0 and wide[-1]["centroid_shift_quarters"] < 0


def test_frozen_rank_admission_drops_a_control_that_is_constant_on_the_support(monkeypatch):
    """Like the rolling windows, a control with no variation on the support is dropped, not fitted."""
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    panel = synthetic(rows=True)
    constant = [dict(r, c2="1.0") if i < 40 else r for i, r in enumerate(panel.rows)]
    controls = panel.controls.copy()
    controls[:40, 1] = 1.0
    panel = lp.Panel(panel.name, panel.quarters, panel.ordinals, panel.y, panel.x, controls, panel.control_ids,
                     panel.hac_lags, "t", "y", tuple(constant))
    op = lp.operator(panel, lp.trailing_weights(panel.ordinals, int(panel.ordinals[39]), 40), NUM)
    assert op.rejected == ("c2",) and op.admitted == ("c1", "c3") and op.design.shape[1] == 4
    assert lp.operator(panel, np.ones(len(panel.quarters)), NUM).admitted == panel.control_ids


def test_influence_shares_sum_to_one_and_rebuild_the_slope():
    panel = synthetic(lags=1)
    op = lp.operator(panel, lp.kernel_weights(panel.ordinals, int(panel.ordinals[30]), 24), NUM)
    share = op.weights * op.residual_x ** 2 / op.partial_variation
    assert share.sum() == pytest.approx(1.0)
    y = panel.y[op.mask]
    assert op.weights @ (op.residual_x * y) / op.partial_variation == pytest.approx(op.slope_row @ y, rel=1e-10)
    infl = lp.influence(op, panel.quarters)
    assert infl["max_share"] == pytest.approx(share.max()) and infl["n_f"] == pytest.approx(1 / (share ** 2).sum())


def test_calibration_targets_are_exact_projections_and_draws_are_reproducible():
    panel = synthetic(lags=1)
    config = copy.deepcopy(CONFIG["calibration"]) | {"replications": 200}
    targets = list(panel.quarters[20:23])
    nuisance = panel.controls[:, 0]
    rows = lp.calibrate(panel, 0, 24, targets, config, nuisance, NUM)
    assert len(rows) == len(targets) * 4 * 2 * 4 * 2
    assert rows == lp.calibrate(panel, 0, 24, targets, config, nuisance, NUM)
    means, _ = lp.means(panel, nuisance, NUM)
    for row in rows:
        op = lp.operator(panel, lp.kernel_weights(panel.ordinals, lp._quarter_ordinal(row["quarter"]), 24), NUM)
        assert row["target"] == pytest.approx(float(lp.apply(op, panel.ordinals, means[row["mean"]], 1)[0]), rel=1e-12)
        # The estimator is linear in y on a fixed design, so it is unbiased for its projection target.
        assert abs(row["mean_estimate"] - row["target"]) < 5 * row["sd_estimate"] / np.sqrt(200)
    assert set(lp.coverage_status(rows)) == set(targets)


def test_parity_rejects_any_departure_from_the_admitted_rolling_rows():
    panel = synthetic(lags=4)
    rolling = []
    for end in panel.ordinals[47:]:
        op = lp.operator(panel, lp.trailing_weights(panel.ordinals, int(end), 48), NUM)
        beta, se = lp.apply(op, panel.ordinals, panel.y, 4)
        rolling.append({"end": rp.quarter_label(int(end)), "n": 48, "beta": float(beta), "se": float(se)})
    full = dict(zip(("beta", "se"), map(float, lp.apply(lp.operator(panel, np.ones(60), NUM), panel.ordinals, panel.y, 4))))
    tolerance = CONFIG["parity_tolerance"]
    assert lp.parity(panel, rolling, full, 48, tolerance, NUM)["rolling_rows"] == len(rolling)
    with pytest.raises(ValueError, match="parity fails"):
        lp.parity(panel, [dict(rolling[0], beta=rolling[0]["beta"] * 1.001), *rolling[1:]], full, 48, tolerance, NUM)
    with pytest.raises(ValueError, match="rows"):
        lp.parity(panel, [dict(rolling[0], n=47)], full, 48, tolerance, NUM)


def test_pinned_inputs_must_match_their_declared_hashes(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n")
    record = {"path": "a.csv", "sha256": hashlib.sha256(b"x\n1\n").hexdigest(), "bytes": 4}
    assert lp.pinned(tmp_path, record) == tmp_path / "a.csv"
    with pytest.raises(ValueError, match="differs"):
        lp.pinned(tmp_path, dict(record, sha256="0" * 64))
    with pytest.raises(ValueError, match="missing"):
        lp.pinned(tmp_path, dict(record, path="b.csv"))


def test_runner_writes_both_panels_and_a_receipt_binding_every_output(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("local_runner_test", ROOT / "scripts" / "run_local_projection.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    deposit, reserve = synthetic(lags=1, seed=5), synthetic(lags=4, seed=6)
    deposit = lp.Panel("deposit", *[getattr(deposit, f) for f in ("quarters", "ordinals", "y", "x", "controls")],
                       ("GDP", "c2", "c3"), 1)
    reserve = lp.Panel("reserve", *[getattr(reserve, f) for f in ("quarters", "ordinals", "y", "x", "controls")],
                       ("quarter_of_year_q2", "c2", "c3"), 4)

    def parity_inputs(panel):
        ones = lp.operator(panel, np.ones(len(panel.quarters)), NUM)
        beta, se = lp.apply(ones, panel.ordinals, panel.y, panel.hac_lags)
        return [], {"beta": float(beta), "se": float(se)}

    config = copy.deepcopy(CONFIG)
    config["calibration"]["replications"] = 50
    (tmp_path / "config").mkdir()
    (tmp_path / "config/local.json").write_text(json.dumps(config))
    (tmp_path / "uv.lock").write_text("lock")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "_verify_producer_commit", lambda root, supplied: "a" * 40)
    monkeypatch.setattr(runner.lp, "load_deposit", lambda root, spec: deposit)
    monkeypatch.setattr(runner.lp, "load_reserve", lambda root, spec: (reserve, SimpleNamespace(sha256="d"), SimpleNamespace(checks={})))
    monkeypatch.setattr(runner.rp, "verify_frozen_sources", lambda root, design: [])
    monkeypatch.setattr(runner, "deposit_parity_inputs", lambda root, spec: parity_inputs(deposit))
    monkeypatch.setattr(runner, "reserve_parity_inputs", lambda root, spec, design, source: parity_inputs(reserve))
    args = SimpleNamespace(producer_commit="a" * 40, config="config/local.json", output_dir="output/local_test")
    receipt_path = runner.run(args)
    receipt = json.loads(receipt_path.read_text())
    for name, record in receipt["outputs"].items():
        assert hashlib.sha256((receipt_path.parent / name).read_bytes()).hexdigest() == record["sha256"]
    with (receipt_path.parent / config["outputs"]["estimates"]).open() as handle:
        rows = list(__import__("csv").DictReader(handle))
    assert {(r["panel"], r["half_width"]) for r in rows} == {(p, str(h)) for p in ("deposit", "reserve") for h in (24, 16, 32)}
    primary = [r for r in rows if r["role"] == "primary" and r["status"] == "reported"]
    assert primary and all(r["coverage_status"].startswith("coverage_") and r["coverage_status"] != "coverage_unverified" for r in primary)
    assert all(r["coverage_status"] == "coverage_unverified" for r in rows if r["role"] == "sensitivity" and r["status"] == "reported")
    with pytest.raises(ValueError, match="already exists"):
        runner.run(args)


def test_ill_conditioned_design_is_solved_accurately_or_withheld():
    """Near-collinear treatment and control: never report a slope the normal equations would corrupt."""
    rng = np.random.default_rng(43)
    panel = synthetic()
    z = rng.standard_normal(60)
    x = z + 1e-8 * rng.standard_normal(60)
    y = 0.6 * x + 0.4 * z + 1e-8 * rng.standard_normal(60)
    panel = lp.Panel("synthetic", panel.quarters, panel.ordinals, y, x, z[:, None], ("z",), 1)
    weights = lp.kernel_weights(panel.ordinals, int(panel.ordinals[30]), 24)
    op = lp.operator(panel, weights, NUM)
    mask = weights > 0
    sw = np.sqrt(weights[mask])
    reference = np.linalg.lstsq(np.column_stack([np.ones(mask.sum()), x[mask], z[mask]]) * sw[:, None], y[mask] * sw,
                                rcond=None)[0][1]
    beta, se = (float(v) for v in lp.apply(op, panel.ordinals, panel.y, 1))
    status = lp.screen(op, beta, se)
    assert status == "numerically_unresolved" or beta == pytest.approx(reference, rel=1e-6)
    strict = lp.operator(panel, weights, dict(NUM, min_reciprocal_condition=1e-6))
    assert lp.screen(strict, beta, se) == "numerically_unresolved"


def test_parity_rejects_unresolved_or_nonfinite_estimates_on_either_side():
    panel = synthetic(lags=1)
    flat = panel.x.copy()
    flat[:48] = 1.0
    flat_panel = lp.Panel("synthetic", panel.quarters, panel.ordinals, panel.y, flat, panel.controls, panel.control_ids, 1)
    full = dict(zip(("beta", "se"), map(float, lp.apply(lp.operator(flat_panel, np.ones(60), NUM), flat_panel.ordinals, flat_panel.y, 1))))
    tolerance = CONFIG["parity_tolerance"]
    with pytest.raises(ValueError, match="not a resolved"):
        lp.parity(flat_panel, [{"end": panel.quarters[47], "n": 48, "beta": 123.0, "se": 45.0}], full, 48, tolerance, NUM)
    good = lp.apply(lp.operator(panel, lp.trailing_weights(panel.ordinals, int(panel.ordinals[47]), 48), NUM), panel.ordinals, panel.y, 1)
    row = {"end": panel.quarters[47], "n": 48, "beta": float(good[0]), "se": float(good[1])}
    whole = dict(zip(("beta", "se"), map(float, lp.apply(lp.operator(panel, np.ones(60), NUM), panel.ordinals, panel.y, 1))))
    assert lp.parity(panel, [row], whole, 48, tolerance, NUM)["beta"] < 1e-12
    for bad in ({"beta": math.nan}, {"se": math.nan}, {"se": 0.0}):
        with pytest.raises(ValueError, match="nonfinite"):
            lp.parity(panel, [dict(row, **bad)], whole, 48, tolerance, NUM)


def test_coverage_is_qualified_only_when_clearly_above_the_floor():
    config = CONFIG["calibration"]
    assert lp.classify(0.888, 0.00705, config) == "coverage_inconclusive_specified_dgp"
    assert lp.classify(0.95, math.sqrt(0.95 * 0.05 / 2000), config) == "coverage_qualified_specified_dgp"
    assert lp.classify(0.85, math.sqrt(0.85 * 0.15 / 2000), config) == "coverage_failed_specified_dgp"
    base = {"quarter": "2010Q1", "mean": "pooled", "scale": "homoskedastic", "ar1": "ar1_0", "innovation": "gaussian"}
    qualified = dict(base, coverage=0.95, status="coverage_qualified_specified_dgp")
    unsure = dict(base, coverage=0.90, status="coverage_inconclusive_specified_dgp")
    failed = dict(base, coverage=0.80, status="coverage_failed_specified_dgp")
    assert lp.coverage_status([qualified, qualified])["2010Q1"]["coverage_status"] == "coverage_qualified_specified_dgp"
    assert lp.coverage_status([qualified, unsure])["2010Q1"]["coverage_status"] == "coverage_inconclusive_specified_dgp"
    status = lp.coverage_status([qualified, unsure, failed])["2010Q1"]
    assert status["coverage_status"] == "coverage_failed_specified_dgp" and status["coverage_min"] == 0.80
    assert (status["coverage_dgps_failed"], status["coverage_dgps_inconclusive"]) == (1, 1)


def test_ar1_errors_follow_the_declared_recurrence_and_unit_variance():
    seq = np.random.SeedSequence([1, 2, 3])
    path = lp.ar1_errors(np.random.default_rng(seq), 30, 5, 0.8, {"id": "gaussian"}, 10)
    shocks = np.random.default_rng(np.random.SeedSequence([1, 2, 3])).standard_normal((40, 5))
    manual = np.empty_like(shocks)
    manual[0] = shocks[0]
    for t in range(1, 40):
        manual[t] = 0.8 * manual[t - 1] + 0.6 * shocks[t]
    assert path == pytest.approx(manual[10:])
    heavy = lp.ar1_errors(np.random.default_rng(5), 2, 200_000, 0.0, {"id": "student_t4", "df": 4}, 0)
    assert heavy.var() == pytest.approx(1.0, abs=0.05)


def test_mean_classes_add_the_declared_slope_and_nuisance_paths():
    panel = synthetic()
    nuisance = panel.controls[:, 0]
    means, residuals = lp.means(panel, nuisance, NUM)
    assert list(means) == list(lp.MEAN_CLASSES)
    assert panel.y - residuals == pytest.approx(means["pooled"])
    steps = (means["slope_steps"] - means["pooled"]) / panel.x
    for quarter, step in zip(panel.quarters, steps):
        assert step == pytest.approx(0.5 if "2008Q4" <= quarter < "2020Q1" else 0.0, abs=1e-9)
    drift = (means["slope_drift"] - means["pooled"]) / panel.x
    assert (drift[0], drift[-1]) == pytest.approx((-0.5, 0.5))


def test_validate_rejects_declarations_the_estimator_would_ignore():
    lp.validate(CONFIG)
    for path, value in ((("estimator", "kernel"), "gaussian"), (("calibration", "bandwidths"), "all"),
                        (("estimator", "sensitivity_half_widths"), [24]), (("calibration", "minimum_acceptable"), 0.99)):
        bad = copy.deepcopy(CONFIG)
        bad[path[0]][path[1]] = value
        with pytest.raises(ValueError):
            lp.validate(bad)
    bad = copy.deepcopy(CONFIG)
    bad["calibration"]["scales"].append({"id": "garch"})
    with pytest.raises(ValueError):
        lp.validate(bad)


def _chain(tmp_path):
    """A minimal sealed receipt -> lane receipt -> panel receipt -> panel chain."""
    def record(path):
        data = (tmp_path / path).read_bytes()
        return {"path": path, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    refresh = tmp_path / "s/run/output/refresh"
    refresh.mkdir(parents=True)
    quarters = rp.quarter_range("2002Q1", "2004Q4")
    lines = ["quarter,t,y,c1"] + [f"{q},{i + 1.5},{2 * i},{(-1) ** i}" for i, q in enumerate(quarters)]
    (refresh / "panel.csv").write_text("\n".join(lines) + "\n")
    panel = record("s/run/output/refresh/panel.csv")
    (refresh / "panel_receipt.json").write_text(json.dumps({"panel": panel, "control_ids": ["c1"]}))
    panel_receipt = record("s/run/output/refresh/panel_receipt.json")
    (tmp_path / "s/open01_receipt.json").write_text(json.dumps(
        {"panel_receipt": dict(panel_receipt, path="output/refresh/panel_receipt.json")}))
    lane = record("s/open01_receipt.json")
    (tmp_path / "s/receipt.json").write_text(json.dumps({"lanes": {"open01": dict(lane, path="open01_receipt.json")}}))
    return {"sealed_refresh_receipt": record("s/receipt.json"), "panel_receipt": panel_receipt, "panel": panel,
            "treatment": "t", "outcome": "y", "controls": ["c1"], "hac_lags": 1,
            "sample": {"start": "2002Q1", "end": "2004Q4", "n": 12}}


def test_deposit_loader_follows_the_sealed_receipt_chain_and_rejects_tampering(tmp_path):
    spec = _chain(tmp_path)
    panel = lp.load_deposit(tmp_path, spec)
    assert panel.quarters[0] == "2002Q1" and panel.x[0] == 1.5 and panel.control_ids == ("c1",)
    lane = json.loads((tmp_path / "s/open01_receipt.json").read_text())
    lane["panel_receipt"]["sha256"] = "0" * 64
    (tmp_path / "s/open01_receipt.json").write_text(json.dumps(lane))
    with pytest.raises(ValueError, match="lane receipt differs"):
        lp.load_deposit(tmp_path, spec)
    spec = _chain(tmp_path / "again")
    (tmp_path / "again/s/run/output/refresh/panel.csv").write_text("quarter,t,y,c1\n")
    with pytest.raises(ValueError, match="differs from its declared hash"):
        lp.load_deposit(tmp_path / "again", spec)
    spec = _chain(tmp_path / "third")
    with pytest.raises(ValueError, match="control order"):
        lp.load_deposit(tmp_path / "third", dict(spec, controls=["c2"]))
