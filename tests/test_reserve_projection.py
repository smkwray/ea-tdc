from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ea_tdc import reserve_projection as rp
from ea_tdc.estimation import _ols
from ea_tdc.open16 import calendar_hac

ROOT = Path(__file__).resolve().parents[1]
H41_TREASURY = ("treasury_general_account", "supplementary_financing_account", "treasury_securities")
SIDES = {"rrp_foreign_official": "absorbing", "foreign_official_deposits": "absorbing", "rrp_others": "absorbing",
         "currency_in_circulation": "absorbing", "treasury_general_account": "absorbing"}
REFRESH_CONFIG = "config/reserve_projection_2026q2.json"


@pytest.fixture(scope="module")
def design() -> rp.Design:
    return rp.load_design(ROOT)


def small(design: rp.Design, *, replicates: int = 19, sims: int = 4, inner: int = 19) -> rp.Design:
    raw = copy.deepcopy(design.raw)
    raw["bootstrap"]["replicates"] = replicates
    raw["calibration"]["replications"] = sims
    raw["calibration"]["bootstrap_replicates"] = inner
    return dataclasses.replace(design, raw=raw)


def synthetic(design: rp.Design, *, seed: int = 7, scale: float = 1.0, offset: float = -0.3,
              quiet: bool = False, foreign: float = 0.0, end: str | None = None) -> tuple[list, list, list, dict]:
    """Pinned-input lookalikes obeying the TDCest identity, in USD million.

    Currency in circulation drains `offset` dollars per TRC dollar, so the
    measured reserve slope moves from one by exactly that covariance. Foreign
    official RRP balances rise `foreign` dollars per TRC dollar; as an absorbing
    factor the ledger signs that as minus the level change. `quiet` removes
    every other factor, bridge and rounding term. The shell ends at the
    design's sample end unless `end` truncates or extends it.
    """
    rng = np.random.default_rng(seed)
    quarters = rp.quarter_range("2002Q1", end or design.sample[-1])
    n = len(quarters)
    sample = set(design.sample)
    factor_keys = [c[2:] for g in design.groups for c in g.components if c.startswith("f_")] + ["rrp_others"]
    c_m, b = rng.normal(0, 50_000, n), rng.normal(10_000, 20_000, n)
    route_on = np.asarray([q >= "2016Q2" for q in quarters])
    routes = {"mmf_route_prop": route_on * rng.gamma(2.0, 8_000, n)}
    routes["mmf_route_lb"] = routes["mmf_route_prop"] * 0.8
    routes["mmf_route_ub"] = routes["mmf_route_prop"] * 1.15
    factors = {k: rng.normal(0, 4_000, n) * (rng.random(n) < 0.6) for k in factor_keys}
    factors["currency_in_circulation"] = offset * (c_m + b) + rng.normal(0, 3_000, n)
    factors["rrp_foreign_official"] = -foreign * (c_m + b) + rng.normal(0, 1_500, n)
    factors["rrp_others"] = -routes["mmf_route_prop"] + rng.normal(0, 2_000, n)
    bridge_cash, bridge_securities = rng.normal(0, 3_000, n), rng.normal(0, 2_000, n)
    rho = rng.integers(-3, 4, n).astype(float)
    parts = {"bridge_outcome_term_deposits": rng.normal(0, 1_000, n),
             "bridge_outcome_z1_flow_vs_level": rng.normal(0, 500, n) * (rng.random(n) < 0.2),
             "bridge_outcome_unattributed": rng.normal(0, 5_000, n)}
    sfp = rng.normal(0, 1_000, n) * route_on
    if quiet:
        factors = {k: np.zeros(n) for k in factors} | {"currency_in_circulation": offset * (c_m + b),
                                                       "rrp_foreign_official": -foreign * (c_m + b)}
        bridge_cash, bridge_securities, rho, sfp = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        parts = {k: np.zeros(n) for k in parts}
    c_h, s_h = c_m + bridge_cash, b + bridge_securities
    y_h = c_h + s_h + sum(factors.values()) + rho
    bridge_outcome = sum(parts.values())
    y_z = y_h + bridge_outcome
    gammas = {"trc_core": np.zeros(n), "trc_preferred": routes["mmf_route_prop"],
              "trc_mmf_lower": routes["mmf_route_lb"], "trc_mmf_upper": routes["mmf_route_ub"]}

    def fmt(value: float, keep: bool = True) -> str:
        return repr(float(value) * scale) if keep else ""

    components, ledger, bridges = [], [], []
    for i, q in enumerate(quarters):
        live = q in sample
        components.append({"quarter": q, "trc_cash_leg": fmt(c_m[i], live), "trc_fed_bridge": fmt(b[i]),
                           "delta_rb_z1": fmt(y_z[i]), **{k: fmt(v[i]) for k, v in routes.items()},
                           **{v: fmt(c_m[i] + b[i] + g[i], live) for v, g in gammas.items()}})
        ledger.append({"quarter": q, "delta_reserve_balances": fmt(y_h[i], live), "c_h": fmt(c_h[i], live),
                       "s_h": fmt(s_h[i], live), "rho": fmt(rho[i], live),
                       "f_treasury_general_account": fmt(c_h[i] - sfp[i], live),
                       "f_supplementary_financing_account": fmt(sfp[i], live),
                       "f_treasury_securities": fmt(s_h[i], live),
                       **{f"f_{k}": fmt(v[i], live) for k, v in factors.items()}})
    for variant, gamma in gammas.items():
        for i, q in enumerate(quarters):
            live = q in sample
            bridges.append({"quarter": q, "trc_variant": variant, "x": fmt(c_m[i] + b[i] + gamma[i], live),
                            "gamma": fmt(gamma[i]), "y_z": fmt(y_z[i]), "y_h": fmt(y_h[i], live),
                            "c_h": fmt(c_h[i], live), "c_m": fmt(c_m[i], live), "s_h": fmt(s_h[i], live),
                            "b": fmt(b[i]), **{f"f_{k}": fmt(v[i], live) for k, v in factors.items()},
                            "bridge_outcome": fmt(bridge_outcome[i], live),
                            **{k: fmt(v[i], live) for k, v in parts.items()},
                            "bridge_cash": fmt(bridge_cash[i], live), "bridge_securities": fmt(bridge_securities[i], live),
                            "rho": fmt(rho[i], live), "n_rrp_others": fmt(factors["rrp_others"][i] - gamma[i], live),
                            "bridge_status": rp.RECONCILED_STATUS if live else "unavailable_no_opening_level"})
    names = ["x", "gamma", "c_m", "b", "y_h", "y_z", "n_rrp_others", "bridge_cash", "bridge_securities", "rho", *parts]
    pins = design.raw["inputs"]
    receipt = {"schema_version": design.raw["upstream"]["receipt_schema_version"],
               "producer_commit": design.raw["upstream"]["producer_commit"], "units": design.raw["units"],
               "outputs": {"quarterly": {"sha256": pins["ledger"]["sha256"]}, "bridges": {"sha256": pins["bridges"]["sha256"]}},
               "inputs": {"trc_quarterly_components_v1": {"sha256": pins["trc_components"]["sha256"]}},
               "factor_map": [{"key": k, "role": "factor", "label": k, "side": SIDES.get(k, "supplying")}
                              for k in [*factor_keys, *H41_TREASURY]]
               + [{"key": "reserve_balances", "role": "outcome", "label": "reserves"}],
               "definitions": {k: k for k in names},
               "column_clocks": {"bridges": {k: {"clock": "test"} for k in [*names, *(f"f_{f}" for f in factor_keys)]}}}
    return components, bridges, ledger, receipt


@pytest.fixture(scope="module")
def produced(design):
    d = small(design)
    panel = rp.build_panel(d, *synthetic(d))
    return d, panel, rp.produce(panel, d)


def _cells(rows, *keys):
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def test_contributions_and_matrix_add_up_for_every_variant_outcome_sample_and_spec(design, produced):
    d, _, result = produced
    tables = result["tables"]
    estimates = {(r["outcome"], r["treatment"], r["sample"], r["spec"]): r for r in tables["estimates"]}
    assert len(estimates) == len(d.outcomes) * len(d.variants) * len(d.samples) * len(d.specs) == 96
    contributions = _cells(tables["contributions"], "outcome", "treatment", "sample", "spec")
    grouped = _cells(tables["grouped_contributions"], "outcome", "treatment", "sample", "spec")
    matrix = _cells(tables["leg_factor_matrix"], "outcome", "treatment", "sample", "spec", "level")
    for key, est in estimates.items():
        target = est["beta"] - 1
        parts = contributions[key]
        assert {r["component"] for r in parts} == set(d.components(key[0]))
        assert sum(r["contribution"] for r in parts) == pytest.approx(target, abs=1e-12)
        groups = [r for r in grouped[key] if r["row_type"] == "group"]
        legs = [r for r in grouped[key] if r["row_type"] == "leg"]
        assert sum(r["contribution"] for r in groups) == pytest.approx(target, abs=1e-12)
        assert sum(r["contribution"] for r in legs) == pytest.approx(target, abs=1e-12)
        by_component = {r["component"]: r["contribution"] for r in parts}
        for level in ("component", "group"):
            cells = matrix[(*key, level)]
            body = [r for r in cells if r["leg"] != "column_sum" and r["column"] != "row_sum"]
            for leg in ("C_M", "B", "Gamma"):
                row_sum = next(r["value"] for r in cells if r["leg"] == leg and r["column"] == "row_sum")
                assert row_sum == pytest.approx(sum(r["value"] for r in body if r["leg"] == leg), abs=1e-12)
                assert row_sum == pytest.approx(next(r["contribution"] for r in legs if r["group"] == leg), abs=1e-12)
            grand = next(r["value"] for r in cells if r["leg"] == "column_sum" and r["column"] == "row_sum")
            assert grand == pytest.approx(target, abs=1e-12)
            if level == "component":
                for r in cells:
                    if r["leg"] == "column_sum" and r["column"] != "row_sum":
                        assert r["value"] == pytest.approx(by_component[r["column"]], abs=1e-12)


def test_offsetting_factor_moves_the_slope_by_its_covariance_share(design):
    # Every TRC dollar meets a 30-cent currency drain and nothing else moves reserves:
    # the reserve slope is 0.70 and the whole departure from one is currency.
    d = small(design)
    panel = rp.build_panel(d, *synthetic(d, offset=-0.3, quiet=True))
    currency = d.components("y_h").index("f_currency_in_circulation")
    for spec in ("conditional", "marginal"):
        for sample, quarters in d.samples:
            fit = rp.fit_cell(panel, d, rp.make_cell(panel, d, "y_h", "trc_core", sample, spec, quarters))
            assert float(fit["beta"]) == pytest.approx(0.7, abs=1e-10)
            assert fit["contributions"][currency] == pytest.approx(-0.3, abs=1e-10)
            assert np.abs(np.delete(fit["contributions"], currency)).max() < 1e-10
    # Here the lagged outcome is exactly 0.7 times lagged TRC: the design must fail, not drop a lag.
    with pytest.raises(ValueError, match="rank deficient"):
        rp.fit_cell(panel, d, rp.make_cell(panel, d, "y_h", "trc_core", "pooled", "lag_sensitivity", d.sample))
    # Adding the MMF route to TRC moves exactly its covariance into RRP others net of the route.
    rrp = d.components("y_h").index("n_rrp_others")
    fit = rp.fit_cell(panel, d, rp.make_cell(panel, d, "y_h", "trc_preferred", "era_2016Q1_2025Q4",
                                             "marginal", d.eras[2][1]))
    xt = fit["resid"][:, 0]
    gamma = panel.column("trc_preferred", "gamma")[rp.make_cell(panel, d, "y_h", "trc_preferred", "era_2016Q1_2025Q4", "marginal", d.eras[2][1]).positions]
    share = float(xt @ (gamma - gamma.mean()) / (xt @ xt))
    assert share > 0.01
    assert fit["contributions"][rrp] == pytest.approx(-share, abs=1e-10)
    assert fit["contributions"][currency] == pytest.approx(-0.3 + 0.3 * share, abs=1e-10)


def test_route_term_is_subtracted_from_rrp_others_exactly_once(design):
    d = small(design)
    rows = synthetic(d)
    for r in rows[1]:
        if r["trc_variant"] == "trc_preferred" and r["bridge_status"] == rp.RECONCILED_STATUS:
            r["n_rrp_others"] = r["f_rrp_others"]  # gross RRP: the route would count twice
    with pytest.raises(ValueError, match="rrp_others_net_of_gamma_once"):
        rp.build_panel(d, *rows)


@pytest.mark.parametrize("variant", ["trc_core", "trc_preferred", "trc_mmf_lower", "trc_mmf_upper"])
def test_identity_equality_is_enforced_for_every_route_sensitivity(design, variant):
    d = small(design)
    panel = rp.build_panel(d, *synthetic(d))
    assert panel.checks["identity_y_h"] < 1e-6 and panel.checks["identity_y_z"] < 1e-6
    rows = synthetic(d)
    target = next(r for r in rows[1] if r["trc_variant"] == variant and r["quarter"] == "2020Q2")
    target["x"] = repr(float(target["x"]) + 1.0)
    with pytest.raises(ValueError):
        rp.build_panel(d, *rows)


def test_common_mask_residualization_and_missing_values_raise(design, produced):
    d, panel, _ = produced
    cell = rp.make_cell(panel, d, "y_z", "trc_preferred", "era_2016Q1_2025Q4", "lag_sensitivity",
                        d.eras[2][1])
    fit = rp.fit_cell(panel, d, cell)
    lags = {k: v[cell.positions] for k, v in rp.cell_lags(panel, "y_z", "trc_preferred").items()}
    z = rp.control_matrix(panel.qoy[cell.positions], d.specs["lag_sensitivity"], lags)
    names, data = rp.cell_columns(panel, d, "y_z", "trc_preferred")
    assert fit["resid"].shape == (len(cell.positions), len(names))
    # Every series (treatment, outcome, each component, each leg) is orthogonal to the same controls.
    np.testing.assert_allclose(z.T @ fit["resid"], 0, atol=1e-6 * np.abs(data[cell.positions]).max())
    rows = synthetic(d)
    next(r for r in rows[1] if r["quarter"] == "2010Q2" and r["trc_variant"] == d.primary_variant)["f_loans"] = ""
    with pytest.raises(ValueError, match="Missing sample value"):
        rp.build_panel(d, *rows)


def test_positive_common_denominator_and_partial_variation_are_reported(design, produced):
    d, panel, result = produced
    for r in result["tables"]["estimates"]:
        assert r["common_denominator"] > 0
        assert 0 < r["partial_variation_share"] <= 1 + 1e-12
    x = panel.column("trc_core", "x").copy()
    flat = dataclasses.replace(panel, by_variant={**panel.by_variant, "trc_core": {**panel.by_variant["trc_core"], "x": np.where(np.isfinite(x), 5.0, x)}})
    cell = rp.make_cell(flat, d, "y_h", "trc_core", "pooled", "marginal", d.sample)
    with pytest.raises(ValueError, match="no partial variation"):
        rp.fit_cell(flat, d, cell)


def test_no_contemporaneous_total_reserves_control(design, tmp_path):
    raw = copy.deepcopy(design.raw)
    for bad in ("TOTRESNS", "y_h", "delta_rb_z1", "f_currency_in_circulation"):
        raw["specifications"]["conditional"]["controls"] = ["quarter_of_year", bad]
        (tmp_path / "config").mkdir(exist_ok=True)
        (tmp_path / rp.CONFIG_LOCATOR).write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="disallowed controls"):
            rp.load_design(tmp_path)
    with pytest.raises(ValueError, match="Disallowed control"):
        rp.control_matrix(np.asarray([1, 2, 3, 4]), ["y_h"], {})
    for controls in design.specs.values():
        assert set(controls) <= {"quarter_of_year", "lag1_treatment", "lag1_outcome"}


def test_lags_are_formed_on_the_full_calendar_before_masking(design, produced):
    d, panel, _ = produced
    lags = rp.cell_lags(panel, "y_h", "trc_preferred")
    i = panel.positions(["2016Q1"])[0]
    assert lags["lag1_outcome"][i] == panel.common["y_h"][i - 1]  # 2015Q4, a masked transition quarter
    assert lags["lag1_treatment"][i] == panel.column("trc_preferred", "x")[i - 1]
    cell = rp.make_cell(panel, d, "y_h", "trc_preferred", "pooled", "lag_sensitivity", d.sample)
    assert cell.dropped == ("2003Q1",)  # 2002Q4 TRC is unavailable, never imputed
    gapped = np.asarray([0, 1, 3, 4])  # a deleted quarter is not bridged by row adjacency
    assert np.isnan(rp._lag1(np.arange(4.0), gapped)[2])


def test_calendar_lag_products_after_a_synthetic_missing_quarter(design, produced):
    d, panel, _ = produced
    full = rp.make_cell(panel, d, "y_h", "trc_preferred", "era_2009Q1_2015Q3", "marginal", d.eras[1][1])
    deleted = dataclasses.replace(full, positions=np.delete(full.positions, 10))
    fit = rp.fit_cell(panel, d, deleted)
    x, y = fit["x"], fit["y"]
    ordinals = panel.ordinals[deleted.positions]
    design_matrix = np.column_stack([np.ones(len(x)), x])
    ols = _ols(y.tolist(), design_matrix.tolist())
    scores = design_matrix * np.asarray(ols.residuals)[:, None]
    meat = scores.T @ scores
    for lag in range(1, 5):
        for a in range(len(x)):
            for b in range(len(x)):
                if ordinals[a] - ordinals[b] == lag:
                    cross = np.outer(scores[a], scores[b])
                    meat += (1 - lag / 5) * (cross + cross.T)
    bread = np.linalg.inv(design_matrix.T @ design_matrix)
    expected = bread @ meat @ bread * len(x) / (len(x) - 2)
    assert fit["hac"][4]["se"] == pytest.approx(np.sqrt(expected[1, 1]), rel=1e-9)
    positional = _ols(y.tolist(), design_matrix.tolist(), covariance_estimator="newey_west", covariance_lags=4)
    assert positional.ses[1] != pytest.approx(fit["hac"][4]["se"], rel=1e-6)


def _flatten(result):
    out = {}
    for name, rows in result["tables"].items():
        for i, r in enumerate(rows):
            for k, v in r.items():
                if isinstance(v, (float, np.floating)) and not k.endswith(("usd_million", "denominator", "ss", "numerator")) and k != "usd_million":
                    out[(name, i, k)] = float(v)
    return out


def test_common_million_to_billion_rescaling_leaves_every_ratio_unchanged(design, produced):
    d, _, base = produced
    scaled = rp.produce(rp.build_panel(d, *synthetic(d, scale=1e-3)), d)
    a, b = _flatten(base), _flatten(scaled)
    assert a.keys() == b.keys() and len(a) > 10_000
    for key, value in a.items():
        assert b[key] == pytest.approx(value, rel=1e-7, abs=1e-9), key
    for r0, r1 in zip(base["tables"]["estimates"], scaled["tables"]["estimates"], strict=True):
        assert r1["common_denominator"] == pytest.approx(r0["common_denominator"] * 1e-6, rel=1e-9)


def test_rank_failure_raises_and_structural_zero_route_leg_is_kept(design, produced):
    d, panel, result = produced
    z = np.column_stack([np.ones(8), np.arange(8.0), np.arange(8.0) * 2])
    with pytest.raises(ValueError, match="rank deficient"):
        rp.residualize(z, np.ones((8, 1)))
    stacked = np.stack([np.column_stack([np.ones(8), np.arange(8.0)]), np.column_stack([np.ones(8), np.zeros(8)])])
    with pytest.raises(ValueError, match="rank deficient"):
        rp.residualize(stacked, np.ones((2, 8, 1)))
    one_quarter = np.asarray([2] * 6)  # a quarter-of-year indicator equal to the intercept
    with pytest.raises(ValueError, match="rank deficient"):
        rp.residualize(rp.control_matrix(one_quarter, ["quarter_of_year"], {}), np.ones((6, 1)))
    gamma_rows = [r for r in result["tables"]["leg_factor_matrix"] if r["leg"] == "Gamma" and r["level"] == "component"]
    early = [r for r in gamma_rows if r["sample"] == "era_2003Q1_2008Q3" or r["treatment"] == "trc_core"]
    assert early and all(r["leg_status"] == "structural_zero_in_sample" and r["value"] == 0.0 for r in early)
    late = [r for r in gamma_rows if r["sample"] == "era_2016Q1_2025Q4" and r["treatment"] == "trc_preferred"]
    assert late and all(r["leg_status"] == "observed" for r in late)


def test_frozen_deposit_producer_sources_are_byte_identical(design, tmp_path):
    verified = rp.verify_frozen_sources(ROOT, design)
    assert {r["path"] for r in verified} >= {"src/ea_tdc/open01.py", "src/ea_tdc/open16.py", "src/ea_tdc/estimation.py",
                                             "src/ea_tdc/covariance.py", "scripts/run_open16_diagnostics.py"}
    for record in design.raw["frozen_deposit_sources"]:
        target = tmp_path / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / record["path"], target)
    rp.verify_frozen_sources(tmp_path, design)
    with (tmp_path / "src/ea_tdc/open16.py").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="Frozen deposit source changed"):
        rp.verify_frozen_sources(tmp_path, design)


def test_pinned_inputs_and_upstream_receipt_are_enforced(design, tmp_path):
    record = dict(design.raw["inputs"]["ledger"])
    target = tmp_path / record["path"]
    target.parent.mkdir(parents=True)
    target.write_text("quarter\n2003Q1\n")
    record["bytes"] = target.stat().st_size
    with pytest.raises(ValueError, match="hash or size differs"):
        rp._pinned(tmp_path, record)
    d = small(design)
    rows = list(synthetic(d))
    rows[3] = {**rows[3], "producer_commit": "0" * 40}
    with pytest.raises(ValueError, match="does not bind"):
        rp.build_panel(d, *rows)


def test_full_calendar_weights_shared_across_every_series_and_era(design, produced, monkeypatch):
    d, panel, _ = produced
    calls = []
    generator = rp.dependent_wild_weights
    def capture(rng, ordinals, bandwidth, replicates):
        weights = generator(rng, ordinals, bandwidth, replicates)
        calls.append(weights)
        return weights
    monkeypatch.setattr(rp, "dependent_wild_weights", capture)
    boot, representation = rp.run_bootstrap(panel, d)
    assert len(calls) == 1  # Independent era/sample/outcome/variant paths fail.
    weights = calls[0]
    for outcome in d.outcomes:
        for variant in d.variants:
            for spec in d.raw["bootstrap"]["specifications"]:
                for sample, quarters in d.samples:
                    pos = panel.positions(quarters)
                    _, data = rp.cell_columns(panel, d, outcome, variant)
                    z = rp.control_matrix(panel.qoy[pos], d.specs[spec], {})
                    pseudo = rp.wild_responses(data[pos], z, weights[:, pos])
                    resid = rp.residualize(np.broadcast_to(z, (len(weights), *z.shape)), pseudo, require=(0,))
                    expected = np.einsum("bn,bn->b", resid[..., 0], resid[..., 1]) / np.einsum("bn,bn->b", resid[..., 0], resid[..., 0])
                    cell = boot[(outcome, variant, sample, spec)]
                    np.testing.assert_allclose(cell["beta"], expected, atol=1e-12)
                    np.testing.assert_allclose(cell["contributions"].sum(axis=1), cell["beta"] - 1, atol=1e-10)
                    np.testing.assert_allclose(cell["matrix_grand_sum"], cell["beta"] - 1, atol=1e-10)
    assert representation["full_calendar_quarters"] == len(panel.quarters)
    assert representation["analysis_quarters"] == len(d.sample)
    assert [r["quarter"] for r in representation["per_quarter"]] == list(d.sample)
    assert all(r["replicates_present"] == len(weights) and r["expected_multiplicity"] == 1
               and r["omission_rate"] == 0 for r in representation["per_quarter"])


def test_wild_bootstrap_keeps_every_observation_and_influential_treatment(design, produced):
    d, panel, result = produced
    pos = panel.positions(d.eras[0][1])
    _, data = rp.cell_columns(panel, d, "y_h", d.primary_variant)
    z = rp.control_matrix(panel.qoy[pos], d.specs["conditional"], {})
    weights = rp.dependent_wild_weights(np.random.default_rng(1), panel.ordinals, 4, 30)
    pseudo = rp.wild_responses(data[pos], z, weights[:, pos])
    assert pseudo.shape == (30, len(pos), data.shape[1])
    np.testing.assert_array_equal(pseudo[..., 0], np.broadcast_to(data[pos, 0], (30, len(pos))))
    assert panel.quarters[pos[-1]] == "2008Q3"
    leverage = [r for r in result["tables"]["influence"] if r["kind"] == "leave_one_out" and r["spec"] != "lag_sensitivity"]
    assert all(r["bootstrap_replicates_present"] == d.raw["bootstrap"]["replicates"]
               and r["bootstrap_omission_rate"] == 0 for r in leverage)
    with pytest.raises(ValueError, match="every observed quarter"):
        rp.wild_responses(data[pos], z, weights[:, pos[:-1]])


def test_bartlett_multiplier_covariance_crosses_era_gaps():
    ordinals = np.array([0, 1, 3, 4, 9])  # missing transition remains a calendar gap
    weights = rp.dependent_wild_weights(np.random.default_rng(4), ordinals, 4, 100000)
    expected = np.maximum(1 - abs(ordinals[:, None] - ordinals[None, :]) / 4, 0)
    np.testing.assert_allclose(np.cov(weights, rowvar=False), expected, atol=0.015)
    assert np.cov(weights, rowvar=False)[1, 2] > 0.45


def test_joint_bootstrap_imposes_equal_slope_null_in_live_paths(design, produced, monkeypatch):
    d, panel, _ = produced
    positions, era, _ = rp._era_arrays(panel, d)
    x, qoy = panel.column(d.primary_variant, "x")[positions], panel.qoy[positions]
    y = panel.common["y_h"][positions] + x * era  # strongly unequal observed slopes
    zero = np.zeros((19, len(x)))
    restricted = rp.era_wild_slopes(x, y, era, qoy, d.specs["conditional"], zero, restricted=True)
    unrestricted = rp.era_wild_slopes(x, y, era, qoy, d.specs["conditional"], zero, restricted=False)
    np.testing.assert_allclose(restricted @ rp.CONTRAST.T, 0, atol=1e-12)
    assert np.max(abs(unrestricted @ rp.CONTRAST.T)) > 0.5
    calls = []
    original = rp.era_wild_slopes
    def checked(*args, restricted):
        calls.append(restricted)
        return original(*args, restricted=restricted)
    monkeypatch.setattr(rp, "era_wild_slopes", checked)
    boot, _ = rp.run_bootstrap(panel, d)
    assert calls and all(calls)  # Production joint tests impose the null.
    calls.clear()
    rp.calibrate(panel, d)
    assert calls and calls[::2] == [True] * (len(calls) // 2)
    assert calls[1::2] == [False] * (len(calls) // 2)  # Coverage uses unrestricted fits.
    weights = rp.dependent_wild_weights(np.random.default_rng(d.raw["bootstrap"]["seed"]),
                                       panel.ordinals, d.raw["bootstrap"]["bandwidth_quarters"],
                                       d.raw["bootstrap"]["replicates"])
    expected = original(x, panel.common["y_h"][positions], era, qoy, d.specs["conditional"],
                        weights[:, positions], restricted=True) @ rp.CONTRAST.T
    np.testing.assert_allclose(boot[("y_h", d.primary_variant, "joint_null", "conditional")]["delta"], expected)


def test_era_contrasts_match_separate_eras_and_report_calibrated_size(design, produced):
    d, panel, result = produced
    joint = [r for r in result["tables"]["era_contrasts"] if r["row_type"] == "joint"]
    assert len(joint) == len(d.outcomes) * len(d.variants) * len(d.specs)
    for r in joint:
        assert r["p_hac4"] == pytest.approx(np.exp(-r["wald_hac4"] / 2))
    primary = [r for r in joint if r["treatment"] == d.primary_variant and r["spec"] in ("conditional", "marginal")]
    assert all(r["era_inference"] != "calibration_not_run_descriptive_only" and r["size_hac_wald"] is not None for r in primary)
    others = [r for r in joint if r["treatment"] != d.primary_variant or r["spec"] == "lag_sensitivity"]
    assert all(r["era_inference"] == "calibration_not_run_descriptive_only" for r in others)
    calibration = [r for r in result["tables"]["calibration"] if r["row_type"] == "size"]
    assert {(r["outcome"], r["spec"], r["dgp"], r["innovation"], r["test"]) for r in calibration} == {
        (o, s, g["id"], v["id"], t) for o in ("y_h", "y_z") for s in ("conditional", "marginal")
        for g in d.raw["calibration"]["dgps"] for v in d.raw["calibration"]["innovations"] for t in ("hac_wald", "bootstrap_fixed_scale_maximum")}
    for r in calibration:
        assert r["status"] == ("calibrated" if r["rejection_rate"] <= r["max_acceptable_size"] else "not_calibrated")
        assert r["size_direction"] == rp.size_direction(r["rejection_rate"], d.raw["calibration"])
    assert [rp.size_direction(v, d.raw["calibration"]) for v in (0.0, 0.05, 0.2)] == ["conservative", "near_nominal", "over_rejects"]


def test_fixed_scale_maximum_band_and_hac_wald_under_equal_slopes():
    rng = np.random.default_rng(3)
    delta = np.asarray([0.2, -0.1])
    star = delta + rng.normal(size=(999, 2)) * [0.1, 0.3]
    out = rp.fixed_scale_maximum_inference(delta, star, 0.95, star - delta)
    assert np.all(out["lower"] < delta) and np.all(out["upper"] > delta)
    assert out["critical_value"] > 1.96 and 0 < out["p_value"] <= 1
    assert rp.fixed_scale_maximum_inference(np.zeros(2), star - delta, 0.95, star - delta)["p_value"] > 0.5
    n = 90
    era = np.repeat([0, 1, 2], 30)
    x = rng.normal(size=n)
    y = 0.6 * x + era + rng.normal(size=n) * 0.5
    out = rp.era_contrast_wald(x, y, era, np.tile([1, 2, 3, 4], 23)[:n], list(range(n)), [], {}, 4)
    assert out["p_value"] > 0.001
    fit = calendar_hac(y, rp.era_design(x, era, np.ones(n), [], {}, 3), list(range(n)), lags=4)
    np.testing.assert_allclose(out["slopes"], fit.beta[:3])


def test_influence_shares_and_leave_one_out_slopes(design, produced):
    d, panel, result = produced
    rows = [r for r in result["tables"]["influence"]
            if (r["outcome"], r["treatment"], r["sample"], r["spec"]) == ("y_h", "trc_preferred", "pooled", "conditional")]
    loo = [r for r in rows if r["kind"] == "leave_one_out"]
    assert len(loo) == 92
    assert sum(r["partial_variation_share"] for r in loo) == pytest.approx(1.0)
    assert sum(r["centered_variation_share"] for r in loo) == pytest.approx(1.0)
    target = next(r for r in loo if r["deleted"] == "2020Q2")
    cell = rp.make_cell(panel, d, "y_h", "trc_preferred", "pooled", "conditional", d.sample)
    kept = dataclasses.replace(cell, positions=cell.positions[[panel.quarters[p] != "2020Q2" for p in cell.positions]])
    assert float(rp.fit_cell(panel, d, kept)["beta"]) == pytest.approx(target["beta_after_deletion"], rel=1e-10)
    blocks = {r["deleted"]: r["n_deleted"] for r in rows if r["kind"] == "block_deletion"}
    assert blocks == {"2008Q3-2009Q2": 4, "2020Q1-2021Q1": 5}


@pytest.mark.parametrize("controls", [[], ["quarter_of_year"]])
def test_restricted_bootstrap_matches_explicit_refit(controls):
    rng = np.random.default_rng(57)
    era = np.repeat(np.arange(3), 24)
    qoy = np.tile(np.arange(1, 5), 18)
    x = rng.normal(size=len(era))
    y = 1 + era + (0.5 + era) * x + rng.normal(size=len(x))
    full = rp.era_design(x, era, qoy, controls, {}, 3)
    restricted = np.column_stack([x, full[:, 3:]])
    fit = np.linalg.lstsq(restricted, y, rcond=None)[0]
    errors = y - restricted @ fit
    weights = rp.dependent_wild_weights(rng, np.arange(len(x)), 4, 20)
    expected = np.asarray([np.linalg.lstsq(full, restricted @ fit + w * errors, rcond=None)[0][:3]
                           for w in weights])
    actual = rp.era_wild_slopes(x, y, era, qoy, controls, weights, restricted=True)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_joint_p_uses_restricted_distribution_and_band_uses_unrestricted():
    rng = np.random.default_rng(5)
    delta = np.array([0.3, -0.2])
    star = delta + rng.normal(size=(999, 2)) * 0.1
    null = rng.normal(size=(999, 2)) * 0.8
    result = rp.fixed_scale_maximum_inference(delta, star, 0.95, null)
    substituted = rp.fixed_scale_maximum_inference(delta, star, 0.95, star - delta)
    assert result["p_value"] > 0.5 and substituted["p_value"] < 0.05
    np.testing.assert_array_equal(result["lower"], substituted["lower"])
    np.testing.assert_array_equal(result["upper"], substituted["upper"])
    with pytest.raises(ValueError, match="eight replicates"):
        rp._summary(np.arange(4.0), 0.95)


def test_every_bootstrap_p_and_endpoint_has_mc_error_and_scope(design, produced):
    d, _, result = produced
    for row in result["tables"]["era_contrasts"]:
        if "bootstrap_p" in row:
            p, b = row["bootstrap_p"], row["bootstrap_replicates"]
            assert row["bootstrap_p_mc_se"] == pytest.approx(np.sqrt(p * (1-p) / b))
            assert row["threshold_status"] in ("near_threshold", "away_from_threshold")
        if "simultaneous_lower" in row:
            assert np.isfinite(row["simultaneous_lower_mc_se"])
            assert np.isfinite(row["simultaneous_upper_mc_se"])
    for outcome in d.outcomes:
        for row in result["tables"][f"thin_era_table_{outcome}"]:
            assert row["era_selection"] == d.raw["era_declaration"]
            if row["row_type"] in ("slope", "contrast"):
                expected = rp.interval_evidence(result["tables"]["calibration"], d, outcome, d.primary_variant,
                                                row["spec"], row["sample"])
                assert row["bootstrap_uncertainty_status"] == expected["bootstrap_uncertainty_status"]
                assert row["calibration_scope"] == expected["calibration_scope"]
                assert row["bootstrap_lower_mc_se"] >= 0 and row["bootstrap_upper_mc_se"] >= 0
            if row["row_type"] == "joint_test":
                assert row["bootstrap_p_mc_se"] > 0
                assert row["calibration_scope"] == "specified_joint_test_null_size_checks_only"
    assert "power" in d.raw["calibration"]["size_rule"] and "describes low power" not in d.raw["calibration"]["size_rule"]


def test_runner_writes_versioned_tables_and_a_receipt_binding_every_hash(design, produced, tmp_path, monkeypatch):
    d, panel, _ = produced
    scripts = ROOT / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("reserve_runner_test", scripts / "run_reserve_projection.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    shutil.copyfile(ROOT / "uv.lock", tmp_path / "uv.lock")
    for record in d.raw["frozen_deposit_sources"]:
        (tmp_path / record["path"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / record["path"], tmp_path / record["path"])
    commit = "a" * 40
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "_verify_producer_commit", lambda root, supplied: commit)
    monkeypatch.setattr(runner, "_verify_source", lambda root: None)
    monkeypatch.setattr(runner.rp, "load_design", lambda root, locator: d)
    monkeypatch.setattr(runner.rp, "load_panel", lambda root, design: panel)
    args = SimpleNamespace(producer_commit=commit, config=rp.CONFIG_LOCATOR,
                           output_dir="output/reports/reserve_projection_test")
    receipt_path = runner.run(args)
    receipt = json.loads(receipt_path.read_text())
    assert receipt["producer_commit"] == commit
    assert receipt["frozen_deposit_sources"]["status"] == "byte_identical"
    assert receipt["uncertainty_definitions"] == d.raw["uncertainty_definitions"]
    assert receipt["calibration_results"] == produced[2]["tables"]["calibration"]
    assert all(r["omission_rate"] == 0 for r in receipt["bootstrap_representation"]["per_quarter"])
    assert set(receipt["inputs"]) == {"trc_components", "bridges", "ledger", "ledger_receipt"}
    assert set(receipt["outputs"]) == set(runner.TABLES.values())
    for name, record in receipt["outputs"].items():
        data = (receipt_path.parent / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record["sha256"] and len(data) == record["bytes"]
    assert str(tmp_path) not in receipt_path.read_text()
    assert receipt["headline"]["primary"] == {"outcome": "y_h", "treatment": "trc_preferred",
                                              "spec": "conditional", "hac_lags": 4}
    with pytest.raises(ValueError, match="already exists"):
        runner.run(args)


def test_coverage_counts_and_labels_cover_every_declared_interval(produced):
    d, _, result = produced
    rows = [r for r in result["tables"]["calibration"] if r["row_type"] == "coverage"]
    expected = {(o, s, g["id"], v["id"], kind, target)
                for o in d.raw["calibration"]["outcomes"] for s in d.raw["calibration"]["specifications"]
                for g in d.raw["calibration"]["dgps"] for v in d.raw["calibration"]["innovations"]
                for kind, target in rp.coverage_targets(d)}
    assert {(r["outcome"], r["spec"], r["dgp"], r["innovation"], r["interval_type"], r["interval_id"])
            for r in rows} == expected
    for r in rows:
        assert r["coverage"] == r["covered"] / r["replications"]
        assert r["monte_carlo_se"] == pytest.approx(np.sqrt(r["coverage"] * (1-r["coverage"]) / r["replications"]))
        assert r["status"] == rp.coverage_status(r["coverage"], r["monte_carlo_se"], d.raw["calibration"]["coverage"])
        if r["interval_id"] == "joint_two_contrasts":
            each = [v["covered"] for v in rows if (v["outcome"], v["spec"], v["dgp"], v["innovation"]) ==
                    (r["outcome"], r["spec"], r["dgp"], r["innovation"]) and "_minus_" in v["interval_id"]]
            assert r["covered"] <= min(each)


def test_failed_band_evidence_cannot_be_labelled_checked_in_live_thin_tables(produced):
    d, _, result = produced
    tables = copy.deepcopy(result["tables"])
    # Stale labels cannot override measured evidence in the publication table.
    for r in tables["calibration"]:
        if r["row_type"] == "coverage":
            r.update(coverage=0.5, monte_carlo_se=0.01, status="coverage_checked_specified_dgp")
    thin = rp._thin_tables(d, tables)
    for outcome in d.outcomes:
        for row in thin[f"thin_era_table_{outcome}"]:
            if row["row_type"] == "contrast" or (row["row_type"] == "slope" and row["sample"] != "pooled"):
                assert row["bootstrap_uncertainty_status"] == "coverage_failed_specified_dgp"
                assert row["coverage_minimum"] == 0.5
    rule = d.raw["calibration"]["coverage"]
    assert rp.coverage_status(0.87, 0.01, rule) == "coverage_failed_specified_dgp"
    assert rp.coverage_status(0.89, 0.01, rule) == "coverage_checked_specified_dgp"
    with pytest.raises(ValueError, match="Malformed coverage"):
        rp.coverage_status(float("nan"), 0.01, rule)


def test_untested_intervals_never_receive_checked_status(produced):
    d, _, result = produced
    for row in result["tables"]["estimates"]:
        if row["sample"] == "pooled" or row["treatment"] != d.primary_variant or row["spec"] == "lag_sensitivity":
            assert row["bootstrap_uncertainty_status"] == "coverage_unverified"
            assert json.loads(row["coverage_evidence"]) == []
        assert row["hac_uncertainty_status"] == "coverage_unverified"
    for name in ("contributions", "grouped_contributions"):
        assert all(r["bootstrap_uncertainty_status"] == "coverage_unverified" for r in result["tables"][name])
    assert rp.interval_evidence([], d, "y_h", d.primary_variant, "conditional", d.eras[0][0])["bootstrap_uncertainty_status"] == "coverage_unverified"
    partial = [r for r in result["tables"]["calibration"] if r["row_type"] == "coverage"][:1]
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        rp.interval_evidence(partial, d, partial[0]["outcome"], d.primary_variant, partial[0]["spec"], partial[0]["interval_id"])


def test_fixed_scale_statistic_is_never_described_as_replicate_studentized(produced):
    d, _, result = produced
    descriptions = json.dumps({"design": d.raw, "tables": result["tables"]}).lower()
    assert "bootstrap-t" not in descriptions and "max-t" not in descriptions
    source = (ROOT / "src/ea_tdc/reserve_projection.py").read_text().lower()
    assert "bootstrap-t" not in source and "max-t" not in source
    assert "fixed_scale_maximum" in d.raw["bootstrap"]["statistic"]
    # The same SD divides every draw; check the critical value independently.
    delta = np.array([0.2, -0.1])
    star = delta + np.random.default_rng(71).normal(size=(99, 2)) * [0.1, 0.7]
    scale = star.std(axis=0, ddof=1)
    expected = np.quantile(np.max(abs(star - delta) / scale, axis=1), 0.95)
    assert rp.fixed_scale_band(delta, star, 0.95)[1] == pytest.approx(expected)


# ------------------------------------------------------------- 2026q2 refresh vintage


@pytest.fixture(scope="module")
def refresh() -> rp.Design:
    return rp.load_design(ROOT, REFRESH_CONFIG)


@pytest.fixture(scope="module")
def refresh_produced(refresh):
    d = small(refresh)
    panel = rp.build_panel(d, *synthetic(d, foreign=0.05))
    return d, panel, rp.produce(panel, d)


def test_refresh_changes_only_vintage_bound_fields(design, refresh):
    # "Unchanged specifications": a refresh may move the sample end, the last era end,
    # the pins and add the declared balance row, but never a control, lag, seed or group.
    admitted, new = design.raw, refresh.raw
    vintage_bound = {"upstream", "inputs", "sample", "eras", "uncertainty_definitions"}
    for key in admitted:
        if key not in vintage_bound:
            assert new[key] == admitted[key], key
    assert set(new) - set(admitted) == {"vintage", "refresh_scope", "output_dir", "balance_coefficients"}
    assert new["sample"]["start"] == admitted["sample"]["start"]
    for old, era in zip(admitted["eras"][:-1], new["eras"][:-1]):
        assert era == old
    last_old, last_new = admitted["eras"][-1], new["eras"][-1]
    assert (last_new["start"], last_new["label"]) == (last_old["start"], last_old["label"])
    assert last_new["end"] == new["sample"]["end"] and last_new["id"].endswith(new["sample"]["end"])
    shared = admitted["uncertainty_definitions"]
    assert {k: new["uncertainty_definitions"][k] for k in shared} == shared
    assert new["upstream"]["producer_commit"] != admitted["upstream"]["producer_commit"]
    assert new["output_dir"] != "output/reports/reserve_projection_v1"


def test_refresh_sample_ends_at_2026q2_with_no_silent_truncation(refresh, refresh_produced):
    assert (refresh.sample[0], refresh.sample[-1], len(refresh.sample)) == ("2003Q1", "2026Q2", 94)
    assert refresh.eras[-1][1][-1] == "2026Q2" and refresh.transitions == ("2008Q4", "2015Q4")
    d, _, result = refresh_produced
    for r in result["tables"]["estimates"]:
        quarters = dict(d.samples)[r["sample"]]
        lag = r["spec"] == "lag_sensitivity"
        # Only the declared lag control may drop its first quarter; every other quarter is used.
        assert r["n"] == len(quarters) - (lag and quarters[0] == "2003Q1")
        assert r["last_quarter"] == quarters[-1]
    pooled = [r for r in result["tables"]["estimates"] if r["sample"] == "pooled"]
    assert {r["last_quarter"] for r in pooled} == {"2026Q2"} and {r["n"] for r in pooled} == {94, 93}
    # A shell that stops early, or an unreconciled 2026Q2 row, must raise rather than shorten the sample.
    with pytest.raises(ValueError, match="outside the bridge shell"):
        rp.build_panel(d, *synthetic(d, end="2025Q4"))
    rows = synthetic(d)
    for r in rows[1]:
        if r["quarter"] == "2026Q2":
            r["bridge_status"] = "unavailable_no_opening_level"
    with pytest.raises(ValueError, match="not a reconciled bridge row"):
        rp.build_panel(d, *rows)


def test_refresh_grouped_contributions_sum_to_beta_minus_one(refresh_produced):
    d, _, result = refresh_produced
    estimates = {(r["outcome"], r["treatment"], r["sample"], r["spec"]): r["beta"] for r in result["tables"]["estimates"]}
    grouped = _cells(result["tables"]["grouped_contributions"], "outcome", "treatment", "sample", "spec")
    assert set(grouped) == set(estimates)
    for key, beta in estimates.items():
        groups = [r for r in grouped[key] if r["row_type"] == "group"]
        assert {r["group"] for r in groups} == {g.id for g in d.groups_for(key[0])}
        assert sum(r["contribution"] for r in groups) == pytest.approx(beta - 1, abs=1e-12)


def test_foreign_official_absorption_lowers_reserves_and_is_reported_with_positive_sign(refresh):
    # Every TRC dollar meets a 30-cent currency drain and a 20-cent rise in foreign official
    # RRP balances: reserves move 0.50 per dollar, the foreign official group contributes
    # -0.20 to beta - 1, and its balance coefficient is +0.20.
    d = small(refresh)
    panel = rp.build_panel(d, *synthetic(d, offset=-0.3, foreign=0.2, quiet=True))
    group = next(g for g in d.groups if g.id == "foreign_official_rrp_and_deposits")
    index = [d.components("y_h").index(c) for c in group.components]
    for spec in ("conditional", "marginal"):
        for sample, quarters in d.samples:
            fit = rp.fit_cell(panel, d, rp.make_cell(panel, d, "y_h", "trc_core", sample, spec, quarters))
            assert float(fit["beta"]) == pytest.approx(0.5, abs=1e-10)
            assert float(fit["contributions"][index].sum()) == pytest.approx(-0.2, abs=1e-10)
            assert fit["balances"]["foreign_official_rrp_and_deposits"]["coefficient"] == pytest.approx(0.2, abs=1e-10)


def test_foreign_official_coefficient_is_minus_its_contribution_and_an_independent_slope(refresh_produced):
    d, panel, result = refresh_produced
    rows = result["tables"]["balance_coefficients"]
    assert len(rows) == len(d.outcomes) * len(d.variants) * len(d.samples) * len(d.specs)
    grouped = {(r["outcome"], r["treatment"], r["sample"], r["spec"], r["group"]): r
               for r in result["tables"]["grouped_contributions"] if r["row_type"] == "group"}
    estimates = {(r["outcome"], r["treatment"], r["sample"], r["spec"]): r for r in result["tables"]["estimates"]}
    group = next(g for g in d.groups if g.id == "foreign_official_rrp_and_deposits")
    lag = d.hac_lags[0]
    for r in rows:
        key = (r["outcome"], r["treatment"], r["sample"], r["spec"])
        contribution = grouped[(*key, r["group"])]
        assert r["coefficient"] == pytest.approx(-contribution["contribution"], abs=1e-12)
        assert r["reserve_beta"] == estimates[key]["beta"]
        # Independent slope: own balance change (minus the signed factors) on TRC and the cell's controls.
        cell = rp.make_cell(panel, d, *key, dict(d.samples)[r["sample"]])
        lags = {k: v[cell.positions] for k, v in rp.cell_lags(panel, r["outcome"], r["treatment"]).items()}
        z = rp.control_matrix(panel.qoy[cell.positions], d.specs[r["spec"]], lags)
        x = panel.column(r["treatment"], "x")[cell.positions]
        balance = -sum(panel.column(r["treatment"], c)[cell.positions] for c in group.components)
        independent = np.linalg.lstsq(np.column_stack([z, x]), balance, rcond=None)[0][-1]
        assert r["coefficient"] == pytest.approx(independent, rel=1e-9, abs=1e-12)
        fit = calendar_hac(balance, np.column_stack([np.ones(len(x)), x, z[:, 1:]]),
                           [int(o) for o in panel.ordinals[cell.positions]], lags=lag)
        assert r[f"se_hac{lag}"] == pytest.approx(fit.ses[1], rel=1e-12)
        assert r["hac_uncertainty_status"] == "coverage_unverified"
        if r["spec"] in d.raw["bootstrap"]["specifications"]:
            # Percentile interval of the negated draws is the mirrored contribution interval.
            assert r["bootstrap_lower"] == pytest.approx(-contribution["bootstrap_upper"], abs=1e-12)
            assert r["bootstrap_upper"] == pytest.approx(-contribution["bootstrap_lower"], abs=1e-12)
            assert r["bootstrap_uncertainty_status"] == "coverage_unverified"
            assert r["calibration_scope"] == "interval_not_tested"
        else:
            assert r["bootstrap_lower"] == "" and r["bootstrap_uncertainty_status"] == "not_bootstrapped_declared_sensitivity"
    # The balance is a reserve factor, not an outcome bridge: both outcomes give one coefficient.
    for r in rows:
        if r["outcome"] == "y_z" and r["spec"] != "lag_sensitivity":
            twin = next(v for v in rows if (v["outcome"], v["treatment"], v["sample"], v["spec"]) ==
                        ("y_h", r["treatment"], r["sample"], r["spec"]))
            assert r["coefficient"] == pytest.approx(twin["coefficient"], abs=1e-12)


def test_thin_balance_table_sits_beside_the_reserve_coefficient(refresh_produced):
    d, _, result = refresh_produced
    thin = result["tables"]["thin_balance_coefficients"]
    expected = {(o, s, sample) for o in d.outcomes for s in d.raw["bootstrap"]["specifications"] for sample, _ in d.samples}
    assert {(r["outcome"], r["spec"], r["sample"]) for r in thin} == expected and len(thin) == len(expected)
    betas = {(r["outcome"], r["spec"], r["sample"]): r["beta"] for r in result["tables"]["estimates"]
             if r["treatment"] == d.primary_variant}
    for r in thin:
        assert r["reserve_beta"] == betas[(r["outcome"], r["spec"], r["sample"])]
        assert r["bootstrap_uncertainty_status"] == r["hac_uncertainty_status"] == "coverage_unverified"


def test_balance_side_must_match_the_ledger_factor_map(refresh, tmp_path):
    d = small(refresh)
    components, bridges, ledger, receipt = synthetic(d)
    for f in receipt["factor_map"]:
        if f["key"] == "foreign_official_deposits":
            f["side"] = "supplying"
    with pytest.raises(ValueError, match="side differs"):
        rp.build_panel(d, components, bridges, ledger, receipt)
    raw = copy.deepcopy(refresh.raw)
    raw["balance_coefficients"][0]["group"] = "outcome_bridge"
    (tmp_path / "config").mkdir()
    (tmp_path / REFRESH_CONFIG).write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="signed-factor group"):
        rp.load_design(tmp_path, REFRESH_CONFIG)


def test_refresh_receipt_binds_every_consumed_input(tmp_path):
    files = {"a": b"alpha", "b": b"beta"}
    records, outputs = {}, {}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
        records[name] = {"path": name, "upstream_path": f"data/processed/{name}",
                         "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        outputs[f"data/processed/{name}"] = {"sha256": records[name]["sha256"], "bytes": len(data)}
    commit = "d" * 40

    def write(receipt: dict) -> None:
        data = json.dumps(receipt).encode()
        (tmp_path / "RECEIPT.json").write_bytes(data)
        records["refresh_receipt"] = {"path": "RECEIPT.json", "upstream_path": "RECEIPT.json",
                                      "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    base = {"vintage": "2026q2", "producers": {"tdcest_exports_and_acquisition": f"{commit} (main)"}, "outputs": outputs}
    write(base)
    bound = rp.verify_refresh_receipt(tmp_path, records, commit, "2026q2")
    assert bound["bindings"] == {"a": "refresh_receipt_output", "b": "refresh_receipt_output"}
    with pytest.raises(ValueError, match="vintage or TDCest producer"):
        rp.verify_refresh_receipt(tmp_path, records, "e" * 40, "2026q2")
    write(base | {"outputs": {**outputs, "data/processed/a": {"sha256": "0" * 64, "bytes": 5}}})
    with pytest.raises(ValueError, match="different object for a"):
        rp.verify_refresh_receipt(tmp_path, records, commit, "2026q2")
    write(base)
    (tmp_path / "c").write_bytes(b"doc")
    records["c"] = {"path": "c", "upstream_path": "docs/c.md", "sha256": hashlib.sha256(b"doc").hexdigest(), "bytes": 3}
    with pytest.raises(ValueError, match="bound by neither"):
        rp.verify_refresh_receipt(tmp_path, records, commit, "2026q2")
    records["c"]["upstream_commit"] = commit
    assert rp.verify_refresh_receipt(tmp_path, records, commit, "2026q2")["bindings"]["c"] == "tracked_file_at_upstream_commit"
    del records["refresh_receipt"]
    with pytest.raises(ValueError, match="requires a pinned"):
        rp.verify_refresh_receipt(tmp_path, records, commit, "2026q2")


def test_runner_writes_refresh_vintage_balance_tables_to_its_declared_directory(refresh_produced, tmp_path, monkeypatch):
    d, panel, _ = refresh_produced
    scripts = ROOT / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("reserve_runner_refresh_test", scripts / "run_reserve_projection.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    shutil.copyfile(ROOT / "uv.lock", tmp_path / "uv.lock")
    for record in d.raw["frozen_deposit_sources"]:
        (tmp_path / record["path"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / record["path"], tmp_path / record["path"])
    commit = "b" * 40
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "_verify_producer_commit", lambda root, supplied: commit)
    monkeypatch.setattr(runner, "_verify_source", lambda root: None)
    monkeypatch.setattr(runner.rp, "load_design", lambda root, locator: d)
    monkeypatch.setattr(runner.rp, "load_panel", lambda root, design: panel)
    receipt_path = runner.run(SimpleNamespace(producer_commit=commit, config=REFRESH_CONFIG, output_dir=None))
    assert receipt_path.parent == tmp_path / d.raw["output_dir"]
    receipt = json.loads(receipt_path.read_text())
    assert receipt["vintage"] == "2026q2" and receipt["config"]["path"] == REFRESH_CONFIG
    assert set(receipt["outputs"]) == set(runner.TABLES.values()) | set(runner.BALANCE_TABLES.values())
    assert receipt["design"]["balance_coefficients"] == d.raw["balance_coefficients"]
    assert {r["sample"] for r in receipt["headline"]["balance_coefficients"]} == {s for s, _ in d.samples}
    assert "refresh_receipt" in receipt["inputs"]


def test_rolling_path_refits_the_pooled_primary_cell_on_consecutive_sample_windows(produced, tmp_path, monkeypatch):
    """Every rolling point must be the admitted pooled estimator on 48 consecutive
    sample quarters, so the path shows sample dependence of one specification."""
    d, panel, result = produced
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("reserve_rolling_test", ROOT / "scripts" / "run_reserve_rolling.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    rows = runner.rolling(panel, d)
    assert len(rows) == len(d.sample) - runner.WINDOW + 1
    for start, row in enumerate(rows):
        window = d.sample[start:start + runner.WINDOW]
        assert (row["window_start_quarter"], row["window_end_quarter"], row["n"]) == (window[0], window[-1], 48)
        assert (row["outcome"], row["treatment"], row["spec"]) == ("y_h", "trc_preferred", "conditional")
    pooled = next(r for r in result["tables"]["estimates"] if (r["outcome"], r["treatment"], r["sample"], r["spec"])
                  == ("y_h", "trc_preferred", "pooled", "conditional"))
    (full,) = runner.rolling(panel, d, window=len(d.sample))
    assert full["beta"] == pytest.approx(pooled["beta"], abs=1e-12)
    assert (full["lower95"], full["upper95"]) == pytest.approx((pooled["ci95_hac4_lower"], pooled["ci95_hac4_upper"]), abs=1e-12)
    with pytest.raises(ValueError, match="inside the declared sample"):
        runner.rolling(panel, d, window=len(d.sample) + 1)

    shutil.copyfile(ROOT / "uv.lock", tmp_path / "uv.lock")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "_verify_producer_commit", lambda root, supplied: "a" * 40)
    monkeypatch.setattr(runner.rp, "load_design", lambda root, locator: d)
    monkeypatch.setattr(runner.rp, "load_panel", lambda root, design: panel)
    args = SimpleNamespace(producer_commit="a" * 40, config=rp.CONFIG_LOCATOR, output_dir="output/rolling_test")
    receipt = json.loads(runner.run(args).read_text())
    table = tmp_path / "output/rolling_test" / runner.TABLE
    assert receipt["outputs"][runner.TABLE]["sha256"] == hashlib.sha256(table.read_bytes()).hexdigest()
    assert receipt["outputs"][runner.TABLE]["rows"] == len(rows)
    with pytest.raises(ValueError, match="already exists"):
        runner.run(args)
