from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from ea_tdc import matched_clock_reserve as mc
from ea_tdc import reserve_projection as rp

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture():
    """Independent weekly event ledger with known signed cash corrections."""
    config = json.loads((ROOT / mc.CONFIG).read_text())
    config["sample"] = {"start": "2006Q1", "end": "2007Q4", "n": 8}
    config["samples"] = [{"id": "pooled", "start": "2006Q1", "end": "2007Q4", "label": "Synthetic"}]
    qs = rp.quarter_range("2006Q1", "2007Q4")
    weekly, quarterly, identity = [], [], []
    common = {k: [] for k in ["y_h", "bridge_cash", "bridge_securities", "rho", "f_rrp_others", "f_unamortized_premiums", "f_unamortized_discounts", "f_currency_in_circulation"]}
    by_variant = {v: {k: [] for k in ["x", "n_rrp_others"]} for v in ["trc_core", "trc_preferred"]}
    rng = np.random.default_rng(51)
    balance = 10000.0
    for q in qs:
        start, end = mc.last_wednesday(rp.quarter_label(rp._quarter_ordinal(q) - 1)), mc.last_wednesday(q)
        dates = [start + timedelta(days=i) for i in range(7, (end - start).days + 1, 7)]
        local, ledgers = [], []
        for d in dates:
            cash, par = rng.normal(0, 500), rng.normal(0, 150)
            primary, redemption, internal, maturity = 2.0, 3.0, -4.0, 1.0
            coupon, remittance, source, premium = 6.0, 2.0, 7.0, 0.5
            core = cash + par + primary - redemption + internal + maturity - coupon + remittance
            bridges = {"non_treasury_signed_factors": -0.3 * core,
                       "rrp_others_net_route": 0.1 * core,
                       "bridge_all_securities_premium_discount": premium,
                       "bridge_treasury_securities_source_inflation": source,
                       "bridge_maturity_payment_clock": -maturity,
                       "bridge_primary_cash_price": -primary,
                       "bridge_redemption_inflation": redemption,
                       "bridge_internal_postings": -internal,
                       "bridge_coupon_remittance_payments": coupon - remittance,
                       "bridge_published_rounding": 0.0}
            outcome = core + math.fsum(bridges.values())
            w = {"date": d.isoformat(), "start_wednesday": (d - timedelta(days=7)).isoformat(),
                 "core_status": "cash_corrected_proxy_secondary_prices_and_entitlements_unobserved",
                 "c_h": cash, "fed_leg_par": par, "fed_leg_cash_corrected": par + primary - redemption + maturity,
                 "trc_core": core, "trc_par_sensitivity": cash + par - coupon + remittance,
                 "primary_cash_adjustment": primary, "redemption_inflation": redemption,
                 "settlement_clock_adjustment": maturity, "j_applied": internal,
                 "soma_coupon_cash": coupon, "cash_remittances": remittance, "delta_reserve_balances": outcome,
                 "reserve_balances_start": balance, "reserve_balances_end": balance + outcome}
            balance += outcome
            h = {"date": d.isoformat(), "variant": "core", "identity_status": "closed_exact_decimal",
                 "trc": core, "delta_reserve_balances": outcome, "route": 0.0, **bridges}
            weekly.append(w); local.append(w); identity.append(h); ledgers.append(h)
        fields = [k for k in local[0] if isinstance(local[0][k], float)]
        summed = {k: math.fsum(r[k] for r in local) for k in fields}
        clock = rng.normal(0, 100)
        b = summed["fed_leg_par"] - summed["soma_coupon_cash"] + summed["cash_remittances"]
        admitted_core = summed["c_h"] - clock + b
        diff = {k: 0.0 for k in config["difference_components"]}
        diff.update(difference_treasury_deposit_clock=clock,
                    difference_primary_cash_price=summed["primary_cash_adjustment"],
                    difference_redemption_inflation=-summed["redemption_inflation"],
                    difference_internal_postings=summed["j_applied"],
                    difference_maturity_payment_clock=summed["settlement_clock_adjustment"])
        quarterly.append({"quarter": q, "start_wednesday": start.isoformat(), "end_wednesday": end.isoformat(),
                          "expected_weeks": len(dates), "n_weeks": len(dates), "quarter_status": mc.AVAILABLE,
                          "unavailable_reasons": "", **summed, "admitted_quarterly_core": admitted_core,
                          **diff, "difference_core": math.fsum(diff.values())})
        common["y_h"].append(summed["delta_reserve_balances"])
        common["bridge_cash"].append(clock)
        common["bridge_securities"].append(math.fsum(r["bridge_treasury_securities_source_inflation"] for r in ledgers) + summed["fed_leg_par"] - b)
        common["rho"].append(0.0)
        common["f_rrp_others"].append(math.fsum(r["rrp_others_net_route"] for r in ledgers))
        common["f_currency_in_circulation"].append(math.fsum(r["non_treasury_signed_factors"] for r in ledgers))
        common["f_unamortized_premiums"].append(len(dates) * 0.5)
        common["f_unamortized_discounts"].append(0.0)
        for v in by_variant:
            route = 100.0 if v == "trc_preferred" else 0.0
            by_variant[v]["x"].append(admitted_core + route)
            by_variant[v]["n_rrp_others"].append(common["f_rrp_others"][-1] - route)
    panel = rp.Panel(qs, np.arange(len(qs)), np.arange(len(qs)) % 4 + 1,
                     {k: np.array(v) for k, v in common.items()},
                     {v: {k: np.array(a) for k, a in columns.items()} for v, columns in by_variant.items()}, {}, {}, {})
    return config, panel, quarterly, weekly, identity


def build(fixture):
    return mc.build_panel(*fixture)


def test_all_variant_contributions_close_and_clock_zero(fixture):
    panel = build(fixture)
    rows = mc.produce(fixture[0], panel)["projections"]
    assert len(rows) == 5
    for row in rows:
        assert row["contribution_sum"] == pytest.approx(row["beta"] - 1, abs=1e-12)
        if row["variant"].startswith("matched"):
            assert row["treasury_deposit_clock"] == 0.0
    # On this independent event fixture, constant per-week bridges residualize
    # away only up to 13/14-week variation; covariance identity must still hold.
    par = panel["variants"]["matched_par"]
    assert np.all(par["components"]["primary_cash_price"] == 0)
    cash = panel["variants"]["matched_cash"]
    q = fixture[2][0]
    assert cash["components"]["cash_reference_treasury_securities"][0] == pytest.approx(q["fed_leg_par"] + 7 * q["n_weeks"])


@pytest.mark.parametrize("component", ["rrp_others", "primary_cash_price", "internal_postings"])
def test_wrong_economic_contribution_fails(fixture, component):
    panel = build(fixture)
    panel["variants"]["matched_core"]["components"][component] += 50 * panel["variants"]["matched_core"]["x"]
    with pytest.raises(ValueError, match="sum to beta"):
        mc.produce(fixture[0], panel)


def test_matched_clock_bridge_cannot_be_hidden_by_offsetting_factor(fixture):
    panel = build(fixture)
    c = panel["variants"]["matched_core"]["components"]
    c["treasury_deposit_clock"] += 100
    c["non_treasury_signed_factors"] -= 100
    with pytest.raises(ValueError, match="exactly zero"):
        mc.produce(fixture[0], panel)


def test_same_mask_required_across_every_row(fixture):
    panel = build(fixture)
    result = mc.produce(fixture[0], panel)
    assert len({r["quarter_mask"] for r in result["projections"]}) == 1
    panel["variants"]["matched_par"]["quarters"] = panel["quarters"][:-1]
    with pytest.raises(ValueError, match="different quarter masks"):
        mc.produce(fixture[0], panel)


@pytest.mark.parametrize("field", ["trc_core", "trc_par_sensitivity", "c_h"])
def test_exposure_equals_sum_not_average_or_calendar_sum(fixture, field):
    config, admitted, quarterly, weekly, identity = fixture
    panel = build(fixture)
    first = quarterly[0]
    rows = [r for r in weekly if first["start_wednesday"] < r["date"] <= first["end_wednesday"]]
    match = next(v for v in config["variants"] if v["column"] == field and v["source"] == "weekly")
    assert panel["variants"][match["id"]]["x"][0] == pytest.approx(math.fsum(r[field] for r in rows))
    quarterly[0][field] += 100
    with pytest.raises(ValueError, match="Weekly sum"):
        build(fixture)


@pytest.mark.parametrize("mutation", ["missing_week", "unavailable_week", "missing_value", "unavailable_quarter"])
def test_incomplete_quarter_never_partially_summed(fixture, mutation):
    config, admitted, quarterly, weekly, identity = fixture
    if mutation == "missing_week":
        weekly.pop(0)
    elif mutation == "unavailable_week":
        weekly[0]["core_status"] = "unavailable_required_input"
    elif mutation == "missing_value":
        weekly[0]["trc_core"] = ""
        with pytest.raises(ValueError, match="Missing required"):
            build(fixture)
        return
    else:
        quarterly[0]["quarter_status"] = "unavailable_required_weeks"
    sums, excluded = mc.aggregate(config, quarterly, weekly, identity)
    assert quarterly[0]["quarter"] not in {r["quarter"] for r in sums}
    assert excluded[0]["quarter"] == quarterly[0]["quarter"]
    with pytest.raises(ValueError, match="Common complete-quarter mask"):
        build(fixture)


def test_clock_endpoints_and_cash_correction_signs(fixture):
    altered = copy.deepcopy(fixture)
    altered[2][0]["end_wednesday"] = "2006-03-31"
    with pytest.raises(ValueError, match="last Wednesdays"):
        build(altered)
    fixture[3][0]["fed_leg_cash_corrected"] += 2 * fixture[3][0]["redemption_inflation"]
    with pytest.raises(ValueError, match="Cash-valued core"):
        build(fixture)


def test_bridge_mutation_can_close_identity_but_not_validate_measurement(fixture):
    # Offset core and payment bridge: closure alone passes, independently pinned
    # event-cash and quarterly-sum checks must reject the changed exposure.
    fixture[4][0]["trc"] += 17
    fixture[4][0]["bridge_coupon_remittance_payments"] -= 17
    with pytest.raises(ValueError, match="Weekly core identity exposure"):
        build(fixture)


def test_difference_components_and_variance_shares(fixture):
    panel = build(fixture)
    summary = mc.produce(fixture[0], panel)["exposure_difference_summary"]
    assert sum(r["partial_additive_variance_share"] for r in summary[:-1]) == pytest.approx(1)
    d = panel["difference_total"] - panel["difference_total"].mean()
    y = panel["y"] - panel["y"].mean()
    assert summary[-1]["raw_covariance_reserves_million_squared"] == pytest.approx(d @ y / len(d))
    fixture[2][0]["difference_primary_cash_price"] *= -1
    with pytest.raises(ValueError, match="Named exposure differences"):
        build(fixture)


def test_transition_is_not_a_single_observation_slope(fixture):
    config = fixture[0]
    config["samples"].append({"id": "transition", "start": "2006Q1", "end": "2006Q1", "label": "Transition"})
    results = mc.produce(config, build(fixture))
    rows = [r for r in results["projections"] if r["sample"] == "transition"]
    assert len(rows) == 5
    assert all(r["n"] == 1 and r["beta"] is None for r in rows)
    assert len(results["transitions"]) == 5
    assert len(results["projections"][0]["quarter_mask"].split("|")) == 8


def test_influence_refits_controls_after_deletion(fixture):
    config = fixture[0]
    config["influence_quarters"] = ["2006Q1"]
    panel = build(fixture)
    result = mc.produce(config, panel)
    for row in result["influence"]:
        if row["diagnostic"] != "specified_quarter":
            continue
        x = panel["variants"][row["variant"]]["x"][1:]
        y = panel["y"][1:]
        q = panel["quarters"][1:]
        z = rp.control_matrix(np.array([rp._quarter_ordinal(a) % 4 + 1 for a in q]), ("quarter_of_year",), {})
        independent = np.linalg.lstsq(np.column_stack([z, x]), y, rcond=None)[0][-1]
        assert row["beta_without_quarter"] == pytest.approx(independent)


def test_admitted_outputs_are_protected_by_exact_hash_and_inventory(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("matched_runner", ROOT / "scripts/run_matched_clock_reserve.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    p = tmp_path / "output/reports/reserve_projection_v1/receipt.json"
    p.parent.mkdir(parents=True)
    p.write_text("admitted")
    locator = str(p.relative_to(tmp_path))
    config = {"admitted_outputs": {locator: {"path": locator, "sha256": rp._sha256(p), "bytes": p.stat().st_size}}}
    runner.verify_admitted(tmp_path, config)
    p.write_text("modified")
    with pytest.raises(ValueError, match="hash or size"):
        runner.verify_admitted(tmp_path, config)
    p.write_text("admitted")
    (p.parent / "extra.csv").write_text("new")
    with pytest.raises(ValueError, match="inventory"):
        runner.verify_admitted(tmp_path, config)


def test_complete_quarter_after_the_sample_end_is_recorded_never_used(fixture):
    # A weekly quarter can be complete before the Financial Accounts cover it. It must
    # stay out of the declared sample and say whether the mixed clock has it at all.
    config, admitted, quarterly, weekly, identity = fixture
    config["sample"] = {"start": "2006Q1", "end": "2007Q3", "n": 7}
    config["samples"] = [{"id": "pooled", "start": "2006Q1", "end": "2007Q3", "label": "Synthetic"}]
    panel = mc.build_panel(config, admitted, quarterly, weekly, identity)
    assert panel["quarters"][-1] == "2007Q3" and len(panel["y"]) == 7
    assert panel["outside_sample"] == [{"quarter": "2007Q4", "n_weeks": quarterly[-1]["n_weeks"],
                                        "status": "outside_declared_sample"}]
    keep = np.arange(7)
    truncated = rp.Panel(admitted.quarters[:7], admitted.ordinals[keep], admitted.qoy[keep],
                         {k: v[keep] for k, v in admitted.common.items()},
                         {v: {k: a[keep] for k, a in cols.items()} for v, cols in admitted.by_variant.items()}, {}, {}, {})
    panel = mc.build_panel(config, truncated, quarterly, weekly, identity)
    assert panel["outside_sample"][0]["status"] == "not_reported_mixed_clock_quarter_unavailable"
    assert all(len(v["x"]) == 7 for v in panel["variants"].values())


def test_pinned_reserve_outputs_may_live_in_a_vintage_directory(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("matched_runner_vintage", ROOT / "scripts/run_matched_clock_reserve.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    records = {}
    for name in ("receipt.json", "thin/table_v1.csv"):
        p = tmp_path / "output/reports/reserve_projection_2026q2" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(name)
        locator = p.relative_to(tmp_path).as_posix()
        records[locator] = {"path": locator, "sha256": rp._sha256(p), "bytes": p.stat().st_size}
    assert runner.verify_admitted(tmp_path, {"admitted_outputs": records}) == records
    (tmp_path / "output/reports/reserve_projection_2026q2/thin/extra.csv").write_text("new")
    with pytest.raises(ValueError, match="inventory"):
        runner.verify_admitted(tmp_path, {"admitted_outputs": records})
    with pytest.raises(ValueError, match="exactly one receipt"):
        runner.verify_admitted(tmp_path, {"admitted_outputs": {k: v for k, v in records.items() if "thin" in k}})


def test_refresh_changes_only_vintage_bound_fields():
    # Same comparisons on a later vintage: only the sample end, the last era end, pins and
    # variant wording may change; components, controls, influence quarters and inference may not.
    admitted = json.loads((ROOT / mc.CONFIG).read_text())
    refresh = json.loads((ROOT / "config/matched_clock_reserve_2026q2.json").read_text())
    vintage_bound = {"sample", "upstream_commit", "upstream_producer_commit", "inputs", "admitted_inputs",
                     "admitted_config", "admitted_outputs", "variants", "samples"}
    for key in admitted:
        if key not in vintage_bound:
            assert refresh[key] == admitted[key], key
    assert set(refresh) - set(admitted) == {"vintage", "refresh_scope", "output_dir"}
    assert "upstream_commit" not in refresh  # bundle copies are bound by the refresh receipt instead
    assert [(v["id"], v["source"], v["column"]) for v in refresh["variants"]] == \
        [(v["id"], v["source"], v["column"]) for v in admitted["variants"]]
    old, new = {s["id"]: s for s in admitted["samples"]}, {s["id"]: s for s in refresh["samples"]}
    assert set(old) == set(new)
    for sid, sample in new.items():
        assert sample["start"] == old[sid]["start"]
        assert sample["end"] == (refresh["sample"]["end"] if old[sid]["end"] == admitted["sample"]["end"] else old[sid]["end"])
    assert refresh["sample"] == {"start": "2006Q1", "end": "2026Q2", "n": len(rp.quarter_range("2006Q1", "2026Q2"))}
    reserve = json.loads((ROOT / refresh["admitted_config"]["path"]).read_text())
    assert reserve["vintage"] == refresh["vintage"] and refresh["admitted_inputs"] == reserve["inputs"]
    receipts = [k for k in refresh["admitted_outputs"] if k.endswith("/receipt.json")]
    assert receipts == [f"{reserve['output_dir']}/receipt.json"]
