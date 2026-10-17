"""Descriptive common-sample reserve projections of measured quarterly exposures."""
from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from ea_tdc import reserve_projection as rp

CONFIG = "config/matched_clock_reserve.json"
AVAILABLE = "available_cash_corrected_proxy"


def verify(path: Path, record: dict) -> Path:
    if not path.is_file() or path.stat().st_size != record["bytes"] or rp._sha256(path) != record["sha256"]:
        raise ValueError(f"Input hash or size differs: {record['path']}")
    return path


def number(row: dict, name: str) -> float:
    value = rp._float(row.get(name), name)
    if not math.isfinite(value):
        raise ValueError(f"Missing required {name}")
    return value


def close(actual, expected, label: str) -> None:
    if not np.allclose(actual, expected, rtol=2e-12, atol=1e-7):
        raise ValueError(f"{label} differs")


def last_wednesday(quarter: str) -> date:
    year, q = map(int, quarter.split("Q"))
    end = date(year + (q == 4), 1 if q == 4 else q * 3 + 1, 1) - timedelta(days=1)
    return end - timedelta(days=(end.weekday() - 2) % 7)


def index(rows: list[dict], key: str) -> dict[str, dict]:
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {key}")
    return result


def aggregate(config: dict, quarterly: list[dict], weekly: list[dict], identity: list[dict]) -> tuple[list[dict], list[dict]]:
    """Rebuild complete quarters on the endpoint clock; never use partial sums."""
    index(quarterly, "quarter")
    weeks = index(weekly, "date")
    core_identity = index([r for r in identity if r["variant"] == "core"], "date")
    fields = [v["column"] for v in config["variants"] if v["source"] == "weekly"]
    fields += ["delta_reserve_balances", "fed_leg_par", "primary_cash_adjustment", "redemption_inflation",
               "settlement_clock_adjustment", "j_applied", "soma_coupon_cash", "cash_remittances"]
    fields = list(dict.fromkeys(fields))
    bridges = [v["weekly"] for v in config["components"].values() if v["weekly"]]
    result, excluded = [], []
    for row in quarterly:
        q = row["quarter"]
        end = last_wednesday(q)
        previous = rp.quarter_label(rp._quarter_ordinal(q) - 1)
        start = last_wednesday(previous)
        dates = [(start + timedelta(days=d)).isoformat() for d in range(7, (end - start).days + 1, 7)]
        if row["start_wednesday"] != start.isoformat() or row["end_wednesday"] != end.isoformat():
            raise ValueError("Quarter endpoints are not adjacent last Wednesdays")
        if int(row["expected_weeks"]) != len(dates):
            raise ValueError("Expected weekly coverage differs from the endpoint clock")
        missing = [d for d in dates if d not in weeks or d not in core_identity
                   or weeks[d]["core_status"] == "unavailable_required_input"
                   or core_identity[d]["identity_status"] != "closed_exact_decimal"]
        if row["quarter_status"] != AVAILABLE or missing or int(row["n_weeks"]) != len(dates):
            excluded.append({"quarter": q, "status": "excluded_incomplete_quarter",
                             "missing_dates": "|".join(missing), "source_status": row["quarter_status"],
                             "source_reasons": row["unavailable_reasons"]})
            continue
        for d in dates:
            w, ledger = weeks[d], core_identity[d]
            if w["start_wednesday"] != (date.fromisoformat(d) - timedelta(days=7)).isoformat():
                raise ValueError("Nonconsecutive weekly interval")
            close(number(w, "delta_reserve_balances"), number(w, "reserve_balances_end") - number(w, "reserve_balances_start"), "Weekly outcome")
            close(number(w, "trc_core"), number(ledger, "trc"), "Weekly core identity exposure")
            close(number(w, "trc_core"), number(w, "c_h") + number(w, "fed_leg_cash_corrected")
                  - number(w, "soma_coupon_cash") + number(w, "cash_remittances") + number(w, "j_applied"), "Cash-valued core")
            close(number(w, "trc_par_sensitivity"), number(w, "c_h") + number(w, "fed_leg_par")
                  - number(w, "soma_coupon_cash") + number(w, "cash_remittances"), "Par sensitivity")
            close(number(w, "fed_leg_cash_corrected"), number(w, "fed_leg_par")
                  + number(w, "settlement_clock_adjustment") + number(w, "primary_cash_adjustment")
                  - number(w, "redemption_inflation"), "Cash corrections")
            close(number(w, "delta_reserve_balances"), number(ledger, "delta_reserve_balances"), "Weekly identity outcome")
            close(number(ledger, "route"), 0, "Core route")
            close(number(ledger, "trc") + sum(number(ledger, b) for b in bridges), number(w, "delta_reserve_balances"), "Weekly accounting identity")
        summed = {f: math.fsum(number(weeks[d], f) for d in dates) for f in fields}
        for f, value in summed.items():
            close(value, number(row, f), f"Weekly sum for {q}:{f}")
        close(summed["delta_reserve_balances"], number(weeks[dates[-1]], "reserve_balances_end")
              - number(weeks[dates[0]], "reserve_balances_start"), "Quarterly reserve telescoping")
        components = {name: math.fsum(number(core_identity[d], spec["weekly"]) for d in dates)
                      if spec["weekly"] else 0.0 for name, spec in config["components"].items()}
        result.append({"quarter": q, "n_weeks": len(dates), "start_wednesday": start.isoformat(),
                       "end_wednesday": end.isoformat(), **summed, "components": components, "source": row})
    return result, excluded


def build_panel(config: dict, admitted: rp.Panel, quarterly: list[dict], weekly: list[dict], identity: list[dict]) -> dict:
    sums, excluded = aggregate(config, quarterly, weekly, identity)
    expected = rp.quarter_range(config["sample"]["start"], config["sample"]["end"])
    selected = {r["quarter"]: r for r in sums if r["quarter"] in expected}
    if tuple(selected) != expected or len(expected) != config["sample"]["n"]:
        raise ValueError("Common complete-quarter mask differs from the declared sample")
    # Complete weekly quarters beyond the declared sample never enter it; record why each is absent.
    outside = [{"quarter": r["quarter"], "n_weeks": r["n_weeks"],
                "status": ("outside_declared_sample" if r["quarter"] in admitted.quarters
                           else "not_reported_mixed_clock_quarter_unavailable")}
               for r in sums if r["quarter"] not in expected]
    positions = admitted.positions(expected)
    y = admitted.common["y_h"][positions]
    close(y, [selected[q]["delta_reserve_balances"] for q in expected], "Common Wednesday reserve outcome")
    variants = {}
    for spec in config["variants"]:
        components = {c: np.zeros(len(expected)) for c in config["components"]}
        if spec["source"] == "admitted":
            variant = spec["column"]
            x = admitted.column(variant, "x")[positions]
            components["treasury_deposit_clock"] = admitted.common["bridge_cash"][positions]
            components["admitted_securities_bridge"] = admitted.common["bridge_securities"][positions]
            components["rrp_others"] = admitted.column(variant, "n_rrp_others")[positions]
            components["published_rounding"] = admitted.common["rho"][positions]
            premiums = ["f_unamortized_premiums", "f_unamortized_discounts"]
            components["all_securities_premium_discount"] = sum(admitted.common[k][positions] for k in premiums)
            others = [k for k in admitted.common if k.startswith("f_") and k != "f_rrp_others" and k not in premiums]
            components["non_treasury_signed_factors"] = sum(admitted.common[k][positions] for k in others)
        else:
            x = np.array([selected[q][spec["column"]] for q in expected])
            components = {c: np.array([selected[q]["components"][c] for q in expected]) for c in components}
            if spec["column"] == "trc_par_sensitivity":
                for c in ("primary_cash_price", "redemption_inflation", "maturity_payment_clock", "internal_postings"):
                    components[c] = np.zeros(len(expected))
            if spec["column"] == "c_h":
                components["cash_reference_treasury_securities"] = np.array([selected[q]["fed_leg_par"] for q in expected]) + components["treasury_securities_source_inflation"]
                for c in ("treasury_securities_source_inflation", "primary_cash_price", "redemption_inflation",
                          "maturity_payment_clock", "internal_postings", "coupon_remittance_payments"):
                    components[c] = np.zeros(len(expected))
            if np.any(components["treasury_deposit_clock"] != 0):
                raise ValueError("Matched-clock cash bridge must be exactly zero")
        close(x + sum(components.values()), y, f"Quarterly accounting for {spec['id']}")
        variants[spec["id"]] = {"x": x, "components": components, "quarters": expected}
    # Independently compare the upstream reconciliation against the admitted ledger.
    differences = {c: np.array([number(selected[q]["source"], c) for q in expected]) for c in config["difference_components"]}
    total = variants["matched_core"]["x"] - variants["admitted_core"]["x"]
    close(total, [number(selected[q]["source"], "difference_core") for q in expected], "Core exposure difference")
    close(sum(differences.values()), total, "Named exposure differences sum")
    close([number(selected[q]["source"], "admitted_quarterly_core") for q in expected], variants["admitted_core"]["x"], "Admitted core reference")
    close(differences["difference_treasury_deposit_clock"], admitted.common["bridge_cash"][positions], "Deposit clock difference")
    return {"quarters": expected, "y": y, "variants": variants, "differences": differences,
            "difference_total": total, "excluded": excluded, "outside_sample": outside, "weekly_sums": sums}


def project(quarters, x, data):
    qoy = np.array([rp._quarter_ordinal(q) % 4 + 1 for q in quarters])
    z = rp.control_matrix(qoy, ("quarter_of_year",), {})
    if len(x) <= z.shape[1] + 1 or rp._scaled_rank(z) != z.shape[1]:
        return None
    resid = rp.residualize(z, np.column_stack([x, data]), require=(0,))
    xt = resid[:, 0]
    return xt @ resid[:, 1:] / (xt @ xt), xt * xt / (xt @ xt)


def produce(config: dict, panel: dict) -> dict[str, list[dict]]:
    quarters = panel["quarters"]
    names = list(config["components"])
    estimates, influence, transition = [], [], []
    for sample in config["samples"]:
        pos = np.array([i for i, q in enumerate(quarters) if sample["start"] <= q <= sample["end"]])
        qs = tuple(quarters[i] for i in pos)
        for variant in config["variants"]:
            v = panel["variants"][variant["id"]]
            if v["quarters"] != quarters:
                raise ValueError("Exposure variants use different quarter masks")
            x, y = v["x"][pos], panel["y"][pos]
            comp = np.column_stack([v["components"][c][pos] for c in names])
            if variant["source"] == "weekly" and np.any(v["components"]["treasury_deposit_clock"] != 0):
                raise ValueError("Matched-clock cash bridge must be exactly zero")
            fitted = project(qs, x, np.column_stack([y, comp]))
            base = {"sample": sample["id"], "sample_label": sample["label"], "variant": variant["id"],
                    "exposure_definition": variant["definition"], "first_quarter": qs[0], "last_quarter": qs[-1], "n": len(qs),
                    "quarter_mask": "|".join(qs), "status": "descriptive" if fitted else "unidentified_single_quarter_with_calendar_controls"}
            if fitted:
                slopes, shares = fitted
                beta, contributions = float(slopes[0]), slopes[1:]
                if not math.isclose(float(contributions.sum()), beta - 1, abs_tol=1e-9):
                    raise ValueError("Contributions fail to sum to beta - 1")
                estimates.append(base | {"beta": beta, **dict(zip(names, map(float, contributions))),
                                         "contribution_sum": float(contributions.sum()), "beta_minus_one": beta - 1})
                order = np.argsort(-shares, kind="stable")
                def deletion(i):
                    keep = np.arange(len(qs)) != i
                    fit = project(tuple(q for q, k in zip(qs, keep) if k), x[keep], y[keep])
                    return float(fit[0][0]) if fit else None
                influence.append(base | {"diagnostic": "concentration", "quarter": qs[order[0]],
                                         "partial_variation_share": float(shares[order[0]]),
                                         "top_three_share": float(shares[order[:3]].sum()),
                                         "beta_without_quarter": deletion(order[0]), "event_status": "in_sample"})
                for q in config["influence_quarters"]:
                    i = qs.index(q) if q in qs else None
                    influence.append(base | {"diagnostic": "specified_quarter", "quarter": q,
                                             "partial_variation_share": float(shares[i]) if i is not None else None,
                                             "top_three_share": None, "beta_without_quarter": deletion(i) if i is not None else None,
                                             "event_status": "in_sample" if i is not None else "outside_sample"})
            else:
                estimates.append(base | {"beta": None, **{c: None for c in names}, "contribution_sum": None, "beta_minus_one": None})
                for q in [*qs, *[q for q in config["influence_quarters"] if q not in qs]]:
                    influence.append(base | {"diagnostic": "specified_quarter", "quarter": q, "partial_variation_share": None,
                                             "top_three_share": None, "beta_without_quarter": None,
                                             "event_status": "unidentified" if q in qs else "outside_sample"})
                for i, q in enumerate(qs):
                    transition.append({"quarter": q, "variant": variant["id"], "exposure": float(x[i]), "reserve_change": float(y[i]),
                                       **{c: float(comp[i, j]) for j, c in enumerate(names)},
                                       "remainder": float(comp[i].sum()), "status": base["status"]})
    values = panel["differences"] | {"difference_core": panel["difference_total"]}
    difference_quarters = [{"quarter": q, **{c: float(v[i]) for c, v in values.items()}} for i, q in enumerate(quarters)]
    ordinals = np.array([rp._quarter_ordinal(q) for q in quarters])
    z = rp.control_matrix(ordinals % 4 + 1, ("quarter_of_year",), {})
    matrix = np.column_stack([panel["difference_total"], panel["y"], *values.values()])
    partial = rp.residualize(z, matrix)
    summaries = []
    for j, c in enumerate(values):
        def statistics(m):
            d, y, a = m[:, 0], m[:, 1], m[:, j + 2]
            return {"additive_variance_share": float(a @ d / (d @ d)),
                    "own_variance_share": float(a @ a / (d @ d)),
                    "covariance_reserves_million_squared": float(a @ y / len(a))}
        raw = matrix - matrix.mean(axis=0)
        summaries.append({"component": c, "n": len(quarters), "first_quarter": quarters[0], "last_quarter": quarters[-1],
                          **{"partial_" + k: v for k, v in statistics(partial).items()},
                          **{"raw_" + k: v for k, v in statistics(raw).items()}})
    close(sum(r["partial_additive_variance_share"] for r in summaries[:-1]), 1, "Additive variance shares")
    return {"projections": estimates, "influence": influence, "transitions": transition,
            "exposure_difference_quarters": difference_quarters, "exposure_difference_summary": summaries,
            "excluded_quarters": panel["excluded"]}
