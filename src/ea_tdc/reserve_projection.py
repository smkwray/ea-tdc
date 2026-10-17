"""Reserve-side same-quarter projections with factor covariance accounting.

beta is the OLS dollar-on-dollar slope of the quarterly reserve change y on
measured TRC x within one fixed mask and control set. With common
residualization, beta - 1 = x'N/x'x for the measured remainder N = y - x, and
every signed H.4.1 factor, named bridge and rho contributes x'N_k/x'x on the
same mask. These are conditional covariance associations, not landing,
retention, or causal responses. Controls are only an intercept, quarter-of-year
indicators and one declared lag sensitivity; the deposit control list is never
used.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ea_tdc.open01 import _quarter_ordinal, _residualize
from ea_tdc.open16 import _scaled_rank, calendar_hac, numerical_tolerance

SCHEMA_VERSION = "ea_tdc_reserve_projection_v1"
CONFIG_LOCATOR = "config/reserve_projection.json"
ALLOWED_CONTROLS = ("quarter_of_year", "lag1_treatment", "lag1_outcome")
RECONCILED_STATUS = "reconciled_mixed_clock"
# TDCest signs each H.4.1 factor +d(level) if supplying and -d(level) if absorbing,
# so a balance's own change is this sign times the sum of its signed factors.
SIDE_SIGN = {"supplying": 1.0, "absorbing": -1.0}
REFRESH_RECEIPT = "refresh_receipt"
# Independent implementations (pure-Python normal equations versus column-scaled
# QR) must agree to this absolute bound in every point cell.
CROSS_CHECK_TOLERANCE = 1e-7
Z95 = 1.959963984540054


def quarter_label(ordinal: int) -> str:
    return f"{ordinal // 4}Q{ordinal % 4 + 1}"


def quarter_range(start: str, end: str) -> tuple[str, ...]:
    first, last = _quarter_ordinal(start), _quarter_ordinal(end)
    if last < first:
        raise ValueError(f"Empty quarter range {start}-{end}")
    return tuple(quarter_label(o) for o in range(first, last + 1))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pinned(root: Path, record: Mapping[str, Any]) -> Path:
    path = (root / record["path"]).resolve()
    path.relative_to(root.resolve())
    if not path.is_file():
        raise FileNotFoundError(f"Missing pinned input: {record['path']}")
    if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
        raise ValueError(f"Pinned input hash or size differs: {record['path']}")
    return path


def verify_refresh_receipt(root: Path, records: Mapping[str, Mapping[str, Any]], upstream_commit: str,
                           vintage: str) -> dict[str, Any]:
    """A refresh vintage consumes only objects its TDCest run receipt binds.

    Each pinned copy must equal a receipt output (path, hash and size), or be a
    tracked TDCest file declared at the same producer commit. Anything else raises.
    """
    if REFRESH_RECEIPT not in records:
        raise ValueError("Refresh vintage requires a pinned TDCest refresh receipt")
    record = records[REFRESH_RECEIPT]
    receipt = json.loads(_pinned(Path(root), record).read_text(encoding="utf-8"))
    producer = str(receipt.get("producers", {}).get("tdcest_exports_and_acquisition", ""))
    if receipt.get("vintage") != vintage or producer.split(" ")[0] != upstream_commit:
        raise ValueError("Refresh receipt vintage or TDCest producer differs from the declared design")
    outputs = receipt.get("outputs", {})
    bindings = {}
    for key, item in records.items():
        if key == REFRESH_RECEIPT:
            continue
        bound = outputs.get(item.get("upstream_path"))
        if bound is not None:
            if bound.get("sha256") != item["sha256"] or bound.get("bytes") != item["bytes"]:
                raise ValueError(f"Refresh receipt binds a different object for {key}")
            bindings[key] = "refresh_receipt_output"
        elif item.get("upstream_commit") == upstream_commit:
            bindings[key] = "tracked_file_at_upstream_commit"
        else:
            raise ValueError(f"Refresh input {key} is bound by neither the refresh receipt nor the producer commit")
    return {"path": record["path"], "sha256": record["sha256"], "vintage": vintage,
            "upstream_producer_commit": upstream_commit, "endpoints": receipt.get("endpoints"),
            "z1_vintage": receipt.get("z1_vintage", {}).get("release"), "bindings": bindings}


def _float(value: Any, label: str) -> float:
    text = str(value if value is not None else "").strip()
    if not text:
        return math.nan
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"Malformed {label}: {text!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite {label}")
    return number


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Required CSV has no rows: {path.name}")
    return rows


# --------------------------------------------------------------------------- design


@dataclass(frozen=True)
class Group:
    id: str
    label: str
    components: tuple[str, ...]
    outcomes: tuple[str, ...]


@dataclass(frozen=True)
class Design:
    raw: dict[str, Any]
    sha256: str
    sample: tuple[str, ...]
    eras: tuple[tuple[str, tuple[str, ...]], ...]
    transitions: tuple[str, ...]
    outcomes: tuple[str, ...]
    variants: dict[str, str | None]
    legs: tuple[tuple[str, str], ...]
    specs: dict[str, tuple[str, ...]]
    hac_lags: tuple[int, ...]
    groups: tuple[Group, ...]
    balances: tuple[dict[str, Any], ...] = ()

    @property
    def primary_outcome(self) -> str:
        return self.outcomes[0]

    @property
    def primary_variant(self) -> str:
        return next(iter(self.variants))

    @property
    def primary_spec(self) -> str:
        return next(iter(self.specs))

    @property
    def samples(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return (("pooled", self.sample), *self.eras)

    def components(self, outcome: str) -> tuple[str, ...]:
        return tuple(c for g in self.groups if outcome in g.outcomes for c in g.components)

    def groups_for(self, outcome: str) -> tuple[Group, ...]:
        return tuple(g for g in self.groups if outcome in g.outcomes)


def load_design(root: Path, locator: str = CONFIG_LOCATOR) -> Design:
    """The single loader for the declared reserve design and its taxonomy."""
    path = Path(root) / locator
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Reserve projection config schema differs")
    sample = quarter_range(raw["sample"]["start"], raw["sample"]["end"])
    if len(sample) != raw["sample"]["n"]:
        raise ValueError("Declared sample size differs from its quarter range")
    eras = tuple((e["id"], quarter_range(e["start"], e["end"])) for e in raw["eras"])
    transitions = tuple(raw["transition_quarters"])
    covered = [q for _, quarters in eras for q in quarters] + list(transitions)
    if sorted(covered, key=_quarter_ordinal) != list(sample):
        raise ValueError("Eras plus transition quarters must partition the pooled sample exactly")
    outcomes = sorted(raw["outcomes"], key=lambda k: raw["outcomes"][k]["role"] != "primary")
    if [raw["outcomes"][o]["role"] for o in outcomes].count("primary") != 1:
        raise ValueError("Exactly one primary outcome is required")
    variants = dict(sorted(raw["treatments"].items(), key=lambda kv: kv[1]["role"] != "primary"))
    if [v["role"] for v in variants.values()].count("primary") != 1:
        raise ValueError("Exactly one primary treatment is required")
    specs = dict(sorted(raw["specifications"].items(), key=lambda kv: kv[1]["role"] != "primary"))
    if [s["role"] for s in specs.values()].count("primary") != 1:
        raise ValueError("Exactly one primary specification is required")
    for spec_id, spec in specs.items():
        unknown = set(spec["controls"]) - set(ALLOWED_CONTROLS)
        if unknown or len(set(spec["controls"])) != len(spec["controls"]):
            # No contemporaneous reserve, deposit, factor or break control may enter.
            raise ValueError(f"Specification {spec_id} declares disallowed controls: {sorted(unknown)}")
    hac = raw["hac"]
    lags = (int(hac["primary_lags"]), *(int(v) for v in hac["sensitivity_lags"]))
    if len(set(lags)) != len(lags) or min(lags) < 0:
        raise ValueError("HAC lags must be distinct and nonnegative")
    groups = tuple(Group(g["id"], g["label"], tuple(g["components"]), tuple(g.get("outcomes", outcomes)))
                   for g in raw["component_groups"])
    for outcome in outcomes:
        names = [c for g in groups if outcome in g.outcomes for c in g.components]
        if len(names) != len(set(names)):
            raise ValueError(f"A component is assigned to more than one group for {outcome}")
        if "f_rrp_others" in names:
            raise ValueError("RRP others must enter only net of the route term (n_rrp_others)")
    legs = tuple((leg["id"], leg["column"]) for leg in raw["legs"])
    if [c for _, c in legs] != ["c_m", "b", "gamma"]:
        raise ValueError("Legs must be the cash leg, Fed bridge and route term")
    for key in ("replicates", "bandwidth_quarters", "monte_carlo_batches"):
        if int(raw["bootstrap"][key]) < 2:
            raise ValueError(f"Bootstrap {key} is too small")
    if raw["bootstrap"]["method"] != "dependent_wild" or raw["bootstrap"]["kernel"] != "bartlett_calendar_distance":
        raise ValueError("Unsupported dependent-wild bootstrap declaration")
    if int(raw["bootstrap"]["replicates"]) < 8:
        raise ValueError("Bootstrap replicates cannot estimate endpoint Monte Carlo error")
    coverage = raw["calibration"]["coverage"]
    if (coverage["nominal_level"] != raw["bootstrap"]["level"] or
            not 0 < coverage["minimum_acceptable"] < coverage["nominal_level"] < 1 or
            coverage["mc_se_multiplier"] < 0):
        raise ValueError("Coverage rule must match the interval level and declare a lower threshold")
    by_id = {g.id: g for g in groups}
    balances = tuple(raw.get("balance_coefficients", ()))
    for balance in balances:
        group = by_id.get(balance["group"])
        # The balance must be one factor group present for every outcome; its sign comes
        # from the upstream factor side, verified against the ledger receipt at load.
        if (group is None or set(group.outcomes) != set(outcomes) or balance["side"] not in SIDE_SIGN
                or not all(c.startswith("f_") for c in group.components)):
            raise ValueError(f"Balance coefficient {balance.get('id')} must name a signed-factor group")
    if len({b["id"] for b in balances}) != len(balances):
        raise ValueError("Balance coefficient ids must be unique")
    if REFRESH_RECEIPT in raw["inputs"] and not raw.get("vintage"):
        raise ValueError("A refresh receipt input requires a declared vintage")
    return Design(raw=raw, sha256=_sha256(path), sample=sample, eras=eras, transitions=transitions,
                  outcomes=tuple(outcomes), variants={k: v["gamma_source"] for k, v in variants.items()},
                  legs=legs, specs={k: tuple(v["controls"]) for k, v in specs.items()},
                  hac_lags=lags, groups=groups, balances=balances)


def verify_frozen_sources(root: Path, design: Design) -> list[dict[str, Any]]:
    """The frozen deposit producers' numerical sources must be byte-identical."""
    verified = []
    for record in design.raw["frozen_deposit_sources"]:
        path = Path(root) / record["path"]
        if not path.is_file() or path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise ValueError(f"Frozen deposit source changed: {record['path']}")
        verified.append(dict(record))
    return verified


# ---------------------------------------------------------------------------- panel


@dataclass(frozen=True)
class Panel:
    quarters: tuple[str, ...]
    ordinals: np.ndarray
    qoy: np.ndarray
    common: dict[str, np.ndarray]
    by_variant: dict[str, dict[str, np.ndarray]]
    labels: dict[str, str]
    clocks: dict[str, str]
    checks: dict[str, Any]

    def column(self, variant: str, name: str) -> np.ndarray:
        return self.by_variant[variant].get(name, self.common.get(name))

    def positions(self, quarters: Sequence[str]) -> np.ndarray:
        index = {q: i for i, q in enumerate(self.quarters)}
        return np.asarray([index[q] for q in quarters], dtype=int)


VARIANT_COLUMNS = ("x", "gamma", "n_rrp_others")


def _assert_close(label: str, actual: np.ndarray, expected: np.ndarray, checks: dict[str, Any]) -> None:
    gap = np.abs(np.asarray(actual) - np.asarray(expected))
    scale = np.maximum(np.abs(actual), np.abs(expected))
    if not np.all(np.isfinite(gap)) or np.any(gap > numerical_tolerance(scale)):
        worst = int(np.nanargmax(gap)) if np.isfinite(gap).any() else 0
        raise ValueError(f"{label} fails at sample position {worst}: gap {float(gap[worst])}")
    checks[label] = max(checks.get(label, 0.0), float(gap.max(initial=0.0)))


def build_panel(design: Design, components: Sequence[Mapping[str, str]], bridges: Sequence[Mapping[str, str]],
                ledger: Sequence[Mapping[str, str]], receipt: Mapping[str, Any]) -> Panel:
    """Validate the pinned TDCest objects against each other and the identity."""
    upstream = design.raw["upstream"]
    inputs = design.raw["inputs"]
    if (receipt.get("schema_version") != upstream["receipt_schema_version"]
            or receipt.get("producer_commit") != upstream["producer_commit"]
            or receipt.get("units") != design.raw["units"]
            or receipt.get("outputs", {}).get("quarterly", {}).get("sha256") != inputs["ledger"]["sha256"]
            or receipt.get("outputs", {}).get("bridges", {}).get("sha256") != inputs["bridges"]["sha256"]
            or receipt.get("inputs", {}).get("trc_quarterly_components_v1", {}).get("sha256") != inputs["trc_components"]["sha256"]):
        raise ValueError("Ledger receipt does not bind the pinned producer, units and objects")
    by_variant_rows: dict[str, list[Mapping[str, str]]] = {}
    for row in bridges:
        by_variant_rows.setdefault(row["trc_variant"], []).append(row)
    if set(by_variant_rows) != set(design.variants):
        raise ValueError("Bridge variants differ from the declared treatments")
    shell = tuple(r["quarter"] for r in by_variant_rows[design.primary_variant])
    ordinals = np.asarray([_quarter_ordinal(q) for q in shell])
    if np.any(np.diff(ordinals) != 1):
        raise ValueError("Bridge quarters must be unique, ordered and consecutive")
    if any(tuple(r["quarter"] for r in rows) != shell for rows in by_variant_rows.values()):
        raise ValueError("Every variant must share the same quarter shell")
    if not set(design.sample) <= set(shell):
        raise ValueError("Declared sample is outside the bridge shell")
    sample = np.asarray([q in set(design.sample) for q in shell])
    for rows in by_variant_rows.values():
        if any(r["bridge_status"] != RECONCILED_STATUS for r, keep in zip(rows, sample) if keep):
            raise ValueError("A sample quarter is not a reconciled bridge row")

    fields = list(bridges[0])
    factor_columns = [c for c in fields if c.startswith("f_")]
    factor_map = {f["key"]: f for f in receipt["factor_map"]}
    h41_treasury = {"treasury_general_account", "supplementary_financing_account", "treasury_securities"}
    expected_factors = {f"f_{k}" for k, f in factor_map.items() if f["role"] == "factor"} - {f"f_{k}" for k in h41_treasury}
    if set(factor_columns) != expected_factors or "f_rrp_others" not in factor_columns:
        raise ValueError("Bridge factor columns differ from the ledger factor map")
    declared = {c for o in design.outcomes for c in design.components(o)}
    if {c for c in declared if c.startswith("f_")} != set(factor_columns) - {"f_rrp_others"}:
        raise ValueError("Config groups must map every signed factor except gross RRP others exactly once")
    for balance in design.balances:
        group = next(g for g in design.groups if g.id == balance["group"])
        if {factor_map.get(c[2:], {}).get("side") for c in group.components} != {balance["side"]}:
            raise ValueError(f"Balance {balance['id']} side differs from the ledger factor map")

    def array(rows: Sequence[Mapping[str, str]], name: str) -> np.ndarray:
        return np.asarray([_float(r.get(name), name) for r in rows])

    primary_rows = by_variant_rows[design.primary_variant]
    common_names = ["y_h", "y_z", "c_m", "b", "c_h", "s_h", *factor_columns, "bridge_outcome",
                    "bridge_cash", "bridge_securities", "rho",
                    *sorted(c for c in declared if c.startswith("bridge_outcome_"))]
    common = {name: array(primary_rows, name) for name in common_names}
    by_variant = {v: {name: array(rows, name) for name in VARIANT_COLUMNS} for v, rows in by_variant_rows.items()}
    checks: dict[str, Any] = {}
    for name, values in [*common.items(), *((f"{v}:{n}", a) for v, cols in by_variant.items() for n, a in cols.items())]:
        if not np.all(np.isfinite(values[sample])):
            raise ValueError(f"Missing sample value for {name}")
    for variant, rows in by_variant_rows.items():
        for name in common_names:
            if not np.array_equal(array(rows, name)[sample], common[name][sample]):
                raise ValueError(f"Common column {name} differs in {variant}")

    comp_by_q = {r["quarter"]: r for r in components}
    comp = {name: np.asarray([_float(comp_by_q.get(q, {}).get(name), name) for q in shell])
            for name in ("trc_cash_leg", "trc_fed_bridge", "delta_rb_z1", *design.variants,
                         *(s for s in design.variants.values() if s))}
    ledger_by_q = {r["quarter"]: r for r in ledger}
    if not set(design.sample) <= set(ledger_by_q):
        raise ValueError("Ledger does not cover the declared sample")
    led = {name: np.asarray([_float(ledger_by_q.get(q, {}).get(name), name) for q in shell])
           for name in ("delta_reserve_balances", "c_h", "s_h", "rho", "f_treasury_general_account",
                        "f_supplementary_financing_account", "f_treasury_securities", *factor_columns)}
    s = sample
    _assert_close("cash_leg_equals_trc_cash_leg", common["c_m"][s], comp["trc_cash_leg"][s], checks)
    _assert_close("fed_bridge_equals_trc_fed_bridge", common["b"][s], comp["trc_fed_bridge"][s], checks)
    _assert_close("y_z_equals_delta_rb_z1", common["y_z"][s], comp["delta_rb_z1"][s], checks)
    _assert_close("y_h_equals_ledger_reserve_change", common["y_h"][s], led["delta_reserve_balances"][s], checks)
    _assert_close("c_h_equals_ledger_tga_plus_sfp", common["c_h"][s],
                  (led["f_treasury_general_account"] + led["f_supplementary_financing_account"])[s], checks)
    _assert_close("s_h_equals_ledger_treasury_securities", common["s_h"][s], led["f_treasury_securities"][s], checks)
    _assert_close("rho_equals_ledger_rho", common["rho"][s], led["rho"][s], checks)
    for name in factor_columns:
        _assert_close("factors_equal_ledger", common[name][s], led[name][s], checks)
    _assert_close("bridge_cash_is_c_h_minus_c_m", common["bridge_cash"][s], (common["c_h"] - common["c_m"])[s], checks)
    _assert_close("bridge_securities_is_s_h_minus_b", common["bridge_securities"][s], (common["s_h"] - common["b"])[s], checks)
    _assert_close("bridge_outcome_is_y_z_minus_y_h", common["bridge_outcome"][s], (common["y_z"] - common["y_h"])[s], checks)
    parts = [c for c in common if c.startswith("bridge_outcome_")]
    _assert_close("bridge_outcome_split_adds_up", common["bridge_outcome"][s], sum(common[c] for c in parts)[s], checks)
    for variant, source in design.variants.items():
        cols = by_variant[variant]
        gamma = comp[source] if source else np.zeros(len(shell))
        _assert_close("gamma_equals_route_source", cols["gamma"][s], gamma[s], checks)
        _assert_close("x_equals_admitted_variant", cols["x"][s], comp[variant][s], checks)
        _assert_close("x_equals_sum_of_legs", cols["x"][s], (common["c_m"] + common["b"] + cols["gamma"])[s], checks)
        # The route already inside TRC is removed from RRP others exactly once.
        _assert_close("rrp_others_net_of_gamma_once", cols["n_rrp_others"][s], (common["f_rrp_others"] - cols["gamma"])[s], checks)
        for outcome in design.outcomes:
            total = sum((cols[c] if c in cols else common[c]) for c in design.components(outcome))
            _assert_close(f"identity_{outcome}", (common[outcome] - cols["x"])[s], total[s], checks)
    checks["identity_rows_per_variant_outcome"] = int(s.sum())

    definitions = receipt.get("definitions", {})
    clocks_source = receipt.get("column_clocks", {}).get("bridges", {})
    labels, clocks = {}, {}
    for name in sorted(declared | {"x", "gamma", "c_m", "b", "y_h", "y_z"}):
        key = name[2:] if name.startswith("f_") else None
        label = factor_map[key]["label"] if key else definitions.get(name)
        clock = clocks_source.get(name, {}).get("clock")
        if not label or not clock:
            raise ValueError(f"Ledger receipt lacks a label or clock for {name}")
        labels[name], clocks[name] = label, clock
    return Panel(quarters=shell, ordinals=ordinals, qoy=ordinals % 4 + 1, common=common,
                 by_variant=by_variant, labels=labels, clocks=clocks, checks=checks)


def load_panel(root: Path, design: Design) -> Panel:
    files = {key: _pinned(Path(root), record) for key, record in design.raw["inputs"].items()}
    refresh = (verify_refresh_receipt(root, design.raw["inputs"], design.raw["upstream"]["producer_commit"],
                                      design.raw["vintage"]) if design.raw.get("vintage") else None)
    receipt = json.loads(files["ledger_receipt"].read_text(encoding="utf-8"))
    panel = build_panel(design, _read_csv(files["trc_components"]), _read_csv(files["bridges"]),
                        _read_csv(files["ledger"]), receipt)
    if refresh:
        panel.checks["refresh_receipt"] = refresh
    return panel


# ----------------------------------------------------------------------- arithmetic


def _ranks(x: np.ndarray) -> np.ndarray:
    """Column-scaled rank, for one design or a stack of replicate designs."""
    if x.ndim == 2:
        return np.asarray(_scaled_rank(x))
    norms = np.linalg.norm(x, axis=-2, keepdims=True)
    return np.linalg.matrix_rank(x / np.where(norms > 0, norms, 1.0))


def residualize(z: np.ndarray, m: np.ndarray, *, require: Sequence[int] = ()) -> np.ndarray:
    """Residuals of every column of m on the same rows and full-rank controls z.

    Stacked leading dimensions are replicates. A rank-deficient control design,
    or a required column (the treatment) without partial variation beyond the
    controls, raises; no column is ever dropped.
    """
    z, m = np.asarray(z, dtype=float), np.asarray(m, dtype=float)
    if z.shape[:-1] != m.shape[:-1] or not np.isfinite(z).all() or not np.isfinite(m).all():
        raise ValueError("Residualization requires aligned finite rows on one common mask")
    n, k = z.shape[-2:]
    if np.any(_ranks(z) != k) or n <= k + len(require):
        raise ValueError("Control design is rank deficient on this mask")
    if require and np.any(_ranks(np.concatenate([z, m[..., list(require)]], axis=-1)) != k + len(require)):
        raise ValueError("Treatment has no partial variation beyond the controls (rank deficient)")
    norms = np.linalg.norm(z, axis=-2, keepdims=True)
    q, _ = np.linalg.qr(z / norms)
    return m - q @ (np.swapaxes(q, -1, -2) @ m)


def _check(label: str, gap: np.ndarray, scale: np.ndarray) -> float:
    gap = np.abs(np.asarray(gap, dtype=float))
    tolerance = numerical_tolerance(np.abs(np.asarray(scale, dtype=float)))
    if not np.all(gap <= tolerance):
        raise ValueError(f"{label} fails: max gap {float(np.nanmax(gap))}")
    return float(gap.max(initial=0.0))


def accounting(resid: np.ndarray, x: int, y: int, components: Sequence[int], legs: Sequence[int]) -> dict[str, Any]:
    """Slope, factor contributions and leg-by-factor matrix on one residualized mask."""
    xt, yt = resid[..., x], resid[..., y]
    nt, lt = resid[..., list(components)], resid[..., list(legs)]
    denominator = np.einsum("...n,...n->...", xt, xt)
    if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0):
        raise ValueError("Common treatment denominator is not positive")
    beta = np.einsum("...n,...n->...", xt, yt) / denominator
    contributions = np.einsum("...n,...nk->...k", xt, nt) / denominator[..., None]
    matrix = np.einsum("...nj,...nk->...jk", lt, nt) / denominator[..., None, None]
    target = beta - 1
    leg_rows = matrix.sum(axis=-1)
    direct_legs = np.einsum("...nj,...n->...j", lt, yt - xt) / denominator[..., None]
    scale = np.maximum(np.abs(contributions).max(axis=-1), np.abs(target))
    gaps = {
        "contribution_sum_gap": _check("Contributions sum to beta - 1", contributions.sum(axis=-1) - target, scale),
        "matrix_column_sum_gap": _check("Leg-by-factor column sums equal contributions",
                                        matrix.sum(axis=-2) - contributions, np.maximum(np.abs(matrix).max(axis=-2), np.abs(contributions))),
        "matrix_row_sum_gap": _check("Leg-by-factor row sums equal leg covariances", leg_rows - direct_legs,
                                     np.maximum(np.abs(matrix).max(axis=-1), np.abs(direct_legs))),
        "matrix_grand_sum_gap": _check("Leg-by-factor grand sum equals beta - 1", matrix.sum(axis=(-2, -1)) - target,
                                       np.maximum(np.abs(matrix).max(axis=(-2, -1)), np.abs(target))),
    }
    return {"denominator": denominator, "beta": beta, "contributions": contributions, "matrix": matrix,
            "leg_rows": leg_rows, "gaps": gaps}


def _lag1(values: np.ndarray, ordinals: np.ndarray) -> np.ndarray:
    """One calendar-quarter lag formed on the full calendar before any masking."""
    lagged = np.full(len(values), np.nan)
    previous = {int(o): v for o, v in zip(ordinals, values)}
    for i, ordinal in enumerate(ordinals):
        lagged[i] = previous.get(int(ordinal) - 1, np.nan)
    return lagged


def control_matrix(qoy: np.ndarray, controls: Sequence[str], lags: Mapping[str, np.ndarray]) -> np.ndarray:
    """Intercept plus declared controls only; never a contemporaneous reserve series."""
    columns = [np.ones(qoy.shape)]
    for control in controls:
        if control == "quarter_of_year":
            columns.extend((qoy == q).astype(float) for q in (2, 3, 4))
        elif control in ("lag1_treatment", "lag1_outcome"):
            columns.append(lags[control])
        else:
            raise ValueError(f"Disallowed control {control}")
    return np.stack(columns, axis=-1)


@dataclass(frozen=True)
class Cell:
    outcome: str
    variant: str
    sample: str
    spec: str
    positions: np.ndarray
    dropped: tuple[str, ...]


def cell_columns(panel: Panel, design: Design, outcome: str, variant: str) -> tuple[list[str], np.ndarray]:
    names = ["x", outcome, *design.components(outcome), *(c for _, c in design.legs)]
    return names, np.column_stack([panel.column(variant, n) for n in names])


def cell_lags(panel: Panel, outcome: str, variant: str) -> dict[str, np.ndarray]:
    return {"lag1_treatment": _lag1(panel.column(variant, "x"), panel.ordinals),
            "lag1_outcome": _lag1(panel.common[outcome], panel.ordinals)}


def make_cell(panel: Panel, design: Design, outcome: str, variant: str, sample: str, spec: str,
              quarters: Sequence[str]) -> Cell:
    positions = panel.positions(quarters)
    controls = design.specs[spec]
    lags = cell_lags(panel, outcome, variant)
    usable = np.ones(len(positions), dtype=bool)
    for control in controls:
        if control in lags:
            usable &= np.isfinite(lags[control][positions])
    dropped = tuple(q for q, keep in zip(quarters, usable) if not keep)
    return Cell(outcome, variant, sample, spec, positions[usable], dropped)


def _normal_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def fit_cell(panel: Panel, design: Design, cell: Cell) -> dict[str, Any]:
    names, data = cell_columns(panel, design, cell.outcome, cell.variant)
    lags = {k: v[cell.positions] for k, v in cell_lags(panel, cell.outcome, cell.variant).items()}
    z = control_matrix(panel.qoy[cell.positions], design.specs[cell.spec], lags)
    m = data[cell.positions]
    resid = residualize(z, m, require=(0,))
    n_comp = len(design.components(cell.outcome))
    result = accounting(resid, 0, 1, range(2, 2 + n_comp), range(2 + n_comp, 5 + n_comp))
    # Independent check against the frozen pure-Python residualizer.
    controls = z[:, 1:].tolist()
    for j in (0, 1):
        reference = np.asarray(_residualize(m[:, j].tolist(), controls))
        scale = max(float(np.abs(m[:, j]).max()), 1.0)
        if np.abs(reference - resid[:, j]).max() > CROSS_CHECK_TOLERANCE * scale:
            raise ValueError("Residualization differs from the frozen reference residualizer")
    x, y = m[:, 0], m[:, 1]
    centered = x - x.mean()
    total_ss = float(centered @ centered)
    hac = {}
    design_matrix = np.column_stack([np.ones(len(x)), x, z[:, 1:]])
    ordinals = [int(o) for o in panel.ordinals[cell.positions]]
    beta = float(result["beta"])
    for lags_count in design.hac_lags:
        fit = calendar_hac(y, design_matrix, ordinals, lags=lags_count)
        gap = abs(fit.beta[1] - beta)
        if gap > CROSS_CHECK_TOLERANCE:
            raise ValueError(f"Calendar HAC coefficient differs from the residualized slope by {gap}")
        se = fit.ses[1]
        hac[lags_count] = {"se": se, "p_beta_zero": _normal_p(beta / se), "p_beta_one": _normal_p((beta - 1) / se),
                           "fwl_gap": gap, "finite_sample_scale": len(x) / (len(x) - design_matrix.shape[1])}
    balances = {}
    components = design.components(cell.outcome)
    for balance in design.balances:
        group = next(g for g in design.groups if g.id == balance["group"])
        index = [components.index(c) for c in group.components]
        sign = SIDE_SIGN[balance["side"]]
        # The balance's own change regressed on TRC is the sign times its covariance contribution.
        coefficient = sign * float(result["contributions"][index].sum())
        series = sign * m[:, [2 + i for i in index]].sum(axis=1)
        balance_hac = {}
        for lags_count in design.hac_lags:
            fit = calendar_hac(series, design_matrix, ordinals, lags=lags_count)
            gap = abs(fit.beta[1] - coefficient)
            if gap > CROSS_CHECK_TOLERANCE:
                raise ValueError(f"Balance {balance['id']} HAC coefficient differs from its signed contribution by {gap}")
            balance_hac[lags_count] = {"se": fit.ses[1], "p_zero": _normal_p(coefficient / fit.ses[1]), "gap": gap}
        balances[balance["id"]] = {"coefficient": coefficient, "sign": sign, "hac": balance_hac}
    return {"names": names, "resid": resid, "x": x, "y": y, "total_centered_ss": total_ss,
            "partial_variation_share": float(result["denominator"]) / total_ss, "k": design_matrix.shape[1],
            "hac": hac, "balances": balances, **result}


# --------------------------------------------------------------------- era contrasts


def era_design(x: np.ndarray, era: np.ndarray, qoy: np.ndarray, controls: Sequence[str],
               lags: Mapping[str, np.ndarray], n_eras: int) -> np.ndarray:
    """Stacked design: era slopes first, then era intercepts and era-specific controls."""
    blocks = [x * (era == e) for e in range(n_eras)]
    for e in range(n_eras):
        within = control_matrix(qoy, controls, lags) * (era == e)[:, None]
        blocks.extend(within.T)
    return np.column_stack(blocks)


CONTRAST = np.asarray([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])


def era_contrast_wald(x: np.ndarray, y: np.ndarray, era: np.ndarray, qoy: np.ndarray, ordinals: Sequence[int],
                      controls: Sequence[str], lags: Mapping[str, np.ndarray], hac_lags: int) -> dict[str, Any]:
    """Joint HAC Wald test of the two era-slope contrasts from one stacked regression."""
    design_matrix = era_design(x, era, qoy, controls, lags, 3)
    fit = calendar_hac(y, design_matrix, [int(o) for o in ordinals], lags=hac_lags)
    slopes = np.asarray(fit.beta[:3])
    covariance = np.asarray(fit.covariance)[:3, :3]
    delta = CONTRAST @ slopes
    v = CONTRAST @ covariance @ CONTRAST.T
    wald = float(delta @ np.linalg.solve(v, delta))
    return {"slopes": slopes, "delta": delta, "delta_se": np.sqrt(np.diag(v)), "wald": wald,
            "p_value": math.exp(-wald / 2.0), "n": len(y), "k": design_matrix.shape[1],
            "finite_sample_scale": len(y) / (len(y) - design_matrix.shape[1])}


# ------------------------------------------------------------------------- bootstrap


def dependent_wild_weights(rng: np.random.Generator, ordinals: np.ndarray,
                           bandwidth: int, replicates: int) -> np.ndarray:
    """Gaussian MA multipliers: Cov(W_t,W_s) = (1 - |t-s|/bandwidth)_+.

    Generate on the entire calendar, including masked transition dates. Taking
    columns afterward preserves dependence across boundaries and calendar gaps.
    No observation is sampled, duplicated, omitted, or given a regression weight.
    """
    ordinals = np.asarray(ordinals, dtype=int)
    if bandwidth < 2 or replicates < 2 or np.any(np.diff(ordinals) <= 0):
        raise ValueError("Wild multipliers require ordered quarters and positive counts")
    span = int(ordinals[-1] - ordinals[0] + 1)
    shocks = rng.standard_normal((replicates, span + bandwidth - 1))
    sums = np.cumsum(np.pad(shocks, ((0, 0), (1, 0))), axis=1)
    weights = (sums[:, bandwidth:] - sums[:, :-bandwidth]) / math.sqrt(bandwidth)
    return weights[:, ordinals - ordinals[0]]


def wild_responses(data: np.ndarray, z: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Unrestricted multivariate residual bootstrap conditional on observed x.

    Regress every column on the same [controls,x], then reweight residuals with
    the same temporal multipliers. The treatment column is fixed exactly; the
    linear identities among outcome, factors and legs are preserved.
    """
    if weights.shape[1] != len(data) or not np.isfinite(weights).all():
        raise ValueError("Wild weights must cover every observed quarter")
    errors = residualize(np.column_stack([z, data[:, 0]]), data)
    errors[:, 0] = 0.0
    return data[None, :, :] - errors[None, :, :] + weights[..., None] * errors[None, :, :]


def _endpoint_mc_se(values: np.ndarray, endpoint, batches: int) -> np.ndarray:
    """Disjoint-batch estimate of simulation error, including endpoint refits."""
    batches = min(batches, len(values) // 4)
    if batches < 2:
        raise ValueError("At least eight replicates are required to estimate endpoint Monte Carlo error")
    endpoints = np.asarray([endpoint(v) for v in np.array_split(values, batches)])
    return endpoints.std(axis=0, ddof=1) / math.sqrt(batches)


def joint_null_inference(delta_hat: np.ndarray, null_star: np.ndarray) -> dict[str, Any]:
    """Joint fixed-scale maximum test against restricted draws, shared by production and calibration."""
    null_se = null_star.std(axis=0, ddof=1)
    if np.any(~np.isfinite(null_se)) or np.any(null_se <= 0):
        raise ValueError("Restricted bootstrap contrast standard error is not positive")
    maximum_null = np.max(np.abs(null_star) / null_se, axis=-1)
    maximum_observed = float(np.max(np.abs(delta_hat) / null_se))
    p = float((1 + np.sum(maximum_null >= maximum_observed)) / (len(maximum_null) + 1))
    return {"maximum_observed": maximum_observed, "p_value": p,
            "p_mc_se": math.sqrt(p * (1 - p) / len(maximum_null)), "replicates": len(maximum_null)}


def fixed_scale_band(delta_hat: np.ndarray, star: np.ndarray, level: float) -> tuple:
    """Maximum absolute centered contrast with one bootstrap SD per contrast.

    Scales are fixed across draws; replicate regression SEs are not computed.
    This exact band construction is shared by production and coverage checks.
    """
    se = star.std(axis=0, ddof=1)
    if np.any(~np.isfinite(se)) or np.any(se <= 0):
        raise ValueError("Bootstrap contrast standard error is not positive")
    maximum = np.max(np.abs(star - delta_hat) / se, axis=-1)
    critical = float(np.quantile(maximum, level))
    return se, critical, delta_hat - critical * se, delta_hat + critical * se


def fixed_scale_maximum_inference(delta_hat: np.ndarray, delta_star: np.ndarray, level: float,
                                  null_star: np.ndarray, batches: int = 20) -> dict[str, Any]:
    """Unrestricted simultaneous band; separately null-imposed joint p-value.

    Each distribution uses its own fixed bootstrap SD scale. The null draws
    come from the restricted common-slope fit, never from recentering the
    unrestricted draws. Band simulation error reestimates scales in each batch.
    """
    def band(star):
        return fixed_scale_band(delta_hat, star, level)

    se, critical, lower, upper = band(delta_star)
    joint = joint_null_inference(delta_hat, null_star)
    endpoint_se = _endpoint_mc_se(delta_star, lambda v: np.stack(band(v)[2:]), batches)
    return {"se": se, "critical_value": critical, "lower": lower, "upper": upper,
            "lower_mc_se": endpoint_se[0], "upper_mc_se": endpoint_se[1],
            **joint}


def era_wild_slopes(x: np.ndarray, y: np.ndarray, era: np.ndarray, qoy: np.ndarray,
                    controls: Sequence[str], weights: np.ndarray, *, restricted: bool) -> np.ndarray:
    """Refit fixed-date era slopes after a restricted or unrestricted wild fit.

    FWL projection maps are exact refits for the fixed observed design. They
    avoid repeating the same QR in calibration; no treatment row can disappear.
    """
    n_eras = 3
    full = era_design(x, era, qoy, controls, {}, n_eras)
    if weights.ndim != 2 or weights.shape[1] != len(y) or not np.isfinite(weights).all():
        raise ValueError("Wild weights must cover every observed quarter")
    model = np.column_stack([x, full[:, n_eras:]]) if restricted else full
    errors = residualize(model, y[:, None])[:, 0]
    pseudo = (y - errors)[None, :] + weights * errors[None, :]
    slopes = []
    for e in range(n_eras):
        mask = era == e
        z = control_matrix(qoy[mask], controls, {})
        xt = residualize(z, x[mask, None], require=(0,))[:, 0]
        # xt is orthogonal to z, so this is the refitted partial-out slope.
        denominator = xt @ xt
        slopes.append(pseudo[:, mask] @ xt / denominator)
    return np.column_stack(slopes)


def run_bootstrap(panel: Panel, design: Design) -> tuple[dict, dict]:
    """One full-calendar path shared across all series, fits, eras and tests."""
    config = design.raw["bootstrap"]
    rng = np.random.default_rng(int(config["seed"]))
    replicates = int(config["replicates"])
    if any(c != "quarter_of_year" for s in config["specifications"] for c in design.specs[s]):
        raise ValueError("Bootstrap specifications cannot carry lag controls")
    # Generated ONCE outside every outcome/spec/sample/variant loop.
    weights = dependent_wild_weights(rng, panel.ordinals, int(config["bandwidth_quarters"]), replicates)
    out = {}
    for spec in config["specifications"]:
        for sample, quarters in design.samples:
            positions = panel.positions(quarters)
            z = control_matrix(panel.qoy[positions], design.specs[spec], {})
            for outcome in design.outcomes:
                n_comp = len(design.components(outcome))
                for variant in design.variants:
                    _, data = cell_columns(panel, design, outcome, variant)
                    accum = {name: [] for name in ("beta", "contributions", "leg_rows", "matrix_grand_sum")}
                    for start in range(0, replicates, 256):
                        w = weights[start:start + 256, positions]
                        pseudo = wild_responses(data[positions], z, w)
                        resid = residualize(np.broadcast_to(z, (*w.shape, z.shape[1])), pseudo, require=(0,))
                        acc = accounting(resid, 0, 1, range(2, 2 + n_comp), range(2 + n_comp, 5 + n_comp))
                        for name in accum:
                            accum[name].append(acc["matrix"].sum(axis=(-2, -1)) if name == "matrix_grand_sum" else acc[name])
                    out[(outcome, variant, sample, spec)] = {name: np.concatenate(v) for name, v in accum.items()}
        positions, era, _ = _era_arrays(panel, design)
        for outcome in design.outcomes:
            for variant in design.variants:
                slopes = era_wild_slopes(panel.column(variant, "x")[positions], panel.common[outcome][positions],
                                         era, panel.qoy[positions], design.specs[spec], weights[:, positions], restricted=True)
                out[(outcome, variant, "joint_null", spec)] = {"delta": slopes @ CONTRAST.T}
    representation = {"replicates": replicates, "full_calendar_quarters": len(panel.quarters),
                      "analysis_quarters": len(design.sample),
                      "per_quarter": [{"quarter": q, "replicates_present": replicates,
                                       "expected_multiplicity": 1.0, "omission_rate": 0.0} for q in design.sample]}
    return out, representation


# ----------------------------------------------------------------------- calibration


def _era_arrays(panel: Panel, design: Design) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    era_positions = [panel.positions(q) for _, q in design.eras]
    positions = np.concatenate(era_positions)
    era = np.concatenate([np.full(len(p), e) for e, p in enumerate(era_positions)])
    return positions, era, era_positions


def calibrate(panel: Panel, design: Design) -> list[dict[str, Any]]:
    """Test size and unrestricted interval coverage under the same specified nulls.

    The null keeps the observed treatment design, including its influential
    quarters: outcomes are the fitted equal-slope mean plus errors scaled by the
    absolute restricted residuals and driven by a calendar AR(1) process.
    """
    config = design.raw["calibration"]
    replications, inner = int(config["replications"]), int(config["bootstrap_replicates"])
    bandwidth = int(design.raw["bootstrap"]["bandwidth_quarters"])
    alpha, ceiling = float(config["nominal_level"]), float(config["max_acceptable_size"])
    variant = config["treatment"]
    positions, era, _ = _era_arrays(panel, design)
    sample_positions = panel.positions(design.sample)
    x_all = panel.column(variant, "x")
    x, qoy, ordinals = x_all[positions], panel.qoy[positions], panel.ordinals[positions]
    adjacent = [(i, j) for i in range(len(ordinals)) for j in range(len(ordinals)) if ordinals[i] - ordinals[j] == 1]
    rows = []
    for outcome, spec in ((o, s) for o in config["outcomes"] for s in config["specifications"]):
        controls = design.specs[spec]
        if any(c != "quarter_of_year" for c in controls):
            raise ValueError("Calibration specifications cannot carry lag controls")
        y = panel.common[outcome][positions]
        null_design = np.column_stack([x, era_design(np.zeros_like(x), era, qoy, controls, {}, 3)[:, 3:]])
        u = residualize(null_design, y[:, None])[:, 0]
        xt = residualize(null_design[:, 1:], x[:, None], require=(0,))[:, 0]
        true_slopes = np.full(3, (xt @ y) / (xt @ xt))
        targets = coverage_targets(design)
        level = float(config["coverage"]["nominal_level"])
        tail = (1 - level) / 2
        phi_hat = float(sum(u[i] * u[j] for i, j in adjacent) / (u @ u))
        for dgp_index, (dgp, innovation) in enumerate((g, v) for g in config["dgps"] for v in config["innovations"]):
            phi = phi_hat if dgp["ar1"] == "estimated_calendar_lag1" else float(dgp["ar1"])
            if not abs(phi) < 1 or dgp["scale"] != "absolute_restricted_residual":
                raise ValueError("Calibration DGP is not a stationary declared design")
            rng = np.random.default_rng(np.random.SeedSequence([int(config["seed"]), dgp_index]))
            rejections = {"hac_wald": 0, "bootstrap_fixed_scale_maximum": 0}
            covered = np.zeros(len(targets), dtype=int)
            for _ in range(replications):
                n = len(sample_positions)
                if innovation["distribution"] == "normal":
                    shocks = rng.standard_normal(n + 256)
                elif innovation["distribution"] == "student_t" and int(innovation["df"]) > 2:
                    df = int(innovation["df"])
                    shocks = rng.standard_t(df, n + 256) * math.sqrt((df - 2) / df)
                else:
                    raise ValueError("Unsupported calibration innovation")
                process = np.empty(len(shocks))
                process[0] = shocks[0]
                for t in range(1, len(shocks)):
                    process[t] = phi * process[t - 1] + math.sqrt(1 - phi * phi) * shocks[t]
                e = np.zeros(len(panel.quarters))
                e[sample_positions] = process[256:]
                y_sim = (y - u) + np.abs(u) * e[positions]
                wald = era_contrast_wald(x, y_sim, era, qoy, ordinals, controls, {}, int(config["hac_lags"]))
                weights = dependent_wild_weights(rng, panel.ordinals, bandwidth, inner)
                null_star = era_wild_slopes(x, y_sim, era, qoy, controls, weights[:, positions], restricted=True) @ CONTRAST.T
                p = joint_null_inference(wald["delta"], null_star)["p_value"]
                rejections["hac_wald"] += wald["p_value"] <= alpha
                rejections["bootstrap_fixed_scale_maximum"] += p <= alpha
                # Reuse the size DGP and weights without advancing its random stream.
                slopes = era_wild_slopes(x, y_sim, era, qoy, controls, weights[:, positions], restricted=False)
                lower, upper = np.quantile(slopes, [tail, 1 - tail], axis=0)
                slope_hits = (lower <= true_slopes) & (true_slopes <= upper)
                _, _, lower, upper = fixed_scale_band(wald["delta"], slopes @ CONTRAST.T, level)
                true_delta = CONTRAST @ true_slopes
                contrast_hits = (lower <= true_delta) & (true_delta <= upper)
                covered += np.r_[slope_hits, contrast_hits, contrast_hits.all()].astype(int)
            for test, count in rejections.items():
                rate = count / replications
                rows.append({"row_type": "size", "outcome": outcome, "treatment": variant, "spec": spec, "dgp": dgp["id"],
                             "innovation": innovation["id"], "innovation_df": innovation.get("df", ""),
                             "ar1": phi, "test": test, "hac_lags": int(config["hac_lags"]),
                             "replications": replications, "bootstrap_replicates": inner,
                             "nominal_level": alpha, "rejections": int(count), "rejection_rate": rate,
                             "monte_carlo_se": math.sqrt(rate * (1 - rate) / replications),
                             "max_acceptable_size": ceiling,
                             "status": "calibrated" if rate <= ceiling else "not_calibrated",
                             "size_direction": size_direction(rate, config)})
            for (interval_type, interval_id), count in zip(targets, covered, strict=True):
                rate = float(count / replications)
                mc_se = math.sqrt(rate * (1 - rate) / replications)
                rows.append({"row_type": "coverage", "test": "coverage", "outcome": outcome,
                             "treatment": variant, "spec": spec, "dgp": dgp["id"],
                             "innovation": innovation["id"], "innovation_df": innovation.get("df", ""),
                             "ar1": phi, "replications": replications, "bootstrap_replicates": inner,
                             "interval_type": interval_type, "interval_id": interval_id,
                             "nominal_level": level, "covered": int(count), "coverage": rate,
                             "monte_carlo_se": mc_se,
                             "minimum_acceptable_coverage": config["coverage"]["minimum_acceptable"],
                             "mc_se_multiplier": config["coverage"]["mc_se_multiplier"],
                             "status": coverage_status(rate, mc_se, config["coverage"])})
    return rows


def coverage_targets(design: Design) -> list[tuple[str, str]]:
    eras = [e for e, _ in design.eras]
    slope, contrast = design.raw["calibration"]["coverage"]["intervals"]
    return ([(slope, e) for e in eras] +
            [(contrast, f"{eras[i + 1]}_minus_{eras[i]}") for i in range(2)] +
            [(contrast, "joint_two_contrasts")])


def coverage_status(rate: float, mc_se: float, rule: Mapping[str, Any]) -> str:
    if not (0 <= rate <= 1 and math.isfinite(mc_se) and mc_se >= 0):
        raise ValueError("Malformed coverage evidence")
    failed = rate + float(rule["mc_se_multiplier"]) * mc_se < float(rule["minimum_acceptable"])
    return "coverage_failed_specified_dgp" if failed else "coverage_checked_specified_dgp"


def interval_evidence(calibration: Sequence[Mapping[str, Any]], design: Design,
                      outcome: str, variant: str, spec: str, interval_id: str) -> dict[str, Any]:
    """Worst specified-DGP status, with every cell retained for inspection."""
    rows = [r for r in calibration if r["row_type"] == "coverage" and
            (r["outcome"], r["treatment"], r["spec"], r["interval_id"]) ==
            (outcome, variant, spec, interval_id)]
    evidence = [{k: r[k] for k in ("dgp", "innovation", "coverage", "monte_carlo_se")} |
                {"status": coverage_status(r["coverage"], r["monte_carlo_se"], design.raw["calibration"]["coverage"])}
                for r in rows]
    status = "coverage_unverified"
    worst = min(rows, key=lambda r: r["coverage"]) if rows else None
    if rows:
        expected = {(g["id"], v["id"]) for g in design.raw["calibration"]["dgps"]
                    for v in design.raw["calibration"]["innovations"]}
        if {(r["dgp"], r["innovation"]) for r in rows} != expected or len(rows) != len(expected):
            raise ValueError("Incomplete or duplicate interval coverage evidence")
        labels = [r["status"] for r in evidence]
        status = ("coverage_failed_specified_dgp" if "coverage_failed_specified_dgp" in labels
                  else "coverage_checked_specified_dgp")
    return {"bootstrap_uncertainty_status": status,
            "coverage_minimum": worst["coverage"] if worst else "",
            "coverage_minimum_mc_se": worst["monte_carlo_se"] if worst else "",
            "coverage_evidence": json.dumps(evidence, separators=(",", ":")),
            "calibration_scope": "specified_equal_slope_coverage_dgps_only" if rows else "interval_not_tested"}


def size_direction(rate: float, config: Mapping[str, Any]) -> str:
    """Descriptive direction only; pass or fail is the declared ceiling alone."""
    if rate > float(config["max_acceptable_size"]):
        return "over_rejects"
    return "conservative" if rate < float(config["conservative_below"]) else "near_nominal"


def era_inference_status(calibration: Sequence[Mapping[str, Any]], outcome: str, spec: str,
                         config: Mapping[str, Any]) -> dict[str, Any]:
    status = {}
    for test in ("hac_wald", "bootstrap_fixed_scale_maximum"):
        rows = [r for r in calibration if r["outcome"] == outcome and r["spec"] == spec and r["test"] == test]
        if not rows:
            status[test] = {"size": None, "status": "calibration_not_run", "direction": None}
            continue
        worst = max(r["rejection_rate"] for r in rows)
        status[test] = {"size": worst, "direction": size_direction(worst, config),
                        "status": "calibrated" if all(r["status"] == "calibrated" for r in rows) else "not_calibrated"}
    flags = [status[t]["status"] == "calibrated" for t in ("hac_wald", "bootstrap_fixed_scale_maximum")]
    label = ("hac_and_bootstrap_calibrated" if all(flags) else "bootstrap_calibrated_only" if flags[1]
             else "hac_calibrated_only" if flags[0]
             else "calibration_not_run_descriptive_only" if status["hac_wald"]["status"] == "calibration_not_run"
             else "not_calibrated_descriptive_only")
    return {"tests": status, "era_inference": label}


# --------------------------------------------------------------------------- tables


def _influence(panel: Panel, design: Design, cell: Cell, fit: Mapping[str, Any]) -> list[dict[str, Any]]:
    lags = {k: v[cell.positions] for k, v in cell_lags(panel, cell.outcome, cell.variant).items()}
    z = control_matrix(panel.qoy[cell.positions], design.specs[cell.spec], lags)
    m = np.column_stack([fit["x"], fit["y"]])
    n = len(m)
    quarters = [panel.quarters[p] for p in cell.positions]
    keep = np.asarray([np.delete(np.arange(n), t) for t in range(n)])
    resid = residualize(z[keep], m[keep], require=(0,))
    loo = np.einsum("bn,bn->b", resid[..., 0], resid[..., 1]) / np.einsum("bn,bn->b", resid[..., 0], resid[..., 0])
    xt = fit["resid"][:, 0]
    centered = fit["x"] - fit["x"].mean()
    beta = float(fit["beta"])
    base = {"outcome": cell.outcome, "treatment": cell.variant, "sample": cell.sample, "spec": cell.spec, "beta": beta}
    rows = [{**base, "kind": "leave_one_out", "deleted": q, "n_deleted": 1, "n": n - 1,
             "centered_variation_share": float(centered[t] ** 2 / fit["total_centered_ss"]),
             "partial_variation_share": float(xt[t] ** 2 / fit["denominator"]),
             "bootstrap_replicates_present": int(design.raw["bootstrap"]["replicates"]) if cell.spec in design.raw["bootstrap"]["specifications"] else "",
             "bootstrap_expected_multiplicity": 1 if cell.spec in design.raw["bootstrap"]["specifications"] else "",
             "bootstrap_omission_rate": 0.0 if cell.spec in design.raw["bootstrap"]["specifications"] else "",
             "beta_after_deletion": float(loo[t]), "change": float(loo[t] - beta)} for t, q in enumerate(quarters)]
    for start, end in design.raw["influence"]["block_deletions"]:
        block = set(quarter_range(start, end))
        mask = np.asarray([q not in block for q in quarters])
        if mask.all():
            continue
        r = residualize(z[mask], m[mask], require=(0,))
        value = float(r[:, 0] @ r[:, 1] / (r[:, 0] @ r[:, 0]))
        rows.append({**base, "kind": "block_deletion", "deleted": f"{start}-{end}",
                     "n_deleted": int((~mask).sum()), "n": int(mask.sum()),
                     "centered_variation_share": float((centered[~mask] ** 2).sum() / fit["total_centered_ss"]),
                     "partial_variation_share": float((xt[~mask] ** 2).sum() / fit["denominator"]),
                     "beta_after_deletion": value, "change": value - beta})
    return rows


def _summary(values: np.ndarray | None, level: float, batches: int = 20) -> dict[str, Any]:
    if values is None:
        return {"bootstrap_se": "", "bootstrap_lower": "", "bootstrap_upper": "",
                "bootstrap_lower_mc_se": "", "bootstrap_upper_mc_se": ""}
    tail = (1 - level) / 2
    low, high = np.quantile(values, [tail, 1 - tail], axis=0)
    mc = _endpoint_mc_se(values, lambda v: np.quantile(v, [tail, 1 - tail], axis=0), batches)
    return {"bootstrap_se": values.std(axis=0, ddof=1), "bootstrap_lower": low, "bootstrap_upper": high,
            "bootstrap_lower_mc_se": mc[0], "bootstrap_upper_mc_se": mc[1]}


def _pick(summary: Mapping[str, Any], index: Any) -> dict[str, Any]:
    return {k: (float(v[index]) if isinstance(v, np.ndarray) else v) for k, v in summary.items()}


def _cell_tables(panel: Panel, design: Design, cell: Cell, fit: Mapping[str, Any],
                 replicates: Mapping[str, np.ndarray] | None) -> dict[str, list[dict[str, Any]]]:
    """Estimate, contribution, grouped and leg-by-factor rows for one cell."""
    level, lag = float(design.raw["bootstrap"]["level"]), design.hac_lags[0]
    summary_for = lambda values: _summary(values, level, int(design.raw["bootstrap"]["monte_carlo_batches"]))
    components, groups = design.components(cell.outcome), design.groups_for(cell.outcome)
    leg_ids = [leg for leg, _ in design.legs]
    beta, d = float(fit["beta"]), float(fit["denominator"])
    key = {"outcome": cell.outcome, "treatment": cell.variant, "sample": cell.sample, "spec": cell.spec}
    boot = (lambda name: replicates[name]) if replicates else (lambda name: None)
    se = fit["hac"][lag]["se"]
    estimate = {**key, "treatment_role": design.raw["treatments"][cell.variant]["role"],
                "spec_role": design.raw["specifications"][cell.spec]["role"], "n": len(cell.positions),
                "first_quarter": panel.quarters[cell.positions[0]], "last_quarter": panel.quarters[cell.positions[-1]],
                "dropped_quarters": ";".join(cell.dropped), "parameters": fit["k"],
                "beta": beta, "beta_minus_one": beta - 1}
    for lags, h in fit["hac"].items():
        estimate |= {f"se_hac{lags}": h["se"], f"p_beta_zero_hac{lags}": h["p_beta_zero"],
                     f"p_beta_one_hac{lags}": h["p_beta_one"]}
    estimate |= {f"ci95_hac{lag}_lower": beta - Z95 * se, f"ci95_hac{lag}_upper": beta + Z95 * se,
                 "finite_sample_scale": fit["hac"][lag]["finite_sample_scale"], "common_denominator": d,
                 "total_centered_ss": fit["total_centered_ss"], "partial_variation_share": fit["partial_variation_share"],
                 **_pick(summary_for(boot("beta")), ()),
                 "bootstrap_status": "dependent_wild_bootstrap_coverage_unverified" if replicates else "not_bootstrapped_declared_sensitivity"}
    total = float(fit["contributions"].sum())
    summary = summary_for(boot("contributions"))
    contributions = [{**key, "group": next(g.id for g in groups if name in g.components), "component": name,
                      "label": panel.labels[name], "clock": panel.clocks[name],
                      "numerator": float(fit["contributions"][k]) * d, "common_denominator": d,
                      "contribution": float(fit["contributions"][k]), **_pick(summary, k),
                      "contribution_sum": total, "beta_minus_one": beta - 1, "adding_up_gap": total - (beta - 1)}
                     for k, name in enumerate(components)]
    members = [[components.index(c) for c in g.components] for g in groups]

    def by_group(values: np.ndarray, axis: int) -> np.ndarray:
        return np.stack([np.take(values, m, axis=axis).sum(axis=axis) for m in members], axis=axis)

    grouped_boot = by_group(replicates["contributions"], -1) if replicates else None
    grouped_summary = summary_for(grouped_boot)
    leg_summary = summary_for(boot("leg_rows"))
    grouped = [{**key, "row_type": "group", "group": g.id, "label": g.label,
                "contribution": float(by_group(fit["contributions"], -1)[i]),
                **_pick(grouped_summary, i)} for i, g in enumerate(groups)]
    grouped += [{**key, "row_type": "leg", "group": leg, "label": design.raw["legs"][j]["label"],
                 "contribution": float(fit["leg_rows"][j]), **_pick(leg_summary, j)}
                for j, leg in enumerate(leg_ids)]
    grouped.append({**key, "row_type": "total", "group": "beta_minus_one", "label": "beta - 1", "contribution": beta - 1,
                    **_pick(summary_for(replicates["beta"] - 1 if replicates else None), ())})
    group_ids = [g.id for g in groups]
    balances = []
    for balance in design.balances:
        values = fit["balances"][balance["id"]]
        g, coefficient = group_ids.index(balance["group"]), values["coefficient"]
        se = values["hac"][lag]["se"]
        row = {**key, "balance": balance["id"], "label": balance["label"], "side": balance["side"],
               "group": balance["group"], "n": len(cell.positions), "first_quarter": estimate["first_quarter"],
               "last_quarter": estimate["last_quarter"], "reserve_beta": beta,
               "group_contribution": float(by_group(fit["contributions"], -1)[g]), "coefficient": coefficient}
        for lags, h in values["hac"].items():
            row |= {f"se_hac{lags}": h["se"], f"p_zero_hac{lags}": h["p_zero"]}
        row |= {f"ci95_hac{lag}_lower": coefficient - Z95 * se, f"ci95_hac{lag}_upper": coefficient + Z95 * se,
                # Bootstrap draws are the signed group-contribution draws of the same replicates.
                **_pick(summary_for(values["sign"] * grouped_boot[:, g] if replicates else None), ()),
                "hac_uncertainty_status": "coverage_unverified",
                "bootstrap_uncertainty_status": "coverage_unverified" if replicates else "not_bootstrapped_declared_sensitivity",
                "calibration_scope": "interval_not_tested",
                "uncertainty_definition": design.raw["uncertainty_definitions"]["balance_coefficient_interval"],
                "definition": balance["definition"]}
        balances.append(row)
    status = {leg: "observed" if np.any(panel.column(cell.variant, column)[cell.positions]) else "structural_zero_in_sample"
              for leg, column in design.legs}
    matrix = []
    for level_name, columns, values in (("component", components, fit["matrix"]),
                                        ("group", [g.id for g in groups], by_group(fit["matrix"], 1))):
        for j, leg in enumerate(leg_ids):
            matrix += [{**key, "level": level_name, "leg": leg, "leg_status": status[leg], "column": column,
                        "value": float(values[j, c])} for c, column in enumerate(columns)]
            matrix.append({**key, "level": level_name, "leg": leg, "leg_status": status[leg], "column": "row_sum",
                           "value": float(values[j].sum())})
        matrix += [{**key, "level": level_name, "leg": "column_sum", "leg_status": "", "column": column,
                    "value": float(values[:, c].sum())} for c, column in enumerate(columns)]
        matrix.append({**key, "level": level_name, "leg": "column_sum", "leg_status": "", "column": "row_sum",
                       "value": float(values.sum())})
    return {"estimates": [estimate], "contributions": contributions, "grouped_contributions": grouped,
            "leg_factor_matrix": matrix, "influence": _influence(panel, design, cell, fit),
            "balance_coefficients": balances}


def _contrast_tables(panel: Panel, design: Design, cells: Mapping[tuple[str, str, str, str], Cell],
                     fits: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
                     boot: Mapping[tuple[str, str, str, str], Mapping[str, np.ndarray]],
                     calibration: Sequence[Mapping[str, Any]], gaps: dict[str, float]) -> list[dict[str, Any]]:
    """Two era-slope contrasts and their joint tests, stacked on the era cells' own rows."""
    level, primary_lag = float(design.raw["bootstrap"]["level"]), design.hac_lags[0]
    era_ids = [e for e, _ in design.eras]
    contrast_ids = [f"{era_ids[1]}_minus_{era_ids[0]}", f"{era_ids[2]}_minus_{era_ids[1]}"]
    rows = []
    for outcome in design.outcomes:
        for variant in design.variants:
            for spec in design.specs:
                era_cells = [cells[(outcome, variant, e, spec)] for e in era_ids]
                positions = np.concatenate([c.positions for c in era_cells])
                era = np.concatenate([np.full(len(c.positions), e) for e, c in enumerate(era_cells)])
                lags = {k: v[positions] for k, v in cell_lags(panel, outcome, variant).items()}
                x, y = panel.column(variant, "x")[positions], panel.common[outcome][positions]
                walds = {lag: era_contrast_wald(x, y, era, panel.qoy[positions], panel.ordinals[positions],
                                                design.specs[spec], lags, lag) for lag in design.hac_lags}
                separate = np.asarray([float(fits[(outcome, variant, e, spec)]["beta"]) for e in era_ids])
                stacked_gap = float(np.abs(walds[primary_lag]["slopes"] - separate).max())
                if stacked_gap > CROSS_CHECK_TOLERANCE:
                    raise ValueError("Stacked era slopes differ from separate era projections")
                gaps["stacked_era_slope_gap"] = max(gaps.get("stacked_era_slope_gap", 0.0), stacked_gap)
                delta = CONTRAST @ separate
                key = {"outcome": outcome, "treatment": variant, "spec": spec, "n": len(positions)}
                bootstrap = None
                if (outcome, variant, era_ids[0], spec) in boot:
                    star = np.column_stack([boot[(outcome, variant, e, spec)]["beta"] for e in era_ids]) @ CONTRAST.T
                    bootstrap = fixed_scale_maximum_inference(delta, star, level, boot[(outcome, variant, "joint_null", spec)]["delta"],
                                                int(design.raw["bootstrap"]["monte_carlo_batches"]))
                calibrated = era_inference_status(
                    calibration if variant == design.raw["calibration"]["treatment"] else [], outcome, spec,
                    design.raw["calibration"])
                for c, contrast in enumerate(contrast_ids):
                    row = {**key, "row_type": "contrast", "contrast": contrast, "estimate": float(delta[c])}
                    row |= {f"se_hac{lag}": float(walds[lag]["delta_se"][c]) for lag in design.hac_lags}
                    if bootstrap:
                        row |= {"bootstrap_se": float(bootstrap["se"][c]), "simultaneous_lower": float(bootstrap["lower"][c]),
                                "simultaneous_upper": float(bootstrap["upper"][c]),
                                "simultaneous_critical_value": bootstrap["critical_value"],
                                "simultaneous_lower_mc_se": float(bootstrap["lower_mc_se"][c]),
                                "simultaneous_upper_mc_se": float(bootstrap["upper_mc_se"][c]),
                                "statistic": design.raw["bootstrap"]["statistic"]}
                    row |= interval_evidence(calibration, design, outcome, variant, spec, contrast)
                    row["uncertainty_status"] = row["bootstrap_uncertainty_status"]
                    row["hac_uncertainty_status"] = "coverage_unverified"
                    rows.append(row)
                joint = {**key, "row_type": "joint", "contrast": "joint_two_contrasts"}
                for lag in design.hac_lags:
                    joint |= {f"wald_hac{lag}": walds[lag]["wald"], f"p_hac{lag}": walds[lag]["p_value"]}
                joint["finite_sample_scale"] = walds[primary_lag]["finite_sample_scale"]
                if bootstrap:
                    joint |= {"bootstrap_p": bootstrap["p_value"], "bootstrap_maximum_observed": bootstrap["maximum_observed"],
                              "bootstrap_p_mc_se": bootstrap["p_mc_se"], "bootstrap_replicates": bootstrap["replicates"],
                              "threshold_status": "near_threshold" if abs(bootstrap["p_value"] - 0.05) <=
                              float(design.raw["bootstrap"]["near_threshold_mc_se_multiplier"]) * bootstrap["p_mc_se"]
                              else "away_from_threshold"}
                tests = calibrated["tests"]
                joint |= {"size_hac_wald": tests["hac_wald"]["size"], "status_hac_wald": tests["hac_wald"]["status"],
                          "direction_hac_wald": tests["hac_wald"]["direction"],
                          "size_bootstrap": tests["bootstrap_fixed_scale_maximum"]["size"],
                          "status_bootstrap": tests["bootstrap_fixed_scale_maximum"]["status"],
                          "direction_bootstrap": tests["bootstrap_fixed_scale_maximum"]["direction"],
                          "era_inference": calibrated["era_inference"]}
                joint["statistic"] = design.raw["bootstrap"]["statistic"]
                family = interval_evidence(calibration, design, outcome, variant, spec, "joint_two_contrasts")
                joint |= {f"band_family_{k}": v for k, v in family.items()}
                rows.append(joint)
    return rows


def _transition_table(panel: Panel, design: Design) -> list[dict[str, Any]]:
    """Transition quarters stay in the pooled sample and are displayed in dollars."""
    rows = []
    for quarter in design.transitions:
        p = panel.positions([quarter])[0]
        for variant in design.variants:
            for outcome in design.outcomes:
                x, y = panel.column(variant, "x")[p], panel.common[outcome][p]
                values = {"x": x, **{leg: panel.column(variant, column)[p] for leg, column in design.legs},
                          "y": y, "remainder": y - x}
                values |= {g.id: sum(panel.column(variant, c)[p] for c in g.components) for g in design.groups_for(outcome)}
                rows += [{"quarter": quarter, "outcome": outcome, "treatment": variant, "item": item,
                          "usd_million": float(value)} for item, value in values.items()]
    return rows


def produce(panel: Panel, design: Design) -> dict[str, Any]:
    """All declared estimates, contributions, matrices, influence, contrasts and calibration."""
    boot, representation = run_bootstrap(panel, design)
    calibration = calibrate(panel, design)
    tables: dict[str, list[dict[str, Any]]] = {k: [] for k in (
        "estimates", "contributions", "grouped_contributions", "leg_factor_matrix", "influence", "balance_coefficients")}
    cells, fits, gaps = {}, {}, {}
    for outcome in design.outcomes:
        for variant in design.variants:
            for spec in design.specs:
                for sample, quarters in design.samples:
                    key = (outcome, variant, sample, spec)
                    cells[key] = make_cell(panel, design, outcome, variant, sample, spec, quarters)
                    fits[key] = fit = fit_cell(panel, design, cells[key])
                    for name, value in [*fit["gaps"].items(), *(("fwl_hac_beta_gap", h["fwl_gap"]) for h in fit["hac"].values()),
                                        *(("balance_hac_coefficient_gap", h["gap"]) for b in fit["balances"].values()
                                          for h in b["hac"].values())]:
                        gaps[name] = max(gaps.get(name, 0.0), value)
                    for name, rows in _cell_tables(panel, design, cells[key], fit, boot.get(key)).items():
                        tables[name].extend(rows)
    tables["era_contrasts"] = _contrast_tables(panel, design, cells, fits, boot, calibration, gaps)
    tables["transition_quarters"] = _transition_table(panel, design)
    tables["calibration"] = calibration
    for row in tables["estimates"]:
        row |= interval_evidence(calibration, design, row["outcome"], row["treatment"], row["spec"], row["sample"])
        row["bootstrap_status"] = row["bootstrap_uncertainty_status"]
        row["hac_uncertainty_status"] = "coverage_unverified"
    for name in ("contributions", "grouped_contributions"):
        for row in tables[name]:
            row["bootstrap_uncertainty_status"] = "coverage_unverified"
    if not design.balances:
        del tables["balance_coefficients"]
    tables |= _thin_tables(design, tables)
    return {"tables": tables, "gaps": gaps, "bootstrap_representation": representation}


def _thin_tables(design: Design, tables: Mapping[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """One era table and one grouped-contribution table per outcome, primary treatment."""
    variant, lag = design.primary_variant, design.hac_lags[0]
    out = {}
    for outcome in design.outcomes:
        era_rows = []
        for spec in design.raw["bootstrap"]["specifications"]:
            for r in tables["estimates"]:
                if (r["outcome"], r["treatment"], r["spec"]) == (outcome, variant, spec):
                    era_rows.append({"row_type": "slope", "spec": spec, "sample": r["sample"],
                                     "first_quarter": r["first_quarter"], "last_quarter": r["last_quarter"], "n": r["n"],
                                     "estimate": r["beta"], f"se_hac{lag}": r[f"se_hac{lag}"],
                                     "hac_lower": r[f"ci95_hac{lag}_lower"], "hac_upper": r[f"ci95_hac{lag}_upper"],
                                     "bootstrap_lower": r["bootstrap_lower"], "bootstrap_upper": r["bootstrap_upper"],
                                     "bootstrap_interval": "pointwise 95% unrestricted dependent-wild percentile",
                                     "partial_variation_share": r["partial_variation_share"],
                                     "bootstrap_lower_mc_se": r["bootstrap_lower_mc_se"],
                                     "bootstrap_upper_mc_se": r["bootstrap_upper_mc_se"]})
            for r in tables["era_contrasts"]:
                if (r["outcome"], r["treatment"], r["spec"]) != (outcome, variant, spec):
                    continue
                if r["row_type"] == "contrast":
                    se = r[f"se_hac{lag}"]
                    era_rows.append({"row_type": "contrast", "spec": spec, "sample": r["contrast"], "n": r["n"],
                                     "estimate": r["estimate"], f"se_hac{lag}": se,
                                     "hac_lower": r["estimate"] - Z95 * se, "hac_upper": r["estimate"] + Z95 * se,
                                     "bootstrap_lower": r["simultaneous_lower"], "bootstrap_upper": r["simultaneous_upper"],
                                     "bootstrap_interval": "simultaneous 95% unrestricted dependent-wild fixed-scale maximum band",
                                     "bootstrap_lower_mc_se": r["simultaneous_lower_mc_se"],
                                     "bootstrap_upper_mc_se": r["simultaneous_upper_mc_se"]})
                else:
                    era_rows.append({"row_type": "joint_test", "spec": spec, "sample": r["contrast"], "n": r["n"],
                                     f"wald_hac{lag}": r[f"wald_hac{lag}"], f"p_hac{lag}": r[f"p_hac{lag}"],
                                     "bootstrap_p": r["bootstrap_p"], "bootstrap_p_mc_se": r["bootstrap_p_mc_se"],
                                     "bootstrap_replicates": r["bootstrap_replicates"], "threshold_status": r["threshold_status"],
                                     "status_hac_wald": r["status_hac_wald"], "status_bootstrap": r["status_bootstrap"],
                                     "size_hac_wald": r["size_hac_wald"],
                                     "size_bootstrap": r["size_bootstrap"], "direction_bootstrap": r["direction_bootstrap"],
                                     "era_inference": r["era_inference"], "statistic": r["statistic"],
                                     **{k: v for k, v in r.items() if k.startswith("band_family_")}})
        for quarter in design.transitions:
            values = {r["item"]: r["usd_million"] for r in tables["transition_quarters"]
                      if (r["quarter"], r["outcome"], r["treatment"]) == (quarter, outcome, variant)}
            era_rows.append({"row_type": "transition_quarter", "sample": quarter, "n": 1,
                             "x_usd_million": values["x"], "y_usd_million": values["y"],
                             "remainder_usd_million": values["remainder"]})
        definitions = design.raw["uncertainty_definitions"]
        for row in era_rows:
            row["era_selection"] = definitions["era_selection"]
            if row["row_type"] in ("slope", "contrast"):
                row["hac_uncertainty_status"] = "coverage_unverified"
                row["hac_definition"] = definitions["hac_interval"]
                row |= interval_evidence(tables["calibration"], design, outcome, variant, row["spec"], row["sample"])
                row["bootstrap_definition"] = definitions["bootstrap_interval"]
            elif row["row_type"] == "joint_test":
                row["uncertainty_status"] = row["era_inference"]
                row["uncertainty_definition"] = definitions["joint_test"]
                row["calibration_scope"] = "specified_joint_test_null_size_checks_only"
            else:
                row["uncertainty_status"] = "observed_transition_no_inference"
        out[f"thin_era_table_{outcome}"] = era_rows
        out[f"thin_grouped_contributions_{outcome}"] = [
            {k: r[k] for k in ("sample", "row_type", "group", "label", "contribution", "bootstrap_se",
                               "bootstrap_lower", "bootstrap_upper", "bootstrap_lower_mc_se", "bootstrap_upper_mc_se")}
            | {"uncertainty_status": "coverage_unverified", "uncertainty_definition": definitions["contribution_interval"]}
            for r in tables["grouped_contributions"]
            if (r["outcome"], r["treatment"], r["spec"]) == (outcome, variant, design.primary_spec)]
    if design.balances:
        # Each declared balance beside the reserve coefficient, primary treatment, bootstrapped specs.
        out["thin_balance_coefficients"] = [
            {k: r[k] for k in ("outcome", "spec", "sample", "first_quarter", "last_quarter", "n", "reserve_beta",
                               "balance", "label", "coefficient", "group_contribution", f"se_hac{lag}",
                               f"ci95_hac{lag}_lower", f"ci95_hac{lag}_upper", "bootstrap_se", "bootstrap_lower",
                               "bootstrap_upper", "bootstrap_lower_mc_se", "bootstrap_upper_mc_se",
                               "hac_uncertainty_status", "bootstrap_uncertainty_status", "calibration_scope",
                               "uncertainty_definition")}
            | {"era_selection": definitions["era_selection"]}
            for r in tables["balance_coefficients"]
            if r["treatment"] == variant and r["spec"] in design.raw["bootstrap"]["specifications"]]
    return out


def write_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing an empty table: {path.name}")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k, "")) for k in fields})
