"""Kernel-weighted local same-quarter projections with fully local refits.

At each target quarter t the admitted specification of a panel (outcome on an
intercept, the measured treatment and the admitted controls) is refitted by
weighted least squares with Epanechnikov weights in calendar-quarter distance.
Every coefficient is local, as in the rolling windows, so a uniform trailing
kernel reproduces the admitted rolling rows and a uniform full-sample kernel the
admitted full-sample estimate. The slope's target is a kernel-weighted
projection coefficient, not an instantaneous structural coefficient, landing,
retention or a causal response.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ea_tdc import reserve_projection as rp
from ea_tdc.open01 import _quarter_ordinal

SCHEMA_VERSION = "ea_tdc_local_projection_v1"
Z95 = rp.Z95
MEAN_CLASSES = ("pooled", "slope_drift", "slope_steps", "nuisance_drift")
SCALES = ("absolute_residual", "homoskedastic")
INNOVATIONS = ("gaussian", "student_t4")
COVERAGE = ("coverage_failed_specified_dgp", "coverage_inconclusive_specified_dgp", "coverage_qualified_specified_dgp")


def validate(config: Mapping[str, Any]) -> None:
    """Reject declarations the implementation would otherwise ignore."""
    estimator, calibration = config["estimator"], config["calibration"]
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Local projection config schema differs")
    if estimator["kernel"] != "epanechnikov" or estimator["id"] != "fully_local_kernel_weighted_least_squares":
        raise ValueError("Only the fully local Epanechnikov estimator is implemented")
    widths = [estimator["primary_half_width"], *estimator["sensitivity_half_widths"]]
    if any(not isinstance(h, int) or h < 2 for h in widths) or len(set(widths)) != len(widths):
        raise ValueError("Half-widths must be distinct integers of at least two quarters")
    numerical = estimator["numerical"]
    if not (0 < numerical["min_reciprocal_condition"] < 1 and 0 < numerical["slope_crosscheck_relative"] < 1):
        raise ValueError("Numerical resolution thresholds must lie in (0, 1)")
    if calibration["bandwidths"] != "primary":
        raise ValueError("Calibration is implemented for the primary bandwidth only")
    if [m["id"] for m in calibration["means"]] != list(MEAN_CLASSES):
        raise ValueError("Calibration mean classes differ from the implemented list")
    if [m["id"] for m in calibration["scales"]] != list(SCALES):
        raise ValueError("Calibration scales differ from the implemented list")
    if [i["id"] for i in calibration["innovations"]] != list(INNOVATIONS) or calibration["innovations"][1]["df"] <= 2:
        raise ValueError("Calibration innovations differ from the implemented list")
    for ar in calibration["ar1"]:
        if ar["id"] != "empirical_ar1" and not (isinstance(ar["value"], (int, float)) and 0 <= ar["value"] < 1):
            raise ValueError(f"Unsupported AR(1) declaration {ar}")
    if not (0 < calibration["minimum_acceptable"] < calibration["nominal_level"] < 1 and calibration["mc_se_multiplier"] > 0):
        raise ValueError("Coverage thresholds are inconsistent")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned(root: Path, record: Mapping[str, Any]) -> Path:
    path = Path(root) / record["path"]
    if not path.is_file():
        raise ValueError(f"Pinned input is missing: {record['path']}")
    if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
        raise ValueError(f"Pinned input differs from its declared hash: {record['path']}")
    return path


def epanechnikov(distance: np.ndarray, half_width: float) -> np.ndarray:
    return np.clip(1.0 - (np.asarray(distance, dtype=float) / half_width) ** 2, 0.0, None)


def kernel_weights(ordinals: np.ndarray, target: int, half_width: float) -> np.ndarray:
    return epanechnikov(np.abs(ordinals - target), half_width)


def trailing_weights(ordinals: np.ndarray, end: int, window: int) -> np.ndarray:
    return ((ordinals <= end) & (ordinals > end - window)).astype(float)


@dataclass(frozen=True)
class Panel:
    name: str
    quarters: tuple[str, ...]
    ordinals: np.ndarray
    y: np.ndarray
    x: np.ndarray
    controls: np.ndarray
    control_ids: tuple[str, ...]
    hac_lags: int
    treatment_id: str = ""
    outcome_id: str = ""
    rows: tuple[dict[str, str], ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        n = len(self.quarters)
        if (self.ordinals.shape != (n,) or self.y.shape != (n,) or self.x.shape != (n,)
                or self.controls.shape != (n, len(self.control_ids))):
            raise ValueError(f"{self.name}: misaligned panel arrays")
        if np.any(np.diff(self.ordinals) <= 0) or len(set(self.quarters)) != n:
            raise ValueError(f"{self.name}: quarters must be unique and ordered")
        if not (np.isfinite(self.y).all() and np.isfinite(self.x).all() and np.isfinite(self.controls).all()):
            raise ValueError(f"{self.name}: nonfinite panel values")


def admitted_controls(panel: Panel, mask: np.ndarray) -> tuple[list[int], list[str]]:
    """Controls admitted on the support; frozen greedy rank rule when raw rows are carried."""
    if not panel.rows:
        return list(range(len(panel.control_ids))), []
    from run_submission_appendix_diagnostics import _admit_controls_rank_aware

    rows = [panel.rows[i] for i in np.flatnonzero(mask)]
    selected, rejected = _admit_controls_rank_aware(
        rows, treatment_id=panel.treatment_id, outcome_id=panel.outcome_id, horizon=0,
        candidate_controls=list(panel.control_ids), covariance_lags=panel.hac_lags)
    return [panel.control_ids.index(c) for c in selected], list(rejected)


def slope_variance(u: np.ndarray, ordinals: np.ndarray, lags: int) -> np.ndarray:
    """Bartlett calendar-distance long-run variance of slope scores u (rows = quarters, columns = draws)."""
    u = np.asarray(u, dtype=float)
    if u.ndim == 1:
        u = u[:, None]
    offsets = np.asarray(ordinals, dtype=int) - int(ordinals[0])
    grid = np.zeros((int(offsets[-1]) + 1, u.shape[1]))
    grid[offsets] = u
    total = (grid * grid).sum(axis=0)
    for lag in range(1, lags + 1):
        total += 2.0 * (1.0 - lag / (lags + 1)) * (grid[lag:] * grid[:-lag]).sum(axis=0)
    return total


@dataclass(frozen=True)
class Operator:
    """Fixed-design local fit at one target: everything except the outcome."""
    mask: np.ndarray
    weights: np.ndarray
    design: np.ndarray
    slope_row: np.ndarray
    residual_maker: np.ndarray
    scale: float
    n_eff: float
    k: int
    rank_ok: bool
    resolved: bool
    reciprocal_condition: float
    partial_variation: float
    residual_x: np.ndarray
    leverage: np.ndarray
    admitted: tuple[str, ...]
    rejected: tuple[str, ...]


def operator(panel: Panel, weights: np.ndarray, numerical: Mapping[str, float]) -> Operator:
    """Column-scaled weighted QR fit; the slope row is cross-checked against an SVD partial-out slope."""
    weights = np.asarray(weights, dtype=float)
    if weights.shape != panel.y.shape or np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("Weights must be finite, nonnegative and aligned with the panel")
    mask = weights > 0
    admitted, rejected = admitted_controls(panel, mask)
    design = np.column_stack([np.ones(int(mask.sum())), panel.x[mask], panel.controls[mask][:, admitted]])
    w = weights[mask]
    n, k = design.shape
    n_eff = float(w.sum() ** 2 / (w ** 2).sum()) if n else 0.0
    root_w = np.sqrt(w)
    weighted = design * root_w[:, None]
    norms = np.linalg.norm(weighted, axis=0)
    names = tuple(panel.control_ids[i] for i in admitted)
    nan = np.full(n, np.nan)
    unresolved = Operator(mask, w, design, nan, np.full((n, n), np.nan), math.nan, n_eff, k, False, False, math.nan,
                          math.nan, nan, nan, names, tuple(rejected))
    if n <= k or not np.all(norms > 0):
        return unresolved
    scaled = weighted / norms
    singular = np.linalg.svd(scaled, compute_uv=False)
    reciprocal = float(singular[-1] / singular[0])
    rank_ok = int(np.linalg.matrix_rank(scaled)) == k
    if not rank_ok:
        return unresolved
    q, r = np.linalg.qr(scaled)
    projector = np.linalg.solve(r, q.T) * root_w[None, :] / norms[:, None]
    nuisance = np.delete(weighted, 1, axis=1)
    coef = np.linalg.lstsq(nuisance, weighted[:, 1], rcond=None)[0]
    residual_x = design[:, 1] - np.delete(design, 1, axis=1) @ coef
    partial = float(w @ residual_x ** 2)
    crosscheck = w * residual_x / partial if partial > 0 else nan
    gap = float(np.max(np.abs(crosscheck - projector[1])) / np.max(np.abs(projector[1])))
    resolved = reciprocal >= numerical["min_reciprocal_condition"] and gap <= numerical["slope_crosscheck_relative"]
    scale = n_eff / (n_eff - k) if n_eff > k else math.nan
    return Operator(mask, w, design, projector[1], np.eye(n) - design @ projector, scale, n_eff, k, True, resolved,
                    reciprocal, partial, residual_x, (q ** 2).sum(axis=1), names, tuple(rejected))


def apply(op: Operator, ordinals: np.ndarray, y: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Slope estimates and standard errors for outcome columns y (full panel rows)."""
    ys = np.asarray(y, dtype=float)[op.mask]
    beta = op.slope_row @ ys
    residuals = op.residual_maker @ ys
    u = op.slope_row[:, None] * (residuals if residuals.ndim == 2 else residuals[:, None])
    se = np.sqrt(op.scale * slope_variance(u, ordinals[op.mask], lags))
    return beta, se if residuals.ndim == 2 else se[0]


def influence(op: Operator, quarters: Sequence[str]) -> dict[str, Any]:
    share = op.weights * op.residual_x ** 2 / op.partial_variation
    support = [q for q, keep in zip(quarters, op.mask) if keep]
    top, peak = int(np.argmax(share)), int(np.argmax(op.leverage))
    return {"max_share": float(share[top]), "max_share_quarter": support[top], "n_f": float(1.0 / (share ** 2).sum()),
            "max_leverage": float(op.leverage[peak]), "max_leverage_quarter": support[peak]}


def screen(op: Operator, beta: float, se: float) -> str:
    if not op.rank_ok:
        return "rank_deficient"
    if not op.resolved:
        return "numerically_unresolved"
    if not op.partial_variation > 0:
        return "no_partial_variation"
    if op.n_eff < max(2 * op.k, op.k + 20):
        return "effective_sample_below_screen"
    if not (math.isfinite(beta) and math.isfinite(se) and se > 0):
        return "nonfinite"
    return "reported"


def local_rows(panel: Panel, half_width: int, role: str, flags: Mapping[str, float],
               numerical: Mapping[str, float]) -> list[dict[str, Any]]:
    first, last = int(panel.ordinals[0]), int(panel.ordinals[-1])
    rows = []
    for quarter, target in zip(panel.quarters, panel.ordinals):
        op = operator(panel, kernel_weights(panel.ordinals, int(target), half_width), numerical)
        beta, se = apply(op, panel.ordinals, panel.y, panel.hac_lags) if op.rank_ok else (math.nan, math.nan)
        beta, se = float(beta), float(se)
        status = screen(op, beta, se)
        reported = status == "reported"
        support = panel.ordinals[op.mask]
        row = {"panel": panel.name, "quarter": quarter, "half_width": half_width, "role": role, "status": status,
               "beta": beta if reported else None, "se": se if reported else None,
               "lower95": beta - Z95 * se if reported else None, "upper95": beta + Z95 * se if reported else None,
               "hac_lags": panel.hac_lags, "finite_sample_scale": op.scale, "n_support": int(op.mask.sum()),
               "n_eff": op.n_eff, "k": op.k, "screen_min_n_eff": max(2 * op.k, op.k + 20),
               "support_start": panel.quarters[int(np.flatnonzero(op.mask)[0])],
               "support_end": panel.quarters[int(np.flatnonzero(op.mask)[-1])],
               "boundary": bool(target - (half_width - 1) < first or target + (half_width - 1) > last),
               "centroid_shift_quarters": float(op.weights @ support / op.weights.sum() - target),
               "reciprocal_condition": op.reciprocal_condition, "controls_rejected": ",".join(op.rejected)}
        if op.rank_ok and op.partial_variation > 0:
            infl = influence(op, panel.quarters)
            row |= infl | {"influence_flag": bool(infl["max_share"] > flags["max_share_above"]
                                                  or infl["n_f"] < flags["n_F_below"]
                                                  or infl["max_leverage"] > flags["max_leverage_above"])}
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- panels


def load_deposit(root: Path, spec: Mapping[str, Any]) -> Panel:
    root = Path(root)
    sealed_path = pinned(root, spec["sealed_refresh_receipt"])
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    lane = sealed["lanes"]["open01"]
    lane_path = sealed_path.parent / lane["path"]
    if _sha256(lane_path) != lane["sha256"]:
        raise ValueError("OPEN-01 lane receipt differs from the sealed refresh receipt")
    run_root = sealed_path.parent / "run"
    lane_receipt = json.loads(lane_path.read_text(encoding="utf-8"))
    receipt_path = pinned(root, spec["panel_receipt"])
    if (run_root / lane_receipt["panel_receipt"]["path"]).resolve() != receipt_path.resolve() or \
            lane_receipt["panel_receipt"]["sha256"] != spec["panel_receipt"]["sha256"]:
        raise ValueError("Panel receipt is not the one bound by the sealed OPEN-01 lane")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    panel_path = pinned(root, spec["panel"])
    if receipt["panel"]["sha256"] != spec["panel"]["sha256"] or receipt["control_ids"] != list(spec["controls"]):
        raise ValueError("Panel or control order differs from the sealed panel receipt")
    with panel_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = tuple(reader)
    if header != ["quarter", spec["treatment"], spec["outcome"], *spec["controls"]]:
        raise ValueError("Deposit panel columns differ from the declared specification")
    quarters = tuple(r["quarter"] for r in rows)
    sample = spec["sample"]
    if quarters != rp.quarter_range(sample["start"], sample["end"]) or len(quarters) != sample["n"]:
        raise ValueError("Deposit panel quarters differ from the declared sample")

    def column(name: str) -> np.ndarray:
        return np.asarray([float(r[name]) for r in rows])

    return Panel("deposit", quarters, np.asarray([_quarter_ordinal(q) for q in quarters]),
                 column(spec["outcome"]), column(spec["treatment"]),
                 np.column_stack([column(c) for c in spec["controls"]]), tuple(spec["controls"]),
                 int(spec["hac_lags"]), spec["treatment"], spec["outcome"], rows)


def load_reserve(root: Path, spec: Mapping[str, Any]) -> tuple[Panel, rp.Design, rp.Panel]:
    design = rp.load_design(Path(root), spec["design_config"])
    if (design.primary_outcome, design.primary_variant, design.primary_spec) != (spec["outcome"], spec["treatment"], spec["spec"]):
        raise ValueError("Reserve panel must be the reserve design's primary cell")
    if design.hac_lags[0] != spec["hac_lags"]:
        raise ValueError("Reserve HAC lags differ from the reserve design")
    source = rp.load_panel(Path(root), design)
    positions = source.positions(design.sample)
    controls = rp.control_matrix(source.qoy[positions], design.specs[design.primary_spec], {})[:, 1:]
    panel = Panel("reserve", design.sample, source.ordinals[positions], source.common[spec["outcome"]][positions],
                  source.column(spec["treatment"], "x")[positions], controls,
                  ("quarter_of_year_q2", "quarter_of_year_q3", "quarter_of_year_q4"), int(spec["hac_lags"]))
    return panel, design, source


# --------------------------------------------------------------------------- parity


def _gap(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1e-12)


def _finite_estimate(beta: float, se: float) -> bool:
    return math.isfinite(beta) and math.isfinite(se) and se > 0


def parity(panel: Panel, rolling: Sequence[Mapping[str, Any]], full: Mapping[str, float], window: int,
           tolerance: Mapping[str, float], numerical: Mapping[str, float]) -> dict[str, Any]:
    """Uniform kernels must reproduce every admitted rolling row and the full-sample estimate."""
    def compare(label: str, weights: np.ndarray, reference: Mapping[str, Any]) -> tuple[float, float, Operator]:
        op = operator(panel, weights, numerical)
        if not (op.rank_ok and op.resolved):
            raise ValueError(f"{panel.name} {label} is not a resolved full-rank fit")
        beta, se = (float(v) for v in apply(op, panel.ordinals, panel.y, panel.hac_lags))
        if not (_finite_estimate(beta, se) and _finite_estimate(float(reference["beta"]), float(reference["se"]))):
            raise ValueError(f"{panel.name} {label} has a nonfinite estimate or reference")
        return _gap(beta, reference["beta"]), _gap(se, reference["se"]), op

    gaps = {"beta": 0.0, "se": 0.0}
    for row in rolling:
        label = f"rolling window ending {row['end']}"
        beta_gap, se_gap, op = compare(label, trailing_weights(panel.ordinals, _quarter_ordinal(row["end"]), window), row)
        if int(op.mask.sum()) != int(row["n"]):
            raise ValueError(f"{panel.name} {label} has {int(op.mask.sum())} rows, admitted {row['n']}")
        if "controls" in row and list(op.admitted) != list(row["controls"]):
            raise ValueError(f"{panel.name} {label} admits different controls")
        gaps["beta"], gaps["se"] = max(gaps["beta"], beta_gap), max(gaps["se"], se_gap)
    gaps["full_beta"], gaps["full_se"], _ = compare("full sample", np.ones(len(panel.quarters)), full)
    if max(gaps["beta"], gaps["full_beta"]) > tolerance["beta_relative"] or max(gaps["se"], gaps["full_se"]) > tolerance["se_relative"]:
        raise ValueError(f"{panel.name} uniform-kernel parity fails: {gaps}")
    return {"rolling_rows": len(rolling), **gaps}


# ---------------------------------------------------------------------- calibration


def ar1_errors(rng: np.random.Generator, span: int, replications: int, rho: float, innovation: Mapping[str, Any],
               burn_in: int) -> np.ndarray:
    total = span + burn_in
    if innovation["id"] == "gaussian":
        shocks = rng.standard_normal((total, replications))
    else:
        df = float(innovation["df"])
        shocks = rng.standard_t(df, (total, replications)) / math.sqrt(df / (df - 2.0))
    path = np.empty_like(shocks)
    path[0] = shocks[0]
    innovation_scale = math.sqrt(1.0 - rho ** 2)
    for t in range(1, total):
        path[t] = rho * path[t - 1] + innovation_scale * shocks[t]
    return path[burn_in:]


def calendar_ar1(residuals: np.ndarray, ordinals: np.ndarray) -> float:
    lagged = {int(o): r for o, r in zip(ordinals, residuals)}
    pairs = [(r, lagged[int(o) - 1]) for o, r in zip(ordinals, residuals) if int(o) - 1 in lagged]
    current, previous = (np.asarray(v) for v in zip(*pairs))
    return float(current @ previous / (previous @ previous))


def means(panel: Panel, nuisance: np.ndarray, numerical: Mapping[str, float]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    op = operator(panel, np.ones(len(panel.quarters)), numerical)
    residuals = op.residual_maker @ panel.y
    fitted = panel.y - residuals
    position = (panel.ordinals - panel.ordinals[0]) / (panel.ordinals[-1] - panel.ordinals[0])
    steps = 0.5 * (panel.ordinals >= _quarter_ordinal("2008Q4")) - 0.5 * (panel.ordinals >= _quarter_ordinal("2020Q1"))
    z = (nuisance - nuisance.mean()) / nuisance.std()
    out = {"pooled": fitted, "slope_drift": fitted + (position - 0.5) * panel.x,
           "slope_steps": fitted + steps * panel.x,
           "nuisance_drift": fitted + 0.25 * panel.y.std() * (2 * position - 1) * z}
    return {m: out[m] for m in MEAN_CLASSES}, residuals


def classify(coverage: float, mc_se: float, config: Mapping[str, Any]) -> str:
    """Failed if clearly below the floor, qualified only if clearly above it, otherwise inconclusive."""
    margin, floor = config["mc_se_multiplier"] * mc_se, config["minimum_acceptable"]
    if coverage + margin < floor:
        return COVERAGE[0]
    return COVERAGE[2] if coverage - margin > floor else COVERAGE[1]


def calibrate(panel: Panel, panel_index: int, half_width: int, targets: Sequence[str], config: Mapping[str, Any],
              nuisance: np.ndarray, numerical: Mapping[str, float]) -> list[dict[str, Any]]:
    mean_paths, residuals = means(panel, nuisance, numerical)
    scales = {"absolute_residual": np.abs(residuals),
              "homoskedastic": np.full(len(residuals), math.sqrt(float(residuals @ residuals) / len(residuals)))}
    ar_values = {a["id"]: (calendar_ar1(residuals, panel.ordinals) if a["id"] == "empirical_ar1" else float(a["value"]))
                 for a in config["ar1"]}
    span = int(panel.ordinals[-1] - panel.ordinals[0] + 1)
    offsets = panel.ordinals - panel.ordinals[0]
    operators = {q: operator(panel, kernel_weights(panel.ordinals, int(o), half_width), numerical)
                 for q, o in zip(panel.quarters, panel.ordinals) if q in set(targets)}
    if len(operators) != len(set(targets)) or not all(op.rank_ok and op.resolved for op in operators.values()):
        raise ValueError("Calibration targets must be resolved sample quarters")
    replications = int(config["replications"])
    rows = []
    index = 0
    for mean_id, mu in mean_paths.items():
        for scale_id in SCALES:
            for ar_id, rho in ar_values.items():
                for innovation in config["innovations"]:
                    rng = np.random.default_rng(np.random.SeedSequence([int(config["seed"]), panel_index, index]))
                    errors = ar1_errors(rng, span, replications, rho, innovation, int(config["burn_in"]))[offsets]
                    simulated = mu[:, None] + scales[scale_id][:, None] * errors
                    for quarter, op in operators.items():
                        beta, se = apply(op, panel.ordinals, simulated, panel.hac_lags)
                        target = float(op.slope_row @ mu[op.mask])
                        coverage = float((np.abs(beta - target) <= Z95 * se).mean())
                        mc_se = math.sqrt(coverage * (1 - coverage) / replications)
                        rows.append({"panel": panel.name, "half_width": half_width, "quarter": quarter, "mean": mean_id,
                                     "scale": scale_id, "ar1": ar_id, "rho": rho, "innovation": innovation["id"],
                                     "dgp_index": index, "replications": replications, "target": target,
                                     "mean_estimate": float(beta.mean()), "sd_estimate": float(beta.std()),
                                     "mean_se": float(se.mean()), "coverage": coverage, "monte_carlo_se": mc_se,
                                     "status": classify(coverage, mc_se, config)})
                    index += 1
    return rows


def coverage_status(calibration: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """A target qualifies only if every declared DGP qualifies; any failure fails it."""
    by_quarter: dict[str, list[Mapping[str, Any]]] = {}
    for row in calibration:
        by_quarter.setdefault(row["quarter"], []).append(row)
    out = {}
    for quarter, rows in by_quarter.items():
        worst = min(rows, key=lambda r: r["coverage"])
        status = COVERAGE[min(COVERAGE.index(r["status"]) for r in rows)]
        out[quarter] = {"coverage_status": status, "coverage_min": worst["coverage"],
                        "coverage_min_dgp": f"{worst['mean']}/{worst['scale']}/{worst['ar1']}/{worst['innovation']}",
                        "coverage_dgps_failed": sum(r["status"] == COVERAGE[0] for r in rows),
                        "coverage_dgps_inconclusive": sum(r["status"] == COVERAGE[1] for r in rows)}
    return out
