#!/usr/bin/env python3
"""Compare a sealed deposit-outcome variant with the sealed vintage it varies.

Both vintages must be sealed and differ only in their declared deposit outcome and,
when the variant declares one, its treatment source column (and the TDCest root that
produced it). The comparison first proves the two projection panels share every
control and the factor scores, and the treatment unless one is declared, then reports the retained OPEN-01 and OPEN-16
results of each side by side, the five-row conditioning and HAC audit of each,
and the cumulative treatment and outcome over the sample. Nothing is refit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_vintage_refresh import (  # noqa: E402
    FACTOR_SCORES_LOCATOR,
    PANEL_LOCATOR,
    _plain,
    _verify_clean_commit,
)

from ea_tdc.open_contract import CANONICAL_CONTROL_IDS, CANONICAL_OUTCOME_ID, CANONICAL_TREATMENT_ID  # noqa: E402
from ea_tdc.vintage import (  # noqa: E402
    Vintage,
    declared_treatment,
    deposit_outcome,
    deposit_outcome_terms,
    load_vintage,
    sha256_file,
    treatment_source_series,
)

HEADLINE = "outputs/tables/tdc_same_quarter_headline.csv"
PRODUCER_RECEIPT = "output/manifests/open01_producer_run_receipt.json"
ROLLING = "output/models/tier2_rolling_selected_credit_rate_pass_through_estimates.csv"
GATE = "output/reports/tier2_pass_through_stability_gate.csv"
CORRELATES = "output/reports/tier2_pass_through_offset_rolling_beta_correlates.csv"
FACTOR_TAIL = "output/reports/submission_factor_tail_robustness.csv"
HAC = "output/reports/submission_hac_bandwidth_sensitivity.csv"
AUDIT_FIELDS = ("audit_row", "variant", "factor_count", "newey_west_lag", "n", "beta", "se", "raw_p_value",
                "lower95", "upper95", "absolute_beta_change_from_k100_lag1", "source_artifact", "source_sha256")
PANDEMIC_ENDPOINTS = ("2022Q3", "2026Q2")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _one(rows: Sequence[Mapping[str, str]], what: str) -> Mapping[str, str]:
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one {what} row, found {len(rows)}")
    return rows[0]


def conditioning_audit(run_root: Path) -> list[dict[str, str]]:
    """k100 at Newey-West lags 1, 4, 6 and 8 and the no-factor fit, for the canonical outcome."""
    tail, hac = run_root / FACTOR_TAIL, run_root / HAC
    by_variant = {r["variant"]: r for r in _rows(tail) if r["outcome"] == CANONICAL_OUTCOME_ID and r["covariance_lags"] == "1"}
    by_lag = {r["covariance_lags"]: r for r in _rows(hac) if r["outcome"] == CANONICAL_OUTCOME_ID}
    selected = [(by_variant["k100_factors_4"], "k100_factors_4", by_variant["k100_factors_4"]["factors_requested"], tail),
                (by_variant["k0_no_factors"], "k0_no_factors", by_variant["k0_no_factors"]["factors_requested"], tail),
                *[(by_lag[lag], "k100_factors_4", "4", hac) for lag in ("4", "6", "8")]]
    base = float(selected[0][0]["beta"])
    if by_lag["1"]["beta"] != selected[0][0]["beta"] or by_lag["1"]["se"] != selected[0][0]["se"]:
        raise ValueError("Factor-tail and HAC tables disagree on the k100 lag-1 fit")
    audit = []
    for number, (row, variant, factors, source) in enumerate(selected, start=1):
        audit.append({"audit_row": str(number), "variant": variant, "factor_count": factors,
                      "newey_west_lag": row["covariance_lags"], "n": row["n"], "beta": row["beta"], "se": row["se"],
                      "raw_p_value": row["p_value"], "lower95": row["lower95"], "upper95": row["upper95"],
                      "absolute_beta_change_from_k100_lag1": repr(abs(float(row["beta"]) - base)),
                      "source_artifact": source.relative_to(run_root).as_posix(), "source_sha256": sha256_file(source)})
    return audit


def write_audit(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def identical_design(baseline: Path, variant: Path, treatment_declared: bool = False) -> dict[str, Any]:
    """The panels may differ in the outcome (and a declared treatment) only; controls and factor scores match exactly."""
    left, right = _rows(baseline / PANEL_LOCATOR), _rows(variant / PANEL_LOCATOR)
    if [r["quarter"] for r in left] != [r["quarter"] for r in right] or set(left[0]) != set(right[0]):
        raise ValueError("Projection panels differ in quarters or columns")
    from ea_tdc.open_contract import OPEN_CONTRACT

    treatment_ids = [CANONICAL_TREATMENT_ID, OPEN_CONTRACT.mmf_lower_treatment_id, OPEN_CONTRACT.mmf_upper_treatment_id]
    shared = [*([] if treatment_declared else treatment_ids), *CANONICAL_CONTROL_IDS]
    allowed = {CANONICAL_OUTCOME_ID, *(treatment_ids if treatment_declared else [])}
    differing = sorted({k for a, b in zip(left, right, strict=True) for k in a if a[k] != b[k]})
    if set(differing) & set(shared) or set(differing) - allowed:
        raise ValueError(f"Variant changes more than the deposit outcome: {differing}")
    scores_identical = sha256_file(baseline / FACTOR_SCORES_LOCATOR) == sha256_file(variant / FACTOR_SCORES_LOCATOR)
    if not scores_identical:
        raise ValueError("Factor scores differ; the frozen factors were not reused")
    return {"quarters": len(left), "columns_differing": differing, "controls_identical": True,
            "treatment_identical": CANONICAL_TREATMENT_ID not in differing, "full_factor_scores_identical": True}


def sealed_configuration(vintage: Vintage, sealed: Mapping[str, Any]) -> None:
    """A seal may bind an earlier commit's configuration that lacked only ``reuse_acquisition``.

    The configuration bytes at the seal's producer commit must hash to the sealed record,
    and equal the current configuration in every other field.
    """
    shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{sealed['producer_commit']}:{vintage.locator}"],
                           check=True, capture_output=True).stdout
    earlier, current = json.loads(shown), dict(vintage.payload)
    differs = {k for k in set(earlier) | set(current) if earlier.get(k) != current.get(k)}
    expected = {k: v for k, v in vintage.record().items() if k != "sha256"}
    if (hashlib.sha256(shown).hexdigest() != sealed["vintage"]["sha256"] or differs - {"reuse_acquisition"}
            or {k: v for k, v in sealed["vintage"].items() if k != "sha256"} != expected):
        raise ValueError(f"{vintage.locator} was sealed under another configuration")


def summarize(vintage: Vintage) -> dict[str, Any]:
    output = ROOT / vintage.payload["output_root"]
    run_root = output / "run"
    seal = output / "receipt.json"
    if not seal.is_file():
        raise ValueError(f"{vintage.locator} is not sealed")
    sealed = json.loads(seal.read_text(encoding="utf-8"))
    if sealed["vintage"] != vintage.record():
        sealed_configuration(vintage, sealed)
    headline = _one([r for r in _rows(run_root / HEADLINE)
                     if r["outcome_id"] == CANONICAL_OUTCOME_ID and r["treatment_id"] == CANONICAL_TREATMENT_ID], "headline")
    acceptance = json.loads((run_root / PRODUCER_RECEIPT).read_text(encoding="utf-8"))["acceptance"]
    checks = acceptance["acceptance_checks"]
    gate = _rows(run_root / GATE)
    overall = _one([r for r in gate if r["test_type"] == "overall_gate"], "overall gate")
    scan = _one([r for r in gate if r["test_type"] == "unknown_break_scan_summary"], "break scan")
    rolling = [r for r in _rows(run_root / ROLLING) if r["outcome"] == CANONICAL_OUTCOME_ID and r["horizon"] == "0"]
    betas = [(float(r["beta"]), r["window_end_quarter"]) for r in rolling]
    by_end = {r["window_end_quarter"]: r for r in rolling}
    legs = {r["leg"]: float(r["contribution"]) for r in _rows(output / "open16/leg_covariance_contributions.csv")
            if r["classification"] == "corrected"}
    pandemic = {r["nominal_end"]: r for r in _rows(output / "open16/pandemic_calendar_deletion.csv")}
    panel = _rows(run_root / PANEL_LOCATOR)
    correlates = {"|".join((r["credit_outcome_id"], r["window_quarters"], r["adjustment"])): r["admission_status"]
                  for r in _rows(run_root / CORRELATES)}
    return {
        "vintage": vintage.record(),
        "sealed_receipt": {"path": seal.relative_to(ROOT).as_posix(), "sha256": sha256_file(seal),
                           "producer_commit": sealed["producer_commit"], "run_id": sealed["open01"]["run_id"]},
        "deposit_outcome_terms": (deposit_outcome_terms(deposit_outcome(vintage)) if deposit_outcome(vintage)
                                  else [("BOGZ1FL764100005Q", 1)]),
        "treatment_source_series": treatment_source_series(vintage),
        "headline": {k: float(headline[k]) for k in ("beta", "se", "lower95", "upper95", "p_value_raw", "p_value_holm")}
        | {"n": int(headline["n"])},
        "acceptance": {"checks_passed": sum(bool(v["passed"]) for v in checks.values()), "checks": len(checks),
                       "status": acceptance["status"], "scientific_status": acceptance["scientific_status"]},
        "stability": {"overall": overall["scientific_status"],
                      "largest_scan_change": float(scan["maximum_abs_beta_change"]),
                      "largest_scan_change_quarter": scan["maximum_abs_beta_change_break_quarter"]},
        "rolling": {"windows": len(rolling), "min": {"beta": min(betas)[0], "window_end": min(betas)[1]},
                    "max": {"beta": max(betas)[0], "window_end": max(betas)[1]},
                    "end_2026q2": {k: float(by_end["2026Q2"][k]) for k in ("beta", "se", "lower95", "upper95")},
                    "negative_windows": sum(beta < 0 for beta, _ in betas)},
        "leg_covariance_contributions": legs,
        "pandemic_calendar_deletion": {q: {k: float(pandemic[q][k]) for k in ("beta", "se", "lower95", "upper95")}
                                       for q in PANDEMIC_ENDPOINTS},
        "credit_screen_admission": correlates,
        "cumulative_usd_billion": {
            "start": vintage.sample_start, "end": vintage.sample_end,
            "treatment": sum(float(r[CANONICAL_TREATMENT_ID]) for r in panel) / 1000.0,
            "outcome": sum(float(r[CANONICAL_OUTCOME_ID]) for r in panel) / 1000.0},
    }


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--producer-commit", required=True)
    args = parser.parse_args(argv)
    _verify_clean_commit(ROOT, args.producer_commit)
    baseline, variant = load_vintage(ROOT, args.baseline), load_vintage(ROOT, args.variant)
    if deposit_outcome(baseline) is not None or deposit_outcome(variant) is None:
        raise ValueError("Compare an admitted-outcome vintage with a declared deposit-outcome variant")
    differs = {k for k in set(baseline.payload) | set(variant.payload) if baseline.payload.get(k) != variant.payload.get(k)}
    treatment_declared = declared_treatment(baseline) is None and declared_treatment(variant) is not None
    expected = {"authority", "output_root", "deposit_outcome", *(("treatment", "tdcest") if treatment_declared else ())}
    if differs != expected:
        raise ValueError(f"Variant differs from its baseline beyond the deposit outcome: {sorted(differs)}")
    target = ROOT / _plain(args.output, "Comparison output")
    if target.exists():
        raise ValueError("Comparison output already exists; no overwrite")
    roots = {name: ROOT / v.payload["output_root"] / "run" for name, v in (("baseline", baseline), ("variant", variant))}
    design = identical_design(roots["baseline"], roots["variant"], treatment_declared)
    summaries = {"baseline": summarize(baseline), "variant": summarize(variant)}
    changed = {k: [summaries["baseline"]["credit_screen_admission"][k], v]
               for k, v in summaries["variant"]["credit_screen_admission"].items()
               if summaries["baseline"]["credit_screen_admission"].get(k) != v}
    target.mkdir(parents=True)
    outputs = {}
    for name, root in roots.items():
        path = target / f"frozen_conditioning_hac_audit_{name}.csv"
        audit = conditioning_audit(root)
        write_audit(path, audit)
        summaries[name]["conditioning_audit"] = audit
        outputs[path.name] = sha256_file(path)
    comparison = {"schema_version": "ea_tdc_deposit_outcome_comparison_v1", "producer_commit": args.producer_commit,
                  "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                  "design_identity": design, "credit_screen_admission_changes": changed, **summaries}
    path = target / "comparison.json"
    path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs[path.name] = sha256_file(path)
    receipt = {"schema_version": "ea_tdc_deposit_outcome_comparison_receipt_v1", "producer_commit": args.producer_commit,
               "baseline": summaries["baseline"]["sealed_receipt"], "variant": summaries["variant"]["sealed_receipt"],
               "outputs": outputs}
    (target / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _verify_clean_commit(ROOT, args.producer_commit)
    return target / "receipt.json"


if __name__ == "__main__":
    result = main()
    print(json.dumps({"receipt": result.relative_to(ROOT).as_posix(), "sha256": sha256_file(result)}))
