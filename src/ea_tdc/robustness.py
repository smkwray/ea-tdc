from __future__ import annotations

import csv
import json
import math
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import fmean, median
from typing import Any

from ea_tdc.designs.quarterly import (
    _coerce_float,
    _load_jobs,
    _previous_quarter,
    _quarter_end_date,
    _quarter_sort_key,
    _stable_float_text,
)
from ea_tdc.estimation import _estimate_rows, _write_estimates_csv
from ea_tdc.paths import ProjectPaths
from ea_tdc.utils import utc_now_iso, write_json


DEFAULT_K_GRID = [100, 200, 300]
DEFAULT_FACTOR_COUNT = 4
RECOMMENDED_K_RELATIVE_FIT_FLOOR = 0.9
DAILY_LAGS = 90
MONTHLY_LAGS = 12
QUARTERLY_LAGS = 8
MIN_COVERAGE = 0.4
TOP_LOADINGS_PER_FACTOR = 12
DEFAULT_REGIME_STATES = [
    "coord_low_reserve_state_l1",
    "coord_on_rrp_drain_state_l1",
    "tsyparty_bank_absorption_share_l1",
    "tsyparty_bank_foreign_private_corr_l1",
    "wamest_bank_reserve_short_share_l1",
    "slrwatch_bank_leverage_pressure_l1",
]
ALTERNATIVE_TREATMENTS = [
    "tdc_base_broad_depository_np_cu_ru_flow",
    "tdc_no_remit_bank_only",
    "tdc_domestic_bank_only_ru_flow",
    "tdc_bank_only_extended_1990",
]


@dataclass(frozen=True)
class ControlUniverseResult:
    panel_path: Path
    meta_path: Path
    columns_path: Path
    quarter_count: int
    feature_count: int


@dataclass(frozen=True)
class QuarterlyRobustnessResult:
    summary_path: Path
    ladder_path: Path
    regime_path: Path
    treatment_path: Path
    control_meta_path: Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_slug(text: str) -> str:
    lowered = "".join(char.lower() if char.isalnum() else "_" for char in str(text))
    while "__" in lowered:
        lowered = lowered.replace("__", "_")
    return lowered.strip("_")


def _repo_relative_text(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _series_key_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("FRED_"):
        stem = stem[5:]
    return _safe_slug(stem)


def _parse_series_rows(path: Path) -> list[tuple[date, float]]:
    rows: list[tuple[date, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            date_text = str(row.get("date", "")).strip()
            value_text = str(row.get("value", "")).strip()
            if not date_text or not value_text or value_text == ".":
                continue
            try:
                rows.append((date.fromisoformat(date_text[:10]), float(value_text)))
            except ValueError:
                continue
    rows.sort(key=lambda item: item[0])
    return rows


def _infer_frequency(observations: list[tuple[date, float]]) -> str:
    if len(observations) < 3:
        return "q"
    gaps = []
    for idx in range(len(observations) - 1):
        gap = (observations[idx + 1][0] - observations[idx][0]).days
        if gap > 0:
            gaps.append(gap)
    if not gaps:
        return "q"
    median_gap = sorted(gaps)[len(gaps) // 2]
    if median_gap <= 10:
        return "d"
    if median_gap <= 45:
        return "m"
    return "q"


def _month_end(year: int, month: int) -> date:
    if month == 12:
        return date(year + 1, 1, 1) - timedelta(days=1)
    return date(year, month + 1, 1) - timedelta(days=1)


def _shift_month_end(anchor: date, lag_index: int) -> date:
    absolute_month = (anchor.year * 12 + (anchor.month - 1)) - lag_index
    year = absolute_month // 12
    month = (absolute_month % 12) + 1
    return _month_end(year, month)


def _shift_quarter_end(anchor: date, lag_index: int) -> date:
    absolute_quarter = (anchor.year * 4 + ((anchor.month - 1) // 3)) - lag_index
    year = absolute_quarter // 4
    quarter_num = (absolute_quarter % 4) + 1
    return _quarter_end_date(f"{year}Q{quarter_num}")


def _value_on_or_before(
    dates: list[date],
    values: list[float],
    target: date,
) -> float | None:
    index = bisect_right(dates, target) - 1
    if index < 0:
        return None
    return values[index]


def _anchor_for_quarter(quarter: str) -> date | None:
    previous = _previous_quarter(quarter)
    if previous is None:
        return None
    return _quarter_end_date(previous)


def _quarter_grid_from_existing_design(paths: ProjectPaths) -> list[str]:
    preferred = paths.bundles / "designs" / "baseline_tdc_lp_deposits__quarterly_bundle.csv"
    if preferred.exists():
        return [str(row.get("quarter", "")).strip() for row in _read_csv(preferred) if str(row.get("quarter", "")).strip()]
    candidates = sorted((paths.bundles / "designs").glob("*__quarterly_bundle.csv"))
    for candidate in candidates:
        quarters = [str(row.get("quarter", "")).strip() for row in _read_csv(candidate) if str(row.get("quarter", "")).strip()]
        if quarters:
            return quarters
    raise FileNotFoundError("No quarterly design bundle found to define the control-universe quarter grid")


def build_control_universe(
    paths: ProjectPaths,
    *,
    quarter_grid: list[str] | None = None,
    daily_lags: int = DAILY_LAGS,
    monthly_lags: int = MONTHLY_LAGS,
    quarterly_lags: int = QUARTERLY_LAGS,
) -> ControlUniverseResult:
    quarter_ids = quarter_grid or _quarter_grid_from_existing_design(paths)
    raw_dir = paths.seed / "interpol" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing interpol raw seed directory: {raw_dir}")

    output_panel = paths.reports / "control_universe_quarterly.csv"
    output_meta = paths.reports / "control_universe_meta.json"
    output_columns = paths.reports / "control_universe_columns.csv"

    feature_columns: list[str] = []
    column_rows: list[dict[str, Any]] = []
    panel_rows = [{"quarter": quarter} for quarter in quarter_ids]
    quarter_row_lookup = {row["quarter"]: row for row in panel_rows}
    frequency_counts = {"d": 0, "m": 0, "q": 0}

    for path in sorted(raw_dir.glob("*.csv")):
        observations = _parse_series_rows(path)
        if not observations:
            continue
        frequency = _infer_frequency(observations)
        if frequency not in frequency_counts:
            continue
        frequency_counts[frequency] += 1
        dates = [item[0] for item in observations]
        values = [item[1] for item in observations]
        base_series = _series_key_from_path(path)
        lag_count = {"d": daily_lags, "m": monthly_lags, "q": quarterly_lags}[frequency]

        for lag in range(1, lag_count + 1):
            column_name = f"{frequency}__{base_series}__lag{lag:03d}"
            nonmissing = 0
            for quarter in quarter_ids:
                anchor = _anchor_for_quarter(quarter)
                if anchor is None:
                    quarter_row_lookup[quarter][column_name] = ""
                    continue
                if frequency == "d":
                    target = anchor - timedelta(days=lag - 1)
                elif frequency == "m":
                    target = _shift_month_end(anchor, lag - 1)
                else:
                    target = _shift_quarter_end(anchor, lag)
                value = _value_on_or_before(dates, values, target)
                if value is None:
                    quarter_row_lookup[quarter][column_name] = ""
                else:
                    quarter_row_lookup[quarter][column_name] = _stable_float_text(value)
                    nonmissing += 1
            feature_columns.append(column_name)
            column_rows.append(
                {
                    "feature_id": column_name,
                    "base_series": base_series,
                    "frequency": frequency,
                    "lag": lag,
                    "source_file": path.name,
                    "nonmissing_quarters": nonmissing,
                    "coverage_share": round(nonmissing / len(quarter_ids), 6) if quarter_ids else 0.0,
                }
            )

    _write_csv(output_panel, panel_rows, fieldnames=["quarter", *feature_columns])
    _write_csv(
        output_columns,
        column_rows,
        fieldnames=["feature_id", "base_series", "frequency", "lag", "source_file", "nonmissing_quarters", "coverage_share"],
    )
    write_json(
        output_meta,
        {
            "generated_at": utc_now_iso(),
            "panel_path": str(output_panel),
            "columns_path": str(output_columns),
            "quarter_count": len(quarter_ids),
            "feature_count": len(feature_columns),
            "series_count_by_frequency": frequency_counts,
            "lag_structure": {
                "daily_lags": daily_lags,
                "monthly_lags": monthly_lags,
                "quarterly_lags": quarterly_lags,
            },
        },
    )
    return ControlUniverseResult(
        panel_path=output_panel,
        meta_path=output_meta,
        columns_path=output_columns,
        quarter_count=len(quarter_ids),
        feature_count=len(feature_columns),
    )


def _merge_control_rows(
    bundle_rows: list[dict[str, str]],
    universe_panel_path: Path,
) -> tuple[list[dict[str, str]], list[str]]:
    universe_rows = _read_csv(universe_panel_path)
    by_quarter = {str(row.get("quarter", "")).strip(): row for row in universe_rows}
    merged_rows: list[dict[str, str]] = []
    feature_ids: list[str] = []
    if universe_rows:
        feature_ids = [column for column in universe_rows[0].keys() if column != "quarter"]
    for row in bundle_rows:
        quarter = str(row.get("quarter", "")).strip()
        merged = dict(row)
        controls = by_quarter.get(quarter, {})
        for feature_id in feature_ids:
            if feature_id not in merged:
                merged[feature_id] = str(controls.get(feature_id, ""))
        merged_rows.append(merged)
    return merged_rows, feature_ids


def _load_alt_treatments(paths: ProjectPaths) -> dict[tuple[str, str], str]:
    standardized_path = paths.bundles / "tdcest" / "standardized_series.csv"
    if not standardized_path.exists():
        return {}
    rows = _read_csv(standardized_path)
    mapping: dict[tuple[str, str], str] = {}
    for row in rows:
        series_id = str(row.get("series_id", "")).strip()
        quarter = str(row.get("period_end", "")).strip()
        if not quarter or series_id not in ALTERNATIVE_TREATMENTS:
            continue
        try:
            quarter_id = f"{datetime.fromisoformat(quarter[:10]).year}Q{((datetime.fromisoformat(quarter[:10]).month - 1) // 3) + 1}"
        except ValueError:
            continue
        mapping[(series_id, quarter_id)] = str(row.get("value", "")).strip()
    return mapping


def _median_impute(values: list[float | None]) -> tuple[list[float], float]:
    observed = [value for value in values if value is not None]
    fill_value = median(observed) if observed else 0.0
    return [fill_value if value is None else value for value in values], float(fill_value)


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_mean = fmean(left)
    right_mean = fmean(right)
    left_var = sum((value - left_mean) ** 2 for value in left)
    right_var = sum((value - right_mean) ** 2 for value in right)
    if left_var <= 0 or right_var <= 0:
        return None
    covariance = sum((lval - left_mean) * (rval - right_mean) for lval, rval in zip(left, right))
    return covariance / math.sqrt(left_var * right_var)


def _screen_features(
    *,
    rows: list[dict[str, str]],
    candidate_ids: list[str],
    treatment_id: str,
    outcome_ids: list[str],
    min_coverage: float,
) -> list[dict[str, Any]]:
    screened: list[dict[str, Any]] = []
    total_rows = len(rows)
    for feature_id in candidate_ids:
        feature_values: list[float] = []
        treatment_values: list[float] = []
        coverage_count = 0
        outcome_corrs: list[float] = []
        paired_outcomes: dict[str, tuple[list[float], list[float]]] = {outcome_id: ([], []) for outcome_id in outcome_ids}
        for row in rows:
            feature_value = _coerce_float(row.get(feature_id, ""))
            if feature_value is None:
                continue
            coverage_count += 1
            treatment_value = _coerce_float(row.get(treatment_id, ""))
            if treatment_value is not None:
                feature_values.append(feature_value)
                treatment_values.append(treatment_value)
            for outcome_id in outcome_ids:
                outcome_value = _coerce_float(row.get(outcome_id, ""))
                if outcome_value is None:
                    continue
                paired_outcomes[outcome_id][0].append(feature_value)
                paired_outcomes[outcome_id][1].append(outcome_value)
        coverage = coverage_count / total_rows if total_rows else 0.0
        if coverage < min_coverage:
            continue
        corr_treat = abs(_pearson(feature_values, treatment_values) or 0.0)
        for outcome_id in outcome_ids:
            left, right = paired_outcomes[outcome_id]
            corr_value = _pearson(left, right)
            if corr_value is not None:
                outcome_corrs.append(abs(corr_value))
        score = round(coverage * (corr_treat + (max(outcome_corrs) if outcome_corrs else 0.0)), 8)
        screened.append(
            {
                "feature_id": feature_id,
                "coverage_share": round(coverage, 6),
                "abs_corr_treatment": round(corr_treat, 6),
                "abs_corr_outcome_max": round(max(outcome_corrs), 6) if outcome_corrs else 0.0,
                "screen_score": score,
            }
        )
    screened.sort(
        key=lambda item: (
            float(item["screen_score"]),
            float(item["coverage_share"]),
            float(item["abs_corr_treatment"]),
            float(item["abs_corr_outcome_max"]),
        ),
        reverse=True,
    )
    return screened


def _normalize_columns(rows: list[dict[str, str]], feature_ids: list[str]) -> tuple[list[list[float]], list[dict[str, float]]]:
    matrix_by_column: list[list[float]] = []
    column_meta: list[dict[str, float]] = []
    for feature_id in feature_ids:
        raw = [_coerce_float(row.get(feature_id, "")) for row in rows]
        imputed, fill_value = _median_impute(raw)
        mean_value = fmean(imputed) if imputed else 0.0
        variance = sum((value - mean_value) ** 2 for value in imputed) / max(len(imputed), 1)
        std_value = math.sqrt(variance)
        if std_value <= 1e-12:
            std_value = 1.0
        matrix_by_column.append([(value - mean_value) / std_value for value in imputed])
        column_meta.append({"fill_value": fill_value, "mean": mean_value, "std": std_value})
    matrix = [[matrix_by_column[col_idx][row_idx] for col_idx in range(len(feature_ids))] for row_idx in range(len(rows))]
    return matrix, column_meta


def _gram_matrix(matrix: list[list[float]]) -> list[list[float]]:
    row_count = len(matrix)
    if row_count == 0:
        return []
    gram = [[0.0 for _ in range(row_count)] for _ in range(row_count)]
    for left_idx in range(row_count):
        gram[left_idx][left_idx] = sum(value * value for value in matrix[left_idx])
        for right_idx in range(left_idx + 1, row_count):
            dot = sum(left * right for left, right in zip(matrix[left_idx], matrix[right_idx]))
            gram[left_idx][right_idx] = dot
            gram[right_idx][left_idx] = dot
    scale = max(row_count - 1, 1)
    for row_idx in range(row_count):
        for col_idx in range(row_count):
            gram[row_idx][col_idx] /= scale
    return gram


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(row[col_idx] * vector[col_idx] for col_idx in range(len(vector))) for row in matrix]


def _power_iteration(
    matrix: list[list[float]],
    *,
    previous_vectors: list[list[float]],
    max_iter: int = 100,
) -> tuple[float, list[float]] | None:
    size = len(matrix)
    if size == 0:
        return None
    vector = [1.0 + (idx / max(size, 1)) for idx in range(size)]
    for base in previous_vectors:
        dot = sum(current * prior for current, prior in zip(vector, base))
        vector = [current - dot * prior for current, prior in zip(vector, base)]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        return None
    vector = [value / norm for value in vector]

    for _ in range(max_iter):
        candidate = _matvec(matrix, vector)
        for base in previous_vectors:
            dot = sum(current * prior for current, prior in zip(candidate, base))
            candidate = [current - dot * prior for current, prior in zip(candidate, base)]
        norm = math.sqrt(sum(value * value for value in candidate))
        if norm <= 1e-12:
            return None
        next_vector = [value / norm for value in candidate]
        shift = math.sqrt(sum((left - right) ** 2 for left, right in zip(next_vector, vector)))
        vector = next_vector
        if shift < 1e-8:
            break
    eigenvalue = sum(vector[idx] * sum(matrix[idx][jdx] * vector[jdx] for jdx in range(size)) for idx in range(size))
    if eigenvalue <= 1e-12:
        return None
    return eigenvalue, vector


def _extract_factor_controls(
    *,
    rows: list[dict[str, str]],
    feature_ids: list[str],
    prefix: str,
    n_factors: int,
) -> tuple[list[str], list[dict[str, str]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows or not feature_ids:
        return [], rows, [], []
    matrix, _ = _normalize_columns(rows, feature_ids)
    gram = _gram_matrix(matrix)
    factors: list[dict[str, Any]] = []
    factor_ids: list[str] = []
    factor_rows = [dict(row) for row in rows]
    eigenvectors: list[list[float]] = []

    for factor_idx in range(1, n_factors + 1):
        eig = _power_iteration(gram, previous_vectors=eigenvectors)
        if eig is None:
            break
        eigenvalue, eigenvector = eig
        eigenvectors.append(eigenvector)
        factor_id = f"{prefix}_f{factor_idx}"
        factor_ids.append(factor_id)
        scale = math.sqrt(max(eigenvalue, 0.0))
        scores = [_stable_float_text(scale * value) for value in eigenvector]
        for row_idx, row in enumerate(factor_rows):
            row[factor_id] = scores[row_idx]
        loadings: list[tuple[str, float]] = []
        if scale > 1e-12:
            for col_idx, feature_id in enumerate(feature_ids):
                loading = sum(matrix[row_idx][col_idx] * eigenvector[row_idx] for row_idx in range(len(rows)))
                loading /= max(scale * max(len(rows) - 1, 1), 1e-12)
                loadings.append((feature_id, loading))
        loadings.sort(key=lambda item: abs(item[1]), reverse=True)
        factors.append(
            {
                "factor_id": factor_id,
                "explained_eigenvalue": round(eigenvalue, 6),
                "top_loadings": [
                    {"feature_id": feature_id, "loading": round(loading, 6)}
                    for feature_id, loading in loadings[:TOP_LOADINGS_PER_FACTOR]
                ],
            }
        )
    factor_column_rows = [
        {
            "quarter": row.get("quarter", ""),
            **{factor_id: row.get(factor_id, "") for factor_id in factor_ids},
        }
        for row in factor_rows
    ]
    factor_loadings_rows: list[dict[str, Any]] = []
    for factor in factors:
        for rank, loading in enumerate(factor["top_loadings"], start=1):
            factor_loadings_rows.append(
                {
                    "factor_id": factor["factor_id"],
                    "rank": rank,
                    "feature_id": loading["feature_id"],
                    "loading": loading["loading"],
                    "explained_eigenvalue": factor["explained_eigenvalue"],
                }
            )
    return factor_ids, factor_rows, factors, factor_loadings_rows


def _base_controls_from_design(design_manifest: dict[str, Any]) -> list[str]:
    return [str(item) for item in design_manifest.get("control_ids", []) if str(item).strip()]


def _estimate_for_rows(
    *,
    job_id: str,
    job: dict[str, Any],
    design_manifest: dict[str, Any],
    rows: list[dict[str, str]],
    control_ids: list[str],
    treatment_id: str,
) -> list[dict[str, str]]:
    estimator = str(job.get("estimator", "")).strip()
    response_type = str(job.get("response_type", "direct_at_h")).strip()
    return _estimate_rows(
        estimator=estimator,
        bundle_rows=rows,
        treatment_id=treatment_id,
        control_ids=control_ids,
        outcome_ids=[str(item) for item in design_manifest.get("outcome_ids", [])],
        horizons=[int(item) for item in design_manifest.get("horizon_grid", [])],
        response_type=response_type,
        job_id=job_id,
        instrument_ids=[str(item) for item in design_manifest.get("instrument_ids", [])],
        state_id=str(job.get("state_id", "")).strip(),
    )


def _estimate_with_adaptive_controls(
    *,
    job_id: str,
    job: dict[str, Any],
    design_manifest: dict[str, Any],
    rows: list[dict[str, str]],
    control_ids: list[str],
    treatment_id: str,
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    candidate_controls = control_ids[:]
    dropped_controls: list[str] = []
    while True:
        try:
            estimates = _estimate_for_rows(
                job_id=job_id,
                job=job,
                design_manifest=design_manifest,
                rows=rows,
                control_ids=candidate_controls,
                treatment_id=treatment_id,
            )
            return estimates, candidate_controls, dropped_controls
        except ValueError:
            if not candidate_controls:
                raise
            dropped_controls.insert(0, candidate_controls[-1])
            candidate_controls = candidate_controls[:-1]


def _summary_from_estimates(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {
            "rows_written": 0,
            "warning_rows": 0,
            "weak_instrument_rows": 0,
            "avg_abs_beta": None,
            "avg_n": None,
        }
    abs_betas = [abs(float(row["beta"])) for row in rows if str(row.get("beta", "")).strip()]
    ns = [float(row["n"]) for row in rows if str(row.get("n", "")).strip()]
    return {
        "rows_written": len(rows),
        "warning_rows": sum(1 for row in rows if str(row.get("warning_flags", "")).strip()),
        "weak_instrument_rows": sum(1 for row in rows if str(row.get("weak_instrument_flag", "")).strip() == "true"),
        "avg_abs_beta": round(sum(abs_betas) / len(abs_betas), 6) if abs_betas else None,
        "avg_n": round(sum(ns) / len(ns), 3) if ns else None,
    }


def _select_recommended_k(screen_rows: list[dict[str, Any]]) -> tuple[int | None, str]:
    candidates = [
        {
            "k_screened": int(item.get("k_screened", 0) or 0),
            "rows_written": int(item.get("rows_written", 0) or 0),
            "warning_rows": int(item.get("warning_rows", 0) or 0),
            "weak_instrument_rows": int(item.get("weak_instrument_rows", 0) or 0),
            "avg_abs_beta": float(item.get("avg_abs_beta", 0.0) or 0.0),
        }
        for item in screen_rows
        if int(item.get("k_screened", 0) or 0) > 0 and int(item.get("rows_written", 0) or 0) > 0
    ]
    if not candidates:
        return None, "No factor-screened branch produced usable estimates."

    min_warning_rows = min(item["warning_rows"] for item in candidates)
    candidates = [item for item in candidates if item["warning_rows"] == min_warning_rows]
    min_weak_rows = min(item["weak_instrument_rows"] for item in candidates)
    candidates = [item for item in candidates if item["weak_instrument_rows"] == min_weak_rows]

    best_avg_abs_beta = max(item["avg_abs_beta"] for item in candidates)
    fit_floor = best_avg_abs_beta * RECOMMENDED_K_RELATIVE_FIT_FLOOR
    near_best = [item for item in candidates if item["avg_abs_beta"] >= fit_floor]
    selected = min(near_best or candidates, key=lambda item: item["k_screened"])
    reason = (
        "Selected the smallest factor-screened branch within "
        f"{int(RECOMMENDED_K_RELATIVE_FIT_FLOOR * 100)}% of the strongest "
        "warning-minimized average absolute response."
    )
    return selected["k_screened"], reason


def _filter_rows_by_regime(rows: list[dict[str, str]], regime_id: str) -> list[dict[str, str]]:
    if regime_id == "exclude_2008_2009":
        return [row for row in rows if not str(row.get("quarter", "")).startswith(("2008Q", "2009Q"))]
    if regime_id == "exclude_2020":
        return [row for row in rows if not str(row.get("quarter", "")).startswith("2020Q")]
    if regime_id.endswith("__high"):
        state_id = regime_id[:-6]
        values = [_coerce_float(row.get(state_id, "")) for row in rows]
        observed = [value for value in values if value is not None]
        if not observed:
            return []
        threshold = median(observed)
        return [row for row in rows if (value := _coerce_float(row.get(state_id, ""))) is not None and value >= threshold]
    if regime_id.endswith("__low"):
        state_id = regime_id[:-5]
        values = [_coerce_float(row.get(state_id, "")) for row in rows]
        observed = [value for value in values if value is not None]
        if not observed:
            return []
        threshold = median(observed)
        return [row for row in rows if (value := _coerce_float(row.get(state_id, ""))) is not None and value < threshold]
    state_id = regime_id
    return [row for row in rows if str(row.get(state_id, "")).strip() == "1"]


def build_quarterly_robustness(
    paths: ProjectPaths,
    *,
    job_id: str,
    k_grid: list[int] | None = None,
    factor_count: int = DEFAULT_FACTOR_COUNT,
    min_coverage: float = MIN_COVERAGE,
) -> QuarterlyRobustnessResult:
    jobs = _load_jobs(paths.config / "dass_job_blueprint.yaml")
    if job_id not in jobs:
        raise KeyError(f"Unknown job_id: {job_id}")
    job = jobs[job_id]
    estimator = str(job.get("estimator", "")).strip()
    if estimator not in {"lp", "lp_iv"}:
        raise ValueError(f"Quarterly robustness currently supports lp/lp_iv jobs only, not '{estimator}'")

    design_manifest_path = paths.manifests / f"{job_id}__design_manifest.json"
    if not design_manifest_path.exists():
        raise FileNotFoundError(f"Missing design manifest for job '{job_id}'")
    design_manifest = json.loads(design_manifest_path.read_text(encoding="utf-8"))
    if str(design_manifest.get("status", "")).strip() != "ready_for_estimation":
        raise ValueError(f"Design manifest for job '{job_id}' is not ready_for_estimation")

    bundle_path = Path(str(design_manifest.get("bundle_path", "")))
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing bundle for job '{job_id}': {bundle_path}")
    bundle_rows = _read_csv(bundle_path)
    bundle_rows.sort(key=lambda row: _quarter_sort_key(str(row.get("quarter", ""))))

    control_universe = build_control_universe(
        paths,
        quarter_grid=[str(row.get("quarter", "")).strip() for row in bundle_rows if str(row.get("quarter", "")).strip()],
    )
    merged_rows, universe_feature_ids = _merge_control_rows(bundle_rows, control_universe.panel_path)
    alt_treatment_map = _load_alt_treatments(paths)
    for row in merged_rows:
        quarter = str(row.get("quarter", "")).strip()
        for treatment_id in ALTERNATIVE_TREATMENTS:
            row.setdefault(treatment_id, alt_treatment_map.get((treatment_id, quarter), ""))

    treatment_id = str(design_manifest.get("treatment_id", "")).strip()
    outcome_ids = [str(item) for item in design_manifest.get("outcome_ids", [])]
    screened = _screen_features(
        rows=merged_rows,
        candidate_ids=universe_feature_ids,
        treatment_id=treatment_id,
        outcome_ids=outcome_ids,
        min_coverage=min_coverage,
    )

    ladder_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    treatment_rows: list[dict[str, Any]] = []
    regime_estimate_rows: list[dict[str, Any]] = []
    treatment_estimate_rows: list[dict[str, Any]] = []
    factor_meta: list[dict[str, Any]] = []
    factor_loading_rows: list[dict[str, Any]] = []

    models_dir = paths.output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = paths.reports
    k_values = sorted({int(item) for item in (k_grid or DEFAULT_K_GRID) if int(item) > 0})
    base_controls = _base_controls_from_design(design_manifest)
    recommended_rows: list[dict[str, str]] | None = None
    recommended_factor_ids: list[str] = []
    recommended_k: int | None = None

    baseline_estimates, baseline_controls_used, baseline_controls_dropped = _estimate_with_adaptive_controls(
        job_id=job_id,
        job=job,
        design_manifest=design_manifest,
        rows=merged_rows,
        control_ids=base_controls,
        treatment_id=treatment_id,
    )
    baseline_path = models_dir / f"{job_id}__robustness_baseline_estimates.csv"
    _write_estimates_csv(baseline_path, baseline_estimates)
    baseline_summary = _summary_from_estimates(baseline_estimates)
    ladder_rows.append(
        {
            "job_id": job_id,
            "run_type": "baseline_core",
            "k_screened": 0,
            "factor_count": 0,
            "control_ids_used": ";".join(baseline_controls_used),
            "dropped_controls": ";".join(baseline_controls_dropped),
            "estimates_path": str(baseline_path),
            **baseline_summary,
        }
    )

    factor_screen_rows: list[dict[str, Any]] = []
    for k_value in k_values:
        selected = [item["feature_id"] for item in screened[:k_value]]
        if not selected:
            continue
        factor_ids, factor_rows, factors, loadings = _extract_factor_controls(
            rows=merged_rows,
            feature_ids=selected,
            prefix=f"dflmx_k{k_value}",
            n_factors=factor_count,
        )
        if not factor_ids:
            continue
        estimates, controls_used, dropped_controls = _estimate_with_adaptive_controls(
            job_id=job_id,
            job=job,
            design_manifest=design_manifest,
            rows=factor_rows,
            control_ids=[*base_controls, *factor_ids],
            treatment_id=treatment_id,
        )
        estimates_path = models_dir / f"{job_id}__robustness_k{k_value}_estimates.csv"
        _write_estimates_csv(estimates_path, estimates)
        summary = _summary_from_estimates(estimates)
        ladder_rows.append(
            {
                "job_id": job_id,
                "run_type": "factor_screen",
                "k_screened": k_value,
                "factor_count": len(factor_ids),
                "control_ids_used": ";".join(controls_used),
                "dropped_controls": ";".join(dropped_controls),
                "estimates_path": str(estimates_path),
                **summary,
            }
        )
        factor_screen_rows.append(
            {
                "k_screened": k_value,
                **summary,
            }
        )
        factor_meta.extend(
            {
                "job_id": job_id,
                "k_screened": k_value,
                **item,
            }
            for item in factors
        )
        factor_loading_rows.extend(
            {
                "job_id": job_id,
                "k_screened": k_value,
                **item,
            }
            for item in loadings
        )
    recommended_k, recommended_k_reason = _select_recommended_k(factor_screen_rows)
    if recommended_k is not None:
        selected = [item["feature_id"] for item in screened[:recommended_k]]
        recommended_factor_ids, recommended_rows, factors, loadings = _extract_factor_controls(
            rows=merged_rows,
            feature_ids=selected,
            prefix=f"dflmx_k{recommended_k}",
            n_factors=factor_count,
        )
        if not any(item.get("job_id") == job_id and item.get("k_screened") == recommended_k for item in factor_meta):
            factor_meta.extend(
                {
                    "job_id": job_id,
                    "k_screened": recommended_k,
                    **item,
                }
                for item in factors
            )
            factor_loading_rows.extend(
                {
                    "job_id": job_id,
                    "k_screened": recommended_k,
                    **item,
                }
                for item in loadings
            )
    else:
        recommended_rows = merged_rows

    regime_filters = ["exclude_2008_2009", "exclude_2020"]
    for state_id in DEFAULT_REGIME_STATES:
        if any(str(row.get(state_id, "")).strip() for row in merged_rows):
            if any(str(row.get(state_id, "")).strip() == "1" for row in merged_rows):
                regime_filters.append(state_id)
            else:
                regime_filters.extend([f"{state_id}__low", f"{state_id}__high"])

    regime_control_ids = [*base_controls, *recommended_factor_ids]
    for regime_id in regime_filters:
        filtered_rows = _filter_rows_by_regime(recommended_rows or merged_rows, regime_id)
        min_rows = 12 if regime_id in DEFAULT_REGIME_STATES or regime_id.endswith(("__low", "__high")) else 24
        if len(filtered_rows) < min_rows:
            continue
        estimates, controls_used, dropped_controls = _estimate_with_adaptive_controls(
            job_id=job_id,
            job=job,
            design_manifest=design_manifest,
            rows=filtered_rows,
            control_ids=regime_control_ids,
            treatment_id=treatment_id,
        )
        summary = _summary_from_estimates(estimates)
        regime_rows.append(
            {
                "job_id": job_id,
                "regime_id": regime_id,
                "rows_in_regime": len(filtered_rows),
                "control_ids_used": ";".join(controls_used),
                "dropped_controls": ";".join(dropped_controls),
                **summary,
            }
        )
        for row in estimates:
            regime_estimate_rows.append({"job_id": job_id, "regime_id": regime_id, **row})

    if estimator == "lp":
        for alternative_treatment in ALTERNATIVE_TREATMENTS:
            if not any(str(row.get(alternative_treatment, "")).strip() for row in merged_rows):
                continue
            estimates, controls_used, dropped_controls = _estimate_with_adaptive_controls(
                job_id=job_id,
                job=job,
                design_manifest=design_manifest,
                rows=recommended_rows or merged_rows,
                control_ids=regime_control_ids,
                treatment_id=alternative_treatment,
            )
            summary = _summary_from_estimates(estimates)
            treatment_rows.append(
                {
                    "job_id": job_id,
                    "treatment_variant": alternative_treatment,
                    "control_ids_used": ";".join(controls_used),
                    "dropped_controls": ";".join(dropped_controls),
                    **summary,
                }
            )
            for row in estimates:
                treatment_estimate_rows.append({"job_id": job_id, "treatment_variant": alternative_treatment, **row})

    ladder_path = reports_dir / f"{job_id}__robustness_ladder.csv"
    regime_path = reports_dir / f"{job_id}__regime_sensitivity.csv"
    treatment_path = reports_dir / f"{job_id}__treatment_sensitivity.csv"
    factor_meta_path = reports_dir / f"{job_id}__dflmx_factor_meta.json"
    factor_loadings_path = reports_dir / f"{job_id}__dflmx_top_loadings.csv"
    screen_path = reports_dir / f"{job_id}__control_screen.csv"
    regime_estimates_path = reports_dir / f"{job_id}__regime_sensitivity_estimates.csv"
    treatment_estimates_path = reports_dir / f"{job_id}__treatment_sensitivity_estimates.csv"
    summary_path = paths.manifests / f"{job_id}__robustness_summary.json"

    _write_csv(
        ladder_path,
        ladder_rows,
        fieldnames=["job_id", "run_type", "k_screened", "factor_count", "control_ids_used", "dropped_controls", "estimates_path", "rows_written", "warning_rows", "weak_instrument_rows", "avg_abs_beta", "avg_n"],
    )
    _write_csv(
        regime_path,
        regime_rows,
        fieldnames=["job_id", "regime_id", "rows_in_regime", "control_ids_used", "dropped_controls", "rows_written", "warning_rows", "weak_instrument_rows", "avg_abs_beta", "avg_n"],
    )
    _write_csv(
        treatment_path,
        treatment_rows,
        fieldnames=["job_id", "treatment_variant", "control_ids_used", "dropped_controls", "rows_written", "warning_rows", "weak_instrument_rows", "avg_abs_beta", "avg_n"],
    )
    _write_csv(
        regime_estimates_path,
        regime_estimate_rows,
        fieldnames=(list(regime_estimate_rows[0].keys()) if regime_estimate_rows else ["job_id", "regime_id"]),
    )
    _write_csv(
        treatment_estimates_path,
        treatment_estimate_rows,
        fieldnames=(list(treatment_estimate_rows[0].keys()) if treatment_estimate_rows else ["job_id", "treatment_variant"]),
    )
    _write_csv(
        factor_loadings_path,
        factor_loading_rows,
        fieldnames=["job_id", "k_screened", "factor_id", "rank", "feature_id", "loading", "explained_eigenvalue"],
    )
    _write_csv(
        screen_path,
        screened,
        fieldnames=["feature_id", "coverage_share", "abs_corr_treatment", "abs_corr_outcome_max", "screen_score"],
    )
    write_json(
        factor_meta_path,
        {
            "job_id": job_id,
            "generated_at": utc_now_iso(),
            "recommended_k": recommended_k,
            "recommended_k_reason": recommended_k_reason,
            "recommended_factor_count": len(recommended_factor_ids),
            "factors": factor_meta,
        },
    )
    write_json(
        summary_path,
        {
            "job_id": job_id,
            "generated_at": utc_now_iso(),
            "control_universe_meta_path": _repo_relative_text(control_universe.meta_path, paths.root),
            "control_universe_columns_path": _repo_relative_text(control_universe.columns_path, paths.root),
            "control_screen_path": _repo_relative_text(screen_path, paths.root),
            "ladder_path": _repo_relative_text(ladder_path, paths.root),
            "regime_path": _repo_relative_text(regime_path, paths.root),
            "treatment_path": _repo_relative_text(treatment_path, paths.root),
            "regime_estimates_path": _repo_relative_text(regime_estimates_path, paths.root),
            "treatment_estimates_path": _repo_relative_text(treatment_estimates_path, paths.root),
            "factor_meta_path": _repo_relative_text(factor_meta_path, paths.root),
            "factor_loadings_path": _repo_relative_text(factor_loadings_path, paths.root),
            "recommended_k": recommended_k,
            "recommended_k_reason": recommended_k_reason,
            "control_universe_feature_count": control_universe.feature_count,
            "screened_feature_count": len(screened),
            "base_controls": base_controls,
            "recommended_factor_ids": recommended_factor_ids,
            "recommended_factor_count": len(recommended_factor_ids),
            "k_grid": k_values,
            "regime_filters_run": [row["regime_id"] for row in regime_rows],
            "treatment_variants_run": [row["treatment_variant"] for row in treatment_rows],
        },
    )

    return QuarterlyRobustnessResult(
        summary_path=summary_path,
        ladder_path=ladder_path,
        regime_path=regime_path,
        treatment_path=treatment_path,
        control_meta_path=control_universe.meta_path,
    )
