from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ea_tdc.designs.events import build_event_design
from ea_tdc.designs.quarterly import build_quarterly_design
from ea_tdc.estimation import build_estimation_snapshot
from ea_tdc.ml_extensions import build_negative_control_mining, build_quarterly_dml, build_quarterly_forest, build_quarterly_tmle
from ea_tdc.paths import ProjectPaths
from ea_tdc.robustness import build_control_universe, build_quarterly_robustness
from ea_tdc.utils import utc_now_iso, write_json


@dataclass(frozen=True)
class ReleaseSnapshotResult:
    summary_path: Path
    summary_csv_path: Path
    jobs_built: int
    ready_jobs: int
    partial_jobs: int


@dataclass(frozen=True)
class ReleaseScorecardResult:
    summary_path: Path
    summary_csv_path: Path
    public_jobs: int
    committed_public_jobs: int
    ready_jobs: int
    estimated_jobs: int


@dataclass(frozen=True)
class ReleaseContractResult:
    summary_path: Path
    summary_csv_path: Path
    active_jobs: int
    main_candidates: int
    appendix_candidates: int
    exploratory_sidecar_jobs: int
    deferred_jobs: int
    blocked_jobs: int


@dataclass(frozen=True)
class ReleaseArtifactContractResult:
    summary_path: Path
    summary_csv_path: Path
    committed_jobs: int
    main_text_artifacts: int
    appendix_artifacts: int


@dataclass(frozen=True)
class RobustnessSnapshotResult:
    summary_path: Path
    summary_csv_path: Path
    jobs_summarized: int
    feature_count: int
    series_count: int


@dataclass(frozen=True)
class EventSidecarScreeningResult:
    summary_path: Path
    summary_csv_path: Path
    signal_count: int
    jobs_summarized: int


@dataclass(frozen=True)
class EventSidecarArtifactPackResult:
    summary_path: Path
    rates_csv_path: Path
    plumbing_csv_path: Path
    manifest_path: Path
    signal_count: int


@dataclass(frozen=True)
class StageCompletionCloseoutResult:
    summary_path: Path
    manifest_path: Path


def _load_jobs(config_path: Path, *, track_field: str | None = "track_in_release_snapshot") -> list[dict[str, Any]]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    jobs = payload.get("jobs", [])
    if not isinstance(jobs, list):
        raise TypeError("Expected 'jobs' list in dass blueprint")
    filtered: list[dict[str, Any]] = []
    for item in jobs:
        if not isinstance(item, dict) or not item.get("job_id"):
            continue
        if track_field is not None and not bool(item.get(track_field, True)):
            continue
        filtered.append(item)
    return filtered


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_int(value: Any) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def _safe_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _robustness_series_count(control_universe_meta: dict[str, Any]) -> int:
    counts = control_universe_meta.get("series_count_by_frequency", {}) or {}
    return sum(_safe_int(value) for value in counts.values())


def _repo_relative_str(value: Any, repo_root: Path) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return text


def _resolve_repo_path(value: Any, repo_root: Path) -> Path:
    text = str(value or "").strip()
    if not text:
        return Path()
    path = Path(text)
    if path.is_absolute():
        return path
    return repo_root / path


def _build_robustness_job_row(
    *,
    repo_root: Path,
    job: dict[str, Any],
    summary: dict[str, Any],
    ladder_rows: list[dict[str, str]],
    treatment_rows: list[dict[str, str]],
    regime_rows: list[dict[str, str]],
    factor_loadings_rows: list[dict[str, str]],
    dml_summary: dict[str, Any] | None = None,
    tmle_summary: dict[str, Any] | None = None,
    forest_summary: dict[str, Any] | None = None,
    negative_control_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dml_rows_written = _safe_int((dml_summary or {}).get("rows_written"))
    tmle_rows_written = _safe_int((tmle_summary or {}).get("rows_written"))
    forest_rows_written = _safe_int((forest_summary or {}).get("rows_written"))
    max_lead_placebo_abs_z = max(
        [
            _safe_float(item.get("max_abs_z"))
            for item in ((negative_control_summary or {}).get("lead_placebos", []) or [])
            if _safe_float(item.get("max_abs_z")) is not None
        ]
        or [0.0]
    )
    ml_public_branch = "none"
    ml_public_branch_label = "No ML branch built"
    ml_public_branch_reason = "No ML robustness estimates are available for this job yet."
    if dml_rows_written > 0:
        ml_public_branch = "dml"
        ml_public_branch_label = "DML"
        ml_public_branch_reason = (
            "Quoted ML branch uses cross-fitted ridge DML because it is the most stable live learner. "
            "Forest remains the conservative cross-check; TMLE remains experimental."
        )
    elif forest_rows_written > 0:
        ml_public_branch = "forest"
        ml_public_branch_label = "Forest"
        ml_public_branch_reason = (
            "Quoted ML branch falls back to the forest learner because DML is unavailable for this job."
        )
    elif tmle_rows_written > 0:
        ml_public_branch = "tmle"
        ml_public_branch_label = "TMLE"
        ml_public_branch_reason = (
            "Quoted ML branch falls back to TMLE because the other learners are unavailable for this job."
        )
    negative_control_signal = "quiet"
    negative_control_signal_label = "Placebo leads are quiet"
    if max_lead_placebo_abs_z >= 3.0:
        negative_control_signal = "cautionary"
        negative_control_signal_label = "Placebo leads are still cautionary"
    elif max_lead_placebo_abs_z >= 2.0:
        negative_control_signal = "mixed"
        negative_control_signal_label = "Placebo leads are mixed"
    recommended_k = _safe_int(summary.get("recommended_k"))
    recommended_k_reason = str(summary.get("recommended_k_reason", "")).strip()
    recommended_factor_ids = [str(item) for item in summary.get("recommended_factor_ids", [])]
    baseline_row = next((row for row in ladder_rows if str(row.get("run_type", "")).strip() == "baseline_core"), {})
    recommended_row = next((row for row in ladder_rows if _safe_int(row.get("k_screened")) == recommended_k), {})
    top_loadings = [
        {
            "factor_id": str(item.get("factor_id", "")).strip(),
            "feature_id": str(item.get("feature_id", "")).strip(),
            "loading_abs": _safe_float(item.get("loading_abs"))
            if _safe_float(item.get("loading_abs")) is not None
            else _safe_float(item.get("loading")),
        }
        for item in factor_loadings_rows[:8]
        if str(item.get("feature_id", "")).strip()
    ]
    return {
        "job_id": str(job.get("job_id", "")).strip(),
        "estimator": str(job.get("estimator", "")).strip(),
        "output_family": str(job.get("output_family", "")).strip(),
        "release1_scope": str(job.get("release1_scope", "")).strip() or "committed",
        "recommended_k": recommended_k,
        "recommended_k_reason": recommended_k_reason,
        "recommended_factor_count": len(recommended_factor_ids),
        "screened_feature_count": _safe_int(summary.get("screened_feature_count")),
        "control_universe_feature_count": _safe_int(summary.get("control_universe_feature_count")),
        "regime_filter_count": len(summary.get("regime_filters_run", []) or []),
        "treatment_variant_count": len(summary.get("treatment_variants_run", []) or []),
        "baseline_avg_abs_beta": _safe_float(baseline_row.get("avg_abs_beta")),
        "recommended_avg_abs_beta": _safe_float(recommended_row.get("avg_abs_beta")),
        "ladder_rows": [
            {
                "run_type": str(row.get("run_type", "")).strip(),
                "label": (
                    "Baseline macro block"
                    if str(row.get("run_type", "")).strip() == "baseline_core"
                    else f"K={_safe_int(row.get('k_screened'))} -> {_safe_int(row.get('factor_count'))} factors"
                ),
                "k_screened": _safe_int(row.get("k_screened")),
                "factor_count": _safe_int(row.get("factor_count")),
                "avg_abs_beta": _safe_float(row.get("avg_abs_beta")),
                "rows_written": _safe_int(row.get("rows_written")),
                "warning_rows": _safe_int(row.get("warning_rows")),
            }
            for row in ladder_rows
        ],
        "treatment_rows": [
            {
                "treatment_variant": str(row.get("treatment_variant", "")).strip(),
                "avg_abs_beta": _safe_float(row.get("avg_abs_beta")),
                "rows_written": _safe_int(row.get("rows_written")),
                "warning_rows": _safe_int(row.get("warning_rows")),
            }
            for row in treatment_rows
            if str(row.get("treatment_variant", "")).strip()
        ],
        "regime_rows": [
            {
                "regime_id": str(row.get("regime_id", "")).strip(),
                "rows_in_regime": _safe_int(row.get("rows_in_regime")),
                "avg_abs_beta": _safe_float(row.get("avg_abs_beta")),
                "rows_written": _safe_int(row.get("rows_written")),
            }
            for row in regime_rows
            if str(row.get("regime_id", "")).strip()
        ],
        "top_loadings": top_loadings,
        "dml": {
            "rows_written": dml_rows_written,
            "avg_nuisance_r2_outcome": _safe_float((dml_summary or {}).get("avg_nuisance_r2_outcome")),
            "avg_nuisance_r2_treatment": _safe_float((dml_summary or {}).get("avg_nuisance_r2_treatment")),
        },
        "tmle": {
            "rows_written": tmle_rows_written,
            "avg_nuisance_r2_outcome": _safe_float((tmle_summary or {}).get("avg_nuisance_r2_outcome")),
            "avg_nuisance_r2_treatment": _safe_float((tmle_summary or {}).get("avg_nuisance_r2_treatment")),
            "avg_tmle_epsilon": _safe_float((tmle_summary or {}).get("avg_tmle_epsilon")),
            "avg_tmle_theta_init": _safe_float((tmle_summary or {}).get("avg_tmle_theta_init")),
            "epsilon_clip_rate": _safe_float((tmle_summary or {}).get("epsilon_clip_rate")),
        },
        "forest": {
            "rows_written": forest_rows_written,
            "avg_nuisance_r2_outcome": _safe_float((forest_summary or {}).get("avg_nuisance_r2_outcome")),
            "avg_nuisance_r2_treatment": _safe_float((forest_summary or {}).get("avg_nuisance_r2_treatment")),
        },
        "negative_controls": {
            "candidate_control_branch": str((negative_control_summary or {}).get("candidate_control_branch", "")).strip(),
            "candidate_count": len((negative_control_summary or {}).get("top_clean_candidates", []) or []),
            "lead_placebo_count": len((negative_control_summary or {}).get("lead_placebos", []) or []),
            "max_lead_placebo_abs_z": max_lead_placebo_abs_z,
            "signal": negative_control_signal,
            "signal_label": negative_control_signal_label,
        },
        "ml_public_branch": ml_public_branch,
        "ml_public_branch_label": ml_public_branch_label,
        "ml_public_branch_reason": ml_public_branch_reason,
        "links": {
            "summary_path": _repo_relative_str(job.get("summary_path", ""), repo_root),
            "ladder_path": _repo_relative_str(summary.get("ladder_path", ""), repo_root),
            "treatment_path": _repo_relative_str(summary.get("treatment_path", ""), repo_root),
            "regime_path": _repo_relative_str(summary.get("regime_path", ""), repo_root),
            "factor_meta_path": _repo_relative_str(summary.get("factor_meta_path", ""), repo_root),
            "factor_loadings_path": _repo_relative_str(summary.get("factor_loadings_path", ""), repo_root),
            "control_screen_path": _repo_relative_str(summary.get("control_screen_path", ""), repo_root),
            "dml_summary_path": _repo_relative_str((dml_summary or {}).get("summary_path", ""), repo_root),
            "dml_estimates_path": _repo_relative_str((dml_summary or {}).get("estimates_path", ""), repo_root),
            "tmle_summary_path": _repo_relative_str((tmle_summary or {}).get("summary_path", ""), repo_root),
            "tmle_estimates_path": _repo_relative_str((tmle_summary or {}).get("estimates_path", ""), repo_root),
            "forest_summary_path": _repo_relative_str((forest_summary or {}).get("summary_path", ""), repo_root),
            "forest_estimates_path": _repo_relative_str((forest_summary or {}).get("estimates_path", ""), repo_root),
            "negative_control_path": _repo_relative_str((negative_control_summary or {}).get("summary_path", ""), repo_root),
            "negative_control_csv_path": _repo_relative_str((negative_control_summary or {}).get("summary_csv_path", ""), repo_root),
        },
    }


def _summarize_estimation(job_id: str, estimation_row: dict[str, Any] | None, estimation_summary: dict[str, Any]) -> dict[str, Any]:
    rows_written = int(str((estimation_row or {}).get("rows_written", "0") or "0"))
    estimation_status = "estimated" if rows_written > 0 else ("estimated_empty" if estimation_row else "not_estimated")
    covariance_estimators_used = [str(item) for item in estimation_summary.get("covariance_estimators_used", [])]
    warning_rows = int(estimation_summary.get("warning_rows", 0) or 0)
    weak_instrument_rows = int(estimation_summary.get("weak_instrument_rows", 0) or 0)
    adaptive_control_rows = int(estimation_summary.get("adaptive_control_rows", 0) or 0)
    small_sample_rows = int(estimation_summary.get("small_sample_rows", 0) or 0)
    publication_risk_flags: list[str] = []
    if weak_instrument_rows > 0:
        publication_risk_flags.append("weak_instrument")
    if adaptive_control_rows > 0:
        publication_risk_flags.append("adaptive_controls")
    if small_sample_rows > 0:
        publication_risk_flags.append("small_sample")
    return {
        "job_id": job_id,
        "rows_written": rows_written,
        "estimation_status": estimation_status,
        "covariance_estimators_used": covariance_estimators_used,
        "warning_rows": warning_rows,
        "weak_instrument_rows": weak_instrument_rows,
        "adaptive_control_rows": adaptive_control_rows,
        "small_sample_rows": small_sample_rows,
        "publication_risk_flags": publication_risk_flags,
        "min_observations": int(estimation_summary.get("min_observations", 0) or 0),
        "max_observations": int(estimation_summary.get("max_observations", 0) or 0),
    }


def _classify_release_contract_row(
    *,
    is_public_job: bool,
    release1_scope: str,
    output_family: str,
    sample_policy: str,
    readiness_status: str,
    estimated_rows_written: int,
    publication_risk_flags: list[str],
) -> tuple[str, str, str]:
    if release1_scope == "deferred":
        if "weak_instrument" in publication_risk_flags:
            return ("deferred_development", "deferred", "deferred_weak_instrument")
        if "small_sample" in publication_risk_flags or "adaptive_controls" in publication_risk_flags:
            return ("deferred_development", "deferred", "deferred_thin_headline_event_sample")
        if readiness_status != "ready_for_estimation":
            return ("deferred_development", "deferred", "deferred_design_not_ready")
        if estimated_rows_written <= 0:
            return ("deferred_development", "deferred", "deferred_not_estimated")
        return ("deferred_development", "deferred", "deferred_by_release_policy")
    if readiness_status != "ready_for_estimation":
        return ("blocked", "blocked", "design_not_ready")
    if estimated_rows_written <= 0:
        return ("blocked", "blocked", "not_estimated")
    if output_family == "supporting_descriptive" or not is_public_job:
        return ("exploratory_sidecar", "sidecar", "descriptive_or_exploratory_sample")
    if "weak_instrument" in publication_risk_flags:
        return ("blocked", "blocked", "weak_instrument")
    if "small_sample" in publication_risk_flags or "adaptive_controls" in publication_risk_flags:
        return ("blocked", "blocked", "headline_event_sample_too_thin")
    if output_family == "headline_identified":
        return ("release1_main_candidate", "main_text", "clean_headline_identified")
    if output_family == "supporting_reduced_form":
        return ("release1_appendix_candidate", "appendix", "clean_supporting_reduced_form")
    return ("release1_appendix_candidate", "appendix", "clean_supported_job")


def _artifact_title(job_id: str, estimator: str, tier: str) -> str:
    base = job_id.replace("_", " ")
    if tier == "release1_main_candidate" and estimator == "lp":
        return f"Main impulse response: {base}"
    if tier == "release1_main_candidate" and estimator == "lp_iv":
        return f"Main IV impulse response: {base}"
    if estimator == "event_lp":
        return f"Event-study support: {base}"
    return f"Supporting result: {base}"


def _artifact_rows_for_job(
    *,
    artifact_index: int,
    contract_row: dict[str, Any],
    design_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    tier = str(contract_row.get("contract_tier", "")).strip()
    if tier not in {"release1_main_candidate", "release1_appendix_candidate"}:
        return []
    estimator = str(contract_row.get("estimator", "")).strip()
    output_family = str(contract_row.get("output_family", "")).strip()
    job_id = str(contract_row.get("job_id", "")).strip()
    outcome_ids = [str(item) for item in design_manifest.get("outcome_ids", [])]
    horizon_grid = [str(item) for item in design_manifest.get("horizon_grid", [])]
    outcomes_text = ",".join(outcome_ids)
    horizons_text = ",".join(horizon_grid)
    rows: list[dict[str, Any]] = []
    if tier == "release1_main_candidate":
        rows.append(
            {
                "artifact_id": f"main_figure_{artifact_index}",
                "artifact_kind": "figure",
                "release_channel": "main_text",
                "job_id": job_id,
                "estimator": estimator,
                "output_family": output_family,
                "title": _artifact_title(job_id, estimator, tier),
                "display_spec": "impulse_response_grid",
                "outcome_ids": outcomes_text,
                "horizons": horizons_text,
                "contract_source": "release1_main_candidate",
                "status": "ready",
            }
        )
        rows.append(
            {
                "artifact_id": f"main_table_{artifact_index}",
                "artifact_kind": "table",
                "release_channel": "main_text",
                "job_id": job_id,
                "estimator": estimator,
                "output_family": output_family,
                "title": f"Main coefficient table: {job_id.replace('_', ' ')}",
                "display_spec": "coefficient_table",
                "outcome_ids": outcomes_text,
                "horizons": horizons_text,
                "contract_source": "release1_main_candidate",
                "status": "ready",
            }
        )
        return rows
    rows.append(
        {
            "artifact_id": f"appendix_table_{artifact_index}",
            "artifact_kind": "table",
            "release_channel": "appendix",
            "job_id": job_id,
            "estimator": estimator,
            "output_family": output_family,
            "title": f"Appendix support table: {job_id.replace('_', ' ')}",
            "display_spec": "supporting_table",
            "outcome_ids": outcomes_text,
            "horizons": horizons_text,
            "contract_source": "release1_appendix_candidate",
            "status": "ready",
        }
    )
    return rows


def build_release_snapshot(paths: ProjectPaths) -> ReleaseSnapshotResult:
    jobs = _load_jobs(paths.config / "dass_job_blueprint.yaml")
    rows: list[dict[str, Any]] = []

    for job in jobs:
        job_id = str(job["job_id"])
        estimator = str(job.get("estimator", "")).strip()
        if estimator == "event_lp":
            built = build_event_design(paths, job_id=job_id)
            diagnostics_path = None
        else:
            built = build_quarterly_design(paths, job_id=job_id)
            diagnostics_path = built.diagnostics_manifest_path

        design_manifest = _read_json(built.design_manifest_path)
        sample_manifest = _read_json(built.sample_manifest_path)
        rows.append(
            {
                "job_id": job_id,
                "estimator": estimator,
                "freq": str(job.get("freq", "")).strip(),
                "release1_scope": str(job.get("release1_scope", "")).strip(),
                "status": str(design_manifest.get("status", "")).strip(),
                "usable_rows": int(design_manifest.get("usable_rows", 0) or 0),
                "sample_policy": str(design_manifest.get("sample_policy", "")).strip(),
                "requested_sample_rows": int(
                    ((design_manifest.get("event_sample_counts", {}) or {}).get("requested_sample_rows", 0) or 0)
                ),
                "headline_eligible_rows": int(
                    ((design_manifest.get("event_sample_counts", {}) or {}).get("headline_eligible_rows", 0) or 0)
                ),
                "reviewed_nonmissing_rows": int(
                    ((design_manifest.get("event_sample_counts", {}) or {}).get("reviewed_nonmissing_rows", 0) or 0)
                ),
                "missing_required_series_count": len(design_manifest.get("missing_required_series", [])),
                "missing_state_ids_count": len(design_manifest.get("missing_state_ids", [])),
                "missing_instrument_ids_count": len(design_manifest.get("missing_instrument_ids", [])),
                "bundle_path": str(built.bundle_path),
                "design_manifest_path": str(built.design_manifest_path),
                "sample_manifest_path": str(built.sample_manifest_path),
                "diagnostics_manifest_path": str(diagnostics_path) if diagnostics_path else "",
                "final_sample_observations": int((sample_manifest.get("rows") or [{}])[-1].get("observations_remaining", 0) or 0),
            }
        )

    ready_jobs = sum(1 for row in rows if row["status"] == "ready_for_estimation")
    partial_jobs = sum(1 for row in rows if row["status"] != "ready_for_estimation")
    summary = {
        "generated_at": utc_now_iso(),
        "jobs_built": len(rows),
        "ready_jobs": ready_jobs,
        "partial_jobs": partial_jobs,
        "rows": rows,
    }

    summary_path = paths.reports / "release_snapshot.json"
    summary_csv_path = paths.reports / "release_snapshot.csv"
    write_json(summary_path, summary)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["job_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return ReleaseSnapshotResult(
        summary_path=summary_path,
        summary_csv_path=summary_csv_path,
        jobs_built=len(rows),
        ready_jobs=ready_jobs,
        partial_jobs=partial_jobs,
    )


def build_release_scorecard(paths: ProjectPaths) -> ReleaseScorecardResult:
    release_snapshot = build_release_snapshot(paths)
    estimation_snapshot = build_estimation_snapshot(paths)
    release_rows = _read_json(release_snapshot.summary_path).get("rows", [])
    estimation_rows = _read_json(estimation_snapshot.summary_path).get("rows", [])
    estimation_by_job = {
        str(row.get("job_id", "")).strip(): row
        for row in estimation_rows
        if str(row.get("job_id", "")).strip()
    }

    rows: list[dict[str, Any]] = []
    estimated_jobs = 0
    committed_public_jobs = 0
    deferred_public_jobs = 0
    jobs_with_warnings = 0
    jobs_with_weak_instruments = 0
    jobs_with_small_sample_flags = 0
    jobs_with_adaptive_controls = 0
    for release_row in release_rows:
        job_id = str(release_row.get("job_id", "")).strip()
        estimation_row = estimation_by_job.get(job_id)
        estimation_summary_path = paths.manifests / f"{job_id}__estimation_summary.json"
        estimation_summary = _read_json(estimation_summary_path) if estimation_summary_path.exists() else {}
        estimation_metrics = _summarize_estimation(job_id, estimation_row, estimation_summary)
        rows_written = int(estimation_metrics["rows_written"])
        estimation_status = str(estimation_metrics["estimation_status"])
        if rows_written > 0:
            estimated_jobs += 1
        covariance_estimators_used = [str(item) for item in estimation_metrics["covariance_estimators_used"]]
        warning_rows = int(estimation_metrics["warning_rows"])
        weak_instrument_rows = int(estimation_metrics["weak_instrument_rows"])
        adaptive_control_rows = int(estimation_metrics["adaptive_control_rows"])
        small_sample_rows = int(estimation_metrics["small_sample_rows"])
        publication_risk_flags = [str(item) for item in estimation_metrics["publication_risk_flags"]]
        release1_scope = str(release_row.get("release1_scope", "")).strip() or "committed"
        if release1_scope == "deferred":
            deferred_public_jobs += 1
        else:
            committed_public_jobs += 1
        if weak_instrument_rows > 0:
            jobs_with_weak_instruments += 1
        if adaptive_control_rows > 0:
            jobs_with_adaptive_controls += 1
        if small_sample_rows > 0:
            jobs_with_small_sample_flags += 1
        if warning_rows > 0:
            jobs_with_warnings += 1
        rows.append(
            {
                "job_id": job_id,
                "estimator": str(release_row.get("estimator", "")).strip(),
                "freq": str(release_row.get("freq", "")).strip(),
                "release1_scope": release1_scope,
                "readiness_status": str(release_row.get("status", "")).strip(),
                "final_sample_observations": int(
                    estimation_metrics["max_observations"]
                    or release_row.get("final_sample_observations", 0)
                    or 0
                ),
                "usable_rows": int(release_row.get("usable_rows", 0) or 0),
                "estimation_status": estimation_status,
                "estimated_rows_written": rows_written,
                "estimates_path": str((estimation_row or {}).get("estimates_path", "")),
                "comparison_path": str((estimation_row or {}).get("comparison_path", "")),
                "sample_policy": str(release_row.get("sample_policy", "")).strip(),
                "requested_sample_rows": int(release_row.get("requested_sample_rows", 0) or 0),
                "headline_eligible_rows": int(release_row.get("headline_eligible_rows", 0) or 0),
                "reviewed_nonmissing_rows": int(release_row.get("reviewed_nonmissing_rows", 0) or 0),
                "min_observations": int(estimation_metrics["min_observations"] or 0),
                "max_observations": int(estimation_metrics["max_observations"] or 0),
                "covariance_estimators_used": ",".join(covariance_estimators_used),
                "warning_rows": warning_rows,
                "weak_instrument_rows": weak_instrument_rows,
                "adaptive_control_rows": adaptive_control_rows,
                "small_sample_rows": small_sample_rows,
                "publication_risk_flags": ",".join(publication_risk_flags),
            }
        )

    summary = {
        "generated_at": utc_now_iso(),
        "public_jobs": len(rows),
        "committed_public_jobs": committed_public_jobs,
        "deferred_public_jobs": deferred_public_jobs,
        "ready_jobs": sum(1 for row in rows if row["readiness_status"] == "ready_for_estimation"),
        "estimated_jobs": estimated_jobs,
        "jobs_with_warnings": jobs_with_warnings,
        "jobs_with_weak_instruments": jobs_with_weak_instruments,
        "jobs_with_small_sample_flags": jobs_with_small_sample_flags,
        "jobs_with_adaptive_controls": jobs_with_adaptive_controls,
        "ready_without_estimates": sum(
            1
            for row in rows
            if row["readiness_status"] == "ready_for_estimation" and row["estimated_rows_written"] == 0
        ),
        "rows": rows,
    }

    summary_path = paths.reports / "release_scorecard.json"
    summary_csv_path = paths.reports / "release_scorecard.csv"
    write_json(summary_path, summary)
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else [
            "job_id",
            "estimator",
            "freq",
            "readiness_status",
            "estimation_status",
            "estimated_rows_written",
            "estimates_path",
            "comparison_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return ReleaseScorecardResult(
        summary_path=summary_path,
        summary_csv_path=summary_csv_path,
        public_jobs=len(rows),
        committed_public_jobs=committed_public_jobs,
        ready_jobs=summary["ready_jobs"],
        estimated_jobs=estimated_jobs,
    )


def build_release_contract(paths: ProjectPaths) -> ReleaseContractResult:
    config_path = paths.config / "dass_job_blueprint.yaml"
    public_jobs = _load_jobs(config_path, track_field="track_in_release_snapshot")
    active_jobs = [
        job
        for job in _load_jobs(config_path, track_field=None)
        if bool(job.get("track_in_release_snapshot", True)) or bool(job.get("track_in_estimation_snapshot", True))
    ]
    for job in active_jobs:
        job_id = str(job["job_id"]).strip()
        estimator = str(job.get("estimator", "")).strip()
        if estimator == "event_lp":
            build_event_design(paths, job_id=job_id)
        else:
            build_quarterly_design(paths, job_id=job_id)
    build_release_scorecard(paths)
    estimation_snapshot = build_estimation_snapshot(paths)
    public_job_ids = {str(job["job_id"]).strip() for job in public_jobs}
    active_job_by_id = {str(job["job_id"]).strip(): job for job in active_jobs}
    estimation_rows = _read_json(estimation_snapshot.summary_path).get("rows", [])
    estimation_by_job = {
        str(row.get("job_id", "")).strip(): row
        for row in estimation_rows
        if str(row.get("job_id", "")).strip()
    }

    rows: list[dict[str, Any]] = []
    main_candidates = 0
    appendix_candidates = 0
    exploratory_sidecar_jobs = 0
    deferred_jobs = 0
    blocked_jobs = 0
    for job_id, job in active_job_by_id.items():
        design_manifest_path = paths.manifests / f"{job_id}__design_manifest.json"
        design_manifest = _read_json(design_manifest_path) if design_manifest_path.exists() else {}
        estimation_summary_path = paths.manifests / f"{job_id}__estimation_summary.json"
        estimation_summary = _read_json(estimation_summary_path) if estimation_summary_path.exists() else {}
        estimation_row = estimation_by_job.get(job_id)
        estimation_metrics = _summarize_estimation(job_id, estimation_row, estimation_summary)
        sample_policy = str(design_manifest.get("sample_policy", "")).strip()
        output_family = str(job.get("output_family", "")).strip()
        release1_scope = str(job.get("release1_scope", "")).strip() or "committed"
        contract_tier, release_channel, contract_reason = _classify_release_contract_row(
            is_public_job=job_id in public_job_ids,
            release1_scope=release1_scope,
            output_family=output_family,
            sample_policy=sample_policy,
            readiness_status=str(design_manifest.get("status", "")).strip(),
            estimated_rows_written=int(estimation_metrics["rows_written"]),
            publication_risk_flags=[str(item) for item in estimation_metrics["publication_risk_flags"]],
        )
        if contract_tier == "release1_main_candidate":
            main_candidates += 1
        elif contract_tier == "release1_appendix_candidate":
            appendix_candidates += 1
        elif contract_tier == "exploratory_sidecar":
            exploratory_sidecar_jobs += 1
        elif contract_tier == "deferred_development":
            deferred_jobs += 1
        else:
            blocked_jobs += 1
        rows.append(
            {
                "job_id": job_id,
                "estimator": str(job.get("estimator", "")).strip(),
                "output_family": output_family,
                "is_public_job": job_id in public_job_ids,
                "release1_scope": release1_scope,
                "sample_policy": sample_policy,
                "readiness_status": str(design_manifest.get("status", "")).strip(),
                "estimated_rows_written": int(estimation_metrics["rows_written"]),
                "publication_risk_flags": ",".join([str(item) for item in estimation_metrics["publication_risk_flags"]]),
                "warning_rows": int(estimation_metrics["warning_rows"]),
                "weak_instrument_rows": int(estimation_metrics["weak_instrument_rows"]),
                "adaptive_control_rows": int(estimation_metrics["adaptive_control_rows"]),
                "small_sample_rows": int(estimation_metrics["small_sample_rows"]),
                "final_sample_observations": int(
                    estimation_metrics["max_observations"]
                    or design_manifest.get("usable_rows", 0)
                    or 0
                ),
                "release_channel": release_channel,
                "contract_tier": contract_tier,
                "contract_reason": contract_reason,
                "design_manifest_path": str(design_manifest_path),
                "estimation_summary_path": str(estimation_summary_path) if estimation_summary_path.exists() else "",
            }
        )

    rows.sort(key=lambda row: (str(row["contract_tier"]), str(row["job_id"])))
    summary = {
        "generated_at": utc_now_iso(),
        "active_jobs": len(rows),
        "main_candidates": main_candidates,
        "appendix_candidates": appendix_candidates,
        "exploratory_sidecar_jobs": exploratory_sidecar_jobs,
        "deferred_jobs": deferred_jobs,
        "blocked_jobs": blocked_jobs,
        "rows": rows,
    }
    summary_path = paths.reports / "release_contract.json"
    summary_csv_path = paths.reports / "release_contract.csv"
    write_json(summary_path, summary)
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else [
            "job_id",
            "estimator",
            "output_family",
            "is_public_job",
            "sample_policy",
            "readiness_status",
            "estimated_rows_written",
            "publication_risk_flags",
            "warning_rows",
            "weak_instrument_rows",
            "adaptive_control_rows",
            "small_sample_rows",
            "final_sample_observations",
            "release_channel",
            "contract_tier",
            "contract_reason",
            "design_manifest_path",
            "estimation_summary_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return ReleaseContractResult(
        summary_path=summary_path,
        summary_csv_path=summary_csv_path,
        active_jobs=len(rows),
        main_candidates=main_candidates,
        appendix_candidates=appendix_candidates,
        exploratory_sidecar_jobs=exploratory_sidecar_jobs,
        deferred_jobs=deferred_jobs,
        blocked_jobs=blocked_jobs,
    )


def build_release_artifact_contract(paths: ProjectPaths) -> ReleaseArtifactContractResult:
    contract = build_release_contract(paths)
    contract_rows = _read_json(contract.summary_path).get("rows", [])
    artifact_rows: list[dict[str, Any]] = []
    committed_jobs = 0
    main_job_index = 0
    appendix_job_index = 0
    for contract_row in contract_rows:
        tier = str(contract_row.get("contract_tier", "")).strip()
        if tier not in {"release1_main_candidate", "release1_appendix_candidate"}:
            continue
        committed_jobs += 1
        job_id = str(contract_row.get("job_id", "")).strip()
        design_manifest_path = Path(str(contract_row.get("design_manifest_path", "")).strip())
        design_manifest = _read_json(design_manifest_path) if design_manifest_path.exists() else {}
        if tier == "release1_main_candidate":
            main_job_index += 1
            artifact_rows.extend(
                _artifact_rows_for_job(
                    artifact_index=main_job_index,
                    contract_row=contract_row,
                    design_manifest=design_manifest,
                )
            )
        else:
            appendix_job_index += 1
            artifact_rows.extend(
                _artifact_rows_for_job(
                    artifact_index=appendix_job_index,
                    contract_row=contract_row,
                    design_manifest=design_manifest,
                )
            )

    main_text_artifacts = sum(1 for row in artifact_rows if row["release_channel"] == "main_text")
    appendix_artifacts = sum(1 for row in artifact_rows if row["release_channel"] == "appendix")
    summary = {
        "generated_at": utc_now_iso(),
        "committed_jobs": committed_jobs,
        "main_text_artifacts": main_text_artifacts,
        "appendix_artifacts": appendix_artifacts,
        "rows": artifact_rows,
    }
    summary_path = paths.reports / "release_artifact_contract.json"
    summary_csv_path = paths.reports / "release_artifact_contract.csv"
    write_json(summary_path, summary)
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(artifact_rows[0].keys()) if artifact_rows else [
            "artifact_id",
            "artifact_kind",
            "release_channel",
            "job_id",
            "estimator",
            "output_family",
            "title",
            "display_spec",
            "outcome_ids",
            "horizons",
            "contract_source",
            "status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(artifact_rows)
    return ReleaseArtifactContractResult(
        summary_path=summary_path,
        summary_csv_path=summary_csv_path,
        committed_jobs=committed_jobs,
        main_text_artifacts=main_text_artifacts,
        appendix_artifacts=appendix_artifacts,
    )


def _event_sidecar_lane_blocks(paths: ProjectPaths) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    models_dir = paths.output / "models"
    job_specs = [
        {
            "job_id": "qra_event_rates_63bd",
            "lane": "rates",
            "title": "Rates benchmark",
            "description": "Medium-horizon rate and term-spread responses on the reviewed event sample.",
        },
        {
            "job_id": "qra_event_risk_21bd",
            "lane": "risk_plumbing",
            "title": "Risk and plumbing benchmark",
            "description": "Short-horizon risk move plus later balance-sheet plumbing responses on the reviewed event sample.",
        },
    ]

    signal_rows: list[dict[str, Any]] = []
    lane_blocks: list[dict[str, Any]] = []
    jobs_summarized = 0
    for spec in job_specs:
        path = models_dir / f"{spec['job_id']}__event_lp_estimates.csv"
        if not path.exists():
            continue
        jobs_summarized += 1
        rows = _read_csv(path)
        signals = []
        for row in rows:
            p_value = _safe_float(row.get("p_value_normal"))
            if p_value is None or p_value >= 0.10:
                continue
            signal = {
                "job_id": spec["job_id"],
                "lane": spec["lane"],
                "outcome": str(row.get("outcome", "")).strip(),
                "horizon": _safe_int(row.get("horizon")),
                "beta": _safe_float(row.get("beta")),
                "p_value_normal": p_value,
                "n": _safe_int(row.get("n")),
                "warning_flags": str(row.get("warning_flags", "")).strip(),
            }
            signals.append(signal)
            signal_rows.append(signal)
        signals.sort(key=lambda item: (item["p_value_normal"], item["horizon"], item["outcome"]))
        lane_blocks.append(
            {
                "job_id": spec["job_id"],
                "lane": spec["lane"],
                "title": spec["title"],
                "description": spec["description"],
                "signals": signals,
            }
        )
    return signal_rows, lane_blocks, jobs_summarized


def build_event_sidecar_screening(paths: ProjectPaths) -> EventSidecarScreeningResult:
    signal_rows, lane_blocks, jobs_summarized = _event_sidecar_lane_blocks(paths)

    summary_csv_path = paths.reports / "event_sidecar_screening.csv"
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "job_id",
            "lane",
            "outcome",
            "horizon",
            "beta",
            "p_value_normal",
            "n",
            "warning_flags",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(signal_rows)

    lines = [
        "# Event Sidecar Screening",
        "",
        "## Verdict",
        "",
        "The event lane is the strongest remaining non-deposit research path in the repo,",
        "but it should stay a **sidecar / descriptive benchmark** because the reviewed sample is still only `14` events.",
        "",
    ]
    for block in lane_blocks:
        lines.extend(
            [
                f"## {block['title']}",
                "",
                block["description"],
                "",
            ]
        )
        if not block["signals"]:
            lines.append("- No `p < 0.10` signals in the current estimate file.")
        else:
            for signal in block["signals"]:
                lines.append(
                    f"- `{signal['outcome']}` at `h={signal['horizon']}`: "
                    f"`beta ≈ {signal['beta']:.6g}`, `p ≈ {signal['p_value_normal']:.4g}`"
                )
        lines.append("")
    lines.extend(
        [
            "## Recommended use",
            "",
            "- Keep the event lane as the next live benchmark surface after deposits.",
            "- Use it for rates and plumbing sidecar evidence, not for broad public causal claims.",
            "- If more work is done here, prioritize compact event artifacts or stronger small-sample inference rather than broader family expansion.",
            "",
        ]
    )
    summary_path = paths.reports / "event_sidecar_screening.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    return EventSidecarScreeningResult(
        summary_path=summary_path,
        summary_csv_path=summary_csv_path,
        signal_count=len(signal_rows),
        jobs_summarized=jobs_summarized,
    )


def build_event_sidecar_artifact_pack(paths: ProjectPaths) -> EventSidecarArtifactPackResult:
    screening = build_event_sidecar_screening(paths)
    signal_rows, lane_blocks, _ = _event_sidecar_lane_blocks(paths)

    rates_rows = [row for row in signal_rows if row["lane"] == "rates"]
    plumbing_rows = [row for row in signal_rows if row["lane"] == "risk_plumbing"]
    rates_csv_path = paths.reports / "event_sidecar_rates_table.csv"
    plumbing_csv_path = paths.reports / "event_sidecar_plumbing_table.csv"
    fieldnames = [
        "job_id",
        "lane",
        "outcome",
        "horizon",
        "beta",
        "p_value_normal",
        "n",
        "warning_flags",
    ]
    for target_path, rows in ((rates_csv_path, rates_rows), (plumbing_csv_path, plumbing_rows)):
        with target_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Event Sidecar Artifact Pack",
        "",
        "## Purpose",
        "",
        "This pack is the compact event-side surface for release closeout.",
        "Use it for sidecar rates and plumbing evidence, not for headline causal claims.",
        "",
    ]
    for block in lane_blocks:
        table_name = (
            "event_sidecar_rates_table.csv"
            if block["lane"] == "rates"
            else "event_sidecar_plumbing_table.csv"
        )
        lines.extend(
            [
                f"## {block['title']}",
                "",
                block["description"],
                "",
                f"Table export: `{table_name}`",
                "",
            ]
        )
        if not block["signals"]:
            lines.append("- No `p < 0.10` signals in the current estimate file.")
        else:
            for signal in block["signals"]:
                lines.append(
                    f"- `{signal['outcome']}` at `h={signal['horizon']}`: "
                    f"`beta ≈ {signal['beta']:.6g}`, `p ≈ {signal['p_value_normal']:.4g}`"
                )
        lines.append("")
    lines.extend(
        [
            "## Caption language",
            "",
            "- Rates/plumbing sidecar on reviewed `n=14` event sample.",
            "- Signals are useful for benchmark orientation, but inference remains fragile at this sample size.",
            "",
        ]
    )
    summary_path = paths.reports / "event_sidecar_artifact_pack.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    manifest_path = paths.manifests / "event_sidecar_artifact_pack_manifest.json"
    write_json(
        manifest_path,
        {
            "generated_at": utc_now_iso(),
            "summary_path": _repo_relative_str(summary_path, paths.root),
            "screening_path": _repo_relative_str(screening.summary_path, paths.root),
            "rates_csv_path": _repo_relative_str(rates_csv_path, paths.root),
            "plumbing_csv_path": _repo_relative_str(plumbing_csv_path, paths.root),
            "signal_count": len(signal_rows),
        },
    )
    return EventSidecarArtifactPackResult(
        summary_path=summary_path,
        rates_csv_path=rates_csv_path,
        plumbing_csv_path=plumbing_csv_path,
        manifest_path=manifest_path,
        signal_count=len(signal_rows),
    )


def build_stage_completion_closeout(paths: ProjectPaths) -> StageCompletionCloseoutResult:
    event_pack = build_event_sidecar_artifact_pack(paths)
    completion_path = paths.reports / "stage_completion_closeout.md"
    completion_path.write_text(
        "\n".join(
            [
                "# Stage Completion Closeout",
                "",
                "## Completed package",
                "",
                "- Headline lane: `baseline_tdc_lp_deposits` on the selected screened branch.",
                "- Mechanism lane: accounting alignment and closeout artifacts.",
                "- Sidecar lane: event rates and plumbing benchmark artifacts.",
                "",
                "## Deferred package",
                "",
                "- Macro-price confirmatory trees remain internal-only and paused.",
                "- IV remains deferred.",
                "- TMLE remains excluded from public evidence.",
                "",
                "## Recommended finalization step",
                "",
                "- Treat the repo as empirically scoped for completion.",
                "- Use the event sidecar artifact pack for any final site/report integration rather than reopening exploratory modeling.",
                "",
                f"Event sidecar artifact pack: `{_repo_relative_str(event_pack.summary_path, paths.root)}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = paths.manifests / "stage_completion_closeout_manifest.json"
    write_json(
        manifest_path,
        {
            "generated_at": utc_now_iso(),
            "summary_path": _repo_relative_str(completion_path, paths.root),
            "event_artifact_pack_path": _repo_relative_str(event_pack.summary_path, paths.root),
        },
    )
    return StageCompletionCloseoutResult(summary_path=completion_path, manifest_path=manifest_path)


def build_robustness_snapshot(
    paths: ProjectPaths,
    *,
    job_ids: list[str] | None = None,
) -> RobustnessSnapshotResult:
    config_path = paths.config / "dass_job_blueprint.yaml"
    active_jobs = _load_jobs(config_path, track_field=None)
    requested_job_ids = {str(item).strip() for item in (job_ids or []) if str(item).strip()}
    target_jobs = [
        job
        for job in active_jobs
        if str(job.get("freq", "")).strip() == "quarterly"
        and str(job.get("estimator", "")).strip() in {"lp", "lp_iv"}
        and (not requested_job_ids or str(job.get("job_id", "")).strip() in requested_job_ids)
    ]

    control_universe_meta_path = paths.reports / "control_universe_meta.json"
    control_universe_meta: dict[str, Any] = {}
    if target_jobs:
        if not control_universe_meta_path.exists():
            for job in target_jobs:
                build_quarterly_design(paths, job_id=str(job.get("job_id", "")).strip())
            build_control_universe(paths)
        control_universe_meta = _read_json(control_universe_meta_path)

    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for job in target_jobs:
        job_id = str(job.get("job_id", "")).strip()
        summary_path = paths.manifests / f"{job_id}__robustness_summary.json"
        if not summary_path.exists():
            build_quarterly_robustness(paths, job_id=job_id)
        if not summary_path.exists():
            continue
        robustness_summary = _read_json(summary_path)
        ladder_rows = _read_csv(_resolve_repo_path(robustness_summary.get("ladder_path", ""), paths.root))
        treatment_rows = _read_csv(_resolve_repo_path(robustness_summary.get("treatment_path", ""), paths.root))
        regime_rows = _read_csv(_resolve_repo_path(robustness_summary.get("regime_path", ""), paths.root))
        factor_loadings_path = _resolve_repo_path(robustness_summary.get("factor_loadings_path", ""), paths.root)
        factor_loadings_rows = _read_csv(factor_loadings_path) if factor_loadings_path.exists() else []
        dml_summary: dict[str, Any] | None = None
        tmle_summary: dict[str, Any] | None = None
        forest_summary: dict[str, Any] | None = None
        negative_control_summary: dict[str, Any] | None = None
        if str(job.get("estimator", "")).strip() == "lp":
            dml_summary_path = paths.manifests / f"{job_id}__dml_summary.json"
            if not dml_summary_path.exists():
                try:
                    build_quarterly_dml(paths, job_id=job_id)
                except Exception:
                    dml_summary = None
            if dml_summary_path.exists():
                dml_summary = _read_json(dml_summary_path)
                dml_summary["summary_path"] = str(dml_summary_path)
            tmle_summary_path = paths.manifests / f"{job_id}__tmle_summary.json"
            if not tmle_summary_path.exists():
                try:
                    build_quarterly_tmle(paths, job_id=job_id)
                except Exception:
                    tmle_summary = None
            if tmle_summary_path.exists():
                tmle_summary = _read_json(tmle_summary_path)
                tmle_summary["summary_path"] = str(tmle_summary_path)
            forest_summary_path = paths.manifests / f"{job_id}__forest_summary.json"
            if not forest_summary_path.exists():
                try:
                    build_quarterly_forest(paths, job_id=job_id)
                except Exception:
                    forest_summary = None
            if forest_summary_path.exists():
                forest_summary = _read_json(forest_summary_path)
                forest_summary["summary_path"] = str(forest_summary_path)
            negative_control_path = paths.reports / f"{job_id}__negative_control_mining.json"
            if not negative_control_path.exists():
                try:
                    build_negative_control_mining(paths, job_id=job_id)
                except Exception:
                    negative_control_summary = None
            if negative_control_path.exists():
                negative_control_summary = _read_json(negative_control_path)
                negative_control_summary["summary_path"] = str(negative_control_path)
                negative_control_summary["summary_csv_path"] = str(paths.reports / f"{job_id}__negative_control_mining.csv")
        detail_row = _build_robustness_job_row(
            repo_root=paths.root,
            job={**job, "summary_path": summary_path},
            summary=robustness_summary,
            ladder_rows=ladder_rows,
            treatment_rows=treatment_rows,
            regime_rows=regime_rows,
            factor_loadings_rows=factor_loadings_rows,
            dml_summary=dml_summary,
            tmle_summary=tmle_summary,
            forest_summary=forest_summary,
            negative_control_summary=negative_control_summary,
        )
        rows.append(detail_row)
        summary_rows.append(
            {
                "job_id": detail_row["job_id"],
                "estimator": detail_row["estimator"],
                "output_family": detail_row["output_family"],
                "release1_scope": detail_row["release1_scope"],
                "recommended_k": detail_row["recommended_k"],
                "recommended_k_reason": detail_row["recommended_k_reason"],
                "recommended_factor_count": detail_row["recommended_factor_count"],
                "screened_feature_count": detail_row["screened_feature_count"],
                "control_universe_feature_count": detail_row["control_universe_feature_count"],
                "regime_filter_count": detail_row["regime_filter_count"],
                "treatment_variant_count": detail_row["treatment_variant_count"],
                "baseline_avg_abs_beta": detail_row["baseline_avg_abs_beta"],
                "recommended_avg_abs_beta": detail_row["recommended_avg_abs_beta"],
                "ml_public_branch": detail_row["ml_public_branch"],
                "ml_public_branch_label": detail_row["ml_public_branch_label"],
                "dml_rows_written": detail_row["dml"]["rows_written"],
                "dml_avg_nuisance_r2_treatment": detail_row["dml"]["avg_nuisance_r2_treatment"],
                "tmle_rows_written": detail_row["tmle"]["rows_written"],
                "tmle_avg_nuisance_r2_treatment": detail_row["tmle"]["avg_nuisance_r2_treatment"],
                "forest_rows_written": detail_row["forest"]["rows_written"],
                "forest_avg_nuisance_r2_treatment": detail_row["forest"]["avg_nuisance_r2_treatment"],
                "negative_control_candidate_count": detail_row["negative_controls"]["candidate_count"],
                "negative_control_signal": detail_row["negative_controls"]["signal"],
                "negative_control_max_lead_placebo_abs_z": detail_row["negative_controls"]["max_lead_placebo_abs_z"],
                "summary_path": detail_row["links"]["summary_path"],
                "ladder_path": detail_row["links"]["ladder_path"],
                "treatment_path": detail_row["links"]["treatment_path"],
                "regime_path": detail_row["links"]["regime_path"],
                "dml_summary_path": detail_row["links"]["dml_summary_path"],
                "tmle_summary_path": detail_row["links"]["tmle_summary_path"],
                "forest_summary_path": detail_row["links"]["forest_summary_path"],
                "negative_control_path": detail_row["links"]["negative_control_path"],
            }
        )

    summary = {
        "generated_at": utc_now_iso(),
        "control_universe": {
            "feature_count": _safe_int(control_universe_meta.get("feature_count")),
            "quarter_count": _safe_int(control_universe_meta.get("quarter_count")),
            "series_count": _robustness_series_count(control_universe_meta),
            "series_count_by_frequency": control_universe_meta.get("series_count_by_frequency", {}) or {},
            "lag_structure": control_universe_meta.get("lag_structure", {}) or {},
            "panel_path": _repo_relative_str(control_universe_meta.get("panel_path", ""), paths.root),
            "columns_path": _repo_relative_str(control_universe_meta.get("columns_path", ""), paths.root),
        },
        "jobs_summarized": len(rows),
        "rows": rows,
    }
    summary_path = paths.reports / "robustness_snapshot.json"
    summary_csv_path = paths.reports / "robustness_snapshot.csv"
    write_json(summary_path, summary)
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else [
            "job_id",
            "estimator",
            "output_family",
            "release1_scope",
            "recommended_k",
            "recommended_k_reason",
            "recommended_factor_count",
            "screened_feature_count",
            "control_universe_feature_count",
            "regime_filter_count",
            "treatment_variant_count",
            "baseline_avg_abs_beta",
            "recommended_avg_abs_beta",
            "ml_public_branch",
            "ml_public_branch_label",
            "dml_rows_written",
            "dml_avg_nuisance_r2_treatment",
            "tmle_rows_written",
            "tmle_avg_nuisance_r2_treatment",
            "forest_rows_written",
            "forest_avg_nuisance_r2_treatment",
            "negative_control_candidate_count",
            "negative_control_signal",
            "negative_control_max_lead_placebo_abs_z",
            "summary_path",
            "ladder_path",
            "treatment_path",
            "regime_path",
            "dml_summary_path",
            "tmle_summary_path",
            "forest_summary_path",
            "negative_control_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return RobustnessSnapshotResult(
        summary_path=summary_path,
        summary_csv_path=summary_csv_path,
        jobs_summarized=len(rows),
        feature_count=_safe_int(control_universe_meta.get("feature_count")),
        series_count=_robustness_series_count(control_universe_meta),
    )
