from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .adapters.accounting import (
    adapt_accounting,
    apply_external_flow_rewrite,
    build_accounting_identity_alignment,
    build_seed_review,
    fill_missing_seed_channels_from_proxy_blocks,
    write_draft_seed_from_proxy_blocks,
)
from .adapters.coordwatch import adapt_coordwatch
from .adapters.qrawatch import adapt_qrawatch
from .adapters.slrwatch import adapt_slrwatch
from .adapters.tdcest import adapt_tdcest
from .adapters.tdcpass import adapt_tdcpass
from .adapters.tsyparty import adapt_tsyparty
from .adapters.wamest import adapt_wamest
from .artifacts import build_release_artifacts
from .config import load_runtime_config, load_source_registry
from .designs.events import build_event_design
from .designs.quarterly import build_quarterly_design
from .download import download_fred_bundle, download_treasury_bundle
from .estimation import build_estimation_snapshot, estimate_job, estimate_quarterly_job
from .iv_lab import build_iv_lab
from .ml_extensions import build_negative_control_mining, build_quarterly_dml, build_quarterly_forest, build_quarterly_tmle
from .paths import project_paths
from .reporting import (
    build_component_sidecar_artifact_pack,
    build_component_sidecar_screening,
    build_event_sidecar_artifact_pack,
    build_event_sidecar_screening,
    build_release_artifact_contract,
    build_release_contract,
    build_robustness_snapshot,
    build_release_scorecard,
    build_release_snapshot,
    build_stage_completion_closeout,
)
from .residualized_shock import build_quarterly_fwl_audit
from .robustness import DEFAULT_CONTROL_POLICY_MODE, build_control_universe, build_quarterly_robustness
from .sanitize import sanitize_output_paths
from .seeds import copy_seed_source
from .site import build_site
from .smoke import run_smoke


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ea-tdc")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke_parser = subparsers.add_parser("smoke", help="Validate config and write an empty manifest bundle")
    smoke_parser.add_argument("--repo-root", default=None, help="Override the repository root")

    fetch_parser = subparsers.add_parser("fetch-fred", help="Download the configured FRED seed manifest")
    fetch_parser.add_argument("--repo-root", default=None, help="Override the repository root")
    fetch_parser.add_argument("--limit", type=int, default=None, help="Optional number of series to download")
    fetch_parser.add_argument("--force", action="store_true", help="Redownload files even if they already exist")

    treasury_parser = subparsers.add_parser("fetch-treasury", help="Download configured Treasury FiscalData datasets")
    treasury_parser.add_argument("--repo-root", default=None, help="Override the repository root")
    treasury_parser.add_argument("--dataset", default=None, help="Optional dataset key from config/treasury_manifest.yaml")
    treasury_parser.add_argument("--max-pages", type=int, default=None, help="Optional page cap for each dataset")
    treasury_parser.add_argument("--force", action="store_true", help="Redownload files even if they already exist")

    seed_parser = subparsers.add_parser("seed-source", help="Copy an optional external seed into project storage")
    seed_parser.add_argument("source_name", help="Source key from config/source_registry.template.yaml")
    seed_parser.add_argument("--repo-root", default=None, help="Override the repository root")

    tdcest_parser = subparsers.add_parser("adapt-tdcest", help="Normalize the local tdcest seed bundle")
    tdcest_parser.add_argument("--repo-root", default=None, help="Override the repository root")
    tdcest_parser.add_argument("--bundle-path", default=None, help="Optional explicit bundle path")

    qrawatch_parser = subparsers.add_parser("adapt-qrawatch", help="Normalize a qrawatch publish directory")
    qrawatch_parser.add_argument("--repo-root", default=None, help="Override the repository root")
    qrawatch_parser.add_argument("--publish-dir", default=None, help="Optional explicit qrawatch publish dir")

    coordwatch_parser = subparsers.add_parser("adapt-coordwatch", help="Normalize a coordwatch publish directory")
    coordwatch_parser.add_argument("--repo-root", default=None, help="Override the repository root")
    coordwatch_parser.add_argument("--publish-dir", default=None, help="Optional explicit coordwatch publish dir")

    accounting_parser = subparsers.add_parser("adapt-accounting", help="Normalize quarterly identity-accounting inputs")
    accounting_parser.add_argument("--repo-root", default=None, help="Override the repository root")
    accounting_parser.add_argument("--seed-path", default=None, help="Optional explicit accounting input csv path")

    accounting_draft_parser = subparsers.add_parser(
        "draft-accounting-seed",
        help="Draft quarterly identity-accounting inputs from the hidden proxy-block bundles",
    )
    accounting_draft_parser.add_argument("--repo-root", default=None, help="Override the repository root")
    accounting_draft_parser.add_argument("--seed-path", default=None, help="Optional explicit accounting input csv path")
    accounting_draft_parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing accounting seed with user-entered rows")

    accounting_review_parser = subparsers.add_parser(
        "review-accounting-seed",
        help="Build a quarter-by-quarter review surface for the accounting identity seed",
    )
    accounting_review_parser.add_argument("--repo-root", default=None, help="Override the repository root")
    accounting_review_parser.add_argument("--seed-path", default=None, help="Optional explicit accounting input csv path")

    accounting_apply_parser = subparsers.add_parser(
        "rewrite-accounting-external-flow",
        help="Apply worksheet-implied external-flow rewrites to the accounting identity seed",
    )
    accounting_apply_parser.add_argument("--repo-root", default=None, help="Override the repository root")
    accounting_apply_parser.add_argument("--seed-path", default=None, help="Optional explicit accounting input csv path")
    accounting_apply_parser.add_argument("--rewrite-csv-path", default=None, help="Optional explicit rewrite worksheet csv path")
    accounting_apply_parser.add_argument("--min-priority", choices=["high", "medium", "low"], default="high", help="Rewrite rows at this priority or higher severity")

    accounting_fill_parser = subparsers.add_parser(
        "fill-accounting-missing-from-proxy",
        help="Fill blank accounting seed channels from the current proxy-block design bundle",
    )
    accounting_fill_parser.add_argument("--repo-root", default=None, help="Override the repository root")
    accounting_fill_parser.add_argument("--seed-path", default=None, help="Optional explicit accounting input csv path")

    accounting_alignment_parser = subparsers.add_parser(
        "build-accounting-alignment",
        help="Compare the public residual deposit component against the rebuilt hidden accounting identity outputs",
    )
    accounting_alignment_parser.add_argument("--repo-root", default=None, help="Override the repository root")

    tsyparty_parser = subparsers.add_parser("adapt-tsyparty", help="Normalize a tsyparty publish directory")
    tsyparty_parser.add_argument("--repo-root", default=None, help="Override the repository root")
    tsyparty_parser.add_argument("--publish-dir", default=None, help="Optional explicit tsyparty publish dir")

    tdcpass_parser = subparsers.add_parser("adapt-tdcpass", help="Normalize a tdcpass publish directory")
    tdcpass_parser.add_argument("--repo-root", default=None, help="Override the repository root")
    tdcpass_parser.add_argument("--publish-dir", default=None, help="Optional explicit tdcpass publish dir")

    wamest_parser = subparsers.add_parser("adapt-wamest", help="Normalize a wamest publish directory")
    wamest_parser.add_argument("--repo-root", default=None, help="Override the repository root")
    wamest_parser.add_argument("--publish-dir", default=None, help="Optional explicit wamest publish dir")

    slrwatch_parser = subparsers.add_parser("adapt-slrwatch", help="Normalize a slrwatch publish directory")
    slrwatch_parser.add_argument("--repo-root", default=None, help="Override the repository root")
    slrwatch_parser.add_argument("--publish-dir", default=None, help="Optional explicit slrwatch publish dir")

    design_parser = subparsers.add_parser("build-quarterly-design", help="Build a quarterly design scaffold from normalized bundles")
    design_parser.add_argument("job_id", help="Job id from config/dass_job_blueprint.yaml")
    design_parser.add_argument("--repo-root", default=None, help="Override the repository root")

    event_design_parser = subparsers.add_parser("build-event-design", help="Build an event design scaffold from normalized event bundles")
    event_design_parser.add_argument("job_id", help="Event job id from config/dass_job_blueprint.yaml")
    event_design_parser.add_argument("--repo-root", default=None, help="Override the repository root")

    snapshot_parser = subparsers.add_parser("build-release-snapshot", help="Build all configured designs and summarize portfolio readiness")
    snapshot_parser.add_argument("--repo-root", default=None, help="Override the repository root")

    release_scorecard_parser = subparsers.add_parser("build-release-scorecard", help="Build public readiness plus estimation scorecard")
    release_scorecard_parser.add_argument("--repo-root", default=None, help="Override the repository root")

    release_contract_parser = subparsers.add_parser("build-release-contract", help="Build Release 1 contract tiers across public and exploratory jobs")
    release_contract_parser.add_argument("--repo-root", default=None, help="Override the repository root")

    release_artifact_contract_parser = subparsers.add_parser(
        "build-release-artifact-contract",
        help="Build figure/table-level Release 1 artifact contracts for committed jobs",
    )
    release_artifact_contract_parser.add_argument("--repo-root", default=None, help="Override the repository root")

    event_sidecar_parser = subparsers.add_parser(
        "build-event-sidecar-screening",
        help="Build a compact rates/plumbing event benchmark summary from live event estimate tables",
    )
    event_sidecar_parser.add_argument("--repo-root", default=None, help="Override the repository root")

    component_sidecar_parser = subparsers.add_parser(
        "build-component-sidecar-screening",
        help="Build a compact component-treatment summary from live quarterly estimate tables",
    )
    component_sidecar_parser.add_argument("--repo-root", default=None, help="Override the repository root")

    component_pack_parser = subparsers.add_parser(
        "build-component-sidecar-artifacts",
        help="Build compact component-treatment artifact tables and summary markdown",
    )
    component_pack_parser.add_argument("--repo-root", default=None, help="Override the repository root")

    event_pack_parser = subparsers.add_parser(
        "build-event-sidecar-artifacts",
        help="Build compact event rates/plumbing artifact tables and summary markdown",
    )
    event_pack_parser.add_argument("--repo-root", default=None, help="Override the repository root")

    completion_parser = subparsers.add_parser(
        "build-stage-completion-closeout",
        help="Build the current stage completion closeout summary from live scoped artifacts",
    )
    completion_parser.add_argument("--repo-root", default=None, help="Override the repository root")

    release_artifacts_parser = subparsers.add_parser(
        "build-release-artifacts",
        help="Render committed Release 1 figures and tables from the artifact contract",
    )
    release_artifacts_parser.add_argument("--repo-root", default=None, help="Override the repository root")

    site_parser = subparsers.add_parser(
        "build-site",
        help="Build a GitHub Pages-ready static site from release artifacts and reports",
    )
    site_parser.add_argument("--repo-root", default=None, help="Override the repository root")

    iv_lab_parser = subparsers.add_parser("build-iv-lab", help="Scan alternative IV candidates for lp_iv jobs")
    iv_lab_parser.add_argument("--repo-root", default=None, help="Override the repository root")
    iv_lab_parser.add_argument("--job-id", default=None, help="Optional lp_iv job id to scan")

    universe_parser = subparsers.add_parser(
        "build-control-universe",
        help="Build the mixed-frequency quarterly control universe from local interpol raw seeds",
    )
    universe_parser.add_argument("--repo-root", default=None, help="Override the repository root")

    robustness_parser = subparsers.add_parser(
        "build-quarterly-robustness",
        help="Run mixed-frequency control, regime, and treatment sensitivity for one quarterly job",
    )
    robustness_parser.add_argument("job_id", help="Quarterly lp or lp_iv job id from config/dass_job_blueprint.yaml")
    robustness_parser.add_argument("--repo-root", default=None, help="Override the repository root")
    robustness_parser.add_argument("--k-grid", default="100,200,300", help="Comma-separated control screen sizes")
    robustness_parser.add_argument("--factor-count", type=int, default=4, help="Number of factor controls to extract")
    robustness_parser.add_argument(
        "--control-policy-mode",
        choices=["off", "balanced", "clean_macro"],
        default=DEFAULT_CONTROL_POLICY_MODE,
        help="Control eligibility policy before factor screening",
    )

    robustness_snapshot_parser = subparsers.add_parser(
        "build-robustness-snapshot",
        help="Summarize quarterly robustness ladders, treatment variants, and regime checks",
    )
    robustness_snapshot_parser.add_argument("--repo-root", default=None, help="Override the repository root")
    robustness_snapshot_parser.add_argument(
        "--job-id",
        action="append",
        default=None,
        help="Optional quarterly job id to limit the snapshot scope; may be repeated",
    )

    dml_parser = subparsers.add_parser(
        "build-quarterly-dml",
        help="Run cross-fitted ridge DML on a ready quarterly lp job",
    )
    dml_parser.add_argument("job_id", help="Quarterly lp job id from config/dass_job_blueprint.yaml")
    dml_parser.add_argument("--repo-root", default=None, help="Override the repository root")
    dml_parser.add_argument("--fold-count", type=int, default=3, help="Cross-fitting fold count")
    dml_parser.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge penalty for nuisance models")
    dml_parser.add_argument("--control-policy-mode", choices=["off", "balanced", "clean_macro"], default=DEFAULT_CONTROL_POLICY_MODE, help="Control eligibility policy if robustness must be built")

    fwl_parser = subparsers.add_parser(
        "build-quarterly-fwl-audit",
        help="Audit a factor-augmented LP by residualizing TDC and outcomes on the same controls",
    )
    fwl_parser.add_argument("job_id", help="Quarterly lp job id from config/dass_job_blueprint.yaml")
    fwl_parser.add_argument("--repo-root", default=None, help="Override the repository root")
    fwl_parser.add_argument("--k-screened", type=int, default=100, help="Number of screened controls to compress into factors")
    fwl_parser.add_argument("--factor-count", type=int, default=4, help="Number of factor controls to extract")
    fwl_parser.add_argument("--control-policy-mode", choices=["off", "balanced", "clean_macro"], default=DEFAULT_CONTROL_POLICY_MODE, help="Control eligibility policy before factor screening")

    tmle_parser = subparsers.add_parser(
        "build-quarterly-tmle",
        help="Run cross-fitted TMLE-style robustness on a ready quarterly lp job",
    )
    tmle_parser.add_argument("job_id", help="Quarterly lp job id from config/dass_job_blueprint.yaml")
    tmle_parser.add_argument("--repo-root", default=None, help="Override the repository root")
    tmle_parser.add_argument("--fold-count", type=int, default=3, help="Cross-fitting fold count")
    tmle_parser.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge penalty for nuisance models")
    tmle_parser.add_argument("--control-policy-mode", choices=["off", "balanced", "clean_macro"], default=DEFAULT_CONTROL_POLICY_MODE, help="Control eligibility policy if robustness must be built")

    forest_parser = subparsers.add_parser(
        "build-quarterly-forest",
        help="Run cross-fitted forest-style orthogonal robustness on a ready quarterly lp job",
    )
    forest_parser.add_argument("job_id", help="Quarterly lp job id from config/dass_job_blueprint.yaml")
    forest_parser.add_argument("--repo-root", default=None, help="Override the repository root")
    forest_parser.add_argument("--fold-count", type=int, default=3, help="Cross-fitting fold count")
    forest_parser.add_argument("--tree-count", type=int, default=40, help="Number of bagged shallow trees")
    forest_parser.add_argument("--max-depth", type=int, default=3, help="Maximum tree depth")
    forest_parser.add_argument("--min-leaf", type=int, default=8, help="Minimum leaf size")
    forest_parser.add_argument("--feature-fraction", type=float, default=0.5, help="Feature subsample share per tree")
    forest_parser.add_argument("--control-policy-mode", choices=["off", "balanced", "clean_macro"], default=DEFAULT_CONTROL_POLICY_MODE, help="Control eligibility policy if robustness must be built")

    negative_controls_parser = subparsers.add_parser(
        "build-negative-controls",
        help="Run lead-placebo and placebo-outcome mining for a quarterly lp job",
    )
    negative_controls_parser.add_argument("job_id", help="Quarterly lp job id from config/dass_job_blueprint.yaml")
    negative_controls_parser.add_argument("--repo-root", default=None, help="Override the repository root")
    negative_controls_parser.add_argument("--top-n", type=int, default=12, help="Number of placebo-outcome candidates to keep")
    negative_controls_parser.add_argument("--control-policy-mode", choices=["off", "balanced", "clean_macro"], default=DEFAULT_CONTROL_POLICY_MODE, help="Control eligibility policy if robustness must be built")

    estimate_job_parser = subparsers.add_parser("estimate-job", help="Estimate a ready supported job from its design bundle")
    estimate_job_parser.add_argument("job_id", help="Job id from config/dass_job_blueprint.yaml")
    estimate_job_parser.add_argument("--repo-root", default=None, help="Override the repository root")

    estimation_parser = subparsers.add_parser("estimate-quarterly-job", help="Estimate a ready quarterly lp job from its design bundle")
    estimation_parser.add_argument("job_id", help="Quarterly lp job id from config/dass_job_blueprint.yaml")
    estimation_parser.add_argument("--repo-root", default=None, help="Override the repository root")

    estimation_snapshot_parser = subparsers.add_parser("build-estimation-snapshot", help="Estimate all ready tracked quarterly lp jobs and summarize outputs")
    estimation_snapshot_parser.add_argument("--repo-root", default=None, help="Override the repository root")

    sanitize_parser = subparsers.add_parser(
        "sanitize-output-paths",
        help="Sanitize absolute local paths from repo output text artifacts",
    )
    sanitize_parser.add_argument("--repo-root", default=None, help="Override the repository root")
    sanitize_parser.add_argument(
        "--path",
        action="append",
        default=None,
        help="Optional repo-relative target path to sanitize; may be repeated. Defaults to output/",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if getattr(args, "repo_root", None) else None

    if args.command == "smoke":
        result = run_smoke(repo_root)
    elif args.command == "fetch-fred":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        result = download_fred_bundle(runtime, paths, limit=args.limit, force=args.force)
    elif args.command == "fetch-treasury":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        result = download_treasury_bundle(
            runtime,
            paths,
            dataset_name=args.dataset,
            max_pages=args.max_pages,
            force=args.force,
        )
    elif args.command == "seed-source":
        sources = {item.name: item for item in load_source_registry(repo_root)}
        if args.source_name not in sources:
            raise SystemExit(f"Unknown source: {args.source_name}")
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        result = copy_seed_source(sources[args.source_name], paths)
    elif args.command == "adapt-tdcest":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        adapted = adapt_tdcest(paths, bundle_path=args.bundle_path)
        result = {
            "standardized_path": str(adapted.standardized_path),
            "manifest_path": str(adapted.manifest_path),
            "rows_written": adapted.rows_written,
            "bundle_hash": adapted.bundle_hash,
        }
    elif args.command == "adapt-qrawatch":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        adapted = adapt_qrawatch(paths, publish_dir=args.publish_dir)
        result = {
            "standardized_path": str(adapted.standardized_path),
            "event_bundle_path": str(adapted.event_bundle_path),
            "manifest_path": str(adapted.manifest_path),
            "series_rows_written": adapted.series_rows_written,
            "event_rows_written": adapted.event_rows_written,
            "bundle_hash": adapted.bundle_hash,
        }
    elif args.command == "adapt-coordwatch":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        adapted = adapt_coordwatch(paths, publish_dir=args.publish_dir)
        result = {
            "standardized_path": str(adapted.standardized_path),
            "manifest_path": str(adapted.manifest_path),
            "rows_written": adapted.rows_written,
            "bundle_hash": adapted.bundle_hash,
        }
    elif args.command == "adapt-accounting":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        adapted = adapt_accounting(paths, seed_path=args.seed_path)
        result = {
            "standardized_path": _display_path(adapted.standardized_path, paths.root),
            "manifest_path": _display_path(adapted.manifest_path, paths.root),
            "rows_written": adapted.rows_written,
            "seed_path": _display_path(adapted.seed_path, paths.root),
            "bundle_hash": adapted.bundle_hash,
        }
    elif args.command == "draft-accounting-seed":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        build_quarterly_design(paths, job_id="baseline_tdc_lp_deposit_source_blocks")
        build_quarterly_design(paths, job_id="baseline_tdc_lp_deposit_source_blocks_pct_gdp")
        drafted = write_draft_seed_from_proxy_blocks(paths, seed_path=args.seed_path, overwrite=args.overwrite)
        result = {
            "seed_path": _display_path(drafted.seed_path, paths.root),
            "reference_path": _display_path(drafted.reference_path, paths.root),
            "manifest_path": _display_path(drafted.manifest_path, paths.root),
            "rows_written": drafted.rows_written,
        }
    elif args.command == "review-accounting-seed":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        reviewed = build_seed_review(paths, seed_path=args.seed_path)
        result = {
            "review_csv_path": _display_path(reviewed.review_csv_path, paths.root),
            "review_md_path": _display_path(reviewed.review_md_path, paths.root),
            "summary_md_path": _display_path(reviewed.summary_md_path, paths.root),
            "rewrite_csv_path": _display_path(reviewed.rewrite_csv_path, paths.root),
            "rewrite_md_path": _display_path(reviewed.rewrite_md_path, paths.root),
            "manifest_path": _display_path(reviewed.manifest_path, paths.root),
            "rows_written": reviewed.rows_written,
            "high_priority_rows": reviewed.high_priority_rows,
        }
    elif args.command == "rewrite-accounting-external-flow":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        rewritten = apply_external_flow_rewrite(
            paths,
            seed_path=args.seed_path,
            rewrite_csv_path=args.rewrite_csv_path,
            min_priority=args.min_priority,
        )
        result = {
            "seed_path": _display_path(rewritten.seed_path, paths.root),
            "manifest_path": _display_path(rewritten.manifest_path, paths.root),
            "rows_updated": rewritten.rows_updated,
            "min_priority": args.min_priority,
        }
    elif args.command == "fill-accounting-missing-from-proxy":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        filled = fill_missing_seed_channels_from_proxy_blocks(paths, seed_path=args.seed_path)
        result = {
            "seed_path": _display_path(filled.seed_path, paths.root),
            "manifest_path": _display_path(filled.manifest_path, paths.root),
            "rows_updated": filled.rows_updated,
            "deposit_substitution_fills": filled.deposit_substitution_fills,
            "public_liquidity_fills": filled.public_liquidity_fills,
        }
    elif args.command == "build-accounting-alignment":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        aligned = build_accounting_identity_alignment(paths)
        result = {
            "csv_path": _display_path(aligned.csv_path, paths.root),
            "md_path": _display_path(aligned.md_path, paths.root),
            "manifest_path": _display_path(aligned.manifest_path, paths.root),
            "rows_written": aligned.rows_written,
        }
    elif args.command == "adapt-tsyparty":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        adapted = adapt_tsyparty(paths, publish_dir=args.publish_dir)
        result = {
            "standardized_path": str(adapted.standardized_path),
            "manifest_path": str(adapted.manifest_path),
            "rows_written": adapted.rows_written,
            "bundle_hash": adapted.bundle_hash,
        }
    elif args.command == "adapt-tdcpass":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        adapted = adapt_tdcpass(paths, publish_dir=args.publish_dir)
        result = {
            "standardized_path": str(adapted.standardized_path),
            "published_reference_path": str(adapted.published_reference_path),
            "manifest_path": str(adapted.manifest_path),
            "rows_written": adapted.rows_written,
            "bundle_hash": adapted.bundle_hash,
        }
    elif args.command == "adapt-wamest":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        adapted = adapt_wamest(paths, publish_dir=args.publish_dir)
        result = {
            "standardized_path": str(adapted.standardized_path),
            "manifest_path": str(adapted.manifest_path),
            "rows_written": adapted.rows_written,
            "bundle_hash": adapted.bundle_hash,
        }
    elif args.command == "adapt-slrwatch":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        adapted = adapt_slrwatch(paths, publish_dir=args.publish_dir)
        result = {
            "standardized_path": str(adapted.standardized_path),
            "manifest_path": str(adapted.manifest_path),
            "rows_written": adapted.rows_written,
            "bundle_hash": adapted.bundle_hash,
        }
    elif args.command == "build-quarterly-design":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        built = build_quarterly_design(paths, job_id=args.job_id)
        result = {
            "bundle_path": str(built.bundle_path),
            "design_manifest_path": str(built.design_manifest_path),
            "diagnostics_manifest_path": str(built.diagnostics_manifest_path) if built.diagnostics_manifest_path else "",
            "sample_manifest_path": str(built.sample_manifest_path),
            "rows_written": built.rows_written,
            "usable_rows": built.usable_rows,
        }
    elif args.command == "build-event-design":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        built = build_event_design(paths, job_id=args.job_id)
        result = {
            "bundle_path": str(built.bundle_path),
            "design_manifest_path": str(built.design_manifest_path),
            "sample_manifest_path": str(built.sample_manifest_path),
            "rows_written": built.rows_written,
            "usable_rows": built.usable_rows,
        }
    elif args.command == "build-release-snapshot":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        snapshot = build_release_snapshot(paths)
        result = {
            "jobs_built": snapshot.jobs_built,
            "partial_jobs": snapshot.partial_jobs,
            "ready_jobs": snapshot.ready_jobs,
            "summary_csv_path": str(snapshot.summary_csv_path),
            "summary_path": str(snapshot.summary_path),
        }
    elif args.command == "build-release-scorecard":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        scorecard = build_release_scorecard(paths)
        result = {
            "public_jobs": scorecard.public_jobs,
            "ready_jobs": scorecard.ready_jobs,
            "estimated_jobs": scorecard.estimated_jobs,
            "summary_csv_path": str(scorecard.summary_csv_path),
            "summary_path": str(scorecard.summary_path),
        }
    elif args.command == "build-release-contract":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        contract = build_release_contract(paths)
        result = {
            "active_jobs": contract.active_jobs,
            "main_candidates": contract.main_candidates,
            "appendix_candidates": contract.appendix_candidates,
            "exploratory_sidecar_jobs": contract.exploratory_sidecar_jobs,
            "blocked_jobs": contract.blocked_jobs,
            "summary_csv_path": str(contract.summary_csv_path),
            "summary_path": str(contract.summary_path),
        }
    elif args.command == "build-release-artifact-contract":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        artifact_contract = build_release_artifact_contract(paths)
        result = {
            "committed_jobs": artifact_contract.committed_jobs,
            "main_text_artifacts": artifact_contract.main_text_artifacts,
            "appendix_artifacts": artifact_contract.appendix_artifacts,
            "summary_csv_path": str(artifact_contract.summary_csv_path),
            "summary_path": str(artifact_contract.summary_path),
        }
    elif args.command == "build-event-sidecar-screening":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        built = build_event_sidecar_screening(paths)
        result = {
            "signal_count": built.signal_count,
            "jobs_summarized": built.jobs_summarized,
            "summary_csv_path": str(built.summary_csv_path),
            "summary_path": str(built.summary_path),
        }
    elif args.command == "build-component-sidecar-screening":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        built = build_component_sidecar_screening(paths)
        result = {
            "signal_count": built.signal_count,
            "jobs_summarized": built.jobs_summarized,
            "summary_csv_path": str(built.summary_csv_path),
            "summary_path": str(built.summary_path),
        }
    elif args.command == "build-component-sidecar-artifacts":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        built = build_component_sidecar_artifact_pack(paths)
        result = {
            "signal_count": built.signal_count,
            "summary_path": str(built.summary_path),
            "reduced_form_csv_path": str(built.reduced_form_csv_path),
            "liquidity_csv_path": str(built.liquidity_csv_path),
            "state_probe_csv_path": str(built.state_probe_csv_path),
            "manifest_path": str(built.manifest_path),
        }
    elif args.command == "build-event-sidecar-artifacts":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        built = build_event_sidecar_artifact_pack(paths)
        result = {
            "signal_count": built.signal_count,
            "summary_path": str(built.summary_path),
            "rates_csv_path": str(built.rates_csv_path),
            "plumbing_csv_path": str(built.plumbing_csv_path),
            "manifest_path": str(built.manifest_path),
        }
    elif args.command == "build-stage-completion-closeout":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        built = build_stage_completion_closeout(paths)
        result = {
            "summary_path": str(built.summary_path),
            "manifest_path": str(built.manifest_path),
        }
    elif args.command == "build-release-artifacts":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        artifact_build = build_release_artifacts(paths)
        result = {
            "artifacts_built": artifact_build.artifacts_built,
            "figure_artifacts": artifact_build.figure_artifacts,
            "table_artifacts": artifact_build.table_artifacts,
            "gallery_path": str(artifact_build.gallery_path),
            "summary_csv_path": str(artifact_build.summary_csv_path),
            "summary_path": str(artifact_build.summary_path),
        }
    elif args.command == "build-site":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        built_site = build_site(paths)
        result = {
            "copied_artifacts": built_site.copied_artifacts,
            "copied_reports": built_site.copied_reports,
            "copied_models": built_site.copied_models,
            "index_path": str(built_site.index_path),
            "sidecar_index_path": str(built_site.sidecar_index_path),
            "summary_path": str(built_site.summary_path),
        }
    elif args.command == "build-iv-lab":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        iv_lab = build_iv_lab(paths, job_id=args.job_id)
        result = {
            "jobs_scanned": iv_lab.jobs_scanned,
            "total_candidates": iv_lab.total_candidates,
            "summary_csv_path": str(iv_lab.summary_csv_path),
            "summary_path": str(iv_lab.summary_path),
        }
    elif args.command == "build-control-universe":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        built = build_control_universe(paths)
        result = {
            "panel_path": str(built.panel_path),
            "meta_path": str(built.meta_path),
            "columns_path": str(built.columns_path),
            "quarter_count": built.quarter_count,
            "feature_count": built.feature_count,
        }
    elif args.command == "build-quarterly-robustness":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        k_grid = [int(item.strip()) for item in str(args.k_grid).split(",") if item.strip()]
        built = build_quarterly_robustness(
            paths,
            job_id=args.job_id,
            k_grid=k_grid,
            factor_count=args.factor_count,
            control_policy_mode=args.control_policy_mode,
        )
        result = {
            "summary_path": str(built.summary_path),
            "ladder_path": str(built.ladder_path),
            "regime_path": str(built.regime_path),
            "treatment_path": str(built.treatment_path),
            "control_meta_path": str(built.control_meta_path),
        }
    elif args.command == "build-robustness-snapshot":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        built = build_robustness_snapshot(paths, job_ids=args.job_id or None)
        result = {
            "summary_path": str(built.summary_path),
            "summary_csv_path": str(built.summary_csv_path),
            "jobs_summarized": built.jobs_summarized,
            "feature_count": built.feature_count,
            "series_count": built.series_count,
        }
    elif args.command == "build-quarterly-dml":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        built = build_quarterly_dml(
            paths,
            job_id=args.job_id,
            fold_count=args.fold_count,
            ridge_alpha=args.ridge_alpha,
            control_policy_mode=args.control_policy_mode,
        )
        result = {
            "estimates_path": str(built.estimates_path),
            "summary_path": str(built.summary_path),
            "rows_written": built.rows_written,
        }
    elif args.command == "build-quarterly-fwl-audit":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        built = build_quarterly_fwl_audit(
            paths,
            job_id=args.job_id,
            k_screened=args.k_screened,
            factor_count=args.factor_count,
            control_policy_mode=args.control_policy_mode,
        )
        result = {
            "estimates_path": str(built.estimates_path),
            "diagnostics_path": str(built.diagnostics_path),
            "summary_path": str(built.summary_path),
            "rows_written": built.rows_written,
        }
    elif args.command == "build-quarterly-tmle":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        built = build_quarterly_tmle(
            paths,
            job_id=args.job_id,
            fold_count=args.fold_count,
            ridge_alpha=args.ridge_alpha,
            control_policy_mode=args.control_policy_mode,
        )
        result = {
            "estimates_path": str(built.estimates_path),
            "summary_path": str(built.summary_path),
            "rows_written": built.rows_written,
        }
    elif args.command == "build-quarterly-forest":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        built = build_quarterly_forest(
            paths,
            job_id=args.job_id,
            fold_count=args.fold_count,
            tree_count=args.tree_count,
            max_depth=args.max_depth,
            min_leaf=args.min_leaf,
            feature_fraction=args.feature_fraction,
            control_policy_mode=args.control_policy_mode,
        )
        result = {
            "estimates_path": str(built.estimates_path),
            "summary_path": str(built.summary_path),
            "rows_written": built.rows_written,
        }
    elif args.command == "build-negative-controls":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        built = build_negative_control_mining(
            paths,
            job_id=args.job_id,
            top_n=args.top_n,
            control_policy_mode=args.control_policy_mode,
        )
        result = {
            "summary_path": str(built.summary_path),
            "summary_csv_path": str(built.summary_csv_path),
            "rows_written": built.rows_written,
        }
    elif args.command == "estimate-job":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        estimated = estimate_job(paths, job_id=args.job_id)
        result = {
            "estimator": estimated.estimator,
            "estimates_path": str(estimated.estimates_path),
            "summary_path": str(estimated.summary_path),
            "comparison_path": str(estimated.comparison_path) if estimated.comparison_path else "",
            "rows_written": estimated.rows_written,
        }
    elif args.command == "estimate-quarterly-job":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        estimated = estimate_quarterly_job(paths, job_id=args.job_id)
        result = {
            "estimator": estimated.estimator,
            "estimates_path": str(estimated.estimates_path),
            "summary_path": str(estimated.summary_path),
            "comparison_path": str(estimated.comparison_path) if estimated.comparison_path else "",
            "rows_written": estimated.rows_written,
        }
    elif args.command == "build-estimation-snapshot":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        snapshot = build_estimation_snapshot(paths)
        result = {
            "jobs_estimated": snapshot.jobs_estimated,
            "summary_csv_path": str(snapshot.summary_csv_path),
            "summary_path": str(snapshot.summary_path),
        }
    elif args.command == "sanitize-output-paths":
        runtime = load_runtime_config(repo_root)
        paths = project_paths(repo_root, data_root=runtime.data_root, output_root=runtime.output_root)
        targets = [paths.root / item for item in (args.path or [])]
        sanitized = sanitize_output_paths(paths, targets=targets or None)
        result = {
            "files_changed": sanitized.files_changed,
            "files_scanned": sanitized.files_scanned,
            "target_count": sanitized.target_count,
        }
    else:
        raise SystemExit(f"Unsupported command: {args.command}")

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
