from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    config: Path
    data: Path
    raw: Path
    raw_fred: Path
    raw_treasury: Path
    seed: Path
    bundles: Path
    output: Path
    manifests: Path
    reports: Path


def project_paths(
    root: Path | str | None = None,
    *,
    data_root: str = "data",
    output_root: str = "output",
) -> ProjectPaths:
    root_path = Path(root or REPO_ROOT).resolve()
    data = root_path / data_root
    raw = data / "raw"
    output = root_path / output_root
    return ProjectPaths(
        root=root_path,
        config=root_path / "config",
        data=data,
        raw=raw,
        raw_fred=raw / "fred",
        raw_treasury=raw / "treasury",
        seed=data / "seed",
        bundles=data / "bundles",
        output=output,
        manifests=output / "manifests",
        reports=output / "reports",
    )


def ensure_repo_dirs(paths: ProjectPaths) -> None:
    expected_dirs = [
        paths.data,
        paths.raw,
        paths.raw_fred,
        paths.raw_treasury,
        paths.seed,
        paths.bundles,
        paths.output,
        paths.manifests,
        paths.reports,
    ]
    for path in expected_dirs:
        path.mkdir(parents=True, exist_ok=True)
