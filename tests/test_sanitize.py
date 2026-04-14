from __future__ import annotations

from pathlib import Path

from ea_tdc.paths import ensure_repo_dirs, project_paths
from ea_tdc.sanitize import sanitize_output_paths


def test_sanitize_output_paths_rewrites_repo_sibling_and_home_paths(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)

    legacy_path = paths.output / "legacy_paths.json"
    repo_target = paths.reports / "custom_report.csv"
    sibling_target = paths.root.parent / "tdcpass" / "output" / "models" / "published.csv"
    cloud_target = Path.home() / "Library" / "CloudStorage" / "GoogleDrive-wray7830@gmail.com" / "My Drive" / "github" / "econark"
    legacy_path.write_text(
        "\n".join(
            [
                f'repo="{repo_target}"',
                f'sibling="{sibling_target}"',
                f'cloud="{cloud_target}"',
            ]
        ),
        encoding="utf-8",
    )

    result = sanitize_output_paths(paths)

    assert result.target_count == 1
    assert result.files_scanned >= 1
    assert result.files_changed >= 1

    sanitized = legacy_path.read_text(encoding="utf-8")
    assert 'repo="output/reports/custom_report.csv"' in sanitized
    assert 'sibling="../tdcpass/output/models/published.csv"' in sanitized
    assert 'cloud="~/Library/CloudStorage/REDACTED_EMAIL/My Drive/github/econark"' in sanitized
    assert str(paths.root) not in sanitized
    assert "wray7830@gmail.com" not in sanitized
