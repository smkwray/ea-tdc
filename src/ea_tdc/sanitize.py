from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

from .paths import ProjectPaths


TEXT_SUFFIXES = {".json", ".md", ".txt", ".html", ".csv", ".svg", ".js", ".css", ".yaml", ".yml"}
EMAIL_PATTERN = re.compile(r"(?P<email>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")


@dataclass(frozen=True)
class PathSanitizationResult:
    target_count: int
    files_scanned: int
    files_changed: int


def _replace_prefix(text: str, prefix: str, replacement: str) -> str:
    if not prefix:
        return text
    if replacement == "":
        return text.replace(f"{prefix}/", "").replace(prefix, "")
    exact = text.replace(prefix, replacement.rstrip("/"))
    if prefix.endswith("/"):
        return exact
    return exact.replace(f"{prefix}/", replacement)


def _sanitize_text(content: str, *, repo_root: Path) -> str:
    repo_root_text = str(repo_root.resolve())
    repo_parent_text = str(repo_root.resolve().parent)
    home_text = str(Path.home().resolve())

    sanitized = content
    sanitized = _replace_prefix(sanitized, repo_root_text, "")
    sanitized = _replace_prefix(sanitized, repo_parent_text, "../")
    sanitized = _replace_prefix(sanitized, home_text, "~/")
    sanitized = EMAIL_PATTERN.sub("REDACTED_EMAIL", sanitized)
    return sanitized


def sanitize_output_paths(
    paths: ProjectPaths,
    *,
    targets: list[Path] | None = None,
) -> PathSanitizationResult:
    scan_targets = targets or [paths.output]
    files_scanned = 0
    files_changed = 0

    for target in scan_targets:
        if not target.exists():
            continue
        for path in target.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            files_scanned += 1
            sanitized = _sanitize_text(content, repo_root=paths.root)
            if sanitized == content:
                continue
            path.write_text(sanitized, encoding="utf-8")
            files_changed += 1

    return PathSanitizationResult(
        target_count=len(scan_targets),
        files_scanned=files_scanned,
        files_changed=files_changed,
    )
