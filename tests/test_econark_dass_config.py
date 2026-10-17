from __future__ import annotations

import ast
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "econark_dass_confirmatory_mortgage_rates_tdc.py"


def _resolved_econark_root(monkeypatch, override: str | None) -> tuple[Path, Path]:
    if override is None:
        monkeypatch.delenv("ECONARK_ROOT", raising=False)
    else:
        monkeypatch.setenv("ECONARK_ROOT", override)

    tree = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"), filename=str(CONFIG_PATH))
    stop = next(
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ECONARK_ROOT" for target in node.targets)
    )
    namespace = {"__file__": str(CONFIG_PATH), "__name__": "__privacy_config_test__"}
    exec(compile(ast.Module(body=tree.body[: stop + 1], type_ignores=[]), str(CONFIG_PATH), "exec"), namespace)
    return namespace["EA_TDC_ROOT"], namespace["ECONARK_ROOT"]


def test_econark_root_defaults_to_repository_sibling(monkeypatch) -> None:
    root, econark_root = _resolved_econark_root(monkeypatch, None)

    assert econark_root == root.parent / "econark"


def test_econark_root_honors_explicit_override(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "provided-econark"
    _, econark_root = _resolved_econark_root(monkeypatch, str(override))

    assert econark_root == override
