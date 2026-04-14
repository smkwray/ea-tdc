from __future__ import annotations

from pathlib import Path

from ea_tdc.download import load_treasury_manifest


def test_treasury_manifest_loads_expected_datasets() -> None:
    manifest = load_treasury_manifest(Path("config/treasury_manifest.yaml"))

    assert "auctions_query" in manifest
    assert manifest["auctions_query"]["endpoint"] == "v1/accounting/od/auctions_query"
    assert "operating_cash_balance" in manifest
