from __future__ import annotations

from pathlib import Path
import importlib.util

from ea_tdc.config import load_runtime_config, load_source_registry


def test_runtime_config_loads_repo_defaults() -> None:
    config = load_runtime_config()

    assert config.project_name == "ea-tdc"
    assert config.package_name == "ea_tdc"
    assert config.fred_series_manifest == Path("config/fred_manifest_seed.csv").resolve()
    assert config.treasury_dataset_manifest == Path("config/treasury_manifest.yaml").resolve()
    assert config.remote.ssh_host == ""
    assert config.remote.run_heavy_jobs_remotely is False


def test_source_registry_has_self_contained_downloads() -> None:
    sources = {source.name: source for source in load_source_registry()}

    assert sources["fred"].kind == "direct_download"
    assert sources["fred"].target_dir == Path("data/raw/fred").resolve()
    assert sources["tdcest_seed"].path_env == "EA_TDC_SEED_TDCEST_BUNDLE"
    assert sources["tdcpass_seed"].path_env == "EA_TDC_SEED_TDCPASS_DIR"
    assert sources["accounting_seed"].path_env == "EA_TDC_SEED_ACCOUNTING_DIR"
    assert sources["wamest_seed"].path_env == "EA_TDC_SEED_WAMEST_DIR"
    assert sources["slrwatch_seed"].path_env == "EA_TDC_SEED_SLRWATCH_DIR"
    assert sources["interpol_seed"].copy_to == Path("data/seed/interpol").resolve()


def _load_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_treast_confirmatory_configs_are_explicitly_marked_non_tdc() -> None:
    config_paths = [
        Path("config/econark_dass_confirmatory.py"),
        Path("config/econark_dass_confirmatory_deposits.py"),
        Path("config/econark_dass_confirmatory_macro_prices.py"),
        Path("config/econark_dass_confirmatory_macro_prices_refined.py"),
    ]
    for config_path in config_paths:
        module = _load_module_from_path(config_path.resolve())
        assert getattr(module, "CONFIRMATORY_SCOPE", "") == "treast_diagnostic"
        assert getattr(module, "PUBLIC_TREATMENT_LABEL", "") == "TREAST"


def test_tdc_confirmatory_configs_use_repo_public_tdc(tmp_path: Path) -> None:
    config_paths = [
        Path("config/econark_dass_confirmatory_deposit_growth.py"),
        Path("config/econark_dass_confirmatory_macro_prices_baaff_tdc.py"),
        Path("config/econark_dass_confirmatory_macro_prices_shortlist_tdc.py"),
        Path("config/econark_dass_confirmatory_macro_prices_refined_tdc.py"),
    ]
    bundle = tmp_path / "data/bundles/tdcpass/standardized_series.csv"
    bundle.parent.mkdir(parents=True)
    bundle.write_text(
        "series_label,period_end,value\n"
        "tdc_bank_only_qoq,2020-06-30,25.0\n"
        "unrelated_series,2020-03-31,999.0\n"
        "tdc_bank_only_qoq,2020-03-31,10.0\n",
        encoding="utf-8",
    )
    (tmp_path / "config").mkdir()
    for config_path in config_paths:
        fixture_config = tmp_path / config_path
        fixture_config.write_bytes(config_path.read_bytes())
        module = _load_module_from_path(fixture_config)
        assert getattr(module, "CONFIRMATORY_SCOPE", "") == "tdc_confirmatory"
        assert getattr(module, "PUBLIC_TREATMENT_LABEL", "") == "tdc_bank_only_qoq"
        assert "tdc_est" in getattr(module, "EXTERNAL_Q_SERIES", {})
        panel = Path(module.EXTERNAL_Q_SERIES["tdc_est"]["path"])
        assert panel.is_relative_to(tmp_path)
        assert panel.read_text(encoding="utf-8").splitlines() == [
            "date,tdc_est",
            "2020-03-31,10.0",
            "2020-06-30,25.0",
        ]
