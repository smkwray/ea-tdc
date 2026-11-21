"""Economic invariants of the declared 2026Q2 vintage rerun.

A vintage may extend the sample and refresh inputs; it may not truncate the
sample, change the 16 controls, reselect the factor features, or loosen gates.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import shutil
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from ea_tdc import open01, open16
from ea_tdc.adapters.tdcest import normalize_schema, read_raw
from ea_tdc.open_contract import (
    CANONICAL_CONTROL_IDS,
    CANONICAL_OUTCOME_ID,
    CANONICAL_TREATMENT_ID,
    CANONICAL_TREATMENT_SOURCE_SERIES,
    EXPECTED_METHOD_TIER_COUNTS,
    OPEN02_CONTRACT,
)
from ea_tdc.paths import project_paths
from ea_tdc.residualized_shock import _load_factor_branch
from ea_tdc.robustness import _extract_factor_controls, _merge_control_rows
from ea_tdc.designs.quarterly import RAW_FRED_LEVEL_CHANGE_SERIES
from ea_tdc.vintage import (
    VINTAGE_ENV,
    declared_treatment,
    deposit_outcome,
    deposit_outcome_fetch,
    deposit_outcome_terms,
    load_vintage,
    open01_tier_counts,
    open01_treatment,
    treatment_series_overrides,
    open16_disclosures,
    open16_sample,
    pinned_factor_features,
    quarter_range,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "config/vintage_2026q2.json"
FROZEN_CONTROL_IDS = (
    "GDP", "gdp_deflator", "FEDFUNDS", "TOTRESNS",
    "tier2_regression_bank_row_tier_pre_component_h15_scaled",
    "tdcpass_strict_loan_core_min_qoq__lag_2", "tdcpass_strict_loan_core_min_qoq__lag_4",
    "tdcpass_strict_loan_consumer_credit_qoq__lag_4", "bank_credit_qoq__lag_4",
    "dgs2__lag_4", "dgs10__lag_1", "dgs10__lag_2",
    "dflmx_k100_f1", "dflmx_k100_f2", "dflmx_k100_f3", "dflmx_k100_f4",
)


def _refresh_module():
    spec = importlib.util.spec_from_file_location("run_vintage_refresh", ROOT / "scripts" / "run_vintage_refresh.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def refresh():
    return _refresh_module()


@pytest.fixture(scope="module")
def vintage():
    return load_vintage(ROOT, CONFIG)


def _vintage_variant(tmp_path: Path, **changes) -> Path:
    payload = json.loads((ROOT / CONFIG).read_text())
    for key, value in changes.items():
        payload[key] = value
    path = tmp_path / "vintage.json"
    path.write_text(json.dumps(payload))
    return path


def _panel(quarters, seed=7):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(len(quarters), 16))
    legs = rng.normal(size=(len(quarters), 3)) * [100, 30, 50]
    c = legs.sum(axis=1)
    y = 0.5 * c + z @ np.arange(16) + rng.normal(size=len(quarters)) * 20
    rows, sources = [], []
    for i, quarter in enumerate(quarters):
        rows.append({"quarter": quarter, CANONICAL_TREATMENT_ID: c[i], CANONICAL_OUTCOME_ID: y[i],
                     **dict(zip(CANONICAL_CONTROL_IDS, z[i], strict=True))})
        sources.append({"quarter": quarter, "C": c[i], "R": legs[i, 0], "J": legs[i, 1], "O": legs[i, 2],
                        "row_tsy_tx": legs[i, 0] + 5.0})
    return rows, sources


def test_committed_vintage_extends_admitted_sample_through_2026q2(vintage):
    assert vintage.quarters[0] == "2002Q1" and vintage.quarters[-1] == "2026Q2"
    assert vintage.observations == len(vintage.quarters) == 98
    assert vintage.quarters[:96] == quarter_range("2002Q1", "2025Q4")
    assert vintage.observation_cutoff == date(2026, 6, 30)
    assert vintage.rolling_endpoints[0] == "2011Q4" and vintage.rolling_endpoints[-1] == "2026Q2"
    assert len(vintage.rolling_endpoints) == 59
    # Historical construction tiers are untouched; only the two appended quarters join the modern tier.
    growth = {k: vintage.method_tier_counts[k] - EXPECTED_METHOD_TIER_COUNTS[k] for k in EXPECTED_METHOD_TIER_COUNTS}
    assert growth == {"pre_component_h15_scaled_backcast": 0, "component_pool_wamest_bucket_backcast": 0,
                      "constrained_component": 2}


def test_controls_and_factor_features_are_the_frozen_specification(vintage, tmp_path):
    assert tuple(CANONICAL_CONTROL_IDS) == FROZEN_CONTROL_IDS
    assert [c for c in CANONICAL_CONTROL_IDS if c.startswith("dflmx_")] == [f"dflmx_k100_f{i}" for i in range(1, 5)]
    pin = vintage.payload["factor_features"]["ordered_top100"]
    target = tmp_path / pin["path"]
    target.parent.mkdir(parents=True)
    features = [f"q__series{i:03d}_x__lag001" for i in range(100)]
    target.write_text(json.dumps(features))
    with pytest.raises(ValueError, match="differs from its vintage pin"):
        pinned_factor_features(vintage, tmp_path)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"sample": {"start": "2002Q1", "end": "2026Q2", "n": 97}}, "count disagrees"),
        ({"sample": {"start": "2002Q2", "end": "2026Q2", "n": 97}}, "without dropping"),
        ({"observation_cutoff": "2026-09-30"}, "last day of the vintage end quarter"),
        ({"method_tier_counts": {"pre_component_h15_scaled_backcast": 32,
                                 "component_pool_wamest_bucket_backcast": 47, "constrained_component": 19}},
         "cannot drop admitted"),
        ({"method_tier_counts": {"pre_component_h15_scaled_backcast": 33,
                                 "component_pool_wamest_bucket_backcast": 47, "constrained_component": 17}},
         "every sample quarter"),
    ],
)
def test_vintage_loader_rejects_sample_or_tier_drift(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        load_vintage(tmp_path, str(_vintage_variant(tmp_path, **changes)))


def test_projection_panel_refuses_silent_truncation_or_extension(refresh, vintage):
    rows, _ = _panel(vintage.quarters)
    as_text = [{k: (v if k == "quarter" else repr(float(v))) for k, v in row.items()} for row in rows]
    assert [r["quarter"] for r in refresh.projection_panel(as_text, vintage.quarters)] == list(vintage.quarters)
    truncated = [dict(r) for r in as_text]
    truncated[-1]["dgs10__lag_1"] = ""
    with pytest.raises(ValueError, match=r"missing=\['2026Q2'\]"):
        refresh.projection_panel(truncated, vintage.quarters)
    extended = [*as_text, {**as_text[-1], "quarter": "2026Q3"}]
    with pytest.raises(ValueError, match=r"extra=\['2026Q3'\]"):
        refresh.projection_panel(extended, vintage.quarters)


def test_open01_tier_binding_is_scoped_to_the_vintage(vintage):
    frozen = dict(EXPECTED_METHOD_TIER_COUNTS)
    with open01_tier_counts(vintage) as binding:
        assert dict(open01.EXPECTED_METHOD_TIER_COUNTS) == dict(vintage.method_tier_counts)
        assert binding["admitted"] == frozen
        with pytest.raises(RuntimeError, match="refusing to rebind"):
            with open01_tier_counts(vintage):
                pass
    assert dict(open01.EXPECTED_METHOD_TIER_COUNTS) == frozen


def test_open16_diagnostics_run_on_exactly_the_vintage_sample(vintage):
    rows, legs = _panel(vintage.quarters)
    with pytest.raises(ValueError, match="96 ordered"):
        open16.validate_panel(rows)
    with open16_sample(vintage):
        joined = open16.join_legs(rows, legs)
        with pytest.raises(ValueError):
            open16.validate_panel(rows[:-1])
        path = open16.pandemic_path(joined, nominal_endpoints=vintage.rolling_endpoints)
        beta = open01._fit_projection(joined, treatment_id=CANONICAL_TREATMENT_ID, outcome_id=CANONICAL_OUTCOME_ID,
                                      control_ids=CANONICAL_CONTROL_IDS).beta
        contributions = open16.covariance_contributions(joined, beta)
    assert len(path) == 59 and path[-1]["nominal_end"] == "2026Q2" and path[0]["nominal_end"] == "2011Q4"
    assert all(row["n_obs"] <= 48 for row in path)
    corrected = {r["leg"]: r["contribution"] for r in contributions if r["classification"] == "corrected"}
    assert corrected["R"] + corrected["J"] + corrected["O"] == pytest.approx(beta - 1, abs=1e-9)
    assert open16.FROZEN_QUARTERS == quarter_range("2002Q1", "2025Q4")


def _write_rows(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_pinned_features_replace_screening_in_the_live_factor_branch(tmp_path, monkeypatch, vintage):
    quarters = quarter_range("2000Q1", "2009Q4")
    rng = np.random.default_rng(3)
    for i in range(13):
        dates = [f"{1998 + k // 4}-{(k % 4) * 3 + 1:02d}-01" for k in range(48)]
        _write_rows(tmp_path / "data/seed/interpol/raw" / f"FRED_ALPHA{i:02d}_alpha{i:02d}.csv", ["date", "value"],
                    zip(dates, rng.normal(size=48).cumsum()))
    treatment = rng.normal(size=len(quarters))
    bundle = tmp_path / "data/bundles/designs/anchor.csv"
    _write_rows(bundle, ["quarter", "T", "Y"],
                [(q, t, 0.5 * t + rng.normal()) for q, t in zip(quarters, treatment, strict=True)])
    manifest = {"bundle_path": str(bundle), "treatment_id": "T", "outcome_ids": ["Y"], "control_ids": []}
    paths = project_paths(tmp_path)

    unpinned_rows, _, _, _ = _load_factor_branch(paths, job_id="anchor", design_manifest=manifest, k_screened=100,
                                                 factor_count=4, control_policy_mode="balanced", min_coverage=0.4)
    merged, features = _merge_control_rows([dict(r) for r in csv.DictReader(bundle.open())],
                                            paths.reports / "control_universe_quarterly.csv")
    pinned = list(reversed(features))[:100]
    pin = vintage.payload["factor_features"]["ordered_top100"]
    (tmp_path / pin["path"]).parent.mkdir(parents=True)
    (tmp_path / pin["path"]).write_text(json.dumps(pinned))
    payload = json.loads((ROOT / CONFIG).read_text())
    payload["factor_features"]["ordered_top100"]["sha256"] = sha256_file(tmp_path / pin["path"])
    (tmp_path / CONFIG).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / CONFIG).write_text(json.dumps(payload))
    monkeypatch.setenv(VINTAGE_ENV, CONFIG)

    pinned_rows, control_ids, screened, count = _load_factor_branch(
        paths, job_id="anchor", design_manifest=manifest, k_screened=100,
        factor_count=4, control_policy_mode="balanced", min_coverage=0.4)
    _, expected_rows, _, _ = _extract_factor_controls(rows=merged, feature_ids=pinned, prefix="dflmx_k100", n_factors=4)
    factor_ids = [f"dflmx_k100_f{i}" for i in range(1, 5)]
    assert control_ids == factor_ids and screened == 100 and count == 4
    assert [[r[k] for k in factor_ids] for r in pinned_rows] == [[r[k] for k in factor_ids] for r in expected_rows]
    assert [[r[k] for k in factor_ids] for r in pinned_rows] != [[r[k] for k in factor_ids] for r in unpinned_rows]
    with pytest.raises(ValueError, match="exactly the admitted K=100"):
        _load_factor_branch(paths, job_id="anchor", design_manifest=manifest, k_screened=200,
                            factor_count=4, control_policy_mode="balanced", min_coverage=0.4)


def test_site_bundle_reconstruction_feeds_the_tdcest_adapter(refresh, tmp_path):
    processed = tmp_path / "processed"
    dates = ["2026-03-31", "2026-06-30"]
    _write_rows(processed / "tdc_estimates.csv", ["date", "tdc_base_bank_only_ru_flow", "tdc_x"],
                [(dates[0], "1.5", ""), (dates[1], "-2.25", "3")])
    _write_rows(processed / "tdc_components.csv", ["date", "bank_depository_tsy_tx"], [(dates[0], "10.0"), (dates[1], "-4.0")])
    _write_rows(processed / "tdc_corrections.csv", ["date", "c"], [(dates[0], "0.0"), (dates[1], "0.0")])
    _write_rows(processed / "quarterly_inputs.csv", ["date", "gdp_deflator", "nominal_gdp_saar_bil"],
                [(dates[1], "131.5", "31500.0")])
    bundle = refresh.reconstruct_site_bundle(processed, generated_at="2026-10-14T15:30:31Z", input_hashes={})
    target = tmp_path / "bundle.json"
    target.write_text(json.dumps(bundle))
    rows = normalize_schema(read_raw(target))
    values = {(r["series_id"], r["period_end"]): r["value"] for r in rows}
    assert values[("tdc_base_bank_only_ru_flow", "2026-06-30")] == "-2.25"
    assert values[("bank_depository_tsy_tx", "2026-03-31")] == "10.0"
    assert values[("gdp_deflator", "2026-03-31")] == "" and values[("gdp_deflator", "2026-06-30")] == "131.5"
    _write_rows(processed / "tdc_components.csv", ["date", "bank_depository_tsy_tx"], [(dates[1], "-4.0")])
    with pytest.raises(ValueError, match="do not align"):
        refresh.reconstruct_site_bundle(processed, generated_at="x", input_hashes={})


def test_observation_cutoff_drops_only_later_dates(refresh):
    rows = [{"date": "2026-06-30", "value": "1"}, {"date": "2026-07-01", "value": "."}, {"date": "2026-10-01", "value": "2"}]
    kept, dropped = refresh.cut_observations(rows, date(2026, 6, 30))
    assert kept == [{"date": "2026-06-30", "value": "1"}] and dropped == 2


def test_factor_sources_map_to_fred_series(refresh):
    columns = [{"feature_id": "d__dgs5_dgs5__lag001", "source_file": "FRED_DGS5_DGS5.csv"},
               {"feature_id": "q__tgdef_net_gov_sav__lag002", "source_file": "FRED_TGDEF_net_gov_sav.csv"},
               {"feature_id": "m__x__lag001", "source_file": "local_x.csv"}]
    assert refresh.factor_source_files(["d__dgs5_dgs5__lag001", "q__tgdef_net_gov_sav__lag002"], columns) == {
        "FRED_DGS5_DGS5.csv": "DGS5", "FRED_TGDEF_net_gov_sav.csv": "TGDEF"}
    with pytest.raises(ValueError, match="not a FRED seed series"):
        refresh.factor_source_files(["m__x__lag001"], columns)


def test_open02_availability_blocks_on_absent_allowlisted_series(refresh):
    def archive(omit: set[str]) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as handle:
            lines = [f"{s.board_series_id}\t{s.official_title}\tLine 1\tS\tunits" for s in OPEN02_CONTRACT.series
                     if s.board_series_id not in omit]
            handle.writestr("data_dictionary/S122_1_t_tu.txt", "\n".join(lines))
        return buffer.getvalue()

    complete = refresh.open02_availability(archive(set()), {"series": []})
    assert complete["disposition"] == "available" and complete["present"] == 20
    blocked = refresh.open02_availability(archive({"FU764023005.Q"}), {"series": []})
    assert blocked["disposition"] == "blocked_allowlisted_series_absent"
    assert blocked["absent_series_ids"] == ["FU764023005.Q"]


def test_refreshed_series_keep_their_admitted_observation_window(refresh, tmp_path):
    admitted = tmp_path / "BOGZ1FL763061100Q.csv"
    _write_rows(admitted, ["date", "value"], [("1945-10-01", "1"), ("1946-01-01", "2")])
    assert refresh.admitted_start(admitted, "1980-01-01") == "1945-10-01"
    assert refresh.admitted_start(tmp_path / "absent.csv", "1980-01-01") == "1980-01-01"


def test_retired_upstream_series_are_carried_without_overwriting_refreshed_values(refresh):
    def row(series_id, period, value):
        return {"series_id": series_id, "period_end": period, "value": value}

    refreshed = [row("tdcpass_strict_loan_core_min_qoq", "2026-06-30", "92.9"),
                 row("tdcpass_strict_loan_core_min_qoq", "2025-12-31", "106.051")]
    admitted = [row("tdcpass_strict_loan_core_min_qoq", "2025-12-31", "106.278"),
                row("tdcpass_domestic_nonbank_deposits_qoq", "2025-12-31", "12.0")]
    merged, retired = refresh.carry_retired_series(refreshed, admitted)
    assert retired == ["tdcpass_domestic_nonbank_deposits_qoq"]
    values = {(r["series_id"], r["period_end"]): r["value"] for r in merged}
    assert values[("tdcpass_strict_loan_core_min_qoq", "2025-12-31")] == "106.051"
    assert values[("tdcpass_domestic_nonbank_deposits_qoq", "2025-12-31")] == "12.0"
    assert ("tdcpass_domestic_nonbank_deposits_qoq", "2026-06-30") not in values


# ------------------------------------------------- acquisition reuse and TDCest receipt addendum

COMMIT = "c" * 40
FRED_ROWS = [{"date": "2026-03-01", "value": "1.0"}, {"date": "2026-06-01", "value": "2.0"},
             {"date": "2026-09-01", "value": "3.0"}]
PROCESSED = {
    "data/processed/tdc_estimates.csv": "date,tdc_base_bank_only_ru_flow\n2026-03-31,1.5\n2026-06-30,-2.25\n",
    "data/processed/tdc_components.csv": "date,bank_depository_tsy_tx\n2026-03-31,10.0\n2026-06-30,-4.0\n",
    "data/processed/tdc_corrections.csv": "date,c\n2026-03-31,0.0\n2026-06-30,0.0\n",
    "data/processed/quarterly_inputs.csv": "date,gdp_deflator,nominal_gdp_saar_bil\n2026-06-30,131.5,31500.0\n",
}
# The correction changes the Treasury cash term, so estimates and components change; the rest do not.
CORRECTED = {
    "data/processed/tdc_estimates.csv": "date,tdc_base_bank_only_ru_flow\n2026-03-31,1.25\n2026-06-30,-2.5\n",
    "data/processed/tdc_components.csv": "date,bank_depository_tsy_tx\n2026-03-31,9.75\n2026-06-30,-4.25\n",
}
ADDENDUM = "RECEIPT_ADDENDUM_correction.json"


def _put(base: Path, locator: str, data: str | bytes) -> str:
    path = base / locator
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    return sha256_file(path)


def _z1_archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        handle.writestr("data_dictionary/S122_1_t_tu.txt", "\n".join(
            f"{s.board_series_id}\t{s.official_title}\tLine 1\tS\tunits" for s in OPEN02_CONTRACT.series))
    return buffer.getvalue()


class _AcquireTree:
    """An EA-TDC root whose earlier acquisition ran on pre-correction TDCest objects.

    TDCest then corrected two outputs in place, kept its original receipt and
    listed the regenerated outputs in an addendum, as the 2026q2 correction did.
    """

    def __init__(self, base: Path, refresh, monkeypatch):
        self.root, self.tdcest, self.refresh, self.monkeypatch = base / "ea", base / "tdcest", refresh, monkeypatch
        root = self.root
        self.payload = json.loads((ROOT / CONFIG).read_text())
        seeds = {"FRED_DGS5_DGS5.csv": "date,value\n2026-03-01,4.0\n", "local_carried.csv": "date,value\n2025-12-01,7.0\n"}
        raw = {name: _put(root, f"data/seed/interpol/raw/{name}", text) for name, text in seeds.items()}
        pins = {"ordered_top100": ("pins/top100.json", json.dumps(["d__dgs5_dgs5__lag001"])),
                "universe_columns": ("pins/universe.csv", "feature_id,source_file\nd__dgs5_dgs5__lag001,FRED_DGS5_DGS5.csv\n"),
                "raw_inventory": ("pins/inventory.json", json.dumps({"raw_inventory_provenance_only_not_consumed": [
                    {"path": f"data/seed/interpol/raw/{name}", "sha256": digest} for name, digest in raw.items()]}))}
        audit = {"strict_source_channels": [{"channel": channel, "series_id": series_id, "end_quarter": "2026Q2"}
                 for channel, series_id in self.payload["tdcpass"]["strict_loan_series"].items()]}
        self.archive = _z1_archive()
        self.payload.update({
            "fred": {**self.payload["fred"], "series_manifest": {
                "path": "config/fred_manifest.csv", "sha256": _put(root, "config/fred_manifest.csv", "series_id\nAAA\nGDP\n")}},
            "factor_features": {**{k: {"path": p, "sha256": _put(root, p, text)} for k, (p, text) in pins.items()},
                                "raw_seed_dir": "data/seed/interpol/raw"},
            "carried_bundles": {"data/bundles/qrawatch/standardized_series.csv":
                                _put(root, "data/bundles/qrawatch/standardized_series.csv", "series_id,period_end,value\n")},
            "tdcpass": {**self.payload["tdcpass"], "root": "../tdcpass", "files": {
                "output/models/proxy_unit_audit.json": _put(base / "tdcpass", "output/models/proxy_unit_audit.json", json.dumps(audit))},
                "admitted_bundle": {"path": "data/bundles/tdcpass/standardized_series.csv",
                                    "sha256": _put(root, "data/bundles/tdcpass/standardized_series.csv", "series_id\n")}},
            "z1_release": {**self.payload["z1_release"], "archive_sha256": hashlib.sha256(self.archive).hexdigest()},
            "open02": {"admitted_source_bundle": {"path": "data/bundles/open02/admitted.json",
                                                  "sha256": _put(root, "data/bundles/open02/admitted.json", json.dumps({"series": []}))}},
        })
        self.original = {loc: _put(self.tdcest, loc, text) for loc, text in PROCESSED.items()}
        listed = ("data/processed/tdc_estimates.csv", "data/processed/tdc_components.csv", "data/processed/quarterly_inputs.csv")
        receipt = {"schema_version": "tdcest_refresh_receipt_v1", "vintage": "2026q2",
                   "producers": {"tdcest_exports_and_acquisition": "a" * 40 + " (main)"},
                   "outputs": {loc: {"sha256": self.original[loc], "bytes": len(PROCESSED[loc])} for loc in listed}}
        self.receipt_sha256 = _put(self.tdcest, "RECEIPT.json", json.dumps(receipt))
        monkeypatch.setattr(refresh, "ROOT", root)
        monkeypatch.setattr(refresh, "_verify_clean_commit", lambda root, commit: commit)
        monkeypatch.setenv(VINTAGE_ENV, CONFIG)

        # The earlier acquisition: live lane on the pre-correction objects, with a recorded fake network.
        self.network: list[str] = []
        monkeypatch.setattr(refresh, "_fred_series", self._fake_series)
        monkeypatch.setattr(refresh.urllib.request, "urlopen", self._fake_urlopen)
        monkeypatch.setenv("FRED_API_KEY", "test-key")
        self.main("config/vintage_old.json", "output/refresh_old", self.tdcest_pins(self.original))
        self.old = root / "output/refresh_old"
        self.old_receipt_sha256 = sha256_file(self.old / "inputs/acquisition_receipt.json")

        # The correction: outputs regenerated in place, original receipt untouched, addendum alongside.
        self.corrected = {**self.original, **{loc: _put(self.tdcest, loc, text) for loc, text in CORRECTED.items()}}
        self.addendum_sha256 = self.write_addendum()

    def _fake_series(self, series_id, key, start):
        self.network.append(series_id)
        return [dict(row) for row in FRED_ROWS], {"series_id": series_id, "last_updated": "2026-10-01 10:00:00-05",
                                                  "realtime_start": "2026-10-14", "observation_end": "2026-09-01"}

    def _fake_urlopen(self, request, timeout=None):
        self.network.append(request.full_url)
        return contextlib.nullcontext(io.BytesIO(self.archive))

    def forbid_network(self) -> list[str]:
        calls: list[str] = []

        def forbidden(*args, **kwargs):
            calls.append(repr(args[:1]))
            raise AssertionError("network call during a reused acquisition")

        for owner, name in ((self.refresh, "_fred_json"), (self.refresh, "_fred_series"),
                            (self.refresh.urllib.request, "urlopen")):
            self.monkeypatch.setattr(owner, name, forbidden)
        self.monkeypatch.delenv("FRED_API_KEY", raising=False)
        return calls

    def write_addendum(self, **changes) -> str:
        addendum = {"schema_version": "tdcest_refresh_receipt_addendum_v1", "vintage": "2026q2",
                    "correction": "Treasury-Fed internal transfers removed from the Treasury cash term",
                    "producers": {"step_commits": {"t03_estimate": "e" * 40}},
                    "outputs": {loc: {"sha256": self.corrected[loc], "bytes": len(text), "status": "regenerated",
                                      "superseded_sha256": self.original[loc]} for loc, text in CORRECTED.items()},
                    **changes}
        return _put(self.tdcest, ADDENDUM, json.dumps(addendum))

    def tdcest_pins(self, files: dict[str, str], addendum: dict | None = None) -> dict:
        tdcest = {**self.payload["tdcest"], "root": "../tdcest", "receipt_sha256": self.receipt_sha256, "files": dict(files)}
        if addendum is not None:
            tdcest["receipt_addendum"] = addendum
        return tdcest

    def corrected_pins(self, **addendum) -> dict:
        return self.tdcest_pins(self.corrected, {"path": ADDENDUM, "sha256": self.addendum_sha256, **addendum})

    def main(self, locator: str, output_root: str, tdcest: dict, **extra) -> Path:
        payload = {**self.payload, "output_root": output_root, "tdcest": tdcest, **extra}
        _put(self.root, locator, json.dumps(payload))
        _put(self.root, f"{output_root}/protected_baseline.json", "{}")
        self.monkeypatch.setenv(VINTAGE_ENV, locator)
        self.refresh.main(["acquire", "--vintage", locator, "--producer-commit", COMMIT])
        return self.root / output_root / "inputs/acquisition_receipt.json"

    def reuse(self, tdcest: dict | None = None, output_root: str = "output/refresh_new", **pin) -> Path:
        declared = {"root": "output/refresh_old", "receipt_sha256": self.old_receipt_sha256,
                    "vintage_config": "config/vintage_old.json", **pin}
        return self.main("config/vintage_new.json", output_root, tdcest or self.corrected_pins(),
                         reuse_acquisition=declared)


@pytest.fixture
def tree(refresh, tmp_path, monkeypatch):
    return _AcquireTree(tmp_path, refresh, monkeypatch)


def _reused_files(receipt: dict) -> list[str]:
    return [*(r["path"] for r in receipt["fred_raw"].values()), *(r["path"] for r in receipt["factor_seed"]["files"].values()),
            receipt["open02_source_availability"]["path"]]


def test_reused_acquisition_copies_every_input_byte_for_byte_with_its_records(tree):
    old = json.loads((tree.old / "inputs/acquisition_receipt.json").read_text())
    assert old["vintage"]["locator"] == "config/vintage_old.json"  # another config, same sample and tiers
    receipt_path = tree.reuse()
    new = json.loads(receipt_path.read_text())
    output = receipt_path.parents[1]
    assert new["vintage"]["locator"] == "config/vintage_new.json"
    for key in ("fred_raw", "factor_seed", "open02_source_availability"):
        assert new[key] == old[key], key
    for locator in _reused_files(new):
        assert (output / locator).read_bytes() == (tree.old / locator).read_bytes(), locator
    assert new["reused_acquisition"] == {
        "root": "output/refresh_old", "receipt_sha256": sha256_file(tree.old / "inputs/acquisition_receipt.json"),
        "vintage_config": {"path": "config/vintage_old.json", "sha256": old["vintage"]["sha256"]}}
    # Only TDCest objects differ from the earlier acquisition.
    assert new["tdcest"]["files"] != old["tdcest"]["files"] and new["carried_bundles"] == old["carried_bundles"]
    vintage = load_vintage(tree.root, "config/vintage_new.json")
    assert tree.refresh._verify_acquisition(vintage)["reused_acquisition"] == new["reused_acquisition"]


def test_reused_acquisition_makes_no_network_call_and_needs_no_fred_key(tree):
    calls = tree.forbid_network()
    tree.monkeypatch.setenv("FRED_API_KEY", "test-key")
    with pytest.raises(AssertionError, match="network call"):
        # The live lane on the same configuration would fetch; the guard sees it.
        tree.main("config/vintage_new.json", "output/refresh_new", tree.corrected_pins())
    assert calls and not (tree.root / "output/refresh_new/inputs").exists()
    calls.clear()
    tree.monkeypatch.delenv("FRED_API_KEY")
    assert tree.reuse().is_file()
    assert calls == []


@pytest.mark.parametrize("target", ["fred", "seed", "open02", "missing"])
def test_tampered_or_missing_reused_input_raises_and_installs_nothing(tree, target):
    tree.forbid_network()
    locator = {"fred": "inputs/fred/AAA.csv", "seed": "inputs/interpol_raw/FRED_DGS5_DGS5.csv",
               "open02": "inputs/open02_source_availability.json", "missing": "inputs/interpol_raw/local_carried.csv"}[target]
    path = tree.old / locator
    if target == "missing":
        path.unlink()
    else:
        data = bytearray(path.read_bytes())
        data[-2] = ord("9") if data[-2] != ord("9") else ord("8")  # same size, different bytes
        path.write_bytes(bytes(data))
    with pytest.raises((ValueError, FileNotFoundError), match="Pinned input changed|missing"):
        tree.reuse()
    assert not any((tree.root / "output/refresh_new" / name).exists() for name in ("inputs", "tdcest", "tdcpass"))


@pytest.mark.parametrize("case", ["substituted", "joint_edit", "unauthenticated_config"])
def test_reuse_takes_only_the_pinned_acquisition_under_its_own_configuration(tree, case):
    # The correction config pins the earlier acquisition by hash; nothing else may stand in for it.
    pin, message = {}, "differs from the pinned receipt"
    if case == "substituted":
        # Another acquisition of the same sample, made the same way, is still not the pinned one.
        alt = tree.main("config/vintage_alt.json", "output/refresh_alt", tree.corrected_pins())
        assert json.loads(alt.read_text())["vintage"]["sample"] == json.loads(
            (tree.old / "inputs/acquisition_receipt.json").read_text())["vintage"]["sample"]
        assert sha256_file(alt) != tree.old_receipt_sha256
        pin = {"root": "output/refresh_alt"}
    elif case == "joint_edit":
        # FRED bytes changed together with their receipt record are a different acquisition.
        path = tree.old / "inputs/fred/AAA.csv"
        path.write_text(path.read_text().replace("2.0", "2.5"))
        receipt = json.loads((tree.old / "inputs/acquisition_receipt.json").read_text())
        receipt["fred_raw"]["AAA"].update(sha256=sha256_file(path), bytes=path.stat().st_size)
        (tree.old / "inputs/acquisition_receipt.json").write_text(json.dumps(receipt))
    else:
        old = tree.root / "config/vintage_old.json"
        old.write_text(old.read_text().replace('"output/refresh_old"', '"output/refresh_moved"'))
        message = "own vintage configuration cannot be authenticated"
    tree.forbid_network()
    with pytest.raises(ValueError, match=message):
        tree.reuse(**pin)
    assert not any((tree.root / "output/refresh_new" / name).exists() for name in ("inputs", "tdcest", "tdcpass"))


def _write_pin(tree, locator: str, text: str) -> dict:
    return {"path": locator, "sha256": _put(tree.root, locator, text)}


@pytest.mark.parametrize("case, field", [
    ("sample", "method_tier_counts', 'observation_cutoff', 'sample"),
    ("feature_lags", "factor_features"),
    ("z1_release", "z1_release"),
    ("fred_start", "fred"),
    ("open02_bundle", "open02"),
    ("tdcest_root", "tdcest.root"),
])
def test_reuse_refuses_a_configuration_that_differs_beyond_the_correction(tree, case, field):
    # Reused FRED, factor-seed and OPEN-02 inputs are valid only for the configuration that
    # acquired them; a correction may change TDCest objects and nothing analytical.
    tree.forbid_network()
    tdcest = tree.corrected_pins()
    if case == "sample":
        tree.payload.update(sample={"start": "2002Q1", "end": "2026Q1", "n": 97}, observation_cutoff="2026-03-31",
                            method_tier_counts={**tree.payload["method_tier_counts"], "constrained_component": 17})
    elif case == "feature_lags":
        # Another lag of the same raw series: the raw-source mapping is identical, the features are not.
        factor = dict(tree.payload["factor_features"])
        factor["ordered_top100"] = _write_pin(tree, "pins/top100_lag2.json", json.dumps(["d__dgs5_dgs5__lag002"]))
        factor["universe_columns"] = _write_pin(tree, "pins/universe_lag2.csv",
                                                "feature_id,source_file\nd__dgs5_dgs5__lag002,FRED_DGS5_DGS5.csv\n")
        old_sources = tree.refresh.factor_source_files(["d__dgs5_dgs5__lag001"], [{"feature_id": "d__dgs5_dgs5__lag001", "source_file": "FRED_DGS5_DGS5.csv"}])
        new_sources = tree.refresh.factor_source_files(["d__dgs5_dgs5__lag002"], [{"feature_id": "d__dgs5_dgs5__lag002", "source_file": "FRED_DGS5_DGS5.csv"}])
        assert old_sources == new_sources
        tree.payload["factor_features"] = factor
    elif case == "z1_release":
        tree.payload["z1_release"] = {**tree.payload["z1_release"], "release_date": "2026-12-11"}
    elif case == "fred_start":
        tree.payload["fred"] = {**tree.payload["fred"], "raw_observation_start": "1990-01-01"}
    elif case == "open02_bundle":
        other = {"series": [{"key": OPEN02_CONTRACT.series[0].key, "observations": [{"date": "2025-12-31"}]}]}
        tree.payload["open02"] = {"admitted_source_bundle": _write_pin(tree, "data/bundles/open02/other.json", json.dumps(other))}
    else:
        shutil.copytree(tree.tdcest, tree.root.parent / "tdcest_copy")
        tdcest["root"] = "../tdcest_copy"
    with pytest.raises(ValueError, match=rf"beyond the correction: \['{field}'\]"):
        tree.reuse(tdcest)
    assert not any((tree.root / "output/refresh_new" / name).exists() for name in ("inputs", "tdcest", "tdcpass"))
    assert not list((tree.root / "output/refresh_new").glob(".acquire-*"))


def test_a_tdcest_receipt_of_another_vintage_is_refused(tree):
    receipt = json.loads((tree.tdcest / "RECEIPT.json").read_text())
    tree.receipt_sha256 = _put(tree.tdcest, "RECEIPT.json", json.dumps({**receipt, "vintage": "2026q1"}))
    with pytest.raises(ValueError, match="refresh receipt was made for another vintage"):
        tree.main("config/vintage_live.json", "output/refresh_live", tree.corrected_pins())
    assert not (tree.root / "output/refresh_live/inputs").exists()


def test_addendum_binds_the_corrected_tdcest_objects(tree):
    tree.forbid_network()
    receipt = json.loads(tree.reuse().read_text())
    bound = {loc: record["bound_by"] for loc, record in receipt["tdcest"]["files"].items()}
    assert bound == {"data/processed/tdc_estimates.csv": "tdcest_receipt_addendum",
                     "data/processed/tdc_components.csv": "tdcest_receipt_addendum",
                     "data/processed/quarterly_inputs.csv": "tdcest_receipt",
                     "data/processed/tdc_corrections.csv": "vintage_pin"}
    assert receipt["tdcest"]["receipt_addendum"]["sha256"] == tree.addendum_sha256
    copied = tree.root / "output/refresh_new/tdcest" / ADDENDUM
    assert sha256_file(copied) == tree.addendum_sha256
    # The reconstructed site bundle carries the corrected values, not the superseded ones.
    bundle = json.loads((tree.root / "output/refresh_new/tdcest/site/data/bundle.json").read_text())
    assert bundle["estimates"]["tdc_base_bank_only_ru_flow"] == [1.25, -2.5]


def test_carried_inputs_of_the_tdcest_receipt_are_checked_and_keep_their_upstream_binding(refresh):
    # A variant root copies some objects unchanged and lists them as carried inputs with the record
    # that binds them upstream; the pin must match that entry, and the binding travels with the file.
    pins = {"data/processed/series.csv": "a" * 64, "data/processed/tdc_corrections.csv": "b" * 64,
            "data/processed/other.csv": "c" * 64}
    listed = {"data/processed/series.csv": {"sha256": "a" * 64}}
    carried = {"data/processed/tdc_corrections.csv": {"sha256": "b" * 64, "source": "refresh_2026q2/x",
                                                      "bound_by": "downstream vintage pin"},
               "copied_tree": {"path": "logs/copied_tree.json", "sha256": "d" * 64}}
    records = refresh.bind_tdcest_files(pins, listed, {}, carried)
    assert {k: v["bound_by"] for k, v in records.items()} == {
        "data/processed/series.csv": "tdcest_receipt", "data/processed/tdc_corrections.csv": "tdcest_receipt_carried_input",
        "data/processed/other.csv": "vintage_pin"}
    assert records["data/processed/tdc_corrections.csv"]["upstream_binding"] == {
        "source": "refresh_2026q2/x", "bound_by": "downstream vintage pin"}
    with pytest.raises(ValueError, match="carried input and vintage pin disagree: data/processed/tdc_corrections.csv"):
        refresh.bind_tdcest_files({**pins, "data/processed/tdc_corrections.csv": "e" * 64}, listed, {}, carried)


def test_corrected_objects_need_the_addendum_because_the_original_receipt_binds_superseded_bytes(tree):
    tree.forbid_network()
    with pytest.raises(ValueError, match="receipt and vintage pin disagree: data/processed/tdc_estimates.csv"):
        tree.reuse(tree.tdcest_pins(tree.corrected))


@pytest.mark.parametrize("case", ["vintage", "schema", "commits", "foreign", "addendum_hash", "stale_pin",
                                  "outside_root", "drive_relative"])
def test_addendum_fails_closed(tree, case):
    tree.forbid_network()
    tdcest = tree.corrected_pins()
    message = "addendum schema, vintage, step commits or outputs differ"
    if case == "commits":
        tdcest["receipt_addendum"]["sha256"] = tree.write_addendum(producers={})
    elif case == "foreign":
        # An addendum from another refresh run supersedes bytes the original receipt never listed.
        outputs = {loc: {"sha256": tree.corrected[loc], "bytes": len(text), "superseded_sha256": "1" * 64}
                   for loc, text in CORRECTED.items()}
        tdcest["receipt_addendum"]["sha256"] = tree.write_addendum(outputs=outputs)
        message = "does not supersede this receipt's outputs"
    elif case == "vintage":
        tdcest["receipt_addendum"]["sha256"] = tree.write_addendum(vintage="2026q3")
    elif case == "schema":
        tdcest["receipt_addendum"]["sha256"] = tree.write_addendum(schema_version="tdcest_refresh_receipt_addendum_v2")
    elif case == "addendum_hash":
        tree.write_addendum(correction="edited after pinning")
        message = "Pinned input changed"
    elif case == "stale_pin":
        tdcest["files"]["data/processed/tdc_components.csv"] = tree.original["data/processed/tdc_components.csv"]
        message = "receipt and vintage pin disagree: data/processed/tdc_components.csv"
    else:
        tdcest["receipt_addendum"]["path"] = f"../tdcest/{ADDENDUM}" if case == "outside_root" else f"C:{ADDENDUM}"
        message = "addendum must be a plain relative locator"
    with pytest.raises(ValueError, match=message):
        tree.reuse(tdcest)
    assert not (tree.root / "output/refresh_new/inputs").exists()


def test_corrected_vintage_changes_only_tdcest_objects_and_can_reuse_the_acquisition(refresh, vintage):
    # The correction reruns the same sample on corrected TDCest objects; everything else, including
    # the FRED and factor-seed acquisition it reuses, must be the 2026q2 vintage's.
    corrected = load_vintage(ROOT, "config/vintage_2026q2_tfx.json")
    old, new = vintage.payload, corrected.payload
    differs = {k for k in set(old) | set(new) if old.get(k) != new.get(k)}
    assert differs == {"authority", "output_root", "tdcest", "reuse_acquisition"} <= set(refresh.REUSE_ALLOWED_DIFFERENCES) | {"authority"}
    assert {k for k in old["tdcest"] if old["tdcest"][k] != new["tdcest"].get(k)} | (set(new["tdcest"]) - set(old["tdcest"])) \
        <= set(refresh.REUSE_ALLOWED_TDCEST_DIFFERENCES)
    reuse = new["reuse_acquisition"]
    assert set(reuse) == {"root", "receipt_sha256", "vintage_config"} and len(reuse["receipt_sha256"]) == 64
    assert reuse["root"] == old["output_root"] and reuse["vintage_config"].startswith(f"{reuse['root']}/")
    assert corrected.vintage_id == vintage.vintage_id and new["output_root"] != old["output_root"]
    tdcest = new["tdcest"]
    assert (tdcest["root"], tdcest["receipt_sha256"]) == (old["tdcest"]["root"], old["tdcest"]["receipt_sha256"])
    assert set(tdcest["files"]) == set(old["tdcest"]["files"]) and tdcest["files"] != old["tdcest"]["files"]
    assert Path(tdcest["receipt_addendum"]["path"]).name == tdcest["receipt_addendum"]["path"]


# ------------------------------------------------- deposit outcome aligned with the treatment's bank group

ALIGNED = "config/vintage_2026q2_tfx_aligned.json"
TREATMENT_BANK_GROUP = ("BOGZ1FL764100005Q", "BOGZ1FL754100005Q", "BOGZ1FL744100005Q")
CREDIT_UNIONS = "BOGZ1FL474100005Q"


def test_aligned_vintage_changes_only_the_deposit_outcome_of_the_corrected_vintage(refresh):
    # The aligned rerun is the corrected vintage with one change: the outcome sums deposits over
    # the bank group the treatment treats as paying with reserves. Credit unions are not in that group.
    corrected = load_vintage(ROOT, "config/vintage_2026q2_tfx.json").payload
    aligned = load_vintage(ROOT, ALIGNED)
    differs = {k for k in set(corrected) | set(aligned.payload) if corrected.get(k) != aligned.payload.get(k)}
    assert differs == {"authority", "output_root", "deposit_outcome"}
    assert "deposit_outcome" in refresh.REUSE_ALLOWED_DIFFERENCES
    declared = deposit_outcome(aligned)
    assert tuple(declared["series"]) == TREATMENT_BANK_GROUP and CREDIT_UNIONS not in declared["series"]
    assert declared["admitted_series"] == RAW_FRED_LEVEL_CHANGE_SERIES[CANONICAL_OUTCOME_ID]
    assert declared["outcome_id"] == CANONICAL_OUTCOME_ID
    assert deposit_outcome(load_vintage(ROOT, "config/vintage_2026q2_tfx.json")) is None
    source = declared["source"]
    assert set(source) <= {"root", "receipt_sha256", "files"}
    if "files" in source:
        assert set(source["files"]) == set(TREATMENT_BANK_GROUP) and len(source["receipt_sha256"]) == 64


@pytest.mark.parametrize("change, value", [
    ("outcome_id", "domestic_nonbank_deposits_qoq"),
    ("admitted_series", "BOGZ1FL474100005Q"),
    ("series", ["BOGZ1FL764100005Q", "BOGZ1FL764100005Q"]),
    ("series", ["BOGZ1FL764100005Q"]),
    ("perimeter", " "),
])
def test_a_deposit_outcome_declaration_that_drifts_from_the_canonical_outcome_is_refused(tmp_path, change, value):
    payload = json.loads((ROOT / ALIGNED).read_text())
    payload["deposit_outcome"][change] = value
    path = tmp_path / "vintage.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Declared deposit outcome"):
        deposit_outcome(load_vintage(tmp_path, "vintage.json"))


def _design_project(tmp_path: Path, levels: dict[str, list[tuple[str, str]]], config: str = ALIGNED,
                    tdcest_rows: tuple[str, ...] = ()) -> Path:
    from test_design_builder import _write_text

    _write_text(tmp_path / "config/dass_job_blueprint.yaml",
                "jobs:\n  - job_id: baseline_tdc_lp_deposits\n    estimator: lp\n    horizons: [0]\n")
    header = ("series_id,series_label,source_family,source_repo,source_table,freq,period_end,release_date,available_at,"
              "vintage_policy,units,value,transform_default,seasonal_adjustment_flag,interpolated_flag,component_group,role,notes")
    _write_text(tmp_path / "data/bundles/tdcest/standardized_series.csv", "\n".join([
        header,
        "tdc_base_bank_only_ru_flow,t,repo_seed_bundle,tdcest,estimates,quarterly,2024-03-31,2024-06-29,2024-06-29,p,"
        "usd_millions,100,none,unknown,false,canonical_headline,treatment,primary_treatment",
        *(f"{series_id},t,repo_processed_csv,tdcest,tier2_regression_series,quarterly,2024-03-31,2024-06-29,"
          f"2024-06-29,p,usd_millions,{value},none,unknown,false,g,treatment,n"
          for series_id, value in (row.split("=") for row in tdcest_rows))]))
    for series_id, rows in levels.items():
        _write_text(tmp_path / f"data/raw/fred/{series_id}.csv", "\n".join(["date,value", *(f"{d},{v}" for d, v in rows)]))
    _write_text(tmp_path / config, (ROOT / config).read_text())
    return tmp_path


def test_aligned_outcome_is_the_quarterly_change_of_the_summed_deposit_levels(tmp_path, monkeypatch):
    # Summing levels before differencing keeps the outcome a change in one balance sheet aggregate,
    # and the residual outcome follows it, so the treatment is unchanged and only the outcome widens.
    from ea_tdc.designs.quarterly import build_quarterly_design
    from ea_tdc.paths import ensure_repo_dirs

    root = _design_project(tmp_path, {
        "BOGZ1FL764100005Q": [("2023-10-01", "900"), ("2024-01-01", "1030")],
        "BOGZ1FL754100005Q": [("2023-10-01", "100"), ("2024-01-01", "150")],
        "BOGZ1FL744100005Q": [("2023-10-01", "10"), ("2024-01-01", "12")],
        CREDIT_UNIONS: [("2023-10-01", "500"), ("2024-01-01", "900")],
    })
    paths = project_paths(root)
    ensure_repo_dirs(paths)
    monkeypatch.delenv(VINTAGE_ENV, raising=False)
    admitted = {r["quarter"]: r for r in csv.DictReader(build_quarterly_design(paths, job_id="baseline_tdc_lp_deposits")
                                                         .bundle_path.open())}
    assert admitted["2024Q1"]["matched_total_deposits"] == "130.0"
    monkeypatch.setenv(VINTAGE_ENV, ALIGNED)
    aligned = {r["quarter"]: r for r in csv.DictReader(build_quarterly_design(paths, job_id="baseline_tdc_lp_deposits")
                                                        .bundle_path.open())}
    assert aligned["2024Q1"]["matched_total_deposits"] == "182.0"  # (1030+150+12) - (900+100+10); credit unions out
    assert aligned["2024Q1"]["other_component_qoq"] == "82.0"
    assert aligned["2024Q1"]["tdc_bank_only_qoq"] == admitted["2024Q1"]["tdc_bank_only_qoq"] == "100"


def test_a_quarter_missing_one_bank_group_raises_instead_of_breaking_the_level(tmp_path, monkeypatch):
    from ea_tdc.designs.quarterly import build_quarterly_design
    from ea_tdc.paths import ensure_repo_dirs

    root = _design_project(tmp_path, {
        "BOGZ1FL764100005Q": [("2023-10-01", "900"), ("2024-01-01", "1030")],
        "BOGZ1FL754100005Q": [("2023-10-01", "100"), ("2024-01-01", "150")],
        "BOGZ1FL744100005Q": [("2023-10-01", "10"), ("2024-01-01", ".")],
    })
    paths = project_paths(root)
    ensure_repo_dirs(paths)
    monkeypatch.setenv(VINTAGE_ENV, ALIGNED)
    with pytest.raises(ValueError, match="do not all report 2024Q1"):
        build_quarterly_design(paths, job_id="baseline_tdc_lp_deposits")


CANONICAL = "config/vintage_2026q2_canonical.json"
CU_TREATMENT = "tdc_tier2_regression_mmf_rrp_prop_depository_institution_np_corp_cu_ru_flow"
OPEN02_CONTRACT_MMF = ("tdc_tier2_regression_mmf_rrp_lb_bank_only_qoq", "tdc_tier2_regression_mmf_rrp_ub_bank_only_qoq")
CANONICAL_TERMS = [("BOGZ1FL703127005Q", 1), ("BOGZ1FL703130005Q", 1), ("BOGZ1FL763123005Q", -1),
                   ("BOGZ1FL313030003Q", -1)]


def test_canonical_vintage_changes_only_the_outcome_and_the_treatment_column_of_the_corrected_vintage():
    corrected = load_vintage(ROOT, "config/vintage_2026q2_tfx.json").payload
    canonical = load_vintage(ROOT, CANONICAL)
    payload = canonical.payload
    differs = {k for k in set(corrected) | set(payload) if corrected.get(k) != payload.get(k)}
    assert differs == {"authority", "output_root", "deposit_outcome", "treatment", "tdcest"}
    # Same frozen design: controls, factors, sample, tiers and the reused acquisition are untouched.
    for key in ("sample", "method_tier_counts", "factor_features", "fred", "z1_release", "reuse_acquisition"):
        assert payload[key] == corrected[key], key
    declared = deposit_outcome(canonical)
    assert deposit_outcome_terms(declared) == CANONICAL_TERMS
    # The admitted outcome series is fetched only to prove the release, never summed.
    assert deposit_outcome_fetch(declared) == [*(s for s, _ in CANONICAL_TERMS), "BOGZ1FL764100005Q"]
    treatment = declared_treatment(canonical)
    assert treatment["source_series"] == CU_TREATMENT and treatment["tdcest_treatment"] == "depository_cu"
    assert treatment["treatment_id"] == CANONICAL_TREATMENT_ID
    # The TDCest root, its receipt and generation time change, and the same nine objects are pinned:
    # the credit-union interest fix regenerates the estimate chain behind all of them.
    before, after = corrected["tdcest"], payload["tdcest"]
    assert {k for k in set(before) | set(after) if before.get(k) != after.get(k)} == {
        "root", "receipt_sha256", "files", "receipt_addendum", "generated_at"}
    assert set(before["files"]) == set(after["files"])
    # The route bounds use the same institutional grouping as the central treatment.
    overrides = treatment_series_overrides(canonical)
    assert overrides == {CANONICAL_TREATMENT_ID: CU_TREATMENT,
                         OPEN02_CONTRACT_MMF[0]: CU_TREATMENT.replace("_prop_", "_lb_"),
                         OPEN02_CONTRACT_MMF[1]: CU_TREATMENT.replace("_prop_", "_ub_")}


@pytest.mark.parametrize("change, value", [
    ("treatment_id", "tdc_bank_only_qoq"),
    ("admitted_source_series", CU_TREATMENT),
    ("source_series", CANONICAL_TREATMENT_SOURCE_SERIES),
    ("source_series", "BOGZ1FL703127005Q"),
    ("tdcest_treatment", ""),
    ("perimeter", " "),
    ("label", "Bank Long"),
    ("route_bound_source_series", {"lower": CU_TREATMENT, "upper": CU_TREATMENT.replace("_prop_", "_ub_")}),
    ("route_bound_source_series", {"lower": CU_TREATMENT.replace("_prop_", "_lb_")}),
])
def test_a_treatment_declaration_that_drifts_from_the_canonical_treatment_is_refused(tmp_path, change, value):
    payload = json.loads((ROOT / CANONICAL).read_text())
    payload["treatment"][change] = value
    (tmp_path / "vintage.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Declared treatment"):
        declared_treatment(load_vintage(tmp_path, "vintage.json"))


@pytest.mark.parametrize("signs", [[1, 1, -1], [1, 1, -1, 0], [1, 1, -1, True]])
def test_deposit_outcome_signs_must_pair_each_series_with_plus_or_minus_one(tmp_path, signs):
    payload = json.loads((ROOT / CANONICAL).read_text())
    payload["deposit_outcome"]["signs"] = signs
    (tmp_path / "vintage.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Declared deposit outcome"):
        deposit_outcome(load_vintage(tmp_path, "vintage.json"))


def test_canonical_outcome_nets_federal_government_deposits_and_uses_the_credit_union_treatment(tmp_path, monkeypatch):
    # Treasury's own deposits are not deposit users' deposits: a rise in them lowers the outcome.
    # The canonical treatment id keeps its name but reads the declared credit-union column.
    from ea_tdc.designs.quarterly import build_quarterly_design
    from ea_tdc.paths import ensure_repo_dirs

    root = _design_project(tmp_path, {
        "BOGZ1FL764100005Q": [("2023-10-01", "900"), ("2024-01-01", "1030")],
        "BOGZ1FL703127005Q": [("2023-10-01", "1000"), ("2024-01-01", "1100")],
        "BOGZ1FL703130005Q": [("2023-10-01", "2000"), ("2024-01-01", "2300")],
        "BOGZ1FL763123005Q": [("2023-10-01", "5"), ("2024-01-01", "25")],
        "BOGZ1FL313030003Q": [("2023-10-01", "2"), ("2024-01-01", "3")],
    }, config=CANONICAL, tdcest_rows=(f"{CANONICAL_TREATMENT_SOURCE_SERIES}=60", f"{CU_TREATMENT}=61.5",
                                      "tdc_tier2_regression_mmf_rrp_lb_bank_only_ru_flow=55",
                                      f"{CU_TREATMENT.replace('_prop_', '_lb_')}=56.5",
                                      "tdc_tier2_regression_mmf_rrp_ub_bank_only_ru_flow=65",
                                      f"{CU_TREATMENT.replace('_prop_', '_ub_')}=66.5"))
    paths = project_paths(root)
    ensure_repo_dirs(paths)
    admitted = {r["quarter"]: r for r in csv.DictReader(build_quarterly_design(paths, job_id="baseline_tdc_lp_deposits")
                                                         .bundle_path.open())}
    assert admitted["2024Q1"]["matched_total_deposits"] == "130.0"
    assert admitted["2024Q1"][CANONICAL_TREATMENT_ID] == "60"
    monkeypatch.setenv(VINTAGE_ENV, CANONICAL)
    canonical = {r["quarter"]: r for r in csv.DictReader(build_quarterly_design(paths, job_id="baseline_tdc_lp_deposits")
                                                          .bundle_path.open())}
    assert canonical["2024Q1"]["matched_total_deposits"] == "379.0"  # (100 + 300) less Treasury's +20 and +1
    assert canonical["2024Q1"][CANONICAL_TREATMENT_ID] == "61.5"
    # The money-fund route bounds follow the same grouping as the central treatment.
    assert admitted["2024Q1"][OPEN02_CONTRACT_MMF[0]] == "55" and canonical["2024Q1"][OPEN02_CONTRACT_MMF[0]] == "56.5"
    assert admitted["2024Q1"][OPEN02_CONTRACT_MMF[1]] == "65" and canonical["2024Q1"][OPEN02_CONTRACT_MMF[1]] == "66.5"
    assert canonical["2024Q1"]["tdc_bank_only_qoq"] == admitted["2024Q1"]["tdc_bank_only_qoq"] == "100"


def test_open01_records_the_declared_treatment_column_only_inside_the_canonical_run():
    admitted_contract = open01.OPEN_CONTRACT
    with open01_treatment(load_vintage(ROOT, CANONICAL)) as source:
        assert open01.CANONICAL_TREATMENT_SOURCE_SERIES == source == CU_TREATMENT
        assert open01.OPEN_CONTRACT.treatment_label == "regression_mmf_rrp_depository_cu_long"
        assert open01.OPEN_CONTRACT.mmf_lower_treatment_id == admitted_contract.mmf_lower_treatment_id
    assert open01.CANONICAL_TREATMENT_SOURCE_SERIES == CANONICAL_TREATMENT_SOURCE_SERIES
    assert open01.OPEN_CONTRACT is admitted_contract
    with open16_disclosures(load_vintage(ROOT, CANONICAL)) as disclosures:
        assert CU_TREATMENT in disclosures["perimeter"] and "corporate credit-union" in disclosures["perimeter"]
    with open01_treatment(load_vintage(ROOT, ALIGNED)) as source:
        assert source == CANONICAL_TREATMENT_SOURCE_SERIES


def test_open16_outputs_state_the_declared_perimeter_only_inside_the_aligned_run():
    aligned = load_vintage(ROOT, ALIGNED)
    admitted = dict(open16.DISCLOSURES)
    with open16_disclosures(aligned) as disclosures:
        assert open16.DISCLOSURES["perimeter"] == disclosures["perimeter"] == aligned.payload["deposit_outcome"]["perimeter"]
        assert {k: v for k, v in disclosures.items() if k != "perimeter"} == {k: v for k, v in admitted.items() if k != "perimeter"}
    assert open16.DISCLOSURES == admitted
    with open16_disclosures(load_vintage(ROOT, "config/vintage_2026q2_tfx.json")) as disclosures:
        assert disclosures == admitted


def _z1_deposit_archive(tables: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        for name, text in tables.items():
            handle.writestr(name, text)
    return buffer.getvalue()


def test_financial_accounts_levels_are_read_by_board_column_and_must_agree_across_tables(refresh):
    archive = _z1_deposit_archive({"csv/F2_s.csv": "date,FL754100005.Q\n2026:Q1,5\n2026:Q2,ND\n",
                                   "csv/L111.csv": "date,FL754100005.Q\n2026:Q1,5\n"})
    assert refresh.z1_levels(archive, "BOGZ1FL754100005Q") == {"2026Q1": 5.0}
    conflicting = _z1_deposit_archive({"csv/A.csv": "date,FL754100005.Q\n2026:Q1,5\n",
                                       "csv/B.csv": "date,FL754100005.Q\n2026:Q1,6\n"})
    with pytest.raises(ValueError, match="disagree"):
        refresh.z1_levels(conflicting, "BOGZ1FL754100005Q")
    with pytest.raises(ValueError, match="not in the Financial Accounts archive"):
        refresh.z1_levels(archive, "BOGZ1FL744100005Q")


DEPOSIT_LEVELS = {"BOGZ1FL764100005Q": ("19114679", "19188648"), "BOGZ1FL754100005Q": ("3108237", "3159915"),
                  "BOGZ1FL744100005Q": ("88486", "92227")}


def _fetch_setup(refresh, tmp_path, monkeypatch, z1_744_q2="92227"):
    columns = list(DEPOSIT_LEVELS)
    values = {s: list(v) for s, v in DEPOSIT_LEVELS.items()}
    values["BOGZ1FL744100005Q"][1] = z1_744_q2
    header = "date," + ",".join(f"{s.removeprefix('BOGZ1').removesuffix('Q')}.Q" for s in columns)
    archive = _z1_deposit_archive({"csv/F2_s.csv": "\n".join(
        [header, "2026:Q1," + ",".join(values[s][0] for s in columns), "2026:Q2," + ",".join(values[s][1] for s in columns)])})
    payload = json.loads((ROOT / ALIGNED).read_text())
    payload["z1_release"]["archive_sha256"] = hashlib.sha256(archive).hexdigest()
    _put(tmp_path, ALIGNED, json.dumps(payload))
    monkeypatch.setattr(refresh, "ROOT", tmp_path)
    monkeypatch.setattr(refresh, "_verify_clean_commit", lambda root, commit: commit)
    monkeypatch.setattr(refresh, "_used_quarters", lambda vintage: ["2026Q1", "2026Q2"])
    monkeypatch.setattr(refresh.urllib.request, "urlopen", lambda request, timeout=None: contextlib.nullcontext(io.BytesIO(archive)))
    monkeypatch.setattr(refresh, "_fred_series", lambda series_id, key, start: (
        [{"date": "2026-01-01", "value": DEPOSIT_LEVELS[series_id][0]}, {"date": "2026-04-01", "value": DEPOSIT_LEVELS[series_id][1]}],
        {"series_id": series_id, "last_updated": "2026-09-11 11:53:04-05", "realtime_start": "2026-11-21",
         "observation_end": "2026-04-01"}))
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    return lambda: refresh.main(["fetch-deposit-outcome", "--vintage", ALIGNED, "--producer-commit", COMMIT])


def test_fetched_deposit_series_must_equal_the_pinned_financial_accounts_release(refresh, tmp_path, monkeypatch):
    _fetch_setup(refresh, tmp_path, monkeypatch)()
    target = tmp_path / "data/bundles/deposit_outcome_2026q2"
    receipt = json.loads((target / "fetch_receipt.json").read_text())
    assert set(receipt["series"]) == set(TREATMENT_BANK_GROUP)
    for series_id, record in receipt["series"].items():
        assert record["z1_release_comparison"]["identical"] == 2
        assert sha256_file(target / "fred" / f"{series_id}.csv") == record["sha256"]
    with pytest.raises(ValueError, match="no overwrite"):
        _fetch_setup(refresh, tmp_path, monkeypatch)()


def test_a_fetched_series_from_another_release_is_refused_and_nothing_is_installed(refresh, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="BOGZ1FL744100005Q differs from the pinned Financial Accounts release"):
        _fetch_setup(refresh, tmp_path, monkeypatch, z1_744_q2="92228")()
    assert not (tmp_path / "data/bundles/deposit_outcome_2026q2").exists()


def _deposit_source(tree, rows: dict[str, list[tuple[str, str]]]) -> dict:
    files, series = {}, {}
    for series_id, values in rows.items():
        files[series_id] = _put(tree.root, f"data/bundles/deposits/fred/{series_id}.csv",
                                "date,value\n" + "".join(f"{d},{v}\n" for d, v in values))
        series[series_id] = {"sha256": files[series_id], "last_updated": "2026-09-11 11:53:04-05",
                             "realtime_start": "2026-11-21", "observation_end": "2026-06-01",
                             "z1_release_comparison": {"identical": 2}}
    receipt = _put(tree.root, "data/bundles/deposits/fetch_receipt.json", json.dumps({"producer_commit": COMMIT, "series": series}))
    return {"outcome_id": CANONICAL_OUTCOME_ID, "admitted_series": "AAA", "series": list(rows), "perimeter": "treatment bank group",
            "source": {"root": "data/bundles/deposits", "receipt_sha256": receipt, "files": files}}


def _acquire_aligned(tree, declared: dict) -> Path:
    reuse = {"root": "output/refresh_old", "receipt_sha256": tree.old_receipt_sha256, "vintage_config": "config/vintage_old.json"}
    return tree.main("config/vintage_new.json", "output/refresh_new", tree.corrected_pins(),
                     reuse_acquisition=reuse, deposit_outcome=declared)


def test_added_deposit_series_join_the_reused_fred_inputs_and_are_verified(tree, monkeypatch):
    monkeypatch.setattr(tree.refresh, "_used_quarters", lambda vintage: ["2026Q1", "2026Q2"])
    tree.forbid_network()
    declared = _deposit_source(tree, {"AAA": [("2026-01-01", "1.0"), ("2026-04-01", "2.0")],
                                      "BBB": [("2026-01-01", "7.0"), ("2026-04-01", "8.0")]})
    receipt = json.loads(_acquire_aligned(tree, declared).read_text())
    inputs = tree.root / "output/refresh_new/inputs"
    # The admitted series keeps the reused bytes; its refetch is evidence only.
    assert sha256_file(inputs / "fred/AAA.csv") == receipt["fred_raw"]["AAA"]["sha256"]
    assert sha256_file(inputs / "fred/BBB.csv") == declared["source"]["files"]["BBB"]
    assert sha256_file(inputs / "deposit_outcome/fred/AAA.csv") == declared["source"]["files"]["AAA"]
    assert receipt["deposit_outcome"]["admitted_refetch_consistency"]["identical"] == 2
    vintage = load_vintage(tree.root, "config/vintage_new.json")
    tree.refresh._verify_acquisition(vintage)
    (inputs / "fred/BBB.csv").write_text("date,value\n2026-01-01,7.5\n")
    with pytest.raises(ValueError, match="Acquired input changed: inputs/fred/BBB.csv"):
        tree.refresh._verify_acquisition(vintage)


@pytest.mark.parametrize("case", ["revised_admitted", "repinned_file"])
def test_deposit_series_from_another_release_or_an_unpinned_file_are_refused(tree, monkeypatch, case):
    monkeypatch.setattr(tree.refresh, "_used_quarters", lambda vintage: ["2026Q1", "2026Q2"])
    admitted_q2 = "2.5" if case == "revised_admitted" else "2.0"
    declared = _deposit_source(tree, {"AAA": [("2026-01-01", "1.0"), ("2026-04-01", admitted_q2)],
                                      "BBB": [("2026-01-01", "7.0"), ("2026-04-01", "8.0")]})
    if case == "repinned_file":
        declared["source"]["files"]["BBB"] = _put(tree.root, "data/bundles/deposits/fred/BBB.csv", "date,value\n2026-01-01,9\n")
        message = "fetch receipt and pin disagree: BBB"
    else:
        message = "Refetched AAA differs from the acquired series"
    with pytest.raises(ValueError, match=message):
        _acquire_aligned(tree, declared)
    assert not (tree.root / "output/refresh_new/inputs").exists()
