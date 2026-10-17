"""Economic invariants of the declared 2026Q2 vintage rerun.

A vintage may extend the sample and refresh inputs; it may not truncate the
sample, change the 16 controls, reselect the factor features, or loosen gates.
"""
from __future__ import annotations

import csv
import importlib.util
import io
import json
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
    EXPECTED_METHOD_TIER_COUNTS,
    OPEN02_CONTRACT,
)
from ea_tdc.paths import project_paths
from ea_tdc.residualized_shock import _load_factor_branch
from ea_tdc.robustness import _extract_factor_controls, _merge_control_rows
from ea_tdc.vintage import (
    VINTAGE_ENV,
    load_vintage,
    open01_tier_counts,
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
