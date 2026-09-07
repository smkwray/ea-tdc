from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from ea_tdc.open_contract import (
    CANONICAL_CONTROL_IDS,
    CANONICAL_OUTCOME_ID,
    CANONICAL_TREATMENT_ID,
    OPEN01_DESIGN_JOB_IDS,
)


@pytest.fixture
def producer(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("frozen_reproduction_test", scripts / "run_frozen_factor_reproduction.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def graph(producer, tmp_path):
    rng = np.random.default_rng(101)
    quarters = ["1945Q4"] + [f"{i // 4}Q{i % 4 + 1}" for i in range(1946 * 4 + 3, 2026 * 4 + 2)]
    features = [f"m__fixture_{i}__lag001" for i in range(104)]
    controls = [k for k in CANONICAL_CONTROL_IDS if not k.startswith("dflmx_")]
    sources = {k.split("__lag_")[0] for k in controls}
    anchor = [{"quarter": q, CANONICAL_TREATMENT_ID: rng.normal(), CANONICAL_OUTCOME_ID: rng.normal(),
               **{k: rng.normal() for k in sources}} for q in quarters]
    universe = [{"quarter": q, **{k: rng.normal() for k in features}} for q in quarters]
    # Identical candidate pair tests retention of incoming order under a stable tie.
    for row in universe:
        row[features[1]] = row[features[0]]
    columns = [{"feature_id": k} for k in features]
    inputs = tmp_path / "inputs"
    producer.write_csv(inputs / "universe.csv", universe)
    producer.write_csv(inputs / "columns.csv", columns)
    producer.write_csv(inputs / "anchor.csv", anchor)
    producer.write_json(inputs / "manifest.json", {"treatment_id": CANONICAL_TREATMENT_ID, "outcome_ids": [CANONICAL_OUTCOME_ID], "control_ids": controls[:4]})
    producer.write_csv(inputs / "headline.csv", [{"outcome_id": CANONICAL_OUTCOME_ID, "treatment_id": CANONICAL_TREATMENT_ID, "beta": 999, "se": 999}])
    endpoints = [f"{2011 + i // 4}Q{i % 4 + 1}" for i in range(3, 60)]
    producer.write_csv(inputs / "rolling.csv", [{"outcome": CANONICAL_OUTCOME_ID, "horizon": "0", "window_end_quarter": q, "beta": 999, "se": 999} for q in endpoints])
    producer.write_json(inputs / "accepted.json", {"contract": {"control_ids": list(CANONICAL_CONTROL_IDS), "treatment_id": CANONICAL_TREATMENT_ID, "design_job_ids": list(OPEN01_DESIGN_JOB_IDS)},
        "designs": {job: {"bundle": {"path": "anchor.csv"}, "design_manifest": {"path": "manifest.json"}} for job in OPEN01_DESIGN_JOB_IDS},
        "retained_outputs": {"fixed": {"same_quarter_headline": {"path": "headline.csv"}, "rolling_estimates": {"path": "rolling.csv"}}}})
    producer.write_json(inputs / producer.CONFIG, {"origin": producer.ORIGIN, "claim_status": "new_pending_review",
        "accepted_receipt": {"path": "accepted.json"}, "graph_cut": {"panel": {"path": "universe.csv"}, "columns": {"path": "columns.csv"}}})
    return anchor, universe, columns


def test_new_reproduction_retains_scores_order_vectors_and_unmatched_comparisons(producer, graph, tmp_path):
    producer.compute_snapshot(tmp_path)
    outputs = tmp_path / "results"
    summary = json.loads((outputs / "summary.json").read_text())
    assert summary["origin"] == "new_frozen_input_reproduction"
    assert not summary["original_coordinate_identity_established"]
    assert not summary["open16_leg_slopes_computed"]
    assert summary["headline"]["archived_beta"] == 999
    assert summary["headline"]["beta"] != 999  # Mismatch is reported, never fitted away.
    assert len(producer.read_csv(outputs / "full_factor_scores.csv")) == 320
    assert len(producer.read_csv(outputs / "sample_factor_scores.csv")) == 96
    assert len(json.loads((outputs / "projection_vectors.json").read_text())) == 58
    screening = producer.read_csv(outputs / "ordered_screening.csv")
    ids = [r["feature_id"] for r in screening]
    assert ids.index("m__fixture_0__lag001") + 1 == ids.index("m__fixture_1__lag001")
    assert json.loads((outputs / "ordered_top100.json").read_text()) == ids[:100]
    assert summary["rolling_windows"] == 57


@pytest.mark.parametrize("mutation", ["column_order", "quarter_order", "nonfinite"])
def test_graph_cut_rejects_order_or_value_changes(producer, graph, mutation):
    anchor, universe, columns = graph
    if mutation == "column_order": columns.reverse()
    elif mutation == "quarter_order": universe[0], universe[1] = universe[1], universe[0]
    else: universe[0][columns[0]["feature_id"]] = "nan"
    with pytest.raises(ValueError): producer.validate_universe(anchor, universe, columns)


def test_frozen_copy_hash_failure_cannot_replace_source(producer, tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("preserve these exact bytes")
    with pytest.raises(ValueError, match="hash/path mismatch"):
        producer.copy_pinned(tmp_path, tmp_path / "snapshot", {"path": source.name, "sha256": "0" * 64})
    assert source.read_text() == "preserve these exact bytes"
    assert not (tmp_path / "snapshot").exists()


def test_frozen_copy_rejects_parent_traversal(producer, tmp_path):
    with pytest.raises(ValueError, match="canonical relative"):
        producer.copy_pinned(tmp_path, tmp_path / "snapshot", {"path": "../outside.csv", "sha256": "0" * 64})
