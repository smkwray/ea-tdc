#!/usr/bin/env python3
"""Rerun the admitted quarterly chains on a declared observation vintage.

Every lane is fail-closed and writes its own receipt under the vintage output
root; admitted outputs are never written. Lanes:

  snapshot      hash the admitted output/data surface before any refresh work
  acquire       retrieve FRED observations with their vintages, copy the
                hash-bound TDCest and TDCpass refresh objects, reconstruct the
                TDCest site bundle, and record OPEN-02 source availability
  materialize   clone the producer commit and place the vintage inputs in it
  open01        run the unmodified OPEN-01 producer inside that clone and export
                its 16-control projection panel
  open16        validate that panel in both pinned numerical runtimes and run
                the bounded OPEN-16 covariance and calendar-deletion diagnostics
  seal          bind inputs, outputs and the unchanged admitted surface
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ea_tdc.vintage import (  # noqa: E402
    VINTAGE_ENV,
    Vintage,
    load_vintage,
    quarter_ordinal,
    sha256_file,
)

DEFAULT_VINTAGE = "config/vintage_2026q2.json"
FRED_BASE = "https://api.stlouisfed.org/fred"
RUN_DIR = "run"
PROTECTED_PREFIXES = ("output", "outputs", "data")
PANEL_LOCATOR = "output/refresh/open01_projection_panel.csv"
FACTOR_SCORES_LOCATOR = "output/refresh/full_factor_scores.csv"
PANEL_RECEIPT_LOCATOR = "output/refresh/open01_projection_panel_receipt.json"
TDCPASS_ADAPTER_FILES = (
    "data/derived/quarterly_panel.csv",
    "output/shocks/unexpected_tdc.csv",
    "output/models/lp_irf_identity_baseline.csv",
    "output/models/result_readiness_summary.json",
)
TDCEST_BUNDLE_SOURCES = {
    "estimates": "tdc_estimates.csv",
    "components": "tdc_components.csv",
    "corrections": "tdc_corrections.csv",
}
TDCEST_REFERENCE_COLUMNS = ("gdp_deflator", "nominal_gdp_saar_bil")
ESTIMATE_TOLERANCE = 1e-7


# --------------------------------------------------------------------------- utilities


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _record(root: Path, path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _copy_verified(source: Path, target: Path, expected_sha256: str) -> None:
    if sha256_file(source) != expected_sha256:
        raise ValueError(f"Pinned input changed: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha256_file(target) != expected_sha256:
        raise ValueError(f"Input changed during copy: {source}")


def _output_root(vintage: Vintage) -> Path:
    return ROOT / vintage.payload["output_root"]


def _verify_clean_commit(root: Path, commit: str) -> str:
    from run_open01_producer import _verify_producer_commit

    return _verify_producer_commit(root, commit)


def _python_env(run_root: Path) -> dict[str, str]:
    """Run-root sources shadow any installed editable checkout."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(run_root / "src"), str(run_root / "scripts")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env[VINTAGE_ENV] = env_vintage_locator(env)
    return env


def env_vintage_locator(env: Mapping[str, str]) -> str:
    return env.get(VINTAGE_ENV) or DEFAULT_VINTAGE


def _run_logged(argv: list[str], *, cwd: Path, env: Mapping[str, str], log: Path) -> dict[str, Any]:
    started = _now()
    result = subprocess.run(argv, cwd=cwd, env=dict(env), capture_output=True, text=True, check=False)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    record = {"argv": argv, "started_at_utc": started, "completed_at_utc": _now(), "exit_code": result.returncode,
              "log": log.name}
    if result.returncode:
        raise RuntimeError(f"Command failed with exit {result.returncode}; see {log}")
    return record


# --------------------------------------------------------------------------- pure transforms


def cut_observations(rows: Sequence[Mapping[str, str]], cutoff: date) -> tuple[list[dict[str, str]], int]:
    """Drop observations dated after the vintage cutoff; never alter retained values."""
    kept = [{"date": row["date"], "value": row["value"]} for row in rows if date.fromisoformat(row["date"][:10]) <= cutoff]
    return kept, len(rows) - len(kept)


def admitted_start(path: Path, default: str) -> str:
    """Keep each refreshed raw series on the observation window of its admitted file."""
    if not path.is_file():
        return default
    rows = _read_csv(path)
    return rows[0]["date"][:10] if rows else default


def carry_retired_series(
    refreshed: Sequence[Mapping[str, str]], admitted: Sequence[Mapping[str, str]]
) -> tuple[list[dict[str, str]], list[str]]:
    """Append admitted rows only for series the refreshed release no longer publishes."""
    present = {row["series_id"] for row in refreshed}
    retired = sorted({row["series_id"] for row in admitted} - present)
    if refreshed and admitted and list(refreshed[0]) != list(admitted[0]):
        raise ValueError("Admitted and refreshed standardized layouts differ")
    keep = set(retired)
    return [dict(row) for row in refreshed] + [dict(row) for row in admitted if row["series_id"] in keep], retired


def factor_source_files(features: Sequence[str], universe_columns: Sequence[Mapping[str, str]]) -> dict[str, str]:
    """Map the raw seed files behind the pinned features to their FRED identifiers."""
    by_feature = {row["feature_id"]: row for row in universe_columns}
    missing = [feature for feature in features if feature not in by_feature]
    if missing:
        raise ValueError(f"Pinned features are absent from the universe columns: {missing[:5]}")
    sources: dict[str, str] = {}
    for feature in features:
        source = by_feature[feature]["source_file"]
        match = re.fullmatch(r"FRED_([A-Z0-9]+)_.+\.csv", source)
        if match is None:
            raise ValueError(f"Factor source file is not a FRED seed series: {source}")
        sources[source] = match.group(1)
    return dict(sorted(sources.items()))


def _csv_number(text: str) -> int | float | None:
    value = text.strip()
    if not value:
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    number = float(value)
    return None if math.isnan(number) else number


def reconstruct_site_bundle(processed: Path, *, generated_at: str, input_hashes: Mapping[str, str]) -> dict[str, Any]:
    """Rebuild the TDCest site-bundle frame payloads from its processed CSVs.

    TDCest writes the same estimates, components, corrections and quarterly
    frames to processed CSVs and to the site bundle; this reproduces that
    payload layout without re-running the TDCest estimator.
    """
    frames: dict[str, tuple[list[str], dict[str, list[Any]]]] = {}
    for key, name in TDCEST_BUNDLE_SOURCES.items():
        rows = _read_csv(processed / name)
        columns = [column for column in rows[0] if column != "date"]
        frames[key] = ([row["date"] for row in rows], {column: [_csv_number(row[column]) for row in rows] for column in columns})
    dates = frames["estimates"][0]
    if len(set(dates)) != len(dates) or dates != sorted(dates):
        raise ValueError("TDCest estimates dates are not unique and ordered")
    for key, (frame_dates, _) in frames.items():
        if frame_dates != dates:
            raise ValueError(f"TDCest {key} dates do not align with estimates")
    quarterly = {row["date"]: row for row in _read_csv(processed / "quarterly_inputs.csv")}
    references = {
        column: [_csv_number(quarterly[d][column]) if d in quarterly else None for d in dates]
        for column in TDCEST_REFERENCE_COLUMNS
    }

    def payload(columns: Mapping[str, list[Any]]) -> dict[str, Any]:
        return {"columns": list(columns), **columns}

    estimates = frames["estimates"][1]
    latest = dates[-1]
    return {
        "bundle_format": "tdc_site_bundle_v4",
        "generated_at_utc": generated_at,
        "summary": {"generated_at_utc": generated_at, "latest_period": latest, "available_methods": list(estimates)},
        "metadata": {
            "generated_at_utc": generated_at,
            "value_units": {"nominal": "Millions of U.S. dollars"},
            "reconstruction": {
                "kind": "ea_tdc_vintage_site_bundle",
                "method": "TDCest site-bundle frame payloads rebuilt from hash-bound processed CSVs",
                "inputs": dict(sorted(input_hashes.items())),
            },
        },
        "dates": dates,
        "estimates": payload(estimates),
        "components": payload(frames["components"][1]),
        "corrections": payload(frames["corrections"][1]),
        "references": payload(references),
    }


def projection_panel(rows: Iterable[Mapping[str, Any]], quarters: Sequence[str]) -> list[dict[str, Any]]:
    """Return the complete-case OPEN-01 panel; it must be exactly the declared sample."""
    from ea_tdc.open_contract import CANONICAL_CONTROL_IDS, CANONICAL_OUTCOME_ID, CANONICAL_TREATMENT_ID

    keys = (CANONICAL_TREATMENT_ID, CANONICAL_OUTCOME_ID, *CANONICAL_CONTROL_IDS)
    complete = []
    for row in sorted(rows, key=lambda item: quarter_ordinal(str(item["quarter"]))):
        values = {}
        for key in keys:
            try:
                value = float(row.get(key, ""))
            except (TypeError, ValueError):
                break
            if not math.isfinite(value):
                break
            values[key] = value
        else:
            complete.append({"quarter": str(row["quarter"]), **values})
    observed = tuple(row["quarter"] for row in complete)
    if observed != tuple(quarters):
        missing = sorted(set(quarters) - set(observed), key=quarter_ordinal)
        extra = sorted(set(observed) - set(quarters), key=quarter_ordinal)
        raise ValueError(
            f"Complete-case sample is not the declared {quarters[0]}-{quarters[-1]} vintage "
            f"(n={len(observed)} vs {len(quarters)}; missing={missing[:6]}; extra={extra[:6]})"
        )
    return complete


def open02_availability(archive: bytes, admitted_bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Check the 20 allowlisted OPEN-02 series against a Board Z.1 archive."""
    from ea_tdc.open_contract import OPEN02_CONTRACT

    dictionaries: dict[str, list[str]] = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as handle:
        for name in sorted(handle.namelist()):
            if name.startswith("data_dictionary/") and name.endswith(".txt"):
                for line in handle.read(name).decode("utf-8-sig").splitlines():
                    series_id = line.split("\t", 1)[0].strip()
                    if series_id:
                        dictionaries.setdefault(series_id, []).append(name)
        legacy_members = sorted(name for name, _ in OPEN02_CONTRACT.source.csv_member_sha256)
        members_present = [name for name in legacy_members if name in handle.namelist()]
    admitted = {series["key"]: series for series in admitted_bundle.get("series", [])}
    records = []
    for series in OPEN02_CONTRACT.series:
        records.append({
            "key": series.key,
            "board_series_id": series.board_series_id,
            "official_title": series.official_title,
            "present": series.board_series_id in dictionaries,
            "dictionary_members": dictionaries.get(series.board_series_id, []),
            "admitted_observations": len(admitted.get(series.key, {}).get("observations", [])),
        })
    absent = [record["board_series_id"] for record in records if not record["present"]]
    return {
        "allowlisted_series": len(records),
        "present": len(records) - len(absent),
        "absent_series_ids": absent,
        "admitted_csv_members_present": members_present,
        "series": records,
        "disposition": "blocked_allowlisted_series_absent" if absent else "available",
    }


# --------------------------------------------------------------------------- snapshot


def protected_inventory(root: Path, output_root: Path) -> dict[str, str]:
    from validate_open16_reproduction import PRODUCTION_SOURCES

    inventory: dict[str, str] = {}
    for prefix in PROTECTED_PREFIXES:
        base = root / prefix
        for directory, names, files in os.walk(base):
            current = Path(directory)
            names[:] = sorted(
                name for name in names
                if name != "__pycache__" and (current / name).resolve() != output_root.resolve()
            )
            for name in files:
                path = current / name
                if path.is_file() and not path.is_symlink():
                    inventory[path.relative_to(root).as_posix()] = sha256_file(path)
    for name in PRODUCTION_SOURCES:
        inventory[name] = sha256_file(root / name)
    return dict(sorted(inventory.items()))


def lane_snapshot(vintage: Vintage, commit: str) -> Path:
    _verify_clean_commit(ROOT, commit)
    output = _output_root(vintage)
    target = output / "protected_baseline.json"
    if target.exists():
        raise ValueError("Protected baseline already exists; no overwrite")
    inventory = protected_inventory(ROOT, output)
    _write_json(target, {"producer_commit": commit, "created_at_utc": _now(), "files": inventory,
                         "count": len(inventory), "vintage": vintage.record()})
    return target


# --------------------------------------------------------------------------- acquire


def _fred_json(endpoint: str, params: Mapping[str, str], key: str) -> tuple[dict[str, Any], str]:
    query = urllib.parse.urlencode({**params, "api_key": key, "file_type": "json"})
    public_url = f"{FRED_BASE}/{endpoint}?" + urllib.parse.urlencode({**params, "file_type": "json"})
    for attempt in range(5):
        try:
            request = urllib.request.Request(f"{FRED_BASE}/{endpoint}?{query}", headers={"User-Agent": "ea-tdc/0.1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
            time.sleep(0.55)
            return payload, public_url
        except Exception:  # noqa: BLE001 - bounded retry, then fail closed
            if attempt == 4:
                raise RuntimeError(f"FRED request failed after retries: {public_url}") from None
            time.sleep(2.0 * (attempt + 1))
    raise AssertionError("unreachable")


def _fred_series(series_id: str, key: str, start: str | None) -> tuple[list[dict[str, str]], dict[str, Any]]:
    meta, meta_url = _fred_json("series", {"series_id": series_id}, key)
    params = {"series_id": series_id}
    if start:
        params["observation_start"] = start
    observations, obs_url = _fred_json("series/observations", params, key)
    rows = [{"date": item["date"], "value": item["value"]} for item in observations.get("observations", [])]
    if not rows:
        raise ValueError(f"FRED returned no observations for {series_id}")
    series = meta["seriess"][0]
    vintage = {
        "series_id": series_id,
        "title": series.get("title"),
        "frequency": series.get("frequency_short"),
        "units": series.get("units"),
        "seasonal_adjustment": series.get("seasonal_adjustment_short"),
        "last_updated": series.get("last_updated"),
        "observation_end": series.get("observation_end"),
        "realtime_start": observations.get("realtime_start"),
        "realtime_end": observations.get("realtime_end"),
        "observations": len(rows),
        "first_date": rows[0]["date"],
        "last_date": rows[-1]["date"],
        "urls": [meta_url, obs_url],
    }
    return rows, vintage


def _write_series_csv(path: Path, rows: Sequence[Mapping[str, str]], *, missing: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "value"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"date": row["date"], "value": missing if row["value"] == "." else row["value"]})


def _overlap(admitted: Path, rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    old = {row["date"]: row["value"] for row in _read_csv(admitted) if row["value"] not in ("", ".")}
    new = {row["date"]: row["value"] for row in rows if row["value"] not in ("", ".")}
    common = sorted(set(old) & set(new))
    gaps = [abs(float(old[d]) - float(new[d])) for d in common]
    return {"admitted_last_date": max(old) if old else None, "common_dates": len(common),
            "identical_values": sum(gap == 0 for gap in gaps), "max_abs_revision": max(gaps, default=0.0)}


def lane_acquire(vintage: Vintage, commit: str) -> Path:
    _verify_clean_commit(ROOT, commit)
    output = _output_root(vintage)
    if not (output / "protected_baseline.json").is_file():
        raise ValueError("Run the snapshot lane before acquisition")
    for name in ("inputs", "tdcest", "tdcpass"):
        if (output / name).exists():
            raise ValueError(f"{name} already acquired; no overwrite")
    key = os.environ.get(vintage.payload["fred"]["api_key_env"], "").strip()
    if not key:
        raise ValueError("FRED API key environment variable is not set")
    stage = Path(tempfile.mkdtemp(prefix=".acquire-", dir=output))
    receipt: dict[str, Any] = {"lane": "acquire", "producer_commit": commit, "started_at_utc": _now(),
                               "vintage": vintage.record()}
    try:
        inputs = stage / "inputs"
        # FRED raw series behind the design builder.
        manifest = vintage.payload["fred"]["series_manifest"]
        manifest_rows = _read_csv(ROOT / manifest["path"])
        if sha256_file(ROOT / manifest["path"]) != manifest["sha256"]:
            raise ValueError("FRED series manifest changed")
        fred_records = {}
        for row in manifest_rows:
            series_id = row["series_id"]
            admitted = ROOT / "data/raw/fred" / f"{series_id}.csv"
            start = admitted_start(admitted, vintage.payload["fred"]["raw_observation_start"])
            rows, record = _fred_series(series_id, key, start)
            path = inputs / "fred" / f"{series_id}.csv"
            _write_series_csv(path, rows, missing=".")
            record["observation_start_requested"] = start
            fred_records[series_id] = {**record, **_record(stage, path),
                                       "admitted_overlap": _overlap(admitted, rows) if admitted.is_file() else None}
        receipt["fred_raw"] = fred_records

        # Raw seed series behind the pinned K=100 factor features.
        factor = vintage.payload["factor_features"]
        for name in ("ordered_top100", "universe_columns", "raw_inventory"):
            _copy_verified(ROOT / factor[name]["path"], inputs / "pins" / factor[name]["path"], factor[name]["sha256"])
        features = _read_json(ROOT / factor["ordered_top100"]["path"])
        sources = factor_source_files(features, _read_csv(ROOT / factor["universe_columns"]["path"]))
        raw_inventory = {
            Path(item["path"]).name: item["sha256"]
            for item in _read_json(ROOT / factor["raw_inventory"]["path"])["raw_inventory_provenance_only_not_consumed"]
        }
        seed_dir = ROOT / factor["raw_seed_dir"]
        seed_records = {}
        for path in sorted(seed_dir.glob("*.csv")):
            if path.name not in raw_inventory:
                raise ValueError(f"Seed file is outside the admitted raw inventory: {path.name}")
            target = inputs / "interpol_raw" / path.name
            if path.name in sources:
                rows, record = _fred_series(sources[path.name], key, None)
                _write_series_csv(target, rows, missing="")
                seed_records[path.name] = {"status": "refreshed", **record, **_record(stage, target),
                                           "admitted_sha256": raw_inventory[path.name],
                                           "admitted_overlap": _overlap(path, rows)}
            else:
                _copy_verified(path, target, raw_inventory[path.name])
                seed_records[path.name] = {"status": "carried_admitted_bytes", **_record(stage, target)}
        if set(seed_records) != set(raw_inventory) or not set(sources) <= set(seed_records):
            raise ValueError("Seed inventory differs from the admitted raw inventory")
        receipt["factor_seed"] = {"refreshed_files": sources, "files": seed_records,
                                  "carried": sum(r["status"] == "carried_admitted_bytes" for r in seed_records.values())}

        # Carried standardized bundles that enter the design grid but no OPEN-01 control.
        receipt["carried_bundles"] = {}
        for locator, digest in vintage.payload["carried_bundles"].items():
            _copy_verified(ROOT / locator, inputs / "carried" / locator, digest)
            receipt["carried_bundles"][locator] = digest

        # TDCest refresh objects and the reconstructed site bundle.
        tdcest = vintage.payload["tdcest"]
        tdcest_root = (ROOT / tdcest["root"]).resolve()
        if sha256_file(tdcest_root / "RECEIPT.json") != tdcest["receipt_sha256"]:
            raise ValueError("TDCest refresh receipt changed")
        tdcest_receipt = _read_json(tdcest_root / "RECEIPT.json")
        _copy_verified(tdcest_root / "RECEIPT.json", stage / "tdcest" / "RECEIPT.json", tdcest["receipt_sha256"])
        listed = tdcest_receipt.get("outputs", {})
        tdcest_records = {}
        for locator, digest in tdcest["files"].items():
            if locator in listed and listed[locator]["sha256"] != digest:
                raise ValueError(f"TDCest receipt and vintage pin disagree: {locator}")
            _copy_verified(tdcest_root / locator, stage / "tdcest" / locator, digest)
            tdcest_records[locator] = {"sha256": digest, "bound_by": "tdcest_receipt" if locator in listed else "vintage_pin"}
        processed = stage / "tdcest" / "data" / "processed"
        bundle = reconstruct_site_bundle(
            processed,
            generated_at=tdcest["generated_at"],
            input_hashes={f"data/processed/{name}": tdcest["files"][f"data/processed/{name}"]
                          for name in (*TDCEST_BUNDLE_SOURCES.values(), "quarterly_inputs.csv")},
        )
        bundle_path = stage / "tdcest" / "site" / "data" / "bundle.json"
        _write_json(bundle_path, bundle)
        receipt["tdcest"] = {"receipt_sha256": tdcest["receipt_sha256"], "files": tdcest_records,
                             "z1_vintage": tdcest_receipt.get("z1_vintage"), "producers": tdcest_receipt.get("producers"),
                             "site_bundle": _record(stage, bundle_path), "dates": [bundle["dates"][0], bundle["dates"][-1]]}

        # TDCpass refreshed strict-loan constructions.
        tdcpass = vintage.payload["tdcpass"]
        tdcpass_root = (ROOT / tdcpass["root"]).resolve()
        for locator, digest in tdcpass["files"].items():
            _copy_verified(tdcpass_root / locator, stage / "tdcpass" / locator, digest)
        retired_source = tdcpass["admitted_bundle"]
        _copy_verified(ROOT / retired_source["path"], inputs / "carried" / "admitted_tdcpass" / Path(retired_source["path"]).name,
                       retired_source["sha256"])
        audit = _read_json(stage / "tdcpass" / "output/models/proxy_unit_audit.json")
        resolved = {item["channel"]: item for item in audit["strict_source_channels"]}
        for channel, series_id in tdcpass["strict_loan_series"].items():
            if resolved[channel]["series_id"] != series_id or resolved[channel]["end_quarter"] != vintage.sample_end:
                raise ValueError(f"TDCpass {channel} resolved to another series or ends early")
        receipt["tdcpass"] = {"producer_commit": tdcpass["producer_commit"], "files": tdcpass["files"],
                              "strict_loan_series": {c: resolved[c] for c in tdcpass["strict_loan_series"]}}

        # OPEN-02 source availability in the pinned Financial Accounts release.
        release = vintage.payload["z1_release"]
        request = urllib.request.Request(release["archive_url"], headers={"User-Agent": "ea-tdc/0.1.0"})
        with urllib.request.urlopen(request, timeout=180) as response:
            archive = response.read()
        if hashlib.sha256(archive).hexdigest() != release["archive_sha256"]:
            raise ValueError("Financial Accounts archive differs from the pinned release")
        admitted = vintage.payload["open02"]["admitted_source_bundle"]
        if sha256_file(ROOT / admitted["path"]) != admitted["sha256"]:
            raise ValueError("Admitted OPEN-02 source bundle changed")
        availability = open02_availability(archive, _read_json(ROOT / admitted["path"]))
        from ea_tdc.open_contract import OPEN02_CONTRACT as contract

        fred_ids = {series.board_series_id: series.fred_id for series in contract.series}
        availability["fred_status_for_absent"] = {
            series_id: _fred_json("series", {"series_id": fred_ids[series_id]}, key)[0]["seriess"][0]
            for series_id in availability["absent_series_ids"]
        }
        availability.update({"release": release, "archive_bytes": len(archive), "checked_at_utc": _now()})
        _write_json(inputs / "open02_source_availability.json", availability)
        receipt["open02_source_availability"] = {**_record(stage, inputs / "open02_source_availability.json"),
                                                 "disposition": availability["disposition"],
                                                 "absent_series_ids": availability["absent_series_ids"]}

        receipt["completed_at_utc"] = _now()
        _write_json(inputs / "acquisition_receipt.json", receipt)
        _verify_clean_commit(ROOT, commit)
        for name in ("inputs", "tdcest", "tdcpass"):
            os.replace(stage / name, output / name)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return output / "inputs" / "acquisition_receipt.json"


# --------------------------------------------------------------------------- materialize


def _verify_acquisition(vintage: Vintage) -> dict[str, Any]:
    output = _output_root(vintage)
    receipt = _read_json(output / "inputs" / "acquisition_receipt.json")
    if receipt.get("vintage") != vintage.record():
        raise ValueError("Acquisition was made under another vintage configuration")
    checks = []
    for record in receipt["fred_raw"].values():
        checks.append(record)
    for record in receipt["factor_seed"]["files"].values():
        checks.append(record)
    for record in checks:
        if sha256_file(output / record["path"]) != record["sha256"]:
            raise ValueError(f"Acquired input changed: {record['path']}")
    for base, files in (("tdcest", vintage.payload["tdcest"]["files"]), ("tdcpass", vintage.payload["tdcpass"]["files"])):
        for locator, digest in files.items():
            if sha256_file(output / base / locator) != digest:
                raise ValueError(f"Acquired {base} object changed: {locator}")
    bundle = receipt["tdcest"]["site_bundle"]
    if sha256_file(output / "tdcest" / "site/data/bundle.json") != bundle["sha256"]:
        raise ValueError("Reconstructed site bundle changed")
    return receipt


def lane_materialize(vintage: Vintage, commit: str) -> Path:
    _verify_clean_commit(ROOT, commit)
    output = _output_root(vintage)
    _verify_acquisition(vintage)
    run_root = output / RUN_DIR
    if run_root.exists():
        raise ValueError("Run root already exists; no overwrite")
    subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", "--no-checkout", str(ROOT), str(run_root)], check=True)
    git = ["git", "-C", str(run_root)]
    for option in ("core.autocrlf", "core.fileMode"):
        # Byte-identical checkout on any transfer target, including Windows hosts.
        subprocess.run([*git, "config", option, "false"], check=True)
    # Remote transfer excludes lock files; the run never reads the dependency lock.
    subprocess.run([*git, "sparse-checkout", "set", "--no-cone", "/*", "!/uv.lock"], check=True)
    subprocess.run([*git, "checkout", "--quiet", "--detach", commit], check=True)
    cutoff = vintage.observation_cutoff
    records: dict[str, Any] = {"fred": {}, "interpol": {}}
    for path in sorted((output / "inputs" / "fred").glob("*.csv")):
        rows, dropped = cut_observations(_read_csv(path), cutoff)
        target = run_root / "data/raw/fred" / path.name
        _write_series_csv(target, rows, missing=".")
        records["fred"][path.stem] = {"dropped_after_cutoff": dropped, **_record(run_root, target)}
    for path in sorted((output / "inputs" / "interpol_raw").glob("*.csv")):
        target = run_root / vintage.payload["factor_features"]["raw_seed_dir"] / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = _read_csv(path)
        dropped = None
        if rows and list(rows[0]) == ["date", "value"]:
            rows, dropped = cut_observations(rows, cutoff)
        if dropped:
            _write_series_csv(target, rows, missing="")
        else:
            # Unchanged bytes; files without a date/value layout never enter the control universe.
            shutil.copyfile(path, target)
        records["interpol"][path.name] = {"dropped_after_cutoff": dropped, **_record(run_root, target)}
    for locator, digest in vintage.payload["carried_bundles"].items():
        _copy_verified(output / "inputs" / "carried" / locator, run_root / locator, digest)
    top100 = vintage.payload["factor_features"]["ordered_top100"]
    _copy_verified(output / "inputs" / "pins" / top100["path"], run_root / top100["path"], top100["sha256"])
    env = _python_env(run_root)
    adapter = _run_logged(
        [sys.executable, "-B", "-m", "ea_tdc", "adapt-tdcpass", "--repo-root", ".", "--publish-dir", "../tdcpass"],
        cwd=run_root, env=env, log=output / "logs" / "materialize_adapt_tdcpass.log",
    )
    standardized = run_root / "data/bundles/tdcpass/standardized_series.csv"
    adapter_record = _record(run_root, standardized)
    admitted_tdcpass = output / "inputs/carried/admitted_tdcpass/standardized_series.csv"
    if sha256_file(admitted_tdcpass) != vintage.payload["tdcpass"]["admitted_bundle"]["sha256"]:
        raise ValueError("Admitted TDCpass bundle changed")
    merged, retired = carry_retired_series(_read_csv(standardized), _read_csv(admitted_tdcpass))
    _write_csv(standardized, merged)
    status = subprocess.run([*git, "status", "--porcelain", "--untracked-files=all"], check=True,
                            capture_output=True, text=True).stdout
    head = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if status or head != commit:
        raise ValueError(f"Run root is not a clean checkout of {commit}: {status[:200]}")
    lane = {"lane": "materialize", "producer_commit": commit, "completed_at_utc": _now(),
            "observation_cutoff": cutoff.isoformat(), "acquisition_receipt": _record(output, output / "inputs/acquisition_receipt.json"),
            "run_root_head": head, "materialized": records, "tdcpass_adapter": adapter,
            "tdcpass_adapter_output": adapter_record,
            "tdcpass_retired_series_carried": retired,
            "tdcpass_standardized": _record(run_root, standardized)}
    target = output / "materialize_receipt.json"
    _write_json(target, lane)
    return target


# --------------------------------------------------------------------------- OPEN-01


def lane_open01(vintage: Vintage, commit: str, run_id: str) -> Path:
    from run_open01_producer import _validate_run_id

    output = _output_root(vintage)
    run_root = output / RUN_DIR
    materialized = _read_json(output / "materialize_receipt.json")
    if materialized["producer_commit"] != commit:
        raise ValueError("Run root was materialized from another commit")
    receipt_path = run_root / "output/manifests/open01_producer_run_receipt.json"
    if receipt_path.exists():
        raise ValueError("OPEN-01 producer receipt already exists in the run root")
    env = _python_env(run_root)
    probe = subprocess.run([sys.executable, "-B", "-c", "import ea_tdc,sys;print(ea_tdc.__file__)"],
                           cwd=run_root, env=env, check=True, capture_output=True, text=True).stdout.strip()
    if Path(probe).resolve().parent != (run_root / "src" / "ea_tdc").resolve():
        raise ValueError(f"Producer would import ea_tdc from outside the run root: {probe}")
    producer = _run_logged(
        [sys.executable, "-B", "scripts/run_open01_producer.py", "--producer-commit", commit,
         "--run-id", _validate_run_id(run_id)],
        cwd=run_root, env=env, log=output / "logs" / "open01_producer.log",
    )
    panel = _run_logged([sys.executable, "-B", "scripts/run_vintage_refresh.py", "export-panel"],
                        cwd=run_root, env=env, log=output / "logs" / "open01_export_panel.log")
    receipt = _read_json(receipt_path)
    sample = receipt["acceptance"]["sample"]
    expected = {"start": vintage.sample_start, "end": vintage.sample_end, "n": vintage.observations,
                "quarter_hash": vintage.quarter_hash}
    if sample != expected:
        raise ValueError(f"OPEN-01 accepted sample {sample} differs from the vintage {expected}")
    lane = {"lane": "open01", "producer_commit": commit, "run_id": run_id, "completed_at_utc": _now(),
            "host": platform.node(),
            "commands": {"producer": producer, "export_panel": panel},
            "producer_receipt": _record(run_root, receipt_path),
            "panel_receipt": _record(run_root, run_root / PANEL_RECEIPT_LOCATOR),
            "accepted_sample": sample, "scientific_status": receipt["acceptance"]["scientific_status"]}
    target = output / "open01_receipt.json"
    _write_json(target, lane)
    return target


def export_panel() -> Path:
    """Run inside the run root: rebuild the factor rows without touching receipted outputs."""
    from dataclasses import replace

    from ea_tdc.open01 import _fit_projection
    from ea_tdc.open_contract import CANONICAL_CONTROL_IDS, CANONICAL_OUTCOME_ID, CANONICAL_TREATMENT_ID
    from ea_tdc.paths import project_paths
    from ea_tdc.residualized_shock import _load_factor_branch
    from ea_tdc.vintage import active_vintage
    from run_pinned_factor_residual_bridge import (
        ANCHOR_JOB_ID,
        CONTROL_POLICY_MODE,
        FACTOR_COUNT,
        K_SCREENED,
        MERGE_JOBS,
        _merge_by_quarter,
    )
    from run_tier2_credit_lead_diagnostics import CREDIT_LAG_SOURCES, LAG_PERIODS, RATE_RISK_LAG_SOURCES, _add_lags

    vintage = active_vintage(ROOT)
    if vintage is None:
        raise ValueError("Panel export requires an active vintage")
    paths = project_paths(ROOT)
    scratch = replace(paths, reports=ROOT / "output/refresh/control_universe")
    bundles = [ROOT / "data/bundles/designs" / f"{job}__quarterly_bundle.csv" for job in MERGE_JOBS]
    manifest = _read_json(paths.manifests / f"{ANCHOR_JOB_ID}__design_manifest.json")
    manifest["bundle_path"] = str(bundles[0])
    factor_rows, _, _, factor_count = _load_factor_branch(
        scratch, job_id=ANCHOR_JOB_ID, design_manifest=manifest, k_screened=K_SCREENED,
        factor_count=FACTOR_COUNT, control_policy_mode=CONTROL_POLICY_MODE, min_coverage=0.4,
    )
    factor_ids = [f"dflmx_k{K_SCREENED}_f{i}" for i in range(1, FACTOR_COUNT + 1)]
    if factor_count != FACTOR_COUNT or [c for c in CANONICAL_CONTROL_IDS if c.startswith("dflmx_")] != factor_ids:
        raise ValueError("Factor controls differ from the frozen contract")
    rows = _merge_by_quarter(factor_rows, bundles)
    _add_lags(rows, CREDIT_LAG_SOURCES, LAG_PERIODS)
    _add_lags(rows, RATE_RISK_LAG_SOURCES, LAG_PERIODS)
    panel = projection_panel(rows, vintage.quarters)
    _write_csv(ROOT / FACTOR_SCORES_LOCATOR, [{"quarter": r["quarter"], **{k: r[k] for k in factor_ids}} for r in factor_rows])
    _write_csv(ROOT / PANEL_LOCATOR, panel)
    headline = [r for r in _read_csv(ROOT / "outputs/tables/tdc_same_quarter_headline.csv")
                if r["outcome_id"] == CANONICAL_OUTCOME_ID and r["treatment_id"] == CANONICAL_TREATMENT_ID]
    rolling = [r for r in _read_csv(ROOT / "output/models/tier2_rolling_selected_credit_rate_pass_through_estimates.csv")
               if r["outcome"] == CANONICAL_OUTCOME_ID and r["horizon"] == "0"]
    if len(headline) != 1 or tuple(r["window_end_quarter"] for r in rolling) != vintage.rolling_endpoints:
        raise ValueError("Retained OPEN-01 headline or rolling inventory differs from the vintage sample")
    gaps = {}
    fit = _fit_projection(panel, treatment_id=CANONICAL_TREATMENT_ID, outcome_id=CANONICAL_OUTCOME_ID,
                          control_ids=CANONICAL_CONTROL_IDS, covariance_lags=1)
    gaps["headline"] = max(abs(fit.beta - float(headline[0]["beta"])), abs(fit.se - float(headline[0]["se"])))
    for stored in rolling:
        end = quarter_ordinal(stored["window_end_quarter"])
        window = [r for r in panel if end - 47 <= quarter_ordinal(r["quarter"]) <= end]
        refit = _fit_projection(window, treatment_id=CANONICAL_TREATMENT_ID, outcome_id=CANONICAL_OUTCOME_ID,
                                control_ids=CANONICAL_CONTROL_IDS, covariance_lags=1)
        gaps[stored["window_end_quarter"]] = max(abs(refit.beta - float(stored["beta"])), abs(refit.se - float(stored["se"])))
    if max(gaps.values()) > ESTIMATE_TOLERANCE:
        raise ValueError(f"Exported panel does not reproduce retained OPEN-01 estimates: {max(gaps.values())}")
    target = ROOT / PANEL_RECEIPT_LOCATOR
    _write_json(target, {"vintage": vintage.record(), "panel": _record(ROOT, ROOT / PANEL_LOCATOR),
                         "full_factor_scores": _record(ROOT, ROOT / FACTOR_SCORES_LOCATOR),
                         "grid": {"quarters": len(factor_rows), "first": factor_rows[0]["quarter"], "last": factor_rows[-1]["quarter"]},
                         "control_ids": list(CANONICAL_CONTROL_IDS), "factor_ids": factor_ids,
                         "estimate_tolerance": ESTIMATE_TOLERANCE, "max_estimate_gap": max(gaps.values()),
                         "windows_compared": len(gaps)})
    return target


# --------------------------------------------------------------------------- OPEN-16


def _rolling_windows(quarters: Sequence[str], endpoints: Sequence[str]) -> list[tuple[str, list[str]]]:
    windows = [("headline", list(quarters))]
    for endpoint in endpoints:
        end = quarter_ordinal(endpoint)
        windows.append((endpoint, [q for q in quarters if end - 47 <= quarter_ordinal(q) <= end]))
    return windows


def open16_worker(panel_path: Path, factors_path: Path, policy_path: Path, destination: Path) -> None:
    """Fit every no-deletion window and its independent scaled-SVD reference."""
    import numpy as np
    from validate_open16_reproduction import (
        compare_scaled,
        input_inventory_hash,
        numerical_sanity,
        rank_projector,
        runtime_identity,
        scaled_geometry,
        svd_reference,
    )

    from ea_tdc.open01 import _fit_projection
    from ea_tdc.open_contract import CANONICAL_CONTROL_IDS, CANONICAL_OUTCOME_ID, CANONICAL_TREATMENT_ID
    from ea_tdc.vintage import active_vintage

    vintage = active_vintage(ROOT)
    policy = _read_json(policy_path)
    panel = [{k: (v if k == "quarter" else float(v)) for k, v in row.items()} for row in _read_csv(panel_path)]
    factors = [c for c in CANONICAL_CONTROL_IDS if c.startswith("dflmx_")]
    full = _read_csv(factors_path)
    factor_rank, _ = rank_projector([[float(r[k]) for k in factors] for r in full])
    sample_rank, _ = rank_projector([[r[k] for k in factors] for r in panel])
    if tuple(r["quarter"] for r in panel) != vintage.quarters or (factor_rank, sample_rank) != (4, 4):
        raise ValueError("Panel inventory or four-factor rank differs")
    windows = []
    for label, quarters in _rolling_windows(vintage.quarters, vintage.rolling_endpoints):
        observed = [r for r in panel if r["quarter"] in set(quarters)]
        fit = _fit_projection(observed, treatment_id=CANONICAL_TREATMENT_ID, outcome_id=CANONICAL_OUTCOME_ID,
                              control_ids=CANONICAL_CONTROL_IDS, covariance_lags=1)
        z = [[1.0, *[r[k] for k in fit.control_ids_used]] for r in observed]
        x = [[row[0], r[CANONICAL_TREATMENT_ID], *row[1:]] for row, r in zip(z, observed, strict=True)]
        y = [r[CANONICAL_OUTCOME_ID] for r in observed]
        geometry = scaled_geometry(x, y, policy)
        conditioning_rank, projector = rank_projector(z)
        _, full_projector = rank_projector(x)
        independent = svd_reference(x, y, geometry)
        evidence = numerical_sanity({"coefficients": fit.fit.beta, "fitted": fit.fit.fitted, "residuals": fit.fit.residuals,
                                     "covariance": fit.fit.covariance, "beta_se": [fit.beta, fit.se]},
                                    geometry["scales"], geometry["outcome_scale"], policy)
        current = {"coefficients": fit.fit.beta, "fitted": fit.fit.fitted, "residuals": fit.fit.residuals,
                   "covariance": evidence["covariance"], "raw_covariance": evidence["raw_covariance"],
                   "beta_se": [fit.beta, fit.se]}
        reference_gaps = compare_scaled(current, independent, geometry["scales"], geometry["outcome_scale"], policy)
        z_geometry = scaled_geometry(z, y, policy)
        projector_gaps = {"conditioning": float(np.linalg.norm(np.asarray(projector) - z_geometry["projector"], 2)),
                          "full_design": float(np.linalg.norm(np.asarray(full_projector) - geometry["projector"], 2))}
        if max(projector_gaps.values()) > policy["projector_tolerance"]:
            raise ValueError("Independent SVD projector differs")
        windows.append({"window": label, "quarters": [r["quarter"] for r in observed],
                        "controls_used": list(fit.control_ids_used), "controls_rejected": list(fit.control_ids_rejected),
                        "conditioning_rank": conditioning_rank, "design_rank": geometry["rank"],
                        "condition": geometry["condition"], "projector": projector, "full_projector": full_projector,
                        "scales": geometry["scales"].tolist(), "outcome_scale": geometry["outcome_scale"],
                        "input_inventory_sha256": input_inventory_hash(
                            [{k: r[k] for k in ("quarter", CANONICAL_OUTCOME_ID, CANONICAL_TREATMENT_ID, *CANONICAL_CONTROL_IDS)}
                             for r in observed],
                            ["quarter", CANONICAL_OUTCOME_ID, CANONICAL_TREATMENT_ID, *CANONICAL_CONTROL_IDS]),
                        "independent_svd": independent, "independent_svd_differences": reference_gaps,
                        "independent_projector_differences": projector_gaps, **current})
    _write_json(destination, {"runtime": runtime_identity(), "factor_rank": factor_rank, "sample_factor_rank": sample_rank,
                              "numerical_policy": policy, "windows": windows})


def compare_runtimes(first: Mapping[str, Any], second: Mapping[str, Any], policy: Mapping[str, Any], expected_windows: int) -> dict[str, Any]:
    import numpy as np
    from validate_open16_reproduction import compare_scaled

    if (len(first["windows"]) != expected_windows or len(second["windows"]) != expected_windows
            or first["runtime"] == second["runtime"]
            or any(r[k] != 4 for r in (first, second) for k in ("factor_rank", "sample_factor_rank"))
            or any(r["numerical_policy"] != policy for r in (first, second))):
        raise ValueError("Validation needs two distinct runtimes, the exact policy, four-factor ranks and every window")
    rows = []
    for a, b in zip(first["windows"], second["windows"], strict=True):
        for key in ("window", "quarters", "controls_used", "controls_rejected", "conditioning_rank", "design_rank",
                    "input_inventory_sha256"):
            if a[key] != b[key]:
                raise ValueError(f"Cross-runtime discrete design identity differs: {key}")
        if max(a["condition"], b["condition"]) > policy["column_normalized_condition_cap"]:
            raise ValueError("Cross-runtime condition cap exceeded")
        gaps = compare_scaled(a, b, a["scales"], a["outcome_scale"], policy)
        for left, right in ((a, b["independent_svd"]), (b, a["independent_svd"]), (a["independent_svd"], b["independent_svd"])):
            for key, value in compare_scaled(left, right, a["scales"], a["outcome_scale"], policy).items():
                gaps[key] = max(gaps[key], value)
        for key in ("projector", "full_projector"):
            gaps[key] = float(np.linalg.norm(np.asarray(a[key]) - np.asarray(b[key]), 2))
            if gaps[key] > policy["projector_tolerance"]:
                raise ValueError("Cross-runtime projector differs")
        rows.append({"window": a["window"], **gaps})
    return {"windows": rows, "maxima": {k: max(r[k] for r in rows) for k in rows[0] if k != "window"}}


def lane_open16(vintage: Vintage, commit: str, comparison_python: str) -> Path:
    import numpy as np  # noqa: F401 - the diagnostic runtime must carry the pinned numerical stack
    from run_open16_diagnostics import _write_calendar_evidence
    from validate_open16_reproduction import runtime_identity

    from ea_tdc import open16
    from ea_tdc.open01 import _fit_projection
    from ea_tdc.open_contract import CANONICAL_CONTROL_IDS, CANONICAL_OUTCOME_ID, CANONICAL_TREATMENT_ID
    from ea_tdc.vintage import open16_sample

    _verify_clean_commit(ROOT, commit)
    output = _output_root(vintage)
    run_root = output / RUN_DIR
    open01 = _read_json(output / "open01_receipt.json")
    if open01["producer_commit"] != commit:
        raise ValueError("OPEN-01 lane ran from another commit")
    for record in (open01["producer_receipt"], open01["panel_receipt"]):
        if sha256_file(run_root / record["path"]) != record["sha256"]:
            raise ValueError(f"OPEN-01 lane object changed: {record['path']}")
    panel_receipt = _read_json(run_root / PANEL_RECEIPT_LOCATOR)
    for record in (panel_receipt["panel"], panel_receipt["full_factor_scores"]):
        if sha256_file(run_root / record["path"]) != record["sha256"]:
            raise ValueError(f"Projection panel changed: {record['path']}")
    authority = _read_json(ROOT / "config/open16_reproduction_authority.json")
    policy = authority["numerical_policy"]
    environments = authority["environments"]
    if runtime_identity() not in environments:
        raise ValueError("Diagnostics must run in a pinned numerical runtime")
    target = output / "open16"
    if target.exists():
        raise ValueError("OPEN-16 vintage output already exists; no overwrite")
    stage = Path(tempfile.mkdtemp(prefix=".open16-", dir=output))
    try:
        _write_json(stage / "validation" / "policy.json", policy)
        env = _python_env(run_root)
        commands = {}
        for name, python in (("primary", sys.executable), ("comparison", comparison_python)):
            commands[name] = _run_logged(
                [python, "-B", "scripts/run_vintage_refresh.py", "open16-worker",
                 "--panel", str(run_root / PANEL_LOCATOR), "--factors", str(run_root / FACTOR_SCORES_LOCATOR),
                 "--policy", str(stage / "validation" / "policy.json"),
                 "--destination", str(stage / f".{name}.json")],
                cwd=run_root, env=env, log=stage / "validation" / f"{name}.log")
        first, second = (_read_json(stage / f".{name}.json") for name in ("primary", "comparison"))
        runtimes = [first["runtime"], second["runtime"]]
        if sorted(json.dumps(r, sort_keys=True) for r in runtimes) != sorted(
                json.dumps(e, sort_keys=True) for e in environments):
            raise ValueError("Validation runtimes are not exactly the two pinned environments")
        windows = 1 + len(vintage.rolling_endpoints)
        comparison = compare_runtimes(first, second, policy, windows)
        headline = [r for r in _read_csv(run_root / "outputs/tables/tdc_same_quarter_headline.csv")
                    if r["outcome_id"] == CANONICAL_OUTCOME_ID and r["treatment_id"] == CANONICAL_TREATMENT_ID][0]
        rolling = {r["window_end_quarter"]: r for r in
                   _read_csv(run_root / "output/models/tier2_rolling_selected_credit_rate_pass_through_estimates.csv")
                   if r["outcome"] == CANONICAL_OUTCOME_ID and r["horizon"] == "0"}
        economics = {}
        for window in first["windows"]:
            stored = headline if window["window"] == "headline" else rolling[window["window"]]
            beta, se = window["beta_se"]
            economics[window["window"]] = max(abs(beta - float(stored["beta"])), abs(se - float(stored["se"])))
        if max(economics.values()) > policy["treatment_beta_se_absolute_tolerance"]:
            raise ValueError("Validated fits do not reproduce the retained OPEN-01 estimates")
        for name, payload in (("primary", first), ("comparison", second)):
            compact = {**payload, "windows": [{k: v for k, v in w.items() if k not in ("projector", "full_projector")}
                                              for w in payload["windows"]]}
            _write_json(stage / "validation" / f"{name}.json", compact)
            (stage / f".{name}.json").unlink()
        gates = {"rank_partition": True, "projector_equivalence": True, "economics_equivalence": True}
        _write_json(stage / "validation" / "numerical_comparison.json",
                    {"numerical_policy": policy, **comparison, "economics_max_gap": max(economics.values()),
                     "commands": commands, "gates": gates})

        panel = _read_csv(run_root / PANEL_LOCATOR)
        legs_dir = output / "tdcest" / "data" / "processed"
        legs = _read_csv(legs_dir / "tdc_regression_legs_v1.csv")
        legs_receipt = _read_json(legs_dir / "tdc_regression_legs_v1_receipt.json")
        if (legs_receipt.get("schema_version") != "regression_legs_v1" or legs_receipt.get("units") != "USD million"
                or legs_receipt.get("sample") != {"start": vintage.sample_start, "end": vintage.sample_end, "n": vintage.observations}
                or legs_receipt.get("output", {}).get("sha256") != sha256_file(legs_dir / "tdc_regression_legs_v1.csv")):
            raise ValueError("Treatment-leg receipt does not bind the vintage legs")
        with open16_sample(vintage) as binding:
            rows = open16.join_legs(panel, legs)
            beta = _fit_projection(rows, treatment_id=CANONICAL_TREATMENT_ID, outcome_id=CANONICAL_OUTCOME_ID,
                                   control_ids=CANONICAL_CONTROL_IDS).beta
            covariance = open16.covariance_contributions(rows, beta)
            rolling_rows = open16.pandemic_path(rows, nominal_endpoints=vintage.rolling_endpoints)
        if abs(beta - float(headline["beta"])) > policy["treatment_beta_se_absolute_tolerance"]:
            raise ValueError("Diagnostic beta differs from the retained OPEN-01 headline")
        calendar = _write_calendar_evidence(stage, rolling_rows)
        _write_csv(stage / "leg_covariance_contributions.csv", covariance)
        _write_csv(stage / "pandemic_calendar_deletion.csv", rolling_rows)
        preflight = {"status": "not_run_scope_excluded", "reason_codes": ["leg_estimation_outside_fresh_scope"],
                     "leg_estimates_computed": False, "source_resolution": legs_receipt["source_resolution"]}
        _write_json(stage / "leg_preflight.json", preflight)
        outputs = {p.relative_to(stage).as_posix(): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                   for p in sorted(stage.rglob("*")) if p.is_file()}
        receipt = {"schema_version": "ea_tdc_vintage_open16_v1", "producer_commit": commit,
                   "status": "completed_bounded_vintage_diagnostics", "scientific_status": "appendix_only",
                   "authority_class": authority["authority_class"], "scope": authority["scope"],
                   "vintage": vintage.record(), "sample_binding": binding,
                   "inputs": {"open01_lane": _record(output, output / "open01_receipt.json"),
                              "panel": panel_receipt["panel"], "full_factor_scores": panel_receipt["full_factor_scores"],
                              "legs": _record(output, legs_dir / "tdc_regression_legs_v1.csv"),
                              "legs_receipt": _record(output, legs_dir / "tdc_regression_legs_v1_receipt.json"),
                              "authority": _record(ROOT, ROOT / "config/open16_reproduction_authority.json")},
                   "validation_gates": gates, "runtimes": runtimes, "headline_beta": beta,
                   "calendar_covariance": calendar, "leg_estimates_computed": False,
                   "leg_preflight_reason_codes": preflight["reason_codes"], "outputs": outputs, **open16.DISCLOSURES}
        _write_json(stage / "receipt.json", receipt)
        _verify_clean_commit(ROOT, commit)
        os.replace(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return target / "receipt.json"


# --------------------------------------------------------------------------- seal


def lane_seal(vintage: Vintage, commit: str) -> Path:
    _verify_clean_commit(ROOT, commit)
    output = _output_root(vintage)
    run_root = output / RUN_DIR
    baseline = _read_json(output / "protected_baseline.json")
    current = protected_inventory(ROOT, output)
    changed = sorted(k for k, v in baseline["files"].items() if current.get(k) != v)
    added = sorted(set(current) - set(baseline["files"]))
    lanes = {}
    for name in ("acquisition", "materialize", "open01", "open16"):
        path = {"acquisition": output / "inputs/acquisition_receipt.json", "materialize": output / "materialize_receipt.json",
                "open01": output / "open01_receipt.json", "open16": output / "open16/receipt.json"}[name]
        lanes[name] = _record(output, path)
    open01 = _read_json(run_root / "output/manifests/open01_producer_run_receipt.json")
    retained = {key: record for key, record in open01["retained_outputs"]["fixed"].items()}
    for record in retained.values():
        if sha256_file(run_root / record["path"]) != record["sha256"]:
            raise ValueError(f"OPEN-01 retained output changed: {record['path']}")
    declared = open01["retained_outputs"]["manifest_declared"]
    for record in declared.values():
        if sha256_file(run_root / record["path"]) != record["sha256"]:
            raise ValueError(f"OPEN-01 declared output changed: {record['path']}")
    open16 = _read_json(output / "open16/receipt.json")
    for name, record in open16["outputs"].items():
        if name != "receipt.json" and sha256_file(output / "open16" / name) != record["sha256"]:
            raise ValueError(f"OPEN-16 output changed: {name}")
    admitted_grid = [r["quarter"] for r in _read_csv(ROOT / "data/bundles/designs/tdc_tier2_mmf_rrp_canonical_full_panel__quarterly_bundle.csv")]
    vintage_grid = [r["quarter"] for r in _read_csv(run_root / "data/bundles/designs/tdc_tier2_mmf_rrp_canonical_full_panel__quarterly_bundle.csv")]
    availability = _read_json(output / "inputs/open02_source_availability.json")
    acquisition = _read_json(output / "inputs/acquisition_receipt.json")
    materialized = _read_json(output / "materialize_receipt.json")
    open01_lane = _read_json(output / "open01_receipt.json")
    receipt = {
        "schema_version": "ea_tdc_vintage_refresh_receipt_v1",
        "producer_commit": commit,
        "sealed_at_utc": _now(),
        "vintage": vintage.record(),
        "authority": vintage.payload["authority"],
        "inputs": {
            "tdcest_receipt_sha256": vintage.payload["tdcest"]["receipt_sha256"],
            "tdcest_files": vintage.payload["tdcest"]["files"],
            "tdcest_site_bundle": acquisition["tdcest"]["site_bundle"],
            "tdcpass": acquisition["tdcpass"],
            "z1_release": vintage.payload["z1_release"],
            "fred_vintages": {k: {f: v[f] for f in ("last_updated", "realtime_start", "observation_end", "sha256")}
                              for k, v in acquisition["fred_raw"].items()},
            "factor_seed_refreshed": {k: {f: v[f] for f in ("last_updated", "realtime_start", "sha256")}
                                      for k, v in acquisition["factor_seed"]["files"].items() if v["status"] == "refreshed"},
            "carried_bundles": acquisition["carried_bundles"],
            "tdcpass_retired_series_carried": materialized["tdcpass_retired_series_carried"],
            "observation_cutoff": materialized["observation_cutoff"],
        },
        "lanes": lanes,
        "open01": {"producer_receipt": _record(output, run_root / "output/manifests/open01_producer_run_receipt.json"),
                   "host": open01_lane["host"], "run_id": open01_lane["run_id"],
                   "retained_outputs": retained,
                   "design_grid": {"admitted_quarters": len(admitted_grid), "vintage_quarters": len(vintage_grid),
                                   "identical": admitted_grid == vintage_grid}},
        "open16": {"receipt": _record(output, output / "open16/receipt.json"),
                   "outputs": {k: v for k, v in open16["outputs"].items()}},
        "open02": {"status": "not_run",
                   "disposition": availability["disposition"],
                   "absent_series_ids": availability["absent_series_ids"],
                   "reason": "allowlisted response series are not published in the pinned release; substitution would change the specification",
                   "availability": _record(output, output / "inputs/open02_source_availability.json")},
        "protected_admitted_surface": {"baseline": _record(output, output / "protected_baseline.json"),
                                       "files": len(baseline["files"]), "changed": changed, "added": added[:200],
                                       "added_count": len(added)},
    }
    target = output / "receipt.json"
    if target.exists():
        raise ValueError("Refresh receipt already exists; no overwrite")
    _write_json(target, receipt)
    return target


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lane", choices=("snapshot", "acquire", "materialize", "open01", "export-panel",
                                         "open16", "open16-worker", "seal"))
    parser.add_argument("--vintage", default=None)
    parser.add_argument("--producer-commit")
    parser.add_argument("--run-id")
    parser.add_argument("--comparison-python")
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--factors", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args(argv)
    if args.lane == "export-panel":
        print(export_panel())
        return
    if args.lane == "open16-worker":
        open16_worker(args.panel, args.factors, args.policy, args.destination)
        return
    locator = args.vintage or env_vintage_locator(os.environ)
    os.environ[VINTAGE_ENV] = locator
    vintage = load_vintage(ROOT, locator)
    if not args.producer_commit:
        parser.error("--producer-commit is required")
    if args.lane == "snapshot":
        result = lane_snapshot(vintage, args.producer_commit)
    elif args.lane == "acquire":
        result = lane_acquire(vintage, args.producer_commit)
    elif args.lane == "materialize":
        result = lane_materialize(vintage, args.producer_commit)
    elif args.lane == "open01":
        if not args.run_id:
            parser.error("--run-id is required")
        result = lane_open01(vintage, args.producer_commit, args.run_id)
    elif args.lane == "open16":
        if not args.comparison_python:
            parser.error("--comparison-python is required")
        result = lane_open16(vintage, args.producer_commit, args.comparison_python)
    else:
        result = lane_seal(vintage, args.producer_commit)
    print(json.dumps({"lane": args.lane, "receipt": str(result.relative_to(ROOT)), "sha256": sha256_file(result)}))


if __name__ == "__main__":
    main()
