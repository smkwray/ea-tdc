from __future__ import annotations

import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from .config import RuntimeConfig
from .paths import ProjectPaths
from .utils import utc_now_iso, write_json

FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"
FRED_GRAPH_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
TREASURY_API_BASE = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/"


def _urlopen_text(url: str, timeout: int = 60) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "ea-tdc/0.1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def _normalize_graph_csv_payload(text: str) -> list[dict[str, str]]:
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        return []
    value_key = next((key for key in rows[0].keys() if key != "observation_date"), None)
    if value_key is None:
        return []
    return [
        {
            "date": str(row.get("observation_date", "")).strip(),
            "value": str(row.get(value_key, "")).strip(),
        }
        for row in rows
        if str(row.get("observation_date", "")).strip()
    ]


def load_fred_seed_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    return rows


def load_treasury_manifest(path: Path) -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    datasets = payload.get("datasets", {})
    if not isinstance(datasets, dict):
        raise TypeError("Expected 'datasets' mapping in treasury manifest")
    clean: dict[str, dict[str, Any]] = {}
    for key, value in datasets.items():
        if not isinstance(value, dict):
            raise TypeError(f"Expected dataset '{key}' to be a mapping")
        clean[str(key)] = value
    return clean


def fred_csv_url(series_id: str, start_date: str | None = None, end_date: str | None = None) -> str:
    params: dict[str, str] = {"id": series_id}
    if start_date:
        params["cosd"] = start_date
    if end_date:
        params["coed"] = end_date
    return f"{FRED_GRAPH_CSV_BASE}?{urllib.parse.urlencode(params)}"


def fred_api_url(series_id: str, api_key: str, start_date: str | None = None, end_date: str | None = None) -> str:
    params: dict[str, str] = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    if start_date:
        params["observation_start"] = start_date
    if end_date:
        params["observation_end"] = end_date
    return f"{FRED_API_BASE}?{urllib.parse.urlencode(params)}"


def _download_series_csv(
    *,
    series_id: str,
    destination: Path,
    api_key: str | None,
    start_date: str | None,
    end_date: str | None,
    allow_graph_csv_fallback: bool,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if api_key:
        url = fred_api_url(series_id, api_key=api_key, start_date=start_date, end_date=end_date)
        payload = json.loads(_urlopen_text(url))
        rows = payload.get("observations", [])
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["date", "value"])
            writer.writeheader()
            for row in rows:
                writer.writerow({"date": row.get("date", ""), "value": row.get("value", "")})
        return {"mode": "fred_api_json", "url": url, "rows": len(rows)}

    if not allow_graph_csv_fallback:
        raise RuntimeError(f"Missing FRED API key for {series_id} and graph CSV fallback disabled")

    url = fred_csv_url(series_id, start_date=start_date, end_date=end_date)
    payload = _normalize_graph_csv_payload(_urlopen_text(url))
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "value"])
        writer.writeheader()
        writer.writerows(payload)
    row_count = len(payload)
    return {"mode": "fred_graph_csv", "url": url, "rows": row_count}


def download_fred_bundle(
    runtime: RuntimeConfig,
    paths: ProjectPaths,
    *,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    manifest_rows = load_fred_seed_manifest(runtime.fred_series_manifest)
    if limit is not None:
        manifest_rows = manifest_rows[:limit]

    api_key = None
    if runtime.fred_api_key_env:
        api_key = __import__("os").environ.get(runtime.fred_api_key_env)

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for item in manifest_rows:
        series_id = (item.get("series_id") or "").strip()
        if not series_id:
            continue
        destination = paths.raw_fred / f"{series_id}.csv"
        if destination.exists() and not force:
            results.append(
                {
                    "series_id": series_id,
                    "path": str(destination),
                    "status": "skipped_existing",
                }
            )
            continue
        try:
            download_meta = _download_series_csv(
                series_id=series_id,
                destination=destination,
                api_key=api_key,
                start_date=runtime.default_start_date,
                end_date=runtime.default_end_date,
                allow_graph_csv_fallback=runtime.allow_graph_csv_fallback,
            )
            results.append(
                {
                    "series_id": series_id,
                    "path": str(destination),
                    "status": "downloaded",
                    **download_meta,
                }
            )
        except Exception as exc:
            errors.append({"series_id": series_id, "error": str(exc)})

    manifest = {
        "kind": "fred_download_manifest",
        "downloaded_at_utc": utc_now_iso(),
        "source_manifest": str(runtime.fred_series_manifest),
        "results": results,
        "errors": errors,
    }
    write_json(paths.manifests / "fred_download_manifest.json", manifest)
    return manifest


def _build_treasury_url(endpoint: str, params: dict[str, Any]) -> str:
    query = urllib.parse.urlencode({str(k): str(v) for k, v in params.items()})
    return f"{TREASURY_API_BASE}{endpoint}?{query}"


def _download_treasury_dataset(
    *,
    dataset_name: str,
    spec: dict[str, Any],
    destination: Path,
    max_pages: int | None,
) -> dict[str, Any]:
    endpoint = str(spec.get("endpoint", "")).strip()
    if not endpoint:
        raise ValueError(f"Treasury dataset '{dataset_name}' is missing an endpoint")
    params = {str(k): v for k, v in dict(spec.get("params", {})).items()}
    params.setdefault("format", "json")
    params.setdefault("page[size]", "10000")

    rows: list[dict[str, Any]] = []
    page = 1
    pages_fetched = 0
    last_url = ""

    while True:
        query = dict(params)
        query["page[number]"] = str(page)
        url = _build_treasury_url(endpoint, query)
        last_url = url
        payload = json.loads(_urlopen_text(url, timeout=120))
        rows.extend(payload.get("data", []) or [])
        pages_fetched += 1

        next_url = (payload.get("links", {}) or {}).get("next")
        if max_pages is not None and pages_fetched >= max_pages:
            break
        if not next_url:
            break
        page += 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(str(key))
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        destination.write_text("", encoding="utf-8")

    return {
        "dataset": dataset_name,
        "endpoint": endpoint,
        "path": str(destination),
        "rows": len(rows),
        "pages_fetched": pages_fetched,
        "url": last_url,
    }


def download_treasury_bundle(
    runtime: RuntimeConfig,
    paths: ProjectPaths,
    *,
    dataset_name: str | None = None,
    max_pages: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    manifest = load_treasury_manifest(runtime.treasury_dataset_manifest)
    selected = {dataset_name: manifest[dataset_name]} if dataset_name else manifest
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for name, spec in selected.items():
        destination = paths.raw_treasury / f"{name}.csv"
        if destination.exists() and not force:
            results.append({"dataset": name, "path": str(destination), "status": "skipped_existing"})
            continue
        try:
            meta = _download_treasury_dataset(
                dataset_name=name,
                spec=spec,
                destination=destination,
                max_pages=max_pages,
            )
            results.append({"status": "downloaded", **meta})
        except Exception as exc:
            errors.append({"dataset": name, "error": str(exc)})

    payload = {
        "kind": "treasury_download_manifest",
        "downloaded_at_utc": utc_now_iso(),
        "source_manifest": str(runtime.treasury_dataset_manifest),
        "results": results,
        "errors": errors,
    }
    write_json(paths.manifests / "treasury_download_manifest.json", payload)
    return payload
