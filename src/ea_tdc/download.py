from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import yaml

from .config import RuntimeConfig
from .open_contract import get_open02_contract
from .paths import ProjectPaths
from .utils import utc_now_iso, write_json

FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"
FRED_GRAPH_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
TREASURY_API_BASE = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/"
_OPEN02_SOURCE_CONTRACT = get_open02_contract().source
OPEN02_BOARD_ARCHIVE_URL = _OPEN02_SOURCE_CONTRACT.archive_url
OPEN02_BOARD_ARCHIVE_RELEASE_DATE = _OPEN02_SOURCE_CONTRACT.release_date
OPEN02_BOARD_ARCHIVE_SHA256 = _OPEN02_SOURCE_CONTRACT.archive_sha256
OPEN02_BOARD_CSV_MEMBER_SHA256 = _OPEN02_SOURCE_CONTRACT.csv_member_sha256
OPEN02_BOARD_DICTIONARY_MEMBER_SHA256 = (
    _OPEN02_SOURCE_CONTRACT.dictionary_member_sha256
)
OPEN02_BOARD_UNIT_LABEL = _OPEN02_SOURCE_CONTRACT.unit_label


def _urlopen_bytes(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ea-tdc/0.1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _urlopen_text(url: str, timeout: int = 60) -> str:
    return _urlopen_bytes(url, timeout=timeout).decode("utf-8")


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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _quarter_number(label: str) -> int:
    if (
        len(label) != 6
        or label[4] != "Q"
        or not label[:4].isdigit()
        or label[5] not in "1234"
    ):
        raise ValueError(f"Malformed quarter label: {label!r}")
    return int(label[:4]) * 4 + int(label[5]) - 1


def _open02_quarters() -> tuple[str, ...]:
    sample = get_open02_contract().sample
    start = _quarter_number(sample.start_quarter)
    end = _quarter_number(sample.end_quarter)
    quarters = tuple(
        f"{number // 4}Q{number % 4 + 1}"
        for number in range(start, end + 1)
    )
    if len(quarters) != sample.observations:
        raise RuntimeError("OPEN-02 sample contract has inconsistent quarter bounds")
    return quarters


def _parse_board_member(
    member_name: str,
    payload: bytes,
    archive_quarters: set[str],
) -> tuple[tuple[str, ...], dict[str, dict[str, str]]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError(f"Board archive member {member_name} is not UTF-8") from None
    reader = csv.reader(io.StringIO(text))
    try:
        header = tuple(next(reader))
    except StopIteration:
        raise ValueError(f"Board archive member {member_name} is empty") from None
    if header.count("date") != 1:
        raise ValueError(
            f"Board archive member {member_name} must contain one date column"
        )
    if len(set(header)) != len(header):
        raise ValueError(f"Board archive member {member_name} has duplicate columns")

    rows: dict[str, dict[str, str]] = {}
    for row_number, values in enumerate(reader, start=2):
        if len(values) != len(header):
            raise ValueError(
                f"Board archive member {member_name} row {row_number} is malformed"
            )
        row = dict(zip(header, values, strict=True))
        quarter = row["date"].strip()
        if quarter not in archive_quarters:
            continue
        if quarter in rows:
            raise ValueError(
                f"Board archive member {member_name} duplicates {quarter}"
            )
        rows[quarter] = row

    missing = sorted(archive_quarters.difference(rows))
    if missing:
        raise ValueError(
            f"Board archive member {member_name} is missing "
            f"{len(missing)} required quarters"
        )
    return header, rows


def _parse_board_dictionary(
    member_name: str,
    payload: bytes,
) -> dict[str, dict[str, str]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError(
            f"Board data dictionary {member_name} is not UTF-8"
        ) from None
    entries: dict[str, dict[str, str]] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split("\t")
        if len(fields) != 5:
            raise ValueError(
                f"Board data dictionary {member_name} line {line_number} "
                "is malformed"
            )
        series_id, description, table_line, table, units = fields
        if not series_id or series_id in entries:
            raise ValueError(
                f"Board data dictionary {member_name} has an empty or duplicate "
                f"series ID at line {line_number}"
            )
        entries[series_id] = {
            "official_description": description,
            "table_line": table_line,
            "table": table,
            "unit_label": units,
        }
    if not entries:
        raise ValueError(f"Board data dictionary {member_name} is empty")
    return entries


def parse_open02_board_archive(archive_bytes: bytes) -> dict[str, Any]:
    """Verify and parse the frozen pre-cutoff Board Z.1 archive."""

    if not isinstance(archive_bytes, bytes) or not archive_bytes:
        raise TypeError("Board archive payload must be nonempty bytes")
    archive_sha256 = _sha256_bytes(archive_bytes)
    if archive_sha256 != OPEN02_BOARD_ARCHIVE_SHA256:
        raise ValueError(
            "Board archive SHA-256 mismatch: "
            f"expected {OPEN02_BOARD_ARCHIVE_SHA256}, got {archive_sha256}"
        )

    csv_member_sha256 = dict(OPEN02_BOARD_CSV_MEMBER_SHA256)
    dictionary_member_sha256 = dict(
        OPEN02_BOARD_DICTIONARY_MEMBER_SHA256
    )
    expected_members = {
        **csv_member_sha256,
        **dictionary_member_sha256,
    }
    member_payloads: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            names = archive.namelist()
            for member_name, expected_sha256 in expected_members.items():
                if names.count(member_name) != 1:
                    raise ValueError(
                        f"Board archive must contain one {member_name} member"
                    )
                payload = archive.read(member_name)
                actual_sha256 = _sha256_bytes(payload)
                if actual_sha256 != expected_sha256:
                    raise ValueError(
                        f"Board archive member {member_name} SHA-256 mismatch: "
                        f"expected {expected_sha256}, got {actual_sha256}"
                    )
                member_payloads[member_name] = payload
    except zipfile.BadZipFile:
        raise ValueError("Board archive payload is not a valid ZIP file") from None

    quarters = _open02_quarters()
    archive_quarters = {
        f"{quarter[:4]}:{quarter[4:]}"
        for quarter in quarters
    }
    headers: dict[str, tuple[str, ...]] = {}
    rows_by_member: dict[str, dict[str, dict[str, str]]] = {}
    for member_name in csv_member_sha256:
        payload = member_payloads[member_name]
        header, rows = _parse_board_member(
            member_name,
            payload,
            archive_quarters,
        )
        headers[member_name] = header
        rows_by_member[member_name] = rows

    dictionaries = {
        member_name: _parse_board_dictionary(
            member_name,
            member_payloads[member_name],
        )
        for member_name in dictionary_member_sha256
    }
    contract = get_open02_contract()
    member_by_series: dict[str, str] = {}
    series_metadata: list[dict[str, Any]] = []
    for series in contract.series:
        locations = [
            member_name
            for member_name, header in headers.items()
            if series.board_series_id in header
        ]
        if len(locations) != 1:
            raise ValueError(
                f"Board series {series.board_series_id} must occur in exactly "
                f"one verified archive member; found {len(locations)}"
            )
        member_name = locations[0]
        dictionary_locations = [
            dictionary_member
            for dictionary_member, entries in dictionaries.items()
            if series.board_series_id in entries
        ]
        if len(dictionary_locations) != 1:
            raise ValueError(
                f"Board series {series.board_series_id} must occur in exactly "
                f"one verified data dictionary; found "
                f"{len(dictionary_locations)}"
            )
        dictionary_member = dictionary_locations[0]
        expected_dictionary_member = (
            member_name.replace("csv/", "data_dictionary/")
            .removesuffix(".csv")
            + ".txt"
        )
        if dictionary_member != expected_dictionary_member:
            raise ValueError(
                f"Board series {series.board_series_id} CSV and dictionary "
                "members do not align"
            )
        dictionary_entry = dictionaries[dictionary_member][
            series.board_series_id
        ]
        official_description = dictionary_entry["official_description"]
        dictionary_side = official_description.rsplit("; ", 1)[-1].casefold()
        if dictionary_side != series.side:
            raise ValueError(
                f"Board series {series.board_series_id} side "
                f"{dictionary_side!r} does not match {series.side!r}"
            )
        expected_description = series.official_title.removesuffix(
            ", Transactions"
        )
        if official_description.casefold() != expected_description.casefold():
            raise ValueError(
                f"Board series {series.board_series_id} official description "
                "does not match the frozen contract"
            )
        if dictionary_entry["unit_label"] != OPEN02_BOARD_UNIT_LABEL:
            raise ValueError(
                f"Board series {series.board_series_id} has unexpected unit label "
                f"{dictionary_entry['unit_label']!r}"
            )
        member_by_series[series.key] = member_name
        series_metadata.append(
            {
                "key": series.key,
                "fred_id": series.fred_id,
                "board_series_id": series.board_series_id,
                "archive_member": member_name,
                "dictionary_member": dictionary_member,
                **dictionary_entry,
                "side": dictionary_side,
                "units": series.units,
                "seasonal_adjustment": series.seasonal_adjustment,
            }
        )

    clean_rows: list[dict[str, Any]] = []
    for quarter in quarters:
        archive_quarter = f"{quarter[:4]}:{quarter[4:]}"
        clean_row: dict[str, Any] = {"quarter": quarter}
        for series in contract.series:
            member_name = member_by_series[series.key]
            raw_value = rows_by_member[member_name][archive_quarter][
                series.board_series_id
            ].strip()
            try:
                value = float(raw_value)
            except ValueError:
                raise ValueError(
                    f"Board series {series.board_series_id} has malformed value "
                    f"at {quarter}: {raw_value!r}"
                ) from None
            if not math.isfinite(value):
                raise ValueError(
                    f"Board series {series.board_series_id} has non-finite value "
                    f"at {quarter}"
                )
            clean_row[series.key] = value
        clean_rows.append(clean_row)

    rows_json = json.dumps(
        clean_rows,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "metadata": {
            "kind": "open02_board_z1_archive",
            "source_url": OPEN02_BOARD_ARCHIVE_URL,
            "release_date": OPEN02_BOARD_ARCHIVE_RELEASE_DATE,
            "observation_vintage_cutoff": (
                contract.sample.observation_vintage_cutoff
            ),
            "archive_sha256": archive_sha256,
            "csv_member_sha256": csv_member_sha256,
            "dictionary_member_sha256": dictionary_member_sha256,
            "rows_sha256": _sha256_bytes(rows_json),
            "sample_start": contract.sample.start_quarter,
            "sample_end": contract.sample.end_quarter,
            "observations": len(clean_rows),
            "series_count": len(contract.series),
            "series": series_metadata,
        },
        "rows": clean_rows,
    }


def fetch_open02_board_archive(*, timeout: int = 120) -> dict[str, Any]:
    """Fetch the one pinned OPEN-02 archive; never fall back to current data."""

    try:
        archive_bytes = _urlopen_bytes(
            OPEN02_BOARD_ARCHIVE_URL,
            timeout=timeout,
        )
    except Exception:
        raise RuntimeError(
            "Federal Reserve OPEN-02 Board archive request failed"
        ) from None
    return parse_open02_board_archive(archive_bytes)


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
