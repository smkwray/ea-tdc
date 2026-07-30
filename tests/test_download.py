from __future__ import annotations

import csv
import hashlib
import io
import zipfile

import pytest

from ea_tdc import download
from ea_tdc.download import (
    _normalize_graph_csv_payload,
    fetch_open02_board_archive,
    parse_open02_board_archive,
)
from ea_tdc.open_contract import get_open02_contract


def test_normalize_graph_csv_payload_maps_observation_date_to_date_value() -> None:
    payload = _normalize_graph_csv_payload(
        "observation_date,SP500\n2024-01-02,4700.0\n2024-01-03,4715.2\n"
    )

    assert payload == [
        {"date": "2024-01-02", "value": "4700.0"},
        {"date": "2024-01-03", "value": "4715.2"},
    ]


def _fixture_archive(
    *,
    missing_quarter: tuple[str, str] | None = None,
    bad_value: tuple[str, str, str] | None = None,
    bad_description: tuple[str, str] | None = None,
    bad_side: tuple[str, str] | None = None,
    bad_unit: tuple[str, str] | None = None,
    duplicate_dictionary_series: tuple[str, str] | None = None,
) -> tuple[bytes, dict[str, bytes]]:
    quarters = tuple(
        f"{year}Q{quarter}"
        for year in range(2002, 2026)
        for quarter in range(1, 5)
    )
    contract = get_open02_contract()
    csv_members = {
        "csv/fu111.csv": [
            series
            for series in contract.series
            if series.board_series_id.startswith("FU76")
        ],
        "csv/fu112.csv": [
            series
            for series in contract.series
            if series.board_series_id.startswith("FU75")
        ],
        "csv/fu113.csv": [
            series
            for series in contract.series
            if series.board_series_id.startswith("FU74")
        ],
    }
    series_index = {
        series.key: index
        for index, series in enumerate(contract.series, start=1)
    }
    payloads: dict[str, bytes] = {}
    for member_name, series_rows in csv_members.items():
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            ["date", *(series.board_series_id for series in series_rows)]
        )
        for quarter_index, quarter in enumerate(quarters):
            if missing_quarter == (member_name, quarter):
                continue
            values: list[str] = []
            for series in series_rows:
                value = str(series_index[series.key] * 1000 + quarter_index)
                if bad_value == (series.key, quarter, member_name):
                    value = "ND"
                values.append(value)
            writer.writerow(
                [f"{quarter[:4]}:{quarter[4:]}", *values]
            )
        payloads[member_name] = stream.getvalue().encode("utf-8")

        dictionary_name = (
            member_name.replace("csv/", "data_dictionary/")
            .removesuffix(".csv")
            + ".txt"
        )
        lines: list[str] = []
        for line_number, series in enumerate(series_rows, start=1):
            description = series.official_title.removesuffix(", Transactions")
            if bad_description and bad_description[0] == series.key:
                description = bad_description[1]
            if bad_side and bad_side[0] == series.key:
                description = (
                    description.rsplit("; ", 1)[0] + f"; {bad_side[1]}"
                )
            unit = download.OPEN02_BOARD_UNIT_LABEL
            if bad_unit and bad_unit[0] == series.key:
                unit = bad_unit[1]
            lines.append(
                "\t".join(
                    (
                        series.board_series_id,
                        description,
                        f"Line {line_number}",
                        member_name,
                        unit,
                    )
                )
            )
        payloads[dictionary_name] = ("\n".join(lines) + "\n").encode("utf-8")

    if duplicate_dictionary_series:
        series_key, destination_member = duplicate_dictionary_series
        board_series_id = next(
            series.board_series_id
            for series in contract.series
            if series.key == series_key
        )
        source_line = next(
            line
            for member_name, payload in payloads.items()
            if member_name.startswith("data_dictionary/")
            for line in payload.decode("utf-8").splitlines()
            if line.startswith(f"{board_series_id}\t")
        )
        payloads[destination_member] += f"{source_line}\n".encode("utf-8")

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for member_name, payload in payloads.items():
            archive.writestr(member_name, payload)
    return stream.getvalue(), payloads


def _pin_fixture_hashes(
    monkeypatch: pytest.MonkeyPatch,
    archive_bytes: bytes,
    member_payloads: dict[str, bytes],
) -> None:
    monkeypatch.setattr(
        download,
        "OPEN02_BOARD_ARCHIVE_SHA256",
        hashlib.sha256(archive_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        download,
        "OPEN02_BOARD_CSV_MEMBER_SHA256",
        tuple(
            (name, hashlib.sha256(member_payloads[name]).hexdigest())
            for name in (
                "csv/fu111.csv",
                "csv/fu112.csv",
                "csv/fu113.csv",
            )
        ),
    )
    monkeypatch.setattr(
        download,
        "OPEN02_BOARD_DICTIONARY_MEMBER_SHA256",
        tuple(
            (name, hashlib.sha256(member_payloads[name]).hexdigest())
            for name in (
                "data_dictionary/fu111.txt",
                "data_dictionary/fu112.txt",
                "data_dictionary/fu113.txt",
            )
        ),
    )


def test_open02_board_archive_pins_completed_release_and_members() -> None:
    assert download.OPEN02_BOARD_ARCHIVE_URL == (
        "https://www.federalreserve.gov/releases/z1/20260319/"
        "z1_csv_files.zip"
    )
    assert download.OPEN02_BOARD_ARCHIVE_SHA256 == (
        "4a758a65a5190987a53e24039d91cc2b09ed55e57a2560bc640fdfe191ceee35"
    )
    assert dict(download.OPEN02_BOARD_CSV_MEMBER_SHA256) == {
        "csv/fu111.csv": (
            "2dc83502d138e5253117a784c28b8fbe"
            "eba0e1460db2439ad243c116c1de9a11"
        ),
        "csv/fu112.csv": (
            "b016e3d742f4dffd61b12947a2e64605"
            "c3338c71507297d925f61f4980f7bae7"
        ),
        "csv/fu113.csv": (
            "e25f4a6843c428bd120fc416974578765"
            "f14253a76d0fb07c79369c46c7df952"
        ),
    }
    assert dict(download.OPEN02_BOARD_DICTIONARY_MEMBER_SHA256) == {
        "data_dictionary/fu111.txt": (
            "df992f6d6868a8665023f558018a2baf"
            "69b24c32933c34221d31acae8c82f1f7"
        ),
        "data_dictionary/fu112.txt": (
            "f91d4217bf99636658068456ccbb7d9c7"
            "c9fabcf268d886238fdd182ad4bd818"
        ),
        "data_dictionary/fu113.txt": (
            "2e51ba3a7b9f24a236ba4ba8dee57da"
            "04c8d61deb86f657d7e02f2a8f3873155"
        ),
    }


def test_fetch_open02_board_archive_returns_sanitized_verified_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_bytes, member_payloads = _fixture_archive()
    _pin_fixture_hashes(monkeypatch, archive_bytes, member_payloads)
    requests: list[tuple[str, int]] = []

    def fake_urlopen(url: str, timeout: int = 60) -> bytes:
        requests.append((url, timeout))
        return archive_bytes

    monkeypatch.setattr(download, "_urlopen_bytes", fake_urlopen)
    result = fetch_open02_board_archive(timeout=17)

    assert requests == [(download.OPEN02_BOARD_ARCHIVE_URL, 17)]
    assert len(result["rows"]) == 96
    assert result["rows"][0]["quarter"] == "2002Q1"
    assert result["rows"][-1]["quarter"] == "2025Q4"
    assert len(result["rows"][0]) == 21
    assert result["metadata"]["observations"] == 96
    assert result["metadata"]["series_count"] == 20
    assert result["metadata"]["observation_vintage_cutoff"] == (
        "2026-05-22T16:56:46Z"
    )
    exception = next(
        series
        for series in result["metadata"]["series"]
        if series["key"] == "agency_us_com_pass"
    )
    assert exception["fred_id"] == "BOGZ1FU763061303Q"
    assert exception["board_series_id"] == "FU763061503.Q"
    assert exception["official_description"].endswith("; Asset")
    assert exception["side"] == "asset"
    assert exception["unit_label"] == download.OPEN02_BOARD_UNIT_LABEL
    assert "api_key" not in repr(result).casefold()


def test_parse_open02_board_archive_rejects_archive_hash_mismatch() -> None:
    with pytest.raises(ValueError, match="archive SHA-256 mismatch"):
        parse_open02_board_archive(b"not-the-pinned-archive")


def test_parse_open02_board_archive_rejects_member_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_bytes, member_payloads = _fixture_archive()
    _pin_fixture_hashes(monkeypatch, archive_bytes, member_payloads)
    hashes = list(download.OPEN02_BOARD_CSV_MEMBER_SHA256)
    hashes[0] = (hashes[0][0], "0" * 64)
    monkeypatch.setattr(
        download,
        "OPEN02_BOARD_CSV_MEMBER_SHA256",
        tuple(hashes),
    )
    with pytest.raises(ValueError, match="member csv/fu111.csv SHA-256 mismatch"):
        parse_open02_board_archive(archive_bytes)


def test_parse_open02_board_archive_rejects_incomplete_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_bytes, member_payloads = _fixture_archive(
        missing_quarter=("csv/fu112.csv", "2017Q3")
    )
    _pin_fixture_hashes(monkeypatch, archive_bytes, member_payloads)
    with pytest.raises(ValueError, match="missing 1 required quarters"):
        parse_open02_board_archive(archive_bytes)


@pytest.mark.parametrize(
    ("fixture_kwargs", "message"),
    [
        (
            {"bad_value": ("loans_aff", "2010Q2", "csv/fu113.csv")},
            "has malformed value",
        ),
        (
            {"bad_description": ("loans_aff", "Wrong loans; asset")},
            "official description",
        ),
        (
            {"bad_side": ("loans_aff", "liability")},
            "does not match",
        ),
        (
            {"bad_unit": ("loans_aff", "Billions of dollars")},
            "unexpected unit label",
        ),
        (
            {
                "duplicate_dictionary_series": (
                    "loans_us",
                    "data_dictionary/fu112.txt",
                )
            },
            "exactly one verified data dictionary; found 2",
        ),
    ],
)
def test_parse_open02_board_archive_rejects_malformed_series_metadata_or_value(
    monkeypatch: pytest.MonkeyPatch,
    fixture_kwargs: dict[str, tuple[str, ...]],
    message: str,
) -> None:
    archive_bytes, member_payloads = _fixture_archive(**fixture_kwargs)
    _pin_fixture_hashes(monkeypatch, archive_bytes, member_payloads)
    with pytest.raises(ValueError, match=message):
        parse_open02_board_archive(archive_bytes)


def test_fetch_open02_board_archive_sanitizes_request_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_urlopen(url: str, timeout: int = 60) -> bytes:
        raise OSError(f"request failed with credential: {url}?api_key=secret")

    monkeypatch.setattr(download, "_urlopen_bytes", failing_urlopen)
    with pytest.raises(RuntimeError) as exc_info:
        fetch_open02_board_archive()

    assert "secret" not in str(exc_info.value)
    assert "api_key" not in str(exc_info.value)
    assert str(exc_info.value) == (
        "Federal Reserve OPEN-02 Board archive request failed"
    )
