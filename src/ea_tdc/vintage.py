"""Declared observation vintages for rerunning admitted quarterly specifications.

A vintage changes sample identity and input pins only. Estimators, controls,
factor features, gates and thresholds remain those of the admitted contracts.
The admitted run is the default: a vintage is active only when ``EA_TDC_VINTAGE``
names a committed vintage configuration relative to the project root.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping

VINTAGE_ENV = "EA_TDC_VINTAGE"
SCHEMA_VERSION = "ea_tdc_vintage_refresh_v1"
FACTOR_FEATURE_COUNT = 100
ROLLING_MINIMUM_OBSERVATIONS = 40
_QUARTER = re.compile(r"^(?P<year>\d{4})Q(?P<quarter>[1-4])$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def quarter_ordinal(quarter: str) -> int:
    match = _QUARTER.fullmatch(str(quarter))
    if match is None:
        raise ValueError(f"Invalid quarter: {quarter!r}")
    return int(match["year"]) * 4 + int(match["quarter"]) - 1


def quarter_label(ordinal: int) -> str:
    return f"{ordinal // 4}Q{ordinal % 4 + 1}"


def quarter_range(start: str, end: str) -> tuple[str, ...]:
    first, last = quarter_ordinal(start), quarter_ordinal(end)
    if last < first:
        raise ValueError(f"Quarter range ends before it starts: {start}..{end}")
    return tuple(quarter_label(value) for value in range(first, last + 1))


def quarter_end(quarter: str) -> date:
    ordinal = quarter_ordinal(quarter) + 1
    return date(ordinal // 4, (ordinal % 4) * 3 + 1, 1) - timedelta(days=1)


def quarter_hash(quarters: tuple[str, ...]) -> str:
    """Same digest as the OPEN-01 sample hash and the OPEN-02 quarter hash."""
    return hashlib.sha256("\n".join(quarters).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Vintage:
    locator: str
    sha256: str
    vintage_id: str
    sample_start: str
    sample_end: str
    observations: int
    observation_cutoff: date
    method_tier_counts: Mapping[str, int]
    payload: Mapping[str, Any] = field(repr=False)

    @property
    def quarters(self) -> tuple[str, ...]:
        return quarter_range(self.sample_start, self.sample_end)

    @property
    def quarter_hash(self) -> str:
        return quarter_hash(self.quarters)

    @property
    def rolling_endpoints(self) -> tuple[str, ...]:
        return self.quarters[ROLLING_MINIMUM_OBSERVATIONS - 1:]

    def record(self) -> dict[str, Any]:
        return {
            "locator": self.locator,
            "sha256": self.sha256,
            "vintage": self.vintage_id,
            "sample": {"start": self.sample_start, "end": self.sample_end, "n": self.observations},
            "quarter_hash": self.quarter_hash,
            "observation_cutoff": self.observation_cutoff.isoformat(),
            "method_tier_counts": dict(self.method_tier_counts),
        }


def _frozen_tier_counts() -> dict[str, int]:
    from ea_tdc.open_contract import OPEN_CONTRACT

    return {tier.tier_id: tier.expected_count for tier in OPEN_CONTRACT.method_tiers}


def load_vintage(root: Path, locator: str) -> Vintage:
    candidate = Path(locator)
    path = (candidate if candidate.is_absolute() else Path(root) / candidate).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Vintage configuration does not exist: {locator}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{locator} is not a {SCHEMA_VERSION} configuration")
    sample = payload.get("sample")
    admitted = payload.get("admitted_sample")
    if not isinstance(sample, dict) or not isinstance(admitted, dict):
        raise ValueError("Vintage must declare sample and admitted_sample")
    quarters = quarter_range(sample["start"], sample["end"])
    if len(quarters) != sample.get("n"):
        raise ValueError("Vintage sample count disagrees with its quarter bounds")
    admitted_quarters = quarter_range(admitted["start"], admitted["end"])
    if len(admitted_quarters) != admitted.get("n") or quarters[: len(admitted_quarters)] != admitted_quarters:
        raise ValueError("Vintage sample must extend the admitted sample without dropping quarters")
    cutoff = date.fromisoformat(str(payload.get("observation_cutoff", "")))
    if cutoff != quarter_end(quarters[-1]):
        raise ValueError("Observation cutoff must be the last day of the vintage end quarter")
    counts = payload.get("method_tier_counts")
    frozen = _frozen_tier_counts()
    if not isinstance(counts, dict) or set(counts) != set(frozen):
        raise ValueError("Vintage method tiers must be exactly the admitted construction tiers")
    if any(type(value) is not int or value < frozen[key] for key, value in counts.items()):
        raise ValueError("Vintage tier counts cannot drop admitted observations")
    if sum(counts.values()) != len(quarters) or sum(frozen.values()) != len(admitted_quarters):
        raise ValueError("Vintage tier counts must cover every sample quarter exactly once")
    return Vintage(
        locator=Path(locator).as_posix(),
        sha256=sha256_file(path),
        vintage_id=str(payload.get("vintage", "")),
        sample_start=quarters[0],
        sample_end=quarters[-1],
        observations=len(quarters),
        observation_cutoff=cutoff,
        method_tier_counts=MappingProxyType(dict(counts)),
        payload=MappingProxyType(payload),
    )


def active_vintage(root: Path) -> Vintage | None:
    locator = os.environ.get(VINTAGE_ENV, "").strip()
    return load_vintage(Path(root), locator) if locator else None


def pinned_factor_features(vintage: Vintage, root: Path) -> list[str]:
    """Return the frozen ordered K=100 feature list; no screening is rerun."""
    record = vintage.payload["factor_features"]["ordered_top100"]
    path = Path(root) / record["path"]
    if not _SHA256.fullmatch(str(record.get("sha256", ""))) or sha256_file(path) != record["sha256"]:
        raise ValueError("Pinned factor-feature list differs from its vintage pin")
    features = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(features, list)
        or len(features) != FACTOR_FEATURE_COUNT
        or len(set(features)) != FACTOR_FEATURE_COUNT
        or not all(isinstance(item, str) and item for item in features)
    ):
        raise ValueError("Pinned factor features must be 100 unique ordered identifiers")
    return list(features)


@contextlib.contextmanager
def _rebound(module: Any, name: str, expected: Any, value: Any) -> Iterator[None]:
    current = getattr(module, name)
    if current != expected:
        raise RuntimeError(f"{module.__name__}.{name} is not the admitted value; refusing to rebind")
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, current)


@contextlib.contextmanager
def open01_tier_counts(vintage: Vintage) -> Iterator[dict[str, Any]]:
    """Extend the construction-tier count check to the declared vintage sample."""
    from ea_tdc import open01

    frozen = _frozen_tier_counts()
    counts = MappingProxyType(dict(vintage.method_tier_counts))
    with _rebound(open01, "EXPECTED_METHOD_TIER_COUNTS", frozen, counts):
        yield {"admitted": frozen, "vintage": dict(counts)}


@contextlib.contextmanager
def open16_sample(vintage: Vintage) -> Iterator[dict[str, Any]]:
    """Bind the OPEN-16 diagnostic sample to the vintage; estimator code is unchanged."""
    from ea_tdc import open16

    admitted = quarter_range(
        vintage.payload["admitted_sample"]["start"], vintage.payload["admitted_sample"]["end"]
    )
    with _rebound(open16, "FROZEN_QUARTERS", admitted, vintage.quarters), _rebound(
        open16,
        "ORIGINAL_ROLLING_ENDPOINTS",
        admitted[ROLLING_MINIMUM_OBSERVATIONS - 1:],
        vintage.rolling_endpoints,
    ):
        yield {
            "admitted": {"start": admitted[0], "end": admitted[-1], "n": len(admitted)},
            "vintage": {"start": vintage.sample_start, "end": vintage.sample_end, "n": vintage.observations},
            "rolling_endpoints": [vintage.rolling_endpoints[0], vintage.rolling_endpoints[-1], len(vintage.rolling_endpoints)],
        }
