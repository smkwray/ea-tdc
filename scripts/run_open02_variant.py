#!/usr/bin/env python3
"""Run the frozen OPEN-02 producer on a declared corrected-input variant.

The admitted producer, its pins and the OPEN-02 contract stay byte-identical. A
committed variant configuration names a corrected TDCest bundle and its
generation time, the observation cutoff, the OPEN-01 run that consumed that
bundle and the hashes of its outputs. The variant's Financial Accounts source is
derived from the admitted bundle without refetching: only the TDCest pairing
stamps change. For one call, the producer's pins and contract are rebound to the
variant's; the estimator code is unchanged.

  --derive-source   write the variant source bundle and manifest, then stop
  (default)         check the variant inputs, run the producer, write a variant receipt
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_open02_producer as producer  # noqa: E402

from ea_tdc.open_contract import OPEN02_CONTRACT  # noqa: E402

SCHEMA = "ea_tdc_open02_inputs_v1"
RECEIPT_LOCATOR = "output/manifests/open02_variant_receipt.json"
# The only source-bundle fields a variant rewrites; the Financial Accounts content is the admitted bytes.
STAMP_FIELDS = ("accepted_tdcest_bundle_generated_at", "observation_vintage_cutoff")


class Variant(NamedTuple):
    name: str
    config: dict[str, Any]
    open01_producer_commit: str
    accepted_input_sha256: dict[str, str]
    source_bundle_locator: str
    source_manifest_locator: str
    tdcest_bundle_sha256: str
    derived_from: dict[str, Any]
    contract: Any


def load_variant(root: Path, locator: str) -> Variant:
    """A committed corrected-input variant: new TDCest pairing, same Financial Accounts.

    Its source locators may not be the admitted ones, so the admitted pair is
    never overwritten, and its cutoff may not precede its TDCest bundle or the
    Board release.
    """
    record = producer._file_record(root, locator)
    payload = producer._load_json_object(producer._project_path(root, locator))
    source = payload.get("source") or {}
    accepted = payload.get("accepted_inputs") or {}
    tdcest = payload.get("tdcest_bundle") or {}
    derived = dict((source.get("derived_from") or {}).get("bundle") or {})
    locators = (source.get("bundle"), source.get("manifest"))
    if (
        payload.get("schema_version") != SCHEMA
        or not str(payload.get("variant", "")).strip()
        or set(accepted) != set(producer.EXPECTED_ACCEPTED_INPUT_SHA256)
        or derived.get("path") != producer.SOURCE_BUNDLE_LOCATOR
        or not all(isinstance(item, str) and item for item in locators)
        or set(locators) & {producer.SOURCE_BUNDLE_LOCATOR, producer.SOURCE_MANIFEST_LOCATOR}
        or type(derived.get("bytes")) is not int
    ):
        raise ValueError("OPEN-02 input variant is malformed or would overwrite the admitted source")
    generated = producer._validate_utc_timestamp(tdcest.get("generated_at"), label="Variant TDCest generated_at")
    cutoff = producer._validate_utc_timestamp(payload.get("observation_vintage_cutoff"), label="Variant cutoff")
    release = producer._validate_utc_timestamp(f"{producer.OPEN02_BOARD_ARCHIVE_RELEASE_DATE}T00:00:00Z",
                                               label="Board release date")
    if cutoff < generated or cutoff < release:
        raise ValueError("Variant observation cutoff precedes its TDCest bundle or the Board release")
    commit = str(payload.get("open01_producer_commit", "")).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Variant OPEN-01 producer must be a full 40-character commit")
    derived["sha256"] = producer._validate_sha256(derived.get("sha256"), label="Admitted source bundle hash")
    contract = replace(OPEN02_CONTRACT, sample=replace(
        OPEN02_CONTRACT.sample,
        accepted_tdcest_bundle_generated_at=str(tdcest["generated_at"]),
        observation_vintage_cutoff=str(payload["observation_vintage_cutoff"]),
    ))
    return Variant(
        name=str(payload["variant"]),
        config=record,
        open01_producer_commit=commit,
        accepted_input_sha256={key: producer._validate_sha256(value, label=f"Variant input hash {key}")
                               for key, value in accepted.items()},
        source_bundle_locator=str(source["bundle"]),
        source_manifest_locator=str(source["manifest"]),
        tdcest_bundle_sha256=producer._validate_sha256(tdcest.get("sha256"), label="Variant TDCest bundle hash"),
        derived_from=derived,
        contract=contract,
    )


def derived_source_bundle(root: Path, variant: Variant) -> dict[str, Any]:
    """The hash-pinned admitted source bundle with only the TDCest pairing stamps replaced."""
    record = variant.derived_from
    admitted = producer._file_record(root, record["path"], expected_sha256=record["sha256"])
    if admitted["bytes"] != record["bytes"]:
        raise ValueError("Admitted OPEN-02 source bundle size changed")
    bundle = producer._load_json_object(producer._project_path(root, record["path"]))
    found = (bundle.get("accepted_tdcest_bundle_generated_at"), bundle.get("observation_vintage_cutoff"),
             (bundle.get("official_release") or {}).get("observation_vintage_cutoff"))
    sample = OPEN02_CONTRACT.sample
    if found != (sample.accepted_tdcest_bundle_generated_at, sample.observation_vintage_cutoff,
                 sample.observation_vintage_cutoff):
        raise ValueError("Admitted OPEN-02 source bundle does not carry the admitted stamps")
    stamps = variant.contract.sample
    bundle["accepted_tdcest_bundle_generated_at"] = stamps.accepted_tdcest_bundle_generated_at
    bundle["observation_vintage_cutoff"] = stamps.observation_vintage_cutoff
    bundle["official_release"]["observation_vintage_cutoff"] = stamps.observation_vintage_cutoff
    return bundle


def derive_source(*, producer_commit: str, run_id: str, variant: Variant, root: Path = ROOT,
                  invocation_argv: list[str] | None = None) -> Path:
    """Write the variant source pair from the admitted bundle; nothing is fetched or overwritten."""
    root = root.resolve()
    commit = producer._verify_producer_commit(root, producer_commit)
    run_id = producer._validate_run_id(run_id)
    started_at = producer._utc_now()
    bundle_path = producer._project_path(root, variant.source_bundle_locator)
    manifest_path = producer._project_path(root, variant.source_manifest_locator)
    if bundle_path.exists() or manifest_path.exists():
        raise ValueError("Variant source pair already exists; no overwrite")
    bundle_bytes = producer._json_bytes(derived_source_bundle(root, variant))
    release = json.loads(bundle_bytes)["official_release"]
    manifest = {
        "schema_version": 1,
        "open_id": "OPEN-02",
        "status": "passed",
        "stage": "source_acquisition",
        "producer_commit": commit,
        "run_id": run_id,
        "argv": list(invocation_argv) if invocation_argv else [
            str(sys.executable), "scripts/run_open02_variant.py", "--producer-commit", commit, "--run-id", run_id,
            "--derive-source"],
        "started_at_utc": started_at,
        "completed_at_utc": producer._utc_now(),
        "observation_vintage": producer.OPEN02_BOARD_ARCHIVE_RELEASE_DATE,
        "observation_vintage_cutoff": variant.contract.sample.observation_vintage_cutoff,
        "official_release": {
            "source_url": release["source_url"],
            "release_date": release["release_date"],
            "archive_sha256": release["archive_sha256"],
            "csv_member_sha256": release["csv_member_sha256"],
            "dictionary_member_sha256": release["dictionary_member_sha256"],
            "normalized_rows_sha256": release["rows_sha256"],
        },
        "source_bundle": {"path": bundle_path.relative_to(root).as_posix(),
                          "sha256": hashlib.sha256(bundle_bytes).hexdigest(), "bytes": len(bundle_bytes)},
        "metadata_gate": {"passed": True, "series_count": release["series_count"],
                          "series_ids": [series.board_series_id for series in OPEN02_CONTRACT.series],
                          "verified_from_bundled_data_dictionaries": True},
        "coverage_gate": {"passed": True, "sample_start": release["sample_start"], "sample_end": release["sample_end"],
                          "observations": release["observations"], "series_count": release["series_count"]},
        "derived_from": {"bundle": dict(variant.derived_from)},
    }
    if producer._git_text(root, "rev-parse", "HEAD").lower() != commit:
        raise ValueError("Checkout HEAD changed during OPEN-02 source derivation")
    try:
        producer._atomic_write_bytes(bundle_path, bundle_bytes)
        producer._atomic_write_bytes(manifest_path, producer._json_bytes(manifest))
    except BaseException:
        for path in (bundle_path, manifest_path):
            path.unlink(missing_ok=True)
        raise
    return manifest_path


@contextlib.contextmanager
def _rebound(variant: Variant) -> Iterator[None]:
    """Bind the producer's pins and contract to the variant for one call, then restore them."""
    if producer.OPEN02_CONTRACT is not OPEN02_CONTRACT:
        raise RuntimeError("OPEN-02 producer is already rebound; refusing to nest variants")
    values = {"EXPECTED_OPEN01_PRODUCER_COMMIT": variant.open01_producer_commit,
              "EXPECTED_ACCEPTED_INPUT_SHA256": dict(variant.accepted_input_sha256),
              "OPEN02_CONTRACT": variant.contract}
    previous = {name: getattr(producer, name) for name in values}
    for name, value in values.items():
        setattr(producer, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(producer, name, value)


def _check_variant_inputs(root: Path, variant: Variant) -> None:
    receipt = producer._load_json_object(producer._project_path(root, producer.OPEN01_RECEIPT_LOCATOR))
    if (receipt.get("tdcest") or {}).get("bundle_hash") != variant.tdcest_bundle_sha256:
        raise ValueError("Accepted OPEN-01 run consumed another TDCest bundle")
    manifest = producer._load_json_object(producer._project_path(root, variant.source_manifest_locator))
    if manifest.get("derived_from") != {"bundle": dict(variant.derived_from)}:
        raise ValueError("Variant source manifest does not record the declared derivation")
    bundle = producer._load_json_object(producer._project_path(root, variant.source_bundle_locator))
    if bundle != derived_source_bundle(root, variant):
        raise ValueError("Variant source bundle differs from the admitted Financial Accounts content")


def run_variant(*, producer_commit: str, run_id: str, variant: Variant, root: Path = ROOT,
                invocation_argv: list[str] | None = None) -> Path:
    """Run the frozen producer on the variant inputs and bind its receipt to the variant."""
    root = root.resolve()
    _check_variant_inputs(root, variant)
    with _rebound(variant):
        producer_receipt = producer.run_open02(
            producer_commit=producer_commit, run_id=run_id, root=root,
            source_bundle_locator=variant.source_bundle_locator,
            source_manifest_locator=variant.source_manifest_locator,
            invocation_argv=invocation_argv)
    receipt = {
        "schema_version": "ea_tdc_open02_variant_receipt_v1",
        "variant": variant.name,
        "config": variant.config,
        "producer_commit": json.loads(producer_receipt.read_text(encoding="utf-8"))["producer_commit"],
        "run_id": run_id,
        "open01_producer_commit": variant.open01_producer_commit,
        "tdcest_bundle_sha256": variant.tdcest_bundle_sha256,
        "accepted_tdcest_bundle_generated_at": variant.contract.sample.accepted_tdcest_bundle_generated_at,
        "observation_vintage_cutoff": variant.contract.sample.observation_vintage_cutoff,
        "accepted_inputs": dict(variant.accepted_input_sha256),
        "derived_from": dict(variant.derived_from),
        "source_bundle": producer._file_record(root, variant.source_bundle_locator),
        "source_manifest": producer._file_record(root, variant.source_manifest_locator),
        "producer_receipt": producer._file_record(root, producer.RECEIPT_LOCATOR),
    }
    path = producer._project_path(root, RECEIPT_LOCATOR)
    producer._atomic_write_json(path, receipt)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", required=True, help="Committed corrected-input variant configuration")
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--derive-source", action="store_true")
    args = parser.parse_args(argv)
    variant = load_variant(ROOT, args.inputs)
    invocation = list(sys.argv if argv is None else ["scripts/run_open02_variant.py", *argv])
    if args.derive_source:
        result = derive_source(producer_commit=args.producer_commit, run_id=args.run_id, variant=variant,
                               invocation_argv=invocation)
    else:
        result = run_variant(producer_commit=args.producer_commit, run_id=args.run_id, variant=variant,
                             invocation_argv=invocation)
    print(json.dumps({"receipt": result.relative_to(ROOT).as_posix(), "sha256": producer._sha256_file(result)}))


if __name__ == "__main__":
    main()
