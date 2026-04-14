from __future__ import annotations

import csv
import json
from pathlib import Path

from ea_tdc.adapters.qrawatch import adapt_qrawatch
from ea_tdc.paths import ensure_repo_dirs, project_paths


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_qrawatch_adapter_writes_series_and_event_bundles(tmp_path: Path) -> None:
    publish_dir = tmp_path / "data" / "seed" / "qrawatch"
    _write_text(
        publish_dir / "ati_quarter_table.csv",
        "\n".join(
            [
                "quarter,financing_need_bn,net_bills_bn,bill_share,missing_coupons_15_bn,missing_coupons_18_bn,missing_coupons_20_bn,ati_baseline_bn,source_quality,public_role",
                "2024Q1,100,10,0.1,1,2,3,5,exact_official_numeric,headline",
            ]
        ),
    )
    _write_text(
        publish_dir / "duration_supply_summary.csv",
        "\n".join(
            [
                "date,coupon_like_total,headline_public_duration_supply,provisional_public_duration_supply,headline_source_quality,fallback_source_quality,qt_proxy,qt_proxy_is_zero_filled,buybacks_accepted",
                "2024-01-10,50,20,22,hybrid,hybrid,3,False,1",
            ]
        ),
    )
    _write_text(
        publish_dir / "qra_event_registry_v2.csv",
        "\n".join(
            [
                "event_id,quarter,release_timestamp_et,release_timestamp_kind,quality_tier,eligibility_blockers,timestamp_precision,separability_status,expectation_status,contamination_status,release_component_count,causal_eligible_component_count",
                "qra_2024_02,2024Q1,2024-02-07T08:30:00-05:00,official_release_time,Tier A,,exact_time,separable,reviewed_benchmark,reviewed_clean,2,1",
            ]
        ),
    )
    _write_text(
        publish_dir / "qra_event_shock_summary.csv",
        "\n".join(
            [
                "quarter,event_id,event_label,event_date_requested,event_date_aligned,event_date_type,shock_bn,schedule_diff_10y_eq_bn,schedule_diff_dynamic_10y_eq_bn,schedule_diff_dv01_usd,gross_notional_delta_bn,shock_review_status,treatment_variant,usable_for_headline_reason,descriptive_headline_reason,claim_scope,headline_bucket,usable_for_headline,usable_for_descriptive_headline,shock_missing_flag,small_denominator_flag,timing_quality,overlap_severity",
                "2024Q1,qra_2024_02,2024 Feb QRA,2024-02-07,2024-02-07,official_release_date,12,,,,,reviewed,canonical_shock_bn,ok,,headline,include,True,True,False,False,exact_timestamp,none",
            ]
        ),
    )

    paths = project_paths(tmp_path)
    ensure_repo_dirs(paths)
    result = adapt_qrawatch(paths)

    assert result.series_rows_written == 12
    assert result.event_rows_written == 1
    assert result.standardized_path.exists()
    assert result.event_bundle_path.exists()
    assert result.manifest_path.exists()

    with result.standardized_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["series_id"] == "ati_baseline_bn" and row["role"] == "treatment" for row in rows)
    assert any(row["series_id"] == "headline_public_duration_supply" and row["role"] == "treatment" for row in rows)

    with result.event_bundle_path.open("r", encoding="utf-8", newline="") as handle:
        event_rows = list(csv.DictReader(handle))
    assert event_rows[0]["event_type"] == "qra_release"
    assert event_rows[0]["treatment_id"] == "canonical_shock_bn"
    assert event_rows[0]["usable_for_headline"] == "true"
    assert event_rows[0]["quality_tier"] == "Tier A"
    assert event_rows[0]["timestamp_precision"] == "exact_time"

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_repo"] == "qrawatch"
    assert manifest["event_rows_written"] == 1
    assert manifest["event_rows_with_treatment"] == 1
    assert manifest["event_rows_usable_for_headline"] == 1
