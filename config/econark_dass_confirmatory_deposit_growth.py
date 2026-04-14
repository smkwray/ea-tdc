from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

# EA-TDC-owned confirmatory config focused on transformed deposit targets.
# This keeps the transformed deposit-growth outcomes but now aligns the
# treatment with the repo's canonical public quarterly TDC object.

CONFIG_DIR = Path(__file__).resolve().parent
EA_TDC_ROOT = CONFIG_DIR.parent
ECONARK_ROOT = Path(
    os.environ.get("ECONARK_ROOT", str(EA_TDC_ROOT.parent / "econark"))
).expanduser()

# Public quarterly TDC in this repo is `tdc_bank_only_qoq`.
# This config uses that canonical public treatment and should remain the first
# confirmatory rerun target for the deposit branch.
CONFIRMATORY_SCOPE = "tdc_confirmatory"
PUBLIC_TREATMENT_LABEL = "tdc_bank_only_qoq"

START_DATE = "1985-03-31"
END_DATE = "2025-12-31"

SERIES_SOURCE = "fetch_dict"
FREDFETCH_PY = str(ECONARK_ROOT / "interpol" / "fredfetch.py")
FETCH_DICT_TXT = str(ECONARK_ROOT / "interpol" / "fetch" / "fetch_dict.txt")
MERGE_FETCH_DICT_METADATA = False
RAW_DIR = str(ECONARK_ROOT / "interpol" / "raw")
FETCH_DATA_CSV = None
FETCH_DATA_FALLBACK_SERIES = []

_PARITY_Q_PANEL = str(
    ECONARK_ROOT
    / "fetchr-parity"
    / "out"
    / "interpol_core_contract_scaffold"
    / "final_lvl.csv"
)
_PUBLIC_TDC_INPUT = EA_TDC_ROOT / "data" / "bundles" / "tdcpass" / "standardized_series.csv"
_PUBLIC_TDC_Q_PANEL = EA_TDC_ROOT / "output" / "econark_dass" / "inputs" / "tdc_public_confirmatory_q.csv"


def _ensure_public_tdc_q_panel() -> str:
    if not _PUBLIC_TDC_INPUT.exists():
        raise FileNotFoundError(f"Missing public TDC bundle: {_PUBLIC_TDC_INPUT}")

    rows: list[dict[str, str]] = []
    with _PUBLIC_TDC_INPUT.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("series_label", "")).strip() != "tdc_bank_only_qoq":
                continue
            period_end = str(row.get("period_end", "")).strip()
            value = str(row.get("value", "")).strip()
            if not period_end or not value:
                continue
            rows.append({"date": period_end, "tdc_est": value})

    rows.sort(key=lambda item: item["date"])
    _PUBLIC_TDC_Q_PANEL.parent.mkdir(parents=True, exist_ok=True)
    with _PUBLIC_TDC_Q_PANEL.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "tdc_est"])
        writer.writeheader()
        writer.writerows(rows)
    return str(_PUBLIC_TDC_Q_PANEL)


_PUBLIC_TDC_Q_PANEL_PATH = _ensure_public_tdc_q_panel()

EXTERNAL_Q_SERIES = {
    "tdc_est": {"path": _PUBLIC_TDC_Q_PANEL_PATH, "column": "tdc_est", "freq": "q"},
    "DEPOSITS": {"path": _PARITY_Q_PANEL, "column": "DEPOSITS", "freq": "q"},
    "RES": {"path": _PARITY_Q_PANEL, "column": "RES", "freq": "q"},
    "GDP": {"path": _PARITY_Q_PANEL, "column": "GDP", "freq": "q"},
    "PCE": {"path": _PARITY_Q_PANEL, "column": "PCE", "freq": "q"},
    "M2": {"path": _PARITY_Q_PANEL, "column": "M2", "freq": "q"},
    "MMF": {"path": _PARITY_Q_PANEL, "column": "MMF", "freq": "q"},
    "TOTBKCR": {"path": _PARITY_Q_PANEL, "column": "TOTBKCR", "freq": "q"},
    "DGS2": {"path": _PARITY_Q_PANEL, "column": "DGS2", "freq": "q"},
    "DGS10": {"path": _PARITY_Q_PANEL, "column": "DGS10", "freq": "q"},
    "Fed_Funds": {"path": _PARITY_Q_PANEL, "column": "Fed_Funds", "freq": "q"},
    "INDPRO": {"path": _PARITY_Q_PANEL, "column": "INDPRO", "freq": "q"},
}

INCLUDE_CONFIG_GENERATED = False
CONFIG_INTERPOL_PY = str(ECONARK_ROOT / "interpol" / "config_interpol.py")
INCLUDE_GENERATED = True
GENERATED_FREQ_POLICY = "coarsest"
APPLY_SAAR_ADJUSTMENTS = False
INFER_RAW_FREQ = True

SERIES_TO_GENERATE = {
    "DEPOSITS_DLOG_Q": {
        "func": lambda df: np.log(df["DEPOSITS"]).diff() * 100.0,
        "components": ["DEPOSITS"],
        "freq": "q",
    },
    "DEPOSITS_D_Q": {
        "func": lambda df: df["DEPOSITS"].diff(),
        "components": ["DEPOSITS"],
        "freq": "q",
    },
}

CUTOFF_POLICY = "quarter_start"
EVENTS_CONFIG_PY = str(ECONARK_ROOT / "dass" / "events.py")
REQUIRE_RAW = False

DAILY_LAGS = 0
WEEKLY_LAGS = 0
MONTHLY_LAGS = 0
QUARTERLY_LAGS = 4
MAX_MISSING_PCT = 80.0
STANDARDIZE = False

PREP_INCLUDE_QUARTER_END = [
    "tdc_est",
    "DEPOSITS",
    "DEPOSITS_DLOG_Q",
    "DEPOSITS_D_Q",
    "RES",
    "GDP",
    "PCE",
    "M2",
    "MMF",
    "TOTBKCR",
    "DGS2",
    "DGS10",
    "Fed_Funds",
    "INDPRO",
]

RUNNER_THREADS = 2
MATH_THREADS = 1
DESIGN_CONCURRENCY = 2
ESTIMATOR_CONCURRENCY = 2
SKIP_EXISTING = False

RUN_V1_GRID = False
RUN_V1_CF = False
RUN_V1_DML = True
RUN_V1_TMLE = True
RUN_V1_LP = True
RUN_IDKIT = False
RUN_ROBUSTNESS_PACK = False
RUN_PLACEBO_DML = False
RUN_BENCHMARKS = False
RUN_D2_MONEY_AGG = False
RUN_BILLS_CONTROL_VARIANTS = False
RUN_HEADLINE_BUNDLE = False

V1_W_SPEC_GRID = [64]
_FORCE_W_SERIES = [
    "tdc_est",
    "DEPOSITS",
    "RES",
    "GDP",
    "PCE",
    "M2",
    "MMF",
    "TOTBKCR",
    "DGS2",
    "DGS10",
    "Fed_Funds",
    "INDPRO",
]

_BASE_JOB = {
    "treatment": "tdc_est",
    "horizons": [0, 1, 2, 4],
    "treatment_mode": "shock",
    "binary": False,
    "folds": 5,
    "make_stationary": False,
    "standardize": False,
    "shock_oos": "fold",
    "shock_w_max": 64,
    "w_max": 64,
    "w_select": "corr_t_then_variance",
    "force_w_series": _FORCE_W_SERIES,
    "n_jobs": RUNNER_THREADS,
}

_OUTCOME_JOBS = [
    {**_BASE_JOB, "outcome": "DEPOSITS_DLOG_Q", "w_tag": "deposit_dlog_q"},
    {**_BASE_JOB, "outcome": "DEPOSITS_D_Q", "w_tag": "deposit_diff_q"},
]

V1_TMLE_DEFAULTS = {
    "tmle_mode": "continuous",
    "density_floor": 0.05,
    "h_clip": 8.0,
    "epsilon_cap": 0.5,
    "nuisance_r2_floor": 0.02,
    "overlap_floor": 0.20,
    "epsilon_theta_ratio_cap": 2.0,
}

_TMLE_VARIANTS = [
    {
        "w_tag": "tmle_ct_base_growth",
        "density_floor": 0.05,
        "h_clip": 8.0,
        "epsilon_cap": 0.5,
        "nuisance_r2_floor": 0.02,
        "overlap_floor": 0.20,
        "epsilon_theta_ratio_cap": 2.0,
    },
    {
        "w_tag": "tmle_ct_loose_overlap_growth",
        "density_floor": 0.02,
        "h_clip": 10.0,
        "epsilon_cap": 0.75,
        "nuisance_r2_floor": 0.01,
        "overlap_floor": 0.10,
        "epsilon_theta_ratio_cap": 3.0,
    },
    {
        "w_tag": "tmle_ct_tight_overlap_growth",
        "density_floor": 0.10,
        "h_clip": 6.0,
        "epsilon_cap": 0.35,
        "nuisance_r2_floor": 0.03,
        "overlap_floor": 0.30,
        "epsilon_theta_ratio_cap": 1.5,
    },
]

V1_TMLE_JOBS = [
    {**job, **V1_TMLE_DEFAULTS, **variant}
    for job in _OUTCOME_JOBS
    for variant in _TMLE_VARIANTS
]
V1_DML_JOBS = [dict(job, hac_lags=4) for job in _OUTCOME_JOBS]
V1_LP_JOBS = [
    dict(
        job,
        hac_lags=4,
        min_obs_per_regressor=1.5,
        max_condition_number=1e10,
        min_treatment_sd=1e-8,
    )
    for job in _OUTCOME_JOBS
]

V1_LP_JOBS_SOURCE = "V1_TMLE_JOBS"
V1_LP_REQUIRE_W_COLS = False

OUT_DIR = str(EA_TDC_ROOT / "output" / "econark_dass" / "deposit_confirmatory_growth_tdc")
OUT_CSV = "stacked_quarterly.csv"
OUT_META_MD = "stacked_quarterly_meta.md"
DESIGN_OUT_DIR = f"{OUT_DIR}/design"
CF_OUT_DIR = f"{OUT_DIR}/cf"
TMLE_OUT_DIR = f"{OUT_DIR}/tmle"
LP_OUT_DIR = f"{OUT_DIR}/lp"
DML_OUT_DIR = f"{OUT_DIR}/dml"
RESULTS_CSV = f"{OUT_DIR}/results.csv"
OVERLAP_MD = f"{OUT_DIR}/overlap.md"
IDKIT_OUT_DIR = f"{OUT_DIR}/id"
