from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# EA-TDC-owned confirmatory config focused on spreads, repo, and inflation.
# This keeps the treatment on TREAST shock mode and tests outcome constructions
# that are closer to the economic channels of interest than raw deposit levels.

CONFIG_DIR = Path(__file__).resolve().parent
EA_TDC_ROOT = CONFIG_DIR.parent
ECONARK_ROOT = Path(
    os.environ.get("ECONARK_ROOT", str(EA_TDC_ROOT.parent / "econark"))
).expanduser()

# Public quarterly TDC in this repo is `tdc_bank_only_qoq`.
# This config intentionally runs raw Fed Treasury holdings (`TREAST`) and must
# not be described as TDC-confirmatory without a treatment switch.
CONFIRMATORY_SCOPE = "treast_diagnostic"
PUBLIC_TREATMENT_LABEL = "TREAST"

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

EXTERNAL_Q_SERIES = {
    "TREAST": {"path": _PARITY_Q_PANEL, "column": "TREAST", "freq": "q"},
    "GDP": {"path": _PARITY_Q_PANEL, "column": "GDP", "freq": "q"},
    "PCE": {"path": _PARITY_Q_PANEL, "column": "PCE", "freq": "q"},
    "M2": {"path": _PARITY_Q_PANEL, "column": "M2", "freq": "q"},
    "RES": {"path": _PARITY_Q_PANEL, "column": "RES", "freq": "q"},
    "DEPOSITS": {"path": _PARITY_Q_PANEL, "column": "DEPOSITS", "freq": "q"},
    "TOTBKCR": {"path": _PARITY_Q_PANEL, "column": "TOTBKCR", "freq": "q"},
    "DGS2": {"path": _PARITY_Q_PANEL, "column": "DGS2", "freq": "q"},
    "DGS10": {"path": _PARITY_Q_PANEL, "column": "DGS10", "freq": "q"},
    "Fed_Funds": {"path": _PARITY_Q_PANEL, "column": "Fed_Funds", "freq": "q"},
    "INDPRO": {"path": _PARITY_Q_PANEL, "column": "INDPRO", "freq": "q"},
    "BAA10Y": {"path": _PARITY_Q_PANEL, "column": "BAA10Y", "freq": "q"},
    "AAA10Y": {"path": _PARITY_Q_PANEL, "column": "AAA10Y", "freq": "q"},
    "TREAS10Y": {"path": _PARITY_Q_PANEL, "column": "TREAS10Y", "freq": "q"},
    "BAAFF": {"path": _PARITY_Q_PANEL, "column": "BAAFF", "freq": "q"},
    "ALL_FFREPO": {"path": _PARITY_Q_PANEL, "column": "all_ffrepo", "freq": "q"},
    "PCEPILFE": {"path": _PARITY_Q_PANEL, "column": "PCEPILFE", "freq": "q"},
    "CPIAUCSL": {"path": _PARITY_Q_PANEL, "column": "CPIAUCSL", "freq": "q"},
}

INCLUDE_CONFIG_GENERATED = False
CONFIG_INTERPOL_PY = str(ECONARK_ROOT / "interpol" / "config_interpol.py")
INCLUDE_GENERATED = True
GENERATED_FREQ_POLICY = "coarsest"
APPLY_SAAR_ADJUSTMENTS = False
INFER_RAW_FREQ = True

SERIES_TO_GENERATE = {
    "BAA_AAA_SPREAD_Q": {
        "func": lambda df: df["BAA10Y"] - df["AAA10Y"],
        "components": ["BAA10Y", "AAA10Y"],
        "freq": "q",
    },
    "AAA_TREAS_SPREAD_Q": {
        "func": lambda df: df["AAA10Y"] - df["TREAS10Y"],
        "components": ["AAA10Y", "TREAS10Y"],
        "freq": "q",
    },
    "BAA_TREAS_SPREAD_Q": {
        "func": lambda df: df["BAA10Y"] - df["TREAS10Y"],
        "components": ["BAA10Y", "TREAS10Y"],
        "freq": "q",
    },
    "BAAFF_D_Q": {
        "func": lambda df: df["BAAFF"].diff(),
        "components": ["BAAFF"],
        "freq": "q",
    },
    "PCEPILFE_DLOG_Q": {
        "func": lambda df: np.log(df["PCEPILFE"]).diff() * 100.0,
        "components": ["PCEPILFE"],
        "freq": "q",
    },
    "CPIAUCSL_DLOG_Q": {
        "func": lambda df: np.log(df["CPIAUCSL"]).diff() * 100.0,
        "components": ["CPIAUCSL"],
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
    "TREAST",
    "GDP",
    "PCE",
    "M2",
    "RES",
    "DEPOSITS",
    "TOTBKCR",
    "DGS2",
    "DGS10",
    "Fed_Funds",
    "INDPRO",
    "BAA10Y",
    "AAA10Y",
    "TREAS10Y",
    "BAAFF",
    "ALL_FFREPO",
    "PCEPILFE",
    "CPIAUCSL",
    "BAA_AAA_SPREAD_Q",
    "AAA_TREAS_SPREAD_Q",
    "BAA_TREAS_SPREAD_Q",
    "BAAFF_D_Q",
    "PCEPILFE_DLOG_Q",
    "CPIAUCSL_DLOG_Q",
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
    "TREAST",
    "GDP",
    "PCE",
    "M2",
    "RES",
    "DEPOSITS",
    "TOTBKCR",
    "DGS2",
    "DGS10",
    "Fed_Funds",
    "INDPRO",
]

_BASE_JOB = {
    "treatment": "TREAST",
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
    {**_BASE_JOB, "outcome": "BAA_AAA_SPREAD_Q", "w_tag": "baa_aaa_spread_q"},
    {**_BASE_JOB, "outcome": "AAA_TREAS_SPREAD_Q", "w_tag": "aaa_treas_spread_q"},
    {**_BASE_JOB, "outcome": "BAA_TREAS_SPREAD_Q", "w_tag": "baa_treas_spread_q"},
    {**_BASE_JOB, "outcome": "ALL_FFREPO", "w_tag": "all_ffrepo"},
    {**_BASE_JOB, "outcome": "PCEPILFE_DLOG_Q", "w_tag": "pcepilfe_dlog_q"},
    {**_BASE_JOB, "outcome": "CPIAUCSL_DLOG_Q", "w_tag": "cpiaucsl_dlog_q"},
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
        "w_tag": "tmle_ct_base_macro_prices",
        "density_floor": 0.05,
        "h_clip": 8.0,
        "epsilon_cap": 0.5,
        "nuisance_r2_floor": 0.02,
        "overlap_floor": 0.20,
        "epsilon_theta_ratio_cap": 2.0,
    },
    {
        "w_tag": "tmle_ct_loose_overlap_macro_prices",
        "density_floor": 0.02,
        "h_clip": 10.0,
        "epsilon_cap": 0.75,
        "nuisance_r2_floor": 0.01,
        "overlap_floor": 0.10,
        "epsilon_theta_ratio_cap": 3.0,
    },
    {
        "w_tag": "tmle_ct_tight_overlap_macro_prices",
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

OUT_DIR = str(EA_TDC_ROOT / "output" / "econark_dass" / "macro_prices_confirmatory")
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
