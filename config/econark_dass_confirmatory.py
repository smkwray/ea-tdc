from __future__ import annotations

import os
from pathlib import Path

# EA-TDC-owned DASS confirmatory config for econark.
# Keep project-specific settings here and invoke econark via:
#   python dass/run/launcher.py --config config/econark_dass_confirmatory.py

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
    ECONARK_ROOT / "fetchr-parity" / "out" / "interpol_core_contract_scaffold" / "final_lvl.csv"
)

EXTERNAL_Q_SERIES = {
    "TREAST": {"path": _PARITY_Q_PANEL, "column": "TREAST", "freq": "q"},
    "DEPOSITS": {"path": _PARITY_Q_PANEL, "column": "DEPOSITS", "freq": "q"},
    "RES": {"path": _PARITY_Q_PANEL, "column": "RES", "freq": "q"},
    "BAAFF": {"path": _PARITY_Q_PANEL, "column": "BAAFF", "freq": "q"},
    "PCEPILFE": {"path": _PARITY_Q_PANEL, "column": "PCEPILFE", "freq": "q"},
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
INCLUDE_GENERATED = False
SERIES_TO_GENERATE = {}
GENERATED_FREQ_POLICY = "coarsest"
APPLY_SAAR_ADJUSTMENTS = False
INFER_RAW_FREQ = True

CUTOFF_POLICY = "quarter_start"
EVENTS_CONFIG_PY = str(ECONARK_ROOT / "dass" / "events.py")
REQUIRE_RAW = False

DAILY_LAGS = 0
WEEKLY_LAGS = 0
MONTHLY_LAGS = 0
QUARTERLY_LAGS = 8
MAX_MISSING_PCT = 80.0
STANDARDIZE = False

PREP_INCLUDE_QUARTER_END = [
    "TREAST",
    "DEPOSITS",
    "RES",
    "BAAFF",
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

V1_W_SPEC_GRID = [32]
_CONFIRMATORY_FORCE_W_SERIES = ["TREAST", "DEPOSITS", "GDP", "Fed_Funds"]

V1_TMLE_DEFAULTS = {
    "treatment_mode": "shock",
    "tmle_mode": "continuous",
    "binary": False,
    "folds": 5,
    "make_stationary": False,
    "standardize": False,
    "shock_oos": "fold",
    "shock_w_max": 32,
    "w_max": 32,
    "w_select": "corr_t_then_variance",
    "force_w_series": _CONFIRMATORY_FORCE_W_SERIES,
    "density_floor": 0.05,
    "h_clip": 8.0,
    "epsilon_cap": 0.5,
    "nuisance_r2_floor": 0.02,
    "overlap_floor": 0.20,
    "epsilon_theta_ratio_cap": 2.0,
    "n_jobs": RUNNER_THREADS,
}

V1_DML_DEFAULTS = {
    "treatment_mode": "shock",
    "binary": False,
    "folds": 5,
    "make_stationary": False,
    "standardize": False,
    "shock_oos": "fold",
    "shock_w_max": 32,
    "w_max": 32,
    "w_select": "corr_t_then_variance",
    "force_w_series": _CONFIRMATORY_FORCE_W_SERIES,
    "hac_lags": 4,
    "n_jobs": RUNNER_THREADS,
}

V1_LP_DEFAULTS = {
    "treatment_mode": "shock",
    "binary": False,
    "folds": 5,
    "make_stationary": False,
    "standardize": False,
    "shock_oos": "fold",
    "shock_w_max": 32,
    "w_max": 32,
    "w_select": "corr_t_then_variance",
    "hac_lags": 4,
    "min_obs_per_regressor": 1.5,
    "max_condition_number": 1e10,
    "min_treatment_sd": 1e-8,
    "n_jobs": RUNNER_THREADS,
}

_CONFIRMATORY_OUTCOMES = ["DEPOSITS", "RES", "BAAFF", "PCEPILFE"]

_TMLE_VARIANTS = [
    {
        "w_tag": "tmle_ct_base",
        "density_floor": 0.05,
        "h_clip": 8.0,
        "epsilon_cap": 0.5,
        "nuisance_r2_floor": 0.02,
        "overlap_floor": 0.20,
        "epsilon_theta_ratio_cap": 2.0,
    },
    {
        "w_tag": "tmle_ct_loose_overlap",
        "density_floor": 0.02,
        "h_clip": 10.0,
        "epsilon_cap": 0.75,
        "nuisance_r2_floor": 0.01,
        "overlap_floor": 0.10,
        "epsilon_theta_ratio_cap": 3.0,
    },
    {
        "w_tag": "tmle_ct_tight_overlap",
        "density_floor": 0.10,
        "h_clip": 6.0,
        "epsilon_cap": 0.35,
        "nuisance_r2_floor": 0.03,
        "overlap_floor": 0.30,
        "epsilon_theta_ratio_cap": 1.5,
    },
]

V1_TMLE_JOBS = [
    {
        "treatment": "TREAST",
        "outcome": outcome,
        "horizons": [0, 1, 2, 4],
        **V1_TMLE_DEFAULTS,
        **variant,
    }
    for outcome in _CONFIRMATORY_OUTCOMES
    for variant in _TMLE_VARIANTS
]

V1_DML_JOBS = [
    {
        "treatment": "TREAST",
        "outcome": outcome,
        "horizons": [0, 1, 2, 4],
        **V1_DML_DEFAULTS,
    }
    for outcome in _CONFIRMATORY_OUTCOMES
]

V1_LP_JOBS = [
    {
        "treatment": "TREAST",
        "outcome": outcome,
        "horizons": [0, 1, 2, 4],
        **V1_LP_DEFAULTS,
    }
    for outcome in _CONFIRMATORY_OUTCOMES
]

V1_LP_JOBS_SOURCE = "V1_TMLE_JOBS"
V1_LP_REQUIRE_W_COLS = False

OUT_DIR = str(EA_TDC_ROOT / "output" / "econark_dass" / "continuous_tmle_confirmatory_auto")
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
