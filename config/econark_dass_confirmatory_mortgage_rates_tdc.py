from __future__ import annotations

import csv
import os
from pathlib import Path

# EA-TDC-owned exploratory DASS config for mortgage-rate channels.
# Treatment is the public TDC bank-only quarterly flow from tdcpass. Outcomes
# are changes in Freddie Mac's 30-year fixed mortgage rate and its spread over
# the 10-year Treasury yield.

CONFIG_DIR = Path(__file__).resolve().parent
EA_TDC_ROOT = CONFIG_DIR.parent
DEFAULT_ECONARK_ROOT = (
    Path.home()
    / "Library"
    / "CloudStorage"
    / "GoogleDrive-wray7830@gmail.com"
    / "My Drive"
    / "github"
    / "econark"
)
ECONARK_ROOT = Path(os.environ.get("ECONARK_ROOT", str(DEFAULT_ECONARK_ROOT))).expanduser()

CONFIRMATORY_SCOPE = "tdc_mortgage_rates"
PUBLIC_TREATMENT_LABEL = "tdc_bank_only_qoq"

START_DATE = "1990-03-31"
# The local MORTGAGE30US seed ends in early 2025Q3; stop at 2025Q2 so the
# mortgage-rate outcomes do not use incomplete or forward-filled quarters.
END_DATE = "2025-06-30"

SERIES_SOURCE = "fetch_dict"
FREDFETCH_PY = str(ECONARK_ROOT / "interpol" / "fredfetch.py")
FETCH_DICT_TXT = str(ECONARK_ROOT / "interpol" / "fetch" / "fetch_dict.txt")
MERGE_FETCH_DICT_METADATA = False
RAW_DIR = str(ECONARK_ROOT / "interpol" / "raw")
FETCH_DATA_CSV = None
FETCH_DATA_FALLBACK_SERIES = []

_LOCAL_Q_PANEL = EA_TDC_ROOT / "data" / "seed" / "interpol" / "out" / "final_lvl.csv"
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
    "GDP": {"path": _LOCAL_Q_PANEL, "column": "GDP", "freq": "q"},
    "PCE": {"path": _LOCAL_Q_PANEL, "column": "PCE", "freq": "q"},
    "M2": {"path": _LOCAL_Q_PANEL, "column": "M2", "freq": "q"},
    "RES": {"path": _LOCAL_Q_PANEL, "column": "RES", "freq": "q"},
    "DEPOSITS": {"path": _LOCAL_Q_PANEL, "column": "DEPOSITS", "freq": "q"},
    "TOTBKCR": {"path": _LOCAL_Q_PANEL, "column": "TOTBKCR", "freq": "q"},
    "DGS2": {"path": _LOCAL_Q_PANEL, "column": "DGS2", "freq": "q"},
    "DGS10": {"path": _LOCAL_Q_PANEL, "column": "DGS10", "freq": "q"},
    "Fed_Funds": {"path": _LOCAL_Q_PANEL, "column": "Fed_Funds", "freq": "q"},
    "INDPRO": {"path": _LOCAL_Q_PANEL, "column": "INDPRO", "freq": "q"},
    "MORTGAGE30US": {"path": _LOCAL_Q_PANEL, "column": "MORTGAGE30US", "freq": "q"},
}

INCLUDE_CONFIG_GENERATED = False
CONFIG_INTERPOL_PY = str(ECONARK_ROOT / "interpol" / "config_interpol.py")
INCLUDE_GENERATED = True
GENERATED_FREQ_POLICY = "coarsest"
APPLY_SAAR_ADJUSTMENTS = False
INFER_RAW_FREQ = True

SERIES_TO_GENERATE = {
    "MORTGAGE30US_D_Q": {
        "func": lambda df: df["MORTGAGE30US"].diff(),
        "components": ["MORTGAGE30US"],
        "freq": "q",
    },
    "MORTGAGE30US_DGS10_SPREAD_D_Q": {
        "func": lambda df: (df["MORTGAGE30US"] - df["DGS10"]).diff(),
        "components": ["MORTGAGE30US", "DGS10"],
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
    "MORTGAGE30US",
    "MORTGAGE30US_D_Q",
    "MORTGAGE30US_DGS10_SPREAD_D_Q",
]

RUNNER_THREADS = 2
MATH_THREADS = 1
DESIGN_CONCURRENCY = 2
ESTIMATOR_CONCURRENCY = 2
SKIP_EXISTING = False

RUN_V1_GRID = False
RUN_V1_CF = False
RUN_V1_DML = False
RUN_V1_TMLE = False
RUN_V1_LP = True
RUN_IDKIT = False
RUN_ROBUSTNESS_PACK = False
RUN_PLACEBO_DML = False
RUN_BENCHMARKS = False
RUN_D2_MONEY_AGG = False
RUN_BILLS_CONTROL_VARIANTS = False
RUN_HEADLINE_BUNDLE = False

_FORCE_W_SERIES = [
    "tdc_est",
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
    "treatment": "tdc_est",
    "horizons": [0, 1, 2, 4, 8],
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
    {**_BASE_JOB, "outcome": "MORTGAGE30US_D_Q", "w_tag": "mortgage30us_d_q"},
    {
        **_BASE_JOB,
        "outcome": "MORTGAGE30US_DGS10_SPREAD_D_Q",
        "w_tag": "mortgage30us_dgs10_spread_d_q",
    },
]

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

V1_LP_JOBS_SOURCE = "V1_LP_JOBS"
V1_LP_REQUIRE_W_COLS = False

OUT_DIR = str(EA_TDC_ROOT / "output" / "econark_dass" / "mortgage_rates_tdc")
OUT_CSV = "stacked_quarterly.csv"
OUT_META_MD = "stacked_quarterly_meta.md"
DESIGN_OUT_DIR = f"{OUT_DIR}/design"
CF_OUT_DIR = f"{OUT_DIR}/cf"
TMLE_OUT_DIR = f"{OUT_DIR}/tmle"
DML_OUT_DIR = f"{OUT_DIR}/dml"
LP_OUT_DIR = f"{OUT_DIR}/lp"
RESULTS_CSV = f"{OUT_DIR}/results.csv"
OVERLAP_MD = f"{OUT_DIR}/overlap.md"
