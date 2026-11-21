# ea-tdc

`ea-tdc` estimates quarterly conditional projections linking measured Treasury Deposit Component (TDC) flows to bank deposits, bank balance-sheet transactions, and Federal Reserve reserve balances. It reads public Federal Reserve and Treasury data alongside quarterly TDC and reserve-factor tables from [tdcest](https://github.com/smkwray/tdcest) and writes CSV tables and JSON run receipts under `output/`.

## What it does

The package combines data adapters, quarterly design builders, and estimation scripts for:

- Quarterly deposit projections and rolling-window stability checks (`scripts/run_open01_producer.py`, `scripts/run_open01_acceptance.py`).
- A quarterly bank-portfolio transaction system covering deposits, loans, Treasury securities, and agency securities (`scripts/run_open02_producer.py`).
- Same-quarter reserve projections, signed H.4.1 reserve-factor decompositions, and size/coverage calibration simulations (`scripts/run_reserve_projection.py`).
- Matched-clock comparisons across quarterly reserve exposure definitions on a common sample (`scripts/run_matched_clock_reserve.py`; see [docs/matched_clock_reserve.md](docs/matched_clock_reserve.md)).
- 48-quarter rolling reserve projections (`scripts/run_reserve_rolling.py`) and kernel-weighted local deposit and reserve projections (`scripts/run_local_projection.py`).
- Covariance and calendar-deletion diagnostics with the control set held fixed (`scripts/run_frozen_factor_reproduction.py`, `scripts/run_open16_diagnostics.py`).

## Data

- FRED macro, banking, and interest-rate series listed in `config/fred_manifest_seed.csv` (configured in `config/runtime.yaml`). Export `FRED_API_KEY` to fetch via the FRED JSON API; if unset, downloads fall back to FRED's public `fredgraph.csv` endpoint.
- Treasury Fiscal Data API datasets (`auctions_query`, `buybacks_operations`, and DTS `operating_cash_balance`) listed in `config/treasury_manifest.yaml`.
- Official Federal Reserve Financial Accounts of the United States (Z.1) CSV archive for the bank-portfolio transaction system.
- Quarterly TDC and reserve-factor outputs from `tdcest`, plus optional local seed exports configured through environment variables in `config/source_registry.template.yaml` (such as `EA_TDC_SEED_TDCEST_BUNDLE`).

Empirical scripts verify SHA-256 hashes against pinned input files declared in `config/*.json`; installing the package provides code and synthetic test fixtures only. To rerun on an alternate committed vintage configuration, set `EA_TDC_VINTAGE` to its repository-relative path.

## Install

Requires Python 3.11 or later. From the repository root:

```bash
uv venv
uv pip install -e '.[dev]'
```

## Usage

Check the CLI and validate configuration manifests offline (writes `output/manifests/bootstrap_manifest_bundle.json`):

```bash
uv run --no-sync python -B -m ea_tdc --help
uv run --no-sync python -B -m ea_tdc smoke
```

Core CLI subcommands include `fetch-fred`, `fetch-treasury`, `seed-source`, `adapt-tdcest`, `build-quarterly-design`, and `estimate-quarterly-job`.

Production scripts under `scripts/` require `--producer-commit` matching a clean committed git HEAD and refuse to overwrite an existing output directory. Inspect a script's required arguments before running it:

```bash
uv run --no-sync python -B scripts/run_reserve_projection.py --help
uv run --no-sync python -B scripts/run_local_projection.py --help
```

## Output

Generated CSV tables, manifests, and provenance receipts are written to ignored local directories under `output/`—for example, `output/manifests/bootstrap_manifest_bundle.json` from `smoke` and per-run directories such as `output/reports/reserve_projection_v1/` from the estimation scripts. Each receipt records the producer commit, config hash, input hashes, runtime versions, and output hashes.

## Limits

- Quarterly deposit and reserve estimates are conditional projections on public-data proxies. The Treasury-flow proxy and US-chartered depository institution deposits cover different balance-sheet scopes, so a slope below one reflects both behavioral offsets and scope differences.
- Factor controls were selected on the full panel, and rolling or kernel-weighted paths share overlapping quarters; read interval coverage labels in the output tables alongside point estimates.
- Simulation calibration runs take substantially longer than the test suite or `smoke` check.

## Tests

```bash
uv run --no-sync python -B -m pytest
uvx ruff check .
```

The test suite uses small fixtures and runs offline without downloaded production inputs.
