# ea-tdc

`ea-tdc` estimates how US bank deposits, bank balance-sheet transactions and Federal Reserve reserve balances move with the Treasury Deposit Component (TDC), the part of the change in deposits at US banks and credit unions that comes from Treasury payments and trades in Treasury securities, as measured by [tdcest](https://github.com/smkwray/tdcest). Reserve estimates also use the Treasury component of reserves (TRC), the change in reserves caused by the same transactions. The estimates are quarterly projections, written as CSV tables with JSON receipts under `output/`.

The estimation scripts check every input against SHA-256 hashes pinned in `config/*.json`, so reproducing an estimate needs the same `tdcest` exports and downloads. The repository ships code and synthetic test fixtures only.

## Data

- Federal Reserve Economic Data (FRED) series listed in `config/fred_manifest_seed.csv`, through the FRED API when `FRED_API_KEY` is set and the public `fredgraph.csv` endpoint otherwise.
- Treasury Fiscal Data: auctions, buybacks and the Daily Treasury Statement (DTS) operating cash balance (`config/treasury_manifest.yaml`).
- The Financial Accounts of the United States (Z.1) CSV archive, for the bank-portfolio system.
- TDC and H.4.1 reserve-factor tables from `tdcest` (H.4.1 is the Federal Reserve's weekly release on factors affecting reserve balances), located through environment variables listed in `config/source_registry.template.yaml`.

Set `EA_TDC_VINTAGE` to a committed vintage file, such as `config/vintage_2026q2_canonical.json`, to rerun on a later sample with the same estimators.

## Install

Python 3.11 or later. From the repository root:

```bash
uv venv
uv pip install -e '.[dev]'
```

## Usage

Validate the configuration offline (writes `output/manifests/bootstrap_manifest_bundle.json`):

```bash
uv run --no-sync python -B -m ea_tdc smoke
```

`python -m ea_tdc --help` lists the data commands. The analyses are scripts that run only from a clean committed checkout, take that commit as `--producer-commit`, and refuse to overwrite an existing output directory:

```bash
uv run --no-sync python -B scripts/run_reserve_projection.py --producer-commit "$(git rev-parse HEAD)"
```

| Script | What it estimates |
|---|---|
| `scripts/run_open01_producer.py`, `scripts/run_open01_acceptance.py` | Deposit projections on TDC, with rolling-window stability and credit checks |
| `scripts/run_open02_producer.py` | Bank-portfolio transactions: deposits, loans, Treasury and agency securities |
| `scripts/run_reserve_projection.py` | Reserve projections, decompositions by H.4.1 factor, and coverage simulations |
| `scripts/run_matched_clock_reserve.py` | Reserve exposure definitions compared on one sample ([details](docs/matched_clock_reserve.md)) |
| `scripts/run_reserve_rolling.py` | The reserve projection in every 48-quarter window |
| `scripts/run_local_projection.py` | Kernel-weighted deposit and reserve paths over time |
| `scripts/run_frozen_factor_reproduction.py` | Factor controls rebuilt with the selection method fixed |
| `scripts/run_open16_diagnostics.py` | Residual-covariance and calendar-deletion diagnostics |

Other scripts run supplementary diagnostics; each takes `--help`.

## Output

Each analysis writes a directory such as `output/reports/reserve_projection_v1/` with CSV tables and a JSON receipt recording the commit, runtime versions, and hashes of the config, inputs and outputs. `output/` is ignored by Git.

## Limits

- Estimates are conditional projections on public-data proxies. A deposit coefficient below one does not separate behavior, coverage differences and measurement error.
- The canonical 2026Q2 vintage measures deposits as Financial Accounts checkable, time and savings deposits at banks and credit unions, less identified federal-government deposits; the default uses deposits at US-chartered depository institutions. Series codes are in the vintage files under `config/`.
- Factor controls were selected on the full sample, and rolling and kernel-weighted paths share quarters, so neighboring estimates are not independent. Read each table's interval-coverage label with its estimates.
- Coverage simulations take much longer than the tests.

## Tests

The tests use small fixtures and run offline:

```bash
uv run --no-sync python -B -m pytest
uvx ruff check .
```

## License

MIT. See [LICENSE](LICENSE).
