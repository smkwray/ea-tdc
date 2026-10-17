# EA-TDC

EA-TDC estimates quarterly associations between measured Treasury Deposit
Component (TDC) flows, bank deposits, and reserve balances. It combines public
data adapters, conditional projections, accounting decompositions, and stability
checks. The quarterly TDC measurement comes from
[TDCest](https://github.com/smkwray/tdcest).

## Interpretation

The central deposit estimate is a same-quarter conditional projection
coefficient. It is not the coefficient in the exact deposit accounting identity,
a causal effect, an immediate landing share, or a retention ratio. The measured
Treasury-flow perimeter and the deposit outcome perimeter differ; the latter
covers US-chartered institutions' currency and deposits. A coefficient below
one therefore does not measure the fraction of a Treasury payment that remains
in deposits.

The deposit construction checks pass, but the stability classification is
**unstable**. The controlling multiple-testing adjustment does not reject the
headline null in the current quarterly sample. Factor controls were selected
using the full panel, including the pandemic; the associated diagnostics are
conditional on that selection.

Reserve projections and their signed factor decompositions are conditional
associations. Simulation checks distinguish restricted joint tests from interval
coverage. A calibrated joint test does not validate era differences, unrestricted
intervals, or a causal interpretation. Failed coverage checks and unavailable
estimates remain explicit in the outputs.

## Analyses

| Analysis | Implementation and scope |
| --- | --- |
| Quarterly deposit projection | `scripts/run_open01_producer.py` and `scripts/run_open01_acceptance.py`; common-sample construction, projection, stability, and influence checks |
| Bank portfolio associations | `scripts/run_open02_producer.py`; joint transaction system with multiple-testing and influence checks; existing results remain limited diagnostics |
| Reserve projections | `scripts/run_reserve_projection.py`; same-quarter reserve change, signed reserve factors, and separate size and coverage simulations |
| Matched settlement clocks | `scripts/run_matched_clock_reserve.py`; compares quarterly exposure definitions on one common sample; [method and interpretation](docs/matched_clock_reserve.md) |
| Rolling and kernel-weighted projections | `scripts/run_reserve_rolling.py` and `scripts/run_local_projection.py`; descriptive variation in the conditional coefficient, with numerical screens and pointwise coverage labels |
| Frozen-control diagnostics | `scripts/run_frozen_factor_reproduction.py` and `scripts/run_open16_diagnostics.py`; explicitly distinguishes fresh reproduction from authentication of historical factor coordinates |

`OPEN-01`, `OPEN-02`, and `OPEN-16` are identifiers retained in the script and
configuration names. Their availability is determined by the input and approval
records each producer validates. Historical factor coordinates remain
unauthenticated; a separate approved fresh-reproduction configuration supports
bounded covariance and calendar-deletion diagnostics. Neither route supplies
conditional component slopes without their separate input checks.

The Financial Accounts vintage used for the quarterly refresh omits three loan
series required by the portfolio system. That refresh remains unavailable for
this system; replacement series are not silently substituted.

## Install and validate

Use Python 3.11 or later and run from a source checkout:

```bash
uv venv
uv pip install -e '.[dev]'
uv run --no-sync python -B -m ea_tdc --help
uv run --no-sync python -B -m pytest
uvx ruff check .
```

A configuration smoke check needs no downloaded data:

```bash
uv run --no-sync python -B -m ea_tdc smoke
```

It validates the manifests and writes a local bootstrap receipt under
`output/manifests/`. It does not run an empirical model. Tests construct small
fixtures; data-backed reproduction requires the exact external inputs described
below.

## Inputs and production runs

`config/runtime.yaml` declares local data/output roots and the public FRED and
Treasury download manifests. `config/source_registry.template.yaml` names the
optional sibling exports and their environment variables. Set those variables
locally when importing a bundle; a private environment file is not part of the
installation.

Empirical producers require the pinned source files, receipts, and sample
contracts named in their configuration. Installing the package does not supply
those datasets. Producers with `--producer-commit` require an exact clean source
commit and verify hashes before estimating. Inspect an interface before running
it, for example:

```bash
uv run --no-sync python -B scripts/run_reserve_projection.py --help
uv run --no-sync python -B scripts/run_local_projection.py --help
```

Do not substitute a newer source file for a pinned vintage or overwrite an
existing result bundle. Production runs write tables and receipts to ignored
local output directories. Calibration runs can be substantially more expensive
than the test suite or smoke check.

## Repository layout

- `src/ea_tdc/`: adapters, designs, estimation, and diagnostics.
- `config/`: source registries, versioned designs, and input-hash contracts.
- `scripts/`: analysis and validation entry points.
- `tests/`: regression and contract tests.
- `docs/`: treatment, outcome, and data-contract registries, plus method notes.
- `data/raw/`, `data/seed/`, `data/bundles/`, and `output/`: local inputs and
  generated results, ignored by Git.

[EconArk](https://github.com/smkwray/econark) provides the related modeling
framework. [TDCpass](https://github.com/smkwray/tdcpass) supplies an accounting
boundary comparison; it is not an independent causal estimate.
