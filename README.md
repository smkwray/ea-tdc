# EA-TDC

Website: [smkwray.github.io/ea-tdc](https://smkwray.github.io/ea-tdc/)

EA-TDC studies the Treasury component of deposits in quarterly public data.

## About

EA-TDC is a research package built on [EconArk](https://github.com/smkwray/econark). It collects the treatment definition, quarterly estimation code, figures, tables, and the static site for the current public release.

The current release centers on three pieces:

- the quarterly deposit response to the baseline TDC estimate
- the deposit-accounting reconstruction that tracks the non-TDC deposit component
- event-side evidence on rates and liquidity plumbing

## Project structure

- `src/ea_tdc/` Python package for data adapters, designs, estimation, reporting, artifacts, and site generation
- `config/` manifests, job definitions, and runtime configuration
- `docs/` static site bundle for GitHub Pages
- `tests/` regression tests

## Build

Run the test suite:

```bash
python -B -m pytest
```

Rebuild the site:

```bash
python -B -m ea_tdc build-site
```
