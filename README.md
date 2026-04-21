# EA-TDC

Website: [smkwray.github.io/ea-tdc](https://smkwray.github.io/ea-tdc/)

EA-TDC studies Treasury Deposit Contribution in quarterly public data.

## About

EA-TDC is a research package built on [EconArk](https://github.com/smkwray/econark). It collects the treatment definition, quarterly estimation code, figures, tables, and the static site for the current public release.

The finished package should be read in this order:

- headline evidence: the quarterly deposit response to the baseline bank-only TDC estimate
- coherence evidence: the deposit-accounting reconstruction, used as a diagnostic cross-check rather than independent validation
- explanatory sidecar: component evidence on RU acquisition, Treasury operating cash, and positive Fed remittances
- sensitivity only: corrected Tier 2 and Tier 3 treatment-ladder variants

This is also the boundary used in sibling `tdcpass`: the broad Treasury-attributed TDC object is not treated as the same thing as a strict independently validated deposit component.

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
