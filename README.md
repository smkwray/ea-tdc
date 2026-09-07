# EA-TDC

Website: [smkwray.github.io/ea-tdc](https://smkwray.github.io/ea-tdc/)

EA-TDC studies Treasury Deposit Contribution in quarterly public data.

## About

EA-TDC is a research package built on [EconArk](https://github.com/smkwray/econark). It collects the treatment definition, quarterly estimation code, figures, tables, and the static site for the current public release.

The finished package should be read in this order:

- headline evidence: the quarterly deposit response to the baseline bank-only TDC estimate
- independent boundary evidence: the narrower `tdcpass` strict source-side comparison, used to separate broad TDC from truly independent non-TDC measurement
- explanatory sidecar: component evidence on RU acquisition, Treasury operating cash, and positive Fed remittances
- sensitivity only: corrected Tier 2 and Tier 3 treatment-ladder variants, including the canonical Tier 2 depository-institution row with bill-discount and proportional MMF/RRP corrections

EA-TDC no longer treats residual/accounting closure as an independent non-TDC measurement lane. That boundary is handled in sibling `tdcpass`, where the broad Treasury-attributed TDC object is kept separate from the narrower strict source-side evidence.

The preferred corrected Tier 2 sensitivity is `tdc_tier2_canonical_di_mmf_rrp_prop_qoq`. Its bank-only, lower/upper MMF/RRP, and non-bill-discount rows remain robustness checks rather than replacements for the public `baseline_tdc_lp_deposits` headline.

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


## Frozen quarterly diagnostics

The diagnostic code preserves all 57 original rolling endpoints, 2011Q4–2025Q4.
Each window spans 48 nominal calendar quarters. The first eight start in
2000Q1–2001Q4 and contain 40–47 available observations; subsequent windows
contain 48 before the fixed pandemic-quarter deletion. Deleted quarters are not
refilled, and HAC pairs follow calendar distance.

The publication runner reads only the committed `config/open16_authority.json`.
Its default status is `unavailable`: accepted factor coordinates and the complete
historical input graph/runtime are not established. Source publication does not
authorize factor restoration. Covariance and pandemic diagnostics remain held;
the three unestablished source-resolution bounds separately withhold leg slopes.

A future approved record must pin a separately reviewed approval receipt and its
canonical `approved_inputs` SHA-256. Every artifact record contains a relative
`path`, `sha256`, and positive `bytes`. That inventory binds the panel and its
provenance receipt, accepted source commit/tree, frozen factor policy, complete
historical graph cut and completeness evidence, runtime identity or numerical
invariance evidence, full and sample factor scores, ordered screening/top-100,
full-vector 57-window equivalence reference, and upstream legs/receipt/producer.
The provenance receipt equals the approved inventory excluding its own record.
Approval requires historical evidence; a newly inventoried directory or matching
projection coefficients cannot establish original coordinate identity.

Only `retained_original_coordinates` and
`deterministically_restored_accepted_graph` are admissible origins. Refreshed or
reselected inputs, estimate matching, and equality of factor spans are outside
this contract. The runner computes no factors and accepts no caller-provided
input hashes. With a separately approved committed record, its invocation is:

```bash
python -B scripts/run_open16_diagnostics.py --producer-commit <exact-clean-commit>
```

## Separate frozen-input reproduction

`run_frozen_factor_reproduction.py` records a new reproduction from the pinned
July control-universe graph cut, its ordered columns, and retained design inputs.
It uses the accepted screening and factor code with unchanged balanced policy,
100 screened features, four factors, and the full panel including the pandemic.
The output retains input bytes, ordered screening and cutoff ties, complete factor
scores, canonical projection vectors, comparisons with archived estimates, and the
current code/runtime receipt. Differences are reported; matching old estimates is
not a selection rule.

```bash
python -B scripts/run_frozen_factor_reproduction.py \
  --producer-commit <exact-clean-commit> \
  --output-dir output/reproductions/<new-run-name>
```

The origin is `new_frozen_input_reproduction`. These results require scientific
review and do not authenticate original factor coordinates, alter historical
outputs or authority, or admit OPEN-16 leg slopes. No source data is downloaded.


The separate `fresh_frozen_conditioning_space` authority is currently pending
validation. It can admit only common-control covariance contributions and fixed,
full-panel-selected-control calendar-row deletion after a passed receipt binds
the production source and stated structural and numerical gates. Until that
receipt is committed, the fresh-reproduction lane fails closed. The vector
reference is the new reproduction; archived OPEN-01 supplies coefficient and
standard-error comparisons only. The failed absolute full-vector comparison is
preserved. Prospective `conditioning_space_v2` compares exact ranks and control
partitions, conditioning/full-design projectors, a column-normalized condition
cap, treatment beta/SE, and scale-invariant vector/covariance errors against both
pinned environments and an independent scaled SVD reference. Its complete policy
is recorded in the authority and must match the validation receipt exactly.
Numeric revalidation reuses the hashed passing raw/screen evidence without a raw
rebuild or factor extraction.

After that exact receipt is pinned, select `--authority-lane fresh-reproduction`
in `run_open16_diagnostics.py`. Historical-coordinate authenticity remains
unavailable, and this route excludes three-leg estimation. Pandemic deletion
continues to condition on factors selected using the full panel including the
pandemic; it is not selection-independent or out-of-sample validation.
