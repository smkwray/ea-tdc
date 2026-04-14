# Event Sidecar Artifact Pack

## Purpose

This pack is the compact event-side surface for release closeout.
Use it for sidecar rates and plumbing evidence, not for headline causal claims.

## Rates benchmark

Medium-horizon rate and term-spread responses on the reviewed event sample.

Table export: `event_sidecar_rates_table.csv`

- `term_spread_10y_3m` at `h=63`: `beta ≈ 2.05753`, `p ≈ 1.158e-08`
- `threefytp10` at `h=63`: `beta ≈ 1.19504`, `p ≈ 2.42e-05`
- `dgs10` at `h=63`: `beta ≈ 2.25111`, `p ≈ 0.0001365`
- `dgs2` at `h=63`: `beta ≈ 1.75623`, `p ≈ 0.006763`
- `repo_spread` at `h=21`: `beta ≈ 0.0309873`, `p ≈ 0.04433`
- `dgs2` at `h=21`: `beta ≈ 0.634783`, `p ≈ 0.07113`

## Risk and plumbing benchmark

Short-horizon risk move plus later balance-sheet plumbing responses on the reviewed event sample.

Table export: `event_sidecar_plumbing_table.csv`

- `reserve_balances_change` at `h=1`: `beta ≈ 915.159`, `p ≈ 9.411e-05`
- `sp500_return` at `h=1`: `beta ≈ -0.0444349`, `p ≈ 0.0002989`
- `tga_balance_change` at `h=21`: `beta ≈ 3513.95`, `p ≈ 0.006408`
- `rrp_balance_change` at `h=21`: `beta ≈ -4.08628`, `p ≈ 0.02243`
- `fed_balance_sheet_change` at `h=21`: `beta ≈ -862.247`, `p ≈ 0.02597`

## Caption language

- Rates/plumbing sidecar on reviewed `n=14` event sample.
- Signals are useful for benchmark orientation, but inference remains fragile at this sample size.
