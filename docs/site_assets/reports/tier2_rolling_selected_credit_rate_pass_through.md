# Rolling Selected-Lag Pass-Through Diagnostic

This diagnostic is descriptive stability evidence for an assumptive project. The rolling correlation rows are scale-free co-movement checks, not causal or canonical pass-through estimates.

Canonical interpretation remains the beta-per-$1 or selected-lag LP/regression pass-through estimate where available. Rolling correlations should only be used as secondary context on whether the deposit association is stable across windows.

- Window length: 48 quarters.
- Minimum observations per window: 40.
- Treatment: `tdc_tier2_regression_mmf_rrp_prop_bank_only_qoq`.

## Latest Window

| Outcome | Window | Rolling beta per $1 TDC | Effect per +$100B TDC | Rolling correlation | n |
|---|---:|---:|---:|---:|---:|
| `matched_total_deposits` | 2014Q3 to 2026Q2 | 0.531 | 53.08 | 0.594 | 46 |
| `other_component_tier2_regression_mmf_rrp_prop_bank_only_qoq` | 2014Q3 to 2026Q2 | -0.469 | -46.92 | -0.452 | 46 |

## Claim Boundary

- Treat sign and broad stability as descriptive evidence only.
- Do not read the rolling correlation as a pass-through share, price effect, or identification result.
- Use the selected-lag regression/LP coefficients for pass-through magnitudes.
