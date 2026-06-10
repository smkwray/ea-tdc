# EA-TDC Pass-Through Regime Validation Memo

This package converts EA-TDC TDC-deposit pass-through diagnostics into source-bound RateWall Assumption Mode inputs. It does not claim causal deposit creation, denominator calibration, holder allocation, incidence, welfare, pricing, or runtime regime selection.

## What Changed Relative To Pooled And Rolling Estimates

- The pooled selected-lag full-sample h0 estimate remains the high historical reference.
- Latest rolling persistence remains elevated, but pandemic-exclusion diagnostics lower it materially.
- Trigger candidates are now explicit, source-hashed, and fail closed unless sample and validation checks support scenario use.

## 2020 Versus 2021 Heterogeneity

- Latest rolling drop-2020 point: 0.248.
- Latest rolling drop-2021 point: 0.743.
- The signs of the drop tests are heterogeneous: dropping 2020 lowers the latest rolling beta sharply, while dropping 2021 raises it. Treat the pandemic block as heterogeneous, not as a single permanent structural break.

## Post-2020 Persistence

- Latest rolling excluding 2020Q1-2021Q4: 0.446.
- The beta does not collapse to zero, so the evidence supports some post-pandemic persistence; however, validation is not strong enough for automatic runtime selection.

## RateWall Use

- Assumption Mode scenario rows allowed: 8.
- Runtime selector rows allowed: 0.
- Current recommendation: import source-backed scenario rows for Assumption Mode review, keep runtime_selector_allowed=false.
- Sample windows in estimates and the RateWall contract are complete cases after transformations, lags, controls, and factor availability; raw data coverage is reported separately.

## No-TOTRESNS Robustness

- No-TOTRESNS robustness leaves the normal-forward coefficient at 0.338 versus 0.342 with contemporaneous TOTRESNS (delta -0.004), within the 0.15 materiality rule; regime ordering remains intact, so publish-and-freeze is supported.

## Runtime Selector Status

No trigger rule is runtime-selector validated. Out-of-sample and false-positive checks are screening diagnostics only, not promotion-grade validation.
