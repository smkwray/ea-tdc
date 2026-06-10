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

## Offset Accounting Implication (Identity, Not A Channel)

- Matched deposits and the same-treatment other component sum to TDC by construction, so the two h0 betas satisfy deposit_beta = 1 + residual_beta exactly.
- Full sample: deposits 0.616; signed residual -0.384. Equivalently, 0.384 of each TDC dollar does not appear as matched commercial-bank deposits in this specification.
- Pre-2020: deposits 0.199, signed residual -0.801. Excluding 2020Q1-2021Q4: deposits 0.342, signed residual -0.658.
- The residual row is an accounting implication of the deposit estimate and carries the same sampling uncertainty; it is not an independently identified channel, and its p-value is not a second discovery.
- The candidate attribution search found no clean named channel for the residual. The bounded interpretation remains a regime/perimeter-sensitive non-TDC residual, not a TGA, ON-RRP/MMF, or rate-competition causal claim.

## Robustness Appendix (Compact)

- HAC bandwidth: the pooled selected-lag h0 beta 0.616 is insensitive to the Newey-West bandwidth (lag 1: p=0.0023, lag 4: p=0.0061, lag 6: p=0.0042, lag 8: p=0.0025).
- Control selection is rank-aware and rejected controls are disclosed per row; regime differences can partly reflect control-set differences. Rejections at h0: latest_rolling_persistence (sample_split_latest_rolling_persistence_window): tier2_regression_bank_row_tier_pre_component_h15_scaled; pandemic_exclusion_drop_2020 (rolling_48q_drop_2020): tier2_regression_bank_row_tier_pre_component_h15_scaled; pandemic_exclusion_drop_2020q1_2021q4 (rolling_48q_drop_2020_2021): tier2_regression_bank_row_tier_pre_component_h15_scaled; pandemic_exclusion_drop_2021 (rolling_48q_drop_2021): tier2_regression_bank_row_tier_pre_component_h15_scaled; reserve_scarcity_or_low_liquidity (sample_split_reserve_scarcity_low_reserve_q25): tier2_regression_bank_row_tier_pre_component_h15_scaled.
- Factor controls are the pinned K=100 surface: K is the screened raw-feature width before compression to four factors, pinned across specifications rather than re-screened per regime cell.
- DML h0 0.695 [0.349, 1.041] is sensitivity-only: it shows the deposit response survives flexible controls and is not the preferred estimate.

## Runtime Selector Status

No trigger rule is runtime-selector validated. Out-of-sample and false-positive checks are screening diagnostics only, not promotion-grade validation.
