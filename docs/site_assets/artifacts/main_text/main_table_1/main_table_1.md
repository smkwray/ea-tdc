# Long-history Tier 2 selected-lag h=0 coefficient table

Local projection | Committed headline design | 2002Q1 to 2025Q4 | 6 outcomes | 96-96 obs

H=0 coefficient table for Matched total deposits, Other component, same Tier 2 treatment, Strict loan core, Mortgages, Consumer credit, Bank credit in response to Long-history Tier 2 TDC. Entries are $B per +$100B TDC, with p-values and observations.

Generated at: 2026-06-10T22:41:36+00:00

## Notes
- Treatment: Long-history Tier 2 TDC
- Response: Direct response at horizon h
- Controls: Real GDP, GDP deflator, Effective federal funds rate, Total reserves, Tier 2 method-tier control, Strict loan-core lag 2, Strict loan-core lag 4, Consumer-credit lag 4, Bank-credit lag 4, 2Y Treasury-yield lag 4, 10Y Treasury-yield lag 1, 10Y Treasury-yield lag 2, Dflmx K100 F1, Dflmx K100 F2, Dflmx K100 F3, Dflmx K100 F4
- Covariance: Newey-West HAC
- Sample span: 2002Q1 to 2025Q4
- Observations: 88 to 96
- Scale: coefficients are $B responses per +$100B TDC.
- Run/specification: regression_mmf_rrp_bank_long_selected_credit_rate_lags. Rows report h=0 effects in $B per +$100B TDC.
- Claim boundary: consumer credit is a guarded candidate margin, not a broad crowding-out headline.
- Artifact source: Long-history Tier 2 selected-lag pass-through

| Outcome | Horizon | Beta | P-value | Observations | Significance |
| --- | --- | --- | --- | --- | --- |
| Matched total deposits | 0 | 61.635 | 0.0023 | 96 | *** |
| Other component, same Tier 2 treatment | 0 | -38.365 | 0.0578 | 96 | * |
| Strict loan core | 0 | -3.750 | 0.2006 | 96 |  |
| Mortgages | 0 | -0.399 | 0.8520 | 96 |  |
| Consumer credit | 0 | -3.351 | 0.0332 | 96 | ** |
| Bank credit | 0 | 7.369 | 0.5272 | 96 |  |
