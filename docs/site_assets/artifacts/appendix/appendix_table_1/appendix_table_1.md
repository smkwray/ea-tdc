# Older baseline K=200 sensitivity

Local projection | Committed headline design | 1945Q4 to 2026Q2 | 2 outcomes | 85-93 obs

Coefficient table for Matched total deposits, Other component (q/q) in response to Baseline TDC estimate. Reported horizons are 0, 1, 2, 4, 8 with the corresponding standard errors and confidence intervals.

Generated at: 2026-06-10T22:58:24+00:00

## Notes
- Treatment: Baseline TDC estimate
- Response: Direct response at horizon h
- Controls: Real GDP, GDP deflator, Effective federal funds rate, Total reserves, Dflmx K200 F1, Dflmx K200 F2, Dflmx K200 F3, Dflmx K200 F4
- Covariance: Newey-West HAC
- Sample span: 1945Q4 to 2026Q2
- Observations: 85 to 93
- Scale: coefficients are responses of quarterly outcomes to the GDP-scaled TDC flow.
- Displayed branch: K=200 screened branch
- Relabeled legacy surface: baseline_tdc_lp_deposits with the K=200 screened branch. This is preserved as appendix/sensitivity output and is no longer the paper-facing main surface.
- Sample endpoint note: 2026Q2 is the latest labeled quarter as of 2026-06-10 and may reflect an in-progress quarter endpoint.
- Artifact source: Deposit responses to the baseline TDC estimate

| Outcome | Horizon | Beta | Se | Lower 95% | Upper 95% | P-value | Observations | Significance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Matched total deposits | 0 | 0.471 | 0.196 | 0.088 | 0.855 | 0.0160 | 93 | ** |
| Matched total deposits | 1 | 0.326 | 0.271 | -0.205 | 0.857 | 0.2293 | 92 |  |
| Matched total deposits | 2 | 0.079 | 0.146 | -0.207 | 0.365 | 0.5883 | 91 |  |
| Matched total deposits | 4 | 0.361 | 0.154 | 0.060 | 0.662 | 0.0187 | 89 | ** |
| Matched total deposits | 8 | 0.283 | 0.169 | -0.049 | 0.615 | 0.0946 | 85 | * |
| Other component (q/q) | 0 | -0.529 | 0.196 | -0.912 | -0.145 | 0.0069 | 93 | *** |
| Other component (q/q) | 1 | 0.230 | 0.274 | -0.307 | 0.766 | 0.4017 | 92 |  |
| Other component (q/q) | 2 | -0.065 | 0.191 | -0.439 | 0.310 | 0.7347 | 91 |  |
| Other component (q/q) | 4 | -0.193 | 0.209 | -0.603 | 0.217 | 0.3560 | 89 |  |
| Other component (q/q) | 8 | -0.109 | 0.110 | -0.324 | 0.106 | 0.3216 | 85 |  |
