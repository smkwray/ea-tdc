# Matched-clock quarterly reserve projection

The 80-quarter comparison covers 2006Q1–2025Q4 on one common mask. Every
exposure uses the change in H.4.1 reserves between adjacent last Wednesdays,
an intercept and quarter-of-year indicators. The pooled sample includes
2008Q4 and 2015Q4. The early era is source-truncated relative to the admitted
2003Q1 start. The admitted primary table uses **preferred TRC**; its common-
sample reference is therefore shown separately from the core-to-core comparison.

| Exposure | Pooled, n=80 | 2006Q1–2008Q3, n=11 | 2009Q1–2015Q3, n=27 | 2016Q1–2025Q4, n=40 |
| --- | ---: | ---: | ---: | ---: |
| Admitted mixed-clock core | 0.505973 | -0.152751 | 0.889661 | 0.483766 |
| Matched weekly-summed cash-corrected core proxy (primary) | 0.536192 | -0.159137 | 0.921693 | 0.539577 |
| Matched weekly-summed par sensitivity | 0.559624 | -0.159430 | 0.921311 | 0.569146 |
| Matched Wednesday cash leg alone | 0.136385 | -0.425771 | 1.312621 | 0.104629 |
| Admitted mixed-clock preferred (reference) | 0.607013 | -0.152751 | 0.889661 | 0.633342 |

Each transition has n=1. Neither has an identified slope with the declared
controls. The projection table carries explicit unavailable cells; the
transition table reports raw exposures, reserve changes and accounting
components. No uncentered ratio is substituted for a conditional projection.

The matched clock removes the Treasury-deposit clock bridge, whose covariance
contribution is exactly zero in all estimated matched rows. It does not remove
unknown secondary settlement prices, coupon entitlement, route ambiguity,
unmatched internal-account uncertainty or leverage. The primary core corrects
supported primary auction cash, paid redemption inflation and verified internal
postings; unsupported corrections remain uncertainty. The par sensitivity
omits the primary, redemption, maturity-payment and internal corrections. The
cash-only reference carries the full H.4.1 Treasury-securities factor separately.
No new route scenario is estimated.

For each estimated row, the signed RRP contribution, other reserve factors and
all named bridges sum to beta minus one within 1e-9. Matched decompositions sum
the upstream weekly identity; admitted decompositions reuse the admitted ledger.
Accounting closure does not validate the measurement proxy. Exact same-clock
TRC has coefficient one by accounting; the reported coefficients are conditional
covariances with measured exposure, never landing, retention or causal responses.
Quarterly aggregation changes the projection functional, so a quarterly
coefficient need not equal an average of weekly coefficients.

Early-era matched-core partial variation is concentrated in 2008Q3: 25.0%
for the largest quarter and 65.6% for the largest three. Deleting 2008Q3 gives
beta -0.000733 rather than -0.159137. The pooled matched-core largest quarter
is 2021Q3 (20.8%; top three 39.2%); deleting it gives beta 0.503736. All variants
have concentration diagnostics and refitted deletion slopes for the largest
quarter, 2008Q3, 2008Q4 and 2020Q1. Dates outside an era are explicitly marked.

The exposure-difference tables provide all 80 quarterly component differences
and raw and quarter-of-year-adjusted summaries. The additive variance share is
Cov(component, total exposure difference)/Var(total exposure difference), so
signed shares sum to one. Own variance divided by total variance is also
reported; those shares need not sum to one because components covary. Covariances
with reserves use n normalization in USD million squared. The adjusted additive
shares are 60.3% deposit clock, 43.5% Fed source/par versus Z.1, -3.8% applied
internal postings, with the remaining components reported separately. A slope
change depends on both numerator and exposure denominator; these covariance
summaries do not assign an additive share of the coefficient change.

Inference is descriptive only. No era tests, bootstrap, intervals or stars are
produced. Eras were declared after the pilot. The admitted HAC era tests
over-reject, unrestricted bands undercover, and the conservative restricted
test reports p=0.18; it establishes neither equality nor inference validity for
this exposure. The admitted table and warning remain intact.

## Reproduction and retained artifacts

`config/matched_clock_reserve.json` pins every consumed upstream file, the
admitted bundle and output inventory, and the design authorities. From a clean
committed checkout, run:

```sh
uv run python scripts/run_matched_clock_reserve.py --producer-commit COMMIT --tdcest-root TDCEST_ROOT --adjudication ADJUDICATION_FILE --design-memo DESIGN_MEMO_FILE
uv run pytest
```

The producer writes only `output/reports/matched_clock_reserve_v1/`:
`projections_v1.csv`, `influence_v1.csv`, `transitions_v1.csv`,
`exposure_difference_quarters_v1.csv`, `exposure_difference_summary_v1.csv`,
`excluded_quarters_v1.csv` and `receipt.json`. The receipt binds input and output
hashes, the committed producer, runtime, common mask and preservation checks.
It distinguishes the weekly artifact producer from the upstream checkout commit.
Outputs and input bundles remain gitignored. Reproduction requires the exact
input bytes and source commit recorded in the receipt.

The economic tests mutate signed bridges, cash corrections, interval endpoints,
weekly sums, completeness, masks and protected outputs. The reserve estimates do not consume a TDCpass bundle.

Preservation checks compare the existing output inventory with the hashes
declared in the configuration; a mismatch stops the producer.
