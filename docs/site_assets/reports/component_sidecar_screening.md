# Component Sidecar Screening

## Verdict

The component lane is the strongest explanatory sidecar in the public package.
RU acquisition is the clearest long-sample component across deposits and liquidity,
Treasury cash is more informative for liquidity accounting than headline deposits,
and positive remittances matter for deposits plus Fed-assets-relative liquidity.
Coefficients are responses of quarterly outcomes to the GDP-scaled component flow unless noted otherwise.

## RU acquisition reduced form

Core RU-acquisition component on deposits, residual deposits, and baseline liquidity outcomes.

- `other_component_qoq` at `h=4`: `beta ≈ -0.500078`, `p ≈ 1.573e-09`
- `reserve_balances_qoq` at `h=0`: `beta ≈ 0.565937`, `p ≈ 2.439e-08`
- `matched_total_deposits` at `h=0`: `beta ≈ 0.80194`, `p ≈ 4.03e-08`
- `repo_spread` at `h=8`: `beta ≈ -1.24252e-05`, `p ≈ 3.327e-05`
- `reserve_balances_net_fed_treasury_qoq` at `h=1`: `beta ≈ -0.293231`, `p ≈ 0.002795`
- `matched_total_deposits` at `h=2`: `beta ≈ 0.315366`, `p ≈ 0.00754`
- `matched_total_deposits` at `h=1`: `beta ≈ 0.422463`, `p ≈ 0.008491`
- `reserve_balances_net_fed_treasury_qoq` at `h=2`: `beta ≈ -0.254475`, `p ≈ 0.04726`
- `other_component_qoq` at `h=8`: `beta ≈ -0.186187`, `p ≈ 0.04994`
- `matched_total_deposits` at `h=8`: `beta ≈ -0.210548`, `p ≈ 0.0622`

## Treasury cash reduced form

Treasury operating cash component on deposits, residual deposits, and baseline liquidity outcomes.

- `other_component_qoq` at `h=0`: `beta ≈ -0.975802`, `p ≈ 0`
- `reserve_balances_net_fed_treasury_qoq` at `h=0`: `beta ≈ 0.580272`, `p ≈ 4.004e-10`
- `reserve_balances_net_fed_treasury_qoq` at `h=8`: `beta ≈ 0.54082`, `p ≈ 3.568e-07`
- `reserve_balances_qoq` at `h=8`: `beta ≈ 0.4756`, `p ≈ 6.243e-05`
- `other_component_qoq` at `h=1`: `beta ≈ -0.262792`, `p ≈ 0.07173`
- `repo_spread` at `h=1`: `beta ≈ -6.39105e-06`, `p ≈ 0.08358`

## Positive remittance reduced form

Positive remittance add-back on deposits, residual deposits, and baseline liquidity outcomes.

- `repo_spread` at `h=8`: `beta ≈ 0.000507092`, `p ≈ 0.0003177`
- `reserve_balances_net_fed_treasury_qoq` at `h=0`: `beta ≈ -12.4335`, `p ≈ 0.003123`
- `other_component_qoq` at `h=1`: `beta ≈ 12.1942`, `p ≈ 0.003371`
- `matched_total_deposits` at `h=1`: `beta ≈ 10.5657`, `p ≈ 0.006544`
- `other_component_qoq` at `h=2`: `beta ≈ 10.9783`, `p ≈ 0.01279`
- `matched_total_deposits` at `h=2`: `beta ≈ 9.35171`, `p ≈ 0.0146`
- `other_component_qoq` at `h=0`: `beta ≈ 9.00173`, `p ≈ 0.02611`
- `matched_total_deposits` at `h=0`: `beta ≈ 6.06733`, `p ≈ 0.06172`
- `reserve_balances_qoq` at `h=0`: `beta ≈ -8.86458`, `p ≈ 0.08239`
- `matched_total_deposits` at `h=4`: `beta ≈ 6.49617`, `p ≈ 0.08287`
- `reserve_balances_qoq` at `h=4`: `beta ≈ 6.34828`, `p ≈ 0.09012`

## RU acquisition liquidity decomposition

RU-acquisition component across reserves, Fed assets, Treasury holdings, and repo plumbing.

- `fed_treasury_holdings_qoq` at `h=0`: `beta ≈ 0.746833`, `p ≈ 5.538e-10`
- `reserve_balances_qoq` at `h=0`: `beta ≈ 0.565937`, `p ≈ 2.439e-08`
- `fed_treasury_holdings_qoq` at `h=2`: `beta ≈ 0.347303`, `p ≈ 1.182e-07`
- `fed_total_assets_qoq` at `h=0`: `beta ≈ 1.05809`, `p ≈ 6.066e-07`
- `repo_spread` at `h=8`: `beta ≈ -1.24252e-05`, `p ≈ 3.327e-05`
- `fed_treasury_holdings_qoq` at `h=1`: `beta ≈ 0.528537`, `p ≈ 0.0004188`
- `fed_total_assets_qoq` at `h=2`: `beta ≈ 0.325948`, `p ≈ 0.002363`
- `reserve_balances_net_fed_treasury_qoq` at `h=1`: `beta ≈ -0.293231`, `p ≈ 0.002795`
- `reserve_balances_net_fed_assets_qoq` at `h=1`: `beta ≈ -0.372979`, `p ≈ 0.01628`
- `reserve_balances_net_fed_assets_qoq` at `h=0`: `beta ≈ -0.492157`, `p ≈ 0.01835`
- `fed_total_assets_qoq` at `h=1`: `beta ≈ 0.608284`, `p ≈ 0.0194`
- `reserve_balances_net_fed_treasury_qoq` at `h=2`: `beta ≈ -0.254475`, `p ≈ 0.04726`
- `reserve_balances_net_fed_assets_qoq` at `h=2`: `beta ≈ -0.233119`, `p ≈ 0.04994`
- `fed_total_assets_qoq` at `h=4`: `beta ≈ 0.25443`, `p ≈ 0.063`

## Treasury cash liquidity decomposition

Treasury-cash component across reserves, Fed assets, Treasury holdings, and repo plumbing.

- `reserve_balances_net_fed_assets_qoq` at `h=0`: `beta ≈ 0.748136`, `p ≈ 3.099e-10`
- `reserve_balances_net_fed_treasury_qoq` at `h=0`: `beta ≈ 0.580272`, `p ≈ 4.004e-10`
- `reserve_balances_net_fed_treasury_qoq` at `h=8`: `beta ≈ 0.54082`, `p ≈ 3.568e-07`
- `reserve_balances_net_fed_assets_qoq` at `h=8`: `beta ≈ 0.489603`, `p ≈ 8.502e-06`
- `reserve_balances_qoq` at `h=8`: `beta ≈ 0.4756`, `p ≈ 6.243e-05`
- `repo_spread` at `h=1`: `beta ≈ -6.39105e-06`, `p ≈ 0.08358`

## Positive remittance liquidity decomposition

Positive remittance component across reserves, Fed assets, Treasury holdings, and repo plumbing.

- `repo_spread` at `h=8`: `beta ≈ 0.000507092`, `p ≈ 0.0003177`
- `reserve_balances_net_fed_assets_qoq` at `h=0`: `beta ≈ -12.3947`, `p ≈ 0.001135`
- `reserve_balances_net_fed_treasury_qoq` at `h=0`: `beta ≈ -12.4335`, `p ≈ 0.003123`
- `reserve_balances_net_fed_assets_qoq` at `h=8`: `beta ≈ -8.53716`, `p ≈ 0.007225`
- `reserve_balances_net_fed_assets_qoq` at `h=1`: `beta ≈ -7.60485`, `p ≈ 0.03516`
- `fed_total_assets_qoq` at `h=8`: `beta ≈ 8.5014`, `p ≈ 0.04478`
- `fed_total_assets_qoq` at `h=4`: `beta ≈ 12.3577`, `p ≈ 0.04879`
- `reserve_balances_qoq` at `h=0`: `beta ≈ -8.86458`, `p ≈ 0.08239`
- `reserve_balances_qoq` at `h=4`: `beta ≈ 6.34828`, `p ≈ 0.09012`

## RU acquisition under low reserves

State-interaction probe for RU-acquisition effects under reserve scarcity.

- `matched_total_deposits` at `h=4`: `interaction beta ≈ 0.841818`, `interaction p ≈ 0.09533`, `low-state beta ≈ 0.226879`, `p ≈ 0.01779`, `high-state beta ≈ 1.0687`, `p ≈ 0.03972`

## Treasury cash under ON RRP drain

State-interaction probe for Treasury-cash effects during ON RRP drain episodes.

- `matched_total_deposits` at `h=4`: `interaction beta ≈ -0.164241`, `interaction p ≈ 0.008`, `low-state beta ≈ -0.00603883`, `p ≈ 0.9395`, `high-state beta ≈ -0.149775`, `p ≈ 0.1107`

## Interpretation

- The component reduced forms are the clearest public summary of which Treasury legs move deposits in the long sample.
- The liquidity decomposition is the companion view for how each component transmits through reserves, Fed assets, and repo conditions.
- The narrow state results are secondary context rather than a standalone public claim.
