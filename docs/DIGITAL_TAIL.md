# SPX Digital Tail: baseline

Standalone research strategy: one far-OTM put financed by N narrow put credit
spreads, all at the same expiry. This is a new construction, separate from the
weekly short-ITM-put scaleout experiments.

## Fixed baseline

| Parameter | Value |
| --- | --- |
| Entry | First eligible session of each calendar month, at most one package |
| Expiry | Closest standard third Friday to 120 calendar days, within 105-135 days |
| Tail | One put, delta nearest -0.01, tolerance +/-0.0025, below both spread strikes |
| Sigma | Nearest-to-spot ATM put annual IV times sqrt(actual calendar DTE / 365) |
| Short strike | Nearest listed strike to spot times exp(-2.5 times sigma) |
| Hedge strike | Immediately adjacent lower listed strike |
| Funding | Minimum integer N covering long purchase and both entry/exit commissions |
| Commission | Assumed USD 1 per contract per side; configurable, not a broker tariff |
| Entry/exit execution | Buy at ask, sell at bid; no additional slippage |
| Exit | One observed trading session before the source expiry date |
| Capital | USD 100,000, configurable |
| Risk caps | 5% per new package, 20% aggregate, based on current marked equity |
| Quantity | One tail contract per eligible entry, maximum 100 financing spreads |

No skew timing filter, rolling, early profit target, cash yield, or financing
income in this baseline. Terminal financing covers the long and commissions;
closing before expiry may still produce a loss even in a quiet market.

## Payoff and accounting

Let Kt < Kl < Ks be the tail, hedge, and short strikes, w = Ks - Kl,
and C the entry credit net of entry and reserved exit commissions, in USD.

```text
expiry PnL(S) = C + 100 * [max(Kt-S, 0) - N*(max(Ks-S, 0)-max(Kl-S, 0))]
max terminal loss = 100*N*w - C
lower breakeven = Kt - max_terminal_loss/100
upper breakeven = Ks - C/(100*N)
```

Risk reservation is the initial terminal-loss budget, not broker margin or a
guarantee on bid/ask liquidation costs. Integer sizing can produce more surplus
credit than requested. Unfundable or over-budget packages are skipped; every
attempt is recorded. New entries are blocked while open packages have stale marks.

Daily equity uses mid marks and reserves exit commissions. Vertical mid marks
are projected into the intersection of [0, width] and quoted bid/ask bounds:
overlapping quotes near expiry can otherwise produce negative vertical values.
Execution always uses the actual leg bids/asks, without that projection.
PnL of the tail and financing spreads sums to package PnL on the same entries;
these are component attributions, not independently sized strategy benchmarks.

## Data limitations

- Existing inputs cover 2010-2023; there is no 2008 crash in this sample.
- No official SET settlement series or AM/PM contract identifiers are present.
  Consequently the baseline uses pre-expiry quote liquidation, not settlement.
  See [Cboe settlement specification](https://www.cboe.com/tradable-products/sp-500/spx-options/spx-specifications).
- Some source chains change between Thursday and Friday expiry labels. Only
  third-Friday expiries are eligible; Thursday/holiday-shifted expiries are
  excluded, not automatically remapped. This reduces early-period coverage.
- ATM IV outside 3%-300%, ATM put delta outside [-0.65, -0.35], crossed quotes,
  ambiguous duplicate contracts, and invalid funding quotes are rejected.
- Missing marks are carried and flagged, not replaced with intrinsic values or
  future quotes. Missing terminal quotes leave a package unresolved. Results
  containing unresolved packages are provisional and not evidence of an edge.
- Expiries past the data endpoint are excluded, avoiding partial terminal trades.
- Mid-only performance is not the execution assumption. Zero cash-return Sharpe
  and daily drawdown depend on the stated initial capital and risk limits.

## Reproduce

```bash
docker compose run --rm -T analytics python -m taleb_kurtosis.digital_tail
```

Optional flags: `--start-date`, `--end-date`, `--data-dir`, `--out-dir`,
`--capital`, `--fee-per-contract`, `--package-risk`, `--total-risk`.

Outputs in `results/digital_tail_atm_iv_baseline/`:

- `manifest.json`: exact parameters, data range, assumptions and diagnostics.
- `trades.parquet`, `daily_equity.parquet`, `summary.csv`: package and portfolio results.
- `entry_audit.csv`: accepted and rejected daily entry attempts.
- `report.html`: offline interactive Plotly report, with embedded JavaScript.

Dashboard preset: **Digital Tail / ATM IV**. Includes equity decomposition,
daily drawdown, monthly PnL, reserved risk, package PnL and selectable expiry payoff.
The expiry payoff is theoretical; historical PnL uses pre-expiry liquidation.

## Baseline result (2010-2023)

Only 10 packages entered, in 2017 and 2019, with 353 of 3,499 sessions having an
open position at the end of the day. Net PnL is -USD 7,038, maximum daily marked
drawdown -USD 8,063.50 (-7.98%), and 3 of 10 packages are profitable. Attribution:
tail puts -USD 1,894; financing spreads -USD 5,144. No unresolved packages or stale
mark sessions; 22 sessions required vertical-mid projection.

The main rejection reasons are unavailable standard-Friday series (1,591 daily
attempts), the package risk cap (888), and nonpositive/invalid executable credit
(664). These are daily attempts, not independent opportunities. This restricted
sample does not test the hypothesis across all market regimes.

The package closed on 2020-03-19 lost USD 7,056, exceeding its USD 4,976 terminal
loss budget. Its 25-point spread cost 35.40 points to close at the separate leg
bid/ask quotes (171.30 - 135.90). This execution assumption is deliberately not
clipped to theoretical fair value. Maximum drawdown on the all-position bid/ask
liquidation equity series is USD 8,479, versus USD 8,063.50 on the main marked
series. Future experiments should separate execution/settlement sensitivity from
changes to the strategy parameters.

## Barrier and profit-target comparison

The separate comparison runner preserves the original baseline artifacts:

```bash
docker compose run --rm --build -T analytics python -m taleb_kurtosis.digital_tail_comparison
docker compose run --rm -T analytics python -m taleb_kurtosis.digital_tail_comparison \
  --capital 250000 --out-dir results/digital_tail_barrier_comparison_250k
```

It evaluates 12 independent portfolios: sigma 2.5, 3.0 and 3.5, each with:

- Pre-expiry exit only (recomputed control with the same series handling).
- Close **all legs** when EOD SPX is at or below the short-put strike fixed at entry.
- The same barrier plus a whole-package profit target of 5% of initial terminal-loss budget.
- The same barrier plus a 10% profit target.

The two profit targets are research sensitivity settings, not optimized or
user-specified thresholds. The barrier can close a losing package. Profit targets
use executable net liquidation PnL after costs, not the long's profit in isolation.
All legs close together at that session's bid/ask snapshot. This is an EOD
execution assumption, not a simulation of an intraday barrier fill; late signals,
gaps and actual execution latency can produce different results. A signal with
missing quotes stays pending until a valid pre-expiry quote for every leg is
available, even if the price subsequently recovers. There is no stale-price fill.

The standard run keeps USD 100,000 capital and 5%/20% risk limits. The separately
labelled USD 250,000 diagnostic keeps those percentages and all construction
rules unchanged. Its purpose is to distinguish financing-unit constraints from
the economic effect of moving the spread farther OTM; it is not a recommendation
to increase capital or risk. Zero trades means no active performance sample.

### Monthly series convention

The old third-Friday-only filter excluded Thursday-labelled monthly chains; that
was not evidence that the source lacked those chains. The new runner uses
`pandas_market_calendars`' `CBOE_Index_Options` sessions to identify monthly AM
settlement and the preceding trading session, including holiday shifts.

Last-trading-day, settlement-day and historical Saturday labels are associated
with a modelled monthly settlement date. Actual source labels are retained:

- New entries require a single source label for the modelled monthly expiry.
- Existing positions first use their current exact source label.
- A label switch is allowed only if that label disappears, exactly one alias
  remains, and no simultaneous-label conflict has been observed for that expiry.
- Simultaneous labels are never combined across legs. Once a conflict has been
  observed, fallback switching is disabled for that expiry. Existing exact-label
  marks remain usable.

This is an explicit **research matching assumption**, not proof of contract
identity. The input lacks SPX/SPXW AM/PM roots; those identifiers are needed for
definitive validation. Decisions use only conflicts observed through the current
session, not future quote availability. Every source-to-model mapping is exported
to `expiry_mapping.csv`, with switch counts on trades and in the summary.
See [Cboe AM settlement](https://cdn.cboe.com/resources/spx/Settlement_of_Standard_AM_Settled_SP_500_Index_Options.pdf)
and the [calendar documentation](https://pandas-market-calendars.readthedocs.io/en/latest/usage.html).

### Comparison outputs

Results: `results/digital_tail_barrier_comparison/`, with a separate `_250k` run.
Both include the usual summary, trades, daily equity, entry audit, manifest and
offline report, plus `package_marks.parquet`, `expiry_mapping.csv` and
`paired_exits.csv`.

The dashboard presets **Digital Tail / Porownanie barier** and **Digital Tail /
Bariery 250k** provide sigma/exit cross-sections, daily drawdown, risk usage,
per-package underlying/barrier and liquidation-PnL paths, and quote diagnostics.
Different exits can change available capital and later entries. The paired-entry
tab therefore compares identical entry dates, expiries, source labels, strikes
and spread counts against each sigma's pre-expiry control. These matched package
sums are not an independently simulated portfolio or a common-entry sigma test.

### Completed comparison (2010-2023)

At USD 100,000, only the 2.5-sigma construction traded (12 packages). The
recomputed pre-expiry control lost USD 7,064 with a USD 7,610 marked drawdown;
barrier exit lost USD 1,514 with USD 1,828 drawdown. Neither profit target fired.
The 3.0/3.5-sigma variants had no entries under the USD 5,000 initial package cap.

At USD 250,000, the barrier-only variants closed 55, 35 and 17 packages for
2.5, 3.0 and 3.5 sigma respectively. Net PnL: -USD 3,143, -USD 6,043 and +USD 1,329;
marked drawdowns: USD 6,267, USD 9,928.50 and USD 3,003. The 3.5-sigma 5% profit
target reduced PnL to -USD 2,576, while its 10% target did not fire. These are small
and differently selected samples, not evidence of a stable edge.

All three USD 250,000 pre-expiry controls have a June 2020 package without a
confirmed terminal fill. Their reported PnL includes carried marks and new entries
are blocked while those positions remain stale. They are **provisional controls**,
not valid complete-period benchmarks. The paired-entry view excludes unresolved
positions on either side, potentially excluding important tail events. All
barrier/profit variants in both runs have zero stale-mark sessions and no
unresolved positions. Original baseline files were not overwritten.
