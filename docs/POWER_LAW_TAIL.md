# Power-Law Tail: SPX research and ES/MES monitor

## Model

Reference: Taleb et al., [Tail Option Pricing Under Power Laws](https://arxiv.org/abs/1908.02347), section III.
This is relative pricing, conditional on an anchor and a tail exponent, not an estimate of physical expected returns.

For `k=K/S0`, integrating the section III downside density on nonnegative prices gives:

```text
g(k, alpha) = (1-k)^(1-alpha) - 1 - (alpha-1)*k
P(K) = P(anchor) * g(K/S0, alpha) / g(anchor/S0, alpha)
0 <= K <= anchor < S0, alpha > 1
```

This real, dimensionally consistent integral form avoids the negative fractional powers and ambiguous dimensional terms in the printed equation (7).
Tests compare it with numerical integration. The anchor normalization cancels the density constant and common discount factor.
Do not apply the call power-law formula to puts, or extrapolate towards ATM.

Baseline alpha is 2.75; anchor is the closest eligible strike to 2.5 sigma, tolerance 0.25 sigma.
Sigma coordinates are `-log(K/S0)/(ATM put IV * sqrt(calendar DTE/365))`.
They do not assert a normal distribution or demonstrate that the Pareto regime has begun.
Sensitivity: alpha 2, 2.75, 3.5 crossed with anchors 2, 2.5, 3 sigma.
No parameters are fitted to subsequent profits. All nine models must price a strike for a robustness claim.

Quotes require finite positive bid, ask >= bid, spread/mid <= 50%, provider IV in [3%,300%], valid negative delta, and unique strikes.
ATM additionally requires put delta in [-0.65,-0.35]. Missing anchors are rejected, not interpolated.
Anchor price is bid; candidate cost is ask plus round-trip assumed commission.
A signal requires model/cost >= 1.10. The anchor slope is checked against the zero-rate put slope bound and the next higher quoted put.
These checks are not a complete arbitrage-free calibration of the entire smile and do not prove liquidity.

## Historical experiment

```bash
docker compose up -d dashboard
docker compose exec -T dashboard python -m taleb_kurtosis.power_law_backtest
docker compose exec -T dashboard python -m unittest tests.test_power_law tests.test_power_law_live
```

Output: `results/power_law_tail`; dashboard: <http://localhost:8501/?view=power-law>.
`--limit-files 2 --output results/power_law_smoke` produces a short diagnostic run.

- Scan all observed eligible monthly SPX chains at 60-180 DTE. Source data are 2010-2023, not including 2008.
- Reuse the documented monthly expiry alias convention from `DIGITAL_TAIL.md`; simultaneous ambiguous roots are excluded.
- Descriptive frequency is averaged equally across observed sessions within each sigma/DTE bin. It is conditional on quote/anchor availability; coverage counts are provided.
- Four independent long-only portfolios: model-selected vs delta ~1% (+/-0.25 percentage points), each with 30 DTE exit or a 5x-premium TP plus 30 DTE exit.
- The delta control uses the same eligible anchored chains, not an unconditional control.
- Capital USD 250,000; monthly premium budget 0.25% of initial capital (USD 625), including both commissions; integer SPX contracts x100; no budget carry-forward.
- First eligible monthly EOD signal; choose expiry closest to 120 DTE within 105-135; model selection maximizes model/cost - 1.
- Execute no earlier than the next observed session at ask, after checking execution quote quality and affordability. A failed order may generate a new signal, never a retroactive replacement.
- Sell at bid when the known 30 DTE date is reached. TP is bid >= 5 * entry ask, followed by execution on a later session; a gap can erase the indicated profit.
- No financing shorts, stock holdings, cash interest, or additional slippage/market impact. The tail sleeve's standalone losses do not measure its benefit to an equity portfolio.
- Daily equity uses liquidation bid minus exit commission. Missing quotes are carried and flagged, not executed; new entries pause while active positions have stale marks.
- Remaining positions close at the final observed bid if available, explicitly labelled end-of-data. Otherwise unresolved; no invented SET, expiry settlement, or future last-quote lookup.

Artifacts include per-month scanned quotes, frequency coverage, alpha/anchor sensitivity, signal/execution audit, trades, daily equity, sample chains, and a self-contained HTML report.

## Completed baseline: 2010-01-04 through 2023-12-29

3499 observed sessions; no unresolved positions or stale portfolio marks in the full run.

| Selection / exit | Closed trades | Net PnL USD | Max drawdown USD |
| --- | ---: | ---: | ---: |
| Model / 30 DTE | 124 | -65,587 | -150,043 |
| Model / TP 5x or 30 DTE | 124 | -37,377 | -44,335 |
| Delta 1% / 30 DTE | 154 | -45,053 | -76,272 |
| Delta 1% / TP 5x or 30 DTE | 154 | -49,143 | -54,675 |

Total premium plus entry commissions: USD 70,451 for model selection, USD 78,107 for the delta control.
The model selected a median 6.59 sigma put with delta -0.004715; the delta control median was 5.57 sigma.
Model/30 DTE equity peaked at USD 334,456 on 2020-03-18 and finished at USD 184,413.
Drawdown measures decline from the running equity peak, including unrealized profits, not just loss from initial capital.
The large peak-to-trough loss illustrates profit giveback; it is not evidence of a short-option liability.

For 105-135 DTE, average daily frequency of the baseline relative-value signal:

| Sigma bin | Observed sessions | Baseline frequency | All nine models agree |
| --- | ---: | ---: | ---: |
| 5-6 | 2213 | 35.65% | 3.43% |
| 6-8 | 1955 | 42.74% | 7.01% |
| 8+ | 1554 | 40.42% | 5.27% |

These are conditional frequencies among eligible quotes, not crash probabilities or trade win rates.
Across all scanned tenors, 8+ sigma at 60-89 DTE has the largest baseline frequency (51.53%, 2032 observed sessions); it is not the four-month strategy.
Results do not establish absolute undervaluation. In this experiment, buying the highest relative-value score was not profitable standalone.
Exits matter: 5x monetization improved the model-selected portfolio but worsened the delta control, so it is not universally preferable.

## Matched-entry scaleout experiment

```bash
docker compose exec -T dashboard python -m taleb_kurtosis.power_law_scaleout
docker compose exec -T dashboard python -m unittest tests.test_power_law_scaleout
```

Output: `results/power_law_scaleout`; dashboard: <http://localhost:8501/?view=power-law-scaleout>.
This replays the exact original model and delta-control entry tickets, prices, quantities, and costs.
The 30 DTE and TP 5x controls must reconcile with the original run before results are published.
No entry selection or sizing is reoptimized based on these outcomes.

- Freeze sigma at purchase: `entry ATM IV * sqrt(entry calendar DTE / 365)`.
- ATM is the nearest put strike of the same source expiry. Invalid IV/delta/quotes are rejected.
  If entry-day ATM data are invalid, use the most recent valid observation for that exact source expiry,
  at most five calendar days old. Record its date, strike, IV, and carry flag; fail if no eligible observation exists.
  This handles corrupt vendor greeks without inventing volatility or looking forward.
- Trigger when `log(entry SPX / current SPX) / frozen sigma` first reaches 3, 4, 5, and each subsequent integer.
  These are cumulative moves over the option horizon, **not** 3-sigma daily returns or distance to the option strike.
- Sell half of what remains at each new level. There is no extra profit condition.
- EOD signal, next available session bid, USD 1 commission per contract per side.
  No intraday barrier fill or backdated execution. A gap queues all crossed levels; a rebound does not cancel a queued sale.
  Missing option quotes delay execution but do not erase an observed underlying trigger.
- Executable quantity convention: floor half to whole contracts, retain at least one. An initial single contract cannot scale out.
  Fractional control implements exact 50%, 25%, 12.5%, etc. of original size with proportional fees; it is theoretical, not executable SPX.
- Residual still exits at 30 DTE, or the end of data with an available quote, **not at expiry**.
  Scheduled liquidation takes precedence over queued partial sales. Missing final quotes remain unresolved.
- Original budget is fixed; partial proceeds do not create additional entries. No financing leg or cash interest is added.

Artifacts include all sale fills with allocated entry cost/PnL, trigger timestamps and barriers,
entry volatility provenance, daily package paths/remaining contracts, equity, drawdown, and an HTML report.
The manifest records the baseline trade-file SHA256 and control reconciliation.

### Scaleout results: 2010-2023

All eight portfolios have 3499 daily observations, zero unresolved positions, and zero stale liquidation marks.
Both controls for both selections also reconcile to the original **daily** equity, not only final PnL.
The same 124 model entry tickets and 154 delta entry tickets are used in every exit variant.

| Selection / exit | Net PnL USD | Max drawdown USD | Packages with sigma sales |
| --- | ---: | ---: | ---: |
| Model / 30 DTE | -65,587 | -150,043 | 0 |
| Model / TP 5x | -37,377 | -44,335 | 0 |
| Model / scaleout whole contracts | -42,320 | -57,772 | 3 |
| Model / scaleout fractional, theoretical | -40,742.50 | -57,772 | 3 |
| Delta 1% / 30 DTE | -45,053 | -76,272 | 0 |
| Delta 1% / TP 5x | -49,143 | -54,675 | 0 |
| Delta 1% / scaleout whole contracts | -46,374 | -57,377 | 2 |
| Delta 1% / scaleout fractional, theoretical | -43,252 | -57,377 | 2 |

Model whole-contract scaleout improved PnL by USD 23,267 and reduced absolute maximum drawdown by 61.5%
relative to 30 DTE, but remained loss-making. TP 5x had better PnL and lower drawdown for model selection in this sample.
All sigma signals occurred in March 2020: seven levels across three model packages and five across two delta packages.
Whole-contract rounding allowed only two partial sales in the delta control; each triggered package started with two contracts.
The model's median entry-frozen 3-sigma barrier is a 21.73% decline from entry SPX, which helps explain the low trigger count.
This is one crash episode, not independent evidence across many crashes. The sample excludes 2008.

There were two carried ATM observations, one per selection on 2019-05-02, both using 2019-05-01 IV.
The entry-day provider ATM IV was approximately 0.00023 with an implausible ATM delta; neither affected a sigma-triggered package.
Sale-level PnL, cash, quantities, and next-session timing all reconcile. Results assume quoted bids are executable at the model size;
there is no volume/depth or additional crisis slippage model. This experiment does not establish positive standalone expected returns.

## Sigma / DTE valuation map with a 3-sigma anchor

Reference specification: `data/wycena_opcji_metoda_taleba.md`.
This is a historical cross-sectional valuation study, **not** a realized-return trading backtest.

```bash
docker compose run --rm -T --no-deps dashboard python -m taleb_kurtosis.power_law_heatmap --workers 4
docker compose run --rm -T --no-deps dashboard python -m unittest tests.test_power_law_heatmap
```

Output: `results/power_law_dte_heatmap_anchor3/report.html`, with standalone PNG heatmaps and CSV tables.
For a matched-sample midpoint comparison, run:

```bash
docker compose run --rm -T --no-deps dashboard python -m taleb_kurtosis.power_law_heatmap --workers 4 --price-basis mid --output results/power_law_dte_heatmap_anchor3_mid
```

The `mid` basis reprices both the 3-sigma anchor and evaluated puts at `(bid + ask) / 2`.
It keeps the original USD 1 per contract per side commission and recomputes all discounts,
frequencies, edges, annual statistics and heatmaps from individual quotes. Quote filters,
anchor selection and the original bid-based slope/splice eligibility checks are unchanged,
so the two variants use the same sample. `median_mid_discount` separately reports gross
mid-to-mid discount without commissions in both variants. Mid prices are hypothetical fills,
not evidence that trades could execute there. Use a separate output directory to preserve the
original conservative results. The default remains `bid_ask`.

Use `--limit-files 2 --min-days 2 --min-years 1 --output results/heatmap_smoke` for a short run.
`--resume` reuses completed monthly files only if source metadata, model/scanner hashes, and parameters match.
`--alpha` changes the primary exponent; the default is 3, with additional fixed 2.5 and 3.5 sensitivity scans.
This intentionally differs from the earlier alpha-2.75 / 2.5-sigma-anchor strategy baseline.

- Scan all available source expiries from one calendar DTE to the longest observed expiry; exclude 0 DTE.
  Do not apply monthly aliases or infer settlement conventions. Never mix dates or expiries when pricing a chain.
- Sigma coordinate: `log(S/K) / (ATM_IV * sqrt(calendar_DTE/365))`, with no drift/rate/dividend correction.
  ATM IV is the nearest-strike put IV of that date and source expiry, with the existing ATM quality checks.
  Invalid ATM observations reject the chain: no carry-forward, interpolation, or forward-looking fill.
- Anchor: nearest eligible quote to 3 sigma, tolerance 0.25 sigma. Actual strike, sigma, bid/ask, and ATM IV are audited.
  Only strikes below the anchor and at least 3 sigma away are scored.
- Primary score: `discount = 1 - (ask + 2*fee/multiplier) / bid_anchored_model`.
  Positive is green, negative red. A 30% discount means acquisition cost is 30% below this relative model, not a 30% return.
  Costs are USD 1 per contract per side, multiplier 100. Mid-anchored/mid-priced discount is saved separately.
- First take the median discount across strikes in each chain/sigma bin, then equal-weight chain medians within each
  date/DTE/sigma bin, then take the median across observed dates. Dense chains and high-activity dates do not dominate.
- Frequency is the equal-date average of equal-chain proportions with positive discount; a separate column uses
  the earlier `model/cost - 1 >= 10%` threshold. Neither is a win rate.
- Grey masks cells with fewer than 100 observed sessions or coverage in fewer than three calendar years.
  Missing data are never treated as zero discount. The coverage map and CSV retain sparse cells.
- The color scale saturates at -100% and +100%; numeric annotations retain actual scores.
  Recurring contracts make sessions dependent, and alpha variants can have different eligible universes.

Artifacts: monthly `chain_bins/` and `chain_audit/`, `daily_bins.parquet`, `summary.csv`, `yearly.csv`,
`ranked_cells.csv`, `audit_counts.csv`, `manifest.json`, and the HTML/PNG heatmaps.
Quoted-spread screening and local anchor slope checks do not establish actual liquidity or full absence of arbitrage.

## Monthly 6-8 sigma midpoint strategy

```bash
docker compose run --build --rm -T --no-deps dashboard python -m taleb_kurtosis.power_law_monthly
docker compose run --rm -T --no-deps dashboard python -m unittest tests.test_power_law_monthly
```

Output: `results/power_law_monthly_6_8sigma_mid/report.html`. Existing outputs are never overwritten;
use `--output` for an independent run, `--limit-files 2` for a smoke test, and `--budget` to change
the default USD 1000 monthly premium-and-commission budget.

- Alpha 3, anchor 3 sigma (+/-0.25), same-chain/date ATM put IV and the heatmap's quote screens.
  Anchor and target are both priced at mid. Positive discount means model exceeds mid plus both commissions;
  there is no additional 10% minimum edge.
- Search all observed source expiries at 165-195 calendar DTE and all eligible puts at 6-8 sigma inclusive.
  Rank by largest positive discount, then tighter spread, proximity to 180 DTE, expiry and strike.
- Search the first exchange session of each month and the following five sessions. Missing data consume
  search opportunities. At most one purchase per month; a sixth-session signal may fill on session seven.
- Freeze the selected contract and its model-implied price limit at signal close. Buy at the next session's
  EOD mid only if below that limit after reserving round-trip commissions. Otherwise cancel and search again
  only if the six-session window is still open. No retrospective substitution with another strike.
- Full SPX contracts (x100), fee USD 1 per contract per side, initial cash USD 250000. Reserve entry plus
  exit fees within the fixed USD 1000 budget. Positions from different months overlap. Both exit policies
  receive identical entries and sizes, subject to available cash in both books.
- Control: exit at the first valid quote at or below 30 calendar DTE. Experimental exit: additionally close
  after a one-session downside log return greater than `5 * prior_session_same_expiry_ATM_IV / sqrt(252)`.
  Only the negative direction triggers. IV is lagged, never taken from the crash close itself or carried
  across missing sessions. Source underlying medians proxy the SPX closing level.
- A crash signal is known at EOD and sells at a later session's mid, not at the trigger day's close.
  Missing quotes delay the persistent exit instruction. Missing marks are carried and flagged; expired
  unresolved positions are never assigned fabricated settlement values. End-of-data exits are separate.
- Calendar: existing `pandas_market_calendars` CBOE_Index_Options calendar. No monthly expiry aliases.
  Input-source sessions outside that calendar and missing exchange sessions are counted in the manifest.
- Reports include matched equity/drawdown curves, annual MTM PnL, trades, entry decisions, crash signals,
  position marks, and source metadata. This is an in-sample experiment selected from the same historical
  heatmap. Mid fills, especially on wide-spread exits and large quantities, are hypothetical; no depth,
  market impact, cash interest or financing positions are modelled.

## ES/MES monitor

<http://localhost:8501/?view=power-law-live>

The implemented network adapter is **CME data via the existing IBKR/TWS dependency**, read-only. Direct CME OAuth/WebSocket integration is not implemented, and no paid subscription is purchased automatically.
TWS or Gateway plus existing CME market-data permissions are required. A new client ID is used with no account fetches and no order methods.
Choose an actual futures contract, discover its option trading classes/expiries, verify European exercise in the CME specification, then request a bid/ask snapshot. No continuous-future or SPY/BTC substitution.
Unknown/American exercise, frozen/delayed data, stale observations (>120 seconds), duplicate strikes, and wrong multipliers block ranking.
IBKR snapshot timestamps are receipt times, not a guaranteed exchange timestamp. Exercise style is user-verified because IBKR ContractDetails does not provide it here.
Manual refresh acquires quotes; a 15-second UI timer rechecks freshness, not an automatic market-data subscription.

European futures adaptation uses the price of the exact matching underlying future in place of SPX spot; this has not been validated by the SPX backtest.
ES multiplier is USD 50 per point, MES USD 5, not a fivefold ES contract.
There is no verified Yahoo/Deribit ES/MES bid/ask-chain adapter. Yahoo's tested endpoint returned HTTP 429. These sources are not silently substituted.

Optional CSV snapshots use these columns (prices in index points, IV as decimal, timezone-aware timestamps):

```text
symbol,option_series,expiry,underlying_contract,underlying_price,underlying_quote_time,quote_time,strike,bid,ask,iv,delta,multiplier,exercise_style,market_data_type
```

`expiry` is ISO YYYY-MM-DD, `exercise_style` must be European, `market_data_type=1` means real-time in the IBKR convention. Imported data remain explicitly labelled snapshots; freshness cannot establish authenticity.

Product caveat: CME [SER 9760](https://www.cmegroup.com/content/dam/cmegroup/notices/ser/2026/06/ser-9760.pdf) stops new listings of the old physically settled Micro option families and transitions towards financial options. The futures-option adapter does not claim support for replacement cash-index option families. Contract identifiers must be discovered, not inferred from "X5".

Official data access: [CME market-data API](https://www.cmegroup.com/market-data/real-time-futures-and-options-data-api.html), [IBKR API](https://www.interactivebrokers.com/docs/tws-api/doc/introduction).
