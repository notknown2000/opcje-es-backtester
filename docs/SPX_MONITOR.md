# SPX Tail Monitor

Separate read-only Streamlit application: <http://localhost:8502>.

```bash
docker compose up -d --build spx-monitor
docker compose run --rm -T --no-deps dashboard python -m unittest tests.test_spx_monitor tests.test_spx_monitor_dashboard
```

Set `SPX_MONITOR_PORT=8503` when 8502 is occupied. The published port binds to
127.0.0.1, not all network interfaces. The repository mount is read-only.
The original dashboard and IBKR adapter are unchanged. No login, API key, order
submission or subscription purchase is required by this Cboe adapter.

## Data

Source: [Cboe delayed SPX quotes](https://www.cboe.com/delayed_quotes/spx/quote_table/),
using its public JSON snapshot at
`https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json` (HTTP redirects followed).
This is a website feed, not a contracted API with a stability or uptime guarantee.
It can change format, throttle requests or become unavailable. No bypass of access
restrictions is implemented. Usage remains subject to the provider's data terms;
this local monitor does not redistribute the feed or provide a public service.

- Delayed approximately 15 minutes, **never labelled real-time or executable**.
- One shared fetch per five minutes while a browser session is open. UI freshness
  is checked every minute when automatic refresh is enabled. No closed-browser worker.
- Manual refresh respects the same five-minute cache. Failures are also cached
  to avoid retry storms; the previous successful session snapshot, if any, is
  explicitly labelled as a failed-refresh fallback, with its original timestamps.
- The top-level naive snapshot timestamp is interpreted as UTC; naive underlying
  `last_trade_time` is interpreted in America/New_York, including DST. These are
  observed feed conventions, not independently verified exchange bid/ask timestamps.
- Snapshot generation time, underlying observation time and retrieval time are separate.
  Individual option last-trade times are **not** used as bid/ask quote times.
  The feed does not certify synchronized, fresh two-sided quotes for every strike.
- Snapshot generation older than 30 minutes or underlying observation older than
  60 minutes blocks the current ranking. An explicit historical-view checkbox can
  display old data, including after the market closes, labelled as historical.
  Future/inconsistent timestamps always block ranking, even in historical mode.
- No historical SPX, synthetic quotes, SPY, futures or crypto are silently substituted.

## Valuation

Defaults: puts at 165-195 calendar DTE, alpha 3, anchor 3 sigma +/-0.25,
mid-to-mid pricing, USD 1 per contract per side, SPX multiplier 100.
DTE is measured from the source underlying observation's New York date, so an
old snapshot is never aged forward while retaining its old prices.

Each root and expiry is independent: SPX and SPXW are not combined. OCC symbols
provide the actual quoted expiry; no monthly-expiry aliases are introduced.
Sigma is `log(S/K) / (ATM_IV * sqrt(DTE/365))`, using the nearest-strike put's IV
of this exact chain and snapshot. The common `power_law.py` bounded downside
kernel is reused, rather than introducing another Taleb formula.

The quote and anchor-slope eligibility checks match the historical mid heatmap:
positive bid, noncrossed bid/ask, finite IV/delta, spread/mid at most 50%,
valid ATM put delta, duplicate-strike exclusion, anchor tolerance and the
original bid-based slope/splice screen. Invalid ATM or absent anchor rejects
the chain, without interpolation. Only puts beyond the anchor and at least
3 sigma are valued. Charts can show other eligible strikes for IV context.

Mid mode anchors the model at anchor mid and compares with target mid plus
round-trip commissions. Bid/ask mode anchors at bid and compares with ask plus
the same commissions. Positive `discount = 1 - cost/model` is green; negative
is red. Gross mid discount and conservative bid/ask discount are separate columns.
This is relative valuation conditional on the anchor and alpha, **not evidence
of absolute undervaluation or positive expected trading returns**.

The heatmap is the median contract discount in each sigma bin, separately for
each expiry/root, not the historical frequency of an opportunity. Empty cells
are unavailable, not zero. Color saturation is +/-50%; values retain full precision.
Alpha 2.5/3/3.5 sensitivity uses the same chain and its eligibility screens;
the robust flag requires positive discounts for all three, without missing models.
CSV exports include the source, underlying and retrieval timestamps, actual
anchor, price basis, fees via all-in cost and the delayed-indicative data label.

General model documentation: [POWER_LAW_TAIL.md](POWER_LAW_TAIL.md).
