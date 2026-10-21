# SPX / ES Options Research

Python research tools for tail-option strategies, Taleb power-law relative
valuation, historical SPX backtests and a standalone delayed SPX monitor.

## Standalone SPX monitor

Requires Docker with Docker Compose and internet access:

```bash
docker compose up -d --build spx-monitor
```

Open <http://localhost:8502>. No historical dataset or API key is required.
The monitor reads Cboe's delayed SPX/SPXW feed and defaults to 165-195 DTE,
same-chain ATM IV, a 3-sigma anchor, alpha 3 and mid-to-mid valuation with fees.
It includes a valuation heatmap, contract ranking, price curves, IV smile,
alpha sensitivity, quote-quality diagnostics and CSV export.

Quotes are delayed approximately 15 minutes. The shared API cache refreshes
every five minutes while the page is open; there is no unattended collector.
The feed is not a guaranteed real-time API. Green/red indicates prices below/above
the assumed relative model, not guaranteed mispricing or executable trading signals.

To use a different port:

```bash
SPX_MONITOR_PORT=8503 docker compose up -d spx-monitor
```

See [SPX monitor documentation](docs/SPX_MONITOR.md) for assumptions and limitations.

## Historical research

The repository includes:

- Weekly tail-ratio and kurtosis experiments, including scaleout exits.
- Tail financing through put credit spreads and barrier exits.
- Power-law option selection, sigma/DTE valuation maps and matched-entry exit tests.
- Monthly 6-8 sigma long-put strategy and crash-triggered monetization.
- Read-only IBKR/TWS integrations and paper signal generation.

Historical source files, Parquet data, generated reports and database contents
are intentionally excluded from Git. Obtain data under your own license.
The historical dashboard requires locally generated outputs in `results/`.

Convert local SPX EOD text files and run an example valuation study:

```bash
docker compose run --rm -T --no-deps analytics python convert_spx_to_parquet.py
docker compose run --rm -T --no-deps analytics python -m taleb_kurtosis.power_law_heatmap --price-basis mid --output results/power_law_dte_heatmap_anchor3_mid
docker compose up -d --build dashboard
```

Open <http://localhost:8501> for the historical dashboard. Data schema and experiment
commands are documented in [Power Law](docs/POWER_LAW_TAIL.md),
[Digital Tail](docs/DIGITAL_TAIL.md), and [IBKR Paper](docs/IBKR_PAPER.md).
The [methodology note](data/wycena_opcji_metoda_taleba.md) is included without datasets.

## Tests

Run the monitor and shared pricing regression tests:

```bash
docker compose run --build --rm -T --no-deps analytics python -m unittest tests.test_spx_monitor tests.test_spx_monitor_dashboard tests.test_power_law tests.test_power_law_live tests.test_power_law_heatmap
```

Other tests live in `tests/`. Some historical dashboard checks require local
backtest artifacts and are skipped when those artifacts are absent.
`tests/browser_spx_monitor.py` is an optional live browser check requiring
Playwright/Chromium and a running monitor; set `SPX_MONITOR_URL` as needed.

## Local configuration

The monitor and backtests do not require MariaDB. To use the optional `db` service,
set `MYSQL_ROOT_PASSWORD` in a local `.env` file; `.env.example` lists configuration
variables. Existing MariaDB volumes keep their initialized credentials.
Do not commit `.env`, broker credentials, API keys or `.streamlit/secrets.toml`.

The SPX monitor binds to localhost and does not place orders. IBKR adapters are
read-only/signal-only; connecting TWS does not imply market-data entitlements.

## Research limitations

The model provides relative tail prices conditional on an anchor and an assumed
exponent, not physical crash probabilities or positive expected returns.
Mid fills are hypothetical. Historical tests have different sizing, cost and
execution assumptions; their raw PnL should not be compared without normalization.
Historical results and parameter selection do not establish out-of-sample profitability.
