# IBKR Paper Testing

This project starts IBKR integration in signal-only mode. It connects to a paper TWS or IB Gateway session, builds the current SPX tail-ratio put candidate, writes a JSON ticket, and does not transmit orders.

## TWS / IB Gateway Setup

1. Log in to the IBKR paper account in Trader Workstation or IB Gateway.
2. In TWS, enable API access: `Global Configuration -> API -> Settings -> Enable ActiveX and Socket Clients`.
3. Use the paper socket port. IBKR defaults are commonly:
   - TWS paper: `7497`
   - IB Gateway paper: `4002`
4. Keep `Read-Only API` enabled for signal generation. Disable it only when we intentionally add order placement later.
5. Confirm you have market data permissions for SPX and SPX options in the paper session.

IBKR notes that TWS/IB Gateway must be running for the TWS API, paper trading can differ from live execution, and paper TWS defaults to port `7497`.

## Read-only access without a live subscription

The Power Law monitor is separate from the paper signal generator:
<http://localhost:8501/?view=power-law-live>.

Do not send an IBKR username, password, 2FA code, or account secret. Log in locally in TWS/Gateway.
Connection setup needs only:

- TWS or IB Gateway, paper or live session;
- the machine/OS running it and the API host reachable from Docker;
- the configured API socket port and an unused client ID;
- enabled socket clients and **Read-Only API** left enabled.

A subscription is not needed merely to establish a socket connection; quote entitlements are separate.
IBKR supports `reqMarketDataType(3)` for delayed data where the exchange/account supports it.
The [IBKR delayed-data table](https://portal.interactivebrokers.com/en/pricing/market-data-pricing.php?menu=B)
lists CME at ten minutes. This does not guarantee a complete ES/MES option chain or all IV/Greek fields for this account.
Request the actual contract and inspect returned market-data type and missing fields before using any quotes.
Delayed streaming does not remove [historical-data subscription requirements](https://interactivebrokers.github.io/tws-api/delayed_data.html).

The current Power Law adapter requests live snapshots and rejects delayed/frozen observations and stale timestamps.
It has **not** been changed to rank delayed quotes. A future indicative/delayed mode must be explicitly labelled,
align underlying/option observation times, and must not present delayed receipt time as live exchange freshness.
No market-data package is purchased automatically; no orders are transmitted.

Official references: [API data types](https://www.interactivebrokers.com/campus/trading-lessons/python-receiving-market-data/),
[TWS API setup](https://www.interactivebrokers.com/campus/trading-lessons/installing-configuring-tws-for-the-api/).

## WSL / Docker On Windows

Addresses and subnet rules below are examples from a previous session, not current network discovery.
Verify the actual host route and scope any firewall exception to the trusted local client; do not expose the API publicly.

If TWS runs on Windows and the strategy runs from WSL/Docker, `127.0.0.1` inside WSL is not the Windows host. Use the Windows WSL gateway address from:

```bash
awk '/nameserver/{print $2; exit}' /etc/resolv.conf
```

In this environment it is:

```text
172.22.32.1
```

First verify that Windows itself can reach TWS:

```powershell
Test-NetConnection -ComputerName 127.0.0.1 -Port 7497
```

Then verify from WSL:

```bash
python3 - <<'PY'
import socket
s = socket.socket()
s.settimeout(3)
s.connect(("172.22.32.1", 7497))
print("OPEN")
PY
```

If Windows succeeds but WSL times out, add a Windows Firewall inbound rule from an elevated PowerShell:

```powershell
New-NetFirewallRule `
  -DisplayName "TWS API 7497 from WSL" `
  -Direction Inbound `
  -Action Allow `
  -Protocol TCP `
  -LocalPort 7497 `
  -InterfaceAlias "vEthernet (WSL)" `
  -RemoteAddress 172.22.32.0/20
```

Also add the current WSL IP to TWS Trusted IPs:

```bash
hostname -I
```

For the current session this was:

```text
172.22.43.195
```

## Generate A Signal

```bash
python -m taleb_kurtosis.ibkr_paper \
  --host 127.0.0.1 \
  --port 7497 \
  --client-id 71
```

From Docker on Windows/WSL, use the WSL gateway:

```bash
docker compose run --rm analytics python -m taleb_kurtosis.ibkr_paper \
  --host 172.22.32.1 \
  --port 7497 \
  --client-id 71
```

The output JSON contains:

- selected expiry near 60 DTE,
- realized volatility from 252 daily SPX closes,
- long put near 5-sigma 60-day downside,
- short put near +0.4-sigma,
- 20:1 ratio,
- mid-price funding check.

Signals are saved under:

```text
results/ibkr_paper_signals/
```

## Safety

The module intentionally does not place orders. First validate several generated tickets manually in TWS paper, especially contract IDs, trading class, expiry, strike, bid/ask, and whether the combo is net credit.
