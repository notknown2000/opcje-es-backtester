"""Read-only CME/IBKR snapshots. No orders, synthetic chains, or proxy tickers."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from taleb_kurtosis.power_law import PricingConfig, score_chain


class DataUnavailable(ValueError):
    pass


def validate_snapshot(frame, symbol, now=None, max_age_seconds=120):
    required = {"symbol", "option_series", "expiry", "underlying_contract", "underlying_price", "underlying_quote_time",
                "quote_time", "strike", "bid", "ask", "iv", "delta", "multiplier", "exercise_style", "market_data_type"}
    if not required.issubset(frame.columns):
        raise DataUnavailable("Missing fields: " + ", ".join(sorted(required - set(frame.columns))))
    if frame.empty or symbol not in {"ES", "MES"} or not frame.symbol.eq(symbol).all():
        raise DataUnavailable("Snapshot must contain only the selected ES or MES product")
    q = frame.copy()
    expected = 50 if symbol == "ES" else 5
    for column in ["strike", "bid", "ask", "iv", "delta", "underlying_price", "multiplier", "market_data_type"]:
        q[column] = pd.to_numeric(q[column], errors="coerce")
    if not q.multiplier.eq(expected).all():
        raise DataUnavailable(f"Expected {expected} USD per index point; prices must be in index points")
    if not q.exercise_style.eq("European").all():
        raise DataUnavailable("European exercise required; unknown/American series are not comparable to this model")
    if not q.market_data_type.eq(1).all():
        raise DataUnavailable("Delayed, frozen or unknown market data cannot be ranked as live executable prices")
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if now.tzinfo is None:
        raise DataUnavailable("UTC-aware current time required")
    for column in ["quote_time", "underlying_quote_time"]:
        q[column] = pd.to_datetime(q[column], utc=True, errors="coerce")
        age = (now - q[column]).dt.total_seconds()
        if age.isna().any() or age.gt(max_age_seconds).any() or age.lt(-5).any():
            raise DataUnavailable("Stale, future-dated or missing option/underlying timestamps")
    q["expiry"] = pd.to_datetime(q.expiry, errors="coerce").dt.normalize()
    if q.expiry.isna().any() or (q.expiry <= now.tz_convert(None).normalize()).any():
        raise DataUnavailable("Expired or invalid expiry")
    if q.underlying_contract.isna().any() or q.underlying_contract.astype(str).str.strip().eq("").any():
        raise DataUnavailable("Exact underlying contract required; continuous futures are not sufficient")
    if q.option_series.isna().any() or q.option_series.astype(str).str.strip().eq("").any():
        raise DataUnavailable("Exact option trading class / series required")
    if q.duplicated(["option_series", "expiry", "underlying_contract", "strike"]).any():
        raise DataUnavailable("Duplicate strikes / mixed option series")
    if not np.isfinite(q.underlying_price).all() or not q.underlying_price.gt(0).all():
        raise DataUnavailable("Missing matching underlying futures quotes")
    return q


def rank_snapshot(frame, symbol, alpha=2.75, anchor_sigma=2.5, fee=1., now=None):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    q = validate_snapshot(frame, symbol, now)
    rows, audit = [], []
    for (series, expiry, underlying), chain in q.groupby(["option_series", "expiry", "underlying_contract"]):
        reference = float(chain.underlying_price.median())
        if (chain.underlying_price.max() - chain.underlying_price.min()) / reference > .001:
            raise DataUnavailable("Underlying quotes are not synchronized within the chain")
        dte = (expiry - now.tz_convert(None).normalize()).days
        normalized = chain.rename(columns=dict(strike="STRIKE", bid="P_BID", ask="P_ASK", iv="P_IV", delta="P_DELTA"))
        cfg = PricingConfig(alpha=alpha, anchor_sigma=anchor_sigma, fee=fee, multiplier=float(chain.multiplier.iloc[0]))
        ranked, reason = score_chain(normalized, reference, dte, cfg)
        audit.append(dict(option_series=series, expiry=expiry, underlying=underlying, rows=len(chain), eligible=len(ranked), reason=reason))
        if not ranked.empty:
            ranked["cost_usd"] = ranked.all_in_ask * cfg.multiplier
            rows.append(ranked)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), pd.DataFrame(audit)


def connect(host, port, client_id):
    from ib_async import IB, StartupFetchNONE

    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    ib = IB()
    ib.RequestTimeout = 20
    try:
        ib.connect(host, port, clientId=client_id, timeout=5, readonly=True, fetchFields=StartupFetchNONE)
        ib.reqMarketDataType(1)
        return ib
    except Exception:
        ib.disconnect()
        raise DataUnavailable("No read-only TWS/Gateway connection. Verify host, port and API access.") from None


def discover_futures(symbol, host="host.docker.internal", port=7497, client_id=81):
    from ib_async import Future

    if symbol not in {"ES", "MES"}:
        raise DataUnavailable("Only ES and MES supported")
    ib = connect(host, port, client_id)
    try:
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        details = ib.reqContractDetails(Future(symbol=symbol, exchange="CME", currency="USD"))
        return [dict(con_id=d.contract.conId, contract=d.contract.localSymbol,
                     expiry=d.contract.lastTradeDateOrContractMonth)
                for d in details if d.contract.lastTradeDateOrContractMonth >= today]
    finally:
        ib.disconnect()


def discover_chains(symbol, future_id, host="host.docker.internal", port=7497, client_id=81):
    ib = connect(host, port, client_id)
    try:
        chains = ib.reqSecDefOptParams(symbol, "CME", "FUT", future_id)
        return [dict(trading_class=c.tradingClass, multiplier=c.multiplier, expiry=expiry)
                for c in chains if c.exchange == "CME" for expiry in sorted(c.expirations)
                if expiry > datetime.now(timezone.utc).strftime("%Y%m%d")]
    finally:
        ib.disconnect()


def fetch_chain(symbol, future_id, trading_class, expiry, european_confirmed=False,
                host="host.docker.internal", port=7497, client_id=81):
    from ib_async import Contract, FuturesOption

    if not european_confirmed:
        raise DataUnavailable("Exercise style must be verified for this exact series in the CME contract specification")
    ib = connect(host, port, client_id)
    try:
        future = ib.qualifyContracts(Contract(conId=future_id, exchange="CME"))[0]
        if future.symbol != symbol or future.secType != "FUT":
            raise DataUnavailable("Underlying is not the requested futures product")
        ft = ib.reqTickers(future)[0]
        if ft.marketDataType != 1 or not (np.isfinite(ft.bid) and np.isfinite(ft.ask) and 0 < ft.bid <= ft.ask):
            raise DataUnavailable("No real-time two-sided quote for the matching future")
        reference = (ft.bid + ft.ask) / 2
        details = ib.reqContractDetails(FuturesOption(symbol=symbol, lastTradeDateOrContractMonth=expiry,
                                                     right="P", exchange="CME", currency="USD", tradingClass=trading_class))
        contracts = [d.contract for d in details if d.underConId == future_id and .1 * reference <= d.contract.strike <= 1.01 * reference]
        if not contracts:
            raise DataUnavailable("No puts with the requested expiry, trading class and underlying")
        if len(contracts) > 300:
            raise DataUnavailable("More than 300 strikes: select a narrower option series before requesting snapshots")
        tickers = []
        for start in range(0, len(contracts), 30):
            tickers.extend(ib.reqTickers(*contracts[start:start+30]))
        rows = []
        for ticker in tickers:
            greeks = ticker.modelGreeks
            rows.append(dict(symbol=symbol, option_series=trading_class, expiry=expiry, underlying_contract=future.localSymbol,
                             underlying_price=reference, underlying_quote_time=ft.time, quote_time=ticker.time,
                             strike=ticker.contract.strike, bid=ticker.bid, ask=ticker.ask,
                             iv=greeks.impliedVol if greeks else np.nan, delta=greeks.delta if greeks else np.nan,
                             multiplier=float(ticker.contract.multiplier), exercise_style="European",
                             market_data_type=ticker.marketDataType, contract=ticker.contract.localSymbol))
        return pd.DataFrame(rows)
    finally:
        ib.disconnect()
