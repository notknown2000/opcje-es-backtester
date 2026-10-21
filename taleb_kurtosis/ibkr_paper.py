from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from statistics import pstdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = PROJECT_ROOT / ".deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

from taleb_kurtosis.backtest import sigma_horizon, target_strike


DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "ibkr_paper_signals"


@dataclass(frozen=True)
class OptionLegSignal:
    action: str
    quantity: int
    symbol: str
    expiry: str
    strike: float
    right: str
    exchange: str
    trading_class: str
    con_id: int | None
    bid: float | None
    ask: float | None
    mid: float | None


@dataclass(frozen=True)
class TailRatioSignal:
    generated_at: str
    account_mode: str
    strategy: str
    underlying_symbol: str
    underlying_price: float
    sigma_window: int
    sigma_daily: float
    sigma_horizon: float
    dte: int
    expiry: str
    long_sigma: float
    short_sigma: float
    ratio: int
    target_long_strike: float
    target_short_strike: float
    net_entry_credit_per_spread: float | None
    net_entry_credit_usd: float | None
    funded: bool
    legs: list[OptionLegSignal]
    notes: list[str]


def realized_vol_from_closes(closes: list[float], window: int) -> float:
    if len(closes) <= window:
        raise ValueError(f"Need at least {window + 1} closes, got {len(closes)}.")
    returns = [math.log(curr / prev) for prev, curr in zip(closes, closes[1:]) if prev > 0 and curr > 0]
    if len(returns) < window:
        raise ValueError(f"Need at least {window} returns, got {len(returns)}.")
    sigma = pstdev(returns[-window:])
    if sigma <= 0 or not math.isfinite(sigma):
        raise ValueError("Realized volatility is not positive.")
    return sigma


def parse_ib_expiry(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def choose_expiry(expirations: list[str], today: date, target_dte: int, min_dte: int, max_dte: int) -> tuple[str, int]:
    candidates: list[tuple[int, str, int]] = []
    for expiry in expirations:
        try:
            expiry_date = parse_ib_expiry(expiry)
        except ValueError:
            continue
        dte = (expiry_date - today).days
        if min_dte <= dte <= max_dte:
            candidates.append((abs(dte - target_dte), expiry, dte))
    if not candidates:
        raise ValueError(f"No expiry found in DTE range {min_dte}-{max_dte}.")
    _, expiry, dte = min(candidates, key=lambda item: (item[0], item[1]))
    return expiry, dte


def nearest_strike(strikes: list[float], target: float) -> float:
    if not strikes:
        raise ValueError("No strikes available.")
    return min(strikes, key=lambda strike: abs(float(strike) - target))


def safe_mid(bid: float | None, ask: float | None, last: float | None = None) -> float | None:
    bid_ok = bid is not None and math.isfinite(bid) and bid > 0
    ask_ok = ask is not None and math.isfinite(ask) and ask > 0
    if bid_ok and ask_ok:
        return (float(bid) + float(ask)) / 2.0
    if last is not None and math.isfinite(last) and last > 0:
        return float(last)
    return None


def import_ib_async() -> Any:
    try:
        import ib_async
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency ib_async. Run `docker compose build analytics` or install requirements.txt."
        ) from exc
    return ib_async


def fetch_signal(args: argparse.Namespace) -> TailRatioSignal:
    ib_async = import_ib_async()
    IB = ib_async.IB
    Index = ib_async.Index
    Option = ib_async.Option

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, timeout=args.timeout)
    try:
        spx = Index("SPX", args.index_exchange, "USD")
        [spx] = ib.qualifyContracts(spx)
        ticker = ib.reqTickers(spx)[0]
        underlying_price = safe_mid(ticker.bid, ticker.ask, ticker.last) or ticker.marketPrice()
        if underlying_price is None or not math.isfinite(underlying_price) or underlying_price <= 0:
            raise ValueError("Could not read a valid SPX market price from IBKR.")

        bars = ib.reqHistoricalData(
            spx,
            endDateTime="",
            durationStr=args.history_duration,
            barSizeSetting="1 day",
            whatToShow=args.history_what_to_show,
            useRTH=True,
            formatDate=1,
        )
        closes = [float(bar.close) for bar in bars if float(bar.close) > 0]
        sigma_daily = realized_vol_from_closes(closes, args.sigma_window)

        chains = ib.reqSecDefOptParams(spx.symbol, "", spx.secType, spx.conId)
        chain = next(
            (
                item
                for item in chains
                if item.tradingClass == args.trading_class and item.exchange in {args.option_chain_exchange, "SMART"}
            ),
            None,
        )
        if chain is None:
            available = sorted({f"{item.exchange}:{item.tradingClass}" for item in chains})
            raise ValueError(f"No matching option chain. Available chains: {available}")

        expiry, dte = choose_expiry(
            sorted(chain.expirations),
            date.today(),
            args.target_dte,
            args.min_dte,
            args.max_dte,
        )
        horizon_sigma = sigma_horizon(sigma_daily, dte)
        long_target = target_strike(underlying_price, horizon_sigma, args.long_sigma, "down")
        short_target = target_strike(underlying_price, horizon_sigma, args.short_sigma, "up")
        strikes = sorted(float(strike) for strike in chain.strikes)
        long_strike = nearest_strike(strikes, long_target)
        short_strike = nearest_strike(strikes, short_target)

        long_put = Option("SPX", expiry, long_strike, "P", args.order_exchange, currency="USD")
        short_put = Option("SPX", expiry, short_strike, "P", args.order_exchange, currency="USD")
        long_put.tradingClass = args.trading_class
        short_put.tradingClass = args.trading_class
        long_put.multiplier = "100"
        short_put.multiplier = "100"
        qualified = ib.qualifyContracts(long_put, short_put)
        long_put, short_put = qualified

        long_ticker, short_ticker = ib.reqTickers(long_put, short_put)
        long_mid = safe_mid(long_ticker.bid, long_ticker.ask, long_ticker.last)
        short_mid = safe_mid(short_ticker.bid, short_ticker.ask, short_ticker.last)

        notes = [
            "Signal only: no order was transmitted.",
            "Manually confirm contract details in TWS paper before trading.",
        ]
        net_credit = None
        net_credit_usd = None
        funded = False
        if long_mid is None or short_mid is None:
            notes.append("Missing bid/ask mid for one or both legs; funding check unavailable.")
        else:
            net_credit = short_mid - args.ratio * long_mid
            net_credit_usd = net_credit * 100.0
            funded = net_credit >= args.min_net_credit
            if not funded:
                notes.append("Target ratio is not fully funded at current mid prices.")

        legs = [
            OptionLegSignal(
                action="BUY",
                quantity=args.ratio,
                symbol="SPX",
                expiry=expiry,
                strike=float(long_put.strike),
                right="P",
                exchange=args.order_exchange,
                trading_class=args.trading_class,
                con_id=getattr(long_put, "conId", None),
                bid=long_ticker.bid,
                ask=long_ticker.ask,
                mid=long_mid,
            ),
            OptionLegSignal(
                action="SELL",
                quantity=1,
                symbol="SPX",
                expiry=expiry,
                strike=float(short_put.strike),
                right="P",
                exchange=args.order_exchange,
                trading_class=args.trading_class,
                con_id=getattr(short_put, "conId", None),
                bid=short_ticker.bid,
                ask=short_ticker.ask,
                mid=short_mid,
            ),
        ]

        return TailRatioSignal(
            generated_at=datetime.now().isoformat(timespec="seconds"),
            account_mode="paper",
            strategy="ratio_tail_put_left_put_5sigma_20x_weekly_candidate",
            underlying_symbol="SPX",
            underlying_price=float(underlying_price),
            sigma_window=args.sigma_window,
            sigma_daily=sigma_daily,
            sigma_horizon=horizon_sigma,
            dte=dte,
            expiry=expiry,
            long_sigma=args.long_sigma,
            short_sigma=args.short_sigma,
            ratio=args.ratio,
            target_long_strike=long_target,
            target_short_strike=short_target,
            net_entry_credit_per_spread=net_credit,
            net_entry_credit_usd=net_credit_usd,
            funded=funded,
            legs=legs,
            notes=notes,
        )
    finally:
        ib.disconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an IBKR paper signal for the SPX tail-ratio put strategy.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497, help="Default TWS paper port is 7497; IB Gateway paper is often 4002.")
    parser.add_argument("--client-id", type=int, default=71)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--index-exchange", default="CBOE")
    parser.add_argument("--option-chain-exchange", default="CBOE")
    parser.add_argument("--order-exchange", default="SMART")
    parser.add_argument("--trading-class", default="SPX")
    parser.add_argument("--history-duration", default="420 D")
    parser.add_argument("--history-what-to-show", default="TRADES")
    parser.add_argument("--sigma-window", type=int, default=252)
    parser.add_argument("--target-dte", type=int, default=60)
    parser.add_argument("--min-dte", type=int, default=45)
    parser.add_argument("--max-dte", type=int, default=75)
    parser.add_argument("--short-sigma", type=float, default=0.4)
    parser.add_argument("--long-sigma", type=float, default=5.0)
    parser.add_argument("--ratio", type=int, default=20)
    parser.add_argument("--min-net-credit", type=float, default=0.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    signal = fetch_signal(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"ibkr_signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = args.out_dir / filename
    output = asdict(signal)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    print(f"\nSaved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
