from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = PROJECT_ROOT / ".deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import pyarrow as pa
import pyarrow.parquet as pq


DATA_DIR = PROJECT_ROOT / "data-parquet"
OUT_DIR = PROJECT_ROOT / "results" / "taleb_tail_ratio"
REQUIRED_COLUMNS = [
    "QUOTE_DATE",
    "UNDERLYING_LAST",
    "EXPIRE_DATE",
    "DTE",
    "STRIKE",
    "C_BID",
    "C_ASK",
    "P_BID",
    "P_ASK",
    "P_IV",
    "P_DELTA",
    "P_GAMMA",
    "P_VEGA",
    "P_THETA",
    "P_RHO",
]
MULTIPLIER = 100.0
SIGMA_HORIZON_MODE = "dte_calendar_to_sessions"
SCALEOUT_EXIT_POLICY = "scaleout_longs_5pct_iv40_itm"
SCALEOUT_CLOSE_SHORT_EXIT_POLICY = "scaleout_longs_close_short_on_first_trigger"
SCALEOUT_LONG_EXIT_REASONS = {"scaleout_drop5", "scaleout_iv40", "scaleout_long_itm"}
SCALEOUT_SHORT_CORE_EXIT_REASON = "scaleout_short_core_after_first_trigger"


@dataclass(frozen=True)
class OptionQuote:
    strike: float
    call_mid: float | None
    put_mid: float | None
    put_iv: float | None = None
    put_delta: float | None = None
    put_gamma: float | None = None
    put_vega: float | None = None
    put_theta: float | None = None
    put_rho: float | None = None


@dataclass(frozen=True)
class Leg:
    option_type: str
    strike: float
    qty: int
    entry_mid: float
    expire_date: date | None = None

    @property
    def side(self) -> str:
        return "long" if self.qty > 0 else "short"


@dataclass
class Trade:
    trade_id: int
    construction: str
    variant: str
    exit_policy: str
    sigma_window: int
    entry_frequency: str
    short_sigma: float
    long_sigma: float
    entry_date: date
    expire_date: date
    dte_entry: float
    underlying_entry: float
    sigma_daily: float
    sigma_t: float
    legs: list[Leg]
    entry_credit: float
    entry_debit: float
    net_entry_cashflow: float
    long_debit: float
    short_credit: float
    sigma_source: str = "realized"
    atm_put_iv: float | None = None
    sigma_60: float | None = None
    strategy_id: str = ""
    sigma_horizon_mode: str = SIGMA_HORIZON_MODE
    ratio_policy: str = ""
    ratio_target: int | None = None
    actual_ratio: float | None = None
    long_put_moneyness: float | None = None
    short_put_moneyness: float | None = None
    long_mid: float | None = None
    short_mid: float | None = None
    tail_cost: float | None = None
    funding_credit: float | None = None
    open_delta: float | None = None
    open_gamma: float | None = None
    open_vega: float | None = None
    open_theta: float | None = None
    open_rho: float | None = None
    open_long_delta: float | None = None
    open_long_gamma: float | None = None
    open_long_vega: float | None = None
    open_long_theta: float | None = None
    open_long_rho: float | None = None
    open_short_delta: float | None = None
    open_short_gamma: float | None = None
    open_short_vega: float | None = None
    open_short_theta: float | None = None
    open_short_rho: float | None = None
    last_mark_date: date | None = None
    last_underlying: float | None = None
    last_exit_value: float | None = None
    last_pnl: float | None = None
    last_long_value: float | None = None
    last_long_pnl: float | None = None
    last_short_pnl: float | None = None
    max_abs_daily_z: float = 0.0
    package_id: str | None = None
    component_type: str | None = None
    bucket_trigger: str | None = None
    long_put_iv_entry: float | None = None
    last_long_put_iv: float | None = None
    trigger_date: date | None = None

    @property
    def total_long_qty(self) -> int:
        return sum(leg.qty for leg in self.legs if leg.qty > 0)

    @property
    def total_short_qty(self) -> int:
        return abs(sum(leg.qty for leg in self.legs if leg.qty < 0))

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.strategy_id,
            self.construction,
            self.variant,
            self.exit_policy,
            self.sigma_window,
            self.entry_frequency,
            self.short_sigma,
            self.long_sigma,
            self.sigma_source,
            self.ratio_policy,
            "none" if self.ratio_target is None else str(self.ratio_target),
        )


def parse_iso_date(value: str) -> date:
    return date.fromisoformat(value)


def parse_csv_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Podaj co najmniej jedno okno sigma.")
    return values


def parse_csv_floats(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Podaj co najmniej jedna wartosc.")
    return values


def parse_csv_choices(value: str, allowed: set[str], name: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(f"Nieznane {name}: {', '.join(unknown)}")
    return values


def clean_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            result = float(text)
        except ValueError:
            return None
    if not math.isfinite(result):
        return None
    return result


def mid_price(bid: object, ask: object) -> float | None:
    bid_value = clean_float(bid)
    ask_value = clean_float(ask)
    if bid_value is None or ask_value is None or bid_value < 0 or ask_value < 0:
        return None
    mid = (bid_value + ask_value) / 2.0
    return mid if mid > 0 else None


def determine_ratio_qty(
    short_mid_price: float,
    long_mid_price: float,
    min_net_credit: float = 0.0,
    max_long_qty: int | None = None,
) -> int:
    if long_mid_price <= 0 or short_mid_price <= 0:
        return 0
    qty = math.floor(short_mid_price / long_mid_price)
    if max_long_qty is not None:
        qty = min(qty, max_long_qty)
    while qty > 0 and short_mid_price - qty * long_mid_price < min_net_credit:
        qty -= 1
    return max(qty, 0)


def sigma_horizon(sigma_daily: float, dte_calendar: float) -> float:
    return sigma_daily * math.sqrt(dte_calendar * 252.0 / 365.0)


def sigma_from_annual_iv(annual_iv: float, target_days: float) -> float:
    return annual_iv * math.sqrt(target_days / 365.0)


def target_strike(underlying: float, sigma_value: float, sigma_multiple: float, direction: str) -> float:
    sign = -1.0 if direction == "down" else 1.0
    return underlying * math.exp(sign * sigma_multiple * sigma_value)


def strategy_id(
    construction: str,
    variant: str,
    exit_policy: str,
    sigma_window: int,
    entry_frequency: str,
    short_sigma: float,
    long_sigma: float,
    sigma_source: str,
    ratio_policy: str,
    ratio_target: int | None,
) -> str:
    ratio_label = "max" if ratio_target is None else str(ratio_target)
    return (
        f"{construction}|{variant}|{exit_policy}|w{sigma_window}|"
        f"{entry_frequency}|short{short_sigma:g}|long{long_sigma:g}|sigsrc{sigma_source}|{ratio_policy}|ratio{ratio_label}"
    )


def trade_expire_date(legs: list[Leg], fallback: date) -> date:
    expiries = [leg.expire_date for leg in legs if leg.expire_date is not None]
    return max(expiries) if expiries else fallback


def iter_parquet_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("spx_eod_*.parquet"))


def file_month_bounds(path: Path) -> tuple[date | None, date | None]:
    stem = path.stem
    suffix = stem.rsplit("_", maxsplit=1)[-1]
    if len(suffix) != 6 or not suffix.isdigit():
        return None, None
    year = int(suffix[:4])
    month = int(suffix[4:])
    start = date(year, month, 1)
    end_month = month + 1
    end_year = year
    if end_month == 13:
        end_month = 1
        end_year += 1
    end = date(end_year, end_month, 1)
    return start, end


def select_files(files: Iterable[Path], start_date: date | None, end_date: date | None) -> list[Path]:
    selected = []
    for path in files:
        month_start, next_month = file_month_bounds(path)
        if month_start is None or next_month is None:
            selected.append(path)
            continue
        if end_date is not None and month_start > end_date:
            continue
        if start_date is not None and next_month <= start_date:
            continue
        selected.append(path)
    return selected


def read_month_rows(path: Path, start_date: date | None, end_date: date | None) -> dict[date, list[dict[str, object]]]:
    table = pq.read_table(path, columns=REQUIRED_COLUMNS)
    rows_by_date: dict[date, list[dict[str, object]]] = defaultdict(list)
    for row in table.to_pylist():
        quote_date = row["QUOTE_DATE"]
        if not isinstance(quote_date, date):
            continue
        if start_date is not None and quote_date < start_date:
            continue
        if end_date is not None and quote_date > end_date:
            continue
        rows_by_date[quote_date].append(row)
    return rows_by_date


def build_chains(rows: list[dict[str, object]]) -> tuple[float | None, dict[date, tuple[float, list[OptionQuote]]]]:
    if not rows:
        return None, {}
    underlying = clean_float(rows[0].get("UNDERLYING_LAST"))
    by_expiry: dict[date, list[OptionQuote]] = defaultdict(list)
    dte_by_expiry: dict[date, float] = {}
    for row in rows:
        expire_date = row.get("EXPIRE_DATE")
        if not isinstance(expire_date, date):
            continue
        strike = clean_float(row.get("STRIKE"))
        dte = clean_float(row.get("DTE"))
        if strike is None or dte is None:
            continue
        by_expiry[expire_date].append(
            OptionQuote(
                strike=strike,
                call_mid=mid_price(row.get("C_BID"), row.get("C_ASK")),
                put_mid=mid_price(row.get("P_BID"), row.get("P_ASK")),
                put_iv=clean_float(row.get("P_IV")),
                put_delta=clean_float(row.get("P_DELTA")),
                put_gamma=clean_float(row.get("P_GAMMA")),
                put_vega=clean_float(row.get("P_VEGA")),
                put_theta=clean_float(row.get("P_THETA")),
                put_rho=clean_float(row.get("P_RHO")),
            )
        )
        dte_by_expiry.setdefault(expire_date, dte)
    chains = {
        expiry: (dte_by_expiry[expiry], sorted(quotes, key=lambda quote: quote.strike))
        for expiry, quotes in by_expiry.items()
        if quotes
    }
    return underlying, chains


def choose_expiry(
    chains: dict[date, tuple[float, list[OptionQuote]]],
    target_dte: float,
    min_dte: float,
    max_dte: float,
) -> tuple[date, float, list[OptionQuote]] | None:
    candidates = [
        (abs(dte - target_dte), expiry, dte, quotes)
        for expiry, (dte, quotes) in chains.items()
        if min_dte <= dte <= max_dte
    ]
    if not candidates:
        return None
    _, expiry, dte, quotes = min(candidates, key=lambda item: (item[0], item[1]))
    return expiry, dte, quotes


def quote_mid(quote: OptionQuote, option_type: str) -> float | None:
    return quote.call_mid if option_type == "call" else quote.put_mid


def nearest_quote(quotes: list[OptionQuote], target_strike: float, option_type: str) -> OptionQuote | None:
    valid_quotes = [quote for quote in quotes if quote_mid(quote, option_type) is not None]
    if not valid_quotes:
        return None
    return min(valid_quotes, key=lambda quote: abs(quote.strike - target_strike))


def nearest_atm_put_iv(quotes: list[OptionQuote], underlying: float) -> float | None:
    valid_quotes = [quote for quote in quotes if quote.put_iv is not None and math.isfinite(float(quote.put_iv))]
    if not valid_quotes:
        return None
    return float(min(valid_quotes, key=lambda quote: abs(quote.strike - underlying)).put_iv or 0.0)


def quote_iv(quote: OptionQuote, option_type: str) -> float | None:
    if option_type == "put":
        return quote.put_iv
    return None


def legs_open_greeks(legs: list[Leg], quotes: list[OptionQuote]) -> dict[str, float | None]:
    quote_by_strike = {quote.strike: quote for quote in quotes}
    basket = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    long_side = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    short_side = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    for leg in legs:
        if leg.option_type != "put":
            continue
        quote = quote_by_strike.get(leg.strike)
        if quote is None:
            continue
        values = {
            "delta": quote.put_delta,
            "gamma": quote.put_gamma,
            "vega": quote.put_vega,
            "theta": quote.put_theta,
            "rho": quote.put_rho,
        }
        for greek, value in values.items():
            if value is None:
                continue
            basket[greek] += leg.qty * float(value)
            if leg.qty > 0:
                long_side[greek] += leg.qty * float(value)
            else:
                short_side[greek] += leg.qty * float(value)
    return {
        "open_delta": basket["delta"],
        "open_gamma": basket["gamma"],
        "open_vega": basket["vega"],
        "open_theta": basket["theta"],
        "open_rho": basket["rho"],
        "open_long_delta": long_side["delta"],
        "open_long_gamma": long_side["gamma"],
        "open_long_vega": long_side["vega"],
        "open_long_theta": long_side["theta"],
        "open_long_rho": long_side["rho"],
        "open_short_delta": short_side["delta"],
        "open_short_gamma": short_side["gamma"],
        "open_short_vega": short_side["vega"],
        "open_short_theta": short_side["theta"],
        "open_short_rho": short_side["rho"],
    }


def option_payoff(option_type: str, strike: float, underlying: float) -> float:
    if option_type == "call":
        return max(underlying - strike, 0.0)
    return max(strike - underlying, 0.0)


def calculate_cashflows(legs: list[Leg]) -> tuple[float, float, float, float, float]:
    credit = sum(abs(leg.qty) * leg.entry_mid * MULTIPLIER for leg in legs if leg.qty < 0)
    debit = sum(leg.qty * leg.entry_mid * MULTIPLIER for leg in legs if leg.qty > 0)
    net = credit - debit
    long_debit = debit
    short_credit = credit
    return credit, debit, net, long_debit, short_credit


def make_fixed_legs(
    variant: str,
    quotes: list[OptionQuote],
    underlying: float,
    sigma_t: float,
    short_sigma: float,
    long_sigma: float,
) -> list[Leg] | None:
    legs: list[Leg] = []
    put_long = nearest_quote(quotes, underlying * math.exp(-long_sigma * sigma_t), "put")
    put_short = nearest_quote(quotes, underlying * math.exp(short_sigma * sigma_t), "put")
    if put_long is None or put_short is None:
        return None
    legs.extend(
        [
            Leg("put", put_long.strike, 2, quote_mid(put_long, "put") or 0.0),
            Leg("put", put_short.strike, -1, quote_mid(put_short, "put") or 0.0),
        ]
    )
    if variant == "symmetric":
        call_long = nearest_quote(quotes, underlying * math.exp(long_sigma * sigma_t), "call")
        call_short = nearest_quote(quotes, underlying * math.exp(-short_sigma * sigma_t), "call")
        if call_long is None or call_short is None:
            return None
        legs.extend(
            [
                Leg("call", call_long.strike, 2, quote_mid(call_long, "call") or 0.0),
                Leg("call", call_short.strike, -1, quote_mid(call_short, "call") or 0.0),
            ]
        )
    return legs


def make_ratio_side(
    option_type: str,
    quotes: list[OptionQuote],
    long_target: float,
    short_target: float,
    min_net_credit: float,
    max_long_qty: int | None,
) -> list[Leg] | None:
    long_quote = nearest_quote(quotes, long_target, option_type)
    short_quote = nearest_quote(quotes, short_target, option_type)
    if long_quote is None or short_quote is None:
        return None
    long_mid = quote_mid(long_quote, option_type)
    short_mid = quote_mid(short_quote, option_type)
    if long_mid is None or short_mid is None:
        return None
    long_qty = determine_ratio_qty(short_mid, long_mid, min_net_credit, max_long_qty)
    if long_qty < 1:
        return None
    return [
        Leg(option_type, long_quote.strike, long_qty, long_mid),
        Leg(option_type, short_quote.strike, -1, short_mid),
    ]


def make_ratio_legs(
    variant: str,
    quotes: list[OptionQuote],
    underlying: float,
    sigma_t: float,
    short_sigma: float,
    long_sigma: float,
    min_net_credit: float,
    max_long_qty: int | None,
) -> list[Leg] | None:
    put_legs = make_ratio_side(
        "put",
        quotes,
        underlying * math.exp(-long_sigma * sigma_t),
        underlying * math.exp(short_sigma * sigma_t),
        min_net_credit,
        max_long_qty,
    )
    if put_legs is None:
        return None
    legs = list(put_legs)
    if variant == "symmetric":
        call_legs = make_ratio_side(
            "call",
            quotes,
            underlying * math.exp(long_sigma * sigma_t),
            underlying * math.exp(-short_sigma * sigma_t),
            min_net_credit,
            max_long_qty,
        )
        if call_legs is None:
            return None
        legs.extend(call_legs)
    return legs


def make_ratio_tail_candidates(
    quotes: list[OptionQuote],
    underlying: float,
    horizon_sigma: float,
    short_sigma: float,
    long_sigma: float,
    ratio_targets: list[int],
    min_net_credit: float,
    max_long_qty: int | None,
) -> tuple[list[tuple[list[Leg], dict[str, object]]], list[dict[str, object]]]:
    long_quote = nearest_quote(
        quotes,
        target_strike(underlying, horizon_sigma, long_sigma, "down"),
        "put",
    )
    short_quote = nearest_quote(
        quotes,
        target_strike(underlying, horizon_sigma, short_sigma, "up"),
        "put",
    )
    if long_quote is None or short_quote is None:
        return [], [{"skip_reason": "missing_quote", "long_sigma": long_sigma}]

    long_mid = quote_mid(long_quote, "put")
    short_mid = quote_mid(short_quote, "put")
    if long_mid is None or short_mid is None or long_mid <= 0 or short_mid <= 0:
        return [], [{"skip_reason": "missing_or_invalid_mid", "long_sigma": long_sigma}]

    candidates: list[tuple[list[Leg], dict[str, object]]] = []
    skipped: list[dict[str, object]] = []

    def build_candidate(qty: int, ratio_policy: str, ratio_target: int | None) -> None:
        if max_long_qty is not None:
            qty = min(qty, max_long_qty)
        net_credit = short_mid - qty * long_mid
        base_meta = {
            "ratio_policy": ratio_policy,
            "ratio_target": ratio_target,
            "actual_ratio": float(qty),
            "long_put_moneyness": long_quote.strike / underlying,
            "short_put_moneyness": short_quote.strike / underlying,
            "long_mid": long_mid,
            "short_mid": short_mid,
            "tail_cost": qty * long_mid * MULTIPLIER,
            "funding_credit": short_mid * MULTIPLIER,
        }
        if qty < 1:
            skipped.append({**base_meta, "skip_reason": "ratio_below_one", "net_credit": net_credit})
            return
        if net_credit < min_net_credit:
            skipped.append({**base_meta, "skip_reason": "insufficient_credit", "net_credit": net_credit})
            return
        candidates.append(
            (
                [
                    Leg("put", long_quote.strike, qty, long_mid),
                    Leg("put", short_quote.strike, -1, short_mid),
                ],
                base_meta,
            )
        )

    for ratio_target in ratio_targets:
        build_candidate(ratio_target, "target", ratio_target)

    max_funded_qty = determine_ratio_qty(short_mid, long_mid, min_net_credit, max_long_qty)
    build_candidate(max_funded_qty, "max_funded", None)
    return candidates, skipped


def make_staggered_ratio_tail_candidates(
    long_quotes: list[OptionQuote],
    short_quotes: list[OptionQuote],
    underlying: float,
    long_horizon_sigma: float,
    short_horizon_sigma: float,
    short_sigma: float,
    long_sigma: float,
    ratio_targets: list[int],
    min_net_credit: float,
    max_long_qty: int | None,
    long_expiry: date,
    short_expiry: date,
) -> tuple[list[tuple[list[Leg], dict[str, object]]], list[dict[str, object]]]:
    long_quote = nearest_quote(
        long_quotes,
        target_strike(underlying, long_horizon_sigma, long_sigma, "down"),
        "put",
    )
    short_quote = nearest_quote(
        short_quotes,
        target_strike(underlying, short_horizon_sigma, short_sigma, "up"),
        "put",
    )
    if long_quote is None or short_quote is None:
        return [], [{"skip_reason": "missing_quote", "long_sigma": long_sigma}]

    long_mid = quote_mid(long_quote, "put")
    short_mid = quote_mid(short_quote, "put")
    if long_mid is None or short_mid is None or long_mid <= 0 or short_mid <= 0:
        return [], [{"skip_reason": "missing_or_invalid_mid", "long_sigma": long_sigma}]

    candidates: list[tuple[list[Leg], dict[str, object]]] = []
    skipped: list[dict[str, object]] = []

    def build_candidate(qty: int, ratio_policy: str, ratio_target: int | None) -> None:
        if max_long_qty is not None:
            qty = min(qty, max_long_qty)
        net_credit = short_mid - qty * long_mid
        base_meta = {
            "ratio_policy": ratio_policy,
            "ratio_target": ratio_target,
            "actual_ratio": float(qty),
            "long_put_moneyness": long_quote.strike / underlying,
            "short_put_moneyness": short_quote.strike / underlying,
            "long_mid": long_mid,
            "short_mid": short_mid,
            "tail_cost": qty * long_mid * MULTIPLIER,
            "funding_credit": short_mid * MULTIPLIER,
            "long_expire_date": long_expiry,
            "short_expire_date": short_expiry,
        }
        if qty < 1:
            skipped.append({**base_meta, "skip_reason": "ratio_below_one", "net_credit": net_credit})
            return
        if net_credit < min_net_credit:
            skipped.append({**base_meta, "skip_reason": "insufficient_credit", "net_credit": net_credit})
            return
        candidates.append(
            (
                [
                    Leg("put", long_quote.strike, qty, long_mid, expire_date=long_expiry),
                    Leg("put", short_quote.strike, -1, short_mid, expire_date=short_expiry),
                ],
                base_meta,
            )
        )

    for ratio_target in ratio_targets:
        build_candidate(ratio_target, "target", ratio_target)

    max_funded_qty = determine_ratio_qty(short_mid, long_mid, min_net_credit, max_long_qty)
    build_candidate(max_funded_qty, "max_funded", None)
    return candidates, skipped


def sigma_for_strikes(
    sigma_source: str,
    sigma_daily: float,
    dte: float,
    quotes: list[OptionQuote],
    underlying: float,
) -> tuple[float | None, float | None, float | None]:
    if sigma_source == "realized":
        horizon = sigma_horizon(sigma_daily, dte)
        return horizon, None, None
    atm_put_iv = nearest_atm_put_iv(quotes, underlying)
    if atm_put_iv is None or atm_put_iv <= 0:
        return None, None, None
    sigma_60 = sigma_from_annual_iv(atm_put_iv, 60.0)
    return sigma_60, atm_put_iv, sigma_60


def mark_trade(
    trade: Trade,
    chains: dict[date, tuple[float, list[OptionQuote]]],
    quote_date: date,
    underlying: float,
) -> bool:
    prices_by_key: dict[tuple[date, str, float], float] = {}
    iv_by_key: dict[tuple[date, str, float], float] = {}
    for expiry, chain in chains.items():
        _, quotes = chain
        for quote in quotes:
            if quote.call_mid is not None:
                prices_by_key[(expiry, "call", quote.strike)] = quote.call_mid
            if quote.put_mid is not None:
                prices_by_key[(expiry, "put", quote.strike)] = quote.put_mid
            if quote.put_iv is not None:
                iv_by_key[(expiry, "put", quote.strike)] = float(quote.put_iv)

    exit_value = 0.0
    long_value = 0.0
    long_pnl = 0.0
    short_pnl = 0.0
    long_put_iv_exit = None
    for leg in trade.legs:
        leg_expiry = leg.expire_date or trade.expire_date
        mid = prices_by_key.get((leg_expiry, leg.option_type, leg.strike))
        if mid is None and quote_date >= leg_expiry:
            mid = option_payoff(leg.option_type, leg.strike, underlying)
        if mid is None:
            return False
        leg_value = leg.qty * mid * MULTIPLIER
        leg_entry_cashflow = -leg.qty * leg.entry_mid * MULTIPLIER
        leg_pnl = leg_entry_cashflow + leg_value
        exit_value += leg_value
        if leg.qty > 0:
            long_value += leg_value
            long_pnl += leg_pnl
            if leg.option_type == "put":
                long_put_iv_exit = iv_by_key.get((leg_expiry, "put", leg.strike), long_put_iv_exit)
        else:
            short_pnl += leg_pnl

    trade.last_mark_date = quote_date
    trade.last_underlying = underlying
    trade.last_exit_value = exit_value
    trade.last_pnl = trade.net_entry_cashflow + exit_value
    trade.last_long_value = long_value
    trade.last_long_pnl = long_pnl
    trade.last_short_pnl = short_pnl
    trade.last_long_put_iv = long_put_iv_exit
    return True


def realized_vol(returns: list[float], window: int) -> float | None:
    if len(returns) < window:
        return None
    sigma = pstdev(returns[-window:])
    return sigma if sigma > 0 and math.isfinite(sigma) else None


def should_profit_exit(trade: Trade, args: argparse.Namespace) -> bool:
    if trade.last_pnl is None or trade.last_long_value is None:
        return False
    if trade.long_debit > 0 and trade.last_long_value >= args.profit_long_multiple * trade.long_debit:
        return True
    pnl_base = max(abs(trade.net_entry_cashflow), MULTIPLIER)
    return trade.last_pnl >= args.profit_pnl_multiple * pnl_base


def should_shock_exit(trade: Trade, daily_z: float | None, threshold: float) -> bool:
    if daily_z is None:
        return False
    trade.max_abs_daily_z = max(trade.max_abs_daily_z, abs(daily_z))
    if trade.variant == "left_put":
        return daily_z <= -threshold
    return abs(daily_z) >= threshold


def should_market_drop_exit(daily_return: float | None, drop_threshold: float) -> bool:
    if daily_return is None:
        return False
    return daily_return <= -abs(drop_threshold)


def is_scaleout_policy(exit_policy: str) -> bool:
    return exit_policy in {SCALEOUT_EXIT_POLICY, SCALEOUT_CLOSE_SHORT_EXIT_POLICY}


def closes_short_core_on_scaleout_trigger(exit_policy: str) -> bool:
    return exit_policy == SCALEOUT_CLOSE_SHORT_EXIT_POLICY


def should_skip_scaleout_entry(long_put_iv_entry: float | None, max_iv: float) -> bool:
    return long_put_iv_entry is not None and long_put_iv_entry > max_iv


def long_put_is_itm(trade: Trade, underlying: float) -> bool:
    return any(leg.option_type == "put" and leg.qty > 0 and underlying <= leg.strike for leg in trade.legs)


def should_exit_at_dte(trade: Trade, quote_date: date, dte_threshold: int) -> bool:
    return (trade.expire_date - quote_date).days <= dte_threshold


def should_exit_combined(trade: Trade, quote_date: date, underlying: float, dte_threshold: int) -> bool:
    return long_put_is_itm(trade, underlying) or should_exit_at_dte(trade, quote_date, dte_threshold)


def should_scaleout_component_exit(
    trade: Trade,
    quote_date: date,
    underlying: float,
    daily_return: float | None,
    args: argparse.Namespace,
) -> str | None:
    if quote_date <= trade.entry_date:
        return None
    if trade.component_type == "long_drop5" and should_market_drop_exit(daily_return, args.market_drop_exit_pct):
        return "scaleout_drop5"
    if (
        trade.component_type == "long_iv40"
        and trade.last_long_put_iv is not None
        and trade.last_long_put_iv >= args.scaleout_exit_iv_threshold
    ):
        return "scaleout_iv40"
    if trade.component_type == "long_itm" and long_put_is_itm(trade, underlying):
        return "scaleout_long_itm"
    return None


def should_close_scaleout_short_core(trade: Trade, triggered_package_ids: set[str]) -> bool:
    return (
        closes_short_core_on_scaleout_trigger(trade.exit_policy)
        and trade.component_type == "short_core"
        and trade.package_id is not None
        and trade.package_id in triggered_package_ids
    )


def trade_to_record(trade: Trade, exit_reason: str) -> dict[str, object]:
    exit_date = trade.last_mark_date or trade.entry_date
    underlying_exit = trade.last_underlying or trade.underlying_entry
    holding_days = max((exit_date - trade.entry_date).days, 0)
    pnl = trade.last_pnl if trade.last_pnl is not None else 0.0
    move_sigma = None
    if trade.sigma_daily > 0:
        denom = trade.sigma_daily * math.sqrt(max(holding_days, 1))
        move_sigma = math.log(underlying_exit / trade.underlying_entry) / denom if denom > 0 else None
    return {
        "trade_id": trade.trade_id,
        "strategy_id": trade.strategy_id,
        "construction": trade.construction,
        "variant": trade.variant,
        "exit_policy": trade.exit_policy,
        "sigma_window": trade.sigma_window,
        "entry_frequency": trade.entry_frequency,
        "short_sigma": trade.short_sigma,
        "long_sigma": trade.long_sigma,
        "sigma_source": trade.sigma_source,
        "sigma_horizon_mode": trade.sigma_horizon_mode,
        "ratio_policy": trade.ratio_policy,
        "ratio_target": trade.ratio_target,
        "actual_ratio": trade.actual_ratio,
        "entry_date": trade.entry_date,
        "exit_date": exit_date,
        "expire_date": trade.expire_date,
        "holding_days": holding_days,
        "dte_entry": trade.dte_entry,
        "underlying_entry": trade.underlying_entry,
        "underlying_exit": underlying_exit,
        "sigma_daily": trade.sigma_daily,
        "sigma_t": trade.sigma_t,
        "sigma_horizon": trade.sigma_t,
        "atm_put_iv": trade.atm_put_iv,
        "sigma_60": trade.sigma_60,
        "long_put_moneyness": trade.long_put_moneyness,
        "short_put_moneyness": trade.short_put_moneyness,
        "long_mid": trade.long_mid,
        "short_mid": trade.short_mid,
        "open_delta": trade.open_delta,
        "open_gamma": trade.open_gamma,
        "open_vega": trade.open_vega,
        "open_theta": trade.open_theta,
        "open_rho": trade.open_rho,
        "open_long_delta": trade.open_long_delta,
        "open_long_gamma": trade.open_long_gamma,
        "open_long_vega": trade.open_long_vega,
        "open_long_theta": trade.open_long_theta,
        "open_long_rho": trade.open_long_rho,
        "open_short_delta": trade.open_short_delta,
        "open_short_gamma": trade.open_short_gamma,
        "open_short_vega": trade.open_short_vega,
        "open_short_theta": trade.open_short_theta,
        "open_short_rho": trade.open_short_rho,
        "tail_cost": trade.tail_cost,
        "funding_credit": trade.funding_credit,
        "entry_credit": trade.entry_credit,
        "entry_debit": trade.entry_debit,
        "net_entry_cashflow": trade.net_entry_cashflow,
        "exit_value": trade.last_exit_value or 0.0,
        "pnl": pnl,
        "pnl_pct_on_debit": pnl / trade.entry_debit if trade.entry_debit > 0 else None,
        "long_qty": trade.total_long_qty,
        "short_qty": trade.total_short_qty,
        "long_debit": trade.long_debit,
        "short_credit": trade.short_credit,
        "long_legs_pnl": trade.last_long_pnl or 0.0,
        "short_legs_pnl": trade.last_short_pnl or 0.0,
        "max_abs_daily_z": trade.max_abs_daily_z,
        "move_sigma": move_sigma,
        "exit_reason": exit_reason,
        "package_id": trade.package_id,
        "component_type": trade.component_type,
        "bucket_trigger": trade.bucket_trigger,
        "trigger_date": trade.trigger_date,
        "long_put_iv_entry": trade.long_put_iv_entry,
        "long_put_iv_exit": trade.last_long_put_iv,
        "legs_json": json.dumps(
            [
                {
                    **leg.__dict__,
                    "expire_date": leg.expire_date.isoformat() if leg.expire_date is not None else None,
                }
                for leg in trade.legs
            ],
            sort_keys=True,
        ),
    }


def aggregate_equity(
    quote_date: date,
    active_trades: list[Trade],
    realized_pnl: dict[tuple[object, ...], float],
) -> list[dict[str, object]]:
    keys = set(realized_pnl)
    keys.update(trade.key for trade in active_trades)
    rows = []
    for key in sorted(keys):
        unrealized = sum(trade.last_pnl or 0.0 for trade in active_trades if trade.key == key)
        realized = realized_pnl.get(key, 0.0)
        rows.append(
            {
                "quote_date": quote_date,
                "strategy_id": key[0],
                "construction": key[1],
                "variant": key[2],
                "exit_policy": key[3],
                "sigma_window": key[4],
                "entry_frequency": key[5],
                "short_sigma": key[6],
                "long_sigma": key[7],
                "sigma_source": key[8],
                "ratio_policy": key[9],
                "ratio_target": key[10],
                "realized_pnl": realized,
                "unrealized_pnl": unrealized,
                "equity_pnl": realized + unrealized,
                "open_trades": sum(1 for trade in active_trades if trade.key == key),
            }
        )
    return rows


def classify_move(move_sigma: float | None) -> str:
    if move_sigma is None:
        return "unknown"
    abs_move = abs(move_sigma)
    if abs_move <= 0.4:
        return "quiet"
    if abs_move <= 2.5:
        return "medium"
    return "tail"


def rollup_scaleout_packages(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    packages: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        packages[str(row.get("package_id") or row["trade_id"])].append(row)
    rolled: list[dict[str, object]] = []
    for _, package_rows in sorted(packages.items()):
        first = package_rows[0]
        short_core = next((row for row in package_rows if row.get("component_type") == "short_core"), package_rows[0])
        longest = max(package_rows, key=lambda row: int(row.get("holding_days") or 0))
        rolled.append(
            {
                **first,
                "trade_id": short_core.get("trade_id"),
                "entry_date": first.get("entry_date"),
                "exit_date": longest.get("exit_date"),
                "expire_date": longest.get("expire_date"),
                "holding_days": longest.get("holding_days"),
                "underlying_exit": longest.get("underlying_exit"),
                "move_sigma": short_core.get("move_sigma"),
                "entry_credit": sum(float(row.get("entry_credit") or 0.0) for row in package_rows),
                "entry_debit": sum(float(row.get("entry_debit") or 0.0) for row in package_rows),
                "net_entry_cashflow": sum(float(row.get("net_entry_cashflow") or 0.0) for row in package_rows),
                "long_debit": sum(float(row.get("long_debit") or 0.0) for row in package_rows),
                "short_credit": sum(float(row.get("short_credit") or 0.0) for row in package_rows),
                "exit_value": sum(float(row.get("exit_value") or 0.0) for row in package_rows),
                "pnl": sum(float(row.get("pnl") or 0.0) for row in package_rows),
                "long_legs_pnl": sum(float(row.get("long_legs_pnl") or 0.0) for row in package_rows),
                "short_legs_pnl": sum(float(row.get("short_legs_pnl") or 0.0) for row in package_rows),
                "long_qty": sum(int(row.get("long_qty") or 0) for row in package_rows),
                "short_qty": max(int(row.get("short_qty") or 0) for row in package_rows),
                "actual_ratio": first.get("actual_ratio"),
                "open_delta": sum(float(row.get("open_delta") or 0.0) for row in package_rows),
                "open_gamma": sum(float(row.get("open_gamma") or 0.0) for row in package_rows),
                "open_vega": sum(float(row.get("open_vega") or 0.0) for row in package_rows),
                "open_theta": sum(float(row.get("open_theta") or 0.0) for row in package_rows),
                "open_rho": sum(float(row.get("open_rho") or 0.0) for row in package_rows),
                "package_id": first.get("package_id"),
            }
        )
    return rolled


def summarize(
    trade_rows: list[dict[str, object]],
    equity_rows: list[dict[str, object]],
    skipped_rows: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in trade_rows:
        groups[row["strategy_id"]].append(row)

    equity_by_key: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in equity_rows:
        equity_by_key[row["strategy_id"]].append(row)

    skipped_by_strategy: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in skipped_rows or []:
        strategy_key = row.get("strategy_id")
        if strategy_key:
            skipped_by_strategy[str(strategy_key)].append(row)

    summary_rows = []
    for key, rows in sorted(groups.items()):
        first = rows[0]
        summary_input_rows = rollup_scaleout_packages(rows) if is_scaleout_policy(str(first["exit_policy"])) else rows
        pnls = [float(row["pnl"]) for row in summary_input_rows]
        ratios = [
            float(row["long_qty"]) / float(row["short_qty"])
            for row in summary_input_rows
            if row["short_qty"] and float(row["short_qty"]) > 0
        ]
        credit_count = sum(1 for row in summary_input_rows if float(row["net_entry_cashflow"]) >= 0)
        regime_pnl = defaultdict(float)
        for row in summary_input_rows:
            regime_pnl[classify_move(row.get("move_sigma"))] += float(row["pnl"])

        equity_values = [float(row["equity_pnl"]) for row in equity_by_key.get(key, [])]
        max_drawdown = 0.0
        sharpe = None
        if equity_values:
            peak = equity_values[0]
            for value in equity_values:
                peak = max(peak, value)
                max_drawdown = min(max_drawdown, value - peak)
            diffs = [b - a for a, b in zip(equity_values, equity_values[1:])]
            if len(diffs) > 1 and pstdev(diffs) > 0:
                sharpe = mean(diffs) / pstdev(diffs) * math.sqrt(252)

        skipped_for_strategy = skipped_by_strategy.get(str(key), [])
        trigger_counts = Counter(
            str(row.get("exit_reason"))
            for row in rows
            if str(row.get("exit_reason")) in SCALEOUT_LONG_EXIT_REASONS
        )
        short_core_close_count = sum(
            1 for row in rows if str(row.get("exit_reason")) == SCALEOUT_SHORT_CORE_EXIT_REASON
        )
        package_buckets_closed = None
        if is_scaleout_policy(str(first["exit_policy"])):
            per_package_closed: list[int] = []
            package_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
            for row in rows:
                package_groups[str(row.get("package_id") or row.get("trade_id"))].append(row)
            for package_rows in package_groups.values():
                closed = sum(
                    1
                    for row in package_rows
                    if str(row.get("component_type", "")).startswith("long_")
                    and str(row.get("exit_reason")) in SCALEOUT_LONG_EXIT_REASONS
                )
                per_package_closed.append(closed)
            package_buckets_closed = mean(per_package_closed) if per_package_closed else None

        summary_rows.append(
            {
                "strategy_id": key,
                "construction": first["construction"],
                "variant": first["variant"],
                "exit_policy": first["exit_policy"],
                "sigma_window": first["sigma_window"],
                "entry_frequency": first["entry_frequency"],
                "short_sigma": first["short_sigma"],
                "long_sigma": first["long_sigma"],
                "sigma_source": first["sigma_source"],
                "sigma_horizon_mode": first["sigma_horizon_mode"],
                "ratio_policy": first["ratio_policy"],
                "ratio_target": first["ratio_target"],
                "trades": len(summary_input_rows),
                "total_pnl": sum(pnls),
                "avg_pnl": mean(pnls) if pnls else 0.0,
                "win_rate": sum(1 for pnl in pnls if pnl > 0) / len(pnls) if pnls else 0.0,
                "max_loss": min(pnls) if pnls else 0.0,
                "max_gain": max(pnls) if pnls else 0.0,
                "max_drawdown": max_drawdown,
                "daily_pnl_sharpe": sharpe,
                "avg_long_short_ratio": mean(ratios) if ratios else None,
                "median_actual_ratio": sorted(ratios)[len(ratios) // 2] if ratios else None,
                "avg_long_put_moneyness": mean(
                    [float(row["long_put_moneyness"]) for row in summary_input_rows if row["long_put_moneyness"] is not None]
                )
                if any(row["long_put_moneyness"] is not None for row in summary_input_rows)
                else None,
                "avg_short_put_moneyness": mean(
                    [float(row["short_put_moneyness"]) for row in summary_input_rows if row["short_put_moneyness"] is not None]
                )
                if any(row["short_put_moneyness"] is not None for row in summary_input_rows)
                else None,
                "avg_open_delta": mean([float(row["open_delta"]) for row in summary_input_rows if row["open_delta"] is not None])
                if any(row["open_delta"] is not None for row in summary_input_rows)
                else None,
                "avg_open_gamma": mean([float(row["open_gamma"]) for row in summary_input_rows if row["open_gamma"] is not None])
                if any(row["open_gamma"] is not None for row in summary_input_rows)
                else None,
                "avg_open_vega": mean([float(row["open_vega"]) for row in summary_input_rows if row["open_vega"] is not None])
                if any(row["open_vega"] is not None for row in summary_input_rows)
                else None,
                "avg_open_theta": mean([float(row["open_theta"]) for row in summary_input_rows if row["open_theta"] is not None])
                if any(row["open_theta"] is not None for row in summary_input_rows)
                else None,
                "avg_open_rho": mean([float(row["open_rho"]) for row in summary_input_rows if row["open_rho"] is not None])
                if any(row["open_rho"] is not None for row in summary_input_rows)
                else None,
                "avg_atm_put_iv": mean([float(row["atm_put_iv"]) for row in summary_input_rows if row["atm_put_iv"] is not None])
                if any(row["atm_put_iv"] is not None for row in summary_input_rows)
                else None,
                "avg_sigma_60": mean([float(row["sigma_60"]) for row in summary_input_rows if row["sigma_60"] is not None])
                if any(row["sigma_60"] is not None for row in summary_input_rows)
                else None,
                "credit_entry_share": credit_count / len(summary_input_rows) if summary_input_rows else 0.0,
                "pnl_quiet": regime_pnl["quiet"],
                "pnl_medium": regime_pnl["medium"],
                "pnl_tail": regime_pnl["tail"],
                "pnl_unknown": regime_pnl["unknown"],
                "scaleout_trigger_drop5_count": trigger_counts.get("scaleout_drop5", 0),
                "scaleout_trigger_iv40_count": trigger_counts.get("scaleout_iv40", 0),
                "scaleout_trigger_itm_count": trigger_counts.get("scaleout_long_itm", 0),
                "scaleout_short_core_close_count": short_core_close_count,
                "avg_scaleout_buckets_closed_pre_expiry": package_buckets_closed,
                "entry_iv_limit_rejections": sum(
                    1 for row in skipped_for_strategy if row.get("skip_reason") == "entry_long_iv_above_limit"
                ),
            }
        )
    return summary_rows


def make_trade(
    trade_id: int,
    construction: str,
    variant: str,
    exit_policy: str,
    sigma_window: int,
    entry_frequency: str,
    quote_date: date,
    underlying: float,
    sigma_daily: float,
    expiry: date,
    dte: float,
    legs: list[Leg],
    args: argparse.Namespace,
    long_sigma: float,
    ratio_policy: str,
    ratio_target: int | None,
    leg_meta: dict[str, object] | None = None,
) -> Trade | None:
    entry_credit, entry_debit, net_cashflow, long_debit, short_credit = calculate_cashflows(legs)
    if construction in {"ratio_credit", "ratio_tail_put", "ratio_tail_put_staggered"} and net_cashflow < args.min_net_credit * MULTIPLIER:
        return None
    leg_meta = leg_meta or {}
    trade_strategy_id = strategy_id(
        construction,
        variant,
        exit_policy,
        sigma_window,
        entry_frequency,
        args.short_sigma,
        long_sigma,
        args.sigma_source,
        ratio_policy,
        ratio_target,
    )
    return Trade(
        trade_id=trade_id,
        construction=construction,
        variant=variant,
        exit_policy=exit_policy,
        sigma_window=sigma_window,
        entry_frequency=entry_frequency,
        short_sigma=args.short_sigma,
        long_sigma=long_sigma,
        entry_date=quote_date,
        expire_date=trade_expire_date(legs, expiry),
        dte_entry=dte,
        underlying_entry=underlying,
        sigma_daily=sigma_daily,
        sigma_t=float(leg_meta.get("sigma_t") if leg_meta.get("sigma_t") is not None else sigma_horizon(sigma_daily, dte)),
        sigma_source=args.sigma_source,
        atm_put_iv=leg_meta.get("atm_put_iv"),
        sigma_60=leg_meta.get("sigma_60"),
        legs=legs,
        entry_credit=entry_credit,
        entry_debit=entry_debit,
        net_entry_cashflow=net_cashflow,
        long_debit=long_debit,
        short_credit=short_credit,
        strategy_id=trade_strategy_id,
        ratio_policy=ratio_policy,
        ratio_target=ratio_target,
        actual_ratio=leg_meta.get("actual_ratio"),
        long_put_moneyness=leg_meta.get("long_put_moneyness"),
        short_put_moneyness=leg_meta.get("short_put_moneyness"),
        long_mid=leg_meta.get("long_mid"),
        short_mid=leg_meta.get("short_mid"),
        tail_cost=leg_meta.get("tail_cost"),
        funding_credit=leg_meta.get("funding_credit"),
        open_delta=leg_meta.get("open_delta"),
        open_gamma=leg_meta.get("open_gamma"),
        open_vega=leg_meta.get("open_vega"),
        open_theta=leg_meta.get("open_theta"),
        open_rho=leg_meta.get("open_rho"),
        open_long_delta=leg_meta.get("open_long_delta"),
        open_long_gamma=leg_meta.get("open_long_gamma"),
        open_long_vega=leg_meta.get("open_long_vega"),
        open_long_theta=leg_meta.get("open_long_theta"),
        open_long_rho=leg_meta.get("open_long_rho"),
        open_short_delta=leg_meta.get("open_short_delta"),
        open_short_gamma=leg_meta.get("open_short_gamma"),
        open_short_vega=leg_meta.get("open_short_vega"),
        open_short_theta=leg_meta.get("open_short_theta"),
        open_short_rho=leg_meta.get("open_short_rho"),
        long_put_iv_entry=leg_meta.get("long_put_iv_entry"),
    )


def make_scaleout_component_trades(
    package_id: str,
    next_trade_id: int,
    quote_date: date,
    base_trade: Trade,
    long_leg: Leg,
    short_leg: Leg,
    quotes: list[OptionQuote],
) -> list[Trade]:
    component_specs = [
        ("short_core", None, [short_leg]),
        ("long_drop5", "drop5", [Leg("put", long_leg.strike, 5, long_leg.entry_mid, long_leg.expire_date)]),
        ("long_iv40", "iv40", [Leg("put", long_leg.strike, 5, long_leg.entry_mid, long_leg.expire_date)]),
        ("long_itm", "itm", [Leg("put", long_leg.strike, 5, long_leg.entry_mid, long_leg.expire_date)]),
        ("long_residual", "residual", [Leg("put", long_leg.strike, 5, long_leg.entry_mid, long_leg.expire_date)]),
    ]
    trades: list[Trade] = []
    current_id = next_trade_id
    for component_type, bucket_trigger, legs in component_specs:
        greek_meta = legs_open_greeks(legs, quotes)
        component = Trade(
            trade_id=current_id,
            construction=base_trade.construction,
            variant=base_trade.variant,
            exit_policy=base_trade.exit_policy,
            sigma_window=base_trade.sigma_window,
            entry_frequency=base_trade.entry_frequency,
            short_sigma=base_trade.short_sigma,
            long_sigma=base_trade.long_sigma,
            entry_date=quote_date,
            expire_date=trade_expire_date(legs, base_trade.expire_date),
            dte_entry=base_trade.dte_entry,
            underlying_entry=base_trade.underlying_entry,
            sigma_daily=base_trade.sigma_daily,
            sigma_t=base_trade.sigma_t,
            legs=legs,
            entry_credit=calculate_cashflows(legs)[0],
            entry_debit=calculate_cashflows(legs)[1],
            net_entry_cashflow=calculate_cashflows(legs)[2],
            long_debit=calculate_cashflows(legs)[3],
            short_credit=calculate_cashflows(legs)[4],
            sigma_source=base_trade.sigma_source,
            atm_put_iv=base_trade.atm_put_iv,
            sigma_60=base_trade.sigma_60,
            strategy_id=base_trade.strategy_id,
            sigma_horizon_mode=base_trade.sigma_horizon_mode,
            ratio_policy=base_trade.ratio_policy,
            ratio_target=base_trade.ratio_target,
            actual_ratio=base_trade.actual_ratio,
            long_put_moneyness=base_trade.long_put_moneyness,
            short_put_moneyness=base_trade.short_put_moneyness,
            long_mid=base_trade.long_mid,
            short_mid=base_trade.short_mid,
            tail_cost=base_trade.tail_cost,
            funding_credit=base_trade.funding_credit,
            open_delta=greek_meta.get("open_delta"),
            open_gamma=greek_meta.get("open_gamma"),
            open_vega=greek_meta.get("open_vega"),
            open_theta=greek_meta.get("open_theta"),
            open_rho=greek_meta.get("open_rho"),
            open_long_delta=greek_meta.get("open_long_delta"),
            open_long_gamma=greek_meta.get("open_long_gamma"),
            open_long_vega=greek_meta.get("open_long_vega"),
            open_long_theta=greek_meta.get("open_long_theta"),
            open_long_rho=greek_meta.get("open_long_rho"),
            open_short_delta=greek_meta.get("open_short_delta"),
            open_short_gamma=greek_meta.get("open_short_gamma"),
            open_short_vega=greek_meta.get("open_short_vega"),
            open_short_theta=greek_meta.get("open_short_theta"),
            open_short_rho=greek_meta.get("open_short_rho"),
            package_id=package_id,
            component_type=component_type,
            bucket_trigger=bucket_trigger,
            long_put_iv_entry=base_trade.long_put_iv_entry,
        )
        trades.append(component)
        current_id += 1
    return trades


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows) if rows else pa.table({})
    pq.write_table(table, path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_equity(out_dir: Path, equity_rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in equity_rows:
        grouped[str(row["strategy_id"])].append(row)
    if not grouped:
        return
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 7))
    for key, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: row["quote_date"])
        label = key
        plt.plot([row["quote_date"] for row in rows], [row["equity_pnl"] for row in rows], label=label, linewidth=1)
    plt.title("Taleb long-kurtosis equity PnL")
    plt.xlabel("Date")
    plt.ylabel("PnL")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(plots_dir / "equity_curves.png", dpi=150)
    plt.close()


def run_backtest(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    files = select_files(iter_parquet_files(args.data_dir), args.start_date, args.end_date)
    if not files:
        raise FileNotFoundError(f"Brak plikow Parquet w {args.data_dir}")

    constructions = ["fixed", "ratio_credit", "ratio_tail_put", "ratio_tail_put_staggered"] if args.construction == "both" else [args.construction]
    variants = ["symmetric", "left_put"] if args.variant == "both" else [args.variant]
    exit_policies = parse_csv_choices(
        args.exit_policies,
        {
            "hold_to_expiry",
            "shock_exit",
            "profit_exit",
            "long_itm_exit",
            "exit_at_30_dte",
            "combined_exit",
            "market_drop_exit",
            SCALEOUT_EXIT_POLICY,
            SCALEOUT_CLOSE_SHORT_EXIT_POLICY,
        },
        "reguly wyjscia",
    )

    active_trades: list[Trade] = []
    closed_trade_rows: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    realized_pnl: dict[tuple[object, ...], float] = defaultdict(float)
    returns_history: list[float] = []
    previous_underlying: float | None = None
    last_entry_week: tuple[int, int] | None = None
    last_entry_date: date | None = None
    cooldown_until_by_strategy: dict[str, date] = {}
    trade_id = 1
    package_id_counter = 1

    for file_index, file_path in enumerate(files, start=1):
        print(f"[{file_index}/{len(files)}] {file_path.name}")
        rows_by_date = read_month_rows(file_path, args.start_date, args.end_date)
        for quote_date in sorted(rows_by_date):
            underlying, chains = build_chains(rows_by_date[quote_date])
            if underlying is None:
                continue

            daily_z_by_trade: dict[int, float | None] = {}
            daily_return = None
            if previous_underlying is not None and previous_underlying > 0:
                daily_return = (underlying / previous_underlying) - 1.0
            for trade in active_trades:
                daily_z = None
                if previous_underlying is not None and trade.sigma_daily > 0:
                    daily_z = math.log(underlying / previous_underlying) / trade.sigma_daily
                daily_z_by_trade[trade.trade_id] = daily_z

            still_active: list[Trade] = []
            scaleout_triggered_packages: set[str] = set()
            for trade in active_trades:
                if quote_date > trade.expire_date and trade.last_mark_date is not None:
                    row = trade_to_record(trade, "hold_to_expiry_last_quote")
                    closed_trade_rows.append(row)
                    realized_pnl[trade.key] += float(row["pnl"])
                    continue
                marked = mark_trade(trade, chains, quote_date, underlying)
                if not marked:
                    still_active.append(trade)
                    continue
                exit_reason = None
                if quote_date >= trade.expire_date:
                    exit_reason = "hold_to_expiry"
                elif is_scaleout_policy(trade.exit_policy):
                    exit_reason = should_scaleout_component_exit(trade, quote_date, underlying, daily_return, args)
                elif trade.exit_policy == "market_drop_exit" and should_market_drop_exit(
                    daily_return,
                    args.market_drop_exit_pct,
                ):
                    exit_reason = "market_drop_exit"
                elif trade.exit_policy == "shock_exit" and should_shock_exit(
                    trade,
                    daily_z_by_trade.get(trade.trade_id),
                    args.shock_sigma,
                ):
                    exit_reason = "shock_exit"
                elif trade.exit_policy == "profit_exit" and should_profit_exit(trade, args):
                    exit_reason = "profit_exit"
                elif trade.exit_policy == "long_itm_exit" and long_put_is_itm(trade, underlying):
                    exit_reason = "long_itm_exit"
                elif trade.exit_policy == "exit_at_30_dte" and should_exit_at_dte(
                    trade,
                    quote_date,
                    args.exit_dte_threshold,
                ):
                    exit_reason = "exit_at_30_dte"
                elif trade.exit_policy == "combined_exit" and should_exit_combined(
                    trade,
                    quote_date,
                    underlying,
                    args.exit_dte_threshold,
                ):
                    if long_put_is_itm(trade, underlying):
                        exit_reason = "long_itm_exit"
                    else:
                        exit_reason = "exit_at_30_dte"

                if exit_reason:
                    if str(exit_reason).startswith("scaleout_"):
                        trade.trigger_date = quote_date
                    if (
                        exit_reason in SCALEOUT_LONG_EXIT_REASONS
                        and closes_short_core_on_scaleout_trigger(trade.exit_policy)
                        and trade.package_id is not None
                    ):
                        scaleout_triggered_packages.add(trade.package_id)
                    row = trade_to_record(trade, exit_reason)
                    closed_trade_rows.append(row)
                    realized_pnl[trade.key] += float(row["pnl"])
                    if exit_reason == "market_drop_exit":
                        cooldown_until_by_strategy[trade.strategy_id] = max(
                            cooldown_until_by_strategy.get(trade.strategy_id, quote_date),
                            quote_date + timedelta(days=args.market_shock_cooldown_days),
                        )
                else:
                    still_active.append(trade)

            if scaleout_triggered_packages:
                still_active_after_short_close: list[Trade] = []
                for trade in still_active:
                    if should_close_scaleout_short_core(trade, scaleout_triggered_packages):
                        marked = mark_trade(trade, chains, quote_date, underlying)
                        if not marked:
                            still_active_after_short_close.append(trade)
                            continue
                        trade.trigger_date = quote_date
                        row = trade_to_record(trade, SCALEOUT_SHORT_CORE_EXIT_REASON)
                        closed_trade_rows.append(row)
                        realized_pnl[trade.key] += float(row["pnl"])
                    else:
                        still_active_after_short_close.append(trade)
                still_active = still_active_after_short_close
            active_trades = still_active

            current_week = (quote_date.isocalendar().year, quote_date.isocalendar().week)
            if args.entry_frequency == "daily":
                should_open_today = True
            elif args.entry_frequency == "weekly":
                should_open_today = current_week != last_entry_week
            else:
                should_open_today = last_entry_date is None or (quote_date - last_entry_date).days >= args.entry_interval_days
            attempted_weekly_entry = False

            if should_open_today:
                for sigma_window in args.sigma_window:
                    sigma_daily = realized_vol(returns_history, sigma_window)
                    if sigma_daily is None:
                        continue
                    expiry_choice = choose_expiry(chains, args.target_dte, args.min_dte, args.max_dte)
                    staggered_short_expiry_choice = choose_expiry(
                        chains,
                        args.short_target_dte,
                        args.short_min_dte,
                        args.short_max_dte,
                    )
                    if expiry_choice is None and staggered_short_expiry_choice is None:
                        continue
                    attempted_weekly_entry = True
                    expiry = None
                    dte = None
                    quotes = None
                    horizon_sigma = None
                    if expiry_choice is not None:
                        expiry, dte, quotes = expiry_choice
                        horizon_sigma, atm_put_iv, sigma_60 = sigma_for_strikes(
                            args.sigma_source,
                            sigma_daily,
                            dte,
                            quotes,
                            underlying,
                        )
                    for construction in constructions:
                        for variant in variants:
                            if construction in {"ratio_tail_put", "ratio_tail_put_staggered"} and variant != "left_put":
                                continue
                            for long_sigma in args.long_sigma:
                                candidate_sets: list[tuple[list[Leg], dict[str, object], str, int | None]]
                                candidate_sets = []
                                if construction == "fixed":
                                    if quotes is None or horizon_sigma is None or expiry is None or dte is None:
                                        continue
                                    legs = make_fixed_legs(
                                        variant,
                                        quotes,
                                        underlying,
                                        horizon_sigma,
                                        args.short_sigma,
                                        long_sigma,
                                    )
                                    if legs:
                                        meta = {
                                            "sigma_t": horizon_sigma,
                                            "atm_put_iv": atm_put_iv,
                                            "sigma_60": sigma_60,
                                            **legs_open_greeks(legs, quotes),
                                        }
                                        candidate_sets.append((legs, meta, "fixed", None))
                                elif construction == "ratio_credit":
                                    if quotes is None or horizon_sigma is None or expiry is None or dte is None:
                                        continue
                                    legs = make_ratio_legs(
                                        variant,
                                        quotes,
                                        underlying,
                                        horizon_sigma,
                                        args.short_sigma,
                                        long_sigma,
                                        args.min_net_credit,
                                        args.max_long_qty,
                                    )
                                    if legs:
                                        meta = {
                                            "sigma_t": horizon_sigma,
                                            "atm_put_iv": atm_put_iv,
                                            "sigma_60": sigma_60,
                                            **legs_open_greeks(legs, quotes),
                                        }
                                        candidate_sets.append((legs, meta, "max_funded", None))
                                elif construction == "ratio_tail_put":
                                    if quotes is None or horizon_sigma is None or expiry is None or dte is None:
                                        if quotes is not None and expiry is not None and dte is not None and args.sigma_source == "atm_put_iv_60d":
                                            skipped_rows.append(
                                                {
                                                    "quote_date": quote_date,
                                                    "construction": construction,
                                                    "variant": variant,
                                                    "sigma_window": sigma_window,
                                                    "entry_frequency": args.entry_frequency,
                                                    "long_sigma": long_sigma,
                                                    "underlying": underlying,
                                                    "expire_date": expiry,
                                                    "dte": dte,
                                                    "skip_reason": "missing_atm_put_iv",
                                                }
                                            )
                                        continue
                                    candidates, skipped = make_ratio_tail_candidates(
                                        quotes,
                                        underlying,
                                        horizon_sigma,
                                        args.short_sigma,
                                        long_sigma,
                                        args.ratio_targets,
                                        args.min_net_credit,
                                        args.max_long_qty,
                                    )
                                    for skipped_row in skipped:
                                        skipped_rows.append(
                                            {
                                                "quote_date": quote_date,
                                                "construction": construction,
                                                "variant": variant,
                                                "sigma_window": sigma_window,
                                                "entry_frequency": args.entry_frequency,
                                                "long_sigma": long_sigma,
                                                "underlying": underlying,
                                                "expire_date": expiry,
                                                "dte": dte,
                                                **skipped_row,
                                            }
                                        )
                                    for legs, meta in candidates:
                                        meta = {
                                            **meta,
                                            "sigma_t": horizon_sigma,
                                            "atm_put_iv": atm_put_iv,
                                            "sigma_60": sigma_60,
                                            **legs_open_greeks(legs, quotes),
                                        }
                                        candidate_sets.append(
                                            (
                                                legs,
                                                meta,
                                                str(meta.get("ratio_policy") or ""),
                                                meta.get("ratio_target") if isinstance(meta.get("ratio_target"), int) else None,
                                            )
                                        )
                                else:
                                    if (
                                        expiry_choice is None
                                        or staggered_short_expiry_choice is None
                                    ):
                                        continue
                                    long_expiry, long_dte, long_quotes = expiry_choice
                                    short_expiry, short_dte, short_quotes = staggered_short_expiry_choice
                                    long_horizon_sigma, long_atm_put_iv, long_sigma_60 = sigma_for_strikes(
                                        args.sigma_source,
                                        sigma_daily,
                                        long_dte,
                                        long_quotes,
                                        underlying,
                                    )
                                    short_horizon_sigma, _, _ = sigma_for_strikes(
                                        args.sigma_source,
                                        sigma_daily,
                                        short_dte,
                                        short_quotes,
                                        underlying,
                                    )
                                    if long_horizon_sigma is None or short_horizon_sigma is None:
                                        skipped_rows.append(
                                            {
                                                "quote_date": quote_date,
                                                "construction": construction,
                                                "variant": variant,
                                                "sigma_window": sigma_window,
                                                "entry_frequency": args.entry_frequency,
                                                "long_sigma": long_sigma,
                                                "underlying": underlying,
                                                "expire_date": long_expiry,
                                                "dte": long_dte,
                                                "short_expire_date": short_expiry,
                                                "short_dte": short_dte,
                                                "skip_reason": "missing_atm_put_iv",
                                            }
                                        )
                                        continue
                                    candidates, skipped = make_staggered_ratio_tail_candidates(
                                        long_quotes,
                                        short_quotes,
                                        underlying,
                                        long_horizon_sigma,
                                        short_horizon_sigma,
                                        args.short_sigma,
                                        long_sigma,
                                        args.ratio_targets,
                                        args.min_net_credit,
                                        args.max_long_qty,
                                        long_expiry,
                                        short_expiry,
                                    )
                                    for skipped_row in skipped:
                                        skipped_rows.append(
                                            {
                                                "quote_date": quote_date,
                                                "construction": construction,
                                                "variant": variant,
                                                "sigma_window": sigma_window,
                                                "entry_frequency": args.entry_frequency,
                                                "long_sigma": long_sigma,
                                                "underlying": underlying,
                                                "expire_date": long_expiry,
                                                "dte": long_dte,
                                                "short_expire_date": short_expiry,
                                                "short_dte": short_dte,
                                                **skipped_row,
                                            }
                                        )
                                    for legs, meta in candidates:
                                        meta = {
                                            **meta,
                                            "sigma_t": long_horizon_sigma,
                                            "atm_put_iv": long_atm_put_iv,
                                            "sigma_60": long_sigma_60,
                                            **legs_open_greeks(legs, long_quotes + short_quotes),
                                        }
                                        candidate_sets.append(
                                            (
                                                legs,
                                                meta,
                                                str(meta.get("ratio_policy") or ""),
                                                meta.get("ratio_target") if isinstance(meta.get("ratio_target"), int) else None,
                                            )
                                        )
                                if not candidate_sets:
                                    continue
                                for legs, leg_meta, ratio_policy, ratio_target in candidate_sets:
                                    base_expiry = expiry if expiry is not None else leg_meta.get("long_expire_date") or leg_meta.get("short_expire_date")
                                    base_dte = dte if dte is not None else long_dte
                                    for exit_policy in exit_policies:
                                        trade = make_trade(
                                            trade_id,
                                            construction,
                                            variant,
                                            exit_policy,
                                            sigma_window,
                                            args.entry_frequency,
                                            quote_date,
                                            underlying,
                                            sigma_daily,
                                            base_expiry,
                                            base_dte,
                                            legs,
                                            args,
                                            long_sigma,
                                            ratio_policy,
                                            ratio_target,
                                            leg_meta,
                                        )
                                        if trade is None:
                                            continue
                                        if is_scaleout_policy(exit_policy):
                                            long_leg = next((leg for leg in trade.legs if leg.option_type == "put" and leg.qty > 0), None)
                                            short_leg = next((leg for leg in trade.legs if leg.option_type == "put" and leg.qty < 0), None)
                                            if long_leg is None or short_leg is None or long_leg.qty != 20:
                                                continue
                                            long_quote = next(
                                                (
                                                    quote
                                                    for quote in (quotes or [])
                                                    if quote.strike == long_leg.strike and quote.put_mid is not None
                                                ),
                                                None,
                                            )
                                            long_put_iv_entry = quote_iv(long_quote, "put") if long_quote is not None else None
                                            if should_skip_scaleout_entry(long_put_iv_entry, args.scaleout_entry_iv_max):
                                                skipped_rows.append(
                                                    {
                                                        "quote_date": quote_date,
                                                        "strategy_id": trade.strategy_id,
                                                        "construction": construction,
                                                        "variant": variant,
                                                        "sigma_window": sigma_window,
                                                        "entry_frequency": args.entry_frequency,
                                                        "long_sigma": long_sigma,
                                                        "short_sigma": args.short_sigma,
                                                        "sigma_source": args.sigma_source,
                                                        "ratio_policy": ratio_policy,
                                                        "ratio_target": ratio_target,
                                                        "exit_policy": exit_policy,
                                                        "underlying": underlying,
                                                        "expire_date": base_expiry,
                                                        "dte": base_dte,
                                                        "skip_reason": "entry_long_iv_above_limit",
                                                        "long_put_iv_entry": long_put_iv_entry,
                                                    }
                                                )
                                                continue
                                            trade.long_put_iv_entry = long_put_iv_entry
                                        cooldown_until = cooldown_until_by_strategy.get(trade.strategy_id)
                                        if cooldown_until is not None and quote_date < cooldown_until:
                                            skipped_rows.append(
                                                {
                                                    "quote_date": quote_date,
                                                    "strategy_id": trade.strategy_id,
                                                    "construction": construction,
                                                    "variant": variant,
                                                    "sigma_window": sigma_window,
                                                    "entry_frequency": args.entry_frequency,
                                                    "long_sigma": long_sigma,
                                                    "underlying": underlying,
                                                    "expire_date": base_expiry,
                                                    "dte": base_dte,
                                                    "ratio_policy": ratio_policy,
                                                    "ratio_target": ratio_target,
                                                    "exit_policy": exit_policy,
                                                    "sigma_source": args.sigma_source,
                                                    "skip_reason": "cooldown_active",
                                                }
                                            )
                                            continue
                                        if is_scaleout_policy(exit_policy):
                                            package_id = f"{trade.strategy_id}|pkg{package_id_counter}"
                                            package_id_counter += 1
                                            component_trades = make_scaleout_component_trades(
                                                package_id,
                                                trade_id,
                                                quote_date,
                                                trade,
                                                long_leg,
                                                short_leg,
                                                quotes or [],
                                            )
                                            for component_trade in component_trades:
                                                mark_trade(component_trade, chains, quote_date, underlying)
                                                active_trades.append(component_trade)
                                            trade_id += len(component_trades)
                                        else:
                                            mark_trade(trade, chains, quote_date, underlying)
                                            active_trades.append(trade)
                                            trade_id += 1

            if args.entry_frequency == "weekly" and attempted_weekly_entry:
                last_entry_week = current_week
            if args.entry_frequency == "every_n_days" and attempted_weekly_entry:
                last_entry_date = quote_date

            equity_rows.extend(aggregate_equity(quote_date, active_trades, realized_pnl))

            if previous_underlying is not None and underlying > 0 and previous_underlying > 0:
                returns_history.append(math.log(underlying / previous_underlying))
            previous_underlying = underlying

    for trade in active_trades:
        if trade.last_mark_date is None:
            continue
        row = trade_to_record(trade, "end_of_data")
        closed_trade_rows.append(row)
        realized_pnl[trade.key] += float(row["pnl"])

    summary_rows = summarize(closed_trade_rows, equity_rows, skipped_rows)
    return closed_trade_rows, equity_rows, summary_rows, skipped_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest strategii Taleba long-kurtosis na opcjach SPX.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Katalog z plikami Parquet.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Katalog wynikow.")
    parser.add_argument("--start-date", type=parse_iso_date, help="Data startu YYYY-MM-DD.")
    parser.add_argument("--end-date", type=parse_iso_date, help="Data konca YYYY-MM-DD.")
    parser.add_argument("--construction", choices=["fixed", "ratio_credit", "ratio_tail_put", "ratio_tail_put_staggered", "both"], default="ratio_tail_put")
    parser.add_argument("--variant", choices=["symmetric", "left_put", "both"], default="left_put")
    parser.add_argument("--exit-policies", default="hold_to_expiry")
    parser.add_argument("--entry-frequency", choices=["daily", "weekly", "every_n_days"], default="daily")
    parser.add_argument("--entry-interval-days", type=int, default=10)
    parser.add_argument("--sigma-window", "--sigma-windows", dest="sigma_window", type=parse_csv_ints, default=[252])
    parser.add_argument("--sigma-source", choices=["realized", "atm_put_iv_60d"], default="realized")
    parser.add_argument("--target-dte", type=float, default=60.0)
    parser.add_argument("--min-dte", type=float, default=45.0)
    parser.add_argument("--max-dte", type=float, default=75.0)
    parser.add_argument("--short-target-dte", type=float, default=30.0)
    parser.add_argument("--short-min-dte", type=float, default=20.0)
    parser.add_argument("--short-max-dte", type=float, default=40.0)
    parser.add_argument("--short-sigma", type=float, default=0.4)
    parser.add_argument("--long-sigma", type=parse_csv_floats, default=[3.0, 4.0, 5.0])
    parser.add_argument("--ratio-targets", type=parse_csv_ints, default=[5, 10, 20, 40])
    parser.add_argument("--min-net-credit", type=float, default=0.0, help="Minimalny kredyt netto na jedna sprzedana strone, przed mnoznikiem.")
    parser.add_argument("--max-long-qty", type=int, help="Opcjonalny limit liczby kupowanych OTM opcji na strone.")
    parser.add_argument("--exit-dte-threshold", type=int, default=30)
    parser.add_argument("--shock-sigma", type=float, default=2.5)
    parser.add_argument("--market-drop-exit-pct", type=float, default=0.05)
    parser.add_argument("--market-shock-cooldown-days", type=int, default=14)
    parser.add_argument("--scaleout-entry-iv-max", type=float, default=0.50)
    parser.add_argument("--scaleout-exit-iv-threshold", type=float, default=0.50)
    parser.add_argument("--profit-long-multiple", type=float, default=3.0)
    parser.add_argument("--profit-pnl-multiple", type=float, default=1.0)
    parser.add_argument("--no-plots", action="store_true", help="Nie generuj wykresow matplotlib.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.data_dir = args.data_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Nie znaleziono katalogu danych: {args.data_dir}")

    trades, equity, summary, skipped = run_backtest(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_table(args.out_dir / "trades.parquet", trades)
    write_table(args.out_dir / "daily_equity.parquet", equity)
    write_csv(args.out_dir / "summary.csv", summary)
    write_csv(args.out_dir / "skipped_ratio_candidates.csv", skipped)
    if not args.no_plots:
        plot_equity(args.out_dir, equity)

    print(f"Zamkniete transakcje: {len(trades)}")
    print(f"Wiersze equity: {len(equity)}")
    print(f"Pominiete kandydaty ratio: {len(skipped)}")
    print(f"Wyniki: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
