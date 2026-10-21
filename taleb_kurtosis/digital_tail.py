"""Monthly SPX tail put financed by limited-loss put credit spreads."""
from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pyarrow.parquet as pq


MULTIPLIER = 100.0
STRATEGY = "digital_tail_atm_iv_baseline"
DEFAULT_OUTPUT = Path("results/digital_tail_atm_iv_baseline")
COLUMNS = ["QUOTE_DATE", "EXPIRE_DATE", "STRIKE", "UNDERLYING_LAST",
           "P_BID", "P_ASK", "P_DELTA", "P_IV"]


@dataclass(frozen=True)
class Config:
    target_dte: int = 120
    min_dte: int = 105
    max_dte: int = 135
    tail_delta: float = 0.01
    delta_tolerance: float = 0.0025
    spread_sigma: float = 2.5
    fee_per_contract: float = 1.0
    initial_capital: float = 100_000.0
    risk_per_package: float = 0.05
    total_risk: float = 0.20
    max_spreads: int = 100
    exit_policy: str = "pre_expiry_bid_ask"
    profit_target_risk: float = 0.0
    expiry_mode: str = "friday_only"

    def __post_init__(self) -> None:
        if not 0 < self.min_dte <= self.target_dte <= self.max_dte:
            raise ValueError("Invalid DTE window")
        if not 0 < self.tail_delta < 1 or not 0 <= self.delta_tolerance < self.tail_delta:
            raise ValueError("Invalid delta target/tolerance")
        if not 0 < self.risk_per_package <= self.total_risk <= 1:
            raise ValueError("Invalid risk limits")
        if self.initial_capital <= 0 or self.spread_sigma <= 0 or self.fee_per_contract < 0 or self.max_spreads < 1:
            raise ValueError("Invalid capital, sigma, fee or spread count")
        if self.exit_policy not in {"pre_expiry_bid_ask", "barrier_close_all"}:
            raise ValueError("Invalid exit policy")
        if self.expiry_mode not in {"friday_only", "monthly_aliases"}:
            raise ValueError("Invalid expiry mode")
        if not 0 <= self.profit_target_risk <= 1:
            raise ValueError("Invalid profit target")
        if self.profit_target_risk and self.exit_policy != "barrier_close_all":
            raise ValueError("Profit target requires barrier_close_all")
        if not all(math.isfinite(float(v)) for v in asdict(self).values() if not isinstance(v, str)):
            raise ValueError("Parameters must be finite")


def strategy_id(cfg: Config) -> str:
    if cfg.expiry_mode == "friday_only" and cfg.exit_policy == "pre_expiry_bid_ask" and cfg.spread_sigma == 2.5:
        return STRATEGY
    exit_name = "hold" if cfg.exit_policy == "pre_expiry_bid_ask" else "barrier"
    return f"digital_s{cfg.spread_sigma:g}_{exit_name}_tp{cfg.profit_target_risk:g}_{cfg.expiry_mode}"


def exit_label(cfg: Config) -> str:
    if cfg.exit_policy == "pre_expiry_bid_ask":
        return "Przed expiry"
    return f"Bariera + TP {cfg.profit_target_risk:.0%}" if cfg.profit_target_risk else "Bariera"


class MonthlyExpiryMap:
    """Research alias convention, not a replacement for missing SPX/SPXW roots."""

    def __init__(self, start: pd.Timestamp, end: pd.Timestamp):
        import pandas_market_calendars as mcal

        self.sessions = mcal.get_calendar("CBOE_Index_Options").valid_days(
            start - pd.Timedelta(days=35), end + pd.Timedelta(days=210), tz=None)
        self.mapping = {}
        self.conflicts: set[pd.Timestamp] = set()
        for month in pd.period_range(start, end + pd.Timedelta(days=180), freq="M"):
            first = month.start_time
            friday = first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)
            settlement = self.sessions[self.sessions <= friday][-1]
            last_trade = self.sessions[self.sessions < settlement][-1]
            for source in {last_trade, settlement, friday, friday + pd.Timedelta(days=1)}:
                self.mapping[source] = settlement

    def normalize(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        frame = frame.copy()
        frame["SOURCE_EXPIRE_DATE"] = frame["EXPIRE_DATE"]
        frame["MONTHLY_EXPIRY"] = frame["EXPIRE_DATE"].map(self.mapping)
        frame = frame.dropna(subset=["MONTHLY_EXPIRY"])
        keys = ["QUOTE_DATE", "MONTHLY_EXPIRY"]
        ambiguous = frame.groupby(keys)["SOURCE_EXPIRE_DATE"].transform("nunique").gt(1)
        audit = frame[[*keys, "SOURCE_EXPIRE_DATE"]].assign(ambiguous=ambiguous).drop_duplicates()
        return frame, audit

    def entry_frame(self, day: pd.DataFrame) -> pd.DataFrame:
        counts = day.groupby("MONTHLY_EXPIRY")["SOURCE_EXPIRE_DATE"].nunique()
        self.conflicts.update(counts[counts > 1].index)
        result = day[day["MONTHLY_EXPIRY"].isin(counts[counts == 1].index)].copy()
        result["EXPIRE_DATE"] = result["MONTHLY_EXPIRY"]
        return result

    def mark(self, position: dict, day: pd.DataFrame, lookup: pd.DataFrame, cfg: Config) -> dict | None:
        source = position.get("last_source_expire_date", position["source_expire_date"])
        if source not in lookup.index.get_level_values("EXPIRE_DATE"):
            # A label switch is an explicit assumption, permitted only for a sole
            # monthly chain and never after simultaneous roots have been observed.
            sources = day.loc[day["MONTHLY_EXPIRY"].eq(position["expire_date"]), "SOURCE_EXPIRE_DATE"].unique()
            if len(sources) != 1 or position["expire_date"] in self.conflicts:
                return None
            source = pd.Timestamp(sources[0])
        marked = mark_package({**position, "expire_date": source}, lookup, cfg)
        if marked is not None:
            if source != position.get("last_source_expire_date", position["source_expire_date"]):
                position["source_label_switches"] = position.get("source_label_switches", 0) + 1
            position["last_source_expire_date"] = source
        return marked


def valid_quote(row: pd.Series) -> bool:
    bid, ask = row["P_BID"], row["P_ASK"]
    return bool(np.isfinite(bid) and np.isfinite(ask) and 0 <= bid <= ask and ask > 0)


def funding_count(tail_ask: float, credit: float, fee: float) -> int | None:
    # Reserve both entry and exit commissions, including the financing legs.
    available = credit * MULTIPLIER - 4 * fee
    if available <= 0:
        return None
    return max(1, math.ceil((tail_ask * MULTIPLIER + 2 * fee) / available))


def expiry_pnl(row: dict | pd.Series, spots: np.ndarray) -> np.ndarray:
    spreads = np.maximum(row["short_strike"] - spots, 0) - np.maximum(row["hedge_strike"] - spots, 0)
    return (row["net_credit_after_costs"] + MULTIPLIER *
            (np.maximum(row["tail_strike"] - spots, 0) - row["spread_count"] * spreads))


def build_candidate(day: pd.DataFrame, when: pd.Timestamp, cfg: Config) -> tuple[dict | None, str]:
    expiries = sorted(day["EXPIRE_DATE"].unique())
    expiries = [pd.Timestamp(e) for e in expiries if cfg.min_dte <= (pd.Timestamp(e) - when).days <= cfg.max_dte]
    if not expiries:
        return None, "no_105_135_dte_expiry"
    # Use consistently identifiable monthly Friday series. Some source rows label
    # the same apparent monthly chain as Thursday; never silently merge contracts.
    if cfg.expiry_mode == "friday_only":
        expiries = [e for e in expiries if e.weekday() == 4 and 15 <= e.day <= 21]
    if not expiries:
        return None, "no_standard_monthly_friday_expiry"
    expiry = min(expiries, key=lambda e: (abs((e - when).days - cfg.target_dte), e))
    chain = day[day["EXPIRE_DATE"] == expiry].sort_values("STRIKE").reset_index(drop=True)
    spot = float(day["UNDERLYING_LAST"].median())
    if not math.isfinite(spot) or spot <= 0:
        return None, "invalid_spot"
    atm = chain.iloc[(chain["STRIKE"] - spot).abs().argmin()]
    iv = float(atm["P_IV"])
    if (not valid_quote(atm) or not math.isfinite(iv) or not 0.03 <= iv <= 3.0
            or not -0.65 <= float(atm["P_DELTA"]) <= -0.35):
        return None, "invalid_atm_iv"
    horizon = iv * math.sqrt((expiry - when).days / 365.0)
    target = spot * math.exp(-cfg.spread_sigma * horizon)
    short_index = int((chain["STRIKE"] - target).abs().argmin())
    if short_index == 0:
        return None, "no_adjacent_lower_strike"
    short, hedge = chain.iloc[short_index], chain.iloc[short_index - 1]
    tails = chain[(chain["STRIKE"] < hedge["STRIKE"]) & (chain["P_DELTA"] < 0)]
    if tails.empty:
        return None, "no_tail_below_spread"
    tail = tails.iloc[(tails["P_DELTA"] + cfg.tail_delta).abs().argmin()]
    if abs(float(tail["P_DELTA"]) + cfg.tail_delta) > cfg.delta_tolerance:
        return None, "tail_delta_outside_tolerance"
    if not all(valid_quote(q) for q in [tail, short, hedge]) or tail["P_BID"] <= 0:
        return None, "invalid_entry_quotes"
    credit = float(short["P_BID"] - hedge["P_ASK"])
    width = float(short["STRIKE"] - hedge["STRIKE"])
    if not 0 < credit < width:
        return None, "nonpositive_or_invalid_credit"
    n = funding_count(float(tail["P_ASK"]), credit, cfg.fee_per_contract)
    if n is None or n > cfg.max_spreads:
        return None, "funding_unavailable_after_costs"
    contracts = 1 + 2 * n
    entry_fee = contracts * cfg.fee_per_contract
    tail_cash = -float(tail["P_ASK"]) * MULTIPLIER - cfg.fee_per_contract
    spread_cash = n * credit * MULTIPLIER - 2 * n * cfg.fee_per_contract
    net_credit = tail_cash + spread_cash - entry_fee
    risk = n * width * MULTIPLIER - net_credit
    return {
        "strategy_id": strategy_id(cfg), "entry_date": when, "expire_date": expiry,
        "source_expire_date": pd.Timestamp(short.get("SOURCE_EXPIRE_DATE", expiry)),
        "dte_entry": (expiry - when).days, "underlying_entry": spot, "atm_iv": iv,
        "sigma_horizon": horizon, "spread_sigma": cfg.spread_sigma,
        "actual_short_sigma": -math.log(float(short["STRIKE"]) / spot) / horizon,
        "tail_delta": float(tail["P_DELTA"]), "tail_strike": float(tail["STRIKE"]),
        "short_strike": float(short["STRIKE"]), "hedge_strike": float(hedge["STRIKE"]),
        "width": width, "spread_count": n, "spread_credit": credit,
        "tail_cost": float(tail["P_ASK"]) * MULTIPLIER,
        "tail_entry_cash": tail_cash, "spread_entry_cash": spread_cash,
        "net_entry_cashflow": tail_cash + spread_cash,
        "entry_fee": entry_fee, "exit_fee_reserve": entry_fee,
        "net_credit_after_costs": net_credit, "risk_budget": risk,
        "lower_breakeven": float(tail["STRIKE"]) - risk / MULTIPLIER,
        "upper_breakeven": float(short["STRIKE"]) - net_credit / (n * MULTIPLIER),
    }, "candidate"


def mark_package(position: dict, lookup: pd.DataFrame, cfg: Config) -> dict | None:
    try:
        qs = [lookup.loc[(position["expire_date"], position[k])]
              for k in ["tail_strike", "short_strike", "hedge_strike"]]
    except KeyError:
        return None
    if not all(valid_quote(q) for q in qs):
        return None
    tail, short, hedge = qs
    n = position["spread_count"]
    mid = lambda q: float(q["P_BID"] + q["P_ASK"]) / 2
    raw_mid = mid(short) - mid(hedge)
    lower = max(0.0, float(short["P_BID"] - hedge["P_ASK"]))
    upper = min(position["width"], float(short["P_ASK"] - hedge["P_BID"]))
    if lower > upper:
        return None
    # Overlapping quotes can give a negative vertical mid near expiry even when
    # executable bid/ask are consistent. Project only the mark, never the fills.
    spread_mid = float(np.clip(raw_mid, lower, upper))
    tail_pnl = position["tail_entry_cash"] + mid(tail) * MULTIPLIER - cfg.fee_per_contract
    spread_pnl = position["spread_entry_cash"] - n * spread_mid * MULTIPLIER - 2 * n * cfg.fee_per_contract
    tail_liq = position["tail_entry_cash"] + float(tail["P_BID"]) * MULTIPLIER - cfg.fee_per_contract
    spread_liq = (position["spread_entry_cash"] + n * float(hedge["P_BID"] - short["P_ASK"]) * MULTIPLIER
                  - 2 * n * cfg.fee_per_contract)
    return {"tail_pnl": tail_pnl, "spread_pnl": spread_pnl,
            "pnl": tail_pnl + spread_pnl, "tail_liq": tail_liq, "spread_liq": spread_liq,
            "liquidation_pnl": tail_liq + spread_liq, "mark_projected": abs(spread_mid - raw_mid) > 1e-8}


def run(data_dir: Path, output: Path, cfg: Config, start: str | None = None, end: str | None = None) -> dict:
    if cfg.expiry_mode != "friday_only" or cfg.exit_policy != "pre_expiry_bid_ask" or cfg.profit_target_risk:
        raise ValueError("Use digital_tail_comparison.run_comparison for barrier/alias experiments")
    files = sorted(data_dir.glob("spx_eod_*.parquet"))
    if start:
        files = [p for p in files if p.stem[-6:] >= start[:7].replace("-", "")]
    if end:
        files = [p for p in files if p.stem[-6:] <= end[:7].replace("-", "")]
    if not files:
        raise ValueError("No input files")
    dates = set()
    for p in files:
        dates.update(pq.read_table(p, columns=["QUOTE_DATE"])["QUOTE_DATE"].unique().to_pylist())
    calendar = pd.DatetimeIndex(sorted(dates))
    calendar = calendar[(calendar >= pd.Timestamp(start or calendar[0])) & (calendar <= pd.Timestamp(end or calendar[-1]))]
    if calendar.empty:
        raise ValueError("No sessions in requested period")
    active, closed, equity_rows, attempts = [], [], [], []
    realized_tail = realized_spread = 0.0
    entered_months: set[tuple[int, int]] = set()
    duplicate_rows = 0
    for i, p in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {p.name}", flush=True)
        frame = pq.read_table(p, columns=COLUMNS, filters=[("DTE", ">=", 0), ("DTE", "<=", cfg.max_dte + 2)]).to_pandas()
        for col in ["QUOTE_DATE", "EXPIRE_DATE"]:
            frame[col] = pd.to_datetime(frame[col])
        for col in COLUMNS[2:]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        duplicates = frame.duplicated(["QUOTE_DATE", "EXPIRE_DATE", "STRIKE"], keep=False)
        duplicate_rows += int(duplicates.sum())
        frame = frame[~duplicates]
        for when, day in frame.groupby("QUOTE_DATE", sort=True):
            if when not in calendar:
                continue
            lookup = day.set_index(["EXPIRE_DATE", "STRIKE"])
            remaining = []
            for pos in active:
                marked = mark_package(pos, lookup, cfg) if when < pos["expire_date"] else None
                if marked is not None:
                    pos.update(marked)
                    pos["last_mark_date"] = when
                else:
                    pos["missing_mark_days"] += 1
                if when >= pos["planned_exit_date"] and marked is not None:
                    pos.update(exit_date=when, exit_reason="pre_expiry_bid_ask", status="closed",
                               pnl=marked["liquidation_pnl"], tail_pnl=marked["tail_liq"], spread_pnl=marked["spread_liq"])
                    realized_tail += pos["tail_pnl"]
                    realized_spread += pos["spread_pnl"]
                    closed.append(pos)
                else:
                    remaining.append(pos)
            active = remaining
            month = (when.year, when.month)
            if month not in entered_months:
                candidate, reason = build_candidate(day, when, cfg)
                detail = {"quote_date": when, "reason": reason}
                if candidate is not None:
                    detail.update({k: candidate[k] for k in ["expire_date", "risk_budget", "spread_count", "net_credit_after_costs"]})
                    current_equity = cfg.initial_capital + realized_tail + realized_spread + sum(x["pnl"] for x in active)
                    if any(x["last_mark_date"] != when for x in active):
                        reason = "stale_portfolio_marks"
                    elif candidate["risk_budget"] > cfg.risk_per_package * max(current_equity, 0):
                        reason = "package_risk_limit"
                    elif candidate["risk_budget"] + sum(x["risk_budget"] for x in active) > cfg.total_risk * max(current_equity, 0):
                        reason = "portfolio_risk_limit"
                    elif candidate["expire_date"] > calendar[-1]:
                        reason = "expiry_beyond_data"
                    else:
                        candidate.update(package_id=f"digital-{len(closed) + len(active) + 1}",
                                         planned_exit_date=calendar[calendar < candidate["expire_date"]][-1],
                                         missing_mark_days=0, last_mark_date=when)
                        marked = mark_package(candidate, lookup, cfg)
                        if marked is None:
                            reason = "invalid_entry_mark"
                        else:
                            candidate.update(marked)
                            active.append(candidate)
                            entered_months.add(month)
                            reason = "entered"
                    detail["reason"] = reason
                attempts.append(detail)
            tail_pnl = realized_tail + sum(x["tail_pnl"] for x in active)
            spread_pnl = realized_spread + sum(x["spread_pnl"] for x in active)
            equity_rows.append({"quote_date": when, "strategy_id": STRATEGY,
                                "equity_pnl": tail_pnl + spread_pnl, "tail_pnl": tail_pnl, "spread_pnl": spread_pnl,
                                "equity": cfg.initial_capital + tail_pnl + spread_pnl,
                                "liquidation_equity": cfg.initial_capital + realized_tail + realized_spread + sum(x["liquidation_pnl"] for x in active),
                                "realized_pnl": realized_tail + realized_spread,
                                "unrealized_pnl": sum(x["pnl"] for x in active), "open_trades": len(active),
                                "reserved_risk": sum(x["risk_budget"] for x in active),
                                "stale_packages": sum(x["last_mark_date"] != when for x in active),
                                "projected_marks": sum(x["mark_projected"] for x in active)})
    # Missing terminal quotes stay unresolved; no invented settlement or stale-price fills.
    for pos in active:
        pos.update(exit_date=pd.NaT, exit_reason="unresolved_missing_quotes", status="unresolved")
    trades = pd.DataFrame(closed + active)
    eq = pd.DataFrame(equity_rows)
    if eq.empty:
        raise ValueError("No daily observations")
    peak = eq["equity"].cummax().clip(lower=cfg.initial_capital)
    eq["drawdown"] = eq["equity"] - peak
    eq["drawdown_pct"] = eq["equity"] / peak - 1
    eq["daily_pnl"] = eq["equity_pnl"].diff().fillna(eq["equity_pnl"].iloc[0])
    returns = eq["equity"].pct_change()
    returns.iloc[0] = eq["equity"].iloc[0] / cfg.initial_capital - 1
    done = trades[trades["status"] == "closed"] if not trades.empty else trades
    summary = {"strategy_id": STRATEGY, "construction": "digital_tail", "exit_policy": "pre_expiry_bid_ask",
               "entry_frequency": "monthly", "sigma_source": "atm_iv_actual_dte", "trades": len(done),
               "total_pnl": float(eq["equity_pnl"].iloc[-1]), "max_drawdown": float(eq["drawdown"].min()),
               "max_drawdown_pct": float(eq["drawdown_pct"].min()),
               "win_rate": float((done["pnl"] > 0).mean()) if len(done) else None,
               "daily_pnl_sharpe": float(returns.mean() / returns.std(ddof=0) * math.sqrt(252)) if returns.std(ddof=0) > 0 else None,
               "tail_pnl": float(eq["tail_pnl"].iloc[-1]), "spread_pnl": float(eq["spread_pnl"].iloc[-1]),
               "max_open_packages": int(eq["open_trades"].max()), "max_reserved_risk": float(eq["reserved_risk"].max()),
               "stale_mark_sessions": int((eq["stale_packages"] > 0).sum()), "unresolved_packages": len(active),
               "projected_mark_sessions": int((eq["projected_marks"] > 0).sum()),
               "ambiguous_quote_rows_excluded": duplicate_rows, "initial_capital": cfg.initial_capital}
    output.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(output / "trades.parquet", index=False)
    eq.to_parquet(output / "daily_equity.parquet", index=False)
    pd.DataFrame([summary]).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(attempts).to_csv(output / "entry_audit.csv", index=False)
    manifest = {"strategy": STRATEGY, "config": asdict(cfg), "data_start": str(calendar[0].date()),
                "data_end": str(calendar[-1].date()), "files": [p.name for p in files],
                "sigma_formula": "ATM put IV * sqrt(actual calendar DTE / 365)",
                "strike_formula": "spot * exp(-spread_sigma * sigma_horizon)",
                "execution": "Entry and exit at bid/ask; fee is an assumed per-contract per-side cost.",
                "marking": "Daily mid with exit fee reserve; vertical mid projected into no-arbitrage/quoted bounds. Missing marks carried and flagged.",
                "atm_quality": "Nearest strike: IV in [3%, 300%], put delta in [-0.65, -0.35]; invalid ATM rejected.",
                "expiry": "Close one observed session before source expiry; no official SET or AM/PM identifiers available.",
                "entry": "First eligible session each month; one tail put; standard third-Friday expiry only; skip expiries beyond data end.",
                "series": "Exact expiry/strike matching. Thursday-labelled monthly chains and holiday-shifted expiry are excluded; no date remapping.",
                "risk": "Terminal payoff loss plus reserved commissions; not broker margin. Limits use current equity.",
                "rates": "No cash interest or financing income; Sharpe uses zero cash return.",
                "summary": summary}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    write_report(output, trades, eq, manifest)
    print(json.dumps(summary, indent=2), flush=True)
    return manifest


def figures(trades: pd.DataFrame, eq: pd.DataFrame, package_id: str | None = None) -> dict[str, go.Figure]:
    plots = {}
    fig = go.Figure()
    for col, label, color in [("equity_pnl", "Pakiet", "#087f8c"), ("tail_pnl", "Daleki put", "#b44a78"), ("spread_pnl", "Spready", "#c28b16")]:
        fig.add_scatter(x=eq["quote_date"], y=eq[col], name=label, line_color=color)
    fig.add_scatter(x=eq["quote_date"], y=eq["liquidation_equity"] - (eq["equity"] - eq["equity_pnl"]),
                    name="Likwidacja bid/ask", line=dict(color="#616975", dash="dot"))
    fig.update_layout(title="Wynik i skladowe (USD)")
    plots["equity"] = fig
    fig = go.Figure(go.Scatter(x=eq["quote_date"], y=eq["drawdown"], fill="tozeroy", line_color="#b53e45", name="Drawdown"))
    fig.update_layout(title="Drawdown portfela (USD)")
    plots["drawdown"] = fig
    fig = go.Figure(go.Scatter(x=eq["quote_date"], y=eq["reserved_risk"], name="Ryzyko pakietow", line_color="#087f8c"))
    fig.update_layout(title="Budzet ryzyka (USD)")
    plots["risk"] = fig
    monthly = eq.set_index("quote_date")["daily_pnl"].resample("MS").sum()
    fig = go.Figure(go.Bar(x=monthly.index, y=monthly, marker_color=np.where(monthly >= 0, "#087f8c", "#b53e45")))
    fig.update_layout(title="Miesieczny PnL (USD)")
    plots["monthly"] = fig
    if not trades.empty:
        row = trades[trades["package_id"] == package_id].iloc[0] if package_id else trades.iloc[0]
        start = max(0, min(row["lower_breakeven"] * 0.95, row["tail_strike"] * 0.8))
        spots = np.unique(np.r_[np.linspace(start, row["underlying_entry"] * 1.1, 501),
                               row["tail_strike"], row["short_strike"], row["hedge_strike"],
                               row["lower_breakeven"], row["upper_breakeven"]])
        fig = go.Figure(go.Scatter(x=spots, y=expiry_pnl(row, spots), line_color="#087f8c", name="PnL"))
        for key, name in [("tail_strike", "Long tail"), ("hedge_strike", "Long spread"), ("short_strike", "Short spread")]:
            fig.add_vline(x=row[key], line_dash="dot")
            fig.add_scatter(x=[row[key]], y=expiry_pnl(row, np.array([row[key]])),
                            mode="markers", name=name, marker_size=8)
        fig.add_hline(y=0, line_color="#888888")
        fig.update_layout(title="Wyplata w expiry (USD)", xaxis_title="SPX w expiry")
        plots["payoff"] = fig
        done = trades[trades["status"] == "closed"]
        fig = go.Figure(go.Scatter(x=done["entry_date"], y=done["pnl"], mode="markers",
                                  text=done["package_id"], marker=dict(color=done["spread_count"], colorscale="Viridis", showscale=True,
                                                                    colorbar=dict(title="Liczba<br>spreadow"))))
        fig.update_layout(title="Wyniki pakietow (USD)", xaxis_title="Data wejscia")
        plots["trades"] = fig
    for fig in plots.values():
        fig.update_layout(template="plotly_white", height=400, margin=dict(l=55, r=20, t=55, b=90),
                          font=dict(family="Arial", size=13), title_font_size=16,
                          legend=dict(orientation="h", y=-0.25))
    return plots


def write_report(output: Path, trades: pd.DataFrame, eq: pd.DataFrame, manifest: dict) -> None:
    plots = figures(trades, eq)
    summary, cfg = manifest["summary"], manifest["config"]
    months = pd.to_datetime(eq["quote_date"]).dt.to_period("M").nunique()
    entry_months = pd.to_datetime(trades["entry_date"]).dt.to_period("M").nunique() if not trades.empty else 0
    metrics = [("PnL netto", f"{summary['total_pnl']:,.0f} USD"),
               ("Max drawdown", f"{summary['max_drawdown']:,.0f} USD"),
               ("Max drawdown (%)", f"{summary['max_drawdown_pct']:.2%}"),
               ("Zamkniete pakiety", str(summary["trades"]))]
    sections = ["<!doctype html><html lang='pl'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
                "<title>SPX Digital Tail</title><style>body{font-family:Arial;margin:24px auto;padding:0 16px;max-width:1200px;color:#222;line-height:1.5}h1{font-size:30px}h2{font-size:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}.table{overflow:auto}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;border-block:1px solid #ddd;padding:16px 0}.metrics dt{font-size:13px;color:#555}.metrics dd{margin:0;font-size:22px}.notice{border-left:3px solid #c28b16;padding-left:12px}</style>",
                "<h1>SPX Digital Tail</h1>",
                f"<p>{manifest['data_start']} - {manifest['data_end']} | monthly | {cfg['target_dte']} DTE | delta -{cfg['tail_delta']} | ATM IV {cfg['spread_sigma']} sigma</p>",
                "<dl class='metrics'>" + "".join(f"<div><dt>{label}</dt><dd>{value}</dd></div>" for label, value in metrics) + "</dl>",
                f"<p>Kapital: {cfg['initial_capital']:,.0f} USD. Budzet straty w expiry: {cfg['risk_per_package']:.0%}/pakiet, {cfg['total_risk']:.0%} lacznie. Koszt: {cfg['fee_per_contract']:.2f} USD/kontrakt/strone.</p>",
                f"<p class='notice'>Miesiace z wejsciem: {entry_months}/{months}. Serie tylko z expiry w trzeci piatek. Budzet straty w expiry nie ogranicza kosztu likwidacji po bid/ask.</p>",
                "<p>Wycena dzienna: mid. Wejscia i wyjscia: bid/ask, z prowizjami. Wyplata w expiry jest teoretyczna; backtest zamyka pakiet przed expiry. Skladowe PnL dotycza tych samych wejsc, nie osobnych portfeli.</p>"]
    if summary["trades"] < 30:
        sections.append("<p class='notice'>Mala proba transakcji: wynik nie rozstrzyga hipotezy o premii za ogon.</p>")
    if summary["unresolved_packages"] or summary["stale_mark_sessions"]:
        sections.append("<p class='notice'>Brakujace wyceny lub nierozliczone pakiety: wynik prowizoryczny.</p>")
    for i, fig in enumerate(plots.values()):
        sections.append(pio.to_html(fig, full_html=False, include_plotlyjs=True if i == 0 else False, config={"responsive": True}))
    if not trades.empty:
        sections += ["<h2>Pakiety</h2><div class='table'>", trades[["package_id", "entry_date", "exit_date", "spread_count",
                     "net_credit_after_costs", "risk_budget", "tail_pnl", "spread_pnl", "pnl"]].to_html(index=False, border=0, float_format=lambda v: f"{v:,.2f}"), "</div>"]
    audit_path = output / "entry_audit.csv"
    if audit_path.exists():
        reasons = pd.read_csv(audit_path)["reason"].value_counts().rename_axis("Powod").reset_index(name="Sesje")
        sections += ["<h2>Decyzje o wejsciu</h2><div class='table'>", reasons.to_html(index=False, border=0), "</div>"]
    sections += ["<details><summary>Parametry i ograniczenia danych</summary><pre>", html.escape(json.dumps(manifest, indent=2)), "</pre></details>"]
    # Plotly's responsive pass can retain legend offsets from the old row count.
    sections.append("""<script>
let redrawTimer;
window.addEventListener('resize', () => {
    clearTimeout(redrawTimer);
    redrawTimer = setTimeout(() => {
        document.querySelectorAll('.js-plotly-plot').forEach(plot => Plotly.redraw(plot));
    }, 300);
});
</script></html>""")
    (output / "report.html").write_text("\n".join(sections), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data-parquet"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--fee-per-contract", type=float, default=1)
    parser.add_argument("--package-risk", type=float, default=0.05)
    parser.add_argument("--total-risk", type=float, default=0.20)
    args = parser.parse_args()
    run(args.data_dir, args.out_dir, Config(initial_capital=args.capital, fee_per_contract=args.fee_per_contract,
                                           risk_per_package=args.package_risk, total_risk=args.total_risk),
        args.start_date, args.end_date)


if __name__ == "__main__":
    main()
