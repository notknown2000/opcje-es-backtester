"""Matched-entry experiment: monetize long puts at fixed entry-vol drawdown levels."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import pyarrow.parquet as pq

from taleb_kurtosis.digital_tail import COLUMNS, MonthlyExpiryMap, valid_quote
from taleb_kurtosis.power_law_backtest import OUTPUT as BASELINE, LongBook

OUTPUT = Path("results/power_law_scaleout")
POLICIES = {"30dte": "30 DTE", "tp5": "TP 5x", "scaleout_integer": "Scaleout / cale",
            "scaleout_fractional": "Scaleout / ulamki (teoria)"}
ENTRY_COLUMNS = ["entry_date", "signal_date", "expire_date", "source_expire_date", "strike",
                 "qty", "entry_ask", "entry_cost", "signal_edge", "signal_delta", "signal_sigma"]


def crossed_level(spot, entry_spot, sigma_horizon):
    if not all(np.isfinite(x) and x > 0 for x in [spot, entry_spot, sigma_horizon]):
        return None
    return max(2, math.floor(math.log(entry_spot / spot) / sigma_horizon + 1e-10))


def sell_half(remaining, fractional=False):
    if not np.isfinite(remaining) or remaining < 0:
        raise ValueError("Invalid remaining quantity")
    if not fractional and not float(remaining).is_integer():
        raise ValueError("Whole contracts required")
    return remaining / 2 if fractional else math.floor(remaining / 2)


def update_atm_cache(day, when, cache):
    for expiry, chain in day.groupby("EXPIRE_DATE"):
        spot = float(chain.UNDERLYING_LAST.median())
        if not np.isfinite(spot) or spot <= 0:
            continue
        atm = chain.iloc[(chain.STRIKE - spot).abs().argmin()]
        if (valid_quote(atm) and atm.P_BID > 0 and atm.P_ASK < atm.STRIKE
                and 2 * (atm.P_ASK - atm.P_BID) / (atm.P_ASK + atm.P_BID) <= .5
                and .03 <= atm.P_IV <= 3 and -.65 <= atm.P_DELTA <= -.35):
            cache[expiry] = dict(atm_quote_date=when, atm_strike=float(atm.STRIKE), entry_atm_iv=float(atm.P_IV))


def entry_parameters(ticket, day, conflicts, atm_cache):
    quote = LongBook.quote(ticket, day, conflicts)
    if quote is None or not np.isclose(float(quote.P_ASK), ticket["entry_ask"]):
        raise ValueError(f"Entry quote does not reconcile: {ticket['package_id']}")
    spot = float(quote.UNDERLYING_LAST)
    atm = atm_cache.get(quote.EXPIRE_DATE)
    if atm is None or not 0 <= (ticket["entry_date"] - atm["atm_quote_date"]).days <= 5:
        raise ValueError(f"Missing reliable entry ATM IV: {ticket['package_id']}")
    dte = (ticket["expire_date"] - ticket["entry_date"]).days
    return dict(ticket, entry_spot=spot, **atm, entry_dte=dte,
                atm_iv_carried=atm["atm_quote_date"] != ticket["entry_date"],
                sigma_horizon=atm["entry_atm_iv"] * math.sqrt(dte / 365), entry_bid=float(quote.P_BID))


class ExitBook:
    def __init__(self, selection, policy, capital=250_000., fee=1., exit_dte=30):
        if policy not in POLICIES:
            raise ValueError("Unknown exit policy")
        self.selection, self.policy = selection, policy
        self.name = f"{selection}_{policy}"
        self.label = f"{'Model' if selection == 'power_law' else 'Delta 1%'} / {POLICIES[policy]}"
        self.capital, self.fee, self.exit_dte = capital, fee, exit_dte
        self.cash = capital
        self.active, self.closed, self.fills, self.signals, self.equity, self.paths = [], [], [], [], [], []

    def open(self, ticket, when):
        if ticket["entry_date"] != when or ticket["signal_date"] >= when:
            raise ValueError("Invalid frozen entry timing")
        if ticket["entry_cost"] > self.cash:
            raise ValueError("Matched entry is unaffordable")
        if not np.isclose(ticket["entry_cost"], ticket["qty"] * (ticket["entry_ask"] * 100 + self.fee)):
            raise ValueError("Entry cost does not reconcile")
        self.cash -= ticket["entry_cost"]
        self.active.append(dict(ticket, strategy_id=self.name, remaining=float(ticket["qty"]), proceeds=0.,
                                last_bid=ticket["entry_bid"], last_mark_date=when, missing_mark_days=0,
                                highest_level=2, pending=[], tp_signal=None))

    def sell(self, pos, when, qty, bid, reason, signal_date=None, level=None):
        if qty <= 0 or qty > pos["remaining"] + 1e-9:
            raise ValueError("Invalid sale quantity")
        proceeds = qty * (100 * bid - self.fee)
        pos["remaining"] -= qty
        pos["proceeds"] += proceeds
        self.cash += proceeds
        self.fills.append(dict(strategy_id=self.name, package_id=pos["package_id"], signal_date=signal_date,
                               exit_date=when, reason=reason, level=level, qty=qty, bid=bid,
                               proceeds=proceeds, pnl=proceeds - pos["entry_cost"] * qty / pos["qty"],
                               remaining=pos["remaining"]))

    def observe(self, pos, when, spot):
        marked = pos["remaining"] * (100 * pos["last_bid"] - self.fee)
        z = math.log(pos["entry_spot"] / spot) / pos["sigma_horizon"] if np.isfinite(spot) and spot > 0 else np.nan
        self.paths.append(dict(strategy_id=self.name, package_id=pos["package_id"], quote_date=when,
                               spot=spot, sigma_drop=z, remaining=pos["remaining"], bid=pos["last_bid"],
                               stale=pos["last_mark_date"] != when,
                               package_pnl=pos["proceeds"] + marked - pos["entry_cost"],
                               realized_pnl=pos["proceeds"] - pos["entry_cost"] * (1 - pos["remaining"] / pos["qty"])))

    def step(self, when, day, conflicts, entries=(), final=False):
        spot = float(day.UNDERLYING_LAST.median()) if len(day) else np.nan
        survivors = []
        for pos in self.active:
            quote = LongBook.quote(pos, day, conflicts) if when < pos["expire_date"] else None
            if quote is None:
                pos["missing_mark_days"] += 1
            else:
                bid = float(quote.P_BID)
                pos.update(last_bid=bid, last_mark_date=when)
                due = (pos["expire_date"] - when).days <= self.exit_dte
                reason = "end_of_data" if final else "30dte" if due else "tp5_next_session" if pos["tp_signal"] is not None else None
                if reason:
                    self.sell(pos, when, pos["remaining"], bid, reason,
                              signal_date=pos["tp_signal"] if reason == "tp5_next_session" else None)
                    pos.update(exit_date=when, exit_reason=reason, status="closed", pnl=pos["proceeds"] - pos["entry_cost"])
                    self.observe(pos, when, spot)
                    self.closed.append(pos)
                    continue
                # Every crossed level is persistent. A bounce or missing quote
                # never retroactively erases a prior EOD liquidation instruction.
                for pending in pos["pending"]:
                    amount = sell_half(pos["remaining"], self.policy == "scaleout_fractional")
                    if amount:
                        self.sell(pos, when, amount, bid, f"sigma_{pending['level']}",
                                  pending["signal_date"], pending["level"])
                pos["pending"] = []
                if self.policy == "tp5" and bid >= 5 * pos["entry_ask"]:
                    pos["tp_signal"] = when
            if self.policy.startswith("scaleout") and not final:
                level = crossed_level(spot, pos["entry_spot"], pos["sigma_horizon"])
                if level is not None and level > pos["highest_level"]:
                    for k in range(pos["highest_level"] + 1, level + 1):
                        pending = dict(level=k, signal_date=when)
                        pos["pending"].append(pending)
                        self.signals.append(dict(strategy_id=self.name, package_id=pos["package_id"],
                                                 spot=spot, barrier=pos["entry_spot"] * math.exp(-k * pos["sigma_horizon"]), **pending))
                    pos["highest_level"] = level
            self.observe(pos, when, spot)
            survivors.append(pos)
        self.active = survivors
        for ticket in entries:
            self.open(ticket, when)
            self.observe(self.active[-1], when, spot)
        equity = self.cash + sum(p["remaining"] * (p["last_bid"] * 100 - self.fee) for p in self.active)
        self.equity.append(dict(strategy_id=self.name, label=self.label, quote_date=when, equity=equity,
                                equity_pnl=equity-self.capital, cash=self.cash, open_trades=len(self.active),
                                stale_positions=sum(p["last_mark_date"] != when for p in self.active)))

    def finish(self):
        eq = pd.DataFrame(self.equity)
        peak = eq.equity.cummax().clip(lower=self.capital)
        eq["drawdown"] = eq.equity - peak
        eq["drawdown_pct"] = eq.equity / peak - 1
        trades = [dict(p, status="unresolved", exit_date=pd.NaT, exit_reason="missing_quotes", pnl=np.nan) for p in self.active] + self.closed
        trades = [{k: v for k, v in p.items() if k not in {"pending", "tp_signal"}} for p in trades]
        summary = dict(strategy_id=self.name, construction="power_law_scaleout", selection=self.selection,
                       exit_policy=self.policy, label=self.label, trades=len(self.closed),
                       total_pnl=float(eq.equity_pnl.iloc[-1]), max_drawdown=float(eq.drawdown.min()),
                       max_drawdown_pct=float(eq.drawdown_pct.min()), premium_spent=sum(p["entry_cost"] for p in trades),
                       scaleout_sales=sum(f["level"] is not None for f in self.fills),
                       monetized_packages=len({f["package_id"] for f in self.fills if f["level"] is not None}),
                       triggered_packages=len({f["package_id"] for f in self.signals}),
                       unresolved=len(self.active), stale_sessions=int(eq.stale_positions.gt(0).sum()),
                       initial_capital=self.capital)
        return summary, pd.DataFrame(trades), eq


def figures(equity):
    result = {}
    for column, title in [("equity_pnl", "PnL (USD)"), ("drawdown", "Drawdown (USD)")]:
        fig = px.line(equity, x="quote_date", y=column, color="label", title=title)
        fig.update_layout(height=490, margin=dict(l=30, r=15, t=50, b=130), xaxis_title=None,
                          yaxis_title="USD", legend=dict(orientation="h", y=-.25, title_text=""))
        result[column] = fig
    return result


def run(data_dir=Path("data-parquet"), baseline=BASELINE, output=OUTPUT):
    manifest = json.loads((baseline / "manifest.json").read_text())
    cfg = manifest["config"]
    original = pd.read_parquet(baseline / "trades.parquet")
    expected = pd.read_csv(baseline / "summary.csv").set_index("strategy_id")
    selections = ["power_law", "delta_1pct"]
    tickets = {}
    for selection in selections:
        q = original[original.strategy_id.eq(f"{selection}_30dte")][ENTRY_COLUMNS].sort_values("entry_date").copy()
        q["package_id"] = [f"{selection}-{i+1:03d}" for i in range(len(q))]
        tickets[selection] = {day: rows.to_dict("records") for day, rows in q.groupby("entry_date")}
    books = [ExitBook(s, policy, cfg["initial_capital"], cfg["fee"], cfg["exit_dte"]) for s in selections for policy in POLICIES]
    mapping = MonthlyExpiryMap(pd.Timestamp(manifest["data_start"]), pd.Timestamp(manifest["data_end"]))
    entry_rows = []
    atm_cache = {}
    for index, filename in enumerate(manifest["files"]):
        print(f"[{index+1}/{len(manifest['files'])}] {filename}", flush=True)
        raw = pq.read_table(data_dir / filename, columns=COLUMNS, filters=[("DTE", ">=", 0), ("DTE", "<=", 183)]).to_pandas()
        for col in COLUMNS:
            raw[col] = pd.to_datetime(raw[col]) if col.endswith("DATE") else pd.to_numeric(raw[col], errors="coerce")
        raw = raw[~raw.duplicated(["QUOTE_DATE", "EXPIRE_DATE", "STRIKE"], keep=False)]
        dates = sorted(raw.QUOTE_DATE.unique())
        raw, _ = mapping.normalize(raw)
        for date in dates:
            when = pd.Timestamp(date)
            day = raw[raw.QUOTE_DATE.eq(when)]
            mapping.entry_frame(day)
            update_atm_cache(day, when, atm_cache)
            entries = {}
            for selection in selections:
                entries[selection] = [entry_parameters(t, day, mapping.conflicts, atm_cache) for t in tickets[selection].get(when, [])]
                entry_rows.extend(dict(t, selection=selection) for t in entries[selection])
            for book in books:
                book.step(when, day, mapping.conflicts, entries[book.selection], final=when == pd.Timestamp(manifest["data_end"]))
    summaries, trades, equity = zip(*(b.finish() for b in books))
    summary, trades, equity = pd.DataFrame(summaries), pd.concat(trades, ignore_index=True), pd.concat(equity, ignore_index=True)
    # Replaying control policies must reproduce the original experiment, not a
    # different set of entries that happens to improve the new exit strategy.
    for selection in selections:
        for policy, old_policy in [("30dte", "30dte"), ("tp5", "tp5_or_30dte")]:
            actual = summary[summary.strategy_id.eq(f"{selection}_{policy}")].iloc[0]
            prior = expected.loc[f"{selection}_{old_policy}"]
            for field in ["total_pnl", "max_drawdown", "trades", "premium_spent", "unresolved"]:
                if not np.isclose(actual[field], prior[field], atol=1e-7, rtol=0):
                    raise ValueError(f"Control regression {selection}/{policy}/{field}: {actual[field]} != {prior[field]}")
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv", index=False)
    trades.to_parquet(output / "trades.parquet", index=False)
    equity.to_parquet(output / "daily_equity.parquet", index=False)
    for name, rows in [("sales", [f for b in books for f in b.fills]), ("signals", [s for b in books for s in b.signals]),
                       ("position_paths", [p for b in books for p in b.paths]), ("entries", entry_rows)]:
        pd.DataFrame(rows).to_parquet(output / f"{name}.parquet", index=False)
    info = dict(kind="power_law_scaleout", baseline=str(baseline), config=cfg,
                baseline_trades_sha256=hashlib.sha256((baseline / "trades.parquet").read_bytes()).hexdigest(),
                data_start=manifest["data_start"], data_end=manifest["data_end"], sessions=manifest["sessions"],
                controls_reconciled=True, sigma="entry ATM IV * sqrt(entry calendar DTE / 365)",
                atm_iv_source="Nearest ATM put; last valid observation for exact source expiry, at most 5 calendar days old at entry",
                entries_with_carried_atm_iv=sum(t["atm_iv_carried"] for t in entry_rows),
                trigger="log(entry SPX / current SPX) / frozen entry sigma >= 3,4,5,...; cumulative, not daily shock",
                execution="EOD signal, next available session bid; all crossed levels persist across rebounds and missing quotes",
                sizing="50% of remaining each new level; integer floor retains at least one contract; fractional is theoretical",
                residual="30 DTE or end-of-data, not expiry; same fixed entries, costs and capital as baseline",
                limitations=["No intraday path: crossings only observed at EOD; no sale at a theoretical barrier price.",
                             "Missing quotes carried/flagged; no invented expiry settlement.",
                             "Fixed historical entry tickets isolate exits; this is not a separately reoptimized entry strategy.",
                             "No profit condition on a sigma sale. Fractional quantities/commissions are not executable SPX trades."])
    (output / "manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    body = "<h1>Power Law: scaleout 3 / 4 / 5 sigma</h1><p>50% pozostalej pozycji na kazdym kolejnym progu. "
    body += "Sigma z ATM IV i DTE na wejsciu; sygnal EOD, wykonanie kolejna sesja bid. Reszta: 30 DTE.</p>" + summary.to_html(index=False)
    for i, fig in enumerate(figures(equity[equity.strategy_id.str.startswith("power_law_")]).values()):
        body += fig.to_html(full_html=False, include_plotlyjs=i == 0)
    (output / "report.html").write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>'+body+'</body></html>', encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data-parquet"))
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.data_dir, args.baseline, args.output)
