"""SPX relative-value tail scan and premium-budgeted, next-session long-only test."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import pyarrow.parquet as pq

from taleb_kurtosis.digital_tail import COLUMNS, MonthlyExpiryMap, valid_quote
from taleb_kurtosis.power_law import PricingConfig, prepare_chain, price_vector, score_prepared

OUTPUT = Path("results/power_law_tail")
SIGMA_BINS = [0, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 8, np.inf]
SIGMA_LABELS = ["<2", "2-2.5", "2.5-3", "3-3.5", "3.5-4", "4-4.5", "4.5-5", "5-6", "6-8", "8+"]
DTE_BINS = [59, 89, 104, 135, 180]
DTE_LABELS = ["60-89", "90-104", "105-135", "136-180"]
LABELS = {"power_law_30dte": "Model / 30 DTE", "power_law_tp5_or_30dte": "Model / TP 5x",
          "delta_1pct_30dte": "Delta 1% / 30 DTE", "delta_1pct_tp5_or_30dte": "Delta 1% / TP 5x"}


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 250_000
    monthly_budget_fraction: float = .0025
    target_dte: int = 120
    min_dte: int = 105
    max_dte: int = 135
    exit_dte: int = 30
    fee: float = 1

    def __post_init__(self):
        if not all(np.isfinite(v) for v in vars(self).values()):
            raise ValueError("Nonfinite configuration")
        if self.initial_capital <= 0 or not 0 < self.monthly_budget_fraction <= .1 or self.fee < 0:
            raise ValueError("Invalid premium budget")
        if not 0 <= self.exit_dte < self.min_dte <= self.target_dte <= self.max_dte <= 180:
            raise ValueError("Invalid DTEs")


def bin_observations(frame):
    q = frame.copy()
    q["sigma_bin"] = pd.cut(q.sigma, SIGMA_BINS, labels=SIGMA_LABELS, right=False).astype(str)
    q["dte_bin"] = pd.cut(q.dte, DTE_BINS, labels=DTE_LABELS).astype(str)
    return q


def daily_frequencies(frame):
    if frame.empty:
        return pd.DataFrame()
    q = bin_observations(frame)
    return q.groupby(["QUOTE_DATE", "sigma_bin", "dte_bin"], observed=True, as_index=False).agg(
        quotes=("edge", "size"), frequency=("underpriced", "mean"),
        median_edge=("edge", "median"), median_delta=("P_DELTA", "median"),
        robust_frequency=("robust_underpriced", "mean"))


def aggregate_frequencies(daily):
    # Equal weight to each observed session within a bin, not to dense chains.
    return daily.groupby(["sigma_bin", "dte_bin"], as_index=False).agg(
        days=("QUOTE_DATE", "nunique"), quotes=("quotes", "sum"), frequency=("frequency", "mean"),
        mean_daily_median_edge=("median_edge", "mean"), median_delta=("median_delta", "median"),
        robust_frequency=("robust_frequency", "mean"))


def scan_day(day, when, pricing):
    results, audit, sensitivity = [], [], []
    reference = float(day.UNDERLYING_LAST.median()) if not day.empty else np.nan
    for expiry, chain in day.groupby("EXPIRE_DATE", sort=True):
        dte = (expiry - when).days
        if not 60 <= dte <= 180:
            continue
        prepared, reason = prepare_chain(chain, reference, dte, pricing)
        scored = pd.DataFrame()
        if reason == "ok":
            scored, reason = score_prepared(prepared, reference, pricing)
        audit.append(dict(quote_date=when, expiry=expiry, reason=reason, rows=len(chain), eligible=len(scored)))
        if scored.empty:
            continue
        models = []
        for alpha in [2., 2.75, 3.5]:
            for anchor in [2., 2.5, 3.]:
                prices, _, why = price_vector(prepared, reference, replace(pricing, alpha=alpha, anchor_sigma=anchor))
                eligible = np.isfinite(prices)
                cost = prepared.P_ASK.to_numpy() + 2 * pricing.fee / pricing.multiplier
                sensitivity.append(dict(quote_date=when, expiry=expiry, alpha=alpha, anchor_sigma=anchor,
                                        reason=why, quotes=int(eligible.sum()),
                                        frequency=float(np.mean(prices[eligible] / cost[eligible] - 1 >= pricing.min_edge)) if eligible.any() else np.nan))
                models.append(prices[scored.index.to_numpy()])
        matrix = np.array(models)
        scored["sensitivity_models"] = np.isfinite(matrix).sum(axis=0)
        # Require all nine combinations for a robustness claim.
        scored["model_low"] = np.min(matrix, axis=0)
        scored["model_high"] = np.max(matrix, axis=0)
        scored["robust_underpriced"] = scored.model_low.ge(scored.all_in_ask * (1 + pricing.min_edge))
        results.append(scored)
    return (pd.concat(results, ignore_index=True) if results else pd.DataFrame()), audit, sensitivity


def choose_signal(scored, when, selection, cfg):
    if scored.empty:
        return None
    q = scored[scored.dte.between(cfg.min_dte, cfg.max_dte)].copy()
    if q.empty:
        return None
    expiry = min(q.EXPIRE_DATE.unique(), key=lambda e: (abs((pd.Timestamp(e) - when).days - cfg.target_dte), e))
    q = q[q.EXPIRE_DATE.eq(expiry)]
    if selection == "power_law":
        q = q[q.underpriced]
        if q.empty:
            return None
        q = q.sort_values(["edge", "relative_spread", "STRIKE"], ascending=[False, True, False])
    else:
        q = q[q.P_DELTA.add(.01).abs().le(.0025)].copy()
        if q.empty:
            return None
        q["delta_distance"] = q.P_DELTA.add(.01).abs()
        q = q.sort_values(["delta_distance", "STRIKE"], ascending=[True, False])
    # Choose without knowing tomorrow's price. An unaffordable contract is skipped
    # on execution, never replaced retroactively with a cheaper winning strike.
    row = q.iloc[0]
    return dict(signal_date=when, expire_date=pd.Timestamp(expiry),
                source_expire_date=pd.Timestamp(row.SOURCE_EXPIRE_DATE), strike=float(row.STRIKE),
                signal_sigma=float(row.sigma), signal_delta=float(row.P_DELTA),
                signal_edge=float(row.edge), signal_model=float(row.model_price),
                signal_ask=float(row.P_ASK), anchor_strike=float(row.anchor_strike), max_relative_spread=.5)


class LongBook:
    def __init__(self, selection, take_profit, cfg):
        self.selection, self.take_profit, self.cfg = selection, take_profit, cfg
        self.name = f"{selection}_{'tp5_or_30dte' if take_profit else '30dte'}"
        self.cash = cfg.initial_capital
        self.active, self.closed, self.equity, self.audit = [], [], [], []
        self.pending = None
        self.entered_months = set()

    @staticmethod
    def quote(pos, day, conflicts):
        q = day[(day.EXPIRE_DATE.eq(pos["source_expire_date"])) & day.STRIKE.eq(pos["strike"])]
        if q.empty and pos["expire_date"] not in conflicts:
            alias = day[day.MONTHLY_EXPIRY.eq(pos["expire_date"])]
            if alias.SOURCE_EXPIRE_DATE.nunique() == 1:
                q = alias[alias.STRIKE.eq(pos["strike"])]
        if len(q) != 1 or not valid_quote(q.iloc[0]):
            return None
        return q.iloc[0]

    def close(self, pos, when, bid, reason):
        proceeds = pos["qty"] * (bid * 100 - self.cfg.fee)
        self.cash += proceeds
        pos.update(exit_date=when, exit_bid=bid, exit_reason=reason, status="closed",
                   pnl=proceeds - pos["entry_cost"])
        self.closed.append(pos)

    def step(self, when, raw, scored, conflicts, final=False):
        remaining = []
        for pos in self.active:
            quote = self.quote(pos, raw, conflicts) if when < pos["expire_date"] else None
            due = (pos["expire_date"] - when).days <= self.cfg.exit_dte
            if quote is not None:
                bid = float(quote.P_BID)
                pos.update(last_bid=bid, last_mark_date=when)
                reason = "end_of_data" if final else "30dte" if due else pos.get("pending_exit")
                if reason:
                    self.close(pos, when, bid, reason)
                    continue
                if self.take_profit and bid >= self.take_profit * pos["entry_ask"]:
                    pos["pending_exit"] = "tp5_next_session"
            else:
                pos["missing_mark_days"] += 1
            remaining.append(pos)
        self.active = remaining
        month = str(when.to_period("M"))
        if self.pending and not final:
            signal, self.pending = self.pending, None
            reason = "entered"
            quote = self.quote(signal, raw, conflicts)
            if month in self.entered_months:
                reason = "month_already_entered"
            elif quote is None or float(quote.P_BID) <= 0:
                reason = "missing_execution_quote"
            elif (float(quote.P_ASK) >= signal["strike"] or
                  2 * (float(quote.P_ASK) - float(quote.P_BID)) / (float(quote.P_ASK) + float(quote.P_BID)) > signal.get("max_relative_spread", .5)):
                reason = "invalid_execution_spread"
            elif any(p["last_mark_date"] != when for p in self.active):
                reason = "stale_portfolio"
            elif (signal["expire_date"] - when).days <= self.cfg.exit_dte:
                reason = "too_late"
            else:
                ask = float(quote.P_ASK)
                cost = ask * 100 + self.cfg.fee
                budget = min(self.cfg.initial_capital * self.cfg.monthly_budget_fraction, self.cash)
                qty = int(max(budget, 0) // (cost + self.cfg.fee))
                if qty == 0:
                    reason = "premium_budget"
                else:
                    pos = dict(signal, strategy_id=self.name, entry_date=when, qty=qty, entry_ask=ask,
                               entry_cost=qty * cost, last_bid=float(quote.P_BID), last_mark_date=when,
                               missing_mark_days=0, status="open")
                    self.cash -= pos["entry_cost"]
                    self.active.append(pos)
                    self.entered_months.add(month)
            self.audit.append(dict(strategy_id=self.name, quote_date=when, reason=reason, **signal))
        if month not in self.entered_months and not final:
            self.pending = choose_signal(scored, when, self.selection, self.cfg)
        liquidation = self.cash + sum(p["qty"] * (p["last_bid"] * 100 - self.cfg.fee) for p in self.active)
        self.equity.append(dict(strategy_id=self.name, quote_date=when, equity=liquidation,
                                equity_pnl=liquidation - self.cfg.initial_capital, cash=self.cash,
                                open_trades=len(self.active), stale_positions=sum(p["last_mark_date"] != when for p in self.active)))

    def finish(self):
        trades = self.closed + [dict(p, status="unresolved", exit_reason="missing_quotes", exit_date=pd.NaT,
                                   pnl=np.nan) for p in self.active]
        eq = pd.DataFrame(self.equity)
        peak = eq.equity.cummax().clip(lower=self.cfg.initial_capital)
        eq["drawdown"] = eq.equity - peak
        eq["drawdown_pct"] = eq.equity / peak - 1
        summary = dict(strategy_id=self.name, construction="power_law_tail", selection=self.selection,
                       exit_policy="tp5_or_30dte" if self.take_profit else "30dte", trades=len(self.closed),
                       total_pnl=float(eq.equity_pnl.iloc[-1]), realized_pnl=sum(p["pnl"] for p in self.closed),
                       max_drawdown=float(eq.drawdown.min()), max_drawdown_pct=float(eq.drawdown_pct.min()),
                       premium_spent=sum(p["entry_cost"] for p in trades), unresolved=len(self.active),
                       stale_sessions=int(eq.stale_positions.gt(0).sum()), initial_capital=self.cfg.initial_capital,
                       win_rate=float(np.mean([p["pnl"] > 0 for p in self.closed])) if self.closed else None)
        return summary, pd.DataFrame(trades), eq


def report_figures(summary, equity, frequencies):
    grid = frequencies.pivot(index="sigma_bin", columns="dte_bin", values="frequency").reindex(index=SIGMA_LABELS, columns=DTE_LABELS)
    equity = equity.assign(variant=equity.strategy_id.map(LABELS))
    plots = {
        "frequency": px.imshow(grid * 100, aspect="auto", zmin=0, zmax=100,
                               color_continuous_scale="Cividis", labels=dict(color="%", x="DTE", y="Sigma ATM IV"),
                               title="Czestosc sygnalow (%)"),
        "equity": px.line(equity, x="quote_date", y="equity_pnl", color="variant", title="PnL long-only (USD)"),
        "drawdown": px.line(equity, x="quote_date", y="drawdown", color="variant", title="Drawdown (USD)"),
    }
    for name, fig in plots.items():
        fig.update_layout(height=480, margin=dict(l=30, r=15, t=55, b=110), title_font_size=16)
        if name != "frequency":
            fig.update_layout(legend=dict(orientation="h", y=-.25, title_text=""), xaxis_title=None, yaxis_title="USD")
    return plots


def write_report(output, summary, equity, frequencies, manifest):
    body = "<h1>SPX Power-Law Tail</h1><p>Wzgledna wycena: alpha 2.75, kotwica 2.5 sigma ATM IV. "
    body += "Niedowartosciowanie: model &gt; ask + koszty o co najmniej 10%. To nie prognoza zysku.</p>"
    body += summary.to_html(index=False)
    for i, fig in enumerate(report_figures(summary, equity, frequencies).values()):
        body += fig.to_html(full_html=False, include_plotlyjs=True if i == 0 else False)
    body += frequencies.to_html(index=False) + "<h2>Ograniczenia</h2><ul>"
    body += "".join(f"<li>{s}</li>" for s in manifest["caveats"]) + "</ul>"
    (output / "report.html").write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head><body>' + body + '</body></html>', encoding="utf-8")


def run(data_dir=Path("data-parquet"), output=OUTPUT, cfg=BacktestConfig(), pricing=PricingConfig(), limit=None):
    files = sorted(data_dir.glob("spx_eod_*.parquet"))
    if limit:
        files = files[:limit]
    if not files:
        raise ValueError("No input parquet files")
    dates = sorted({pd.Timestamp(d) for p in files for d in pq.read_table(p, columns=["QUOTE_DATE"]).column(0).unique().to_pylist()})
    mapping = MonthlyExpiryMap(dates[0], dates[-1])
    books = [LongBook(selection, tp, cfg) for selection in ["power_law", "delta_1pct"] for tp in [0, 5]]
    output.mkdir(parents=True, exist_ok=True)
    parts = output / "scan"
    parts.mkdir(exist_ok=True)
    daily, audit, sensitivity, examples = [], [], [], []
    excluded_duplicates = 0
    for index, path in enumerate(files):
        print(f"[{index + 1}/{len(files)}] {path.name}", flush=True)
        raw = pq.read_table(path, columns=COLUMNS, filters=[("DTE", ">=", 0), ("DTE", "<=", 183)]).to_pandas()
        for column in COLUMNS:
            raw[column] = pd.to_datetime(raw[column]) if column.endswith("DATE") else pd.to_numeric(raw[column], errors="coerce")
        duplicates = raw.duplicated(["QUOTE_DATE", "EXPIRE_DATE", "STRIKE"], keep=False)
        excluded_duplicates += int(duplicates.sum())
        raw = raw[~duplicates]
        month_dates = sorted(raw.QUOTE_DATE.unique())
        raw, _ = mapping.normalize(raw)
        observations = []
        for when in month_dates:
            when = pd.Timestamp(when)
            day = raw[raw.QUOTE_DATE.eq(when)]
            entries = mapping.entry_frame(day)
            scored, day_audit, variants = scan_day(entries, when, pricing)
            audit.extend(day_audit)
            sensitivity.extend(variants)
            if not scored.empty:
                observations.append(scored)
                daily.append(daily_frequencies(scored))
                if not examples or when.year != pd.Timestamp(examples[-1].QUOTE_DATE.iloc[0]).year:
                    examples.append(scored)
                last_snapshot = scored
            for book in books:
                book.step(when, day, scored, mapping.conflicts, final=when == dates[-1])
        if observations:
            pd.concat(observations, ignore_index=True).to_parquet(parts / path.name, index=False)
    if not daily:
        raise ValueError("No valid anchored tail quotes")
    daily = pd.concat(daily, ignore_index=True)
    frequencies = aggregate_frequencies(daily)
    summaries, trades, equity = zip(*(book.finish() for book in books))
    summary, trades, equity = pd.DataFrame(summaries), pd.concat(trades, ignore_index=True), pd.concat(equity, ignore_index=True)
    summary.to_csv(output / "summary.csv", index=False)
    trades.to_parquet(output / "trades.parquet", index=False)
    equity.to_parquet(output / "daily_equity.parquet", index=False)
    daily.to_parquet(output / "daily_frequencies.parquet", index=False)
    frequencies.to_csv(output / "frequencies.csv", index=False)
    pd.DataFrame(audit).to_csv(output / "scan_audit.csv", index=False)
    pd.DataFrame(sensitivity).to_parquet(output / "sensitivity.parquet", index=False)
    pd.DataFrame([a for book in books for a in book.audit]).to_csv(output / "entry_audit.csv", index=False)
    pd.concat(examples, ignore_index=True).to_parquet(output / "examples.parquet", index=False)
    last_snapshot.to_parquet(output / "last_snapshot.parquet", index=False)
    manifest = dict(kind="power_law_tail", data_start=str(dates[0].date()), data_end=str(dates[-1].date()),
                    sessions=len(dates), config=asdict(cfg), pricing=asdict(pricing), excluded_duplicate_rows=excluded_duplicates,
                    source="https://arxiv.org/abs/1908.02347", files=[p.name for p in files],
                    caveats=["Relative value, not a physical expected-return estimate; alpha and tail threshold assumed, not fitted.",
                             "SPX source has no AM/PM roots or official SET; monthly aliases follow existing research convention.",
                             "Anchor bid / executable ask plus round-trip fee; quote filter is not a liquidity guarantee.",
                             "First eligible monthly signal, next observed session ask; fixed 0.25% initial-capital monthly premium budget.",
                             "No stocks, short options, cash interest, slippage beyond bid/ask, or market impact.",
                             "Exit 30 DTE or 5x entry ask signal followed by next-session bid. End-of-data liquidation labelled separately.",
                             "Missing marks carried and flagged; missing exits unresolved. Never synthesize settlement.",
                             "Delta control uses the same eligible anchored chains; not an unconditional delta strategy.",
                             "Sensitivity is descriptive, not an out-of-sample parameter optimization."])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(output, summary, equity, frequencies, manifest)
    print(summary.to_string(index=False), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data-parquet"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limit-files", type=int)
    args = parser.parse_args()
    run(args.data_dir, args.output, limit=args.limit_files)
