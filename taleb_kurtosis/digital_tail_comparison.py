"""Paired sigma/exit experiments for the standalone financed SPX tail strategy."""
from __future__ import annotations

import argparse
import html
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pyarrow.parquet as pq

try:
    from taleb_kurtosis.digital_tail import COLUMNS, Config, MonthlyExpiryMap, build_candidate, exit_label, strategy_id
except ModuleNotFoundError as exc:
    if exc.name != "taleb_kurtosis":
        raise
    from digital_tail import COLUMNS, Config, MonthlyExpiryMap, build_candidate, exit_label, strategy_id


DEFAULT_OUTPUT = Path("results/digital_tail_barrier_comparison")


def experiment_configs(base: Config) -> list[Config]:
    return [replace(base, spread_sigma=sigma, expiry_mode="monthly_aliases", exit_policy=policy,
                    profit_target_risk=target)
            for sigma in (2.5, 3.0, 3.5)
            for policy, target in [("pre_expiry_bid_ask", 0.0), ("barrier_close_all", 0.0),
                                   ("barrier_close_all", 0.05), ("barrier_close_all", 0.10)]]


def exit_signal(position: dict, when: pd.Timestamp, spot: float, marked: dict | None, cfg: Config) -> str | None:
    if cfg.exit_policy == "barrier_close_all" and np.isfinite(spot) and spot <= position["short_strike"]:
        return "barrier_close_all"
    if (cfg.profit_target_risk and marked is not None
            and marked["liquidation_pnl"] >= cfg.profit_target_risk * position["risk_budget"]):
        return "package_profit_target"
    if when >= position["planned_exit_date"]:
        return "pre_expiry_bid_ask"
    return None


class DigitalTailPortfolio:
    def __init__(self, cfg: Config, dates: pd.DatetimeIndex, expiry_map: MonthlyExpiryMap):
        self.cfg, self.dates, self.expiry_map = cfg, dates, expiry_map
        self.id = strategy_id(cfg)
        self.active: list[dict] = []
        self.closed: list[dict] = []
        self.equity_rows: list[dict] = []
        self.attempts: list[dict] = []
        self.paths: list[dict] = []
        self.entered_months: set[tuple[int, int]] = set()
        self.realized_tail = self.realized_spread = 0.0

    def record_path(self, pos: dict, when: pd.Timestamp, spot: float, closed: bool = False) -> None:
        self.paths.append({"strategy_id": self.id, "package_id": pos["package_id"], "quote_date": when,
                           "underlying": spot, "pnl": pos["pnl"], "liquidation_pnl": pos["liquidation_pnl"],
                           "tail_liq": pos["tail_liq"], "spread_liq": pos["spread_liq"],
                           "fresh": pos["last_mark_date"] == when, "closed": closed,
                           "source_expire_date": pos["last_source_expire_date"]})

    def step(self, when: pd.Timestamp, day: pd.DataFrame, entry_day: pd.DataFrame,
             lookup: pd.DataFrame, spot: float, candidates: dict) -> None:
        cfg = self.cfg
        remaining = []
        for pos in self.active:
            marked = self.expiry_map.mark(pos, day, lookup, cfg) if when < pos["expire_date"] else None
            if marked is not None:
                pos.update(marked, last_mark_date=when)
            else:
                pos["missing_mark_days"] += 1
            signal = exit_signal(pos, when, spot, marked, cfg)
            if signal and pos["pending_exit_reason"] is None:
                pos.update(pending_exit_reason=signal, exit_signal_date=when)
            if pos["pending_exit_reason"] and marked is not None:
                pos.update(exit_date=when, exit_reason=pos["pending_exit_reason"], status="closed",
                           pnl=marked["liquidation_pnl"], tail_pnl=marked["tail_liq"], spread_pnl=marked["spread_liq"],
                           exit_underlying=spot, exit_dte=(pos["expire_date"] - when).days)
                self.realized_tail += pos["tail_pnl"]
                self.realized_spread += pos["spread_pnl"]
                self.closed.append(pos)
                self.record_path(pos, when, spot, closed=True)
            else:
                remaining.append(pos)
                self.record_path(pos, when, spot)
        self.active = remaining
        month = (when.year, when.month)
        if month not in self.entered_months:
            # Construction is identical across exit policies; cache today's three candidates.
            if cfg.spread_sigma not in candidates:
                candidates[cfg.spread_sigma] = build_candidate(entry_day, when, cfg)
            template, reason = candidates[cfg.spread_sigma]
            detail = {"strategy_id": self.id, "quote_date": when, "reason": reason}
            if template is not None:
                candidate = {**template, "strategy_id": self.id}
                detail.update({k: candidate[k] for k in ["expire_date", "source_expire_date", "risk_budget",
                                                       "spread_count", "net_credit_after_costs"]})
                equity = cfg.initial_capital + self.realized_tail + self.realized_spread + sum(x["pnl"] for x in self.active)
                if any(x["last_mark_date"] != when for x in self.active):
                    reason = "stale_portfolio_marks"
                elif candidate["risk_budget"] > cfg.risk_per_package * max(equity, 0):
                    reason = "package_risk_limit"
                elif candidate["risk_budget"] + sum(x["risk_budget"] for x in self.active) > cfg.total_risk * max(equity, 0):
                    reason = "portfolio_risk_limit"
                elif candidate["expire_date"] > self.dates[-1]:
                    reason = "expiry_beyond_data"
                else:
                    candidate.update(package_id=f"{self.id}-{len(self.closed) + len(self.active) + 1}",
                                     planned_exit_date=self.expiry_map.sessions[self.expiry_map.sessions < candidate["expire_date"]][-1],
                                     missing_mark_days=0, last_mark_date=when, source_label_switches=0,
                                     last_source_expire_date=candidate["source_expire_date"],
                                     pending_exit_reason=None, exit_signal_date=pd.NaT)
                    marked = self.expiry_map.mark(candidate, day, lookup, cfg)
                    if marked is None:
                        reason = "invalid_entry_mark"
                    else:
                        candidate.update(marked)
                        self.active.append(candidate)
                        self.entered_months.add(month)
                        self.record_path(candidate, when, spot)
                        reason = "entered"
                detail["reason"] = reason
            self.attempts.append(detail)
        tail = self.realized_tail + sum(x["tail_pnl"] for x in self.active)
        spread = self.realized_spread + sum(x["spread_pnl"] for x in self.active)
        self.equity_rows.append({"strategy_id": self.id, "quote_date": when, "equity_pnl": tail + spread,
                                "tail_pnl": tail, "spread_pnl": spread, "equity": cfg.initial_capital + tail + spread,
                                "liquidation_equity": cfg.initial_capital + self.realized_tail + self.realized_spread + sum(x["liquidation_pnl"] for x in self.active),
                                "realized_pnl": self.realized_tail + self.realized_spread,
                                "unrealized_pnl": sum(x["pnl"] for x in self.active), "open_trades": len(self.active),
                                "reserved_risk": sum(x["risk_budget"] for x in self.active),
                                "stale_packages": sum(x["last_mark_date"] != when for x in self.active),
                                "projected_marks": sum(x["mark_projected"] for x in self.active)})

    def finish(self) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
        cfg = self.cfg
        for pos in self.active:
            pos.update(exit_date=pd.NaT, exit_reason="unresolved_missing_quotes", status="unresolved")
        trades = pd.DataFrame(self.closed + self.active)
        eq = pd.DataFrame(self.equity_rows)
        peak = eq["equity"].cummax().clip(lower=cfg.initial_capital)
        eq["drawdown"] = eq["equity"] - peak
        eq["drawdown_pct"] = eq["equity"] / peak - 1
        eq["liquidation_drawdown"] = eq["liquidation_equity"] - eq["liquidation_equity"].cummax().clip(lower=cfg.initial_capital)
        eq["daily_pnl"] = eq["equity_pnl"].diff().fillna(eq["equity_pnl"].iloc[0])
        done = pd.DataFrame(self.closed)
        summary = {"strategy_id": self.id, "construction": "digital_tail", "spread_sigma": cfg.spread_sigma,
                   "exit_policy": cfg.exit_policy, "profit_target_risk": cfg.profit_target_risk,
                   "exit_label": exit_label(cfg), "label": f"{cfg.spread_sigma:g} sigma | {exit_label(cfg)}",
                   "expiry_mode": cfg.expiry_mode, "entry_frequency": "monthly", "sigma_source": "atm_iv_actual_dte",
                   "initial_capital": cfg.initial_capital, "trades": len(done), "total_pnl": float(eq["equity_pnl"].iloc[-1]),
                   "realized_pnl": self.realized_tail + self.realized_spread,
                   "max_drawdown": float(eq["drawdown"].min()), "max_drawdown_pct": float(eq["drawdown_pct"].min()),
                   "liquidation_max_drawdown": float(eq["liquidation_drawdown"].min()),
                   "win_rate": float(done["pnl"].gt(0).mean()) if len(done) else None,
                   "tail_pnl": float(eq["tail_pnl"].iloc[-1]), "spread_pnl": float(eq["spread_pnl"].iloc[-1]),
                   "barrier_exits": sum(x["exit_reason"] == "barrier_close_all" for x in self.closed),
                   "profit_exits": sum(x["exit_reason"] == "package_profit_target" for x in self.closed),
                   "expiry_exits": sum(x["exit_reason"] == "pre_expiry_bid_ask" for x in self.closed),
                   "delayed_exits": sum(x["exit_signal_date"] < x["exit_date"] for x in self.closed),
                   "unresolved_packages": len(self.active), "stale_mark_sessions": int(eq["stale_packages"].gt(0).sum()),
                   "projected_mark_sessions": int(eq["projected_marks"].gt(0).sum()),
                   "source_label_switches": sum(x["source_label_switches"] for x in self.closed + self.active),
                   "max_open_packages": int(eq["open_trades"].max()), "max_reserved_risk": float(eq["reserved_risk"].max()),
                   "active_sessions": int(eq["open_trades"].gt(0).sum())}
        return summary, trades, eq


def paired_exits(trades: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Exit effect only on identical entries and financing quantities, per sigma."""
    rows = []
    keys = ["entry_date", "expire_date", "source_expire_date", "tail_strike", "short_strike", "hedge_strike", "spread_count"]
    if trades.empty:
        return pd.DataFrame()
    done = trades[trades["status"].eq("closed")]
    for sigma, variants in summary.groupby("spread_sigma"):
        control_id = variants.loc[variants["exit_policy"].eq("pre_expiry_bid_ask"), "strategy_id"].iloc[0]
        control = done[done["strategy_id"].eq(control_id)]
        for _, variant in variants[variants["strategy_id"].ne(control_id)].iterrows():
            other = done[done["strategy_id"].eq(variant["strategy_id"])]
            pairs = other.merge(control, on=keys, suffixes=("_variant", "_control"), validate="one_to_one")
            for _, p in pairs.iterrows():
                rows.append({"strategy_id": variant["strategy_id"], "spread_sigma": sigma, "exit_label": variant["exit_label"],
                             "entry_date": p["entry_date"], "package_id": p["package_id_variant"],
                             "control_package_id": p["package_id_control"], "exit_date": p["exit_date_variant"],
                             "control_exit_date": p["exit_date_control"], "exit_reason": p["exit_reason_variant"],
                             "pnl": p["pnl_variant"], "control_pnl": p["pnl_control"],
                             "pnl_difference": p["pnl_variant"] - p["pnl_control"]})
    return pd.DataFrame(rows)


def run_comparison(data_dir: Path, output: Path, base: Config, start: str | None = None, end: str | None = None) -> dict:
    files = sorted(data_dir.glob("spx_eod_*.parquet"))
    files = [p for p in files if (not start or p.stem[-6:] >= start[:7].replace("-", ""))
             and (not end or p.stem[-6:] <= end[:7].replace("-", ""))]
    if not files:
        raise ValueError("No input files")
    dates = set()
    file_dates = {}
    for p in files:
        file_dates[p] = pq.read_table(p, columns=["QUOTE_DATE"])["QUOTE_DATE"].unique().to_pylist()
        dates.update(file_dates[p])
    calendar = pd.DatetimeIndex(sorted(dates))
    calendar = calendar[(calendar >= pd.Timestamp(start or calendar[0])) & (calendar <= pd.Timestamp(end or calendar[-1]))]
    if calendar.empty:
        raise ValueError("No sessions in requested range")
    expiry_map = MonthlyExpiryMap(calendar[0], calendar[-1])
    configs = experiment_configs(base)
    engines = [DigitalTailPortfolio(cfg, calendar, expiry_map) for cfg in configs]
    mappings = []
    duplicates = 0
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name}", flush=True)
        raw = pq.read_table(path, columns=COLUMNS, filters=[("DTE", ">=", 0), ("DTE", "<=", base.max_dte + 4)]).to_pandas()
        for col in ["QUOTE_DATE", "EXPIRE_DATE"]:
            raw[col] = pd.to_datetime(raw[col])
        for col in COLUMNS[2:]:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        raw = raw[raw["QUOTE_DATE"].isin(calendar)]
        duplicate_mask = raw.duplicated(["QUOTE_DATE", "EXPIRE_DATE", "STRIKE"], keep=False)
        duplicates += int(duplicate_mask.sum())
        raw = raw[~duplicate_mask]
        monthly, audit = expiry_map.normalize(raw)
        mappings.append(audit)
        by_day = {when: day for when, day in raw.groupby("QUOTE_DATE")}
        for when in sorted(pd.Timestamp(d) for d in file_dates[path] if pd.Timestamp(d) in calendar):
            all_day = by_day.get(when, raw.iloc[:0])
            day = monthly[monthly["QUOTE_DATE"].eq(when)]
            entry_day = expiry_map.entry_frame(day)
            lookup = day.set_index(["EXPIRE_DATE", "STRIKE"])
            spot = float(all_day["UNDERLYING_LAST"].median())
            candidates: dict = {}
            for engine in engines:
                engine.step(when, day, entry_day, lookup, spot, candidates)
    summaries, trade_frames, equities = [], [], []
    for engine in engines:
        summary, trades, equity = engine.finish()
        summaries.append(summary)
        if not trades.empty:
            trade_frames.append(trades)
        equities.append(equity)
    summary = pd.DataFrame(summaries)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(columns=["strategy_id", "package_id", "status"])
    equity = pd.concat(equities, ignore_index=True)
    audit = pd.DataFrame([r for engine in engines for r in engine.attempts])
    paths = pd.DataFrame([r for engine in engines for r in engine.paths])
    mapping = pd.concat(mappings, ignore_index=True)
    paired = paired_exits(trades, summary)
    manifest = {"kind": "digital_tail_comparison", "data_start": str(calendar[0].date()), "data_end": str(calendar[-1].date()),
                "configs": {strategy_id(c): asdict(c) for c in configs}, "files": [p.name for p in files],
                "execution": "All legs closed together at same EOD bid/ask snapshot; fees per contract per side. No intraday touch/fill model.",
                "barrier": "Fixed entry short-put strike. Trigger when EOD SPX <= strike; exit even at a package loss.",
                "profit_target": "5% and 10% of initial terminal-loss budget, tested on net liquidation PnL of the whole package, not long-only PnL.",
                "expiry": "Fallback exit on CBOE session before modelled monthly AM settlement. No official SET settlement used.",
                "series": "Research convention: monthly last-trading-day, settlement-day and historical Saturday labels mapped to CBOE monthly settlement. Raw expiry retained for marking. Simultaneous labels excluded from new entries; existing positions keep their current raw label. Switch only when it disappears and a sole alias remains, before any observed label conflict. AM/PM root identity cannot be verified from these files.",
                "missing_quotes": "Carry and flag missing marks. Exit signal remains pending until all legs have valid quotes. Never invent a settlement or fill after expiry. Stale positions block new entries.",
                "risk": "Same 100% integer unit construction, capital and 5%/20% default limits for every variant. Terminal-loss budget is not broker margin or a cap on bid/ask liquidation loss.",
                "comparison": "Independent portfolios may choose different entry dates after exits change capital/risk availability. paired_exits.csv isolates identical entries against each sigma's hold control; it is not a standalone portfolio.",
                "sigma_formula": "ATM put IV * sqrt(actual calendar DTE / 365); Kshort nearest spot * exp(-sigma * horizon).",
                "rates": "No cash yield or financing income. One tail contract per entry; no volatility/skew timing filter.",
                "ambiguous_quote_rows_excluded": duplicates,
                "simultaneous_label_days": int(mapping.loc[mapping["ambiguous"], ["QUOTE_DATE", "MONTHLY_EXPIRY"]].drop_duplicates().shape[0]),
                "summary": summaries}
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv", index=False)
    trades.to_parquet(output / "trades.parquet", index=False)
    equity.to_parquet(output / "daily_equity.parquet", index=False)
    paths.to_parquet(output / "package_marks.parquet", index=False)
    audit.to_csv(output / "entry_audit.csv", index=False)
    mapping.to_csv(output / "expiry_mapping.csv", index=False)
    paired.to_csv(output / "paired_exits.csv", index=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    write_comparison_report(output, summary, equity, manifest)
    print(summary[["label", "trades", "total_pnl", "max_drawdown", "barrier_exits", "profit_exits", "unresolved_packages"]].to_string(index=False), flush=True)
    return manifest


def comparison_figures(summary: pd.DataFrame, equity: pd.DataFrame) -> dict[str, go.Figure]:
    plots = {}
    colors = ["#087f8c", "#b44a78", "#c28b16", "#5264a2"]
    for col, title in [("equity_pnl", "PnL netto (USD)"), ("drawdown", "Drawdown (USD)"), ("reserved_risk", "Budzet ryzyka (USD)")]:
        fig = go.Figure()
        for i, (_, row) in enumerate(summary.iterrows()):
            eq = equity[equity["strategy_id"].eq(row["strategy_id"])]
            fig.add_scatter(x=eq["quote_date"], y=eq[col], name=row["label"], line_color=colors[i % len(colors)])
        fig.update_layout(title=title, template="plotly_white", height=440, title_font_size=16,
                          margin=dict(l=55, r=15, t=50, b=120), font=dict(size=12),
                          legend=dict(orientation="h", y=-0.3))
        plots[col] = fig
    return plots


def package_figures(trade: pd.Series, paths: pd.DataFrame) -> dict[str, go.Figure]:
    fig = go.Figure(go.Scatter(x=paths["quote_date"], y=paths["underlying"], name="SPX", line_color="#087f8c"))
    for col, name, color in [("short_strike", "Bariera", "#b53e45"), ("hedge_strike", "Hedge spreadu", "#c28b16"), ("tail_strike", "Long tail", "#5264a2")]:
        fig.add_scatter(x=paths["quote_date"], y=[trade[col]] * len(paths), name=name, line=dict(color=color, dash="dot"))
    pnl = go.Figure()
    for col, name, color in [("liquidation_pnl", "Pakiet", "#087f8c"), ("tail_liq", "Long tail", "#b44a78"), ("spread_liq", "Spready", "#c28b16")]:
        pnl.add_scatter(x=paths["quote_date"], y=paths[col].where(paths["fresh"]), name=name, line_color=color)
    for chart, title in [(fig, "SPX i bariera pakietu"), (pnl, "PnL likwidacyjny (USD)")]:
        if pd.notna(trade.get("exit_date")):
            chart.add_vline(x=pd.Timestamp(trade["exit_date"]).timestamp() * 1000, line_dash="dash", line_color="#666666")
        chart.update_layout(title=title, template="plotly_white", height=420, title_font_size=16,
                            margin=dict(l=55, r=15, t=50, b=100), legend=dict(orientation="h", y=-0.3))
    return {"spot": fig, "pnl": pnl}


def write_comparison_report(output: Path, summary: pd.DataFrame, equity: pd.DataFrame, manifest: dict) -> None:
    cfg = next(iter(manifest["configs"].values()))
    sections = ["<!doctype html><html lang='pl'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
                "<title>SPX Digital Tail: porownanie</title><style>body{font:15px Arial;line-height:1.5;max-width:1200px;margin:24px auto;padding:0 16px;color:#222}h1{font-size:28px}h2{font-size:21px}pre{white-space:pre-wrap;overflow-wrap:anywhere}.table{overflow:auto}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}</style>",
                "<h1>SPX Digital Tail: bariera i realizacja zysku</h1>",
                f"<p>{manifest['data_start']} - {manifest['data_end']} | ATM IV | 120 DTE | delta -0.01</p>",
                f"<p>Kapital: {cfg['initial_capital']:,.0f} USD | limit/pakiet: {cfg['risk_per_package']:.0%} | limit laczny: {cfg['total_risk']:.0%} | koszt: {cfg['fee_per_contract']:.2f} USD/kontrakt/strone</p>",
                "<p>Bariera: EOD SPX &lt;= strike short puta. Zamkniecie calego pakietu po bid/ask tej sesji. TP: 5% / 10% poczatkowego budzetu ryzyka. Brak symulacji intraday.</p>",
                "<p>Daty serii sa dopasowywane wedlug jawnej konwencji badawczej; pliki nie identyfikuja AM/PM. Nierozliczone pakiety i nieswieze wyceny oznaczaja wynik prowizoryczny: przeniesione wyceny blokuja kolejne wejscia. Takie kontrole nie sa pelnym benchmarkiem. Zero transakcji nie jest wynikiem aktywnej strategii.</p>",
                "<div class='table'>", summary[["label", "trades", "total_pnl", "max_drawdown", "barrier_exits", "profit_exits", "stale_mark_sessions", "unresolved_packages"]].to_html(index=False, border=0, float_format=lambda x: f"{x:,.2f}"), "</div>"]
    first = True
    for sigma, group in summary.groupby("spread_sigma"):
        sections.append(f"<h2>{sigma:g} sigma</h2>")
        for fig in list(comparison_figures(group, equity).values())[:2]:
            sections.append(pio.to_html(fig, full_html=False, include_plotlyjs=first, config={"responsive": True}))
            first = False
    sections += ["<details><summary>Zalozenia i parametry</summary><pre>", html.escape(json.dumps(manifest, indent=2)), "</pre></details>",
                 "<script>let timer;addEventListener('resize',()=>{clearTimeout(timer);timer=setTimeout(()=>document.querySelectorAll('.js-plotly-plot').forEach(p=>Plotly.redraw(p)),300)});</script></html>"]
    (output / "report.html").write_text("\n".join(sections), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data-parquet"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--package-risk", type=float, default=0.05)
    parser.add_argument("--total-risk", type=float, default=0.20)
    parser.add_argument("--fee-per-contract", type=float, default=1)
    args = parser.parse_args()
    run_comparison(args.data_dir, args.out_dir, Config(initial_capital=args.capital, risk_per_package=args.package_risk,
                                                      total_risk=args.total_risk, fee_per_contract=args.fee_per_contract),
                   args.start_date, args.end_date)


if __name__ == "__main__":
    main()
