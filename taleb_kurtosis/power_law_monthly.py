"""Monthly 6-8 sigma long puts, matched 30-DTE and lagged daily-crash exits."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import plotly.express as px
import pyarrow.parquet as pq

from taleb_kurtosis.digital_tail import COLUMNS, valid_quote
from taleb_kurtosis.power_law import PricingConfig, prepare_chain, price_vector

OUTPUT = Path("results/power_law_monthly_6_8sigma_mid")
POLICIES = {"30dte": "30 DTE", "crash5_or_30dte": "Spadek >5 sigma / 30 DTE"}


@dataclass(frozen=True)
class Config:
    initial_capital: float = 250_000.
    monthly_budget: float = 1_000.
    fee: float = 1.
    min_sigma: float = 6.
    max_sigma: float = 8.
    min_dte: int = 165
    target_dte: int = 180
    max_dte: int = 195
    exit_dte: int = 30
    retry_sessions: int = 5
    crash_sigma: float = 5.

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Finite parameters required")
        if not 0 < self.monthly_budget <= self.initial_capital or self.fee < 0:
            raise ValueError("Invalid capital, budget or fee")
        if not 3 < self.min_sigma < self.max_sigma or self.crash_sigma <= 0:
            raise ValueError("Invalid sigma thresholds")
        if not 0 <= self.exit_dte < self.min_dte <= self.target_dte <= self.max_dte:
            raise ValueError("Invalid DTE window")
        for name in ["min_dte", "target_dte", "max_dte", "exit_dte", "retry_sessions"]:
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError("Session and DTE parameters must be nonnegative integers")


def reference_spot(day):
    spots = day.UNDERLYING_LAST.to_numpy(dtype=float)
    return float(np.median(spots)) if len(spots) and np.isfinite(spots).all() and (spots > 0).all() else np.nan


def prepare(chain, when, expiry, pricing):
    spot = reference_spot(chain)
    if not np.isfinite(spot) or (chain.UNDERLYING_LAST.max() - chain.UNDERLYING_LAST.min()) / spot > .001:
        return pd.DataFrame(), "inconsistent_spot", spot
    q, reason = prepare_chain(chain, spot, (expiry - when).days, pricing)
    return q, reason, spot


def select_signal(day, when, cfg, pricing):
    """Select using this close only, without knowing the next execution price."""
    choices, audit = [], []
    for expiry, chain in day.groupby("EXPIRE_DATE", sort=True):
        dte = (expiry - when).days
        if not cfg.min_dte <= dte <= cfg.max_dte:
            continue
        q, reason, spot = prepare(chain, when, expiry, pricing)
        record = dict(quote_date=when, expiry=expiry, reason=reason, candidates=0, underpriced=0)
        if reason == "ok":
            prices, anchor_index, reason = price_vector(q, spot, pricing)
            record["reason"] = reason
            if reason == "ok":
                anchor = q.iloc[anchor_index]
                model = prices * (anchor.P_BID + anchor.P_ASK) / (2 * anchor.P_BID)
                keep = np.isfinite(model) & q.sigma.between(cfg.min_sigma, cfg.max_sigma).to_numpy()
                selected = q[keep].copy()
                mid = (selected.P_BID + selected.P_ASK) / 2
                selected["model_mid"] = model[keep]
                selected["mid"] = mid
                selected["discount"] = 1 - (mid + 2 * cfg.fee / 100) / selected.model_mid
                record.update(candidates=len(selected), underpriced=int(selected.discount.gt(0).sum()))
                for row in selected[selected.discount.gt(0)].itertuples():
                    choices.append(dict(signal_date=when, signal_month=str(when.to_period("M")),
                                        expiry=expiry, strike=float(row.STRIKE), signal_sigma=float(row.sigma),
                                        signal_atm_iv=float(row.atm_iv), signal_spot=spot, signal_dte=dte,
                                        signal_mid=float(row.mid), signal_model_mid=float(row.model_mid),
                                        signal_discount=float(row.discount), signal_delta=float(row.P_DELTA),
                                        signal_spread=float(row.relative_spread), anchor_strike=float(anchor.STRIKE),
                                        anchor_mid=float((anchor.P_BID + anchor.P_ASK) / 2),
                                        limit_mid=float(row.model_mid - 2 * cfg.fee / 100)))
        audit.append(record)
    if not choices:
        return None, audit
    choices.sort(key=lambda p: (-p["signal_discount"], p["signal_spread"],
                               abs(p["signal_dte"] - cfg.target_dte), p["expiry"], p["strike"]))
    return choices[0], audit


def quote_index(day):
    return day[~day.duplicated(["EXPIRE_DATE", "STRIKE"], keep=False)].set_index(["EXPIRE_DATE", "STRIKE"])


def quote_for(index, pos, when):
    if when >= pos["expiry"] or (pos["expiry"], pos["strike"]) not in index.index:
        return None
    quote = index.loc[(pos["expiry"], pos["strike"])]
    if not valid_quote(quote) or quote.P_ASK >= pos["strike"]:
        return None
    return quote


def atm_snapshot(day, when, expiries, pricing):
    result = {}
    for expiry, chain in day[day.EXPIRE_DATE.isin(expiries)].groupby("EXPIRE_DATE"):
        if expiry <= when:
            continue
        q, reason, _ = prepare(chain, when, expiry, pricing)
        if reason == "ok":
            result[expiry] = dict(date=when, iv=float(q.atm_iv.iloc[0]))
    return result


def daily_crash(log_return, previous_iv, threshold=5.):
    if not np.isfinite(log_return) or not np.isfinite(previous_iv) or previous_iv <= 0:
        return False, np.nan
    z = -log_return / (previous_iv / math.sqrt(252))
    return z > threshold, z


class Book:
    def __init__(self, policy, cfg):
        if policy not in POLICIES:
            raise ValueError("Unknown exit policy")
        self.policy, self.cfg = policy, cfg
        self.cash = cfg.initial_capital
        self.active, self.closed, self.equity, self.events, self.paths = [], [], [], [], []

    def open(self, ticket, when):
        if ticket["signal_date"] >= when or ticket["entry_date"] != when:
            raise ValueError("Entry must follow the signal")
        if ticket["entry_cost"] > self.cash:
            raise ValueError("Insufficient cash for matched entry")
        self.cash -= ticket["entry_cost"]
        self.active.append(dict(ticket, strategy_id=self.policy, last_mid=ticket["entry_mid"],
                                last_mark_date=when, missing_mark_days=0, missing_risk_days=0,
                                pending_crash=None, status="open"))

    def step(self, when, index, log_return, previous_atm, previous_session, final=False):
        remaining = []
        for pos in self.active:
            quote = quote_for(index, pos, when)
            reason = None
            due = (pos["expiry"] - when).days <= self.cfg.exit_dte
            if quote is not None:
                mid = float((quote.P_BID + quote.P_ASK) / 2)
                pos.update(last_mid=mid, last_mark_date=when)
                reason = "crash5_next_session" if pos["pending_crash"] else "30dte" if due else "end_of_data" if final else None
                if reason:
                    proceeds = pos["qty"] * (mid * 100 - self.cfg.fee)
                    self.cash += proceeds
                    pos.update(exit_date=when, exit_mid=mid, exit_bid=float(quote.P_BID), exit_ask=float(quote.P_ASK),
                               exit_spread=float((quote.P_ASK - quote.P_BID) / mid), exit_reason=reason,
                               exit_signal_date=pos["pending_crash"] or when, exit_proceeds=proceeds,
                               pnl=proceeds - pos["entry_cost"], status="closed")
            else:
                pos["missing_mark_days"] += 1
            if not reason and self.policy == "crash5_or_30dte" and not final and not pos["pending_crash"]:
                previous = previous_atm.get(pos["expiry"])
                iv = previous["iv"] if previous and previous["date"] == previous_session else np.nan
                hit, z = daily_crash(log_return, iv, self.cfg.crash_sigma)
                if not np.isfinite(z):
                    pos["missing_risk_days"] += 1
                if hit:
                    pos["pending_crash"] = when
                    self.events.append(dict(entry_id=pos["entry_id"], quote_date=when, expiry=pos["expiry"],
                                            daily_log_return=log_return, previous_atm_iv=iv,
                                            previous_atm_date=previous_session, daily_drop_sigma=z))
            marked_pnl = pos["qty"] * (100 * pos["last_mid"] - self.cfg.fee) - pos["entry_cost"]
            self.paths.append(dict(strategy_id=self.policy, entry_id=pos["entry_id"], quote_date=when,
                                   mid=pos["last_mid"], pnl=marked_pnl, stale=pos["last_mark_date"] != when))
            if reason:
                self.closed.append(pos)
            else:
                remaining.append(pos)
        self.active = remaining

    def record_equity(self, when):
        marked = sum(p["qty"] * (100 * p["last_mid"] - self.cfg.fee) for p in self.active)
        self.equity.append(dict(strategy_id=self.policy, quote_date=when, cash=self.cash,
                                equity=self.cash + marked, equity_pnl=self.cash + marked - self.cfg.initial_capital,
                                open_trades=len(self.active), stale_positions=sum(p["last_mark_date"] != when for p in self.active)))

    def finish(self):
        trades = self.closed + [dict(p, status="unresolved", exit_date=pd.NaT, exit_reason="missing_quotes", pnl=np.nan)
                                for p in self.active]
        eq = pd.DataFrame(self.equity)
        peaks = eq.equity.cummax().clip(lower=self.cfg.initial_capital)
        eq["drawdown"] = eq.equity - peaks
        eq["drawdown_pct"] = eq.equity / peaks - 1
        eq["daily_pnl"] = eq.equity.diff().fillna(eq.equity - self.cfg.initial_capital)
        pnl = [p["pnl"] for p in self.closed]
        return dict(strategy_id=self.policy, entries=len(trades), closed_trades=len(pnl),
                    total_pnl=float(eq.equity_pnl.iloc[-1]), realized_pnl=float(sum(pnl)),
                    max_drawdown=float(eq.drawdown.min()), max_drawdown_pct=float(eq.drawdown_pct.min()),
                    win_rate=float(np.mean(np.array(pnl) > 0)) if pnl else np.nan,
                    premium_spent=sum(p["entry_cost"] for p in trades),
                    crash_exits=sum(p["exit_reason"] == "crash5_next_session" for p in self.closed),
                    dte_exits=sum(p["exit_reason"] == "30dte" for p in self.closed),
                    end_of_data_exits=sum(p["exit_reason"] == "end_of_data" for p in self.closed),
                    unresolved=len(self.active), stale_sessions=int(eq.stale_positions.gt(0).sum()),
                    missing_risk_days=sum(p["missing_risk_days"] for p in trades)), pd.DataFrame(trades), eq


class Experiment:
    def __init__(self, cfg=Config(), pricing=None):
        self.cfg = cfg
        self.pricing = pricing or PricingConfig(alpha=3., anchor_sigma=3., min_edge=0., fee=cfg.fee)
        self.books = [Book(policy, cfg) for policy in POLICIES]
        self.previous_atm, self.previous_spot, self.previous_session = {}, np.nan, None
        self.pending = None
        self.entered_months = set()
        self.entries, self.entry_audit, self.scan_audit, self.market = [], [], [], []

    def step(self, when, day, session_in_month, final=False):
        index = quote_index(day)
        spot = reference_spot(day)
        log_return = math.log(spot / self.previous_spot) if np.isfinite(spot) and np.isfinite(self.previous_spot) else np.nan
        for book in self.books:
            book.step(when, index, log_return, self.previous_atm, self.previous_session, final)
        month = str(when.to_period("M"))
        if self.pending is not None:
            signal, self.pending = self.pending, None
            quote = quote_for(index, signal, when)
            reason = "entered"
            if final or signal["signal_month"] != month or month in self.entered_months:
                reason = "execution_window_closed"
            elif quote is None:
                reason = "missing_execution_quote"
            else:
                mid = float((quote.P_BID + quote.P_ASK) / 2)
                if quote.P_BID <= 0 or (quote.P_ASK - quote.P_BID) / mid > self.pricing.max_relative_spread:
                    reason = "invalid_execution_spread"
                elif mid >= signal["limit_mid"]:
                    reason = "mid_above_frozen_model_limit"
                elif (signal["expiry"] - when).days <= self.cfg.exit_dte:
                    reason = "too_late"
                else:
                    budget = max(0., min(self.cfg.monthly_budget, *(b.cash for b in self.books)))
                    qty = math.floor(budget / (mid * 100 + 2 * self.cfg.fee))
                    if qty < 1:
                        reason = "premium_budget"
                    else:
                        ticket = dict(signal, entry_id=f"entry_{len(self.entries)+1:04d}", entry_date=when,
                                      entry_dte=(signal["expiry"] - when).days,
                                      qty=qty, entry_mid=mid, entry_bid=float(quote.P_BID), entry_ask=float(quote.P_ASK),
                                      entry_cost=qty * (mid * 100 + self.cfg.fee))
                        self.entries.append(ticket)
                        self.entered_months.add(month)
                        for book in self.books:
                            book.open(ticket, when)
            self.entry_audit.append(dict(quote_date=when, month=month, phase="execution", reason=reason, **signal))
        if month not in self.entered_months and not final and session_in_month <= 1 + self.cfg.retry_sessions:
            self.pending, audit = select_signal(day, when, self.cfg, self.pricing)
            self.scan_audit.extend(audit)
            eligible = sum(a["candidates"] for a in audit)
            self.entry_audit.append(dict(quote_date=when, month=month, phase="search", session_in_month=session_in_month,
                                        reason="signal" if self.pending else "no_undervalued_put" if eligible else "no_eligible_quotes",
                                        candidates=eligible, underpriced=sum(a["underpriced"] for a in audit)))
        if month not in self.entered_months and session_in_month == 2 + self.cfg.retry_sessions:
            self.entry_audit.append(dict(quote_date=when, month=month, phase="window", reason="month_skipped"))
        for book in self.books:
            book.record_equity(when)
        expiries = {p["expiry"] for book in self.books for p in book.active}
        self.previous_atm = atm_snapshot(day, when, expiries, self.pricing)
        self.previous_spot, self.previous_session = spot, when
        self.market.append(dict(quote_date=when, spot=spot, daily_log_return=log_return, source_rows=len(day)))


def write_report(output, summary, equity, annual, entries, audit, manifest):
    cfg = manifest['config']
    q = equity.assign(variant=equity.strategy_id.map(POLICIES))
    a = annual.assign(variant=annual.strategy_id.map(POLICIES), year=annual.year.astype(str))
    figures = [px.line(q, x="quote_date", y="equity_pnl", color="variant", title="PnL po mid (USD)"),
               px.line(q, x="quote_date", y="drawdown", color="variant", title="Drawdown (USD)"),
               px.bar(a, x="year", y="pnl", color="variant", barmode="group", title="Roczny PnL mark-to-market (USD)")]
    pieces = ['<!doctype html><html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
              '<title>SPX: miesieczne puty 6-8 sigma</title><style>body{margin:0;font:15px Arial,sans-serif;color:#202526;background:white}'
              'main{max-width:1300px;margin:auto;padding:24px}h1{font-size:26px}h2{font-size:20px}p{line-height:1.5}'
              '.table,.chart{overflow-x:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}'
              '.chart>div{min-width:0}a{color:#087f4f}pre{white-space:pre-wrap;overflow-wrap:anywhere}@media(max-width:600px){main{padding:16px}}</style></head><body><main>',
              '<h1>SPX: miesieczne puty 6-8 sigma</h1>',
              f"<p>{manifest['data_start']} - {manifest['data_end']} | {manifest['sessions']} sesji | mid / mid | alpha 3 | kotwica 3 sigma</p>",
              f"<p>Najwieksze dodatnie dyskonto po kosztach w zakresie {cfg['min_sigma']:g}-{cfg['max_sigma']:g} sigma i {cfg['min_dte']}-{cfg['max_dte']} DTE. "
              f"Pierwsza sesja miesiaca + {cfg['retry_sessions']} kolejnych sesji. "
              'Sygnal EOD, zakup po mid kolejnej sesji ponizej zamrozonego limitu modelowego. '
              f"Wyjscie przy DTE &lt;={cfg['exit_dte']} lub w kolejnej sesji po dziennym spadku &gt;{cfg['crash_sigma']:g} sigma. "
              'Sigma dzienna: ATM IV tego samego terminu z poprzedniej sesji / sqrt(252).</p>',
              f"<p>Kapital poczatkowy: {manifest['config']['initial_capital']:,.0f} USD. Budzet miesieczny: {manifest['config']['monthly_budget']:,.0f} USD; "
              f"pelne kontrakty SPX x100, prowizja {manifest['config']['fee']:g} USD na strone, rezerwa na prowizje wyjscia. "
              'Oba warianty maja identyczne wejscia. Mid to hipotetyczna cena wykonania, nie potwierdzona transakcja.</p>',
              '<h2>Wyniki</h2><div class="table">' + summary.to_html(index=False, float_format=lambda x: f"{x:,.4f}") + '</div>']
    for i, fig in enumerate(figures):
        fig.update_layout(height=490, margin=dict(l=65, r=20, t=65, b=100), title_font_size=18,
                          legend=dict(orientation="h", y=-.2, title_text=""), xaxis_title=None, yaxis_title="USD")
        if i == 2:
            fig.update_xaxes(type="category")
        pieces.append('<section class="chart">' + fig.to_html(full_html=False, include_plotlyjs=i == 0,
                       config={"displaylogo": False, "responsive": True}) + '</section>')
    pieces.extend(['<h2>Roczne wyniki</h2><div class="table">', annual.to_html(index=False), '</div>',
                   '<h2>Decyzje wejscia</h2><div class="table">', audit.groupby(['phase', 'reason']).size().rename('count').reset_index().to_html(index=False), '</div>',
                   '<p><a href="trades.csv">Transakcje CSV</a> | <a href="entries.csv">Wejscia CSV</a> | <a href="crash_events.csv">Sygnaly krachu CSV</a> | '
                   '<a href="annual.csv">Wyniki roczne CSV</a> | <a href="performance.png">Wykres PNG</a></p>',
                   '<h2>Ograniczenia</h2><ul>', ''.join('<li>' + html.escape(s) + '</li>' for s in manifest['caveats']), '</ul>',
                   '<details><summary>Parametry i audyt</summary><pre>', html.escape(json.dumps(manifest, indent=2)), '</pre></details></main></body></html>'])
    (output / 'report.html').write_text(''.join(pieces), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for policy, group in equity.groupby('strategy_id'):
        axes[0].plot(group.quote_date, group.equity_pnl, label=POLICIES[policy])
        axes[1].plot(group.quote_date, group.drawdown, label=POLICIES[policy])
    for ax, title in zip(axes, ['PnL po mid (USD)', 'Drawdown (USD)']):
        ax.set_title(title)
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'performance.png', dpi=150)
    plt.close(fig)


def run(data_dir=Path('data-parquet'), output=OUTPUT, cfg=Config(), limit_files=None):
    files = sorted(data_dir.glob('spx_eod_*.parquet'))
    if limit_files is not None:
        if limit_files < 1:
            raise ValueError('Positive file limit required')
        files = files[:limit_files]
    if not files:
        raise ValueError('No input parquet files')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output directory; existing results are never overwritten')
    source_dates = {}
    for path in files:
        for date in pq.read_table(path, columns=['QUOTE_DATE']).column(0).unique().to_pylist():
            date = pd.Timestamp(date)
            if pd.isna(date) or date in source_dates:
                raise ValueError('Missing or duplicate source session across files')
            source_dates[date] = path
    first, last = min(source_dates), max(source_dates)
    calendar = mcal.get_calendar('CBOE_Index_Options').valid_days(
        start_date=first.replace(day=1), end_date=last).tz_localize(None)
    calendar = calendar[calendar >= first]
    month_schedule = mcal.get_calendar('CBOE_Index_Options').valid_days(
        start_date=first.replace(day=1), end_date=last).tz_localize(None)
    ordinals = pd.Series(np.arange(len(month_schedule)), index=month_schedule).groupby(month_schedule.to_period('M')).cumcount() + 1
    if calendar.empty:
        raise ValueError('No exchange sessions in source interval')
    output.mkdir(parents=True, exist_ok=True)
    engine = Experiment(cfg)
    current_file, frames, loaded = None, {}, 0
    for when in calendar:
        path = source_dates.get(when)
        if path is not None and path != current_file:
            raw = pq.read_table(path, columns=COLUMNS).to_pandas()
            for column in COLUMNS:
                raw[column] = pd.to_datetime(raw[column], errors='raise') if column.endswith('DATE') else pd.to_numeric(raw[column], errors='coerce')
            if raw[['QUOTE_DATE', 'EXPIRE_DATE']].isna().any().any():
                raise ValueError('Missing quote or expiry date')
            raw = raw[(raw.EXPIRE_DATE - raw.QUOTE_DATE).dt.days.between(1, cfg.max_dte)]
            frames = {date: group for date, group in raw.groupby('QUOTE_DATE')}
            current_file = path
            loaded += 1
            print(f'[{loaded}/{len(files)}] {path.name}', flush=True)
        day = frames.get(when, pd.DataFrame(columns=COLUMNS)) if path is not None else pd.DataFrame(columns=COLUMNS)
        engine.step(when, day, int(ordinals.loc[when]), final=when == calendar[-1])
    summaries, trades, equity = zip(*(book.finish() for book in engine.books))
    summary, trades, equity = pd.DataFrame(summaries), pd.concat(trades, ignore_index=True), pd.concat(equity, ignore_index=True)
    annual = equity.assign(year=equity.quote_date.dt.year).groupby(['strategy_id', 'year'], as_index=False).agg(
        pnl=('daily_pnl', 'sum'), end_equity=('equity', 'last'), max_drawdown=('drawdown', 'min'), stale_sessions=('stale_positions', lambda x: int(x.gt(0).sum())))
    entries = pd.DataFrame(engine.entries) if engine.entries else pd.DataFrame(columns=['entry_id', 'entry_date', 'signal_date', 'expiry', 'strike', 'qty', 'entry_mid', 'entry_cost'])
    audit = pd.DataFrame(engine.entry_audit) if engine.entry_audit else pd.DataFrame(columns=['quote_date', 'month', 'phase', 'reason'])
    if trades.empty:
        trades = pd.DataFrame(columns=['strategy_id', 'entry_id', 'entry_date', 'exit_date', 'status', 'pnl'])
    events = pd.DataFrame([event for book in engine.books for event in book.events],
                          columns=['entry_id', 'quote_date', 'expiry', 'daily_log_return', 'previous_atm_iv', 'previous_atm_date', 'daily_drop_sigma'])
    for name, frame in [('summary', summary), ('trades', trades), ('entries', entries), ('entry_audit', audit),
                        ('annual', annual), ('scan_audit', pd.DataFrame(engine.scan_audit)), ('crash_events', events)]:
        frame.to_csv(output / f'{name}.csv', index=False)
    equity.to_parquet(output / 'daily_equity.parquet', index=False)
    pd.DataFrame(engine.market).to_parquet(output / 'market.parquet', index=False)
    pd.DataFrame([row for book in engine.books for row in book.paths]).to_parquet(output / 'position_paths.parquet', index=False)
    manifest = dict(kind='power_law_monthly_mid', config=asdict(cfg), pricing=asdict(engine.pricing),
                    data_start=str(calendar[0].date()), data_end=str(calendar[-1].date()), sessions=len(calendar),
                    source_sessions=len(source_dates), missing_source_sessions=sum(day not in source_dates for day in calendar),
                    non_calendar_source_sessions=sum(day not in calendar for day in source_dates),
                    traded_months=len(engine.entered_months), months=len(calendar.to_period('M').unique()),
                    files=[dict(name=p.name, size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in files],
                    scanner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    model_sha256=hashlib.sha256(Path(__file__).with_name('power_law.py').read_bytes()).hexdigest(),
                    caveats=[
                        'In-sample research: sigma/DTE were chosen using the same 2010-2023 history, not an out-of-sample trading validation.',
                        'Exact source expiries, no monthly aliases or synthetic settlement. Selection uses the same ATM IV, quote and bid-anchor slope screens as the heatmap; the valuation anchor is repriced at mid.',
                        'Signals use the first six exchange sessions, not calendar days. One purchase per month; a signal on session six may execute on session seven. Missing data do not extend the search window.',
                        'Entry is next-session EOD mid below the prior signal model minus round-trip fees. Unfilled signals are cancelled, not retroactively replaced. Sigma/DTE eligibility is measured on the signal date.',
                        'Both controls share entries and quantities. Premium plus both commissions is capped at USD 1000 and available cash in both books. No fractional contracts, leverage or reinvestment of gains into a larger budget.',
                        'Crash means strictly negative close-to-close log return exceeding 5 times previous-session same-expiry ATM IV / sqrt(252). SPX close is proxied by the source underlying median. No stale-IV fallback or multi-session return across missing source sessions.',
                        'Crash instruction is persistent and executes on the next session with a valid exact-contract quote, never at the trigger close. Missing marks and risk inputs are counted; no missing quote is imputed as an executable price.',
                        'Exit at first valid quote on/after 30 calendar DTE; final data liquidation is labelled separately. Unresolved positions use stale marks and can make total PnL unreliable.',
                        'Mid fills, including wide-spread exits, are hypothetical. No depth, volume, queue priority, market impact or intraday observations; large contract quantities may not be executable.',
                        'No cash interest, stocks, financing shorts or dividends. Positive relative discount is not proof of expected profit. Calendar is CBOE_Index_Options.'
                    ])
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    write_report(output, summary, equity, annual, entries, audit, manifest)
    print(summary.to_string(index=False), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data-parquet'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--limit-files', type=int)
    parser.add_argument('--budget', type=float, default=1000.)
    args = parser.parse_args()
    run(args.data_dir, args.output, Config(monthly_budget=args.budget), args.limit_files)
