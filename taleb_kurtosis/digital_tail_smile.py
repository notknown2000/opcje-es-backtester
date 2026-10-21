"""Reproducible, equal-day-weight IV diagnostics, independent of trade outcomes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq

try:
    from taleb_kurtosis.digital_tail import COLUMNS, Config, MonthlyExpiryMap, build_candidate, valid_quote
except ModuleNotFoundError as exc:
    if exc.name != "taleb_kurtosis":
        raise
    from digital_tail import COLUMNS, Config, MonthlyExpiryMap, build_candidate, valid_quote

OUTPUT = Path("results/digital_tail_smile_diagnostics")
GRID = np.round(np.arange(-5, 2.01, .1), 2)


def interpolate(x, y, grid=GRID, max_gap=.5):
    """No extrapolation or interpolation across gaps wider than half a sigma."""
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    result = np.full(len(grid), np.nan)
    for i, point in enumerate(grid):
        exact = np.flatnonzero(np.isclose(x, point, atol=1e-10, rtol=0))
        if len(exact):
            result[i] = y[exact[0]]
            continue
        right = np.searchsorted(x, point)
        if 0 < right < len(x) and x[right] - x[right - 1] <= max_gap:
            result[i] = np.interp(point, x[right - 1:right + 1], y[right - 1:right + 1])
    return result


def mean_curve(curves):
    return curves.groupby("sigma", as_index=False).agg(
        iv=("iv", "mean"), relative_iv=("relative_iv", "mean"), days=("iv", "count"))


def matched_zones(curves):
    zones = curves[curves.sigma.isin([-3.5, -3, -2.5])]
    complete = zones.pivot(index="date", columns="sigma", values="iv").reindex(columns=[-3.5, -3, -2.5]).dropna().index
    return mean_curve(zones[zones.date.isin(complete)])


def smile_figure(curves, normalized=True):
    key = "relative_iv" if normalized else "iv"
    scale = 1 if normalized else 100
    fig = go.Figure()
    for day, rows in curves.groupby("date", sort=True):
        fig.add_scatter(x=rows.sigma, y=rows[key] * scale, name=str(day)[:10],
                        mode="lines", line=dict(width=1), opacity=.5, connectgaps=False)
    avg = mean_curve(curves)
    fig.add_scatter(x=avg.sigma, y=avg[key] * scale, name="Srednia", mode="lines",
                    line=dict(color="#161616", width=4), customdata=avg.days,
                    hovertemplate="sigma=%{x}<br>IV=%{y:.3f}<br>dni=%{customdata}<extra>Srednia</extra>")
    for sigma in [2.5, 3, 3.5]:
        fig.add_vline(x=-sigma, line_dash="dot", line_color="#888888")
    fig.update_layout(xaxis_title="log(K / SPX) / (ATM IV * sqrt(DTE / 365))",
                      yaxis_title="IV / ATM IV" if normalized else "IV (%)", height=480,
                      margin=dict(l=40, r=20, t=30, b=90),
                      legend=dict(orientation="h", y=-.25))
    return fig


def run(data_dir=Path("data-parquet"), output=OUTPUT, seed=42, count=10):
    files = sorted(data_dir.glob("spx_eod_*.parquet"))
    sources = {}
    for path in files:
        dates = pq.read_table(path, columns=["QUOTE_DATE"]).to_pandas()["QUOTE_DATE"]
        for date in pd.to_datetime(dates.unique()):
            sources.setdefault(date, []).append(path)
    dates = sorted(sources)
    mapping = MonthlyExpiryMap(dates[0], dates[-1])
    cfg = Config(expiry_mode="monthly_aliases")
    curves, samples, funding, audit, raw = [], [], [], [], []
    columns = COLUMNS + ["C_IV", "C_BID", "C_ASK", "C_DELTA"]
    for index in np.random.default_rng(seed).permutation(len(dates)):
        when = dates[index]
        day = pd.concat([pq.read_table(p, columns=columns,
                        filters=[("QUOTE_DATE", "=", when.date())]).to_pandas() for p in sources[when]], ignore_index=True)
        for col in columns:
            day[col] = pd.to_datetime(day[col]) if col.endswith("DATE") else pd.to_numeric(day[col], errors="coerce")
        day, _ = mapping.normalize(day)
        day = mapping.entry_frame(day)
        expiries = [e for e in day.EXPIRE_DATE.unique() if 105 <= (pd.Timestamp(e) - when).days <= 135]
        reason = "no_unambiguous_expiry"
        if expiries:
            expiry = pd.Timestamp(min(expiries, key=lambda e: (abs((pd.Timestamp(e) - when).days - 120), e)))
            chain = day[day.EXPIRE_DATE.eq(expiry)].sort_values("STRIKE")
            spot = day.UNDERLYING_LAST.median()
            reason = "invalid_spot_or_duplicate_strikes"
            if np.isfinite(spot) and spot > 0 and not chain.STRIKE.duplicated().any():
                atm = chain.iloc[(chain.STRIKE - spot).abs().argmin()]
                iv = atm.P_IV
                reason = "invalid_atm"
                if valid_quote(atm) and .03 <= iv <= 3 and -.65 <= atm.P_DELTA <= -.35:
                    dte = (expiry - when).days
                    x = np.log(chain.STRIKE / spot) / (iv * np.sqrt(dte / 365))
                    put = chain.STRIKE.le(spot)
                    wing_iv = chain.P_IV.where(put, chain.C_IV)
                    bid = chain.P_BID.where(put, chain.C_BID)
                    ask = chain.P_ASK.where(put, chain.C_ASK)
                    delta = chain.P_DELTA.where(put, chain.C_DELTA)
                    valid = (bid.gt(0) & ask.ge(bid) & np.isfinite(ask) & wing_iv.between(.03, 3)
                             & delta.where(put, -delta).between(-1, 0, inclusive="neither") & np.isfinite(x))
                    reason = "insufficient_quotes"
                    if valid.sum() >= 5:
                        values = interpolate(x[valid], wing_iv[valid])
                        curves.append(pd.DataFrame(dict(date=str(when.date()), sigma=GRID, iv=values, relative_iv=values / iv)))
                        raw.append(pd.DataFrame(dict(date=str(when.date()), strike=chain.loc[valid, "STRIKE"],
                                                    sigma=x[valid], iv=wing_iv[valid], side=np.where(put[valid], "put", "call"))))
                        samples.append(dict(date=str(when.date()), expiry=str(expiry.date()),
                                            source_expiry=str(chain.SOURCE_EXPIRE_DATE.iloc[0].date()),
                                            dte=dte, spot=spot, atm_iv=iv, valid_quotes=int(valid.sum()),
                                            rejected_quotes=int((~valid).sum())))
                        for sigma in [2.5, 3, 3.5]:
                            candidate, why = build_candidate(day, when, Config(expiry_mode="monthly_aliases", spread_sigma=sigma))
                            row = dict(date=str(when.date()), sigma=sigma, reason=why)
                            if candidate:
                                row.update({k: candidate[k] for k in ["spread_count", "spread_credit", "width", "risk_budget", "actual_short_sigma"]})
                                row["credit_width"] = candidate["spread_credit"] / candidate["width"]
                            funding.append(row)
                        reason = "selected"
        audit.append(dict(date=str(when.date()), reason=reason))
        if len(samples) == count:
            break
    if len(samples) != count:
        raise ValueError(f"Only {len(samples)} eligible dates")
    output.mkdir(parents=True, exist_ok=True)
    all_curves = pd.concat(curves, ignore_index=True)
    all_curves.to_parquet(output / "curves.parquet", index=False)
    pd.concat(raw, ignore_index=True).to_parquet(output / "quotes.parquet", index=False)
    for name, rows in [("samples", samples), ("funding", funding), ("selection_audit", audit)]:
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    mean_curve(all_curves).to_csv(output / "mean.csv", index=False)
    matched_zones(all_curves).to_csv(output / "matched_zones.csv", index=False)
    manifest = dict(seed=seed, count=count, population=len(dates), data_start=str(dates[0].date()),
                    data_end=str(dates[-1].date()), attempted=len(audit), max_interpolation_gap_sigma=.5,
                    sampling="Random permutation of observed sessions; first 10 with eligible monthly chain and valid ATM/quotes. No trade/PnL filter.")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    body = "<h1>SPX: 10 losowych sesji, ~120 DTE</h1><p>Seed 42. OTM puty po lewej, OTM calle po prawej. "
    body += "IV dostawcy, nie przeliczone z modelu. Srednia jednakowo wazy dni; liczba obserwacji zmienia sie po osi. "
    body += "Brak ekstrapolacji. Krzywe nie dowodza przewagi 3,5 sigma.</p>"
    for normalized in [True, False]:
        body += smile_figure(all_curves, normalized).to_html(full_html=False, include_plotlyjs=True if normalized else False)
    body += "<h2>Wspolne dni dla 2,5 / 3 / 3,5 sigma</h2>" + matched_zones(all_curves).to_html(index=False)
    body += pd.DataFrame(samples).to_html(index=False) + pd.DataFrame(funding).to_html(index=False)
    (output / "report.html").write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head><body>' + body + '</body></html>')
    print(json.dumps(manifest))
    print(mean_curve(all_curves).query("sigma in [-3.5, -3, -2.5]").to_string(index=False))


if __name__ == "__main__":
    run()
