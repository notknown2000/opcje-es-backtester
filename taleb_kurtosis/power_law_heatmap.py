"""Historical same-chain relative put valuation, not a trading PnL backtest."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq

from taleb_kurtosis.digital_tail import COLUMNS
from taleb_kurtosis.power_law import PricingConfig, prepare_chain, price_vector

OUTPUT = Path("results/power_law_dte_heatmap_anchor3")
SIGMA_EDGES = [3, 3.5, 4, 4.5, 5, 6, 7, 8, 10, 12, 15, np.inf]
SIGMA_LABELS = ["3-3.5", "3.5-4", "4-4.5", "4.5-5", "5-6", "6-7", "7-8", "8-10", "10-12", "12-15", "15+"]
DTE_EDGES = [1, 8, 15, 31, 61, 91, 121, 181, 271, 366, 541, 731, 1096, np.inf]
DTE_LABELS = ["1-7", "8-14", "15-30", "31-60", "61-90", "91-120", "121-180", "181-270", "271-365", "366-540", "541-730", "731-1095", "1096+"]
KEYS = ["alpha", "sigma_bin", "dte_bin"]
CHAIN_COLUMNS = ["quote_date", "expiry", "alpha", "sigma_bin", "dte_bin", "quotes", "discount",
                 "mid_discount", "frequency", "frequency_10pct", "median_edge"]
COLOR_SCALE = [[0, "#b42338"], [.5, "#ffffff"], [1, "#087f4f"]]


def dte_label(dte):
    if not np.isfinite(dte) or dte < 1:
        return None
    return DTE_LABELS[np.searchsorted(DTE_EDGES, dte, side="right") - 1]


def scan_chain(chain, when, expiry, alphas, pricing, price_basis="bid_ask"):
    """Only information from this source expiry and observation date is used."""
    if price_basis not in {"bid_ask", "mid"}:
        raise ValueError("price_basis must be bid_ask or mid")
    dte = (expiry - when).days
    reference = float(chain.UNDERLYING_LAST.median())
    reason = "ok"
    if dte < 1:
        reason = "nonpositive_dte"
    elif not np.isfinite(reference) or reference <= 0:
        reason = "invalid_spot"
    elif not np.isfinite(chain.UNDERLYING_LAST).all() or (chain.UNDERLYING_LAST.max() - chain.UNDERLYING_LAST.min()) / reference > .001:
        reason = "inconsistent_spot"
    prepared = pd.DataFrame()
    if reason == "ok":
        prepared, reason = prepare_chain(chain, reference, dte, pricing)
    records, audits = [], []
    for alpha in alphas:
        audit = dict(quote_date=when, expiry=expiry, dte=dte, dte_bin=dte_label(dte), alpha=alpha,
                     raw_quotes=len(chain), eligible_quotes=0, reason=reason, reference=reference,
                     duplicate_strike_rows=int(chain.STRIKE.duplicated(keep=False).sum()),
                     atm_iv=np.nan, anchor_sigma=np.nan, anchor_strike=np.nan, anchor_bid=np.nan, anchor_ask=np.nan)
        if reason != "ok":
            audits.append(audit)
            continue
        audit["atm_iv"] = float(prepared.atm_iv.iloc[0])
        prices, anchor_index, why = price_vector(prepared, reference, replace(pricing, alpha=alpha))
        audit["reason"] = why
        if why != "ok":
            audits.append(audit)
            continue
        anchor = prepared.iloc[anchor_index]
        audit.update(anchor_sigma=float(anchor.sigma), anchor_strike=float(anchor.STRIKE),
                     anchor_bid=float(anchor.P_BID), anchor_ask=float(anchor.P_ASK))
        # Exclude the anchor and inward strikes; no inference about <3-sigma puts.
        keep = np.isfinite(prices) & (prices > 0) & prepared.sigma.ge(3).to_numpy()
        q = prepared[keep]
        model = prices[keep]
        mid_model = model * (anchor.P_BID + anchor.P_ASK) / (2 * anchor.P_BID)
        mid = (q.P_BID.to_numpy() + q.P_ASK.to_numpy()) / 2
        mid_discount = 1 - mid / mid_model
        if price_basis == "mid":
            model = mid_model
        cost = (mid if price_basis == "mid" else q.P_ASK.to_numpy()) + 2 * pricing.fee / pricing.multiplier
        discount = 1 - cost / model
        edge = model / cost - 1
        bins = np.searchsorted(SIGMA_EDGES, q.sigma.to_numpy(), side="right") - 1
        audit["eligible_quotes"] = int(keep.sum())
        if not keep.any():
            audit["reason"] = "no_quotes_beyond_3sigma"
        for bin_id in np.unique(bins):
            selected = bins == bin_id
            records.append(dict(quote_date=when, expiry=expiry, alpha=alpha, sigma_bin=SIGMA_LABELS[bin_id],
                                dte_bin=dte_label(dte), quotes=int(selected.sum()),
                                discount=float(np.median(discount[selected])),
                                mid_discount=float(np.median(mid_discount[selected])),
                                frequency=float(np.mean(discount[selected] > 0)),
                                frequency_10pct=float(np.mean(edge[selected] >= pricing.min_edge)),
                                median_edge=float(np.median(edge[selected]))))
        audits.append(audit)
    return records, audits


def daily_bins(chains):
    # Dense strike grids and dates with many expiries must not dominate the map.
    return chains.groupby(["quote_date"] + KEYS, as_index=False, observed=True).agg(
        chains=("expiry", "size"), quotes=("quotes", "sum"), discount=("discount", "mean"),
        mid_discount=("mid_discount", "mean"), frequency=("frequency", "mean"),
        frequency_10pct=("frequency_10pct", "mean"), median_edge=("median_edge", "mean"))


def summarize(daily, min_days=100, min_years=3):
    q = daily.assign(year=pd.to_datetime(daily.quote_date).dt.year)
    result = q.groupby(KEYS, as_index=False, observed=True).agg(
        days=("quote_date", "nunique"), years=("year", "nunique"), chains=("chains", "sum"), quotes=("quotes", "sum"),
        median_discount=("discount", "median"), mean_discount=("discount", "mean"),
        median_mid_discount=("mid_discount", "median"), frequency=("frequency", "mean"),
        frequency_10pct=("frequency_10pct", "mean"), median_edge=("median_edge", "median"),
        discount_p10=("discount", lambda x: x.quantile(.1)), discount_p90=("discount", lambda x: x.quantile(.9)))
    result["sufficient_coverage"] = result.days.ge(min_days) & result.years.ge(min_years)
    return result


def matrix(summary, alpha, column, mask=True):
    q = summary[summary.alpha.eq(alpha)].copy()
    if mask:
        q.loc[~q.sufficient_coverage, column] = np.nan
    return q.pivot(index="sigma_bin", columns="dte_bin", values=column).reindex(index=SIGMA_LABELS, columns=DTE_LABELS)


def heatmap(summary, alpha, metric="median_discount", price_basis="bid_ask"):
    coverage = metric == "days"
    frequency = metric == "frequency"
    grid = matrix(summary, alpha, metric, mask=not coverage)
    days = matrix(summary, alpha, "days", mask=False)
    years = matrix(summary, alpha, "years", mask=False)
    quotes = matrix(summary, alpha, "quotes", mask=False)
    z = grid.to_numpy() * (1 if coverage else 100)
    title = "Pokrycie: liczba sesji" if coverage else "Czestosc niedowartosciowania (%)" if frequency else "Dyskonto wzgledem modelu (%)"
    custom = np.stack([days.to_numpy(), years.to_numpy(), quotes.to_numpy()], axis=-1)
    labels = [[("" if not np.isfinite(v) else f"{v:.0f}" if coverage else f"{v:.1f}%") for v in row] for row in z]
    trace = go.Heatmap(x=DTE_LABELS, y=SIGMA_LABELS, z=z, customdata=custom, text=labels,
                       colorscale="Blues" if coverage else "Greens" if frequency else COLOR_SCALE,
                       zmin=0 if coverage or frequency else -100, zmax=float(days.max().max()) if coverage else 100,
                       texttemplate="%{text}", textfont_size=11,
                       hoverongaps=False, xgap=2, ygap=2,
                       colorbar=dict(title="Sesje" if coverage else "%", thickness=14),
                       hovertemplate="DTE: %{x}<br>Sigma: %{y}<br>Wartosc: %{z:.2f}<br>Sesje: %{customdata[0]:.0f}<br>Lata: %{customdata[1]:.0f}<br>Kwotowania: %{customdata[2]:.0f}<extra></extra>")
    fig = go.Figure(trace)
    basis_label = "mid / mid + koszty" if price_basis == "mid" else "bid / ask + koszty"
    fig.update_layout(title=f"{title} | alpha {alpha:g} | anchor 3 sigma | {basis_label}", height=650,
                      margin=dict(l=75, r=65, t=75, b=120), plot_bgcolor="#e5e7eb", paper_bgcolor="white",
                      font=dict(family="Arial", size=12, color="#202526"),
                      xaxis=dict(title="Dni do wygasniecia (DTE)", type="category", tickangle=-40, fixedrange=True),
                      yaxis=dict(title="Odleglosc puta (sigma ATM IV)", type="category",
                                 categoryorder="array", categoryarray=SIGMA_LABELS,
                                 autorange="reversed", fixedrange=True))
    return fig


def save_png(summary, alpha, path, metric="median_discount", price_basis="bid_ask"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    coverage, frequency = metric == "days", metric == "frequency"
    grid = matrix(summary, alpha, metric, mask=not coverage).to_numpy() * (1 if coverage else 100)
    cmap = plt.get_cmap("Blues" if coverage else "Greens").copy() if coverage or frequency else LinearSegmentedColormap.from_list("discount", [c for _, c in COLOR_SCALE])
    cmap.set_bad("#e5e7eb")
    fig, ax = plt.subplots(figsize=(17, 9))
    im = ax.imshow(np.ma.masked_invalid(grid), aspect="auto", cmap=cmap,
                   vmin=0 if coverage or frequency else -100, vmax=max(1, np.nanmax(grid)) if coverage else 100)
    for i, j in np.ndindex(grid.shape):
        if np.isfinite(grid[i, j]):
            value = grid[i, j]
            text = f"{value:.0f}" if coverage else f"{value:.1f}%"
            dark = value > (np.nanmax(grid) * .6 if coverage else 60) or (not coverage and not frequency and value < -60)
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="white" if dark else "#202526")
    ax.set_xticks(range(len(DTE_LABELS)), DTE_LABELS, rotation=40, ha="right")
    ax.set_yticks(range(len(SIGMA_LABELS)), SIGMA_LABELS)
    ax.set_xlabel("Dni do wygasniecia (DTE)")
    ax.set_ylabel("Odleglosc puta (sigma ATM IV)")
    title = "Pokrycie: liczba sesji" if coverage else "Czestosc niedowartosciowania" if frequency else "Dyskonto ceny zakupu wzgledem modelu Taleba"
    basis_label = "mid / mid + koszty" if price_basis == "mid" else "bid / ask + koszty"
    ax.set_title(f"{title} | alpha {alpha:g} | kotwica 3 sigma | {basis_label}", pad=18)
    fig.colorbar(im, ax=ax, label="Sesje" if coverage else "%", shrink=.85)
    cost_label, anchor_label = ("mid", "mid") if price_basis == "mid" else ("ask", "bid")
    fig.text(.06, .02, f"Szare: brak / niewystarczajace dane. Dyskonto = 1 - ({cost_label} + koszty) / model; kotwica {anchor_label}. To nie jest stopa zwrotu.", fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(output, summary, audit, config):
    alpha = config["primary_alpha"]
    price_basis = config.get("price_basis", "bid_ask")
    cost_label, anchor_label = ("mid", "mid") if price_basis == "mid" else ("ask", "bid")
    for metric, name in [("median_discount", "heatmap"), ("frequency", "heatmap_frequency"), ("days", "heatmap_coverage")]:
        save_png(summary, alpha, output / f"{name}.png", metric, price_basis)
    ranked = summary[summary.alpha.eq(alpha) & summary.sufficient_coverage].sort_values("median_discount", ascending=False)
    ranked.to_csv(output / "ranked_cells.csv", index=False)
    table = ranked[["sigma_bin", "dte_bin", "median_discount", "median_mid_discount", "frequency", "days", "years"]].head(20).copy()
    for col in ["median_discount", "median_mid_discount", "frequency"]:
        table[col] = table[col].map(lambda v: f"{100*v:.1f}%")
    audit_counts = audit[audit.alpha.eq(alpha)].groupby("reason", as_index=False).agg(chains=("raw_quotes", "size"), rows=("raw_quotes", "sum"))
    audit_counts.to_csv(output / "audit_counts.csv", index=False)
    pieces = ["""<!doctype html><html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Taleb: sigma / DTE</title><style>body{font:15px Arial,sans-serif;color:#202526;margin:0;background:#fff}main{max-width:1400px;margin:auto;padding:24px}h1{font-size:26px}h2{font-size:20px;margin-top:32px}p{line-height:1.5;max-width:1050px}.chart{overflow-x:auto}.chart>div{min-width:1100px}.table{overflow-x:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}a{color:#087f4f}details{margin:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}@media(max-width:600px){main{padding:16px}}</style></head><body><main>
<h1>Taleb: niedowartosciowanie putow SPX</h1>""",
              f"<p>{config['data_start']} - {config['data_end']} | {config['sessions']} sesji | alpha {alpha:g} | kotwica 3 sigma (+/- 0.25)</p>",
              "<p><b>Zielony: cena zakupu ponizej modelu. Czerwony: powyzej. Szary: brak dostatecznych danych.</b> "
              f"Dyskonto = 1 - ({cost_label} + prowizje w obie strony) / model z kotwicy {anchor_label}. +30% oznacza koszt zakupu o 30% nizszy od modelu, nie 30% zysku.</p>",
              f"<p>Wariant: <b>{price_basis}</b>. Prowizja: {config['pricing']['fee']:g} USD za kontrakt na strone. "
              "Kolumna median_mid_discount pokazuje mid-mid bez prowizji. Mid nie gwarantuje ceny wykonania. "
              "Filtry plynnosci, kotwica i kwalifikacja lancuchow pozostaja takie same jak w wariancie bid-ask.</p>",
              f"<p>Sigma = log(S/K) / (IV ATM tego dnia i lancucha x sqrt(DTE/365)), bez korekty stop i dywidend. "
              f"Minimum pokrycia: {config['min_days']} sesji i {config['min_years']} lat. "
              "Agregacja: mediana strike'ow w lancuchu, rowna waga lancuchow w sesji, mediana po sesjach. "
              "Dyskonto kolorowane w zakresie -100% do +100%; liczby poza skala zachowuja wartosc, kolor jest nasycony.</p>"]
    for i, metric in enumerate(["median_discount", "frequency", "days"]):
        fig = heatmap(summary, alpha, metric, price_basis)
        pieces.append('<section class="chart">' + fig.to_html(full_html=False, include_plotlyjs=i == 0, config={"displaylogo": False, "responsive": True}) + "</section>")
    pieces.append('<h2>Najwyzsze dyskonto: pola z wystarczajacym pokryciem</h2><div class="table">' + table.to_html(index=False, border=0) + "</div>")
    pieces.append('<p><a href="heatmap.png">Heatmapa PNG</a> | <a href="summary.csv">Wyniki CSV</a> | <a href="yearly.csv">Wyniki roczne CSV</a></p>')
    for other in config["alphas"]:
        if other != alpha:
            pieces.append(f'<details><summary>Wrazliwosc: alpha {other:g}</summary><div class="chart">' + heatmap(summary, other, price_basis=price_basis).to_html(full_html=False, include_plotlyjs=False) + "</div></details>")
    pieces.append("<h2>Kontrola danych</h2><div class=\"table\">" + audit_counts.to_html(index=False, border=0) + "</div>")
    pieces.append("<h2>Ograniczenia</h2><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in config["caveats"]) + "</ul>")
    pieces.append('<details><summary>Parametry i pochodzenie danych</summary><pre>' + html.escape(json.dumps(config, indent=2)) + "</pre></details></main></body></html>")
    (output / "report.html").write_text("".join(pieces), encoding="utf-8")


def scan_file(file, output, alphas, pricing, price_basis="bid_ask"):
    raw = pq.read_table(file, columns=COLUMNS).to_pandas()
    for col in COLUMNS:
        raw[col] = pd.to_datetime(raw[col], errors="raise") if col.endswith("DATE") else pd.to_numeric(raw[col], errors="coerce")
    if raw.QUOTE_DATE.isna().any() or raw.EXPIRE_DATE.isna().any():
        raise ValueError(f"Missing chain dates in {file.name}")
    rows, audit = [], []
    for (when, expiry), chain in raw.groupby(["QUOTE_DATE", "EXPIRE_DATE"], sort=True):
        bins, checks = scan_chain(chain, when, expiry, alphas, pricing, price_basis)
        rows.extend(bins)
        audit.extend(checks)
    pd.DataFrame(rows, columns=CHAIN_COLUMNS).to_parquet(output / "chain_bins" / file.name, index=False)
    pd.DataFrame(audit).to_parquet(output / "chain_audit" / file.name, index=False)
    return f"{file.name}: {len(audit)//len(alphas)} chains, {len(rows)} bins"


def run(data_dir=Path("data-parquet"), output=OUTPUT, alpha=3., limit_files=None, min_days=100, min_years=3, resume=False, workers=4, price_basis="bid_ask"):
    if price_basis not in {"bid_ask", "mid"}:
        raise ValueError("price_basis must be bid_ask or mid")
    if min_days < 1 or min_years < 1 or workers < 1 or (limit_files is not None and limit_files < 1):
        raise ValueError("Positive coverage and file limits required")
    pricing = PricingConfig(alpha=alpha, anchor_sigma=3.)
    alphas = sorted({2.5, alpha, 3.5})
    files = sorted(data_dir.glob("spx_eod_*.parquet"))
    if limit_files:
        files = files[:limit_files]
    if not files:
        raise ValueError("No SPX parquet files")
    source_doc = Path("data/wycena_opcji_metoda_taleba.md")
    config = dict(version=2, pricing=asdict(pricing), price_basis=price_basis, primary_alpha=alpha, alphas=alphas,
                  min_days=min_days, min_years=min_years, source_directory=str(data_dir.resolve()),
                  files=[dict(name=p.name, size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in files],
                  source_document_sha256=hashlib.sha256(source_doc.read_bytes()).hexdigest(),
                  model_sha256=hashlib.sha256(Path(__file__).with_name("power_law.py").read_bytes()).hexdigest(),
                  scanner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "run_config.json"
    if config_path.exists():
        if not resume or json.loads(config_path.read_text()) != config:
            raise ValueError("Output already exists or parameters changed; use a new output path or --resume with identical parameters")
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    for dirname in ["chain_bins", "chain_audit"]:
        (output / dirname).mkdir(exist_ok=True)
    pending = []
    for file in files:
        bin_path, audit_path = output / "chain_bins" / file.name, output / "chain_audit" / file.name
        if resume and bin_path.exists() and audit_path.exists():
            print(f"cached {file.name}", flush=True)
            continue
        pending.append(file)
    if workers == 1:
        for i, file in enumerate(pending, start=len(files)-len(pending)+1):
            print(f"[{i}/{len(files)}] " + scan_file(file, output, alphas, pricing, price_basis), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(scan_file, file, output, alphas, pricing, price_basis) for file in pending]
            for i, job in enumerate(as_completed(jobs), start=len(files)-len(pending)+1):
                print(f"[{i}/{len(files)}] " + job.result(), flush=True)
    chains = pd.concat([pd.read_parquet(output / "chain_bins" / p.name) for p in files], ignore_index=True)
    audit = pd.concat([pd.read_parquet(output / "chain_audit" / p.name) for p in files], ignore_index=True)
    if chains.empty:
        raise ValueError("No eligible chains; inspect chain_audit files")
    daily = daily_bins(chains)
    summary = summarize(daily, min_days, min_years)
    yearly = pd.concat([summarize(g, min_days=1, min_years=1).assign(year=year)
                        for year, g in daily.groupby(daily.quote_date.dt.year)], ignore_index=True)
    daily.to_parquet(output / "daily_bins.parquet", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    yearly.to_csv(output / "yearly.csv", index=False)
    config.update(data_start=str(audit.quote_date.min().date()), data_end=str(audit.quote_date.max().date()),
                  sessions=int(audit.quote_date.nunique()), max_calendar_dte=int(audit.dte.max()),
                  source_chains=int(len(audit) // len(alphas)),
                  caveats=["Relative valuation under a chosen alpha/anchor, not proof of absolute mispricing or positive trading returns.",
                           "All observed source expiries; calendar DTE from source dates, no monthly alias or expiry settlement assumed. 0 DTE excluded.",
                           "Quotes/ATM IV must be valid on the same date and source expiry. No stale-IV carry, interpolation, or future data.",
                           "Anchor is the nearest eligible quote to 3 sigma within 0.25; actual anchor coordinates and prices are audited.",
                           ("Mid-anchored model versus mid plus USD 1 per contract per side (SPX x100). Mid fills are hypothetical; gross mid-to-mid discount is also saved in CSV."
                            if price_basis == "mid" else "Bid-anchored model versus ask plus USD 1 per contract per side (SPX x100). Mid-to-mid comparison is also saved in CSV."),
                           "Both price bases use identical quote eligibility, anchor selection and bid-based slope/splice screening to isolate the pricing effect.",
                           "Bid > 0, spread/mid <= 50%, IV 3%-300%, valid delta; ATM delta -0.65 to -0.35. Duplicate strikes excluded.",
                           "Anchor slope/splice checks are partial zero-rate bounds, not a complete static-arbitrage calibration. No yield curves are available.",
                           "No verified AM/PM root identifiers, open interest, synchronized exchange timestamps, or order-book depth. Liquidity is screened by quoted spread only.",
                           "Coverage/frequency are conditional on eligible quotes. Missing observations are not zero discounts. Alpha comparisons may have different coverage.",
                           "Sessions repeat overlapping contracts and are not independent trials. There is no 2008 data or out-of-sample trading test."])
    (output / "manifest.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_report(output, summary, audit, config)
    print(summary[summary.alpha.eq(alpha) & summary.sufficient_coverage].sort_values("median_discount", ascending=False).head(12).to_string(index=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data-parquet"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--alpha", type=float, default=3.)
    parser.add_argument("--limit-files", type=int)
    parser.add_argument("--min-days", type=int, default=100)
    parser.add_argument("--min-years", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--price-basis", choices=["bid_ask", "mid"], default="bid_ask")
    args = parser.parse_args()
    run(args.data_dir, args.output, args.alpha, args.limit_files, args.min_days, args.min_years, args.resume, args.workers, args.price_basis)
