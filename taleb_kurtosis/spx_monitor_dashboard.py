"""Standalone delayed SPX tail monitor; no order or account access."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taleb_kurtosis.spx_monitor import (
    REFRESH_SECONDS, FeedError, evaluate_snapshot, fetch_snapshot, freshness,
)

GREEN, RED, TEAL = "#087f4f", "#b42338", "#007c91"


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def get_snapshot():
    try:
        return fetch_snapshot(), None
    except FeedError as exc:
        return None, str(exc)


def chart_style(fig, title, height=390):
    fig.update_layout(title=title, height=height, margin=dict(l=20, r=20, t=55, b=45),
                      paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#25282b", size=12),
                      legend=dict(orientation="h", y=-.2), hovermode="closest")
    fig.update_xaxes(gridcolor="#eeeeee", zeroline=False)
    fig.update_yaxes(gridcolor="#eeeeee", zeroline=False)
    return fig


def discount_map(q):
    edges = [3, 3.5, 4, 4.5, 5, 6, 7, 8, 10, 12, np.inf]
    labels = ["3-3.5", "3.5-4", "4-4.5", "4.5-5", "5-6", "6-7", "7-8", "8-10", "10-12", "12+"]
    q = q.assign(sigma_bin=pd.cut(q.sigma, edges, labels=labels, right=False),
                 chain=q.series + " / " + q.expiry.dt.strftime("%d.%m.%y") + "<br>" + q.dte.astype(str) + " DTE")
    grid = q.pivot_table(index="chain", columns="sigma_bin", values="discount", aggfunc="median", observed=True)
    grid = grid.reindex(columns=labels).sort_index()
    counts = q.pivot_table(index="chain", columns="sigma_bin", values="discount", aggfunc="size", observed=True)
    counts = counts.reindex(index=grid.index, columns=labels).fillna(0)
    fig = go.Figure(go.Heatmap(x=labels, y=grid.index, z=grid.to_numpy() * 100, customdata=counts.to_numpy(),
                             colorscale=[[0, RED], [.5, "#ffffff"], [1, GREEN]], zmin=-50, zmax=50,
                             hoverongaps=False, xgap=2, ygap=2,
                             colorbar=dict(title="%", thickness=12),
                             hovertemplate="%{y}<br>%{x} sigma<br>Mediana: %{z:.2f}%<br>Kontrakty: %{customdata}<extra></extra>"))
    fig.update_xaxes(title="Sigma z ATM IV", type="category", categoryorder="array", categoryarray=labels,
                     tickangle=-45, tickfont_size=10)
    fig.update_yaxes(type="category", tickfont_size=10)
    return chart_style(fig, "Mediana dyskonta (%)", max(300, 120 + 55 * len(grid)))


def curve_figures(q, smile, anchor):
    q = q.sort_values("STRIKE")
    prices = go.Figure()
    for col, label, color in [("P_BID", "Bid", "#858585"), ("P_ASK", "Ask", RED),
                              ("mid", "Mid", TEAL), ("model_price", "Model", GREEN)]:
        prices.add_scatter(x=q.STRIKE, y=q[col], mode="lines+markers", name=label,
                           line=dict(color=color, width=2), marker_size=4)
    prices.update_yaxes(type="log", title="Punkty indeksowe / log")
    prices.update_xaxes(title="Strike")
    smile = smile[smile.eligible_quote & smile.sigma.between(-.5, 12)].sort_values("sigma")
    iv = go.Figure(go.Scatter(x=smile.sigma, y=smile.P_IV * 100, mode="lines+markers",
                              line=dict(color=TEAL), marker_size=4, customdata=smile.STRIKE,
                              hovertemplate="%{x:.2f} sigma<br>IV %{y:.2f}%<br>Strike %{customdata}<extra></extra>"))
    iv.add_vline(x=anchor, line_dash="dash", line_color="#767676")
    iv.add_vrect(x0=6, x1=8, fillcolor=GREEN, opacity=.08, line_width=0)
    iv.update_xaxes(title="Sigma z ATM IV")
    iv.update_yaxes(title="IV (%)")
    return chart_style(prices, "Cena rynkowa i model"), chart_style(iv, "Usmiech IV / puty")


def render_snapshot(snapshot, dte_range, alpha, fee, spread, basis, series, sigma_range, status, allow_stale):
    now = pd.Timestamp.now(tz="UTC")
    fresh, message = freshness(snapshot, now)
    st.warning("Cboe / opoznienie ok. 15 min / wycena orientacyjna. Brak czasu aktualizacji kazdego bid/ask.")
    st.caption(f"Snapshot UTC: {snapshot.generated_at:%Y-%m-%d %H:%M:%S} | "
               f"SPX UTC: {snapshot.underlying_at:%Y-%m-%d %H:%M:%S} | "
               f"Pobrano UTC: {snapshot.fetched_at:%Y-%m-%d %H:%M:%S}")
    if not fresh:
        st.error(message)
        if not allow_stale or message.startswith("Niespojny"):
            return
        st.warning("ARCHIWALNY SNAPSHOT: ranking dotyczy daty danych, nie biezacego rynku.")
    ranked, audit, smiles = evaluate_snapshot(snapshot, *dte_range, alpha, fee, spread, basis)
    if not fresh and not ranked.empty:
        ranked = ranked.assign(data_type="historical_snapshot")
    if series != "Wszystkie":
        ranked = ranked[ranked.series.eq(series)] if not ranked.empty else ranked
        audit = audit[audit.series.eq(series)] if not audit.empty else audit
    if not ranked.empty:
        ranked = ranked[ranked.sigma.between(*sigma_range)]
    a, b, c, d = st.columns(4)
    a.metric("SPX", f"{snapshot.spot:,.2f}")
    b.metric("Lancuchy / wycenione", f"{int(audit.reason.eq('ok').sum()) if not audit.empty else 0}")
    c.metric("Ponizej modelu", str(int(ranked.discount.gt(0).sum())) if not ranked.empty else "0")
    d.metric("Powyzej modelu", str(int(ranked.discount.lt(0).sum())) if not ranked.empty else "0")
    st.caption("Dyskonto = 1 - (cena + koszty obu stron) / model. Wycena wzgledna, nie prognoza zysku. Mid nie gwarantuje wykonania.")
    ranking, curves, diagnostics = st.tabs(["Wycena", "Lancuch i IV", "Jakosc danych"])
    with ranking:
        if ranked.empty:
            st.info("Brak wycenionych kontraktow dla wybranych filtrow. Szczegoly odrzucen: Jakosc danych.")
        else:
            st.plotly_chart(discount_map(ranked), width="stretch", config={"displayModeBar": False})
            selected = ranked if status == "Wszystkie" else ranked[ranked.valuation.eq(status)]
            columns = ["option", "series", "expiry", "dte", "STRIKE", "sigma", "P_BID", "mid", "P_ASK",
                       "model_price", "discount", "gross_mid_discount", "bid_ask_discount", "relative_spread",
                       "valuation", "robust_underpriced", "P_IV", "atm_iv", "cost_usd", "open_interest"]
            table = selected[columns].copy()
            styles = table.style.map(lambda v: f"color: {GREEN if v > 0 else RED if v < 0 else '#555'}",
                                      subset=["discount", "gross_mid_discount", "bid_ask_discount"])
            config = {col: st.column_config.NumberColumn(label, format="percent") for col, label in [
                ("discount", "Dyskonto netto"), ("gross_mid_discount", "Mid / mid brutto"),
                ("bid_ask_discount", "Bid / ask netto"), ("relative_spread", "Spread / mid"),
                ("P_IV", "IV puta"), ("atm_iv", "ATM IV")]}
            config.update({"expiry": st.column_config.DateColumn("Wygasanie"), "STRIKE": "Strike",
                           "model_price": st.column_config.NumberColumn("Model (pkt)", format="%.3f"),
                           "sigma": st.column_config.NumberColumn("Sigma", format="%.2f"),
                           "robust_underpriced": "Dyskonto: alpha 2.5 / 3 / 3.5", "cost_usd": "Koszt + 2 prowizje (USD)"})
            st.dataframe(styles, hide_index=True, width="stretch", column_config=config, height=410)
            st.download_button("CSV", selected.to_csv(index=False), "spx_tail_valuation.csv", icon=":material/download:")
    with curves:
        if ranked.empty:
            st.info("Brak lancucha z poprawna kotwica 3 sigma w wybranym zakresie.")
        else:
            choices = ranked[["series", "expiry", "dte"]].drop_duplicates().sort_values(["expiry", "series"])
            keys = list(zip(choices.series, choices.expiry))
            chosen = st.selectbox("Seria / wygasanie", keys, format_func=lambda x: f"{x[0]} / {x[1]:%Y-%m-%d}")
            q = ranked[ranked.series.eq(chosen[0]) & ranked.expiry.eq(chosen[1])]
            full = smiles[smiles.series.eq(chosen[0]) & smiles.expiry.eq(chosen[1])]
            row = q.iloc[0]
            st.caption(f"ATM IV {row.atm_iv:.2%} | {row.dte} DTE | kotwica K {row.anchor_strike:g} / "
                       f"{row.anchor_sigma:.3f} sigma / mid {row.anchor_mid:.3f} pkt | alpha {alpha:g}")
            for fig in curve_figures(q, full, row.anchor_sigma):
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            sensitivity = q[["option", "sigma", "discount_alpha_2.5", "discount_alpha_3", "discount_alpha_3.5"]]
            st.dataframe(sensitivity, hide_index=True, width="stretch", column_config={
                f"discount_alpha_{a}": st.column_config.NumberColumn(f"Alpha {a}", format="percent")
                for a in ["2.5", "3", "3.5"]})
    with diagnostics:
        st.dataframe(audit, hide_index=True, width="stretch")
        st.caption(f"Kontrakty w feedzie: {snapshot.raw_contracts:,} | puty SPX/SPXW: {len(snapshot.puts):,} | "
                   f"Niepoprawne identyfikatory: {snapshot.rejected_symbols}")
        st.caption("SPX i SPXW sa liczone osobno. Kotwica 3 sigma +/- 0.25; brak kotwicy lub ATM IV blokuje wycene. "
                   "Kontrola nachylenia kotwicy pozostaje po bid, jak w historycznej heatmapie mid.")
        st.caption("Sigma = log(S/K) / (ATM IV * sqrt(DTE/365)). Alpha jest zalozeniem, nie estymacja z przyszlych zwrotow. "
                   "Znacznik snapshotu: UTC; czas SPX bez strefy: America/New_York. Brak potwierdzenia synchronizacji kazdej opcji z indeksem.")
        st.link_button("Cboe / SPX", "https://www.cboe.com/delayed_quotes/spx/quote_table/", icon=":material/open_in_new:")
        st.link_button("Model / artykul", "https://arxiv.org/abs/1908.02347", icon=":material/article:")


def main():
    st.set_page_config(page_title="SPX / Tail Monitor", page_icon=":material/query_stats:", layout="wide")
    st.markdown("""<style>
    .block-container {max-width:1500px;padding-top:2rem;}
    h1 {font-size:1.8rem !important;letter-spacing:0 !important;}
    h2,h3,p,span,label {letter-spacing:0 !important;}
    [data-testid="stMetricValue"] {font-size:1.65rem;}
    @media(max-width:600px) { .block-container {padding:1rem;} }
    </style>""", unsafe_allow_html=True)
    st.title("SPX / Tail Monitor")
    st.caption("PUT / 180 DTE / POWER LAW / KOTWICA 3 SIGMA")
    with st.sidebar:
        st.subheader("Rynek i wycena")
        st.caption("Cboe / SPX + SPXW / opoznione")
        dte = st.slider("Dni do wygasniecia", 30, 365, (165, 195), step=1)
        series = st.selectbox("Seria", ["Wszystkie", "SPX", "SPXW"])
        basis_label = st.segmented_control("Ceny", ["Mid / mid", "Bid / ask"], default="Mid / mid")
        alpha = st.number_input("Alpha", min_value=1.1, max_value=6., value=3., step=.25)
        fee = st.number_input("Prowizja / kontrakt / strona (USD)", min_value=0., max_value=20., value=1., step=.25)
        spread = st.slider("Maks. spread / mid (%)", 5, 100, 50, step=5) / 100
        sigma = st.slider("Sigma / ranking", 3., 15., (3., 12.), step=.5)
        status = st.selectbox("Wycena / tabela", ["Wszystkie", "Ponizej modelu", "Powyzej modelu"])
        auto = st.toggle("Automatyczne odswiezanie", value=True)
        allow_stale = st.checkbox("Podglad nieaktualnego snapshotu", value=False)
        st.button("", icon=":material/refresh:", help="Odswiez widok. Wspolny cache API: 5 minut.")
        st.caption("API co 5 min podczas otwartej sesji. Tylko odczyt; bez zlecen.")

    @st.fragment(run_every=60 if auto else None)
    def panel():
        with st.spinner("Pobieranie SPX..."):
            snapshot, error = get_snapshot()
        if snapshot is not None:
            st.session_state["last_spx_snapshot"] = snapshot
        else:
            st.error(error)
            snapshot = st.session_state.get("last_spx_snapshot")
            if snapshot is not None:
                st.warning("Ostatni poprawny snapshot; odswiezenie API nie powiodlo sie.")
        if snapshot is None:
            st.info("Brak danych rynkowych. Nie podstawiono danych historycznych ani syntetycznych.")
            st.link_button("Status / zrodlo Cboe", "https://www.cboe.com/delayed_quotes/spx/quote_table/")
            return
        render_snapshot(snapshot, dte, alpha, fee, spread, "bid_ask" if basis_label == "Bid / ask" else "mid",
                        series, sigma, status, allow_stale)
    panel()


if __name__ == "__main__":
    main()
