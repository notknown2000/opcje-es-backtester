from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from taleb_kurtosis.power_law_backtest import LABELS, aggregate_frequencies, report_figures
from taleb_kurtosis.power_law_live import DataUnavailable, discover_chains, discover_futures, fetch_chain, rank_snapshot


def price_figure(q):
    fig = go.Figure()
    q = q.sort_values("STRIKE")
    for column, label, color in [("P_BID", "Bid", "#737373"), ("P_ASK", "Ask", "#c43e55"),
                                 ("model_price", "Power law", "#148579")]:
        fig.add_scatter(x=q.STRIKE, y=q[column], mode="lines+markers", name=label, line_color=color)
    fig.update_layout(xaxis_title="Strike", yaxis_title="Cena (punkty indeksowe)", yaxis_type="log",
                      height=430, margin=dict(l=30, r=15, t=30, b=30), legend=dict(orientation="h"))
    return fig


@st.cache_data(ttl=60, show_spinner=False)
def read_results(path, revision):
    base = Path(path)
    return {name: pd.read_parquet(base / f"{name}.parquet") for name in
            ["daily_frequencies", "sensitivity", "examples"]}


def render_backtest(results_dir, summary, trades, equity):
    base = Path(results_dir)
    manifest = json.loads((base / "manifest.json").read_text())
    data = read_results(results_dir, (base / "manifest.json").stat().st_mtime_ns)
    st.subheader("Power Law / SPX Long Tail")
    st.caption(f"{manifest['data_start']} - {manifest['data_end']} | alpha 2.75 | kotwica 2.5 sigma ATM IV | SPX x100")
    st.warning("Niedowartosciowanie wzgledem kotwicy i przyjetego alpha, nie prognoza dodatniego PnL. Prog: model >= 110% (ask + koszty).")
    tabs = st.tabs(["Mapa wycen", "Backtest", "Alpha i kotwica", "Lancuchy", "Audyt"])
    frequencies = aggregate_frequencies(data["daily_frequencies"])
    figures = report_figures(summary, equity, frequencies)
    with tabs[0]:
        years = sorted(pd.to_datetime(data["daily_frequencies"].QUOTE_DATE).dt.year.unique())
        selected = st.selectbox("Okres", ["Caly okres"] + [str(y) for y in years])
        daily = data["daily_frequencies"]
        if selected != "Caly okres":
            daily = daily[pd.to_datetime(daily.QUOTE_DATE).dt.year.eq(int(selected))]
        freq = aggregate_frequencies(daily)
        st.plotly_chart(report_figures(summary, equity, freq)["frequency"], width="stretch")
        minimum = st.number_input("Minimum sesji w przedziale", 1, 4000, min(100, len(daily.QUOTE_DATE.unique())), step=25)
        display = freq[freq.days.ge(minimum)].sort_values("frequency", ascending=False)
        st.dataframe(display, hide_index=True, width="stretch", column_config={
            "frequency": st.column_config.NumberColumn("Czestosc", format="percent"),
            "robust_frequency": st.column_config.NumberColumn("Wszystkie 9 modeli", format="percent"),
            "mean_daily_median_edge": st.column_config.NumberColumn("Model / koszt - 1", format="percent")})
        st.download_button("CSV", display.to_csv(index=False), "power_law_frequency.csv", icon=":material/download:")
    with tabs[1]:
        st.link_button("Scaleout 3 / 4 / 5 sigma", "?view=power-law-scaleout", icon=":material/open_in_new:")
        st.caption("Kapital 250 000 USD | budzet miesieczny 625 USD | sygnal EOD -> kolejna sesja ask | sprzedaz bid | 1 USD/kontrakt/strona")
        table = summary.assign(variant=summary.strategy_id.map(LABELS))[
            ["variant", "trades", "total_pnl", "max_drawdown", "max_drawdown_pct", "premium_spent"]]
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "variant": "Wariant", "trades": "Transakcje", "total_pnl": st.column_config.NumberColumn("PnL USD", format="%.0f"),
            "max_drawdown": st.column_config.NumberColumn("Max DD USD", format="%.0f"),
            "max_drawdown_pct": st.column_config.NumberColumn("Max DD %", format="percent"),
            "premium_spent": st.column_config.NumberColumn("Premie + wejscia USD", format="%.0f")})
        if summary.unresolved.gt(0).any() or summary.stale_sessions.gt(0).any():
            st.warning("Wynik zawiera nieaktualne wyceny lub nierozliczone pozycje.")
        st.plotly_chart(figures["equity"], width="stretch")
        st.plotly_chart(figures["drawdown"], width="stretch")
        strategy = st.selectbox("Transakcje", summary.strategy_id.tolist(), format_func=lambda x: LABELS.get(x, x))
        st.dataframe(trades[trades.strategy_id.eq(strategy)], hide_index=True, width="stretch")
    with tabs[2]:
        s = data["sensitivity"]
        available = s[s.quotes.gt(0)]
        per_day = available.groupby(["quote_date", "alpha", "anchor_sigma"], as_index=False).frequency.mean()
        table = per_day.groupby(["alpha", "anchor_sigma"], as_index=False).agg(frequency=("frequency", "mean"), sessions=("frequency", "size"))
        st.plotly_chart(px.density_heatmap(table, x="anchor_sigma", y="alpha", z="frequency", histfunc="avg",
                                           color_continuous_scale="Cividis", title="Alpha / kotwica"), width="stretch")
        st.caption("Czestosc jest warunkowa: kombinacje maja rozne zbiory kwalifikujacych sie opcji.")
        st.dataframe(table, hide_index=True, width="stretch")
    with tabs[3]:
        examples = data["examples"]
        dates = sorted(examples.QUOTE_DATE.unique())
        date = st.selectbox("Sesja", dates, format_func=lambda x: str(pd.Timestamp(x).date()))
        q = examples[examples.QUOTE_DATE.eq(date)]
        expiry = st.selectbox("Wygasanie", sorted(q.EXPIRE_DATE.unique()), format_func=lambda x: str(pd.Timestamp(x).date()))
        q = q[q.EXPIRE_DATE.eq(expiry)]
        st.plotly_chart(price_figure(q), width="stretch")
        st.dataframe(q[["STRIKE", "P_DELTA", "sigma", "P_BID", "P_ASK", "model_price", "edge", "model_low", "model_high", "sensitivity_models"]], hide_index=True, width="stretch")
    with tabs[4]:
        st.json(manifest, expanded=False)
        audit = pd.read_csv(base / "scan_audit.csv")
        st.dataframe(audit.groupby("reason", as_index=False).agg(chains=("rows", "size"), rows=("rows", "sum")), hide_index=True)
        st.link_button("Artykul", "https://arxiv.org/abs/1908.02347", icon=":material/open_in_new:")


@st.cache_data(ttl=60, show_spinner=False)
def read_scaleout_results(path, revision):
    base = Path(path)
    return {name: pd.read_parquet(base / f"{name}.parquet") for name in
            ["entries", "sales", "signals", "position_paths"]}


def render_scaleout(results_dir, summary, trades, equity):
    from taleb_kurtosis.power_law_scaleout import POLICIES, figures

    base = Path(results_dir)
    manifest = json.loads((base / "manifest.json").read_text())
    data = read_scaleout_results(results_dir, (base / "manifest.json").stat().st_mtime_ns)
    st.subheader("Power Law / Scaleout")
    st.caption(f"{manifest['data_start']} - {manifest['data_end']} | SPX x100 | te same wejscia | kapital 250 000 USD")
    st.caption("Progi 3, 4, 5... sigma: skumulowany spadek logarytmiczny od zakupu. "
               "Sigma = ATM IV na wejsciu x sqrt(DTE na wejsciu / 365), bez aktualizacji.")
    if manifest.get("entries_with_carried_atm_iv", 0):
        st.caption(f"Niepoprawne IV ATM przy zakupie: {manifest['entries_with_carried_atm_iv']} wejsc. "
                   "Ostatni poprawny odczyt tej samej serii, maks. 5 dni wczesniej; daty w audycie.")
    st.caption("50% pozostalych longow na kazdym nowym progu. Sygnal EOD, sprzedaz na kolejnej dostepnej sesji po bid. "
               "Reszta: 30 DTE / koniec danych, nie wygasanie. Bez warunku dodatniego zysku przy sprzedazy.")
    st.warning("Cale kontrakty: polowa zaokraglona w dol, ostatni kontrakt zostaje do 30 DTE. "
               "Ulamki: teoretyczne 50% / 25% / 12.5%...; niewykonalne dla pojedynczych kontraktow SPX.")
    selection = st.selectbox("Dobor putow", ["power_law", "delta_1pct"],
                             format_func=lambda x: "Model power law" if x == "power_law" else "Delta okolo 1%",
                             key="pl_scaleout_selection")
    subset = summary[summary.selection.eq(selection)].copy()
    ids = subset.strategy_id.tolist()
    eq = equity[equity.strategy_id.isin(ids)].copy()
    labels = dict(zip(subset.strategy_id, subset.exit_policy.map(POLICIES)))
    eq["label"] = eq.strategy_id.map(labels)
    tabs = st.tabs(["Porownanie", "Pakiety", "Audyt"])
    with tabs[0]:
        table = subset.assign(variant=subset.exit_policy.map(POLICIES))[
            ["variant", "trades", "total_pnl", "max_drawdown", "max_drawdown_pct", "triggered_packages", "monetized_packages", "scaleout_sales"]]
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "variant": "Wyjscie", "trades": "Pakiety", "total_pnl": st.column_config.NumberColumn("PnL USD", format="%.2f"),
            "max_drawdown": st.column_config.NumberColumn("Max DD USD", format="%.2f"),
            "max_drawdown_pct": st.column_config.NumberColumn("Max DD %", format="percent"),
            "triggered_packages": "Pakiety z sygnalem", "monetized_packages": "Pakiety ze sprzedaza sigma",
            "scaleout_sales": "Sprzedaze sigma"})
        for fig in figures(eq).values():
            st.plotly_chart(fig, width="stretch")
        annual = eq.sort_values("quote_date").assign(year=eq.quote_date.dt.year).groupby(["strategy_id", "year"]).equity.last()
        annual = annual.groupby(level=0).diff().fillna(annual - manifest["config"]["initial_capital"])
        annual = annual.rename("PnL USD").reset_index().assign(variant=lambda f: f.strategy_id.map(labels))
        st.dataframe(annual.pivot(index="year", columns="variant", values="PnL USD"), width="stretch")
        sales = data["sales"]
        sales = sales[sales.strategy_id.isin(ids) & sales.level.notna()].copy()
        if not sales.empty:
            sales["variant"] = sales.strategy_id.map(labels)
            st.dataframe(sales.groupby(["variant", "level"], as_index=False).agg(
                sales=("qty", "size"), contracts=("qty", "sum"), proceeds_usd=("proceeds", "sum"),
                allocated_pnl_usd=("pnl", "sum")), hide_index=True, width="stretch")
        st.download_button("CSV", table.to_csv(index=False), f"{selection}_scaleout.csv", icon=":material/download:")
    with tabs[1]:
        entries = data["entries"][data["entries"].selection.eq(selection)].set_index("package_id")
        signals = data["signals"]
        signals = signals[signals.strategy_id.eq(f"{selection}_scaleout_integer")]
        levels = signals.groupby("package_id")["level"].max().to_dict()
        packages = sorted(entries.index, key=lambda p: (-levels.get(p, 0), entries.loc[p, "entry_date"]))
        package = st.selectbox("Pakiet", packages, key=f"pl_scaleout_package_{selection}", format_func=lambda p:
                              f"{str(entries.loc[p, 'entry_date'])[:10]} / K {entries.loc[p, 'strike']:g} / {entries.loc[p, 'qty']:g} szt. / prog {levels.get(p, '-')}")
        entry = entries.loc[package]
        st.caption(f"SPX przy zakupie: {entry.entry_spot:,.2f} | ATM IV: {entry.entry_atm_iv:.2%} z {str(entry.atm_quote_date)[:10]} | "
                   f"DTE: {entry.entry_dte:.0f} | sigma horyzontu: {entry.sigma_horizon:.2%}")
        paths = data["position_paths"]
        paths = paths[paths.package_id.eq(package) & paths.strategy_id.isin(ids)].copy()
        paths["label"] = paths.strategy_id.map(labels)
        spot = paths[paths.strategy_id.eq(f"{selection}_30dte")]
        fig = px.line(spot, x="quote_date", y="spot", title="SPX / progi zamrozone przy zakupie")
        for level in range(3, max(5, int(levels.get(package, 5))) + 1):
            fig.add_hline(y=entry.entry_spot * math.exp(-level * entry.sigma_horizon),
                          line_dash="dot", annotation_text=f"{level} sigma", annotation_position="top left")
        fig.update_layout(height=380, margin=dict(l=30, r=15, t=55, b=30), xaxis_title=None, yaxis_title="SPX")
        st.plotly_chart(fig, width="stretch")
        for column, title in [("package_pnl", "PnL pakietu (USD)"), ("remaining", "Pozostale kontrakty")]:
            fig = px.line(paths, x="quote_date", y=column, color="label", title=title,
                          line_shape="hv" if column == "remaining" else "linear")
            fig.update_layout(height=460, margin=dict(l=30, r=15, t=50, b=130), xaxis_title=None,
                              yaxis_title=None, legend=dict(orientation="h", y=-.25, title_text=""))
            st.plotly_chart(fig, width="stretch")
        policy = st.selectbox("Sprzedaze", ["scaleout_integer", "scaleout_fractional", "30dte", "tp5"],
                              format_func=POLICIES.get, key="pl_scaleout_policy")
        fills = data["sales"]
        fills = fills[fills.package_id.eq(package) & fills.strategy_id.eq(f"{selection}_{policy}")]
        st.dataframe(fills[["signal_date", "exit_date", "reason", "qty", "bid", "proceeds", "pnl", "remaining"]], hide_index=True, width="stretch")
        st.dataframe(trades[trades.package_id.eq(package) & trades.strategy_id.isin(ids)][
            ["strategy_id", "entry_cost", "proceeds", "pnl", "exit_date", "exit_reason"]], hide_index=True, width="stretch")
    with tabs[2]:
        st.json(manifest, expanded=False)
        st.dataframe(subset[["label", "premium_spent", "unresolved", "stale_sessions"]], hide_index=True, width="stretch")
        if subset.unresolved.gt(0).any() or subset.stale_sessions.gt(0).any():
            st.warning("Wynik zawiera nieaktualne wyceny lub nierozliczone pozycje.")
        st.dataframe(data["signals"][data["signals"].strategy_id.isin(ids)], hide_index=True, width="stretch")
        st.dataframe(data["entries"][data["entries"].selection.eq(selection)][
            ["package_id", "entry_date", "atm_quote_date", "atm_strike", "entry_atm_iv", "atm_iv_carried", "sigma_horizon"]],
            hide_index=True, width="stretch")


def render_live():
    st.subheader("Power Law / ES - MES")
    st.caption("CME | tylko odczyt | ES x50 USD, MES x5 USD | ceny w punktach indeksowych")
    cols = st.columns(3)
    symbol = cols[0].selectbox("Produkt", ["ES", "MES"])
    alpha = cols[1].number_input("Alpha", 1.1, 8., 2.75, .05)
    anchor = cols[2].number_input("Kotwica (sigma ATM IV)", 1.5, 5., 2.5, .25)
    provider = st.selectbox("Zrodlo", ["CME przez IBKR / TWS", "Snapshot CSV", "Yahoo Finance", "Deribit"])
    if provider in {"Yahoo Finance", "Deribit"}:
        st.warning("Brak zweryfikowanego lancucha bid/ask opcji ES/MES z tego zrodla. Ranking niedostepny; brak zamiennika SPY/BTC.")
        return
    identity = (symbol, provider)
    if st.session_state.get("pl_live_identity") != identity:
        for key in ["pl_futures", "pl_chains", "pl_snapshot"]:
            st.session_state.pop(key, None)
        st.session_state["pl_live_identity"] = identity
    if provider == "Snapshot CSV":
        upload = st.file_uploader("Snapshot opcji CME (CSV)", type="csv")
        if upload is None:
            st.info("Brak danych. Wymagany aktualny snapshot bid/ask z identyfikatorem kontraktu bazowego i czasami kwotowan.")
            return
        try:
            frame = pd.read_csv(upload)
        except (ValueError, pd.errors.ParserError):
            st.error("Nieprawidlowy CSV")
            return
    else:
        with st.expander("Polaczenie TWS / Gateway", expanded=True):
            host = st.text_input("Host", "host.docker.internal")
            c1, c2 = st.columns(2)
            port = c1.number_input("Port", 1, 65535, 7497)
            client_id = c2.number_input("Client ID", 1, 9999, 81)
        connection = dict(host=host, port=int(port), client_id=int(client_id))
        fingerprint = (symbol, host, port, client_id)
        if st.session_state.get("pl_connection") != fingerprint:
            for key in ["pl_futures", "pl_chains", "pl_snapshot"]:
                st.session_state.pop(key, None)
            st.session_state["pl_connection"] = fingerprint
        if st.button("Pobierz kontrakty", icon=":material/sync:"):
            try:
                st.session_state.pop("pl_snapshot", None)
                st.session_state["pl_futures"] = discover_futures(symbol, **connection)
            except Exception as exc:
                st.error(str(exc) if isinstance(exc, DataUnavailable) else "Blad pobierania kontraktow CME")
        futures = st.session_state.get("pl_futures", [])
        if not futures:
            st.info("Brak polaczonego lancucha live. Wymagane TWS/Gateway oraz uprawnienia do danych CME.")
            return
        future = st.selectbox("Kontrakt bazowy", futures, format_func=lambda f: f"{f['contract']} / {f['expiry']}")
        if st.button("Pobierz serie opcji", icon=":material/search:"):
            try:
                st.session_state.pop("pl_snapshot", None)
                st.session_state["pl_chains"] = (future["con_id"], discover_chains(symbol, future["con_id"], **connection))
            except Exception as exc:
                st.error(str(exc) if isinstance(exc, DataUnavailable) else "Blad pobierania serii opcji")
        chain_id, chains = st.session_state.get("pl_chains", (None, []))
        if chain_id != future["con_id"] or not chains:
            return
        chain = st.selectbox("Seria", chains, format_func=lambda c: f"{c['trading_class']} / {c['expiry']} / x{c['multiplier']}")
        signature = (future["con_id"], chain["trading_class"], chain["expiry"])
        european = st.checkbox("Zweryfikowano europejski styl wykonania tej serii w specyfikacji CME", key=f"pl_style_{signature}")
        if st.button("Odswiez kwotowania", icon=":material/refresh:", disabled=not european):
            st.session_state.pop("pl_snapshot", None)
            try:
                with st.spinner("CME bid/ask..."):
                    snapshot = fetch_chain(symbol, future["con_id"], chain["trading_class"], chain["expiry"], european, **connection)
                    st.session_state["pl_snapshot"] = (signature, snapshot)
            except Exception as exc:
                st.error(str(exc) if isinstance(exc, DataUnavailable) else "Blad kwotowan; brak rankingu live")
        cached_signature, frame = st.session_state.get("pl_snapshot", (None, None))
        if cached_signature != signature or frame is None:
            return
        st.caption("CME przez IBKR: czas odbioru snapshotu; styl wykonania potwierdzony przez uzytkownika. Nie jest to zweryfikowany timestamp transakcji gieldowej.")
    @st.fragment(run_every="15s")
    def display_snapshot():
        try:
            ranked, audit = rank_snapshot(frame, symbol, alpha, anchor)
        except (ValueError, KeyError, TypeError) as exc:
            st.error(str(exc))
            return
        st.caption(f"Ostatni snapshot: {pd.to_datetime(frame.quote_time, utc=True).max()} | waznosc maks. 120 s | odswiezanie reczne")
        st.warning("Model wzgledny z cena wlasciwego futures jako kotwica bazowa. Nie jest to bezposrednia walidacja wynikow historycznych SPX.")
        if ranked.empty:
            st.info("Brak kwalifikujacych sie opcji / kotwicy")
        else:
            ranked = ranked.sort_values("edge", ascending=False)
            groups = list(ranked.groupby(["option_series", "expiry", "underlying_contract"]).groups)
            selected = st.selectbox("Lancuch cen", groups,
                                    format_func=lambda x: f"{x[0]} / {str(x[1])[:10]} / {x[2]}")
            ranked = ranked[ranked.option_series.eq(selected[0]) & ranked.expiry.eq(selected[1]) & ranked.underlying_contract.eq(selected[2])]
            st.plotly_chart(price_figure(ranked), width="stretch")
            st.dataframe(ranked[["STRIKE", "P_BID", "P_ASK", "P_DELTA", "sigma", "model_price", "edge", "cost_usd", "underpriced"]], hide_index=True, width="stretch")
            st.download_button("CSV", ranked.to_csv(index=False), f"{symbol}_power_law.csv", icon=":material/download:")
        st.dataframe(audit, hide_index=True)
    display_snapshot()
