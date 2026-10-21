from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Streamlit's console entry point adds the script directory, not the repo root.
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DEFAULT_RESULTS_DIR = Path("results/taleb_tail_ratio")
PRESETS = {
    "Brak": {},
    "Power Law / Long Tail": {
        "results_dir": "results/power_law_tail",
        "filters": {},
    },
    "Power Law / Scaleout": {
        "results_dir": "results/power_law_scaleout",
        "filters": {},
    },
    "Digital Tail / Porównanie barier": {
        "results_dir": "results/digital_tail_barrier_comparison",
        "filters": {},
    },
    "Digital Tail / Bariery 250k": {
        "results_dir": "results/digital_tail_barrier_comparison_250k",
        "filters": {},
    },
    "Digital Tail / ATM IV": {
        "results_dir": "results/digital_tail_atm_iv_baseline",
        "filters": {},
    },
    "Weekly 30 DTE": {
        "results_dir": "results/taleb_tail_ratio_weekly_4x20_itm_all_exits",
        "filters": {
            "long_sigma": [4.0],
            "short_sigma": [0.4],
            "ratio_target": [20.0],
            "entry_frequency": ["weekly"],
            "exit_policy": ["exit_at_30_dte"],
        },
    },
}


def opis_wyjscia(exit_policy: object) -> str:
    labels = {
        "hold_to_expiry": "Trzymanie do wygaśnięcia",
        "long_itm_exit": "Zamknięcie po wejściu long puta ITM",
        "exit_at_30_dte": "Zamknięcie przy 30 DTE",
        "combined_exit": "Zamknięcie: ITM albo 30 DTE",
        "shock_exit": "Zamknięcie po dziennym szoku",
        "market_drop_exit": "Zamknięcie po dziennym spadku > 5%",
        "scaleout_longs_5pct_iv40_itm": "Częściowa realizacja longów: -5% / IV>=50% / long ITM",
        "scaleout_longs_close_short_on_first_trigger": "Częściowa realizacja longów + zamknięcie shorta po pierwszym triggerze",
        "profit_exit": "Zamknięcie po celu zysku",
        "scaleout_drop5": "Realizacja paczki longów po spadku > 5%",
        "scaleout_iv40": "Realizacja paczki longów przy IV >= 50%",
        "scaleout_long_itm": "Realizacja paczki longów po wejściu long puta ITM",
        "scaleout_short_core_after_first_trigger": "Zamknięcie short core po pierwszym triggerze longów",
        "hold_to_expiry_last_quote": "Zamknięcie na ostatnim dostępnym kwotowaniu",
        "end_of_data": "Koniec danych",
    }
    return labels.get(str(exit_policy), str(exit_policy))


def opis_konstrukcji(row: pd.Series) -> str:
    parts = [opis_wyjscia(row.get("exit_policy", ""))]
    if "short_sigma" in row and pd.notna(row["short_sigma"]):
        parts.append(f"short {float(row['short_sigma']):g}σ")
    if "long_sigma" in row and pd.notna(row["long_sigma"]):
        parts.append(f"long {float(row['long_sigma']):g}σ")
    if "ratio_target" in row and pd.notna(row["ratio_target"]):
        parts.append(f"ratio {int(float(row['ratio_target']))}")
    elif row.get("ratio_policy") == "max_funded":
        parts.append("max funded")
    return " | ".join(parts)


def opis_wartosci_filtra(column: str, value: object) -> str:
    if column == "exit_policy":
        return opis_wyjscia(value)
    if column == "entry_frequency":
        return {
            "daily": "codziennie",
            "weekly": "co tydzień",
            "every_n_days": "co N dni",
        }.get(str(value), str(value))
    if column == "ratio_policy":
        return {
            "target": "docelowe ratio",
            "max_funded": "maksymalnie finansowane",
            "fixed": "stała konstrukcja",
        }.get(str(value), str(value))
    if column == "sigma_source":
        return {
            "realized": "realized vol",
            "atm_put_iv_60d": "ATM put IV przeskalowana do 60 dni",
        }.get(str(value), str(value))
    return str(value)


def available_results_dirs(root: Path = Path("results")) -> list[str]:
    if not root.exists():
        return [str(DEFAULT_RESULTS_DIR)]
    dirs = sorted(str(path) for path in root.iterdir() if path.is_dir())
    default = str(DEFAULT_RESULTS_DIR)
    if default not in dirs:
        dirs.insert(0, default)
    return dirs


@st.cache_data(show_spinner=False)
def load_results(results_dir: str, revision: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path(results_dir)
    summary_path = base / "summary.csv"
    trades_path = base / "trades.parquet"
    equity_path = base / "daily_equity.parquet"

    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
    equity = pd.read_parquet(equity_path) if equity_path.exists() else pd.DataFrame()

    for frame in [trades, equity]:
        for column in ["entry_date", "exit_date", "expire_date", "quote_date"]:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column])
    if not summary.empty:
        summary["opis_wyjscia"] = summary["exit_policy"].map(opis_wyjscia) if "exit_policy" in summary.columns else ""
        summary["opis_strategii"] = summary.apply(opis_konstrukcji, axis=1)
    if not trades.empty:
        trades["opis_wyjscia"] = trades["exit_reason"].map(opis_wyjscia) if "exit_reason" in trades.columns else ""
    return summary, trades, equity


def ensure_filter_state(frame: pd.DataFrame, column: str, key: str) -> list[object]:
    if column not in frame.columns or frame.empty:
        st.session_state[key] = []
        return []
    values = sorted(frame[column].dropna().unique().tolist())
    current = st.session_state.get(key)
    if not current:
        st.session_state[key] = values
        return values
    valid = [value for value in current if value in values]
    if not valid:
        valid = values
    st.session_state[key] = valid
    return valid


def multiselect_existing(label: str, frame: pd.DataFrame, column: str, key: str) -> list[object]:
    if column not in frame.columns or frame.empty:
        return []
    values = sorted(frame[column].dropna().unique().tolist())
    ensure_filter_state(frame, column, key)
    return st.sidebar.multiselect(
        label,
        values,
        default=st.session_state[key],
        key=key,
        format_func=lambda value: opis_wartosci_filtra(column, value),
    )


def apply_filters(frame: pd.DataFrame, filters: dict[str, list[object]]) -> pd.DataFrame:
    filtered = frame.copy()
    for column, values in filters.items():
        if column in filtered.columns and values:
            filtered = filtered[filtered[column].isin(values)]
    return filtered


def drawdown_frame(equity: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return equity
    rows = []
    for strategy, group in equity.sort_values("quote_date").groupby("strategy_id", dropna=False):
        group = group.copy()
        group["peak"] = group["equity_pnl"].cummax()
        group["drawdown"] = group["equity_pnl"] - group["peak"]
        group["strategy_id"] = strategy
        rows.append(group)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def apply_selected_preset(preset_name: str | None = None) -> None:
    preset_name = preset_name or st.session_state.get("preset_name", "Brak")
    preset = PRESETS.get(preset_name, {})
    if not preset:
        return
    results_dir = str(preset.get("results_dir", DEFAULT_RESULTS_DIR))
    st.session_state["preset_name"] = preset_name
    st.session_state["selected_results_dir"] = results_dir
    st.session_state["use_custom_dir"] = False
    if results_dir.startswith("results/digital_tail_barrier_comparison"):
        st.session_state["digital_cmp_dimension"] = "Odległość spreadu"
        st.session_state["digital_cmp_policy"] = "Bariera"
    for column, values in preset.get("filters", {}).items():
        st.session_state[f"filter_{column}"] = values


def trade_risk_budget(row: pd.Series) -> float | None:
    if "legs_json" not in row or "net_entry_cashflow" not in row:
        return None
    try:
        legs = json.loads(row["legs_json"])
    except Exception:
        return None
    put_longs = [float(leg["strike"]) for leg in legs if leg.get("option_type") == "put" and int(leg.get("qty", 0)) > 0]
    put_shorts = [float(leg["strike"]) for leg in legs if leg.get("option_type") == "put" and int(leg.get("qty", 0)) < 0]
    if not put_longs or not put_shorts:
        return None
    long_strike = max(put_longs)
    short_strike = min(put_shorts)
    width = max(short_strike - long_strike, 0.0)
    budget = width * 100.0 - float(row["net_entry_cashflow"])
    return budget if budget > 0 else None


def risk_budget_from_legs(legs: list[dict[str, object]], net_entry_cashflow: float) -> float | None:
    put_longs = [
        float(leg["strike"])
        for leg in legs
        if leg.get("option_type") == "put" and int(leg.get("qty", 0)) > 0
    ]
    put_shorts = [
        float(leg["strike"])
        for leg in legs
        if leg.get("option_type") == "put" and int(leg.get("qty", 0)) < 0
    ]
    if not put_longs or not put_shorts:
        return None
    long_strike = max(put_longs)
    short_strike = min(put_shorts)
    width = max(short_strike - long_strike, 0.0)
    budget = width * 100.0 - net_entry_cashflow
    return budget if budget > 0 else None


def is_scaleout_trades_frame(trades: pd.DataFrame) -> bool:
    if trades.empty or "exit_policy" not in trades.columns:
        return False
    return trades["exit_policy"].astype(str).isin(
        ["scaleout_longs_5pct_iv40_itm", "scaleout_longs_close_short_on_first_trigger"]
    ).any()


def scaleout_component_count(trades: pd.DataFrame) -> int:
    if "package_id" not in trades.columns or trades.empty:
        return 1
    counts = trades.groupby("package_id", dropna=False).size()
    if counts.empty:
        return 1
    return max(1, int(round(float(counts.median()))))


def rollup_scaleout_trades_for_kelly(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or not {"package_id", "legs_json", "net_entry_cashflow", "pnl"}.issubset(trades.columns):
        return trades.copy()

    rolled_rows: list[dict[str, object]] = []
    for package_id, group in trades.groupby("package_id", dropna=False):
        legs: list[dict[str, object]] = []
        for legs_json in group["legs_json"].dropna():
            try:
                loaded_legs = json.loads(legs_json)
            except Exception:
                continue
            if isinstance(loaded_legs, list):
                legs.extend(loaded_legs)

        net_entry_cashflow = float(group["net_entry_cashflow"].sum())
        risk_budget = risk_budget_from_legs(legs, net_entry_cashflow)
        if risk_budget is None:
            continue

        first = group.iloc[0]
        exit_reasons = (
            group["exit_reason"].astype(str).tolist()
            if "exit_reason" in group.columns
            else []
        )
        scaled_out = sum(
            reason in {"scaleout_drop5", "scaleout_iv40", "scaleout_long_itm"}
            for reason in exit_reasons
        )
        rolled_rows.append(
            {
                "strategy_id": first.get("strategy_id"),
                "entry_date": group["entry_date"].min() if "entry_date" in group.columns else first.get("entry_date"),
                "exit_date": group["exit_date"].max() if "exit_date" in group.columns else first.get("exit_date"),
                "pnl": float(group["pnl"].sum()),
                "risk_budget": risk_budget,
                "net_entry_cashflow": net_entry_cashflow,
                "package_id": package_id,
                "kelly_unit": "pakiet scaleout",
                "scaleout_buckets_closed": scaled_out,
            }
        )

    return pd.DataFrame(rolled_rows)


def prepare_kelly_trades(trades: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if is_scaleout_trades_frame(trades):
        rolled = rollup_scaleout_trades_for_kelly(trades)
        return rolled, "pakiet scaleout"

    prepared = trades.copy()
    prepared["risk_budget"] = prepared.apply(trade_risk_budget, axis=1)
    prepared = prepared[prepared["risk_budget"].notna() & (prepared["risk_budget"] > 0)].copy()
    prepared["kelly_unit"] = "transakcja"
    return prepared, "transakcja"


def empirical_kelly_fraction(returns: pd.Series) -> tuple[float, pd.DataFrame]:
    clean = returns.replace([pd.NA, float("inf"), float("-inf")], pd.NA).dropna().astype(float)
    if clean.empty:
        return 0.0, pd.DataFrame(columns=["fraction", "growth"])
    worst = float(clean.min())
    if worst < 0:
        upper = min(5.0, 0.999 / abs(worst))
    else:
        upper = 5.0
    fractions = [upper * step / 400.0 for step in range(401)]
    growth_rows: list[dict[str, float]] = []
    best_fraction = 0.0
    best_growth = float("-inf")
    for fraction in fractions:
        wealth = 1.0 + fraction * clean
        if (wealth <= 0).any():
            growth = float("-inf")
        else:
            growth = float(pd.Series(wealth).map(math.log).mean())
        growth_rows.append({"fraction": fraction, "growth": growth})
        if growth > best_growth:
            best_growth = growth
            best_fraction = fraction
    return best_fraction, pd.DataFrame(growth_rows)


def render_digital_tail(results_dir: str, summary: pd.DataFrame, trades: pd.DataFrame, equity: pd.DataFrame) -> None:
    try:
        from taleb_kurtosis.digital_tail import figures
    except ModuleNotFoundError as exc:
        if exc.name != "taleb_kurtosis":
            raise
        # The Streamlit CLI puts the script directory, not the repo, on sys.path.
        from digital_tail import figures

    manifest = json.loads((Path(results_dir) / "manifest.json").read_text())
    if manifest.get("kind") == "digital_tail_comparison":
        render_digital_tail_comparison(results_dir, summary, trades, equity, manifest)
        return
    cfg = manifest["config"]
    st.subheader("SPX Digital Tail")
    st.button("Porównanie 2,5σ / 3σ / 3,5σ · 250 tys. USD", icon=":material/compare_arrows:",
              on_click=apply_selected_preset, args=("Digital Tail / Bariery 250k",), key="open_digital_comparison")
    st.caption(f"{manifest['data_start']} – {manifest['data_end']} | Monthly | "
               f"{cfg['target_dte']} DTE | delta −{cfg['tail_delta']} | {cfg['spread_sigma']}σ ATM IV")
    row = summary.iloc[0]
    cols = st.columns(4)
    cols[0].metric("PnL netto (USD)", f"{row['total_pnl']:,.0f}")
    cols[1].metric("Max drawdown (USD)", f"{row['max_drawdown']:,.0f}")
    cols[2].metric("Max drawdown (%)", f"{row['max_drawdown_pct']:.2%}")
    cols[3].metric("Zamknięte pakiety", int(row["trades"]))
    st.caption(f"Kapitał: {cfg['initial_capital']:,.0f} USD | limit/pakiet: {cfg['risk_per_package']:.0%} | "
               f"limit łączny: {cfg['total_risk']:.0%} | założony koszt/kontrakt/stronę: {cfg['fee_per_contract']:.2f} USD")
    if row["unresolved_packages"] or row["stale_mark_sessions"]:
        st.warning(f"Nieświeże wyceny: {int(row['stale_mark_sessions'])} sesji. "
                   f"Pakiety bez potwierdzonego zamknięcia: {int(row['unresolved_packages'])}. "
                   "PnL i drawdown mogą być zniekształcone przez brakujące kwotowania.")
    st.caption("Wejścia i wyjścia po bid/ask; dzienna wycena po mid z rezerwą kosztów wyjścia. "
               "Zamknięcie sesję przed datą expiry z pliku; brak oficjalnych danych settlementu SET i identyfikatorów AM/PM.")
    active_sessions = int(equity["open_trades"].gt(0).sum())
    entry_months = pd.to_datetime(trades["entry_date"]).dt.to_period("M").nunique() if not trades.empty else 0
    months = pd.to_datetime(equity["quote_date"]).dt.to_period("M").nunique()
    st.caption(f"Miesiące z wejściem: {entry_months}/{months} | sesje z otwartą pozycją na koniec dnia: "
               f"{active_sessions}/{len(equity)}. Tylko serie z expiry w trzeci piątek miesiąca.")
    if row["trades"] < 30:
        st.warning("Mała próba transakcji: wynik nie rozstrzyga hipotezy o premii za ogon.")
    if not trades.empty:
        exceeded = trades[trades["status"].eq("closed") & trades["pnl"].lt(-trades["risk_budget"] - 1e-6)]
        if not exceeded.empty:
            st.warning(f"W {len(exceeded)} pakiecie/pakietach strata przy zamknięciu po bid/ask przekroczyła "
                       "budżet straty w expiry. Ten budżet nie jest limitem kosztu wcześniejszej likwidacji.")
    overview, payoff, audit = st.tabs(["Wynik", "Wypłata i pakiety", "Dane i parametry"])
    selected = None
    with payoff:
        if not trades.empty:
            selected = st.selectbox("Pakiet", trades["package_id"].tolist(), key="digital_package")
    plots = figures(trades, equity, selected)
    with overview:
        for name in ["equity", "drawdown", "monthly", "risk"]:
            st.plotly_chart(plots[name], use_container_width=True, key=f"digital_{name}")
        st.caption("Daleki put i spready to składowe tych samych pakietów, nie osobne portfele o równym ryzyku.")
    with payoff:
        if not trades.empty:
            st.plotly_chart(plots["payoff"], use_container_width=True, key="digital_payoff")
            detail = trades[trades["package_id"] == selected]
            st.dataframe(detail[["entry_date", "expire_date", "tail_strike", "short_strike", "hedge_strike",
                                 "spread_count", "tail_delta", "atm_iv", "risk_budget", "lower_breakeven",
                                 "net_credit_after_costs", "pnl", "status"]], use_container_width=True)
            st.caption("Wykres przedstawia teoretyczne PnL w expiry po rezerwie prowizji. "
                       "Historyczny wynik obejmuje wcześniejsze zamknięcie po bid/ask.")
            st.plotly_chart(plots["trades"], use_container_width=True, key="digital_trades")
            st.dataframe(trades, use_container_width=True)
    with audit:
        attempts = pd.read_csv(Path(results_dir) / "entry_audit.csv")
        st.dataframe(attempts["reason"].value_counts().rename_axis("Powód").reset_index(name="Sesje"), use_container_width=True)
        st.dataframe(attempts, use_container_width=True)
        st.json(manifest)


def render_digital_tail_comparison(results_dir: str, summary: pd.DataFrame, trades: pd.DataFrame,
                                   equity: pd.DataFrame, manifest: dict) -> None:
    try:
        from taleb_kurtosis.digital_tail_comparison import comparison_figures, package_figures
    except ModuleNotFoundError as exc:
        if exc.name != "taleb_kurtosis":
            raise
        from digital_tail_comparison import comparison_figures, package_figures

    root = Path(results_dir)
    labels = summary.set_index("strategy_id")["label"].to_dict()
    st.subheader("SPX Digital Tail: porównanie barier")
    st.caption(f"{manifest['data_start']} – {manifest['data_end']} | 120 DTE | delta −0,01 | 2,5 / 3 / 3,5σ ATM IV")
    variant = st.selectbox("Wariant", summary["strategy_id"].tolist(), index=1,
                           format_func=labels.get, key="digital_cmp_variant")
    row = summary[summary["strategy_id"].eq(variant)].iloc[0]
    cfg = manifest["configs"][variant]
    cols = st.columns(4)
    for col, label, value in zip(cols, ["PnL netto (USD)", "Max drawdown", "Zamknięte pakiety", "Bariera / cel zysku"],
                                 [f"{row['total_pnl']:,.0f}", f"{row['max_drawdown_pct']:.2%}", str(int(row["trades"])),
                                  f"{int(row['barrier_exits'])} / {int(row['profit_exits'])}"]):
        col.metric(label, value)
    st.caption(f"Kapitał: {cfg['initial_capital']:,.0f} USD | limit/pakiet: {cfg['risk_per_package']:.0%} | "
               f"limit łączny: {cfg['total_risk']:.0%} | koszt: {cfg['fee_per_contract']:.2f} USD/kontrakt/stronę")
    st.caption("Bariera = strike short puta z wejścia. Sygnał: SPX na koniec dnia ≤ bariera. "
               "Zamknięcie wszystkich nóg po bid/ask tej sesji, również przy stracie pakietu. "
               "TP = 5% lub 10% początkowego budżetu ryzyka, po kosztach całego pakietu. Brak symulacji dotknięć intraday.")
    if row["unresolved_packages"] or row["stale_mark_sessions"]:
        st.warning(f"Wynik prowizoryczny: {int(row['unresolved_packages'])} nierozliczonych pakietów; "
                   f"{int(row['stale_mark_sessions'])} sesji z nieświeżą wyceną.")
    elif row["trades"] == 0:
        st.warning("Brak transakcji dla tego wariantu przy przyjętym kapitale, limitach i warunkach finansowania. "
                   "0 USD nie jest wynikiem aktywnej strategii.")
    elif row["trades"] < 30:
        st.warning("Mała próba transakcji. Różna liczba wejść nie pozwala przypisać całej różnicy PnL samej regule wyjścia.")
    st.caption("Daty miesięcznych serii dopasowane według kalendarza CBOE. Zmiany etykiet są założeniem badawczym: "
               "brak identyfikatorów AM/PM w plikach. Starszy baseline z filtrem piątkowym pozostaje osobnym runem.")
    provisional = summary["unresolved_packages"].gt(0) | summary["stale_mark_sessions"].gt(0)
    if provisional.any():
        st.warning("Część wariantów w tabeli ma wynik prowizoryczny. Nierozliczone pozycje pozostają po ostatniej wycenie "
                   "i blokują nowe wejścia; ich PnL i drawdown nie są pełnym wynikiem strategii.")
    overview, packages, paired_tab, audit_tab, smile_tab, entry_tab = st.tabs(
        ["Porównanie", "Pakiet", "Wspólne wejścia", "Dane", "Uśmiech IV", "Wejścia i utrzymanie"])
    with overview:
        columns = {"spread_sigma": "Sigma", "exit_label": "Wyjście", "trades": "Pakiety", "total_pnl": "PnL USD",
                   "max_drawdown": "DD USD", "max_drawdown_pct": "DD %", "liquidation_max_drawdown": "DD bid/ask USD",
                   "win_rate": "Win rate", "barrier_exits": "Bariera", "profit_exits": "TP",
                   "unresolved_packages": "Nierozliczone"}
        table = summary[list(columns)].rename(columns=columns)
        table.insert(2, "Status", ["Prowizoryczny" if bad else "Brak transakcji" if n == 0 else "Rozliczony"
                                   for bad, n in zip(provisional, summary["trades"])])
        st.dataframe(table.style.format({"Sigma": "{:g}", "PnL USD": "{:,.0f}", "DD USD": "{:,.0f}",
                                        "DD %": "{:.2%}", "Win rate": "{:.1%}", "DD bid/ask USD": "{:,.0f}"}, na_rep="—"),
                     hide_index=True, use_container_width=True, height=458)
        dimension = st.radio("Przekrój wykresów", ["Reguły wyjścia", "Odległość spreadu"], index=1,
                             horizontal=True, key="digital_cmp_dimension")
        if dimension == "Reguły wyjścia":
            sigma = st.selectbox("Odległość spreadu", sorted(summary["spread_sigma"].unique()),
                                 format_func=lambda x: f"{x:g}σ ATM IV", key="digital_cmp_sigma")
            chart_rows = summary[summary["spread_sigma"].eq(sigma)]
        else:
            policies = summary["exit_label"].drop_duplicates().tolist()
            policy = st.selectbox("Reguła wyjścia", policies, index=policies.index("Bariera") if "Bariera" in policies else 0,
                                  key="digital_cmp_policy")
            chart_rows = summary[summary["exit_label"].eq(policy)]
        plots = comparison_figures(chart_rows, equity)
        for name in ["equity_pnl", "drawdown"]:
            st.plotly_chart(plots[name], use_container_width=True, key=f"digital_cmp_{name}")
        with st.expander("Wykorzystanie budżetu ryzyka"):
            st.plotly_chart(plots["reserved_risk"], use_container_width=True, key="digital_cmp_risk")
        st.caption("DD główne: dzienna wycena mid z rezerwą kosztów; DD bid/ask: koszt likwidacji całego portfela. "
                   "Budżet straty w expiry nie jest limitem straty przy wcześniejszym zamknięciu.")
    with packages:
        selected_trades = trades[trades["strategy_id"].eq(variant)]
        if selected_trades.empty:
            st.info("Brak otwartych pakietów dla tego wariantu.")
        else:
            names = {r["package_id"]: f"{r['entry_date']:%Y-%m-%d} | {r['spread_count']} spreadów | {r['pnl']:,.0f} USD"
                     for _, r in selected_trades.iterrows()}
            package = st.selectbox("Pakiet", selected_trades["package_id"].tolist(), format_func=names.get,
                                   key=f"digital_cmp_package_{variant}")
            trade = selected_trades[selected_trades["package_id"].eq(package)].iloc[0]
            paths = pd.read_parquet(root / "package_marks.parquet")
            paths = paths[paths["package_id"].eq(package)]
            exit_names = {"barrier_close_all": "Bariera", "package_profit_target": "Cel zysku pakietu",
                          "pre_expiry_bid_ask": "Przed expiry", "unresolved_missing_quotes": "Brak rozliczenia"}
            st.caption(f"Wyjście: {exit_names.get(trade['exit_reason'], trade['exit_reason'])} | "
                       f"long tail: {trade['tail_pnl']:,.0f} USD | spready: {trade['spread_pnl']:,.0f} USD | "
                       f"budżet początkowy: {trade['risk_budget']:,.0f} USD")
            for name, fig in package_figures(trade, paths).items():
                st.plotly_chart(fig, use_container_width=True, key=f"digital_cmp_package_{name}")
            st.dataframe(selected_trades[["entry_date", "exit_date", "exit_reason", "tail_strike", "short_strike",
                                          "hedge_strike", "spread_count", "risk_budget", "pnl", "source_label_switches",
                                          "missing_mark_days", "status"]], hide_index=True, use_container_width=True)
    with paired_tab:
        st.caption("Porównanie z wyjściem przed expiry przy identycznych wejściach, strike’ach i liczbie spreadów, osobno dla każdej sigmy. "
                   "To analiza pakietów, nie osobny portfel ani porównanie sigm na tych samych datach.")
        if provisional.any():
            st.warning("Tylko pakiety zamknięte po obu stronach porównania. Wykluczenie nierozliczonych pozycji może "
                       "usunąć ważne zdarzenia ogonowe i zniekształcić ocenę reguły wyjścia.")
        try:
            pairs = pd.read_csv(root / "paired_exits.csv")
        except pd.errors.EmptyDataError:
            pairs = pd.DataFrame()
        if pairs.empty:
            st.info("Brak wspólnych zamkniętych pakietów.")
        else:
            grouped = pairs.groupby(["spread_sigma", "exit_label"], sort=False).agg(
                pakiety=("package_id", "size"), pnl=("pnl", "sum"), kontrola=("control_pnl", "sum"),
                roznica=("pnl_difference", "sum")).reset_index()
            st.dataframe(grouped.rename(columns={"spread_sigma": "Sigma", "exit_label": "Wyjście", "pakiety": "Wspólne pakiety",
                                                 "pnl": "PnL wariantu", "kontrola": "PnL przed expiry", "roznica": "Różnica USD"}),
                         hide_index=True, use_container_width=True)
            st.dataframe(pairs[pairs["strategy_id"].eq(variant)], hide_index=True, use_container_width=True)
    with audit_tab:
        attempts = pd.read_csv(root / "entry_audit.csv")
        attempts = attempts[attempts["strategy_id"].eq(variant)]
        if "risk_budget" in attempts and attempts["risk_budget"].notna().any():
            st.caption(f"Najmniejszy budżet finansowalnego kandydata: {attempts['risk_budget'].min():,.0f} USD; "
                       f"początkowy limit/pakiet: {cfg['initial_capital'] * cfg['risk_per_package']:,.0f} USD.")
        st.dataframe(attempts["reason"].value_counts().rename_axis("Powód").reset_index(name="Sesje"),
                     hide_index=True, use_container_width=True)
        st.caption(f"Zmiany etykiety w otwartych pakietach: {int(row['source_label_switches'])}. "
                   f"Opóźnione zamknięcia: {int(row['delayed_exits'])}. "
                   f"Daty z równoczesnymi etykietami serii (cały zbiór): {manifest['simultaneous_label_days']}.")
        with st.expander("Próby wejścia"):
            st.dataframe(attempts, hide_index=True, use_container_width=True)
        with st.expander("Dopasowania dat serii"):
            st.dataframe(pd.read_csv(root / "expiry_mapping.csv"), hide_index=True, use_container_width=True)
        with st.expander("Założenia i parametry"):
            st.json(manifest)
    with smile_tab:
        try:
            from taleb_kurtosis.digital_tail_smile import OUTPUT, smile_figure, matched_zones
        except ModuleNotFoundError as exc:
            if exc.name != "taleb_kurtosis":
                raise
            from digital_tail_smile import OUTPUT, smile_figure, matched_zones
        if not (OUTPUT / "manifest.json").exists():
            st.info("Brak zapisanej próby IV.")
        else:
            sample_meta = json.loads((OUTPUT / "manifest.json").read_text())
            st.caption(f"10 losowych kwalifikujących się sesji z {sample_meta['population']} dni; seed {sample_meta['seed']}. "
                       "Najbliższa miesięczna seria do 120 DTE (105–135). Bez selekcji po wyniku transakcji.")
            st.caption("OTM puty po lewej, OTM calle po prawej; IV dostawcy. Sigma z ATM puta i faktycznego DTE. "
                       "Średnia równo waży dni, bez ekstrapolacji i interpolacji luk >0,5σ. "
                       "Pokrycie zmienia się po osi; 10 dni nie dowodzi przewagi konkretnej bariery.")
            curves = pd.read_parquet(OUTPUT / "curves.parquet")
            normalized = st.radio("Miara IV", ["IV / ATM IV", "IV (%)"], horizontal=True) == "IV / ATM IV"
            st.plotly_chart(smile_figure(curves, normalized), use_container_width=True, key="digital_smile")
            mean = pd.read_csv(OUTPUT / "mean.csv")
            st.dataframe(mean[mean.sigma.isin([-3.5, -3, -2.5])].rename(
                columns={"sigma": "Pozycja σ", "iv": "Średnia IV", "relative_iv": "Średnia IV/ATM", "days": "Liczba dni"}),
                hide_index=True, use_container_width=True)
            st.caption("Porównanie trzech barier wyłącznie na wspólnych dniach z pełnym pokryciem:")
            st.dataframe(matched_zones(curves), hide_index=True, use_container_width=True)
            st.dataframe(pd.read_csv(OUTPUT / "samples.csv"), hide_index=True, use_container_width=True)
            with st.expander("Finansowanie na tych samych 10 sesjach"):
                st.caption("Kandydaci przed limitami kapitału. Premia wykonywalna = bid short − ask hedge. "
                           "Podwyższona IV obu nóg nie gwarantuje korzystnej premii spreadu.")
                st.dataframe(pd.read_csv(OUTPUT / "funding.csv"), hide_index=True, use_container_width=True)
    with entry_tab:
        st.subheader("Niezależne wejścia miesięczne; brak rolowania")
        st.caption("Pierwsza sesja miesiąca spełniająca warunki; odrzucone wejście jest ponawiane w kolejnych sesjach. "
                   "Maksymalnie jeden nowy pakiet miesięcznie: 1 put delta około −0,01 oraz minimalna liczba spreadów "
                   "finansująca zakup i prowizje. Termin najbliższy 120 DTE w oknie 105–135.")
        st.caption("Poprzednie pakiety mogą pozostawać otwarte. Bariera i strike’i nie zmieniają się po wejściu. "
                   "Wyjście zamyka wszystkie nogi; nie uruchamia automatycznego odkupienia ani nowego wejścia w tym samym miesiącu. "
                   "Bez sygnału wcześniejszego wyjścia: ostatnia sesja przed modelowaną datą rozliczenia, po bid/ask.")
        selected = trades[trades.strategy_id.eq(variant)].copy()
        if not selected.empty:
            selected["holding_days"] = (pd.to_datetime(selected.exit_date) - pd.to_datetime(selected.entry_date)).dt.days
            st.dataframe(selected[["entry_date", "exit_date", "dte_entry", "holding_days", "exit_reason", "spread_count", "pnl"]],
                         hide_index=True, use_container_width=True)
            closed = selected[selected.status.eq("closed")]
            if len(closed):
                best = closed.pnl.max()
                st.caption(f"Najlepszy zamknięty pakiet: {best:,.0f} USD; pozostałe razem: {closed.pnl.sum() - best:,.0f} USD. "
                           "To diagnostyka koncentracji, nie wynik alternatywnego backtestu bez tego zdarzenia.")


def main() -> None:
    st.set_page_config(page_title="Dashboard Strategii Taleba", layout="wide")
    st.title("Dashboard Strategii Taleba")

    view = st.query_params.get("view")
    if "workspace_view" not in st.session_state:
        st.session_state["workspace_view"] = "ES / MES Live" if view == "power-law-live" else "Strategie"
    workspace = st.sidebar.radio("Widok", ["Strategie", "ES / MES Live"], key="workspace_view")
    if workspace == "ES / MES Live":
        from taleb_kurtosis.power_law_dashboard import render_live
        render_live()
        return
    linked_presets = {"digital-tail-250k": "Digital Tail / Bariery 250k",
                      "digital-tail-100k": "Digital Tail / Porównanie barier",
                      "power-law": "Power Law / Long Tail",
                      "power-law-scaleout": "Power Law / Scaleout"}
    if view in linked_presets and st.session_state.get("applied_view") != view:
        apply_selected_preset(linked_presets[view])
        st.session_state["applied_view"] = view

    result_options = available_results_dirs()
    default_dir = str(DEFAULT_RESULTS_DIR)
    if "selected_results_dir" not in st.session_state:
        st.session_state["selected_results_dir"] = default_dir
    selected_index = (
        result_options.index(st.session_state["selected_results_dir"])
        if st.session_state["selected_results_dir"] in result_options
        else (result_options.index(default_dir) if default_dir in result_options else 0)
    )

    st.sidebar.subheader("Szybki wybór strategii")
    st.sidebar.selectbox(
        "Preset",
        list(PRESETS.keys()),
        key="preset_name",
        on_change=apply_selected_preset,
    )
    if st.session_state.get("preset_name") == "Weekly 30 DTE":
        st.sidebar.success("Preset ustawia run i filtry dla strategii weekly z wyjściem przy 30 DTE.")

    st.sidebar.subheader("Wybór wyników")
    selected_dir = st.sidebar.selectbox("Run wyników", result_options, index=selected_index, key="selected_results_dir")
    use_custom_dir = st.sidebar.checkbox("Użyj własnej ścieżki", value=False, key="use_custom_dir")
    results_dir = (
        st.sidebar.text_input("Własny katalog wyników", selected_dir)
        if use_custom_dir
        else selected_dir
    )
    revision = max((p.stat().st_mtime_ns for name in ["summary.csv", "trades.parquet", "daily_equity.parquet"]
                    if (p := Path(results_dir) / name).exists()), default=0)
    summary, trades, equity = load_results(results_dir, revision)

    if summary.empty:
        st.warning("Nie znaleziono pliku summary.csv w wybranym katalogu wyników.")
        return

    if "construction" in summary and summary["construction"].eq("power_law_scaleout").all():
        from taleb_kurtosis.power_law_dashboard import render_scaleout
        render_scaleout(results_dir, summary, trades, equity)
        return

    if "construction" in summary and summary["construction"].eq("power_law_tail").all():
        from taleb_kurtosis.power_law_dashboard import render_backtest
        render_backtest(results_dir, summary, trades, equity)
        return

    if "construction" in summary and summary["construction"].eq("digital_tail").all():
        render_digital_tail(results_dir, summary, trades, equity)
        return

    available_exits = sorted(summary["exit_policy"].dropna().unique().tolist()) if "exit_policy" in summary.columns else []
    active_run_col, exit_run_col = st.columns([2, 1])
    with active_run_col:
        st.info(
            f"**Aktywny run:** `{results_dir}`\n\n"
            f"**Dostępne wyjścia:** {', '.join(opis_wyjscia(value) for value in available_exits) if available_exits else 'brak'}"
        )
    with exit_run_col:
        if "exit_at_30_dte" not in available_exits:
            only_exits = ", ".join(opis_wyjscia(value) for value in available_exits) if available_exits else "brak"
            st.warning(f"Ten run zawiera tylko: {only_exits}")
        else:
            st.success("Ten run zawiera strategię z wyjściem przy 30 DTE.")

    filters = {
        "long_sigma": multiselect_existing("Long sigma", summary, "long_sigma", "filter_long_sigma"),
        "short_sigma": multiselect_existing("Short sigma", summary, "short_sigma", "filter_short_sigma"),
        "sigma_source": multiselect_existing("Źródło sigma", summary, "sigma_source", "filter_sigma_source"),
        "ratio_policy": multiselect_existing("Polityka ratio", summary, "ratio_policy", "filter_ratio_policy"),
        "ratio_target": multiselect_existing("Docelowe ratio", summary, "ratio_target", "filter_ratio_target"),
        "sigma_window": multiselect_existing("Okno sigma", summary, "sigma_window", "filter_sigma_window"),
        "entry_frequency": multiselect_existing("Częstotliwość wejścia", summary, "entry_frequency", "filter_entry_frequency"),
        "exit_policy": multiselect_existing("Reguła wyjścia", summary, "exit_policy", "filter_exit_policy"),
    }
    filtered_summary = apply_filters(summary, filters)
    strategy_ids = filtered_summary["strategy_id"].dropna().unique().tolist() if "strategy_id" in filtered_summary else []

    filtered_trades = trades[trades["strategy_id"].isin(strategy_ids)] if strategy_ids and "strategy_id" in trades else trades.iloc[0:0]
    filtered_equity = equity[equity["strategy_id"].isin(strategy_ids)] if strategy_ids and "strategy_id" in equity else equity.iloc[0:0]

    if st.session_state.get("preset_name") == "Weekly 30 DTE":
        st.caption(
            "Aktywny preset: **Weekly 30 DTE** — "
            "run `results/taleb_tail_ratio_weekly_4x20_itm_all_exits`, "
            "long 4σ, short 0.4σ, ratio 20, wejście weekly, wyjście `exit_at_30_dte`."
        )

    overview_tab, kelly_tab = st.tabs(["Przegląd", "Kelly"])

    with overview_tab:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Strategie", len(filtered_summary))
        metric_cols[1].metric("Transakcje", int(filtered_summary["trades"].sum()) if "trades" in filtered_summary else 0)
        metric_cols[2].metric("Total PnL", f"{filtered_summary['total_pnl'].sum():,.0f}" if "total_pnl" in filtered_summary else "0")
        best = filtered_summary.sort_values("total_pnl", ascending=False).head(1)
        metric_cols[3].metric("Najlepszy PnL", f"{best['total_pnl'].iloc[0]:,.0f}" if not best.empty else "0")

        st.subheader("Ranking")
        ranking_columns = [
            "opis_strategii",
            "strategy_id",
            "total_pnl",
            "daily_pnl_sharpe",
            "max_drawdown",
            "win_rate",
            "entry_frequency",
            "short_sigma",
            "sigma_source",
            "avg_long_short_ratio",
            "avg_long_put_moneyness",
            "avg_short_put_moneyness",
            "avg_open_delta",
            "avg_open_gamma",
            "avg_open_vega",
            "avg_open_theta",
            "avg_open_rho",
            "scaleout_trigger_drop5_count",
            "scaleout_trigger_iv40_count",
            "scaleout_trigger_itm_count",
            "scaleout_short_core_close_count",
            "avg_scaleout_buckets_closed_pre_expiry",
            "entry_iv_limit_rejections",
            "trades",
        ]
        visible_columns = [column for column in ranking_columns if column in filtered_summary.columns]
        st.dataframe(filtered_summary.sort_values("total_pnl", ascending=False)[visible_columns], use_container_width=True)

        chart_col, heat_col = st.columns([2, 1])
        with chart_col:
            st.subheader("Krzywa Equity")
            if not filtered_equity.empty:
                fig = px.line(
                    filtered_equity.sort_values("quote_date"),
                    x="quote_date",
                    y="equity_pnl",
                    color="strategy_id",
                    hover_data=["open_trades", "realized_pnl", "unrealized_pnl"],
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Brak wierszy equity dla aktualnych filtrów.")

        with heat_col:
            st.subheader("Sigma x Ratio")
            if {"long_sigma", "ratio_policy", "ratio_target", "total_pnl"}.issubset(filtered_summary.columns):
                heat = filtered_summary.copy()
                heat["ratio_label"] = heat["ratio_policy"].astype(str) + ":" + heat["ratio_target"].fillna("max").astype(str)
                heat = heat.groupby(["long_sigma", "ratio_label"], as_index=False)["total_pnl"].sum()
                fig = px.density_heatmap(heat, x="ratio_label", y="long_sigma", z="total_pnl", histfunc="sum")
                st.plotly_chart(fig, use_container_width=True)

        dd = drawdown_frame(filtered_equity)
        if not dd.empty:
            st.subheader("Drawdown")
            fig = px.line(dd, x="quote_date", y="drawdown", color="strategy_id")
            st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.subheader("Rozkład PnL")
            if not filtered_trades.empty and "pnl" in filtered_trades:
                fig = px.histogram(filtered_trades, x="pnl", color="strategy_id", nbins=80)
                st.plotly_chart(fig, use_container_width=True)
        with right:
            st.subheader("Najlepsze / Najgorsze transakcje")
            if not filtered_trades.empty:
                detail_columns = [
                    "strategy_id",
                    "entry_date",
                    "exit_date",
                    "pnl",
                    "net_entry_cashflow",
                    "actual_ratio",
                    "long_put_moneyness",
                    "short_put_moneyness",
                    "package_id",
                    "component_type",
                    "bucket_trigger",
                    "long_put_iv_entry",
                    "long_put_iv_exit",
                    "trigger_date",
                    "legs_json",
                ]
                detail_columns = [column for column in detail_columns if column in filtered_trades.columns]
                worst = filtered_trades.sort_values("pnl").head(10)[detail_columns]
                best_trades = filtered_trades.sort_values("pnl", ascending=False).head(10)[detail_columns]
                st.caption("Najgorsze")
                st.dataframe(worst, use_container_width=True)
                st.caption("Najlepsze")
                st.dataframe(best_trades, use_container_width=True)

    with kelly_tab:
        st.subheader("Dobór kapitału metodą Kelly'ego")
        if filtered_summary.empty or filtered_trades.empty:
            st.info("Wybierz co najmniej jedną strategię z transakcjami, aby policzyć Kelly'ego.")
        else:
            strategy_options = filtered_summary.sort_values("total_pnl", ascending=False)["strategy_id"].dropna().tolist()
            selected_strategy = st.selectbox("Strategia źródłowa do Kelly'ego", strategy_options, index=0)
            bankroll = st.number_input("Kapitał", min_value=1000.0, value=100000.0, step=1000.0)
            use_half_kelly = st.checkbox("Pokaż Half-Kelly jako główną rekomendację", value=True)

            raw_strategy_trades = filtered_trades[filtered_trades["strategy_id"] == selected_strategy].copy()
            strategy_equity = filtered_equity[filtered_equity["strategy_id"] == selected_strategy].copy()
            strategy_trades, kelly_unit = prepare_kelly_trades(raw_strategy_trades)

            if strategy_trades.empty:
                st.warning("Nie udało się wyznaczyć dodatniego budżetu ryzyka dla wybranej strategii.")
            else:
                strategy_trades["risk_return"] = strategy_trades["pnl"] / strategy_trades["risk_budget"]
                kelly_fraction, growth_curve = empirical_kelly_fraction(strategy_trades["risk_return"])
                half_kelly = kelly_fraction / 2.0
                quarter_kelly = kelly_fraction / 4.0
                max_open_raw = (
                    int(strategy_equity["open_trades"].max())
                    if "open_trades" in strategy_equity and not strategy_equity.empty
                    else 1
                )
                if kelly_unit == "pakiet scaleout":
                    max_open = math.ceil(max_open_raw / scaleout_component_count(raw_strategy_trades))
                else:
                    max_open = max_open_raw
                overlap_adjusted = half_kelly / max(max_open, 1)
                primary_fraction = half_kelly if use_half_kelly else kelly_fraction

                metrics = st.columns(5)
                metrics[0].metric("Frakcja Kelly'ego", f"{kelly_fraction:.2%}")
                metrics[1].metric("Połowa Kelly'ego", f"{half_kelly:.2%}")
                metrics[2].metric("Ćwiartka Kelly'ego", f"{quarter_kelly:.2%}")
                metrics[3].metric("Maks. liczba nakładających się pozycji", max_open)
                metrics[4].metric("Skorygowany Half-Kelly", f"{overlap_adjusted:.2%}")

                dollars = bankroll * primary_fraction
                conservative_dollars = bankroll * overlap_adjusted
                median_risk = float(strategy_trades["risk_budget"].median())
                mean_risk = float(strategy_trades["risk_budget"].mean())
                selected_row = filtered_summary[filtered_summary["strategy_id"] == selected_strategy].head(1)
                selected_exit = opis_wyjscia(selected_row["exit_policy"].iloc[0]) if not selected_row.empty else "nieznane"
                st.write(
                    f"Kelly liczony jest dla runu `{results_dir}` i strategii z wyjściem **{selected_exit}**. "
                    f"Jednostka próbki: **{kelly_unit}**."
                )
                st.write(
                    f"Na podstawie empirycznego rozkładu zwrotów na jednostkę ryzyka "
                    f"surowa frakcja Kelly'ego wynosi `{kelly_fraction:.2%}`. "
                    f"Dla kapitału `{bankroll:,.0f}` oznacza to około `{dollars:,.0f}` "
                    f"sugerowanego kapitału ryzyka na nową pozycję lub `{conservative_dollars:,.0f}` "
                    f"w wersji konserwatywnej po uwzględnieniu nakładających się pozycji."
                )

                approx_contracts = math.floor(conservative_dollars / median_risk) if median_risk > 0 else 0
                st.write(
                    f"Mediana budżetu ryzyka na próbkę w tej serii to `{median_risk:,.0f}`, "
                    f"a średnia to `{mean_risk:,.0f}`. "
                    f"Przy skorygowanym Half-Kelly daje to około `{approx_contracts}` pełnych pozycji "
                    f"przy medianowym ryzyku."
                )

                chart_col, dist_col = st.columns(2)
                with chart_col:
                    st.subheader("Krzywa wzrostu Kelly'ego")
                    if not growth_curve.empty:
                        fig = px.line(growth_curve, x="fraction", y="growth")
                        st.plotly_chart(fig, use_container_width=True)
                with dist_col:
                    st.subheader("Zwroty z transakcji na jednostkę ryzyka")
                    fig = px.histogram(strategy_trades, x="risk_return", nbins=60)
                    st.plotly_chart(fig, use_container_width=True)

                detail_columns = [
                    "entry_date",
                    "exit_date",
                    "pnl",
                    "risk_budget",
                    "risk_return",
                    "net_entry_cashflow",
                    "package_id",
                    "scaleout_buckets_closed",
                ]
                detail_columns = [column for column in detail_columns if column in strategy_trades.columns]
                detail = strategy_trades[detail_columns].copy()
                st.subheader("Próbka do Kelly'ego")
                st.dataframe(detail.sort_values("risk_return"), use_container_width=True)


if __name__ == "__main__":
    main()
