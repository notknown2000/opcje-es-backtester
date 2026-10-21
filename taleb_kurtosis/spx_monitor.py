"""Read-only SPX delayed feed and same-chain Taleb relative valuation."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from taleb_kurtosis.power_law import PricingConfig, prepare_chain, price_vector

SOURCE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"
MAX_BYTES = 40 * 1024 * 1024
REFRESH_SECONDS = 300


class FeedError(ValueError):
    pass


@dataclass
class Snapshot:
    puts: pd.DataFrame
    spot: float
    generated_at: pd.Timestamp
    underlying_at: pd.Timestamp
    fetched_at: pd.Timestamp
    raw_contracts: int
    rejected_symbols: int


def timestamp(value, naive_zone):
    if value is None or value == "":
        raise FeedError("Brak czasu danych w odpowiedzi Cboe.")
    try:
        result = pd.Timestamp(value)
        if pd.isna(result):
            raise ValueError("NaT")
        if result.tzinfo is None:
            result = result.tz_localize(naive_zone, ambiguous="raise", nonexistent="raise")
        return result.tz_convert("UTC")
    except (ValueError, TypeError) as exc:
        raise FeedError("Niepoprawny czas danych Cboe.") from exc


def parse_snapshot(payload, fetched_at=None):
    if not isinstance(payload, dict) or payload.get("symbol") not in {"_SPX", "SPX", "^SPX"}:
        raise FeedError("Odpowiedz nie jest lancuchem SPX.")
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("symbol") not in {"_SPX", "SPX", "^SPX"}:
        raise FeedError("Niepoprawny instrument bazowy.")
    try:
        spot = float(data["current_price"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FeedError("Brak poziomu SPX.") from exc
    if not np.isfinite(spot) or spot <= 0:
        raise FeedError("Niepoprawny poziom SPX.")
    options = data.get("options")
    if not isinstance(options, list) or not options:
        raise FeedError("Cboe nie zwrocilo kontraktow opcyjnych.")
    q = pd.DataFrame(options)
    required = {"option", "bid", "ask", "iv", "delta"}
    if not required.issubset(q.columns):
        raise FeedError("Zmieniony format API: brak pol kontraktu, bid/ask lub IV/delta.")
    # OCC identifiers keep AM SPX and PM SPXW series separate, even on the same date.
    parts = q.option.astype(str).str.extract(
        r"^(?P<series>SPXW?)(?P<date>\d{6})(?P<side>[CP])(?P<strike>\d{8})$"
    )
    parts["expiry"] = pd.to_datetime("20" + parts.date, format="%Y%m%d", errors="coerce")
    valid = parts.expiry.notna() & parts.series.notna()
    puts = q.loc[valid & parts.side.eq("P")].copy()
    if puts.empty:
        raise FeedError("Brak poprawnych putow SPX/SPXW.")
    puts["series"] = parts.loc[puts.index, "series"]
    puts["expiry"] = parts.loc[puts.index, "expiry"]
    puts["STRIKE"] = pd.to_numeric(parts.loc[puts.index, "strike"]) / 1000
    for source, target in [("bid", "P_BID"), ("ask", "P_ASK"), ("iv", "P_IV"), ("delta", "P_DELTA")]:
        puts[target] = pd.to_numeric(puts[source], errors="coerce")
    for column in ["volume", "open_interest", "bid_size", "ask_size"]:
        puts[column] = pd.to_numeric(puts.get(column, np.nan), errors="coerce")
    # Snapshot envelope uses UTC; underlying last_trade_time uses New York local time.
    # Neither timestamp certifies freshness of the individual option bid/ask.
    return Snapshot(
        puts.reset_index(drop=True), spot, timestamp(payload.get("timestamp"), "UTC"),
        timestamp(data.get("last_trade_time"), "America/New_York"),
        timestamp(fetched_at if fetched_at is not None else pd.Timestamp.now(tz="UTC"), "UTC"),
        len(options), int((~valid).sum()),
    )


def fetch_snapshot():
    request = Request(SOURCE_URL, headers={"Accept": "application/json", "User-Agent": "SPXTailMonitor/1.0"})
    try:
        with urlopen(request, timeout=30) as response:
            body = response.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES:
            raise FeedError("Odpowiedz Cboe przekracza limit 40 MB.")
        return parse_snapshot(json.loads(body))
    except HTTPError as exc:
        raise FeedError(f"Cboe HTTP {exc.code}. Feed niedostepny; brak nowych notowan.") from None
    except (URLError, TimeoutError, OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FeedError("Nie udalo sie pobrac poprawnego snapshotu Cboe. Sprawdz siec lub sprobuj pozniej.") from exc


def freshness(snapshot, now=None):
    now = timestamp(now if now is not None else pd.Timestamp.now(tz="UTC"), "UTC")
    generation_age = (now - snapshot.generated_at).total_seconds()
    spot_age = (now - snapshot.underlying_at).total_seconds()
    if min(generation_age, spot_age) < -60 or snapshot.underlying_at > snapshot.generated_at + pd.Timedelta(minutes=1):
        return False, "Niespojny czas danych; ranking zablokowany."
    if generation_age > 1800 or spot_age > 3600:
        return False, "Nieaktualny snapshot / rynek zamkniety. Ranking biezacy zablokowany."
    return True, "Dane opoznione ok. 15 min; wiek poszczegolnych bid/ask nieznany."


def evaluate_snapshot(snapshot, dte_min=165, dte_max=195, alpha=3., fee=1.,
                      max_spread=.5, basis="mid"):
    if basis not in {"mid", "bid_ask"}:
        raise ValueError("Unknown price basis")
    if not 1 <= dte_min <= dte_max <= 1095:
        raise ValueError("Invalid DTE range")
    cfg = PricingConfig(alpha=alpha, anchor_sigma=3., min_edge=0., fee=fee, max_relative_spread=max_spread)
    when = snapshot.underlying_at.tz_convert("America/New_York").tz_localize(None).normalize()
    source = snapshot.puts.assign(dte=(snapshot.puts.expiry - when).dt.days)
    source = source[source.dte.between(dte_min, dte_max)]
    ranked, audits, smiles = [], [], []
    for (series, expiry), chain in source.groupby(["series", "expiry"], sort=True):
        dte = int(chain.dte.iloc[0])
        q, reason = prepare_chain(chain, snapshot.spot, dte, cfg)
        audit = dict(series=series, expiry=expiry, dte=dte, raw_quotes=len(chain), eligible_quotes=0,
                     ranked_quotes=0, reason=reason, atm_iv=np.nan, anchor_strike=np.nan,
                     anchor_sigma=np.nan, anchor_mid=np.nan,
                     duplicate_rows=int(chain.STRIKE.duplicated(keep=False).sum()))
        if q.empty:
            audits.append(audit)
            continue
        smiles.append(q.assign(series=series, expiry=expiry))
        audit.update(atm_iv=float(q.atm_iv.iloc[0]), eligible_quotes=int(q.eligible_quote.sum()))
        prices, anchor_index, reason = price_vector(q, snapshot.spot, cfg)
        audit["reason"] = reason
        if reason != "ok":
            audits.append(audit)
            continue
        anchor = q.iloc[anchor_index]
        anchor_mid = (anchor.P_BID + anchor.P_ASK) / 2
        audit.update(anchor_strike=float(anchor.STRIKE), anchor_sigma=float(anchor.sigma), anchor_mid=anchor_mid)
        keep = np.isfinite(prices) & (prices > 0) & q.sigma.ge(3).to_numpy()
        result = q.loc[keep].copy()
        model_mid = prices[keep] * anchor_mid / anchor.P_BID
        result["mid"] = (result.P_BID + result.P_ASK) / 2
        cost = (result.mid if basis == "mid" else result.P_ASK) + 2 * fee / cfg.multiplier
        model = model_mid if basis == "mid" else prices[keep]
        result = result.assign(
            series=series, expiry=expiry, model_price=model, mid_model=model_mid,
            discount=1 - cost / model, gross_mid_discount=1 - result.mid / model_mid,
            bid_ask_discount=1 - (result.P_ASK + 2 * fee / cfg.multiplier) / prices[keep],
            cost_usd=cost * cfg.multiplier, anchor_strike=anchor.STRIKE,
            anchor_sigma=anchor.sigma, anchor_mid=anchor_mid, alpha=alpha, basis=basis,
            source_generated_at=snapshot.generated_at, underlying_at=snapshot.underlying_at,
            fetched_at=snapshot.fetched_at, data_type="delayed_indicative", source_url=SOURCE_URL,
            fee_per_side=fee, multiplier=cfg.multiplier,
        )
        # Sensitivities reuse the same raw chain and quote/slope screens as the heatmap.
        for exponent in [2.5, 3., 3.5]:
            alternative, index, why = price_vector(q, snapshot.spot, replace(cfg, alpha=exponent))
            column = f"discount_alpha_{exponent:g}"
            result[column] = np.nan
            if why == "ok":
                alt_anchor = q.iloc[index]
                alt_model = alternative[keep]
                if basis == "mid":
                    alt_model = alt_model * (alt_anchor.P_BID + alt_anchor.P_ASK) / (2 * alt_anchor.P_BID)
                result[column] = 1 - cost / alt_model
        sensitivity = result[[f"discount_alpha_{a:g}" for a in [2.5, 3., 3.5]]]
        result["robust_underpriced"] = sensitivity.notna().all(axis=1) & sensitivity.gt(0).all(axis=1)
        result["valuation"] = np.where(result.discount > 0, "Ponizej modelu", "Powyzej modelu")
        result.loc[result.discount.abs() < 1e-10, "valuation"] = "Przy modelu"
        audit.update(ranked_quotes=len(result), reason="ok" if len(result) else "no_quotes_beyond_3sigma")
        audits.append(audit)
        if not result.empty:
            ranked.append(result)
    concat = lambda frames: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result = concat(ranked)
    if not result.empty:
        result = result.sort_values("discount", ascending=False).reset_index(drop=True)
    return result, pd.DataFrame(audits), concat(smiles)
