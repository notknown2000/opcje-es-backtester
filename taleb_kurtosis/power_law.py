"""Relative put pricing from the bounded downside density in arXiv:1908.02347 III."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PricingConfig:
    alpha: float = 2.75
    anchor_sigma: float = 2.5
    anchor_tolerance: float = .25
    max_relative_spread: float = .5
    min_edge: float = .10
    fee: float = 1.0
    multiplier: float = 100.0

    def __post_init__(self):
        if not all(np.isfinite(v) for v in vars(self).values()):
            raise ValueError("Parameters must be finite")
        if self.alpha <= 1 or self.anchor_sigma <= 0 or self.anchor_tolerance <= 0:
            raise ValueError("alpha > 1 and positive anchor parameters required")
        if not 0 < self.max_relative_spread <= 2 or self.min_edge < 0 or self.fee < 0 or self.multiplier <= 0:
            raise ValueError("Invalid quote/cost limits")


def put_shape(strikes, reference: float, alpha: float):
    """Integral of (r-x)*r**(-alpha-1), x=1-K/S0, r in [x,1], up to a constant.

    Integrating the paper's density on S>=0 avoids the negative fractional
    powers/dimensional ambiguity in the printed equation (7).
    """
    strikes = np.asarray(strikes, dtype=float)
    if not np.isfinite(reference) or reference <= 0 or not np.isfinite(alpha) or alpha <= 1:
        raise ValueError("Positive reference and alpha > 1 required")
    if np.any(~np.isfinite(strikes) | (strikes < 0) | (strikes >= reference)):
        raise ValueError("Put strikes must satisfy 0 <= K < reference")
    k = strikes / reference
    result = np.expm1((1 - alpha) * np.log1p(-k)) - (alpha - 1) * k
    # Taylor expansion preserves precision for strikes very close to zero.
    small = k < 1e-5
    return np.where(small, alpha * (alpha - 1) / 2 * k**2 *
                    (1 + (alpha + 1) * k / 3 + (alpha + 1) * (alpha + 2) * k**2 / 12), result)


def relative_put_price(strikes, reference: float, anchor_strike: float, anchor_price: float, alpha: float):
    if not np.isfinite(anchor_price) or anchor_price <= 0 or not 0 < anchor_strike < reference:
        raise ValueError("Invalid anchor")
    strikes = np.asarray(strikes, dtype=float)
    if np.any(strikes > anchor_strike):
        raise ValueError("Only extrapolation deeper into the put tail is allowed")
    return anchor_price * put_shape(strikes, reference, alpha) / put_shape(anchor_strike, reference, alpha)


def anchor_slope(reference, strike, price, alpha):
    k = strike / reference
    return price * (alpha - 1) * np.expm1(-alpha * np.log1p(-k)) / (reference * put_shape(strike, reference, alpha))


def prepare_chain(chain: pd.DataFrame, reference: float, dte: int, cfg=PricingConfig(), atm_iv=None):
    """Normalize quotes once; the alpha/anchor sensitivity shares this frame."""
    required = ["STRIKE", "P_BID", "P_ASK", "P_IV", "P_DELTA"]
    q = chain.copy()
    for c in required:
        q[c] = pd.to_numeric(q[c], errors="coerce")
    q = q[~q.STRIKE.duplicated(keep=False)].sort_values("STRIKE").reset_index(drop=True)
    if q.empty or not np.isfinite(reference) or reference <= 0 or dte <= 0:
        return pd.DataFrame(), "invalid_chain"
    mid = (q.P_BID + q.P_ASK) / 2
    valid = (np.isfinite(q[required]).all(axis=1) & q.P_BID.gt(0) & q.P_ASK.ge(q.P_BID)
             & q.P_ASK.lt(q.STRIKE) & q.STRIKE.gt(0)
             & ((q.P_ASK - q.P_BID) / mid).le(cfg.max_relative_spread)
             & q.P_IV.between(.03, 3) & q.P_DELTA.between(-1, 0, inclusive="neither"))
    atm = q.iloc[(q.STRIKE - reference).abs().argmin()]
    if atm_iv is None:
        if not valid.iloc[atm.name] or not -.65 <= atm.P_DELTA <= -.35:
            return pd.DataFrame(), "invalid_atm"
        atm_iv = float(atm.P_IV)
    if not np.isfinite(atm_iv) or not .03 <= atm_iv <= 3:
        return pd.DataFrame(), "invalid_atm"
    horizon = atm_iv * np.sqrt(dte / 365)
    q["sigma"] = -np.log(q.STRIKE / reference) / horizon
    q["relative_spread"] = (q.P_ASK - q.P_BID) / mid
    q["eligible_quote"] = valid
    q["atm_iv"] = float(atm_iv)
    q["reference"] = reference
    q["dte"] = dte
    return q, "ok"


def price_vector(q, reference, cfg):
    """NaN for ineligible/inward strikes; arrays keep sensitivity scans cheap."""
    strikes = q.STRIKE.to_numpy()
    bid, ask = q.P_BID.to_numpy(), q.P_ASK.to_numpy()
    valid = q.eligible_quote.to_numpy()
    distance = np.abs(q.sigma.to_numpy() - cfg.anchor_sigma)
    indices = np.flatnonzero(valid & (distance <= cfg.anchor_tolerance))
    prices = np.full(len(q), np.nan)
    if not len(indices):
        return prices, None, "no_liquid_anchor"
    anchor = indices[np.argmin(distance[indices])]
    # A put price derivative cannot exceed one (zero-rate bound). Also check
    # that the splice is compatible with at least the next higher quoted put.
    slope = anchor_slope(reference, strikes[anchor], bid[anchor], cfg.alpha)
    higher = np.flatnonzero(valid & (strikes > strikes[anchor]))
    if slope > 1 or (len(higher) and slope > (ask[higher[0]] - bid[anchor]) / (strikes[higher[0]] - strikes[anchor]) + 1e-9):
        return prices, None, "anchor_slope_bound"
    deeper = valid & (strikes < strikes[anchor])
    if not deeper.any():
        return prices, None, "no_deeper_quotes"
    prices[deeper] = relative_put_price(strikes[deeper], reference, strikes[anchor], bid[anchor], cfg.alpha)
    return prices, anchor, "ok"


def score_prepared(q, reference, cfg):
    prices, index, reason = price_vector(q, reference, cfg)
    if reason != "ok":
        return pd.DataFrame(), reason
    anchor = q.iloc[index]
    eligible = np.isfinite(prices)
    result = q[eligible].copy()
    cost = result.P_ASK.to_numpy() + 2 * cfg.fee / cfg.multiplier
    edge = prices[eligible] / cost - 1
    return result.assign(model_price=prices[eligible], all_in_ask=cost, edge=edge,
                         underpriced=edge >= cfg.min_edge, anchor_strike=float(anchor.STRIKE),
                         anchor_bid=float(anchor.P_BID), anchor_actual_sigma=float(anchor.sigma), alpha=cfg.alpha), "ok"


def score_chain(chain: pd.DataFrame, reference: float, dte: int, cfg=PricingConfig(), atm_iv=None):
    """Same-expiry chain in index points. Sigma locates anchors, not a Pareto fit.

    European futures options may use the matching futures as reference, an
    explicitly separate adaptation from the historical SPX spot experiment.
    """
    q, reason = prepare_chain(chain, reference, dte, cfg, atm_iv)
    return score_prepared(q, reference, cfg) if reason == "ok" else (q, reason)
