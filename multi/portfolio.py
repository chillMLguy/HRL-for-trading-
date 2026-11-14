"""
Portfolio construction for the multi-asset pipeline (brief section 5) and the
rule-based portfolios of section 9.

Every portfolio in the project gets its positions from `build_positions`; training
environments, evaluation and baselines must not compute positions any other way.
The only exception is `spy` (buy and hold), a constant unscaled position that by
definition has no volatility scaling and no caps.

    1. sleeves      b[i,t]      = min(SLEEVE_VOL / vol[i,t], B_MAX)
    2. core         w_core      = beta_base · b
                    k[t]        = PORT_VOL / max(sqrt(w_core' Cov w_core), eps)
                    x_core      = k · w_core
    3. tilts        x_raw       = G · (beta / beta_base) · m · x_core
    4. caps         ex-ante vol above VOL_CAP  → scale all positions by one factor
                    gross above LEV_MAX        → scale all positions by one factor

The scale k comes from the core, never from the tilted portfolio, so tilts change
total risk one for one.  Neutral inputs (m = 1, beta = beta_base, G = 1) reproduce
the core exactly.

Returns follow multi/data.py: x[t] is decided at close t and earns Rex[t+1]; the cost
c · sum|x[t] − x[t−1]| is charged in the same return, and the first day pays for the
move from zero.
"""
import numpy as np

import config
from common.stats import ann_sharpe, bootstrap_sharpe_ci, max_drawdown

RULE_BASED = ["core", "core_trend", "equal_weight", "sixty_forty", "spy_vt", "spy"]


def sleeve_sizes(vol, sleeve_vol=None, b_max=None):
    """Step 1: each asset scaled to the same volatility, capped at B_MAX."""
    sleeve_vol = config.SLEEVE_VOL if sleeve_vol is None else sleeve_vol
    b_max = config.B_MAX if b_max is None else b_max
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.minimum(sleeve_vol / np.asarray(vol, dtype=np.float64), b_max)


def ex_ante_vol(x, cov):
    """sqrt(x[t]' Cov[t] x[t]) for every row."""
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.einsum("ti,tij,tj->t", x, cov, x))


def build_positions(vol, cov, m=None, beta=None, G=None, base=None, sleeve=None,
                    caps=True, return_info=False):
    """
    Positions x[t, i] as fractions of capital (section 5).

    vol    (T, N) annualised asset vol;  cov (T, N, N) annualised covariance
    m      (T, N) specialist multipliers in [0, 2], default 1
    beta   (T, N) or (N,) risk budgets summing to 1, default `base`
    G      (T,) or scalar gross risk scalar in [0.75, 1.25], default 1
    base   (N,) base budgets, default config.MULTI_BASE_BUDGETS
    sleeve (T, N) sleeve sizes in place of step 1 (capital-weighted baselines)
    caps   apply step 4 (False only in tests)

    Rows where vol or cov are not yet defined give NaN positions.
    """
    vol = np.asarray(vol, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    T, N = vol.shape
    base = np.asarray(config.MULTI_BASE_BUDGETS if base is None else base,
                      dtype=np.float64)
    b = sleeve_sizes(vol) if sleeve is None else np.broadcast_to(
        np.asarray(sleeve, dtype=np.float64), (T, N))

    w_core = base * b
    v_core = ex_ante_vol(w_core, cov)
    k = config.PORT_VOL / np.maximum(v_core, config.MULTI_VCORE_EPS)
    x_core = k[:, None] * w_core

    x = x_core
    if m is not None:
        x = np.asarray(m, dtype=np.float64) * x
    if beta is not None:
        if (base <= 0).any():
            raise ValueError("Budgets can only be tilted when every base budget > 0.")
        x = (np.asarray(beta, dtype=np.float64) / base) * x
    if G is not None:
        g = np.asarray(G, dtype=np.float64)
        x = (g[:, None] if g.ndim == 1 else g) * x
    x_uncapped = x

    vol_bind = np.zeros(T, dtype=bool)
    lev_bind = np.zeros(T, dtype=bool)
    if caps:
        v = ex_ante_vol(x, cov)
        with np.errstate(invalid="ignore", divide="ignore"):
            vol_bind = v > config.VOL_CAP
            x = np.where(vol_bind[:, None], x * (config.VOL_CAP / v)[:, None], x)
            gross = np.abs(x).sum(axis=1)
            lev_bind = gross > config.LEV_MAX
            x = np.where(lev_bind[:, None], x * (config.LEV_MAX / gross)[:, None], x)

    if not return_info:
        return x
    return x, dict(b=b, k=k, v_core=v_core, x_core=x_core, x_uncapped=x_uncapped,
                   vol_bind=vol_bind, lev_bind=lev_bind,
                   ex_ante_vol=ex_ante_vol(x, cov), gross=np.abs(x).sum(axis=1))


def net_returns(x, rex, start, end, cost_pct=None):
    """
    Net excess returns of the positions decided at closes start, ..., end − 1.

        r[t+1] = sum_i x[t,i] · Rex[t+1,i]  −  c · sum_i |x[t,i] − x[t−1,i]|

    with x[start − 1] = 0, so the first day pays for the move from zero.  All arrays
    returned are aligned to the realisation rows start + 1, ..., end.
    """
    c = config.COST_PCT if cost_pct is None else cost_pct
    x = np.asarray(x, dtype=np.float64)
    rex = np.asarray(rex, dtype=np.float64)
    if not 0 <= start < end < len(rex):
        raise ValueError(f"Need 0 ≤ start < end < {len(rex)}, got {start}, {end}.")
    X = x[start:end]
    if not np.isfinite(X).all():
        raise ValueError("Positions undefined on some decision day (inputs not warm?).")
    nxt = rex[start + 1:end + 1]
    if not np.isfinite(nxt).all():
        raise ValueError("Excess returns undefined on some realisation day.")
    prev = np.vstack([np.zeros((1, X.shape[1])), X[:-1]])
    turnover = np.abs(X - prev).sum(axis=1)
    pnl = (X * nxt).sum(axis=1)
    cost = c * turnover
    return dict(r=pnl - cost, pnl=pnl, cost=cost, turnover=turnover,
                gross=np.abs(X).sum(axis=1))


def fixed_weights(tickers, weights):
    """Base vector with the given capital weights, zero elsewhere."""
    w = np.zeros(len(tickers))
    for t, v in weights.items():
        w[tickers.index(t)] = v
    return w


def rule_based_positions(inputs, trend_tilt):
    """
    Positions of the six rule-based portfolios of section 9, each as (x, info).

    core          section 5 with neutral inputs
    core_trend    m = 1 + clip(12-month trend feature, −1, 1), fixed budgets
    equal_weight  one eighth of capital per asset, scaled to PORT_VOL, same caps
    sixty_forty   60% SPY / 40% IEF, scaled the same way
    spy_vt        SPY alone, scaled the same way
    spy           SPY at weight 1, unscaled and uncapped (info is None)
    """
    vol, cov, tickers = inputs.vol, inputs.cov, inputs.tickers
    N = len(tickers)
    ones = np.ones_like(vol)
    out = {
        "core": build_positions(vol, cov, return_info=True),
        "core_trend": build_positions(vol, cov, m=1.0 + trend_tilt, return_info=True),
        "equal_weight": build_positions(vol, cov, base=np.full(N, 1.0 / N),
                                        sleeve=ones, return_info=True),
        "sixty_forty": build_positions(
            vol, cov, base=fixed_weights(tickers, {"SPY": 0.6, "IEF": 0.4}),
            sleeve=ones, return_info=True),
        "spy_vt": build_positions(vol, cov, base=fixed_weights(tickers, {"SPY": 1.0}),
                                  sleeve=ones, return_info=True),
    }
    spy = np.tile(fixed_weights(tickers, {"SPY": 1.0}), (len(vol), 1))
    out["spy"] = (spy, None)
    return out


def _r(v, nd):
    return round(float(v), nd)


def portfolio_metrics(res, info=None, start=None, end=None, ann=None):
    """
    One row of the summary table for a result of `net_returns`.
    `info`, `start`, `end` add how often each cap binds on the decision days.
    """
    ann = ann or config.MULTI_BARS_PER_YEAR
    r = res["r"]
    eq = np.cumprod(1.0 + r)
    lo, hi = bootstrap_sharpe_ci(r, ann)
    row = {
        "days": len(r),
        "sharpe": _r(ann_sharpe(r, ann), 3),
        "sharpe_ci_lo": _r(lo, 3),
        "sharpe_ci_hi": _r(hi, 3),
        "ann_ret_pct": _r(r.mean() * ann * 100, 2),
        "ann_vol_pct": _r(r.std() * np.sqrt(ann) * 100, 2),
        "max_dd_pct": _r(max_drawdown(eq) * 100, 2),
        "turnover_per_year": _r(res["turnover"].mean() * ann, 2),
        "cost_per_year_pct": _r(res["cost"].mean() * ann * 100, 3),
        "gross_mean": _r(res["gross"].mean(), 3),
        "gross_max": _r(res["gross"].max(), 3),
    }
    if info is not None:
        row["vol_cap_days_pct"] = _r(info["vol_bind"][start:end].mean() * 100, 2)
        row["lev_cap_days_pct"] = _r(info["lev_bind"][start:end].mean() * 100, 2)
    else:
        row["vol_cap_days_pct"] = row["lev_cap_days_pct"] = float("nan")
    return row
