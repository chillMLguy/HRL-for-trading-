"""
Causal risk estimates and features for the multi-asset pipeline (brief sections 4, 6).

Every value on row t uses prices up to and including close t, and nothing later:
EWMA and rolling windows only, never centred windows, bfill or shift(-1).  The test
"no look-ahead" in tests/test_multi.py changes every price after a random day and
checks that nothing up to that day moves.

    vol[t]    annualised EWMA std of daily returns, span VOL_EWMA_SPAN (33),
              min_periods 20, floored at MULTI_VOL_FLOOR (0.02)
    cov[t]    252 × EWMA covariance of the eight daily returns, span 60, min_periods 60
    feats[t]  the twelve market features 0-11 of the specialists' observation,
              shape (T, N, 12).  Features 8-11 are the same for every asset and are
              also the allocator's market context.  Feature 12 (own previous tilt) and
              the asset identity are added by the specialist environment.

All inputs to the learners are dimensionless and clipped.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config

FEATURE_NAMES = [
    "z21", "z63", "z126", "z252",          # 0-3 trend
    "z1", "z5",                            # 4-5 reversal
    "vol_ratio",                           # 6
    "vol_level",                           # 7
    "equity_stress",                       # 8  (feature 7 of SPY)
    "stock_bond_corr",                     # 9  (SPY-IEF 60-day correlation)
    "avg_corr",                            # 10 (mean pairwise 60-day correlation)
    "trend_breadth",                       # 11 (2 × share with z126 > 0 − 1)
]
CONTEXT_SLICE = slice(8, 12)


@dataclass
class MarketInputs:
    """Everything the portfolio and the learners read, one row per trading day."""
    dates: pd.DatetimeIndex
    tickers: list
    rex: np.ndarray      # (T, N) excess return realised at close t
    rf: np.ndarray       # (T,)
    vol: np.ndarray      # (T, N)
    cov: np.ndarray      # (T, N, N)
    feats: np.ndarray    # (T, N, 12) market features 0-11
    ctx: np.ndarray      # (T, 4) features 8-11

    @property
    def n_assets(self):
        return len(self.tickers)

    def z(self, h):
        """The trend/reversal feature z(h), shape (T, N)."""
        horizons = list(config.MULTI_TREND_HORIZONS) + list(config.MULTI_REVERSAL_HORIZONS)
        return self.feats[:, :, horizons.index(h)]

    def first_warm_index(self):
        """First row on which vol, cov and every market feature are defined."""
        ok = (np.isfinite(self.vol).all(axis=1)
              & np.isfinite(self.cov).all(axis=(1, 2))
              & np.isfinite(self.feats).all(axis=(1, 2)))
        if not ok.any():
            raise ValueError("Features never become warm on this sample.")
        first = int(np.argmax(ok))
        if not ok[first:].all():
            raise ValueError("Features turn undefined again after warming up.")
        return first


def ewma_vol(R, span=None, min_periods=None, floor=None, bars_per_year=None):
    """Annualised causal EWMA volatility of each column of R, floored."""
    span = span or config.VOL_EWMA_SPAN
    mp = min_periods or config.MULTI_VOL_MIN_PERIODS
    floor = config.MULTI_VOL_FLOOR if floor is None else floor
    bpy = bars_per_year or config.MULTI_BARS_PER_YEAR
    vol = R.ewm(span=span, min_periods=mp).std() * np.sqrt(bpy)
    return vol.clip(lower=floor)


def ewma_cov(R, span=None, min_periods=None, bars_per_year=None):
    """Annualised causal EWMA covariance matrices, shape (T, N, N)."""
    span = span or config.MULTI_COV_SPAN
    mp = min_periods or config.MULTI_COV_MIN_PERIODS
    bpy = bars_per_year or config.MULTI_BARS_PER_YEAR
    T, N = R.shape
    c = R.ewm(span=span, min_periods=mp).cov()
    # pandas returns rows (date, column) in date order then column order
    c = c.reindex(pd.MultiIndex.from_product([R.index, R.columns]))
    return c.to_numpy(dtype=np.float64).reshape(T, N, N) * bpy


def zscore_returns(prices, vol, h, bars_per_year=None, clip=None):
    """z(h) = (ln P[t] − ln P[t−h]) / (daily vol[t] · √h), clipped."""
    bpy = bars_per_year or config.MULTI_BARS_PER_YEAR
    clip = config.MULTI_Z_CLIP if clip is None else clip
    logp = np.log(prices)
    dvol = vol / np.sqrt(bpy)
    z = (logp - logp.shift(h)) / (dvol * np.sqrt(h))
    return z.clip(-clip, clip)


def _log_ratio(num, den, clip):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(num / den)
    return out.clip(-clip, clip)


def vol_ratio(R, windows=None, clip=None):
    """Feature 6: ln(std of the last 10 returns / std of the last 60), clipped."""
    short, long_ = windows or config.MULTI_VOLRATIO_WINDOWS
    clip = config.MULTI_LOG_CLIP if clip is None else clip
    s = R.rolling(short, min_periods=short).std()
    l = R.rolling(long_, min_periods=long_).std()
    return _log_ratio(s, l, clip)


def vol_level(vol, window=None, clip=None):
    """Feature 7: ln(vol[t] / median of vol over the previous 252 days), clipped."""
    window = window or config.MULTI_VOL_LEVEL_WINDOW
    clip = config.MULTI_LOG_CLIP if clip is None else clip
    med = vol.shift(1).rolling(window, min_periods=window).median()
    return _log_ratio(vol, med, clip)


def rolling_corr_matrices(R, window=None):
    """Rolling correlation matrices, shape (T, N, N); NaN until the window is full."""
    window = window or config.MULTI_CORR_WINDOW
    T, N = R.shape
    c = R.rolling(window, min_periods=window).corr()
    c = c.reindex(pd.MultiIndex.from_product([R.index, R.columns]))
    return c.to_numpy(dtype=np.float64).reshape(T, N, N)


def context_features(R, vol, z_breadth, tickers):
    """Features 8-11, shape (T, 4); the same for every asset on a given day."""
    i_spy, i_ief = tickers.index("SPY"), tickers.index("IEF")
    vlev = vol_level(vol)
    stress = vlev.iloc[:, i_spy].to_numpy()
    corr = rolling_corr_matrices(R)
    N = len(tickers)
    stock_bond = corr[:, i_spy, i_ief]
    off = ~np.eye(N, dtype=bool)
    avg_corr = corr[:, off].mean(axis=1)            # NaN until the window is full
    zb = z_breadth.to_numpy()
    with np.errstate(invalid="ignore"):
        breadth = 2.0 * (zb > 0).mean(axis=1) - 1.0
    breadth[np.isnan(zb).any(axis=1)] = np.nan
    return np.column_stack([stress, stock_bond, avg_corr, breadth])


def market_features(prices, R, vol):
    """The twelve market features 0-11, shape (T, N, 12), and the context (T, 4)."""
    tickers = list(prices.columns)
    zs = {h: zscore_returns(prices, vol, h)
          for h in list(config.MULTI_TREND_HORIZONS) + list(config.MULTI_REVERSAL_HORIZONS)}
    per_asset = [zs[h].to_numpy() for h in config.MULTI_TREND_HORIZONS]
    per_asset += [zs[h].to_numpy() for h in config.MULTI_REVERSAL_HORIZONS]
    per_asset += [vol_ratio(R).to_numpy(), vol_level(vol).to_numpy()]
    ctx = context_features(R, vol, zs[config.MULTI_BREADTH_HORIZON], tickers)
    T, N = prices.shape
    feats = np.empty((T, N, config.MULTI_N_MARKET_FEATURES), dtype=np.float64)
    for j, f in enumerate(per_asset):
        feats[:, :, j] = f
    feats[:, :, CONTEXT_SLICE] = ctx[:, None, :]
    return feats, ctx


def compute_inputs(universe):
    """MarketInputs for a multi.data.Universe."""
    vol = ewma_vol(universe.R)
    cov = ewma_cov(universe.R)
    feats, ctx = market_features(universe.prices, universe.R, vol)
    return MarketInputs(
        dates=universe.dates, tickers=universe.tickers,
        rex=universe.Rex.to_numpy(dtype=np.float64),
        rf=universe.rf.to_numpy(dtype=np.float64),
        vol=vol.to_numpy(dtype=np.float64), cov=cov, feats=feats, ctx=ctx)


def trend_rule_tilt(inputs):
    """Tilt of the core_trend baseline: clip(z(252), −1, 1); m = 1 + tilt."""
    c = config.MULTI_TREND_RULE_CLIP
    return np.clip(inputs.z(config.MULTI_TREND_RULE_HORIZON), -c, c)
