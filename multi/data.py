"""
Data for the multi-asset pipeline: the eight-ETF universe on its common dates, the
T-bill rate, and excess returns (brief section 4).

Prices and the rate come only from the local caches written by download_multi.py.
Unlike common.data.load_prices, nothing here falls back to yfinance: a missing cache
is an error, so every run sees byte-identical data.

Alignment rules
  * Only dates on which all eight tickers have a price are kept.  Prices are never
    forward-filled (that would create zero returns and understate volatility).
  * The T-bill yield is forward-filled onto the price calendar, and only the yield.
  * Everything is indexed by trading day t of the common calendar:

        R[t]   = P[t] / P[t-1] - 1          return realised at close t (NaN at t = 0)
        rf[t]  = IRX[t-1] / 100 / 252       rate for close t-1 → close t, i.e. the
                                            last yield known at the start of the period
        Rex[t] = R[t] - rf[t]

    A position x[t] decided at close t earns Rex[t+1].  This is the brief's
    R[i,t+1], rf[t+1] and Rex[i,t+1], stored on the row of the day they are realised.
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from common.data import load_prices


@dataclass
class Universe:
    """Prices, the known yield and returns on the common calendar."""
    prices: pd.DataFrame   # P, one column per ticker in config order
    irx: pd.Series         # last ^IRX yield known at each date, % per year
    R: pd.DataFrame        # simple returns realised at each date
    rf: pd.Series          # daily risk-free rate realised at each date
    Rex: pd.DataFrame      # excess returns realised at each date

    @property
    def dates(self):
        return self.prices.index

    @property
    def tickers(self):
        return list(self.prices.columns)


def _require(path, what):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"{what} cache {path!r} not found. Run `python download_multi.py` first. "
            "The multi-asset pipeline never downloads data on the fly.")


def load_prices_wide(start, end, data_csv=None, tickers=None):
    """
    Adjusted closes of `tickers` on the dates all of them share, start ≤ date < end.
    `end` is exclusive, as in common.data.load_prices.
    """
    csv = data_csv or config.MULTI_DATA_CSV
    tickers = list(tickers or config.MULTI_TICKERS)
    _require(csv, "Price")
    cols = [load_prices(t, start, end, data_csv=csv, verbose=False).rename(t)
            for t in tickers]
    prices = pd.concat(cols, axis=1, join="inner").sort_index()
    prices.index.name = "Date"
    check_alignment(prices, tickers)
    return prices


def load_riskfree_known(dates, riskfree_csv=None):
    """
    The last ^IRX yield (% per year) known at each of `dates`: yields dated on or
    before each date, forward-filled across gaps in the yield series only.
    """
    csv = riskfree_csv or config.RISKFREE_CSV
    _require(csv, "Risk-free rate")
    df = pd.read_csv(csv, parse_dates=["Date"], index_col="Date")
    col = config.RISKFREE_TICKER if config.RISKFREE_TICKER in df.columns else "Close"
    if col not in df.columns:
        raise ValueError(f"{csv} has no {config.RISKFREE_TICKER} column "
                         f"(cols: {list(df.columns)})")
    y = df[col].dropna().astype(np.float64).sort_index()
    y = y[~y.index.duplicated(keep="last")]
    known = y.reindex(y.index.union(dates)).ffill().reindex(dates)
    if known.isna().any():
        first = known.index[known.isna()][0].date()
        raise ValueError(f"No T-bill yield known on or before {first}. Re-run "
                         "download_multi.py with an earlier start; do not use zero.")
    known.name = "irx"
    return known


def excess_returns(prices, irx_known, bars_per_year=None):
    """(R, rf, Rex) on the rows of `prices`, realised-day convention (module doc)."""
    bpy = bars_per_year or config.MULTI_BARS_PER_YEAR
    R = prices / prices.shift(1) - 1.0
    rf = irx_known.reindex(prices.index).shift(1) / 100.0 / bpy
    rf.name = "rf"
    Rex = R.sub(rf, axis=0)
    return R, rf, Rex


def universe_from_frames(prices, irx_known):
    """Universe from an aligned price frame and the known-yield series (also for tests)."""
    check_alignment(prices, list(prices.columns))
    irx = irx_known.reindex(prices.index)
    if irx.isna().any():
        raise ValueError("Risk-free yield missing on some price dates.")
    R, rf, Rex = excess_returns(prices, irx)
    return Universe(prices=prices, irx=irx, R=R, rf=rf, Rex=Rex)


def load_universe(start, end, data_csv=None, riskfree_csv=None, tickers=None):
    """Prices, rate and excess returns for start ≤ date < end, from the caches."""
    prices = load_prices_wide(start, end, data_csv=data_csv, tickers=tickers)
    irx = load_riskfree_known(prices.index, riskfree_csv=riskfree_csv)
    return universe_from_frames(prices, irx)


def check_alignment(prices, tickers):
    """Raise unless `prices` is a clean, aligned panel of the given tickers."""
    if list(prices.columns) != list(tickers):
        raise ValueError(f"Columns {list(prices.columns)} != tickers {list(tickers)}")
    idx = prices.index
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        raise ValueError("Price index must be strictly increasing.")
    if prices.isna().any().any():
        raise ValueError("Missing prices after alignment.")
    if (prices.values <= 0).any():
        raise ValueError("Non-positive prices.")
    if len(prices) < 2:
        raise ValueError("Fewer than two common dates.")


def first_common_date(data_csv=None, tickers=None):
    """First date on which every ticker has a price, read from the cache."""
    csv = data_csv or config.MULTI_DATA_CSV
    _require(csv, "Price")
    df = pd.read_csv(csv, parse_dates=["Date"], index_col="Date")
    tickers = list(tickers or config.MULTI_TICKERS)
    return df[tickers].dropna(how="any").index[0]
