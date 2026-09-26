"""
Price loading with a local CSV cache (F9 reproducibility).

Every training / evaluation script loads prices through `load_prices`.  When a
cache CSV exists (config.DATA_CSV or --data_csv) the prices come from it —
one download, byte-identical data for every fold, seed and rerun (yfinance
adjustments can drift between calls, which silently changes results).

Create the cache once with:  python download_data.py
"""
import os

import numpy as np
import pandas as pd


def load_prices(ticker, start, end, interval="1d", data_csv=None,
                verbose=True):
    """Return a pd.Series of close prices indexed by timestamp."""
    csv = data_csv
    if csv is None:
        try:
            import config
            csv = getattr(config, "DATA_CSV", None)
        except Exception:
            csv = None

    if csv and os.path.isfile(csv):
        df = pd.read_csv(csv, parse_dates=["Date"], index_col="Date")
        col = ticker if ticker in df.columns else "Close"
        if col not in df.columns:
            raise ValueError(f"{csv} has no column for {ticker} (cols: "
                             f"{list(df.columns)})")
        s = df[col].dropna()
        s = s.loc[(s.index >= pd.Timestamp(start)) &
                  (s.index < pd.Timestamp(end))]
        if verbose:
            print(f"  {len(s)} bars loaded from cache {csv} "
                  f"({start} → {end}).")
        if len(s) == 0:
            raise ValueError(f"No bars in cache for {start} → {end}. "
                             f"Re-run download_data.py with a wider range.")
        return s.astype(np.float64)

    import yfinance as yf
    if verbose:
        print(f"  Downloading {ticker} {start} → {end} "
              f"(interval={interval}) from yfinance ...")
    df = yf.download(ticker, start=start, end=end, interval=interval,
                     auto_adjust=True, progress=False)
    prices = df["Close"].squeeze().dropna()
    if verbose:
        print(f"  {len(prices)} bars loaded.")
    if len(prices) == 0:
        raise ValueError("yfinance returned no data — create a local cache "
                         "with download_data.py or pass --data_csv.")
    return prices


def warm_start_date(test_start, warm_bars, bars_per_week=5):
    """Calendar date roughly `warm_bars` trading bars before test_start (F8).
    Generous padding — extra history is harmless, missing history is not."""
    days = int(warm_bars * 7 / bars_per_week) + 30
    return (pd.Timestamp(test_start) - pd.Timedelta(days=days)).strftime(
        "%Y-%m-%d")


def trading_start_index(prices, test_start):
    """Index of the first bar at/after test_start in a warm price series."""
    idx = prices.index.searchsorted(pd.Timestamp(test_start))
    return int(idx)
