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
        s = s.astype(np.float64)
        s.attrs["data_csv"] = csv      # VIX is read from the same file (resolve_vix)
        return s

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


def load_vix(dates, data_csv=None, ticker=None, verbose=False):
    """
    VIX close on each of `dates` (the traded asset's calendar), from the cache.

    Only VIX is forward-filled, over the rare days the asset trades and VIX has no
    print; the asset's prices are never filled.  Raises if the cache has no VIX
    column or no VIX value on or before the first date: a missing VIX must never
    be replaced silently (re-run download_data.py).
    """
    import config
    csv = data_csv or getattr(config, "DATA_CSV", None)
    ticker = ticker or getattr(config, "VIX_TICKER", "^VIX")
    if not csv or not os.path.isfile(csv):
        raise FileNotFoundError(f"VIX needs the data cache {csv!r}; run "
                                "`python download_data.py` first.")
    df = pd.read_csv(csv, parse_dates=["Date"], index_col="Date")
    if ticker not in df.columns:
        raise ValueError(f"{csv} has no {ticker} column (cols: {list(df.columns)}); "
                         "re-run `python download_data.py`.")
    v = df[ticker].dropna().astype(np.float64).sort_index()
    v = v[~v.index.duplicated(keep="last")]
    idx = pd.DatetimeIndex(dates)
    out = v.reindex(v.index.union(idx)).ffill().reindex(idx)
    if out.isna().any():
        first = out.index[out.isna()][0].date()
        raise ValueError(f"No {ticker} value on or before {first}; re-run "
                         "download_data.py with an earlier start.")
    if verbose:
        n_filled = int((~idx.isin(v.index)).sum())
        print(f"  {ticker}: {len(out)} bars from {csv} ({n_filled} forward-filled)")
    return out


def vix_settings():
    """How the observation uses VIX: saved next to every model (config.json)."""
    import config
    return {"use_vix": bool(getattr(config, "USE_VIX", False)),
            "vix_lag_days": int(getattr(config, "VIX_LAG_DAYS", 0)),
            "vix_ticker": getattr(config, "VIX_TICKER", "^VIX")}


def check_vix_settings(cfg, where="model"):
    """
    Raise if a saved model was trained with other VIX settings than config.py.
    The observation has the same size with or without VIX, so a mismatch would
    otherwise run silently on inputs the model never saw.  Configs saved before
    the spy-vix branch have no VIX keys and 20 inputs; SB3 rejects those anyway.
    """
    if "use_vix" not in cfg:
        return
    now = vix_settings()
    diff = {k: (cfg.get(k), v) for k, v in now.items() if cfg.get(k) != v}
    if diff:
        hint, drop = [], []
        if cfg.get("use_vix") != now["use_vix"]:
            (hint.append("HRL_USE_VIX=0") if cfg.get("use_vix") is False
             else drop.append("HRL_USE_VIX"))
        if cfg.get("vix_lag_days") != now["vix_lag_days"]:
            (hint.append(f"HRL_VIX_LAG={cfg.get('vix_lag_days')}")
             if cfg.get("vix_lag_days") != 1 else drop.append("HRL_VIX_LAG"))
        if cfg.get("vix_ticker") != now["vix_ticker"]:
            drop.append("a different VIX_TICKER in config.py")
        fix = "; ".join(
            ([f"run the script as:  {' '.join(hint)} python ..."] if hint else []) +
            ([f"unset {' and '.join(drop)} (the model used the default)"] if drop else []))
        fix = fix[:1].upper() + fix[1:]
        raise SystemExit(f"{where} was trained with other VIX settings "
                         f"(saved, current): {diff}. {fix}, or retrain.")


def check_vix_dir(model_dir, where=None):
    """check_vix_settings on <model_dir>/config.json, if the model has one."""
    import json
    p = os.path.join(model_dir, "config.json")
    if os.path.isfile(p):
        with open(p) as f:
            check_vix_settings(json.load(f), where or model_dir)
