"""
One-time download of the multi-asset universe and the T-bill rate → local CSV caches.

    python download_multi.py

Writes
  data/multi_prices.csv   Date + one adjusted close (auto_adjust=True) per ticker of
                          config.MULTI_TICKERS.  Each ticker keeps its own dates; the
                          loader (multi/data.py) keeps only the dates all eight share.
  data/riskfree.csv       Date + the 13-week T-bill yield ^IRX in percent per year.

Every multi-asset script reads these files and never calls yfinance, so all folds,
seeds and reruns see byte-identical data (same reason as download_data.py).

The script stops with an error if any ticker or ^IRX comes back empty.  A missing
risk-free rate must never be replaced by zero: every reported number is an excess
return over it.
"""
import argparse
import os
import sys
import time

import pandas as pd
import yfinance as yf

import config


def _close_series(df, ticker):
    """The adjusted close column of one yfinance download, or None."""
    if df is None or len(df) == 0 or "Close" not in df.columns.get_level_values(0):
        return None
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close[ticker] if ticker in close.columns else close.iloc[:, 0]
    close = close.dropna().astype(float).rename(ticker)
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close.index.name = "Date"
    return close.sort_index() if len(close) else None


def _download_one(ticker, start, end, retries, wait, timeout):
    """One ticker, retried: parallel requests fail together on a flaky connection."""
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(ticker, start=start, end=end, interval="1d",
                             auto_adjust=True, progress=False, threads=False,
                             timeout=timeout)
        except Exception as e:                       # network errors, bad responses
            print(f"  {ticker}: {type(e).__name__}: {e}")
            df = None
        s = _close_series(df, ticker)
        if s is not None:
            print(f"  {ticker}: {len(s)} bars")
            return s
        if attempt < retries:
            print(f"  {ticker}: attempt {attempt}/{retries} failed, retrying in {wait}s")
            time.sleep(wait)
    print(f"  {ticker}: FAILED after {retries} attempts")
    return pd.Series(dtype=float, name=ticker)


def _closes(tickers, start, end, retries=3, wait=5, timeout=30):
    """Adjusted closes as a DataFrame with one column per ticker (union of dates)."""
    cols = [_download_one(t, start, end, retries, wait, timeout) for t in tickers]
    close = pd.concat(cols, axis=1).reindex(columns=tickers)
    close.index.name = "Date"
    return close.sort_index()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=config.MULTI_DOWNLOAD_START)
    p.add_argument("--end", default=None, help="exclusive; default: today")
    p.add_argument("--prices_out", default=config.MULTI_DATA_CSV)
    p.add_argument("--riskfree_out", default=config.RISKFREE_CSV)
    p.add_argument("--retries", type=int, default=3, help="attempts per ticker")
    p.add_argument("--wait", type=float, default=5, help="seconds between attempts")
    p.add_argument("--timeout", type=float, default=30, help="seconds per request")
    args = p.parse_args()
    net = dict(retries=args.retries, wait=args.wait, timeout=args.timeout)

    tickers = list(config.MULTI_TICKERS)
    print(f"Downloading {', '.join(tickers)} from {args.start} ...")
    prices = _closes(tickers, args.start, args.end, **net)

    empty = [t for t in tickers if prices[t].dropna().empty]
    if empty:
        sys.exit(f"ERROR: no data for {empty}. Nothing written. If the messages above "
                 "say the host could not be resolved or the connection timed out, "
                 "the problem is the network (DNS, VPN, firewall), not the tickers.")
    if (prices.dropna(how="all") <= 0).any().any():
        sys.exit("ERROR: non-positive prices in the download. Nothing written.")

    print(f"\n{'ticker':<8}{'first date':>14}{'last date':>14}{'bars':>8}")
    for t in tickers:
        s = prices[t].dropna()
        print(f"{t:<8}{str(s.index[0].date()):>14}{str(s.index[-1].date()):>14}"
              f"{len(s):>8}")
    common = prices.dropna(how="any")
    print(f"\nFirst common date: {common.index[0].date()}  "
          f"(last {common.index[-1].date()}, {len(common)} common bars)")
    late = prices.apply(lambda s: s.first_valid_index()).idxmax()
    print(f"Latest-starting ticker: {late}")

    print(f"\nDownloading {config.RISKFREE_TICKER} ...")
    irx = _closes([config.RISKFREE_TICKER], args.start, args.end, **net)
    irx = irx[config.RISKFREE_TICKER].dropna()
    if irx.empty:
        sys.exit(f"ERROR: {config.RISKFREE_TICKER} could not be downloaded. "
                 "Do not substitute a zero rate; nothing written.")
    print(f"{config.RISKFREE_TICKER}: {irx.index[0].date()} → {irx.index[-1].date()}, "
          f"{len(irx)} bars, range {irx.min():.3f}% to {irx.max():.3f}%")
    missing = common.index.difference(irx.index)
    print(f"Common price dates without a yield on the same day: {len(missing)} "
          "(the loader forward-fills the yield, never the prices)")

    for path in (args.prices_out, args.riskfree_out):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    prices.reset_index().to_csv(args.prices_out, index=False)
    irx.rename(config.RISKFREE_TICKER).rename_axis("Date").reset_index().to_csv(
        args.riskfree_out, index=False)
    print(f"\nSaved → {args.prices_out}, {args.riskfree_out}")


if __name__ == "__main__":
    main()
