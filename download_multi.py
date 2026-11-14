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

import pandas as pd
import yfinance as yf

import config


def _closes(tickers, start, end):
    """Adjusted closes as a DataFrame with one column per ticker (union of dates)."""
    df = yf.download(tickers, start=start, end=end, interval="1d",
                     auto_adjust=True, progress=False, group_by="column")
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=tickers)
    close = df["Close"] if "Close" in df.columns.get_level_values(0) else df
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close.reindex(columns=tickers)
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close.index.name = "Date"
    return close.sort_index()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=config.MULTI_DOWNLOAD_START)
    p.add_argument("--end", default=None, help="exclusive; default: today")
    p.add_argument("--prices_out", default=config.MULTI_DATA_CSV)
    p.add_argument("--riskfree_out", default=config.RISKFREE_CSV)
    args = p.parse_args()

    tickers = list(config.MULTI_TICKERS)
    print(f"Downloading {', '.join(tickers)} from {args.start} ...")
    prices = _closes(tickers, args.start, args.end)

    empty = [t for t in tickers if prices[t].dropna().empty]
    if empty:
        sys.exit(f"ERROR: no data for {empty}. Nothing written.")
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
    irx = _closes([config.RISKFREE_TICKER], args.start, args.end)
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
