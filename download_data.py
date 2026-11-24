"""
One-time download of the traded asset and VIX → local CSV cache (F9 reproducibility).

    python download_data.py                         # SPY + ^VIX daily, from 2005
    python download_data.py --start 1993-01-29      # longer history

One yfinance call fetches both tickers (adjusted closes, auto_adjust=True, so SPY's
dividends are in its returns) and writes data/spy_vix.csv with the columns
Date, SPY, ^VIX.  Each column keeps its own dates; common.data.load_prices reads
SPY and common.data.load_vix aligns VIX to SPY's calendar (forward-filling VIX
only).  Every other script reads this file instead of hitting yfinance, so all
folds / seeds / reruns use byte-identical data.

The script stops without writing anything if either ticker comes back empty: a
missing VIX must never be replaced silently.
"""
import argparse
import os
import sys
import warnings

import pandas as pd
import yfinance as yf

import config

# yfinance calls pandas with arguments pandas 3.1 deprecates; harmless noise
warnings.filterwarnings("ignore", message="The inplace keyword")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker",   default=config.TICKER)
    p.add_argument("--vix",      default=config.VIX_TICKER)
    p.add_argument("--start",    default=config.DOWNLOAD_START)
    p.add_argument("--end",      default=None)
    p.add_argument("--interval", default=config.INTERVAL)
    p.add_argument("--out",      default=config.DATA_CSV)
    args = p.parse_args()

    tickers = [args.ticker, args.vix]
    print(f"Downloading {', '.join(tickers)} from {args.start} ...")
    df = yf.download(tickers, start=args.start, end=args.end,
                     interval=args.interval, auto_adjust=True, progress=False)
    close = df["Close"] if "Close" in df.columns.get_level_values(0) else df
    close = close.reindex(columns=tickers)
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close.index.name = "Date"

    empty = [t for t in tickers if close[t].dropna().empty]
    if empty:
        sys.exit(f"ERROR: no data for {empty}. Nothing written. Yahoo sometimes "
                 "returns empty data; run the script again in a few minutes.")
    if (close.dropna(how="all") <= 0).any().any():
        sys.exit("ERROR: non-positive values in the download. Nothing written.")

    for t in tickers:
        s = close[t].dropna()
        print(f"  {t:<6} {len(s):>6} bars  {s.index[0].date()} → {s.index[-1].date()}"
              f"  (min {s.min():.2f}, max {s.max():.2f})")
    asset = close[args.ticker].dropna()
    no_vix = asset.index.difference(close[args.vix].dropna().index)
    print(f"  {args.ticker} days without a {args.vix} close: {len(no_vix)} "
          "(VIX is forward-filled on those days, prices never are)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    close.dropna(how="all").reset_index().to_csv(args.out, index=False)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
