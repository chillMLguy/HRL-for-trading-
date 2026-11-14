"""
Download the multi-asset universe and the T-bill rate in one yfinance call, then
split it into the cache files the pipeline reads.

    python download_multi.py                       # download + split
    python download_multi.py --raw some_file.csv   # split a file saved earlier with
                                                   # yf.download(...).to_csv(...)

The download is a single call, the one that works reliably against Yahoo:

    yf.download(MULTI_TICKERS + ["^IRX"], period="max", auto_adjust=True).to_csv(raw)

Writes
  data/multi_raw.csv      the yfinance output as it came (kept for reproducibility)
  data/multi_prices.csv   Date + adjusted close per ETF.  Each ETF keeps its own dates;
                          multi/data.py keeps only the dates all eight share, so the
                          sample starts when the last ETF (DBC) starts.
  data/multi_ohlcv.csv    Date, Ticker, Open, High, Low, Close, Volume (unused so far)
  data/riskfree.csv       Date + the ^IRX yield, % per year

Every other multi-asset script reads these files and never calls yfinance.
"""
import argparse
import os
import sys

import pandas as pd

import config

FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def read_raw(path):
    """A yf.download(...).to_csv(...) file as a frame with (field, ticker) columns."""
    raw = pd.read_csv(path, header=[0, 1], index_col=0)
    raw.index = pd.to_datetime(raw.index, errors="coerce")
    raw = raw[raw.index.notna()]                     # drops a stray "Date" row, if any
    raw.index.name = "Date"
    return raw.apply(pd.to_numeric, errors="coerce").sort_index()


def split_raw(raw, tickers, rf_ticker):
    """(prices, ohlcv, irx) from the raw frame; raises if anything is missing."""
    have = set(raw.columns.get_level_values(1))
    missing = [t for t in tickers + [rf_ticker]
               if t not in have or raw[("Close", t)].dropna().empty]
    if missing:
        raise ValueError(f"No data for {missing}. Run the download again later "
                         "(Yahoo sometimes returns empty data); do not fill it in.")
    prices = pd.concat({t: raw[("Close", t)] for t in tickers}, axis=1)
    prices = prices.dropna(how="all")
    if (prices <= 0).any().any():
        raise ValueError("Non-positive prices in the download.")
    ohlcv = pd.concat(
        [pd.DataFrame({f: raw[(f, t)] for f in FIELDS if (f, t) in raw.columns})
         .dropna(subset=["Close"]).assign(Ticker=t) for t in tickers])
    ohlcv = ohlcv.reset_index().reindex(columns=["Date", "Ticker"] + FIELDS)
    irx = raw[("Close", rf_ticker)].dropna().rename(rf_ticker)
    return prices, ohlcv, irx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=None,
                   help="split this saved yfinance CSV instead of downloading")
    args = p.parse_args()
    tickers, rf = list(config.MULTI_TICKERS), config.RISKFREE_TICKER
    raw_path = args.raw or config.MULTI_RAW_CSV

    if args.raw is None:
        import yfinance as yf
        print(f"Downloading {', '.join(tickers + [rf])} ...")
        data = yf.download(tickers + [rf], period="max", auto_adjust=True)
        os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)
        data.to_csv(raw_path)
        print(f"Saved the raw download → {raw_path}")

    try:
        prices, ohlcv, irx = split_raw(read_raw(raw_path), tickers, rf)
    except ValueError as e:
        sys.exit(f"ERROR: {e} Nothing else written.")

    print(f"\n{'ticker':<8}{'first date':>14}{'last date':>14}{'bars':>8}")
    for t in tickers + [rf]:
        s = irx if t == rf else prices[t].dropna()
        print(f"{t:<8}{str(s.index[0].date()):>14}{str(s.index[-1].date()):>14}"
              f"{len(s):>8}")
    common = prices.dropna(how="any")
    print(f"\nFirst common date: {common.index[0].date()}  ({len(common)} common bars)")

    prices.reset_index().to_csv(config.MULTI_DATA_CSV, index=False)
    ohlcv.to_csv(config.MULTI_OHLCV_CSV, index=False)
    irx.rename_axis("Date").reset_index().to_csv(config.RISKFREE_CSV, index=False)
    print(f"Saved → {config.MULTI_DATA_CSV}, {config.MULTI_OHLCV_CSV}, "
          f"{config.RISKFREE_CSV}")


if __name__ == "__main__":
    main()
