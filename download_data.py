"""
One-time price download → local CSV cache (F9 reproducibility).

    python download_data.py                    # ^DJI daily, full history
    python download_data.py --ticker ^GSPC

Every other script then reads data/prices.csv (config.DATA_CSV) instead of
hitting yfinance, so all folds / seeds / reruns use byte-identical data.
"""
import argparse
import os

import pandas as pd
import yfinance as yf

import config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker",   default=config.TICKER)
    p.add_argument("--start",    default="2005-01-01")
    p.add_argument("--end",      default=None)
    p.add_argument("--interval", default=config.INTERVAL)
    p.add_argument("--out",      default=config.DATA_CSV)
    args = p.parse_args()

    print(f"Downloading {args.ticker} from {args.start} ...")
    df = yf.download(args.ticker, start=args.start, end=args.end,
                     interval=args.interval, auto_adjust=True, progress=False)
    prices = df["Close"].squeeze().dropna()
    print(f"{len(prices)} bars ({prices.index[0].date()} → "
          f"{prices.index[-1].date()})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = pd.DataFrame({"Date": prices.index, "Close": prices.values})
    out.to_csv(args.out, index=False)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
