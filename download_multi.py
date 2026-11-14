"""
One-time download of the multi-asset universe and the T-bill rate → local CSV caches.

    python download_multi.py

Writes
  data/multi_prices.csv   Date + one adjusted close (auto_adjust=True) per ticker of
                          config.MULTI_TICKERS.  Each ticker keeps its own dates; the
                          loader (multi/data.py) keeps only the dates all eight share,
                          so the sample starts when the last ETF (DBC) starts.
  data/multi_ohlcv.csv    Date, Ticker, Open, High, Low, Close, Volume from the same
                          download (OHLC adjusted by the same factor as the close).
                          Not read by any model yet; cached now so that range and
                          volume features, if they are ever tested, use the same data.
  data/riskfree.csv       Date + the 13-week T-bill yield ^IRX in percent per year.

Every multi-asset script reads these files and never calls yfinance, so all folds,
seeds and reruns see byte-identical data (same reason as download_data.py).

Each ticker is fetched on its own with retries, because parallel requests fail
together on a flaky connection.  The script stops with an error if any ticker or
^IRX comes back empty.  A missing risk-free rate must never be replaced by zero:
every reported number is an excess return over it.
"""
import argparse
import os
import sys
import time

import pandas as pd
import yfinance as yf

import config

FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def _bars(df, ticker):
    """Single-level OHLCV frame of one yfinance download, or None if empty."""
    if df is None or len(df) == 0:
        return None
    if isinstance(df.columns, pd.MultiIndex):            # (Price, Ticker)
        lvl = df.columns.get_level_values(-1)
        df = df.loc[:, lvl == ticker] if ticker in set(lvl) else df
        df = df.droplevel(-1, axis=1)
    if "Close" not in df.columns:
        return None
    out = df.reindex(columns=FIELDS).astype(float)
    out = out[out["Close"].notna()]
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    out.index.name = "Date"
    return out.sort_index() if len(out) else None


def _download_one(ticker, start, end, retries, wait, timeout):
    """One ticker, retried.  Returns its OHLCV frame or None."""
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(ticker, start=start, end=end, interval="1d",
                             auto_adjust=True, progress=False, threads=False,
                             timeout=timeout)
        except Exception as e:                       # network errors, bad responses
            print(f"  {ticker}: {type(e).__name__}: {e}")
            df = None
        bars = _bars(df, ticker)
        if bars is not None:
            print(f"  {ticker}: {len(bars)} bars")
            return bars
        if attempt < retries:
            print(f"  {ticker}: attempt {attempt}/{retries} failed, retrying in {wait}s")
            time.sleep(wait)
    print(f"  {ticker}: FAILED after {retries} attempts")
    return None


def _download(tickers, start, end, retries=3, wait=5, timeout=30):
    """{ticker: OHLCV frame or None}."""
    return {t: _download_one(t, start, end, retries, wait, timeout) for t in tickers}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=config.MULTI_DOWNLOAD_START)
    p.add_argument("--end", default=None, help="exclusive; default: today")
    p.add_argument("--prices_out", default=config.MULTI_DATA_CSV)
    p.add_argument("--ohlcv_out", default=config.MULTI_OHLCV_CSV)
    p.add_argument("--riskfree_out", default=config.RISKFREE_CSV)
    p.add_argument("--retries", type=int, default=3, help="attempts per ticker")
    p.add_argument("--wait", type=float, default=5, help="seconds between attempts")
    p.add_argument("--timeout", type=float, default=30, help="seconds per request")
    args = p.parse_args()
    net = dict(retries=args.retries, wait=args.wait, timeout=args.timeout)

    tickers = list(config.MULTI_TICKERS)
    print(f"Downloading {', '.join(tickers)} from {args.start} ...")
    bars = _download(tickers, args.start, args.end, **net)
    empty = [t for t, b in bars.items() if b is None]
    if empty:
        sys.exit(f"ERROR: no data for {empty}. Nothing written. If the messages above "
                 "say the host could not be resolved or the connection timed out, "
                 "the problem is the network (DNS, VPN, firewall), not the tickers.")
    prices = pd.concat([bars[t]["Close"].rename(t) for t in tickers], axis=1)
    prices.index.name = "Date"
    if (prices <= 0).any().any():
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
    irx_bars = _download([config.RISKFREE_TICKER], args.start, args.end, **net)
    irx_bars = irx_bars[config.RISKFREE_TICKER]
    if irx_bars is None:
        sys.exit(f"ERROR: {config.RISKFREE_TICKER} could not be downloaded. "
                 "Do not substitute a zero rate; nothing written.")
    irx = irx_bars["Close"].rename(config.RISKFREE_TICKER)
    print(f"{config.RISKFREE_TICKER}: {irx.index[0].date()} → {irx.index[-1].date()}, "
          f"{len(irx)} bars, range {irx.min():.3f}% to {irx.max():.3f}%")
    missing = common.index.difference(irx.index)
    print(f"Common price dates without a yield on the same day: {len(missing)} "
          "(the loader forward-fills the yield, never the prices)")

    for path in (args.prices_out, args.ohlcv_out, args.riskfree_out):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    prices.reset_index().to_csv(args.prices_out, index=False)
    ohlcv = pd.concat([bars[t].assign(Ticker=t) for t in tickers]).reset_index()
    ohlcv[["Date", "Ticker"] + FIELDS].to_csv(args.ohlcv_out, index=False)
    irx.rename_axis("Date").reset_index().to_csv(args.riskfree_out, index=False)
    print(f"\nSaved → {args.prices_out}, {args.ohlcv_out}, {args.riskfree_out}")


if __name__ == "__main__":
    main()
