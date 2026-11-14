"""
Milestone M1 report: the fixed core and the rule-based baselines over the whole sample.

    python download_multi.py                          # once
    python report_core_multi.py --outdir runs/multi_core

Builds the six rule-based portfolios of the brief (section 9) with no learning:

    core          risk parity, scaled to PORT_VOL       (the benchmark)
    core_trend    core with the 12-month trend rule as tilt
    equal_weight  1/8 of capital per asset, scaled to PORT_VOL
    sixty_forty   60% SPY / 40% IEF, scaled to PORT_VOL
    spy_vt        SPY alone, scaled to PORT_VOL
    spy           SPY buy and hold

Decisions run from the first warm date (MULTI_WARM_BARS after the first common date)
to the second-to-last trading day of --end_year, so the last return is realised on
the last trading day of --end_year.  Prices after --end_year are never loaded.

Outputs in --outdir
  core_returns.csv        date (day the return is realised) + one column per portfolio,
                          daily net excess return after costs
  core_metrics.csv        Sharpe with 95% CI, return, vol, max drawdown, turnover,
                          cost, gross exposure, share of days each cap binds
  core_yearly_sharpe.csv  Sharpe per calendar year
  core_yearly_vol.csv     realised volatility per calendar year
  core_sleeves.csv        per asset: mean vol, share of days at the vol floor and at
                          B_MAX, mean and max position in the core
  core_positions.csv      the core's daily positions
  config.json             every setting and a hash of both data files

The core's parameters are fixed by the brief: this report describes how the core
behaves, it is not a place to tune it.
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

import config
from common.stats import ann_sharpe
from multi.data import first_common_date, load_universe
from multi.features import compute_inputs, trend_rule_tilt
from multi.portfolio import (RULE_BASED, net_returns, portfolio_metrics,
                             rule_based_positions)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def training_window_years(first_decision):
    """Length in years of each walk-forward fold's training window."""
    out = {}
    for y in config.MULTI_WF_TEST_YEARS:
        lo = max(pd.Timestamp(f"{y - config.MULTI_WF_TRAIN_YEARS}-01-01"), first_decision)
        out[y] = (pd.Timestamp(f"{y}-01-01") - lo).days / 365.25
    return out


def multi_config():
    """Every constant of the multi-asset block of config.py."""
    keys = ["COST_PCT", "VOL_EWMA_SPAN"] + [
        k for k in vars(config) if k.isupper() and (
            k.startswith(("MULTI_", "SPEC_", "ALLOC_", "RISKFREE_"))
            or k in ("SLEEVE_VOL", "B_MAX", "PORT_VOL", "VOL_CAP", "LEV_MAX"))]
    return {k: getattr(config, k) for k in keys}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="runs/multi_core")
    p.add_argument("--end_year", type=int, default=2025)
    p.add_argument("--prices_csv", default=config.MULTI_DATA_CSV)
    p.add_argument("--riskfree_csv", default=config.RISKFREE_CSV)
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    end = f"{args.end_year + 1}-01-01"          # exclusive
    u = load_universe(config.MULTI_DOWNLOAD_START, end, data_csv=args.prices_csv,
                      riskfree_csv=args.riskfree_csv)
    inp = compute_inputs(u)
    T = len(u.dates)
    w = config.MULTI_WARM_BARS
    first_warm = inp.first_warm_index()
    if first_warm > w:
        raise SystemExit(f"Features need {first_warm} bars, more than "
                         f"MULTI_WARM_BARS = {w}.")
    start, stop = w, T - 1                      # decisions [start, stop)
    dates = u.dates[start + 1:stop + 1]         # realisation days

    print(f"First common date      {u.dates[0].date()}  "
          f"(cache: {first_common_date(args.prices_csv).date()})")
    print(f"Features warm after    {first_warm} bars")
    print(f"First decision date    {u.dates[start].date()}  "
          f"(first common date + {w} bars)")
    print(f"Returns realised       {dates[0].date()} → {dates[-1].date()}  "
          f"({len(dates)} days)")

    feas = training_window_years(u.dates[start])
    ok = [y for y, yrs in feas.items() if yrs >= config.MULTI_MIN_TRAIN_YEARS]
    print("Training window per test year (years from the first decision date or "
          f"Jan 1 of Y−{config.MULTI_WF_TRAIN_YEARS}, whichever is later):")
    print("  " + ", ".join(f"{y}: {yrs:.2f}" for y, yrs in feas.items()))
    print(f"First feasible test year (≥ {config.MULTI_MIN_TRAIN_YEARS} years): "
          f"{ok[0] if ok else 'none'}")

    positions = rule_based_positions(inp, trend_rule_tilt(inp))
    returns, rows = {}, []
    for name in RULE_BASED:
        x, info = positions[name]
        res = net_returns(x, inp.rex, start, stop)
        returns[name] = res["r"]
        rows.append({"portfolio": name, **portfolio_metrics(res, info, start, stop)})
    metrics = pd.DataFrame(rows)
    ret = pd.DataFrame(returns, index=pd.Index(dates, name="date"))

    years = ret.index.year
    yearly_sharpe = ret.groupby(years).apply(
        lambda d: d.apply(lambda s: ann_sharpe(s.values))).round(3)
    yearly_vol = (ret.groupby(years).std() * np.sqrt(config.MULTI_BARS_PER_YEAR)
                  * 100).round(2)
    yearly_sharpe.index.name = yearly_vol.index.name = "year"

    x_core, info = positions["core"]
    sl = slice(start, stop)
    sleeves = pd.DataFrame({
        "mean_vol_pct": (inp.vol[sl].mean(axis=0) * 100).round(2),
        "vol_floor_days_pct": ((inp.vol[sl] <= config.MULTI_VOL_FLOOR).mean(axis=0)
                               * 100).round(2),
        "b_max_days_pct": ((info["b"][sl] >= config.B_MAX).mean(axis=0) * 100).round(2),
        "mean_position": x_core[sl].mean(axis=0).round(3),
        "max_position": x_core[sl].max(axis=0).round(3),
    }, index=pd.Index(u.tickers, name="ticker"))
    core_extra = {
        "core_k_mean": round(float(info["k"][sl].mean()), 3),
        "core_k_min": round(float(info["k"][sl].min()), 3),
        "core_k_max": round(float(info["k"][sl].max()), 3),
    }

    ret.to_csv(os.path.join(args.outdir, "core_returns.csv"))
    metrics.to_csv(os.path.join(args.outdir, "core_metrics.csv"), index=False)
    yearly_sharpe.to_csv(os.path.join(args.outdir, "core_yearly_sharpe.csv"))
    yearly_vol.to_csv(os.path.join(args.outdir, "core_yearly_vol.csv"))
    sleeves.to_csv(os.path.join(args.outdir, "core_sleeves.csv"))
    pd.DataFrame(x_core[sl], index=pd.Index(u.dates[sl], name="date"),
                 columns=u.tickers).round(6).to_csv(
        os.path.join(args.outdir, "core_positions.csv"))
    cfg = {
        "script": "report_core_multi.py", "args": vars(args),
        "first_common_date": str(u.dates[0].date()),
        "first_decision_date": str(u.dates[start].date()),
        "first_return_date": str(dates[0].date()),
        "last_return_date": str(dates[-1].date()),
        "return_days": len(dates), "features_warm_after_bars": first_warm,
        "training_window_years": {str(k): round(v, 2) for k, v in feas.items()},
        "first_feasible_test_year": ok[0] if ok else None,
        "data_sha256": {args.prices_csv: _sha256(args.prices_csv),
                        args.riskfree_csv: _sha256(args.riskfree_csv)},
        "config": multi_config(), **core_extra,
    }
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\nRule-based portfolios, daily net excess returns after costs\n")
    print(metrics.set_index("portfolio").to_string())
    print("\nCore sleeves\n")
    print(sleeves.to_string())
    print(f"\nCore scale k: mean {core_extra['core_k_mean']}, "
          f"min {core_extra['core_k_min']}, max {core_extra['core_k_max']}")
    print("\nSharpe per year\n")
    print(yearly_sharpe.to_string())
    print(f"\nSaved → {args.outdir}")


if __name__ == "__main__":
    main()
