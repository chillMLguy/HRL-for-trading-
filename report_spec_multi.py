"""
Milestone M2 report: the specialists on the first walk-forward fold.

    python report_spec_multi.py --outroot runs/multi_wf --test_year 2015

Reads the fold's cross-fit stream (<fold>/crossfit, or the chosen search config's) and
the final specialists (<fold>/models/specialists), and writes <fold>/m2/:

  active_sharpe.csv      active Sharpe against the core of the specialists and of the
                         trend rule on the same days: on fit windows (in sample) and on
                         the cross-fit blocks (out of sample), with 95% intervals and the
                         timing-only diagnostic
  spec_diagnostics.csv   per asset (section 9): mean tilt, share of days with |a| > 0.95,
                         lag-1 autocorrelation, correlation with the trend-rule tilt, mean
                         correlation between seeds, correlation of fold and final tilts
  training_curves.csv    fit-window active Sharpe at every tenth of training, mean over seeds
  m2_summary.json        the above in one place, plus cap binding and training time

Nothing here touches the test year.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import config
from common.stats import bootstrap_sharpe_ci
from multi.data import load_universe, training_window
from multi.features import compute_inputs, trend_rule_tilt
from multi.portfolio import (active_sharpe, build_positions, ex_ante_vol, net_returns,
                             tilt_returns)
from multi.spec_env import load_ensemble, specialist_tilts
from search_multi import load_settings
from train_crossfit_multi import stream_tilts


def active_row(name, where, inputs, tilts, b0, b1):
    r_core = net_returns(build_positions(inputs.vol, inputs.cov), inputs.rex, b0, b1)["r"]
    r = tilt_returns(inputs, tilts, b0, b1)["r"]
    const = np.tile(tilts.mean(axis=0), (len(tilts), 1))
    r_const = tilt_returns(inputs, const, b0, b1)["r"]
    lo, hi = bootstrap_sharpe_ci(r - r_core)
    return {"layer": name, "where": where, "first": str(inputs.dates[b0].date()),
            "last": str(inputs.dates[b1 - 1].date()), "days": b1 - b0,
            "active_sharpe": round(active_sharpe(r, r_core), 3),
            "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
            "timing_only_active_sharpe": round(active_sharpe(r, r_const), 3),
            "mean_tilt": round(float(tilts.mean()), 3)}


def corr_cols(a, b):
    return np.array([np.corrcoef(a[:, i], b[:, i])[0, 1] for i in range(a.shape[1])])


def lag1(a):
    return np.array([np.corrcoef(a[1:, i], a[:-1, i])[0, 1] for i in range(a.shape[1])])


def cap_share(inputs, tilts, b0, b1):
    m = np.ones_like(inputs.vol)
    m[b0:b1] = 1.0 + tilts
    _, info = build_positions(inputs.vol, inputs.cov, m=m, return_info=True)
    return (round(float(info["vol_bind"][b0:b1].mean() * 100), 2),
            round(float(info["lev_bind"][b0:b1].mean() * 100), 2))


def curves(pattern):
    frames = [pd.read_csv(f) for f in sorted(glob.glob(pattern))]
    if not frames:
        return None
    df = pd.concat(frames)
    return df.groupby("step")[["fit_active_sharpe", "mean_abs_tilt"]].mean().round(3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outroot", default=os.path.join("runs", "multi_wf"))
    p.add_argument("--test_year", type=int, default=config.MULTI_WF_TEST_YEARS[0])
    args = p.parse_args()
    Y = args.test_year
    fold = os.path.join(args.outroot, f"fold{Y}")
    out = os.path.join(fold, "m2")
    os.makedirs(out, exist_ok=True)
    settings = load_settings(args.outroot)
    choice = settings.get("spec_choice", "default")
    cf_root = fold if choice == "default" else os.path.join(fold, "search", choice)

    u = load_universe(config.MULTI_DOWNLOAD_START, f"{Y}-01-01")
    inputs = compute_inputs(u)
    lo, hi = training_window(u.dates, Y, config.MULTI_WARM_BARS)
    trend = trend_rule_tilt(inputs)
    stream = pd.read_csv(os.path.join(cf_root, "crossfit", "tilts.csv"), parse_dates=["date"])
    with open(os.path.join(cf_root, "crossfit", "blocks.json")) as f:
        blocks = json.load(f)["blocks"]
    s0, A = stream_tilts(stream, inputs)
    s1 = s0 + len(A)

    rows = []
    # out of sample: the stream, the whole of it and block by block
    rows.append(active_row("specialists", "cross-fit blocks", inputs, A, s0, s1))
    rows.append(active_row("trend rule", "cross-fit blocks", inputs, trend[s0:s1], s0, s1))
    for blk in blocks:
        b0, b1 = blk["fit_hi"], blk["fit_hi"] + int((stream["fold"] == blk["fold"]).sum())
        a = A[b0 - s0:b1 - s0]
        rows.append(active_row("specialists", f"block {blk['fold']}", inputs, a, b0, b1))
        rows.append(active_row("trend rule", f"block {blk['fold']}", inputs, trend[b0:b1],
                               b0, b1))
    # in sample: each fold's ensemble on its own fit window, the final one on the window
    fold_tilts_seed = {}
    for blk in blocks:
        j = blk["fold"]
        dirs = sorted(glob.glob(os.path.join(cf_root, "crossfit", f"fold{j}",
                                             "specialists", "seed*")))
        models = load_ensemble(dirs)
        d0, d1 = blk["fit_lo"], blk["fit_hi"] - 1
        rows.append(active_row("specialists", f"fit window fold {j} (in sample)", inputs,
                               specialist_tilts(models, inputs, d0, d1), d0, d1))
        b0, b1 = blk["fit_hi"], blk["fit_hi"] + int((stream["fold"] == j).sum())
        fold_tilts_seed[j] = [specialist_tilts([m], inputs, b0, b1) for m in models]
    final_dirs = sorted(glob.glob(os.path.join(fold, "models", "specialists", "seed*")))
    final = load_ensemble(final_dirs)
    rows.append(active_row("specialists", "fit window final (in sample)", inputs,
                           specialist_tilts(final, inputs, lo, hi - 1), lo, hi - 1))
    rows.append(active_row("trend rule", "fit window final", inputs, trend[lo:hi - 1],
                           lo, hi - 1))
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, "active_sharpe.csv"), index=False)

    # per-asset diagnostics on the stream days
    final_on_stream = specialist_tilts(final, inputs, s0, s1)
    seed_corr = []
    for j, per_seed in fold_tilts_seed.items():
        for x in range(len(per_seed)):
            for y in range(x + 1, len(per_seed)):
                seed_corr.append(corr_cols(per_seed[x], per_seed[y]))
    diag = pd.DataFrame({
        "mean_tilt": A.mean(axis=0),
        "mean_trend_tilt": trend[s0:s1].mean(axis=0),
        "share_abs_gt_095_pct": (np.abs(A) > 0.95).mean(axis=0) * 100,
        "lag1_autocorr": lag1(A),
        "corr_with_trend_tilt": corr_cols(A, trend[s0:s1]),
        "mean_corr_between_seeds": np.mean(seed_corr, axis=0) if seed_corr else np.nan,
        "corr_fold_vs_final": corr_cols(A, final_on_stream),
    }, index=pd.Index(inputs.tickers, name="ticker")).round(3)
    diag.to_csv(os.path.join(out, "spec_diagnostics.csv"))

    cur_final = curves(os.path.join(fold, "models", "specialists", "seed*", "diag.csv"))
    cur_fold = curves(os.path.join(cf_root, "crossfit", "fold*", "specialists", "seed*",
                                   "diag.csv"))
    tc = pd.concat({"final": cur_final, "folds": cur_fold}, axis=1)
    tc.to_csv(os.path.join(out, "training_curves.csv"))

    secs = []
    for f in glob.glob(os.path.join(fold, "**", "specialists", "seed*", "config.json"),
                       recursive=True):
        with open(f) as fh:
            c = json.load(fh)
        secs.append((c["steps"], c["train_seconds"]))
    secs = pd.DataFrame(secs, columns=["steps", "seconds"])
    timing = (secs.groupby("steps")["seconds"].agg(["count", "mean"]).round(0)
              .reset_index().to_dict("records"))

    caps_spec = cap_share(inputs, A, s0, s1)
    caps_trend = cap_share(inputs, trend[s0:s1], s0, s1)
    summary = {
        "test_year": Y, "settings": settings, "stream_root": cf_root,
        "stream_days": [str(inputs.dates[s0].date()), str(inputs.dates[s1 - 1].date()), len(A)],
        "active_sharpe": rows, "diagnostics": diag.reset_index().to_dict("records"),
        "cap_days_pct_on_stream": {"core_spec": dict(zip(["vol_cap", "lev_cap"], caps_spec)),
                                   "core_trend": dict(zip(["vol_cap", "lev_cap"], caps_trend))},
        "train_seconds_by_steps": timing,
    }
    with open(os.path.join(out, "m2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(f"Stream: {summary['stream_days']}, settings: {choice}")
    print("\nActive Sharpe against the core\n")
    print(table.to_string(index=False))
    print("\nSpecialist diagnostics on the cross-fit blocks\n")
    print(diag.to_string())
    print(f"\nCap days % on the stream: core_spec vol {caps_spec[0]}, lev {caps_spec[1]}; "
          f"core_trend vol {caps_trend[0]}, lev {caps_trend[1]}")
    print("\nTraining curves (fit-window active Sharpe, mean over seeds)\n")
    print(tc.to_string())
    print(f"\nTraining time: {timing}")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
