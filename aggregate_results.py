"""
R7 — pool walk-forward results and compute honest statistics.

Reads every <root>/fold*/seed*/phase2_returns.csv, then reports per strategy:

  * pooled OOS Sharpe (all folds chained chronologically, seed-averaged
    daily returns) with a stationary-bootstrap 95% CI
  * Probabilistic Sharpe Ratio vs 0 and Deflated Sharpe Ratio (trials =
    every strategy×seed pooled Sharpe that was produced by the sweep)
  * per-fold Sharpe table (mean over seeds) and per-seed pooled Sharpes
  * paired bootstrap p-values: meta_rl vs equal_weight and vs vol_target_bh

Outputs under <root>:
  walkforward_summary.csv    per_fold_sharpe.csv    pooled_returns.csv
  walkforward_report.txt

The charts for these numbers are in the dashboard:  python dashboard.py
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from common.stats import (ann_sharpe, bootstrap_sharpe_diff_pvalue,
                          deflated_sharpe, summary_row)


def load_cells(root):
    cells = {}
    for path in sorted(glob.glob(os.path.join(root, "fold*", "seed*",
                                              "phase2_returns.csv"))):
        m = re.search(r"fold(\d+)[/\\]seed(\d+)", path)
        if not m:
            continue
        fold, seed = int(m.group(1)), int(m.group(2))
        df = pd.read_csv(path, parse_dates=["date"])
        cells[(fold, seed)] = df
    return cells


def aggregate(cells, ann=252):
    """
    Pool a {(fold, seed): returns DataFrame} grid.  Returns a dict with:
    folds, seeds, strategies, pooled_mean (date × strategy, seed-averaged),
    summary, per_fold (strategy × fold Sharpe, seed mean), per_seed_sharpe
    ({strategy: {seed: Sharpe}}) and tests (paired bootstrap vs benchmarks).
    """
    folds = sorted({f for f, _ in cells})
    seeds = sorted({s for _, s in cells})
    strategies = [c for c in next(iter(cells.values())).columns if c != "date"]

    # ── per-seed pooled series (folds chained chronologically) ────────────
    pooled_by_seed = {}       # seed → DataFrame(date × strategy)
    for s in seeds:
        parts = [cells[(f, s)] for f in folds if (f, s) in cells]
        if parts:
            pooled_by_seed[s] = (pd.concat(parts)
                                 .sort_values("date")
                                 .set_index("date"))

    # seed-mean daily returns (benchmarks are identical across seeds)
    pooled_mean = (pd.concat(pooled_by_seed.values())
                   .groupby(level=0).mean().sort_index())

    # trials for the Deflated Sharpe: every strategy×seed pooled Sharpe
    trial_sharpes = []
    per_seed_sharpe = {}
    for name in strategies:
        per_seed_sharpe[name] = {}
        for s, df in pooled_by_seed.items():
            r = df[name].values
            sr = r.mean() / max(r.std(), 1e-12)     # per-period units
            per_seed_sharpe[name][s] = ann_sharpe(r, ann)
            trial_sharpes.append(sr)

    # ── headline summary ──────────────────────────────────────────────────
    rows = []
    for name in strategies:
        r = pooled_mean[name].values
        row = summary_row(r, ann=ann, name=name)
        row["dsr"] = round(deflated_sharpe(r, trial_sharpes, ann), 3)
        srs = list(per_seed_sharpe[name].values())
        row["seed_sharpe_mean"] = round(float(np.mean(srs)), 3)
        row["seed_sharpe_std"]  = round(float(np.std(srs)), 3)
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("strategy") \
                                .sort_values("sharpe", ascending=False)

    # ── per-fold Sharpe (mean over seeds) ─────────────────────────────────
    per_fold = {}
    for f in folds:
        for name in strategies:
            vals = [ann_sharpe(cells[(f, s)][name].values, ann)
                    for s in seeds if (f, s) in cells]
            per_fold.setdefault(name, {})[f] = round(float(np.mean(vals)), 3)
    per_fold_df = pd.DataFrame(per_fold).T

    # ── paired tests: meta_rl vs the two benchmarks that matter ───────────
    tests = []
    if "meta_rl" in strategies:
        for bench in ["meta_rl_leaked", "equal_weight", "vol_target_bh",
                      "buy_and_hold"]:
            if bench in strategies:
                d, pv = bootstrap_sharpe_diff_pvalue(
                    pooled_mean["meta_rl"].values,
                    pooled_mean[bench].values, ann)
                tests.append({"comparison": f"meta_rl − {bench}",
                              "sharpe_diff": round(d, 3),
                              "p_value_H0_diff<=0": round(pv, 4)})

    return {"folds": folds, "seeds": seeds, "strategies": strategies,
            "pooled_mean": pooled_mean, "summary": summary,
            "per_fold": per_fold_df, "per_seed_sharpe": per_seed_sharpe,
            "tests": pd.DataFrame(tests)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs/wf")
    p.add_argument("--ann",  type=int, default=252)
    args = p.parse_args()

    cells = load_cells(args.root)
    if not cells:
        print(f"No phase2_returns.csv found under {args.root}. "
              f"Run walk_forward.py first.")
        return

    res = aggregate(cells, args.ann)
    folds, seeds = res["folds"], res["seeds"]
    print(f"Loaded {len(cells)} cells: folds {folds} × seeds {seeds}")
    print(f"Strategies: {res['strategies']}\n")

    # ── report ────────────────────────────────────────────────────────────
    sep = "═" * 96
    lines = [sep, "  WALK-FORWARD RESULTS "
             f"({len(folds)} folds × {len(seeds)} seeds, pooled OOS)", sep,
             res["summary"].to_string(), "",
             "  Per-fold Sharpe (seed mean):", res["per_fold"].to_string(), ""]
    if len(res["tests"]):
        lines += ["  Paired stationary-bootstrap tests:",
                  res["tests"].to_string(index=False), ""]
    lines += [sep]
    report = "\n".join(lines)
    print(report)

    res["pooled_mean"].to_csv(os.path.join(args.root, "pooled_returns.csv"))
    res["summary"].to_csv(os.path.join(args.root, "walkforward_summary.csv"))
    res["per_fold"].to_csv(os.path.join(args.root, "per_fold_sharpe.csv"))
    with open(os.path.join(args.root, "walkforward_report.txt"), "w") as f:
        f.write(report + "\n")
    print("  Charts for these numbers:  python dashboard.py")


if __name__ == "__main__":
    main()
