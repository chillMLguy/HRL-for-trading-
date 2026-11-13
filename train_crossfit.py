"""
Build the out-of-sample proposal stream for cross-fitted meta training.

For each block at the end of the training window (see common/crossfit.py):

  1. train the three agents on the bars before the block   (train_agents.py)
  2. fit the HMM on the same bars                          (train_phase1.py)
  3. roll both over the block and record what they propose

Outputs under <outdir>/crossfit/:
  fold<k>/        models and proposals of each fold; finished folds are skipped
  proposals.csv   the joined stream that train_phase2.py --crossfit reads
  config.json     the block dates and settings

This costs one extra Phase 0 per fold.  run_all.py --crossfit runs it for you.
"""
import argparse
import json
import os
import subprocess
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from common.crossfit import fold_edges, stream_path
from common.data import load_prices
from env.trading_env import TradingEnv
from regime.allocators import AGENT_ORDER
from regime.hmm_regime import HMMRegimeDetector
import config


def sh(cmd):
    print("  $", " ".join(cmd), flush=True)
    if subprocess.run(cmd).returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}")


def propose_block(fold_dir, prices, start, end, cost_pct, ann):
    """What fold_dir's agents and HMM propose on bars [start, end) of `prices`."""
    lo = max(0, start - config.WARM_BARS)
    warm = prices.iloc[lo: min(end + 1, len(prices))]
    first = start - lo
    actions = {}
    for name in AGENT_ORDER:
        model = SAC.load(os.path.join(fold_dir, "models", f"{name}_agent",
                                      "best_model"))
        env = TradingEnv(warm, preset=name, bars_per_year=ann, cost_pct=cost_pct,
                         eval_mode=True, start_index=first)
        obs, _ = env.reset()
        rows, done = [], False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            rows.append(float(np.clip(np.asarray(action).ravel()[0], -1.0, 1.0)))
            obs, _, done, _, _ = env.step(action)
        actions[f"a_{name}"] = rows
    n = len(rows)

    hmm = HMMRegimeDetector.load(os.path.join(fold_dir, "models", "phase1",
                                              "hmm_model.pkl"))
    hmm_obs, valid_start = HMMRegimeDetector.build_observations(warm)
    posterior = hmm.filtered_proba(hmm_obs)[first - valid_start: first - valid_start + n]

    frame = pd.DataFrame(actions, index=pd.Index(warm.index[first: first + n],
                                                 name="date"))
    for k in range(posterior.shape[1]):
        frame[f"hmm_{k}"] = posterior[:, k]
    return frame


def main():
    p = argparse.ArgumentParser(
        description="Build the out-of-sample proposal stream for the meta-agent")
    p.add_argument("--outdir",      required=True)
    p.add_argument("--ticker",      default=config.TICKER)
    p.add_argument("--train_start", default=config.TRAIN_START)
    p.add_argument("--train_end",   default=config.TRAIN_END)
    p.add_argument("--interval",    default=config.INTERVAL,
                   choices=list(config.BARS_PER_YEAR.keys()))
    p.add_argument("--folds",       type=int, default=config.CF_FOLDS)
    p.add_argument("--min_train_frac", type=float, default=config.CF_MIN_TRAIN_FRAC,
                   help="Share of the window that fold 1 fits on")
    p.add_argument("--seed",        type=int, default=config.SEED)
    p.add_argument("--cost_pct",    type=float, default=config.COST_PCT)
    p.add_argument("--workers",     type=int, default=3)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--full",  action="store_true")
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--data_csv", default=config.DATA_CSV)
    args = p.parse_args()

    if os.path.isfile(os.path.join(args.outdir, "models", "cnn_features",
                                   "cnn_model.pt")):
        raise SystemExit("Cross-fitting does not support CNN features yet: the "
                         "fold agents would not see them. Use a run folder "
                         "without models/cnn_features.")

    ann = config.BARS_PER_YEAR[args.interval]
    prices = load_prices(args.ticker, args.train_start, args.train_end,
                         args.interval, data_csv=args.data_csv)
    edges = fold_edges(len(prices), args.folds, args.min_train_frac)
    root = os.path.join(args.outdir, "crossfit")
    os.makedirs(root, exist_ok=True)

    py = sys.executable
    size = (["--quick"] if args.quick else []) + (["--full"] if args.full else []) \
        + (["--steps", str(args.steps)] if args.steps else [])
    base = ["--ticker", args.ticker, "--interval", args.interval,
            "--data_csv", args.data_csv]

    blocks, parts = [], []
    for k, (start, end) in enumerate(edges, start=1):
        fold_dir = os.path.join(root, f"fold{k}")
        part_path = os.path.join(fold_dir, "proposals.csv")
        # The fold fits on [train_start, block start): load_prices excludes `end`
        fit_end = prices.index[start].strftime("%Y-%m-%d")
        last = prices.index[end - 1].strftime("%Y-%m-%d")
        blocks.append({"fold": k, "fit_start": args.train_start, "fit_end": fit_end,
                       "block_start": fit_end, "block_last": last})
        print(f"\n=== Fold {k}/{len(edges)}: fit {args.train_start} to {fit_end}, "
              f"propose {fit_end} to {last} ===", flush=True)

        if os.path.isfile(part_path):
            print("  already finished, skipping")
        else:
            fold_seed = args.seed + 1000 * k
            sh([py, "train_agents.py", "--start", args.train_start, "--end", fit_end,
                "--outdir", fold_dir, "--seed", str(fold_seed), "--no_cnn",
                "--workers", str(args.workers), "--cost_pct", str(args.cost_pct)]
               + size + base)
            sh([py, "train_phase1.py", "--train_start", args.train_start,
                "--train_end", fit_end, "--outdir", fold_dir,
                "--seed", str(fold_seed), "--cost_pct", str(args.cost_pct)] + base)
            propose_block(fold_dir, prices, start, end, args.cost_pct,
                          ann).to_csv(part_path)

        part = pd.read_csv(part_path, parse_dates=["date"], index_col="date")
        part.insert(0, "fold", k)
        parts.append(part)

    pd.concat(parts).to_csv(stream_path(args.outdir))
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump({"folds": args.folds, "min_train_frac": args.min_train_frac,
                   "train_start": args.train_start, "train_end": args.train_end,
                   "cost_pct": args.cost_pct, "seed": args.seed,
                   "agent_order": AGENT_ORDER, "blocks": blocks}, f, indent=2)
    print(f"\n✓ Proposal stream → {stream_path(args.outdir)} "
          f"({sum(len(x) for x in parts)} bars)")


if __name__ == "__main__":
    main()
