"""
F9 — one command regenerates a full single-split run (phase 0 → 1 → 2 → eval)
from one set of arguments, so every CSV/model in an outdir comes from the SAME
configuration.

    python run_all.py --outdir split2024 \
        --train_start 2013-01-01 --train_end 2023-12-31 \
        --test_start 2024-01-01 --test_end 2024-12-31 --full
"""
import argparse
import os
import subprocess
import sys

import config


def sh(cmd):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir",      default="run")
    p.add_argument("--ticker",      default=config.TICKER)
    p.add_argument("--train_start", default=config.TRAIN_START)
    p.add_argument("--train_end",   default=config.TRAIN_END)
    p.add_argument("--test_start",  default=config.TEST_START)
    p.add_argument("--test_end",    default=config.TEST_END)
    p.add_argument("--seed",        type=int, default=config.SEED)
    p.add_argument("--cost_pct",    type=float, default=config.COST_PCT)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--full",  action="store_true")
    p.add_argument("--steps0", type=int, default=0)
    p.add_argument("--steps2", type=int, default=0)
    p.add_argument("--no_cash", action="store_true")
    p.add_argument("--no_hmm",  action="store_true")
    p.add_argument("--vol_target", type=float, default=config.VOL_TARGET)
    p.add_argument("--meta_reward", default=config.META_REWARD)
    p.add_argument("--data_csv", default=config.DATA_CSV)
    args = p.parse_args()

    py = sys.executable
    os.makedirs(args.outdir, exist_ok=True)
    base = ["--data_csv", args.data_csv, "--ticker", args.ticker]
    size = (["--quick"] if args.quick else []) + \
           (["--full"] if args.full else [])

    p0 = [py, "train_agents.py", "--start", args.train_start,
          "--end", args.train_end, "--outdir", args.outdir,
          "--seed", str(args.seed), "--cost_pct", str(args.cost_pct)] \
          + size + base
    if args.steps0:
        p0 += ["--steps", str(args.steps0)]
    sh(p0)

    sh([py, "train_phase1.py", "--train_start", args.train_start,
        "--train_end", args.train_end, "--outdir", args.outdir,
        "--seed", str(args.seed)] + base)

    p2 = [py, "train_phase2.py", "--train_start", args.train_start,
          "--train_end", args.train_end, "--modeldir", args.outdir,
          "--outdir", args.outdir, "--seed", str(args.seed),
          "--cost_pct", str(args.cost_pct),
          "--vol_target", str(args.vol_target),
          "--meta_reward", args.meta_reward] + size + base
    if args.steps2:
        p2 += ["--steps", str(args.steps2)]
    if args.no_cash:
        p2.append("--no_cash")
    if args.no_hmm:
        p2.append("--no_hmm")
    sh(p2)

    sh([py, "evaluate_phase2.py", "--test_start", args.test_start,
        "--test_end", args.test_end, "--modeldir", args.outdir,
        "--outdir", args.outdir] + base)

    print(f"\n✓ Full pipeline complete → {args.outdir}")


if __name__ == "__main__":
    main()
