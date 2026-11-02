"""
R7 — walk-forward protocol: rolling annual folds × multiple seeds.

For each test year Y in config.WF_TEST_YEARS and each seed S in
config.WF_SEEDS:

    train window : [Y − WF_TRAIN_YEARS, Y − 1]   (sliding, not expanding)
    test window  : year Y  (traded in full thanks to warm windows, F8)
    output dir   : <outroot>/fold<Y>/seed<S>/

runs phase 0 → 1 → 2 → evaluate_phase2 through the standard CLI scripts, so a
fold/seed cell is exactly one reproducible pipeline run.  Already-completed
cells (phase2_returns.csv present) are skipped, so the sweep is resumable.

Afterwards:   python aggregate_results.py --root <outroot>
              python dashboard.py

Example (full sweep):
    python download_data.py
    python walk_forward.py --outroot runs/wf --steps0 200000 --steps2 150000
Quick sanity sweep:
    python walk_forward.py --outroot runs/wf_quick --steps0 30000 \
        --steps2 30000 --years 2022 2024 --seeds 0
"""
import argparse
import os
import subprocess
import sys
import time

import config


def sh(cmd):
    print("  $", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outroot", default="runs/wf")
    p.add_argument("--ticker",  default=config.TICKER)
    p.add_argument("--years",   type=int, nargs="+",
                   default=config.WF_TEST_YEARS)
    p.add_argument("--seeds",   type=int, nargs="+", default=config.WF_SEEDS)
    p.add_argument("--train_years", type=int, default=config.WF_TRAIN_YEARS)
    p.add_argument("--steps0",  type=int, default=200_000,
                   help="Phase 0 timesteps per agent")
    p.add_argument("--steps2",  type=int, default=150_000,
                   help="Phase 2 timesteps")
    p.add_argument("--workers", type=int, default=3,
                   help="Parallel Phase 0 agent trainings per cell")
    p.add_argument("--cost_pct", type=float, default=config.COST_PCT)
    p.add_argument("--data_csv", default=config.DATA_CSV)
    p.add_argument("--no_cnn",  action="store_true", default=True,
                   help="CNN off by default in the sweep")
    # Meta ablation passthroughs
    p.add_argument("--no_cash", action="store_true")
    p.add_argument("--no_hmm",  action="store_true")
    p.add_argument("--vol_target", type=float, default=config.VOL_TARGET)
    p.add_argument("--meta_reward", default=config.META_REWARD,
                   choices=["dsr", "composite"])
    args = p.parse_args()

    py = sys.executable
    t_start = time.time()
    cells = [(y, s) for y in args.years for s in args.seeds]
    print(f"Walk-forward: {len(cells)} cells "
          f"({len(args.years)} folds × {len(args.seeds)} seeds)")

    for k, (year, seed) in enumerate(cells, 1):
        d = os.path.join(args.outroot, f"fold{year}", f"seed{seed}")
        os.makedirs(d, exist_ok=True)
        if os.path.isfile(os.path.join(d, "phase2_returns.csv")):
            print(f"[{k}/{len(cells)}] fold{year}/seed{seed} — done, skip")
            continue

        tr_start = f"{year - args.train_years}-01-01"
        tr_end   = f"{year - 1}-12-31"
        te_start = f"{year}-01-01"
        te_end   = f"{year}-12-31"
        print(f"\n[{k}/{len(cells)}] fold{year}/seed{seed}   "
              f"train {tr_start}→{tr_end}  test {te_start}→{te_end}")

        base = ["--data_csv", args.data_csv, "--ticker", args.ticker]

        sh([py, "train_agents.py", "--start", tr_start, "--end", tr_end,
            "--outdir", d, "--seed", str(seed), "--steps", str(args.steps0),
            "--workers", str(args.workers), "--cost_pct", str(args.cost_pct),
            "--no_cnn"] + base)

        sh([py, "train_phase1.py", "--train_start", tr_start,
            "--train_end", tr_end, "--outdir", d, "--seed", str(seed)]
           + base)

        p2 = [py, "train_phase2.py", "--train_start", tr_start,
              "--train_end", tr_end, "--modeldir", d, "--outdir", d,
              "--seed", str(seed), "--steps", str(args.steps2),
              "--cost_pct", str(args.cost_pct), "--no_cnn",
              "--vol_target", str(args.vol_target),
              "--meta_reward", args.meta_reward] + base
        if args.no_cash:
            p2.append("--no_cash")
        if args.no_hmm:
            p2.append("--no_hmm")
        sh(p2)

        sh([py, "evaluate_phase2.py", "--test_start", te_start,
            "--test_end", te_end, "--modeldir", d, "--outdir", d,
            "--no_cnn"] + base)

    print(f"\n✓ Walk-forward complete in {(time.time()-t_start)/60:.1f} min.")
    print(f"  Aggregate with:  python aggregate_results.py --root "
          f"{args.outroot}")


if __name__ == "__main__":
    main()
