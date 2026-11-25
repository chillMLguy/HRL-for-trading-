"""
R7 — walk-forward protocol: rolling annual folds × multiple seeds.

For each test year Y in config.WF_TEST_YEARS and each seed S in
config.WF_SEEDS:

    train window : [WF_TRAIN_START, Y − 1]        (expanding, the default)
                   [Y − N, Y − 1] with --train_years N (sliding)
    test window  : year Y  (traded in full thanks to warm windows, F8)
    output dir   : <outroot>/fold<Y>/seed<S>/

runs phase 0 → 1 → 2 → evaluate_phase2 through the standard CLI scripts, so a
fold/seed cell is exactly one reproducible pipeline run.  Already-completed
cells (phase2_returns.csv present) are skipped, so the sweep is resumable.

Afterwards:   python aggregate_results.py --root <outroot>
              python dashboard.py

Example (full sweep):
    python download_data.py
    python walk_forward.py --outroot runs/wf --crossfit
Pipeline check (3 passes, mechanics only):
    python walk_forward.py --outroot runs/wf_check --quick --years 2022 2024 --seeds 0

Network size, weight decay and training length take the same flags as run_all.py
(--net_arch, --meta_net_arch, --weight_decay, --passes0/2, --steps0/2, --quick).
"""
import argparse
import os
import subprocess
import sys
import time

import config
from common.training import forward_args, parse_net_arch


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
    p.add_argument("--train_start", default=config.WF_TRAIN_START,
                   help="Start of every fold's (expanding) training window")
    p.add_argument("--train_years", type=int, default=config.WF_TRAIN_YEARS,
                   help="N > 0: slide, training on the N years before each test "
                        "year instead (0 = expanding from --train_start)")
    p.add_argument("--net_arch", type=parse_net_arch, default=config.AGENT_NET_ARCH,
                   help="Agents' hidden layers, e.g. 32,32, or 'linear'")
    p.add_argument("--meta_net_arch", type=parse_net_arch,
                   default=config.META_NET_ARCH, help="Meta-agent's hidden layers")
    p.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    p.add_argument("--passes0", type=float, default=config.AGENT_PASSES,
                   help="Agent steps = passes × agent training bars")
    p.add_argument("--passes2", type=float, default=config.META_PASSES,
                   help="Meta steps = passes × meta training bars")
    p.add_argument("--steps0", type=int, default=0, help="Fixed agent steps")
    p.add_argument("--steps2", type=int, default=0, help="Fixed meta steps")
    p.add_argument("--quick", action="store_true",
                   help=f"{config.QUICK_PASSES} passes: a pipeline check, not a result")
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
    # Cross-fitted meta training
    p.add_argument("--crossfit", action="store_true",
                   help="Train the meta-agent on out-of-sample agent behaviour")
    p.add_argument("--cf_folds", type=int, default=config.CF_FOLDS)
    p.add_argument("--no_compare_leaked", action="store_true",
                   help="With --crossfit, skip the in-sample comparison meta-agent")
    p.add_argument("--no_vix", action="store_true",
                   help="Ablation: VIX inputs set to zero (same observation size)")
    p.add_argument("--vix_lag", type=int, default=None,
                   help=f"Days VIX lags the asset (default {config.VIX_LAG_DAYS})")
    args = p.parse_args()
    # Every script started below reads these in config.py, so the whole run uses
    # one VIX setting; each model's config.json records it and evaluation checks it.
    if args.no_vix:
        os.environ["HRL_USE_VIX"] = "0"
    if args.vix_lag is not None:
        os.environ["HRL_VIX_LAG"] = str(args.vix_lag)

    size0 = forward_args(argparse.Namespace(
        net_arch=args.net_arch, weight_decay=args.weight_decay, passes=args.passes0,
        steps=args.steps0, quick=args.quick))
    size2 = forward_args(argparse.Namespace(
        net_arch=args.meta_net_arch, weight_decay=args.weight_decay,
        passes=args.passes2, steps=args.steps2, quick=args.quick))

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

        tr_start = (f"{year - args.train_years}-01-01" if args.train_years > 0
                    else args.train_start)
        tr_end   = f"{year - 1}-12-31"
        te_start = f"{year}-01-01"
        te_end   = f"{year}-12-31"
        print(f"\n[{k}/{len(cells)}] fold{year}/seed{seed}   "
              f"train {tr_start}→{tr_end}  test {te_start}→{te_end}")

        base = ["--data_csv", args.data_csv, "--ticker", args.ticker]

        sh([py, "train_agents.py", "--start", tr_start, "--end", tr_end,
            "--outdir", d, "--seed", str(seed),
            "--workers", str(args.workers), "--cost_pct", str(args.cost_pct),
            "--no_cnn"] + size0 + base)

        sh([py, "train_phase1.py", "--train_start", tr_start,
            "--train_end", tr_end, "--outdir", d, "--seed", str(seed)]
           + base)

        if args.crossfit:
            sh([py, "train_crossfit.py", "--train_start", tr_start,
                "--train_end", tr_end, "--outdir", d, "--seed", str(seed),
                "--workers", str(args.workers),
                "--cost_pct", str(args.cost_pct), "--folds", str(args.cf_folds)]
               + size0 + base)

        p2 = [py, "train_phase2.py", "--train_start", tr_start,
              "--train_end", tr_end, "--modeldir", d, "--outdir", d,
              "--seed", str(seed),
              "--cost_pct", str(args.cost_pct), "--no_cnn",
              "--vol_target", str(args.vol_target),
              "--meta_reward", args.meta_reward] + size2 + base
        if args.no_cash:
            p2.append("--no_cash")
        if args.no_hmm:
            p2.append("--no_hmm")
        if args.crossfit:
            sh(p2 + ["--crossfit"])
            if not args.no_compare_leaked:
                sh(p2 + ["--stream_dates", "--meta_dir", "phase2_leaked"])
        else:
            sh(p2)

        sh([py, "evaluate_phase2.py", "--test_start", te_start,
            "--test_end", te_end, "--modeldir", d, "--outdir", d,
            "--no_cnn"] + base)

        sh([py, "overfit_check.py", "--rundir", d, "--data_csv", args.data_csv])

    print(f"\n✓ Walk-forward complete in {(time.time()-t_start)/60:.1f} min.")
    print(f"  Aggregate with:  python aggregate_results.py --root "
          f"{args.outroot}")


if __name__ == "__main__":
    main()
