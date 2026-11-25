"""
How much of a cross-fitted result is training noise?

One run trains one cross-fitted and one leaked meta-agent, so a single seed decides
the comparison.  This holds the agents, the HMM and the proposal stream of a finished
run fixed and retrains only the two meta-agents across seeds, then scores them all
on the run's test window.

    python run_all.py --outdir runs/crossfit_2024 --crossfit
    python meta_seed_study.py --rundir runs/crossfit_2024 --seeds 0 1 2 3

Each new meta-agent copies the network, weight decay and number of training steps
of the run's own meta-agent of the same kind (from its config.json), so only the
seed differs.  Meta-agents go to <rundir>/meta_seeds/, and finished ones are
skipped.  Writes <rundir>/meta_seed_study.csv, which includes the
run's own two meta-agents.
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

from common.data import load_prices, warm_start_date, trading_start_index, check_vix_settings
from common.stats import ann_sharpe
from env.meta_env import load_subagents, meta_env_from_config
from regime.hmm_regime import HMMRegimeDetector
import config


def score(meta_path, warm, first, subagents, hmm, ann):
    """Test Sharpe and exposure of the meta-agent saved in meta_path."""
    with open(os.path.join(meta_path, "config.json")) as f:
        cfg = json.load(f)
    check_vix_settings(cfg, meta_path)
    meta = SAC.load(os.path.join(meta_path, "best_model"))
    env = meta_env_from_config(warm, subagents, hmm if cfg.get("use_hmm", True) else None,
                               cfg, cfg["cost_pct"], ann=ann, start_index=first,
                               eval_mode=True)
    obs, _ = env.reset()
    rets, pos, done = [], [], False
    while not done:
        action, _ = meta.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        rets.append(info["net_ret"])
        pos.append(info["blended"])
    rets, pos = np.array(rets), np.array(pos)
    return {"val_sharpe": round(cfg["best_val_sharpe"], 3),
            "test_sharpe": round(ann_sharpe(rets, ann), 3),
            "mean_abs_position": round(float(np.abs(pos).mean()), 3),
            "total_return_pct": round(float(np.prod(1.0 + rets) - 1.0) * 100, 2)}


def main():
    p = argparse.ArgumentParser(
        description="Retrain only the meta-agents of a cross-fitted run across seeds")
    p.add_argument("--rundir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--steps", type=int, default=0,
                   help="Override the run's number of meta-agent training steps")
    p.add_argument("--data_csv", default=config.DATA_CSV)
    args = p.parse_args()

    primary = os.path.join(args.rundir, "models", "phase2")
    with open(os.path.join(primary, "config.json")) as f:
        cfg = json.load(f)
    if cfg.get("training_mode") != "crossfit":
        raise SystemExit(f"{args.rundir} is not a cross-fitted run.")
    with open(os.path.join(args.rundir, "phase2_eval_config.json")) as f:
        eval_cfg = json.load(f)

    check_vix_settings(cfg, primary)      # before hours of training, not after
    kinds = {"crossfit": ("--crossfit", cfg)}
    leaked_cfg_path = os.path.join(args.rundir, "models", "phase2_leaked", "config.json")
    if os.path.isfile(leaked_cfg_path):
        with open(leaked_cfg_path) as f:
            kinds["leaked"] = ("--stream_dates", json.load(f))
    out = os.path.join(args.rundir, "meta_seeds")
    command = [sys.executable, "train_phase2.py", "--train_start", cfg["train_start"],
               "--train_end", cfg["train_end"], "--modeldir", args.rundir, "--outdir", out,
               "--ticker", cfg["ticker"], "--cost_pct", str(cfg["cost_pct"]),
               "--data_csv", args.data_csv]

    for seed in args.seeds:
        for kind, (flag, kcfg) in kinds.items():
            name = f"{kind}_{seed}"
            if os.path.isfile(os.path.join(out, "models", name, "config.json")):
                continue
            # Same network, weight decay and steps as the run's own meta-agent of
            # this kind; runs from before R19 used plain Adam (weight decay 0).
            arch = ",".join(str(h) for h in kcfg["net_arch"]) or "linear"
            same = ["--net_arch", arch,
                    "--weight_decay", str(kcfg.get("weight_decay", 0.0)),
                    "--steps", str(args.steps or kcfg["timesteps"])]
            print(f"Training {name} ...", flush=True)
            subprocess.run(command + same + ["--seed", str(seed), flag,
                                             "--meta_dir", name],
                           check=True, stdout=subprocess.DEVNULL)

    ann = cfg.get("bars_per_year", 252)
    warm = load_prices(cfg["ticker"],
                       warm_start_date(eval_cfg["test_start"], config.WARM_BARS),
                       eval_cfg["test_end"], cfg.get("interval", config.INTERVAL),
                       data_csv=args.data_csv, verbose=False)
    first = trading_start_index(warm, eval_cfg["test_start"])
    subagents = load_subagents(args.rundir, verbose=False)
    hmm = HMMRegimeDetector.load(os.path.join(args.rundir, "models", "phase1",
                                              "hmm_model.pkl"))

    targets = [(cfg["seed"], "crossfit", primary)]
    if "leaked" in kinds:
        targets.append((cfg["seed"], "leaked",
                        os.path.join(args.rundir, "models", "phase2_leaked")))
    targets += [(s, k, os.path.join(out, "models", f"{k}_{s}"))
                for s in args.seeds for k in kinds]
    rows = [dict(seed=s, kind=k, **score(path, warm, first, subagents, hmm, ann))
            for s, k, path in targets]
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.rundir, "meta_seed_study.csv"), index=False)

    wide = table.pivot(index="seed", columns="kind",
                       values=["val_sharpe", "test_sharpe", "mean_abs_position"])
    print(wide.to_string())
    print(table.groupby("kind")[["val_sharpe", "test_sharpe"]]
          .agg(["mean", "std"]).round(2).to_string())
    tests = table.pivot(index="seed", columns="kind", values="test_sharpe")
    if "leaked" in tests:
        gap = tests["crossfit"] - tests["leaked"]
        print(f"\nCross-fitted minus leaked test Sharpe: mean {gap.mean():+.2f}, "
              f"sd {gap.std():.2f}, cross-fitted higher in {(gap > 0).sum()} "
              f"of {len(gap)}")
    print("All seeds share one test path, so this measures training noise only.")


if __name__ == "__main__":
    main()
