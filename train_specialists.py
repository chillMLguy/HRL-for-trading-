"""
Fit a seed ensemble of asset specialists on one fit window (brief section 6).

    python train_specialists.py --outdir runs/multi_wf/fold2015 --test_year 2015
    python train_specialists.py --outdir runs/x --fit_start 2007-04-18 --fit_end 2012-01-01 \\
        --seed_base 2015010

Each seed is one SAC network shared by the eight assets (mode shared_id adds the
asset's one-hot to the observation).  The final model of a fixed step budget is kept;
no checkpoint is ever chosen.  The specialist is the ensemble of seeds, combined in
multi.spec_env.specialist_tilts.  Every tenth of training the single seed's active
Sharpe on its own fit window is logged to diag.csv, for diagnostics only.

Seeds run in parallel worker processes (OMP_NUM_THREADS=1 each).  A seed whose
model.zip and config.json exist is skipped, so the script is resumable.

Layout:  <model_root>/seed<k>/model.zip, config.json, diag.csv
"""
import argparse
import json
import multiprocessing as mp
import os
import time
import warnings

import numpy as np

import config


def sac_kwargs(steps):
    return dict(config.MULTI_SAC_KWARGS,
                buffer_size=min(config.MULTI_BUFFER_MAX, steps),
                policy_kwargs=dict(net_arch=list(config.MULTI_NET_ARCH),
                                   optimizer_kwargs=dict(
                                       weight_decay=config.MULTI_WEIGHT_DECAY)))


def fit_window_active_sharpe(models, inputs, fit_lo, fit_hi, mode, feats=None):
    """Active Sharpe of core_spec against core on the fit window's decisions."""
    from multi.portfolio import active_sharpe, build_positions, net_returns, tilt_returns
    from multi.spec_env import specialist_tilts
    d_lo, d_hi = fit_lo, fit_hi - 1
    tilts = specialist_tilts(models, inputs, d_lo, d_hi, mode=mode, feats=feats)
    r_spec = tilt_returns(inputs, tilts, d_lo, d_hi)["r"]
    r_core = net_returns(build_positions(inputs.vol, inputs.cov), inputs.rex,
                         d_lo, d_hi)["r"]
    return active_sharpe(r_spec, r_core), tilts


def _train_seed(job):
    """One seed, in its own process (top level so multiprocessing can pickle it)."""
    os.environ["OMP_NUM_THREADS"] = "1"
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from multi.spec_env import SpecialistEnv

    inputs, cfg, seed_dir = job["inputs"], job["cfg"], job["seed_dir"]
    fit_lo, fit_hi, steps, mode = cfg["fit_lo"], cfg["fit_hi"], cfg["steps"], cfg["mode"]
    env = SpecialistEnv(inputs, fit_lo, fit_hi, mode=mode,
                        spec_lambda=cfg["spec_lambda"], reward_scale=cfg["reward_scale"],
                        seed=cfg["seed"])
    model = SAC("MlpPolicy", env, seed=cfg["seed"], device="cpu", verbose=0,
                **sac_kwargs(steps))

    class Diag(BaseCallback):
        def __init__(self):
            super().__init__()
            self.every = max(1, steps // config.MULTI_DIAG_POINTS)
            self.rows = []

        def _on_step(self):
            if self.num_timesteps % self.every == 0:
                sr, tilts = fit_window_active_sharpe([self.model], inputs, fit_lo,
                                                     fit_hi, mode)
                self.rows.append((self.num_timesteps, sr, float(np.abs(tilts).mean())))
            return True

    diag = Diag()
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=diag)
    seconds = time.time() - t0
    os.makedirs(seed_dir, exist_ok=True)
    model.save(os.path.join(seed_dir, "model.zip"))
    with open(os.path.join(seed_dir, "diag.csv"), "w") as f:
        f.write("step,fit_active_sharpe,mean_abs_tilt\n")
        for st, sr, mt in diag.rows:
            f.write(f"{st},{sr:.4f},{mt:.4f}\n")
    cfg = dict(cfg, train_seconds=round(seconds, 1),
               mu_s=[float(v) for v in env.mu_s])
    with open(os.path.join(seed_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return seed_dir, seconds


def seed_config(inputs, fit_lo, fit_hi, seed, steps, mode, spec_lambda,
                reward_scale=None):
    d = inputs.dates
    return {
        "layer": "specialist", "mode": mode, "seed": int(seed), "steps": int(steps),
        "spec_lambda": float(spec_lambda),
        "reward_scale": float(config.SPEC_REWARD_SCALE if reward_scale is None
                              else reward_scale),
        "fit_lo": int(fit_lo), "fit_hi": int(fit_hi),
        "fit_first_decision": str(d[fit_lo].date()),
        "fit_last_decision": str(d[fit_hi - 2].date()),
        "fit_last_data_bar": str(d[fit_hi - 1].date()),
        "episode_bars": config.SPEC_EPISODE_BARS, "cost_pct": config.COST_PCT,
        "sac": {k: v for k, v in sac_kwargs(steps).items() if k != "policy_kwargs"},
        "net_arch": list(config.MULTI_NET_ARCH),
        "weight_decay": config.MULTI_WEIGHT_DECAY,
        "features": "market features 0-11 + own previous tilt"
                    + (" + asset one-hot" if mode == "shared_id" else ""),
        "tickers": list(inputs.tickers),
    }


_KEYS = ("mode", "seed", "steps", "spec_lambda", "reward_scale", "fit_lo", "fit_hi")


def train_ensemble(inputs, fit_lo, fit_hi, model_root, seeds, steps, mode=None,
                   spec_lambda=None, reward_scale=None, workers=2):
    """
    Train (or reuse) one model per seed in `seeds` on the fit window [fit_lo, fit_hi).
    Returns the seed directories, in seed order.
    """
    specs = [(fit_lo, fit_hi, seed, os.path.join(model_root, f"seed{k}"))
             for k, seed in enumerate(seeds)]
    return train_jobs(inputs, specs, steps, mode, spec_lambda, reward_scale, workers)


def train_jobs(inputs, specs, steps, mode=None, spec_lambda=None, reward_scale=None,
               workers=2):
    """
    Train (or reuse) one model per (fit_lo, fit_hi, seed, seed_dir) in `specs`, all in
    one worker pool, so that several fit windows share the cores.  Returns the dirs.
    """
    mode = mode or config.SPEC_MODE
    lam = config.SPEC_LAMBDA if spec_lambda is None else spec_lambda
    jobs, dirs = [], []
    for fit_lo, fit_hi, seed, seed_dir in specs:
        dirs.append(seed_dir)
        cfg = seed_config(inputs, fit_lo, fit_hi, seed, steps, mode, lam, reward_scale)
        done = os.path.join(seed_dir, "config.json")
        if os.path.isfile(os.path.join(seed_dir, "model.zip")) and os.path.isfile(done):
            with open(done) as f:
                old = json.load(f)
            if any(old.get(key) != cfg[key] for key in _KEYS):
                raise SystemExit(f"{seed_dir} holds a model trained with other settings; "
                                 "use another output folder.")
            continue
        jobs.append({"inputs": inputs, "cfg": cfg, "seed_dir": seed_dir})
    if jobs:
        print(f"  training {len(jobs)} of {len(specs)} models × {steps:,} steps "
              f"in {min(workers, len(jobs))} workers", flush=True)
        if workers > 1 and len(jobs) > 1:
            with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
                for seed_dir, sec in pool.imap_unordered(_train_seed, jobs):
                    print(f"    {seed_dir}: {sec / 60:.1f} min", flush=True)
        else:
            for job in jobs:
                seed_dir, sec = _train_seed(job)
                print(f"    {seed_dir}: {sec / 60:.1f} min", flush=True)
    return dirs


def main():
    from multi.data import load_universe, training_window
    from multi.features import compute_inputs

    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--test_year", type=int, default=None,
                   help="fit on the whole training window of this test year")
    p.add_argument("--fit_start", default=None)
    p.add_argument("--fit_end", default=None, help="exclusive")
    p.add_argument("--seed_base", type=int, default=None,
                   help="seed of model k = seed_base + k (default 1000·Y)")
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--spec_mode", default=config.SPEC_MODE)
    p.add_argument("--spec_lambda", type=float, default=config.SPEC_LAMBDA)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    steps = args.steps or (config.SPEC_STEPS_QUICK if args.quick else config.SPEC_STEPS)
    n_seeds = args.seeds or (config.SPEC_SEEDS_QUICK if args.quick else config.SPEC_SEEDS)
    end_year = args.test_year or int(args.fit_end[:4]) + 1
    u = load_universe(config.MULTI_DOWNLOAD_START, f"{end_year}-01-01")
    inputs = compute_inputs(u)
    if args.test_year:
        fit_lo, fit_hi = training_window(u.dates, args.test_year, config.MULTI_WARM_BARS)
        seed_base = args.seed_base if args.seed_base is not None else 1000 * args.test_year
    else:
        import pandas as pd
        fit_lo = max(int(u.dates.searchsorted(pd.Timestamp(args.fit_start))),
                     config.MULTI_WARM_BARS)
        fit_hi = int(u.dates.searchsorted(pd.Timestamp(args.fit_end)))
        seed_base = args.seed_base or 0
    root = os.path.join(args.outdir, "models", "specialists")
    dirs = train_ensemble(inputs, fit_lo, fit_hi, root,
                          [seed_base + k for k in range(n_seeds)], steps,
                          args.spec_mode, args.spec_lambda, workers=args.workers)
    from multi.spec_env import load_ensemble
    sr, _ = fit_window_active_sharpe(load_ensemble(dirs), inputs, fit_lo, fit_hi,
                                     args.spec_mode)
    print(f"Ensemble of {len(dirs)}: fit-window active Sharpe {sr:.3f}")


if __name__ == "__main__":
    main()
