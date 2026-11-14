"""
Train the allocators of one test year on its cross-fit stream (brief section 7).

    python train_allocator.py --outdir runs/multi_wf/fold2015 --test_year 2015

Two allocators are trained on the same days, the stream's out-of-sample behaviour:

    full        sees the specialists' tilts from the stream (m = 1 + tilt)
    core_alloc  sees m = 1 (no specialists)

All cross-fit blocks are used and the final model of a fixed step budget is kept, so
no validation block is needed.  Each allocator is a seed ensemble; inference takes the
mean of the seeds' raw actions before the mapping to beta and G
(multi.alloc_env.run_allocator).  The stream never contains tilts from specialists
fitted on the same days (train_crossfit_multi.py).

`crossfit_scores` gives each allocator an out-of-sample score inside the training
window: each block in turn is held out, the allocator is trained on the other blocks
(skipping the ALLOC_PURGE_AFTER days after the held-out block, whose observations
contain its returns), and it is run on the held-out block.  search_multi.py --layer 2
uses this score; the final allocators are trained on every block.

Layout:  <fold>/models/allocator_<name>/seed<k>/model.zip, config.json
         <run>/alloc_crossfit/holdout<j>/allocator_<name>/seed<k>/...
Resumable: a seed whose model.zip and config.json exist is skipped.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
import warnings

import numpy as np
import pandas as pd

import config

NAMES = ("full", "core_alloc")


def stream_arrays(inputs, stream):
    """(s0, s1, m_full, hmm): stream decision bars [s0, s1) and full-length arrays."""
    from train_crossfit_multi import stream_tilts
    s0, tilts = stream_tilts(stream, inputs)
    s1 = s0 + len(tilts)
    T, N = inputs.vol.shape
    m = np.ones((T, N))
    m[s0:s1] = 1.0 + tilts
    K = config.N_STATES
    hmm = np.full((T, K), np.nan)
    hmm[s0:s1] = stream[[f"hmm_{k}" for k in range(K)]].to_numpy(np.float64)
    return s0, s1, m, hmm


def alloc_inputs(inputs, stream, name):
    from multi.alloc_env import AllocInputs
    _, _, m, hmm = stream_arrays(inputs, stream)
    return AllocInputs(inputs, m if name == "full" else np.ones_like(m), hmm)


def stream_blocks(stream, inputs):
    """[(fold, b0, b1)] decision bars of each block of the stream."""
    out = []
    for j in sorted(stream["fold"].unique()):
        rows = stream[stream["fold"] == j]
        b0 = int(inputs.dates.get_loc(rows["date"].iloc[0]))
        out.append((int(j), b0, b0 + len(rows)))
    return out


def train_days(s0, s1, blocks=None, holdout=None):
    """Allocator training days: the stream, minus a held-out block and its purge."""
    days = np.arange(s0, s1)
    if holdout is None:
        return days
    _, h0, h1 = next(b for b in blocks if b[0] == holdout)
    keep = (days < h0) | (days >= h1 + config.ALLOC_PURGE_AFTER)
    return days[keep]


def _train_alloc(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import SAC
    from multi.alloc_env import AllocInputs, AllocatorEnv
    from train_specialists import sac_kwargs

    cfg, seed_dir = job["cfg"], job["seed_dir"]
    ai = AllocInputs(job["inputs"], job["m"], job["hmm"])
    env = AllocatorEnv(ai, train_days=job["days"], turnover_pen=cfg["turnover_pen"],
                       seed=cfg["seed"])
    model = SAC("MlpPolicy", env, seed=cfg["seed"], device="cpu", verbose=0,
                **sac_kwargs(cfg["steps"]))
    t0 = time.time()
    model.learn(total_timesteps=cfg["steps"])
    seconds = time.time() - t0
    os.makedirs(seed_dir, exist_ok=True)
    model.save(os.path.join(seed_dir, "model.zip"))
    cfg = dict(cfg, train_seconds=round(seconds, 1), mu_q=[float(v) for v in env.mu_q],
               n_episode_starts=int(len(env.starts)))
    with open(os.path.join(seed_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return seed_dir, seconds


_KEYS = ("name", "seed", "steps", "turnover_pen", "holdout", "first_day", "last_day",
         "n_days")


def train_allocators(inputs, stream, root, seeds, steps, names=NAMES, holdout=None,
                     turnover_pen=None, workers=2):
    """Train (or reuse) one model per seed for each allocator; returns {name: dirs}."""
    from train_specialists import sac_kwargs
    pen = config.ALLOC_TURNOVER_PEN if turnover_pen is None else turnover_pen
    s0, s1, m, hmm = stream_arrays(inputs, stream)
    blocks = stream_blocks(stream, inputs)
    days = train_days(s0, s1, blocks, holdout)
    d = inputs.dates
    jobs, dirs = [], {}
    for name in names:
        mm = m if name == "full" else np.ones_like(m)
        dirs[name] = []
        for k, seed in enumerate(seeds):
            seed_dir = os.path.join(root, f"allocator_{name}", f"seed{k}")
            dirs[name].append(seed_dir)
            cfg = {"layer": "allocator", "name": name, "seed": int(seed),
                   "steps": int(steps), "turnover_pen": float(pen),
                   "holdout": holdout, "first_day": str(d[days[0]].date()),
                   "last_day": str(d[days[-1]].date()), "n_days": int(len(days)),
                   "reward_scale": config.ALLOC_REWARD_SCALE,
                   "alloc_lambda": config.ALLOC_LAMBDA, "step_days": config.ALLOC_STEP,
                   "episode_decisions": config.ALLOC_EPISODE_BARS // config.ALLOC_STEP,
                   "kappa": config.ALLOC_KAPPA, "g_range": config.ALLOC_G_RANGE,
                   "purge_after": config.ALLOC_PURGE_AFTER, "cost_pct": config.COST_PCT,
                   "sac": {k_: v for k_, v in sac_kwargs(steps).items()
                           if k_ != "policy_kwargs"},
                   "net_arch": list(config.MULTI_NET_ARCH),
                   "weight_decay": config.MULTI_WEIGHT_DECAY}
            done = os.path.join(seed_dir, "config.json")
            if os.path.isfile(os.path.join(seed_dir, "model.zip")) and os.path.isfile(done):
                with open(done) as f:
                    old = json.load(f)
                if any(old.get(key) != cfg[key] for key in _KEYS):
                    raise SystemExit(f"{seed_dir} holds a model trained with other "
                                     "settings; use another output folder.")
                continue
            jobs.append({"inputs": inputs, "m": mm, "hmm": hmm, "days": days,
                         "cfg": cfg, "seed_dir": seed_dir})
    if jobs:
        print(f"  training {len(jobs)} allocator models × {steps:,} steps in "
              f"{min(workers, len(jobs))} workers", flush=True)
        if workers > 1 and len(jobs) > 1:
            with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
                for seed_dir, sec in pool.imap_unordered(_train_alloc, jobs):
                    print(f"    {seed_dir}: {sec / 60:.1f} min", flush=True)
        else:
            for job in jobs:
                seed_dir, sec = _train_alloc(job)
                print(f"    {seed_dir}: {sec / 60:.1f} min", flush=True)
    return dirs


def reward_check(ai, days, turnover_pen=None, episodes=40, seed=0):
    """Reward under uniformly random actions: its std (scale check) and penalty ratio."""
    from multi.alloc_env import AllocatorEnv
    env = AllocatorEnv(ai, train_days=days, turnover_pen=turnover_pen, seed=seed)
    rng = np.random.default_rng(seed)
    R, P = [], []
    for _ in range(episodes):
        env.reset()
        done = False
        while not done:
            _, r, _, done, info = env.step(rng.uniform(-1, 1, ai.N + 1))
            R.append(r)
            P.append(info["penalty"])
    R, P = np.array(R), np.array(P)
    return {"reward_std": round(float(R.std()), 4), "mean_penalty": round(float(P.mean()), 4),
            "penalty_ratio": round(float(P.mean() / R.std()), 4)}


def policy_check(models, ai, start, end, mu_q, turnover_pen=None):
    """Reward and penalty of a trained ensemble walking start .. end − 1."""
    from multi.alloc_env import AllocatorEnv, _mean_action
    env = AllocatorEnv(ai, eval_start=start, eval_end=end, mu_q=mu_q,
                       turnover_pen=turnover_pen)
    obs, _ = env.reset()
    R, P, done = [], [], False
    while not done:
        obs, r, _, done, info = env.step(_mean_action(models, obs))
        R.append(r)
        P.append(info["penalty"])
    R, P = np.array(R), np.array(P)
    return {"reward_std": round(float(R.std()), 4), "mean_penalty": round(float(P.mean()), 4),
            "penalty_ratio": round(float(P.mean() / R.std()), 4)}


def _diag(out):
    """Allocator diagnostics of section 9 from run_allocator output."""
    dec = out["decision"]
    beta, u, G = out["beta"][dec], out["u"][dec], out["G"][dec]
    base = np.asarray(config.MULTI_BASE_BUDGETS)
    prev = np.vstack([base, beta[:-1]])
    return {"decisions": int(dec.sum()),
            "mean_budget_change": round(float(np.abs(beta - prev).sum(axis=1).mean()), 4),
            "share_any_u_beyond_095_pct": round(float((np.abs(u) > 0.95).any(axis=1).mean()
                                                      * 100), 1),
            "mean_G": round(float(G.mean()), 4),
            "mean_beta": [round(float(v), 4) for v in beta.mean(axis=0)]}


def crossfit_scores(inputs, stream, run_root, test_year, seeds_per_holdout, steps,
                    names=NAMES, turnover_pen=None, workers=2):
    """
    Out-of-sample score of each allocator on the stream: block j is scored by models
    trained without it.  Returns ({name: summary}, {name: per-day frame}).
    """
    from common.stats import bootstrap_sharpe_ci
    from multi.alloc_env import load_allocator, run_allocator
    from multi.portfolio import active_sharpe, build_positions, net_returns
    Y = test_year
    s0, s1, m, hmm = stream_arrays(inputs, stream)
    blocks = stream_blocks(stream, inputs)
    all_dirs = {}
    for j, _, _ in blocks:
        seeds = [1000 * Y + 10 * j + k for k in range(seeds_per_holdout)]
        all_dirs[j] = train_allocators(
            inputs, stream, os.path.join(run_root, "alloc_crossfit", f"holdout{j}"),
            seeds, steps, names, holdout=j, turnover_pen=turnover_pen, workers=workers)
    summary, frames = {}, {}
    for name in names:
        ai = alloc_inputs(inputs, stream, name)
        x = np.full_like(ai.x_fix, np.nan)
        per_block, seed_corr, diags = [], [], []
        for j, b0, b1 in blocks:
            models = load_allocator(all_dirs[j][name])
            out = run_allocator(models, ai, b0, b1)
            x[b0:b1] = out["x"]
            diags.append(_diag(out))
            acts = []
            for mdl in models:
                o = run_allocator([mdl], ai, b0, b1)
                acts.append(np.column_stack([o["u"], o["g"]])[o["decision"]])
            for a in range(len(acts)):
                for b in range(a + 1, len(acts)):
                    cs = [np.corrcoef(acts[a][:, i], acts[b][:, i])[0, 1]
                          for i in range(acts[a].shape[1])
                          if acts[a][:, i].std() > 0 and acts[b][:, i].std() > 0]
                    if cs:
                        seed_corr.append(np.mean(cs))
            r_l = net_returns(x, inputs.rex, b0, b1)["r"]
            r_b = net_returns(ai.x_fix, inputs.rex, b0, b1)["r"]
            per_block.append({"block": j, "first": str(inputs.dates[b0].date()),
                              "last": str(inputs.dates[b1 - 1].date()),
                              "active_sharpe": round(active_sharpe(r_l, r_b), 3)})
        r_l = net_returns(x, inputs.rex, s0, s1)
        r_b = net_returns(ai.x_fix, inputs.rex, s0, s1)
        lo, hi = bootstrap_sharpe_ci(r_l["r"] - r_b["r"])
        n_dec = sum(dg["decisions"] for dg in diags)
        summary[name] = {
            "baseline": "core_spec" if name == "full" else "core",
            "days": s1 - s0, "first": str(inputs.dates[s0].date()),
            "last": str(inputs.dates[s1 - 1].date()),
            "active_sharpe": round(active_sharpe(r_l["r"], r_b["r"]), 3),
            "ci": [round(lo, 3), round(hi, 3)],
            "active_ret_pct_yr": round(float((r_l["r"] - r_b["r"]).mean() * 252 * 100), 2),
            "turnover_per_year": round(float(r_l["turnover"].mean() * 252), 1),
            "baseline_turnover_per_year": round(float(r_b["turnover"].mean() * 252), 1),
            "per_block": per_block,
            "mean_budget_change": round(sum(dg["mean_budget_change"] * dg["decisions"]
                                            for dg in diags) / n_dec, 4),
            "share_any_u_beyond_095_pct": round(sum(dg["share_any_u_beyond_095_pct"]
                                                    * dg["decisions"] for dg in diags)
                                                / n_dec, 1),
            "mean_G": round(sum(dg["mean_G"] * dg["decisions"] for dg in diags) / n_dec, 4),
            "mean_corr_between_seeds": (round(float(np.mean(seed_corr)), 3)
                                        if seed_corr else None),
        }
        frames[name] = pd.DataFrame({"date": inputs.dates[s0 + 1:s1 + 1],
                                     name: r_l["r"], "baseline": r_b["r"]})
    return summary, frames


def main():
    from multi.data import load_universe
    from multi.spec_env import specialist_tilts  # noqa: F401  (import check)
    from search_multi import load_settings

    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True, help="the fold folder")
    p.add_argument("--test_year", type=int, required=True)
    p.add_argument("--stream", default=None, help="default: <outdir>/crossfit/tilts.csv")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--turnover_pen", type=float, default=None)
    p.add_argument("--settings_root", default=os.path.join("runs", "multi_wf"))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    from multi.features import compute_inputs
    u = load_universe(config.MULTI_DOWNLOAD_START, f"{args.test_year}-01-01")
    inputs = compute_inputs(u)
    stream = pd.read_csv(args.stream or os.path.join(args.outdir, "crossfit", "tilts.csv"),
                         parse_dates=["date"])
    st = load_settings(args.settings_root)
    steps = args.steps or (config.ALLOC_STEPS_QUICK if args.quick
                           else st.get("alloc_steps", config.ALLOC_STEPS))
    n_seeds = args.seeds or (config.ALLOC_SEEDS_QUICK if args.quick else config.ALLOC_SEEDS)
    pen = (args.turnover_pen if args.turnover_pen is not None
           else st.get("alloc_turnover_pen", config.ALLOC_TURNOVER_PEN))
    seeds = [1000 * args.test_year + k for k in range(n_seeds)]
    root = os.path.join(args.outdir, "models")
    train_allocators(inputs, stream, root, seeds, steps, turnover_pen=pen,
                     workers=args.workers)
    s0, s1, _, _ = stream_arrays(inputs, stream)
    for name in NAMES:
        ai = alloc_inputs(inputs, stream, name)
        print(name, "random-action check:", reward_check(ai, np.arange(s0, s1), pen))


if __name__ == "__main__":
    main()
