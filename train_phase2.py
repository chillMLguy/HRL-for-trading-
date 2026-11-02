"""
Phase 2 — train the meta-RL allocator (R1 cash arm, R2 vol targeting,
R3 DSR reward; F3 shadow-book sub-agent observations; F10 Sharpe-based
model selection).

Prerequisites:  train_agents.py  →  train_phase1.py

What the meta-agent trains on:
  default            the final agents on the first (1 - val_frac) of the window.
                     These are days the agents were fitted to, so their behaviour
                     is far better than it will be on new data.
  --crossfit         the out-of-sample proposal stream from train_crossfit.py.
  --stream_dates     the final agents again, on the same dates as the stream.  Differs
                     from --crossfit only in where the proposals come from, which
                     makes it the clean comparison for measuring the leak.

Outputs (under <outdir>/models/<meta_dir>/, phase2 by default):
  best_model.zip    — best meta-agent by VALIDATION SHARPE
  final_model.zip   — meta-agent at end of training
  config.json       — all meta hyper-parameters (read back by evaluate_phase2)
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from env.meta_env import MetaTradingEnv, load_subagents, detect_cnn_path
from regime.hmm_regime import HMMRegimeDetector
from regime.allocators import AGENT_ORDER
from common.crossfit import load_stream, stream_arrays, stream_windows
from common.data import load_prices
from common.callbacks import SharpeEvalCallback
import config


BARS_PER_YEAR = config.BARS_PER_YEAR


def main():
    p = argparse.ArgumentParser(
        description="Phase 2 — Train the meta-RL allocator")
    p.add_argument("--ticker",      default=config.TICKER)
    p.add_argument("--train_start", default=config.TRAIN_START)
    p.add_argument("--train_end",   default=config.TRAIN_END)
    p.add_argument("--modeldir",    default=".",
                   help="Where Phase 0 sub-agents + Phase 1 HMM live.")
    p.add_argument("--outdir",      default=".")
    p.add_argument("--interval",    default=config.INTERVAL,
                   choices=list(BARS_PER_YEAR.keys()))
    p.add_argument("--cost_pct",    type=float, default=config.COST_PCT)
    p.add_argument("--seed",        type=int, default=config.SEED)
    p.add_argument("--val_frac",    type=float, default=0.15)
    p.add_argument("--no_cnn",      action="store_true")
    p.add_argument("--no_hmm",      action="store_true",
                   help="Ablation: no HMM regime observation.")
    p.add_argument("--data_csv",    default=None)

    # Where the agents' behaviour comes from
    source = p.add_mutually_exclusive_group()
    source.add_argument("--crossfit", action="store_true",
                        help="Train on out-of-sample proposals (train_crossfit.py).")
    source.add_argument("--stream_dates", action="store_true",
                        help="Final agents' in-sample proposals on the stream's dates.")
    p.add_argument("--episode_bars", type=int, default=None,
                   help="Random-start episode length in bars, 0 = always replay one "
                        "path. Default: config.META_TRAIN_EPISODE_BARS with --crossfit "
                        "or --stream_dates, otherwise 0.")
    p.add_argument("--meta_dir", default="phase2",
                   help="Folder under models/ and logs/ for this meta-agent.")

    # Training budget
    p.add_argument("--quick", action="store_true", help="40k steps, [64,64]")
    p.add_argument("--full",  action="store_true", help="300k steps, [256,256]")
    p.add_argument("--steps", default=0, type=int,
                   help="Explicit timestep override (walk-forward use)")

    # Meta design (R1/R2/R3) — defaults from config.py
    p.add_argument("--no_cash", action="store_true",
                   help="Ablation: disable the cash arm (R1).")
    p.add_argument("--vol_target", type=float, default=config.VOL_TARGET,
                   help="Annualized vol target for the overlay (0 = off, R2).")
    p.add_argument("--vol_target_cap", type=float,
                   default=config.VOL_TARGET_CAP)
    p.add_argument("--meta_reward", choices=["dsr", "composite"],
                   default=config.META_REWARD,
                   help="R3 DSR reward (default) or the mean-variance "
                        "composite (ablation).")
    p.add_argument("--dsr_eta", type=float, default=config.DSR_ETA)
    p.add_argument("--lam_var_meta", type=float, default=12.0,
                   help="Variance aversion of the composite reward.")
    p.add_argument("--alpha",       type=float, default=0.05)
    p.add_argument("--beta",        type=float, default=0.05)
    p.add_argument("--dd_free",     type=float, default=0.03)
    p.add_argument("--dd_max",      type=float, default=0.10)
    p.add_argument("--logit_scale", type=float, default=2.0)
    p.add_argument("--perf_window", type=int,   default=20)
    args = p.parse_args()

    if args.steps > 0:
        timesteps, net_arch, mode = args.steps, [128, 128], "CUSTOM"
    elif args.quick:
        timesteps, net_arch, mode = 40_000, [64, 64], "QUICK"
    elif args.full:
        timesteps, net_arch, mode = 300_000, [256, 256], "FULL"
    else:
        timesteps, net_arch, mode = 150_000, [128, 128], "NORMAL"

    ann = BARS_PER_YEAR[args.interval]
    use_cash = not args.no_cash
    use_stream_dates = args.crossfit or args.stream_dates
    episode_bars = args.episode_bars if args.episode_bars is not None else (
        config.META_TRAIN_EPISODE_BARS if use_stream_dates else 0)
    mode = ("crossfit" if args.crossfit else
            "in_sample_stream_dates" if args.stream_dates else "in_sample")
    out_dir = os.path.join(args.outdir, "models", args.meta_dir)
    log_dir = os.path.join(args.outdir, "logs", args.meta_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  Phase 2 — Meta-RL allocator  ({mode}: {timesteps:,} steps)")
    print(f"  {args.ticker}   {args.train_start} → {args.train_end}")
    print(f"  reward={args.meta_reward}  cash_arm={use_cash}  "
          f"vol_target={args.vol_target}")
    print(f"  training data: {mode}  episode_bars={episode_bars or 'full path'}")
    print(f"{'='*64}")

    prices = load_prices(args.ticker, args.train_start, args.train_end,
                         args.interval, data_csv=args.data_csv)
    if use_stream_dates:
        stream = load_stream(args.modeldir)
        windows = stream_windows(stream, prices, config.WARM_BARS)
        train_prices, train_part, train_first = windows["train"]
        val_prices, val_part, val_first = windows["val"]
        n_train, n_val = len(train_part), len(val_part)
        n_folds = int(stream["fold"].max())
    else:
        split = int(len(prices) * (1.0 - args.val_frac))
        train_prices, val_prices = prices.iloc[:split], prices.iloc[split:]
        train_first = val_first = None
        n_train, n_val = len(train_prices), len(val_prices)
        n_folds = None
    print(f"  Train: {n_train} bars | Val: {n_val} bars")

    subagents = {}
    if not args.crossfit:
        print("\n  Loading frozen Phase 0 sub-agents ...")
        subagents = load_subagents(args.modeldir)
        if not subagents:
            print("\n  ERROR: no sub-agents found. Run train_agents.py first.")
            return

    cnn_path = None if (args.no_cnn or args.crossfit) \
        else detect_cnn_path(args.modeldir)
    print(f"  CNN features: {'enabled' if cnn_path else 'disabled'}")

    hmm = None
    if args.crossfit:
        print(f"  Replaying recorded proposals and HMM posteriors ({n_folds} folds)"
              + (" without the HMM (--no_hmm)" if args.no_hmm else ""))
    elif not args.no_hmm:
        hmm_path = os.path.join(args.modeldir, "models", "phase1",
                                "hmm_model.pkl")
        if os.path.isfile(hmm_path):
            hmm = HMMRegimeDetector.load(hmm_path)
            print(f"  HMM loaded ({hmm.n_states} states)")
        else:
            print("  ⚠  HMM not found — training WITHOUT regime obs.")
    else:
        print("  HMM disabled (--no_hmm ablation).")

    meta_kwargs = dict(
        use_cash=use_cash, vol_target=args.vol_target,
        vol_target_cap=args.vol_target_cap, meta_reward=args.meta_reward,
        dsr_eta=args.dsr_eta, lam_var_meta=args.lam_var_meta,
        alpha=args.alpha, beta=args.beta,
        dd_free=args.dd_free, dd_max=args.dd_max,
        logit_scale=args.logit_scale, perf_window=args.perf_window,
    )

    def recorded(part, slice_):
        """Stream arrays for a slice; empty when using live agents."""
        if not args.crossfit:
            return {}
        proposals, hmm_post = stream_arrays(part, slice_, AGENT_ORDER)
        return {"proposal_stream": proposals,
                "hmm_stream": None if args.no_hmm else hmm_post}

    train_recorded = recorded(train_part, train_prices) if use_stream_dates else {}
    val_recorded = recorded(val_part, val_prices) if use_stream_dates else {}

    def make_train():
        return Monitor(MetaTradingEnv(
            train_prices, subagents, hmm_detector=hmm, ann=ann,
            cost_pct=args.cost_pct, eval_mode=False,
            cnn_model_path=cnn_path, start_index=train_first,
            train_episode_bars=episode_bars, **train_recorded, **meta_kwargs))

    train_env = DummyVecEnv([make_train])
    val_env = MetaTradingEnv(
        val_prices, subagents, hmm_detector=hmm, ann=ann,
        cost_pct=args.cost_pct, eval_mode=True,
        cnn_model_path=cnn_path, start_index=val_first,
        **val_recorded, **meta_kwargs)

    eval_cb = SharpeEvalCallback(
        val_env, ann=ann, eval_freq=max(timesteps // 20, 1000),
        best_model_save_path=out_dir, verbose=0)

    print(f"\n  Meta-obs dim: {train_env.observation_space.shape[0]}  |  "
          f"action dim: {train_env.action_space.shape[0]}")
    print("  Training meta-agent ...")

    model = SAC(
        "MlpPolicy", train_env,
        learning_rate=3e-4,
        buffer_size=min(100_000, timesteps),
        batch_size=256, gamma=0.99, tau=0.005, ent_coef="auto",
        policy_kwargs=dict(net_arch=net_arch),
        tensorboard_log=log_dir, seed=args.seed, verbose=0,
    )
    model.learn(total_timesteps=timesteps, callback=eval_cb,
                progress_bar=True)
    model.save(os.path.join(out_dir, "final_model"))

    cfg = {
        "ticker": args.ticker,
        "train_start": args.train_start, "train_end": args.train_end,
        "interval": args.interval, "bars_per_year": ann,
        "cost_pct": args.cost_pct, "seed": args.seed,
        "timesteps": timesteps, "net_arch": net_arch,
        "use_cash": use_cash, "vol_target": args.vol_target,
        "vol_target_cap": args.vol_target_cap,
        "meta_reward": args.meta_reward, "dsr_eta": args.dsr_eta,
        "lam_var_meta": args.lam_var_meta,
        "alpha": args.alpha, "beta": args.beta,
        "dd_free": args.dd_free, "dd_max": args.dd_max,
        "logit_scale": args.logit_scale, "perf_window": args.perf_window,
        "use_hmm": (not args.no_hmm) if args.crossfit else hmm is not None,
        "use_cnn": cnn_path is not None,
        "agent_order": AGENT_ORDER,
        "training_mode": mode, "episode_bars": episode_bars,
        "val_frac": args.val_frac, "crossfit_folds": n_folds,
        "train_bars": n_train, "val_bars": n_val,
        "obs_dim": int(train_env.observation_space.shape[0]),
        "best_val_sharpe": eval_cb.best_sharpe,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n  Best val Sharpe: {eval_cb.best_sharpe:+.3f}")
    print(f"  Saved best_model.zip / final_model.zip / config.json → {out_dir}")
    print("\n  Phase 2 training complete.\n")


if __name__ == "__main__":
    main()
