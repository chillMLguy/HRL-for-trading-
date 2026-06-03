"""
Phase 2 — train the meta-RL allocator.

A single SAC meta-agent learns to blend the three FROZEN Phase 0 sub-agents
(aggressive / balanced / conservative).  At each step it sees the HMM regime
posterior, multi-timeframe market features, the sub-agents' proposed actions
and their recent performance, and outputs a 3-vector that is soft-maxed into
allocation weights.  See env/meta_env.py for the full observation / reward spec.

This replaces the fixed weight-mapping matrix of the Phase 1 HMMAllocator with a
learned, regime-conditioned policy.

Usage
-----
  python train_phase2.py --ticker "^DJI" \
      --train_start 2013-01-01 --train_end 2023-12-31 \
      --modeldir . --outdir . --interval 1d

  # fast smoke test:
  python train_phase2.py --quick

Prerequisites (run these first):
  python train_agents.py   ...     # Phase 0 sub-agents  → models/<name>_agent/
  python train_phase1.py   ...     # fits the HMM        → models/phase1/hmm_model.pkl

Outputs (under <outdir>/models/phase2/):
  best_model.zip    — best meta-agent on the validation slice (EvalCallback)
  final_model.zip   — meta-agent at end of training
  config.json       — all meta hyper-parameters
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import yfinance as yf

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from env.meta_env import MetaTradingEnv, load_subagents, detect_cnn_path
from regime.hmm_regime import HMMRegimeDetector
from regime.allocators import AGENT_ORDER
import config


BARS_PER_YEAR = config.BARS_PER_YEAR


def make_meta_env(prices, subagents, hmm, ann, cost_pct, meta_kwargs,
                  eval_mode, cnn_path):
    def _init():
        return Monitor(MetaTradingEnv(
            prices, subagents, hmm_detector=hmm, ann=ann, cost_pct=cost_pct,
            eval_mode=eval_mode, cnn_model_path=cnn_path, **meta_kwargs))
    return DummyVecEnv([_init])


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
    p.add_argument("--val_frac",    type=float, default=0.15,
                   help="Last fraction of training used for EvalCallback "
                        "model selection.")
    p.add_argument("--no_cnn",      action="store_true")
    p.add_argument("--no_hmm",      action="store_true",
                   help="Ablation: train the meta-agent without HMM regime obs.")

    # Training budget
    p.add_argument("--quick", action="store_true",
                   help="40k steps, [64,64] net — quick smoke test.")
    p.add_argument("--full",  action="store_true",
                   help="300k steps, [256,256] net.")

    # Meta hyper-parameters (env-side reward shaping)
    p.add_argument("--lam_meta",    type=float, default=0.5)
    p.add_argument("--alpha",       type=float, default=0.05,
                   help="Diversity (entropy) bonus weight.")
    p.add_argument("--beta",        type=float, default=0.05,
                   help="Turnover penalty weight.")
    p.add_argument("--dd_free",     type=float, default=0.03)
    p.add_argument("--dd_max",      type=float, default=0.10)
    p.add_argument("--logit_scale", type=float, default=2.0)
    p.add_argument("--perf_window", type=int,   default=20)
    args = p.parse_args()

    if args.quick:
        timesteps, net_arch = 40_000, [64, 64]
        mode = "QUICK"
    elif args.full:
        timesteps, net_arch = 300_000, [256, 256]
        mode = "FULL"
    else:
        timesteps, net_arch = 150_000, [128, 128]
        mode = "NORMAL"

    ann = BARS_PER_YEAR[args.interval]
    out_dir = os.path.join(args.outdir, "models", "phase2")
    log_dir = os.path.join(args.outdir, "logs", "phase2")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  Phase 2 — Meta-RL allocator  ({mode}: {timesteps:,} steps, "
          f"net {net_arch})")
    print(f"  {args.ticker}   {args.train_start} → {args.train_end}")
    print(f"{'='*64}")

    # ── 1. Data ───────────────────────────────────────────────────────────
    print("\n  Downloading training data ...")
    df = yf.download(args.ticker, start=args.train_start, end=args.train_end,
                     interval=args.interval, auto_adjust=True, progress=False)
    prices = df["Close"].squeeze().dropna()
    print(f"  {len(prices)} bars loaded.")

    split = int(len(prices) * (1.0 - args.val_frac))
    train_prices = prices.iloc[:split]
    val_prices   = prices.iloc[split:]
    print(f"  Train: {len(train_prices)} bars | Val: {len(val_prices)} bars")

    # ── 2. Frozen sub-agents + CNN extractor ──────────────────────────────
    print("\n  Loading frozen Phase 0 sub-agents ...")
    subagents = load_subagents(args.modeldir)
    if not subagents:
        print("\n  ERROR: no sub-agents found. Run train_agents.py first.")
        return

    cnn_path = None if args.no_cnn else detect_cnn_path(args.modeldir)
    print(f"  CNN features: {'enabled' if cnn_path else 'disabled'}")

    # ── 3. HMM regime detector ─────────────────────────────────────────────
    hmm = None
    if not args.no_hmm:
        hmm_path = os.path.join(args.modeldir, "models", "phase1",
                                "hmm_model.pkl")
        if os.path.isfile(hmm_path):
            hmm = HMMRegimeDetector.load(hmm_path)
            print(f"  HMM loaded ({hmm.n_states} states) ← {hmm_path}")
        else:
            print(f"  ⚠  HMM not found at {hmm_path}; training WITHOUT regime "
                  f"obs. Run train_phase1.py for the regime-conditioned model.")
    else:
        print("  HMM disabled (--no_hmm ablation).")

    # ── 4. Build envs ──────────────────────────────────────────────────────
    meta_kwargs = dict(
        lam_meta=args.lam_meta, alpha=args.alpha, beta=args.beta,
        dd_free=args.dd_free, dd_max=args.dd_max,
        logit_scale=args.logit_scale, perf_window=args.perf_window,
    )
    train_env = make_meta_env(train_prices, subagents, hmm, ann, args.cost_pct,
                              meta_kwargs, eval_mode=False, cnn_path=cnn_path)
    val_env   = make_meta_env(val_prices, subagents, hmm, ann, args.cost_pct,
                              meta_kwargs, eval_mode=True, cnn_path=cnn_path)

    eval_cb = EvalCallback(
        val_env, best_model_save_path=out_dir, log_path=log_dir,
        eval_freq=max(timesteps // 20, 1000), n_eval_episodes=1,
        deterministic=True, verbose=0)

    # ── 5. Train ───────────────────────────────────────────────────────────
    print(f"\n  Meta-obs dim: {train_env.observation_space.shape[0]}  |  "
          f"action dim: {train_env.action_space.shape[0]}")
    print(f"  Reward shaping: λ_meta={args.lam_meta}  α(div)={args.alpha}  "
          f"β(turn)={args.beta}  logit_scale={args.logit_scale}")
    print("\n  Training meta-agent ...")

    model = SAC(
        "MlpPolicy", train_env,
        learning_rate=3e-4,
        buffer_size=min(100_000, timesteps),
        batch_size=256, gamma=0.99, tau=0.005, ent_coef="auto",
        policy_kwargs=dict(net_arch=net_arch),
        tensorboard_log=log_dir, seed=args.seed, verbose=0,
    )
    model.learn(total_timesteps=timesteps, callback=eval_cb, progress_bar=True)
    model.save(os.path.join(out_dir, "final_model"))

    # ── 6. Persist config ──────────────────────────────────────────────────
    cfg = {
        "ticker": args.ticker,
        "train_start": args.train_start, "train_end": args.train_end,
        "interval": args.interval, "bars_per_year": ann,
        "cost_pct": args.cost_pct, "seed": args.seed,
        "timesteps": timesteps, "net_arch": net_arch,
        "lam_meta": args.lam_meta, "alpha": args.alpha, "beta": args.beta,
        "dd_free": args.dd_free, "dd_max": args.dd_max,
        "logit_scale": args.logit_scale, "perf_window": args.perf_window,
        "use_hmm": hmm is not None, "use_cnn": cnn_path is not None,
        "agent_order": AGENT_ORDER,
        "obs_dim": int(train_env.observation_space.shape[0]),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n  Saved best_model.zip / final_model.zip / config.json → {out_dir}")
    print("\n  Phase 2 training complete.\n")


if __name__ == "__main__":
    main()
