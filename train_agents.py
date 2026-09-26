import argparse
import json
import os
import warnings
import multiprocessing as mp
warnings.filterwarnings("ignore")

import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from env.trading_env import TradingEnv
from common.data import load_prices
from common.callbacks import SharpeEvalCallback
from config import AGENT_PRESETS
import config


BARS_PER_YEAR = config.BARS_PER_YEAR


def train_split(prices, ratio=0.8):
    split = int(len(prices) * ratio)
    return prices.iloc[:split], prices.iloc[split:]


def _train_worker(args):
    """Top-level function (required for multiprocessing pickle)."""
    (agent_name, train_prices, eval_prices,
     outdir, seed, timesteps, net_arch, bars_per_year, cost_pct,
     cnn_model_path, episode_bars) = args

    warnings.filterwarnings("ignore")

    def make_train():
        return Monitor(TradingEnv(
            train_prices, preset=agent_name, bars_per_year=bars_per_year,
            cost_pct=cost_pct, cnn_model_path=cnn_model_path,
            train_episode_bars=episode_bars))          # F11 random-start eps

    train_env = DummyVecEnv([make_train])
    # F10: model selection on validation Sharpe (full path, eval mode)
    sharpe_eval_env = TradingEnv(
        eval_prices, preset=agent_name, bars_per_year=bars_per_year,
        cost_pct=cost_pct, cnn_model_path=cnn_model_path, eval_mode=True)

    model_dir = os.path.join(outdir, "models", f"{agent_name}_agent")
    log_dir   = os.path.join(outdir, "logs",   f"{agent_name}_agent")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)

    eval_cb = SharpeEvalCallback(
        sharpe_eval_env, ann=bars_per_year,
        eval_freq=max(timesteps // 20, 1000),
        best_model_save_path=model_dir, verbose=0)

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=min(100_000, timesteps),
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=net_arch),
        tensorboard_log=log_dir,
        seed=seed,
        verbose=0,
    )

    model.learn(total_timesteps=timesteps, callback=eval_cb,
                progress_bar=True)
    model.save(os.path.join(model_dir, "final_model"))

    # F9: persist exactly how this agent was trained
    preset = AGENT_PRESETS[agent_name]
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump({
            "agent": agent_name, "preset": preset, "seed": seed,
            "timesteps": timesteps, "net_arch": net_arch,
            "cost_pct": cost_pct, "bars_per_year": bars_per_year,
            "episode_bars": episode_bars,
            "reward": "quadratic_utility+dd_zones",
            "best_val_sharpe": eval_cb.best_sharpe,
        }, f, indent=2)

    print(f"  [{agent_name}] done — best val Sharpe "
          f"{eval_cb.best_sharpe:+.3f}")
    return agent_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",   default=config.TICKER)
    parser.add_argument("--start",    default=config.TRAIN_START)
    parser.add_argument("--end",      default=config.TRAIN_END)
    parser.add_argument("--interval", default=config.INTERVAL,
                        choices=list(BARS_PER_YEAR.keys()))
    parser.add_argument("--outdir",  default=".")
    parser.add_argument("--seed",    default=config.SEED, type=int)
    parser.add_argument("--quick",   action="store_true",
                        help="50k steps, [64,64] net")
    parser.add_argument("--full",    action="store_true",
                        help="500k steps, [256,256] net")
    parser.add_argument("--steps",   default=0, type=int,
                        help="Explicit timestep override (walk-forward use)")
    parser.add_argument("--workers", default=3, type=int)
    parser.add_argument("--agents", default=None, nargs="+",
                        choices=list(AGENT_PRESETS.keys()))
    parser.add_argument("--cost_pct", default=config.COST_PCT, type=float)
    parser.add_argument("--episode_bars", default=config.TRAIN_EPISODE_BARS,
                        type=int,
                        help="Random-start training episode length in bars "
                             "(0 = full-path episodes)")
    parser.add_argument("--no_cnn", action="store_true")
    parser.add_argument("--data_csv", default=None,
                        help="Local price cache CSV (see download_data.py)")
    args = parser.parse_args()

    if args.steps > 0:
        timesteps = args.steps
        net_arch  = [128, 128]
        print(f"  Mode: CUSTOM ({timesteps:,} steps, [128,128] net)")
    elif args.quick:
        timesteps = 50_000
        net_arch  = [64, 64]
        print("  Mode: QUICK (50k steps, [64,64] net)")
    elif args.full:
        timesteps = 500_000
        net_arch  = [256, 256]
        print("  Mode: FULL (500k steps, [256,256] net)")
    else:
        timesteps = 200_000
        net_arch  = [128, 128]
        print("  Mode: NORMAL (200k steps, [128,128] net)")

    bars_per_year = BARS_PER_YEAR[args.interval]

    cnn_model_path = None
    if not args.no_cnn:
        cnn_path = os.path.join(args.outdir, "models", "cnn_features",
                                "cnn_model.pt")
        if os.path.exists(cnn_path):
            cnn_model_path = cnn_path
            print(f"  CNN model found: {cnn_path}")
        else:
            print("  CNN model not found — CNN features disabled.")

    selected = args.agents if args.agents else list(AGENT_PRESETS.keys())

    print(f"\n=== Phase 0: Risk-ladder agents (R4) ===")
    print(f"  Ticker   : {args.ticker}   {args.start} → {args.end}")
    print(f"  Cost/side: {args.cost_pct*100:.3f}%   Steps: {timesteps:,}"
          f"   Episode bars: {args.episode_bars or 'full path'}")
    for name in selected:
        p = AGENT_PRESETS[name]
        print(f"    {name:14s} lam_var={p['lam_var']:<6} "
              f"max_pos={p['max_pos']:<4} dd_max={p['dd_max']}")

    prices = load_prices(args.ticker, args.start, args.end, args.interval,
                         data_csv=args.data_csv)
    train_prices, eval_prices = train_split(prices)
    print(f"  Train: {len(train_prices)} bars | Val: {len(eval_prices)} bars\n")

    worker_args = [
        (name, train_prices, eval_prices,
         args.outdir, args.seed + i, timesteps, net_arch, bars_per_year,
         args.cost_pct, cnn_model_path, args.episode_bars)
        for i, name in enumerate(selected)
    ]

    n_workers = min(args.workers, len(selected))
    if n_workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            pool.map(_train_worker, worker_args)
    else:
        for wa in worker_args:
            _train_worker(wa)

    print("\n✓ All agents trained.")


if __name__ == "__main__":
    main()
