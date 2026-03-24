
import argparse
import os
import warnings
import multiprocessing as mp
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from env.trading_env import TradingEnv, AgentType


BARS_PER_YEAR = {"1d": 252, "1h": 1638, "30m": 3276, "15m": 6552}


def download_data(ticker, start, end, interval="1h"):
    print(f"  Downloading {ticker} {start} → {end} (interval={interval})...")
    df = yf.download(ticker, start=start, end=end, interval=interval,
                     auto_adjust=True, progress=False)
    prices = df["Close"].squeeze().dropna()
    print(f"  {len(prices)} bars loaded.")
    return prices


def train_split(prices, ratio=0.8):
    split = int(len(prices) * ratio)
    return prices.iloc[:split], prices.iloc[split:]


def make_env(prices, agent_type, bars_per_year=1638, cost_pct=0.0002):
    def _init():
        return Monitor(TradingEnv(prices, agent_type=agent_type,
                                  bars_per_year=bars_per_year,
                                  cost_pct=cost_pct))
    return DummyVecEnv([_init])


def _train_worker(args):
    """
    Top-level function (required for multiprocessing pickle).
    Runs one agent training to completion and returns the model path.
    """
    (agent_name, train_prices, eval_prices,
     outdir, seed, timesteps, net_arch, bars_per_year, cost_pct) = args

    warnings.filterwarnings("ignore")
    agent_type = AgentType(agent_name)

    train_env = make_env(train_prices, agent_type, bars_per_year, cost_pct)
    eval_env  = make_env(eval_prices,  agent_type, bars_per_year, cost_pct)

    model_dir = os.path.join(outdir, "models", f"{agent_name}_agent")
    log_dir   = os.path.join(outdir, "logs",   f"{agent_name}_agent")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=max(timesteps // 20, 1000),  # evaluate 20 times total
        n_eval_episodes=1,                      # was 5 — saves 80% eval time
        deterministic=True,
        verbose=0,
    )

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
    print(f"  [{agent_name}] done.")
    return agent_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",   default="SPY")
    parser.add_argument("--start",    default="2024-04-01")
    parser.add_argument("--end",      default="2025-10-31")
    parser.add_argument("--interval", default="1h",
                        choices=list(BARS_PER_YEAR.keys()),
                        help="Bar interval (default: 1h). Note: yfinance limits "
                             "intraday history to ~730 days.")
    parser.add_argument("--outdir",  default=".")
    parser.add_argument("--seed",    default=42, type=int)
    parser.add_argument("--quick",   action="store_true",
                        help="50k steps, [64,64] net — full loop in ~5 min")
    parser.add_argument("--full",    action="store_true",
                        help="500k steps, [256,256] net — original quality")
    parser.add_argument("--workers", default=3, type=int,
                        help="Parallel processes (default=3, one per agent)")
    parser.add_argument("--agents", default=None, nargs="+",
                        choices=[a.value for a in AgentType],
                        help="Which agents to train (default: all). "
                             "e.g. --agents aggressive cvar rachev")
    parser.add_argument("--cost_pct", default=0.0002, type=float,
                        help="One-way transaction cost fraction "
                             "(default 0.0002 = 0.02%% for intraday). "
                             "Use 0.001 for daily/conservative estimate.")
    args = parser.parse_args()

    if args.quick:
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

    print(f"\n=== Phase 0: Baseline Flat Agents ===")
    print(f"  Ticker   : {args.ticker}")
    print(f"  Interval : {args.interval}  ({bars_per_year} bars/year)")
    print(f"  Period   : {args.start} → {args.end}")
    print(f"  Cost/side: {args.cost_pct*100:.3f}%")
    print(f"  Steps    : {timesteps:,} per agent")
    print(f"  Network  : {net_arch}")
    print(f"  Workers  : {args.workers} (parallel)")

    prices = download_data(args.ticker, args.start, args.end, args.interval)
    train_prices, eval_prices = train_split(prices)
    print(f"  Train: {len(train_prices)} bars | Eval: {len(eval_prices)} bars\n")

    selected = (
        [AgentType(a) for a in args.agents]
        if args.agents else list(AgentType)
    )
    print(f"  Agents  : {[a.value for a in selected]}")

    worker_args = [
        (agent.value, train_prices, eval_prices,
         args.outdir, args.seed + i, timesteps, net_arch, bars_per_year,
         args.cost_pct)
        for i, agent in enumerate(selected)
    ]

    # Spawn one process per agent — they run truly in parallel
    n_workers = min(args.workers, len(AgentType))
    if n_workers > 1:
        # 'spawn' context avoids macOS fork issues with numpy/torch
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            pool.map(_train_worker, worker_args)
    else:
        for wa in worker_args:
            _train_worker(wa)

    print("\n✓ All agents trained.")
    print("  Run: python evaluate_agents.py --ticker", args.ticker)


if __name__ == "__main__":
    main()