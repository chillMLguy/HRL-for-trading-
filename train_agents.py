"""
Phase 0 — Agent Training
========================
Trains three SAC agents (conservative, neutral, aggressive) on
the same price series. Each agent is identical in architecture
and hyperparameters — only the reward function differs.

Usage
-----
    python train_agents.py --ticker SPY --start 2010-01-01 --end 2022-12-31
    python train_agents.py --ticker BTC-USD --start 2018-01-01 --end 2023-12-31

Outputs
-------
    models/conservative_agent/  — SB3 SAC model
    models/neutral_agent/
    models/aggressive_agent/
    logs/                       — TensorBoard logs per agent
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor

from env.trading_env import TradingEnv, AgentType


# ------------------------------------------------------------------
# Hyperparameters (shared across all three agents)
# ------------------------------------------------------------------
TIMESTEPS      = 50000  # total env steps per agent
EVAL_FREQ      = 10000    # evaluate every N steps
LEARNING_RATE  = 3e-4
BATCH_SIZE     = 256
BUFFER_SIZE    = 10000
GAMMA          = 0.99
TAU            = 0.005
ENT_COEF       = "auto"    # SAC auto-tunes entropy coefficient

# Network: [256, 256] for policy and Q networks
POLICY_KWARGS  = dict(net_arch=[256, 256])


def download_data(ticker: str, start: str, end: str) -> pd.Series:
    """Download adjusted close prices via yfinance."""
    print(f"  Downloading {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    prices = df["Close"].squeeze().dropna()
    print(f"  {len(prices)} trading days loaded.")
    return prices


def train_split(prices: pd.Series, train_ratio: float = 0.8):
    """Chronological train / test split (no shuffling — avoids lookahead bias)."""
    split = int(len(prices) * train_ratio)
    return prices.iloc[:split], prices.iloc[split:]    


def make_env(prices: pd.Series, agent_type: AgentType, seed: int = 0):
    """Factory for a monitored, vectorised environment."""
    def _init():
        env = TradingEnv(prices, agent_type=agent_type)
        env = Monitor(env)
        return env
    return DummyVecEnv([_init])


def train_agent(
    agent_type: AgentType,
    train_prices: pd.Series,
    eval_prices: pd.Series,
    output_dir: str,
    seed: int = 42,
) -> SAC:
    name = agent_type.value
    print(f"\n{'='*60}")
    print(f"  Training: {name.upper()} agent")
    print(f"{'='*60}")

    # Environments
    train_env = make_env(train_prices, agent_type, seed)
    eval_env  = make_env(eval_prices,  agent_type, seed)

    model_dir = os.path.join(output_dir, "models", f"{name}_agent")
    log_dir   = os.path.join(output_dir, "logs",   f"{name}_agent")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=5,
        deterministic=True,
        verbose=0,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=model_dir,
        name_prefix=name,
    )

    # Model (SAC — handles continuous actions naturally)
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        ent_coef=ENT_COEF,
        policy_kwargs=POLICY_KWARGS,
        tensorboard_log=log_dir,
        seed=seed,
        verbose=0,
    )

    model.learn(
        total_timesteps=TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    print(f"  Saved → {final_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Phase 0 — Train three flat RL agents")
    parser.add_argument("--ticker",  default="SPY",        help="yfinance ticker symbol")
    parser.add_argument("--start",   default="2010-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default="2022-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--outdir",  default=".",          help="Output directory")
    parser.add_argument("--seed",    default=42, type=int, help="Random seed")
    args = parser.parse_args()

    print("\n=== Phase 0: Baseline Flat Agents ===")
    print(f"  Ticker : {args.ticker}")
    print(f"  Period : {args.start} → {args.end}")

    # Data
    prices = download_data(args.ticker, args.start, args.end)
    train_prices, eval_prices = train_split(prices, train_ratio=0.8)
    print(f"  Train: {len(train_prices)} days | Eval: {len(eval_prices)} days")

    # Train all three agents
    agents = {}
    for agent_type in AgentType:
        agents[agent_type] = train_agent(
            agent_type=agent_type,
            train_prices=train_prices,
            eval_prices=eval_prices,
            output_dir=args.outdir,
            seed=args.seed,
        )

    print("\n✓ All three agents trained. Run evaluate_agents.py to compare performance.")


if __name__ == "__main__":
    main()
