"""
Overfitting check: how each saved model scores on the data it was fitted to.

The test year alone cannot show overfitting, and evaluate_phase2.py only rolls the
models over the test year.  This script rolls them over their training and
validation windows too and writes <rundir>/window_sharpe.csv, so dashboard.py can
show the gap for every run:

    model        agent or allocator
    window       train, val, or stream (an agent on its out-of-sample blocks)
    proposals    in_sample / out_of_sample: which kind of agent behaviour this
                 row saw (blank for an agent on its own training data)
    sharpe       annualized, net of costs
    bars         days in the window
    bh_sharpe    buy-and-hold on the same days

run_all.py and walk_forward.py run this automatically.  It needs the model files,
which git does not track, so run it by hand on the machine that trained them:

    python overfit_check.py --rundir runs/latest
"""
import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from common.crossfit import load_stream, stream_arrays, stream_windows
from common.data import load_prices
from common.stats import ann_sharpe
from env.meta_env import load_subagents, meta_env_from_config
from env.trading_env import TradingEnv
from evaluate_phase1 import run_allocator
from regime.allocators import AGENT_ORDER, EqualWeightAllocator
from regime.hmm_regime import HMMRegimeDetector
import config


def _row(model, window, proposals, returns, bh, ann):
    return {"model": model, "window": window, "proposals": proposals,
            "sharpe": round(ann_sharpe(returns, ann), 3), "bars": len(returns),
            "bh_sharpe": round(ann_sharpe(bh, ann), 3)}


def _simple_returns(prices, first):
    """Next-bar simple returns from bar `first` on."""
    p = prices.values.astype(np.float64)
    return p[first + 1:] / p[first:-1] - 1.0


def agent_rows(subagents, prices, cost, ann):
    """Each agent on the two windows train_agents.py gave it: 80% fit, 20% select."""
    cut = int(len(prices) * config.AGENT_TRAIN_RATIO)
    rows = []
    for name in AGENT_ORDER:
        for window, part in (("train", prices.iloc[:cut]), ("val", prices.iloc[cut:])):
            env = TradingEnv(part, preset=name, bars_per_year=ann, cost_pct=cost,
                             eval_mode=True)
            obs, _ = env.reset()
            rets, mkt, done = [], [], False
            while not done:
                action, _ = subagents[name].predict(obs, deterministic=True)
                obs, _, done, _, info = env.step(action)
                rets.append(info["net_ret"])
                mkt.append(info["price_ret"])
            rows.append(_row(name, window, "", np.array(rets), np.array(mkt), ann))
    return rows


def _positions(actions):
    """Agent actions in [-1, 1] to positions, using each agent's cap."""
    caps = np.array([config.AGENT_PRESETS[a]["max_pos"] for a in AGENT_ORDER])
    return actions * caps


def _net(position, market, cost):
    return position * market - np.abs(np.diff(position, prepend=0.0)) * cost


def stream_rows(stream, prices, cost, ann):
    """Agents and equal weight on the out-of-sample stream: the honest numbers."""
    rows = []
    windows = stream_windows(stream, prices, config.WARM_BARS)
    for window, (slice_, part, first) in windows.items():
        proposals, _ = stream_arrays(part, slice_, AGENT_ORDER)
        bars = np.arange(first, first + len(part))
        market = _simple_returns(slice_, first)[: len(bars)]
        pos = _positions(proposals[bars[: len(market)]])
        rows.append(_row("equal_weight", window, "out_of_sample",
                         _net(pos.mean(axis=1), market, cost), market, ann))
    # each agent over the whole stream
    proposals = stream[[f"a_{a}" for a in AGENT_ORDER]].values.astype(np.float64)
    idx = prices.index.searchsorted(stream.index)
    p = prices.values.astype(np.float64)
    keep = idx + 1 < len(p)
    market = p[idx[keep] + 1] / p[idx[keep]] - 1.0
    pos = _positions(proposals[keep])
    for i, name in enumerate(AGENT_ORDER):
        rows.append(_row(name, "stream", "out_of_sample",
                         _net(pos[:, i], market, cost), market, ann))
    return rows


def _meta_returns(meta, env):
    obs, _ = env.reset()
    rets, mkt, done = [], [], False
    while not done:
        action, _ = meta.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        rets.append(info["net_ret"])
        mkt.append(info["price_ret"])
    return np.array(rets), np.array(mkt)


def meta_rows(name, meta_dir, rundir, prices, subagents, hmm, ann):
    """One meta-agent on the exact training and validation windows it was given."""
    meta_path = os.path.join(rundir, "models", meta_dir)
    with open(os.path.join(meta_path, "config.json")) as f:
        cfg = json.load(f)
    meta = SAC.load(os.path.join(meta_path, "best_model"))
    cost = cfg.get("cost_pct", config.COST_PCT)
    mode = cfg.get("training_mode", "in_sample")
    use_hmm = hmm if cfg.get("use_hmm", True) else None
    rows = []

    if mode == "in_sample":
        cut = int(len(prices) * (1.0 - cfg.get("val_frac", 0.15)))
        windows = {"train": (prices.iloc[:cut], None, None),
                   "val": (prices.iloc[cut:], None, None)}
    else:
        windows = stream_windows(load_stream(rundir), prices, config.WARM_BARS)

    for window, (slice_, part, first) in windows.items():
        if mode == "crossfit":
            proposals, hmm_post = stream_arrays(part, slice_, AGENT_ORDER)
            env = meta_env_from_config(
                slice_, {}, None, cfg, cost, ann=ann, start_index=first, eval_mode=True,
                proposal_stream=proposals,
                hmm_stream=hmm_post if cfg.get("use_hmm", True) else None)
            label = "out_of_sample"
        else:
            env = meta_env_from_config(slice_, subagents, use_hmm, cfg, cost, ann=ann,
                                       start_index=first, eval_mode=True)
            label = "in_sample"
        rets, mkt = _meta_returns(meta, env)
        rows.append(_row(name, window, label, rets, mkt, ann))
        if mode != "crossfit":
            ew = run_allocator(EqualWeightAllocator(), subagents, slice_, ann, cost,
                               None, start_index=first)
            rows.append(_row("equal_weight", window, "in_sample",
                             np.asarray(ew["returns"], dtype=np.float64), mkt, ann))
    return rows


def main():
    p = argparse.ArgumentParser(
        description="Score saved models on their training and validation windows")
    p.add_argument("--rundir", required=True,
                   help="Run folder holding models/ (the output of run_all.py)")
    p.add_argument("--out", default=None,
                   help="Output CSV (default: <rundir>/window_sharpe.csv)")
    p.add_argument("--data_csv", default=config.DATA_CSV)
    args = p.parse_args()

    cfg_path = os.path.join(args.rundir, "models", "phase2", "config.json")
    if not os.path.isfile(os.path.join(args.rundir, "models", "phase2",
                                       "best_model.zip")):
        raise SystemExit(f"No trained models under {args.rundir}/models. Model files "
                         "are not in git, so run this on the machine that trained them.")
    with open(cfg_path) as f:
        cfg = json.load(f)
    ann = cfg.get("bars_per_year", 252)
    cost = cfg.get("cost_pct", config.COST_PCT)
    prices = load_prices(cfg.get("ticker", config.TICKER), cfg["train_start"],
                         cfg["train_end"], cfg.get("interval", config.INTERVAL),
                         data_csv=args.data_csv, verbose=False)
    subagents = load_subagents(args.rundir, verbose=False)
    hmm = HMMRegimeDetector.load(os.path.join(args.rundir, "models", "phase1",
                                              "hmm_model.pkl"))

    rows = agent_rows(subagents, prices, cost, ann)
    rows += meta_rows("meta_rl", "phase2", args.rundir, prices, subagents, hmm, ann)
    if os.path.isfile(os.path.join(args.rundir, "models", "phase2_leaked",
                                   "best_model.zip")):
        rows += meta_rows("meta_rl_leaked", "phase2_leaked", args.rundir, prices,
                          subagents, hmm, ann)
    if cfg.get("training_mode") == "crossfit":
        rows += stream_rows(load_stream(args.rundir), prices, cost, ann)

    table = pd.DataFrame(rows)
    out = args.out or os.path.join(args.rundir, "window_sharpe.csv")
    table.to_csv(out, index=False)
    print("\nSharpe by window")
    print(table.to_string(index=False))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
