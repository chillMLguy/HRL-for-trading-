"""
Phase 1 evaluation — fixed-rule allocators on test data (rewritten):
  * F3 — sub-agents observe their OWN shadow books (see env/shadow.py)
  * F8 — warm feature/HMM history; trading starts exactly at test_start
  * F9 — evaluation cost defaults to the cost the agents were trained with
  * adds the vol-targeted B&H benchmark (R2)

Outputs: phase1_equity_curves.csv / phase1_weights.csv / phase1_actions.csv /
phase1_metrics.csv
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from env.trading_env import TradingEnv, AGENT_PRESETS
from env.shadow import ShadowBooks
from evaluate_agents import (sharpe, buy_and_hold, vol_target_buy_and_hold,
                             label_regimes, compute_metrics, regime_table)
from regime.hmm_regime import HMMRegimeDetector
from regime.allocators import (EqualWeightAllocator,
                               VolatilityRegimeAllocator,
                               HMMAllocator, AGENT_ORDER, N_AGENTS)
from common.data import load_prices, warm_start_date, trading_start_index
import config


BARS_PER_YEAR = config.BARS_PER_YEAR


def trained_cost(modeldir, fallback=config.COST_PCT):
    """F9: read the training cost from any Phase 0 agent config."""
    for name in AGENT_ORDER:
        p = os.path.join(modeldir, "models", f"{name}_agent", "config.json")
        if os.path.isfile(p):
            with open(p) as f:
                return json.load(f).get("cost_pct", fallback)
    return fallback


def run_allocator(alloc, agents, prices, ann, cost_pct,
                  cnn_model_path=None, start_index=None):
    """
    Run one fixed-rule allocator over the test period.

    Sub-agents are frozen advisors proposing actions from their OWN shadow
    book observations (F3); the blend of their POSITION proposals (action ×
    per-agent cap) is executed in a single portfolio env.
    """
    portfolio_env = TradingEnv(
        prices, lam_var=0.0, max_pos=1.0, bars_per_year=ann,
        cost_pct=cost_pct, eval_mode=True,
        cnn_model_path=cnn_model_path, start_index=start_index,
    )
    portfolio_env.reset()
    warmup = portfolio_env.start_index

    caps = np.array([AGENT_PRESETS[n]["max_pos"] for n in AGENT_ORDER])
    books = ShadowBooks(N_AGENTS, cost_pct, caps, start_index=warmup)

    equity_list = [float(portfolio_env.initial_capital)]
    returns_list, weight_history, action_history = [], [], []
    agent_action_history = {name: [] for name in AGENT_ORDER}

    done = False
    t = 0
    while not done:
        # Proposals from each agent's own shadow observation (F3)
        prop_a = np.zeros(N_AGENTS)
        for i, name in enumerate(AGENT_ORDER):
            if name in agents:
                obs_i = books.agent_obs(portfolio_env, i)
                action, _ = agents[name].predict(obs_i, deterministic=True)
                prop_a[i] = float(np.clip(action.flat[0], -1.0, 1.0))
            agent_action_history[name].append(prop_a[i] * caps[i])

        current_price_idx = warmup + t
        weights = alloc.get_weights(
            t, prices.iloc[:current_price_idx + 1],
            price_index=current_price_idx)
        weight_history.append(weights.copy())

        # Blend POSITION proposals (action × per-agent cap)
        blended = float(np.clip(np.dot(weights, prop_a * caps), -1.0, 1.0))
        action_history.append(blended)

        _, _, done, truncated, info = portfolio_env.step(
            np.array([blended], dtype=np.float32))

        books.step(prop_a, info["price_ret"])
        books.mark_flat(portfolio_env.t)

        returns_list.append(info["net_ret"])
        equity_list.append(info["equity"])
        t += 1

    return {
        "returns":       np.array(returns_list, dtype=np.float32),
        "equity":        np.array(equity_list, dtype=np.float32),
        "positions":     np.array(action_history, dtype=np.float32),
        "weights":       np.array(weight_history, dtype=np.float64),
        "agent_actions": {k: np.array(v, dtype=np.float32)
                          for k, v in agent_action_history.items()},
        "warmup":        warmup,
    }


def print_metrics_table(results, ann):
    metrics_list = {name: compute_metrics(r["returns"], r["equity"], ann)
                    for name, r in results.items()}
    df = pd.DataFrame(metrics_list)
    print(f"\n{'═'*70}")
    print(f"  Phase 1 — Allocator Comparison")
    print(f"{'═'*70}")
    print(df.to_string())
    print(f"{'═'*70}\n")
    return df


def sanity_checks(results):
    print("\n  Sanity Checks:")
    all_ok = True
    n_steps = None
    for aname, res in results.items():
        w = res.get("weights")
        if w is not None and len(w):
            sums = w.sum(axis=1)
            if "w_cash" in res:            # meta with cash arm (R1)
                sums = sums + np.asarray(res["w_cash"])[:len(sums)]
            if not np.allclose(sums, 1.0, atol=1e-5):
                print(f"    ✗  {aname}: weights (incl. cash) do not sum to 1")
                all_ok = False
            if np.any(w < -1e-6):
                print(f"    ✗  {aname}: negative weights")
                all_ok = False
        eq = res["equity"]
        if np.any(np.isnan(eq)) or np.any(np.isinf(eq)):
            print(f"    ✗  {aname}: NaN/Inf in equity")
            all_ok = False
        if n_steps is None:
            n_steps = len(res["returns"])
        elif len(res["returns"]) != n_steps:
            print(f"    ✗  {aname}: step count mismatch")
            all_ok = False
    if all_ok:
        print(f"    ✓  All checks passed ({n_steps} steps per allocator)")
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 — Evaluate fixed-rule allocators on test data")
    parser.add_argument("--ticker",     default=config.TICKER)
    parser.add_argument("--test_start", default=config.TEST_START)
    parser.add_argument("--test_end",   default=config.TEST_END)
    parser.add_argument("--modeldir",   default=".")
    parser.add_argument("--interval",   default=config.INTERVAL,
                        choices=list(BARS_PER_YEAR.keys()))
    parser.add_argument("--cost_pct",   type=float, default=None,
                        help="Default: training cost from Phase 0 config")
    parser.add_argument("--no_cnn",     action="store_true")
    parser.add_argument("--outdir",     default=".")
    parser.add_argument("--data_csv",   default=None)
    args = parser.parse_args()

    ann = BARS_PER_YEAR[args.interval]
    cost = args.cost_pct if args.cost_pct is not None \
        else trained_cost(args.modeldir)

    cnn_path = None
    if not args.no_cnn:
        candidate = os.path.join(args.modeldir, "models", "cnn_features",
                                 "cnn_model.pt")
        if os.path.isfile(candidate):
            cnn_path = candidate

    print(f"\n{'='*60}")
    print(f"  Phase 1 Evaluation — {args.ticker}  "
          f"{args.test_start} → {args.test_end}  (cost {cost*1e4:.1f} bp)")
    print(f"{'='*60}")

    # F8: warm history
    warm_start = warm_start_date(args.test_start, config.WARM_BARS)
    prices = load_prices(args.ticker, warm_start, args.test_end,
                         args.interval, data_csv=args.data_csv)
    start_idx = trading_start_index(prices, args.test_start)
    print(f"  Warm history: {start_idx} bars; trading starts "
          f"{prices.index[start_idx].date()}")

    print(f"\n  Loading Phase 0 agents ...")
    agents = {}
    for name in AGENT_ORDER:
        model_path = os.path.join(
            args.modeldir, "models", f"{name}_agent", "best_model")
        if os.path.isfile(model_path + ".zip"):
            agents[name] = SAC.load(model_path)
            print(f"    ✓  {name}")
        else:
            print(f"    ✗  {name} — NOT FOUND, will use action=0")

    if not agents:
        print("\n  ERROR: No agents loaded.")
        return

    p1dir = os.path.join(args.modeldir, "models", "phase1")
    vol_alloc = VolatilityRegimeAllocator(
        prices, ann,
        thresholds_path=os.path.join(p1dir, "vol_thresholds.json"))
    hmm_detector = HMMRegimeDetector.load(
        os.path.join(p1dir, "hmm_model.pkl"))
    allocators = [EqualWeightAllocator(), vol_alloc,
                  HMMAllocator(hmm_detector)]

    results = {}
    for alloc in allocators:
        print(f"\n  Running {alloc.name} allocator ...")
        results[alloc.name] = run_allocator(
            alloc, agents, prices, ann, cost, cnn_path,
            start_index=start_idx)
        eq = results[alloc.name]["equity"]
        print(f"    Final equity: {eq[-1]:.4f}  "
              f"({(eq[-1]/eq[0]-1)*100:+.2f}%)")

    warmup = results["equal_weight"]["warmup"]
    n_steps = len(results["equal_weight"]["returns"])
    bh  = buy_and_hold(prices, warmup, n_steps)
    vbh = vol_target_buy_and_hold(prices, warmup, n_steps, ann,
                                  cost_pct=cost)

    sanity_checks(results)
    metrics_df = print_metrics_table(results, ann)
    metrics_df["buy_and_hold"] = pd.Series(
        compute_metrics(bh["returns"], bh["equity"], ann))
    metrics_df["vol_target_bh"] = pd.Series(
        compute_metrics(vbh["returns"], vbh["equity"], ann))
    print("  Benchmarks: buy_and_hold / vol_target_bh appended.")

    # Per-regime breakdown
    regimes = label_regimes(prices, ann)
    label_offset = max(20, ann // 13)
    regimes = regimes[max(0, warmup - label_offset):]
    for aname, res in results.items():
        r = res["returns"]
        reg = regimes[:len(r)] if len(regimes) >= len(r) \
            else np.concatenate([regimes,
                                 np.full(len(r) - len(regimes), "mid_vol")])
        rt = regime_table(r, reg[:len(r)], ann)
        if not rt.empty:
            print(f"\n  [{aname}]")
            print(rt.to_string())

    # Save CSVs
    os.makedirs(args.outdir, exist_ok=True)
    eq_df = pd.DataFrame({"step": np.arange(n_steps + 1)})
    for aname in results:
        eq_df[aname] = results[aname]["equity"]
    eq_df["buy_and_hold"] = bh["equity"]
    eq_df["vol_target_bh"] = vbh["equity"]
    eq_df.to_csv(os.path.join(args.outdir, "phase1_equity_curves.csv"),
                 index=False)

    rows = []
    for aname in results:
        w = results[aname]["weights"]
        for step in range(len(w)):
            row = {"step": step, "allocator": aname}
            for i, agent_name in enumerate(AGENT_ORDER):
                row[f"w_{agent_name}"] = w[step, i]
            rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(args.outdir, "phase1_weights.csv"), index=False)

    rows = []
    for aname in results:
        res = results[aname]
        for step in range(len(res["positions"])):
            row = {"step": step, "allocator": aname,
                   "blended_action": res["positions"][step]}
            for agent_name in AGENT_ORDER:
                row[f"{agent_name}_action"] = \
                    res["agent_actions"][agent_name][step]
            rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(args.outdir, "phase1_actions.csv"), index=False)

    metrics_df.to_csv(os.path.join(args.outdir, "phase1_metrics.csv"))
    print(f"\n  Phase 1 evaluation complete → {args.outdir}\n")


if __name__ == "__main__":
    main()
