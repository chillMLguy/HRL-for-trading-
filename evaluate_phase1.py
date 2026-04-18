"""
Phase 1 evaluation — run all allocators on test data, compare metrics.

Usage
-----
  python evaluate_phase1.py --ticker "^DJI" \
      --test_start 2022-01-01 --test_end 2022-12-31 \
      --modeldir . --interval 1d

Outputs:
  phase1_equity_curves.csv
  phase1_weights.csv
  phase1_actions.csv
  phase1_metrics.csv
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import SAC

from env.trading_env import TradingEnv, AGENT_PRESETS
from evaluate_agents import (sharpe, sortino, max_drawdown, calmar,
                             cvar_metric, buy_and_hold, label_regimes,
                             compute_metrics, regime_table)
from regime.hmm_regime import HMMRegimeDetector
from regime.allocators import (EqualWeightAllocator,
                               VolatilityRegimeAllocator,
                               HMMAllocator, AGENT_ORDER, N_AGENTS)


BARS_PER_YEAR = {"1d": 252, "1h": 1638, "30m": 3276, "15m": 6552}


def run_allocator(alloc, agents, prices, ann, cost_pct,
                  cnn_model_path=None):
    """
    Run one allocator over the test period.

    Uses a single portfolio TradingEnv (λ=0) — agents are frozen advisors
    that propose actions from the shared observation.

    Returns dict with keys: returns, equity, positions, weights,
                             agent_actions
    """
    portfolio_env = TradingEnv(
        prices, lam=0.0, bars_per_year=ann,
        cost_pct=cost_pct, eval_mode=True,
        cnn_model_path=cnn_model_path,
    )
    obs, _ = portfolio_env.reset()
    warmup = portfolio_env.warmup

    equity_list = [1.0]
    returns_list = []
    weight_history = []
    action_history = []
    agent_action_history = {name: [] for name in AGENT_ORDER}

    done = False
    t = 0

    while not done:
        # Each agent proposes an action from the SAME observation
        proposed = {}
        for name in AGENT_ORDER:
            if name in agents:
                action, _ = agents[name].predict(obs, deterministic=True)
                proposed[name] = float(np.clip(action.flat[0], -1.0, 1.0))
            else:
                proposed[name] = 0.0  # missing agent → flat
            agent_action_history[name].append(proposed[name])

        # Get weights from allocator
        current_price_idx = warmup + t
        weights = alloc.get_weights(
            t, prices.iloc[:current_price_idx + 1],
            price_index=current_price_idx)
        weight_history.append(weights.copy())

        # Blend actions
        blended = sum(weights[i] * proposed[AGENT_ORDER[i]]
                      for i in range(N_AGENTS))
        blended = float(np.clip(blended, -1.0, 1.0))
        action_history.append(blended)

        # Step portfolio env
        obs, reward, done, truncated, info = portfolio_env.step(
            np.array([blended], dtype=np.float32))

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
    """Print a formatted comparison metrics table."""
    alloc_names = list(results.keys())
    metrics_list = {}
    for aname in alloc_names:
        r = results[aname]["returns"]
        eq = results[aname]["equity"]
        metrics_list[aname] = compute_metrics(r, eq, ann)

    df = pd.DataFrame(metrics_list)
    print(f"\n{'═'*70}")
    print(f"  Phase 1 — Allocator Comparison")
    print(f"{'═'*70}")
    print(df.to_string())
    print(f"{'═'*70}\n")
    return df


def sanity_checks(results):
    """Run and print sanity checks."""
    print("\n  Sanity Checks:")
    all_ok = True
    n_steps = None

    for aname, res in results.items():
        w = res["weights"]
        # Weight validity
        sums = w.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-5):
            print(f"    ✗  {aname}: weights do not sum to 1 "
                  f"(range {sums.min():.6f} – {sums.max():.6f})")
            all_ok = False
        if np.any(w < -1e-6):
            print(f"    ✗  {aname}: negative weights detected")
            all_ok = False

        # Blended action range
        pos = res["positions"]
        if np.any(np.abs(pos) > 1.0 + 1e-6):
            print(f"    ✗  {aname}: blended action outside [-1, 1]")
            all_ok = False

        # NaN / Inf
        eq = res["equity"]
        if np.any(np.isnan(eq)) or np.any(np.isinf(eq)):
            print(f"    ✗  {aname}: NaN/Inf in equity")
            all_ok = False

        # Step count alignment
        if n_steps is None:
            n_steps = len(res["returns"])
        elif len(res["returns"]) != n_steps:
            print(f"    ✗  {aname}: step count mismatch "
                  f"({len(res['returns'])} vs {n_steps})")
            all_ok = False

    if all_ok:
        print(f"    ✓  All checks passed ({n_steps} steps per allocator)")
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 — Evaluate all allocators on test data")
    parser.add_argument("--ticker",     default="^DJI")
    parser.add_argument("--test_start", default="2022-01-01")
    parser.add_argument("--test_end",   default="2022-12-31")
    parser.add_argument("--modeldir",   default=".")
    parser.add_argument("--interval",   default="1d",
                        choices=list(BARS_PER_YEAR.keys()))
    parser.add_argument("--cost_pct",   type=float, default=0.0005)
    parser.add_argument("--no_cnn",     action="store_true")
    parser.add_argument("--outdir",     default=".",
                        help="Where to save CSV results")
    args = parser.parse_args()

    ann = BARS_PER_YEAR[args.interval]
    cnn_path = None if args.no_cnn else None  # extend if CNN path needed

    # ── 1. Download test data ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase 1 Evaluation — {args.ticker}  "
          f"{args.test_start} → {args.test_end}")
    print(f"{'='*60}")

    print(f"\n  Downloading test data ...")
    df = yf.download(args.ticker, start=args.test_start,
                     end=args.test_end, interval=args.interval,
                     auto_adjust=True, progress=False)
    prices = df["Close"].squeeze().dropna()
    print(f"  {len(prices)} bars loaded.")

    # ── 2. Load Phase 0 agents ────────────────────────────────────
    print(f"\n  Loading Phase 0 agents ...")
    agents = {}
    for name in AGENT_ORDER:
        model_path = os.path.join(
            args.modeldir, "models", f"{name}_agent", "best_model")
        if os.path.isfile(model_path + ".zip"):
            agents[name] = SAC.load(model_path)
            print(f"    ✓  {name} (λ={AGENT_PRESETS[name]})")
        else:
            print(f"    ✗  {name} — NOT FOUND, will use action=0")

    if not agents:
        print("\n  ERROR: No agents loaded. Cannot evaluate.")
        return

    # ── 3. Initialize allocators ──────────────────────────────────
    print(f"\n  Initializing allocators ...")

    equal_alloc = EqualWeightAllocator()

    # Vol-regime: load thresholds from Phase 1 training
    th_path = os.path.join(args.modeldir, "models", "phase1",
                           "vol_thresholds.json")
    vol_alloc = VolatilityRegimeAllocator(
        prices, ann, thresholds_path=th_path)
    print(f"    Vol thresholds: q33={vol_alloc.q33:.6f}, "
          f"q67={vol_alloc.q67:.6f}")

    # HMM: load fitted model
    hmm_path = os.path.join(args.modeldir, "models", "phase1",
                            "hmm_model.pkl")
    hmm_detector = HMMRegimeDetector.load(hmm_path)
    hmm_alloc = HMMAllocator(hmm_detector)
    print(f"    HMM loaded ({hmm_detector.n_states} states)")

    allocators = [equal_alloc, vol_alloc, hmm_alloc]

    # ── 4. Run all allocators ─────────────────────────────────────
    results = {}
    for alloc in allocators:
        print(f"\n  Running {alloc.name} allocator ...")
        results[alloc.name] = run_allocator(
            alloc, agents, prices, ann, args.cost_pct, cnn_path)
        eq = results[alloc.name]["equity"]
        print(f"    Final equity: {eq[-1]:.4f}  "
              f"({(eq[-1]/eq[0]-1)*100:+.2f}%)")

    # ── 5. Buy-and-hold benchmark ─────────────────────────────────
    bh = buy_and_hold(prices)

    # ── 6. Sanity checks ──────────────────────────────────────────
    sanity_checks(results)

    # ── 7. Metrics table ──────────────────────────────────────────
    metrics_df = print_metrics_table(results, ann)

    # Add buy-and-hold to metrics
    bh_metrics = compute_metrics(bh["returns"], bh["equity"], ann)
    metrics_df["buy_and_hold"] = pd.Series(bh_metrics)
    print("  Buy & Hold:")
    for k, v in bh_metrics.items():
        print(f"    {k:20s}  {v}")

    # ── 8. Per-regime breakdown ───────────────────────────────────
    print(f"\n  Per-regime breakdown (vol-based labels):")
    regimes = label_regimes(prices, ann)
    for aname, res in results.items():
        r = res["returns"]
        # Align regime labels to returns length
        reg = regimes[:len(r)] if len(regimes) >= len(r) \
            else np.concatenate([regimes,
                                 np.full(len(r)-len(regimes), "mid_vol")])
        rt = regime_table(r, reg[:len(r)], ann)
        if not rt.empty:
            print(f"\n  [{aname}]")
            print(rt.to_string())

    # ── 9. Save CSVs ──────────────────────────────────────────────
    os.makedirs(args.outdir, exist_ok=True)
    alloc_names = list(results.keys())
    n_steps = len(results[alloc_names[0]]["returns"])

    # Equity curves
    eq_df = pd.DataFrame({"step": np.arange(n_steps + 1)})
    for aname in alloc_names:
        eq_df[aname] = results[aname]["equity"]
    # Align buy-and-hold equity to same length
    bh_eq = bh["equity"]
    warmup = results[alloc_names[0]]["warmup"]
    bh_aligned = bh_eq[warmup:warmup + n_steps + 1]
    if len(bh_aligned) == n_steps + 1:
        bh_aligned = bh_aligned / bh_aligned[0]  # rebase
        eq_df["buy_and_hold"] = bh_aligned
    eq_path = os.path.join(args.outdir, "phase1_equity_curves.csv")
    eq_df.to_csv(eq_path, index=False)
    print(f"\n  Saved → {eq_path}")

    # Weights
    rows = []
    for aname in alloc_names:
        w = results[aname]["weights"]
        for step in range(len(w)):
            row = {"step": step, "allocator": aname}
            for i, agent_name in enumerate(AGENT_ORDER):
                row[f"w_{agent_name}"] = w[step, i]
            rows.append(row)
    w_df = pd.DataFrame(rows)
    w_path = os.path.join(args.outdir, "phase1_weights.csv")
    w_df.to_csv(w_path, index=False)
    print(f"  Saved → {w_path}")

    # Actions
    rows = []
    for aname in alloc_names:
        res = results[aname]
        for step in range(len(res["positions"])):
            row = {"step": step, "allocator": aname,
                   "blended_action": res["positions"][step]}
            for agent_name in AGENT_ORDER:
                row[f"{agent_name}_action"] = \
                    res["agent_actions"][agent_name][step]
            rows.append(row)
    a_df = pd.DataFrame(rows)
    a_path = os.path.join(args.outdir, "phase1_actions.csv")
    a_df.to_csv(a_path, index=False)
    print(f"  Saved → {a_path}")

    # Metrics
    m_path = os.path.join(args.outdir, "phase1_metrics.csv")
    metrics_df.to_csv(m_path)
    print(f"  Saved → {m_path}")

    print(f"\n  Phase 1 evaluation complete.\n")


if __name__ == "__main__":
    main()
