"""
Phase 2 evaluation — meta-RL allocator vs. the Phase 1 baselines on test data.

Rolls out the learned meta-agent and, for a head-to-head comparison on the same
window, re-runs the Phase 1 allocators (equal-weight, vol-regime, fixed-matrix
HMM) plus a buy-and-hold benchmark.  All series share the agents' trading window
so the CSVs line up step-for-step.

Usage
-----
  python evaluate_phase2.py --ticker "^DJI" \
      --test_start 2024-01-01 --test_end 2024-12-31 --modeldir . --outdir .

Outputs:
  phase2_equity_curves.csv   (step + one column per strategy + buy_and_hold)
  phase2_weights.csv         (per-step agent weights, incl. meta_rl)
  phase2_actions.csv         (blended + per-agent proposed actions)
  phase2_metrics.csv         (full metric table)
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import SAC

from env.meta_env import (MetaTradingEnv, load_subagents, detect_cnn_path)
from evaluate_agents import (buy_and_hold, label_regimes, compute_metrics,
                             regime_table, sharpe)
from evaluate_phase1 import run_allocator, sanity_checks
from regime.hmm_regime import HMMRegimeDetector
from regime.allocators import (EqualWeightAllocator, VolatilityRegimeAllocator,
                               HMMAllocator, AGENT_ORDER, N_AGENTS)
import config


BARS_PER_YEAR = config.BARS_PER_YEAR
META_NAME = "meta_rl"


def run_meta_agent(meta_model, env):
    """
    Roll out the meta-agent over `env` (a MetaTradingEnv in eval_mode).

    Returns the same dict shape as evaluate_phase1.run_allocator so the
    downstream metrics / CSV / plotting code is shared.
    """
    obs, _ = env.reset()
    equity_list = [float(env._env.initial_capital)]
    returns_list, action_history, weight_history = [], [], []
    agent_action_history = {name: [] for name in AGENT_ORDER}

    done = False
    while not done:
        # Proposals used for THIS step's blend (env updates them after step()).
        proposals = env._proposals.copy()
        for i, name in enumerate(AGENT_ORDER):
            agent_action_history[name].append(float(proposals[i]))

        action, _ = meta_model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)

        returns_list.append(info["net_ret"])
        equity_list.append(info["equity"])
        action_history.append(info["blended"])
        weight_history.append(info["weights"].copy())

    return {
        "returns":       np.array(returns_list, dtype=np.float32),
        "equity":        np.array(equity_list, dtype=np.float32),
        "positions":     np.array(action_history, dtype=np.float32),
        "weights":       np.array(weight_history, dtype=np.float64),
        "agent_actions": {k: np.array(v, dtype=np.float32)
                          for k, v in agent_action_history.items()},
        "warmup":        env.warmup,
    }


def main():
    p = argparse.ArgumentParser(
        description="Phase 2 — Evaluate meta-RL allocator vs. baselines")
    p.add_argument("--ticker",     default=config.TICKER)
    p.add_argument("--test_start", default=config.TEST_START)
    p.add_argument("--test_end",   default=config.TEST_END)
    p.add_argument("--modeldir",   default=".")
    p.add_argument("--outdir",     default=".")
    p.add_argument("--interval",   default=config.INTERVAL,
                   choices=list(BARS_PER_YEAR.keys()))
    p.add_argument("--cost_pct",   type=float, default=config.COST_PCT)
    p.add_argument("--no_cnn",     action="store_true")
    p.add_argument("--meta_model", default="best_model",
                   choices=["best_model", "final_model"])
    args = p.parse_args()

    ann = BARS_PER_YEAR[args.interval]
    cnn_path = None if args.no_cnn else detect_cnn_path(args.modeldir)

    print(f"\n{'='*64}")
    print(f"  Phase 2 Evaluation — {args.ticker}  "
          f"{args.test_start} → {args.test_end}")
    print(f"{'='*64}")

    # ── 1. Data ───────────────────────────────────────────────────────────
    print("\n  Downloading test data ...")
    df = yf.download(args.ticker, start=args.test_start, end=args.test_end,
                     interval=args.interval, auto_adjust=True, progress=False)
    prices = df["Close"].squeeze().dropna()
    print(f"  {len(prices)} bars loaded.  CNN features: "
          f"{'enabled' if cnn_path else 'disabled'}")

    # ── 2. Frozen sub-agents ───────────────────────────────────────────────
    print("\n  Loading frozen sub-agents ...")
    subagents = load_subagents(args.modeldir)
    if not subagents:
        print("\n  ERROR: no sub-agents found. Run train_agents.py first.")
        return

    # ── 3. HMM + meta-agent + Phase 1 allocators ──────────────────────────
    p1dir = os.path.join(args.modeldir, "models", "phase1")
    hmm = HMMRegimeDetector.load(os.path.join(p1dir, "hmm_model.pkl"))

    meta_path = os.path.join(args.modeldir, "models", "phase2",
                             args.meta_model)
    if not os.path.isfile(meta_path + ".zip"):
        print(f"\n  ERROR: meta-agent not found at {meta_path}.zip. "
              f"Run train_phase2.py first.")
        return
    meta_model = SAC.load(meta_path)
    print(f"  Meta-agent loaded ← {meta_path}.zip")

    # Read meta hyper-parameters so the eval env matches training.
    cfg_path = os.path.join(args.modeldir, "models", "phase2", "config.json")
    meta_cfg = {}
    if os.path.isfile(cfg_path):
        import json
        with open(cfg_path) as f:
            meta_cfg = json.load(f)

    # ── 4. Run the meta-agent ──────────────────────────────────────────────
    print("\n  Running meta-RL allocator ...")
    meta_env = MetaTradingEnv(
        prices, subagents, hmm_detector=hmm, ann=ann, cost_pct=args.cost_pct,
        lam_meta=meta_cfg.get("lam_meta", 0.5),
        alpha=meta_cfg.get("alpha", 0.05), beta=meta_cfg.get("beta", 0.05),
        dd_free=meta_cfg.get("dd_free", 0.03),
        dd_max=meta_cfg.get("dd_max", 0.10),
        logit_scale=meta_cfg.get("logit_scale", 2.0),
        perf_window=meta_cfg.get("perf_window", 20),
        eval_mode=True, cnn_model_path=cnn_path,
    )
    results = {META_NAME: run_meta_agent(meta_model, meta_env)}
    eq = results[META_NAME]["equity"]
    print(f"    Final equity: {eq[-1]:.4f}  ({(eq[-1]/eq[0]-1)*100:+.2f}%)")

    # ── 5. Run Phase 1 baselines on the same window ───────────────────────
    th_path = os.path.join(p1dir, "vol_thresholds.json")
    allocators = [
        EqualWeightAllocator(),
        VolatilityRegimeAllocator(prices, ann, thresholds_path=th_path),
        HMMAllocator(hmm),
    ]
    for alloc in allocators:
        print(f"  Running {alloc.name} baseline ...")
        results[alloc.name] = run_allocator(
            alloc, subagents, prices, ann, args.cost_pct, cnn_path)
        eq = results[alloc.name]["equity"]
        print(f"    Final equity: {eq[-1]:.4f}  "
              f"({(eq[-1]/eq[0]-1)*100:+.2f}%)")

    strat_names = list(results.keys())

    # ── 6. Buy-and-hold benchmark (aligned to the trading window) ──────────
    warmup  = results[META_NAME]["warmup"]
    n_steps = len(results[META_NAME]["returns"])
    bh = buy_and_hold(prices, warmup=warmup, n_steps=n_steps)

    # ── 7. Sanity checks ───────────────────────────────────────────────────
    sanity_checks(results)

    # ── 8. Metrics table ───────────────────────────────────────────────────
    metrics = {name: compute_metrics(r["returns"], r["equity"], ann)
               for name, r in results.items()}
    metrics["buy_and_hold"] = compute_metrics(bh["returns"], bh["equity"], ann)
    metrics_df = pd.DataFrame(metrics)
    print(f"\n{'═'*78}")
    print("  Phase 2 — Meta-RL vs. Baselines")
    print(f"{'═'*78}")
    print(metrics_df.to_string())
    print(f"{'═'*78}")

    # ── 9. Per-regime breakdown (vol-based labels) ─────────────────────────
    print("\n  Per-regime Sharpe (vol-based labels):")
    regimes = label_regimes(prices, ann)
    label_offset = max(20, ann // 13)
    regimes = regimes[max(0, warmup - label_offset):]
    reg_rows = {}
    for name, res in results.items():
        r = res["returns"]
        reg = np.array(regimes[:len(r)])
        r = r[:len(reg)]
        for reg_lbl in ["low_vol", "mid_vol", "high_vol"]:
            mask = reg == reg_lbl
            rs = r[mask]
            reg_rows.setdefault(name, {})[reg_lbl] = (
                round(sharpe(rs, ann), 3) if len(rs) >= 5 else float("nan"))
    print(pd.DataFrame(reg_rows).T.to_string())

    # ── 10. Save CSVs ──────────────────────────────────────────────────────
    os.makedirs(args.outdir, exist_ok=True)

    # Equity curves
    eq_df = pd.DataFrame({"step": np.arange(n_steps + 1)})
    for name in strat_names:
        eq_df[name] = results[name]["equity"]
    if len(bh["equity"]) == n_steps + 1:
        eq_df["buy_and_hold"] = bh["equity"]
    eq_path = os.path.join(args.outdir, "phase2_equity_curves.csv")
    eq_df.to_csv(eq_path, index=False)

    # Weights
    rows = []
    for name in strat_names:
        w = results[name]["weights"]
        for step in range(len(w)):
            row = {"step": step, "allocator": name}
            for i, agent_name in enumerate(AGENT_ORDER):
                row[f"w_{agent_name}"] = w[step, i]
            rows.append(row)
    w_path = os.path.join(args.outdir, "phase2_weights.csv")
    pd.DataFrame(rows).to_csv(w_path, index=False)

    # Actions
    rows = []
    for name in strat_names:
        res = results[name]
        for step in range(len(res["positions"])):
            row = {"step": step, "allocator": name,
                   "blended_action": res["positions"][step]}
            for agent_name in AGENT_ORDER:
                row[f"{agent_name}_action"] = \
                    res["agent_actions"][agent_name][step]
            rows.append(row)
    a_path = os.path.join(args.outdir, "phase2_actions.csv")
    pd.DataFrame(rows).to_csv(a_path, index=False)

    # Metrics
    m_path = os.path.join(args.outdir, "phase2_metrics.csv")
    metrics_df.to_csv(m_path)

    for pth in (eq_path, w_path, a_path, m_path):
        print(f"  Saved → {pth}")
    print("\n  Phase 2 evaluation complete.\n")


if __name__ == "__main__":
    main()
