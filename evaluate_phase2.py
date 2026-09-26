"""
Phase 2 evaluation — meta-RL allocator vs. baselines on test data (rewritten):
  * F8 — warm feature/HMM history; trading starts exactly at test_start
  * F9 — env is rebuilt from the phase-2 config.json (reward mode, cash arm,
         vol target, costs) so eval always matches training
  * adds vol-targeted B&H to the head-to-head

Outputs (under --outdir):
  phase2_equity_curves.csv   phase2_returns.csv (per-step, for aggregation)
  phase2_weights.csv         phase2_actions.csv
  phase2_metrics.csv
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from env.meta_env import MetaTradingEnv, load_subagents, detect_cnn_path
from evaluate_agents import (buy_and_hold, vol_target_buy_and_hold,
                             label_regimes, compute_metrics, sharpe)
from evaluate_phase1 import run_allocator, sanity_checks, trained_cost
from regime.hmm_regime import HMMRegimeDetector
from regime.allocators import (EqualWeightAllocator, VolatilityRegimeAllocator,
                               HMMAllocator, AGENT_ORDER, N_AGENTS)
from common.data import load_prices, warm_start_date, trading_start_index
import config


BARS_PER_YEAR = config.BARS_PER_YEAR
META_NAME = "meta_rl"


def run_meta_agent(meta_model, env):
    obs, _ = env.reset()
    equity_list = [float(env._env.initial_capital)]
    returns_list, action_history, weight_history = [], [], []
    lever_history, cash_history = [], []
    agent_action_history = {name: [] for name in AGENT_ORDER}

    done = False
    while not done:
        proposals = env._proposals_pos.copy()
        for i, name in enumerate(AGENT_ORDER):
            agent_action_history[name].append(float(proposals[i]))

        action, _ = meta_model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)

        returns_list.append(info["net_ret"])
        equity_list.append(info["equity"])
        action_history.append(info["blended"])
        weight_history.append(info["agent_weights"].copy())
        lever_history.append(info["lever"])
        cash_history.append(info["w_cash"])

    return {
        "returns":       np.array(returns_list, dtype=np.float32),
        "equity":        np.array(equity_list, dtype=np.float32),
        "positions":     np.array(action_history, dtype=np.float32),
        "weights":       np.array(weight_history, dtype=np.float64),
        "lever":         np.array(lever_history, dtype=np.float32),
        "w_cash":        np.array(cash_history, dtype=np.float32),
        "agent_actions": {k: np.array(v, dtype=np.float32)
                          for k, v in agent_action_history.items()},
        "warmup":        env.start_index,
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
    p.add_argument("--cost_pct",   type=float, default=None,
                   help="Default: cost from phase2 config.json")
    p.add_argument("--no_cnn",     action="store_true")
    p.add_argument("--meta_model", default="best_model",
                   choices=["best_model", "final_model"])
    p.add_argument("--data_csv",   default=None)
    args = p.parse_args()

    ann = BARS_PER_YEAR[args.interval]
    cnn_path = None if args.no_cnn else detect_cnn_path(args.modeldir)

    print(f"\n{'='*64}")
    print(f"  Phase 2 Evaluation — {args.ticker}  "
          f"{args.test_start} → {args.test_end}")
    print(f"{'='*64}")

    # ── 1. Data with warm history (F8) ────────────────────────────────────
    warm_start = warm_start_date(args.test_start, config.WARM_BARS)
    prices = load_prices(args.ticker, warm_start, args.test_end,
                         args.interval, data_csv=args.data_csv)
    start_idx = trading_start_index(prices, args.test_start)
    print(f"  Warm history: {start_idx} bars; trading starts "
          f"{prices.index[start_idx].date()}  CNN: "
          f"{'on' if cnn_path else 'off'}")

    # ── 2. Frozen sub-agents ───────────────────────────────────────────────
    print("\n  Loading frozen sub-agents ...")
    subagents = load_subagents(args.modeldir)
    if not subagents:
        print("\n  ERROR: no sub-agents found. Run train_agents.py first.")
        return

    # ── 3. HMM + meta-agent + config ──────────────────────────────────────
    p1dir = os.path.join(args.modeldir, "models", "phase1")
    hmm = HMMRegimeDetector.load(os.path.join(p1dir, "hmm_model.pkl"))

    meta_path = os.path.join(args.modeldir, "models", "phase2",
                             args.meta_model)
    if not os.path.isfile(meta_path + ".zip"):
        print(f"\n  ERROR: meta-agent not found at {meta_path}.zip.")
        return
    meta_model = SAC.load(meta_path)
    print(f"  Meta-agent loaded ← {meta_path}.zip")

    cfg_path = os.path.join(args.modeldir, "models", "phase2", "config.json")
    meta_cfg = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            meta_cfg = json.load(f)

    # F9: cost = the cost the meta-agent was trained with, unless overridden
    cost = args.cost_pct if args.cost_pct is not None \
        else meta_cfg.get("cost_pct", trained_cost(args.modeldir))
    print(f"  Cost/side: {cost*1e4:.1f} bp  reward={meta_cfg.get('meta_reward')}"
          f"  cash_arm={meta_cfg.get('use_cash')}  "
          f"vol_target={meta_cfg.get('vol_target')}")

    use_hmm_in_env = hmm if meta_cfg.get("use_hmm", True) else None

    # ── 4. Run the meta-agent (env rebuilt from training config) ──────────
    print("\n  Running meta-RL allocator ...")
    meta_env = MetaTradingEnv(
        prices, subagents, hmm_detector=use_hmm_in_env, ann=ann,
        cost_pct=cost,
        use_cash=meta_cfg.get("use_cash", config.META_USE_CASH),
        vol_target=meta_cfg.get("vol_target", config.VOL_TARGET),
        vol_target_cap=meta_cfg.get("vol_target_cap", config.VOL_TARGET_CAP),
        meta_reward=meta_cfg.get("meta_reward", config.META_REWARD),
        dsr_eta=meta_cfg.get("dsr_eta", config.DSR_ETA),
        lam_var_meta=meta_cfg.get("lam_var_meta", 12.0),
        alpha=meta_cfg.get("alpha", 0.05), beta=meta_cfg.get("beta", 0.05),
        dd_free=meta_cfg.get("dd_free", 0.03),
        dd_max=meta_cfg.get("dd_max", 0.10),
        logit_scale=meta_cfg.get("logit_scale", 2.0),
        perf_window=meta_cfg.get("perf_window", 20),
        start_index=start_idx, eval_mode=True, cnn_model_path=cnn_path,
    )
    results = {META_NAME: run_meta_agent(meta_model, meta_env)}
    eq = results[META_NAME]["equity"]
    print(f"    Final equity: {eq[-1]:.4f}  ({(eq[-1]/eq[0]-1)*100:+.2f}%)")

    # ── 5. Phase 1 baselines on the same window ───────────────────────────
    th_path = os.path.join(p1dir, "vol_thresholds.json")
    allocators = [
        EqualWeightAllocator(),
        VolatilityRegimeAllocator(prices, ann, thresholds_path=th_path),
        HMMAllocator(hmm),
    ]
    for alloc in allocators:
        print(f"  Running {alloc.name} baseline ...")
        results[alloc.name] = run_allocator(
            alloc, subagents, prices, ann, cost, cnn_path,
            start_index=start_idx)
        eq = results[alloc.name]["equity"]
        print(f"    Final equity: {eq[-1]:.4f}  "
              f"({(eq[-1]/eq[0]-1)*100:+.2f}%)")

    strat_names = list(results.keys())

    # ── 6. Benchmarks ──────────────────────────────────────────────────────
    warmup  = results[META_NAME]["warmup"]
    n_steps = len(results[META_NAME]["returns"])
    bh  = buy_and_hold(prices, warmup, n_steps)
    vbh = vol_target_buy_and_hold(prices, warmup, n_steps, ann,
                                  cost_pct=cost)

    # ── 7. Sanity + metrics ───────────────────────────────────────────────
    sanity_checks(results)
    metrics = {name: compute_metrics(r["returns"], r["equity"], ann)
               for name, r in results.items()}
    metrics["buy_and_hold"]  = compute_metrics(bh["returns"], bh["equity"], ann)
    metrics["vol_target_bh"] = compute_metrics(vbh["returns"], vbh["equity"],
                                               ann)
    metrics_df = pd.DataFrame(metrics)
    print(f"\n{'═'*78}")
    print("  Phase 2 — Meta-RL vs. Baselines")
    print(f"{'═'*78}")
    print(metrics_df.to_string())
    print(f"{'═'*78}")

    mr = results[META_NAME]
    print(f"\n  meta_rl diagnostics: mean w_cash={mr['w_cash'].mean():.3f}  "
          f"mean lever={mr['lever'].mean():.3f}  "
          f"mean net pos={mr['positions'].mean():+.3f}  "
          f"mean |pos|={np.abs(mr['positions']).mean():.3f}")

    # ── 8. Per-regime Sharpe ──────────────────────────────────────────────
    print("\n  Per-regime Sharpe (vol-based labels):")
    regimes = label_regimes(prices, ann)
    label_offset = max(20, ann // 13)
    regimes = regimes[max(0, warmup - label_offset):]
    reg_rows = {}
    all_results = dict(results)
    all_results["buy_and_hold"] = bh
    all_results["vol_target_bh"] = vbh
    for name, res in all_results.items():
        r = res["returns"]
        reg = np.array(regimes[:len(r)])
        r = r[:len(reg)]
        for reg_lbl in ["low_vol", "mid_vol", "high_vol"]:
            rs = r[reg == reg_lbl]
            reg_rows.setdefault(name, {})[reg_lbl] = (
                round(sharpe(rs, ann), 3) if len(rs) >= 5 else float("nan"))
    print(pd.DataFrame(reg_rows).T.to_string())

    # ── 9. Save CSVs ──────────────────────────────────────────────────────
    os.makedirs(args.outdir, exist_ok=True)

    # Dates of the traded window (for aggregation across folds)
    trade_dates = prices.index[warmup + 1: warmup + 1 + n_steps]

    eq_df = pd.DataFrame({"step": np.arange(n_steps + 1)})
    for name in strat_names:
        eq_df[name] = results[name]["equity"]
    eq_df["buy_and_hold"]  = bh["equity"]
    eq_df["vol_target_bh"] = vbh["equity"]
    eq_df.to_csv(os.path.join(args.outdir, "phase2_equity_curves.csv"),
                 index=False)

    # Per-step returns — the file aggregate_results.py pools across folds
    ret_df = pd.DataFrame({"date": trade_dates[:n_steps]})
    for name in strat_names:
        ret_df[name] = results[name]["returns"][:n_steps]
    ret_df["buy_and_hold"]  = bh["returns"][:n_steps]
    ret_df["vol_target_bh"] = vbh["returns"][:n_steps]
    ret_df.to_csv(os.path.join(args.outdir, "phase2_returns.csv"),
                  index=False)

    rows = []
    for name in strat_names:
        w = results[name]["weights"]
        for step in range(len(w)):
            row = {"step": step, "allocator": name}
            for i, agent_name in enumerate(AGENT_ORDER):
                row[f"w_{agent_name}"] = w[step, i]
            if name == META_NAME:
                row["w_cash"] = results[name]["w_cash"][step]
                row["lever"]  = results[name]["lever"][step]
            rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(args.outdir, "phase2_weights.csv"), index=False)

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
    pd.DataFrame(rows).to_csv(
        os.path.join(args.outdir, "phase2_actions.csv"), index=False)

    metrics_df.to_csv(os.path.join(args.outdir, "phase2_metrics.csv"))

    # F9: record exactly how this evaluation was run
    with open(os.path.join(args.outdir, "phase2_eval_config.json"), "w") as f:
        json.dump({"ticker": args.ticker, "test_start": args.test_start,
                   "test_end": args.test_end, "cost_pct": cost,
                   "meta_model": args.meta_model, "warm_bars": start_idx,
                   "meta_cfg": meta_cfg}, f, indent=2)

    print(f"\n  Phase 2 evaluation complete → {args.outdir}\n")


if __name__ == "__main__":
    main()
