import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from env.trading_env import TradingEnv, AGENT_PRESETS
from common.data import load_prices, warm_start_date, trading_start_index
from common.stats import sharpe, max_drawdown, cvar_metric, compute_metrics
import config


# ── Regime labeling (diagnostics) ──────────────────────────────────

def label_regimes(prices, ann=252, q33=None, q67=None):
    """
    Label each bar low/mid/high vol.  Pass q33/q67 from the TRAINING period
    for out-of-sample labels; falls back to in-window tertiles (diagnostics
    only) when not provided.
    """
    window = max(20, ann // 13)
    log_r = np.log(prices / prices.shift(1)).dropna()
    rv    = log_r.rolling(window).std() * np.sqrt(ann)
    rv    = rv.dropna()
    if q33 is None or q67 is None:
        q33, q67 = rv.quantile(0.33), rv.quantile(0.67)
    labels = np.where(rv <= q33, "low_vol",
             np.where(rv <= q67, "mid_vol", "high_vol"))
    return labels


def regime_table(r, regimes, ann=252):
    rows = []
    for reg in ["low_vol", "mid_vol", "high_vol"]:
        mask = regimes == reg
        rs   = r[mask[:len(r)]]
        if len(rs) < 5:
            continue
        eq = np.cumprod(1 + rs)
        rows.append({
            "Regime":        reg,
            "Bars":          int(mask.sum()),
            "Ann. ret (%)":  round(rs.mean() * ann * 100, 2),
            "Sharpe":        round(sharpe(rs, ann), 3),
            "MaxDD (%)":     round(max_drawdown(eq) * 100, 2),
            "CVaR 95% (%)":  round(cvar_metric(rs) * 100, 3),
            "Win rate (%)":  round((rs > 0).mean() * 100, 1),
        })
    return pd.DataFrame(rows).set_index("Regime")


# ── Rollout ────────────────────────────────────────────────────────

def rollout(model, prices, agent_name, bars_per_year=252, cost_pct=0.0002,
            cnn_model_path=None, start_index=None):
    env = TradingEnv(prices, preset=agent_name,
                     bars_per_year=bars_per_year, cost_pct=cost_pct,
                     eval_mode=True, cnn_model_path=cnn_model_path,
                     start_index=start_index)
    obs, _ = env.reset()
    returns, equity, positions = [], [float(env.initial_capital)], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        returns.append(info["net_ret"])
        equity.append(info["equity"])
        positions.append(info["position"])
    return {
        "returns":   np.array(returns,   dtype=np.float32),
        "equity":    np.array(equity,    dtype=np.float32),
        "positions": np.array(positions, dtype=np.float32),
        "warmup":    int(env.start_index),
    }


# ── Benchmarks ────────────────────────────────────────────────────

def buy_and_hold(prices, warmup=0, n_steps=None):
    """B&H equity over [warmup, warmup+n_steps], rebased to 1.0 (simple rets)."""
    p = np.asarray(prices.values if hasattr(prices, "values") else prices,
                   dtype=np.float64)
    simple_r = np.diff(p) / p[:-1]
    eq_full = np.concatenate([[1.0], np.cumprod(1.0 + simple_r)])

    if n_steps is None:
        eq_window = eq_full
        ret_window = simple_r
    else:
        end = warmup + n_steps + 1
        eq_window = eq_full[warmup:end]
        ret_window = simple_r[warmup:warmup + n_steps]

    eq_window = (eq_window / eq_window[0]).astype(np.float32)
    return {
        "returns":   ret_window.astype(np.float32),
        "equity":    eq_window,
        "positions": np.ones(len(ret_window), dtype=np.float32),
        "warmup":    int(warmup),
    }


def vol_target_buy_and_hold(prices, warmup, n_steps, ann=252,
                            vol_target=config.VOL_TARGET,
                            cap=config.VOL_TARGET_CAP,
                            cost_pct=config.COST_PCT):
    """
    R2 benchmark: long-only position scaled to `vol_target` by trailing EWMA
    vol (Moreira & Muir style).  A much tougher — and more honest — benchmark
    for the RL stack than plain B&H.
    """
    from env.meta_env import ewma_vol_series
    p = np.asarray(prices.values if hasattr(prices, "values") else prices,
                   dtype=np.float64)
    simple_r = np.diff(p) / p[:-1]
    vol = ewma_vol_series(prices, bars_per_year=ann)

    pos_prev, rets, poss = 0.0, [], []
    for k in range(n_steps):
        t = warmup + k
        v = vol[t]
        pos = float(np.clip(vol_target / v, 0.0, cap)) \
            if np.isfinite(v) and v > 1e-4 else 1.0
        cost = abs(pos - pos_prev) * cost_pct
        rets.append(pos * simple_r[t] - cost)
        poss.append(pos)
        pos_prev = pos
    rets = np.asarray(rets, dtype=np.float32)
    eq = np.concatenate([[1.0], np.cumprod(1.0 + rets)]).astype(np.float32)
    return {"returns": rets, "equity": eq,
            "positions": np.asarray(poss, dtype=np.float32),
            "warmup": int(warmup)}


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",   default=config.TICKER)
    parser.add_argument("--start",    default=config.TEST_START)
    parser.add_argument("--end",      default=config.TEST_END)
    parser.add_argument("--interval", default=config.INTERVAL,
                        choices=list(config.BARS_PER_YEAR.keys()))
    parser.add_argument("--modeldir", default=".")
    parser.add_argument("--agents",   default=None, nargs="+",
                        choices=list(AGENT_PRESETS.keys()))
    parser.add_argument("--cost_pct", default=None, type=float,
                        help="Default: the cost each agent was TRAINED with "
                             "(from its config.json)")
    parser.add_argument("--no_cnn", action="store_true")
    parser.add_argument("--data_csv", default=None)
    args = parser.parse_args()

    ann = config.BARS_PER_YEAR[args.interval]

    cnn_model_path = None
    if not args.no_cnn:
        cnn_path = os.path.join(args.modeldir, "models", "cnn_features",
                                "cnn_model.pt")
        if os.path.exists(cnn_path):
            cnn_model_path = cnn_path

    print(f"\n=== Phase 0 Evaluation — {args.ticker} "
          f"({args.start} → {args.end}) ===")

    # F8: warm history so the full test window is traded
    warm_start = warm_start_date(args.start, config.WARM_BARS)
    prices = load_prices(args.ticker, warm_start, args.end, args.interval,
                         data_csv=args.data_csv)
    start_idx = trading_start_index(prices, args.start)
    print(f"  Warm history: {start_idx} bars; trading starts "
          f"{prices.index[start_idx].date()}")

    candidate_names = args.agents if args.agents else list(AGENT_PRESETS.keys())
    results = {}
    for name in candidate_names:
        path = os.path.join(args.modeldir, "models",
                            f"{name}_agent", "best_model")
        if not os.path.isfile(path + ".zip"):
            print(f"  [skip] {name} — model not found")
            continue
        # F9: evaluate at the cost the agent was trained with, unless overridden
        cost = args.cost_pct
        cfg_path = os.path.join(args.modeldir, "models",
                                f"{name}_agent", "config.json")
        if cost is None:
            cost = config.COST_PCT
            if os.path.isfile(cfg_path):
                with open(cfg_path) as f:
                    cost = json.load(f).get("cost_pct", cost)
        print(f"  Rolling out: {name} (cost {cost*1e4:.1f} bp)...")
        model = SAC.load(path)
        results[name] = rollout(model, prices, name, ann, cost,
                                cnn_model_path, start_index=start_idx)

    if not results:
        print("\n  No models found. Run train_agents.py first.")
        return

    sample = next(iter(results.values()))
    warmup, n_steps = sample["warmup"], len(sample["positions"])
    results["buy_&_hold"] = buy_and_hold(prices, warmup, n_steps)
    results["vol_target_b&h"] = vol_target_buy_and_hold(
        prices, warmup, n_steps, ann)

    sep = "=" * 80
    print(f"\n{sep}\n  OVERALL PERFORMANCE\n{sep}")
    overall = {name: compute_metrics(r["returns"], r["equity"], ann)
               for name, r in results.items()}
    print(pd.DataFrame(overall).T.to_string())

    # Position-size ladder check (R4 diagnostic)
    print(f"\n  Exposure ladder (mean |position|):")
    for name in candidate_names:
        if name in results:
            print(f"    {name:14s} {np.abs(results[name]['positions']).mean():.3f}")

    # Regime-conditional Sharpe
    regimes = label_regimes(prices, ann)
    label_offset = max(20, ann // 13)
    regimes = regimes[max(0, warmup - label_offset):]
    reg_rows = {}
    for name, r in results.items():
        ret = r["returns"]
        reg = np.array(regimes[:len(ret)])
        ret = ret[:len(reg)]
        for regime in ["low_vol", "mid_vol", "high_vol"]:
            rs = ret[reg == regime]
            reg_rows.setdefault(name, {})[regime] = \
                round(sharpe(rs, ann), 3) if len(rs) >= 5 else float("nan")
    df_regime_sharpe = pd.DataFrame(reg_rows).T
    print(f"\n{sep}\n  REGIME-CONDITIONAL SHARPE\n{sep}")
    print(df_regime_sharpe.to_string())

    # Proposal correlation (diversity diagnostic)
    min_len = min(len(r["positions"]) for r in results.values())
    pos_df  = pd.DataFrame(
        {n: r["positions"][:min_len] for n, r in results.items()})
    print(f"\n  POSITION CORRELATION MATRIX")
    print(pos_df.corr().round(3).to_string())

    eq_df = pd.DataFrame(
        {n: r["equity"][:min_len + 1] for n, r in results.items()})
    eq_path = os.path.join(args.modeldir, "equity_curves.csv")
    eq_df.to_csv(eq_path, index=False)
    df_regime_sharpe.to_csv(os.path.join(args.modeldir, "regime_table.csv"))
    print(f"\n  Saved → {eq_path}")


if __name__ == "__main__":
    main()
