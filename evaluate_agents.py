
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import SAC
from env.trading_env import TradingEnv, AGENT_PRESETS


BARS_PER_YEAR = {"1d": 252, "1h": 1638, "30m": 3276, "15m": 6552}

# ── Metrics ───────────────────────────────────────────────────────

def sharpe(r, ann=1638):
    s = r.std()
    return float(np.sqrt(ann) * r.mean() / s) if s > 1e-10 else 0.0

def sortino(r, ann=1638):
    down = r[r < 0]
    ds = down.std() if len(down) > 1 else 1e-10
    return float(np.sqrt(ann) * r.mean() / ds)

def max_drawdown(eq):
    peak = np.maximum.accumulate(eq)
    dd   = (peak - eq) / np.where(peak > 0, peak, 1e-10)
    return float(dd.max())

def calmar(r, eq, ann=1638):
    mdd = max_drawdown(eq)
    return float(r.mean() * ann / mdd) if mdd > 1e-6 else 0.0

def cvar_metric(r, alpha=0.95):
    """Historical CVaR (Expected Shortfall) at confidence alpha."""
    var = np.percentile(r, (1 - alpha) * 100)
    tail = r[r <= var]
    return float(tail.mean()) if len(tail) > 0 else 0.0

def omega_metric(r, threshold=0.0):
    """Omega ratio at given threshold."""
    gains  = (r[r > threshold] - threshold).sum()
    losses = (threshold - r[r < threshold]).sum()
    return float(gains / losses) if losses > 1e-10 else 2.0

def compute_metrics(r, eq, ann=1638):
    return {
        "Total ret (%)":  round((eq[-1] / eq[0] - 1) * 100, 2),
        "Ann. ret (%)":   round(r.mean() * ann * 100, 2),
        "Sharpe":         round(sharpe(r, ann), 3),
        "Sortino":        round(sortino(r, ann), 3),
        "MaxDD (%)":      round(max_drawdown(eq) * 100, 2),
        "Calmar":         round(calmar(r, eq, ann), 3),
        "CVaR 95% (%)":   round(cvar_metric(r) * 100, 3),
        "Omega":          round(omega_metric(r), 3),
        "Win rate (%)":   round((r > 0).mean() * 100, 1),
    }


# ── Regime labeling ────────────────────────────────────────────────

def label_regimes(prices, ann=1638):
    """
    Label each bar low_vol / mid_vol / high_vol using rolling realized vol tertiles.
    Window is ~20 trading days worth of bars, annualized with ann factor.
    Returns np.ndarray aligned to prices after warmup.
    """
    window = max(20, ann // 13)   # ~20 trading days worth of bars
    log_r = np.log(prices / prices.shift(1)).dropna()
    rv    = log_r.rolling(window).std() * np.sqrt(ann)
    rv    = rv.dropna()
    q33, q67 = rv.quantile(0.33), rv.quantile(0.67)
    labels = np.where(rv <= q33, "low_vol",
             np.where(rv <= q67, "mid_vol", "high_vol"))
    return labels


def regime_table(r, regimes, ann=1638):
    """Slice r by regime and compute per-regime metrics."""
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

def rollout(model, prices, lam, bars_per_year=1638, cost_pct=0.0002):
    env = TradingEnv(prices, lam=lam,
                     bars_per_year=bars_per_year, cost_pct=cost_pct,
                     eval_mode=True)   # no hard DD termination during eval
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
    }


# ── Buy-and-hold benchmark ────────────────────────────────────────

def buy_and_hold(prices):
    log_r = np.diff(np.log(prices.values)).astype(np.float32)
    eq    = np.cumprod(1 + log_r)
    eq    = np.concatenate([[1.0], eq])
    return {"returns": log_r, "equity": eq,
            "positions": np.ones(len(log_r), dtype=np.float32)}


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",   default="SPY")
    parser.add_argument("--start",    default="2025-11-01")
    parser.add_argument("--end",      default="2026-03-15")
    parser.add_argument("--interval", default="1h",
                        choices=list(BARS_PER_YEAR.keys()),
                        help="Bar interval (default: 1h)")
    parser.add_argument("--modeldir", default=".")
    parser.add_argument("--agents",   default=None, nargs="+",
                        choices=list(AGENT_PRESETS.keys()),
                        help="Subset of agents to evaluate (default: all found)")
    parser.add_argument("--cost_pct", default=0.0002, type=float,
                        help="One-way transaction cost — must match training value")
    args = parser.parse_args()

    ann = BARS_PER_YEAR[args.interval]

    print(f"\n=== Phase 0 Evaluation — {args.ticker} "
          f"({args.start} → {args.end}, interval={args.interval}) ===\n")

    # ── Download ──────────────────────────────────────────────────
    df = yf.download(args.ticker, start=args.start, end=args.end,
                     interval=args.interval, auto_adjust=True, progress=False)
    prices = df["Close"].squeeze().dropna()
    print(f"  {len(prices)} bars in test set.")

    # ── Regime labels ─────────────────────────────────────────────
    regimes = label_regimes(prices, ann)
    counts  = {r: (regimes == r).sum() for r in ["low_vol","mid_vol","high_vol"]}
    print(f"  Regime distribution: {counts}\n")

    # ── Determine which agents to evaluate ───────────────────────
    candidate_names = args.agents if args.agents else list(AGENT_PRESETS.keys())
    to_eval = []
    for name in candidate_names:
        path = os.path.join(args.modeldir, "models",
                            f"{name}_agent", "best_model.zip")
        if os.path.exists(path):
            to_eval.append(name)
        else:
            print(f"  [skip] {name} — model not found at {path}")

    if not to_eval:
        print("\n  No models found. Run train_agents.py first.")
        return

    # ── Rollout ───────────────────────────────────────────────────
    results = {}
    for name in to_eval:
        lam  = AGENT_PRESETS[name]
        path = os.path.join(args.modeldir, "models",
                            f"{name}_agent", "best_model")
        print(f"  Rolling out: {name} (λ={lam})...")
        model = SAC.load(path)
        results[name] = rollout(model, prices, lam, ann, args.cost_pct)

    # Add buy-and-hold benchmark
    results["buy_&_hold"] = buy_and_hold(prices)
    print(f"  Buy-and-hold benchmark added.")

    # ── Overall metrics ───────────────────────────────────────────
    sep = "=" * 80
    print(f"\n{sep}")
    print("  OVERALL PERFORMANCE")
    print(sep)
    overall = {name: compute_metrics(r["returns"], r["equity"], ann)
               for name, r in results.items()}
    df_overall = pd.DataFrame(overall).T
    print(df_overall.to_string())

    # ── Regime-conditional table ──────────────────────────────────
    print(f"\n{sep}")
    print("  REGIME-CONDITIONAL SHARPE  (key thesis table)")
    print(sep)

    # Compact cross-table: rows=agents, cols=regimes
    reg_rows = {}
    for name, r in results.items():
        ret = r["returns"]
        reg = np.array(regimes[:len(ret)])
        ret = ret[:len(reg)]
        for regime in ["low_vol", "mid_vol", "high_vol"]:
            mask = reg == regime
            rs   = ret[mask]
            s    = round(sharpe(rs, ann), 3) if len(rs) >= 5 else float("nan")
            reg_rows.setdefault(name, {})[regime] = s
    df_regime_sharpe = pd.DataFrame(reg_rows).T
    df_regime_sharpe.index.name = "agent"
    print(df_regime_sharpe.to_string())

    # Verbose per-agent regime breakdown
    print(f"\n{sep}")
    print("  REGIME-CONDITIONAL DETAIL (per agent)")
    print(sep)
    all_regime_dfs = {}
    for name, r in results.items():
        ret = r["returns"]
        reg = np.array(regimes[:len(ret)])
        ret = ret[:len(reg)]
        rt  = regime_table(ret, reg, ann)
        all_regime_dfs[name] = rt
        print(f"\n  [{name.upper()}]")
        print(rt.to_string())

    # ── Position correlation ──────────────────────────────────────
    print(f"\n{sep}")
    print("  POSITION CORRELATION MATRIX")
    print(sep)
    min_len = min(len(r["positions"]) for r in results.values())
    pos_df  = pd.DataFrame(
        {n: r["positions"][:min_len] for n, r in results.items()})
    print(pos_df.corr().round(3).to_string())
    print("\n  Low correlation = agents genuinely disagree = HLA adds value")

    # ── Save outputs ──────────────────────────────────────────────
    eq_df = pd.DataFrame(
        {n: r["equity"][:min_len + 1] for n, r in results.items()})
    eq_path = os.path.join(args.modeldir, "equity_curves.csv")
    eq_df.to_csv(eq_path, index=False)

    reg_path = os.path.join(args.modeldir, "regime_table.csv")
    df_regime_sharpe.to_csv(reg_path)

    print(f"\n  Saved → {eq_path}")
    print(f"  Saved → {reg_path}")

    # ── Summary: winner per regime ────────────────────────────────
    print(f"\n{sep}")
    print("  WINNER PER REGIME (highest Sharpe)")
    print(sep)
    for regime in ["low_vol", "mid_vol", "high_vol"]:
        col = df_regime_sharpe[regime].dropna()
        if col.empty:
            continue
        winner = col.idxmax()
        score  = col.max()
        print(f"  {regime:10s}  →  {winner:22s}  (Sharpe {score:.3f})")

    print(f"\n✓ Done. Run plot_results.py --csvpath equity_curves.csv "
          f"--ticker {args.ticker} --start {args.start} --end {args.end}")


if __name__ == "__main__":
    main()
