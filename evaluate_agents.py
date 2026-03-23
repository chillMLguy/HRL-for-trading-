import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from stable_baselines3 import SAC

from env.trading_env import TradingEnv, AgentType


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def sharpe(returns: np.ndarray, rf: float = 0.0, ann: int = 252) -> float:
    """Annualized Sharpe ratio."""
    excess = returns - rf / ann
    if excess.std() < 1e-10:
        return 0.0
    return float(np.sqrt(ann) * excess.mean() / excess.std())


def sortino(returns: np.ndarray, rf: float = 0.0, ann: int = 252) -> float:
    """Annualized Sortino ratio (uses downside deviation only)."""
    excess = returns - rf / ann
    downside = excess[excess < 0]
    dd_std = downside.std() if len(downside) > 1 else 1e-10
    return float(np.sqrt(ann) * excess.mean() / dd_std)


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown."""
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / np.where(peak > 0, peak, 1e-10)
    return float(dd.max())


def calmar(returns: np.ndarray, equity_curve: np.ndarray, ann: int = 252) -> float:
    """Calmar ratio = annualized return / max drawdown."""
    ann_ret = float(np.mean(returns) * ann)
    mdd = max_drawdown(equity_curve)
    return ann_ret / mdd if mdd > 1e-6 else 0.0


def compute_metrics(returns: np.ndarray, equity: np.ndarray) -> dict:
    return {
        "Total return (%)":  round(float((equity[-1] / equity[0] - 1) * 100), 2),
        "Ann. return (%)":   round(float(np.mean(returns) * 252 * 100), 2),
        "Sharpe":            round(sharpe(returns), 3),
        "Sortino":           round(sortino(returns), 3),
        "Max drawdown (%)":  round(max_drawdown(equity) * 100, 2),
        "Calmar":            round(calmar(returns, equity), 3),
        "Win rate (%)":      round(float(np.mean(returns > 0) * 100), 1),
    }


# ------------------------------------------------------------------
# Rollout
# ------------------------------------------------------------------

def rollout(model: SAC, prices: pd.Series, agent_type: AgentType) -> dict:
    """
    Run a trained model on a price series. Returns a dict with:
      - returns  : np.ndarray of step returns
      - equity   : np.ndarray of portfolio value
      - positions: np.ndarray of position at each step
    """
    env = TradingEnv(prices, agent_type=agent_type)
    obs, _ = env.reset()
    returns, equity, positions = [], [env.initial_capital], []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        returns.append(info["net_ret"])
        equity.append(info["equity"])
        positions.append(info["position"])

    return {
        "returns":   np.array(returns, dtype=np.float32),
        "equity":    np.array(equity,  dtype=np.float32),
        "positions": np.array(positions, dtype=np.float32),
    }


# ------------------------------------------------------------------
# Regime labeling
# ------------------------------------------------------------------

def label_regimes(prices: pd.Series, n_warmup: int = 20) -> pd.Series:
    """
    Label each bar as 'low_vol', 'mid_vol', 'high_vol' using
    realized 20-day vol tertiles on the test set.

    Returns a pd.Series aligned with prices[n_warmup:].
    """
    log_rets = np.log(prices / prices.shift(1)).dropna()
    rv = log_rets.rolling(20).std() * np.sqrt(252)
    rv = rv.dropna()

    q33 = rv.quantile(0.33)
    q67 = rv.quantile(0.67)

    def bucket(v):
        if v <= q33:
            return "low_vol"
        elif v <= q67:
            return "mid_vol"
        else:
            return "high_vol"

    return rv.map(bucket)


def regime_conditional_metrics(
    returns: np.ndarray,
    regimes: np.ndarray,
) -> pd.DataFrame:
    """
    Slice returns by regime label and compute metrics per slice.
    regimes must be same length as returns.
    """
    rows = []
    for regime in ["low_vol", "mid_vol", "high_vol"]:
        mask = regimes == regime
        r = returns[mask]
        if len(r) < 5:
            continue
        eq = np.cumprod(1 + r)
        rows.append({
            "Regime": regime,
            "N bars":          int(mask.sum()),
            "Ann. return (%)": round(float(np.mean(r) * 252 * 100), 2),
            "Sharpe":          round(sharpe(r), 3),
            "Max DD (%)":      round(max_drawdown(eq) * 100, 2),
        })
    return pd.DataFrame(rows).set_index("Regime")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 0 — Evaluate agents")
    parser.add_argument("--ticker",   default="SPY",        help="yfinance ticker")
    parser.add_argument("--start",    default="2023-01-01", help="Test period start")
    parser.add_argument("--end",      default="2024-12-31", help="Test period end")
    parser.add_argument("--modeldir", default=".",          help="Directory with models/")
    args = parser.parse_args()

    print("\n=== Phase 0: Evaluation ===")

    # --- Download test data ---
    print(f"  Downloading {args.ticker} test data...")
    df = yf.download(args.ticker, start=args.start, end=args.end, auto_adjust=True, progress=False)
    prices = df["Close"].squeeze().dropna()
    print(f"  {len(prices)} bars in test set.")

    # --- Regime labels ---
    regimes_series = label_regimes(prices, n_warmup=20)
    # Align to env steps (env starts at t=warmup=20, so n_bars-21 steps)
    regime_arr = regimes_series.values[-len(prices) + 20 + 1:]

    # --- Load & rollout each agent ---
    results = {}
    for agent_type in AgentType:
        name = agent_type.value
        model_path = os.path.join(args.modeldir, "models", f"{name}_agent", "best_model")
        if not os.path.exists(model_path + ".zip"):
            print(f"  [WARN] Model not found: {model_path}.zip — skipping.")
            continue
        print(f"  Rolling out: {name}...")
        model = SAC.load(model_path)
        results[name] = rollout(model, prices, agent_type)

    if not results:
        print("  No models found. Train agents first with train_agents.py.")
        return

    # --- Overall metrics table ---
    print("\n" + "="*65)
    print("  OVERALL PERFORMANCE (full test period)")
    print("="*65)
    rows = {}
    for name, r in results.items():
        rows[name] = compute_metrics(r["returns"], r["equity"])
    metrics_df = pd.DataFrame(rows).T
    print(metrics_df.to_string())

    # --- Regime-conditional table ---
    print("\n" + "="*65)
    print("  REGIME-CONDITIONAL PERFORMANCE")
    print("="*65)
    for name, r in results.items():
        rets = r["returns"]
        reg  = regime_arr[:len(rets)]  # align lengths
        print(f"\n  [{name.upper()}]")
        rcm = regime_conditional_metrics(rets, reg)
        print(rcm.to_string())

    # --- Position correlation ---
    print("\n" + "="*65)
    print("  POSITION CORRELATION MATRIX")
    print("="*65)
    min_len = min(len(r["positions"]) for r in results.values())
    pos_df = pd.DataFrame(
        {name: r["positions"][:min_len] for name, r in results.items()}
    )
    print(pos_df.corr().round(3).to_string())

    # --- Save equity curves ---
    eq_df = pd.DataFrame(
        {name: r["equity"][:min_len + 1] for name, r in results.items()}
    )
    out_path = os.path.join(args.modeldir, "equity_curves.csv")
    eq_df.to_csv(out_path, index=False)
    print(f"\n  Equity curves saved → {out_path}")

    print("\n✓ Evaluation complete.")
    print("  Key question: which agent performs best in each volatility regime?")
    print("  That table is the empirical motivation for the Phase 1 hierarchy.")


if __name__ == "__main__":
    main()
