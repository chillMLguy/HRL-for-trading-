"""
Phase 2 visualisation — reads the phase2_*.csv files from evaluate_phase2.py.

Usage
-----
  python plot_phase2.py --datadir .

Outputs:
  phase2_equity.png            — equity curves (meta-RL highlighted) vs. B&H
  phase2_drawdown.png          — drawdown comparison
  phase2_rolling_sharpe.png    — rolling Sharpe comparison
  phase2_weight_evolution.png  — stacked agent weights per strategy
  phase2_meta_diagnostics.png  — meta-RL weight stack + entropy + turnover
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from regime.allocators import AGENT_ORDER


# ── Style ──────────────────────────────────────────────────────────────────

STRAT_COLORS = {
    "meta_rl":      "#d62728",   # crimson — the learned allocator
    "hmm":          "#FF9800",
    "vol_regime":   "#2196F3",
    "equal_weight": "#888888",
    "buy_and_hold": "#000000",
}
STRAT_LABELS = {
    "meta_rl":      "Meta-RL",
    "hmm":          "HMM (fixed)",
    "vol_regime":   "Vol Regime",
    "equal_weight": "Equal Weight",
    "buy_and_hold": "Buy & Hold",
}
AGENT_COLORS = {
    "aggressive":   "#2ecc71",
    "balanced":     "#9b59b6",
    "conservative": "#e74c3c",
}


def _setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#cccccc", "axes.labelcolor": "#333333",
        "xtick.color": "#666666", "ytick.color": "#666666",
        "text.color": "#333333", "grid.color": "#eeeeee",
        "grid.linewidth": 0.5, "font.family": "sans-serif",
        "axes.titlesize": 11, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
    })


def _strat_names(eq_df):
    skip = {"step", "buy_and_hold"}
    names = [c for c in eq_df.columns if c not in skip]
    # Draw meta_rl last so it sits on top.
    return sorted(names, key=lambda n: n == "meta_rl")


def _style(name):
    color = STRAT_COLORS.get(name, "#333")
    lw = 2.4 if name == "meta_rl" else 1.4
    return color, lw


def _drawdown(eq):
    peak = np.maximum.accumulate(eq)
    return (eq - peak) / np.where(peak > 0, peak, 1e-10) * 100


def _rolling_sharpe(returns, window=60, ann=252):
    r = pd.Series(returns)
    return (r.rolling(window).mean()
            / r.rolling(window).std().replace(0, np.nan)) * np.sqrt(ann)


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_equity(eq_df, outdir):
    fig, ax = plt.subplots(figsize=(14, 5))
    for name in _strat_names(eq_df):
        c, lw = _style(name)
        ax.plot(eq_df["step"], eq_df[name], color=c, lw=lw,
                label=STRAT_LABELS.get(name, name))
    if "buy_and_hold" in eq_df.columns:
        ax.plot(eq_df["step"], eq_df["buy_and_hold"],
                color=STRAT_COLORS["buy_and_hold"], lw=1.2, ls="--",
                label="Buy & Hold")
    ax.set_title("Phase 2 — Portfolio Equity (Meta-RL vs. Baselines)")
    ax.set_xlabel("Step"); ax.set_ylabel("Cumulative Equity")
    ax.legend(loc="best", framealpha=0.9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, outdir, "phase2_equity.png")


def plot_drawdown(eq_df, outdir):
    fig, ax = plt.subplots(figsize=(14, 5))
    for name in _strat_names(eq_df):
        c, lw = _style(name)
        ax.plot(eq_df["step"], _drawdown(eq_df[name].values),
                color=c, lw=lw, label=STRAT_LABELS.get(name, name))
    if "buy_and_hold" in eq_df.columns:
        ax.plot(eq_df["step"], _drawdown(eq_df["buy_and_hold"].values),
                color=STRAT_COLORS["buy_and_hold"], lw=1.0, ls="--",
                label="Buy & Hold")
    ax.set_title("Phase 2 — Drawdown Comparison")
    ax.set_xlabel("Step"); ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="best", framealpha=0.9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, outdir, "phase2_drawdown.png")


def plot_rolling_sharpe(eq_df, outdir, window=60, ann=252):
    fig, ax = plt.subplots(figsize=(14, 5))
    for name in _strat_names(eq_df):
        eq = eq_df[name].values
        rets = np.diff(eq) / eq[:-1]
        c, lw = _style(name)
        rs = _rolling_sharpe(rets, window, ann)
        ax.plot(np.arange(len(rs)), rs, color=c, lw=lw, alpha=0.9,
                label=STRAT_LABELS.get(name, name))
    ax.axhline(0, color="#aaa", lw=0.5, ls="--")
    ax.set_title(f"Phase 2 — Rolling {window}-bar Sharpe")
    ax.set_xlabel("Step"); ax.set_ylabel("Sharpe")
    ax.legend(loc="best", framealpha=0.9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, outdir, "phase2_rolling_sharpe.png")


def plot_weight_evolution(w_df, outdir):
    allocs = list(w_df["allocator"].unique())
    n = len(allocs)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.0 * n), sharex=True)
    if n == 1:
        axes = [axes]
    agent_cols = [f"w_{a}" for a in AGENT_ORDER]

    for ax, name in zip(axes, allocs):
        sub = w_df[w_df["allocator"] == name].sort_values("step")
        steps = sub["step"].values
        weights = sub[agent_cols].values
        ax.stackplot(steps, *weights.T, labels=AGENT_ORDER,
                     colors=[AGENT_COLORS.get(a, "#999") for a in AGENT_ORDER],
                     alpha=0.85)
        ax.set_ylabel("Weight"); ax.set_ylim(0, 1)
        ax.set_title(STRAT_LABELS.get(name, name))
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Step")
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend(h, l, loc="upper right", fontsize=7,
                   ncol=len(AGENT_ORDER), framealpha=0.9)
    fig.suptitle("Phase 2 — Agent Weight Evolution", fontsize=12, y=1.005)
    fig.tight_layout()
    _save(fig, outdir, "phase2_weight_evolution.png")


def plot_meta_diagnostics(w_df, outdir):
    """Meta-RL weight stack + allocation entropy + turnover over time."""
    if "meta_rl" not in set(w_df["allocator"]):
        print("  [skip] meta diagnostics — no meta_rl weights found")
        return
    sub = w_df[w_df["allocator"] == "meta_rl"].sort_values("step")
    steps = sub["step"].values
    agent_cols = [f"w_{a}" for a in AGENT_ORDER]
    W = sub[agent_cols].values                      # (T, N)

    p = np.clip(W, 1e-12, 1.0)
    entropy = -(p * np.log(p)).sum(axis=1) / np.log(W.shape[1])
    turnover = np.concatenate([[0.0], np.abs(np.diff(W, axis=0)).sum(axis=1)])

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1]})
    axes[0].stackplot(steps, *W.T, labels=AGENT_ORDER,
                      colors=[AGENT_COLORS.get(a, "#999") for a in AGENT_ORDER],
                      alpha=0.85)
    axes[0].set_ylabel("Weight"); axes[0].set_ylim(0, 1)
    axes[0].set_title("Meta-RL — Learned Allocation")
    axes[0].legend(loc="upper right", fontsize=8, ncol=len(AGENT_ORDER))
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(steps, entropy, color="#d62728", lw=1.4)
    axes[1].set_ylabel("Entropy"); axes[1].set_ylim(0, 1.05)
    axes[1].axhline(1.0, color="#aaa", lw=0.5, ls="--")
    axes[1].set_title("Allocation entropy  (1 = perfectly diversified, "
                      "0 = collapsed to one agent)")
    axes[1].grid(True, alpha=0.2)

    axes[2].plot(steps, turnover, color="#2196F3", lw=1.0)
    axes[2].set_ylabel("Turnover"); axes[2].set_xlabel("Step")
    axes[2].set_title(r"Allocation turnover  $\Sigma|w_t - w_{t-1}|$")
    axes[2].grid(True, alpha=0.2)

    fig.tight_layout()
    _save(fig, outdir, "phase2_meta_diagnostics.png")


def _save(fig, outdir, fname):
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 — comparison plots")
    parser.add_argument("--datadir", default=".",
                        help="Directory with phase2_*.csv files")
    args = parser.parse_args()

    _setup_style()
    d = args.datadir
    eq_path = os.path.join(d, "phase2_equity_curves.csv")
    w_path = os.path.join(d, "phase2_weights.csv")
    for pth in (eq_path, w_path):
        if not os.path.isfile(pth):
            print(f"  ERROR: {pth} not found. Run evaluate_phase2.py first.")
            return

    eq_df = pd.read_csv(eq_path)
    w_df = pd.read_csv(w_path)

    print(f"\n  Generating Phase 2 plots from {d}/ ...\n")
    plot_equity(eq_df, d)
    plot_drawdown(eq_df, d)
    plot_rolling_sharpe(eq_df, d)
    plot_weight_evolution(w_df, d)
    plot_meta_diagnostics(w_df, d)
    print("\n  All Phase 2 plots saved.\n")


if __name__ == "__main__":
    main()
