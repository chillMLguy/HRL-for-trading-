"""
Phase 1 visualisation — reads CSVs from evaluate_phase1.py and produces plots.

Usage
-----
  python plot_phase1.py --datadir .

Outputs:
  phase1_equity.png
  phase1_actions_on_price.png
  phase1_weight_evolution.png
  phase1_drawdown.png
  phase1_rolling_sharpe.png
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from regime.allocators import AGENT_ORDER


# ── Style ──────────────────────────────────────────────────────────

ALLOC_COLORS = {
    "equal_weight": "#888888",
    "vol_regime":   "#2196F3",
    "hmm":          "#FF9800",
    "buy_and_hold": "#000000",
}
AGENT_COLORS = {
    "aggressive":   "#2ecc71",   # green
    "growth":       "#3498db",   # blue-ish
    "balanced":     "#9b59b6",   # purple
    "conservative": "#e74c3c",   # red
}
ALLOC_LABELS = {
    "equal_weight": "Equal Weight",
    "vol_regime":   "Vol Regime",
    "hmm":          "HMM",
    "buy_and_hold": "Buy & Hold",
}


def _setup_style():
    """Set a clean, light plotting style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#cccccc",
        "axes.labelcolor":  "#333333",
        "xtick.color":      "#666666",
        "ytick.color":      "#666666",
        "text.color":       "#333333",
        "grid.color":       "#eeeeee",
        "grid.linewidth":   0.5,
        "font.family":      "sans-serif",
        "axes.titlesize":   11,
        "axes.labelsize":   9,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
    })


def _alloc_names_from_equity(eq_df):
    """Extract allocator names from equity CSV columns."""
    skip = {"step", "buy_and_hold"}
    return [c for c in eq_df.columns if c not in skip]


def _drawdown(eq):
    peak = np.maximum.accumulate(eq)
    return (eq - peak) / np.where(peak > 0, peak, 1e-10) * 100


def _rolling_sharpe(returns, window=60, ann=252):
    r = pd.Series(returns)
    mu = r.rolling(window).mean()
    sig = r.rolling(window).std()
    return (mu / sig.replace(0, np.nan)) * np.sqrt(ann)


# ── Plot 1: Equity Comparison ─────────────────────────────────────

def plot_equity(eq_df, outdir):
    fig, ax = plt.subplots(figsize=(14, 5))
    allocs = _alloc_names_from_equity(eq_df)

    for aname in allocs:
        ax.plot(eq_df["step"], eq_df[aname],
                color=ALLOC_COLORS.get(aname, "#333"),
                lw=1.5, label=ALLOC_LABELS.get(aname, aname))

    if "buy_and_hold" in eq_df.columns:
        ax.plot(eq_df["step"], eq_df["buy_and_hold"],
                color=ALLOC_COLORS["buy_and_hold"],
                lw=1.2, ls="--", label="Buy & Hold")

    ax.set_title("Phase 1 — Portfolio Equity Comparison")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Equity")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(outdir, "phase1_equity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ── Plot 2: Actions on Price ──────────────────────────────────────

def plot_actions_on_price(act_df, eq_df, prices_path, outdir):
    """
    Top subplot: price series.
    Bottom subplot: blended action per allocator.

    If no prices_path is given, uses equity as a proxy.
    """
    fig, (ax_price, ax_act) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]})

    allocs = _alloc_names_from_equity(eq_df)

    # Price proxy — use equal_weight equity as the shape indicator
    if allocs:
        ax_price.plot(eq_df["step"], eq_df[allocs[0]],
                      color="#333", lw=1.0, alpha=0.7)
    ax_price.set_ylabel("Equity")
    ax_price.set_title("Phase 1 — Blended Action on Price")
    ax_price.grid(True, alpha=0.3)

    for aname in allocs:
        sub = act_df[act_df["allocator"] == aname]
        ax_act.plot(sub["step"].values, sub["blended_action"].values,
                    color=ALLOC_COLORS.get(aname, "#333"),
                    lw=0.8, alpha=0.8,
                    label=ALLOC_LABELS.get(aname, aname))

    ax_act.set_ylabel("Blended Action")
    ax_act.set_xlabel("Step")
    ax_act.set_ylim(-1.1, 1.1)
    ax_act.axhline(0, color="#aaa", lw=0.5, ls="--")
    ax_act.legend(loc="best", framealpha=0.9, fontsize=8)
    ax_act.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(outdir, "phase1_actions_on_price.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ── Plot 3: Weight Evolution ──────────────────────────────────────

def plot_weight_evolution(w_df, outdir):
    allocs = w_df["allocator"].unique()
    n = len(allocs)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    agent_cols = [c for c in w_df.columns if c.startswith("w_")]
    # Map w_<name> → <name>
    agent_names = [c.replace("w_", "") for c in agent_cols]

    for ax, aname in zip(axes, allocs):
        sub = w_df[w_df["allocator"] == aname].sort_values("step")
        steps = sub["step"].values
        weights = sub[agent_cols].values  # (T, N_AGENTS)

        # Stacked area
        ax.stackplot(
            steps, *weights.T,
            labels=[ALLOC_LABELS.get(a, a) for a in agent_names],
            colors=[AGENT_COLORS.get(a, "#999") for a in agent_names],
            alpha=0.8)

        # Green/red background tint based on sentiment
        # sentiment = weight on most aggressive − weight on most conservative
        w_agg = weights[:, 0]  # aggressive
        w_con = weights[:, -1]  # conservative
        sentiment = w_agg - w_con

        for i in range(len(steps) - 1):
            s = sentiment[i]
            if abs(s) > 0.02:
                c = "#2ecc71" if s > 0 else "#e74c3c"
                alpha = min(abs(s) * 0.3, 0.15)
                ax.axvspan(steps[i], steps[i+1], color=c, alpha=alpha,
                           lw=0, zorder=0)

        ax.set_ylabel("Weight")
        ax.set_ylim(0, 1)
        ax.set_title(ALLOC_LABELS.get(aname, aname))
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Step")
    # Legend on first axis
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper right",
                   fontsize=7, ncol=len(agent_names), framealpha=0.9)

    fig.suptitle("Phase 1 — Agent Weight Evolution", fontsize=12, y=1.01)
    fig.tight_layout()
    path = os.path.join(outdir, "phase1_weight_evolution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ── Plot 4: Drawdown ──────────────────────────────────────────────

def plot_drawdown(eq_df, outdir):
    fig, ax = plt.subplots(figsize=(14, 5))
    allocs = _alloc_names_from_equity(eq_df)

    for aname in allocs:
        dd = _drawdown(eq_df[aname].values)
        ax.plot(eq_df["step"], dd,
                color=ALLOC_COLORS.get(aname, "#333"),
                lw=1.2, label=ALLOC_LABELS.get(aname, aname))

    if "buy_and_hold" in eq_df.columns:
        dd = _drawdown(eq_df["buy_and_hold"].values)
        ax.plot(eq_df["step"], dd, color=ALLOC_COLORS["buy_and_hold"],
                lw=1.0, ls="--", label="Buy & Hold")

    ax.set_title("Phase 1 — Drawdown Comparison")
    ax.set_xlabel("Step")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(outdir, "phase1_drawdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ── Plot 5: Rolling Sharpe ────────────────────────────────────────

def plot_rolling_sharpe(eq_df, act_df, outdir, window=60, ann=252):
    """Compute rolling Sharpe from equity curves (daily returns)."""
    fig, ax = plt.subplots(figsize=(14, 5))
    allocs = _alloc_names_from_equity(eq_df)

    for aname in allocs:
        eq = eq_df[aname].values
        rets = np.diff(eq) / eq[:-1]
        rs = _rolling_sharpe(rets, window=window, ann=ann)
        ax.plot(np.arange(len(rs)), rs,
                color=ALLOC_COLORS.get(aname, "#333"),
                lw=1.0, alpha=0.8,
                label=ALLOC_LABELS.get(aname, aname))

    ax.axhline(0, color="#aaa", lw=0.5, ls="--")
    ax.set_title(f"Phase 1 — Rolling {window}-day Sharpe Ratio")
    ax.set_xlabel("Step")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(outdir, "phase1_rolling_sharpe.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 — Generate comparison plots")
    parser.add_argument("--datadir", default=".",
                        help="Directory containing phase1_*.csv files")
    args = parser.parse_args()

    _setup_style()
    d = args.datadir

    # Load CSVs
    eq_path  = os.path.join(d, "phase1_equity_curves.csv")
    w_path   = os.path.join(d, "phase1_weights.csv")
    a_path   = os.path.join(d, "phase1_actions.csv")

    for p in [eq_path, w_path, a_path]:
        if not os.path.isfile(p):
            print(f"  ERROR: {p} not found. Run evaluate_phase1.py first.")
            return

    eq_df  = pd.read_csv(eq_path)
    w_df   = pd.read_csv(w_path)
    act_df = pd.read_csv(a_path)

    print(f"\n  Generating Phase 1 plots from {d}/ ...\n")

    plot_equity(eq_df, d)
    plot_actions_on_price(act_df, eq_df, None, d)
    plot_weight_evolution(w_df, d)
    plot_drawdown(eq_df, d)
    plot_rolling_sharpe(eq_df, act_df, d)

    print(f"\n  All plots saved.\n")


if __name__ == "__main__":
    main()
