"""
Unified visualisation for Phase 0, Phase 1, and combined results.

Usage
-----
  python plot_all.py --datadir .

Reads:
  equity_curves.csv          (Phase 0 — from evaluate_agents.py)
  phase1_equity_curves.csv   (Phase 1 — from evaluate_phase1.py)
  phase1_weights.csv
  phase1_actions.csv
  phase1_metrics.csv

Outputs:
  ── Phase 0 ──
  p0_equity.png              Equity curves (indexed to 100)
  p0_drawdown.png            Drawdown series
  p0_rolling_sharpe.png      Rolling Sharpe
  p0_return_dist.png         Bar return distributions
  p0_metrics_bar.png         Metrics bar chart

  ── Phase 1 ──
  p1_equity.png              Allocator equity comparison
  p1_drawdown.png            Allocator drawdowns
  p1_rolling_sharpe.png      Rolling Sharpe (with B&H)
  p1_weight_evolution.png    Agent weight evolution (stacked)
  p1_actions_on_price.png    Blended actions on price
  p1_metrics_bar.png         Metrics bar chart (with B&H Sharpe)

  ── Combined ──
  combined_equity.png        Phase 0 agents + Phase 1 allocators + B&H
  combined_sharpe_bar.png    Sharpe comparison bar chart (all)
  combined_metrics_table.png Metrics summary table
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

# ═══════════════════════════════════════════════════════════════════
#  Style & Palette
# ═══════════════════════════════════════════════════════════════════

# --- Phase 0 agent colours ---
P0_COLORS = {
    "aggressive":        "#E74C3C",
    "growth":            "#E67E22",
    "balanced":          "#3498DB",
    "conservative":      "#1ABC9C",
    "ultra_conservative": "#9B59B6",
    "buy_&_hold":        "#7F8C8D",
}

# --- Phase 1 allocator colours ---
P1_COLORS = {
    "equal_weight": "#95A5A6",
    "vol_regime":   "#2980B9",
    "hmm":          "#E67E22",
    "buy_and_hold": "#2C3E50",
}

P1_LABELS = {
    "equal_weight": "Equal Weight",
    "vol_regime":   "Vol-Regime",
    "hmm":          "HMM",
    "buy_and_hold": "Buy & Hold",
}

AGENT_COLORS = {
    "aggressive":   "#27AE60",
    "growth":       "#2980B9",
    "balanced":     "#8E44AD",
    "conservative": "#E74C3C",
}


def _apply_style():
    """Clean, academic-quality plotting style."""
    plt.rcParams.update({
        "figure.facecolor":     "#FAFBFC",
        "axes.facecolor":       "#FFFFFF",
        "axes.edgecolor":       "#D5D8DC",
        "axes.labelcolor":      "#2C3E50",
        "axes.titlesize":       12,
        "axes.titleweight":     "semibold",
        "axes.labelsize":       10,
        "xtick.color":          "#566573",
        "ytick.color":          "#566573",
        "xtick.labelsize":      8.5,
        "ytick.labelsize":      8.5,
        "text.color":           "#2C3E50",
        "grid.color":           "#EAECEE",
        "grid.linewidth":       0.6,
        "grid.linestyle":       "--",
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Helvetica", "DejaVu Sans", "Arial"],
        "legend.fontsize":      8.5,
        "legend.framealpha":    0.92,
        "legend.edgecolor":     "#D5D8DC",
        "figure.dpi":           150,
        "savefig.dpi":          200,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.15,
    })


def _color_for(name):
    """Universal colour lookup across Phase 0 and Phase 1."""
    if name in P0_COLORS:
        return P0_COLORS[name]
    if name in P1_COLORS:
        return P1_COLORS[name]
    return "#566573"


def _label_for(name):
    if name in P1_LABELS:
        return P1_LABELS[name]
    return name.replace("_", " ").title()


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _drawdown(eq):
    """Drawdown in percent from equity array."""
    peak = np.maximum.accumulate(eq)
    return (eq - peak) / np.where(peak > 0, peak, 1e-10) * 100


def _rolling_sharpe(returns, window=60, ann=252):
    r = pd.Series(returns)
    mu = r.rolling(window).mean()
    sig = r.rolling(window).std()
    return (mu / sig.replace(0, np.nan)) * np.sqrt(ann)


def _sharpe(r, ann=252):
    s = np.std(r)
    return float(np.sqrt(ann) * np.mean(r) / s) if s > 1e-10 else 0.0


def _sortino(r, ann=252):
    down = r[r < 0]
    ds = np.std(down) if len(down) > 1 else 1e-10
    return float(np.sqrt(ann) * np.mean(r) / ds)


def _max_drawdown(eq):
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, 1e-10)
    return float(dd.max())


def _calmar(r, eq, ann=252):
    mdd = _max_drawdown(eq)
    return float(np.mean(r) * ann / mdd) if mdd > 1e-6 else 0.0


def _cvar(r, alpha=0.95):
    var = np.percentile(r, (1 - alpha) * 100)
    tail = r[r <= var]
    return float(tail.mean()) if len(tail) > 0 else 0.0


def _omega(r, threshold=0.0):
    gains = (r[r > threshold] - threshold).sum()
    losses = (threshold - r[r < threshold]).sum()
    return float(gains / losses) if losses > 1e-10 else 2.0


def _compute_metrics(r, eq, ann=252):
    return {
        "Total ret (%)":  round((eq[-1] / eq[0] - 1) * 100, 2),
        "Ann. ret (%)":   round(np.mean(r) * ann * 100, 2),
        "Sharpe":         round(_sharpe(r, ann), 3),
        "Sortino":        round(_sortino(r, ann), 3),
        "MaxDD (%)":      round(_max_drawdown(eq) * 100, 2),
        "Calmar":         round(_calmar(r, eq, ann), 3),
        "CVaR 95% (%)":   round(_cvar(r) * 100, 3),
        "Omega":          round(_omega(r), 3),
        "Win rate (%)":   round(float((r > 0).mean()) * 100, 1),
    }


def _add_watermark(fig, text="HRL Trading System"):
    fig.text(0.99, 0.005, text, fontsize=7, color="#BDC3C7",
             ha="right", va="bottom", style="italic")


def _save(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path, facecolor=fig.get_facecolor())
    print(f"  Saved → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Phase 0 Plots
# ═══════════════════════════════════════════════════════════════════

def p0_equity(df, outdir, ann):
    """Equity curves indexed to 100."""
    agents = list(df.columns)
    fig, ax = plt.subplots(figsize=(13, 5))

    for name in agents:
        lw = 1.8 if name == "buy_&_hold" else 1.3
        ls = "--" if name == "buy_&_hold" else "-"
        alpha = 0.65 if name == "buy_&_hold" else 0.9
        ax.plot(df[name].values, color=_color_for(name),
                lw=lw, ls=ls, alpha=alpha, label=_label_for(name))

    ax.axhline(100, color="#BDC3C7", lw=0.6, ls=":", alpha=0.6)
    ax.set_ylabel("Portfolio Value (indexed 100)")
    ax.set_xlabel("Step")
    ax.set_title("Phase 0 — λ-Spectrum Agent Equity Curves", pad=12)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True)
    _add_watermark(fig)
    _save(fig, outdir, "p0_equity.png")


def p0_drawdown(df, outdir):
    """Drawdown series for Phase 0 agents."""
    agents = list(df.columns)
    fig, ax = plt.subplots(figsize=(13, 4))

    for name in agents:
        dd = _drawdown(df[name].values)
        lw = 1.5 if name == "buy_&_hold" else 1.1
        ls = "--" if name == "buy_&_hold" else "-"
        ax.plot(dd, color=_color_for(name), lw=lw, ls=ls,
                alpha=0.85, label=_label_for(name))

    ax.fill_between(range(len(dd)), dd, 0, alpha=0.03, color="#E74C3C")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Step")
    ax.set_title("Phase 0 — Drawdown Comparison", pad=12)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=8)
    ax.grid(True)
    _add_watermark(fig)
    _save(fig, outdir, "p0_drawdown.png")


def p0_rolling_sharpe(df, outdir, ann):
    """Rolling Sharpe for Phase 0 agents."""
    agents = list(df.columns)
    window = max(10, ann // 13)
    fig, ax = plt.subplots(figsize=(13, 4))

    for name in agents:
        r = df[name].pct_change().dropna()
        rs = r.rolling(window).apply(
            lambda x: np.sqrt(ann) * x.mean() / (x.std() + 1e-10))
        lw = 1.4 if name == "buy_&_hold" else 1.0
        ls = "--" if name == "buy_&_hold" else "-"
        ax.plot(rs.values, color=_color_for(name), lw=lw, ls=ls,
                alpha=0.85, label=_label_for(name))

    ax.axhline(0, color="#BDC3C7", lw=0.8, ls=":")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xlabel("Step")
    ax.set_title(f"Phase 0 — Rolling {window}-bar Sharpe Ratio", pad=12)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True)
    _add_watermark(fig)
    _save(fig, outdir, "p0_rolling_sharpe.png")


def p0_return_dist(df, outdir):
    """Return distribution histograms for Phase 0 agents."""
    agents = [a for a in df.columns if a != "buy_&_hold"]
    fig, ax = plt.subplots(figsize=(10, 5))

    for name in agents:
        r = df[name].pct_change().dropna().values * 100
        ax.hist(r, bins=70, alpha=0.35, color=_color_for(name),
                label=_label_for(name), density=True, histtype="stepfilled",
                edgecolor=_color_for(name), linewidth=0.5)
        ax.axvline(np.mean(r), color=_color_for(name), lw=1.3,
                   ls="--", alpha=0.7)

    ax.set_xlabel("Bar Return (%)")
    ax.set_ylabel("Density")
    ax.set_title("Phase 0 — Return Distribution", pad=12)
    ax.legend(framealpha=0.9)
    ax.grid(True)
    _add_watermark(fig)
    _save(fig, outdir, "p0_return_dist.png")


def p0_metrics_bar(df, outdir, ann):
    """Grouped bar chart of key metrics for Phase 0 agents."""
    agents = list(df.columns)
    metrics_data = {}
    for name in agents:
        eq = df[name].values
        r = np.diff(eq) / eq[:-1]
        metrics_data[name] = _compute_metrics(r, eq / eq[0], ann)

    metric_keys = ["Sharpe", "Sortino", "Calmar", "Omega"]
    labels = [_label_for(n) for n in agents]
    x = np.arange(len(metric_keys))
    width = 0.8 / len(agents)

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, name in enumerate(agents):
        vals = [metrics_data[name][k] for k in metric_keys]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.9,
                      label=_label_for(name), color=_color_for(name),
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        # Value labels
        for bar, val in zip(bars, vals):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=6.5,
                    color="#2C3E50")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, fontsize=10)
    ax.set_ylabel("Value")
    ax.set_title("Phase 0 — Risk-Adjusted Metrics Comparison", pad=12)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax.axhline(0, color="#BDC3C7", lw=0.6)
    ax.grid(True, axis="y")
    _add_watermark(fig)
    _save(fig, outdir, "p0_metrics_bar.png")


# ═══════════════════════════════════════════════════════════════════
#  Phase 1 Plots
# ═══════════════════════════════════════════════════════════════════

def _p1_alloc_names(eq_df):
    skip = {"step", "buy_and_hold"}
    return [c for c in eq_df.columns if c not in skip]


def p1_equity(eq_df, outdir):
    """Allocator equity comparison."""
    allocs = _p1_alloc_names(eq_df)
    fig, ax = plt.subplots(figsize=(13, 5))

    for aname in allocs:
        ax.plot(eq_df["step"], eq_df[aname],
                color=P1_COLORS.get(aname, "#566573"),
                lw=1.5, label=P1_LABELS.get(aname, aname))

    if "buy_and_hold" in eq_df.columns:
        ax.plot(eq_df["step"], eq_df["buy_and_hold"],
                color=P1_COLORS["buy_and_hold"],
                lw=1.3, ls="--", alpha=0.7, label="Buy & Hold")

    ax.axhline(1.0, color="#BDC3C7", lw=0.6, ls=":", alpha=0.6)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Equity")
    ax.set_title("Phase 1 — Portfolio Equity Comparison", pad=12)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True)
    _add_watermark(fig)
    _save(fig, outdir, "p1_equity.png")


def p1_drawdown(eq_df, outdir):
    """Drawdown comparison for allocators."""
    allocs = _p1_alloc_names(eq_df)
    fig, ax = plt.subplots(figsize=(13, 4))

    for aname in allocs:
        dd = _drawdown(eq_df[aname].values)
        ax.plot(eq_df["step"], dd,
                color=P1_COLORS.get(aname, "#566573"),
                lw=1.2, label=P1_LABELS.get(aname, aname))

    if "buy_and_hold" in eq_df.columns:
        dd = _drawdown(eq_df["buy_and_hold"].values)
        ax.plot(eq_df["step"], dd, color=P1_COLORS["buy_and_hold"],
                lw=1.0, ls="--", alpha=0.7, label="Buy & Hold")

    ax.set_xlabel("Step")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Phase 1 — Drawdown Comparison", pad=12)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True)
    _add_watermark(fig)
    _save(fig, outdir, "p1_drawdown.png")


def p1_rolling_sharpe(eq_df, outdir, window=60, ann=252):
    """Rolling Sharpe for allocators — includes Buy & Hold."""
    allocs = _p1_alloc_names(eq_df)
    fig, ax = plt.subplots(figsize=(13, 4.5))

    for aname in allocs:
        eq = eq_df[aname].values
        rets = np.diff(eq) / eq[:-1]
        rs = _rolling_sharpe(rets, window=window, ann=ann)
        ax.plot(np.arange(len(rs)), rs,
                color=P1_COLORS.get(aname, "#566573"),
                lw=1.1, alpha=0.85,
                label=P1_LABELS.get(aname, aname))

    # ── Buy & Hold rolling Sharpe ──
    if "buy_and_hold" in eq_df.columns:
        eq_bh = eq_df["buy_and_hold"].values
        rets_bh = np.diff(eq_bh) / eq_bh[:-1]
        rs_bh = _rolling_sharpe(rets_bh, window=window, ann=ann)
        ax.plot(np.arange(len(rs_bh)), rs_bh,
                color=P1_COLORS["buy_and_hold"],
                lw=1.2, ls="--", alpha=0.65, label="Buy & Hold")

    ax.axhline(0, color="#BDC3C7", lw=0.8, ls=":")
    ax.set_xlabel("Step")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title(f"Phase 1 — Rolling {window}-day Sharpe Ratio (incl. Buy & Hold)",
                 pad=12)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True)
    _add_watermark(fig)
    _save(fig, outdir, "p1_rolling_sharpe.png")


def p1_weight_evolution(w_df, outdir):
    """Stacked area of agent weights per allocator."""
    allocs = w_df["allocator"].unique()
    n = len(allocs)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    agent_cols = [c for c in w_df.columns if c.startswith("w_")]
    agent_names = [c.replace("w_", "") for c in agent_cols]

    for ax, aname in zip(axes, allocs):
        sub = w_df[w_df["allocator"] == aname].sort_values("step")
        steps = sub["step"].values
        weights = sub[agent_cols].values

        ax.stackplot(
            steps, *weights.T,
            labels=[_label_for(a) for a in agent_names],
            colors=[AGENT_COLORS.get(a, "#999") for a in agent_names],
            alpha=0.82)

        ax.set_ylabel("Weight")
        ax.set_ylim(0, 1)
        ax.set_title(P1_LABELS.get(aname, aname), fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper right",
                   fontsize=7.5, ncol=len(agent_names), framealpha=0.9)

    fig.suptitle("Phase 1 — Agent Weight Evolution per Allocator",
                 fontsize=12, fontweight="semibold", y=1.01)
    fig.tight_layout()
    _add_watermark(fig)
    _save(fig, outdir, "p1_weight_evolution.png")


def p1_actions_on_price(act_df, eq_df, outdir):
    """Blended action per allocator overlaid on equity proxy."""
    allocs = _p1_alloc_names(eq_df)
    fig, (ax_price, ax_act) = plt.subplots(
        2, 1, figsize=(13, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]})

    if allocs:
        ax_price.plot(eq_df["step"], eq_df[allocs[0]],
                      color="#2C3E50", lw=1.0, alpha=0.7)
    if "buy_and_hold" in eq_df.columns:
        ax_price.plot(eq_df["step"], eq_df["buy_and_hold"],
                      color=P1_COLORS["buy_and_hold"],
                      lw=0.8, ls="--", alpha=0.5, label="Buy & Hold")
    ax_price.set_ylabel("Equity")
    ax_price.set_title("Phase 1 — Blended Action on Price", pad=10)
    ax_price.grid(True)

    for aname in allocs:
        sub = act_df[act_df["allocator"] == aname]
        ax_act.plot(sub["step"].values, sub["blended_action"].values,
                    color=P1_COLORS.get(aname, "#566573"),
                    lw=0.7, alpha=0.8,
                    label=P1_LABELS.get(aname, aname))

    ax_act.set_ylabel("Blended Action")
    ax_act.set_xlabel("Step")
    ax_act.set_ylim(-1.15, 1.15)
    ax_act.axhline(0, color="#BDC3C7", lw=0.8, ls=":")
    ax_act.legend(loc="best", framealpha=0.9, fontsize=8)
    ax_act.grid(True)

    fig.tight_layout()
    _add_watermark(fig)
    _save(fig, outdir, "p1_actions_on_price.png")


def p1_metrics_bar(eq_df, m_df, outdir, ann=252):
    """
    Grouped bar chart of Sharpe, Sortino, Calmar, MaxDD for Phase 1
    allocators AND Buy & Hold.
    """
    # If we have a metrics CSV, use it; otherwise compute from equity
    if m_df is not None and not m_df.empty:
        strategies = list(m_df.columns)
        metrics_data = {}
        for s in strategies:
            metrics_data[s] = m_df[s].to_dict()
    else:
        strategies = _p1_alloc_names(eq_df)
        if "buy_and_hold" in eq_df.columns:
            strategies.append("buy_and_hold")
        metrics_data = {}
        for s in strategies:
            eq = eq_df[s].values
            r = np.diff(eq) / eq[:-1]
            metrics_data[s] = _compute_metrics(r, eq / eq[0], ann)

    metric_keys = ["Sharpe", "Sortino", "MaxDD (%)", "Calmar"]
    labels = [P1_LABELS.get(s, _label_for(s)) for s in strategies]
    x = np.arange(len(metric_keys))
    width = 0.8 / len(strategies)

    fig, ax = plt.subplots(figsize=(12, 5.5))

    bar_colors = [P1_COLORS.get(s, _color_for(s)) for s in strategies]

    for i, (s, col) in enumerate(zip(strategies, bar_colors)):
        vals = []
        for k in metric_keys:
            v = metrics_data[s].get(k, 0.0)
            vals.append(float(v) if v is not None else 0.0)
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.88,
                      label=labels[i], color=col,
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            h = bar.get_height()
            offset = 0.03 if h >= 0 else -0.12
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=6.5,
                    color="#2C3E50", fontweight="medium")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, fontsize=10)
    ax.set_ylabel("Value")
    ax.set_title("Phase 1 — Risk-Adjusted Metrics (incl. Buy & Hold)", pad=12)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax.axhline(0, color="#BDC3C7", lw=0.6)
    ax.grid(True, axis="y")
    _add_watermark(fig)
    _save(fig, outdir, "p1_metrics_bar.png")


# ═══════════════════════════════════════════════════════════════════
#  Combined Plots  (Phase 0 + Phase 1)
# ═══════════════════════════════════════════════════════════════════

def combined_equity(p0_df, p1_eq_df, outdir, ann):
    """
    Single chart: Phase 0 agents + Phase 1 allocators + Buy & Hold,
    all rebased to 1.0.
    """
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(16, 5.5),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.25})

    # Left: Phase 0 agents
    p0_agents = list(p0_df.columns)
    for name in p0_agents:
        eq = p0_df[name].values / p0_df[name].values[0]
        lw = 1.5 if name == "buy_&_hold" else 1.2
        ls = "--" if name == "buy_&_hold" else "-"
        alpha = 0.6 if name == "buy_&_hold" else 0.9
        ax0.plot(eq, color=_color_for(name), lw=lw, ls=ls, alpha=alpha,
                 label=_label_for(name))

    ax0.axhline(1.0, color="#BDC3C7", lw=0.6, ls=":")
    ax0.set_ylabel("Normalised Equity")
    ax0.set_xlabel("Step")
    ax0.set_title("Phase 0 — Individual Agents", fontsize=11, pad=10)
    ax0.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    ax0.grid(True)

    # Right: Phase 1 allocators
    allocs = _p1_alloc_names(p1_eq_df)
    for aname in allocs:
        ax1.plot(p1_eq_df["step"], p1_eq_df[aname],
                 color=P1_COLORS.get(aname, "#566573"),
                 lw=1.5, label=P1_LABELS.get(aname, aname))

    if "buy_and_hold" in p1_eq_df.columns:
        ax1.plot(p1_eq_df["step"], p1_eq_df["buy_and_hold"],
                 color=P1_COLORS["buy_and_hold"],
                 lw=1.3, ls="--", alpha=0.65, label="Buy & Hold")

    ax1.axhline(1.0, color="#BDC3C7", lw=0.6, ls=":")
    ax1.set_xlabel("Step")
    ax1.set_title("Phase 1 — Meta-Controlled Allocators", fontsize=11, pad=10)
    ax1.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    ax1.grid(True)

    fig.suptitle("Equity Comparison — Phase 0 vs Phase 1",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02,
             "Note: Panels may reflect different test periods / bar "
             "intervals. Each panel's B&H matches its own evaluation setup.",
             ha="center", fontsize=7, color="#95A5A6", style="italic")
    fig.tight_layout()
    _add_watermark(fig)
    _save(fig, outdir, "combined_equity.png")


def combined_sharpe_bar(p0_df, p1_eq_df, outdir, p0_ann, p1_ann=252):
    """
    Single bar chart: Sharpe ratios of Phase 0 agents (excl. B&H),
    Phase 1 allocators, and ONE Buy & Hold from Phase 1 evaluation.

    Phase 0 and Phase 1 may use different test periods / bar intervals,
    so Phase 0's B&H is excluded to avoid showing two contradictory
    passive benchmarks.  Phase 0's own B&H remains in p0_*.png plots
    where the comparison is valid.
    """
    data = {}

    # Phase 0 — skip buy_&_hold (different eval setup)
    for name in p0_df.columns:
        if name == "buy_&_hold":
            continue
        eq = p0_df[name].values
        r = np.diff(eq) / eq[:-1]
        data[f"P0: {_label_for(name)}"] = {
            "sharpe": _sharpe(r, p0_ann),
            "color": _color_for(name),
            "phase": 0,
        }

    # Phase 1 allocators
    allocs = _p1_alloc_names(p1_eq_df)
    for aname in allocs:
        eq = p1_eq_df[aname].values
        r = np.diff(eq) / eq[:-1]
        data[f"P1: {P1_LABELS.get(aname, aname)}"] = {
            "sharpe": _sharpe(r, p1_ann),
            "color": P1_COLORS.get(aname, "#566573"),
            "phase": 1,
        }

    # Single Buy & Hold — from Phase 1 evaluation only
    if "buy_and_hold" in p1_eq_df.columns:
        eq_bh = p1_eq_df["buy_and_hold"].values
        r_bh = np.diff(eq_bh) / eq_bh[:-1]
        data["Buy & Hold"] = {
            "sharpe": _sharpe(r_bh, p1_ann),
            "color": P1_COLORS["buy_and_hold"],
            "phase": -1,
        }

    names = list(data.keys())
    sharpes = [data[n]["sharpe"] for n in names]
    colors = [data[n]["color"] for n in names]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.2), 5.5))

    bars = ax.bar(range(len(names)), sharpes, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.7, width=0.7)

    # Value labels
    for bar, val in zip(bars, sharpes):
        h = bar.get_height()
        offset = 0.03 if h >= 0 else -0.08
        va = "bottom" if h >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                f"{val:.2f}", ha="center", va=va, fontsize=8,
                color="#2C3E50", fontweight="medium")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio — All Strategies Comparison", pad=12)
    ax.axhline(0, color="#BDC3C7", lw=0.8)
    ax.grid(True, axis="y")

    # Phase separation line
    p0_count = sum(1 for d in data.values() if d["phase"] == 0)
    if p0_count < len(names):
        ax.axvline(p0_count - 0.5, color="#D5D8DC", lw=1.0, ls="--")
        ax.text(p0_count / 2, ax.get_ylim()[1] * 0.95, "Phase 0",
                ha="center", fontsize=8, color="#7F8C8D", style="italic")
        mid_p1 = p0_count + (len(names) - p0_count) / 2
        ax.text(mid_p1, ax.get_ylim()[1] * 0.95, "Phase 1 + B&H",
                ha="center", fontsize=8, color="#7F8C8D", style="italic")

    # Caveat footnote
    fig.text(0.5, -0.02,
             "Note: Phase 0 and Phase 1 may use different test periods / "
             "bar intervals. B&H shown is from the Phase 1 evaluation only.",
             ha="center", fontsize=7, color="#95A5A6", style="italic")

    _add_watermark(fig)
    _save(fig, outdir, "combined_sharpe_bar.png")


def combined_metrics_table(p0_df, p1_eq_df, outdir, p0_ann, p1_ann=252):
    """
    Render a publication-quality metrics table as an image.
    Rows = strategies, Columns = metrics.
    Phase 0 B&H is excluded — only one B&H (from Phase 1) is shown.
    """
    all_metrics = {}

    # Phase 0 — skip buy_&_hold
    for name in p0_df.columns:
        if name == "buy_&_hold":
            continue
        eq = p0_df[name].values
        r = np.diff(eq) / eq[:-1]
        all_metrics[f"P0 · {_label_for(name)}"] = _compute_metrics(
            r, eq / eq[0], p0_ann)

    # Phase 1
    allocs = _p1_alloc_names(p1_eq_df)
    for aname in allocs:
        eq = p1_eq_df[aname].values
        r = np.diff(eq) / eq[:-1]
        all_metrics[f"P1 · {P1_LABELS.get(aname, aname)}"] = \
            _compute_metrics(r, eq / eq[0], p1_ann)

    if "buy_and_hold" in p1_eq_df.columns:
        eq_bh = p1_eq_df["buy_and_hold"].values
        r_bh = np.diff(eq_bh) / eq_bh[:-1]
        all_metrics["Buy & Hold"] = _compute_metrics(
            r_bh, eq_bh / eq_bh[0], p1_ann)

    df_m = pd.DataFrame(all_metrics).T
    display_cols = ["Total ret (%)", "Ann. ret (%)", "Sharpe", "Sortino",
                    "MaxDD (%)", "Calmar", "CVaR 95% (%)", "Win rate (%)"]
    df_m = df_m[[c for c in display_cols if c in df_m.columns]]

    # Render table
    n_rows = len(df_m)
    n_cols = len(df_m.columns)
    fig_height = max(3, 0.4 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")

    tbl = ax.table(
        cellText=df_m.values.round(3),
        rowLabels=df_m.index,
        colLabels=df_m.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.4)

    # Style header
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D5D8DC")
        if row == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold", fontsize=7.5)
        elif col == -1:
            cell.set_facecolor("#ECF0F1")
            cell.set_text_props(fontweight="semibold", fontsize=7.5)
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 == 0 else "#F8F9FA")

    fig.suptitle("Performance Metrics — All Strategies",
                 fontsize=12, fontweight="bold", y=0.98)
    fig.text(0.5, 0.02,
             "Note: Phase 0 and Phase 1 rows may reflect different test "
             "periods / bar intervals. B&H is from the Phase 1 evaluation.",
             ha="center", fontsize=7, color="#95A5A6", style="italic")
    _add_watermark(fig)
    _save(fig, outdir, "combined_metrics_table.png")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

BARS_PER_YEAR = {"1d": 252, "1h": 1638, "30m": 3276, "15m": 6552}


def main():
    parser = argparse.ArgumentParser(
        description="Generate all Phase 0 / Phase 1 / Combined plots")
    parser.add_argument("--datadir", default=".",
                        help="Directory containing CSV outputs")
    parser.add_argument("--outdir", default=None,
                        help="Output directory for plots (default: datadir)")
    parser.add_argument("--p0_interval", default="1d",
                        choices=list(BARS_PER_YEAR.keys()),
                        help="Bar interval used for Phase 0 evaluation")
    parser.add_argument("--p1_interval", default="1d",
                        choices=list(BARS_PER_YEAR.keys()),
                        help="Bar interval used for Phase 1 evaluation")
    args = parser.parse_args()

    _apply_style()

    d = args.datadir
    outdir = args.outdir or d
    os.makedirs(outdir, exist_ok=True)

    p0_ann = BARS_PER_YEAR[args.p0_interval]
    p1_ann = BARS_PER_YEAR[args.p1_interval]

    # ── Load Phase 0 data ─────────────────────────────────────────
    p0_path = os.path.join(d, "equity_curves.csv")
    p0_df = None
    if os.path.isfile(p0_path):
        raw = pd.read_csv(p0_path)
        p0_df = raw / raw.iloc[0] * 100  # rebase to 100
        print(f"\n  Phase 0 data loaded: {list(p0_df.columns)}")
    else:
        print(f"\n  [skip] Phase 0 — {p0_path} not found")

    # ── Load Phase 1 data ─────────────────────────────────────────
    p1_eq_path = os.path.join(d, "phase1_equity_curves.csv")
    p1_w_path  = os.path.join(d, "phase1_weights.csv")
    p1_a_path  = os.path.join(d, "phase1_actions.csv")
    p1_m_path  = os.path.join(d, "phase1_metrics.csv")

    p1_eq_df = p1_w_df = p1_a_df = p1_m_df = None

    if os.path.isfile(p1_eq_path):
        p1_eq_df = pd.read_csv(p1_eq_path)
        allocs = _p1_alloc_names(p1_eq_df)
        print(f"  Phase 1 equity loaded: {allocs}")
    else:
        print(f"  [skip] Phase 1 — {p1_eq_path} not found")

    if os.path.isfile(p1_w_path):
        p1_w_df = pd.read_csv(p1_w_path)
    if os.path.isfile(p1_a_path):
        p1_a_df = pd.read_csv(p1_a_path)
    if os.path.isfile(p1_m_path):
        p1_m_df = pd.read_csv(p1_m_path, index_col=0)
    else:
        p1_m_df = None

    # ── Generate Phase 0 plots ────────────────────────────────────
    if p0_df is not None:
        print(f"\n  ── Phase 0 plots ──")
        p0_equity(p0_df, outdir, p0_ann)
        p0_drawdown(p0_df, outdir)
        p0_rolling_sharpe(p0_df, outdir, p0_ann)
        p0_return_dist(p0_df, outdir)
        p0_metrics_bar(p0_df, outdir, p0_ann)

    # ── Generate Phase 1 plots ────────────────────────────────────
    if p1_eq_df is not None:
        print(f"\n  ── Phase 1 plots ──")
        p1_equity(p1_eq_df, outdir)
        p1_drawdown(p1_eq_df, outdir)
        p1_rolling_sharpe(p1_eq_df, outdir, ann=p1_ann)

        if p1_w_df is not None:
            p1_weight_evolution(p1_w_df, outdir)
        if p1_a_df is not None:
            p1_actions_on_price(p1_a_df, p1_eq_df, outdir)

        p1_metrics_bar(p1_eq_df, p1_m_df, outdir, ann=p1_ann)

    # ── Generate Combined plots ───────────────────────────────────
    if p0_df is not None and p1_eq_df is not None:
        print(f"\n  ── Combined plots ──")
        combined_equity(p0_df, p1_eq_df, outdir, p0_ann)
        combined_sharpe_bar(p0_df, p1_eq_df, outdir, p0_ann, p1_ann)
        combined_metrics_table(p0_df, p1_eq_df, outdir, p0_ann, p1_ann)
    elif p0_df is not None:
        print(f"\n  [skip] Combined plots — Phase 1 data missing")
    elif p1_eq_df is not None:
        print(f"\n  [skip] Combined plots — Phase 0 data missing")

    print(f"\n  ✓ All plots saved to {outdir}/\n")


if __name__ == "__main__":
    main()
