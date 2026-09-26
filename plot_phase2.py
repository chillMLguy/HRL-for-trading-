"""
Phase 2 visualisation — reads the phase2_*.csv files written by
evaluate_phase2.py (run automatically at the end of run_all.py).

Usage
-----
  python plot_phase2.py --datadir <outdir>

Outputs (into the same directory):
  phase2_equity.png            — equity curves vs. B&H and vol-target B&H
  phase2_drawdown.png          — drawdown comparison
  phase2_rolling_sharpe.png    — rolling Sharpe comparison
  phase2_weight_evolution.png  — stacked weights per strategy (meta incl. cash)
  phase2_meta_diagnostics.png  — meta weights + leverage + entropy + turnover
  phase2_positions.png         — executed meta position vs. sub-agent proposals

Colors are a validated colorblind-safe categorical set; benchmarks are neutral
dashed lines so line style (not only hue) separates them.
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regime.allocators import AGENT_ORDER


# ── Style ──────────────────────────────────────────────────────────────────
# Strategy hues: 3 categorical slots validated for all-pairs CVD separation;
# benchmarks are neutrals with distinct dash patterns (secondary encoding).
STRAT_COLORS = {
    "meta_rl":       "#2a78d6",   # blue — the learned allocator (highlighted)
    "hmm":           "#eb6834",   # orange
    "vol_regime":    "#1baf7a",   # aqua
    "equal_weight":  "#8a897f",   # neutral gray (naive baseline)
    "buy_and_hold":  "#0b0b0b",   # black dashed benchmark
    "vol_target_bh": "#52514e",   # dark gray dash-dot benchmark
}
STRAT_STYLE = {                    # (linewidth, linestyle)
    "meta_rl":       (2.4, "-"),
    "hmm":           (1.3, "-"),
    "vol_regime":    (1.3, "-"),
    "equal_weight":  (1.1, "-"),
    "buy_and_hold":  (1.2, "--"),
    "vol_target_bh": (1.2, "-."),
}
STRAT_LABELS = {
    "meta_rl":       "Meta-RL",
    "hmm":           "HMM (fixed map)",
    "vol_regime":    "Vol Regime",
    "equal_weight":  "Equal Weight",
    "buy_and_hold":  "Buy & Hold",
    "vol_target_bh": "Vol-target B&H",
}
# Weight-stack hues (adjacent-pair validated: orange, blue, aqua, yellow)
AGENT_COLORS = {
    "aggressive":   "#eb6834",
    "balanced":     "#2a78d6",
    "conservative": "#1baf7a",
    "cash":         "#eda100",
}
INK, INK2 = "#0b0b0b", "#52514e"


def _setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#d9d8d3", "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "grid.color": "#eceae4",
        "grid.linewidth": 0.6, "font.family": "sans-serif",
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _order(names):
    """Benchmarks first (under), meta_rl last (on top)."""
    rank = {"buy_and_hold": 0, "vol_target_bh": 1, "equal_weight": 2,
            "vol_regime": 3, "hmm": 4, "meta_rl": 9}
    return sorted(names, key=lambda n: rank.get(n, 5))


def _line(ax, x, y, name):
    lw, ls = STRAT_STYLE.get(name, (1.2, "-"))
    ax.plot(x, y, color=STRAT_COLORS.get(name, "#333"), lw=lw, ls=ls,
            label=STRAT_LABELS.get(name, name),
            zorder=5 if name == "meta_rl" else 3)


def _endlabels(ax, series):
    """Selective direct labels at the right end — greedy collision avoidance:
    labels are placed in priority order (meta first, then benchmarks) and a
    label is skipped when its end value sits too close to one already placed.
    Text wears ink, not the series color; the legend still names every line."""
    prio = {"meta_rl": 0, "buy_and_hold": 1, "vol_target_bh": 2}
    ymin, ymax = ax.get_ylim()
    min_gap = (ymax - ymin) * 0.04
    placed = []
    for name, x_end, y_end in sorted(series, key=lambda s: prio.get(s[0], 5)):
        if any(abs(y_end - y) < min_gap for y in placed):
            continue
        ax.annotate(f" {STRAT_LABELS.get(name, name)}", (x_end, y_end),
                    fontsize=7, color=INK2, va="center")
        placed.append(y_end)


def _drawdown(eq):
    peak = np.maximum.accumulate(eq)
    return (eq - peak) / np.where(peak > 0, peak, 1e-10) * 100


def _load_dates(datadir, n_steps):
    """Date axis from phase2_returns.csv when available; else step index."""
    p = os.path.join(datadir, "phase2_returns.csv")
    if os.path.isfile(p):
        d = pd.read_csv(p, parse_dates=["date"])["date"]
        if len(d) >= n_steps:
            return d.iloc[:n_steps].values, True
    return np.arange(n_steps), False


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_equity(eq_df, dates, outdir):
    fig, ax = plt.subplots(figsize=(12, 5))
    names = _order([c for c in eq_df.columns if c != "step"])
    ends = []
    for name in names:
        y = eq_df[name].values[1:]         # equity after each dated step
        _line(ax, dates, y, name)
        ends.append((name, dates[-1], y[-1]))
    _endlabels(ax, ends)
    ax.set_title("Portfolio equity — Meta-RL vs. baselines and benchmarks")
    ax.set_ylabel("Cumulative equity (start = 1.0)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.6)
    ax.margins(x=0.01)
    fig.tight_layout()
    _save(fig, outdir, "phase2_equity.png")


def plot_drawdown(eq_df, dates, outdir):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for name in _order([c for c in eq_df.columns if c != "step"]):
        _line(ax, dates, _drawdown(eq_df[name].values)[1:], name)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.6)
    ax.margins(x=0.01)
    fig.tight_layout()
    _save(fig, outdir, "phase2_drawdown.png")


def plot_rolling_sharpe(ret_df, outdir, window=60, ann=252):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    dates = ret_df["date"].values if "date" in ret_df else np.arange(len(ret_df))
    for name in _order([c for c in ret_df.columns if c != "date"]):
        r = pd.Series(ret_df[name].values)
        rs = (r.rolling(window).mean()
              / r.rolling(window).std().replace(0, np.nan)) * np.sqrt(ann)
        _line(ax, dates, rs.values, name)
    ax.axhline(0, color="#b6b5ad", lw=0.8, ls=":")
    ax.set_title(f"Rolling {window}-bar Sharpe (annualized)")
    ax.set_ylabel("Sharpe")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.6)
    ax.margins(x=0.01)
    fig.tight_layout()
    _save(fig, outdir, "phase2_rolling_sharpe.png")


def _weight_stack(ax, sub, title):
    """Stacked weights for one allocator; includes cash when present (R1)."""
    steps = sub["step"].values
    cols   = [f"w_{a}" for a in AGENT_ORDER]
    labels = list(AGENT_ORDER)
    colors = [AGENT_COLORS[a] for a in AGENT_ORDER]
    if "w_cash" in sub.columns and sub["w_cash"].notna().any():
        cols += ["w_cash"]
        labels += ["cash"]
        colors += [AGENT_COLORS["cash"]]
    W = sub[cols].fillna(0.0).values
    ax.stackplot(steps, *W.T, labels=labels, colors=colors, alpha=0.9,
                 edgecolor="white", linewidth=0.6)   # white gaps between bands
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("weight")
    ax.set_title(title, fontsize=9)
    ax.grid(True, axis="y", alpha=0.4)
    ax.margins(x=0)


def plot_weight_evolution(w_df, outdir):
    allocs = [a for a in ["meta_rl", "hmm", "vol_regime", "equal_weight"]
              if a in set(w_df["allocator"])]
    fig, axes = plt.subplots(len(allocs), 1, figsize=(12, 2.4 * len(allocs)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, allocs):
        _weight_stack(ax, w_df[w_df["allocator"] == name].sort_values("step"),
                      STRAT_LABELS.get(name, name))
    axes[0].legend(loc="upper right", fontsize=7, ncol=4)
    axes[-1].set_xlabel("Step")
    fig.suptitle("Allocation weights per strategy", fontweight="bold", y=1.0)
    fig.tight_layout()
    _save(fig, outdir, "phase2_weight_evolution.png")


def plot_meta_diagnostics(w_df, outdir):
    """Meta-RL: weight stack (+cash), vol-target leverage, entropy, turnover."""
    if "meta_rl" not in set(w_df["allocator"]):
        print("  [skip] meta diagnostics — no meta_rl weights found")
        return
    sub = w_df[w_df["allocator"] == "meta_rl"].sort_values("step")
    steps = sub["step"].values
    cols = [f"w_{a}" for a in AGENT_ORDER]
    has_cash  = "w_cash" in sub.columns and sub["w_cash"].notna().any()
    has_lever = "lever" in sub.columns and sub["lever"].notna().any()
    W = sub[cols + (["w_cash"] if has_cash else [])].fillna(0.0).values

    p = np.clip(W, 1e-12, 1.0)
    entropy = -(p * np.log(p)).sum(axis=1) / np.log(W.shape[1])
    turnover = np.concatenate([[0.0], np.abs(np.diff(W, axis=0)).sum(axis=1)])

    n_panels = 3 + int(has_lever)
    heights = [2.2] + ([1] if has_lever else []) + [1, 1]
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.2 * n_panels),
                             sharex=True, gridspec_kw={"height_ratios": heights})
    k = 0
    _weight_stack(axes[k], sub, "Meta-RL — learned allocation"
                  + (" (incl. cash arm)" if has_cash else ""))
    axes[k].legend(loc="upper right", fontsize=7, ncol=4)
    k += 1

    if has_lever:
        axes[k].plot(steps, sub["lever"].values, color="#2a78d6", lw=1.4)
        axes[k].axhline(1.0, color="#b6b5ad", lw=0.8, ls=":")
        axes[k].set_ylabel("leverage")
        axes[k].set_title("Vol-target overlay leverage (R2)", fontsize=9)
        axes[k].grid(True, axis="y", alpha=0.4)
        k += 1

    axes[k].plot(steps, entropy, color="#eb6834", lw=1.2)
    axes[k].set_ylim(0, 1.05)
    axes[k].set_ylabel("entropy")
    axes[k].set_title("Allocation entropy (1 = fully mixed, 0 = collapsed)",
                      fontsize=9)
    axes[k].grid(True, axis="y", alpha=0.4)
    k += 1

    axes[k].plot(steps, turnover, color="#1baf7a", lw=1.0)
    axes[k].set_ylabel("turnover")
    axes[k].set_xlabel("Step")
    axes[k].set_title(r"Weight turnover  $\Sigma_i\,|w_{i,t} - w_{i,t-1}|$",
                      fontsize=9)
    axes[k].grid(True, axis="y", alpha=0.4)

    fig.tight_layout()
    _save(fig, outdir, "phase2_meta_diagnostics.png")


def plot_positions(a_df, dates, outdir):
    """Executed meta position vs. the sub-agents' proposed positions."""
    if "meta_rl" not in set(a_df["allocator"]):
        return
    sub = a_df[a_df["allocator"] == "meta_rl"].sort_values("step")
    x = dates[:len(sub)]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for agent in AGENT_ORDER:
        ax.plot(x, sub[f"{agent}_action"].values, lw=0.9, alpha=0.65,
                color=AGENT_COLORS[agent], label=f"{agent} proposal")
    ax.plot(x, sub["blended_action"].values, lw=2.2, color=INK,
            label="executed position")
    ax.axhline(0, color="#b6b5ad", lw=0.8, ls=":")
    ax.set_title("Executed position vs. sub-agent proposals (position units)")
    ax.set_ylabel("position")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.6)
    ax.margins(x=0.01)
    fig.tight_layout()
    _save(fig, outdir, "phase2_positions.png")


def _save(fig, outdir, fname):
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 — comparison plots")
    parser.add_argument("--datadir", default=".",
                        help="Directory with phase2_*.csv files")
    parser.add_argument("--window", type=int, default=60,
                        help="Rolling-Sharpe window (bars)")
    args = parser.parse_args()

    _setup_style()
    d = args.datadir
    eq_path = os.path.join(d, "phase2_equity_curves.csv")
    w_path  = os.path.join(d, "phase2_weights.csv")
    for pth in (eq_path, w_path):
        if not os.path.isfile(pth):
            print(f"  ERROR: {pth} not found. Run evaluate_phase2.py first.")
            return

    eq_df = pd.read_csv(eq_path)
    w_df  = pd.read_csv(w_path)
    n_steps = len(eq_df) - 1
    dates, dated = _load_dates(d, n_steps)

    print(f"\n  Generating Phase 2 plots from {d}/ "
          f"({'date' if dated else 'step'} axis) ...\n")
    plot_equity(eq_df, dates, d)
    plot_drawdown(eq_df, dates, d)

    ret_path = os.path.join(d, "phase2_returns.csv")
    if os.path.isfile(ret_path):
        plot_rolling_sharpe(pd.read_csv(ret_path, parse_dates=["date"]),
                            d, window=args.window)

    plot_weight_evolution(w_df, d)
    plot_meta_diagnostics(w_df, d)

    a_path = os.path.join(d, "phase2_actions.csv")
    if os.path.isfile(a_path):
        plot_positions(pd.read_csv(a_path), dates, d)

    print("\n  All Phase 2 plots saved.\n")


if __name__ == "__main__":
    main()
