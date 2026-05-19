"""
Unified visualisation for Phase 0 and Phase 1 results.

Replaces plot_results.py, plot_phase1.py, and plot_all.py with a single
file that produces 15 PNGs in a consistent light-academic style.

Usage
-----
  python plot.py --plots all
  python plot.py --plots p0_equity,signals --outdir plots/
  python plot.py --plots all --ticker ^DJI \
                 --test_start 2022-01-01 --test_end 2022-12-31

Reads (any missing CSV → the affected plots are skipped, not errored):
  equity_curves.csv          (Phase 0 — from evaluate_agents.py)
  phase1_equity_curves.csv   (Phase 1 — from evaluate_phase1.py)
  phase1_weights.csv
  phase1_actions.csv
  phase1_metrics.csv
  regime_table.csv           (optional — rendered as heatmap)

Outputs in --outdir (default = --datadir):
  p0_equity.png, p0_drawdown.png, p0_rolling_sharpe.png,
  p0_return_dist.png, p0_metrics_bar.png,
  p1_equity.png, p1_drawdown.png, p1_rolling_sharpe.png,
  p1_weight_evolution.png, p1_actions_on_price.png, p1_metrics_bar.png,
  combined_equity.png, combined_sharpe_bar.png,
  combined_metrics_table.png,
  signals.png                (requires saved models + network for ^DJI)
  regime_table.png           (only if regime_table.csv present)
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
import matplotlib.patches as mpatches

import config


# ═══════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════

BARS_PER_YEAR = config.BARS_PER_YEAR

P0_COLORS = {
    "aggressive":         "#E74C3C",
    "growth":             "#E67E22",
    "balanced":           "#3498DB",
    "conservative":       "#1ABC9C",
    "ultra_conservative": "#9B59B6",
    "buy_&_hold":         "#7F8C8D",
}

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

SIGNAL_BUY  = "#2ECC71"
SIGNAL_SELL = "#E74C3C"
SIGNAL_FLIP = "#F39C12"


PLOT_REGISTRY = [
    "p0_equity", "p0_drawdown", "p0_rolling_sharpe",
    "p0_return_dist", "p0_metrics_bar",
    "p1_equity", "p1_drawdown", "p1_rolling_sharpe",
    "p1_weight_evolution", "p1_actions_on_price", "p1_metrics_bar",
    "combined_equity", "combined_sharpe_bar", "combined_metrics_table",
    "signals", "regime_table",
]


# ═══════════════════════════════════════════════════════════════════
#  Style
# ═══════════════════════════════════════════════════════════════════

def _apply_style():
    plt.rcParams.update({
        "figure.facecolor":   "#FAFBFC",
        "axes.facecolor":     "#FFFFFF",
        "axes.edgecolor":     "#D5D8DC",
        "axes.labelcolor":    "#2C3E50",
        "axes.titlesize":     12,
        "axes.titleweight":   "semibold",
        "axes.labelsize":     10,
        "xtick.color":        "#566573",
        "ytick.color":        "#566573",
        "xtick.labelsize":    8.5,
        "ytick.labelsize":    8.5,
        "text.color":         "#2C3E50",
        "grid.color":         "#EAECEE",
        "grid.linewidth":     0.6,
        "grid.linestyle":     "--",
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Helvetica", "DejaVu Sans", "Arial"],
        "legend.fontsize":    8.5,
        "legend.framealpha":  0.92,
        "legend.edgecolor":   "#D5D8DC",
        "figure.dpi":         150,
        "savefig.dpi":        200,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.15,
    })


def _color_for(name):
    if name in P0_COLORS:
        return P0_COLORS[name]
    if name in P1_COLORS:
        return P1_COLORS[name]
    return "#566573"


def _label_for(name):
    if name in P1_LABELS:
        return P1_LABELS[name]
    return name.replace("_", " ").title()


def _watermark(fig, text="HRL Trading System"):
    fig.text(0.99, 0.005, text, fontsize=7, color="#BDC3C7",
             ha="right", va="bottom", style="italic")


def _save(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path, facecolor=fig.get_facecolor())
    print(f"  Saved → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Loaders (return None on missing/invalid — caller skips that plot)
# ═══════════════════════════════════════════════════════════════════

def _load_p0(datadir):
    """Phase 0 equity curves, rebased to 100. Returns DataFrame or None."""
    path = os.path.join(datadir, "equity_curves.csv")
    if not os.path.isfile(path):
        return None
    raw = pd.read_csv(path)
    raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]
    if raw.empty or len(raw) < 2:
        print(f"  [warn] {path} is empty or too short")
        return None
    return raw / raw.iloc[0] * 100


def _load_p1_equity(datadir):
    path = os.path.join(datadir, "phase1_equity_curves.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    if "step" not in df.columns:
        df = df.reset_index().rename(columns={"index": "step"})
    return df


def _load_p1_weights(datadir):
    path = os.path.join(datadir, "phase1_weights.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    if "allocator" not in df.columns or "step" not in df.columns:
        print(f"  [warn] {path} missing required columns")
        return None
    return df


def _load_p1_actions(datadir):
    path = os.path.join(datadir, "phase1_actions.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    if "allocator" not in df.columns or "blended_action" not in df.columns:
        print(f"  [warn] {path} missing required columns")
        return None
    return df


def _load_p1_metrics(datadir):
    path = os.path.join(datadir, "phase1_metrics.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path, index_col=0)


def _load_regime_table(datadir):
    path = os.path.join(datadir, "regime_table.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def _load_prices(ticker, start, end, interval, cache_dir):
    """Download (or cache-hit) close prices. Returns pd.Series or None."""
    os.makedirs(cache_dir, exist_ok=True)
    safe_ticker = ticker.replace("^", "").replace("/", "_")
    cache_path = os.path.join(
        cache_dir, f"{safe_ticker}_{start}_{end}_{interval}.csv")

    if os.path.isfile(cache_path):
        s = pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0]
        if len(s) > 0:
            print(f"  Prices loaded from cache: {cache_path}")
            return s.dropna()

    try:
        import yfinance as yf
    except ImportError:
        print("  [warn] yfinance not installed — cannot fetch prices")
        return None

    print(f"  Downloading {ticker} {start} → {end} (interval={interval})...")
    try:
        raw = yf.download(ticker, start=start, end=end, interval=interval,
                          auto_adjust=True, progress=False)
    except Exception as e:
        print(f"  [warn] yfinance download failed: {e}")
        return None

    if raw.empty:
        print(f"  [warn] yfinance returned empty data for {ticker}")
        return None
    prices = raw["Close"].squeeze().dropna()
    prices.to_csv(cache_path)
    return prices


# ═══════════════════════════════════════════════════════════════════
#  Metrics & primitives
# ═══════════════════════════════════════════════════════════════════

def _drawdown(eq):
    """Drawdown in percent from equity array."""
    peak = np.maximum.accumulate(eq)
    return (eq - peak) / np.where(peak > 0, peak, 1e-10) * 100


def _rolling_sharpe(returns, window, ann):
    r = pd.Series(returns)
    mu = r.rolling(window).mean()
    sig = r.rolling(window).std()
    return (mu / sig.replace(0, np.nan)) * np.sqrt(ann)


def _adaptive_window(ann):
    """Rolling Sharpe window scaled to bars-per-year."""
    return max(20, ann // 8)


def _sharpe(r, ann):
    s = float(np.std(r))
    return float(np.sqrt(ann) * np.mean(r) / s) if s > 1e-10 else 0.0


def _sortino(r, ann):
    down = r[r < 0]
    ds = float(np.std(down)) if len(down) > 1 else 1e-10
    return float(np.sqrt(ann) * np.mean(r) / ds)


def _max_drawdown(eq):
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, 1e-10)
    return float(dd.max())


def _calmar(r, eq, ann):
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


def _compute_metrics(r, eq, ann):
    return {
        "Total ret (%)": round((eq[-1] / eq[0] - 1) * 100, 2),
        "Ann. ret (%)":  round(float(np.mean(r)) * ann * 100, 2),
        "Sharpe":        round(_sharpe(r, ann), 3),
        "Sortino":       round(_sortino(r, ann), 3),
        "MaxDD (%)":     round(_max_drawdown(eq) * 100, 2),
        "Calmar":        round(_calmar(r, eq, ann), 3),
        "CVaR 95% (%)":  round(_cvar(r) * 100, 3),
        "Omega":         round(_omega(r), 3),
        "Win rate (%)":  round(float((r > 0).mean()) * 100, 1),
    }


def _p1_alloc_names(eq_df):
    skip = {"step", "buy_and_hold"}
    return [c for c in eq_df.columns if c not in skip]


# ═══════════════════════════════════════════════════════════════════
#  Phase 0 plots
# ═══════════════════════════════════════════════════════════════════

def plot_p0_equity(df, outdir, ann):
    fig, ax = plt.subplots(figsize=(13, 5))
    for name in df.columns:
        lw = 1.8 if name == "buy_&_hold" else 1.3
        ls = "--" if name == "buy_&_hold" else "-"
        alpha = 0.65 if name == "buy_&_hold" else 0.9
        ax.plot(df[name].values, color=_color_for(name),
                lw=lw, ls=ls, alpha=alpha, label=_label_for(name))
    ax.axhline(100, color="#BDC3C7", lw=0.6, ls=":", alpha=0.6)
    ax.set_ylabel("Portfolio Value (indexed 100)")
    ax.set_xlabel("Step")
    ax.set_title("Phase 0 — λ-Spectrum Agent Equity Curves", pad=12)
    ax.legend(loc="upper left")
    ax.grid(True)
    _watermark(fig)
    _save(fig, outdir, "p0_equity.png")


def plot_p0_drawdown(df, outdir):
    fig, ax = plt.subplots(figsize=(13, 4))
    last_dd = None
    for name in df.columns:
        dd = _drawdown(df[name].values)
        last_dd = dd
        lw = 1.5 if name == "buy_&_hold" else 1.1
        ls = "--" if name == "buy_&_hold" else "-"
        ax.plot(dd, color=_color_for(name), lw=lw, ls=ls,
                alpha=0.85, label=_label_for(name))
    if last_dd is not None:
        ax.fill_between(range(len(last_dd)), last_dd, 0,
                        alpha=0.03, color="#E74C3C")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Step")
    ax.set_title("Phase 0 — Drawdown Comparison", pad=12)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True)
    _watermark(fig)
    _save(fig, outdir, "p0_drawdown.png")


def plot_p0_rolling_sharpe(df, outdir, ann):
    window = _adaptive_window(ann)
    fig, ax = plt.subplots(figsize=(13, 4))
    for name in df.columns:
        r = df[name].pct_change().dropna()
        rs = _rolling_sharpe(r.values, window, ann)
        lw = 1.4 if name == "buy_&_hold" else 1.0
        ls = "--" if name == "buy_&_hold" else "-"
        ax.plot(rs.values, color=_color_for(name), lw=lw, ls=ls,
                alpha=0.85, label=_label_for(name))
    ax.axhline(0, color="#BDC3C7", lw=0.8, ls=":")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xlabel("Step")
    ax.set_title(f"Phase 0 — Rolling {window}-bar Sharpe Ratio", pad=12)
    ax.legend(loc="best")
    ax.grid(True)
    _watermark(fig)
    _save(fig, outdir, "p0_rolling_sharpe.png")


def plot_p0_return_dist(df, outdir):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in df.columns:
        if name == "buy_&_hold":
            continue
        r = df[name].pct_change().dropna().values * 100
        ax.hist(r, bins=70, alpha=0.35, color=_color_for(name),
                label=_label_for(name), density=True, histtype="stepfilled",
                edgecolor=_color_for(name), linewidth=0.5)
        ax.axvline(float(np.mean(r)), color=_color_for(name),
                   lw=1.3, ls="--", alpha=0.7)
    ax.set_xlabel("Bar Return (%)")
    ax.set_ylabel("Density")
    ax.set_title("Phase 0 — Return Distribution", pad=12)
    ax.legend()
    ax.grid(True)
    _watermark(fig)
    _save(fig, outdir, "p0_return_dist.png")


def plot_p0_metrics_bar(df, outdir, ann):
    agents = list(df.columns)
    metrics_data = {}
    for name in agents:
        eq = df[name].values
        r = np.diff(eq) / eq[:-1]
        metrics_data[name] = _compute_metrics(r, eq / eq[0], ann)

    metric_keys = ["Sharpe", "Sortino", "Calmar", "Omega"]
    x = np.arange(len(metric_keys))
    width = 0.8 / len(agents)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, name in enumerate(agents):
        vals = [metrics_data[name][k] for k in metric_keys]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.9,
                      label=_label_for(name), color=_color_for(name),
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=6.5, color="#2C3E50")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, fontsize=10)
    ax.set_ylabel("Value")
    ax.set_title("Phase 0 — Risk-Adjusted Metrics Comparison", pad=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.axhline(0, color="#BDC3C7", lw=0.6)
    ax.grid(True, axis="y")
    _watermark(fig)
    _save(fig, outdir, "p0_metrics_bar.png")


# ═══════════════════════════════════════════════════════════════════
#  Phase 1 plots
# ═══════════════════════════════════════════════════════════════════

def plot_p1_equity(eq_df, outdir):
    fig, ax = plt.subplots(figsize=(13, 5))
    for aname in _p1_alloc_names(eq_df):
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
    ax.legend(loc="best")
    ax.grid(True)
    _watermark(fig)
    _save(fig, outdir, "p1_equity.png")


def plot_p1_drawdown(eq_df, outdir):
    fig, ax = plt.subplots(figsize=(13, 4))
    for aname in _p1_alloc_names(eq_df):
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
    ax.legend(loc="lower left")
    ax.grid(True)
    _watermark(fig)
    _save(fig, outdir, "p1_drawdown.png")


def plot_p1_rolling_sharpe(eq_df, outdir, ann):
    window = _adaptive_window(ann)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    steps = eq_df["step"].values
    for aname in _p1_alloc_names(eq_df):
        eq = eq_df[aname].values
        rets = np.diff(eq) / eq[:-1]
        rs = _rolling_sharpe(rets, window, ann)
        ax.plot(steps[1:], rs.values,
                color=P1_COLORS.get(aname, "#566573"),
                lw=1.1, alpha=0.85,
                label=P1_LABELS.get(aname, aname))
    if "buy_and_hold" in eq_df.columns:
        eq_bh = eq_df["buy_and_hold"].values
        rets_bh = np.diff(eq_bh) / eq_bh[:-1]
        rs_bh = _rolling_sharpe(rets_bh, window, ann)
        ax.plot(steps[1:], rs_bh.values,
                color=P1_COLORS["buy_and_hold"],
                lw=1.2, ls="--", alpha=0.65, label="Buy & Hold")
    ax.axhline(0, color="#BDC3C7", lw=0.8, ls=":")
    ax.set_xlabel("Step")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title(f"Phase 1 — Rolling {window}-bar Sharpe Ratio", pad=12)
    ax.legend(loc="best")
    ax.grid(True)
    _watermark(fig)
    _save(fig, outdir, "p1_rolling_sharpe.png")


def plot_p1_weight_evolution(w_df, outdir):
    allocs = list(w_df["allocator"].unique())
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

        # Sentiment tinting: green when aggressive > conservative, red opposite.
        # Restored from plot_phase1.py — was silently dropped in plot_all.py.
        if len(agent_names) >= 2:
            w_agg = weights[:, 0]
            w_con = weights[:, -1]
            sentiment = w_agg - w_con
            for i in range(len(steps) - 1):
                s = sentiment[i]
                if abs(s) > 0.02:
                    c = "#2ECC71" if s > 0 else "#E74C3C"
                    a = min(abs(float(s)) * 0.3, 0.15)
                    ax.axvspan(steps[i], steps[i + 1], color=c,
                               alpha=a, lw=0, zorder=0)

        ax.set_ylabel("Weight")
        ax.set_ylim(0, 1)
        ax.set_title(P1_LABELS.get(aname, aname), fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper right",
                   fontsize=7.5, ncol=len(agent_names))

    fig.suptitle("Phase 1 — Agent Weight Evolution per Allocator",
                 fontsize=12, fontweight="semibold", y=1.01)
    fig.tight_layout()
    _watermark(fig)
    _save(fig, outdir, "p1_weight_evolution.png")


def plot_p1_actions_on_price(act_df, eq_df, outdir, prices=None):
    """
    Top: actual prices (yfinance) when available, else equity proxy.
    Bottom: blended action per allocator.
    """
    fig, (ax_price, ax_act) = plt.subplots(
        2, 1, figsize=(13, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]})

    allocs = _p1_alloc_names(eq_df)
    steps = eq_df["step"].values

    if prices is not None and len(prices) >= len(steps):
        px = prices.values[:len(steps)]
        ax_price.plot(steps, px, color="#2C3E50", lw=1.1, alpha=0.85,
                      label="Price")
        ax_price.set_ylabel("Price")
        ax_price.set_title("Phase 1 — Blended Action on Price", pad=10)
    else:
        if prices is not None:
            print(f"  [warn] prices length {len(prices)} < steps {len(steps)} — "
                  "falling back to equity proxy")
        if allocs:
            ax_price.plot(steps, eq_df[allocs[0]].values,
                          color="#2C3E50", lw=1.0, alpha=0.7,
                          label="Equity proxy")
        ax_price.set_ylabel("Equity (proxy)")
        ax_price.set_title("Phase 1 — Blended Action on Equity (price proxy)",
                           pad=10)

    if "buy_and_hold" in eq_df.columns and prices is None:
        ax_price.plot(steps, eq_df["buy_and_hold"].values,
                      color=P1_COLORS["buy_and_hold"],
                      lw=0.8, ls="--", alpha=0.5, label="Buy & Hold")

    ax_price.legend(loc="best", fontsize=8)
    ax_price.grid(True)

    for aname in allocs:
        sub = act_df[act_df["allocator"] == aname]
        ax_act.plot(sub["step"].values, sub["blended_action"].values,
                    color=P1_COLORS.get(aname, "#566573"),
                    lw=0.8, alpha=0.85,
                    label=P1_LABELS.get(aname, aname))

    ax_act.set_ylabel("Blended Action")
    ax_act.set_xlabel("Step")
    ax_act.set_ylim(-1.15, 1.15)
    ax_act.axhline(0, color="#BDC3C7", lw=0.8, ls=":")
    ax_act.legend(loc="best", fontsize=8)
    ax_act.grid(True)

    fig.tight_layout()
    _watermark(fig)
    _save(fig, outdir, "p1_actions_on_price.png")


def plot_p1_metrics_bar(eq_df, m_df, outdir, ann):
    if m_df is not None and not m_df.empty:
        strategies = list(m_df.columns)
        metrics_data = {s: m_df[s].to_dict() for s in strategies}
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
    for i, s in enumerate(strategies):
        col = P1_COLORS.get(s, _color_for(s))
        vals = []
        for k in metric_keys:
            v = metrics_data[s].get(k, 0.0)
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(0.0)
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
    ax.legend(loc="upper right", fontsize=8)
    ax.axhline(0, color="#BDC3C7", lw=0.6)
    ax.grid(True, axis="y")
    _watermark(fig)
    _save(fig, outdir, "p1_metrics_bar.png")


# ═══════════════════════════════════════════════════════════════════
#  Combined plots
# ═══════════════════════════════════════════════════════════════════

def plot_combined_equity(p0_df, p1_eq_df, outdir):
    """Both panels rebased to 100 — fixes the p0=100 / p1=raw mismatch."""
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(16, 5.5),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.25})

    # p0_df is already rebased to 100 by _load_p0
    for name in p0_df.columns:
        lw = 1.5 if name == "buy_&_hold" else 1.2
        ls = "--" if name == "buy_&_hold" else "-"
        alpha = 0.6 if name == "buy_&_hold" else 0.9
        ax0.plot(p0_df[name].values, color=_color_for(name),
                 lw=lw, ls=ls, alpha=alpha, label=_label_for(name))
    ax0.axhline(100, color="#BDC3C7", lw=0.6, ls=":")
    ax0.set_ylabel("Indexed Equity (start=100)")
    ax0.set_xlabel("Step")
    ax0.set_title("Phase 0 — Individual Agents", fontsize=11, pad=10)
    ax0.legend(loc="upper left", fontsize=7.5)
    ax0.grid(True)

    # Rebase Phase 1 to 100 for visual consistency.
    allocs = _p1_alloc_names(p1_eq_df)
    for aname in allocs:
        eq = p1_eq_df[aname].values
        ax1.plot(p1_eq_df["step"], eq / eq[0] * 100,
                 color=P1_COLORS.get(aname, "#566573"),
                 lw=1.5, label=P1_LABELS.get(aname, aname))
    if "buy_and_hold" in p1_eq_df.columns:
        eq_bh = p1_eq_df["buy_and_hold"].values
        ax1.plot(p1_eq_df["step"], eq_bh / eq_bh[0] * 100,
                 color=P1_COLORS["buy_and_hold"],
                 lw=1.3, ls="--", alpha=0.65, label="Buy & Hold")
    ax1.axhline(100, color="#BDC3C7", lw=0.6, ls=":")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Indexed Equity (start=100)")
    ax1.set_title("Phase 1 — Meta-Controlled Allocators", fontsize=11, pad=10)
    ax1.legend(loc="upper left", fontsize=7.5)
    ax1.grid(True)

    fig.suptitle("Equity Comparison — Phase 0 vs Phase 1",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02,
             "Note: Panels may reflect different test periods. Both equity "
             "series rebased to 100 for visual comparability.",
             ha="center", fontsize=7, color="#95A5A6", style="italic")
    fig.tight_layout()
    _watermark(fig)
    _save(fig, outdir, "combined_equity.png")


def plot_combined_sharpe_bar(p0_df, p1_eq_df, outdir, p0_ann, p1_ann):
    data = {}
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

    for aname in _p1_alloc_names(p1_eq_df):
        eq = p1_eq_df[aname].values
        r = np.diff(eq) / eq[:-1]
        data[f"P1: {P1_LABELS.get(aname, aname)}"] = {
            "sharpe": _sharpe(r, p1_ann),
            "color": P1_COLORS.get(aname, "#566573"),
            "phase": 1,
        }

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
    ax.set_title("Sharpe Ratio — All Strategies", pad=12)
    ax.axhline(0, color="#BDC3C7", lw=0.8)
    ax.grid(True, axis="y")

    p0_count = sum(1 for d in data.values() if d["phase"] == 0)
    if 0 < p0_count < len(names):
        ax.axvline(p0_count - 0.5, color="#D5D8DC", lw=1.0, ls="--")
        ymax = ax.get_ylim()[1]
        ax.text(p0_count / 2 - 0.5, ymax * 0.95, "Phase 0",
                ha="center", fontsize=8, color="#7F8C8D", style="italic")
        ax.text(p0_count + (len(names) - p0_count) / 2 - 0.5,
                ymax * 0.95, "Phase 1 + B&H",
                ha="center", fontsize=8, color="#7F8C8D", style="italic")

    fig.text(0.5, -0.02,
             "Note: Phase 0 and Phase 1 may use different test periods. "
             "B&H shown is from the Phase 1 evaluation only.",
             ha="center", fontsize=7, color="#95A5A6", style="italic")
    _watermark(fig)
    _save(fig, outdir, "combined_sharpe_bar.png")


def plot_combined_metrics_table(p0_df, p1_eq_df, outdir, p0_ann, p1_ann):
    all_metrics = {}
    for name in p0_df.columns:
        if name == "buy_&_hold":
            continue
        eq = p0_df[name].values
        r = np.diff(eq) / eq[:-1]
        all_metrics[f"P0 · {_label_for(name)}"] = _compute_metrics(
            r, eq / eq[0], p0_ann)

    for aname in _p1_alloc_names(p1_eq_df):
        eq = p1_eq_df[aname].values
        r = np.diff(eq) / eq[:-1]
        all_metrics[f"P1 · {P1_LABELS.get(aname, aname)}"] = _compute_metrics(
            r, eq / eq[0], p1_ann)

    if "buy_and_hold" in p1_eq_df.columns:
        eq_bh = p1_eq_df["buy_and_hold"].values
        r_bh = np.diff(eq_bh) / eq_bh[:-1]
        all_metrics["Buy & Hold"] = _compute_metrics(
            r_bh, eq_bh / eq_bh[0], p1_ann)

    df_m = pd.DataFrame(all_metrics).T
    display_cols = ["Total ret (%)", "Ann. ret (%)", "Sharpe", "Sortino",
                    "MaxDD (%)", "Calmar", "CVaR 95% (%)", "Win rate (%)"]
    df_m = df_m[[c for c in display_cols if c in df_m.columns]]

    n_rows = len(df_m)
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
             "periods. B&H is from the Phase 1 evaluation.",
             ha="center", fontsize=7, color="#95A5A6", style="italic")
    _watermark(fig)
    _save(fig, outdir, "combined_metrics_table.png")


# ═══════════════════════════════════════════════════════════════════
#  Signals (price + agent positions, requires saved models)
# ═══════════════════════════════════════════════════════════════════

def _rollout_positions(model, prices, lam, bars_per_year, cnn_model_path):
    from env.trading_env import TradingEnv
    env = TradingEnv(prices, lam=lam, bars_per_year=bars_per_year,
                     eval_mode=True, cnn_model_path=cnn_model_path)
    obs, _ = env.reset()
    warmup = env.warmup
    positions = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        positions.append(info["position"])
    return np.array(positions, dtype=np.float32), warmup


def _detect_trades(pos, threshold=0.1):
    buys, sells, flips = [], [], []
    for i in range(1, len(pos)):
        p, c = pos[i - 1], pos[i]
        if (p > 0.1 and c < -0.1) or (p < -0.1 and c > 0.1):
            flips.append(i)
        elif p < threshold and c >= threshold:
            buys.append(i)
        elif p > -threshold and c <= -threshold:
            sells.append(i)
    return buys, sells, flips


def _signals_panel(ax, px, pos, name, color):
    n = min(len(px), len(pos))
    px, pos = px[:n], pos[:n]
    xs = np.arange(n)

    ax.plot(xs, px, color="#2C3E50", lw=1.0, alpha=0.85, zorder=2)

    for i in range(n - 1):
        p = pos[i]
        if p > 0.05:
            ax.axvspan(i, i + 1, alpha=min(p * 0.18, 0.18),
                       color="#3498DB", lw=0)
        elif p < -0.05:
            ax.axvspan(i, i + 1, alpha=min(-p * 0.18, 0.18),
                       color="#E74C3C", lw=0)

    buys, sells, flips = _detect_trades(pos)
    if buys:
        ax.scatter(buys, px[buys], marker="^", color=SIGNAL_BUY,
                   s=45, zorder=5, lw=0)
    if sells:
        ax.scatter(sells, px[sells], marker="v", color=SIGNAL_SELL,
                   s=45, zorder=5, lw=0)
    if flips:
        ax.scatter(flips, px[flips], marker="D", color=SIGNAL_FLIP,
                   s=28, zorder=5, lw=0)

    ax2 = ax.twinx()
    ax2.plot(xs, pos, color=color, lw=0.9, alpha=0.65, ls="--")
    ax2.axhline(0, color="#BDC3C7", lw=0.3, alpha=0.6)
    ax2.set_ylim(-2, 2)
    ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax2.set_ylabel("position", fontsize=8, color=color)
    ax2.tick_params(labelsize=7, colors=color)

    legend = [
        mpatches.Patch(color="#3498DB", alpha=0.5, label="long"),
        mpatches.Patch(color="#E74C3C", alpha=0.5, label="short"),
        plt.Line2D([0], [0], marker="^", color="w",
                   markerfacecolor=SIGNAL_BUY, ms=7, ls="None", label="go long"),
        plt.Line2D([0], [0], marker="v", color="w",
                   markerfacecolor=SIGNAL_SELL, ms=7, ls="None", label="go short"),
        plt.Line2D([0], [0], marker="D", color="w",
                   markerfacecolor=SIGNAL_FLIP, ms=6, ls="None", label="flip"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=7, ncol=3)
    ax.set_ylabel("Price")
    ax.set_title(f"{_label_for(name)}  —  {len(buys)} buys · "
                 f"{len(sells)} sells · {len(flips)} flips")
    ax.grid(True, alpha=0.4)


def plot_signals(prices, modeldir, outdir, interval, ticker, start, end,
                 cnn_model_path=None):
    try:
        from stable_baselines3 import SAC
        from env.trading_env import AGENT_PRESETS
    except ImportError as e:
        print(f"  [warn] cannot import SB3/env: {e} — skipping signals")
        return

    if prices is None or len(prices) == 0:
        print("  [skip signals] no prices available")
        return

    ann = BARS_PER_YEAR.get(interval, 252)
    signal_agents = []
    for name in AGENT_PRESETS:
        path = os.path.join(modeldir, "models", f"{name}_agent",
                            "best_model.zip")
        if os.path.exists(path):
            signal_agents.append(name)
        else:
            print(f"  [skip signals] {name} — no model at {path}")

    if not signal_agents:
        print("  [skip signals] no agent models found")
        return

    fig, axes = plt.subplots(
        len(signal_agents), 1,
        figsize=(16, 4.5 * len(signal_agents)))
    if len(signal_agents) == 1:
        axes = [axes]

    for ax, name in zip(axes, signal_agents):
        try:
            lam = AGENT_PRESETS[name]
            model_path = os.path.join(modeldir, "models", f"{name}_agent",
                                      "best_model")
            model = SAC.load(model_path)
            print(f"  Rolling out {name} (λ={lam})...")
            pos, warmup = _rollout_positions(
                model, prices, lam, ann, cnn_model_path)
            px_plot = prices.values[warmup:]
            color = AGENT_COLORS.get(name, _color_for(name))
            _signals_panel(ax, px_plot, pos, name, color)
        except Exception as e:
            ax.set_title(f"{_label_for(name)} — error: {e}")
            print(f"  [err] {name}: {e}")

    fig.suptitle(
        f"{ticker}  {start} → {end}  |  "
        "blue=long · red=short · ▲buy · ▼sell · ◆flip",
        fontsize=10, y=1.005)
    fig.tight_layout()
    _watermark(fig)
    _save(fig, outdir, "signals.png")


# ═══════════════════════════════════════════════════════════════════
#  Regime table heatmap (optional)
# ═══════════════════════════════════════════════════════════════════

def plot_regime_table(r_df, outdir):
    if "agent" not in r_df.columns:
        print("  [skip regime_table] missing 'agent' column")
        return
    df = r_df.set_index("agent")
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.6 * len(df) + 1.5)))
    im = ax.imshow(df.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, fontsize=9)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels([_label_for(i) for i in df.index], fontsize=9)
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            ax.text(j, i, f"{df.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=8, color="#2C3E50")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title("Per-Regime Agent Allocation", pad=10)
    fig.tight_layout()
    _watermark(fig)
    _save(fig, outdir, "regime_table.png")


# ═══════════════════════════════════════════════════════════════════
#  Main / dispatch
# ═══════════════════════════════════════════════════════════════════

def _resolve_plots(arg):
    if arg in (None, "", "all"):
        return list(PLOT_REGISTRY)
    requested = [p.strip() for p in arg.split(",") if p.strip()]
    unknown = [p for p in requested if p not in PLOT_REGISTRY]
    if unknown:
        raise SystemExit(
            f"Unknown plot name(s): {unknown}\n"
            f"Available: {', '.join(PLOT_REGISTRY)}")
    return requested


def main():
    parser = argparse.ArgumentParser(
        description="Unified plotting for the HRL trading system.")
    parser.add_argument("--datadir", default=".",
                        help="Directory containing CSV outputs")
    parser.add_argument("--outdir", default=None,
                        help="Output directory (default: datadir)")
    parser.add_argument("--interval", default=config.INTERVAL,
                        choices=list(BARS_PER_YEAR.keys()),
                        help="Bar interval used for both phases")
    parser.add_argument("--plots", default="all",
                        help=("Comma-separated subset of plot names, or "
                              f"'all'. Available: {', '.join(PLOT_REGISTRY)}"))
    parser.add_argument("--ticker", default=config.TICKER,
                        help="Ticker for signals + actions-on-price")
    parser.add_argument("--test_start", default=config.TEST_START)
    parser.add_argument("--test_end", default=config.TEST_END)
    parser.add_argument("--modeldir", default=".",
                        help="Root containing models/<agent>_agent/")
    parser.add_argument("--cache_dir", default=".cache",
                        help="Where to cache yfinance downloads")
    parser.add_argument("--no_cnn", action="store_true",
                        help="Skip CNN features in signals rollout")
    args = parser.parse_args()

    _apply_style()

    outdir = args.outdir or args.datadir
    os.makedirs(outdir, exist_ok=True)
    requested = _resolve_plots(args.plots)
    ann = BARS_PER_YEAR[args.interval]

    print(f"\n  Plotting → {outdir}/  (interval={args.interval}, "
          f"ann={ann})")
    print(f"  Requested: {', '.join(requested)}\n")

    p0_df = _load_p0(args.datadir)
    p1_eq = _load_p1_equity(args.datadir)
    p1_w  = _load_p1_weights(args.datadir)
    p1_a  = _load_p1_actions(args.datadir)
    p1_m  = _load_p1_metrics(args.datadir)
    r_df  = _load_regime_table(args.datadir)

    # Load prices once if we'll need them.
    needs_prices = any(p in requested for p in
                       ("signals", "p1_actions_on_price"))
    prices = None
    if needs_prices:
        prices = _load_prices(args.ticker, args.test_start, args.test_end,
                              args.interval, args.cache_dir)

    # CNN path (signals only — currently unused but keep wiring in place)
    cnn_path = None
    if not args.no_cnn:
        candidate = os.path.join(args.modeldir, "models", "cnn_features",
                                 "cnn_model.pt")
        if os.path.isfile(candidate):
            cnn_path = candidate

    # ── Dispatch table ────────────────────────────────────────────
    def _need(name, *required):
        if name not in requested:
            return False
        for r in required:
            if r is None:
                print(f"  [skip {name}] required input missing")
                return False
        return True

    if _need("p0_equity", p0_df):
        plot_p0_equity(p0_df, outdir, ann)
    if _need("p0_drawdown", p0_df):
        plot_p0_drawdown(p0_df, outdir)
    if _need("p0_rolling_sharpe", p0_df):
        plot_p0_rolling_sharpe(p0_df, outdir, ann)
    if _need("p0_return_dist", p0_df):
        plot_p0_return_dist(p0_df, outdir)
    if _need("p0_metrics_bar", p0_df):
        plot_p0_metrics_bar(p0_df, outdir, ann)

    if _need("p1_equity", p1_eq):
        plot_p1_equity(p1_eq, outdir)
    if _need("p1_drawdown", p1_eq):
        plot_p1_drawdown(p1_eq, outdir)
    if _need("p1_rolling_sharpe", p1_eq):
        plot_p1_rolling_sharpe(p1_eq, outdir, ann)
    if _need("p1_weight_evolution", p1_w):
        plot_p1_weight_evolution(p1_w, outdir)
    if _need("p1_actions_on_price", p1_a, p1_eq):
        plot_p1_actions_on_price(p1_a, p1_eq, outdir, prices=prices)
    if _need("p1_metrics_bar", p1_eq):
        plot_p1_metrics_bar(p1_eq, p1_m, outdir, ann)

    if _need("combined_equity", p0_df, p1_eq):
        plot_combined_equity(p0_df, p1_eq, outdir)
    if _need("combined_sharpe_bar", p0_df, p1_eq):
        plot_combined_sharpe_bar(p0_df, p1_eq, outdir, ann, ann)
    if _need("combined_metrics_table", p0_df, p1_eq):
        plot_combined_metrics_table(p0_df, p1_eq, outdir, ann, ann)

    if "signals" in requested:
        plot_signals(prices, args.modeldir, outdir, args.interval,
                     args.ticker, args.test_start, args.test_end,
                     cnn_model_path=cnn_path)

    if _need("regime_table", r_df):
        plot_regime_table(r_df, outdir)

    print(f"\n  ✓ Done. Files in {outdir}/\n")


if __name__ == "__main__":
    main()
