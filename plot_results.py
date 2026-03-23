import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── Style ──────────────────────────────────────────────────────────
PALETTE = {
    "conservative": "#1d7874",   # teal
    "neutral":      "#4a6fa5",   # blue
    "aggressive":   "#c75146",   # coral
    "market":       "#888888",   # grey (buy-and-hold)
}

plt.rcParams.update({
    "figure.facecolor":  "#0e1117",
    "axes.facecolor":    "#141820",
    "axes.edgecolor":    "#2a2f3d",
    "axes.labelcolor":   "#c8cdd8",
    "xtick.color":       "#7a7f8e",
    "ytick.color":       "#7a7f8e",
    "text.color":        "#c8cdd8",
    "grid.color":        "#1e2330",
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
})


def load_equity(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize to 100 at start
    return df / df.iloc[0] * 100


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (equity - peak) / peak * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvpath", default="equity_curves.csv")
    args = parser.parse_args()

    df = load_equity(args.csvpath)
    agents = [c for c in df.columns if c in PALETTE]

    fig = plt.figure(figsize=(14, 10), facecolor="#0e1117")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)

    # ── 1. Equity curves ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title("Equity curves — Phase 0 flat agents (indexed to 100)")
    for name in agents:
        ax1.plot(df[name].values, color=PALETTE[name], lw=1.5, label=name)
    ax1.axhline(100, color="#ffffff", lw=0.4, alpha=0.3, linestyle="--")
    ax1.set_ylabel("Portfolio value")
    ax1.legend(loc="upper left", framealpha=0.2, fontsize=8)
    ax1.grid(True)

    # ── 2. Drawdown curves ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_title("Drawdown (%) — conservative should be smallest")
    for name in agents:
        dd = drawdown_series(df[name])
        ax2.plot(dd.values, color=PALETTE[name], lw=1.2, label=name, alpha=0.85)
    ax2.fill_between(
        range(len(df)),
        drawdown_series(df[agents[0]]).values,
        0, color=PALETTE[agents[0]], alpha=0.06
    )
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(loc="lower left", framealpha=0.2, fontsize=8)
    ax2.grid(True)

    # ── 3. Rolling Sharpe (60-day) ────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_title("Rolling Sharpe (60-day window)")
    for name in agents:
        r = df[name].pct_change().dropna()
        rs = r.rolling(60).apply(
            lambda x: np.sqrt(252) * x.mean() / (x.std() + 1e-10)
        )
        ax3.plot(rs.values, color=PALETTE[name], lw=1.1, alpha=0.85, label=name)
    ax3.axhline(0, color="#ffffff", lw=0.5, alpha=0.25)
    ax3.set_ylabel("Sharpe")
    ax3.legend(framealpha=0.2, fontsize=8)
    ax3.grid(True)

    # ── 4. Return distribution ────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.set_title("Daily return distribution")
    for name in agents:
        r = df[name].pct_change().dropna().values * 100
        ax4.hist(
            r, bins=60, alpha=0.45,
            color=PALETTE[name], label=name,
            density=True, histtype="stepfilled"
        )
        ax4.axvline(np.mean(r), color=PALETTE[name], lw=1.2, linestyle="--", alpha=0.8)
    ax4.set_xlabel("Daily return (%)")
    ax4.set_ylabel("Density")
    ax4.legend(framealpha=0.2, fontsize=8)
    ax4.grid(True)

    fig.suptitle(
        "Phase 0 — Flat Agent Comparison\n"
        "Goal: each agent should show regime-specific strengths that motivate the HLA",
        fontsize=11, y=0.98, color="#e0e4ef"
    )

    plt.savefig("phase0_results.png", dpi=150, bbox_inches="tight", facecolor="#0e1117")
    print("Saved → phase0_results.png")
    plt.show()


if __name__ == "__main__":
    main()
