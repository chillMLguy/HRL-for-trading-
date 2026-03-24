import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.cm as cm

# ── Auto colour palette — works for any number of agents ──────────

_FIXED = {
    "conservative": "#1d7874",
    "neutral":      "#4a6fa5",
    "aggressive":   "#c75146",
    "cvar":         "#9b59b6",
    "omega":        "#e67e22",
    "rachev":       "#2ecc71",
    "buy_&_hold":   "#888888",
}
_TAB10 = plt.get_cmap("tab10").colors

def agent_color(name, idx):
    return _FIXED.get(name, _TAB10[idx % len(_TAB10)])

BUY_COLOR  = "#2ecc71"
SELL_COLOR = "#e74c3c"

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


# ── Helpers ────────────────────────────────────────────────────────

def load_equity(path):
    df = pd.read_csv(path)
    # Normalize all columns to 100 at start
    return df / df.iloc[0] * 100


def drawdown_series(s):
    peak = s.cummax()
    return (s - peak) / peak * 100


def rollout_positions(model, prices, agent_type, bars_per_year=1638):
    from env.trading_env import TradingEnv
    env = TradingEnv(prices, agent_type=agent_type, bars_per_year=bars_per_year)
    obs, _ = env.reset()
    warmup = env.warmup
    positions = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        positions.append(info["position"])
    return np.array(positions, dtype=np.float32), warmup


def detect_trades(pos, threshold=0.1):
    buys, sells, flips = [], [], []
    for i in range(1, len(pos)):
        p, c = pos[i-1], pos[i]
        if (p > 0.1 and c < -0.1) or (p < -0.1 and c > 0.1):
            flips.append(i)
        elif p < threshold and c >= threshold:
            buys.append(i)
        elif p > -threshold and c <= -threshold:
            sells.append(i)
    return buys, sells, flips


def plot_signals_panel(ax, px, pos, name, color):
    n  = min(len(px), len(pos))
    px, pos, xs = px[:n], pos[:n], np.arange(n)

    ax.plot(xs, px, color="#c8cdd8", lw=0.9, alpha=0.7, zorder=2)

    for i in range(n - 1):
        p = pos[i]
        if   p >  0.05: ax.axvspan(i, i+1, alpha=min( p*0.25, 0.25), color="#4a6fa5", lw=0)
        elif p < -0.05: ax.axvspan(i, i+1, alpha=min(-p*0.25, 0.25), color="#c75146", lw=0)

    buys, sells, flips = detect_trades(pos)
    if buys:  ax.scatter(buys,  px[buys],  marker="^", color=BUY_COLOR,  s=45, zorder=5, lw=0)
    if sells: ax.scatter(sells, px[sells], marker="v", color=SELL_COLOR, s=45, zorder=5, lw=0)
    if flips: ax.scatter(flips, px[flips], marker="D", color="#f39c12",  s=28, zorder=5, lw=0)

    ax2 = ax.twinx()
    ax2.plot(xs, pos, color=color, lw=0.9, alpha=0.5, ls="--")
    ax2.axhline(0, color="#fff", lw=0.3, alpha=0.2)
    ax2.set_ylim(-2, 2)
    ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax2.set_ylabel("position", fontsize=7, color=color)
    ax2.tick_params(labelsize=7, colors=color)

    legend = [
        mpatches.Patch(color="#4a6fa5", alpha=0.5, label="long"),
        mpatches.Patch(color="#c75146", alpha=0.5, label="short"),
        plt.Line2D([0],[0], marker="^", color="w", markerfacecolor=BUY_COLOR,  ms=7, ls="None", label="go long"),
        plt.Line2D([0],[0], marker="v", color="w", markerfacecolor=SELL_COLOR, ms=7, ls="None", label="go short"),
        plt.Line2D([0],[0], marker="D", color="w", markerfacecolor="#f39c12",  ms=6, ls="None", label="flip"),
    ]
    ax.legend(handles=legend, loc="upper left", framealpha=0.2, fontsize=7, ncol=3)
    ax.set_ylabel("price")
    ax.set_title(f"{name}  —  {len(buys)} buys · {len(sells)} sells · {len(flips)} flips")
    ax.grid(True, alpha=0.4)


# ── Main ───────────────────────────────────────────────────────────

def main():
    BARS_PER_YEAR = {"1d": 252, "1h": 1638, "30m": 3276, "15m": 6552}

    parser = argparse.ArgumentParser()
    parser.add_argument("--csvpath",  default="equity_curves.csv")
    parser.add_argument("--ticker",   default=None)
    parser.add_argument("--start",    default="2025-11-01")
    parser.add_argument("--end",      default="2026-03-15")
    parser.add_argument("--interval", default="1h",
                        choices=list(BARS_PER_YEAR.keys()))
    parser.add_argument("--modeldir", default=".")
    args = parser.parse_args()

    bars_per_year = BARS_PER_YEAR[args.interval]

    df     = load_equity(args.csvpath)
    agents = list(df.columns)
    colors = {n: agent_color(n, i) for i, n in enumerate(agents)}

    print(f"  Plotting {len(agents)} agents: {agents}")

    # ── Figure 1: performance overview ────────────────────────────
    fig1 = plt.figure(figsize=(14, 10), facecolor="#0e1117")
    gs   = gridspec.GridSpec(3, 2, figure=fig1, hspace=0.45, wspace=0.32)

    # Equity curves
    ax1 = fig1.add_subplot(gs[0, :])
    ax1.set_title("Equity curves (indexed to 100)")
    for name in agents:
        lw = 2.0 if name == "buy_&_hold" else 1.4
        ls = "--" if name == "buy_&_hold" else "-"
        ax1.plot(df[name].values, color=colors[name], lw=lw, ls=ls, label=name)
    ax1.axhline(100, color="#ffffff", lw=0.4, alpha=0.3, ls="--")
    ax1.set_ylabel("Portfolio value")
    ax1.legend(loc="upper left", framealpha=0.2, fontsize=8)
    ax1.grid(True)

    # Drawdown
    ax2 = fig1.add_subplot(gs[1, :])
    ax2.set_title("Drawdown (%)")
    for name in agents:
        dd = drawdown_series(df[name])
        ax2.plot(dd.values, color=colors[name], lw=1.1, label=name, alpha=0.85)
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(loc="lower left", framealpha=0.2, fontsize=8)
    ax2.grid(True)

    # Rolling Sharpe — window and annualisation scaled to bar frequency
    roll_window = max(10, bars_per_year // 13)   # ~20 trading days
    ax3 = fig1.add_subplot(gs[2, 0])
    ax3.set_title(f"Rolling Sharpe ({roll_window}-bar)")
    for name in agents:
        r  = df[name].pct_change().dropna()
        rs = r.rolling(roll_window).apply(
            lambda x: np.sqrt(bars_per_year) * x.mean() / (x.std() + 1e-10))
        ax3.plot(rs.values, color=colors[name], lw=1.1, alpha=0.85, label=name)
    ax3.axhline(0, color="#ffffff", lw=0.5, alpha=0.25)
    ax3.set_ylabel("Sharpe")
    ax3.legend(framealpha=0.2, fontsize=8)
    ax3.grid(True)

    # Return distribution
    ax4 = fig1.add_subplot(gs[2, 1])
    ax4.set_title("Bar return distribution")
    for name in agents:
        r = df[name].pct_change().dropna().values * 100
        ax4.hist(r, bins=60, alpha=0.4, color=colors[name],
                 label=name, density=True, histtype="stepfilled")
        ax4.axvline(np.mean(r), color=colors[name], lw=1.2, ls="--", alpha=0.8)
    ax4.set_xlabel("Bar return (%)")
    ax4.set_ylabel("Density")
    ax4.legend(framealpha=0.2, fontsize=8)
    ax4.grid(True)

    fig1.suptitle("Phase 0 — Agent Comparison", fontsize=11, y=0.98, color="#e0e4ef")
    plt.savefig("phase0_results.png", dpi=150, bbox_inches="tight", facecolor="#0e1117")
    print("Saved → phase0_results.png")

    # ── Figure 2: trade signals (optional) ────────────────────────
    if args.ticker:
        try:
            import yfinance as yf
            from stable_baselines3 import SAC
            from env.trading_env import AgentType
        except ImportError as e:
            print(f"  [WARN] {e}")
            plt.show()
            return

        print(f"\n  Downloading {args.ticker} for signal plot...")
        raw    = yf.download(args.ticker, start=args.start, end=args.end,
                             interval=args.interval,
                             auto_adjust=True, progress=False)
        prices = raw["Close"].squeeze().dropna()

        # Only plot agents that have a model saved (skip buy_&_hold)
        signal_agents = []
        for name in agents:
            if name == "buy_&_hold":
                continue
            path = os.path.join(args.modeldir, "models",
                                f"{name}_agent", "best_model.zip")
            if os.path.exists(path):
                signal_agents.append(name)
            else:
                print(f"  [skip signals] {name} — no model found")

        if not signal_agents:
            print("  No models found for signal plot.")
            plt.show()
            return

        fig2, axes = plt.subplots(
            len(signal_agents), 1,
            figsize=(16, 5 * len(signal_agents)),
            facecolor="#0e1117",
        )
        if len(signal_agents) == 1:
            axes = [axes]

        for ax, name in zip(axes, signal_agents):
            try:
                at    = AgentType(name)
                model = SAC.load(os.path.join(
                    args.modeldir, "models", f"{name}_agent", "best_model"))
                print(f"  Rolling out {name}...")
                pos, warmup = rollout_positions(model, prices, at, bars_per_year)
                px_plot = prices.values[warmup:]
                plot_signals_panel(ax, px_plot, pos, name, colors[name])
            except Exception as e:
                ax.set_title(f"{name} — error: {e}")
                print(f"  [ERR] {name}: {e}")

        fig2.suptitle(
            f"{args.ticker}  {args.start} → {args.end}  |  "
            "blue=long · red=short · ▲buy · ▼sell · ◆flip",
            fontsize=9, y=1.005, color="#e0e4ef"
        )
        fig2.tight_layout()
        plt.savefig("phase0_signals.png", dpi=150,
                    bbox_inches="tight", facecolor="#0e1117")
        print("Saved → phase0_signals.png")

    plt.show()


if __name__ == "__main__":
    main()
