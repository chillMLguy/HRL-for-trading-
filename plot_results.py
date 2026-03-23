import argparse
import os
import warnings
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
 
# ── Style ──────────────────────────────────────────────────────────
PALETTE = {
    "conservative": "#1d7874",
    "neutral":      "#4a6fa5",
    "aggressive":   "#c75146",
    "market":       "#888888",
}
 
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
 
def load_equity(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df / df.iloc[0] * 100
 
 
def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (equity - peak) / peak * 100
 
 
def rollout_with_signals(model, prices, agent_type):
    from env.trading_env import TradingEnv
    env = TradingEnv(prices, agent_type=agent_type)
    obs, _ = env.reset()
    positions = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        positions.append(info["position"])
    return np.array(positions, dtype=np.float32)
 
 
def detect_trades(positions: np.ndarray, threshold: float = 0.1):
    buys, sells, flips = [], [], []
    for i in range(1, len(positions)):
        prev, curr = positions[i - 1], positions[i]
        sign_change = (prev > 0.1 and curr < -0.1) or (prev < -0.1 and curr > 0.1)
        if sign_change:
            flips.append(i)
        elif prev < threshold and curr >= threshold:
            buys.append(i)
        elif prev > -threshold and curr <= -threshold:
            sells.append(i)
    return buys, sells, flips
 
 
def plot_trade_signals(ax, prices_plot, positions, agent_name, color):
    n = min(len(prices_plot), len(positions))
    px = prices_plot[:n]
    pos = positions[:n]
    xs = np.arange(n)
 
    # Price line
    ax.plot(xs, px, color="#c8cdd8", lw=0.9, alpha=0.7, zorder=2)
 
    # Background shading — intensity proportional to position size
    for i in range(n - 1):
        p = pos[i]
        if p > 0.05:
            ax.axvspan(i, i + 1, alpha=min(p * 0.25, 0.25),
                       color="#4a6fa5", linewidth=0)
        elif p < -0.05:
            ax.axvspan(i, i + 1, alpha=min(-p * 0.25, 0.25),
                       color="#c75146", linewidth=0)
 
    # Trade markers
    buys, sells, flips = detect_trades(pos)
    if buys:
        bx = np.array(buys)
        ax.scatter(bx, px[bx], marker="^", color=BUY_COLOR,
                   s=45, zorder=5, alpha=0.95, linewidths=0)
    if sells:
        sx = np.array(sells)
        ax.scatter(sx, px[sx], marker="v", color=SELL_COLOR,
                   s=45, zorder=5, alpha=0.95, linewidths=0)
    if flips:
        fx = np.array(flips)
        ax.scatter(fx, px[fx], marker="D", color="#f39c12",
                   s=28, zorder=5, alpha=0.85, linewidths=0)
 
    # Secondary axis: position over time
    ax2 = ax.twinx()
    ax2.plot(xs, pos, color=color, lw=0.9, alpha=0.55, linestyle="--")
    ax2.axhline(0, color="#ffffff", lw=0.3, alpha=0.2)
    ax2.set_ylim(-2.0, 2.0)
    ax2.set_ylabel("position", fontsize=7, color=color)
    ax2.tick_params(labelsize=7, colors=color)
    ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
 
    legend_elements = [
        mpatches.Patch(color="#4a6fa5", alpha=0.5, label="long"),
        mpatches.Patch(color="#c75146", alpha=0.5, label="short"),
        plt.Line2D([0], [0], marker="^", color="w",
                   markerfacecolor=BUY_COLOR, markersize=7,
                   label="go long", linestyle="None"),
        plt.Line2D([0], [0], marker="v", color="w",
                   markerfacecolor=SELL_COLOR, markersize=7,
                   label="go short", linestyle="None"),
        plt.Line2D([0], [0], marker="D", color="w",
                   markerfacecolor="#f39c12", markersize=6,
                   label="flip", linestyle="None"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              framealpha=0.2, fontsize=7, ncol=3)
    ax.set_ylabel("price")
    ax.set_title(f"Trade signals — {agent_name}   "
                 f"({len(buys)} buys · {len(sells)} sells · {len(flips)} flips)")
    ax.grid(True, alpha=0.4)
 
 
# ── Main ───────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvpath",  default="equity_curves.csv")
    parser.add_argument("--ticker",   default=None,
                        help="Ticker for trade signal plot, e.g. SPY")
    parser.add_argument("--start",    default="2023-01-01")
    parser.add_argument("--end",      default="2024-12-31")
    parser.add_argument("--modeldir", default=".")
    args = parser.parse_args()
 
    df = load_equity(args.csvpath)
    agents = [c for c in df.columns if c in PALETTE]
 
    # ── Figure 1: performance overview ────────────────────────────
    fig1 = plt.figure(figsize=(14, 10), facecolor="#0e1117")
    gs1 = gridspec.GridSpec(3, 2, figure=fig1, hspace=0.45, wspace=0.32)
 
    ax1 = fig1.add_subplot(gs1[0, :])
    ax1.set_title("Equity curves (indexed to 100)")
    for name in agents:
        ax1.plot(df[name].values, color=PALETTE[name], lw=1.5, label=name)
    ax1.axhline(100, color="#ffffff", lw=0.4, alpha=0.3, linestyle="--")
    ax1.set_ylabel("Portfolio value")
    ax1.legend(loc="upper left", framealpha=0.2, fontsize=8)
    ax1.grid(True)
 
    ax2 = fig1.add_subplot(gs1[1, :])
    ax2.set_title("Drawdown (%)")
    for name in agents:
        dd = drawdown_series(df[name])
        ax2.plot(dd.values, color=PALETTE[name], lw=1.2, label=name, alpha=0.85)
    ax2.fill_between(range(len(df)), drawdown_series(df[agents[0]]).values,
                     0, color=PALETTE[agents[0]], alpha=0.06)
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(loc="lower left", framealpha=0.2, fontsize=8)
    ax2.grid(True)
 
    ax3 = fig1.add_subplot(gs1[2, 0])
    ax3.set_title("Rolling Sharpe (60-day)")
    for name in agents:
        r = df[name].pct_change().dropna()
        rs = r.rolling(60).apply(
            lambda x: np.sqrt(252) * x.mean() / (x.std() + 1e-10))
        ax3.plot(rs.values, color=PALETTE[name], lw=1.1, alpha=0.85, label=name)
    ax3.axhline(0, color="#ffffff", lw=0.5, alpha=0.25)
    ax3.set_ylabel("Sharpe")
    ax3.legend(framealpha=0.2, fontsize=8)
    ax3.grid(True)
 
    ax4 = fig1.add_subplot(gs1[2, 1])
    ax4.set_title("Daily return distribution")
    for name in agents:
        r = df[name].pct_change().dropna().values * 100
        ax4.hist(r, bins=60, alpha=0.45, color=PALETTE[name],
                 label=name, density=True, histtype="stepfilled")
        ax4.axvline(np.mean(r), color=PALETTE[name], lw=1.2,
                    linestyle="--", alpha=0.8)
    ax4.set_xlabel("Daily return (%)")
    ax4.set_ylabel("Density")
    ax4.legend(framealpha=0.2, fontsize=8)
    ax4.grid(True)
 
    fig1.suptitle("Phase 0 — Flat Agent Comparison",
                  fontsize=11, y=0.98, color="#e0e4ef")
    plt.savefig("phase0_results.png", dpi=150,
                bbox_inches="tight", facecolor="#0e1117")
    print("Saved → phase0_results.png")
 
    # ── Figure 2: trade signals ────────────────────────────────────
    if args.ticker:
        try:
            import yfinance as yf
            from stable_baselines3 import SAC
            from env.trading_env import AgentType
        except ImportError as e:
            print(f"  [WARN] Cannot plot trade signals: {e}")
            plt.show()
            return
 
        print(f"\n  Downloading {args.ticker} for signal plot...")
        raw = yf.download(args.ticker, start=args.start,
                          end=args.end, auto_adjust=True, progress=False)
        prices = raw["Close"].squeeze().dropna()
        prices_plot = prices.values[20:]   # skip warmup bars
 
        fig2, axes = plt.subplots(
            len(agents), 1,
            figsize=(16, 5 * len(agents)),
            facecolor="#0e1117",
        )
        if len(agents) == 1:
            axes = [axes]
 
        for ax, name in zip(axes, agents):
            model_path = os.path.join(
                args.modeldir, "models", f"{name}_agent", "best_model")
            if not os.path.exists(model_path + ".zip"):
                ax.set_title(f"{name} — model not found at {model_path}")
                continue
 
            agent_type = AgentType(name)
            model = SAC.load(model_path)
            print(f"  Rolling out {name}...")
            positions = rollout_with_signals(model, prices, agent_type)
            plot_trade_signals(ax, prices_plot, positions,
                               agent_name=name, color=PALETTE[name])
 
        fig2.suptitle(
            f"{args.ticker}  {args.start} → {args.end}   |   "
            "blue = long · red = short · ▲ go long · ▼ go short · ◆ flip",
            fontsize=9, y=1.005, color="#e0e4ef"
        )
        fig2.tight_layout()
        plt.savefig("phase0_signals.png", dpi=150,
                    bbox_inches="tight", facecolor="#0e1117")
        print("Saved → phase0_signals.png")
 
    plt.show()
 
 
if __name__ == "__main__":
    main()
 