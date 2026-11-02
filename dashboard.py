"""
Results dashboard: one command, one self-contained interactive HTML file.

    python dashboard.py                        # every run found under runs/
    python dashboard.py --runs runs/split2024  # only these run folders
    python dashboard.py --out my.html --window 120

The output opens in any browser with no server and no internet connection.
A selector switches between runs, and every chart has a camera button that
saves a high-resolution PNG (for LaTeX, switch the page to the light theme
first).

Sections, per run:

  1. Overview       headline numbers and all runs side by side
  2. Single agents  the frozen Phase 0 agents traded alone, against benchmarks
  3. Model testing  Meta-RL (Phase 2) vs. rule-based allocators (Phase 1) vs.
                    benchmarks: equity, drawdown, metrics, rolling Sharpe,
                    return distribution, correlation, weights, regimes, ...
  4. Walk-forward   pooled statistics, per-fold Sharpe, seed dispersion
                    (walk-forward experiments only)
  5. Run details    training configuration, validation vs. test Sharpe

Only the CSV and JSON files written by evaluate_phase2.py are read, so no
model files are needed and nothing is retrained.  A folder counts as a run
when it holds phase2_returns.csv; <dir>/fold*/seed*/ folders are pooled into
one walk-forward experiment, the same way aggregate_results.py pools them.

Single-agent results are computed from the positions saved in
phase2_actions.csv.  In Phases 1 and 2 each frozen agent trades its own shadow
book (env/shadow.py), so those positions are what the agent does when trading
alone, and  net_ret = pos_t * ret_t - |pos_t - pos_(t-1)| * cost.
"""
import argparse
import datetime as dt
import html
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from scipy import stats as sps

import config
from aggregate_results import aggregate
from common.stats import (bootstrap_sharpe_ci, bootstrap_sharpe_diff_pvalue,
                          compute_metrics, max_drawdown, probabilistic_sharpe)

ANN = config.BARS_PER_YEAR.get(config.INTERVAL, 252)
AGENTS = list(config.AGENT_PRESETS)
ALLOCATORS = ["meta_rl", "hmm", "vol_regime", "equal_weight"]
BENCHMARKS = ["vol_target_bh", "buy_and_hold"]
REGIMES = ["low_vol", "mid_vol", "high_vol"]

LABEL = {
    "meta_rl": "Meta-RL", "hmm": "HMM", "vol_regime": "Vol regime",
    "equal_weight": "Equal weight", "buy_and_hold": "Buy & hold",
    "vol_target_bh": "Vol-target B&H", "aggressive": "Aggressive",
    "balanced": "Balanced", "conservative": "Conservative", "cash": "Cash",
    "low_vol": "Low vol", "mid_vol": "Mid vol", "high_vol": "High vol",
}


def label(name):
    return LABEL.get(name, name)


# ── Palette ─────────────────────────────────────────────────────────────────
# Each entity keeps one hue on every chart.  Allocators and the agents-plus-cash
# stack are separately validated for colorblind separation in light and dark
# mode.  Benchmarks are neutrals with dash patterns, so style separates them
# as well as hue.  The page's dark theme swaps each key below for its value.
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, BASELINE, DIV_MID = "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
BLUE, RED = "#2a78d6", "#e34948"
COLOR = {
    "meta_rl": BLUE, "hmm": "#eb6834", "vol_regime": "#1baf7a",
    "equal_weight": "#eda100", "aggressive": RED, "balanced": "#6250d6",
    "conservative": "#008300", "cash": "#e87ba4", "buy_and_hold": INK,
    "vol_target_bh": INK2,
}
DARK_MAP = {
    SURFACE: "#1a1a19", INK: "#f0efec", INK2: "#c3c2b7", GRID: "#2c2c2a",
    BASELINE: "#383835", DIV_MID: "#383835", BLUE: "#3987e5",
    "#eb6834": "#d95926", "#1baf7a": "#199e70", "#eda100": "#c98500",
    RED: "#e66767", "#6250d6": "#9085e9", "#e87ba4": "#d55181",
}
DIVERGING = [[0.0, RED], [0.5, DIV_MID], [1.0, BLUE]]
DASH = {"buy_and_hold": "dash", "vol_target_bh": "dashdot"}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
# Draw benchmarks underneath and Meta-RL on top; legends read the other way.
Z_RANK = {"buy_and_hold": 0, "vol_target_bh": 1, "equal_weight": 2,
          "vol_regime": 3, "hmm": 4, "conservative": 5, "balanced": 6,
          "aggressive": 7, "meta_rl": 9}
LEGEND_RANK = {"meta_rl": 1, "hmm": 2, "vol_regime": 3, "equal_weight": 4,
               "aggressive": 5, "balanced": 6, "conservative": 7, "cash": 8,
               "vol_target_bh": 9, "buy_and_hold": 10}


# ═══════════════════════════════════════════════════════════════════════════
#  Loading
# ═══════════════════════════════════════════════════════════════════════════

def _read_json(*parts):
    path = os.path.join(*parts)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_cell(path):
    """Read one evaluated run folder (all outputs of evaluate_phase2.py)."""
    returns = (pd.read_csv(os.path.join(path, "phase2_returns.csv"),
                           parse_dates=["date"]).set_index("date").astype(float))
    dates = returns.index
    actions = pd.read_csv(os.path.join(path, "phase2_actions.csv"))
    weights = pd.read_csv(os.path.join(path, "phase2_weights.csv"))
    eval_cfg = _read_json(path, "phase2_eval_config.json")
    cost = float(eval_cfg.get("cost_pct", config.COST_PCT))

    # Single agents: positions are the same under every allocator (shadow books)
    first = actions[actions["allocator"] == actions["allocator"].iloc[0]]
    first = first.sort_values("step")
    agent_pos = pd.DataFrame(
        {a: first[f"{a}_action"].values for a in AGENTS
         if f"{a}_action" in first}, index=dates[first["step"].values],
        dtype=float)
    held_before = agent_pos.shift(1).fillna(0.0)
    agent_ret = (agent_pos.mul(returns["buy_and_hold"], axis=0)
                 - (agent_pos - held_before).abs() * cost)

    exec_pos = actions.pivot(index="step", columns="allocator",
                             values="blended_action").astype(float)
    exec_pos.index = dates[exec_pos.index]

    wmap = {}
    for name, sub in weights.groupby("allocator"):
        sub = sub.sort_values("step").drop(columns="allocator")
        idx = dates[sub["step"].values]
        wmap[name] = sub.drop(columns="step").set_index(idx).astype(float)

    return {
        "path": path, "returns": returns, "agent_ret": agent_ret,
        "agent_pos": agent_pos, "exec_pos": exec_pos, "weights": wmap,
        "cost": cost, "eval_cfg": eval_cfg,
        "meta_cfg": _read_json(path, "models", "phase2", "config.json")
        or eval_cfg.get("meta_cfg", {}),
        "agent_cfg": {a: _read_json(path, "models", f"{a}_agent", "config.json")
                      for a in AGENTS},
        "thresholds": _read_json(path, "models", "phase1", "vol_thresholds.json"),
    }


def _load_price_cache():
    if not os.path.isfile(config.DATA_CSV):
        return None
    df = pd.read_csv(config.DATA_CSV, parse_dates=["Date"], index_col="Date")
    return df["Close"].dropna() if "Close" in df else None


def vol_regimes(returns, thresholds, prices):
    """
    Low / mid / high-vol label for every traded day.  The label uses only what
    was known when the position was set: rolling realized vol up to the
    previous close, against the training-period terciles when the run saved
    them (the same rule as the Vol regime allocator), else in-window terciles.
    """
    window = max(20, ANN // 13)
    dates = returns.index
    rv = None
    if prices is not None:
        simple = prices.pct_change().reindex(dates)
        if (simple - returns["buy_and_hold"]).abs().max() < 1e-3:
            rv = np.log(prices / prices.shift(1)).rolling(window).std() \
                 * np.sqrt(ANN)
    if rv is None:      # the cache holds another series: use the traded window
        rv = np.log1p(returns["buy_and_hold"]).rolling(window).std() \
             * np.sqrt(ANN)
    rv = rv.shift(1).reindex(dates)
    q33 = thresholds.get("q33", rv.quantile(0.33))
    q67 = thresholds.get("q67", rv.quantile(0.67))
    out = pd.Series("mid_vol", index=dates, dtype=object)
    out[rv <= q33] = "low_vol"
    out[rv > q67] = "high_vol"
    out[rv.isna()] = "n/a"
    return out


def _pool(frames):
    return pd.concat(frames).groupby(level=0).mean().sort_index()


def build_run(run_id, cells, prices, wf_cells=None):
    """Merge one cell (single run) or many (walk-forward) into a run."""
    names = sorted({n for c in cells for n in c["weights"]})
    regimes = (pd.concat([vol_regimes(c["returns"], c["thresholds"], prices)
                          for c in cells]).groupby(level=0).first())
    returns = _pool([c["returns"] for c in cells])
    run = {
        "id": run_id, "cells": cells, "wf_cells": wf_cells,
        "returns": returns,
        "agent_ret": _pool([c["agent_ret"] for c in cells]),
        "agent_pos": _pool([c["agent_pos"] for c in cells]),
        "exec_pos": _pool([c["exec_pos"] for c in cells]),
        "weights": {n: _pool([c["weights"][n] for c in cells
                              if n in c["weights"]]) for n in names},
        "regimes": regimes.reindex(returns.index),
    }
    run["strategies"] = [s for s in ALLOCATORS + BENCHMARKS
                         if s in returns.columns]
    run["agents"] = [a for a in AGENTS if a in run["agent_ret"].columns]
    run["metrics"] = metrics_frame(run)
    return run


def discover(paths):
    """Return ([single-run folders], {walk-forward root: [cell folders]})."""
    singles, walkforward = [], {}
    for root in paths:
        for here, subdirs, files in os.walk(root):
            subdirs[:] = sorted(d for d in subdirs if d not in ("models", "logs"))
            if "phase2_returns.csv" not in files:
                continue
            m = re.search(r"^(.*)[/\\]fold\d+[/\\]seed\d+$", os.path.normpath(here))
            if m:
                walkforward.setdefault(m.group(1), []).append(here)
            else:
                singles.append(here)
    return singles, walkforward


def load_runs(paths):
    prices = _load_price_cache()
    singles, walkforward = discover(paths)
    runs = []
    for path in singles:
        runs.append(build_run(os.path.normpath(path), [load_cell(path)], prices))
    for root, cell_paths in walkforward.items():
        cells, wf_cells = [], {}
        for path in sorted(cell_paths):
            m = re.search(r"fold(\d+)[/\\]seed(\d+)", path)
            cell = load_cell(path)
            cells.append(cell)
            wf_cells[(int(m.group(1)), int(m.group(2)))] = \
                cell["returns"].reset_index()
        runs.append(build_run(root, cells, prices, wf_cells=wf_cells))

    def rank(run):
        folds = {k[0] for k in (run["wf_cells"] or {})}
        steps = run_steps(run)[0] or 0
        return (0 if len(folds) > 1 else 1 if not run["wf_cells"] else 2, -steps,
                run["id"])
    return sorted(runs, key=rank)


def run_steps(run):
    """(Phase 0 steps per agent, Phase 2 steps) of a run's first cell."""
    cell = run["cells"][0]
    p0 = [c.get("timesteps") for c in cell["agent_cfg"].values() if c]
    return (p0[0] if p0 else None), cell["meta_cfg"].get("timesteps")


# ═══════════════════════════════════════════════════════════════════════════
#  Statistics
# ═══════════════════════════════════════════════════════════════════════════

def equity(r):
    """Equity including the starting 1.0, so len(equity) == len(r) + 1."""
    return np.r_[1.0, np.cumprod(1.0 + np.asarray(r, dtype=np.float64))]


def drawdown_pct(r):
    eq = equity(r)
    peak = np.maximum.accumulate(eq)
    return ((eq - peak) / peak * 100.0)[1:]


def rolling_sharpe(s, window):
    return (s.rolling(window).mean() / s.rolling(window).std().replace(0, np.nan)
            * np.sqrt(ANN))


def worst_drawdown(r, dates):
    """Peak date, trough date and recovery date of the deepest drawdown."""
    eq = equity(r)
    peak = np.maximum.accumulate(eq)
    trough = int(np.argmax((peak - eq) / peak))
    if trough == 0:
        return "–", "–", "No drawdown"
    start = int(np.argmax(eq[:trough + 1]))
    later = np.nonzero(eq[trough:] >= eq[start])[0]
    day = lambda i: dates[i - 1].strftime("%Y-%m-%d") if i > 0 else "start"
    recovered = (f"Recovered {day(trough + int(later[0]))}" if len(later)
                 else "Not recovered by the end of the window")
    return day(start), day(trough), recovered


def all_returns(run):
    frames = [run["returns"][run["strategies"]]]
    if run["agents"]:
        frames.append(run["agent_ret"][run["agents"]])
    return pd.concat(frames, axis=1)


def metrics_frame(run):
    """One row per strategy and agent.  Base metrics match phase2_metrics.csv."""
    rets = all_returns(run)
    bh = rets["buy_and_hold"].values
    rows = {}
    for name in rets.columns:
        r = rets[name].values
        row = compute_metrics(r, equity(r), ANN)
        lo, hi = bootstrap_sharpe_ci(r, ANN)
        beta = float(np.cov(r, bh)[0, 1] / bh.var(ddof=1))
        row.update({
            "Ann. vol (%)": round(r.std() * np.sqrt(ANN) * 100, 2),
            "CI lo": round(lo, 3), "CI hi": round(hi, 3),
            "PSR": round(probabilistic_sharpe(r), 3),
            "Skew": round(float(sps.skew(r)), 2),
            "Ex. kurt": round(float(sps.kurtosis(r)), 2),
            "Beta": round(beta, 2),
            "Alpha (%)": round((r.mean() - beta * bh.mean()) * ANN * 100, 2),
        })
        pos = None
        if name in run["exec_pos"]:
            pos = run["exec_pos"][name].values
        elif name in run["agent_pos"]:
            pos = run["agent_pos"][name].values
        elif name == "buy_and_hold":
            pos = np.ones(len(r))
        if pos is not None:
            row["Avg |pos|"] = round(float(np.abs(pos).mean()), 3)
            row["Net pos"] = round(float(pos.mean()), 3)
            row["Turnover/yr"] = round(
                float(np.abs(np.diff(pos, prepend=0.0)).mean() * ANN), 1)
        rows[name] = row
    return pd.DataFrame(rows).T


def regime_sharpe(rets, regimes):
    days = {g: int((regimes == g).sum()) for g in REGIMES}
    rows = {}
    for name in rets.columns:
        rows[name] = {}
        for g in REGIMES:
            r = rets.loc[regimes == g, name].values
            s = r.std() if len(r) >= 5 else 0.0
            rows[name][g] = np.sqrt(ANN) * r.mean() / s if s > 1e-10 else np.nan
    return pd.DataFrame(rows).T[REGIMES], days


# ═══════════════════════════════════════════════════════════════════════════
#  Chart helpers
# ═══════════════════════════════════════════════════════════════════════════

def _vals(a, nd=4):
    return [None if not np.isfinite(v) else round(float(v), nd)
            for v in np.asarray(a, dtype=np.float64)]


def _days(index):
    return index.strftime("%Y-%m-%d").tolist()


def _style(fig, height=340, legend=True, hover="x unified", top=None, right=20,
           yfmt=None, ysuffix="", ytitle=None, date_x=True):
    fig.update_layout(
        template="none", height=height, paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE, hovermode=hover, showlegend=legend,
        font=dict(family=FONT, size=12, color=INK2),
        margin=dict(l=58, r=right, t=(36 if legend else 14) if top is None
                    else top, b=34),
        legend=dict(orientation="h", x=0, xanchor="left", y=1.0,
                    yanchor="bottom", font=dict(size=11, color=INK2),
                    bgcolor="rgba(0,0,0,0)", itemwidth=36),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=GRID,
                        font=dict(family=FONT, size=12, color=INK)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=BASELINE,
                     ticks="", tickfont=dict(size=11, color=MUTED))
    if date_x:
        fig.update_xaxes(type="date", showspikes=True, spikemode="across",
                         spikethickness=1, spikecolor=BASELINE, spikedash="solid",
                         spikesnap="data")
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, griddash="solid",
                     zerolinecolor=BASELINE, zerolinewidth=1, linecolor=BASELINE,
                     ticks="", tickfont=dict(size=11, color=MUTED),
                     tickformat=yfmt, ticksuffix=ysuffix)
    if ytitle:
        fig.update_yaxes(title=dict(text=ytitle, font=dict(size=11, color=INK2),
                                    standoff=6))
    return fig


def _line_trace(x, y, name, fmt=".3f", suffix="", nd=4, hidden=False,
                color=None, width=None):
    meta = name == "meta_rl"
    return go.Scatter(
        x=x, y=_vals(y, nd), mode="lines", name=label(name),
        legendrank=LEGEND_RANK.get(name, 50),
        visible="legendonly" if hidden else True,
        line=dict(color=color or COLOR.get(name, INK2),
                  width=width or (2.6 if meta else 1.6),
                  dash=DASH.get(name, "solid")),
        hovertemplate=f"<b>%{{y:{fmt}}}{suffix}</b> {label(name)}<extra></extra>",
    )


def _by_z(names):
    return sorted(names, key=lambda n: Z_RANK.get(n, 8))


def _end_labels(fig, x_end, ends):
    """Label line ends selectively: Meta-RL and benchmarks first, skip collisions."""
    values = [v for _, v in ends]
    gap = max((max(values) - min(values)) * 0.07, 1e-9)
    placed = []
    order = {"meta_rl": 0, "buy_and_hold": 1, "vol_target_bh": 2}
    for name, y in sorted(ends, key=lambda e: order.get(e[0], 5)):
        if any(abs(y - p) < gap for p in placed):
            continue
        placed.append(y)
        fig.add_annotation(x=x_end, y=y, text=label(name), xanchor="left",
                           xshift=6, showarrow=False, name="endlabel",
                           font=dict(size=11, color=INK2))


def _subtitle_style(fig, left=True):
    for a in fig.layout.annotations:
        a.font = dict(size=11, color=INK2)
        if left:
            a.x, a.xanchor = 0, "left"
    return fig


def chart(title, sub, fig, span="half"):
    data = json.loads(fig.to_json())
    data["layout"].pop("template", None)
    return {"type": "chart", "title": title, "sub": sub, "span": span,
            "height": fig.layout.height, "data": data["data"],
            "layout": data["layout"]}


def block_html(markup, span="full"):
    return {"type": "html", "html": markup, "span": span}


# ═══════════════════════════════════════════════════════════════════════════
#  Charts
# ═══════════════════════════════════════════════════════════════════════════

def fig_equity(rets, hidden=()):
    x = _days(rets.index)
    fig = go.Figure()
    ends = []
    for name in _by_z(rets.columns):
        eq = equity(rets[name].values)[1:]
        off = name in hidden
        fig.add_trace(_line_trace(x, eq, name, hidden=off))
        if not off:
            ends.append((name, float(eq[-1])))
    _end_labels(fig, x[-1], ends)
    return _style(fig, height=380, right=104, yfmt=".2f",
                  ytitle="Equity (start = 1.0)")


def fig_drawdown(rets):
    x = _days(rets.index)
    fig = go.Figure()
    for name in _by_z(rets.columns):
        fig.add_trace(_line_trace(x, drawdown_pct(rets[name].values), name,
                                  fmt=".2f", suffix="%", nd=3))
    return _style(fig, yfmt=".0f", ysuffix="%", ytitle="Drop from running peak")


def _ranked_bars(names, values, fmt, xtitle, custom=None, hover=None):
    order = np.argsort(values)[::-1]
    names = [names[i] for i in order]
    values = [values[i] for i in order]
    fig = go.Figure(go.Bar(
        orientation="h", y=[label(n) for n in names], x=_vals(values, 3),
        marker=dict(color=[COLOR.get(n, INK2) for n in names], cornerradius=4),
        text=[fmt.format(v) for v in values], textposition="outside",
        cliponaxis=False, textfont=dict(size=11, color=INK),
        customdata=[custom[i] for i in order] if custom else None,
        hovertemplate=hover or "<b>%{x:.2f}</b> %{y}<extra></extra>",
    ))
    fig.update_layout(bargap=0.42)
    _style(fig, height=64 + 34 * len(names), legend=False, hover="closest",
           date_x=False, right=64)
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)", type="category",
                     tickfont=dict(size=12, color=INK2))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=True,
                     zerolinecolor=BASELINE, title=dict(text=xtitle,
                     font=dict(size=11, color=INK2)), rangemode="tozero")
    fig.update_layout(margin=dict(l=108, b=48))
    return fig


def fig_max_drawdown(rets):
    names = list(rets.columns)
    values = [max_drawdown(equity(rets[n].values)) * 100 for n in names]
    custom = [list(worst_drawdown(rets[n].values, rets.index)) for n in names]
    hover = ("<b>%{x:.2f}%</b> %{y}<br>Peak %{customdata[0]}<br>"
             "Trough %{customdata[1]}<br>%{customdata[2]}<extra></extra>")
    return _ranked_bars(names, values, "{:.2f}%", "Maximum drawdown (%)",
                        custom, hover)


def fig_rolling_sharpe(rets, window):
    x = _days(rets.index)
    fig = go.Figure()
    for name in _by_z(rets.columns):
        fig.add_trace(_line_trace(x, rolling_sharpe(rets[name], window).values,
                                  name, fmt=".2f", nd=3))
    return _style(fig, yfmt=".1f", ytitle=f"Annualized Sharpe, {window}-day window")


def fig_rolling_vol(rets, window, target=None):
    x = _days(rets.index)
    fig = go.Figure()
    for name in _by_z(rets.columns):
        vol = rets[name].rolling(window).std() * np.sqrt(ANN) * 100
        fig.add_trace(_line_trace(x, vol.values, name, fmt=".1f", suffix="%", nd=3))
    if target:
        fig.add_hline(y=target * 100, line=dict(color=MUTED, width=1),
                      annotation_text=f"Target {target:.0%}",
                      annotation_font=dict(size=11, color=INK2),
                      annotation_position="top left")
    return _style(fig, yfmt=".0f", ysuffix="%",
                  ytitle=f"Annualized vol, {window}-day window")


def fig_distribution(rets):
    """Density lines stay readable when several distributions overlap."""
    pct = rets * 100.0
    lo = float(pct.quantile(0.002).min())
    hi = float(pct.quantile(0.998).max())
    pad = (hi - lo) * 0.05
    grid = np.linspace(lo - pad, hi + pad, 240)
    fig = go.Figure()
    for name in _by_z(pct.columns):
        v = pct[name].values
        if v.std() < 1e-6:
            continue
        fig.add_trace(_line_trace(grid.round(3).tolist(), sps.gaussian_kde(v)(grid),
                                  name, fmt=".2f", nd=3))
    fig.add_vline(x=0, line=dict(color=BASELINE, width=1))
    _style(fig, date_x=False, yfmt=".1f", ytitle="Density")
    fig.update_xaxes(ticksuffix="%", title=dict(text="Daily return",
                     font=dict(size=11, color=INK2)), showspikes=True,
                     spikemode="across", spikethickness=1, spikecolor=BASELINE,
                     spikedash="solid")
    fig.update_layout(margin=dict(b=48))
    return fig


def _heatmap(z, xlabels, ylabels, zmin, zmax, fmt, suffix="", height=None,
             hover=None):
    z = np.asarray(z, dtype=np.float64)
    text = [["" if not np.isfinite(v) else fmt.format(v) + suffix for v in row]
            for row in z]
    fig = go.Figure(go.Heatmap(
        z=[_vals(row, 3) for row in z], x=xlabels, y=ylabels, zmin=zmin, zmax=zmax,
        zmid=0, colorscale=DIVERGING, xgap=2, ygap=2, text=text,
        texttemplate="%{text}", textfont=dict(size=11), hoverongaps=False,
        hovertemplate=hover or "%{y} · %{x}<br><b>%{text}</b><extra></extra>",
        colorbar=dict(thickness=10, outlinewidth=0, len=0.9,
                      tickfont=dict(size=10, color=MUTED)),
    ))
    _style(fig, height=height or 60 + 38 * len(ylabels), legend=False,
           hover="closest", date_x=False, top=10)
    fig.update_xaxes(type="category", side="bottom", linecolor="rgba(0,0,0,0)")
    fig.update_yaxes(type="category", autorange="reversed",
                     gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)",
                     tickfont=dict(size=11, color=INK2))
    fig.update_layout(margin=dict(l=104))
    return fig


def fig_correlation(rets):
    corr = rets.corr().values.copy()
    corr[np.triu_indices_from(corr, k=1)] = np.nan
    names = [label(n) for n in rets.columns]
    fig = _heatmap(corr, [n.replace(" ", "<br>", 1) for n in names], names, -1, 1,
                   "{:.2f}", height=84 + 44 * len(names))
    fig.update_xaxes(tickangle=0)
    fig.update_layout(margin=dict(b=48))
    return fig


def fig_regime_sharpe(rets, regimes):
    table, days = regime_sharpe(rets, regimes)
    x = [f"{label(g)} ({days[g]}d)" for g in REGIMES]
    limit = max(1.0, min(4.0, float(np.nanmax(np.abs(table.values)))))
    return _heatmap(table.values, x, [label(n) for n in table.index],
                    -limit, limit, "{:.2f}")


def fig_periodic_returns(rets):
    """Returns per month, or per year once there are too many months to read."""
    months = rets.index.to_period("M").nunique()
    freq = "M" if months <= 24 else "Y"
    groups = rets.groupby(rets.index.to_period(freq))
    table = groups.apply(lambda g: (1.0 + g).prod() - 1.0) * 100.0
    xlabels = [p.strftime("%b %y") if freq == "M" else str(p) for p in table.index]
    limit = max(1.0, float(np.nanmax(np.abs(table.values))))
    return _heatmap(table.T.values, xlabels, [label(n) for n in table.columns],
                    -limit, limit, "{:.1f}", suffix="%")


def fig_sharpe_ci(metrics, order):
    """Point estimate with bootstrap CI, so noise is visible next to the gaps."""
    rows = metrics.loc[order].sort_values("Sharpe", ascending=False)
    fig = go.Figure()
    for name, row in rows.iterrows():
        color = COLOR.get(name, INK2)
        fig.add_trace(go.Scatter(
            x=[row["Sharpe"]], y=[label(name)], mode="markers", showlegend=False,
            marker=dict(size=11, color=color, line=dict(color=SURFACE, width=2)),
            error_x=dict(type="data", symmetric=False, width=0, thickness=1.6,
                         array=[row["CI hi"] - row["Sharpe"]],
                         arrayminus=[row["Sharpe"] - row["CI lo"]], color=color),
            hovertemplate=(f"<b>%{{x:.2f}}</b> {label(name)}<br>95% CI "
                           f"{row['CI lo']:.2f} to {row['CI hi']:.2f}<extra></extra>"),
        ))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    _style(fig, height=64 + 34 * len(rows), legend=False, hover="closest",
           date_x=False)
    fig.update_yaxes(autorange="reversed", type="category",
                     gridcolor="rgba(0,0,0,0)", tickfont=dict(size=12, color=INK2))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, title=dict(
        text="Annualized Sharpe", font=dict(size=11, color=INK2)))
    fig.update_layout(margin=dict(l=108, b=48))
    return fig


def fig_exposure_ladder(run):
    names = run["agents"]
    mean_abs = [float(run["agent_pos"][a].abs().mean()) for a in names]
    caps = [config.AGENT_PRESETS[a]["max_pos"] for a in names]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        orientation="h", y=[label(a) for a in names], x=_vals(mean_abs, 3),
        marker=dict(color=[COLOR[a] for a in names], cornerradius=4),
        text=[f"{v:.2f}" for v in mean_abs], textposition="outside",
        cliponaxis=False, textfont=dict(size=11, color=INK),
        hovertemplate="<b>%{x:.3f}</b> mean |position| %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        y=[label(a) for a in names], x=caps, mode="markers",
        marker=dict(symbol="line-ns", size=22, line=dict(color=INK, width=2)),
        hovertemplate="Hard cap <b>%{x:.2f}</b> %{y}<extra></extra>",
    ))
    fig.update_layout(bargap=0.45)
    _style(fig, height=64 + 44 * len(names), legend=False, hover="closest",
           date_x=False)
    fig.update_yaxes(autorange="reversed", type="category",
                     gridcolor="rgba(0,0,0,0)", tickfont=dict(size=12, color=INK2))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, range=[0, 1.12],
                     zeroline=True, zerolinecolor=BASELINE, title=dict(
                     text="Position size (1.0 = fully invested)",
                     font=dict(size=11, color=INK2)))
    fig.update_layout(margin=dict(l=96, b=48))
    return fig


def fig_positions_on_price(run, frame, names, panel_height=112, hidden_area=False):
    """Index level on top; each strategy's position below on a shared date axis."""
    x = _days(frame.index)
    rows = len(names) + 1
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.035,
        row_heights=[1.5] + [1] * len(names),
        subplot_titles=["Index level (start = 100)"]
                       + [f"{label(n)} position" for n in names])
    level = equity(run["returns"]["buy_and_hold"].reindex(frame.index).values)[1:] * 100
    fig.add_trace(_line_trace(x, level, "buy_and_hold", fmt=".1f", nd=2,
                              color=INK, width=1.6), row=1, col=1)
    fig.data[-1].line.dash = "solid"
    fig.data[-1].hovertemplate = "<b>%{y:.1f}</b> Index<extra></extra>"
    limit = max(1.0, float(frame[names].abs().max().max())) * 1.05
    for i, name in enumerate(names, start=2):
        trace = _line_trace(x, frame[name].values, name, fmt=".2f", nd=3, width=1.4)
        trace.fill = "tozeroy"
        trace.fillcolor = _wash(COLOR.get(name, INK2))
        trace.line.dash = "solid"
        fig.add_trace(trace, row=i, col=1)
        fig.update_yaxes(range=[-limit, limit], row=i, col=1, nticks=5)
    _style(fig, height=70 + panel_height * (rows + 0.5), legend=False, yfmt=None)
    fig.update_yaxes(tickformat=".1f")
    fig.update_yaxes(tickformat=".0f", row=1, col=1)
    fig.update_layout(margin=dict(t=24))
    return _subtitle_style(fig)


def _wash(hex_color, alpha=0.12):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def fig_weight_stacks(run):
    names = [a for a in ALLOCATORS if a in run["weights"]]
    fig = make_subplots(rows=len(names), cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=[label(n) for n in names])
    for i, name in enumerate(names, start=1):
        w = run["weights"][name]
        x = _days(w.index)
        parts = [(a, f"w_{a}") for a in AGENTS]
        if "w_cash" in w and w["w_cash"].fillna(0).abs().max() > 0:
            parts.append(("cash", "w_cash"))
        for part, col in parts:
            fig.add_trace(go.Scatter(
                x=x, y=_vals(w[col].fillna(0.0), 3), mode="lines", name=label(part),
                stackgroup=f"s{i}", legendgroup=part, showlegend=i == 1,
                legendrank=LEGEND_RANK[part], fillcolor=COLOR[part],
                line=dict(color=SURFACE, width=0.8),
                hovertemplate=f"<b>%{{y:.0%}}</b> {label(part)}<extra></extra>",
            ), row=i, col=1)
    _style(fig, height=70 + 132 * len(names), yfmt=".0%")
    fig.update_yaxes(range=[0, 1], nticks=3, showgrid=False)
    fig.update_layout(margin=dict(t=52), legend=dict(
        yref="container", y=1.0, yanchor="top"))
    return _subtitle_style(fig)


def fig_weights_by_regime(run):
    names = [a for a in ("meta_rl", "hmm", "vol_regime") if a in run["weights"]]
    fig = make_subplots(rows=1, cols=len(names), shared_yaxes=True,
                        horizontal_spacing=0.04,
                        subplot_titles=[label(n) for n in names])
    regimes = run["regimes"]
    counts = {g: int((regimes == g).sum()) for g in REGIMES}
    x = [f"{label(g)}<br>{counts[g]}d" for g in REGIMES]
    for j, name in enumerate(names, start=1):
        w = run["weights"][name].reindex(regimes.index)
        parts = [(a, f"w_{a}") for a in AGENTS]
        if "w_cash" in w and w["w_cash"].fillna(0).abs().max() > 0:
            parts.append(("cash", "w_cash"))
        for part, col in parts:
            means = [float(w.loc[regimes == g, col].fillna(0).mean())
                     if counts[g] else 0.0 for g in REGIMES]
            fig.add_trace(go.Bar(
                x=x, y=_vals(means, 3), name=label(part), legendgroup=part,
                showlegend=j == 1, legendrank=LEGEND_RANK[part],
                marker=dict(color=COLOR[part], line=dict(color=SURFACE, width=2)),
                text=[f"{m:.0%}" if m >= 0.1 else "" for m in means],
                textposition="inside", insidetextanchor="middle",
                textfont=dict(size=11, color="#ffffff"), textangle=0,
                hovertemplate=f"<b>%{{y:.1%}}</b> {label(part)}, %{{x}}<extra></extra>",
            ), row=1, col=j)
    fig.update_layout(barmode="stack", bargap=0.3)
    _style(fig, height=380, hover="closest", date_x=False, yfmt=".0%")
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(margin=dict(t=60, b=48), legend=dict(
        yref="container", y=1.0, yanchor="top"))
    return _subtitle_style(fig, left=False)


def fig_excess(rets):
    """Meta-RL minus each alternative: above zero means the RL layer is ahead."""
    x = _days(rets.index)
    meta = equity(rets["meta_rl"].values)[1:]
    fig = go.Figure()
    for name in _by_z([c for c in rets.columns if c != "meta_rl"]):
        other = equity(rets[name].values)[1:]
        fig.add_trace(_line_trace(x, (meta / other - 1.0) * 100.0, name,
                                  fmt="+.2f", suffix="%", nd=3))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    return _style(fig, yfmt=".0f", ysuffix="%", ytitle="Meta-RL relative to each line")


def fig_meta_diagnostics(run):
    w = run["weights"]["meta_rl"]
    x = _days(w.index)
    cols = [f"w_{a}" for a in AGENTS]
    has_cash = "w_cash" in w and w["w_cash"].fillna(0).abs().max() > 1e-6
    has_lever = "lever" in w and (w["lever"].fillna(1) - 1).abs().max() > 1e-6
    W = w[cols + (["w_cash"] if has_cash else [])].fillna(0.0).values
    p = np.clip(W, 1e-12, 1.0)
    entropy = -(p * np.log(p)).sum(axis=1) / np.log(W.shape[1])
    turnover = np.r_[0.0, np.abs(np.diff(W, axis=0)).sum(axis=1)]

    panels = []
    if has_cash:
        panels.append(("Cash weight", w["w_cash"].values, COLOR["cash"], ".0%"))
    if has_lever:
        panels.append(("Vol-target leverage", w["lever"].values, BLUE, ".2f"))
    panels.append(("Allocation entropy (1 = evenly mixed, 0 = one agent)",
                   entropy, BLUE, ".2f"))
    panels.append(("Daily weight turnover, sum of |change|", turnover, BLUE, ".2f"))
    titles = [f"{t}, mean {np.nanmean(v):{f}}" for t, v, _, f in panels]
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True,
                        vertical_spacing=0.07, subplot_titles=titles)
    for i, (t, v, c, f) in enumerate(panels, start=1):
        fig.add_trace(go.Scatter(
            x=x, y=_vals(v, 3), mode="lines", line=dict(color=c, width=1.5),
            hovertemplate=f"<b>%{{y:{f}}}</b> {t.split(',')[0]}<extra></extra>",
        ), row=i, col=1)
    _style(fig, height=50 + 118 * len(panels), legend=False)
    fig.update_yaxes(nticks=4)
    fig.update_layout(margin=dict(t=26))
    return _subtitle_style(fig)


def fig_proposals(run, height=340):
    """What the meta-agent executed against what its agents proposed."""
    frame = run["agent_pos"]
    x = _days(frame.index)
    fig = go.Figure()
    for a in run["agents"]:
        fig.add_trace(_line_trace(x, frame[a].values, a, fmt=".2f", nd=3, width=1.1))
    fig.add_trace(go.Scatter(
        x=x, y=_vals(run["exec_pos"]["meta_rl"].values, 3), mode="lines",
        name="Executed (Meta-RL)", legendrank=0, line=dict(color=INK, width=2.2),
        hovertemplate="<b>%{y:.2f}</b> Executed<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    return _style(fig, height=height, yfmt=".1f", ytitle="Position (1.0 = fully long)")


def fig_per_fold(per_fold):
    names = [n for n in ALLOCATORS + BENCHMARKS if n in per_fold.index]
    table = per_fold.loc[names]
    limit = max(1.0, min(4.0, float(np.nanmax(np.abs(table.values)))))
    return _heatmap(table.values, [str(c) for c in table.columns],
                    [label(n) for n in names], -limit, limit, "{:.2f}")


def fig_seed_dispersion(per_seed):
    names = [n for n in ALLOCATORS + BENCHMARKS if n in per_seed]
    fig = go.Figure()
    for name in names:
        seeds = sorted(per_seed[name])
        fig.add_trace(go.Scatter(
            x=_vals([per_seed[name][s] for s in seeds], 3),
            y=[label(name)] * len(seeds), mode="markers", showlegend=False,
            customdata=seeds,
            marker=dict(size=11, color=COLOR.get(name, INK2),
                        line=dict(color=SURFACE, width=2)),
            hovertemplate=(f"<b>%{{x:.2f}}</b> {label(name)}, "
                           "seed %{customdata}<extra></extra>"),
        ))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    _style(fig, height=64 + 34 * len(names), legend=False, hover="closest",
           date_x=False)
    fig.update_yaxes(autorange="reversed", type="category",
                     gridcolor="rgba(0,0,0,0)", tickfont=dict(size=12, color=INK2))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, title=dict(
        text="Pooled annualized Sharpe of each seed", font=dict(size=11, color=INK2)))
    fig.update_layout(margin=dict(l=108, b=48))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  HTML blocks
# ═══════════════════════════════════════════════════════════════════════════

def _esc(x):
    return html.escape(str(x), quote=True)


def _table(headers, rows, caption=None, bold=None):
    """rows: [[first, cell, ...]]; bold: set of (row, col) positions to emphasize."""
    bold = bold or set()
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = ""
    for i, row in enumerate(rows):
        cells = ""
        for j, cell in enumerate(row):
            text = _esc(cell)
            if (i, j) in bold:
                text = f"<strong>{text}</strong>"
            cells += f"<td>{text}</td>"
        body += f"<tr>{cells}</tr>"
    cap = f'<p class="caption">{_esc(caption)}</p>' if caption else ""
    return (f'{cap}<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")


PERFORMANCE_COLUMNS = [
    ("Total ret (%)", "Total return", "{:+.2f}%", 1),
    ("Ann. ret (%)", "Ann. return", "{:+.2f}%", 1),
    ("Ann. vol (%)", "Ann. vol", "{:.2f}%", 0),
    ("Sharpe", "Sharpe", "{:.2f}", 1),
    ("CI", "95% CI", None, 0),
    ("PSR", "PSR (>0)", "{:.2f}", 0),
    ("Sortino", "Sortino", "{:.2f}", 1),
    ("Calmar", "Calmar", "{:.2f}", 1),
    ("Omega", "Omega", "{:.2f}", 1),
    ("Win rate (%)", "Win rate", "{:.1f}%", 0),
]
RISK_COLUMNS = [
    ("MaxDD (%)", "Max drawdown", "{:.2f}%", -1),
    ("CVaR 95% (%)", "CVaR 95%", "{:.2f}%", 1),
    ("Skew", "Skew", "{:.2f}", 0),
    ("Ex. kurt", "Excess kurtosis", "{:.2f}", 0),
    ("Beta", "Beta to B&H", "{:.2f}", 0),
    ("Alpha (%)", "Alpha per year", "{:+.2f}%", 0),
    ("Avg |pos|", "Mean |position|", "{:.2f}", 0),
    ("Net pos", "Mean net position", "{:+.2f}", 0),
    ("Turnover/yr", "Turnover per year", "{:.1f}", 0),
]


def _metric_rows(frame, columns):
    rows, bold = [], set()
    for name, row in frame.iterrows():
        cells = [label(name)]
        for key, _, fmt, _ in columns:
            if key == "CI":
                cells.append(f"{row['CI lo']:.2f} to {row['CI hi']:.2f}")
            elif key not in row or pd.isna(row[key]):
                cells.append("–")
            else:
                cells.append(fmt.format(row[key]))
        rows.append(cells)
    for j, (key, _, _, sense) in enumerate(columns, start=1):
        if sense:
            bold.add((int(np.argmax(frame[key].astype(float).values * sense)), j))
    return rows, bold


def metrics_table(metrics, order):
    """Two tables sorted by Sharpe; the best value in a scored column is bold."""
    frame = metrics.loc[order].sort_values("Sharpe", ascending=False)
    out = ""
    for title, columns in (("Performance", PERFORMANCE_COLUMNS),
                           ("Risk and behaviour", RISK_COLUMNS)):
        rows, bold = _metric_rows(frame, columns)
        out += _table(["Strategy"] + [c[1] for c in columns], rows, caption=title,
                      bold=bold)
    return out


def tile(name, value, note=""):
    return (f'<div class="tile"><div class="tile-label">{_esc(name)}</div>'
            f'<div class="tile-value">{_esc(value)}</div>'
            f'<div class="tile-note">{_esc(note)}</div></div>')


def paired_tests(run):
    rets = run["returns"]
    if "meta_rl" not in rets:
        return []
    out = []
    for other in [s for s in run["strategies"] if s != "meta_rl"]:
        diff, p = bootstrap_sharpe_diff_pvalue(rets["meta_rl"].values,
                                               rets[other].values, ANN)
        out.append((other, diff, p))
    return out


def _verdict_word(p):
    if p < 0.05:
        return "Meta-RL higher (5% level)"
    if p > 0.95:
        return "Meta-RL lower (5% level)"
    return "No significant difference"


def _comparison_phrase(p, other):
    if p < 0.05:
        return f"significantly above {other}"
    if p > 0.95:
        return f"significantly below {other}"
    return f"not significantly different from {other}"


def takeaway(run, tests):
    m = run["metrics"]
    if "meta_rl" not in m.index:
        return ""
    meta = m.loc["meta_rl"]
    p_by_name = {n: p for n, _, p in tests}
    phrases = []
    for bench in ("buy_and_hold", "vol_target_bh"):
        if bench in m.index:
            other = f"{label(bench)} ({m.loc[bench, 'Sharpe']:.2f})"
            phrases.append(_comparison_phrase(p_by_name[bench], other))
    rule = [a for a in ALLOCATORS[1:] if a in m.index]
    if rule:
        best = max(rule, key=lambda a: m.loc[a, "Sharpe"])
        other = (f"the best rule-based allocator, {label(best)} "
                 f"({m.loc[best, 'Sharpe']:.2f})")
        phrases.append(_comparison_phrase(p_by_name[best], other))
    text = (f"Meta-RL Sharpe is {meta['Sharpe']:.2f} "
            f"(95% CI {meta['CI lo']:.2f} to {meta['CI hi']:.2f}).")
    if phrases:
        joined = (", ".join(phrases[:-1]) + " and " + phrases[-1]
                  if len(phrases) > 1 else phrases[0])
        text += f" It is {joined}, on one-sided paired bootstrap tests at the 5% level."
    return text


def run_window(run):
    d = run["returns"].index
    return f"{d[0]:%Y-%m-%d} to {d[-1]:%Y-%m-%d}"


def runs_table(runs):
    rows = []
    for run in runs:
        m, cell = run["metrics"], run["cells"][0]
        p0, p2 = run_steps(run)
        train = (f"{cell['meta_cfg'].get('train_start', '?')} to "
                 f"{cell['meta_cfg'].get('train_end', '?')}"
                 if not run["wf_cells"] else "rolling")
        sharpe = lambda n: f"{m.loc[n, 'Sharpe']:.2f}" if n in m.index else "–"
        rule = [a for a in ALLOCATORS[1:] if a in m.index]
        best = max((m.loc[a, "Sharpe"] for a in rule), default=np.nan)
        rows.append([
            run["id"], train, run_window(run),
            f"{p0:,}" if p0 else "?", f"{p2:,}" if p2 else "?",
            f"{cell['cost'] * 1e4:.1f}", sharpe("meta_rl"),
            "–" if np.isnan(best) else f"{best:.2f}",
            sharpe("vol_target_bh"), sharpe("buy_and_hold"),
        ])
    return _table(["Run", "Train", "Test", "Agent steps", "Meta steps", "Cost (bp)",
                   "Meta-RL", "Best rule", "Vol-target B&H", "Buy & hold"], rows,
                  caption="Sharpe of every run. Test windows differ, so compare "
                          "each row against its own benchmarks.")


# ═══════════════════════════════════════════════════════════════════════════
#  Sections
# ═══════════════════════════════════════════════════════════════════════════

def section_overview(run, runs, tests):
    m = run["metrics"]
    tiles = []
    if "meta_rl" in m.index:
        meta, bh = m.loc["meta_rl"], m.loc["buy_and_hold"]
        best = m.loc[run["strategies"], "Sharpe"].astype(float).idxmax()
        tiles = [
            tile("Meta-RL Sharpe", f"{meta['Sharpe']:.2f}",
                 f"95% CI {meta['CI lo']:.2f} to {meta['CI hi']:.2f}"),
            tile("Meta-RL total return", f"{meta['Total ret (%)']:+.2f}%",
                 f"Buy & hold {bh['Total ret (%)']:+.2f}%"),
            tile("Meta-RL max drawdown", f"{meta['MaxDD (%)']:.2f}%",
                 f"Buy & hold {bh['MaxDD (%)']:.2f}%"),
            tile("Highest Sharpe", label(best), f"{m.loc[best, 'Sharpe']:.2f}"),
        ]
    bars = len(run["returns"])
    years = run["returns"].index.year.nunique()
    caution = ("One test year gives wide Sharpe intervals, so draw conclusions from "
               "the walk-forward results." if years == 1 else "")
    note = ""
    p0, p2 = run_steps(run)
    if p0 and p2 and p2 < 100_000:
        note = (f'<p class="flag">Short training budget ({p0:,} agent steps, '
                f"{p2:,} meta steps). This run checks the pipeline; it is not "
                f"a thesis result.</p>")
    blocks = [block_html(
        f'<div class="tiles">{"".join(tiles)}</div>'
        f'<p class="takeaway">{_esc(takeaway(run, tests))}</p>'
        f'<p class="muted">{bars} out-of-sample trading days, {_esc(run_window(run))}. '
        f"{caution}</p>{note}")]
    if len(runs) > 1:
        blocks.append(block_html(runs_table(runs)))
    return {"id": "overview", "title": "Overview", "intro": "", "blocks": blocks}


def section_agents(run, window):
    names = run["agents"]
    if not names:
        return None
    benchmarks = [b for b in BENCHMARKS if b in run["returns"]]
    rets = pd.concat([run["agent_ret"][names], run["returns"][benchmarks]], axis=1)
    blocks = [
        block_html(metrics_table(run["metrics"], names + benchmarks)),
        chart("Equity", "Each frozen agent trading alone, after costs, against the "
              "two benchmarks.", fig_equity(rets), "full"),
        chart("Drawdown", "Percent below the running equity peak.", fig_drawdown(rets)),
        chart("Maximum drawdown", "Deepest loss from peak. Hover for its dates.",
              fig_max_drawdown(rets)),
        chart("Rolling Sharpe", f"Annualized over {window} trading days.",
              fig_rolling_sharpe(rets, window)),
        chart("Daily return distribution",
              "Smoothed density. A narrow peak means small, steady daily moves.",
              fig_distribution(rets)),
        chart("Correlation of daily returns",
              "Agents only diversify the blend if their returns differ.",
              fig_correlation(rets)),
        chart("Exposure ladder",
              "Mean |position| of each agent; the black tick is its hard cap "
              "from config.py. The risk ladder only exists if these bars differ.",
              fig_exposure_ladder(run)),
        chart("Sharpe by volatility regime",
              "Regime set from realized vol at the previous close, using "
              "training-period terciles when saved. Days in each regime in "
              "brackets.", fig_regime_sharpe(rets, run["regimes"]), "full"),
        chart("Period returns", "Compounded return per period.",
              fig_periodic_returns(rets), "full"),
        chart("Positions over the test window",
              "Above zero is long, below zero is short, on one scale, so size "
              "differences are directly visible.",
              fig_positions_on_price(run, run["agent_pos"], names), "full"),
    ]
    return {"id": "agents", "title": "Single agents",
            "intro": "Phase 0: the frozen SAC agents on the risk ladder, each "
                     "traded alone on the test window.", "blocks": blocks}


def section_models(run, window, tests):
    names = [a for a in ALLOCATORS if a in run["returns"]]
    order = names + [b for b in BENCHMARKS if b in run["returns"]]
    rets = run["returns"][order]
    with_agents = pd.concat([rets, run["agent_ret"][run["agents"]]], axis=1)
    blocks = [
        block_html(metrics_table(run["metrics"], order)),
        chart("Equity", "Meta-RL against the Phase 1 allocators and benchmarks. "
              "Agents are in the legend, switched off; click to add them.",
              fig_equity(with_agents, hidden=run["agents"]), "full"),
        chart("Sharpe with 95% confidence intervals",
              "Stationary bootstrap over all models, agents and benchmarks.",
              fig_sharpe_ci(run["metrics"], order + run["agents"])),
        chart("Maximum drawdown", "Deepest loss from peak. Hover for its dates.",
              fig_max_drawdown(rets)),
        chart("Drawdown", "Percent below the running equity peak.",
              fig_drawdown(rets), "full"),
    ]
    if "meta_rl" in rets:
        blocks.append(chart(
            "Meta-RL relative to each alternative",
            "Above zero means Meta-RL has grown more since the start of the window.",
            fig_excess(rets), "full"))
    blocks += [
        chart("Rolling Sharpe", f"Annualized over {window} trading days.",
              fig_rolling_sharpe(rets, window)),
    ]
    target = run["cells"][0]["meta_cfg"].get("vol_target")
    blocks += [
        chart("Rolling volatility",
              "Realized risk of each strategy against the overlay's vol target.",
              fig_rolling_vol(rets, window, target or None)),
        chart("Daily return distribution",
              "Smoothed density. A narrow peak means small, steady daily moves.",
              fig_distribution(rets)),
        chart("Correlation of daily returns",
              "High correlation means the allocators end up holding similar books.",
              fig_correlation(rets)),
        chart("Agent weights over time",
              "Share of capital given to each agent. Only Meta-RL can hold cash.",
              fig_weight_stacks(run), "full"),
        chart("Average weights by volatility regime",
              "The thesis claim in one chart: does capital move to the conservative "
              "agent when volatility rises? Vol regime and HMM do this by design.",
              fig_weights_by_regime(run), "full"),
    ]
    if "meta_rl" in run["weights"]:
        diagnostics = fig_meta_diagnostics(run)
        blocks.append(chart(
            "Meta-RL diagnostics",
            "How it allocates: de-risking, overlay sizing, concentration and churn.",
            diagnostics))
        blocks.append(chart(
            "Executed position vs. agent proposals",
            "The position Meta-RL ran, after cash and the vol overlay, against what "
            "each agent proposed.", fig_proposals(run, diagnostics.layout.height)))
    blocks += [
        chart("Executed position by allocator",
              "Same scale in every panel, so the size of each bet can be compared.",
              fig_positions_on_price(run, run["exec_pos"],
                                     [a for a in names if a in run["exec_pos"]]),
              "full"),
        chart("Sharpe by volatility regime",
              "Where each strategy earns its risk-adjusted return. Days in each "
              "regime in brackets.", fig_regime_sharpe(rets, run["regimes"]), "full"),
        chart("Period returns", "Compounded return per period.",
              fig_periodic_returns(rets), "full"),
    ]
    if tests:
        rows = [[f"Meta-RL vs. {label(n)}", f"{d:+.2f}", f"{p:.3f}", _verdict_word(p)]
                for n, d, p in tests]
        blocks.append(block_html(
            "<h3>Paired bootstrap tests</h3>"
            '<p class="muted">Both strategies are resampled on the same days. p is '
            "the share of resamples where the Meta-RL Sharpe is not above the other "
            "strategy's.</p>" +
            _table(["Comparison", "Sharpe difference", "p", "Result"], rows)))
    return {"id": "models", "title": "Model testing",
            "intro": "Phase 2 Meta-RL against the Phase 1 allocators (HMM, Vol regime, "
                     "Equal weight) and the benchmarks.", "blocks": blocks}


def section_walkforward(run):
    if not run["wf_cells"]:
        return None
    res = aggregate(run["wf_cells"], ANN)
    folds, seeds = res["folds"], res["seeds"]
    planned = len(config.WF_TEST_YEARS) * len(config.WF_SEEDS)
    done = len(run["wf_cells"])
    keys = [("bars", "Days", "{:.0f}"), ("ann_ret_pct", "Ann. return", "{:+.2f}%"),
            ("sharpe", "Sharpe", "{:.2f}"), ("sharpe_ci_lo", "CI low", "{:.2f}"),
            ("sharpe_ci_hi", "CI high", "{:.2f}"), ("psr_vs_0", "PSR (>0)", "{:.2f}"),
            ("dsr", "Deflated Sharpe", "{:.2f}"), ("sortino", "Sortino", "{:.2f}"),
            ("max_dd_pct", "Max DD", "{:.2f}%"),
            ("total_ret_pct", "Total return", "{:+.2f}%"),
            ("seed_sharpe_mean", "Seed mean", "{:.2f}"),
            ("seed_sharpe_std", "Seed std", "{:.2f}")]
    rows = [[label(n)] + [f.format(r[k]) for k, _, f in keys]
            for n, r in res["summary"].iterrows()]
    flag = ""
    if done < planned:
        flag = (f'<p class="flag">Incomplete: {done} of the {planned} fold × seed '
                f"cells in config.py are finished, so everything on this page "
                f"covers only part of the planned test years.</p>")
    fold_text = (f"Test year {folds[0]}" if len(folds) == 1 else
                 f"Test years {', '.join(map(str, folds))}")
    seed_text = f"seed{'s' if len(seeds) > 1 else ''} {', '.join(map(str, seeds))}"
    blocks = [block_html(
        f'<p class="muted">{fold_text}; {seed_text}. Daily returns are averaged over '
        f"seeds and chained across test years. The other sections on this page show "
        f"the same pooled series.</p>{flag}"
        + _table(["Strategy"] + [k[1] for k in keys], rows))]
    if len(folds) > 1:
        blocks.append(chart("Sharpe by test year", "Mean over seeds.",
                            fig_per_fold(res["per_fold"]),
                            "half" if len(seeds) > 1 else "full"))
    if len(seeds) > 1:
        blocks.append(chart(
            "Seed dispersion",
            "Each dot is one seed's pooled Sharpe. Spread here is training noise.",
            fig_seed_dispersion(res["per_seed_sharpe"]),
            "half" if len(folds) > 1 else "full"))
    if len(res["tests"]):
        rows = [[r["comparison"].replace("meta_rl − ", "Meta-RL vs. "),
                 f"{r['sharpe_diff']:+.2f}", f"{r['p_value_H0_diff<=0']:.3f}"]
                for _, r in res["tests"].iterrows()]
        for row in rows:
            for key in ("equal_weight", "vol_target_bh", "buy_and_hold"):
                row[0] = row[0].replace(key, label(key))
        blocks.append(block_html("<h3>Paired bootstrap tests, pooled</h3>" + _table(
            ["Comparison", "Sharpe difference", "p (H0: difference ≤ 0)"], rows)))
    return {"id": "walkforward", "title": "Walk-forward",
            "intro": "Rolling folds and seeds: the protocol from walk_forward.py.",
            "blocks": blocks}


def section_details(run):
    cell = run["cells"][0]
    meta = cell["meta_cfg"]
    p0, p2 = run_steps(run)
    onoff = lambda v: "unknown" if v is None else ("on" if v else "off")
    items = [
        ("Folder", run["id"]),
        ("Test window", run_window(run)),
        ("Trading days", len(run["returns"])),
        ("Ticker", meta.get("ticker", config.TICKER)),
        ("Agent steps", f"{p0:,}" if p0 else "unknown"),
        ("Meta-agent steps", f"{p2:,}" if p2 else "unknown"),
        ("Transaction cost", f"{cell['cost'] * 1e4:.1f} bp per side"),
    ]
    if not run["wf_cells"]:
        items.insert(1, ("Train window", f"{meta.get('train_start', '?')} to "
                                         f"{meta.get('train_end', '?')}"))
        items.append(("Seed", meta.get("seed", "unknown")))
    items += [
        ("Meta reward", meta.get("meta_reward", "unknown")),
        ("Cash arm", onoff(meta.get("use_cash"))),
        ("Vol target", f"{meta['vol_target']:.0%} (max leverage "
                       f"{meta.get('vol_target_cap', '?')})"
         if meta.get("vol_target") else "off"),
        ("HMM regime input", onoff(meta.get("use_hmm"))),
        ("CNN features", onoff(meta.get("use_cnn"))),
    ]
    config_html = _table(["Setting", "Value"], [[k, v] for k, v in items])

    m = run["metrics"]
    rows = []
    for a in run["agents"]:
        preset = config.AGENT_PRESETS[a]
        vals = [c["agent_cfg"][a].get("best_val_sharpe") for c in run["cells"]
                if c["agent_cfg"].get(a)]
        vals = [v for v in vals if v is not None]
        rows.append([label(a), preset["lam_var"], preset["max_pos"],
                     f"{preset['dd_free']:.0%} / {preset['dd_max']:.0%}",
                     f"{np.mean(vals):.2f}" if vals else "–",
                     f"{m.loc[a, 'Sharpe']:.2f}"])
    vals = [c["meta_cfg"].get("best_val_sharpe") for c in run["cells"]]
    vals = [v for v in vals if v is not None]
    if "meta_rl" in m.index:
        rows.append(["Meta-RL", "–", meta.get("vol_target_cap", "–"),
                     f"{meta.get('dd_free', 0):.0%} / {meta.get('dd_max', 0):.0%}"
                     if meta.get("dd_max") else "–",
                     f"{np.mean(vals):.2f}" if vals else "–",
                     f"{m.loc['meta_rl', 'Sharpe']:.2f}"])
    agents_html = _table(["Model", "lam_var", "Max pos", "DD free / max",
                          "Val. Sharpe", "Test Sharpe"], rows,
                         caption="A large drop from validation to test points to "
                                 "overfitting or a regime change.")
    method = (
        '<p class="muted">All returns are daily, simple and net of transaction costs. '
        "Sharpe uses no risk-free rate and annualizes by √252. Intervals are stationary "
        "bootstrap percentiles (2,000 resamples, mean block 20 days). PSR is the "
        "Probabilistic Sharpe Ratio against zero. Beta and alpha come from a regression "
        "on buy & hold. Single-agent results are computed from the positions in "
        "phase2_actions.csv. Turnover is the yearly sum of |position change|.</p>")
    return {"id": "details", "title": "Run details", "intro": "", "blocks": [
        block_html(config_html, "half"), block_html(agents_html, "half"),
        block_html(method)]}


def build_payload(runs, window):
    payload = []
    for run in runs:
        tests = paired_tests(run)
        sections = [section_overview(run, runs, tests),
                    section_walkforward(run),
                    section_agents(run, window),
                    section_models(run, window, tests),
                    section_details(run)]
        payload.append({"id": run["id"],
                        "sections": [s for s in sections if s]})
    return payload


# ═══════════════════════════════════════════════════════════════════════════
#  Page
# ═══════════════════════════════════════════════════════════════════════════

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HRL Trading Results</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e;
  --muted: #6f6d67; --line: #e1e0d9; --ring: rgba(11,11,11,.10); --accent: #2a78d6;
  --flag-bg: #fdf3dc; --flag-ink: #5f4300;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7;
    --muted: #a5a39b; --line: #2c2c2a; --ring: rgba(255,255,255,.10); --accent: #3987e5;
    --flag-bg: #3a2d0c; --flag-ink: #f5dc9f;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7;
  --muted: #a5a39b; --line: #2c2c2a; --ring: rgba(255,255,255,.10); --accent: #3987e5;
  --flag-bg: #3a2d0c; --flag-ink: #f5dc9f;
}
* { box-sizing: border-box; }
html { scroll-padding-top: 112px; }
body { margin: 0; background: var(--page); color: var(--ink);
       font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { position: sticky; top: 0; z-index: 10; background: var(--page);
         border-bottom: 1px solid var(--line); }
.bar { max-width: 1320px; margin: 0 auto; padding: 10px 24px;
       display: flex; flex-wrap: wrap; align-items: center; gap: 8px 18px; }
h1 { font-size: 17px; font-weight: 650; margin: 0; }
label { font-size: 13px; color: var(--ink2); display: flex; align-items: center; gap: 6px; }
select, button { font: inherit; font-size: 13px; color: var(--ink); background: var(--surface);
                 border: 1px solid var(--line); border-radius: 6px; padding: 5px 9px;
                 min-height: 32px; max-width: 100%; }
button { cursor: pointer; margin-left: auto; }
select:focus-visible, button:focus-visible, nav a:focus-visible {
  outline: 2px solid var(--accent); outline-offset: 1px; }
nav { display: flex; flex-wrap: wrap; gap: 2px 14px; flex-basis: 100%; }
nav a { font-size: 13px; color: var(--ink2); text-decoration: none; padding: 2px 0; }
nav a:hover { color: var(--ink); text-decoration: underline; }
main { max-width: 1320px; margin: 0 auto; padding: 8px 24px 64px; }
section { margin-top: 28px; }
h2 { font-size: 21px; font-weight: 650; margin: 0 0 2px; }
h3 { font-size: 14px; font-weight: 600; margin: 0; }
.intro { margin: 0 0 12px; color: var(--ink2); }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 12px; }
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px;
        padding: 14px 14px 8px; margin: 0; min-width: 0; }
.full { grid-column: 1 / -1; }
.card figcaption { margin-bottom: 4px; }
.card figcaption p { margin: 1px 0 0; font-size: 13px; color: var(--ink2); }
.plot { width: 100%; }
.card > h3 { margin-bottom: 4px; }
.tiles { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.tile-label { font-size: 13px; color: var(--ink2); }
.tile-value { font-size: 27px; font-weight: 650; line-height: 1.25; overflow-wrap: anywhere; }
.tile-note { font-size: 13px; color: var(--muted); }
.takeaway { margin: 14px 0 2px; }
.muted { margin: 6px 0; font-size: 13px; color: var(--ink2); }
.flag { margin: 8px 0; padding: 7px 10px; border-radius: 6px; font-size: 13px;
        background: var(--flag-bg); color: var(--flag-ink); }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; font-size: 13px; width: 100%; font-variant-numeric: tabular-nums; }
.caption { margin: 0 0 4px; font-size: 13px; font-weight: 600; }
.table-wrap + .caption { margin-top: 16px; }
th, td { padding: 5px 10px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--line); }
th { font-weight: 600; color: var(--ink2); vertical-align: bottom; }
th:first-child, td:first-child { text-align: left; position: sticky; left: 0; background: var(--surface); }
tbody tr:last-child td { border-bottom: 0; }
footer { max-width: 1320px; margin: 0 auto; padding: 0 24px 32px; font-size: 12px; color: var(--muted); }
@media (max-width: 960px) {
  .grid { grid-template-columns: minmax(0, 1fr); }
  .tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 640px) {
  .bar, main, footer { padding-left: 16px; padding-right: 16px; }
  .tile-value { font-size: 22px; }
  html { scroll-padding-top: 150px; }
}
</style>
</head>
<body>
<header><div class="bar">
  <h1>HRL Trading Results</h1>
  <label>Run <select id="run"></select></label>
  <button id="theme" type="button"></button>
  <nav id="nav" aria-label="Sections"></nav>
</div></header>
<main id="root"></main>
<footer>Generated __DATE__ by dashboard.py from the CSV files in each run folder.</footer>
<script>__PLOTLY__</script>
<script type="application/json" id="payload">__PAYLOAD__</script>
<script>
(function () {
  var D = JSON.parse(document.getElementById("payload").textContent);
  var root = document.getElementById("root");
  var select = document.getElementById("run");
  var nav = document.getElementById("nav");
  var themeBtn = document.getElementById("theme");
  var html = document.documentElement;
  var observer = null;
  var keys = Object.keys(D.dark).join("|");
  var darkRe = new RegExp(keys, "gi");
  var CONFIG = { responsive: true, displaylogo: false,
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "zoomIn2d", "zoomOut2d",
      "toggleSpikelines", "hoverClosestCartesian", "hoverCompareCartesian"],
    toImageButtonOptions: { format: "png", scale: 3 } };

  function isDark() {
    if (html.dataset.theme) return html.dataset.theme === "dark";
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  function themed(spec) {
    var text = JSON.stringify({ data: spec.data, layout: spec.layout });
    if (isDark()) text = text.replace(darkRe, function (m) { return D.dark[m.toLowerCase()]; });
    return JSON.parse(text);
  }
  function fitNarrow(t, width) {
    if (width >= 560) return;
    var l = t.layout;
    if (l.annotations) {
      var kept = l.annotations.filter(function (a) { return a.name !== "endlabel"; });
      if (kept.length !== l.annotations.length) { l.annotations = kept; l.margin.r = 16; }
    }
    if (l.legend) { l.legend.entrywidth = 112; l.legend.entrywidthmode = "pixels"; }
    Object.keys(l).forEach(function (k) {
      if (/^xaxis/.test(k) && l[k].type === "date") { l[k].nticks = 4; l[k].tickangle = 0; }
    });
    t.data.forEach(function (d) {
      if (d.visible === "legendonly") d.showlegend = false;
      if (d.type === "heatmap" && d.z[0].length > 8) d.texttemplate = "";
    });
  }
  function draw(el) {
    var spec = el.__spec, t = themed(spec);
    fitNarrow(t, el.clientWidth);
    var cfg = Object.assign({}, CONFIG, { toImageButtonOptions: Object.assign(
      {}, CONFIG.toImageButtonOptions, { filename: spec.file, height: spec.height, width: 1100 }) });
    Plotly.react(el, t.data, t.layout, cfg);
    el.__drawn = true;
  }
  function slug(s) { return s.replace(/[^a-z0-9]+/gi, "_").toLowerCase(); }

  function render(id) {
    var run = D.runs.filter(function (r) { return r.id === id; })[0] || D.runs[0];
    select.value = run.id;
    if (observer) observer.disconnect();
    root.querySelectorAll(".plot").forEach(function (el) { if (el.__drawn) Plotly.purge(el); });
    root.textContent = "";
    nav.textContent = "";
    observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { draw(e.target); observer.unobserve(e.target); }
      });
    }, { rootMargin: "500px 0px" });

    run.sections.forEach(function (sec) {
      var a = document.createElement("a");
      a.href = "#" + sec.id;
      a.textContent = sec.title;
      nav.appendChild(a);

      var s = document.createElement("section");
      s.id = sec.id;
      var h = document.createElement("h2");
      h.textContent = sec.title;
      s.appendChild(h);
      if (sec.intro) {
        var p = document.createElement("p");
        p.className = "intro";
        p.textContent = sec.intro;
        s.appendChild(p);
      }
      var grid = document.createElement("div");
      grid.className = "grid";
      sec.blocks.forEach(function (b) {
        var cls = "card" + (b.span === "full" ? " full" : "");
        if (b.type === "html") {
          var d = document.createElement("div");
          d.className = cls;
          d.innerHTML = b.html;       // built server-side from escaped text
          grid.appendChild(d);
          return;
        }
        var f = document.createElement("figure");
        f.className = cls;
        var c = document.createElement("figcaption");
        var t = document.createElement("h3");
        t.textContent = b.title;
        var sub = document.createElement("p");
        sub.textContent = b.sub;
        c.appendChild(t);
        c.appendChild(sub);
        var plot = document.createElement("div");
        plot.className = "plot";
        plot.style.height = b.height + "px";
        plot.setAttribute("role", "img");
        plot.setAttribute("aria-label", b.title + ". " + b.sub);
        plot.__spec = { data: b.data, layout: b.layout, height: b.height,
                        file: slug(run.id + "_" + sec.id + "_" + b.title) };
        f.appendChild(c);
        f.appendChild(plot);
        grid.appendChild(f);
        observer.observe(plot);
      });
      s.appendChild(grid);
      root.appendChild(s);
    });
  }

  function paintThemeButton() { themeBtn.textContent = isDark() ? "Light theme" : "Dark theme"; }
  themeBtn.addEventListener("click", function () {
    html.dataset.theme = isDark() ? "light" : "dark";
    paintThemeButton();
    root.querySelectorAll(".plot").forEach(function (el) { if (el.__drawn) draw(el); });
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
    paintThemeButton();
    root.querySelectorAll(".plot").forEach(function (el) { if (el.__drawn) draw(el); });
  });

  D.runs.forEach(function (r) {
    var o = document.createElement("option");
    o.value = r.id;
    o.textContent = r.id;
    select.appendChild(o);
  });
  select.addEventListener("change", function () { window.scrollTo(0, 0); render(select.value); });
  paintThemeButton();
  render(D.runs[0].id);
})();
</script>
</body>
</html>
"""


def write_page(payload, out):
    body = json.dumps({"runs": payload, "dark": DARK_MAP}, separators=(",", ":"))
    page = (PAGE
            .replace("__DATE__", dt.date.today().isoformat())
            .replace("__PAYLOAD__", body.replace("</", "<\\/"))
            .replace("__PLOTLY__", get_plotlyjs()))
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)


def main():
    p = argparse.ArgumentParser(
        description="Build one interactive HTML dashboard from evaluated runs.")
    p.add_argument("--runs", nargs="+", default=["runs"],
                   help="Run folders, or folders to search for runs (default: runs)")
    p.add_argument("--out", default="dashboard.html", help="Output HTML file")
    p.add_argument("--window", type=int, default=60,
                   help="Rolling window in trading days (default: 60)")
    args = p.parse_args()

    runs = load_runs(args.runs)
    if not runs:
        sys.exit(f"No phase2_returns.csv found under {', '.join(args.runs)}.\n"
                 "Create results first, for example:  "
                 "python run_all.py --outdir runs/sanity --quick")

    payload = build_payload(runs, args.window)
    write_page(payload, args.out)

    print(f"{len(runs)} run(s):")
    for run in runs:
        m = run["metrics"]
        sharpe = f"{m.loc['meta_rl', 'Sharpe']:+.2f}" if "meta_rl" in m.index else "n/a"
        print(f"  {run['id']:<28} {run_window(run)}   Meta-RL Sharpe {sharpe}")
    size = os.path.getsize(args.out) / 1e6
    print(f"Saved → {args.out} ({size:.1f} MB). Open it in a browser.")


if __name__ == "__main__":
    main()
