"""
Checks for dashboard.py and common/stats.py (run: python tests/test_dashboard.py).

Uses a synthetic run folder with hand-checkable numbers, plus every real run
under runs/.  Needs no trained models.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config
import dashboard as d
from aggregate_results import aggregate, load_cells
from common.stats import (bootstrap_sharpe_ci, bootstrap_sharpe_diff_pvalue,
                          compute_metrics)

OK = []
COST = 0.0005


def check(name, cond):
    OK.append((name, bool(cond)))
    print(("  ✓ " if cond else "  ✗ FAIL ") + name)


def write_run(path, seed=0, start="2022-01-03", n=120, cash=True, extra_meta=False):
    """Minimal evaluate_phase2.py output with known positions and returns."""
    rng = np.random.default_rng(seed)
    os.makedirs(path, exist_ok=True)
    dates = pd.bdate_range(start, periods=n)
    allocs = ["meta_rl", "equal_weight", "vol_regime", "hmm"]
    if extra_meta:
        allocs.insert(1, "meta_rl_leaked")
    bh = rng.normal(4e-4, 0.01, n)

    returns = {"date": dates}
    action_rows, weight_rows = [], []
    agent_pos = {a: np.round(rng.uniform(-1, 1, n)
                             * config.AGENT_PRESETS[a]["max_pos"], 6)
                 for a in d.AGENTS}
    for alloc in allocs:
        w = rng.dirichlet([2, 2, 2, 1] if cash else [2, 2, 2], n)
        blended = sum(w[:, i] * agent_pos[a] for i, a in enumerate(d.AGENTS))
        turnover = np.abs(np.diff(blended, prepend=0.0))
        returns[alloc] = blended * bh - turnover * COST
        for t in range(n):
            action_rows.append({"step": t, "allocator": alloc,
                                "blended_action": blended[t],
                                **{f"{a}_action": agent_pos[a][t] for a in d.AGENTS}})
            row = {"step": t, "allocator": alloc,
                   **{f"w_{a}": w[t, i] for i, a in enumerate(d.AGENTS)}}
            if alloc.startswith("meta_rl"):
                row["w_cash"] = w[t, 3] if cash else 0.0
                row["lever"] = 1.2 if cash else 1.0
            weight_rows.append(row)
    returns["buy_and_hold"] = bh
    returns["vol_target_bh"] = bh * 0.8

    pd.DataFrame(returns).to_csv(os.path.join(path, "phase2_returns.csv"), index=False)
    pd.DataFrame(action_rows).to_csv(os.path.join(path, "phase2_actions.csv"), index=False)
    pd.DataFrame(weight_rows).to_csv(os.path.join(path, "phase2_weights.csv"), index=False)
    with open(os.path.join(path, "phase2_eval_config.json"), "w") as f:
        json.dump({"cost_pct": COST}, f)
    return agent_pos, bh


tmp = tempfile.mkdtemp()

# ── Single-agent results from saved positions ───────────────────────────────
agent_pos, bh = write_run(os.path.join(tmp, "one"))
cell = d.load_cell(os.path.join(tmp, "one"))
a = "balanced"
p = agent_pos[a]
expected = p * bh - np.abs(np.diff(p, prepend=0.0)) * COST
check("agent return = position × market return − cost × |position change|",
      np.allclose(cell["agent_ret"][a].values, expected, atol=1e-9))
check("first day pays cost for entering from flat",
      abs(cell["agent_ret"][a].iloc[0]
          - (p[0] * bh[0] - abs(p[0]) * COST)) < 1e-9)

# ── Basic statistics ────────────────────────────────────────────────────────
r = np.array([0.10, -0.10, 0.05])
check("equity starts at 1 and compounds",
      np.allclose(d.equity(r), [1.0, 1.1, 0.99, 1.0395]))
check("drawdown is measured from the running peak",
      np.allclose(d.drawdown_pct(r), [0.0, -10.0, -5.5]))
peak, trough, recovered = d.worst_drawdown(
    np.array([0.1, -0.2, 0.3, 0.01]), pd.bdate_range("2024-01-01", periods=4))
check("worst drawdown dates: peak, trough, recovery",
      (peak, trough, recovered) == ("2024-01-01", "2024-01-02", "Recovered 2024-01-03"))

# ── Regimes use only information from before the traded day ─────────────────
dates = pd.bdate_range("2022-01-03", periods=80)
ret = pd.DataFrame({"buy_and_hold": np.random.default_rng(1).normal(0, 0.01, 80)},
                   index=dates)
prices = pd.Series(100 * np.cumprod(1 + ret["buy_and_hold"].values), index=dates)
th = {"q33": 0.12, "q67": 0.18}
base = d.vol_regimes(ret, th, prices)
shocked_ret = ret.copy()
shocked_ret.iloc[50, 0] = 0.20
shocked_px = pd.Series(100 * np.cumprod(1 + shocked_ret["buy_and_hold"].values),
                       index=dates)
after = d.vol_regimes(shocked_ret, th, shocked_px)
check("a shock on day t leaves labels up to and including day t unchanged",
      (base.iloc[:51] == after.iloc[:51]).all())
check("the same shock changes the regime from day t+1 on",
      after.iloc[51] == "high_vol")
check("days without enough history are labelled n/a, not guessed",
      (base.iloc[:20] == "n/a").all())

# ── Bootstrap ───────────────────────────────────────────────────────────────
rng = np.random.default_rng(3)
strong = rng.normal(2e-3, 0.01, 1000)
weak = rng.normal(0.0, 0.01, 1000)
lo, hi = bootstrap_sharpe_ci(strong)
sr = np.sqrt(252) * strong.mean() / strong.std()
check("Sharpe CI brackets the point estimate", lo < sr < hi)
check("Sharpe CI is reproducible", (lo, hi) == bootstrap_sharpe_ci(strong))
_, p_up = bootstrap_sharpe_diff_pvalue(strong, weak)
_, p_down = bootstrap_sharpe_diff_pvalue(weak, strong)
check("paired test: p is small when A is clearly better", p_up < 0.05)
check("paired test: p is large when A is clearly worse", p_down > 0.95)

# ── Walk-forward pooling matches aggregate_results.py ───────────────────────
wf = os.path.join(tmp, "wf")
for fold, start in ((2022, "2022-01-03"), (2023, "2023-01-02")):
    for seed in (0, 1):
        write_run(os.path.join(wf, f"fold{fold}", f"seed{seed}"),
                  seed=fold + seed, start=start)
runs = d.load_runs([wf])
pooled = aggregate(load_cells(wf))["pooled_mean"]
check("a fold/seed folder tree becomes one walk-forward run",
      len(runs) == 1 and len(runs[0]["wf_cells"]) == 4)
check("pooled returns equal aggregate_results.py",
      np.allclose(runs[0]["returns"][pooled.columns].values, pooled.values))

# ── Full build, including a run with no cash arm or overlay ─────────────────
write_run(os.path.join(tmp, "nocash"), cash=False)
runs = d.load_runs([os.path.join(tmp, "one"), wf, os.path.join(tmp, "nocash")])
payload = d.build_payload(runs, 20)
out = os.path.join(tmp, "dash.html")
d.write_page(payload, out)
page = open(out, encoding="utf-8").read()
ids = {r["id"].split(os.sep)[-1]: [s["id"] for s in r["sections"]] for r in payload}
check("sections: single-agent section comes before model testing",
      ids["one"] == ["overview", "agents", "models", "overfitting", "details"])
check("walk-forward runs add their own section",
      ids["wf"] == ["overview", "walkforward", "agents", "models", "overfitting",
                    "details"])

titles = {b["title"] for r in payload for s in r["sections"] if s["id"] == "models"
          for b in s["blocks"] if b["type"] == "chart"}
wanted = {"Equity", "Maximum drawdown", "Drawdown", "Rolling Sharpe",
          "Daily return distribution", "Correlation of daily returns",
          "Agent weights over time", "Meta-RL diagnostics"}
check("model testing has every requested chart", wanted <= titles)

nocash = [r for r in payload if r["id"].endswith("nocash")][0]
stack = [b for s in nocash["sections"] for b in s["blocks"]
         if b.get("title") == "Agent weights over time"][0]
check("no cash band when the run has no cash arm",
      not any(t["name"] == "Cash" for t in stack["data"]))
check("page embeds plotly.js and loads no external script",
      len(page) > 1_000_000 and '<script src="http' not in page)

# ── Overfitting check ───────────────────────────────────────────────────────
one = [r for r in payload if r["id"].endswith("one")][0]
over = [s for s in one["sections"] if s["id"] == "overfitting"][0]
check("a run without window_sharpe.csv says how to create it",
      "overfit_check.py" in over["blocks"][0]["html"])

fit_dir = os.path.join(tmp, "fit")
write_run(fit_dir)
rows = []
for model, train, val in (("aggressive", 6.0, 1.0), ("balanced", 7.0, 0.5),
                          ("conservative", 8.0, 0.0)):
    rows.append(dict(model=model, window="train", proposals="", sharpe=train,
                     bars=500, bh_sharpe=0.7))
    rows.append(dict(model=model, window="val", proposals="", sharpe=val,
                     bars=100, bh_sharpe=0.7))
for model in ("equal_weight", "meta_rl"):
    rows.append(dict(model=model, window="train", proposals="in_sample", sharpe=9.0,
                     bars=500, bh_sharpe=0.7))
    rows.append(dict(model=model, window="val", proposals="in_sample", sharpe=1.0,
                     bars=100, bh_sharpe=0.7))
pd.DataFrame(rows).to_csv(os.path.join(fit_dir, "window_sharpe.csv"), index=False)
fit_runs = d.load_runs([fit_dir])
table = d.window_rows(fit_runs[0])
check("window table has agents, equal weight and Meta-RL",
      [r["model"] for r in table] == d.AGENTS + ["equal_weight", "meta_rl"])
check("window table takes test Sharpe from the run's metrics",
      table[-1]["test"] == float(fit_runs[0]["metrics"].loc["meta_rl", "Sharpe"]))
text, warning = d.overfit_summary(fit_runs[0])
check("summary reports the mean agent training Sharpe", "7.00" in text)
check("a training Sharpe above the threshold raises a warning", "memorized" in warning)
fit_page = d.build_payload(fit_runs, 20)[0]
over = [s for s in fit_page["sections"] if s["id"] == "overfitting"][0]
check("the section has the train/validation/test chart",
      [t["name"] for b in over["blocks"] if b["type"] == "chart" for t in b["data"]]
      == ["Training data", "Validation data", "Test data"])

# ── A cross-fitted run shows the leaked meta-agent as its own strategy ──────
cf_dir = os.path.join(tmp, "cf")
write_run(cf_dir, extra_meta=True)
os.makedirs(os.path.join(cf_dir, "models", "phase2"))
with open(os.path.join(cf_dir, "models", "phase2", "config.json"), "w") as f:
    json.dump({"training_mode": "crossfit", "crossfit_folds": 4,
               "episode_bars": 252}, f)
cf_runs = d.load_runs([cf_dir])
check("meta_rl_leaked is picked up as a strategy",
      "meta_rl_leaked" in cf_runs[0]["strategies"])
check("best rule-based allocator ignores meta_rl_leaked",
      "meta_rl_leaked" not in d.RULE_ALLOCATORS)
cf_tests = d.paired_tests(cf_runs[0])
check("paired tests include the leaked comparison",
      "meta_rl_leaked" in [n for n, _, _ in cf_tests])
check("the takeaway mentions the comparison",
      "in-sample agent behaviour" in d.takeaway(cf_runs[0], cf_tests))
cf_page = d.build_payload(cf_runs, 20)[0]
weights = [b for s in cf_page["sections"] for b in s["blocks"]
           if b.get("title") == "Agent weights over time"][0]
check("weight chart has a panel for the leaked meta-agent",
      any(a.get("text") == "Meta-RL (leaked)" for a in weights["layout"]["annotations"]))
details = [s for s in cf_page["sections"] if s["id"] == "details"][0]
check("run details name the training data",
      "Out-of-sample agent behaviour" in details["blocks"][0]["html"])

# ── Real runs: recomputed numbers must match what evaluation wrote ──────────
root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
singles, walkforward = d.discover([root])
folders = singles + [p for cells in walkforward.values() for p in cells]
for path in folders:
    name = os.path.relpath(path, root)
    cell = d.load_cell(path)
    csv = pd.read_csv(os.path.join(path, "phase2_metrics.csv"), index_col=0)
    worst = 0.0
    for col in csv.columns:
        ret_ = cell["returns"][col].values
        got = compute_metrics(ret_, d.equity(ret_))
        worst = max(worst, max(abs(got[k] - csv.loc[k, col]) for k in got))
    check(f"{name}: metrics match phase2_metrics.csv (worst diff {worst:.4f})",
          worst < 0.02)
    eq = pd.read_csv(os.path.join(path, "phase2_equity_curves.csv"))
    gap = max(np.abs(d.equity(cell["returns"][c].values) - eq[c].values).max()
              for c in csv.columns)
    check(f"{name}: equity matches phase2_equity_curves.csv", gap < 1e-4)

passed = sum(c for _, c in OK)
print(f"\n{passed}/{len(OK)} checks passed")
sys.exit(0 if passed == len(OK) else 1)
