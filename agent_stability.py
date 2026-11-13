"""
Do retrained agents behave alike?

Cross-fitting assumes the agents the meta-agent learns from behave like the agents it
is deployed with.  This rolls every agent set of a cross-fitted run (one per fold,
plus the final agents) over the run's test year and compares them:

    python agent_stability.py --rundir runs/crossfit_2024

Writes <rundir>/agent_stability.csv with one row per agent and set:

    agent           aggressive, balanced, conservative, or equal_weight
    set             fold1 ... foldN, final, or ensemble (the mean action of all sets)
    test_sharpe     annualized, net of costs
    mean_position   average net position
    corr_final      correlation of its actions with the final agents' actions

Every set was fitted before the test year, so nothing here looks ahead.  Sets differ
in both seed and fit window, so this cannot separate the two causes.
"""
import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from common.data import load_prices, warm_start_date, trading_start_index
from common.stats import ann_sharpe
from regime.allocators import AGENT_ORDER
from train_crossfit import propose_block
import config

ACTIONS = [f"a_{name}" for name in AGENT_ORDER]


def _row(agent, name, position, action, final, market, cost, ann):
    net = position * market - np.abs(np.diff(position, prepend=0.0)) * cost
    return {"agent": agent, "set": name,
            "test_sharpe": round(ann_sharpe(net, ann), 3),
            "mean_position": round(float(position.mean()), 3),
            "corr_final": round(float(np.corrcoef(action, final)[0, 1]), 3)}


def main():
    p = argparse.ArgumentParser(description="Compare retrained agents on the test year")
    p.add_argument("--rundir", required=True, help="A cross-fitted run with its models")
    p.add_argument("--data_csv", default=config.DATA_CSV)
    args = p.parse_args()

    with open(os.path.join(args.rundir, "phase2_eval_config.json")) as f:
        eval_cfg = json.load(f)
    with open(os.path.join(args.rundir, "crossfit", "config.json")) as f:
        folds = json.load(f)["folds"]
    with open(os.path.join(args.rundir, "models", "phase2", "config.json")) as f:
        train_cfg = json.load(f)
    ann = train_cfg.get("bars_per_year", 252)
    cost = train_cfg["cost_pct"]

    prices = load_prices(train_cfg["ticker"],
                         warm_start_date(eval_cfg["test_start"], config.WARM_BARS),
                         eval_cfg["test_end"], train_cfg.get("interval", config.INTERVAL),
                         data_csv=args.data_csv, verbose=False)
    first = trading_start_index(prices, eval_cfg["test_start"])

    sets = {f"fold{k}": os.path.join(args.rundir, "crossfit", f"fold{k}")
            for k in range(1, folds + 1)}
    sets["final"] = args.rundir
    actions = {name: propose_block(path, prices, first, len(prices), cost, ann)[ACTIONS]
               for name, path in sets.items()}
    actions["ensemble"] = sum(actions.values()) / len(actions)

    p_ = prices.values.astype(np.float64)
    bars = prices.index.searchsorted(actions["final"].index)
    bars = bars[bars + 1 < len(p_)]
    market = p_[bars + 1] / p_[bars] - 1.0
    caps = np.array([config.AGENT_PRESETS[a]["max_pos"] for a in AGENT_ORDER])

    rows = []
    final = actions["final"].values[: len(market)]
    for name, frame in actions.items():
        values = frame.values[: len(market)]
        for j, agent in enumerate(AGENT_ORDER):
            rows.append(_row(agent, name, values[:, j] * caps[j], values[:, j],
                             final[:, j], market, cost, ann))
        rows.append(_row("equal_weight", name, (values * caps).mean(axis=1),
                         (values * caps).mean(axis=1), (final * caps).mean(axis=1),
                         market, cost, ann))
    table = pd.DataFrame(rows)
    out = os.path.join(args.rundir, "agent_stability.csv")
    table.to_csv(out, index=False)

    print(table.pivot(index="agent", columns="set", values="test_sharpe")
          [list(actions)].to_string())
    single = [name for name in actions if name != "ensemble"]
    print("\nAction correlation between agent sets, same agent:")
    for j, agent in enumerate(AGENT_ORDER):
        corr = pd.DataFrame({n: actions[n].iloc[:, j].values for n in single}).corr().values
        pairs = corr[np.triu_indices(len(single), 1)]
        print(f"  {agent:13s} mean {pairs.mean():+.2f}, "
              f"min {pairs.min():+.2f}, max {pairs.max():+.2f}")
    inside = [actions[n].corr().values[np.triu_indices(len(AGENT_ORDER), 1)].mean()
              for n in single]
    print(f"Between different agents in one set: mean {np.mean(inside):+.2f}")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
