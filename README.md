# Phase 0 — Flat Baseline Agents

Three independent SAC agents trained on the same single asset.
The only difference between them is their risk exposure in **reward function**.
This is the empirical foundation for the hierarchical system.

## Project structure

```
phase0/
├── env/
│   └── trading_env.py      ← Custom Gymnasium environment + 3 reward functions
├── train_agents.py         ← Trains all three agents, saves models/
├── evaluate_agents.py      ← Loads models, computes metrics + regime table
├── plot_results.py         ← Plots equity curves, drawdown, rolling Sharpe
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Train (uses SPY 2010–2022 by default)
python train_agents.py --ticker SPY --start 2010-01-01 --end 2022-12-31

# 2. Evaluate on held-out test period
python evaluate_agents.py --ticker SPY --start 2023-01-01 --end 2024-12-31

# 3. Plot
python plot_results.py --csvpath equity_curves.csv
```
