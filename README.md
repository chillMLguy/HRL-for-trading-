# Hierarchical RL Trading System
 
Master's thesis project: a multi-agent reinforcement learning system for trading, where diverse risk-profile agents are coordinated by a regime-aware meta-controller.

## Phase 0 — Independent Agents
 
Four SAC agents trained on the same asset (^DJI) with identical observations but different risk-aversion parameters (λ)

Each agent sees 20 features (momentum, volatility, higher-order stats, portfolio state, CNN latent features) and outputs a continuous position in [-1, 1].
 
```bash
python train_agents.py --ticker "^DJI" --start 2011-01-01 --end 2021-12-31 --interval 1d
python evaluate_agents.py --ticker "^DJI" --start 2022-01-01 --end 2022-12-31
```
 
## Phase 1 — Portfolio Allocation via Regime Detection
 
A high-level allocator blends the four agents' proposed actions using weights determined by market regime. Three methods are compared:
 
- **Equal Weight** — constant 1/3 split (baseline)
- **Volatility Regime** — rule-based: rolling vol percentiles map to fixed weight profiles (aggressive in calm markets, conservative in turbulent)
- **HMM** — Gaussian Hidden Markov Model fitted on log returns + rolling volatility; posterior state probabilities drive agent weights through a learned mapping
 
```bash
python train_phase1.py --ticker "^DJI" --train_start 2011-01-01 --train_end 2021-12-31
python evaluate_phase1.py --ticker "^DJI" --test_start 2022-01-01 --test_end 2022-12-31
python plot_phase1.py
```
 
## Setup
 
```bash
pip install -r requirements.txt
```