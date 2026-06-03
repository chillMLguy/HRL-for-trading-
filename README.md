# Hierarchical RL Trading System

Master's thesis project: a multi-agent reinforcement learning system for trading
the Dow Jones Index (`^DJI`), where diverse risk-profile agents are coordinated
by a regime-aware meta-controller. See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)
for the full research framing and competitive positioning.

The system is built bottom-up in phases. Each phase adds one layer of the
hierarchy and is independently trainable, evaluable, and plottable.

## Phase 0 — Independent λ-spectrum agents

Three SAC agents trained on `^DJI` with identical 20-feature observations
(momentum, volatility, higher-order stats, portfolio state, CNN latent features)
but different risk-aversion parameters `λ`:

| Agent | λ | Behaviour |
|-------|-----|-----------|
| `aggressive` | 0.0 | large positions, return-seeking |
| `balanced` | 0.75 | moderate risk |
| `conservative` | 1.5 | small positions, tight drawdowns |

Each agent outputs a continuous position in `[-1, 1]` (full short → full long).

```bash
python train_agents.py    --ticker "^DJI" --start 2013-01-01 --end 2023-12-31 --interval 1d
python evaluate_agents.py --ticker "^DJI" --start 2024-01-01 --end 2024-12-31 --interval 1d
python plot_results.py
```

## Phase 1 — Regime-based allocation (fixed rules)

A high-level allocator blends the three agents' proposed actions into a single
executed position. Three **non-learned** allocators are compared:

- **Equal Weight** — constant `1/3` split (baseline)
- **Volatility Regime** — rolling-vol percentile thresholds → fixed weight profiles
- **HMM** — Gaussian HMM (3 states) on log-returns + rolling vol; posterior state
  probabilities map to agent weights through a fixed matrix

```bash
python train_phase1.py    --ticker "^DJI" --train_start 2013-01-01 --train_end 2023-12-31
python evaluate_phase1.py --ticker "^DJI" --test_start 2024-01-01 --test_end 2024-12-31
python plot_phase1.py
```

## Phase 2 — Meta-RL allocator (learned)

A single SAC **meta-agent** replaces Phase 1's fixed weight-mapping matrix. The
sub-agents are **frozen** and act as a fixed part of the environment; the
meta-agent observes the HMM regime posterior, multi-timeframe market features,
the sub-agents' proposed actions and their recent performance, and outputs a
3-vector that is soft-maxed into allocation weights.

Reward (per step), see [env/meta_env.py](env/meta_env.py):

```
R_meta = net_ret / vol_scale  −  λ_meta·(dd_dev + dd_penalty)   ← base mean-risk term
         + α·H(weights)/log(N)         (diversity bonus — anti-collapse)
         − β·Σ|wₜ − wₜ₋₁|              (turnover penalty — anti-churn)
```

The HMM posterior fed to the meta-agent uses **causal forward-filtering**
(`HMMRegimeDetector.filtered_proba`), so there is no look-ahead bias.

```bash
python train_phase2.py    --ticker "^DJI" --train_start 2013-01-01 --train_end 2023-12-31
python evaluate_phase2.py --ticker "^DJI" --test_start 2024-01-01 --test_end 2024-12-31
python plot_phase2.py
```

`evaluate_phase2.py` re-runs the Phase 1 baselines on the same window, so the
output CSVs and plots contain a direct head-to-head: `meta_rl` vs. `hmm` vs.
`vol_regime` vs. `equal_weight` vs. buy-and-hold.

Useful flags for `train_phase2.py`:

| Flag | Meaning |
|------|---------|
| `--quick` / `--full` | 40k / 300k steps (default 150k) |
| `--lam_meta` | meta risk aversion (default 0.5) |
| `--alpha` / `--beta` | diversity-bonus / turnover-penalty weights |
| `--logit_scale` | softmax temperature for action → weights (default 2.0) |
| `--no_hmm` | ablation: train without regime observation |

## Train the whole system (end-to-end)

Run the phases in order — each consumes the artifacts of the previous one.
Defaults come from [config.py](config.py) (train `2013–2023`, test `2024`,
interval `1d`).

```bash
# 0. (optional) pre-train the CNN pattern extractor used in the 20-feature obs
python pretrain_cnn.py

# 1. Phase 0 — train the three frozen sub-agents → models/<name>_agent/best_model.zip
python train_agents.py --full          # or --quick for a fast loop

# 2. Phase 1 — fit the HMM + vol thresholds → models/phase1/
python train_phase1.py

# 3. Phase 2 — train the meta-RL allocator → models/phase2/
python train_phase2.py --full

# 4. Evaluate the full stack on the held-out test year and plot
python evaluate_phase2.py
python plot_phase2.py
```

**Walk-forward validation.** Repeat steps 1–4 for each split, pointing
`--outdir`/`--modeldir` at a per-split directory so models and CSVs don't
clobber each other:

```bash
# Split 1: train 2011–2021 → test 2022 (bear market)
python train_agents.py  --start 2011-01-01 --end 2021-12-31 --outdir split1 --full
python train_phase1.py  --train_start 2011-01-01 --train_end 2021-12-31 --outdir split1
python train_phase2.py  --train_start 2011-01-01 --train_end 2021-12-31 --modeldir split1 --outdir split1 --full
python evaluate_phase2.py --test_start 2022-01-01 --test_end 2022-12-31 --modeldir split1 --outdir split1
python plot_phase2.py   --datadir split1

# Split 2: train 2013–2023 → test 2024 (bull market) — config.py defaults
python train_agents.py  --outdir split2 --full
python train_phase1.py  --outdir split2
python train_phase2.py  --modeldir split2 --outdir split2 --full
python evaluate_phase2.py --modeldir split2 --outdir split2
python plot_phase2.py   --datadir split2
```

## Setup

```bash
pip install -r requirements.txt
```
