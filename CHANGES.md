# The August 2026 rework: what changed and why

> Historical record of the rework behind the current code. **For how to run the
> code, see [README.md](README.md).** The older results and plots mentioned here
> were removed from the working tree and remain in git history at commit `ebd3b23`.

Every change is tagged with the ID from the analysis doc
(*claude/repo-analysis-and-research-ideas.md* in the claude.ai project).
F = correctness fix, R = research feature.

## Correctness fixes

| ID | Change | Where |
|----|--------|-------|
| F1 | Reward credit assignment: the reward of step *t* is now `position_t × simple_ret_{t→t+1} − cost_t` — the action's payoff and its cost sit in the same reward (previously the PnL landed one step late, so the only action-dependent part of the immediate reward was the cost). PnL stream unchanged; learning signal much cleaner. | `env/trading_env.py step()` |
| F2 | "Recent performance" meta-features are now each sub-agent's realized shadow-book net returns (cost-inclusive, correctly time-aligned). Previously current proposals were paired with the *previous* bar's return. | `env/shadow.py`, `env/meta_env.py` |
| F3 | Frozen sub-agents observe their **own shadow book** (position / unreal PnL / time-in-trade / drawdown), not the blended portfolio's — they now run on-distribution. Applied in both Phase 1 (`run_allocator`) and Phase 2. | `env/shadow.py`, `TradingEnv.obs_with_book` |
| F4 | PnL compounds **simple** returns (log returns remain the basis of features). A permanently-full-long agent now exactly replicates buy-and-hold (verified by test); previously it lagged B&H by ≈ σ²/2 per year. | `env/trading_env.py` |
| F6 | Reward scale is stationary: the old `ret / own-downside-dev` scaling (orders-of-magnitude jumps, bad for SAC's entropy tuning) is gone. | reward rewrite (R4) |
| F8 | Warm evaluation windows: eval scripts load `WARM_BARS` (~1y) of history before `test_start`; features and the HMM forward filter are warm and **the full test year is traded** (259 bars vs ~190 before). Metrics start exactly at the test boundary. | `common/data.py`, all `evaluate_*.py` |
| F9 | Reproducibility: local CSV price cache (`download_data.py` → `data/prices.csv`) so every fold/seed/rerun uses identical data; eval cost defaults to the **training** cost read from the saved configs (they were mismatched between splits); each output dir gets its full config written next to it; `run_all.py` regenerates a whole split from one command. | `common/data.py`, `download_data.py`, `run_all.py`, config dumps |
| F10 | Model selection on validation **Sharpe** (deterministic rollout), not mean shaped reward over one episode. | `common/callbacks.py`, both train scripts |
| F11 | Random-start training episodes (`TRAIN_EPISODE_BARS`, default 504): each episode is a random ~2-year window instead of always replaying the identical full path — decorrelates the replay buffer, regularizes against single-path memorization. | `env/trading_env.py reset()` |
| F12 | `pretrain_cnn.py` defaults no longer overlap test years (was SPY-1h 2024→2025 — a leak if rerun); `max_hold` 390 → 60 so `time_in_trade` is informative on daily bars. | `pretrain_cnn.py`, `env/trading_env.py` |

## Research features

### R4 — a risk ladder that actually differentiates (`env/trading_env.py`, `config.py`)
Old reward divided returns by the agent's *own* downside deviation → roughly
scale-invariant in position size → aggressive/balanced/conservative all traded
~the same size (mean |pos| 0.59/0.43/0.43 in the committed runs).
New per-step reward (Ritter 2017 quadratic utility + the existing quadratic
drawdown zones):

    r = REWARD_SCALE · (net_ret − ½·lam_var·net_ret²) − dd_penalty

with per-agent `lam_var` (2 / 12 / 40), hard exposure caps `max_pos`
(1.0 / 0.6 / 0.3) and per-agent drawdown zones (dd_max 0.20 / 0.12 / 0.08).
Optimal size scales like μ/(lam_var·σ²), so λ now controls sizing directly;
the caps guarantee the ladder structurally.

### R1 — cash arm for the meta-agent (`env/meta_env.py`)
The meta-action is a softmax over **{aggressive, balanced, conservative, CASH}**
(`--no_cash` for the ablation). The blend is over *position* proposals
(action × per-agent cap); the cash weight simply scales gross exposure down.
The meta-agent can finally de-risk on its own instead of hoping the
conservative agent shorts.

### R2 — volatility-target overlay + honest benchmark
Executed position = blended × clip(`VOL_TARGET` / EWMA-vol, 0, `VOL_TARGET_CAP`)
(Moreira–Muir vol-managed sizing; `--vol_target 0` disables). The execution
env's `max_pos` widens to the cap so the overlay can mildly lever calm markets.
Every evaluation now also reports **vol_target_bh** — vol-targeted buy-and-hold —
which is the benchmark the thesis should be judged against, not plain B&H.

### R3 — Differential Sharpe Ratio meta-reward (`DifferentialSharpe` in `env/meta_env.py`)
Default meta reward = Moody–Saffell DSR of executed net returns
(+ diversity bonus − turnover penalty − drawdown-zone penalty).
`--meta_reward composite` keeps a mean–variance composite for the ablation
chapter. DSR has stable O(1) scale — fixes the SAC reward-scale problem at the
meta level.

### R7 — walk-forward protocol + honest statistics
* `walk_forward.py` — rolling folds (train `WF_TRAIN_YEARS`=8y → test 1y) over
  `WF_TEST_YEARS` × `WF_SEEDS`, each cell a full phase 0→1→2→eval pipeline in
  its own directory; resumable (finished cells are skipped).
* `evaluate_phase2.py` now writes `phase2_returns.csv` (dated per-step returns)
  per cell.
* `aggregate_results.py` — pools OOS returns across folds (seed-averaged),
  reports Sharpe with **stationary-bootstrap 95% CIs**, **Probabilistic Sharpe
  Ratio**, **Deflated Sharpe Ratio** (trials = every strategy×seed produced),
  per-fold Sharpe table, per-seed dispersion, and **paired bootstrap tests**
  of meta_rl vs equal-weight / vol-target-B&H / B&H, plus a pooled equity plot.
* `common/stats.py` — the statistics implementations.

## How to run

```bash
pip install -r requirements.txt
python download_data.py                  # once → data/prices.csv

# quick end-to-end sanity (~15 min CPU)
python run_all.py --outdir sanity --quick

# full single split
python run_all.py --outdir split2024 --full

# THE thesis experiment: 9 folds × 3 seeds (heavy — run overnight / on GPU box)
python walk_forward.py --outroot runs/wf --steps0 200000 --steps2 150000
python aggregate_results.py --root runs/wf

# ablations (same fold grid, different outroot):
python walk_forward.py --outroot runs/wf_nocash   --no_cash ...
python walk_forward.py --outroot runs/wf_novol    --vol_target 0 ...
python walk_forward.py --outroot runs/wf_composite --meta_reward composite ...
python walk_forward.py --outroot runs/wf_nohmm    --no_hmm ...
```

`tests/test_envs.py` (18 mechanical checks: reward timing, B&H equivalence,
caps, random starts, shadow books, DSR, cash arm, vol overlay) — run
`python tests/test_envs.py` after any env change.

## Compute budget hints
One fold/seed cell at 200k/150k steps ≈ 1.5–3 h on a laptop CPU
(set `OMP_NUM_THREADS=1` per worker and parallelize agents). 9×3 grid ≈ a few
days sequential — trim to 5 folds × 3 seeds or 9 × 2 if needed; keep at least
2022 and 2024 style years in every trimmed grid.

## Old results are not comparable
The committed CSVs/models predate these changes (and were themselves mixed
configs — see analysis doc F9). Regenerate everything through `run_all.py` /
`walk_forward.py` before citing numbers.

---

# November 2026: cross-fitted meta training and the overfitting check

Follows the diagnosis in *claude/results-diagnosis-2026-11.md* in the claude.ai project: agents
scored a Sharpe of 5 to 9 on their training data and around zero on test, and 94% of the
meta-agent's training days were days the agents had been fitted to.

| ID | Change | Where |
|----|--------|-------|
| R12 | **Cross-fitted meta training.** `--crossfit` refits agents and HMM on the bars before each block at the end of the training window, records what they propose on that block, and trains the meta-agent only on that stream. The last block is validation. | `train_crossfit.py`, `common/crossfit.py`, `env/meta_env.py`, `train_phase2.py` |
| R13 | **Paired leak ablation.** Each cross-fitted run also trains `meta_rl_leaked` on the same dates, budget and seed with the final agents' in-sample behaviour, and evaluates it on the same test path. | `train_phase2.py --stream_dates`, `evaluate_phase2.py` |
| R14 | **Random-start meta episodes.** F11 had only reached the agents, so the meta-agent replayed one fixed path. On by default for both meta-agents above; `--episode_bars` sets it anywhere. | `env/meta_env.py`, `train_phase2.py` |
| R15 | **Seed study for the ablation.** Retrains only the two meta-agents across seeds with agents, HMM and stream fixed. | `meta_seed_study.py` |
| F13 | **Overfitting check for every run.** `window_sharpe.csv` records each model's Sharpe on its training and validation data, and the dashboard shows it next to test. | `overfit_check.py`, `dashboard.py` |
| F14 | **Agent stability check.** Rolls every fold's agents and the final agents over the test year and compares their actions and Sharpe. | `agent_stability.py` |

Runs without `--crossfit` behave exactly as before, so older results stay comparable.

