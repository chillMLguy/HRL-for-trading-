# Multi-asset risk-budget hierarchy: results

Results of the multi-asset pipeline on branch `risk-budget-hierarchy`, built to the
implementation brief of 13 Nov 2026. Every number below comes from code in this repository;
anything marked *pending* has not been run yet.

## Status

| Milestone | State |
|---|---|
| M1: data and core, no training | Code and tests done. Real-data report pending: `data/multi_prices.csv` and `data/riskfree.csv` have to be downloaded with `python download_multi.py` on a machine that can reach Yahoo Finance |
| M2: specialists on the first fold | Not started |
| M3: allocator on the first fold, one whole fold | Not started |
| M4: walk-forward grid | Not started, runs only on request |

## M1: the core and the rule-based baselines

*Pending data.* Run `python report_core_multi.py --outdir runs/multi_core`; it writes the
tables below to that folder.

- First common date, first decision date, first feasible test year: *pending*
- Summary per portfolio (`core_metrics.csv`): Sharpe with 95% interval, annual return,
  volatility, maximum drawdown, turnover, cost, mean and maximum gross exposure, share of
  days each cap binds: *pending*
- Core sleeves (`core_sleeves.csv`): days at the vol floor and at `B_MAX`, mean and
  maximum position per asset: *pending*
- Sharpe per year (`core_yearly_sharpe.csv`): *pending*

## Settings search (brief section 8)

No configuration tried yet. Every configuration and its cross-fitted score will be listed
here, because the number tried enters the Deflated Sharpe.

## Deviations from the brief and interpretations

| # | Where | What and why |
|---|---|---|
| 1 | `report_core_multi.py` | New script, not in the brief's file table. M1 asks for a whole-sample report of the rule-based portfolios before any learning exists, and `evaluate_multi.py` is defined per test year. It reuses `multi/portfolio.py`, which `evaluate_multi.py` will use too. |
| 2 | `multi/portfolio.py` | `spy` (buy and hold) is the one portfolio that does not go through `build_positions`: it is a constant weight of 1 in `SPY`, unscaled and uncapped by definition. Passing it through step 4 would cut it whenever `SPY` volatility exceeds `VOL_CAP`, and it would no longer be buy and hold. It uses the same return and cost function as every other portfolio. |
| 3 | `multi/data.py` | Dating convention: return rows are dated by the day the return is realised (the brief's `t+1`). In `multi_returns.csv` the `date` is the close at which the return lands, so test year Y holds the returns realised in Y, and its first position is decided at the last close of Y−1. |
| 4 | `multi/features.py` | Feature 7 "median of `vol` over the previous 252 days" is read as the 252 days before t, excluding t. |
| 5 | `multi/features.py` | EWMA std and covariance use pandas' defaults (`adjust=True`, bias-corrected, around the EWMA mean), as `ewma_vol_series` in `env/meta_env.py` does. |
| 6 | `config.py` | Constants beyond section 14 were added to the new block so the modules hold no magic numbers: feature horizons, windows and clips, the `1e-6` guard in `k`, `SPEC_INFER_LEADIN = 20`, `MULTI_MIN_TRAIN_YEARS = 7.5`, quick-preset values. None changes a value the brief sets. |
