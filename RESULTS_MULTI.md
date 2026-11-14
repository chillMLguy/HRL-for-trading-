# Multi-asset risk-budget hierarchy: results

Results of the multi-asset pipeline on branch `risk-budget-hierarchy`, built to the
implementation brief of 13 Nov 2026. Every number below comes from code in this repository
and can be regenerated with the command given next to it; anything marked *pending* has
not been run yet.

## Status

| Milestone | State |
|---|---|
| M1: data and core, no training | Done, reviewed |
| M2: specialists on the first fold | Done, awaiting review (below) |
| M3: allocator on the first fold, one whole fold | Not started |
| M4: walk-forward grid | Not started, runs only on request |

## M1: the core and the rule-based baselines

```bash
python download_multi.py --raw data/multi_raw.csv   # split the committed download
python report_core_multi.py --outdir runs/multi_core
```

All series are daily net excess returns over the T-bill rate, after 2 bp per unit traded.
These are whole-sample numbers (2007 to 2025). They include every future test year, so
nothing in the core may be changed because of them.

### Data

| Item | Value |
|---|---|
| Source | One `yf.download(tickers + ["^IRX"], period="max", auto_adjust=True)` call, saved as `data/multi_raw.csv` on 14 Nov 2026 |
| First common date | 2006-02-06 (DBC's first trading day on Yahoo; all other ETFs start earlier) |
| First decision date | 2007-04-18 (first common date + 300 bars; features are warm after 272) |
| Returns reported | 2007-04-19 to 2025-12-31, 4,707 days |
| First feasible test year | 2015, with 7.71 years of training data (2016 onward: 8.00) |
| Dates lost to misalignment | None: every date after 2006-02-06 has all eight prices |
| Risk-free rate | `^IRX`; three bond-market holidays (e.g. 2006-10-09) forward-filled; seven slightly negative yields in March 2020 (minimum −0.105%) kept as published; mean 1.41% a year over the reported period |

Sanity checks on the data: the largest one-day moves match known events (SPY +14.5% on
2008-10-13 and −10.9% on 2020-03-16, GLD −8.8% on 2013-04-15). IEF's mean excess return of
about 2% a year is only possible with distributions included, so the prices are adjusted.

### Summary per portfolio (`core_metrics.csv`)

| Portfolio | Sharpe | 95% interval | Return % | Vol % | Max DD % | Turnover / yr | Cost % / yr | Gross mean | Gross max | Vol cap days % | Lev cap days % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `core` | 0.54 | [0.07, 0.99] | 5.64 | 10.42 | 23.0 | 10.3 | 0.21 | 1.43 | 2.50 | 0 | 2.57 |
| `core_trend` | 0.57 | [0.13, 1.04] | 7.47 | 13.10 | 25.0 | 22.8 | 0.46 | 1.73 | 2.50 | 41.96 | 18.59 |
| `equal_weight` | 0.43 | [-0.02, 0.87] | 4.46 | 10.29 | 23.5 | 4.9 | 0.10 | 1.12 | 2.50 | 0 | 0.36 |
| `sixty_forty` | 0.75 | [0.30, 1.20] | 7.69 | 10.30 | 22.8 | 5.4 | 0.11 | 1.24 | 2.50 | 0 | 2.91 |
| `spy_vt` | 0.64 | [0.20, 1.07] | 6.56 | 10.34 | 24.8 | 3.4 | 0.07 | 0.72 | 1.86 | 0 | 0 |
| `spy` | 0.54 | [0.14, 0.97] | 10.64 | 19.85 | 56.1 | 0.1 | 0.00 | 1.00 | 1.00 | – | – |

Return and volatility are annualised; return is the arithmetic mean of daily excess returns.
Intervals are stationary-bootstrap 95% intervals (`common.stats.bootstrap_sharpe_ci`).

### The core's sleeves (`core_sleeves.csv`)

| Asset | Mean vol % | Days at vol floor % | Days at B_MAX % | Mean position | Max position |
|---|---|---|---|---|---|
| SPY | 16.8 | 0 | 0 | 0.17 | 0.45 |
| EFA | 18.5 | 0 | 0 | 0.15 | 0.41 |
| EEM | 23.4 | 0 | 0 | 0.12 | 0.27 |
| IEF | 6.6 | 0 | 2.25 | 0.39 | 0.84 |
| TLT | 14.4 | 0 | 0 | 0.17 | 0.40 |
| GLD | 16.2 | 0 | 0 | 0.16 | 0.38 |
| DBC | 17.7 | 0 | 0 | 0.15 | 0.38 |
| VNQ | 23.1 | 0 | 0 | 0.14 | 0.35 |

The core's scale `k` averaged 1.81 (range 1.24 to 3.75).

### Sharpe per year (`core_yearly_sharpe.csv`)

2007 covers 19 April to 31 December only.

| Year | `core` | `core_trend` | `equal_weight` | `sixty_forty` | `spy_vt` | `spy` |
|---|---|---|---|---|---|---|
| 2007 | 0.97 | 1.06 | 0.49 | -0.06 | -0.24 | -0.10 |
| 2008 | -0.83 | -0.43 | -1.00 | -1.25 | -1.35 | -0.94 |
| 2009 | 1.20 | 0.26 | 1.41 | 1.32 | 1.36 | 1.01 |
| 2010 | 1.91 | 1.60 | 1.37 | 1.42 | 0.97 | 0.87 |
| 2011 | 0.89 | 0.56 | 0.40 | 0.60 | -0.04 | 0.20 |
| 2012 | 1.16 | 1.07 | 1.11 | 1.24 | 0.82 | 1.23 |
| 2013 | -0.32 | 0.49 | -0.32 | 2.06 | 2.35 | 2.58 |
| 2014 | 0.75 | 1.26 | 0.52 | 1.53 | 0.78 | 1.18 |
| 2015 | -0.99 | -0.47 | -1.01 | -0.11 | -0.21 | 0.15 |
| 2016 | 0.94 | 0.01 | 0.76 | 0.77 | 0.74 | 0.91 |
| 2017 | 2.36 | 2.28 | 2.36 | 3.10 | 2.65 | 2.83 |
| 2018 | -0.92 | -1.32 | -1.08 | -0.48 | -0.22 | -0.30 |
| 2019 | 2.76 | 2.54 | 2.45 | 2.30 | 1.66 | 2.08 |
| 2020 | 0.60 | 0.85 | 0.30 | 0.72 | 0.36 | 0.66 |
| 2021 | 0.82 | 0.56 | 1.20 | 1.54 | 1.66 | 2.01 |
| 2022 | -1.50 | -1.15 | -1.34 | -1.52 | -1.02 | -0.79 |
| 2023 | 0.29 | -0.03 | 0.29 | 1.00 | 1.27 | 1.48 |
| 2024 | 0.13 | 0.11 | 0.17 | 0.80 | 1.23 | 1.44 |
| 2025 | 1.32 | 1.44 | 1.46 | 0.58 | 0.36 | 0.73 |

### Findings

**1. The core is a sound benchmark.** Realised volatility is 10.42% against the 10% target,
and between 8.8% (2009) and 12.0% (2008) in every calendar year. The volatility cap never
binds. The leverage cap binds on 2.6% of days, all in calm years (2014, 2017, 2018, 2020).
`B_MAX` binds only for IEF (2.3% of days) and the volatility floor never. Turnover is about
10 times capital a year, costing 0.21% a year. The core's Sharpe of 0.54 [0.07, 0.99] matches
SPY's with less than half its maximum drawdown (23% against 56%).

**2. The core's weak spot is the onset of a crash.** In September and October 2008 the core
ran at 17.7% and 19.8% realised volatility. Two things lifted its scale `k` to its maximum of
3.75 on 2008-09-22. Correlations between the eight assets had collapsed to an average of
−0.03 (SPY and TLT at −0.70), and the 60-day covariance lagged the 33-day volatility by
15 to 20% for most assets. The volatility cap did not catch it because it uses the same
lagging covariance. This is the textbook risk-parity behaviour in a correlation break. The
brief fixes these parameters and these are whole-sample numbers, so this is reported and not
tuned. A faster or range-based risk estimate could only be studied as a separate ablation.

**3. The simple trend rule leaves the specialists no visible room in this sample.**
`core_trend` beats `core` by 0.03 in Sharpe (bootstrap p = 0.41). Its active Sharpe against
the core is 0.28 with an interval of [−0.14, 0.73], which includes zero. The trend tilt
averages +0.29 and is positive on 77% of days, so `core_trend` mostly runs more risk than the
core (13.1% volatility). A hindsight check holds each asset's average tilt constant. It
uses future data and is a diagnostic only. It reaches a Sharpe of 0.563 and leaves
`core_trend` an active Sharpe of −0.003 against it. In 2007–2025 the 12-month trend rule's
edge over the core is therefore average exposure, not timing. The specialists are paid for
timing only, so this sets expectations: a "no" for layer 1 is the likely outcome, and it
would be a clean result. Numbers: `core_diagnostics.json`.

**4. The caps shape the trend portfolio.** For `core_trend` the volatility cap binds on 42%
of days and the leverage cap on 19%, scaling positions to 0.84 on average on those days. A
broad upward tilt hits the caps, and then a specialist's reward, computed for its sleeve
alone, no longer equals its effect inside the portfolio. Specialist tilts should average
close to zero by construction, so they should bind less often. M2 will report how often the
caps bind for `core_spec`.

**5. A volatility-targeted 60/40 is the strongest simple portfolio here.** Its Sharpe of 0.75
beats the core's 0.54 in both halves of the sample (2007–14: 0.82 against 0.64; 2015–25: 0.70
against 0.47). In this US-led period EFA, EEM, DBC and VNQ had poor risk-adjusted returns;
DBC earned 1.0% a year over the T-bill rate at 19% volatility. For the thesis this means "a
learned layer beats the core" would not mean "beats the best simple portfolio", which is why
`sixty_forty` and `spy_vt` are in the grid. The allocator cannot earn this gap by
permanently favouring US assets, because the reward removes the payoff of a constant action
(principle 2).

**6. Two expected effects show up.** Sizing by risk beats sizing by capital (`core` 0.54
against `equal_weight` 0.43), and volatility targeting improves SPY (0.64 against 0.54).

## Feature check: high–low range and volume (pre-registered)

Decided with the owner on 14 Nov 2026 after M1, and written here and committed **before the
check was run**. It is a deviation from the brief: section 8 does not list feature changes
among the settings that may differ from the defaults.

**Candidate features**, computed per asset and day from `data/multi_ohlcv.csv`, causal, clipped:

- `range_vol_ratio` = ln( sqrt(mean of PK² over the last 10 days) / sqrt(mean of PK² over the
  last 60 days) ), clipped to [−2, 2], where PK² = (ln(High/Low))² / (4 ln 2) is the Parkinson
  daily variance. It is the high–low analogue of feature 6.
- `abnormal_volume` = ln( mean volume over the last 5 days / mean volume over the last 60
  days ), clipped to [−2, 2].

**Method.** Data: the training window of test year 2015 only (decisions from 2007-04-18
to 2014-12-30, every return inside the window). Target: the untilted sleeve's excess return
`s = b · Rex` summed over the next 5 days, and over the next day as a secondary check, with
each asset's mean over the fitted part removed (as `mu_s` does in the specialist reward).
Model: ridge regression with `alpha = 1` on standardised features, pooled over the eight
assets. It is fitted on the cross-fit structure the specialists use (`fold_edges(n, 4, 0.4)`):
for each block, fitted on the decisions before it, with the last 5 decisions before the
block left out, and predicting the block. Score: the correlation of predictions and outcomes
over all block days and assets. Three feature sets: features 0–11 (base), base +
`range_vol_ratio`, base + `abnormal_volume`. The uncertainty of each improvement over base
comes from a paired stationary bootstrap over dates (mean block 20 days, 2,000 draws).

**Decision rule.** A candidate feature joins the specialist observation in one extra
configuration of the layer-1 settings search only if its improvement over base, for the
5-day target, has a 95% interval entirely above zero. If neither passes, no feature
configuration is tried and the features are dropped for good. If both pass, the extra
configuration has both.

**Result** (`python feature_check_multi.py`, files in `runs/multi_wf/fold2015/feature_check/`).
Blocks start 2010-05-17, 2011-07-12, 2012-09-05 and 2013-11-01, with 1,161 block days for the
5-day target.

| Target | Features | Out-of-sample correlation | Change vs base | 95% interval |
|---|---|---|---|---|
| 5 days | base (0–11) | 0.0178 | | |
| 5 days | base + `range_vol_ratio` | 0.0188 | +0.0010 | [−0.0074, +0.0085] |
| 5 days | base + `abnormal_volume` | 0.0147 | −0.0031 | [−0.0099, +0.0034] |
| 1 day | base (0–11) | 0.0330 | | |
| 1 day | base + `range_vol_ratio` | 0.0336 | +0.0007 | [−0.0022, +0.0037] |
| 1 day | base + `abnormal_volume` | 0.0317 | −0.0013 | [−0.0029, +0.0002] |

Neither feature passes, so **no feature configuration is tried and both are dropped**. The
range adds nothing measurable once features 6 and 7 (the close-to-close volatility ratio
and level) are present, and volume slightly hurts. The base features themselves predict
little: a correlation of 0.018 with the next 5 days' sleeve returns and 0.033 with the next
day's. That matches the single-asset finding in section 2 of the brief (about 0.05) and is
the ceiling the specialists work under.

## M2: the specialists on the first fold (test year 2015)

```bash
python search_multi.py --layer 1          # default stream + 4 variants, freezes settings.json
python train_specialists.py --outdir runs/multi_wf/fold2015 --test_year 2015
python report_spec_multi.py               # writes runs/multi_wf/fold2015/m2/
```

Training window of 2015: decisions 2007-04-18 to 2014-12-30 (1,941). Cross-fit blocks
(out of sample): 2010-05-17 to 2014-12-30, 1,165 days in four blocks of about 291 days. Each
block's three specialists were fitted only on data before it. Nothing here touches 2015.

**Reward scale check.** Under uniformly random tilts on the fit window, the reward has a
standard deviation of 0.38 (0.38–0.39 for λ = 5, 10, 20), inside the brief's 0.1–3. So
`SPEC_REWARD_SCALE` stays at 100.

### Active Sharpe against the core (`m2/active_sharpe.csv`)

Net excess returns after costs. "Timing only" holds each asset's average tilt constant and
measures what is left. It uses the period's own average tilt (hindsight), is a diagnostic
only, and never enters a decision.

| Layer | Where | Days | Active Sharpe | 95% interval | Timing only | Mean tilt |
|---|---|---|---|---|---|---|
| Specialists | cross-fit blocks (out of sample) | 1,165 | **0.31** | [−0.54, 1.22] | 0.39 | 0.01 |
| Trend rule | same days | 1,165 | 0.89 | [−0.09, 1.91] | −0.15 | 0.39 |
| Specialists | block 1 (2010-05 to 2011-07) | 291 | 1.45 | [0.03, 2.98] | 0.87 | 0.10 |
| Specialists | block 2 (2011-07 to 2012-09) | 291 | 1.03 | [−0.64, 2.99] | 1.08 | 0.01 |
| Specialists | block 3 (2012-09 to 2013-10) | 291 | −0.56 | [−2.04, 1.08] | 0.75 | 0.07 |
| Specialists | block 4 (2013-11 to 2014-12) | 292 | −1.01 | [−2.34, 0.54] | −0.20 | −0.12 |
| Specialists | fit window, fold 1 (in sample) | 775 | 8.67 | [7.33, 10.50] | 8.67 | 0.09 |
| Specialists | fit window, fold 2 (in sample) | 1,066 | 7.91 | [6.71, 9.15] | 8.02 | 0.02 |
| Specialists | fit window, fold 3 (in sample) | 1,357 | 6.72 | [5.73, 7.72] | 6.90 | −0.04 |
| Specialists | fit window, fold 4 (in sample) | 1,648 | 5.95 | [5.23, 6.65] | 6.01 | −0.04 |
| Specialists | fit window, final ensemble (in sample) | 1,941 | 5.99 | [5.19, 6.74] | 6.04 | 0.03 |
| Trend rule | same window | 1,941 | 0.41 | [−0.22, 1.08] | −0.04 | 0.34 |

Costs on the cross-fit blocks: `core_spec` turns over 123 times capital a year against 10.7
for the core, and pays 2.46% a year in costs against 0.21%. **Before costs** the specialists'
active Sharpe is 0.75 [−0.13, 1.69] (active return 3.8% a year at 5.1% tracking error);
after costs it is 0.31.

### Specialist diagnostics on the cross-fit blocks (`m2/spec_diagnostics.csv`)

| Asset | Mean tilt | Mean trend tilt | Days with abs(tilt) > 0.95 % | Lag-1 autocorr | Corr with trend tilt | Mean corr between seeds | Corr fold vs final |
|---|---|---|---|---|---|---|---|
| SPY | −0.19 | 0.80 | 5.4 | 0.67 | 0.06 | 0.40 | 0.49 |
| EFA | −0.14 | 0.45 | 9.4 | 0.58 | −0.52 | 0.47 | 0.62 |
| EEM | 0.02 | 0.23 | 4.8 | 0.53 | −0.14 | 0.45 | 0.57 |
| IEF | 0.16 | 0.55 | 6.4 | 0.66 | −0.34 | 0.46 | 0.50 |
| TLT | 0.05 | 0.33 | 8.2 | 0.72 | −0.42 | 0.44 | 0.49 |
| GLD | 0.15 | 0.11 | 5.5 | 0.73 | −0.02 | 0.41 | 0.33 |
| DBC | 0.12 | −0.08 | 4.2 | 0.67 | −0.23 | 0.41 | 0.46 |
| VNQ | −0.07 | 0.74 | 9.9 | 0.71 | 0.16 | 0.51 | 0.60 |

"Corr fold vs final" compares the stream's tilts with the final ensemble's on the same days,
which the final ensemble was trained on; the test-year version comes with M3. The caps bind
for `core_spec` on 10.8% (volatility) and 4.2% (leverage) of the block days, against 39.4% and
24.1% for `core_trend`.

Training curve, mean over seeds (`m2/training_curves.csv`): the fit-window active Sharpe
rises steadily through training, without levelling off: 2.0 / 4.2 / 5.2 / 6.4 at 6k / 18k /
30k / 60k steps for the fold models, and 1.2 / 2.4 / 3.3 / 4.6 per single final seed.

**Training time:** 7.0 minutes per 60,000-step model (422 s on average over 41 runs, two
running in parallel on a 2-core machine), 3.6 minutes per 30,000 and 14.1 minutes per 120,000.

### Findings

**1. No evidence that the specialists time the assets.** Out of sample the active Sharpe is
0.31 with an interval of [−0.54, 1.22]. Their average tilt is close to zero (0.01) as the
reward intends, so this number is timing, not exposure. By block it falls from 1.45 and 1.03
in 2010–12 to −0.56 and −1.01 in 2012–14. Each block's interval is about ±1.5 wide, so that
decline is not reliable either.

**2. They memorise their training data, as the old agents did.** The active Sharpe on their
own fit windows is 5.9 to 8.7, against 0.31 out of sample. It shrinks as the fit window grows
(8.7 on 775 days, 5.9 on 1,648), and it keeps rising with training steps. Sharing one network
across assets, averaging seeds, and making every input dimensionless did not prevent it. The
features carry very little to learn: the pre-registered check found a correlation of 0.018
between features 0–11 and the next 5 days' sleeve returns.

**3. Trading costs eat more than half of what there is.** The tilt changes by 0.33 a day on
average (lag-1 autocorrelation 0.53–0.73), so `core_spec` trades about 12 times as much as the
core. The reward does charge for the extra trading, but 2 bp per unit is small next to the
noise in the active return.

**4. The policies are unstable.** Seeds trained on the same data correlate 0.40–0.51, and fold
tilts against final tilts 0.33–0.62. That is barely better than the old system's 0.33.

**5. The trend rule is not a timing benchmark on these days either.** Its active Sharpe of
0.89 on the blocks becomes −0.15 once its average tilt (+0.39) is held constant. It comes from
holding more risk, as in M1. Pre-registered comparison 2 (`core_spec` against `core_trend`) is
therefore partly a comparison of timing with exposure. I will report the timing-only
diagnostic next to it, without changing the comparison.

## Settings search (brief section 8)

Layer 1, first fold only, scored by the cross-fitted active Sharpe of `core_spec` against
`core` on the blocks above (`runs/multi_wf/fold2015/search_layer1.csv`):

| Configuration | Steps | λ | Active Sharpe | 95% interval | Timing only | Mean tilt |
|---|---|---|---|---|---|---|
| default | 60,000 | 10 | 0.309 | [−0.544, 1.220] | 0.392 | 0.013 |
| lambda_5 | 60,000 | 5 | 0.078 | [−0.828, 1.007] | 0.211 | −0.027 |
| lambda_20 | 60,000 | 20 | 0.142 | [−0.756, 1.100] | 0.267 | −0.015 |
| steps_30k | 30,000 | 10 | 0.307 | [−0.540, 1.238] | 0.349 | 0.045 |
| steps_120k | 120,000 | 10 | 0.379 | [−0.544, 1.320] | 0.441 | 0.016 |

The best change (120,000 steps) beats the defaults by 0.07, under the 0.1 margin, so **the
defaults are kept** and frozen in `runs/multi_wf/settings.json` for every fold. Every interval
spans about −0.8 to +1.3: the differences between configurations are noise. Configurations
tried so far: 5 for layer 1, plus the feature check, which ran no configuration. Layer 2
follows in M3.

## Deviations from the brief and interpretations

| # | Where | What and why |
|---|---|---|
| 1 | `report_core_multi.py` | New script, not in the brief's file table. M1 asks for a whole-sample report of the rule-based portfolios before any learning exists, and `evaluate_multi.py` is defined per test year. It reuses `multi/portfolio.py`, which `evaluate_multi.py` will use too. |
| 2 | `multi/portfolio.py` | `spy` (buy and hold) is the one portfolio that does not go through `build_positions`: it is a constant weight of 1 in `SPY`, unscaled and uncapped by definition. Passing it through step 4 would cut it whenever `SPY` volatility exceeds `VOL_CAP`, and it would no longer be buy and hold. It uses the same return and cost function as every other portfolio. |
| 3 | `multi/data.py` | Dating convention: return rows are dated by the day the return is realised (the brief's `t+1`). In `multi_returns.csv` the `date` is the close at which the return lands, so test year Y holds the returns realised in Y, and its first position is decided at the last close of Y−1. |
| 4 | `multi/features.py` | Feature 7 "median of `vol` over the previous 252 days" is read as the 252 days before t, excluding t. |
| 5 | `multi/features.py` | EWMA std and covariance use pandas' defaults (`adjust=True`, bias-corrected, around the EWMA mean), as `ewma_vol_series` in `env/meta_env.py` does. |
| 6 | `config.py` | Constants beyond section 14 were added to the new block so the modules hold no magic numbers: feature horizons, windows and clips, the `1e-6` guard in `k`, `SPEC_INFER_LEADIN = 20`, `MULTI_MIN_TRAIN_YEARS = 7.5`, quick-preset values, file paths. None changes a value the brief sets. |
| 7 | `download_multi.py` | The download is one `yf.download(..., period="max", auto_adjust=True)` call for the eight ETFs and `^IRX` together, saved unchanged as `data/multi_raw.csv` and then split into `multi_prices.csv` and `riskfree.csv`. Per-ticker requests with a start date were throttled by Yahoo on the owner's machine; the single call was not. The brief's start date (2005-01-01) now only bounds what the loaders read. |
| 8 | `download_multi.py` | `data/multi_ohlcv.csv` (open, high, low, close, volume) is cached from the same download. No model reads it. It is kept for a possible test of range and volume features, which is not decided and would itself be a deviation (section 8 does not list feature changes among the allowed settings). |
| 9 | `report_core_multi.py` | `core_diagnostics.json` adds a hindsight check (each asset's average trend tilt held constant) to separate timing from average exposure in `core_trend`. It uses future data, is labelled as a diagnostic, and never enters a comparison. |
| 10 | new scripts | `feature_check_multi.py` (pre-registered feature check), `search_multi.py` (settings search, which writes `settings.json`) and `report_spec_multi.py` (M2 report) are not in the brief's file table. They implement steps the brief asks for (sections 8 and 12) and reuse `multi/` and `train_crossfit_multi.py`. |
| 11 | `multi/spec_env.py` | Mode `separate` raises a clear error. The brief marks it as an ablation run only on request; it will be built when requested. |
| 12 | `multi/data.py`, `train_crossfit_multi.py` | Fit windows end strictly before what they are evaluated on: a fold's last training reward uses the return realised at the bar before its block starts. The stream and the final fit stop at the last decision whose return is realised inside the training window, so the stream ends on 2014-12-30, not 2014-12-31. |
| 13 | `search_multi.py`, `report_spec_multi.py` | A timing-only active Sharpe (average tilt held constant, hindsight) is reported next to the brief's active Sharpe. It never enters a decision; the search uses the brief's criterion. |
| 14 | `report_spec_multi.py` | The correlation of fold and final specialists is computed on the cross-fit blocks in M2 (where the final ensemble is in sample). The brief's test-year version (section 8) comes with the M3 evaluation. |
