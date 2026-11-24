# ── Asset & data ─────────────────────────
# Branch spy-vix: the traded asset is the SPY ETF (adjusted closes, so dividends
# are in the returns) and the agents also see the VIX index.  The ^DJI runs of
# earlier branches used TICKER = "^DJI" with data/prices.csv (still in data/).
import os

TICKER     = "SPY"
VIX_TICKER = "^VIX"
INTERVAL   = "1d"

# Local data cache: Date + one column per ticker (SPY, ^VIX).  All scripts read
# prices and VIX from this file instead of hitting yfinance — one download,
# identical data for every fold/seed/rerun.  Create it with:
#     python download_data.py
DATA_CSV = "data/spy_vix.csv"
DOWNLOAD_START = "2005-01-01"

# ── VIX inputs ───────────────────────────
# Four VIX features are appended to every agent's observation and to the
# meta-agent's (env/trading_env.py).  USE_VIX = False keeps the observation size
# but fills the VIX features with zeros (the "no VIX" ablation).
# run_all.py / walk_forward.py --no_vix and --vix_lag set the two environment
# variables below, so every script they start sees the same setting.  To run a
# single script on a no-VIX run:  HRL_USE_VIX=0 python evaluate_phase2.py ...
USE_VIX        = os.environ.get("HRL_USE_VIX", "1") != "0"
# The VIX close is set at 4:15 pm ET, 15 minutes after SPY's 4:00 pm close, so a
# decision at SPY's close uses the VIX close of the previous day by default
# (strictly no look-ahead).  Set 0 to use the same-day VIX close.
VIX_LAG_DAYS   = int(os.environ.get("HRL_VIX_LAG", "1"))
VIX_REF_LEVEL  = 20.0      # vix_level = ln(VIX / 20)
VIX_REL_WINDOW = 252       # vix_rel   = ln(VIX / median of the previous 252 days)
VIX_CHG_DAYS   = 5         # vix_chg   = ln(VIX / VIX 5 days earlier)

# ── Train / test splits (single-split runs) ──
TRAIN_START = "2013-01-01"
TRAIN_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2024-12-31"

# Warm history prepended to evaluation windows (F8): features / HMM filter are
# computed over [test_start − WARM_BARS, test_end] but trading + metrics start
# exactly at test_start.  ~1 trading year by default.
WARM_BARS = 260

# ── Execution & training ──────────────
# One-way transaction cost per unit of position traded (must match train & eval —
# enforced).  0.2 bp: SPY's bid-ask spread is about 1 cent on a $400-600 price
# (~0.2 bp, so ~0.1 bp per side) plus a broker commission of a similar size.
# The ^DJI runs used 0.0002 (2 bp).
COST_PCT = 0.00002
SEED     = 42

# Random-start training episodes (F11): each training episode is a random
# window of this many bars (0 = full-path episodes, old behaviour).
TRAIN_EPISODE_BARS = 504          # ~2 trading years

# ── Sub-agent risk ladder (F5/R4) ────────
# Reward:  r = REWARD_SCALE·(net_ret − ½·lam_var·net_ret²) − dd_penalty
# (Ritter-2017 quadratic utility; optimal size ≈ μ/(lam_var·σ²) → real ladder)
# max_pos is a hard exposure cap; dd_free/dd_max are per-agent drawdown zones.
REWARD_SCALE = 100.0
AGENT_PRESETS = {
    "aggressive":   dict(lam_var=2.0,  max_pos=1.0, dd_free=0.04, dd_max=0.20),
    "balanced":     dict(lam_var=12.0, max_pos=0.6, dd_free=0.03, dd_max=0.12),
    "conservative": dict(lam_var=40.0, max_pos=0.3, dd_free=0.02, dd_max=0.08),
}

# ── Meta layer (R1/R2/R3) ────────────────
META_USE_CASH   = True     # R1: 4th softmax arm = cash
META_REWARD     = "dsr"    # "dsr" (R3, default) or "composite" (old reward, ablation)
DSR_ETA         = 1.0 / 60.0
VOL_TARGET      = 0.10     # R2: annualized vol target for the position overlay (0 = off)
VOL_TARGET_CAP  = 1.5      # max leverage the overlay may apply
VOL_EWMA_SPAN   = 33       # EWMA span for realized-vol estimate (~RiskMetrics 0.94)

# Agents fit on this share of their window and select on the rest (train_agents.py)
AGENT_TRAIN_RATIO = 0.8

# ── Cross-fitted meta training ───────────
# Agents and HMM are refit CF_FOLDS times, each time on the bars before one block
# at the end of the training window, so the meta-agent only ever sees behaviour on
# unseen data.  See common/crossfit.py.
CF_FOLDS          = 4
CF_MIN_TRAIN_FRAC = 0.5    # share of the window that fold 1 fits on
META_TRAIN_EPISODE_BARS = 252   # random-start meta episodes, ~1 trading year

# ── HMM ─────────────────────────
N_STATES     = 3
HMM_EMBARGO  = 5      # drop last N training obs to reduce
                      # train→test contamination
HMM_VAL_FRAC = 0.05   # last X% of (training − embargo) is the
                      # validation slice for log-likelihood reporting

# ── Walk-forward protocol (R7) ───────────
WF_TEST_YEARS  = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
WF_TRAIN_YEARS = 8      # each fold trains on the 8 years before its test year
WF_SEEDS       = [0, 1, 2]

# ── Annualization ───────────────
BARS_PER_YEAR = {"1d": 252, "1h": 1638, "30m": 3276, "15m": 6552}
