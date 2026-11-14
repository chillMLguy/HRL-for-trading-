# ── Asset & data ─────────────────────────
TICKER   = "^DJI"
INTERVAL = "1d"

# Optional local price cache (CSV with Date,Close columns).  When set (or when
# --data_csv is passed), all scripts read prices from this file instead of
# hitting yfinance — one download, identical data for every fold/seed/rerun.
# Create it with:  python download_data.py
DATA_CSV = "data/prices.csv"

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
COST_PCT = 0.0002     # one-way transaction cost (must match train & eval — enforced)
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


# ── Multi-asset risk-budget hierarchy ──
# Constants of the multi-asset pipeline (package multi/, scripts *_multi.py,
# train_specialists.py, train_allocator.py).  Nothing above this line is used
# differently by it, and nothing in the single-asset pipeline reads this block.
# Values come from the implementation brief (Nov 2026), section 14.  The core's
# parameters are fixed by design: report how they behave, never tune them on results.
import math as _math

# Data
MULTI_TICKERS        = ["SPY", "EFA", "EEM", "IEF", "TLT", "GLD", "DBC", "VNQ"]
MULTI_DATA_CSV       = "data/multi_prices.csv"   # Date + one adjusted close per ticker
RISKFREE_TICKER      = "^IRX"                    # 13-week T-bill yield, % per year
RISKFREE_CSV         = "data/riskfree.csv"
MULTI_DOWNLOAD_START = "2005-01-01"
MULTI_WARM_BARS      = 300      # feature history before any window, never traded
MULTI_BARS_PER_YEAR  = 252

# Core portfolio (brief section 5)
MULTI_VOL_FLOOR        = 0.02   # floor on annualised asset vol (VOL_EWMA_SPAN above)
MULTI_VOL_MIN_PERIODS  = 20
MULTI_COV_SPAN         = 60
MULTI_COV_MIN_PERIODS  = 60
SLEEVE_VOL             = 0.10   # each sleeve sized to this vol
B_MAX                  = 3.0    # maximum sleeve size, fraction of capital
PORT_VOL               = 0.10   # core portfolio volatility target
VOL_CAP                = 0.15   # ex-ante vol cap, every portfolio
LEV_MAX                = 2.5    # gross exposure cap, every portfolio
MULTI_BASE_BUDGETS     = [1.0 / 8] * 8
MULTI_VCORE_EPS        = 1e-6   # numerical guard in k = PORT_VOL / max(v_core, eps)

# Features (brief section 6)
MULTI_TREND_HORIZONS    = (21, 63, 126, 252)   # features 0-3
MULTI_REVERSAL_HORIZONS = (1, 5)               # features 4-5
MULTI_Z_CLIP            = 3.0
MULTI_VOLRATIO_WINDOWS  = (10, 60)             # feature 6
MULTI_LOG_CLIP          = 2.0                  # features 6-8
MULTI_VOL_LEVEL_WINDOW  = 252                  # feature 7: median of the previous 252 days
MULTI_CORR_WINDOW       = 60                   # features 9-10
MULTI_BREADTH_HORIZON   = 126                  # feature 11
MULTI_TREND_RULE_HORIZON = 252                 # core_trend: m = 1 + clip(z(252), -1, 1)
MULTI_TREND_RULE_CLIP   = 1.0
MULTI_N_MARKET_FEATURES = 12                   # features 0-11; 12 = own previous tilt

# Specialists (brief section 6)
SPEC_MODE          = "shared_id"     # shared_id | shared | separate
SPEC_EPISODE_BARS  = 252
SPEC_REWARD_SCALE  = 100.0
SPEC_LAMBDA        = 10.0
SPEC_STEPS         = 60_000
SPEC_STEPS_QUICK   = 15_000
SPEC_SEEDS         = 5
SPEC_SEEDS_QUICK   = 2
SPEC_INFER_LEADIN  = 20              # inference starts this many bars early, previous tilt 0

# Allocator (brief section 7)
ALLOC_STEP           = 5             # trading days per decision
ALLOC_KAPPA          = _math.log(1.5)
ALLOC_G_RANGE        = 0.25
ALLOC_REWARD_SCALE   = 100.0
ALLOC_LAMBDA         = 10.0
ALLOC_TURNOVER_PEN   = 0.5
ALLOC_PERF_WINDOW    = 60
ALLOC_EPISODE_BARS   = 252           # 50 decisions of ALLOC_STEP days
ALLOC_STEPS          = 30_000
ALLOC_STEPS_QUICK    = 8_000
ALLOC_SEEDS          = 5
ALLOC_SEEDS_QUICK    = 2
ALLOC_SHARPE_CLIP    = 3.0           # sleeve-performance input, clipped then / 3
ALLOC_DD_SCALE       = 10.0          # drawdown input × 10, clipped to [0, 3]
ALLOC_DD_CLIP        = 3.0

# Both learners
MULTI_NET_ARCH     = [64, 64]
MULTI_SAC_KWARGS   = dict(learning_rate=3e-4, batch_size=256, gamma=0.9, tau=0.005,
                          ent_coef="auto")
MULTI_BUFFER_MAX   = 100_000         # buffer_size = min(MULTI_BUFFER_MAX, steps)
MULTI_WEIGHT_DECAY = 1e-4
MULTI_DIAG_POINTS  = 10              # log fit-window active Sharpe every tenth of training

# Protocol (brief section 8)
MULTI_CF_FOLDS          = 4
MULTI_CF_FOLDS_QUICK    = 2
MULTI_CF_MIN_TRAIN_FRAC = 0.4
MULTI_CF_SEEDS          = 3
MULTI_CF_SEEDS_QUICK    = 1
MULTI_WF_TEST_YEARS     = list(range(2015, 2026))
MULTI_WF_TRAIN_YEARS    = 8
MULTI_MIN_TRAIN_YEARS   = 7.5        # a training window must hold at least this much
