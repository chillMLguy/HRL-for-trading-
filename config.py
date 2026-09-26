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
