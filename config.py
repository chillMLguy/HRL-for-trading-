# ── Asset & data ─────────────────────────
TICKER   = "^DJI"
INTERVAL = "1d"

# ── train test splits ────────────────────

TRAIN_START = "2013-01-01"
TRAIN_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2024-12-31"

# ── Execution & training ──────────────
COST_PCT = 0.0002     
SEED     = 42

# ── HMM ─────────────────────────
N_STATES     = 3
HMM_EMBARGO  = 5      # drop last N training obs to reduce
                      # train→test contamination
HMM_VAL_FRAC = 0.05   # last X% of (training − embargo) is the
                      # validation slice for log-likelihood reporting

# ── Annualization ───────────────
BARS_PER_YEAR = {"1d": 252, "1h": 1638, "30m": 3276, "15m": 6552}
