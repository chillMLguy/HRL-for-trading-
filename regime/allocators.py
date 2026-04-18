"""
Portfolio allocation strategies that assign weights to Phase 0 agents.

All allocators share a common interface:
  get_weights(t, prices_up_to_t) → np.ndarray  (sums to 1, non-negative)

The weight vector is ordered to match AGENT_ORDER (derived from AGENT_PRESETS).
"""

import json
import numpy as np
import pandas as pd

from env.trading_env import AGENT_PRESETS

# Frozen ordered list — matches AGENT_PRESETS insertion order.
AGENT_ORDER = list(AGENT_PRESETS.keys())
N_AGENTS = len(AGENT_ORDER)


# ── Base ───────────────────────────────────────────────────────────

class BaseAllocator:
    """Interface for portfolio allocators."""

    def get_weights(self, t, prices_up_to_t, **kwargs):
        """
        Return weight vector of length N_AGENTS.
        Weights must sum to 1.0 and be non-negative.
        """
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError


# ── 1. Equal Weight ────────────────────────────────────────────────

class EqualWeightAllocator(BaseAllocator):
    """Constant 1/N allocation across all agents."""

    def __init__(self):
        self._w = np.full(N_AGENTS, 1.0 / N_AGENTS)

    def get_weights(self, t, prices_up_to_t, **kwargs):
        return self._w.copy()

    @property
    def name(self):
        return "equal_weight"


# ── 2. Volatility Regime ──────────────────────────────────────────

class VolatilityRegimeAllocator(BaseAllocator):
    """
    Rule-based allocation using rolling realized-vol percentile thresholds.

    Thresholds are computed from training data and frozen.  At each step
    the current rolling vol is compared to those thresholds to select a
    regime label → weight map.

    With 4 agents (aggressive, growth, balanced, conservative), the
    default weight map favors:
      low_vol  → aggressive & growth
      mid_vol  → growth & balanced
      high_vol → balanced & conservative
    """

    DEFAULT_WEIGHT_MAP = {
        "low_vol":  np.array([0.6, 0.3, 0.1]),
        "mid_vol":  np.array([0.3, 0.4, 0.3]),
        "high_vol": np.array([0.3, 0.2, 0.5]),
    }

    def __init__(self, prices, ann, q33=None, q67=None,
                 weight_map=None, thresholds_path=None):
        """
        Parameters
        ----------
        prices : pd.Series — full price series (train or test)
        ann    : int — bars_per_year (252 for daily)
        q33, q67 : float | None — pre-computed thresholds. If None and
                   thresholds_path is given, load from JSON.
        weight_map : dict | None — override default weight map
        thresholds_path : str | None — JSON file with {q33, q67, ann}
        """
        self.ann = ann
        self.weight_map = weight_map or self.DEFAULT_WEIGHT_MAP

        # Validate weight map
        for k, v in self.weight_map.items():
            assert len(v) == N_AGENTS, (
                f"Weight map '{k}' has {len(v)} entries, "
                f"need {N_AGENTS} (agents: {AGENT_ORDER})")
            assert abs(v.sum() - 1.0) < 1e-6, (
                f"Weight map '{k}' does not sum to 1")

        # Load or use provided thresholds
        if q33 is not None and q67 is not None:
            self.q33, self.q67 = float(q33), float(q67)
        elif thresholds_path is not None:
            with open(thresholds_path) as f:
                th = json.load(f)
            self.q33, self.q67 = th["q33"], th["q67"]
        else:
            raise ValueError("Must provide either (q33, q67) or "
                             "thresholds_path")

        # Pre-compute rolling vol for the full price series so we can
        # look up vol at each timestep in O(1).
        window = max(20, ann // 13)
        p = pd.Series(prices.values if hasattr(prices, "values")
                       else np.asarray(prices))
        log_r = np.log(p / p.shift(1))
        self._rolling_vol = (log_r.rolling(window).std()
                             * np.sqrt(ann)).values

    def _label_at(self, t):
        """Return regime label at timestep *t* (index into the price series)."""
        v = self._rolling_vol[t]
        if np.isnan(v):
            return "mid_vol"  # warmup fallback
        if v <= self.q33:
            return "low_vol"
        elif v <= self.q67:
            return "mid_vol"
        else:
            return "high_vol"

    def get_weights(self, t, prices_up_to_t, price_index=None, **kwargs):
        """
        Parameters
        ----------
        t           : int — step counter (0-based within episode)
        prices_up_to_t : pd.Series (ignored — we use pre-computed vol)
        price_index : int | None — absolute index into the full price series.
                      If None, uses len(prices_up_to_t)-1.
        """
        idx = price_index if price_index is not None else len(prices_up_to_t) - 1
        label = self._label_at(idx)
        return self.weight_map[label].copy()

    @property
    def name(self):
        return "vol_regime"

    @staticmethod
    def compute_thresholds(prices_train, ann):
        """
        Compute vol-regime percentile thresholds from training data.

        Returns (q33, q67) as floats.
        """
        window = max(20, ann // 13)
        p = pd.Series(prices_train.values if hasattr(prices_train, "values")
                       else np.asarray(prices_train))
        log_r = np.log(p / p.shift(1)).dropna()
        rv = log_r.rolling(window).std() * np.sqrt(ann)
        rv = rv.dropna()
        q33 = float(rv.quantile(0.33))
        q67 = float(rv.quantile(0.67))
        return q33, q67


# ── 3. HMM Allocator ──────────────────────────────────────────────

class HMMAllocator(BaseAllocator):
    """
    Uses fitted HMM posterior probabilities as the basis for agent weights.

    Because the HMM has K states but we may have N ≠ K agents, a
    (K × N) weight-mapping matrix converts posterior probs → agent weights:

        agent_weights = posterior @ weight_map   (shape: N)

    Default weight_map (3 states → 4 agents) encodes:
      state 0 (low-vol)   → favor aggressive + growth
      state 1 (mid-vol)   → favor growth + balanced
      state 2 (high-vol)  → favor balanced + conservative
    """

    # Rows = HMM states (low-vol → high-vol)
    # Cols = agents in AGENT_ORDER (aggressive, growth, balanced, conservative)
    DEFAULT_WEIGHT_MAP_3x4 = np.array([
        [0.45, 0.30, 0.15, 0.10],   # state 0: low-vol
        [0.10, 0.25, 0.40, 0.25],   # state 1: mid-vol
        [0.05, 0.10, 0.30, 0.55],   # state 2: high-vol
    ])

    def __init__(self, hmm_detector, weight_map=None):
        """
        Parameters
        ----------
        hmm_detector : HMMRegimeDetector (already fitted)
        weight_map   : np.ndarray of shape (n_states, N_AGENTS) or None
        """
        self.hmm = hmm_detector
        K = hmm_detector.n_states

        if weight_map is not None:
            self._wmap = np.asarray(weight_map, dtype=np.float64)
        elif K == 3 and N_AGENTS == 4:
            self._wmap = self.DEFAULT_WEIGHT_MAP_3x4.copy()
        elif K == N_AGENTS:
            # Identity-like: state i → agent i
            self._wmap = np.eye(K)
        else:
            raise ValueError(
                f"No default weight map for {K} HMM states → "
                f"{N_AGENTS} agents.  Provide weight_map explicitly.")

        assert self._wmap.shape == (K, N_AGENTS), \
            f"weight_map shape {self._wmap.shape} != ({K}, {N_AGENTS})"
        # Each row should sum to 1
        for i in range(K):
            assert abs(self._wmap[i].sum() - 1.0) < 1e-6, \
                f"weight_map row {i} sums to {self._wmap[i].sum()}"

        self._equal = np.full(N_AGENTS, 1.0 / N_AGENTS)

    def get_weights(self, t, prices_up_to_t, **kwargs):
        """
        Build HMM observations from prices seen so far, run forward
        algorithm, and return the posterior-weighted agent allocation.
        """
        from regime.hmm_regime import HMMRegimeDetector

        obs, valid_start = HMMRegimeDetector.build_observations(prices_up_to_t)

        # Not enough history for rolling vol → fall back to equal weight
        if len(obs) < 2:
            return self._equal.copy()

        try:
            posterior = self.hmm.predict_proba(obs)  # (T, K)
        except Exception:
            return self._equal.copy()

        # Last timestep's posterior → agent weights via mapping matrix
        p = posterior[-1]                             # (K,)
        weights = p @ self._wmap                      # (N_AGENTS,)

        # Numerical safety
        weights = np.clip(weights, 0.0, None)
        s = weights.sum()
        if s > 1e-10:
            weights /= s
        else:
            weights = self._equal.copy()

        return weights

    @property
    def name(self):
        return "hmm"
