"""
Phase 2 — Meta-RL environment.

A learned SAC/PPO *meta-agent* replaces the fixed weight-mapping matrix of the
Phase 1 HMMAllocator.  It observes the market regime + the frozen sub-agents'
proposals and outputs a continuous 3-vector that is soft-maxed into allocation
weights over the three λ-spectrum sub-agents.  The blended action is executed in
a single portfolio TradingEnv (λ=0 execution semantics, meta-λ in the reward).

The sub-agents (aggressive / balanced / conservative) are FROZEN: they act as a
fixed part of the environment, exactly as in Phase 1's `run_allocator`.

──────────────────────────────────────────────────────────────────────────────
Meta-observation layout (length = K + 12 + 3·N_AGENTS;  21 for K=3, N=3)

  [0:K]            HMM filtered posterior  P(state_t | o_1…o_t)   (no look-ahead)
  [K:K+3]          vol_10, vol_20, vol_60                         (market vol)
  [K+3:K+7]        ret_5, ret_10, ret_20, ret_60                  (market momentum)
  [K+7]            portfolio drawdown
  [K+8]            current blended position
  [K+9 : K+9+N]    sub-agent proposed actions
  [.. : ..+N]      sub-agent recent performance (cum. hypothetical return)
  [.. : ..+N]      previous allocation weights

Meta-action: Box(-1, 1, shape=(N_AGENTS,))  →  softmax(logit_scale · a) → weights

Meta-reward (per step):

    R_meta = net_ret / vol_scale  −  λ_meta · (dd_dev + dd_penalty)     ← base env
             + α · H(weights)/log(N)            (diversity bonus, anti-collapse)
             − β · Σ|w_t − w_{t−1}|             (turnover penalty, anti-churn)

The base term is produced by the wrapped TradingEnv (lam=λ_meta, meta dd zones);
α and β are the meta-level shaping terms motivated in PROJECT_CONTEXT.md.
"""

import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC

from env.trading_env import TradingEnv
from regime.allocators import AGENT_ORDER, N_AGENTS
from regime.hmm_regime import HMMRegimeDetector


# ── Helpers ────────────────────────────────────────────────────────────────

def softmax_weights(action, logit_scale=2.0):
    """
    Map an unconstrained meta-action in [-1, 1]^N to a simplex of allocation
    weights via a temperature-scaled softmax.

    `logit_scale` controls how decisive the allocation can become: with the
    default 2.0 a saturated action like [1, -1, -1] yields ≈[0.96, 0.02, 0.02],
    so the meta-agent *can* concentrate on one sub-agent, but the soft form
    (plus the entropy bonus in the reward) discourages hard collapse.
    """
    z = logit_scale * np.asarray(action, dtype=np.float64).ravel()
    z -= z.max()                       # numerical stability
    e = np.exp(z)
    return (e / e.sum()).astype(np.float64)


def load_subagents(modeldir, agent_order=AGENT_ORDER, verbose=True):
    """
    Load the frozen Phase 0 SAC sub-agents from `<modeldir>/models/<name>_agent/
    best_model.zip`.  Returns dict name → SAC model (missing agents are simply
    absent; MetaTradingEnv treats a missing agent as a constant flat proposal).
    """
    agents = {}
    for name in agent_order:
        path = os.path.join(modeldir, "models", f"{name}_agent", "best_model")
        if os.path.isfile(path + ".zip"):
            agents[name] = SAC.load(path)
            if verbose:
                print(f"    ✓  sub-agent {name}")
        elif verbose:
            print(f"    ✗  sub-agent {name} — NOT FOUND (proposes 0)")
    return agents


def detect_cnn_path(modeldir):
    """Return the CNN feature-extractor path if present, else None."""
    candidate = os.path.join(modeldir, "models", "cnn_features", "cnn_model.pt")
    return candidate if os.path.isfile(candidate) else None


# ── Meta environment ─────────────────────────────────────────────────────────

class MetaTradingEnv(gym.Env):
    """
    Gymnasium environment whose *agent* is the meta-controller.

    Parameters
    ----------
    prices          : pd.Series — full price series for the episode window
    subagents       : dict name → frozen SAC model (see load_subagents)
    hmm_detector    : fitted HMMRegimeDetector (or None to disable regime obs)
    ann             : bars_per_year (252 for daily)
    cost_pct        : one-way transaction cost (must match sub-agent training)
    lam_meta        : risk aversion of the meta reward's base term
    alpha           : weight on the entropy (diversity) bonus
    beta            : weight on the turnover penalty
    dd_free, dd_max : meta-level drawdown zones (tighter than the sub-agents')
    logit_scale     : softmax temperature for action → weights
    perf_window     : look-back (bars) for the per-agent recent-performance feat
    eval_mode       : if True, drawdown does NOT terminate the episode
    cnn_model_path  : CNN feature extractor for the wrapped TradingEnv's obs
    deterministic_subagents : predict sub-agent actions deterministically
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices,
        subagents,
        hmm_detector=None,
        ann:          int   = 252,
        cost_pct:     float = 0.0002,
        lam_meta:     float = 0.5,
        alpha:        float = 0.05,
        beta:         float = 0.05,
        dd_free:      float = 0.03,
        dd_max:       float = 0.10,
        logit_scale:  float = 2.0,
        perf_window:  int   = 20,
        eval_mode:    bool  = False,
        cnn_model_path: str = None,
        deterministic_subagents: bool = True,
    ):
        super().__init__()

        self.prices       = prices
        self.subagents    = subagents
        self.hmm          = hmm_detector
        self.ann          = ann
        self.cost_pct     = cost_pct
        self.lam_meta     = float(lam_meta)
        self.alpha        = float(alpha)
        self.beta         = float(beta)
        self.logit_scale  = float(logit_scale)
        self.perf_window  = int(perf_window)
        self.eval_mode    = eval_mode
        self.cnn_path     = cnn_model_path
        self.det_sub      = deterministic_subagents

        # Wrapped single-portfolio execution env. λ=λ_meta so its per-step
        # reward already carries the mean-risk base term; we add the meta
        # shaping terms (entropy / turnover) on top in step().
        self._env = TradingEnv(
            prices, lam=lam_meta, bars_per_year=ann, cost_pct=cost_pct,
            dd_free=dd_free, dd_max=dd_max, eval_mode=eval_mode,
            cnn_model_path=cnn_model_path,
        )
        self.warmup = self._env.warmup

        # Number of HMM states feeding the obs (0 if no detector).
        self._K = hmm_detector.n_states if hmm_detector is not None else 0

        # Precompute the causal (forward-filtered) HMM posteriors ONCE for the
        # whole series, indexed by bar → O(1) lookup per step, no look-ahead.
        self._post = None
        self._valid_start = 0
        if hmm_detector is not None:
            obs, self._valid_start = HMMRegimeDetector.build_observations(prices)
            self._post = hmm_detector.filtered_proba(obs)   # (T_obs, K)

        # Per-agent recent-performance ring buffers (hypothetical agent returns)
        self._perf = np.zeros((N_AGENTS, self.perf_window), dtype=np.float64)
        self._perf_i = 0
        self._perf_n = 0

        # Observation / action spaces
        self._n_obs = self._K + 3 + 4 + 1 + 1 + 3 * N_AGENTS
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_AGENTS,), dtype=np.float32)

        self._base_obs   = None
        self._proposals  = None
        self._prev_w     = None

    # ── Sub-agent proposals ─────────────────────────────────────────────────

    def _propose(self, base_obs):
        """Each frozen sub-agent proposes a position in [-1, 1] from base_obs."""
        out = np.zeros(N_AGENTS, dtype=np.float64)
        for i, name in enumerate(AGENT_ORDER):
            model = self.subagents.get(name)
            if model is None:
                continue
            a, _ = model.predict(base_obs, deterministic=self.det_sub)
            out[i] = float(np.clip(np.asarray(a).ravel()[0], -1.0, 1.0))
        return out

    # ── HMM posterior lookup (causal) ───────────────────────────────────────

    def _hmm_post(self, t):
        if self._post is None:
            return np.empty(0, dtype=np.float64)
        i = t - self._valid_start
        if i < 0:
            return np.full(self._K, 1.0 / self._K)
        if i >= len(self._post):
            i = len(self._post) - 1
        return self._post[i]

    # ── Observation assembly ─────────────────────────────────────────────────

    def _make_obs(self):
        b = self._base_obs
        t = self._env.t
        perf = self._perf_sum()

        parts = [
            self._hmm_post(t),                         # K regime posteriors
            np.array([b[4], b[5], b[6]]),              # vol_10/20/60
            np.array([b[0], b[1], b[2], b[3]]),        # ret_5/10/20/60
            np.array([b[16]]),                         # drawdown
            np.array([b[13]]),                         # current position
            self._proposals,                           # sub-agent actions (N)
            perf,                                      # recent performance (N)
            self._prev_w,                              # previous weights (N)
        ]
        return np.concatenate(parts).astype(np.float32)

    def _perf_sum(self):
        """Per-agent cumulative hypothetical return over the perf window."""
        if self._perf_n == 0:
            return np.zeros(N_AGENTS, dtype=np.float64)
        valid = self._perf if self._perf_n == self.perf_window \
            else self._perf[:, :self._perf_n]
        return valid.sum(axis=1)

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._base_obs, _ = self._env.reset()
        self._proposals   = self._propose(self._base_obs)
        self._prev_w      = np.full(N_AGENTS, 1.0 / N_AGENTS)
        self._perf[:]     = 0.0
        self._perf_i      = 0
        self._perf_n      = 0
        return self._make_obs(), {}

    def step(self, action):
        weights = softmax_weights(action, self.logit_scale)

        # Blend the frozen sub-agents' proposals → single executed position.
        blended = float(np.clip(np.dot(weights, self._proposals), -1.0, 1.0))

        # Price return applied to the position on this step (see TradingEnv).
        t0 = self._env.t
        price_ret = float(self._env.returns[t0 - 1])

        self._base_obs, base_reward, done, truncated, info = self._env.step(
            np.array([blended], dtype=np.float32))

        # Record each sub-agent's hypothetical (gross) return for the perf feat.
        self._perf[:, self._perf_i] = self._proposals * price_ret
        self._perf_i = (self._perf_i + 1) % self.perf_window
        self._perf_n = min(self._perf_n + 1, self.perf_window)

        # Meta-level reward shaping.
        diversity = self._entropy(weights)
        turnover  = float(np.abs(weights - self._prev_w).sum())
        meta_reward = float(base_reward
                            + self.alpha * diversity
                            - self.beta * turnover)

        # Advance state for the next decision.
        self._proposals = self._propose(self._base_obs)
        self._prev_w    = weights

        info = dict(info)
        info.update({
            "weights":       weights,
            "blended":       blended,
            "proposals":     self._proposals.copy(),
            "diversity":     diversity,
            "turnover":      turnover,
            "base_reward":   float(base_reward),
        })
        if self._K:
            info["hmm_post"] = self._hmm_post(self._env.t)

        return self._make_obs(), meta_reward, done, truncated, info

    @staticmethod
    def _entropy(w):
        """Shannon entropy of the weight vector, normalised to [0, 1]."""
        p = np.clip(w, 1e-12, 1.0)
        h = -np.sum(p * np.log(p))
        return float(h / np.log(len(w)))
