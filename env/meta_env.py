"""
Phase 2 — Meta-RL environment (rewritten: F2/F3/F8 fixes + R1/R2/R3).

A learned SAC meta-agent blends the FROZEN Phase 0 sub-agents.  Changes vs the
original version:

  * F3 — each frozen sub-agent now observes its OWN shadow book (position /
    unreal PnL / time-in-trade / drawdown), not the blended portfolio's, so
    the frozen policies run on-distribution.  See env/shadow.py.
  * F2 — the "recent performance" features are the trailing sums of each
    agent's hypothetical net returns (cost-inclusive, correctly time-aligned).
  * R1 — the action is a softmax over {agents..., CASH}: the meta-agent can
    de-risk to cash instead of only re-mixing the agents.
  * R2 — optional volatility-target overlay: the blended position is scaled by
    clip(vol_target / EWMA realized vol, 0, cap).  The overlay may lever up to
    `vol_target_cap` (the execution env's max_pos is widened accordingly).
  * R3 — default reward is the Differential Sharpe Ratio (Moody & Saffell)
    of the executed net returns, plus the diversity bonus, turnover penalty
    and the drawdown-zone penalty.  `meta_reward="composite"` keeps a
    mean–variance composite for ablation.
  * F8 — pass `start_index` so evaluation windows include warm feature/HMM
    history but trading starts exactly at the test boundary.
  * Cross-fitted training — pass `proposal_stream` and `hmm_stream` to replay
    recorded out-of-sample behaviour instead of querying live agents (see
    common/crossfit.py), and `train_episode_bars` for random-start episodes.

Meta-observation layout (K = HMM states, N = number of sub-agents, C = 1 if
cash arm else 0):

  [0:K]        HMM filtered posterior (causal)
  [K:K+3]      vol_10, vol_20, vol_60
  [K+3:K+7]    ret_5, ret_10, ret_20, ret_60
  [K+7]        portfolio drawdown        (execution env)
  [K+8]        current executed position (position units; can exceed 1 with lever)
  [K+9]        current vol-target leverage (1.0 when overlay off)
  [.. +N]      sub-agent proposed positions (action × its own max_pos)
  [.. +N]      sub-agent shadow performance (trailing net return sum)
  [.. +N+C]    previous allocation weights

Meta-action: Box(-1, 1, shape=(N+C,)) → softmax(logit_scale · a) → weights.
"""

import os

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC

from env.trading_env import TradingEnv
from env.shadow import ShadowBooks
from regime.allocators import AGENT_ORDER, N_AGENTS
from regime.hmm_regime import HMMRegimeDetector
from config import AGENT_PRESETS
import config as CFG


# ── Helpers ────────────────────────────────────────────────────────────────

def softmax_weights(action, logit_scale=2.0):
    """Map an unconstrained action in [-1, 1]^M to simplex weights."""
    z = logit_scale * np.asarray(action, dtype=np.float64).ravel()
    z -= z.max()
    e = np.exp(z)
    return (e / e.sum()).astype(np.float64)


def load_subagents(modeldir, agent_order=AGENT_ORDER, verbose=True):
    """Load frozen Phase 0 SAC sub-agents from <modeldir>/models/<name>_agent/."""
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
    candidate = os.path.join(modeldir, "models", "cnn_features", "cnn_model.pt")
    return candidate if os.path.isfile(candidate) else None


def ewma_vol_series(prices, span=CFG.VOL_EWMA_SPAN, bars_per_year=252):
    """
    Annualized EWMA realized vol usable at bar t WITHOUT look-ahead:
    vol[t] uses simple returns up to the move into bar t (index t-1) —
    the same information set as the env features at bar t.
    """
    arr = prices.values if hasattr(prices, "values") else np.asarray(prices)
    arr = np.asarray(arr, dtype=np.float64)
    r = pd.Series(arr).pct_change()
    s = r.ewm(span=span, min_periods=5).std() * np.sqrt(bars_per_year)
    vol = s.values                       # vol[i] uses returns up to index i
    out = np.full(len(arr), np.nan)
    out[:] = vol                          # index t ← ewm through return at t
    return out


def meta_env_from_config(prices, subagents, hmm_detector, cfg, cost_pct, **kwargs):
    """Build the env a saved meta-agent was trained for, from its config.json."""
    return MetaTradingEnv(
        prices, subagents, hmm_detector=hmm_detector, cost_pct=cost_pct,
        use_cash=cfg.get("use_cash", CFG.META_USE_CASH),
        vol_target=cfg.get("vol_target", CFG.VOL_TARGET),
        vol_target_cap=cfg.get("vol_target_cap", CFG.VOL_TARGET_CAP),
        meta_reward=cfg.get("meta_reward", CFG.META_REWARD),
        dsr_eta=cfg.get("dsr_eta", CFG.DSR_ETA),
        lam_var_meta=cfg.get("lam_var_meta", 12.0),
        alpha=cfg.get("alpha", 0.05), beta=cfg.get("beta", 0.05),
        dd_free=cfg.get("dd_free", 0.03), dd_max=cfg.get("dd_max", 0.10),
        logit_scale=cfg.get("logit_scale", 2.0),
        perf_window=cfg.get("perf_window", 20), **kwargs)


class DifferentialSharpe:
    """
    Moody & Saffell's Differential Sharpe Ratio on percent-scale returns.

      A_t = A_{t-1} + η (r_t − A_{t-1});  B_t = B_{t-1} + η (r_t² − B_{t-1})
      D_t = (B_{t-1} ΔA_t − ½ A_{t-1} ΔB_t) / (B_{t-1} − A_{t-1}²)^{3/2}

    For the first `warmup` steps (moments not yet formed) the plain scaled
    return is emitted instead.
    """

    def __init__(self, eta=CFG.DSR_ETA, warmup=20, ret_scale=100.0):
        self.eta, self.warmup, self.ret_scale = eta, int(warmup), ret_scale
        self.reset()

    def reset(self):
        self.A, self.B, self.n = 0.0, 0.0, 0

    def step(self, net_ret):
        r = float(net_ret) * self.ret_scale
        self.n += 1
        dA = r - self.A
        dB = r * r - self.B
        if self.n <= self.warmup:
            self.A += self.eta * dA
            self.B += self.eta * dB
            return r
        denom = max(self.B - self.A * self.A, 1e-8) ** 1.5
        dsr = (self.B * dA - 0.5 * self.A * dB) / denom
        self.A += self.eta * dA
        self.B += self.eta * dB
        return float(np.clip(dsr, -10.0, 10.0))


# ── Meta environment ─────────────────────────────────────────────────────────

class MetaTradingEnv(gym.Env):
    """
    Parameters
    ----------
    prices          : pd.Series — price series (warm history + trading window)
    subagents       : dict name → frozen SAC model
    hmm_detector    : fitted HMMRegimeDetector (or None)
    ann             : bars_per_year
    cost_pct        : one-way transaction cost
    use_cash        : R1 — add a cash arm to the softmax
    vol_target      : R2 — annualized vol target (0 disables the overlay)
    vol_target_cap  : max leverage the overlay may apply
    meta_reward     : "dsr" (R3) or "composite"
    lam_var_meta    : variance aversion of the composite reward
    alpha, beta     : diversity bonus / turnover penalty weights
    dd_free, dd_max : meta-level drawdown zones (termination in training)
    logit_scale     : softmax temperature
    perf_window     : trailing window for the shadow performance features
    start_index     : F8 — first bar at which trading starts
    eval_mode       : True → no drawdown termination, full window
    proposal_stream : optional (n_bars, N) array of recorded agent actions in
                      [-1, 1], one row per bar of `prices`.  Replaces `subagents`.
    hmm_stream      : optional (n_bars, K) array of recorded posteriors.  Replaces
                      `hmm_detector`.
    train_episode_bars : if > 0 and not eval_mode, each episode is a random window
                      of this many bars.  0 = always replay the full path.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices,
        subagents,
        hmm_detector=None,
        ann:          int   = 252,
        cost_pct:     float = 0.0002,
        use_cash:     bool  = CFG.META_USE_CASH,
        vol_target:   float = CFG.VOL_TARGET,
        vol_target_cap: float = CFG.VOL_TARGET_CAP,
        meta_reward:  str   = CFG.META_REWARD,
        lam_var_meta: float = 12.0,
        dsr_eta:      float = CFG.DSR_ETA,
        alpha:        float = 0.05,
        beta:         float = 0.05,
        dd_free:      float = 0.03,
        dd_max:       float = 0.10,
        logit_scale:  float = 2.0,
        perf_window:  int   = 20,
        start_index:  int   = None,
        eval_mode:    bool  = False,
        cnn_model_path: str = None,
        deterministic_subagents: bool = True,
        proposal_stream = None,
        hmm_stream = None,
        train_episode_bars: int = 0,
    ):
        super().__init__()

        assert meta_reward in ("dsr", "composite")
        self.prices       = prices
        self.subagents    = subagents
        self.hmm          = hmm_detector
        self.ann          = ann
        self.cost_pct     = cost_pct
        self.use_cash     = bool(use_cash)
        self.vol_target   = float(vol_target)
        self.vol_cap      = float(vol_target_cap)
        self.meta_reward  = meta_reward
        self.alpha        = float(alpha)
        self.beta         = float(beta)
        self.logit_scale  = float(logit_scale)
        self.perf_window  = int(perf_window)
        self.eval_mode    = eval_mode
        self.det_sub      = deterministic_subagents

        exec_cap = self.vol_cap if self.vol_target > 0 else 1.0
        self._env = TradingEnv(
            prices,
            lam_var=(lam_var_meta if meta_reward == "composite" else 0.0),
            max_pos=exec_cap,
            bars_per_year=ann, cost_pct=cost_pct,
            dd_free=dd_free, dd_max=dd_max, eval_mode=eval_mode,
            cnn_model_path=cnn_model_path, start_index=start_index,
            train_episode_bars=train_episode_bars,
        )
        self.warmup      = self._env.warmup
        self.start_index = self._env.start_index

        # Per-agent hard caps (positions are blended in position units)
        self._caps = np.array([AGENT_PRESETS[n]["max_pos"]
                               for n in AGENT_ORDER])

        # Vol-target leverage series (causal)
        self._ewma_vol = ewma_vol_series(prices, bars_per_year=ann)

        # Recorded behaviour (cross-fitted training) replaces live models
        self._stream = None
        if proposal_stream is not None:
            self._stream = np.asarray(proposal_stream, dtype=np.float64)
            if len(self._stream) != len(prices):
                raise ValueError("proposal_stream needs one row per price bar")
        self._hmm_stream = None
        if hmm_stream is not None:
            self._hmm_stream = np.asarray(hmm_stream, dtype=np.float64)
            if len(self._hmm_stream) != len(prices):
                raise ValueError("hmm_stream needs one row per price bar")

        # HMM causal posteriors precomputed once over the full series
        self._K = hmm_detector.n_states if hmm_detector is not None else 0
        if self._hmm_stream is not None:
            self._K = self._hmm_stream.shape[1]
        self._post = None
        self._valid_start = 0
        if hmm_detector is not None and self._hmm_stream is None:
            obs, self._valid_start = HMMRegimeDetector.build_observations(prices)
            self._post = hmm_detector.filtered_proba(obs)

        # Shadow books for the frozen sub-agents (F2/F3)
        self._books = ShadowBooks(N_AGENTS, cost_pct, self._caps,
                                  perf_window=self.perf_window,
                                  start_index=self.start_index)

        self._dsr = DifferentialSharpe(eta=dsr_eta)

        # Spaces
        self._n_act = N_AGENTS + (1 if self.use_cash else 0)
        self._n_obs = self._K + 3 + 4 + 3 + N_AGENTS + N_AGENTS + self._n_act
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._n_act,), dtype=np.float32)

        self._proposals_a   = None   # raw sub-agent actions in [-1, 1]
        self._proposals_pos = None   # proposals in position units (a × cap)
        self._prev_w        = None
        self._lever         = 1.0

    # ── Sub-agent proposals (from their own shadow obs — F3) ───────────────

    def _propose(self):
        a = np.zeros(N_AGENTS, dtype=np.float64)
        if self._stream is not None:
            # Bars outside the recorded window are only ever seen as the terminal
            # observation of an episode, never traded on.
            a = np.clip(np.nan_to_num(self._stream[self._env.t]), -1.0, 1.0)
            self._proposals_a   = a
            self._proposals_pos = a * self._caps
            return
        for i, name in enumerate(AGENT_ORDER):
            model = self.subagents.get(name)
            if model is None:
                continue
            obs_i = self._books.agent_obs(self._env, i)
            act, _ = model.predict(obs_i, deterministic=self.det_sub)
            a[i] = float(np.clip(np.asarray(act).ravel()[0], -1.0, 1.0))
        self._proposals_a   = a
        self._proposals_pos = a * self._caps

    # ── HMM posterior lookup (causal) ──────────────────────────────────────

    def _hmm_post(self, t):
        if self._hmm_stream is not None:
            row = self._hmm_stream[t]
            return row if np.isfinite(row).all() else np.full(self._K, 1.0 / self._K)
        if self._post is None:
            return np.empty(0, dtype=np.float64)
        i = t - self._valid_start
        if i < 0:
            return np.full(self._K, 1.0 / self._K)
        if i >= len(self._post):
            i = len(self._post) - 1
        return self._post[i]

    # ── Vol-target leverage ────────────────────────────────────────────────

    def _leverage(self, t):
        if self.vol_target <= 0:
            return 1.0
        v = self._ewma_vol[t]
        if not np.isfinite(v) or v < 1e-4:
            return 1.0
        return float(np.clip(self.vol_target / v, 0.0, self.vol_cap))

    # ── Observation assembly ───────────────────────────────────────────────

    def _make_obs(self):
        e = self._env
        t = e.t
        f = e._feat
        parts = [
            self._hmm_post(t),
            np.array([f["vol_10"][t], f["vol_20"][t], f["vol_60"][t]]),
            np.array([f["ret_5"][t], f["ret_10"][t],
                      f["ret_20"][t], f["ret_60"][t]]),
            np.array([e._current_drawdown()]),
            np.array([e.position]),
            np.array([self._lever]),
            self._proposals_pos,
            self._books.perf(),
            self._prev_w,
        ]
        return np.concatenate(parts).astype(np.float32)

    # ── Gym API ────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._env.reset(seed=seed)
        self._books.reset(self._env.t)
        self._dsr.reset()
        self._prev_w = np.full(self._n_act, 1.0 / self._n_act)
        self._lever  = self._leverage(self._env.t)
        self._propose()
        return self._make_obs(), {}

    def step(self, action):
        weights = softmax_weights(action, self.logit_scale)

        # R1: blend agent POSITIONS; the cash arm (last weight) contributes 0.
        raw = float(np.dot(weights[:N_AGENTS], self._proposals_pos))

        # R2: vol-target overlay
        t0 = self._env.t
        self._lever = self._leverage(t0)
        executed = float(np.clip(raw * self._lever,
                                 -self._env.max_pos, self._env.max_pos))

        # Execution env acts in [-1,1] × its own max_pos
        env_action = np.array([executed / self._env.max_pos], dtype=np.float32)
        _, base_reward, done, truncated, info = self._env.step(env_action)

        # Shadow books earn the SAME realized return (F2 time-aligned)
        self._books.step(self._proposals_a, info["price_ret"])
        self._books.mark_flat(self._env.t)

        # Reward
        diversity = self._entropy(weights)
        turnover  = float(np.abs(weights - self._prev_w).sum())
        if self.meta_reward == "dsr":
            core = self._dsr.step(info["net_ret"]) - info["dd_penalty"]
        else:
            core = base_reward
        meta_reward = float(core + self.alpha * diversity - self.beta * turnover)

        # Advance decision state
        self._propose()
        self._prev_w = weights

        info = dict(info)
        info.update({
            "weights":       weights,               # incl. cash arm if enabled
            "agent_weights": weights[:N_AGENTS],
            "w_cash":        float(weights[-1]) if self.use_cash else 0.0,
            "raw_blend":     raw,
            "lever":         self._lever,
            "blended":       executed,
            "proposals":     self._proposals_pos.copy(),
            "shadow_perf":   self._books.perf(),
            "diversity":     diversity,
            "turnover":      turnover,
            "base_reward":   float(base_reward),
        })
        if self._K:
            info["hmm_post"] = self._hmm_post(self._env.t)

        return self._make_obs(), meta_reward, done, truncated, info

    @staticmethod
    def _entropy(w):
        p = np.clip(w, 1e-12, 1.0)
        h = -np.sum(p * np.log(p))
        return float(h / np.log(len(w)))
