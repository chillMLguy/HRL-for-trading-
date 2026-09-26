"""
Shadow books for the frozen sub-agents (F2/F3).

In Phase 1/2 the sub-agents are frozen advisors.  Previously they were fed the
BLENDED portfolio's state features (position / unreal PnL / time-in-trade /
drawdown of the shared execution env) — inputs they never saw in Phase 0
training — and the meta-agent's "recent performance" feature paired *current*
proposals with the *previous* bar's return.

A ShadowBook tracks, for each sub-agent, the portfolio it WOULD have run had it
traded alone (its own position, equity, drawdown, costs).  This gives:
  * on-distribution observations for each frozen policy (obs[13..16] come from
    its own book), and
  * a correct "recent performance" feature: the trailing sum of each agent's
    hypothetical net returns, costs included.
"""
import numpy as np


class ShadowBooks:
    def __init__(self, n_agents, cost_pct, max_pos, perf_window=20,
                 start_index=0):
        self.n = n_agents
        self.cost_pct = float(cost_pct)
        self.max_pos = np.asarray(max_pos, dtype=np.float64) \
            if np.ndim(max_pos) else np.full(n_agents, float(max_pos))
        self.perf_window = int(perf_window)
        self.reset(start_index)

    def reset(self, start_index=0):
        self.pos     = np.zeros(self.n)
        self.equity  = np.ones(self.n)
        self.peak    = np.ones(self.n)
        self.trade_start = np.full(self.n, int(start_index))
        self._perf   = np.zeros((self.n, self.perf_window))
        self._perf_i = 0
        self._perf_n = 0

    # ── stepping ────────────────────────────────────────────────

    def step(self, proposals, price_ret):
        """
        Each agent moves to its proposed position (scaled by its own cap) and
        earns `price_ret` on it, paying its own transaction costs.
        Call AFTER the execution env has stepped, with the same price_ret.
        Returns the per-agent net returns.
        """
        target = np.clip(np.asarray(proposals, dtype=np.float64),
                         -1.0, 1.0) * self.max_pos
        cost    = np.abs(target - self.pos) * self.cost_pct
        net_ret = target * float(price_ret) - cost

        self.equity *= (1.0 + net_ret)
        self.peak    = np.maximum(self.peak, self.equity)
        self.pos     = target

        self._perf[:, self._perf_i] = net_ret
        self._perf_i = (self._perf_i + 1) % self.perf_window
        self._perf_n = min(self._perf_n + 1, self.perf_window)
        return net_ret

    def mark_flat(self, t):
        """Reset trade_start for agents that are ~flat (mirrors TradingEnv)."""
        flat = np.abs(self.pos) < 0.01
        self.trade_start[flat] = int(t)

    # ── features ────────────────────────────────────────────────

    def drawdown(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            dd = (self.peak - self.equity) / np.where(self.peak > 0,
                                                      self.peak, 1.0)
        return np.clip(dd, 0.0, None)

    def perf(self):
        """Trailing sum of each agent's hypothetical net returns (F2 fixed:
        realized, cost-inclusive, correctly time-aligned)."""
        if self._perf_n == 0:
            return np.zeros(self.n)
        valid = self._perf if self._perf_n == self.perf_window \
            else self._perf[:, :self._perf_n]
        return valid.sum(axis=1)

    def agent_obs(self, base_env, i):
        """Observation for sub-agent i at the env's current bar, with the
        agent's OWN book in the portfolio-state slots (F3)."""
        return base_env.obs_with_book(self.pos[i],
                                      int(self.trade_start[i]),
                                      float(self.drawdown()[i]))
