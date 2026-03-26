"""
Observation space (11 features):
  [0]  ret_5    — ~1-week rolling return  (scaled by bar size)
  [1]  ret_10   — ~2-week rolling return
  [2]  ret_20   — ~1-month rolling return
  [3]  vol_10   — ~2-week realized volatility (annualized)
  [4]  vol_20   — ~1-month realized volatility (annualized)
  [5]  rsi_14   — RSI(~14 days), scaled to [-1, 1]
  [6]  position — current position in [-1, 1]
  [7]  unreal_pnl — unrealized PnL, normalized
  [8]  time_in_trade — bars held / max_hold, in [0, 1]
  [9]  drawdown — current drawdown from peak equity
  [10] vol_ratio — vol_10 / vol_20  (vol regime signal)

Action space: continuous [-1, 1]
  -1 = full short, 0 = flat, +1 = full long
  Intermediate values = fractional positions

Reward function (unified, parametric):
  r_t = net_ret / vol_scale  -  λ * (dd_dev + dd_penalty)

  λ controls risk aversion on a smooth spectrum:
    λ = 0.0  → aggressive   
    λ = 0.25 → growth   
    λ = 0.75 → balanced  
    λ = 1.5  → conservative   
    λ = 3.0  → ultra-conservative 
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Named presets for convenience (maps name → lambda value)
AGENT_PRESETS = {
    "aggressive": 0.0,
    "growth": 0.25,
    "balanced": 0.75,
    "conservative": 1.5,
    "ultra_conservative": 3.0,
}


def _precompute_features(returns: np.ndarray, bars_per_year: int = 1638) -> dict:
    n = len(returns)
    N = n + 1
    ANN = np.float32(np.sqrt(bars_per_year))
    scale = max(1, bars_per_year // 252)   # 1 for daily, 6 for 1h

    w5  = 5  * scale   # ~1 week
    w10 = 10 * scale   # ~2 weeks
    w20 = 20 * scale   # ~1 month
    w14 = 14 * scale   # RSI period (~14 trading days)

    ret_5  = np.empty(N, dtype=np.float32)
    ret_10 = np.empty(N, dtype=np.float32)
    ret_20 = np.empty(N, dtype=np.float32)
    vol_10 = np.empty(N, dtype=np.float32)
    vol_20 = np.empty(N, dtype=np.float32)
    rsi_14 = np.empty(N, dtype=np.float32)

    cum = np.concatenate([[0.0], np.cumsum(returns)]).astype(np.float32)

    for t in range(N):
        tc = min(t, n)
        ret_5[t]  = cum[tc] - cum[max(0, tc - w5)]
        ret_10[t] = cum[tc] - cum[max(0, tc - w10)]
        ret_20[t] = cum[tc] - cum[max(0, tc - w20)]

        wv10 = returns[max(0, tc - w10): tc]
        wv20 = returns[max(0, tc - w20): tc]
        vol_10[t] = np.std(wv10) * ANN if len(wv10) > 1 else 0.0
        vol_20[t] = np.std(wv20) * ANN if len(wv20) > 1 else 0.0

        wr = returns[max(0, tc - w14): tc]
        if len(wr) < 2:
            rsi_14[t] = 0.0
        else:
            gains  = wr[wr > 0]
            losses = -wr[wr < 0]
            ag = gains.mean()  if len(gains)  > 0 else 0.0
            al = losses.mean() if len(losses) > 0 else 1e-8
            rsi_14[t] = np.float32((100.0 / (1.0 + ag / al)) / 50.0 - 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        vol_ratio = np.where(vol_20 > 1e-8, vol_10 / vol_20, np.float32(1.0))

    return {
        "ret_5":     ret_5,
        "ret_10":    ret_10,
        "ret_20":    ret_20,
        "vol_10":    vol_10,
        "vol_20":    vol_20,
        "rsi":       rsi_14,
        "vol_ratio": vol_ratio.astype(np.float32),
        "_scale":    scale,
    }


class TradingEnv(gym.Env):
    """
    Parameters
    prices          : pd.Series or np.ndarray of close prices
    lam             : float — risk aversion parameter (λ). 0 = aggressive, 3 = ultra-conservative
    bars_per_year   : 252 for daily, 1638 for 1h; drives ANN factor & lookback scaling
    cost_pct        : one-way transaction cost (default 0.02% — realistic for intraday)
    max_hold        : normalisation for time_in_trade feature
    initial_capital : starting equity
    dd_free         : drawdown level below which no penalty applies (default 3%)
    dd_max          : drawdown level at which episode terminates (default 15%)
    buf_size        : rolling window length for reward statistics
                      (auto-scaled by bar frequency if left at 0)
    eval_mode       : if True, drawdown does NOT terminate the episode (for full equity curves)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices,
        lam:             float = 0.75,
        bars_per_year:   int   = 1638,
        cost_pct:        float = 0.0001,
        max_hold:        int   = 390,
        initial_capital: float = 1.0,
        dd_free:         float = 0.03,
        dd_max:          float = 0.15,
        buf_size:        int   = 0,
        eval_mode:       bool  = False,
    ):
        super().__init__()

        arr = prices.values if hasattr(prices, "values") else np.asarray(prices)
        self.returns   = np.diff(np.log(arr)).astype(np.float32)
        self.n_bars    = len(arr)
        self._feat     = _precompute_features(self.returns, bars_per_year)

        scale = self._feat["_scale"]

        self.lam             = np.float32(lam)
        self.cost_pct        = np.float32(cost_pct)
        self.max_hold        = max_hold
        self.initial_capital = np.float32(initial_capital)
        self.dd_free         = np.float32(dd_free)
        self.dd_max          = np.float32(dd_max)
        self.dd_range        = np.float32(dd_max - dd_free)
        self.eval_mode       = eval_mode
        self.warmup          = 20 * scale   # ~1 month of bars to warm up features

        _buf_size = buf_size if buf_size > 0 else 50 * scale  # ~50 trading days
        self.buf_size = _buf_size

        # Ring buffer stores vol-normalized returns
        self._rbuf   = np.zeros(_buf_size, dtype=np.float32)
        self._rbuf_i = 0
        self._rbuf_n = 0

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.float32(-1.0), high=np.float32(1.0),
            shape=(1,), dtype=np.float32)

        self.t = self.position = self.equity = None
        self.peak_equity = self.trade_start = None

    # ── Gym interface ──────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t           = self.warmup
        self.position    = np.float32(0.0)
        self.equity      = self.initial_capital
        self.peak_equity = self.initial_capital
        self.trade_start = self.warmup
        self._rbuf[:]    = 0.0
        self._rbuf_i     = 0
        self._rbuf_n     = 0
        return self._get_obs(), {}

    def step(self, action):
        action    = np.float32(np.clip(
            np.asarray(action, dtype=np.float32).flat[0], -1.0, 1.0))
        prev_pos  = self.position
        cost      = abs(action - prev_pos) * self.cost_pct
        price_ret = self.returns[self.t - 1]
        net_ret   = np.float32(prev_pos * price_ret - cost)

        self.equity      *= np.float32(1.0 + net_ret)
        self.peak_equity  = max(self.peak_equity, self.equity)
        self.position     = action

        # Compute reward (includes ring buffer update)
        reward, dd_terminated = self._compute_reward(net_ret)

        self.t += 1
        done = self.t >= self.n_bars - 1

        # Hard drawdown termination (training only)
        if dd_terminated and not self.eval_mode:
            done = True

        if abs(self.position) < np.float32(0.01):
            self.trade_start = self.t

        info = {
            "equity":   float(self.equity),
            "position": float(self.position),
            "net_ret":  float(net_ret),
            "drawdown": float(self._current_drawdown()),
            "lam":      float(self.lam),
        }
        if dd_terminated:
            info["terminated_by"] = "drawdown"

        return self._get_obs(), float(reward), done, False, info

    # ── Unified reward function ────────────────────────────────────

    def _compute_reward(self, net_ret):
        """
        r_t = net_ret / vol_scale  -  λ * (dd_dev + dd_penalty)

        Returns (reward, dd_terminated) where dd_terminated is True if
        drawdown exceeded dd_max.
        """
        # 1. Vol-scale normalization
        vol_scale = max(self._feat["vol_20"][self.t], np.float32(1e-6))
        norm_ret  = np.float32(net_ret / vol_scale)

        # 2. Store vol-normalized return in ring buffer
        self._rbuf[self._rbuf_i] = norm_ret
        self._rbuf_i = (self._rbuf_i + 1) % self.buf_size
        self._rbuf_n = min(self._rbuf_n + 1, self.buf_size)

        # 3. Downside deviation (Component A)
        buf = self._buf()
        negative_returns = buf[buf < 0]
        dd_dev = np.float32(negative_returns.std()) \
            if len(negative_returns) >= 2 else np.float32(0.0)

        # 4. Quadratic drawdown penalty with zones (Component B)
        dd = self._current_drawdown()
        dd_terminated = False

        if dd < self.dd_free:
            dd_penalty = np.float32(0.0)
        elif dd < self.dd_max:
            dd_penalty = np.float32(
                ((dd - self.dd_free) / self.dd_range) ** 2)
        else:
            dd_terminated = True
            dd_penalty = np.float32(1.0)

        # 5. Combined reward
        penalty = dd_dev + dd_penalty
        reward  = float(norm_ret - self.lam * penalty)

        return reward, dd_terminated

    def _buf(self):
        """Return valid portion of the ring buffer as a numpy array."""
        return self._rbuf if self._rbuf_n == self.buf_size \
               else self._rbuf[:self._rbuf_n]

    # ── Observation ────────────────────────────────────────────────

    def _get_obs(self):
        t = self.t
        f = self._feat
        cum_since = float(np.sum(self.returns[self.trade_start: t]))
        obs       = np.empty(11, dtype=np.float32)
        obs[0]  = f["ret_5"][t]
        obs[1]  = f["ret_10"][t]
        obs[2]  = f["ret_20"][t]
        obs[3]  = f["vol_10"][t]
        obs[4]  = f["vol_20"][t]
        obs[5]  = f["rsi"][t]
        obs[6]  = self.position
        obs[7]  = self.position * np.float32(cum_since)
        obs[8]  = np.float32(min((t - self.trade_start) / self.max_hold, 1.0))
        obs[9]  = self._current_drawdown()
        obs[10] = f["vol_ratio"][t]
        return obs

    # ── Helpers ────────────────────────────────────────────────────

    def _current_drawdown(self):
        if self.peak_equity <= 0:
            return np.float32(0.0)
        return np.float32((self.peak_equity - self.equity) / self.peak_equity)
