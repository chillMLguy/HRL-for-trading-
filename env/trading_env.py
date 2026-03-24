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

All lookback windows are multiplied by (bars_per_year // 252) so they
represent the same calendar duration regardless of bar frequency.
  daily (252 bars/yr):  scale=1  → windows: 5, 10, 20, 14 bars
  1-hour (1638 bars/yr): scale=6 → windows: 30, 60, 120, 84 bars

Action space: continuous [-1, 1]
  -1 = full short, 0 = flat, +1 = full long
  Intermediate values = fractional positions
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from enum import Enum


class AgentType(Enum):
    CONSERVATIVE = "conservative"
    NEUTRAL      = "neutral"
    AGGRESSIVE   = "aggressive"
    CVAR         = "cvar"
    OMEGA        = "omega"
    RACHEV       = "rachev"


def _precompute_features(returns: np.ndarray, bars_per_year: int = 1638) -> dict:
    """
    Compute every rolling feature once.
    Arrays have size n+1 so index t is always valid up to t=n.
    bars_per_year: 252 for daily, 1638 for 1-hour (252 * 6.5).
    All lookback windows are scaled by bars_per_year // 252 so they
    represent the same calendar duration regardless of bar frequency.
    """
    n     = len(returns)
    N     = n + 1
    ANN   = np.float32(np.sqrt(bars_per_year))
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
    ----------
    prices          : pd.Series or np.ndarray of close prices
    agent_type      : AgentType enum — selects reward function
    bars_per_year   : 252 for daily, 1638 for 1h; drives ANN factor & lookback scaling
    cost_pct        : one-way transaction cost (default 0.02% — realistic for intraday)
    max_hold        : normalisation for time_in_trade feature
    initial_capital : starting equity
    lambda_drawdown : drawdown penalty weight  (CONSERVATIVE)
    lambda_vol      : rolling-vol penalty weight (CONSERVATIVE)
    cvar_alpha      : CVaR confidence level, e.g. 0.95 = worst 5% (CVAR)
    cvar_beta       : CVaR aversion weight (CVAR)
    omega_threshold : minimum acceptable return threshold (OMEGA)
    rachev_alpha    : tail quantile for Rachev ratio (RACHEV)
    buf_size        : rolling window length for reward statistics
                      (auto-scaled by bar frequency if left at 0)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices,
        agent_type:      AgentType = AgentType.NEUTRAL,
        bars_per_year:   int   = 1638,
        cost_pct:        float = 0.0002,
        max_hold:        int   = 390,
        initial_capital: float = 1.0,
        # CONSERVATIVE params
        lambda_drawdown: float = 0.5,
        lambda_vol:      float = 0.3,
        # CVAR params
        cvar_alpha:      float = 0.95,
        cvar_beta:       float = 0.1,
        # OMEGA params
        omega_threshold: float = 0.0,
        # RACHEV params
        rachev_alpha:    float = 0.05,
        # shared
        buf_size:        int   = 0,    # 0 = auto (50 trading days worth of bars)
    ):
        super().__init__()

        arr = prices.values if hasattr(prices, "values") else np.asarray(prices)
        self.returns   = np.diff(np.log(arr)).astype(np.float32)
        self.n_bars    = len(arr)
        self._feat     = _precompute_features(self.returns, bars_per_year)

        scale = self._feat["_scale"]

        self.agent_type      = agent_type
        self.cost_pct        = np.float32(cost_pct)
        self.max_hold        = max_hold
        self.initial_capital = np.float32(initial_capital)
        self.lambda_dd       = np.float32(lambda_drawdown)
        self.lambda_vol      = np.float32(lambda_vol)
        self.cvar_alpha      = cvar_alpha
        self.cvar_beta       = np.float32(cvar_beta)
        self.omega_thresh    = np.float32(omega_threshold)
        self.rachev_alpha    = rachev_alpha
        self.warmup          = 20 * scale   # ~1 month of bars to warm up features

        _buf_size = buf_size if buf_size > 0 else 50 * scale  # ~50 trading days
        self.buf_size = _buf_size

        # Ring buffer — large enough for CVaR/Omega/Rachev windows
        self._rbuf   = np.zeros(_buf_size, dtype=np.float32)
        self._rbuf_i = 0
        self._rbuf_n = 0

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.float32(-1.0), high=np.float32(1.0),
            shape=(1,), dtype=np.float32)

        self._reward_fn = {
            AgentType.CONSERVATIVE: self._reward_conservative,
            AgentType.NEUTRAL:      self._reward_neutral,
            AgentType.AGGRESSIVE:   self._reward_aggressive,
            AgentType.CVAR:         self._reward_cvar,
            AgentType.OMEGA:        self._reward_omega,
            AgentType.RACHEV:       self._reward_rachev,
        }[agent_type]

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

        self._rbuf[self._rbuf_i] = net_ret
        self._rbuf_i = (self._rbuf_i + 1) % self.buf_size
        self._rbuf_n = min(self._rbuf_n + 1, self.buf_size)

        reward = self._reward_fn(net_ret)

        self.t += 1
        done = self.t >= self.n_bars - 1

        if abs(self.position) < np.float32(0.01):
            self.trade_start = self.t

        info = {
            "equity":   float(self.equity),
            "position": float(self.position),
            "net_ret":  float(net_ret),
            "drawdown": float(self._current_drawdown()),
        }
        return self._get_obs(), float(reward), done, False, info

    # ── Reward functions ───────────────────────────────────────────

    def _buf(self):
        """Return valid portion of the ring buffer as a numpy array."""
        return self._rbuf if self._rbuf_n == self.buf_size \
               else self._rbuf[:self._rbuf_n]

    def _reward_conservative(self, net_ret):
        """
            r = net_ret - λ_dd·drawdown - λ_vol·std(recent_returns)
        """
        dd  = self._current_drawdown()
        rv  = self._buf().std() if self._rbuf_n >= 2 else np.float32(0.0)
        return float(net_ret
                     - self.lambda_dd  * max(np.float32(0.0), dd)
                     - self.lambda_vol * rv)

    def _reward_neutral(self, net_ret):
        """
            r = net_ret - 0.3·std(negative returns only)
        """
        neg = self._buf()
        neg = neg[neg < 0]
        ds  = neg.std() if len(neg) >= 2 else np.float32(0.0)
        return float(net_ret - np.float32(0.3) * ds)

    def _reward_aggressive(self, net_ret):
        """
            r = net_ret + 0.1·sign(net_ret)·|ret_5|
        """
        ret_5 = self._feat["ret_5"][self.t]
        return float(net_ret + np.float32(0.1) * np.sign(net_ret) * abs(ret_5))

    def _reward_cvar(self, net_ret):

        buf = self._buf()
        if self._rbuf_n < 20:
            return float(net_ret)

        var_threshold = np.percentile(buf, (1.0 - self.cvar_alpha) * 100.0)
        tail          = buf[buf <= var_threshold]
        cvar          = tail.mean() if len(tail) > 0 else np.float32(0.0)

        return float(net_ret - self.cvar_beta * abs(cvar))

    def _reward_omega(self, net_ret):
        buf = self._buf()
        if self._rbuf_n < 10:
            return float(net_ret)

        excess = buf - self.omega_thresh
        gains  = excess[excess > 0].sum()
        losses = (-excess[excess < 0]).sum()

        if losses < 1e-10:
            omega = np.float32(2.0)   # cap — avoid division by zero
        else:
            omega = np.float32(gains / losses)

        return float(net_ret + np.float32(0.05) * (omega - np.float32(1.0)))

    def _reward_rachev(self, net_ret):

        buf = self._buf()
        if self._rbuf_n < 20:
            return float(net_ret)

        q_lo = np.percentile(buf, self.rachev_alpha * 100.0)
        q_hi = np.percentile(buf, (1.0 - self.rachev_alpha) * 100.0)

        up_tail   = buf[buf >= q_hi]
        down_tail = buf[buf <= q_lo]

        etl_up   = up_tail.mean()   if len(up_tail)   > 0 else np.float32(0.0)
        etl_down = abs(down_tail.mean()) if len(down_tail) > 0 else np.float32(1e-8)

        rachev = np.float32(etl_up / max(etl_down, np.float32(1e-8)))

        return float(net_ret + np.float32(0.05) * (rachev - np.float32(1.0)))

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
