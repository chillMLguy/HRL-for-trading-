"""
Observation space (20 features):

  MARKET — Momentum:
  [0]  ret_5         — ~1-week rolling return  (scaled by bar size)
  [1]  ret_10        — ~2-week rolling return
  [2]  ret_20        — ~1-month rolling return
  [3]  ret_60        — ~3-month rolling return                  (NEW — Axis D)

  MARKET — Volatility:
  [4]  vol_10        — ~2-week realized volatility (annualized)
  [5]  vol_20        — ~1-month realized volatility (annualized)
  [6]  vol_60        — ~3-month realized volatility (annualized) (NEW — Axis D)
  [7]  vol_ratio     — vol_10 / vol_20  (vol regime signal)

  MARKET — Higher-order statistics:
  [8]  rsi_14        — RSI(~14 days), scaled to [-1, 1]
  [9]  skew_20       — rolling skewness over ~1 month            (NEW — Axis A)
  [10] kurt_20       — rolling excess kurtosis, clipped [-2, 10] (NEW — Axis A)
  [11] zscore_20     — price z-score over ~1 month, clipped [-4, 4] (NEW — Axis B)
  [12] autocorr_20   — rolling lag-1 autocorrelation ~1 month    (NEW — Axis C)

  PORTFOLIO STATE:
  [13] position      — current position in [-1, 1]
  [14] unreal_pnl    — unrealized PnL, normalized
  [15] time_in_trade — bars held / max_hold, in [0, 1]
  [16] drawdown      — current drawdown from peak equity

  CNN PATTERN FEATURES (Axis E — optional, zeros if no CNN model):
  [17] cnn_feat_0    — CNN latent dimension 0
  [18] cnn_feat_1    — CNN latent dimension 1
  [19] cnn_feat_2    — CNN latent dimension 2

All lookback windows are multiplied by (bars_per_year // 252) so they
represent the same calendar duration regardless of bar frequency.
  daily (252 bars/yr):  scale=1  → windows: 5, 10, 20, 60, 14 bars
  1-hour (1638 bars/yr): scale=6 → windows: 30, 60, 120, 360, 84 bars

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

N_OBS = 20  # total observation features


def _precompute_features(returns: np.ndarray,
                         bars_per_year: int = 1638,
                         cnn_model_path: str = None) -> dict:
    """
    Compute every rolling feature once.
    Arrays have size N = n+1 so index t is always valid up to t=n.

    Parameters
    ----------
    returns        : 1-D array of log returns
    bars_per_year  : 252 for daily, 1638 for 1h
    cnn_model_path : path to pre-trained CNN .pt file (optional)
    """
    n     = len(returns)
    N     = n + 1
    ANN   = np.float32(np.sqrt(bars_per_year))
    scale = max(1, bars_per_year // 252)   # 1 for daily, 6 for 1h

    w5  = 5  * scale   # ~1 week
    w10 = 10 * scale   # ~2 weeks
    w20 = 20 * scale   # ~1 month
    w60 = 60 * scale   # ~3 months
    w14 = 14 * scale   # RSI period (~14 trading days)

    # ── Allocate arrays ──────────────────────────────────────────

    ret_5      = np.empty(N, dtype=np.float32)
    ret_10     = np.empty(N, dtype=np.float32)
    ret_20     = np.empty(N, dtype=np.float32)
    ret_60     = np.empty(N, dtype=np.float32)
    vol_10     = np.empty(N, dtype=np.float32)
    vol_20     = np.empty(N, dtype=np.float32)
    vol_60     = np.empty(N, dtype=np.float32)
    rsi_14     = np.empty(N, dtype=np.float32)
    skew_20    = np.empty(N, dtype=np.float32)
    kurt_20    = np.empty(N, dtype=np.float32)
    zscore_20  = np.empty(N, dtype=np.float32)
    autocorr_20 = np.empty(N, dtype=np.float32)

    cum = np.concatenate([[0.0], np.cumsum(returns)]).astype(np.float32)

    # ── Main feature loop ────────────────────────────────────────

    for t in range(N):
        tc = min(t, n)

        # Momentum (cumulative returns over window)
        ret_5[t]  = cum[tc] - cum[max(0, tc - w5)]
        ret_10[t] = cum[tc] - cum[max(0, tc - w10)]
        ret_20[t] = cum[tc] - cum[max(0, tc - w20)]
        ret_60[t] = cum[tc] - cum[max(0, tc - w60)]

        # Volatility (annualized std of returns)
        wv10 = returns[max(0, tc - w10): tc]
        wv20 = returns[max(0, tc - w20): tc]
        wv60 = returns[max(0, tc - w60): tc]
        vol_10[t] = np.std(wv10) * ANN if len(wv10) > 1 else 0.0
        vol_20[t] = np.std(wv20) * ANN if len(wv20) > 1 else 0.0
        vol_60[t] = np.std(wv60) * ANN if len(wv60) > 1 else 0.0

        # RSI
        wr = returns[max(0, tc - w14): tc]
        if len(wr) < 2:
            rsi_14[t] = 0.0
        else:
            gains  = wr[wr > 0]
            losses = -wr[wr < 0]
            ag = gains.mean()  if len(gains)  > 0 else 0.0
            al = losses.mean() if len(losses) > 0 else 1e-8
            rsi_14[t] = np.float32((100.0 / (1.0 + ag / al)) / 50.0 - 1.0)

        # ── Axis A: Skewness & Kurtosis (w20 window) ────────────
        w_20 = returns[max(0, tc - w20): tc]
        if len(w_20) >= 3:
            m = w_20.mean()
            s = w_20.std()
            if s > 1e-10:
                z = (w_20 - m) / s
                skew_20[t] = np.float32(np.mean(z ** 3))
            else:
                skew_20[t] = 0.0
        else:
            skew_20[t] = 0.0

        if len(w_20) >= 4:
            m = w_20.mean()
            s = w_20.std()
            if s > 1e-10:
                z = (w_20 - m) / s
                kurt_20[t] = np.float32(np.clip(
                    np.mean(z ** 4) - 3.0, -2.0, 10.0))
            else:
                kurt_20[t] = 0.0
        else:
            kurt_20[t] = 0.0

        # ── Axis B: Z-score (w20 window) ────────────────────────
        if len(w_20) >= 5:
            cum_ret = w_20.sum()
            s = w_20.std() * np.sqrt(len(w_20))
            zscore_20[t] = np.float32(np.clip(
                cum_ret / s if s > 1e-10 else 0.0, -4.0, 4.0))
        else:
            zscore_20[t] = 0.0

        # ── Axis C: Lag-1 autocorrelation (w20 window) ──────────
        if len(w_20) >= 5:
            r1 = w_20[:-1]
            r2 = w_20[1:]
            m1, m2 = r1.mean(), r2.mean()
            s1, s2 = r1.std(), r2.std()
            if s1 > 1e-10 and s2 > 1e-10:
                autocorr_20[t] = np.float32(
                    np.mean((r1 - m1) * (r2 - m2)) / (s1 * s2))
            else:
                autocorr_20[t] = 0.0
        else:
            autocorr_20[t] = 0.0

    # ── Vol ratio ────────────────────────────────────────────────
    with np.errstate(invalid="ignore", divide="ignore"):
        vol_ratio = np.where(vol_20 > 1e-8, vol_10 / vol_20,
                             np.float32(1.0))

    # ── Axis E: CNN features ──────────────────────────
    cnn_feat_0 = np.zeros(N, dtype=np.float32)
    cnn_feat_1 = np.zeros(N, dtype=np.float32)
    cnn_feat_2 = np.zeros(N, dtype=np.float32)

    if cnn_model_path is not None:
        try:
            import torch
            from env.cnn_model import PricePatternCNN
            import json, os

            meta_path = os.path.join(os.path.dirname(cnn_model_path),
                                     "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                w_cnn = meta.get("input_length", w60)
                latent_dim = meta.get("latent_dim", 3)
            else:
                w_cnn = w60
                latent_dim = 3

            model = PricePatternCNN(input_length=w_cnn,
                                    latent_dim=latent_dim)
            model.load_state_dict(torch.load(cnn_model_path,
                                             map_location="cpu",
                                             weights_only=True))
            model.eval()

            # Batch-extract CNN features for all valid timesteps
            inputs = []
            valid_indices = []
            for t in range(N):
                tc = min(t, n)
                if tc >= w_cnn:
                    window = returns[tc - w_cnn: tc].copy()
                    # Vol-normalize the window
                    v = vol_20[t]
                    if v > 1e-6:
                        window = window / (v / ANN)  # undo annualization
                    inputs.append(window)
                    valid_indices.append(t)

            if inputs:
                batch = torch.tensor(np.array(inputs), dtype=torch.float32)
                latent = model.extract_features(batch).numpy()
                for i, t in enumerate(valid_indices):
                    cnn_feat_0[t] = latent[i, 0]
                    cnn_feat_1[t] = latent[i, 1] if latent_dim > 1 else 0.0
                    cnn_feat_2[t] = latent[i, 2] if latent_dim > 2 else 0.0

            print(f"  CNN features loaded from {cnn_model_path} "
                  f"({len(valid_indices)} bars with CNN features)")

        except Exception as e:
            print(f"  [WARN] CNN feature extraction failed: {e}")
            print("  Falling back to zero CNN features.")

    return {
        "ret_5":       ret_5,
        "ret_10":      ret_10,
        "ret_20":      ret_20,
        "ret_60":      ret_60,
        "vol_10":      vol_10,
        "vol_20":      vol_20,
        "vol_60":      vol_60,
        "rsi":         rsi_14,
        "vol_ratio":   vol_ratio.astype(np.float32),
        "skew_20":     skew_20,
        "kurt_20":     kurt_20,
        "zscore_20":   zscore_20,
        "autocorr_20": autocorr_20,
        "cnn_feat_0":  cnn_feat_0,
        "cnn_feat_1":  cnn_feat_1,
        "cnn_feat_2":  cnn_feat_2,
        "_scale":      scale,
    }


class TradingEnv(gym.Env):
    """
    Parameters
    ----------
    prices          : pd.Series or np.ndarray of close prices
    lam             : float — risk aversion parameter (λ). 0 = aggressive, 3 = ultra-conservative
    bars_per_year   : 252 for daily, 1638 for 1h; drives ANN factor & lookback scaling
    cost_pct        : one-way transaction cost (default 0.01% — realistic for intraday)
    max_hold        : normalisation for time_in_trade feature
    initial_capital : starting equity
    dd_free         : drawdown level below which no penalty applies (default 3%)
    dd_max          : drawdown level at which episode terminates (default 15%)
    buf_size        : rolling window length for reward statistics
                      (auto-scaled by bar frequency if left at 0)
    eval_mode       : if True, drawdown does NOT terminate the episode (for full equity curves)
    cnn_model_path  : path to pre-trained CNN .pt file (optional, None = no CNN features)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices,
        lam:             float = 0.75,
        bars_per_year:   int   = 1638,
        cost_pct:        float = 0.0002,
        max_hold:        int   = 390,
        initial_capital: float = 1.0,
        dd_free:         float = 0.03,
        dd_max:          float = 0.15,
        buf_size:        int   = 0,
        eval_mode:       bool  = False,
        cnn_model_path:  str   = None,
    ):
        super().__init__()

        arr = prices.values if hasattr(prices, "values") else np.asarray(prices)
        self.returns   = np.diff(np.log(arr)).astype(np.float32)
        self.n_bars    = len(arr)
        self._feat     = _precompute_features(self.returns, bars_per_year,
                                              cnn_model_path)

        scale = self._feat["_scale"]

        self.lam             = np.float32(lam)
        self.cost_pct        = np.float32(cost_pct)
        self.max_hold        = max_hold
        self.initial_capital = np.float32(initial_capital)
        self.dd_free         = np.float32(dd_free)
        self.dd_max          = np.float32(dd_max)
        self.dd_range        = np.float32(dd_max - dd_free)
        self.eval_mode       = eval_mode
        self.warmup          = 60 * scale   

        _buf_size = buf_size if buf_size > 0 else 50 * scale 
        self.buf_size = _buf_size

        # Ring buffer stores vol-normalized returns
        self._rbuf   = np.zeros(_buf_size, dtype=np.float32)
        self._rbuf_i = 0
        self._rbuf_n = 0

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(N_OBS,), dtype=np.float32)
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
          r_t = net_ret / downside_dev  -  λ * (dd_dev + dd_penalty)

        Returns (reward, dd_terminated) where dd_terminated is True if
        drawdown exceeded dd_max.
        """
        # 1. Store raw return in ring buffer
        self._rbuf[self._rbuf_i] = np.float32(net_ret)
        self._rbuf_i = (self._rbuf_i + 1) % self.buf_size
        self._rbuf_n = min(self._rbuf_n + 1, self.buf_size)

        # 2. Downside deviation from buffer (shared by Sortino scale + penalty)
        buf = self._buf()
        negative_returns = buf[buf < 0]
        dd_dev = np.float32(negative_returns.std()) \
            if len(negative_returns) >= 2 else np.float32(0.0)

        # 3. Sortino-style scaling: divide return by downside deviation
        #    Falls back to vol_20 when not enough history to estimate dd_dev.
        if dd_dev > np.float32(1e-6):
            sortino_scale = dd_dev
        else:
            sortino_scale = max(self._feat["vol_20"][self.t], np.float32(1e-6))
        norm_ret = np.float32(net_ret / sortino_scale)

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
        obs       = np.empty(N_OBS, dtype=np.float32)

        # Market — Momentum
        obs[0]  = f["ret_5"][t]
        obs[1]  = f["ret_10"][t]
        obs[2]  = f["ret_20"][t]
        obs[3]  = f["ret_60"][t]

        # Market — Volatility
        obs[4]  = f["vol_10"][t]
        obs[5]  = f["vol_20"][t]
        obs[6]  = f["vol_60"][t]
        obs[7]  = f["vol_ratio"][t]

        # Market — Higher-order statistics
        obs[8]  = f["rsi"][t]
        obs[9]  = f["skew_20"][t]
        obs[10] = f["kurt_20"][t]
        obs[11] = f["zscore_20"][t]
        obs[12] = f["autocorr_20"][t]

        # Portfolio state
        obs[13] = self.position
        obs[14] = self.position * np.float32(cum_since)
        obs[15] = np.float32(min((t - self.trade_start) / self.max_hold, 1.0))
        obs[16] = self._current_drawdown()

        # CNN pattern features
        obs[17] = f["cnn_feat_0"][t]
        obs[18] = f["cnn_feat_1"][t]
        obs[19] = f["cnn_feat_2"][t]

        return obs

    # ── Helpers ────────────────────────────────────────────────────

    def _current_drawdown(self):
        if self.peak_equity <= 0:
            return np.float32(0.0)
        return np.float32((self.peak_equity - self.equity) / self.peak_equity)
