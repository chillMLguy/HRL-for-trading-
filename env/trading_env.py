"""
Observation space (20 features):

  MARKET — Momentum:
  [0]  ret_5         — ~1-week rolling return
  [1]  ret_10        — ~2-week rolling return
  [2]  ret_20        — ~1-month rolling return
  [3]  ret_60        — ~3-month rolling return

  MARKET — Volatility:
  [4]  vol_10        — ~2-week realized volatility (annualized)
  [5]  vol_20        — ~1-month realized volatility (annualized)
  [6]  vol_60        — ~3-month realized volatility (annualized)
  [7]  vol_ratio     — vol_10 / vol_20  (vol regime signal)

  MARKET — Higher-order statistics:
  [8]  rsi_14        — RSI(~14 days), scaled to [-1, 1]
  [9]  skew_20       — rolling skewness over ~1 month
  [10] kurt_20       — rolling excess kurtosis, clipped [-2, 10]
  [11] zscore_20     — price z-score over ~1 month, clipped [-4, 4]
  [12] autocorr_20   — rolling lag-1 autocorrelation ~1 month

  PORTFOLIO STATE:
  [13] position      — current position (in units of max long = 1)
  [14] unreal_pnl    — unrealized PnL, normalized
  [15] time_in_trade — bars held / max_hold, in [0, 1]
  [16] drawdown      — current drawdown from peak equity

  CNN PATTERN FEATURES (optional, zeros if no CNN model):
  [17..19] cnn_feat_0..2

Action space: continuous [-1, 1], scaled internally by `max_pos`
  (per-agent hard exposure cap — part of the R4 risk ladder).

Reward function (F5/R4 — Ritter-2017 per-step quadratic utility + drawdown zones):

    r_t = REWARD_SCALE · (net_ret_t − ½·lam_var·net_ret_t²)  −  dd_penalty_t

  * net_ret_t = position_t · simple_ret_{t→t+1} − cost_t   (F1: the reward of
    step t is the PnL of the action taken at step t — no one-step credit lag)
  * lam_var sets risk aversion: optimal size ≈ μ/(lam_var·σ²), so the λ-spectrum
    now genuinely differentiates position sizes (the old reward divided returns
    by the agent's own downside deviation, which is scale-invariant in position
    size — all agents traded ~the same size).
  * dd_penalty: 0 below dd_free, quadratic in (dd_free, dd_max), 1 + episode
    termination above dd_max (training only).  dd zones are per-agent.

PnL uses SIMPLE returns (F4) so a full-long agent exactly replicates
buy-and-hold; log returns remain the basis of all features.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import AGENT_PRESETS, REWARD_SCALE  # re-exported for other modules

N_OBS = 20  # total observation features

# Defaults for env-internal lookbacks. Both are scaled by the per-frequency
# factor inside TradingEnv (×1 for daily, ×6 for 1h, etc.).
DEFAULT_WARMUP_BARS = 60   # bars before trading can start (need rolling stats)
DEFAULT_MAX_HOLD    = 60   # normalisation for the time_in_trade feature (F12:
                           # was 390 — an intraday leftover that pinned the
                           # feature near zero on daily bars)


def _precompute_features(returns: np.ndarray,
                         bars_per_year: int = 252,
                         cnn_model_path: str = None) -> dict:
    """
    Compute every rolling feature once (from LOG returns).
    Arrays have size N = n+1 so index t is always valid up to t=n.
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

        # Skewness & Kurtosis (w20 window)
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

        # Z-score (w20 window)
        if len(w_20) >= 5:
            cum_ret = w_20.sum()
            s = w_20.std() * np.sqrt(len(w_20))
            zscore_20[t] = np.float32(np.clip(
                cum_ret / s if s > 1e-10 else 0.0, -4.0, 4.0))
        else:
            zscore_20[t] = 0.0

        # Lag-1 autocorrelation (w20 window)
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

    with np.errstate(invalid="ignore", divide="ignore"):
        vol_ratio = np.where(vol_20 > 1e-8, vol_10 / vol_20,
                             np.float32(1.0))

    # CNN features (optional)
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

            inputs = []
            valid_indices = []
            for t in range(N):
                tc = min(t, n)
                if tc >= w_cnn:
                    window = returns[tc - w_cnn: tc].copy()
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
    lam_var         : float — variance-aversion λ in the quadratic-utility
                      reward (R4).  0 = pure return (execution env).
    max_pos         : float — hard cap on |position| (per-agent ladder, R4).
                      May exceed 1 for the meta execution env when the
                      vol-target overlay is allowed to lever up (R2).
    bars_per_year   : 252 for daily, 1638 for 1h
    cost_pct        : one-way transaction cost
    dd_free, dd_max : drawdown penalty zones; > dd_max terminates (training)
    reward_scale    : multiplies the utility term (SAC reward-scale hygiene)
    start_index     : first bar at which trading may start (F8 warm windows).
                      Features/HMM history before it are used but not traded.
    train_episode_bars : if > 0 and not eval_mode, each episode is a random
                      window of this many bars (F11).  0 = full path.
    eval_mode       : if True — full path from start_index, no dd termination
    preset          : optional name from config.AGENT_PRESETS — fills
                      lam_var / max_pos / dd_free / dd_max in one go.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices,
        lam_var:         float = 12.0,
        max_pos:         float = 1.0,
        bars_per_year:   int   = 252,
        cost_pct:        float = 0.0002,
        max_hold:        int   = DEFAULT_MAX_HOLD,
        initial_capital: float = 1.0,
        dd_free:         float = 0.03,
        dd_max:          float = 0.15,
        reward_scale:    float = REWARD_SCALE,
        eval_mode:       bool  = False,
        cnn_model_path:  str   = None,
        start_index:     int   = None,
        train_episode_bars: int = 0,
        preset:          str   = None,
    ):
        super().__init__()

        if preset is not None:
            p = AGENT_PRESETS[preset]
            lam_var, max_pos = p["lam_var"], p["max_pos"]
            dd_free, dd_max  = p["dd_free"], p["dd_max"]

        arr = prices.values if hasattr(prices, "values") else np.asarray(prices)
        arr = np.asarray(arr, dtype=np.float64)
        self.n_bars = len(arr)
        # Log returns drive the FEATURES; simple returns drive the PnL (F4).
        self.returns        = np.diff(np.log(arr)).astype(np.float32)
        self.simple_returns = (arr[1:] / arr[:-1] - 1.0).astype(np.float32)
        self._feat = _precompute_features(self.returns, bars_per_year,
                                          cnn_model_path)

        scale = self._feat["_scale"]

        self.lam_var         = np.float32(lam_var)
        self.max_pos         = np.float32(max_pos)
        self.cost_pct        = np.float32(cost_pct)
        self.max_hold        = max_hold * scale
        self.initial_capital = np.float32(initial_capital)
        self.dd_free         = np.float32(dd_free)
        self.dd_max          = np.float32(dd_max)
        self.dd_range        = np.float32(dd_max - dd_free)
        self.reward_scale    = np.float32(reward_scale)
        self.eval_mode       = eval_mode
        self.warmup          = DEFAULT_WARMUP_BARS * scale
        self.start_index     = max(self.warmup,
                                   start_index if start_index is not None else 0)
        self.train_episode_bars = int(train_episode_bars)

        if self.start_index >= self.n_bars - 2:
            raise ValueError("start_index leaves no bars to trade")

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(N_OBS,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.float32(-1.0), high=np.float32(1.0),
            shape=(1,), dtype=np.float32)

        self.t = self.position = self.equity = None
        self.peak_equity = self.trade_start = self._ep_end = None

    # ── Gym API ────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        t0     = self.start_index
        ep_end = self.n_bars - 1          # last valid t is n_bars-2 (uses ret[t])

        # F11: random-start episodes during training
        if (not self.eval_mode) and self.train_episode_bars > 0:
            max_start = self.n_bars - 2 - self.train_episode_bars
            if max_start > t0:
                t0 = int(self.np_random.integers(t0, max_start + 1))
            ep_end = min(ep_end, t0 + self.train_episode_bars)

        self.t           = t0
        self._ep_end     = ep_end
        self.position    = np.float32(0.0)
        self.equity      = self.initial_capital
        self.peak_equity = self.initial_capital
        self.trade_start = t0
        return self._get_obs(), {}

    def step(self, action):
        # Action in [-1, 1] → position in [-max_pos, max_pos]
        a = np.float32(np.clip(
            np.asarray(action, dtype=np.float32).flat[0], -1.0, 1.0))
        target_pos = a * self.max_pos

        prev_pos  = self.position
        cost      = abs(target_pos - prev_pos) * self.cost_pct
        # F1: the position chosen NOW earns the return revealed on the next
        # bar, and that PnL is the reward of THIS step (no one-step credit lag;
        # the trade cost and its payoff sit in the same reward).
        price_ret = self.simple_returns[self.t]
        net_ret   = np.float32(target_pos * price_ret - cost)

        self.equity      *= np.float32(1.0 + net_ret)
        self.peak_equity  = max(self.peak_equity, self.equity)
        self.position     = target_pos

        reward, dd_penalty, dd_terminated = self._compute_reward(net_ret)

        self.t += 1
        done = (self.t >= self._ep_end) or (self.t >= self.n_bars - 1)
        if dd_terminated and not self.eval_mode:
            done = True

        if abs(self.position) < np.float32(0.01):
            self.trade_start = self.t

        info = {
            "equity":     float(self.equity),
            "position":   float(self.position),
            "net_ret":    float(net_ret),
            "price_ret":  float(price_ret),
            "cost":       float(cost),
            "drawdown":   float(self._current_drawdown()),
            "dd_penalty": float(dd_penalty),
            "lam_var":    float(self.lam_var),
        }
        if dd_terminated:
            info["terminated_by"] = "drawdown"

        return self._get_obs(), float(reward), done, False, info

    # ── Reward (R4: quadratic utility + drawdown zones) ────────────

    def _compute_reward(self, net_ret):
        utility = np.float32(net_ret - 0.5 * self.lam_var * net_ret * net_ret)

        dd = self._current_drawdown()
        dd_terminated = False
        if dd < self.dd_free:
            dd_penalty = np.float32(0.0)
        elif dd < self.dd_max:
            dd_penalty = np.float32(((dd - self.dd_free) / self.dd_range) ** 2)
        else:
            dd_terminated = True
            dd_penalty = np.float32(1.0)

        reward = float(self.reward_scale * utility - dd_penalty)
        return reward, dd_penalty, dd_terminated

    # ── Observation ────────────────────────────────────────────────

    def _get_obs(self):
        return self.obs_with_book(self.position,
                                  self.trade_start,
                                  self._current_drawdown())

    def obs_with_book(self, position, trade_start, drawdown):
        """
        Build the 20-feature observation at the current bar with an arbitrary
        portfolio book (F3: lets frozen sub-agents see their OWN shadow book
        instead of the blended portfolio's).
        """
        t = self.t
        f = self._feat
        cum_since = float(np.sum(self.returns[trade_start: t]))
        obs       = np.empty(N_OBS, dtype=np.float32)

        obs[0]  = f["ret_5"][t]
        obs[1]  = f["ret_10"][t]
        obs[2]  = f["ret_20"][t]
        obs[3]  = f["ret_60"][t]
        obs[4]  = f["vol_10"][t]
        obs[5]  = f["vol_20"][t]
        obs[6]  = f["vol_60"][t]
        obs[7]  = f["vol_ratio"][t]
        obs[8]  = f["rsi"][t]
        obs[9]  = f["skew_20"][t]
        obs[10] = f["kurt_20"][t]
        obs[11] = f["zscore_20"][t]
        obs[12] = f["autocorr_20"][t]
        obs[13] = np.float32(position)
        obs[14] = np.float32(position * cum_since)
        obs[15] = np.float32(min((t - trade_start) / self.max_hold, 1.0))
        obs[16] = np.float32(drawdown)
        obs[17] = f["cnn_feat_0"][t]
        obs[18] = f["cnn_feat_1"][t]
        obs[19] = f["cnn_feat_2"][t]
        return obs

    # ── Helpers ────────────────────────────────────────────────────

    def _current_drawdown(self):
        if self.peak_equity <= 0:
            return np.float32(0.0)
        return np.float32((self.peak_equity - self.equity) / self.peak_equity)
