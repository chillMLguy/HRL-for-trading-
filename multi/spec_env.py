"""
Layer 1: the asset specialists (brief section 6).

A specialist looks at one asset and outputs one tilt a in [−1, 1] a day, which sets
that asset's multiplier m = 1 + a on its core position.  It is paid only for timing:

    p[t]      = m[t] · b[t]                 sleeve position with the tilt
    s[t+1]    = b[t] · Rex[t+1]             what the untilted sleeve earns
    mu_s      = mean of s over the fit window, per asset, computed before training
    active[t] = (m[t] − 1) · (s[t+1] − mu_s) − c · (|p[t] − p[t−1]| − |b[t] − b[t−1]|)
    reward[t] = SPEC_REWARD_SCALE · (active[t] − ½ · SPEC_LAMBDA · active[t]²)

A tilt that never changes earns nothing on average over the fit window, so the
network cannot score by learning which assets rose in the training years.  The
reward is for training only; every reported number uses Rex (multi/portfolio.py).

Fit windows follow multi/data.py: a fit window is a bar range [fit_lo, fit_hi) of
data, its decisions are fit_lo .. fit_hi − 2, and each decision earns the return
realised at the next bar, so no reward uses data at or after bar fit_hi.

`specialist_tilts` turns a seed ensemble into a table of daily tilts: for each day the
tilt is the mean of the seeds' deterministic actions, and that mean is what goes into
the next day's feature 12 (own previous tilt).
"""
import numpy as np
import gymnasium as gym

import config
from multi.portfolio import sleeve_sizes

MODES = ("shared_id", "shared", "separate")


def obs_dim(n_features, n_assets, mode):
    return n_features + 1 + (n_assets if mode == "shared_id" else 0)


def sleeve_returns(b, rex):
    """s[t] = b[t] · Rex[t+1]: the untilted sleeve's return of decision t (NaN at the end)."""
    s = np.full_like(b, np.nan)
    s[:-1] = b[:-1] * rex[1:]
    return s


def fit_mu_s(s, fit_lo, fit_hi):
    """mu_s per asset: mean of s over the decisions fit_lo .. fit_hi − 2."""
    return s[fit_lo:fit_hi - 1].mean(axis=0)


class SpecialistEnv(gym.Env):
    """
    One episode = one asset over SPEC_EPISODE_BARS consecutive decisions inside the
    fit window (training), or one given asset over a given decision range (evaluation).

    inputs     multi.features.MarketInputs
    feats      (T, N, F) observation features; default inputs.feats (features 0-11)
    fit_lo/hi  fit window bars [fit_lo, fit_hi); mu_s is computed on it
    eval_asset, eval_start, eval_end   evaluation mode: walk decisions [start, end)
    """
    metadata = {"render_modes": []}

    def __init__(self, inputs, fit_lo, fit_hi, mode=None, feats=None,
                 spec_lambda=None, reward_scale=None, cost_pct=None,
                 episode_bars=None, assets=None, eval_asset=None, eval_start=None,
                 eval_end=None, seed=None):
        super().__init__()
        self.mode = mode or config.SPEC_MODE
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.lam = config.SPEC_LAMBDA if spec_lambda is None else spec_lambda
        self.scale = config.SPEC_REWARD_SCALE if reward_scale is None else reward_scale
        self.c = config.COST_PCT if cost_pct is None else cost_pct
        self.episode_bars = episode_bars or config.SPEC_EPISODE_BARS
        self.feats = np.asarray(inputs.feats if feats is None else feats, np.float64)
        T, N, F = self.feats.shape
        self.N = N
        self.b = sleeve_sizes(inputs.vol)
        self.s = sleeve_returns(self.b, inputs.rex)
        self.fit_lo, self.fit_hi = int(fit_lo), int(fit_hi)
        self.d_lo, self.d_hi = self.fit_lo, self.fit_hi - 1        # decisions [d_lo, d_hi)
        if self.d_lo < 1 or self.d_hi > T - 1:
            raise ValueError("Fit window must leave one bar before it and one after.")
        win = slice(self.d_lo, self.d_hi)
        if not (np.isfinite(self.feats[win]).all() and np.isfinite(self.s[win]).all()
                and np.isfinite(self.b[self.d_lo - 1:self.d_hi]).all()):
            raise ValueError("Features, sleeves or returns undefined in the fit window.")
        self.mu_s = fit_mu_s(self.s, self.fit_lo, self.fit_hi)
        self.assets = list(range(N)) if assets is None else list(assets)
        self.eval_asset = eval_asset
        self.eval_start, self.eval_end = eval_start, eval_end
        if eval_asset is None and self.d_hi - self.d_lo < self.episode_bars:
            raise ValueError("Fit window shorter than one episode.")
        self.eye = np.eye(N, dtype=np.float32)
        dim = obs_dim(F, N, self.mode)
        self.observation_space = gym.spaces.Box(-10.0, 10.0, (dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        self._rng = np.random.default_rng(seed)
        self.t = self.i = self.steps = 0
        self.prev_a = 0.0
        self.p_prev = self.b_prev = 0.0

    def _obs(self):
        parts = [self.feats[self.t, self.i].astype(np.float32),
                 np.array([self.prev_a], dtype=np.float32)]
        if self.mode == "shared_id":
            parts.append(self.eye[self.i])
        return np.concatenate(parts)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.eval_asset is not None:
            self.i = int(self.eval_asset)
            self.t = int(self.eval_start)
        else:
            self.i = int(self._rng.choice(self.assets))
            self.t = int(self._rng.integers(self.d_lo, self.d_hi - self.episode_bars + 1))
        self.steps = 0
        self.prev_a = 0.0
        self.b_prev = self.b[self.t - 1, self.i]
        self.p_prev = self.b_prev                 # starting untilted costs nothing
        return self._obs(), {}

    def active_return(self, t, i, a, p_prev, b_prev):
        """The timing payoff of tilt a on asset i at decision t (reward before scaling)."""
        b = self.b[t, i]
        p = (1.0 + a) * b
        cost = self.c * (abs(p - p_prev) - abs(b - b_prev))
        return a * (self.s[t, i] - self.mu_s[i]) - cost, p

    def step(self, action):
        a = float(np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[0], -1.0, 1.0))
        t, i = self.t, self.i
        active, p = self.active_return(t, i, a, self.p_prev, self.b_prev)
        reward = self.scale * (active - 0.5 * self.lam * active * active)
        self.p_prev, self.b_prev, self.prev_a = p, self.b[t, i], a
        self.t += 1
        self.steps += 1
        if self.eval_asset is not None:
            truncated = self.t >= self.eval_end
        else:
            truncated = self.steps >= self.episode_bars
        info = {"t": t, "asset": i, "tilt": a, "active": active}
        return self._obs(), float(reward), False, truncated, info


def _predict_mean(models, obs):
    acts = [np.asarray(m.predict(obs, deterministic=True)[0], dtype=np.float64)
            .reshape(len(obs)) for m in models]
    return np.mean(acts, axis=0)


def specialist_tilts(models, inputs, start, end, mode=None, feats=None, leadin=None):
    """
    Ensemble tilts a[t, i] for decision days start .. end − 1, shape (end − start, N).

    models   list of trained models (modes shared_id / shared); the tilt is the mean
             of their deterministic actions
    Inference starts `leadin` days before `start` with previous tilt 0, so the tilt in
    force on the first used day does not depend on the artificial start.
    """
    mode = mode or config.SPEC_MODE
    if mode == "separate":
        raise NotImplementedError("mode 'separate' is an ablation run only on request "
                                  "(brief section 12).")
    leadin = config.SPEC_INFER_LEADIN if leadin is None else leadin
    feats = np.asarray(inputs.feats if feats is None else feats, np.float64)
    T, N, F = feats.shape
    t0 = start - leadin
    if t0 < 0 or not np.isfinite(feats[t0:end]).all():
        raise ValueError("Features undefined in the inference window (lead-in included).")
    eye = np.eye(N, dtype=np.float32)
    prev = np.zeros(N)
    out = np.empty((end - start, N))
    for t in range(t0, end):
        obs = [feats[t].astype(np.float32), prev[:, None].astype(np.float32)]
        if mode == "shared_id":
            obs.append(eye)
        a = np.clip(_predict_mean(models, np.concatenate(obs, axis=1)), -1.0, 1.0)
        if t >= start:
            out[t - start] = a
        prev = a
    return out


def tilts_to_multipliers(tilts, start, T, N):
    """Full (T, N) multiplier array: 1 + tilt on start .. start + len − 1, 1 elsewhere."""
    m = np.ones((T, N))
    m[start:start + len(tilts)] = 1.0 + tilts
    return m


def load_ensemble(model_dirs):
    from stable_baselines3 import SAC
    import os
    return [SAC.load(os.path.join(d, "model.zip"), device="cpu") for d in model_dirs]
