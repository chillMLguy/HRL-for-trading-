"""
Layer 2: the allocator (brief section 7).

Once every ALLOC_STEP (5) trading days the allocator shifts risk between the eight
assets and scales total risk, both within fixed bounds:

    action  u[i] in [−1, 1] per asset, g in [−1, 1]
    beta[i] = beta_base[i] · exp(KAPPA · u[i]) / sum_j beta_base[j] · exp(KAPPA · u[j])
    G       = 1 + G_RANGE · g

It is paid only for what it adds over fixed budgets.  One step covers the H days
d = t .. t+H−1, with two books (each with its own previous positions):

    x[i,d]      positions with the allocator's beta and G       (multi.portfolio)
    x_fix[i,d]  positions with beta_base and G = 1, same m
    rho[i,d]    = G · beta[i] / beta_base[i]
    q[i,d+1]    = x_fix[i,d] · Rex[i,d+1]
    mu_q[i]     = mean of q[i,.] over the allocator's training days
    active[d]   = sum_i (x − x_fix)[i,d] · Rex[i,d+1] − sum_i (rho[i,d] − 1) · mu_q[i]
                  − c · (sum_i |x[i,d] − x[i,d−1]| − sum_i |x_fix[i,d] − x_fix[i,d−1]|)
    A           = sum of active[d] over the step
    reward      = SCALE · (A − ½ · LAMBDA · A²) − TURNOVER_PEN · sum_i |beta − beta_prev|

The environment replays recorded daily inputs and never calls a specialist: vol, Cov,
Rex, the market context, HMM posteriors and the multipliers m (the specialists'
out-of-sample tilts, or 1 for the `core_alloc` allocator).  The evaluation mode walks
a given range deciding every H days; allocator inference uses it, so a deployed
allocator sees exactly what it saw in training.

Observation, 26 numbers for eight assets:
    3  HMM posteriors                 8  current budgets ln(beta / beta_base) / KAPPA
    4  market context (features 8-11) 1  previous g
    8  60-day Sharpe of each tilted   1  ln k (core scale), clipped to [−2, 2]
       sleeve, clipped ±3, / 3        1  drawdown within the episode × 10, clipped [0, 3]
"""
import numpy as np
import gymnasium as gym

import config
from multi.portfolio import build_positions, sleeve_sizes


def budgets_from_action(u, base=None, kappa=None):
    """beta from the per-asset actions u (..., N); the zero action gives beta_base."""
    base = np.asarray(config.MULTI_BASE_BUDGETS if base is None else base, np.float64)
    kappa = config.ALLOC_KAPPA if kappa is None else kappa
    w = base * np.exp(kappa * np.asarray(u, np.float64))
    return w / w.sum(axis=-1, keepdims=True)


def gross_from_action(g, g_range=None):
    g_range = config.ALLOC_G_RANGE if g_range is None else g_range
    return 1.0 + g_range * np.asarray(g, np.float64)


def sleeve_performance(inputs, m, cost_pct=None, window=None):
    """
    (T, N) allocator input: annualised Sharpe of each tilted sleeve's net excess
    return over the last `window` days, clipped to ±3, divided by 3.  The tilted
    sleeve's return realised at t is p[t−1] · Rex[t] − c · |p[t−1] − p[t−2]| with
    p = m · b.  Uses returns realised up to and including t only.
    """
    c = config.COST_PCT if cost_pct is None else cost_pct
    window = window or config.ALLOC_PERF_WINDOW
    p = m * sleeve_sizes(inputs.vol)
    r = np.full_like(p, np.nan)
    r[2:] = p[1:-1] * inputs.rex[2:] - c * np.abs(p[1:-1] - p[:-2])
    import pandas as pd
    df = pd.DataFrame(r)
    mu = df.rolling(window, min_periods=window).mean().to_numpy()
    sd = df.rolling(window, min_periods=window).std().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        sr = mu / sd * np.sqrt(config.MULTI_BARS_PER_YEAR)
    clip = config.ALLOC_SHARPE_CLIP
    return np.clip(sr, -clip, clip) / clip


class AllocInputs:
    """Daily arrays an allocator replays: everything except its own actions."""

    def __init__(self, inputs, m, hmm):
        self.inputs = inputs
        self.T, self.N = inputs.vol.shape
        self.m = np.asarray(m, np.float64)
        self.hmm = np.asarray(hmm, np.float64)
        self.x_fix, info = build_positions(inputs.vol, inputs.cov, m=self.m,
                                           return_info=True)
        self.logk = np.clip(np.log(info["k"]), -config.ALLOC_LOGK_CLIP,
                            config.ALLOC_LOGK_CLIP)
        self.perf = sleeve_performance(inputs, self.m)
        self.q = np.full_like(self.x_fix, np.nan)
        self.q[:-1] = self.x_fix[:-1] * inputs.rex[1:]          # q of decision d

    def obs_ok(self, d):
        return (np.isfinite(self.hmm[d]).all() and np.isfinite(self.perf[d]).all()
                and np.isfinite(self.inputs.ctx[d]).all() and np.isfinite(self.logk[d])
                and np.isfinite(self.x_fix[d]).all())


def mu_q_of(ai, days):
    """mu_q per asset over the given decision days (computed before training)."""
    return ai.q[np.asarray(days)].mean(axis=0)


class AllocatorEnv(gym.Env):
    """
    Training: an episode is ALLOC_EPISODE_BARS // ALLOC_STEP decisions whose days all
    lie in `train_days`; any start day is allowed, so every weekday phase is seen.
    Evaluation: eval_start / eval_end walk decisions every H days over [start, end);
    the last step may be shorter.
    """
    metadata = {"render_modes": []}

    def __init__(self, ai, train_days=None, mu_q=None, eval_start=None, eval_end=None,
                 reward_scale=None, alloc_lambda=None, turnover_pen=None, cost_pct=None,
                 step_days=None, episode_bars=None, caps=True, seed=None):
        super().__init__()
        self.ai = ai
        self.N = ai.N
        self.K = ai.hmm.shape[1]
        self.H = step_days or config.ALLOC_STEP
        self.n_steps = (episode_bars or config.ALLOC_EPISODE_BARS) // self.H
        self.scale = config.ALLOC_REWARD_SCALE if reward_scale is None else reward_scale
        self.lam = config.ALLOC_LAMBDA if alloc_lambda is None else alloc_lambda
        self.pen = config.ALLOC_TURNOVER_PEN if turnover_pen is None else turnover_pen
        self.c = config.COST_PCT if cost_pct is None else cost_pct
        self.caps = caps
        self.base = np.asarray(config.MULTI_BASE_BUDGETS, np.float64)
        self.eval_start, self.eval_end = eval_start, eval_end
        self.mu_q = np.zeros(self.N) if mu_q is None else np.asarray(mu_q, np.float64)
        if eval_start is None:
            days = np.asarray(sorted(train_days))
            allowed = np.zeros(ai.T, dtype=bool)
            allowed[days] = True
            L = self.H * self.n_steps
            ok = [s for s in days if s + L <= ai.T - 1 and allowed[s:s + L].all()
                  and ai.obs_ok(s) and ai.obs_ok(s + L)]      # last obs feeds the buffer
            if not ok:
                raise ValueError("No training episode fits in the given days.")
            self.starts = np.asarray(ok)
            if mu_q is None:
                self.mu_q = mu_q_of(ai, days)
        dim = self.K + 4 + 2 * self.N + 3
        self.observation_space = gym.spaces.Box(-10.0, 10.0, (dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.N + 1,), np.float32)
        self._rng = np.random.default_rng(seed)
        self.t = self.steps = 0

    def _obs(self):
        a, t = self.ai, self.t
        dd = min(max((1.0 - self.eq / self.peak) * config.ALLOC_DD_SCALE, 0.0),
                 config.ALLOC_DD_CLIP)
        return np.concatenate([
            a.hmm[t], a.inputs.ctx[t], a.perf[t],
            np.log(self.beta / self.base) / config.ALLOC_KAPPA,
            [self.g_prev, a.logk[t], dd]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.t = (int(self.eval_start) if self.eval_start is not None
                  else int(self._rng.choice(self.starts)))
        self.steps = 0
        self.beta = self.base.copy()
        self.g_prev = 0.0
        self.eq = self.peak = 1.0
        self.x_prev = self.ai.x_fix[self.t - 1].copy()     # both books start from the
        self.xf_prev = self.x_prev.copy()                  # fixed portfolio's day before
        if not self.ai.obs_ok(self.t) or not np.isfinite(self.x_prev).all():
            raise ValueError(f"Inputs undefined at decision day {self.t}.")
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float64).reshape(-1), -1.0, 1.0)
        u, g = a[:self.N], float(a[self.N])
        beta = budgets_from_action(u, self.base)
        G = float(gross_from_action(g))
        t0 = self.t
        end = t0 + self.H
        if self.eval_end is not None:
            end = min(end, int(self.eval_end))
        d = np.arange(t0, end)
        ai = self.ai
        x = build_positions(ai.inputs.vol[d], ai.inputs.cov[d], m=ai.m[d],
                            beta=np.tile(beta, (len(d), 1)), G=np.full(len(d), G),
                            caps=self.caps)
        xf = ai.x_fix[d] if self.caps else build_positions(
            ai.inputs.vol[d], ai.inputs.cov[d], m=ai.m[d], caps=False)
        rex = ai.inputs.rex[d + 1]
        rho = G * beta / self.base
        drift = float(((rho - 1.0) * self.mu_q).sum())
        active = np.empty(len(d))
        r_x = np.empty(len(d))
        for j in range(len(d)):
            pnl_x = float((x[j] * rex[j]).sum())
            pnl_f = float((xf[j] * rex[j]).sum())
            turn_x = float(np.abs(x[j] - self.x_prev).sum())
            turn_f = float(np.abs(xf[j] - self.xf_prev).sum())
            active[j] = (pnl_x - pnl_f) - drift - self.c * (turn_x - turn_f)
            r_x[j] = pnl_x - self.c * turn_x
            self.eq *= 1.0 + r_x[j]
            self.peak = max(self.peak, self.eq)
            self.x_prev, self.xf_prev = x[j], xf[j]
        A = float(active.sum())
        penalty = self.pen * float(np.abs(beta - self.beta).sum())
        reward = self.scale * (A - 0.5 * self.lam * A * A) - penalty
        self.beta, self.g_prev = beta, g
        self.t = end
        self.steps += 1
        if self.eval_end is not None:
            truncated = self.t >= self.eval_end
        else:
            truncated = self.steps >= self.n_steps
        info = {"t0": t0, "days": len(d), "active": active, "A": A, "penalty": penalty,
                "beta": beta, "G": G, "u": u, "g": g, "x": x, "r_x": r_x}
        obs = self._obs() if not truncated or self.t < ai.T else np.zeros(
            self.observation_space.shape, np.float32)
        return obs, float(reward), False, truncated, info


def _mean_action(models, obs):
    acts = [np.asarray(m.predict(obs, deterministic=True)[0], np.float64).reshape(-1)
            for m in models]
    return np.mean(acts, axis=0)


def run_allocator(models, ai, start, end, mu_q=None, policy=None):
    """
    Walk decisions start .. end − 1 every H days with the ensemble's mean raw action
    (or `policy(obs)`), holding beta and G in between.  Returns per-day arrays:
    x (positions), beta, G, u, decision (bool), and the per-step rewards.
    """
    env = AllocatorEnv(ai, eval_start=start, eval_end=end, mu_q=mu_q)
    obs, _ = env.reset()
    n = end - start
    out = {"x": np.empty((n, ai.N)), "beta": np.empty((n, ai.N)), "G": np.empty(n),
           "u": np.empty((n, ai.N)), "g": np.empty(n), "decision": np.zeros(n, bool),
           "rewards": [], "penalties": []}
    done = False
    while not done:
        act = policy(obs) if policy is not None else _mean_action(models, obs)
        obs, r, _, done, info = env.step(act)
        i0, k = info["t0"] - start, info["days"]
        out["x"][i0:i0 + k] = info["x"]
        out["beta"][i0:i0 + k] = info["beta"]
        out["G"][i0:i0 + k] = info["G"]
        out["u"][i0:i0 + k] = info["u"]
        out["g"][i0:i0 + k] = info["g"]
        out["decision"][i0] = True
        out["rewards"].append(r)
        out["penalties"].append(info["penalty"])
    return out


def load_allocator(model_dirs):
    from stable_baselines3 import SAC
    import os
    return [SAC.load(os.path.join(d, "model.zip"), device="cpu") for d in model_dirs]
