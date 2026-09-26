"""
Mechanical correctness tests for the rewritten envs (run: python tests/test_envs.py).
Covers: F1 reward timing, F4 simple-return PnL, R4 caps, F11 random starts,
F8 start_index, shadow books (F2/F3), meta env shapes / cash arm / vol overlay
/ DSR finiteness, and full-long == buy-and-hold equivalence.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from env.trading_env import TradingEnv
from env.shadow import ShadowBooks
from env.meta_env import (MetaTradingEnv, DifferentialSharpe,
                          softmax_weights, ewma_vol_series)

rng = np.random.default_rng(0)
N = 900
prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(3e-4, 0.01, N))))
OK = []


def check(name, cond):
    OK.append((name, bool(cond)))
    print(("  ✓ " if cond else "  ✗ FAIL ") + name)


# ── F1/F4: reward timing & PnL indexing ─────────────────────────────────────
env = TradingEnv(prices, lam_var=0.0, max_pos=1.0, cost_pct=0.0, eval_mode=True)
obs, _ = env.reset()
t0 = env.t
_, r1, *_ , info1 = env.step(np.array([1.0]))
expected = float(prices.iloc[t0 + 1] / prices.iloc[t0] - 1.0)
check("F1: step-k reward pays action-k's next-bar return",
      abs(info1["net_ret"] - expected) < 1e-9)
check("F1: reward == reward_scale * net_ret when lam_var=0, dd=0",
      abs(r1 - env.reward_scale * info1["net_ret"]) < 1e-6)

# full-long equals buy-and-hold (F4)
env = TradingEnv(prices, lam_var=0.0, cost_pct=0.0, eval_mode=True)
env.reset()
start = env.t
done = False
while not done:
    _, _, done, _, info = env.step(np.array([1.0]))
bh = float(prices.iloc[env.t] / prices.iloc[start] - 1.0)
check("F4: always-full-long equity == buy-and-hold",
      abs(info["equity"] - (1 + bh)) < 1e-6)

# ── R4: exposure caps ───────────────────────────────────────────────────────
env = TradingEnv(prices, lam_var=0.0, max_pos=0.3, eval_mode=True)
env.reset()
_, _, _, _, info = env.step(np.array([1.0]))
check("R4: max_pos caps the position", abs(info["position"] - 0.3) < 1e-6)

# ── F11: random episode starts ──────────────────────────────────────────────
env = TradingEnv(prices, train_episode_bars=100, eval_mode=False)
starts = set()
for s in range(8):
    env.reset(seed=s)
    starts.add(env.t)
check("F11: random-start episodes vary the start bar", len(starts) > 3)
env.reset(seed=1)
steps = 0
done = False
while not done:
    _, _, done, _, _ = env.step(np.array([0.0]))
    steps += 1
check("F11: episode length == train_episode_bars", steps == 100)

# ── F8: start_index ─────────────────────────────────────────────────────────
env = TradingEnv(prices, start_index=400, eval_mode=True)
env.reset()
check("F8: trading starts at start_index", env.t == 400)

# ── Shadow books (F2/F3) ────────────────────────────────────────────────────
books = ShadowBooks(2, cost_pct=0.0, max_pos=[1.0, 0.3], start_index=0)
nr = books.step([1.0, 1.0], 0.01)
check("F3: shadow books respect per-agent caps",
      abs(nr[0] - 0.01) < 1e-12 and abs(nr[1] - 0.003) < 1e-12)
check("F2: perf = trailing sum of realized shadow returns",
      np.allclose(books.perf(), [0.01, 0.003]))

# ── DSR ─────────────────────────────────────────────────────────────────────
d = DifferentialSharpe(eta=1 / 60, warmup=5)
vals = [d.step(x) for x in rng.normal(2e-4, 0.01, 300)]
check("R3: DSR finite and bounded", np.all(np.isfinite(vals))
      and np.max(np.abs(vals)) <= 10.0)

# ── softmax / meta env ──────────────────────────────────────────────────────
w = softmax_weights([1, -1, -1, 0], 2.0)
check("R1: softmax weights sum to 1", abs(w.sum() - 1) < 1e-12)

class ConstAgent:
    def __init__(self, a): self.a = float(a)
    def predict(self, obs, deterministic=True):
        return np.array([self.a], dtype=np.float32), None

subs = {"aggressive": ConstAgent(1.0), "balanced": ConstAgent(0.5),
        "conservative": ConstAgent(1.0)}

menv = MetaTradingEnv(prices, subs, hmm_detector=None, cost_pct=0.0,
                      use_cash=True, vol_target=0.10, meta_reward="dsr",
                      eval_mode=True)
obs, _ = menv.reset()
check("meta: obs dim matches space",
      obs.shape[0] == menv.observation_space.shape[0])
check("meta: action dim = N_agents + cash", menv.action_space.shape[0] == 4)

# all weight on cash → executed position 0
_, _, _, _, info = menv.step(np.array([-1, -1, -1, 1], dtype=np.float32))
check("R1: full cash weight → ~zero executed position",
      abs(info["blended"]) < 0.12)

# all weight on conservative (cap 0.3, proposal 1.0)
menv2 = MetaTradingEnv(prices, subs, hmm_detector=None, cost_pct=0.0,
                       use_cash=False, vol_target=0.0, meta_reward="composite",
                       eval_mode=True)
menv2.reset()
_, _, _, _, i2 = menv2.step(np.array([-1, -1, 1], dtype=np.float32))
check("R4: meta blends POSITION proposals (conservative ≈ 0.3·1.0)",
      0.15 < i2["blended"] < 0.40)

# vol overlay levers a low-vol series up to the cap
vol = ewma_vol_series(prices)
check("R2: ewma vol finite after warmup", np.isfinite(vol[100:]).all())
lever_seen = []
menv.reset()
for _ in range(50):
    _, _, _, _, inf = menv.step(np.array([1, -1, -1, -1], dtype=np.float32))
    lever_seen.append(inf["lever"])
check("R2: leverage stays within [0, cap]",
      min(lever_seen) >= 0 and max(lever_seen) <= menv.vol_cap + 1e-9)

# rollout to completion — no NaNs
menv.reset()
done, rets = False, []
while not done:
    a = menv.action_space.sample()
    _, r, done, _, inf = menv.step(a)
    rets.append(inf["net_ret"])
    assert np.isfinite(r)
check("meta: full random rollout finite", np.all(np.isfinite(rets)))

n_fail = sum(1 for _, ok in OK if not ok)
print(f"\n{len(OK) - n_fail}/{len(OK)} checks passed")
sys.exit(1 if n_fail else 0)
