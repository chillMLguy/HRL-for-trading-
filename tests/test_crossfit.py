"""
Checks for cross-fitted meta training (run: python tests/test_crossfit.py).

The two claims that matter: a fold never fits on the bars it proposes on, and
replaying recorded proposals behaves exactly like querying live agents.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from common.crossfit import fold_edges, load_stream, stream_arrays, stream_windows
from env.meta_env import MetaTradingEnv
from overfit_check import _net, _positions, stream_rows
from regime.allocators import AGENT_ORDER, EqualWeightAllocator
from evaluate_phase1 import run_allocator
import config

OK = []


def check(name, cond):
    OK.append((name, bool(cond)))
    print(("  ✓ " if cond else "  ✗ FAIL ") + name)


rng = np.random.default_rng(0)
N = 1200
dates = pd.bdate_range("2015-01-01", periods=N)
prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(3e-4, 0.01, N))), index=dates)


class ObsAgent:
    """Deterministic agent whose action depends on the observation."""
    def __init__(self, k): self.k = k
    def predict(self, obs, deterministic=True):
        return np.array([np.tanh(self.k * obs[2] * 10 + obs[13])], dtype=np.float32), None


subs = {"aggressive": ObsAgent(1.0), "balanced": ObsAgent(-1.0),
        "conservative": ObsAgent(0.5)}

# ── Fold boundaries ─────────────────────────────────────────────────────────
edges = fold_edges(N, 4, 0.5)
check("folds: blocks are consecutive", all(edges[i][1] == edges[i + 1][0]
                                           for i in range(len(edges) - 1)))
check("folds: blocks start after the first min_train_frac", edges[0][0] == N // 2)
check("folds: blocks reach the end of the window", edges[-1][1] == N)
check("folds: every block has bars", all(b > a for a, b in edges))
try:
    fold_edges(N, 1, 0.5)
    check("folds: one fold is rejected (nothing left to validate on)", False)
except ValueError:
    check("folds: one fold is rejected (nothing left to validate on)", True)

# train_crossfit.py fits a fold on load_prices(start, end=bar `start` of the block),
# and load_prices excludes `end`, so the block's first bar is never fitted on.
from common.data import load_prices
csv = os.path.join(tempfile.mkdtemp(), "p.csv")
pd.DataFrame({"Date": dates, "Close": prices.values}).to_csv(csv, index=False)
for start, _ in edges:
    fit_end = prices.index[start].strftime("%Y-%m-%d")
    fitted = load_prices("X", "2015-01-01", fit_end, data_csv=csv, verbose=False)
    if fitted.index[-1] >= prices.index[start]:
        check("folds: fit window ends before its block", False)
        break
else:
    check("folds: fit window ends before its block", True)

# ── Recorded proposals behave exactly like live agents ──────────────────────
kw = dict(cost_pct=0.0002, use_cash=True, vol_target=0.10, meta_reward="dsr",
          eval_mode=True, start_index=200)
live = MetaTradingEnv(prices, subs, **kw)
obs_live, _ = live.reset()
recorded = np.full((len(prices), len(AGENT_ORDER)), np.nan)
actions = rng.uniform(-1, 1, size=(400, 4)).astype(np.float32)
obs_seq, rets_live = [obs_live], []
for a in actions:
    recorded[live._env.t] = live._proposals_a
    o, _, done, _, info = live.step(a)
    obs_seq.append(o)
    rets_live.append(info["net_ret"])
recorded[live._env.t] = live._proposals_a

replay = MetaTradingEnv(prices, {}, proposal_stream=recorded, **kw)
obs_replay, _ = replay.reset()
same_obs, rets_replay = [np.allclose(obs_replay, obs_live, atol=1e-6)], []
for i, a in enumerate(actions):
    o, _, _, _, info = replay.step(a)
    same_obs.append(np.allclose(o, obs_seq[i + 1], atol=1e-6))
    rets_replay.append(info["net_ret"])
check("replay: observations match live agents at every step", all(same_obs))
check("replay: net returns match live agents", np.allclose(rets_live, rets_replay, atol=1e-7))

class Boom:
    def predict(self, *a, **k):
        raise AssertionError("a live agent was queried during replay")

try:
    guard = MetaTradingEnv(prices, {n: Boom() for n in AGENT_ORDER},
                           proposal_stream=recorded, **kw)
    guard.reset()
    guard.step(actions[0])
    check("replay: live agents are never queried", True)
except AssertionError:
    check("replay: live agents are never queried", False)

# Recorded HMM posteriors feed the same observation slots
post = rng.dirichlet([1, 1, 1], len(prices))
with_hmm = MetaTradingEnv(prices, {}, proposal_stream=recorded, hmm_stream=post, **kw)
o, _ = with_hmm.reset()
check("replay: recorded posterior is in the observation",
      np.allclose(o[:3], post[with_hmm._env.t], atol=1e-6))
check("replay: obs space includes the posterior",
      o.shape[0] == with_hmm.observation_space.shape[0] == replay.observation_space.shape[0] + 3)
bad = post.copy()
bad[with_hmm._env.t] = np.nan
gap = MetaTradingEnv(prices, {}, proposal_stream=recorded, hmm_stream=bad, **kw)
o, _ = gap.reset()
check("replay: a missing posterior falls back to uniform", np.allclose(o[:3], 1 / 3, atol=1e-6))
try:
    MetaTradingEnv(prices, {}, proposal_stream=recorded[:-1], **kw)
    check("replay: a stream of the wrong length is rejected", False)
except ValueError:
    check("replay: a stream of the wrong length is rejected", True)

# ── Random-start meta episodes ──────────────────────────────────────────────
ep = dict(kw, eval_mode=False, start_index=200)
env = MetaTradingEnv(prices, {}, proposal_stream=recorded, train_episode_bars=100, **ep)
starts = set()
for s in range(8):
    env.reset(seed=s)
    starts.add(env._env.t)
check("episodes: starts vary", len(starts) > 3)
check("episodes: never start before start_index", min(starts) >= 200)
env.reset(seed=3)
first = env._env.t
env.reset(seed=3)
check("episodes: a seed reproduces the start", env._env.t == first)
fixed = MetaTradingEnv(prices, {}, proposal_stream=recorded, **ep)
fixed.reset(seed=1)
a = fixed._env.t
fixed.reset(seed=2)
check("episodes: 0 bars keeps the old fixed path", a == fixed._env.t == 200)

# ── Stream files ────────────────────────────────────────────────────────────
run = tempfile.mkdtemp()
os.makedirs(os.path.join(run, "crossfit"))
rows = []
for k, (lo, hi) in enumerate(edges, start=1):
    for t in range(lo, min(hi, N - 1)):
        rows.append({"date": dates[t], "fold": k,
                     **{f"a_{n}": recorded[t, i] if np.isfinite(recorded[t, i]) else 0.25
                        for i, n in enumerate(AGENT_ORDER)},
                     "hmm_0": 0.5, "hmm_1": 0.3, "hmm_2": 0.2})
pd.DataFrame(rows).to_csv(os.path.join(run, "crossfit", "proposals.csv"), index=False)
stream = load_stream(run)
win = stream_windows(stream, prices, config.WARM_BARS)
tr_slice, tr_part, tr_first = win["train"]
va_slice, va_part, va_first = win["val"]
check("stream: validation is the last fold only", set(va_part["fold"]) == {4})
check("stream: training is every other fold", set(tr_part["fold"]) == {1, 2, 3})
check("stream: training ends before validation starts",
      tr_part.index[-1] < va_part.index[0])
check("stream: trading starts on the slice's first stream bar",
      tr_slice.index[tr_first] == tr_part.index[0]
      and va_slice.index[va_first] == va_part.index[0])
check("stream: warm history comes before trading", tr_first == config.WARM_BARS)
check("stream: the last decision has a next bar to earn on",
      tr_slice.index.get_loc(tr_part.index[-1]) + 1 < len(tr_slice))
props, hmm = stream_arrays(tr_part, tr_slice, AGENT_ORDER)
check("stream: arrays have one row per bar", len(props) == len(hmm) == len(tr_slice))
check("stream: bars before the stream are empty", np.isnan(props[:tr_first]).all())
check("stream: bars of validation are not visible to training",
      np.isnan(props[-1]).all())
train_env = MetaTradingEnv(tr_slice, {}, proposal_stream=props, hmm_stream=hmm,
                           cost_pct=0.0002, use_cash=True, vol_target=0.10,
                           meta_reward="dsr", start_index=tr_first,
                           train_episode_bars=100)
train_env.reset(seed=0)
done, n = False, 0
while not done:
    _, r, done, _, info = train_env.step(train_env.action_space.sample())
    n += 1
    assert np.isfinite(r)
check("stream: a full training episode runs on recorded data", n <= 100)

# ── Equal weight on recorded proposals matches the Phase 1 allocator ────────
class Replay:
    """Live-agent stand-in that plays back one recorded action per call."""
    def __init__(self, seq): self.seq, self.i = seq, 0
    def predict(self, obs, deterministic=True):
        v = self.seq[self.i]
        self.i += 1
        return np.array([v], dtype=np.float32), None


cost = 0.0002
for window, (slice_, part, first_bar) in win.items():
    actions_w = part[[f"a_{a}" for a in AGENT_ORDER]].values
    agents = {a: Replay(np.r_[actions_w[:, i], actions_w[-1:, i]])
              for i, a in enumerate(AGENT_ORDER)}
    ew = run_allocator(EqualWeightAllocator(), agents, slice_, 252, cost, None,
                       start_index=first_bar)
    table = pd.DataFrame(stream_rows(stream, prices, cost, 252))
    sharpe = table[(table.model == "equal_weight") & (table.window == window)].sharpe.iloc[0]
    r = np.asarray(ew["returns"], dtype=np.float64)
    expected = np.sqrt(252) * r.mean() / r.std()
    check(f"overfit check: equal weight on {window} stream matches run_allocator",
          abs(sharpe - expected) < 0.02)
table = pd.DataFrame(stream_rows(stream, prices, cost, 252))
check("overfit check: each agent gets an out-of-sample stream row",
      set(table[table.window == "stream"].model) == set(AGENT_ORDER))
check("overfit check: stream rows are labelled out of sample",
      set(table.proposals) == {"out_of_sample"})
pos = _positions(np.array([[1.0, 1.0, 1.0]]))
check("overfit check: positions use each agent's cap",
      np.allclose(pos, [[config.AGENT_PRESETS[a]["max_pos"] for a in AGENT_ORDER]]))
check("overfit check: costs are charged on position changes",
      np.allclose(_net(np.array([1.0, 1.0, 0.0]), np.array([0.01, 0.01, 0.01]), 0.001),
                  [0.01 - 0.001, 0.01, -0.001]))

n_fail = sum(1 for _, ok in OK if not ok)
print(f"\n{len(OK) - n_fail}/{len(OK)} checks passed")
sys.exit(1 if n_fail else 0)
