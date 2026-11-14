"""
Mechanical tests for the multi-asset pipeline (run: python tests/test_multi.py).

Numbers refer to the test table in section 11 of the implementation brief.  All
tests use synthetic prices and run in seconds.

Milestone M1 (data and core):
   1  data alignment          common dates only, no forward-filled prices, Rex = R − rf
   2  no look-ahead           changing prices after day t moves nothing up to t
   3  neutral identity        (core part) neutral inputs reproduce the core exactly
   6  costs                   c × sum|Δx|, first day pays for the move from zero
   7  caps                    one common scale factor, cap holds exactly
   8  volatility target       realised vol of the core within 15% of PORT_VOL
  13  benchmarks              SPY at weight 1, equal capital weights, 60/40, spy_vt
plus sanity checks on the features.

Milestone M2 (specialists):
   3  neutral identity        SpecialistEnv with action 0 earns reward 0
   4  reward timing           step t pays the return from t to t+1, inside the fit window
   5  constant action         a constant tilt earns nothing on average over the fit window
  11  ensemble                one model = its action, several = their mean, reproducible
  12  cross-fit dates         every stream row comes from models fitted before its block

Milestone M3 (allocator):
   3  neutral identity        the zero action gives the fixed-budget positions, reward 0
   4  reward timing           one step pays the H days' returns from d to d+1
   5  constant action         a constant non-zero action earns nothing on average
   9  budget mapping          beta_base at zero, sums to 1, ratios in [1/2.25, 2.25]
  10  weekly hold             beta and G change only every H trading days
"""
import json
import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config
from multi.data import (excess_returns, first_common_date, load_prices_wide,
                        load_riskfree_known, load_universe, universe_from_frames)
from multi.features import compute_inputs, trend_rule_tilt, FEATURE_NAMES
from multi.portfolio import (build_positions, ex_ante_vol, fixed_weights,
                             net_returns, rule_based_positions, sleeve_sizes)

TICKERS = list(config.MULTI_TICKERS)
N = len(TICKERS)
OK = []


def check(name, cond):
    OK.append((name, bool(cond)))
    print(("  ✓ " if cond else "  ✗ FAIL ") + name)


def max_abs_diff(a, b):
    """Largest |a − b| where both are finite; inf if the NaN patterns differ."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return np.inf
    m = ~np.isnan(a)
    return float(np.abs(a[m] - b[m]).max()) if m.any() else 0.0


VOLS = np.array([0.16, 0.18, 0.24, 0.07, 0.14, 0.16, 0.20, 0.22])
LOADS = np.array([0.8, 0.8, 0.8, -0.3, -0.3, 0.1, 0.4, 0.7])


def true_cov(vols=VOLS, loads=LOADS):
    """Annualised covariance from a one-factor correlation structure."""
    C = np.outer(loads, loads)
    np.fill_diagonal(C, 1.0)
    return C * np.outer(vols, vols)


def synthetic_prices(n, seed, start="2010-01-04", drift=2e-4):
    rng = np.random.default_rng(seed)
    R = rng.multivariate_normal(np.full(N, drift), true_cov() / 252, size=n)
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame(100.0 * np.cumprod(1.0 + R, axis=0), index=dates,
                        columns=TICKERS)


def synthetic_irx(dates, seed):
    rng = np.random.default_rng(seed)
    y = np.clip(2.0 + np.cumsum(rng.normal(0, 0.03, len(dates))), 0.0, None)
    return pd.Series(y, index=dates)


# ── 1. Data alignment ───────────────────────────────────────────────────────
print("1. data alignment")
tmp = tempfile.mkdtemp()
raw = synthetic_prices(400, seed=1)
raw.iloc[:37, TICKERS.index("DBC")] = np.nan          # DBC starts later
gaps = [100, 101, 250]                                # EEM misses three days
raw.iloc[gaps, TICKERS.index("EEM")] = np.nan
price_csv = os.path.join(tmp, "multi_prices.csv")
raw.rename_axis("Date").reset_index().to_csv(price_csv, index=False)

irx_raw = synthetic_irx(raw.index, seed=2)
irx_raw = irx_raw.drop(irx_raw.index[[0, 5, 6, 7, 120]])   # gaps in the yield
early = pd.Series([1.5], index=[raw.index[0] - pd.Timedelta(days=7)])
irx_raw = pd.concat([early, irx_raw])
rf_csv = os.path.join(tmp, "riskfree.csv")
irx_raw.rename(config.RISKFREE_TICKER).rename_axis("Date").reset_index().to_csv(
    rf_csv, index=False)

u = load_universe("2000-01-01", "2100-01-01", data_csv=price_csv, riskfree_csv=rf_csv)
expected_idx = raw.dropna(how="any").index
check("1: eight series share one index", list(u.prices.columns) == TICKERS
      and u.prices.index.equals(expected_idx))
check("1: no missing values after alignment", not u.prices.isna().any().any())
check("1: dates missing for one ticker are dropped, not filled",
      not any(raw.index[g] in u.prices.index for g in gaps))
check("1: first common date is the latest start",
      first_common_date(price_csv) == raw.index[37] == u.dates[0])

# rf[t] = last yield known at t−1, / 100 / 252, yields forward-filled only
exp_rf = [np.nan]
for t in range(1, len(u.dates)):
    known = irx_raw[irx_raw.index <= u.dates[t - 1]]
    exp_rf.append(known.iloc[-1] / 100.0 / 252)
check("1: rf[t+1] = IRX[t] / 100 / 252 with the yield forward-filled",
      max_abs_diff(u.rf.values, np.array(exp_rf)) < 1e-15)
R_manual = raw.loc[u.dates].values[1:] / raw.loc[u.dates].values[:-1] - 1.0
check("1: R is the simple return between common dates",
      max_abs_diff(u.R.values[1:], R_manual) < 1e-12)
check("1: Rex = R − rf to 1e-12",
      max_abs_diff(u.Rex.values, u.R.values - u.rf.values[:, None]) < 1e-12)
zero = net_returns(np.zeros((len(u.dates), N)), u.Rex.values, 1, len(u.dates) - 1)
check("1: all-zero positions earn exactly 0", np.all(zero["r"] == 0.0))
try:
    load_riskfree_known(u.dates, riskfree_csv=os.path.join(tmp, "missing.csv"))
    check("1: missing risk-free cache is an error, never zero", False)
except FileNotFoundError:
    check("1: missing risk-free cache is an error, never zero", True)
try:
    load_prices_wide("2000-01-01", "2100-01-01", data_csv=os.path.join(tmp, "no.csv"))
    check("1: missing price cache is an error (no yfinance fallback)", False)
except FileNotFoundError:
    check("1: missing price cache is an error (no yfinance fallback)", True)

# yfinance output saved with .to_csv → the cache files
import download_multi as dm
yf_cols = pd.MultiIndex.from_product([["Close", "High", "Low", "Open", "Volume"],
                                      sorted(TICKERS + ["^IRX"])], names=["Price", "Ticker"])
yf_raw = pd.DataFrame(np.nan, index=raw.index.rename("Date"), columns=yf_cols)
for tk in TICKERS:
    for fld in ["Close", "High", "Low", "Open"]:
        yf_raw[(fld, tk)] = raw[tk]
    yf_raw[("Volume", tk)] = 1e6
yf_raw[("Close", "^IRX")] = synthetic_irx(raw.index, seed=2)
raw_csv = os.path.join(tmp, "yf_raw.csv")
yf_raw.to_csv(raw_csv)
p_split, o_split, i_split = dm.split_raw(dm.read_raw(raw_csv), TICKERS, "^IRX")
check("1: a saved yf.download CSV splits into closes, OHLCV and the yield",
      list(p_split.columns) == TICKERS
      and max_abs_diff(p_split.values, raw.dropna(how="all").values) < 1e-9
      and set(o_split["Ticker"]) == set(TICKERS) and len(i_split) == len(raw))
try:
    dm.split_raw(dm.read_raw(raw_csv).drop(columns="DBC", level=1), TICKERS, "^IRX")
    check("1: a ticker missing from the download is an error", False)
except ValueError:
    check("1: a ticker missing from the download is an error", True)

# ── features: sanity ────────────────────────────────────────────────────────
print("features")
P = synthetic_prices(900, seed=3)
irx = synthetic_irx(P.index, seed=4)
U = universe_from_frames(P, irx)
INP = compute_inputs(U)
W = config.MULTI_WARM_BARS
f = INP.feats[W:]
check("features: warm within MULTI_WARM_BARS", INP.first_warm_index() <= W)
check("features: finite after warm-up", np.isfinite(f).all())
check("features: twelve market features", INP.feats.shape == (900, N, 12)
      and len(FEATURE_NAMES) == 12)
check("features: z-scores within ±3", np.abs(f[:, :, :6]).max() <= config.MULTI_Z_CLIP)
check("features: log ratios within ±2", np.abs(f[:, :, 6:9]).max() <= config.MULTI_LOG_CLIP)
check("features: correlations within [−1, 1]", np.abs(f[:, :, 9:11]).max() <= 1 + 1e-12)
check("features: breadth within [−1, 1]", np.abs(f[:, :, 11]).max() <= 1.0)
check("features: 8-11 identical across assets",
      np.all(INP.feats[W:, :, 8:12] == INP.feats[W:, :1, 8:12]))
check("features: vol floored", np.nanmin(INP.vol) >= config.MULTI_VOL_FLOOR)
t = 500
z63 = (np.log(P.values[t]) - np.log(P.values[t - 63])) / (INP.vol[t] / np.sqrt(252) * np.sqrt(63))
check("features: z(63) matches its definition",
      max_abs_diff(INP.z(63)[t], np.clip(z63, -3, 3)) < 1e-12)
check("features: covariance is symmetric",
      max_abs_diff(INP.cov[W:], np.transpose(INP.cov[W:], (0, 2, 1))) < 1e-14)

# ── 2. No look-ahead ────────────────────────────────────────────────────────
print("2. no look-ahead")
rng = np.random.default_rng(5)
T = len(P)
m_rand = rng.uniform(0, 2, (T, N))
u_rand = rng.uniform(-1, 1, (T, N))
beta_rand = np.exp(config.ALLOC_KAPPA * u_rand) / 8
beta_rand /= beta_rand.sum(axis=1, keepdims=True)
G_rand = 1 + config.ALLOC_G_RANGE * rng.uniform(-1, 1, T)


def everything(inputs):
    tt = trend_rule_tilt(inputs)
    pos = rule_based_positions(inputs, tt)
    out = {"vol": inputs.vol, "cov": inputs.cov, "feats": inputs.feats,
           "ctx": inputs.ctx, "rex": inputs.rex, "trend_tilt": tt,
           "x_tilted": build_positions(inputs.vol, inputs.cov, m=m_rand,
                                       beta=beta_rand, G=G_rand)}
    out.update({f"x_{k}": x for k, (x, _) in pos.items()})
    return out


base_out = everything(INP)
for cut in rng.integers(320, 880, size=5):
    P2 = P.copy()
    P2.iloc[cut + 1:] *= np.exp(rng.normal(0, 0.05, (T - cut - 1, N)))
    irx2 = irx.copy()
    irx2.iloc[cut + 1:] += 3.0                  # changes rf from row cut + 2 on
    new_out = everything(compute_inputs(universe_from_frames(P2, irx2)))
    moved = [k for k in base_out
             if max_abs_diff(base_out[k][:cut + 1], new_out[k][:cut + 1]) > 1e-12]
    check(f"2: nothing up to day {cut} moves when later prices change"
          + (f" (moved: {moved})" if moved else ""), not moved)
    check(f"2: the later prices did change the inputs after day {cut}",
          max_abs_diff(base_out["vol"][cut + 1:], new_out["vol"][cut + 1:]) > 0)

# ── 3. Neutral identity (core part) ─────────────────────────────────────────
print("3. neutral identity (core)")
x_core, info_core = build_positions(INP.vol, INP.cov, return_info=True)
base = np.asarray(config.MULTI_BASE_BUDGETS)
x_neutral = build_positions(INP.vol, INP.cov, m=np.ones((T, N)),
                            beta=np.tile(base, (T, 1)), G=np.ones(T))
check("3: neutral inputs give the core's positions to 1e-12",
      max_abs_diff(x_neutral, x_core) < 1e-12)
r_core = net_returns(x_core, INP.rex, W, T - 1)["r"]
r_neutral = net_returns(x_neutral, INP.rex, W, T - 1)["r"]
check("3: neutral inputs give the core's returns to 1e-12",
      max_abs_diff(r_neutral, r_core) < 1e-12)
w_core = base * sleeve_sizes(INP.vol)
check("3: core = k · beta_base · b with k from the core's own ex-ante vol",
      max_abs_diff(x_core[W:], (config.PORT_VOL / ex_ante_vol(w_core, INP.cov))[W:, None]
                   * w_core[W:]) < 1e-12)
check("3: tilts are not renormalised back to PORT_VOL (m = 0.5 halves risk)",
      max_abs_diff(ex_ante_vol(build_positions(INP.vol, INP.cov, m=np.full((T, N), 0.5)),
                               INP.cov)[W:], 0.5 * ex_ante_vol(x_core, INP.cov)[W:]) < 1e-12)

# ── 6. Costs ────────────────────────────────────────────────────────────────
print("6. costs")
x_rand = rng.uniform(-1, 2, (T, N))
c = config.COST_PCT
res = net_returns(x_rand, INP.rex, W, T - 1, cost_pct=c)
res0 = net_returns(x_rand, INP.rex, W, T - 1, cost_pct=0.0)
dx = np.abs(np.diff(x_rand[W - 1:T - 1], axis=0)).sum(axis=1)
dx[0] = np.abs(x_rand[W]).sum()
check("6: cost = c × sum of absolute position changes",
      max_abs_diff(res["cost"], c * dx) < 1e-15)
check("6: the first day pays for the move from zero",
      abs(res["cost"][0] - c * np.abs(x_rand[W]).sum()) < 1e-15)
check("6: net = gross P&L − cost", max_abs_diff(res["r"], res0["r"] - res["cost"]) < 1e-15)
pos = rule_based_positions(INP, trend_rule_tilt(INP))
all_costed = True
for name, (x, _) in pos.items():
    a = net_returns(x, INP.rex, W, T - 1)
    b = net_returns(x, INP.rex, W, T - 1, cost_pct=0.0)
    all_costed &= (a["cost"][0] > 0) and max_abs_diff(b["r"] - a["r"], a["cost"]) < 1e-15
check("6: every rule-based series carries its costs", all_costed)

# ── 7. Caps ─────────────────────────────────────────────────────────────────
print("7. caps")


def flat_cov(vols, rho, T=3):
    """Unequal vols, one correlation: positions differ across assets."""
    vols = np.asarray(vols, dtype=np.float64)
    C = np.full((N, N), rho)
    np.fill_diagonal(C, 1.0)
    S = C * np.outer(vols, vols)
    return np.tile(vols, (T, 1)), np.tile(S, (T, 1, 1))


def common_factor(x, x_raw):
    ratio = x / x_raw
    return np.abs(ratio - ratio[:, :1]).max() < 1e-14, ratio[:, 0]


vol7, cov7 = flat_cov(np.linspace(0.20, 0.40, N), 0.5)    # vol cap only
kw = dict(m=np.full((3, N), 2.0), G=np.full(3, 1.25))
x7, i7 = build_positions(vol7, cov7, return_info=True, **kw)
same, fac = common_factor(x7, i7["x_uncapped"])
check("7: VOL_CAP binds and scales all positions by one factor",
      i7["vol_bind"].all() and not i7["lev_bind"].any() and same and (fac < 1).all())
check("7: VOL_CAP holds exactly", max_abs_diff(ex_ante_vol(x7, cov7), np.full(3, config.VOL_CAP)) < 1e-12)

vol7, cov7 = flat_cov(np.linspace(0.02, 0.05, N), 0.95)   # leverage cap only
x7, i7 = build_positions(vol7, cov7, return_info=True)
same, fac = common_factor(x7, i7["x_uncapped"])
check("7: LEV_MAX binds and scales all positions by one factor",
      i7["lev_bind"].all() and not i7["vol_bind"].any() and same and (fac < 1).all())
check("7: LEV_MAX holds exactly", max_abs_diff(np.abs(x7).sum(axis=1), np.full(3, config.LEV_MAX)) < 1e-12)

x7, i7 = build_positions(vol7, cov7, return_info=True, **kw)   # both caps
same, _ = common_factor(x7, i7["x_uncapped"])
check("7: with both caps binding, still one common factor and both caps hold",
      i7["vol_bind"].all() and i7["lev_bind"].all() and same
      and (ex_ante_vol(x7, cov7) <= config.VOL_CAP + 1e-12).all()
      and max_abs_diff(np.abs(x7).sum(axis=1), np.full(3, config.LEV_MAX)) < 1e-12)
check("7: positions differ across assets in these cases", np.ptp(x7[0]) > 0.01)
check("7: B_MAX caps sleeve sizes",
      np.all(i7["b"][:, 0] == config.B_MAX) and np.all(i7["b"] <= config.B_MAX))

# ── 8. Volatility target ────────────────────────────────────────────────────
print("8. volatility target")
P8 = synthetic_prices(4000, seed=8, drift=0.0)
inp8 = compute_inputs(universe_from_frames(P8, pd.Series(0.0, index=P8.index)))
x8 = build_positions(inp8.vol, inp8.cov)
r8 = net_returns(x8, inp8.rex, W, len(P8) - 1)["r"]
rv = r8.std() * np.sqrt(252)
check(f"8: core realised vol {rv:.4f} within 15% of PORT_VOL {config.PORT_VOL}",
      abs(rv / config.PORT_VOL - 1) < 0.15)

# ── 13. Benchmarks ──────────────────────────────────────────────────────────
print("13. benchmarks")
i_spy, i_ief = TICKERS.index("SPY"), TICKERS.index("IEF")
spy = net_returns(pos["spy"][0], INP.rex, W, T - 1)
exp = INP.rex[W + 1:T, i_spy].copy()
exp[0] -= config.COST_PCT
check("13: SPY at weight 1 earns SPY's excess return minus the initial cost",
      max_abs_diff(spy["r"], exp) < 1e-15)
xe = pos["equal_weight"][0][W:]
check("13: equal_weight holds equal capital weights",
      np.abs(xe - xe[:, :1]).max() < 1e-14 and (xe > 0).all())
xs = pos["sixty_forty"][0][W:]
others = [i for i in range(N) if i not in (i_spy, i_ief)]
check("13: sixty_forty holds SPY : IEF = 60 : 40 and nothing else",
      max_abs_diff(xs[:, i_spy] / xs[:, i_ief], np.full(len(xs), 1.5)) < 1e-12
      and np.all(xs[:, others] == 0))
xv, iv = pos["spy_vt"]
ok = ~(iv["vol_bind"] | iv["lev_bind"])[W:]
check("13: spy_vt holds SPY only, at PORT_VOL / sigma_SPY",
      np.all(xv[W:, others + [i_ief]] == 0)
      and max_abs_diff((xv[W:, i_spy] * np.sqrt(INP.cov[W:, i_spy, i_spy]))[ok],
                       np.full(ok.sum(), config.PORT_VOL)) < 1e-12)
check("13: scaled baselines hit PORT_VOL ex ante when no cap binds",
      all(np.abs(i["ex_ante_vol"][W:][~(i["vol_bind"] | i["lev_bind"])[W:]]
                 - config.PORT_VOL).max() < 1e-12
          for k, (_, i) in pos.items() if k in ("core", "equal_weight", "sixty_forty")))

# ── Milestone M2: specialists ───────────────────────────────────────────────
from multi.spec_env import SpecialistEnv, specialist_tilts, sleeve_returns
from multi.data import training_window

print("3. neutral identity (specialist environment)")
FIT_LO, FIT_HI = 320, 700
env = SpecialistEnv(INP, FIT_LO, FIT_HI, seed=0)
all_zero = True
for ep in range(3):
    env.reset()
    done = False
    while not done:
        _, r, _, done, info = env.step(np.array([0.0]))
        all_zero &= (r == 0.0) and (info["active"] == 0.0)
check("3: SpecialistEnv with action 0 returns reward 0 at every step (costs on)", all_zero)

print("4. reward timing (specialist)")
envt = SpecialistEnv(INP, FIT_LO, FIT_HI, cost_pct=0.0, spec_lambda=0.0, reward_scale=1.0,
                     eval_asset=2, eval_start=400, eval_end=410)
envt.reset()
_, r4, *_ = envt.step(np.array([0.7]))
b4 = sleeve_sizes(INP.vol)
exp4 = 0.7 * (b4[400, 2] * INP.rex[401, 2] - envt.mu_s[2])
check("4: the reward of step t pays the tilt times the sleeve return from t to t+1",
      abs(r4 - exp4) < 1e-15)
starts, lengths, last_ok = [], [], True
for ep in range(20):
    env.reset()
    starts.append(env.t)
    n, done = 0, False
    while not done:
        _, _, _, done, info = env.step(np.array([0.3]))
        n += 1
        last_ok &= info["t"] + 1 <= FIT_HI - 1
    lengths.append(n)
check("4: training episodes are 252 decisions inside the fit window",
      all(L == config.SPEC_EPISODE_BARS for L in lengths)
      and min(starts) >= FIT_LO and max(starts) <= FIT_HI - 1 - config.SPEC_EPISODE_BARS)
check("4: no reward uses a return at or after the end of the fit window", last_ok)

print("5. constant action earns nothing (specialist)")
means = []
for i in range(N):
    e5 = SpecialistEnv(INP, FIT_LO, FIT_HI, cost_pct=0.0, eval_asset=i,
                       eval_start=FIT_LO, eval_end=FIT_HI - 1)
    e5.reset()
    act, done = [], False
    while not done:
        _, _, _, done, info = e5.step(np.array([0.5]))
        act.append(info["active"])
    means.append(np.mean(act))
check(f"5: constant tilt +0.5: mean active over the fit window = 0 (max {max(map(abs, means)):.1e})",
      max(map(abs, means)) < 1e-10)

print("11. ensemble")


class Stub:
    """predict() like SB3: a fixed linear map of the observation."""
    def __init__(self, w):
        self.w = w

    def predict(self, obs, deterministic=True):
        return np.tanh(obs @ self.w)[:, None], None


rng11 = np.random.default_rng(11)
dim = INP.feats.shape[2] + 1 + N
stubs = [Stub(rng11.normal(0, 0.3, dim)) for _ in range(3)]
t1 = specialist_tilts(stubs[:1], INP, 500, 530)
manual, prev = [], np.zeros(N)
for t in range(500 - config.SPEC_INFER_LEADIN, 530):
    obs = np.concatenate([INP.feats[t], prev[:, None], np.eye(N)], axis=1).astype(np.float32)
    a = np.tanh(obs @ stubs[0].w)
    if t >= 500:
        manual.append(a)
    prev = a
check("11: one model: the tilt is its deterministic action, fed back as feature 12",
      max_abs_diff(t1, np.array(manual)) < 1e-6)
t3 = specialist_tilts(stubs, INP, 500, 530)
manual, prev = [], np.zeros(N)
for t in range(500 - config.SPEC_INFER_LEADIN, 530):
    obs = np.concatenate([INP.feats[t], prev[:, None], np.eye(N)], axis=1).astype(np.float32)
    a = np.mean([np.tanh(obs @ s_.w) for s_ in stubs], axis=0)
    if t >= 500:
        manual.append(a)
    prev = a
check("11: several models: the tilt is their mean, and the mean is fed back",
      max_abs_diff(t3, np.array(manual)) < 1e-6)
check("11: inference twice gives identical output",
      np.array_equal(t3, specialist_tilts(stubs, INP, 500, 530)))
from stable_baselines3 import SAC
sac = SAC("MlpPolicy", SpecialistEnv(INP, FIT_LO, FIT_HI), seed=0, device="cpu",
          learning_starts=10_000)
obs0 = np.concatenate([INP.feats[600], np.zeros((N, 1)), np.eye(N)], axis=1).astype(np.float32)
ts = specialist_tilts([sac], INP, 600, 601, leadin=0)
check("11: works with a real SAC model (deterministic action)",
      max_abs_diff(ts[0], sac.predict(obs0, deterministic=True)[0][:, 0]) < 1e-6)

print("12. cross-fit dates")
from train_crossfit_multi import build_stream, stream_scores
P12 = synthetic_prices(1500, seed=12)
inp12 = compute_inputs(universe_from_frames(P12, synthetic_irx(P12.index, seed=13)))
run12 = os.path.join(tmp, "cf12")
stream = build_stream(inp12, 2015, run12, folds=2, seeds=1, steps=300, workers=1)
lo12, hi12 = training_window(inp12.dates, 2015, W)
ok12 = True
for j in sorted(stream["fold"].unique()):
    rows = stream[stream["fold"] == j]
    with open(os.path.join(run12, "crossfit", f"fold{j}", "block.json")) as f:
        blk = json.load(f)
    with open(os.path.join(run12, "crossfit", f"fold{j}", "specialists", "seed0",
                           "config.json")) as f:
        sc = json.load(f)
    fit_end = pd.Timestamp(sc["fit_last_data_bar"])
    ok12 &= fit_end < rows["date"].min() and fit_end == pd.Timestamp(blk["fit_last_data_bar"])
check("12: every row comes from models whose fit window ended before its block", ok12)
check("12: blocks are consecutive and the stream ends where returns leave the window",
      inp12.dates[inp12.dates.get_loc(stream["date"].iloc[0]):hi12 - 1].equals(
          pd.DatetimeIndex(stream["date"])))
check("12: the stream has a_<ticker> and hmm_<k> columns",
      all(f"a_{t}" in stream for t in TICKERS)
      and all(f"hmm_{k}" in stream for k in range(config.N_STATES))
      and np.allclose(stream[[f"hmm_{k}" for k in range(config.N_STATES)]].sum(axis=1), 1))
sc12, _ = stream_scores(stream, inp12)
check("12: rerunning reuses every finished fold (resumable)",
      build_stream(inp12, 2015, run12, folds=2, seeds=1, steps=300, workers=1)
      .equals(stream) and np.isfinite(sc12["spec_active_sharpe"]))

# ── Milestone M3: allocator ─────────────────────────────────────────────────
from multi.alloc_env import (AllocInputs, AllocatorEnv, budgets_from_action,
                             gross_from_action, run_allocator)
from train_allocator import train_days

print("9. budget mapping")
rng9 = np.random.default_rng(9)
base9 = np.asarray(config.MULTI_BASE_BUDGETS)
U = np.vstack([rng9.uniform(-1, 1, (2000, N)), np.eye(N), -np.eye(N),
               np.where(np.eye(N) > 0, 1.0, -1.0), np.where(np.eye(N) > 0, -1.0, 1.0)])
B9 = budgets_from_action(U)
ratio9 = B9 / base9
check("9: the zero action gives beta_base", np.array_equal(budgets_from_action(np.zeros(N)), base9))
check("9: budgets sum to 1", np.abs(B9.sum(axis=1) - 1).max() < 1e-12)
check(f"9: every beta / beta_base in [1/2.25, 2.25] (range {ratio9.min():.4f}–{ratio9.max():.4f})",
      ratio9.min() >= 1 / 2.25 - 1e-12 and ratio9.max() <= 2.25 + 1e-12)
G9 = gross_from_action(rng9.uniform(-1, 1, 1000))
check("9: G in [0.75, 1.25] and G = 1 for g = 0",
      G9.min() >= 0.75 and G9.max() <= 1.25 and gross_from_action(0.0) == 1.0)

# a synthetic stream on INP: tilts and HMM posteriors on bars 320 .. 879
S0, S1 = 320, 880
m_syn = np.ones((T, N))
m_syn[S0:S1] = 1 + rng9.uniform(-1, 1, (S1 - S0, N))
hmm_syn = np.full((T, 3), np.nan)
hmm_syn[S0:S1] = rng9.dirichlet(np.ones(3), S1 - S0)
AI = AllocInputs(INP, m_syn, hmm_syn)

print("3. neutral identity (allocator environment)")
z3 = run_allocator(None, AI, S0, S1, policy=lambda o: np.zeros(N + 1))
check("3: AllocatorEnv with the zero action returns the fixed-budget positions exactly",
      np.array_equal(z3["x"], AI.x_fix[S0:S1]))
check("3: ... and reward 0 at every step (costs and turnover penalty on)",
      all(r == 0.0 for r in z3["rewards"]))
env3 = AllocatorEnv(AI, train_days=np.arange(S0, S1), seed=3)
check("3: observation has 26 numbers for eight assets",
      env3.observation_space.shape == (26,) and env3.action_space.shape == (N + 1,))

print("4. reward timing (allocator)")
env4 = AllocatorEnv(AI, eval_start=500, eval_end=600, mu_q=np.zeros(N), cost_pct=0.0,
                    alloc_lambda=0.0, reward_scale=1.0, turnover_pen=0.0)
env4.reset()
act4 = rng9.uniform(-1, 1, N + 1)
_, r4a, *_ = env4.step(act4)
d4 = np.arange(500, 500 + config.ALLOC_STEP)
x4 = build_positions(INP.vol[d4], INP.cov[d4], m=m_syn[d4],
                     beta=np.tile(budgets_from_action(act4[:N]), (len(d4), 1)),
                     G=np.full(len(d4), gross_from_action(act4[N])))
exp4a = float(((x4 - AI.x_fix[d4]) * INP.rex[d4 + 1]).sum())
check("4: one step pays the H days' returns from d to d+1 (d = t .. t+H−1)",
      abs(r4a - exp4a) < 1e-14)
check("4: one step spans ALLOC_STEP days", env4.t == 500 + config.ALLOC_STEP)

print("5. constant action earns nothing (allocator)")
xf_nc = build_positions(INP.vol, INP.cov, m=m_syn, caps=False)
q_nc = np.full_like(xf_nc, np.nan)
q_nc[:-1] = xf_nc[:-1] * INP.rex[1:]
days5 = np.arange(S0, S1 - 1)
mu5 = q_nc[days5].mean(axis=0)
act5 = np.concatenate([rng9.uniform(-1, 1, N), [0.6]])
env5 = AllocatorEnv(AI, eval_start=S0, eval_end=S1 - 1, mu_q=mu5, cost_pct=0.0, caps=False,
                    turnover_pen=0.0)
env5.reset()
act_all, done = [], False
while not done:
    _, _, _, done, info = env5.step(act5)
    act_all.extend(info["active"])
check(f"5: constant non-zero action: mean active over the fit window = 0 "
      f"(|mean| {abs(np.mean(act_all)):.1e}, {len(act_all)} days)",
      len(act_all) == len(days5) and abs(np.mean(act_all)) < 1e-10)

env4b = AllocatorEnv(AI, eval_start=500, eval_end=600)
env4b.reset()
a1, a2 = rng9.uniform(-1, 1, N + 1), rng9.uniform(-1, 1, N + 1)
o1, *_ = env4b.step(a1)
_, _, _, _, i2 = env4b.step(a2)
b1_, b2_ = budgets_from_action(a1[:N]), budgets_from_action(a2[:N])
check("4: the turnover penalty is TURNOVER_PEN × sum|beta − previous beta|",
      abs(i2["penalty"] - config.ALLOC_TURNOVER_PEN * np.abs(b2_ - b1_).sum()) < 1e-12)
check("4: the observation shows the budgets and g in force",
      np.allclose(o1[3 + 4 + N:3 + 4 + 2 * N], np.log(b1_ / base9) / config.ALLOC_KAPPA, atol=1e-6)
      and abs(o1[3 + 4 + 2 * N] - a1[N]) < 1e-6)

print("10. weekly hold")
z10 = run_allocator(None, AI, 503, 640, policy=lambda o: rng9.uniform(-1, 1, N + 1))
dec10 = np.flatnonzero(z10["decision"])
same = all(np.array_equal(z10["beta"][i], z10["beta"][i - 1]) and z10["G"][i] == z10["G"][i - 1]
           for i in range(1, len(z10["G"])) if not z10["decision"][i])
moved = all(not np.array_equal(z10["beta"][i], z10["beta"][i - 1]) for i in dec10[1:])
check("10: decisions fall every H trading days from the start",
      np.array_equal(dec10, np.arange(0, 640 - 503, config.ALLOC_STEP)))
check("10: beta and G change only on decision days", same and moved)

print("12b. allocator training days")
blocks12 = [(1, 320, 600), (2, 600, 700), (3, 700, 880)]
td = train_days(320, 880, blocks12, holdout=2)
check("12b: a held-out block and the ALLOC_PURGE_AFTER days after it are not trained on",
      not np.any((td >= 600) & (td < 700 + config.ALLOC_PURGE_AFTER))
      and td.min() == 320 and td.max() == 879)
env12 = AllocatorEnv(AI, train_days=td, seed=0)
L12 = config.ALLOC_EPISODE_BARS // config.ALLOC_STEP * config.ALLOC_STEP
check("12b: no training episode crosses the held-out block",
      len(env12.starts) > 0
      and all(not np.any((np.arange(s_, s_ + L12) >= 600) & (np.arange(s_, s_ + L12) < 760))
              for s_ in env12.starts))

n_fail = sum(1 for _, ok in OK if not ok)
print(f"\n{len(OK) - n_fail}/{len(OK)} checks passed")
sys.exit(1 if n_fail else 0)
