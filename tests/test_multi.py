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
plus sanity checks on the features.  Tests 3 (environments), 4, 5, 9-12 arrive with
the specialist and allocator environments (milestones M2 and M3).
"""
import os
import sys
import tempfile

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

n_fail = sum(1 for _, ok in OK if not ok)
print(f"\n{len(OK) - n_fail}/{len(OK)} checks passed")
sys.exit(1 if n_fail else 0)
