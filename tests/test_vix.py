"""
Tests for the VIX inputs of branch spy-vix (run: python tests/test_vix.py).

Covers: load_vix alignment (VIX forward-filled on the asset's calendar, never the
prices; errors instead of silent gaps), the four feature definitions, no look-ahead
(including the one-day VIX lag), the zero-VIX ablation, automatic lookup by date,
the meta-agent's VIX inputs, the saved-settings check, and download_data.py's file
layout (with a fake yfinance).
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config
from common.data import check_vix_settings, load_prices, load_vix, vix_settings
from env.trading_env import N_OBS, VIX_FEATURES, TradingEnv, _vix_features
from env.meta_env import MetaTradingEnv

OK = []


def check(name, cond):
    OK.append((name, bool(cond)))
    print(("  ✓ " if cond else "  ✗ FAIL ") + name)


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except BaseException:
        return False
    return False


rng = np.random.default_rng(7)
N = 700
dates = pd.bdate_range("2012-01-02", periods=N)
prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(3e-4, 0.01, N))), index=dates)
vix = pd.Series(np.exp(np.log(18) + np.cumsum(rng.normal(0, 0.05, N))), index=dates)

tmp = tempfile.mkdtemp()
csv = os.path.join(tmp, "spy_vix.csv")
vix_gappy = vix.drop(vix.index[[0, 100, 101, 400]])          # VIX misses 4 SPY days
vix_gappy = pd.concat([pd.Series([17.0], index=[dates[0] - pd.Timedelta(days=3)]),
                       vix_gappy])                            # and starts earlier
pd.DataFrame({"SPY": prices, "^VIX": vix_gappy}).rename_axis("Date").reset_index() \
    .to_csv(csv, index=False)
config.DATA_CSV = csv
config.USE_VIX = True

# ── load_vix ────────────────────────────────────────────────────────────────
print("load_vix")
v = load_vix(dates)
exp = vix_gappy.reindex(vix_gappy.index.union(dates)).ffill().reindex(dates)
check("VIX on the asset's calendar, forward-filled over its gaps",
      v.index.equals(dates) and np.allclose(v.values, exp.values))
check("a gap takes the last known VIX, not a later one",
      v.iloc[100] == vix.iloc[99] and v.iloc[101] == vix.iloc[99])
check("prices themselves are never filled (load_prices reads SPY as stored)",
      load_prices("SPY", "2000-01-01", "2100-01-01", verbose=False).index.equals(dates))
early = pd.bdate_range("2011-06-01", periods=5)
check("no VIX on or before the first date is an error, never zero",
      raises(lambda: load_vix(early), ValueError))
nocol = os.path.join(tmp, "novix.csv")
pd.DataFrame({"Date": dates, "SPY": prices.values}).to_csv(nocol, index=False)
check("a cache without a VIX column is an error",
      raises(lambda: load_vix(dates, data_csv=nocol), ValueError))
check("a missing cache is an error",
      raises(lambda: load_vix(dates, data_csv=os.path.join(tmp, "none.csv")),
             FileNotFoundError))

# ── feature definitions ─────────────────────────────────────────────────────
print("features")
env0 = TradingEnv(prices, vix=vix.values, eval_mode=True)
f = env0._feat
for lag in (0, 1):
    feats = _vix_features(vix.values, f["vol_20"], lag=lag)
    t = 450
    vt = vix.iloc[t - lag]
    med = np.median(vix.values[t - lag - config.VIX_REL_WINDOW:t - lag])
    ok = (abs(feats["vix_level"][t] - np.clip(np.log(vt / config.VIX_REF_LEVEL), -2, 2)) < 1e-5
          and abs(feats["vix_rel"][t] - np.clip(np.log(vt / med), -2, 2)) < 1e-5
          and abs(feats["vix_chg"][t] - np.clip(np.log(vt / vix.iloc[t - lag - 5]), -2, 2)) < 1e-5
          and abs(feats["vrp"][t] - np.clip(vt / 100 - f["vol_20"][t], -1, 1)) < 1e-5)
    check(f"lag {lag}: vix_level, vix_rel, vix_chg and vrp match their definitions", ok)
check("default lag is config.VIX_LAG_DAYS (1)",
      config.VIX_LAG_DAYS == 1 and abs(f["vix_level"][450]
                                       - np.log(vix.iloc[449] / 20)) < 1e-5)
check("observation has 24 inputs, VIX at 20-23",
      N_OBS == 24 and env0.observation_space.shape == (24,))
env0.reset()
o = env0._get_obs()
check("obs[20:24] are the VIX features at the current bar",
      np.allclose(o[20:24], [f[k][env0.t] for k in VIX_FEATURES]))
check("features finite and inside their clips",
      all(np.isfinite(f[k]).all() for k in VIX_FEATURES)
      and np.abs(f["vix_level"]).max() <= 2 and np.abs(f["vrp"]).max() <= 1)

# ── no look-ahead ───────────────────────────────────────────────────────────
print("no look-ahead")
for cut in rng.integers(300, 650, size=4):
    v2 = vix.values.copy()
    v2[cut + 1:] *= rng.uniform(0.5, 2.0, N - cut - 1)
    g = TradingEnv(prices, vix=v2, eval_mode=True)._feat
    check(f"changing VIX after bar {cut} changes no feature up to {cut}",
          all(np.array_equal(g[k][:cut + 1], f[k][:cut + 1]) for k in VIX_FEATURES))
    v3 = vix.values.copy()
    v3[cut] *= 3.0
    h = TradingEnv(prices, vix=v3, eval_mode=True)._feat
    check(f"with the 1-day lag, VIX of bar {cut} is not used at bar {cut}",
          all(h[k][cut] == f[k][cut] for k in VIX_FEATURES)
          and any(h[k][cut + 1] != f[k][cut + 1] for k in VIX_FEATURES))

# ── ablation and automatic lookup ───────────────────────────────────────────
print("ablation and lookup")
z = TradingEnv(prices, vix=False, eval_mode=True)
check("vix=False gives zero VIX features (same observation size)",
      not z.vix_on and all(not z._feat[k].any() for k in VIX_FEATURES)
      and z.observation_space.shape == (24,))
config.USE_VIX = False
check("config.USE_VIX = False gives zero VIX features",
      not TradingEnv(prices, eval_mode=True).vix_on)
config.USE_VIX = True
auto = TradingEnv(prices, eval_mode=True)
ref = TradingEnv(prices, vix=load_vix(dates).values, eval_mode=True)
check("with USE_VIX, VIX is looked up by the dates of the prices",
      auto.vix_on and all(np.array_equal(auto._feat[k], ref._feat[k]) for k in VIX_FEATURES))
part = prices.iloc[200:500]
sub = TradingEnv(part, eval_mode=True)
check("a slice of the prices looks up the same VIX dates",
      np.array_equal(sub._feat["vix_level"][100:],
                     _vix_features(load_vix(part.index).values,
                                   sub._feat["vol_20"])["vix_level"][100:]))
other = os.path.join(tmp, "other.csv")
vix_other = vix * 1.5
pd.DataFrame({"SPY": prices, "^VIX": vix_other}).rename_axis("Date").reset_index() \
    .to_csv(other, index=False)
p_other = load_prices("SPY", "2000-01-01", "2100-01-01", data_csv=other, verbose=False)
e_other = TradingEnv(p_other.iloc[50:], eval_mode=True)
check("VIX comes from the same file as the prices (--data_csv), even after slicing",
      np.array_equal(e_other._feat["vix_level"],
                     _vix_features(vix_other.values[50:], e_other._feat["vol_20"])["vix_level"]))
check("prices without dates and no vix argument is an error (never silent zeros)",
      raises(lambda: TradingEnv(prices.reset_index(drop=True), eval_mode=True), ValueError))


class ConstAgent:
    def __init__(self, a): self.a = float(a)
    def predict(self, obs, deterministic=True):
        assert obs.shape == (24,)
        return np.array([self.a], dtype=np.float32), None


subs = {"aggressive": ConstAgent(1.0), "balanced": ConstAgent(0.5),
        "conservative": ConstAgent(0.2)}
menv = MetaTradingEnv(prices, subs, hmm_detector=None, eval_mode=True)
mobs, _ = menv.reset()
K = 0
e = menv._env
check("meta-agent observation carries the four VIX inputs after the returns",
      np.allclose(mobs[K + 7:K + 11], [e._feat[k][e.t] for k in VIX_FEATURES]))
menv0 = MetaTradingEnv(prices, subs, hmm_detector=None, eval_mode=True, vix=False)
check("meta vix=False zeroes them, same size",
      menv0.reset()[0].shape == mobs.shape and not menv0.reset()[0][7:11].any())
check("sub-agents see a 24-input observation through their shadow books",
      np.isfinite(menv.step(np.zeros(4, dtype=np.float32))[1]))

# ── saved settings ──────────────────────────────────────────────────────────
print("saved settings")
saved = {"cost_pct": 2e-5, **vix_settings()}
check("matching VIX settings pass", not raises(lambda: check_vix_settings(saved)))
check("a model trained without VIX is refused when USE_VIX is on",
      raises(lambda: check_vix_settings({**saved, "use_vix": False}), SystemExit))
check("a different VIX lag is refused",
      raises(lambda: check_vix_settings({**saved, "vix_lag_days": 0}), SystemExit))
check("configs from before this branch (no VIX keys) are left to SB3's shape check",
      not raises(lambda: check_vix_settings({"cost_pct": 2e-4})))
try:
    check_vix_settings({**saved, "use_vix": False})
    msg = ""
except SystemExit as e:
    msg = str(e)
check("the refusal says how to run a no-VIX model (HRL_USE_VIX=0)", "HRL_USE_VIX=0" in msg)

import json
import subprocess
from common.data import check_vix_dir
adir = os.path.join(tmp, "models", "balanced_agent")
os.makedirs(adir, exist_ok=True)
with open(os.path.join(adir, "config.json"), "w") as f:
    json.dump({**saved, "use_vix": False}, f)
check("agent loaders check the agent's config.json",
      raises(lambda: check_vix_dir(adir), SystemExit)
      and not raises(lambda: check_vix_dir(os.path.join(tmp, "no_such_dir"))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def cfg_in_child(env_extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("HRL_")}
    env.update(env_extra)
    out = subprocess.run([sys.executable, "-c",
                          "import config; print(config.USE_VIX, config.VIX_LAG_DAYS)"],
                         cwd=ROOT, env=env, capture_output=True, text=True)
    return out.stdout.split()
check("by default scripts use VIX with a 1-day lag", cfg_in_child({}) == ["True", "1"])
check("HRL_USE_VIX=0 / HRL_VIX_LAG=0 (set by --no_vix / --vix_lag) reach config.py",
      cfg_in_child({"HRL_USE_VIX": "0", "HRL_VIX_LAG": "0"}) == ["False", "0"])

def flags_reach_children(script, argv):
    """Parse the runner's arguments, stop at its first subprocess, report the env."""
    code = f"""
import os, sys, subprocess
sys.argv = {[script] + argv!r}
def stop(cmd, *a, **k):
    print(os.environ.get('HRL_USE_VIX'), os.environ.get('HRL_VIX_LAG')); sys.exit(0)
subprocess.run = stop
import runpy; runpy.run_path({script!r}, run_name='__main__')
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith("HRL_")}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                         capture_output=True, text=True)
    return out.stdout.split()[-2:]
check("run_all.py --no_vix --vix_lag 0 passes both to every script it starts",
      flags_reach_children("run_all.py", ["--no_vix", "--vix_lag", "0",
                                          "--outdir", os.path.join(tmp, "r")])
      == ["0", "0"])
check("walk_forward.py --no_vix passes it on too",
      flags_reach_children("walk_forward.py", ["--no_vix", "--years", "2024", "--seeds", "0",
                                               "--outroot", os.path.join(tmp, "w")])
      == ["0", "None"])

# ── download_data.py layout (fake yfinance) ─────────────────────────────────
print("download_data.py")
import download_data as dd
cols = pd.MultiIndex.from_product([["Close", "High", "Low", "Open", "Volume"],
                                   ["SPY", "^VIX"]], names=["Price", "Ticker"])
fake = pd.DataFrame(1.0, index=dates, columns=cols)
fake[("Close", "SPY")] = prices.values
fake[("Close", "^VIX")] = vix.values
fake.iloc[3, list(cols).index(("Close", "^VIX"))] = np.nan
dd.yf.download = lambda *a, **k: fake
out = os.path.join(tmp, "dl.csv")
sys.argv = ["download_data.py", "--out", out]
dd.main()
got = pd.read_csv(out, parse_dates=["Date"], index_col="Date")
check("the cache has Date, SPY and ^VIX columns",
      list(got.columns) == ["SPY", "^VIX"] and np.allclose(got["SPY"].values, prices.values))
check("load_vix reads it and fills the one missing VIX close",
      load_vix(dates, data_csv=out).iloc[3] == vix.iloc[2])

n_fail = sum(1 for _, ok in OK if not ok)
print(f"\n{len(OK) - n_fail}/{len(OK)} checks passed")
sys.exit(1 if n_fail else 0)
