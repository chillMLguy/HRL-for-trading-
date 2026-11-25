"""
Checks for R19: network size, weight decay, training length in passes, and the
longer training window.  No real training: SAC runs for a few hundred steps on
synthetic prices.

    python tests/test_training.py
"""
import argparse
import json
import os
import runpy
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch

import config
config.USE_VIX = False
from common.data import load_prices
from common.training import (forward_args, parse_net_arch, sac_policy_kwargs,
                             training_steps)

PASS, FAIL = 0, 0


def check(name, ok):
    global PASS, FAIL
    print(("✓ " if ok else "✗ FAIL ") + name)
    PASS, FAIL = PASS + bool(ok), FAIL + (not ok)


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    return False


# ── defaults ──────────────────────────────────────────────────────────────────
check("defaults: [32,32] networks, AdamW weight decay 0.01, 15 passes",
      config.AGENT_NET_ARCH == [32, 32] and config.META_NET_ARCH == [32, 32]
      and config.WEIGHT_DECAY == 0.01
      and config.AGENT_PASSES == 15 and config.META_PASSES == 15)
check("default training window starts with SPY in 1993; walk-forward expands from it",
      config.TRAIN_START == "1993-02-01" and config.DOWNLOAD_START == "1993-01-01"
      and config.WF_TRAIN_YEARS == 0 and config.WF_TRAIN_START == config.TRAIN_START)

# ── --net_arch parsing ────────────────────────────────────────────────────────
check("--net_arch 32,32 / 64 / [16, 16] / linear",
      parse_net_arch("32,32") == [32, 32] and parse_net_arch("64") == [64]
      and parse_net_arch("[16, 16]") == [16, 16] and parse_net_arch("linear") == [])
check("--net_arch rejects non-numbers and zero sizes",
      raises(lambda: parse_net_arch("big"), argparse.ArgumentTypeError)
      and raises(lambda: parse_net_arch("32,0"), argparse.ArgumentTypeError))

# ── steps from passes ─────────────────────────────────────────────────────────
check("steps = passes × training bars",
      training_steps(3623, 15) == (54345, "15 passes"))
check("a longer window gets proportionally more steps at the same passes",
      training_steps(2 * 3623, 15)[0] == 2 * training_steps(3623, 15)[0])
check("--quick uses QUICK_PASSES, --steps overrides everything",
      training_steps(3623, 15, quick=True)[0] == config.QUICK_PASSES * 3623
      and training_steps(3623, 15, steps=777, quick=True) == (777, "fixed steps"))
check("very short windows get at least MIN_STEPS",
      training_steps(100, 15)[0] == config.MIN_STEPS)
fwd = forward_args(argparse.Namespace(net_arch=[], weight_decay=0.0, passes=7.0,
                                      steps=0, quick=False))
check("linear networks and weight decay 0 forward to child scripts",
      fwd == ["--net_arch", "linear", "--weight_decay", "0.0", "--passes", "7.0"])

# ── SB3 SAC gets the network and AdamW ────────────────────────────────────────
from stable_baselines3 import SAC
from env.trading_env import TradingEnv

rng = np.random.default_rng(0)
N = 700
dates = pd.bdate_range("2010-01-04", periods=N)
prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(3e-4, 0.01, N))), index=dates)
tmp = tempfile.mkdtemp()

for arch in ([8, 8], []):
    env = TradingEnv(prices.iloc[:400], preset="balanced", vix=False)
    model = SAC("MlpPolicy", env, policy_kwargs=sac_policy_kwargs(arch, 0.05),
                learning_starts=10, batch_size=16, buffer_size=200, seed=0, verbose=0)
    model.learn(60)
    opts = [model.actor.optimizer, model.critic.optimizer]
    linears = [m for m in model.actor.latent_pi if isinstance(m, torch.nn.Linear)]
    check(f"SAC {arch or 'linear'}: actor and critics use AdamW with weight decay 0.05",
          all(isinstance(o, torch.optim.AdamW) for o in opts)
          and all(g["weight_decay"] == 0.05 for o in opts for g in o.param_groups))
    check(f"SAC {arch or 'linear'}: hidden layers as asked",
          [m.out_features for m in linears] == arch)
    path = os.path.join(tmp, f"m{len(arch)}")
    model.save(path)
    again = SAC.load(path)
    check(f"SAC {arch or 'linear'}: saves and loads with AdamW",
          isinstance(again.actor.optimizer, torch.optim.AdamW))

# weight decay pulls weights that get no gradient toward zero; plain Adam does not
w = torch.nn.Parameter(torch.ones(4))
opt = torch.optim.AdamW([w], lr=0.1, weight_decay=0.5)
for _ in range(20):
    opt.zero_grad()
    (0.0 * w.sum()).backward()
    opt.step()
check("AdamW weight decay shrinks unused weights", float(w.detach().abs().max()) < 0.5)

# ── train_agents.py records what it did ───────────────────────────────────────
csv = os.path.join(tmp, "prices.csv")
pd.DataFrame({"SPY": prices}).rename_axis("Date").reset_index().to_csv(csv, index=False)
out = os.path.join(tmp, "run")
env = {**os.environ, "HRL_USE_VIX": "0", "OMP_NUM_THREADS": "1"}
r = subprocess.run([sys.executable, "train_agents.py", "--start", "2010-01-01",
                    "--end", "2013-01-01", "--outdir", out, "--agents", "balanced",
                    "--workers", "1", "--no_cnn", "--net_arch", "8,8",
                    "--weight_decay", "0.05", "--steps", "1200", "--data_csv", csv],
                   cwd=ROOT, env=env, capture_output=True, text=True)
cfg_path = os.path.join(out, "models", "balanced_agent", "config.json")
cfg = json.load(open(cfg_path)) if os.path.isfile(cfg_path) else {}
if r.returncode != 0:
    print(r.stdout[-2000:], r.stderr[-2000:])
check("train_agents.py saves network, weight decay, steps and passes in config.json",
      cfg.get("net_arch") == [8, 8] and cfg.get("weight_decay") == 0.05
      and cfg.get("optimizer") == "AdamW" and cfg.get("timesteps") == 1200
      and cfg.get("train_bars") == int(N * config.AGENT_TRAIN_RATIO)
      and abs(cfg.get("passes", 0) - 1200 / cfg.get("train_bars", 1)) < 0.01)
check("the best checkpoint loads",
      os.path.isfile(os.path.join(out, "models", "balanced_agent", "best_model.zip")))

# ── run_all.py / walk_forward.py forward the settings ─────────────────────────
def first_commands(script, argv, n=1):
    """Run a launcher until it starts n child scripts; return their command lines."""
    code = f"""
import sys, subprocess, json
sys.argv = {[script] + argv!r}
seen = []
def stop(cmd, *a, **k):
    seen.append(cmd)
    if len(seen) >= {n}:
        print(json.dumps(seen)); sys.exit(0)
    class R: returncode = 0
    return R()
subprocess.run = stop
import runpy; runpy.run_path({script!r}, run_name='__main__')
"""
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                       text=True, env=env)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        print(r.stdout[-1500:], r.stderr[-1500:])
        return [[]] * n


def value(cmd, flag):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


cmds = first_commands("run_all.py", ["--outdir", os.path.join(tmp, "ra"), "--crossfit",
                                     "--net_arch", "16,16", "--meta_net_arch", "8",
                                     "--weight_decay", "0.02", "--passes0", "7",
                                     "--passes2", "4"], n=4)
agents, crossfit, meta = cmds[0], cmds[2], cmds[3]
check("run_all.py: agents get --net_arch 16,16, --weight_decay 0.02, --passes 7",
      value(agents, "--net_arch") == "16,16" and value(agents, "--weight_decay") == "0.02"
      and value(agents, "--passes") == "7.0")
check("run_all.py: the cross-fit fold agents get the agents' settings",
      value(crossfit, "--net_arch") == "16,16" and value(crossfit, "--passes") == "7.0")
check("run_all.py: the meta-agent gets its own network and passes",
      "train_phase2.py" in meta[1] and value(meta, "--net_arch") == "8"
      and value(meta, "--passes") == "4.0" and value(meta, "--weight_decay") == "0.02")
check("run_all.py defaults: [32,32] everywhere, 15 passes, no fixed steps",
      value(first_commands("run_all.py", ["--outdir", os.path.join(tmp, "rb")])[0],
            "--net_arch") == "32,32"
      and "--steps" not in first_commands("run_all.py",
                                          ["--outdir", os.path.join(tmp, "rb")])[0])

wf = first_commands("walk_forward.py", ["--outroot", os.path.join(tmp, "wf"),
                                        "--years", "2022", "--seeds", "0"])[0]
check("walk_forward.py: folds expand from 1993 by default",
      value(wf, "--start") == "1993-02-01" and value(wf, "--end") == "2021-12-31")
wf8 = first_commands("walk_forward.py", ["--outroot", os.path.join(tmp, "wf8"),
                                         "--years", "2022", "--seeds", "0",
                                         "--train_years", "8", "--steps0", "999"])[0]
check("walk_forward.py --train_years 8 slides; --steps0 fixes the steps",
      value(wf8, "--start") == "2014-01-01" and value(wf8, "--steps") == "999")

# ── meta_seed_study.py copies the run's own meta-agent settings ───────────────
run = os.path.join(tmp, "seedrun")
base = {"training_mode": "crossfit", "train_start": "2010-01-01",
        "train_end": "2012-12-31", "ticker": "SPY", "cost_pct": 2e-5, "seed": 42,
        "use_vix": False, "vix_lag_days": 1, "vix_ticker": "^VIX"}
for sub, extra in (("phase2", {"net_arch": [16, 16], "weight_decay": 0.02,
                               "timesteps": 1234}),
                   ("phase2_leaked", {"net_arch": [64, 64], "timesteps": 999})):
    os.makedirs(os.path.join(run, "models", sub))
    json.dump({**base, **extra}, open(os.path.join(run, "models", sub, "config.json"), "w"))
json.dump({"test_start": "2013-01-01", "test_end": "2013-12-31"},
          open(os.path.join(run, "phase2_eval_config.json"), "w"))
seed_cmds = first_commands("meta_seed_study.py", ["--rundir", run, "--seeds", "0"], n=2)
cf = next((c for c in seed_cmds if "--crossfit" in c), [])
lk = next((c for c in seed_cmds if "--stream_dates" in c), [])
check("meta_seed_study.py: new cross-fitted seeds copy the run's network, decay and steps",
      value(cf, "--net_arch") == "16,16" and value(cf, "--weight_decay") == "0.02"
      and value(cf, "--steps") == "1234")
check("meta_seed_study.py: leaked seeds copy theirs; runs before R19 get Adam (decay 0)",
      value(lk, "--net_arch") == "64,64" and value(lk, "--weight_decay") == "0.0"
      and value(lk, "--steps") == "999")

# ── the longer window needs the longer download ───────────────────────────────
check("load_prices refuses a cache that starts years after the requested start",
      raises(lambda: load_prices("SPY", "1993-02-01", "2013-01-01", data_csv=csv,
                                 verbose=False), ValueError))
check("a start a few days before the first bar is fine (holidays, weekends)",
      not raises(lambda: load_prices("SPY", "2010-01-01", "2013-01-01", data_csv=csv,
                                     verbose=False)))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
