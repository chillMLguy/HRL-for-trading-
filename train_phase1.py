"""
Phase 1 training — fits HMM on training data and saves artifacts.

Usage
-----
  python train_phase1.py --ticker "^DJI" \
      --train_start 2011-01-01 --train_end 2021-12-31 \
      --outdir . --n_states 3 --seed 42

Outputs (under <outdir>/models/phase1/):
  hmm_model.pkl        — fitted, state-sorted HMM
  vol_thresholds.json  — {q33, q67, ann}
  config.json          — all training parameters
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import yfinance as yf

from env.trading_env import AGENT_PRESETS
from regime.hmm_regime import HMMRegimeDetector
from regime.allocators import VolatilityRegimeAllocator, AGENT_ORDER
import config


BARS_PER_YEAR = config.BARS_PER_YEAR


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 — Fit HMM and compute vol thresholds")
    parser.add_argument("--ticker",      default=config.TICKER)
    parser.add_argument("--train_start", default=config.TRAIN_START)
    parser.add_argument("--train_end",   default=config.TRAIN_END)
    parser.add_argument("--outdir",      default=".")
    parser.add_argument("--n_states",    type=int, default=config.N_STATES)
    parser.add_argument("--seed",        type=int, default=config.SEED)
    parser.add_argument("--interval",    default=config.INTERVAL,
                        choices=list(BARS_PER_YEAR.keys()))
    parser.add_argument("--cost_pct",    type=float, default=config.COST_PCT)
    parser.add_argument("--embargo",     type=int, default=config.HMM_EMBARGO,
                        help="Drop last N training obs to reduce "
                             "train→test contamination.")
    parser.add_argument("--val_frac",    type=float,
                        default=config.HMM_VAL_FRAC,
                        help="Hold out last X%% of (training−embargo) "
                             "as a validation slice for log-likelihood "
                             "reporting.")
    parser.add_argument("--no_scaler",   action="store_true",
                        help="Skip StandardScaler on HMM observations.")
    args = parser.parse_args()

    ann = BARS_PER_YEAR[args.interval]
    out_dir = os.path.join(args.outdir, "models", "phase1")
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Download training data ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase 1 Training — {args.ticker}  "
          f"{args.train_start} → {args.train_end}")
    print(f"{'='*60}")

    print(f"\n  Downloading {args.ticker} ...")
    df = yf.download(args.ticker, start=args.train_start,
                     end=args.train_end, interval=args.interval,
                     auto_adjust=True, progress=False)
    prices = df["Close"].squeeze().dropna()
    print(f"  {len(prices)} bars loaded.")

    # ── 2. Verify Phase 0 models exist ────────────────────────────
    print(f"\n  Checking Phase 0 models ...")
    models_dir = os.path.join(args.outdir, "models")
    all_found = True
    for name in AGENT_ORDER:
        p = os.path.join(models_dir, f"{name}_agent", "best_model.zip")
        if os.path.isfile(p):
            print(f"    ✓  {name}_agent/best_model.zip")
        else:
            print(f"    ✗  {name}_agent/best_model.zip  — NOT FOUND")
            all_found = False
    if not all_found:
        print("\n  ⚠  Some Phase 0 models are missing. "
              "Phase 1 evaluation will fail for those agents.\n"
              "  Run train_agents.py first, or pass --outdir pointing "
              "to the directory containing models/.\n")

    # ── 3. Fit HMM ────────────────────────────────────────────────
    print(f"\n  Fitting HMM ({args.n_states} states) ...")
    hmm = HMMRegimeDetector(n_states=args.n_states,
                            random_state=args.seed,
                            use_scaler=not args.no_scaler)
    obs, valid_start = HMMRegimeDetector.build_observations(prices)
    print(f"  Observation matrix: {obs.shape}  "
          f"(valid from price index {valid_start})")

    # Drop the final `embargo` observations to reduce contamination
    # between the end of the training window and the start of the test
    # window, then hold out a validation slice for log-likelihood
    # reporting (no model selection — just a sanity check on the fit).
    embargo = max(0, int(args.embargo))
    n_total = len(obs)
    fit_end = n_total - embargo
    val_size = int(args.val_frac * fit_end)
    val_start = max(1, fit_end - val_size)
    obs_fit = obs[:val_start]
    obs_val = obs[val_start:fit_end]
    print(f"  Embargo: dropping last {embargo} obs "
          f"(of {n_total}) before fit")
    print(f"  Fit slice:  obs[0:{val_start}]   ({len(obs_fit)} rows)")
    print(f"  Val slice:  obs[{val_start}:{fit_end}] ({len(obs_val)} rows)")

    hmm.fit(obs_fit)
    hmm.summary()
    if len(obs_val) > 0:
        val_ll = hmm.score(obs_val)
        print(f"  Validation log-likelihood: {val_ll:.3f}  "
              f"(per-obs: {val_ll / len(obs_val):.4f})")

    hmm_path = os.path.join(out_dir, "hmm_model.pkl")
    hmm.save(hmm_path)
    print(f"  Saved → {hmm_path}")

    # ── 4. Compute volatility-regime thresholds ───────────────────
    print(f"\n  Computing volatility-regime thresholds ...")
    q33, q67 = VolatilityRegimeAllocator.compute_thresholds(prices, ann)
    print(f"  q33 = {q33:.6f}   q67 = {q67:.6f}")

    th_path = os.path.join(out_dir, "vol_thresholds.json")
    with open(th_path, "w") as f:
        json.dump({"q33": q33, "q67": q67, "ann": ann}, f, indent=2)
    print(f"  Saved → {th_path}")

    # ── 5. Save config ────────────────────────────────────────────
    cfg = {
        "ticker": args.ticker,
        "train_start": args.train_start,
        "train_end": args.train_end,
        "interval": args.interval,
        "bars_per_year": ann,
        "cost_pct": args.cost_pct,
        "n_states": args.n_states,
        "seed": args.seed,
        "n_train_bars": len(prices),
        "agent_presets": {k: float(v) for k, v in AGENT_PRESETS.items()},
        "agent_order": AGENT_ORDER,
    }
    cfg_path = os.path.join(out_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Saved → {cfg_path}")

    print(f"\n  Phase 1 training complete.\n")


if __name__ == "__main__":
    main()
