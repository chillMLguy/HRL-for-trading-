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


BARS_PER_YEAR = {"1d": 252, "1h": 1638, "30m": 3276, "15m": 6552}


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 — Fit HMM and compute vol thresholds")
    parser.add_argument("--ticker",      default="^DJI")
    parser.add_argument("--train_start", default="2011-01-01")
    parser.add_argument("--train_end",   default="2021-12-31")
    parser.add_argument("--outdir",      default=".")
    parser.add_argument("--n_states",    type=int, default=3)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--interval",    default="1d",
                        choices=list(BARS_PER_YEAR.keys()))
    parser.add_argument("--cost_pct",    type=float, default=0.0005)
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
                            random_state=args.seed)
    obs, valid_start = HMMRegimeDetector.build_observations(prices)
    print(f"  Observation matrix: {obs.shape}  "
          f"(valid from price index {valid_start})")

    hmm.fit(obs)
    hmm.summary()

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
