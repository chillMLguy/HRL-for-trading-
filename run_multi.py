"""
One test year of the multi-asset hierarchy end to end (mirrors run_all.py).

    python run_multi.py --outdir runs/multi_wf/fold2015 --test_year 2015
    python run_multi.py --outdir runs/multi_quick --quick      # mechanics only, minutes

Stages, each skipped when its outputs exist:

  1. inputs              prices, the T-bill rate and features (recomputed, seconds)
  2. final specialists   seed ensemble on the whole training window
  3. cross-fit stream    specialists and HMM refitted before each block, recorded on
                         the block → crossfit/tilts.csv
  4. allocators          `full` and `core_alloc` on the stream
  5. evaluation          the test year with warm history (evaluate_multi.py), which
                         also fits the final HMM on the whole training window

Settings come from <settings_root>/settings.json (frozen by search_multi.py on the
first fold) or the defaults in config.py.  --quick uses the reduced steps, seeds and
cross-fit folds of the brief; its numbers check mechanics and are never results.
The seconds each stage took go to <outdir>/timing.json.
"""
import argparse
import glob
import json
import os
import time

import config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--test_year", type=int, default=config.MULTI_WF_TEST_YEARS[0])
    p.add_argument("--settings_root", default=os.path.join("runs", "multi_wf"))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from multi.data import load_universe, training_window
    from multi.features import compute_inputs
    from search_multi import load_settings
    from train_allocator import train_allocators
    from train_crossfit_multi import build_stream
    from train_specialists import train_ensemble
    from evaluate_multi import evaluate

    Y, q = args.test_year, args.quick
    st = load_settings(args.settings_root)
    spec_steps = config.SPEC_STEPS_QUICK if q else st.get("spec_steps", config.SPEC_STEPS)
    spec_lambda = st.get("spec_lambda", config.SPEC_LAMBDA)
    alloc_steps = (config.ALLOC_STEPS_QUICK if q
                   else st.get("alloc_steps", config.ALLOC_STEPS))
    alloc_pen = st.get("alloc_turnover_pen", config.ALLOC_TURNOVER_PEN)
    spec_seeds = config.SPEC_SEEDS_QUICK if q else config.SPEC_SEEDS
    alloc_seeds = config.ALLOC_SEEDS_QUICK if q else config.ALLOC_SEEDS
    cf_folds = config.MULTI_CF_FOLDS_QUICK if q else config.MULTI_CF_FOLDS
    cf_seeds = config.MULTI_CF_SEEDS_QUICK if q else config.MULTI_CF_SEEDS
    os.makedirs(args.outdir, exist_ok=True)
    run_cfg = {"test_year": Y, "quick": q, "spec_steps": spec_steps,
               "spec_lambda": spec_lambda, "spec_seeds": spec_seeds,
               "alloc_steps": alloc_steps, "alloc_turnover_pen": alloc_pen,
               "alloc_seeds": alloc_seeds, "cf_folds": cf_folds, "cf_seeds": cf_seeds,
               "settings_source": st}
    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)
    if q:
        # quick runs keep their own settings, never the frozen ones of the real grid
        with open(os.path.join(args.outdir, "settings.json"), "w") as f:
            json.dump({"spec_steps": spec_steps, "spec_lambda": spec_lambda,
                       "alloc_steps": alloc_steps, "alloc_turnover_pen": alloc_pen,
                       "quick": True}, f, indent=2)

    timing = {}
    path = os.path.join(args.outdir, "timing.json")
    if os.path.isfile(path):
        timing = json.load(open(path))

    def stage(name, fn):
        t = time.time()
        print(f"\n=== {name} ===", flush=True)
        out = fn()
        sec = round(time.time() - t, 1)
        if sec > 5 or name not in timing:     # keep the time of the run that did the work
            timing[name] = sec
        with open(path, "w") as f:
            json.dump(timing, f, indent=2)
        print(f"  {name}: {sec / 60:.1f} min", flush=True)
        return out

    u = stage("1 inputs", lambda: load_universe(config.MULTI_DOWNLOAD_START, f"{Y}-01-01"))
    inputs = compute_inputs(u)
    lo, hi = training_window(inputs.dates, Y, config.MULTI_WARM_BARS)
    stage("2 final specialists", lambda: train_ensemble(
        inputs, lo, hi, os.path.join(args.outdir, "models", "specialists"),
        [1000 * Y + k for k in range(spec_seeds)], spec_steps, spec_lambda=spec_lambda,
        workers=args.workers))
    stream = stage("3 cross-fit stream", lambda: build_stream(
        inputs, Y, args.outdir, folds=cf_folds, seeds=cf_seeds, steps=spec_steps,
        spec_lambda=spec_lambda, workers=args.workers))
    stage("4 allocators", lambda: train_allocators(
        inputs, stream, os.path.join(args.outdir, "models"),
        [1000 * Y + k for k in range(alloc_seeds)], alloc_steps, turnover_pen=alloc_pen,
        workers=args.workers))
    settings_root = args.outdir if q else args.settings_root
    returns, metrics, comps, window, diag = stage(
        "5 evaluation", lambda: evaluate(args.outdir, Y, settings_root))

    import pandas as pd
    pd.set_option("display.width", 220)
    print(f"\nTest year {Y}{' (QUICK: mechanics only)' if q else ''}\n")
    print(metrics.set_index("portfolio")[["sharpe", "ann_ret_pct", "ann_vol_pct",
                                          "max_dd_pct", "turnover_per_year",
                                          "gross_mean"]].to_string())
    print("\n" + comps[["first", "second", "sharpe_diff", "p_value", "active_sharpe"]]
          .to_string(index=False))
    print(f"\nTiming (s): {timing}")


if __name__ == "__main__":
    main()
