"""
The settings search of brief section 8, on the first walk-forward fold only.

    python search_multi.py --layer 1 --outroot runs/multi_wf

Layer 1 (specialists): the defaults and one change at a time from them,

    default      SPEC_STEPS = 60,000, SPEC_LAMBDA = 10
    steps_30k    30,000 steps          steps_120k   120,000 steps
    lambda_5     SPEC_LAMBDA = 5       lambda_20    SPEC_LAMBDA = 20

each scored by the cross-fitted active Sharpe of core_spec against core on the
cross-fit blocks of the first fold (data before 2015 for test year 2015).  If no
configuration beats the defaults by more than 0.1, the defaults are kept.  The result
is frozen in <outroot>/settings.json for every fold.  The feature configuration was
not run: the pre-registered feature check rejected both candidates.

The default configuration's stream is the fold's real stream (<fold>/crossfit, with
HMM posteriors); the others go to <fold>/search/<name>/crossfit without HMM.  Every
configuration tried is recorded, because the count enters the Deflated Sharpe.
Resumable: finished folds and seeds are reused.
"""
import argparse
import json
import os

import pandas as pd

import config

LAYER1 = [  # name, setting, value, steps, lambda
    ("default", None, None, config.SPEC_STEPS, config.SPEC_LAMBDA),
    ("lambda_5", "SPEC_LAMBDA", 5.0, config.SPEC_STEPS, 5.0),
    ("lambda_20", "SPEC_LAMBDA", 20.0, config.SPEC_STEPS, 20.0),
    ("steps_30k", "SPEC_STEPS", 30_000, 30_000, config.SPEC_LAMBDA),
    ("steps_120k", "SPEC_STEPS", 120_000, 120_000, config.SPEC_LAMBDA),
]
MARGIN = 0.1


def settings_path(outroot):
    return os.path.join(outroot, "settings.json")


def load_settings(outroot):
    """Frozen settings, or the defaults if no search has been run."""
    path = settings_path(outroot)
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {"spec_steps": config.SPEC_STEPS, "spec_lambda": config.SPEC_LAMBDA}


def run_layer1(outroot, test_year, workers, only=None):
    from multi.data import load_universe
    from multi.features import compute_inputs
    from train_crossfit_multi import build_stream, stream_scores

    u = load_universe(config.MULTI_DOWNLOAD_START, f"{test_year}-01-01")
    inputs = compute_inputs(u)
    fold = os.path.join(outroot, f"fold{test_year}")
    rows = []
    for name, setting, value, steps, lam in LAYER1:
        if only and name not in only:
            continue
        rundir = fold if name == "default" else os.path.join(fold, "search", name)
        print(f"\n=== {name}: steps {steps:,}, lambda {lam} ===", flush=True)
        stream = build_stream(inputs, test_year, rundir, steps=steps, spec_lambda=lam,
                              workers=workers, with_hmm=(name == "default"))
        sc, _ = stream_scores(stream, inputs)
        row = {"config": name, "setting": setting or "-", "value": value,
               "spec_steps": steps, "spec_lambda": lam, **sc}
        row["active_ci_lo"], row["active_ci_hi"] = row.pop("spec_active_ci")
        row.pop("trend_active_ci")
        rows.append(row)
        print(f"  cross-fitted active Sharpe {row['spec_active_sharpe']:.3f} "
              f"[{row['active_ci_lo']:.3f}, {row['active_ci_hi']:.3f}], timing only "
              f"{row['spec_timing_only_active_sharpe']:.3f}", flush=True)
    return pd.DataFrame(rows)


def decide(table):
    """The brief's rule: keep the defaults unless a change beats them by more than 0.1."""
    base = float(table.loc[table["config"] == "default", "spec_active_sharpe"].iloc[0])
    others = table[table["config"] != "default"]
    best = others.loc[others["spec_active_sharpe"].idxmax()] if len(others) else None
    if best is not None and best["spec_active_sharpe"] > base + MARGIN:
        return best["config"], int(best["spec_steps"]), float(best["spec_lambda"])
    d = table[table["config"] == "default"].iloc[0]
    return "default", int(d["spec_steps"]), float(d["spec_lambda"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, choices=[1], default=1,
                   help="layer 2 (allocator) arrives with milestone M3")
    p.add_argument("--outroot", default=os.path.join("runs", "multi_wf"))
    p.add_argument("--test_year", type=int, default=config.MULTI_WF_TEST_YEARS[0])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--only", nargs="*", default=None, help="run only these configs")
    args = p.parse_args()

    table = run_layer1(args.outroot, args.test_year, args.workers, args.only)
    out_csv = os.path.join(args.outroot, f"fold{args.test_year}", "search_layer1.csv")
    table.to_csv(out_csv, index=False)
    pd.set_option("display.width", 200)
    print("\n" + table[["config", "spec_steps", "spec_lambda", "spec_active_sharpe",
                        "active_ci_lo", "active_ci_hi", "spec_timing_only_active_sharpe",
                        "spec_mean_tilt"]].to_string(index=False))
    if args.only or len(table) < len(LAYER1):
        print("\nPartial search: settings not frozen.")
        return
    chosen, steps, lam = decide(table)
    settings = load_settings(args.outroot)
    settings.update({
        "spec_steps": steps, "spec_lambda": lam, "spec_choice": chosen,
        "spec_rule": f"keep defaults unless a change beats them by more than {MARGIN} "
                     "in cross-fitted active Sharpe (first fold only)",
        "spec_configs_tried": table["config"].tolist(),
        "chosen_on_test_year": args.test_year,
    })
    with open(settings_path(args.outroot), "w") as f:
        json.dump(settings, f, indent=2)
    print(f"\nChosen: {chosen} (steps {steps:,}, lambda {lam}) → frozen in "
          f"{settings_path(args.outroot)}")


if __name__ == "__main__":
    main()
