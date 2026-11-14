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

Layer 2 (allocator), on the frozen layer-1 stream:

    default      ALLOC_STEPS = 30,000, ALLOC_TURNOVER_PEN = 0.5
    steps_15k    15,000 steps          steps_60k    60,000 steps
    pen_0.25     ALLOC_TURNOVER_PEN = 0.25          pen_1.0   1.0

scored by the active Sharpe of `full` against `core_spec` on the stream, where each
block is scored by allocators trained without it (train_allocator.crossfit_scores).
The default configuration also scores `core_alloc` against `core`.

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
LAYER2 = [  # name, setting, value, steps, turnover penalty
    ("default", None, None, config.ALLOC_STEPS, config.ALLOC_TURNOVER_PEN),
    ("steps_15k", "ALLOC_STEPS", 15_000, 15_000, config.ALLOC_TURNOVER_PEN),
    ("steps_60k", "ALLOC_STEPS", 60_000, 60_000, config.ALLOC_TURNOVER_PEN),
    ("pen_0.25", "ALLOC_TURNOVER_PEN", 0.25, config.ALLOC_STEPS, 0.25),
    ("pen_1.0", "ALLOC_TURNOVER_PEN", 1.0, config.ALLOC_STEPS, 1.0),
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


def layer1_stream_root(outroot, test_year):
    st = load_settings(outroot)
    fold = os.path.join(outroot, f"fold{test_year}")
    choice = st.get("spec_choice", "default")
    return fold if choice == "default" else os.path.join(fold, "search", choice)


def run_layer2(outroot, test_year, workers, only=None, seeds=None):
    from multi.data import load_universe
    from multi.features import compute_inputs
    from train_allocator import crossfit_scores

    u = load_universe(config.MULTI_DOWNLOAD_START, f"{test_year}-01-01")
    inputs = compute_inputs(u)
    fold = os.path.join(outroot, f"fold{test_year}")
    stream = pd.read_csv(os.path.join(layer1_stream_root(outroot, test_year), "crossfit",
                                      "tilts.csv"), parse_dates=["date"])
    seeds = seeds or config.ALLOC_CF_SEEDS
    rows = []
    for name, setting, value, steps, pen in LAYER2:
        if only and name not in only:
            continue
        rundir = fold if name == "default" else os.path.join(fold, "search", f"alloc_{name}")
        names = ("full", "core_alloc") if name == "default" else ("full",)
        print(f"\n=== allocator {name}: steps {steps:,}, turnover pen {pen} ===", flush=True)
        summary, frames = crossfit_scores(inputs, stream, rundir, test_year, seeds, steps,
                                          names, turnover_pen=pen, workers=workers)
        out = os.path.join(rundir, "alloc_crossfit")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump({"steps": steps, "turnover_pen": pen, "seeds_per_holdout": seeds,
                       **summary}, f, indent=2)
        for nm, fr in frames.items():
            fr.to_csv(os.path.join(out, f"returns_{nm}.csv"), index=False)
        sm = summary["full"]
        row = {"config": name, "setting": setting or "-", "value": value,
               "alloc_steps": steps, "alloc_turnover_pen": pen,
               "active_sharpe": sm["active_sharpe"], "active_ci_lo": sm["ci"][0],
               "active_ci_hi": sm["ci"][1], "mean_budget_change": sm["mean_budget_change"],
               "share_any_u_beyond_095_pct": sm["share_any_u_beyond_095_pct"],
               "mean_G": sm["mean_G"]}
        if "core_alloc" in summary:
            row["core_alloc_active_sharpe"] = summary["core_alloc"]["active_sharpe"]
        rows.append(row)
        for nm, sm in summary.items():
            print(f"  {nm} vs {sm['baseline']}: active Sharpe {sm['active_sharpe']:.3f} "
                  f"[{sm['ci'][0]:.3f}, {sm['ci'][1]:.3f}], blocks "
                  f"{[b['active_sharpe'] for b in sm['per_block']]}", flush=True)
    return pd.DataFrame(rows)


def decide2(table):
    base = float(table.loc[table["config"] == "default", "active_sharpe"].iloc[0])
    others = table[table["config"] != "default"]
    best = others.loc[others["active_sharpe"].idxmax()] if len(others) else None
    if best is not None and best["active_sharpe"] > base + MARGIN:
        return best["config"], int(best["alloc_steps"]), float(best["alloc_turnover_pen"])
    d = table[table["config"] == "default"].iloc[0]
    return "default", int(d["alloc_steps"]), float(d["alloc_turnover_pen"])


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
    p.add_argument("--layer", type=int, choices=[1, 2], default=1)
    p.add_argument("--outroot", default=os.path.join("runs", "multi_wf"))
    p.add_argument("--test_year", type=int, default=config.MULTI_WF_TEST_YEARS[0])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--only", nargs="*", default=None, help="run only these configs")
    args = p.parse_args()

    if args.layer == 2:
        table = run_layer2(args.outroot, args.test_year, args.workers, args.only)
        out_csv = os.path.join(args.outroot, f"fold{args.test_year}", "search_layer2.csv")
        if not args.only:
            table.to_csv(out_csv, index=False)
        pd.set_option("display.width", 200)
        print("\n" + table.to_string(index=False))
        if args.only or len(table) < len(LAYER2):
            print("\nPartial search: settings not frozen.")
            return
        chosen, steps, pen = decide2(table)
        settings = load_settings(args.outroot)
        settings.update({
            "alloc_steps": steps, "alloc_turnover_pen": pen, "alloc_choice": chosen,
            "alloc_rule": f"keep defaults unless a change beats them by more than {MARGIN} "
                          "in held-out-block active Sharpe of full vs core_spec "
                          "(first fold only)",
            "alloc_configs_tried": table["config"].tolist(),
        })
        with open(settings_path(args.outroot), "w") as f:
            json.dump(settings, f, indent=2)
        print(f"\nChosen: {chosen} (steps {steps:,}, turnover pen {pen}) → frozen in "
              f"{settings_path(args.outroot)}")
        return

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
