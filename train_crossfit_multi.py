"""
The cross-fitted tilt stream of one test year (brief section 8, stage 3).

    python train_crossfit_multi.py --outdir runs/multi_wf/fold2015 --test_year 2015

The training window's decisions are cut into consecutive blocks with
common.crossfit.fold_edges(n, folds, min_train_frac).  For each block, specialists
(several seeds) and an HMM are fitted on the bars before the block only, and their
tilts and regime posteriors are recorded on the block:

    decisions:  [ history ............ | block 1 | block 2 | block 3 | block 4 ]
    fold j fits on the bars before block j and records block j

The blocks joined are crossfit/tilts.csv (date, fold, a_<ticker>, hmm_<k>), the
out-of-sample behaviour the allocator trains on.  Only decisions whose return is
realised inside the training window are in the stream, so its last day is the
second-to-last trading day of the year before the test year.

The HMM follows train_phase1.py, applied to the core portfolio's equity curve:
build_observations(E), drop the last HMM_EMBARGO observations, fit, filtered_proba.

Resumable: a fold whose tilts.csv exists is skipped, and so is every finished seed.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import config
from common.crossfit import fold_edges


def stream_blocks(lo, hi, folds, min_train_frac):
    """Absolute decision bars [(b0, b1), ...] of the blocks of the window [lo, hi)."""
    n_dec = hi - 1 - lo                          # decisions with a return in the window
    return [(lo + a, lo + b) for a, b in fold_edges(n_dec, folds, min_train_frac)]


def core_equity(inputs, lo, end_bar):
    """Core equity curve E on bars lo+1 .. end_bar (returns of decisions lo .. end_bar−1)."""
    from multi.portfolio import build_positions, net_returns
    r = net_returns(build_positions(inputs.vol, inputs.cov), inputs.rex, lo, end_bar)["r"]
    return pd.Series(np.cumprod(1.0 + r), index=inputs.dates[lo + 1:end_bar + 1])


def fit_core_hmm(inputs, lo, fit_hi):
    """HMM fitted on the core's equity curve over the bars [lo, fit_hi) only."""
    from regime.hmm_regime import HMMRegimeDetector
    obs, _ = HMMRegimeDetector.build_observations(core_equity(inputs, lo, fit_hi - 1))
    det = HMMRegimeDetector(n_states=config.N_STATES)
    det.fit(obs[:len(obs) - config.HMM_EMBARGO])
    return det


def hmm_posteriors(det, inputs, lo, b0, b1):
    """Filtered (causal) posteriors on decision bars b0 .. b1−1, shape (b1 − b0, K)."""
    from regime.hmm_regime import HMMRegimeDetector
    E = core_equity(inputs, lo, b1 - 1)
    obs, valid = HMMRegimeDetector.build_observations(E)
    post = det.filtered_proba(obs)
    first_bar = lo + 1 + valid                   # bar of obs row 0
    if b0 < first_bar:
        raise ValueError("Block starts before the HMM has observations.")
    return post[b0 - first_bar:b1 - first_bar]


def stream_frame(inputs, b0, b1, fold, tilts, post=None):
    df = pd.DataFrame({"date": inputs.dates[b0:b1], "fold": fold})
    for i, t in enumerate(inputs.tickers):
        df[f"a_{t}"] = tilts[:, i]
    if post is not None:
        for k in range(post.shape[1]):
            df[f"hmm_{k}"] = post[:, k]
    return df


def build_stream(inputs, test_year, rundir, folds=None, seeds=None, steps=None,
                 mode=None, spec_lambda=None, workers=2, with_hmm=True):
    """Train every fold's specialists, record the stream, write crossfit/tilts.csv."""
    from multi.data import training_window
    from multi.spec_env import load_ensemble, specialist_tilts
    from train_specialists import train_jobs

    folds = folds or config.MULTI_CF_FOLDS
    seeds = seeds or config.MULTI_CF_SEEDS
    steps = steps or config.SPEC_STEPS
    mode = mode or config.SPEC_MODE
    lam = config.SPEC_LAMBDA if spec_lambda is None else spec_lambda
    lo, hi = training_window(inputs.dates, test_year, config.MULTI_WARM_BARS)
    blocks = stream_blocks(lo, hi, folds, config.MULTI_CF_MIN_TRAIN_FRAC)
    root = os.path.join(rundir, "crossfit")
    os.makedirs(root, exist_ok=True)
    d = inputs.dates

    specs, todo = [], []
    for j, (b0, b1) in enumerate(blocks, start=1):
        fold_dir = os.path.join(root, f"fold{j}")
        if os.path.isfile(os.path.join(fold_dir, "tilts.csv")):
            continue
        todo.append(j)
        for k in range(seeds):
            specs.append((lo, b0, 1000 * test_year + 10 * j + k,
                          os.path.join(fold_dir, "specialists", f"seed{k}")))
    if specs:
        print(f"Cross-fit {test_year}: folds {todo}, {seeds} seeds each", flush=True)
        train_jobs(inputs, specs, steps, mode, lam, workers=workers)

    parts, meta = [], []
    for j, (b0, b1) in enumerate(blocks, start=1):
        fold_dir = os.path.join(root, f"fold{j}")
        part = os.path.join(fold_dir, "tilts.csv")
        info = {"fold": j, "fit_lo": lo, "fit_hi": b0,
                "fit_first_decision": str(d[lo].date()),
                "fit_last_data_bar": str(d[b0 - 1].date()),
                "block_first": str(d[b0].date()), "block_last": str(d[b1 - 1].date()),
                "seeds": [1000 * test_year + 10 * j + k for k in range(seeds)],
                "steps": steps, "mode": mode, "spec_lambda": lam}
        if not os.path.isfile(part):
            dirs = [os.path.join(fold_dir, "specialists", f"seed{k}") for k in range(seeds)]
            tilts = specialist_tilts(load_ensemble(dirs), inputs, b0, b1, mode=mode)
            post = None
            if with_hmm:
                det = fit_core_hmm(inputs, lo, b0)
                det.save(os.path.join(fold_dir, "hmm.pkl"))
                post = hmm_posteriors(det, inputs, lo, b0, b1)
            stream_frame(inputs, b0, b1, j, tilts, post).to_csv(part, index=False)
            with open(os.path.join(fold_dir, "block.json"), "w") as f:
                json.dump(info, f, indent=2)
            print(f"  fold {j}: fit to {info['fit_last_data_bar']}, block "
                  f"{info['block_first']} → {info['block_last']}", flush=True)
        parts.append(pd.read_csv(part, parse_dates=["date"]))
        meta.append(info)

    stream = pd.concat(parts, ignore_index=True)
    stream.to_csv(os.path.join(root, "tilts.csv"), index=False)
    with open(os.path.join(root, "blocks.json"), "w") as f:
        json.dump({"test_year": test_year, "window_first": str(d[lo].date()),
                   "window_last_data_bar": str(d[hi - 1].date()), "blocks": meta},
                  f, indent=2)
    return stream


def stream_tilts(stream, inputs):
    """(first decision bar, tilts array) of a stream, checking its days are contiguous."""
    b0 = int(inputs.dates.get_loc(stream["date"].iloc[0]))
    if not inputs.dates[b0:b0 + len(stream)].equals(pd.DatetimeIndex(stream["date"])):
        raise ValueError("Stream dates are not consecutive trading days.")
    return b0, stream[[f"a_{t}" for t in inputs.tickers]].to_numpy(np.float64)


def stream_scores(stream, inputs):
    """Cross-fitted active Sharpe of core_spec and core_trend against core on the stream days."""
    from common.stats import bootstrap_sharpe_ci
    from multi.features import trend_rule_tilt
    from multi.portfolio import active_sharpe, build_positions, net_returns, tilt_returns
    b0, tilts = stream_tilts(stream, inputs)
    b1 = b0 + len(tilts)
    r_core = net_returns(build_positions(inputs.vol, inputs.cov), inputs.rex, b0, b1)["r"]
    r_spec = tilt_returns(inputs, tilts, b0, b1)["r"]
    r_trend = tilt_returns(inputs, trend_rule_tilt(inputs)[b0:b1], b0, b1)["r"]
    out = {"days": len(tilts), "first": str(inputs.dates[b0].date()),
           "last": str(inputs.dates[b1 - 1].date())}
    for name, r in (("spec", r_spec), ("trend", r_trend)):
        lo, hi = bootstrap_sharpe_ci(r - r_core)
        out[f"{name}_active_sharpe"] = round(active_sharpe(r, r_core), 3)
        out[f"{name}_active_ci"] = [round(lo, 3), round(hi, 3)]
    return out, dict(core=r_core, spec=r_spec, trend=r_trend)


def main():
    from multi.data import load_universe
    from multi.features import compute_inputs

    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True, help="the fold folder, e.g. runs/multi_wf/fold2015")
    p.add_argument("--test_year", type=int, required=True)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--spec_lambda", type=float, default=None)
    p.add_argument("--spec_mode", default=config.SPEC_MODE)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no_hmm", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    u = load_universe(config.MULTI_DOWNLOAD_START, f"{args.test_year}-01-01")
    inputs = compute_inputs(u)
    q = args.quick
    stream = build_stream(
        inputs, args.test_year, args.outdir,
        folds=config.MULTI_CF_FOLDS_QUICK if q else config.MULTI_CF_FOLDS,
        seeds=config.MULTI_CF_SEEDS_QUICK if q else config.MULTI_CF_SEEDS,
        steps=args.steps or (config.SPEC_STEPS_QUICK if q else config.SPEC_STEPS),
        mode=args.spec_mode, spec_lambda=args.spec_lambda, workers=args.workers,
        with_hmm=not args.no_hmm)
    scores, _ = stream_scores(stream, inputs)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
