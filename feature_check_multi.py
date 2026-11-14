"""
Pre-registered check of two candidate specialist features (RESULTS_MULTI.md, "Feature
check"): does the high–low range or volume add predictive information about the next
days' sleeve returns beyond features 0-11?

    python feature_check_multi.py --outdir runs/multi_wf/fold2015/feature_check

No reinforcement learning and no test-year data: a ridge regression on the training
window of the first walk-forward test year (2015), cross-fitted on the same blocks the
specialists use.  For each block the ridge is fitted on the decisions before it, minus
the last `h` (so no target overlaps the block), and predicts the block.  The score is
the correlation between predictions and outcomes over all block days and assets; each
candidate's improvement over the base set gets a paired stationary-bootstrap interval
over dates.  The decision rule is fixed in RESULTS_MULTI.md and applied here.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import config
from common.crossfit import fold_edges
from common.stats import _stationary_bootstrap_indices
from multi.data import load_ohlcv, load_universe, training_window
from multi.features import FEATURE_NAMES, compute_inputs, range_volume_features
from multi.portfolio import sleeve_sizes

N_BOOT, BLOCK = 2000, 20


def targets(b, rex, h):
    """y[t, i] = sum_{j=1..h} b[t+j−1, i] · Rex[t+j, i]   (NaN where undefined)."""
    s = np.full_like(b, np.nan)
    s[:-1] = b[:-1] * rex[1:]                    # s[t] = return of decision t
    y = np.full_like(b, np.nan)
    for t in range(len(b) - h):
        y[t] = s[t:t + h].sum(axis=0)
    return y


def ridge_fit_predict(X_fit, y_fit, X_new, alpha):
    mu, sd = X_fit.mean(axis=0), X_fit.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    A, B = (X_fit - mu) / sd, (X_new - mu) / sd
    yc = y_fit - y_fit.mean()
    beta = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ yc)
    return B @ beta + y_fit.mean()


def crossfit_predictions(X, y, lo, edges, h, alpha):
    """Predictions and demeaned outcomes, both (n_block_days, N), and block days."""
    preds, outs, days = [], [], []
    for b0, b1 in edges:
        b0, b1 = lo + b0, lo + b1
        fit = np.arange(lo, b0 - h)
        blk = np.arange(b0, b1)
        blk = blk[np.isfinite(y[blk]).all(axis=1)]          # target inside the window
        mu = y[fit].mean(axis=0)                              # per-asset mean, fit part
        Xf = X[fit].reshape(-1, X.shape[2])
        yf = (y[fit] - mu).reshape(-1)
        p = ridge_fit_predict(Xf, yf, X[blk].reshape(-1, X.shape[2]), alpha)
        preds.append(p.reshape(len(blk), -1))
        outs.append(y[blk] - mu)
        days.append(blk)
    return np.vstack(preds), np.vstack(outs), np.concatenate(days)


def _corr_from_sums(n, sx, sy, sxx, syy, sxy):
    num = n * sxy - sx * sy
    den = np.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2))
    return num / den


def pooled_corr(p, y):
    return float(np.corrcoef(p.reshape(-1), y.reshape(-1))[0, 1])


def paired_bootstrap(p_base, p_new, y, seed=0):
    """Bootstrap draws of corr(new) − corr(base), resampling whole dates."""
    n_dates, N = y.shape
    idx = _stationary_bootstrap_indices(n_dates, BLOCK, np.random.default_rng(seed),
                                        N_BOOT)
    def draws(p):
        per = [p.sum(1), y.sum(1), (p * p).sum(1), (y * y).sum(1), (p * y).sum(1)]
        s = [q[idx].sum(axis=1) for q in per]
        return _corr_from_sums(N * n_dates, *s)
    return draws(p_new) - draws(p_base)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=os.path.join(
        "runs", "multi_wf", f"fold{config.MULTI_FCHECK_TEST_YEAR}", "feature_check"))
    p.add_argument("--test_year", type=int, default=config.MULTI_FCHECK_TEST_YEAR)
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    u = load_universe(config.MULTI_DOWNLOAD_START, f"{args.test_year}-01-01")
    inp = compute_inputs(u)
    lo, hi = training_window(u.dates, args.test_year, config.MULTI_WARM_BARS)
    extra = range_volume_features(load_ohlcv(u.dates, u.tickers))
    base = inp.feats
    sets = {"base": base,
            "base+range_vol_ratio": np.concatenate([base, extra[:, :, :1]], axis=2),
            "base+abnormal_volume": np.concatenate([base, extra[:, :, 1:]], axis=2)}
    for name, X in sets.items():
        if not np.isfinite(X[lo:hi]).all():
            raise SystemExit(f"{name}: features not defined on the whole window")
    b = sleeve_sizes(inp.vol)
    n_dec = hi - 1 - lo
    edges = fold_edges(n_dec, config.MULTI_CF_FOLDS, config.MULTI_CF_MIN_TRAIN_FRAC)
    print(f"Training window of {args.test_year}: decisions {u.dates[lo].date()} → "
          f"{u.dates[hi - 2].date()} ({n_dec}); blocks start "
          f"{[str(u.dates[lo + e0].date()) for e0, _ in edges]}")

    rows, decision = [], {}
    for h in config.MULTI_FCHECK_HORIZONS:
        y = targets(b, inp.rex, h)
        y[hi - h:] = np.nan                        # targets must end inside the window
        res = {name: crossfit_predictions(X, y, lo, edges, h,
                                          config.MULTI_FCHECK_RIDGE_ALPHA)
               for name, X in sets.items()}
        p_base, y_out, days = res["base"]
        c_base = pooled_corr(p_base, y_out)
        rows.append({"horizon_days": h, "features": "base", "oos_corr": round(c_base, 4),
                     "delta": None, "delta_ci_lo": None, "delta_ci_hi": None,
                     "block_days": len(days)})
        for name in ["base+range_vol_ratio", "base+abnormal_volume"]:
            p_new, y_new, _ = res[name]
            assert np.array_equal(y_new, y_out)
            c_new = pooled_corr(p_new, y_out)
            d = paired_bootstrap(p_base, p_new, y_out)
            lo_ci, hi_ci = np.percentile(d, [2.5, 97.5])
            rows.append({"horizon_days": h, "features": name,
                         "oos_corr": round(c_new, 4), "delta": round(c_new - c_base, 4),
                         "delta_ci_lo": round(float(lo_ci), 4),
                         "delta_ci_hi": round(float(hi_ci), 4),
                         "block_days": len(days)})
            if h == config.MULTI_FCHECK_HORIZONS[0]:
                decision[name.split("+")[1]] = bool(lo_ci > 0)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.outdir, "feature_check.csv"), index=False)
    passed = [f for f, ok in decision.items() if ok]
    out = {"test_year": args.test_year, "decision_horizon_days": config.MULTI_FCHECK_HORIZONS[0],
           "passed": passed, "rule": "improvement over base, 5-day target, 95% interval > 0",
           "base_features": FEATURE_NAMES, "ridge_alpha": config.MULTI_FCHECK_RIDGE_ALPHA,
           "bootstrap": {"draws": N_BOOT, "mean_block_days": BLOCK},
           "window": [str(u.dates[lo].date()), str(u.dates[hi - 2].date())]}
    with open(os.path.join(args.outdir, "feature_check.json"), "w") as f:
        json.dump(out, f, indent=2)
    pd.set_option("display.width", 160)
    print("\n" + table.to_string(index=False))
    print(f"\nPassed the pre-registered rule: {passed or 'none'}")
    print(f"Saved → {args.outdir}")


if __name__ == "__main__":
    main()
