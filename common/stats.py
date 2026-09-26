"""
R7 — statistics for honest performance claims.

  * sharpe / sortino / drawdown on pooled OOS daily returns
  * stationary-bootstrap confidence intervals for Sharpe (Politis–Romano)
  * bootstrap test for a Sharpe DIFFERENCE between two strategies (paired)
  * Probabilistic / Deflated Sharpe Ratio (Bailey & López de Prado)
"""
import numpy as np
from scipy import stats as sps

EULER_GAMMA = 0.5772156649015329


def ann_sharpe(r, ann=252):
    r = np.asarray(r, dtype=np.float64)
    s = r.std()
    return float(np.sqrt(ann) * r.mean() / s) if s > 1e-12 else 0.0


def _stationary_bootstrap_indices(n, block, rng):
    """One stationary-bootstrap resample of indices (mean block length =
    `block`)."""
    idx = np.empty(n, dtype=np.int64)
    p = 1.0 / block
    t = rng.integers(0, n)
    for i in range(n):
        idx[i] = t
        if rng.random() < p:
            t = rng.integers(0, n)
        else:
            t = (t + 1) % n
    return idx


def bootstrap_sharpe_ci(r, ann=252, n_boot=2000, block=20, alpha=0.05,
                        seed=0):
    """Percentile CI for the annualized Sharpe under a stationary bootstrap."""
    r = np.asarray(r, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(r)
    sims = np.empty(n_boot)
    for b in range(n_boot):
        sims[b] = ann_sharpe(r[_stationary_bootstrap_indices(n, block, rng)],
                             ann)
    lo, hi = np.percentile(sims, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_sharpe_diff_pvalue(r_a, r_b, ann=252, n_boot=2000, block=20,
                                 seed=0):
    """
    Paired stationary-bootstrap p-value for H0: Sharpe(a) ≤ Sharpe(b).
    Resamples the SAME time indices for both series (preserves dependence).
    """
    r_a = np.asarray(r_a, dtype=np.float64)
    r_b = np.asarray(r_b, dtype=np.float64)
    n = min(len(r_a), len(r_b))
    r_a, r_b = r_a[:n], r_b[:n]
    rng = np.random.default_rng(seed)
    obs = ann_sharpe(r_a, ann) - ann_sharpe(r_b, ann)
    count = 0
    for b in range(n_boot):
        idx = _stationary_bootstrap_indices(n, block, rng)
        d = ann_sharpe(r_a[idx], ann) - ann_sharpe(r_b[idx], ann)
        if d <= 0:
            count += 1
    return float(obs), float(count / n_boot)


def probabilistic_sharpe(r, sr_benchmark=0.0):
    """
    PSR (Bailey & López de Prado 2012): probability that the TRUE per-period
    Sharpe exceeds `sr_benchmark` (per-period units), accounting for sample
    length, skew and kurtosis.
    """
    r = np.asarray(r, dtype=np.float64)
    T = len(r)
    if T < 10 or r.std() < 1e-12:
        return float("nan")
    sr = r.mean() / r.std()
    g3 = sps.skew(r)
    g4 = sps.kurtosis(r, fisher=False)
    denom = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2))
    z = (sr - sr_benchmark) * np.sqrt(T - 1) / denom
    return float(sps.norm.cdf(z))


def deflated_sharpe(r, trial_sharpes, ann=252):
    """
    DSR: PSR against the expected MAX Sharpe of `len(trial_sharpes)` random
    trials with the observed cross-trial variance — i.e. "is this Sharpe more
    than what selection over N variants would produce by luck?"
    `trial_sharpes` = per-period Sharpes of every configuration that was tried.
    """
    trial_sharpes = np.asarray(trial_sharpes, dtype=np.float64)
    N = len(trial_sharpes)
    if N < 2:
        return probabilistic_sharpe(r, 0.0)
    v = trial_sharpes.var(ddof=1)
    sr_star = np.sqrt(v) * ((1 - EULER_GAMMA) * sps.norm.ppf(1 - 1.0 / N)
                            + EULER_GAMMA * sps.norm.ppf(1 - 1.0 / (N * np.e)))
    return probabilistic_sharpe(r, sr_star)


def max_drawdown(eq):
    eq = np.asarray(eq, dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    return float(((peak - eq) / np.where(peak > 0, peak, 1e-10)).max())


def summary_row(r, ann=252, name=""):
    r = np.asarray(r, dtype=np.float64)
    eq = np.cumprod(1.0 + r)
    lo, hi = bootstrap_sharpe_ci(r, ann)
    return {
        "strategy":      name,
        "bars":          len(r),
        "ann_ret_pct":   round(r.mean() * ann * 100, 2),
        "sharpe":        round(ann_sharpe(r, ann), 3),
        "sharpe_ci_lo":  round(lo, 3),
        "sharpe_ci_hi":  round(hi, 3),
        "psr_vs_0":      round(probabilistic_sharpe(r), 3),
        "sortino":       round(np.sqrt(ann) * r.mean()
                               / max(r[r < 0].std(), 1e-12), 3)
                          if (r < 0).sum() > 1 else float("nan"),
        "max_dd_pct":    round(max_drawdown(eq) * 100, 2),
        "total_ret_pct": round((eq[-1] - 1) * 100, 2),
    }
