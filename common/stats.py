"""
Every performance statistic used by the project lives here.

  * the per-strategy metric block written to the phase*_metrics.csv files
    (Sharpe, Sortino, max drawdown, Calmar, CVaR, Omega, win rate)
  * stationary-bootstrap confidence intervals for Sharpe (Politis–Romano)
  * bootstrap test for a Sharpe DIFFERENCE between two strategies (paired)
  * Probabilistic / Deflated Sharpe Ratio (Bailey & López de Prado)
"""
import numpy as np
from scipy import stats as sps

EULER_GAMMA = 0.5772156649015329


# ── Metric block (phase*_metrics.csv, evaluate_*.py, dashboard.py) ──────────

def sharpe(r, ann=252):
    s = r.std()
    return float(np.sqrt(ann) * r.mean() / s) if s > 1e-10 else 0.0


def sortino(r, ann=252):
    down = r[r < 0]
    ds = down.std() if len(down) > 1 else 1e-10
    return float(np.sqrt(ann) * r.mean() / ds)


def calmar(r, eq, ann=252):
    mdd = max_drawdown(eq)
    return float(r.mean() * ann / mdd) if mdd > 1e-6 else 0.0


def cvar_metric(r, alpha=0.95):
    var = np.percentile(r, (1 - alpha) * 100)
    tail = r[r <= var]
    return float(tail.mean()) if len(tail) > 0 else 0.0


def omega_metric(r, threshold=0.0):
    gains  = (r[r > threshold] - threshold).sum()
    losses = (threshold - r[r < threshold]).sum()
    return float(gains / losses) if losses > 1e-10 else 2.0


def compute_metrics(r, eq, ann=252):
    return {
        "Total ret (%)":  round((eq[-1] / eq[0] - 1) * 100, 2),
        "Ann. ret (%)":   round(r.mean() * ann * 100, 2),
        "Sharpe":         round(sharpe(r, ann), 3),
        "Sortino":        round(sortino(r, ann), 3),
        "MaxDD (%)":      round(max_drawdown(eq) * 100, 2),
        "Calmar":         round(calmar(r, eq, ann), 3),
        "CVaR 95% (%)":   round(cvar_metric(r) * 100, 3),
        "Omega":          round(omega_metric(r), 3),
        "Win rate (%)":   round((r > 0).mean() * 100, 1),
    }


# ── Bootstrap / deflated Sharpe ─────────────────────────────────────────────

def ann_sharpe(r, ann=252):
    r = np.asarray(r, dtype=np.float64)
    s = r.std()
    return float(np.sqrt(ann) * r.mean() / s) if s > 1e-12 else 0.0


def _ann_sharpe_rows(x, ann=252):
    """Annualized Sharpe of each row of a (n_boot, n) return matrix."""
    s = x.std(axis=1)
    safe = np.where(s > 1e-12, s, 1.0)
    return np.where(s > 1e-12, np.sqrt(ann) * x.mean(axis=1) / safe, 0.0)


def _stationary_bootstrap_indices(n, block, rng, n_boot=1):
    """
    (n_boot, n) matrix of stationary-bootstrap resample indices (mean block
    length = `block`).  Each path starts at a random bar, then at every step
    either jumps to a new random bar (probability 1/block) or moves on to the
    next bar, wrapping around.
    """
    restart = rng.random((n_boot, n)) < 1.0 / block
    fresh = rng.integers(0, n, size=(n_boot, n))
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = fresh[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(restart[:, t], fresh[:, t],
                             (idx[:, t - 1] + 1) % n)
    return idx


def bootstrap_sharpe_ci(r, ann=252, n_boot=2000, block=20, alpha=0.05,
                        seed=0):
    """Percentile CI for the annualized Sharpe under a stationary bootstrap."""
    r = np.asarray(r, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = _stationary_bootstrap_indices(len(r), block, rng, n_boot)
    sims = _ann_sharpe_rows(r[idx], ann)
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
    idx = _stationary_bootstrap_indices(n, block, rng, n_boot)
    diffs = _ann_sharpe_rows(r_a[idx], ann) - _ann_sharpe_rows(r_b[idx], ann)
    return float(obs), float((diffs <= 0).mean())


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
