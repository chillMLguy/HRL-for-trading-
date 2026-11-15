"""
One test year of the multi-asset hierarchy (brief section 8 stage 5, and section 9).

    python evaluate_multi.py --outdir runs/multi_wf/fold2015 --test_year 2015

Needs, in the fold folder: the final specialists (models/specialists), the cross-fit
stream (crossfit/tilts.csv) and both final allocators (models/allocator_full,
models/allocator_core_alloc).  Fits the final HMM on the whole training window
(models/hmm_final.pkl), then runs the test year with warm history: decisions from the
last close of the year before to the second-to-last close of the test year, so the
returns are those realised on the test year's trading days.

Writes to the fold folder:
  multi_returns.csv        date + the nine series, daily net excess returns after costs
  multi_positions.csv      date, portfolio, one column per ticker (core, core_spec,
                           core_alloc, full)
  multi_tilts.csv          the final specialists' tilts on the test-year decision days
  multi_budgets.csv        date, allocator, decision, G, beta_<ticker>
  multi_metrics.csv        summary per portfolio for this year
  multi_comparisons.csv    the five pre-registered pairs, on this year alone
  multi_window_sharpe.csv  active Sharpe of each learned layer on its fit window (in
                           sample), on the cross-fit blocks and on the test year
  multi_diagnostics.json   core, specialist and allocator diagnostics of section 9
  multi_eval_config.json   settings, dates, models and data hashes

One year is one draw: these numbers describe the fold, they are not a result.
"""
import argparse
import glob
import hashlib
import json
import os
import warnings

import numpy as np
import pandas as pd

import config
from common.stats import (ann_sharpe, bootstrap_sharpe_ci, bootstrap_sharpe_diff_pvalue)
from multi.alloc_env import AllocInputs, load_allocator, run_allocator
from multi.data import load_universe, training_window
from multi.features import compute_inputs, trend_rule_tilt
from multi.portfolio import (RULE_BASED, active_sharpe, build_positions, net_returns,
                             portfolio_metrics, rule_based_positions)
from multi.spec_env import load_ensemble, specialist_tilts

warnings.filterwarnings("ignore")

SERIES = ["core", "core_trend", "core_spec", "core_alloc", "full", "equal_weight",
          "sixty_forty", "spy_vt", "spy"]
COMPARISONS = [("core_spec", "core", "Do the specialists add anything?"),
               ("core_spec", "core_trend", "Do learned specialists beat a simple trend rule?"),
               ("full", "core_spec", "Does the allocator add anything on top of the specialists?"),
               ("core_alloc", "core", "Does the allocator add anything by itself?"),
               ("full", "core", "Does the whole hierarchy beat the fixed core?")]


def _dirs(path):
    d = sorted(glob.glob(os.path.join(path, "seed*")))
    if not d:
        raise SystemExit(f"No models in {path}")
    return d


def _corr_cols(a, b):
    out = []
    for i in range(a.shape[1]):
        if a[:, i].std() > 0 and b[:, i].std() > 0:
            out.append(np.corrcoef(a[:, i], b[:, i])[0, 1])
        else:
            out.append(np.nan)
    return np.array(out)


def _seed_corr(per_seed):
    cs = [np.nanmean(_corr_cols(per_seed[a], per_seed[b]))
          for a in range(len(per_seed)) for b in range(a + 1, len(per_seed))]
    return round(float(np.nanmean(cs)), 3) if cs else None


def alloc_diag(out):
    dec = out["decision"]
    beta, u, G = out["beta"][dec], out["u"][dec], out["G"][dec]
    prev = np.vstack([np.asarray(config.MULTI_BASE_BUDGETS), beta[:-1]])
    return {"decisions": int(dec.sum()),
            "mean_budget_change": round(float(np.abs(beta - prev).sum(axis=1).mean()), 4),
            "share_any_u_beyond_095_pct": round(float((np.abs(u) > 0.95).any(axis=1).mean()
                                                      * 100), 1),
            "mean_G": round(float(G.mean()), 4),
            "mean_beta": [round(float(v), 4) for v in beta.mean(axis=0)]}


def evaluate(fold, test_year, settings_root=None):
    from search_multi import load_settings
    from train_allocator import alloc_inputs, policy_check, stream_arrays
    from train_crossfit_multi import fit_core_hmm, hmm_posteriors, stream_scores

    Y = test_year
    settings = load_settings(settings_root or os.path.dirname(os.path.normpath(fold)))
    u = load_universe(config.MULTI_DOWNLOAD_START, f"{Y + 1}-01-01")
    inp = compute_inputs(u)
    T, N = inp.vol.shape
    d = inp.dates
    lo, hi = training_window(d, Y, config.MULTI_WARM_BARS)
    t0, t1 = hi - 1, T - 1                       # test decisions [t0, t1)
    if d[t1].year != Y or d[t0 + 1].year != Y:
        raise SystemExit(f"Data do not cover {Y}: last date {d[-1].date()}.")
    ret_dates = d[t0 + 1:t1 + 1]

    # specialists: final ensemble, with history for the allocator's 60-day input
    spec_dirs = _dirs(os.path.join(fold, "models", "specialists"))
    spec = load_ensemble(spec_dirs)
    tm0 = t0 - config.ALLOC_PERF_WINDOW - 2
    tilts_all = specialist_tilts(spec, inp, tm0, t1)
    m_full = np.ones((T, N))
    m_full[tm0:t1] = 1.0 + tilts_all
    tilts = tilts_all[t0 - tm0:]

    # final HMM on the whole training window
    det = fit_core_hmm(inp, lo, hi)
    det.save(os.path.join(fold, "models", "hmm_final.pkl"))
    hmm = np.full((T, config.N_STATES), np.nan)
    hmm[t0:t1] = hmm_posteriors(det, inp, lo, t0, t1)

    # allocators
    runs, ai = {}, {}
    for name, m in (("full", m_full), ("core_alloc", np.ones((T, N)))):
        ai[name] = AllocInputs(inp, m, hmm)
        models = load_allocator(_dirs(os.path.join(fold, "models", f"allocator_{name}")))
        runs[name] = run_allocator(models, ai[name], t0, t1)
        runs[name]["models"] = models

    # positions of the nine portfolios (full-length arrays, test rows filled)
    trend = trend_rule_tilt(inp)
    rule = rule_based_positions(inp, trend)
    pos, info = {}, {}
    for name in RULE_BASED:
        pos[name], info[name] = rule[name]
    pos["core_spec"], info["core_spec"] = build_positions(inp.vol, inp.cov, m=m_full,
                                                          return_info=True)
    for name in ("full", "core_alloc"):
        r = runs[name]
        beta = np.tile(np.asarray(config.MULTI_BASE_BUDGETS), (T, 1))
        G = np.ones(T)
        beta[t0:t1], G[t0:t1] = r["beta"], r["G"]
        x, inf = build_positions(inp.vol, inp.cov, m=ai[name].m, beta=beta, G=G,
                                 return_info=True)
        if np.abs(x[t0:t1] - r["x"]).max() > 1e-12:
            raise SystemExit(f"{name}: positions differ between the environment and "
                             "build_positions")
        pos[name], info[name] = x, inf

    results = {n: net_returns(pos[n], inp.rex, t0, t1) for n in SERIES}
    returns = pd.DataFrame({n: results[n]["r"] for n in SERIES},
                           index=pd.Index(ret_dates, name="date"))

    metrics = pd.DataFrame([{"portfolio": n, **portfolio_metrics(results[n], info[n], t0, t1)}
                            for n in SERIES])

    comps = []
    for a, b, q in COMPARISONS:
        ra, rb = returns[a].values, returns[b].values
        diff, p = bootstrap_sharpe_diff_pvalue(ra, rb)
        lo_ci, hi_ci = bootstrap_sharpe_ci(ra - rb)
        comps.append({"first": a, "second": b, "question": q,
                      "sharpe_first": round(ann_sharpe(ra), 3),
                      "sharpe_second": round(ann_sharpe(rb), 3),
                      "sharpe_diff": round(diff, 3), "p_value": round(p, 3),
                      "active_sharpe": round(active_sharpe(ra, rb), 3),
                      "active_ci_lo": round(lo_ci, 3), "active_ci_hi": round(hi_ci, 3),
                      "scope": f"test year {Y} only"})

    # overfit check: fit window (in sample), cross-fit blocks, test year
    stream = pd.read_csv(os.path.join(fold, "crossfit", "tilts.csv"), parse_dates=["date"])
    s0, s1, m_stream, hmm_stream = stream_arrays(inp, stream)
    win = []
    core_x = pos["core"]

    def act(xa, xb, a, b):
        return round(active_sharpe(net_returns(xa, inp.rex, a, b)["r"],
                                   net_returns(xb, inp.rex, a, b)["r"]), 3)

    fit_tilts = specialist_tilts(spec, inp, lo, hi - 1)
    m_fit = np.ones((T, N))
    m_fit[lo:hi - 1] = 1.0 + fit_tilts
    win.append({"layer": "specialists", "baseline": "core", "window": "fit (in sample)",
                "first": str(d[lo].date()), "last": str(d[hi - 2].date()),
                "active_sharpe": act(build_positions(inp.vol, inp.cov, m=m_fit), core_x,
                                     lo, hi - 1)})
    sc, _ = stream_scores(stream, inp)
    win.append({"layer": "specialists", "baseline": "core", "window": "cross-fit blocks",
                "first": sc["first"], "last": sc["last"],
                "active_sharpe": sc["spec_active_sharpe"]})
    win.append({"layer": "specialists", "baseline": "core", "window": "test",
                "first": str(ret_dates[0].date()), "last": str(ret_dates[-1].date()),
                "active_sharpe": round(active_sharpe(returns["core_spec"], returns["core"]), 3)})
    choice = settings.get("alloc_choice", "default")
    cf_root = fold if choice == "default" else os.path.join(fold, "search", f"alloc_{choice}")
    cf_path = os.path.join(cf_root, "alloc_crossfit", "summary.json")
    cf = json.load(open(cf_path)) if os.path.isfile(cf_path) else {}
    penalty = {}
    for name, base in (("full", "core_spec"), ("core_alloc", "core")):
        ai_s = alloc_inputs(inp, stream, name)
        models = runs[name]["models"]
        ins = run_allocator(models, ai_s, s0, s1)
        x_in = np.full_like(ai_s.x_fix, np.nan)
        x_in[s0:s1] = ins["x"]
        win.append({"layer": f"allocator {name}", "baseline": base,
                    "window": "fit (in sample)", "first": str(d[s0].date()),
                    "last": str(d[s1 - 1].date()),
                    "active_sharpe": act(x_in, ai_s.x_fix, s0, s1)})
        win.append({"layer": f"allocator {name}", "baseline": base,
                    "window": "cross-fit blocks (held out)",
                    "first": str(d[s0].date()), "last": str(d[s1 - 1].date()),
                    "active_sharpe": cf.get(name, {}).get("active_sharpe")})
        win.append({"layer": f"allocator {name}", "baseline": base, "window": "test",
                    "first": str(ret_dates[0].date()), "last": str(ret_dates[-1].date()),
                    "active_sharpe": round(active_sharpe(returns[name], returns[base]), 3)})
        with open(os.path.join(_dirs(os.path.join(fold, "models",
                                                  f"allocator_{name}"))[0],
                               "config.json")) as f:
            c0 = json.load(f)
        penalty[name] = policy_check(models, ai_s, s0, s1, c0["mu_q"],
                                     turnover_pen=c0["turnover_pen"])
    window = pd.DataFrame(win)

    # diagnostics (section 9)
    tsl = slice(t0, t1)
    core_res = results["core"]
    diag = {"core": {
        "realised_vol_pct": round(float(core_res["r"].std() * np.sqrt(252) * 100), 2),
        "target_vol_pct": config.PORT_VOL * 100,
        "vol_cap_days_pct": round(float(info["core"]["vol_bind"][tsl].mean() * 100), 2),
        "lev_cap_days_pct": round(float(info["core"]["lev_bind"][tsl].mean() * 100), 2),
        "gross_mean": round(float(core_res["gross"].mean()), 3),
        "gross_max": round(float(core_res["gross"].max()), 3)}}
    per_seed = [specialist_tilts([mdl], inp, t0, t1) for mdl in spec]
    fold_corr = {}
    for fd in sorted(glob.glob(os.path.join(fold, "crossfit", "fold*", "specialists"))):
        j = os.path.basename(os.path.dirname(fd))
        ft = specialist_tilts(load_ensemble(_dirs(fd)), inp, t0, t1)
        fold_corr[j] = round(float(np.nanmean(_corr_cols(tilts, ft))), 3)
    tr = trend[t0:t1]
    diag["specialists"] = {
        "per_asset": pd.DataFrame({
            "mean_tilt": tilts.mean(axis=0),
            "share_abs_gt_095_pct": (np.abs(tilts) > 0.95).mean(axis=0) * 100,
            "lag1_autocorr": [np.corrcoef(tilts[1:, i], tilts[:-1, i])[0, 1]
                              for i in range(N)],
            "corr_with_trend_tilt": _corr_cols(tilts, tr)},
            index=inp.tickers).round(3).reset_index(names="ticker").to_dict("records"),
        "mean_corr_between_seeds": _seed_corr(per_seed),
        "corr_final_vs_fold_specialists_on_test_year": fold_corr,
        "turnover_per_year": round(float(results["core_spec"]["turnover"].mean() * 252), 1),
    }
    diag["allocators"] = {}
    for name in ("full", "core_alloc"):
        r = runs[name]
        seeds_out = [run_allocator([mdl], ai[name], t0, t1) for mdl in r["models"]]
        acts = [np.column_stack([o["u"], o["g"]])[o["decision"]] for o in seeds_out]
        diag["allocators"][name] = {**alloc_diag(r), "mean_corr_between_seeds": _seed_corr(acts),
                                    "penalty_check_on_stream_in_sample": penalty[name]}

    # files
    returns.to_csv(os.path.join(fold, "multi_returns.csv"))
    rows = []
    for name in ("core", "core_spec", "core_alloc", "full"):
        df = pd.DataFrame(pos[name][tsl], columns=inp.tickers)
        df.insert(0, "portfolio", name)
        df.insert(0, "date", d[tsl])
        rows.append(df)
    pd.concat(rows).round(6).to_csv(os.path.join(fold, "multi_positions.csv"), index=False)
    pd.DataFrame(tilts, index=pd.Index(d[tsl], name="date"),
                 columns=inp.tickers).round(6).to_csv(os.path.join(fold, "multi_tilts.csv"))
    bud = []
    for name in ("full", "core_alloc"):
        r = runs[name]
        df = pd.DataFrame(r["beta"], columns=[f"beta_{t}" for t in inp.tickers])
        df.insert(0, "G", r["G"])
        df.insert(0, "decision", r["decision"])
        df.insert(0, "allocator", name)
        df.insert(0, "date", d[tsl])
        bud.append(df)
    pd.concat(bud).round(6).to_csv(os.path.join(fold, "multi_budgets.csv"), index=False)
    metrics.to_csv(os.path.join(fold, "multi_metrics.csv"), index=False)
    pd.DataFrame(comps).to_csv(os.path.join(fold, "multi_comparisons.csv"), index=False)
    window.to_csv(os.path.join(fold, "multi_window_sharpe.csv"), index=False)
    with open(os.path.join(fold, "multi_diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2, default=float)
    sha = {}
    for path in (config.MULTI_DATA_CSV, config.RISKFREE_CSV):
        with open(path, "rb") as f:
            sha[path] = hashlib.sha256(f.read()).hexdigest()
    with open(os.path.join(fold, "multi_eval_config.json"), "w") as f:
        json.dump({"test_year": Y, "settings": settings,
                   "training_window": [str(d[lo].date()), str(d[hi - 1].date())],
                   "first_decision": str(d[t0].date()),
                   "returns": [str(ret_dates[0].date()), str(ret_dates[-1].date()),
                               len(ret_dates)],
                   "specialists": spec_dirs,
                   "allocators": {n: _dirs(os.path.join(fold, "models", f"allocator_{n}"))
                                  for n in ("full", "core_alloc")},
                   "hmm": os.path.join(fold, "models", "hmm_final.pkl"),
                   "data_sha256": sha}, f, indent=2)
    return returns, metrics, pd.DataFrame(comps), window, diag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True, help="the fold folder")
    p.add_argument("--test_year", type=int, required=True)
    p.add_argument("--settings_root", default=None,
                   help="folder of settings.json (default: the fold's parent)")
    args = p.parse_args()
    returns, metrics, comps, window, diag = evaluate(args.outdir, args.test_year,
                                                     args.settings_root)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(f"\nTest year {args.test_year} (one year: a description, not a result)\n")
    print(metrics.set_index("portfolio").to_string())
    print("\nPre-registered comparisons, this year only\n")
    print(comps.drop(columns=["question", "scope"]).to_string(index=False))
    print("\nOverfit check (active Sharpe)\n")
    print(window.to_string(index=False))
    print("\nDiagnostics\n" + json.dumps({k: v for k, v in diag.items()}, indent=1,
                                         default=float)[:3000])


if __name__ == "__main__":
    main()
