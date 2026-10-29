#!/usr/bin/env python3
"""
Step 7 - psychometric properties of the NMF factors.

Supports the Methods section "Psychometric properties": weighted alpha and test-retest kappa and
ICC in Fig. 2d and Table 1, and all statistics with their confidence strata in Extended Data
Fig. 6.  Estimators are documented in src/slm/psychometrics.py.

Each factor j is treated as the composite y = sum_i w_i x_i of the 52 symptom ratings x_i,
weighted by its loadings w_i = H[j, i].  Columns of the output (one row per factor):

  alpha             weighted Cronbach's alpha,
                    alpha = k_eff/(k_eff - 1) * (1 - sum_i var(w_i x_i) / var(y)),
                    k_eff = (sum_i w_i)^2 / sum_i w_i^2
  k_eff             effective number of items
  n_items_top50     number of items that carry 50% of the factor's weight
  alpha_top50       ordinary (unweighted) alpha of those items
  discrim_rest_top  item discrimination: correlation of each item with the rest of the composite,
                    averaged over the items that carry 50% of the weight
  discrim_rest_bottom   the same for as many items with the lowest weights (comparison)
  kappa             test-retest weighted kappa (linear weights) of the composite binned in
                    tertiles, over all pairs of adjacent windows (same user, 14 days apart)
  kappa_npairs      number of these pairs
  icc               test-retest ICC(A,1) of the continuous composite on the same pairs

Confidence strata (suffixes _lo and _hi; model confidence of the ratings, 0-100):
  alpha_lo, alpha_hi                 windows whose factor-weighted confidence sum_i w_i c_i is in
                                     the bottom / top 10% (params.ALPHA_CONF_QUANTILES)
  discrim_rest_{top,bottom}_{lo,hi}  for each item, the windows in the bottom / top quartile of
                                     that item's own confidence (params.DISCRIM_CONF_QUANTILES)
  kappa_lo, kappa_hi, icc_lo, icc_hi, kappa_lo_npairs, kappa_hi_npairs
                                     pairs of adjacent windows that are both in the bottom / top
                                     quartile of factor-weighted confidence
                                     (params.KAPPA_CONF_QUANTILES)

95% intervals (column suffix _ci; percentile bootstrap):
  alpha*, kappa*    bootstrap over users, --n-boot resamples (200 in the manuscript)
  discrim_rest_*    bootstrap over windows (rows), 100 resamples

Inputs
  data/processed/<run>/factors<tag>.parquet   ratings and confidences of the confident windows (step 6)
  results/<run>/06_nmf<tag>.npz              loadings H and factor titles (step 6)
Outputs
  results/<run>/07_psychometrics<tag>.pkl    one row per factor; intervals as (lower, upper) tuples
  results/<run>/07_psychometrics<tag>.csv    the same; each interval column <col>_ci is split into
                                             <col>_ci_lo95 and <col>_ci_hi95 (placed at the end)

Usage (run from scripts/)
  python3 07_psychometrics.py --run llama31_8b --n-boot 200
  add --tag _k18 for the K = 18 sensitivity solution
"""
import argparse

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from slm import params as P
from slm import psychometrics as PM
from slm.io import Timer
from slm.parse import symptom_matrix

ap = argparse.ArgumentParser(description="Internal consistency, item discrimination and test-retest "
                                         "reliability of the NMF factors.")
ap.add_argument("--run", required=True,
                help="annotation run, i.e. the folder under data/processed/ and results/ (main: llama31_8b)")
ap.add_argument("--n-boot", type=int, default=200,
                help="bootstrap resamples of users for the intervals of alpha and kappa")
ap.add_argument("--tag", default="",
                help="suffix of the factor solution (factors<tag>.parquet, 06_nmf<tag>.npz), e.g. _k18")
args = ap.parse_args()

pdir = P.processed_dir(args.run)
rdir = P.results_dir(args.run)
df = pd.read_parquet(pdir / f"factors{args.tag}.parquet")
nmf = np.load(rdir / f"06_nmf{args.tag}.npz", allow_pickle=True)
H, titles = nmf["H"], list(nmf["titles"])         # loadings (K x 52) and factor names
X = symptom_matrix(df, "rating"); C = symptom_matrix(df, "confidence")   # windows x 52 each
n_win, n_sym = X.shape

# ---- row bootstrap of item discrimination ----------------------------------------------------
# Windows (rows) are resampled with replacement, N_ROW_BOOT times per statistic, from a single
# generator (seed 0) that runs through all statistics and factors in a fixed order.  The random
# stream behind the reported intervals also served, for every factor and before its six
# item-rest statistics, six bootstrap runs of an alternative item selection that is not
# reported.  skip_row_boot draws and discards those resamples, so that the intervals equal those
# of the analysis reported in the manuscript.
rng = np.random.RandomState(0)
N_ROW_BOOT = 100
N_SKIPPED_RUNS = 6


def row_boot(stat, data, n=N_ROW_BOOT):
    """Bootstrap over rows: `stat` (function array -> float) evaluated on n resamples of the rows
    of `data` drawn with replacement from the shared generator `rng`.  A resample on which stat
    fails gives NaN (its rows are drawn all the same).  Returns the n values."""
    out = np.full(n, np.nan)
    for b in range(n):
        idx = rng.randint(0, data.shape[0], data.shape[0])
        try:
            out[b] = stat(data[idx])
        except Exception:  # noqa: BLE001
            pass
    return out


def skip_row_boot(n_runs, n_rows, n=N_ROW_BOOT):
    """Advance the shared generator `rng` past the resamples of n_runs row-bootstrap runs of n
    resamples each, over n_rows rows (draws are made exactly as in row_boot and discarded)."""
    for _ in range(n_runs * n):
        rng.randint(0, n_rows, n_rows)


rows = []
for j, title in enumerate(titles):
    w = H[j]                                       # loadings of factor j: the item weights (52,)
    with Timer(f"factor {j+1} {title}"):
        r = {"factor": j + 1, "title": title}
        wc = C @ w                                 # factor-weighted confidence of every window

        # ---- step: internal consistency (weighted alpha, overall and in confidence strata) ----
        # each call of hierarchical_bootstrap uses its own generator seeded with the factor index
        q_lo, q_hi = P.ALPHA_CONF_QUANTILES        # bottom and top 10% of factor-weighted confidence
        lo, hi = wc <= np.quantile(wc, q_lo), wc >= np.quantile(wc, q_hi)
        for suf, m in [("", np.ones(len(df), bool)), ("_lo", lo), ("_hi", hi)]:
            sub = df[m]
            st = (lambda d: PM.alpha_weighted(symptom_matrix(d), w))
            r["alpha" + suf] = st(sub)
            bs = PM.hierarchical_bootstrap(st, sub, n=args.n_boot, seed=j)
            r["alpha" + suf + "_ci"] = PM.ci(bs)
        r["k_eff"] = PM.k_effective(w)
        top = PM.top_weight_items(w, 0.5)          # the items that carry 50% of the weight
        r["n_items_top50"] = int(len(top))
        r["alpha_top50"] = PM.alpha_unweighted(X[:, top]) if len(top) > 1 else np.nan

        # ---- step: item discrimination (item-rest r), with a row bootstrap ----
        # XC holds ratings (columns 0..51) and confidences (52..103) of the same windows, so that
        # a row resample keeps each window's ratings and confidences together.
        XC = np.c_[X, C]
        skip_row_boot(N_SKIPPED_RUNS, n_win)       # see the note on the row bootstrap above
        dq_lo, dq_hi = P.DISCRIM_CONF_QUANTILES    # bottom and top quartile of each item's confidence
        for lab, low in [("top", False), ("bottom", True)]:
            st = (lambda d, low=low: PM.top_discrim(d, w, P.DISCRIM_TOP_FRAC, low))
            r[f"discrim_rest_{lab}"] = st(X)
            r[f"discrim_rest_{lab}_ci"] = PM.ci(row_boot(st, X))
            for cl, q, dirn in [("lo", dq_lo, "le"), ("hi", dq_hi, "ge")]:
                st2 = (lambda d, low=low, q=q, dirn=dirn: PM.top_discrim(
                    d[:, :n_sym], w, P.DISCRIM_TOP_FRAC, low, conf=d[:, n_sym:], conf_q=q, conf_dir=dirn))
                r[f"discrim_rest_{lab}_{cl}"] = st2(XC)
                r[f"discrim_rest_{lab}_{cl}_ci"] = PM.ci(row_boot(st2, XC))

        # ---- step: test-retest reliability over adjacent windows (kappa and ICC) ----
        raw = X @ w                                # composite score y of every window
        # tertile bins of the composite over all windows; the same bins serve the confidence strata
        # and every bootstrap resample (the top edge is raised so that the maximum falls inside)
        bins = np.quantile(raw, np.arange(0, 1.01, 1 / P.KAPPA_N_BINS)); bins[-1] += 0.1
        kq_lo, kq_hi = P.KAPPA_CONF_QUANTILES      # bottom and top quartile of factor-weighted confidence
        q25, q75 = np.quantile(wc, kq_lo), np.quantile(wc, kq_hi)
        for suf, m in [("", np.ones(len(df), bool)), ("_lo", wc <= q25), ("_hi", wc >= q75)]:
            # pairs are formed within the stratum, so both windows of a pair belong to it
            sub = df[m].copy()
            sub["_raw"] = raw[m]

            def st(d):
                """Kappa of the adjacent pairs of table d (NaN with fewer than three pairs)."""
                pr_ = PM.adjacent_pairs(d, d["_raw"].to_numpy())
                return PM.kappa_binned(pr_, bins) if len(pr_) > 2 else np.nan
            pr = PM.adjacent_pairs(sub, sub["_raw"].to_numpy())
            r["kappa" + suf] = st(sub)
            r["kappa" + suf + "_npairs"] = int(len(pr))
            # relabel: a user drawn twice contributes two separate series, never pairs across copies
            r["kappa" + suf + "_ci"] = PM.ci(PM.hierarchical_bootstrap(st, sub, n=args.n_boot, relabel=True,
                                                                       seed=j))
            r["icc" + suf] = PM.icc_a1(pr) if len(pr) > 2 else np.nan
        rows.append(r)

# ---- step: outputs ---------------------------------------------------------------------------------
out = pd.DataFrame(rows)
out.to_pickle(rdir / f"07_psychometrics{args.tag}.pkl")
flat = out.copy()
for c in flat.columns:                             # split every interval column into two columns
    if flat[c].map(lambda v: isinstance(v, tuple)).any():
        flat[c + "_lo95"] = flat[c].map(lambda v: v[0]); flat[c + "_hi95"] = flat[c].map(lambda v: v[1])
        flat = flat.drop(columns=c)
flat.to_csv(rdir / f"07_psychometrics{args.tag}.csv", index=False)
print(flat[["title", "k_eff", "alpha", "alpha_top50", "discrim_rest_top", "kappa", "kappa_hi", "icc",
            "kappa_npairs"]].round(3).to_string())
