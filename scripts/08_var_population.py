#!/usr/bin/env python3
"""
Step 8 - panel vector autoregression of the factor scores within users.

Supports the Methods section "Factor dynamics within individuals" and the Results section
"Longitudinal analysis of the factors": Granger-predictive pairs and impulse responses (Fig. 3b,c),
the P value matrix, step responses, lag-order selection and the distribution of P values
(Extended Data Fig. 7a,c-e), and the calibration of the tests under a simulated null (Methods).
The lag-1 coefficients of Fig. 3a come from step 19, which refits the same model.  The
estimators are documented in src/slm/var.py.

Analyses (keys of the output dictionary)
  corrected         main model, bias-corrected: panel VAR on pairs of windows of the same user
                    exactly 14 days apart, scores centered within user, half-panel jackknife bias
                    correction, lag order chosen by BIC among 1 to 3 on a common sample; Wald
                    tests from a bootstrap over users (--n-boot resamples), Benjamini-Hochberg FDR
                    over the 132 ordered pairs, bootstrap bands of impulse and step responses
  corrected_p2      the same model with lag order 2 (sensitivity to the lag order)
  corrected_within  within estimator at the BIC order without bias correction, with
                    cluster-robust Wald tests (users as clusters) and bootstrap bands
  null_calibration  false-positive rates of both tests in --n-null panels simulated with each
                    user's observation pattern, the fitted own-lag dynamics and the innovation
                    covariance of the data, but no lagged effects between factors

Inputs
  data/processed/<run>/factors<tag>.parquet   factor scores nmf_0..nmf_{K-1} (0-100) of the
                                              confident windows (step 6)
  results/<run>/06_nmf<tag>.npz              factor titles (step 6)
Outputs
  results/<run>/08_var_population<tag>.npy    dict {corrected, corrected_p2, corrected_within,
                                              null_calibration} with all arrays (coefficients
                                              A, P values p, FDR q values q, responses irf / cum
                                              and their bands); matrices are indexed
                                              [response, predictor]
  results/<run>/08_var_population<tag>.json   summary of each model: order, nobs (lagged pairs),
                                              n_speakers, n_sig (pairs with q < 0.05), spectral
                                              radius rho (rho_within: without bias correction),
                                              bias_correction, significant pairs as
                                              "predictor -> response"; and null_calibration
  results/<run>/08_var_order_ic<tag>.csv      AIC and BIC of lag orders 1-3 on the common sample

Usage (run from scripts/)
  python3 08_var_population.py --run llama31_8b --n-boot 500
  add --tag _k18 for the K = 18 sensitivity solution
"""
import argparse

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from slm import params as P
from slm import var as V
from slm.io import Timer, save_json

ap = argparse.ArgumentParser(description="Panel VAR of the factor scores within users.")
ap.add_argument("--run", required=True,
                help="annotation run, i.e. the folder under data/processed/ and results/ (main: llama31_8b)")
ap.add_argument("--n-boot", type=int, default=500,
                help="bootstrap resamples of users for the tests and response bands (500 in the manuscript)")
ap.add_argument("--n-null", type=int, default=20,
                help="number of panels simulated for the null calibration (20 in the manuscript)")
ap.add_argument("--tag", default="",
                help="suffix of the factor solution (factors<tag>.parquet, 06_nmf<tag>.npz), e.g. _k18")
args = ap.parse_args()

pdir = P.processed_dir(args.run)
rdir = P.results_dir(args.run)
df = pd.read_parquet(pdir / f"factors{args.tag}.parquet")
titles = list(np.load(rdir / f"06_nmf{args.tag}.npz", allow_pickle=True)["titles"])
keys = [f"nmf_{j}" for j in range(len(titles))]     # score columns, in the order of titles
out = {}

# ---- step: lag order by BIC on a common sample ---------------------------------------------------
# All orders are compared on the same rows: the windows whose three predecessors (starting 14, 28
# and 42 days before) were all observed, with scores centered within user.  For order p the first K*p
# columns of the lag design hold lags 1..p.  With the ML residual covariance Sigma = E'E / n and
# npar = K (K p + 1) coefficients (with intercepts):
#   AIC = log det Sigma + 2 npar / n,   BIC = log det Sigma + log(n) npar / n.
with Timer("bias-corrected panel VAR: order selection"):
    ic = []
    pmax = 3
    Ymax, Xmax, clmax = V.lag_design(df, keys, pmax, demean=True)
    n = len(Ymax); k = len(keys)
    for p in range(1, pmax + 1):
        Xp = np.c_[np.ones(n), Xmax[:, :k * p]]
        B, *_ = np.linalg.lstsq(Xp, Ymax, rcond=None)
        E = Ymax - Xp @ B
        Sig = E.T @ E / n
        ld = np.linalg.slogdet(Sig)[1]
        npar = k * (k * p + 1)
        ic.append({"p": p, "aic": ld + 2 * npar / n, "bic": ld + np.log(n) * npar / n, "n": n})
    ic = pd.DataFrame(ic)
    p_sel = int(ic.loc[ic["bic"].idxmin(), "p"])
print(ic.round(4).to_string()); print("BIC order:", p_sel)

# ---- step: bias-corrected panel VAR (main model) and its order-2 sensitivity analysis ----------
# Half-panel jackknife estimate, Wald tests and bands from --n-boot bootstrap resamples of users
# (the bootstrap seed is the lag order).
for label, p in [("corrected", p_sel), ("corrected_p2", 2)]:
    with Timer(f"{label}: bias-corrected panel VAR({p})"):
        fit = V.fit_panel_var_hpj(df, keys, p, n_boot=args.n_boot, seed=p)
    out[label] = {**fit, "n_sig": int(np.nansum(fit["q"] < P.FDR_Q))}
    print(f"{label}: nobs={fit['nobs']} users={fit['n_speakers']} sig={out[label]['n_sig']} "
          f"rho={fit['rho']:.3f} (within estimator rho={fit['rho_within']:.3f})")

# ---- step: sensitivity analysis - within estimator without bias correction -----------------------
with Timer(f"corrected_within: panel VAR({p_sel}), no bias correction"):
    fit = V.fit_panel_var(df, keys, p_sel, demean=True)
    pv, qv, F = V.granger_panel(fit)                  # cluster-robust Wald tests
    irf = V.irf_panel(fit, n_boot=args.n_boot)        # responses with bootstrap bands (seed 0)
out["corrected_within"] = {"order": p_sel, "nobs": fit["nobs"], "n_speakers": fit["n_speakers"],
                           "p": pv, "q": qv, "F": F, "n_sig": int(np.nansum(qv < P.FDR_Q)),
                           "rho": V.companion_spectral_radius(fit["A"]), "A": np.array(fit["A"]), **irf}

# ---- step: calibration of both tests under a null without lagged effects between factors --------
# Panels are simulated with the observation pattern of every user, the own-lag coefficients of the
# bias-corrected model and the residual covariance of the within estimator (fit above).
with Timer(f"null calibration ({args.n_null} simulated panels)"):
    out["null_calibration"] = V.null_calibration(df, keys, out["corrected"], fit, n_sim=args.n_null, seed=11)
print("null calibration:", out["null_calibration"])

# ---- step: outputs ---------------------------------------------------------------------------------
np.save(rdir / f"08_var_population{args.tag}.npy", out, allow_pickle=True)
ic.to_csv(rdir / f"08_var_order_ic{args.tag}.csv", index=False)
SUMMARY_FIELDS = ("order", "nobs", "n_speakers", "n_sig", "rho", "rho_within", "bias_correction")
summ = {k: {kk: vv for kk, vv in v.items() if kk in SUMMARY_FIELDS}
        for k, v in out.items() if k != "null_calibration"}
summ["null_calibration"] = out["null_calibration"]
for k in out:
    if k == "null_calibration":
        continue
    sig = np.argwhere(out[k]["q"] < P.FDR_Q)            # (response i, predictor j) with q < 0.05
    summ[k]["significant_pairs"] = [f"{titles[j]} -> {titles[i]}" for i, j in sig]
save_json(summ, rdir / f"08_var_population{args.tag}.json")
print(summ)
