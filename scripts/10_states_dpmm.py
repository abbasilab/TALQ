#!/usr/bin/env python3
"""
Step 10 - discrete mental states: truncated Dirichlet process Gaussian mixture of the factor profiles
(Methods, "Discrete mental states"; Fig. 4a,b; Extended Data Fig. 8).

The factor vectors of all confident windows are clustered with a truncated Dirichlet process Gaussian
mixture (scikit-learn BayesianGaussianMixture, 100 components, full covariances, 10 initializations)
at each of 14 hyperparameter settings, steps 0 (least regularized) to 13 (most regularized), which
interpolate between keypoints of the weight concentration prior (1e-8 to 1e-14), the mean precision
prior (0.001 to 3) and the covariance regularization (10 to 150); see slm.states.  At every step the
most populated components that together cover 90 % of the windows are retained and each window is
assigned to the retained component with the highest posterior probability.  Each step is scored by
AIC, AICc and BIC computed with the parameter count of a Gaussian mixture with full covariances and the
retained components, and with the number of windows as the sample size.  The step is selected by the
minimum of one criterion (--select, default BIC).  In the main analysis BIC is lowest at step 5
(11 states), whereas AIC and AICc are lowest at the most complex setting (step 0, 39 states).

The states of the selected step are coded 0..S-1 in decreasing order of their number of windows and
provisionally named S1..S<S>; 10c_name_states.py then gives them descriptive labels.  Every step is
fitted with its own fixed seed, drawn from P.DPMM_SEED, so the fit of a step does not depend on which
other steps are run.

Inputs   data/processed/<run>/factors<tag>.parquet   factor scores nmf_0..nmf_<K-1> per window (step 6)
         results/<run>/06_nmf<tag>.npz               factor titles
Outputs  data/processed/<run>/states<tag>.parquet    speaker_id, window_start; step_0..step_13: retained
                                                     mixture component of the window at each step
                                                     (component index, not consecutive); state: state
                                                     code of the selected step (0 = most common)
         results/<run>/10_dpmm_sweep<tag>.csv        one row per step: hyperparameters, n_states
                                                     (retained), n_states_raw (components with windows
                                                     before truncation), converged, n_params, loglik,
                                                     aic, aicc, bic, n (Extended Data Fig. 8)
         results/<run>/10_state_centroids<tag>.csv   mean factor scores and number of windows per state
         results/<run>/10_states<tag>.json           selected_step, select_rule, n_states, step_min_bic,
                                                     n_states_min_bic, state_names, n_windows, naming,
                                                     params (hyperparameters of the selected step)
         results/<run>/10_dpmm_model<tag>.joblib     fitted mixture of the selected step with its retained
                                                     components and the map from component to state code,
                                                     to assign further factor vectors to the same states

usage: python3 10_states_dpmm.py --run llama31_8b --n-init 10                 (step 5, 11 states)
       python3 10_states_dpmm.py --run llama31_8b --n-init 10 --tag _k18      (K = 18 sensitivity analysis)
"""
import argparse

import _setup  # noqa: F401
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from slm import params as P
from slm.io import Timer, save_json
from slm.states import dpmm_param_sweep, fit_dpmm_step, gmm_information_criteria

ap = argparse.ArgumentParser(description="Dirichlet process Gaussian mixture states of the factor "
                                         "profiles.")
ap.add_argument("--run", required=True,
                help="annotation run: folder under data/processed/ and results/ (main analysis: llama31_8b)")
ap.add_argument("--n-init", type=int, default=P.DPMM_FIT["n_init"],
                help="variational initializations per step; the one with the best lower bound is kept "
                     "(10 in the main analysis)")
ap.add_argument("--steps", default=None,
                help="comma-separated sweep steps to fit (default: all 14); the step is selected among "
                     "the fitted steps")
ap.add_argument("--n-jobs", type=int, default=2, help="number of steps fitted in parallel")
ap.add_argument("--select", default="bic", choices=["bic", "aic", "aicc"],
                help="information criterion whose minimum over the fitted steps selects the step "
                     "(default: bic)")
ap.add_argument("--step", type=int, default=None,
                help="use this step instead of the criterion minimum (it must be among the fitted steps)")
ap.add_argument("--tag", default="", help="suffix of the input factor files and of all outputs (e.g. _k18)")
args = ap.parse_args()

pdir = P.processed_dir(args.run)
rdir = P.results_dir(args.run)

# ---- factor scores and the hyperparameter settings ----
df = pd.read_parquet(pdir / f"factors{args.tag}.parquet")
titles = list(np.load(rdir / f"06_nmf{args.tag}.npz", allow_pickle=True)["titles"])
keys = [f"nmf_{j}" for j in range(len(titles))]
X = df[keys].to_numpy(float)                       # (n_windows, n_factors), 0-100 scale
steps = dpmm_param_sweep()                         # 14 steps x (weight_concentration_prior,
                                                   # mean_precision_prior, reg_covar)
sel = list(range(len(steps))) if args.steps is None else [int(s) for s in args.steps.split(",")]
if args.step is not None and args.step not in sel:
    ap.error(f"--step {args.step} is not among the fitted steps {sel}")
# one seed per step, drawn for all 14 steps whatever subset is fitted
rng = np.random.RandomState(P.DPMM_SEED)
seeds = rng.randint(0, 2**31 - 1, size=len(steps))


def run_step(i):
    """Fit sweep step i (see slm.states.fit_dpmm_step); returns (i, model, raw, states, retained)."""
    prm = steps.iloc[i].to_dict()
    fit = dict(P.DPMM_FIT)
    fit["n_init"] = args.n_init
    m, raw, st, keep = fit_dpmm_step(X, prm, int(seeds[i]), fit=fit)
    return i, m, raw, st, keep


# ---- the sweep ----
with Timer(f"DPMM sweep over steps {sel}"):
    res = Parallel(n_jobs=args.n_jobs)(delayed(run_step)(i) for i in sel)

rows, st_cols, models, keeps = [], {}, {}, {}
for i, m, raw, st, keep in res:
    keeps[i] = keep
    ic = gmm_information_criteria(m, X, st)        # n_params, loglik, aic, aicc, bic, n
    rows.append({"step": i, **steps.iloc[i].to_dict(), "n_states": int(len(np.unique(st))),
                 "n_states_raw": int(len(np.unique(raw))), "converged": bool(m.converged_), **ic})
    st_cols[f"step_{i}"] = st
    models[i] = m
sweep = pd.DataFrame(rows).sort_values("step")
sweep.to_csv(rdir / f"10_dpmm_sweep{args.tag}.csv", index=False)
print(sweep[["step", "n_states", "aic", "aicc", "bic", "converged"]].to_string())

# ---- selection of the step ----
if args.step is not None:
    use, rule = args.step, "step"
elif len(sweep) == 1:
    use, rule = int(sweep["step"].iloc[0]), args.select
else:
    use, rule = int(sweep.loc[sweep[args.select].idxmin(), "step"]), args.select   # NaN AICc values skipped
print(f"selected step {use} ({rule})")

# ---- states of the selected step: codes in decreasing order of occupancy ----
out = df[["speaker_id", "window_start"]].copy()
for k, v in st_cols.items():
    out[k] = v
st = st_cols[f"step_{use}"]
labs = np.unique(st)                                          # retained component indices, ascending
cent = np.array([X[st == l].mean(axis=0) for l in labs])      # centroid of each component
order = list(np.argsort(-np.bincount(np.searchsorted(labs, st))))   # positions in labs, largest first
names = [f"S{i+1}" for i in range(len(labs))]                 # provisional; 10c_name_states.py names them
remap = {labs[o]: new for new, o in enumerate(order)}         # component index -> state code
out["state"] = [remap[s] for s in st]
cdf = pd.DataFrame(cent[order], columns=titles)
cdf.insert(0, "state", names)
cdf["n_windows"] = [int((st == labs[o]).sum()) for o in order]
cdf.to_csv(rdir / f"10_state_centroids{args.tag}.csv", index=False)
# the selected model, so that further factor vectors can be assigned to these states
joblib.dump({"model": models[use], "keep": keeps[use], "remap": remap, "names": names, "step": use,
             "factor_titles": titles}, rdir / f"10_dpmm_model{args.tag}.joblib")
i_bic = sweep["bic"].idxmin()
summ = {"selected_step": use, "select_rule": rule, "n_states": int(len(labs)),
        "step_min_bic": int(sweep.loc[i_bic, "step"]),
        "n_states_min_bic": int(sweep.loc[i_bic, "n_states"]),
        "state_names": names, "n_windows": cdf["n_windows"].tolist(), "naming": "ordered by occupancy",
        "params": steps.iloc[use].to_dict()}
print(cdf.round(1).to_string())
out.to_parquet(pdir / f"states{args.tag}.parquet", index=False)
save_json(summ, rdir / f"10_states{args.tag}.json")
