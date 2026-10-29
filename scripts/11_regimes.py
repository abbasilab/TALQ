#!/usr/bin/env python3
"""
Step 11 - dynamical regimes: mixtures of first-order Markov chains over the discrete states
(Methods, "Dynamical regimes"; Fig. 4c,d, Fig. 5d,e; Extended Data Figs. 9 and 10).

* Sequences: the state sequences of users with at least 10 confident windows and no gap longer than
  57 days (249 users and 4,436 windows in the main analysis; slm.markov.good_seqs).
* Population chain: occupancy of the states and transition matrices pooled over users, with
  transitions across gaps counted as single steps (A_pop) and between windows exactly 14 days apart
  only (A_pop_gap, Fig. 4c,d).
* Regime sweep: mixtures of 1-20 Markov chains fitted by hard-assignment EM (20 iterations, 1,000
  random restarts, occupancy as the distribution of initial states, transitions across gaps counted as
  single steps; slm.markov.em_hard), with AIC, AICc and BIC from the classification log likelihood,
  k = K [(S - 1) + S (S - 1)] parameters and n = number of windows (Extended Data Fig. 9a).  The
  number of regimes is the AICc minimum (four in the main analysis).
* Final model: the selected number of regimes with 10,000 restarts (Fig. 5d,e; Extended Data Figs. 9d
  and 10a).  Regimes are numbered in the order of the mixture components of the best restart (Regime 1
  is component 0, and so on); the numbering is fixed by the seeds.
* Fractions of users per regime, with 95 % intervals from a bootstrap over users: each resample is
  refitted (50 restarts) and its regimes are matched to the final ones by the Hungarian algorithm on the
  L1 distance between their transition matrices and initial distributions.
* Symptom burden (Extended Data Fig. 9c): for each factor, the share of each user's windows at or above
  the 90th percentile of the factor scores of the windows of Regime 1, averaged over the users of each
  regime.

Inputs   data/processed/<run>/factors<tag>.parquet, states<tag>.parquet   (steps 6 and 10/10b)
         results/<run>/06_nmf<tag>.npz (factor titles), 10_states<tag>.json (state names)
Outputs  results/<run>/11_regime_sweep<tag>.csv    n_regimes, loglik, n_params, aic, aicc, bic, min_cluster
         results/<run>/11_regime_labels<tag>.csv   speaker_id, regime (1..K), n_windows
         results/<run>/11_regime_burden<tag>.csv   rows Regime 1..K, one column per factor
         results/<run>/11_regimes<tag>.npy         dict: state_names, titles, p0_pop, A_pop, A_pop_gap, K,
                                                   K_aicc, K_bic, p0 (K, S), A (K, S, S) [regime, to, from],
                                                   z (regime code 0..K-1 per user), speakers, frac,
                                                   frac_boot, seqs, steps
         results/<run>/11_regimes<tag>.json        n_sequences, n_windows, n_states, K_aicc, K_bic, K_used,
                                                   fractions, fractions_ci, loglik_final

usage: python3 11_regimes.py --run llama31_8b --n-init-sweep 1000 --n-init-final 10000 --n-boot 200
       (the same with --tag _8states or --tag _k18 for the sensitivity analyses)
"""
import argparse

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from slm import markov as M
from slm import params as P
from slm.io import Timer, load_json, save_json

ap = argparse.ArgumentParser(description="Mixtures of Markov chains (dynamical regimes) over the "
                                         "discrete states.")
ap.add_argument("--run", required=True,
                help="annotation run: folder under data/processed/ and results/ (main analysis: llama31_8b)")
ap.add_argument("--n-init-sweep", type=int, default=P.EM_N_INIT,
                help="random restarts of the hard-assignment EM for each number of regimes in the sweep "
                     "(default and main analysis: 1,000)")
ap.add_argument("--n-init-final", type=int, default=10_000,
                help="random restarts of the final model (default and main analysis: 10,000)")
ap.add_argument("--n-regimes", type=int, default=None,
                help="number of regimes of the final model (default: the AICc minimum of the sweep)")
ap.add_argument("--n-boot", type=int, default=200,
                help="bootstrap resamples of users for the intervals of the regime fractions (default: 200)")
ap.add_argument("--n-jobs", type=int, default=2, help="parallel workers for the EM restarts")
ap.add_argument("--tag", default="", help="suffix of the state solution, e.g. _8states or _k18")
args = ap.parse_args()

pdir = P.processed_dir(args.run)
rdir = P.results_dir(args.run)

# ---- states of the confident windows, and their names ----
fac = pd.read_parquet(pdir / f"factors{args.tag}.parquet")
st = pd.read_parquet(pdir / f"states{args.tag}.parquet")
titles = list(np.load(rdir / f"06_nmf{args.tag}.npz", allow_pickle=True)["titles"])
keys = [f"nmf_{j}" for j in range(len(titles))]
df = fac.merge(st[["speaker_id", "window_start", "state"]], on=["speaker_id", "window_start"])
S = int(df["state"].max()) + 1
_sj = rdir / f"10_states{args.tag}.json"
_names = load_json(_sj).get("state_names") if _sj.exists() else None
state_names = _names if _names and len(_names) == S else [f"S{i+1}" for i in range(S)]

# ---- sequences of the included users (sorted by speaker_id) ----
seqd = M.good_seqs(df, "state", keep_steps=True)
spk = sorted(seqd)
seqs = [seqd[s][0] for s in spk]                   # state codes in time order
steps = [seqd[s][1] for s in spk]                  # window positions (14-day steps from the first window)
n_obs = int(sum(len(s) for s in seqs))             # number of windows, the n of the information criteria
print(f"{len(seqs)} sequences, {n_obs} windows, {S} states")

# ---- population chain: occupancy and transitions pooled over users (no smoothing) ----
occ, trans, first = M.sequence_counts(seqs, S)                  # transitions across gaps as single steps
p0_pop = occ.sum(0) / occ.sum()
A_pop = M.normalize_columns(trans.sum(0))
A_pop_gap = M.normalize_columns(M.sequence_counts(seqs, S, steps, gap_aware=True)[1].sum(0))   # Fig. 4c

# ---- regime sweep: 1-20 regimes, hard-assignment EM, information criteria ----
sweep = []
for K in range(P.N_REGIMES_RANGE[0], P.N_REGIMES_RANGE[1] + 1):
    with Timer(f"hard EM, {K} regimes"):
        f = M.em_hard(seqs, K, n_states=S, n_init=args.n_init_sweep, seed=1000 + K, n_jobs=args.n_jobs)
    k = M.n_params_markov_mixture(K, S)
    aic, aicc, bic = M.info_criteria(f["loglik"], k, n_obs)
    sweep.append({"n_regimes": K, "loglik": f["loglik"], "n_params": k, "aic": aic, "aicc": aicc, "bic": bic,
                  "min_cluster": int(np.bincount(f["z"], minlength=K).min())})    # users, smallest regime
sweep = pd.DataFrame(sweep)
sweep.to_csv(rdir / f"11_regime_sweep{args.tag}.csv", index=False)
K_aicc = int(sweep.loc[sweep["aicc"].idxmin(), "n_regimes"])
K_bic = int(sweep.loc[sweep["bic"].idxmin(), "n_regimes"])
K = args.n_regimes or K_aicc
print(sweep.round(1).to_string()); print(f"AICc -> {K_aicc}, BIC -> {K_bic}; using {K}")

# ---- final model ----
with Timer(f"final fit, {K} regimes, {args.n_init_final} restarts"):
    fin = M.em_hard(seqs, K, n_states=S, n_init=args.n_init_final, seed=7, n_jobs=args.n_jobs)
# Regimes keep the numbering of the components of the best restart: Regime r + 1 is component r.
z = np.asarray(fin["z"])                           # regime code (0..K-1) of every user, in the order of spk
p0s, As = fin["p0"], fin["A"]                      # (K, S) occupancy (initial) distributions and
                                                   # (K, S, S) transition matrices [regime, to, from]
lab = pd.DataFrame({"speaker_id": spk, "regime": z + 1, "n_windows": [len(s) for s in seqs]})
lab.to_csv(rdir / f"11_regime_labels{args.tag}.csv", index=False)

# ---- fractions of users per regime, bootstrap over users ----
rng = np.random.RandomState(11)
fr_boot = []
with Timer("bootstrap regime fractions"):
    for b in range(args.n_boot):
        idx = rng.randint(0, len(seqs), len(seqs))                   # users drawn with replacement
        fb = M.em_hard([seqs[i] for i in idx], K, n_states=S, n_init=50, seed=b)
        # match the bootstrap regimes to the final ones: L1 distance of transition matrices plus
        # initial distributions, minimized over one-to-one assignments
        D = np.array([[np.abs(fb["A"][a] - As[c]).sum() + np.abs(fb["p0"][a] - p0s[c]).sum()
                       for c in range(K)] for a in range(K)])
        ra, ca = linear_sum_assignment(D)
        zz = np.array([dict(zip(ra, ca))[v] for v in fb["z"]])     # bootstrap labels in final numbering
        fr_boot.append(np.bincount(zz, minlength=K) / len(zz))
fr_boot = np.array(fr_boot)                         # (n_boot, K)
frac = np.bincount(z, minlength=K) / len(z)

# ---- symptom burden relative to Regime 1 (Extended Data Fig. 9c) ----
lab_map = dict(zip(spk, z))                         # speaker_id -> regime code
dseq = df[df["speaker_id"].isin(spk)].copy()        # all windows of the included users
dseq["regime"] = dseq["speaker_id"].map(lab_map)
ref = dseq[dseq["regime"] == 0]                     # windows of the users of Regime 1
burden = {}
for j, t in enumerate(titles):
    thr = np.quantile(ref[keys[j]], 0.9)            # 90th percentile of the factor in Regime 1
    per_user = dseq.groupby("speaker_id").apply(lambda g: np.mean(g[keys[j]] >= thr), include_groups=False)
    reg = pd.Series(lab_map)
    burden[t] = [float(per_user[reg[reg == r].index].mean()) for r in range(K)]   # mean over users
burden = pd.DataFrame(burden, index=[f"Regime {r+1}" for r in range(K)])
burden.to_csv(rdir / f"11_regime_burden{args.tag}.csv")

# ---- outputs ----
np.save(rdir / f"11_regimes{args.tag}.npy", {
    "state_names": state_names, "titles": titles, "p0_pop": p0_pop, "A_pop": A_pop, "A_pop_gap": A_pop_gap,
    "K": K, "K_aicc": K_aicc, "K_bic": K_bic, "p0": p0s, "A": As, "z": z, "speakers": spk,
    "frac": frac, "frac_boot": fr_boot, "seqs": seqs, "steps": steps}, allow_pickle=True)
summ = {"n_sequences": len(seqs), "n_windows": n_obs, "n_states": S, "K_aicc": K_aicc, "K_bic": K_bic,
        "K_used": K, "fractions": frac.round(4).tolist(),
        "fractions_ci": [[float(np.quantile(fr_boot[:, r], .025)), float(np.quantile(fr_boot[:, r], .975))]
                         for r in range(K)],
        "loglik_final": fin["loglik"]}
save_json(summ, rdir / f"11_regimes{args.tag}.json")
print(summ)
print(burden.round(3).to_string())
