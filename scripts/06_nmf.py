#!/usr/bin/env python3
"""
Step 6 - nonnegative matrix factorization of the symptom profiles (staNMF).

Supports the Methods section "Latent mental factors": the choice of K = 12 by the staNMF
instability (Fig. 2b), the final model and its loadings (Fig. 2a, Table 1, Extended Data
Fig. 5d) and the K = 18 sensitivity solution (Extended Data Fig. 5e).  The methods are documented
in src/slm/nmf.py.

What the script does
  * stability sweep (optional): for each K in --sweep-ks (or --k-min..--k-max), --reps NMF fits
    from different random initializations (seed --seed) and their staNMF instability;
  * final model: --reps-final fits at K = --k (seed --seed + 1); the fit with the lowest
    reconstruction error is kept, and its instability is added to the sweep table;
  * factor scores are rescaled to 0-100 by each factor's maximum over windows;
  * names and order of the factors: at K = 12, every fitted component is assigned to one of the
    12 named factors by the reference signatures SIG (below), which fixes the factor order
    (columns nmf_0..nmf_11 follow SIG); at any other K, factors are ordered by total loading and
    named "Factor 1", "Factor 2", ... or by the YAML list given with --names.

Inputs
  data/processed/<run>/collated_confident.parquet   confident windows: mean rating and confidence
                                                    of each of the 52 symptoms (step 5; --input)
  config/symptoms.yaml                              symptom order and titles
  --names PATH (optional)                           factor names for K other than 12

Outputs (<tag> = --tag)
  data/processed/<run>/factors<tag>.parquet  speaker_id, window_start, scores nmf_0..nmf_{K-1}
                                             (0-100), then the remaining columns of the input
  results/<run>/06_nmf<tag>.npz              W (unscaled scores, windows x K), H (loadings, K x 52),
                                             titles, codes (symptom short codes), errs (errors of
                                             all final fits), order (fitted component of each factor)
  results/<run>/06_nmf_sweep<tag>.csv        K, instability_W, instability_H, n_reps, min_err
  results/<run>/06_factor_table<tag>.csv     items that carry 25%, 50% and 90% of each factor's weight
  results/<run>/06_nmf_summary<tag>.json     K, data shape, titles, naming rule, signature shares,
                                             instability of the final K
  With --sweep-only, only 06_nmf_sweep<tag>.csv is written.

Usage (main analysis, run from scripts/)
  python3 06_nmf.py --run llama31_8b --reps 20 --reps-final 100 --sweep-ks none
  python3 06_nmf.py --run llama31_8b --sweep-only --sweep-ks 11,13,14 --reps 100 --tag _fine
  python3 06_nmf.py --run llama31_8b --sweep-only --k-min 3 --k-max 25 --reps 20 --tag _broad
K = 18 sensitivity analysis
  python3 06_nmf.py --run llama31_8b --k 18 --reps-final 100 --sweep-ks none --tag _k18 \
      --names config/factor_names_k18.yaml
  python3 06_nmf.py --run llama31_8b --sweep-only --sweep-ks 17,19 --reps 100 --tag _k18_fine
"""
import argparse
from pathlib import Path

import _setup  # noqa: F401
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment

from slm import params as P
from slm.io import Timer, save_json
from slm.nmf import scale_scores, sta_nmf, weight_fraction_sets
from slm.parse import symptom_matrix
from slm.prompts import load_symptoms

ap = argparse.ArgumentParser(description="NMF of the symptom profiles with the staNMF stability criterion.")
ap.add_argument("--run", required=True,
                help="annotation run, i.e. the folder under data/processed/ and results/ (main: llama31_8b)")
ap.add_argument("--k-min", type=int, default=P.NMF_K_RANGE_SWEEP[0],
                help="smallest K of the stability sweep when --sweep-ks is not given (default 3)")
ap.add_argument("--k-max", type=int, default=P.NMF_K_RANGE_SWEEP[1],
                help="largest K of the stability sweep when --sweep-ks is not given (default 25)")
ap.add_argument("--reps", type=int, default=50,
                help="random initializations per K in the stability sweep")
ap.add_argument("--reps-final", type=int, default=P.NMF_N_REPLICATES,
                help="random initializations of the final model at K = --k (default 100)")
ap.add_argument("--k", type=int, default=P.NMF_K, help="number of factors of the final model (default 12)")
ap.add_argument("--n-jobs", type=int, default=2,
                help="parallel workers for the NMF fits (results do not depend on it)")
ap.add_argument("--seed", type=int, default=12345,
                help="seed of the sweep; the final model uses seed + 1")
ap.add_argument("--input", default="collated_confident.parquet",
                help="table of window ratings in data/processed/<run>/ (step 5)")
ap.add_argument("--tag", default="",
                help="suffix of every output file, e.g. _k18 for the K = 18 sensitivity analysis")
ap.add_argument("--sweep-ks", default=None,
                help="comma-separated list of K for the stability sweep (overrides --k-min/--k-max); "
                     "'none' = no sweep")
ap.add_argument("--sweep-only", action="store_true",
                help="run the sweep only and write 06_nmf_sweep<tag>.csv (no final model)")
ap.add_argument("--names", default=None,
                help="YAML list of factor names for K other than 12, in order of decreasing total "
                     "loading (e.g. config/factor_names_k18.yaml; a relative path may be given "
                     "relative to the working directory or to the repository root)")
args = ap.parse_args()

pdir = P.processed_dir(args.run)
rdir = P.results_dir(args.run); rdir.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(pdir / args.input)
X = symptom_matrix(df)                         # windows x 52 mean ratings, symptoms in config order
syms = load_symptoms(); codes = [s["shortcode"] for s in syms]
print(f"X = {X.shape}")

# ---- step: reference signatures of the 12 named factors -------------------------------------
# For each named factor of the main solution, two to four of its most heavily weighted symptoms
# (Table 1 lists the items that carry 50% of each factor's weight; together the signature
# symptoms carry 24-65% of a factor's weight in the main solution, see signature_match in the
# outputs).  At K = 12 every fitted component is assigned to one named factor (below), which
# gives every fit and refit of the main model the same names and a fixed order.  The order of
# this dictionary is the order of the factors in all outputs (columns nmf_0..nmf_11).
SIG = {
    "Anxious tension": ["anxiety", "ip_sensitivity", "feeling_bothered", "inner_tension"],
    "Social strain": ["others_unfriendly", "others_dislike", "loneliness"],
    "Positive wellbeing": ["feeling_happy", "feeling_needed", "feeling_good"],
    "Anhedonia": ["interest_loss", "feeling_inability", "pleasure_loss", "pessimism"],
    "Emotional pain": ["grief", "sadness", "crying"],
    "Sleep disturbance": ["insomnia_late", "insomnia_early", "insomnia_middle", "hypersomnia"],
    "Physical distress": ["appetite_decrease", "somatization", "weight_decrease", "energy_loss"],
    "Fearful preoccupation": ["hypochondriasis", "panic", "anxiety", "arousal"],
    "Emotional instability": ["irritability", "reactivity", "feeling_bothered"],
    "Negative self-reflection": ["guilt", "self_failure", "self_criticalness", "self_punishment"],
    "Self-harm tendency": ["si", "self_punishment"],
    "Overwhelm": ["life_full", "effort_excess", "concentration_dysfunction"],
}
use_sig = args.k == len(SIG)


def read_names(path: str, k: int) -> tuple[list[str], Path]:
    """Read the factor names of a solution with k factors from a YAML list.

    path : file name as given on the command line; a relative path that does not exist relative
           to the working directory is looked up relative to the repository root, so that
           config/... works both from the repository root and from scripts/
    k    : number of factors; the list must hold exactly k distinct names

    Returns (names, resolved path).
    """
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = P.ROOT / p
    names = yaml.safe_load(p.read_text())
    if not (isinstance(names, list) and len(names) == k and all(isinstance(n, str) for n in names)
            and len(set(names)) == k):
        raise SystemExit(f"{p} must be a YAML list of {k} distinct names (one per factor)")
    return names, p


if args.names and use_sig and not args.sweep_only:
    raise SystemExit(f"--names applies to K other than {len(SIG)}; at K = {len(SIG)} the factors "
                     "are named by their reference signatures")
names_k = read_names(args.names, args.k) if (args.names and not args.sweep_only) else None

# ---- step: stability sweep -------------------------------------------------------------------
if args.sweep_ks is None:
    ks = list(range(args.k_min, args.k_max + 1))
elif args.sweep_ks.lower() == "none":
    ks = []
else:
    ks = [int(k) for k in args.sweep_ks.split(",")]
if not args.sweep_only:
    ks = [k for k in ks if k != args.k]          # the final fit supplies the instability at K = k
sweep = None
if ks:
    with Timer(f"staNMF sweep K={ks} ({args.reps} reps)"):
        sweep = sta_nmf(X, ks, n_replicates=args.reps, seed=args.seed, n_jobs=args.n_jobs)

rows = []
if sweep:
    for k, a, b in zip(sweep["ks"], sweep["instability"], sweep["instability_H"]):
        rows.append({"K": k, "instability_W": a, "instability_H": b, "n_reps": args.reps,
                     "min_err": float(sweep["fits"][k]["err"].min())})
if args.sweep_only:
    pd.DataFrame(rows).sort_values("K").to_csv(rdir / f"06_nmf_sweep{args.tag}.csv", index=False)
    print(pd.DataFrame(rows).sort_values("K").to_string())
    raise SystemExit

# ---- step: final model -----------------------------------------------------------------------
# --reps-final fits from random initializations (generator seeded with --seed + 1, independent of
# the sweep); the fit with the lowest reconstruction error is the final model, and the mean
# pairwise distance among all fits is the instability of K = --k.
with Timer(f"final K={args.k} ({args.reps_final} reps)"):
    fin = sta_nmf(X, [args.k], n_replicates=args.reps_final, seed=args.seed + 1, n_jobs=args.n_jobs)
rows.append({"K": args.k, "instability_W": fin["instability"][0], "instability_H": fin["instability_H"][0],
             "n_reps": args.reps_final, "min_err": float(fin["fits"][args.k]["err"].min())})
sw = pd.DataFrame(rows).sort_values("K")
sw.to_csv(rdir / f"06_nmf_sweep{args.tag}.csv", index=False)

W, H = fin["fits"][args.k]["W"], fin["fits"][args.k]["H"]   # windows x K scores, K x 52 loadings

# ---- step: names and order of the factors ------------------------------------------------------
match_score = None
if use_sig:
    # Ssig[i, c] = share of component c's total weight carried by the signature symptoms of named
    # factor i.  The Hungarian algorithm assigns the components to the names one to one so that
    # the summed shares are maximal.
    titles = list(SIG)
    Hn = H / H.sum(axis=1, keepdims=True)           # loadings of each component as shares of its total
    Ssig = np.zeros((len(titles), H.shape[0]))
    for i, n in enumerate(titles):
        v = np.zeros(len(codes))
        for c in SIG[n]:
            v[codes.index(c)] = 1.0                     # indicator of the signature symptoms
        Ssig[i] = Hn @ v
    r, c = linear_sum_assignment(-Ssig)
    order = list(c[np.argsort(r)])                  # order[i] = fitted component named titles[i]
    match_score = [float(Ssig[i, order[i]]) for i in range(len(order))]
    naming = "assigned to the reference signatures of the 12 named factors"
else:
    # Other K: decreasing total loading (row sums of H), named by --names or numbered
    order = list(np.argsort(-H.sum(axis=1)))
    if names_k is not None:
        titles = names_k[0]
        naming = f"ordered by total loading; names from {names_k[1].name}"
    else:
        titles = [f"Factor {i+1}" for i in range(H.shape[0])]
        naming = "ordered by total loading"
W, H = W[:, order], H[order]

# ---- step: scores and outputs ----------------------------------------------------------------
scores = scale_scores(W)                            # 0-100: each factor's maximum over windows -> 100
out = df[["speaker_id", "window_start"]].copy()
for j in range(scores.shape[1]):
    out[f"nmf_{j}"] = scores[:, j]
out = pd.concat([out.reset_index(drop=True),
                 df.drop(columns=["speaker_id", "window_start"]).reset_index(drop=True)], axis=1)
out.to_parquet(pdir / f"factors{args.tag}.parquet", index=False)

# factor table: the heaviest items that carry 25%, 50% and 90% of each factor's weight
tab = []
for j in range(H.shape[0]):
    sets = weight_fraction_sets(H[j])
    h = H[j] / H[j].sum()                           # loadings as shares of the factor's weight
    tab.append({"factor": j + 1, "title": titles[j],
                "top25": "; ".join(syms[i]["title"] for i in sets[0.25]),
                "top50": "; ".join(syms[i]["title"] for i in sets[0.5]),
                "n_items_90": len(sets[0.9]),
                "max_weight_frac": float(h.max()),           # share of the single heaviest item
                "signature_match": match_score[j] if match_score else np.nan})
pd.DataFrame(tab).to_csv(rdir / f"06_factor_table{args.tag}.csv", index=False)
np.savez(rdir / f"06_nmf{args.tag}.npz", W=W, H=H, titles=np.array(titles), codes=np.array(codes),
         errs=fin["fits"][args.k]["err"], order=np.array(order))
save_json({"K": args.k, "X_shape": X.shape, "titles": titles, "naming": naming,
           "signature_match": match_score,
           "final_instability_W": fin["instability"][0], "final_instability_H": fin["instability_H"][0]},
          rdir / f"06_nmf_summary{args.tag}.json")
print(pd.DataFrame(tab)[["factor", "title", "top25", "signature_match"]].to_string())
print(sw.to_string())
