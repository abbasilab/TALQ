#!/usr/bin/env python3
"""
Step 10c - descriptive labels of the discrete states of a solution (after step 10 or 10b; Fig. 4a,b).

States are labeled by the rule of the Methods ("Discrete mental states"; slm.states.describe_states):
the factors whose centroid exceeds the mean over states by at least one standard deviation, at most
two per label in decreasing order of their standardized centroid (e.g. "SH+SS" for Self-harm tendency
and Social strain), and the highest factor in parentheses when no factor qualifies (e.g. "(AT)").
States with the same label are told apart by their burden, the mean centroid over the factors that are
not positive: "lower", "higher" (and "highest").  The same rule names every solution: the 11 states of
the main analysis, the eight states of the sensitivity analysis and the states of the K = 18 factors.

The integer state codes are unchanged; only the names are written.

Inputs   data/processed/<run>/states<tag>.parquet, factors<tag>.parquet   (step 10 or 10b; step 6)
         results/<run>/06_nmf<tag>.npz                                    factor titles
Outputs  results/<run>/10_states<tag>.json             state_names and naming updated (other keys kept)
         results/<run>/10_state_centroids<tag>.csv     rewritten: state (label), mean factor scores,
                                                       n_windows, one row per state code

usage: python3 10c_name_states.py --run llama31_8b
       python3 10c_name_states.py --run llama31_8b --tag _8states
       python3 10c_name_states.py --run llama31_8b --tag _k18
"""
import argparse

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from slm import params as P
from slm.io import load_json, save_json
from slm.states import describe_states

ap = argparse.ArgumentParser(description="Descriptive labels of the discrete states.")
ap.add_argument("--run", default=P.MAIN_RUN,
                help="annotation run: folder under data/processed/ and results/ (default: %(default)s)")
ap.add_argument("--tag", default="", help="suffix of the state solution, e.g. _8states or _k18")
args = ap.parse_args()
pdir, rdir = P.processed_dir(args.run), P.results_dir(args.run)

# ---- centroids of the states (mean factor scores of their windows) ----
st = pd.read_parquet(pdir / f"states{args.tag}.parquet")
fac = pd.read_parquet(pdir / f"factors{args.tag}.parquet")
titles = list(np.load(rdir / f"06_nmf{args.tag}.npz", allow_pickle=True)["titles"])
keys = [f"nmf_{j}" for j in range(len(titles))]
d = fac.merge(st[["speaker_id", "window_start", "state"]], on=["speaker_id", "window_start"])
S = int(d["state"].max()) + 1
cent = np.array([d.loc[d["state"] == s, keys].mean().to_numpy() for s in range(S)])   # (S, n_factors)

# ---- labels ----
names = describe_states(cent, titles)
naming = "descriptive labels from the centroids"

# ---- outputs ----
jp = rdir / f"10_states{args.tag}.json"
js = load_json(jp) if jp.exists() else {}
js.update({"state_names": names, "naming": naming})
save_json(js, jp)
cp = rdir / f"10_state_centroids{args.tag}.csv"
cdf = pd.DataFrame(cent, columns=titles)
cdf.insert(0, "state", names)
cdf["n_windows"] = [int((d["state"] == s).sum()) for s in range(S)]
cdf.to_csv(cp, index=False)
print(naming, names)
