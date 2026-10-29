#!/usr/bin/env python3
"""
Step 10b - an alternative discretization from the same Dirichlet process mixture sweep, for the
sensitivity analysis with eight states (Methods, "Discrete mental states" and "Dynamical regimes";
Results, "Recurrent mental states" and "Individuals differ mainly in the states they occupy").

Step 10 stores the truncated state assignment of every sweep step.  This script takes another step out
of that table, chosen by --rule:
  adjacent        the step after the one selected in step 10 (selected_step + 1 of 10_states.json), i.e.
                  the adjacent, more regularized setting; step 6 with eight states in the main analysis;
  bic, aic, aicc  the step at which the criterion of 10_dpmm_sweep.csv is lowest;
or given directly with --step.  The states are coded 0..S-1 in decreasing order of their number of
windows and provisionally named S1..S<S> (10c_name_states.py gives them descriptive labels).  The
outputs carry the suffix --tag, and the factor table and factor model of the run are linked under the
same suffix, so that steps 10c, 11, 12 and 19 can be run with --tag on the alternative states.

Inputs   data/processed/<run>/states.parquet, factors.parquet       (steps 10 and 6)
         results/<run>/10_dpmm_sweep.csv, 10_states.json, 06_nmf.npz
Outputs  data/processed/<run>/states<tag>.parquet      speaker_id, window_start, state
         results/<run>/10_state_centroids<tag>.csv     mean factor scores and number of windows per state
         results/<run>/10_states<tag>.json             selected_step, select_rule, n_states, state_names,
                                                       n_windows
         data/processed/<run>/factors<tag>.parquet -> factors.parquet   (relative symbolic links)
         results/<run>/06_nmf<tag>.npz -> 06_nmf.npz

usage: python3 10b_alt_states.py --run llama31_8b --rule adjacent --tag _8states
"""
import argparse
import os

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from slm import params as P
from slm.io import load_json, save_json

ap = argparse.ArgumentParser(description="Alternative state solution from the Dirichlet process "
                                         "mixture sweep.")
ap.add_argument("--run", required=True,
                help="annotation run: folder under data/processed/ and results/ (main analysis: llama31_8b)")
ap.add_argument("--rule", default="adjacent", choices=["adjacent", "bic", "aic", "aicc"],
                help="adjacent: the step after the one selected in step 10 (more regularized); "
                     "bic/aic/aicc: the step with the lowest criterion (default: adjacent)")
ap.add_argument("--step", type=int, default=None, help="use this sweep step (overrides --rule)")
ap.add_argument("--tag", required=True,
                help="suffix of the outputs, e.g. _8states; must not be empty, so that the main "
                     "solution is not overwritten")
args = ap.parse_args()
if not args.tag:
    ap.error("--tag must not be empty")
pdir, rdir = P.processed_dir(args.run), P.results_dir(args.run)

# ---- choice of the step ----
sw = pd.read_csv(rdir / "10_dpmm_sweep.csv")
if args.step is not None:
    step, rule = args.step, "step"
elif args.rule == "adjacent":
    step, rule = int(load_json(rdir / "10_states.json")["selected_step"]) + 1, args.rule
else:
    step, rule = int(sw.loc[sw[args.rule].idxmin(), "step"]), args.rule
if step not in set(sw["step"].astype(int)):
    ap.error(f"step {step} is not in 10_dpmm_sweep.csv (steps {sw['step'].tolist()})")

# ---- state codes in decreasing order of occupancy ----
st = pd.read_parquet(pdir / "states.parquet")
fac = pd.read_parquet(pdir / "factors.parquet")
titles = list(np.load(rdir / "06_nmf.npz", allow_pickle=True)["titles"])
raw = st[f"step_{step}"].to_numpy()                        # retained mixture component of each window
labs, cnt = np.unique(raw, return_counts=True)
order = labs[np.argsort(-cnt)]                             # components, largest first
remap = {o: i for i, o in enumerate(order)}                # component index -> state code
out = st[["speaker_id", "window_start"]].copy()
out["state"] = [remap[x] for x in raw]
out.to_parquet(pdir / f"states{args.tag}.parquet", index=False)

# ---- centroids, summary and links to the factor files ----
d = fac.merge(out, on=["speaker_id", "window_start"])
keys = [f"nmf_{j}" for j in range(len(titles))]
cent = d.groupby("state")[keys].mean(); cent.columns = titles
names = [f"S{i+1}" for i in range(len(order))]
cent.insert(0, "state", names); cent["n_windows"] = d.groupby("state").size().to_numpy()
cent.to_csv(rdir / f"10_state_centroids{args.tag}.csv", index=False)
save_json({"selected_step": step, "select_rule": rule, "n_states": len(order), "state_names": names,
           "n_windows": cent["n_windows"].tolist()}, rdir / f"10_states{args.tag}.json")
for src, dst in [(pdir / "factors.parquet", pdir / f"factors{args.tag}.parquet"),
                 (rdir / "06_nmf.npz", rdir / f"06_nmf{args.tag}.npz")]:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.name, dst)                              # relative link within the same folder
print(f"step {step} ({rule}): {len(order)} states", cent["n_windows"].tolist())
