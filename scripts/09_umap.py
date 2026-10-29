#!/usr/bin/env python3
"""
Step 9 - UMAP embedding of the factor profiles of all confident windows (Fig. 4a).

Each window's vector of factor scores (12 factors on the 0-100 scale) is embedded with uniform manifold
approximation and projection (UMAP; 20 neighbors, minimum distance 0.08, local connectivity 1,
correlation metric, fixed seed; P.UMAP_CONFIG and P.UMAP_SEED) in two dimensions, the embedding shown
in Fig. 4a, and with the same settings in three dimensions, which shows how much of the local structure
the two-dimensional display gives up.  The embedding serves only for display: states are estimated on
the factor scores themselves (step 10).  Trustworthiness with k = 20 neighbors (correlation metric)
measures how well each embedding preserves the neighborhoods of the factor space (Methods, "Discrete
mental states"); it is computed on a random subsample of 3,000 windows, the same for both embeddings,
because its memory grows with the square of the number of windows.

Inputs   data/processed/<run>/factors<tag>.parquet   factor scores nmf_0..nmf_11 per window (step 6)
Outputs  data/processed/<run>/umap<tag>.parquet      speaker_id, window_start, umap3_0..umap3_2,
                                                    umap2_0, umap2_1
         results/<run>/09_umap<tag>.json             trustworthiness_3d_k20, trustworthiness_2d_k20

usage: python3 09_umap.py --run llama31_8b
"""
import argparse

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness

from slm import params as P
from slm.io import Timer, save_json
from slm.states import run_umap

ap = argparse.ArgumentParser(description="UMAP embedding of the factor profiles (Fig. 4a).")
ap.add_argument("--run", required=True,
                help="annotation run: folder under data/processed/ and results/ (main analysis: llama31_8b)")
ap.add_argument("--tag", default="", help="suffix of the factor table and of the outputs (e.g. _k18)")
args = ap.parse_args()
pdir = P.processed_dir(args.run)
rdir = P.results_dir(args.run)

# ---- factor scores of the confident windows ----
df = pd.read_parquet(pdir / f"factors{args.tag}.parquet")
keys = [c for c in df.columns if c.startswith("nmf_")]
X = df[keys].to_numpy(float)

# ---- embeddings in three and two dimensions, and their trustworthiness ----
res = {}
for d in (3, 2):
    with Timer(f"UMAP {d}-D"):
        E = run_umap(X, n_components=d)
    for i in range(d):
        df[f"umap{d}_{i}"] = E[:, i]
    # trustworthiness on a fixed subsample (the full computation needs O(n^2) memory)
    idx = np.random.RandomState(0).choice(len(X), min(3000, len(X)), replace=False)
    res[f"trustworthiness_{d}d_k20"] = float(trustworthiness(X[idx], E[idx], n_neighbors=20,
                                                               metric="correlation"))

# ---- outputs ----
emb = df[["speaker_id", "window_start"] + [c for c in df.columns if c.startswith("umap")]]
emb.to_parquet(pdir / f"umap{args.tag}.parquet", index=False)
save_json(res, rdir / f"09_umap{args.tag}.json")
print(res)
