#!/usr/bin/env python3
"""
Step 12c - replication of the factor structure in the two communities.

Supports the Methods section "Latent mental factors" (replicability) and the Results section
"Twelve replicable mental health factors": factors refitted in the users who wrote mostly in
r/depression (370 users) and in those who wrote mostly in r/Anxiety (111 users), compared with
refits in random subsets of users of the same sizes (Extended Data Fig. 5b).

Method
  * community of a user: r/depression if more than half of the utterances in the user's scored
    windows (all windows with posts, confident or not) were written in r/depression, otherwise
    r/Anxiety;
  * the NMF with K factors (K of the main solution, 12) is refitted on the confident windows of
    each group; the refit is the fit with the lowest error among --n-init random initializations
    (at most 4,000 iterations each);
  * the refitted loadings are matched one to one to the main loadings (largest summed cosine
    similarity) and compared by Tucker's congruence phi;
  * the same refit in --n-random random subsets of users of the size of each group gives the
    congruence expected from the smaller sample alone.

Inputs
  data/processed/<run>/factors.parquet    ratings of the confident windows (step 6)
  results/<run>/06_nmf.npz               main loadings H and factor titles (step 6)
  data/processed/windows.parquet         utterance ids of every scored window (step 3)
  data/interim/cohort_texts.parquet      community (subreddit) of every utterance (step 2)
Outputs
  results/<run>/12c_community_congruence.csv  phi of every factor: group, kind ("community" or
                                              "random_<i>"), title, phi
  results/<run>/12c_community.json            group sizes, share of r/depression utterances,
                                              suicidal-ideation ratings by group, and per group
                                              the median and minimum phi, the number of factors
                                              with phi >= 0.85 and >= 0.95, and the median phi of
                                              the random subsets (median and range)

Usage (run from scripts/)
  python3 12c_community_replication.py --run llama31_8b
"""
import argparse

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from slm import params as P
from slm.io import Timer, save_json
from slm.nmf import _fit_one, match_factors, tucker_congruence
from slm.parse import symptom_matrix

ap = argparse.ArgumentParser(description="Refits of the factor model in the two communities.")
ap.add_argument("--run", default=P.MAIN_RUN,
                help="annotation run, i.e. the folder under data/processed/ and results/ "
                     "(default llama31_8b)")
ap.add_argument("--n-init", type=int, default=5,
                help="random initializations of every refit; the fit with the lowest error is kept")
ap.add_argument("--n-random", type=int, default=10,
                help="random subsets of users per community, of the size of that community")
args = ap.parse_args()
pdir, rdir = P.processed_dir(args.run), P.results_dir(args.run)

fac = pd.read_parquet(pdir / "factors.parquet")
fac["speaker_id"] = fac["speaker_id"].astype(str)
nmf = np.load(rdir / "06_nmf.npz", allow_pickle=True)
H, titles = nmf["H"], list(nmf["titles"])         # main loadings (K x 52) and factor names
K = H.shape[0]

# ---- step: community of each user ------------------------------------------------------------------
# majority of the utterances in the user's scored windows (one row per utterance id)
win = pd.read_parquet(P.PROCESSED / "windows.parquet", columns=["speaker_id", "post_ids"])
win["speaker_id"] = win["speaker_id"].astype(str)
ex = win.explode("post_ids").dropna(subset=["post_ids"]).rename(columns={"post_ids": "id"})
ut = pd.read_parquet(P.INTERIM / "cohort_texts.parquet", columns=["id", "subreddit"])
ex = ex.merge(ut, on="id", how="left")
share = (ex.assign(dep=ex["subreddit"].str.lower().eq("depression"))
         .groupby("speaker_id")["dep"].mean())     # share of the user's utterances in r/depression
comm = np.where(share > 0.5, "r/depression", "r/Anxiety")
comm = pd.Series(comm, index=share.index)
fac["community"] = fac["speaker_id"].map(comm)
X = symptom_matrix(fac)                            # confident windows x 52 ratings
# suicidal ideation (si) ratings by group (Results: usually rated zero in the r/Anxiety group, in
# which Self-harm tendency did not replicate)
out = {"n_users": {}, "n_windows": {}, "share_depression_median": float(share.median()),
       "si_rating_median": {g: float(fac.loc[fac["community"] == g, "si_rating"].median())
                            for g in ["r/depression", "r/Anxiety"]},
       "si_rating_mean": {g: float(fac.loc[fac["community"] == g, "si_rating"].mean())
                          for g in ["r/depression", "r/Anxiety"]}}


def refit(mask, seed):
    """Refit the NMF on the windows selected by `mask` (boolean, n_windows) and compare it with
    the main loadings.

    The seeds of the --n-init initializations are drawn from a generator seeded with `seed`; the
    fit with the lowest reconstruction error is kept (at most 4,000 iterations per fit).  Its
    factors are matched one to one to the main factors.  Returns Tucker's phi of every main
    factor (K,), in the order of the main factors.
    """
    best = None
    rng_init = np.random.RandomState(seed)
    for _ in range(args.n_init):
        W_, H_, e, _ = _fit_one(X[mask], K, rng_init.randint(1e9), P.NMF_TOL, 4000)
        if best is None or e < best[1]:
            best = (H_, e)
    perm, _ = match_factors(best[0], H)            # refitted factor paired with each main factor
    return tucker_congruence(best[0][perm], H)


# ---- step: refits in each community and in size-matched random subsets of users ------------------
rows = []
rng = np.random.RandomState(11)                    # draws the random subsets of both groups, in turn
users = fac["speaker_id"].unique()
for g in ["r/depression", "r/Anxiety"]:
    m = (fac["community"] == g).to_numpy()
    out["n_users"][g] = int(fac.loc[m, "speaker_id"].nunique()); out["n_windows"][g] = int(m.sum())
    with Timer(f"refit in {g} ({out['n_users'][g]} users, {m.sum()} windows)"):
        phi = refit(m, 1 if g == "r/depression" else 2)
    rows += [{"group": g, "kind": "community", "title": t, "phi": float(p)} for t, p in zip(titles, phi)]
    # size-matched random subsets of users (any community); refit seeds 100, 101, ...
    n_u = out["n_users"][g]
    with Timer(f"{args.n_random} random subsets of {n_u} users"):
        for r in range(args.n_random):
            sub = set(rng.choice(users, n_u, replace=False))
            mm = fac["speaker_id"].isin(sub).to_numpy()
            phi_r = refit(mm, 100 + r)
            rows += [{"group": g, "kind": f"random_{r}", "title": t, "phi": float(p)}
                     for t, p in zip(titles, phi_r)]

# ---- step: outputs ---------------------------------------------------------------------------------
tab = pd.DataFrame(rows)
tab.to_csv(rdir / "12c_community_congruence.csv", index=False)
for g in ["r/depression", "r/Anxiety"]:
    c = tab[(tab["group"] == g) & (tab["kind"] == "community")]
    # median phi over factors of every random subset
    rnd = tab[(tab["group"] == g) & (tab["kind"] != "community")].groupby("kind")["phi"].median()
    out[g] = {"phi_median": float(c["phi"].median()), "phi_min": float(c["phi"].min()),
              "phi_min_factor": c.loc[c["phi"].idxmin(), "title"],
              "n_ge_085": int((c["phi"] >= 0.85).sum()), "n_ge_095": int((c["phi"] >= 0.95).sum()),
              "random_phi_median_median": float(rnd.median()),
              "random_phi_median_range": [float(rnd.min()), float(rnd.max())]}
save_json(out, rdir / "12c_community.json")
print(out)
