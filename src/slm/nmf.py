"""
Nonnegative matrix factorization (NMF) of the symptom profiles and the staNMF stability criterion.

Supports the Methods sections "Latent mental factors" (choice of K, final model, Fig. 2a,b,
Table 1, Extended Data Fig. 5e) and the replication analyses described there (Tucker
congruence after optimal matching: Fig. 2c, Extended Data Fig. 5b,c).  Used by
scripts/06_nmf.py (factorization), scripts/12_robustness.py and scripts/12c_community_replication.py
(refits and matching), and by the figure scripts (combined_sweep, weight_fraction_sets).

Model
    The matrix X of windows by symptoms (one row per confident window, 52 columns of mean ratings
    on the 0-100 scale) is approximated by X ~ W H with nonnegative factor scores W (windows x K)
    and nonnegative loadings H (K x 52).  Each fit uses scikit-learn's NMF with coordinate descent,
    the Frobenius loss and a tolerance of 1e-5 (params.NMF_TOL).

Initialization
    Every fit starts from its own random initialization: the initial window scores W0 are K
    randomly chosen symptom columns of X, and the initial loadings H0 are drawn uniformly on
    [0, 1).  The seed of each fit is drawn from one generator per call of sta_nmf, so a single
    seed determines all fits of a run.

Stability (staNMF; Wu et al. 2016)
    For a given K, the solutions of n fits from different initializations are compared pairwise
    with the Amari-type distance of sta_diss, and the instability of K is the mean distance over
    all n (n - 1) / 2 pairs (0 = every initialization finds the same factors).  The distance is
    computed on the window scores W (instability_W, the criterion reported in the manuscript,
    e.g. 0.015 at K = 12) and on the loadings H (instability_H, shown alongside in Fig. 2b).

Final model and scaling
    The final model at the chosen K is the fit with the lowest reconstruction error among its
    initializations, and its scores are rescaled to 0-100 by the maximum of each factor over
    windows (scale_scores).

Matching and congruence
    Factors of two solutions are paired one to one by the assignment that maximizes the summed
    cosine similarity of their loading vectors (match_factors; this cosine is Tucker's congruence
    coefficient), and each pair is then summarized by Tucker's congruence phi (tucker_congruence;
    1 for proportional loadings).
"""

from __future__ import annotations

import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

from . import params as P


# --------------------------------------------------------------------------------------------
# Stability across random initializations (staNMF)
# --------------------------------------------------------------------------------------------

def sta_diss(D1: np.ndarray, D2: np.ndarray) -> float:
    """Amari-type distance between two factor solutions (Wu et al. 2016).

    D1, D2 : arrays (n, K) whose K columns are the factors of two solutions, either window
             scores W (n = windows) or transposed loadings H.T (n = 52 symptoms).

    C is the K x K matrix of Pearson correlations between the columns of D1 (rows of C) and the
    columns of D2 (columns of C).  The distance is

        diss = (2K - sum_j max_i C_ij - sum_i max_j C_ij) / (2K),

    which is 0 when every factor of one solution has a perfectly correlated counterpart in the
    other, whatever the order of the factors.  A factor that is zero everywhere has an undefined
    correlation; it is set to 0, i.e. treated as having no counterpart.

    Returns the distance as a float.
    """
    K = D1.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        # np.corrcoef stacks the 2K columns; the off-diagonal block holds corr(D1[:, i], D2[:, j])
        C = np.nan_to_num(np.corrcoef(D1.T, D2.T)[:K, K:], nan=0.0)
    acc = 2.0 * K - np.nanmax(C, axis=0).sum() - np.nanmax(C, axis=1).sum()
    return (1.0 / (2.0 * K)) * acc


def _fit_one(X, K, seed, tol, max_iter):
    """One NMF fit of X with K factors from a random initialization.

    X        : array (n_windows, 52), nonnegative ratings
    K        : number of factors
    seed     : seed of the initialization (numpy RandomState)
    tol      : tolerance of the coordinate-descent solver (params.NMF_TOL = 1e-5)
    max_iter : maximum number of iterations (params.NMF_MAX_ITER = 10,000 for the final model and
               the stability sweeps; 4,000 for the refits of scripts 12 and 12c)

    Initialization: a random permutation of the 52 symptom columns is drawn and its first K
    columns of X become the initial window scores W0; the initial loadings H0 are uniform on
    [0, 1).  Iterations that stop at max_iter without reaching the tolerance are accepted (the
    convergence warning is silenced).

    Returns (W, H, err, n_iter): scores W (n_windows, K), loadings H (K, 52), the reconstruction
    error ||X - W H||_F divided by the number of windows (used only to rank fits), and the number
    of iterations used.
    """
    rng = np.random.RandomState(seed)
    n_features = X.shape[1]
    idx = rng.permutation(n_features)[:K]                  # K randomly chosen symptom columns
    W0 = X[:, idx].copy(order="C")                         # ... serve as initial window scores
    H0 = rng.uniform(size=(K, n_features)).copy(order="C")
    m = NMF(n_components=K, init="custom", tol=tol, max_iter=max_iter)   # solver "cd", Frobenius loss
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        W = m.fit_transform(X, W=W0, H=H0)
    return W, m.components_, m.reconstruction_err_ / X.shape[0], m.n_iter_


def sta_nmf(X: np.ndarray, ks, n_replicates: int = P.NMF_N_REPLICATES, seed: int = 0,
            tol: float = P.NMF_TOL, max_iter: int = P.NMF_MAX_ITER, n_jobs: int = -1,
            verbose: bool = True) -> dict:
    """staNMF: fit NMF from `n_replicates` random initializations for each K and measure instability.

    X            : array (n_windows, 52), nonnegative ratings
    ks           : iterable of numbers of factors, processed in the given order
    n_replicates : initializations per K (n); the instability averages n (n - 1) / 2 pairs
    seed         : seed of the generator that draws one seed per fit (np.random.RandomState); the
                   seeds of each K are drawn after those of the K listed before it
    tol, max_iter: solver settings passed to _fit_one
    n_jobs       : parallel workers (joblib); results do not depend on it, because every fit has its
                   own seed and results are collected in input order
    verbose      : print one line per K

    Returns a dict with
        "ks"            : list of K
        "instability"   : mean pairwise sta_diss of the window scores W, one value per K
        "instability_H" : the same for the loadings H
        "fits"          : {K: {"W", "H": scores and loadings of the fit with the lowest error,
                               "err": errors of all fits (n_replicates,), "best": index of that fit,
                               "seeds": seeds of all fits, "n_iter": iterations of all fits}}
    """
    X = np.ascontiguousarray(X, dtype=float)
    rng = np.random.RandomState(seed)
    out = {"ks": list(ks), "instability": [], "instability_H": [], "fits": {}}
    for K in ks:
        seeds = rng.randint(0, 2**31 - 1, size=n_replicates)
        fits = Parallel(n_jobs=n_jobs)(delayed(_fit_one)(X, K, s, tol, max_iter) for s in seeds)
        Ws = [f[0] for f in fits]; Hs = [f[1] for f in fits]
        errs = np.array([f[2] for f in fits])
        dW, dH = [], []
        for b1 in range(n_replicates):                     # all unordered pairs of fits
            for b2 in range(b1 + 1, n_replicates):
                dW.append(sta_diss(Ws[b1], Ws[b2]))            # factors as columns of W (windows x K)
                dH.append(sta_diss(Hs[b1].T, Hs[b2].T))        # factors as columns of H.T (52 x K)
        out["instability"].append(float(np.mean(dW)))
        out["instability_H"].append(float(np.mean(dH)))
        best = int(np.argmin(errs))                        # final model: lowest reconstruction error
        out["fits"][K] = {"W": Ws[best], "H": Hs[best], "err": errs, "best": best,
                          "seeds": seeds, "n_iter": [f[3] for f in fits]}
        if verbose:
            print(f"  K={K:2d}  instability(W)={np.mean(dW):.4f}  instability(H)={np.mean(dH):.4f}"
                  f"  min err={errs.min():.4f}")
    return out


# --------------------------------------------------------------------------------------------
# Scores, matching and summaries of a solution
# --------------------------------------------------------------------------------------------

def scale_scores(W: np.ndarray) -> np.ndarray:
    """Rescale factor scores to 0-100: each column of W (n_windows, K) is multiplied by
    100 / (its maximum over windows), so the highest-scoring window of every factor scores 100.
    The rescaling leaves the reconstruction unchanged if H is divided by the same factors, but
    only the scores are rescaled here (the loadings H are stored as fitted)."""
    return W * (100.0 / W.max(axis=0, keepdims=True))


def match_factors(H: np.ndarray, H_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pair the factors of a solution with those of a reference solution, one to one.

    H     : loadings (K, n_symptoms) of the solution to be matched
    H_ref : loadings (K_ref, n_symptoms) of the reference solution, K_ref <= K

    The similarity of two factors is the cosine of their loading vectors, i.e. Tucker's
    congruence coefficient.  The Hungarian algorithm (scipy linear_sum_assignment)
    finds the one-to-one pairing with the largest summed similarity.

    Returns (perm, sim): perm (K_ref,) gives for each reference factor i the row of H paired with
    it, so H[perm] is aligned with H_ref; sim (K_ref,) holds the cosines of the pairs.
    """
    from scipy.optimize import linear_sum_assignment

    a = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
    b = H_ref / np.maximum(np.linalg.norm(H_ref, axis=1, keepdims=True), 1e-12)
    S = b @ a.T                                            # S[i, c] = cosine(H_ref[i], H[c])
    r, c = linear_sum_assignment(-S)                       # maximize the summed similarity
    perm = np.empty(H_ref.shape[0], dtype=int)
    perm[r] = c
    return perm, S[r, c]


def tucker_congruence(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tucker's congruence coefficient between matched rows of A and B (both (K, n_symptoms)):
    phi_i = sum_j A_ij B_ij / sqrt(sum_j A_ij^2 * sum_j B_ij^2).  Returns an array (K,);
    phi = 1 for proportional loadings, and 0.85 and 0.95 are the conventional thresholds of fair
    and good similarity used in Fig. 2c and Extended Data Fig. 5."""
    num = (A * B).sum(axis=1)
    den = np.sqrt((A * A).sum(axis=1) * (B * B).sum(axis=1))
    return num / np.maximum(den, 1e-12)


def weight_fraction_sets(h: np.ndarray, fracs=(0.25, 0.5, 0.9)) -> dict:
    """Most heavily weighted items of one factor that together carry a given share of its weight.

    h     : loadings of one factor (n_symptoms,)
    fracs : shares of the total weight sum(h)

    For each share f, returns the smallest set of items, taken in order of decreasing loading,
    whose loadings sum to at least f * sum(h) (Table 1 lists the set for f = 0.5).
    Returns {f: array of item indices, heaviest first}.
    """
    order = np.argsort(-h)
    cum = np.cumsum(h[order]) / h.sum()                    # cumulative share of the weight
    out = {}
    for f in fracs:
        k = int(np.searchsorted(cum, f) + 1)               # first position where the share reaches f
        out[f] = order[:k]
    return out


def combined_sweep(rdir, tags=("", "_fine", "_broad")):
    """Merge the staNMF instability tables of one run (results/<run>/06_nmf_sweep<tag>.csv).

    rdir : results folder of the run
    tags : file tags to merge; "" is the table written with the final model (K = 12 at 100
           initializations), "_fine" and "_broad" the sweeps of the Methods

    For each K the estimate with the most initializations is kept.  Returns a DataFrame with
    columns K, instability_W, instability_H, n_reps, min_err and source (the tag without its
    underscore, "main" for the empty tag), sorted by K; empty if no table exists.
    """
    import pandas as pd
    from pathlib import Path
    parts = []
    for t in tags:
        p = Path(rdir) / f"06_nmf_sweep{t}.csv"
        if p.exists():
            d = pd.read_csv(p)
            d["source"] = t.strip("_") or "main"
            parts.append(d)
    if not parts:
        return pd.DataFrame(columns=["K", "instability_W", "instability_H", "n_reps", "min_err", "source"])
    d = pd.concat(parts, ignore_index=True)
    d = d.sort_values(["K", "n_reps"], ascending=[True, False]).drop_duplicates("K")
    return d.reset_index(drop=True)
