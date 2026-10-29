"""
UMAP embedding and discrete mental states (truncated Dirichlet process Gaussian mixture).

Supports the Methods section "Discrete mental states", Fig. 4a,b and Extended Data Fig. 8, through
scripts 09_umap.py, 10_states_dpmm.py, 10b_alt_states.py and 10c_name_states.py.

Embedding.  The factor vectors of the confident windows (12 factor scores on the 0-100 scale) are
embedded with UMAP (20 neighbors, minimum distance 0.08, correlation metric; P.UMAP_CONFIG).  The
embedding is used for display only (Fig. 4a); states are estimated on the factor scores themselves.

States.  The factor vectors are clustered with a truncated Dirichlet process Gaussian mixture fitted by
variational inference (scikit-learn BayesianGaussianMixture with a Dirichlet process prior on the
weights, at most 100 components, full covariances, 10 initializations; P.DPMM_FIT).  The mixture is
fitted at 14 hyperparameter settings ("steps", dpmm_param_sweep) that run from the least regularized
(step 0) to the most regularized (step 13): the prior on the weight concentration decreases from
1e-8 to 1e-14 (smaller values favor fewer components), the prior on the precision of the means
increases from 0.001 to 3 (stronger shrinkage of the component means toward the mean of the data) and
the regularization added to the diagonal of the covariances increases from 10 to 150 (squared points
of the 0-100 scale; broader components).  At every step, the most populated components that together
cover 90 % of the windows are retained and every window is assigned to the retained component with the
highest posterior probability (relabel_truncated).

Model selection.  Each truncated mixture is scored as an ordinary Gaussian mixture made of its retained
components (gmm_information_criteria), with K = number of retained components, d = number of factors
and n = number of windows:
    k    = K (d + d (d + 1) / 2) + (K - 1)        means, full covariances and K - 1 free weights
    AIC  = 2 k - 2 log L
    AICc = AIC + (2 k^2 + 2 k) / (n - k - 1)      (NaN when n - k - 1 <= 0)
    BIC  = k ln(n) - 2 log L
The main analysis uses the BIC minimum (step 5, 11 states); the sensitivity analysis uses the
adjacent, more regularized step (step 6, eight states).

Labels.  describe_states names each state by the factors that characterize its centroid (Methods:
factors whose centroid exceeds the mean over states by at least one standard deviation, at most two
per label, the highest factor in parentheses when none qualifies).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture

from . import params as P


def run_umap(X: np.ndarray, n_components: int = 3, seed: int = P.UMAP_SEED, **kw):
    """UMAP embedding of the rows of X.

    X            : (n_windows, n_factors) array of factor scores.
    n_components : dimension of the embedding (2 for Fig. 4a).
    seed         : random_state of UMAP; fixing it makes the embedding reproducible.
    kw           : overrides of P.UMAP_CONFIG (n_neighbors 20, min_dist 0.08, local_connectivity 1,
                   metric "correlation").
    Returns an (n_windows, n_components) array of coordinates.
    """
    from umap import UMAP          # imported here: umap-learn is slow to import and only needed here

    cfg = dict(P.UMAP_CONFIG); cfg.update(kw)
    return UMAP(random_state=seed, n_components=n_components, **cfg).fit_transform(X)


def dpmm_param_sweep(keypoints=P.DPMM_PARAM_KEYPOINTS, ns=P.DPMM_N_INTERP, kinds=P.DPMM_PARAM_KINDS):
    """Hyperparameter settings ("steps") of the Dirichlet process mixture sweep.

    keypoints : dict hyperparameter -> list of keypoint values (four per hyperparameter).
    ns        : ns[i] = number of equal intervals between keypoints i and i + 1.
    kinds     : dict hyperparameter -> "log" (intervals equal on a log10 scale) or "linear".
    Returns a DataFrame with one row per step and one column per hyperparameter (the keyword
    arguments weight_concentration_prior, mean_precision_prior and reg_covar of
    BayesianGaussianMixture).  With ns = [6, 5, 2] there are 6 + 5 + 2 + 1 = 14 steps, and the
    keypoints fall on steps 0, 6, 11 and 13 (step 12, for example, lies halfway between the last two
    keypoints: weight concentration 1e-13, mean precision sqrt(3), covariance regularization 125).
    """
    out = {k: [] for k in keypoints}
    for k, vs in keypoints.items():
        for i, (v0, v1) in enumerate(zip(vs[:-1], vs[1:])):
            out[k].append(v0)                                   # the keypoint itself
            if kinds.get(k, "linear") == "log":
                vi = np.logspace(np.log10(v0), np.log10(v1), ns[i] + 1)
            else:
                vi = np.linspace(v0, v1, ns[i] + 1)
            out[k] += list(vi[1:-1])                            # the ns[i] - 1 interior points
        out[k].append(vs[-1])                                   # the last keypoint
    return pd.DataFrame(out)


def relabel_truncated(model, X, states_raw, frac=P.DPMM_TRUNC_FRAC):
    """Truncate a fitted mixture to its most populated components and reassign every window.

    model      : fitted BayesianGaussianMixture.
    X          : (n_windows, n_factors) data the model was fitted on.
    states_raw : (n_windows,) component of maximum posterior probability of each window (model.predict).
    frac       : share of windows covered by the retained components (0.9).
    Components are sorted by their number of windows and retained, from the largest down, as long as
    their cumulative share of windows does not exceed `frac`; the component that would carry the
    cumulative share above `frac` and all smaller ones are dropped (if the largest component alone
    exceeds `frac`, it is the only one retained).  Every window, including those of dropped components,
    is then assigned to the retained component with the highest posterior probability.
    Returns (states, keep): states (n_windows,) holds the mixture's own indices (0-99) of the retained
    components (not consecutive codes), keep the retained component indices in decreasing size.
    """
    labs, counts = np.unique(states_raw, return_counts=True)
    order = np.argsort(-counts)                                  # components from the largest down
    cum = np.cumsum(counts[order]) / counts.sum()
    i_cut = int(np.where(cum > frac)[0][0]) - 1                  # last position with cum <= frac
    keep = labs[order][: i_cut + 1]
    if len(keep) == 0:
        keep = labs[order][:1]
    proba = model.predict_proba(X)                               # (n_windows, n_components) posteriors
    pr = proba[:, keep]
    return keep[np.argmax(pr, axis=1)], keep


def fit_dpmm_step(X, step_params: dict, seed: int, n_components=P.DPMM_N_COMPONENTS_MAX,
                  fit=P.DPMM_FIT, frac=P.DPMM_TRUNC_FRAC):
    """Fit the Dirichlet process Gaussian mixture at one hyperparameter setting and truncate it.

    X           : (n_windows, n_factors) factor scores.
    step_params : one row of dpmm_param_sweep() as a dict (weight_concentration_prior,
                  mean_precision_prior, reg_covar).
    seed        : random_state of the fit (one fixed seed per step, drawn in 10_states_dpmm.py).
    n_components: truncation level of the variational approximation (100).
    fit         : remaining BayesianGaussianMixture arguments (full covariances, tol 1e-3,
                  max_iter 1000, n_init initializations of which the best lower bound is kept).
    frac        : coverage of the retained components (see relabel_truncated).
    Returns (model, raw, states, keep): the fitted model, the maximum-posterior component of every
    window before truncation, the component of every window after truncation and the retained
    components.
    """
    m = BayesianGaussianMixture(n_components=n_components, random_state=seed,
                                weight_concentration_prior_type="dirichlet_process",
                                **step_params, **fit)
    m.fit(X)
    raw = m.predict(X)
    states, keep = relabel_truncated(m, X, raw, frac)
    return m, raw, states, keep


def gmm_information_criteria(model, X, states):
    """AIC, AICc and BIC of a truncated mixture, scored as a Gaussian mixture of its retained components.

    model  : fitted BayesianGaussianMixture.
    X      : (n_windows, n_factors) data.
    states : (n_windows,) retained component of every window (relabel_truncated); the retained
             components are the distinct values.
    The log likelihood is that of a Gaussian mixture whose components are the retained components,
    with the variational point estimates of their means and covariances (model.means_,
    model.covariances_) and their weights renormalized to sum to 1:
        log L = sum_i log sum_c w_c N(x_i; mu_c, Sigma_c).
    Parameters: k = K (d + d (d + 1) / 2) + (K - 1) for K retained components in d dimensions (means,
    full covariances, free weights); sample size n = number of windows.
    Returns a dict with n_params, loglik, aic, aicc, bic and n (see _criteria).
    """
    from scipy.special import logsumexp
    from scipy.stats import multivariate_normal

    used = np.unique(states)
    w = model.weights_[used]; w = w / w.sum()                    # renormalized over retained components
    comp = np.column_stack([multivariate_normal(model.means_[c], model.covariances_[c],
                                                allow_singular=True).logpdf(X) for c in used])
    ll = float(logsumexp(comp + np.log(w)[None, :], axis=1).sum())
    d = X.shape[1]; K = len(used)
    k = K * (d + d * (d + 1) / 2) + (K - 1)
    return _criteria(k, ll, X.shape[0])


def _criteria(k, ll, n):
    """Information criteria from a parameter count k, a log likelihood ll and a sample size n.

    AIC = 2k - 2 ll;  AICc = AIC + (2k^2 + 2k) / (n - k - 1), NaN when n - k - 1 <= 0 (the correction
    is defined only for n > k + 1);  BIC = k ln(n) - 2 ll.
    Returns a dict with keys n_params, loglik, aic, aicc, bic, n (the columns of 10_dpmm_sweep.csv).
    """
    aic = 2 * k - 2 * ll
    aicc = aic + (2 * k * k + 2 * k) / (n - k - 1) if n - k - 1 > 0 else np.nan
    bic = k * np.log(n) - 2 * ll
    return {"n_params": k, "loglik": ll, "aic": aic, "aicc": aicc, "bic": bic, "n": n}


# Abbreviations of the factor titles used in the state labels
FACTOR_ABBR = {"Anxious tension": "AT", "Social strain": "SS", "Positive wellbeing": "PW",
               "Anhedonia": "AN", "Emotional pain": "EP", "Sleep disturbance": "SD",
               "Physical distress": "PD", "Fearful preoccupation": "FP", "Emotional instability": "EI",
               "Negative self-reflection": "NSR", "Self-harm tendency": "SH", "Overwhelm": "OW",
               # factors that appear only in the K = 18 sensitivity analysis
               "Depressed mood": "DM", "Diurnal variation": "DV", "Withdrawal and indecision": "WI",
               "Tearfulness": "TF", "Feeling needed": "FN", "Panic and arousal": "PA",
               "Appetite and weight change": "AW", "Fatigue and slowing": "FS", "Illness worry": "IW",
               "Grief": "GR"}
# Factors of positive content, left out of the burden that separates states with the same label
POSITIVE_FACTORS = {"Positive wellbeing", "Feeling needed"}


def describe_states(centroids: np.ndarray, factor_titles: list[str], z_min: float = 1.0) -> list[str]:
    """Short descriptive labels of states from their centroids.

    centroids     : (n_states, n_factors) mean factor score of each state.
    factor_titles : factor names in column order (abbreviated with FACTOR_ABBR; unknown titles by
                    their first three letters).
    z_min         : threshold in standard deviations (1).
    Each factor is standardized across states, z = (centroid - mean over states) / SD over states
    (population SD).  A state is labeled by the factors with z >= z_min, at most two, in decreasing
    order of z and joined by "+" (e.g. "SH+SS"); when no factor reaches z_min, by its highest factor in
    parentheses (e.g. "(AT)").  States that receive the same label are separated by their burden, the
    mean centroid over the factors that are not positive (POSITIVE_FACTORS excluded): the suffixes
    "lower", "higher" and "highest" (third and any further state) are added in increasing order of
    burden.
    Returns a list of n_states labels in the row order of `centroids`.
    """
    C = np.asarray(centroids, float)
    Z = (C - C.mean(axis=0)) / (C.std(axis=0) + 1e-9)             # 1e-9 guards against a constant factor
    abbr = [FACTOR_ABBR.get(t, t[:3]) for t in factor_titles]
    neg = [i for i, t in enumerate(factor_titles) if t not in POSITIVE_FACTORS]
    burden = C[:, neg].mean(axis=1)
    labels = []
    for z in Z:
        top = [j for j in np.argsort(-z) if z[j] >= z_min][:2]
        labels.append("+".join(abbr[j] for j in top) if top else f"({abbr[int(np.argmax(z))]})")
    # Duplicates: each group of equal labels is handled on its own, so the result does not depend on
    # the iteration order of the set (suffixed labels contain a space and cannot match another label).
    for lab in set(labels):
        idx = [i for i, x in enumerate(labels) if x == lab]
        if len(idx) > 1:
            for rank, i in enumerate(sorted(idx, key=lambda i: burden[i])):
                labels[i] = f"{lab} {['lower', 'higher', 'highest'][min(rank, 2)]}"
    return labels
