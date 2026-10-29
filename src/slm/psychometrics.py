"""
Psychometric properties of the NMF factors: internal consistency, item discrimination and
test-retest reliability, with bootstrap intervals.

Supports the Methods section "Psychometric properties" (Fig. 2d, Table 1, Extended Data Fig. 6),
through scripts/07_psychometrics.py; icc_a1 is also used by scripts/14b_validation_local.py.

Every factor is treated as a composite y = sum_i w_i x_i of the 52 symptom ratings x_i, weighted
by the factor's loadings w_i (one row of H).

* Internal consistency: weighted Cronbach's alpha of the composite,
      alpha = k_eff / (k_eff - 1) * (1 - sum_i var(w_i x_i) / var(y)),
      k_eff = (sum_i w_i)^2 / sum_i w_i^2,
  which reduces to the ordinary k / (k - 1) prefactor for equal weights (alpha_weighted), and
  ordinary alpha over the items that carry 50% of the weight (alpha_unweighted).
* Item discrimination: correlation of an item with the rest of the composite (the composite
  without the item's own contribution), averaged over a set of items (top_discrim).
* Test-retest reliability over pairs of adjacent windows of the same user (adjacent_pairs):
  weighted kappa with linear weights on binned scores (kappa_binned) and ICC(A,1) on the
  continuous scores (icc_a1).
* Intervals: bootstrap over users (hierarchical_bootstrap) and percentile intervals (ci).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import params as P


# --------------------------------------------------------------------------------------------
# Internal consistency
# --------------------------------------------------------------------------------------------

def alpha_weighted(data: np.ndarray, weights: np.ndarray) -> float:
    """Weighted Cronbach's alpha of the composite y = sum_i w_i x_i.

    data    : ratings (n_windows, k)
    weights : item weights w (k,), here the loadings of one factor

    alpha = k_eff / (k_eff - 1) * (1 - sum_i var(w_i x_i) / var(y)), with the effective number of
    items k_eff = (sum_i w_i)^2 / sum_i w_i^2 (k_eff = k for equal weights, so that the formula
    reduces to the ordinary alpha).  Variances are population variances (ddof = 0), whose
    normalization cancels in the ratio.  Returns alpha as a float.
    """
    k = data.shape[1]
    ys = data @ weights.reshape(k, 1)                  # composite y of every window
    ys_var = np.var(ys)
    yks_var = np.var(data * weights.reshape(1, k), axis=0)   # var(w_i x_i) of every item
    keff = weights.sum() ** 2 / (weights ** 2).sum()
    pref = keff / (keff - 1)
    return float(pref * (1 - yks_var.sum() / ys_var))


def k_effective(weights: np.ndarray) -> float:
    """Effective number of items of a weighted composite, k_eff = (sum_i w_i)^2 / sum_i w_i^2.
    It equals k for k equal weights and is small when a few items carry most of the weight
    (6.3-21.3 for the 12 factors)."""
    return float(weights.sum() ** 2 / (weights ** 2).sum())


def alpha_unweighted(data: np.ndarray) -> float:
    """Ordinary Cronbach's alpha of the unweighted sum of the k columns of data (n_windows, k):
    alpha = k / (k - 1) * (1 - sum_i var(x_i) / var(sum_i x_i)), population variances."""
    k = data.shape[1]
    tot = data.sum(axis=1)
    return float(k / (k - 1) * (1 - np.var(data, axis=0).sum() / np.var(tot)))


def top_weight_items(weights: np.ndarray, frac: float = 0.5, low: bool = False) -> np.ndarray:
    """Items that carry a share `frac` of a factor's weight, or as many items of lowest weight.

    weights : item weights (k,)
    frac    : share of the total weight
    low     : False -> the smallest set of most heavily weighted items whose weights sum to at least
              frac * sum(w) (the same set as nmf.weight_fraction_sets and Table 1);
              True  -> the same number of items with the lowest weights

    Returns the item indices (heaviest first for low=False; for low=True, the lightest items in
    order of decreasing weight).
    """
    idx = np.argsort(-weights)                         # items by decreasing weight
    cum = np.cumsum(weights[idx]) / weights.sum()      # cumulative share of the weight
    k_use = int(np.where(cum >= frac)[0][0]) + 1       # first position where the share reaches frac
    return idx[-k_use:] if low else idx[:k_use]


# --------------------------------------------------------------------------------------------
# Item discrimination
# --------------------------------------------------------------------------------------------

def item_rest_r(data: np.ndarray, weights: np.ndarray, item: int) -> float:
    """Item-rest correlation: Pearson r between item `item` and the composite of the other items,
    sum_{i != item} w_i x_i (the item's own contribution is removed, so the correlation is not
    inflated by the item correlating with itself).

    data : ratings (n_windows, k); weights : item weights (k,).  Returns r as a float.
    """
    ys = (data @ weights.reshape(-1, 1)).ravel()       # full composite
    ys = ys - weights[item] * data[:, item]            # ... minus the item's contribution
    return float(np.corrcoef(data[:, item], ys)[0, 1])


def top_discrim(data, weights, frac=P.DISCRIM_TOP_FRAC, low=False, conf=None, conf_q=None, conf_dir=None):
    """Mean item-rest correlation over the items that carry `frac` of a factor's weight.

    data     : ratings (n_windows, k)
    weights  : loadings of the factor (k,)
    frac     : share of the weight (0.5: the items that carry 50% of the weight)
    low      : True -> the same number of items with the lowest weights instead (comparison set)
    conf     : optional confidences (n_windows, k); with conf, the correlation of each item is computed
               only in the windows in which that item's confidence lies in a stratum:
    conf_q   : quantile that bounds the stratum (0.25 or 0.75)
    conf_dir : "le" -> confidence <= its conf_q quantile (bottom stratum);
               "ge" -> confidence >= its conf_q quantile (top stratum)

    In a stratum, each item is correlated with its rest composite over the selected windows only.
    Returns the mean over the items, ignoring items whose correlation is undefined (e.g. constant
    within a stratum).
    """
    items = top_weight_items(weights, frac, low)
    vals = []
    for i in items:
        if conf is None:
            vals.append(item_rest_r(data, weights, i))
        else:
            th = np.quantile(conf[:, i], conf_q)       # the item's own confidence quantile
            m = conf[:, i] <= th if conf_dir == "le" else conf[:, i] >= th
            vals.append(item_rest_r(data[m], weights, i))
    return float(np.nanmean(vals))


# --------------------------------------------------------------------------------------------
# Test-retest reliability
# --------------------------------------------------------------------------------------------

def adjacent_pairs(df: pd.DataFrame, values: np.ndarray, dt_max_days: float = P.KAPPA_DT_MAX_DAYS,
                   group_key="speaker_id", t_key="window_start") -> np.ndarray:
    """All pairs of consecutive windows of the same user that are less than dt_max_days apart.

    df          : one row per window, with the user (group_key) and the window start (t_key)
    values      : score of every row of df (same order as df)
    dt_max_days : maximum distance in days; 14.01 keeps exactly the windows 14 days apart, i.e.
                  adjacent windows (window starts lie on a 14-day grid of wall-clock times)

    Windows are sorted by user and start; every window is paired with the next window of the same
    user if that window starts less than dt_max_days after it.  A window can therefore appear in
    two pairs (as the second window of one pair and the first window of the next).
    Returns an array (n_pairs, 2): values of the first and of the second window of every pair.
    """
    d = df[[group_key, t_key]].copy()
    d["_v"] = values
    d = d.sort_values([group_key, t_key], kind="stable")
    g = d[group_key].to_numpy()
    t = d[t_key].to_numpy()
    v = d["_v"].to_numpy()
    lim = np.timedelta64(int(round(dt_max_days * 86400e6)), "us")   # dt_max_days in microseconds
    same = g[1:] == g[:-1]                             # consecutive rows of the same user
    dt = (t[1:] - t[:-1]) < lim                        # ... at most dt_max_days apart
    m = same & dt
    return np.c_[v[:-1][m], v[1:][m]]


def kappa_binned(pairs: np.ndarray, bins: np.ndarray) -> float:
    """Weighted kappa with linear weights between the two scores of each pair, after binning.

    pairs : array (n_pairs, 2) of continuous scores (first window, second window of each pair)
    bins  : increasing bin edges (n_bins + 1,), e.g. the tertiles of the score over all windows;
            bins are right-open (np.digitize), and scores outside the edges go to the first or
            last bin

    With O the n_bins x n_bins table of observed pair counts, E the counts expected from the two
    marginal distributions (n_pairs * p_first(a) * p_second(b)) and linear disagreement weights
    w_ab = |a - b| / (n_bins - 1):
        kappa = 1 - sum_ab w_ab O_ab / sum_ab w_ab E_ab.
    Returns kappa as a float.
    """
    n_bins = len(bins) - 1
    ii = np.arange(n_bins)
    w = np.abs(ii[None, :] - ii[:, None]) / (n_bins - 1)   # linear disagreement weights
    b = np.digitize(pairs, bins) - 1                   # bin index of every score, 0..n_bins-1
    b = np.clip(b, 0, n_bins - 1)
    O = np.zeros((n_bins, n_bins))
    np.add.at(O, (b[:, 0], b[:, 1]), 1)                # observed table
    p0 = np.bincount(b[:, 0], minlength=n_bins) / len(b)   # marginal of the first window
    p1 = np.bincount(b[:, 1], minlength=n_bins) / len(b)   # marginal of the second window
    E = len(b) * np.outer(p0, p1)                      # expected table under independence
    return float(1 - (w * O).sum() / (w * E).sum())


def icc_a1(pairs: np.ndarray) -> float:
    """ICC(A,1): two-way random-effects intraclass correlation, absolute agreement, single
    measurement (McGraw and Wong 1996), for n subjects rated by k raters.

    pairs : array (n, k) of ratings; for test-retest, rows are window pairs and the two columns
            the first and the second window of the pair

    With the mean squares of rows (MS_R), columns (MS_C) and residual (MS_E) of the two-way
    layout:  ICC(A,1) = (MS_R - MS_E) / (MS_R + (k - 1) MS_E + k (MS_C - MS_E) / n).
    Returns the ICC as a float.
    """
    x = np.asarray(pairs, float)
    n, k = x.shape
    gm = x.mean()
    ms_r = k * ((x.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    ms_c = n * ((x.mean(axis=0) - gm) ** 2).sum() / (k - 1)
    ss_e = ((x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + gm) ** 2).sum()
    ms_e = ss_e / ((n - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n))


# --------------------------------------------------------------------------------------------
# Bootstrap intervals
# --------------------------------------------------------------------------------------------

def hierarchical_bootstrap(stat, df: pd.DataFrame, n: int = 200, key: str = "speaker_id",
                           relabel: bool = False, seed: int = 0) -> np.ndarray:
    """Bootstrap over users: resample users with replacement and recompute a statistic.

    stat    : function DataFrame -> float, applied to each resampled table
    df      : one row per window, with the user in column `key`
    n       : number of resamples
    relabel : True -> every drawn copy of a user gets its own label ("<user>___<draw>"), so that a
              user drawn twice contributes two separate series (needed for statistics that pair
              windows within users, such as test-retest kappa)
    seed    : seed of a new generator (np.random.RandomState) created for this call, so the
              resamples depend only on the seed and on the users of df (in order of first
              appearance), not on other bootstrap calls

    Each resample draws as many users as df contains and stacks all their windows.  A resample in
    which `stat` fails gives NaN.  Returns the n statistics (n,).
    """
    rng = np.random.RandomState(seed)
    groups = {k: g for k, g in df.groupby(key, sort=False)}   # users in order of first appearance
    keys = np.array(list(groups.keys()), dtype=object)
    out = np.full(n, np.nan)
    for b in range(n):
        sel = rng.choice(len(keys), len(keys), replace=True)
        blocks = []
        for j, i in enumerate(sel):
            g = groups[keys[i]]
            if relabel:
                g = g.copy(); g[key] = f"{keys[i]}___{j}"
            blocks.append(g)
        try:
            out[b] = stat(pd.concat(blocks, ignore_index=True))
        except Exception:  # noqa: BLE001
            out[b] = np.nan
    return out


def ci(x, level=0.95):
    """Percentile interval of bootstrap statistics x, ignoring non-finite values.
    Returns (lower, upper) = the (1 - level)/2 and (1 + level)/2 quantiles, or (nan, nan)."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    a = (1 - level) / 2
    return (float(np.quantile(x, a)), float(np.quantile(x, 1 - a)))
