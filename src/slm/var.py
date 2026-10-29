"""
Panel vector autoregression (VAR) of the factor scores within users.

Supports the Methods section "Factor dynamics within individuals" (Fig. 3, Extended Data Fig. 7),
through scripts/08_var_population.py (main model, order selection, sensitivity analyses, null
calibration).

Model.  For user u and window t, the K factor scores y_ut (here K = 12, 0-100 scale) follow

    y_ut = c + A_1 y_u,t-1 + ... + A_p y_u,t-p + e_ut,

where lags link only windows of the same user that are exactly 1, ..., p windows (14 days each)
apart: a window enters the design only if all its p predecessors were observed, so lags never
cross gaps or users.  Scores are centered within user (stable differences between users are
removed), and the coefficients are estimated by ordinary least squares on the stacked pairs.
A_l[i, j] is the effect of factor j at lag l on factor i (in code, A[l - 1][i, j]).

Bias correction.  Centering within short panels (7,094 confident windows of 481 users) biases
the autoregressive coefficients toward zero (Nickell 1981).  The half-panel jackknife (Dhaene and
Jochmans 2015) removes the leading term of this bias:

    A = 2 A_full - (A_first + A_second) / 2,

where A_first and A_second are estimated on the first and second halves of each user's span,
each half centered separately.  This is the main ("bias-corrected") estimator.

Inference.  A bootstrap over users (the whole procedure, including the jackknife, is repeated on
users resampled with replacement) gives the covariance of the lagged coefficients of every
ordered pair of factors, a Wald test of j -> i ("j Granger-predicts i") with a chi-square
reference distribution, Benjamini-Hochberg FDR control over the K (K - 1) = 132 ordered pairs, and
95% percentile bands of impulse and step responses.  The within estimator without bias
correction is kept as a sensitivity analysis, with cluster-robust Wald tests (clusters = users).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import params as P


# --------------------------------------------------------------------------------------------
# Lag design and the within (fixed-effects) estimator
# --------------------------------------------------------------------------------------------

def lag_design(df: pd.DataFrame, keys: list[str], order: int, demean: bool = True,
               step_days: float = P.WINDOW_DAYS):
    """Stack the lagged pairs of all users: rows whose `order` predecessors were all observed.

    df        : one row per window with columns speaker_id, window_start and the scores `keys`
    keys      : score columns (K factors)
    order     : lag order p
    demean    : center every user's scores on that user's mean over all of their windows
    step_days : window length; a window's position on the user's grid is
                round((window_start - first window_start) / 14 days)

    Returns (Y, X, clusters): Y (n, K) current scores, X (n, K * p) lagged scores ordered
    [lag 1 | lag 2 | ... | lag p], and the user of every row (n,).  Users are processed in
    sorted order.
    """
    Ys, Xs, cl = [], [], []
    groups = df.groupby("speaker_id", sort=True)
    for s, g in groups:
        g = g.sort_values("window_start")
        V = g[keys].to_numpy(float)
        if demean:
            V = V - V.mean(axis=0, keepdims=True)
        t = (g["window_start"] - g["window_start"].iloc[0]).dt.total_seconds().to_numpy() / 86400.0
        step = np.rint(t / step_days).astype(int)          # grid position of every window
        pos = {st: i for i, st in enumerate(step)}
        for i, st in enumerate(step):
            lags = [pos.get(st - L) for L in range(1, order + 1)]
            if any(l is None for l in lags):              # a predecessor was not observed
                continue
            Ys.append(V[i]); Xs.append(np.concatenate([V[l] for l in lags])); cl.append(s)
    Y = np.array(Ys); X = np.array(Xs)
    return Y, X, np.array(cl, dtype=object)


def fit_panel_var(df, keys, order, demean=True, const=True):
    """Within estimator of the panel VAR (no bias correction): OLS of Y on [1, X] from lag_design.

    Returns a dict with A (list of p arrays (K, K), A[l][i, j] = effect of factor j at lag l + 1 on
    factor i), B (coefficient matrix (1 + K p, K)), the design X (with the constant column), Y,
    residuals E, clusters (user of every row), keys, order, const, nobs (number of lagged pairs)
    and n_speakers.
    """
    Y, X, cl = lag_design(df, keys, order, demean)
    n, k = Y.shape
    Xc = np.c_[np.ones(n), X] if const else X
    B, *_ = np.linalg.lstsq(Xc, Y, rcond=None)          # (1+k*p) x k
    E = Y - Xc @ B
    # rows of B: [constant | lag 1 (k rows) | lag 2 | ...]; column i is the equation of factor i
    A = [B[(1 if const else 0) + L * k:(1 if const else 0) + (L + 1) * k, :].T for L in range(order)]
    return {"A": A, "B": B, "X": Xc, "Y": Y, "E": E, "clusters": cl, "keys": keys,
            "order": order, "const": const, "nobs": n, "n_speakers": len(set(cl))}


def cluster_robust_cov(fit, eq: int):
    """Cluster-robust (sandwich) covariance of the coefficients of equation `eq` of a within fit,
    with users as clusters:  V = c (X'X)^-1 [sum_g X_g' e_g e_g' X_g] (X'X)^-1, with the
    small-sample factor c = G/(G - 1) * (N - 1)/(N - P) (G users, N rows, P regressors)."""
    X, e, cl = fit["X"], fit["E"][:, eq], fit["clusters"]
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(cl):
        m = cl == c
        sc = X[m].T @ e[m]                                 # score contribution of one user
        meat += np.outer(sc, sc)
    G, N, K = len(np.unique(cl)), X.shape[0], X.shape[1]
    adj = G / (G - 1) * (N - 1) / (N - K)
    return adj * XtX_inv @ meat @ XtX_inv


def granger_panel(fit):
    """Cluster-robust Wald tests of the within fit: does factor j Granger-predict factor i?

    For every ordered pair (i, j), i != j, the p lagged coefficients of j in the equation of i are
    tested jointly: W = b' V^+ b with the cluster-robust covariance V, and F = W / p is referred to
    an F distribution with (p, G - 1) degrees of freedom (G users).  P values are FDR-adjusted
    (Benjamini-Hochberg) over all K (K - 1) pairs.

    Returns (P, q, F), each (K, K) with row = response i and column = predictor j (NaN diagonal).
    """
    from statsmodels.stats.multitest import fdrcorrection

    k, p, off = len(fit["keys"]), fit["order"], (1 if fit["const"] else 0)
    G = fit["n_speakers"]
    P_ = np.full((k, k), np.nan); F = np.full((k, k), np.nan)
    for i in range(k):
        V = cluster_robust_cov(fit, i)
        b = fit["B"][:, i]
        for j in range(k):
            if i == j:
                continue
            idx = [off + L * k + j for L in range(p)]          # rows of B holding j's lags
            bb = b[idx]; VV = V[np.ix_(idx, idx)]
            w = float(bb @ np.linalg.pinv(VV) @ bb)
            F[i, j] = w / p
            P_[i, j] = stats.f.sf(F[i, j], p, G - 1)
    q = np.full_like(P_, np.nan); m = ~np.isnan(P_)
    q[m] = fdrcorrection(P_[m].ravel())[1]
    return P_, q, F


# --------------------------------------------------------------------------------------------
# Half-panel jackknife (bias-corrected) panel VAR with a bootstrap over users
# --------------------------------------------------------------------------------------------

def _user_design(V, step, order):
    """Lagged pairs of one user's centered scores V (n_windows, K) at grid positions `step`:
    returns Y (n_pairs, K) and X (n_pairs, K * order), keeping only windows whose `order`
    predecessors were all observed (as in lag_design)."""
    pos = {st: i for i, st in enumerate(step)}
    Ys, Xs = [], []
    for i, st in enumerate(step):
        lags = [pos.get(st - L) for L in range(1, order + 1)]
        if any(l is None for l in lags):
            continue
        Ys.append(V[i]); Xs.append(np.concatenate([V[l] for l in lags]))
    k = V.shape[1]
    return (np.array(Ys).reshape(-1, k), np.array(Xs).reshape(-1, k * order))


def panel_blocks(df: pd.DataFrame, keys: list[str], order: int, step_days: float = P.WINDOW_DAYS):
    """Per-user design blocks for the full series and for its two halves.

    Each user's span (first to last observed grid position) is split at its midpoint: windows at
    positions <= midpoint form the first half ("h1"), the others the second half ("h2").  The full
    series and each half are centered separately on their own means, and lagged pairs are formed
    within each part only.  A part with fewer than order + 1 windows gets empty blocks.

    Returns {user: {"full" | "h1" | "h2": (Y, X)}} with Y (n_pairs, K) and X (n_pairs, K * order).
    """
    blocks = {}
    for s, g in df.groupby("speaker_id", sort=True):
        g = g.sort_values("window_start")
        V = g[keys].to_numpy(float)
        t = (g["window_start"] - g["window_start"].iloc[0]).dt.total_seconds().to_numpy() / 86400.0
        step = np.rint(t / step_days).astype(int)
        mid = 0.5 * (step[0] + step[-1])                   # midpoint of the user's span
        parts = {"full": np.ones(len(step), bool), "h1": step <= mid, "h2": step > mid}
        b = {}
        for name, m in parts.items():
            if m.sum() < order + 1:
                b[name] = (np.zeros((0, len(keys))), np.zeros((0, len(keys) * order)))
                continue
            Vm = V[m] - V[m].mean(axis=0, keepdims=True)   # centered within the part
            b[name] = _user_design(Vm, step[m], order)
        blocks[s] = b
    return blocks


def _ols_A(blocks, users, part, k, order):
    """OLS (with intercept) of the stacked blocks `part` of `users` (repeats allowed, as in a
    bootstrap resample).  Returns (A, n): A (order, K, K) with A[l][i, j] = effect of factor j at
    lag l + 1 on factor i, and the number of stacked pairs n."""
    Y = np.vstack([blocks[u][part][0] for u in users])
    X = np.vstack([blocks[u][part][1] for u in users])
    Xc = np.c_[np.ones(len(X)), X]
    B, *_ = np.linalg.lstsq(Xc, Y, rcond=None)
    return np.array([B[1 + L * k:1 + (L + 1) * k, :].T for L in range(order)]), len(Y)


def hpj_A(blocks, users, k, order):
    """Half-panel jackknife estimate A = 2 A_full - (A_h1 + A_h2) / 2 for a set of users.
    Returns (A_hpj, A_full, n_pairs_full), the arrays of shape (order, K, K)."""
    A, n = _ols_A(blocks, users, "full", k, order)
    A1, _ = _ols_A(blocks, users, "h1", k, order)
    A2, _ = _ols_A(blocks, users, "h2", k, order)
    return 2 * A - 0.5 * (A1 + A2), A, n


def fit_panel_var_hpj(df, keys, order, n_boot=500, seed=0, horizon=P.IRF_HORIZON):
    """Bias-corrected panel VAR(order) with inference from a bootstrap over users.

    df      : one row per window with speaker_id, window_start and the scores `keys`
    keys    : score columns (K factors)
    order   : lag order p
    n_boot  : bootstrap resamples of users (500 in the manuscript)
    seed    : seed of the bootstrap generator (np.random.RandomState)
    horizon : number of two-week steps of the impulse and step responses (12 = 24 weeks)

    Estimation: half-panel jackknife (hpj_A) on all users with at least one lagged pair.
    Bootstrap: users are drawn with replacement and the jackknife estimate is recomputed on the
    stacked blocks of the drawn users (blocks are centered once, per user and half).
    Tests: for every ordered pair (i, j), i != j, the Wald statistic W = a' V^+ a of the p lagged
    coefficients a = A[:, i, j] with their bootstrap covariance V, P = P(chi2_p > W), and
    Benjamini-Hochberg q values over all K (K - 1) pairs.
    Responses: moving-average coefficients Phi_h (ma_coefs), i.e. the response of every factor h
    steps after a unit impulse in one factor; their cumulative sums are the responses to a
    sustained unit change (step responses).  Bands are the 2.5% and 97.5% bootstrap quantiles.

    Returns a dict:
        A (p, K, K) bias-corrected coefficients; A_within (p, K, K) within estimate without
        correction; p, q, wald (K, K) with row = response i, column = predictor j; nobs (lagged
        pairs); n_speakers; order; bias_correction; irf, cum (horizon + 1, K, K) indexed
        [h, response, impulse]; irf_lo, irf_hi, cum_lo, cum_hi (bootstrap bands); rho, rho_within
        (spectral radius of the companion matrix; < 1 for a stable process).
    """
    from statsmodels.stats.multitest import fdrcorrection

    k = len(keys)
    blocks = panel_blocks(df, keys, order)
    users = np.array(sorted(u for u, b in blocks.items() if len(b["full"][0])), dtype=object)
    if len(users) < 5:
        raise ValueError(f"only {len(users)} users with adjacent windows")
    Ahat, Awithin, n = hpj_A(blocks, users, k, order)
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        sel = users[rng.randint(0, len(users), len(users))]     # users drawn with replacement
        boots.append(hpj_A(blocks, sel, k, order)[0])
    boots = np.array(boots)                                  # (n_boot, p, k, k)
    P_ = np.full((k, k), np.nan); W = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            b = Ahat[:, i, j]                                # the p lags of j in the equation of i
            V = np.atleast_2d(np.cov(boots[:, :, i, j].T))   # their bootstrap covariance (p x p)
            w = float(b @ np.linalg.pinv(V) @ b)
            W[i, j] = w
            P_[i, j] = stats.chi2.sf(w, order)
    q = np.full_like(P_, np.nan); m = ~np.isnan(P_)
    q[m] = fdrcorrection(P_[m].ravel())[1]                   # Benjamini-Hochberg over all pairs
    irf = ma_coefs(list(Ahat), horizon)
    bi = np.array([ma_coefs(list(Ab), horizon) for Ab in boots])
    bc = np.cumsum(bi, axis=1)                               # step responses of every resample
    return {"A": Ahat, "A_within": Awithin, "p": P_, "q": q, "wald": W, "nobs": int(n),
            "n_speakers": int(len(users)), "order": order, "bias_correction": "half-panel jackknife",
            "irf": irf, "cum": np.cumsum(irf, axis=0),
            "irf_lo": np.quantile(bi, 0.025, axis=0), "irf_hi": np.quantile(bi, 0.975, axis=0),
            "cum_lo": np.quantile(bc, 0.025, axis=0), "cum_hi": np.quantile(bc, 0.975, axis=0),
            "rho": companion_spectral_radius(list(Ahat)),
            "rho_within": companion_spectral_radius(list(Awithin))}


# --------------------------------------------------------------------------------------------
# Calibration of the tests under a null without lagged effects between factors
# --------------------------------------------------------------------------------------------

def null_calibration(df, keys, hpj_fit, within_fit, n_sim=20, seed=0, n_boot=100):
    """False-positive rates of the two Granger tests in panels simulated without lagged effects
    between factors.

    df         : the observed panel (only each user's observation pattern is used)
    keys       : score columns (K factors)
    hpj_fit    : bias-corrected fit (fit_panel_var_hpj); the diagonal of its lag-1 matrix, clipped
                 to [-0.95, 0.95], gives the own-lag (autoregressive) coefficient of every factor,
                 the same for all users
    within_fit : within fit (fit_panel_var); the covariance of its residuals is the covariance of
                 the simulated innovations
    n_sim      : number of simulated panels
    seed       : seed of the simulation generator (np.random.RandomState)
    n_boot     : bootstrap resamples of users for the bias-corrected test of each simulated panel

    For every user, a VAR(1) path with diagonal coefficient matrix A0 (no cross-lagged effects)
    is simulated from x = A0 x + L e (L the Cholesky factor of the innovation covariance) for 30
    burn-in steps followed by as many steps as the user's span, and the path is observed at the
    user's observed grid positions.  Both tests are applied with lag order 1: the bias-corrected
    bootstrap Wald test and the cluster-robust Wald test of the within estimator.  The bootstrap
    seed of each simulated panel is drawn from the simulation generator after that panel.

    Returns the mean over panels of hpj_fp_rate and within_fp_rate (share of the K (K - 1) pairs
    with P < 0.05) and of hpj_n_fdr and within_n_fdr (pairs with q < 0.05), and n_sim.
    """
    rng = np.random.RandomState(seed)
    k = len(keys)
    A0 = np.diag(np.clip(np.diag(hpj_fit["A"][0]), -0.95, 0.95))   # own-lag dynamics only
    Sig = np.cov(within_fit["E"].T)                     # innovation covariance (K x K)
    Lc = np.linalg.cholesky(Sig + 1e-9 * np.eye(k))
    pats = []                                           # observed grid positions of every user
    for s, g in df.groupby("speaker_id", sort=True):
        t = (g["window_start"] - g["window_start"].min()).dt.total_seconds().to_numpy() / 86400.0
        pats.append((s, np.sort(np.rint(t / P.WINDOW_DAYS).astype(int))))
    res = {"hpj_fp_rate": [], "hpj_n_fdr": [], "within_fp_rate": [], "within_n_fdr": []}
    for _ in range(n_sim):
        rows = []
        for s, steps in pats:
            T = int(steps.max()) + 1
            x = np.zeros(k); path = []
            for t in range(T + 30):
                x = A0 @ x + Lc @ rng.normal(size=k); path.append(x.copy())
            path = np.array(path[30:])                  # drop the burn-in
            for st in steps:                            # observe the path at the user's positions
                rows.append({"speaker_id": s,
                             "window_start": pd.Timestamp("2000-01-01") + pd.Timedelta(days=14 * int(st)),
                             **{kk: path[st, j] for j, kk in enumerate(keys)}})
        sd = pd.DataFrame(rows)
        h = fit_panel_var_hpj(sd, keys, 1, n_boot=n_boot, seed=int(rng.randint(1e9)))
        off = ~np.eye(k, dtype=bool)                    # the K (K - 1) ordered pairs
        res["hpj_fp_rate"].append(float(np.mean(h["p"][off] < 0.05)))
        res["hpj_n_fdr"].append(int(np.sum(h["q"][off] < 0.05)))
        w = fit_panel_var(sd, keys, 1, demean=True)
        pw, qw, _ = granger_panel(w)
        res["within_fp_rate"].append(float(np.mean(pw[off] < 0.05)))
        res["within_n_fdr"].append(int(np.sum(qw[off] < 0.05)))
    return {k_: float(np.mean(v)) for k_, v in res.items()} | {"n_sim": n_sim}


# --------------------------------------------------------------------------------------------
# Responses and stability
# --------------------------------------------------------------------------------------------

def ma_coefs(A: list[np.ndarray], horizon: int) -> np.ndarray:
    """Moving-average coefficients of a VAR(p) with coefficient matrices A = [A_1, ..., A_p]:
    Phi_0 = I and Phi_h = sum_{l=1..min(h, p)} Phi_{h-l} A_l.  Phi_h[i, j] is the response of
    factor i, h steps (h x 14 days) after a unit impulse in factor j (no orthogonalization).
    Returns an array (horizon + 1, K, K) indexed [h, response, impulse]."""
    k = A[0].shape[0]
    Phi = [np.eye(k)]
    for h in range(1, horizon + 1):
        acc = np.zeros((k, k))
        for j in range(1, min(h, len(A)) + 1):
            acc += Phi[h - j] @ A[j - 1]
        Phi.append(acc)
    return np.array(Phi)                                   # (H+1, k, k): [h, to, from]


def irf_panel(fit, horizon=P.IRF_HORIZON, n_boot=500, seed=0):
    """Impulse and step responses of a within fit (fit_panel_var), with bootstrap bands.

    Responses are the moving-average coefficients of fit["A"] (irf) and their cumulative sums
    (cum, response to a sustained unit change).  With n_boot > 0, users are resampled with
    replacement, the OLS is refitted on their stacked rows of the fitted design (scores stay
    centered as in the full data), and irf_lo, irf_hi, cum_lo, cum_hi are the 2.5% and 97.5%
    quantiles over resamples.  Returns a dict of arrays (horizon + 1, K, K).
    """
    irf = ma_coefs(fit["A"], horizon)
    out = {"irf": irf, "cum": np.cumsum(irf, axis=0)}
    if n_boot:
        rng = np.random.RandomState(seed)
        spk = np.array(sorted(set(fit["clusters"])), dtype=object)
        # pre-build per-user design blocks for fast resampling
        Y, X, cl = fit["Y"], fit["X"], fit["clusters"]
        blocks = {s: (Y[cl == s], X[cl == s]) for s in spk}
        boots_irf, boots_cum = [], []
        k, p, off = len(fit["keys"]), fit["order"], (1 if fit["const"] else 0)
        for _ in range(n_boot):
            sel = spk[rng.randint(0, len(spk), len(spk))]
            Yb = np.vstack([blocks[s][0] for s in sel]); Xb = np.vstack([blocks[s][1] for s in sel])
            Bb, *_ = np.linalg.lstsq(Xb, Yb, rcond=None)
            Ab = [Bb[off + L * k:off + (L + 1) * k, :].T for L in range(p)]
            ib = ma_coefs(Ab, horizon)
            boots_irf.append(ib); boots_cum.append(np.cumsum(ib, axis=0))
        bi, bc = np.array(boots_irf), np.array(boots_cum)
        out.update(irf_lo=np.quantile(bi, 0.025, axis=0), irf_hi=np.quantile(bi, 0.975, axis=0),
                   cum_lo=np.quantile(bc, 0.025, axis=0), cum_hi=np.quantile(bc, 0.975, axis=0))
    return out


def companion_spectral_radius(A: list[np.ndarray]) -> float:
    """Largest absolute eigenvalue of the companion matrix of a VAR(p) with matrices A (each
    K x K).  The process is stable (responses decay) when it is below 1."""
    k, p = A[0].shape[0], len(A)
    C = np.zeros((k * p, k * p))
    C[:k, :] = np.hstack(A)
    if p > 1:
        C[k:, :-k] = np.eye(k * (p - 1))
    return float(np.max(np.abs(np.linalg.eigvals(C))))
