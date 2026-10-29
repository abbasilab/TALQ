"""
Course of the factor scores over the year: slow change within people, the reliability of person-level
features and a mixture of course types.

* ``semivariogram``: half the mean squared difference between two windows of the same person, by the
  number of windows between them, relative to the same quantity after shuffling each person's windows in
  time. A ratio below 1 at short lags and above 1 at long lags means that windows close in time are more
  alike than windows months apart, that is, slow change.
* ``feature_reliability``: correlation across people of level, variability and course (the least-squares
  slope over the year) estimated from the odd and from the even windows of each person, two interleaved
  halves that cover the same year. Fortnightly dynamics (inertia, the lag-1 autocorrelation, and
  instability, the mean squared successive difference) need consecutive windows and are compared between
  the first and second halves of the year instead.
* ``half_course``: slope and quarter means of one score in the odd and in the even windows of a person,
  used to test whether a course type assigned from one half predicts the course in the other half.
* ``CourseMixture``: a multivariate growth mixture model with random intercepts. Every person belongs to
  one of K course types, each with its own trajectory (linear and quadratic terms in time) for every factor;
  people differ in their own level through a random intercept per factor with a full covariance, and
  windows share a full residual covariance across factors. Fitted by expectation-maximization with exact
  updates (an orthogonal transform of each person's windows separates the person mean from the deviations
  around it).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

N_WINDOWS = 27  # windows per year (index 0-26)


def time_basis(window_index):
    """Centered linear and quadratic terms in time, with time running from -0.5 to 0.5 over the year."""
    u = np.asarray(window_index, float) / (N_WINDOWS - 1) - 0.5
    return np.c_[u, u ** 2 - 1 / 12]


# --------------------------------------------------------------------------------- slow change
def semivariogram(groups, n_shuffle=20, seed=0, max_lag=N_WINDOWS - 1):
    """groups: list of (window_index (n,), values (n, p)) per person.

    Returns lags (1..max_lag), the observed semivariance (max_lag, p), its mean under shuffling (max_lag, p),
    and the number of window pairs at each lag."""
    p = groups[0][1].shape[1]

    def one(rng):
        num = np.zeros((max_lag + 1, p)); cnt = np.zeros(max_lag + 1)
        for t, Y in groups:
            if rng is not None:
                Y = Y[rng.permutation(len(Y))]
            iu = np.triu_indices(len(t), 1)
            h = np.abs(t[:, None] - t[None, :])[iu].astype(int)
            sq = 0.5 * (Y[:, None, :] - Y[None, :, :]) ** 2
            np.add.at(num, h, sq[iu])
            np.add.at(cnt, h, 1)
        return num[1:] / np.maximum(cnt[1:, None], 1), cnt[1:]

    obs, cnt = one(None)
    rng = np.random.default_rng(seed)
    null = np.mean([one(rng)[0] for _ in range(n_shuffle)], axis=0)
    return np.arange(1, max_lag + 1), obs, null, cnt


# ------------------------------------------------------------------------ person-level features
def _features(t, y):
    """Level, variability and course (slope per year) of one series; inertia and instability when the
    series has enough exactly consecutive windows."""
    out = {"level": y.mean(), "variability": y.std(ddof=1)}
    tt = np.asarray(t, float) / (N_WINDOWS - 1)
    X = np.c_[np.ones_like(tt), tt - tt.mean()]
    out["course"] = np.linalg.lstsq(X, y, rcond=None)[0][1]
    cons = np.diff(t) == 1
    if cons.sum() >= 3:
        a = y[:-1][cons] - y.mean(); b = y[1:][cons] - y.mean()
        out["inertia"] = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))
        out["instability"] = float(np.mean(np.diff(y)[cons] ** 2))
    else:
        out["inertia"] = out["instability"] = np.nan
    return out


def _residualize(F, col, on):
    """Residual of a feature after regression on other features (level and its square, variability)."""
    X = np.c_[np.ones(len(F))] if not on else np.c_[np.ones(len(F)), *[F[c] for c in on]]
    y = F[col].to_numpy(float)
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def feature_reliability(groups, n_shuffle=20, seed=0, min_windows=10, min_half=5):
    """Split-half reliability of person-level features of one score.

    groups: list of (window_index, values (n,)). Returns a dict of Pearson correlations: level, variability
    and course from interleaved halves (odd and even windows), the same for time-shuffled series (mean
    over shuffles), and inertia and instability (residualized on level, its square and variability) from
    the first and second halves of the year."""
    import pandas as pd

    def interleaved(rng):
        A, B = [], []
        for t, y in groups:
            if len(t) < min_windows:
                continue
            if rng is not None:
                y = y[rng.permutation(len(y))]
            o = np.arange(len(t)) % 2 == 0
            A.append(_features(t[o], y[o])); B.append(_features(t[~o], y[~o]))
        A, B = pd.DataFrame(A), pd.DataFrame(B)
        return {k: float(np.corrcoef(A[k], B[k])[0, 1]) for k in ("level", "variability", "course")}, len(A)

    obs, n = interleaved(None)
    rng = np.random.default_rng(seed)
    nulls = [interleaved(rng)[0] for _ in range(n_shuffle)]
    out = {"n_people": n, **obs}
    for k in ("level", "variability", "course"):
        v = np.array([x[k] for x in nulls])
        out[f"{k}_null_mean"] = float(v.mean()); out[f"{k}_null_p95"] = float(np.quantile(v, 0.95))
    # halves of the year: fortnightly dynamics beyond level and variability
    A, B = [], []
    for t, y in groups:
        a = t <= (N_WINDOWS - 1) // 2
        if a.sum() < min_half or (~a).sum() < min_half:
            continue
        A.append(_features(t[a], y[a])); B.append(_features(t[~a], y[~a]))
    A, B = pd.DataFrame(A), pd.DataFrame(B)
    ok = A[["inertia", "instability"]].notna().all(axis=1) & B[["inertia", "instability"]].notna().all(axis=1)
    A, B = A[ok].reset_index(drop=True), B[ok].reset_index(drop=True)
    for F in (A, B):
        F["level2"] = F["level"] ** 2
        F["log_instability"] = np.log1p(F["instability"])
    out["halves_n_people"] = int(ok.sum())
    for k in ("level", "variability", "course"):
        out[f"halves_{k}"] = float(np.corrcoef(A[k], B[k])[0, 1])
    out["halves_inertia"] = float(np.corrcoef(_residualize(A, "inertia", ["level", "level2", "variability"]),
                                              _residualize(B, "inertia", ["level", "level2", "variability"]))[0, 1])
    out["halves_instability"] = float(np.corrcoef(
        _residualize(A, "log_instability", ["level", "level2", "variability"]),
        _residualize(B, "log_instability", ["level", "level2", "variability"]))[0, 1])
    return out


def half_course(t, y):
    """Course of one score in the odd and in the even windows of a person (two interleaved halves).

    Returns {"slope_odd", "slope_even"}: least-squares slope per year, and {"q_odd", "q_even"}: the mean in
    each quarter of the year (windows 0-6, 7-13, 14-20 and 21-26; NaN for a quarter without windows)."""
    t = np.asarray(t); y = np.asarray(y, float)
    o = np.arange(len(t)) % 2 == 0
    out = {}
    for part, m in (("odd", o), ("even", ~o)):
        out[f"slope_{part}"] = _features(t[m], y[m])["course"]
        q = np.digitize(t[m], [7, 14, 21])
        out[f"q_{part}"] = [float(y[m][q == k].mean()) if (q == k).any() else np.nan for k in range(4)]
    return out


# ------------------------------------------------------------------------------ course mixture
class Person:
    """One person's windows: person means and deviations of the scores and of the time basis."""
    __slots__ = ("n", "ybar", "phibar", "Yc", "Phic")

    def __init__(self, Y, Phi):
        self.n = len(Y)
        self.ybar = Y.mean(0); self.phibar = Phi.mean(0)
        self.Yc = Y - self.ybar; self.Phic = Phi - self.phibar


def make_person(window_index, Y, part=None, perm=None):
    """Person from window indices and scores; part 0 or 1 keeps the odd or the even windows (in time order),
    perm reorders the scores (a shuffle in time)."""
    t = np.asarray(window_index, float); Y = np.asarray(Y, float)
    if perm is not None:
        Y = Y[perm]
    if part is not None:
        o = np.arange(len(t)) % 2 == part
        t, Y = t[o], Y[o]
    return Person(Y, time_basis(t))


def _logdet(A):
    return np.linalg.slogdet(A)[1]


class CourseMixture:
    """Multivariate growth mixture model with random intercepts (see the module docstring)."""

    def __init__(self, K, n_iter=500, tol=1e-6, seed=0):
        self.K, self.n_iter, self.tol, self.seed = K, n_iter, tol, seed

    def class_loglik(self, persons):
        """(n_persons, K) log likelihood of each person's windows under each course type."""
        ll = np.empty((len(persons), self.K))
        Si = np.linalg.inv(self.Sig); ldS = _logdet(self.Sig); P = self.P
        for i, p in enumerate(persons):
            A = self.Sig + p.n * self.Om
            Ai = np.linalg.inv(A); ldA = _logdet(A)
            for k in range(self.K):
                rbar = p.ybar - self.mu - p.phibar @ self.B[k]
                D = p.Yc - p.Phic @ self.B[k]
                ll[i, k] = -0.5 * (ldA + p.n * rbar @ Ai @ rbar + (p.n - 1) * ldS
                                   + np.einsum("tf,fg,tg->", D, Si, D) + p.n * P * np.log(2 * np.pi))
        return ll

    def posterior(self, persons):
        ll = self.class_loglik(persons) + self.logpi[None]
        return np.exp(ll - logsumexp(ll, axis=1, keepdims=True))

    def fit(self, persons):
        rng = np.random.default_rng(self.seed)
        K = self.K; P = persons[0].Yc.shape[1]; J = persons[0].Phic.shape[1]
        self.P, self.J = P, J
        N = len(persons); ntot = sum(p.n for p in persons)
        self.mu = np.mean([p.ybar for p in persons], 0)
        self.Sig = np.cov(np.vstack([p.Yc for p in persons]).T) + 1e-6 * np.eye(P)
        self.Om = np.cov(np.array([p.ybar for p in persons]).T) * 0.8 + 1e-6 * np.eye(P)
        tau = rng.dirichlet(np.ones(K), size=N)          # random soft start
        self.logpi = np.log(tau.mean(0))
        self.B = np.zeros((K, J, P))
        prev = -np.inf
        for it in range(self.n_iter):
            Si = np.linalg.inv(self.Sig)
            Ai_all = [np.linalg.inv(self.Sig + p.n * self.Om) for p in persons]
            # course coefficients of each type: exact generalized least squares given the covariances
            for k in range(K):
                lhs = np.zeros((J * P, J * P)); rhs = np.zeros(J * P)
                for i, p in enumerate(persons):
                    w = tau[i, k]
                    if w < 1e-8:
                        continue
                    Ai = Ai_all[i]
                    lhs += w * (np.kron(p.Phic.T @ p.Phic, Si) + p.n * np.kron(np.outer(p.phibar, p.phibar), Ai))
                    rhs += w * ((Si @ p.Yc.T @ p.Phic).T.ravel() + p.n * np.kron(p.phibar, Ai @ (p.ybar - self.mu)))
                self.B[k] = np.linalg.solve(lhs + 1e-8 * np.eye(J * P), rhs).reshape(J, P)
            # grand mean
            lhs = np.zeros((P, P)); rhs = np.zeros(P)
            for i, p in enumerate(persons):
                for k in range(K):
                    lhs += tau[i, k] * p.n * Ai_all[i]
                    rhs += tau[i, k] * p.n * Ai_all[i] @ (p.ybar - p.phibar @ self.B[k])
            self.mu = np.linalg.solve(lhs, rhs)
            # covariances, with the random intercepts as missing data
            Om_new = np.zeros((P, P)); Sig_new = np.zeros((P, P))
            Omi = np.linalg.inv(self.Om)
            for i, p in enumerate(persons):
                V = np.linalg.inv(Omi + p.n * Si)
                for k in range(K):
                    w = tau[i, k]
                    if w < 1e-10:
                        continue
                    rbar = p.ybar - self.mu - p.phibar @ self.B[k]
                    m = V @ (p.n * Si @ rbar)
                    D = p.Yc - p.Phic @ self.B[k]
                    e = rbar - m
                    Om_new += w * (np.outer(m, m) + V)
                    Sig_new += w * (D.T @ D + p.n * (np.outer(e, e) + V))
            self.Om = Om_new / N + 1e-6 * np.eye(P)
            self.Sig = Sig_new / ntot + 1e-6 * np.eye(P)
            # course types
            ll = self.class_loglik(persons) + self.logpi[None]
            tot = logsumexp(ll, axis=1)
            tau = np.exp(ll - tot[:, None])
            self.logpi = np.log(tau.mean(0).clip(1e-12))
            L = tot.sum()
            if L - prev < self.tol * abs(L):
                break
            prev = L
        self.tau, self.loglik, self.n_iter_run = tau, float(L), it + 1
        self.n_par = P + K * J * P + P * (P + 1) + (K - 1)
        self.bic = -2 * L + self.n_par * np.log(N)
        Pm = tau.clip(1e-300)
        self.entropy = 1 - (-(Pm * np.log(Pm)).sum()) / (N * np.log(K)) if K > 1 else 1.0
        return self


def fit_best(persons, K, n_starts=10, seed=0):
    """Best of n_starts random starts by log likelihood."""
    best = None
    for s in range(1 if K == 1 else n_starts):
        m = CourseMixture(K, seed=seed * 1000 + s).fit(persons)
        if best is None or m.loglik > best.loglik:
            best = m
    return best


def order_by_size(m):
    """Relabel the course types so that the largest comes first."""
    o = np.argsort(-np.bincount(m.tau.argmax(1), minlength=m.K))
    m.B, m.tau, m.logpi = m.B[o], m.tau[:, o], m.logpi[o]
    return m


def match_types(Bref, B):
    """Map the course types of B onto those of Bref by the correlation of their coefficients."""
    K = len(Bref)
    C = np.array([[np.corrcoef(Bref[a].ravel(), B[b].ravel())[0, 1] for b in range(K)] for a in range(K)])
    _, c = linear_sum_assignment(-C)
    return {c[a]: a for a in range(K)}
