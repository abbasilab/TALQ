"""
Mixtures of first-order Markov chains over discrete states ("dynamical regimes") and mixtures that
model state occupancy only.

Supports the Methods section "Dynamical regimes", Fig. 4c,d, Fig. 5 and Extended Data Figs. 9 and 10,
through scripts 11_regimes.py and 12_robustness.py (parts D and F).

Sequences (good_seqs).  A user's sequence is the list of states of the user's confident windows in
time order.  Users enter the analysis if they have at least 10 such windows and no gap longer than
57 days between consecutive ones, that is, at most three missing windows in a row (249 users and 4,436
windows in the main analysis).  steps[i] is the number of 14-day windows from the user's first included
window to window i, so a missing window shows as a jump of two (e.g. 0, 1, 3, 4).

Transitions are counted in one of two ways (sequence_counts):
  * gaps bridged (gap_aware=False; hard-assignment EM and the regimes it defines): consecutive windows
    of a sequence form a transition even when missing windows lie between them, so a transition across
    a gap counts as a single step, and only the first window of the sequence is scored by the initial
    distribution;
  * gaps respected (gap_aware=True; soft EM, cross-validation, persistence): only windows exactly 14
    days apart form a transition, and the first window of every contiguous run is scored by the initial
    distribution, so that every window enters the likelihood exactly once and Markov and occupancy
    models are compared on the same windows.
Transition matrices are column-stochastic: A[to, from] = P(state `to` at window t + 1 | state `from`
at window t).

Hard-assignment EM (em_hard; the sweep over 1-20 regimes and the final model).  Each regime r has an
initial distribution p0_r over the S states and a transition matrix A_r.  From random parameters,
every iteration assigns each user to the regime under which the sequence has the highest
classification log likelihood, log p0_r(s_1) + sum_t log A_r(s_t | s_t-1), and re-estimates p0_r as
the occupancy of all windows of the regime's users (occupancy serves as the distribution of initial
states) and A_r from their transition counts.  Each restart runs 20 iterations; of 1,000 random
restarts (10,000 for the final model) the one with the highest classification log likelihood is kept.
The number of regimes is chosen by AICc with k = K [(S - 1) + S (S - 1)] parameters and n = number of
windows (n_params_markov_mixture, info_criteria).

Soft EM (em_soft).  The mixture likelihood with mixing weights w_r, sum_r w_r p(sequence | r):
responsibilities replace hard assignments, all counts are smoothed by adding a Dirichlet pseudo-count
alpha = 1, and transitions respect gaps.  With occupancy_only=True the same routine fits a mixture of
multinomial distributions over states, which models occupancy only and no dynamics.

Cross-validation (heldout_loglik).  Users are split into five folds; the model fitted on four folds
scores the sequences of the users of the fifth by the mixture likelihood, and the summed log
likelihood over the five held-out folds is divided by the number of held-out windows (cross-validated
log likelihood per window, in nats).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from . import params as P


# ---------------------------------------------------------------- sequences ---

def good_seqs(df: pd.DataFrame, label_key="state", gap_max_days=P.SEQ_GAP_MAX_DAYS,
              len_min=P.SEQ_LEN_MIN, keep_steps=False):
    """State sequences of the users included in the regime analysis.

    df           : one row per confident window, with speaker_id, window_start (datetime) and
                   the label column `label_key` (integer state code).
    gap_max_days : longest allowed gap between consecutive windows is gap_max_days + 1 days
                   (56 + 1 = 57 days).  Window starts lie on a grid of exactly 14 days, so gaps are
                   14, 28, 42, 56, 70, ... days and the extra day only guards against rounding.
    len_min      : minimum number of windows (10).
    keep_steps   : also return the window positions.
    Users are visited in sorted order of speaker_id and their windows in time order.
    Returns dict speaker_id -> list of state codes, or, with keep_steps, speaker_id -> (states, steps)
    where steps[i] = round((window_start_i - window_start_0) / 14 days), e.g. [0, 1, 3, 4, ...].
    """
    lim = (gap_max_days + 1) * 86400.0                          # in seconds
    out = {}
    for s, g in df.groupby("speaker_id", sort=True):
        g = g.sort_values("window_start")
        if len(g) < len_min:
            continue
        gaps = g["window_start"].diff().dt.total_seconds().to_numpy()[1:]
        if np.sum(gaps > lim) > 0:
            continue
        labels = g[label_key].astype(int).tolist()
        if keep_steps:
            t = (g["window_start"] - g["window_start"].iloc[0]).dt.total_seconds().to_numpy() / 86400.0
            out[s] = (labels, np.rint(t / P.WINDOW_DAYS).astype(int).tolist())
        else:
            out[s] = labels
    return out


def transition_pairs(seq, steps=None, gap_aware=False):
    """(from, to) pairs of consecutive windows of one sequence.

    With gap_aware and steps given, only pairs of windows exactly one step (14 days) apart; otherwise
    every pair of consecutive windows, across gaps.
    """
    if not gap_aware or steps is None:
        return list(zip(seq[:-1], seq[1:]))
    return [(a, b) for a, b, sa, sb in zip(seq[:-1], seq[1:], steps[:-1], steps[1:]) if sb - sa == 1]


def sequence_counts(seqs, n_states, steps=None, gap_aware=False):
    """Occupancy, transition and initial-window counts of each sequence.

    seqs      : list of sequences (lists of state codes 0..n_states-1).
    n_states  : number of states S.
    steps     : list of window positions per sequence (good_seqs with keep_steps); used only with
                gap_aware.
    gap_aware : False - every consecutive pair is a transition and only the first window of the
                sequence is an initial window; True - only pairs exactly one step apart are transitions,
                and the window after each gap starts a new run and counts as an initial window.
    Returns (occ, trans, first):
        occ   (n_seq, S)    number of windows in each state;
        trans (n_seq, S, S) transition counts, trans[i, to, from];
        first (n_seq, S)    number of windows scored by the initial distribution, by state.
    With gap_aware, first.sum() + trans.sum() = occ.sum() for every sequence (each window is scored once).
    """
    occ = np.zeros((len(seqs), n_states))
    trans = np.zeros((len(seqs), n_states, n_states))       # [seq, to, from]
    first = np.zeros((len(seqs), n_states))
    for i, s in enumerate(seqs):
        np.add.at(occ[i], s, 1)
        st = steps[i] if steps is not None else None
        if gap_aware and st is not None:
            first[i, s[0]] += 1
            for a, b, sa, sb in zip(s[:-1], s[1:], st[:-1], st[1:]):
                if sb - sa == 1:
                    trans[i, b, a] += 1
                else:
                    first[i, b] += 1                        # first window of a new contiguous run
        else:
            first[i, s[0]] = 1
            for a, b in transition_pairs(s, st, False):
                trans[i, b, a] += 1
    return occ, trans, first


def normalize_columns(M):
    """Column-stochastic matrix from counts M[to, from]; columns without counts become uniform."""
    M = M.copy()
    z = M.sum(axis=0)
    M[:, z == 0] = 1.0
    return M / M.sum(axis=0, keepdims=True)


# ----------------------------------------------------- hard-assignment EM ---

def em_hard_single(occ, trans, first, n_regimes, n_iter, rng, eps=1e-6):
    """One restart of hard-assignment ("classification") EM for a mixture of Markov chains.

    occ, trans, first : counts from sequence_counts (gaps bridged in the main analysis).
    n_regimes         : number of regimes K.
    n_iter            : number of iterations (fixed, 20; no convergence test).
    rng               : np.random.RandomState of this restart; it draws the initial parameters and the
                        jitter below, so a restart is fully determined by its seed.
    eps               : upper bound of the uniform jitter added to every count.
    Initialization: p0_r and every column of A_r are uniform random draws, normalized.
    E-step: ll[n, r] = first[n] . log p0_r + sum_ij trans[n, i, j] log A_r[i, j]; z[n] = argmax_r.
    M-step, for each regime: p0_r = occupancy of all windows of its users (uniform if the regime is
    empty); A_r = their transition counts normalized by column (columns without counts uniform).  A
    jitter drawn from U(0, eps) is added to every count, so that no probability is exactly zero and
    every sequence keeps a finite likelihood under every regime.
    Returns (total, p0, A, z): the classification log likelihood sum_n ll[n, z[n]] of the final
    parameters at the assignments of the last E-step, p0 (K, S), A (K, S, S) and z (n_seq,).
    """
    n_seq, S = occ.shape
    p0 = rng.uniform(size=(n_regimes, S)); p0 /= p0.sum(axis=1, keepdims=True)
    A = rng.uniform(size=(n_regimes, S, S)); A /= A.sum(axis=1, keepdims=True)   # columns sum to 1
    z = np.zeros(n_seq, dtype=int)
    for _ in range(n_iter):
        ll = (first @ np.log(p0).T) + np.einsum("nij,rij->nr", trans, np.log(A))
        z = np.argmax(ll, axis=1)
        for r in range(n_regimes):
            m = z == r
            c0 = occ[m].sum(axis=0)                             # occupancy as initial distribution
            if c0.sum() == 0:
                c0 = np.ones(S)
            c0 = c0 + rng.uniform(0, eps, size=S)
            p0[r] = c0 / c0.sum()
            cA = trans[m].sum(axis=0)
            cA[:, cA.sum(axis=0) == 0] = 1.0
            cA = cA + rng.uniform(0, eps, size=(S, S))
            A[r] = cA / cA.sum(axis=0, keepdims=True)
    # the returned log likelihood: final parameters, assignments of the last E-step
    ll = (first @ np.log(p0).T) + np.einsum("nij,rij->nr", trans, np.log(A))
    total = float(ll[np.arange(n_seq), z].sum())
    return total, p0, A, z


def em_hard(seqs, n_regimes, n_states=None, n_iter=P.EM_N_ITER, n_init=P.EM_N_INIT, seed=0,
            steps=None, gap_aware=False, n_jobs=1):
    """Mixture of first-order Markov chains by hard-assignment EM, best of n_init random restarts.

    seqs      : list of state sequences (users).
    n_regimes : number of regimes K.
    n_states  : number of states S (default: highest code in seqs + 1).
    n_iter    : iterations per restart (20).
    n_init    : random restarts (1,000 in the sweep, 10,000 for the final model).
    seed      : seeds the generator that draws one seed per restart, so the result does not depend on
                n_jobs or on the order in which restarts finish.
    steps, gap_aware : transition counting (sequence_counts); the regimes use gaps bridged.
    n_jobs    : parallel workers (joblib).
    Returns dict with loglik (classification log likelihood of the best restart; the first one in
    restart order if several are equal), p0 (K, S), A (K, S, S), z (n_seq,) regime of every sequence,
    n_states and n_regimes.
    """
    n_states = n_states or (max(max(s) for s in seqs) + 1)
    occ, trans, first = sequence_counts(seqs, n_states, steps, gap_aware)
    rng0 = np.random.RandomState(seed)
    seeds = rng0.randint(0, 2**31 - 1, size=n_init)

    def run(sd):
        """One restart with its own generator."""
        return em_hard_single(occ, trans, first, n_regimes, n_iter, np.random.RandomState(sd))

    if n_jobs == 1:
        res = [run(sd) for sd in seeds]
    else:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs)(delayed(run)(sd) for sd in seeds)
    best = max(res, key=lambda r: r[0])
    return {"loglik": best[0], "p0": best[1], "A": best[2], "z": best[3],
            "n_states": n_states, "n_regimes": n_regimes}


def n_params_markov_mixture(n_regimes, n_states):
    """Free parameters of K Markov chains on S states, as used for the regime sweep.

    Per regime: S - 1 initial (occupancy) probabilities and S (S - 1) transition probabilities, so
    k = K [(S - 1) + S (S - 1)].  Mixing proportions are not counted because the classification
    likelihood maximized by hard-assignment EM does not contain them.
    """
    return n_regimes * ((n_states - 1) + (n_states * n_states - n_states))


def info_criteria(ll, k, n):
    """(AIC, AICc, BIC) from a log likelihood ll, k parameters and n observations.

    AIC = 2k - 2 ll;  AICc = AIC + (2k^2 + 2k) / (n - k - 1), +inf when n - k - 1 <= 0 (so that such
    models are never selected);  BIC = k ln(n) - 2 ll.  For the regimes, n is the number of windows.
    """
    aic = 2 * k - 2 * ll
    aicc = aic + (2 * k * k + 2 * k) / (n - k - 1) if n - k - 1 > 0 else np.inf
    bic = k * np.log(n) - 2 * ll
    return aic, aicc, bic


# ------------------------------------------------------------- soft EM ---

def em_soft(seqs, n_regimes, n_states=None, n_iter=200, n_init=50, seed=0, alpha=1.0,
            steps=None, gap_aware=True, tol=1e-6, occupancy_only=False):
    """Mixture of first-order Markov chains (or of multinomials) by EM with soft assignments.

    seqs, n_regimes, n_states : as in em_hard.
    n_iter         : maximum iterations per restart.
    n_init         : random restarts; the one with the highest observed-data log likelihood is kept.
    seed           : seed of the generator shared by all restarts (drawn in sequence).
    alpha          : Dirichlet pseudo-count added to every count before normalizing (1: add-one
                     smoothing), so that states or transitions unseen in training keep a nonzero
                     probability for held-out users.
    steps, gap_aware : transition counting (sequence_counts); gaps respected by default.
    tol            : stop when the log likelihood improves by less than tol * |log likelihood|.
    occupancy_only : fit a mixture of multinomial distributions over states (no dynamics).
    Each restart starts from responsibilities R ~ Dirichlet(1, ..., 1) per user and alternates
      M-step: w_r = mean_n R[n, r];
              Markov: pi0_r from the initial-window counts, A_r[:, j] from the transition counts out of
              state j, both weighted by R[:, r] and smoothed by alpha;
              occupancy: theta_r from the occupancy counts weighted by R[:, r], smoothed by alpha;
      E-step: L[n, r] = log w_r + log p(sequence n | regime r); R = softmax_r(L);
              log likelihood = sum_n logsumexp_r L[n, r].
    Note that the initial distribution pi0 is estimated from the initial windows (first window of each
    contiguous run), not from the occupancy as in em_hard.
    Returns dict with loglik, R (n_seq, K) responsibilities, z = argmax_r R, w (K,), and pi0 (K, S) and
    A (K, S, S), or theta (K, S) with occupancy_only; plus n_params = (K - 1) + K (S - 1) for
    occupancy or (K - 1) + K [(S - 1) + S (S - 1)] for Markov mixtures, n_obs (windows), n_states,
    n_regimes and occupancy_only.
    """
    n_states = n_states or (max(max(s) for s in seqs) + 1)
    occ, trans, first = sequence_counts(seqs, n_states, steps, gap_aware)
    rng = np.random.RandomState(seed)
    best = None
    for _ in range(n_init):
        R = rng.dirichlet(np.ones(n_regimes), size=len(seqs))    # random initial responsibilities
        prev = -np.inf
        for it in range(n_iter):
            w = R.mean(axis=0) + 1e-12                          # 1e-12 keeps log(w) finite
            if occupancy_only:
                th = (R.T @ occ) + alpha
                th /= th.sum(axis=1, keepdims=True)
                L = occ @ np.log(th).T
            else:
                pi0 = (R.T @ first) + alpha
                pi0 /= pi0.sum(axis=1, keepdims=True)
                cA = np.einsum("nr,nij->rij", R, trans) + alpha     # [regime, to, from]
                A = cA / cA.sum(axis=1, keepdims=True)              # normalized over `to`
                L = first @ np.log(pi0).T + np.einsum("nij,rij->nr", trans, np.log(A))
            L = L + np.log(w)[None, :]
            ll = float(logsumexp(L, axis=1).sum())
            R = np.exp(L - logsumexp(L, axis=1, keepdims=True))
            if ll - prev < tol * abs(ll):
                break
            prev = ll
        params = {"w": w, "theta": th} if occupancy_only else {"w": w, "pi0": pi0, "A": A}
        if best is None or ll > best["loglik"]:
            best = {"loglik": ll, "R": R, "z": R.argmax(axis=1), **params}
    S, K = n_states, n_regimes
    k = (K - 1) + (K * (S - 1) if occupancy_only else K * ((S - 1) + S * (S - 1)))
    best.update(n_params=k, n_obs=int(occ.sum()), n_states=S, n_regimes=K,
                occupancy_only=occupancy_only)
    return best


def heldout_loglik(seqs, n_regimes, n_folds=5, seed=0, steps=None, occupancy_only=False, **kw):
    """Cross-validated log likelihood per window of a Markov or occupancy mixture (folds of users).

    seqs           : list of state sequences (users).
    n_regimes      : number of mixture components K.
    n_folds        : number of folds (5).
    seed           : seeds both the assignment of users to folds and em_soft, so that Markov and
                     occupancy mixtures, and observed and shuffled sequences, are compared on the same
                     folds.
    steps          : window positions, for gap-aware counting.
    occupancy_only : multinomial mixture instead of a Markov mixture.
    kw             : passed to em_soft (n_init, gap_aware, ...); gap_aware also sets the counting of the
                     held-out sequences (default True).
    For each fold, em_soft is fitted on the other folds, and each held-out sequence is scored by its
    mixture likelihood log sum_r w_r p(sequence | r).  Returns the total held-out log likelihood
    divided by the number of held-out windows (all windows of all users, each scored once).
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(seqs))
    folds = np.array_split(idx, n_folds)
    tot, nwin = 0.0, 0
    for f in folds:
        tr = [i for i in idx if i not in set(f)]                 # training users, in permuted order
        tr_seqs = [seqs[i] for i in tr]; te_seqs = [seqs[i] for i in f]
        tr_steps = [steps[i] for i in tr] if steps is not None else None
        te_steps = [steps[i] for i in f] if steps is not None else None
        S = max(max(s) for s in seqs) + 1
        m = em_soft(tr_seqs, n_regimes, n_states=S, seed=seed, steps=tr_steps,
                    occupancy_only=occupancy_only, **kw)
        occ, trans, first = sequence_counts(te_seqs, S, te_steps, kw.get("gap_aware", True))
        if occupancy_only:
            L = occ @ np.log(m["theta"]).T
        else:
            L = first @ np.log(m["pi0"]).T + np.einsum("nij,rij->nr", trans, np.log(m["A"]))
        L = L + np.log(m["w"])[None, :]
        tot += float(logsumexp(L, axis=1).sum()); nwin += int(occ.sum())
    return tot / nwin


def shuffle_within(seqs, rng):
    """Null model: permute the order of each user's windows (keeps occupancy, destroys temporal order).

    Only the states are permuted; the window positions (steps) stay where they are, so each user's
    pattern of gaps is kept.  rng is an np.random.RandomState; one permutation is drawn per user.
    """
    return [list(rng.permutation(s)) for s in seqs]


def adjusted_rand(a, b):
    """Adjusted Rand index between two labelings of the same items (1 = identical partitions)."""
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))
