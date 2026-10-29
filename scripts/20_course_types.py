#!/usr/bin/env python3
"""
Step 20 - course of overall severity over the year and course types (Fig. 6).

Supports the Results section "People differ in the course of their symptoms over months", the Methods
section "Course of symptoms over months" and Fig. 6.  Fortnightly transitions carried little information
beyond the states each person occupied (steps 10-12).  This step asks whether slower change does: whether
windows months apart differ more than adjacent ones, whether the course of each person over the year can
be measured reliably, and whether people fall into types with different courses.

  1. Semivariogram of every factor and of overall severity (the mean of the 11 factors other than Positive
     wellbeing): half the mean squared difference between two windows of a user as a function of the
     number of windows between them, relative to its mean over --n-shuffle shuffles of each user's windows
     in time (slm.course.semivariogram).
  2. Split-half reliability of level (mean), variability (SD) and course (least-squares slope over the
     year), estimated from the odd and from the even windows of each user with at least 10 windows (two
     interleaved halves covering the same year) and correlated across users, with a null for course from
     shuffles in time; inertia (lag-1 autocorrelation) and instability (log mean squared successive
     difference) compared between the two halves of the year after regression on level, its square and
     variability (slm.course.feature_reliability).
  3. Course types: multivariate growth mixture model (slm.course.CourseMixture) with K = 1..--k-max types
     on the 12 factors; each type has its own linear and quadratic trend for every factor, around a random
     intercept per user and factor with a full covariance, and windows share a full residual covariance.
     Fitted by EM with --n-starts random starts on the observed windows and on windows shuffled within
     each user.  K is chosen by BIC (sample size = users); the held-out log likelihood of users left out
     of fitting (five folds) is reported as well.
  4. Reproducibility of the types: models fitted on 80% of users classify the odd and the even windows of
     each remaining user with at least 10 windows.  Agreement of the two classifications is Cohen's kappa of
     the assigned types and the correlation of the log odds of the smallest type.  Whether the type assigned
     from one half predicts the course of overall severity in the other half is the correlation of the log
     odds with the slope in the other half and the difference in that slope between the smallest type and
     the others; the quarter means in the other half, by assigned type, are drawn in Fig. 6f.  All against
     --n-null repetitions on windows shuffled in time.  The model is also refitted in the users who wrote
     mostly in r/depression (as in step 12c).
  5. Description of each type by quarter of the year (windows 0-6, 7-13, 14-20, 21-26): factors, overall
     severity, the BDI-II and PHQ-9 completed by the same model (linked windows, step 18), sentiment (VADER
     compound, a lexicon measure independent of the model, step 13), the share of windows in the euthymic
     state (PW) and words per window, with 95% intervals of selected means from 200 bootstrap resamples of
     users.

Inputs (<run> is --run; the step that writes each file in parentheses)
  data/processed/<run>/factors.parquet, results/<run>/06_nmf.npz    factor scores and titles (step 6)
  data/processed/<run>/states.parquet, results/<run>/10_states.json state of every window (steps 10, 10c)
  data/processed/windows.parquet                                    window index, words, utterances (step 3)
  data/processed/lexicon_windows.parquet                            VADER compound per window (step 13)
  data/processed/inventories.parquet                                BDI-II and PHQ-9 totals (step 18)
  data/interim/cohort_texts.parquet                                 community of every utterance (step 2)
Outputs
  results/<run>/20_course.json           semivariogram summary, feature reliability, K by BIC and by held-out
                                          likelihood, type sizes, agreement between halves (observed and
                                          null), refit in r/depression, the larger solution (K + 1) in brief
  results/<run>/20_course.npz            semivariogram ratios and pair counts, loadings of the types (B),
                                          posteriors (tau), severity by quarter in the half of the windows not
                                          used to assign the type (ho_q_*), bootstrap intervals of the
                                          quarter means
  results/<run>/20_course_sweep.csv      log likelihood, BIC, entropy, sizes and held-out log likelihood per
                                          window for every K, observed and shuffled
  results/<run>/20_course_labels.csv     type and posterior probabilities of every user (contains speaker_id;
                                          pseudonymized by step 99 before deposit)
  results/<run>/20_course_quarters.csv   means by type and quarter of the year

Usage (run from scripts/; about one hour with two processes)
  python3 20_course_types.py --run llama31_8b [--k-max 6] [--n-starts 10] [--n-null 10] [--n-jobs 2]
"""
import argparse

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import logsumexp
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score

from slm import course as C
from slm import params as P
from slm.io import Timer, load_json, save_json

ap = argparse.ArgumentParser(description="Course of overall severity over the year and course types.")
ap.add_argument("--run", default=P.MAIN_RUN,
                help="annotation run, i.e. the folder under data/processed/ and results/ (default: %(default)s)")
ap.add_argument("--k-max", type=int, default=6, help="largest number of course types fitted (default: 6)")
ap.add_argument("--n-starts", type=int, default=10,
                help="random starts of every fit on all users; the fit with the highest likelihood is kept")
ap.add_argument("--n-starts-cv", type=int, default=4,
                help="random starts of the fits on 80%% of users for the held-out likelihood (default: 4)")
ap.add_argument("--n-null", type=int, default=10,
                help="repetitions of the agreement between halves on windows shuffled in time (default: 10)")
ap.add_argument("--n-shuffle", type=int, default=20,
                help="shuffles in time for the semivariogram and the reliability of course (default: 20)")
ap.add_argument("--n-jobs", type=int, default=2, help="parallel processes (default: 2)")
ap.add_argument("--tag", default="",
                help="suffix of the factor, state and output files of an alternative solution (default: none)")
args = ap.parse_args()
pdir, rdir = P.processed_dir(args.run), P.results_dir(args.run)

# ------------------------------------------------------------------------------------ inputs
fac = pd.read_parquet(pdir / f"factors{args.tag}.parquet")
titles = [str(t) for t in np.load(rdir / f"06_nmf{args.tag}.npz", allow_pickle=True)["titles"]]
keys = [f"nmf_{j}" for j in range(len(titles))]
win = pd.read_parquet(P.PROCESSED / "windows.parquet", columns=["speaker_id", "window_index", "window_start",
                                                                 "n_words", "post_ids"])
d = fac[["speaker_id", "window_start"] + keys].merge(
    win[["speaker_id", "window_start", "window_index", "n_words"]], on=["speaker_id", "window_start"], how="left")
assert d["window_index"].notna().all()
d = d.rename(columns=dict(zip(keys, titles))).sort_values(["speaker_id", "window_index"]).reset_index(drop=True)
d["window_index"] = d["window_index"].astype(int)
others = [t for t in titles if t != "Positive wellbeing"]
d["Severity"] = d[others].mean(axis=1)
st = pd.read_parquet(pdir / f"states{args.tag}.parquet")[["speaker_id", "window_start", "state"]]
d = d.merge(st, on=["speaker_id", "window_start"], how="left")
state_names = load_json(rdir / f"10_states{args.tag}.json").get("state_names", [])
euthymic = state_names.index("PW") if "PW" in state_names else 0
lex = P.PROCESSED / "lexicon_windows.parquet"
if lex.exists():
    d = d.merge(pd.read_parquet(lex, columns=["speaker_id", "window_index", "vader_compound"]),
                on=["speaker_id", "window_index"], how="left")
inv = P.PROCESSED / "inventories.parquet"
if inv.exists():
    d = d.merge(pd.read_parquet(inv, columns=["speaker_id", "window_start", "bdi2_total", "phq9_total"]),
                on=["speaker_id", "window_start"], how="left")
# community of each user: majority of the utterances in their scored windows (as in step 12c)
ex = win[["speaker_id", "post_ids"]].explode("post_ids").dropna(subset=["post_ids"]).rename(columns={"post_ids": "id"})
ut = pd.read_parquet(P.INTERIM / "cohort_texts.parquet", columns=["id", "subreddit"])
share = (ex.merge(ut, on="id", how="left").assign(dep=lambda x: x["subreddit"].str.lower().eq("depression"))
         .groupby("speaker_id")["dep"].mean())
d["depression_community"] = d["speaker_id"].map(share > 0.5).fillna(False).astype(bool)

ids = sorted(d["speaker_id"].unique())
G = {s: g for s, g in d.groupby("speaker_id")}
out = {"n_users": len(ids), "n_windows": int(len(d)), "factors": titles, "severity": "mean of the factors other "
       "than Positive wellbeing"}
arr = {}

# ------------------------------------------------------------------------ 1. slow change
with Timer("semivariograms"):
    cols = titles + ["Severity"]
    lags, obs, null, cnt = C.semivariogram([(G[s]["window_index"].to_numpy(), G[s][cols].to_numpy(float))
                                            for s in ids], n_shuffle=args.n_shuffle, seed=1)
    ratio = obs / null
    arr.update(vario_lags=lags, vario_ratio=ratio, vario_pairs=cnt, vario_cols=np.array(cols))
    b = cols.index("Severity")
    # lag of 1 window, lags of 19-22 windows (38-44 weeks, the longest with at least 700 pairs of windows)
    out["semivariogram_severity"] = {"lag1": float(ratio[0, b]), "lags_19_22_mean": float(ratio[18:22, b].mean()),
                                     "lag_ratio_cross_1": int(lags[np.argmax(ratio[:, b] >= 1)])}
    print(out["semivariogram_severity"])

# ------------------------------------------------------------------ 2. person-level features
with Timer("reliability of person-level features"):
    rel = {}
    for c in cols:
        rel[c] = C.feature_reliability([(G[s]["window_index"].to_numpy(), G[s][c].to_numpy(float)) for s in ids],
                                       n_shuffle=args.n_shuffle, seed=2)
    out["feature_reliability"] = rel
    print({k: round(v, 3) for k, v in rel["Severity"].items() if isinstance(v, float)})


# ------------------------------------------------------------------------- 3. course types
def persons(perm_seed=None, subset=None, part=None):
    """Users as slm.course.Person objects: all users (or `subset`), with their windows permuted in time when
    perm_seed is given (the shuffle null), and only their odd (part=0) or even (part=1) windows when part is
    given."""
    rng = np.random.default_rng(perm_seed) if perm_seed is not None else None
    use = subset if subset is not None else ids
    return [C.make_person(G[s]["window_index"].to_numpy(), G[s][titles].to_numpy(float), part=part,
                          perm=rng.permutation(len(G[s])) if rng is not None else None) for s in use]


P_obs, P_shuf = persons(), persons(perm_seed=11)


def sweep_one(K, which):
    """Best of --n-starts fits with K types on the observed or the shuffled windows of all users."""
    m = C.fit_best(P_obs if which == "observed" else P_shuf, K, n_starts=args.n_starts, seed=K)
    return {"K": K, "data": which, "loglik": m.loglik, "bic": m.bic, "entropy": m.entropy,
            "sizes": np.bincount(m.tau.argmax(1), minlength=K).tolist(), "n_par": m.n_par}, m


with Timer("course types, K = 1..k_max, observed and shuffled"):
    res = Parallel(n_jobs=args.n_jobs)(delayed(sweep_one)(K, w) for w in ("observed", "shuffled")
                                       for K in range(1, args.k_max + 1))
    sweep = pd.DataFrame([r for r, _ in res])
    models = {(r["data"], r["K"]): m for r, m in res}
folds = np.arange(len(ids)) % 5
np.random.default_rng(99).shuffle(folds)


def heldout_one(K, which, f):
    """Log likelihood of the users of fold f under the best fit with K types on the other four folds."""
    Pa = P_obs if which == "observed" else P_shuf
    tr = [p for p, ff in zip(Pa, folds) if ff != f]; te = [p for p, ff in zip(Pa, folds) if ff == f]
    m = C.fit_best(tr, K, n_starts=args.n_starts_cv, seed=50 + f)
    return K, which, float(logsumexp(m.class_loglik(te) + m.logpi[None], axis=1).sum())


with Timer("held-out log likelihood (five folds over users)"):
    ho = Parallel(n_jobs=args.n_jobs)(delayed(heldout_one)(K, w, f) for w in ("observed", "shuffled")
                                      for K in range(1, args.k_max + 1) for f in range(5))
    hod = pd.DataFrame(ho, columns=["K", "data", "ll"]).groupby(["data", "K"])["ll"].sum() / len(d)
    sweep["heldout_ll_per_window"] = [hod[(r.data, r.K)] for r in sweep.itertuples()]
sweep.to_csv(rdir / f"20_course_sweep{args.tag}.csv", index=False)
print(sweep.round(4).to_string())
obs_sw = sweep[sweep["data"] == "observed"].set_index("K")
K = int(obs_sw["bic"].idxmin())
out["K_bic"] = K
out["K_heldout"] = int(obs_sw["heldout_ll_per_window"].idxmax())
main = C.order_by_size(models[("observed", K)])
lab = main.tau.argmax(1)
out["types"] = {"K": K, "sizes": np.bincount(lab, minlength=K).tolist(), "entropy": main.entropy,
                "mean_max_posterior": float(main.tau.max(1).mean())}
arr.update(B=main.B, tau=main.tau, mu=main.mu)
pd.DataFrame({"speaker_id": ids, "type": lab + 1, **{f"p_type{k + 1}": main.tau[:, k] for k in range(K)}}) \
    .to_csv(rdir / f"20_course_labels{args.tag}.csv", index=False)
# the next larger solution, for the text (when it was fitted)
alt = C.order_by_size(models[("observed", K + 1)]) if K < args.k_max else None
alt_lab = alt.tau.argmax(1) if alt is not None else np.zeros(len(ids), int)

# --------------------------------------------------------------- 4. reproducibility of the types
keep = [i for i, s in enumerate(ids) if len(G[s]) >= 10]


def severity_halves(perm_seed=None):
    """Slope and quarter means of overall severity in the odd and in the even windows of every user with at
    least 10 windows (slm.course.half_course; None for the others), with each user's windows permuted in time
    as in persons(perm_seed)."""
    rng = np.random.default_rng(perm_seed) if perm_seed is not None else None
    out = []
    for s in ids:
        y = G[s]["Severity"].to_numpy(float)
        if rng is not None:
            y = y[rng.permutation(len(y))]                     # drawn for every user, as in persons()
        out.append(C.half_course(G[s]["window_index"].to_numpy(), y) if len(y) >= 10 else None)
    return out


def halves_agreement(perm_seed, n_starts):
    """Fit on 80% of users and classify the odd and the even windows of each remaining user (five folds).

    Returns the agreement of the two classifications (Cohen's kappa of the types, correlation of the log odds
    of the smallest type of the main model), the correlation of each fold's types with the main types, and
    whether the type assigned from one half predicts the course of overall severity in the other half: the
    correlation of the log odds with the slope in the other half (mean of both directions) and the difference
    in that slope between users assigned to the smallest type and the others (points per year).  "records"
    holds, for both directions, the user, the assignment and the quarter means in the other half."""
    Pa = persons(perm_seed=perm_seed)
    Po, Pe = persons(perm_seed=perm_seed, part=0), persons(perm_seed=perm_seed, part=1)
    hc = severity_halves(perm_seed)
    lo, le, ro, re_, corr, who = [], [], [], [], [], []
    for f in range(5):
        tr = [Pa[i] for i in range(len(ids)) if folds[i] != f]
        m = C.order_by_size(C.fit_best(tr, K, n_starts=n_starts, seed=10 + f))
        mp = C.match_types(main.B, m.B)                         # type of m -> type of the main model
        inv = {v: k for k, v in mp.items()}
        corr.append([float(np.corrcoef(main.B[a].ravel(), m.B[inv[a]].ravel())[0, 1]) for a in range(K)])
        te = [i for i in keep if folds[i] == f]
        for Pp, L, R in ((Po, lo, ro), (Pe, le, re_)):
            ll = m.class_loglik([Pp[i] for i in te]) + m.logpi[None]
            L += [mp[x] for x in ll.argmax(1)]
            small = inv[K - 1]                                   # smallest type of the main model
            R += list(ll[:, small] - logsumexp(np.delete(ll, small, axis=1), axis=1))
        who += te
    # the type assigned from one half against the course of severity in the other half
    so = np.array([hc[i]["slope_odd"] for i in who]); se = np.array([hc[i]["slope_even"] for i in who])
    ao, ae = np.array(lo) == K - 1, np.array(le) == K - 1
    course_r = (np.corrcoef(ro, se)[0, 1] + np.corrcoef(re_, so)[0, 1]) / 2
    slope_other, assigned = np.r_[se, so], np.r_[ao, ae]
    slope_diff = (slope_other[assigned].mean() - slope_other[~assigned].mean()
                  if assigned.any() and (~assigned).any() else np.nan)
    q_other = np.r_[np.array([hc[i]["q_even"] for i in who]), np.array([hc[i]["q_odd"] for i in who])]
    return {"kappa": float(cohen_kappa_score(lo, le)), "logodds_r": float(np.corrcoef(ro, re_)[0, 1]),
            "n": len(who), "fold_type_corr": corr, "course_r": float(course_r), "slope_diff": float(slope_diff),
            "records": {"who": np.r_[who, who], "assigned_small": assigned, "q_other": q_other}}


with Timer("agreement between interleaved halves, observed and shuffled"):
    if K == 1:                                                    # a single type: nothing to agree on
        out["halves"] = None
    else:
        hv = Parallel(n_jobs=args.n_jobs)(delayed(halves_agreement)(s, n) for s, n in
                                          [(None, args.n_starts)] + [(1000 + k, 3) for k in range(args.n_null)])
        o, nl = hv[0], hv[1:]
        rec = o.pop("records")
        out["halves"] = {**o, "kappa_null": [x["kappa"] for x in nl],
                         "logodds_r_null": [x["logodds_r"] for x in nl],
                         "logodds_r_spearman_brown": 2 * o["logodds_r"] / (1 + o["logodds_r"]),
                         "course_r_null": [x["course_r"] for x in nl], "slope_diff_null": [x["slope_diff"] for x in nl]}
        print({k: v for k, v in out["halves"].items() if k != "fold_type_corr"})
        # Fig. 6f: severity by quarter in the half of the windows not used to assign the type (rows: other types,
        # smallest type), with 95% intervals from 200 bootstrap resamples of users
        groups = [~rec["assigned_small"], rec["assigned_small"]]
        arr["ho_q_mean"] = np.array([np.nanmean(rec["q_other"][g], axis=0) for g in groups])
        arr["ho_q_n"] = np.array([np.isfinite(rec["q_other"][g]).sum(axis=0) for g in groups])
        pos = {u: np.flatnonzero(rec["who"] == u) for u in np.unique(rec["who"])}
        users = np.array(list(pos)); rng_b = np.random.default_rng(6); bq = []
        for _ in range(200):
            ix = np.concatenate([pos[u] for u in rng_b.choice(users, len(users))])
            a_, q_ = rec["assigned_small"][ix], rec["q_other"][ix]
            bq.append([np.nanmean(q_[~a_], axis=0), np.nanmean(q_[a_], axis=0)])
        arr["ho_q_lo"], arr["ho_q_hi"] = np.nanpercentile(np.array(bq), [2.5, 97.5], axis=0)

with Timer("refit in users who wrote mostly in r/depression"):
    dep_ids = [s for s in ids if G[s]["depression_community"].iloc[0]]
    md = C.order_by_size(C.fit_best(persons(subset=dep_ids), K, n_starts=args.n_starts, seed=33))
    mp = C.match_types(main.B, md.B); inv = {v: k for k, v in mp.items()}
    idx = [ids.index(s) for s in dep_ids]
    out["depression_community"] = {
        "n_users": len(dep_ids), "sizes": [int((np.array([mp[x] for x in md.tau.argmax(1)]) == k).sum()) for k in range(K)],
        "type_corr": [float(np.corrcoef(main.B[a].ravel(), md.B[inv[a]].ravel())[0, 1]) for a in range(K)],
        "ari_vs_main": float(adjusted_rand_score(lab[idx], [mp[x] for x in md.tau.argmax(1)]))}
    print(out["depression_community"])

# ------------------------------------------------------------------------------ 5. description
d["type"] = d["speaker_id"].map(dict(zip(ids, lab)))
d["alt_type"] = d["speaker_id"].map(dict(zip(ids, alt_lab)))
d["quarter"] = pd.cut(d["window_index"], [-1, 6, 13, 20, 26], labels=[1, 2, 3, 4]).astype(int)
d["euthymic"] = (d["state"] == euthymic).astype(float)
desc_cols = titles + ["Severity", "euthymic"] + [c for c in ("vader_compound", "phq9_total", "bdi2_total") if c in d]
qt = d.groupby(["type", "quarter"])[desc_cols].mean()
qt["median_words"] = d.groupby(["type", "quarter"])["n_words"].median()
qt["n_windows"] = d.groupby(["type", "quarter"]).size()
for c in ("phq9_total", "bdi2_total"):
    if c in d:
        qt[f"n_{c}"] = d.groupby(["type", "quarter"])[c].count()
qt.reset_index().to_csv(rdir / f"20_course_quarters{args.tag}.csv", index=False)
per = d.groupby("speaker_id").agg(type=("type", "first"), dep=("depression_community", "first"))
out["type_community_share"] = per.groupby("type")["dep"].mean().round(4).tolist()
# 95% intervals of the quarter means of severity and of selected measures (bootstrap over users)
rng = np.random.default_rng(5)
boot = {c: [] for c in ("Severity", "Positive wellbeing", "vader_compound", "phq9_total", "euthymic")
        if c in d}
for _ in range(200):
    pick = rng.choice(ids, len(ids))
    bd = d.set_index("speaker_id").loc[pick].reset_index()
    m_ = bd.groupby(["type", "quarter"])[list(boot)].mean()
    for c in boot:
        boot[c].append(m_[c].reindex(qt.index))
ci = {c: np.nanpercentile(np.array(v), [2.5, 97.5], axis=0) for c, v in boot.items()}
for c, v in ci.items():
    arr[f"q_{c.replace(' ', '_')}_lo"] = v[0]; arr[f"q_{c.replace(' ', '_')}_hi"] = v[1]
arr["q_index"] = np.array([f"{a}-{b}" for a, b in qt.index])
# the larger solution in brief
if alt is not None:
    aq = d.groupby(["alt_type", "quarter"])[titles].mean()
    out["alt_types"] = {"K": K + 1, "sizes": np.bincount(alt_lab, minlength=K + 1).tolist(),
                        "community_share": d.groupby("speaker_id").agg(t=("alt_type", "first"),
                                                                       dep=("depression_community", "first"))
                        .groupby("t")["dep"].mean().round(4).tolist(),
                        "quarter_means": {f"{a}-{q}": {t: float(aq.loc[(a, q), t]) for t in titles}
                                          for a, q in aq.index}}
np.savez(rdir / f"20_course{args.tag}.npz", **arr)
save_json(out, rdir / f"20_course{args.tag}.json")
print("done")
