#!/usr/bin/env python3
"""
Step 12 - reliability of the ratings, replicability of the factors, and tests of the states and regimes.

 A   Replicate reliability (Results, "A mental health scoring system built on an LLM"; Fig. 1c):
     ICC(1) of single ratings across the five samples of each window and symptom, and the reliability of
     the mean of the samples, for every symptom.
 B   Text volume and the confidence filter (same Results section; Extended Data Fig. 1): words per
     window in the windows that pass the confidence filter and in those that do not; Spearman
     correlations of the word count with confidence, ratings and factor scores (pooled and within users).
 C   Split-half replicability of the factors (Fig. 2c, Table 1): the factorization refitted in both halves
     of random splits of users (20 in the main analysis), Tucker congruence with the main loadings after
     optimal matching of factors.
 C2  Confidence thresholds (Extended Data Fig. 5c): the factorization refitted on the windows whose
     confidence exceeds 30, 40 or 50 for every symptom (40 is the threshold of the main analysis).
 D   Markov versus occupancy mixtures (Methods, "Dynamical regimes"; Fig. 4d, Fig. 5a-c, Extended Data
     Fig. 9b):
     (i)   cross-validated log likelihood per window (five folds of users) of Markov mixtures and of
           occupancy (multinomial) mixtures with 1-8 components, for the observed sequences and for
           sequences shuffled in time within users;
     (ii)  shuffle null: each user's windows permuted in time (keeps occupancy), applied to the Markov
           advantage (Markov minus occupancy, cross-validated, at the number of regimes of step 11) and
           to state persistence (mean probability of staying in a state between windows 14 days apart);
     (iii) agreement (adjusted Rand index) of the regime labels of step 11 with those of soft-assignment
           EM and of an occupancy mixture with the same number of components;
     (iv)  agreement of the regime labels with 50 bootstrap refits.
 E   Gaussian hidden Markov model with as many states as the mixture, fitted to the factor scores of the
     contiguous runs of windows (Fig. 5c, Extended Data Fig. 10b): agreement with the mixture states and
     mean probability of staying in a state.
 F   Stability of regime membership within people (Fig. 5c): Markov mixtures (and occupancy mixtures) are
     fitted on four fifths of the users, and the first and second halves of the year of each remaining
     user are classified separately; agreement between halves and Cohen's kappa, against a permutation
     null.
Parts run in the order A, B, C, C2, D, F, E.  Each part uses its own fixed seed, so its results do not
depend on which other parts are run.  Results of parts from previous runs are kept in the JSON output.

Inputs   data/processed/<run>/requests.parquet                    one row per stored response (step 5; A)
         data/processed/windows.parquet                           words per window (step 3; part B)
         data/processed/<run>/collated_nona.parquet               windows with all 52 symptoms rated (B, C2)
         data/processed/<run>/factors<tag>.parquet, states<tag>.parquet   (steps 6 and 10/10b)
         results/<run>/06_nmf<tag>.npz (loadings H, factor titles), 11_regimes<tag>.npy (D, F)
Outputs  results/<run>/12A_replicate_reliability<tag>.csv   symptom, icc1, icc_mean, sd_within, sd_between,
                                                            n_windows
         results/<run>/12C_splithalf_congruence<tag>.csv    title, phi_median, phi_p05, phi_min
         results/<run>/12D_heldout<tag>.csv                 K, markov, occupancy, markov_shuffled,
                                                            occupancy_shuffled
         results/<run>/12F_regime_halves<tag>.csv           speaker_id, markov_h1, markov_h2, occ_h1, occ_h2
         results/<run>/12_robustness<tag>.json              keys A, B, C, C2, D_shuffle, D_agreement, F, E

usage: python3 12_robustness.py --run llama31_8b --n-splits 20 --n-shuffle 200
       python3 12_robustness.py --run llama31_8b --parts D,F,E --n-shuffle 200 --tag _8states
       python3 12_robustness.py --run llama31_8b --parts C,D,E,F --n-splits 20 --n-shuffle 200 --tag _k18
"""
import argparse

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from scipy import stats

from slm import markov as M
from slm import params as P
from slm.io import Timer, load_json, save_json
from slm.nmf import _fit_one, match_factors, tucker_congruence
from slm.parse import symptom_matrix
from slm.prompts import load_symptoms

ap = argparse.ArgumentParser(description="Reliability, replicability and robustness analyses.")
ap.add_argument("--run", required=True,
                help="annotation run: folder under data/processed/ and results/ (main analysis: llama31_8b)")
ap.add_argument("--parts", default="A,B,C,D,E,F",
                help="comma-separated parts to run: A, B, C (split halves, followed by C2), C2, D, E, F "
                     "(default: all)")
ap.add_argument("--n-splits", type=int, default=20,
                help="random splits of users in part C (main analysis: 20)")
ap.add_argument("--n-shuffle", type=int, default=20,
                help="within-user time shuffles of the null in part D(ii) (main analysis: 200)")
ap.add_argument("--tag", default="", help="suffix of the factor and state solution, e.g. _8states or _k18")
args = ap.parse_args()
parts = set(args.parts.split(","))

pdir = P.processed_dir(args.run)
rdir = P.results_dir(args.run)
fac = pd.read_parquet(pdir / f"factors{args.tag}.parquet")        # confident windows: ratings, factor scores
nmf = np.load(rdir / f"06_nmf{args.tag}.npz", allow_pickle=True)
H, titles = nmf["H"], list(nmf["titles"])                         # H: (n_factors, 52) main loadings
keys = [f"nmf_{j}" for j in range(len(titles))]
codes = [s["shortcode"] for s in load_symptoms()]
out = {}

# ------------------------------------------------------------------- A
if "A" in parts:
    with Timer("A. replicate reliability"):
        req = pd.read_parquet(pdir / "requests.parquet",
                              columns=["speaker_id", "window_start", "symptom_shortcode", "replicate",
                                       "rating"])
        # the same set of requests as the window summaries of step 5: accounts run by software removed,
        # one rating per request (the first, when the stored outputs hold a request more than once)
        req = req[~req["speaker_id"].isin(P.BOT_SPEAKERS)]
        req = req.drop_duplicates(["speaker_id", "window_start", "symptom_shortcode", "replicate"])
        req = req.dropna(subset=["rating"])                         # unparseable or out-of-range responses
        g = req.groupby(["speaker_id", "window_start", "symptom_shortcode"])["rating"]
        m, v, n = g.mean(), g.var(ddof=1), g.size()                 # per window and symptom, over samples
        rows = []
        for sc, idx in m.groupby(level="symptom_shortcode").groups.items():
            mm, vv, nn = m.loc[idx], v.loc[idx], n.loc[idx]
            ok = nn >= 2                                            # windows with at least two valid samples
            # one-way random-effects ANOVA with windows as groups; group sizes differ slightly, so the
            # mean number of samples n0 stands in for the group size
            msw = float(vv[ok].mean())                              # within-window mean square
            n0 = float(nn[ok].mean())
            msb = float(n0 * mm[ok].var(ddof=1))                    # between-window mean square
            icc1 = (msb - msw) / (msb + (n0 - 1) * msw)             # ICC(1): reliability of one sample
            icck = (msb - msw) / msb                                # reliability of the mean of n0 samples
            rows.append({"symptom": sc, "icc1": icc1, "icc_mean": icck, "sd_within": np.sqrt(msw),
                         "sd_between": float(mm[ok].std()), "n_windows": int(ok.sum())})
        rel = pd.DataFrame(rows).sort_values("icc1")
        rel.to_csv(rdir / f"12A_replicate_reliability{args.tag}.csv", index=False)
        out["A"] = {"icc1_median": float(rel["icc1"].median()), "icc1_range": [float(rel["icc1"].min()),
                    float(rel["icc1"].max())], "icc_mean_median": float(rel["icc_mean"].median()),
                    # share of window-symptom pairs whose samples are all identical
                    "frac_identical_replicates": float((v[n >= 2] == 0).mean())}
        print(out["A"])

# ------------------------------------------------------------------- B
if "B" in parts:
    with Timer("B. text volume and the confidence filter"):
        win = pd.read_parquet(P.PROCESSED / "windows.parquet",
                              columns=["speaker_id", "window_start", "n_posts", "n_words", "n_chars"])
        win["window_start"] = pd.to_datetime(win["window_start"])
        nona = pd.read_parquet(pdir / "collated_nona.parquet")      # all windows with complete ratings
        d = nona.merge(win, on=["speaker_id", "window_start"], how="left")
        conf_cols = [f"{c}_confidence" for c in codes]
        d["passes_filter"] = (d[conf_cols] > P.CONFIDENCE_THRESH).all(axis=1)   # confidence > 40 for all 52
        d["min_conf"] = d[conf_cols].min(axis=1); d["mean_conf"] = d[conf_cols].mean(axis=1)
        d["mean_rating"] = d[[f"{c}_rating" for c in codes]].mean(axis=1)
        res = {"filter_pass_rate": float(d["passes_filter"].mean()),
               "words_median_pass": float(d.loc[d["passes_filter"], "n_words"].median()),
               "words_median_fail": float(d.loc[~d["passes_filter"], "n_words"].median()),
               "mannwhitney_p_words": float(stats.mannwhitneyu(
                   d.loc[d["passes_filter"], "n_words"], d.loc[~d["passes_filter"], "n_words"]).pvalue),
               "rho_words_meanconf": float(stats.spearmanr(d["n_words"], d["mean_conf"]).statistic),
               "rho_words_meanrating": float(stats.spearmanr(d["n_words"], d["mean_rating"]).statistic)}
        # factor scores of the confident windows
        f2 = fac.merge(win, on=["speaker_id", "window_start"], how="left")
        rho = {t: float(stats.spearmanr(f2["n_words"], f2[k]).statistic) for t, k in zip(titles, keys)}

        def wp(x):
            """Deviation from the user's mean (within-person association)."""
            return x - x.groupby(f2["speaker_id"]).transform("mean")

        rho_w = {t: float(stats.spearmanr(wp(np.log1p(f2["n_words"])), wp(f2[k])).statistic)
                 for t, k in zip(titles, keys)}
        per_sym = {c: float(stats.spearmanr(d["n_words"], d[f"{c}_rating"]).statistic) for c in codes}
        res.update({"rho_words_factor": rho, "rho_words_factor_within_user": rho_w,
                    "rho_words_symptom": per_sym})
        out["B"] = res
        print({k: v for k, v in res.items() if not isinstance(v, dict)})

# ------------------------------------------------------------------- C
if "C" in parts:
    with Timer("C. split-half factor replicability"):
        X = symptom_matrix(fac)                                     # (n_windows, 52) mean ratings
        users = fac["speaker_id"].unique()
        rng = np.random.RandomState(3)                              # draws the splits and the NMF seeds
        phis = []
        for s in range(args.n_splits):
            half = set(rng.choice(users, len(users) // 2, replace=False))
            m = fac["speaker_id"].isin(half).to_numpy()
            for part in (m, ~m):
                best = None
                for r in range(3):                                  # lowest error of three initializations
                    # same K and tolerance as the main model, at most 4,000 coordinate-descent iterations
                    W_, H_, e, _ = _fit_one(X[part], H.shape[0], rng.randint(1e9), P.NMF_TOL, 4000)
                    if best is None or e < best[1]:
                        best = (H_, e)
                perm, sim = match_factors(best[0], H)               # optimal matching to the main factors
                phis.append(tucker_congruence(best[0][perm], H))
        phis = np.array(phis)                                       # (2 * n_splits, n_factors)
        cong = pd.DataFrame({"title": titles, "phi_median": np.median(phis, 0),
                             "phi_p05": np.quantile(phis, 0.05, 0), "phi_min": phis.min(0)})
        cong.to_csv(rdir / f"12C_splithalf_congruence{args.tag}.csv", index=False)
        out["C"] = {"phi_median_over_factors": float(np.median(phis)), "n_splits": int(args.n_splits),
                    "n_factors_phi_ge_0.85_median": int((cong["phi_median"] >= 0.85).sum()),
                    "n_factors_phi_ge_0.95_median": int((cong["phi_median"] >= 0.95).sum())}
        print(cong.round(3).to_string())
if "C" in parts or "C2" in parts:
    with Timer("C2. confidence-threshold sensitivity"):
        nona = pd.read_parquet(pdir / "collated_nona.parquet")
        conf_cols = [f"{c}_confidence" for c in codes]
        sens = []
        rng_c2 = np.random.RandomState(11)
        for th in (30, 40, 50):
            sub = nona[(nona[conf_cols] > th).all(axis=1)]          # confidence above th for every symptom
            Xs = symptom_matrix(sub)
            best = None
            for r in range(5):                                      # lowest error of five initializations
                W_, H_, e, _ = _fit_one(Xs, H.shape[0], int(rng_c2.randint(1e9)), P.NMF_TOL, 4000)
                if best is None or e < best[1]:
                    best = (H_, e)
            perm, _ = match_factors(best[0], H)
            phi = tucker_congruence(best[0][perm], H)
            sens.append({"threshold": th, "n_windows": len(sub), "n_users": sub["speaker_id"].nunique(),
                         "phi_median": float(np.median(phi)), "phi_min": float(phi.min()),
                         "phi_min_factor": titles[int(np.argmin(phi))],
                         "n_below_085": int((phi < 0.85).sum()), "n_below_095": int((phi < 0.95).sum()),
                         "phi": {t: float(p) for t, p in zip(titles, phi)}})
        out["C2"] = sens
        print(pd.DataFrame([{k: v for k, v in s_.items() if k != "phi"} for s_ in sens]))

# ------------------------------------------------------------------- D
if "D" in parts:
    st = pd.read_parquet(pdir / f"states{args.tag}.parquet")
    df = fac.merge(st[["speaker_id", "window_start", "state"]], on=["speaker_id", "window_start"])
    seqd = M.good_seqs(df, "state", keep_steps=True)                # the sequences of step 11
    spk = sorted(seqd); seqs = [seqd[s][0] for s in spk]; steps = [seqd[s][1] for s in spk]
    S = int(df["state"].max()) + 1
    reg = np.load(rdir / f"11_regimes{args.tag}.npy", allow_pickle=True).item()
    K = int(reg["K"])                                               # number of regimes selected in step 11
    with Timer("D(i). held-out likelihood: Markov vs occupancy-only mixtures (observed and time-shuffled)"):
        cv = []
        rng_cv = np.random.RandomState(21)
        shuffled = [M.shuffle_within(seqs, rng_cv) for _ in range(2)]   # two shuffled copies of the data
        for k in range(1, 9):
            row = {"K": k,
                   "markov": M.heldout_loglik(seqs, k, steps=steps, n_init=10, gap_aware=True),
                   "occupancy": M.heldout_loglik(seqs, k, steps=steps, n_init=10, occupancy_only=True)}
            # shuffled sequences keep each user's window positions, hence the same gaps
            row["markov_shuffled"] = float(np.mean([M.heldout_loglik(sh, k, steps=steps, n_init=5,
                                                                     gap_aware=True) for sh in shuffled]))
            row["occupancy_shuffled"] = float(np.mean([M.heldout_loglik(sh, k, steps=steps, n_init=5,
                                                                        occupancy_only=True)
                                                       for sh in shuffled]))
            cv.append(row)
        cv = pd.DataFrame(cv)
        cv.to_csv(rdir / f"12D_heldout{args.tag}.csv", index=False)
        print(cv.round(4).to_string())
    with Timer("D(ii). time-shuffle null"):
        rng = np.random.RandomState(5)

        def dyn_advantage(ss):
            """Held-out log likelihood per window of the Markov mixture minus that of the occupancy mixture
            with K components; same folds, seeds and restarts as in D(i), so the observed value equals the
            difference of the D(i) columns at K."""
            a = M.heldout_loglik(ss, K, steps=steps, n_init=10, gap_aware=True)
            b = M.heldout_loglik(ss, K, steps=steps, n_init=10, occupancy_only=True)
            return a - b

        def persistence(ss):
            """Mean over states of the probability of staying in a state, from transitions pooled over
            users between windows exactly 14 days apart."""
            o, t, _ = M.sequence_counts(ss, S, steps, gap_aware=True)
            A = M.normalize_columns(t.sum(0))
            return float(np.mean(np.diag(A)))

        d_real, p_real = dyn_advantage(seqs), persistence(seqs)
        d_null, p_null = [], []
        for i in range(args.n_shuffle):
            sh = M.shuffle_within(seqs, rng)          # keeps each user's occupancy and gap pattern
            d_null.append(dyn_advantage(sh)); p_null.append(persistence(sh))
        # one-sided permutation P values, (number of null values >= observed + 1) / (n_shuffle + 1)
        out["D_shuffle"] = {
            "dyn_advantage_real": d_real, "dyn_advantage_null_mean": float(np.mean(d_null)),
            "dyn_advantage_null_p95": float(np.quantile(d_null, 0.95)),
            "p_dyn": float((np.sum(np.array(d_null) >= d_real) + 1) / (len(d_null) + 1)),
            "persistence_real": p_real, "persistence_null_mean": float(np.mean(p_null)),
            "persistence_null_p95": float(np.quantile(p_null, 0.95)),
            "p_persistence": float((np.sum(np.array(p_null) >= p_real) + 1) / (len(p_null) + 1)),
            "n_shuffles": args.n_shuffle,
            "dyn_advantage_null": [float(x) for x in d_null], "persistence_null": [float(x) for x in p_null]}
        print({k: v for k, v in out["D_shuffle"].items() if not isinstance(v, list)})
    with Timer("D(iii)-(iv). soft EM agreement and bootstrap stability"):
        z_hard = reg["z"]                                           # regime labels of step 11 (hard EM)
        soft = M.em_soft(seqs, K, n_states=S, steps=steps, gap_aware=True, n_init=50, seed=3)
        occm = M.em_soft(seqs, K, n_states=S, steps=steps, occupancy_only=True, n_init=50, seed=3)
        rng = np.random.RandomState(9)
        aris = []
        for b in range(50):
            # distinct users of a bootstrap resample (each user once), refitted with 500 restarts
            idx = np.unique(rng.randint(0, len(seqs), len(seqs)))
            fb = M.em_hard([seqs[i] for i in idx], K, n_states=S, n_init=500, seed=b)
            aris.append(M.adjusted_rand(z_hard[idx], fb["z"]))
        out["D_agreement"] = {"ari_soft_vs_hard": M.adjusted_rand(z_hard, soft["z"]),
                              "ari_occupancy_vs_hard": M.adjusted_rand(z_hard, occm["z"]),
                              "ari_bootstrap_median": float(np.median(aris)),
                              "ari_bootstrap_iqr": [float(np.quantile(aris, .25)),
                                                    float(np.quantile(aris, .75))]}
        print(out["D_agreement"])

# ------------------------------------------------------------------- F
if "F" in parts:
    with Timer("F. within-person stability of regime membership (first vs second half-year)"):
        from sklearn.metrics import cohen_kappa_score
        st = pd.read_parquet(pdir / f"states{args.tag}.parquet")
        df = fac.merge(st[["speaker_id", "window_start", "state"]], on=["speaker_id", "window_start"])
        seqd = M.good_seqs(df, "state", keep_steps=True)
        spk = sorted(seqd)
        reg = np.load(rdir / f"11_regimes{args.tag}.npy", allow_pickle=True).item()
        K, S = int(reg["K"]), int(df["state"].max()) + 1
        # cross-fitted: Markov mixtures (hard EM, as in step 11) and occupancy mixtures are fitted on the
        # users of four folds, then the two halves of the year of each user of the fifth fold are classified
        rng_f = np.random.RandomState(17)
        order_u = rng_f.permutation(len(spk))
        folds = np.array_split(order_u, 5)
        rows = []
        for f in folds:
            test = set(f.tolist())
            tr = [seqd[spk[i]][0] for i in range(len(spk)) if i not in test]
            fh = M.em_hard(tr, K, n_states=S, n_init=500, seed=int(rng_f.randint(1e6)))
            # mix 0.1 % of a uniform distribution into the fitted probabilities, so that states and
            # transitions absent from the training users do not make a half-year impossible
            lp0 = np.log((1 - 1e-3) * fh["p0"] + 1e-3 / S); lA = np.log((1 - 1e-3) * fh["A"] + 1e-3 / S)
            occm = M.em_soft(tr, K, n_states=S, occupancy_only=True, n_init=20, seed=3)
            ltheta = np.log(occm["theta"])

            def classify(seq, stp):
                """(Markov regime, occupancy component) of one half-sequence: the component under which it
                is most likely (classification likelihood, without mixing weights; gaps bridged as in
                step 11)."""
                o, t, f_ = M.sequence_counts([seq], S, [stp], gap_aware=False)
                return (int(np.argmax(f_ @ lp0.T + np.einsum("nij,rij->nr", t, lA))),
                        int(np.argmax(o @ ltheta.T)))

            for i in f:
                seq, stp = seqd[spk[i]]
                # halves: windows less than 13 steps (26 weeks) after the user's first window, and the rest
                h1 = [(a, b) for a, b in zip(seq, stp) if b < 13]
                h2 = [(a, b) for a, b in zip(seq, stp) if b >= 13]
                if len(h1) >= 3 and len(h2) >= 3:                   # at least three windows in each half
                    m1, o1 = classify([a for a, _ in h1], [b for _, b in h1])
                    m2, o2 = classify([a for a, _ in h2], [b for _, b in h2])
                    rows.append((spk[i], m1, m2, o1, o2))
        hf = pd.DataFrame(rows, columns=["speaker_id", "markov_h1", "markov_h2", "occ_h1", "occ_h2"])
        rng = np.random.RandomState(13)
        res = {"n_users": int(len(hf))}
        for nm in ["markov", "occ"]:
            a, b = hf[f"{nm}_h1"].to_numpy(), hf[f"{nm}_h2"].to_numpy()
            agree = float(np.mean(a == b))
            # null: second halves permuted across users, 1,000 times
            null = [np.mean(a == rng.permutation(b)) for _ in range(1000)]
            res[nm] = {"agreement": agree, "kappa": float(cohen_kappa_score(a, b)),
                       "null_mean": float(np.mean(null)),
                       "p": float((np.sum(np.array(null) >= agree) + 1) / 1001)}
        hf.to_csv(rdir / f"12F_regime_halves{args.tag}.csv", index=False)
        out["F"] = res
        print(res)

# ------------------------------------------------------------------- E
if "E" in parts:
    with Timer("E. Gaussian HMM"):
        from hmmlearn.hmm import GaussianHMM
        from threadpoolctl import threadpool_limits
        st = pd.read_parquet(pdir / f"states{args.tag}.parquet")
        df = fac.merge(st[["speaker_id", "window_start", "state"]], on=["speaker_id", "window_start"])
        # contiguous runs (windows exactly 14 days apart) of at least two windows, from all users
        runs, lab = [], []
        for s, g in df.sort_values(["speaker_id", "window_start"]).groupby("speaker_id"):
            step = np.rint((g["window_start"] - g["window_start"].iloc[0]).dt.total_seconds() / 86400 / 14
                           ).astype(int)
            brk = np.r_[0, np.where(np.diff(step) != 1)[0] + 1, len(g)]   # run boundaries
            for a, b in zip(brk[:-1], brk[1:]):
                if b - a >= 2:
                    runs.append(g[keys].to_numpy(float)[a:b]); lab.append(g["state"].to_numpy()[a:b])
        Xc = np.vstack(runs); lens = [len(r) for r in runs]
        S = int(df["state"].max()) + 1
        # single-threaded linear algebra: EM of a Gaussian HMM with full covariances is sensitive to the
        # order of floating-point sums, so multithreaded BLAS can make reruns differ
        with threadpool_limits(limits=1):
            best = None
            for seed in range(3):                                       # three initializations
                # at most 100 EM iterations, stopping when the log likelihood gains less than 0.1;
                # min_covar = 1 (squared points of the 0-100 scale) floors the covariance diagonals
                h = GaussianHMM(n_components=S, covariance_type="full", n_iter=100, tol=0.1,
                                random_state=seed, min_covar=1.0)
                h.fit(Xc, lens)
                sc = h.score(Xc, lens)
                if best is None or sc > best[1]:
                    best = (h, sc)
            h = best[0]
            dec = h.predict(Xc, lens)                                   # most likely (Viterbi) state sequence
        out["E"] = {"ari_hmm_vs_dpmm": M.adjusted_rand(np.concatenate(lab), dec),
                    "hmm_mean_self_transition": float(np.mean(np.diag(h.transmat_))),
                    "n_runs": len(runs), "n_windows": int(sum(lens))}
        print(out["E"])

# ---- merge with the parts written by previous runs of this script ----
jpath = rdir / f"12_robustness{args.tag}.json"
prev = load_json(jpath) if jpath.exists() else {}
prev.update(out)
save_json(prev, jpath)
