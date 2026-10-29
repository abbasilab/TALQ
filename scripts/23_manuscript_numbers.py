#!/usr/bin/env python3
"""
Step 23 - the numbers quoted in the manuscript, computed from the result files.

Supports the Results and Methods: every number of the text that comes from the analysis is
formatted as it is quoted (rounding, thousands separators, typographic minus, ranges with an en
dash) and written under a short key to one file.  A key is left out when its input does not exist,
so the script can run on a partial set of results.  Keys starting with "_branch_" are booleans that
check a qualitative statement of the text (for example that K = 12 is the most stable dimensionality
between 11 and 14); keys ending in "_WORD" spell a count as a word.

Inputs (results/<run>/ unless stated otherwise; the step that writes each file in parentheses) and
the keys computed from them
  05_parse_report.json (5)            N_EXPORT_ROWS; collation counts N_COLLATED, N_NONA, N_CONF,
                                      N_CONF_USERS, PCT_CONF_DROP
  data/processed/windows.parquet (3)  windows, utterances and words of the 491 users: N_AUTOMOD_WIN,
                                      N_WIN_COHORT, N_UTT_COHORT, PCT_POSTS_COHORT, MED_WORDS_COHORT,
                                      MED_UTT_COHORT, IQR_UTT_COHORT, IQR_WORDS_COHORT,
                                      MED_WIN_PER_USER, N_REQ_COHORT
  data/processed/<run>/requests.parquet (5)
                                      stored responses and parse outcomes: N_RATINGS_COHORT(_M),
                                      N_VALID_RATINGS(_M), N_NO_RESPONSE, PCT_NO_RESPONSE, N_UNPARSED,
                                      PCT_UNPARSED, PCT_PARSE_JSON, PCT_PARSE_FALLBACK
  12_robustness.json (12)             A: ICC1_MEDIAN, ICC1_MIN, ICC1_MAX, ICC_MEAN_MEDIAN (and
                                      12A_replicate_reliability.csv: ICC1_LOWEST); B: RHO_WORDS_CONF,
                                      WORDS_PASS, WORDS_FAIL; C: PHI_MEDIAN, N_PHI_085, N_SPLITS;
                                      C2: PHI_THRESH, THRESH<30|50>_*; D_shuffle: DYN_ADV, DYN_NULL,
                                      P_DYN, PSTAY, PSTAY_NULL, P_PSTAY; D_agreement: ARI_BOOT,
                                      ARI_SOFT, STABILITY_WORD, ARI_OCC_HARD, ARI_BOOT_IQR;
                                      E: ARI_HMM, HMM_STAY; F: HALF_N, HALF_AGREE, HALF_NULL,
                                      HALF_KAPPA, HALF_P, HALF_OCC_KAPPA, HALF_OCC_AGREE
  12c_community.json, 12c_community_congruence.csv (12c)
                                      COMM_*_DEP, COMM_*_ANX, COMM_SI_MED_DEP, COMM_SI_MED_ANX
  07_psychometrics.csv (7)            ALPHA_MIN, ALPHA_MAX, KEFF_RANGE
  06_nmf.npz (6)                      NMF_REPS (fits of the final model)
  06_nmf_sweep<tag>.csv (6), tags "", _fine, _broad, _k18, _k18_fine (the sweeps of Fig. 2b; for each
                                      K the estimate with the most initializations): NMF_INSTAB_12,
                                      NMF_INSTAB_NEAR, NMF_NEAR_KS, NMF_BROAD_KS, NMF_BROAD_REPS,
                                      NMF_BROAD_MAX, NMF_LOWER_KS, NMF_MORE_STABLE_KS
  08_var_population.json, .npy (8)    N_GRANGER, TOP_EDGES, AR_MIN, AR_MAX, VAR_RHO, N_P05_PAIRS,
                                      N_CHANCE_P05(_INT), VAR_N_USERS, VAR_NOBS, EX_CAUSE, EX_EFFECT,
                                      EX_SIZE, IRF_CAUSE, IRF_EFFECT, IRF_2W, IRF_2W_CI,
                                      IRF_WEEKS_BELOW1; null calibration: N_NULL_SIM, HPJ_FP,
                                      HPJ_NFDR, WITHIN_FP, WITHIN_NFDR
  09_umap.json (9)                    TRUST_2D
  10_states.json, 10_dpmm_sweep.csv (10)
                                      N_STATES, DPMM_STEP, DPMM_N_STEPS, DPMM_STEP0_STATES,
                                      N_STATES_MIN_BIC
  11_regimes.json (11)                N_SEQ, N_SEQ_WIN, K_AICC, K_BIC, REG_FRAC_RANGE
  12D_heldout.csv (12)                HO_BEST_MARKOV(_K), HO_BEST_OCC(_K), HO_K1_ADV, HO_OCC_BETTER_FROM
  *_8states files (10b, 11, 12)       eight-state sensitivity analysis: S8_*
  20_course.json, 20_course_quarters.csv, 20_course_sweep.csv (20)
                                      course of overall severity and course types: CRS_VARIO_*, CRS_REL_*,
                                      CRS_HALVES_COURSE, CRS_K_BIC, CRS_K_HELDOUT, CRS_P_*, CRS_DEP_SHARE,
                                      CRS_FALL_RANGE, CRS_PHQ_*, CRS_EU_*, CRS_ALT_*, CRS_FOLD_CORR,
                                      CRS_COURSE_R(_NULL), CRS_SLOPE_DIFF, CRS_KAPPA(_NULL_MAX),
                                      CRS_LOGODDS_R(_NULL_MAX)
  results/validation/14b_<tag>_metrics.csv, 14b_<tag>_summary.json (14b; tag of the Llama-3.1 run)
                                      AUC_TBE, AUC_PBU, AUC_LOST, RHO_SEV, AUC_SEV, LEN_*, AUC_CTRL_RANGE,
                                      CTRL_*, AUC_LEX_RANGE, RHO_LEX_SEV, AUC_SUP_RANGE, RHO_SUP_SEV,
                                      LEX_*, VAL_*, JSON_VALID, REFUSAL_*, CAL_*, RATING_RECOVERED,
                                      LOCAL_RUNTIME, OTHER_MODELS(_N_POSTS)
  results/validation/14c_model_comparison.json (14c)
                                      QW_* and LL_* (second model against Llama-3.1-8B)
  results/validation/14d_<tag>_predictive.csv (14d), 14e_<tag>_single_symptom.csv, 14e_differences.csv (14e)
                                      PRED_* (prediction of the human labels from all scored symptoms: AUROC of
                                      the lasso, its range, lexicon features, neural network, best single
                                      symptom, second model against Llama-3.1-8B)
  results/inventories/18_summary.json (18)
                                      INV_*
Output
  results/<run>/manuscript_numbers.json

Usage (run from scripts/)
  python3 23_manuscript_numbers.py --run llama31_8b
"""
import argparse
import glob
import json

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from slm import params as P
from slm.io import load_json, save_json
from slm.nmf import combined_sweep
from slm.prompts import load_symptoms

ap = argparse.ArgumentParser(description="Numbers quoted in the manuscript, from the result files.")
ap.add_argument("--run", default=P.MAIN_RUN,
                help="annotation run, i.e. the folder under data/processed/ and results/ "
                     "(default llama31_8b)")
args = ap.parse_args()
R = P.results_dir(args.run)                       # results of the run
PD = P.processed_dir(args.run)                    # processed tables of the run
VD = P.RESULTS / "validation"                     # validation results (steps 14b, 14c)
N = {}


# ---- formatting helpers ---------------------------------------------------------------------------

def jl(p):
    """Contents of a JSON file, or {} if it does not exist."""
    return load_json(p) if p.exists() else {}


def f2(x, d=2):
    """Number with d decimals and a typographic minus; magnitudes below half a unit of the last
    decimal keep one significant digit (0.0004 instead of 0.00)."""
    if x != 0 and abs(x) < 0.5 * 10 ** (-d):
        d = int(np.ceil(-np.log10(abs(x))))
    s = f"{x:.{d}f}"
    return "−" + s[1:] if s.startswith("-") else s


NUMWORD = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
           9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def rng_str(a, b, d=2):
    """Range "a–b" (en dash), or "a to b" when the lower end is negative."""
    return f"{f2(a, d)} to {f2(b, d)}" if a < 0 else f"{f2(a, d)}–{f2(b, d)}"


def fphi(x):
    """Tucker congruence: three decimals between 0.995 and 1, so that 0.996 is not printed as 1.00."""
    return f"{x:.3f}" if 0.995 <= x < 1 else f2(x)


SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def pfmt(p):
    """P value: two significant digits, or m × 10⁻ᵏ below 0.001 (Unicode superscripts)."""
    if p >= 0.001:
        return f"{p:.2g}"
    m, e = f"{p:.1e}".split("e")
    return f"{m} × 10" + f"-{int(e[1:])}".translate(SUP)


def and_list(xs):
    """'a, b and c' from a list of strings."""
    return (", ".join(xs[:-1]) + " and " + xs[-1]) if len(xs) > 1 else "".join(xs)


# ---- parsing and collation (step 5) -----------------------------------------------------------------
rep = jl(R / "05_parse_report.json")
if rep:
    N["N_EXPORT_ROWS"] = f"{rep['n_rows']:,}"                  # stored responses, all accounts
    col = rep.get("collation")
    if col:
        N["N_COLLATED"] = f"{col['n_collated']:,}"; N["N_NONA"] = f"{col['n_nona']:,}"
        N["N_CONF"] = f"{col['n_confident']:,}"; N["N_CONF_USERS"] = str(col["n_speakers_confident"])
        # share of the windows with all 52 ratings that the confidence filter removes
        N["PCT_CONF_DROP"] = f2(100 * (1 - col["n_confident"] / col["n_nona"]), 1)

# ---- windows of the 491 users (step 3; the moderation bot excluded) ---------------------------------
win = pd.read_parquet(P.PROCESSED / "windows.parquet",
                      columns=["speaker_id", "n_words", "n_posts", "n_submissions"])
N["N_AUTOMOD_WIN"] = str(int((win["speaker_id"].isin(P.BOT_SPEAKERS)).sum()))
wh = win[~win["speaker_id"].isin(P.BOT_SPEAKERS)]
N["N_WIN_COHORT"] = f"{len(wh):,}"
N["N_UTT_COHORT"] = f"{int(wh['n_posts'].sum()):,}"          # n_posts counts utterances
# share of posts (submissions) among the utterances
N["PCT_POSTS_COHORT"] = f2(100 * wh["n_submissions"].sum() / wh["n_posts"].sum(), 1)
N["MED_WORDS_COHORT"] = f"{int(wh['n_words'].median() + 0.5):,}"               # half up (416.5 -> 417)
N["MED_UTT_COHORT"] = f"{wh['n_posts'].median():,.0f}"
_q = wh["n_posts"].quantile([0.25, 0.75]); N["IQR_UTT_COHORT"] = f"{_q.iloc[0]:,.0f}–{_q.iloc[1]:,.0f}"
_q = wh["n_words"].quantile([0.25, 0.75]); N["IQR_WORDS_COHORT"] = f"{_q.iloc[0]:,.0f}–{_q.iloc[1]:,.0f}"
N["MED_WIN_PER_USER"] = f"{wh.groupby('speaker_id').size().median():,.0f}"
N["N_REQ_COHORT"] = f"{len(wh) * 52 * 5:,}"                 # 52 symptoms x 5 samples per window

# ---- stored responses and parse outcomes of the 491 users (step 5) ---------------------------------
_rq = PD / "requests.parquet"
if _rq.exists():
    _r = pd.read_parquet(_rq, columns=["speaker_id", "window_start", "symptom_shortcode", "replicate",
                                       "parse_status"])
    # without the bot; the 33 requests that were answered twice count once
    _r = _r[~_r["speaker_id"].isin(P.BOT_SPEAKERS)].drop_duplicates(
        ["speaker_id", "window_start", "symptom_shortcode", "replicate"])
    _nr = len(_r)
    N["N_RATINGS_COHORT"] = f"{_nr:,}"; N["N_RATINGS_COHORT_M"] = f"{_nr / 1e6:.2f} million"
    _ps = _r["parse_status"].astype(str).value_counts()
    # missing: responses that could not be parsed ("fail"); "empty" (no response text) does not
    # occur in the stored outputs of the main run
    _nmiss = int(_ps.get("fail", 0)) + int(_ps.get("empty", 0))
    _nv = _nr - _nmiss                                                           # valid ratings
    N["N_VALID_RATINGS"] = f"{_nv:,}"; N["N_VALID_RATINGS_M"] = f"{_nv / 1e6:.2f} million"
    N["N_NO_RESPONSE"] = f"{len(wh) * 52 * 5 - _nr:,}"
    N["PCT_NO_RESPONSE"] = f2(100 * (1 - _nr / (len(wh) * 52 * 5)), 2)
    N["N_UNPARSED"] = f"{int(_ps.get('fail', 0)):,}"
    N["PCT_UNPARSED"] = f2(100 * _ps.get("fail", 0) / _nr, 2)
    N["PCT_PARSE_JSON"] = f2(100 * _ps.get("json", 0) / _nr, 1)
    N["PCT_PARSE_FALLBACK"] = f2(100 * (_ps.get("json_block", 0) + _ps.get("regex", 0)) / _nr, 1)

# ---- replicate reliability, text volume and the confidence filter, split halves (step 12) -----------
rob = jl(R / "12_robustness.json")
if "A" in rob:
    a = rob["A"]
    N["ICC1_MEDIAN"] = f2(a["icc1_median"])
    N["ICC1_MIN"], N["ICC1_MAX"] = f2(a["icc1_range"][0]), f2(a["icc1_range"][1])
    N["ICC_MEAN_MEDIAN"] = f2(a["icc_mean_median"])            # reliability of the mean of five samples
    rel = pd.read_csv(R / "12A_replicate_reliability.csv")
    syms = {s["shortcode"]: s["title"] for s in load_symptoms()}
    _low = [f"“{syms[c]}”" for c in rel.nsmallest(3, "icc1")["symptom"]]
    N["ICC1_LOWEST"] = "the items " + ", ".join(_low[:-1]) + " and " + _low[-1]
if "B" in rob:
    b = rob["B"]
    N["RHO_WORDS_CONF"] = f2(b["rho_words_meanconf"]); N["WORDS_PASS"] = f"{b['words_median_pass']:,.0f}"
    N["WORDS_FAIL"] = f"{int(b['words_median_fail'] + 0.5):,}"            # half up (292.5 -> 293)
if "C" in rob:
    N["PHI_MEDIAN"] = fphi(rob["C"]["phi_median_over_factors"])
    N["N_PHI_085"] = str(rob["C"]["n_factors_phi_ge_0.85_median"])
    N["N_SPLITS"] = str(rob["C"].get("n_splits", 20))

# ---- replication in the two communities (step 12c) ---------------------------------------------------
cm_ = jl(R / "12c_community.json")
if cm_:
    cc_ = pd.read_csv(R / "12c_community_congruence.csv")
    for g, pre in [("r/depression", "DEP"), ("r/Anxiety", "ANX")]:
        if g in cm_:
            N[f"COMM_N_{pre}"] = str(cm_["n_users"][g]); N[f"COMM_NWIN_{pre}"] = f"{cm_['n_windows'][g]:,}"
            N[f"COMM_PHI_{pre}"] = fphi(cm_[g]["phi_median"]); N[f"COMM_PHIMIN_{pre}"] = f2(cm_[g]["phi_min"])
            N[f"COMM_PHIMIN_FAC_{pre}"] = cm_[g]["phi_min_factor"]
            N[f"COMM_RND_{pre}"] = fphi(cm_[g]["random_phi_median_median"])
            N[f"COMM_N085_{pre}"] = str(cm_[g]["n_ge_085"])
            ca_ = cc_[(cc_["group"] == g) & (cc_["kind"] == "community")]
            below = ca_[ca_["phi"] < 0.85].sort_values("phi")
            N[f"COMM_BELOW085_{pre}"] = " and ".join(f"{t} ({p:.2f})"
                                                     for t, p in zip(below["title"], below["phi"]))
            # random subsets of the same size in which at least one factor fell below 0.85
            rnd_ = cc_[(cc_["group"] == g) & (cc_["kind"] != "community")].groupby("kind")["phi"].min()
            N[f"COMM_RND_FAIL_{pre}"] = NUMWORD.get(int((rnd_ < 0.85).sum()), str(int((rnd_ < 0.85).sum())))
            N[f"COMM_RND_N_{pre}"] = NUMWORD.get(len(rnd_), str(len(rnd_)))
    if "si_rating_median" in cm_:
        N["COMM_SI_MED_ANX"] = f2(cm_["si_rating_median"]["r/Anxiety"], 0)
        N["COMM_SI_MED_DEP"] = f2(cm_["si_rating_median"]["r/depression"], 0)
    N["_branch_community"] = True

# ---- confidence thresholds 30 and 50, regimes and dynamics (step 12, parts C2, D, E, F) --------------
if "C2" in rob:
    c2 = pd.DataFrame(rob["C2"])
    N["PHI_THRESH"] = (f"{fphi(c2.loc[c2['threshold'] != 40, 'phi_median'].min())}–"
                       f"{fphi(c2.loc[c2['threshold'] != 40, 'phi_median'].max())}")
    for th in (30, 50):
        r_ = c2[c2["threshold"] == th]
        if len(r_):
            r_ = r_.iloc[0]
            N[f"THRESH{th}_N"] = f"{int(r_['n_windows']):,}"; N[f"THRESH{th}_MIN_PHI"] = f2(r_["phi_min"])
            if "phi_min_factor" in r_:
                N[f"THRESH{th}_MIN_FACTOR"] = r_["phi_min_factor"]
                N[f"THRESH{th}_N_BELOW_085"] = str(int(r_["n_below_085"]))
                N[f"_branch_thresh{th}_failure"] = bool(r_["phi_min"] < 0.85)
if "D_shuffle" in rob:
    d = rob["D_shuffle"]
    N["DYN_ADV"] = f2(d["dyn_advantage_real"], 3); N["DYN_NULL"] = f2(d["dyn_advantage_null_mean"], 3)
    N["P_DYN"] = f"{d['p_dyn']:.2g}"
    N["PSTAY"], N["PSTAY_NULL"] = f2(d["persistence_real"]), f2(d["persistence_null_mean"])
    N["P_PSTAY"] = f"{d['p_persistence']:.2g}"
    N["_branch_dynamics"] = bool(d["dyn_advantage_real"] > d["dyn_advantage_null_p95"]
                                 and d["dyn_advantage_real"] > 0)
if "D_agreement" in rob:
    g = rob["D_agreement"]
    N["ARI_BOOT"] = f2(g["ari_bootstrap_median"]); N["ARI_SOFT"] = f2(g["ari_soft_vs_hard"])
    N["STABILITY_WORD"] = ("highly" if g["ari_bootstrap_median"] > 0.8 else
                           "moderately" if g["ari_bootstrap_median"] > 0.5 else "only weakly")
if "E" in rob:
    N["ARI_HMM"] = f2(rob["E"]["ari_hmm_vs_dpmm"])
if "F" in rob:
    f = rob["F"]
    N["HALF_N"] = str(f["n_users"]); N["HALF_AGREE"] = f2(100 * f["markov"]["agreement"], 0)
    N["HALF_NULL"] = f2(100 * f["markov"]["null_mean"], 0); N["HALF_KAPPA"] = f2(f["markov"]["kappa"])
    N["HALF_P"] = f"{f['markov']['p']:.2g}"; N["HALF_OCC_KAPPA"] = f2(f["occ"]["kappa"])

# ---- psychometric properties (step 7) -----------------------------------------------------------------
if (R / "07_psychometrics.csv").exists():
    ps = pd.read_csv(R / "07_psychometrics.csv")
    N["ALPHA_MIN"], N["ALPHA_MAX"] = f2(ps["alpha"].min()), f2(ps["alpha"].max())
    if "k_eff" in ps:
        N["KEFF_RANGE"] = f"{ps['k_eff'].min():.1f}–{ps['k_eff'].max():.1f}"

# ---- factorization and staNMF stability (step 6) -------------------------------------------------------
if (R / "06_nmf.npz").exists():
    N["NMF_REPS"] = str(len(np.load(R / "06_nmf.npz", allow_pickle=True)["errs"]))   # fits of the final model
# the sweeps shown in Fig. 2b: for each K the estimate with the most initializations
cs = combined_sweep(R, tags=("", "_fine", "_broad", "_k18", "_k18_fine"))
if len(cs) > 1:
    s_ = cs.set_index("K")
    near = [k for k in (11, 13, 14) if k in s_.index and s_.loc[k, "n_reps"] >= 100]
    if 12 in s_.index:
        N["NMF_INSTAB_12"] = f2(s_.loc[12, "instability_W"], 3)
    if near:
        N["NMF_INSTAB_NEAR"] = and_list([f"{s_.loc[k, 'instability_W']:.3f}" for k in near])
        N["NMF_NEAR_KS"] = and_list([str(k) for k in near])
        N["_branch_k12_most_stable_near"] = bool(all(s_.loc[k, "instability_W"] > s_.loc[12, "instability_W"]
                                                     for k in near))
    br = cs[cs["source"] == "broad"]
    if len(br) and 12 in s_.index:
        N["NMF_BROAD_KS"] = f"{int(cs['K'].min())}–{int(cs['K'].max())}"
        N["NMF_BROAD_REPS"] = str(int(br["n_reps"].min()))
        N["NMF_BROAD_MAX"] = f2(cs["instability_W"].max(), 3)
        inst12 = s_.loc[12, "instability_W"]
        lower = cs[(cs["K"] > 12) & (cs["instability_W"] < inst12)]["K"].tolist()
        N["NMF_LOWER_KS"] = ", ".join(str(int(k)) for k in lower)
        # every K of the sweeps more stable than K = 12, written as runs ("3–6, 8, 10 and 18")
        _ks = sorted(int(k) for k in cs.loc[cs["instability_W"] < inst12, "K"])
        _runs, _cur = [], []
        for k in _ks:
            if _cur and k == _cur[-1] + 1:
                _cur.append(k)
            else:
                if _cur:
                    _runs.append(_cur)
                _cur = [k]
        if _cur:
            _runs.append(_cur)
        _txt = [f"{r[0]}–{r[-1]}" if len(r) > 2 else ", ".join(str(x) for x in r) for r in _runs]
        N["NMF_MORE_STABLE_KS"] = and_list(_txt)
        N["_branch_nmf_more_stable_elsewhere"] = bool(_ks)

# ---- panel VAR (step 8); matrices are indexed [response i, predictor j] -------------------------------
vj = jl(R / "08_var_population.json")
if vj and (R / "08_var_population.npy").exists():
    N["N_GRANGER"] = str(vj["corrected"]["n_sig"])            # pairs with FDR q < 0.05 (bias-corrected model)
    sp_ = vj["corrected"]["significant_pairs"]
    _e = [x.replace(" -> ", " → ") for x in sp_[:4]]
    N["TOP_EDGES"] = (" and ".join(_e) if len(_e) <= 2 else ", ".join(_e[:-1]) + " and " + _e[-1]
                      + (" among others" if len(sp_) > 4 else ""))
    va = np.load(R / "08_var_population.npy", allow_pickle=True).item()
    cor = va["corrected"]
    sc = {(i, j) for i, j in zip(*np.where(cor["q"] < P.FDR_Q))}
    titles = list(np.load(R / "06_nmf.npz", allow_pickle=True)["titles"])
    A1 = np.asarray(cor["A"])[0]                                # lag-1 coefficients
    N["AR_MIN"], N["AR_MAX"] = f2(np.diag(A1).min()), f2(np.diag(A1).max())
    N["VAR_RHO"] = f2(cor["rho"])
    _offc = ~np.eye(np.asarray(cor["p"]).shape[0], dtype=bool)  # the 132 ordered pairs
    N["N_P05_PAIRS"] = str(int(np.nansum((np.asarray(cor["p"], float) < 0.05) & _offc)))
    N["N_CHANCE_P05"] = f2(0.05 * _offc.sum(), 1)
    N["N_CHANCE_P05_INT"] = NUMWORD.get(int(round(0.05 * _offc.sum())), str(int(round(0.05 * _offc.sum()))))
    N["VAR_N_USERS"] = str(cor.get("n_speakers", ""))
    N["VAR_NOBS"] = f"{cor['nobs']:,}"
    # response 24 weeks after a sustained change of 25 points
    cum = cor["cum"][-1] * P.STEP_SCALE
    sig = [(i, j) for i, j in sc if i != j]
    if sig:
        i, j = max(sig, key=lambda ij: abs(cum[ij]))
        N["EX_CAUSE"], N["EX_EFFECT"], N["EX_SIZE"] = titles[j], titles[i], f2(abs(cum[i, j]), 1)
        irf = cor["irf"] * P.IRF_IMPULSE_SCALE          # impulse of 50 points; horizons in two-week steps
        i, j = max(sig, key=lambda ij: abs(irf[1][ij]))
        N["IRF_CAUSE"], N["IRF_EFFECT"] = titles[j], titles[i]
        N["IRF_2W"] = f2(irf[1][i, j], 1)
        N["IRF_2W_CI"] = rng_str(cor["irf_lo"][1][i, j] * P.IRF_IMPULSE_SCALE,
                                 cor["irf_hi"][1][i, j] * P.IRF_IMPULSE_SCALE, 1)
        # first horizon from which the response stays below 1 point, in weeks
        below = [h for h in range(1, irf.shape[0])
                 if all(abs(irf[k][i, j]) < 1.0 for k in range(h, irf.shape[0]))]
        N["IRF_WEEKS_BELOW1"] = str(2 * below[0]) if below else ""
nc = vj.get("null_calibration")
if nc:
    N["N_NULL_SIM"] = str(nc["n_sim"]); N["HPJ_FP"] = f2(100 * nc["hpj_fp_rate"], 1)
    N["HPJ_NFDR"] = f2(nc["hpj_n_fdr"], 2); N["WITHIN_FP"] = f2(100 * nc["within_fp_rate"], 1)
    N["WITHIN_NFDR"] = f2(nc["within_n_fdr"], 2)

# ---- UMAP, states and regimes (steps 9, 10, 11) and held-out likelihoods (step 12) ----------------------
uj = jl(R / "09_umap.json")
if uj:
    N["TRUST_2D"] = f2(uj["trustworthiness_2d_k20"])
sj = jl(R / "10_states.json")
if sj:
    N["N_STATES"] = str(sj["n_states"])
    N["DPMM_STEP"] = str(sj.get("selected_step", P.DPMM_STEP_USED))
    if (R / "10_dpmm_sweep.csv").exists():
        _sw = pd.read_csv(R / "10_dpmm_sweep.csv")
        N["DPMM_N_STEPS"] = str(len(_sw))
        N["DPMM_STEP0_STATES"] = str(int(_sw.loc[_sw["step"] == 0, "n_states"].iloc[0]))
        # AIC and AICc both favor the least regularized setting (step 0)
        N["_branch_dpmm_aic_most_complex"] = bool(int(_sw.loc[_sw["aic"].idxmin(), "step"]) == 0
                                                  and int(_sw.loc[_sw["aicc"].idxmin(), "step"]) == 0)
    if sj.get("n_states_min_bic") is not None:
        N["N_STATES_MIN_BIC"] = str(sj["n_states_min_bic"])
rj = jl(R / "11_regimes.json")
if rj:
    N["N_SEQ"] = str(rj["n_sequences"]); N["N_SEQ_WIN"] = f"{rj['n_windows']:,}"
    N["K_AICC"] = str(rj["K_aicc"])
    N["K_BIC"] = str(rj["K_bic"])
    N["REG_FRAC_RANGE"] = f"{100 * min(rj['fractions']):.0f}–{100 * max(rj['fractions']):.0f}%"
if "E" in rob:
    N["HMM_STAY"] = f2(rob["E"]["hmm_mean_self_transition"])
if "D_agreement" in rob:
    N["ARI_OCC_HARD"] = f2(rob["D_agreement"]["ari_occupancy_vs_hard"])
    N["ARI_BOOT_IQR"] = rng_str(*rob["D_agreement"]["ari_bootstrap_iqr"])
if "F" in rob:
    N["HALF_OCC_AGREE"] = f2(100 * rob["F"]["occ"]["agreement"], 0)
# held-out log likelihood per window: Markov versus occupancy-only mixtures over the number of components
if (R / "12D_heldout.csv").exists():
    cv = pd.read_csv(R / "12D_heldout.csv")
    bm, bo = cv.loc[cv["markov"].idxmax()], cv.loc[cv["occupancy"].idxmax()]
    N["HO_BEST_MARKOV_K"], N["HO_BEST_MARKOV"] = str(int(bm["K"])), f2(bm["markov"], 3)
    N["HO_BEST_OCC_K"], N["HO_BEST_OCC"] = str(int(bo["K"])), f2(bo["occupancy"], 3)
    N["HO_K1_ADV"] = f2(float(cv.loc[cv["K"] == 1, "markov"].iloc[0]
                              - cv.loc[cv["K"] == 1, "occupancy"].iloc[0]), 2)
    k_cross = cv.loc[cv["occupancy"] >= cv["markov"], "K"]
    N["HO_OCC_BETTER_FROM"] = str(int(k_cross.min())) if len(k_cross) else ""
    N["_branch_occupancy_wins"] = bool(bo["occupancy"] >= bm["markov"])

# ---- eight-state sensitivity analysis: the adjacent, more regularized step (steps 10b, 11, 12) ----------
s8, r8 = jl(R / "10_states_8states.json"), jl(R / "11_regimes_8states.json")
b8 = jl(R / "12_robustness_8states.json")
if s8:
    N["S8_STEP"] = str(s8.get("selected_step", "")); N["S8_N_STATES"] = str(s8["n_states"])
    N["S8_N_STATES_WORD"] = NUMWORD.get(int(s8["n_states"]), str(s8["n_states"]))
if r8:
    N["S8_K_AICC"] = str(r8["K_aicc"]); N["S8_N_SEQ"] = str(r8["n_sequences"])
if "D_shuffle" in b8:
    d = b8["D_shuffle"]
    N["S8_DYN_ADV"] = f2(d["dyn_advantage_real"], 3); N["S8_DYN_NULL"] = f2(d["dyn_advantage_null_mean"], 3)
    N["S8_P_DYN"] = f"{d['p_dyn']:.2g}"
    N["S8_PSTAY"], N["S8_PSTAY_NULL"] = f2(d["persistence_real"]), f2(d["persistence_null_mean"])
if "D_agreement" in b8:
    N["S8_ARI_BOOT"] = f2(b8["D_agreement"]["ari_bootstrap_median"])
if "F" in b8:
    N["S8_HALF_KAPPA"] = f2(b8["F"]["markov"]["kappa"]); N["S8_HALF_OCC_KAPPA"] = f2(b8["F"]["occ"]["kappa"])
if "E" in b8:
    N["S8_ARI_HMM"] = f2(b8["E"]["ari_hmm_vs_dpmm"])
if (R / "12D_heldout_8states.csv").exists():
    cv8 = pd.read_csv(R / "12D_heldout_8states.csv")
    N["S8_HO_BEST_MARKOV"] = f2(cv8["markov"].max(), 3); N["S8_HO_BEST_OCC"] = f2(cv8["occupancy"].max(), 3)

# ---- course of overall severity over the year and course types (step 20) -------------------------------
cj = jl(R / "20_course.json")
if cj:
    sv, rel = cj["semivariogram_severity"], cj["feature_reliability"]["Severity"]
    N["CRS_VARIO_LAG1"], N["CRS_VARIO_LONG"] = f2(sv["lag1"]), f2(sv["lags_19_22_mean"])   # 2 and 38-44 weeks
    N["CRS_REL_N"] = str(rel["n_people"])
    N["CRS_REL_COURSE"] = f2(rel["course"])
    N["CRS_REL_COURSE_SB"] = f2(2 * rel["course"] / (1 + rel["course"]))      # Spearman-Brown, full series
    N["CRS_REL_COURSE_NULL"] = f"{abs(rel['course_null_mean']):.2f}"
    N["CRS_REL_VARIABILITY"] = f2(rel["variability"])
    N["CRS_REL_INERTIA"], N["CRS_REL_INSTAB"] = f2(rel["halves_inertia"]), f2(rel["halves_instability"])
    N["CRS_HALVES_COURSE"] = f2(rel["halves_course"])
    N["CRS_K_BIC"], N["CRS_K_HELDOUT"] = str(cj["K_bic"]), str(cj["K_heldout"])
    sizes = cj["types"]["sizes"]; n_ = sum(sizes)
    stable, remit = 0, len(sizes) - 1                     # the largest and the smallest type
    N["CRS_P_STABLE"] = f"{100 * sizes[stable] / n_:.0f}"; N["CRS_P_REMIT"] = f"{100 * sizes[remit] / n_:.0f}"
    N["CRS_DEP_SHARE"] = f"{100 * cj['type_community_share'][remit]:.0f}"
    qt = pd.read_csv(R / "20_course_quarters.csv").set_index(["type", "quarter"])
    drop = [qt.loc[(remit, 1), t] - qt.loc[(remit, 4), t] for t in
            ["Self-harm tendency", "Negative self-reflection", "Social strain", "Sleep disturbance", "Overwhelm"]]
    N["CRS_FALL_RANGE"] = f"{min(drop):.0f}–{max(drop):.0f}"
    if "phq9_total" in qt:
        N["CRS_PHQ_REMIT_Q1"] = f"{qt.loc[(remit, 1), 'phq9_total']:.0f}"
        N["CRS_PHQ_REMIT_Q4"] = f"{qt.loc[(remit, 4), 'phq9_total']:.0f}"
        N["CRS_PHQ_STABLE"] = f"{qt.loc[(stable, 1), 'phq9_total']:.0f}"
    N["CRS_EU_Q1"] = f"{100 * qt.loc[(remit, 1), 'euthymic']:.0f}"
    N["CRS_EU_Q4"] = f"{100 * qt.loc[(remit, 4), 'euthymic']:.0f}"
    if "alt_types" in cj:
        al = cj["alt_types"]; k_ = int(np.argmin(al["sizes"]))
        N["CRS_ALT_SMALL"] = f"{100 * al['sizes'][k_] / sum(al['sizes']):.0f}"
        N["CRS_ALT_ANX"] = f"{100 * (1 - al['community_share'][k_]):.0f}"     # share from r/Anxiety
    hv = cj["halves"]
    if hv:
        fc = [c[remit] for c in hv["fold_type_corr"]]
        N["CRS_FOLD_CORR"] = rng_str(min(fc), max(fc))
        N["CRS_COURSE_R"] = f2(hv["course_r"])
        N["CRS_COURSE_R_NULL"] = rng_str(min(hv["course_r_null"]), max(hv["course_r_null"]))
        N["CRS_SLOPE_DIFF"] = f2(hv["slope_diff"], 1)
        N["CRS_KAPPA"], N["CRS_KAPPA_NULL_MAX"] = f2(hv["kappa"]), f2(max(hv["kappa_null"]))
        N["CRS_LOGODDS_R"], N["CRS_LOGODDS_NULL_MAX"] = f2(hv["logodds_r"]), f2(max(hv["logodds_r_null"]))
    sw_ = pd.read_csv(R / "20_course_sweep.csv")
    ob = sw_[sw_["data"] == "observed"].set_index("K")["heldout_ll_per_window"]
    sh = sw_[sw_["data"] == "shuffled"].set_index("K")["heldout_ll_per_window"]
    # the held-out likelihood favors more than one type on the observed windows and one type when shuffled
    N["_branch_heldout_more_types"] = bool((ob.drop(1) > ob[1]).all() and int(sh.idxmax()) == 1)

# ---- validation with the reduced protocol on a single local computer (step 14b) ---------------------
# reference: the Llama-3.1-8B-Instruct run served by llama.cpp (or by Ollama when that is the only one)
ref_tag = next((t for t in ["llama31_8b_q4km_llamacpp", "llama31_8b_q4km_ollama"]
                if (VD / f"14b_{t}_metrics.csv").exists()), None)
if ref_tag:
    v = pd.read_csv(VD / f"14b_{ref_tag}_metrics.csv")
    sm = jl(VD / f"14b_{ref_tag}_summary.json")

    def ci_s(r, col):
        """'estimate, 95% CI lo–hi' of metric col of a metrics row."""
        return f"{r[col]:.2f}, 95% CI {r[f'{col}_ci_lo']:.2f}–{r[f'{col}_ci_hi']:.2f}"
    comp = v[(v["source"] == "LM-QA") & (v["predictor"] == "mapped-symptom composite")]
    for k, (ds, lab, col) in {"AUC_TBE": ("irf", "tbe", "auroc"), "AUC_PBU": ("irf", "pbu", "auroc"),
                              "AUC_LOST": ("lost", "low_self_esteem", "auroc"),
                              "RHO_SEV": ("severity", "severity", "spearman"),
                              "AUC_SEV": ("severity", "severity", "auroc")}.items():
        r = comp[(comp["dataset"] == ds) & (comp["label"] == lab)]
        if len(r):
            N[k] = ci_s(r.iloc[0], col)
    adj = v[(v["source"] == "LM-QA")
            & (v["predictor"] == "mapped-symptom composite, adjusted for post length")]
    if len(adj) and len(comp):
        dd = (adj.set_index(["dataset", "label"])["auroc"]
              - comp.set_index(["dataset", "label"])["auroc"]).dropna()
        if len(dd):
            N["LEN_ADJ_DIFF_RANGE"] = rng_str(dd.min(), dd.max())
    ln = v[(v["source"] == "lexicon") & (v["predictor"] == "Post length (log tokens)")]
    if len(ln):
        N["LEN_AUC_RANGE"] = rng_str(ln["auroc"].min(), ln["auroc"].max())
    ctrl = v[(v["source"] == "LM-QA") & (v["predictor"] == "control-symptom composite")]
    if len(ctrl):
        N["AUC_CTRL_RANGE"] = rng_str(ctrl["auroc"].min(), ctrl["auroc"].max())
        N["CTRL_DIFF_RANGE"] = rng_str(ctrl["vs_composite_diff"].min(), ctrl["vs_composite_diff"].max())
        n_sig_ = int((ctrl["vs_composite_diff_ci_lo"] > 0).sum())   # composite better than the controls
        N["CTRL_N_SIG"] = NUMWORD.get(n_sig_, str(n_sig_))
        N["CTRL_N"] = NUMWORD.get(len(ctrl), str(len(ctrl)))
        _nw = NUMWORD.get(len(ctrl), len(ctrl))
        N["CTRL_SIG_PHRASE"] = (f"all {_nw} labels" if n_sig_ == len(ctrl)
                                else f"{NUMWORD.get(n_sig_, n_sig_)} of {_nw} labels")
    lxb = v[(v["source"] == "lexicon") & v["dataset"].isin(["irf", "lost"])]
    if len(lxb):                                                 # best lexicon feature of each label
        _best = lxb.groupby("label")["auroc"].max()
        N["AUC_LEX_RANGE"] = rng_str(_best.min(), _best.max())
    lxs = v[(v["source"] == "lexicon") & (v["dataset"] == "severity")]
    if len(lxs):
        N["RHO_LEX_SEV"] = f2(lxs["spearman"].max())
    sup = v[(v["source"] == "supervised")]
    if len(sup):
        sb = sup[sup["dataset"].isin(["irf", "lost"])]
        N["AUC_SUP_RANGE"] = rng_str(sb["auroc"].min(), sb["auroc"].max())
        ss = sup[sup["dataset"] == "severity"]
        if len(ss):
            N["RHO_SUP_SEV"] = f2(ss["spearman"].iloc[0])
    cmpd = v[v["source"] == "comparison"]                        # composite minus the best lexicon feature
    if len(cmpd):
        N["LEX_DIFF_RANGE"] = rng_str(cmpd["diff"].min(), cmpd["diff"].max())
        N["LEX_DIFF_PMAX"] = f"{cmpd['p_boot_le0'].max():.2g}"
        nb_, ns_ = int((cmpd["diff"] > 0).sum()), int((cmpd["diff_ci_lo"] > 0).sum())
        N["LEX_N_BETTER"] = NUMWORD.get(nb_, str(nb_)); N["LEX_N_SIG"] = NUMWORD.get(ns_, str(ns_))
        N["LEX_N"] = NUMWORD.get(len(cmpd), str(len(cmpd)))
        N["LEX_SIG_PHRASE"] = ("significant for none" if ns_ == 0 else
                               f"significant for {NUMWORD.get(ns_, ns_)}" if ns_ < len(cmpd) else
                               "significant for all")
        N["_branch_lex_all_better"] = bool(nb_ == len(cmpd))
    n_items = v.groupby("dataset")["n_items"].max()
    dsn = {"irf": "interpersonal risk factors", "lost": "low self-esteem", "severity": "depression severity"}
    N["VAL_N_POSTS"] = (f"{int(n_items.iloc[0])} posts per dataset" if n_items.nunique() == 1 else
                        ", ".join(f"{int(n_items[d])} posts for {dsn.get(d, d)}" for d in n_items.index))
    pl = sm.get("planned") or {}
    if pl:
        N["VAL_PLANNED_POSTS"] = str(pl["posts_per_dataset"])
        N["VAL_PLANNED_CAL"] = str(pl["calibration_windows"])
        N["VAL_PLANNED_REQ"] = f"{pl['requests']:,}"
        N["VAL_DONE_REQ"] = f"{sm['n_requests']:,}"
        N["VAL_PCT_DONE"] = f2(100 * sm["frac_of_planned_requests"], 0)
        N["_branch_val_complete"] = bool(sm["frac_of_planned_requests"] >= 0.999)
        N["VAL_INTERIM_NOTE"] = ("" if N["_branch_val_complete"] else
                                 f" [interim: {sm['n_requests']:,} of {pl['requests']:,} planned requests"
                                 " completed]")

    # wording of the validity statements, from the lower ends of the 95% intervals
    def _lo(ds, lab, col):
        """Lower end of the 95% interval of metric col of the composite for one label."""
        r = comp[(comp["dataset"] == ds) & (comp["label"] == lab)]
        return float(r.iloc[0][f"{col}_ci_lo"]) if len(r) else float("nan")
    disc = [nm for (ds, lab, nm) in [("irf", "tbe", "thwarted belongingness"),
                                     ("irf", "pbu", "perceived burdensomeness"),
                                     ("lost", "low_self_esteem", "low self-esteem")]
            if _lo(ds, lab, "auroc") > 0.5]
    sev_ok = _lo("severity", "severity", "spearman") > 0
    _rs = comp[(comp["dataset"] == "severity") & (comp["label"] == "severity")]["spearman"]
    sev_word = "tracked" if (len(_rs) and float(_rs.iloc[0]) >= 0.4) else "correlated weakly with"
    lst = and_list(disc)
    # shorter form for the abstract: the two interpersonal labels as one group
    _ab = ["interpersonal risk factors" if x in ("thwarted belongingness", "perceived burdensomeness") else x
           for x in disc]
    lst_ab = and_list(list(dict.fromkeys(_ab)))
    _bin = comp[comp["dataset"].isin(["irf", "lost"])]["auroc"]
    _rng = f" (area under the curve {_bin.min():.2f}–{_bin.max():.2f})" if len(_bin) else ""
    if disc:
        N["VAL_ABSTRACT_CLAUSE"] = ("In posts annotated by humans, prespecified symptoms discriminated "
                                    f"{lst_ab}{_rng}"
                                    + (f" and {sev_word} depression severity." if sev_ok
                                       else ", and their association with depression severity was weaker."))
        N["VAL_DISC_CLAUSE"] = (f"In posts annotated by humans, they discriminated {lst}"
                                + (f" and {sev_word} depression severity." if sev_ok
                                   else ", and their association with depression severity was weaker."))
    else:
        N["VAL_ABSTRACT_CLAUSE"] = ("In posts annotated by humans, prespecified symptoms discriminated the"
                                    " labels only weakly.")
        N["VAL_DISC_CLAUSE"] = "In posts annotated by humans, they discriminated the labels only weakly."
    N["_branch_sev_ok"] = bool(sev_ok)
    N["VAL_N_REQ"] = f"{sum(int(n) for d, n in sm.get('n_by_dataset', {}).items() if d != 'calibration'):,}"
    N["JSON_VALID"] = f2(100 * sm.get("frac_strict_json", float("nan")), 1)
    N["REFUSAL_PCT"] = f2(100 * sm.get("frac_refusal", float("nan")), 1)
    N["REFUSAL_ITEMS"] = str(sm.get("n_items_any_refusal", ""))
    cps = sm.get("calibration_parse_status_local", {})          # shares of the local answers
    if cps:
        N["CAL_BARE_JSON_LOCAL"] = f2(100 * cps.get("json", 0), 0)
    ops = sm.get("calibration_cohort_parse_status", {})         # counts of the cohort answers
    if ops:
        N["CAL_BARE_JSON_COHORT"] = f2(100 * ops.get("json", 0) / max(sum(ops.values()), 1), 0)
    N["RATING_RECOVERED"] = f2(100 * sm.get("frac_rating_recovered", float("nan")), 1)
    be = (sm.get("backend") or ["?"])[0]
    N["LOCAL_RUNTIME"] = {"ollama": "Ollama, which runs llama.cpp", "llamacpp": "llama.cpp"}.get(be, be)
    # calibration windows: one local sample against the cohort samples (Extended Data Fig. 3c)
    cal = sm.get("calibration_all", {})
    m_ = cal.get("Local (1 sample) vs mean of 4 cohort samples", {})
    c_ = cal.get("Cohort sample vs mean of the other 4 (ceiling)", {})
    if m_:
        N["CAL_RHO"] = f2(m_["spearman"]); N["CAL_ICC"] = f2(m_["icc_a1"]); N["CAL_BIAS"] = f2(m_["bias"], 1)
        N["CAL_MAE"] = f2(m_["mae"], 1)
        # number of calibration windows: their requests divided by the 8 symptoms scored per window
        N["CAL_N_WINDOWS"] = str(sm.get("n_by_dataset", {}).get("calibration", 0) // 8)
    if c_:
        N["CAL_RHO_CEIL"] = f2(c_["spearman"]); N["CAL_ICC_CEIL"] = f2(c_["icc_a1"])
    N["_branch_validation"] = bool(len(comp))
    others = sorted(t.split("14b_")[1].rsplit("_metrics", 1)[0]
                    for t in glob.glob(str(VD / "14b_*_metrics.csv")))
    others = [t for t in others if t != ref_tag and not t.startswith("llama31_8b_q4km")]
    if others:                                          # further models scored with the same protocol
        _sm = {t: jl(VD / f"14b_{t}_summary.json") for t in others}
        _nm = [(_sm[t].get("model_name") or P.MODEL_NAMES.get(t, t)).split(" (")[0] for t in others]
        N["OTHER_MODELS"] = and_list(_nm)
        _np = [min(v for d_, v in (_sm[t].get("n_items_by_dataset") or {}).items() if d_ != "calibration")
               for t in others if _sm[t].get("n_items_by_dataset")]
        if _np:
            N["OTHER_MODELS_N_POSTS"] = str(min(_np))
        N["_branch_models"] = True

# ---- paired comparison of the two models on the validation posts (step 14c) -----------------------------
mc14 = jl(VD / "14c_model_comparison.json")
if mc14.get("labels"):
    _ref = mc14["models"][mc14["reference"]]
    _oth_tag = sorted(t for t in mc14["models"] if t != mc14["reference"])[0]
    _oth = mc14["models"][_oth_tag]
    _lab = pd.DataFrame(mc14["labels"])
    _lab = _lab[_lab["model"] == _oth_tag]
    N["QW_NAME"] = _oth["name"].split(" (")[0]
    N["QW_PCT_JSON"] = f2(_oth["pct_strict_json"], 0); N["LL_PCT_JSON"] = f2(_ref["pct_strict_json"], 0)
    N["QW_PCT_RECOVERED"] = f2(_oth["pct_rating_recovered"], 1)
    N["QW_N_REFUSALS"] = NUMWORD.get(_oth["refusals"], f"{_oth['refusals']:,}")
    N["LL_PCT_REFUSAL"] = f2(_ref["pct_refusal"], 1)
    N["LL_PCT_REFUSAL_PBU"] = f2(_ref["pct_refusal_pbu_positive"], 1)
    N["QW_RHO_LEN"] = f2(_oth["rho_length_posts"]); N["LL_RHO_LEN"] = f2(_ref["rho_length_posts"])
    N["QW_RHO_LEN_WIN"] = f2(_oth["rho_length_windows"]); N["LL_RHO_LEN_WIN"] = f2(_ref["rho_length_windows"])
    N["QW_CAL_RHO"] = f2(_oth["cal_rho_vs_cohort"])
    N["QW_PCT_ZERO"] = f2(_oth["pct_zero"], 0); N["LL_PCT_ZERO"] = f2(_ref["pct_zero"], 0)
    N["QW_REQ_H"] = f"{_oth['requests_per_hour']:,.0f}"; N["LL_REQ_H"] = f"{_ref['requests_per_hour']:,.0f}"
    N["QW_HOURS"] = f2(_oth["active_hours"], 0); N["LL_HOURS"] = f2(_ref["active_hours"], 0)
    _nm = {("irf", "tbe"): "thwarted belongingness", ("irf", "pbu"): "perceived burdensomeness",
           ("lost", "low_self_esteem"): "low self-esteem", ("severity", "severity"): "severity"}
    _auc = _lab[_lab["metric"].isin(["auc", "sev_auc"])].copy()
    _auc["sig"] = (_auc["diff_lo"] > 0) | (_auc["diff_hi"] < 0)  # 95% interval of the difference excludes 0
    _better = _auc[_auc["sig"] & (_auc["diff"] > 0)]
    _worse = _auc[_auc["sig"] & (_auc["diff"] < 0)]
    _ns = _auc[~_auc["sig"]]
    if len(_better) == 1 and len(_worse) == 0:
        r_ = _better.iloc[0]
        N["QW_BETTER_LABEL"] = _nm[(r_["dataset"], r_["label"])]
        N["QW_BETTER_AUC"] = f2(r_["other"]); N["LL_BETTER_AUC"] = f2(r_["ref"])
        N["QW_BETTER_DIFF"] = f"{r_['diff']:.2f}, 95% CI {r_['diff_lo']:.2f}–{r_['diff_hi']:.2f}"
        N["QW_NS_DIFF_RANGE"] = rng_str(_ns["diff"].min(), _ns["diff"].max())
        N["_branch_qw_one_better"] = True
    N["QW_AUC_RANGE"] = rng_str(_auc["other"].min(), _auc["other"].max())
    N["_branch_qw_compare"] = True

# ---- prediction of the human labels from all scored symptoms (steps 14d, 14e) ---------------------------
# Results paragraph on validation, abstract (range of AUROC), Fig. 1d and Extended Data Fig. 3
_pf = VD / f"14d_{ref_tag}_predictive.csv" if ref_tag else None
if _pf is not None and _pf.exists():
    pr_ = pd.read_csv(_pf).set_index(["label", "model"])
    _labs = ["pbu", "tbe", "low_self_esteem", "severity"]

    def _pci(lab, model="lasso", col="auroc"):
        """'estimate, 95% CI lo–hi' of a predictive metric of step 14d."""
        r = pr_.loc[(lab, model)]
        return f"{r[col]:.2f}, 95% CI {r[col + '_ci_lo']:.2f}–{r[col + '_ci_hi']:.2f}"
    for k, lab in {"PRED_AUC_PBU": "pbu", "PRED_AUC_TBE": "tbe", "PRED_AUC_LOST": "low_self_esteem",
                   "PRED_AUC_SEV": "severity"}.items():
        N[k] = _pci(lab)
    N["PRED_RHO_SEV"] = _pci("severity", col="spearman")
    _a = [pr_.loc[(lb, "lasso"), "auroc"] for lb in _labs]
    N["PRED_AUC_RANGE"] = rng_str(min(_a), max(_a))
    _lx = [pr_.loc[(lb, "lexicon lasso"), "auroc"] for lb in _labs[:3]]
    N["PRED_LEX_RANGE"] = rng_str(min(_lx), max(_lx))
    N["PRED_LEX_SEV"] = f2(pr_.loc[("severity", "lexicon lasso"), "auroc"])
    N["PRED_MLP_MAX_GAP"] = f2(max(abs(pr_.loc[(lb, "MLP"), "auroc"] - pr_.loc[(lb, "lasso"), "auroc"])
                                   for lb in _labs))
    _fs = VD / f"14e_{ref_tag}_single_symptom.csv"
    if _fs.exists():
        N["PRED_SINGLE_SEV"] = f2(pd.read_csv(_fs).set_index("label").loc["severity", "auroc"])
    _fd = VD / "14e_differences.csv"
    if _fd.exists():
        D_ = pd.read_csv(_fd)
        d_ = D_[(D_["comparison"] == "lasso minus best single symptom") & (D_["model_tag"] == ref_tag)
                & (D_["metric"] == "auroc")].set_index("label")
        # combining symptoms improves on the best single symptom only for severity
        N["_branch_pred_single_only_sev"] = bool(d_.loc["severity", "ci_lo"] > 0
                                                 and (d_.drop("severity")["ci_lo"] <= 0).all())
        q_ = D_[D_["comparison"].str.endswith(f"minus {ref_tag} (lasso)") & (D_["metric"] == "auroc")]
        if len(q_):
            q_ = q_.set_index("label")
            r_ = q_.loc["low_self_esteem"]
            N["PRED_QW_LOST_DIFF"] = f"{r_['diff']:.2f}, 95% CI {r_['ci_lo']:.2f}–{r_['ci_hi']:.2f}"
            N["_branch_pred_qw_all_better"] = bool((q_["diff"] > 0).all())
            _qt = q_["model_tag"].iloc[0]
            _qp = VD / f"14d_{_qt}_predictive.csv"
            if _qp.exists():
                N["PRED_QW_AUC_LOST"] = f2(pd.read_csv(_qp).set_index(["label", "model"])
                                           .loc[("low_self_esteem", "lasso"), "auroc"])
                N["PRED_LL_AUC_LOST"] = f2(pr_.loc[("low_self_esteem", "lasso"), "auroc"])

# ---- standard inventories answered by the same model (step 18) ------------------------------------------
inv = jl(P.RESULTS / "inventories" / "18_summary.json")
if inv:
    N["INV_N_WINDOWS"] = f"{inv['n_windows_joined']:,}"; N["INV_N_USERS"] = str(inv["n_users_joined"])
    # windows linked to a single cohort member (the moderation bot excluded)
    if "n_windows_linked_cohort" in inv:
        N["INV_N_LINKED"] = f"{inv['n_windows_linked_cohort']:,}"
        N["INV_N_USERS_LINKED"] = str(inv["n_users_linked_cohort"])
    wt = inv.get("window_text_check", {})
    if wt:
        N["INV_TEXT_CHECK"] = (f"{wt['n_identical_posts_text']} of {wt['n_windows']} windows"
                               + (", all from one user" if wt.get("n_users") == 1 else
                                  f", from {wt['n_users']} users" if wt.get("n_users") else ""))
    t_ = inv["totals"]

    def rci(k):
        """'rho, 95% CI lo–hi' of one comparison of totals."""
        r = t_[k]
        return f"{r['rho']:.2f}, 95% CI {r['ci_lo']:.2f}–{r['ci_hi']:.2f}"
    kb = "BDI-II total vs LM-QA BDI-II analogue (mapped battery items)"
    kp = "PHQ-9 total vs LM-QA PHQ-9 analogue (mapped battery items)"
    kk = "BDI-II total vs PHQ-9 total (same model, same role)"
    N["INV_RHO_BDI"] = rci(kb); N["INV_RHO_PHQ"] = rci(kp); N["INV_RHO_BDI_PHQ"] = f2(t_[kk]["rho"])
    N["INV_N_BDI"] = f"{t_[kb]['n']:,}"; N["INV_NU_BDI"] = str(t_[kb]["n_users"])
    N["INV_N_PHQ"] = f"{t_[kp]['n']:,}"; N["INV_NU_PHQ"] = str(t_[kp]["n_users"])
    N["INV_WRHO_BDI"] = f2(t_[kb]["within_rho"]); N["INV_WRHO_PHQ"] = f2(t_[kp]["within_rho"])
    it = inv["items"]
    N["INV_CONV_BDI"] = f2(it["bdi2"]["median_convergent"])
    N["INV_DISC_BDI"] = f2(it["bdi2"]["median_discriminant"])
    N["INV_CONV_PHQ"] = f2(it["phq9"]["median_convergent"])
    N["INV_DISC_PHQ"] = f2(it["phq9"]["median_discriminant"])
    N["INV_TOP3_BDI"] = f"{it['bdi2']['n_items_mapped_top3']} of {it['bdi2']['n_items']}"
    N["INV_TOP3_PHQ"] = f"{it['phq9']['n_items_mapped_top3']} of {it['phq9']['n_items']}"
    N["_branch_inventories"] = True

# ---- counts spelled as words, and output -------------------------------------------------------------------
for k_ in ["N_GRANGER", "K_AICC", "K_BIC", "S8_K_AICC", "N_STATES", "HO_OCC_BETTER_FROM",
           "IRF_WEEKS_BELOW1", "CRS_K_BIC", "CRS_K_HELDOUT"]:
    if k_ in N and str(N[k_]).isdigit():
        N[k_ + "_WORD"] = NUMWORD.get(int(N[k_]), N[k_])
save_json(N, R / "manuscript_numbers.json")
print(json.dumps(N, indent=1, ensure_ascii=False))
