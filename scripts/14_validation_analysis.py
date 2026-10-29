#!/usr/bin/env python3
"""
Step 14 - validation against human annotations on every post of each dataset
(Methods, "Validation against human annotations").

The manuscript reports the reduced protocol scored on a single local computer (300 posts per
dataset; step 14b, 14b_validation_local.py).  This step evaluates the comparators on all posts of
each human-annotated dataset and, for every model whose ratings of the full datasets exist, the
TALQ ratings as well.  Predictors:

  * TALQ ratings of any model with outputs in data/llm_outputs/<model>_val_<dataset>/ (written by
    annotation/run_vllm_annotation.py, for example `bash annotation/run_gpu_jobs.sh validation`):
      - the pre-specified composite: mean rating of the symptoms mapped to the label in
        config/validation_mapping.yaml, inverse items entering as 100 - rating (labels marked
        "aggregate: max" use the maximum instead of the mean);
      - every one of the 52 symptoms on its own (per-symptom table), which shows whether the mapped
        symptoms discriminate the label better than the others (specificity);
      - factor scores: each post's 52 ratings are projected on the loadings H of the main analysis
        by nonnegative least squares, and the factor named in the mapping is evaluated (for the
        ordinal severity label, also the sum of all factor scores);
  * lexicon baselines (step 13): VADER negativity and compound score (reversed), the Empath
    categories negative emotion, sadness, shame and suffering, and the rates of first person
    singular pronouns and of absolutist words;
  * supervised reference: logistic regression with L2 regularization on TF-IDF features of word
    unigrams and bigrams, with predictions from stratified fivefold cross-validation.  It is trained
    on the labels, so it is an upper reference rather than a competitor.

Datasets: interpersonal risk factors (thwarted belongingness `tbe`, perceived burdensomeness `pbu`),
LoST (low self-esteem) and depression severity on four levels (minimum, mild, moderate, severe coded
0-3), as in the manuscript, and two secondary datasets that the manuscript does not report:
DepressionEmo (labels from a vote of zero-shot classifiers, treated as weak labels) and ReDSM5
(gated; evaluated only when its labels were prepared in step 4).

Metrics: AUROC for binary labels; for the ordinal severity label, Spearman rho over the four levels
and the AUROC of severe versus minimum.  Intervals: 95% percentile bootstrap over posts.

Inputs:  data/processed/validation_labels.parquet (labels, step 4)
         data/processed/lexicon_validation.parquet (lexicon features, step 13)
         annotation/requests/validation_<dataset>.parquet (post texts, step 4)
         config/validation_mapping.yaml (pre-specified label <-> symptom mapping)
         results/<main-run>/06_nmf.npz (factor loadings H and titles, step 6)
         data/llm_outputs/<model>_val_<dataset>/*.parquet (optional model outputs)
Outputs: results/validation/14_validation_metrics.csv     one row per dataset, label, source, predictor
         results/validation/14_validation_per_symptom.csv every symptom (only with model outputs)
         results/validation/14_validation_sources.json    models whose outputs were found

usage: python3 14_validation_analysis.py --main-run llama31_8b --n-boot 1000
"""
import argparse
import glob

import _setup  # noqa: F401
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.optimize import nnls
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from slm import params as P
from slm.io import save_json
from slm.parse import parse_one
from slm.prompts import load_symptoms

ap = argparse.ArgumentParser(description="Validation against human annotations on the full datasets.")
ap.add_argument("--main-run", default=P.MAIN_RUN,
                help="annotation run whose NMF loadings (results/<run>/06_nmf.npz) give the factor scores")
ap.add_argument("--n-boot", type=int, default=1000,
                help="bootstrap resamples of posts for the 95%% intervals")
args = ap.parse_args()

out_dir = P.RESULTS / "validation"; out_dir.mkdir(parents=True, exist_ok=True)
labels = pd.read_parquet(P.PROCESSED / "validation_labels.parquet")
lex = pd.read_parquet(P.PROCESSED / "lexicon_validation.parquet")
mapping = yaml.safe_load(open(P.CONFIG / "validation_mapping.yaml"))
codes = [s["shortcode"] for s in load_symptoms()]       # the 52 symptoms in battery order
# One generator for every bootstrap of this script: the intervals depend on the order of the calls
# below (lexicon, supervised, then TALQ predictors), which must therefore stay as it is.
rng = np.random.RandomState(0)


def metric(y, s, ordinal=False):
    """Discrimination of the labels `y` by the scores `s` (posts with a missing score are dropped).

    Binary labels: {"auroc", "n"}; the AUROC is NaN unless both classes are present.
    Ordinal labels (severity 0-3): {"spearman", "auroc", "n"} with Spearman rho over the four
    levels and the AUROC of severe (3) versus minimum (0), NaN when at most 10 posts have either level.
    """
    y = np.asarray(y); s = np.asarray(s, float)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if ordinal:
        rho = stats.spearmanr(y, s).statistic
        mm = np.isin(y, [0, 3])
        auc = roc_auc_score(y[mm] == 3, s[mm]) if mm.sum() > 10 else np.nan
        return {"spearman": rho, "auroc": auc, "n": int(len(y))}
    return {"auroc": roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan, "n": int(len(y))}


def boot_all(y, s, ordinal=False):
    """95% percentile bootstrap intervals of every metric of `metric()`, from the same resamples.

    Draws args.n_boot resamples of posts (with replacement) from the shared generator `rng`;
    resamples on which a metric cannot be computed are skipped.  Returns {"<metric>_ci_lo",
    "<metric>_ci_hi"} for "auroc" (and "spearman" for ordinal labels).
    """
    y = np.asarray(y); s = np.asarray(s, float)
    keys = ["spearman", "auroc"] if ordinal else ["auroc"]
    vals = {k: [] for k in keys}
    for _ in range(args.n_boot):
        i = rng.randint(0, len(y), len(y))
        try:
            m = metric(y[i], s[i], ordinal)
            for k in keys:
                vals[k].append(m[k])
        except Exception:  # noqa: BLE001  (e.g. a resample with a single class)
            pass
    out = {}
    for k, v in vals.items():
        v = np.array(v, float); v = v[np.isfinite(v)]
        out[f"{k}_ci_lo"] = float(np.quantile(v, .025)) if len(v) else np.nan
        out[f"{k}_ci_hi"] = float(np.quantile(v, .975)) if len(v) else np.nan
    return out


def boot_ci(y, s, ordinal=False, key="auroc"):
    """Interval of the primary metric `key` as a (lo, hi) tuple, plus the dict of all intervals."""
    b = boot_all(y, s, ordinal)
    return (b[f"{key}_ci_lo"], b[f"{key}_ci_hi"]), b


def ci_cols(y, s, ordinal=False, key="auroc"):
    """Interval columns of one result row: "ci" (primary metric, split into ci_lo/ci_hi before
    saving) and the "<metric>_ci_lo/hi" columns of every metric."""
    (lo, hi), b = boot_ci(y, s, ordinal, key)
    return {"ci": (lo, hi), **b}


rows, per_sym = [], []

# ---- post texts for the supervised reference -------------------------------------------------------
# Each post was rendered like a one-post window, "'<text>'" (step 4); the slice removes the quotes.
texts = {}
for ds in ["irf", "lost", "severity", "depemo", "redsm5"]:
    f = P.ROOT / "annotation" / "requests" / f"validation_{ds}.parquet"
    if not f.exists():
        continue
    v = pd.read_parquet(f)
    texts.update(dict(zip(v["item_id"], v["posts_text"].str.slice(1, -1))))

# ---- comparators: lexicon baselines and the supervised reference ----------------------------------
for ds, labs in [(k, v) for k, v in mapping.items() if not k.startswith("_")]:   # "_..." keys: notes
    L = labels[labels["dataset"] == ds].dropna(axis=1, how="all")    # keep only this dataset's labels
    if L.empty or not (lex["dataset"] == ds).any():
        continue
    lx = lex[lex["dataset"] == ds].set_index("item_id").loc[L["item_id"]]   # same post order as L
    for lab, spec in labs.items():
        if lab not in L:
            continue
        y = L[lab].to_numpy(); ordinal = bool(spec.get("ordinal", False))
        key = "spearman" if ordinal else "auroc"          # primary metric (the "ci" column)
        # lexicon baselines; VADER compound is reversed so that higher means more negative
        cands = {"VADER negativity": lx["vader_neg"], "VADER compound (reversed)": -lx["vader_compound"],
                 "First-person singular rate": lx["style_first_person"],
                 "Absolutist-word rate": lx["style_absolutist"],
                 "Empath negative_emotion": lx["empath_negative_emotion"],
                 "Empath sadness": lx["empath_sadness"],
                 "Empath shame": lx["empath_shame"], "Empath suffering": lx["empath_suffering"]}
        for nm, s in cands.items():
            r = metric(y, s.to_numpy(), ordinal)
            rows.append({"dataset": ds, "label": lab, "source": "lexicon", "predictor": nm, **r,
                         **ci_cols(y, s.to_numpy(), ordinal, key)})
        # supervised reference: TF-IDF of word 1-2 grams (terms in >= 2 posts, sublinear term
        # frequency) and L2 logistic regression (C = 4), out-of-fold predictions from stratified
        # fivefold cross-validation.  Score: probability of the positive class (binary labels) or
        # the expected level sum_k k * p_k (ordinal label).
        X_t = [texts[i] for i in L["item_id"]]
        clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                            LogisticRegression(max_iter=2000, C=4.0))
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        pr = cross_val_predict(clf, X_t, y, cv=cv, method="predict_proba")
        score = pr[:, 1] if pr.shape[1] == 2 else pr @ np.arange(pr.shape[1])
        rows.append({"dataset": ds, "label": lab, "source": "supervised",
                     "predictor": "TF-IDF + logistic (5-fold CV)",
                     **metric(y, score, ordinal), **ci_cols(y, score, ordinal, key)})

# ---- TALQ ratings of the full datasets (if any model's outputs exist) --------------------------
H = None
nmf_path = P.results_dir(args.main_run) / "06_nmf.npz"
if nmf_path.exists():
    nmf = np.load(nmf_path, allow_pickle=True)
    H, FT = nmf["H"], list(nmf["titles"])               # H: factors x 52 loadings; FT: factor titles
llm_dirs = sorted(glob.glob(str(P.LLM_OUTPUTS / "*_val_*")))
models = sorted({d.split("/")[-1].rsplit("_val_", 1)[0] for d in llm_dirs})    # "<model>_val_<ds>"
for model in models:
    for ds, labs in [(k, v) for k, v in mapping.items() if not k.startswith("_")]:
        dd = P.LLM_OUTPUTS / f"{model}_val_{ds}"
        if not dd.exists():
            continue
        raw = pd.concat([pd.read_parquet(p) for p in glob.glob(str(dd / "*.parquet"))], ignore_index=True)
        raw = raw[raw["replicate"] >= 0]                 # replicate -1: prompt too long, not generated
        parsed = [parse_one(r) for r in raw["response"]]    # (reasoning, rating, confidence, status)
        raw["rating"] = [p[1] for p in parsed]; raw["confidence"] = [p[2] for p in parsed]
        raw.loc[(raw["rating"] < 0) | (raw["rating"] > 100), "rating"] = np.nan    # off the 0-100 scale
        # post x symptom table of ratings averaged over the replicates, all 52 symptoms as columns
        wide = raw.groupby(["item_id", "symptom_shortcode"])["rating"].mean().unstack().reindex(columns=codes)
        L = labels[labels["dataset"] == ds].set_index("item_id").loc[wide.index]
        Xs = wide.to_numpy(float)
        F = None
        if H is not None:
            # factor scores by nonnegative least squares, x ~ H^T f with f >= 0; missing ratings
            # are replaced by the mean of all ratings of this dataset
            Xf = np.nan_to_num(Xs, nan=np.nanmean(Xs))
            F = np.array([nnls(H.T, x)[0] for x in Xf])
        for lab, spec in labs.items():
            if lab not in L:
                continue
            y = L[lab].to_numpy(); ordinal = bool(spec.get("ordinal", False))
            key = "spearman" if ordinal else "auroc"
            pos = [codes.index(c) for c in spec.get("symptoms", [])]
            neg = [codes.index(c) for c in spec.get("inverse_symptoms", [])]
            agg = np.nanmax if spec.get("aggregate") == "max" else np.nanmean
            comp = agg(np.c_[Xs[:, pos], 100 - Xs[:, neg]] if neg else Xs[:, pos], axis=1)
            rows.append({"dataset": ds, "label": lab, "source": f"LM-QA {model}",
                         "predictor": "mapped-symptom composite",
                         **metric(y, comp, ordinal), **ci_cols(y, comp, ordinal, key)})
            # every symptom on its own (point estimates only)
            allm = []
            for j, c in enumerate(codes):
                mtr = metric(y, Xs[:, j], ordinal)
                allm.append(mtr[key])
                per_sym.append({"model": model, "dataset": ds, "label": lab, "symptom": c, key: mtr[key],
                                "mapped": c in spec.get("symptoms", []) + spec.get("inverse_symptoms", [])})
            if F is not None and spec.get("factor") in FT:
                j = FT.index(spec["factor"])
                rows.append({"dataset": ds, "label": lab, "source": f"LM-QA {model}",
                             "predictor": f"NMF factor: {spec['factor']}", **metric(y, F[:, j], ordinal),
                             **ci_cols(y, F[:, j], ordinal, key)})
            if F is not None and ordinal:
                tot = F.sum(axis=1)
                rows.append({"dataset": ds, "label": lab, "source": f"LM-QA {model}",
                             "predictor": "sum of NMF factors",
                             **metric(y, tot, ordinal), **ci_cols(y, tot, ordinal, key)})

# ---- save ------------------------------------------------------------------------------------------
res = pd.DataFrame(rows)
res["ci_lo"] = res["ci"].map(lambda c: c[0]); res["ci_hi"] = res["ci"].map(lambda c: c[1])
res.drop(columns="ci").to_csv(out_dir / "14_validation_metrics.csv", index=False)
if per_sym:
    pd.DataFrame(per_sym).to_csv(out_dir / "14_validation_per_symptom.csv", index=False)
save_json({"models_with_outputs": models}, out_dir / "14_validation_sources.json")
print(res.drop(columns="ci").round(3).to_string())
