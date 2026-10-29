#!/usr/bin/env python3
"""
Step 14d - predictive validity: how well the TALQ symptom ratings of a post, taken together, predict the
labels that human annotators gave the post (Results, "A mental health scoring system built on an LLM";
Fig. 1d; Extended Data Fig. 3a,b,e).

The posts are those of the reduced validation protocol (step 14b inputs, one rating per post and symptom).
Each dataset was scored on the symptoms related to its labels and, for the interpersonal and self-esteem
datasets, on three unrelated control symptoms (config/validation_mapping.yaml).  For every label, all
symptoms scored on its dataset are predictors (thwarted belongingness and perceived burdensomeness: 12
symptoms; low self-esteem: 8; depression severity: 20); no symptom is assigned to a label.

Models, each inside a cross-validation over posts (5 folds, repeated --n-repeats times, stratified by label):
  * lasso: L1-penalized logistic regression for the binary labels and L1-penalized linear regression of the
    severity level (0 minimum to 3 severe), with the penalty chosen by an inner 5-fold cross-validation;
  * MLP: one hidden layer of 16 units, with the L2 penalty chosen by an inner 5-fold cross-validation;
  * baseline: the same lasso on nine lexicon and length features of the same posts (VADER negativity and
    compound, Empath negative emotion, sadness, shame and suffering, first person singular and absolutist
    word rates, log length).
Ratings are standardized and missing ratings (refusals, unparsed responses) replaced by the median, both
estimated on the training folds.  Predictions are averaged over the repetitions; discrimination is the AUROC
(severity: Spearman rho with the level and the AUROC of severe against minimum), with 95% intervals from
--n-boot bootstrap resamples of posts.  The symptoms that carry the prediction are those that the lasso
keeps (nonzero coefficient) in at least 80% of the training folds, with the sign of the coefficient.

Inputs:  local_validation/outputs/<model-tag>.jsonl   raw answers (run_validation.py)
         data/processed/validation_labels.parquet, data/processed/lexicon_validation.parquet (steps 4, 13)
Outputs (per model tag) in results/validation/:
         14d_<tag>_predictive.csv   one row per label and model: AUROC (and Spearman rho for severity) with
                                    95% intervals and the standard deviation over repetitions
         14d_<tag>_selection.csv    every symptom: share of training folds in which the lasso kept it, mean
                                    coefficient (standardized ratings)
         14d_<tag>_predictions.csv  cross-validated prediction of every post, averaged over repetitions

usage: python3 14d_validation_predictive.py [--n-repeats 10]
"""
import argparse
import glob
import json
import warnings
from pathlib import Path

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from slm import params as P
from slm.parse import parse_one
from slm.prompts import load_symptoms

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
ap = argparse.ArgumentParser(description="Prediction of human labels from the TALQ ratings of each post.")
ap.add_argument("--outputs", default=str(P.ROOT / "local_validation" / "outputs" / "*.jsonl"),
                help="glob of result files of local_validation/run_validation.py (one per annotating model)")
ap.add_argument("--n-repeats", type=int, default=10, help="repetitions of the fivefold cross-validation (default: 10)")
ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples of posts (default: 2000)")
ap.add_argument("--seed", type=int, default=0, help="seed of the folds and of the bootstrap (default: 0)")
args = ap.parse_args()

out_dir = P.RESULTS / "validation"
labels = pd.read_parquet(P.PROCESSED / "validation_labels.parquet")
lex = pd.read_parquet(P.PROCESSED / "lexicon_validation.parquet")
LEX = ["vader_neg", "vader_compound", "style_first_person", "style_absolutist", "empath_negative_emotion",
       "empath_sadness", "empath_shame", "empath_suffering", "log_tokens"]
lex["log_tokens"] = np.log1p(lex["n_tokens"])
TITLE = {s["shortcode"]: s["title"] for s in load_symptoms()}
TASKS = [("irf", "tbe", False), ("irf", "pbu", False), ("lost", "low_self_esteem", False),
         ("severity", "severity", True)]
ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]


def lasso(ordinal):
    if ordinal:
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LassoCV(cv=5, alphas=50, max_iter=20000, random_state=0))
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegressionCV(Cs=20, cv=5, penalty="l1", solver="liblinear", scoring="roc_auc",
                                              max_iter=5000, random_state=0))


def mlp(ordinal):
    net = (MLPRegressor if ordinal else MLPClassifier)(hidden_layer_sizes=(16,), max_iter=3000, random_state=0)
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), net)
    key = f"{pipe.steps[-1][0]}__alpha"
    return GridSearchCV(pipe, {key: ALPHAS}, cv=5, scoring="neg_mean_squared_error" if ordinal else "roc_auc")


def predict(model, X, ordinal):
    return model.predict(X) if ordinal else model.predict_proba(X)[:, 1]


def coefs(model, ordinal):
    est = model.steps[-1][1]
    return np.ravel(est.coef_)


def metric(y, s, ordinal):
    if ordinal:
        mm = np.isin(y, [0, 3])
        return {"spearman": float(stats.spearmanr(y, s).statistic),
                "auroc": float(roc_auc_score(y[mm] == 3, s[mm]))}
    return {"auroc": float(roc_auc_score(y, s))}


def boot_ci(y, s, ordinal, rng):
    vals = {}
    for _ in range(args.n_boot):
        i = rng.randint(0, len(y), len(y))
        try:
            m = metric(y[i], s[i], ordinal)
        except ValueError:
            continue
        for k, v in m.items():
            vals.setdefault(k, []).append(v)
    return {f"{k}_ci_lo": float(np.nanquantile(v, .025)) for k, v in vals.items()} | \
           {f"{k}_ci_hi": float(np.nanquantile(v, .975)) for k, v in vals.items()}


for fn in sorted(glob.glob(args.outputs)):
    tag = Path(fn).stem
    raw = pd.DataFrame([json.loads(x) for x in open(fn) if x.strip()]).drop_duplicates("key", keep="last")
    raw = raw[raw["dataset"].isin(["irf", "lost", "severity"])].copy()
    parsed = [parse_one(r) for r in raw["response"]]
    raw["rating"] = [p[1] for p in parsed]
    raw.loc[(raw["rating"] < 0) | (raw["rating"] > 100), "rating"] = np.nan
    rows, sel_rows, pred_rows = [], [], []
    for ds, lab, ordinal in TASKS:
        wide = raw[raw["dataset"] == ds].groupby(["item_id", "symptom_shortcode"])["rating"].mean().unstack()
        L = labels[labels["dataset"] == ds].set_index("item_id").reindex(wide.index)
        ok = L[lab].notna().to_numpy()
        X, y = wide.to_numpy(float)[ok], L[lab].to_numpy(float)[ok]
        syms = list(wide.columns)
        strat = y.astype(int)
        # baseline: the nine lexicon and length features of the same posts, combined by the same lasso
        XL = lex[lex["dataset"] == ds].set_index("item_id").reindex(wide.index[ok])[LEX].to_numpy(float)
        rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.n_repeats, random_state=args.seed)
        for name, make, XX in (("lasso", lasso, X), ("MLP", mlp, X), ("lexicon lasso", lasso, XL)):
            pred = np.zeros((args.n_repeats, len(y)))
            fold_metric, nz = [], []
            for k, (tr, te) in enumerate(rcv.split(XX, strat)):
                m = clone(make(ordinal)).fit(XX[tr], y[tr] if ordinal else y[tr].astype(int))
                pred[k // 5, te] = predict(m, XX[te], ordinal)
                if name == "lasso":
                    nz.append(coefs(m, ordinal))
            for r in range(args.n_repeats):
                fold_metric.append(metric(y, pred[r], ordinal))
            avg = pred.mean(axis=0)
            pred_rows += [{"model_tag": tag, "dataset": ds, "label": lab, "model": name, "item_id": i_, "y": float(y_),
                           "prediction": float(p_)} for i_, y_, p_ in zip(wide.index[ok], y, avg)]
            rng = np.random.RandomState(args.seed)
            met = metric(y, avg, ordinal)
            rows.append({"model_tag": tag, "dataset": ds, "label": lab, "model": name, "n_posts": int(len(y)),
                         "n_predictors": XX.shape[1], **met, **boot_ci(y, avg, ordinal, rng),
                         **{f"{k}_repeat_sd": float(np.std([f[k] for f in fold_metric])) for k in met}})
            if name == "lasso":
                C = np.array(nz)
                for j, s in enumerate(syms):
                    sel_rows.append({"model_tag": tag, "dataset": ds, "label": lab, "symptom": s,
                                     "title": TITLE.get(s, s), "selected_frac": float(np.mean(C[:, j] != 0)),
                                     "mean_coef": float(C[:, j].mean())})
            print(tag, lab, name, {k: round(v, 3) for k, v in rows[-1].items() if isinstance(v, float)})
    pd.DataFrame(rows).to_csv(out_dir / f"14d_{tag}_predictive.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(out_dir / f"14d_{tag}_selection.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out_dir / f"14d_{tag}_predictions.csv", index=False)
print("done")
