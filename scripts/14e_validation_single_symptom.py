#!/usr/bin/env python3
"""
Step 14e - comparators for the predictive validation of step 14d (Results, "A mental health scoring system
built on an LLM"; Methods, "Validation against human annotations").

Best single symptom: for every label, the scored symptom that best separates the labels in the training
folds (largest |AUROC - 0.5| for the binary labels, largest |Spearman rho| with the level for severity) is
chosen within each training fold of the same cross-validation as step 14d (5 folds over posts, repeated
--n-repeats times, stratified by label, same seed), with its sign, and its ratings are the prediction for the
held-out posts.  Missing ratings are replaced by the training median.  Predictions are averaged over the
repetitions and summarized as in step 14d (AUROC, for severity also Spearman rho with the level and the
AUROC of severe against minimum depression), with 95% intervals from --n-boot bootstrap resamples of posts.

Paired differences (same posts, --n-boot joint bootstrap resamples of posts): lasso minus best single symptom,
lasso minus the lexicon lasso and neural network minus lasso for each annotating model, and each other
annotating model minus the reference model (Llama-3.1-8B) for the lasso.

Inputs:  local_validation/outputs/<model-tag>.jsonl   raw answers (run_validation.py)
         data/processed/validation_labels.parquet      human labels (step 4)
         results/validation/14d_<tag>_predictions.csv  cross-validated predictions of step 14d
Outputs in results/validation/:
         14e_<tag>_single_symptom.csv              one row per label: AUROC (and Spearman rho for severity)
                                                   with 95% intervals, the symptom chosen most often and the
                                                   share of folds in which it was chosen
         14e_<tag>_single_symptom_predictions.csv  cross-validated prediction of every post
         14e_differences.csv                       paired differences with 95% intervals and the bootstrap
                                                   share of differences at or below zero

usage: python3 14e_validation_single_symptom.py [--n-repeats 10]
"""
import argparse
import glob
import json
from pathlib import Path

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from slm import params as P
from slm.parse import parse_one

ap = argparse.ArgumentParser(description="Best single symptom and paired differences for the predictive validation.")
ap.add_argument("--outputs", default=str(P.ROOT / "local_validation" / "outputs" / "*.jsonl"),
                help="glob of result files of local_validation/run_validation.py (one per annotating model)")
ap.add_argument("--n-repeats", type=int, default=10, help="repetitions of the fivefold cross-validation (default: 10)")
ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples of posts (default: 2000)")
ap.add_argument("--seed", type=int, default=0, help="seed of the folds and of the bootstrap (default: 0)")
args = ap.parse_args()

out_dir = P.RESULTS / "validation"
labels = pd.read_parquet(P.PROCESSED / "validation_labels.parquet")
TASKS = [("irf", "tbe", False), ("irf", "pbu", False), ("lost", "low_self_esteem", False),
         ("severity", "severity", True)]


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
    raw["rating"] = [parse_one(r)[1] for r in raw["response"]]
    raw.loc[(raw["rating"] < 0) | (raw["rating"] > 100), "rating"] = np.nan
    rows, pred_rows = [], []
    for ds, lab, ordinal in TASKS:
        wide = raw[raw["dataset"] == ds].groupby(["item_id", "symptom_shortcode"])["rating"].mean().unstack()
        L = labels[labels["dataset"] == ds].set_index("item_id").reindex(wide.index)
        ok = L[lab].notna().to_numpy()
        X, y = wide.to_numpy(float)[ok], L[lab].to_numpy(float)[ok]
        rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.n_repeats, random_state=args.seed)
        pred = np.zeros((args.n_repeats, len(y)))
        picks = []
        for k, (tr, te) in enumerate(rcv.split(X, y.astype(int))):
            med = np.nanmedian(X[tr], axis=0)
            Xtr = np.where(np.isnan(X[tr]), med, X[tr])
            Xte = np.where(np.isnan(X[te]), med, X[te])
            if ordinal:
                sc = np.array([stats.spearmanr(Xtr[:, j], y[tr]).statistic for j in range(X.shape[1])])
                j = int(np.nanargmax(np.abs(sc))); sgn = np.sign(sc[j])
            else:
                sc = np.array([roc_auc_score(y[tr], Xtr[:, j]) for j in range(X.shape[1])])
                j = int(np.argmax(np.abs(sc - 0.5))); sgn = np.sign(sc[j] - 0.5)
            pred[k // 5, te] = sgn * Xte[:, j]
            picks.append(wide.columns[j])
        avg = pred.mean(axis=0)
        met = metric(y, avg, ordinal)
        top = pd.Series(picks).value_counts()
        pred_rows += [{"model_tag": tag, "dataset": ds, "label": lab, "model": "best single symptom", "item_id": i_,
                       "y": float(y_), "prediction": float(p_)} for i_, y_, p_ in zip(wide.index[ok], y, avg)]
        rows.append({"model_tag": tag, "dataset": ds, "label": lab, "model": "best single symptom",
                     "n_posts": int(len(y)), **met, **boot_ci(y, avg, ordinal, np.random.RandomState(args.seed)),
                     "symptom_most_chosen": top.index[0], "share_of_folds": float(top.iloc[0] / len(picks))})
        print(tag, lab, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()})
    pd.DataFrame(rows).to_csv(out_dir / f"14e_{tag}_single_symptom.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out_dir / f"14e_{tag}_single_symptom_predictions.csv", index=False)


# ---------------------------------------------------------------- paired differences between predictions
def paired(a, b, ordinal, key, rng):
    """a, b: predictions indexed by item_id (columns y, prediction); difference a - b in metric key."""
    ids = a.index.intersection(b.index)
    y = a.loc[ids, "y"].to_numpy(float)
    sa, sb = a.loc[ids, "prediction"].to_numpy(float), b.loc[ids, "prediction"].to_numpy(float)
    est = metric(y, sa, ordinal)[key] - metric(y, sb, ordinal)[key]
    d = []
    for _ in range(args.n_boot):
        i = rng.randint(0, len(y), len(y))
        try:
            d.append(metric(y[i], sa[i], ordinal)[key] - metric(y[i], sb[i], ordinal)[key])
        except ValueError:
            continue
    d = np.asarray(d)
    return {"n_posts": int(len(ids)), "diff": float(est), "ci_lo": float(np.quantile(d, .025)),
            "ci_hi": float(np.quantile(d, .975)), "p_boot_le0": float(np.mean(d <= 0))}


preds = {}
for f in sorted(out_dir.glob("14d_*_predictions.csv")):
    t_ = f.name[len("14d_"):-len("_predictions.csv")]
    g = pd.read_csv(f)
    f_single = out_dir / f"14e_{t_}_single_symptom_predictions.csv"
    if f_single.exists():
        g = pd.concat([g, pd.read_csv(f_single)], ignore_index=True)
    preds[t_] = g
ref = next((t for t in ["llama31_8b_q4km_llamacpp", "llama31_8b_q4km_ollama"] if t in preds), None)
diff_rows = []
for ds, lab, ordinal in TASKS:
    keys = ["auroc", "spearman"] if ordinal else ["auroc"]

    def get(t_, m_):
        g = preds[t_]
        return g[(g["dataset"] == ds) & (g["label"] == lab) & (g["model"] == m_)].set_index("item_id")

    for t_ in preds:
        for a_, b_ in [("lasso", "best single symptom"), ("lasso", "lexicon lasso"), ("MLP", "lasso")]:
            A_, B_ = get(t_, a_), get(t_, b_)
            if len(A_) and len(B_):
                for key in keys:
                    diff_rows.append({"comparison": f"{a_} minus {b_}", "model_tag": t_, "dataset": ds, "label": lab,
                                      "metric": key, **paired(A_, B_, ordinal, key, np.random.RandomState(args.seed))})
    for t_ in preds:
        if ref is None or t_ == ref:
            continue
        A_, B_ = get(t_, "lasso"), get(ref, "lasso")
        if len(A_) and len(B_):
            for key in keys:
                diff_rows.append({"comparison": f"{t_} minus {ref} (lasso)", "model_tag": t_, "dataset": ds,
                                  "label": lab, "metric": key,
                                  **paired(A_, B_, ordinal, key, np.random.RandomState(args.seed))})
D = pd.DataFrame(diff_rows)
D.to_csv(out_dir / "14e_differences.csv", index=False)
print(D.round(3).to_string())
print("done")
