#!/usr/bin/env python3
"""
Step 14c - paired comparison of two language models on the reduced validation protocol
(Methods, "Additional language models"; Results, paragraph on Qwen3.6-35B-A3B and Extended Data
Fig. 3e).

Every model was run by local_validation/run_validation.py on the same posts and calibration windows
and analyzed by step 14b.  Every other result file of local_validation/outputs/ (except further
Llama-3.1-8B runs) is compared with the reference Llama-3.1-8B-Instruct run.  The script reports:
  - discrimination of every label by the mapped-symptom composite (AUROC; for severity the AUROC of
    severe versus minimum and Spearman rho over the four levels), with the paired difference
    (other - reference), its 95% interval from 2,000 paired bootstrap resamples of posts and a
    two-sided bootstrap P value; only posts with a composite from both models are used;
  - for every model: refusals (overall and for posts labeled for perceived burdensomeness), answers
    that are valid JSON, ratings recovered by the parser, share of ratings equal to zero and number
    of distinct rating values;
  - dependence of the ratings on text length: Spearman rho between the mean rating of each item
    and its log10 word count, over the validation posts and over the calibration windows;
  - agreement (Spearman) of each model with the cohort ratings of the main run (mean of five
    samples) on the calibration windows;
  - throughput on the local computer (requests per hour of active running).

Inputs:  local_validation/outputs/<model-tag>.jsonl (raw answers of every model)
         local_validation/requests/posts.jsonl, calibration_windows.jsonl (texts, for word counts)
         results/validation/14b_<tag>_{item_scores.csv,summary.json} (step 14b, every model)
         results/validation/14b_<reference tag>_calibration_pairs.csv (cohort means, step 14b)
         data/processed/validation_labels.parquet
Outputs: results/validation/14c_model_comparison.json    {"reference", "models": properties per
                                                          model tag, "labels": the paired comparisons}
         results/validation/14c_model_comparison_labels.csv  the paired comparisons as a table

usage: python3 14c_model_comparison_local.py
"""
import argparse
import json
import re

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from slm import params as P
from slm.io import save_json
from slm.parse import parse_one

ap = argparse.ArgumentParser(description="Paired comparison of models on the reduced validation protocol.")
ap.add_argument("--ref", default="llama31_8b_q4km_llamacpp",
                help="model tag of the reference run (Llama-3.1-8B-Instruct served by llama.cpp)")
ap.add_argument("--n-boot", type=int, default=2000, help="paired bootstrap resamples of posts")
ap.add_argument("--seed", type=int, default=0, help="seed of the bootstrap resamples")
args = ap.parse_args()

VD = P.RESULTS / "validation"
OUT = P.ROOT / "local_validation" / "outputs"
LABELS = [("irf", "tbe"), ("irf", "pbu"), ("lost", "low_self_esteem"), ("severity", "severity")]
# a refusal: no rating and a declining phrase in the answer (same pattern as step 14b)
REFUSAL = re.compile(r"\bI (?:cannot|can't|can’t|won't|will not)\b|\bnot able to provide\b"
                     r"|\bI(?:'m| am) (?:unable|not able)\b|\bI'm sorry, but\b", re.I)


def load_raw(tag):
    """Raw answers of one model tag, parsed as in step 14b.

    One row per request (the last answer if a request was answered twice) with the run_validation.py
    fields plus rating (NaN if missing or off the 0-100 scale), status (parse outcome) and refusal.
    """
    rows = [json.loads(x) for x in open(OUT / f"{tag}.jsonl") if x.strip()]
    d = pd.DataFrame(rows).drop_duplicates("key", keep="last")
    parsed = [parse_one(r) for r in d["response"]]
    d["rating"] = [p[1] for p in parsed]
    d["status"] = [p[3] for p in parsed]
    d.loc[(d["rating"] < 0) | (d["rating"] > 100), "rating"] = np.nan
    d["refusal"] = d["rating"].isna() & d["response"].astype(str).str.contains(REFUSAL)
    return d


def words(text):
    """Number of whitespace-separated words of a text."""
    return len(re.findall(r"\S+", str(text)))


# ---- word counts of every post and calibration window; cohort means of the calibration pairs -------
REQ = P.ROOT / "local_validation" / "requests"
posts = pd.DataFrame([json.loads(x) for x in open(REQ / "posts.jsonl")])
cal = pd.DataFrame([json.loads(x) for x in open(REQ / "calibration_windows.jsonl")])
nwords = {**{r.item_id: words(r.posts_text) for r in posts.itertuples()},
          **{r.item_id: words(r.posts_text) for r in cal.itertuples()}}
pairs_ref = pd.read_csv(VD / f"14b_{args.ref}_calibration_pairs.csv")    # cohort means per window x symptom
cohort = pairs_ref.set_index(["item_id", "symptom_shortcode"])["cohort_mean"]

# the other models: every non-empty result file except the reference and other Llama-3.1-8B runs
others = sorted(p.stem for p in OUT.glob("*.jsonl")
                if p.stat().st_size > 0 and p.stem != args.ref and not p.stem.startswith("llama31_8b_q4km"))
raw = {t: load_raw(t) for t in [args.ref] + others}
items = {t: pd.read_csv(VD / f"14b_{t}_item_scores.csv") for t in [args.ref] + others}
summ = {t: json.load(open(VD / f"14b_{t}_summary.json")) for t in [args.ref] + others}


def model_props(t):
    """Properties of the model tag `t` over all its requests (validation posts and calibration windows).

    Active hours: time from the first request to the last answer minus every pause longer than
    10 minutes between consecutive answers (the run was stopped and resumed).  Shares are percentages.
    """
    d = raw[t]
    val = d[d["dataset"] != "calibration"]
    span = d["t_end"].sort_values().to_numpy()
    gaps = np.diff(span)                                          # seconds between consecutive answers
    active_h = (d["t_end"].max() - d["t_start"].min() - gaps[gaps > 600].sum()) / 3600
    # dependence on text length: mean rating per item vs log10(word count + 1)
    mr = d.groupby("item_id")["rating"].mean()
    lw = np.log10(pd.Series({i: nwords.get(i, np.nan) for i in mr.index}) + 1)
    vi = [i for i in mr.index if not i.startswith("calib")]          # validation posts
    ci = [i for i in mr.index if i.startswith("calib")]              # calibration windows
    # calibration: this model's rating vs the cohort mean of five samples, same window x symptom
    c = d[d["dataset"] == "calibration"].set_index(["item_id", "symptom_shortcode"])["rating"]
    j = pd.concat([c.rename("m"), cohort.rename("o")], axis=1, join="inner").dropna()
    return {"name": summ[t].get("model_name", t),
            "n_requests": int(len(d)),
            "refusals": int(d["refusal"].sum()), "pct_refusal": float(100 * d["refusal"].mean()),
            "items_with_refusal": int(d.loc[d["refusal"], "item_id"].nunique()),
            "pct_refusal_pbu_positive": None,                          # filled in below
            "pct_strict_json": float(100 * (d["status"] == "json").mean()),
            "pct_rating_recovered": float(100 * d["rating"].notna().mean()),
            "pct_zero": float(100 * (val["rating"] == 0).mean()),
            "n_distinct_values": int(val["rating"].dropna().nunique()),
            "mean_rating": float(val["rating"].mean()),
            "rho_length_posts": float(stats.spearmanr(mr[vi], lw[vi], nan_policy="omit").statistic),
            "rho_length_windows": float(stats.spearmanr(mr[ci], lw[ci], nan_policy="omit").statistic),
            "cal_rho_vs_cohort": float(stats.spearmanr(j["m"], j["o"]).statistic),
            "active_hours": float(active_h), "requests_per_hour": float(len(d) / active_h),
            "output_tokens_median": float(pd.to_numeric(d["n_output_tokens"], errors="coerce").median())}


props = {t: model_props(t) for t in [args.ref] + others}
# refusals among requests about posts labeled for perceived burdensomeness (pbu = 1)
lab = pd.read_parquet(P.PROCESSED / "validation_labels.parquet")
for t in props:
    d = raw[t]
    ipbu = d[(d["dataset"] == "irf")].merge(lab[["item_id", "pbu"]], on="item_id")
    props[t]["pct_refusal_pbu_positive"] = float(100 * ipbu.loc[ipbu["pbu"] == 1, "refusal"].mean())


def metric(y, s, kind):
    """One discrimination metric over the posts where labels and scores are finite.

    kind "auc": AUROC of a binary label; "sev_auc": AUROC of severe (3) versus minimum (0);
    "rho": Spearman rho over the four severity levels.  NaN when undefined.
    """
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    if kind == "rho":
        return stats.spearmanr(y, s).statistic
    if kind == "sev_auc":                       # severe (3) versus minimum (0)
        m = np.isin(y, [0, 3])
        return roc_auc_score(y[m] == 3, s[m]) if m.sum() > 2 and len(set(y[m])) == 2 else np.nan
    return roc_auc_score(y, s) if len(set(y)) == 2 else np.nan


# ---- paired comparison of the composites, label by label ------------------------------------------
# One generator for all comparisons, used in the order model -> label -> metric.
rng = np.random.RandomState(args.seed)
rows = []
for t in others:
    for ds, lb in LABELS:
        a = items[args.ref]; b = items[t]
        a = a[(a["dataset"] == ds) & (a["label"] == lb)].set_index("item_id")
        b = b[(b["dataset"] == ds) & (b["label"] == lb)].set_index("item_id")
        common = a.index.intersection(b.index)
        # posts with a composite from both models (Llama-3.1 declined every mapped symptom of a few posts)
        common = common[np.isfinite(a.loc[common, "composite"].to_numpy(float))
                        & np.isfinite(b.loc[common, "composite"].to_numpy(float))]
        y = a.loc[common, "y"].to_numpy(float)
        sa, sb = a.loc[common, "composite"].to_numpy(float), b.loc[common, "composite"].to_numpy(float)
        kinds = ["sev_auc", "rho"] if ds == "severity" else ["auc"]
        for kind in kinds:
            ma, mb = metric(y, sa, kind), metric(y, sb, kind)
            boots = []
            n = len(common)
            for _ in range(args.n_boot):
                ix = rng.randint(0, n, n)                 # the same resample for both models
                boots.append(metric(y[ix], sb[ix], kind) - metric(y[ix], sa[ix], kind))
            boots = np.array(boots, float)
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            # two-sided bootstrap P value: twice the smaller tail share of the differences around 0
            p_two = min(1.0, 2 * min(np.nanmean(boots <= 0), np.nanmean(boots >= 0)))
            rows.append({"model": t, "dataset": ds, "label": lb, "metric": kind, "n_items": int(n),
                         "ref": float(ma), "other": float(mb), "diff": float(mb - ma), "diff_lo": float(lo),
                         "diff_hi": float(hi), "p_two_sided": float(p_two)})
cmp = pd.DataFrame(rows)
cmp.to_csv(VD / "14c_model_comparison_labels.csv", index=False)
save_json({"reference": args.ref, "models": props, "labels": rows}, VD / "14c_model_comparison.json")
print(pd.DataFrame(props).T.to_string())
print(cmp.round(3).to_string())
