#!/usr/bin/env python3
"""
Step 14b - validation against human annotations with the reduced protocol run on a single local
computer (Methods, "Validation against human annotations" and "Additional language models";
Results, Fig. 1d and Extended Data Fig. 3).

local_validation/run_validation.py scores balanced samples of posts from three human-annotated
Reddit datasets, and a sample of cohort windows (the calibration set), with one sample per request
from a locally served model (llama.cpp or Ollama).  Only the mapped and control symptoms are scored.
This script analyzes every result file local_validation/outputs/<model-tag>.jsonl, one per model
(for example llama31_8b_q4km_llamacpp, qwen36_35b_a3b_iq4xs_llamacpp):

1. Behavior of the rater: share of answers that are valid JSON and of ratings recovered by the parser
   (slm.parse.parse_one), refusals, length of the answers and answers that reached the token limit.
2. Discrimination of the human labels (interpersonal risk factors: thwarted belongingness `tbe` and
   perceived burdensomeness `pbu`; LoST: low self-esteem; depression severity on four levels, 0-3),
   with the pre-specified mapping config/validation_mapping.yaml.  Its SHA-256 is compared with the
   record config/validation_mapping.record.json, made before any model output on these posts
   existed.  Predictors of each label:
     - mapped-symptom composite: mean rating of the mapped symptoms, inverse items entering as
       100 - rating (AUROC; for severity also Spearman rho over the four levels, and the AUROC of
       severe versus minimum);
     - the same composite with its linear dependence on post length (log tokens) removed, the slope
       being fitted without the labels;
     - control-symptom composite (interpersonal risk factors and LoST): mean of three symptoms
       without a conceptual link to the labels, chosen in advance (initial insomnia, decreased
       appetite, gastrointestinal problems; expected AUROC about 0.5), with its paired difference
       from the composite;
     - every scored symptom on its own;
     - nine lexicon and length features (step 13) and the supervised reference (TF-IDF features and
       logistic regression, predictions from fivefold cross-validation over the full dataset), on
       the same posts, with the paired difference between the composite and the best lexicon feature.
   Intervals: 95% percentile bootstrap over posts (2,000 resamples); paired differences use the same
   resamples for both predictors.
3. Calibration: the calibration windows were rated once each on eight symptoms by the local model.
   These ratings are compared with the cohort ratings of the main run for the same windows and
   symptoms (mean of the five samples, same parser): Spearman and Pearson correlation, ICC(A,1),
   bias (local - cohort), Bland-Altman limits of agreement and mean absolute difference.  For a
   like-for-like comparison, each cohort sample is held out in turn; the local sample and the
   held-out cohort sample are each compared with the mean of the remaining four cohort samples, and
   the results are averaged over the five held-out samples.  The second comparison is the ceiling
   for a single sample (Extended Data Fig. 3c).

Comparison labels of the calibration table (column "comparison", keys of "calibration_all" in the
summary): "Local (1 sample) vs cohort mean of 5", "Local (1 sample) vs mean of 4 cohort samples",
"Cohort sample vs mean of the other 4 (ceiling)".

Inputs:  local_validation/outputs/<model-tag>.jsonl   raw answers (run_validation.py)
         local_validation/requests/posts.jsonl, calibration_windows.jsonl
                                     the scheduled items with their symptoms and rank (sample order),
                                     written by step 4b (04b_prepare_local_validation.py)
         data/processed/validation_labels.parquet, data/processed/lexicon_validation.parquet
         annotation/requests/validation_<dataset>.parquet   all posts (TF-IDF reference; step 4)
         data/processed/calibration_window_map.parquet      calibration item -> cohort window (step 4b)
         data/processed/llama31_8b/requests.parquet         cohort ratings per sample (step 5)
         config/validation_mapping.yaml, config/validation_mapping.record.json
Outputs (per model tag) in results/validation/:
         14b_<tag>_metrics.csv            one row per dataset, label, source and predictor
         14b_<tag>_per_symptom.csv        every scored symptom
         14b_<tag>_item_scores.csv        composite and control composite of every post
         14b_<tag>_calibration.csv        agreement with the cohort ratings, per symptom and overall
         14b_<tag>_calibration_pairs.csv  window x symptom: cohort_mean, cohort_n, local
         14b_<tag>_summary.json           run description, parse rates, refusals, calibration summary

usage: python3 14b_validation_local.py
"""
import argparse
import glob
import hashlib
import json
import warnings
from pathlib import Path

import _setup  # noqa: F401
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import roc_auc_score

from slm import params as P
from slm.io import save_json
from slm.parse import parse_one
from slm.psychometrics import icc_a1

# constant inputs (a symptom rated 0 for every post) and empty bootstrap resamples produce warnings
# that do not affect the results
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="An input array is constant")
ap = argparse.ArgumentParser(description="Reduced validation protocol scored on a single local computer.")
ap.add_argument("--outputs", default=str(P.ROOT / "local_validation" / "outputs" / "*.jsonl"),
                help="glob of the result files of local_validation/run_validation.py (one per model tag)")
ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples of posts")
ap.add_argument("--planned-preset", default="standard",
                help="preset of local_validation/run_validation.py that the protocol fixed in advance; "
                     "the summary reports the planned numbers of items and requests and the share completed")
ap.add_argument("--seed", type=int, default=0,
                help="seed of the bootstrap resamples (one generator per model)")
args = ap.parse_args()

# ---- inputs shared by all models -------------------------------------------------------------------
out_dir = P.RESULTS / "validation"; out_dir.mkdir(parents=True, exist_ok=True)
map_path = P.CONFIG / "validation_mapping.yaml"
mapping = yaml.safe_load(open(map_path))
pre = json.load(open(P.CONFIG / "validation_mapping.record.json"))      # SHA-256 recorded in advance
map_sha = hashlib.sha256(open(map_path, "rb").read()).hexdigest()
controls = mapping["_reduced_protocol"]["controls"]              # control symptoms per dataset
labels = pd.read_parquet(P.PROCESSED / "validation_labels.parquet")
lex = pd.read_parquet(P.PROCESSED / "lexicon_validation.parquet")
# the nine lexicon and length features (Methods); the VADER compound score is reversed, so that
# higher values mean more negative text
LEXICON = {"VADER negativity": lambda d: d["vader_neg"],
           "VADER compound (reversed)": lambda d: -d["vader_compound"],
           "First-person singular rate": lambda d: d["style_first_person"],
           "Absolutist-word rate": lambda d: d["style_absolutist"],
           "Empath negative_emotion": lambda d: d["empath_negative_emotion"],
           "Empath sadness": lambda d: d["empath_sadness"], "Empath shame": lambda d: d["empath_shame"],
           "Empath suffering": lambda d: d["empath_suffering"],
           "Post length (log tokens)": lambda d: np.log1p(d["n_tokens"])}


# ----------------------------------------------------------------------------- metrics
def metric(y, s, ordinal):
    """Discrimination of labels `y` by scores `s` over the posts where both are finite.

    Binary labels: {"auroc", "n"}.  Ordinal labels (severity 0-3): {"spearman", "auroc", "n"}, with
    Spearman rho over the four levels and the AUROC of severe (3) versus minimum (0).  A metric is
    NaN when it is undefined (a single class, or at most two posts for rho).
    """
    y = np.asarray(y, float); s = np.asarray(s, float)
    m = np.isfinite(s) & np.isfinite(y)
    y, s = y[m], s[m]
    if ordinal:
        rho = stats.spearmanr(y, s).statistic if len(y) > 2 else np.nan
        mm = np.isin(y, [0, 3])
        auc = roc_auc_score(y[mm] == 3, s[mm]) if len(np.unique(y[mm])) == 2 else np.nan
        return {"spearman": float(rho), "auroc": float(auc), "n": int(len(y))}
    auc = roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan
    return {"auroc": float(auc), "n": int(len(y))}


def boot_idx(n, rng):
    """args.n_boot bootstrap resamples (index arrays of length n, drawn with replacement)."""
    return [rng.randint(0, n, n) for _ in range(args.n_boot)]


def ci_of(y, s, ordinal, idx):
    """95% percentile intervals of every metric of `metric()` over the resamples `idx`.

    Returns {"auroc_ci_lo", "auroc_ci_hi"} (and the "spearman_" pair for ordinal labels); resamples
    on which a metric is undefined are skipped.
    """
    keys = ["spearman", "auroc"] if ordinal else ["auroc"]
    vals = {k: [] for k in keys}
    for i in idx:
        try:
            m = metric(y[i], s[i], ordinal)
        except ValueError:
            continue
        for k in keys:
            vals[k].append(m[k])
    out = {}
    for k, v in vals.items():
        v = np.array(v, float); v = v[np.isfinite(v)]
        out[f"{k}_ci_lo"] = float(np.quantile(v, .025)) if len(v) else np.nan
        out[f"{k}_ci_hi"] = float(np.quantile(v, .975)) if len(v) else np.nan
    return out


def paired_diff(y, s1, s2, ordinal, idx, key):
    """Paired difference metric(s1) - metric(s2) of the same posts, for the metric `key`.

    Returns {"diff": estimate on all posts, "diff_ci_lo"/"diff_ci_hi": 95% percentile interval over
    the resamples `idx` (both predictors evaluated on the same resample), "p_boot_le0": share of
    resamples with a difference <= 0 (one-sided bootstrap P value for s1 better than s2)}.
    """
    d = []
    for i in idx:
        try:
            d.append(metric(y[i], s1[i], ordinal)[key] - metric(y[i], s2[i], ordinal)[key])
        except ValueError:
            continue
    d = np.array(d, float); d = d[np.isfinite(d)]
    est = metric(y, s1, ordinal)[key] - metric(y, s2, ordinal)[key]
    return {"diff": float(est), "diff_ci_lo": float(np.quantile(d, .025)),
            "diff_ci_hi": float(np.quantile(d, .975)), "p_boot_le0": float(np.mean(d <= 0))}


# ----------------------------------------------------------------------------- TF-IDF reference
def tfidf_cv_scores(ds, lab, ordinal):
    """Supervised reference for one label: out-of-fold scores for every post of the full dataset.

    TF-IDF of word 1-2 grams (terms in >= 2 posts, sublinear term frequency) and L2 logistic
    regression (C = 4), with predictions from stratified fivefold cross-validation over all posts of
    the dataset (an upper reference, since it is trained on the labels).  Score: probability of the
    positive class, or the expected level sum_k k * p_k for the ordinal label.  Returns a Series
    indexed by item_id, or None if the dataset's post texts are missing.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    f = P.ROOT / "annotation" / "requests" / f"validation_{ds}.parquet"
    if not f.exists():
        return None
    v = pd.read_parquet(f, columns=["item_id", "posts_text"])
    L = labels[labels["dataset"] == ds].set_index("item_id")
    v = v[v["item_id"].isin(L.index)]
    y = L.loc[v["item_id"], lab].to_numpy()
    ok = np.isfinite(y)
    v, y = v[ok], y[ok].astype(int)
    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                        LogisticRegression(max_iter=2000, C=4.0))
    # each post was rendered like a one-post window, "'<text>'" (step 4); the slice removes the quotes
    pr = cross_val_predict(clf, v["posts_text"].str.slice(1, -1).tolist(), y,
                           cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")
    score = pr[:, 1] if pr.shape[1] == 2 else pr @ np.arange(pr.shape[1])
    return pd.Series(score, index=v["item_id"].to_numpy())


# ----------------------------------------------------------------------------- calibration reference
def cohort_replicates(cal_items):
    """Cohort ratings of the calibration windows, one row per item x symptom x sample.

    The calibration items (calib_000, ...) are mapped to their cohort window (speaker_id,
    window_start) with data/processed/calibration_window_map.parquet and joined with the parsed
    ratings of the main run (requests.parquet of step 5: one row per window, symptom and sample,
    rating NaN when no valid rating was returned).  All 52 symptoms of each window are returned.
    Returns a DataFrame with columns item_id, symptom_shortcode, replicate, rating, parse_status.
    """
    cmap = pd.read_parquet(P.PROCESSED / "calibration_window_map.parquet")
    cmap["window_start"] = pd.to_datetime(cmap["window_start"]).astype("datetime64[ns]")
    req = pd.read_parquet(P.processed_dir(P.MAIN_RUN) / "requests.parquet",
                          columns=["speaker_id", "window_start", "symptom_shortcode", "replicate", "rating",
                                   "parse_status"])
    # categorical / differently typed keys -> plain strings and nanosecond timestamps for the join
    req["speaker_id"] = req["speaker_id"].astype(str)
    req["window_start"] = pd.to_datetime(req["window_start"]).astype("datetime64[ns]")
    req["symptom_shortcode"] = req["symptom_shortcode"].astype(str)
    m = cmap[cmap["item_id"].isin(cal_items)].merge(req, on=["speaker_id", "window_start"], how="left")
    m = m.drop_duplicates(["item_id", "symptom_shortcode", "replicate"])   # one row per sample
    return m[["item_id", "symptom_shortcode", "replicate", "rating", "parse_status"]]


def agreement(a, b):
    """Agreement of two sets of ratings of the same window x symptom pairs (0-100 points).

    Pairs with a missing value are dropped; with fewer than 5 pairs only {"n"} is returned.
    Returns n, Spearman rho, Pearson r, ICC(A,1) (two-way random effects, absolute agreement),
    bias = mean(a - b), the Bland-Altman limits of agreement bias +- 1.96 SD(a - b) ("loa_lo",
    "loa_hi") and the mean absolute difference ("mae").
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 5:
        return {"n": int(len(a))}
    d = a - b
    return {"n": int(len(a)), "spearman": float(stats.spearmanr(a, b).statistic),
            "pearson": float(np.corrcoef(a, b)[0, 1]), "icc_a1": icc_a1(np.c_[a, b]),
            "bias": float(d.mean()), "loa_lo": float(d.mean() - 1.96 * d.std(ddof=1)),
            "loa_hi": float(d.mean() + 1.96 * d.std(ddof=1)), "mae": float(np.abs(d).mean())}


# ----------------------------------------------------------------------------- planned protocol
# EXPECTED: number of symptoms scheduled for every item (to recognize items left incomplete when a
# run was stopped).  PLANNED: items and requests of the preset fixed in advance.  Items are
# scheduled in the order of their `rank` (a random order, balanced across label levels), and a preset
# with n posts per dataset and m calibration windows takes the items with rank < n (< m).
EXPECTED = {}
# posts per dataset and calibration windows of each preset; must equal PRESETS in run_validation.py
PRESETS = {"smoke": (5, 5), "quick": (80, 40), "standard": (300, 80), "extended": (800, 150)}
_np, _nc = PRESETS[args.planned_preset]
PLANNED = {"preset": args.planned_preset, "posts_per_dataset": _np, "calibration_windows": _nc,
           "items": {}, "requests": 0}
for _f in ["posts.jsonl", "calibration_windows.jsonl"]:
    with open(P.ROOT / "local_validation" / "requests" / _f) as _h:
        for _line in _h:
            _r = json.loads(_line)
            EXPECTED[_r["item_id"]] = len(_r["symptoms"])
            _ds = _r.get("dataset", "calibration")
            if _r["rank"] < (_nc if _f.startswith("calibration") else _np):
                PLANNED["items"][_ds] = PLANNED["items"].get(_ds, 0) + 1
                PLANNED["requests"] += len(_r["symptoms"])

# ----------------------------------------------------------------------------- one model at a time
files = sorted(glob.glob(args.outputs))
if not files:
    raise SystemExit(f"no result files match {args.outputs}")
tfidf_cache = {}                                   # (dataset, label) -> TF-IDF scores; model-independent
for fn in files:
    tag = Path(fn).stem                            # model tag = file name, e.g. llama31_8b_q4km_llamacpp
    # ---- load the raw answers (one JSON object per line; a line cut off by a stopped run is skipped)
    rows = []
    with open(fn) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        print(f"{fn}: no results yet, skipped")
        continue
    # a request answered twice (an answer redone after a restart) counts once, with its last answer
    raw = pd.DataFrame(rows).drop_duplicates("key", keep="last")
    # use only items whose scheduled symptoms are all done (a run stopped mid-item leaves partial items)
    n_sched = raw["item_id"].map(EXPECTED)
    n_done = raw.groupby("item_id")["symptom_shortcode"].transform("nunique")
    n_partial = int(raw.loc[n_done < n_sched, "item_id"].nunique())
    raw = raw[n_done >= n_sched].copy()
    # ---- parse: rating and confidence from the JSON answer (embedded JSON or regex as fallbacks)
    parsed = [parse_one(r) for r in raw["response"]]
    raw["rating"] = [p[1] for p in parsed]; raw["confidence"] = [p[2] for p in parsed]
    raw["parse_status"] = [p[3] for p in parsed]
    raw.loc[(raw["rating"] < 0) | (raw["rating"] > 100), "rating"] = np.nan    # off the 0-100 scale
    # refusals: no rating and the model declined (Llama-3.1 safety behavior on explicit self-harm content)
    raw["refusal"] = raw["rating"].isna() & raw["response"].astype(str).str.contains(
        r"\bI (?:cannot|can't|can’t|won't|will not)\b|\bnot able to provide\b"
        r"|\bI(?:'m| am) (?:unable|not able)\b|\bI'm sorry, but\b", case=False, regex=True)
    stop = raw.get("stop_type", pd.Series(index=raw.index, dtype=object)).astype(str)
    # ---- run description
    _names = raw["model_name"].dropna().astype(str).unique().tolist() if "model_name" in raw else []
    summary = {"model_tag": tag, "model_name": _names[0] if _names else P.MODEL_NAMES.get(tag, tag),
               # answers that contain a reasoning ("thinking") part; the Qwen runs switch thinking off
               "thinking_answers": int(((pd.to_numeric(raw.get("reasoning_chars"), errors="coerce")
                                         .fillna(0) > 0)
                                        if "reasoning_chars" in raw
                                        else pd.Series(False, index=raw.index)).sum()),
               "file": str(Path(fn).relative_to(P.ROOT)) if P.ROOT in Path(fn).parents else fn,
               "n_requests": int(len(raw)), "n_partial_items_left_out": n_partial, "planned": PLANNED,
               "frac_of_planned_requests": float(len(raw) / max(PLANNED["requests"], 1)),
               "n_items_by_dataset": raw.groupby("dataset")["item_id"].nunique().to_dict(),
               "backend": sorted(raw.get("backend", pd.Series(["?"])).dropna().unique().tolist()),
               # model file reported by the server; only the file name is kept (no local folders)
               "server_model": sorted(raw.get("server_model", pd.Series(["?"])).dropna().astype(str)
                                      .map(lambda s: Path(s).name).unique().tolist()),
               "parse_status": raw["parse_status"].value_counts().to_dict(),
               "frac_strict_json": float((raw["parse_status"] == "json").mean()),
               "frac_rating_recovered": float(raw["rating"].notna().mean()),
               "output_tokens_median": float(pd.to_numeric(raw.get("n_output_tokens"),
                                                           errors="coerce").median()),
               # stop type "limit" (llama.cpp) or "length" (Ollama): the answer reached max new tokens
               "frac_hit_token_limit": float(stop.isin(["limit", "length"]).mean()),
               "n_by_dataset": raw["dataset"].value_counts().to_dict(),
               "frac_refusal": float(raw["refusal"].mean()),
               "refusal_by_dataset": raw.groupby("dataset")["refusal"].mean().round(4).to_dict(),
               "n_items_any_refusal": int(raw.loc[raw["refusal"], "item_id"].nunique()),
               "mapping_sha256": map_sha, "mapping_matches_prespecified": map_sha == pre["sha256"]}
    if "t_start" in raw and "t_end" in raw:        # wall-clock span of the run (epoch seconds -> hours)
        span_h = (raw["t_end"].max() - raw["t_start"].min()) / 3600
        summary["wall_hours"] = float(span_h)
        summary["requests_per_min"] = float(len(raw) / max(span_h * 60, 1e-9))
    # ---- discrimination of the human labels
    rng = np.random.RandomState(args.seed)         # the same seed for every model
    res, per_sym, item_scores = [], [], []
    for ds in ["irf", "lost", "severity"]:
        d = raw[raw["dataset"] == ds]
        if d.empty:
            continue
        # post x symptom table (one sample per request, so the mean is that sample's rating)
        wide = d.groupby(["item_id", "symptom_shortcode"])["rating"].mean().unstack()
        items = wide.index.to_numpy()
        L = labels[labels["dataset"] == ds].set_index("item_id").loc[items]
        lx = lex[lex["dataset"] == ds].set_index("item_id").reindex(items)
        idx = boot_idx(len(items), rng)            # one set of resamples for every predictor of the dataset
        for lab, spec in mapping[ds].items():
            if lab not in L:
                continue
            y = L[lab].to_numpy(float); ordinal = bool(spec.get("ordinal", False))
            # paired comparisons on the AUROC scale for every label (severity: severe vs minimum),
            # as in Fig. 1d
            key = "auroc"
            pos = [c for c in spec.get("symptoms", []) if c in wide]
            neg = [c for c in spec.get("inverse_symptoms", []) if c in wide]
            # composite: NaN-aware mean of the mapped ratings, inverse items as 100 - rating; NaN for
            # a post without any rating of a mapped symptom (e.g. every request declined)
            parts = [wide[pos].to_numpy(float)] + ([100 - wide[neg].to_numpy(float)] if neg else [])
            comp = np.nanmean(np.concatenate(parts, axis=1), axis=1)
            miss = ~np.isfinite(comp)
            base = {"model_tag": tag, "dataset": ds, "label": lab, "n_items": int(len(items)),
                    "n_positive": int(np.nansum(y > 0)) if not ordinal else None,
                    "n_items_no_rating": int(miss.sum()),
                    "frac_no_rating_by_level": {str(int(k)): float(miss[y == k].mean())
                                                for k in np.unique(y[np.isfinite(y)])}}
            res.append({**base, "source": "LM-QA", "predictor": "mapped-symptom composite",
                        "n_symptoms": len(pos) + len(neg), **metric(y, comp, ordinal),
                        **ci_of(y, comp, ordinal, idx)})
            # the same composite with the linear effect of post length removed: residual of an
            # ordinary least squares fit of the composite on log(1 + tokens), fitted without the labels
            ll = np.log1p(lx["n_tokens"].to_numpy(float))
            ok_ = np.isfinite(comp) & np.isfinite(ll)
            adj = np.full(len(comp), np.nan)
            if ok_.sum() > 3:
                b1, b0 = np.polyfit(ll[ok_], comp[ok_], 1)
                adj[ok_] = comp[ok_] - (b0 + b1 * ll[ok_])
            res.append({**base, "source": "LM-QA",
                        "predictor": "mapped-symptom composite, adjusted for post length",
                        "n_symptoms": len(pos) + len(neg), **metric(y, adj, ordinal),
                        **ci_of(y, adj, ordinal, idx)})
            # control-symptom composite (none for severity) and the per-post scores (Extended Data Fig. 3b)
            ctrl = [c for c in controls.get(ds, []) if c in wide]
            cs_all = np.nanmean(wide[ctrl].to_numpy(float), axis=1) if ctrl else np.full(len(items), np.nan)
            item_scores += [{"model_tag": tag, "dataset": ds, "label": lab, "item_id": i_, "y": float(y_),
                             "composite": float(c_), "control": float(k_)}
                            for i_, y_, c_, k_ in zip(items, y, comp, cs_all)]
            if ctrl:
                cs = np.nanmean(wide[ctrl].to_numpy(float), axis=1)
                res.append({**base, "source": "LM-QA", "predictor": "control-symptom composite",
                            "n_symptoms": len(ctrl), **metric(y, cs, ordinal), **ci_of(y, cs, ordinal, idx),
                            **{f"vs_composite_{k}": v
                               for k, v in paired_diff(y, comp, cs, ordinal, idx, key).items()}})
            # every scored symptom on its own, with its role for this label (Extended Data Fig. 3a)
            for c in wide.columns:
                mt = metric(y, wide[c].to_numpy(float), ordinal)
                per_sym.append({"model_tag": tag, "dataset": ds, "label": lab, "symptom": c,
                                "role": "mapped" if c in pos else "mapped (inverse)" if c in neg
                                else "control" if c in ctrl else "other",
                                **mt, "frac_missing": float(wide[c].isna().mean())})
            # lexicon and length features; the best one (highest AUROC) is compared with the composite
            best, best_v = None, -np.inf
            for nm, fx in LEXICON.items():
                s = fx(lx).to_numpy(float)
                mt = metric(y, s, ordinal)
                res.append({**base, "source": "lexicon", "predictor": nm, **mt, **ci_of(y, s, ordinal, idx)})
                if np.isfinite(mt[key]) and mt[key] > best_v:
                    best, best_v = (nm, s), mt[key]
            if best is not None:
                pdiff = paired_diff(y, comp, best[1], ordinal, idx, key)
                res.append({**base, "source": "comparison", "predictor": f"LM-QA composite - {best[0]}",
                            **{k: v for k, v in pdiff.items()}})
            # supervised reference (computed once per label, reused for every model)
            k_t = (ds, lab)
            if k_t not in tfidf_cache:
                tfidf_cache[k_t] = tfidf_cv_scores(ds, lab, ordinal)
            ts = tfidf_cache[k_t]
            if ts is not None:
                s = ts.reindex(items).to_numpy(float)
                res.append({**base, "source": "supervised", "predictor": "TF-IDF + logistic (5-fold CV)",
                            **metric(y, s, ordinal), **ci_of(y, s, ordinal, idx)})
    # ---- calibration against the cohort ratings of the main run
    cal = raw[raw["dataset"] == "calibration"]
    cal_rows = []
    if not cal.empty:
        cohort = cohort_replicates(cal["item_id"].unique())
        # parse outcomes of the cohort ratings of these windows (all 52 symptoms x 5 samples)
        summary["calibration_cohort_parse_status"] = (cohort["parse_status"].astype(str)
                                                      .value_counts().to_dict())
        # cohort mean and number of valid samples per window x symptom; joined with the local
        # rating, which keeps the eight calibration symptoms
        om = cohort.groupby(["item_id", "symptom_shortcode"])["rating"].agg(["mean", "count"])
        local = cal.groupby(["item_id", "symptom_shortcode"])["rating"].mean().rename("local")
        j = om.join(local, how="inner")
        (j.reset_index().rename(columns={"mean": "cohort_mean", "count": "cohort_n"}).assign(model_tag=tag)
         .to_csv(out_dir / f"14b_{tag}_calibration_pairs.csv", index=False))
        # cohort ratings with one column per sample (replicate 0-4), rows aligned with j
        piv = cohort.pivot_table(index=["item_id", "symptom_shortcode"], columns="replicate", values="rating")
        piv = piv.reindex(j.index)
        reps = [c for c in piv.columns]
        for sym, g in list(j.groupby(level="symptom_shortcode")) + [("all symptoms", j)]:
            row = {"model_tag": tag, "symptom": sym, "comparison": "Local (1 sample) vs cohort mean of 5",
                   **agreement(g["local"], g["mean"])}
            cal_rows.append(row)
            # like for like: a single sample against the mean of the same four cohort samples.
            # For each held-out cohort sample r: local vs mean(others) and sample r vs mean(others),
            # averaged over r.  The second is the agreement expected if the local deployment behaved
            # exactly like the cohort annotation (the ceiling for one sample).
            pv = piv.loc[g.index]
            loc_, ceil = [], []
            for r in reps:
                others = pv.drop(columns=r).mean(axis=1)      # NaN-aware mean of the other samples
                loc_.append(agreement(g["local"], others))
                ceil.append(agreement(pv[r], others))
            for name, lst in [("Local (1 sample) vs mean of 4 cohort samples", loc_),
                              ("Cohort sample vs mean of the other 4 (ceiling)", ceil)]:
                d = pd.DataFrame(lst)
                cal_rows.append({"model_tag": tag, "symptom": sym, "comparison": name,
                                 **{k: float(d[k].mean()) for k in d.columns if k != "n"},
                                 "n": int(d["n"].mean()) if "n" in d else None})
    # ---- save
    met = pd.DataFrame(res)
    met.to_csv(out_dir / f"14b_{tag}_metrics.csv", index=False)
    pd.DataFrame(per_sym).to_csv(out_dir / f"14b_{tag}_per_symptom.csv", index=False)
    pd.DataFrame(item_scores).to_csv(out_dir / f"14b_{tag}_item_scores.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(out_dir / f"14b_{tag}_calibration.csv", index=False)
    if cal_rows:
        allc = [r for r in cal_rows if r["symptom"] == "all symptoms"]
        summary["calibration_all"] = {r["comparison"]: {k: r.get(k)
                                                        for k in ["n", "spearman", "icc_a1", "bias", "mae"]}
                                      for r in allc}
        # parse outcomes of the local calibration answers, as shares
        calj = (raw[raw["dataset"] == "calibration"]["parse_status"]
                .value_counts(normalize=True).round(3).to_dict())
        summary["calibration_parse_status_local"] = calj
    save_json(summary, out_dir / f"14b_{tag}_summary.json")
    show = met[met["source"].isin(["LM-QA", "supervised", "comparison"])]
    print(json.dumps(summary, indent=1, default=str))
    print(show.round(3).to_string())
    if cal_rows:
        print(pd.DataFrame(cal_rows).round(3).to_string())
