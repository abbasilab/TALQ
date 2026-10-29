#!/usr/bin/env python3
"""
Step 18 - agreement of the TALQ battery with two standard inventories completed by the same model
(Methods, "Agreement with standard inventories completed by the same model"; Results, paragraph on
the BDI-II and PHQ-9; Extended Data Fig. 4).

In a separate annotation job (P.INVENTORY_JOB), Llama-3.1-8B-Instruct answered the Beck Depression
Inventory (BDI-II, 21 items) and the Patient Health Questionnaire (PHQ-9, 9 items) for the windows of
the annotation, in the first person.  For each item it chose an option scored from 0 to 3, five times
per item (1,479,190 responses).  This script parses the answers, links the windows to their authors,
computes the totals and compares them with the 52-item battery of the main run.

Linkage of windows to authors.  The inventory job stored the same placeholder speaker_id in every
output document, so its outputs are keyed by the start of the window only.  Because window starts lie
on a grid of 14 days shared across users, a start identifies its author only when a single account
has a window starting at that time.  The inventory job was run on the windows of all 492 annotated
accounts, the 491 cohort members and the moderation bot AutoModerator, so the owner of a start is
looked up in data/processed/windows.parquet, which holds the windows of all 492 accounts (step 3): a
start shared with the bot is as ambiguous as one shared by two cohort members, and is left out.
Starts with more answers than one account's 30 items x 5 samples are left out as well.  The bot's
windows with a start of their own are linked to the bot: they are counted in n_windows_linked and
n_users_linked (not in n_windows_linked_cohort and n_users_linked_cohort) and drop out when the
inventories are joined with the battery, from which step 5 removed the bot.
Where the exported documents of the job include the posts (sample_full_docs.jsonl), two checks are
made: the author of the posts against the linked author (possible only for linked windows) and the
post text against the battery window with the same author and start, which checks the construction
of the windows but not the linkage.  An export with the column posts_speaker_id (author of the first
post, added by annotation/export_llm_outputs.py; the export analyzed in the manuscript does not have
it) allows every window to be linked by its author instead (--link-by-posts-author).

Analyses (pre-specified item correspondence config/inventory_item_mapping.yaml, Supplementary
Table 3; its SHA-256 is compared with the record config/inventory_item_mapping.record.json):
  * parse rates (each answer is scored by the digit of the chosen option, 0-3);
  * totals: BDI-II (0-63) and PHQ-9 (0-27), sums of the item means over the samples, for windows with
    every item answered; Spearman correlation with the analog built from the mapped battery items,
    with the mean of all 52 battery items and with the 12 factor scores, pooled and within person
    (scores centered within user, users with >= 3 windows), with 95% intervals from a bootstrap over
    users;
  * items: the correlation of each inventory item with its mapped battery items (convergent) and its
    median correlation with the other battery items (discriminant), and the rank of the mapped items
    among the 52.

Inputs:  data/llm_outputs[/llm_export]/<job>/part-*.parquet (stored answers; P.LLM_EXPORT) and
         sample_full_docs.jsonl (complete documents, optional), as written by
         annotation/export_llm_outputs.py
         data/processed/windows.parquet (windows of all 492 annotated accounts, step 3)
         data/processed/llama31_8b/collated_nona.parquet (windows with all 52 symptoms rated, step 5)
         data/processed/llama31_8b/factors.parquet, results/llama31_8b/06_nmf.npz (factors, step 6)
         config/inventory_item_mapping.yaml, config/inventory_item_mapping.record.json
Outputs: data/processed/inventories.parquet (totals and item means per linked window)
         results/inventories/18_totals.csv   correlations of the totals (pooled and within person)
         results/inventories/18_items.csv    convergent and discriminant correlations per item
         results/inventories/18_summary.json parse rates, linkage counts and checks, summaries

usage: python3 18_inventories.py
"""
import argparse
import glob
import hashlib
import json
import re

import _setup  # noqa: F401
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from scipy import stats

from slm import params as P
from slm.io import Timer, save_json
from slm.parse import _first_json_object
from slm.prompts import load_symptoms

ap = argparse.ArgumentParser(description="BDI-II and PHQ-9 completed by the same model versus the battery.")
ap.add_argument("--job", default=P.INVENTORY_JOB,
                help="job id of the stored inventory answers, read from data/llm_outputs/llm_export/<job>/ "
                     "(or data/llm_outputs/<job>/)")
ap.add_argument("--n-boot", type=int, default=1000,
                help="bootstrap resamples of users for the intervals of the correlations")
ap.add_argument("--link-by-posts-author", action="store_true",
                help="link windows by the author of their posts (column posts_speaker_id of the export) "
                     "instead of by unique window start; not used in the manuscript")
args = ap.parse_args()

src = P.LLM_EXPORT / args.job
odir = P.RESULTS / "inventories"; odir.mkdir(parents=True, exist_ok=True)
map_path = P.CONFIG / "inventory_item_mapping.yaml"
mapping = yaml.safe_load(open(map_path))          # {inventory: {item number: {title, symptoms, aggregate}}}
pre = json.load(open(P.CONFIG / "inventory_item_mapping.record.json"))       # SHA-256 recorded in advance
codes = [s["shortcode"] for s in load_symptoms()]  # the 52 battery symptoms
N_ITEMS = {"bdi2": 21, "phq9": 9}
summary = {"job": args.job, "mapping_sha256": hashlib.sha256(open(map_path, "rb").read()).hexdigest()}
summary["mapping_matches_prespecified"] = summary["mapping_sha256"] == pre["sha256"]

# ---------------------------------------------------------------------------- load + parse
files = sorted(glob.glob(str(src / "part-*.parquet")))
have = pq.ParquetFile(files[0]).schema_arrow.names
cols = ["speaker_id", "window_start", "inventory", "question_number", "replicate", "response"]
cols += [c for c in ["posts_speaker_id", "n_posts"] if c in have]    # added by newer exports only
with Timer("load"):
    raw = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
raw["window_start"] = pd.to_datetime(raw["window_start"]).astype("datetime64[ns]")
_LEAD = re.compile(r"^\s*([0-3])(?:[ab]|\.0+)?\s*$")       # "2", "2a"/"2b" (BDI-II items 16, 18), "2.0"
_RAW = re.compile(r'"?rating"?\s*[:=]\s*"?([0-3])')        # fallback: first digit after "rating:"


def parse_option(resp):
    """One inventory answer -> (option score 0-3 or NaN, parse status).

    The answer is read as JSON, else as the first JSON object embedded in the text
    (slm.parse._first_json_object), and its "rating" field must be an option 0-3.  BDI-II items 16
    (sleep) and 18 (appetite) offer options 1a/1b to 3a/3b, which are scored by their digit, as the
    BDI-II prescribes.  Without a usable JSON object, a regular expression reads the first digit after
    "rating".  Status: "json", "json_block" (embedded JSON), "regex", "invalid_option" (a rating that
    is not an option 0-3) or "fail".
    """
    s = str(resp)
    obj = None
    try:
        obj = json.loads(s)
        status = "json"
    except Exception:  # noqa: BLE001
        obj = _first_json_object(s)
        status = "json_block" if obj is not None else None
    if isinstance(obj, dict) and "rating" in obj:
        m = _LEAD.match(str(obj["rating"]))
        return (float(m.group(1)), status) if m else (np.nan, "invalid_option")
    m = _RAW.search(s)
    return (float(m.group(1)), "regex") if m else (np.nan, "fail")


with Timer("parse"):
    parsed = [parse_option(r) for r in raw["response"]]
raw["rating"] = np.array([p[0] for p in parsed], float)
raw["parse_status"] = [p[1] for p in parsed]
del parsed
summary["n_rows"] = int(len(raw))
summary["parse_status"] = raw["parse_status"].value_counts().to_dict()
summary["frac_valid_rating"] = float(np.isfinite(raw["rating"]).mean())
summary["stored_speaker_ids"] = raw["speaker_id"].value_counts().head(3).to_dict()   # the placeholder
raw = raw.drop(columns=["response"])

# ---------------------------------------------------------------------------- linkage
# windows of all 492 annotated accounts, bot included (the accounts the inventory job was run on)
win = pd.read_parquet(P.PROCESSED / "windows.parquet", columns=["speaker_id", "window_start"])
win["window_start"] = pd.to_datetime(win["window_start"]).astype("datetime64[ns]")
win["speaker_id"] = win["speaker_id"].astype(str)
if args.link_by_posts_author:
    if "posts_speaker_id" not in raw or not raw["posts_speaker_id"].notna().all():
        raise SystemExit("--link-by-posts-author needs an export with posts_speaker_id in every row")
    raw["user"] = raw["posts_speaker_id"].astype(str)
    summary["linkage"] = "posts_speaker_id (exported true author)"
else:
    # link by window start: only starts that belong to a single account
    n_users_at = win.groupby("window_start")["speaker_id"].nunique()
    single = n_users_at[n_users_at == 1].index
    owner = win[win["window_start"].isin(single)].set_index("window_start")["speaker_id"]
    rows_at = raw.groupby("window_start").size()
    full = rows_at[rows_at <= 150].index                   # at most one user's 30 items x 5 samples
    keep = raw["window_start"].isin(single) & raw["window_start"].isin(full)
    raw = raw[keep].copy()
    raw["user"] = raw["window_start"].map(owner)
    summary["linkage"] = "window start owned by a single cohort member"
    # check against the complete documents exported with the job (their posts name the author)
    chk = []
    sf = src / "sample_full_docs.jsonl"
    if sf.exists():
        for line in open(sf):
            d = json.loads(line)
            ws = pd.Timestamp(d["window_start"]).tz_localize(None) if pd.Timestamp(d["window_start"]).tzinfo \
                else pd.Timestamp(d["window_start"])
            if ws in owner.index:
                chk.append(owner.loc[ws] == d["posts"][0]["speaker_id"])
    summary["linkage_check"] = {"n_sample_docs_in_linked_windows": len(chk),
                                "n_correct": int(np.sum(chk)) if chk else 0}
    # the windows of the inventory job are those of the symptom job: compare the documents' post text
    # with the battery window of the same author and start
    if sf.exists():
        wt = pd.read_parquet(P.PROCESSED / "windows.parquet",
                             columns=["speaker_id", "window_start", "posts_text"])
        wt["window_start"] = pd.to_datetime(wt["window_start"]).astype("datetime64[ns]")
        wt = wt.set_index(["speaker_id", "window_start"])["posts_text"]
        same = {}
        for line in open(sf):
            d = json.loads(line)
            ws = pd.Timestamp(d["window_start"])
            ws = ws.tz_localize(None) if ws.tzinfo else ws
            k = (d["posts"][0]["speaker_id"], ws)
            if k not in same:
                same[k] = bool(k in wt.index and wt.loc[k] == d["posts_text"])
        summary["window_text_check"] = {"n_windows": len(same),
                                        "n_identical_posts_text": int(sum(same.values())),
                                        "n_users": len({k[0] for k in same})}
    # share of linked windows with all 150 answers stored (30 items x 5 samples)
    summary["rows_per_linked_window_150"] = float((raw.groupby("window_start").size() == 150).mean())
summary["n_rows_linked"] = int(len(raw))
summary["n_windows_linked"] = int(raw.groupby(["user", "window_start"]).ngroups)
summary["n_users_linked"] = int(raw["user"].nunique())
# the same counts without the bot account(s) (P.BOT_SPEAKERS): linked windows of cohort members
_coh = raw[~raw["user"].isin(P.BOT_SPEAKERS)]
summary["n_windows_linked_cohort"] = int(_coh.groupby(["user", "window_start"]).ngroups)
summary["n_users_linked_cohort"] = int(_coh["user"].nunique())
del _coh

# ---------------------------------------------------------------------------- item means and totals
# item score = mean over the valid samples of the item (n_valid of n_samples)
item = (raw.groupby(["user", "window_start", "inventory", "question_number"])["rating"]
        .agg(["mean", "count", "size"]).reset_index())
item = item.rename(columns={"mean": "rating", "count": "n_valid", "size": "n_samples"})
wide = item.pivot_table(index=["user", "window_start"], columns=["inventory", "question_number"],
                        values="rating")
tot = pd.DataFrame(index=wide.index)
for inv, n in N_ITEMS.items():
    sub = wide[inv] if inv in wide.columns.get_level_values(0) else None
    if sub is None:
        continue
    complete = sub.notna().sum(axis=1) == n                # total only when every item has a score
    tot[f"{inv}_total"] = sub.sum(axis=1).where(complete)
tot = tot.reset_index().rename(columns={"user": "speaker_id"})
wide_items = wide.copy(); wide_items.columns = [f"{a}_{b}" for a, b in wide_items.columns]   # e.g. bdi2_16
wide_items = wide_items.reset_index().rename(columns={"user": "speaker_id"})
out = tot.merge(wide_items, on=["speaker_id", "window_start"])
out.to_parquet(P.PROCESSED / "inventories.parquet", index=False)

# ---------------------------------------------------------------------------- join with the battery
# battery ratings of the main run (mean of the five samples), windows with all 52 symptoms rated;
# the bot is not in these tables, so its linked windows drop out here
coll = pd.read_parquet(P.processed_dir(P.MAIN_RUN) / "collated_nona.parquet")
coll["speaker_id"] = coll["speaker_id"].astype(str)
coll["window_start"] = pd.to_datetime(coll["window_start"]).astype("datetime64[ns]")
j = out.merge(coll, on=["speaker_id", "window_start"], how="inner")
# factor scores exist only for the confident windows of the main analysis (NaN elsewhere)
fac_path = P.processed_dir(P.MAIN_RUN) / "factors.parquet"
nmf_path = P.results_dir(P.MAIN_RUN) / "06_nmf.npz"
FT = None
if fac_path.exists() and nmf_path.exists():
    FT = list(np.load(nmf_path, allow_pickle=True)["titles"])
    fac = pd.read_parquet(fac_path,
                          columns=["speaker_id", "window_start"] + [f"nmf_{i}" for i in range(len(FT))])
    fac["speaker_id"] = fac["speaker_id"].astype(str)
    fac["window_start"] = pd.to_datetime(fac["window_start"]).astype("datetime64[ns]")
    j = j.merge(fac, on=["speaker_id", "window_start"], how="left")
summary["n_windows_joined"] = int(len(j))
summary["n_users_joined"] = int(j["speaker_id"].nunique())


def battery(df, spec):
    """Battery counterpart of one inventory item: mean (or maximum, for "aggregate: max") of the
    ratings of its mapped battery symptoms, per window (NaN-aware)."""
    v = df[[f"{c}_rating" for c in spec["symptoms"]]].to_numpy(float)
    return np.nanmax(v, axis=1) if spec.get("aggregate") == "max" else np.nanmean(v, axis=1)


# analog of each inventory: mean over its items of their battery counterparts (0-100 scale)
for inv in N_ITEMS:
    j[f"{inv}_analogue"] = np.nanmean(np.column_stack([battery(j, s) for s in mapping[inv].values()]), axis=1)
j["battery_mean"] = j[[f"{c}_rating" for c in codes]].mean(axis=1)


def corr_ci(a, b, groups=None, n_boot=args.n_boot, seed=0):
    """Spearman rho of a and b with a 95% percentile interval from a bootstrap over users.

    Pairs with a missing value are dropped; users (`groups`) are resampled with replacement, each
    bringing all its windows (a fresh generator with the same seed for every call).  Returns
    {"rho", "ci_lo", "ci_hi", "n" (windows), "n_users"}.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    g = np.asarray(groups)[ok] if groups is not None else np.arange(len(a))
    rho = stats.spearmanr(a, b).statistic
    rng = np.random.RandomState(seed)
    ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}
    bs = []
    for _ in range(n_boot):
        sel = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        bs.append(stats.spearmanr(a[sel], b[sel]).statistic)
    return {"rho": float(rho), "ci_lo": float(np.nanquantile(bs, .025)),
            "ci_hi": float(np.nanquantile(bs, .975)), "n": int(len(a)), "n_users": int(len(ug))}


def within(df, x, y, min_n=3):
    """Within-person correlation of columns x and y: both centered on each user's mean, users with at
    least `min_n` windows with both values; corr_ci of the centered values, keys prefixed "within_"."""
    d = df[["speaker_id", x, y]].dropna()
    cnt = d.groupby("speaker_id")[x].transform("size")
    d = d[cnt >= min_n]
    if d.empty:
        return {"n": 0}
    dx = d[x] - d.groupby("speaker_id")[x].transform("mean")
    dy = d[y] - d.groupby("speaker_id")[y].transform("mean")
    r = corr_ci(dx, dy, d["speaker_id"])
    return {f"within_{k}": v for k, v in r.items()}


# ---------------------------------------------------------------------------- totals
rows = []
pairs = [("bdi2_total", "phq9_total", "BDI-II total vs PHQ-9 total (same model, same role)"),
         ("bdi2_total", "bdi2_analogue", "BDI-II total vs LM-QA BDI-II analogue (mapped battery items)"),
         ("phq9_total", "phq9_analogue", "PHQ-9 total vs LM-QA PHQ-9 analogue (mapped battery items)"),
         ("bdi2_total", "battery_mean", "BDI-II total vs mean of all 52 battery items"),
         ("phq9_total", "battery_mean", "PHQ-9 total vs mean of all 52 battery items")]
if FT is not None:
    for i, t in enumerate(FT):
        pairs += [("bdi2_total", f"nmf_{i}", f"BDI-II total vs factor: {t}"),
                  ("phq9_total", f"nmf_{i}", f"PHQ-9 total vs factor: {t}")]
for x, y, label in pairs:
    if x in j and y in j:
        rows.append({"comparison": label, "x": x, "y": y, **corr_ci(j[x], j[y], j["speaker_id"]),
                     **within(j, x, y)})
tot_tab = pd.DataFrame(rows)
tot_tab.to_csv(odir / "18_totals.csv", index=False)

# ---------------------------------------------------------------------------- items
# convergent and discriminant correlations of every inventory item
item_rows = []
for inv in N_ITEMS:
    for q, spec in mapping[inv].items():
        col = f"{inv}_{q}"
        if col not in j:
            continue
        x = j[col].to_numpy(float)
        conv = stats.spearmanr(x, battery(j, spec), nan_policy="omit").statistic      # convergent
        r_all = {c: stats.spearmanr(x, j[f"{c}_rating"], nan_policy="omit").statistic for c in codes}
        mapped = set(spec["symptoms"])
        others = np.array([v for c, v in r_all.items() if c not in mapped], float)       # discriminant
        # rank of the best mapped symptom among the 52: 1 + number of unmapped symptoms that
        # correlate more strongly with the inventory item
        rank = 1 + int(np.sum(others > max(r_all[c] for c in mapped)))
        item_rows.append({"inventory": inv, "item": q, "title": spec["title"],
                          "mapped": ";".join(spec["symptoms"]), "convergent_rho": float(conv),
                          "discriminant_median_rho": float(np.nanmedian(others)),
                          "discriminant_max_rho": float(np.nanmax(others)),
                          "best_unmapped": max((c for c in codes if c not in mapped), key=lambda c: r_all[c]),
                          "rank_of_mapped_among_52": rank,
                          "frac_windows_nonzero": float(np.nanmean(x > 0)), "sd": float(np.nanstd(x))})
items_tab = pd.DataFrame(item_rows)
items_tab.to_csv(odir / "18_items.csv", index=False)

# ---------------------------------------------------------------------------- summary
summary["items"] = {inv: {"median_convergent": float(g["convergent_rho"].median()),
                          "median_discriminant": float(g["discriminant_median_rho"].median()),
                          "n_items_mapped_top1": int((g["rank_of_mapped_among_52"] == 1).sum()),
                          "n_items_mapped_top3": int((g["rank_of_mapped_among_52"] <= 3).sum()),
                          "n_items": int(len(g))}
                    for inv, g in items_tab.groupby("inventory")}
summary["totals"] = {r["comparison"]: {k: r[k] for k in ["rho", "ci_lo", "ci_hi", "n", "n_users"]}
                     | ({"within_rho": r.get("within_rho")} if "within_rho" in r else {})
                     for r in rows[:5]}                    # the five comparisons that are not factors
# distribution of the totals over the usual severity bands of each inventory
# (BDI-II 0-13, 14-19, 20-28, 29-63; PHQ-9 0-4, 5-9, 10-14, 15-19, 20-27)
summary["bdi2_severity_bands"] = (pd.cut(j["bdi2_total"], [-0.1, 13, 19, 28, 63],
                                         labels=["minimal", "mild", "moderate", "severe"])
                                  .value_counts(normalize=True).round(3).to_dict())
summary["phq9_severity_bands"] = (pd.cut(j["phq9_total"], [-0.1, 4, 9, 14, 19, 27],
                                         labels=["minimal", "mild", "moderate", "mod-severe", "severe"])
                                  .value_counts(normalize=True).round(3).to_dict())
save_json(summary, odir / "18_summary.json")
print(json.dumps(summary, indent=1, default=str))
print(tot_tab.round(3).to_string())
print(items_tab.round(3).to_string())
