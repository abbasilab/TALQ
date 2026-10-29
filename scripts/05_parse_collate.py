#!/usr/bin/env python3
"""
Step 5 - parse the stored model responses and build the window x symptom tables (Methods,
"Language model questionnaire" and "Symptom profiles"; Fig. 1b,c, Extended Data Figs. 1 and 2).

Every response is parsed as JSON; failing that, an embedded JSON object (for example in a fenced
code block) is extracted; failing that, rating and confidence are read with regular expressions;
otherwise the response is missing.  Values outside the 0-100 scale are missing (slm.parse).  Then:

  * duplicated outputs of the same request (same account, window, symptom and sample; 33 in the
    main run) are reduced to their first occurrence in export order;
  * the moderation bot (P.BOT_SPEAKERS) is removed, which leaves the 491 users;
  * ratings and confidences are averaged over the available samples of each window and symptom
    (missing values ignored);
  * the tables are restricted to users with at least 200 window-symptom pairs, to windows with all
    52 symptoms rated, and to windows whose mean confidence exceeds 40 for every symptom
    (7,094 windows from 481 users in the main run).

Inputs (parquet parts with one row per response: speaker_id, window_start, symptom_title and/or
symptom_shortcode, replicate, response)
  run llama31_8b   data/llm_outputs/[llm_export/]<P.SYMPTOM_JOB>/   (stored outputs of the annotation job)
  run synthetic    data/llm_outputs/_synthetic/<P.SYMPTOM_JOB>/      (tests/make_synthetic_llm_export.py)
  other runs       --src, or data/llm_outputs/<run>/                 (annotation/run_vllm_annotation.py)
Outputs (in data/processed/<run>/ and results/<run>/)
  requests.parquet           one row per stored response: speaker_id, window_start, symptom_shortcode,
                             symptom_number, replicate, rating, confidence, parse_status, export_row
                             (all accounts and duplicates kept)
  summary.parquet            one row per (account, window, symptom): replicate means, SDs and counts
  collated.parquet           one row per window of the users with >= 200 window-symptom pairs:
                             <shortcode>_rating and <shortcode>_confidence for the 52 symptoms
  collated_nona.parquet      windows with all 52 ratings and confidences available
  collated_confident.parquet windows of collated_nona with mean confidence > 40 for every symptom
                             (the input of the factor analysis, step 6)
  05_parse_report.json       row counts, parse-status counts, number of duplicated requests, and
                             the counts of the collation (key "collation")

Usage (from scripts/):
  python3 05_parse_collate.py --run llama31_8b
  python3 05_parse_collate.py --run synthetic              (pipeline test only)
  python3 05_parse_collate.py --run <name> --src <folder>  (another annotation run)
"""
import argparse
import glob
import os
from pathlib import Path

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from slm import params as P
from slm.io import Timer, save_json
from slm.parse import collate, parse_responses, summarize_replicates

ap = argparse.ArgumentParser(
    description="Parse stored model responses and build the window x symptom tables.")
ap.add_argument("--run", default=P.MAIN_RUN,
                help="name of the annotation run; outputs go to data/processed/<run>/ and "
                     f"results/<run>/ (default: {P.MAIN_RUN})")
ap.add_argument("--src", default=None,
                help="folder with the parquet parts of the stored responses (default: "
                     f"{P.MAIN_RUN}: data/llm_outputs/[llm_export/]<job>/; synthetic: "
                     "data/llm_outputs/_synthetic/<job>/; other runs: data/llm_outputs/<run>/)")
args = ap.parse_args()

# ---- step: locate the stored responses ----
default_src = {P.MAIN_RUN: P.LLM_EXPORT / P.SYMPTOM_JOB,
               "synthetic": P.LLM_OUTPUTS / "_synthetic" / P.SYMPTOM_JOB}
src = Path(args.src) if args.src else default_src.get(args.run, P.LLM_OUTPUTS / args.run)
parts = sorted(glob.glob(str(src / "*.parquet")))       # part-00000.parquet, ... in export order
assert parts, f"no parquet parts in {src}"
outd = P.processed_dir(args.run); outd.mkdir(parents=True, exist_ok=True)
resd = P.results_dir(args.run); resd.mkdir(parents=True, exist_ok=True)

# ---- step: load the parts (only the columns needed; strings as categoricals to save memory) ----
with Timer(f"loading {len(parts)} parts from {src}"):
    import pyarrow.parquet as pq
    keep = ["speaker_id", "window_start", "symptom_title", "symptom_shortcode", "replicate", "response",
            "item_id", "window_index"]
    frames = []
    for p in parts:
        names = pq.ParquetFile(p).schema_arrow.names
        f = pd.read_parquet(p, columns=[c for c in keep if c in names])
        for c in ["speaker_id", "symptom_title", "symptom_shortcode"]:
            if c in f:
                f[c] = f[c].astype("category")
        frames.append(f)
    raw = pd.concat(frames, ignore_index=True)
    del frames
    for c in ["speaker_id", "symptom_title", "symptom_shortcode"]:
        if c in raw:
            raw[c] = raw[c].astype("category")
if "replicate" in raw:
    raw = raw[raw["replicate"] >= 0].copy()        # GPU runner: rows of prompts skipped as too long
raw["window_start"] = pd.to_datetime(raw["window_start"])
# window_start is a naive Pacific wall-clock time (slm.windows); an export that returns it as a
# timezone-aware UTC value is converted back to the same naive value
if getattr(raw["window_start"].dt, "tz", None) is not None:
    raw["window_start"] = raw["window_start"].dt.tz_convert("UTC").dt.tz_localize(None)
raw["export_row"] = np.arange(len(raw))            # position in the concatenated export
# source folder for the report: relative to the repository root when inside it (no machine paths)
src_abs = Path(os.path.abspath(src))               # absolute, ".." removed, symlinks kept
src_rel = str(src_abs.relative_to(P.ROOT)) if src_abs.is_relative_to(P.ROOT) else str(src)
report = {"run": args.run, "src": src_rel, "n_rows": int(len(raw)),
          "n_requests_expected": P.EXPECTED_N_REQUESTS,   # requests submitted (all 492 accounts)
          "n_speakers": int(raw["speaker_id"].nunique())}

# ---- step: parse every response ----
responses = raw.pop("response").tolist()
with Timer("parsing"):
    parsed = parse_responses(raw, responses=responses)
del responses
report["parse_status"] = parsed["parse_status"].value_counts().to_dict()

keep = ["speaker_id", "window_start", "symptom_shortcode", "symptom_number", "replicate",
        "response_rating", "response_confidence", "parse_status", "export_row"]
req = parsed[keep].rename(columns={"response_rating": "rating", "response_confidence": "confidence"})
req.to_parquet(outd / "requests.parquet", index=False, compression="zstd")

# ---- step: duplicates and bot removal ----
# duplicated outputs of the same request (e.g. re-queued jobs): first occurrence in export order
dkey = ["speaker_id", "window_start", "symptom_shortcode", "replicate"]
dup = parsed.duplicated(dkey, keep="first") if "replicate" in parsed else pd.Series(False, index=parsed.index)
report["n_duplicate_requests"] = int(dup.sum())
d = parsed[~dup.to_numpy()]
d = d[~d["speaker_id"].isin(P.BOT_SPEAKERS)]       # the moderation bot is not a user

# ---- step: replicate means and window x symptom tables ----
with Timer("summarizing replicates"):
    summ = summarize_replicates(d)
coll, nona, conf, rep = collate(summ)
summ.to_parquet(outd / "summary.parquet", index=False)
coll.to_parquet(outd / "collated.parquet", index=False)
nona.to_parquet(outd / "collated_nona.parquet", index=False)
conf.to_parquet(outd / "collated_confident.parquet", index=False)
report["collation"] = rep
print("collation", rep)

save_json(report, resd / "05_parse_report.json")
print({k: v for k, v in report.items() if k != "collation"})
