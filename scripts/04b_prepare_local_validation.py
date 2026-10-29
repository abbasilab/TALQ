#!/usr/bin/env python3
"""
Step 4b - request files of the reduced validation protocol run on a single local computer
(local_validation/; Methods, "Validation against human annotations" and "Additional language models").

local_validation/run_validation.py scores the items listed in local_validation/requests/.  This step
writes them:

  posts.jsonl                800 posts per dataset (interpersonal risk factors `irf`, LoST `lost`,
                             depression severity `severity`), balanced over the label values; one JSON
                             object per line with item_id, dataset, rank, posts_text and symptoms
  calibration_windows.jsonl  150 cohort windows (calib_000 to calib_149), the same fields with dataset
                             "calibration" and the eight calibration symptoms
  symptoms.json              the battery of config/symptoms.yaml with each title as shown to the model
                             in the annotation job (config/symptoms_job_overrides.yaml)
  task_symptom.json          the prompt templates of config/task_symptom.yaml

Sample of posts.  One numpy RandomState(20261001) is used for the three datasets in turn (irf, lost,
severity).  Within a dataset, the posts are grouped by their combination of label values (irf: the four
combinations of tbe and pbu; lost: the two levels of low self-esteem; severity: the four levels), taken
in sorted order.  The posts of each combination, in the row order of validation_labels.parquet, are
permuted and the first 800 / c are kept, where c is the number of combinations.  Rank r is post r // c
of combination r mod c, so every prefix of the ranks is balanced; the presets of run_validation.py take
the items with rank below 5, 80, 300 (the protocol of the manuscript) or 800.  Each post is scored on the
symptoms mapped to its dataset's labels in config/validation_mapping.yaml (inverse items included),
followed by the control symptoms of the reduced protocol.

Calibration windows.  data/processed/calibration_window_map.parquet is the record of the 150 cohort
windows used in the manuscript (item_id -> speaker_id, window_start, window_index).  It identifies
accounts and is not distributed with the repository; the list is available from the authors.
  * If the record exists, calibration_windows.jsonl is rebuilt from it: one item per window in the
    order of item_id (which gives the rank), with the window's posts_text from
    data/processed/windows.parquet.  The procedure that drew this set cannot be regenerated.  It was
    not a draw from all windows: no window in it has more than 5,773 characters of post text (5,807
    characters of posts_text, which adds the quotes and blank lines), consistent with a cap that keeps
    every prompt within the 4,096-token context of a server slot.
  * Without the record, a new set is drawn and its record written: 150 windows without replacement,
    with the same RandomState continued after the posts, among the windows of accounts other than the
    bot (P.BOT_SPEAKERS) whose posts_text has at most 6,000 characters.  The calibration results of
    such a set differ from those reported in the manuscript.

Inputs:  config/validation_mapping.yaml, config/symptoms.yaml, config/symptoms_job_overrides.yaml,
         config/task_symptom.yaml
         data/processed/validation_labels.parquet            labels (step 4; --labels)
         annotation/requests/validation_<dataset>.parquet   post texts (step 4; --validation-dir)
         data/processed/windows.parquet                     cohort windows (step 3; --windows)
         data/processed/calibration_window_map.parquet      record of the calibration set, if present
Outputs: local_validation/requests/{posts.jsonl, calibration_windows.jsonl, symptoms.json,
         task_symptom.json} (--out-dir), and data/processed/calibration_window_map.parquet when a new
         calibration set is drawn

usage: python3 04b_prepare_local_validation.py       (after steps 3 and 4)
"""
import argparse
import json
from pathlib import Path

import _setup  # noqa: F401
import numpy as np
import pandas as pd
import yaml

from slm import params as P
from slm.prompts import load_symptoms, load_task

SEED = 20261001                     # seed of the sample (posts, then a new calibration set if needed)
DATASETS = ["irf", "lost", "severity"]   # the three datasets of the manuscript, in sampling order
POSTS_PER_DATASET = 800             # the largest preset of run_validation.py ("extended")
N_CALIBRATION = 150                 # calibration windows of the largest preset
MAX_CALIBRATION_CHARS = 6000        # new calibration sets only: longest posts_text (characters)
# the eight calibration symptoms, rated by the local model and compared with the cohort ratings
CALIBRATION_SYMPTOMS = ["sadness", "anxiety", "insomnia_early", "worthlessness", "si", "loneliness",
                        "interest_loss", "feeling_happy"]

ap = argparse.ArgumentParser(description="Request files of the reduced validation protocol.")
ap.add_argument("--labels", type=Path, default=P.PROCESSED / "validation_labels.parquet",
                help="labels of the validation posts (step 4)")
ap.add_argument("--validation-dir", type=Path, default=P.ROOT / "annotation" / "requests",
                help="folder with validation_<dataset>.parquet, the posts rendered by step 4")
ap.add_argument("--windows", type=Path, default=P.PROCESSED / "windows.parquet",
                help="cohort windows with their posts_text (step 3)")
ap.add_argument("--calibration-map", type=Path, default=P.PROCESSED / "calibration_window_map.parquet",
                help="record of the calibration windows: read if it exists, otherwise written for a new draw")
ap.add_argument("--out-dir", type=Path, default=P.ROOT / "local_validation" / "requests",
                help="folder of the request files")
args = ap.parse_args()


def write_jsonl(path, rows):
    """Write one JSON object per line (UTF-8, non-ASCII characters kept as they are, "\\n" endings)."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def item_symptoms(spec, controls):
    """Symptoms scored for every post of a dataset, in this order: for each label of the dataset (in
    the order of the mapping) its mapped symptoms and inverse symptoms, then the control symptoms;
    duplicates are dropped.  `spec` is the dataset's entry of the validation mapping."""
    syms = [c for lab in spec.values() for c in lab.get("symptoms", []) + lab.get("inverse_symptoms", [])]
    return list(dict.fromkeys(syms + list(controls)))


def balanced_sample(lab, label_cols, rs, n):
    """item_ids of n posts in rank order, balanced over the combinations of the label columns.

    lab: the dataset's rows of validation_labels.parquet (in file order); label_cols: the label
    columns; rs: the shared numpy RandomState; n: posts to keep.  The combinations (groups of
    DataFrame.groupby, in sorted order) are permuted one after the other with rs, the first n // c
    posts of each are kept, and rank r takes post r // c of combination r mod c.
    """
    groups = [g for _, g in lab.groupby(label_cols, sort=True)]
    c = len(groups)
    k = n // c                                      # posts per combination
    perms = []
    for g in groups:
        assert len(g) >= k, f"only {len(g)} posts with labels {g[label_cols].iloc[0].tolist()}"
        perms.append(g["item_id"].to_numpy()[rs.permutation(len(g))[:k]])
    return [perms[r % c][r // c] for r in range(c * k)]


def as_ns(s):
    """Timestamps as datetime64[ns], so that keys written with other resolutions compare equal."""
    return pd.to_datetime(s).astype("datetime64[ns]")


args.out_dir.mkdir(parents=True, exist_ok=True)
mapping = yaml.safe_load(open(P.CONFIG / "validation_mapping.yaml"))
controls = mapping["_reduced_protocol"]["controls"]
labels = pd.read_parquet(args.labels)
rs = np.random.RandomState(SEED)

# ---- step: validation posts ------------------------------------------------------------------------
rows = []
for ds in DATASETS:
    spec = mapping[ds]                              # {label: {symptoms, inverse_symptoms, ...}}
    label_cols = list(spec)                         # e.g. ["tbe", "pbu"] for irf
    ids = balanced_sample(labels[labels["dataset"] == ds], label_cols, rs, POSTS_PER_DATASET)
    texts = pd.read_parquet(args.validation_dir / f"validation_{ds}.parquet",
                            columns=["item_id", "posts_text"])
    text_of = dict(zip(texts["item_id"], texts["posts_text"]))
    syms = item_symptoms(spec, controls.get(ds, []))
    rows += [{"item_id": str(i), "dataset": ds, "rank": r, "posts_text": text_of[i], "symptoms": syms}
             for r, i in enumerate(ids)]
    lab_ds = labels.set_index("item_id").loc[ids, label_cols]
    print(f"{ds}: {len(ids)} posts, {len(syms)} symptoms per post; label combinations among the first 300:",
          lab_ds.iloc[:300].value_counts().sort_index().to_dict())
write_jsonl(args.out_dir / "posts.jsonl", rows)

# ---- step: calibration windows -----------------------------------------------------------------------
win = pd.read_parquet(args.windows, columns=["speaker_id", "window_start", "window_index", "posts_text"])
if args.calibration_map.exists():
    # the record of the manuscript's calibration set
    cmap = pd.read_parquet(args.calibration_map)
    source = f"record {args.calibration_map}"
else:
    # a new set: windows of non-bot accounts whose posts_text has at most MAX_CALIBRATION_CHARS
    # characters (in the row order of windows.parquet), drawn without replacement with the continued
    # RandomState
    cand = win[~win["speaker_id"].isin(P.BOT_SPEAKERS)
               & (win["posts_text"].str.len() <= MAX_CALIBRATION_CHARS)].reset_index(drop=True)
    pick = rs.permutation(len(cand))[:N_CALIBRATION]
    cmap = cand.iloc[pick][["speaker_id", "window_start", "window_index"]].reset_index(drop=True)
    cmap.insert(0, "item_id", [f"calib_{i:03d}" for i in range(len(cmap))])
    args.calibration_map.parent.mkdir(parents=True, exist_ok=True)
    cmap.to_parquet(args.calibration_map, index=False)
    source = f"new draw among {len(cand):,} windows, record written to {args.calibration_map}"
cmap = cmap.sort_values("item_id").reset_index(drop=True)          # rank = order of item_id
key = ["speaker_id", "window_start"]
cal = cmap.assign(window_start=as_ns(cmap["window_start"])).merge(
    win.assign(window_start=as_ns(win["window_start"]))[key + ["posts_text"]], on=key, how="left")
assert cal["posts_text"].notna().all(), "calibration windows missing from windows.parquet"
write_jsonl(args.out_dir / "calibration_windows.jsonl",
            [{"item_id": i, "dataset": "calibration", "rank": r, "posts_text": t,
              "symptoms": CALIBRATION_SYMPTOMS}
             for r, (i, t) in enumerate(zip(cal["item_id"], cal["posts_text"]))])
print(f"calibration: {len(cal)} windows ({source}); longest posts_text {cal['posts_text'].str.len().max():,} "
      "characters")

# ---- step: battery and prompt templates (JSON, indented by one space, no final newline) -------------
battery = [{k: (s["prompt_title"] if k == "title" else v) for k, v in s.items() if k != "prompt_title"}
           for s in load_symptoms()]                # title as shown to the model ('Indiciciveness')
with open(args.out_dir / "symptoms.json", "w", encoding="utf-8") as f:
    json.dump(battery, f, indent=1)
with open(args.out_dir / "task_symptom.json", "w", encoding="utf-8") as f:
    json.dump(load_task("symptom"), f, indent=1)
print(f"wrote {args.out_dir}/posts.jsonl ({len(rows):,} posts), calibration_windows.jsonl, symptoms.json, "
      "task_symptom.json")
