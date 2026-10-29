#!/usr/bin/env python3
"""
Step 3 - two-week windows and the request set of the annotation job (Methods, "Cohort and
observation windows" and "Language model questionnaire"; Extended Data Fig. 1).

Builds the 27 windows of 14 days of each of the 492 accounts that meet the activity criteria
(cohort.csv, in_cohort == True).  The moderation bot is included because its windows were annotated
with the others; its ratings are removed when the outputs are parsed (step 5).  Each onset is
converted with slm.cohort.annotation_onset, window boundaries are Pacific wall-clock times
(slm.windows), and the accounts are processed in the sorted order of their names, which was the
order of the annotation job's account list.  The windows, and their row order, are therefore those
of the annotation job and match the stored model outputs.

Inputs
  data/processed/cohort.csv                 (step 2)
  data/interim/cohort_texts.parquet         (step 2)
  config/symptoms.yaml, config/symptoms_job_overrides.yaml, config/task_symptom.yaml
Outputs
  data/processed/windows.parquet            one row per window with utterances (9,913 rows; 9,886
                                            without the bot): speaker_id, window_index, window_start,
                                            n_posts (utterances), n_submissions (posts), n_chars,
                                            n_words, post_ids, posts_text
  annotation/requests/windows.parquet       columns needed by the GPU runner
                                            (annotation/run_vllm_annotation.py): speaker_id,
                                            window_index, window_start, n_posts, posts_text
  annotation/requests/symptoms.yaml         the battery, with each title as shown to the model
  annotation/requests/task_symptom.yaml     the prompt templates
  results/03_windows_summary.json           window and request counts and text volume per window,
                                            over all 492 accounts (the bot's 27 windows included)

Usage (from scripts/):
  python3 03_build_windows.py
"""
import argparse
import shutil

import _setup  # noqa: F401
import pandas as pd
import yaml

from slm import params as P
from slm.cohort import annotation_onset
from slm.io import Timer, save_json
from slm.prompts import load_symptoms
from slm.windows import build_windows, n_requests

# --help prints this description; the step has no options
argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

# ---- step: accounts and onsets as used by the annotation job ----
coh = pd.read_csv(P.PROCESSED / "cohort.csv")
c = coh[coh["in_cohort"]]                          # 492 accounts, bot included
acc = pd.DataFrame({"speaker_id": c["speaker_id"].to_numpy(),
                    "active_start": annotation_onset(c["active_start"].to_numpy())})
# sorted order of the account names (code-point order, as Python's sorted): the order of the
# annotation job's account list, which determines the row order of windows.parquet
acc = acc.sort_values("speaker_id", kind="stable").reset_index(drop=True)

# ---- step: windows ----
utts = pd.read_parquet(P.INTERIM / "cohort_texts.parquet")
with Timer("building windows"):
    win = build_windows(acc, utts)

win.to_parquet(P.PROCESSED / "windows.parquet", index=False, compression="zstd")

# ---- step: request files for the GPU runner ----
req_dir = P.ROOT / "annotation" / "requests"
req_dir.mkdir(parents=True, exist_ok=True)
win[["speaker_id", "window_index", "window_start", "n_posts", "posts_text"]].to_parquet(
    req_dir / "windows.parquet", index=False, compression="zstd")
# The runner shows each symptom's `title` to the model; use the title as shown in the annotation
# job (prompt_title: 'Indiciciveness' for item 17), with the other fields as in config/symptoms.yaml.
battery = [{k: (s["prompt_title"] if k == "title" else v) for k, v in s.items() if k != "prompt_title"}
           for s in load_symptoms()]
with open(req_dir / "symptoms.yaml", "w") as f:
    yaml.safe_dump(battery, f, sort_keys=False)
shutil.copy(P.CONFIG / "task_symptom.yaml", req_dir / "task_symptom.yaml")

# ---- step: summary (all 492 accounts, i.e. every request submitted to the model) ----
nreq = n_requests(win)                             # windows x 52 symptoms x 5 samples
per_spk = win.groupby("speaker_id").size()         # windows with utterances per account
summ = {
    "n_speakers": int(win["speaker_id"].nunique()),
    "n_windows_nonempty": int(len(win)),
    "n_requests": nreq,
    "n_requests_submitted": P.EXPECTED_N_REQUESTS,           # requests submitted to the model
    "n_requests_matches_submitted": nreq == P.EXPECTED_N_REQUESTS,
    "windows_per_speaker_median": float(per_spk.median()),
    "windows_per_speaker_min_max": [int(per_spk.min()), int(per_spk.max())],
    "posts_per_window_median": float(win["n_posts"].median()),            # utterances per window
    "posts_per_window_iqr": [float(win["n_posts"].quantile(q)) for q in (0.25, 0.75)],
    "words_per_window_median": float(win["n_words"].median()),
    "words_per_window_p95": float(win["n_words"].quantile(0.95)),
    "words_per_window_max": int(win["n_words"].max()),
    "chars_total": int(win["n_chars"].sum()),
}
save_json(summ, P.RESULTS / "03_windows_summary.json")
print(summ)
