#!/usr/bin/env python3
"""
Step 2 - cohort selection (Methods, "Cohort and observation windows"; Extended Data Fig. 1).

For every account with at least 20 utterances, the daily posting rate (a sum of Epanechnikov
kernels extending 28 days to either side of each utterance) is evaluated on a daily grid.  Accounts
with at least 360 active days (rate >= 0.2 utterances per day, not necessarily contiguous) meet the
activity criteria; their onset is the first active day (slm.cohort).  Of the 492 accounts that
meet the criteria, one (AutoModerator, a moderation bot) is not a user, which leaves 491 users.
A second pass over the corpora then extracts the full records, text included, of the 492 accounts.
The bot is kept in this table because its windows were annotated with the others (step 3); its
ratings are removed in step 5.

Inputs
  data/interim/utterance_index.parquet     (step 1)
  data/raw/depression.corpus.zip, data/raw/Anxiety.corpus.zip
Outputs
  data/processed/cohort.csv                one row per account with >= 20 utterances (33,520 rows):
                                           speaker_id, n_posts (utterances), active_start (onset,
                                           epoch s), active_end, n_active_days, active_time_s,
                                           in_cohort, active_duration_days, is_bot
  data/interim/cohort_texts.parquet        every utterance of the 492 accounts with in_cohort, in
                                           corpus file order: id, speaker_id, conversation_id,
                                           reply_to, timestamp, text, score, subreddit, corpus,
                                           corpus_rank, line_no (read by steps 3, 13, 15 and 12c)
  results/02_cohort_summary.json           counts of the selection and a description of the users

Usage (from scripts/):
  python3 02_select_cohort.py
"""
import argparse

import _setup  # noqa: F401
import pandas as pd

from slm import params as P
from slm.cohort import select_cohort
from slm.corpus import corpus_zip_paths, extract_texts, speaker_timestamps
from slm.io import Timer, save_json

# --help prints this description; the step has no options
argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

# ---- step: timestamps of the accounts with >= 20 utterances ----
index = pd.read_parquet(P.INTERIM / "utterance_index.parquet",
                        columns=["speaker_id", "timestamp", "corpus", "line_no"])
# first and last timestamp of the whole corpus: bounds of the daily grid shared by all accounts
t_min, t_max = float(index["timestamp"].min()), float(index["timestamp"].max())
with Timer("grouping timestamps"):
    counts = index.groupby("speaker_id").size()                    # utterances per account name
    keep = set(counts[counts >= P.MIN_POSTS].index) - set(P.EXCLUDE_SPEAKERS)
    sts = speaker_timestamps(index[index["speaker_id"].isin(keep)])  # timestamps in file order
print(f"{len(counts)} speakers; {len(sts)} with >= {P.MIN_POSTS} utterances (excl. [deleted])")

# ---- step: posting rates, active days and onsets ----
with Timer("rate functions"):
    coh = select_cohort(sts, t_min, t_max)
coh["is_bot"] = coh["speaker_id"].isin(P.BOT_SPEAKERS)
P.PROCESSED.mkdir(parents=True, exist_ok=True)
coh.to_csv(P.PROCESSED / "cohort.csv", index=False)

# ---- step: summary of the selection ----
c = coh[coh["in_cohort"]]                  # accounts that meet the activity criteria (492)
users = c[~c["is_bot"]]                    # the users of the manuscript (491)
summ = {
    "n_speakers_total": int(len(counts)),                 # account names, "[deleted]" included
    "n_speakers_ge20": int(len(sts)),                     # >= 20 utterances, "[deleted]" excluded
    "t_min": t_min, "t_max": t_max,                       # epoch seconds (UTC)
    "n_meeting_activity_criteria": int(len(c)),
    "bots_meeting_activity_criteria": sorted(c.loc[c["is_bot"], "speaker_id"]),
    "n_users": int(len(users)),
    # posting and onset of the users: utterances in the whole corpus, active days, onset dates
    "users_n_posts_median": float(users["n_posts"].median()),
    "users_n_posts_iqr": [float(users["n_posts"].quantile(q)) for q in (0.25, 0.75)],
    "users_active_days_median": float(users["n_active_days"].median()),
    "users_onset_range_utc": [str(pd.to_datetime(users["active_start"].min(), unit="s")),
                              str(pd.to_datetime(users["active_start"].max(), unit="s"))],
}
del index, sts                             # free memory before the second pass

# ---- step: texts of the accounts that meet the activity criteria (second pass) ----
with Timer("extracting cohort texts"):
    texts = extract_texts(corpus_zip_paths(), set(c["speaker_id"]))
P.INTERIM.mkdir(parents=True, exist_ok=True)
texts.to_parquet(P.INTERIM / "cohort_texts.parquet", index=False, compression="zstd")
summ["n_cohort_text_rows"] = int(len(texts))             # utterances of the 492 accounts

save_json(summ, P.RESULTS / "02_cohort_summary.json")
print(summ)
