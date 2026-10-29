#!/usr/bin/env python3
"""
Step 15 - description of the cohort (Methods, "Cohort and observation windows"; Extended Data
Fig. 1a,d).  No language model is needed.

Describes the 491 users, i.e. the accounts that meet the activity criteria without the moderation
bot (P.BOT_SPEAKERS): onset years, posting volume, the share of posts among their utterances, the
share of r/depression, and windows per user.  The active days of all 33,520 accounts with at
least 20 utterances are summarized for comparison.

Inputs
  data/processed/cohort.csv            (step 2)
  data/interim/cohort_texts.parquet    (step 2)
  data/processed/windows.parquet       (step 3)
Outputs
  data/processed/user_description.parquet   one row per user: speaker_id, n (utterances in the
                                            corpus), frac_posts, frac_depression, n_windows,
                                            utt_in_windows, posts_in_windows, frac_posts_in_windows
  results/15_cohort.json                    summary numbers (onset_year_counts: Extended Data Fig. 1d)

Usage (from scripts/):
  python3 15_cohort_description.py
"""
import argparse

import _setup  # noqa: F401
import pandas as pd

from slm import params as P
from slm.io import save_json

# --help prints this description; the step has no options
argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

# ---- step: the 491 users and their utterances and windows ----
coh = pd.read_csv(P.PROCESSED / "cohort.csv")            # all accounts with >= 20 utterances
c = coh[coh["in_cohort"] & ~coh["is_bot"]].copy()        # meet the activity criteria, bot excluded
users = set(c["speaker_id"])
t = pd.read_parquet(P.INTERIM / "cohort_texts.parquet", columns=["speaker_id", "reply_to", "subreddit"])
t = t[t["speaker_id"].isin(users)].copy()
w = pd.read_parquet(P.PROCESSED / "windows.parquet",
                    columns=["speaker_id", "window_index", "n_posts", "n_words", "n_submissions"])
w = w[w["speaker_id"].isin(users)]

# ---- step: per-user description ----
t["is_post"] = t["reply_to"].isna()                      # posts have no parent; comments do
per = t.groupby("speaker_id").agg(n=("is_post", "size"), frac_posts=("is_post", "mean"),
                                  frac_depression=("subreddit", lambda s: (s == "depression").mean()))
# windows with utterances, and the utterances and posts inside them (n_posts counts utterances)
ww = w.groupby("speaker_id").agg(n_windows=("window_index", "size"), utt_in_windows=("n_posts", "sum"),
                                 posts_in_windows=("n_submissions", "sum"))
per = per.join(ww)
per["frac_posts_in_windows"] = per["posts_in_windows"] / per["utt_in_windows"]
c["onset_year"] = pd.to_datetime(c["active_start"], unit="s").dt.year     # calendar year (UTC)


def med_iqr(x):
    """[median, 25th percentile, 75th percentile] of a Series, as floats."""
    return [float(x.median()), float(x.quantile(.25)), float(x.quantile(.75))]


# ---- step: summary numbers ----
res = {
    "n_cohort": int(len(c)),                                               # users (491)
    "n_bots_excluded": int((coh["in_cohort"] & coh["is_bot"]).sum()),
    "onset_year_counts": c["onset_year"].value_counts().sort_index().to_dict(),
    "utterances_per_user_median_iqr": med_iqr(c["n_posts"]),
    "frac_posts_per_user_median_iqr": med_iqr(per["frac_posts"]),
    "frac_posts_overall": float(t["is_post"].mean()),
    "n_users_without_any_post_in_windows": int((per["posts_in_windows"] == 0).sum()),
    "frac_depression_overall": float((t["subreddit"] == "depression").mean()),
    "users_only_depression": int((per["frac_depression"] == 1).sum()),
    "users_only_anxiety": int((per["frac_depression"] == 0).sum()),
    "windows_per_user_median_iqr": med_iqr(per["n_windows"]),
    "active_days_median": float(c["n_active_days"].median()),
    "ge20_accounts_active_days_median": float(coh["n_active_days"].median()),   # all 33,520 accounts
}
per.reset_index().to_parquet(P.PROCESSED / "user_description.parquet", index=False)
save_json(res, P.RESULTS / "15_cohort.json")
print(res)
