#!/usr/bin/env python3
"""
Step 1 - index every utterance of the two ConvoKit corpora (Methods, "Data source").

Streams data/raw/depression.corpus.zip and data/raw/Anxiety.corpus.zip (in this order, which
defines the corpus file order used by later steps) and writes an index of all utterances without
their text.  The index is the input of the cohort selection (step 2).

Inputs
  data/raw/depression.corpus.zip, data/raw/Anxiety.corpus.zip   (data/raw/download_convokit.py)
Outputs
  data/interim/utterance_index.parquet   one row per utterance (4,547,682 rows): id, speaker_id,
                                         conversation_id, reply_to, timestamp, score, subreddit,
                                         n_chars, corpus (0 = r/depression, 1 = r/Anxiety), line_no
  results/01_corpus_summary.json         counts reported in the Methods (4,547,682 utterances from
                                         549,783 account names, "[deleted]" included)

Usage (from scripts/):
  python3 01_ingest_corpus.py
"""
import argparse

import _setup  # noqa: F401

from slm import params as P
from slm.corpus import build_index, corpus_zip_paths
from slm.io import Timer, save_json

# --help prints this description; the step has no options
argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

# ---- step: stream both corpora into the utterance index ----
zips = corpus_zip_paths()                        # r/depression first, then r/Anxiety
with Timer("streaming corpora"):
    index = build_index(zips)

P.INTERIM.mkdir(parents=True, exist_ok=True)
index.to_parquet(P.INTERIM / "utterance_index.parquet", index=False, compression="zstd")

# ---- step: corpus summary ----
summ = {
    "n_utterances": int(len(index)),                                      # posts and comments
    "n_by_corpus": index.groupby("corpus").size().rename(dict(enumerate(P.SUBREDDITS))).to_dict(),
    "n_posts (reply_to null)": int(index["reply_to"].isna().sum()),       # posts (not comments)
    "n_speakers": int(index["speaker_id"].nunique()),                     # account names
    "n_deleted_utterances": int((index["speaker_id"] == "[deleted]").sum()),
    "t_min": int(index["timestamp"].min()), "t_max": int(index["timestamp"].max()),   # epoch s, UTC
}
save_json(summ, P.RESULTS / "01_corpus_summary.json")
print(summ)
