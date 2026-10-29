#!/usr/bin/env python3
"""
Step 13 - lexicon and style features, the comparators of the validation (Methods, "Validation
against human annotations"; Fig. 1d).  No language model is needed.

Features (slm.lexicon): VADER sentiment, the rates of the 194 Empath categories, and the rates of
first person singular pronouns (Rude et al., 2004) and of absolutist words (Al-Mosaiwi and
Johnstone, 2018).  They are computed for
  * every cohort window: VADER scores are computed per utterance and averaged over the utterances
    of the window; Empath and style rates are averaged with weights proportional to the number of
    tokens of each utterance, i.e. pooled over the tokens of the window;
  * every validation post (step 4), from the post text without the enclosing quotes.

Inputs
  data/processed/windows.parquet                    (step 3)
  data/interim/cohort_texts.parquet                 (step 2)
  annotation/requests/validation_{irf,lost,severity,depemo}.parquet   (step 4)
Outputs
  data/processed/lexicon_windows.parquet     one row per window: speaker_id, window_index,
                                             window_start, n_tokens, vader_*, empath_*, style_*
  data/processed/lexicon_validation.parquet  one row per validation post: item_id, dataset,
                                             vader_*, empath_*, n_tokens, style_*

Usage (from scripts/):
  python3 13_lexicon_features.py
"""
import argparse

import _setup  # noqa: F401
import pandas as pd

from slm import params as P
from slm.io import Timer
from slm.lexicon import empath_frame, style_frame, vader_frame

# --help prints this description; the step has no options
argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

# ---- step: features of every utterance of the cohort accounts ----
win = pd.read_parquet(P.PROCESSED / "windows.parquet",
                      columns=["speaker_id", "window_index", "window_start", "post_ids"])
utt = pd.read_parquet(P.INTERIM / "cohort_texts.parquet", columns=["id", "text"])
utt["text"] = utt["text"].fillna("")
with Timer(f"post-level features for {len(utt)} posts"):
    feats = pd.concat([utt[["id"]].reset_index(drop=True), vader_frame(utt["text"].tolist()),
                       empath_frame(utt["text"].tolist()), style_frame(utt["text"].tolist())], axis=1)
feats = feats.set_index("id")
emp_cols = [c for c in feats.columns if c.startswith("empath_")]

# ---- step: aggregate to windows ----
rows = []
with Timer("aggregating to windows"):
    for _, w in win.iterrows():
        f = feats.loc[list(w["post_ids"])]                       # utterances of the window
        n = f["n_tokens"].to_numpy()
        tot = max(n.sum(), 1.0)                                  # avoid 0/0 for windows without tokens
        r = {"speaker_id": w["speaker_id"], "window_index": w["window_index"],
             "window_start": w["window_start"], "n_tokens": float(n.sum())}
        for c in ["vader_compound", "vader_neg", "vader_pos", "vader_neu"]:
            r[c] = float(f[c].mean())                            # mean over utterances
        for c in emp_cols + ["style_first_person", "style_absolutist"]:
            r[c] = float((f[c].to_numpy() * n).sum() / tot)      # token-weighted rate
        rows.append(r)
lw = pd.DataFrame(rows)
lw.to_parquet(P.PROCESSED / "lexicon_windows.parquet", index=False)
print(lw.shape)

# ---- step: features of the validation posts ----
vals = []
for name in ["irf", "lost", "severity", "depemo"]:
    v = pd.read_parquet(P.ROOT / "annotation" / "requests" / f"validation_{name}.parquet")
    txt = v["posts_text"].str.slice(1, -1).tolist()      # strip the '...' quoting
    with Timer(f"validation {name}"):
        f = pd.concat([v[["item_id", "dataset"]].reset_index(drop=True), vader_frame(txt), empath_frame(txt),
                       style_frame(txt)], axis=1)
    vals.append(f)
pd.concat(vals, ignore_index=True).to_parquet(P.PROCESSED / "lexicon_validation.parquet", index=False)
print("done")
