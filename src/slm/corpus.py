"""
Reading the ConvoKit subreddit corpora directly from their zip archives (Methods, "Data source").

The corpora of r/depression and r/Anxiety (data/raw/<subreddit>.corpus.zip, downloaded and
checksummed by data/raw/download_convokit.py) contain every post and comment ("utterances") of the
two communities up to 31 October 2018: 4,547,682 utterances from 549,783 account names, including
the placeholder "[deleted]".  Each line of the archive member `utterances.jsonl` is one utterance.
This module streams those lines, without loading a corpus into memory, and returns the fields the
analysis uses:

    id, speaker_id, conversation_id, reply_to, timestamp, text, score, subreddit, corpus

`reply_to` is empty (None) for posts and holds the parent id for comments; `timestamp` is in
seconds since 1970-01-01 UTC.

File order.  Every utterance also carries its position in the corpus files: the rank of its archive
in corpus_zip_paths() (0 = r/depression, 1 = r/Anxiety) and its line number in utterances.jsonl.
This order is used twice: the posting-rate kernels of an account are added in file order
(slm.cohort), and the utterances of a window are listed in file order in the text shown to the
model (slm.windows), as in the annotation job, so that the windows match the stored model outputs.

Used by scripts/01_ingest_corpus.py (utterance index) and scripts/02_select_cohort.py (timestamps
for the posting rates; texts of the accounts that meet the activity criteria).
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from . import params as P

# Columns of the utterance index (data/interim/utterance_index.parquet).  Unlike the text records,
# the index stores the archive rank in `corpus` (0, 1) and the text length instead of the text.
INDEX_COLUMNS = ["id", "speaker_id", "conversation_id", "reply_to", "timestamp",
                 "score", "subreddit", "n_chars", "corpus", "line_no"]


def corpus_zip_paths() -> list[Path]:
    """
    Paths of the corpus archives, in the order that defines the archive rank.

    Returns:
        [data/raw/depression.corpus.zip, data/raw/Anxiety.corpus.zip] (the order of P.SUBREDDITS).
    """
    return [P.RAW / f"{name}.corpus.zip" for name in P.SUBREDDITS]


def _utterance_member(z: zipfile.ZipFile) -> str:
    """Name of the `utterances.jsonl` member of an opened corpus archive (the first match)."""
    cands = [n for n in z.namelist() if n.endswith("utterances.jsonl")]
    if not cands:
        raise FileNotFoundError("no utterances.jsonl inside zip")
    return cands[0]


def iter_utterances(zip_path: str | Path) -> Iterator[tuple[int, dict]]:
    """
    Stream the utterances of one corpus archive.

    Args:
        zip_path: path of a <name>.corpus.zip archive.

    Yields:
        (line_no, record) for every non-empty line of utterances.jsonl.  line_no counts the lines
        of the file from 0 (blank lines are skipped but counted).  record is a dict with the keys
        id, speaker_id, conversation_id, reply_to, timestamp, text, score, subreddit and corpus
        (the archive's base name, "depression" or "Anxiety").  ConvoKit releases before 2.3 call
        the speaker and conversation fields "user" and "root"; both spellings are accepted.
        `subreddit` falls back to the corpus name when the utterance metadata lack it.
    """
    zip_path = Path(zip_path)
    corpus = zip_path.name.split(".")[0]                 # "depression.corpus.zip" -> "depression"
    with zipfile.ZipFile(zip_path) as z:
        with z.open(_utterance_member(z)) as fb:
            for line_no, raw in enumerate(io.TextIOWrapper(fb, encoding="utf-8")):
                if not raw.strip():
                    continue
                u = json.loads(raw)
                meta = u.get("meta") or {}
                rec = {
                    "id": u.get("id"),
                    # ConvoKit <2.3 corpora use "user" / "root"; >=2.3 "speaker" / "conversation_id"
                    "speaker_id": u.get("speaker", u.get("user")),
                    "conversation_id": u.get("conversation_id", u.get("root")),
                    "reply_to": u.get("reply_to"),
                    "timestamp": u.get("timestamp"),
                    "text": u.get("text", ""),
                    "score": meta.get("score"),
                    "subreddit": meta.get("subreddit", corpus),
                    "corpus": corpus,
                }
                yield line_no, rec


def build_index(zip_paths: list[str | Path], chunk: int = 500_000) -> pd.DataFrame:
    """
    Stream every corpus once and return an index of all utterances, without their text.

    Args:
        zip_paths: archives in corpus order (corpus_zip_paths()); the position of an archive in
            this list is stored in the `corpus` column.
        chunk: number of rows buffered before they are converted to a DataFrame (limits memory).

    Returns:
        DataFrame with INDEX_COLUMNS, one row per utterance, in file order (4,547,682 rows for the
        two corpora): id, speaker_id, conversation_id, reply_to (None for posts), timestamp (int64
        seconds, UTC), score, subreddit, n_chars (length of the text in characters), corpus
        (archive rank) and line_no (line in utterances.jsonl).
    """
    parts, buf = [], {c: [] for c in INDEX_COLUMNS}
    for rank, zp in enumerate(zip_paths):
        for line_no, r in iter_utterances(zp):
            buf["id"].append(r["id"]); buf["speaker_id"].append(r["speaker_id"])
            buf["conversation_id"].append(r["conversation_id"]); buf["reply_to"].append(r["reply_to"])
            buf["timestamp"].append(r["timestamp"]); buf["score"].append(r["score"])
            buf["subreddit"].append(r["subreddit"]); buf["n_chars"].append(len(r["text"] or ""))
            buf["corpus"].append(rank); buf["line_no"].append(line_no)
            if len(buf["id"]) >= chunk:
                parts.append(pd.DataFrame(buf)); buf = {c: [] for c in INDEX_COLUMNS}
    if buf["id"]:
        parts.append(pd.DataFrame(buf))
    index = pd.concat(parts, ignore_index=True)
    index["timestamp"] = index["timestamp"].astype("int64")
    return index


def extract_texts(zip_paths: list[str | Path], speakers: set[str]) -> pd.DataFrame:
    """
    Second pass over the corpora: full records, text included, of the utterances of `speakers`.

    Args:
        zip_paths: archives in corpus order (corpus_zip_paths()).
        speakers: set of account names whose utterances are kept.

    Returns:
        DataFrame in file order with the columns id, speaker_id, conversation_id, reply_to,
        timestamp (int64 seconds, UTC), text, score, subreddit, corpus (archive name), corpus_rank
        (archive rank) and line_no.  Step 2 writes it, for the 492 accounts that meet the activity
        criteria, to data/interim/cohort_texts.parquet.
    """
    rows = []
    for rank, zp in enumerate(zip_paths):
        for line_no, r in iter_utterances(zp):
            if r["speaker_id"] in speakers:
                r = dict(r); r["corpus_rank"] = rank; r["line_no"] = line_no
                rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df["timestamp"] = df["timestamp"].astype("int64")
    return df


def speaker_timestamps(index: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Timestamps of each account's utterances, in file order.

    Args:
        index: utterance index, or a subset of its rows, with the columns speaker_id, timestamp,
            corpus and line_no.

    Returns:
        dict speaker_id -> int64 array of timestamps (seconds, UTC), ordered by (corpus, line_no).
        Accounts appear in the order of their first utterance in file order.  The order of the
        timestamps determines the order in which slm.cohort.rate_epanechnikov adds kernel values.
    """
    idx = index.sort_values(["corpus", "line_no"], kind="stable")
    out: dict[str, np.ndarray] = {}
    for s, g in idx.groupby("speaker_id", sort=False):   # sort=False: order of first appearance
        out[s] = g["timestamp"].to_numpy()
    return out
