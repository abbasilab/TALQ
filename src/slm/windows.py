"""
Two-week observation windows and the text shown to the model (Methods, "Cohort and observation
windows" and "Language model questionnaire").

For each account (scripts/03_build_windows.py passes the 492 accounts that meet the activity
criteria, with onsets converted by slm.cohort.annotation_onset):

    onset_local    = onset as a naive wall-clock time in Pacific time (America/Los_Angeles)
    window starts  = onset_local - 1 day + k x 14 days, k = 0, ..., 26  (27 windows; the last one
                     starts 363 days after onset), i.e. np.arange(onset_local - 1 day,
                     onset_local + 365 days, 14 days)
    utterance time = naive wall-clock time in Pacific time
    window k       = the utterances with  start_k <= time < start_k + 14 days, in corpus file order

Windows without utterances are not scored and are not returned (9,913 windows with utterances for
the 492 accounts, 9,886 for the 491 users).  The text of a window (`posts_text`) lists its
utterances, each enclosed in single quotes and separated by a blank line; the system message of the
prompt contains it verbatim (slm.prompts).

Time zone and daylight saving time: all comparisons use naive Pacific wall-clock times, so a window
boundary stays at the same local clock time all year and moves by one hour in UTC when daylight
saving time starts or ends.  The onset convention of the annotation job is described in
slm.cohort.annotation_onset.  Windows are built exactly as for the annotation job, so that they
match the stored model outputs (2,577,380 requests = 9,913 windows x 52 symptoms x 5 samples).
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from . import params as P


def to_local_naive(ts, tz: str = P.WINDOW_TZ) -> pd.Series:
    """
    Convert epoch seconds to naive wall-clock datetimes in `tz`.

    The result equals what datetime.fromtimestamp returns on a computer set to that time zone.

    Args:
        ts: epoch seconds (array-like of integers).
        tz: IANA time zone name (America/Los_Angeles).

    Returns:
        pandas Series of naive datetime64 values (no time zone attached).
    """
    s = pd.to_datetime(pd.Series(ts, dtype="int64"), unit="s", utc=True)
    return s.dt.tz_convert(tz).dt.tz_localize(None)


def window_starts_for(active_start_epoch: float, tz: str = P.WINDOW_TZ) -> list[pd.Timestamp]:
    """
    Start times of the 27 windows of one account.

    Args:
        active_start_epoch: onset in epoch seconds, as returned by slm.cohort.annotation_onset.
        tz: time zone of the wall-clock times.

    Returns:
        list of naive pandas Timestamps (Pacific wall-clock time): onset - 1 day + k x 14 days for
        every k with start < onset + 365 days (k = 0, ..., 26).
    """
    start = to_local_naive([int(active_start_epoch)], tz).iloc[0]
    # add back the fractional second of the onset (zero on the daily grid of slm.cohort)
    start = start + pd.Timedelta(seconds=float(active_start_epoch) - int(active_start_epoch))
    d = timedelta(days=P.ACTIVE_DELAY_DAYS)       # the first window starts 1 day before onset
    dur = timedelta(days=P.ACTIVE_DURATION_DAYS)  # starts are generated up to 365 days after onset
    w = timedelta(days=P.WINDOW_DAYS)             # window length and spacing: 14 days
    out, t = [], start.to_pydatetime() - d
    stop = start.to_pydatetime() + dur
    while t < stop:                          # == np.arange(start - d, start + dur, w)
        out.append(pd.Timestamp(t))
        t = t + w
    return out


def posts_text(texts: list[str]) -> str:
    """
    Text of a window as inserted into the prompt.

    Each utterance is enclosed in single quotes and the utterances are separated by a blank line:
    `'\\n\\n'.join([f"'{t}'" for t in texts])`.  Step 4 renders a single validation post the same
    way, as a window with one utterance.

    Args:
        texts: utterance texts in corpus file order.

    Returns:
        the joined string.
    """
    return "\n\n".join([f"'{t}'" for t in texts])


def build_windows(cohort: pd.DataFrame, utts: pd.DataFrame, tz: str = P.WINDOW_TZ) -> pd.DataFrame:
    """
    One row per (account, window) with at least one utterance.

    Args:
        cohort: columns speaker_id and active_start (onset from slm.cohort.annotation_onset, epoch
            seconds).  Rows are processed in the given order, which becomes the row order of the
            result.
        utts: utterances of these accounts with the columns speaker_id, timestamp, text, id,
            reply_to, corpus_rank and line_no (data/interim/cohort_texts.parquet).
        tz: time zone of the window boundaries.

    Returns:
        DataFrame (data/processed/windows.parquet) with the columns
          speaker_id;
          window_index: position k (0-26) of the window in the account's 27-window grid;
          window_start: start of the window, naive Pacific wall-clock time;
          n_posts: number of utterances (posts and comments) in the window;
          n_submissions: number of posts (utterances without reply_to) in the window;
          n_chars: total number of characters of the utterance texts;
          n_words: total number of whitespace-separated words;
          post_ids: ids of the utterances, in corpus file order;
          posts_text: text shown to the model (see posts_text).
    """
    utts = utts.sort_values(["corpus_rank", "line_no"], kind="stable").copy()   # corpus file order
    utts["t_local"] = to_local_naive(utts["timestamp"].to_numpy(), tz).to_numpy()
    rows = []
    wdelta = pd.Timedelta(days=P.WINDOW_DAYS)
    by_speaker = {s: g for s, g in utts.groupby("speaker_id", sort=False)}
    for _, c in cohort.iterrows():
        g = by_speaker.get(c["speaker_id"])
        if g is None:
            continue
        tl = g["t_local"].to_numpy()
        for i_w, ws in enumerate(window_starts_for(c["active_start"], tz)):
            m = (tl >= np.datetime64(ws)) & (tl < np.datetime64(ws + wdelta))   # start <= t < end
            if not m.any():
                continue                          # windows without utterances are not scored
            gw = g[m]
            texts = gw["text"].fillna("").tolist()
            rows.append({
                "speaker_id": c["speaker_id"],
                "window_index": i_w,
                "window_start": ws,
                "n_posts": int(m.sum()),
                "n_submissions": int(gw["reply_to"].isna().sum()) if "reply_to" in gw else np.nan,
                "n_chars": int(sum(len(t) for t in texts)),
                "n_words": int(sum(len(t.split()) for t in texts)),
                "post_ids": list(gw["id"]),
                "posts_text": posts_text(texts),
            })
    return pd.DataFrame(rows)


def n_requests(windows: pd.DataFrame, n_symptoms: int = 52, n_replicates: int = P.N_REPLICATES) -> int:
    """
    Number of model requests for a set of windows: windows x symptoms x samples.

    Args:
        windows: one row per window with utterances.
        n_symptoms: items of the battery (52).
        n_replicates: samples per window and symptom (5).

    Returns:
        int (2,577,380 for the 9,913 windows of the 492 accounts).
    """
    return int(len(windows) * n_symptoms * n_replicates)
