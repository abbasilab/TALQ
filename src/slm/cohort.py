"""
Cohort selection: posting rate, activity criteria and onset (Methods, "Cohort and observation
windows"; Extended Data Fig. 1).

Procedure (run by scripts/02_select_cohort.py):

  * all utterances (posts and comments) of r/depression and r/Anxiety;
  * accounts with at least 20 utterances (P.MIN_POSTS), excluding the placeholder "[deleted]"
    (33,520 accounts);
  * the daily posting rate of an account is the sum of Epanechnikov kernels centered on its
    utterances, each extending 28 days to either side (P.RATE_KERNEL_WIDTH_DAYS):
        rate(t) = sum_i k(t - t_i),   k(x) = (1/s) * 3/4 * (1 - (x/s)^2) for |x| <= s, else 0,
    with s = 28 days in seconds, so the rate is in utterances per second (x 86,400 per day);
  * the rate is evaluated on one daily grid shared by all accounts, t_min + j days (j = 0, 1, ...,
    up to t_max), where t_min and t_max are the first and last timestamps of the whole corpus; for
    these corpora the grid points fall at 07:20:35 UTC;
  * a day (grid point) is active when the rate is at least 0.2 utterances per day
    (P.ACTIVE_THRESH_PER_DAY);
  * accounts with at least 360 active days in total (P.MIN_ACTIVE_DAYS), which need not be
    contiguous, meet the activity criteria (column `in_cohort`; 492 accounts).  One of them,
    AutoModerator, is a moderation bot (P.BOT_SPEAKERS).  Its windows were annotated with the
    others, and it is removed when the ratings are parsed (step 5) and from the cohort description
    (step 15), which leaves the 491 users of the manuscript;
  * onset (`active_start`) is the first active day (a grid point, in epoch seconds).

Numerical note: each kernel is evaluated only on the grid points around its support, and the
contributions are added in file order (slm.corpus.speaker_timestamps).  Changing either can alter
the last bits of the rate and thereby the classification of a day whose rate lies within rounding
error of the threshold, so neither should be changed.

annotation_onset() converts onsets into the form used by the annotation job to build the windows
(see its docstring and slm.windows); scripts/03_build_windows.py applies it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import params as P

DAY = 86400.0                                     # seconds per day


def eval_grid(t_min: float, t_max: float, step_days: float = P.RATE_EVAL_STEP_DAYS) -> np.ndarray:
    """
    Evaluation grid of the posting rate: t_min, t_min + 1 day, ... (strictly below t_max).

    Args:
        t_min, t_max: first and last timestamp of the whole corpus (epoch seconds).
        step_days: grid spacing in days (1).

    Returns:
        float array of grid points in epoch seconds (one per day).
    """
    return np.arange(t_min, t_max, step_days * DAY)


def rate_epanechnikov(ts: np.ndarray, t_eval: np.ndarray,
                      width_days: float = P.RATE_KERNEL_WIDTH_DAYS) -> np.ndarray:
    """
    Posting rate of one account at the grid points `t_eval`, in utterances per second.

    Each utterance at time t adds k(t_eval - t) = (1/s) * 3/4 * (1 - u^2) * [|u| <= 1], with
    u = (t_eval - t) / s and s = width_days * 86,400 s.  The kernel integrates to 1, so the sum is a
    rate in events per second.  For speed, each kernel is evaluated only on the slice [lo, hi) of
    the grid that covers its support plus a margin of one to two grid points; grid points of the
    slice outside the support receive exactly 0 from the indicator.

    Args:
        ts: utterance timestamps of the account (epoch seconds) in file order; NaN entries are
            skipped.  Contributions are added in this order.
        t_eval: evaluation grid (epoch seconds, equally spaced; see eval_grid).
        width_days: kernel half-width in days (28).

    Returns:
        float array with the shape of t_eval.
    """
    scale = width_days * DAY                      # kernel half-width in seconds
    t0 = t_eval[0]
    dt = t_eval[1] - t_eval[0] if len(t_eval) > 1 else DAY
    out = np.zeros(t_eval.shape)
    n = len(t_eval)
    for t in ts:
        if np.isnan(t):
            continue
        # grid indices whose points may lie within [t - scale, t + scale], with a safety margin
        lo = max(int(np.floor((t - scale - t0) / dt)) - 1, 0)
        hi = min(int(np.ceil((t + scale - t0) / dt)) + 2, n)
        if lo >= hi:
            continue
        x = t_eval[lo:hi] - t
        u = x / scale
        k = (1 / scale) * ((np.abs(u) <= 1.0) * (3 / 4.0) * (1.0 - np.power(u, 2.0)))
        out[lo:hi] = out[lo:hi] + k
    return out


def select_cohort(speaker_ts: dict[str, np.ndarray], t_min: float, t_max: float,
                  min_posts: int = P.MIN_POSTS,
                  exclude: list[str] = P.EXCLUDE_SPEAKERS,
                  active_thresh_per_day: float = P.ACTIVE_THRESH_PER_DAY,
                  min_active_days: float = P.MIN_ACTIVE_DAYS) -> pd.DataFrame:
    """
    Apply the activity criteria to every account with at least `min_posts` utterances.

    Args:
        speaker_ts: dict account -> utterance timestamps in file order
            (slm.corpus.speaker_timestamps).
        t_min, t_max: first and last timestamp of the whole corpus (epoch seconds); they define
            the daily grid shared by all accounts.
        min_posts: minimum number of utterances (20).
        exclude: account names never considered (["[deleted]"]).
        active_thresh_per_day: rate threshold of an active day (0.2 utterances per day).
        min_active_days: minimum number of active days (360).

    Returns:
        DataFrame with one row per account with >= min_posts utterances (33,520 rows), in the order
        of `speaker_ts`, and the columns
          speaker_id;
          n_posts: number of utterances (posts and comments) in the corpus;
          active_start: onset, the first active grid point (epoch seconds; NaN without active days);
          active_end: last active grid point + 1 day (epoch seconds);
          n_active_days: number of active days;
          active_time_s: n_active_days in seconds;
          in_cohort: True if the account meets the activity criteria (>= min_active_days);
          active_duration_days: (active_end - active_start) in days, inactive gaps included.
    """
    t_eval = eval_grid(t_min, t_max)
    thresh = active_thresh_per_day / DAY          # threshold in utterances per second
    rows = []
    for s, ts in speaker_ts.items():
        if s in exclude or len(ts) < min_posts:
            continue
        r = rate_epanechnikov(np.asarray(ts, dtype=float), t_eval)
        idx = np.sort(np.where(r >= thresh)[0])   # grid indices of the active days (ascending)
        row = {"speaker_id": s, "n_posts": len(ts)}
        if len(idx) == 0:
            row.update(active_start=np.nan, active_end=np.nan, n_active_days=0)
        else:
            row.update(active_start=float(t_eval[idx[0]]),
                       active_end=float(t_eval[idx[-1]] + DAY),
                       n_active_days=int(len(idx)))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["active_time_s"] = DAY * df["n_active_days"]
    df["in_cohort"] = df["active_time_s"] >= DAY * min_active_days
    df["active_duration_days"] = (df["active_end"] - df["active_start"]) / DAY
    return df


def annotation_onset(t_epoch, tz: str = P.WINDOW_TZ) -> np.ndarray:
    """
    Onsets in the form used by the annotation job to build the windows.

    The annotation job stored each onset as a naive wall-clock time in Pacific time
    (America/Los_Angeles, P.WINDOW_TZ) and read that naive time back as if it were UTC.  The stored
    value is therefore t + utcoffset(t), 7 h (daylight time) or 8 h (standard time) before the kernel
    onset t.  The windows are built from this value in Pacific wall-clock time (slm.windows), so in
    absolute time the first window begins one day plus 7-8 h (6-9 h around changes of daylight
    saving time) before the kernel onset, rather than exactly one day before it.  Written as naive
    Pacific times, the window starts (`window_start`) lie one day plus 14-16 h before the kernel
    onset written in UTC.  Example: a kernel onset at 2014-07-01 07:20:35 UTC is stored as
    2014-07-01 00:20:35, and its first window starts at 2014-06-29 17:20:35 Pacific time
    (2014-06-30 00:20:35 UTC).  The windows keep this convention so that they match the stored model
    outputs.

    Args:
        t_epoch: kernel onsets (cohort.csv `active_start`, epoch seconds), array-like without NaN.
        tz: time zone of the wall-clock conversion.

    Returns:
        float array of onsets in epoch seconds, same length as t_epoch.  A fractional second of the
        input is carried over unchanged (it is zero on the daily grid).
    """
    s = pd.to_datetime(pd.Series(np.asarray(t_epoch, dtype="int64")), unit="s", utc=True)
    local_naive = s.dt.tz_convert(tz).dt.tz_localize(None)       # Pacific wall-clock time, no zone
    frac = np.asarray(t_epoch, dtype=float) - np.asarray(t_epoch, dtype="int64")
    # seconds since 1970-01-01 of the naive wall-clock time, i.e. the wall-clock time read as UTC
    return (local_naive - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy() + frac
