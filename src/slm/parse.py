"""
Parsing of model responses and collation into window x symptom tables (Methods, "Language model
questionnaire" and "Symptom profiles"; Extended Data Figs. 1 and 2).

Each stored response should be a JSON object with the fields reasoning, rating and confidence
(config/task_symptom.yaml).  parse_one() reads a response in four stages:

  1. the whole response as JSON                                      status "json"
  2. else the first embedded JSON object, for example inside a
     fenced code block or after introductory text                    status "json_block"
  3. else rating and confidence read with regular expressions        status "regex"
  4. else missing                                                    status "fail"

A missing response is "empty".  Ratings and confidences outside the 0-100 scale are set to missing
(parse_responses).  For the main run, 93.0% of responses were valid JSON, 6.9% were read in stages
2-3, and 3,779 (0.15%) were unparseable; these figures are computed over the 491 users by
scripts/23_manuscript_numbers.py.

summarize_replicates() averages the five samples of each window and symptom, ignoring missing
ratings; collate() builds the window x symptom tables and applies the selection of the Methods
("Symptom profiles"): users with at least 200 window-symptom pairs, windows with all 52 symptoms
rated, and windows whose mean confidence exceeds 40 for every symptom (7,094 windows, 481 users).

Used by scripts/05_parse_collate.py (`python3 05_parse_collate.py --run llama31_8b`); parse_one is
also used by the validation scripts (14, 14b, 14c), and symptom_matrix by the factor analyses.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from . import params as P
from .prompts import load_symptoms

# "rating": 70, rating = 70, "rating": "70" ... (first match; optional quotes, ':' or '=')
_RE_RATING = re.compile(r'"?rating"?\s*[:=]\s*"?(-?\d+(?:\.\d+)?)')
_RE_CONF = re.compile(r'"?confidence"?\s*[:=]\s*"?(-?\d+(?:\.\d+)?)')


def _first_json_object(s: str):
    """
    First balanced {...} substring of `s` that parses as a JSON object.

    Scans from each "{" and tracks the brace depth, ignoring braces inside JSON strings (with
    backslash escapes).  When a balanced substring does not parse, the scan restarts at the next "{".

    Args:
        s: response text.

    Returns:
        the decoded dict, or None if no balanced substring parses.
    """
    start = s.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:                    # balanced substring s[start:i + 1]
                    try:
                        obj = json.loads(s[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except Exception:  # noqa: BLE001
                        break
        start = s.find("{", start + 1)
    return None


def _to_float(v):
    """float(v) if it is finite, else NaN (also for None, non-numeric strings and other types)."""
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except Exception:  # noqa: BLE001
        return np.nan


# reasoning block of models that emit one (removed before parsing)
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def parse_one(resp: str, rating_key: str = "rating", conf_key: str = "confidence"):
    """
    Parse one model response.

    A reasoning block (<think>...</think>, for example the empty one left by a chat template with
    thinking switched off) is removed first; then the stages of the module docstring are tried in
    order.  Values that are not finite numbers become NaN (the 0-100 range is checked by
    parse_responses).

    Args:
        resp: raw response text (None or NaN for a missing response).
        rating_key, conf_key: JSON field names of the rating and the confidence.

    Returns:
        (reasoning, rating, confidence, status): reasoning text or None, rating and confidence as
        floats (NaN when missing), status one of "json", "json_block", "regex", "fail", "empty".
    """
    if resp is None or (isinstance(resp, float) and np.isnan(resp)):
        return None, np.nan, np.nan, "empty"
    s = _THINK.sub("", str(resp)).strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return (obj.get("reasoning"), _to_float(obj.get(rating_key)),
                    _to_float(obj.get(conf_key)), "json")
    except Exception:  # noqa: BLE001
        pass
    obj = _first_json_object(s)
    if obj is not None:
        return (obj.get("reasoning"), _to_float(obj.get(rating_key)),
                _to_float(obj.get(conf_key)), "json_block")
    mr, mc = _RE_RATING.search(s), _RE_CONF.search(s)
    if mr or mc:
        return (None, _to_float(mr.group(1)) if mr else np.nan,
                _to_float(mc.group(1)) if mc else np.nan, "regex")
    return None, np.nan, np.nan, "fail"


def attach_symptom_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add symptom_shortcode (where missing) and symptom_number from the column symptom_title.

    Both the display title and the title shown to the model (prompt_title, for example
    'Indiciciveness') map to the same battery entry.

    Args:
        df: one row per request with a symptom_title column (modified in place).

    Returns:
        df with the columns symptom_shortcode and symptom_number.
    """
    spec = {s["title"]: s for s in load_symptoms()}
    spec.update({s["prompt_title"]: s for s in load_symptoms()})
    if "symptom_shortcode" not in df or df["symptom_shortcode"].isna().any():
        df["symptom_shortcode"] = df["symptom_title"].map(lambda t: spec[t]["shortcode"])
    df["symptom_number"] = df["symptom_title"].map(lambda t: spec[t]["number"])
    return df


def parse_responses(raw: pd.DataFrame, responses=None, keep_reasoning: bool = False) -> pd.DataFrame:
    """
    Parse every response of a run (parse_one) and attach symptom codes.

    Args:
        raw: one row per request, in export order, with the key columns (speaker_id,
            window_start, symptom_title, replicate, ...) and optionally `response`.
        responses: the response strings in the order of `raw`; passed separately for the
            2.5-million-row export to save memory (default: raw["response"]).
        keep_reasoning: also keep the free-text reasoning (column response_reasoning).

    Returns:
        copy of `raw` without `response`, plus response_rating and response_confidence (floats;
        NaN when missing, unparseable or outside 0-100), parse_status (categorical) and the
        symptom codes of attach_symptom_codes.
    """
    df = raw.drop(columns=["response"], errors="ignore").copy()
    resp = responses if responses is not None else raw["response"].tolist()
    res = [parse_one(r) for r in resp]
    status = np.array([r[3] for r in res], dtype=object)
    if keep_reasoning:
        df["response_reasoning"] = [r[0] for r in res]
    df["response_rating"] = np.array([r[1] for r in res], dtype=float)
    df["response_confidence"] = np.array([r[2] for r in res], dtype=float)
    df["parse_status"] = pd.Categorical(status)
    del res
    # out-of-range values are invalid on the 0-100 rubric (the parse status is kept)
    for c in ["response_rating", "response_confidence"]:
        bad = (df[c] < 0) | (df[c] > 100)
        df.loc[bad, c] = np.nan
    return attach_symptom_codes(df)


def summarize_replicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean and standard deviation over the replicates of each (account, window, symptom).

    Missing ratings and confidences are ignored: the mean and SD (ddof = 0) are taken over the
    available samples, and are NaN only when no sample of the pair has a value.

    Args:
        df: parsed requests (parse_responses), one row per response.

    Returns:
        DataFrame sorted by (speaker_id, window_start, symptom_shortcode) with the columns
        rating_mean, rating_std, rating_n, confidence_mean, confidence_std, confidence_n, where
        *_n is the number of stored responses of the pair (missing values included).
    """
    keys = ["speaker_id", "window_start", "symptom_shortcode"]
    d = df[keys + ["response_rating", "response_confidence"]].copy()
    for c in keys:
        if isinstance(d[c].dtype, pd.CategoricalDtype):
            d[c] = d[c].astype(str)               # plain strings: no unused category combinations
    g = d.groupby(keys, sort=True, observed=True)
    parts = {}
    for col, name in [("response_rating", "rating"), ("response_confidence", "confidence")]:
        mean = g[col].mean()                       # skips NaN
        std = g[col].std(ddof=0)
        n = g[col].size()                          # responses, NaN included
        std = std.where(g[col].count() > 0)        # SD undefined without any value
        parts[f"{name}_mean"] = mean; parts[f"{name}_std"] = std; parts[f"{name}_n"] = n
    out = pd.DataFrame(parts)[["rating_mean", "rating_std", "rating_n", "confidence_mean", "confidence_std",
                               "confidence_n"]].reset_index()
    return out


def collate(summary: pd.DataFrame, n_points_min: int = P.N_POINTS_MIN,
            confidence_thresh: float = P.CONFIDENCE_THRESH):
    """
    Window x symptom tables and the selection of windows for the analyses (Methods, "Symptom
    profiles").

    Steps:
      * keep users with at least n_points_min (200) window-symptom pairs in `summary`;
      * pivot to one row per (speaker_id, window_start) with the columns <shortcode>_rating and
        <shortcode>_confidence (replicate means), interleaved in battery order;
      * collated: every window of these users that has at least one response;
      * collated_nona: windows in which all 52 ratings and confidences are available;
      * collated_confident: windows of collated_nona whose mean confidence exceeds
        confidence_thresh (40) for every symptom.

    Args:
        summary: output of summarize_replicates.
        n_points_min: minimum number of window-symptom pairs per user.
        confidence_thresh: confidence threshold (points on the 0-100 scale).

    Returns:
        (collated, collated_nona, collated_confident, report): three DataFrames and a dict of row
        and user counts (n_speakers_summary, n_speakers_good, n_summary_rows, n_summary_rows_good,
        n_collated, n_nona, n_confident, n_speakers_confident).
    """
    symptoms = [s["shortcode"] for s in load_symptoms()]
    counts = summary.groupby("speaker_id").size()  # window-symptom pairs per user
    good = counts[counts >= n_points_min].index
    s = summary[summary["speaker_id"].isin(good)]
    wide_r = s.pivot_table(index=["speaker_id", "window_start"], columns="symptom_shortcode",
                           values="rating_mean", aggfunc="first", dropna=False)
    wide_c = s.pivot_table(index=["speaker_id", "window_start"], columns="symptom_shortcode",
                           values="confidence_mean", aggfunc="first", dropna=False)
    wide_r = wide_r.reindex(columns=symptoms)
    wide_c = wide_c.reindex(columns=symptoms)
    coll = pd.concat([wide_r.add_suffix("_rating"), wide_c.add_suffix("_confidence")], axis=1)
    # column order: rating and confidence of each symptom, interleaved, in battery order
    order = []
    for sc in symptoms:
        order += [f"{sc}_rating", f"{sc}_confidence"]
    coll = coll[order].reset_index()
    # pivot_table(dropna=False) can add (speaker, window) combinations that do not exist;
    # keep only the windows present in the summary
    present = s.groupby(["speaker_id", "window_start"]).size().reset_index()[["speaker_id", "window_start"]]
    coll = present.merge(coll, on=["speaker_id", "window_start"], how="left")
    nona = coll.dropna()
    conf_cols = [f"{sc}_confidence" for sc in symptoms]
    confident = nona[(nona[conf_cols] > confidence_thresh).all(axis=1)]
    report = {"n_speakers_summary": int(summary["speaker_id"].nunique()),
              "n_speakers_good": int(len(good)),
              "n_summary_rows": int(len(summary)), "n_summary_rows_good": int(len(s)),
              "n_collated": int(len(coll)), "n_nona": int(len(nona)),
              "n_confident": int(len(confident)),
              "n_speakers_confident": int(confident["speaker_id"].nunique())}
    return coll, nona, confident, report


def symptom_matrix(df: pd.DataFrame, kind: str = "rating") -> np.ndarray:
    """
    Window x symptom matrix from a collated table.

    Args:
        df: collated table (one row per window).
        kind: "rating" or "confidence".

    Returns:
        float array (n_windows, 52), columns in battery order.
    """
    symptoms = [s["shortcode"] for s in load_symptoms()]
    return df[[f"{s}_{kind}" for s in symptoms]].to_numpy(dtype=float)
