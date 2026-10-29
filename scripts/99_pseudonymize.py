#!/usr/bin/env python3
"""
Step 99 - shareable copy of the ratings and derived data, with pseudonymous user identifiers.

Supports the Data availability statement: the model ratings and the derived window-level data of
the main run can be shared without account names, calendar dates or any text.

What is released (folder release/, written from the main run llama31_8b)
  ratings_replicate_level.parquet  one row per stored response of the 491 users: user, window,
                                   symptom_shortcode, symptom_number, replicate (0-4), rating,
                                   confidence (0-100; missing if the response could not be parsed)
                                   and parse_status (json, json_block, regex, fail); duplicated
                                   outputs of a request are reduced to the first one, as in step 5
  collated_confident.csv.gz        the 7,094 confident windows: mean rating and confidence of the
                                   52 symptoms (<shortcode>_rating, <shortcode>_confidence)
  factors<tag>.csv.gz              the same windows with the factor scores nmf_0.. (0-100) in the
                                   order of the factor titles of results/<run>/06_nmf<tag>.npz
                                   (tag "" = K = 12; "_k18" = K = 18 sensitivity solution)
  states<tag>.csv.gz               discrete state of every confident window (column state; for the
                                   main solution also the labels at every hyperparameter step,
                                   step_0..step_13); tags "", "_8states", "_k18"
  regime_labels<tag>.csv           regime (1..K) and number of windows of every user in the regime
                                   analysis; tags "", "_8states", "_k18"
  course_types.csv                 course type (1..K) of every user and the posterior probability of
                                   each type (p_type1..p_typeK), from the growth mixture model of step 20
  symptoms.yaml                    the symptom battery (config/symptoms.yaml)
  README.txt                       this description in short, with the factor titles of every nmf_<j>
                                   column (from 06_nmf<tag>.npz) and the state names (10_states<tag>.json)
  Files whose source does not exist are skipped.

Pseudonymization
  * user: "u" + the first 12 hex digits of SHA-256(salt + account name).  The salt is a random
    32-hex-digit string created once and kept in release_salt.txt at the repository root; it must
    not be shared or committed, because anyone with the salt can test guesses of account names.
    The same salt gives the same identifiers in every release.
  * window: index of the two-week window in the user's first year of activity (0 = the window that
    starts one day before onset, 26 = the last), taken from data/processed/windows.parquet.  It
    replaces the window start, so no calendar date is released, and it is the same in every file.
  * No text is released: columns are taken from an explicit list, and the script stops if a
    released table would contain any other text column (post text, model reasoning or responses).

Inputs
  data/processed/llama31_8b/requests.parquet, collated_confident.parquet, factors<tag>.parquet,
  states<tag>.parquet (steps 5, 6, 10, 10b); results/llama31_8b/11_regime_labels<tag>.csv (step 11),
  20_course_labels.csv (step 20), 06_nmf<tag>.npz (step 6), 10_states<tag>.json (steps 10, 10b, 10c);
  data/processed/windows.parquet (step 3; only speaker_id, window_start and window_index are read)
Outputs
  release/ (see above) and, on the first run, release_salt.txt

Usage (run from scripts/)
  python3 99_pseudonymize.py
"""
import argparse
import hashlib
import json
import secrets
import shutil
import textwrap

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from slm import params as P
from slm.prompts import load_symptoms

# --help prints this description; the step has no options
argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

RUN = P.MAIN_RUN
pdir, rdir = P.processed_dir(RUN), P.results_dir(RUN)
out = P.ROOT / "release"; out.mkdir(exist_ok=True)

# ---- step: salt of the pseudonymous identifiers -----------------------------------------------------
salt_f = P.ROOT / "release_salt.txt"
if not salt_f.exists():
    salt_f.write_text(secrets.token_hex(16))
salt = salt_f.read_text().strip()


def pid(u):
    """Pseudonymous identifier of account name u: "u" + 12 hex digits of SHA-256(salt + u)."""
    return "u" + hashlib.sha256((salt + str(u)).encode()).hexdigest()[:12]


# ---- step: window index of every (account, window start) ------------------------------------------
# only these three columns are read; windows.parquet also holds the post text
win = pd.read_parquet(P.PROCESSED / "windows.parquet", columns=["speaker_id", "window_start", "window_index"])
win["speaker_id"] = win["speaker_id"].astype(str)
win["window_start"] = pd.to_datetime(win["window_start"]).astype("datetime64[ns]")

# text columns allowed in a release: the pseudonymous user and two fixed vocabularies
TEXT_OK = {"user", "symptom_shortcode", "parse_status"}


def anon(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Release version of a table with columns speaker_id [, window_start] and `columns`.

    Keeps only `columns`; replaces speaker_id by the pseudonymous user and window_start by the
    window index; stops with an error if any released column other than TEXT_OK is not numeric.
    Returns the new table with user [, window] first.
    """
    d = df[["speaker_id"] + (["window_start"] if "window_start" in df else []) + columns].copy()
    d["speaker_id"] = d["speaker_id"].astype(str)
    if "window_start" in d:
        d["window_start"] = pd.to_datetime(d["window_start"]).astype("datetime64[ns]")
        d = d.merge(win, on=["speaker_id", "window_start"], how="left", validate="many_to_one")
        if d["window_index"].isna().any():
            raise SystemExit("windows without an index in windows.parquet")
        d.insert(1, "window", d.pop("window_index").astype(int))
        d = d.drop(columns=["window_start"])
    d.insert(0, "user", d.pop("speaker_id").map(pid))
    bad = [c for c in d.columns if c not in TEXT_OK and not is_numeric_dtype(d[c])]
    if bad:
        raise SystemExit(f"non-numeric columns would be released: {bad}")
    return d


written = []

# ---- step: replicate-level ratings -------------------------------------------------------------------
req_f = pdir / "requests.parquet"
if req_f.exists():
    r = pd.read_parquet(req_f, columns=["speaker_id", "window_start", "symptom_shortcode", "symptom_number",
                                        "replicate", "rating", "confidence", "parse_status"])
    r["speaker_id"] = r["speaker_id"].astype(str)
    r = r[~r["speaker_id"].isin(P.BOT_SPEAKERS)]            # the moderation bot is not a user
    # duplicated outputs of the same request: first occurrence in export (file) order, as in step 5
    r = r[~r.duplicated(["speaker_id", "window_start", "symptom_shortcode", "replicate"], keep="first")]
    for c in ["symptom_shortcode", "parse_status"]:
        r[c] = r[c].astype(str)
    anon(r, ["symptom_shortcode", "symptom_number", "replicate", "rating", "confidence", "parse_status"]) \
        .to_parquet(out / "ratings_replicate_level.parquet", index=False)
    written.append("ratings_replicate_level.parquet")

# ---- step: window-level tables ------------------------------------------------------------------------
# rating and confidence columns of the 52 symptoms, in the order of the battery
rc_cols = [f"{s['shortcode']}_{kind}" for s in load_symptoms() for kind in ("rating", "confidence")]
cc_f = pdir / "collated_confident.parquet"
if cc_f.exists():
    anon(pd.read_parquet(cc_f), rc_cols).to_csv(out / "collated_confident.csv.gz", index=False)
    written.append("collated_confident.csv.gz")
for tag in ["", "_k18"]:                                    # main factor solution and K = 18
    f = pdir / f"factors{tag}.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        nmf_cols = [c for c in d.columns if c.startswith("nmf_")]
        anon(d, nmf_cols + [c for c in rc_cols if c in d]).to_csv(out / f"factors{tag}.csv.gz", index=False)
        written.append(f"factors{tag}.csv.gz")
for tag in ["", "_8states", "_k18"]:                        # main states and the sensitivity analyses
    sf = pdir / f"states{tag}.parquet"
    if sf.exists():
        d = pd.read_parquet(sf)
        anon(d, [c for c in d.columns if c.startswith("step_")] + ["state"]) \
            .to_csv(out / f"states{tag}.csv.gz", index=False)
        written.append(f"states{tag}.csv.gz")
    rl = rdir / f"11_regime_labels{tag}.csv"
    if rl.exists():
        anon(pd.read_csv(rl), ["regime", "n_windows"]).to_csv(out / f"regime_labels{tag}.csv", index=False)
        written.append(f"regime_labels{tag}.csv")
cl = rdir / "20_course_labels.csv"                         # course types over the year (step 20)
if cl.exists():
    d = pd.read_csv(cl)
    anon(d, ["type"] + [c for c in d.columns if c.startswith("p_type")]).to_csv(out / "course_types.csv",
                                                                                 index=False)
    written.append("course_types.csv")
shutil.copy(P.CONFIG / "symptoms.yaml", out / "symptoms.yaml")
written.append("symptoms.yaml")

# ---- step: short description of the release, with the factor titles and state names ------------------
lines = ["Ratings of 52 depression symptoms by Llama-3.1-8B-Instruct for two-week windows of Reddit posts, "
         "and data derived from them (factor scores, states, regimes, course types).  No text and no dates "
         "are included.",
         "user: pseudonymous identifier (salted SHA-256 of the account name; the salt is not released).",
         "window: index of the two-week window in the user's first year of activity (0 = the window that "
         "starts one day before onset); the same index in every file.",
         "Ratings, confidences and factor scores are on the 0-100 scale; state, regime and course type are "
         "integer labels.",
         "Files: " + ", ".join(written) + "."]
for tag in ["", "_k18"]:                                    # names of the factor columns
    npz = rdir / f"06_nmf{tag}.npz"
    if f"factors{tag}.csv.gz" in written and npz.exists():
        titles = list(np.load(npz, allow_pickle=True)["titles"])
        lines.append(f"factors{tag}.csv.gz: "
                     + "; ".join(f"nmf_{j} = {t}" for j, t in enumerate(titles)) + ".")
for tag in ["", "_8states", "_k18"]:                        # names of the states
    sj = rdir / f"10_states{tag}.json"
    if f"states{tag}.csv.gz" in written and sj.exists():
        names = json.loads(sj.read_text()).get("state_names") or []
        if names:
            lines.append(f"states{tag}.csv.gz: "
                         + "; ".join(f"state {i} = {n}" for i, n in enumerate(names)) + ".")
# paragraphs wrapped at 100 characters, without splitting hyphenated names
(out / "README.txt").write_text("\n\n".join(textwrap.fill(s, 100, break_on_hyphens=False) for s in lines)
                                + "\n")
print("release written to", out, ":", ", ".join(written))
