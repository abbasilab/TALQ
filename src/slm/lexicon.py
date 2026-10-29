"""
Lexicon and style features, the comparators of the validation (Methods, "Validation against human
annotations"; Fig. 1d).

* VADER sentiment (vaderSentiment package): negative, neutral, positive and compound scores.
* Empath categories (Fast et al., 2016), applied with the lexicon vendored in
  third_party/empath/categories.tsv (194 categories): count of tokens that belong to a category
  divided by the number of tokens, as Empath's analyze(normalize=True).
* Style: rates of first person singular pronouns and of absolutist words per token.

The manuscript compares nine of these features with the model ratings: VADER negativity and
compound, the Empath categories negative_emotion, sadness, shame and suffering, the two style rates
and post length, log(1 + n_tokens) (scripts/14b_validation_local.py; scripts/14_validation_analysis.py
uses the same lexicon features).  Tokens are runs of lower-case letters and apostrophes ([a-z']+) in
the lower-cased text.

Used by scripts/13_lexicon_features.py (`python3 13_lexicon_features.py`).
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from . import params as P

_TOK = re.compile(r"[a-z']+")


@lru_cache(maxsize=1)
def empath_lexicon() -> dict[str, set[str]]:
    """
    The Empath lexicon: dict category -> set of its terms (third_party/empath/categories.tsv,
    one tab-separated line per category: name, then terms).  Read once and cached.
    """
    cats = {}
    with open(P.ROOT / "third_party" / "empath" / "categories.tsv") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            cats[cols[0]] = set(cols[1:])
    return cats


@lru_cache(maxsize=1)
def _inv():
    """Inverted Empath lexicon: dict term -> list of the categories that contain it (cached)."""
    inv = {}
    for c, terms in empath_lexicon().items():
        for t in terms:
            inv.setdefault(t, []).append(c)
    return inv


def empath_counts(text: str) -> tuple[dict, int]:
    """
    Empath category counts of one text.

    Args:
        text: raw text (None is treated as empty).

    Returns:
        (counts, n_tokens): dict category -> number of tokens of the text in that category (a token
        counts once for every category that contains it), and the number of tokens.
    """
    toks = _TOK.findall((text or "").lower())
    inv = _inv()
    cnt: dict[str, float] = {}
    for t in toks:
        for c in inv.get(t, ()):
            cnt[c] = cnt.get(c, 0.0) + 1.0
    return cnt, len(toks)


def empath_frame(texts: list[str]) -> pd.DataFrame:
    """
    Empath rates of a list of texts.

    Args:
        texts: list of strings.

    Returns:
        DataFrame with one row per text: empath_<category> (count / number of tokens; 0 for a text
        without tokens) for the 194 categories in alphabetical order, and n_tokens.
    """
    cats = sorted(empath_lexicon())
    M = np.zeros((len(texts), len(cats))); n = np.zeros(len(texts))
    idx = {c: i for i, c in enumerate(cats)}
    for r, t in enumerate(texts):
        cnt, n[r] = empath_counts(t)
        for c, v in cnt.items():
            M[r, idx[c]] = v
    with np.errstate(invalid="ignore", divide="ignore"):
        M = M / n[:, None]                        # rates per token; 0/0 -> NaN, set to 0 below
    df = pd.DataFrame(np.nan_to_num(M), columns=[f"empath_{c}" for c in cats])
    df["n_tokens"] = n
    return df


_VADER = None                                     # analyzer, created on first use


def vader_frame(texts: list[str]) -> pd.DataFrame:
    """
    VADER polarity scores of a list of texts.

    Args:
        texts: list of strings (None is treated as empty).

    Returns:
        DataFrame with one row per text and the columns vader_neg, vader_neu, vader_pos and
        vader_compound (SentimentIntensityAnalyzer.polarity_scores).
    """
    global _VADER
    if _VADER is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _VADER = SentimentIntensityAnalyzer()
    rows = [_VADER.polarity_scores(t or "") for t in texts]
    return pd.DataFrame(rows).add_prefix("vader_")


# first person singular pronouns (Rude et al., 2004) and absolutist words (Al-Mosaiwi and
# Johnstone, 2018), matched as whole words in the lower-cased text
FIRST_PERSON = re.compile(r"\b(i|me|my|mine|myself)\b")
ABSOLUTIST = re.compile(r"\b(always|never|nothing|completely|totally|entire|everything|constantly|definitely|"
                        r"absolutely|whole|every|all)\b")


def style_frame(texts: list[str]) -> pd.DataFrame:
    """
    Style rates of a list of texts.

    Args:
        texts: list of strings (None is treated as empty).

    Returns:
        DataFrame with one row per text: style_first_person and style_absolutist, the number of
        matches divided by the number of tokens (at least 1).
    """
    rows = []
    for t in texts:
        low = (t or "").lower()
        n = max(len(_TOK.findall(low)), 1)
        rows.append({"style_first_person": len(FIRST_PERSON.findall(low)) / n,
                     "style_absolutist": len(ABSOLUTIST.findall(low)) / n})
    return pd.DataFrame(rows)
