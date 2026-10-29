#!/usr/bin/env python3
"""
Step 22 - Extended Data figures (Extended Data Figs. 1-10) of the manuscript, numbered in the order in
which the main text first cites them.

  ed1_cohort        ED Fig. 1   cohort, text volume and the confidence filter
  ed2_parsing       ED Fig. 2   response parsing (routes, ratings by route), window flow, confidence per
                                symptom
  ed3_validation    ED Fig. 3   prediction of human annotations in detail, from the reduced protocol run on a
                                single local computer: weights of the lasso for each label, predicted
                                severity by level, agreement of the local deployment with the cohort
                                annotation, other annotating models
  ed4_inventories   ED Fig. 4   the TALQ battery versus BDI-II and PHQ-9 completed by the same model
  ed5_factors       ED Fig. 5   symptom correlations, factor replication by community and confidence
                                threshold, composition of each factor, comparison with solutions of 10 and
                                18 factors
  ed6_psychometrics ED Fig. 6   internal consistency under alternative estimators and confidence strata,
                                item discrimination
  ed7_var           ED Fig. 7   vector autoregression within individuals: Granger predictive pairs, step
                                responses, lag order, distribution of the lagged P values
  ed8_states        ED Fig. 8   selection of the number of discrete states (DPMM hyperparameter sweep)
  ed9_regimes       ED Fig. 9   model selection, sensitivity and description of the dynamical regimes
  ed10_sequences    ED Fig. 10  example state sequences per regime and a Gaussian hidden Markov model
Panels with missing inputs are drawn as labeled placeholders.

Inputs (<run> is --run; the step that writes each file in parentheses)
  data/processed/cohort.csv, user_description.parquet, windows.parquet       (steps 2, 15, 3)
  data/processed/inventories.parquet, results/inventories/18_*.csv          (step 18)
  data/processed/<run>/requests.parquet, summary.parquet, collated_nona.parquet,
      collated_confident.parquet, results/<run>/05_parse_report.json        (step 5)
  data/processed/<run>/factors.parquet, results/<run>/06_nmf.npz,
      06_nmf_k10.npz, 06_nmf_k18.npz                                        (step 6)
  results/<run>/07_psychometrics.csv                                        (step 7)
  results/<run>/08_var_population.npy, 08_var_order_ic.csv                  (step 8)
  results/<run>/10_dpmm_sweep.csv, 10_states.json, 10_states_8states.json,
      data/processed/<run>/states.parquet                                   (steps 10, 10b, 10c)
  results/<run>/11_regimes.npy, 11_regime_sweep.csv, 11_regime_burden.csv   (step 11)
  results/<run>/12_robustness.json, 12D_heldout.csv, 12D_heldout_8states.csv (step 12)
  results/<run>/12c_community_congruence.csv, 12c_community.json           (step 12c)
  results/validation/14b_<model>_*.csv, 14b_<model>_summary.json            (step 14b)
  results/validation/14d_<model>_*.csv                                     (step 14d)
Outputs  figures/extended_data/ED_Fig1..ED_Fig10.{pdf,svg,png,tif,pptx}

usage: python3 22_figures_extended.py [--run llama31_8b] [--only 1,2,...,10]   (numbers as above)
"""
import argparse

import _setup  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats

from slm import params as P
from slm import plotting as PL
from slm.io import load_json
from slm.parse import symptom_matrix
from slm.prompts import load_symptoms

ap = argparse.ArgumentParser(description="Extended Data figures (Extended Data Figs. 1-10).")
ap.add_argument("--run", default=P.MAIN_RUN,
                help="annotation run whose results are drawn (default: %(default)s)")
ap.add_argument("--only", default=None,
                help="comma-separated Extended Data figure numbers to draw, 1-10 (default: all)")
ap.add_argument("--subdir", default="extended_data",
                help="folder under figures/ for the outputs (default: extended_data)")
args = ap.parse_args()
only = set(args.only.split(",")) if args.only else None
PL.setup()
MM = PL.MM
INK, INK2, MUTED, GRID, AXIS = PL.INK, PL.INK2, PL.MUTED, PL.GRID, PL.AXIS
RC_ = P.results_dir(args.run)                    # results of the run
PC_ = P.processed_dir(args.run)                  # processed tables of the run
SYM = load_symptoms()
codes = [s["shortcode"] for s in SYM]
stitle = {s["shortcode"]: s["title"] for s in SYM}


def want(k):
    """True when Extended Data figure k ("1".."10") was requested with --only, or when --only is not given."""
    return only is None or k in only


def pending(ax, msg):
    """Placeholder panel: a dashed light frame with msg (the missing input) in the center."""
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(GRID); s.set_linestyle((0, (2, 2)))
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=6)


def lab(ax, s, x=-0.25, y=1.06):
    """Panel letter s at (x, y) in axes coordinates (see slm.plotting.panel_label)."""
    PL.panel_label(ax, s, x=x, y=y)


def jl(p):
    """Contents of the JSON file p, or an empty dict when it does not exist."""
    return load_json(p) if p.exists() else {}


def plain_log_ticks(ax, ticks):
    """Major ticks of a logarithmic x axis labeled as plain numbers (1,000 rather than 10^3)."""
    lo, hi = ax.get_xlim()
    tk = [t for t in ticks if lo <= t <= hi]
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xticks(tk); ax.set_xticklabels([f"{t:,}" for t in tk])
    ax.set_xlim(lo, hi)


# ====================================================================== ED1
def ed1_cohort():
    """Extended Data Fig. 1: cohort, text volume and the confidence filter.

    a  utterances per user in the two corpora (histogram, logarithmic bins); b  scored windows per user,
    out of 27; c  words per scored window (logarithmic bins); d  year of onset of sustained activity;
    e  fraction of complete windows (all 52 symptoms rated) that pass the confidence filter (confidence
    above 40 for every symptom), by decile of words per window (x: median words of the decile);
    f  median of the mean model confidence of each window, by the same deciles.
    All panels describe the 491 users: AutoModerator, a moderation bot that met the activity criteria and
    whose windows were annotated with the others, is excluded (P.BOT_SPEAKERS).
    Inputs: user_description.parquet (step 15), windows.parquet (step 3), cohort.csv (step 2),
    collated_nona.parquet (step 5).  Returns the path of the PNG.
    """
    fig = plt.figure(figsize=(180 * MM, 95 * MM))
    gs = GridSpec(2, 4, figure=fig, wspace=0.55, hspace=0.75, left=0.07, right=0.985, top=0.94, bottom=0.11)
    ud = (pd.read_parquet(P.PROCESSED / "user_description.parquet")
          if (P.PROCESSED / "user_description.parquet").exists() else None)
    win = pd.read_parquet(P.PROCESSED / "windows.parquet",
                          columns=["speaker_id", "window_start", "window_index", "n_posts", "n_submissions",
                                   "n_words"])
    win["window_start"] = pd.to_datetime(win["window_start"])
    # describe the 491 people: the moderation bot's windows were annotated but are not analyzed
    win = win[~win["speaker_id"].isin(P.BOT_SPEAKERS)]
    if ud is not None and "speaker_id" in ud:
        ud = ud[~ud["speaker_id"].isin(P.BOT_SPEAKERS)]
    coh = pd.read_csv(P.PROCESSED / "cohort.csv")
    coh = coh[coh["in_cohort"] & ~coh["speaker_id"].isin(P.BOT_SPEAKERS)]
    # onset (first active day, epoch seconds) -> calendar year
    onset_years = pd.to_datetime(coh["active_start"], unit="s").dt.year.value_counts().sort_index()
    # a utterances per user
    ax = fig.add_subplot(gs[0, 0])
    ncol = "n_utterances" if ud is not None and "n_utterances" in ud else "n"   # utterances in the corpora
    if ud is not None and ncol in ud:
        # 30 logarithmic bins; cohort members have at least 20 utterances
        ax.hist(ud[ncol], bins=np.logspace(np.log10(max(20, ud[ncol].min())), np.log10(ud[ncol].max()), 30),
                color=PL.SLOTS[0], lw=0)
        ax.set_xscale("log")
        plain_log_ticks(ax, (100, 300, 1000, 3000, 10000))
    else:
        pending(ax, "user description (step 15)")
    ax.set_xlabel("Utterances per user (corpus)"); ax.set_ylabel("Users"); PL.hgrid(ax); lab(ax, "a", -0.4)
    # b windows per user
    ax = fig.add_subplot(gs[0, 1])
    nw = win.groupby("speaker_id").size()
    ax.hist(nw, bins=np.arange(0.5, 28.5, 1), color=PL.SLOTS[0], lw=0)   # one bin per count, 1-27
    ax.set_xlabel("Scored windows per user (of 27)"); ax.set_ylabel("Users"); PL.hgrid(ax); lab(ax, "b", -0.4)
    # c words per window
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(win["n_words"].clip(lower=1), bins=np.logspace(0, np.log10(win["n_words"].max()), 40),
            color=PL.SLOTS[0], lw=0)
    ax.set_xscale("log"); ax.set_xlabel("Words per window"); ax.set_ylabel("Windows")
    PL.hgrid(ax); lab(ax, "c", -0.4)
    plain_log_ticks(ax, (1, 10, 100, 1000, 10000))
    # d onset year
    ax = fig.add_subplot(gs[0, 3])
    ks = [int(k) for k in onset_years.index]
    ax.bar(ks, onset_years.to_numpy(), color=PL.SLOTS[0], width=0.7, lw=0)
    ax.set_xticks(ks[::2]); ax.set_xticklabels([str(k) for k in ks[::2]])   # every second year labeled
    ax.set_xlabel("Year of activity onset"); ax.set_ylabel("Users"); PL.hgrid(ax); lab(ax, "d", -0.4)
    # e/f confidence filter vs volume
    nona_p = PC_ / "collated_nona.parquet"
    if nona_p.exists():
        nona = pd.read_parquet(nona_p)                 # windows with all 52 symptoms rated
        d = nona.merge(win, on=["speaker_id", "window_start"], how="left")
        cc = [f"{c}_confidence" for c in codes]
        d["pass"] = (d[cc] > P.CONFIDENCE_THRESH).all(axis=1)    # the filter of the main analyses
        d["mean_conf"] = d[cc].mean(axis=1)
        # deciles of words per window: bin edges at the 0, 10, ..., 100 % quantiles, bins 0-9
        bins = np.quantile(d["n_words"], np.linspace(0, 1, 11))
        d["bin"] = np.clip(np.searchsorted(bins, d["n_words"], side="right") - 1, 0, 9)
        g = d.groupby("bin").agg(words=("n_words", "median"), pr=("pass", "mean"),
                                 conf=("mean_conf", "median"))
        ax = fig.add_subplot(gs[1, 0:2])
        ax.plot(g["words"], g["pr"], "-o", ms=2.6, lw=1.0, color=PL.SLOTS[0])
        ax.set_xscale("log"); ax.set_ylim(0, 1); ax.set_xlabel("Words per window (decile median)")
        ax.set_ylabel("Fraction passing the\nconfidence filter"); PL.hgrid(ax); lab(ax, "e", -0.17)
        plain_log_ticks(ax, (30, 100, 300, 1000, 3000))
        ax = fig.add_subplot(gs[1, 2:4])
        ax.plot(g["words"], g["conf"], "-o", ms=2.6, lw=1.0, color=PL.SLOTS[0])
        ax.set_xscale("log"); ax.set_xlabel("Words per window (decile median)")
        ax.set_ylabel("Median of the mean LLM\nconfidence per window"); PL.hgrid(ax); lab(ax, "f", -0.17)
        plain_log_ticks(ax, (30, 100, 300, 1000, 3000))
    else:
        pending(fig.add_subplot(gs[1, :]), "collated windows (step 5)")
    return PL.save(fig, "ED_Fig1", args.subdir)


# ====================================================================== ED2
# parse routes of step 5 (parse_status of requests.parquet) and their axis labels
PARSE_ROUTES = [("json", "valid\nJSON"), ("json_block", "embedded\nJSON"), ("regex", "regular\nexpression"),
                ("fail", "unparse-\nable")]


def cohort_responses(cols):
    """Responses of the 491 people (moderation bot excluded), one row per request.

    cols : columns of requests.parquet to read.  A request stored more than once counts once (its first
    row), as in the window summaries of step 5.
    """
    r = pd.read_parquet(P.processed_dir(args.run) / "requests.parquet", columns=cols)
    r = r[~r["speaker_id"].isin(P.BOT_SPEAKERS)]
    return r.drop_duplicates(["speaker_id", "window_start", "symptom_shortcode", "replicate"])


def ed2_parsing():
    """Extended Data Fig. 2: response parsing and window flow.

    a  parse route of all responses of the 491 users (log scale, percentages above the bars): valid
       JSON, JSON object embedded in other text, read by regular expressions, or unparseable (missing).
    b  distribution of the ratings of each route, in bins of 10 points centered on multiples of 10 (the
       ratings cluster there), with the mean rating of the route in the legend.
    c  windows remaining at each step: scored windows, windows of users with at least 200 rated
       window-symptom pairs, windows with all 52 symptoms rated, windows with confidence above 40 for
       every symptom; the title gives the share of all requested window-symptom pairs (scored windows x
       52) with five valid ratings.
    d  median and interquartile range over windows of the model confidence (mean of the five samples)
       for each symptom, sorted by the median; the dashed line marks the filter threshold of 40.
    Inputs: requests.parquet, 05_parse_report.json (key "collation"), summary.parquet (step 5),
    windows.parquet (step 3).  Returns the path of the PNG.
    """
    fig = plt.figure(figsize=(180 * MM, 125 * MM))
    gs = GridSpec(2, 3, figure=fig, wspace=0.55, hspace=0.62, left=0.07, right=0.985, top=0.9, bottom=0.2,
                  height_ratios=[1, 0.8])
    rep = jl(P.results_dir(args.run) / "05_parse_report.json")
    rq = P.processed_dir(args.run) / "requests.parquet"
    r = (cohort_responses(["speaker_id", "window_start", "symptom_shortcode", "replicate", "rating",
                           "parse_status"])
         if rq.exists() else None)
    # a parse routes
    ax = fig.add_subplot(gs[0, 0])
    if r is not None:
        vc = r["parse_status"].value_counts()
        x = np.arange(len(PARSE_ROUTES)); n = len(r)
        vals = [100 * vc.get(c, 0) / n for c, _ in PARSE_ROUTES]   # percentage of responses per route
        ax.bar(x, vals, width=0.6, color=PL.SLOTS[0], lw=0)
        for xv, v in zip(x, vals):
            ax.text(xv, v * 1.25, f"{v:.2f}%" if v < 1 else f"{v:.1f}%", ha="center", va="bottom", fontsize=5,
                    color=INK2)
        ax.set_yscale("log"); ax.set_ylim(0.05, 400)
        ax.set_xticks(x); ax.set_xticklabels([nm for _, nm in PARSE_ROUTES], fontsize=5.2)
        ax.set_ylabel("Responses (%)"); PL.hgrid(ax)
        ax.set_title(f"{n:,} responses", fontsize=5.5, color=INK2, pad=2)
    else:
        pending(ax, "requests (step 5)")
    lab(ax, "a", -0.45)
    # b rating distribution by parse route
    ax = fig.add_subplot(gs[0, 1])
    if r is not None:
        bins = np.arange(-5, 106, 10)                        # ratings cluster on multiples of 10
        for k, (c, nm) in enumerate(PARSE_ROUTES[:3]):       # unparseable responses have no rating
            v = r.loc[r["parse_status"] == c, "rating"].dropna().to_numpy(float)
            if len(v):
                h, _ = np.histogram(v, bins=bins)
                # percentage of the route's ratings in each bin, plotted at the bin centers
                ax.plot(bins[:-1] + 5, 100 * h / h.sum(), "-o", ms=2.2, color=PL.SLOTS[k], lw=0.9,
                        label=f"{nm.replace(chr(10), ' ').replace('- ', '')} (mean {v.mean():.0f})")
        ax.set_xlim(-4, 104); ax.set_xticks(range(0, 101, 20)); ax.set_xlabel("Rating (bins of 10 points)")
        ax.set_ylabel("Responses of each route (%)")
        ax.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2,
                  borderaxespad=0.2)
        PL.hgrid(ax)
    else:
        pending(ax, "requests (step 5)")
    lab(ax, "b", -0.45)
    # c window flow
    ax = fig.add_subplot(gs[0, 2])
    if rep and "collation" in rep:
        # scored windows of the 491 users (bot excluded), then the counts of the collation of step 5
        nwin = int((~pd.read_parquet(P.PROCESSED / "windows.parquet", columns=["speaker_id"])["speaker_id"]
                    .isin(P.BOT_SPEAKERS)).sum())
        steps = [(nwin, "scored"), (rep["collation"]["n_collated"], "users with\n≥ 200 ratings"),
                 (rep["collation"]["n_nona"], "all 52\nrated"),
                 (rep["collation"]["n_confident"], "confidence\n> 40")]
        x = np.arange(len(steps))
        ax.bar(x, [v for v, _ in steps], width=0.6, color=PL.SLOTS[0], lw=0)
        for xv, (v, _) in zip(x, steps):
            ax.text(xv, v + 150, f"{v:,}", ha="center", va="bottom", fontsize=5, color=INK2)
        ax.set_xticks(x); ax.set_xticklabels([nm for _, nm in steps], fontsize=5.0); ax.set_ylabel("Windows")
        ax.set_ylim(0, max(v for v, _ in steps) * 1.15); PL.hgrid(ax)
        if r is not None:                  # over all requested window–symptom pairs (scored windows × 52)
            # valid ratings per window and symptom; pairs with all five samples valid
            full = (r.dropna(subset=["rating"])
                    .groupby(["speaker_id", "window_start", "symptom_shortcode"]).size())
            ax.set_title(f"{100 * (full == 5).sum() / (nwin * 52):.1f}% of window–symptom pairs\n"
                         "have all five ratings", fontsize=5.3, color=INK2, pad=2)
    else:
        pending(ax, "parse report (step 5)")
    lab(ax, "c", -0.42)
    # d confidence by symptom (all 52)
    ax = fig.add_subplot(gs[1, :])
    sp = PC_ / "summary.parquet"
    if sp.exists():
        sm = pd.read_parquet(sp, columns=["symptom_shortcode", "confidence_mean"])
        # quartiles over windows of each symptom's confidence, symptoms sorted by the median
        q = (sm.groupby("symptom_shortcode")["confidence_mean"].quantile([0.25, 0.5, 0.75]).unstack()
             .reindex(codes))
        q = q.sort_values(0.5)
        x = np.arange(len(q))
        ax.vlines(x, q[0.25], q[0.75], color=PL.BLUE_RAMP[3], lw=1.6)
        ax.scatter(x, q[0.5], s=6, color=PL.SLOTS[0], lw=0, zorder=3)
        ax.axhline(P.CONFIDENCE_THRESH, color=MUTED, lw=0.6, ls=(0, (3, 2)))   # dashed: threshold of 40
        ax.set_xticks(x); ax.set_xticklabels([stitle[c] for c in q.index], rotation=90, fontsize=4.6)
        ax.set_xlim(-0.8, len(q) - 0.2)
        ax.set_ylabel("LLM confidence\n(median, IQR over windows)"); PL.hgrid(ax)
    else:
        pending(ax, "summary (step 5)")
    lab(ax, "d", -0.06, 1.04)
    return PL.save(fig, "ED_Fig2", args.subdir)


# ====================================================================== ED5
def ed5_factors():
    """Symptom correlations, replication of the factors, their composition and alternative numbers of factors.

    Extended Data Fig. 5:
    a  Spearman correlations between the 52 symptom ratings across confident windows (blue negative, red
       positive), symptoms ordered by average-linkage hierarchical clustering with correlation distance.
    b  Tucker congruence with the main loadings of the factors refitted in the users who wrote most of
       their utterances in r/depression (blue) or in r/Anxiety (vermillion); light bars give the range
       over ten random subsets of users of the same size as each group; dashed line 0.85, dotted 0.95.
    c  congruence with the main loadings of factors refitted at confidence thresholds of 30, 40 and 50
       (lowest error of five initializations): median (black) and minimum (vermillion) over factors; the
       least similar factor is named when its congruence is below 0.85; windows retained at each
       threshold at the bottom.
    d  composition of each factor (in the order of Table 1, with the square of its clinical family): its
       eight heaviest symptoms with their share of the factor's weight, shaded by where each falls in the
       cumulative weight (the darkest bars together carry the first 25%).  The full loading matrix is
       Fig. 2a.
    e  for each main factor, the Tucker congruence of its best match among the factors of the K = 10
       (open) and K = 18 (filled) solutions; dashed line 0.85.
    Inputs: collated_confident.parquet (step 5), 12c_community_congruence.csv, 12c_community.json
    (step 12c), 12_robustness.json (step 12, part C2), 06_nmf.npz, 06_nmf_k10.npz, 06_nmf_k18.npz
    (step 6).  Returns the path of the PNG.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from scipy.cluster.hierarchy import leaves_list, linkage
    fig = plt.figure(figsize=(180 * MM, 228 * MM))
    gs = GridSpec(2, 2, figure=fig, wspace=0.62, hspace=0.32, left=0.17, right=0.985, top=0.985, bottom=0.505,
                  width_ratios=[1.0, 0.8])
    conf = PC_ / "collated_confident.parquet"
    X = symptom_matrix(pd.read_parquet(conf))          # (windows, 52) mean ratings of the confident windows
    # a symptom correlations
    ax = fig.add_subplot(gs[:, 0])
    C = stats.spearmanr(X).statistic                   # (52, 52)
    # leaf order of the clustering of the symptoms (columns of X), correlation distance 1 - r
    order = leaves_list(linkage(X.T, "average", metric="correlation"))
    im = PL.heat(ax, C[np.ix_(order, order)], cmap=PL.div_cmap(), vmin=-1, vmax=1, edge=0)
    ax.set_xticks([]); ax.set_yticks(range(len(codes)))
    ax.set_yticklabels([PL.SYMPTOM_SHORT.get(codes[i], stitle[codes[i]]) for i in order], fontsize=5.0)
    ax.tick_params(length=0, pad=1)
    PL.no_frame(ax)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.03, pad=0.015, aspect=30)
    cb.outline.set_visible(False)
    cb.set_label("Spearman ρ between symptoms", fontsize=5.3)
    cb.ax.tick_params(labelsize=5, length=1.5)
    lab(ax, "a", -0.42, 1.0)
    # b replication in the two communities
    ax = fig.add_subplot(gs[0, 1])
    cc = pd.read_csv(RC_ / "12c_community_congruence.csv")   # group, kind, title, phi
    t0 = list(np.load(RC_ / "06_nmf.npz", allow_pickle=True)["titles"])
    t = [t0[j] for j in PL.family_order(t0)]; y = np.arange(len(t))        # the order of Fig. 2
    for k, (g, col) in enumerate([("r/depression", PL.BLUE), ("r/Anxiety", PL.VERMILLION)]):
        off = (k - 0.5) * 0.36
        # range of phi over the random subsets of the group's size (every kind other than "community")
        rnd = cc[(cc["group"] == g) & (cc["kind"] != "community")].groupby("title")["phi"]
        lo_, hi_ = rnd.min().reindex(t), rnd.max().reindex(t)
        ax.hlines(y + off, lo_, hi_, color="#c6dbef" if k == 0 else "#f4c3a5", lw=2.2, zorder=1)
        cm_ = cc[(cc["group"] == g) & (cc["kind"] == "community")].set_index("title")["phi"].reindex(t)
        nu = jl(RC_ / "12c_community.json").get("n_users", {}).get(g)
        ax.scatter(cm_, y + off, s=7, color=col, zorder=3, lw=0,
                   label=f"Mostly {g}" + (f" (n = {nu})" if nu else ""))
    ax.axvline(0.95, color=MUTED, lw=0.5, ls=(0, (1, 2)))
    ax.axvline(0.85, color=MUTED, lw=0.5, ls=(0, (3, 2)))
    ax.set_yticks(y); ax.set_yticklabels(t, fontsize=5.2)
    ax.set_ylim(len(t) - 0.5, -0.5); ax.set_xlim(0, 1.02)
    ax.tick_params(axis="y", length=0)
    PL.factor_tags(ax, t)
    ax.set_xlabel("Tucker φ with the main loadings")
    h_, l_ = ax.get_legend_handles_labels()
    h_.append(Line2D([], [], color="#bdbdbd", lw=2.2))
    l_.append("Range over 10 random\nsubsets of the same size")
    ax.legend(h_, l_, fontsize=4.9, loc="upper left", bbox_to_anchor=(0.0, 0.93), handlelength=1.2)
    lab(ax, "b", -0.78)
    # c confidence-threshold sensitivity
    ax = fig.add_subplot(gs[1, 1])
    rob = jl(RC_ / "12_robustness.json")
    c2 = pd.DataFrame(rob["C2"])                       # one row per threshold (30, 40, 50)
    ax.plot(c2["threshold"], c2["phi_median"], "-o", ms=2.4, color=INK, label="Median φ")
    ax.plot(c2["threshold"], c2["phi_min"], "-o", ms=2.4, color=PL.VERMILLION, label="Minimum φ")
    for _, r in c2.iterrows():
        ax.text(r["threshold"], 0.02, f"{int(r['n_windows']):,}\nwindows", fontsize=4.8, ha="center",
                color=INK2)
    ax.set_ylim(0, 1.02); ax.set_xticks(c2["threshold"]); ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Congruence with main loadings"); ax.legend(fontsize=5.0, loc="center right")
    for _, r in c2.iterrows():
        if r["phi_min"] < 0.85:                        # name the factor that did not replicate
            ax.annotate(r["phi_min_factor"], (r["threshold"], r["phi_min"]), xytext=(-5, -1),
                        textcoords="offset points", ha="right", va="top", fontsize=5, color=INK2)
    lab(ax, "c", -0.32)
    # d composition of each factor: its heaviest symptoms, shaded by the cumulative share of the weight
    nmf = np.load(RC_ / "06_nmf.npz", allow_pickle=True)
    H, FT = nmf["H"], list(nmf["titles"])
    Hn = H / H.sum(axis=1, keepdims=True)              # each factor's weights as shares summing to 1
    fam_order = ["Anxiety", "Social", "Positive affect", "Somatic", "Affective lability", "Cognitive load",
                 "Depressive affect", "Self-referential"]
    # factors by clinical family (the order of Table 1)
    f_order = sorted(range(len(FT)),
                     key=lambda j: (fam_order.index(PL.FAMILY.get(FT[j], ("Anxiety", 0))[0]), j))
    cum_bins = [(0.25, "First 25% of the weight"), (0.50, "25–50%"), (0.90, "50–90%"), (1.01, "Remaining 10%")]
    shades = ["#08306b", "#4a7fb5", "#a9c7e4", "#e3edf7"]
    n_show = 8                                         # symptoms shown per factor
    gd = GridSpec(1, 2, figure=fig, left=0.115, right=0.985, top=0.415, bottom=0.03, wspace=0.62,
                  width_ratios=[1.0, 0.26])
    sub = gd[0, 0].subgridspec(4, 3, wspace=1.15, hspace=0.62)   # 4 rows x 3 columns of factors
    for k, j in enumerate(f_order):
        ax = fig.add_subplot(sub[k // 3, k % 3])
        h = Hn[j]
        o = np.argsort(-h)                             # symptoms from the heaviest down
        cum = np.cumsum(h[o])
        # bin of the cumulative share before each item (+1e-9: an item starting on a boundary is in the next)
        shade = np.minimum(np.searchsorted([b_ for b_, _ in cum_bins], cum - h[o] + 1e-9, side="right"),
                           len(cum_bins) - 1)
        top = o[:n_show]; yy = np.arange(n_show)
        ax.barh(yy, h[top], height=0.78, color=[shades[q] for q in shade[:n_show]], lw=0)
        ax.set_yticks(yy)
        ax.set_yticklabels([PL.SYMPTOM_SHORT.get(codes[i], stitle[codes[i]]) for i in top], fontsize=4.7)
        ax.set_ylim(n_show - 0.4, -0.6); ax.set_xlim(0, 0.3); ax.set_xticks([0, 0.1, 0.2, 0.3])
        ax.set_xticklabels(["0", "0.1", "0.2", "0.3"]); ax.tick_params(axis="x", labelsize=4.8)
        ax.tick_params(axis="y", length=0, pad=1.0)
        ax.set_title(FT[j], fontsize=5.6, fontweight="bold", loc="left", pad=1.5, x=-0.02)
        # square in the color of the factor's family, left of the title
        ax.add_patch(plt.Rectangle((-0.1, 1.035), 0.065, 0.085, transform=ax.transAxes,
                                   fc=PL.factor_color(FT[j]), ec="none", clip_on=False))
        if k // 3 == 3:
            ax.set_xlabel("Share of the factor's weight", fontsize=5.2)
        if k == 0:
            lab(ax, "d", -1.25, 1.12)
    fig.legend(handles=[Patch(color=c, label=l) for c, (_b, l) in zip(shades, cum_bins)], loc="upper left",
               bbox_to_anchor=(0.12, 0.445), ncol=4, fontsize=5.0, handlelength=1.0)
    # e best match of each main factor among the factors of the K = 10 and K = 18 solutions
    ax = fig.add_subplot(gd[0, 1])
    rows = [FT[j] for j in f_order]
    nrm = lambda A: A / np.linalg.norm(A, axis=1, keepdims=True)  # noqa: E731
    yy = np.arange(len(rows))
    for k, (path, nm, col, mfc) in enumerate([(RC_ / "06_nmf_k10.npz", "K = 10", INK, "white"),
                                             (RC_ / "06_nmf_k18.npz", "K = 18", PL.VERMILLION,
                                              PL.VERMILLION)]):
        if not path.exists():
            continue
        Hk = np.load(path) if path.suffix == ".npy" else np.load(path, allow_pickle=True)["H"]
        # Tucker congruence = cosine of two loading vectors; best match of each main factor (not one-to-one)
        phi = (nrm(H) @ nrm(Hk).T).max(axis=1)
        ax.plot(phi[f_order], yy + (k - 0.5) * 0.3, "o", ms=2.4, mec=col, mfc=mfc, mew=0.6, ls="none",
                label=nm)
    ax.axvline(0.85, color=MUTED, lw=0.5, ls=(0, (3, 2)))
    ax.set_yticks(yy); ax.set_yticklabels(rows, fontsize=5.2); ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    PL.factor_tags(ax, rows)
    ax.set_xlim(0.6, 1.01); ax.set_xlabel("Tucker φ of the best match")
    ax.legend(fontsize=5.0, loc="lower left", handlelength=0.8)
    lab(ax, "e", -1.55, 1.03)
    return PL.save(fig, "ED_Fig5", args.subdir)


# ====================================================================== ED6
def ed6_psychometrics():
    """Extended Data Fig. 6: internal consistency under alternative estimators and confidence strata.

    One row per factor (squares give the clinical family); colored markers per estimator (legend under
    each panel).
    a  Cronbach's alpha: the weighted composite with prefactor k_eff/(k_eff - 1) (main estimate, blue);
       ordinary alpha over the items that carry 50 % of the weight (green); the weighted coefficient in
       windows in the top (orange) and bottom (vermillion) 10 % of factor-weighted confidence.
    b  item discrimination, the correlation of each item with the rest of the composite, averaged over
       the items that carry 50 % of the weight (blue) and over as many items with the lowest weights
       (vermillion).
    Input: 07_psychometrics.csv (step 7).  Returns the path of the PNG.
    """
    fig = plt.figure(figsize=(180 * MM, 80 * MM))
    gs = GridSpec(1, 2, figure=fig, wspace=0.1, left=0.16, right=0.985, top=0.95, bottom=0.3)
    if not (RC_ / "07_psychometrics.csv").exists():
        pending(fig.add_subplot(gs[0, :]), "psychometrics (step 7)")
        return PL.save(fig, "ED_Fig6", args.subdir)
    pc = pd.read_csv(RC_ / "07_psychometrics.csv").set_index("title")
    t0 = list(pc.index); t = [t0[j] for j in PL.family_order(t0)]; y = np.arange(len(t))   # the order of Fig. 2
    hi_lab, lo_lab = "windows in the top 10% of confidence", "windows in the bottom 10% of confidence"
    # (column prefix, x label, [(column, legend label, color), ...]) for each panel
    panels = [("alpha", "Cronbach's α",
               [("alpha", "weighted composite, prefactor k_eff/(k_eff − 1)", PL.SLOTS[0]),
                ("alpha_top50", "unweighted, items with 50% of the weight", PL.SLOTS[2]),
                ("alpha_hi", hi_lab, PL.SLOTS[3]),
                ("alpha_lo", lo_lab, PL.SLOTS[1])]),
              ("discrim", "Item discrimination (item–rest r)",
               [("discrim_rest_top", "items with 50% of the weight", PL.SLOTS[0]),
                ("discrim_rest_bottom", "same number, lowest weight", PL.SLOTS[1])])]
    for k, (key, name, series) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        ns = len(series)
        for m, (col, nm, color) in enumerate(series):
            if col not in pc:
                continue
            v = pc[col].reindex(t)
            # markers of the series side by side within each factor's row
            ax.plot(v, y + (m - (ns - 1) / 2) * 0.18, "o", ms=2.5, color=color, mec="white", mew=0.3,
                    label=nm, ls="none")
        ax.set_yticks(y)
        ax.set_yticklabels(t if k == 0 else [""] * len(t), fontsize=5.4)
        ax.set_ylim(len(t) - 0.5, -0.5)
        if k == 0:
            PL.factor_tags(ax, t)
        else:
            ax.tick_params(axis="y", length=0)
        ax.set_xlabel(name)
        ax.legend(fontsize=4.9, loc="upper center", bbox_to_anchor=(0.5, -0.17), handlelength=0.8)
        lab(ax, "ab"[k], -0.06 if k else -0.62)
    return PL.save(fig, "ED_Fig6", args.subdir)


# ====================================================================== ED7
def ed7_var():
    """Extended Data Fig. 7: vector autoregression within individuals.

    Panel VAR(1) within users on windows exactly 14 days apart, bias-corrected with the half-panel
    jackknife, with Wald tests based on a covariance from a bootstrap over users.  Rows and columns are in
    the order of Fig. 2 and carry the squares of the factors' clinical families.
    a  Granger predictive pairs (rows: response, columns: predictor): pale blue cells P < 0.05 without
       correction, blue cells FDR q < 0.05; the title gives both counts.
    b  response of each factor (rows) 24 weeks after a sustained change of STEP_SCALE points in another
       factor (columns), blue negative to red positive; dots mark responses whose 95 % bootstrap interval
       excludes zero (without correction for multiple comparisons).
    c  AIC and BIC of lag orders 1-3 on a common sample.
    d  distribution of the P values of the 132 lagged pairs (bins of 0.05); the dashed line marks the
       count per bin expected without any coupling (uniform P values).
    Inputs: 08_var_population.npy (key "corrected": the bias-corrected panel VAR), 08_var_order_ic.csv
    (step 8), 06_nmf.npz.  Returns the path of the PNG.
    """
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(180 * MM, 150 * MM))
    gt = GridSpec(1, 2, figure=fig, wspace=0.12, left=0.15, right=0.93, top=0.915, bottom=0.505)    # a, b
    gb = GridSpec(1, 2, figure=fig, wspace=0.32, left=0.085, right=0.97, top=0.33, bottom=0.075,    # c, d
                  width_ratios=[0.7, 1.3])
    p = RC_ / "08_var_population.npy"
    if not p.exists():
        pending(fig.add_subplot(gt[0, :]), "VAR (step 8)"); return PL.save(fig, "ED_Fig7", args.subdir)
    var = np.load(p, allow_pickle=True).item()
    titles0 = list(np.load(RC_ / "06_nmf.npz", allow_pickle=True)["titles"])
    o = PL.family_order(titles0)                       # rows and columns in the order of Fig. 2
    oo = np.ix_(o, o)
    titles = [titles0[j] for j in o]
    short = titles
    nT = len(titles)
    off = ~np.eye(nT, dtype=bool)                      # the 132 cross-lagged pairs
    cor = var["corrected"]
    # a Granger predictive pairs
    ax = fig.add_subplot(gt[0, 0])
    q = np.asarray(cor["q"], float)[oo].copy(); pv = np.asarray(cor["p"], float)[oo].copy()
    # 0: P >= 0.05; 1: P < 0.05 without FDR control; 2: FDR q < 0.05 (diagonal: NaN, drawn white)
    Mx = np.where(np.isfinite(q), (pv < 0.05).astype(float) + (q < P.FDR_Q).astype(float), np.nan)
    cm = plt.matplotlib.colors.ListedColormap(["#f3f2ee", PL.BLUE_RAMP[2], PL.SLOTS[0]]); cm.set_bad("#ffffff")
    ax.imshow(np.ma.masked_invalid(Mx), cmap=cm, vmin=0, vmax=2, interpolation="nearest")
    ax.set_xticks(range(nT)); ax.set_xticklabels(short, rotation=90, fontsize=5)
    ax.set_yticks(range(nT)); ax.set_yticklabels(short, fontsize=5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    PL.factor_tags(ax, titles, axis="y", size_pt=3.6, gap_pt=1.0, label_gap_pt=1.5)
    PL.factor_tags(ax, titles, axis="x", size_pt=3.6, gap_pt=1.0, label_gap_pt=1.5)
    ax.set_title(f"Panel VAR({cor['order']}) corrected for bias, {int(np.nansum((pv < 0.05) & off))} pairs "
                 f"P < 0.05, {int(np.nansum((q < P.FDR_Q) & off))} at q < {P.FDR_Q}", fontsize=5.3, color=INK2,
                 pad=3)
    ax.set_ylabel("Response (row) ← predictor (column)", fontsize=5.3)
    fig.legend(handles=[Patch(color=PL.BLUE_RAMP[2], label="P < 0.05, uncorrected"),
                        Patch(color=PL.SLOTS[0], label=f"FDR q < {P.FDR_Q}")],
               fontsize=5, loc="upper left", bbox_to_anchor=(0.15, 0.995), ncol=2, handlelength=0.9)
    lab(ax, "a", -0.5, 1.08)
    # b step responses at 24 weeks
    ax = fig.add_subplot(gt[0, 1])
    cum = np.asarray(cor["cum"][-1])[oo] * P.STEP_SCALE                # cumulative response at the last horizon, [to, from]
    Mx = cum.copy(); np.fill_diagonal(Mx, np.nan)
    v = float(np.nanmax(np.abs(Mx))) or 1.0            # symmetric color range
    cmd = PL.div_cmap(); cmd.set_bad("#ffffff")
    im = ax.imshow(np.ma.masked_invalid(Mx), cmap=cmd, vmin=-v, vmax=v, interpolation="nearest")
    lo_, hi_ = np.asarray(cor["cum_lo"][-1])[oo], np.asarray(cor["cum_hi"][-1])[oo]
    yy, xx = np.where(((lo_ > 0) | (hi_ < 0)) & ~np.eye(nT, dtype=bool))   # interval excludes zero
    ax.scatter(xx, yy, s=2.4, color=INK, marker="o", lw=0)
    ax.set_xticks(range(nT)); ax.set_xticklabels(short, rotation=90, fontsize=5)
    ax.set_yticks(range(nT)); ax.set_yticklabels([""] * nT)
    for sp in ax.spines.values():
        sp.set_visible(False)
    PL.factor_tags(ax, titles, axis="y", size_pt=3.6, gap_pt=1.0, label_gap_pt=1.5)
    PL.factor_tags(ax, titles, axis="x", size_pt=3.6, gap_pt=1.0, label_gap_pt=1.5)
    ax.set_title(f"Response 24 weeks after a sustained change of {P.STEP_SCALE:.0f} points in the column factor",
                 fontsize=5.3, color=INK2, pad=3)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.outline.set_visible(False)
    cb.ax.tick_params(length=1.5, labelsize=5); cb.set_label("Change (points)", fontsize=5.2)
    lab(ax, "b", -0.08, 1.08)
    # c order selection
    ax = fig.add_subplot(gb[0, 0])
    icp = RC_ / "08_var_order_ic.csv"
    if icp.exists():
        ic = pd.read_csv(icp)                          # p, aic, bic, n
        ax.plot(ic["p"], ic["bic"], "-o", ms=2.6, color=PL.SLOTS[0], label="BIC")
        ax.plot(ic["p"], ic["aic"], "-o", ms=2.6, color=PL.SLOTS[1], label="AIC")
        ax.set_xticks(ic["p"]); ax.set_xlabel("Lag order p"); ax.set_ylabel("Information criterion")
        ax.legend(fontsize=5.2)
        ax.set_title(f"Common sample, n = {int(ic['n'].iloc[0]):,}", fontsize=5.3, color=INK2, pad=2)
    lab(ax, "c", -0.45, 1.08)
    # d distribution of the cross-lagged P values
    ax = fig.add_subplot(gb[0, 1])
    bins = np.linspace(0, 1, 21)                       # 20 bins of 0.05
    pvv = pv[off]; pvv = pvv[np.isfinite(pvv)]
    h, _ = np.histogram(pvv, bins=bins)
    ax.step(bins[:-1] + 0.025, h, where="mid", color=PL.SLOTS[0], lw=1.0,
            label=f"Observed ({int((pvv < 0.05).sum())} below 0.05)")
    ax.axhline(off.sum() / (len(bins) - 1), color=MUTED, lw=0.7, ls=(0, (3, 2)),
               label=f"Expected without coupling ({off.sum() / (len(bins) - 1):.1f} per bin)")
    ax.set_xlim(0, 1); ax.set_xlabel("P value of the 132 lagged pairs"); ax.set_ylabel("Pairs per 0.05 bin")
    ax.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.4,
              columnspacing=1.0)
    lab(ax, "d", -0.2, 1.08)
    return PL.save(fig, "ED_Fig7", args.subdir)


# ====================================================================== ED8
def ed8_states():
    """Extended Data Fig. 8: selection of the number of discrete states.

    a  AIC, AICc and BIC across the 14 hyperparameter steps of the Dirichlet process mixture, computed
       with the parameter count of a Gaussian mixture with full covariances and the retained components
       and with the number of windows as the sample size; each criterion is scaled to range from 0 to 1
       and steps run from the least to the most regularized setting; the vertical line marks the selected
       step (BIC minimum).
    b  number of retained states at each step, with the selected step annotated.
    Inputs: 10_dpmm_sweep.csv and 10_states.json (step 10).  Returns the path of the PNG.
    """
    fig = plt.figure(figsize=(180 * MM, 58 * MM))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35, left=0.07, right=0.8, top=0.9, bottom=0.2)
    p = RC_ / "10_dpmm_sweep.csv"
    if not p.exists():
        pending(fig.add_subplot(gs[0, :]), "DPMM sweep (step 10)")
        return PL.save(fig, "ED_Fig8", args.subdir)
    sw = pd.read_csv(p)
    js = jl(RC_ / "10_states.json")
    sel = js.get("selected_step", P.DPMM_STEP_USED)
    ax = fig.add_subplot(gs[0, 0])
    for m, (c, nm) in enumerate([("aic", "AIC"), ("aicc", "AICc"), ("bic", "BIC")]):
        y = sw[c]
        # min-max scaling, so the three criteria share one axis (1e-12 avoids a division by zero)
        ax.plot(sw["step"], (y - y.min()) / (y.max() - y.min() + 1e-12), "-o", ms=2.2, color=PL.SLOTS[m],
                lw=1.0, label=nm)
    ax.axvline(sel, color=MUTED, lw=0.6)
    ax.set_xlabel("Hyperparameter step (more regularized →)"); ax.set_ylabel("Criterion (min–max scaled)")
    ax.set_xticks(sw["step"]); ax.tick_params(axis="x", labelsize=5)
    ax.legend(fontsize=5.2, loc="upper left", bbox_to_anchor=(1.0, 1.0)); PL.hgrid(ax)
    lab(ax, "a", -0.22)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(sw["step"], sw["n_states"], "-o", ms=2.4, color=PL.SLOTS[0])
    ax.axvline(sel, color=MUTED, lw=0.6)
    ns = int(sw.loc[sw["step"] == sel, "n_states"].iloc[0])
    ax.annotate(f"selected step {sel}, {ns} states", (sel, ns), xytext=(8, 14), textcoords="offset points",
                fontsize=5.2, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    ax.set_xlabel("Hyperparameter step (more regularized →)"); ax.set_ylabel("Retained states")
    ax.set_xticks(sw["step"]); ax.tick_params(axis="x", labelsize=5); PL.hgrid(ax)
    lab(ax, "b", -0.22)
    return PL.save(fig, "ED_Fig8", args.subdir)


# ====================================================================== ED9
def ed9_regimes():
    """Extended Data Fig. 9: model selection, sensitivity and description of the dynamical regimes.

    a  AIC, AICc and BIC of mixtures of Markov chains with 1 to 20 regimes, fitted with hard assignments,
       each scaled to range from 0 to 1.
    b  advantage in cross-validated log likelihood per window of Markov mixtures over occupancy mixtures
       by number of components, for the main solution (11 states) and for the eight states of the
       adjacent hyperparameter step; dashed lines show the same after shuffling each user's windows in
       time; the horizontal line marks zero.
    c  symptom burden: for each factor, the share of each user's windows at or above the 90th percentile
       of Regime 1, averaged within each regime.
    d  transition matrices of the regimes, P(to | from), estimated by the hard-assignment EM from
       consecutive confident windows (a gap of up to three missing windows counts as one step); titles
       give the share of users in each regime.
    Inputs: 11_regime_sweep.csv, 11_regime_burden.csv, 11_regimes.npy (step 11), 12D_heldout.csv,
    12D_heldout_8states.csv (step 12), 10_states.json, 10_states_8states.json (steps 10, 10b, 10c),
    06_nmf.npz.  Returns the path of the PNG.
    """
    rp = RC_ / "11_regimes.npy"
    reg = np.load(rp, allow_pickle=True).item() if rp.exists() else None
    K = int(reg["K"]) if reg else 1                    # number of regimes (AICc minimum of step 11)
    fig = plt.figure(figsize=(180 * MM, 118 * MM))
    gt = GridSpec(1, 3, figure=fig, wspace=0.7, left=0.07, right=0.97, top=0.94, bottom=0.6,
                  width_ratios=[1, 1.15, 0.9])
    # panel d: one transition matrix per regime
    gb = GridSpec(1, K, figure=fig, wspace=0.12, left=0.1, right=0.97, top=0.4, bottom=0.12)
    # a information criteria of the regime sweep
    ax = fig.add_subplot(gt[0, 0])
    p = RC_ / "11_regime_sweep.csv"
    if p.exists():
        sw = pd.read_csv(p)
        for m, c in enumerate(["aic", "aicc", "bic"]):
            y = sw[c]; ax.plot(sw["n_regimes"], (y - y.min()) / (y.max() - y.min() + 1e-12), "-", lw=1.0,
                               color=PL.SLOTS[m], label=c.upper().replace("AICC", "AICc"))
        ax.set_xlabel("Regimes (hard EM)"); ax.set_ylabel("Criterion (min–max scaled)")
        ax.legend(fontsize=5.2)
        ax.set_xticks([1, 5, 10, 15, 20]); PL.hgrid(ax)
    lab(ax, "a", -0.35)
    # b Markov minus occupancy-only, held-out, for two discretizations
    ax = fig.add_subplot(gt[0, 1])
    n_main = len(jl(RC_ / "10_states.json").get("state_names", [])) or "main"
    n_alt = len(jl(RC_ / "10_states_8states.json").get("state_names", [])) or 8
    series = [(RC_ / "12D_heldout.csv", f"{n_main} states (main)", PL.SLOTS[0]),
              (RC_ / "12D_heldout_8states.csv", f"{n_alt} states (adjacent step)", PL.SLOTS[1])]
    any_ = False
    for f_, nm, col in series:
        if f_.exists():
            cv = pd.read_csv(f_)
            ax.plot(cv["K"], cv["markov"] - cv["occupancy"], "-o", ms=2.3, lw=1.0, color=col, label=nm)
            ax.plot(cv["K"], cv["markov_shuffled"] - cv["occupancy_shuffled"], ls=(0, (3, 2)), lw=0.8,
                    color=col)
            any_ = True
    if any_:
        ax.axhline(0, color=AXIS, lw=0.6)
        ax.set_xlabel("Mixture components")
        ax.set_ylabel("Markov minus occupancy\ncross-validated log lik. per window")
        ax.set_xticks(range(1, 9)); PL.hgrid(ax)
        from matplotlib.lines import Line2D
        h_, l_ = ax.get_legend_handles_labels()
        h_.append(Line2D([], [], color=INK2, lw=0.8, ls=(0, (3, 2))))
        l_.append("windows shuffled within person")
        ax.legend(h_, l_, fontsize=5, loc="upper right", handlelength=1.6, borderaxespad=0.2)
    else:
        pending(ax, "held-out likelihood (step 12 D)")
    lab(ax, "b", -0.4)
    # c burden by regime
    ax = fig.add_subplot(gt[0, 2])
    bp = RC_ / "11_regime_burden.csv"
    if bp.exists() and reg is not None:
        b = pd.read_csv(bp, index_col=0)               # rows Regime 1..K, one column per factor
        t0 = list(np.load(RC_ / "06_nmf.npz", allow_pickle=True)["titles"])
        titles = [t0[j] for j in PL.family_order(t0)]                     # the order of Fig. 2
        B = b[titles].to_numpy(float)
        # factors as rows, regimes as columns; color range at least 0-0.3
        im = ax.imshow(B.T, aspect="auto", cmap=PL.seq_cmap(), vmin=0, vmax=max(0.3, float(np.nanmax(B))),
                       interpolation="nearest")
        ax.set_yticks(range(len(titles))); ax.set_yticklabels(titles, fontsize=5)
        ax.set_xticks(range(K)); ax.set_xticklabels([f"R{r+1}" for r in range(K)], fontsize=5.5)
        ax.tick_params(length=0, pad=1)
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.04); cb.outline.set_visible(False)
        cb.ax.tick_params(length=1.5, labelsize=5)
        cb.set_label("Share of windows ≥ R1's\n90th percentile", fontsize=5.2)
    lab(ax, "c", -1.0)
    # d regime transition matrices
    if reg is not None:
        names = list(reg["state_names"]); S = len(names)
        vmax = float(np.max(reg["A"]))                 # one color scale for all regimes
        for r in range(K):
            a = fig.add_subplot(gb[0, r])
            im = a.imshow(reg["A"][r], cmap=PL.seq_cmap(), vmin=0, vmax=vmax, interpolation="nearest")
            a.set_title(f"Regime {r+1} ({100*reg['frac'][r]:.0f}% of users)", fontsize=5.6, pad=3, color=INK)
            a.set_xticks(range(S)); a.set_yticks(range(S))
            a.set_xticklabels(names, rotation=90, fontsize=5)
            a.set_yticklabels(names if r == 0 else [""] * S, fontsize=5)
            a.tick_params(length=0, pad=1)
            for sp in a.spines.values():
                sp.set_visible(False)
            if r == 0:
                a.set_ylabel("To (window t + 1)", fontsize=5.3)
                lab(a, "d", -0.62, 1.1)
            a.set_xlabel("From (window t)", fontsize=5.3)
        cb = fig.colorbar(im, ax=fig.axes[-K:], fraction=0.012, pad=0.01); cb.outline.set_visible(False)
        cb.ax.tick_params(length=1.5, labelsize=5); cb.set_label("P(to | from)", fontsize=5.2)
    return PL.save(fig, "ED_Fig9", args.subdir)


# ====================================================================== ED10
def ed10_sequences():
    """Extended Data Fig. 10: example state sequences and a hidden Markov model.

    a  state sequences of six randomly chosen users in each regime (fixed seed): rows are users, columns
       are windows of two weeks counted from the user's first confident window, and white cells mark
       windows without a confident rating; the eight most common states have their own colors and the
       remaining states share one gray.
    b  a Gaussian hidden Markov model with as many states as the mixture, fitted to the factor scores of
       contiguous runs of windows: its adjusted Rand index against the states of the Dirichlet process
       mixture and its mean probability of remaining in the same state.
    Inputs: 11_regimes.npy (step 11), 12_robustness.json (step 12, part E).  Returns the path of the PNG.
    """
    fig = plt.figure(figsize=(180 * MM, 80 * MM))
    rp = RC_ / "11_regimes.npy"
    if not rp.exists():
        gs = GridSpec(1, 1, figure=fig); pending(fig.add_subplot(gs[0, 0]), "regimes (step 11)")
        return PL.save(fig, "ED_Fig10", args.subdir)
    reg = np.load(rp, allow_pickle=True).item()
    K = int(reg["K"]); names = list(reg["state_names"]); S = len(names)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2.6, 1], wspace=0.25, left=0.07, right=0.985, top=0.9,
                  bottom=0.12)
    sub = gs[0, 0].subgridspec(K, 1, hspace=0.6)
    rng = np.random.RandomState(1)                     # choice of the example users (display only)
    # the eight most common states get their own colors; further states share one gray ("other states")
    st_cols = [PL.SLOTS[i] if i < len(PL.SLOTS) else "#b4b2a9" for i in range(S)]
    cm = plt.matplotlib.colors.ListedColormap(st_cols)
    for r in range(K):
        a = fig.add_subplot(sub[r, 0])
        idx = [i for i, z in enumerate(reg["z"]) if z == r]       # users (sequences) of regime r
        pick = rng.choice(idx, size=min(6, len(idx)), replace=False) if idx else []
        # users x 27 window positions; positions without a confident window stay NaN (white)
        M = np.full((len(pick), 27), np.nan)
        for k, i in enumerate(pick):
            for s, t in zip(reg["seqs"][i], reg["steps"][i]):
                if t < 27:
                    M[k, t] = s
        a.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cm, vmin=-0.5, vmax=S - 0.5,
                 interpolation="nearest")
        a.set_yticks([]); a.set_xticks([] if r < K - 1 else [0, 6, 13, 20, 26])
        if r == K - 1:
            a.set_xticklabels(["0", "12", "26", "40", "52"])
            a.set_xlabel("Weeks since the first confident window")
        a.set_ylabel(f"R{r+1}", rotation=0, ha="right", va="center", fontsize=6)
        for sp in a.spines.values():
            sp.set_visible(False)
        if r == 0:
            a.set_title("Six random users per regime (columns are windows, white cells have no "
                        "confident window)", fontsize=5.5, color=INK2, pad=2)
            a.text(-0.06, 1.25, "a", transform=a.transAxes, fontsize=8, fontweight="bold")
    from matplotlib.patches import Patch
    handles = [Patch(color=st_cols[i], label=n) for i, n in enumerate(names[:len(PL.SLOTS)])]
    if S > len(PL.SLOTS):
        handles.append(Patch(color="#b4b2a9", label="other states"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.07, -0.02), ncol=len(handles), fontsize=5,
               handlelength=1, frameon=False)
    ax = fig.add_subplot(gs[0, 1])
    rob = jl(RC_ / "12_robustness.json")
    if "E" in rob:
        e = rob["E"]
        ax.bar([0, 1], [e["ari_hmm_vs_dpmm"], e["hmm_mean_self_transition"]], color=PL.SLOTS[0], width=0.6,
               lw=0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["ARI, HMM\nvs DPMM states", "HMM mean\nP(stay)"], fontsize=5.3)
        ax.set_ylim(0, 1); PL.hgrid(ax)
        ax.set_title(f"Gaussian HMM on {e['n_windows']:,} windows\nin {e['n_runs']:,} contiguous runs",
                     fontsize=5.3, color=INK2, pad=2)
    else:
        pending(ax, "HMM (step 12 E)")
    lab(ax, "b", -0.35)
    return PL.save(fig, "ED_Fig10", args.subdir)


# ====================================================================== ED3
def validation_tags():
    """Result tags of the reduced protocol run on a single local computer (results/validation/14b_<tag>_*).

    Returns (ref, others): ref is the Llama-3.1-8B-Instruct run served by llama.cpp (or by Ollama when only
    that run exists, else the first tag); others are the runs of other models.  Further Llama-3.1-8B runs
    and tags starting with "fake" (pipeline tests without a model) are left out.
    """
    import glob as _g
    tags = sorted(x.split("14b_")[1].rsplit("_metrics", 1)[0]
                  for x in _g.glob(str(P.RESULTS / "validation" / "14b_*_metrics.csv")))
    ref = next((t for t in ["llama31_8b_q4km_llamacpp", "llama31_8b_q4km_ollama"] if t in tags),
               tags[0] if tags else None)
    others = [t for t in tags if t != ref and not t.startswith("llama31_8b_q4km")
              and not t.startswith("fake")]
    return ref, others


# (dataset, label) of the validation data -> panel title
LABEL_NAMES = {("irf", "tbe"): "Thwarted belongingness", ("irf", "pbu"): "Perceived burdensomeness",
               ("lost", "low_self_esteem"): "Low self-esteem",
               ("severity", "severity"): "Depression severity"}


def ed3_validation():
    """Extended Data Fig. 3: prediction of human annotations in detail, from the reduced protocol.

    Panels a-d show Llama-3.1-8B-Instruct, the annotating model of the main analyses, run on a single
    local computer.  Panels are placed in millimetres (ax_mm) so that the long symptom names fit.
    a  weight of every scored symptom in the lasso of step 14d for each label (mean standardized coefficient
       over the 50 training folds: log odds per standard deviation of the rating for the binary labels,
       severity levels per standard deviation for severity), one small panel per label in the order of
       Fig. 1d; blue when the lasso kept the symptom in at least 80 % of the training folds, gray otherwise;
       a dagger marks the control symptoms.
    b  cross-validated lasso prediction of the severity level (0 minimum to 3 severe, step 14d) by
       annotated level: posts as jittered dots, median (horizontal bar) and interquartile range.
    c  agreement (ICC(A,1)) of one sample of the local deployment with the mean of four samples of the
       cohort annotation for the same calibration windows, per symptom and over all symptoms (filled),
       and the ceiling: one cohort sample against the mean of the other four (open).
    d  local rating against the mean of the five samples of the cohort annotation, all eight calibration
       symptoms; the line marks identity.
    e  prediction by the cross-validated lasso for each annotating model, AUROC with 95 % bootstrap
       interval (severity: severe versus minimum depression).
    Inputs: results/validation/14b_<model>_per_symptom.csv, _calibration.csv, _calibration_pairs.csv,
    _summary.json (step 14b); 14d_<model>_selection.csv, _predictions.csv, _predictive.csv (step 14d).
    Returns the path of the PNG.
    """
    ref, others = validation_tags()
    W_, H_ = 180, 150                                         # figure size in mm; panels are placed in mm
    fig = plt.figure(figsize=(W_ * MM, H_ * MM))

    def ax_mm(x, y, w, h):
        return fig.add_axes([x / W_, y / H_, w / W_, h / H_])

    V = P.RESULTS / "validation"
    if ref is None:
        pending(ax_mm(10, 10, 160, 130), "reduced-protocol validation (step 14b; needs local_validation outputs)")
        return PL.save(fig, "ED_Fig3", args.subdir)
    ps = pd.read_csv(V / f"14b_{ref}_per_symptom.csv")
    sel = pd.read_csv(V / f"14d_{ref}_selection.csv")          # lasso selection frequency and weights (step 14d)
    order3 = [("irf", "pbu"), ("irf", "tbe"), ("lost", "low_self_esteem"), ("severity", "severity")]  # as in Fig. 1d
    labs = [k for k in order3 if ((sel["dataset"] == k[0]) & (sel["label"] == k[1])).any()]
    # a: weight of every scored symptom in the lasso of each label (mean standardized coefficient over the
    # 50 training folds); blue when kept in at least 80% of the folds, daggers mark the control symptoms
    names_ = {c: PL.SYMPTOM_SHORT.get(c, stitle[c]) if len(stitle[c]) > 26 else stitle[c] for c in codes}
    xs_mm = [27, 70, 113, 156]
    for k, (ds, lb) in enumerate(labs):
        g = sel[(sel["dataset"] == ds) & (sel["label"] == lb)].copy()
        roles = ps[(ps["dataset"] == ds) & (ps["label"] == lb)].set_index("symptom")["role"]
        g["control"] = g["symptom"].map(roles).eq("control")
        g = g.sort_values("mean_coef", ascending=False)
        ax = ax_mm(xs_mm[k], 82, 17, 58)
        yy = np.arange(len(g))
        kept = g["selected_frac"].to_numpy() >= 0.8
        ax.barh(yy, g["mean_coef"], height=0.72, color=[PL.SLOTS[0] if k_ else "#c9c9c9" for k_ in kept], lw=0)
        ax.axvline(0, color=AXIS, lw=0.5)
        ax.set_yticks(yy)
        ax.set_yticklabels([names_.get(c, c) + (" †" if ctl else "") for c, ctl in zip(g["symptom"], g["control"])],
                           fontsize=5.0)
        ax.set_ylim(len(g) - 0.5, -0.5)
        ax.tick_params(axis="y", length=0, pad=1.5)
        ax.spines["left"].set_visible(False)
        lim = max(0.3, float(np.ceil(g["mean_coef"].abs().max() * 10) / 10))
        ax.set_xlim(-lim if g["mean_coef"].min() < 0 else -0.05 * lim, lim)
        ax.locator_params(axis="x", nbins=3)
        ax.set_xlabel("Weight (levels per s.d.)" if ds == "severity" else "Weight (log odds per s.d.)", fontsize=5.2)
        ax.set_title(LABEL_NAMES[(ds, lb)], fontsize=5.6, fontweight="bold", loc="left", x=-0.02, pad=3)
        if k == 0:
            lab(ax, "a", -1.45, 1.05)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=PL.SLOTS[0], label="kept by the lasso in at least 80% of the training folds"),
                        Patch(color="#c9c9c9", label="kept less often"),
                        Patch(color="none", label="† control symptom")],
               fontsize=5.2, loc="upper left", bbox_to_anchor=(0.15, 1.0), ncol=3, handlelength=1.0,
               columnspacing=1.6)
    # b: cross-validated lasso prediction of the severity level, by annotated level (step 14d)
    ax = ax_mm(13, 22, 28, 44)
    f_pred = V / f"14d_{ref}_predictions.csv"
    if f_pred.exists():
        pr = pd.read_csv(f_pred)
        g = pr[(pr["dataset"] == "severity") & (pr["model"] == "lasso")]
        levels = ["Minimum", "Mild", "Moderate", "Severe"]
        for k in range(4):
            v = g.loc[g["y"] == k, "prediction"].to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            ax.scatter(np.full(len(v), k) + rng_jit(len(v), 0.22), v, s=2, color=PL.SLOTS[0], alpha=0.35, lw=0)
            q1, q2, q3 = np.quantile(v, [.25, .5, .75])
            ax.plot([k - 0.3, k + 0.3], [q2, q2], color=INK, lw=1.0)
            ax.plot([k, k], [q1, q3], color=INK, lw=0.8)
        ax.set_xticks(range(4)); ax.set_xticklabels(levels, fontsize=5.2, rotation=30, ha="right",
                                                    rotation_mode="anchor")
        ax.set_xlabel("Annotated severity"); ax.set_ylabel("Predicted severity level\n(lasso, cross-validated)")
        PL.hgrid(ax)
    else:
        pending(ax, "predictions (step 14d)")
    lab(ax, "b", -0.5, 1.03)
    # c: calibration per symptom against the ceiling
    ax = ax_mm(71, 22, 22, 44)
    f_cal = V / f"14b_{ref}_calibration.csv"
    if f_cal.exists() and len(pd.read_csv(f_cal)):
        cal = pd.read_csv(f_cal)
        like = cal["comparison"] == "Local (1 sample) vs mean of 4 cohort samples"     # like for like with the ceiling
        loc_ = cal[like if like.any() else cal["comparison"].str.startswith("Local")].drop_duplicates("symptom").set_index("symptom")
        ceil = cal[cal["comparison"].str.startswith("Cohort")].set_index("symptom")
        order = [x for x in loc_.sort_values("icc_a1").index if x != "all symptoms"] + ["all symptoms"]
        yy = np.arange(len(order))
        ax.hlines(yy, ceil.loc[order, "icc_a1"], loc_.loc[order, "icc_a1"], color=GRID, lw=0.8)
        ax.scatter(ceil.loc[order, "icc_a1"], yy, s=8, facecolor="white", edgecolor=MUTED, lw=0.7, zorder=3,
                   label="one cohort sample against the other four")
        ax.scatter(loc_.loc[order, "icc_a1"], yy, s=9, color=PL.SLOTS[0], lw=0, zorder=4,
                   label="local deployment, one sample against\nthe mean of four cohort samples")
        ax.set_yticks(yy)
        ax.set_yticklabels([("All symptoms" if x == "all symptoms" else stitle.get(x, x)) for x in order], fontsize=5.0)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_xlabel("ICC(A,1)")
        ax.legend(fontsize=4.9, loc="upper left", bbox_to_anchor=(-1.25, -0.16), handletextpad=0.3)
    else:
        pending(ax, "calibration windows")
    lab(ax, "c", -1.3, 1.03)
    # d: calibration scatter
    ax = ax_mm(107, 22, 28, 44)
    f_pairs = V / f"14b_{ref}_calibration_pairs.csv"
    if f_pairs.exists():
        pr = pd.read_csv(f_pairs)
        ax.scatter(pr["cohort_mean"], pr["local"], s=3, color=PL.SLOTS[0], alpha=0.4, lw=0)
        ax.plot([0, 100], [0, 100], color=AXIS, lw=0.6)
        ax.set_xlim(-2, 102); ax.set_ylim(-2, 102); ax.set_xticks([0, 50, 100]); ax.set_yticks([0, 50, 100])
        ax.set_xlabel("Cohort annotation\n(mean of five samples)"); ax.set_ylabel("Local deployment (one sample)")
        PL.hgrid(ax)
    else:
        pending(ax, "calibration pairs")
    lab(ax, "d", -0.42, 1.03)
    # e: annotating models, cross-validated lasso of step 14d
    ax = ax_mm(155, 22, 22, 44)
    if others:
        tags = [ref] + others
        cols = {t: PL.SLOTS[i % 8] for i, t in enumerate(tags)}
        lo_all = []
        for i, t in enumerate(tags):
            if not (V / f"14d_{t}_predictive.csv").exists():
                continue
            m = pd.read_csv(V / f"14d_{t}_predictive.csv")
            m = m[m["model"] == "lasso"]
            for k, (ds, lb) in enumerate(labs):
                r = m[(m["dataset"] == ds) & (m["label"] == lb)]
                if not len(r):
                    continue
                r = r.iloc[0]
                lo_all.append(float(r["auroc_ci_lo"]))
                yv = k + (i - (len(tags) - 1) / 2) * 0.2
                ax.plot([r["auroc_ci_lo"], r["auroc_ci_hi"]], [yv, yv], color=cols[t], lw=0.8)
                nm_ = (load_json(V / f"14b_{t}_summary.json").get("model_name") if (V / f"14b_{t}_summary.json").exists()
                       else None) or P.MODEL_NAMES.get(t, t)
                ax.plot(r["auroc"], yv, "o", ms=2.8, color=cols[t], mec="white", mew=0.3, label=nm_ if k == 0 else None)
        ylab = {("severity", "severity"): "Severe vs\nminimum\ndepression"}
        ax.set_yticks(range(len(labs)))
        ax.set_yticklabels([ylab.get(x, LABEL_NAMES[x].replace(" ", "\n", 1)) for x in labs], fontsize=5.0)
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis(); ax.set_xlim(0.5, 1); ax.set_xticks([0.5, 0.75, 1]); ax.set_xticklabels(["0.5", "0.75", "1"])
        ax.set_xlabel("AUROC (95% CI)")
        ax.legend(fontsize=4.9, loc="upper left", bbox_to_anchor=(-0.75, -0.16), handletextpad=0.3)
        lab(ax, "e", -0.85, 1.03)
    else:
        ax.set_axis_off()
    return PL.save(fig, "ED_Fig3", args.subdir)


def rng_jit(n, s=0.18):
    """Horizontal jitter for n points, uniform in (-s, s); seeded by n, so it is the same on every run."""
    return np.random.RandomState(n).uniform(-s, s, n)


# ====================================================================== ED4
def ed4_inventories():
    """TALQ battery versus BDI-II and PHQ-9 administered by the same model (step 18).

    Extended Data Fig. 4:
    a  BDI-II total (21 items, each the mean of five samples, answered in the first person) against the
       TALQ analog of the BDI-II, the mean of the mapped battery items, for the linked windows with
       complete battery ratings; the text gives Spearman rho pooled and within person and the numbers of
       windows and users.
    b  the same for the PHQ-9.
    c  Spearman correlation of each inventory item with its mapped battery item or items (filled) and its
       median correlation with the other battery items (open).
    d  Spearman correlation within users of each factor score with the BDI-II and PHQ-9 totals, after
       centering within each user, factors sorted by the PHQ-9 correlation.
    Inputs: data/processed/inventories.parquet, results/inventories/18_totals.csv, 18_items.csv (step 18),
    collated_nona.parquet (step 5), config/inventory_item_mapping.yaml.  Returns the path of the PNG.
    """
    d = P.RESULTS / "inventories"
    fig = plt.figure(figsize=(180 * MM, 105 * MM))
    gs = GridSpec(2, 3, figure=fig, wspace=0.95, hspace=0.55, left=0.075, right=0.985, top=0.93, bottom=0.1,
                  width_ratios=[1, 1.35, 1.2])
    if not (d / "18_totals.csv").exists():
        pending(fig.add_subplot(gs[0, :]), "inventory comparison (step 18)")
        return PL.save(fig, "ED_Fig4", args.subdir)
    tot = pd.read_csv(d / "18_totals.csv"); items = pd.read_csv(d / "18_items.csv")
    inv = pd.read_parquet(P.PROCESSED / "inventories.parquet")       # totals of the linked windows
    coll = pd.read_parquet(PC_ / "collated_nona.parquet")             # windows with complete battery ratings
    # common key types for the merge
    coll["speaker_id"] = coll["speaker_id"].astype(str)
    coll["window_start"] = pd.to_datetime(coll["window_start"]).astype("datetime64[ns]")
    inv["window_start"] = pd.to_datetime(inv["window_start"]).astype("datetime64[ns]")
    j = inv.merge(coll, on=["speaker_id", "window_start"])
    import yaml
    mp = yaml.safe_load(open(P.CONFIG / "inventory_item_mapping.yaml"))

    def analogue(df, name):
        """TALQ analog of inventory `name`: per inventory item, the mean of its mapped battery ratings (the
        maximum when the mapping says aggregate: max), then the mean over items (0-100 scale)."""
        parts = []
        for spec in mp[name].values():
            v = df[[f"{c}_rating" for c in spec["symptoms"]]].to_numpy(float)
            parts.append(np.nanmax(v, axis=1) if spec.get("aggregate") == "max" else np.nanmean(v, axis=1))
        return np.nanmean(np.column_stack(parts), axis=1)

    for k, (name, lab_, xmax) in enumerate([("bdi2", "BDI-II", 63), ("phq9", "PHQ-9", 27)]):   # max totals
        ax = fig.add_subplot(gs[k, 0])
        x = j[f"{name}_total"].to_numpy(float); y = analogue(j, name)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=2.2, color=PL.SLOTS[0], alpha=0.35, lw=0, rasterized=True)
        r = tot[(tot["x"] == f"{name}_total") & (tot["y"] == f"{name}_analogue")].iloc[0]
        ax.text(0.04, 0.97, f"ρ = {r['rho']:.2f}\nwithin person {r['within_rho']:.2f}\n"
                f"{int(r['n']):,} windows\n{int(r['n_users'])} users", transform=ax.transAxes, va="top",
                fontsize=5.0, color=INK2)
        ax.set_xlim(0, xmax); ax.set_ylim(0, 100)
        ax.set_xlabel(f"{lab_} total\n(same model, answering in the first person)")
        ax.set_ylabel(f"{PL.METHOD} {lab_} analog\n(mapped items, 0–100)")
        PL.hgrid(ax); lab(ax, "ab"[k], -0.36)
    # c: item level, convergent vs discriminant
    ax = fig.add_subplot(gs[:, 1])
    # PHQ-9 items first, each inventory sorted by the convergent correlation (from the bottom up)
    it = items.sort_values(["inventory", "convergent_rho"], ascending=[False, True]).reset_index(drop=True)
    yy = np.arange(len(it))
    ax.hlines(yy, it["discriminant_median_rho"], it["convergent_rho"], color=GRID, lw=0.8)
    ax.scatter(it["discriminant_median_rho"], yy, s=6, facecolor="white", edgecolor=MUTED, lw=0.6, zorder=3,
               label="median, other battery items")
    ax.scatter(it["convergent_rho"], yy, s=7, color=PL.SLOTS[0], lw=0, zorder=4,
               label="mapped battery item(s)")
    lbl = [("PHQ-9 " if r["inventory"] == "phq9" else "BDI-II ") + str(r["title"]) for _, r in it.iterrows()]
    ax.set_yticks(yy); ax.set_yticklabels(lbl, fontsize=5)
    ax.set_ylim(-0.8, len(it) - 0.2)
    ax.set_xlim(-0.05, 1); ax.set_xlabel("Spearman ρ across windows")
    ax.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.45, 1.0), handletextpad=0.3, ncol=1)
    lab(ax, "c", -0.95, 1.03)
    # d: factor correlations with the inventory totals (within person)
    ax = fig.add_subplot(gs[:, 2])
    fr = tot[tot["comparison"].str.contains("factor:")].copy()       # rows "... factor: <title>"
    fr["factor"] = fr["comparison"].str.split("factor: ").str[1]
    fr["inv"] = np.where(fr["x"] == "bdi2_total", "BDI-II", "PHQ-9")
    order = (fr[fr["inv"] == "PHQ-9"].sort_values("within_rho")["factor"]).tolist()
    for k, (nm, col) in enumerate([("BDI-II", PL.SLOTS[0]), ("PHQ-9", PL.SLOTS[1])]):
        g = fr[fr["inv"] == nm].set_index("factor").loc[order]
        yv = np.arange(len(order)) + (k - 0.5) * 0.38
        ax.barh(yv, g["within_rho"], height=0.36, color=col, lw=0, label=nm)
    ax.axvline(0, color=AXIS, lw=0.6)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=5.5)
    ax.set_ylim(-0.6, len(order) - 0.4)
    PL.factor_tags(ax, order)                          # squares in the color of each factor's family
    ax.set_xlabel("Spearman ρ within users\nwith the inventory total")
    ax.legend(fontsize=5.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1, ncol=2)
    lab(ax, "d", -0.95, 1.03)
    return PL.save(fig, "ED_Fig4", args.subdir)


if __name__ == "__main__":
    # Extended Data figure number -> drawing function; a failure (for example a missing input) is printed
    # with its traceback and the remaining figures are still drawn
    jobs = {"1": ed1_cohort, "2": ed2_parsing, "3": ed3_validation, "4": ed4_inventories, "5": ed5_factors,
            "6": ed6_psychometrics, "7": ed7_var, "8": ed8_states, "9": ed9_regimes, "10": ed10_sequences}
    for k, f in jobs.items():
        if want(k):
            try:
                print("wrote", f())
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"[fail] ED{k}: {e}")
