#!/usr/bin/env python3
"""
Step 21 - main-text figures (Figs. 1-6) and Table 1 of the manuscript.

Figures follow the Nature Mental Health artwork guidelines (180 mm wide, Arial-type text of
5-7 pt, thin black axes, no grid lines, color only where it carries meaning; see
src/slm/plotting.py).  Each figure is written as PDF and SVG (vector), PNG and TIFF (600 dpi) and an
editable PPTX.

  fig1    Fig. 1  TALQ ratings of a year of online writing: a, study design; b, ratings of all 52
                  symptoms across the year of one user; c, agreement between the five samples of the
                  rater; d, prediction of human annotations of single Reddit posts from their ratings
                  (lasso and a small neural network, cross-validated over posts) and comparators.
  fig2    Fig. 2  Twelve replicable factors: a, loadings of the 52 symptoms on the 12 factors; b, staNMF
                  instability across numbers of factors; c, replication in disjoint halves of the cohort;
                  d, internal consistency.
  fig3    Fig. 3  Lagged coupling between factors within individuals (panel VAR): a, autoregressive
                  coefficients; b, network of lagged effects with the pairs declared Granger predictive;
                  c, impulse responses of those pairs, with their lagged coefficients.
  fig4    Fig. 4  Discrete states of the factor profiles: a, each state on the UMAP embedding; b, mean
                  factor scores of each state; c, population transition matrix; d, persistence of each
                  state against its occupancy.
  fig5    Fig. 5  Individuals differ mainly in the states they occupy: a, cross-validated log likelihood
                  of Markov and occupancy mixtures; b, time-shuffle nulls; c, stability of the
                  classifications; d, occupancy and e, persistence of the states within each regime.
  fig6    Fig. 6  Course of symptoms over months: a, semivariogram relative to shuffled windows; b,
                  reliability of person-level features; c, number of course types; d, overall severity by
                  quarter in each type; e, change of each factor; f, agreement between interleaved halves;
                  g, markers not used to define the types.
  table1  Table 1 factors, the items that carry 50 % of their weight, DSM-5-TR correspondence, weighted
                  alpha and split-half congruence.

Every factor carries the color of its clinical family (plotting.FAMILY_COLORS) in Figs. 2-4 and 6 and in
Table 1.

Inputs (<run> is --run; the step that writes each file in parentheses)
  data/processed/windows.parquet                    windows, their index in the year (step 3)
  data/processed/<run>/summary.parquet              mean rating per window and symptom (step 5)
  data/processed/<run>/factors.parquet              confident windows with factor scores (step 6)
  data/processed/<run>/umap.parquet                 2-D UMAP coordinates (step 9)
  data/processed/<run>/states.parquet               state of every confident window (step 10)
  results/<run>/06_nmf.npz, 06_factor_table.csv     loadings, factor titles, items per factor (step 6)
  results/<run>/06_nmf_sweep<tag>.csv               staNMF sweeps, tags "", _fine, _broad, _k18, _k18_fine
                                                    (step 6)
  results/<run>/07_psychometrics.csv                alpha, kappa, ICC with bootstrap intervals (step 7)
  results/<run>/08_var_population.npy               bias-corrected panel VAR, P and FDR q values, impulse
                                                    responses (step 8)
  results/<run>/10_states.json, 10_state_centroids.csv   state names and centroids (steps 10, 10c)
  results/<run>/11_regimes.npy                      regimes, occupancy and transition matrices (step 11)
  results/<run>/12A_replicate_reliability.csv, 12C_splithalf_congruence.csv, 12D_heldout.csv,
  results/<run>/12_robustness.json                  reliability, replication, Markov versus occupancy
                                                    mixtures, shuffle nulls, stability (step 12)
  results/validation/14b_<model>_metrics.csv        validation on human-annotated posts (step 14b)
  results/validation/14d_<model>_predictive.csv     prediction of the human labels (step 14d)
  results/<run>/20_course.json, 20_course.npz, 20_course_quarters.csv, 20_course_sweep.csv
                                                    course types over the year (step 20)
Outputs  figures/main/Fig1..Fig6.{pdf,svg,png,tif,pptx}; results/<run>/table1.csv

usage: python3 21_figures_main.py [--run llama31_8b] [--only 1,2,3,4,5,6,T1]
"""
import argparse

import _setup  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle

from slm import params as P
from slm import plotting as PL
from slm.io import load_json
from slm.nmf import combined_sweep
from slm.prompts import load_symptoms

ap = argparse.ArgumentParser(description="Main-text figures (Figs. 1-6) and Table 1.")
ap.add_argument("--run", default=P.MAIN_RUN,
                help="annotation run whose results are drawn (default: %(default)s)")
ap.add_argument("--only", default=None,
                help="comma-separated subset to draw: 1-6 (figures) and T1 (Table 1); default all")
ap.add_argument("--subdir", default="main", help="folder under figures/ for the outputs (default: main)")
args = ap.parse_args()
only = set(args.only.split(",")) if args.only else None
PL.setup()
MM = PL.MM
INK, INK2, MUTED, LIGHT = PL.INK, PL.INK2, PL.MUTED, PL.LIGHT

R = P.results_dir(args.run)         # analysis results of the run
PD = P.processed_dir(args.run)      # processed tables (factors, states, UMAP)
SYM = load_symptoms()
codes = [s["shortcode"] for s in SYM]
sym_title = {s["shortcode"]: s["title"] for s in SYM}
SHORT = PL.SYMPTOM_SHORT                         # compact symptom labels for dense axes


def slab(c):
    """Short label of a symptom (full titles are in Supplementary Table 1)."""
    return SHORT.get(c, sym_title[c])


def want(k):
    """True when item k ("1".."6", "T1") was requested with --only, or when --only is not given."""
    return only is None or k in only


# ------------------------------------------------------------------ shared inputs
nmf = np.load(R / "06_nmf.npz", allow_pickle=True)
H, W, FT = nmf["H"], nmf["W"], list(nmf["titles"])     # loadings (factors x symptoms), scores, factor names
Hn = H / H.sum(axis=1, keepdims=True)                   # each factor's weights as shares summing to 1
fac = pd.read_parquet(PD / "factors.parquet")
FAM_ORDER = ["Anxiety", "Social", "Positive affect", "Somatic", "Affective lability", "Cognitive load",
             "Depressive affect", "Self-referential"]
# factors ordered by clinical family (the order of Table 1)
f_order = sorted(range(len(FT)), key=lambda j: (FAM_ORDER.index(PL.FAMILY.get(FT[j], ("Anxiety", 0))[0]), j))
# symptoms ordered by the factor that carries their largest share, then by that share
best = Hn.argmax(axis=0)
pos = {j: i for i, j in enumerate(f_order)}
s_order = sorted(range(len(codes)), key=lambda i: (pos[best[i]], -Hn[best[i], i]))


def arrow(ax, x0, y0, x1, y1):
    """Arrow from (x0, y0) to (x1, y1), in the millimeter coordinates of the Fig. 1a schematic."""
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=5, lw=0.6, color=INK,
                                 shrinkA=0, shrinkB=0))


def box(ax, x, y, w, h, fc="white", lw=0.5):
    """Rectangle with a black outline: lower-left corner (x, y), width w and height h in millimeters."""
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=INK, lw=lw))


# ======================================================================= Fig 1
def fig1():
    """Fig. 1: the scoring system and the reliability and validity of its ratings.

    a  study design, drawn as a schematic in millimeters: one user's year of 27 windows of 14 days, the
       posts of one window and one item of the 52-symptom questionnaire going into the LLM rater, its
       structured response (rating, confidence, justification) and the resulting record of 52 symptoms x
       27 windows per person.  Post times and the cells of the record are random illustrations (fixed
       seeds), not data; the window count is that of the 491 users.
    b  ratings (mean of the five samples, 0-100) of all 52 symptoms across the year of one user, in the
       order of the battery (Supplementary Table 1); gray cells are windows without posts.
    c  agreement between the five samples: ICC(1) of single ratings of each symptom against its mean
       rating; the dashed line marks the median; the three least reliable symptoms are labeled.
    d  prediction of human annotations of single posts (reduced protocol run locally with
       Llama-3.1-8B-Instruct, step 14d): cross-validated AUROC with 95 % bootstrap interval of a lasso
       (blue) and a neural network (purple) on all symptoms scored on each dataset, of the same lasso on
       nine lexicon and length features (orange) and of a supervised TF-IDF classifier trained on all posts
       of each dataset (green, step 14b); for severity, severe versus minimum depression; thin rules
       separate the four labels; the dashed line marks chance (0.5).
    Inputs: windows.parquet, summary.parquet, 12A_replicate_reliability.csv,
    results/validation/14d_<model>_predictive.csv, 14b_<model>_metrics.csv.  Returns the path of the PNG.
    """
    fig = plt.figure(figsize=(180 * MM, 150 * MM))
    # a: schematic drawn in millimeters on an axes without frame
    axa = fig.add_axes([0.0, 0.745, 1.0, 0.255])
    axa.set_xlim(0, 180); axa.set_ylim(0, 38); axa.set_axis_off()
    # number of scored windows of the 491 users: the moderation bot's windows were annotated but are not
    # part of the cohort
    _w = pd.read_parquet(P.PROCESSED / "windows.parquet", columns=["speaker_id"])
    n_win = int((~_w["speaker_id"].isin(P.BOT_SPEAKERS)).sum())

    # one person's year: 27 windows of two weeks, with posts as ticks
    rng = np.random.RandomState(3)                   # illustrative post times (fixed seed), not data
    x0, x1, ytl = 6, 60, 31.5
    axa.plot([x0, x1], [ytl, ytl], color=INK, lw=0.5)
    for k in range(28):                              # 28 boundaries delimit the 27 windows
        xx = x0 + (x1 - x0) * k / 27
        axa.plot([xx, xx], [ytl - 0.8, ytl + 0.8], color=INK, lw=0.4)
    for k in range(27):
        n = rng.poisson(2.2)
        xs = x0 + (x1 - x0) * (k + rng.uniform(0.1, 0.9, n)) / 27
        axa.plot(xs, np.full(n, ytl + 2.0), "|", ms=3, mew=0.5, color=MUTED)
    kh = 9                                           # the highlighted window
    axa.add_patch(Rectangle((x0 + (x1 - x0) * kh / 27, ytl - 1.6), (x1 - x0) / 27, 3.2, fc=PL.BLUE, ec="none",
                            alpha=0.35))
    axa.text(x0, ytl + 4.6, "One person, first year of sustained activity (27 windows of 14 days)",
             fontsize=5.5, ha="left", va="bottom")
    axa.text(x1 + 1.5, ytl, f"491 people\n{n_win:,} windows", fontsize=5.5, ha="left", va="center",
             color=INK2, linespacing=1.2)

    # inputs: the posts of one window (from the left) and one questionnaire item (from above)
    def lines_block(x, y, w, h, n):
        """A page with n gray text lines (the last one shorter), standing for posts."""
        box(axa, x, y, w, h)
        for i in range(n):
            yy = y + h - 1.6 - i * 1.5
            axa.plot([x + 1.2, x + w - 1.2 - (3 if i == n - 1 else 0)], [yy, yy], color=LIGHT, lw=1.0,
                     solid_capstyle="butt")
    for k in range(3):                               # three stacked pages
        lines_block(8 + k * 1.2, 12.4 - k * 1.2, 24, 8.5, 5)
    # dashed link from the highlighted window to its posts
    axa.plot([x0 + (x1 - x0) * (kh + 0.5) / 27, 21], [ytl - 1.8, 21.4], color=PL.BLUE, lw=0.5,
             ls=(0, (2, 1.5)))
    axa.text(20.5, 8.6, "Posts and comments\nin the window", fontsize=5.5, ha="center", va="top",
             linespacing=1.2)
    # questionnaire item, above the rater
    box(axa, 80, 27.2, 30, 9.6)
    axa.text(95, 35.6, "Item i of 52", fontsize=5.5, ha="center", va="top", fontweight="bold")
    axa.text(95, 32.4, "for example, loneliness\n0 absent to 100 every day", fontsize=5.2, ha="center",
             va="top", linespacing=1.2, color=INK2)
    axa.text(111.5, 32.0, "Symptom questionnaire\n(7 depression scales)", fontsize=5.5, ha="left",
             va="center", linespacing=1.2)
    # the rater
    box(axa, 80, 9.5, 30, 13.5, fc="#eef3f8", lw=0.6)
    axa.text(95, 20.4, "LLM rater", fontsize=6, ha="center", va="top", fontweight="bold")
    axa.text(95, 17.2, "Llama-3.1-8B-Instruct\nopen weights, fixed version\n5 samples per item", fontsize=5.2,
             ha="center", va="top", linespacing=1.25, color=INK2)
    arrow(axa, 34.4, 16.2, 79.6, 16.2)
    arrow(axa, 95, 26.8, 95, 23.4)
    # outputs
    box(axa, 120, 9.5, 24, 13.5)
    for i, t in enumerate(["Rating, 0 to 100", "Confidence, 0 to 100", "Justification"]):
        axa.text(122, 20.1 - i * 3.6, t, fontsize=5.4, ha="left", va="top")
    arrow(axa, 110.4, 16.2, 119.6, 16.2)
    axa.text(132, 8.6, "Structured response\n(JSON)", fontsize=5.5, ha="center", va="top", linespacing=1.2)
    # the resulting record: symptoms x windows
    gx, gy, gw, gh = 152, 7.6, 24, 17
    rng2 = np.random.RandomState(7)                  # illustrative cell values (fixed seed), not data
    M = rng2.beta(1.4, 2.6, size=(13, 18))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            c = PL.seq_cmap()(0.08 + 0.85 * M[i, j])
            axa.add_patch(Rectangle((gx + j * gw / 18, gy + gh - (i + 1) * gh / 13), gw / 18, gh / 13, fc=c,
                                    ec="white", lw=0.15))
    axa.add_patch(Rectangle((gx, gy), gw, gh, fc="none", ec=INK, lw=0.5))
    arrow(axa, 144.4, 16.2, 151.6, 16.2)
    axa.text(gx + gw / 2, gy - 1.2, "52 symptoms × 27 windows\nper person", fontsize=5.5, ha="center",
             va="top", linespacing=1.2)
    axa.text(gx + gw / 2, gy + gh + 1.0, "2.54 million ratings", fontsize=5.5, ha="center", va="bottom",
             color=INK2)
    # the framework: questionnaire item, LLM rater and structured response inside a dashed frame
    from matplotlib.patches import FancyBboxPatch
    axa.add_patch(FancyBboxPatch((77, 2.8), 70, 35.0, boxstyle="round,pad=0,rounding_size=1.5", fc="none",
                                 ec=MUTED, lw=0.6, ls=(0, (3, 2)), clip_on=False, zorder=0))
    axa.text(145.8, 37.2, PL.METHOD, fontsize=7, fontweight="bold", ha="right", va="top", color=INK)
    PL.fig_label(fig, "a", 0.004, 0.995)

    # b: one person's year of ratings
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.25, 0.8, 0.85], wspace=0.7, left=0.13, right=0.985,
                  top=0.70, bottom=0.075)
    ax = fig.add_subplot(gs[0, 0])
    summ = pd.read_parquet(PD / "summary.parquet")
    win = pd.read_parquet(P.PROCESSED / "windows.parquet",
                          columns=["speaker_id", "window_index", "window_start"])
    win["window_start"] = pd.to_datetime(win["window_start"])
    # the user ranked sixth by the number of confident windows: a densely observed year, chosen by rule
    nwin = fac.groupby("speaker_id").size()
    user = nwin.sort_values(ascending=False).index[min(5, len(nwin) - 1)]   # a densely observed user
    u = summ[summ["speaker_id"] == user].merge(win, on=["speaker_id", "window_start"])
    # symptoms x the 27 window positions; windows without posts were not scored and stay NaN (gray)
    M = np.full((len(codes), 27), np.nan)
    for r in u.itertuples():
        M[codes.index(r.symptom_shortcode), int(r.window_index)] = r.rating_mean
    # rows in the order of the battery (Supplementary Table 1): the factors are introduced only in Fig. 2
    im = PL.heat(ax, M, vmin=0, vmax=100, edge=0.15)
    ax.set_yticks(range(len(codes))); ax.set_yticklabels([slab(c) for c in codes], fontsize=5.0)
    ax.set_xticks([0, 6, 13, 20, 26]); ax.set_xticklabels(["0", "12", "26", "40", "52"])  # 2 weeks per window
    ax.set_xlabel("Weeks since onset")
    PL.no_frame(ax)
    ax.tick_params(axis="y", length=0, pad=1.2)
    cax = ax.inset_axes([1.06, 0.0, 0.045, 0.28])   # colorbar to the right of the heat map
    cb = fig.colorbar(im, cax=cax); cb.outline.set_visible(False)
    cb.ax.tick_params(length=1.5, width=0.4, labelsize=5, pad=1); cb.set_ticks([0, 50, 100])
    cb.set_label("Rating", fontsize=5.5, labelpad=2)
    PL.panel_label(ax, "b", x=-0.62, y=1.0)

    # c: agreement between the five samples
    ax = fig.add_subplot(gs[0, 1])
    rel = pd.read_csv(R / "12A_replicate_reliability.csv").set_index("symptom")
    mr = summ.groupby("symptom_shortcode")["rating_mean"].mean()   # mean rating over all scored windows
    x = mr.reindex(rel.index); y = rel["icc1"]
    ax.scatter(x, y, s=7, color=INK, lw=0, zorder=3)
    ax.axhline(y.median(), color=MUTED, lw=0.5, ls=(0, (3, 2)))   # dashed: median ICC(1)
    ax.set_xlabel("Mean rating"); ax.set_ylabel("Agreement between samples, ICC(1)")
    lo_y = max(0.0, np.floor((y.min() - 0.05) * 10) / 10)          # lower limit: next 0.1 below the minimum
    ax.set_ylim(lo_y, 1.0); ax.set_xlim(0, max(60, float(np.ceil(x.max() / 10) * 10)))
    ax.text(ax.get_xlim()[1], y.median() + 0.008, f"median {y.median():.2f}", fontsize=5, color=INK2,
            va="bottom", ha="right")
    # labels of the three least reliable symptoms, stacked below their points; the uppermost label sits
    # further right so that the leader lines never cross
    lab = list(y.nsmallest(3).index)
    for k, c in enumerate(sorted(lab, key=lambda c: y[c])):
        yt = lo_y + (1 - lo_y) * [0.055, 0.165, 0.26][k]
        xs = ax.get_xlim()[1] * [0.42, 0.42, 0.5][k]
        ax.annotate(slab(c), (x[c], y[c]), xytext=(xs, yt), textcoords="data", fontsize=5, color=INK2,
                    va="center", arrowprops=dict(arrowstyle="-", lw=0.3, color=MUTED, shrinkA=0, shrinkB=1.5))
    PL.panel_label(ax, "c", x=-0.42, y=1.0)

    # d: prediction of human labels from the ratings of each post (cross-validated over posts, step 14d)
    ax = fig.add_subplot(gs[0, 2])
    V = P.RESULTS / "validation"
    # results of Llama-3.1-8B-Instruct in the local deployment (llama.cpp; the Ollama run if only that exists)
    ref_tag = next((t for t in ["llama31_8b_q4km_llamacpp", "llama31_8b_q4km_ollama"]
                    if (V / f"14d_{t}_predictive.csv").exists()), None)
    pr = pd.read_csv(V / f"14d_{ref_tag}_predictive.csv")     # lasso, MLP and lexicon lasso (step 14d)
    v = pd.read_csv(V / f"14b_{ref_tag}_metrics.csv")         # supervised reference trained on all posts (14b)
    rows = [("irf", "pbu", "Perceived\nburdensomeness"), ("irf", "tbe", "Thwarted\nbelongingness"),
            ("lost", "low_self_esteem", "Low\nself-esteem"),
            ("severity", "severity", "Severe vs\nminimum\ndepression")]     # (dataset, label, row title)
    series = [(f"{PL.METHOD} ratings, lasso", PL.BLUE, "o"), (f"{PL.METHOD} ratings, neural network", PL.PURPLE, "^"),
              ("Lexicon and length features, lasso", PL.ORANGE, "s"), ("Supervised text classifier", PL.GREEN, "D")]
    for yi, (ds, lab_, name) in enumerate(rows):
        p_ = pr[(pr["dataset"] == ds) & (pr["label"] == lab_)].set_index("model")
        sup = v[(v["dataset"] == ds) & (v["label"] == lab_) & (v["source"] == "supervised")]
        picks = [p_.loc[m] if m in p_.index else None for m in ("lasso", "MLP", "lexicon lasso")]
        picks.append(sup.iloc[0] if len(sup) else None)
        for k, (r, (nm, col, mk)) in enumerate(zip(picks, series)):
            if r is None:
                continue
            yy = yi + (k - 1.5) * 0.18               # the four series side by side within the label's row
            ax.plot([r["auroc_ci_lo"], r["auroc_ci_hi"]], [yy, yy], color=col, lw=0.7)
            ax.plot(r["auroc"], yy, mk, ms=2.8 if mk not in ("D", "^") else 2.5, color=col, mew=0)
    ax.axvline(0.5, color=MUTED, lw=0.5, ls=(0, (3, 2)))           # dashed: chance
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[2] for r in rows], fontsize=5.3)
    ax.set_ylim(len(rows) - 0.45, -0.55); ax.set_xlim(0.4, 1.0); ax.set_xlabel("AUROC (95% CI)")
    ax.tick_params(axis="y", length=0)
    # thin rules between the four labels, extended under the row titles
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for yb in np.arange(len(rows) - 1) + 0.5:
        ax.plot([-0.52, 1.0], [yb, yb], transform=tr, color=LIGHT, lw=0.6, clip_on=False, zorder=0)
    hs = [Line2D([], [], marker=mk, ls="-", lw=0.7, ms=2.6, color=c, mew=0, label=n) for n, c, mk in series]
    ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(-0.55, -0.09), fontsize=5.1, ncol=1,
              handlelength=1.6)
    PL.panel_label(ax, "d", x=-0.55, y=1.0)
    return PL.save(fig, "Fig1", args.subdir)


# ======================================================================= Fig 2
def fig2():
    """Fig. 2: twelve replicable mental health factors.

    a  loadings of the 52 symptoms (columns) on the 12 factors (rows, in the order of Table 1), each row
       scaled to its maximum; symptoms are grouped by the factor that carries their largest share.  The
       colored square before each factor marks its clinical family (plotting.FAMILY_COLORS), the same colors
       as in Figs. 3, 4 and 6 and Table 1.
    b  staNMF instability across numbers of factors K, computed on window scores (W, black) and on
       loadings (H, vermillion); filled markers for 100 initializations, open markers for 20; the dashed
       line marks K = 12.
    c  replication in disjoint halves of the cohort: Tucker congruence of each factor between the loadings
       fitted in each half and in the full sample; dots give the median over both halves of 20 random
       splits and ticks the 5th percentile; dashed line 0.85 (fair similarity), dotted line 0.95
       (equality).
    d  internal consistency: weighted Cronbach's alpha of each factor with its 95% interval from 200
       bootstrap resamples of users.
    Inputs: 06_nmf.npz, 06_nmf_sweep<tag>.csv, 12C_splithalf_congruence.csv, 07_psychometrics.csv.
    Returns the path of the PNG.
    """
    fig = plt.figure(figsize=(180 * MM, 152 * MM))
    # a: loadings of the 52 symptoms on the 12 factors, each row scaled to its maximum
    ax = fig.add_axes([0.165, 0.635, 0.765, 0.35])
    Hr = H[f_order] / H[f_order].max(axis=1, keepdims=True)
    im = PL.heat(ax, Hr[:, s_order], vmin=0, vmax=1, edge=0.2)
    ax.set_yticks(range(len(f_order))); ax.set_yticklabels([FT[j] for j in f_order], fontsize=5.6)
    ax.set_xticks(range(len(codes))); ax.set_xticklabels([slab(codes[i]) for i in s_order], rotation=90,
                                                         fontsize=5.0)
    ax.tick_params(axis="x", length=0, pad=1.5)
    PL.no_frame(ax)
    # family squares between the axis and the factor names (the tick labels are moved to make room)
    PL.factor_tags(ax, [FT[j] for j in f_order], size_pt=4.6, gap_pt=1.2, label_gap_pt=2.2)
    cax = ax.inset_axes([1.015, 0.25, 0.012, 0.75])
    cb = fig.colorbar(im, cax=cax); cb.outline.set_visible(False)
    cb.ax.tick_params(length=1.5, width=0.4, labelsize=5, pad=1); cb.set_ticks([0, 0.5, 1])
    cb.set_label("Loading relative to\nthe factor's maximum", fontsize=5.3, labelpad=2)
    PL.panel_label(ax, "a", x=-0.205, y=1.0)

    gb = GridSpec(1, 1, figure=fig, left=0.075, right=0.32, top=0.395, bottom=0.085)
    gcd = GridSpec(1, 2, figure=fig, left=0.53, right=0.985, top=0.395, bottom=0.085, wspace=0.12,
                   width_ratios=[1.0, 0.85])
    # b: staNMF stability across K
    ax = fig.add_subplot(gb[0, 0])
    # all sweeps of the run; for each K, the one with the most initializations
    sw = combined_sweep(R, tags=("", "_fine", "_broad", "_k18", "_k18_fine"))
    many = sw["n_reps"].to_numpy() >= 100            # filled markers: 100 initializations
    for col_, key_, lab_ in [(INK, "instability_W", "Window scores (W)"), (PL.VERMILLION, "instability_H",
                                                                            "Loadings (H)")]:
        ax.plot(sw["K"], sw[key_], "-", lw=0.7, color=col_, label=lab_)
        ax.plot(sw["K"][many], sw[key_][many], "o", ms=2.3, color=col_, lw=0)
        ax.plot(sw["K"][~many], sw[key_][~many], "o", ms=2.3, mfc="white", mec=col_, mew=0.6, lw=0)
    ax.axvline(P.NMF_K, color=MUTED, lw=0.5, ls=(0, (3, 2)))          # dashed: K = 12
    ax.set_ylim(0, 0.07); ax.set_xlim(2, 26)
    ax.text(P.NMF_K, 0.0705, "K = 12", fontsize=5.2, color=INK2, va="bottom", ha="center")
    ax.set_xlabel("Number of factors, K"); ax.set_ylabel("Instability (staNMF)")
    ax.legend(fontsize=5.0, loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=1, handlelength=1.4)
    PL.panel_label(ax, "b", x=-0.42, y=1.02)

    # c: replication in split halves
    ax = fig.add_subplot(gcd[0, 0])
    names = [FT[j] for j in f_order]
    c = pd.read_csv(R / "12C_splithalf_congruence.csv").set_index("title").loc[names]
    y = np.arange(len(c))
    ax.hlines(y, c["phi_p05"], c["phi_median"], color=MUTED, lw=0.6)
    # median over the 40 half-sample fits, in the color of the factor's family
    ax.scatter(c["phi_median"], y, s=8, c=[PL.factor_color(t) for t in names], lw=0, zorder=3)
    ax.plot(c["phi_p05"], y, "|", ms=3.5, color=INK, mew=0.6)      # 5th percentile
    ax.axvline(0.85, color=MUTED, lw=0.5, ls=(0, (3, 2)))
    ax.axvline(0.95, color=MUTED, lw=0.5, ls=(0, (1, 1.5)))
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=5.2); ax.set_ylim(len(c) - 0.4, -0.6)
    ax.set_xlim(0.8, 1.0); ax.set_xticks([0.8, 0.85, 0.9, 0.95, 1.0])
    ax.set_xticklabels(["0.8", "", "0.9", "", "1"])
    ax.set_xlabel("Tucker φ between split halves")
    PL.factor_tags(ax, names)
    PL.panel_label(ax, "c", x=-0.82, y=1.02)
    ax_c = ax

    # d: internal consistency
    ax = fig.add_subplot(gcd[0, 1], sharey=ax_c)     # same rows as c
    ps = pd.read_csv(R / "07_psychometrics.csv").set_index("title").loc[names]
    y = np.arange(len(ps))
    ax.hlines(y, ps["alpha_ci_lo95"], ps["alpha_ci_hi95"], color=MUTED, lw=0.6)
    ax.scatter(ps["alpha"], y, s=8, c=[PL.factor_color(t) for t in names], lw=0, zorder=3)
    ax.tick_params(axis="y", labelleft=False, length=0)
    ax.set_xlim(0.4, 1); ax.set_xticks([0.4, 0.6, 0.8, 1.0]); ax.set_xticklabels(["0.4", "0.6", "0.8", "1"])
    ax.set_xlabel("Weighted Cronbach's α (95% CI)")
    PL.panel_label(ax, "d", x=-0.08, y=1.02)
    return PL.save(fig, "Fig2", args.subdir)


# ======================================================================= Fig 3
def fig3():
    """Fig. 3: lagged coupling between factors within individuals (bias-corrected panel VAR(1), step 8).

    a  autoregressive coefficient of each factor (the diagonal of A; one lag is one window of two weeks)
       with its bootstrap 95% interval, factors sorted by the coefficient.
    b  network of the off-diagonal coefficients A[i, j]: an arrow runs from factor j at window t to the
       factor i it predicts at window t + 1.  Black arrows, labeled with their
       coefficient: pairs that survive Benjamini-Hochberg control (q < 0.05) over the 132 directed pairs;
       gray arrows: P < 0.05 before correction; dashed arrows: negative coefficients.  Factors in a
       surviving pair are named in bold; nodes carry the color of the factor's family.
    c  response of the later factor to an impulse of IRF_IMPULSE_SCALE points in the earlier one, for the
       (at most two) surviving pairs, with bootstrap 95% bands.
    Inputs: 08_var_population.npy (key "corrected").  Returns the path of the PNG.
    """
    from matplotlib.patches import Circle
    var = np.load(R / "08_var_population.npy", allow_pickle=True).item()
    cor = var["corrected"]
    A = np.asarray(cor["A"])[0]                                    # [to, from]
    pv, qv = np.asarray(cor["p"], float), np.asarray(cor["q"], float)
    # the impulse response at horizon 1 equals A, so its bootstrap band is the interval of A
    lo1, hi1 = np.asarray(cor["irf_lo"])[1], np.asarray(cor["irf_hi"])[1]
    nF = len(FT)
    off = ~np.eye(nF, dtype=bool)
    fig = plt.figure(figsize=(180 * MM, 92 * MM))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.72, 1.45, 0.75], wspace=0.3, left=0.135, right=0.985,
                  top=0.94, bottom=0.12)
    # a: lag-1 autoregression of each factor
    ax = fig.add_subplot(gs[0, 0])
    ar, l0, h0 = np.diag(A), np.diag(lo1), np.diag(hi1)
    order = list(np.argsort(-ar))
    yy = np.arange(nF)
    ax.hlines(yy, l0[order], h0[order], color=MUTED, lw=0.6)
    ax.scatter(ar[order], yy, s=8, c=[PL.factor_color(FT[i]) for i in order], lw=0, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.5)
    ax.set_yticks(yy); ax.set_yticklabels([FT[i] for i in order], fontsize=5.2)
    ax.set_ylim(nF - 0.4, -0.6)
    ax.set_xlim(-0.05, 0.25); ax.set_xticks([0, 0.1, 0.2])
    ax.set_xlabel("Autoregressive coefficient,\nlag of two weeks (95% CI)")
    PL.factor_tags(ax, [FT[i] for i in order])
    PL.panel_label(ax, "a", x=-1.02, y=1.02)
    # b: network of lagged effects; arrows run from the earlier factor to the later one
    ax = fig.add_subplot(gs[0, 1])
    ax.set_aspect("equal"); ax.set_xlim(-1.62, 1.62); ax.set_ylim(-1.62, 1.3); ax.set_axis_off()
    ang = {j: np.deg2rad(90 - 360 * k / nF) for k, j in enumerate(f_order)}   # clockwise from the top
    pos = {j: np.array([np.cos(a), np.sin(a)]) for j, a in ang.items()}
    rn = 0.1                                                       # radius of a node
    sig_nom = [(i, j) for i in range(nF) for j in range(nF) if off[i, j] and pv[i, j] < 0.05]
    fdr = [(i, j) for (i, j) in sig_nom if qv[i, j] < P.FDR_Q]
    for (i, j) in sorted(sig_nom, key=lambda ij: qv[ij] < P.FDR_Q):  # surviving pairs drawn last (on top)
        strong = qv[i, j] < P.FDR_Q
        a0, a1 = pos[j], pos[i]
        u = (a1 - a0) / np.linalg.norm(a1 - a0)
        ax.add_patch(FancyArrowPatch(a0 + u * (rn + 0.02), a1 - u * (rn + 0.03), connectionstyle="arc3,rad=0.14",
                                     arrowstyle="-|>", mutation_scale=6 if strong else 4.5,
                                     lw=1.3 if strong else 0.5, color=INK if strong else "#a6a6a6",
                                     ls="-" if A[i, j] > 0 else (0, (2, 1.2)), zorder=3 if strong else 2))
    for j in f_order:
        x0, y0 = pos[j]
        ax.add_patch(Circle((x0, y0), rn, fc=PL.factor_color(FT[j]), ec="white", lw=0.6, zorder=5))
        a = ang[j]; lx, ly = 1.2 * np.cos(a), 1.16 * np.sin(a)
        ha = "center" if abs(np.cos(a)) < 0.2 else ("left" if np.cos(a) > 0 else "right")
        bold = any(j in pr for pr in fdr)
        ax.text(lx, ly, FT[j], fontsize=5.3, ha=ha, va="center", fontweight="bold" if bold else "normal",
                zorder=5)
    n05 = len(sig_nom); chance = 0.05 * off.sum()
    ax.text(0, -1.6, f"{n05} of {int(off.sum())} directed pairs at P < 0.05 ({chance:.1f} expected by chance), "
                     f"{len(fdr)} at FDR q < 0.05", fontsize=5.2, ha="center", va="bottom", color=INK2)
    hs = [Line2D([], [], color=INK, lw=1.3, label="FDR q < 0.05"),
          Line2D([], [], color="#a6a6a6", lw=0.5, label="P < 0.05, uncorrected"),
          Line2D([], [], color="#a6a6a6", lw=0.5, ls=(0, (2, 1.2)), label="Negative coefficient")]
    ax.legend(handles=hs, fontsize=5.0, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=3, handlelength=1.6,
              columnspacing=1.2)
    PL.panel_label(ax, "b", x=0.0, y=0.98)
    # c: responses to an impulse of IRF_IMPULSE_SCALE points for the pairs that survived FDR control
    sig = sorted(fdr, key=lambda ij: -abs(cor["irf"][1][ij]))[:2]
    sub = gs[0, 2].subgridspec(max(len(sig), 1), 1, hspace=0.85)
    hh = np.arange(cor["irf"].shape[0]) * 2                        # horizon in weeks
    s = P.IRF_IMPULSE_SCALE
    for k, (i, j) in enumerate(sig):
        a = fig.add_subplot(sub[k, 0])
        col = PL.factor_color(FT[j])
        a.fill_between(hh, cor["irf_lo"][:, i, j] * s, cor["irf_hi"][:, i, j] * s, color=col, alpha=0.22, lw=0)
        a.plot(hh, cor["irf"][:, i, j] * s, color=col, lw=0.9)
        a.axhline(0, color=MUTED, lw=0.4)
        a.text(0.0, 1.05, f"{FT[j]} → {FT[i]}\nlagged coefficient {A[i, j]:.2f}", transform=a.transAxes,
               fontsize=5.3, ha="left", va="bottom", linespacing=1.3)
        a.set_xticks([0, 6, 12, 18, 24]); a.set_xlim(0, 24)
        a.set_ylim(-0.6, 7.2); a.set_yticks([0, 3, 6])
        a.set_ylabel("Points", fontsize=5.3)
        if k == len(sig) - 1:
            a.set_xlabel(f"Weeks after an impulse of {s:.0f} points", fontsize=5.5)
        else:
            a.set_xticklabels([])
        if k == 0:
            PL.panel_label(a, "c", x=-0.36, y=1.12)
    return PL.save(fig, "Fig3", args.subdir)


# ======================================================================= Fig 4
def fig4():
    """Fig. 4: discrete states of the factor profiles (Dirichlet process Gaussian mixture, 11 states).

    a  each state on the two-dimensional UMAP embedding of all confident windows: its windows in blue over
       all windows in light gray; titles give the state label and its percentage of windows.  Labels name
       the factors at least one standard deviation above their mean across states (the highest factor in
       parentheses when none qualifies; 'higher' and 'lower' separate states with the same label).
    b  mean factor scores (0-100) of each state, factors in the order of Table 1 with the colored square of
       their clinical family.
    c  population transition matrix between exactly adjacent windows, P(to | from): columns are the state
       at window t, rows the state at window t + 1.
    d  probability of remaining in each state (bars, the diagonal of c) against its occupancy (dots,
       share of all confident windows); the text compares the mean persistence with its mean over 200
       shuffles of each user's windows in time.
    Inputs: states.parquet, umap.parquet, 10_states.json, 10_state_centroids.csv, 11_regimes.npy,
    12_robustness.json.  Returns the path of the PNG.
    """
    st = pd.read_parquet(PD / "states.parquet")
    um = pd.read_parquet(PD / "umap.parquet")
    js = load_json(R / "10_states.json")
    names = js.get("state_names") or [f"S{i+1}" for i in range(int(st["state"].max()) + 1)]
    S = len(names)                                   # state codes 0..S-1, in decreasing order of occupancy
    d = um.merge(st[["speaker_id", "window_start", "state"]], on=["speaker_id", "window_start"])
    reg = np.load(R / "11_regimes.npy", allow_pickle=True).item()
    cent = pd.read_csv(R / "10_state_centroids.csv").set_index("state")
    fig = plt.figure(figsize=(180 * MM, 122 * MM))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.35, 1.0, 0.75], height_ratios=[1, 1], wspace=0.55,
                  hspace=0.62, left=0.015, right=0.985, top=0.96, bottom=0.13)
    # a: each state on the shared embedding
    sub = gs[:, 0].subgridspec(int(np.ceil(S / 3)), 3, hspace=0.3, wspace=0.06)   # three states per row
    x, y = d["umap2_0"].to_numpy(), d["umap2_1"].to_numpy()
    for s in range(S):
        a = fig.add_subplot(sub[s // 3, s % 3])
        m = d["state"].to_numpy() == s
        # point clouds are rasterized (thousands of points) inside the vector figure
        a.scatter(x[~m], y[~m], s=0.25, color=LIGHT, lw=0, rasterized=True)
        a.scatter(x[m], y[m], s=0.45, color=PL.BLUE, lw=0, rasterized=True)
        a.set_xticks([]); a.set_yticks([])
        PL.no_frame(a)
        a.text(0.5, 1.0, f"{names[s]} ({100*m.mean():.0f}%)", transform=a.transAxes, fontsize=5.3,
               ha="center", va="bottom")
        if s == 0:
            PL.panel_label(a, "a", x=-0.05, y=1.12)
    # b: mean factor scores of each state
    ax = fig.add_subplot(gs[0, 1:])
    # rows in state-code order (via the names), columns in the order of Table 1
    C = (cent[[FT[j] for j in f_order]].loc[names] if set(names) <= set(cent.index)
         else cent[[FT[j] for j in f_order]])
    im = PL.heat(ax, C.to_numpy(float), vmin=0, vmax=70)
    ax.set_yticks(range(len(C))); ax.set_yticklabels(C.index, fontsize=5.3)
    ax.set_xticks(range(len(FT))); ax.set_xticklabels([FT[j] for j in f_order], rotation=40, ha="right",
                                                       rotation_mode="anchor", fontsize=5.1)
    ax.tick_params(length=0, pad=1.5)
    PL.no_frame(ax)
    PL.factor_tags(ax, [FT[j] for j in f_order], axis="x", size_pt=3.6, gap_pt=1.0, label_gap_pt=1.4)
    PL.thin_colorbar(fig, im, ax, "Mean factor score", fraction=0.025, pad=0.015, aspect=18)
    PL.panel_label(ax, "b", x=-0.14, y=1.03)
    # c: population transitions between exactly adjacent windows
    ax = fig.add_subplot(gs[1, 1])
    A = reg["A_pop_gap"] if "A_pop_gap" in reg else reg["A_pop"]    # [to, from], columns sum to 1
    im = PL.heat(ax, A, vmin=0, vmax=max(0.5, float(np.max(A))))
    ax.set_xticks(range(S)); ax.set_xticklabels(names, rotation=90, fontsize=4.9)
    ax.set_yticks(range(S)); ax.set_yticklabels(names, fontsize=4.9)
    ax.set_xlabel("From (window t)", fontsize=5.5); ax.set_ylabel("To (window t + 1)", fontsize=5.5)
    ax.tick_params(length=0, pad=1)
    PL.no_frame(ax)
    ax.set_aspect("equal")
    PL.thin_colorbar(fig, im, ax, "P(to | from)", fraction=0.045, pad=0.03, aspect=16)
    PL.panel_label(ax, "c", x=-0.42, y=1.03)
    # d: persistence against occupancy
    ax = fig.add_subplot(gs[1, 2])
    rob = load_json(R / "12_robustness.json") if (R / "12_robustness.json").exists() else {}
    occ = np.bincount(d["state"], minlength=S) / len(d)       # share of all confident windows
    stay = np.diag(A)                                          # P(stay) between adjacent windows
    yy = np.arange(S)
    ax.barh(yy, stay, height=0.62, color="#bdbdbd", lw=0, label="P(stay)")
    ax.plot(occ, yy, "o", ms=2.3, color=INK, lw=0, label="Occupancy")
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=5.0); ax.set_ylim(S - 0.4, -0.6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 0.6); ax.set_xlabel("Probability")
    ax.legend(fontsize=5.0, loc="lower right", handlelength=0.9)
    if "D_shuffle" in rob:
        # mean P(stay) observed and in the time-shuffle null of step 12, part D(ii)
        dsh = rob["D_shuffle"]
        ax.text(0.98, 0.32, f"Mean P(stay)\n{dsh['persistence_real']:.2f} observed\n"
                            f"{dsh['persistence_null_mean']:.2f} shuffled", transform=ax.transAxes,
                fontsize=5.0, ha="right", va="bottom", color=INK2, linespacing=1.2)
    PL.panel_label(ax, "d", x=-0.62, y=1.03)
    return PL.save(fig, "Fig4", args.subdir)


# ======================================================================= Fig 5
def fig5():
    """Fig. 5: individuals differ mainly in the states they occupy.

    a  cross-validated log likelihood per window of the state sequences of users left out of fitting
       (five folds over users) for mixtures of first-order Markov chains (blue) and mixtures that model
       state occupancy only (vermillion), by number of components; solid lines with markers for the
       observed sequences, dashed lines for sequences shuffled in time within users; the dotted line marks
       the number of regimes selected by AICc for the hard-assignment fits (four).
    b  time-shuffle nulls: the Markov minus occupancy difference at that number of components (left) and
       the mean probability of staying in a state (right), observed (black dot) and in 200 shuffles
       within users (gray dots, spread horizontally for visibility), with the permutation P value.
    c  stability of the classifications: adjusted Rand index (black) of the regime labels with bootstrap
       refits (median and interquartile range), with soft-assignment EM and with an occupancy mixture,
       and between the states of the Dirichlet process mixture and of a Gaussian HMM; Cohen's kappa
       (vermillion) between the classifications of the two halves of each user's year by Markov regimes
       and by occupancy mixtures fitted on the other users.
    d  occupancy of the states within each regime (share of the regime's windows; also the regime's
       distribution of initial states); row labels give the share of users.
    e  probability of remaining in each state within each regime (diagonal of the regime's transition
       matrix, consecutive confident windows); cells of states with less than 2 % of the regime's windows
       are left empty (gray).
    Inputs: 11_regimes.npy, 12D_heldout.csv, 12_robustness.json.  Returns the path of the PNG.
    """
    reg = np.load(R / "11_regimes.npy", allow_pickle=True).item()
    rob = load_json(R / "12_robustness.json") if (R / "12_robustness.json").exists() else {}
    names = list(reg["state_names"]); S = len(names); K = int(reg["K"])   # K: regimes selected by AICc
    fig = plt.figure(figsize=(180 * MM, 112 * MM))
    gt = GridSpec(1, 3, figure=fig, width_ratios=[1.05, 0.7, 0.9], wspace=1.0, left=0.07, right=0.985,
                  top=0.94, bottom=0.6)
    gb = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.1, left=0.075, right=0.94, top=0.43,
                  bottom=0.12)
    # a: likelihood of the sequences of people withheld from fitting
    ax = fig.add_subplot(gt[0, 0])
    cv = pd.read_csv(R / "12D_heldout.csv")
    ax.plot(cv["K"], cv["markov"], "-o", ms=2.2, lw=0.8, color=PL.BLUE, label="Markov mixture")
    ax.plot(cv["K"], cv["occupancy"], "-o", ms=2.2, lw=0.8, color=PL.VERMILLION, label="Occupancy mixture")
    # dashed: the same models on sequences shuffled in time within users (mean of two shuffles)
    ax.plot(cv["K"], cv["markov_shuffled"], ls=(0, (3, 2)), lw=0.7, color=PL.BLUE, label="Markov, shuffled")
    ax.plot(cv["K"], cv["occupancy_shuffled"], ls=(0, (3, 2)), lw=0.7, color=PL.VERMILLION,
            label="Occupancy, shuffled")
    ax.axvline(K, color=MUTED, lw=0.5, ls=(0, (1, 1.5)))      # dotted: number of regimes (AICc)
    ax.set_xlabel("Number of mixture components"); ax.set_ylabel("Held-out log likelihood per window")
    ax.set_xticks(cv["K"])
    ax.legend(fontsize=5.0, loc="lower right", handlelength=1.6)
    PL.panel_label(ax, "a", x=-0.4, y=1.03)
    # b: shuffle nulls
    sub = gt[0, 1].subgridspec(1, 2, wspace=1.3)
    dsh = rob["D_shuffle"]
    rng = np.random.RandomState(0)                   # horizontal jitter of the null values (display only)
    for k, (key, nm) in enumerate([("dyn_advantage", f"Markov minus occupancy,\n{K} components (per window)"),
                                   ("persistence", "Mean probability of\nstaying in a state")]):
        a = fig.add_subplot(sub[0, k])
        null = np.asarray(dsh.get(f"{key}_null", [dsh[f"{key}_null_mean"]]), float)
        a.scatter(rng.uniform(-0.18, 0.18, len(null)), null, s=2.5, color="#bdbdbd", lw=0, label="Shuffled")
        a.plot([0.55], [dsh[f"{key}_real"]], "o", ms=3.0, color=INK, label="Observed")
        a.set_xlim(-0.45, 0.9); a.set_xticks([]); a.spines["bottom"].set_visible(False)
        a.set_ylabel(nm, fontsize=5.2)
        pv = dsh["p_dyn" if key == "dyn_advantage" else "p_persistence"]   # one-sided permutation P value
        a.text(0.5, 1.0, f"P = {pv:.2g}", transform=a.transAxes, fontsize=5.1, ha="center", va="bottom")
        if k == 0:
            PL.panel_label(a, "b", x=-2.0, y=1.03)
            a.legend(fontsize=5.0, loc="upper center", bbox_to_anchor=(1.3, -0.03), ncol=2, handlelength=0.6,
                     columnspacing=0.8, handletextpad=0.3)
    # c: stability of the classifications
    ax = fig.add_subplot(gt[0, 2])
    rows = []                                        # (label, value, interquartile range or None, kind)
    g = rob["D_agreement"]
    rows += [("Bootstrap refits", g["ari_bootstrap_median"], g["ari_bootstrap_iqr"], "ARI"),
             ("Hard vs soft EM", g["ari_soft_vs_hard"], None, "ARI"),
             ("Markov vs occupancy mixture", g["ari_occupancy_vs_hard"], None, "ARI")]
    if "E" in rob:
        rows += [("States, DPMM vs HMM", rob["E"]["ari_hmm_vs_dpmm"], None, "ARI")]
    f_ = rob["F"]
    rows += [("Year halves, Markov regimes", f_["markov"]["kappa"], None, "κ"),
             ("Year halves, occupancy mixture", f_["occ"]["kappa"], None, "κ")]
    yy = np.arange(len(rows))
    for yv, (lab, v, iqr, kind) in zip(yy, rows):
        col = INK if kind == "ARI" else PL.VERMILLION
        if iqr is not None:
            ax.hlines(yv, iqr[0], iqr[1], color=col, lw=0.8)
        ax.plot([v], [yv], "o", ms=2.6, color=col)
        ax.text(v + 0.03, yv, f"{v:.2f}", va="center", ha="left", fontsize=5.1)
    ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], fontsize=5.2)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1); ax.set_xlabel("Agreement")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=INK, ms=2.6, label="Adjusted Rand index"),
                       Line2D([], [], marker="o", ls="", color=PL.VERMILLION, ms=2.6, label="Cohen's κ")],
              fontsize=5.0, loc="upper right", bbox_to_anchor=(1.02, 1.04), handletextpad=0.2)
    PL.panel_label(ax, "c", x=-0.85, y=1.03)
    # d, e: occupancy and persistence within each regime
    occ_r = np.asarray(reg["p0"])                    # (K, S) occupancy = initial distribution of each regime
    stay_r = np.array([np.diag(reg["A"][r]) for r in range(K)])
    stay_r = np.where(occ_r >= 0.02, stay_r, np.nan)      # too few windows in the state to estimate P(stay)
    rl = [f"R{r+1} ({100*reg['frac'][r]:.0f}%)" for r in range(K)]   # regime and its share of users
    for k, (Mx, lab, pl, vmax) in enumerate([(occ_r, "Share of windows", "d", 0.5),
                                             (stay_r, "P(stay)", "e", 1.0)]):
        a = fig.add_subplot(gb[0, k])
        im = PL.heat(a, Mx, vmin=0, vmax=vmax)
        # the value of every cell, without the leading zero; white text on dark cells
        for r in range(K):
            for s_ in range(S):
                v = Mx[r, s_]
                if np.isfinite(v):
                    a.text(s_, r, f"{v:.2f}", ha="center", va="center",
                           fontsize=4.6, color="white" if v > 0.55 * vmax else INK)
        a.set_xticks(range(S)); a.set_xticklabels(names, rotation=90, fontsize=5.0)
        a.set_yticks(range(K)); a.set_yticklabels(rl if k == 0 else [""] * K, fontsize=5.3)
        a.tick_params(length=0, pad=1.5)
        PL.no_frame(a)
        PL.thin_colorbar(fig, im, a, lab, fraction=0.035, pad=0.015, aspect=12)
        PL.panel_label(a, pl, x=-0.17 if k == 0 else -0.04, y=1.04)
    return PL.save(fig, "Fig5", args.subdir)


# ======================================================================= Fig 6
TYPE_COLORS = [INK, PL.BLUE]          # stable majority, remitting type (K = 2)
Q_LABELS = ["0–3", "3–6", "6–9", "9–12"]


def fig6():
    """Fig. 6: people differ in the course of their symptoms over months (step 20).

    a  semivariogram of overall severity (black; the mean of the 11 factors other than Positive wellbeing)
       and of each factor (colored by family): half the mean squared difference between two windows of a
       user by the weeks between them, relative to its mean over 20 shuffles in time; lags with at least
       700 pairs of windows.
    b  split-half reliability of level, variability and course (odd against even windows) and of inertia
       and instability (first against second half of the year) for each factor (dots) and for overall
       severity (bars); the gray band reaches the 95th percentile of the course correlation after shuffling.
    c  BIC of growth mixture models with 1-6 course types relative to one type, observed and shuffled.
    d  overall severity by quarter of the year in each type, with 95% bootstrap intervals.
    e  change of each factor from the first to the last quarter in the smallest (remitting, filled) and
       the largest (stable, open) type.
    f  course in held-out windows: overall severity by quarter in the half of the windows (odd or even) not
       used to assign the type, for users assigned to the remitting type and the others; the text gives the
       correlation between the log odds of the type and the slope in the other half, and its range after
       shuffling.
    g  measures not used to define the types, by quarter: PHQ-9 completed by the same model, VADER compound
       sentiment, share of windows in the euthymic state and median words per window.
    Inputs: 20_course.json, 20_course.npz, 20_course_quarters.csv, 20_course_sweep.csv.
    Returns the path of the PNG.
    """
    J = load_json(R / "20_course.json")
    Z = np.load(R / "20_course.npz", allow_pickle=True)
    qt = pd.read_csv(R / "20_course_quarters.csv").set_index(["type", "quarter"])
    sw = pd.read_csv(R / "20_course_sweep.csv")
    K = int(J["types"]["K"]); sizes = J["types"]["sizes"]; n = sum(sizes)
    tnames = ["Stable", "Remitting"] if K == 2 else [f"Type {k + 1}" for k in range(K)]
    tlab = [f"{tnames[k]} ({100 * sizes[k] / n:.0f}%)" for k in range(K)]
    fig = plt.figure(figsize=(180 * MM, 168 * MM))
    g1 = GridSpec(1, 3, figure=fig, left=0.075, right=0.985, top=0.965, bottom=0.715, wspace=0.42,
                  width_ratios=[1.0, 1.15, 0.8])
    g2 = GridSpec(1, 3, figure=fig, left=0.075, right=0.985, top=0.6, bottom=0.31, wspace=0.55,
                  width_ratios=[0.85, 1.1, 0.6])
    g3 = GridSpec(1, 4, figure=fig, left=0.075, right=0.985, top=0.2, bottom=0.055, wspace=0.55)
    # a: semivariogram relative to shuffled windows
    ax = fig.add_subplot(g1[0, 0])
    lags = Z["vario_lags"] * 2; ratio = Z["vario_ratio"]; cols = list(Z["vario_cols"]); pairs = Z["vario_pairs"]
    ok = pairs >= 700                     # lags with at least 700 pairs of windows (up to about 44 weeks)
    for j, c in enumerate(cols):
        if c == "Severity":
            continue
        ax.plot(lags[ok], ratio[ok, j], color=PL.factor_color(c), lw=0.5, alpha=0.6)
    b = cols.index("Severity")
    ax.plot(lags[ok], ratio[ok, b], color=INK, lw=1.3, label="Overall severity")
    ax.axhline(1, color=MUTED, lw=0.5, ls=(0, (3, 2)))
    ax.set_xlim(0, 46); ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlabel("Weeks between two windows"); ax.set_ylabel("Dissimilarity relative to\nshuffled windows")
    ax.legend(handles=[Line2D([], [], color=INK, lw=1.3, label="Overall severity"),
                       Line2D([], [], color=MUTED, lw=0.5, label="Each factor")], fontsize=5.0, loc="lower right")
    PL.panel_label(ax, "a", x=-0.36, y=1.02)
    # b: reliability of person-level features
    ax = fig.add_subplot(g1[0, 1])
    rel = J["feature_reliability"]
    feats = [("level", "Level"), ("variability", "Variability"), ("course", "Course"),
             ("halves_inertia", "Inertia"), ("halves_instability", "Instability")]
    rng = np.random.RandomState(1)
    for x, (k, nm) in enumerate(feats):
        v = [rel[t][k] for t in FT]
        ax.scatter(x + rng.uniform(-0.17, 0.17, len(v)), v, s=6, c=[PL.factor_color(t) for t in FT], lw=0, zorder=3)
        ax.plot([x - 0.26, x + 0.26], [rel["Severity"][k]] * 2, color=INK, lw=1.3, zorder=4)
    cn = [rel[t]["course_null_p95"] for t in FT + ["Severity"]]
    ax.add_patch(Rectangle((1.7, -0.2), 0.6, max(cn) + 0.2, fc="#ececec", ec="none", zorder=1))
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.axvline(2.5, color=LIGHT, lw=0.6)
    ax.set_xticks(range(len(feats))); ax.set_xticklabels([f[1] for f in feats], fontsize=5.4)
    ax.set_xlim(-0.5, len(feats) - 0.5); ax.set_ylim(-0.2, 1.0)
    ax.set_ylabel("Split-half reliability (r)")
    ax.text(1.0, 0.995, "Odd vs even windows", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=5.0, color=INK2)
    ax.text(3.5, 0.995, "Year halves", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=5.0, color=INK2)
    ax.legend(handles=[Line2D([], [], color=INK, lw=1.3, label="Overall severity"),
                       Line2D([], [], marker="o", ls="", ms=2.4, color=MUTED, label="Each factor"),
                       Patch(color="#ececec", label="Course, shuffled\n(95th percentile)")],
              fontsize=4.9, loc="upper right", bbox_to_anchor=(1.0, 0.92), handlelength=1.2)
    PL.panel_label(ax, "b", x=-0.27, y=1.02)
    # c: number of course types
    ax = fig.add_subplot(g1[0, 2])
    for which, col, ls in [("observed", INK, "-"), ("shuffled", MUTED, (0, (3, 2)))]:
        q = sw[sw["data"] == which].sort_values("K")
        ax.plot(q["K"], q["bic"] - q["bic"].iloc[0], marker="o", ms=2.4, color=col, lw=0.8, ls=ls,
                label="Observed" if which == "observed" else "Shuffled in time")
    ax.axhline(0, color=MUTED, lw=0.4)
    ax.set_xticks(range(1, int(sw["K"].max()) + 1)); ax.set_xlabel("Number of course types")
    ax.set_ylabel("BIC relative to one type")
    ax.legend(fontsize=5.0, loc="upper left")
    PL.panel_label(ax, "c", x=-0.42, y=1.02)
    # d: overall severity by quarter of the year in each type
    ax = fig.add_subplot(g2[0, 0])
    qi = list(Z["q_index"])
    for k in range(K):
        y = [qt.loc[(k, q), "Severity"] for q in range(1, 5)]
        lo = [Z["q_Severity_lo"][qi.index(f"{k}-{q}")] for q in range(1, 5)]
        hi = [Z["q_Severity_hi"][qi.index(f"{k}-{q}")] for q in range(1, 5)]
        xx = np.arange(4) + (k - 0.5) * 0.08
        ax.vlines(xx, lo, hi, color=TYPE_COLORS[k], lw=0.7)
        ax.plot(xx, y, "-o", ms=2.8, lw=1.0, color=TYPE_COLORS[k], label=tlab[k])
    ax.set_xticks(range(4)); ax.set_xticklabels(Q_LABELS); ax.set_xlabel("Months since onset")
    ax.set_ylabel("Overall severity (95% CI)")
    ax.legend(fontsize=5.0, loc="upper right")
    PL.panel_label(ax, "d", x=-0.36, y=1.02)
    ax_d = ax
    # e: change in each factor from the first to the last quarter
    ax = fig.add_subplot(g2[0, 1])
    names = [FT[j] for j in f_order]
    yy = np.arange(len(names))
    for k in range(K):
        ch = [qt.loc[(k, 4), t] - qt.loc[(k, 1), t] for t in names]
        if k == 0:
            ax.scatter(ch, yy, s=9, facecolors="white", edgecolors=MUTED, lw=0.6, zorder=3, label=tlab[k])
        else:
            ax.scatter(ch, yy, s=10, c=[PL.factor_color(t) for t in names], lw=0, zorder=4, label=tlab[k])
            for c_, y_ in zip(ch, yy):
                ax.plot([0, c_], [y_, y_], color=LIGHT, lw=0.6, zorder=1)
    ax.axvline(0, color=MUTED, lw=0.5)
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=5.2); ax.set_ylim(len(names) - 0.4, -0.6)
    ax.set_xlabel("Change from the first to the last quarter (points)")
    PL.factor_tags(ax, names)
    # filled marker in neutral ink: the dots of the remitting type take the color of their factor
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.0, mfc="white", mec=MUTED, mew=0.6, label=tlab[0])] +
                      [Line2D([], [], marker="o", ls="", ms=3.0, color=INK2, mew=0, label=t_) for t_ in tlab[1:]],
              fontsize=5.0, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handletextpad=0.2)
    PL.panel_label(ax, "e", x=-0.75, y=1.02)
    # f: course in held-out windows: the type assigned from one half of a user's windows (odd or even) against
    # overall severity by quarter in the other half
    ax = fig.add_subplot(g2[0, 2])
    hv = J["halves"]
    if hv is not None and "ho_q_mean" in Z.files:
        for k, col in enumerate([INK, PL.BLUE]):                 # rows: other types, smallest (remitting) type
            xx = np.arange(4) + (k - 0.5) * 0.08
            ax.vlines(xx, Z["ho_q_lo"][k], Z["ho_q_hi"][k], color=col, lw=0.7)
            ax.plot(xx, Z["ho_q_mean"][k], "-o", ms=2.8, lw=1.0, color=col)
        nl = np.asarray(hv["course_r_null"], float)
        ax.text(0.97, 0.97, f"r = {hv['course_r']:.2f}\nshuffled {nl.min():.2f} to {nl.max():.2f}".replace("-", "−"),
                transform=ax.transAxes, ha="right", va="top", fontsize=5.0, color=INK2, linespacing=1.25)
    ax.set_ylim(ax_d.get_ylim())
    ax.set_xticks(range(4)); ax.set_xticklabels(Q_LABELS); ax.set_xlabel("Months since onset")
    ax.set_ylabel("Overall severity in the\nother half of the windows")
    PL.panel_label(ax, "f", x=-0.62, y=1.02)
    # f -> g: external markers by quarter in each type
    ext = [("phq9_total", "PHQ-9 total\n(completed by the model)"), ("vader_compound", "Sentiment\n(VADER compound)"),
           ("euthymic", "Windows in the\neuthymic state (share)"), ("median_words", "Words per window\n(median)")]
    for i, (c, nm) in enumerate(ext):
        a = fig.add_subplot(g3[0, i])
        if c not in qt:
            a.text(0.5, 0.5, "not available", transform=a.transAxes, ha="center", fontsize=5)
            continue
        for k in range(K):
            y = [qt.loc[(k, q), c] for q in range(1, 5)]
            key_lo = f"q_{c}_lo"
            xx = np.arange(4) + (k - 0.5) * 0.08
            if key_lo in Z.files:
                lo = [Z[key_lo][qi.index(f"{k}-{q}")] for q in range(1, 5)]
                hi = [Z[f"q_{c}_hi"][qi.index(f"{k}-{q}")] for q in range(1, 5)]
                a.vlines(xx, lo, hi, color=TYPE_COLORS[k], lw=0.6)
            a.plot(xx, y, "-o", ms=2.4, lw=0.9, color=TYPE_COLORS[k])
        a.set_xticks(range(4)); a.set_xticklabels(Q_LABELS, fontsize=5.0)
        a.set_xlabel("Months since onset", fontsize=5.4)
        a.set_ylabel(nm, fontsize=5.3)
        if c == "median_words":
            a.set_yscale("log"); a.yaxis.set_minor_locator(plt.NullLocator())
            a.set_yticks([100, 200, 400, 800]); a.set_yticklabels(["100", "200", "400", "800"])
        if i == 0:
            PL.panel_label(a, "g", x=-0.5, y=1.04)
    return PL.save(fig, "Fig6", args.subdir)


# ===================================================================== Table 1
DSM = {  # mapping of factors to DSM-5-TR criteria (MDD A1-A9; GAD A-C), chosen by the authors
    "Anxious tension": "GAD A, C1 (restlessness or feeling keyed up) and C4 (irritability)",
    "Social strain": "None (interpersonal, not a DSM criterion)",
    "Positive wellbeing": "MDD A1 and A2 (inverse)",
    "Anhedonia": "MDD A2 (anhedonia) and A1 (empty mood)",
    "Emotional pain": "MDD A1 (depressed mood)",
    "Sleep disturbance": "MDD A4 and GAD C6",
    "Physical distress": ("MDD A3 (appetite or weight), A5 (psychomotor retardation) and A6 (fatigue), "
                          "with somatic symptoms"),
    "Fearful preoccupation": "GAD A, with panic and illness anxiety",
    "Emotional instability": "GAD C4 (irritability)",
    "Negative self-reflection": "MDD A7 (worthlessness or guilt)",
    "Self-harm tendency": "MDD A9 (suicidality)",
    "Overwhelm": "MDD A5 (psychomotor agitation) and A8 (concentration), GAD C3",
}


def table1():
    """Table 1: one row per factor (order of Table 1) with the items that carry 50 % of its weight (step 6),
    the DSM-5-TR correspondence (DSM above), weighted alpha (step 7) and the median split-half Tucker
    congruence (step 12, part C).  Writes and returns results/<run>/table1.csv."""
    tab = pd.read_csv(R / "06_factor_table.csv")
    ps = pd.read_csv(R / "07_psychometrics.csv").set_index("title")
    cg = pd.read_csv(R / "12C_splithalf_congruence.csv").set_index("title")
    rows = []
    for j in f_order:
        t = FT[j]
        r = tab[tab["title"] == t].iloc[0]
        rows.append({"Factor": t, "Items that carry 50% of the weight": r["top50"].replace("; ", ", "),
                     "DSM-5-TR correspondence": DSM.get(t, ""),
                     "α": f"{ps.loc[t, 'alpha']:.2f}", "Congruence φ": f"{cg.loc[t, 'phi_median']:.2f}"})
    out = pd.DataFrame(rows)
    out.to_csv(R / "table1.csv", index=False)
    print(out.to_string())
    return R / "table1.csv"


if __name__ == "__main__":
    # draw the requested items; a failure (for example a missing input) is printed with its traceback and
    # the remaining items are still drawn
    jobs = {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5, "6": fig6, "T1": table1}
    for k, f in jobs.items():
        if want(k):
            try:
                print("wrote", f())
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"[fail] {k}: {e}")
