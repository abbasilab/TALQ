"""
Figure style for the manuscript (Nature Mental Health format): colors, font sizes, line widths, helpers
for panels, heat maps and colorbars, and saving of every figure in five formats.

Used by scripts/21_figures_main.py (Figs. 1-6, Table 1) and scripts/22_figures_extended.py (Extended
Data Figs. 1-10); both call setup() once and save(fig, name, subdir) for each figure.

Conventions, following the journal's artwork guidelines:
* Arial-type sans serif text (Liberation Sans, which has Arial's metrics, is used for layout;
  the PowerPoint export names the font Arial); 5-7 pt text at final size, 8 pt bold lowercase
  panel letters;
* figures 180 mm wide (double column), thin black axes without top and right spines, no grid
  lines, outward ticks;
* colors from the Okabe-Ito palette, which stays distinguishable for readers with color
  vision deficiency; a single blue ramp for magnitudes (heat maps) and a blue-white-red ramp
  for signed values;
* explanations go in the figure legend, not in panel titles.

Outputs  figures/<subdir>/<name>.pdf and .svg (vector, text kept as text), .png and .tif (600 dpi; the TIFF
         is RGB with lossless LZW compression) and .pptx (editable shapes, see slm.pptx_export); subdir is
         "main" or "extended_data".

usage:   from slm import plotting as PL
         PL.setup()
         fig = plt.figure(figsize=(180 * PL.MM, 120 * PL.MM))
         ...
         PL.save(fig, "Fig4", "main")
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from . import params as P

MM = 1 / 25.4                                    # inches per millimeter (matplotlib sizes are in inches)
SINGLE, ONE_HALF, DOUBLE = 89 * MM, 120 * MM, 183 * MM   # journal column widths: single, 1.5, double

# name of the scoring framework as printed on figure labels (Fig. 1d, Extended Data Fig. 4)
METHOD = "TALQ"

# ---- ink and neutral tones -------------------------------------------------------------
INK = "#000000"        # text, axes, primary marks
INK2 = "#404040"       # secondary text
MUTED = "#8c8c8c"      # reference lines, null distributions
LIGHT = "#d9d9d9"      # background marks (for example all windows behind a highlighted state)
GRID = "#ebebeb"       # very light guides where a reference grid is needed
AXIS = "#000000"       # axis lines (spines)
SURFACE = "#ffffff"    # figure background

# ---- Okabe-Ito categorical colors (fixed order, never cycled) ---------------------------
BLUE, VERMILLION, GREEN, ORANGE = "#0072B2", "#D55E00", "#009E73", "#E69F00"
SKY, PURPLE, YELLOW = "#56B4E9", "#CC79A7", "#F0E442"
SLOTS = [BLUE, VERMILLION, GREEN, ORANGE, PURPLE, SKY, INK2, YELLOW]   # order of categorical series

# ---- sequential and diverging ramps ------------------------------------------------------
# ColorBrewer "Blues" with a white low end, for magnitudes (seq_cmap)
BLUE_RAMP = ["#ffffff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c",
             "#08306b"]
DIV_NEG, DIV_MID, DIV_POS = "#2166ac", "#f7f7f7", "#b2182b"   # ColorBrewer RdBu ends and center (div_cmap)

# Factors are grouped by clinical family; the family sets their order in tables and figures and their color
# (FAMILY_COLORS), which marks every factor in the figures next to its name (factor_tags).
FAMILY = {  # factor -> (family, order)
    "Anxious tension": ("Anxiety", 0), "Fearful preoccupation": ("Anxiety", 0),
    "Social strain": ("Social", 1),
    "Positive wellbeing": ("Positive affect", 2),
    "Sleep disturbance": ("Somatic", 3), "Physical distress": ("Somatic", 3),
    "Emotional instability": ("Affective lability", 4),
    "Overwhelm": ("Cognitive load", 5),
    "Anhedonia": ("Depressive affect", 6), "Emotional pain": ("Depressive affect", 6),
    "Negative self-reflection": ("Self-referential", 7), "Self-harm tendency": ("Self-referential", 7),
}
# Abbreviations of the 12 factors shown on dense axes (the same as in slm.states.FACTOR_ABBR)
FACTOR_ABBR = {"Anxious tension": "AT", "Fearful preoccupation": "FP", "Social strain": "SS",
               "Positive wellbeing": "PW", "Sleep disturbance": "SD", "Physical distress": "PD",
               "Emotional instability": "EI", "Overwhelm": "OW", "Anhedonia": "AN", "Emotional pain": "EP",
               "Negative self-reflection": "NSR", "Self-harm tendency": "SH"}
REGIME_COLORS = SLOTS[:5]                        # one categorical color per regime (up to five)

# Compact symptom labels used on dense figure axes (shortcode -> label).  Keys are the shortcodes of
# config/symptoms.yaml; the shortcode of Indecisiveness is spelled "indiciciveness", as in the stored
# ratings.  Symptoms not listed here are shown with their title.
SYMPTOM_SHORT = {
    "others_unfriendly": "Others unfriendly", "others_dislike": "Others dislike me",
    "feeling_good": "Feeling as good as others", "feeling_needed": "Feeling needed",
    "ip_sensitivity": "Interpersonal sensitivity", "concentration_dysfunction": "Concentration problems",
    "indiciciveness": "Indecisiveness", "insomnia_middle": "Middle insomnia",
    "insomnia_early": "Initial insomnia",
    "insomnia_late": "Late insomnia", "anxiety": "Anxious mood", "panic": "Panic / phobic",
    "self_failure": "Past failure", "life_full": "Life overfull", "gi_dysfunction": "GI problems",
    "somatization": "Somatic complaints", "arousal": "Sympathetic arousal",
}


# One color per clinical family of factors, used for the squares next to factor names and for markers that
# stand for factors.  Factors of the same family share a color and are always named as well.
FAMILY_COLORS = {
    "Anxiety": "#2a78d6", "Social": "#eb6834", "Positive affect": "#1baf7a", "Somatic": "#eda100",
    "Affective lability": "#e87ba4", "Cognitive load": "#008300", "Depressive affect": "#4a3aa7",
    "Self-referential": "#e34948",
}


def factor_color(title: str) -> str:
    """Color of the clinical family of a factor (FAMILY_COLORS); gray (MUTED) for names that are not factors."""
    fam = FAMILY.get(title)
    return FAMILY_COLORS[fam[0]] if fam else MUTED


# order of the clinical families on figure axes (the order of Fig. 2 and Table 1)
FAMILY_ORDER = ["Anxiety", "Social", "Positive affect", "Somatic", "Affective lability", "Cognitive load",
                "Depressive affect", "Self-referential"]


def family_order(titles) -> list:
    """Indices of the factors in `titles` ordered by clinical family (FAMILY_ORDER), then by their index.

    Used to show factors in the same order in every figure, e.g. [titles[j] for j in family_order(titles)].
    """
    return sorted(range(len(titles)),
                  key=lambda j: (FAMILY_ORDER.index(FAMILY.get(titles[j], ("Anxiety", 0))[0]), j))


def factor_tags(ax, titles, axis="y", size_pt=4.2, gap_pt=1.5, label_gap_pt=2.0):
    """Small squares in the color of each factor between the tick labels and the axes.

    titles : factor names in tick order (tick k at data coordinate k), on the y axis (axis="y") or the x axis.
    The squares are size_pt points wide, gap_pt points from the axes, and the tick labels are moved outward
    by label_gap_pt more, so that labels and squares never overlap.  Call after the axis limits are set."""
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory
    fig = ax.figure
    pos = ax.get_position()
    w_ax_pt = pos.width * fig.get_figwidth() * 72
    h_ax_pt = pos.height * fig.get_figheight() * 72
    if axis == "y":
        lo, hi = ax.get_ylim(); span = abs(hi - lo)
        h_data = size_pt / h_ax_pt * span                       # square height in data units
        tr = blended_transform_factory(ax.transAxes, ax.transData) + \
            ScaledTranslation(-(gap_pt + size_pt) / 72, 0, fig.dpi_scale_trans)
        w_frac = size_pt / w_ax_pt
        for k, t in enumerate(titles):
            ax.add_patch(Rectangle((0, k - h_data / 2), w_frac, h_data, transform=tr, fc=factor_color(t), ec="none",
                                   clip_on=False))
        ax.tick_params(axis="y", length=0, pad=gap_pt + size_pt + label_gap_pt)
    else:
        lo, hi = ax.get_xlim(); span = abs(hi - lo)
        w_data = size_pt / w_ax_pt * span
        tr = blended_transform_factory(ax.transData, ax.transAxes) + \
            ScaledTranslation(0, -(gap_pt + size_pt) / 72, fig.dpi_scale_trans)
        h_frac = size_pt / h_ax_pt
        for k, t in enumerate(titles):
            ax.add_patch(Rectangle((k - w_data / 2, 0), w_data, h_frac, transform=tr, fc=factor_color(t), ec="none",
                                   clip_on=False))
        ax.tick_params(axis="x", length=0, pad=gap_pt + size_pt + label_gap_pt)


def seq_cmap():
    """Sequential colormap from white to dark blue (BLUE_RAMP) for magnitudes, e.g. mean factor scores of
    states or transition probabilities.  Returns a new LinearSegmentedColormap."""
    return LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)


def div_cmap():
    """Diverging colormap blue - near white - red for signed values such as correlations; meant for a range
    symmetric about zero (vmin = -vmax).  Returns a new LinearSegmentedColormap."""
    return LinearSegmentedColormap.from_list("blue_red", [DIV_NEG, DIV_MID, DIV_POS])


def setup():
    """Apply the figure style to matplotlib's global settings (rcParams); call once before drawing.

    Sizes are in points at final print size; line widths of 0.4-0.9 pt.  Figures are saved with a tight
    bounding box (0.02 inch padding), PNG at 600 dpi, and fonts embedded as TrueType (fonttype 42) so
    that text stays text in PDF and PostScript files; SVG text is kept as text.
    """
    mpl.rcParams.update({
        # Liberation Sans first: metric-compatible with Arial and freely available, so the layout matches
        # the Arial text of the PowerPoint export
        "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "font.size": 6, "axes.titlesize": 6, "axes.labelsize": 6, "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5, "legend.fontsize": 5.5, "figure.titlesize": 7,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.5, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.labelpad": 2.0, "axes.titlepad": 2.0,
        "xtick.color": INK, "ytick.color": INK, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "xtick.direction": "out", "ytick.direction": "out",
        "text.color": INK, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 0.9, "lines.markersize": 3, "lines.solid_capstyle": "butt",
        "patch.linewidth": 0.5, "hatch.linewidth": 0.4,
        "legend.frameon": False, "legend.handlelength": 1.2, "legend.handletextpad": 0.4,
        "legend.borderaxespad": 0.2, "legend.labelspacing": 0.3, "legend.columnspacing": 0.8,
        # 150 dpi on screen; 600 dpi in saved raster files, cropped to the drawn content
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        # TrueType fonts embedded in PDF/PS (editable text); SVG keeps text as text
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.prop_cycle": mpl.cycler(color=SLOTS), "image.interpolation": "nearest",
        "mathtext.default": "regular",
    })


def save(fig, name: str, subdir: str = "main", pptx: bool = True, formats=("pdf", "svg", "png", "tif")):
    """Save a figure as PDF, SVG, PNG, TIFF and an editable PPTX, then close it.

    fig     : matplotlib Figure.
    name    : file name without extension (e.g. "Fig4", "ED_Fig8").
    subdir  : folder under figures/ ("main" or "extended_data").
    pptx    : also write the PowerPoint version (slm.pptx_export.save_pptx); a failure of that export
              is printed and does not stop the figure script.
    formats : file formats written by matplotlib; "tif" is the 600-dpi rendering converted to RGB on a white
              background and compressed losslessly (LZW), and the SVG names Arial first among its fonts.
    Returns the path of the PNG file.
    """
    out = P.FIGURES / subdir
    out.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        if ext == "tif":
            import io

            from PIL import Image
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            im = Image.open(buf)
            rgb = Image.new("RGB", im.size, "white")
            rgb.paste(im, mask=im.split()[-1] if im.mode == "RGBA" else None)
            dpi = mpl.rcParams["savefig.dpi"]
            rgb.save(out / f"{name}.tif", compression="tiff_lzw", dpi=(dpi, dpi))
        else:
            fig.savefig(out / f"{name}.{ext}")
            if ext == "svg":                # Liberation Sans has the metrics of Arial; name Arial first
                f_svg = out / f"{name}.svg"
                f_svg.write_text(f_svg.read_text(encoding="utf-8").replace("'Liberation Sans', 'Arial'",
                                                                            "'Arial', 'Liberation Sans'"),
                                 encoding="utf-8")
    if pptx:
        try:
            from .pptx_export import save_pptx       # imported here: python-pptx is needed only for this
            save_pptx(fig, out / f"{name}.pptx")
        except Exception as e:  # noqa: BLE001
            print(f"[pptx] {name}: {e}")
    plt.close(fig)
    return out / f"{name}.png"


def panel_label(ax, s, x=-0.18, y=1.06):
    """Bold lowercase panel letter (8 pt) at (x, y) in axes coordinates (0-1 span the axes; negative x is
    to the left of the axes, y > 1 above it)."""
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left", color=INK)


def fig_label(fig, s, x, y):
    """Bold lowercase panel letter (8 pt) with its top-left corner at (x, y) in figure coordinates (0-1)."""
    fig.text(x, y, s, fontsize=8, fontweight="bold", va="top", ha="left", color=INK)


def hgrid(ax):
    """Placeholder for a horizontal reference grid: draws nothing and returns None, because this style
    has no grid lines.  The Extended Data figure script calls it on panels where a grid could be shown."""
    return None


def ci_err(ax, x, lo, hi, color=INK2, lw=0.6):
    """Vertical interval lines from lo to hi at positions x (arrays or scalars), lw in points."""
    ax.vlines(x, lo, hi, color=color, linewidth=lw)


def despine(ax, left=True, bottom=True):
    """Hide all spines except, optionally, the left and bottom ones."""
    for name, sp in ax.spines.items():
        sp.set_visible((name == "left" and left) or (name == "bottom" and bottom))


def no_frame(ax):
    """Hide all four spines (for heat maps and embeddings)."""
    for sp in ax.spines.values():
        sp.set_visible(False)


def thin_colorbar(fig, mappable, ax, label, **kw):
    """A slim colorbar without outline, with small ticks.

    fig, mappable, ax : as for Figure.colorbar (mappable is, e.g., the result of heat()).
    label             : colorbar label (5.5 pt).
    kw                : further Figure.colorbar arguments; defaults fraction 0.03, pad 0.02, aspect 22.
    Returns the Colorbar.
    """
    kw.setdefault("fraction", 0.03)
    kw.setdefault("pad", 0.02)
    kw.setdefault("aspect", 22)
    cb = fig.colorbar(mappable, ax=ax, **kw)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1.5, width=0.4, labelsize=5, pad=1)
    cb.set_label(label, fontsize=5.5, labelpad=2)
    return cb


def heat(ax, M, cmap=None, vmin=0, vmax=None, edge=0.25):
    """Heat map drawn with pcolormesh (one vector cell per value, editable in PowerPoint); rows run top to
    bottom as in a table.

    ax         : axes to draw on.
    M          : 2-D array (rows x columns); NaN cells are shown light gray.
    cmap       : colormap (default seq_cmap()); a copy is modified, so a shared colormap is not changed.
    vmin, vmax : color limits.
    edge       : width in points of the white lines between cells (0 for none).
    Cell (i, j) is centered on x = j, y = i, so tick positions are the row and column indices.
    Returns the QuadMesh (for a colorbar).
    """
    M = np.ma.masked_invalid(np.asarray(M, float))
    cm = cmap or seq_cmap()
    if hasattr(cm, "copy"):
        cm = cm.copy()
    cm.set_bad("#f0f0f0")
    # cell edges at -0.5, 0.5, ..., n - 0.5 so that cells are centered on integer positions
    im = ax.pcolormesh(np.arange(M.shape[1] + 1) - 0.5, np.arange(M.shape[0] + 1) - 0.5, M, cmap=cm,
                       vmin=vmin, vmax=vmax, edgecolors="white", linewidth=edge, shading="flat")
    ax.set_xlim(-0.5, M.shape[1] - 0.5)
    ax.set_ylim(M.shape[0] - 0.5, -0.5)              # inverted y axis: row 0 at the top
    return im
