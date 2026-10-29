"""
Export matplotlib figures as editable PowerPoint slides.

Used by slm.plotting.save, which writes a .pptx version next to the PDF and PNG of every figure of
scripts 21_figures_main.py and 22_figures_extended.py (figures/<subdir>/<name>.pptx).

Why this module exists
----------------------
Journals ask for figures as vector files, and co-authors often want to adjust
labels, colors or layout by hand.  A PDF or SVG can be edited in Illustrator,
but most labs work in PowerPoint.  This module draws a matplotlib figure
directly into a .pptx slide, so that every element becomes a native shape:

* lines (axes, ticks, error bars, curves) become PowerPoint lines or open
  freeform shapes, with their width, color and dash pattern;
* bars, patches and filled areas become closed freeform shapes;
* circular markers become ovals;
* every text element (tick labels, axis labels, panel letters, legends)
  becomes a text box in Arial, so it can be retyped or restyled;
* the shapes of each axes (each panel) are grouped, so a panel can be moved
  or resized as a unit, and ungrouped for detailed edits.

Artists that would produce thousands of shapes (for example the dense point
clouds of a UMAP embedding, or the smooth color ramp of a colorbar) are drawn
as embedded images when the artist is flagged ``rasterized=True``, exactly as
matplotlib does for PDF output.

How it works
------------
matplotlib separates *what* is drawn (artists) from *how* it is drawn
(renderers).  The PDF and SVG backends are renderers that turn drawing
primitives (paths, text, images) into file commands.  ``RendererPPTX`` below is
such a renderer: it receives the same primitives and turns them into shapes
with python-pptx.  ``MixedModeRenderer`` (from matplotlib) handles rasterized
artists by drawing them with the Agg raster renderer and passing the resulting
image to ``RendererPPTX.draw_image``.  Primitives that RendererPPTX does not
implement (markers, collections such as scatter plots) reach it through the
default implementations of matplotlib's RendererBase, which call draw_path once
per marker or collection member.

Coordinates: the figure is drawn at 72 dots per inch, so one renderer unit is
one typographic point.  matplotlib's y axis points up, PowerPoint's points
down, and PowerPoint measures positions in EMU (12,700 EMU per point, 914,400
per inch).  Paths are clipped to their axes here, because PowerPoint shapes
have no clipping.

Usage
-----
    from slm.pptx_export import save_pptx
    save_pptx(fig, "Fig2.pptx")          # one slide, the size of the figure
    save_pptx([fig1, fig2], "all.pptx")  # one slide per figure
"""
from __future__ import annotations

import re
from io import BytesIO

import numpy as np
from lxml import etree
from matplotlib.backend_bases import GraphicsContextBase, RendererBase
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.colors import to_rgba
from matplotlib.path import Path
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

# digits and signs -> Unicode superscripts (for exponents of log-axis tick labels)
_SUPERSCRIPT = str.maketrans("0123456789-\u2212+",
                             "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b\u207b\u207a")


def plain_math(s: str) -> str:
    """Matplotlib mathtext as plain Unicode text for PowerPoint.

    Tick labels of logarithmic axes arrive as '$\\mathdefault{10^{3}}$', which becomes '10\u00b3'.  Font
    commands (\\mathdefault, \\mathrm, \\mathit, \\mathbf) keep their content, exponents become Unicode
    superscripts, \\times and \\cdot become their characters and the dollar signs are dropped.  Other
    mathtext is left as written (the figures write symbols as Unicode text).
    """
    s = s.replace("$", "")
    for cmd in ("\\mathdefault", "\\mathrm", "\\mathit", "\\mathbf"):
        while cmd + "{" in s:
            i = s.index(cmd + "{")
            j = k = i + len(cmd) + 1               # j: first character inside the braces
            depth = 1
            while k < len(s) and depth:            # find the matching closing brace
                depth += {"{": 1, "}": -1}.get(s[k], 0)
                k += 1
            s = s[:i] + s[j:k - 1] + s[k:]
    s = re.sub(r"\^\{([^{}]*)\}", lambda m: m.group(1).translate(_SUPERSCRIPT), s)
    s = re.sub(r"\^([0-9\-\u2212+])", lambda m: m.group(1).translate(_SUPERSCRIPT), s)
    return s.replace("\\times", "\u00d7").replace("\\cdot", "\u00b7")


EMU_PER_PT = 12700            # PowerPoint distances are in English Metric Units
# Arial's ascent (0.905 em) minus descent (0.212 em), halved: the height above the baseline at which
# PowerPoint centers a line of text in a box that is anchored in the middle
ARIAL_LINE_CENTER = 0.3465    # (ascent - descent) / 2 of Arial, in em: where PowerPoint centers a line
FONT_MAP = {                  # fonts used for layout -> font named in the slide
    "Liberation Sans": "Arial", "Arial": "Arial", "DejaVu Sans": "Arial", "Helvetica": "Arial",
}
GROUPED = {"axes", "legend"}  # matplotlib artist groups that become PowerPoint groups
_UNIT_CIRCLE = Path.unit_circle()   # the path matplotlib uses for circular markers (cubic Bezier arcs)


# --------------------------------------------------------------------------- geometry helpers
def _clip_segment(p0, p1, box):
    """Liang-Barsky clipping of one segment to box = (x0, y0, x1, y1); returns the visible part or None.

    The segment is parametrized as P(t) = p0 + t (p1 - p0), 0 <= t <= 1.  For each of the four box
    edges, the inside condition reads p * t <= q.  An edge with p < 0 is crossed from outside to inside
    at t = q / p, which raises the entry parameter t0; an edge with p > 0 is crossed from inside to
    outside, which lowers the exit parameter t1.  The visible part is P(t0)..P(t1), empty when t0 > t1.
    p0, p1 : (x, y) end points; box : (xmin, ymin, xmax, ymax), all in points.
    Returns ((x, y), (x, y)) or None.
    """
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    # (p, q) for the edges x >= xmin, x <= xmax, y >= ymin, y <= ymax
    for p, q in ((-dx, x0 - box[0]), (dx, box[2] - x0), (-dy, y0 - box[1]), (dy, box[3] - y0)):
        if p == 0:
            if q < 0:
                return None                      # parallel to this edge and outside it
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)                      # entering the half-plane
        else:
            t1 = min(t1, t)                      # leaving the half-plane
        if t0 > t1:
            return None
    return (x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy)


def _clip_polyline(pts, box):
    """Clip an open polyline to a box; returns a list of visible polylines (each a list of points).

    Each segment is clipped with _clip_segment.  A visible piece continues the current polyline when it
    starts where the polyline ends (within 1e-6 pt); otherwise the line has left the box and re-entered
    it, and a new polyline starts.  Polylines with fewer than two points are dropped.
    """
    out, cur = [], []
    for a, b in zip(pts[:-1], pts[1:]):
        seg = _clip_segment(a, b, box)
        if seg is None:
            if len(cur) > 1:
                out.append(cur)
            cur = []
            continue
        s0, s1 = seg
        if cur and np.allclose(cur[-1], s0, atol=1e-6):
            cur.append(s1)
        else:
            if len(cur) > 1:
                out.append(cur)
            cur = [s0, s1]
    if len(cur) > 1:
        out.append(cur)
    return out


def _clip_polygon(pts, box):
    """Sutherland-Hodgman clipping of a closed polygon to a box (keeps fills inside the axes).

    The polygon is clipped successively against the four half-planes x >= xmin, x <= xmax, y >= ymin and
    y <= ymax.  For each edge prev -> cur: if cur is inside, the crossing point is added first when prev
    is outside, then cur; if only prev is inside, only the crossing point is added.  A concave polygon
    can come out with zero-width parts along the box edge, which does not change the filled area.
    pts : list of (x, y); box : (xmin, ymin, xmax, ymax).  Returns a list of (x, y), possibly empty.
    """
    def clip(poly, inside, cross):
        """One pass of the algorithm against one half-plane."""
        res = []
        for i in range(len(poly)):
            cur, prev = poly[i], poly[i - 1]     # i = 0 pairs the first vertex with the last (closed)
            if inside(cur):
                if not inside(prev):
                    res.append(cross(prev, cur))
                res.append(cur)
            elif inside(prev):
                res.append(cross(prev, cur))
        return res

    def x_cross(xc):
        """Crossing of segment a -> b with the vertical line x = xc (1e-12 avoids a division by zero)."""
        return lambda a, b: (xc, a[1] + (b[1] - a[1]) * (xc - a[0]) / ((b[0] - a[0]) or 1e-12))

    def y_cross(yc):
        """Crossing of segment a -> b with the horizontal line y = yc."""
        return lambda a, b: (a[0] + (b[0] - a[0]) * (yc - a[1]) / ((b[1] - a[1]) or 1e-12), yc)

    poly = [tuple(p) for p in pts]
    for inside, cross in ((lambda p: p[0] >= box[0], x_cross(box[0])),
                          (lambda p: p[0] <= box[2], x_cross(box[2])),
                          (lambda p: p[1] >= box[1], y_cross(box[1])),
                          (lambda p: p[1] <= box[3], y_cross(box[3]))):
        if not poly:
            break
        poly = clip(poly, inside, cross)
    return poly


def _subpaths(path, transform):
    """Split a matplotlib path into (points, closed) subpaths in display coordinates, curves flattened.

    path      : matplotlib Path (vertices and codes).
    transform : transform from path coordinates to display coordinates (points, y up).
    iter_segments(curves=False) converts Bezier curves into line segments and, with remove_nans=True,
    splits the path at NaN vertices (a gap in a plotted line).  Each MOVETO starts a subpath; CLOSEPOLY
    marks the current subpath as closed.  Returns a list of (list of (x, y), closed).
    """
    subs, cur, closed = [], [], False
    for verts, code in path.iter_segments(transform, curves=False, simplify=False, remove_nans=True):
        if code == Path.MOVETO:
            if len(cur) > 1 or (len(cur) == 1 and closed):
                subs.append((cur, closed))
            cur, closed = [tuple(verts[-2:])], False      # the last two numbers are the end point
        elif code == Path.CLOSEPOLY:
            closed = True
        else:  # LINETO (curves were converted to line segments)
            cur.append(tuple(verts[-2:]))
    if len(cur) > 1:
        subs.append((cur, closed))
    return subs


# --------------------------------------------------------------------------- XML helpers
def _strip_style(shape):
    """Remove the theme style reference python-pptx adds, so no theme fill, outline or shadow leaks in.

    python-pptx gives new autoshapes, connectors and freeforms a <p:style> element that points to the
    theme's line, fill, effect and font styles; without it, a shape shows only the properties set here.
    """
    st = shape._element.find(qn("p:style"))
    if st is not None:
        shape._element.remove(st)


def _set_alpha(fill_or_line_color_parent, alpha):
    """Add <a:alpha> to the solid color under an spPr or ln element (python-pptx has no API for this).

    fill_or_line_color_parent : lxml element that contains <a:srgbClr> elements (a fill, a line or a
                                text run).
    alpha                     : opacity 0-1; nothing is written when it is (nearly) 1.
    DrawingML expresses opacity in thousandths of a percent: val = 100000 is fully opaque.
    """
    if alpha >= 0.999:
        return
    for clr in fill_or_line_color_parent.iter(qn("a:srgbClr")):
        a = etree.SubElement(clr, qn("a:alpha"))
        a.set("val", str(int(round(alpha * 100000))))


def _rgb(c):
    """RGB(A) color with components in 0-1 -> python-pptx RGBColor (0-255 per channel; alpha ignored)."""
    r, g, b = (int(round(v * 255)) for v in c[:3])
    return RGBColor(r, g, b)


# --------------------------------------------------------------------------- the renderer
class RendererPPTX(RendererBase):
    """A matplotlib renderer whose drawing primitives become python-pptx shapes on one slide.

    slide     : python-pptx slide to draw on.
    width_pt, height_pt : size of the drawn area in points (reported to matplotlib as the canvas size).
    image_dpi : resolution of rasterized artists and resampled images.
    origin    : figure position (x, y_top), in points with y up, placed at the slide's top-left corner
                (the top-left corner of the tight bounding box, see _draw_on_slide).
    """

    def __init__(self, slide, width_pt, height_pt, image_dpi=300, origin=(0.0, 0.0)):
        super().__init__()
        self.slide = slide
        self.width, self.height = width_pt, height_pt
        self.origin = origin                 # figure point placed at the slide's top-left corner (x, y_top)
        self.image_dpi = image_dpi           # resolution of rasterized artists (MixedModeRenderer)
        self._stack = []                     # open matplotlib groups: [name, pptx group or None]

    # ---- capabilities matplotlib asks every renderer about -------------------------------
    def flipy(self):
        """Whether y increases downwards: no, as in the PDF backend."""
        return False                         # y increases upwards, as in PDF

    def get_canvas_width_height(self):
        """Canvas size in points."""
        return self.width, self.height

    def option_image_nocomposite(self):
        """Ask matplotlib not to merge the images of an axes into one."""
        return True                          # keep every image as its own picture

    def option_scale_image(self):
        """Accept images with an affine transform (draw_image's transform argument)."""
        return True                          # we can place unsampled images with an affine transform

    def points_to_pixels(self, points):
        """Points to renderer units: identity, because the figure is drawn at 72 dpi."""
        return points                        # one unit is one point

    def get_image_magnification(self):
        """Factor by which matplotlib resamples images: image_dpi / 72."""
        return self.image_dpi / 72.0         # images (imshow) resampled at image_dpi, as in the PDF backend

    def new_gc(self):
        """A new graphics context (line and fill properties of the next primitive)."""
        return GraphicsContextBase()

    # ---- grouping: one PowerPoint group per axes and per legend --------------------------
    def open_group(self, s, gid=None):
        """Called by matplotlib when an artist group starts (s = "axes", "legend", "line2d", ...)."""
        self._stack.append([s, None])

    def close_group(self, s):
        """Called by matplotlib when the innermost open group ends."""
        if self._stack:
            self._stack.pop()

    def _shapes(self):
        """The shape collection new shapes go into (creating PowerPoint groups lazily, so no empty groups).

        Walks the open matplotlib groups from the outermost; each "axes" or "legend" group becomes a
        PowerPoint group shape (named "panel" or "legend") the first time a shape is added inside it, and
        nested groups (a legend inside an axes) become nested group shapes.
        """
        parent = self.slide.shapes
        for entry in self._stack:
            if entry[0] in GROUPED:
                if entry[1] is None:
                    entry[1] = parent.add_group_shape()
                    entry[1].name = "panel" if entry[0] == "axes" else entry[0]
                parent = entry[1].shapes
        return parent

    # ---- coordinate conversion ------------------------------------------------------------
    def _emu(self, x, y):
        """Display point (pt, y up) -> slide position (EMU, y down), relative to the slide's top-left corner.

        X = (x - x_origin) * 12700 and Y = (y_top - y) * 12700, rounded to whole EMU.
        """
        return (int(round((x - self.origin[0]) * EMU_PER_PT)),
                int(round((self.origin[1] - y) * EMU_PER_PT)))

    # ---- line and fill properties -----------------------------------------------------------
    def _style_line(self, shape, gc):
        """Copy width, color, alpha, dash pattern, cap and join from the graphics context to a shape outline.

        shape : python-pptx shape (autoshape, connector or freeform).
        gc    : matplotlib GraphicsContextBase of the primitive.
        A line width or opacity of zero removes the outline.  The opacity is the graphics context's alpha
        when it is forced (an artist's alpha), otherwise the alpha of the line color.
        """
        lw = gc.get_linewidth()
        rgba = gc.get_rgb()
        alpha = gc.get_alpha() if gc.get_forced_alpha() else (rgba[3] if len(rgba) > 3 else 1.0)
        if lw <= 0 or alpha <= 0:
            shape.line.fill.background()     # no outline
            return
        shape.line.width = Emu(int(round(lw * EMU_PER_PT)))
        shape.line.color.rgb = _rgb(rgba)
        ln = shape._element.spPr.find(qn("a:ln"))
        _set_alpha(ln, alpha)
        # matplotlib cap styles -> DrawingML line caps (flat, round, square)
        cap = {"butt": "flat", "round": "rnd", "projecting": "sq"}.get(gc.get_capstyle(), "flat")
        ln.set("cap", cap)
        offset, dashes = gc.get_dashes()     # dash and gap lengths in points; the offset is not used
        if dashes is not None and len(dashes) > 1:
            # custom dash pattern, in thousandths of a percent of the line width
            cust = etree.SubElement(ln, qn("a:custDash"))
            for d, sp in zip(dashes[0::2], dashes[1::2]):
                ds = etree.SubElement(cust, qn("a:ds"))
                ds.set("d", str(int(round(100000 * d / lw))))
                ds.set("sp", str(int(round(100000 * sp / lw))))
        join = {"miter": "a:miter", "round": "a:round", "bevel": "a:bevel"}.get(gc.get_joinstyle(), "a:round")
        etree.SubElement(ln, qn(join))
        # schema order inside a:ln is fill, dash, join, ends: move the dash/join after the fill
        for tag in ("a:custDash", "a:miter", "a:round", "a:bevel"):
            for el in ln.findall(qn(tag)):
                ln.remove(el); ln.append(el)

    def _style_fill(self, shape, gc, rgbFace):
        """Solid fill of a shape from the face color rgbFace (RGBA, 0-1), or no fill when rgbFace is None or
        fully transparent; the opacity follows the same rule as in _style_line."""
        if rgbFace is None:
            shape.fill.background()
            return
        alpha = gc.get_alpha() if gc.get_forced_alpha() else (rgbFace[3] if len(rgbFace) > 3 else 1.0)
        if alpha <= 0:
            shape.fill.background()
            return
        shape.fill.solid()
        shape.fill.fore_color.rgb = _rgb(rgbFace)
        _set_alpha(shape._element.spPr.find(qn("a:solidFill")), alpha)

    # ---- primitives ----------------------------------------------------------------------------
    def draw_path(self, gc, path, transform, rgbFace=None):
        """Draw a path: circular markers as ovals, fills as closed freeforms, strokes as lines.

        gc        : graphics context (stroke properties and the clip rectangle of the axes).
        path      : matplotlib Path in its own coordinates.
        transform : path coordinates -> display coordinates (points, y up).
        rgbFace   : fill color (RGBA, 0-1) or None for an unfilled path.
        Fills are clipped to the axes with _clip_polygon and strokes with _clip_polyline, since PowerPoint
        shapes cannot be clipped.
        """
        box = gc.get_clip_rectangle()
        box = None if box is None else (box.x0, box.y0, box.x1, box.y1)
        if rgbFace is not None and len(rgbFace) > 3 and rgbFace[3] == 0 and not gc.get_forced_alpha():
            rgbFace = None                   # fully transparent fill
        # circles (scatter and line markers) become native ovals, so they stay circles when edited
        if (len(path.vertices) == len(_UNIT_CIRCLE.vertices) and path.codes is not None
                and np.array_equal(path.codes, _UNIT_CIRCLE.codes)
                and np.allclose(path.vertices, _UNIT_CIRCLE.vertices)):
            # center and radii in points: images of the unit circle's center (0, 0) and corner (1, 1)
            c = transform.transform((0.0, 0.0)); e = transform.transform((1.0, 1.0))
            rx, ry = abs(e[0] - c[0]), abs(e[1] - c[1])
            if box is not None and not (box[0] - rx <= c[0] <= box[2] + rx
                                        and box[1] - ry <= c[1] <= box[3] + ry):
                return                       # marker entirely outside the axes
            left, top = self._emu(c[0] - rx, c[1] + ry)     # top-left corner of the bounding square
            shp = self._shapes().add_shape(MSO_SHAPE.OVAL, left, top,
                                           max(1, int(round(2 * rx * EMU_PER_PT))),      # at least 1 EMU
                                           max(1, int(round(2 * ry * EMU_PER_PT))))
            _strip_style(shp)
            self._style_fill(shp, gc, rgbFace)
            self._style_line(shp, gc)
            return
        subs = _subpaths(path, transform)
        if not subs:
            return
        stroke = gc.get_linewidth() > 0
        fill_polys = []
        if rgbFace is not None:
            # filled shape: one freeform with one contour per subpath that encloses an area
            for pts, _closed in subs:
                if len(pts) >= 3:
                    poly = _clip_polygon(pts, box) if box is not None else pts
                    if len(poly) >= 3:
                        fill_polys.append(poly)
        # a subpath of three or more points encloses an area and is outlined as a closed contour
        all_closed = all(closed or len(pts) >= 3 for pts, closed in subs)
        if fill_polys:
            # the outline goes on the same shape when every subpath is a closed outline
            self._freeform(fill_polys, closed=True, gc=gc if all_closed else None, rgbFace=rgbFace)
            if all_closed or not stroke:
                return
        if not stroke:
            return
        # stroke: clip each polyline; straight two-point lines (ticks, error bars) become connectors
        for pts, closed in subs:
            pts = list(pts) + ([pts[0]] if closed else [])   # close the outline explicitly
            pieces = _clip_polyline(pts, box) if box is not None else [pts]
            for piece in pieces:
                if len(piece) == 2:
                    (x0, y0), (x1, y1) = piece
                    a, b = self._emu(x0, y0), self._emu(x1, y1)
                    con = self._shapes().add_connector(MSO_CONNECTOR.STRAIGHT, a[0], a[1], b[0], b[1])
                    _strip_style(con)
                    self._style_line(con, gc)
                else:
                    self._freeform([piece], closed=False, gc=gc, rgbFace=None)

    def _freeform(self, polys, closed, gc, rgbFace):
        """One freeform shape from one or more contours (lists of display points).

        polys   : list of contours, each a list of (x, y) display points.
        closed  : close every contour (filled shapes) or leave them open (lines).
        gc      : graphics context for the outline, or None for a fill without outline.
        rgbFace : fill color (RGBA, 0-1) or None.
        python-pptx's FreeformBuilder takes local coordinates; with scale=1.0 they are slide EMU, and the
        shape is placed at the bounding box of all contours.
        """
        pts0 = [self._emu(*p) for p in polys[0]]
        xs = [p[0] for poly in polys for p in (self._emu(*q) for q in poly)]
        ys = [p[1] for poly in polys for p in (self._emu(*q) for q in poly)]
        if max(xs) == min(xs):
            pts0[0] = (pts0[0][0] + 1, pts0[0][1])   # avoid a zero-width path box
        if max(ys) == min(ys):
            pts0[0] = (pts0[0][0], pts0[0][1] + 1)   # avoid a zero-height path box
        fb = self._shapes().build_freeform(pts0[0][0], pts0[0][1], scale=1.0)
        fb.add_line_segments(pts0[1:], close=closed)
        for poly in polys[1:]:                       # further contours of the same shape
            pe = [self._emu(*p) for p in poly]
            fb.move_to(*pe[0])
            fb.add_line_segments(pe[1:], close=closed)
        shp = fb.convert_to_shape()
        _strip_style(shp)
        if gc is None:                       # fill only (the outline is drawn separately)
            self._style_fill(shp, GraphicsContextBase(), rgbFace)
            shp.line.fill.background()
            return
        self._style_fill(shp, gc, rgbFace if closed else None)
        self._style_line(shp, gc)

    def draw_image(self, gc, x, y, im, transform=None):
        """Place an RGBA image (row 0 = bottom, matplotlib's convention for vector backends) as a picture.

        gc        : graphics context (unused).
        x, y      : position of the image's lower-left corner in points.
        im        : (rows, columns, 4) uint8 RGBA array.
        transform : None for rasterized artists and resampled images, whose pixels are at image_dpi;
                    otherwise an affine transform that maps the unit square onto the image area, relative
                    to (x, y) (unsampled images; without rotation or shear).
        """
        h, w = im.shape[:2]
        if w == 0 or h == 0:
            return
        if transform is None:                # from MixedModeRenderer: pixels at image_dpi
            x0, y0 = x, y
            wpt, hpt = 72.0 * w / self.image_dpi, 72.0 * h / self.image_dpi
        else:                                # unsampled image: affine transform of the unit square
            # matrix [[a, c, e], [b, d, f]]: a and d are the width and height, e and f the offset
            a, b, c, d, e, f = transform.frozen().to_values()
            x0, y0, wpt, hpt = x + e, y + f, a, d
        arr = np.asarray(im)[::-1]           # PNG rows run top to bottom
        buf = BytesIO()
        Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
        buf.seek(0)
        left, top = self._emu(x0, y0 + hpt)  # the picture is placed by its top-left corner
        self._shapes().add_picture(buf, left, top, int(round(wpt * EMU_PER_PT)), int(round(hpt * EMU_PER_PT)))

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        """A text box in Arial whose baseline and alignment anchor match matplotlib's layout.

        gc     : graphics context (text color and alpha).
        x, y   : start of the text's baseline in points (y up), as laid out by matplotlib.
        s      : the string; prop : FontProperties; angle : rotation in degrees, counterclockwise.
        ismath : the string contains mathtext ($...$); the dollar signs are removed.
        mtext  : the Text artist (for its horizontal alignment), or None.
        The box is wider and taller than the text, so that the slightly different metrics of Arial in
        PowerPoint neither wrap nor clip it, and it is anchored so that the text keeps its alignment.
        """
        if not s.strip():
            return
        if ismath:
            s = plain_math(s)               # mathtext (for example log-axis tick labels) as Unicode
        w, _h, _d = self.get_text_width_height_descent(s, prop, ismath=False)   # width in points
        size = prop.get_size_in_points()
        bw, bh = w * 1.08 + 0.5 * size, size * 1.25          # generous box, no wrapping
        th = np.deg2rad(angle)
        ha = mtext.get_horizontalalignment() if (mtext is not None and angle % 360 == 0) else "left"
        if angle % 360 == 0:
            # anchor the box where matplotlib anchored the text, so alignment survives small font differences
            cy = y + ARIAL_LINE_CENTER * size    # vertical center of the line (the box is anchored middle)
            if ha == "center":
                left = x + w / 2 - bw / 2; align = PP_ALIGN.CENTER
            elif ha == "right":
                left = x + w - bw; align = PP_ALIGN.RIGHT
            else:
                left = x; align = PP_ALIGN.LEFT
            l, t = self._emu(left, cy + bh / 2)
            rot = 0.0
        else:
            # rotated text: place the box by its center, rotated about that center
            # center = baseline start + the vector (w / 2, ARIAL_LINE_CENTER * size) rotated by the angle
            cx = x + (w / 2) * np.cos(th) - ARIAL_LINE_CENTER * size * np.sin(th)
            cy = y + (w / 2) * np.sin(th) + ARIAL_LINE_CENTER * size * np.cos(th)
            l, t = self._emu(cx - bw / 2, cy + bh / 2)
            align = PP_ALIGN.CENTER
            rot = (-angle) % 360.0           # PowerPoint rotates clockwise, matplotlib counterclockwise
        tb = self._shapes().add_textbox(l, t, int(round(bw * EMU_PER_PT)), int(round(bh * EMU_PER_PT)))
        tb.rotation = rot
        tf = tb.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.word_wrap = False
        tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = align
        r = p.add_run()
        r.text = s
        f = r.font
        f.name = FONT_MAP.get(prop.get_name(), "Arial")
        f.size = Pt(size)
        weight = prop.get_weight()
        # bold for named bold weights or numeric weights of 600 and above
        f.bold = (weight in ("bold", "heavy", "black", "semibold", "demibold", "extra bold")
                  or (isinstance(weight, (int, float)) and weight >= 600))
        f.italic = prop.get_style() in ("italic", "oblique")
        rgba = gc.get_rgb()
        f.color.rgb = _rgb(rgba)
        alpha = gc.get_alpha() if gc.get_forced_alpha() else (rgba[3] if len(rgba) > 3 else 1.0)
        if alpha < 0.999:
            _set_alpha(r._r, alpha)          # r._r is the run's XML element (<a:r>)


# --------------------------------------------------------------------------- public entry point
def _tight_box(fig, pad_pt):
    """Bounding box (in points) of everything drawn, as matplotlib's savefig(bbox_inches='tight') uses.

    fig    : matplotlib Figure.
    pad_pt : padding added on every side, in points.
    The figure is laid out at 72 dpi (its dpi is restored afterwards); get_tightbbox returns inches, which
    are converted to points.  Returns (x0, y0, x1, y1) in figure points, y up from the bottom edge.
    """
    old_dpi = fig.dpi
    fig.set_dpi(72)
    try:
        bb = fig.get_tightbbox(fig.canvas.get_renderer())       # in inches
    finally:
        fig.set_dpi(old_dpi)
    x0, y0, x1, y1 = bb.x0 * 72 - pad_pt, bb.y0 * 72 - pad_pt, bb.x1 * 72 + pad_pt, bb.y1 * 72 + pad_pt
    return x0, y0, x1, y1


def _draw_on_slide(fig, slide, image_dpi, box):
    """Draw one figure on one slide (figure drawn at 72 dpi so that one unit is one point).

    fig       : matplotlib Figure.
    slide     : python-pptx slide.
    image_dpi : resolution of rasterized artists.
    box       : (x0, y0, x1, y1) area of the figure to show, in points (from _tight_box); its top-left
                corner (x0, y1) is placed at the slide's top-left corner.
    The figure's dpi and the hidden background patches are restored afterwards, so the figure can still be
    saved in other formats.
    """
    old_dpi = fig.dpi
    hidden = []
    # white figure and axes backgrounds would become large shapes that get in the way when editing
    for patch in [fig.patch] + [ax.patch for ax in fig.axes]:
        if patch.get_visible() and to_rgba(patch.get_facecolor())[:3] == (1.0, 1.0, 1.0):
            patch.set_visible(False); hidden.append(patch)
    try:
        fig.set_dpi(72)
        w_in, h_in = fig.get_size_inches()
        x0, y0, x1, y1 = box
        # MixedModeRenderer draws rasterized artists with Agg at image_dpi and hands the images to
        # RendererPPTX.draw_image; everything else goes to RendererPPTX directly
        renderer = MixedModeRenderer(fig, w_in, h_in, image_dpi,
                                     RendererPPTX(slide, x1 - x0, y1 - y0, image_dpi, origin=(x0, y1)))
        fig.draw(renderer)
    finally:
        fig.set_dpi(old_dpi)
        for patch in hidden:
            patch.set_visible(True)


def save_pptx(figs, path, image_dpi=300, tight=True, pad_pt=4.0):
    """Save one figure (or a list of figures, one per slide) as an editable .pptx file.

    figs      : a matplotlib Figure or a list of Figures.
    path      : output file (.pptx).
    image_dpi : resolution of rasterized artists (300).
    tight     : crop each slide to what is drawn (like savefig(bbox_inches='tight')), so panel letters
                and legends outside the axes grid are kept; otherwise use the full figure.
    pad_pt    : padding around the tight bounding box, in points.
    The figure keeps its print size (for example 180 mm wide for a double-column Nature figure); with
    several figures, the slide size is that of the largest.  Returns path.
    """
    figs = figs if isinstance(figs, (list, tuple)) else [figs]
    boxes = []
    for fig in figs:
        if tight:
            boxes.append(_tight_box(fig, pad_pt))
        else:
            w, h = fig.get_size_inches() * 72
            boxes.append((0.0, 0.0, w, h))
    prs = Presentation()
    w_pt = max(b[2] - b[0] for b in boxes)
    h_pt = max(b[3] - b[1] for b in boxes)
    prs.slide_width = Emu(int(round(w_pt * EMU_PER_PT)))
    prs.slide_height = Emu(int(round(h_pt * EMU_PER_PT)))
    blank = prs.slide_layouts[6]             # "Blank" layout of python-pptx's default template
    for fig, box in zip(figs, boxes):
        slide = prs.slides.add_slide(blank)
        _draw_on_slide(fig, slide, image_dpi, box)
    prs.save(str(path))
    return path
