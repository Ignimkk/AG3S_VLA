"""CVPR 2027 Fig. 1 v3: old Fig. 1 (teaser) + old Fig. 2 (three-axis concept) merged into one full-width figure.

Spec: paper/figures/fig01_teaser/FIG1_V3_FINAL_PROMPT_KO.md (2026-11-09).
  top strip (optional)  SafeLIBERO spatial Level I, task 2, episode 9 (seed 7, init state 9): VLA only vs VLA + AG3S
  three cards           identity / extent / permission, each "Prior" (left) vs "AG3S (ours)" (right), square panels

Every raster comes from paper/figures/fig01_teaser/v3/assets/ (made by fig1_v3_assets.py; provenance in assets.json).
Prior panels are crops of earlier baseline documentation figures (previous container runs, LIBERO scenes); the
only edits are (a) inpainting raster name tags / legends that would print below 7 pt and (b) re-tinting the
Grounded-SAM-2 mask from blue to orange so it follows this figure's colour code (orange = selected target).
Both edits are recorded per panel in the sidecar. No number is printed in the figure.

Layout is authored at the CVPR text width (6.875 in), so font sizes are printed sizes (floor 7 pt).
The Fig class / fit() / layout_audit() are copied verbatim from cvpr_v1_compose.py (that module reads files from the
previous container at import time and cannot be imported here); the original is not modified.

    .venv-ag3s/bin/python paper/tools/plotting/cvpr_v3_fig1_compose.py          # both variants
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.text as mtext  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3]
V3 = REPO / "paper/figures/fig01_teaser/v3"
ASSETS = V3 / "assets"
TEXTWIDTH_IN = 6.875          # cvpr.sty \textwidth, letter, two column
FS = 7.0                      # font floor (pt, printed size)
PDF_DPI, PNG_DPI = 400, 300   # raster panels inside the PDF / preview PNG
STRICT_LAYOUT = os.environ.get("CVPR_V1_STRICT", "1") != "0"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "mathtext.fontset": "stix", "axes.linewidth": 0.5, "savefig.facecolor": "white"})

# same colour code as cvpr_v1 (orange = selected target, blue = protected neighbour, gold = seeds / contact links,
# purple = attached object, red ✗ prior, green ✓ ours)
C = {"target": "#E8711A", "neigh": "#2E75B6", "gold": "#D9A406", "goldfill": "#FFF2CC", "attached": "#7030A0",
     "red": "#C62828", "green": "#2E7D32", "navyedge": "#5B7FA6", "ctrledge": "#7B6A9A", "ink": "#1F1F1F",
     "mute": "#555555", "darkgold": "#8C6D00"}

RUN_VLA = "20261108T233915_4ef922024674_qe-vla-e1a-spatial-LI-t2-LI-vla_only"
RUN_AG3S = ("20261109T080310_a811331c11d8_qe-pe123458-e1a-spatial-LI-t2-LI-vla_v11ab-AG3S-commonto-att-termsoff-pass-zr-ee-"
            "hdp-loose2-obst4-pts-nocarve-fh-t100-pe123458-C")


def sha(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


ASSET_META = json.loads((ASSETS / "assets.json").read_text())


# ------------------------------------------------------------------------------------------------ copied helpers
def fit(a, asp, anchor=(0.5, 0.5)):
    """Centre-crop ``a`` to aspect ``asp`` (w/h); anchor = where the kept window sits in the removable range."""
    h, w = a.shape[:2]
    if w / h > asp:
        nw = int(h * asp); x0 = int((w - nw) * anchor[0]); return a[:, x0:x0 + nw], (x0, 0, nw, h)
    nh = int(w / asp); y0 = int((h - nh) * anchor[1]); return a[y0:y0 + nh], (0, y0, w, nh)




class Fig:
    """Inch-based drawing on a figure of exactly W x H inches."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.fig = plt.figure(figsize=(W, H))
        self.A = None            # (v3: assets tracked by the caller)
        self.printed = []       # (text, meaning, source, key)
        self.segments = []      # [(x0, y0, x1, y1), kind] in inches: arrows and lines (for the overlap audit)
        self.panels = []        # [(x, y, w, h)] in inches: image panels
        self.badges = []        # Text artists of in-panel badges (equal-height audit)

    def F(self, x, y, w, h):
        return [x / self.W, y / self.H, w / self.W, h / self.H]

    def text(self, x, y, s, size=FS, **kw):
        kw.setdefault("ha", "left"); kw.setdefault("va", "center")
        return self.fig.text(x / self.W, y / self.H, s, fontsize=size, **kw)

    def panel(self, rect, a, border="#BBBBBB", lw=0.8, anchor=(0.5, 0.5), badge=None, bcol=None, bsize=FS, zorder=3,
              labels=None, interpolation="lanczos", badge_y=0.10):
        x, y, w, h = rect
        self.panels.append((x, y, w, h))
        ax = self.fig.add_axes(self.F(x, y, w, h), zorder=zorder)
        img, (cx0, cy0, cw, ch) = fit(a, w / h, anchor)
        ax.imshow(img, aspect="auto", interpolation=interpolation)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(border); sp.set_linewidth(lw)
        if badge:
            self.badges.append(ax.text(0.10 / w, badge_y / h, badge, transform=ax.transAxes, fontsize=bsize, fontweight="bold", color="white",
                                       va="baseline", bbox=dict(boxstyle="round,pad=0.25", fc=bcol, ec="none"), zorder=5, linespacing=1.05))
        if labels:     # vector replacements for raster label boxes: (x, y, text[, va]) in pixel coords of ``a``
            for lab_ in labels:
                lx, ly, t = lab_[:3]
                va = lab_[3] if len(lab_) > 3 else "bottom"
                u, v = (lx - cx0) / cw, 1 - (ly - cy0) / ch
                if not (0 <= u <= 1 and 0 <= v <= 1):
                    continue
                v = v + (0.07 if va == "bottom" else -0.07)      # keep the labelled object visible
                mx = min(0.42 / w, 0.45)
                lo = min((0.02 if va == "bottom" else 0.16) / h + 0.05 / h, 0.45)
                hi = min((0.16 if va == "bottom" else 0.02) / h + 0.05 / h, 0.45)
                u, v = float(np.clip(u, mx, 1 - mx)), float(np.clip(v, lo, 1 - hi))
                ax.text(u, v, t.replace("other fruit", "fruit"), transform=ax.transAxes, fontsize=FS, fontweight="bold", color=C["ink"],
                        ha="center", va=va, bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#333333", lw=0.5), zorder=6)
        return ax

    def band(self, x, y, w, h, fc, ec, lw=0.7, z=-1, ls="-", r=0.006):
        self.fig.add_artist(FancyBboxPatch((x / self.W, y / self.H), w / self.W, h / self.H, boxstyle=f"round,pad=0,rounding_size={r}",
                                           transform=self.fig.transFigure, fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))

    def box(self, x, y, w, h, s, fc, ec="#555555", size=FS, bold=False, ls="-", lw=0.7, color=None, z=2):
        self.band(x, y, w, h, fc, ec, lw=lw, z=z, ls=ls, r=0.004)
        self.text(x + w / 2, y + h / 2, s, size=size, ha="center", va="center", fontweight="bold" if bold else "normal",
                  color=color or C["ink"], zorder=z + 3, linespacing=1.1)

    def arrow(self, a, b, color="#555555", lw=0.8, ms=5, ls="-", z=4):
        self.segments.append(((a[0], a[1], b[0], b[1]), "arrow"))
        self.fig.add_artist(FancyArrowPatch((a[0] / self.W, a[1] / self.H), (b[0] / self.W, b[1] / self.H), transform=self.fig.transFigure,
                                            arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, ls=ls, zorder=z,
                                            shrinkA=0, shrinkB=0))

    def line(self, pts, color, lw=0.8, z=1, ls="-"):
        for p0, p1 in zip(pts[:-1], pts[1:]):
            self.segments.append(((p0[0], p0[1], p1[0], p1[1]), "line"))
        self.fig.add_artist(Line2D([p[0] / self.W for p in pts], [p[1] / self.H for p in pts], transform=self.fig.transFigure,
                                   color=color, lw=lw, zorder=z, ls=ls))


    def audit(self):
        sizes = [t.get_fontsize() for t in self.fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()]
        return float(min(sizes)), float(max(sizes)), len(sizes)

    # ------------------------------------------------------------------ tidiness audit (fourth pass)
    def layout_audit(self, tol_pt=1.0, border_in=0.05):
        """Display-space checks: no text-text overlap, no text-arrow/line crossing (unless the text has an opaque
        background box drawn above the line), text >= border_in from every image-panel border, equal badge heights.
        Returns a dict of violation lists (empty = clean)."""
        fig = self.fig
        fig.canvas.draw()
        ren = fig.canvas.get_renderer()
        dpi = fig.dpi
        tol = tol_pt * dpi / 72.0
        texts = []
        for t in fig.findobj(mtext.Text):
            if not (t.get_text().strip() and t.get_visible()):
                continue
            try:
                bb = t.get_window_extent(ren)
            except Exception:
                continue
            if not np.isfinite([bb.x0, bb.y0, bb.x1, bb.y1]).all() or bb.width == 0:
                continue
            patch = t.get_bbox_patch()
            opaque = False
            if patch is not None:
                try:
                    bb = patch.get_window_extent(ren)
                    fc = patch.get_facecolor()
                    opaque = fc[3] > 0.5
                except Exception:
                    pass
            texts.append((t, bb, opaque, t.get_zorder()))
        out = {"text_text": [], "text_segment": [], "panel_border": [], "badge_heights": []}

        def shrink(bb, d):
            from matplotlib.transforms import Bbox
            return Bbox.from_extents(bb.x0 + d, bb.y0 + d, bb.x1 - d, bb.y1 - d)
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                a, b = shrink(texts[i][1], tol / 2), shrink(texts[j][1], tol / 2)
                if a.overlaps(b):
                    out["text_text"].append((texts[i][0].get_text()[:30], texts[j][0].get_text()[:30]))
        # segments (inches -> display)
        def disp(x, y):
            return fig.transFigure.transform((x / self.W, y / self.H))

        def seg_hits_box(p, q, bb):
            # Liang-Barsky clip of segment p-q against bbox
            x0, y0, x1, y1 = bb.x0, bb.y0, bb.x1, bb.y1
            dx, dy = q[0] - p[0], q[1] - p[1]
            t0, t1 = 0.0, 1.0
            for pk, qk in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
                if pk == 0:
                    if qk < 0:
                        return False
                    continue
                r = qk / pk
                if pk < 0:
                    if r > t1:
                        return False
                    t0 = max(t0, r)
                else:
                    if r < t0:
                        return False
                    t1 = min(t1, r)
            return t0 <= t1
        for (x0, y0, x1, y1), kind in self.segments:
            p, q = disp(x0, y0), disp(x1, y1)
            for t, bb, opaque, z in texts:
                if opaque and z >= 4:
                    continue           # label box intentionally covers the line
                if seg_hits_box(p, q, shrink(bb, tol)):
                    out["text_segment"].append((kind, round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2), t.get_text()[:30]))
        # text vs panel borders
        bd = border_in * dpi
        for (px, py, pw, ph) in self.panels:
            P0, P1 = disp(px, py), disp(px + pw, py + ph)
            for t, bb, opaque, z in texts:
                inside = bb.x0 >= P0[0] and bb.x1 <= P1[0] and bb.y0 >= P0[1] and bb.y1 <= P1[1]
                if inside:
                    d = min(bb.x0 - P0[0], P1[0] - bb.x1, bb.y0 - P0[1], P1[1] - bb.y1)
                else:
                    dx = max(P0[0] - bb.x1, bb.x0 - P1[0], 0); dy = max(P0[1] - bb.y1, bb.y0 - P1[1], 0)
                    d = (dx ** 2 + dy ** 2) ** 0.5
                    if dx == 0 and dy == 0:
                        d = 0.0        # straddles the border
                if d < bd - 0.5:
                    out["panel_border"].append((t.get_text()[:30], round(d / dpi, 3), "inside" if inside else "outside"))
        hs = []
        for t in self.badges:
            if "\n" in t.get_text():
                continue
            patch = t.get_bbox_patch()
            hs.append(patch.get_window_extent(ren).height / dpi * 72 if patch is not None else t.get_window_extent(ren).height / dpi * 72)
        if hs and max(hs) - min(hs) > 0.5:
            out["badge_heights"].append([round(h, 2) for h in hs])
        return out


# ------------------------------------------------------------------------------------------------ raster edits
def inpaint(a, boxes, pad=5, iters=800):
    """Fill each (x0, y0, x1, y1) box by harmonic (4-neighbour average) diffusion from its border."""
    a = a.astype(float).copy()
    H, W = a.shape[:2]
    m = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in boxes:
        m[max(y0 - pad, 0):min(y1 + pad, H), max(x0 - pad, 0):min(x1 + pad, W)] = True
    a[m] = a[~m].mean(0)
    for _ in range(iters):
        p = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="edge")
        nb = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4.0
        a[m] = nb[m]
    return np.clip(a, 0, 255).astype(np.uint8), int(m.sum())


def retint_blue_to_orange(a):
    """Re-tint the saturated blue mask fill to the target orange hue (saturation / value kept)."""
    hsv = mcolors.rgb_to_hsv(a.astype(float) / 255.0)
    sel = (hsv[..., 0] > 0.52) & (hsv[..., 0] < 0.70) & (hsv[..., 1] > 0.35)
    hsv[sel, 0] = mcolors.rgb_to_hsv(np.array(mcolors.to_rgb(C["target"])))[0]
    return (mcolors.hsv_to_rgb(hsv) * 255).round().astype(np.uint8), int(sel.sum())


# label boxes (pixel coords of the asset crop) whose raster text would print at ~1.5 pt in a 1 in panel
TAGS = {
    "crop_aegis__v1-aegis-scene-detection-ellipsoid_r0c1": [(55, 157, 148, 171), (97, 220, 186, 238)],
    "crop_knows__v1-knows-scene-ellipsoids_r0c0": [(132, 127, 224, 139), (147, 140, 206, 150), (0, 163, 79, 175),
                                                   (312, 212, 399, 225), (256, 329, 305, 342), (6, 464, 88, 499)],
}
TRIM = {"crop_grounded_sam2_track__v2-sl3-track-strip_r0c0": 2}     # light panel frame of the source strip


class Panels:
    """Loads each asset once, applies the recorded edits, square-crops it and keeps the provenance for the sidecar."""

    def __init__(self):
        self.used = []

    def get(self, name, slot, kind):
        p = ASSETS / f"{name}.png"
        a = np.asarray(Image.open(p).convert("RGB"))
        edits = []
        if name in TRIM:
            t = TRIM[name]
            a = a[t:-t, t:-t]
            edits.append(f"trimmed {t} px source-panel frame on each side")
        if name in TAGS:
            a, n = inpaint(a, TAGS[name])
            edits.append(f"inpainted {len(TAGS[name])} raster name tag / legend boxes ({n} px) that would print below 7 pt: "
                         + json.dumps(TAGS[name]))
        if name.startswith("crop_grounded_sam2"):
            a, n = retint_blue_to_orange(a)
            edits.append(f"re-tinted the blue Grounded-SAM-2 per-frame mask fill to orange {C['target']} ({n} px; hue only) "
                         "to follow this figure's colour code; green contour (ground-truth target) and orange contour "
                         "(same-category distractor) of the source figure kept")
        sq, (cx0, cy0, cw, ch) = fit(a, 1.0)
        meta = ASSET_META.get(f"{name}.png", {})
        rec = dict(slot=slot, file=str(p.relative_to(REPO)), sha256=sha(p), size_px=list(a.shape[1::-1]),
                   square_crop_px=[cx0, cy0, cw, ch], edits=edits, panel_kind=kind, provenance=meta)
        if kind == "recorded run":
            rec["run"] = RUN_VLA if name.startswith("top_vla") else RUN_AG3S
        self.used.append(rec)
        return sq


# ------------------------------------------------------------------------------------------------ content
INSTR = "pick up the black bowl on the stove and place it on the plate"    # episodes.jsonl task_description, t2
TOP = [("top_vla_step111", "VLA only", C["red"], "✗ bottle toppled", C["red"]),
       ("top_ag3s_step111", "VLA + AG3S", C["green"], "✓ bottle upright", C["gold"]),
       ("top_ag3s_step139", "VLA + AG3S, later", C["green"], "✓ bowl placed", C["gold"])]
CARDS = [
    dict(title="1 What: identity", sub="the noun's attention names the object", ec=C["gold"], tc=C["darkgold"], fc="#FFFDF5",
         plabel="Prior: segmenter", prior="crop_grounded_sam2_track__v2-sl3-track-strip_r0c0", pbadge="✗ other bowl",
         ours="c1_ours_attention", obadge="✓ target bowl"),
    dict(title="2 Where it ends: extent", sub="depth continuity sets the boundary", ec=C["navyedge"], tc="#2F5597", fc="#F7FAFD",
         plabel="Prior: ellipsoid", prior="crop_aegis__v1-aegis-scene-detection-ellipsoid_r0c1", pbadge="✗ one blob",
         ours="c2_ours_extent", obadge="✓ surface only"),
    dict(title="3 Who and when: permission", sub="contact links only, in the allowed stage", ec=C["ctrledge"], tc="#5B4A7A",
         fc="#F8F6FB", plabel="Prior: EE filter", prior="crop_knows__v1-knows-scene-ellipsoids_r0c0", pbadge="✗ EE path only",
         ours="c3_ours_links", obadge="✓ held bowl"),
]
LEGEND = [("patch", C["target"], "selected target surface"), ("patch", C["neigh"], "protected neighbor"),
          ("patch", C["gold"], "seeds / contact links"), ("patch", C["attached"], "attached object")]

GAP, MARGIN = 0.06, 0.02
CW = (TEXTWIDTH_IN - 2 * MARGIN - 2 * GAP) / 3            # card width
P = (CW - 2 * 0.06 - 0.06) / 2                            # square panel side inside a card
CARD_TOP_PAD = 0.43                                       # title, subtitle, row label above the panels
CH = CARD_TOP_PAD + P + 0.05                              # card height
T = 1.13                                                  # top-strip panel side


def legend(f, x, y, ncol, colw, rowh=0.13):
    for k, (_, col, lab) in enumerate(LEGEND):
        cx, cy = x + (k % ncol) * colw, y - (k // ncol) * rowh
        f.fig.add_artist(Rectangle(((cx) / f.W, (cy - 0.035) / f.H), 0.12 / f.W, 0.07 / f.H, transform=f.fig.transFigure,
                                   fc=col, ec="none", zorder=3))
        f.text(cx + 0.16, cy, lab, size=FS, color=C["ink"])


def cards(f, PN, y0):
    for i, c in enumerate(CARDS):
        x = MARGIN + i * (CW + GAP)
        top = y0 + CH
        f.band(x, y0, CW, CH, c["fc"], c["ec"], lw=0.9)
        f.text(x + 0.06, top - 0.09, c["title"], fontweight="bold", color=c["tc"])
        f.text(x + 0.06, top - 0.205, c["sub"], style="italic", color=C["mute"])
        px = [x + 0.06, x + 0.06 + P + 0.06]
        f.text(px[0], top - 0.32, c["plabel"], fontweight="bold", color=C["red"])
        f.text(px[1], top - 0.32, "AG3S (ours)", fontweight="bold", color=C["green"])
        py = top - CARD_TOP_PAD - P
        f.panel((px[0], py, P, P), PN.get(c["prior"], f"{c['title']} / prior", "previous-container figure crop"),
                border=C["red"], lw=0.9, badge=c["pbadge"], bcol=C["red"], badge_y=0.11)
        f.panel((px[1], py, P, P), PN.get(c["ours"], f"{c['title']} / ours", "recorded run"),
                border=C["gold"], lw=0.9, badge=c["obadge"], bcol=C["green"], badge_y=0.11)


def fig01_v3(top_strip: bool):
    W = TEXTWIDTH_IN
    PN = Panels()
    if top_strip:
        y_cards = MARGIN
        y_top = y_cards + CH + 0.05
        H = round(y_top + T + 0.18, 3)
    else:
        y_cards = 0.17
        H = round(y_cards + CH + MARGIN, 3)
    f = Fig(W, H)
    cards(f, PN, y_cards)
    if top_strip:
        x_strip = W - MARGIN - 3 * T - 2 * GAP
        for k, (name, lab, lcol, badge, bord) in enumerate(TOP):
            x = x_strip + k * (T + GAP)
            f.text(x, y_top + T + 0.105, lab, fontweight="bold", color=lcol)
            f.panel((x, y_top, T, T), PN.get(name, f"top / {lab}", "recorded run"), border=bord, lw=0.9, badge=badge,
                    bcol=C["red"] if badge.startswith("✗") else C["green"], badge_y=0.11)
        lx = MARGIN + 0.04
        ly = y_top + T + 0.105
        f.text(lx, ly, "SafeLIBERO, Level I: one scene, two runs", fontweight="bold", color=C["ink"])
        f.text(lx, ly - 0.15, f"“{INSTR}”", style="italic", color=C["ink"])
        f.text(lx, ly - 0.30, "obstacle: wine bottle next to the bowl's path", color=C["mute"])
        f.text(lx, ly - 0.43, "frozen VLA alone vs. with AG3S, same initial state and seed", color=C["mute"])
        f.text(lx, ly - 0.62, "Below, per axis: closest prior practice vs. AG3S", fontweight="bold", color=C["ink"])
        f.text(lx, ly - 0.75, "prior: earlier baseline runs on other LIBERO scenes", color=C["mute"])
        f.text(lx, ly - 0.88, "ours: the AG3S episode shown above", color=C["mute"])
        legend(f, lx, y_top + 0.17, ncol=2, colw=1.55)
    else:
        legend(f, MARGIN + 0.04, 0.085, ncol=4, colw=1.62)
    stem = "fig01_v3" if top_strip else "fig01_v3_notop"
    save(f, PN, stem, dict(top_strip=top_strip))


def save(f, PN, stem, extra):
    mn, mx, cnt = f.audit()
    assert mn >= FS - 1e-6, f"{stem}: font below {FS} pt: {mn}"
    viol = f.layout_audit()
    n_viol = sum(len(v) for v in viol.values())
    print(f"[{stem}] {f.W:.3f} x {f.H:.3f} in, layout audit: " + ", ".join(f"{k}={len(v)}" for k, v in viol.items()))
    for k, v in viol.items():
        for item in v[:40]:
            print("   ", k, item)
    texts = [t.get_text() for t in f.fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()]
    if STRICT_LAYOUT:
        assert n_viol == 0, f"{stem}: {n_viol} layout violations"
    pdf = V3 / f"{stem}.pdf"
    f.fig.savefig(pdf, dpi=PDF_DPI)
    f.fig.savefig(pdf.with_suffix(".png"), dpi=PNG_DPI)
    plt.close(f.fig)
    side = dict(
        artifact=stem, pdf=str(pdf.relative_to(REPO)), png=str(pdf.with_suffix(".png").relative_to(REPO)),
        size_in=[f.W, f.H], textwidth_in=TEXTWIDTH_IN, raster_dpi_in_pdf=PDF_DPI, png_dpi=PNG_DPI,
        fonts=dict(family="DejaVu Sans", pdf_fonttype=42, min_pt=mn, max_pt=mx, n_text_items=cnt, floor_pt=FS),
        printed_text=texts, printed_numbers=[],
        layout_audit=dict(tolerance_pt=1.0, min_text_to_panel_border_in=0.05, violations={k: len(v) for k, v in viol.items()},
                          n_panels=len(f.panels), n_texts=cnt),
        panels=PN.used,
        runs=dict(vla_only=RUN_VLA, ag3s=RUN_AG3S,
                  note="VLA-only run (2026-11-08, commit 4ef922024674) and AG3S run (2026-11-09, commit a811331c11d8) differ "
                       "in commit; seed 7 and initial state 9 are the same. SafeLIBERO spatial Level I task 2 episode 9."),
        prior_scenes=dict(
            identity="Grounded-SAM-2 per-frame, instruction prompt; LIBERO spatial t2 e0 (task 'from table center', no obstacle), "
                     "head camera obs 10; source AG3S/ag3s/docs/figures/baselines/grounded_sam2_track/v2-sl3-track-strip.png "
                     "(BL-4, 2026-10-24, previous container)",
            extent="AEGIS detector_names variant; LIBERO spatial t4_e0 (bowl in the top drawer of the wooden cabinet, no obstacle), "
                   "call 8 obs 50, open-loop replay; detector boxes of flat_stove_1 and wooden_cabinet_1 both span the cabinet, "
                   "so their ellipsoids (semi-axes ~0.15 m) contain the target bowl; source aegis/v1-aegis-scene-detection-ellipsoid.png",
            permission="KNOWS; LIBERO spatial t4_e0 call 12 obs 70, object ellipsoids fitted from simulator segmentation "
                       "(privileged) and the nominal (white) vs filtered (orange) end-effector path; no attached-object model. "
                       "The green ellipse is the flat_stove_1 object ellipsoid (not an end-effector ellipsoid); "
                       "source knows/v1-knows-scene-ellipsoids.png"),
        honesty=["prior panels are different scenes (LIBERO, no obstacle) from previous-container runs, not the top-row episode",
                 "c1 ours uses per-point attention values; the stored 16x16 attention_head map is not used",
                 "camera renders are 256 px upsampled 3x (768 px), no further sharpening",
                 "no number printed in the figure"],
        script="paper/tools/plotting/cvpr_v3_fig1_compose.py", generated=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        git_head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
        is_measured_result=False, scope="conceptual comparison on real renders / recorded baseline figures; no numbers",
    )
    side.update(extra)
    pdf.with_suffix(".json").write_text(json.dumps(side, indent=1, ensure_ascii=False, default=str) + "\n")
    print(f"wrote {pdf.relative_to(REPO)}  min font {mn} pt")


if __name__ == "__main__":
    which = sys.argv[1:] or ["top", "notop"]
    for w in which:
        fig01_v3(top_strip=(w == "top"))
