"""CVPR 2027 v1 main figures (Fig 1 teaser, Fig 2 overview, Fig 3 pipeline, Fig 6 episode strip).

Everything is laid out natively at the CVPR text width (6.875 in) so that every font size in the file is the
printed size (floor: 7 pt, audited automatically and written to the sidecar). Photographic panels are MuJoCo
replay renders reused read-only from paper/figures/ifg_overview/assets (IFG study, development episodes),
paper/figures/paperbanana/figure2/assets (older serving run; contact-interface illustration only) and the new
hurdle frames in paper/figures/cvpr_v1/assets (cvpr_v1_fig1_scenes.py). No research code is imported.

Numbers printed on the figures are copied from the files listed in NUMBERS below and repeated, with their JSON
pointer, in each figure's sidecar. Nothing is re-measured here.

    .venv-ag3s/bin/python paper/tools/plotting/cvpr_v1_compose.py            # all four
    .venv-ag3s/bin/python paper/tools/plotting/cvpr_v1_compose.py fig01 fig03  # subset
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
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.text as mtext  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3]
STORAGE = pathlib.Path("/mnt/dev/AG3S_VLA-storage")
IFG_A = REPO / "paper/figures/ifg_overview/assets"
PB2_A = REPO / "paper/figures/paperbanana/figure2/assets"
PB3_A = REPO / "paper/figures/paperbanana/figure3/assets"
SC_A = REPO / "paper/figures/cvpr_v1/assets"
OUT = REPO / "paper/figures/cvpr_v1"
TEXTWIDTH_IN = 6.875          # cvpr.sty \textwidth, letter, two column
FS = 7.0                      # font floor (pt, printed size)
PDF_DPI, PNG_DPI = 400, 300   # raster panels inside the PDF / preview PNG
STRICT_LAYOUT = os.environ.get("CVPR_V1_STRICT", "1") != "0"   # set CVPR_V1_STRICT=0 to only report layout violations

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "mathtext.fontset": "stix", "axes.linewidth": 0.5, "savefig.facecolor": "white"})

# paperbanana shared colour code (paper/figures/paperbanana/README.md) + IFG draft colours
C = {"target": "#E8711A", "neigh": "#2E75B6", "gold": "#D9A406", "goldfill": "#FFF2CC", "attached": "#7030A0",
     "hold": "#C00000", "red": "#C62828", "green": "#2E7D32", "navy": "#DEEBF7", "navyedge": "#5B7FA6",
     "field": "#DAE8FC", "state": "#FBE5D6", "stateedge": "#C55A11", "ctrl": "#E4DFEC", "ctrledge": "#7B6A9A",
     "grey": "#E7E6E6", "ink": "#1F1F1F", "teal": "#2B7A78", "tealfill": "#DDF0EE", "mute": "#555555",
     "bar_prior": "#9E9E9E", "darkgold": "#8C6D00"}

# ------------------------------------------------------------------------------------------------ sources
HELD = STORAGE / "results/development/20261016T030741+0900_f15e587290f6_19ea09c7a8b8/heldout_locked/summary.json"
HELD_LOCAL = REPO / "AG3S/ag3s/docs/figures/ifg/ifg-heldout-locked.json"
T43W = REPO / "AG3S/ag3s/docs/handoff/T43W.verify.json"
HURDLE = STORAGE / "results/development/20261017T171201+0900_f15e587290f6_137b5f75d9e3/verify/IFG-HURDLE/agg.json"
T58V_PATH = REPO / "AG3S/ag3s/docs/handoff/T58.verify.json"


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jget(path, pointer):
    """Read a JSON pointer (``/a/b/0``) from a file."""
    cur = json.loads(pathlib.Path(path).read_text())
    for tok in pointer.strip("/").split("/"):
        cur = cur[int(tok)] if isinstance(cur, list) else cur[tok]
    return cur


def load_numbers():
    """Every number any figure may print, read from its source; the sidecars repeat this table."""
    held = json.loads(HELD.read_text())["main_success"]
    hur = json.loads(HURDLE.read_text())["arms"]
    e0 = json.loads(T43W.read_text())["numbers"]["summary_per_arm"]["E0/H"]
    n = {
        "heldout_iou_GN": dict(value=held["G:N"]["iou"]["episode_mean"], source=HELD, key="/main_success/G:N/iou/episode_mean"),
        "heldout_iou_A": dict(value=held["A"]["iou"]["episode_mean"], source=HELD, key="/main_success/A/iou/episode_mean"),
        "heldout_iou_B": dict(value=held["B"]["iou"]["episode_mean"], source=HELD, key="/main_success/B/iou/episode_mean"),
        "heldout_prec_GN": dict(value=held["G:N"]["precision"]["episode_mean"], source=HELD, key="/main_success/G:N/precision/episode_mean"),
        "heldout_prec_B": dict(value=held["B"]["precision"]["episode_mean"], source=HELD, key="/main_success/B/precision/episode_mean"),
        "heldout_nfp_GN": dict(value=held["G:N"]["neighbor_fp_sum"], source=HELD, key="/main_success/G:N/neighbor_fp_sum"),
        "heldout_nfp_A": dict(value=held["A"]["neighbor_fp_sum"], source=HELD, key="/main_success/A/neighbor_fp_sum"),
        "heldout_nfp_B": dict(value=held["B"]["neighbor_fp_sum"], source=HELD, key="/main_success/B/neighbor_fp_sum"),
        "heldout_n_episodes": dict(value=held["B"]["iou"]["n_episodes"], source=HELD, key="/main_success/B/iou/n_episodes"),
        "heldout_n_frames": dict(value=held["B"]["n_valid_frames"], source=HELD, key="/main_success/B/n_valid_frames"),
        "heldout_paired_A_GN": dict(value=held["paired_iou"]["A - G:N"], source=HELD, key="/main_success/paired_iou/A - G:N"),
        "hurdle_E0_success": dict(value=e0["success"], source=T43W, key="/numbers/summary_per_arm/E0~1H/success"),
        "hurdle_E0_n": dict(value=e0["n"], source=T43W, key="/numbers/summary_per_arm/E0~1H/n"),
        "hurdle_E0_obstacle_stop": dict(value=e0["runs_obstacle_stop"], source=T43W, key="/numbers/summary_per_arm/E0~1H/runs_obstacle_stop"),
        "hurdle_SURF_success_ct": dict(value=hur["SURF"]["S_contact"], source=HURDLE, key="/arms/SURF/S_contact"),
        "hurdle_SURF_n": dict(value=hur["SURF"]["n"], source=HURDLE, key="/arms/SURF/n"),
        "hurdle_C1a_success_ct": dict(value=hur["C1a"]["S_contact"], source=HURDLE, key="/arms/C1a/S_contact"),
    }
    t58 = json.loads(T58V_PATH.read_text())["numbers"]["on_diagnosis"]["1830"]["cover_rows"]
    n["t58_outside_mm"] = dict(value=[r for r in t58 if r["t"] == 55][0]["closest_point_outside_spheres_mm"], source=T58V_PATH,
                               key="/numbers/on_diagnosis/1830/cover_rows[t=55]/closest_point_outside_spheres_mm")
    # derived, printed as given in the figure brief
    n["heldout_nfp_reduction_B_pct"] = dict(value=100.0 * (1 - n["heldout_nfp_B"]["value"] / n["heldout_nfp_GN"]["value"]),
                                            source=HELD, key="100*(1 - /main_success/B/neighbor_fp_sum / /main_success/G:N/neighbor_fp_sum)")
    n["heldout_nfp_reduction_A_pct"] = dict(value=100.0 * (1 - n["heldout_nfp_A"]["value"] / n["heldout_nfp_GN"]["value"]),
                                            source=HELD, key="100*(1 - /main_success/A/neighbor_fp_sum / /main_success/G:N/neighbor_fp_sum)")
    # cross-checks against the brief (fail loudly if the sources moved)
    assert abs(n["heldout_iou_GN"]["value"] - 0.773) < 5e-4 and abs(n["heldout_iou_B"]["value"] - 0.891) < 5e-4
    assert abs(n["heldout_iou_A"]["value"] - 0.876) < 5e-4
    assert round(n["heldout_nfp_GN"]["value"]) == 176070 and round(n["heldout_nfp_B"]["value"]) == 13582
    assert n["hurdle_E0_success"]["value"] == 0 and n["hurdle_E0_n"]["value"] == 14 and n["hurdle_E0_obstacle_stop"]["value"] == 13
    assert n["hurdle_SURF_success_ct"]["value"] == 7 and n["hurdle_SURF_n"]["value"] == 14 and n["hurdle_C1a_success_ct"]["value"] == 11
    assert n["heldout_n_episodes"]["value"] == 34 and n["heldout_n_frames"]["value"] == 2523
    return n


NUMBERS = load_numbers()


# ------------------------------------------------------------------------------------------------ images
class Assets:
    """Tracks every raster file a figure uses (for the sidecar) and loads crops."""

    def __init__(self):
        self.used = {}

    def path(self, name):
        for d in (SC_A, IFG_A, PB2_A, PB3_A):
            if (d / f"{name}.png").exists():
                return d / f"{name}.png"
        raise FileNotFoundError(name)

    def meta(self, name):
        p = self.path(name).with_suffix(".json")
        return json.loads(p.read_text()) if p.exists() else {}

    def im(self, name, crop=None):
        p = self.path(name)
        x = Image.open(p).convert("RGB")
        if crop:
            w, h = x.size
            x = x.crop((int(crop[0] * w), int(crop[1] * h), int(crop[2] * w), int(crop[3] * h)))
        rec = self.used.setdefault(name, dict(file=str(p.relative_to(REPO)), sha256=sha(p), crops=[]))
        if p.with_suffix(".json").exists():
            rec["sidecar"] = str(p.with_suffix(".json").relative_to(REPO))
            m = self.meta(name)
            for k in ("frame", "run", "note", "episode", "control_step", "arm"):
                if k in m:
                    rec[k] = m[k]
        if crop and list(crop) not in rec["crops"]:
            rec["crops"].append(list(crop))
        return np.asarray(x)


def fit(a, asp, anchor=(0.5, 0.5)):
    """Centre-crop ``a`` to aspect ``asp`` (w/h); anchor = where the kept window sits in the removable range."""
    h, w = a.shape[:2]
    if w / h > asp:
        nw = int(h * asp); x0 = int((w - nw) * anchor[0]); return a[:, x0:x0 + nw], (x0, 0, nw, h)
    nh = int(w / asp); y0 = int((h - nh) * anchor[1]); return a[y0:y0 + nh], (0, y0, w, nh)


# ------------------------------------------------------------------------- raster label replacement (>= 7 pt)
_FONT = None


def _font():
    global _FONT
    if _FONT is None:
        p = pathlib.Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"
        _FONT = ImageFont.truetype(str(p), 34)       # same face and size as ifg_overview_render.label()
    return _FONT


def _render_label(text):
    """Interior of the label box as ifg_overview_render.label() draws it: text anchored "mm", rectangle padded
    8 x 6 px with a 2 px outline, so the white interior is the text bbox + 12 x 8 px."""
    f = _font()
    im = Image.new("L", (900, 140), 255)
    dr = ImageDraw.Draw(im)
    dr.text((450, 70), text, font=f, fill=20, anchor="mm")
    x0, y0, x1, y1 = dr.textbbox((450, 70), text, font=f, anchor="mm")
    return np.asarray(im.crop((x0 - 6, y0 - 4, x1 + 6, y1 + 4)))


def find_labels(a, candidates):
    """Locate the white label boxes drawn by ifg_overview_render.label() and name them by template match.

    Candidate strings are rendered with the same face/size; a box is accepted when its size matches a template
    within 8 px and the column ink profiles correlate (robust to sub-pixel offsets).
    Returns a list of (x0, y0, x1, y1, text) in pixel coordinates of ``a``."""
    from scipy import ndimage
    white = (a[..., 0] >= 248) & (a[..., 1] >= 248) & (a[..., 2] >= 248)
    lab, n = ndimage.label(white)
    out = []
    temps = {t: _render_label(t) for t in dict.fromkeys(candidates)}

    def prof(g):
        v = (255.0 - g).sum(0); v = v - v.mean(); nrm = np.linalg.norm(v)
        return v / nrm if nrm > 0 else v

    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if not (90 <= w <= 600 and 28 <= h <= 70):
            continue
        if (lab[sl] == i).mean() < 0.55:
            continue
        gray = a[sl].mean(-1)
        best = None
        for t, tp in temps.items():
            if abs(tp.shape[1] - w) > 5 or abs(tp.shape[0] - h) > 5:
                continue
            tp2 = np.asarray(Image.fromarray(tp).resize((w, h)), float)
            c = float(np.dot(prof(tp2), prof(gray)))
            if best is None or c > best[0]:
                best = (c, t)
        if best is not None and best[0] > 0.7:
            out.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop, best[1]))
    return out


def inpaint_boxes(a, boxes, pad=4):
    """Fill each label box by blending horizontally between the pixels just left and right of it (vector labels go on top)."""
    a = a.copy().astype(float)
    H, W = a.shape[:2]
    for x0, y0, x1, y1, _ in boxes:
        X0, Y0, X1, Y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, W), min(y1 + pad, H)
        left = a[Y0:Y1, max(X0 - 4, 0):X0].mean(1)        # (rows, 3)
        right = a[Y0:Y1, X1:min(X1 + 4, W)].mean(1)
        t = np.linspace(0, 1, X1 - X0)[None, :, None]
        a[Y0:Y1, X0:X1] = left[:, None, :] * (1 - t) + right[:, None, :] * t
    return a.astype(np.uint8)


# ------------------------------------------------------------------------------------------------ figure helpers
class Fig:
    """Inch-based drawing on a figure of exactly W x H inches."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.fig = plt.figure(figsize=(W, H))
        self.A = Assets()
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

    def note(self, text, meaning, key):
        """Register a printed number with its source."""
        n = NUMBERS[key]
        self.printed.append(dict(text=text, meaning=meaning, value=n["value"], source=str(n["source"]), key=n["key"]))
        return text

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

    def save(self, stem, extra):
        mn, mx, cnt = self.audit()
        assert mn >= FS - 1e-6, f"{stem}: font below {FS} pt: {mn}"
        viol = self.layout_audit()
        n_viol = sum(len(v) for v in viol.values())
        print(f"[{stem}] layout audit: " + ", ".join(f"{k}={len(v)}" for k, v in viol.items()))
        for k, v in viol.items():
            for item in v[:40]:
                print("   ", k, item)
        if STRICT_LAYOUT:
            assert n_viol == 0, f"{stem}: {n_viol} layout violations"
        pdf = OUT / f"{stem}.pdf"
        self.fig.savefig(pdf, dpi=PDF_DPI)
        self.fig.savefig(pdf.with_suffix(".png"), dpi=PNG_DPI)
        plt.close(self.fig)
        sources = {str(p): sha(p) for p in {pathlib.Path(x["source"]) for x in self.printed}}
        side = dict(
            artifact=stem, pdf=str(pdf.relative_to(REPO)), png=str(pdf.with_suffix(".png").relative_to(REPO)),
            size_in=[self.W, self.H], aspect=round(self.W / self.H, 3), textwidth_in=TEXTWIDTH_IN,
            raster_dpi_in_pdf=PDF_DPI, png_dpi=PNG_DPI,
            fonts=dict(family="DejaVu Sans", math="stix", pdf_fonttype=42, min_pt=mn, max_pt=mx, n_text_items=cnt, floor_pt=FS),
            layout_audit=dict(tolerance_pt=1.0, min_text_to_panel_border_in=0.05, violations={k: len(v) for k, v in viol.items()},
                              n_segments_checked=len(self.segments), n_panels=len(self.panels), n_texts=cnt),
            script="paper/tools/plotting/cvpr_v1_compose.py", generated=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            git_head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
            printed_numbers=self.printed, number_sources=sources, assets=self.A.used,
            registration_status=("numbers copied from raw summary.json / verify.json / agg.json as listed; they are not yet "
                                 "registered in paper/evidence/RESULTS.md (planned EXP003 held-out locked, EXP004 closed-loop hurdle)"),
        )
        side.update(extra)
        pdf.with_suffix(".json").write_text(json.dumps(side, indent=1, default=str) + "\n")
        print(f"wrote {pdf.relative_to(REPO)}  {self.W:.3f} x {self.H:.3f} in  aspect {self.W / self.H:.2f}  min font {mn} pt")


def fmt(v, nd=3):
    return f"{v:.{nd}f}"


# ================================================================================================= Figure 1
def fig01(cl=True):
    """cl=False: no closed-loop rows in the measured box (fig01_teaser_noCL)."""
    W, H = TEXTWIDTH_IN, (2.88 if cl else 2.70)
    f = Fig(W, H)
    A = f.A
    # ---------------- header
    x0, pw, gap, ph = 0.20, 1.90, 0.08, (0.85 if cl else 0.80)
    y2 = 0.80 if cl else 0.73
    y1 = y2 + ph + 0.08
    f.text(0.20, y1 + ph + 0.225, "“Put the apple in the basket.”", size=7.5, style="italic")
    for k, t in enumerate(("Pick beside the bar", "Carry over the bar")):
        f.text(x0 + k * (pw + gap) + pw / 2, y1 + ph + 0.095, t, size=7.5, fontweight="bold", ha="center")
    # ---------------- row bands and labels
    f.band(0.0, y1 - 0.03, x0 + 2 * pw + gap + 0.03, ph + 0.06, "#FDECEA", "#E8A9A3")
    f.band(0.0, y2 - 0.03, x0 + 2 * pw + gap + 0.03, ph + 0.06, "#FFF7DD", C["gold"])
    f.text(0.095, y1 + ph / 2, "VLA only", size=7.5, fontweight="bold", color=C["red"], rotation=90, ha="center")
    f.text(0.095, y2 + ph / 2, "VLA + AG3S", size=7.5, fontweight="bold", color=C["darkgold"], rotation=90, ha="center")
    # ---------------- scenes (anchor = vertical window inside the 4:3 render)
    f.panel((x0, y1, pw, ph), A.im("vla_pick"), border=C["red"], lw=1.0, anchor=(0.5, 0.42),
            badge="grasps the apple", bcol="#6D6D6D")
    f.panel((x0 + pw + gap, y1, pw, ph), A.im("vla_carry_fail"), border=C["red"], lw=1.0, anchor=(0.5, 0.33),
            badge="✗ hand hits the bar → stopped", bcol=C["red"])
    f.panel((x0, y2, pw, ph), A.im("ag3s_pick"), border=C["gold"], lw=1.2, anchor=(0.5, 0.42),
            badge="grasps the apple", bcol=C["green"])
    f.panel((x0 + pw + gap, y2, pw, ph), A.im("ag3s_carry"), border=C["gold"], lw=1.2, anchor=(0.5, 0.33),
            badge="✓ clears the bar and places it", bcol=C["green"])
    # ---------------- measured box
    bx, by, bw = 4.18, y2 - 0.03, W - 4.18 - 0.01
    bt = H - 0.025
    bh = bt - by
    f.band(bx, by, bw, bh, "#FAFAFA", "#CCCCCC")
    f.text(bx + bw / 2, bt - 0.105, "Measured", size=8, fontweight="bold", ha="center")
    f.text(bx + bw / 2, bt - 0.235,
           f"held-out grounding · {f.note('34', 'held-out episodes', 'heldout_n_episodes')} episodes · "
           f"{f.note('2,523', 'held-out frames', 'heldout_n_frames')} frames", size=FS, ha="center", color=C["mute"])
    # bar charts
    cw, chh = 1.12, (0.50 if cl else 1.0)
    cy = bt - 0.60 - chh
    specs = (("target-surface\nIoU  ↑", (NUMBERS["heldout_iou_GN"]["value"], NUMBERS["heldout_iou_B"]["value"]),
              (fmt(NUMBERS["heldout_iou_GN"]["value"]), fmt(NUMBERS["heldout_iou_B"]["value"])), "heldout_iou_GN", "heldout_iou_B"),
             ("neighbor-surface\nleakage  ↓", (NUMBERS["heldout_nfp_GN"]["value"], NUMBERS["heldout_nfp_B"]["value"]),
              (f"{NUMBERS['heldout_nfp_GN']['value']:,.0f}", f"{NUMBERS['heldout_nfp_B']['value']:,.0f}"), "heldout_nfp_GN", "heldout_nfp_B"))
    for k, (title, vals, labels, key_b, key_o) in enumerate(specs):
        cx = bx + 0.19 + k * (cw + 0.22)
        ax = f.fig.add_axes(f.F(cx, cy, cw, chh))
        ax.bar([0, 1], vals, width=0.62, color=[C["bar_prior"], C["gold"]], zorder=2)
        top = max(vals)
        ax.set_ylim(0, top * 1.3); ax.set_xlim(-0.6, 1.6)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Coarse", "AG3S"], fontsize=FS)
        ax.tick_params(axis="x", length=0, pad=1.5); ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.text(0, vals[0] + top * 0.04, f.note(labels[0], title.replace("\n", " ") + " baseline G:N", key_b), ha="center", va="bottom", fontsize=FS)
        ax.text(1, vals[1] + top * 0.04, f.note(labels[1], title.replace("\n", " ") + " ours B", key_o), ha="center", va="bottom", fontsize=FS, fontweight="bold")
        f.text(cx + cw / 2, cy + chh + 0.17, title, size=FS, fontweight="bold", ha="center", linespacing=1.05)
        if k == 1:
            red = NUMBERS["heldout_nfp_reduction_B_pct"]["value"]
            ax.text(1, top * 0.44, f.note(f"−{red:.0f} %", "neighbour FP mass reduction, B vs G:N", "heldout_nfp_reduction_B_pct"),
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=C["darkgold"])
    if cl:
        # closed loop: 14 paired runs as squares
        f.text(bx + bw / 2, by + 0.66, f"hurdle scenes · {f.note('14', 'paired runs', 'hurdle_E0_n')} paired runs (simulation)",
               size=FS, fontweight="bold", ha="center")
        sq_y, sq_h = by + 0.28, 0.30
        ax = f.fig.add_axes(f.F(bx + 0.86, sq_y, 1.36, sq_h)); ax.set_xlim(-0.2, 14.2); ax.set_ylim(-0.45, 1.45); ax.axis("off")
        rows = ((1.0, "VLA only", NUMBERS["hurdle_E0_success"]["value"], "hurdle_E0_success", "hurdle_E0_n", C["red"]),
                (0.0, "VLA + AG3S", NUMBERS["hurdle_SURF_success_ct"]["value"], "hurdle_SURF_success_ct", "hurdle_SURF_n", C["darkgold"]))
        for yy, lab, k, keyk, keyn, col in rows:
            for i in range(14):
                ax.add_patch(Rectangle((i + 0.1, yy - 0.32), 0.8, 0.64, fc=C["gold"] if i < k else "white", ec="#777777", lw=0.5))
            ty = sq_y + sq_h * (yy + 0.45) / 1.9
            f.text(bx + 0.82, ty, lab, size=FS, ha="right", color=col, fontweight="bold")
            f.text(bx + 2.26, ty, f.note(f"{k} / {NUMBERS[keyn]['value']}", f"task success (contact rule), {lab}", keyk), size=FS, fontweight="bold")
        f.text(bx + bw / 2, by + 0.19, "task success without a forbidden contact", size=FS, ha="center", color=C["mute"])
    f.text(bx + bw / 2, by + 0.065, "frozen VLA · no extra training · no segmenter", size=FS, ha="center", style="italic", color="#333333")
    # ---------------- how strip
    hw, hh, hy, ag = 1.56, (0.56 if cl else 0.50), 0.02, 0.14
    hx0 = (W - (4 * hw + 3 * ag)) / 2
    mem = A.im("p_memory")
    cand = [f"{ {'target': 'target', 'destination': 'basket', 'other_object': 'other fruit'}[c]} {v:.2f}" for v, c in A.meta("p_memory")["scores"]]
    boxes = find_labels(mem, cand)
    mem_clean = inpaint_boxes(mem, boxes)
    labels = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[4]) for b in boxes]
    items = [("t_noun_attention", (0.0, 0.3, 1.0, 1.0), "1  Noun attention → seeds", C["gold"], None),
             ("t_surface_v2", (0.08, 0.15, 0.92, 0.95), "2  Depth → target surface", C["gold"], None),
             ("p_memory", (0.24, 0.26, 1.0, 0.62), "3  Features → same object?", C["teal"], labels),
             ("t_permission_v2", (0.08, 0.15, 0.92, 0.95), "4  Contact links only", C["ctrledge"], None)]
    for k, (name, cr, t, bc, lab) in enumerate(items):
        x = hx0 + k * (hw + ag)
        f.text(x + hw / 2, hy + hh + 0.115, t, size=FS, fontweight="bold", ha="center")
        if name == "p_memory":
            h_, w_ = mem_clean.shape[:2]
            sub = mem_clean[int(cr[1] * h_):int(cr[3] * h_), int(cr[0] * w_):int(cr[2] * w_)]
            lab2 = [(lx - int(cr[0] * w_), ly - int(cr[1] * h_), t2) for lx, ly, t2 in lab]
            f.panel((x, hy, hw, hh), sub, border=bc, lw=1.0, labels=lab2)
            A.im("p_memory", cr)
            for lx, ly, t2 in lab:
                if cr[1] * h_ <= ly <= cr[3] * h_ and cr[0] * w_ <= lx <= cr[2] * w_:
                    f.printed.append(dict(text=t2.replace("other fruit", "fruit"), meaning="centered prefix-L8 cosine to the remembered region (illustration, dev episode)",
                                          source=str(IFG_A / "p_memory.json"), key="/scores"))
        else:
            f.panel((x, hy, hw, hh), A.im(name, cr), border=bc, lw=1.0)
        if k < 3:
            f.arrow((x + hw + 0.01, hy + hh / 2), (x + hw + ag - 0.01, hy + hh / 2), color=C["gold"], lw=1.0, ms=6)
    scenes = json.loads((SC_A / "fig1_scenes_index.json").read_text())
    f.save("fig01_teaser" if cl else "fig01_teaser_noCL", dict(
        variant=("with closed-loop rows" if cl else "closed-loop rows removed (user choice); bar charts enlarged"),
        layout="2x2 hurdle scenes (left) + measured box (right) + how strip (bottom); authored at text width",
        baseline_label="Coarse = previous grounding (radius growth + clustering + ball exemption + sphere/capsule robot filter), arm G:N",
        scenes=scenes,
        scene_choice=("episode 1968, seed 19681 for both rows (paired). VLA only = T43 W E0 run (2026-10-05); VLA + AG3S = IFG-HURDLE SURF "
                      "run (2026-10-17), same episode, seed, obstacle config and profile. Both rows show two moments of ONE run each. "
                      "The VLA-only run grasps the apple (as 11/14 E0 runs do) and is stopped when the hand contacts the bar during the "
                      "carry (13/14 E0 runs end in an obstacle stop); the SURF run is success_ct with no hurdle or neighbour contact rows."),
        closed_loop_note=("E0 arm recorded 2026-10-05 (T43 W, numbers.summary_per_arm['E0/H']); SURF arm recorded 2026-10-17 "
                          "(IFG-HURDLE agg.json arms.SURF.S_contact); same 7 episodes x 2 seeds. Coarse-extent ablation C1a S_contact = "
                          f"{NUMBERS['hurdle_C1a_success_ct']['value']}/14 (not printed on the figure)."),
        held_out_arms=dict(baseline="G:N (radius + clustering style reference, rounds/R6 arm G:N)", ours="B (A + temporal identity switch)",
                           A_without_memory=dict(iou=NUMBERS["heldout_iou_A"]["value"], neighbor_fp=NUMBERS["heldout_nfp_A"]["value"],
                                                 reduction_pct=NUMBERS["heldout_nfp_reduction_A_pct"]["value"],
                                                 paired_A_minus_GN=NUMBERS["heldout_paired_A_GN"]["value"]),
                           precision=dict(GN=NUMBERS["heldout_prec_GN"]["value"], B=NUMBERS["heldout_prec_B"]["value"])),
        how_strip="IFG development episode T53 ep1842; panels 2 and 4 re-rendered (cvpr_v1_fig3_assets.py: target mesh tinted orange, neighbour blue, nodes unchanged); panel 3 raster labels replaced by vector text",
        dev_set_numbers_omitted="0.837 / 0.893 (dev) intentionally not printed",
    ))


def fig01_noCL():
    fig01(cl=False)


# ================================================================================================= Figure 2
def fig02():
    W, H = TEXTWIDTH_IN, 3.25
    f = Fig(W, H)
    A = f.A
    # ---------------- top strip
    sy, sh = H - 0.05 - 0.82, 0.82
    f.band(0.02, sy, W - 0.04, sh, "#FBFBFB", "#CCCCCC")
    f.box(0.08, sy + 0.45, 0.70, 0.24, "RGB +\ninstruction", C["grey"])
    f.box(0.08, sy + 0.08, 0.70, 0.24, "depth +\nrobot model", C["grey"])
    f.box(0.92, sy + 0.38, 0.78, 0.30, "frozen VLA", C["navy"], ec=C["navyedge"], bold=True)
    f.arrow((0.78, sy + 0.57), (0.92, sy + 0.55))
    outs = [(sy + 0.70, "actions", C["navy"], C["navyedge"]), (sy + 0.525, "noun attention", C["goldfill"], C["gold"]),
            (sy + 0.35, "internal features", C["tealfill"], C["teal"]), (sy + 0.175, "hidden state", C["state"], C["stateedge"])]
    for yy, t, fc, ec in outs:
        f.box(1.84, yy - 0.075, 0.86, 0.15, t, fc, ec=ec)
        f.arrow((1.70, sy + 0.53), (1.84, yy), color=ec)
    f.box(2.86, sy + 0.47, 1.22, 0.21, "noun seeds → identity", C["goldfill"], ec=C["gold"], bold=True)
    f.box(2.86, sy + 0.245, 1.22, 0.20, "feature memory →\nsame object? · alert", C["goldfill"], ec=C["gold"])
    f.box(2.86, sy + 0.03, 1.22, 0.19, "surface graph + robot\nmask → extent", C["field"], ec=C["navyedge"], bold=True)
    f.arrow((2.70, sy + 0.525), (2.86, sy + 0.575), color=C["gold"]); f.arrow((2.70, sy + 0.35), (2.86, sy + 0.345), color=C["teal"])
    # depth + robot model -> surface graph, routed below the read-out column
    f.line([(0.78, sy + 0.20), (1.77, sy + 0.20), (1.77, sy + 0.045), (2.80, sy + 0.045), (2.80, sy + 0.125)], "#777777")
    f.arrow((2.80, sy + 0.125), (2.86, sy + 0.125), color="#777777")
    f.box(4.22, sy + 0.41, 1.08, 0.27, "LTE: link-specific\nfield $\\delta_{i,t}$", C["ctrl"], ec=C["ctrledge"], bold=True)
    f.box(4.22, sy + 0.03, 1.08, 0.32, "HSR: phase / grasp\nstate (target →\nattached → released)", C["state"], ec=C["stateedge"])
    f.arrow((4.08, sy + 0.575), (4.22, sy + 0.58), color=C["gold"]); f.arrow((4.08, sy + 0.345), (4.22, sy + 0.52), color=C["gold"])
    f.arrow((4.08, sy + 0.125), (4.22, sy + 0.46), color=C["navyedge"])
    # hidden state -> HSR, routed along the strip bottom
    f.line([(2.70, sy + 0.175), (2.78, sy + 0.175), (2.78, sy + 0.012), (4.15, sy + 0.012), (4.15, sy + 0.19)], C["stateedge"])
    f.arrow((4.15, sy + 0.19), (4.22, sy + 0.19), color=C["stateedge"])
    f.arrow((4.76, sy + 0.35), (4.76, sy + 0.41), color=C["stateedge"])
    f.box(5.44, sy + 0.41, 0.78, 0.27, "CCR: refine +\ncheck prefix", C["ctrl"], ec=C["ctrledge"], bold=True)
    f.arrow((5.30, sy + 0.545), (5.44, sy + 0.545), color=C["ctrledge"])
    f.text(6.33, sy + 0.615, "execute", size=FS, fontweight="bold", color=C["green"])
    f.text(6.33, sy + 0.47, "hold", size=FS, fontweight="bold", color=C["hold"])
    f.arrow((6.22, sy + 0.58), (6.31, sy + 0.615), color=C["green"]); f.arrow((6.22, sy + 0.51), (6.31, sy + 0.47), color=C["hold"])
    f.text(5.44, sy + 0.29, "exception withheld on alert", size=FS, color=C["mute"], style="italic")
    f.text(5.44, sy + 0.175, "or uncertain grasp", size=FS, color=C["mute"], style="italic")
    f.line([(2.70, sy + 0.755), (5.83, sy + 0.755)], C["navyedge"], lw=1.0); f.arrow((5.83, sy + 0.755), (5.83, sy + 0.68), color=C["navyedge"])
    f.text(3.62, sy + 0.755, "nominal action chunk (policy unchanged)", size=FS, color=C["navyedge"], fontweight="bold", ha="center",
           bbox=dict(fc="#FBFBFB", ec="none", pad=0.6), zorder=6)
    # ---------------- cards: prior practice (top panel, paperbanana figure-2 concept) vs AG3S (two panels)
    cy = 0.22
    ch = sy - 0.09 - cy
    cards = [(0.03, 2.20, "1 What: identity (ATS)", "the noun's attention names the object", C["gold"], C["darkgold"], "#FFFDF5",
              "Prior: external segmenter", ("v2_segmenter_mask", (0.0, 0.52, 1.0, 1.0), (0.5, 1.0)), "extra model + class prompt",
              [("c1_ours_seed_region", (0.0, 0.0, 1.0, 0.68), (0.6, 0.3), "✓ noun seeds"), ("c3_ours_alert", (0.15, 0.2, 0.9, 0.9), (0.5, 0.5), "✓ wrong→alert")]),
             (2.29, 2.20, "2 Where it ends: extent (ATS)", "depth continuity sets the boundary", C["navyedge"], "#2F5597", "#F7FAFD",
              "Prior: volume exemption", ("f2b_ball_render", (0.0, 0.08, 1.0, 0.75), (0.5, 0.5)), "freed for every link;\nneighbor erased",
              [("c2_ours_surface_region_v2", (0.1, 0.25, 0.9, 1.0), (0.5, 0.5), "✓ kept apart"), ("c2_ours_render_mask_v2", (0.15, 0.1, 0.85, 0.9), (0.5, 0.4), "✓ fruit kept")]),
             (4.55, 2.30, "3 Who/when: permission (HSR·LTE)", "contact links only, in the allowed state", C["ctrledge"], "#5B4A7A", "#F8F6FB",
              "Prior: fixed object role", ("v2_role_prior", (0.1, 0.3, 1.0, 0.8), (0.5, 0.5)), "held object not modelled",
              [("f2c_link_render", (0.1, 0.25, 0.9, 0.85), (0.3, 0.5), "✓ contact links"), ("v2_role_attached", (0.42, 0.25, 0.75, 0.75), (0.5, 0.5), "✓ attached")])]
    for x, w, t, sub, ec, tc, fc, ptitle, (pname, pcrop, panc), pbadge, ours in cards:
        f.band(x, cy, w, ch, fc, ec, lw=0.9)
        ct = cy + ch
        f.text(x + 0.06, ct - 0.10, t, size=FS, fontweight="bold", color=tc)
        f.text(x + 0.06, ct - 0.22, sub, size=FS, style="italic", color=C["mute"])
        f.text(x + 0.06, ct - 0.34, ptitle, size=FS, fontweight="bold", color=C["red"])
        iw = w - 0.12
        f.panel((x + 0.06, ct - 0.45 - 0.68, iw, 0.68), A.im(pname, pcrop), border=C["red"], lw=0.9, anchor=panc, badge=pbadge, bcol=C["red"])
        f.text(x + 0.06, ct - 1.235, "AG3S (ours)", size=FS, fontweight="bold", color=C["green"])
        ow = (iw - 0.08) / 2
        for k, (name, crop, anc, badge) in enumerate(ours):
            f.panel((x + 0.06 + k * (ow + 0.08), ct - 1.34 - 0.62, ow, 0.62), A.im(name, crop), border=C["gold"], lw=0.9, anchor=anc,
                    badge=badge, bcol=C["green"])
    f.text(0.03, 0.155, "Replay renders of recorded frames: orange = selected target, blue = protected neighbor, gold = seeds / contact links,",
           size=FS, color="#666666")
    f.text(0.03, 0.055, "purple = attached object. Prior rows: segmenter mask = simulator label as a stand-in; volume exemption, fixed role = recorded runs.",
           size=FS, color="#666666")
    f.save("fig02_overview", dict(
        layout="top: framework strip (RGB + instruction → frozen VLA → read-outs; depth + robot model → geometric layer; LTE field → CCR); "
               "bottom: three contribution cards, prior practice (top panel) vs AG3S (two panels)",
        cards={"1_identity": ["v2_segmenter_mask", "c1_ours_seed_region", "c3_ours_alert"],
               "2_extent": ["f2b_ball_render", "c2_ours_surface_region_v2", "c2_ours_render_mask_v2"],
               "3_permission": ["v2_role_prior", "f2c_link_render", "v2_role_attached"]},
        prior_provenance=dict(
            segmenter_mask="illustration: T53 ep1842 chunk 10 right-wrist camera re-rendered from the recorded state; the simulator GT target label "
                           "(gt/ep1842.npz label_small, class 0) stands in for an ideal external segmenter mask (paperbanana figure2 v2_segmenter_mask)",
            volume_exemption="recorded run: T53 C1a ep1842 chunk 17, recorded ball (surface-fit centre, radius 60.2 mm) removes every voxel inside it "
                             "for every link, including the neighbour's near side (paperbanana figure2 f2b_ball_render)",
            fixed_role="recorded run: T43 V11AB ep1982 s19822 chunk 22 carry state; red glow marks the held object a fixed-role filter does not model "
                       "(paperbanana figure2 v2_role_prior); the AG3S 'attached' panel is the same recorded state with the attached sphere"),
        ours_assets_note="c2 panels re-rendered by cvpr_v1_fig3_assets.py (target mesh tinted orange, neighbour blue; nodes unchanged); "
                         "f2c_link_render / v2_role_attached come from the earlier serving configuration (contact-interface illustration)",
        is_measured_result=False, scope="conceptual overview on real replay renders; no numbers printed",
    ))


# ================================================================================================= Figure 3 + Figure 6
def fig03_legacy():
    W, H = TEXTWIDTH_IN, 3.3
    f = Fig(W, H)
    A = f.A
    top, bot = H - 0.40, 0.05
    g = 0.08
    cols = [(0.03, 1.26, "Frozen VLA", C["navy"], C["navyedge"]),
            (0.03 + 1.26 + g, 1.33, "Surface graph\n(ATS)", "#F2F2F2", "#888888"),
            (0.03 + 1.26 + 2 * g + 1.33, 1.33, "Seeds → region\n(ATS)", C["goldfill"], C["gold"]),
            (0.03 + 1.26 + 3 * g + 2 * 1.33, 1.33, "Identity and state\n(ATS · HSR)", C["tealfill"], C["teal"]),
            (0.03 + 1.26 + 4 * g + 3 * 1.33, 1.22, "Contact interface\n(LTE · CCR)", "#F3F0F7", C["ctrledge"])]
    for x, w, t, fc, ec in cols:
        f.band(x, bot, w, top - bot, fc, ec, lw=0.8, ls="--", z=-2, r=0.008)
        f.text(x + w / 2, top + 0.155, t, size=7.5, fontweight="bold", style="italic", ha="center", va="center", linespacing=1.0)
    R = lambda k: cols[k][0] + cols[k][1]       # right edge of column k
    # ---- column 1: frozen VLA
    x, w = cols[0][0], cols[0][1]
    tw = (w - 0.10 - 0.04) / 3
    for i in range(3):
        f.panel((x + 0.05 + i * (tw + 0.02), top - 0.35, tw, 0.29), A.im(f"p_in_{i}"), lw=0.5)
        f.panel((x + 0.05 + i * (tw + 0.02), top - 1.99, tw, 0.29), A.im(f"p_att_{i}"), lw=0.5)
    f.text(x + w / 2, top - 0.46, "RGB (3 views),", size=FS, ha="center"); f.text(x + w / 2, top - 0.57, "instruction, state", size=FS, ha="center")
    f.box(x + 0.03, top - 0.95, w - 0.06, 0.30, "frozen VLA\nserving pass", C["navy"], ec=C["navyedge"], bold=True)
    f.arrow((x + w / 2, top - 0.95), (x + w / 2, top - 1.02), color=C["navyedge"])
    f.box(x + 0.03, top - 1.62, w - 0.06, 0.60, "an instrumented copy\nexposes:\nnoun attention (L10 H7)\nprefix-L8 features\nKV-L4 sub-task probe", "#FFFFFF")
    f.arrow((x + w / 2, top - 1.62), (x + w / 2, top - 1.69), color=C["navyedge"])
    f.text(x + w / 2, top - 2.10, "noun attention", size=FS, ha="center"); f.text(x + w / 2, top - 2.21, "3 × 16 × 16", size=FS, ha="center")
    f.box(x + 0.03, bot + 0.08, w - 0.06, 0.36, "depth + camera poses\n+ robot model (FK)", C["grey"])
    # ---- column 2: surface graph
    x, w = cols[1][0], cols[1][1]
    iw, ih = w - 0.10, 0.86
    yA = top - 0.06 - ih                    # upper image row (shared by columns 2-5)
    yB = yA - 0.36 - ih                     # lower image row (columns 2-3)
    f.panel((x + 0.05, yA, iw, ih), A.im("p_robot_mask_v2", (0.2, 0.55, 0.85, 1.0)), border="#888888")
    f.text(x + w / 2, yA - 0.08, "robot mask by rendering", size=FS, ha="center", va="top")
    f.text(x + w / 2, yA - 0.19, "whole target kept (orange)", size=FS, ha="center", va="top", color=C["mute"])
    f.panel((x + 0.05, yB, iw, ih), A.im("p_surface_nodes", (0.1, 0.05, 0.9, 0.95)), border="#888888")
    f.text(x + w / 2, yB - 0.08, "surface graph, 5 mm nodes", size=FS, ha="center", va="top")
    f.text(x + w / 2, yB - 0.19, "depth-continuous edges only", size=FS, ha="center", va="top", color=C["mute"])
    f.arrow((R(0), bot + 0.26), (x, yB + 0.25), color="#777777")
    f.arrow((R(0), top - 1.84), (x, yB + ih - 0.1), color=C["gold"], lw=1.0)
    # ---- column 3: seeds → region
    x, w = cols[2][0], cols[2][1]
    f.panel((x + 0.05, yA, iw, ih), A.im("p_membership", (0.1, 0.05, 0.9, 0.95)), border=C["gold"], lw=1.0)
    f.text(x + w / 2, yA - 0.08, "seeds → membership $m$", size=FS, ha="center", va="top")
    f.text(x + w / 2, yA - 0.32, r"$(\Lambda+L)\,m=\Lambda y$", size=9, ha="center", va="center")   # manuscript Eq. (5)
    f.text(x + w / 2, yA - 0.48, "support: negative evidence", size=FS, ha="center", va="center", color=C["mute"], style="italic")
    yB3 = yA - 0.68 - ih
    f.panel((x + 0.05, yB3, iw, ih), A.im("p_region_v2", (0.1, 0.05, 0.9, 0.95)), border=C["gold"], lw=1.0)
    f.text(x + w / 2, yB3 - 0.08, "largest-seed-mass region", size=FS, ha="center", va="top")
    f.text(x + w / 2, yB3 - 0.19, "neighbor stays blue", size=FS, ha="center", va="top", color=C["mute"])
    f.arrow((R(1), yA + ih / 2), (x, yA + ih / 2), color=C["gold"], lw=1.0)
    f.arrow((R(1), yB + ih / 2), (x, yB3 + ih / 2), color=C["gold"], lw=1.0)
    f.arrow((x + w / 2, yA - 0.55), (x + w / 2, yB3 + ih + 0.01), color=C["gold"], lw=1.0)
    # ---- column 4: identity and state
    x, w = cols[3][0], cols[3][1]
    mem = A.im("p_memory")
    cand = [f"{ {'target': 'target', 'destination': 'basket', 'other_object': 'other fruit'}[c]} {v:.2f}" for v, c in A.meta("p_memory")["scores"]]
    boxes = find_labels(mem, cand)
    mem_clean = inpaint_boxes(mem, boxes)
    cr = (0.1, 0.1, 0.95, 0.95)
    h_, w_ = mem.shape[:2]
    sub = mem_clean[int(cr[1] * h_):int(cr[3] * h_), int(cr[0] * w_):int(cr[2] * w_)]
    labs = [((b[0] + b[2]) / 2 - int(cr[0] * w_), (b[1] + b[3]) / 2 - int(cr[1] * h_), b[4],
             "bottom" if b[4].startswith("target") else "top") for b in boxes]      # fruit labels go below their objects
    f.panel((x + 0.05, yA, iw, ih), sub, border=C["teal"], lw=1.0, labels=labs); A.im("p_memory", cr)
    for _, _, t, _ in labs:
        f.printed.append(dict(text=t.replace("other fruit", "fruit"), meaning="centered prefix-L8 cosine to the region remembered at approach (illustration, dev episode)",
                              source=str(IFG_A / "p_memory.json"), key="/scores"))
    f.text(x + w / 2, yA - 0.08, "cosine to memory", size=FS, ha="center", va="top")
    f.box(x + 0.05, yA - 0.66, iw, 0.42, "switch on a clear margin;\nalert when agreement is\nlow → exception withheld", C["tealfill"], ec=C["teal"])
    f.box(x + 0.05, yA - 1.40, iw, 0.64, "phase / grasp state\nsub-task probe, gripper,\nhand–target geometry\n→ target · attached\n→ released", C["state"], ec=C["stateedge"])
    f.arrow((R(2), yA + ih / 2), (x, yA + ih / 2), color=C["gold"], lw=1.0)
    f.arrow((R(2), yB3 + ih / 2), (x, yA - 1.08), color=C["gold"], lw=1.0)
    f.line([(R(0), top - 1.32), (R(0) + 0.04, top - 1.32), (R(0) + 0.04, bot + 0.02), (x + w / 2, bot + 0.02)], C["stateedge"])
    f.arrow((x + w / 2, bot + 0.02), (x + w / 2, yA - 1.40), color=C["stateedge"])
    # ---- column 5: contact interface
    x, w = cols[4][0], cols[4][1]
    iw5 = w - 0.10
    f.panel((x + 0.05, yA, iw5, ih), A.im("t_permission_v2", (0.12, 0.05, 0.88, 0.95)), border=C["ctrledge"], lw=1.0)
    f.text(x + w / 2, yA - 0.08, "contact links only", size=FS, ha="center", va="top")
    ey = yA - 1.22
    f.band(x + 0.05, ey, iw5, 1.0, "#FFFFFF", C["ctrledge"], lw=0.7, z=2, r=0.004)
    f.text(x + 0.09, ey + 0.905, r"$\delta_{i,t}(x)=d_t^{\setminus\mathcal{X}}(x)$", size=FS, va="center", zorder=6)
    f.text(x + 0.20, ey + 0.745, r"if $i\in\mathcal{L}$ and $e_t=1$", size=FS, va="center", zorder=6)
    f.text(x + 0.09, ey + 0.605, r"$\delta_{i,t}(x)=d_t(x)$  otherwise", size=FS, va="center", zorder=6)
    f.text(x + 0.09, ey + 0.455, "$e_t$: grounding valid,", size=FS, va="center", color="#333333", zorder=6)
    f.text(x + 0.09, ey + 0.33, "no alert, allowed phase", size=FS, va="center", color="#333333", zorder=6)
    f.text(x + 0.09, ey + 0.19, "$\\mathcal{X}_t$: accepted target", size=FS, va="center", color="#333333", zorder=6)
    f.text(x + 0.09, ey + 0.065, "surface only", size=FS, va="center", color="#333333", zorder=6)
    f.box(x + 0.05, ey - 0.50, iw5, 0.36, "CCR: SQP refinement\n+ clearance check", C["ctrl"], ec=C["ctrledge"])
    f.text(x + 0.22, ey - 0.64, "execute", size=FS, fontweight="bold", color=C["green"])
    f.text(x + w - 0.20, ey - 0.64, "hold", size=FS, fontweight="bold", color=C["hold"], ha="right")
    f.arrow((x + w / 2, ey - 0.50), (x + w / 2, ey - 0.57), color=C["ctrledge"])
    f.arrow((x + w / 2, ey), (x + w / 2, ey - 0.14), color=C["ctrledge"])
    f.arrow((R(3), yA - 0.45), (x, yA - 0.45), color=C["teal"])
    f.arrow((R(3), yA - 1.08), (x, yA - 1.08), color=C["stateedge"])
    # action lane
    f.line([(R(0), top - 0.80), (R(0) + 0.04, top - 0.80), (R(0) + 0.04, H - 0.06), (W - 0.015, H - 0.06), (W - 0.015, ey - 0.32)],
           C["navyedge"], lw=1.0)
    f.arrow((W - 0.015, ey - 0.32), (x + w - 0.05, ey - 0.32), color=C["navyedge"])
    f.text(3.5, H - 0.06, "nominal action chunk (policy unchanged)", size=FS, color=C["navyedge"], fontweight="bold", ha="center",
           bbox=dict(fc="white", ec="none", pad=0.6), zorder=6)
    f.save("fig03_pipeline", dict(
        layout="five stages, one development episode (T53 ep1842) for the real panels; episode strip moved to fig06_episode_strip",
        rerendered=dict(stage2_top="p_robot_mask_v2 (T53_ep1842_f025, wrist_cam_r): whole observed target orange, rendered robot mask purple (no partial overlay)",
                        stage3_bottom="p_region_v2 (T53_ep1842_f016): target mesh tinted orange, neighbour blue, nodes unchanged",
                        stage5_top="t_permission_v2 (T53_ep1842_f016): same tint, contact links gold", script="paper/tools/plotting/cvpr_v1_fig3_assets.py"),
        notation="Eq. (linkquery) of 03_method.tex: delta_{i,t}(x) = d_t^{setminus X}(x) if i in L and e_t = 1, else d_t(x) (X_t = accepted target surface); membership equation printed as the manuscript Eq. (5): (Lambda + L) m = Lambda y",
        method_constants_omitted="switch margin / alert threshold values are not printed (described in words)",
        terminology="stage titles carry the module acronyms (ATS / HSR / LTE / CCR); attention and features come from an instrumented copy of the frozen checkpoint",
        badges_removed="'not yet evaluated with this output' removed; Table 1 reports the closed loop",
        is_measured_result=False,
    ))


# ================================================================================================= Figure 3 (v2: two-row story)
def fig03_storyboard():
    """Two rows, ten real panels, one short label each. Row 1 = ATS (which object, where it ends, is it still it);
    row 2 = cuRobo scene field -> link-specific exemption -> grasp confirmed -> attached object -> refine + check."""
    W, H = TEXTWIDTH_IN, 2.66
    f = Fig(W, H)
    A = f.A
    ph, g, px0 = 1.12, 0.15, 0.31
    pw = (W - px0 - 0.03 - 4 * g) / 5
    y2 = 0.05
    y1 = y2 + ph + 0.32
    X = [px0 + k * (pw + g) for k in range(5)]
    # row bands + rotated row titles
    f.band(0.01, y1 - 0.04, W - 0.02, ph + 0.08, "#FFF8E6", C["gold"], lw=0.8, z=-2, r=0.008)
    f.band(0.01, y2 - 0.04, W - 0.02, ph + 0.08, "#F4F1F8", C["ctrledge"], lw=0.8, z=-2, r=0.008)
    f.text(0.135, y1 + ph / 2, "Ground the target\n(ATS)", size=7.5, fontweight="bold", color=C["darkgold"], rotation=90,
           ha="center", va="center", linespacing=1.05)
    f.text(0.135, y2 + ph / 2, "Constrain the motion\n(HSR · LTE · CCR)", size=7, fontweight="bold", color="#5B4A7A", rotation=90,
           ha="center", va="center", linespacing=1.05)
    bc = "#2B2B2B"

    def pan(k, row, name, crop, badge, border, anchor=(0.5, 0.5), labels=None, img=None, badge_y=0.10):
        y = y1 if row == 0 else y2
        a = img if img is not None else A.im(name, crop)
        return f.panel((X[k], y, pw, ph), a, border=border, lw=1.0, anchor=anchor, badge=badge, bcol=bc, labels=labels, badge_y=badge_y)

    # ---------------- row 1: ATS
    pan(0, 0, "t_noun_attention", (0.0, 0.12, 1.0, 1.0), "1 noun attention", C["gold"])
    f.text(X[0] + 0.10, y1 + ph - 0.16, "frozen VLA", size=FS, fontweight="bold", color=C["navyedge"], va="center",
           bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C["navyedge"], lw=0.5), zorder=7)
    pan(1, 0, "p_robot_mask_v2", (0.18, 0.52, 0.86, 1.0), "2 robot mask", "#888888")
    pan(2, 0, "p_surface_nodes", (0.12, 0.08, 0.92, 0.95), "3 surface graph", "#888888")
    pan(3, 0, "p_region_v2", (0.12, 0.05, 0.88, 0.95), "4 target surface", C["gold"])
    mem = A.im("p_memory")
    cand = [f"{ {'target': 'target', 'destination': 'basket', 'other_object': 'other fruit'}[c]} {v:.2f}" for v, c in A.meta("p_memory")["scores"]]
    boxes = find_labels(mem, cand)
    keep = [b for b in boxes if b[4].startswith("target")] + sorted([b for b in boxes if not b[4].startswith("target")], key=lambda b: -float(b[4].split()[-1]))[:1]
    mem_clean = inpaint_boxes(mem, boxes)
    cr = (0.1, 0.1, 0.95, 0.95)
    h_, w_ = mem.shape[:2]
    sub = mem_clean[int(cr[1] * h_):int(cr[3] * h_), int(cr[0] * w_):int(cr[2] * w_)]
    labs = [((b[0] + b[2]) / 2 - int(cr[0] * w_), (b[1] + b[3]) / 2 - int(cr[1] * h_), b[4], "bottom" if b[4].startswith("target") else "top") for b in keep]
    pan(4, 0, None, None, "5 same object?", C["teal"], labels=labs, img=sub, badge_y=ph - 0.20)
    A.im("p_memory", cr)
    for _, _, t, _ in labs:
        f.printed.append(dict(text=t.replace("other fruit", "fruit"), meaning="centered prefix-L8 cosine to the region remembered at approach (illustration, dev episode)",
                              source=str(IFG_A / "p_memory.json"), key="/scores"))
    for k in range(4):
        f.arrow((X[k] + pw + 0.02, y1 + ph / 2), (X[k + 1] - 0.02, y1 + ph / 2), color=C["gold"], lw=1.0, ms=6)

    # ---------------- row 2: field -> exemption -> grasp -> attached -> refine
    pan(0, 1, "f3_scene_field", (0.0, 0.0, 0.9, 1.0), "6 cuRobo ESDF", C["navyedge"], anchor=(0.5, 0.6))
    pan(1, 1, "t_permission_v2", (0.12, 0.05, 0.88, 0.95), "7 contact links", C["ctrledge"])
    # 8: task + grasp state (vector, recorded probe output of the same episode)
    sig = json.loads((IFG_A / "p_phase_signals.json").read_text())["rows"]
    ax = f.fig.add_axes(f.F(X[2], y2, pw, ph))
    f.panels.append((X[2], y2, pw, ph))
    n = len(sig)
    cols = {"pre": "#FCE9D9", "held": "#E8DDF3", "placed": "#E3EFF9"}
    names = {"pre": "target", "held": "attached", "placed": "released"}
    seg, start = [], 0
    for i in range(1, n + 1):
        if i == n or sig[i]["grasp_state"] != sig[start]["grasp_state"]:
            seg.append((start, i, sig[start]["grasp_state"])); start = i
    for j, (a0, a1, gs) in enumerate(seg):
        ax.axvspan(a0, a1, color=cols.get(gs, "#EEEEEE"), lw=0, zorder=0)
        half = (len(names.get(gs, gs)) * 0.62 * FS / 72 / 2 + 0.07) * n / pw     # half label width + 0.07 in, in data units
        xc = float(np.clip((a0 + a1) / 2, half, n - half))
        ax.text(xc, 1.38 if j % 2 == 0 else 1.17, names.get(gs, gs), ha="center", va="center", fontsize=FS, fontweight="bold",
                color={"pre": C["target"], "held": C["attached"], "placed": C["neigh"]}.get(gs, C["ink"]),
                bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.85), zorder=4)
    xs = [i + 0.5 for i in range(n)]
    ax.plot(xs, [r["hsr"][1] for r in sig], color=C["stateedge"], lw=1.2, ls="--", marker="o", ms=2.2, zorder=3)
    ax.plot(xs, [min(1.0, 3.0 * r["grip"]) * 0 + (1.0 - r["grip"]) * 0 for r in sig], alpha=0)   # keep axis span
    ax.text(n * 0.5, 0.50, "probe\np(place)", ha="center", va="center", fontsize=FS, color=C["stateedge"], linespacing=1.0,
            bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.85), zorder=4)
    ax.set_xlim(0, n); ax.set_ylim(-0.12, 1.55); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(C["stateedge"]); sp.set_linewidth(1.0)
    f.badges.append(ax.text(0.10 / pw, 0.10 / ph, "8 grasp state", transform=ax.transAxes, fontsize=FS, fontweight="bold", color="white",
                            va="baseline", bbox=dict(boxstyle="round,pad=0.25", fc=bc, ec="none"), zorder=5))
    f.printed.append(dict(text="(curve)", meaning="recorded sub-task probe p(place) and annotated grasp state at 7 key moments of T53 ep1842",
                          source=str(IFG_A / "p_phase_signals.json"), key="/rows/*/hsr[1], /rows/*/grasp_state"))
    pan(3, 1, "f3_attached_object", (0.18, 0.12, 0.88, 0.92), "9 carried object", C["attached"])
    # 10: refine + check (schematic)
    ax = f.fig.add_axes(f.F(X[4], y2, pw, ph))
    f.panels.append((X[4], y2, pw, ph))
    t = np.linspace(0, 1, 16)
    xr, yr = 0.08 + 0.84 * t, 0.10 + 0.70 * t
    bump = 0.23 * np.exp(-((t - 0.45) / 0.18) ** 2)
    xo, yo = xr - bump * 0.55, yr + bump
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((0.47, 0.42), 0.20, 0.20 * pw / (ph / 1.16), color=C["neigh"], alpha=0.85, zorder=1))
    ax.plot(xr, yr, color="#777777", ls="--", lw=1.0, zorder=2)
    ax.plot(xo, yo, color=C["gold"], lw=1.6, zorder=3)
    ax.scatter(xo[:8], yo[:8], s=6, color=C["gold"], zorder=4)
    ax.scatter(xo[8:], yo[8:], s=6, facecolor="white", edgecolor=C["gold"], lw=0.6, zorder=4)
    ax.text(0.93, 0.36, "VLA\nchunk", ha="right", va="center", fontsize=FS, color="#666666", linespacing=1.0)
    ax.text(0.70, 1.00, "execute", ha="center", va="center", fontsize=FS, fontweight="bold", color=C["green"])
    ax.text(0.70, 0.88, "or hold", ha="center", va="center", fontsize=FS, fontweight="bold", color=C["hold"])
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.14); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(C["ctrledge"]); sp.set_linewidth(1.0)
    f.badges.append(ax.text(0.10 / pw, 0.10 / ph, "10 refine, check", transform=ax.transAxes, fontsize=FS, fontweight="bold", color="white",
                            va="baseline", bbox=dict(boxstyle="round,pad=0.25", fc=bc, ec="none"), zorder=5))
    for k in range(4):
        f.arrow((X[k] + pw + 0.02, y2 + ph / 2), (X[k + 1] - 0.02, y2 + ph / 2), color=C["ctrledge"], lw=1.0, ms=6)
    # ---------------- row 1 -> row 2: accepted surface + no alert feed the link-specific field
    ym = y2 + ph + 0.16
    x4, x5, x7 = X[3] + pw / 2, X[4] + pw / 2, X[1] + pw / 2
    f.line([(x4, y1), (x4, ym)], C["gold"], lw=1.0, z=3)
    f.line([(x5, y1), (x5, ym), (x7, ym)], C["gold"], lw=1.0, z=3)
    f.arrow((x7, ym), (x7, y2 + ph + 0.005), color=C["gold"], lw=1.0, ms=6)
    f.text((x7 + x4) / 2, ym, r"accepted surface $\mathcal{X}_t$, no alert", size=FS, ha="center", va="center", color=C["darkgold"],
           fontweight="bold", bbox=dict(fc="white", ec="none", pad=0.8), zorder=6)
    f.save("fig03_storyboard", dict(
        version="v2 two-row story (2026-10-18): ten real panels, one label each; long text moved to the LaTeX caption",
        layout=("row 1 ATS: 1 noun attention (frozen VLA) -> 2 rendered robot mask -> 3 depth-continuous surface graph -> 4 target surface -> "
                "5 same-object check; row 2: 6 cuRobo TSDF->ESDF scene field -> 7 contact links may enter the target surface -> "
                "8 grasp state (target -> attached -> released, probe p(place)) -> 9 carried object as attached geometry -> 10 refine + check"),
        panels_provenance={
            "1-5": "IFG development episode T53 ep1842 (paper/figures/ifg_overview/assets, cvpr_v1/assets *_v2 re-renders)",
            "6 f3_scene_field": "recorded cuRobo main ESDF (5 mm) of T53 C1a ep1842 chunk 17, voxels with |d| <= 4 mm near the target "
                                "(paperbanana figure3); the coarse-extent serving run of the same episode, same cuRobo field construction",
            "7 t_permission_v2": "configured contact rule visualised on T53 ep1842 (gold = active-hand contact links)",
            "8": "recorded sub-task probe output and annotated grasp state at 7 key moments of T53 ep1842 (p_phase_signals.json)",
            "9 f3_attached_object": "attached sphere recorded at attach (chunk 19) of T53 C1a ep1842, carried with the hand by FK (paperbanana figure3)",
            "10": "schematic (not data): policy chunk (dashed) vs refined chunk (gold) around an obstacle; filled dots = 8 executed rows, hollow = look-ahead"},
        is_measured_result=False,
    ))


# ================================================================================================= Figure 3 (v3: architecture diagram)
def _field_slices():
    """2-D illustration of the two distance fields (exact Euclidean distance transform of a synthetic cross-section)."""
    from scipy import ndimage
    ny, nx = 100, 120
    yy, xx = np.mgrid[0:ny, 0:nx]
    table = yy < 14
    tgt = ((xx - 50) / 19.0) ** 2 + ((yy - 37) / 23.0) ** 2 <= 1.0
    nb = (xx - 83) ** 2 + (yy - 31) ** 2 <= 16.5 ** 2
    full = table | tgt | nb
    free = table | nb
    return ndimage.distance_transform_edt(~full), ndimage.distance_transform_edt(~free), tgt, nb, table


def fig03():
    """Method figure as an architecture diagram: inputs -> frozen VLA -> ATS (+HSR) -> LTE -> CCR. Each block shows only the
    representation it produces; real frames appear only for the seed patches and the selected target surface."""
    W, H = TEXTWIDTH_IN, 3.05
    f = Fig(W, H)
    A = f.A
    top = H - 0.20                     # bands start below the action lane
    bot = 0.06
    xin, win = 0.03, 0.80
    xv, wv = 0.92, 0.98
    xa, wa = 2.00, 1.98
    xl, wl = 4.08, 1.66
    xc, wc = 5.82, 1.03
    yh = 1.20                          # ATS / HSR split
    # ---------------- bands + headers
    f.band(xv, yh, wv, top - yh, C["navy"], C["navyedge"], lw=0.8, z=-2)
    f.band(xa, yh, wa, top - yh, "#FFF8E6", C["gold"], lw=0.8, z=-2)
    f.band(xa, bot, wa, yh - bot - 0.08, C["state"], C["stateedge"], lw=0.8, z=-2)
    f.band(xl, bot, wl, top - bot, "#F4F1F8", C["ctrledge"], lw=0.8, z=-2)
    f.band(xc, bot, wc, top - bot, "#F1F1F1", "#777777", lw=0.8, z=-2)
    hy = top - 0.11
    f.text(xv + 0.22, hy, "Frozen VLA", size=7.5, fontweight="bold", color=C["navyedge"])
    f.text(xa + 0.07, hy, "ATS  which object, where it ends", size=7.5, fontweight="bold", color=C["darkgold"])
    f.text(xa + 0.07, yh - 0.19, "HSR  when its role changes", size=7.5, fontweight="bold", color=C["stateedge"])
    f.text(xl + 0.07, hy, "LTE  who may touch it", size=7.5, fontweight="bold", color="#5B4A7A")
    f.text(xc + 0.07, hy, "CCR", size=7.5, fontweight="bold", color="#444444")

    # ---------------- inputs
    th = 0.40
    for i in range(3):
        f.panel((xin, top - 0.06 - (i + 1) * th - i * 0.04, win, th), A.im(f"p_in_{i}"), lw=0.5)
    yrgb = top - 0.06 - 3 * th - 0.08
    f.text(xin + win / 2, yrgb - 0.06, "RGB, 3 views", size=FS, ha="center", va="top")
    f.box(xin, 0.62, win, 0.36, "“Put the pear\nin the basket.”", "#FFFFFF", ec="#999999", size=FS)
    f.box(xin, 0.14, win, 0.36, "depth +\nrobot state", C["grey"], ec="#999999", size=FS)

    # ---------------- frozen VLA block: token strip + three taps
    vb_y0, vb_y1 = 1.56, top - 0.24
    f.band(xv + 0.06, vb_y0, wv - 0.12, vb_y1 - vb_y0, "#FFFFFF", C["navyedge"], lw=0.7, z=1, r=0.004)
    f.text(xv + wv / 2 + 0.04, vb_y1 - 0.11, r"$\pi_{0.5}$, no update", size=FS, ha="center", va="center", zorder=5)
    tok_y, ts = vb_y0 + 0.30, 0.085
    for j in range(8):                                    # image | text | action tokens
        col = "#BFBFBF" if j < 3 else (C["gold"] if j == 4 else ("#E9E2C8" if j < 6 else "#9DC3E6"))
        f.band(xv + 0.12 + j * (ts + 0.012), tok_y, ts, ts, col, "#666666", lw=0.4, z=3, r=0.001)
    f.text(xv + 0.12 + 1.5 * (ts + 0.012), tok_y - 0.09, "image", size=FS, ha="center", va="center", color="#555555")
    f.text(xv + 0.12 + 4.5 * (ts + 0.012), tok_y + 0.19, "“pear”", size=FS, ha="center", va="center", color=C["darkgold"], fontweight="bold")
    f.text(xv + 0.12 + 6.5 * (ts + 0.012), tok_y - 0.09, "action", size=FS, ha="center", va="center", color=C["navyedge"])
    f.arrow((xin + win, 2.10), (xv + 0.06, 2.10), color="#777777")
    f.line([(xin + win, 0.80), (xv - 0.04, 0.80), (xv - 0.04, 1.80)], "#777777")
    f.arrow((xv - 0.04, 1.80), (xv + 0.06, 1.80), color="#777777")
    taps = [(vb_y1 - 0.36, "noun attn.", C["gold"]), (vb_y0 + 0.52, "features", C["teal"])]
    for yy_, t_, c_ in taps:
        f.text(xv + wv - 0.10, yy_ + 0.075, t_, size=FS, ha="right", va="center", color=c_, fontweight="bold")
    # action lane
    f.line([(xv + 0.13, vb_y1), (xv + 0.13, H - 0.07), (xc + wc - 0.12, H - 0.07)], C["navyedge"], lw=1.1)
    f.arrow((xc + wc - 0.12, H - 0.07), (xc + wc - 0.12, top - 0.26), color=C["navyedge"], lw=1.1)
    f.text((xa + xl + wl) / 2, H - 0.07, r"action chunk $A_t$ (policy output, unchanged)", size=FS, ha="center", va="center",
           color=C["navyedge"], fontweight="bold", bbox=dict(fc="white", ec="none", pad=0.6), zorder=6)

    # ---------------- ATS: (a) seeds  (b) surface graph + membership  (c) result ; identity memory row
    ps, py = 0.56, top - 0.30 - 0.56
    xa1, xa2, xa3 = xa + 0.08, xa + 0.08 + ps + 0.10, xa + 0.08 + 2 * (ps + 0.10)
    f.panel((xa1, py, ps, ps), A.im("t_noun_attention", (0.18, 0.25, 0.86, 1.0)), border=C["gold"], lw=0.9)
    f.text(xa1 + ps / 2, py - 0.06, "seed patches", size=FS, ha="center", va="top")
    # (b) schematic graph
    ax = f.fig.add_axes(f.F(xa2, py, ps, ps)); f.panels.append((xa2, py, ps, ps))
    rng = np.random.default_rng(3)
    def blob(cx, cy, rx, ry, n):
        a = rng.uniform(0, 2 * np.pi, n); r = np.sqrt(rng.uniform(0, 1, n))
        return np.c_[cx + rx * r * np.cos(a), cy + ry * r * np.sin(a)]
    P = blob(0.36, 0.52, 0.17, 0.22, 26); N = blob(0.72, 0.42, 0.14, 0.14, 16)
    T = np.c_[np.linspace(0.06, 0.94, 10), np.full(10, 0.12)]
    for G, ec in ((P, "#B0B0B0"), (N, "#B0B0B0"), (T, "#B0B0B0")):
        for i in range(len(G)):
            for j in range(i + 1, len(G)):
                if np.linalg.norm(G[i] - G[j]) < 0.13:
                    ax.plot(*G[[i, j]].T, color=ec, lw=0.4, zorder=1)
    ax.plot([0.555, 0.555], [0.30, 0.66], color=C["red"], lw=0.8, ls=(0, (2, 1.5)), zorder=2)
    ax.scatter(*P.T, s=7, color=C["target"], zorder=3); ax.scatter(*N.T, s=7, color=C["neigh"], zorder=3)
    ax.scatter(*T.T, s=7, color="#8C8C8C", zorder=3)
    ax.scatter(*P[:4].T, s=20, marker="*", color=C["gold"], edgecolor="#6B5200", lw=0.3, zorder=4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#888888"); sp.set_linewidth(0.7)
    f.text(xa2 + ps / 2, py - 0.06, r"$(\Lambda+L)\,m=\Lambda y$", size=FS, ha="center", va="top")
    f.panel((xa3, py, ps, ps), A.im("p_region_v2", (0.25, 0.25, 0.80, 0.85)), border=C["gold"], lw=0.9)
    f.text(xa3 + ps / 2, py - 0.06, r"surface $\mathcal{T}_t$", size=FS, ha="center", va="top")
    f.arrow((xa1 + ps + 0.015, py + ps / 2), (xa2 - 0.015, py + ps / 2), color=C["gold"])
    f.arrow((xa2 + ps + 0.015, py + ps / 2), (xa3 - 0.015, py + ps / 2), color=C["gold"])
    f.arrow((xv + wv - 0.06, vb_y1 - 0.36 + 0.075), (xa1, py + ps * 0.75), color=C["gold"])
    # identity memory: raw vs centered cosine (frame of the R10 study)
    my = yh + 0.06
    ax = f.fig.add_axes(f.F(xa + 0.55, my + 0.04, 0.78, 0.36))
    vals = [(0.980, 0.975), (0.595, 0.413)]
    for k, (vt, vo) in enumerate(vals):
        ax.bar([k * 1.0 - 0.17], [vt], width=0.30, color=C["target"]); ax.bar([k * 1.0 + 0.17], [vo], width=0.30, color="#9E9E9E")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["raw", "centered"], fontsize=FS); ax.tick_params(axis="x", length=0, pad=1)
    ax.set_yticks([]); ax.set_ylim(0, 1.15); ax.set_xlim(-0.55, 1.55)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    f.text(xa + 0.07, my + 0.29, "same", size=FS, color=C["teal"], fontweight="bold")
    f.text(xa + 0.07, my + 0.17, "object?", size=FS, color=C["teal"], fontweight="bold")
    f.text(xa + 1.40, my + 0.32, "target", size=FS, color=C["target"], fontweight="bold")
    f.text(xa + 1.40, my + 0.20, "other", size=FS, color="#777777", fontweight="bold")
    f.text(xa + 1.40, my + 0.08, "low → alert", size=FS, color=C["red"])
    f.arrow((xv + wv - 0.06, vb_y0 + 0.595), (xa + 0.06, my + 0.23), color=C["teal"])
    for v in (0.980, 0.975, 0.595, 0.413):
        f.printed.append(dict(text="(bar)", meaning="cosine to the remembered region, raw vs centered (R10 example frame: target vs strongest alternative)",
                              value=v, source="paper/figures/ifg_results/exp/exp-R10-centering.json", key="caption values of the R10 example"))

    # ---------------- HSR: phase and grasp state chips
    def chips(y, items, cols, label):
        f.text(xa + 0.07, y, label, size=FS, color="#555555", va="center")
        x = xa + 0.55
        for k, (t, c) in enumerate(zip(items, cols)):
            w = 0.36
            f.box(x, y - 0.08, w, 0.16, t, c, ec="#777777", size=FS)
            if k < len(items) - 1:
                f.arrow((x + w + 0.01, y), (x + w + 0.08, y), color="#777777", ms=4)
            x += w + 0.09
    chips(0.74, ["pick", "place", "home"], ["#FCE9D9", "#FCE9D9", "#FCE9D9"], "phase")
    chips(0.44, ["free", "held", "free"], ["#FFFFFF", "#E8DDF3", "#FFFFFF"], "grasp")
    f.text(xa + 0.55, 0.20, "probe + gripper + geometry", size=FS, color="#555555", va="center")
    f.line([(xv + wv / 2, vb_y0), (xv + wv / 2, 0.60), (xa - 0.02, 0.60)], C["stateedge"])
    f.arrow((xa - 0.02, 0.60), (xa + 0.04, 0.60), color=C["stateedge"])
    f.text(xv + wv / 2 + 0.04, 1.00, "KV", size=FS, color=C["stateedge"], fontweight="bold")

    # ---------------- LTE: scene field -> two fields -> attached object
    f.box(xl + 0.10, top - 0.50, wl - 0.20, 0.24, "cuRobo TSDF → ESDF", "#FFFFFF", ec=C["navyedge"], size=FS, bold=True)
    f.line([(xin + win / 2, 0.14), (xin + win / 2, 0.025), (xl - 0.04, 0.025), (xl - 0.04, top - 0.38)], "#777777")
    f.arrow((xl - 0.04, top - 0.38), (xl + 0.10, top - 0.38), color="#777777")
    d_full, d_free, tgt, nb, table = _field_slices()
    fs_, fy = 0.68, top - 0.62 - 0.68
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("fld", ["#2E75B6", "#BDD7EE", "#FFFFFF"])
    for k, (d, lab, query) in enumerate(((d_full, "other links", "wrist"), (d_free, "contact links", "finger"))):
        x0 = xl + 0.08 + k * (fs_ + 0.14)
        ax = f.fig.add_axes(f.F(x0, fy, fs_, fs_)); f.panels.append((x0, fy, fs_, fs_))
        ax.imshow(np.minimum(d, 30), origin="lower", cmap=cmap, vmin=0, vmax=30, interpolation="bilinear")
        ax.contour(d, levels=[7], colors=[C["neigh"]], linewidths=0.6, linestyles="--")
        ax.contourf(table.astype(float), levels=[0.5, 1.5], colors=["#C69C6D"])
        ax.contourf(nb.astype(float), levels=[0.5, 1.5], colors=[C["neigh"]])
        if k == 0:
            ax.contourf(tgt.astype(float), levels=[0.5, 1.5], colors=[C["target"]])
            ax.add_patch(plt.Circle((22, 70), 7, fc="#9E9E9E", ec="#444444", lw=0.5))
        else:
            ax.contour(tgt.astype(float), levels=[0.5], colors=[C["target"]], linewidths=0.8, linestyles=":")
            ax.add_patch(plt.Circle((36, 58), 7, fc=C["gold"], ec="#6B5200", lw=0.5))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(C["ctrledge"]); sp.set_linewidth(0.8)
        f.text(x0 + fs_ / 2, fy - 0.06, (r"$d_t$" if k == 0 else r"$d_t^{\setminus\mathcal{X}}$"), size=FS, ha="center", va="top")
        f.text(x0 + fs_ / 2, fy - 0.24, lab, size=FS, ha="center", va="top")
    f.arrow((xl + wl / 2, top - 0.50), (xl + wl / 2, fy + fs_ + 0.01), color=C["navyedge"])
    f.text(xl + 0.08, fy - 0.46, r"$\delta_i=d_t^{\setminus\mathcal{X}}$ iff $i\in\mathcal{L}\wedge e_t$", size=FS, va="center")
    # attached after grasp
    f.panel((xl + 0.08, bot + 0.06, 0.62, 0.50), A.im("f3_attached_object", (0.30, 0.20, 0.82, 0.78)), border=C["attached"], lw=0.9)
    f.text(xl + 0.78, bot + 0.40, "after grasp:", size=FS, color=C["attached"], fontweight="bold")
    f.text(xl + 0.78, bot + 0.27, "object becomes", size=FS, color=C["attached"])
    f.text(xl + 0.78, bot + 0.14, "attached spheres", size=FS, color=C["attached"])
    f.arrow((xa3 + ps + 0.02, py + ps / 2), (xl + 0.08 + fs_ + 0.14 + fs_ / 2, fy + fs_ + 0.01), color=C["gold"])
    f.arrow((xa + wa - 0.02, my + 0.20), (xl + 0.06, fy + 0.20), color=C["teal"])
    f.arrow((xa + wa - 0.02, 0.44), (xl + 0.06, bot + 0.31), color=C["stateedge"])

    # ---------------- CCR: refine chunk, check prefix
    ax = f.fig.add_axes(f.F(xc + 0.06, top - 1.30, wc - 0.12, 0.98)); f.panels.append((xc + 0.06, top - 1.30, wc - 0.12, 0.98))
    t = np.linspace(0, 1, 16)
    xr, yr = 0.08 + 0.84 * t, 0.10 + 0.72 * t
    bump = 0.24 * np.exp(-((t - 0.45) / 0.17) ** 2)
    xo, yo = xr - bump * 0.55, yr + bump
    from matplotlib.patches import Ellipse
    pw_, ph_ = wc - 0.12, 0.98
    ax.add_patch(Ellipse((0.47, 0.43), 0.20, 0.20 * pw_ / ph_, color=C["neigh"], alpha=0.85, zorder=1))
    ax.plot(xr, yr, color="#777777", ls="--", lw=1.0, zorder=2)
    ax.plot(xo, yo, color=C["gold"], lw=1.6, zorder=3)
    ax.scatter(xo[:8], yo[:8], s=6, color=C["gold"], zorder=4)
    ax.scatter(xo[8:], yo[8:], s=6, facecolor="white", edgecolor=C["gold"], lw=0.6, zorder=4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#777777"); sp.set_linewidth(0.7)
    f.text(xc + 0.10, top - 1.43, r"refine $A_t$", size=FS, va="center")
    f.text(xc + 0.10, top - 1.575, r"s.t. $h_{ik}\geq 0$", size=FS, va="center")
    f.text(xc + 0.10, top - 1.72, "run first $K$", size=FS, va="center")
    f.box(xc + 0.10, 0.62, wc - 0.20, 0.26, "check prefix", "#FFFFFF", ec="#777777", size=FS, bold=True)
    f.arrow((xc + wc / 2, top - 1.79), (xc + wc / 2, 0.88), color="#777777")
    f.text(xc + 0.14, 0.38, "execute", size=FS, fontweight="bold", color=C["green"])
    f.text(xc + wc - 0.14, 0.38, "hold", size=FS, fontweight="bold", color=C["hold"], ha="right")
    f.arrow((xc + wc / 2 - 0.08, 0.62), (xc + 0.30, 0.46), color=C["green"]); f.arrow((xc + wc / 2 + 0.08, 0.62), (xc + wc - 0.30, 0.46), color=C["hold"])
    f.arrow((xl + wl - 0.02, fy + fs_ / 2), (xc + 0.06, fy + fs_ / 2), color=C["ctrledge"], lw=1.0)
    f.save("fig03_pipeline", dict(
        version="v3 architecture diagram (2026-10-18): inputs -> frozen VLA -> ATS (+HSR) -> LTE -> CCR; replaces the ten-frame storyboard (kept as fig03_storyboard)",
        real_frames=dict(inputs="p_in_0..2 (T53 ep1842 policy cameras)", seeds="t_noun_attention (noun attention with seed patches, T53 ep1842)",
                         target_surface="p_region_v2 (selected target surface, T53 ep1842)",
                         attached="f3_attached_object (recorded attached sphere, T53 C1a ep1842; same attachment as ours)"),
        schematic=dict(surface_graph="illustration of depth-continuous edges, seeds (stars), support (grey) and the depth break (red dashed)",
                       fields="exact 2-D Euclidean distance transforms of a synthetic cross-section: d_t with target, d_t^{\\X} without; dashed = clearance margin",
                       ccr="illustration of reference vs refined chunk; filled = executed rows"),
        identity_bars="raw vs centered cosine of the R10 example frame (target 0.980 / 0.595, strongest alternative 0.975 / 0.413), as printed in the manuscript",
        is_measured_result=False,
    ))


def fig06():
    W, H = TEXTWIDTH_IN, 1.8
    f = Fig(W, H)
    A = f.A
    ep = json.loads((IFG_A / "tp_episode.json").read_text())
    sig = json.loads((IFG_A / "p_phase_signals.json").read_text())
    moms = ["M1_approach", "M2_pre_grasp", "M3_grasp", "M4_carry", "M5_pre_place", "M6_post_place", "M7_retreat"]
    f.text(0.03, H - 0.10, "One episode, moment by moment", size=7.5, fontweight="bold")
    f.text(2.08, H - 0.10, "(development episode T53 ep1842)", size=FS, color=C["mute"])
    sw, sh, sg, sx0, sy = 0.56, 0.52, 0.03, 0.03, H - 0.22 - 0.52
    for i, mm in enumerate(moms):
        x = sx0 + i * (sw + sg)
        f.panel((x, sy, sw, sh), A.im(f"p_strip_{mm}", (0.1, 0.0, 0.9, 1.0)), border=C["gold"], lw=0.8)
        iou = ep["moments"][mm]["iou_recomputed"]
        lab = mm.split("_", 1)[1].replace("_", " ")
        f.text(x + sw / 2, sy - 0.075, lab, size=FS, ha="center")
        f.text(x + sw / 2, sy - 0.185, f"IoU {iou:.2f}", size=FS, ha="center", color=C["target"], fontweight="bold")
        f.printed.append(dict(text=f"{iou:.2f}", meaning=f"surface IoU at {mm} (recomputed offline, dev episode)",
                              source=str(IFG_A / "tp_episode.json"), key=f"/moments/{mm}/iou_recomputed"))
    f.text(sx0, 0.30, "Top: observed target surface (orange) and protected neighbors (blue)", size=FS, color="#444444")
    f.text(sx0, 0.19, "at the 7 recorded key moments; numbers = surface IoU. Right: inputs of", size=FS, color="#444444")
    f.text(sx0, 0.08, "the phase / grasp model; background = annotated grasp state.", size=FS, color="#444444")
    rows = sorted([r for r in sig["rows"]], key=lambda r: r["moment"])
    xx = np.arange(len(rows))
    ax = f.fig.add_axes(f.F(4.45, 0.47, 2.38, 0.86))
    bands = {"pre": "#DEEBF7", "held": "#E4D3F0", "placed": "#E2F0D9"}
    for xi, r in zip(xx, rows):
        ax.axvspan(xi - 0.5, xi + 0.5, color=bands.get(r["grasp_state"], "white"), lw=0)
    ax.plot(xx, [r["hand_d"] for r in rows], "o-", color=C["neigh"], lw=1.0, ms=2.5, label="hand–target dist. (m)")
    ax.plot(xx, [r["height"] - 0.8 for r in rows], "s-", color="#548235", lw=1.0, ms=2.5, label="target height − 0.8 m")
    ax.plot(xx, [1 - r["grip"] for r in rows], "^-", color=C["gold"], lw=1.0, ms=2.5, label="gripper closing")
    ax.plot(xx, [r["hsr"][1] for r in rows], "d--", color=C["stateedge"], lw=1.0, ms=2.5, label="probe p(place)")
    ax.set_xticks(xx); ax.set_xticklabels([r["moment"].split("_", 1)[1].replace("_", "\n") for r in rows], fontsize=FS)
    ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["0", "0.5", "1"], fontsize=FS)
    ax.tick_params(length=2, pad=1.5)
    ax.set_ylim(-0.05, 1.05); ax.set_xlim(-0.5, len(rows) - 0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=FS, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.1, columnspacing=0.7,
              handletextpad=0.35, borderaxespad=0.1)
    for xi, r in zip(xx, rows):
        pass
    f.save("fig06_episode_strip", dict(
        layout="7 key-moment close-ups (left) + vector re-plot of the phase / grasp inputs from p_phase_signals.json (right)",
        episode=ep["episode"], instruction="Put the pear in the basket.",
        phase_rows=[{k: r[k] for k in ("frame_id", "moment", "grasp_state", "hand_d", "height", "grip", "hsr")} for r in rows],
        phase_source=str((IFG_A / "p_phase_signals.json").relative_to(REPO)),
        is_measured_result=False, scope="one development episode; qualitative",
    ))


# ================================================================================================= Figure 4 (held-out)
HELD_RUN = HELD.parent          # heldout_locked/
ARM_ROWS = {"G:N": ("R6", "G:N"), "A-fk": ("R6", "top1-only-ns:N"), "A": ("R10b", "R6"), "B": ("R10b", "Mem-first:prefix_L8:d0.05")}
ARM_LABEL = {"G:N": "Coarse", "A-fk": "Surface (vol. mask)", "A": "Surface", "B": "Surface + ID (ours)"}
ARM_COLOR = {"G:N": C["bar_prior"], "A-fk": "#9DB4CC", "A": "#F0D27A", "B": C["gold"]}


def heldout_per_episode():
    """Per-episode IoU means of the locked arms, recomputed from the per-frame rows exactly as summary.json aggregates them
    (main = success episodes, valid frames = the G:N row's ``valid`` flag, frame -> episode mean). Asserted against summary.json."""
    S = json.loads(HELD.read_text())
    frames = json.loads((HELD_RUN / "frames.json").read_text())["frames"]
    per = {a: {} for a in ARM_ROWS}
    n_valid = 0
    for fr in frames:
        if fr["kind"] != "success":
            continue
        fid, eid = fr["frame_id"], fr["episode_id"]
        r6 = {r["arm_id"]: r for r in map(json.loads, (HELD_RUN / "run/rounds/R6/rows" / f"{fid}.jsonl").read_text().splitlines()) if r}
        if r6["G:N"].get("valid") is not True:
            continue
        r10 = {r["arm_id"]: r for r in map(json.loads, (HELD_RUN / "run/rounds/R10b/rows" / f"{fid}.jsonl").read_text().splitlines()) if r}
        n_valid += 1
        for a, (rnd, arm_id) in ARM_ROWS.items():
            per[a].setdefault(eid, []).append((r6 if rnd == "R6" else r10)[arm_id]["iou"])
    ep = {a: {e: float(np.mean(v)) for e, v in per[a].items()} for a in ARM_ROWS}
    for a in ARM_ROWS:
        ref = S["main_success"][a]["iou"]["episode_mean"]
        assert len(ep[a]) == 34 and abs(np.mean(list(ep[a].values())) - ref) < 1e-9, (a, np.mean(list(ep[a].values())), ref)
    assert n_valid == S["main_success"]["B"]["n_valid_frames"]
    return ep, S


def fig04():
    W, H = 3.25, 2.3
    f = Fig(W, H)
    ep, S = heldout_per_episode()
    eps = sorted(ep["G:N"], key=lambda e: ep["G:N"][e])
    base = np.array([ep["G:N"][e] for e in eps]); ours = np.array([ep["B"][e] for e in eps])
    pr = S["main_success"]["paired_iou"]["B - G:N"]
    assert int((ours > base).sum()) == pr["better"] and int((ours < base).sum()) == pr["worse"]
    # ---- left: paired per-episode IoU
    ax = f.fig.add_axes(f.F(0.36, 0.44, 1.32, 1.20))
    xx = np.arange(len(eps))
    for i in range(len(eps)):
        ax.plot([i, i], [base[i], ours[i]], color=(C["green"] if ours[i] > base[i] else C["hold"]), lw=0.7, alpha=0.85, zorder=2)
    ax.scatter(xx, base, s=7, color=C["bar_prior"], zorder=3, label=ARM_LABEL["G:N"], linewidths=0)
    ax.scatter(xx, ours, s=7, color=C["gold"], zorder=4, label=ARM_LABEL["B"], linewidths=0)
    ax.text(0.97, 0.05, f"{pr['better']} improved · {pr['worse']} worse", transform=ax.transAxes, fontsize=FS, ha="right", va="bottom", color=C["ink"])
    for k_, v_ in (("better", pr["better"]), ("worse", pr["worse"])):
        f.printed.append(dict(text=str(v_), meaning=f"episodes where B {k_} than G:N (paired)", value=v_, source=str(HELD), key=f"/main_success/paired_iou/B - G:N/{k_}"))
    ax.set_xlim(-1, len(eps)); ax.set_ylim(0.3, 1.0)
    ax.set_xticks([]); ax.set_yticks([0.4, 0.6, 0.8, 1.0]); ax.set_yticklabels(["0.4", "0.6", "0.8", "1.0"], fontsize=FS)
    ax.tick_params(length=2, pad=1.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("held-out episodes,\nsorted by baseline IoU", fontsize=FS, labelpad=2, linespacing=1.05)
    ax.set_ylabel("IoU (episode mean)", fontsize=FS, labelpad=2)
    f.text(0.36 + 0.66, H - 0.38, "paired per-episode IoU", size=FS, fontweight="bold", ha="center")
    f.text(0.36 + 0.66, H - 0.50, "green = improved, red = worse", size=FS, ha="center", color=C["mute"])
    # ---- right: grouped bars by grasp state
    states = ["pre", "held", "placed"]
    arms = ["G:N", "A-fk", "A", "B"]
    vals = {a: [S["by_grasp_state"][st][a]["iou"]["episode_mean"] for st in states] for a in arms}
    ax2 = f.fig.add_axes(f.F(1.98, 0.44, 1.22, 1.20))
    bw = 0.19
    for k, a in enumerate(arms):
        ax2.bar(np.arange(3) + (k - 1.5) * bw, vals[a], bw, color=ARM_COLOR[a], label=ARM_LABEL[a], zorder=2, linewidth=0)
    ax2.set_xticks(np.arange(3)); ax2.set_xticklabels(["before\ngrasp", "in hand", "placed"], fontsize=FS)
    ax2.set_ylim(0, 1.0); ax2.set_yticks([0, 0.5, 1.0]); ax2.set_yticklabels(["0", "0.5", "1"], fontsize=FS)
    ax2.tick_params(length=2, pad=1.5)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    hs, ls = ax2.get_legend_handles_labels()
    f.fig.legend(hs, ls, frameon=False, fontsize=FS, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.0, handletextpad=0.4,
                 columnspacing=1.2, borderaxespad=0.15, labelspacing=0.25)
    f.text(1.98 + 0.61, H - 0.38, "IoU by grasp state", size=FS, fontweight="bold", ha="center")
    f.text(1.98 + 0.61, H - 0.50, "held-out episode means", size=FS, ha="center", color=C["mute"])
    for a in arms:
        for st, v in zip(states, vals[a]):
            f.printed.append(dict(text=f"bar {ARM_LABEL[a]} / {st}", meaning="IoU episode mean by grasp state (bar height, no label printed)", value=v,
                                  source=str(HELD), key=f"/by_grasp_state/{st}/{a}/iou/episode_mean"))
    f.printed.append(dict(text="(dots)", meaning="per-episode IoU means, G:N and B, 34 success episodes", value=None, source=str(HELD),
                          key="recomputed from heldout_locked/frames.json + run/rounds/R6/rows/*.jsonl (arm G:N, valid flag) + run/rounds/R10b/rows/*.jsonl (arm Mem-first:prefix_L8:d0.05); asserted equal to /main_success/*/iou/episode_mean"))
    f.save("fig04_heldout", dict(
        layout="single column: left paired per-episode IoU (G:N vs B, connectors), right grouped bars of IoU by grasp state for G:N / A-fk / A / B",
        per_episode=dict(episodes=eps, GN=base.tolist(), B=ours.tolist(), A=[ep["A"][e] for e in eps], A_fk=[ep["A-fk"][e] for e in eps],
                         source_rows=[str(HELD_RUN / "frames.json"), str(HELD_RUN / "run/rounds/R6/rows"), str(HELD_RUN / "run/rounds/R10b/rows")],
                         validity="G:N row 'valid' flag (R10b rows carry no flag); 2523 valid frames of 34 success episodes",
                         paired_B_minus_GN=pr, paired_A_minus_GN=S["main_success"]["paired_iou"]["A - G:N"]),
        by_grasp_state={a: dict(zip(states, vals[a])) for a in arms}, arm_labels=ARM_LABEL,
        note=("the connectors show B vs G:N: 32 improved / 2 worse (summary.json paired_iou); the 34/34 statement holds for A vs G:N "
              "(A - G:N better 34, worse 0), not for B"),
        is_measured_result=True,
    ))


# ================================================================================================= Figure 7 (failure)
T58V = REPO / "AG3S/ag3s/docs/handoff/T58.verify.json"


def fig07():
    W, H = 3.25, 2.0
    f = Fig(W, H)
    A = f.A
    meta = A.meta("t58_ep1830_contact")
    cover = json.loads(T58V.read_text())["numbers"]["on_diagnosis"]["1830"]["cover_rows"]
    assert [r["t"] for r in cover] == [r["t"] for r in meta["cover_rows"]]
    last = [r for r in cover if r["t"] == meta["control_step"]][0]
    assert abs(last["closest_point_outside_spheres_mm"] - meta["closest_point_outside_spheres_mm"]) < 0.05
    # ---- left: scene
    crop = (0.20, 0.26, 0.80, 0.86)
    img = A.im("t58_ep1830_contact", crop)
    px, py = meta["hand_point_px"]; iw, ih = meta["image_px"]
    ax = f.panel((0.03, 0.05, 1.86, 1.86), img, border=C["hold"], lw=1.0, badge="✗ palm touches the post", bcol=C["hold"])
    u = (px / iw - crop[0]) / (crop[2] - crop[0]); v = 1 - (py / ih - crop[1]) / (crop[3] - crop[1])
    gap_txt = f.note(f"{last['closest_point_outside_spheres_mm']:.0f} mm", "closest hand point outside the sphere union at the contact step",
                     "t58_outside_mm")
    ax.annotate(f"closest hand point:\n{gap_txt} outside the\nquery spheres", xy=(u, v), xycoords="axes fraction", xytext=(0.04, 0.80),
                textcoords="axes fraction", fontsize=FS, color=C["ink"], ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#333333", lw=0.5),
                arrowprops=dict(arrowstyle="-|>", color=C["hold"], lw=0.7, shrinkA=0, shrinkB=3, mutation_scale=6), zorder=7)
    ax.text(0.96, 0.30, "query spheres\n(fingers, wrist)", transform=ax.transAxes, fontsize=FS, color=C["ink"], ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc=C["goldfill"], ec=C["gold"], lw=0.6), zorder=7)
    f.text(0.03, H - 0.05, "adjacent post, first contact step (replay)", size=FS, color=C["mute"])
    # ---- right: cover rows
    ax2 = f.fig.add_axes(f.F(2.33, 0.42, 0.87, 1.08))
    t = [r["t"] for r in cover]
    ax2.plot(t, [r["sphere_union_post_mm"] for r in cover], "o-", color=C["gold"], lw=1.0, ms=2.5, label="query spheres", zorder=3)
    ax2.plot(t, [r["real_hand_post_mm"] for r in cover], "o-", color=C["hold"], lw=1.0, ms=2.5, label="hand mesh", zorder=4)
    ax2.axhline(0, color="#777777", lw=0.5)
    ax2.set_xticks([t[0], t[-1]]); ax2.set_xticklabels([str(t[0]), str(t[-1])], fontsize=FS)
    ax2.set_yticks([0, 20, 40]); ax2.set_yticklabels(["0", "20", "40"], fontsize=FS)
    ax2.set_ylim(-3, 45); ax2.set_xlim(t[0] - 0.4, t[-1] + 0.4)
    ax2.tick_params(length=2, pad=1.5)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    ax2.set_xlabel("control step", fontsize=FS, labelpad=2)
    ax2.set_ylabel("distance to post (mm)", fontsize=FS, labelpad=2)
    ax2.legend(frameon=False, fontsize=FS, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, handlelength=1.1, handletextpad=0.4,
               borderaxespad=0.0, labelspacing=0.2)
    f.text(2.33 + 0.40, H - 0.05, "spheres vs hand", size=FS, fontweight="bold", ha="center")
    for r in cover:
        for k in ("real_hand_post_mm", "sphere_union_post_mm"):
            f.printed.append(dict(text=f"(curve point t={r['t']})", meaning=k, value=r[k], source=str(T58V),
                                  key=f"/numbers/on_diagnosis/1830/cover_rows[t={r['t']}]/{k}"))
    f.save("fig07_failure", dict(
        layout="single column: left replay render of ON_ep1830_s18301 at the first robot-post contact step with the left-hand query spheres "
               "(gold) and the real closest hand point (red); right: hand-mesh vs sphere-union distance to the post over the cover rows",
        scene=meta,
        provenance=("T58 (2026-10-13), AG3S ON arm = coarse-extent serving configuration C1a + F1 + W1; the adjacent-post failure also occurs "
                    "with the VLA alone (E0) in all 3 T58 episodes (T58.verify.json). Sphere set and pose reconstructed exactly as "
                    "verify/T58/p/diag_cover_t58.py (asserted against diag/cover_ON_ep1830.json)."),
        is_measured_result=True,
    ))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["fig01", "fig01_noCL", "fig02", "fig03", "fig04", "fig06", "fig07"]
    for w in which:
        globals()[w]()
