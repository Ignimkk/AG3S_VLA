"""CVPR 2027 Fig. 1 v4: one SafeLIBERO scene, prior safety layer (left) vs AG3S (right).

Layout follows the user's figurelabs mock (2026-11-09): shared instruction bubble, two halves on the SAME episode
(same initial state and seed), a main frame per half at the same step, an end-of-episode thumbnail with a colour key,
and three zoomed insets per half in the same axis order (identity, extent, permission).
Every raster is a recorded frame or an overlay of recorded geometry made by fig1_v4_assets.py (provenance in
v4/assets/assets.json); nothing is generated or repainted. No number is printed in the figure.

Helpers (Fig with layout_audit, fit, colour code, font setup) are imported from cvpr_v3_fig1_compose.py.

    .venv-ag3s/bin/python paper/tools/plotting/cvpr_v4_fig1_compose.py
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import subprocess
import sys

import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cvpr_v3_fig1_compose as v3  # noqa: E402
from cvpr_v3_fig1_compose import C, FS, PDF_DPI, PNG_DPI, REPO, STRICT_LAYOUT, TEXTWIDTH_IN, Fig, sha  # noqa: E402

V4 = REPO / "paper/figures/fig01_teaser/v4"
ASSETS = V4 / "assets"
META = json.loads((ASSETS / "assets.json").read_text())
INSTR = "pick up the black bowl on the stove and place it on the plate"
GREY_BOX = "#3C3C3C"


class Panels:
    def __init__(self):
        self.used = []

    def get(self, name, slot, centre, width, aspect):
        """Crop ``name`` (768 px) to a window of ``width`` px and aspect w/h centred at ``centre`` (clamped)."""
        p = ASSETS / f"{name}.png"
        a = np.asarray(Image.open(p).convert("RGB"))
        H, W = a.shape[:2]
        w = int(round(min(width, W, H * aspect))); h = int(round(w / aspect))
        x0 = int(np.clip(centre[0] - w / 2, 0, W - w)); y0 = int(np.clip(centre[1] - h / 2, 0, H - h))
        self.used.append(dict(slot=slot, file=str(p.relative_to(REPO)), sha256=sha(p), crop_px=[x0, y0, w, h],
                              provenance=META.get(f"{name}.png", {})))
        return a[y0:y0 + h, x0:x0 + w], (x0, y0, w, h)


def fig01_v4():
    W = TEXTWIDTH_IN
    M, MID, GAPH = 0.03, TEXTWIDTH_IN / 2, 0.07
    HALF = MID - M - GAPH                                     # usable width of one half
    MAIN_H = 1.30; MAIN_ASP = 1.60; MAIN_W = MAIN_H * MAIN_ASP
    COL_W = HALF - MAIN_W - 0.08                              # thumbnail + key column
    INS_W = (HALF - 2 * 0.07) / 3; INS_H = 0.70; INS_ASP = INS_W / INS_H
    LAB_H = 0.27
    H = round(0.30 + 0.20 + MAIN_H + 0.08 + INS_H + LAB_H + 0.03, 3)
    f = Fig(W, H)
    PN = Panels()
    halves = {"L": M, "R": MID + GAPH}

    # ---------------------------------------------------------------- shared instruction bubble + divider
    by = H - 0.17
    f.text(MID, by, f"“{INSTR}”", size=8, style="italic", ha="center", color=C["ink"], zorder=6,
           bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.8", fc="white", ec="#9A9A9A", lw=0.8))
    f.fig.add_artist(Polygon([((MID - 0.10) / W, (by - 0.115) / H), ((MID - 0.02) / W, (by - 0.115) / H),
                              ((MID - 0.13) / W, (by - 0.20) / H)], closed=True, transform=f.fig.transFigure,
                             fc="white", ec="#9A9A9A", lw=0.8, zorder=5))
    top_main = H - 0.30 - 0.20
    f.line([(MID, M), (MID, top_main)], "#B0B0B0", lw=0.8)
    f.text(W - M - 0.04, by, "same scene \u00b7 same seed", size=FS, ha="right", color=C["mute"], zorder=6,
           bbox=dict(boxstyle="round,pad=0.3", fc="#F4F4F4", ec="#C8C8C8", lw=0.6))

    # ---------------------------------------------------------------- headers
    hy = H - 0.30 - 0.09
    f.text(halves["L"], hy, "Prior: external detector + safety filter", size=8.5, fontweight="bold", color=C["red"])
    f.text(halves["R"], hy, "AG3S: the policy's own attention grounds contact", size=8.5, fontweight="bold", color=C["green"])

    # ---------------------------------------------------------------- main frames (same step) + end thumbnails + key
    y_main = top_main - MAIN_H
    for side, name, badge, bcol, bord in (("L", "l_main", "✗ bottle knocked over", C["red"], C["red"]),
                                          ("R", "r_main", "✓ bottle untouched", C["green"], C["gold"])):
        img, _ = PN.get(name, f"{side} main (same step)", (384, 330), 768, MAIN_ASP)
        f.panel((halves[side], y_main, MAIN_W, MAIN_H), img, border=bord, lw=1.0, badge=badge, bcol=bcol, badge_y=0.11)
    # "missed" tag on the toppled bottle (AEGIS frame, obs 110; bottle around px (500, 470) of 768)
    _, (cx0, cy0, cw, ch) = PN.used[0]["crop_px"][:2], PN.used[0]["crop_px"]
    bx = halves["L"] + (505 - cx0) / cw * MAIN_W; byy = y_main + (1 - (455 - cy0) / ch) * MAIN_H
    f.arrow((bx + 0.30, byy + 0.30), (bx + 0.06, byy + 0.05), color=C["red"], lw=1.0, ms=6)
    f.text(bx + 0.31, byy + 0.36, "no box", size=FS, fontweight="bold", color=C["red"], ha="center", zorder=7,
           bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C["red"], lw=0.6))

    keys = {"L": [("dash", GREY_BOX, "detector box"), ("patch", C["red"], "hand ellipsoid")],
            "R": [("patch", C["neigh"], "kept as obstacle"), ("patch", C["target"], "target surface"),
                  ("patch", C["gold"], "seeds / contact links"), ("patch", C["attached"], "attached bowl")]}
    ends = {"L": ("o_aegis_final", "episode end"), "R": ("o_ag3s_final", "episode end")}
    for side in ("L", "R"):
        cx = halves[side] + MAIN_W + 0.08
        th = 0.66; tw = COL_W
        img, _ = PN.get(ends[side][0], f"{side} end of episode", (384, 360), 768, tw / th)
        ty = top_main - th
        f.panel((cx, ty, tw, th), img, border="#9A9A9A", lw=0.7)
        f.text(cx, ty - 0.10, ends[side][1], size=FS, color=C["mute"])
        ky = ty - 0.25
        for k, (kind, col, lab) in enumerate(keys[side]):
            yy = ky - k * 0.12
            if kind == "dash":
                f.line([(cx, yy), (cx + 0.15, yy)], col, lw=1.0, ls=(0, (2, 1.5)))
            else:
                f.band(cx, yy - 0.035, 0.15, 0.07, col, "none", lw=0, z=3, r=0.01)
            f.text(cx + 0.20, yy, lab, size=FS, color=C["ink"])

    # ---------------------------------------------------------------- insets (same axis order on both halves)
    ins = {"L": [("l_boxes", (384, 450), 560, "Identity", "bottle never boxed"),
                 ("l_boxes", (165, 345), 330, "Extent", "one box, two objects"),
                 ("l_ee", (590, 375), 300, "Permission", "bowl not modelled")],
           "R": [("r_identity", (300, 345), 300, "Identity", "attention seeds bowl"),
                 ("r_extent", (280, 330), 270, "Extent", "bowl surface only"),
                 ("r_permission", (380, 215), 330, "Permission", "contact links + bowl")]}
    y_ins = y_main - 0.08 - INS_H
    for side in ("L", "R"):
        icon, icol, bord = ("✗", C["red"], C["red"]) if side == "L" else ("✓", C["green"], C["gold"])
        for k, (name, ctr, wpx, axis, desc) in enumerate(ins[side]):
            x = halves[side] + k * (INS_W + 0.07)
            img, _ = PN.get(name, f"{side} inset {axis}", ctr, wpx, INS_ASP)
            f.panel((x, y_ins, INS_W, INS_H), img, border=bord, lw=0.9)
            f.text(x, y_ins - 0.10, icon, size=8, fontweight="bold", color=icol)
            f.text(x + 0.12, y_ins - 0.10, f"{k + 1} {axis}", size=FS, fontweight="bold", color=C["ink"])
            f.text(x, y_ins - 0.215, desc, size=FS, color=C["mute"])
    save(f, PN, "fig01_v4")


def save(f, PN, stem):
    mn, mx, cnt = f.audit()
    assert mn >= FS - 1e-6, f"{stem}: font below {FS} pt: {mn}"
    viol = f.layout_audit()
    n_viol = sum(len(v) for v in viol.values())
    print(f"[{stem}] {f.W:.3f} x {f.H:.3f} in, layout audit: " + ", ".join(f"{k}={len(v)}" for k, v in viol.items()))
    for k, v in viol.items():
        for item in v[:40]:
            print("   ", k, item)
    texts = [t.get_text() for t in f.fig.findobj(v3.mtext.Text) if t.get_text().strip() and t.get_visible()]
    if STRICT_LAYOUT:
        assert n_viol == 0, f"{stem}: {n_viol} layout violations"
    pdf = V4 / f"{stem}.pdf"
    f.fig.savefig(pdf, dpi=PDF_DPI)
    f.fig.savefig(pdf.with_suffix(".png"), dpi=PNG_DPI)
    v3.plt.close(f.fig)
    side = dict(
        artifact=stem, pdf=str(pdf.relative_to(REPO)), png=str(pdf.with_suffix(".png").relative_to(REPO)),
        size_in=[f.W, f.H], textwidth_in=TEXTWIDTH_IN, raster_dpi_in_pdf=PDF_DPI, png_dpi=PNG_DPI,
        fonts=dict(family="DejaVu Sans", pdf_fonttype=42, min_pt=mn, max_pt=mx, n_text_items=cnt, floor_pt=FS),
        printed_text=texts, printed_numbers=[],
        layout_audit=dict(tolerance_pt=1.0, min_text_to_panel_border_in=0.05, violations={k: len(v) for k, v in viol.items()},
                          n_panels=len(f.panels), n_texts=cnt),
        panels=PN.used, episode=META.get("_episode"),
        honesty=["every panel is a recorded frame of SafeLIBERO spatial Level I task 2 episode 9 (init 9, seed 7) from the "
                 "named run, or an overlay of geometry recorded in that run (fig1_v4_assets.py); no generated imagery",
                 "left main / identity / extent: AEGIS run (detector boxes recorded once per episode); left permission: "
                 "KNOWS run (end-effector ellipsoid from knows/adapter.py GRIPPER_SEMI_AXES at the recorded eef pose); "
                 "KNOWS obstacle ellipsoids are not drawn because their rotation was not recorded",
                 "right panels: AG3S final run pe123458; overlays are recorded points / voxels / spheres mapped to depth pixels",
                 "the AEGIS and AG3S runs differ in commit (89b424595b57 vs a811331c11d8); same initial state and seed",
                 "init 9 lies in the init 5-14 split already used to choose the configuration (dev-4a note, 2026-11-09)",
                 "no number printed in the figure"],
        layout_source="user's figurelabs mock (2026-11-09) restructured with recorded frames",
        script="paper/tools/plotting/cvpr_v4_fig1_compose.py", generated=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        git_head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
        is_measured_result=False, scope="qualitative same-episode comparison; no numbers",
    )
    pdf.with_suffix(".json").write_text(json.dumps(side, indent=1, ensure_ascii=False, default=str) + "\n")
    print(f"wrote {pdf.relative_to(REPO)}  min font {mn} pt")


if __name__ == "__main__":
    fig01_v4()
