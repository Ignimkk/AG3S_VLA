"""Fill the Figure Labs layout of the method figure with recorded-scene renders.

The layout (`Method-Pipeline-Redraw-image-1.png`, Figure Labs draft, 1584 x 672) is kept as the
background. Every illustrative image slot is replaced by an existing MuJoCo replay render of a
recorded closed-loop run (read only). Text, chips and arrows of the layout are not changed, except
the in-slot labels of "Attention seeds" and the refinement legend, which are redrawn on top.

Run (CPU only):
    .venv-ag3s/bin/python paper/tools/plotting/fig03_realscene_compose.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib

from PIL import Image, ImageDraw, ImageFont

REPO = pathlib.Path(__file__).resolve().parents[3]
LAYOUT = REPO / "Method-Pipeline-Redraw-image-1.png"
OUT = REPO / "paper/figures/fig03_realscene"
S = 3  # output scale over the 1584 x 672 layout
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

P2 = "paper/figures/paperbanana/figure3/assets_v2/panels/"
IFG = "paper/figures/ifg_overview/assets/"
F2 = "paper/figures/paperbanana/figure2/assets/"
LTE = "AG3S/ag3s/docs/figures/core-eval/paper/paper-lte-link-field.png"

# Source pre-crops (source px). The two field panels share one window inside panels (b) and (c) of the
# LTE figure, above the legend of (c), so both show the same region at the same scale.
PRECROP = {"map_other": (496, 139, 657, 310), "map_contact": (880, 139, 1041, 310)}
ZOOM = {"attached": 2.5}

# slot -> (layout box x0, y0, x1, y1 incl. border, source, fit, focus (fx, fy), what it shows)
# Stage labels of the strip follow the physics ground truth of T64 Part B for ep1842
# (frame index -> stage: 0-17 approach, 18-20 grasp, 21-25 lift, 26-27 transport, 28-31 placement, 32+ retreat).
SLOTS = {
    "head":        ((17, 92, 148, 207), P2 + "01a_input_head.png", "cover", (0.50, 0.50), "ep1842 f016 head camera"),
    "left_wrist":  ((164, 92, 294, 207), P2 + "01b_input_left_wrist.png", "cover", (0.50, 0.50), "ep1842 f016 left wrist"),
    "right_wrist": ((310, 92, 441, 207), P2 + "01c_input_right_wrist.png", "cover", (0.55, 0.55), "ep1842 f016 right wrist"),
    "attention":   ((324, 289, 412, 375), IFG + "p_att_2.png", "cover", (0.46, 0.50), "ep1842 f016 right wrist, noun attention (L10 H7)"),
    "seeds":       ((505, 92, 621, 207), P2 + "02_ats_seed_patches.png", "cover", (0.45, 0.55), "ep1842 f016 seed patches"),
    "grow":        ((661, 92, 776, 207), P2 + "03_ats_membership.png", "cover", (0.50, 0.50), "ep1842 f016 graph membership"),
    "surface":     ((817, 92, 932, 207), P2 + "04_ats_target_surface.png", "cover", (0.55, 0.50), "ep1842 f016 target (orange) vs neighbour (blue)"),
    "map_other":   ((980, 92, 1133, 255), LTE, "cover", (0.50, 0.50), "core eval regression 954 k1 req 23, full-scene field (wrist, arm links)"),
    "map_contact": ((1148, 92, 1300, 255), LTE, "cover", (0.50, 0.50), "same request, target-free field (fingers, palm) + contact-link spheres"),
    "attached":    ((1249, 346, 1303, 420), F2 + "v2_role_attached.png", "cover", (0.58, 0.48), "ep1982 chunk 22, attached sphere"),
    "refine":      ((1340, 92, 1565, 285), P2 + "09_ccr_refined_path.png", "cover", (0.50, 0.50), "ep1982 chunk 20, reference vs refined"),
    "s_approach":  ((134, 485, 364, 607), IFG + "p_strip_M1_approach.png", "cover", (0.50, 0.55), "ep1842 f010 (truth: approach)"),
    "s_grasp":     ((373, 485, 603, 607), IFG + "p_strip_M3_grasp.png", "cover", (0.50, 0.50), "ep1842 f020 (truth: grasp)"),
    "s_lift":      ((612, 485, 842, 607), IFG + "p_strip_M4_carry.png", "cover", (0.50, 0.50), "ep1842 f025 (truth: lift)"),
    "s_transport": ((852, 485, 1081, 607), P2 + "08_lte_attached_object.png", "cover", (0.60, 0.58), "ep1842 chunk 26 (truth: transport), attached sphere"),
    "s_place":     ((1091, 485, 1321, 607), IFG + "p_strip_M5_pre_place.png", "cover", (0.50, 0.55), "ep1842 f031 (truth: placement)"),
    "s_retreat":   ((1330, 485, 1560, 607), IFG + "p_strip_M7_retreat.png", "cover", (0.50, 0.55), "ep1842 f037 (truth: retreat)"),
}
BORDER = (58, 58, 58)


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cover(img: Image.Image, w: int, h: int, focus, zoom: float = 1.0) -> Image.Image:
    sw, sh = img.size
    scale = max(w / sw, h / sh) * zoom
    cw, ch = w / scale, h / scale
    fx, fy = focus
    x0 = min(max(fx * sw - cw / 2, 0), sw - cw)
    y0 = min(max(fy * sh - ch / 2, 0), sh - ch)
    return img.crop((round(x0), round(y0), round(x0 + cw), round(y0 + ch))).resize((w, h), Image.LANCZOS)


def contain(img: Image.Image, w: int, h: int) -> Image.Image:
    sw, sh = img.size
    scale = min(w / sw, h / sh)
    nw, nh = round(sw * scale), round(sh * scale)
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(img.resize((nw, nh), Image.LANCZOS), ((w - nw) // 2, (h - nh) // 2))
    return canvas


def swatches(draw, x, y, items, size):
    """Small colour key drawn inside a slot (replaces the cropped-out legend)."""
    font = ImageFont.truetype(FONT, size)
    rows = []
    for col, txt in items:
        rows.append((col, txt, draw.textbbox((0, 0), txt, font=font)))
    wmax = max(b[2] for _c, _t, b in rows)
    lh = round(size * 1.25)
    pad = size // 3
    draw.rounded_rectangle((x - pad, y - pad, x + size + pad + wmax + 2 * pad, y + lh * len(rows) + pad // 2),
                           radius=pad, fill=(255, 255, 255, 220))
    for i, (col, txt, _b) in enumerate(rows):
        yy = y + i * lh
        draw.rectangle((x, yy + size // 6, x + size, yy + size), fill=col)
        draw.text((x + size + pad, yy), txt, font=font, fill=(20, 20, 20))


def label(draw, xy, text, size, fill=(20, 20, 20), bg=(255, 255, 255, 215)):
    font = ImageFont.truetype(FONT, size)
    x, y = xy
    l, t, r, b = draw.textbbox((x, y), text, font=font)
    pad = size // 4
    draw.rounded_rectangle((l - pad, t - pad, r + pad, b + pad), radius=pad, fill=bg)
    draw.text((x, y), text, font=font, fill=fill)
    return r + pad


def main():
    base = Image.open(LAYOUT).convert("RGB")
    W, H = base.size
    out = base.resize((W * S, H * S), Image.LANCZOS)

    srcs = {k: Image.open(REPO / v[1]).convert("RGB") for k, v in SLOTS.items()}
    for k, box in PRECROP.items():
        srcs[k] = srcs[k].crop(box)

    draw = ImageDraw.Draw(out, "RGBA")
    for k, (box, _src, fit, focus, _what) in SLOTS.items():
        x0, y0, x1, y1 = (v * S for v in box)
        ix0, iy0, ix1, iy1 = x0 + S, y0 + S, x1 - S + 1, y1 - S + 1
        w, h = ix1 - ix0, iy1 - iy0
        tile = cover(srcs[k], w, h, focus, ZOOM.get(k, 1.0)) if fit == "cover" else contain(srcs[k], w, h)
        out.paste(tile, (ix0, iy0))
        draw.rectangle((x0, y0, x1, y1), outline=BORDER, width=S)

    # colour keys of the two field panels (colours as rendered in the LTE figure)
    fk = 10 * S
    x0, _y0, _x1, y1 = (v * S for v in SLOTS["map_other"][0])
    swatches(draw, x0 + 5 * S, y1 - 19 * S, [((140, 140, 150), "scene surface")], fk)
    x0, _y0, _x1, y1 = (v * S for v in SLOTS["map_contact"][0])
    swatches(draw, x0 + 5 * S, y1 - 32 * S, [((214, 40, 40), "target removed"), ((70, 120, 220), "contact links")], fk)

    # in-slot labels of the layout that the paste covered
    x0, y0 = SLOTS["seeds"][0][0] * S, SLOTS["seeds"][0][1] * S
    label(draw, (x0 + 4 * S, y0 + 3 * S), "Attention seeds", 12 * S)
    x0, y0, x1, y1 = (v * S for v in SLOTS["refine"][0])
    fs = 11 * S
    lx, ly = x0 + 6 * S, y1 - 34 * S
    draw.rounded_rectangle((lx - 3 * S, ly - 3 * S, lx + 90 * S, ly + 29 * S), radius=3 * S, fill=(255, 255, 255, 215))
    font = ImageFont.truetype(FONT, fs)
    for i, (txt, col, dashed) in enumerate((("Original", (35, 35, 35), True), ("Refined", (212, 160, 23), False))):
        yy = ly + 6 * S + i * 14 * S
        if dashed:
            for xx in range(lx, lx + 18 * S, 6 * S):
                draw.line((xx, yy, xx + 3 * S, yy), fill=col, width=2 * S)
        else:
            draw.line((lx, yy, lx + 18 * S, yy), fill=col, width=2 * S)
        draw.text((lx + 22 * S, yy - fs // 2 - S), txt, font=font, fill=(20, 20, 20))

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "fig03_method_pipeline_realscene.png"
    pdf = OUT / "fig03_method_pipeline_realscene.pdf"
    out.save(png, dpi=(round(W * S / 7.16), round(W * S / 7.16)))
    out.save(pdf, resolution=W * S / 7.16)

    side = {
        "asset": str(png.relative_to(REPO)),
        "script": "paper/tools/plotting/fig03_realscene_compose.py",
        "layout": {"path": str(LAYOUT.relative_to(REPO)), "sha256": sha(LAYOUT), "size": [W, H], "origin": "Figure Labs draft"},
        "scale": S,
        "size_px": list(out.size),
        "width_in": 7.16,
        "is_measured_result": False,
        "renderer": "MuJoCo 3.11.0 replays of recorded qpos (existing assets, read only)",
        "slots": {k: {"box_layout_px": list(v[0]), "source": v[1], "sha256": sha(REPO / v[1]), "fit": v[2],
                      "focus": v[3], "precrop_src_px": list(PRECROP[k]) if k in PRECROP else None,
                      "zoom": ZOOM.get(k, 1.0), "shows": v[4]} for k, v in SLOTS.items()},
        "strip_stage_truth": {
            "episode": "T53 C1a ep1842 (re-simulated in T64 Part B, bit-identical)",
            "source": "/mnt/dev/AG3S_VLA-storage/results/development/20261029T013324+0900_b3b69311f948_7da63daefa49/verify/T64/partB/partB.json",
            "rule": "frame index -> physics ground-truth stage; 0-17 approach, 18-20 grasp, 21-25 lift, 26-27 transport, 28-31 placement, 32+ retreat",
        },
        "episodes": {
            "ep1842": "pear touching an orange (inputs, attention, segmentation, strip)",
            "regression 954 k1": "apple next to a banana, core evaluation 2026-10-28 (collision-map panels)",
            "ep1982": "apple carried over a bar (refinement, attached-object icon)",
        },
        "limits": [
            "Panels come from three recorded episodes; the caption must not imply one episode.",
            "The collision-map panels are cropped from paper-lte-link-field.png (b, c); its legend is replaced by in-slot keys.",
            "The transport frame (ep1842 chunk 26) uses a different camera from the other five strip frames.",
            "Layout text is the Figure Labs raster upscaled x3; final text should be re-set as vector.",
            "No performance number is printed in any panel.",
        ],
    }
    (OUT / "fig03_method_pipeline_realscene.json").write_text(json.dumps(side, indent=1))
    print(png, out.size)


if __name__ == "__main__":
    main()
