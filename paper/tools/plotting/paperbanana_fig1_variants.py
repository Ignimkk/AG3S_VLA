"""Compose the two Figure 1 variants (layout drafts for PaperBanana).

A  performance teaser: VLA-only failures (pick, place) over AG3S successes, a "how" strip, a small
   performance column.
B  limitation-first: what existing approaches get wrong, then why AG3S is needed.

Scenes: paperbanana_fig1_scenes.py (T40 hurdle replays). Mechanism panels: the T53 renders of
paperbanana_render.py. Performance bars are PLACEHOLDER values for layout only (labelled on the chart).

    MUJOCO_GL=osmesa .venv-ag3s/bin/python paper/tools/plotting/paperbanana_fig1_variants.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paperbanana_render as pr  # noqa: E402

ROOT = pr.OUT / "figure1/variants"
A = ROOT / "assets"
C = {"red": "#C62828", "green": "#2E7D32", "gold": "#D9A406", "ink": "#1F1F1F", "grey": "#9E9E9E",
     "vla": "#9E9E9E", "ours": "#D9A406"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

PLACEHOLDER = {"Task success": (45, 85), "Collision-free": (40, 95), "Safe success": (25, 82)}


def im(path, crop=None):
    x = Image.open(path).convert("RGB")
    if crop:
        w, h = x.size
        x = x.crop((int(crop[0] * w), int(crop[1] * h), int(crop[2] * w), int(crop[3] * h)))
    return np.asarray(x)


def fit(arr, aspect, anchor=(0.5, 0.5)):
    """Crop `arr` (never stretch) to width/height = `aspect` around `anchor` (fractions)."""
    h, w = arr.shape[:2]
    if w / h > aspect:
        nw = int(round(h * aspect)); x = int(round((w - nw) * anchor[0]))
        return arr[:, x:x + nw]
    nh = int(round(w / aspect)); y = int(round((h - nh) * anchor[1]))
    return arr[y:y + nh]


def pad(arr, aspect):
    """Pad `arr` with white to width/height = `aspect` (for charts that must not be cropped)."""
    h, w = arr.shape[:2]
    if w / h < aspect:
        nw = int(round(h * aspect)); out = np.full((h, nw, 3), 255, np.uint8); x = (nw - w) // 2
        out[:, x:x + w] = arr[..., :3]
    else:
        nh = int(round(w / aspect)); out = np.full((nh, w, 3), 255, np.uint8); y = (nh - h) // 2
        out[y:y + h] = arr[..., :3]
    return out


def panel(fig, rect, arr, title=None, badge=None, badge_col=None, border="#BBBBBB", lw=1.6, tsize=11, anchor=(0.5, 0.5),
          keep=False):
    ax = fig.add_axes(rect)
    fw, fh = fig.get_size_inches()
    aspect = rect[2] * fw / (rect[3] * fh)
    ax.imshow(pad(arr, aspect) if keep else fit(arr, aspect, anchor), aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(border); s.set_linewidth(lw)
    if title:
        ax.set_title(title, fontsize=tsize, fontweight="bold", color=C["ink"], pad=4)
    if badge:
        ax.text(0.02, 0.04, badge, transform=ax.transAxes, fontsize=11, fontweight="bold", color="white",
                ha="left", va="bottom", bbox=dict(boxstyle="round,pad=0.35", fc=badge_col, ec="none", alpha=0.95))
    return ax


def arrow(fig, p0, p1, color="#555555", lw=2.2):
    fig.add_artist(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=16,
                                   color=color, lw=lw))


def band(fig, x, y, w, h, fc, ec):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.01",
                                  transform=fig.transFigure, fc=fc, ec=ec, lw=1.4, zorder=-2))


# ------------------------------------------------------------------ small charts
def hsr_chart():
    run = pr.Run()
    p, states = [], []
    for j in range(len(run.chunks)):
        s = json.loads(str(np.load(run.chunks[j])["summary_json"]))
        sp = s.get("subtask", {}).get("p") or {}
        p.append([sp.get("pick", np.nan), sp.get("place", np.nan), sp.get("home", np.nan)])
        states.append(s.get("grasp", {}).get("state"))
    p = np.asarray(p, float)
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    lab = np.nanargmax(np.nan_to_num(p, nan=-1), axis=1)
    cols = ["#DEEBF7", "#FBE5D6", "#E2F0D9"]
    for j, l in enumerate(lab):
        ax.axvspan(j - 0.5, j + 0.5, color=cols[l], lw=0)
    for k, (name, col) in enumerate(zip(["pick", "place", "home"], ["#2E75B6", "#C55A11", "#548235"])):
        ax.plot(np.arange(len(p)), p[:, k], color=col, lw=2.2, label=name)
    ax.set_ylim(-0.05, 1.12); ax.set_yticks([0, 1]); ax.set_xticks([])
    ax.set_xlabel("time (action chunks)"); ax.set_ylabel("probe p")
    ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.13))
    ax.spines[["top", "right"]].set_visible(False)
    out = A / "hsr_timeline.png"
    fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)
    out.with_suffix(".json").write_text(json.dumps({"run": "T53 C1a ep1842", "note": "recorded sub-task probe p per chunk",
                                                    "n_chunks": len(p)}, indent=1) + "\n")
    return out


def perf_chart(vertical=True):
    fig, axs = plt.subplots(len(PLACEHOLDER), 1, figsize=(2.9, 6.4))
    for ax, (name, (b, o)) in zip(axs, PLACEHOLDER.items()):
        bars = ax.barh([1, 0], [b, o], color=[C["vla"], C["ours"]], height=0.62)
        for bar, v in zip(bars, [b, o]):
            ax.text(v + 2, bar.get_y() + bar.get_height() / 2, f"{v}", va="center", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 112); ax.set_yticks([1, 0]); ax.set_yticklabels(["VLA only", "+ AG3S"], fontsize=10)
        ax.set_xticks([]); ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.set_title(f"{name} (%)", fontsize=11, fontweight="bold", loc="left")
    fig.text(0.5, 0.005, "placeholder values", ha="center", fontsize=8.5, color="#888888", style="italic")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = A / "perf_placeholder.png"
    fig.savefig(out, dpi=220); plt.close(fig)
    out.with_suffix(".json").write_text(json.dumps({"values": PLACEHOLDER, "is_measured_result": False,
                                                    "note": "PLACEHOLDER layout values; keep the placeholder label or replace with sourced values whose status is shown"},
                                                   indent=1) + "\n")
    return out


# ------------------------------------------------------------------ variant A
def variant_a(hsr, perf):
    fig = plt.figure(figsize=(16, 10.2))
    crop = (0.0, 0.06, 1.0, 0.86)
    x0, w, gap = 0.115, 0.318, 0.012
    r1, r2, h = 0.625, 0.300, 0.300
    fig.text(x0 + w / 2, 0.952, "Pick: obstacle next to the target", ha="center", fontsize=13, fontweight="bold")
    fig.text(x0 + w + gap + w / 2, 0.952, "Place: hurdle on the carry path", ha="center", fontsize=13, fontweight="bold")
    band(fig, 0.008, r1 - 0.012, 0.765, h + 0.024, "#FDECEA", "#E8A9A3")
    band(fig, 0.008, r2 - 0.012, 0.765, h + 0.024, "#FFF7DD", C["gold"])
    fig.text(0.058, r1 + h / 2, "VLA only", ha="center", va="center", fontsize=14, fontweight="bold", color=C["red"],
             rotation=90)
    fig.text(0.058, r2 + h / 2, "VLA + AG3S\n(ours)", ha="center", va="center", fontsize=14, fontweight="bold",
             color="#9A7300", rotation=90)
    panel(fig, [x0, r1, w, h], im(A / "vla_pick_fail.png", crop), badge="✗  collides → pick fails",
          badge_col=C["red"], border=C["red"])
    panel(fig, [x0 + w + gap, r1, w, h], im(A / "vla_place_fail.png", crop), badge="✗  hits the hurdle → place fails",
          badge_col=C["red"], border=C["red"])
    panel(fig, [x0, r2, w, h], im(A / "ag3s_pick_ok.png", crop), badge="✓  avoids, then grasps",
          badge_col=C["green"], border=C["gold"], lw=2.2)
    panel(fig, [x0 + w + gap, r2, w, h], im(A / "ag3s_place_ok.png", crop), badge="✓  clears the hurdle, places",
          badge_col=C["green"], border=C["gold"], lw=2.2)

    # how strip
    fig.text(0.012, 0.245, "How", fontsize=13, fontweight="bold", color=C["ink"])
    hw, hh, hy = 0.172, 0.205, 0.022
    xs = [0.012 + k * (hw + 0.022) for k in range(4)]
    items = [(im(pr.OUT / "figure1/assets/f1_wrist_attention.png", (0.10, 0.20, 0.95, 0.98)), "1  VLA attention"),
             (im(pr.OUT / "figure1/assets/f1_grounding_scene.png", (0.12, 0.08, 0.88, 0.82)), "2  3D target grounding"),
             (im(hsr), "3  Sub-task recognition (HSR)", True),
             (im(pr.OUT / "figure2/assets/f2c_ownership_render.png", (0.13, 0.02, 0.92, 0.86)), "4  Link-specific contact field")]
    for k, (item, x) in enumerate(zip(items, xs)):
        arr, title = item[0], item[1]
        panel(fig, [x, hy, hw, hh], arr, title=title, border=C["gold"] if k != 2 else "#BBBBBB", tsize=10.5,
              keep=len(item) > 2)
        if k < 3:
            arrow(fig, (x + hw + 0.002, hy + hh / 2), (x + hw + 0.020, hy + hh / 2), color=C["gold"])

    # performance column
    band(fig, 0.79, 0.30, 0.20, 0.645, "#FAFAFA", "#CCCCCC")
    fig.text(0.89, 0.925, "Performance", ha="center", fontsize=13, fontweight="bold")
    panel(fig, [0.80, 0.315, 0.18, 0.59], im(perf), border="#FAFAFA", keep=True)
    fig.text(0.89, 0.23, "frozen VLA · no extra training\nno external segmenter", ha="center", fontsize=10.5,
             style="italic", color="#444444")
    out = ROOT / "variant_A_performance.png"
    fig.savefig(out, dpi=170, facecolor="white"); plt.close(fig)
    print("wrote", out.relative_to(pr.REPO))


# ------------------------------------------------------------------ variant B
def variant_b():
    fig = plt.figure(figsize=(16, 9.6))
    crop2 = (0.13, 0.02, 0.92, 0.86)
    band(fig, 0.008, 0.535, 0.984, 0.44, "#FDECEA", "#E8A9A3")
    fig.text(0.02, 0.945, "Existing approaches cannot tell needed contact from collision", fontsize=14,
             fontweight="bold", color=C["red"])
    w, h, y = 0.313, 0.345, 0.555
    tops = [(im(A / "vla_place_fail.png", (0.0, 0.06, 1.0, 0.86)), "(a) VLA only", "✗  no 3D constraint: hits the hurdle"),
            (im(pr.OUT / "figure2/assets/f2a_scene_render.png", crop2), "(b) Geometry-only safety filter",
             "✗  target is an obstacle: grasp refused"),
            (im(pr.OUT / "figure2/assets/f2b_ball_render.png", crop2), "(c) Global target exclusion",
             "✗  neighbour erased: unsafe contact")]
    for k, (arr, title, badge) in enumerate(tops):
        panel(fig, [0.02 + k * (w + 0.012), y, w, h], arr, title=title, badge=badge, badge_col=C["red"], border=C["red"])

    band(fig, 0.008, 0.02, 0.984, 0.495, "#FFF7DD", C["gold"])
    fig.text(0.02, 0.482, "AG3S: ground the target in 3D, allow contact only where the task needs it", fontsize=14,
             fontweight="bold", color="#9A7300")
    mw, mh, my = 0.150, 0.185, 0.235
    mech = [(im(pr.OUT / "figure1/assets/f1_wrist_attention.png", (0.10, 0.20, 0.95, 0.98)), "VLA attention"),
            (im(pr.OUT / "figure2/assets/f2c_ownership_render.png", crop2), "Boundary-preserving extent"),
            (im(pr.OUT / "figure2/assets/f2c_link_render.png", (0.10, 0.0, 0.90, 0.85)), "Contact links only")]
    for k, (arr, title) in enumerate(mech):
        x = 0.02 + k * (mw + 0.018)
        panel(fig, [x, my, mw, mh], arr, title=title, border=C["gold"], tsize=10.5)
        if k < 2:
            arrow(fig, (x + mw + 0.001, my + mh / 2), (x + mw + 0.017, my + mh / 2), color=C["gold"])
    fig.text(0.02, 0.17, "• the frozen VLA's own attention picks the target (no segmenter)\n"
                         "• depth gives its observed extent; the touching neighbour is kept\n"
                         "• only the grasping links may enter that extent; the hurdle stays an obstacle",
             fontsize=10.5, va="top", color=C["ink"], linespacing=1.6)
    ow, oh, oy = 0.212, 0.385, 0.045
    outs = [(im(A / "ag3s_pick_ok.png", (0.0, 0.06, 1.0, 0.86)), "✓  pick next to an obstacle"),
            (im(A / "ag3s_place_ok.png", (0.0, 0.06, 1.0, 0.86)), "✓  place over the hurdle")]
    arrow(fig, (0.515, 0.3275), (0.537, 0.3275), color=C["gold"], lw=3)
    for k, (arr, badge) in enumerate(outs):
        panel(fig, [0.542 + k * (ow + 0.012), oy, ow, oh], arr, badge=badge, badge_col=C["green"], border=C["gold"],
              lw=2.2, anchor=(0.62, 0.4))
    out = ROOT / "variant_B_limitation.png"
    fig.savefig(out, dpi=170, facecolor="white"); plt.close(fig)
    print("wrote", out.relative_to(pr.REPO))


def main():
    A.mkdir(parents=True, exist_ok=True)
    hsr = hsr_chart()
    perf = perf_chart()
    variant_a(hsr, perf)
    variant_b()


if __name__ == "__main__":
    main()
