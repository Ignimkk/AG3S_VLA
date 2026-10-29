"""Compose the IFG-revised overview montage (same layout family as paperbanana figure2 reference_montage_v2).

Top: framework strip of the adopted configuration B. Bottom: three questions, each "prior vs AG3S" on real
replay renders made by ifg_overview_render.py. Numbers printed on the figure are read from the asset sidecars.

    .venv-ag3s/bin/python paper/tools/plotting/ifg_overview_compose.py
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3]
A = REPO / "paper/figures/ifg_overview/assets"
OUT = REPO / "paper/figures/ifg_overview/overview_montage.png"
C = {"red": "#C62828", "green": "#2E7D32", "gold": "#D9A406", "ink": "#1F1F1F", "navy": "#C9DCF0",
     "goldfill": "#FFF2CC", "field": "#DAE8FC", "state": "#FBE5D6", "ctrl": "#E4DFEC", "grey": "#E7E6E6",
     "teal": "#2B7A78", "tealfill": "#DDF0EE", "brown": "#A0450F"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def meta(name):
    return json.loads((A / f"{name}.json").read_text())


def im(name, crop=None):
    x = Image.open(A / f"{name}.png").convert("RGB")
    if crop:
        w, h = x.size
        x = x.crop((int(crop[0] * w), int(crop[1] * h), int(crop[2] * w), int(crop[3] * h)))
    return np.asarray(x)


def fit(a, asp, anchor=(0.5, 0.5)):
    h, w = a.shape[:2]
    if w / h > asp:
        nw = int(h * asp); x0 = int((w - nw) * anchor[0]); return a[:, x0:x0 + nw]
    nh = int(w / asp); y0 = int((h - nh) * anchor[1]); return a[y0:y0 + nh]


def panel(fig, rect, a, badge=None, bcol=None, border="#BBBBBB", lw=1.6, anchor=(0.5, 0.5)):
    ax = fig.add_axes(rect, zorder=3)
    fw, fh = fig.get_size_inches()
    ax.imshow(fit(a, rect[2] * fw / (rect[3] * fh), anchor), aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(border); s.set_linewidth(lw)
    if badge:
        ax.text(0.025, 0.04, badge, transform=ax.transAxes, fontsize=9.3, fontweight="bold", color="white", ha="left",
                va="bottom", bbox=dict(boxstyle="round,pad=0.3", fc=bcol, ec="none"))


def blk(fig, x, y, w, h, text, fc, ec="#555555", size=9, bold=False, ls="-"):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.002,rounding_size=0.006",
                                  transform=fig.transFigure, fc=fc, ec=ec, lw=1.2, ls=ls, zorder=2))
    fig.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, zorder=5,
             fontweight="bold" if bold else "normal", color=C["ink"])


def arr(fig, a, b, color="#555555", lw=1.6, ms=11, ls="-"):
    fig.add_artist(FancyArrowPatch(a, b, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=ms, color=color,
                                   lw=lw, ls=ls, zorder=4))


def band(fig, x, y, w, h, fc, ec, z=-2, lw=1.4):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.01",
                                  transform=fig.transFigure, fc=fc, ec=ec, lw=lw, zorder=z))


def tag(fig, x, y, text, fc, size=9.5):
    fig.text(x, y, text, fontsize=size, fontweight="bold", color="white", ha="left", va="center", zorder=6,
             bbox=dict(boxstyle="round,pad=0.3", fc=fc, ec="none"))


def compose():
    m = {k: meta(k) for k in ("c1_prior_seeds_on_support", "c1_ours_seed_region", "c2_prior_radius_cluster",
                              "c2_ours_surface_region", "c2_prior_sphere_mask", "c3_prior_raw_cosine",
                              "c3_ours_centered_cosine", "c3_ours_alert")}
    fig = plt.figure(figsize=(17, 11.2))
    fig.text(0.012, 0.972, "AG3S: the frozen VLA says which object; observed depth says where its surface ends",
             fontsize=14.5, fontweight="bold", color=C["ink"])

    # ---------------------------------------------------------------- framework strip (configuration B)
    band(fig, 0.008, 0.735, 0.984, 0.215, "#FBFBFB", "#CCCCCC")
    blk(fig, 0.016, 0.838, 0.07, 0.06, "RGB +\ninstruction", C["grey"], size=9.2)
    blk(fig, 0.016, 0.745, 0.07, 0.06, "depth +\nrobot model", C["grey"], size=9.2)
    blk(fig, 0.105, 0.833, 0.085, 0.07, "frozen VLA\n(no retraining)", C["navy"], size=10.2, bold=True)
    arr(fig, (0.086, 0.868), (0.105, 0.868))
    outs = [(0.912, "actions", C["navy"], "#5B7FA6"), (0.868, "noun attention", C["goldfill"], C["gold"]),
            (0.824, "internal features", C["tealfill"], C["teal"]), (0.780, "hidden state", C["state"], "#C55A11")]
    for yy, t, fc, ec in outs:
        blk(fig, 0.21, yy - 0.016, 0.085, 0.032, t, fc, ec=ec, size=8.8)
        arr(fig, (0.19, 0.868), (0.21, yy), color=ec)
    # column 2: seeds, memory, surface graph
    blk(fig, 0.32, 0.848, 0.15, 0.04, "eligible attention seeds\n(robot / support patches excluded)", C["goldfill"], ec=C["gold"], size=8.6, bold=True)
    tag(fig, 0.322, 0.899, "Q1  Which object", C["gold"], 9)
    arr(fig, (0.295, 0.868), (0.32, 0.868), color=C["gold"])
    blk(fig, 0.32, 0.808, 0.15, 0.032, "centered feature memory", C["tealfill"], ec=C["teal"], size=8.6, bold=True)
    arr(fig, (0.295, 0.824), (0.32, 0.824), color=C["teal"])
    blk(fig, 0.32, 0.742, 0.15, 0.036, "rendered robot mask +\n5 mm depth-continuous surface graph", "#F2F2F2", size=8.3)
    arr(fig, (0.086, 0.758), (0.32, 0.758))
    # Q2 box
    fig.add_artist(FancyBboxPatch((0.5, 0.80), 0.21, 0.098, boxstyle="round,pad=0.004,rounding_size=0.01",
                                  transform=fig.transFigure, fc="#FFF2CC", ec=C["gold"], lw=2.2, zorder=2))
    tag(fig, 0.505, 0.892, "Q2  Which surface", C["gold"], 9)
    fig.text(0.605, 0.868, "observed target surface", ha="center", fontsize=10.5, fontweight="bold", zorder=5)
    fig.text(0.605, 0.838, r"$\min_m\ (m-s)^{\top}\Lambda(m-s)+\lambda\, m^{\top}Lm$", ha="center", fontsize=12.5, zorder=5)
    fig.text(0.605, 0.811, "keep the connected region with the largest seed mass", ha="center", fontsize=8.3,
             color="#555555", style="italic", zorder=5)
    arr(fig, (0.47, 0.868), (0.5, 0.862), color=C["gold"]); arr(fig, (0.47, 0.76), (0.52, 0.80), color="#777777")
    # memory line runs under Q2 to Q3
    fig.add_artist(Line2D([0.47, 0.485, 0.485, 0.722], [0.824, 0.824, 0.786, 0.786], transform=fig.transFigure, color=C["teal"], lw=1.6, zorder=3))
    arr(fig, (0.722, 0.786), (0.745, 0.845), color=C["teal"])
    blk(fig, 0.745, 0.845, 0.1, 0.05, "Q3  same object?\nswitch (B) + alert", C["tealfill"], ec=C["teal"], size=8.6, bold=True)
    arr(fig, (0.71, 0.87), (0.745, 0.87), color=C["gold"])
    blk(fig, 0.745, 0.748, 0.1, 0.05, "phase / grasp:\nHSR + gripper + 3D", C["state"], ec="#C55A11", size=8.6)
    fig.add_artist(Line2D([0.295, 0.31, 0.31, 0.728], [0.780, 0.780, 0.739, 0.739], transform=fig.transFigure, color="#C55A11", lw=1.6, zorder=3))
    arr(fig, (0.728, 0.739), (0.745, 0.76), color="#C55A11")
    arr(fig, (0.70, 0.80), (0.745, 0.785), color=C["gold"])
    blk(fig, 0.868, 0.79, 0.115, 0.06, "contact interface\nlink-specific field\n(not yet evaluated)", "#FFFFFF", ec="#7B6A9A", size=8.4, ls="--")
    arr(fig, (0.845, 0.865), (0.868, 0.835), color=C["teal"], ls="--"); arr(fig, (0.845, 0.773), (0.868, 0.805), color="#C55A11", ls="--")
    fig.add_artist(Line2D([0.295, 0.925], [0.912, 0.912], transform=fig.transFigure, color="#5B7FA6", lw=2))
    arr(fig, (0.925, 0.912), (0.925, 0.851), color="#5B7FA6")
    fig.text(0.62, 0.92, "nominal action chunk (unchanged policy)", ha="center", fontsize=9, color="#5B7FA6", fontweight="bold")
    fig.text(0.868, 0.772, "exception withheld on alert", fontsize=7.8, color="#555555", style="italic")

    # ---------------------------------------------------------------- three questions
    cols = [("Q1  Which object?", "the noun's attention picks seed patches", C["gold"], "#9A7300"),
            ("Q2  Which surface belongs to it?", "depth continuity, not a 3D radius, sets the boundary", C["gold"], "#9A7300"),
            ("Q3  Is it still the same object?", "internal features check identity over time", C["teal"], C["teal"])]
    W, gap, x0 = 0.322, 0.009, 0.008
    for k, (t, sub, colr, tc) in enumerate(cols):
        x = x0 + k * (W + gap)
        band(fig, x, 0.03, W, 0.687, "#FFFDF5" if k < 2 else "#F6FBFA", colr)
        fig.text(x + 0.01, 0.69, t, fontsize=13, fontweight="bold", color=tc)
        fig.text(x + 0.01, 0.668, sub, fontsize=9.5, style="italic", color="#555555")
    ph = 0.255

    def col(k, prior_title, priors, ours_title, ours, tc):
        x = x0 + k * (W + gap) + 0.008
        ww = W - 0.016
        fig.text(x, 0.645, prior_title, fontsize=10, fontweight="bold", color=C["red"])
        n = len(priors); pw = (ww - 0.006 * (n - 1)) / n
        for i, (a, badge, anc) in enumerate(priors):
            panel(fig, [x + i * (pw + 0.006), 0.375, pw, ph], a, badge=badge, bcol=C["red"], border=C["red"], anchor=anc)
        fig.text(x, 0.346, ours_title, fontsize=10, fontweight="bold", color=tc)
        n = len(ours); ow = (ww - 0.006 * (n - 1)) / n
        for i, (a, badge, cap, anc) in enumerate(ours):
            panel(fig, [x + i * (ow + 0.006), 0.088, ow, ph - 0.012], a, badge=badge, bcol=C["green"],
                  border=C["gold"] if k < 2 else C["teal"], lw=2.0, anchor=anc)
            fig.text(x + i * (ow + 0.006) + ow / 2, 0.076, cap, ha="center", va="top", fontsize=8.6, color="#444444")

    q1a, q1b = m["c1_prior_seeds_on_support"], m["c1_ours_seed_region"]
    col(0, "Prior: support-dominated attention patches",
        [(im("c1_prior_seeds_on_support", (0.0, 0.0, 1.0, 0.68)),
          f"✗ seeds on the basket floor → empty region (IoU {q1a['iou']:.2f})", (0.5, 0.5))],
        "AG3S: eligible noun-attention seeds",
        [(im("c1_ours_noun_attention"), None, "noun attention + seed patch", (0.5, 0.5)),
         (im("c1_ours_seed_region", (0.0, 0.0, 1.0, 0.68)), f"✓ IoU {q1b['iou']:.2f}", "seed → surface region (orange)", (0.6, 0.3))],
        "#9A7300")
    q2a, q2b, q2m = m["c2_prior_radius_cluster"], m["c2_ours_surface_region"], m["c2_prior_sphere_mask"]
    col(1, "Prior: 3 cm growth + DBSCAN, robot volume mask",
        [(im("c2_prior_radius_cluster", (0.1, 0.25, 0.9, 1.0)), f"✗ apple swallowed (IoU {q2a['iou']:.2f})", (0.5, 0.5)),
         (im("c2_prior_sphere_mask", (0.15, 0.1, 0.85, 0.9)), f"✗ held fruit erased ({q2m['render_nodes_absent_under_sphere']} nodes)", (0.5, 0.4))],
        "AG3S: depth-continuous surface + rendered robot mask",
        [(im("c2_ours_surface_region", (0.1, 0.25, 0.9, 1.0)), f"✓ IoU {q2b['iou']:.2f}", "neighbour stays protected (blue)", (0.5, 0.5)),
         (im("c2_ours_render_mask", (0.15, 0.1, 0.85, 0.9)), None, "fruit between the fingers kept", (0.5, 0.4))],
        "#9A7300")
    q3c, q3a = m["c3_ours_centered_cosine"], m["c3_ours_alert"]
    col(2, "Prior: raw feature similarity",
        [(im("c3_prior_raw_cosine", (0.0, 0.1, 1.0, 0.85)), "✗ every region ≈ 0.97: no identity signal", (0.5, 0.5))],
        "AG3S: centered memory → switch + alert",
        [(im("c3_ours_centered_cosine", (0.18, 0.2, 0.85, 0.9)), None, "target separates after centering", (0.5, 0.5)),
         (im("c3_ours_alert", (0.2, 0.25, 0.85, 0.85)), f"✓ alert {q3a['score']:.2f} < {q3a['threshold']:.2f}",
          "wrong pick (red) → exception withheld", (0.5, 0.5))],
        C["teal"])
    fig.text(0.012, 0.008, "MuJoCo replays of recorded development frames (32 episodes, all dev); overlays = 5 mm observed-surface nodes recomputed "
             "offline. Orange = selected target surface, red = wrongly selected or erased, blue = protected neighbour, gold = seeds. "
             "Contact interface not yet evaluated in closed loop.", fontsize=7.6, color="#666666")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, facecolor="white"); plt.close(fig)
    side = dict(figure=str(OUT.relative_to(REPO)), script="paper/tools/plotting/ifg_overview_compose.py",
                assets={k: dict(frame=v["frame"], **{q: v[q] for q in ("iou", "score", "threshold", "rule", "scores",
                                                                      "render_nodes_absent_under_sphere", "neighbor_fp") if q in v})
                        for k, v in m.items()},
                is_measured_result=False, scope="development frames only; illustrative examples chosen by fixed rules")
    OUT.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float) + "\n")
    print("wrote", OUT.relative_to(REPO))


if __name__ == "__main__":
    compose()
