"""Compose the PaperBanana reference montages (F1-F3) from the replay renders.

Run after paperbanana_render.py:
    MUJOCO_GL=osmesa .venv-ag3s/bin/python paper/tools/plotting/paperbanana_compose.py

Also draws two small charts for F3 from recorded data (TCP path of the CCR chunk, per-chunk state
timeline). Each montage is a layout draft for PaperBanana's single reference image; every raster in it
is an asset with its own JSON sidecar.
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paperbanana_render as pr  # noqa: E402

OUT = pr.OUT
C = {"target": "#E8711A", "neigh": "#2E75B6", "gold": "#D9A406", "grey": "#7F7F7F", "purple": "#7030A0",
     "red": "#C00000", "green": "#2E7D32", "ink": "#1F1F1F", "navy": "#DEEBF7", "goldfill": "#FFF2CC",
     "field": "#DAE8FC", "state": "#FBE5D6", "ctrl": "#E4DFEC"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def img(fig_no, name, crop=None):
    im = Image.open(OUT / f"figure{fig_no}/assets/{name}.png").convert("RGB")
    if crop:
        w, h = im.size
        im = im.crop((int(crop[0] * w), int(crop[1] * h), int(crop[2] * w), int(crop[3] * h)))
    return np.asarray(im)


def show(ax, a, title=None, tag=None, tagcol="#1F1F1F", border=None):
    ax.imshow(a)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(border is not None)
        if border:
            s.set_edgecolor(border); s.set_linewidth(2.2)
    if title:
        ax.set_title(title, fontsize=11.5, fontweight="bold", color=C["ink"], pad=5)
    if tag:
        ax.text(0.03, 0.95, tag, transform=ax.transAxes, fontsize=10.5, fontweight="bold", color="white",
                va="top", ha="left", bbox=dict(boxstyle="round,pad=0.3", fc=tagcol, ec="none", alpha=0.92))


def arrow(fig, p0, p1, color="#404040", lw=2.0, ms=16, style="-|>", ls="-"):
    fig.add_artist(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle=style, mutation_scale=ms,
                                   color=color, lw=lw, linestyle=ls))


def chip(fig, x, y, color, text, kind="dot"):
    if kind == "dot":
        fig.add_artist(Circle((x, y), 0.006, transform=fig.transFigure, color=color))
    else:
        fig.add_artist(FancyBboxPatch((x - 0.008, y - 0.007), 0.016, 0.014, boxstyle="round,pad=0.002",
                                      transform=fig.transFigure, fc=color, ec="none"))
    fig.text(x + 0.012, y, text, va="center", fontsize=10, color=C["ink"])


def save(fig, fig_no):
    p = OUT / f"figure{fig_no}/reference_montage.png"
    fig.savefig(p, dpi=170, facecolor="white")
    plt.close(fig)
    print("wrote", p.relative_to(pr.REPO))


# ------------------------------------------------------------------ F1
def montage1():
    fig = plt.figure(figsize=(16, 7.6))
    fig.text(0.5, 0.962, "“Put the pear in the basket.”", ha="center", va="center", fontsize=15,
             style="italic", color=C["ink"])
    keys = [("approach", "Approach"), ("grasp", "Grasp"), ("carry", "Carry"), ("release", "Release")]
    w, gap, x0, y0, h = 0.229, 0.0163, 0.02, 0.505, 0.425
    for k, (name, lab) in enumerate(keys):
        ax = fig.add_axes([x0 + k * (w + gap), y0, w, h])
        show(ax, img(1, f"f1_key_{name}", (0.08, 0.02, 0.92, 0.80)), tag=f"{k + 1}  {lab}", tagcol="#333333")
        if k < 3:
            arrow(fig, (x0 + (k + 1) * w + k * gap + 0.001, y0 + h / 2), (x0 + (k + 1) * (w + gap) - 0.001, y0 + h / 2))
    fig.text(x0, y0 - 0.022, "Closed-loop execution of a frozen VLA with AG3S (recorded run, re-rendered)",
             fontsize=9.5, color="#555555", va="center")

    panels = [("f1_wrist_attention", (0.10, 0.20, 0.95, 0.98), "VLA attention on the wrist view"),
              ("f1_grounding_scene", (0.12, 0.08, 0.88, 0.82), "3D target grounding"),
              ("f1_contact_scene", (0.10, 0.04, 0.86, 0.80), "Contact only on the target")]
    w2, h2, y2 = 0.222, 0.385, 0.03
    xs = [0.02, 0.265, 0.51]
    for (name, crop, title), x in zip(panels, xs):
        ax = fig.add_axes([x, y2, w2, h2])
        show(ax, img(1, name, crop), title=title, border="#BBBBBB")
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(fig, (a + w2 + 0.003, y2 + h2 / 2), (b - 0.003, y2 + h2 / 2), color=C["gold"], lw=2.4)

    bx, by, bw, bh = 0.752, 0.05, 0.228, 0.39
    fig.add_artist(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.006", transform=fig.transFigure,
                                  fc="#FAFAFA", ec="#CCCCCC", zorder=-1))
    fig.text(bx + 0.012, by + bh - 0.03, "What AG3S adds", fontsize=12.5, fontweight="bold", color=C["ink"], va="top")
    lines = ["Frozen VLA attention picks the target", "Depth gives its observed 3D extent",
             "Only contact links may touch that extent", "The touching neighbour stays an obstacle"]
    for i, t in enumerate(lines):
        fig.text(bx + 0.012, by + bh - 0.085 - i * 0.05, "• " + t, fontsize=10.5, color=C["ink"], va="top")
    chip(fig, bx + 0.02, by + 0.07, C["target"], "target object", kind="box")
    chip(fig, bx + 0.125, by + 0.07, C["neigh"], "neighbour", kind="box")
    chip(fig, bx + 0.02, by + 0.03, C["gold"], "contact links", kind="box")
    chip(fig, bx + 0.125, by + 0.03, "#E05A1A", "attention", kind="box")
    save(fig, 1)


# ------------------------------------------------------------------ F2
def montage2():
    fig = plt.figure(figsize=(16, 5.2))
    fig.text(0.5, 0.95, "Finding the target is not enough: its boundary and the querying link decide contact",
             ha="center", va="center", fontsize=14, fontweight="bold", color=C["ink"])
    cols = [("f2a_scene_render", (0.13, 0.02, 0.92, 0.86), "(a) All surfaces as obstacles",
             "grasp blocked: target repels the fingers", C["red"], "#BBBBBB"),
            ("f2b_ball_render", (0.13, 0.02, 0.92, 0.86), "(b) Global target exclusion (ball)",
             "neighbour surface erased for every link", C["red"], "#BBBBBB"),
            ("f2c_ownership_render", (0.13, 0.02, 0.92, 0.86), "(c) Ours: boundary-preserving extent",
             "only target-owned voxels removed", C["green"], C["gold"]),
            ("f2c_link_render", (0.10, 0.0, 0.90, 0.85), "(d) Ours: link-specific queries",
             "active hand: target-free · other hand: scene", C["green"], C["gold"])]
    w, gap, x0, y0, h = 0.236, 0.0127, 0.0155, 0.20, 0.64
    for k, (name, crop, title, verdict, vc, bc) in enumerate(cols):
        x = x0 + k * (w + gap)
        ax = fig.add_axes([x, y0, w, h])
        show(ax, img(2, name, crop), title=title, border=bc)
        fig.text(x + w / 2, y0 - 0.045, verdict, ha="center", va="center", fontsize=10.5, color=vc, fontweight="bold")
    y = 0.05
    chip(fig, 0.05, y, C["neigh"], "obstacle voxels (scene field)", kind="box")
    chip(fig, 0.235, y, "#E58A8A", "removed by the ball", kind="box")
    chip(fig, 0.385, y, "#F2B48A", "removed: target-owned", kind="box")
    chip(fig, 0.545, y, C["gold"], "contact links (active hand)", kind="box")
    chip(fig, 0.725, y, "#9E9E9E", "other links", kind="box")
    fig.text(0.985, y, "same recorded frame · 1 cm voxels", va="center", ha="right", fontsize=9, color="#666666")
    save(fig, 2)


# ------------------------------------------------------------------ F3 charts
def charts3():
    run, sc = pr.Run(), pr.Scene(320, 240)
    meta = json.loads((OUT / "figure3/assets/f3_ccr_paths.json").read_text())
    j = meta["chunk"]
    z = np.load(run.chunks[j]); q = run.step(j)["qpos"]
    ref = pr.ee_path(sc, q, np.asarray(z["reference_chunk"], float)[:16])
    refd = pr.ee_path(sc, q, np.asarray(z["refined_chunk"], float)[:16])
    c = run.chunk(j)
    tgt = np.asarray(c["target_points"], float)
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.scatter(tgt[:, 0] * 100, tgt[:, 2] * 100, s=10, c=C["target"], alpha=0.35, lw=0, label="target points")
    ax.plot(refd[:, 0] * 100, refd[:, 2] * 100, "-", color=C["gold"], lw=3.0, label="CCR refined", zorder=2)
    ax.plot(ref[:, 0] * 100, ref[:, 2] * 100, "--", color="#222222", lw=2.0, label="policy reference", zorder=3)
    ax.scatter(refd[:8, 0] * 100, refd[:8, 2] * 100, s=34, c=C["gold"], ec="white", lw=0.6, zorder=4,
               label="executed rows")
    ax.set_xlabel("x (cm)"); ax.set_ylabel("z (cm)")
    ax.set_aspect("equal"); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, framealpha=0.9, fontsize=8.5, loc="lower right")
    ax.set_title("Tool-centre path (side view)", fontsize=11.5)
    p = OUT / "figure3/assets/f3_ccr_tcp.png"
    fig.savefig(p, dpi=220, bbox_inches="tight"); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps({
        "asset": str(p.relative_to(pr.REPO)), "chunk": j, "run": "T53 C1a", "episode": pr.KEY,
        "note": "FK (site right_ee) of the recorded reference and refined chunks, 16 planned rows; target points of the chunk.",
        "max_tcp_shift_mm": meta["max_tcp_shift_mm"], "is_measured_result": False}, indent=1) + "\n")

    states, p_sub, src, hold = [], [], [], []
    for jj in range(len(run.chunks)):
        s = json.loads(str(np.load(run.chunks[jj])["summary_json"]))
        states.append(s.get("grasp", {}).get("state"))
        sp = s.get("subtask", {}).get("p") or {}
        p_sub.append([sp.get("pick", np.nan), sp.get("place", np.nan), sp.get("home", np.nan)])
        src.append(s.get("exclusion", {}).get("source"))
        hold.append(any(r.get("action") == "hold" for r in s.get("verdict", {}).get("reasons", [])))
    p_sub = np.asarray(p_sub, float)
    n = len(states)
    cols = {"searching": "#F2F2F2", "latched": "#DEEBF7", "closing": "#FFE699", "held": "#E4D3F0", "placed": "#E2F0D9"}
    fig, axs = plt.subplots(2, 1, figsize=(9, 2.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.5], "hspace": 0.12})
    for jj, sname in enumerate(states):
        axs[0].axvspan(jj - 0.5, jj + 0.5, color=cols.get(sname, "white"), lw=0)
    for sname in cols:
        idx = [jj for jj, v in enumerate(states) if v == sname]
        if idx:
            axs[0].text(np.mean(idx), 0.5, sname, ha="center", va="center", fontsize=9)
    hj = [jj for jj, h in enumerate(hold) if h]
    axs[0].scatter(hj, [0.12] * len(hj), marker="v", c=C["red"], s=30, zorder=3)
    axs[0].set_ylim(0, 1)
    axs[0].set_yticks([]); axs[0].set_ylabel("grasp\nstate", rotation=0, ha="right", va="center", fontsize=9)
    for k, (lab, col) in enumerate(zip(["pick", "place", "home"], ["#5B9BD5", "#ED7D31", "#70AD47"])):
        axs[1].plot(np.arange(n), p_sub[:, k], color=col, lw=1.8, label=lab)
    axs[1].set_ylabel("sub-task\nprobe", rotation=0, ha="right", va="center", fontsize=9)
    axs[1].set_yticks([0, 1]); axs[1].legend(frameon=False, ncol=3, fontsize=8, loc="center right")
    axs[1].set_xlabel("action chunk (533 ms nominal each)", fontsize=9)
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    p = OUT / "figure3/assets/f3_timeline.png"
    fig.savefig(p, dpi=220, bbox_inches="tight"); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps({
        "asset": str(p.relative_to(pr.REPO)), "run": "T53 C1a", "episode": pr.KEY, "n_chunks": n,
        "states": states, "hold_chunks": hj, "exclusion_source": src,
        "note": "Recorded per-chunk grasp-latch state, sub-task probe probabilities and HOLD chunks (red).",
        "is_measured_result": False}, indent=1) + "\n")


# ------------------------------------------------------------------ F3 montage
def box(fig, x, y, w, h, fc, ec="#555555", lw=1.2, ls="-", z=1):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.006",
                                  transform=fig.transFigure, fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=z))


def label(fig, x, y, t, size=10, bold=False, color=None, ha="center", va="center", style="normal"):
    fig.text(x, y, t, fontsize=size, fontweight="bold" if bold else "normal", color=color or C["ink"],
             ha=ha, va=va, style=style, zorder=5)


def thumb(fig, rect, fig_no, name, crop=None, border="#BBBBBB"):
    ax = fig.add_axes(rect, zorder=3)
    show(ax, img(fig_no, name, crop), border=border)
    return ax


def elbow(fig, pts, color="#404040", lw=2.0, ls="-"):
    """Orthogonal polyline through `pts` (figure coords) with an arrowhead on the last segment."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        fig.add_artist(Line2D([a[0], b[0]], [a[1], b[1]], transform=fig.transFigure, color=color, lw=lw, ls=ls,
                              zorder=4, solid_capstyle="round"))
    fig.add_artist(FancyArrowPatch(pts[-2], pts[-1], transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=15, color=color, lw=lw, linestyle=ls, zorder=4))


def montage3():
    fig = plt.figure(figsize=(18, 9.0))
    S = [(0.008, 0.162), (0.178, 0.192), (0.378, 0.150), (0.536, 0.250), (0.794, 0.198)]
    titles = ["Frozen VLA", "Multi-view 3D grounding", "Boundary-preserving extent",
              "Contact-aware geometry", "Constrained chunk refinement"]
    fills = [(C["navy"], "#5B7FA6"), (C["goldfill"], C["gold"]), (C["goldfill"], C["gold"]),
             ("#EEF4FB", "#5B7FA6"), (C["ctrl"], "#7B6A9A")]
    top, bottom = 0.905, 0.315
    for (x, w), t, (fc, ec) in zip(S, titles, fills):
        box(fig, x, bottom, w, top - bottom, fc, ec=ec, lw=1.6, ls="--", z=-2)
        label(fig, x + w / 2, top + 0.025, t, size=13, bold=True, style="italic")
    cx = [x + w / 2 for x, w in S]
    LANE = 0.875

    # 1 frozen VLA
    for k, n in enumerate(["head", "left_wrist", "right_wrist"]):
        thumb(fig, [0.016 + k * 0.0505, 0.745, 0.0485, 0.098], 3, f"f3_in_{n}")
    label(fig, cx[0], 0.728, "RGB-D (3 views), instruction, state", size=9)
    box(fig, 0.020, 0.640, 0.138, 0.065, "#C9DCF0", z=2)
    label(fig, cx[0], 0.683, "VLA (frozen)", size=11.5, bold=True)
    label(fig, cx[0], 0.657, "serving pass", size=9)
    box(fig, 0.020, 0.555, 0.138, 0.058, "#E6EEF7", z=2)
    label(fig, cx[0], 0.594, "extraction pass", size=10.5, bold=True)
    label(fig, cx[0], 0.570, "same checkpoint", size=9)
    for k, n in enumerate(["head", "left_wrist", "right_wrist"]):
        thumb(fig, [0.016 + k * 0.0505, 0.432, 0.0485, 0.098], 3, f"f3_att_{n}")
    label(fig, cx[0], 0.415, "attention, 3 × 16 × 16", size=9)
    box(fig, 0.020, 0.330, 0.138, 0.060, C["state"], ec="#C55A11", z=2)
    label(fig, cx[0], 0.369, "sub-task probe", size=10.5, bold=True)
    label(fig, cx[0], 0.346, "pick / place / home", size=9)
    elbow(fig, [(cx[0], 0.640), (cx[0], 0.613)], color="#5B7FA6")
    elbow(fig, [(cx[0], 0.555), (cx[0], 0.532)], color="#5B7FA6")
    elbow(fig, [(cx[0], 0.405), (cx[0], 0.390)], color="#C55A11")

    # 2 grounding
    thumb(fig, [0.186, 0.615, 0.176, 0.235], 3, "f3_lifted_attention", (0.05, 0.08, 0.95, 0.92))
    label(fig, cx[1], 0.598, "lift + fuse, attention seeds", size=9.5)
    thumb(fig, [0.186, 0.355, 0.176, 0.215], 3, "f3_target_cluster", (0.05, 0.08, 0.95, 0.92))
    label(fig, cx[1], 0.338, "grow → filter → confirm (3 frames)", size=9.5)
    elbow(fig, [(cx[1], 0.587), (cx[1], 0.572)], color=C["gold"])
    elbow(fig, [(0.166, 0.481), (0.172, 0.481), (0.172, 0.73), (0.186, 0.73)])          # attention -> lifting
    elbow(fig, [(0.158, 0.360), (0.176, 0.360), (0.176, 0.40), (0.186, 0.40)], color="#C55A11")  # probe -> confirm

    # 3 extent
    thumb(fig, [0.385, 0.560, 0.136, 0.29], 2, "f2c_ownership_render", (0.15, 0.05, 0.90, 0.85), border=C["gold"])
    label(fig, cx[2], 0.538, "ball = outer bound;", size=9.5)
    label(fig, cx[2], 0.515, "keep neighbour-owned voxels", size=9.5)
    box(fig, 0.392, 0.385, 0.122, 0.075, "#FFE699", ec=C["gold"], z=2)
    label(fig, cx[2], 0.432, "target-free", size=11, bold=True)
    label(fig, cx[2], 0.407, "fine layer", size=11, bold=True)
    elbow(fig, [(0.362, 0.46), (0.372, 0.46), (0.372, 0.70), (0.385, 0.70)], color=C["gold"])
    elbow(fig, [(cx[2], 0.502), (cx[2], 0.460)], color=C["gold"])

    # 4 contact-aware geometry
    thumb(fig, [0.544, 0.665, 0.115, 0.185], 3, "f3_scene_field", (0.12, 0.05, 0.92, 0.90))
    label(fig, 0.6015, 0.648, "cuRobo TSDF → ESDF", size=9)
    thumb(fig, [0.665, 0.665, 0.115, 0.185], 2, "f2c_link_render", (0.10, 0.0, 0.90, 0.85), border=C["gold"])
    label(fig, 0.7225, 0.648, "query spheres per link", size=9)
    box(fig, 0.548, 0.535, 0.228, 0.075, "#FFE699", ec=C["gold"], z=2)
    label(fig, cx[3], 0.582, "link-specific allowed collision", size=11.5, bold=True)
    label(fig, cx[3], 0.553, "contact links → target-free  ·  others → scene field", size=9)
    box(fig, 0.548, 0.330, 0.110, 0.150, C["state"], ec="#C55A11", z=2)
    label(fig, 0.603, 0.455, "grasp state", size=10.5, bold=True)
    label(fig, 0.603, 0.428, "searching → latched", size=8.8)
    label(fig, 0.603, 0.408, "→ closing → held", size=8.8)
    label(fig, 0.603, 0.388, "→ placed", size=8.8)
    label(fig, 0.603, 0.355, "executed gripper feedback", size=8.3, color="#555555")
    thumb(fig, [0.666, 0.330, 0.110, 0.150], 3, "f3_attached_object", (0.12, 0.02, 0.88, 0.98))
    label(fig, 0.721, 0.318, "held → attached object", size=9, color=C["purple"])
    elbow(fig, [(0.6015, 0.638), (0.6015, 0.610)], color="#5B7FA6")
    elbow(fig, [(0.7225, 0.638), (0.7225, 0.610)], color=C["gold"])
    elbow(fig, [(0.603, 0.480), (0.603, 0.535)], color="#C55A11")
    elbow(fig, [(0.721, 0.480), (0.721, 0.535)], color=C["purple"])
    elbow(fig, [(0.514, 0.42), (0.530, 0.42), (0.530, 0.572), (0.548, 0.572)], color=C["gold"])

    # 5 CCR
    box(fig, 0.806, 0.735, 0.174, 0.085, "#D5CCE6", ec="#7B6A9A", z=2)
    label(fig, cx[4], 0.792, "executed-horizon SQP", size=11, bold=True)
    label(fig, cx[4], 0.762, "d(x) − r ≥ m  ·  8 executed + 8 look-ahead", size=8.8)
    thumb(fig, [0.812, 0.455, 0.162, 0.255], 3, "f3_ccr_tcp", border=None)
    box(fig, 0.835, 0.345, 0.116, 0.055, "#FFFFFF", ec="#7B6A9A", z=2)
    label(fig, cx[4], 0.372, "post-check gate", size=10.5, bold=True)
    label(fig, 0.845, 0.325, "execute", size=10, bold=True, color=C["green"])
    label(fig, 0.94, 0.325, "hold", size=10, bold=True, color=C["red"])
    elbow(fig, [(cx[4], 0.735), (cx[4], 0.712)], color="#7B6A9A")
    elbow(fig, [(cx[4], 0.455), (cx[4], 0.400)], color="#7B6A9A")
    elbow(fig, [(0.776, 0.572), (0.790, 0.572), (0.790, 0.777), (0.806, 0.777)], color=C["gold"])
    # nominal chunk lane
    elbow(fig, [(0.158, 0.672), (0.168, 0.672), (0.168, LANE), (cx[4], LANE), (cx[4], 0.820)], color="#5B7FA6", lw=2.4)
    fig.text(0.47, LANE, "nominal action chunk (50 × 16)", fontsize=10, fontweight="bold", color="#5B7FA6",
             ha="center", va="center", zorder=6, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))
    # feedback
    elbow(fig, [(0.835, 0.36), (0.790, 0.36), (0.790, 0.40), (0.658, 0.40)], color="#888888", lw=1.6, ls="--")
    label(fig, 0.792, 0.338, "feedback", size=8.5, color="#666666", style="italic", ha="left")

    ax = fig.add_axes([0.10, 0.015, 0.80, 0.27])
    show(ax, np.asarray(Image.open(OUT / "figure3/assets/f3_timeline.png").convert("RGB")))
    label(fig, 0.012, 0.15, "One episode,\nchunk by chunk", size=10.5, bold=True, ha="left")
    save(fig, 3)


def main():
    montage1()
    montage2()
    charts3()
    montage3()


if __name__ == "__main__":
    main()
