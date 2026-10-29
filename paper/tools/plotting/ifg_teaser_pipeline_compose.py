"""Compose the IFG-revised Figure 1 teaser and Figure 3 pipeline from ifg_teaser_pipeline_render.py assets.

New files only (the paperbanana F1/F3 references are kept unchanged):
    paper/figures/ifg_overview/teaser_montage.png   (+ .json)
    paper/figures/ifg_overview/pipeline_montage.png (+ .json)

    .venv-ag3s/bin/python paper/tools/plotting/ifg_teaser_pipeline_compose.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ifg_overview_compose import A, C, REPO, arr, blk, band, im, meta, panel, tag  # noqa: E402

OUTDIR = REPO / "paper/figures/ifg_overview"
EP = json.loads((A / "tp_episode.json").read_text())
IOU = {k: v["iou_recomputed"] for k, v in EP["moments"].items()}


def title_tag(fig, x, y, text, fc="#333333", size=10.5):
    fig.text(x, y, text, fontsize=size, fontweight="bold", color="white", ha="left", va="center", zorder=7,
             bbox=dict(boxstyle="round,pad=0.25", fc=fc, ec="none", alpha=0.85))


def legend_rows(fig, x, y, items, dy=0.04, size=9.5):
    for i, (c, t) in enumerate(items):
        yy = y - i * dy
        fig.add_artist(FancyBboxPatch((x, yy - 0.008), 0.018, 0.016, boxstyle="square,pad=0", transform=fig.transFigure,
                                      fc=c, ec="none", zorder=6))
        fig.text(x + 0.024, yy, t, fontsize=size, va="center", zorder=6)


def trim(path, pad=30):
    from PIL import Image, ImageChops
    x = Image.open(path).convert("RGB")
    b = ImageChops.difference(x, Image.new("RGB", x.size, "white")).convert("L").point(lambda v: 255 if v > 8 else 0).getbbox()
    x.crop((max(b[0] - pad, 0), max(b[1] - pad, 0), min(b[2] + pad, x.width), min(b[3] + pad, x.height))).save(path)


# ------------------------------------------------------------------------------------------- teaser
def teaser():
    fig = plt.figure(figsize=(16, 7.6))
    fig.text(0.5, 0.955, "“Put the pear in the basket.”", ha="center", fontsize=16, style="italic")
    keys = [("M1_approach", "1  Approach"), ("M3_grasp", "2  Grasp"), ("M4_carry", "3  Carry"), ("M6_post_place", "4  Placed")]
    w, h, y0, gap = 0.229, 0.36, 0.53, 0.0163
    for i, (mom, lab) in enumerate(keys):
        x = 0.02 + i * (w + gap)
        panel(fig, [x, y0, w, h], im(f"t_key_{mom}", (0.04, 0.0, 0.96, 0.92)), border="#BBBBBB", lw=1.0)
        title_tag(fig, x + 0.008, y0 + h - 0.03, lab)
        fig.text(x + w - 0.008, y0 + 0.02, f"surface IoU {IOU[mom]:.2f}", ha="right", fontsize=8.5, color="white", zorder=7,
                 bbox=dict(boxstyle="round,pad=0.2", fc="#E8711A", ec="none"))
        if i < 3:
            arr(fig, (x + w + 0.002, y0 + h / 2), (x + w + gap - 0.002, y0 + h / 2), color="#444444", lw=2, ms=12)
    fig.text(0.02, 0.495, "Recorded rollout of the frozen VLA, re-rendered. Orange: observed target surface recomputed offline at each moment "
             "(development episode).", fontsize=9.5, color="#555555")
    # detail row
    w2, h2, y1 = 0.222, 0.40, 0.04
    det = [("t_noun_attention", "1  Identify: noun attention", "the policy's attention to “pear”;\ngold = seed patches", C["gold"]),
           ("t_surface", "2  Recover: observed surface", "depth continuity keeps the touching\norange separate (blue)", C["gold"]),
           ("t_permission", "3  Authorize: contact links only", "fingers (gold) may touch the pear;\nthe orange stays an obstacle", "#7B6A9A")]
    for i, (name, t, cap, bc) in enumerate(det):
        x = 0.02 + i * (w2 + 0.022)
        fig.text(x + w2 / 2, y1 + h2 + 0.012, t, ha="center", fontsize=11.5, fontweight="bold")
        crop = None if name == "t_noun_attention" else (0.12, 0.05, 0.88, 0.95)
        panel(fig, [x, y1 + 0.06, w2, h2 - 0.06], im(name, crop), border=bc, lw=2.0)
        fig.text(x + w2 / 2, y1 + 0.052, cap, ha="center", va="top", fontsize=8.8, color="#444444")
        if i < 2:
            arr(fig, (x + w2 + 0.003, y1 + 0.06 + (h2 - 0.06) / 2), (x + w2 + 0.019, y1 + 0.06 + (h2 - 0.06) / 2), color=C["gold"], lw=2.2, ms=13)
    fig.text(0.02 + 2 * (w2 + 0.022) + w2 - 0.004, y1 + 0.075, "configured rule ·\nnot yet closed-loop", ha="right", fontsize=7.8,
             color="white", zorder=7, bbox=dict(boxstyle="round,pad=0.25", fc="#7B6A9A", ec="none"))
    # what AG3S adds
    bx = 0.745
    band(fig, bx, y1, 0.235, h2 + 0.02, "#FAFAFA", "#CCCCCC", z=-1)
    fig.text(bx + 0.015, y1 + h2 - 0.03, "What AG3S adds", fontsize=13, fontweight="bold")
    pts = ["The frozen VLA's noun attention\npicks the target", "Depth continuity recovers its\nobserved surface",
           "Centered features check it is\nstill the same object", "Only contact links may touch\nthat surface; neighbours stay"]
    for i, t in enumerate(pts):
        fig.text(bx + 0.015, y1 + h2 - 0.085 - i * 0.062, "• " + t, fontsize=9.6, va="top")
    legend_rows(fig, bx + 0.015, y1 + 0.06, [("#E8711A", "target surface"), ("#2E75B6", "neighbour (protected)")], dy=0.032, size=9)
    legend_rows(fig, bx + 0.135, y1 + 0.06, [("#FFCC0D", "seeds / contact links"), ("#7B6A9A", "rule (downstream)")], dy=0.032, size=9)
    out = OUTDIR / "teaser_montage.png"
    fig.savefig(out, dpi=180, facecolor="white"); plt.close(fig)
    out.with_suffix(".json").write_text(json.dumps(dict(
        figure=str(out.relative_to(REPO)), script="paper/tools/plotting/ifg_teaser_pipeline_compose.py", episode=EP["episode"],
        surface_iou=IOU, assets={k: meta(k)["note"] for k in ("t_noun_attention", "t_surface", "t_permission")},
        is_measured_result=False, scope="one development episode; the permission panel illustrates the configured rule only"),
        indent=1, default=float) + "\n")
    print("wrote", out.relative_to(REPO))


# ------------------------------------------------------------------------------------------- pipeline
def pipeline():
    fig = plt.figure(figsize=(18, 10.2))
    cols = [(0.008, 0.17, "Frozen VLA", "#DEEBF7", "#5B7FA6"),
            (0.184, 0.19, "Observed surface graph", "#F2F2F2", "#888888"),
            (0.380, 0.19, "Seeds → membership → region", "#FFF2CC", C["gold"]),
            (0.576, 0.20, "Identity and task state", "#E3F2F0", C["teal"]),
            (0.782, 0.21, "Contact interface (downstream)", "#F3F0F7", "#7B6A9A")]
    yb, hb = 0.335, 0.6
    for x, w, t, fc, ec in cols:
        fig.add_artist(FancyBboxPatch((x, yb), w, hb, boxstyle="round,pad=0,rounding_size=0.008", transform=fig.transFigure,
                                      fc=fc, ec=ec, lw=1.6, ls="--", zorder=-1))
        fig.text(x + w / 2, yb + hb + 0.036, t, ha="center", fontsize=13, fontweight="bold", style="italic")
    # column 1
    tw = 0.051
    for i in range(3):
        panel(fig, [0.016 + i * (tw + 0.003), 0.82, tw, 0.075], im(f"p_in_{i}"), lw=1.0)
        panel(fig, [0.016 + i * (tw + 0.003), 0.465, tw, 0.075], im(f"p_att_{i}"), lw=1.0)
    fig.text(0.093, 0.802, "RGB (3 views), instruction, state", ha="center", fontsize=8.6)
    blk(fig, 0.02, 0.715, 0.146, 0.065, "VLA (frozen)\nserving pass", C["navy"], size=10.5, bold=True)
    arr(fig, (0.093, 0.715), (0.093, 0.69), color="#5B7FA6")
    blk(fig, 0.02, 0.6, 0.146, 0.088, "same pass exposes\nnoun attention (L10 H7)\nprefix-L8 features\nKV-L4 (sub-task probe)", "#FFFFFF", size=8.4)
    arr(fig, (0.093, 0.6), (0.093, 0.545), color="#5B7FA6")
    fig.text(0.093, 0.447, "noun attention, 3 × 16 × 16", ha="center", fontsize=8.6)
    blk(fig, 0.02, 0.35, 0.146, 0.06, "depth + camera poses\n+ robot model (FK)", C["grey"], size=8.6)
    # column 2
    panel(fig, [0.192, 0.69, 0.174, 0.2], im("p_robot_mask", (0.2, 0.55, 0.85, 1.0)), border="#888888")
    fig.text(0.279, 0.672, "wrist view while carrying: rendered robot mask\n(purple); sphere filter would also erase red", ha="center", va="top", fontsize=8.2)
    panel(fig, [0.192, 0.42, 0.174, 0.2], im("p_surface_nodes", (0.1, 0.05, 0.9, 0.95)), border="#888888")
    fig.text(0.279, 0.402, "5 mm nodes from 3 views; edges only\nacross depth-continuous neighbour pixels", ha="center", va="top", fontsize=8.2)
    arr(fig, (0.166, 0.38), (0.192, 0.44), color="#777777")
    # column 3
    panel(fig, [0.388, 0.69, 0.174, 0.2], im("p_membership", (0.1, 0.05, 0.9, 0.95)), border=C["gold"], lw=2)
    fig.text(0.475, 0.672, "eligible seeds (gold) → membership m", ha="center", va="top", fontsize=8.4)
    fig.text(0.475, 0.615, r"$(\Lambda+\lambda L)\,m=\Lambda s$", ha="center", fontsize=13)
    fig.text(0.475, 0.59, "support = negative evidence; no seed → unknown", ha="center", fontsize=7.8, color="#555555", style="italic")
    panel(fig, [0.388, 0.38, 0.174, 0.19], im("p_region", (0.1, 0.05, 0.9, 0.95)), border=C["gold"], lw=2)
    fig.text(0.475, 0.374, f"top-1 region (IoU {meta('p_region')['iou']:.2f}); touching orange stays blue",
             ha="center", va="top", fontsize=8.0)
    arr(fig, (0.366, 0.79), (0.388, 0.79), color=C["gold"], lw=2); arr(fig, (0.366, 0.52), (0.388, 0.52), color=C["gold"], lw=2)
    arr(fig, (0.17, 0.5), (0.192, 0.5), color=C["gold"], lw=2)   # attention seeds are lifted onto the surface nodes
    # column 4
    panel(fig, [0.584, 0.69, 0.184, 0.2], im("p_memory", (0.1, 0.1, 0.95, 0.95)), border=C["teal"], lw=2)
    fig.text(0.676, 0.672, "carry: centered cosine to the region\nremembered at approach", ha="center", va="top", fontsize=8.2)
    blk(fig, 0.588, 0.53, 0.176, 0.075, "switch region if margin ≥ 0.05  (B)\nalert if c ≤ 0.59 → withhold exception", C["tealfill"],
        ec=C["teal"], size=8.4, bold=True)
    blk(fig, 0.588, 0.415, 0.176, 0.085, "phase / grasp state\nLR on HSR + gripper\n+ hand distance, height, size", C["state"], ec="#C55A11", size=8.4)
    arr(fig, (0.562, 0.475), (0.588, 0.46), color=C["gold"]); arr(fig, (0.676, 0.64), (0.676, 0.605), color=C["teal"])
    arr(fig, (0.562, 0.8), (0.584, 0.8), color=C["gold"], lw=2)
    fig.add_artist(Line2D([0.166, 0.181, 0.181, 0.64], [0.64, 0.64, 0.322, 0.322], transform=fig.transFigure, color="#C55A11",
                          lw=1.5, zorder=1))
    arr(fig, (0.64, 0.322), (0.64, 0.413), color="#C55A11")
    # column 5
    blk(fig, 0.79, 0.565, 0.194, 0.085, "", "#FFFFFF", ec="#7B6A9A")
    fig.text(0.887, 0.625, r"$\delta_{i}=d(\cdot\,;\mathcal{S}\setminus\mathcal{R})$  if $i\in\mathcal{L}$, gate open", ha="center", fontsize=10.5, zorder=6)
    fig.text(0.887, 0.588, r"$\delta_{i}=d(\cdot\,;\mathcal{S})$  otherwise;   $\mathcal{R}\subseteq\mathcal{S}\cap\mathrm{supp}(\mathcal{T})$",
             ha="center", fontsize=10.5, zorder=6)
    panel(fig, [0.80, 0.69, 0.174, 0.2], im("t_permission", (0.12, 0.05, 0.88, 0.95)), border="#7B6A9A", lw=2)
    fig.text(0.887, 0.672, "configured rule: contact links (gold) only", ha="center", va="top", fontsize=8.2)
    blk(fig, 0.80, 0.43, 0.174, 0.06, "existing CCR (SQP)\n+ executed-prefix post-check", C["ctrl"], ec="#7B6A9A", size=8.6, ls="--")
    fig.text(0.835, 0.39, "execute", fontsize=10, fontweight="bold", color=C["green"])
    fig.text(0.925, 0.39, "hold", fontsize=10, fontweight="bold", color=C["red"])
    arr(fig, (0.887, 0.43), (0.887, 0.405), color="#7B6A9A")
    tag(fig, 0.792, 0.355, "not yet evaluated with this output", "#7B6A9A", 8.5)
    arr(fig, (0.764, 0.567), (0.79, 0.615), color=C["teal"], ls="--"); arr(fig, (0.764, 0.457), (0.79, 0.585), color="#C55A11", ls="--")
    arr(fig, (0.887, 0.565), (0.887, 0.49), color="#7B6A9A")
    # action lane
    fig.add_artist(Line2D([0.166, 0.181, 0.181, 0.987, 0.987], [0.748, 0.748, 0.953, 0.953, 0.46], transform=fig.transFigure,
                          color="#5B7FA6", lw=2, zorder=1))
    arr(fig, (0.987, 0.46), (0.974, 0.46), color="#5B7FA6")
    fig.text(0.58, 0.947, "nominal action chunk (policy unchanged)", ha="center", fontsize=9.5, color="#5B7FA6", fontweight="bold",
             bbox=dict(fc="white", ec="none", pad=1))
    # bottom: one episode
    fig.text(0.012, 0.25, "One episode,\nmoment by moment", fontsize=12, fontweight="bold")
    moms = ["M1_approach", "M2_pre_grasp", "M3_grasp", "M4_carry", "M5_pre_place", "M6_post_place", "M7_retreat"]
    sw = 0.066
    for i, mm in enumerate(moms):
        x = 0.12 + i * (sw + 0.004)
        panel(fig, [x, 0.17, sw, 0.125], im(f"p_strip_{mm}", (0.1, 0.0, 0.9, 1.0)), border=C["gold"], lw=1.2)
        fig.text(x + sw / 2, 0.157, f"{mm.split('_', 1)[1].replace('_', ' ')}  {IOU[mm]:.2f}", ha="center", fontsize=8)
    ax = fig.add_axes([0.12 + 7 * (sw + 0.004) + 0.01, 0.09, 0.39, 0.235])
    ax.imshow(im("p_phase_signals")); ax.axis("off")
    fig.text(0.12, 0.13, "Top: observed target surface (orange) and protected neighbours (blue) at the 7 recorded key moments; "
             "numbers = surface IoU.\nRight: inputs of the phase / grasp model over the same moments (background = annotated grasp state).",
             fontsize=8.6, color="#444444", va="top")
    out = OUTDIR / "pipeline_montage_v2.png"
    fig.savefig(out, dpi=170, facecolor="white"); plt.close(fig)
    trim(out)
    out.with_suffix(".json").write_text(json.dumps(dict(
        figure=str(out.relative_to(REPO)), script="paper/tools/plotting/ifg_teaser_pipeline_compose.py", episode=EP["episode"],
        surface_iou=IOU, memory_scores=meta("p_memory")["scores"], region_iou=meta("p_region")["iou"],
        robot_mask=dict((k, meta("p_robot_mask")[k]) for k in ("render_px", "sphere_only_px")),
        is_measured_result=False, scope="one development episode; contact interface and CCR are downstream and not evaluated with this output"),
        indent=1, default=float) + "\n")
    print("wrote", out.relative_to(REPO))


# ------------------------------------------------------------------------------------------- teaser v2
VA = REPO / "paper/figures/paperbanana/figure1/variants/assets"
R9 = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261015T181706+0900_30955b6ca8ef_d4b3bafb79ba/rounds/R9/paired_summary.json")


def imp(path, crop=None):
    import numpy as np
    from PIL import Image
    x = Image.open(path).convert("RGB")
    if crop:
        w, h = x.size
        x = x.crop((int(crop[0] * w), int(crop[1] * h), int(crop[2] * w), int(crop[3] * h)))
    return np.asarray(x)


def teaser_v2():
    """Figure-1 variant-A layout (failure vs success + how + measured box) for the IFG-revised method."""
    arms = json.loads(R9.read_text())["arms"]
    base, ours = arms["T54region+rm:N"], arms["top1-only-ns+rm:N"]
    fig = plt.figure(figsize=(16, 10.6))
    fig.text(0.012, 0.968, "AG3S: let a frozen VLA touch the object it names, and nothing beside it", fontsize=16, fontweight="bold")
    crop = (0.0, 0.06, 1.0, 0.86)
    x0, w, gap = 0.115, 0.318, 0.012
    r1, r2, h = 0.6, 0.29, 0.275
    fig.text(x0 + w / 2, 0.905, "Pick: obstacle next to the target", ha="center", fontsize=13, fontweight="bold")
    fig.text(x0 + w + gap + w / 2, 0.905, "Place: hurdle on the carry path", ha="center", fontsize=13, fontweight="bold")
    band(fig, 0.008, r1 - 0.012, 0.765, h + 0.024, "#FDECEA", "#E8A9A3")
    band(fig, 0.008, r2 - 0.012, 0.765, h + 0.024, "#FFF7DD", C["gold"])
    fig.text(0.058, r1 + h / 2, "VLA only", ha="center", va="center", fontsize=14, fontweight="bold", color=C["red"], rotation=90)
    fig.text(0.058, r2 + h / 2, "VLA + AG3S\n(ours)", ha="center", va="center", fontsize=14, fontweight="bold", color="#9A7300", rotation=90)
    panel(fig, [x0, r1, w, h], imp(VA / "vla_pick_fail.png", crop), badge="✗  hits the bar beside the apple → pick fails", bcol=C["red"], border=C["red"])
    panel(fig, [x0 + w + gap, r1, w, h], imp(VA / "vla_place_fail.png", crop), badge="✗  hits the hurdle → place fails", bcol=C["red"], border=C["red"])
    panel(fig, [x0, r2, w, h], imp(VA / "ag3s_pick_ok.png", crop), badge="✓  touches only the apple, grasps", bcol=C["green"], border=C["gold"], lw=2.2)
    panel(fig, [x0 + w + gap, r2, w, h], imp(VA / "ag3s_place_ok.png", crop), badge="✓  carries it over the hurdle, places", bcol=C["green"], border=C["gold"], lw=2.2)
    fig.text(0.115, r2 - 0.03, "Recorded closed-loop runs in one scene (same seed per column); red / gold line = recorded hand path. "
             "AG3S row: an earlier AG3S configuration of the same policy.", fontsize=8.6, color="#666666", style="italic")
    # how strip
    fig.text(0.012, 0.248, "How it works", fontsize=13, fontweight="bold")
    hw, hh, hy = 0.172, 0.165, 0.02
    items = [("t_noun_attention", None, "1  Noun attention → seeds", "which object (the policy's own attention)"),
             ("t_surface", (0.12, 0.05, 0.88, 0.95), "2  Depth → target surface", "where it ends; touching neighbour stays blue"),
             ("p_memory", (0.1, 0.1, 0.95, 0.95), "3  Features → same object?", "identity check over time (switch / alert)"),
             ("t_permission", (0.12, 0.05, 0.88, 0.95), "4  Contact links only", "fingers may touch it; other links may not")]
    for k, (name, cr, t, cap) in enumerate(items):
        x = 0.012 + k * (hw + 0.022)
        fig.text(x + hw / 2, hy + hh + 0.035, t, ha="center", fontsize=10.5, fontweight="bold")
        panel(fig, [x, hy + 0.012, hw, hh + 0.012], im(name, cr), border=C["teal"] if k == 2 else ("#7B6A9A" if k == 3 else C["gold"]), lw=2.0)
        fig.text(x + hw / 2, hy + hh + 0.014, "", ha="center")
        fig.text(x + hw / 2, hy - 0.005, cap, ha="center", va="top", fontsize=8.4, color="#444444")
        if k < 3:
            arr(fig, (x + hw + 0.002, hy + 0.012 + (hh + 0.012) / 2), (x + hw + 0.02, hy + 0.012 + (hh + 0.012) / 2), color=C["gold"], lw=2.2, ms=13)
    # measured box
    band(fig, 0.79, 0.275, 0.20, 0.655, "#FAFAFA", "#CCCCCC")
    fig.text(0.89, 0.902, "Measured so far", ha="center", fontsize=13, fontweight="bold")
    fig.text(0.89, 0.878, "target-surface grounding, same robot mask", ha="center", fontsize=8.8, color="#555555", style="italic")
    ax = fig.add_axes([0.825, 0.42, 0.15, 0.37])
    labels = ["IoU", "precision"]
    import numpy as np
    xx = np.arange(2)
    bv = [base["iou"], base["precision"]]; ov = [ours["iou"], ours["precision"]]
    ax.barh(xx + 0.2, bv, 0.36, color="#9E9E9E", label="radius + clustering")
    ax.barh(xx - 0.2, ov, 0.36, color=C["gold"], label="AG3S surface graph")
    for yv, v in zip(xx + 0.2, bv):
        ax.text(v + 0.01, yv, f"{v:.3f}", va="center", fontsize=9.5)
    for yv, v in zip(xx - 0.2, ov):
        ax.text(v + 0.01, yv, f"{v:.3f}", va="center", fontsize=9.5, fontweight="bold")
    ax.set_yticks(xx); ax.set_yticklabels(labels, fontsize=10); ax.invert_yaxis()
    ax.set_xlim(0.6, 1.08); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(axis="x", labelsize=8)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=1)
    fig.text(0.89, 0.305, "development set: 180 frames / 32 episodes\nepisode means, offline\nclosed-loop safety: not yet measured",
             ha="center", fontsize=8.6, color="#555555")
    fig.text(0.89, 0.235, "frozen VLA · no extra training\nno external segmenter", ha="center", fontsize=10.5, style="italic", color="#444444")
    out = OUTDIR / "teaser_montage_v2.png"
    fig.savefig(out, dpi=170, facecolor="white"); plt.close(fig)
    trim(out)
    out.with_suffix(".json").write_text(json.dumps(dict(
        figure=str(out.relative_to(REPO)), script="paper/tools/plotting/ifg_teaser_pipeline_compose.py",
        layout="paperbanana figure1 variant A (failure vs success, how strip, measured box)",
        scenes={k: json.loads((VA / f"{k}.json").read_text()).get("run") for k in ("vla_pick_fail", "vla_place_fail", "ag3s_pick_ok", "ag3s_place_ok")},
        scene_note="T43 ep1982 overbar paired runs: E0 (VLA only) vs V11AB (earlier AG3S configuration); not configuration B",
        measured=dict(source=str(R9), radius_clustering=base, ag3s_surface_graph=ours),
        how_episode=EP["episode"], is_measured_result=True), indent=1, default=float) + "\n")
    print("wrote", out.relative_to(REPO))


# ------------------------------------------------------------------------------------------- Figure 1 at paper width
TEXTWIDTH_IN = 6.875   # cvpr.sty \textwidth (two-column, letter)


def teaser_fig1(height_in=3.2):
    """Variant-A teaser laid out natively at the paper text width (no down-scaling; 6-8 pt text)."""
    import numpy as np
    W, H = TEXTWIDTH_IN, height_in
    fig = plt.figure(figsize=(W, H))
    F = lambda x, y, w, h: [x / W, y / H, w / W, h / H]

    def pnl(rect_in, a, border, lw=1.0, badge=None, bcol=None, bsize=5.8, anchor=(0.5, 0.5)):
        ax = fig.add_axes(F(*rect_in), zorder=3)
        ax.imshow(fit_(a, rect_in[2] / rect_in[3], anchor), aspect="auto", interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(border); sp.set_linewidth(lw)
        if badge:
            ax.text(0.02, 0.04, badge, transform=ax.transAxes, fontsize=bsize, fontweight="bold", color="white", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.22", fc=bcol, ec="none"))

    def bnd(x, y, w, h, fc, ec, lw=0.8):
        fig.add_artist(FancyBboxPatch((x / W, y / H), w / W, h / H, boxstyle="round,pad=0,rounding_size=0.006",
                                      transform=fig.transFigure, fc=fc, ec=ec, lw=lw, zorder=-1))

    def arw(a, b, color, lw=1.2, ms=7):
        from matplotlib.patches import FancyArrowPatch
        fig.add_artist(FancyArrowPatch((a[0] / W, a[1] / H), (b[0] / W, b[1] / H), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, zorder=4))

    crop = (0.0, 0.06, 1.0, 0.86)
    # ---- scenes
    lx, x0, pw, gap = 0.0, 0.25, 2.07, 0.05
    ph, y2, y1 = 0.98, 0.935, 1.985
    fig.text((x0 + pw / 2) / W, 3.075 / H, "Pick: obstacle next to the target", ha="center", fontsize=7.5, fontweight="bold")
    fig.text((x0 + pw + gap + pw / 2) / W, 3.075 / H, "Place: hurdle on the carry path", ha="center", fontsize=7.5, fontweight="bold")
    bnd(lx, y1 - 0.03, x0 + 2 * pw + gap + 0.03, ph + 0.06, "#FDECEA", "#E8A9A3")
    bnd(lx, y2 - 0.03, x0 + 2 * pw + gap + 0.03, ph + 0.06, "#FFF7DD", C["gold"])
    fig.text(0.12 / W, (y1 + ph / 2) / H, "VLA only", ha="center", va="center", rotation=90, fontsize=7.5, fontweight="bold", color=C["red"])
    fig.text(0.12 / W, (y2 + ph / 2) / H, "VLA + AG3S", ha="center", va="center", rotation=90, fontsize=7.5, fontweight="bold", color="#9A7300")
    pnl((x0, y1, pw, ph), imp(VA / "vla_pick_fail.png", crop), C["red"], badge="✗ hits the bar beside the apple → pick fails", bcol=C["red"], anchor=(0.6, 0.5))
    pnl((x0 + pw + gap, y1, pw, ph), imp(VA / "vla_place_fail.png", crop), C["red"], badge="✗ hits the hurdle → place fails", bcol=C["red"], anchor=(0.6, 0.5))
    pnl((x0, y2, pw, ph), imp(VA / "ag3s_pick_ok.png", crop), C["gold"], lw=1.4, badge="✓ touches only the apple, grasps", bcol=C["green"], anchor=(0.6, 0.5))
    pnl((x0 + pw + gap, y2, pw, ph), imp(VA / "ag3s_place_ok.png", crop), C["gold"], lw=1.4, badge="✓ carries it over the hurdle, places", bcol=C["green"], anchor=(0.6, 0.5))
    # ---- measured box
    arms = json.loads(R9.read_text())["arms"]
    base, ours = arms["T54region+rm:N"], arms["top1-only-ns+rm:N"]
    bx, bw = 4.52, W - 4.52 - 0.01
    bnd(bx, y2 - 0.03, bw, 2 * ph + 0.18, "#FAFAFA", "#CCCCCC")
    fig.text((bx + bw / 2) / W, 2.93 / H, "Measured so far", ha="center", fontsize=7.8, fontweight="bold")
    fig.text((bx + bw / 2) / W, 2.81 / H, "target-surface grounding, same robot mask", ha="center", fontsize=5.8, style="italic", color="#555555")
    ax = fig.add_axes(F(bx + 0.42, 1.36, bw - 0.62, 1.0))
    xx = np.arange(2)
    bv, ov = [base["iou"], base["precision"]], [ours["iou"], ours["precision"]]
    ax.barh(xx + 0.2, bv, 0.36, color="#9E9E9E", label="radius + clustering")
    ax.barh(xx - 0.2, ov, 0.36, color=C["gold"], label="AG3S surface graph")
    for yv, v in zip(xx + 0.2, bv):
        ax.text(v + 0.008, yv, f"{v:.3f}", va="center", fontsize=6)
    for yv, v in zip(xx - 0.2, ov):
        ax.text(v + 0.008, yv, f"{v:.3f}", va="center", fontsize=6, fontweight="bold")
    ax.set_yticks(xx); ax.set_yticklabels(["IoU", "precision"], fontsize=6.5); ax.invert_yaxis()
    ax.set_xlim(0.6, 1.07); ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=5.5, length=2); ax.tick_params(axis="y", length=2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    ax.legend(frameon=False, fontsize=5.8, loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=2, handlelength=1.0, columnspacing=0.8)
    fig.text((bx + bw / 2) / W, 1.17 / H, "dev set: 180 frames / 32 episodes, offline\nclosed-loop safety: not yet measured",
             ha="center", va="top", fontsize=5.6, color="#555555")
    fig.text((bx + bw / 2) / W, 0.985 / H, "frozen VLA · no extra training · no segmenter", ha="center", va="top", fontsize=5.8,
             style="italic", color="#333333")
    # ---- how it works
    items = [("t_noun_attention", (0.0, 0.3, 1.0, 1.0), "1  Noun attention → seeds", C["gold"]),
             ("t_surface", (0.08, 0.15, 0.92, 0.95), "2  Depth → target surface", C["gold"]),
             ("p_memory", (0.22, 0.18, 1.0, 0.72), "3  Features → same object?", C["teal"]),
             ("t_permission", (0.08, 0.15, 0.92, 0.95), "4  Contact links only", "#7B6A9A")]
    hw, hh, hy, ag = 1.53, 0.6, 0.0, 0.16
    hx0 = (W - (4 * hw + 3 * ag)) / 2
    for k, (name, cr, t, bc) in enumerate(items):
        x = hx0 + k * (hw + ag)
        fig.text((x + hw / 2) / W, (hy + hh + 0.07) / H, t, ha="center", fontsize=6.6, fontweight="bold")
        pnl((x, hy + 0.02, hw, hh), im(name, cr), bc, lw=1.2)
        if k < 3:
            arw((x + hw + 0.02, hy + 0.02 + hh / 2), (x + hw + ag - 0.02, hy + 0.02 + hh / 2), C["gold"], lw=1.4, ms=8)
    outp = OUTDIR / "teaser_fig1_textwidth.pdf"
    fig.savefig(outp, facecolor="white"); fig.savefig(outp.with_suffix(".png"), dpi=300, facecolor="white"); plt.close(fig)
    outp.with_suffix(".json").write_text(json.dumps(dict(
        figure=str(outp.relative_to(REPO)), png=str(outp.with_suffix(".png").relative_to(REPO)),
        size_in=[W, H], textwidth_source="paper/latex/cvpr.sty \\textwidth=6.875in", min_font_pt=5.5,
        scenes={k: json.loads((VA / f"{k}.json").read_text()).get("run") for k in ("vla_pick_fail", "vla_place_fail", "ag3s_pick_ok", "ag3s_place_ok")},
        scene_note="T43 ep1982 overbar paired runs: E0 (VLA only) vs V11AB (earlier AG3S configuration); not configuration B",
        measured=dict(source=str(R9), radius_clustering=base, ag3s_surface_graph=ours), how_episode=EP["episode"]),
        indent=1, default=float) + "\n")
    print("wrote", outp.relative_to(REPO))


def fit_(a, asp, anchor=(0.5, 0.5)):
    h, w = a.shape[:2]
    if w / h > asp:
        nw = int(h * asp); x0 = int((w - nw) * anchor[0]); return a[:, x0:x0 + nw]
    nh = int(w / asp); y0 = int((h - nh) * anchor[1]); return a[y0:y0 + nh]


if __name__ == "__main__":
    import sys as _s
    which = _s.argv[1:] or ["teaser_v2", "pipeline"]
    for w in which:
        globals()[w]()
