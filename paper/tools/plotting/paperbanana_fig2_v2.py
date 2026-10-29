"""Figure 2 v2 draft: contribution-centred overview (What / Who / When) + compact framework.

Top: one frozen VLA emits actions, attention and hidden state; they meet in a single link-specific
query rule (the contact representation) used by CCR. Bottom: three contribution axes, each "prior vs AG3S"
on recorded data.
  What  -- external segmenter mask (GT labels of T53 ep1842 stand in for an ideal segmenter) vs the
           frozen VLA's own attention lifted to a 3D cluster (T53 ep1842).
  Who   -- global ball exclusion vs boundary-preserving, link-specific queries (T53 ep1842 renders).
  When  -- fixed role vs phase-aware role: target -> attached object -> released (T43 V11AB ep1982).

    MUJOCO_GL=osmesa .venv-ag3s/bin/python paper/tools/plotting/paperbanana_fig2_v2.py
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
from matplotlib.lines import Line2D  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paperbanana_fig2_overview as f2  # noqa: E402
import paperbanana_render as pr  # noqa: E402

OUT = f2.OUT
A = f2.A
F2 = pr.OUT / "figure2/assets"
F1 = pr.OUT / "figure1/assets"
C = f2.C
RED = (0.85, 0.12, 0.10, 1.0)


def save(img, name, note, extra=None):
    return f2.save(img, name, note, extra)


# ------------------------------------------------------------------ renders
def seg_mask_render():
    """T53 ep1842: GT target mask (ideal external segmenter) on the re-rendered active wrist camera."""
    run, sc = pr.Run(), pr.Scene(1280, 960)
    j = pr.CH["approach"]
    sc.set(run.step(j)["qpos"])
    rgb = sc.render("wrist_cam_r").astype(float) / 255.0
    lab = run.gt["label_small"][j, 2]                     # cam_right_wrist, class index 0 = target
    m = np.asarray(Image.fromarray((lab == 0).astype(np.uint8) * 255).resize((1280, 960), Image.NEAREST)) > 0
    col = np.array([0.20, 0.75, 0.35])
    out = rgb.copy()
    out[m] = 0.45 * rgb[m] + 0.55 * col
    edge = m & ~(np.asarray(Image.fromarray(m.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(7))) > 0)
    out[edge] = col
    p = save((out * 255).astype(np.uint8), "v2_segmenter_mask",
             "T53 ep1842 chunk 10: right-wrist camera re-rendered from the recorded state; the GT target "
             "label (gt/ep1842.npz label_small, class 0) stands in for an ideal external segmenter mask.",
             {"chunk": j, "episode": "T53 C1a ep1842"})
    return p


def role_renders():
    sc, cs = f2.Scene(), f2.chunks()
    made = {}
    hb = mujoco.mj_name2id(sc.m, mujoco.mjtObj.mjOBJ_BODY, "EE_BODY_L")
    ca = f2.dump(cs, f2.CH["attach"])
    att = ca["summary"]["grasp"]["attached"]["held"]["spheres"]
    sc.at(f2.CH["attach"]); R0, p0 = sc.d.xmat[hb].reshape(3, 3).copy(), sc.d.xpos[hb].copy()
    local = [(R0.T @ (np.asarray(s_["centre_m"]) - p0), s_["radius_mm"] / 1000) for s_ in att]

    jc = 22
    sc.at(jc); R1, p1 = sc.d.xmat[hb].reshape(3, 3).copy(), sc.d.xpos[hb].copy()
    cam = {"lookat": (p1 + np.array([0.0, -0.05, -0.08])).tolist(), "dist": 0.62, "az": 150, "el": -16}
    sc.tint(f2.fs.GRIPPER, f2.GOLD, mix=0.5)
    ex = [("sphere", p1 + R1 @ lc, r, f2.PURPLE) for lc, r in local]
    made["attached"] = save(sc.render(cam, ex), "v2_role_attached",
                            "Carry (chunk 22, latch HELD): the apple is an attached sphere on the hand (purple), so it is "
                            "checked against the bar like any robot link.", {"chunk": jc})
    sc.at(jc)
    apple = sc.body("apple")
    ex = [("sphere", apple, 0.055, (1.0, 0.1, 0.05, 0.22)), ("sphere", apple, 0.040, (1.0, 0.1, 0.05, 0.25))]
    made["attached_prior"] = save(sc.render(cam, ex), "v2_role_prior",
                                  "Same carry state; red glow marks a held object that a fixed-role filter does not model "
                                  "(illustration on the recorded state).", {"chunk": jc})

    jr = 34
    c = f2.dump(cs, jr); sc.at(jr)
    o, vs = np.asarray(c["esdf_origin"], float), float(c["esdf_voxel_size"])
    main = np.asarray(c["esdf_distance"], np.float32)
    apple = sc.body("apple")
    ii, pts = f2.surface_voxels(main, o, vs, apple, np.array([0.05, 0.05, 0.05]), z_min=0.80)
    sc.fade(("apple",), 0.25)
    ex = [("sphere", p, vs * 0.62, (0.18, 0.46, 0.71, 0.9)) for p in pts]
    camr = {"lookat": apple.tolist(), "dist": 0.42, "az": 150, "el": -50}
    made["released"] = save(sc.render(camr, ex), "v2_role_released",
                            "After placing (chunk 34, HSR = home, exclusion released): the apple in the basket is an "
                            "obstacle again in the recorded scene ESDF (blue voxels).",
                            {"chunk": jr, "subtask": c["summary"]["subtask"]["label"]})
    made["target"] = A / "ex_field.png"
    return made


# ------------------------------------------------------------------ compose helpers
im, panel, blk, arr, band = f2.im, f2.panel, f2.blk, f2.arr, f2.band


def tag(fig, x, y, text, fc, size=9.5):
    fig.text(x, y, text, fontsize=size, fontweight="bold", color="white", ha="left", va="center", zorder=6,
             bbox=dict(boxstyle="round,pad=0.3", fc=fc, ec="none"))


def compose(seg, roles):
    fig = plt.figure(figsize=(17, 11.2))
    # ---------------- framework strip
    fig.text(0.012, 0.972, "AG3S: one frozen VLA decides what may be touched, by which link, and when",
             fontsize=14.5, fontweight="bold", color=C["ink"])
    band(fig, 0.008, 0.735, 0.984, 0.215, "#FBFBFB", "#CCCCCC")
    y0 = 0.80
    blk(fig, 0.018, y0, 0.075, 0.07, "RGB-D\n+ instruction", C["grey"], size=9.5)
    blk(fig, 0.115, y0 - 0.005, 0.09, 0.08, "frozen VLA\n(no retraining)", C["navy"], size=10.5, bold=True)
    arr(fig, (0.093, y0 + 0.035), (0.115, y0 + 0.035))
    # three outputs
    outs = [(0.905, "actions", C["navy"], "#5B7FA6"), (0.835, "attention", C["goldfill"], C["gold"]),
            (0.765, "hidden state", C["state"], "#C55A11")]
    for yy, t, fc, ec in outs:
        blk(fig, 0.23, yy - 0.012, 0.075, 0.034, t, fc, ec=ec, size=9)
        arr(fig, (0.205, y0 + 0.035), (0.23, yy + 0.005), color=ec)
    # C1 what, C3 when
    blk(fig, 0.33, 0.815, 0.135, 0.05, "3D target from attention\n(no segmenter)", C["goldfill"], ec=C["gold"], size=9.5, bold=True)
    blk(fig, 0.33, 0.748, 0.135, 0.05, "object role: target →\nattached → released", C["state"], ec="#C55A11", size=9.5, bold=True)
    tag(fig, 0.333, 0.876, "C1  What", C["gold"]); tag(fig, 0.333, 0.735 + 0.0005, "C3  When", "#C55A11")
    arr(fig, (0.305, 0.84), (0.33, 0.84), color=C["gold"]); arr(fig, (0.305, 0.77), (0.33, 0.773), color="#C55A11")
    # C2 who: the query rule
    fig.add_artist(FancyBboxPatch((0.49, 0.752), 0.30, 0.13, boxstyle="round,pad=0.004,rounding_size=0.01",
                                  transform=fig.transFigure, fc="#FFF2CC", ec=C["gold"], lw=2.2, zorder=2))
    tag(fig, 0.495, 0.876, "C2  Who", C["gold"])
    fig.text(0.64, 0.858, "link-specific contact representation", ha="center", fontsize=11, fontweight="bold", zorder=5)
    fig.text(0.64, 0.812, r"$d_\ell(\mathbf{x}) = d^{\,\mathrm{free}}(\mathbf{x})\ \ \mathrm{if}\ \ \ell \in \mathcal{L}_{\mathrm{contact}}(\phi)$",
             ha="center", fontsize=14, zorder=5)
    fig.text(0.64, 0.777, r"$d_\ell(\mathbf{x}) = d^{\,\mathrm{scene}}(\mathbf{x})\ \ \mathrm{otherwise}$", ha="center", fontsize=14, zorder=5)
    fig.text(0.64, 0.757, "free = scene minus the target's own observed extent", ha="center", fontsize=8.8,
             color="#555555", style="italic", zorder=5)
    arr(fig, (0.465, 0.84), (0.49, 0.83), color=C["gold"]); arr(fig, (0.465, 0.773), (0.49, 0.785), color="#C55A11")
    blk(fig, 0.815, 0.79, 0.08, 0.055, "CCR\n(SQP)", C["ctrl"], ec="#7B6A9A", size=10.5, bold=True)
    blk(fig, 0.912, 0.79, 0.07, 0.055, "execute\nor hold", "#FFFFFF", ec="#7B6A9A", size=9.5)
    arr(fig, (0.79, 0.817), (0.815, 0.817), color=C["gold"]); arr(fig, (0.895, 0.817), (0.912, 0.817), color="#7B6A9A")
    # actions lane to CCR
    fig.add_artist(Line2D([0.305, 0.855], [0.905, 0.905], transform=fig.transFigure, color="#5B7FA6", lw=2))
    arr(fig, (0.855, 0.905), (0.855, 0.845), color="#5B7FA6")
    fig.text(0.58, 0.913, "nominal action chunk", ha="center", fontsize=9, color="#5B7FA6", fontweight="bold")

    # ---------------- three contribution axes
    cols = [("C1  What to touch", "where the target's geometry comes from", C["gold"]),
            ("C2  Who may touch it", "which surfaces are freed, and for which links", C["gold"]),
            ("C3  When it changes", "the object's role follows the task phase", "#C55A11")]
    W, gap, x0 = 0.322, 0.009, 0.008
    for k, (t, sub, colr) in enumerate(cols):
        x = x0 + k * (W + gap)
        band(fig, x, 0.012, W, 0.705, "#FFFDF5", colr)
        fig.text(x + 0.01, 0.69, t, fontsize=13, fontweight="bold", color="#9A7300" if k < 2 else "#A0450F")
        fig.text(x + 0.01, 0.668, sub, fontsize=9.5, style="italic", color="#555555")
    ph, pw = 0.255, 0.15
    # row labels
    def pair(k, prior_img, prior_badge, ours_imgs, ours_badge, prior_title, ours_title, anchors=None):
        x = x0 + k * (W + gap) + 0.008
        fig.text(x, 0.645, prior_title, fontsize=10, fontweight="bold", color=C["red"])
        panel(fig, [x, 0.375, W - 0.016, ph], prior_img, badge=prior_badge, bcol=C["red"], border=C["red"])
        fig.text(x, 0.345, ours_title, fontsize=10, fontweight="bold", color="#9A7300" if k < 2 else "#A0450F")
        n = len(ours_imgs)
        ww = (W - 0.016 - 0.006 * (n - 1)) / n
        for i, (img, cap) in enumerate(ours_imgs):
            panel(fig, [x + i * (ww + 0.006), 0.07, ww, ph], img, border=C["gold"], lw=2.0)
            if cap:
                fig.text(x + i * (ww + 0.006) + ww / 2, 0.055, cap, ha="center", va="top", fontsize=8.6, color="#444444")
        if ours_badge:
            tag(fig, x + 0.004, 0.083, ours_badge, C["green"], size=9)

    crop2 = (0.13, 0.02, 0.92, 0.86)
    pair(0, im(seg, (0.12, 0.40, 0.92, 1.0)), "✗ extra model + class prompt",
         [(im(F1 / "f1_wrist_attention.png", (0.10, 0.20, 0.95, 0.98)), "frozen-VLA attention"),
          (im(F1 / "f1_grounding_scene.png", (0.12, 0.08, 0.88, 0.82)), "lifted 3D target")],
         None, "Prior: external segmenter", "AG3S: the policy's own attention → 3D")
    pair(1, im(F2 / "f2b_ball_render.png", crop2), "✗ freed for every link; neighbour erased",
         [(im(F2 / "f2c_ownership_render.png", crop2), "only target-owned voxels"),
          (im(F2 / "f2c_link_render.png", (0.10, 0.0, 0.90, 0.85)), "only the grasping links")],
         None, "Prior: global target exclusion", "AG3S: boundary-preserving + link-specific")
    pair(2, im(roles["attached_prior"]), "✗ held object not modelled",
         [(im(roles["target"]), "target: fingers may touch"),
          (im(roles["attached"]), "attached: avoids the bar"),
          (im(roles["released"]), "released: obstacle again")],
         None, "Prior: fixed object role", "AG3S: HSR + grasp state switch the role")
    out = OUT / "reference_montage_v2.png"
    fig.savefig(out, dpi=170, facecolor="white"); plt.close(fig)
    print("wrote", out.relative_to(f2.REPO))


if __name__ == "__main__":
    compose(seg_mask_render(), role_renders())
