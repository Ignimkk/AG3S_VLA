"""Real-scene assets for the IFG-revised Figure 1 (teaser) and Figure 3 (pipeline).

One development episode (T53 ep1842, "Put the pear in the basket.", an orange touches the pear; the same
episode as the paperbanana F1/F3 references) is replayed in MuJoCo at its 7 key moments. Overlays are the
5 mm observed-surface nodes recomputed with the IFG study code (read-only import): noun-attention seeds,
graph-energy membership, the top-1 region, robot masks, centered feature agreement. Nothing is
re-simulated and the contact-permission panel only visualises the configured rule (not evaluated).

    MUJOCO_GL=osmesa PYTHONPATH=. .venv-ag3s/bin/python paper/tools/plotting/ifg_teaser_pipeline_render.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.cm as mcm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ifg_overview_render as O  # noqa: E402

XF, M, R = O.XF, O.M, O.R
EP = "T53_ep1842"
MOM = {"M1_approach": "T53_ep1842_f010", "M2_pre_grasp": "T53_ep1842_f016", "M3_grasp": "T53_ep1842_f020",
       "M4_carry": "T53_ep1842_f025", "M5_pre_place": "T53_ep1842_f031", "M6_post_place": "T53_ep1842_f032",
       "M7_retreat": "T53_ep1842_f037"}
WIDE = {"lookat": [0.56, -0.14, 1.06], "dist": 1.28, "az": 200, "el": -24}   # paperbanana F1 viewpoint
HANDS = {"R": ("EE_BODY_R", "ee_finger_r1", "ee_finger_r2"), "L": ("EE_BODY_L", "ee_finger_l1", "ee_finger_l2")}


class Ep(O.Replay):
    """One replay scene re-posed at each key moment of the episode."""

    def __init__(self):
        super().__init__(MOM["M1_approach"])
        self.base = self.m.geom_rgba.copy()

    def pose(self, fid):
        with np.load(O.RUN / "s0/replay" / f"{fid}.npz") as z:
            self.d.qpos[:] = z["qpos"]
        mujoco.mj_forward(self.m, self.d)
        self.m.geom_rgba[:] = self.base

    def body(self, n):
        return self.d.xpos[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)].copy()

    def tint(self, names, rgb, mix=0.75):
        ids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
        g = np.isin(self.m.geom_bodyid, [i for i in ids if i >= 0])
        self.m.geom_rgba[g, :3] = (1 - mix) * self.base[g, :3] + mix * np.asarray(rgb[:3])

    def cam_image(self, name):
        return self.render(None, mjcam=name)


def grounding(fr):
    s = XF.seed_and_g(fr, True)
    return s, O.top1(fr, s)


def heat(img, att, alpha_max=0.75):
    H, W = img.shape[:2]
    a = np.asarray(Image.fromarray((att / att.max() * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC), float) / 255
    al = np.clip(a * 1.3, 0, alpha_max)[..., None]
    return ((img / 255 * (1 - al) + mcm.get_cmap("inferno")(a)[..., :3] * al) * 255).astype(np.uint8)


def run():
    ep = Ep()
    frames, side = {}, {}
    rows = XF.rows_of(O.RUN, "rounds/R6", "top1-only-ns+rm:N")
    for mom, fid in MOM.items():
        fr = XF.Frame(O.RUN, fid, mask="render", feats=("prefix_L8",) if mom in ("M1_approach", "M4_carry") else ())
        s, reg = grounding(fr)
        frames[mom] = (fr, s, reg)
        side[mom] = dict(frame=fid, iou_recomputed=fr.iou(reg), iou_row=rows[fid].get("iou"))
    # active hand: the one closest to the target at grasp
    fr3 = frames["M3_grasp"][0]; ep.pose(MOM["M3_grasp"])
    ctr3 = fr3.geo.pos[fr3.gt].mean(0)
    hand = min(HANDS, key=lambda h: np.linalg.norm(ep.body(HANDS[h][0]) - ctr3))

    # ---- teaser: wide keyframes with the grounded target surface
    for mom in ("M1_approach", "M3_grasp", "M4_carry", "M6_post_place"):
        fr, s, reg = frames[mom]; ep.pose(MOM[mom])
        img = ep.render(WIDE, [(fr.geo.pos[reg], O.ORANGE, O.HALF * 1.3)])
        O.save(f"t_key_{mom}", img, MOM[mom], "Wide replay of the recorded state; orange = recomputed top-1 target surface (5 mm nodes).",
               dict(episode=EP, moment=mom, iou=side[mom]["iou_recomputed"], camera=WIDE))

    # ---- pre-grasp close-up frame used by most detail panels
    fr, s, reg = frames["M2_pre_grasp"]; fid = MOM["M2_pre_grasp"]; ep.pose(fid)
    ctr = fr.geo.pos[fr.gt].mean(0)
    near = O.neighbours(fr, ctr, 0.14) & ~reg
    close = O.cam_at(ctr, 0.42, 150, -34)
    # noun attention on the camera whose seed patches hold most target pixels
    cm = fr.class_map
    per = np.zeros(3, int)
    for p in s["six"]:
        cc, rr, col = int(p) // 256, (int(p) % 256) // 16, int(p) % 16
        h, w = cm.shape[1] // 16, cm.shape[2] // 16
        per[cc] += int((cm[cc, rr * h:(rr + 1) * h, col * w:(col + 1) * w] == 0).sum())
    c = int(per.argmax())
    ys, xs = np.nonzero(cm[c] == 0)
    O.save_attention("t_noun_attention", ep.cam_image(ep.cams[c]), fr.fin.maps["N"][c * 256:(c + 1) * 256].reshape(16, 16),
                     s["six"], c, fid, dict(episode=EP, moment="M2_pre_grasp", camera=ep.cams[c], target_px_in_seed_patches=per.tolist()),
                     crop=(xs.mean() / cm.shape[2], ys.mean() / cm.shape[1]), half=0.3)
    O.save("t_surface", ep.render(close, [(fr.geo.pos[near], O.BLUE, O.HALF)] + O.eval_layers(fr, reg)), fid,
           "Observed target surface (orange) and the touching neighbour's observed surface (blue).",
           dict(episode=EP, moment="M2_pre_grasp", iou=fr.iou(reg), neighbour_nodes=int(near.sum()), camera=close))
    ep.tint(HANDS[hand], O.GOLD[:3])
    O.save("t_permission", ep.render(close, [(fr.geo.pos[near], O.BLUE, O.HALF), (fr.geo.pos[reg], (0.91, 0.44, 0.10, 0.35), O.HALF)]), fid,
           "Configured contact rule, visualised: only the active-hand links (gold) may enter the target surface (faint orange); "
           "the neighbour (blue) stays an obstacle for every link. Not evaluated in closed loop.",
           dict(episode=EP, moment="M2_pre_grasp", active_hand=hand, camera=close, is_rule_illustration=True))
    ep.m.geom_rgba[:] = ep.base

    # ---- pipeline assets
    for ci, cn in enumerate(ep.cams):
        img = ep.cam_image(cn)
        Image.fromarray(img).save(O.OUT / f"p_in_{ci}.png")
        Image.fromarray(heat(img, fr.fin.maps["N"][ci * 256:(ci + 1) * 256].reshape(16, 16))).save(O.OUT / f"p_att_{ci}.png")
    (O.OUT / "p_inputs.json").write_text(json.dumps(dict(frame=fid, cameras=ep.cams, map="N (L10H7)",
                                                         note="p_in_* = re-rendered policy cameras, p_att_* = with noun attention"), indent=1))
    # all observed surface nodes in the workspace (support light, others grey)
    win = np.linalg.norm(fr.geo.pos - ctr[None], axis=1) <= 0.30
    O.save("p_surface_nodes", ep.render(O.cam_at(ctr, 0.62, 150, -38), [(fr.geo.pos[win & fr.geo.support], (0.55, 0.72, 0.88, 1.0), O.HALF),
                                                                         (fr.geo.pos[win & ~fr.geo.support], (0.45, 0.45, 0.45, 1.0), O.HALF)]),
           fid, "5 mm observed-surface nodes within 30 cm of the target: support plane (light blue) and other surfaces (grey).",
           dict(episode=EP, nodes=int(win.sum()), support_nodes=int((win & fr.geo.support).sum())))
    # membership m and seeds
    m = s["m"]; roi = m >= 0
    cmap = mcm.get_cmap("Reds")
    roi = roi & ~fr.geo.support   # support nodes are protected scene; not drawn here
    lay = [(fr.geo.pos[roi & (m < 0.05)], (0.6, 0.6, 0.6, 1.0), O.HALF)]
    for lo, hi in ((0.05, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        k = roi & (m >= lo) & (m < hi)
        lay.append((fr.geo.pos[k], (*cmap(0.25 + 0.75 * (lo + hi) / 2)[:3], 1.0), O.HALF))
    seeds = (s["sl"] > 0) & ~fr.geo.support
    lay.append((fr.geo.pos[seeds], O.GOLD, O.HALF * 1.15))
    O.save("p_membership", ep.render(close, lay), fid, "Graph-energy membership m of non-support nodes in the 15 cm ROI (grey < 0.05, pale -> dark red ~ 1); gold = seed-mass nodes.",
           dict(episode=EP, roi_nodes=int(roi.sum()), seed_nodes=int(seeds.sum())))
    O.save("p_region", ep.render(close, [(fr.geo.pos[near], O.BLUE, O.HALF)] + O.eval_layers(fr, reg)), fid,
           "Top-1 seed-mass region (orange); neighbour surfaces (blue) stay protected.", dict(episode=EP, iou=fr.iou(reg)))
    # rendered vs sphere robot mask on the active wrist camera while carrying
    fid4 = MOM["M4_carry"]; ep.pose(fid4)
    cw = {"R": 2, "L": 1}[hand]
    img = ep.cam_image(ep.cams[cw]).astype(float)
    with np.load(O.RUN / "s0/replay" / f"{fid4}.npz") as z:
        fk = z["robot_mask"][cw]
    with np.load(O.RUN / "rounds/R6/robot_mask" / f"{fid4}.npz") as z:
        rm = z["robot_mask"][cw]
    H, W = img.shape[:2]
    up = lambda a: np.asarray(Image.fromarray(a.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 0
    fk, rm = up(fk), up(rm)
    out = img.copy()
    out[rm] = 0.35 * img[rm] + 0.65 * np.array([140, 90, 200])
    extra = fk & ~rm
    out[extra] = 0.15 * img[extra] + 0.85 * np.array([220, 0, 0])
    O.save("p_robot_mask", out.astype(np.uint8), fid4, "Active wrist camera while carrying: purple = rendered robot-surface mask; "
           "red = pixels the sphere/capsule volume filter removes in addition (incl. the held fruit).",
           dict(episode=EP, moment="M4_carry", camera=ep.cams[cw], render_px=int(rm.sum()), sphere_only_px=int(extra.sum())))
    # centered memory: first-frame choice vs regions while carrying
    fm, fc = frames["M1_approach"][0], frames["M4_carry"][0]
    sc = O.region_scores(fm, fc, True)
    ctr4 = fc.geo.pos[fc.gt].mean(0); cam4 = O.cam_at(ctr4 + np.array([0.0, 0.05, -0.08]), 0.80, 150, -38)
    cm2 = mcm.get_cmap("viridis"); lay, labels = [], []
    for v, k, cl in sc:
        lay.append((fc.geo.pos[k], (*cm2(float(np.clip((v + 0.45) / 1.1, 0, 1)))[:3], 1.0), O.HALF))
        labels.append((fc.geo.pos[k].mean(0), f"{O.NAMES.get(cl, cl)} {v:.2f}"))
    O.save("p_memory", O.label(ep.render(cam4, lay), ep, cam4, labels[:3]), fid4,
           "Centered prefix-L8 cosine of candidate regions (M4 carry) to the region remembered at M1 approach.",
           dict(episode=EP, memory_frame=MOM["M1_approach"], scores=[(round(v, 3), cl) for v, _, cl in sc[:5]]))
    # moment strip
    for mom, fid_ in MOM.items():
        f_, s_, r_ = frames[mom]; ep.pose(fid_)
        c_ = f_.geo.pos[f_.gt].mean(0)
        nb = O.neighbours(f_, c_, 0.12) & ~r_
        O.save(f"p_strip_{mom}", ep.render(O.cam_at(c_, 0.40, 150, -30), [(f_.geo.pos[nb], O.BLUE, O.HALF)] + O.eval_layers(f_, r_)),
               fid_, "Key-moment close-up with the recomputed target surface.", dict(episode=EP, moment=mom, iou=side[mom]["iou_recomputed"]))
    ep.close()
    # phase signals of the episode (R12 features)
    F = [json.loads(l) for l in open(O.RUN / "rounds/R12/features.jsonl")]
    F = sorted([r for r in F if r["episode_id"] == EP], key=lambda r: r["moment"])
    fig, ax = plt.subplots(figsize=(8.0, 2.4))
    x = np.arange(len(F))
    bands = {"pre": "#DEEBF7", "held": "#E4D3F0", "placed": "#E2F0D9"}
    for xi, r in zip(x, F):
        ax.axvspan(xi - 0.5, xi + 0.5, color=bands.get(r["grasp_state"], "white"), lw=0)
    ax.plot(x, [r["hand_d"] for r in F], "o-", color="#2E75B6", lw=2, label="target-hand distance (m)")
    ax.plot(x, [r["height"] - 0.8 for r in F], "s-", color="#548235", lw=2, label="target height - 0.8 m")
    ax.plot(x, [1 - r["grip"] for r in F], "^-", color="#C99A06", lw=2, label="gripper closing")
    ax.plot(x, [r["hsr"][1] for r in F], "d--", color="#C55A11", lw=2, label="HSR p(place)")
    ax.set_xticks(x); ax.set_xticklabels([r["moment"].split("_", 1)[1].replace("_", " ") for r in F], fontsize=10)
    ax.set_ylim(-0.05, 1.05); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.42))
    for xi, r in zip(x, F):
        ax.text(xi, -0.0, r["grasp_state"], ha="center", va="bottom", fontsize=7.5, color="#555555")
    fig.tight_layout(); fig.savefig(O.OUT / "p_phase_signals.png", dpi=220); plt.close(fig)
    (O.OUT / "p_phase_signals.json").write_text(json.dumps(dict(episode=EP, rows=[{k: r[k] for k in ("frame_id", "moment", "grasp_state", "hand_d", "height", "grip", "hsr")} for r in F],
                                                                note="grasp_state from recorded annotation (background bands)"), indent=1))
    (O.OUT / "tp_episode.json").write_text(json.dumps(dict(episode=EP, moments=side, active_hand=hand), indent=1, default=float))
    print("done", side)


if __name__ == "__main__":
    run()
