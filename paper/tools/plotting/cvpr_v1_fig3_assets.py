"""Re-rendered IFG panels for CVPR v1 (fourth pass): the selected target reads as a whole object.

Same frames, cameras and 5 mm observed-surface node overlays as paper/figures/ifg_overview/assets
(ifg_overview_render.py / ifg_teaser_pipeline_render.py, imported read-only), with two presentation changes:
 (1) 3-D panels: the target object's mesh is tinted orange (paper target colour) and the touching neighbour fruit's
     mesh light blue underneath the nodes, so the selected object reads as a whole; the nodes remain the observed
     surface the method actually uses (orange = selected target surface, blue = protected neighbour surface).
 (2) wrist-view robot-mask panel: every observed target pixel (label class 'target' outside the rendered robot mask)
     is coloured orange and the rendered robot mask purple (the whole target reads as one object; no partial overlay).
Nothing is re-simulated or re-measured; the region / membership / IoU values come from the same study code.

    MUJOCO_GL=osmesa PYTHONPATH=. .venv-ag3s/bin/python paper/tools/plotting/cvpr_v1_fig3_assets.py
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.cm as mcm  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ifg_overview_render as O  # noqa: E402

XF, RUN = O.XF, O.RUN
REPO = O.REPO
OUT = REPO / "paper/figures/cvpr_v1/assets"
OLD = REPO / "paper/figures/ifg_overview/assets"
HANDS = {"R": ("EE_BODY_R", "ee_finger_r1", "ee_finger_r2"), "L": ("EE_BODY_L", "ee_finger_l1", "ee_finger_l2")}
FRUITS = ("apple", "banana", "orange", "pear")
ORANGE_RGB, BLUE_RGB, GOLD_RGB = (0.91, 0.44, 0.10), (0.35, 0.58, 0.80), (0.95, 0.70, 0.02)
TINT_T, TINT_N = 0.6, 0.5


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Ep(O.Replay):
    def __init__(self, fid):
        super().__init__(fid)
        self.base = self.m.geom_rgba.copy()

    def pose(self, fid):
        with np.load(RUN / "s0/replay" / f"{fid}.npz") as z:
            self.d.qpos[:] = z["qpos"]
        mujoco.mj_forward(self.m, self.d)
        self.m.geom_rgba[:] = self.base

    def body(self, n):
        return self.d.xpos[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)].copy()

    def tint(self, names, rgb, mix):
        ids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
        g = np.isin(self.m.geom_bodyid, [i for i in ids if i >= 0])
        self.m.geom_rgba[g, :3] = (1 - mix) * self.base[g, :3] + mix * np.asarray(rgb)


def labels(fid):
    return np.load(RUN / "s0/labels" / f"{fid}.npz", allow_pickle=True)


def save(name, img, fid, note, extra, srcs):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.png"
    Image.fromarray(img).save(p)
    meta = dict(asset=str(p.relative_to(REPO)), frame=fid, run=str(RUN), renderer=f"MuJoCo {mujoco.__version__} replay of recorded qpos (s0/replay)",
                script="paper/tools/plotting/cvpr_v1_fig3_assets.py", replaces=str((OLD / f"{name[:-3]}.json").relative_to(REPO)),
                sources={str(s): sha(s) for s in srcs}, note=note, is_measured_result=False, **extra)
    p.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=float) + "\n")
    print("wrote", p.relative_to(REPO), flush=True)


def tint_scene(ep, fr, target, ctr, with_neighbours=True):
    ep.tint([target], ORANGE_RGB, TINT_T)
    neigh = [f for f in FRUITS if f != target and np.linalg.norm(ep.body(f) - ctr) <= 0.16]
    if with_neighbours:
        ep.tint(neigh, BLUE_RGB, TINT_N)
    return neigh


def main():
    # ------------------------------------------------------------ T53 ep1842, M2 pre-grasp (fig01 panels 2/4, fig03 stages 3/5)
    fid = "T53_ep1842_f016"
    lab = labels(fid); target = str(lab["target"])
    fr = XF.Frame(RUN, fid, mask="render")
    s = XF.seed_and_g(fr, True); reg = O.top1(fr, s)
    ctr = fr.geo.pos[fr.gt].mean(0)
    near = O.neighbours(fr, ctr, 0.14) & ~reg
    close = O.cam_at(ctr, 0.42, 150, -34)
    ep = Ep(fid); ep.pose(fid)
    neigh = tint_scene(ep, fr, target, ctr)
    srcs = [RUN / "s0/replay" / f"{fid}.npz", RUN / "s0/labels" / f"{fid}.npz"]
    tint_note = (f"object meshes tinted for legibility: target '{target}' orange, touching neighbour(s) {neigh} light blue; "
                 "overlaid nodes = recomputed 5 mm observed-surface nodes (orange = selected target surface, blue = protected neighbour)")
    save("t_surface_v2", ep.render(close, [(fr.geo.pos[near], O.BLUE, O.HALF)] + O.eval_layers(fr, reg)), fid,
         "Observed target surface (orange nodes) and the touching neighbour's observed surface (blue nodes). " + tint_note,
         dict(iou=fr.iou(reg), neighbour_nodes=int(near.sum()), camera=close, target=target, tinted=dict(target=target, neighbours=neigh)), srcs)
    save("p_region_v2", ep.render(close, [(fr.geo.pos[near], O.BLUE, O.HALF)] + O.eval_layers(fr, reg)), fid,
         "Top-1 seed-mass region (orange nodes); neighbour surfaces (blue nodes) stay protected. " + tint_note,
         dict(iou=fr.iou(reg), camera=close, target=target, tinted=dict(target=target, neighbours=neigh)), srcs)
    hand = json.loads((OLD / "tp_episode.json").read_text())["active_hand"]
    ep.tint(HANDS[hand], GOLD_RGB, 0.75)
    save("t_permission_v2", ep.render(close, [(fr.geo.pos[near], O.BLUE, O.HALF), (fr.geo.pos[reg], (0.91, 0.44, 0.10, 0.35), O.HALF)]), fid,
         "Configured contact rule, visualised: only the active-hand links (gold) may enter the target surface; the neighbour (blue) stays an "
         "obstacle for every link. " + tint_note,
         dict(active_hand=hand, camera=close, target=target, tinted=dict(target=target, neighbours=neigh, hand=list(HANDS[hand])), is_rule_illustration=True), srcs)
    ep.m.geom_rgba[:] = ep.base

    # ------------------------------------------------------------ T53 ep1842, M4 carry: wrist view robot mask (fig03 stage 2)
    fid4 = "T53_ep1842_f025"
    ep.pose(fid4)
    cw = {"R": 2, "L": 1}[hand]
    img = ep.render(None, mjcam=ep.cams[cw]).astype(float)
    with np.load(RUN / "s0/replay" / f"{fid4}.npz") as z:
        fk = z["robot_mask"][cw]
    with np.load(RUN / "rounds/R6/robot_mask" / f"{fid4}.npz") as z:
        rm = z["robot_mask"][cw]
    lab4 = labels(fid4)
    tgt = lab4["class_map"][cw] == 0            # class 0 = 'target'
    H, W = img.shape[:2]
    up = lambda a: np.asarray(Image.fromarray(a.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 0
    fk, rm, tgt = up(fk), up(rm), up(tgt)
    out = img.copy()
    sel = tgt & ~rm
    out[sel] = 0.45 * img[sel] + 0.55 * np.array([232, 113, 26])          # orange: every observed target pixel
    out[rm] = 0.35 * img[rm] + 0.65 * np.array([140, 90, 200])            # purple: rendered robot mask
    extra = fk & ~rm                                                       # removed in addition by the volume filter (counted, not drawn)
    save("p_robot_mask_v2", out.astype(np.uint8), fid4,
         "Active wrist camera while carrying: orange = every observed target pixel (label class 'target' outside the robot), purple = rendered "
         "robot-surface mask. The pixels a sphere/capsule volume filter would remove in addition are counted in volume_filter_extra_px but not drawn.",
         dict(moment="M4_carry", camera=ep.cams[cw], render_px=int(rm.sum()), volume_filter_extra_px=int(extra.sum()),
              target_px_outside_robot=int(sel.sum()), target=str(lab4["target"])),
         [RUN / "s0/replay" / f"{fid4}.npz", RUN / "rounds/R6/robot_mask" / f"{fid4}.npz", RUN / "s0/labels" / f"{fid4}.npz"])
    ep.close()

    # ------------------------------------------------------------ fig02 card 2: surface region (T53_ep1849_f018) and render mask (T54c_ep1817_f020)
    meta = json.loads((OLD / "c2_ours_surface_region.json").read_text())
    fid2 = meta["frame"]; cam2 = meta["camera"]
    lab2 = labels(fid2); target2 = str(lab2["target"])
    fr2 = XF.Frame(RUN, fid2, mask="render")
    s2 = XF.seed_and_g(fr2, True); cm2 = O.top1(fr2, s2)
    ctr2 = fr2.geo.pos[fr2.gt].mean(0)
    neigh_nodes = O.neighbours(fr2, ctr2) & ~cm2
    ep2 = Ep(fid2); ep2.pose(fid2)
    neigh2 = tint_scene(ep2, fr2, target2, ctr2)
    save("c2_ours_surface_region_v2", ep2.render(cam2, [(fr2.geo.pos[neigh_nodes], O.BLUE, O.HALF)] + O.eval_layers(fr2, cm2)), fid2,
         "Depth-continuous surface graph, top-1 seed-mass region (orange nodes); neighbour surfaces stay blue (protected). "
         f"Meshes tinted: target '{target2}' orange, neighbours {neigh2} light blue.",
         dict(iou=fr2.iou(cm2), camera=cam2, target=target2, tinted=dict(target=target2, neighbours=neigh2)),
         [RUN / "s0/replay" / f"{fid2}.npz", RUN / "s0/labels" / f"{fid2}.npz"])
    ep2.close()
    meta3 = json.loads((OLD / "c2_ours_render_mask.json").read_text())
    fid3 = meta3["frame"]; cam3 = meta3["camera"]
    lab3 = labels(fid3); target3 = str(lab3["target"])
    frm = XF.Frame(RUN, fid3, mask="render")
    ctr3 = frm.geo.pos[frm.gt].mean(0)
    ep3 = Ep(fid3); ep3.pose(fid3)
    tint_scene(ep3, frm, target3, ctr3, with_neighbours=False)
    save("c2_ours_render_mask_v2", ep3.render(cam3, [(frm.geo.pos[frm.gt], O.ORANGE, O.HALF)]), fid3,
         f"Observed target nodes left after the rendered robot-surface mask (orange nodes). Mesh tinted: target '{target3}' orange.",
         dict(camera=cam3, target=target3, target_nodes_render=int(frm.gt.sum()), tinted=dict(target=target3, neighbours=[])),
         [RUN / "s0/replay" / f"{fid3}.npz", RUN / "s0/labels" / f"{fid3}.npz"])
    ep3.close()
    print("done")


if __name__ == "__main__":
    main()
