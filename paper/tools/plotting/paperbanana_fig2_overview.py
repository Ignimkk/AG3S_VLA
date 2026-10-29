"""Figure 2 (overview) draft: how AG3S differs from existing pipelines + AG3S on one concrete scene.

Scene: T43 V11AB (AG3S main configuration), episode 1982, seed 19822, red overbar t43p2_overbar -- the
same scene as Figure 1. Renders are MuJoCo replays of the recorded qpos (frames.jsonl) with the run's
hurdle; overlays come from the paired server chunk dumps (attention, fused cloud, target cluster, scene
ESDF and the recorded target-free ESDF, query spheres, attached sphere, reference / refined chunks).

    MUJOCO_GL=osmesa .venv-ag3s/bin/python paper/tools/plotting/paperbanana_fig2_overview.py
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
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "AG3S/ag3s/experiments/studies"))
sys.path.insert(0, str(REPO / "AG3S/ag3s/experiments"))
sys.path.insert(0, str(REPO / "AG3S"))
import t53_records  # noqa: E402
import paperbanana_fig1_scenes as fs  # noqa: E402
from paperbanana_fig1_variants import fit, pad  # noqa: E402

T43 = fs.T43
RUN = T43 / "V11/runs/V11AB/H/V11AB_ep1982_s19822"
SERVER = T43 / "V11/runs/V11AB"
OUT = REPO / "paper/figures/paperbanana/figure2"
A = OUT / "assets"
F1 = REPO / "paper/figures/paperbanana/figure1/variants/assets"

CH = {"attention": 8, "ground": 5, "field": 5, "attach": 16}
TARGET = (0.91, 0.44, 0.10, 1.0)
NEIGH = (0.18, 0.46, 0.71, 1.0)
GOLD = (0.95, 0.70, 0.02, 1.0)
PURPLE = (0.44, 0.19, 0.63, 0.45)
C = {"red": "#C62828", "green": "#2E7D32", "gold": "#D9A406", "ink": "#1F1F1F", "navy": "#C9DCF0",
     "goldfill": "#FFF2CC", "field": "#DAE8FC", "state": "#FBE5D6", "ctrl": "#E4DFEC", "grey": "#E7E6E6"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
CAM = {"lookat": [0.51, 0.25, 0.92], "dist": 0.58, "az": 270, "el": -30}


class Scene(fs.Replay):
    def __init__(self):
        super().__init__()
        self.rows = self.load(RUN)
        self.base = self.m.geom_rgba.copy()

    def at(self, j):
        self.m.geom_rgba[:] = self.base
        self.set(self.rows[min(8 * j, len(self.rows) - 1)])

    def fade(self, names, alpha):
        ids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
        g = np.isin(self.m.geom_bodyid, [i for i in ids if i >= 0])
        self.m.geom_rgba[g, 3] = alpha

    def tint(self, names, rgb, mix=0.7):
        ids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
        g = np.isin(self.m.geom_bodyid, [i for i in ids if i >= 0])
        self.m.geom_rgba[g, :3] = (1 - mix) * self.base[g, :3] + mix * np.asarray(rgb[:3])

    def body(self, n):
        return self.d.xpos[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)].copy()

    def cam_image(self, cam, W=1280, H=960):
        r = mujoco.Renderer(self.m, H, W)
        r.update_scene(self.d, cam)
        img = r.render().copy(); r.close()
        return img

    def box_extras(self, pts, half, rgba):
        return [("sphere", p, half * 1.2, rgba) for p in pts]


def chunks():
    segs = t53_records.segments(sorted(SERVER.glob("server_*/server_constraints/run_*")))
    pc = np.load(RUN / "traj.npz")["predicted_chunks"]
    i = t53_records.pair_segment(segs, list(pc), tol=1e-4)
    if i is None:
        raise SystemExit("no server segment pairs with this run")
    return list(segs[i])


def dump(cs, j):
    z = np.load(cs[j], allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["summary"] = json.loads(str(d.pop("summary_json")))
    return d


def save(img, name, note, extra=None):
    A.mkdir(parents=True, exist_ok=True)
    p = A / f"{name}.png"
    Image.fromarray(img).save(p)
    p.with_suffix(".json").write_text(json.dumps({"asset": str(p.relative_to(REPO)), "run": str(RUN), "note": note,
                                                  **(extra or {})}, indent=1, default=str) + "\n")
    return p


def overlay(rgb, att):
    a = np.asarray(att, float)
    a = (a - a.min()) / max(a.max() - a.min(), 1e-12)
    up = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(rgb.shape[1::-1], Image.BICUBIC)) / 255.0
    al = np.clip((up - 0.08) / 0.6, 0, 1)[..., None] * 0.85
    return ((rgb / 255.0 * (1 - al) + plt.cm.inferno(up)[..., :3] * al) * 255).astype(np.uint8)


def surface_voxels(dist, origin, vs, centre, half, z_min=0.80, shell=0.004):
    ii = np.argwhere(np.abs(dist) <= shell)
    pts = origin[None] + (ii + 0.5) * vs
    keep = np.all(np.abs(pts - centre[None]) <= half, axis=1) & (pts[:, 2] > z_min)
    return ii[keep], pts[keep]


# ------------------------------------------------------------------ renders
def renders():
    sc, cs = Scene(), chunks()
    made = {}
    j = CH["attention"]; c = dump(cs, j); sc.at(j)
    made["wrist_att"] = save(overlay(sc.cam_image("wrist_cam_l"), c["attention_left_wrist"]), "ex_wrist_attention",
                             "Active (left) wrist camera re-rendered from the recorded state + served 16x16 attention.", {"chunk": j})
    made["head_att"] = save(overlay(sc.cam_image("zed_left"), c["attention_head"]), "ex_head_attention",
                            "Head camera re-rendered + served attention.", {"chunk": j})

    j = CH["ground"]; c = dump(cs, j); sc.at(j)
    cloud = np.asarray(c["filtered_cloud"], float); sup = np.asarray(c["support_mask"], bool)
    tgt = np.asarray(c["target_points"], float); centre = tgt.mean(0)
    pear = sc.body("pear")
    win = np.linalg.norm(cloud - centre[None], axis=1) <= 0.22
    nb = np.linalg.norm(cloud - pear[None], axis=1) <= 0.05
    sc.fade(("apple", "pear", "banana", "orange"), 0.25)
    ex = [("sphere", p, 0.0032, (0.72, 0.72, 0.72, 1.0)) for p in cloud[win & ~sup & ~nb]]
    ex += [("sphere", p, 0.0042, NEIGH) for p in cloud[win & nb & ~sup]] + [("sphere", p, 0.0048, TARGET) for p in tgt]
    made["ground"] = save(sc.render(CAM, ex), "ex_grounding",
                          "Recorded fused cloud: target cluster (orange), pear neighbour (blue; coloured by its replayed body "
                          "position, display only), other observed surfaces incl. the overbar (grey).", {"chunk": j})

    # task- and link-conditioned field: scene ESDF vs recorded target-free ESDF
    j = CH["field"]; c = dump(cs, j); sc.at(j)
    o, vs = np.asarray(c["esdf_origin"], float), float(c["esdf_voxel_size"])
    main = np.asarray(c["esdf_distance"], np.float32); free = np.asarray(c["esdf_free_distance"], np.float32)
    tgt = np.asarray(c["target_points"], float); centre = tgt.mean(0)
    ii, pts = surface_voxels(main, o, vs, np.array([0.52, 0.22, 0.95]), np.array([0.20, 0.20, 0.16]), z_min=0.835)
    removed = free[tuple(ii.T)] > 0.008
    names = np.array(c["summary"]["sphere_link_names"]); auth = set(c["summary"]["exclusion"]["authorized_links"])
    sph, rad = np.asarray(c["sphere_centres"], float), np.asarray(c["sphere_radii"], float)
    contact = np.isin(names, list(auth))
    sc.fade(("apple", "pear"), 0.2)
    sc.tint(fs.GRIPPER, GOLD)
    ex = [("sphere", p, vs * 0.62, (0.91, 0.44, 0.10, 0.30) if r_ else (0.18, 0.46, 0.71, 0.9))
          for p, r_ in zip(pts, removed)]
    ex += [("sphere", p, r, (0.95, 0.70, 0.0, 0.85)) for p, r, k in zip(sph, rad, contact) if k]
    made["field"] = save(sc.render(CAM, ex), "ex_field",
                         "Recorded scene ESDF (5 mm) surface voxels near the target; voxels absent from the recorded "
                         "target-free ESDF are faint orange (removed for contact links only). Gold spheres = contact-link "
                         "queries (left gripper); the overbar and the pear stay obstacles.",
                         {"chunk": j, "n_voxels": int(len(pts)), "n_removed": int(removed.sum()),
                          "authorized_links": sorted(auth)})
    # same voxels, target kept for everyone (what a geometry-only filter sees)
    ex = [("sphere", p, vs * 0.62, (0.18, 0.46, 0.71, 0.9)) for p in pts]
    sc.m.geom_rgba[:] = sc.base; sc.fade(("apple", "pear"), 0.2)
    made["field_all"] = save(sc.render(CAM, ex), "ex_field_all_obstacles",
                             "Same recorded scene-ESDF voxels with the target kept: every surface is an obstacle.", {"chunk": j})

    # CCR + attached object during carry: chunk with the largest executed-row TCP shift after attach
    adr = [sc.m.jnt_qposadr[mujoco.mj_name2id(sc.m, mujoco.mjtObj.mjOBJ_JOINT, f"left_arm_{k}")] for k in range(7)]
    def tcp(q, rows):
        out = []
        for row in rows:
            qq = np.array(q, float); qq[adr] = row[0:7]; sc.set(qq); out.append(sc.d.site_xpos[sc.site].copy())
        return np.asarray(out)
    best = (-1, None)
    for jj in range(CH["attach"] + 1, 30):
        z = np.load(cs[jj]); q = sc.rows[min(8 * jj, len(sc.rows) - 1)]
        a = tcp(q, z["reference_chunk"][:8]); b = tcp(q, z["refined_chunk"][:8])
        sh = float(np.linalg.norm(a - b, axis=1).max())
        if sh > best[0]:
            best = (sh, jj)
    j = best[1]; c = dump(cs, j); q = sc.rows[min(8 * j, len(sc.rows) - 1)]
    ref = tcp(q, c["reference_chunk"][:16]); refd = tcp(q, c["refined_chunk"][:16])
    ca = dump(cs, CH["attach"]); att = ca["summary"]["grasp"]["attached"]["held"]["spheres"]
    hb = mujoco.mj_name2id(sc.m, mujoco.mjtObj.mjOBJ_BODY, "EE_BODY_L")
    sc.at(CH["attach"]); R0, p0 = sc.d.xmat[hb].reshape(3, 3).copy(), sc.d.xpos[hb].copy()
    local = [(R0.T @ (np.asarray(s_["centre_m"]) - p0), s_["radius_mm"] / 1000) for s_ in att]
    sc.at(j); R1, p1 = sc.d.xmat[hb].reshape(3, 3), sc.d.xpos[hb]
    ex = [("line", (ref[i], ref[i + 1]), 0.004, (0.2, 0.2, 0.2, 1.0)) for i in range(0, len(ref) - 1, 2)]
    ex += [("line", (refd[i], refd[i + 1]), 0.005, GOLD) for i in range(len(refd) - 1)]
    ex += [("sphere", p, 0.007, GOLD) for p in refd[:8]]
    ex += [("sphere", p1 + R1 @ lc, r, PURPLE) for lc, r in local]
    mid = ((ref.mean(0) + refd.mean(0)) / 2).tolist()
    made["ccr"] = save(sc.render({"lookat": mid, "dist": 0.62, "az": 150, "el": -18}, ex), "ex_ccr",
                       "Carry chunk with the largest executed-row tool-centre shift: policy reference (dashed dark) vs "
                       "SQP-refined (gold), FK of the recorded chunks; attached-object sphere (purple) carried with the hand.",
                       {"chunk": j, "max_tcp_shift_mm": best[0] * 1000})

    # state timeline
    states, p = [], []
    for jj in range(len(cs)):
        s_ = json.loads(str(np.load(cs[jj])["summary_json"]))
        states.append((s_.get("grasp") or {}).get("state"))
        sp = (s_.get("subtask") or {}).get("p") or {}
        p.append([sp.get("pick", np.nan), sp.get("place", np.nan), sp.get("home", np.nan)])
    p = np.asarray(p, float)
    cols = {"searching": "#F2F2F2", "latched": "#DEEBF7", "closing": "#FFE699", "held": "#E4D3F0", "placed": "#E2F0D9"}
    fig, axs = plt.subplots(2, 1, figsize=(4.4, 3.0), sharex=True, gridspec_kw={"height_ratios": [1, 1.5], "hspace": 0.1})
    for jj, s_ in enumerate(states):
        axs[0].axvspan(jj - 0.5, jj + 0.5, color=cols.get(s_, "white"), lw=0)
    for s_ in ("latched", "held", "placed"):
        idx = [k for k, v in enumerate(states) if v == s_]
        if idx:
            axs[0].text(np.mean(idx), 0.5, s_, ha="center", va="center", fontsize=9)
    axs[0].set_ylim(0, 1); axs[0].set_yticks([]); axs[0].set_ylabel("grasp", rotation=0, ha="right", va="center")
    for k, (lab, col) in enumerate(zip(["pick", "place", "home"], ["#2E75B6", "#C55A11", "#548235"])):
        axs[1].plot(np.arange(len(p)), p[:, k], color=col, lw=2, label=lab)
    axs[1].set_yticks([0, 1]); axs[1].set_ylabel("HSR", rotation=0, ha="right", va="center")
    axs[1].legend(frameon=False, ncol=3, fontsize=8.5, loc="center right")
    axs[1].set_xlabel("action chunk")
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    tp = A / "ex_timeline.png"
    fig.savefig(tp, dpi=220, bbox_inches="tight"); plt.close(fig)
    made["timeline"] = tp
    return made


# ------------------------------------------------------------------ compose
def im(p, crop=None):
    x = Image.open(p).convert("RGB")
    if crop:
        w, h = x.size
        x = x.crop((int(crop[0] * w), int(crop[1] * h), int(crop[2] * w), int(crop[3] * h)))
    return np.asarray(x)


def panel(fig, rect, arr, title=None, badge=None, bcol=None, border="#BBBBBB", lw=1.6, keep=False, tsize=11,
          anchor=(0.5, 0.5)):
    ax = fig.add_axes(rect, zorder=3)
    fw, fh = fig.get_size_inches()
    asp = rect[2] * fw / (rect[3] * fh)
    ax.imshow(pad(arr, asp) if keep else fit(arr, asp, anchor), aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(border); s.set_linewidth(lw)
    if title:
        ax.set_title(title, fontsize=tsize, fontweight="bold", color=C["ink"], pad=4)
    if badge:
        ax.text(0.03, 0.05, badge, transform=ax.transAxes, fontsize=9.5, fontweight="bold", color="white", ha="left",
                va="bottom", bbox=dict(boxstyle="round,pad=0.3", fc=bcol, ec="none"))


def blk(fig, x, y, w, h, text, fc, ec="#555555", size=9, bold=False):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.002,rounding_size=0.006",
                                  transform=fig.transFigure, fc=fc, ec=ec, lw=1.1, zorder=2))
    fig.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, zorder=5,
             fontweight="bold" if bold else "normal", color=C["ink"])


def arr(fig, a, b, color="#555555", lw=1.6, ms=11):
    fig.add_artist(FancyArrowPatch(a, b, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=ms, color=color,
                                   lw=lw, zorder=4))


def band(fig, x, y, w, h, fc, ec, z=-2):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.01",
                                  transform=fig.transFigure, fc=fc, ec=ec, lw=1.4, zorder=z))


def compose(made):
    fig = plt.figure(figsize=(17, 10.2))
    fig.text(0.012, 0.968, "(a) AG3S vs. existing pipelines", fontsize=14, fontweight="bold", color=C["ink"])
    cards = [(0.008, "End-to-end VLA", "#FDECEA", "#E8A9A3"), (0.338, "VLA + geometric safety filter", "#FDECEA", "#E8A9A3"),
             (0.668, "AG3S (ours)", "#FFF7DD", C["gold"])]
    for x, t, fc, ec in cards:
        band(fig, x, 0.535, 0.324, 0.415, fc, ec)
        fig.text(x + 0.012, 0.925, t, fontsize=12.5, fontweight="bold", color=C["red"] if ec != C["gold"] else "#9A7300")
    # card 1 flow
    y = 0.865
    blk(fig, 0.02, y, 0.07, 0.04, "RGB +\ninstruction", C["grey"], size=8.5)
    blk(fig, 0.115, y, 0.07, 0.04, "VLA", C["navy"], bold=True)
    blk(fig, 0.21, y, 0.07, 0.04, "actions", C["grey"], size=9)
    arr(fig, (0.09, y + 0.02), (0.115, y + 0.02)); arr(fig, (0.185, y + 0.02), (0.21, y + 0.02))
    fig.text(0.02, 0.835, "no 3D geometry; contact is never checked", fontsize=9, style="italic", color="#555555")
    panel(fig, [0.02, 0.548, 0.30, 0.275], im(F1 / "vla_place_fail.png"), badge="✗ collides with the hurdle",
          bcol=C["red"], border=C["red"])
    # card 2 flow
    x0 = 0.35
    blk(fig, x0, y, 0.06, 0.04, "RGB-D", C["grey"], size=9)
    blk(fig, x0 + 0.08, y, 0.09, 0.04, "ESDF of all\nsurfaces", C["field"], size=8.5)
    blk(fig, x0 + 0.19, y, 0.06, 0.04, "VLA", C["navy"], bold=True)
    blk(fig, x0 + 0.08, y - 0.055, 0.17, 0.035, "safety filter on actions", C["ctrl"], size=8.5)
    arr(fig, (x0 + 0.06, y + 0.02), (x0 + 0.08, y + 0.02))
    arr(fig, (x0 + 0.125, y), (x0 + 0.125, y - 0.02)); arr(fig, (x0 + 0.22, y), (x0 + 0.22, y - 0.02))
    fig.text(x0, 0.776, "target is an obstacle too, or removed for every\nlink by an external segmenter", fontsize=9,
             style="italic", color="#555555", va="center")
    panel(fig, [x0 - 0.01 + 0.0, 0.548, 0.30, 0.205], im(made["field_all"]), badge="✗ target repels the fingers: grasp blocked",
          bcol=C["red"], border=C["red"], anchor=(0.5, 0.8))
    # card 3 flow
    x0 = 0.678
    blk(fig, x0, y, 0.06, 0.04, "RGB-D +\ninstruction", C["grey"], size=8)
    blk(fig, x0 + 0.075, y, 0.06, 0.04, "frozen\nVLA", C["navy"], size=8.5, bold=True)
    blk(fig, x0 + 0.15, y, 0.075, 0.04, "attention →\n3D target", C["goldfill"], ec=C["gold"], size=8.5, bold=True)
    blk(fig, x0 + 0.24, y, 0.065, 0.04, "sub-task\n(HSR)", C["state"], ec="#C55A11", size=8.5)
    blk(fig, x0 + 0.075, y - 0.058, 0.15, 0.038, "task- & link-conditioned field", C["goldfill"], ec=C["gold"], size=8.5, bold=True)
    blk(fig, x0 + 0.24, y - 0.058, 0.065, 0.038, "CCR", C["ctrl"], ec="#7B6A9A", size=9, bold=True)
    arr(fig, (x0 + 0.06, y + 0.02), (x0 + 0.075, y + 0.02)); arr(fig, (x0 + 0.135, y + 0.02), (x0 + 0.15, y + 0.02))
    arr(fig, (x0 + 0.1875, y), (x0 + 0.1875, y - 0.02), color=C["gold"]); arr(fig, (x0 + 0.2725, y), (x0 + 0.21, y - 0.02), color="#C55A11")
    arr(fig, (x0 + 0.225, y - 0.039), (x0 + 0.24, y - 0.039), color=C["gold"])
    fig.text(x0, 0.776, "the VLA's own attention picks the target; only the\ngrasping links may touch it",
             fontsize=9, style="italic", color="#555555", va="center")
    panel(fig, [x0 - 0.0, 0.548, 0.30, 0.205], im(F1 / "ag3s_place_ok.png"), badge="✓ grasps the target, clears the hurdle",
          bcol=C["green"], border=C["gold"], lw=2.2)

    # (b) concrete example
    fig.text(0.012, 0.495, "(b) AG3S on one scene: “Put the apple in the basket.”", fontsize=14, fontweight="bold")
    band(fig, 0.008, 0.015, 0.984, 0.465, "#FFFDF5", C["gold"])
    w, h, yb = 0.183, 0.335, 0.088
    items = [(made["wrist_att"], "① Frozen-VLA attention", None, False),
             (made["ground"], "② 3D target grounding", None, False),
             (made["field"], "③ Task- & link-conditioned field", None, False),
             (made["timeline"], "④ Sub-task & grasp state", None, True),
             (made["ccr"], "⑤ Constrained chunk refinement", None, False)]
    for k, (p, t, _, keep) in enumerate(items):
        x = 0.016 + k * (w + 0.0138)
        panel(fig, [x, yb, w, h], im(p), title=t, border=C["gold"] if k in (1, 2) else "#BBBBBB", keep=keep, tsize=10.5)
        if k < 4:
            arr(fig, (x + w + 0.001, yb + h / 2), (x + w + 0.0128, yb + h / 2), color=C["gold"], lw=2.2, ms=14)
    caps = ["attention of the frozen policy\npeaks on the apple", "apple cluster (orange) vs. touching\npear (blue) and the bar",
            "orange: removed for contact links\nonly; bar and pear stay obstacles", "target locked while placing;\nheld object attached",
            "dashed: policy; gold: refined;\npurple: attached object"]
    for k, cpt in enumerate(caps):
        fig.text(0.016 + k * (w + 0.0138) + w / 2, yb - 0.004, cpt, ha="center", va="top", fontsize=8.6, color="#444444")
    out = OUT / "reference_montage.png"
    fig.savefig(out, dpi=170, facecolor="white"); plt.close(fig)
    print("wrote", out.relative_to(REPO))


if __name__ == "__main__":
    compose(renders())
