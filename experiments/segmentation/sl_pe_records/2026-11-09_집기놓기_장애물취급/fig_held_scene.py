"""(offline, 2026-11-09 04:5x dev-4a) real scene: the held bowl in the obstacle-object set (pe12345 LIBERO s9 t9) and what the
implemented SL-PE8 tracker (`AG3S.ag3s.stages.held_motion.HeldMotionTracker`, run on the recorded clouds with the MuJoCo hand
pose and the recorded fingertip) would remove. Side view (x-z, base frame) of selected held calls.
Also prints, per held call, the fraction of the observed GT bowl points removed and other-object points removed (GT scoring).
Run: SCENE_DIR=<scene24L9> python fig_held_scene.py '<run glob>' <stem> <out.png> <seq,seq,...>"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, "/mnt/dev/AG3S_VLA/outputs/2026-11-08_PE_경로끝점_identity")
sys.path.insert(0, "/mnt/dev/AG3S_VLA")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pe_signal as PS  # noqa: E402
from held_obst_replay import frame_obstacles  # noqa: E402
from AG3S.ag3s.stages.held_motion import HeldMotionTracker  # noqa: E402

Q = PS.Q
HAND = ("panda_hand", "panda_leftfinger", "panda_rightfinger")


def main():
    run = next(p for p in sorted(Q.E.RUNS.glob(sys.argv[1])) if not p.name.startswith("INVALID"))
    stem, out, show = sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4].split(",")]
    eps = [json.loads(l) for l in open(run / "episodes.jsonl")]
    order = [f"{e['suite']}_t{e['task_id']}_e{e['episode_idx']}" for e in eps]
    ch = PS.episode_chunks(run, order)
    info, m, rg, cache, stc = PS.scene_for(stem)
    d = mujoco.MjData(m)
    hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot0_right_hand")
    geom = np.load(run / "episodes" / f"{stem}_geom.npz")
    grow = {int(o): i for i, o in enumerate(geom["obs_index"])}
    calls = {int(json.loads(l)["seq"]): json.loads(l) for l in open(run / "episodes" / f"{stem}_calls.jsonl")}
    tr = HeldMotionTracker()
    panels = []
    for ts, cp in ch[stem]:
        c = calls[ts]
        st = (c.get("grasp") or {}).get("state")
        z = np.load(cp)
        i = grow[int(c["obs_index"])]
        d.qpos[:] = geom["sim_state_image"][i][1:1 + m.nq]
        mujoco.mj_forward(m, d)
        Tw = geom["T_world_base"][i]
        Twh = np.eye(4)
        Twh[:3, :3] = d.xmat[hb].reshape(3, 3)
        Twh[:3, 3] = d.xpos[hb]
        T = np.linalg.inv(Tw) @ Twh
        pts = z["filtered_cloud"].astype(float)[~z["support_mask"].astype(bool)]
        tip = (((c.get("grasp") or {}).get("tip_alignment") or {}).get("now") or {}).get("tip_m")
        held = st == "held"
        pre = z["target_points"].astype(float) if (not held and "target_points" in z.files) else None
        rm = tr.update(pts, T, None if tip is None else np.asarray(tip), held, pre)
        S2, G2 = cache.world(d)
        bowl = S2[rg[G2] == "target"]
        Pw = pts @ Tw[:3, :3].T + Tw[:3, 3]
        on_bowl = cKDTree(bowl).query(Pw)[0] <= 0.01
        if held:
            print(f"{ts:3d} held: bowl pts {int(on_bowl.sum()):3d} removed {int((rm & on_bowl).sum()):3d} | other removed "
                  f"{int((rm & ~on_bowl).sum()):3d} | verified {tr.last.get('n_verified')}", flush=True)
        if ts in show:
            obst = frame_obstacles(z, None)
            s = json.loads(str(z["summary_json"]))
            names = np.array(s["sphere_link_names"])
            hm = np.isin(names, HAND)
            pc = (c.get("to") or {}).get("passthrough_clear") or {}
            w = ((pc.get("obstacle_rows") or {}).get("worst") or {})
            panels.append(dict(seq=ts, st=st, pts=pts, obst=obst, rm=rm, sph=z["sphere_centres"][hm], rad=z["sphere_radii"][hm],
                               worst=w.get("point_m") if pc.get("reason") == "obstacle" else None, clr=w.get("clearance_m")))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6))
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, panels):
        hand = p["sph"].mean(0)
        win = (np.abs(p["pts"][:, 0] - hand[0]) < 0.25) & (np.abs(p["pts"][:, 2] - hand[2]) < 0.25)
        ax.scatter(p["pts"][win, 0], p["pts"][win, 2], s=2, c="#c9c8c2", label="observed cloud (non-support)")
        ow = (np.abs(p["obst"][:, 0] - hand[0]) < 0.25) & (np.abs(p["obst"][:, 2] - hand[2]) < 0.25)
        ax.scatter(p["obst"][ow, 0], p["obst"][ow, 2], s=6, c="#eb6834", label="obstacle-object set (current rule)")
        r = p["rm"] & win
        ax.scatter(p["pts"][r, 0], p["pts"][r, 2], s=8, facecolors="none", edgecolors="#2a78d6", linewidths=0.8,
                   label="removed by SL-PE8 (moved with the hand)")
        for cc, rr in zip(p["sph"], p["rad"]):
            ax.add_patch(plt.Circle((cc[0], cc[2]), rr, fill=False, color="#52514e", lw=0.4))
        if p["worst"] is not None:
            ax.plot(p["worst"][0], p["worst"][2], marker="*", ms=14, color="#e34948", ls="none",
                    label="sphere of the row that refused the pass-through")
        ax.set_aspect("equal")
        ax.set_xlim(hand[0] - 0.25, hand[0] + 0.25)
        ax.set_ylim(hand[2] - 0.25, hand[2] + 0.2)
        ax.set_title(f"call {p['seq']} ({p['st']})" + ("" if p["worst"] is None else f", refusing row {p['clr'] * 1e3:.0f} mm"), fontsize=10)
        ax.set_xlabel("x (m, base)")
        ax.set_ylabel("z (m, base)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9, markerscale=2)
    fig.suptitle(f"{run.name[16:58]} {stem}: held bowl vs obstacle-object set (side view; grey circles = hand / finger spheres)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(out, dpi=140, facecolor="#fcfcfb")
    print(out)


if __name__ == "__main__":
    main()
