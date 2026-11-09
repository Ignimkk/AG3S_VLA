"""Rule A scene figure (2026-11-08, dev-30): the recorded cloud around the VLA's grasp point, recorded identity (R0) vs the
object cut out under the grasp point (SL-PE1, `stages/close_seed.split_at_seed`), GT target surface for scoring only.
Top (x-y) and side (x-z) views in the robot base frame, one row per case. Writes pe_scene.png + pe_scene.json."""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial import cKDTree

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/mnt/dev/AG3S_VLA")
import pe_signal as PS  # noqa: E402
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
from AG3S.ag3s.stages.close_seed import split_at_seed  # noqa: E402

Q = PS.Q
CASES = [("libero_spatial_t9_e0", 9, "t9 seed 8: bowl on the cabinet"),
         ("libero_spatial_t4_e0", 21, "t4 seed 8: bowl in the drawer"),
         ("libero_spatial_t3_e0", 8, "t3 seed 8: bowl on the cookie box")]


def main():
    ko = figstyle.use_korean()
    run = next(Q.E.RUNS.glob("*gt24att-libero9s8*"))
    eps = [json.loads(l) for l in open(run / "episodes.jsonl")]
    order = [f"{e['suite']}_t{e['task_id']}_e{e['episode_idx']}" for e in eps]
    ch = PS.episode_chunks(run, order)
    fig, axes = plt.subplots(len(CASES), 2, figsize=(11, 4.2 * len(CASES)))
    figstyle.style_axes(fig, axes.ravel())
    C = figstyle.CATEGORICAL
    side = {}
    for row, (stem, seq, title) in enumerate(CASES):
        info, m, rg, cache, stc = PS.scene_for(stem)
        d = mujoco.MjData(m)
        geom = np.load(run / "episodes" / f"{stem}_geom.npz")
        grow = {int(o): i for i, o in enumerate(geom["obs_index"])}
        calls = {int(json.loads(l)["seq"]): json.loads(l) for l in open(run / "episodes" / f"{stem}_calls.jsonl")}
        cp = dict(ch[stem])[seq]
        call = calls[seq]
        pc = call["grasp"]["tip_alignment"]["plan_close"]
        seed = np.asarray(pc["tip_m"], float)
        i = grow[int(call["obs_index"])]
        d.qpos[:] = geom["sim_state_image"][i][1:1 + m.nq]
        mujoco.mj_forward(m, d)
        S1, G1 = stc.world(d)
        S2, G2 = cache.world(d)
        tree = Q.SplitTree(cKDTree(S1), rg[G1], S2, rg[G2])
        Tw = geom["T_world_base"][i]
        W = lambda X: np.asarray(X).reshape(-1, 3) @ Tw[:3, :3].T + Tw[:3, 3]
        z = np.load(cp)
        cloud = z["filtered_cloud"].astype(float)
        sup = z["support_mask"].astype(bool)
        dist, j = tree.query(W(cloud))
        lab = np.where(dist <= 0.010, tree.roles[j], "none")
        idx, rec = split_at_seed(cloud, seed, exclude_mask=sup)
        r0 = z["target_points"].astype(float)
        if not len(r0):   # occluded identity: the kept geometry is in summary.manipulated only; use the last recorded
            r0 = np.zeros((0, 3))
        near = np.linalg.norm(cloud - seed, axis=1) <= 0.25
        r0_near = r0[np.linalg.norm(r0 - seed, axis=1) <= 0.25] if len(r0) else r0
        pe_lab = lab[idx] if idx is not None else np.array([])
        r0_role = (Q.role_counts(W(r0), tree, tree.roles, 0.010) or {}).get("majority") if len(r0) else "no target"
        side[stem] = {"seq": seq, "seed_m": seed.round(4).tolist(), "plan_close_row": int(pc["row"]), "cut": rec,
                      "pe_purity": None if idx is None else round(float(np.mean(pe_lab == "target")), 3),
                      "pe_recall": None if idx is None else round(float(np.isin(np.flatnonzero(lab == "target"), idx).mean()), 3),
                      "r0_majority": r0_role, "r0_centroid_m": None if not len(r0) else r0.mean(0).round(4).tolist()}
        for col, (a, b, xl, yl) in enumerate(((0, 1, "x (m)", "y (m)"), (0, 2, "x (m)", "z (m)"))):
            ax = axes[row, col]
            P = cloud[near]
            ax.scatter(P[:, a], P[:, b], s=2, c=figstyle.GRID_INK, alpha=0.5, lw=0,
                       label="cloud (support-plane points included)")
            T = cloud[near & (lab == "target")]
            ax.scatter(T[:, a], T[:, b], s=14, facecolors="none", edgecolors=figstyle.INK, lw=0.5,
                       label="GT target surface (scoring only)")
            if len(r0_near):
                ax.scatter(r0_near[:, a], r0_near[:, b], s=5, c=C[1], lw=0, label=f"recorded identity R0 ({r0_role})")
            if idx is not None:
                ax.scatter(cloud[idx, a], cloud[idx, b], s=5, c=C[0], lw=0, label="SL-PE1 cut under the grasp point")
            ax.scatter([seed[a]], [seed[b]], marker="*", s=160, c=C[2], edgecolors=figstyle.INK, lw=0.6,
                       label="grasp point (plan_close tip)", zorder=5)
            if col == 1 and rec.get("local_support_top_m") is not None:
                ax.axhline(rec["local_support_top_m"], color=C[3], lw=1, ls="--", label="local support top")
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(f"{title} — call {seq} — {'top' if col == 0 else 'side'}", fontsize=9)
        if len(r0) and not len(r0_near):
            axes[row, 0].text(0.02, 0.02, f"R0 ({r0_role}) is outside this 25 cm crop: centroid "
                              f"{np.round(r0.mean(0), 3).tolist()} m", transform=axes[row, 0].transAxes,
                              fontsize=8, color=figstyle.INK_2)
        if r0_role == "no target" and len(r0) == 0:
            axes[row, 0].text(0.02, 0.02, "R0: no recorded points this call", transform=axes[row, 0].transAxes,
                              fontsize=8, color=figstyle.INK_2)
    fig.suptitle("Target identity: attention grounding (R0) vs the object under the VLA's grasp point (SL-PE1), "
                 "gt24att LIBERO seed 8", fontsize=10)
    h, l = axes[0, 1].get_legend_handles_labels()
    extra = [plt.Line2D([], [], marker="o", ls="", color=C[1], label="recorded identity R0")]
    seen = {}
    for ax in axes.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(ll.split(" (")[0] if ll.startswith("recorded") else ll, hh)
    fig.legend(list(seen.values()), list(seen.keys()), loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(HERE / "pe_scene.png", dpi=150)
    (HERE / "pe_scene.json").write_text(json.dumps(side, indent=1))
    print(json.dumps(side, indent=1))


if __name__ == "__main__":
    main()
