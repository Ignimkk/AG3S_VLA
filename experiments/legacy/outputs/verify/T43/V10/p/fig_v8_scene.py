"""T43 V8 scene figure: the --finger-cover finger body spheres (270-sphere chunk record, V8 server) overlaid on the real MuJoCo hand.
Rows: (1) closing moment ep1925 s19253 t128 (apple between the fingers), (2) after release near the bar ep1807 s18073 t264 (CT: finger-bar contact 5.8 mm at t270, obstacle stop).
Pose = the chunk's planned row 0 (arm joints of refined_chunk[0]; the chunk record's sphere_centres match this pose, 7 mm centroid check), finger / other joints = planning-row qpos.
Spheres: ee_finger_* (224 = 4 x 56 in the record; the active hand's 112 drawn) orange, ee_left palm 23 spheres blue. usage: fig_v8_scene.py OUT_PNG OUT_JSON"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/V8/p'))
import scene_probe as SP  # noqa: E402
import mujoco  # noqa: E402
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
RY = SP.RY
V = SP.V; an = json.load(open(V / "analysis/analysis.json"))["runs"]
C = figstyle.CATEGORICAL
hexrgb = lambda h, a: tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)) + (a,)
COL = {"ee_finger_l1": hexrgb(C[1], 0.55), "ee_finger_l2": hexrgb(C[3], 0.55), "ee_left": hexrgb(C[0], 0.40)}
MOMENTS = [("closing, apple between the fingers", "V8/H/V8_ep1925_s19253", 128), ("after release, finger near the bar (CT contact 5.8 mm at t270, obstacle stop)", "V8/H/V8_ep1807_s18073", 264)]
LEFT = list(range(20, 27))
CAMS = [[dict(az=200.0, el=-25.0, dist=0.34), dict(az=140.0, el=-20.0, dist=0.34), dict(az=180.0, el=-18.0, dist=0.9)],
        [dict(az=90.0, el=-5.0, dist=0.42), dict(az=150.0, el=-15.0, dist=0.40), dict(az=180.0, el=-18.0, dist=0.9)]]
kor = figstyle.use_korean()
fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.6)); figstyle.style_axes(fig, axes); meta = []
for r_, (lab, key, t) in enumerate(MOMENTS):
    r = an[key]; c = next(x for x in r["chunks"] if x["t"] == t)
    z = np.load(c["server_chunk"], allow_pickle=True); cen, rad, links = SP.chunk_spheres(c["server_chunk"]); ref = z["refined_chunk"]
    M, ctrl = RY.scene_for(r["run_dir"], "left")
    rows = [json.loads(l) for l in open(r["run_dir"] + "/frames.jsonl") if l.strip()]
    qp = np.array(next(x for x in rows if x["kind"] == "planning" and x["t_step"] == t)["qpos"], float)
    q = qp.copy(); q[LEFT] = ref[0, :7]
    m, d = M["m"], M["d"]; B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    d.qpos[:] = q; mujoco.mj_kinematics(m, d); b = B("base"); W = d.xpos[b] + cen @ d.xmat[b].reshape(3, 3).T
    sel = [i for i, l in enumerate(links) if l in COL]
    sph = [(W[i], float(rad[i]), COL[links[i]]) for i in sel]
    ctr = np.mean([W[i] for i in sel if links[i] != "ee_left"], 0)
    for cc, cam in enumerate(CAMS[r_]):
        cm = dict(lookat=tuple(ctr), distance=cam["dist"], azimuth=cam["az"], elevation=cam["el"])
        if cc == 2:
            cm["lookat"] = (0.50, 0.0, 1.0)
        img = RY.render(M, q, cm, sph); axes[r_, cc].imshow(img); axes[r_, cc].set_xticks([]); axes[r_, cc].set_yticks([])
        axes[r_, cc].set_title(("close-up az %d el %d" % (cam["az"], cam["el"])) if cc < 2 else "wider view (az 180)", fontsize=9, color=figstyle.INK)
    nl = {l: int(sum(1 for x in links if x == l)) for l in sorted(set(links))}
    fr = {l: [round(1e3 * float(rad[[i for i, x in enumerate(links) if x == l]].min()), 2), round(1e3 * float(rad[[i for i, x in enumerate(links) if x == l]].max()), 2)] for l in sorted(set(links))}
    axes[r_, 0].set_ylabel(f"{lab}\n{key.split('/')[-1]}  t {t}", fontsize=9, color=figstyle.INK)
    meta.append({"moment": lab, "run": key, "t": t, "server_chunk": c["server_chunk"], "n_spheres_by_link": nl, "radius_mm_range_by_link": fr, "spheres_drawn": len(sph),
                 "pose": "qpos of the planning row with left-arm joints (q index 20..26) = refined_chunk[0]", "view_centre_world": [float(x) for x in ctr],
                 "ct_finger_contacts": [(x["body"], x["phase"], x["first_t"], round(x["max_pen_mm"], 2)) for x in json.load(open(V / "ct.json"))["runs"][key]["contacts"] if x["body"].startswith("ee_finger")]})
handles = [plt.Line2D([0], [0], marker="o", ls="", color=hexrgb(C[1], 0.9), label="ee_finger_l1 body spheres (URDF name; MJCF ee_finger_l2)"),
           plt.Line2D([0], [0], marker="o", ls="", color=hexrgb(C[3], 0.9), label="ee_finger_l2 body spheres"),
           plt.Line2D([0], [0], marker="o", ls="", color=hexrgb(C[0], 0.9), label="ee_left palm spheres (23, unchanged)")]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
fig.suptitle("T43 V8 --finger-cover both: finger body spheres on the real hand (56 per finger, r 2.5-8.4 mm; 270 constraint spheres in the server record)", fontsize=11, color=figstyle.INK)
fig.subplots_adjust(left=0.05, right=0.995, top=0.93, bottom=0.07, hspace=0.12, wspace=0.03)
out = pathlib.Path(sys.argv[1]); out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=105, facecolor=figstyle.SURFACE); plt.close(fig)
json.dump({"figure": str(out), "script": __file__, "panels": meta, "cams": CAMS, "sources": [str(V / "analysis/analysis.json"), str(V / "ct.json")]}, open(sys.argv[2], "w"), indent=1, default=float)
print("ok", out)
