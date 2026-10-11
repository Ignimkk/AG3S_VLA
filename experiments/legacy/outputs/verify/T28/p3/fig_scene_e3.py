"""T28 Phase 3 real-scene overlay just before the grasp. usage: fig_scene_e3.py COND RUN [SEQ]
Chunk: SEQ, default = the last planning row with t_step <= the run's first close command t (applied_ctrl.gripper[0] < 0.5).
MuJoCo re-posed with that planning row's qpos (objects included). Overlays in MuJoCo world (server base-frame arrays -> world with the
MuJoCo base body pose): recorded constraint spheres of ee_left / ee_finger_l1 / ee_finger_l2 (npz sphere_centres + sphere_radii,
the server FK at that frame with the T29 finger joints), GT apple collision mesh, GT left finger / palm collision meshes,
manipulated target points (npz target_points), the exclusion ball (esdf_stats.target_free.ball: centre, radius, z_min cut),
filtered cloud points within 0.15 m of the apple (npz filtered_cloud, subsampled), and the worst-row query point.
Third panel: the recorded third-person mp4 frame at that t."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np, mujoco
import imageio.v2 as iio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-3'))
from AG3S.ag3s.experiments.common import figstyle as fs
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
from p13a_spheres import body_mesh_world, XML
fs.use_korean(); C = fs.CATEGORICAL
F = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t28'))
V3 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T28/p3'))
cond, run = sys.argv[1], sys.argv[2]
AN = json.loads((V3 / f"{cond}_analysis.json").read_text())["runs"][run]
s = AN["summary"]
seq = int(sys.argv[3]) if len(sys.argv) > 3 else max(e["seq"] for e in AN["chunks"] if e["t"] <= (s["t_first_close_cmd"] or 10 ** 9))
e = next(x for x in AN["chunks"] if x["seq"] == seq)
OUT = pathlib.Path(f"${AG3S_PROJECT_ROOT}/outputs/verify/T28/{cond}_rec")
rows = {json.loads(l)["seq"]: json.loads(l) for l in open(OUT / run / "frames.jsonl") if '"planning"' in l}
pr = rows[seq]
m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m); scene = TransportScene.attach(m, d)
d.qpos[:] = pr["qpos"]; mujoco.mj_forward(m, d)
z = np.load(OUT / e["server_chunk"], allow_pickle=True)
sm = json.loads(str(z["summary_json"])); names = np.array(sm["sphere_link_names"])
Tb = scene.body_pose("base")
w = lambda P: np.asarray(P, float) @ Tb[:3, :3].T + Tb[:3, 3]
Cw = w(z["sphere_centres"]); R = z["sphere_radii"].astype(float)
tp = w(z["target_points"]) if z["target_points"].size else np.zeros((0, 3))
ball = (sm.get("esdf_stats") or {}).get("target_free", {})
ball = ball.get("ball") if isinstance(ball, dict) else None
apple = body_mesh_world(m, d, "apple", True)
meshes = {"ee_finger_l1": body_mesh_world(m, d, "ee_finger_l1", True), "ee_finger_l2": body_mesh_world(m, d, "ee_finger_l2", True),
          "EE_BODY_L": body_mesh_world(m, d, "EE_BODY_L", True)}
fc = w(z["filtered_cloud"]); ac = apple.mean(0)
near = fc[np.linalg.norm(fc - ac, axis=1) < 0.15]
rng = np.random.default_rng(0); near = near[rng.choice(len(near), min(len(near), 4000), replace=False)] if len(near) else near
mv = pr.get("max_violation_pair") or {}
wp = w(mv["point_m"]) if mv.get("point_m") is not None else None
col = {"ee_left": fs.INK_2, "ee_finger_l1": C[0], "ee_finger_l2": C[4]}
fig = plt.figure(figsize=(18, 6.5)); gs = fig.add_gridspec(1, 3)
side = {"condition": cond, "run": run, "seq": seq, "t": pr["t_step"], "server_chunk": e["server_chunk"],
        "finger_joints": sm.get("finger_joints"), "manipulated": sm.get("manipulated"), "exclusion": sm.get("exclusion"),
        "ball": ball, "worst_row": {k: mv.get(k) for k in ("link", "tier", "obstacle", "clearance_m")},
        "worst_nearest_body": e["worst"].get("nearest_body"), "worst_inside_apple": e["worst"].get("inside_apple"),
        "gate": pr.get("gate"), "latch": e.get("grasp_latch"), "apple_centre_world": ac.tolist(),
        "n_target_points": int(len(tp)), "n_cloud_points_within_0.15m_of_apple_drawn": int(len(near))}
for j, (a, b, lab) in enumerate(((0, 1, "top view x-y (world)"), (0, 2, "side view x-z (world)"))):
    ax = fig.add_subplot(gs[0, j])
    ax.scatter(near[:, a], near[:, b], s=1, color=fs.GRID_INK, alpha=0.6, label="filtered cloud (<0.15 m of apple)")
    ax.scatter(apple[:, a], apple[:, b], s=2, color=C[1], alpha=0.35, label="apple mesh (GT)")
    for n, Vm in meshes.items():
        ax.scatter(Vm[::3, a], Vm[::3, b], s=1, color=C[2] if n == "EE_BODY_L" else col[n], alpha=0.2, label=f"{n} mesh (GT)")
    if len(tp):
        ax.scatter(tp[:, a], tp[:, b], s=6, color=C[3], marker="^", label=f"manipulated target points ({len(tp)})")
    if ball:
        bc = w(ball["centre_m"]); r = ball["radius_mm"] / 1000
        ax.add_patch(Circle((bc[a], bc[b]), r, fill=False, ls="--", lw=1.5, ec=C[7]))
        ax.plot([], [], ls="--", color=C[7], label=f"exclusion ball r {ball['radius_mm']:.1f} mm")
        if b == 2:
            zc = w([ball["centre_m"][0], ball["centre_m"][1], ball["z_min_m"]])[2]
            ax.axhline(zc, color=C[7], lw=0.8, ls=":", label="ball z_min cut")
    for L in ("ee_left", "ee_finger_l1", "ee_finger_l2"):
        for c, r in zip(Cw[names == L], R[names == L]):
            ax.add_patch(Circle((c[a], c[b]), r, fill=True, alpha=0.5, lw=0.6, ec=col[L], fc=col[L]))
        ax.plot([], [], "o", color=col[L], label=f"{L} constraint spheres ({(names == L).sum()})")
    if wp is not None:
        ax.plot(wp[a], wp[b], "x", color=C[7], ms=10, mew=2, label=f"worst row {mv.get('link')} {1000 * mv.get('clearance_m', 0):.1f} mm")
    ctr = np.vstack([apple, *meshes.values()]).mean(0)
    ax.set_xlim(ctr[a] - 0.13, ctr[a] + 0.13); ax.set_ylim(ctr[b] - 0.13, ctr[b] + 0.13); ax.set_aspect("equal")
    ax.set_title(f"{cond} {run} t={pr['t_step']} (seq {seq}): {lab}", fontsize=10); ax.set_xlabel("xyz"[a] + " (m)"); ax.set_ylabel("xyz"[b] + " (m)")
    if j == 0:
        ax.legend(fontsize=6.5, loc="lower left", markerscale=1.5)
    fs.style_axes(fig, [ax])
ax = fig.add_subplot(gs[0, 2])
rd = iio.get_reader(s["mp4"]); n = rd.count_frames(); k = min(max(pr["t_step"] - 1, 0), n - 1)
ax.imshow(rd.get_data(k)[40:420, 60:600]); rd.close(); ax.set_xticks([]); ax.set_yticks([])
fjl = (sm.get("finger_joints") or {}).get("opening_norm", {}).get("left")
ax.set_title(f"third-person mp4 frame {k}; finger model opening {fjl:.2f}, exclusion {(sm.get('exclusion') or {}).get('source')}", fontsize=9)
side["mp4"] = s["mp4"]; side["mp4_frame"] = k
fig.suptitle(f"T28 P3 real scene just before the grasp — gripper constraint spheres (T29 finger joints, scale 0.05), apple, exclusion geometry", fontsize=11)
fig.patch.set_facecolor(fs.SURFACE); fig.tight_layout()
p = F / f"t28-p3-scene-pregrasp-{cond}.png"
assert not p.exists(), p
fig.savefig(p, dpi=120); plt.close(fig)
p.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float)); print(p)
