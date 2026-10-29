"""V-T45 scene figure: MuJoCo hand pose of a pre-grasp chunk with the palm spheres off (23) vs on (121) overlaid,
and world-frame projections with the target points, other cloud points and the nearest field surface.
usage: fig_scene.py   (CPU, MuJoCo osmesa)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import os, sys, json, argparse
os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T45/V/p'))
import numpy as np, mujoco, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from AG3S.ag3s.experiments.common import figstyle as FS
import vt45 as V
from AG3S.trajopt.experiments import ta_offline as TA

KO = FS.use_korean(); T = lambda ko, en: ko if KO else en
O = pathlib.Path(_ag3s_workspace_path('outputs/verify/T45/V'))
F = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t45'))
CAT = FS.CATEGORICAL
chunks = json.load(open(O / "agg_cmp_chunks.json"))
CASES = [("C1a_ep3_s31", 28, T("새 HOLD (on 에서 palm −17.8 mm)", "new HOLD under on (palm -17.8 mm)"))]
# a second, benign pre-grasp chunk: rel -1, palm worst under 'on' and no collision rows
for c in chunks:
    o = c["variants"]["PC"]["eval_under"]["PC"]
    if c["rel"] == -1 and o["worst_is_palm"] and o["n_collision"] == 0 and c["run"] != "C1a_ep3_s31":
        CASES.append((c["run"], c["seq"], T("palm 이 최악 행, HOLD 없음", "palm is worst row, no HOLD")))
        break
ns = argparse.Namespace(hm=False, no_plane_rows=False)
XML = None
fig = plt.figure(figsize=(18, 7.6))
meta = {}
for ci, (name, seq, tag) in enumerate(CASES):
    entry = V.load_entry(str(O / "index_c1a.json"), name)
    sides = {"base": V.Side(name, entry, ns, ()), "PC": V.Side(name, entry, ns, V.PC_ARGS)}
    run = sides["PC"].run
    inp, _ = run.inputs(seq)
    own = np.load(O / "raw" / "cmp" / f"{name}.cmp.npz")[f"s{seq}_PC_own"]
    traj = run.layout.chunk_to_trajectory(own)[:, :V.K]
    k = V.K - 1
    spheres = {}
    for v in ("base", "PC"):
        r = sides[v].run
        lin = r.meter(V.K); lin.set_joint_parameter_path(r.finger_path(inp["summary"], inp["ref_chunk"], V.K))
        cen = lin.sphere_states(traj, inp["q_now"])[0][k]
        pi = V.group_indices(r)["palm"]
        spheres[v] = (cen[pi], np.asarray(lin.robot_radii)[pi])
    # MuJoCo pose at step k
    xml = str(run.sargs.model_xml)
    M = mujoco.MjModel.from_xml_path(xml); D = mujoco.MjData(M)
    row = run.plan[seq]
    qpos = np.asarray(row["qpos"], float).copy()
    full = run.layout.full_q(traj[:, k:k + 1], inp["q_now"])[:, 0]
    for i, jn in enumerate(run.robot.joint_names):
        if jn in run.qadr:
            qpos[run.qadr[jn]] = full[i]
    D.qpos[:] = qpos; D.qvel[:] = 0; mujoco.mj_forward(M, D)
    body = "EE_BODY_L" if run.hand == "left" else "EE_BODY_R"
    bid = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, body)
    c_pc = spheres["PC"][0]
    look = c_pc.mean(0)
    meta[name] = {"seq": seq, "step": k, "palm_body_xpos": D.xpos[bid].tolist(), "palm_sphere_mean": look.tolist(),
                  "n_spheres": {v: int(len(spheres[v][1])) for v in spheres}}
    z = inp["z"]
    tp = np.asarray(z["target_points"], float)
    cloud = np.asarray(z["filtered_cloud"], float)
    sm = np.asarray(z["support_mask"], bool)
    ti = np.zeros(len(cloud), bool); ti[np.asarray(z["target_point_indices"], int)] = True
    near = np.linalg.norm(cloud - look, axis=1) < 0.20
    other = cloud[near & ~sm & ~ti]
    sup = cloud[near & sm]
    rr = mujoco.Renderer(M, 420, 560)
    imgs = []
    for mode in ("none", "base", "PC"):
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = look; cam.distance = 0.36; cam.azimuth = 150 if run.hand == "left" else 30; cam.elevation = -28
        rr.update_scene(D, cam)
        if mode != "none":
            cen, rad = spheres[mode]
            rgba = [0.16, 0.47, 0.84, 0.45] if mode == "base" else [0.92, 0.41, 0.20, 0.35]
            for c, r_ in zip(cen, rad):
                g = rr.scene.geoms[rr.scene.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r_, 0, 0]), c, np.eye(3).flatten(), np.array(rgba, np.float32))
                rr.scene.ngeom += 1
        imgs.append(rr.render().copy())
    rr.close()
    titles = [T("MuJoCo 손 자세 (step 7)", "MuJoCo hand pose (step 7)"), T(f"off: palm 구 {len(spheres['base'][1])} (r {spheres['base'][1].min()*1e3:.1f}–{spheres['base'][1].max()*1e3:.1f} mm)", f"off: {len(spheres['base'][1])} palm spheres"),
              T(f"on: palm 구 {len(spheres['PC'][1])} (r {spheres['PC'][1].min()*1e3:.1f}–{spheres['PC'][1].max()*1e3:.1f} mm)", f"on: {len(spheres['PC'][1])} palm spheres")]
    top = 0.97 - ci * 0.50
    for j in range(3):
        ax = fig.add_axes([0.01 + j * 0.205, top - 0.42, 0.2, 0.40]); ax.imshow(imgs[j]); ax.axis("off")
        ax.set_title(titles[j], fontsize=9, color=FS.INK)
    # projections
    for pj, (a, b, an, bn) in enumerate(((0, 1, "x", "y"), (0, 2, "x", "z"))):
        ax = fig.add_axes([0.63 + pj * 0.185, top - 0.40, 0.17, 0.34])
        ax.scatter(sup[:, a], sup[:, b], s=0.6, color=FS.GRID_INK, rasterized=True, label=T("지지면 점", "support pts"))
        ax.scatter(other[:, a], other[:, b], s=1.5, color=CAT[3], rasterized=True, label=T("기타 점", "other pts"))
        ax.scatter(tp[:, a], tp[:, b], s=1.5, color=CAT[2], rasterized=True, label="target")
        for v, col in (("base", CAT[0]), ("PC", CAT[1])):
            cen, rad = spheres[v]
            for c, r_ in zip(cen, rad):
                ax.add_patch(Circle((c[a], c[b]), r_, fill=False, lw=0.5 if v == "PC" else 0.9, color=col, alpha=0.8))
        ax.set_aspect("equal"); ax.set_xlim(look[a] - 0.12, look[a] + 0.12); ax.set_ylim(look[b] - 0.12, look[b] + 0.12)
        ax.set_xlabel(f"{an} (m)", fontsize=8); ax.set_ylabel(f"{bn} (m)", fontsize=8); ax.tick_params(labelsize=7)
        FS.style_axes(fig, [ax])
        if pj == 0:
            ax.legend(fontsize=6, frameon=False, loc="upper right", markerscale=4)
            ax.set_title(f"{name} seq {seq}: {tag}", fontsize=8.5, color=FS.INK, loc="left")
    z.close()
fig.text(0.01, 0.005, T("파랑 = off 의 palm 구, 주황 = on 의 palm 구 (같은 궤적 · 같은 자세에 두 모델을 얹었다). 지지면 = recorded support_mask 점. 투영은 palm 구 중심 ±12 cm.\n"
                       "chunk = 해당 run 의 close 직전 (rel −1), on 해(refined)의 8번째 step.", "Blue = palm spheres off, orange = on (both models on the same pose). Projections +-12 cm around the palm sphere centroid."),
         fontsize=8.5, color=FS.INK_2)
fig.savefig(F / "t45v-scene.png", dpi=130)
json.dump({"cases": [dict(run=n, seq=s, tag=t) for n, s, t in CASES], "meta": meta}, open(F / "t45v-scene.json", "w"), indent=1)
print("ok", meta)
