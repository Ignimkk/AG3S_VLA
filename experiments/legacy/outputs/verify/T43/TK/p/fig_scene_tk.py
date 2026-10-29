"""T43 TK real-scene figure (verifier, CPU only).  usage: T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v8 fig_scene_tk.py OUT_PNG KEY:T [KEY:T ...]
Per contact (row t of raw/tk_V8.json 'contacts'): MuJoCo render with a free camera looking along the bar axis (azimuth = bar axis
direction, elevation -8 deg, lookat = EE of M(t)).
  left  panel: actual state M(t) (control row t+1 qpos, or row t qpos + measured arm at an obstacle-stop row)
  right panel: the same state with the active arm at P(t) = refined row k (= C(t)); apple moved rigidly with the EE (carry contacts)
  markers (both panels): EE_BODY positions of P (orange) and M (blue) for rows t-10..t, larger marker at row t.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, pathlib, sys
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p')); sys.path.insert(0, _ag3s_workspace_path(''))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np, mujoco  # noqa: E402
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
import hold_w as HW  # noqa: E402
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

out_png = pathlib.Path(sys.argv[1]); specs = [s.rsplit(":", 1) for s in sys.argv[2:]]
assert not out_png.exists(), f"{out_png} exists"
TK = json.load(open(_ag3s_workspace_path('outputs/verify/T43/TK/raw/tk_V8.json')))
cl = HW.load_client(); kor = figstyle.use_korean()
ORANGE = figstyle.CATEGORICAL[1]; BLUE = figstyle.CATEGORICAL[0]


def hexrgba(h, a=1.0):
    h = h.lstrip("#"); return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [a], np.float32)


fig, axes = plt.subplots(len(specs), 2, figsize=(12, 4.6 * len(specs)))
axes = np.atleast_2d(axes); side_json = {"script": __file__, "snapshot": str(HW.HY), "panels": []}
for row, (key, t) in enumerate(specs):
    t = int(t); R = TK["runs"][key]; rd = pathlib.Path(R["run_dir"]); c = next(x for x in R["contacts"] if x["t"] == t)
    man = json.load(open(rd / "manifest.json")); prof = man["scene"].get("obstacle_profile"); ep = man["scene"]["episode_index"]
    z = np.load(rd / "traj.npz", allow_pickle=True); seed = int(z["policy_seed"])
    stt = rd.parent / f"{rd.name}.start"
    cfg = next((l.split("=", 1)[1].split(":", 1)[0] for l in stt.read_text().splitlines() if l.startswith("obstacle=") and l != "obstacle=:"), None)
    L = HW.reset(cl, ep, seed, prof, cfg); m, d = L["m"], L["d"]; hurdle = sorted(L["obstacle_manager"].collision_geom_ids)
    side = R["side"]; qadr = np.asarray(L["idx"][f"{side}_q"][:7], int); c0 = 0 if side == "left" else 8
    ee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "EE_BODY_L" if side == "left" else "EE_BODY_R")
    ab = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple"); aq = int(m.jnt_qposadr[m.body_jntadr[ab]])
    rows = [json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()]; ctrl = {r["t_step"]: r for r in rows if r["kind"] == "control"}
    MQ = np.asarray(z["measured_qpos"], float)
    if t + 1 in ctrl:
        qM = np.asarray(ctrl[t + 1]["qpos"], float)
    else:
        qM = np.asarray(ctrl[t]["qpos"], float); qM[qadr] = MQ[t, c0:c0 + 7]
    bar = next(g for g in hurdle if "bar" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""))
    d.qpos[:] = qM; mujoco.mj_forward(m, d)
    axis = d.geom_xmat[bar].reshape(3, 3)[:, 2]; az = float(np.degrees(np.arctan2(axis[1], axis[0])))
    T0 = np.eye(4); T0[:3, :3] = d.xmat[ee].reshape(3, 3); T0[:3, 3] = d.xpos[ee]; look = d.xpos[ee].copy()
    qP = qM.copy()
    # P arm joints: refined row k re-read from the server chunk of the control row's chunk_seq
    an = json.load(open(_ag3s_workspace_path('outputs/verify/T43/V8/analysis/analysis.json')))["runs"][key]
    seg = an["server_segment"]; first = pathlib.Path(seg["first"]); i0 = int(first.stem.split("_")[1]); seq = int(ctrl[t]["chunk_seq"])
    for n in range(int(seg["n"])):
        p = first.parent / f"chunk_{i0 + n:05d}.npz"
        with np.load(p, allow_pickle=True) as zz:
            if int(zz["t_step"]) == seq:
                Pk = np.asarray(zz["refined_chunk"], float)[c["k"], c0:c0 + 7]; break
    qP[qadr] = Pk
    if c["kind"] == "apple":
        d.qpos[:] = qP; mujoco.mj_kinematics(m, d)
        T1 = np.eye(4); T1[:3, :3] = d.xmat[ee].reshape(3, 3); T1[:3, 3] = d.xpos[ee]
        Ra = np.zeros(9); mujoco.mju_quat2Mat(Ra, qM[aq + 3:aq + 7]); Ta = np.eye(4); Ta[:3, :3] = Ra.reshape(3, 3); Ta[:3, 3] = qM[aq:aq + 3]
        Tn = T1 @ np.linalg.inv(T0) @ Ta; qa = np.zeros(4); mujoco.mju_mat2Quat(qa, Tn[:3, :3].reshape(-1)); qP[aq:aq + 3] = Tn[:3, 3]; qP[aq + 3:aq + 7] = qa
    ts = list(range(max(0, t - 10), t + 1))
    marks = [(R["rows"][tt]["EE_P"], 0.006 if tt == t else 0.0035, hexrgba(ORANGE)) for tt in ts if R["rows"][tt]["EE_P"]] + \
            [(R["rows"][tt]["EE_M"], 0.006 if tt == t else 0.0035, hexrgba(BLUE)) for tt in ts]
    for col, (q, lab) in enumerate(((qM, "M(t) measured" if not kor else "M(t) 측정"), (qP, "P(t)=C(t) planned/commanded" if not kor else "P(t)=C(t) 계획·명령"))):
        d.qpos[:] = q; mujoco.mj_forward(m, d)
        rr = mujoco.Renderer(m, 480, 640); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = look; cam.azimuth = az + 180; cam.elevation = -12; cam.distance = 0.9
        rr.update_scene(d, camera=cam)
        for cc, r, rgba in marks:
            s = rr.scene
            if s.ngeom < s.maxgeom:
                mujoco.mjv_initGeom(s.geoms[s.ngeom], int(mujoco.mjtGeom.mjGEOM_SPHERE), np.array([r, 0, 0]), np.asarray(cc, float), np.eye(3).reshape(-1), rgba); s.ngeom += 1
        img = rr.render(); rr.close()
        ax = axes[row, col]; ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        dist = c["bar_M_mm"] if col == 0 else c["bar_P_mm"]
        ax.set_title(f"{key.split('/')[-1]}  t={t} k={c['k']} ({c['kind']}:{c['body']})\n{lab}: bar distance {dist:+.1f} mm", fontsize=9)
    side_json["panels"].append({"key": key, "t": t, "k": c["k"], "kind": c["kind"], "body": c["body"], "bar_M_mm": c["bar_M_mm"],
                                "bar_P_mm": c["bar_P_mm"], "CM_mm": c["CM_mm"], "camera": {"lookat": look.tolist(), "azimuth": az + 180, "elevation": -12, "distance": 0.9},
                                "marker_rows": ts, "EE_P": [R["rows"][tt]["EE_P"] for tt in ts], "EE_M": [R["rows"][tt]["EE_M"] for tt in ts]})
fig.suptitle("T43 TK - contact moment: measured M vs planned P (orange = P EE path, blue = M EE path, rows t-10..t)", fontsize=10)
fig.tight_layout(); fig.savefig(out_png, dpi=130)
json.dump(side_json, open(out_png.with_suffix(".json"), "w"), indent=1)
print("wrote", out_png)
