"""T43 HX real-scene figure (verifier, CPU only).
usage: T43W_SNAP=<snap> PYTHONPATH=${AG3S_PROJECT_ROOT} fig_scene_hx.py OUT_PNG ARM KEY T_CARRY T_RETURN
Row 1 = carry row T_CARRY, row 2 = return row T_RETURN of KEY (raw/hx_<ARM>.json).
  col 1: MuJoCo render of M(t) (measured post state); col 2: same with both arms at C(t) = P(t) (apple moved rigidly with the EE);
         free camera looking along the bar axis (azimuth = bar axis + 180, elevation -12 deg, distance 0.9 m, lookat = EE of M(t)).
         markers: EE body positions of C (orange) and M (blue) for the window rows t-12..t.
  col 3: the crossing seen along the bar axis (world y-z): per window row, the point of the group geometry (carry: apple_geom,
         return: finger collision geoms) nearest to the hurdle (mj_geomDistance fromto, group side) for C (orange), M (blue) and the
         servo prediction S (green dashed); bar = circle r 12 mm at its axis, dotted ring = bar + 10 mm obstacle margin.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, pathlib, sys
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p')); sys.path.insert(0, _ag3s_workspace_path(''))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np, mujoco  # noqa: E402
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
import hold_w as HW  # noqa: E402
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

out_png = pathlib.Path(sys.argv[1]); arm, key = sys.argv[2], sys.argv[3]; tc, tr = int(sys.argv[4]), int(sys.argv[5])
assert not out_png.exists(), f"{out_png} exists"
RAW = json.load(open(f"{_ag3s_workspace_path()}/outputs/verify/T43/HX/raw/hx_{arm}.json"))["runs"][key]
rows = {r["t"]: r for r in RAW["rows"]}
cl = HW.load_client(); kor = figstyle.use_korean()
BLUE, ORANGE, GREEN = figstyle.CATEGORICAL[0], figstyle.CATEGORICAL[1], figstyle.CATEGORICAL[2]


def hexrgba(h, a=1.0):
    h = h.lstrip("#"); return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [a], np.float32)


ANA = {"V7g": _ag3s_workspace_path('outputs/verify/T43/CT/raw/analysis_v7g.json'), "V8": _ag3s_workspace_path('outputs/verify/T43/V8/analysis/analysis.json'),
       "V5": _ag3s_workspace_path('outputs/verify/T43/W/analysis/analysis.json')}
a = json.load(open(ANA[arm]))["runs"][key]; rd = pathlib.Path(a["run_dir"])
man = json.load(open(rd / "manifest.json")); prof = man["scene"].get("obstacle_profile"); ep = man["scene"]["episode_index"]
z = np.load(rd / "traj.npz", allow_pickle=True); seed = int(z["policy_seed"])
stt = rd.parent / f"{rd.name}.start"
cfg = next((l.split("=", 1)[1].split(":", 1)[0] for l in stt.read_text().splitlines() if l.startswith("obstacle=") and l != "obstacle=:"), None)
L = HW.reset(cl, ep, seed, prof, cfg); m, d = L["m"], L["d"]; hurdle = sorted(L["obstacle_manager"].collision_geom_ids)
side = RAW["side"]; SD = side[0].upper()
qL = np.asarray(L["idx"]["left_q"][:7], int); qR = np.asarray(L["idx"]["right_q"][:7], int)
ee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"EE_BODY_{SD}")
ab = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple"); aq = int(m.jnt_qposadr[m.body_jntadr[ab]])
fr = [json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()]; ctrl = {r["t_step"]: r for r in fr if r["kind"] == "control"}
EA = np.asarray(z["executed_actions"], float); MQ = np.asarray(z["measured_qpos"], float)
bar = next(g for g in hurdle if "bar" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""))


def qM_of(t):
    if t + 1 in ctrl:
        return np.asarray(ctrl[t + 1]["qpos"], float)
    q = np.asarray(ctrl[t]["qpos"], float).copy(); q[qL] = MQ[t, 0:7]; q[qR] = MQ[t, 8:15]; return q


def qC_of(t, qM, carry):
    d.qpos[:] = qM; mujoco.mj_kinematics(m, d)
    T0 = np.eye(4); T0[:3, :3] = d.xmat[ee].reshape(3, 3); T0[:3, 3] = d.xpos[ee]
    q = qM.copy(); q[qL] = EA[t, 0:7]; q[qR] = EA[t, 8:15]
    if carry:
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        T1 = np.eye(4); T1[:3, :3] = d.xmat[ee].reshape(3, 3); T1[:3, 3] = d.xpos[ee]
        Ra = np.zeros(9); mujoco.mju_quat2Mat(Ra, qM[aq + 3:aq + 7]); Ta = np.eye(4); Ta[:3, :3] = Ra.reshape(3, 3); Ta[:3, 3] = qM[aq:aq + 3]
        Tn = T1 @ np.linalg.inv(T0) @ Ta; qa = np.zeros(4); mujoco.mju_mat2Quat(qa, Tn[:3, :3].reshape(-1)); q[aq:aq + 3] = Tn[:3, 3]; q[aq + 3:aq + 7] = qa
    return q


figstyle.use_korean()
fig, axes = plt.subplots(2, 3, figsize=(16, 9.6), gridspec_kw={"width_ratios": [1.25, 1.25, 1.0]})
side_json = {"script": __file__, "arm": arm, "key": key, "snapshot": str(HW.HY), "panels": []}
d.qpos[:] = qM_of(tc); mujoco.mj_forward(m, d)
axis = d.geom_xmat[bar].reshape(3, 3)[:, 2]; az = float(np.degrees(np.arctan2(axis[1], axis[0]))); barc = d.geom_xpos[bar].copy()
for r_i, (t, ph) in enumerate(((tc, "carry"), (tr, "return"))):
    R = rows[t]; qM = qM_of(t); qC = qC_of(t, qM, ph == "carry")
    d.qpos[:] = qM; mujoco.mj_forward(m, d); look = d.xpos[ee].copy()
    ts = [tt for tt in range(t - 12, t + 1) if tt in rows]
    marks = [(rows[tt]["ee_C"], 0.006 if tt == t else 0.0035, hexrgba(ORANGE)) for tt in ts] + \
            [(rows[tt]["ee_M"], 0.006 if tt == t else 0.0035, hexrgba(BLUE)) for tt in ts]
    grp = "apple" if ph == "carry" else "fingers"
    for col, (q, lab, gd) in enumerate(((qM, "M(t) measured", R["geo_M"]), (qC, "C(t) = P(t) commanded / planned", R["geo_C"]))):
        d.qpos[:] = q; mujoco.mj_forward(m, d)
        rr = mujoco.Renderer(m, 480, 640); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = look; cam.azimuth = az + 180; cam.elevation = -12; cam.distance = 0.9
        rr.update_scene(d, camera=cam)
        for cc, rad, rgba in marks:
            s = rr.scene
            if s.ngeom < s.maxgeom:
                mujoco.mjv_initGeom(s.geoms[s.ngeom], int(mujoco.mjtGeom.mjGEOM_SPHERE), np.array([rad, 0, 0]), np.asarray(cc, float), np.eye(3).reshape(-1), rgba); s.ngeom += 1
        img = rr.render(); rr.close()
        ax = axes[r_i, col]; ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        txt = "  ".join(f"{g}: {v:+.1f}" for g, v in gd.items())
        ax.set_title(f"{key.split('/')[-1]} {ph} t={t} k={R['k']}\n{lab} - bar distance mm: {txt}", fontsize=8.5)
    # path panel (world y-z, looking along bar axis x)
    ax = axes[r_i, 2]
    win = [rows[tt] for tt in sorted(rows) if rows[tt]["phase"] == ph and rows[tt]["in_window"]]
    gg = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "apple_geom")] if ph == "carry" else \
        [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"ee_finger_{side[0]}1"), mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"ee_finger_{side[0]}2")) and (m.geom_contype[g] or m.geom_conaffinity[g])]
    ft = np.zeros(6)

    def near_pt(q):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d); best = (np.inf, None)
        for g1 in gg:
            for g2 in hurdle:
                v = mujoco.mj_geomDistance(m, d, int(g1), int(g2), 0.5, ft)
                if v < best[0]:
                    best = (v, ft[:3].copy())
        return best
    paths = {"C": [], "M": [], "S": []}
    for w in win:
        qm = qM_of(w["t"]); paths["M"].append(near_pt(qm)); paths["C"].append(near_pt(qC_of(w["t"], qm, ph == "carry")))
        if "pred_q14" in w:
            a16 = EA[w["t"]].copy(); a16[0:7] = w["pred_q14"][:7]; a16[8:15] = w["pred_q14"][7:]
            d.qpos[:] = qm; mujoco.mj_kinematics(m, d); T0 = np.eye(4); T0[:3, :3] = d.xmat[ee].reshape(3, 3); T0[:3, 3] = d.xpos[ee]
            qs = qm.copy(); qs[qL] = a16[0:7]; qs[qR] = a16[8:15]
            if ph == "carry":
                d.qpos[:] = qs; mujoco.mj_kinematics(m, d); T1 = np.eye(4); T1[:3, :3] = d.xmat[ee].reshape(3, 3); T1[:3, 3] = d.xpos[ee]
                Ra = np.zeros(9); mujoco.mju_quat2Mat(Ra, qm[aq + 3:aq + 7]); Ta = np.eye(4); Ta[:3, :3] = Ra.reshape(3, 3); Ta[:3, 3] = qm[aq:aq + 3]
                Tn = T1 @ np.linalg.inv(T0) @ Ta; qa = np.zeros(4); mujoco.mju_mat2Quat(qa, Tn[:3, :3].reshape(-1)); qs[aq:aq + 3] = Tn[:3, 3]; qs[aq + 3:aq + 7] = qa
            paths["S"].append(near_pt(qs))
    for nm, col_, ls, lw, lab in (("C", ORANGE, "-", 1.4, "C = P (command / plan)"), ("M", BLUE, "-", 2.6, "M (measured)"), ("S", GREEN, "--", 1.2, "S (servo model, offline)")):
        P_ = np.asarray([p_[1] for p_ in paths[nm]], float)
        if len(P_):
            ax.plot(P_[:, 1], P_[:, 2], ls, color=col_, lw=lw, marker="o", ms=2.5, label=lab)
    j_ = [i for i, w in enumerate(win) if w["t"] == t]
    if j_:
        for nm, col_ in (("C", ORANGE), ("M", BLUE)):
            ax.plot(paths[nm][j_[0]][1][1], paths[nm][j_[0]][1][2], "s", color=col_, ms=8, mfc="none", mew=1.5)
    th = np.linspace(0, 2 * np.pi, 100)
    ax.fill(barc[1] + 0.012 * np.cos(th), barc[2] + 0.012 * np.sin(th), color=figstyle.INK_2, alpha=0.8, label="bar (r 12 mm)")
    ax.plot(barc[1] + 0.022 * np.cos(th), barc[2] + 0.022 * np.sin(th), ":", color=figstyle.INK_2, lw=1, label="bar + 10 mm margin")
    ax.set_aspect("equal"); ax.set_xlabel("world y (m)"); ax.set_ylabel("world z (m)"); ax.invert_xaxis()
    ax.set_title(f"{ph} window rows {win[0]['t']}-{win[-1]['t']} (n={len(win)}): {'apple' if ph == 'carry' else 'finger'} point nearest the hurdle\n(square = row {t})", fontsize=8.5)
    ax.legend(fontsize=7, loc="best"); figstyle.style_axes(fig, [ax])
    side_json["panels"].append({"phase": ph, "t": t, "k": R["k"], "geo_M_mm": R["geo_M"], "geo_C_mm": R["geo_C"], "geo_S_mm": R.get("geo_S"),
                                "servo_err_mm": R.get("servo_err_mm"), "cmd_err_mm": R.get("cmd_err_mm"),
                                "camera": {"lookat": look.tolist(), "azimuth": az + 180, "elevation": -12, "distance": 0.9},
                                "window_rows": [w["t"] for w in win], "nearest_group": "apple" if ph == "carry" else "fingers",
                                "nearest_pt_C": [p_[1].tolist() for p_ in paths["C"]], "nearest_pt_M": [p_[1].tolist() for p_ in paths["M"]],
                                "nearest_pt_S": [p_[1].tolist() for p_ in paths["S"]], "dist_C_mm": [1e3 * p_[0] for p_ in paths["C"]],
                                "dist_M_mm": [1e3 * p_[0] for p_ in paths["M"]], "dist_S_mm": [1e3 * p_[0] for p_ in paths["S"]], "bar_centre_world": barc.tolist(), "bar_axis_world": axis.tolist()})
fig.suptitle(f"T43 HX - hurdle crossing, {key}: measured M vs commanded/planned P (= C) and servo-model prediction S", fontsize=11)
fig.tight_layout(); fig.savefig(out_png, dpi=125)
json.dump(side_json, open(out_png.with_suffix(".json"), "w"), indent=1)
print("wrote", out_png)
