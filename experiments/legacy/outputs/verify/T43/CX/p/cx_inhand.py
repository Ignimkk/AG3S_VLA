"""T43 CX — in-hand geometry of the apple per control row, V11AB vs V11ABC (verifier, CPU only, numbers only).
usage: T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v11 cx_inhand.py OUT_JSON OUT_NPZ_DIR

State of row t = the run's control row t 'qpos' (full MuJoCo qpos incl. the apple free joint; the CT replays reproduce it, qpos_mismatch_rows 0),
model = hurdle_metrics.model() (model_transport_pick_place_obstacles.xml of the snapshot).
Per row t from grasp_t - 16 to min(t_sep + 6, 599) (t_sep, grasp_t from raw/cx_cases.json):
  finger bodies: MJCF bodies ee_finger_l1 / ee_finger_l2 (MJCF names; the URDF names of the left fingers are swapped, T43 OC — not used here).
  gap_mm         |p(l2) - p(l1)| (body origins);  axis u = (p(l2) - p(l1)) / gap.
  off_axis_mm    (apple - mid) . u  (apple centre along the closing axis; 0 = centred between the fingers)
  off_app_mm     (apple - mid) . a, a = site left_ee z axis (approach axis) ;  off_perp_mm = the rest (norm).
  d_l1_mm / d_l2_mm   mj_geomDistance(apple_geom, nearest collision geom of that finger body) (< 0 = penetration), distmax 50 mm.
  rot_deg        rotation of the apple relative to the left_ee frame since row grasp_t (angle of R_ee(t)^T R_apple(t) vs the same at grasp_t).
  tcp_vz, tcp_az measured TCP (site left_ee) vertical velocity (m/s) and acceleration (m/s^2) from the row states (differences x 15, x 15^2)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, pathlib
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p'))
import hurdle_metrics as HM  # noqa: E402

X = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/CX/raw')); V11 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V11'))


def rot_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def main():
    C = json.load(open(X / "cx_cases.json"))["cases"]; nd = pathlib.Path(sys.argv[2]); nd.mkdir(parents=True, exist_ok=True)
    HM.set_arm("left"); M = HM.model(); m, d = M["m"], M["d"]
    B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    coll = lambda g: bool(m.geom_contype[g] or m.geom_conaffinity[g])
    fg = {n: [g for g in range(m.ngeom) if coll(g) and int(m.geom_bodyid[g]) == B(n)] for n in ("ee_finger_l1", "ee_finger_l2")}
    ag = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "apple_geom"); ab = B("apple")
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_ee")
    ft = np.zeros(6)
    res = {"definitions": __doc__, "finger_geoms": {k: v for k, v in fg.items()}, "cases": {}}
    for ck, c in C.items():
        res["cases"][ck] = {}
        for arm in ("V11AB", "V11ABC"):
            gt = c["arms"][arm]["timeline"].get("grasp_t_used"); ts = c["arms"][arm]["timeline"].get("t_sep")
            if gt is None:
                res["cases"][ck][arm] = None; continue
            rd = pathlib.Path(json.load(open(V11 / f"analysis/analysis_{arm}.json"))["runs"][c["arms"][arm]["key"]]["run_dir"])
            ctrl = {r["t_step"]: r for r in (json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()) if r["kind"] == "control"}
            t0 = max(int(gt) - 16, 0); t1 = min((ts if ts is not None else int(gt) + 120) + 6, 599)
            rows = []; R0 = None
            for t in range(t0, t1 + 1):
                d.qpos[:] = np.asarray(ctrl[t]["qpos"], float); mujoco.mj_forward(m, d)
                p1, p2 = d.xpos[B("ee_finger_l1")].copy(), d.xpos[B("ee_finger_l2")].copy(); mid = 0.5 * (p1 + p2)
                gap = np.linalg.norm(p2 - p1); u = (p2 - p1) / gap
                Re = d.site_xmat[site].reshape(3, 3).copy(); a = Re[:, 2]
                ap = d.xpos[ab].copy(); Ra = d.xmat[ab].reshape(3, 3).copy(); v = ap - mid
                rel = Re.T @ Ra
                if t == int(gt):
                    R0 = rel
                dl = {n: min(mujoco.mj_geomDistance(m, d, ag, g, 0.05, ft) for g in gs) for n, gs in fg.items()}
                rows.append({"t": t, "gap_mm": 1e3 * gap, "off_axis_mm": 1e3 * float(v @ u), "off_app_mm": 1e3 * float(v @ a),
                             "off_perp_mm": 1e3 * float(np.linalg.norm(v - (v @ u) * u - (v @ a) * a)),
                             "d_l1_mm": 1e3 * dl["ee_finger_l1"], "d_l2_mm": 1e3 * dl["ee_finger_l2"],
                             "rel_R": rel, "tcp": d.site_xpos[site].copy()})
            for r in rows:
                r["rot_deg"] = rot_angle(R0.T @ r.pop("rel_R")) if R0 is not None else None
            tcp = np.array([r.pop("tcp") for r in rows]); vz = np.r_[np.nan, np.diff(tcp[:, 2]) * 15]; az = np.r_[np.nan, np.diff(vz) * 15]
            for r, x, y in zip(rows, vz, az):
                r["tcp_vz"] = float(x); r["tcp_az"] = float(y)
            np.savez_compressed(nd / f"inhand_{arm}_{ck.replace(':', '_')}.npz", **{k: np.array([r[k] if r[k] is not None else np.nan for r in rows], float) for k in rows[0]})
            at = lambda k, tt: next((r[k] for r in rows if r["t"] == tt), None)
            summ = {"grasp_t": gt, "t_sep": ts}
            for lab, tt in (("at_grasp", int(gt)), ("at_grasp_p8", int(gt) + 8), ("at_tsep_m8", None if ts is None else ts - 8), ("at_tsep_m2", None if ts is None else ts - 2)):
                summ[lab] = None if tt is None else {k: at(k, tt) for k in ("gap_mm", "off_axis_mm", "off_app_mm", "off_perp_mm", "d_l1_mm", "d_l2_mm", "rot_deg", "tcp_vz", "tcp_az")}
            if ts is not None:
                pre = [r for r in rows if int(gt) <= r["t"] < ts]
                summ["pre_tsep_tcp_az_max"] = float(np.nanmax([r["tcp_az"] for r in pre])) if pre else None
                summ["pre_tsep_tcp_vz_max"] = float(np.nanmax([r["tcp_vz"] for r in pre])) if pre else None
                last = [r for r in rows if ts - 6 <= r["t"] <= ts]
                summ["tsep_m6_to_0"] = [{k: (round(r[k], 3) if isinstance(r[k], float) else r[k]) for k in ("t", "d_l1_mm", "d_l2_mm", "off_axis_mm", "off_app_mm", "rot_deg", "tcp_vz", "tcp_az")} for r in last]
                fl = next((r["t"] for r in rows if r["t"] >= int(gt) and (r["d_l1_mm"] > 0.5 or r["d_l2_mm"] > 0.5)), None)
                summ["first_finger_gap_gt_0p5mm"] = fl
                summ["first_finger_gap_which"] = None if fl is None else [k for k in ("d_l1_mm", "d_l2_mm") if at(k, fl) > 0.5]
            res["cases"][ck][arm] = summ
            print(ck, arm, json.dumps({k: summ[k] for k in ("at_grasp", "at_tsep_m2") if summ.get(k)}, default=lambda x: round(x, 2))[:600], flush=True)
    pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
