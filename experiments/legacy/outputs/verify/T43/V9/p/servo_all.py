"""T43 V9 servo prediction vs actual M on EXECUTED rows (verifier, CPU, recorded data only). Same method as T43/HX/p/hx_measure.py (servo.py ServoObserver over the
run's chunks in order, q_now = planning-row qpos, exec_feedback of the planning row; ServoModel.rollout(q, v, C rows of the chunk, tau) = predicted end-of-row arm joints),
but over ALL control rows of every run (H and B), not only rows from grasp_t.  M(t) = control row t+1 qpos (last row: row t qpos with the arms = measured_qpos[t]).
C(t) = executed_actions[t].  Per row: servo_err_mm = |EE_BODY(pred) - EE_BODY(M)| (active arm), cmd_err_mm = |EE_BODY(C) - EE_BODY(M)|, servo_err_joint_deg_max over the
active arm's 7 joints, cmd_err_joint_deg_max likewise.  Categories (hx_agg rule): stop = last row of a run whose M is not row t+1; hold = executed false; exec = the rest.
usage: servo_all.py ARM OUT_JSON   ARM = V8 | V9      (T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v8)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, pathlib, sys
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/DX/p'))
import numpy as np, mujoco  # noqa: E402
import hold_w as HW  # noqa: E402
import dx_common as C  # noqa: E402
from AG3S.trajopt.servo import ServoModel, ServoObserver, servo_params_from_xml  # noqa: E402
T43 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43'))
ANALYSIS = {"V8": T43 / "V8/analysis/analysis.json", "V9": T43 / "V9/analysis/analysis.json"}
XML = "${AG3S_PROJECT_ROOT}-t43v8/robots/rby1/description/models/rby1a/mujoco/model_transport.xml"


def main():
    arm, outp = sys.argv[1], pathlib.Path(sys.argv[2])
    runs = json.load(open(ANALYSIS[arm]))["runs"]; cl = HW.load_client()
    names = [f"left_arm_{j}" for j in range(7)] + [f"right_arm_{j}" for j in range(7)]
    sm = ServoModel(servo_params_from_xml(XML, names, control_hz=15.0)); res = {"definitions": __doc__, "arm": arm, "runs": {}}
    for key in sorted(runs):
        a = runs[key]; rd = pathlib.Path(a["run_dir"]); man = json.load(open(rd / "manifest.json")); prof = man["scene"].get("obstacle_profile"); ep = int(man["scene"]["episode_index"])
        z = np.load(rd / "traj.npz", allow_pickle=True); seed = int(z["policy_seed"]); stt = rd.parent / f"{rd.name}.start"
        cfg = next((l.split("=", 1)[1].split(":", 1)[0] for l in stt.read_text().splitlines() if l.startswith("obstacle=") and l != "obstacle=:"), None)
        L = HW.reset(cl, ep, seed, prof, cfg); m, d = L["m"], L["d"]
        side = "right" if ep == 1834 else "left"; SD = side[0].upper()
        qL = np.asarray(L["idx"]["left_q"][:7], int); qR = np.asarray(L["idx"]["right_q"][:7], int); q14 = np.r_[qL, qR]
        ee_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"EE_BODY_{SD}")
        sl = slice(0, 7) if side == "left" else slice(7, 14)
        def ee(q):
            d.qpos[:] = q; mujoco.mj_kinematics(m, d); return d.xpos[ee_b].copy()
        rows_f = [json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()]
        ctrl = {r["t_step"]: r for r in rows_f if r["kind"] == "control"}; plan = {r["seq"]: r for r in rows_f if r["kind"] == "planning"}
        seg = C.lit(a.get("server_segment")); chunks = C.segment_chunks(seg) if seg and seg != "None" else {}
        EA = np.asarray(z["executed_actions"], float); MQ = np.asarray(z["measured_qpos"], float); T = len(EA); last_t = T - 1
        obs = ServoObserver(sm); pred = {}
        for seq in sorted(plan):
            pr = plan[seq]; qn = np.asarray(pr["qpos"], float)[q14]; fb = pr.get("exec_feedback") or {}
            if not (isinstance(fb, dict) and fb.get("applied_arm") is not None) and seq in chunks:
                with np.load(chunks[seq], allow_pickle=True) as zz:
                    fb = (json.loads(str(zz["summary_json"])).get("exec_feedback") or {})
            if isinstance(fb, dict) and isinstance(fb.get("applied_arm"), str):
                fb = dict(fb); fb["applied_arm"] = C.lit(fb["applied_arm"])
            st_ = obs.update(qn, fb, seq)
            rows_c = sorted(t for t, r in ctrl.items() if r["chunk_seq"] == seq)
            if not rows_c:
                continue
            U = np.stack([np.r_[EA[t, 0:7], EA[t, 8:15]] for t in rows_c], axis=1); ro = sm.rollout(st_.q, st_.v, U, st_.tau)
            for j, t in enumerate(rows_c):
                pred[t] = (ro["q"][:, j], st_.source)
        out = []
        for t in range(T):
            if t not in pred or t not in ctrl:
                continue
            r = ctrl[t]; ex = bool(r.get("executed", True)); q_pre = np.asarray(r["qpos"], float)
            if t + 1 in ctrl:
                qM = np.asarray(ctrl[t + 1]["qpos"], float); msrc = "row t+1 qpos"
            else:
                qM = q_pre.copy(); qM[qL] = MQ[t, 0:7]; qM[qR] = MQ[t, 8:15]; msrc = "row t qpos + measured arms"
            def with_arms(a14):
                q = qM.copy(); q[qL] = a14[0:7]; q[qR] = a14[7:14]; return q
            pq = pred[t][0]; c14 = np.r_[EA[t, 0:7], EA[t, 8:15]]; jm = np.r_[qM[qL], qM[qR]]
            eM = ee(qM); eS = ee(with_arms(pq)); eC = ee(with_arms(c14))
            cat = "stop" if (t == last_t and msrc != "row t+1 qpos") else ("hold" if not ex else "exec")
            out.append({"t": t, "cat": cat, "servo_err_mm": 1e3 * float(np.linalg.norm(eS - eM)), "cmd_err_mm": 1e3 * float(np.linalg.norm(eC - eM)),
                        "servo_err_joint_deg": float(np.degrees(np.abs(pq - jm)[sl]).max()), "cmd_err_joint_deg": float(np.degrees(np.abs(c14 - jm)[sl]).max()), "src": pred[t][1]})
        res["runs"][key] = {"grasp_t": a.get("grasp_t"), "n_rows": len(out), "side": side, "rows": out}
        print(key, len(out), flush=True)
        outp.write_text(json.dumps(res))
if __name__ == "__main__":
    main()
