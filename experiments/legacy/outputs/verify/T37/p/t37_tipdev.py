"""T37 — fingertip deviation refined - reference per chunk, with the grasp-latch state and the continuity gate record.

usage: MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} <shim py> t37_tipdev.py COND [SRC_TAG]  -> p/<COND>_tipdev.json
       SRC_TAG (default T37) selects outputs/verify/<SRC_TAG>/{COND}_rec + p/<COND>_analysis.json (so T34 runs can be measured
       with the same code; output p/<COND>_tipdev_<SRC_TAG>.json when SRC_TAG != T37).

Per paired chunk (p/<COND>_analysis.json 'server_chunk'):
  state          : server summary grasp.state (latch state recorded with that chunk)
  gate           : to.metrics continuity_active / continuity_reason / continuity_off_by (T37 records; absent in T34 records)
  fingertip dev  : fixed_input_ablation.Meter.compare(refined, reference, q_now) as T35-diag b_ablation.py
                   ('recorded_refined_vs_reference'): fingertip = mean of the left finger constraint-sphere centres (FK),
                   finger joint path from summary.finger_joints.opening_norm + reference cols 7/15 (FingerJointMap.path);
                   disp per planned step (mm) = |tip(refined) - tip(reference)|; max over the planned horizon and over
                   executed rows 0-7; joint_dev_max_deg over arm joints and planned steps.
  settings       : fixed_input_ablation.parse_server_log + resolve_settings (server log + manifest), as b_ablation.setup.
  q_now          : paired local planning row qpos (T.q_now_from_qpos).
No solve is run; recorded chunks only."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))
import AG3S.trajopt.experiments.fixed_input_ablation as T  # noqa: E402
from AG3S.ag3s.robot_models.gripper_state import FingerJointMap  # noqa: E402

COND = sys.argv[1]
SRC = sys.argv[2] if len(sys.argv) > 2 else "T37"
V = pathlib.Path(f"${AG3S_PROJECT_ROOT}/outputs/verify/{SRC}")
OUTV = pathlib.Path(_ag3s_workspace_path('outputs/verify/T37/p'))
PORT = {"E3a": 8204, "E3b": 8206}


def setup(run_dir):
    from AG3S.trajopt.limits import build_limits
    from AG3S.trajopt.types import ChunkLayout
    rec = V / f"{COND}_rec"
    port = PORT[COND]
    args = T.build_parser().parse_args(["--constraints", str(rec / f"server_{port}/server_constraints/run_0000"), "--frames",
                                        str(run_dir / "frames.jsonl"), "--server-log", str(rec / f"serve_safe_{port}.log"), "--out", "/dev/null"])
    settings = T.resolve_settings(args, T.parse_server_log(args.server_log), json.loads((run_dir / "manifest.json").read_text()))
    robot, qadr, _ = T.build_robot(settings)
    layout = ChunkLayout.rby1(robot.joint_names)
    planning = T.to_config(settings, collision=True, cost=dict(settings.cost), sqp={})
    limits = build_limits(robot, layout, dt=planning.horizon.dt, config=planning.limits)
    meter = T.Meter(robot, layout, planning.horizon.planned, settings.hand, limits)
    return dict(robot=robot, qadr=qadr, layout=layout, meter=meter, fmap=FingerJointMap(robot))


def main():
    A = json.loads((V / "p" / f"{COND}_analysis.json").read_text())["runs"]
    out = {"condition": COND, "source": SRC, "definitions": __doc__, "runs": {}}
    for run in sorted(A):
        run_dir = V / f"{COND}_rec" / run
        rows = {int(x["seq"]): x for x in (json.loads(l) for l in (run_dir / "frames.jsonl").read_text().splitlines() if l.strip())
                if x.get("kind") == "planning"}
        sh = setup(run_dir)
        lin, layout, robot = sh["meter"].lin, sh["layout"], sh["robot"]
        res = []
        for c in A[run]["chunks"]:
            if not c.get("server_chunk"):
                continue
            z = np.load(V / f"{COND}_rec" / c["server_chunk"], allow_pickle=False)
            s = json.loads(str(z["summary_json"]))
            tm = (s.get("to") or {}).get("metrics") or {}
            row = rows[c["seq"]]
            q_now = T.q_now_from_qpos(row["qpos"], sh["qadr"])
            refc = np.asarray(z["reference_chunk"], np.float64)
            meas = (s.get("finger_joints") or {}).get("opening_norm") or {}
            Pp, _ = sh["fmap"].path(meas, {"left": refc[:, 7], "right": refc[:, 15]}, lin.horizon, default=robot.param_vector())
            lin.set_attached(None, None)
            lin.set_joint_parameter_path(Pp)
            H = lin.horizon
            ref = layout.chunk_to_trajectory(refc)[:, :H]
            fin = layout.chunk_to_trajectory(np.asarray(z["refined_chunk"], np.float64))[:, :H]
            cmp = sh["meter"].compare(fin, ref, q_now)
            disp = np.asarray(cmp["fingertip_disp_mm_per_step"])
            ec = s.get("exec_continuity") or {}
            res.append({"t": c["t"], "seq": c["seq"], "state": (s.get("grasp") or {}).get("state"),
                        "continuity_active": tm.get("continuity_active"), "continuity_reason": tm.get("continuity_reason"),
                        "continuity_off_by": tm.get("continuity_off_by"), "exec_continuity_source": ec.get("source"),
                        "returned": tm.get("returned"), "sqp_iterations": tm.get("sqp_iterations"),
                        "n_executed": c.get("n_executed"), "n_control": c.get("n_control"),
                        "fingertip_disp_mm_max": float(disp.max()), "fingertip_disp_mm_max_rows0_7": float(disp[:8].max()),
                        "fingertip_disp_mm_per_step": disp.tolist(), "joint_dev_max_deg": cmp["joint_dev_max_deg"],
                        "palm_rot_deg_max": cmp["palm_rot_deg_max"], "horizon": int(H)})
        sel = [r for r in res if r["state"] in ("closing", "held")]
        out["runs"][run] = {
            "chunks": res,
            "n_chunks": len(res),
            "n_closing": sum(r["state"] == "closing" for r in res), "n_held": sum(r["state"] == "held" for r in res),
            "n_gate_on": sum(r["continuity_off_by"] is not None for r in res),
            "gate_on_t": [r["t"] for r in res if r["continuity_off_by"] is not None],
            "closing_held_t": [r["t"] for r in sel],
            "closing_held_tip_mm": [[r["t"], r["state"], round(r["fingertip_disp_mm_max"], 3)] for r in sel],
            "closing_held_tip_mm_max": max((r["fingertip_disp_mm_max"] for r in sel), default=None),
            "closing_held_tip_mm_median": float(np.median([r["fingertip_disp_mm_max"] for r in sel])) if sel else None,
            "closing_tip_mm_max": max((r["fingertip_disp_mm_max"] for r in sel if r["state"] == "closing"), default=None),
            "held_tip_mm_max": max((r["fingertip_disp_mm_max"] for r in sel if r["state"] == "held"), default=None),
            "other_tip_mm_max": max((r["fingertip_disp_mm_max"] for r in res if r["state"] not in ("closing", "held")), default=None),
            "n_closing_held_tip_gt_1mm": sum(1 for r in sel if r["fingertip_disp_mm_max"] > 1.0),
        }
        o = out["runs"][run]
        print(run, "closing", o["n_closing"], "held", o["n_held"], "gate_on", o["n_gate_on"], "tip max c/h", o["closing_tip_mm_max"], o["held_tip_mm_max"],
              "other", o["other_tip_mm_max"], ">1mm", o["n_closing_held_tip_gt_1mm"], flush=True)
    name = f"{COND}_tipdev.json" if SRC == "T37" else f"{COND}_tipdev_{SRC}.json"
    (OUTV / name).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
