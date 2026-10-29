"""AG3S-GEO-GRASP — the close gate offline on recorded closed-loop requests: "selected" vs "valid-pose", and whether
the planned close pose was a valid grasp on the ground-truth apple (evaluation only).

For each run of a pilot server record (one server serves several runs; `order.txt` + the client step counts split it),
the task state is replayed request by request (`GeoGraspController.step`, recorded cloud / target / policy chunk),
and on every request where the policy chunk closes inside the execution window the gate is evaluated on the planned
close pose of the recorded chunk (arm columns of `refined_chunk`, grip columns of `reference_chunk` — the policy's
command before any deferral). Ground truth: `close_pose_check` on the apple's collision mesh surface (samples + face
normals) and the obstacle surfaces, with the same thresholds except the friction cone (true cone atan(mu), no margin).

usage: python -m AG3S.trajopt.experiments.geo_gate_offline --out OUT.json --gt-dir DIR --apple-geoms NPZ SERVER_DIR CLIENT_RUNS_DIR
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import types

import numpy as np


def main(argv=None):
    from AG3S.ag3s.experiments.studies.geo_grasp_offline import GroundTruth
    from AG3S.ag3s.grasp.candidates import GeoGraspConfig, close_pose_check
    from AG3S.ag3s.grasp.gripper import RBY1_LEFT
    from AG3S.ag3s.grasp.surface import SurfaceEstimate
    from AG3S.trajopt import wire
    from AG3S.trajopt.config import TrajOptConfig
    from AG3S.trajopt.experiments import fixed_input_ablation as FIA
    from AG3S.trajopt.experiments import ta_offline as TA
    from AG3S.trajopt.experiments.geo_grasp_replay import _manip
    from AG3S.trajopt.geo_grasp import GeoGraspController

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--apple-geoms", required=True)
    ap.add_argument("server_dir")
    ap.add_argument("runs_dir")
    a = ap.parse_args(argv)
    srv, runs = pathlib.Path(a.server_dir), pathlib.Path(a.runs_dir)
    order = (srv / "order.txt").read_text().split()
    chunks = sorted((srv / "server_constraints" / "run_0000").glob("chunk_*.npz"))
    zz = np.load(a.apple_geoms, allow_pickle=True)
    key = [k for k in zz.files if k.startswith("i") and json.loads(str(zz[k]))["contype"] == 1][0]
    apple_cv = zz["v" + key[1:]]
    sargs, _ = TA.server_args(chunks[0].parent)
    robot, qadr = TA.build_robot(sargs)
    from AG3S.trajopt.types import ChunkLayout
    layout = ChunkLayout.rby1(robot.joint_names)
    cols = wire.gripper_columns(layout.nq_opt // 2)
    gcfg = GeoGraspConfig(friction_mu=0.8)
    gt_cfg = dataclasses.replace(gcfg, normal_margin_deg=0.0, target_margin=-0.001)
    out, pos = {"runs": {}}, 0
    for name in order:
        rows = [json.loads(x) for x in open(runs / name / "frames.jsonl") if x.strip()]
        plan = {int(r["seq"]): r for r in rows if r["kind"] == "planning"}
        steps = sorted((runs / name / "ag3s" / "run_0000").glob("step_*.npz"))
        seg = chunks[pos:pos + len(steps)]
        pos += len(steps)
        case = name.split("_")[1] + "_" + name.split("_")[2]
        gt_path = pathlib.Path(a.gt_dir) / f"{case}_{name.split('_')[3]}.npz"
        if not gt_path.exists():
            continue                                        # no ground truth extracted for this scene
        gt = GroundTruth(gt_path, collision_vertices=apple_cv)
        cost = TrajOptConfig().with_overrides({"cost": {"geo_grasp": True, "geo_close_gate": "valid-pose"}}).cost
        ctl = GeoGraspController(cost, robot, layout, {"left": "ee_left", "right": "ee_right"}, cols, 0.85,
                                 TrajOptConfig().limits, horizon=16, execution_length=8)
        ctl.gcfg = gcfg
        recs = []
        for k, (f, st) in enumerate(zip(seg, steps)):
            z = np.load(f, allow_pickle=True)
            s = json.loads(str(z["summary_json"]))
            seq = int(s.get("t_step", k + 1))
            row = plan.get(seq)
            if row is None:
                continue
            q_now = FIA.q_now_from_qpos(row["qpos"], qadr)
            phase = str((s.get("grasp") or {}).get("state") or "searching")
            manip, cloud = _manip(s, z)
            ref = np.asarray(z["reference_chunk"], np.float64)
            ctl.step(phase=phase, hand="left", manip=manip, cloud_points=cloud, q_now=q_now, chunk=ref)
            if phase != "latched":
                continue
            planned = np.asarray(z["refined_chunk"], np.float64).copy()
            planned[:, list(cols)] = ref[:, list(cols)]
            c = ctl.close_row(planned, "left")
            if c is None or c >= 8:
                continue
            q_c = ctl._chunk_q(planned, c, q_now)
            T = ctl._pose(q_c, "ee_left")
            cs = ctl._planners["left"].last_set
            ok_obs, info_obs = close_pose_check(T, RBY1_LEFT, gcfg, None if cs is None else cs.surface,
                                                None if cs is None else cs.scene, None if cs is None else cs.shell)
            qpos = np.asarray(np.load(st)["qpos"], np.float64)
            apple, apple_c = gt.apple(qpos)
            surf_gt = SurfaceEstimate(apple, gt.apple_normals(qpos), np.zeros(len(apple)), np.full(len(apple), 99),
                                      apple.mean(axis=0), "gt", None)
            ok_gt, info_gt = close_pose_check(T, RBY1_LEFT, gt_cfg, surf_gt, gt.obstacle_points(apple_c), apple)
            sel = ctl.selected
            near = None
            if sel is not None:
                from AG3S.ag3s.grasp.candidates import grasp_rot_angle
                dp = float(np.linalg.norm(T[:3, 3] - sel.T[:3, 3]))
                dr = grasp_rot_angle(T[:3, :3], sel.T[:3, :3])
                near = bool(dp <= cost.geo_close_tol_pos and dr <= np.radians(cost.geo_close_tol_rot_deg))
            rec_gate = (s.get("grasp") or {}).get("geo", {}).get("close_gate") or {}
            pad_c = T[:3, 3] + T[:3, :3] @ np.array([0.0, 0.0, -RBY1_LEFT.pad_centre_depth])
            hand_now = ctl._pose(q_now, "ee_left")
            vla_c = ctl._pose(ctl._chunk_q(ref, c, q_now), "ee_left")
            vla_pad = vla_c[:3, 3] + vla_c[:3, :3] @ np.array([0.0, 0.0, -RBY1_LEFT.pad_centre_depth])
            recs.append({"seq": seq, "close_row": c, "recorded_decision": rec_gate.get("decision"),
                         "pad_centre_to_apple_mm": round(float(np.linalg.norm(pad_c - apple.mean(axis=0))) * 1000, 1),
                         "vla_unmodified_pad_to_apple_mm": round(float(np.linalg.norm(vla_pad - apple.mean(axis=0))) * 1000, 1),
                         "palm_now_to_apple_mm": round(float(np.linalg.norm(hand_now[:3, 3] - apple.mean(axis=0))) * 1000, 1),
                         "selected_rule": "close" if near else "defer",
                         "valid_pose_rule": "close" if (near or ok_obs is None or ok_obs) else "defer",
                         "observed_check": {"ok": ok_obs, **{k2: info_obs.get(k2) for k2 in ("reason", "width_mm", "normal_angle_deg")}},
                         "gt_check": {"ok": ok_gt, **{k2: info_gt.get(k2) for k2 in ("reason", "width_mm", "normal_angle_deg")}}})
        out["runs"][name] = recs
        print(name, len(recs), "close attempts;",
              "valid-pose closes", sum(r["valid_pose_rule"] == "close" for r in recs),
              "| of those GT-valid", sum(r["valid_pose_rule"] == "close" and bool(r["gt_check"]["ok"]) for r in recs),
              "| selected-rule closes", sum(r["selected_rule"] == "close" for r in recs),
              "| GT-valid attempts", sum(bool(r["gt_check"]["ok"]) for r in recs), flush=True)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


if __name__ == "__main__":
    main()
