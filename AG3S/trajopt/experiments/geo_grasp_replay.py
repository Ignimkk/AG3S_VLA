"""AG3S-GEO-GRASP S2 — offline replay of recorded closed-loop requests with the 6-D geometric goal (B0 / B1 / B2).

Per run (an E3 closed-loop record: client run dir + server chunk records) every request is re-solved by the recorded
server configuration (`ta_offline.Run`) in server mode (each setting's own last chunk is its continuity reference):

* B0 — the recorded configuration;
* B1 — B0 + `cost.geo_grasp` (task state, 6-D goal, close gate);
* B2 — B1 + `geo_track_scale 0.25` + `geo_continuity_off` (+ carry restore over 3 requests).

The GPU seed batches and the multi-seed re-solve are off for all three (no GPU here; CPU seeds would dominate the
timing) — said in the output. The task state reads the ATS target as in S1 (the recorded cloud within the manipulated
object's radius; the record has no point indices). Replay limit: the inputs of request n+1 are the recorded ones, so a
setting's different output never changes what it sees next (open loop on the record).

usage: python -m AG3S.trajopt.experiments.geo_grasp_replay --out DIR INDEX_JSON KEY [KEY ...]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
import types
from typing import Any

import numpy as np

SETTINGS = ("B0", "B1", "B2")


def _manip(summary: dict, z) -> Any:
    m = summary.get("manipulated") or None
    if m is None or m.get("centroid") is None:
        return None, np.zeros((0, 3))
    cloud = np.asarray(z["filtered_cloud"], np.float64)
    fin = np.all(np.isfinite(cloud), axis=1)
    sup = np.asarray(z["support_mask"], bool) if "support_mask" in z.files else np.zeros(len(cloud), bool)
    cen = np.asarray(m["centroid"], np.float64)
    near = fin & ~sup & (np.linalg.norm(cloud - cen, axis=1) <= float(m.get("radius") or 0.04) + 0.005)
    geom = types.SimpleNamespace(points=cloud[near]) if near.any() else None
    return types.SimpleNamespace(id=int(m["id"]), state=str(m.get("state")), centroid=cen, geometry=geom), cloud[fin]


def run(key: str, entry: dict, out_dir: pathlib.Path, max_chunks: int = 0, keep_palm: bool = False,
        goal_mode: str = "redirect", gt=None) -> dict[str, Any]:
    from AG3S.trajopt import wire
    from AG3S.trajopt.experiments import ta_offline as TA
    from AG3S.trajopt.experiments import tp_timing as TP
    from AG3S.trajopt.geo_grasp import GeoGraspController
    from AG3S.trajopt.grasp_anchor import LinkPoseFK, rotation_errors
    from AG3S.trajopt.refiner import TrajOptChunkRefiner

    TP.install()
    ns = argparse.Namespace(hm=False, no_plane_rows=False)
    R = TA.Run(key, entry, ns)
    base = R.base_cfg.with_overrides({"sqp": {"gpu_seeds": "off", "gpu_seeds_check": "all", "multi_seed": 0},
                                      "collision": {"held_obstacle_margin": False}})
    cfgs = {"B0": base,
            "B1": base.with_overrides({"cost": {"geo_grasp": True, "geo_goal_mode": goal_mode}}),
            "B2": base.with_overrides({"cost": {"geo_grasp": True, "geo_track_scale": 0.25,
                                                "geo_continuity_off": True, "geo_goal_mode": goal_mode}})}
    holder: dict[str, Any] = {}
    refiners = {v: TrajOptChunkRefiner(R.robot, R.layout, lambda _c: holder["value"], cfgs[v]) for v in SETTINGS}
    cols = wire.gripper_columns(R.layout.nq_opt // 2)
    geos = {v: GeoGraspController(cfgs[v].cost, R.robot, R.layout, {"left": "ee_left", "right": "ee_right"}, cols,
                                  0.85, cfgs[v].limits, horizon=int(cfgs[v].horizon.planned),
                                  execution_length=int(cfgs[v].horizon.execution_length))
            for v in ("B1", "B2")}
    H = int(base.horizon.planned)
    pose_fk = LinkPoseFK(R.robot, R.layout, H, "ee_left")
    names = [str(R.layout.joint_names[i]) for i in R.layout.q_indices]
    own_prev: dict[str, Any] = {}
    out: dict[str, Any] = {"key": key, "settings": {v: cfgs[v].to_dict()["cost"] for v in SETTINGS},
                           "note": "gpu_seeds off, multi_seed 0 for all settings (offline, CPU)", "chunks": []}
    seqs = sorted(R.chunks)
    if max_chunks:
        seqs = seqs[:max_chunks]
    prev_seq = None
    for seq in seqs:
        inp, why = R.inputs(seq)
        if inp is None:
            out["chunks"].append({"seq": seq, "skipped": why})
            prev_seq = None
            continue
        s, q_now, fb, z = inp["summary"], inp["q_now"], inp["feedback"], inp["z"]
        scene, _ = TP.prepare_field(inp["scene"], "cpu")
        phase = str((s.get("grasp") or {}).get("state") or "searching")
        manip, cloud = _manip(s, z)
        prev_chunk = None
        if prev_seq is not None and prev_seq == seq - 1:
            with np.load(R.chunks[prev_seq], allow_pickle=True) as pz:
                prev_chunk = np.asarray(pz["refined_chunk"], np.float64)
        ctx0: dict[str, Any] = {"t_step": seq, "previous_physical_chunk": prev_chunk}
        n_exec = fb.get("n_exec") if fb.get("available") else None
        if n_exec is not None:
            ctx0["previous_executed_steps"] = int(n_exec)
        rec: dict[str, Any] = {"seq": seq, "phase": phase, "target_state": None if manip is None else manip.state,
                               "settings": {}}
        ref_traj = R.layout.chunk_to_trajectory(inp["ref_chunk"])[:, :H]
        goal_for_b0 = None
        for v in SETTINGS:
            cfg = cfgs[v]
            ref = refiners[v]
            ref.optimizer.linearizer.set_joint_parameter_path(R.finger_path(s, inp["ref_chunk"], H))
            holder["value"] = (scene, q_now, True)
            ctx = dict(ctx0)
            if own_prev.get(v) is not None and ctx.get("previous_physical_chunk") is not None:
                ctx["previous_physical_chunk"] = own_prev[v]
            geo_rec, geo_ms = None, 0.0
            if v in geos:
                g = geos[v]
                t0 = time.perf_counter()
                g.step(phase=phase, hand="left", manip=manip, cloud_points=cloud, q_now=q_now, chunk=inp["ref_chunk"])
                anchor = g.anchor(q_now, inp["ref_chunk"])
                if anchor is not None:
                    ctx["grasp_anchor"] = anchor
                    if goal_for_b0 is None:
                        goal_for_b0 = anchor
                off = g.continuity_off()
                if off:
                    ctx["continuity_off"] = off
                sc = g.track_scale(names)
                if sc is not None:
                    ctx["joint_track_scale"] = sc
                geo_ms = (time.perf_counter() - t0) * 1000.0
            w0 = time.perf_counter()
            own = np.asarray(ref.refine(inp["ref_chunk"], ctx), np.float64)
            wall = (time.perf_counter() - w0) * 1000.0
            own_prev[v] = own
            res = ref.last_result
            traj = np.asarray(res.trajectory, np.float64)[:, :H]
            e: dict[str, Any] = {"status": res.status.value, "max_violation_mm": round(float(res.max_violation) * 1e3, 3),
                                 "to_ms": round(wall, 1), "geo_ms": round(geo_ms, 1), "total_ms": round(wall + geo_ms, 1),
                                 "dev_from_vla_deg_rows0_7": round(float(np.degrees(np.abs(traj[:, :8] - ref_traj[:, :8]).max())), 3)}
            if v in geos:
                g = geos[v]
                gated, hold = g.close_gate(own, q_now)
                e["geo"] = {k: g.last.get(k) for k in ("valid", "reason", "event", "generation", "n_candidates", "grasp",
                                                        "anchor", "track", "continuity_off", "close_gate", "started",
                                                        "failed")}
                e["geo"]["hold"] = hold
                e["grasp_anchor"] = res.metrics.get("grasp_anchor")
            if gt is not None and phase == "latched":
                # the planned close of this setting judged on the ground-truth apple (evaluation only)
                planned = np.asarray(own, np.float64).copy()
                planned[:, list(cols)] = np.asarray(inp["ref_chunk"], np.float64)[:, list(cols)]
                cr = np.flatnonzero(planned[:, cols[0]] < 0.85)
                if cr.size and cr[0] < 8:
                    from AG3S.ag3s.grasp.candidates import GeoGraspConfig, close_pose_check
                    from AG3S.ag3s.grasp.gripper import RBY1_LEFT
                    from AG3S.ag3s.grasp.surface import SurfaceEstimate

                    qpos = np.asarray(R.plan[seq]["qpos"], np.float64)
                    apple, apple_c = gt.apple(qpos)
                    full = R.layout.full_q(R.layout.chunk_to_trajectory(planned)[:, int(cr[0]):int(cr[0]) + 1], q_now)[:, 0]
                    T = R.robot.link_pose(full, "ee_left")
                    pad = T[:3, 3] + T[:3, :3] @ np.array([0, 0, -RBY1_LEFT.pad_centre_depth])
                    surf_gt = SurfaceEstimate(apple, gt.apple_normals(qpos), np.zeros(len(apple)), np.full(len(apple), 99),
                                              apple.mean(axis=0), "gt", None)
                    ok_gt, info = close_pose_check(T, RBY1_LEFT, GeoGraspConfig(friction_mu=0.8, normal_margin_deg=0.0),
                                                   surf_gt, gt.obstacle_points(apple_c), apple)
                    e["close_gt"] = {"close_row": int(cr[0]), "pad_to_apple_mm": round(float(np.linalg.norm(pad - apple.mean(axis=0))) * 1000, 1),
                                     "ok": ok_gt, "reason": info.get("reason")}
            if keep_palm:
                pp = pose_fk.pose(traj, q_now)[0]
                pr = pose_fk.pose(ref_traj, q_now)[0]
                e["palm_path"] = np.round(pp, 5).tolist()
                rec["vla_palm_path"] = np.round(pr, 5).tolist()
            rec["settings"][v] = e
        # B0 measured against the goal B1 used on this request (same definition), when B1 had one
        if goal_for_b0 is not None:
            rows = np.flatnonzero(goal_for_b0.goal_mask(H))
            if rows.size:
                for v in SETTINGS:
                    traj = np.asarray(refiners[v].last_result.trajectory, np.float64)[:, :H]
                    p, _, Rr, _ = pose_fk.pose(traj, q_now)
                    rec["settings"][v]["goal_b1"] = {
                        "kind": goal_for_b0.goal_kind,
                        "pos_err_mm": round(float(np.max(np.linalg.norm(p[rows] - goal_for_b0.goal, axis=1))) * 1000, 2),
                        "rot_err_deg": round(math.degrees(float(np.max(np.linalg.norm(
                            rotation_errors(Rr, rows, goal_for_b0.goal_R), axis=1)))), 2)}
        out["chunks"].append(rec)
        z.close()
        prev_seq = seq
    out["n_goal_chunks"] = {v: geos[v].n_goal_chunks for v in geos}
    (out_dir / f"{key.replace('/', '_')}.json").write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--keep-palm", action="store_true", help="also store each setting's planned palm path (figures)")
    ap.add_argument("--goal-mode", default="redirect", choices=("redirect", "correct"))
    ap.add_argument("--gt-dir", default=None, help="ground truth npz per scene (geo_grasp_offline) — judge planned closes")
    ap.add_argument("--apple-geoms", default=None)
    ap.add_argument("index")
    ap.add_argument("keys", nargs="+")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = json.load(open(a.index))["runs"]
    for k in a.keys:
        t = time.time()
        gt = None
        if a.gt_dir:
            from AG3S.ag3s.experiments.studies.geo_grasp_offline import GroundTruth
            zz = np.load(a.apple_geoms, allow_pickle=True)
            kk = [x for x in zz.files if x.startswith("i") and json.loads(str(zz[x]))["contype"] == 1][0]
            name = pathlib.Path(idx[k]["run_dir"]).name                     # AG3S_case1_L2_s13453
            gt = GroundTruth(pathlib.Path(a.gt_dir) / f"{name[5:13]}_{name[14:]}.npz", collision_vertices=zz["v" + kk[1:]])
        r = run(k, idx[k], out, a.max_chunks, a.keep_palm, a.goal_mode, gt)
        print(k, "chunks", len(r["chunks"]), "goal chunks", r["n_goal_chunks"], f"{time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
