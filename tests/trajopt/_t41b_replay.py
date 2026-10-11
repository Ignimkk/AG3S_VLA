"""T41 b helper — re-solve a recorded T40 G E3b chunk with the recorded inputs (fixed-input replay).

Inputs are restored as `tests/trajopt/test_t37_grasp_continuity_replay.py` does (the `fixed_input_ablation`
functions): server-log settings, the chunk npz's distance field (coarse + fine + target-free layers), the
paired client planning row's `qpos` -> `q_now`, contact-policy masks, the finger path from the measured
opening, no wall clock, `max_iterations` = the recorded SQP iteration count. The continuity input follows
the recorded `exec_feedback` (`n_exec == 0` -> none, else the previous recorded refined chunk aligned by
`n_exec`) — what `SafePolicy._continuity_input` would hand the refiner.

Run as a script it replays a fixed list of chunks and writes one npz — the HEAD-vs-tree bit-identity test
runs it under a `git archive` of the pre-T41b benchmark (`PYTHONPATH=<archive>`) so both sides use the
same harness code and only `benchmark/` differs.
"""

from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

import numpy as np

G = pathlib.Path(_ag3s_workspace_path('legacy/outputs/verify/T40/G/runs/E3b'))
SERVER = G / "server_8240_20261003T135313"
CHUNKS = SERVER / "server_constraints/run_0000"
#: run dir per recorded chunk index, and the chunk index of a planning seq (`chunk = seq + offset`).
RUNS = {"C1807": (G / "C/E3b_ep1807_s18071", 102), "B1968": (G / "B/E3b_ep1968_s19681", 130)}
#: chunks replayed for bit identity: C1807 seq 24-28 (the T40 I window: HOLD x3 then the contact chunk)
#: and B1968 seq 15 (no obstacle, just before the grasp — the T41 a fixture chunk).
IDENTITY = (("C1807", 24), ("C1807", 25), ("C1807", 26), ("C1807", 27), ("C1807", 28), ("B1968", 15))

_SETUP: dict = {}


def available() -> bool:
    return (SERVER / "serve_safe.log").exists() and all(
        (d / "frames.jsonl").exists() for d, _ in RUNS.values())


def _setup(name: str) -> dict:
    if name in _SETUP:
        return _SETUP[name]
    from AG3S.ag3s.robot_models.gripper_state import FingerJointMap
    from AG3S.trajopt.experiments import fixed_input_ablation as T
    from AG3S.trajopt.limits import build_limits
    from AG3S.trajopt.types import ChunkLayout

    run, offset = RUNS[name]
    args = T.build_parser().parse_args([
        "--constraints", str(CHUNKS), "--frames", str(run / "frames.jsonl"),
        "--server-log", str(SERVER / "serve_safe.log"), "--out", "/dev/null"])
    st = T.resolve_settings(args, T.parse_server_log(args.server_log),
                            json.loads((run / "manifest.json").read_text()))
    robot, qadr, _ = T.build_robot(st)
    layout = ChunkLayout.rby1(robot.joint_names)
    _, _, _, maxd, unk = T.contact_policy(st, robot.sphere_link_names, has_target=False,
                                          has_free=False, attached=False)
    base = T.to_config(st, collision=True, cost=dict(st.cost), sqp={})
    limits = build_limits(robot, layout, dt=base.horizon.dt, config=base.limits)
    rows = {int(x["seq"]): x for x in (json.loads(line) for line in
                                       (run / "frames.jsonl").read_text().splitlines()
                                       if line.strip()) if x.get("kind") == "planning"}
    _SETUP[name] = dict(T=T, st=st, robot=robot, qadr=qadr, layout=layout, offset=offset,
                        outside=maxd if unk == "free" else -maxd, rows=rows, limits=limits,
                        meter=T.Meter(robot, layout, base.horizon.planned, st.hand, limits),
                        fmap=FingerJointMap(robot))
    return _SETUP[name]


def summary(name: str, seq: int) -> dict:
    sh = _setup(name)
    z = np.load(CHUNKS / f"chunk_{seq + sh['offset']:05d}.npz", allow_pickle=False)
    return json.loads(str(z["summary_json"]))


def replay(name: str, seq: int, *, limits: dict | None = None, at_rest: bool = False) -> dict:
    """One chunk. `limits` overrides the `limits` section (e.g. `{"rest_start": True}`); `at_rest` puts
    `start_at_rest` in the refiner context (what `SafePolicy` does when it decides the robot is at rest)."""
    sh = _setup(name)
    T, st, robot, layout = sh["T"], sh["st"], sh["robot"], sh["layout"]
    z = np.load(CHUNKS / f"chunk_{seq + sh['offset']:05d}.npz", allow_pickle=False)
    assert int(z["t_step"]) == seq, (name, seq, int(z["t_step"]))
    s = json.loads(str(z["summary_json"]))
    tm = (s.get("to") or {}).get("metrics") or {}
    q_now = T.q_now_from_qpos(sh["rows"][seq]["qpos"], sh["qadr"])
    field = T.build_field(z, outside_distance=sh["outside"])
    tp = z["target_points"] if "target_points" in z.files else None
    margin, mask, _, _, _ = T.contact_policy(
        st, robot.sphere_link_names, has_target=bool(s.get("has_target")) and tp is not None,
        has_free=bool(field.target_free_layers), attached=False)
    lin = sh["meter"].lin
    scene = T.make_scene(field, lin.robot_radii, st.esdf_margin, target_points=tp,
                         margin=margin, mask=mask)
    refc = np.asarray(z["reference_chunk"], np.float64)
    meas = (s.get("finger_joints") or {}).get("opening_norm") or {}
    path, _ = sh["fmap"].path(meas, {"left": refc[:, 7], "right": refc[:, 15]}, lin.horizon,
                              default=robot.param_vector())
    cfg = T.to_config(st, collision=True, cost=dict(st.cost),
                      sqp={"time_budget_ms": T.NO_CLOCK_MS,
                           "max_iterations": max(1, int(tm.get("sqp_iterations") or 1))})
    if limits:
        cfg = cfg.with_overrides({"limits": dict(limits)})
    runner = T.Instrumented(robot, layout, cfg, {"value": (scene, q_now, s.get("status") == "ok")})
    runner.refiner.optimizer.linearizer.set_joint_parameter_path(path)
    fb = s.get("exec_feedback") or {}
    ctx: dict = {"t_step": seq}
    if fb.get("available") and int(fb.get("n_exec") or 0) > 0:
        zp = np.load(CHUNKS / f"chunk_{seq + sh['offset'] - 1:05d}.npz", allow_pickle=False)
        ctx["previous_physical_chunk"] = np.asarray(zp["refined_chunk"], np.float64)
        ctx["previous_executed_steps"] = int(fb["n_exec"])
    if at_rest:
        ctx["start_at_rest"] = True
    result, _ = runner.run(refc, ctx)
    q0 = q_now[layout.q_indices]
    rec_traj = layout.chunk_to_trajectory(np.asarray(z["refined_chunk"], np.float64))
    return {
        "chunk": np.asarray(result.chunk, np.float64),
        "trajectory": np.asarray(result.trajectory, np.float64),
        "status": result.status.value,
        "safe": bool(result.status.safe),
        "clearance_m": float(result.metrics["clearance_m"]),
        "metrics": result.metrics,
        "q_now_opt": q0,
        "step0_offset_rad": np.abs(result.trajectory[:, 0] - q0),
        "recorded_step0_offset_rad": np.abs(rec_traj[:, 0] - q0),
        "max_step": np.asarray(sh["limits"].max_step, np.float64),
        "max_step_change": np.asarray(sh["limits"].max_step_change, np.float64),
        "feedback": fb,
    }


def main(out_path: str) -> int:
    import benchmark

    arrays, meta = {}, {"benchmark_file": benchmark.__file__}
    for name, seq in IDENTITY:
        r = replay(name, seq)
        arrays[f"{name}_{seq}_chunk"] = r["chunk"]
        meta[f"{name}_{seq}"] = {"status": r["status"], "clearance_m": r["clearance_m"],
                                 "returned": r["metrics"].get("returned")}
    np.savez(out_path, meta=json.dumps(meta), **arrays)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
