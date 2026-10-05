"""T43 Q helper — re-solve a recorded E3b chunk with the recorded inputs, with or without the servo /
sweep rows (fixed-input replay, the `_t41b_replay` method).

Inputs as `tests/trajopt/_t41b_replay.py`: server-log settings, the chunk npz's distance field, the paired
client planning row's `qpos` -> `q_now`, contact-policy masks, the finger path from the measured opening,
no wall clock, `max_iterations` = the recorded SQP iteration count, continuity from the recorded
`exec_feedback`. Additionally:

* `obstacle_margin` — the T41 a margin as the T41 a server ran it (`--obstacle-margin 0.010`): support
  planes refitted from the record's filtered cloud with AG3S `fit_support_surfaces` (the verifier's
  `replay_t41.py` method).
* the servo observer is **run over the recorded history** (planning-row `qpos` + each chunk's recorded
  `exec_feedback.applied_arm`, exactly what the server received) up to the chunk before, so the replayed
  chunk starts from the velocity the live server would have estimated.

Run as a script it replays a fixed list of chunks with the flags off and writes one npz — the bit-identity
test runs it once against the working tree and once against a copy whose T43 Q files are reverted to the
pre-T43 commit, so both sides use the same harness and only the T43 Q edits differ.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys

import numpy as np

G40 = pathlib.Path("/mnt/dev/work/outputs/verify/T40/G/runs/E3b")
G41 = pathlib.Path("/mnt/dev/work/outputs/verify/T41/a/runs/E3b")
#: name -> (server dir, run dir, chunk offset (chunk index = seq + offset), obstacle margin of that server)
RUNS = {
    "T40C1807": (G40 / "server_8240_20261003T135313", G40 / "C/E3b_ep1807_s18071", 102, 0.0),
    "T40B1968": (G40 / "server_8240_20261003T135313", G40 / "B/E3b_ep1968_s19681", 130, 0.0),
    "T41aC1807": (G41 / "server_8260_20261005T072218", G41 / "C/E3b_ep1807_s18071", 74, 0.010),
}
#: off replays for bit identity: T40 C1807 seq 24-28 (HOLD x3 then the t=218 contact chunk), B1968 seq 15,
#: T41 a C1807 seq 34-35 (the t=276 contact chunk and the one before).
IDENTITY = (("T40C1807", 24), ("T40C1807", 26), ("T40C1807", 28), ("T40B1968", 15),
            ("T41aC1807", 34), ("T41aC1807", 35))

_SETUP: dict = {}


def available(names=None) -> bool:
    names = names or tuple(RUNS)
    return all((RUNS[n][0] / "serve_safe.log").exists() and (RUNS[n][1] / "frames.jsonl").exists()
               for n in names)


def _setup(name: str) -> dict:
    if name in _SETUP:
        return _SETUP[name]
    from benchmark.ag3s.robot_models.gripper_state import FingerJointMap
    from benchmark.trajopt.experiments import fixed_input_ablation as T
    from benchmark.trajopt.limits import build_limits
    from benchmark.trajopt.types import ChunkLayout

    server, run, offset, margin = RUNS[name]
    chunks = server / "server_constraints/run_0000"
    args = T.build_parser().parse_args([
        "--constraints", str(chunks), "--frames", str(run / "frames.jsonl"),
        "--server-log", str(server / "serve_safe.log"), "--out", "/dev/null"])
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
                        chunks=chunks, margin=margin, run=run,
                        outside=maxd if unk == "free" else -maxd, rows=rows, limits=limits,
                        meter=T.Meter(robot, layout, base.horizon.planned, st.hand, limits),
                        fmap=FingerJointMap(robot))
    return _SETUP[name]


def npz(name: str, seq: int):
    sh = _setup(name)
    z = np.load(sh["chunks"] / f"chunk_{seq + sh['offset']:05d}.npz", allow_pickle=False)
    assert int(z["t_step"]) == seq, (name, seq, int(z["t_step"]))
    return z


def summary(name: str, seq: int) -> dict:
    return json.loads(str(npz(name, seq)["summary_json"]))


def q_now(name: str, seq: int) -> np.ndarray:
    sh = _setup(name)
    return sh["T"].q_now_from_qpos(sh["rows"][seq]["qpos"], sh["qadr"])


def support_planes(z) -> np.ndarray:
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.stages.support_surface import fit_support_surfaces
    from benchmark.ag3s.types import PointCloud

    surf, _ = fit_support_surfaces(PointCloud(points=np.asarray(z["filtered_cloud"], np.float64)),
                                   AG3SConfig().support_surface, timestamp=0.0)
    return np.asarray([[*np.asarray(s.normal, float), float(s.offset)] for s in surf],
                      float).reshape(-1, 4)


def history_state(name: str, seq: int, model):
    """`ServoObserver` run over seq 1 .. seq-1 with what the server received; returns the observer."""
    from benchmark.trajopt import wire
    from benchmark.trajopt.servo import ServoObserver

    sh = _setup(name)
    obs = ServoObserver(model)
    qi = sh["layout"].q_indices
    for s in range(1, seq):
        if s not in sh["rows"]:
            continue
        fb = wire.unpack_exec_feedback({"exec_feedback": summary(name, s).get("exec_feedback")})
        obs.update(q_now(name, s)[qi], fb, s)
    return obs


def scene_of(name: str, seq: int, *, obstacle_margin: float | None = None):
    sh = _setup(name)
    T, st, robot = sh["T"], sh["st"], sh["robot"]
    z = npz(name, seq)
    s = json.loads(str(z["summary_json"]))
    field = T.build_field(z, outside_distance=sh["outside"])
    tp = z["target_points"] if "target_points" in z.files else None
    margin, mask, _, _, _ = T.contact_policy(
        st, robot.sphere_link_names, has_target=bool(s.get("has_target")) and tp is not None,
        has_free=bool(field.target_free_layers), attached=False)
    lin = sh["meter"].lin
    scene = T.make_scene(field, lin.robot_radii, st.esdf_margin, target_points=tp,
                         margin=margin, mask=mask)
    om = sh["margin"] if obstacle_margin is None else float(obstacle_margin)
    if om > 0.0:
        scene = dataclasses.replace(scene, obstacle_margin=om, support_planes=support_planes(z))
    return scene, s


def replay(name: str, seq: int, *, overrides: dict | None = None,
           obstacle_margin: float | None = None, max_iterations: int | None = None) -> dict:
    """One chunk. `overrides` = TrajOptConfig overrides (e.g. `{"limits": {"servo_model": True}}`)."""
    from benchmark.trajopt import wire
    from benchmark.trajopt.refiner import TrajOptChunkRefiner

    sh = _setup(name)
    T, st, robot, layout = sh["T"], sh["st"], sh["robot"], sh["layout"]
    z = npz(name, seq)
    scene, s = scene_of(name, seq, obstacle_margin=obstacle_margin)
    tm = (s.get("to") or {}).get("metrics") or {}
    qn = q_now(name, seq)
    refc = np.asarray(z["reference_chunk"], np.float64)
    meas = (s.get("finger_joints") or {}).get("opening_norm") or {}
    lin = sh["meter"].lin
    path, _ = sh["fmap"].path(meas, {"left": refc[:, 7], "right": refc[:, 15]}, lin.horizon,
                              default=robot.param_vector())
    iters = max_iterations or max(1, int(tm.get("sqp_iterations") or 1))
    cfg = T.to_config(st, collision=True, cost=dict(st.cost),
                      sqp={"time_budget_ms": T.NO_CLOCK_MS, "max_iterations": iters})
    if overrides:
        cfg = cfg.with_overrides(overrides)
    holder = {"value": (scene, qn, s.get("status") == "ok")}
    refiner = TrajOptChunkRefiner(robot, layout, lambda _ctx: holder["value"], cfg)
    refiner.reset()
    opt = refiner.optimizer
    opt.linearizer.set_joint_parameter_path(path)
    fb_raw = s.get("exec_feedback") or {}
    ctx: dict = {"t_step": seq}
    if fb_raw.get("available") and int(fb_raw.get("n_exec") or 0) > 0:
        zp = npz(name, seq - 1)
        ctx["previous_physical_chunk"] = np.asarray(zp["refined_chunk"], np.float64)
        ctx["previous_executed_steps"] = int(fb_raw["n_exec"])
    if getattr(opt, "observer", None) is not None:          # absent in the pre-T43 code
        hist = history_state(name, seq, opt.servo)
        opt.observer._last, opt.observer._last_seq = hist._last, hist._last_seq
        ctx["servo_feedback"] = {"feedback": wire.unpack_exec_feedback(
            {"exec_feedback": fb_raw}), "seq": seq}
    refiner.refine(refc, ctx)
    result = refiner.last_result
    assert result is not None, refiner.last_failure
    return {"result": result, "chunk": np.asarray(result.chunk, np.float64),
            "trajectory": np.asarray(result.trajectory, np.float64),
            "status": result.status.value, "safe": bool(result.status.safe),
            "clearance_m": float(result.metrics["clearance_m"]), "metrics": result.metrics,
            "q_now": qn, "scene": scene, "optimizer": opt, "summary": s,
            "recorded_refined": np.asarray(z["refined_chunk"], np.float64)}


def check_plan(name: str, seq: int, chunk: np.ndarray, *, overrides: dict,
               obstacle_margin: float | None = None) -> dict:
    """The path check (servo / sweep) of a **given** plan, without solving: what the new rows would
    have said about it. Returns the `path_check` record and the waypoint worst clearance."""
    from benchmark.trajopt import wire
    from benchmark.trajopt.refiner import TrajOptChunkRefiner

    sh = _setup(name)
    T, st, robot, layout = sh["T"], sh["st"], sh["robot"], sh["layout"]
    scene, s = scene_of(name, seq, obstacle_margin=obstacle_margin)
    qn = q_now(name, seq)
    z = npz(name, seq)
    refc = np.asarray(z["reference_chunk"], np.float64)
    meas = (s.get("finger_joints") or {}).get("opening_norm") or {}
    cfg = T.to_config(st, collision=True, cost=dict(st.cost), sqp={}).with_overrides(overrides)
    opt = TrajOptChunkRefiner(robot, layout, lambda _ctx: None, cfg).optimizer  # planning horizon
    path, _ = sh["fmap"].path(meas, {"left": refc[:, 7], "right": refc[:, 15]}, opt.horizon,
                              default=robot.param_vector())
    opt.linearizer.set_joint_parameter_path(path)
    state = None
    if opt.observer is not None:
        hist = history_state(name, seq, opt.servo)
        opt.observer._last, opt.observer._last_seq = hist._last, hist._last_seq
        state = opt.observer.update(qn[layout.q_indices], wire.unpack_exec_feedback(
            {"exec_feedback": s.get("exec_feedback")}), seq)
    traj = layout.chunk_to_trajectory(np.asarray(chunk, np.float64))[:, :opt.horizon]
    opt.path.begin(qn, scene, state)
    worst, record = opt.path.violation(traj, scene, record=True)
    wp, pair = opt.linearizer.worst_row(traj, qn, scene)
    return {"path": record, "path_clearance_m": float(worst), "waypoint_clearance_m": float(wp),
            "waypoint_pair": pair, "trajectory": traj, "q_now": qn, "optimizer": opt,
            "scene": scene, "servo_state": state}


def main(out_path: str) -> int:
    import benchmark

    arrays, meta = {}, {"benchmark_file": benchmark.__file__}
    for name, seq in IDENTITY:
        r = replay(name, seq)
        arrays[f"{name}_{seq}_chunk"] = r["chunk"]
        meta[f"{name}_{seq}"] = {"status": r["status"], "clearance_m": r["clearance_m"],
                                 "returned": r["metrics"].get("returned"),
                                 "metric_keys": sorted(r["metrics"])}
    np.savez(out_path, meta=json.dumps(meta), **arrays)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
