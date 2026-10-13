"""SL5a offline check: recorded LIBERO dev episodes through the Panda V11AB server path (no policy model).

For every recorded policy call of a pi0.5-only dev run (SL3/SL4b), the same request the live client sends is
built (official keys are not needed: the policy is a stub that returns the **recorded** chunk and the
recorded V11AB-cell attention), and passed to ``AG3S.adapters.panda.server.build_safe_server`` — i.e. the real
``SafePolicy`` (perception, ESDF, TrajOpt CCR, gate, verdict) with the Panda injections — and the response is
converted back to EE deltas. The trajectory is the recorded one (open loop: V11AB's output is not executed).

Writes a new run (``create_run``): ``ccr/<stem>.jsonl`` (one row per call) and ``summary.json``.

    .venv-openpi-live/bin/python benchmark/runners/ccr_replay.py --source-run <dev run> --attention-run <audit run> \
        --task-ids 0 --run-id sl5a-ccr-t0e0 [--esdf-backend legacy]   # GPU (curobo, V11AB) under the shared lock
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class RecordedPolicy:
    """Stub policy: the recorded chunk and attention of the current call (set before each request)."""

    def __init__(self):
        self.chunk = None
        self.attention = None

    def infer(self, obs, **kwargs):
        return {"actions": np.asarray(self.chunk), "attention": dict(self.attention)}


def panda_server_args(esdf_backend: str = "curobo"):
    from AG3S.adapters.panda import profile

    argv = list(profile.v11ab_server_argv())
    i = argv.index("--config")
    argv[i + 1] = "pi05_libero"
    argv[argv.index("--checkpoint") + 1] = "<offline: recorded chunks>"
    argv += ["--robot", "panda"]
    if esdf_backend != "curobo":
        argv[argv.index("--esdf-backend") + 1] = esdf_backend
        j = argv.index("--target-field-policy")
        del argv[j:j + 2]  # serve_safe refuses it without the cuRobo backend (as the SL5a golden does)
    return profile.parse_v11ab_args(argv=argv), argv


def prev_step_world(ep_arrays: dict, t: int) -> np.ndarray:
    """Realised EE step ending at observation ``t`` (world): what the live client sends."""
    from scipy.spatial.transform import Rotation as Rot

    if t <= 0:
        return np.zeros(6)
    p, q = ep_arrays["obs_robot0_eef_pos"], ep_arrays["obs_robot0_eef_quat"]
    return np.concatenate([p[t] - p[t - 1], (Rot.from_quat(q[t]) * Rot.from_quat(q[t - 1]).inv()).as_rotvec()])


def recorded_feedback(conv, seq: int, ep_arr: dict, t0: int, t_next, geom, row_of) -> dict:
    """exec_feedback for the next request: the recorded rows (= the policy's reference) were executed."""
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt import wire

    n = 5 if t_next is None else int(t_next - t0)
    n = max(0, min(n, conv.joint_chunk.shape[0]))
    g = conv.joint_chunk[:n, 7]
    pair = np.stack([g, np.ones_like(g)], 1)
    row_next = row_of.get(t0 + n)
    opening = 1.0 if row_next is None else ps.gripper_opening_norm(geom["robot_state/left_wrist"][row_next])
    return wire.make_exec_feedback(seq=seq, t_step_start=t0, executed=[True] * n, executed_chunk="reference",
                                   ipc="offline-replay", hold_kind=None, hold_reason=None, planned_gripper=pair,
                                   applied_gripper=pair, applied_arm=conv.joint_chunk[:n, :7],
                                   measured_gripper=np.array([opening, 1.0]))


def request_for(geom, row: int, seq: int, reset: bool) -> dict:
    from benchmark.adapters.libero_panda import wire_pack

    req = dict(wire_pack.wire_from_geom(geom, row))
    req.update({"ag3s/phase": "approach", "ag3s/active_manipulators": ["left"], "ag3s/seq": int(seq),
                "ag3s/reset": bool(reset),
                "panda/robot_state": np.asarray(geom["robot_state/left_wrist"][row], np.float64),
                "panda/T_world_base": np.asarray(geom["T_world_base"][row], np.float64)})
    return req


def _j(v):
    if isinstance(v, dict):
        return {str(k): _j(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_j(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    return v


def run(args) -> int:
    from AG3S.adapters.panda import server as ps
    from benchmark.runners import perception_replay as pr
    from common.run_artifacts import create_run

    sources = pr.index_sources(args.source_run, args.attention_run, task_ids=args.task_ids)
    srv_args, argv = panda_server_args(args.esdf_backend)
    stub = RecordedPolicy()
    served, info = ps.build_safe_server(stub, srv_args)
    settings = {"step": "SL5a", "purpose": "offline CCR replay (recorded chunks, open loop)",
                "server_argv": argv, "panda_server": _j(info), "sources": _j({k: v for k, v in sources.items() if k != "episodes"}),
                "episodes": [e["stem"] for e in sources["episodes"]], "limit_calls": args.limit_calls}
    run_dir = create_run(run_id=args.run_id, settings=settings)
    print(run_dir, flush=True)
    out = run_dir / "ccr"
    out.mkdir()
    spec = pr.AttentionSpec(mapping="unmirror")
    totals = {"calls": 0, "rows": 0, "passthrough_rows": 0, "corrected_rows": 0, "saturated_rows": 0,
              "unsafe_calls": 0, "passthrough_rows_bit_identical": 0, "t_ms": []}
    for ep in sources["episodes"]:
        dev_dir = Path(ep["dev_run_dir"])
        with np.load(dev_dir / ep["geom_npz"]) as g:
            geom = {k: g[k] for k in g.files}
        with np.load(dev_dir / ep["npz"]) as e:
            chunks = np.asarray(e["action_chunks"])
            chunk_t = np.asarray(e["chunk_t"]).astype(np.int64)
            ep_arr = {k: np.asarray(e[k]) for k in ("obs_robot0_eef_pos", "obs_robot0_eef_quat")}
        with np.load(ep["attn_npz"]) as a:
            attn = {k: a[k] for k in ("obs_index", "global_step", spec.grid_key())}
        grids, _ = pr.episode_attention_grids(attn, spec)
        row_of = {int(o): i for i, o in enumerate(np.asarray(geom["obs_index"]).reshape(-1))}
        served.reset()
        feedback = None
        n = len(chunk_t) if not args.limit_calls else min(args.limit_calls, len(chunk_t))
        with open(out / f"{ep['stem']}.jsonl", "x") as f:
            for k in range(n):
                row = row_of[int(chunk_t[k])]
                stub.chunk = chunks[k].astype(np.float32)
                stub.attention = {"cam_high": grids[k, 0], "cam_left_wrist": grids[k, 1]}
                t0 = time.perf_counter()
                req = request_for(geom, row, k, k == 0)
                req["panda/prev_step_world"] = prev_step_world(ep_arr, int(chunk_t[k]))
                if feedback is not None:
                    req["ag3s/exec_feedback"] = feedback
                resp = served.infer(req)
                conv_obj = served.chunk_policy.last
                T_wb = np.asarray(geom["T_world_base"][row], np.float64)
                pred = [T_wb[:3, :3] @ pb + T_wb[:3, 3] for pb, _ in conv_obj.targets_base]
                t0i = int(chunk_t[k])
                n_exec = min(5, len(ep_arr["obs_robot0_eef_pos"]) - 1 - t0i)
                pred_err = [float(np.linalg.norm(pred[j] - ep_arr["obs_robot0_eef_pos"][t0i + 1 + j])) for j in range(max(n_exec, 0))]
                dt = (time.perf_counter() - t0) * 1000.0
                conv = resp["panda/conversion"]
                pt = conv["passthrough_rows"]
                same = int(sum(np.array_equal(resp["actions"][r], stub.chunk[r]) for r in pt))
                rec = {"stem": ep["stem"], "call": k, "obs_index": int(chunk_t[k]), "safe": bool(resp.get("safe")),
                       "verdict_reasons": _j(resp.get("verdict_reasons")), "ag3s_status": resp.get("ag3s_status"),
                       "trajopt_status": resp.get("trajopt_status"), "max_violation_m": _j(resp.get("max_violation_m")),
                       "conversion": _j(conv), "passthrough_bit_identical": same, "t_server_ms": dt,
                       "forward_pred_err_m_executed_rows": pred_err,
                       "timing_ms": _j(resp.get("timing_ms")), "to": _j(resp.get("to")), "ag3s": _j(resp.get("ag3s")),
                       "max_abs_action_change": float(np.max(np.abs(resp["actions"].astype(np.float64)
                                                                     - stub.chunk.astype(np.float64))))}
                f.write(json.dumps(rec) + "\n")
                # Open loop: the recorded (reference) rows are what ran; tell the server so (exec_feedback, T18).
                feedback = recorded_feedback(conv_obj, k, ep_arr, int(chunk_t[k]),
                                             int(chunk_t[k + 1]) if k + 1 < len(chunk_t) else None, geom, row_of)
                totals["calls"] += 1
                totals["rows"] += conv["rows"]
                totals["passthrough_rows"] += len(pt)
                totals["corrected_rows"] += len(conv["corrected_rows"])
                totals["saturated_rows"] += len(conv["saturated_rows"])
                totals["passthrough_rows_bit_identical"] += same
                totals["unsafe_calls"] += int(not rec["safe"])
                totals["t_ms"].append(dt)
                print(f"{ep['stem']} call {k} safe={rec['safe']} pass={len(pt)} corr={len(conv['corrected_rows'])} "
                      f"sat={len(conv['saturated_rows'])} maxda={rec['max_abs_action_change']:.4f} {dt:.0f}ms", flush=True)
    t = np.asarray(totals.pop("t_ms")[1:] or [np.nan])
    totals["t_server_ms_p50_p95_excl_first"] = [float(np.nanpercentile(t, 50)), float(np.nanpercentile(t, 95))]
    (run_dir / "summary.json").write_text(json.dumps(totals, indent=1))
    print(json.dumps(totals))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source-run", required=True)
    ap.add_argument("--attention-run", required=True)
    ap.add_argument("--task-ids", type=lambda s: [int(x) for x in s.split(",")], default=None)
    ap.add_argument("--limit-calls", type=int, default=None)
    ap.add_argument("--esdf-backend", default="curobo", choices=("curobo", "legacy"))
    ap.add_argument("--run-id", required=True)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
