"""LIBERO client for paired VLA-only vs VLA+V11AB runs (SL5a; **Python 3.8**, ``.venv-libero``).

The episode loop is the official one (``libero_pi05_client.run_tasks`` / ``run_episode``, protocol constants and
reset history unchanged). What this module adds is the request/response handling, identical for both server
modes, so the two conditions differ only in the server:

request (every policy call) = official keys (image, wrist_image, state, prompt) +
  * ``ag3s/*`` camera geometry of the same observation (SL1 adapter: depth mm, K, T_base_cam, robot_state, stamp),
    ``ag3s/phase=approach``, ``ag3s/active_manipulators=[left]``, ``ag3s/seq``, ``ag3s/reset`` (first call),
    ``ag3s/policy_seed`` (first call, per episode), ``ag3s/exec_feedback`` (V11AB mode, from the second call);
  * ``panda/robot_state`` (9-D, image instant), ``panda/T_world_base``, ``panda/prev_step_world`` (last realised EE
    step, world frame); from the second call ``panda/exec_applied_osc`` (the OSC rows actually applied for
    the previous chunk, confirmed after they ran; the server turns them into ``exec_feedback.applied_arm``).

response: ``actions`` (EE deltas, already converted by the server in V11AB mode) and, in V11AB mode, ``safe`` —
``safe == False`` executes **HOLD** rows (zero arm delta, last gripper command), as the V11AB client holds.

Per episode it writes ``episodes/<stem>_calls.jsonl`` (one row per policy call: mode, safe, verdict reasons, AG3S /
TrajOpt status, conversion record, max action change, timings) and ``episodes/<stem>_calls.npz`` (per call:
policy chunk, executed rows, refined/reference joint chunks when present).
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import libero_pi05_client as base  # noqa: E402

PHASE = "approach"            # V11AB client: --safe-phase approach (profile.v11ab_client_phase)
MANIPULATORS = ["left"]       # Panda = V11AB left manipulator (SL2)
FINGER_OPEN_M = 0.04


def episode_seed(seed, task_id, episode_idx):
    """Deterministic per-episode policy seed, the same in both modes (paired)."""
    return int(seed) * 100000 + int(task_id) * 1000 + int(episode_idx)


def _rotvec_between(q1_xyzw, q0_xyzw):
    from scipy.spatial.transform import Rotation as Rot

    return (Rot.from_quat(q1_xyzw) * Rot.from_quat(q0_xyzw).inv()).as_rotvec()


class SafeClient(object):
    """Wraps the websocket policy client; one instance per episode (``policy_for``)."""

    def __init__(self, ws_client, seed, task_id, episode_idx, replan_steps):
        self.ws = ws_client
        self.seed = episode_seed(seed, task_id, episode_idx)
        self.replan = int(replan_steps)
        self.geometry = None
        self.eef = []          # (pos, quat) per observation, world frame
        self.calls = []        # per-call log rows
        self.arrays = collections.defaultdict(list)
        self.seq = 0
        self.last_gripper = -1.0
        self.sent = None       # the chunk the last call handed to the env loop; its feedback is confirmed later
        self.feedback_log = []
        self.camera = None
        self.wire_pack = None

    # --- run_tasks hooks --------------------------------------------------------------------------------------
    def observer(self, geometry):
        self.geometry = geometry
        self.camera, self.wire_pack = geometry.camera, geometry.wire_pack

        def observe(env, obs):
            geometry.observe(env, obs)
            self.eef.append((np.array(obs["robot0_eef_pos"], np.float64), np.array(obs["robot0_eef_quat"], np.float64)))
        return observe

    def get_server_metadata(self):
        return self.ws.get_server_metadata()

    # --- one policy call ----------------------------------------------------------------------------------------
    def _request(self, element):
        frames = self.geometry.frames[-1]
        req = dict(element)
        req.update(self.wire_pack.pack_cameras(frames, stamp_mode=self.camera.STAMP_MODE))
        first = self.seq == 0
        req.update({"ag3s/phase": PHASE, "ag3s/active_manipulators": list(MANIPULATORS), "ag3s/seq": self.seq,
                    "ag3s/reset": bool(first)})
        if first:
            req["ag3s/policy_seed"] = self.seed
        cam0 = self.geometry.cameras[-1]  # left_wrist robot_state (both cameras share the image instant)
        rs9 = np.asarray(frames[cam0]["robot_state"], np.float64)
        if self.sent is not None:
            fb, osc = self._confirmed_feedback(rs9)
            req["ag3s/exec_feedback"] = fb
            req["panda/exec_applied_osc"] = osc
        req["panda/robot_state"] = rs9
        req["panda/T_world_base"] = np.asarray(frames[self.geometry.cameras[0]]["T_world_base"], np.float64)
        if len(self.eef) >= 2:
            (p1, q1), (p0, q0) = self.eef[-1], self.eef[-2]
            req["panda/prev_step_world"] = np.concatenate([p1 - p0, _rotvec_between(q1, q0)])
        else:
            req["panda/prev_step_world"] = np.zeros(6)
        return req, rs9

    def _confirmed_feedback(self, rs9):
        """exec_feedback (T18 form) of the last chunk, confirmed **after** its rows ran.

        ``n_steps`` = env steps actually taken since that call (observations appended since), not the rows
        handed over. ``measured_gripper`` = this request's capture (wire contract). ``applied_arm`` is filled by
        the Panda server from ``panda/exec_applied_osc`` (the OSC rows actually applied, HOLD rows included),
        with the conversion of the request that sent them (``action.applied_command_rows``).
        """
        s = self.sent
        n = max(0, min(s["applied"].shape[0], len(self.eef) - 1 - s["t_call"]))
        applied, planned = s["applied"][:n], s["planned"][:n]
        g_app = (1.0 - np.clip(applied[:, 6], -1, 1)) / 2.0
        g_plan = (1.0 - np.clip(planned[:, 6], -1, 1)) / 2.0
        safe = s["safe"]
        fb = {"available": True, "seq": s["seq"], "t_step_start": s["t_call"], "n_steps": int(n),
              "n_exec": int(n) if safe else 0, "n_hold": 0 if safe else int(n),
              "executed": [bool(safe)] * n, "executed_chunk": "refined" if safe else "none", "ipc": "websocket",
              "hold_kind": None if safe else "unsafe", "hold_reason": None if safe else "server safe=False",
              "planned_gripper": np.stack([g_plan, np.ones_like(g_plan)], 1).astype(np.float32),
              "applied_gripper": np.stack([g_app, np.ones_like(g_app)], 1).astype(np.float32),
              "measured_gripper": np.array([abs(float(rs9[7])) / FINGER_OPEN_M, 1.0], np.float32)}
        self.feedback_log.append({"seq": s["seq"], "n_steps": int(n), "handed": int(s["applied"].shape[0]),
                                  "measured_gripper": float(fb["measured_gripper"][0]), "obs_index": len(self.eef) - 1})
        return fb, applied.astype(np.float64)

    def infer(self, element):
        req, rs9 = self._request(element)
        t0 = time.perf_counter()
        resp = self.ws.infer(req)
        dt = (time.perf_counter() - t0) * 1000.0
        mode = resp.get("panda/mode", "unknown")
        actions = np.asarray(resp["actions"])
        safe = bool(resp.get("safe", True)) if mode == "vla_v11ab" else True
        n = min(self.replan, actions.shape[0])
        if safe:
            executed = np.array(actions[:n], copy=True)
            if n:
                self.last_gripper = float(executed[n - 1, 6])
        else:
            executed = np.zeros((n, 7), actions.dtype)
            executed[:, 6] = self.last_gripper
        out = np.array(actions, copy=True)
        out[:n] = executed
        # V11AB mode: remember what goes to the env loop; the feedback is confirmed at the next request (T18).
        if mode == "vla_v11ab":
            self.sent = {"seq": int(self.seq), "t_call": int(len(self.eef) - 1), "safe": bool(safe),
                         "applied": np.asarray(executed, np.float64), "planned": np.asarray(actions[:n], np.float64)}
        row = {"seq": int(self.seq), "obs_index": int(len(self.eef) - 1), "mode": mode, "safe": safe,
               "hold": not safe, "t_client_ms": dt,
               "verdict_reasons": base._jsonable(resp.get("verdict_reasons")),
               "ag3s_status": resp.get("ag3s_status"), "trajopt_status": resp.get("trajopt_status"),
               "max_violation_m": base._jsonable(resp.get("max_violation_m")),
               "ag3s": base._jsonable(resp.get("ag3s")), "to": base._jsonable(resp.get("to")),
               "conversion": base._jsonable(resp.get("panda/conversion")),
               "grasp": base._jsonable(resp.get("panda/grasp")),
               "latch_signal": base._jsonable(resp.get("panda/latch_signal")),
               "timing_ms": base._jsonable(resp.get("timing_ms")), "policy_seed": resp.get("policy_seed"),
               "feedback_sent": self.feedback_log[-1] if (self.feedback_log and "ag3s/exec_feedback" in req) else None,
               "feedback_server": base._jsonable(resp.get("panda/feedback"))}
        if "panda/actions_policy" in resp:
            pol = np.asarray(resp["panda/actions_policy"], np.float64)
            row["max_abs_action_change_executed"] = float(np.max(np.abs(executed.astype(np.float64) - pol[:n]))) if n else 0.0
            self.arrays["actions_policy"].append(pol)
            self.arrays["joint_refined"].append(np.asarray(resp["panda/joint_refined"], np.float32))
            self.arrays["joint_reference"].append(np.asarray(resp["panda/joint_reference"], np.float32))
            self.arrays["predicted_ee_base"].append(np.asarray(resp["panda/predicted_ee_base"], np.float32))
            self.arrays["T_world_base"].append(np.asarray(req["panda/T_world_base"], np.float64))
        self.arrays["actions_response"].append(np.asarray(actions, np.float64))
        self.arrays["actions_executed"].append(np.asarray(executed, np.float64))
        self.arrays["call_obs_index"].append(len(self.eef) - 1)
        self.calls.append(row)
        self.seq += 1
        return {"actions": out, "server_timing": resp.get("server_timing"), "policy_timing": resp.get("policy_timing")}

    def save(self, episodes_dir, stem):
        path = episodes_dir / (stem + "_calls.jsonl")
        with open(str(path), "x") as f:
            for row in self.calls:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        arrays = {}
        for key, val in self.arrays.items():
            try:
                arrays[key] = np.stack(val)
            except ValueError:
                arrays[key] = np.array(val, dtype=object)
        tmp = base._tmp_sibling(episodes_dir / (stem + "_calls.npz"), ".npz")
        np.savez_compressed(str(tmp), **{k: v for k, v in arrays.items() if v.dtype != object})
        base._exclusive_publish(tmp, episodes_dir / (stem + "_calls.npz"))
        holds = sum(1 for r in self.calls if r["hold"])
        changed = [r.get("max_abs_action_change_executed", 0.0) for r in self.calls]
        return {"calls_log": str(path.name), "calls_npz": stem + "_calls.npz", "policy_seed": self.seed,
                "server_mode": self.calls[0]["mode"] if self.calls else None, "hold_calls": holds,
                "calls": len(self.calls), "max_abs_action_change_executed": float(max(changed) if changed else 0.0)}


def main(argv=None):
    parser = base.build_parser()
    parser.description = __doc__.splitlines()[0]
    args = parser.parse_args(argv)
    args.capture_geometry = True  # the request carries the camera geometry in both modes
    base.check_geometry_arguments(args)
    run_dir = base.validate_run_dir(args.run_dir)
    base.require_libero_config()
    from libero.libero import benchmark

    episodes_dir = run_dir / "episodes"
    (run_dir / "videos").mkdir(exist_ok=True)
    episodes_dir.mkdir(exist_ok=True)
    jsonl = open(str(run_dir / "episodes.jsonl"), "x")
    np.random.seed(args.seed)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    max_steps = base.MAX_STEPS[args.task_suite_name]
    ws, connector = base.make_policy_client(args.host, args.port)

    def policy_for(task_id, episode_idx):
        return SafeClient(ws, args.seed, task_id, episode_idx, args.replan_steps), connector, {}

    def observer_for(client, geometry, task_id, episode_idx):
        return client.observer(geometry)

    def after_episode(client, stem, row):
        return {"safe_client": client.save(episodes_dir, stem)}

    with jsonl:
        errors = base.run_tasks(args, run_dir, jsonl, task_suite, max_steps, policy_for,
                                observer_for=observer_for, after_episode=after_episode)
    return 1 if errors else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
