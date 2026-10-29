"""T65 — re-send a recorded closed-loop episode's requests to a running `serve_safe` (fixed-input replay).

`handoff/T65.task.md`: the attention source is the only thing that differs between the replays; the robot
trajectory is the recorded one. For every recorded request k of a client run directory:

* observation: the recorded policy images (HWC -> CHW, as `pi05_infer.build_obs` sends them), the recorded
  16-D state and the episode prompt (`traj.npz`);
* AG3S cameras: re-rendered from the recorded ``qpos`` / ``qvel`` with the recorded model
  (`policy_record.replay_scene` + `pose_scene`, `TransportScene.capture`) — float depth as the original client
  sent it, not the 1 mm ``uint16`` copy in ``step_*.npz``;
* ``exec_feedback``: the planning row's own ``exec_feedback`` (what the client actually sent with request k);
* stamps: the observation row's ``camera_stamps`` (``stamp_mode = sim_frozen``), ``reset`` on k = 0,
  ``seq = k + 1``, ``policy_seed`` on the first request only (`pi05_infer`), phase / manipulators from the
  client argv.

The server's answer is not used (no control loop); the server records its constraint chunks as usual.
Run with PYTHONPATH = the server's snapshot so `wire` / `policy_record` are the server's own versions.

usage: python t65_replay_client.py --client-dir DIR --port 8292 [--limit N]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np


def _arg(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def load_client(client_dir: pathlib.Path) -> dict:
    client_dir = pathlib.Path(client_dir)
    ap = client_dir.with_name(client_dir.name + ".argv.json")
    if ap.is_file():
        argv = json.loads(ap.read_text())["argv"]
    else:
        # occlusion launcher (`occ/launch_occ_CORE.sh`): phase approach · left arm, seed = the `_s<SEED>` suffix
        import re

        m = re.search(r"_s(\d+)$", client_dir.name)
        argv = ["--safe-phase", "approach", "--safe-manipulators", "left"] + (
            ["--policy-seed", m.group(1)] if m else [])
    rows = [json.loads(x) for x in (client_dir / "frames.jsonl").read_text().splitlines() if x.strip()]
    obs_rows = [r for r in rows if r.get("kind") == "observation"]
    plan_rows = [r for r in rows if r.get("kind") == "planning"]
    tj = np.load(client_dir / "traj.npz", allow_pickle=True)
    prompt = str(tj["prompt"]) if "prompt" in tj.files else None
    return {"argv": argv, "obs_rows": obs_rows, "plan_rows": plan_rows, "prompt": prompt,
            "record_dir": client_dir / "ag3s" / "run_0000"}


def replay(client_dir: pathlib.Path, port: int, limit: int | None = None, log=print) -> dict:
    from openpi_client import websocket_client_policy

    from AG3S.ag3s.experiments.sources import policy_record as pr
    from AG3S.trajopt import wire

    c = load_client(client_dir)
    rec = pr.load_run(c["record_dir"])
    scene = pr.replay_scene(rec)
    argv = c["argv"]
    phase = _arg(argv, "--safe-phase", "approach")
    manip = [m for m in str(_arg(argv, "--safe-manipulators", "")).split(",") if m]
    seed = _arg(argv, "--policy-seed")
    prompt = c["prompt"]
    if prompt is None:
        raise SystemExit(f"{client_dir}: no prompt in traj.npz")
    n = len(rec.steps)
    if len(c["plan_rows"]) < n or len(c["obs_rows"]) < n:
        raise SystemExit(f"{client_dir}: {n} steps but {len(c['plan_rows'])} planning / {len(c['obs_rows'])} "
                         "observation rows")
    policy = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=int(port))
    cams = tuple(wire.DEFAULT_CAMERAS)
    sent, t0 = 0, time.time()
    for k, step in enumerate(rec.steps[: (limit or n)]):
        pl, ob = c["plan_rows"][k], c["obs_rows"][k]
        if int(pl["t_step"]) != int(step.t_step) or int(ob["t_step"]) != int(step.t_step):
            raise SystemExit(f"{client_dir}: request {k} t_step mismatch ({step.t_step} / {pl['t_step']} / "
                             f"{ob['t_step']})")
        pr.pose_scene(scene, step)
        depth, K, T, state = {}, {}, {}, {}
        for cam in cams:
            fr = scene.capture(cam)
            depth[cam] = np.asarray(fr.depth, np.float64)
            K[cam] = np.asarray(fr.camera_intrinsics, np.float64)
            T[cam] = np.asarray(fr.T_base_cam, np.float64)
            state[cam] = np.asarray(fr.robot_state, np.float64)
        stamps = {cam: float(ob["camera_stamps"][cam]) for cam in cams}
        obs = {"state": np.asarray(step.state, np.float64),
               "images": {key: np.ascontiguousarray(np.transpose(np.asarray(img), (2, 0, 1)))
                          for key, img in step.images.items()},
               "prompt": prompt}
        fb = pl.get("exec_feedback") or wire.no_exec_feedback("replay: no exec_feedback in the planning row")
        req = wire.pack_request(obs, cameras=cams, depth=depth, intrinsics=K, extrinsics=T, robot_state=state,
                                stamps=stamps, phase=phase, active_manipulators=manip, reset=(k == 0),
                                seq=int(pl.get("seq", k + 1)), exec_feedback=fb, render_stamps=dict(stamps),
                                stamp_mode="sim_frozen",
                                policy_seed=(int(seed) if (k == 0 and seed is not None) else None))
        policy.infer(req)
        sent += 1
    log(f"[t65] {client_dir.name}: {sent} requests in {time.time() - t0:.0f}s")
    return {"client_dir": str(client_dir), "sent": sent, "seconds": time.time() - t0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--client-dir", required=True, type=pathlib.Path)
    ap.add_argument("--port", type=int, default=8292)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    print(json.dumps(replay(a.client_dir, a.port, a.limit)))
    sys.stdout.flush()
