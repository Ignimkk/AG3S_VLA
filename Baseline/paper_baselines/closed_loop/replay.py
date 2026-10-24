"""Recorded-observation replay (CPU smoke without the simulator): rebuild the exact wire requests
``libero_safe_client.py`` sent in a recorded run, send them to a method server, compare the responses.

The request of policy call ``k`` of an episode is rebuilt from the run's files as the client builds it
(``SafeClient._request`` + ``libero_pi05_client.run_episode``):

* policy keys: ``resize_with_pad(raw[::-1, ::-1], 224)`` of the recorded agentview / eye-in-hand image at the call's
  observation index (``chunk_t[k]``), the recorded ``policy_state[k]``, the task prompt;
* ``ag3s/*``: ``wire_pack.pack_cameras`` of that observation's ``_geom.npz`` row (depth mm, K, T_base_cam,
  robot_state, stamp), ``phase=approach``, ``active_manipulators=[left]``, ``seq``, ``reset`` / ``policy_seed`` at k=0;
* ``panda/*``: ``robot_state`` of the last wire camera, ``T_world_base`` of the first, ``prev_step_world`` from the
  recorded EE pose of the last two observations.

The episode is **open loop** (call k+1 is the recorded state, not the result of the served chunk).

    .venv-openpi-live/bin/python -m Baseline.paper_baselines.closed_loop.replay --port 8123 \
        --run <SL5c VLA-only run> --stem safelibero_spatial_t0_e0 --calls 0-11 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
PHASE, MANIPULATORS = "approach", ["left"]


def _rotvec_between(q1_xyzw, q0_xyzw):
    from scipy.spatial.transform import Rotation as Rot

    return (Rot.from_quat(q1_xyzw) * Rot.from_quat(q0_xyzw).inv()).as_rotvec()


def resolve_run(run) -> pathlib.Path:
    import os

    p = pathlib.Path(run)
    if p.is_dir():
        return p
    root = pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", PROJECT_ROOT.parent / "AG3S_VLA-storage"))
    return root / "results/benchmark" / str(run)


class RecordedEpisode:
    def __init__(self, run, stem: str):
        self.run = resolve_run(run)
        self.stem = stem
        rows = [json.loads(l) for l in (self.run / "episodes.jsonl").read_text().splitlines() if l.strip()]
        self.row = next(r for r in rows if r["npz"].endswith(f"/{stem}.npz"))
        self.z = dict(np.load(self.run / "episodes" / f"{stem}.npz"))
        self.g = np.load(self.run / "episodes" / f"{stem}_geom.npz")
        self.calls = dict(np.load(self.run / "episodes" / f"{stem}_calls.npz"))
        self.policy_seed = (self.row.get("safe_client") or {}).get("policy_seed")
        self.cameras = [str(c) for c in self.g["cameras"]]
        self.obs_index = np.asarray(self.g["obs_index"])
        self.n_calls = int(len(self.z["chunk_t"]))

    def request(self, k: int) -> dict:
        from openpi_client import image_tools

        sys.path.insert(0, str(PROJECT_ROOT))
        from benchmark.adapters.libero_panda import wire_pack

        t = int(self.z["chunk_t"][k])
        img = np.ascontiguousarray(self.z["obs_agentview_image"][t][::-1, ::-1])
        wrist = np.ascontiguousarray(self.z["obs_robot0_eye_in_hand_image"][t][::-1, ::-1])
        req = {"observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224)),
               "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224)),
               "observation/state": np.asarray(self.z["policy_state"][k], np.float64),
               "prompt": str(self.row["task_description"])}
        r = int(np.flatnonzero(self.obs_index == t)[0])
        frames = {}
        for cam in self.cameras:
            frames[cam] = {"depth_mm": np.asarray(self.g[f"depth_mm/{cam}"][r]), "K": self.g[f"K/{cam}"][r],
                           "T_base_cam": self.g[f"T_base_cam/{cam}"][r], "robot_state": self.g[f"robot_state/{cam}"][r],
                           "stamp": float(self.g[f"stamp/{cam}"][r])}
        req.update(wire_pack.pack_cameras(frames, stamp_mode=str(self.g["stamp_mode"])))
        req.update({"ag3s/phase": PHASE, "ag3s/active_manipulators": list(MANIPULATORS), "ag3s/seq": int(k),
                    "ag3s/reset": k == 0})
        if k == 0 and self.policy_seed is not None:
            req["ag3s/policy_seed"] = int(self.policy_seed)
        req["panda/robot_state"] = np.asarray(self.g[f"robot_state/{self.cameras[-1]}"][r], np.float64)
        req["panda/T_world_base"] = np.asarray(self.g["T_world_base"][r], np.float64)
        p1, q1 = self.z["obs_robot0_eef_pos"][t], self.z["obs_robot0_eef_quat"][t]
        if t >= 1:
            p0, q0 = self.z["obs_robot0_eef_pos"][t - 1], self.z["obs_robot0_eef_quat"][t - 1]
            req["panda/prev_step_world"] = np.concatenate([p1 - p0, _rotvec_between(q1, q0)])
        else:
            req["panda/prev_step_world"] = np.zeros(6)
        return req


def readonly(request: dict) -> dict:
    """The request as the server sees it after the openpi msgpack decoder: every array read-only."""
    out = {}
    for k, v in request.items():
        if isinstance(v, np.ndarray):
            v = v.copy()
            v.setflags(write=False)
        out[k] = v
    return out


def parse_calls(text: str, n: int) -> list[int]:
    if text == "all":
        return list(range(n))
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [c for c in out if c < n]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--calls", default="0-5")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from openpi_client import websocket_client_policy

    ep = RecordedEpisode(a.run, a.stem)
    calls = parse_calls(a.calls, ep.n_calls)
    if calls and calls[0] != 0:
        calls = [0] + calls  # the episode starts with the reset call (policy_seed)
    client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", a.port)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, served, recorded = [], [], []
    for k in calls:
        req = ep.request(k)
        t0 = time.perf_counter()
        resp = client.infer(req)
        dt = (time.perf_counter() - t0) * 1e3
        act = np.asarray(resp["actions"], np.float64)
        rec = np.asarray(ep.calls["actions_response"][k], np.float64)
        rows.append({"call": k, "obs_index": int(ep.z["chunk_t"][k]), "client_ms": dt, "mode": resp.get("panda/mode"),
                     "timing_ms": resp.get("timing_ms"), "policy_seed_echo": resp.get("policy_seed"),
                     "shape": list(act.shape), "max_abs_vs_recorded": float(np.abs(act - rec).max()),
                     "bit_identical_to_recorded": bool(act.shape == rec.shape and np.array_equal(act, rec))})
        served.append(act)
        recorded.append(rec)
        print(json.dumps(rows[-1]), flush=True)
    np.savez(out / f"{a.stem}_replay.npz", calls=np.asarray(calls), served=np.stack(served), recorded=np.stack(recorded))
    (out / f"{a.stem}_replay.json").write_text(json.dumps({"run": str(ep.run), "stem": a.stem, "rows": rows,
                                                           "server_metadata": client.get_server_metadata()},
                                                          indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
