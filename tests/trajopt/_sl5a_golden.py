"""SL5a golden: the RB-Y1 ``SafePolicy`` / ``serve_safe`` path on a fixed offline input.

SL5a adds optional injection points to ``AG3S/trajopt/safe_policy.py`` and ``serve_safe.py`` (Panda).
Their defaults must reproduce the RB-Y1 behaviour **bit for bit**. This module captures that behaviour on a
fixed, deterministic CPU input and compares a later run against it.

What runs (everything that ``serve_safe.main`` does for V11AB except loading openpi):

* flags: the V11AB server flags read from ``scripts/benchmark/v11ab/start_server.sh`` / ``env.sh``
  (``AG3S.adapters.panda.profile.v11ab_server_argv``) with ``--esdf-backend legacy`` (CPU, deterministic;
  the cuRobo backend needs a GPU) and without ``--target-field-policy`` (serve_safe refuses it on the legacy
  backend), parsed by ``serve_safe.build_parser`` + its own checks;
* ``to_config = serve_safe.trajopt_config_from_args(args)``, ``ag3s = serve_safe.build_ag3s(...)`` with the
  keywords of ``main``'s call (``profile.v11ab_build_kwargs``), ``static_geometry`` / ``attention_fn`` as in
  ``main``; ``SafePolicy(stub, ...)`` with no recorder, not shadow;
* input: the RB-Y1 transport scene (``TransportScene``, osmesa, 240x320), three wire cameras, the 16-D client
  state, a stub policy whose 50-row chunk moves the left arm and closes the left gripper in the third request,
  Gaussian attention on the apple (``mujoco_source.gaussian_attention``) under the policy camera names; the
  gripper opening of the scene set per request (:data:`OPENINGS`); six requests (reset, then exec_feedback of the executed refined rows each time, the scene advanced to the last
  executed row).

Inputs (wire requests) and outputs (every response key except wall-clock timings, ``SafePolicy._exec_summary()``,
``metadata``, the refiner's trajectory) are stored in ``tests/trajopt/data/sl5a_rby1_golden.npz``.

    # capture (before the core change) -- writes the fixture, refuses to overwrite
    MUJOCO_GL=osmesa .venv-ag3s/bin/python -m tests.trajopt._sl5a_golden --write tests/trajopt/data/sl5a_rby1_golden.npz
    # compare the working tree against the fixture
    MUJOCO_GL=osmesa .venv-ag3s/bin/python -m tests.trajopt._sl5a_golden --check tests/trajopt/data/sl5a_rby1_golden.npz
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/trajopt/data/sl5a_rby1_golden.npz"
HEIGHT, WIDTH = 240, 320
N_REQUESTS = 6
#: measured left/right gripper opening (client norm, 1 = open) set in the scene before each capture
OPENINGS = (1.0, 1.0, 1.0, 0.75, 0.72, 0.72)
CHUNK_H = 50
#: response keys whose value is a wall clock (not part of the comparison)
TIMING_KEYS = ("timing_ms",)
TO_TIMING_KEYS = ("solve_ms",)
#: field-provenance keys that are a wall clock
FIELD_TIMING_KEYS = ("built_at",)
CAMERAS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
POLICY_CAMERAS = {"zed_left": "cam_high", "wrist_cam_l": "cam_left_wrist", "wrist_cam_r": "cam_right_wrist"}


# --- the server, as serve_safe.main builds it --------------------------------------------------------
#: V11AB flags the CPU (legacy ESDF) golden cannot take: serve_safe refuses ``--target-field-policy`` without
#: the cuRobo backend (the target-free layer exists only there). Recorded in the fixture provenance.
CPU_DROPPED_FLAGS = ("--target-field-policy",)


def v11ab_cpu_argv() -> list[str]:
    """V11AB server argv with ``--esdf-backend legacy`` and :data:`CPU_DROPPED_FLAGS` (and their value) removed."""
    from AG3S.adapters.panda import profile as pf

    argv = pf.v11ab_server_argv()
    i = argv.index("--esdf-backend")
    argv[i + 1] = "legacy"
    for flag in CPU_DROPPED_FLAGS:
        if flag in argv:
            j = argv.index(flag)
            del argv[j:j + 2]
    return argv


def build_server(policy):
    from AG3S.adapters.panda import profile as pf
    from AG3S.trajopt import serve_safe as ss
    from AG3S.trajopt.safe_policy import SafePolicy

    args = pf.parse_v11ab_args(argv=v11ab_cpu_argv())
    to_config = ss.trajopt_config_from_args(args)
    kw = pf.v11ab_build_kwargs(args)
    ag3s = ss.build_ag3s(args.model_xml, plan_horizon_steps=to_config.horizon.planned,
                         rows_per_step=to_config.reduction.rows_per_step, **kw)
    served = SafePolicy(policy, ag3s=ag3s,
                        static_geometry=ss.load_static_geometry(args.static_geometry, args.model_xml,
                                                                links=args.links),
                        to_config=to_config, attention_fn=ss.attention_extractor(), recorder=None,
                        shadow=args.shadow)
    return served, args


# --- the input ---------------------------------------------------------------------------------------
class StubPolicy:
    """Returns the prepared chunk (and attention) of request ``i``; ``reset`` is counted, not acted on."""

    def __init__(self, chunks, attention):
        self.chunks, self.attention, self.i, self.resets = chunks, attention, 0, 0

    @property
    def metadata(self):
        return {"stub": "sl5a_golden"}

    def reset(self):
        self.resets += 1

    def infer(self, _obs, **_):
        i = min(self.i, len(self.chunks) - 1)
        self.i += 1
        return {"actions": np.array(self.chunks[i], np.float64),
                "attention": {k: np.array(v, np.float32) for k, v in self.attention[i].items()}}


def _state16(scene) -> np.ndarray:
    from AG3S.ag3s.robot_models.gripper_state import load_client_convention, openings_from_mujoco

    q = scene.robot_state()
    names = list(__import__("AG3S.ag3s.robot_models", fromlist=["DEFAULT_RBY1_JOINTS"]).DEFAULT_RBY1_JOINTS)
    open_ = openings_from_mujoco(scene.model, scene.data, load_client_convention())
    left = [q[names.index(f"left_arm_{i}")] for i in range(7)]
    right = [q[names.index(f"right_arm_{i}")] for i in range(7)]
    return np.asarray(left + [open_["left"]] + right + [open_["right"]], np.float64)


def _chunk(state16: np.ndarray, request: int) -> np.ndarray:
    """50 x 16: left arm swept toward a fixed offset, right arm still; grippers open, left closing in request 2."""
    t = np.linspace(0.0, 1.0, CHUNK_H)[:, None]
    offset = np.array([0.20, 0.10, -0.10, -0.25, 0.05, 0.10, 0.0])
    out = np.tile(state16, (CHUNK_H, 1))
    out[:, 0:7] = state16[0:7] + t * offset * (1.0 + 0.25 * request)
    out[:, 7] = 1.0 if request < 2 else 0.0
    out[:, 15] = 1.0
    if request == 2:
        out[:3, 7] = 1.0
        out[3:, 7] = np.linspace(0.8, 0.0, CHUNK_H - 3)
    return out


def _set_opening(scene, norm: float) -> None:
    """Both hands to ``norm`` the client's way (driven joint = norm * RBY1_GRIPPER_OPEN, partner mirrored as the
    MJCF ``left_finger`` / ``right_finger`` equality ``polycoef="0 -1"``)."""
    from AG3S.ag3s.robot_models.gripper_state import load_client_convention

    conv = load_client_convention()
    for hand, partner in (("left", "gripper_finger_l2"), ("right", "gripper_finger_r2")):
        value = conv.driven_value(norm)
        scene.data.qpos[scene._qadr[conv.driven_joints[hand]]] = value  # noqa: SLF001
        scene.data.qpos[scene._qadr[partner]] = -value  # noqa: SLF001
    scene.mujoco.mj_forward(scene.model, scene.data)


def build_requests():
    """``(requests, chunks, attention)`` -- deterministic, rendered once."""
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene, gaussian_attention
    from AG3S.trajopt import wire

    scene = TransportScene(height=HEIGHT, width=WIDTH)
    names = list(__import__("AG3S.ag3s.robot_models", fromlist=["DEFAULT_RBY1_JOINTS"]).DEFAULT_RBY1_JOINTS)
    requests, chunks, attention = [], [], []
    K = 8
    previous = None
    for i in range(N_REQUESTS):
        if previous is not None:
            row = previous[K - 1]
            for j in range(7):
                scene.set_joint(f"left_arm_{j}", float(row[j]))
        _set_opening(scene, OPENINGS[i])
        frames = {cam: scene.capture(cam) for cam in CAMERAS}
        state = _state16(scene)
        apple = scene.body_position_in_base("apple")
        attention.append({POLICY_CAMERAS[c]: gaussian_attention(frames[c], apple) for c in CAMERAS})
        chunk = _chunk(state, i)
        chunks.append(chunk)
        if previous is None:
            feedback = wire.no_exec_feedback("first chunk of the episode")
        else:
            planned = np.stack([previous[:K, 7], previous[:K, 15]], 1)
            feedback = wire.make_exec_feedback(
                seq=i - 1, t_step_start=(i - 1) * K, executed=[True] * K, executed_chunk="refined",
                ipc="ok", hold_kind=None, hold_reason=None, planned_gripper=planned,
                applied_gripper=planned, applied_arm=np.concatenate([previous[:K, 0:7], previous[:K, 8:15]], 1),
                measured_gripper=state[[7, 15]])
        stamp = 100.0 + i
        req = wire.pack_request(
            {"state": state, "prompt": "pick up the apple"}, cameras=CAMERAS,
            depth={c: frames[c].depth for c in CAMERAS},
            intrinsics={c: frames[c].camera_intrinsics for c in CAMERAS},
            extrinsics={c: frames[c].T_base_cam for c in CAMERAS},
            robot_state={c: frames[c].robot_state for c in CAMERAS},
            stamps={c: stamp for c in CAMERAS}, phase="approach", active_manipulators=["left"],
            reset=(i == 0), seq=i, exec_feedback=feedback, stamp_mode="sim_frozen")
        requests.append(req)
        # the "executed" rows of the next feedback are the policy's own chunk here (the stub client does not
        # wait for the response); what matters is that the inputs are fixed.
        previous = chunk
    scene.close()
    del names
    return requests, chunks, attention


# --- run + serialise -----------------------------------------------------------------------------------
def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return {"__ndarray__": True, "dtype": str(value.dtype), "shape": list(value.shape),
                "data": [repr(float(x)) if value.dtype.kind == "f" else _jsonable(x.item()
                         if hasattr(x, "item") else x) for x in value.reshape(-1)]}
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return {"__float__": repr(value)}
    if hasattr(value, "value") and not isinstance(value, (str, bytes, int, bool)):
        return _jsonable(value.value)
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return value


def _strip_timing(response: dict) -> dict:
    out = {k: v for k, v in response.items() if k not in TIMING_KEYS}
    if isinstance(out.get("to"), dict):
        out["to"] = {k: v for k, v in out["to"].items() if k not in TO_TIMING_KEYS}
    if isinstance(out.get("field"), dict):
        out["field"] = {k: v for k, v in out["field"].items() if k not in FIELD_TIMING_KEYS}
    return out


def run(requests, chunks, attention) -> list[dict]:
    """Serve the requests; return one JSON-able record per request."""
    policy = StubPolicy(chunks, attention)
    served, _ = build_server(policy)
    out = []
    for req in requests:
        resp = served.infer(dict(req))
        result = served.refiner.last_result
        out.append({
            "response": _jsonable(_strip_timing(resp)),
            "exec_summary": _jsonable(served._exec_summary()),  # noqa: SLF001
            "metadata": _jsonable(served.metadata),
            "trajectory": _jsonable(None if result is None else np.asarray(result.trajectory)),
            "finger": _jsonable(served._last_finger),  # noqa: SLF001
            "layout": _jsonable({"gripper_columns": list(served.gripper_columns),
                                 "action_columns": served.layout.action_columns,
                                 "q_indices": served.layout.q_indices,
                                 "nq_model": served.layout.nq_model}),
        })
    return out


def _requests_to_arrays(requests) -> dict:
    arrays = {"n_requests": np.array(len(requests))}
    meta = []
    for i, req in enumerate(requests):
        keys = {}
        for k, v in req.items():
            if isinstance(v, np.ndarray):
                arrays[f"req{i}|{k}"] = v
                keys[k] = "ndarray"
            elif isinstance(v, dict):
                keys[k] = {"dict": _jsonable(v)}
                for kk, vv in v.items():
                    if isinstance(vv, np.ndarray):
                        arrays[f"req{i}|{k}|{kk}"] = vv
            else:
                keys[k] = {"value": v if not isinstance(v, float) else {"__float__": repr(v)}}
        meta.append(keys)
    arrays["requests_json"] = np.array(json.dumps(meta))
    return arrays


def _arrays_to_requests(z) -> list[dict]:
    meta = json.loads(str(z["requests_json"]))
    out = []
    for i, keys in enumerate(meta):
        req = {}
        for k, spec in keys.items():
            if spec == "ndarray":
                req[k] = np.asarray(z[f"req{i}|{k}"])
            elif "dict" in spec:
                d = {}
                for kk, vv in spec["dict"].items():
                    name = f"req{i}|{k}|{kk}"
                    d[kk] = np.asarray(z[name]) if name in z.files else _restore(vv)
                req[k] = d
            else:
                req[k] = _restore(spec["value"])
        out.append(req)
    return out


def _restore(v):
    if isinstance(v, dict) and "__float__" in v:
        return float(v["__float__"])
    if isinstance(v, list):
        return [_restore(x) for x in v]
    if isinstance(v, dict):
        return {k: _restore(x) for k, x in v.items()}
    return v


def load_fixture(path=FIXTURE):
    with np.load(path) as z:
        requests = _arrays_to_requests(z)
        chunks = [np.asarray(z[f"chunk{i}"]) for i in range(int(z["n_requests"]))]
        att_meta = json.loads(str(z["attention_json"]))
        attention = [{cam: np.asarray(z[f"att{i}|{cam}"]) for cam in att_meta[i]} for i in range(len(att_meta))]
        golden = json.loads(str(z["golden_json"]))
        provenance = json.loads(str(z["provenance_json"]))
    return requests, chunks, attention, golden, provenance


def write_fixture(path) -> dict:
    import subprocess

    path = pathlib.Path(path)
    if path.exists():
        raise FileExistsError(path)
    requests, chunks, attention = build_requests()
    golden = run(requests, chunks, attention)
    arrays = _requests_to_arrays(requests)
    for i, c in enumerate(chunks):
        arrays[f"chunk{i}"] = c
    for i, a in enumerate(attention):
        for cam, v in a.items():
            arrays[f"att{i}|{cam}"] = v
    arrays["attention_json"] = np.array(json.dumps([sorted(a) for a in attention]))
    arrays["golden_json"] = np.array(json.dumps(golden))
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "AG3S/trajopt", "AG3S/ag3s"],
                           capture_output=True, text=True).stdout.strip()
    provenance = {"head": head, "core_dirty": dirty, "argv": v11ab_cpu_argv(), "cpu_dropped_flags": list(CPU_DROPPED_FLAGS), "height": HEIGHT, "width": WIDTH,
                  "n_requests": N_REQUESTS, "chunk_h": CHUNK_H, "openings": list(OPENINGS)}
    arrays["provenance_json"] = np.array(json.dumps(provenance))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as stream:
        np.savez_compressed(stream, **arrays)
    return provenance


def _drop_wall_clock(record):
    """Remove the wall-clock keys from a stored record (a fixture written before they were excluded)."""
    for rec in record:
        field = rec.get("response", {}).get("field")
        if isinstance(field, dict):
            for k in FIELD_TIMING_KEYS:
                field.pop(k, None)
    return record


def diff(a, b, where="") -> list[str]:
    """Every place two JSON-able records differ (exact)."""
    out = []
    if type(a) is not type(b):
        return [f"{where}: type {type(a).__name__} != {type(b).__name__}"]
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{where}/{k}: only in {'golden' if k in a else 'current'}")
            else:
                out.extend(diff(a[k], b[k], f"{where}/{k}"))
    elif isinstance(a, list):
        if len(a) != len(b):
            return [f"{where}: length {len(a)} != {len(b)}"]
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(diff(x, y, f"{where}[{i}]"))
    elif a != b:
        out.append(f"{where}: {str(a)[:120]} != {str(b)[:120]}")
    return out


def check_fixture(path=FIXTURE) -> list[str]:
    requests, chunks, attention, golden, _ = load_fixture(path)
    current = json.loads(json.dumps(run(requests, chunks, attention)))
    return diff(_drop_wall_clock(golden), _drop_wall_clock(current))


# --- serve_safe level -----------------------------------------------------------------------------------
SERVE_SAFE_FIXTURE = ROOT / "tests/trajopt/data/sl5a_serve_safe_golden.json"
#: argparse keys SL5a adds (absent from the pre-SL5a namespace; compared separately for their defaults)
SL5A_NEW_ARG_KEYS = ("robot",)
#: Options added after the golden for benchmark-branch improvements (V11AB-pb*), with their off default. They are
#: left out of the comparison **only while off** in the V11AB / bare argv, so the golden keeps meaning "V11AB as
#: served"; an option that is on (or a changed default) still shows up as a difference.
PB_DEFAULT_OFF = {"reach_at_capture": False}


def _drop_default_off(d: dict, values: dict) -> dict:
    return {k: v for k, v in d.items() if not (k in PB_DEFAULT_OFF and values.get(k) == PB_DEFAULT_OFF[k])}


def serve_safe_record() -> dict:
    """V11AB (cuRobo, as served) and a bare argv through serve_safe's parser, TrajOpt config, build kwargs and
    the attention extractor -- the serve_safe-level pieces SL5a touches."""
    from AG3S.adapters.panda import profile as pf
    from AG3S.trajopt import serve_safe as ss

    out = {}
    bare = ["--config", "c", "--checkpoint", "k", "--model-xml", "m.xml"]
    for name, argv in (("v11ab", pf.v11ab_server_argv()), ("bare", bare)):
        ap = ss.build_parser()
        args = ap.parse_args(argv)
        ss.reject_bad_flag_combinations(ap, args)
        ss.resolve_subtask_gate(args)
        values = vars(args)
        out[name] = {"namespace": _jsonable(_drop_default_off(
                         {k: v for k, v in vars(args).items() if k not in SL5A_NEW_ARG_KEYS}, values)),
                     "trajopt_config": _jsonable(ss.trajopt_config_from_args(args).to_dict()),
                     "build_kwargs": _jsonable(_drop_default_off(dict(pf.v11ab_build_kwargs(args)), values)),
                     "build_call": _drop_default_off(pf.v11ab_build_call_source(), values)}
    rng = np.random.default_rng(0)
    raw = {k: rng.random((16, 16)).astype(np.float32)
           for k in ("cam_high", "cam_left_wrist", "cam_right_wrist", "zed_left", "other", "base_0_rgb")}
    extract = ss.attention_extractor()
    out["attention_extractor"] = {
        "full": _jsonable(extract({}, {"attention": raw})),
        "empty": _jsonable(extract({}, {"attention": {}})),
        "missing": _jsonable(extract({}, {})),
    }
    return out


def write_serve_safe_fixture(path=SERVE_SAFE_FIXTURE) -> None:
    path = pathlib.Path(path)
    with open(path, "x") as stream:
        json.dump(serve_safe_record(), stream, indent=1, sort_keys=True)


def check_serve_safe_fixture(path=SERVE_SAFE_FIXTURE) -> list[str]:
    golden = json.loads(pathlib.Path(path).read_text())
    current = json.loads(json.dumps(serve_safe_record(), sort_keys=True))
    return diff(golden, current)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", metavar="NPZ")
    g.add_argument("--check", metavar="NPZ")
    g.add_argument("--write-serve-safe", metavar="JSON")
    g.add_argument("--check-serve-safe", metavar="JSON")
    args = ap.parse_args(argv)
    if args.write_serve_safe:
        write_serve_safe_fixture(args.write_serve_safe)
        return 0
    if args.check_serve_safe:
        problems = check_serve_safe_fixture(args.check_serve_safe)
        print(json.dumps({"identical": not problems, "n_differences": len(problems), "first": problems[:20]}))
        return 1 if problems else 0
    if args.write:
        print(json.dumps(write_fixture(args.write)))
        return 0
    problems = check_fixture(args.check)
    print(json.dumps({"identical": not problems, "n_differences": len(problems), "first": problems[:20]}))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
