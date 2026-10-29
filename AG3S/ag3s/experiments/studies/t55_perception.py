"""T55 — the serving AG3S grounding re-run on re-simulated observations, feeding the strong rule (CPU; GPU backend optional).

`AG3S/ag3s/docs/handoff/T55.task.md` 추가 사전 고정 2 (user decision 2026-10-13): perception = the **fixed
serving L8H2 grounding re-run on the re-sim's own image and depth**, perception history updated from those
observations, for every condition (the unintervened one included). This is the ``reground`` latch mode — the
primary mode of `t55_study`. The re-sim npz's ``record`` mode (the server's recorded perception, stale after an
intervention) is the sensitivity mode and ``oracle`` (MuJoCo reach) the auxiliary diagnostic.

What runs (serving code, unchanged)
-----------------------------------
* **The request** is packed the way the T53 client packs it (`AG3S.trajopt.client.SafeRemoteClient._pack`):
  at every policy frame of the re-simulation (`t55_resim.simulate(observer=...)`, i.e. after the client's
  `build_obs` and before any ``mj_step``) the three AG3S cameras (`wire.DEFAULT_CAMERAS`) are captured with
  `TransportScene.attach(m, d).capture` (480x640 depth, intrinsics, base-frame extrinsics, robot state), every
  camera stamped with the one frozen sim instant (``stamp_mode = sim_frozen``; sim time instead of
  ``time.monotonic()``), ``phase = approach`` and ``active_manipulators = (used arm,)`` (the client's
  ``--safe-phase`` / ``--safe-manipulators``), ``reset`` on the first request, ``seq = k + 1``, and the execution
  feedback of the previous chunk as `wire.make_exec_feedback` builds it (`ExecutionLog.close`): the record's
  ``executed`` flags, the re-simulated ``applied_gripper`` / ``applied_arm`` (``d.ctrl``) and the measured
  opening at this capture; ``executed_chunk`` / ``hold_kind`` / ``hold_reason`` / ``planned_gripper`` from the
  T53 record's planning row (dev records have none: ``refined`` / None / the executed row's gripper).
* **The server** is a `SafePolicy` built exactly as `serve_safe.main` builds it from the T53 rollout server's
  own command line (`server_proc.txt`; `t54_replay.c1a_build_kwargs` -> `serve_safe.build_ag3s`,
  `serve_safe.trajopt_config_from_args`, `serve_safe.load_static_geometry`, `serve_safe.attention_extractor`),
  one per (re-sim, prompt), and called with `SafePolicy.infer(request)` request by request. Its wrapped policy
  is a stub that answers with the reference chunk of that request (T53: the server chunk's
  ``reference_chunk``; dev: the step record's chunk) and the **re-extracted attention** of that frame
  (`t55_attention`: serving `AttentionPolicy`, L8H2 / denoise 0 / last, + the ``kv_L4_v1`` subtask label of the
  same pass). The only part replaced is the trajectory optimiser: `PerceptionOnlyRefiner.refine` calls the
  policy's own `_scene_fn` (finger state -> `_advance_grasp` (feedback -> latch -> attach / detach) -> AG3S
  flags -> `process_multi_debug` -> `_run_latch` -> grasp record) exactly once, as the real refiner does
  first, and returns the reference chunk (no TO; the latch, AG3S state and grasp record do not depend on it).
* ESDF backend: ``legacy`` (default, CPU — the T54 ``c1a`` profile: grounding runs before the field; the two
  cuRobo-only field settings are dropped and recorded) or ``server`` (the server's own ``curobo`` backend;
  needs `.venv-openpi-live`, the GPU and the shared lock).

Attention sources (``--attention-source``)
------------------------------------------
``t55``     primary — `t55_attention` output for every (re-sim, prompt), instruction prompts included (the
            changed prompt's own attention and subtask label; `T55.task.md` 추가 사전 고정 2).
``server``  diagnostic, test ``E0`` / ``orig`` only — the attention the serving AG3S actually received
            (server chunk ``attention_*``, float16 as recorded) and the recorded subtask probabilities: isolates
            the replay machinery from attention-extraction parity.
``t53``     diagnostic, test ``E0`` / ``orig`` only — T53's producer attention (``attention_16d_<key>.npz``,
            denoise 0 / last / L8H2, float16) with the recorded subtask probabilities (the T54 input).

Checks
------
* pairing: every frame's observation images must equal the stored re-sim npz images (and qpos) — the attention
  was extracted from those images; a mismatch stops the job.
* serving parity (test ``E0`` / ``orig``, server chunks present): offline grasp record vs the server's recorded
  ``summary_json.grasp`` — latch state per request, attach / detach / rejection events, manipulated id, reach,
  target centroid. Numbers only; the verdict is the verifier's.

Outputs: ``$AG3S_RUN_DIR/verify/T55/perception/<set>/<source>-<backend>/<key>__<tag>__<prompt_tag>.npz`` (+
``.json``): per request ``latch_reground__phase`` / ``__attach`` / ``__detach`` (the `t55_study` latch trace),
``phase_before``, ``revoked``, ``slipped``, ``rejected``, ``reach_mm``, ``manipulated_id``, ``has_target``,
``target_centroid``, ``scene_fn_failure`` and the full grasp record per request (``grasp_json``).

Usage::

    source scripts/setup/env.sh; export AG3S_RUN_DIR=<the T55 run>; R=<T53 run>
    M=AG3S.ag3s.experiments.studies.t55_perception
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 .venv-ag3s/bin/python -m $M --set dev --t53-run-dir $R --workers 2
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 .venv-ag3s/bin/python -m $M --set test --t53-run-dir $R \\
        --episodes-root $R/verify/T53/runs/C1a/R --server-roots $S --workers 2
    # diagnostics (test E0 / orig only, CPU): --attention-source server   |   --attention-source t53
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import pathlib
import sys
import time
from typing import Optional, Sequence

import numpy as np

STEP = "T55"
PERC_VERSION = "t55-perception-v1"
ATTN_SOURCES = ("t55", "server", "t53")
BACKENDS = ("legacy", "server")
#: The client's ``--safe-phase`` of every T53 rollout (`meta.json` argv) — used for the dev records too.
CLIENT_PHASE = "approach"
CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
#: Server chunk attention keys -> policy camera (`ConstraintRecordWriter`: AG3S CameraID names).
SERVER_ATTN_KEYS = {"cam_high": "attention_head", "cam_left_wrist": "attention_left_wrist",
                    "cam_right_wrist": "attention_right_wrist"}
#: Server argv entries that differ between the two T53 servers and do not reach `SafePolicy`.
SERVER_ARGV_IGNORED = ("--port", "--run-dir", "--record-constraints")
GPU_LOCK = "/tmp/ag3s-vla-gpu-validation.lock"


# ------------------------------------------------------------------------------ the server


def server_procs(t53_run_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted((pathlib.Path(t53_run_dir) / "verify" / "T53" / "runs" / "C1a").glob("server_*/server_proc.txt"))


def _strip_argv(argv: Sequence[str]) -> list[str]:
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in SERVER_ARGV_IGNORED:
            skip = True
            continue
        out.append(a)
    return out


def serving_argv(t53_run_dir: pathlib.Path) -> tuple[list[str], dict]:
    """The T53 rollout server's `serve_safe` argv; refuses when the servers' SafePolicy arguments differ."""
    from AG3S.ag3s.experiments.studies import t54_replay as rp

    procs = server_procs(t53_run_dir)
    if not procs:
        raise SystemExit(f"no server_*/server_proc.txt under {t53_run_dir}/verify/T53/runs/C1a")
    argvs = [rp.server_argv(p) for p in procs]
    stripped = {tuple(_strip_argv(a)) for a in argvs}
    if len(stripped) != 1:
        raise SystemExit(f"the T53 servers ran different SafePolicy arguments: {sorted(stripped)}")
    return argvs[0], {"server_procs": [str(p) for p in procs], "argv": argvs[0],
                      "ignored_differences": list(SERVER_ARGV_IGNORED)}


class StubPolicy:
    """The served policy's place in `SafePolicy`: answers with what `set_result` gave for this request."""

    def __init__(self):
        self._result: Optional[dict] = None
        self.metadata: dict = {}

    def set_result(self, result: dict) -> None:
        self._result = dict(result)

    def infer(self, obs, **kwargs) -> dict:
        if self._result is None:
            raise RuntimeError("StubPolicy.infer without set_result")
        out, self._result = self._result, None
        return out

    def reset(self) -> None:
        """`SafePolicy.reset` (a ``reset`` request) runs *before* this request's ``infer`` — keep its result."""


class PerceptionOnlyRefiner:
    """`TrajOptChunkRefiner` stand-in: calls ``scene_fn`` once (as the real `refine` does first) and returns the
    chunk. A ``scene_fn`` exception is recorded (``last_failure``) like the real refiner's fallback."""

    def __init__(self, scene_fn):
        self.scene_fn = scene_fn
        self.last_result = None
        self.last_failure: Optional[str] = None
        self.last_continuity: dict = {}
        self.optimizer = None
        self.failures = 0

    def refine(self, physical_chunk, context: Optional[dict] = None):
        self.last_failure = None
        self.last_continuity = {"used": False, "aligned_from_step": None,
                                "source": "perception-only replay (no trajectory optimisation)"}
        try:
            self.scene_fn(context)
        except Exception as exc:  # noqa: BLE001 — the real refiner swallows it too; recorded per request
            self.last_failure = f"{type(exc).__name__}: {exc}"
            self.failures += 1
        return np.asarray(physical_chunk, np.float64)

    def reset(self) -> None:
        self.last_result = None
        self.last_failure = None


def build_safe_policy(argv: Sequence[str], *, backend: str = "legacy"):
    """``(SafePolicy, stub, provenance)`` — `serve_safe.main`'s construction from the server argv, with the
    ESDF backend `backend` (``legacy`` or ``server`` = the argv's own) and the TO replaced (module docstring)."""
    from AG3S.ag3s.experiments.studies import t54_replay as rp
    from AG3S.trajopt import serve_safe as ss
    from AG3S.trajopt.safe_policy import SafePolicy

    if backend not in BACKENDS:
        raise ValueError(f"backend {backend!r} not in {BACKENDS}")
    ap = ss.build_parser()
    args = ap.parse_args(list(argv))
    ss.reject_bad_flag_combinations(ap, args)
    ss.resolve_subtask_gate(args)
    be = "legacy" if backend == "legacy" else str(args.esdf_backend)
    xml, kw, rec = rp.c1a_build_kwargs(list(argv), backend=be)
    if rec["placed_no_seed_certify"]:
        raise SystemExit("T55 fixes clustering.placed_no_seed_certify = off; the server had it on")
    stub = StubPolicy()
    ag = ss.build_ag3s(xml, **kw)
    sp = SafePolicy(stub, ag3s=ag,
                    static_geometry=ss.load_static_geometry(args.static_geometry, args.model_xml, links=args.links),
                    to_config=ss.trajopt_config_from_args(args), attention_fn=ss.attention_extractor(),
                    recorder=None, shadow=args.shadow)
    sp.refiner = PerceptionOnlyRefiner(sp._scene_fn)  # noqa: SLF001 — the TO is the one part not replayed
    from AG3S.ag3s.experiments.studies import t55_resim as rs

    prov = {"backend": be, "model_xml": xml, "latch_config": repr(sp._latch.config),  # noqa: SLF001
            "t54_replay_md5": rs.md5_of(pathlib.Path(rp.__file__)),
            "build_record": rec, "subtask_gate": bool(args.subtask_gate),
            "build_kwargs": {k: (v if isinstance(v, (int, float, str, bool, type(None), dict, list)) else repr(v))
                             for k, v in kw.items()}}
    return sp, stub, prov


# ------------------------------------------------------------------------------ the request


def wire_feedback(k: int, chunk_steps: dict, measured_gripper: Sequence[float], *, horizon: int,
                  record_fb: Optional[dict], executed_rows: Optional[np.ndarray]) -> dict:
    """The client's `ExecutionLog.close` for request k (k = 0: no chunk has run yet)."""
    from AG3S.trajopt import wire

    if k == 0 or not chunk_steps.get("executed"):
        return wire.no_exec_feedback("no chunk has run yet in this episode" if k == 0 else
                                     f"chunk seq {k} was opened but ran no control step")
    ex = [bool(x) for x in chunk_steps["executed"]]
    n = len(ex)
    rf = record_fb if (record_fb and record_fb.get("available") and
                       [bool(x) for x in (record_fb.get("executed") or [])] == ex) else None
    if rf is not None and rf.get("planned_gripper") is not None:
        planned = np.asarray(rf["planned_gripper"], np.float64)
    elif executed_rows is not None:
        planned = np.asarray(executed_rows, np.float64)[:, [7, 15]]
    else:
        planned = np.asarray(chunk_steps["applied_gripper"], np.float64)
    return wire.make_exec_feedback(
        seq=int(k), t_step_start=int((k - 1) * horizon), executed=ex,
        executed_chunk=str((rf or {}).get("executed_chunk") or "refined"), ipc=str((rf or {}).get("ipc") or "ok"),
        hold_kind=(rf or {}).get("hold_kind"), hold_reason=(rf or {}).get("hold_reason"),
        planned_gripper=planned.reshape(n, 2),
        applied_gripper=np.asarray(chunk_steps["applied_gripper"], np.float64).reshape(n, 2),
        applied_arm=np.asarray(chunk_steps["applied_arm"], np.float64).reshape(n, -1),
        measured_gripper=np.asarray(measured_gripper, np.float64))


def pack_request(scene, obs: dict, *, k: int, used_arm: str, sim_time: float, feedback: dict,
                 prompt: Optional[str] = None) -> dict:
    """`SafeRemoteClient._pack` on the re-sim's ``m`` / ``d`` (`TransportScene.attach`), one frozen stamp."""
    from AG3S.trajopt import wire

    cams = tuple(wire.DEFAULT_CAMERAS)
    depth, K, T, state = {}, {}, {}, {}
    for cam in cams:
        fr = scene.capture(cam)
        depth[cam] = np.asarray(fr.depth, np.float64)
        K[cam] = np.asarray(fr.camera_intrinsics, np.float64)
        T[cam] = np.asarray(fr.T_base_cam, np.float64)
        state[cam] = np.asarray(fr.robot_state, np.float64)
    stamps = {cam: float(sim_time) for cam in cams}
    o = dict(obs)
    if prompt is not None:
        o["prompt"] = prompt
    return wire.pack_request(o, cameras=cams, depth=depth, intrinsics=K, extrinsics=T, robot_state=state,
                             stamps=stamps, phase=CLIENT_PHASE, active_manipulators=[used_arm], reset=(k == 0),
                             seq=int(k) + 1, exec_feedback=feedback, render_stamps=dict(stamps),
                             stamp_mode="sim_frozen")


# ------------------------------------------------------------------------------ per-request record


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def request_row(sp) -> dict:
    """What the strong rule and the serving parity read after one `SafePolicy.infer`."""
    from AG3S.trajopt.safe_policy import _jsonable

    rec = dict(sp._last_grasp_record or {})  # noqa: SLF001
    ev = rec.get("evidence") or {}
    cs = sp._last_constraint_set  # noqa: SLF001
    tgt = getattr(cs, "target", None)
    manip = getattr(sp.ag3s, "manipulated", None)
    return {"phase": str(rec.get("state", sp._latch.phase.value)),  # noqa: SLF001
            "phase_before": str(rec.get("state_before", "")),
            "attach": bool(rec.get("attach", False)), "detach": bool(rec.get("detach", False)),
            "rejected": rec.get("rejected") is not None, "revoked": rec.get("attach_revoked") is not None,
            "slipped": bool(rec.get("held_slip", False)),
            "reach_mm": _f(ev.get("reach_mm")),
            "manipulated_id": -1 if ev.get("manipulated_id") is None else int(ev["manipulated_id"]),
            "manipulated_state": str(ev.get("manipulated_state") or ""),
            "has_target": bool(getattr(cs, "has_target", False)),
            "target_centroid": (np.asarray(tgt.centroid, np.float64) if tgt is not None
                                else np.full(3, np.nan)),
            "manipulated_now": -1 if manip is None else int(manip.id),
            "scene_fn_failure": str(getattr(sp.refiner, "last_failure", None) or ""),
            "grasp_json": json.dumps(_jsonable(rec), ensure_ascii=False, sort_keys=True)}


ROW_KEYS = ("phase", "phase_before", "attach", "detach", "rejected", "revoked", "slipped", "reach_mm",
            "manipulated_id", "manipulated_state", "has_target", "target_centroid", "manipulated_now",
            "scene_fn_failure", "grasp_json")


def rows_to_arrays(rows: Sequence[dict]) -> dict[str, np.ndarray]:
    out = {}
    for k in ROW_KEYS:
        v = [r[k] for r in rows]
        if k in ("phase", "phase_before"):
            out[k] = np.asarray(v, dtype="U10")
        elif k in ("manipulated_state", "scene_fn_failure", "grasp_json"):
            out[k] = np.asarray(v, dtype=str)
        else:
            out[k] = np.asarray(v)
    return out


def latch_arrays(arr: dict) -> dict[str, np.ndarray]:
    """The ``latch_reground__*`` keys `t55_study` reads (same names as the re-sim npz's latch modes)."""
    return {f"latch_reground__{k}": arr[k] for k in ("phase", "phase_before", "attach", "detach", "revoked", "slipped")}


# ------------------------------------------------------------------------------ serving parity


def serving_parity(arr: dict, grasp: Sequence[dict], centroids: Optional[np.ndarray] = None) -> dict:
    """Offline grasp record vs the server's recorded ``summary_json.grasp`` (numbers only)."""
    n = min(len(arr["phase"]), len(grasp))
    rec_state = [str((g or {}).get("state")) for g in grasp[:n]]
    got = [str(x) for x in arr["phase"][:n]]
    same = [a == b for a, b in zip(got, rec_state)]

    def events(key):
        return [i for i in range(n) if bool((grasp[i] or {}).get(key))]

    rec_rej = [i for i in range(n) if (grasp[i] or {}).get("rejected") is not None]
    rec_rev = [i for i in range(n) if (grasp[i] or {}).get("attach_revoked") is not None]
    rec_reach = np.asarray([_f(((grasp[i] or {}).get("evidence") or {}).get("reach_mm")) for i in range(n)])
    rec_mid = [((grasp[i] or {}).get("evidence") or {}).get("manipulated_id") for i in range(n)]
    reach = np.asarray(arr["reach_mm"][:n], np.float64)
    both = np.isfinite(reach) & np.isfinite(rec_reach)
    d_reach = np.abs(reach - rec_reach)[both]
    mid_pairs = [(int(arr["manipulated_id"][i]), -1 if rec_mid[i] is None else int(rec_mid[i])) for i in range(n)]
    held_o = np.asarray([x == "held" for x in got])
    held_r = np.asarray([x == "held" for x in rec_state])
    states = sorted(set(got) | set(rec_state))
    out = {"n_requests": n, "state_equal": int(sum(same)), "state_equal_share": (sum(same) / n) if n else None,
           "first_state_mismatch": next((i for i, e in enumerate(same) if not e), None),
           "held_frames_offline": int(held_o.sum()), "held_frames_recorded": int(held_r.sum()),
           "held_equal_frames": int((held_o == held_r).sum()),
           "state_confusion": {f"{a}->{b}": int(sum(1 for x, y in zip(rec_state, got) if x == a and y == b))
                               for a in states for b in states
                               if any(x == a and y == b for x, y in zip(rec_state, got))},
           "attach": {"offline": [int(i) for i in np.flatnonzero(arr["attach"][:n])], "recorded": events("attach")},
           "detach": {"offline": [int(i) for i in np.flatnonzero(arr["detach"][:n])], "recorded": events("detach")},
           "rejected": {"offline": [int(i) for i in np.flatnonzero(arr["rejected"][:n])], "recorded": rec_rej},
           "revoked": {"offline": [int(i) for i in np.flatnonzero(arr["revoked"][:n])], "recorded": rec_rev},
           "reach_mm_abs_diff": ({"n": int(both.sum()), "median": float(np.median(d_reach)),
                                  "max": float(d_reach.max())} if both.any() else {"n": 0}),
           "reach_presence_equal": int((np.isfinite(reach) == np.isfinite(rec_reach)).sum()),
           "manipulated_id_equal": int(sum(a == b for a, b in mid_pairs)),
           "states": {"offline": got, "recorded": rec_state}}
    if centroids is not None:
        c = np.asarray(centroids, np.float64)[:n]
        tc = np.asarray(arr["target_centroid"][:n], np.float64)
        ok = np.isfinite(c).all(1) & np.isfinite(tc).all(1)
        d = np.linalg.norm(c - tc, axis=1)[ok]
        out["target_centroid_dist_m"] = ({"n": int(ok.sum()), "median": float(np.median(d)), "max": float(d.max())}
                                         if ok.any() else {"n": 0})
        out["has_target_equal"] = int((np.isfinite(c).all(1) == np.asarray(arr["has_target"][:n], bool)).sum())
    return out


# ------------------------------------------------------------------------------ attention inputs


def server_chunk_inputs(files: Sequence[pathlib.Path]) -> dict:
    """Per request from the server chunks: reference chunk, recorded attention (policy cams), subtask, centroid."""
    ref, att, sub, cen = [], [], [], []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            ref.append(np.asarray(z["reference_chunk"], np.float64))
            att.append(np.stack([np.asarray(z[SERVER_ATTN_KEYS[c]], np.float32) if SERVER_ATTN_KEYS[c] in z.files
                                 else np.full((16, 16), np.nan, np.float32) for c in CAMS]))
            s = json.loads(str(z["summary_json"]))
            p = (s.get("subtask") or {}).get("p")
            sub.append(None if not p else {"p": {str(k): float(v) for k, v in p.items()},
                                           "argmax": max(p, key=p.get), "probe": "recorded"})
            cen.append(np.asarray(z["target_centroid"], np.float64) if "target_centroid" in z.files
                       and z["target_centroid"].shape == (3,) and s.get("has_target") else np.full(3, np.nan))
    return {"reference_chunk": ref, "attention": np.asarray(att), "subtask": sub, "target_centroid": np.asarray(cen)}


def dev_reference_chunks(src) -> list[np.ndarray]:
    out = []
    for f in src.step_files:
        with np.load(f, allow_pickle=False) as z:
            out.append(np.asarray(z["actions"], np.float64))
    return out


def t55_attention_inputs(path: pathlib.Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        att = np.asarray(z["attention"], np.float32)
        sub = [json.loads(str(x)) if str(x) else None for x in z["subtask_json"]]
        stamp = json.loads(str(z["stamp"]))
    return {"attention": att, "subtask": sub, "stamp": stamp}


def t53_attention_inputs(t53_run_dir: pathlib.Path, key: str) -> np.ndarray:
    """``(F, 3, 16, 16)`` float32 of T53's producer npz at denoise 0 / last / L8 / H2."""
    from AG3S.ag3s.experiments.studies import t53_records as rc
    from AG3S.trajopt import attention_policy as apol

    rc.configure(run_dir=t53_run_dir)
    with np.load(rc.npz_path(key), allow_pickle=False) as z:
        ds = [int(x) for x in z["denoise_steps"]]
        ag = [str(x) for x in z["aggregations"]]
        cams = [str(x) for x in z["cameras"]]
        a = z["attention"][:, ds.index(apol._DENOISE_STEP), ag.index(apol._AGG), apol._LAYER, apol._HEAD]  # noqa: SLF001
    return np.asarray(a, np.float32)[:, [cams.index(c) for c in CAMS]]


# ------------------------------------------------------------------------------ jobs


def perception_dir(set_name: str, source: str, backend: str, *, forbid=()) -> pathlib.Path:
    from AG3S.ag3s.experiments.studies import t55_resim as rs

    return rs.out_root(f"perception/{set_name}/{source}-{backend}", forbid=forbid)


def attention_dir(set_name: str, *, forbid=()) -> pathlib.Path:
    from AG3S.ag3s.experiments.studies import t55_resim as rs

    return rs.out_root(f"attention/{set_name}", forbid=forbid)


def plan_jobs(resim_dir: pathlib.Path, *, set_name: str, source: str, attn_dir: Optional[pathlib.Path],
              keys: Optional[Sequence[str]] = None, tags: Optional[Sequence[str]] = None,
              instruction: bool = True, meta_ds: Optional[dict] = None) -> list[dict]:
    """One job per re-sim (key, tag); its prompts are E0's orig + instruction prompts, else orig."""
    from AG3S.ag3s.experiments.studies import t55_features as tf

    if source in ("server", "t53") and set_name != "test":
        raise SystemExit(f"--attention-source {source} is a test E0 / orig diagnostic")
    groups: dict[tuple, dict] = {}
    for j in tf.resim_jobs(resim_dir, attn_dir or resim_dir, set_name=set_name, keys=keys, tags=tags,
                           instruction=instruction and source == "t55", meta_ds=meta_ds):
        if source != "t55" and not (j["tag"] == "E0" and j["prompt_tag"] == "orig"):
            continue
        g = groups.setdefault((j["key"], j["tag"]), {"key": j["key"], "tag": j["tag"], "src": j["src"],
                                                     "set": set_name, "prompts": []})
        g["prompts"].append({"prompt_tag": j["prompt_tag"], "prompt": j["prompt"], "noun": j["noun"],
                             "attention": j["out"] if source == "t55" else None})
    return [groups[k] for k in sorted(groups)]


def _source_for(key: str, set_name: str, args_ns: dict):
    from AG3S.ag3s.experiments.studies import t55_resim as rs

    if set_name == "dev":
        return rs.dev_source(key)
    return rs.t53_source(key, t53_run_dir=pathlib.Path(args_ns["t53_run_dir"]),
                         episodes_root=(None if args_ns.get("episodes_root") is None
                                        else pathlib.Path(args_ns["episodes_root"])),
                         server_roots=[pathlib.Path(p) for p in args_ns.get("server_roots") or []])


def _variant(resim_npz: pathlib.Path):
    """(Intervention or None, frame or None, meta) of a stored re-sim run."""
    from AG3S.ag3s.experiments.studies import t55_resim as rs

    with np.load(resim_npz, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
    ivp = meta.get("intervention")
    if not ivp:
        return None, None, meta
    p = {k: v for k, v in ivp.items() if k != "tag"}
    p["park_pos"] = tuple(p.get("park_pos", rs.PARK_POS))
    iv = rs.Intervention(**p)
    if iv.tag != ivp.get("tag"):
        raise SystemExit(f"{resim_npz}: intervention tag {ivp.get('tag')} != {iv.tag}")
    return iv, int(meta["info"]["frame"]), meta


def run_job(job: dict, *, source: str, backend: str, out_dir: pathlib.Path, client_path: pathlib.Path,
            serving: list, args_ns: dict, force: bool, run_meta: dict) -> dict:
    """One re-simulation of (key, tag) with one offline `SafePolicy` per prompt."""
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
    from AG3S.ag3s.experiments.studies import t55_resim as rs

    todo = [p for p in job["prompts"] if force or not (out_dir / f"{job['key']}__{job['tag']}__{p['prompt_tag']}.npz").is_file()]
    if not todo:
        return {"key": job["key"], "tag": job["tag"], "cached": True}
    src = _source_for(job["key"], job["set"], args_ns)
    resim_npz = pathlib.Path(job["src"])
    iv, frame, rmeta = _variant(resim_npz)
    with np.load(resim_npz, allow_pickle=False) as z:
        ref_imgs = {c: np.asarray(z[f"image_{c}"]) for c in CAMS}
        ref_qpos = np.asarray(z["qpos"], np.float64)
    if ref_imgs["cam_high"].ndim != 4:
        raise SystemExit(f"{resim_npz} has no images (re-simulated with --no-images)")
    F = src.n_frames
    # inputs per prompt
    srv = server_chunk_inputs(src.ref_chunk_files) if src.ref_chunk_files else None
    chunks = srv["reference_chunk"] if srv is not None else dev_reference_chunks(src)
    inputs = []
    for p in todo:
        if source == "t55":
            if not pathlib.Path(p["attention"]).is_file():
                raise SystemExit(f"attention missing: {p['attention']} (t55_attention extract first)")
            a = t55_attention_inputs(pathlib.Path(p["attention"]))
            if a["stamp"].get("source_md5") != rs.md5_of(resim_npz) or a["stamp"].get("prompt") != p["prompt"]:
                raise SystemExit(f"{p['attention']} was extracted from another re-sim / prompt")
            att, sub = a["attention"], a["subtask"]
        elif source == "server":
            if srv is None:
                raise SystemExit(f"[{src.key}] no server chunks — --attention-source server unavailable")
            att, sub = srv["attention"], srv["subtask"]
        else:
            att, sub = t53_attention_inputs(pathlib.Path(args_ns["t53_run_dir"]), src.key), \
                (srv["subtask"] if srv is not None else [None] * F)
        if len(att) != F or len(sub) != F or len(chunks) != F:
            raise SystemExit(f"[{src.key}] {len(att)} attention / {len(sub)} subtask / {len(chunks)} chunks for {F} frames")
        inputs.append((p, att, sub))
    # one SafePolicy per prompt
    logging.disable(logging.WARNING)
    built, prov = [], None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        for _ in inputs:
            sp, stub, prov = build_safe_policy(serving, backend=backend)
            built.append((sp, stub))
    rows: list[list[dict]] = [[] for _ in inputs]
    pairing = {"image_mismatch_frames": [], "qpos_mismatch_frames": []}
    state = {"scene": None}
    rec_fb = src.ref_feedback

    def observer(k: int, ctx: dict) -> None:
        obs = ctx["obs"]
        for c in CAMS:
            if not np.array_equal(rs._hwc(obs["images"][c]), ref_imgs[c][k]):  # noqa: SLF001
                pairing["image_mismatch_frames"].append(int(k))
                break
        if not np.array_equal(np.asarray(ctx["d"].qpos, np.float64), ref_qpos[k]):
            pairing["qpos_mismatch_frames"].append(int(k))
        if pairing["image_mismatch_frames"] or pairing["qpos_mismatch_frames"]:
            raise RuntimeError(f"[{src.key} {job['tag']}] frame {k}: the re-simulation does not pair with {resim_npz}")
        if state["scene"] is None:
            state["scene"] = TransportScene.attach(ctx["m"], ctx["d"])
        st = np.asarray(obs["state"], np.float64)
        steps = ctx["chunk_steps"]
        fb = wire_feedback(k, steps, (st[7], st[15]), horizon=8,
                           record_fb=None if rec_fb is None else rec_fb[k],
                           executed_rows=(src.actions[steps["t"]] if steps.get("t") else None))
        for (p, att, sub), (sp, stub), out in zip(inputs, built, rows):
            req = pack_request(state["scene"], obs, k=k, used_arm=src.used_arm, sim_time=ctx["sim_time"],
                               feedback=fb, prompt=p["prompt"])
            result = {"actions": np.asarray(chunks[k], np.float64),
                      "attention": {c: np.asarray(att[k, ci], np.float32) for ci, c in enumerate(CAMS)}}
            if sub[k] is not None:
                result["subtask"] = sub[k]
            stub.set_result(result)
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                sp.infer(req)
            out.append(request_row(sp))
        if buf.tell() > 2_000_000:
            buf.seek(0)
            buf.truncate()

    cl = rs.load_client(client_path)
    t0 = time.time()
    try:
        rs.simulate(cl, src, intervention=iv, frame=frame, keep_images=False, observer=observer)
    finally:
        if state["scene"] is not None:
            state["scene"].close()
        logging.disable(logging.NOTSET)
    seconds = time.time() - t0
    results = {}
    for (p, att, sub), r in zip(inputs, rows):
        arr = rows_to_arrays(r)
        meta = {**run_meta, "version": PERC_VERSION, "created": rs.now_kst(), "key": src.key, "tag": job["tag"],
                "prompt_tag": p["prompt_tag"], "prompt": p["prompt"], "noun": p["noun"], "attention_source": source,
                "attention_file": p["attention"], "backend": backend, "server": prov, "resim": str(resim_npz),
                "resim_md5": rs.md5_of(resim_npz), "resim_baseline": rmeta.get("baseline"),
                "intervention": None if iv is None else iv.params(), "frame": frame, "pairing": pairing,
                "seconds_all_prompts": seconds, "n_requests": len(r),
                "scene_fn_failures": int(sum(1 for x in r if x["scene_fn_failure"])),
                "phase_counts": {s: int(sum(1 for x in r if x["phase"] == s)) for s in sorted({x["phase"] for x in r})}}
        if job["tag"] == "E0" and p["prompt_tag"] == "orig" and src.ref_grasp is not None:
            meta["serving_parity"] = serving_parity(arr, src.ref_grasp,
                                                    None if srv is None else srv["target_centroid"])
        path = out_dir / f"{src.key}__{job['tag']}__{p['prompt_tag']}.npz"
        rs.save_npz_atomic(path, **arr, **latch_arrays(arr), meta=np.asarray(json.dumps(rs.jsonable(meta), sort_keys=True)),
                           key=np.asarray(src.key), tag=np.asarray(job["tag"]), prompt_tag=np.asarray(p["prompt_tag"]),
                           prompt=np.asarray(p["prompt"]), noun=np.asarray(p["noun"]))
        rs.write_json(path.with_suffix(".json"), meta)
        sp_ = meta.get("serving_parity") or {}
        results[p["prompt_tag"]] = {"phase_counts": meta["phase_counts"],
                                    "scene_fn_failures": meta["scene_fn_failures"],
                                    **({"state_equal": f"{sp_['state_equal']}/{sp_['n_requests']}",
                                        "attach": sp_["attach"], "detach": sp_["detach"]} if sp_ else {})}
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs" / f"{src.key}__{job['tag']}.log.txt").write_text(buf.getvalue()[-200_000:])
    return {"key": src.key, "tag": job["tag"], "seconds": seconds, "prompts": results}


def _worker(job: dict) -> dict:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    try:
        return run_job(job["job"], **job["kw"])
    except Exception as exc:  # noqa: BLE001 — one job's failure is reported, others continue
        import traceback

        return {"key": job["job"]["key"], "tag": job["job"]["tag"], "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-3000:]}


# ------------------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", required=True, choices=("dev", "test"))
    ap.add_argument("--t53-run-dir", required=True, help="the T53 run (read-only): server argv, test records, client")
    ap.add_argument("--episodes-root", default=None)
    ap.add_argument("--server-roots", nargs="*", default=[])
    ap.add_argument("--client", default=None)
    ap.add_argument("--records", nargs="+", default=None)
    ap.add_argument("--tags", nargs="+", default=None)
    ap.add_argument("--no-instruction", action="store_true")
    ap.add_argument("--attention-source", default="t55", choices=ATTN_SOURCES)
    ap.add_argument("--esdf-backend", default="legacy", choices=BACKENDS,
                    help="legacy (CPU, default) or server (the server's curobo: GPU, lock, .venv-openpi-live)")
    ap.add_argument("--gpu-lock", default=GPU_LOCK)
    ap.add_argument("--workers", type=int, default=1, help="processes (<= 4; each holds one AG3S per prompt)")
    ap.add_argument("--force", action="store_true")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> dict:
    from AG3S.ag3s.experiments.studies import t55_resim as rs

    ap = build_parser()
    args = ap.parse_args(argv)
    if not 1 <= args.workers <= 4:
        ap.error("--workers in 1..4")
    if args.esdf_backend == "server":
        from AG3S.ag3s.experiments.studies.t53_extract import lock_is_held

        if not lock_is_held(args.gpu_lock):
            raise SystemExit(f"--esdf-backend server uses the GPU: run under flock {args.gpu_lock}")
    forbid = [pathlib.Path(args.t53_run_dir)]
    resim_dir = rs.out_root(f"resim/{args.set}", forbid=forbid)
    out_dir = perception_dir(args.set, args.attention_source, args.esdf_backend, forbid=forbid)
    out_dir.mkdir(parents=True, exist_ok=True)
    adir = attention_dir(args.set, forbid=forbid) if args.attention_source == "t55" else None
    jobs = plan_jobs(resim_dir, set_name=args.set, source=args.attention_source, attn_dir=adir, keys=args.records,
                     tags=args.tags, instruction=not args.no_instruction)
    serving, sprov = serving_argv(pathlib.Path(args.t53_run_dir))
    client = pathlib.Path(args.client) if args.client else rs.default_client(pathlib.Path(args.t53_run_dir))
    run_meta = {"step": STEP, "argv": sys.argv, "args": vars(args), "python": sys.executable,
                "serving": sprov, "client": str(client), "client_md5": rs.md5_of(client)}
    args_ns = {"t53_run_dir": args.t53_run_dir, "episodes_root": args.episodes_root,
               "server_roots": args.server_roots}
    kw = {"source": args.attention_source, "backend": args.esdf_backend, "out_dir": out_dir, "client_path": client,
          "serving": serving, "args_ns": args_ns, "force": args.force, "run_meta": run_meta}
    t0 = time.time()
    payload = [{"job": j, "kw": kw} for j in jobs]
    if args.workers == 1:
        rows = [_worker(p) for p in payload]
    else:
        import concurrent.futures as cf
        import multiprocessing as mp

        with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
            rows = list(ex.map(_worker, payload))
    for r in rows:
        print(f"[perception {args.set} {args.attention_source}-{args.esdf_backend}] {r['key']} {r['tag']}: "
              f"{r.get('error') or ('cached' if r.get('cached') else r.get('prompts'))}", flush=True)
    summary = {**run_meta, "created": rs.now_kst(), "n_jobs": len(jobs), "rows": rows, "wall_s": time.time() - t0,
               "errors": [r for r in rows if r.get("error")]}
    rs.write_json(out_dir / f"summary_{time.strftime('%Y%m%dT%H%M%S')}.json", summary)
    return summary


__all__ = ["ATTN_SOURCES", "BACKENDS", "PerceptionOnlyRefiner", "StubPolicy", "build_safe_policy", "latch_arrays",
           "main", "pack_request", "plan_jobs", "request_row", "rows_to_arrays", "server_chunk_inputs",
           "serving_argv", "serving_parity", "wire_feedback"]


if __name__ == "__main__":
    main()
