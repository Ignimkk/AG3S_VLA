"""T55 — physics re-simulation of recorded rollouts with interventions, and the offline strong rule (CPU).

`AG3S/ag3s/docs/handoff/T55.task.md` 사전 고정 3 / 구현 1. The recorded **executed actions** are run in MuJoCo
from the episode's initial state — **no per-step qpos overwrite**. Fingers, contacts, ``applied_gripper``,
the measured opening and the three policy images therefore react to whether an object is in the hand.

How a record is re-simulated (E0 = no intervention)
---------------------------------------------------
* **Initial state = the client's own reset.** The client file (`integration/rby1/bringup/pi05_infer.py`;
  default: the T53 run snapshot's copy, else this checkout's) is loaded as a module and its ``main()`` is
  run with a minimal argv (model, episode, ``--warmup-steps`` / ``--start-delay`` of the record) until it
  asks for the remote policy; that call is stubbed to raise and the frame locals (``m``, ``d``, ``idx``,
  ``act``, the 299x224 policy renderer, the policy-hidden geoms) are captured — the T43W/T44 replay method
  (``hold_w.reset``), so the reset (keyframe, ``hold_keyframe_pose``, ``reset_randomized_scene`` with its
  1.5 s settle, ``mj_forward``) is the client's code, not a copy. Run outputs and logging are stubbed (no
  file is written). Then the record's warm-up: ``--warmup-steps N`` = N ``mj_step``; ``--start-delay 0`` =
  none (a wall-clock start delay cannot be re-simulated and is refused).
* **Control loop = the client's.** Every control step: (frame boundary every ``OPEN_LOOP_HORIZON`` = 8 steps)
  observation with the client's ``build_obs`` inside ``hidden_from_policy_render``; then the client's
  ``apply_action(action_format, action, ...)`` with the recorded executed action (T53: ``traj.npz
  executed_actions`` — the chunk row or the HOLD row that was actually applied; dev records without a
  trajectory: the first 8 rows of each recorded chunk, see "dev records" below), then
  ``steps_per_action`` (33) ``mj_step``. ``applied_gripper`` = ``d.ctrl[gripper] / RBY1_GRIPPER_OPEN``
  right after ``apply_action`` (the client's ``applied_ctrl``); measured opening at the next observation =
  ``|q(finger_x1)| / 0.045`` (the client's ``rby1_state``). HOLD decisions (which steps were executed) are
  the record's — the executed action of a HOLD step is the recorded HOLD row.
* **Execution feedback** per request k >= 1 = `wire`-shaped dict of chunk k-1 (``executed`` flags of the
  record, re-simulated ``applied_gripper``, re-simulated measured opening at frame k, both rounded to
  float32 as `wire.make_exec_feedback` does); request 0 has none (``available: False``), as in serving.
* **Physical state per frame** = `t48b_scene.classify` of every fruit on ``d`` as it is at the observation —
  its contacts / kinematics are those of the last ``mj_step``'s forward pass, i.e. exactly the pose the
  camera saw (the T48b / T53 ``render_qpos`` convention, without reconstructing it).

Interventions (parameters are CLI flags; dev fixes them)
--------------------------------------------------------
Applied right after the observation of policy frame ``f`` and ``offset_steps`` control steps later (0 = before
that frame's first action). The frames ``> f`` are the condition window (``window_start = f + 1``; with
offset > 0 the frame f+1 still follows the intervention).

* ``empty`` — remove the target before closing starts. ``f = f_close - lead_frames`` where ``f_close`` is the
  policy frame whose chunk contains the first **executed** used-hand step with ``applied_gripper <
  closed_below`` (0.85 = `LatchConfig.gripper_closed_below`, the strong rule's closing command).
  Method ``park``: the target's geoms lose all contacts (contype = conaffinity = 0) and its free joint is
  moved to ``park_pos`` (far behind / below the scene, outside every policy camera) with zero velocity,
  and re-pinned there after every control step (it would otherwise free-fall; it touches nothing).
* ``slip`` — detach the object from the hand after hold onset. ``f = f_hold + delay_frames`` where
  ``f_hold`` is the E0 frame at which raw P_task (target held by the used arm) has been true for
  ``after_frames`` consecutive frames. Methods: ``park`` (as above: the object leaves the scene) and
  ``ghost`` (the object keeps its velocity and falls out of the closed fingers: contacts between it and the
  active hand subtree — `t48b_scene.HAND_BODY` and below, fingers included — are switched off with a private
  collision bit, every other contact partner of the object is kept; the bits are restored at the first
  control-step boundary where the object is more than ``ghost_clear_m`` away from every hand geom,
  ``mj_geomDistance``; restore step reported, never restored within ``ghost_max_steps`` = reported).
  A slip intervention needs an E0 hold; records without one get ``not_applicable``.

Offline strong rule (`grasp_latch`, unchanged)
----------------------------------------------
``GraspLatch(LatchConfig(slip_detach=<record>), evidence=True, external_destination=True)`` — the
`SafePolicy` construction — fed per request with ``grasp_signal_from_feedback(re-sim feedback)`` and
``observe_grasp``. The non-feedback inputs (target lock, manipulated id/state, reach, placed, held
observation, attach rejection) come from one of two **perception** modes stored here — the **primary** mode
(user decision 2026-10-13) is neither: ``reground`` (`t55_perception`) re-runs the serving AG3S on the re-sim's own
image and depth. ``record`` is kept as the sensitivity mode (stale perception), ``oracle`` as the auxiliary
diagnostic:

* ``record`` (T53 only): the server's recorded ``summary_json.grasp`` of the same request (evidence
  ``reach_mm`` / ``manipulated_id`` / ``manipulated_state`` / ``placed``; ``attach_revoked.held_observation``
  or ``hold_checks`` for the held observation; ``rejected`` -> `revert_attach`), and the target lock
  (``update`` called ``confirm_frames`` times with one label at the end of the first request whose recorded
  end-of-request ``state`` is not ``searching`` — `SafePolicy._finish_grasp_record` writes the phase after
  ``_run_latch``). After an intervention these inputs are still the recorded ones —
  i.e. AG3S perception is assumed unchanged; only the execution feedback reacts.
* ``oracle``: lock at the end of request ``confirm_frames - 1`` (a stable label from request 0); manipulated id 0, state ``visible``; reach = MuJoCo
  distance between the parent link (URDF ``ee_finger_l1`` = **MJCF ``ee_finger_l2``** on the left — the T43
  name swap; right ``ee_finger_r1`` both) and the target body, the target position frozen at its last
  pre-park value; placed = target in the crate volume and not held; no held observation; no rejection.

Pre-intervention latch parity: the latch of a variant must equal the E0 latch on requests ``<= f`` (same
feedback); E0 latch (record mode) vs the recorded ``grasp.state`` is the E0 check of the offline rule.

dev records (T47 / T51 calibration)
-----------------------------------
They were recorded without ``--safe-remote`` and without ``--trajectory-out``: every step executed, the
executed action = the chunk row, but the step npz stores the chunk as **float32** while the policy server
returned float64 (state float64 + delta). The E0 of a dev record therefore uses rounded actions and is not
bit-identical (measured: ep1807 qpos |Δ| ~2e-8 rad until the grasp, then the fruit diverges; `long` /
`ep1800` — recorded before the head-keyframe fix — are excluded from the dev set, `DEV_EXCLUDED`).

**Baseline of a set (user decision 2026-10-13, `T55.task.md` 추가 사전 고정 1)** — `BASELINE_OF_SET`:

* ``dev`` -> ``unintervened_resim``: the dev baseline of every condition is the **unintervened re-simulation**
  (E0), not the recording. Every condition (E0 and the interventions) is the same re-simulation, and images,
  features and predicates are all made from it. The difference to the recording is still measured and reported
  per frame (`e0_parity`, ``per_frame``), but an exact match is not required.
* ``test`` -> ``record``: T53 re-simulates bit-exactly; an E0 that is not exact refuses the interventions of
  that record (STOP and report — never replaced by a qpos overwrite).

Slip baseline: a slip needs a hold of the unintervened re-sim. A record whose E0 never reaches one is
**excluded from the slip evaluation** (`slip_baseline`: ``excluded`` + reason, written to the variant json,
the record summary and the stage summary), because there is no baseline state to compare an intervention with.

Observer hook (`simulate(observer=...)`): called at every policy frame right after the observation and the
execution feedback of that request were made, before any ``mj_step`` — the moment the client packs its request
(`t55_perception` renders the serving depth there and runs the offline `SafePolicy`). The hook must not step
or edit the physics; `t55_perception` checks that the images it saw equal the stored re-sim images.

Outputs (`$AG3S_RUN_DIR/verify/T55/resim/<set>/`; refuses to write inside the T53 run)::

    <key>__E0.npz / .json          frames, images, physics, feedback, latch (record / oracle), parity
    <key>__<variant tag>.npz/.json  same for an intervention (+ window_start, intervention record)
    summary_<stage>.json

Usage::

    source scripts/setup/env.sh; export AG3S_RUN_DIR=...   # a new T55 run
    M=AG3S.ag3s.experiments.studies.t55_resim; R=<T53 run>; E=$R/verify/T53/runs/C1a/R
    S="$R/verify/T53/runs/C1a/server_8290_* $R/verify/T53/runs/C1a/server_8291_*"
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 .venv-ag3s/bin/python -m $M --set dev --t53-run-dir $R --variants E0
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 .venv-ag3s/bin/python -m $M --set test --t53-run-dir $R \\
        --episodes-root $E --server-roots $S --variants E0 empty slip --workers 4
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as _dt
import hashlib
import importlib.util
import inspect
import io
import json
import os
import pathlib
import sys
import time
from typing import Any, Optional, Sequence

import numpy as np

STEP = "T55"
SETS = ("dev", "test")
#: T55 dev records (lead, user decision 2026-10-13): T47 `long` / `ep1800` are excluded — recorded before the
#: X2 fix with head_1 = 0.0 (cam_high faces the wall).
DEV_KEYS = ("t2", "headfix", "ep1807", "ep1850", "ep1900", "ep0", "ep500", "ep1200")
DEV_EXCLUDED = ("long", "ep1800")
VARIANT_KINDS = ("E0", "empty", "slip")
EMPTY_METHODS = ("park",)
SLIP_METHODS = ("park", "ghost")
#: Offline strong-rule perception modes. ``record`` / ``oracle`` live in the re-sim npz; ``reground`` (primary,
#: user decision 2026-10-13) is `t55_perception`'s serving AG3S re-run on the re-sim observations.
PERCEPTION_MODES = ("record", "oracle")
PRIMARY_PERCEPTION = "reground"
#: The dev baseline is the unintervened re-sim; the test baseline is the (bit-exact) record.
BASELINE_OF_SET = {"dev": "unintervened_resim", "test": "record"}
SLIP_EXCLUSION_REASON = "unintervened re-sim never reaches a hold (no baseline state)"
#: Far behind and below the scene: outside every policy camera (checked by a segmentation test).
PARK_POS = (-20.0, 0.0, -5.0)
#: Private collision bit of the ghost release (model_transport uses bits 0-1).
GHOST_BIT = 1 << 24
#: URDF parent link of `SafePolicy.DEFAULT_GRASP_LINKS` -> the MJCF body of the same physical finger.
#: Left fingers are swapped between MJCF and URDF (T43 OC): URDF ee_finger_l1 = MJCF ee_finger_l2.
PARENT_BODY_MJCF = {"left": "ee_finger_l2", "right": "ee_finger_r1"}
RESIM_VERSION = "t55-resim-v2"
HAND_INDEX = {"left": 0, "right": 1}
#: Arm joints per side of the rby1_16d action (the client's ``ARM_JOINT_DIM``; `applied_ctrl` arm order).
ARM_DIM_16D = 7


# ------------------------------------------------------------------------------ plumbing


def now_kst() -> str:
    return _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).isoformat(timespec="seconds")


def jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, pathlib.Path):
        return str(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def write_json(path: pathlib.Path, payload: Any) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(jsonable(payload), indent=1, ensure_ascii=False))
    tmp.replace(path)


def save_npz_atomic(path: pathlib.Path, **arrays) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name[:-4] + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def md5_of(path: pathlib.Path) -> str:
    return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[4]


def out_root(sub: str = "", *, forbid: Sequence[pathlib.Path] = ()) -> pathlib.Path:
    """``$AG3S_RUN_DIR/verify/T55/<sub>``; refuses without a run dir or inside a forbidden run (the T53 run)."""
    from common.run_output import resolve_output_path

    run = os.environ.get("AG3S_RUN_DIR")
    if not run:
        raise SystemExit('AG3S_RUN_DIR is not set — export AG3S_RUN_DIR="$(.venv-openpi-live/bin/python '
                         '-m common.run_output)" (a new T55 run)')
    rel = "verify/T55" + (f"/{sub}" if sub else "")
    root = resolve_output_path(rel, run_dir=run, default_name=rel)
    for f in forbid:
        f = pathlib.Path(f).resolve()
        if root.resolve() == f or f in root.resolve().parents:
            raise SystemExit(f"refusing to write inside {f} (read-only input run)")
    return root


# ------------------------------------------------------------------------------ record sources


@dataclasses.dataclass
class Source:
    key: str
    set_name: str
    episode: int
    target: str
    used_arm: str
    prompt: str
    record_dir: pathlib.Path
    step_files: list
    t_steps: np.ndarray            # (F,) recorded observation t_step
    actions: np.ndarray            # (N, 16) executed actions (float64)
    executed: np.ndarray           # (N,) bool
    warmup_steps: Optional[int]
    start_delay: Optional[float]
    seed: Optional[int] = None
    ref_measured: Optional[np.ndarray] = None        # (N, 16) traj measured_qpos
    ref_applied_gripper: Optional[np.ndarray] = None  # (N, 2) frames control applied_ctrl.gripper
    ref_feedback: Optional[list] = None               # per request: frames planning exec_feedback
    ref_grasp: Optional[list] = None                  # per request: server summary_json.grasp
    slip_detach: bool = False
    action_source: str = ""
    ref_chunk_files: Optional[list] = None            # per request: server chunk npz (T53 only)
    notes: list = dataclasses.field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return int(len(self.step_files))

    @property
    def n_steps(self) -> int:
        return int(len(self.actions))

    def describe(self) -> dict:
        return {"key": self.key, "set": self.set_name, "episode": self.episode, "target": self.target,
                "used_arm": self.used_arm, "prompt": self.prompt, "record_dir": str(self.record_dir),
                "n_frames": self.n_frames, "n_steps": self.n_steps, "warmup_steps": self.warmup_steps,
                "start_delay": self.start_delay, "seed": self.seed, "action_source": self.action_source,
                "has_ref_measured": self.ref_measured is not None,
                "has_ref_feedback": self.ref_feedback is not None,
                "has_ref_grasp": self.ref_grasp is not None, "slip_detach": self.slip_detach,
                "n_executed": int(self.executed.sum()), "notes": list(self.notes)}


def prompt_fruit(prompt: str) -> Optional[str]:
    from AG3S.ag3s.experiments.studies import t53_records as rc

    return rc.prompt_fruit(prompt)


def _argv_value(argv: Sequence[str], flag: str) -> Optional[str]:
    argv = list(argv)
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def dataset_meta() -> dict[int, dict]:
    from AG3S.ag3s.experiments.studies import t53_records as rc

    return rc.load_dataset_meta()


def dev_source(key: str, *, horizon: int = 8, meta_ds: Optional[dict] = None) -> Source:
    """A T47 / T51-calibration record (step npz only; no safe-remote, no trajectory)."""
    from AG3S.ag3s.experiments.studies import t51_records as rc51

    if key not in DEV_KEYS:
        raise SystemExit(f"unknown dev key {key!r}; have {list(DEV_KEYS)}")
    rec = rc51.record_path(key)
    meta = json.loads((rec / "meta.json").read_text())
    argv = meta.get("argv") or []
    files = sorted(rec.glob("step_*.npz"))
    if not files or len(files) != int(meta.get("n_steps", len(files))):
        raise SystemExit(f"[{key}] {len(files)} step files, meta n_steps {meta.get('n_steps')}")
    t_steps, acts = [], []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            t_steps.append(int(z["t_step"]))
            acts.append(np.asarray(z["actions"][:horizon], np.float64))
    t_steps = np.asarray(t_steps, np.int64)
    if not np.array_equal(t_steps, np.arange(len(files)) * horizon):
        raise SystemExit(f"[{key}] t_step {t_steps[:5]}... is not 0, {horizon}, 2*{horizon}, ...")
    max_steps = int(_argv_value(argv, "--max-steps") or len(files) * horizon)
    actions = np.concatenate(acts)[:max_steps]
    ep = int(_argv_value(argv, "--episode-index"))
    if ep != int(rc51.EPISODE_OF[key]):
        raise SystemExit(f"[{key}] argv episode {ep} != registry {rc51.EPISODE_OF[key]}")
    meta_ds = meta_ds if meta_ds is not None else dataset_meta()
    used = str(meta_ds[ep]["used_arm"])
    target = prompt_fruit(meta["prompt"])
    if target != str(meta_ds[ep]["target_fruit"]):
        raise SystemExit(f"[{key}] prompt fruit {target} != dataset target {meta_ds[ep]['target_fruit']}")
    warm = _argv_value(argv, "--warmup-steps")
    delay = _argv_value(argv, "--start-delay")
    return Source(key=key, set_name="dev", episode=ep, target=target, used_arm=used, prompt=meta["prompt"],
                  record_dir=rec, step_files=files, t_steps=t_steps, actions=actions,
                  executed=np.ones(len(actions), bool),
                  warmup_steps=None if warm is None else int(warm),
                  start_delay=2.0 if (delay is None and warm is None) else (None if delay is None else float(delay)),
                  action_source="step npz chunk rows [:8] (float32 record of float64 actions)",
                  notes=["no safe-remote: every step executed, no HOLD, no exec feedback / latch in the record"])


def t53_source(key: str, *, t53_run_dir: pathlib.Path, episodes_root: Optional[pathlib.Path] = None,
               server_roots: Sequence[pathlib.Path] = (), horizon: int = 8,
               meta_ds: Optional[dict] = None) -> Source:
    """A T53 final-test rollout (step npz + traj.npz + frames.jsonl + server chunks), read-only."""
    from AG3S.ag3s.experiments.studies import t53_records as rc

    rc.configure(run_dir=t53_run_dir, episodes_root=episodes_root)
    key = rc.resolve_keys([key])[0]
    files = rc.locate(key)
    if files.trajectory is None or files.frames is None:
        raise SystemExit(f"[{key}] needs traj.npz and frames.jsonl ({files.as_dict()})")
    steps = sorted(files.record.glob("step_*.npz"))
    meta = rc.read_meta(files.record)
    probs = rc.record_problems(key, meta)
    if probs:
        raise SystemExit(f"[{key}] record does not match the T53 registry: {probs}")
    t_steps = np.asarray([int(np.load(f, allow_pickle=False)["t_step"]) for f in steps], np.int64)
    with np.load(files.trajectory, allow_pickle=False) as z:
        actions = np.asarray(z["executed_actions"], np.float64)
        measured = np.asarray(z["measured_qpos"], np.float64)
        css = np.asarray(z["chunk_start_steps"], np.int64)
        warm = int(z["warmup_steps"]) if "warmup_steps" in z.files else None
        seed = int(z["policy_seed"]) if "policy_seed" in z.files else None
        ctrl_h = int(z["execution_length"]) if "execution_length" in z.files else horizon
    if ctrl_h != horizon or not np.array_equal(css, t_steps):
        raise SystemExit(f"[{key}] chunk_start_steps / execution_length do not match the step records")
    rows = rc.read_frames(files.frames)
    control = [r for r in rows if r.get("kind") == "control"]
    planning = [r for r in rows if r.get("kind") == "planning"]
    if len(control) != len(actions) or len(planning) != len(steps):
        raise SystemExit(f"[{key}] {len(control)} control rows / {len(actions)} actions, "
                         f"{len(planning)} planning rows / {len(steps)} steps")
    executed = np.asarray([bool(r.get("executed")) for r in control], bool)
    applied = np.asarray([np.asarray((r.get("applied_ctrl") or {}).get("gripper", [np.nan, np.nan]), np.float64)
                          for r in control]).reshape(-1, 2)
    feedback = [r.get("exec_feedback") for r in planning]
    grasp = None
    chunk_files = None
    notes = []
    chunks = rc.locate_chunks(key, ep_dir=files.episode_dir, server_roots=list(server_roots), planning=planning)
    if chunks["chunks"]:
        if len(chunks["chunks"]) != len(steps):
            raise SystemExit(f"[{key}] {len(chunks['chunks'])} server chunks for {len(steps)} requests")
        grasp = []
        chunk_files = list(chunks["chunks"])
        for c in chunks["chunks"]:
            with np.load(c, allow_pickle=False) as z:
                grasp.append((json.loads(str(z["summary_json"])).get("grasp") or {}))
        notes.append(f"server chunks: {chunks['source']}")
    else:
        notes.append(f"no server chunks ({chunks['source']}) — record-mode latch unavailable")
    meta_ds = meta_ds if meta_ds is not None else dataset_meta()
    ep = rc.episode_of(key).episode
    return Source(key=key, set_name="test", episode=int(ep), target=rc.target_of(key),
                  used_arm=str(meta_ds[int(ep)]["used_arm"]), prompt=str(meta["prompt"]),
                  record_dir=files.record, step_files=steps, t_steps=t_steps, actions=actions,
                  executed=executed, warmup_steps=warm, start_delay=None, seed=seed,
                  ref_measured=measured, ref_applied_gripper=applied, ref_feedback=feedback, ref_grasp=grasp,
                  slip_detach=bool(grasp and any("held_slip" in g for g in grasp)),
                  action_source="traj.npz executed_actions", notes=notes, ref_chunk_files=chunk_files)


# ------------------------------------------------------------------------------ the client


class _Captured(Exception):
    pass


def default_client(t53_run_dir: Optional[pathlib.Path]) -> pathlib.Path:
    if t53_run_dir is not None:
        snap = pathlib.Path(t53_run_dir) / "snapshot" / "integration" / "rby1" / "bringup" / "pi05_infer.py"
        if snap.is_file():
            return snap
    return repo_root() / "integration" / "rby1" / "bringup" / "pi05_infer.py"


def load_client(path: pathlib.Path):
    """The client file as a module, run outputs / logging stubbed (the replay writes nothing)."""
    path = pathlib.Path(path).resolve()
    if not path.is_file():
        raise SystemExit(f"client {path} not found")
    spec = importlib.util.spec_from_file_location(f"_t55_client_{md5_of(path)[:8]}", path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    mod.prepare_run_outputs = lambda *a, **k: None
    mod.configure_run_logging = lambda *a, **k: None
    return mod


def client_argv(src: Source) -> list[str]:
    argv = ["pi05_infer.py", "--model", "rby1_randomized_pick_place_16d", "--remote", "localhost:1",
            "--episode-index", str(src.episode), "--headless", "--speed", "0"]
    if src.warmup_steps is not None:
        argv += ["--warmup-steps", str(int(src.warmup_steps))]
    elif src.start_delay is not None and float(src.start_delay) == 0.0:
        argv += ["--start-delay", "0"]
    else:
        raise SystemExit(f"[{src.key}] start delay {src.start_delay} s is wall-clock paced — not re-simulable")
    return argv


def client_reset(cl, src: Source) -> dict:
    """Run the client's ``main()`` up to the remote-policy load and capture its locals."""
    cap: dict = {}

    def stub(remote):
        cap.update(inspect.currentframe().f_back.f_locals)
        raise _Captured

    orig_load, orig_argv = cl.load_remote_policy, sys.argv
    cl.load_remote_policy = stub
    sys.argv = client_argv(src)
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            cl.main()
    except _Captured:
        pass
    finally:
        cl.load_remote_policy = orig_load
        sys.argv = orig_argv
    need = ("m", "d", "idx", "act", "renderer_pol", "mcfg", "args")
    missing = [k for k in need if k not in cap]
    if missing:
        raise RuntimeError(f"client main() did not reach load_remote_policy (missing {missing}); "
                           f"output tail: {out.getvalue()[-800:]}")
    if cap["mcfg"]["action_format"] != "rby1_16d" or cap["mcfg"]["obs_format"] != "rby1_16d":
        raise RuntimeError(f"unexpected formats {cap['mcfg']}")
    return cap


# ------------------------------------------------------------------------------ interventions


@dataclasses.dataclass(frozen=True)
class Intervention:
    kind: str                       # "empty" | "slip"
    method: str                     # "park" | "ghost"
    lead_frames: int = 0            # empty
    after_frames: int = 2           # slip
    delay_frames: int = 0           # slip
    offset_steps: int = 0
    park_pos: tuple = PARK_POS
    ghost_clear_m: float = 0.005
    ghost_max_steps: int = 30

    def __post_init__(self):
        if self.kind not in ("empty", "slip"):
            raise ValueError(f"kind {self.kind!r}")
        allowed = EMPTY_METHODS if self.kind == "empty" else SLIP_METHODS
        if self.method not in allowed:
            raise ValueError(f"{self.kind} method must be one of {allowed}, got {self.method!r}")
        if min(self.lead_frames, self.delay_frames, self.offset_steps) < 0 or self.after_frames < 1:
            raise ValueError("lead/delay/offset >= 0, after_frames >= 1")

    @property
    def tag(self) -> str:
        if self.kind == "empty":
            return f"empty_L{self.lead_frames}O{self.offset_steps}_{self.method}"
        return f"slip_A{self.after_frames}D{self.delay_frames}O{self.offset_steps}_{self.method}"

    def params(self) -> dict:
        return dataclasses.asdict(self) | {"tag": self.tag}


def first_closing_frame(src: Source, horizon: int, closed_below: float) -> Optional[int]:
    """Policy frame of the first executed used-hand step whose applied gripper < `closed_below`."""
    col = 7 if src.used_arm == "left" else 15
    g = np.clip(np.asarray(src.actions[:, col], np.float64), 0.0, 1.0)  # apply_action clips
    hit = np.flatnonzero(src.executed & (g < float(closed_below)))
    return None if hit.size == 0 else int(hit[0] // horizon)


def hold_onset_frame(p_task_raw: Sequence[bool], after_frames: int) -> Optional[int]:
    run = 0
    for i, v in enumerate(np.asarray(p_task_raw, bool)):
        run = run + 1 if v else 0
        if run >= int(after_frames):
            return i
    return None


def intervention_frame(iv: Intervention, src: Source, e0_p_task_raw: Optional[Sequence[bool]], *,
                       horizon: int, closed_below: float) -> tuple[Optional[int], str]:
    if iv.kind == "empty":
        fc = first_closing_frame(src, horizon, closed_below)
        if fc is None:
            return None, "no executed closing command of the used hand"
        f = fc - iv.lead_frames
        return (f, f"f_close={fc}") if f >= 0 else (None, f"f_close={fc} < lead {iv.lead_frames}")
    if e0_p_task_raw is None:
        return None, "slip needs the E0 predicates"
    fh = hold_onset_frame(e0_p_task_raw, iv.after_frames)
    if fh is None:
        return None, f"no hold of {iv.after_frames} consecutive frames in E0"
    f = fh + iv.delay_frames
    if f >= len(e0_p_task_raw):
        return None, f"f_hold={fh} + delay beyond the episode"
    return f, f"f_hold={fh}"


class _Mods:
    """Model / data edits of an intervention, kept to undo or re-pin."""

    def __init__(self, m, sidx, fruit: str):
        import mujoco

        self.mujoco = mujoco
        self.m = m
        self.body = int(sidx.fruit_body[fruit])
        self.geoms = [int(g) for g in np.flatnonzero(m.geom_bodyid == self.body)]
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{fruit}_free")
        self.qadr = int(m.jnt_qposadr[j])
        self.vadr = int(m.jnt_dofadr[j])
        self.parked: Optional[np.ndarray] = None
        self.saved: dict[int, tuple[int, int]] = {}
        self.ghost_hand: list[int] = []
        self.restored_at: Optional[int] = None

    def park(self, d, pos: Sequence[float]) -> None:
        for g in self.geoms:
            self.saved.setdefault(g, (int(self.m.geom_contype[g]), int(self.m.geom_conaffinity[g])))
            self.m.geom_contype[g] = 0
            self.m.geom_conaffinity[g] = 0
        q = np.array(d.qpos[self.qadr:self.qadr + 7], np.float64)
        q[:3] = np.asarray(pos, np.float64)
        self.parked = q
        self.pin(d)

    def pin(self, d) -> None:
        if self.parked is not None:
            d.qpos[self.qadr:self.qadr + 7] = self.parked
            d.qvel[self.vadr:self.vadr + 6] = 0.0

    def ghost(self, hand_geoms: Sequence[int]) -> dict:
        m = self.m
        hand = set(int(g) for g in hand_geoms)
        obj = [g for g in self.geoms if (int(m.geom_contype[g]) | int(m.geom_conaffinity[g])) != 0]
        if not obj:
            raise RuntimeError("object has no colliding geom")
        oc = [(int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for g in obj]
        partners = []
        for g in range(int(m.ngeom)):
            if g in hand or g in self.geoms:
                continue
            ct, ca = int(m.geom_contype[g]), int(m.geom_conaffinity[g])
            if (ct | ca) == 0:
                continue
            if any((ct & a) or (c & ca) for c, a in oc):
                partners.append(g)
        for g in obj + partners:
            self.saved.setdefault(g, (int(m.geom_contype[g]), int(m.geom_conaffinity[g])))
        for g in obj:
            m.geom_contype[g] = GHOST_BIT
            m.geom_conaffinity[g] = GHOST_BIT
        for g in partners:
            m.geom_conaffinity[g] = int(m.geom_conaffinity[g]) | GHOST_BIT
        self.ghost_hand = sorted(hand)
        return {"object_geoms": obj, "n_partners": len(partners), "hand_geoms": len(hand)}

    def hand_distance(self, d, cap: float) -> float:
        best = float(cap)
        fromto = np.zeros(6)
        for g in self.geoms:
            for h in self.ghost_hand:
                best = min(best, float(self.mujoco.mj_geomDistance(self.m, d, g, h, float(cap), fromto)))
        return best

    def restore(self) -> None:
        for g, (ct, ca) in self.saved.items():
            self.m.geom_contype[g] = ct
            self.m.geom_conaffinity[g] = ca
        self.saved.clear()


def hand_geoms(m, sidx, hand: str) -> list[int]:
    """Collision geoms of the subtree of the hand body (`t48b_scene.HAND_BODY`, fingers included)."""
    root = int(sidx.hand_body[hand])
    out = []
    for g in range(int(m.ngeom)):
        b = int(m.geom_bodyid[g])
        x = b
        while x > 0:
            if x == root:
                break
            x = int(m.body_parentid[x])
        if x == root and (int(m.geom_contype[g]) | int(m.geom_conaffinity[g])) != 0:
            out.append(g)
    return out


# ------------------------------------------------------------------------------ simulation


def wire32(x) -> np.ndarray:
    """float64 -> float32 -> float64: the value the server unpacks from the wire."""
    return np.asarray(np.asarray(x, np.float64).astype(np.float32), np.float64)


def _hwc(img_chw) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(np.asarray(img_chw), (1, 2, 0)))


def simulate(cl, src: Source, *, intervention: Optional[Intervention] = None, frame: Optional[int] = None,
             keep_images: bool = True, observer=None) -> dict:
    """One re-simulation of `src` (module docstring). `frame` = the intervention frame (resolved).

    `observer(k, ctx)` (optional) is called at policy frame k after the observation and that request's
    execution feedback, before the frame's first action: ``ctx`` = ``m``, ``d``, ``obs`` (the client's
    `build_obs` dict), ``feedback`` (this request's), ``chunk_steps`` (the previous chunk's per-step
    ``executed`` / ``applied_gripper`` / ``applied_arm`` / ``t``), ``t`` (control step), ``sim_time``.
    It must not step or edit the physics."""
    import mujoco

    from AG3S.ag3s.experiments.studies import t48b_scene as sc

    cap = client_reset(cl, src)
    m, d, idx, act, ren = cap["m"], cap["d"], cap["idx"], cap["act"], cap["renderer_pol"]
    hidden = cap.get("policy_hidden_geoms") or ()
    H = int(cl.OPEN_LOOP_HORIZON)
    spa = max(1, int(round(1.0 / (cl.CTRL_HZ * m.opt.timestep))))
    sidx = sc.SceneIndex.from_model(m)
    grip_open = float(cl.RBY1_GRIPPER_OPEN)
    fruits = list(sc.FRUITS)
    parent = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PARENT_BODY_MJCF[src.used_arm])
    if parent < 0:
        raise RuntimeError(f"MJCF body {PARENT_BODY_MJCF[src.used_arm]} missing")
    t0 = time.time()
    try:
        if src.warmup_steps:
            for _ in range(int(src.warmup_steps)):
                mujoco.mj_step(m, d)
        N, F = src.n_steps, src.n_frames
        if N > F * H or N <= (F - 1) * H:
            raise RuntimeError(f"[{src.key}] {N} actions for {F} frames of {H}")
        rec: dict[str, list] = {k: [] for k in (
            "image_cam_high", "image_cam_left_wrist", "image_cam_right_wrist", "state", "qpos", "qvel",
            "measured_gripper", "phys_state", "phys_hand", "finger_contact", "in_crate", "fruit_pos",
            "penetration", "parent_pos")}
        applied_g = np.zeros((N, 2))
        applied_arm = np.zeros((N, 2 * ARM_DIM_16D))
        measured = np.zeros((N, 16))
        feedback: list[dict] = []
        chunk_exec, chunk_applied, chunk_t = [], [], []
        mods: Optional[_Mods] = None
        iv_rec: dict = {"applied": False}
        target_pos_frozen: list = []
        last_target = None
        for t in range(N):
            k, s = divmod(t, H)
            if s == 0:
                with cl.hidden_from_policy_render(m, hidden):
                    obs = cl.build_obs("rby1_16d", m, d, ren, idx, src.prompt)
                st = np.asarray(obs["state"], np.float64)
                for c in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                    rec[f"image_{c}"].append(_hwc(obs["images"][c]) if keep_images else np.zeros((0,), np.uint8))
                rec["state"].append(st)
                rec["qpos"].append(np.array(d.qpos, np.float64))
                rec["qvel"].append(np.array(d.qvel, np.float64))
                mg = np.array([st[7], st[15]], np.float64)
                rec["measured_gripper"].append(mg)
                ph, hd, fc, ic, fp = [], [], [], [], []
                for fr in fruits:
                    p = sc.classify(m, d, sidx, fr)
                    ph.append(p.state)
                    hd.append(p.hand or "")
                    fc.append([p.finger_contact["left"], p.finger_contact["right"]])
                    b = sidx.fruit_body[fr]
                    ic.append(sc.point_in_crate(d.xpos[b], d.xpos[sidx.crate_body], d.xmat[sidx.crate_body],
                                                sidx.crate_volume))
                    fp.append(np.array(d.xpos[b], np.float64))
                rec["phys_state"].append(ph)
                rec["phys_hand"].append(hd)
                rec["finger_contact"].append(fc)
                rec["in_crate"].append(ic)
                rec["fruit_pos"].append(fp)
                rec["penetration"].append(sc.max_penetration(
                    m, d, [sidx.fruit_body[f] for f in fruits]
                    + [b for h in sc.HANDS for b in sidx.finger_body[h]]))
                rec["parent_pos"].append(np.array(d.xpos[parent], np.float64))
                # oracle reach: the target's position, frozen at its last value once it is parked
                target_parked = mods is not None and mods.parked is not None \
                    and mods.body == sidx.fruit_body[src.target]
                if not target_parked or last_target is None:
                    last_target = np.array(d.xpos[sidx.fruit_body[src.target]], np.float64)
                target_pos_frozen.append(last_target)
                # execution feedback of the previous chunk (client: ExecutionLog.close at this observation)
                if k == 0 or not chunk_exec:
                    feedback.append({"available": False, "reason": "no chunk has run yet in this episode"})
                else:
                    ex = list(chunk_exec)
                    # wire format: float32 arrays (`wire.make_exec_feedback` / `_rows`) — the server's
                    # latch saw these rounded values, so the offline latch gets the same.
                    feedback.append({"available": True, "seq": int(k), "t_step_start": int((k - 1) * H),
                                     "n_steps": len(ex), "n_exec": int(sum(ex)), "executed": ex,
                                     "applied_gripper": wire32(chunk_applied).tolist(),
                                     "measured_gripper": wire32(mg).tolist()})
                if observer is not None:
                    observer(int(k), {"m": m, "d": d, "obs": obs, "feedback": feedback[-1], "t": int(t),
                                      "sim_time": float(d.time),
                                      "chunk_steps": {"t": list(chunk_t), "executed": list(chunk_exec),
                                                      "applied_gripper": [list(x) for x in chunk_applied],
                                                      "applied_arm": [applied_arm[i].tolist() for i in chunk_t]}})
                chunk_exec, chunk_applied, chunk_t = [], [], []
            if intervention is not None and frame is not None and t == frame * H + intervention.offset_steps:
                obj = src.target
                mods = _Mods(m, sidx, obj)
                iv_rec = {"applied": True, "step": int(t), "frame": int(frame), "object": obj,
                          "object_pos_before": np.array(d.xpos[mods.body], np.float64).tolist()}
                if intervention.method == "park":
                    mods.park(d, intervention.park_pos)
                else:
                    iv_rec["ghost"] = mods.ghost(hand_geoms(m, sidx, src.used_arm))
            cl.apply_action("rby1_16d", np.asarray(src.actions[t], np.float64), d, idx, act)
            g = np.array([d.ctrl[act["left_grip_a"]], d.ctrl[act["right_grip_a"]]], np.float64) / grip_open
            applied_g[t] = g
            applied_arm[t] = [d.ctrl[a] for a in act["left_a"][:ARM_DIM_16D]] + \
                [d.ctrl[a] for a in act["right_a"][:ARM_DIM_16D]]
            chunk_exec.append(bool(src.executed[t]))
            chunk_applied.append(g.tolist())
            chunk_t.append(int(t))
            for _ in range(spa):
                mujoco.mj_step(m, d)
            if mods is not None:
                mods.pin(d)
                if intervention is not None and intervention.method == "ghost" and mods.saved \
                        and mods.restored_at is None:
                    dist = mods.hand_distance(d, cap=0.05)
                    if dist > float(intervention.ghost_clear_m):
                        mods.restore()
                        mods.restored_at = int(t)
                    elif t - iv_rec["step"] >= int(intervention.ghost_max_steps):
                        iv_rec["ghost_not_restored"] = True
            lq = [d.qpos[i] for i in idx["left_q"]]
            rq = [d.qpos[i] for i in idx["right_q"]]
            measured[t] = np.concatenate([lq, [abs(d.qpos[idx["left_grip_q"]]) / abs(grip_open)],
                                          rq, [abs(d.qpos[idx["right_grip_q"]]) / abs(grip_open)]])
        if mods is not None and intervention is not None and intervention.method == "ghost":
            iv_rec["ghost_restored_step"] = mods.restored_at
    finally:
        for r in (cap.get("renderer_pol"), cap.get("renderer_rec")):
            with contextlib.suppress(Exception):
                r.close()
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["phys_state"] = out["phys_state"].astype("U12")
    out["phys_hand"] = out["phys_hand"].astype("U8")
    out.update({"t_step": np.arange(F, dtype=np.int64) * H, "applied_gripper": applied_g, "measured": measured,
                "applied_arm": applied_arm,
                "executed": np.asarray(src.executed, bool), "target_pos_frozen": np.asarray(target_pos_frozen),
                "fruits": np.asarray(fruits)})
    return {"arrays": out, "feedback": feedback, "intervention": iv_rec, "seconds": time.time() - t0,
            "spa": spa, "horizon": H, "nq": int(m.nq), "nv": int(m.nv)}


# ------------------------------------------------------------------------------ predicates of a run


def held_by(arrays: dict) -> dict[str, list]:
    fruits = [str(f) for f in arrays["fruits"]]
    st, hd = arrays["phys_state"], arrays["phys_hand"]
    return {f: [(str(hd[i, j]) or None) if str(st[i, j]) == "held" else None for i in range(len(st))]
            for j, f in enumerate(fruits)}


def raw_preds(arrays: dict, target: str, used_arm: str) -> dict:
    from AG3S.ag3s.experiments.studies import t55_predicate as pr

    fruits = [str(f) for f in arrays["fruits"]]
    return pr.raw_predicates(held_by(arrays), target, used_arm,
                             in_crate=np.asarray(arrays["in_crate"])[:, fruits.index(target)])


# ------------------------------------------------------------------------------ offline latch


def perception_from_record(grasp: Sequence[dict], cfg) -> tuple[list[dict], Optional[int]]:
    """Per request: the recorded non-feedback inputs of `observe_grasp`, and the lock request."""
    per, lock = [], None
    for k, g in enumerate(grasp):
        g = g or {}
        ev = g.get("evidence") or {}
        # the recorded ``state`` is the phase at the end of the request (after `_run_latch`'s ``update``),
        # ``state_before`` the phase before ``observe_grasp``: the lock happened in the first request that
        # ends outside ``searching``
        if lock is None and g.get("state") not in (None, "searching"):
            lock = k
        held = None
        rv = g.get("attach_revoked") or {}
        if rv.get("held_observation"):
            held = dict(rv["held_observation"])
        else:
            hc = ev.get("hold_checks") or {}
            if "observed_distance_mm" in hc:
                held = {"centre_distance_m": float(hc["observed_distance_mm"]) / 1000.0,
                        "held_radius_m": float(hc["observed_limit_mm"]) / 1000.0 - float(cfg.revoke_centre_slack),
                        "hand_rise_m": float(hc["hand_rise_mm"]) / 1000.0,
                        "object_rise_m": float(hc["object_rise_mm"]) / 1000.0}
        rej = g.get("rejected")
        per.append({"reach_m": None if ev.get("reach_mm") is None else float(ev["reach_mm"]) / 1000.0,
                    "manipulated_id": ev.get("manipulated_id"), "manipulated_state": ev.get("manipulated_state"),
                    "placed": bool(ev.get("placed", False)), "held_observation": held,
                    "reject_reason": (rej.get("reason") if isinstance(rej, dict) else (str(rej) if rej else None))})
    return per, lock


def perception_oracle(arrays: dict, target: str, used_arm: str, cfg) -> tuple[list[dict], int]:
    fruits = [str(f) for f in arrays["fruits"]]
    j = fruits.index(target)
    parent = np.asarray(arrays["parent_pos"], np.float64)
    tpos = np.asarray(arrays["target_pos_frozen"], np.float64)
    st = arrays["phys_state"][:, j]
    in_crate = np.asarray(arrays["in_crate"])[:, j]
    per = []
    for k in range(len(parent)):
        per.append({"reach_m": float(np.linalg.norm(parent[k] - tpos[k])), "manipulated_id": 0,
                    "manipulated_state": "visible", "placed": bool(in_crate[k] and str(st[k]) != "held"),
                    "held_observation": None, "reject_reason": None})
    return per, int(cfg.confirm_frames) - 1


def latch_config(slip_detach: bool):
    from AG3S.trajopt.grasp_latch import LatchConfig

    return LatchConfig(slip_detach=bool(slip_detach))


def offline_latch(feedback: Sequence[dict], perception: Sequence[dict], *, lock_request: Optional[int],
                  used_arm: str, cfg) -> dict[str, np.ndarray]:
    """`grasp_latch` evidence mode on given feedback + perception (module docstring)."""
    from AG3S.trajopt.grasp_latch import GraspLatch, GraspPhase, grasp_signal_from_feedback

    if len(feedback) != len(perception):
        raise ValueError(f"{len(feedback)} feedback vs {len(perception)} perception requests")
    latch = GraspLatch(cfg, evidence=True, external_destination=True)
    hand = HAND_INDEX[used_arm]
    cols: dict[str, list] = {k: [] for k in ("phase", "phase_before", "attach", "detach", "revoked", "slipped",
                                             "note", "closing_steps", "opening")}
    for k, (fb, p) in enumerate(zip(feedback, perception)):
        before = latch.phase
        sig = grasp_signal_from_feedback(fb, hand, cfg.gripper_closed_below)
        ev = latch.observe_grasp(sig, reach_m=p["reach_m"], manipulated_id=p["manipulated_id"],
                                 manipulated_state=p["manipulated_state"], placed=bool(p["placed"]),
                                 held_observation=(p.get("held_observation") if latch.holding else None))
        if ev.attach and p.get("reject_reason"):
            ev = latch.revert_attach(str(p["reject_reason"]))
        if lock_request is not None and k == int(lock_request) and latch.phase is GraspPhase.SEARCHING:
            # `SafePolicy._run_latch` after the constraints: the same label `confirm_frames` times locks
            for _ in range(int(cfg.confirm_frames)):
                latch.update(label="obj0", score=1.0, runner_up=0.0)
        cols["phase"].append(latch.phase.value)
        cols["phase_before"].append(before.value)
        cols["attach"].append(bool(ev.attach))
        cols["detach"].append(bool(ev.detach))
        cols["revoked"].append(bool(ev.revoked))
        cols["slipped"].append(bool(ev.slipped))
        cols["note"].append(str(ev.note)[:300])
        cols["closing_steps"].append(int(sig.closing_steps))
        cols["opening"].append(np.nan if sig.opening is None else float(sig.opening))
    out = {k: np.asarray(v) for k, v in cols.items()}
    out["phase"] = out["phase"].astype("U10")
    out["phase_before"] = out["phase_before"].astype("U10")
    return out


def run_latches(src: Source, res: dict) -> dict[str, dict]:
    cfg = latch_config(src.slip_detach)
    out = {}
    per, lock = perception_oracle(res["arrays"], src.target, src.used_arm, cfg)
    out["oracle"] = offline_latch(res["feedback"], per, lock_request=lock, used_arm=src.used_arm, cfg=cfg)
    if src.ref_grasp is not None:
        per, lock = perception_from_record(src.ref_grasp, cfg)
        out["record"] = offline_latch(res["feedback"], per, lock_request=lock, used_arm=src.used_arm, cfg=cfg)
    return out


# ------------------------------------------------------------------------------ E0 parity


def _maxabs(a, b) -> Optional[float]:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return None
    return float(np.nanmax(np.abs(a - b))) if a.size else 0.0


def _feedback_equal(a: Optional[dict], b: Optional[dict]) -> bool:
    a, b = a or {}, b or {}
    if bool(a.get("available")) != bool(b.get("available")):
        return False
    if not a.get("available"):
        return True
    return (list(map(bool, a.get("executed") or [])) == list(map(bool, b.get("executed") or []))
            and _maxabs(a.get("applied_gripper"), b.get("applied_gripper")) == 0.0
            and _maxabs(a.get("measured_gripper"), b.get("measured_gripper")) == 0.0)


def e0_parity(src: Source, res: dict, latches: Optional[dict] = None) -> dict:
    """E0 re-simulation vs the record — numbers per quantity; ``exact`` only when every one is 0."""
    a = res["arrays"]
    F = src.n_frames
    per = {"qpos": [], "qvel": [], "state": [], "img_max": [], "img_mad": []}
    first_bad = None
    for k, f in enumerate(src.step_files):
        with np.load(f, allow_pickle=False) as z:
            dq = _maxabs(a["qpos"][k], z["qpos"])
            dv = _maxabs(a["qvel"][k], z["qvel"])
            ds = _maxabs(a["state"][k], z["state"])
            im, mad = 0, 0.0
            if a["image_cam_high"].ndim == 4:
                for c in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                    diff = np.abs(a[f"image_{c}"][k].astype(np.int16) - np.asarray(z[f"image_{c}"], np.int16))
                    im, mad = max(im, int(diff.max())), max(mad, float(diff.mean()))
        for name, v in (("qpos", dq), ("qvel", dv), ("state", ds), ("img_max", im), ("img_mad", mad)):
            per[name].append(np.nan if v is None else float(v))
        if first_bad is None and any((v is None) or v != 0 for v in (dq, dv, ds, im)):
            first_bad = k
    out: dict = {"n_frames": F,
                 "qpos_max_abs": float(np.nanmax(per["qpos"])), "qvel_max_abs": float(np.nanmax(per["qvel"])),
                 "state_max_abs": float(np.nanmax(per["state"])), "image_max_abs": int(np.nanmax(per["img_max"])),
                 "image_mad_max": float(np.nanmax(per["img_mad"])),
                 "frames_mismatched": int(sum(1 for i in range(F) if any(
                     (not np.isfinite(per[n][i])) or per[n][i] != 0 for n in ("qpos", "qvel", "state", "img_max")))),
                 "first_mismatch_frame": first_bad,
                 "per_frame": {n: per[n] for n in ("qpos", "qvel", "state", "img_max", "img_mad")}}
    if a["image_cam_high"].ndim != 4:
        out["image_note"] = "images not kept — image parity not measured"
    if src.ref_measured is not None:
        d = np.abs(a["measured"] - src.ref_measured).max(1)
        out["measured_max_abs"] = float(d.max())
        out["measured_rows_mismatched"] = int((d != 0).sum())
        Hs = int(res.get("horizon", 8))
        out["per_frame"]["measured"] = [float(d[i * Hs:(i + 1) * Hs].max(initial=0.0)) for i in range(F)]
    if src.ref_applied_gripper is not None:
        out["applied_gripper_max_abs"] = _maxabs(a["applied_gripper"], src.ref_applied_gripper)
    else:
        col = np.clip(src.actions[:, [7, 15]], 0.0, 1.0)
        out["applied_gripper_vs_clipped_action_max_abs"] = _maxabs(a["applied_gripper"], col)
    if src.ref_feedback is not None:
        eq = [_feedback_equal(x, y) for x, y in zip(res["feedback"], src.ref_feedback)]
        out["feedback_requests_equal"] = int(sum(eq))
        out["feedback_first_mismatch"] = next((i for i, e in enumerate(eq) if not e), None)
    if latches and "record" in latches and src.ref_grasp is not None:
        rec_state = [str((g or {}).get("state")) for g in src.ref_grasp]
        got = [str(x) for x in latches["record"]["phase"]]
        same = [x == y for x, y in zip(got, rec_state)]
        out["latch_record_equal"] = int(sum(same))
        out["latch_record_first_mismatch"] = next((i for i, e in enumerate(same) if not e), None)
        out["latch_record_states"] = {"offline": got, "recorded": rec_state}
    keys0 = ["qpos_max_abs", "qvel_max_abs", "state_max_abs", "image_max_abs"]
    exact = all(out[k] == 0 for k in keys0)
    if "measured_max_abs" in out:
        exact &= out["measured_max_abs"] == 0
    if "applied_gripper_max_abs" in out:
        exact &= out["applied_gripper_max_abs"] == 0
    if "feedback_requests_equal" in out:
        exact &= out["feedback_requests_equal"] == F
    if "latch_record_equal" in out:
        exact &= out["latch_record_equal"] == F
    out["exact"] = bool(exact)
    return out


# ------------------------------------------------------------------------------ one record


def variant_payload(src: Source, res: dict, latches: dict, *, tag: str, window_start: Optional[int],
                    meta: dict) -> dict[str, np.ndarray]:
    a = dict(res["arrays"])
    for mode, lt in latches.items():
        for k, v in lt.items():
            a[f"latch_{mode}__{k}"] = v
    fb = res["feedback"]
    a["fb_available"] = np.asarray([bool(f.get("available")) for f in fb])
    a["window_start"] = np.asarray(-1 if window_start is None else int(window_start), np.int64)
    a["meta"] = np.asarray(json.dumps(jsonable(meta), sort_keys=True))
    a["tag"] = np.asarray(tag)
    a["key"] = np.asarray(src.key)
    a["target"] = np.asarray(src.target)
    a["used_arm"] = np.asarray(src.used_arm)
    a["prompt"] = np.asarray(src.prompt)
    a["episode"] = np.asarray(int(src.episode), np.int64)
    return a


def pre_intervention_latch_parity(e0_latch: dict, v_latch: dict, frame: int) -> dict:
    out = {}
    for mode in sorted(set(e0_latch) & set(v_latch)):
        a = [str(x) for x in e0_latch[mode]["phase"][: frame + 1]]
        b = [str(x) for x in v_latch[mode]["phase"][: frame + 1]]
        out[mode] = {"requests": frame + 1, "equal": int(sum(x == y for x, y in zip(a, b))),
                     "states_e0": a}
    return out


def slip_baseline(e0_p_task_raw: Sequence[bool], after_frames: int) -> dict:
    """Does the unintervened re-sim reach a hold (raw P_task for `after_frames` consecutive frames)?"""
    fh = hold_onset_frame(e0_p_task_raw, after_frames)
    out = {"after_frames": int(after_frames), "e0_hold_frame": fh, "excluded": fh is None}
    if fh is None:
        out["reason"] = SLIP_EXCLUSION_REASON
    return out


def process_record(src: Source, variants: Sequence, *, client_path: pathlib.Path, out_dir: pathlib.Path,
                   force: bool, keep_images: bool, run_meta: dict) -> dict:
    """E0 (always) + the requested interventions of one record. Returns the summary row.

    Baseline (`BASELINE_OF_SET`): dev -> the unintervened re-sim (interventions run whatever the E0 parity;
    the parity numbers are reported); test -> the record (interventions refused unless E0 is exact)."""
    from AG3S.trajopt.grasp_latch import LatchConfig

    baseline = BASELINE_OF_SET[src.set_name]
    cl = load_client(client_path)
    H = int(cl.OPEN_LOOP_HORIZON)
    closed_below = float(LatchConfig().gripper_closed_below)
    base_meta = {**run_meta, "source": src.describe(), "client": str(client_path), "client_md5": md5_of(client_path),
                 "version": RESIM_VERSION, "created": now_kst(), "baseline": baseline}
    summary: dict = {"key": src.key, "baseline": baseline, "variants": {}}
    p0 = out_dir / f"{src.key}__E0.npz"
    res0 = simulate(cl, src, keep_images=keep_images)
    lat0 = run_latches(src, res0)
    par = e0_parity(src, res0, lat0)
    raw0 = raw_preds(res0["arrays"], src.target, src.used_arm)
    slips = sorted({int(iv.after_frames) for iv in variants if iv.kind == "slip"})
    sb = {a: slip_baseline(raw0["p_task"], a) for a in slips}
    meta0 = {**base_meta, "variant": "E0", "parity": par, "seconds": res0["seconds"],
             "e0_physical_hold": {"p_task_raw_frames": int(np.sum(raw0["p_task"])),
                                  "p_grasp_raw_frames": int(np.sum(raw0["p_grasp"]))},
             "slip_baseline": {str(a): v for a, v in sb.items()}}
    if p0.exists() and not force:
        summary["variants"]["E0"] = {"cached": True}
    else:
        save_npz_atomic(p0, **variant_payload(src, res0, lat0, tag="E0", window_start=None, meta=meta0))
        write_json(p0.with_suffix(".json"), meta0)
    summary["e0_parity"] = {k: v for k, v in par.items() if k not in ("per_frame", "latch_record_states")}
    summary["variants"]["E0"] = {"seconds": res0["seconds"], "exact": par["exact"]}
    summary["slip_excluded"] = {}
    for iv in variants:
        tag = iv.tag
        p = out_dir / f"{src.key}__{tag}.npz"
        if iv.offset_steps >= H:
            raise ValueError(f"{tag}: offset_steps {iv.offset_steps} must be < {H} (one chunk)")
        if baseline == "record" and not par["exact"]:
            summary["variants"][tag] = {"refused": "test E0 not exact — STOP and report (no qpos overwrite)"}
            continue
        if iv.kind == "slip" and sb[iv.after_frames]["excluded"]:
            why = sb[iv.after_frames]["reason"]
            summary["variants"][tag] = {"not_applicable": why, "excluded_from_slip_evaluation": True}
            summary["slip_excluded"][tag] = why
            write_json(out_dir / f"{src.key}__{tag}.json", {
                **base_meta, "variant": tag, "not_applicable": why, "excluded_from_slip_evaluation": True,
                "slip_baseline": sb[iv.after_frames], "intervention": iv.params()})
            continue
        f, why = intervention_frame(iv, src, raw0["p_task"], horizon=H, closed_below=closed_below)
        if f is None:
            summary["variants"][tag] = {"not_applicable": why}
            write_json(out_dir / f"{src.key}__{tag}.json", {**base_meta, "variant": tag, "not_applicable": why,
                                                           "intervention": iv.params()})
            continue
        if p.exists() and not force:
            summary["variants"][tag] = {"cached": True}
            continue
        res = simulate(cl, src, intervention=iv, frame=f, keep_images=keep_images)
        lat = run_latches(src, res)
        raw = raw_preds(res["arrays"], src.target, src.used_arm)
        ws = f + 1
        info = {"frame": f, "why": why, "window_start": ws, **res["intervention"],
                "pre_latch_parity": pre_intervention_latch_parity(lat0, lat, f),
                "pre_qpos_equal": bool(np.array_equal(res["arrays"]["qpos"][: f + 1], res0["arrays"]["qpos"][: f + 1])),
                "post_p_grasp_raw_frames": int(raw["p_grasp"][ws:].sum()),
                "post_p_task_raw_frames": int(raw["p_task"][ws:].sum()),
                "post_penetration_max_m": float(np.max(res["arrays"]["penetration"][ws:])) if ws < len(raw["p_task"]) else None}
        meta = {**base_meta, "variant": tag, "intervention": iv.params(), "info": info, "seconds": res["seconds"],
                "e0_exact": par["exact"]}
        if iv.kind == "slip":
            meta["slip_baseline"] = sb[iv.after_frames]
        save_npz_atomic(p, **variant_payload(src, res, lat, tag=tag, window_start=ws, meta=meta))
        write_json(p.with_suffix(".json"), meta)
        summary["variants"][tag] = {k: v for k, v in info.items() if k != "pre_latch_parity"} | {
            "pre_latch_equal": {m: v["equal"] == v["requests"] for m, v in info["pre_latch_parity"].items()},
            "seconds": res["seconds"]}
    return summary


def _worker(job: dict) -> dict:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    src = job["source"]
    try:
        return process_record(src, job["variants"], client_path=job["client"], out_dir=job["out_dir"],
                              force=job["force"], keep_images=job["keep_images"], run_meta=job["run_meta"])
    except Exception as exc:  # noqa: BLE001 — one record's failure is reported, others continue
        import traceback

        return {"key": src.key, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-3000:]}


# ------------------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", required=True, choices=SETS)
    ap.add_argument("--records", nargs="+", default=None, help="subset (dev keys / T53 keys)")
    ap.add_argument("--t53-run-dir", default=None, help="the T53 run (read-only): test records + client snapshot")
    ap.add_argument("--episodes-root", default=None, help="T53 episode dirs (default <run>/verify/T53)")
    ap.add_argument("--server-roots", nargs="*", default=[], help="T53 server record roots (record-mode latch)")
    ap.add_argument("--client", default=None, help="client file (default: T53 snapshot copy, else checkout)")
    ap.add_argument("--variants", nargs="+", default=["E0"], choices=VARIANT_KINDS)
    ap.add_argument("--empty-lead-frames", type=int, default=0)
    ap.add_argument("--empty-offset-steps", type=int, default=0)
    ap.add_argument("--empty-method", default="park", choices=EMPTY_METHODS)
    ap.add_argument("--slip-after-frames", type=int, default=2)
    ap.add_argument("--slip-delay-frames", type=int, default=0)
    ap.add_argument("--slip-offset-steps", type=int, default=0)
    ap.add_argument("--slip-method", default="ghost", choices=SLIP_METHODS)
    ap.add_argument("--ghost-clear-mm", type=float, default=5.0)
    ap.add_argument("--ghost-max-steps", type=int, default=30)
    ap.add_argument("--no-images", action="store_true", help="E0 parity only, no images kept (smoke)")
    ap.add_argument("--workers", type=int, default=1, help="processes (<= 4)")
    ap.add_argument("--force", action="store_true")
    return ap


def interventions_from_args(args) -> list[Intervention]:
    out = []
    if "empty" in args.variants:
        out.append(Intervention("empty", args.empty_method, lead_frames=args.empty_lead_frames,
                                offset_steps=args.empty_offset_steps))
    if "slip" in args.variants:
        out.append(Intervention("slip", args.slip_method, after_frames=args.slip_after_frames,
                                delay_frames=args.slip_delay_frames, offset_steps=args.slip_offset_steps,
                                ghost_clear_m=args.ghost_clear_mm / 1000.0, ghost_max_steps=args.ghost_max_steps))
    return out


def sources_from_args(args) -> list[Source]:
    meta_ds = dataset_meta()
    if args.set == "dev":
        keys = args.records or list(DEV_KEYS)
        return [dev_source(k, meta_ds=meta_ds) for k in keys]
    if not args.t53_run_dir:
        raise SystemExit("--set test needs --t53-run-dir")
    from AG3S.ag3s.experiments.studies import t53_records as rc

    keys = rc.resolve_keys(args.records)
    return [t53_source(k, t53_run_dir=pathlib.Path(args.t53_run_dir),
                       episodes_root=None if args.episodes_root is None else pathlib.Path(args.episodes_root),
                       server_roots=[pathlib.Path(p) for p in args.server_roots], meta_ds=meta_ds) for k in keys]


def main(argv: Optional[Sequence[str]] = None) -> dict:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not 1 <= args.workers <= 4:
        ap.error("--workers in 1..4")
    forbid = [pathlib.Path(args.t53_run_dir)] if args.t53_run_dir else []
    out_dir = out_root(f"resim/{args.set}", forbid=forbid)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = pathlib.Path(args.client) if args.client else default_client(
        pathlib.Path(args.t53_run_dir) if args.t53_run_dir else None)
    srcs = sources_from_args(args)
    ivs = interventions_from_args(args)
    run_meta = {"step": STEP, "argv": sys.argv, "args": vars(args), "python": sys.executable,
                "mujoco_gl": os.environ.get("MUJOCO_GL"), "omp_threads": os.environ.get("OMP_NUM_THREADS")}
    jobs = [{"source": s, "variants": ivs, "client": client, "out_dir": out_dir, "force": args.force,
             "keep_images": not args.no_images, "run_meta": run_meta} for s in srcs]
    t0 = time.time()
    if args.workers == 1:
        rows = [_worker(j) for j in jobs]
    else:
        import concurrent.futures as cf
        import multiprocessing as mp

        with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
            rows = list(ex.map(_worker, jobs))
    for r in rows:
        status = r.get("error") or {k: ("exact" if v.get("exact") else
                                        v.get("refused") or v.get("not_applicable") or "ok")
                                    for k, v in r.get("variants", {}).items()}
        print(f"[resim {args.set}] {r['key']}: {status}", flush=True)
    summary = {**run_meta, "created": now_kst(), "client": str(client), "client_md5": md5_of(client),
               "baseline": BASELINE_OF_SET[args.set], "interventions": [iv.params() for iv in ivs],
               "slip_excluded": {r["key"]: r["slip_excluded"] for r in rows if r.get("slip_excluded")},
               "records": rows, "wall_s": time.time() - t0}
    write_json(out_dir / f"summary_{'_'.join(args.variants)}_{time.strftime('%Y%m%dT%H%M%S')}.json", summary)
    return summary


__all__ = ["BASELINE_OF_SET", "DEV_KEYS", "Intervention", "Source", "client_reset", "default_client", "dev_source", "e0_parity",
           "first_closing_frame", "hand_geoms", "held_by", "hold_onset_frame", "intervention_frame", "load_client",
           "main", "offline_latch", "perception_from_record", "perception_oracle", "process_record", "raw_preds",
           "run_latches", "simulate", "slip_baseline", "t53_source"]


if __name__ == "__main__":
    main()
