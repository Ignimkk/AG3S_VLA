"""``serve_safe --robot panda`` — the V11AB server for the LIBERO Panda (SL5a).

Same ``SafePolicy`` (V11AB objective, constraints, gate, latch) with the Panda injected through its optional
arguments; the RB-Y1 path of ``serve_safe.main`` is untouched (``--robot rby1`` never reaches this module).

| mode | served object | what the client gets |
|---|---|---|
| ``--no-safe`` (VLA-only) | ``PandaVLAOnly(SeededPolicy(policy))`` | pi05_libero's own EE-delta chunk |
| default (VLA + V11AB) | ``PandaSafeServer(SafePolicy(ChunkToJointPolicy(AttentionPolicy(policy))))`` | EE-delta chunk after V11AB (``action.py``) + V11AB response keys |

Both modes take the **same request** from ``benchmark/runners/libero_safe_client.py`` (official LIBERO policy keys
+ ``ag3s/*`` camera geometry + ``panda/*`` robot state) and both strip everything but the official keys before
pi05_libero sees the observation, so with the same ``policy_seed`` and the same observations the policy's chunks
are the same in both modes.

Panda differences from the RB-Y1 server (all recorded in ``panda_server_config.json`` in the run's ``logs/``):

| what | V11AB (RB-Y1) | Panda | reason |
|---|---|---|---|
| AG3S | ``build_ag3s`` RB-Y1 | ``profile.build_panda_ag3s`` (SL4) | robot geometry; config diff = contact links + ESDF z |
| chunk | 16-D absolute joints | ``[q7, g]`` from EE deltas (``action.py``) | pi05_libero outputs OSC deltas |
| horizon | H 50, K 8, 15 Hz, plan 16 | H 10, K 5, 20 Hz, plan 10 | pi05_libero chunk 10, client replans every 5, LIBERO 20 Hz; plan = 2K as V11AB |
| joint limits | RB-Y1 MJCF | adapter URDF (Franka) + acceleration 7.5 rad/s² | Panda limits; URDF has no acceleration (min of the Franka datasheet values) |
| attention | ``cam_high``→``zed_left`` … | ``cam_high``→``head``, ``cam_left_wrist``→``left_wrist``, un-mirror | SL1/SL3 |
| subtask probe | ``kv_L4_v1`` (RB-Y1) | none | RB-Y1-trained; the gate is inert without a label |
| gripper | 2 hands, RB-Y1 client convention | 1 hand ``left``, ``panda_finger_joint1 / 0.04`` | Panda |
| static geometry | ``--static-geometry`` (default none) | none | RB-Y1 shapes |

``--record-constraints DIR`` (SL5j): the same per-call constraint record as the RB-Y1 server
(``ConstraintRecordWriter(background=True)``, ``AG3S/ag3s/experiments/sources/constraint_record.py``), passed to
``SafePolicy(recorder=...)`` by :func:`build_safe_server` (:func:`constraint_recorder`, meta in
:func:`recorder_meta`). :func:`main` closes it when ``serve_forever`` ends -- normally, on SIGINT (the launcher's
``stop_own_process``) and on SIGTERM (``serve_safe.exit_cleanly_on_sigterm``) -- so the queued chunks are written
and ``meta.json`` exists. Without the flag the recorder is ``None`` and nothing else changes.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import pathlib
from typing import Any, Optional

import numpy as np

from AG3S.adapters.panda import action as pa
from AG3S.adapters.panda import model as pm

#: pi05_libero / LIBERO timing (SL0: chunk 10, replan 5, ``ControlEnv`` control_freq 20).
PANDA_HORIZON = {"horizon": 10, "execution_length": 5, "control_hz": 20.0, "plan_horizon": 10}
#: Franka Panda joint acceleration limits (rad/s²) per the Franka datasheet: 15, 7.5, 10, 12.5, 15, 20, 20.
#: ``LimitsConfig.default_acceleration`` is one scalar, so the smallest is used (conservative).
FRANKA_ACCEL_LIMITS = (15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0)
PANDA_DEFAULT_ACCELERATION = min(FRANKA_ACCEL_LIMITS)
ATTENTION_ALIAS = {"cam_high": "head", "cam_left_wrist": "left_wrist"}
#: Reverse-conversion mode of this server (``action.CONVERSION_MODES``): env ``AG3S_PANDA_CONVERSION`` (A | B).
ENV_CONVERSION = "AG3S_PANDA_CONVERSION"
GRIPPER_CALIBRATION = pm.DESCRIPTION_DIR / "gripper_calibration.json"
#: SL11 §5: thresholds of the latch when its opening is the two-finger gap (read only with ``--gripper-gap-opening``).
GRIPPER_GAP_CALIBRATION = pm.DESCRIPTION_DIR / "gripper_gap_calibration.json"
GRIPPER_GAP_FULL_M = 0.08      # both fingers fully open: 2 x 0.04 m


def conversion_mode() -> str:
    mode = os.environ.get(ENV_CONVERSION, "A").strip().upper()
    if mode not in pa.CONVERSION_MODES:
        raise ValueError(f"{ENV_CONVERSION}={mode!r}: expected one of {sorted(pa.CONVERSION_MODES)}")
    return mode


def gripper_calibration() -> dict:
    return json.loads(GRIPPER_CALIBRATION.read_text())


def gripper_gap_calibration() -> dict:
    return json.loads(GRIPPER_GAP_CALIBRATION.read_text())


def panda_latch_config(gap_opening: bool = False):
    """V11AB ``LatchConfig`` defaults with only ``empty_below`` calibrated for the Panda gripper.

    SL11 §5 ``gap_opening``: the latch's opening is the two-finger gap (:func:`gripper_gap_norm`), so the three thresholds that
    read a *closed on nothing / blocked by something* off the opening are re-fitted to that scale
    (``gripper_gap_calibration.json``): ``empty_below``, ``blocked_gap`` and ``hold_empty_below``, and the HSR6 attach-assist also needs
    the reach check (``assist_requires_reach``). Everything else is unchanged.
    """
    from AG3S.trajopt.grasp_latch import LatchConfig

    if not gap_opening:
        return LatchConfig(empty_below=float(gripper_calibration()["panda"]["empty_below"]))
    c = gripper_gap_calibration()["thresholds"]
    return LatchConfig(empty_below=float(c["empty_below"]), blocked_gap=float(c["blocked_gap"]),
                       hold_empty_below=float(c["hold_empty_below"]), assist_requires_reach=bool(c["assist_requires_reach"]))
ATTENTION_CAMERAS = ("head", "left_wrist")
HANDS = {pm.MANIPULATOR_SIDE: 0}
GRIPPER_COLUMNS = (pa.GRIPPER_COLUMN,)
#: request keys only the Panda adapter reads (never reach the policy)
PANDA_PREFIX = "panda/"
KEY_ROBOT_STATE = "panda/robot_state"      # (9,) q7 + 2 MuJoCo finger qpos, image instant
KEY_T_WORLD_BASE = "panda/T_world_base"    # (4, 4)
KEY_ACTION_ADAPTER = pa.ACTION_ADAPTER_KEY     # SL11: the LIBERO-Safety client's action adapter (absent from other clients)
KEY_PREV_STEP = "panda/prev_step_world"    # (6,) last realised EE step, world frame: dp (m), drotvec (rad)


def panda_gripper_convention():
    from AG3S.ag3s.robot_models.gripper_state import ClientGripperConvention

    return ClientGripperConvention(open_command_m=0.04, driven_joints={pm.MANIPULATOR_SIDE: "panda_finger_joint1"},
                                   source="AG3S/adapters/panda/server.py: LIBERO Panda finger joint1 open limit 0.04 m "
                                          "(adapter URDF); normalised opening = q / 0.04")


def check_response_adapter(response_name: str, adapter) -> None:
    """SL11 -- the fork response model is expressed for the LIBERO-scale action the LIBERO-Safety client rescales with
    ``libero_to_safety``; on any other client / adapter its beta would be wrong, so the request is refused (not guessed).
    The default model has no requirement (``adapter`` is not looked at: those requests are unchanged)."""
    if response_name == "robosuite141":
        return
    adapter = None if adapter is None else str(np.asarray(adapter).item() if hasattr(adapter, "shape") else adapter)
    if adapter != pa.FORK_REQUIRED_ADAPTER:
        raise ValueError(f"--response-model {response_name} needs the LIBERO-Safety client with action_adapter "
                         f"{pa.FORK_REQUIRED_ADAPTER!r} (request {KEY_ACTION_ADAPTER}={adapter!r}); refusing to convert with a "
                         "model that does not describe the env")


def strip_panda_keys(obs: dict) -> tuple[dict, dict]:
    keep = {k: v for k, v in obs.items() if not k.startswith(PANDA_PREFIX)}
    panda = {k: v for k, v in obs.items() if k.startswith(PANDA_PREFIX)}
    return keep, panda


#: SafePolicy reads the capture-time gripper opening from the policy observation's ``state`` at
#: ``gripper_columns`` (T29 convention, 1 = open); the Panda request carries it in ``panda/robot_state``.
STATE_KEY = "state"


def gripper_gap_norm(robot_state9) -> float:
    """SL11 §5: ``(q1 - q2) / 0.08`` clipped to [0, 1] -- the gap between the two finger pads over its full width (both
    MuJoCo finger joints, ``q2`` is negative when open). Unlike ``|q1| / 0.04`` it does not read two fingers pushed to
    one side together (``q1 = 0, q2 = -0.02``) as a nearly closed hand."""
    rs = np.asarray(robot_state9, np.float64).reshape(-1)
    if rs.shape[0] != len(pm.JOINT_MAP):
        raise ValueError(f"Panda robot_state must have {len(pm.JOINT_MAP)} entries, got {rs.shape[0]}")
    gap = float(rs[7]) - float(rs[8])      # MuJoCo qpos: finger_joint1 in [0, 0.04], finger_joint2 in [-0.04, 0]
    return float(min(1.0, max(0.0, gap / GRIPPER_GAP_FULL_M)))


def gripper_opening_norm(robot_state9) -> float:
    """``|panda_finger_joint1| / 0.04`` from the adapter's 9-D robot_state (q7 + 2 MuJoCo finger qpos)."""
    _q, fingers = pm.split_robot_state(robot_state9)
    return abs(float(fingers["panda_finger_joint1"])) / 0.04


def adapt_request(obs: dict) -> dict:
    """V11AB wire for the Panda: ``ag3s/robot_state/<cam>`` 9-D -> 7 arm joints (the model's nq), and
    ``state`` = ``[q7, opening]`` so ``SafePolicy`` reads the capture-time opening at column 7."""
    from AG3S.trajopt import wire

    out = dict(obs)
    prefix = wire.PREFIX + "robot_state/"
    for key in [k for k in out if k.startswith(prefix)]:
        rs = np.asarray(out[key], np.float64).reshape(-1)
        if rs.shape[0] == len(pm.JOINT_MAP):
            out[key], _ = pm.split_robot_state(rs)
    if KEY_ROBOT_STATE in out:
        q7, _ = pm.split_robot_state(out[KEY_ROBOT_STATE])
        out[STATE_KEY] = np.concatenate([q7, [gripper_opening_norm(out[KEY_ROBOT_STATE])]])
    return out


class PandaVLAOnly:
    """VLA-only: drop ``ag3s/*`` and ``panda/*``; move ``ag3s/policy_seed`` to ``policy_seed`` for ``SeededPolicy``."""

    def __init__(self, policy):
        self._policy = policy

    @property
    def metadata(self):
        return getattr(self._policy, "metadata", {})

    def reset(self):
        r = getattr(self._policy, "reset", None)
        if callable(r):
            r()

    def infer(self, obs: dict, **kwargs) -> dict:
        from AG3S.trajopt import wire

        clean = {k: v for k, v in obs.items() if not k.startswith(wire.PREFIX) and not k.startswith(PANDA_PREFIX)}
        seed_key = wire.PREFIX + wire.POLICY_SEED
        if seed_key in obs:
            clean[wire.POLICY_SEED] = obs[seed_key]
        result = dict(self._policy.infer(clean, **kwargs))
        result["panda/mode"] = "vla_only"
        return result


class ChunkToJointPolicy:
    """Inside ``SafePolicy``: pops ``panda/*``, calls the policy, turns its EE-delta chunk into ``[q7, g]``."""

    def __init__(self, policy, kin: pa.PandaKinematics, response: Optional[dict] = None, response_name: str = "robosuite141"):
        self._policy = policy
        self.kin = kin
        self.response = response              # SL11: None = ``action.RESPONSE`` (robosuite 1.4.1), the earlier behaviour
        self.response_name = response_name
        self.last: Optional[pa.Conversion] = None
        self.last_result_actions = None

    @property
    def metadata(self):
        return getattr(self._policy, "metadata", {})

    def reset(self):
        r = getattr(self._policy, "reset", None)
        if callable(r):
            r()

    def infer(self, obs: dict, **kwargs) -> dict:
        obs, panda = strip_panda_keys(obs)
        obs.pop(STATE_KEY, None)  # added by PandaSafeServer for SafePolicy's measured opening; not a policy key
        if KEY_ROBOT_STATE not in panda or KEY_T_WORLD_BASE not in panda:
            raise ValueError(f"the Panda request needs {KEY_ROBOT_STATE} and {KEY_T_WORLD_BASE}")
        q_now, _fingers = pm.split_robot_state(panda[KEY_ROBOT_STATE])
        check_response_adapter(self.response_name, panda.get(KEY_ACTION_ADAPTER))
        result = dict(self._policy.infer(obs, **kwargs))
        delta = np.asarray(result["actions"])
        prev = panda.get(KEY_PREV_STEP)
        prev = None if prev is None else (np.asarray(prev, np.float64)[:3], np.asarray(prev, np.float64)[3:6])
        self.last = pa.delta_chunk_to_joint_chunk(self.kin, delta, q_now, np.asarray(panda[KEY_T_WORLD_BASE]),
                                                  prev_step_world=prev, response=self.response)
        self.last_result_actions = delta
        result["actions"] = self.last.joint_chunk
        return result


class PandaSafeServer:
    """Outside ``SafePolicy``: refined joint chunk -> EE-delta chunk (``action.joint_chunk_to_delta_chunk``)."""

    def __init__(self, safe, chunk_policy: ChunkToJointPolicy, mode: str = "A", observe_only: bool = False,
                 gap_opening: bool = False):
        self._policy = safe
        self.chunk_policy = chunk_policy
        self.mode = mode
        #: SL11 ``--observe-only`` (diagnostic): everything is computed and recorded as without it, but the response the
        #: client executes is the policy's own chunk, bit for bit, with ``safe`` True (never a HOLD). Off: the code below
        #: is not reached and the response is the one of the earlier versions.
        self.observe_only = bool(observe_only)
        #: SL11 §5 ``--gripper-gap-opening``: the exec feedback's ``measured_gripper`` (what the grasp latch reads as the measured
        #: opening) is the two-finger gap of this request's ``panda/robot_state`` instead of ``|q1| / 0.04``. Off: not touched.
        self.gap_opening = bool(gap_opening)
        self._sent = {}  # seq -> (Conversion, refined, reference) of the chunks sent, for the next exec_feedback

    @property
    def metadata(self):
        return getattr(self._policy, "metadata", {})

    @property
    def recorder(self):
        """``SafePolicy``'s ``ConstraintRecordWriter`` (``None`` without ``--record-constraints``)."""
        return getattr(self._policy, "recorder", None)

    def reset(self):
        self._sent.clear()
        self._policy.reset()

    def _feedback(self, obs: dict) -> dict:
        """``exec_feedback.applied_arm`` from the OSC rows the client actually applied (``panda/exec_applied_osc``).

        The client confirms the feedback after the rows ran (``n_steps`` = steps actually executed). Here the
        applied OSC commands become joint command rows with the conversion of the request that sent them
        (``action.applied_command_rows``). Without that conversion or those rows the feedback is withdrawn
        (``available=False`` with the reason) rather than filled with a substitute.
        """
        osc = obs.pop("panda/exec_applied_osc", None)
        fb = obs.get("ag3s/exec_feedback")
        if not isinstance(fb, dict) or not fb.get("available"):
            return {"used": False}
        seq = int(fb["seq"])
        sent = self._sent.get(seq)
        why = None
        if sent is None:
            why = f"no chunk sent for feedback seq {seq} in this server session"
        elif osc is None:
            why = "client sent no panda/exec_applied_osc"
        elif np.asarray(osc).shape[0] != int(fb.get("n_steps", -1)):
            why = f"panda/exec_applied_osc has {np.asarray(osc).shape[0]} rows, n_steps={fb.get('n_steps')}"
        if why is not None:
            obs["ag3s/exec_feedback"] = {"available": False, "reason": f"panda adapter: {why}"}
            return {"used": False, "withdrawn": why}
        conv, refined, reference = sent
        arm = pa.applied_command_rows(self.chunk_policy.kin, conv, np.asarray(osc, np.float64))
        obs["ag3s/exec_feedback"] = dict(fb, applied_arm=arm.astype(np.float32))
        n = arm.shape[0]
        rec = {"used": True, "seq": seq, "n_steps": n, "n_exec": int(fb.get("n_exec", 0)),
               "executed_chunk": fb.get("executed_chunk")}
        if n:
            rec["applied_minus_refined_rad_max"] = float(np.abs(arm - np.asarray(refined, np.float64)[:n, :pa.ARM_DOF]).max())
            rec["applied_minus_reference_rad_max"] = float(np.abs(arm - np.asarray(reference, np.float64)[:n, :pa.ARM_DOF]).max())
            rec["applied_equals_reference_rows"] = int(sum(np.array_equal(arm[k].astype(np.float32),
                                                                           np.asarray(reference, np.float32)[k, :pa.ARM_DOF])
                                                           for k in range(n)))
        return rec

    @staticmethod
    def _gap_feedback(obs: dict, fb_rec: dict) -> dict:
        """SL11 §5 -- replace ``measured_gripper`` of the available exec feedback by the two-finger gap opening of this request
        (the capture instant the feedback is confirmed at). The single-finger value it had stays in the record."""
        fb = obs.get("ag3s/exec_feedback")
        if KEY_ROBOT_STATE not in obs or not isinstance(fb, dict) or not fb.get("available"):
            return fb_rec
        rs = np.asarray(obs[KEY_ROBOT_STATE], np.float64).reshape(-1)
        old = np.asarray(fb["measured_gripper"], np.float64).reshape(-1)
        new = old.copy()
        new[0] = gripper_gap_norm(rs)
        obs["ag3s/exec_feedback"] = dict(fb, measured_gripper=new.astype(np.float32))
        return {**fb_rec, "gripper_gap_opening": {"single_finger": float(old[0]), "gap": float(new[0]),
                                                  "q1": float(rs[7]), "q2": float(rs[8])}}

    def _g1_deferred(self) -> bool:
        """SL6-A A8 — did the geometric grasp (``SafePolicy._geo``) defer the close on the chunk just refined."""
        geo = getattr(self._policy, "_geo", None)
        if geo is None:
            return False
        gate = (getattr(geo, "last", None) or {}).get("close_gate") or {}
        return bool(gate.get("applied")) and gate.get("decision") == "defer"

    def infer(self, obs: dict, **kwargs) -> dict:
        obs = dict(obs)
        if obs.get("ag3s/reset"):
            self._sent.clear()  # a new episode: seq restarts, earlier chunks are not this episode's
        fb_rec = self._feedback(obs)
        if self.gap_opening:
            fb_rec = self._gap_feedback(obs, fb_rec)
        response = dict(self._policy.infer(adapt_request(obs), **kwargs))
        conv = self.chunk_policy.last
        refined, reference = response["actions"], response.get("actions_reference")
        if conv is None or reference is None:
            raise RuntimeError("SafePolicy returned no reference chunk for the Panda conversion")
        actions, rec = pa.joint_chunk_to_delta_chunk(self.chunk_policy.kin, conv, refined, reference, mode=self.mode)
        deferred = self._g1_deferred()
        if deferred:
            # SL6-A A8 — G1 deferred the close on this chunk: send the refined (open) gripper rows (G1 off: never here)
            actions, rec["g1_deferred_close"] = pa.apply_deferred_close(actions, refined, reference)
        real = pa.realisation_record(self.chunk_policy.kin, conv, actions, np.asarray(refined, np.float64),
                                     PANDA_HORIZON["execution_length"])
        # response-model EE positions (robot base frame) for the executed rows of the actions sent
        response["panda/predicted_ee_base"] = np.asarray(real.pop("p_hat_base"), np.float32)
        rec["realisation"] = real
        response["panda/joint_refined"] = np.asarray(refined, np.float32)
        response["panda/joint_reference"] = np.asarray(reference, np.float32)
        response["panda/actions_policy"] = np.asarray(conv.delta)
        response["panda/conversion"] = {**conv.record(), **rec}
        response["panda/mode"] = "vla_v11ab"
        response["actions"] = actions
        response["panda/feedback"] = fb_rec
        if "ag3s/seq" in obs:
            self._sent = {k: v for k, v in self._sent.items() if k > int(obs["ag3s/seq"]) - 3}
            self._sent[int(obs["ag3s/seq"])] = (conv, np.asarray(refined, np.float32), np.asarray(reference, np.float32))
        # logging only (read, not changed): the grasp-latch decision and the gripper value it saw, per chunk
        from AG3S.trajopt.safe_policy import _jsonable

        response["panda/grasp"] = _jsonable(getattr(self._policy, "_last_grasp_record", None) or {})
        response["panda/latch_signal"] = _jsonable(getattr(self._policy, "_last_latch_signal", None) or {})
        hsr6 = hsr6_call_record(self._policy)
        if hsr6 is not None:   # SL6-H H2 — key only with an HSR6 use flag on (V11AB / pb responses unchanged)
            response["panda/hsr6"] = hsr6
        oracle = oracle_call_record(self._policy)
        if oracle is not None:    # SL12 -- key only with --target-extent oracle-visible (every other response unchanged)
            response["panda/oracle"] = oracle
        surface = surface_call_record(self._policy)
        if surface is not None:   # SL7i — key only with the attention-surface extent on (V11AB / pb / pc responses unchanged)
            response["panda/surface"] = surface
        if self.observe_only:     # SL11 — key and the policy chunk only with --observe-only
            self._observe_only_response(response, conv, actions)
        return response

    @staticmethod
    def _observe_only_response(response: dict, conv: pa.Conversion, would_be: np.ndarray) -> None:
        """SL11 ``--observe-only``: replace the executable part of ``response`` by the policy's own chunk.

        ``actions`` becomes ``conv.delta`` (the chunk as received, its dtype, bit for bit) and ``safe`` True, so the client
        executes the policy and never holds. What the server would have sent -- the converted actions and the verdict --
        stays in the record: ``panda/would_be_actions`` (the converted chunk), ``panda/observe`` (``would_be_safe`` /
        ``would_be_hold``, the rows that differ from the policy, the largest difference). The other verdict keys
        (``verdict_reasons``, ``trajopt_status``, ``max_violation_m``, ``notes``) are the server's own, unchanged."""
        policy = np.array(conv.delta, copy=True)
        would = np.asarray(would_be)
        diff = np.abs(would.astype(np.float64) - policy.astype(np.float64))
        would_safe = bool(response.get("safe", True))
        response["panda/would_be_actions"] = np.array(would, copy=True)
        response["panda/observe"] = {"observe_only": True, "would_be_safe": would_safe, "would_be_hold": not would_safe,
                                     "would_be_rows_changed": [int(k) for k in np.flatnonzero(np.any(diff > 0.0, axis=1))],
                                     "would_be_max_abs_action_change": float(diff.max()) if diff.size else 0.0}
        response["actions"] = policy
        response["safe"] = True


def hsr6_call_record(safe) -> Optional[dict]:
    """SL6-H H2 — this call's HSR6 record for the client's ``*_calls.jsonl`` (logging only, read after the solve):
    ``phase`` (raw argmax), ``p`` (probabilities), ``probe``, ``flags``, ``changed`` per use flag and ``detail`` --
    with ⑤ on, ``detail.placement_contact`` = ``on`` / ``attached`` / ``applied`` / ``rows_released`` /
    ``held_rows_released`` (``SafePolicy._hsr6_placement_contact``). ``None`` when HSR6 is off (no key)."""
    if getattr(safe, "hsr6", None) is None:
        return None
    from AG3S.trajopt.safe_policy import _jsonable

    return _jsonable(getattr(safe, "_last_hsr6", None) or {})


def oracle_call_record(safe) -> Optional[dict]:
    """SL12 -- this call's visible-target-oracle record for the client's ``*_calls.jsonl`` (logging only, read after the solve):
    the scalars of ``AG3S._oracle_frame`` (segmentation present, cameras, target, memory counts) and of the target-free layer's
    ``oracle`` stats (``applied`` / ``why``, ``identity_mode``, ``pipeline_code``, ``identity_matches_gt`` = the AG3S-chosen
    manipulated object is the GT target, ``n_target_samples`` / ``n_other_samples``, ``n_band_cells``). ``None`` when the version
    does not use ``--target-extent oracle-visible`` (no key)."""
    ag3s = getattr(safe, "ag3s", None)
    esdf = getattr(getattr(ag3s, "config", None), "esdf", None)
    if ag3s is None or getattr(esdf, "target_extent", None) != "oracle_visible":
        return None
    from AG3S.trajopt.safe_policy import _jsonable

    scalars = lambda d: ({k: v for k, v in d.items() if isinstance(v, (bool, int, float, str)) or v is None}
                         if isinstance(d, dict) else {})
    frame = getattr(ag3s, "_oracle_frame", None)
    rec = scalars(frame)
    if isinstance(frame, dict):
        rec["cameras"] = list(frame.get("cameras") or [])
        if isinstance(frame.get("memory"), dict):
            rec["memory"] = scalars(frame["memory"])
    stats = getattr(getattr(ag3s, "_last_esdf_field", None), "stats", None)
    tf = stats.get("target_free") if isinstance(stats, dict) else None
    orc = tf.get("oracle") if isinstance(tf, dict) else None
    if isinstance(orc, dict):
        rec["field"] = scalars(orc)
        if isinstance(orc.get("pipeline_instance"), dict):
            rec["field"]["pipeline_instance"] = scalars(orc["pipeline_instance"])
        for k in ("volume_bbox_lo", "volume_bbox_hi"):   # SL13 (keys only with --oracle-target-volume)
            if isinstance(orc.get(k), (list, tuple)):
                rec["field"][k] = [float(v) for v in orc[k]]
    return _jsonable(rec)


def surface_call_record(safe) -> Optional[dict]:
    """SL7i — this call's attention-surface record for the client's ``*_calls.jsonl`` (logging only, read after the
    solve): the scalars of ``AG3S._surface_frame`` (``applied``, ``why``, ...) and of its ``t78`` ``region`` /
    ``target_check`` / ``hold`` / ``hull`` / ``seed_ats_gate`` parts. ``state`` = ``"surface"`` (a region was applied
    this frame) / ``"fallback"`` (no region: the version's fallback -- ball with keep-last off, nothing with keep-last +
    no-ball) / ``"none"`` (no surface record yet this episode). SL9: with the development proposals A / B on, also
    ``solid`` / ``removed_sign_free`` / ``hand_voxel_m`` (:func:`_solid_sign_free_record`). ``None`` when the version
    does not use the attention-surface extent (no key)."""
    ag3s = getattr(safe, "ag3s", None)
    esdf = getattr(getattr(ag3s, "config", None), "esdf", None)
    if ag3s is None or getattr(esdf, "target_extent", None) != "attention_surface":
        return None
    from AG3S.trajopt.safe_policy import _jsonable

    keep = lambda d: {k: v for k, v in (d or {}).items()
                      if isinstance(v, (bool, int, float, str)) or v is None} if isinstance(d, dict) else None
    surf = getattr(ag3s, "_surface_frame", None)
    if not isinstance(surf, dict):
        return {"state": "none"}
    t78 = surf.get("t78") or {}
    rec = {**keep(surf), "state": "surface" if surf.get("applied") else "fallback"}
    for part in ("region", "target_check", "hold", "hull", "seed_ats_gate"):
        if isinstance(t78.get(part), dict):
            rec[part] = keep(t78[part])
    rec.update(_solid_sign_free_record(ag3s, keep))   # SL9 -- keys only with the A / B flags (else nothing is added)
    return _jsonable(rec)


def _solid_sign_free_record(ag3s, keep) -> dict:
    """SL9 -- scalar summaries of the development proposals' records, read (logging only) from the last ESDF field's
    ``stats["target_free"]`` (``AG3S._last_esdf_field``, set by every field build): A ``solid`` = the pipeline's
    ``attention_surface.solid`` (``applied`` / ``why``, ``volume_cm3``, ``n_projected`` = the R-perp points projected on
    the support plane, ``shadow`` / ``dn_ever_target`` parts) + ``voxels`` (the builder's removed-seed counts per path,
    the tau-protected counts, ``n_removed_shadow`` and ``n_removed_shadow_by_camera``); B ``removed_sign_free`` =
    ``oracle_split.removed_sign_free`` (``applied``, ``n_flipped`` ...); ``hand_voxel_m`` = the voxel size of the
    target-free hand layer tier. ``{}`` when neither record exists (flags off, no field yet, or none this frame)."""
    stats = getattr(getattr(ag3s, "_last_esdf_field", None), "stats", None)
    tf = stats.get("target_free") if isinstance(stats, dict) else None
    if not isinstance(tf, dict):
        return {}
    ints = lambda d: ({str(k): v for k, v in d.items() if isinstance(v, (bool, int, float))}
                      if isinstance(d, dict) else None)
    out: dict = {}
    sol = (tf.get("attention_surface") or {}).get("solid") if isinstance(tf.get("attention_surface"), dict) else None
    if isinstance(sol, dict):
        s = keep(sol)
        for part in ("shadow", "dn_ever_target"):
            if isinstance(sol.get(part), dict):
                s[part] = keep(sol[part])
                if part == "shadow" and isinstance(sol[part].get("cameras"), dict):
                    s[part]["target_px_by_camera"] = ints(sol[part]["cameras"])
        vox = sol.get("voxels")
        if isinstance(vox, dict):
            s["voxels"] = keep(vox)
            if isinstance(vox.get("n_removed_shadow_by_camera"), dict):
                s["voxels"]["n_removed_shadow_by_camera"] = ints(vox["n_removed_shadow_by_camera"])
        out["solid"] = s
    split = tf.get("oracle_split")
    rsf = split.get("removed_sign_free") if isinstance(split, dict) else None
    if isinstance(rsf, dict):
        out["removed_sign_free"] = keep(rsf)
    if out:
        tiers = tf.get("tiers")
        tier = next((t for t in tiers if isinstance(t, dict) and t.get("name") == "fine"),
                    tiers[0] if tiers and isinstance(tiers[0], dict) else None) if isinstance(tiers, list) else None
        if tier is not None and isinstance(tier.get("voxel_size"), (int, float)):
            out["hand_voxel_m"] = float(tier["voxel_size"])
        seq = (tf.get("geometry_version") or {}).get("request_seq") if isinstance(tf.get("geometry_version"), dict) else None
        if seq is not None:
            out["field_request_seq"] = int(seq)
    return out


def _replace_horizon_limits(to_config):
    h = dataclasses.replace(to_config.horizon, **PANDA_HORIZON)
    lim = dataclasses.replace(to_config.limits, source="urdf", model_xml=None,
                              default_acceleration=PANDA_DEFAULT_ACCELERATION)
    return dataclasses.replace(to_config, horizon=h, limits=lim)


#: serve_safe options added by the AG3S final-configuration merge (main 3ff415d5, 2026-10-29) that the Panda server
#: has not ported: this module builds its own SafePolicy / AG3S config, so most of them would be ignored or applied in
#: part without a word. :func:`check_unported_flags` refuses any of them set away from its parser default. Allowed:
#: ``--run-dir``, ``--blas-threads`` (applied by ``serve_safe.main`` before the Panda branch) and ``--no-roi-carve``
#: (``profile.V11AB_PRE_MERGE_DEFAULTS``). Remove a name only together with its Panda port.
#: SL6-A A10 (2026-10-29): the options of the AG3S final configuration (``AG3S-CORE-EVAL-A6.impl.md`` §3) are ported
#: (:data:`PORTED_FINAL_CONFIG_DESTS`) and no longer here; every other merge option still is.
UNPORTED_FINAL_CONFIG_DESTS: tuple[str, ...] = (
    'confidence_gate', 'confidence_gate_threshold', 'free_space_carve', 'free_space_carve_margin_mm',
    'geo_arm_check', 'geo_carry_requests', 'geo_close_gate', 'geo_close_timeout', 'geo_close_tol_deg',
    'geo_close_tol_mm', 'geo_correct_deg', 'geo_correct_max_offset', 'geo_correct_mm', 'geo_grasp_mu',
    'geo_grasp_selection', 'geo_obstacle_memory', 'geo_obstacle_source', 'geo_time_budget_ms', 'geo_w_pos',
    'geo_w_rot', 'geo_wrist_check', 'grasp_anchor_radius_mm', 'grasp_goal_box', 'grasp_goal_window', 'held_3d_max_m',
    'held_3d_rule', 'held_permission_parent_links', 'hybrid_identity', 'hybrid_other_exclude_voxels',
    'latch_carry_from_closing', 'latch_reach_hand_relative', 'latch_reach_kept_unseen',
    'nonauthorized_target_margin', 'nontarget_shell_voxels', 'oracle_extent_identity', 'oracle_max_dist_voxels',
    'owned_ambiguous', 'owned_tie_voxels', 'owned_unowned', 'palm_cover', 'palm_cover_overflow',
    'palm_shell_overflow', 'profile_substages', 'qp_backend', 'recovery_retreat', 'robot_free_shrink_mm',
    'robot_mask', 'robot_occupied_free', 'roi_pad_mm', 'stem_cap_bottom_fit_frac', 'stem_cap_radius_mm',
    'stem_cap_top_mm', 'target_free_coarse', 'target_free_coarse_sign_floor',
    'target_free_coarse_support', 'target_layer', 'target_layer_reset_mm', 'target_layer_tol_mm', 'target_stem_cap',
    'task_gate', 'task_gate_shadow', 'task_probe', 'tsdf_frustum_decay', 'tsdf_time_decay', 'wrist_cover_overflow',
) + (
    # main 02fe59e6 merge (2026-11-03, user ruling): every option it adds is refused until its Panda port. SL7 ported the
    # P0m ones (attention-surface extent + its checks with the Panda head, GPU surface selection, pre-grasp hull fill,
    # contact allowance, --subtask-from-latch: see PORTED_FINAL_CONFIG_DESTS; SL7b: pass-through when clear); still
    # refused: target memory sticky, surface split / crease cut / reattach (dropped or not adopted in P0m). Refused for good:
    # RB-Y1-trained assets (hsr5, hidden probe), MuJoCo ground-truth stage, RB-Y1 gripper geometry (grasp set, grasp
    # corridor), dropped / not adopted options.
    'grasp_corridor_ahead_mm', 'grasp_corridor_allow', 'grasp_corridor_min_target_points',
    'grasp_set_budget_ms', 'grasp_set_field_only', 'grasp_set_full_horizon', 'grasp_set_gate_mm', 'grasp_set_to',
    'grasp_set_track_scale', 'hsr_gt_stage', 'hsr5', 'hsr6_objective_window',
    'support_surface_offset_mm', 'surface_crease_angle_deg', 'surface_crease_cut',
    'surface_crease_reattach_ratio', 'surface_crease_voxel_mm',
    'surface_memory_target_sticky', 'surface_reattach_hull', 'surface_split',
    'target_vacancy_delta_mm', 'target_vacancy_min_points', 'target_vacancy_p_empty',
    'target_vacancy_p_present', 'target_vacancy_release', 'target_vacancy_samples',
    # main a0b16d0a (T91-T93, 2026-11-03): refused until each STEP's Panda port (development transfer note
    # BENCHMARK_T90_T93_TRANSFER_20261103_KO.md): wrist-depth phase machine (T91c), VLA flow-matching guidance (T93,
    # needs an EE-delta -> IK Jacobian map on the Panda). The phase from the grasp latch (T92) is ported (SL7).
    'subtask_from_phase_machine', 'fm_guide', 'fm_guide_scale', 'fm_guide_steps',
    'fm_guide_max_step_rad', 'fm_guide_sweep',
) + (
    # main 4ecf8d61 (T96-T97, 2026-11-04; transfer note BENCHMARK_T96_T96E_PALM_TRANSFER_20261104_KO.md): T97 crease
    # protrusion / hand freeze is not ablated in SL9 (task SL9 §1); the real-size palm cover is not applicable on the
    # Panda (SL9 §4-1: the Panda palm shell at its 5 mm margin leaves 0 of the panda_hand mesh samples uncovered).
    'surface_crease_protrusion', 'surface_crease_protrusion_area', 'surface_crease_hand_freeze_mm',
    'palm_cover_margin',
)
#: SL6-A A10 — the final-configuration options this server now applies (A1-A9, A12); listed for the record and the test.
#: SL6-H: + ``hsr6_placement_contact`` (HSR6 ⑤, an HSR6 use flag: refused without ``--hsr6-probe`` like the others).
PORTED_FINAL_CONFIG_DESTS: tuple[str, ...] = (
    'attention_source', 'attention_split_partition', 'attention_split_radius_mm', 'attention_split_ratio_source',
    'continuity_after_grasp', 'deflection_rate_after_grasp', 'geo_continuity_off', 'geo_goal_mode', 'geo_grasp',
    'geo_track_scale', 'grasp_anchor', 'grasp_anchor_path_weight', 'hsr6_attach_assist', 'hsr6_held_gate',
    'hsr6_objective', 'hsr6_placement_contact', 'hsr6_probe', 'hsr6_target_exempt', 'kept_geometry_follow',
    'neighbour_contact_mm',
    'neighbour_palm_contact_mm', 'nontarget_evidence', 'owned_max_dist_voxels', 'palm_shell', 'palm_shell_margin',
    'placed_no_seed_certify', 'roi_carve', 'roi_carve_device', 'roi_carve_footprint', 'roi_carve_footprint_pad_px',
    'roi_carve_footprint_robot', 'split_supersedes_anchor', 'target_exclusion', 'target_permission_strict',
    'to_gpu_seeds', 'to_gpu_seeds_check', 'to_multi_seed', 'wrist_track_scale',
    # SL7 (P0m, 2026-11-03): attention-surface extent and its post-processing (`--target-extent attention-surface` only,
    # see check_unported_flags), GPU surface selection, hull fill, contact allowance, the gate label from the latch (T92)
    'attention_surface_device', 'subtask_from_latch', 'surface_contact_allowance_mm', 'surface_grasp_hold',
    'surface_hull_fill', 'surface_keep_last', 'surface_no_ball', 'surface_target_check', 'target_extent',
    # SL7b (user ruling 2026-11-04): pass-through when the VLA reference is clear (T88) -- applied through
    # serve_safe.trajopt_config_from_args (sqp.passthrough_when_clear), which this server already calls
    'passthrough_when_clear',
    # SL7f (lead self-ruling 2026-11-04, user autonomous window): ATS gate of the surface seed map (esdf key via
    # serve_safe.target_exclusion_section, like the other surface flags)
    'surface_seed_ats_gate',
    # SL9 (lead ruling 2026-11-04, task SL9): the development proposals A (target as a closed solid: T96 / T96c / T96d) and
    # B (removed-seed sign fix: T96e); esdf keys via serve_safe.target_exclusion_section like surface_seed_ats_gate
    'surface_solid_owned', 'surface_solid_protect_mm', 'surface_solid_dilate_mm', 'surface_solid_shadow_mm',
    'surface_solid_dn_ever_target', 'surface_solid_dn_support', 'surface_removed_sign_free',
)


def _hidden_probe_source(args) -> bool:
    """main 02fe59e6 merge — whether ``--attention-source`` names the hidden-probe target map (``hidden_probe[:NPZ]``)."""
    spec = getattr(args, "attention_source", None)
    return spec is not None and str(spec).strip().split(":", 1)[0] == "hidden_probe"


#: SL12: options of ``--target-extent oracle-visible`` (the oracle's identity and candidate reach). They stay in
#: :data:`UNPORTED_FINAL_CONFIG_DESTS` (refused with any other extent, attention-surface included) and are accepted only there.
SL12_ORACLE_DESTS: tuple[str, ...] = ("oracle_extent_identity", "oracle_max_dist_voxels")


def check_unported_flags(args, ap) -> None:
    """Refuse a final-configuration option the Panda server would not apply as the RB-Y1 server does, and (SL6-A)
    the ported ones in a form the Panda cannot run: an HSR6 flag without ``--hsr6-probe`` (the default probe is the
    RB-Y1 one, A10), ``--palm-shell right`` (the Panda is the one ``left`` hand)."""
    from AG3S.trajopt import serve_safe as ss

    parser = ap if ap is not None else ss.build_parser()
    oracle_ported = getattr(args, "target_extent", None) == "oracle-visible"   # SL12: the identity / reach options are read there
    bad = sorted(d for d in UNPORTED_FINAL_CONFIG_DESTS
                 if hasattr(args, d) and getattr(args, d) != parser.get_default(d)
                 and not (oracle_ported and d in SL12_ORACLE_DESTS))
    message = None
    if bad:
        message = ("--robot panda: not ported to the Panda server yet (AG3S final-configuration merge 2026-10-29): "
                   + ", ".join("--" + d.replace("_", "-") for d in bad))
    elif ss.hsr6_flags_from_args(args) is not None and not getattr(args, "hsr6_probe", None):
        message = ("--robot panda: --hsr6-* needs --hsr6-probe FILE (the Panda probe, SL6-B B4); the default probe is "
                   "the RB-Y1 one (kv_L4 of another policy and robot)")
    elif getattr(args, "target_extent", None) not in (None, "estimated", "attention-surface", "oracle-visible"):
        # SL7: the extent flag itself is ported, but only the attention-surface one (the hybrid needs the RB-Y1 ball-owned
        # split protection and a rendered-robot mask). SL12: oracle-visible too -- a diagnostic that reads the LIBERO clients'
        # GT code image (``--oracle-segmentation``, ``benchmark/adapters/libero_panda/oracle_codes.py``)
        message = (f"--robot panda: --target-extent {args.target_extent} is not ported (SL7 ported attention-surface, SL12 "
                   "oracle-visible; attention-surface-hybrid is RB-Y1's)")
    elif _hidden_probe_source(args):
        # main 02fe59e6 merge: the hidden-probe target map is a probe trained on RB-Y1 16D hidden states (3 cameras)
        message = ("--robot panda: --attention-source hidden_probe is an RB-Y1-trained probe (not ported; needs a "
                   "Panda probe)")
    elif getattr(args, "palm_shell", "off") not in (None, "off", "on", "left"):
        message = f"--robot panda: --palm-shell {args.palm_shell} (the Panda is the single left hand; use on / left)"
    if message:
        if ap is None:
            raise ValueError(message)
        ap.error(message)


def check_record_flags(args) -> None:
    """``--record-constraints`` checks, before the checkpoint is loaded (same refusals as the RB-Y1 server + one).

    * the ESDF backend must be one the writer can record (``SUPPORTED_ESDF_BACKENDS``, ``serve_safe.py`` RB-Y1
      branch): otherwise every chunk fails and the run leaves an empty directory;
    * Panda only: refused with ``--no-safe`` (no ``SafePolicy``, nothing to record; the RB-Y1 server ignores it).
    """
    if not getattr(args, "record_constraints", None):
        return
    from AG3S.ag3s.experiments.sources.constraint_record import SUPPORTED_ESDF_BACKENDS

    if getattr(args, "no_safe", False):
        raise SystemExit("--record-constraints 는 --no-safe (VLA-only) Panda server 에서 기록할 것이 없습니다 "
                         "(SafePolicy 가 없다). 그대로 띄우면 빈 디렉토리만 남습니다")
    if args.esdf_backend not in SUPPORTED_ESDF_BACKENDS:
        raise SystemExit(f"--record-constraints 는 esdf backend {args.esdf_backend!r} 의 거리장을 실을 수 없습니다 "
                         f"(지원: {list(SUPPORTED_ESDF_BACKENDS)}). 그대로 띄우면 청크마다 기록이 실패하고 "
                         f"**빈 디렉토리**가 남습니다")


def recorder_meta(args, to_config, *, conversion: str, ag3s_info: dict) -> dict:
    """``ConstraintRecordWriter`` meta: the RB-Y1 server's keys (``serve_safe.py`` RB-Y1 branch, same rules for the
    optional ones) with the Panda's effective values, plus ``robot`` / ``ag3s_version`` / ``version_flags`` /
    ``conversion`` / Panda geometry. ``grasp_contact`` is added by :func:`build_safe_server` once applied.

    Effective, not argv, where the Panda differs: ``model_xml`` is ``None`` (the Panda reads its URDF; the RB-Y1
    MJCF in argv is not used), ``joint_limits`` / ``sqp`` / ``plan_horizon`` come from the Panda ``to_config``
    (``_replace_horizon_limits``), ``subtask.probe`` is ``None`` (no Panda probe). ``sphere_options`` /
    ``finger_cover`` keep the RB-Y1 rule (key only when given) and ``v11ab_build_kwargs_not_applied`` says they are
    not applied to the Panda models.
    """
    from AG3S.adapters.panda import profile
    from AG3S.trajopt import serve_safe as ss
    from AG3S.trajopt.safe_policy import _jsonable

    meta = {"config": args.config, "checkpoint": args.checkpoint, "model_xml": None, "links": args.links,
            **({"shadow": True} if args.shadow else {}),
            **({"exclude_links": list(args.exclude_links)} if args.exclude_links else {}),
            **({"self_collision": False} if args.no_self_collision else {}),
            **({"target_field_policy": args.target_field_policy} if args.target_field_policy != "relax" else {}),
            **({"collision": "off"} if args.no_collision else {}),
            **({"limits": "off"} if args.no_limits else {}),
            "joint_limits": ss.joint_limits_meta(to_config.limits),
            "sqp": ss.sqp_meta(to_config.sqp),
            "plan_horizon": int(to_config.horizon.planned),
            **({"sphere_options": ss.sphere_options(args)} if ss.sphere_options(args) else {}),
            **({"gripper_cover": ss.gripper_cover_options(args)} if ss.gripper_cover_options(args) else {}),
            **({"finger_cover": ss.finger_cover_options(args)} if ss.finger_cover_options(args) else {}),
            **({"self_filter": ss.self_filter_options(args)} if ss.self_filter_options(args) else {}),
            "subtask": {"gate": bool(args.subtask_gate), "probe": None,
                        **ss.subtask_from_latch_record_meta(args),   # SL7 (T92) — only when on
                        # SL6-A A11 — key only when on (as the RB-Y1 meta); inert on the Panda (no `home` label) unless
                        # the gate label comes from the latch (SL7: PLACED -> `home`, the rule can fire)
                        **({"placed_no_seed_certify": True,
                            **({} if ss.subtask_from_latch_on(args) else {"placed_no_seed_certify_inert_on_panda": True})}
                           if getattr(args, "placed_no_seed_certify", False) else {})},
            # SL6-A — the RB-Y1 meta keys of the final configuration, only when on (V11AB / pb meta unchanged)
            **({"palm_shell": ss.palm_shell_options(args)} if ss.palm_shell_options(args) else {}),
            **ss.t57a_record_meta(args),
            **ss.hsr6_record_meta(args),
            **ss.ifg_cl1_record_meta(args),   # SL7 — attention-surface extent flags, only the ones on
            # --- Panda (SL5j) ---
            "robot": "panda", "robot_urdf": str(pm.PANDA_URDF),
            "ag3s_version": profile.match_ag3s_version(args),
            "version_flags": profile.version_flag_values(args),
            "conversion": {"mode": conversion, "name": pa.CONVERSION_MODES[conversion]},
            "esdf_backend": args.esdf_backend,
            "horizon": dict(PANDA_HORIZON),
            "constraint_links": ag3s_info.get("constraint_links"),
            "n_constraint_spheres": ag3s_info.get("n_constraint_spheres"),
            "esdf_bounds": ag3s_info.get("esdf_bounds"),
            "v11ab_build_kwargs_not_applied": ["constraint_sphere_options", "finger_cover"]}
    return _jsonable(meta)


def constraint_recorder(args, to_config, *, conversion: str, ag3s_info: dict):
    """``ConstraintRecordWriter`` for ``--record-constraints DIR`` as the RB-Y1 server makes it, else ``None``."""
    if not getattr(args, "record_constraints", None):
        return None
    check_record_flags(args)
    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

    # T38 B5 — npz writes off the response path; same content and order; flushed / closed by `main`.
    recorder = ConstraintRecordWriter(args.record_constraints, esdf_mode=args.record_constraints_esdf,
                                      background=True,
                                      meta=recorder_meta(args, to_config, conversion=conversion, ag3s_info=ag3s_info))
    logging.info("recording AG3S constraint diagnostics to %s (esdf=%s, backend=%s)",
                 recorder.run_dir, args.record_constraints_esdf, args.esdf_backend)
    if args.record_constraints_esdf == "occupancy" and args.esdf_backend == "curobo":
        logging.warning(
            "!!! --record-constraints-esdf occupancy 는 cuRobo backend 에서 점유 배열을 남기지 않습니다 !!!\n"
            "    block-sparse TSDF 에는 legacy 의 3 상태(FREE/OCCUPIED/UNKNOWN)에 대응하는 배열이 없습니다. "
            "격자 메타·계층 정보는 남고, **거리장 자체를 보려면 `full` 을 쓰십시오** (계층마다 float16 격자: "
            "주 계층 · coarse · target 없는 계층).")
    return recorder


def close_recorder(served) -> Optional[dict]:
    """Flush and close the served object's recorder (``meta.json``); ``None`` if it has none. Never raises."""
    recorder = getattr(served, "recorder", None)
    if recorder is None:
        return None
    try:
        recorder.close()
    except Exception:  # noqa: BLE001 -- the server is stopping; say it loudly, do not mask the exit cause
        logging.exception("!!! constraint recorder close failed (%s) — meta.json may be missing !!!",
                          getattr(recorder, "run_dir", "?"))
        return {"closed": False}
    return {"closed": True, "run_dir": str(recorder.run_dir), "n_chunks": recorder.meta.get("n_chunks"),
            "n_failed": getattr(recorder, "n_failed", 0)}


#: SL6-A A3 — the Panda values of SafePolicy's robot injections (defaults there = RB-Y1). ``arm_joint_names`` index j
#: (0-6) = the RB-Y1 ``{hand}_arm_{j}``: IFG-TOW's wrist joints (4, 5, 6) are ``panda_joint5..7``, the same wrist set.
PANDA_ARM_JOINT_NAMES = {pm.MANIPULATOR_SIDE: tuple(pm.PANDA_ARM_JOINTS)}
#: IFG-NBC finger links (RB-Y1: the grasp link minus its digits, ``ee_finger_l*``): the Panda grasp pair, both fingers.
PANDA_NEIGHBOUR_FINGER_LINKS = {pm.MANIPULATOR_SIDE: tuple(pm.PANDA_FINGER_LINKS)}
#: The SafePolicy note for the A11 feature that has no Panda label.
PLACED_NO_SEED_INERT = ("--placed-no-seed-certify certifies no_seed frames after PLACED in the 3-class subtask label "
                        "'home' (ag3s/types.py); the Panda server has no 3-class subtask probe (subtask label None), so "
                        "the rule never fires: inert on panda (not fixed, task SL6 A11)")
#: AttentionPolicy noun rule of the LIBERO prompts (SL6-B B1, ``attention_source.NounTokens.from_config(rule=)``).
PANDA_NOUN_RULE = "libero"


def panda_gpu_seed_robot():
    """SL6-A A4 — the cuRobo seed batch's robot constants for the Panda: adapter URDF, base ``panda_link0`` (fixed
    root: the world needs no per-chunk pose), tool frame ``panda_hand`` (the hold link the grasp goal names)."""
    from AG3S.trajopt.gpu_seeds import GpuSeedRobot

    return GpuSeedRobot(name="panda", urdf=str(pm.PANDA_URDF), base_link="panda_link0", tool_frames=("panda_hand",),
                        fk_root_link="panda_link0")


def safe_policy_injections() -> dict:
    """SL6-A A3 / A6 / A8 — the Panda geometry for SafePolicy's injection points (each changes nothing while its
    feature is off, so a V11AB / pb server is unchanged)."""
    from AG3S.adapters.panda import hand_geometry as hg
    from AG3S.adapters.panda import profile
    from AG3S.ag3s.grasp.gripper import PANDA_LEFT

    return {"arm_joint_names": dict(PANDA_ARM_JOINT_NAMES),
            "neighbour_finger_links": dict(PANDA_NEIGHBOUR_FINGER_LINKS),
            "hold_site_offset_m": tuple(hg.PANDA_HOLD_SITE_OFFSET_M),
            "palm_links": tuple(profile.PANDA_HOLD_LINKS.values()),
            "geo_gripper": {pm.MANIPULATOR_SIDE: PANDA_LEFT}}


def _noun_tokens(config_name: str):
    """``NounTokens`` with the LIBERO rule when the attention-source module offers one (SL6-B B1), else the default
    (fruit) rule -- recorded by the caller."""
    import inspect

    from AG3S.trajopt.attention_source import NounTokens

    if "rule" in inspect.signature(NounTokens.from_config).parameters:
        return NounTokens.from_config(config_name, rule=PANDA_NOUN_RULE), PANDA_NOUN_RULE
    logging.warning("!!! attention_source.NounTokens has no LIBERO rule: the noun lookup uses the fruit rule !!!")
    return NounTokens.from_config(config_name), "fruit"


def attention_policy_kwargs(args, *, tokenizer: bool = True) -> tuple[dict, dict]:
    """SL6-A A1 — ``(AttentionPolicy keyword args, record)`` as the RB-Y1 ``serve_safe.main`` builds them (T57-A
    attention source, AG3S-CORE-EVAL-HSR6 probe); ``({}, {})`` when every one of those flags is off (the V11AB
    AttentionPolicy). The task gate / confidence gate stay unported (refused before this). ``tokenizer=False`` (no
    attention model: tests, ``--no-attention``) skips the noun tokens (they need the policy's SentencePiece model)."""
    from AG3S.trajopt import serve_safe as ss

    kw: dict = {}
    rec: dict = {}
    if getattr(args, "attention_source", None):
        from AG3S.trajopt.attention_source import log_source, parse_attention_source

        source = parse_attention_source(args.attention_source)
        kw = {"attention_source": source, "task_probe": None, "default_prompt": args.default_prompt}
        rec["attention_source"] = source.label
        if source.is_noun:
            if tokenizer:
                kw["noun_tokens"], rec["noun_rule"] = _noun_tokens(args.config)
            else:
                rec["noun_rule"] = "not built (no attention model)"
        log_source(source, None)
    flags = ss.hsr6_flags_from_args(args)
    if flags is not None:
        from AG3S.trajopt.hsr6 import load_hsr6_probe

        import hashlib
        import pathlib

        kw["hsr6_probe"] = load_hsr6_probe(args.hsr6_probe)
        rec["hsr6_probe"] = {"path": str(args.hsr6_probe), "name": getattr(kw["hsr6_probe"], "name", None),
                             "type": type(kw["hsr6_probe"]).__name__,
                             "classes": list(getattr(kw["hsr6_probe"], "classes", ()) or ()),
                             "sha256": hashlib.sha256(pathlib.Path(args.hsr6_probe).read_bytes()).hexdigest()}
        logging.warning("HSR6 (panda): probe %s (%s) · uses %s · raw argmax per request · objective scope %s",
                        rec["hsr6_probe"]["name"], args.hsr6_probe, flags.names(),
                        getattr(args, "hsr6_objective_scope", "all"))
    return kw, rec


def gpu_seeds_setup(safe, to_config, *, warm_up: bool) -> Optional[dict]:
    """SL6-A A4 — with ``--to-gpu-seeds`` (else None): the Panda robot for the seed batches and, with ``warm_up``, the
    batches built and their CUDA graphs captured before the first chunk (RB-Y1 ``serve_safe.main``). A build that fails
    does **not** stop the server (each triggered chunk then solves the seeds on the CPU, `gpu_seeds.solve_seeds`) but is
    said loudly here and recorded (``warm_up_error``) -- not only in the per-chunk record."""
    if getattr(to_config.sqp, "gpu_seeds", "off") == "off":
        return None
    from AG3S.trajopt import gpu_seeds as gs

    robot = panda_gpu_seed_robot()
    gs.set_robot(safe.refiner.optimizer, robot)
    rec: dict = {"backend": str(to_config.sqp.gpu_seeds), "check": str(getattr(to_config.sqp, "gpu_seeds_check", "all")),
                 "n_seeds": int(getattr(to_config.sqp, "multi_seed", 0) or 0), "robot": robot.record(),
                 "warm_up": None}
    if not warm_up:
        rec["warm_up"] = "skipped (caller)"
        return rec
    try:
        rec["warm_up"] = gs.warm_up(safe.refiner.optimizer)
        logging.warning("TO GPU seeds warm-up (panda): %s", rec["warm_up"])
    except Exception as exc:  # noqa: BLE001 - the server stays up (CPU seeds per chunk), loudly
        rec["warm_up_error"] = f"{type(exc).__name__}: {exc}"
        logging.warning("!!! TO GPU seeds (panda): the cuRobo seed batch could not be built — %s. Every chunk that "
                        "triggers multi-seed will solve the %d seeds on the CPU (gpu_seeds.solve_seeds fallback) !!!",
                        rec["warm_up_error"], rec["n_seeds"])
    return rec


#: SL10 §2.2 -- OSQP's iteration cap with the action rows. The rows (coefficients ~10^2: 1 / (beta scale) of a difference
#: of EE poses) make the ADMM slow: 2000-6000 iterations for a QP that takes 300-1000 without them (measured on the
#: synthetic Panda chunks of ``tests/benchmark/test_sl10_action_box.py``), and the default cap (4000) ended a first QP as
#: ``maximum iterations reached`` -- a failed QP returns the iterate unchanged, so the box would silently switch the
#: correction off. Only with ``--to-action-box``; the SQP iteration count (3) and its wall budget are unchanged.
ACTION_BOX_QP_MAX_ITER = 30000


def action_box_qp_config(to_config):
    return dataclasses.replace(to_config, qp=dataclasses.replace(to_config.qp, max_iter=ACTION_BOX_QP_MAX_ITER))


def action_box_kwargs(args, to_config, chunk_policy: "ChunkToJointPolicy", mode: str) -> dict:
    """SL10 §2.2 -- ``{"action_box_fn": fn}`` for ``SafePolicy`` with ``--to-action-box``, else ``{}`` (the SafePolicy
    arguments are then exactly the earlier ones). ``fn()`` is the action-limit box of the chunk the policy just produced
    (``ChunkToJointPolicy.last``, the adapter's own conversion). The rows are the executed ones
    (``PANDA_HORIZON["execution_length"]``). With the look-ahead gate closed the K-row window optimizer carries the same
    rows."""
    if not getattr(args, "to_action_box", False):
        return {}
    from AG3S.trajopt.action_box import ACTION_BOX_MARGIN_DEFAULT

    margin = getattr(args, "to_action_box_margin", None)
    margin = ACTION_BOX_MARGIN_DEFAULT if margin is None else float(margin)
    n_rows = int(PANDA_HORIZON["execution_length"])

    def action_box_fn():
        conv = chunk_policy.last
        return None if conv is None else pa.action_box_from_conversion(conv, margin=margin, n_rows=n_rows, mode=mode)

    return {"action_box_fn": action_box_fn}


def build_safe_server(policy, args, *, attention_model=None, gpu_warm_up: bool = True):
    """Everything below the openpi policy (also used by tests with a stub policy and no attention model).

    SL6-A: the final-configuration options are wired as the RB-Y1 ``serve_safe.main`` wires them -- attention source
    and HSR6 probe into ``AttentionPolicy``, ``hsr6=`` / ``hsr6_objective_scope=`` into ``SafePolicy``, the GPU seed
    warm-up (``gpu_warm_up``; tests on a CPU host pass False) -- each only when its flag is on, plus the Panda
    geometry for SafePolicy's robot injections (:func:`safe_policy_injections`)."""
    from AG3S.adapters.panda import attention as pat
    from AG3S.adapters.panda import profile
    from AG3S.ag3s.robot_models.gripper_state import FingerJointMap
    from AG3S.trajopt import serve_safe as ss
    from AG3S.trajopt.safe_policy import SafePolicy

    kw = profile.v11ab_build_kwargs(args)
    ag3s, ag3s_info = profile.build_panda_ag3s(kw=kw)
    served_policy = policy
    attn_kw, attn_rec = attention_policy_kwargs(args, tokenizer=attention_model is not None)
    if attention_model is not None:
        from AG3S.trajopt.attention_policy import AttentionPolicy

        served_policy = AttentionPolicy(policy, attention_model, subtask_probe=None, **attn_kw)
    constraint = ag3s.constraint_robot_model
    kin = pa.PandaKinematics(constraint)
    response_name = getattr(args, "response_model", "robosuite141")   # SL11
    gap_opening = bool(getattr(args, "gripper_gap_opening", False))     # SL11 §5
    chunk_policy = ChunkToJointPolicy(served_policy, kin, response=pa.response_for(response_name), response_name=response_name)
    to_config = _replace_horizon_limits(ss.trajopt_config_from_args(args))
    if getattr(args, "to_action_box", False):
        to_config = action_box_qp_config(to_config)   # SL10 §2.2: a QP budget the action rows can converge in
    attention_fn = ss.attention_extractor(alias=ATTENTION_ALIAS, cameras=ATTENTION_CAMERAS,
                                          transform=lambda _cam, grid: pat.unmirror_grid(np.asarray(grid)))
    mode = conversion_mode()
    # SL5j — None without --record-constraints (then the SafePolicy arguments are the pre-SL5j ones)
    recorder = constraint_recorder(args, to_config, conversion=mode, ag3s_info=ag3s_info)
    hsr6 = ss.hsr6_flags_from_args(args)
    scope = getattr(args, "hsr6_objective_scope", "all")
    # SL6-A — SafePolicy's Panda geometry + the HSR6 arguments only with a final-configuration option on: a V11AB / pb
    # server constructs SafePolicy with exactly the pre-SL6 arguments (tests/benchmark/test_panda_sl5j_recorder.py).
    abox_kw = action_box_kwargs(args, to_config, chunk_policy, mode)   # SL10 §2.2: {} unless --to-action-box
    final_kw = ({**safe_policy_injections(), **({"hsr6": hsr6} if hsr6 is not None else {}),
                 **({"hsr6_objective_scope": scope} if scope != "all" else {}),
                 **ss.subtask_from_latch_kwargs(args)}   # SL7 (T92): {} unless --subtask-from-latch
                if final_options_on(args) else {})
    safe = SafePolicy(chunk_policy, ag3s=ag3s, to_config=to_config, attention_fn=attention_fn, latch=panda_latch_config(gap_opening=gap_opening),
                      grasp_links=dict(profile.PANDA_GRASP_LINKS), hold_links=dict(profile.PANDA_HOLD_LINKS),
                      static_geometry=None, recorder=recorder, shadow=bool(getattr(args, "shadow", False)),
                      layout=pa.panda_chunk_layout(constraint.joint_names), gripper_columns=GRIPPER_COLUMNS,
                      finger_map=FingerJointMap(constraint, convention=panda_gripper_convention()), hands=HANDS,
                      **final_kw, **abox_kw)
    grasp_contact = ss.apply_grasp_contact_flags(safe, args)  # V11AB-pb4 / pb5 (all off = V11AB)
    gpu = gpu_seeds_setup(safe, to_config, warm_up=gpu_warm_up)
    inert = []
    ss.announce_subtask_from_latch(args)   # SL7 (T92): one start line when on
    if getattr(args, "placed_no_seed_certify", False) and not ss.subtask_from_latch_on(args):
        # with --subtask-from-latch the gate label has `home` (latch PLACED), so the rule is not inert (SL7)
        inert.append({"flag": "--placed-no-seed-certify", "why": PLACED_NO_SEED_INERT})
        logging.warning("!!! %s !!!", PLACED_NO_SEED_INERT)
    info = {"ag3s": ag3s_info, "horizon": PANDA_HORIZON, "grasp_contact": grasp_contact, "conversion_mode": {"mode": mode, "name": pa.CONVERSION_MODES[mode]},
            "latch": {"empty_below": panda_latch_config().empty_below, "calibration": str(GRIPPER_CALIBRATION),
                      "calibration_record": gripper_calibration()},
            "limits": {"source": "urdf", "urdf": str(pm.PANDA_URDF), "default_acceleration": PANDA_DEFAULT_ACCELERATION,
                       "franka_accel_limits": FRANKA_ACCEL_LIMITS},
            "to_config": {"horizon": dataclasses.asdict(to_config.horizon), "planned": to_config.horizon.planned},
            "attention": {"alias": ATTENTION_ALIAS, "cameras": ATTENTION_CAMERAS, "unmirror": True,
                          "cell": "V11AB (attention_policy._LAYER/_HEAD/_AGG/_DENOISE_STEP)",
                          "subtask_probe": None, "loaded": attention_model is not None},
            "hands": HANDS, "gripper_columns": GRIPPER_COLUMNS, "gripper_convention": panda_gripper_convention().record(),
            "conversion": pa.conventions(), "static_geometry": None,
            "v11ab_build_kwargs_not_applied": ["constraint_sphere_options", "finger_cover"]}
    if gap_opening:   # SL11 §5 -- key only with the flag (the earlier servers' dump is unchanged)
        gc = panda_latch_config(gap_opening=True)
        info["latch"]["gripper_gap_opening"] = {"calibration": str(GRIPPER_GAP_CALIBRATION), "record": gripper_gap_calibration(),
                                                "empty_below": gc.empty_below, "blocked_gap": gc.blocked_gap,
                                                "hold_empty_below": gc.hold_empty_below,
                                                "assist_requires_reach": gc.assist_requires_reach}
    final = final_config_record(args, safe, to_config, attn_rec, gpu, inert)
    if response_name != "robosuite141":   # SL11 -- key only with the fork model (the earlier servers' dump is unchanged)
        info["response_model"] = {"name": response_name, "pos": chunk_policy.response["pos"],
                                  "rot": chunk_policy.response["rot"], "fit": chunk_policy.response["path"],
                                  "requires_action_adapter": pa.FORK_REQUIRED_ADAPTER}
    if final:   # SL6-A — key only with a final-configuration option on (V11AB / pb config unchanged)
        info["final_config"] = final
    if recorder is not None:  # key only when recording (the panda_server_config.json of other runs is unchanged)
        from AG3S.trajopt.safe_policy import _jsonable

        recorder.meta["grasp_contact"] = _jsonable(grasp_contact)
        if final:
            recorder.meta["final_config"] = _jsonable(final)
        info["record_constraints"] = {"dir": str(args.record_constraints), "run_dir": str(recorder.run_dir),
                                      "esdf_mode": recorder.esdf_mode, "background": recorder.background,
                                      "ag3s_version": recorder.meta.get("ag3s_version")}
    if getattr(args, "observe_only", False):   # SL11 (off: the constructor call is the one of the earlier versions)
        info["observe_only"] = True
        logging.warning("!!! --observe-only: the client gets the policy's action bit for bit and is never held; the "
                        "verdict and the converted actions the server computed are recorded (panda/observe) !!!")
        return PandaSafeServer(safe, chunk_policy, mode=mode, observe_only=True,
                               **({"gap_opening": True} if gap_opening else {})), info
    return PandaSafeServer(safe, chunk_policy, mode=mode, **({"gap_opening": True} if gap_opening else {})), info


def final_options_on(args) -> list[str]:
    """SL6-A — the ported final-configuration options ``args`` sets away from their parser default (``[]`` for every
    V11AB / pb argv)."""
    from AG3S.trajopt import serve_safe as ss

    parser = ss.build_parser()
    return sorted(d for d in PORTED_FINAL_CONFIG_DESTS + ("hsr6_objective_scope",)
                  if getattr(args, d, None) != parser.get_default(d))


def surface_record(args, ag3s) -> dict:
    """SL7 -- what the Panda applies of the P0 target extent (attention-surface): the AG3S config keys, the camera
    handling (head + left_wrist real, right_wrist a dummy), the robot mask (FK spheres) and the hull-fill palm point."""
    from AG3S.ag3s import config as ag3s_config
    from AG3S.ag3s.runtime import attention_surface as asurf

    e = ag3s.config.esdf
    return {"target_extent": e.target_extent, "seed_map": getattr(args, "attention_source", None),
            "surface_target_check": e.surface_target_check, "surface_keep_last": bool(e.surface_keep_last),
            "surface_grasp_hold": bool(e.surface_grasp_hold), "surface_no_ball": bool(e.surface_no_ball),
            "surface_hull_fill": bool(e.surface_hull_fill), "surface_hull_fill_palm_mm": float(e.surface_hull_fill_palm_mm),
            "surface_contact_allowance_mm": e.surface_contact_allowance_mm,
            "device_requested": getattr(args, "attention_surface_device", None),
            "device_in_effect": ag3s_config.ATTENTION_SURFACE_DEVICE,
            "camera_order": list(asurf.CAMERA_ORDER), "real_cameras": list(ATTENTION_CAMERAS),
            "dummy_cameras": [n for n in asurf.CAMERA_ORDER if n not in ATTENTION_CAMERAS],
            "pad_missing_cameras": bool(ag3s.surface_pad_missing_cameras),
            "robot_mask": e.robot_mask, "arm": asurf.ARM_SPHERES if e.robot_mask != "render" else asurf.ARM_RENDER,
            "hull_fill_palm_point": "mean of the grasping gripper's finger-link constraint spheres "
                                    "(pipeline._grasp_palm_points; the A6 hold site is not used here)"}


def subtask_latch_record(safe) -> dict:
    """SL7 -- the 3-stage gate label comes from the grasp latch (T92): searching / latched / closing -> pick, held ->
    place, placed -> home; the gate (``clustering.subtask_gate``) and its debounce as the config has them."""
    from AG3S.trajopt.safe_policy import LATCH_SUBTASK

    c = safe.ag3s.config.clustering
    return {"source": "latch", "subtask_gate": bool(c.subtask_gate), "confirm_frames": c.subtask_confirm_frames,
            "target_confirm_frames": c.target_confirm_frames,
            "label_of_phase": {getattr(k, "value", str(k)): v for k, v in LATCH_SUBTASK.items()},
            "kv_L4_v1_probe": None}


def final_config_record(args, safe, to_config, attn_rec: dict, gpu: Optional[dict], inert: list) -> dict:
    """SL6-A — what the Panda server applies of the AG3S final configuration (VA4 config dump): ``{}`` when no
    final-configuration option is on (every ported dest at its parser default)."""
    from AG3S.adapters.panda import hand_geometry as hg
    from AG3S.adapters.panda import profile
    from AG3S.trajopt import serve_safe as ss
    from AG3S.trajopt.safe_policy import _jsonable

    on = final_options_on(args)
    if not on:
        return {}
    col = to_config.collision
    shell = getattr(safe.constraint_robot_model, "palm_shell_report", None)
    return _jsonable({
        "version": profile.match_ag3s_version(args),
        "options_on": on,
        "attention": attn_rec,
        "hsr6": None if safe.hsr6 is None else {"flags": safe.hsr6.names(), "objective_scope": safe.hsr6_objective_scope,
                                                "probe": getattr(args, "hsr6_probe", None)},
        "palm_shell": None if shell is None else {
            "n_spheres": shell["n_spheres_after"] - shell["n_spheres_before"],
            "n_spheres_before": shell["n_spheres_before"], "n_spheres_after": shell["n_spheres_after"],
            "table": shell["links"][hg.PALM_LINK], "source": shell["source"], "shell_digest": shell["shell_digest"],
            "margin_m": getattr(col, "palm_shell_margin", None)},
        "neighbour_palm_contact": None if not float(getattr(col, "neighbour_palm_contact_depth", 0.0) or 0.0) > 0.0 else {
            "site": "panda_grip_site", "site_offset_m": list(hg.PANDA_HOLD_SITE_OFFSET_M), "hold_link": hg.PALM_LINK,
            "depth_m": float(col.neighbour_palm_contact_depth), "radius_m": float(col.neighbour_palm_contact_radius),
            "gate_m": float(col.neighbour_palm_contact_gate),
            "palm_rows": int(sum(1 for n in safe.constraint_robot_model.sphere_link_names if n == hg.PALM_LINK))},
        "neighbour_contact": None if not float(getattr(col, "neighbour_contact_depth", 0.0) or 0.0) > 0.0 else {
            "finger_links": safe.neighbour_finger_links, "depth_m": float(col.neighbour_contact_depth)},
        "wrist_track": None if float(getattr(to_config.cost, "wrist_track_scale", 1.0)) >= 1.0 else {
            "joints": [safe.arm_joint_names[pm.MANIPULATOR_SIDE][int(j)] for j in to_config.cost.wrist_track_joints],
            "scale": float(to_config.cost.wrist_track_scale)},
        "multi_seed": None if int(getattr(to_config.sqp, "multi_seed", 0) or 0) <= 0 else {
            "n": int(to_config.sqp.multi_seed), "palm_links": list(safe.refiner.optimizer.__dict__.get("palm_links", ()))},
        "gpu_seeds": gpu,
        "geo_grasp": None if safe._geo is None else {
            "spec": safe._geo._spec(pm.MANIPULATOR_SIDE).hold_link,
            "frame_R_HG": safe._geo._spec(pm.MANIPULATOR_SIDE).frame,
            "max_opening_m": safe._geo._spec(pm.MANIPULATOR_SIDE).max_opening,
            "arm_joints": list(safe._geo._arm_names(pm.MANIPULATOR_SIDE) or ()),
            "gripper_column_override": "refined gripper column sent on chunks where G1 deferred the close "
                                       "(action.apply_deferred_close)"},
        # A12 — the continuity / deflection-rate terms off before the grasp (latch searching · latched · closing)
        "before_grasp_terms_off": {
            "continuity_after_grasp": bool(getattr(to_config.cost, "continuity_after_grasp", False)),
            "deflection_rate_after_grasp": bool(getattr(to_config.cost, "deflection_rate_after_grasp", False)),
            "w_continuity": float(to_config.cost.w_continuity),
            "w_deflection_rate": float(getattr(to_config.cost, "w_deflection_rate", 0.0) or 0.0),
            "geo_continuity_off": bool(getattr(to_config.cost, "geo_continuity_off", False))},
        "grasp_anchor": None if float(getattr(to_config.cost, "w_grasp_goal", 0.0) or 0.0) <= 0.0 else {
            "w_grasp_goal": float(to_config.cost.w_grasp_goal), "gripper_column": int(pa.GRIPPER_COLUMN)},
        # SL7 (P0m) — keys only with the option on (the earlier versions' dump is unchanged)
        **({"attention_surface": surface_record(args, safe.ag3s)} if getattr(args, "target_extent", None) == "attention-surface"
           else {}),
        **({"oracle_visible": {"target_extent": "oracle_visible", "identity": getattr(args, "oracle_extent_identity", None) or "pipeline",
                               "needs_client_flag": "--oracle-segmentation (metadata oracle_segmentation_required)",
                               "cameras": ["head", "left_wrist"], "code_table": "0 robot/none, 1 target, 2 static, >=3 other objects",
                               **({"target_volume": True} if getattr(args, "oracle_target_volume", False) else {})}}
           if getattr(args, "target_extent", None) == "oracle-visible" else {}),
        **({"subtask_from_latch": subtask_latch_record(safe)} if ss.subtask_from_latch_on(args) else {}),
        "inert_on_panda": inert,
        "request_count_rules_unchanged": "hold_follow_to / geo_close_timeout / geo_carry_requests / HSR6 raw argmax are "
                                         "request counts; Panda requests are 0.25 s apart (RB-Y1 0.53 s)",
    })


def _write_config(record: dict) -> Optional[str]:
    run = os.environ.get("AG3S_RUN_DIR")
    if not run:
        return None
    path = pathlib.Path(run) / "logs" / "panda_server_config.json"
    with open(path, "x") as f:
        json.dump(record, f, indent=1, default=str)
    return str(path)


def main(args, ap) -> None:
    from openpi.policies import policy_config as _policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _config

    from AG3S.trajopt.policy_seed import SeededPolicy, describe_rng_owner, describe_xla_determinism

    check_unported_flags(args, ap)  # before the checkpoint is loaded (no-op with every new flag at its default)
    check_record_flags(args)  # SL5j: refuse before the checkpoint is loaded (no-op without --record-constraints)
    policy = _policy_config.create_trained_policy(_config.get_config(args.config), args.checkpoint,
                                                  default_prompt=args.default_prompt)
    record: dict[str, Any] = {"robot": "panda", "config": args.config, "checkpoint": args.checkpoint,
                              "mode": "vla_only" if args.no_safe else "vla_v11ab", "port": args.port}
    if args.no_safe:
        served = PandaVLAOnly(SeededPolicy(policy))
    else:
        attn = None
        if not args.no_attention:
            from AG3S.trajopt.attention_policy import load_attention_model

            logging.info("loading a second copy of the checkpoint for attention extraction")
            attn = load_attention_model(args.config, args.checkpoint)
        served, info = build_safe_server(policy, args, attention_model=attn)
        record.update(info)
    record["config_path"] = _write_config(record)
    logging.info("panda server: %s", json.dumps({k: record[k] for k in ("mode", "port")}))
    logging.info("%s", describe_rng_owner(served))
    logging.info("%s", describe_xla_determinism(os.environ.get("XLA_FLAGS")))
    recording = getattr(served, "recorder", None) is not None
    if recording:
        from AG3S.trajopt import serve_safe as ss

        ss.exit_cleanly_on_sigterm()  # SIGTERM -> SystemExit, so the `finally` below runs (as the RB-Y1 server)
    try:
        websocket_policy_server.WebsocketPolicyServer(policy=served, host="0.0.0.0", port=args.port,
                                                      metadata=getattr(served, "metadata", {})).serve_forever()
    finally:
        # SL5j — normal return, SIGINT (KeyboardInterrupt; launcher `stop_own_process`) and SIGTERM (SystemExit):
        # write the queued chunks and meta.json. Without a recorder this is a no-op.
        if recording:
            logging.info("constraint recorder: %s", close_recorder(served))
