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


def conversion_mode() -> str:
    mode = os.environ.get(ENV_CONVERSION, "A").strip().upper()
    if mode not in pa.CONVERSION_MODES:
        raise ValueError(f"{ENV_CONVERSION}={mode!r}: expected one of {sorted(pa.CONVERSION_MODES)}")
    return mode


def gripper_calibration() -> dict:
    return json.loads(GRIPPER_CALIBRATION.read_text())


def panda_latch_config():
    """V11AB ``LatchConfig`` defaults with only ``empty_below`` calibrated for the Panda gripper."""
    from AG3S.trajopt.grasp_latch import LatchConfig

    return LatchConfig(empty_below=float(gripper_calibration()["panda"]["empty_below"]))
ATTENTION_CAMERAS = ("head", "left_wrist")
HANDS = {pm.MANIPULATOR_SIDE: 0}
GRIPPER_COLUMNS = (pa.GRIPPER_COLUMN,)
#: request keys only the Panda adapter reads (never reach the policy)
PANDA_PREFIX = "panda/"
KEY_ROBOT_STATE = "panda/robot_state"      # (9,) q7 + 2 MuJoCo finger qpos, image instant
KEY_T_WORLD_BASE = "panda/T_world_base"    # (4, 4)
KEY_PREV_STEP = "panda/prev_step_world"    # (6,) last realised EE step, world frame: dp (m), drotvec (rad)


def panda_gripper_convention():
    from AG3S.ag3s.robot_models.gripper_state import ClientGripperConvention

    return ClientGripperConvention(open_command_m=0.04, driven_joints={pm.MANIPULATOR_SIDE: "panda_finger_joint1"},
                                   source="AG3S/adapters/panda/server.py: LIBERO Panda finger joint1 open limit 0.04 m "
                                          "(adapter URDF); normalised opening = q / 0.04")


def strip_panda_keys(obs: dict) -> tuple[dict, dict]:
    keep = {k: v for k, v in obs.items() if not k.startswith(PANDA_PREFIX)}
    panda = {k: v for k, v in obs.items() if k.startswith(PANDA_PREFIX)}
    return keep, panda


#: SafePolicy reads the capture-time gripper opening from the policy observation's ``state`` at
#: ``gripper_columns`` (T29 convention, 1 = open); the Panda request carries it in ``panda/robot_state``.
STATE_KEY = "state"


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

    def __init__(self, policy, kin: pa.PandaKinematics):
        self._policy = policy
        self.kin = kin
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
        result = dict(self._policy.infer(obs, **kwargs))
        delta = np.asarray(result["actions"])
        prev = panda.get(KEY_PREV_STEP)
        prev = None if prev is None else (np.asarray(prev, np.float64)[:3], np.asarray(prev, np.float64)[3:6])
        self.last = pa.delta_chunk_to_joint_chunk(self.kin, delta, q_now, np.asarray(panda[KEY_T_WORLD_BASE]),
                                                  prev_step_world=prev)
        self.last_result_actions = delta
        result["actions"] = self.last.joint_chunk
        return result


class PandaSafeServer:
    """Outside ``SafePolicy``: refined joint chunk -> EE-delta chunk (``action.joint_chunk_to_delta_chunk``)."""

    def __init__(self, safe, chunk_policy: ChunkToJointPolicy, mode: str = "A"):
        self._policy = safe
        self.chunk_policy = chunk_policy
        self.mode = mode
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

    def infer(self, obs: dict, **kwargs) -> dict:
        obs = dict(obs)
        if obs.get("ag3s/reset"):
            self._sent.clear()  # a new episode: seq restarts, earlier chunks are not this episode's
        fb_rec = self._feedback(obs)
        response = dict(self._policy.infer(adapt_request(obs), **kwargs))
        conv = self.chunk_policy.last
        refined, reference = response["actions"], response.get("actions_reference")
        if conv is None or reference is None:
            raise RuntimeError("SafePolicy returned no reference chunk for the Panda conversion")
        actions, rec = pa.joint_chunk_to_delta_chunk(self.chunk_policy.kin, conv, refined, reference, mode=self.mode)
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
        return response


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
UNPORTED_FINAL_CONFIG_DESTS: tuple[str, ...] = (
    'attention_source', 'attention_split_partition', 'attention_split_radius_mm', 'attention_split_ratio_source',
    'confidence_gate', 'confidence_gate_threshold', 'continuity_after_grasp', 'deflection_rate_after_grasp',
    'free_space_carve', 'free_space_carve_margin_mm', 'geo_arm_check', 'geo_carry_requests', 'geo_close_gate',
    'geo_close_timeout', 'geo_close_tol_deg', 'geo_close_tol_mm', 'geo_continuity_off', 'geo_correct_deg',
    'geo_correct_max_offset', 'geo_correct_mm', 'geo_goal_mode', 'geo_grasp', 'geo_grasp_mu', 'geo_grasp_selection',
    'geo_obstacle_memory', 'geo_obstacle_source', 'geo_time_budget_ms', 'geo_track_scale', 'geo_w_pos', 'geo_w_rot',
    'geo_wrist_check', 'grasp_anchor', 'grasp_anchor_path_weight', 'grasp_anchor_radius_mm', 'grasp_goal_box',
    'grasp_goal_window', 'held_3d_max_m', 'held_3d_rule', 'held_permission_parent_links', 'hsr6_attach_assist',
    'hsr6_held_gate', 'hsr6_objective', 'hsr6_probe', 'hsr6_target_exempt', 'hybrid_identity',
    'hybrid_other_exclude_voxels', 'kept_geometry_follow', 'latch_carry_from_closing', 'latch_reach_hand_relative',
    'latch_reach_kept_unseen', 'neighbour_contact_mm', 'neighbour_palm_contact_mm', 'nonauthorized_target_margin',
    'nontarget_evidence', 'nontarget_shell_voxels', 'oracle_extent_identity', 'oracle_max_dist_voxels',
    'owned_ambiguous', 'owned_max_dist_voxels', 'owned_tie_voxels', 'owned_unowned', 'palm_cover',
    'palm_cover_overflow', 'palm_shell', 'palm_shell_margin', 'palm_shell_overflow', 'placed_no_seed_certify',
    'profile_substages', 'qp_backend', 'recovery_retreat', 'robot_free_shrink_mm', 'robot_mask',
    'robot_occupied_free', 'roi_carve', 'roi_carve_device', 'roi_carve_footprint', 'roi_carve_footprint_pad_px',
    'roi_carve_footprint_robot', 'roi_pad_mm', 'split_supersedes_anchor', 'stem_cap_bottom_fit_frac',
    'stem_cap_radius_mm', 'stem_cap_top_mm', 'target_exclusion', 'target_extent', 'target_free_coarse',
    'target_free_coarse_sign_floor', 'target_free_coarse_support', 'target_layer', 'target_layer_reset_mm',
    'target_layer_tol_mm', 'target_permission_strict', 'target_stem_cap', 'task_gate', 'task_gate_shadow',
    'task_probe', 'to_gpu_seeds', 'to_gpu_seeds_check', 'to_multi_seed', 'tsdf_frustum_decay', 'tsdf_time_decay',
    'wrist_cover_overflow', 'wrist_track_scale',
)


def check_unported_flags(args, ap) -> None:
    """Refuse a final-configuration option the Panda server would not apply as the RB-Y1 server does."""
    from AG3S.trajopt import serve_safe as ss

    parser = ap if ap is not None else ss.build_parser()
    bad = sorted(d for d in UNPORTED_FINAL_CONFIG_DESTS
                 if hasattr(args, d) and getattr(args, d) != parser.get_default(d))
    if bad:
        message = ("--robot panda: not ported to the Panda server yet (AG3S final-configuration merge 2026-10-29): "
                   + ", ".join("--" + d.replace("_", "-") for d in bad))
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
            "subtask": {"gate": bool(args.subtask_gate), "probe": None},
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


def build_safe_server(policy, args, *, attention_model=None):
    """Everything below the openpi policy (also used by tests with a stub policy and no attention model)."""
    from AG3S.adapters.panda import attention as pat
    from AG3S.adapters.panda import profile
    from AG3S.ag3s.robot_models.gripper_state import FingerJointMap
    from AG3S.trajopt import serve_safe as ss
    from AG3S.trajopt.safe_policy import SafePolicy

    kw = profile.v11ab_build_kwargs(args)
    ag3s, ag3s_info = profile.build_panda_ag3s(kw=kw)
    served_policy = policy
    if attention_model is not None:
        from AG3S.trajopt.attention_policy import AttentionPolicy

        served_policy = AttentionPolicy(policy, attention_model, subtask_probe=None)
    constraint = ag3s.constraint_robot_model
    kin = pa.PandaKinematics(constraint)
    chunk_policy = ChunkToJointPolicy(served_policy, kin)
    to_config = _replace_horizon_limits(ss.trajopt_config_from_args(args))
    attention_fn = ss.attention_extractor(alias=ATTENTION_ALIAS, cameras=ATTENTION_CAMERAS,
                                          transform=lambda _cam, grid: pat.unmirror_grid(np.asarray(grid)))
    mode = conversion_mode()
    # SL5j — None without --record-constraints (then the SafePolicy arguments are the pre-SL5j ones)
    recorder = constraint_recorder(args, to_config, conversion=mode, ag3s_info=ag3s_info)
    safe = SafePolicy(chunk_policy, ag3s=ag3s, to_config=to_config, attention_fn=attention_fn, latch=panda_latch_config(),
                      grasp_links=dict(profile.PANDA_GRASP_LINKS), hold_links=dict(profile.PANDA_HOLD_LINKS),
                      static_geometry=None, recorder=recorder, shadow=bool(getattr(args, "shadow", False)),
                      layout=pa.panda_chunk_layout(constraint.joint_names), gripper_columns=GRIPPER_COLUMNS,
                      finger_map=FingerJointMap(constraint, convention=panda_gripper_convention()), hands=HANDS)
    grasp_contact = ss.apply_grasp_contact_flags(safe, args)  # V11AB-pb4 / pb5 (all off = V11AB)
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
    if recorder is not None:  # key only when recording (the panda_server_config.json of other runs is unchanged)
        from AG3S.trajopt.safe_policy import _jsonable

        recorder.meta["grasp_contact"] = _jsonable(grasp_contact)
        info["record_constraints"] = {"dir": str(args.record_constraints), "run_dir": str(recorder.run_dir),
                                      "esdf_mode": recorder.esdf_mode, "background": recorder.background,
                                      "ag3s_version": recorder.meta.get("ag3s_version")}
    return PandaSafeServer(safe, chunk_policy, mode=mode), info


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
