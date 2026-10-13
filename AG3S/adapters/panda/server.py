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

    def __init__(self, safe, chunk_policy: ChunkToJointPolicy):
        self._policy = safe
        self.chunk_policy = chunk_policy

    @property
    def metadata(self):
        return getattr(self._policy, "metadata", {})

    def reset(self):
        self._policy.reset()

    def infer(self, obs: dict, **kwargs) -> dict:
        response = dict(self._policy.infer(adapt_request(obs), **kwargs))
        conv = self.chunk_policy.last
        refined, reference = response["actions"], response.get("actions_reference")
        if conv is None or reference is None:
            raise RuntimeError("SafePolicy returned no reference chunk for the Panda conversion")
        actions, rec = pa.joint_chunk_to_delta_chunk(self.chunk_policy.kin, conv, refined, reference)
        response["panda/joint_refined"] = np.asarray(refined, np.float32)
        response["panda/joint_reference"] = np.asarray(reference, np.float32)
        response["panda/actions_policy"] = np.asarray(conv.delta)
        response["panda/conversion"] = {**conv.record(), **rec}
        response["panda/mode"] = "vla_v11ab"
        response["actions"] = actions
        return response


def _replace_horizon_limits(to_config):
    h = dataclasses.replace(to_config.horizon, **PANDA_HORIZON)
    lim = dataclasses.replace(to_config.limits, source="urdf", model_xml=None,
                              default_acceleration=PANDA_DEFAULT_ACCELERATION)
    return dataclasses.replace(to_config, horizon=h, limits=lim)


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
    safe = SafePolicy(chunk_policy, ag3s=ag3s, to_config=to_config, attention_fn=attention_fn,
                      grasp_links=dict(profile.PANDA_GRASP_LINKS), hold_links=dict(profile.PANDA_HOLD_LINKS),
                      static_geometry=None, recorder=None, shadow=bool(getattr(args, "shadow", False)),
                      layout=pa.panda_chunk_layout(constraint.joint_names), gripper_columns=GRIPPER_COLUMNS,
                      finger_map=FingerJointMap(constraint, convention=panda_gripper_convention()), hands=HANDS)
    info = {"ag3s": ag3s_info, "horizon": PANDA_HORIZON,
            "limits": {"source": "urdf", "urdf": str(pm.PANDA_URDF), "default_acceleration": PANDA_DEFAULT_ACCELERATION,
                       "franka_accel_limits": FRANKA_ACCEL_LIMITS},
            "to_config": {"horizon": dataclasses.asdict(to_config.horizon), "planned": to_config.horizon.planned},
            "attention": {"alias": ATTENTION_ALIAS, "cameras": ATTENTION_CAMERAS, "unmirror": True,
                          "cell": "V11AB (attention_policy._LAYER/_HEAD/_AGG/_DENOISE_STEP)",
                          "subtask_probe": None, "loaded": attention_model is not None},
            "hands": HANDS, "gripper_columns": GRIPPER_COLUMNS, "gripper_convention": panda_gripper_convention().record(),
            "conversion": pa.conventions(), "static_geometry": None,
            "v11ab_build_kwargs_not_applied": ["constraint_sphere_options", "finger_cover"]}
    return PandaSafeServer(safe, chunk_policy), info


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
    websocket_policy_server.WebsocketPolicyServer(policy=served, host="0.0.0.0", port=args.port,
                                                  metadata=getattr(served, "metadata", {})).serve_forever()
