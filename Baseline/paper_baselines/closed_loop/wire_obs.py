"""SafeLIBERO / LIBERO wire request (``benchmark/runners/libero_safe_client.py``) -> ``SceneObservation``.

What the request carries (SL1 / SL5a; ``benchmark/adapters/libero_panda/wire_pack.py``):

| key | content |
|---|---|
| ``observation/image``, ``observation/wrist_image`` | policy RGB 224 px = ``resize_with_pad(raw[::-1, ::-1])`` (agentview / eye-in-hand) |
| ``observation/state`` | ``eef_pos (world) + quat2axisangle(eef_quat) + gripper_qpos (2)`` |
| ``ag3s/cameras``, ``ag3s/depth/<cam>`` (uint16 mm), ``ag3s/K/<cam>``, ``ag3s/T_base_cam/<cam>``, ``ag3s/robot_state/<cam>`` (9) | depth camera geometry, OpenCV pixel frame |
| ``panda/robot_state`` (9), ``panda/T_world_base`` (4, 4), ``panda/prev_step_world`` (6) | Panda adapter keys |
| ``prompt`` | task instruction |

Frames follow ``methods/safety_common.load_recorded_call`` (the offline smoke input): RGB in the OpenCV frame
(``raw[::-1]``; here rebuilt from the policy image as ``policy[:, ::-1]`` resized to the depth size), depth metres,
``ee_pose`` = robosuite ``robot0_eef`` pose in the robot base frame. No privileged key is produced.
"""
from __future__ import annotations

import re

import numpy as np

from Baseline.paper_baselines.methods.interface import SceneObservation

PREFIX = "ag3s/"
WIRE_CAMERAS = {"head": "observation/image", "left_wrist": "observation/wrist_image"}
DEPTH_SCALE_MM = 1000.0
FINGER_OPEN_M = 0.04


def rotvec_to_mat(v) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(np.array(v, np.float64).reshape(3)).as_matrix()  # copy: msgpack arrays are read-only


def policy_rgb_to_opencv(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Policy image (``resize(raw[::-1, ::-1])``) -> OpenCV-frame RGB (``raw[::-1]``) at ``hw`` (bilinear)."""
    from PIL import Image

    a = np.ascontiguousarray(np.asarray(img, np.uint8)[:, ::-1])
    if a.shape[:2] != tuple(hw):
        a = np.asarray(Image.fromarray(a).resize((int(hw[1]), int(hw[0])), Image.BILINEAR))
    return np.ascontiguousarray(a, np.uint8)


_TARGET_RULES = (
    r"^pick up (?:both )?(?:the )?(.+?)(?: (?:between|next to|on|in|from|that|which|and|at|to)\b.*)?$",
    r"^put both (?:the )?(.+?) and .*$",
    r"^put (?:both )?(?:the )?(.+?)(?: (?:on|in|into|onto|inside|to|at|under|next)\b.*)?$",
    r"^place (?:both )?(?:the )?(.+?)(?: (?:on|in|into|onto|and|to|at)\b.*)?$",
    r"^push the (.+?)(?: (?:to|into|towards|on)\b.*)?$",
    r"^(?:open|close) the (?:top |middle |bottom )?(.+?)(?: (?:of|and|on)\b.*)?$",
    r"^turn (?:on|off) the (.+?)(?: (?:and)\b.*)?$",
    r"^(?:stack|move|lift|grab|take) the (.+?)(?: (?:on|in|into|onto|and|to|from)\b.*)?$",
)


def target_phrase_from_instruction(instruction: str) -> str | None:
    """Noun phrase of the manipulated object, parsed from the instruction text (no simulator metadata)."""
    s = " ".join(str(instruction).lower().strip().rstrip(".").split())
    for rule in _TARGET_RULES:
        m = re.match(rule, s)
        if m:
            return m.group(1).strip() or None
    return None


def scene_observation(request: dict, *, last_gripper_cmd: float | None = None, seq: int | None = None) -> SceneObservation:
    """One policy request -> ``SceneObservation`` (base frame, OpenCV pixel frame, non-privileged)."""
    cams = [str(c) for c in request.get(PREFIX + "cameras", [])]
    if not cams:
        raise KeyError("the request has no ag3s/cameras (client must run with the camera geometry)")
    depth, K, T, rgb = {}, {}, {}, {}
    for cam in cams:
        d = np.asarray(request[f"{PREFIX}depth/{cam}"])
        depth[cam] = d.astype(np.float64) / DEPTH_SCALE_MM
        K[cam] = np.array(request[f"{PREFIX}K/{cam}"], np.float64)
        T[cam] = np.array(request[f"{PREFIX}T_base_cam/{cam}"], np.float64)
        key = WIRE_CAMERAS.get(cam)
        if key in request:
            rgb[cam] = policy_rgb_to_opencv(request[key], d.shape[:2])
    # np.array (copies): the openpi msgpack decoder hands out read-only arrays; methods / scipy may write
    state = np.array(request["observation/state"], np.float64).reshape(-1)
    T_wb = np.array(request["panda/T_world_base"], np.float64).reshape(4, 4)
    rs9 = np.array(request.get("panda/robot_state", request[f"{PREFIX}robot_state/{cams[0]}"]), np.float64).reshape(-1)
    T_we = np.eye(4)
    T_we[:3, :3], T_we[:3, 3] = rotvec_to_mat(state[3:6]), state[:3]
    ee_pose = np.linalg.inv(T_wb) @ T_we
    prev = np.array(request.get("panda/prev_step_world", np.zeros(6)), np.float64).reshape(-1)
    instruction = str(request.get("prompt", ""))
    extras = {"robot_state": rs9, "T_world_base": T_wb, "prev_ee_step_world": prev[:3].copy(),
              "finger_opening_m": float(np.clip(abs(rs9[7]) if len(rs9) > 7 else FINGER_OPEN_M, 0.0, FINGER_OPEN_M)),
              "gripper_closed": bool(last_gripper_cmd is not None and last_gripper_cmd > 0.0),
              "policy_state": state, "seq": seq, "source": "wire"}
    return SceneObservation(rgb=rgb, depth=depth, intrinsics=K, T_base_cam=T, q=rs9[:7].copy(), ee_pose=ee_pose,
                            gripper=float(state[6] - state[7]) if len(state) >= 8 else 0.0, instruction=instruction,
                            target_phrase=target_phrase_from_instruction(instruction), extras=extras)


# --- SceneObservation <-> framed dict (server <-> sidecar) ---------------------------------------------------------
def obs_to_dict(obs: SceneObservation) -> dict:
    return {"rgb": dict(obs.rgb), "depth": dict(obs.depth), "intrinsics": dict(obs.intrinsics),
            "T_base_cam": dict(obs.T_base_cam), "q": np.asarray(obs.q), "ee_pose": np.asarray(obs.ee_pose),
            "gripper": float(obs.gripper), "instruction": obs.instruction, "target_phrase": obs.target_phrase,
            "extras": dict(obs.extras)}


def obs_from_dict(d: dict) -> SceneObservation:
    return SceneObservation(rgb=dict(d["rgb"]), depth=dict(d["depth"]), intrinsics=dict(d["intrinsics"]),
                            T_base_cam=dict(d["T_base_cam"]), q=np.asarray(d["q"]), ee_pose=np.asarray(d["ee_pose"]),
                            gripper=float(d["gripper"]), instruction=d["instruction"], target_phrase=d.get("target_phrase"),
                            extras=dict(d.get("extras") or {}))


__all__ = ["WIRE_CAMERAS", "obs_from_dict", "obs_to_dict", "policy_rgb_to_opencv", "scene_observation",
           "target_phrase_from_instruction"]
