"""One recorded LIBERO policy call -> ``SceneObservation`` + its recorded raw action chunk (smoke input).

Reads a rollout run recorded with ``--capture-geometry --capture-segmentation`` through A's
``eval/offline_grounding.Episode`` (OpenCV frame, wire depth, K, T_base_cam, GT segmentation labels). The GT labels
fill **privileged** masks only (``privileged/target_mask``, ``privileged/robot_mask``, ``privileged/ignore_mask``);
a non-privileged run passes a grounding provider's mask as ``extras["target_mask"]`` instead.
"""
from __future__ import annotations

import pathlib

import numpy as np

from Baseline.paper_baselines.methods.interface import PRIVILEGED_PREFIX, SceneObservation

SL3_DEV_RUN = "20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0"


def quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q, np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_recorded_call(run: str | pathlib.Path = SL3_DEV_RUN, episode: int | str = 0, call: int = 1, *,
                       ignore_classes=("support", "mount", "destination"), cameras=("head", "left_wrist")):
    """``(SceneObservation, raw_chunk (H, 7), meta)`` of policy call ``call`` of an episode of ``run``."""
    from Baseline.paper_baselines.eval import metrics as M
    from Baseline.paper_baselines.eval import offline_grounding as og

    run_dir = og.resolve_run(str(run))
    eps = og.load_episodes(run_dir)
    ep_row = eps[episode] if isinstance(episode, int) else next(e for e in eps if e["stem"] == episode)
    ep = og.Episode(ep_row, cameras=cameras)
    t = int(ep.chunk_t[call])
    fr = ep.frame(t)
    with np.load(ep_row["npz_path"], allow_pickle=False) as z:
        chunk = np.asarray(z["action_chunks"][call], np.float64)
        eef_pos = np.asarray(z["obs_robot0_eef_pos"][t], np.float64)
        eef_quat = np.asarray(z["obs_robot0_eef_quat"][t], np.float64)
        q = np.asarray(z["obs_robot0_joint_pos"][t], np.float64)
        grip = np.asarray(z["obs_robot0_gripper_qpos"][t], np.float64)
        prev = np.asarray(z["obs_robot0_eef_pos"][max(t - 1, 0)], np.float64)
    with np.load(ep_row["geom_path"], allow_pickle=False) as g:
        T_wb = np.asarray(g["T_world_base"][fr["geom_row"]], np.float64)
    T_we = np.eye(4)
    T_we[:3, :3], T_we[:3, 3] = quat_xyzw_to_R(eef_quat), eef_pos
    ee_base = np.linalg.inv(T_wb) @ T_we
    cls = M.CLASSES
    tmask = {c: fr["labels"][c] == cls.index("target") for c in cameras}
    rmask = {c: np.isin(fr["labels"][c], [cls.index("robot")]) for c in cameras}
    imask = {c: np.isin(fr["labels"][c], [cls.index(k) for k in ignore_classes]) for c in cameras}
    extras = {PRIVILEGED_PREFIX + "target_mask": tmask, PRIVILEGED_PREFIX + "robot_mask": rmask,
              PRIVILEGED_PREFIX + "ignore_mask": imask, "T_world_base": T_wb,
              "prev_ee_step_world": eef_pos - prev, "finger_opening_m": float(abs(grip[0])),
              "source": {"run": run_dir.name, "episode": ep.stem, "call": int(call), "obs_index": t}}
    obs = SceneObservation(rgb=fr["rgb"], depth=fr["depth_m"], intrinsics=fr["K"], T_base_cam=fr["T"], q=q,
                           ee_pose=ee_base, gripper=float(abs(grip[0])), instruction=ep.meta.get("task_description", ""),
                           target_phrase=str(ep.target["target"]), extras=extras)
    meta = {"run": run_dir.name, "episode": ep.stem, "call": int(call), "obs_index": t, "target": ep.target["target"],
            "destination": ep.target.get("destination"), "ignore_classes": list(ignore_classes),
            "npz": str(ep_row["npz_path"]), "geom": str(ep_row["geom_path"])}
    return obs, chunk, meta


__all__ = ["SL3_DEV_RUN", "load_recorded_call", "quat_xyzw_to_R"]
