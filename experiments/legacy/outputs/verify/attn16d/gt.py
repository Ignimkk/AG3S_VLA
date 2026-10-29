"""Ground truth for the 16D attention-cell re-validation.

For each recorded frame and each of the three policy cameras:
  * segmentation at the *policy* camera setup: 299x224 render -> per-body float mask ->
    PIL BILINEAR resize to 224x224 (the same op `resize_policy_image` applies to RGB) ->
    14x14-pixel patch mean = coverage on the 16x16 SigLIP grid.
  * segmentation at 640x480 (the 14D `attention_report.ground_truth` geometry; `patch_coverage`)
    -> used for the px visibility rules (min-target-px 200, min-body-px 100) exactly as 14D, and
    as a cross-check of the coverage.
  * RGB re-render at the policy setup (reflections off, 299x224 -> 224x224) compared with the
    recorded policy image (mean |diff|), to show the qpos replay reproduces what the policy saw.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} .venv-openpi-live/bin/python gt.py NAME RECORD_DIR
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import sys
import pathlib

import numpy as np
import mujoco
from PIL import Image

from AG3S.ag3s.experiments.sources.policy_record import (
    ATTENTION_GRID, CAMERA_BINDINGS, POLICY_CAMERA_NAMES, load_run, patch_coverage, pose_scene,
    replay_scene)
from AG3S.ag3s.experiments.reports.attention_report import CANDIDATE_BODIES

OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/attn16d'))
SRC_H, SRC_W = 224, 299  # POLICY_SOURCE_HEIGHT / WIDTH (rby1_manipulation.data.recording)
POL = 224


def seg_bodies(renderer, model, data, camera):
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera)
    seg = np.asarray(renderer.render())
    renderer.disable_segmentation_rendering()
    oid, otype = seg[..., 0], seg[..., 1]
    is_geom = otype == int(mujoco.mjtObj.mjOBJ_GEOM)
    body = np.full(oid.shape, -1, np.int32)
    body[is_geom] = model.geom_bodyid[oid[is_geom]]
    return body


def main(name: str, rec_dir: str) -> None:
    rec = load_run(rec_dir)
    scene = replay_scene(rec, height=480, width=640)  # nq + joint-order check
    model, data = scene.model, scene.data
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in CANDIDATE_BODIES]
    assert all(b >= 0 for b in body_ids), body_ids
    r_pol = mujoco.Renderer(model, height=SRC_H, width=SRC_W)
    r_640 = mujoco.Renderer(model, height=480, width=640)
    n, nc, nb = len(rec), len(POLICY_CAMERA_NAMES), len(body_ids)
    cov_pol = np.zeros((n, nc, nb, ATTENTION_GRID, ATTENTION_GRID), np.float32)
    area_pol = np.zeros((n, nc, nb), np.float32)
    cov_640 = np.zeros_like(cov_pol)
    area_640 = np.zeros((n, nc, nb), np.float32)
    rgb_mad = np.zeros((n, nc), np.float32)
    rgb_render = np.zeros((n, nc, POL, POL, 3), np.uint8)
    ph = POL // ATTENTION_GRID
    for i, step in enumerate(rec):
        pose_scene(scene, step)
        for ci, key in enumerate(POLICY_CAMERA_NAMES):
            cam = CAMERA_BINDINGS[key][0]
            # RGB, as pi05_infer.render_cam(match_rby1_dataset=True)
            r_pol.update_scene(data, camera=cam)
            r_pol.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            img = r_pol.render()
            img = np.asarray(Image.fromarray(img).resize((POL, POL), resample=Image.Resampling.BILINEAR))
            rgb_render[i, ci] = img
            rgb_mad[i, ci] = np.abs(img.astype(np.float32) - step.images[key].astype(np.float32)).mean()
            # segmentation, policy geometry
            lab = seg_bodies(r_pol, model, data, cam)
            for k, b in enumerate(body_ids):
                m = Image.fromarray((lab == b).astype(np.float32), mode="F")
                m = np.asarray(m.resize((POL, POL), resample=Image.Resampling.BILINEAR), np.float32)
                m = np.clip(m, 0.0, 1.0)
                area_pol[i, ci, k] = m.sum()
                cov_pol[i, ci, k] = m.reshape(ATTENTION_GRID, ph, ATTENTION_GRID, ph).mean(axis=(1, 3))
            # segmentation, 640x480 (14D geometry)
            lab640 = seg_bodies(r_640, model, data, cam)
            cov_640[i, ci], area_640[i, ci] = patch_coverage(lab640, body_ids)
        print(f"{name} frame {i + 1}/{n}  rgb_mad={rgb_mad[i].round(2).tolist()}", flush=True)
    t_step = np.asarray([s.t_step for s in rec], np.int64)
    state = np.stack([s.state for s in rec])
    np.savez_compressed(
        OUT / f"gt_{name}.npz", cov_pol=cov_pol, area_pol=area_pol, cov_640=cov_640,
        area_640=area_640, rgb_mad=rgb_mad, rgb_render=rgb_render, t_step=t_step, state=state,
        bodies=np.asarray(CANDIDATE_BODIES), cameras=np.asarray(POLICY_CAMERA_NAMES),
        record=np.asarray(str(pathlib.Path(rec_dir).resolve())), prompt=np.asarray(rec.prompt))
    print("wrote", OUT / f"gt_{name}.npz", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
