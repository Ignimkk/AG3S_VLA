"""T44 E2 harness (`ag3s/experiments/live/exclusion_e2.py`) — the CPU-checkable pieces.

The cuRobo path (four target-free tiers on one integrator) is the GPU smoke in `T44I0I1.impl.md`.
Here: the demo frame rule, the posed-frame scene, GT voxel labels, and the GT attention guard.
"""

from __future__ import annotations

import json
import pathlib
import types

import numpy as np
import pytest

E = pytest.importorskip("benchmark.ag3s.experiments.live.exclusion_e2")

LABEL = {n: i for i, n in enumerate(E.LABELS)}


def test_demo_frames_are_evenly_spaced_over_pregrasp_to_close():
    meta = {"events": {"pregrasp_start_frame": 68, "grasp_close_start_frame": 104}}
    assert E.demo_frames(meta, 4) == [68, 80, 92, 104]
    assert E.demo_frames(meta, 1) == [68]


def test_voxel_labels_majority_shared_nearest_and_hull():
    from scipy.spatial import Delaunay

    vs, origin = 0.005, np.zeros(3)
    pts = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.0, 0.001, 0.0],     # voxel 0: target ×3
                    [0.05, 0.0, 0.0], [0.051, 0.0, 0.0],                          # voxel 10: tgt + dis
                    [0.1, 0.0, 0.0]])                                             # near voxel 21
    lab = np.array([LABEL["target"]] * 3 + [LABEL["target"], LABEL["distractor"],
                                            LABEL["support"]], np.int16)
    cube = np.array([[x, y, z] for x in (0.2, 0.3) for y in (-0.05, 0.05) for z in (-0.05, 0.05)])
    ijk = np.array([[0, 0, 0], [10, 0, 0], [21, 0, 0], [50, 0, 0], [80, 0, 0]])
    out = E.voxel_labels(pts, lab, origin, vs, ijk,
                         hulls=[(Delaunay(cube), LABEL["distractor"])])
    assert [E.LABELS[i] for i in out] == ["target", "shared", "support", "distractor",
                                          "unlabelled"]


def test_nearest_other_xy():
    pf = {"target_fruit": "apple",
          "fruit_poses": {"apple": [0.5, 0.3, 0.85, 1, 0, 0, 0], "pear": [0.5, 0.37, 0.86, 1, 0, 0, 0],
                          "banana": [0.5, -0.3, 0.84, 1, 0, 0, 0]}}
    assert E.nearest_other_xy(pf) == pytest.approx(0.07)


def _scene():
    pytest.importorskip("mujoco")
    from benchmark.ag3s.experiments.sources.mujoco_source import TransportScene
    from benchmark.ag3s.runtime.asset_path import resolve_asset

    try:
        xml = resolve_asset("pi05_TO_hybrid/rby1_description/models/rby1a/mujoco/model_transport.xml",
                            what="transport model")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"16D transport model not found: {exc}")
    return TransportScene(str(xml), settle_steps=0)


def test_pose_posed_puts_fruits_arms_and_grippers_where_the_record_says():
    scene = _scene()
    try:
        st = np.zeros(16)
        st[0:7] = [-0.27, 0.37, -0.01, -1.92, 0.28, 1.48, -1.54]
        st[7] = 1.0
        st[8:15] = [-0.30, -0.40, -0.01, -1.93, -0.31, 1.48, -1.56]
        st[15] = 0.5
        pf = {"state16": st.tolist(), "target_fruit": "apple", "used_arm": "left",
              "fruit_poses": {"apple": [0.45, 0.30, 0.85, 1, 0, 0, 0],
                              "orange": [0.52, 0.30, 0.85, 1, 0, 0, 0],
                              "banana": [0.46, -0.30, 0.84, 1, 0, 0, 0],
                              "pear": [0.55, -0.30, 0.86, 1, 0, 0, 0]},
              "basket_pose": [0.52, -0.02, 0.88, 1, 0, 0, 0]}
        E.pose_posed(scene, pf)
        np.testing.assert_allclose(scene.body_position("apple"), [0.45, 0.30, 0.85], atol=1e-9)
        np.testing.assert_allclose(scene.body_position("crate"), [0.52, -0.02, 0.88], atol=1e-9)
        q = scene.data.qpos
        adr = lambda n: scene._qadr[n]                                     # noqa: E731
        assert q[adr("left_arm_3")] == pytest.approx(-1.92)
        assert q[adr("right_arm_4")] == pytest.approx(-0.31)
        assert q[adr("gripper_finger_l1")] == pytest.approx(E.RBY1_GRIPPER_OPEN)
        assert q[adr("gripper_finger_r1")] == pytest.approx(0.5 * E.RBY1_GRIPPER_OPEN)
    finally:
        scene.close()


def test_gt_attention_skips_cameras_that_do_not_see_the_target():
    scene = types.SimpleNamespace(
        mujoco=types.SimpleNamespace(mj_name2id=lambda m, t, n: 7,
                                     mjtObj=types.SimpleNamespace(mjOBJ_BODY=1)),
        model=None)
    K = np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]])
    seen = types.SimpleNamespace(body_ids=np.full((48, 64), 7), hw=(48, 64),
                                 camera_intrinsics=K, T_base_cam=np.eye(4))
    unseen = types.SimpleNamespace(body_ids=np.full((48, 64), 3), hw=(48, 64),
                                   camera_intrinsics=K, T_base_cam=np.eye(4))
    out = E.gt_attention(scene, {"a": seen, "b": unseen}, "apple", np.array([0.0, 0.0, 1.0]))
    assert set(out) == {"a"}


def test_cli_inputs_are_exclusive_in_practice(tmp_path):
    ap = E._parser()
    a = ap.parse_args(["--demo-episodes", "0,5", "--out", str(tmp_path)])
    assert a.demo_episodes == "0,5" and a.run is None and a.demo_frames_per_episode == 4
    pf = tmp_path / "pf.json"
    pf.write_text(json.dumps([]))
    a = ap.parse_args(["--posed-frames", str(pf), "--out", str(tmp_path), "--identity", "gt_label"])
    assert a.identity == "gt_label" and isinstance(a.posed_frames, pathlib.Path)
