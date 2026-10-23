"""PAPER-METHODS C: common interface (``methods/interface.py``) and shared safety helpers (``safety_common.py``).

    CUDA_VISIBLE_DEVICES="" .venv-baselines/bin/python -m pytest Baseline/paper_baselines/methods/tests -q
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from Baseline.paper_baselines.methods import interface as I
from Baseline.paper_baselines.methods import safety_common as SC

RUN = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark/20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0")
needs_run = pytest.mark.skipif(not (RUN / "episodes").is_dir(), reason="SL3 dev recorded run not available")


def test_registry_resolves_safety_filters():
    for name in ("knows", "aegis", "multilink_cbf"):
        f = I.get_method(name, obstacle_source="ellipsoids")
        assert isinstance(f, I.SafetyFilter), name
        assert f.name == name
        prov = f.provenance()
        assert prov["method"] == name and "cite" in prov
    with pytest.raises(KeyError):
        I.get_method("no_such_method_xyz")
    assert {"knows", "aegis", "multilink_cbf"} <= set(I.available_methods())


def test_register_method_and_convention():
    @I.register_method("_c_dummy")
    class Dummy:
        name = "_c_dummy"

        def provenance(self):
            return {}

        def make_guidance_fn(self, obs, norm):
            return lambda x, t: x

    g = I.get_method("_c_dummy")
    assert isinstance(g, I.SamplingGuidance)


@pytest.mark.parametrize("norm", [{"q01": np.array([-1.0, 0.0, 2.0]), "q99": np.array([1.0, 4.0, 3.0])},
                                  {"mean": np.array([0.5, -1.0, 2.0]), "std": np.array([2.0, 0.5, 1.0])}])
def test_normalize_roundtrip_keeps_padding(norm):
    x = np.random.default_rng(0).normal(size=(2, 10, 5))  # 3 stats dims + 2 padded dims
    y = I.unnormalize_actions(x, norm)
    assert np.allclose(I.normalize_actions(y, norm), x)
    assert np.array_equal(y[..., 3:], x[..., 3:])


def test_validate_filter_result():
    A = np.zeros((4, 7))
    I.validate_filter_result(I.FilterResult(A.copy(), np.zeros(4, bool), {}), A)
    with pytest.raises(ValueError):
        I.validate_filter_result(I.FilterResult(A[:3], np.zeros(3, bool), {}), A)
    with pytest.raises(ValueError):
        I.validate_filter_result(I.FilterResult(A.copy(), np.zeros(4, int), {}), A)
    bad = A.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        I.validate_filter_result(I.FilterResult(bad, np.zeros(4, bool), {}), A)


def test_so3_helpers_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        v = rng.normal(size=3) * rng.uniform(0.01, 3.0)
        R = SC.axisangle_to_mat(v)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.allclose(SC.axisangle_to_mat(SC.mat_to_axisangle(R)), R, atol=1e-9)


def test_propagate_ee_uses_measured_gain():
    T = np.eye(4)
    a = np.array([1.0, -1.0, 0.5, 0, 0, 0.2, -1.0])
    T1 = SC.propagate_ee(T, a)
    assert np.allclose(T1[:3, 3], SC.POS_GAIN_M * a[:3])
    assert np.allclose(SC.mat_to_axisangle(T1[:3, :3]), SC.ROT_GAIN_RAD * a[3:6])


def test_backproject_pinhole():
    K = np.array([[100.0, 0, 15.5], [0, 100.0, 15.5], [0, 0, 1]])
    d = np.full((32, 32), 1.5)
    p = SC.backproject(d, K, np.eye(4))
    assert p.shape == (1024, 3) and np.allclose(p[:, 2], 1.5)
    assert len(SC.backproject(np.full((32, 32), 2.5), K, np.eye(4))) == 0  # beyond range_max
    m = np.zeros((32, 32), bool)
    m[15, 15] = True
    q = SC.backproject(d, K, np.eye(4), m)
    assert np.allclose(q, [[-0.0075, -0.0075, 1.5]])


@needs_run
def test_recorded_call_frames_agree_with_panda_fk():
    from AG3S.adapters.panda import model as panda

    obs, A, meta = SC.load_recorded_call(RUN, "libero_spatial_t0_e0", 3)
    assert A.shape == (10, 7) and meta["obs_index"] == 25
    assert obs.rgb["head"].shape == (256, 256, 3) and obs.depth["head"].shape == (256, 256)
    m = panda.build_panda_bare_model()
    q = panda.set_finger_parameters(m, obs.extras["robot_state"])
    T_fk = m.link_pose(q, "panda_grip_site")
    # the image-instant robot_state vs the step-return eef obs differ by one physics substep (< 2 mm)
    assert np.linalg.norm(T_fk[:3, 3] - obs.ee_pose[:3, 3]) < 2e-3
    # ee_pose keeps the robosuite ``robot0_eef_quat`` frame (what AEGIS reads); the URDF grip-site frame is that
    # frame turned by -90 deg about its z axis
    Rz = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(T_fk[:3, :3], obs.ee_pose[:3, :3] @ Rz, atol=2e-2)
    assert obs.privileged("target") == "akita_black_bowl_1"


@needs_run
def test_gt_seg_clouds_exclude_target_and_support():
    obs, _, _ = SC.load_recorded_call(RUN, "libero_spatial_t0_e0", 0)
    src = SC.ObstacleCloudSource("gt_seg")
    c = src(obs)
    assert c.privileged and c.source == "gt_seg"
    assert "akita_black_bowl_1" not in c.points and "plate_1" not in c.points  # task objects excluded
    assert "main_table" not in c.points
    assert {"glazed_rim_porcelain_ramekin_1", "wooden_cabinet_1"} <= set(c.points)
    ram = c.points["glazed_rim_porcelain_ramekin_1"]
    assert 0.0 < ram[:, 2].mean() < 0.1  # on the table, base frame
    src_t = SC.ObstacleCloudSource("gt_seg", exclude="target")
    assert "plate_1" in src_t(obs).points


class FakeDetector:
    """Returns a fixed box for every prompt (tests the AEGIS crop path without torch)."""

    def __init__(self, box):
        self.box = np.asarray(box, float).reshape(1, 4)

    def provenance(self):
        return {"detector": "fake"}

    def detect(self, rgb, text, box_threshold, text_threshold):
        return self.box.copy(), np.array([0.9]), [text]


@needs_run
def test_detector_crop_and_robot_filter():
    obs, _, _ = SC.load_recorded_call(RUN, "libero_spatial_t0_e0", 0)
    seg = obs.privileged("segmentation_geom")["head"]
    ids = obs.privileged("instances_geom_ids")["glazed_rim_porcelain_ramekin_1"]
    v, u = np.nonzero(np.isin(seg, ids))
    box = [u.min(), v.min(), u.max() + 1, v.max() + 1]
    src = SC.ObstacleCloudSource("detector", detector=FakeDetector(box), detector_prompt="obstacle_names")
    c = src(obs)
    assert c.privileged  # obstacle names are LIBERO metadata
    assert set(c.points) == set(obs.privileged("obstacle_names"))
    # a box over the gripper: the robot self-filter removes the arm points
    from AG3S.adapters.panda import model as panda

    m = panda.build_panda_bare_model()
    q = panda.set_finger_parameters(m, obs.extras["robot_state"])
    C, R = m.sphere_centers_numeric(q)
    full = np.ones((256, 256), bool)
    arm_box = [0, 0, 256, 256]
    pts_all = SC.backproject(obs.depth["head"], obs.intrinsics["head"], obs.T_base_cam["head"], full)
    kept = SC.ObstacleCloudSource("detector", detector=FakeDetector(arm_box)).remove_robot(obs, pts_all)
    gap = min(float((np.linalg.norm(kept - c, axis=1) - r).min()) for c, r in zip(C, R))
    assert len(kept) < len(pts_all) and gap > 0.0
