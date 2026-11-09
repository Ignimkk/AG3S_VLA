"""KNOWS F3 (track freeze while the arm intervenes, paper Sec. 3.2) and F4a (attention target exclusion, Eq. 2-4).

Synthetic scenes, no simulator: one pinhole camera at the base origin looking along +z (OpenCV frame), a bump-shaped
object on a plane at 1 m, Panda collision spheres replaced by a chosen sphere (``robot_spheres`` patched in the adapter
module). ``.venv-baselines``::

    PYTHONPATH=$PWD .venv-baselines/bin/python -m pytest -q Baseline/paper_baselines/methods/knows/tests/test_knows_f3_f4a.py
"""
from __future__ import annotations

import numpy as np
import pytest

from Baseline.paper_baselines.methods.interface import PRIVILEGED_PREFIX, SceneObservation
from Baseline.paper_baselines.methods.knows import adapter as KA
from Baseline.paper_baselines.methods.knows.cbf.ellipsoid import Ellipsoid
from Baseline.paper_baselines.methods.safety_common import ObstacleCloudSource, ellipsoid_silhouette

H = W = 64
K = np.array([[100.0, 0, 31.5], [0, 100.0, 31.5], [0, 0, 1.0]])
T_CAM = np.eye(4)  # camera at the base origin, optical axis +z
FAR_SPHERES = (np.array([[5.0, 5.0, 5.0]]), np.array([0.02]))


def scene(col0: int, *, seg_id: int = 5):
    """Depth (bump of 3 cm on a 1 m plane) and GT seg of one object whose mask starts at column ``col0``."""
    d = np.ones((H, W))
    seg = np.full((H, W), -1, np.int32)
    v, u = np.mgrid[0:H, 0:W]
    r = np.hypot((u - (col0 + 8)) / 8.0, (v - 32) / 8.0)
    inside = r <= 1.0
    d[inside] = 1.0 - 0.03 * np.sqrt(1.0 - r[inside] ** 2)
    seg[inside] = seg_id
    return d, seg


def obs_of(col0: int, extras=None):
    d, seg = scene(col0)
    ee = np.eye(4)
    ee[:3, 3] = [0.0, 0.6, 0.2]  # far from the object: the CBF never binds
    ex = {PRIVILEGED_PREFIX + "segmentation_geom": {"head": seg},
          PRIVILEGED_PREFIX + "instances_geom_ids": {"bowl": [5]},
          PRIVILEGED_PREFIX + "target": None, PRIVILEGED_PREFIX + "destination": None, **(extras or {})}
    return SceneObservation(rgb={"head": np.zeros((H, W, 3), np.uint8)}, depth={"head": d}, intrinsics={"head": K},
                            T_base_cam={"head": T_CAM}, q=np.zeros(7), ee_pose=ee, gripper=0.08,
                            instruction="pick up the bowl", extras=ex)


def centre(f, obs):
    f.filter(obs, np.zeros((1, 7)))
    return f.cur_c["bowl"].copy()


@pytest.fixture
def spheres(monkeypatch):
    box = {"s": FAR_SPHERES}
    monkeypatch.setattr(KA, "robot_spheres", lambda obs, cache=None: box["s"])
    return box


# --- geometry helper -------------------------------------------------------------------------------------------------
def test_silhouette_is_the_projected_ellipsoid():
    sil = ellipsoid_silhouette([0.0, 0.0, 1.0], np.eye(3) * 0.05 ** 2, K, T_CAM, (H, W))
    # a sphere of radius 5 cm at 1 m: image radius ~ f r / sqrt(d^2 - r^2) = 5.0 px around the principal point
    v, u = np.nonzero(sil)
    assert abs(u.mean() - 31.5) < 0.6 and abs(v.mean() - 31.5) < 0.6
    assert 60 < sil.sum() < 100  # pi * 5^2 = 78.5
    behind = ellipsoid_silhouette([0.0, 0.0, -1.0], np.eye(3) * 0.05 ** 2, K, T_CAM, (H, W))
    assert not behind.any()
    inside = ellipsoid_silhouette([0.0, 0.0, 0.0], np.eye(3) * 0.05 ** 2, K, T_CAM, (H, W))
    assert inside.all()


# --- F3 ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("source", ["gt_seg", "given"])
def test_track_freezes_only_while_the_arm_is_over_it(spheres, source):
    f = KA.KnowsSafetyFilter(source, track_freeze="arm", diagnostics=False)
    g = KA.KnowsSafetyFilter(source, track_freeze="off", diagnostics=False)

    def o(col):
        ob = obs_of(col)
        if source == "given":  # the depth-lift arm: the cloud comes in as points (centroid association)
            from Baseline.paper_baselines.methods.safety_common import backproject

            seg = ob.extras[PRIVILEGED_PREFIX + "segmentation_geom"]["head"]
            ob.extras["obstacle_points"] = {"bowl": backproject(ob.depth["head"], K, T_CAM, seg == 5)}
        return ob

    if source == "given":  # 'given' names are re-associated by centroid: keep the track name stable for the check
        for flt in (f, g):
            flt.assoc_gate_m = 1.0
    c0 = centre(f, o(10)), centre(g, o(10))
    c1 = centre(f, o(14)), centre(g, o(14))  # moved 4 px, arm far: both follow
    assert np.allclose(c1[0], c1[1]) and c1[0][0] > c0[0][0] + 0.02
    spheres["s"] = (np.array([[c1[0][0], 0.0, 0.975]]), np.array([0.02]))  # the arm over the object
    c2 = centre(f, o(20)), centre(g, o(20))
    assert np.allclose(c2[0], c1[0])  # F3: frozen at its last position
    assert c2[1][0] > c1[1][0] + 0.03  # off: the partial / moved mask is followed
    assert f.perception_info["frozen"] == {"bowl": {"head": f.perception_info["frozen"]["bowl"]["head"]}}
    assert "frozen" not in g.perception_info
    spheres["s"] = FAR_SPHERES  # the arm leaves: the track is re-measured
    c3 = centre(f, o(20))
    assert c3[0] > c1[0][0] + 0.03 and f.perception_info["frozen"] == {}


def test_freeze_off_equals_freeze_on_when_the_arm_never_intervenes(spheres):
    f = KA.KnowsSafetyFilter("gt_seg", track_freeze="arm")
    g = KA.KnowsSafetyFilter("gt_seg", track_freeze="off")
    for col in (10, 12, 15, 9):
        ra, rb = f.filter(obs_of(col), np.full((1, 7), 0.1)), g.filter(obs_of(col), np.full((1, 7), 0.1))
        assert np.array_equal(ra.actions, rb.actions)
        assert ra.info["obstacles"] == rb.info["obstacles"]


def test_provenance_declares_the_rules():
    p = KA.KnowsSafetyFilter("gt_seg").provenance()
    assert p["track_freeze"]["rule"] == "arm" and p["track_freeze"]["margin_m"] == 0.02
    assert p["track_freeze"]["min_pixels"] == 1 and p["target_exclusion"] == {"rule": "source"}
    p = KA.KnowsSafetyFilter("gt_seg", track_freeze="off").provenance()
    assert p["track_freeze"] == {"rule": "off (port before F3)"}
    p = KA.KnowsSafetyFilter("detector", detector=object(), target_exclusion="attention").provenance()
    t = p["target_exclusion"]
    assert (t["layer"], t["head"], t["K_policy_calls"], t["delta"], t["beta"]) == (12, 3, 5, 7.3e-6, -1.0)
    assert p["obstacle"]["target_box"] is False
    with pytest.raises(ValueError):
        KA.KnowsSafetyFilter("gt_seg", target_exclusion="attention")


# --- F4a ---------------------------------------------------------------------------------------------------------------
def two_tracks():
    return {"left": Ellipsoid(np.array([-0.15, 0.0, 1.0]), np.eye(3) * 0.04 ** 2),
            "right": Ellipsoid(np.array([0.15, 0.0, 1.0]), np.eye(3) * 0.04 ** 2)}


def head_obs(grid=None, seq=0):
    d = np.ones((256, 256))
    K2 = np.array([[200.0, 0, 127.5], [0, 200.0, 127.5], [0, 0, 1.0]])
    ex = {} if grid is None else {KA.ATTENTION_KEY: {"grid": grid, "seq": seq}}
    return SceneObservation(rgb={"head": np.zeros((256, 256, 3), np.uint8)}, depth={"head": d}, intrinsics={"head": K2},
                            T_base_cam={"head": T_CAM}, q=np.zeros(7), ee_pose=np.eye(4), gripper=0.0,
                            instruction="", extras=ex)


def grid_left(hi=4e-3, lo=1e-4):
    g = np.full((16, 16), lo)
    g[:, :8] = hi
    return g


def test_attention_target_eq2_to_4():
    at = KA.AttentionTarget()
    tracks = two_tracks()
    assert at.update(head_obs(grid_left()), tracks) == "left"
    assert at.info["gap"] >= KA.ATTENTION_DELTA and at.info["frames"] == 1
    assert at.info["alpha"]["left"] == pytest.approx(at.info["alpha"]["right"])  # symmetric masks, 224-px units
    # a filter step of the same policy call (same seq): the call's target is kept, no new frame
    assert at.update(head_obs(grid_left(), seq=0), tracks) == "left" and len(at.frames) == 1
    assert at.update(head_obs(None), tracks) == "left"
    # uniform attention: equal densities -> gap 0 < delta -> no target (Eq. 4)
    at2 = KA.AttentionTarget()
    assert at2.update(head_obs(np.full((16, 16), 1e-3)), tracks) is None and at2.info["reason"] == "gap < delta"
    # one track: no lead to measure -> no target
    at3 = KA.AttentionTarget()
    assert at3.update(head_obs(grid_left()), {"left": tracks["left"]}) is None


def test_attention_window_is_the_last_k_calls():
    at = KA.AttentionTarget(k=2)
    tracks = two_tracks()
    for s in range(3):
        at.update(head_obs(grid_left(), seq=s), tracks)
    assert len(at.frames) == 2 and at.target == "left"
    right = grid_left()[:, ::-1]
    at.update(head_obs(right, seq=3), tracks)  # window = {left-call, right-call}: densities equal -> none
    assert at.target is None
    at.update(head_obs(right, seq=4), tracks)
    assert at.target == "right"


def test_filter_drops_the_attention_target_from_the_obstacle_set():
    f = KA.KnowsSafetyFilter("detector", detector=object(), target_exclusion="attention", diagnostics=False)
    f._obstacles = lambda obs: two_tracks()
    r = f.filter(head_obs(grid_left()), np.zeros((1, 7)))
    assert r.info["target"]["target"] == "left" and sorted(r.info["obstacles"]) == ["right"]
    r = f.filter(head_obs(np.full((16, 16), 1e-3), seq=1), np.zeros((1, 7)))  # window {left, uniform}: still left
    assert r.info["target"]["target"] == "left"
    f.reset()
    r = f.filter(head_obs(np.full((16, 16), 1e-3)), np.zeros((1, 7)))
    assert r.info["target"]["target"] is None and sorted(r.info["obstacles"]) == ["left", "right"]


class FakeDetector:
    """Target phrase -> one box; 'object' -> the target's box and another one."""

    def detect(self, rgb, text, box_threshold, text_threshold):
        if text == "object":
            return np.array([[10.0, 10, 30, 30], [40, 40, 60, 60]]), np.array([0.9, 0.8]), ["object", "object"]
        return np.array([[11.0, 10, 30, 31]]), np.array([0.7]), [text]


def test_source_target_box_option():
    ob = head_obs()
    ob.depth["head"] = ob.depth["head"] - 0.01 * np.random.default_rng(0).random((256, 256))
    ob.target_phrase = "bowl"
    kw = dict(detector_prompt="all_objects", crop_mode="box", robot_filter=False, detector=FakeDetector(),
              cameras=("head",))
    a = ObstacleCloudSource("detector", **kw)(ob)
    b = ObstacleCloudSource("detector", target_box=False, **kw)(ob)
    assert len(a.points) == 1 and len(b.points) == 2
    assert "target_box" not in ObstacleCloudSource("detector", **kw).provenance()
