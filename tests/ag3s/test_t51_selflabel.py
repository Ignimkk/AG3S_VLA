"""T51 grasp-event self label (`t51_selflabel`) and the 3D self fields (`t51_replay.self_fields`).

Pinned:
  - projection geometry: a known camera and point -> the pinhole pixel; a point behind the camera is
    not projected; the projected sphere centre matches the centroid of a MuJoCo segmentation render of
    that sphere (TransportScene's own `intrinsics` / `T_base_cam`, so the convention is the one the GT
    cache and the attention lifting use);
  - disc rasterisation (area, coverage sums), GT centroid, patch lookup;
  - grasp-event detection and the contact predicate (synthetic sequences + a two-finger MJCF);
  - object radius from pad separation; 3D self identity.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t51_replay as rp
from AG3S.ag3s.experiments.studies import t51_selflabel as sl


def _cam(f=100.0, cx=320.0, cy=240.0):
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def test_project_known_camera():
    K = _cam()
    T = np.eye(4)  # camera at the base origin, looking along +z (OpenCV)
    u, v, z = sl.project([0.1, -0.05, 2.0], K, T)
    assert (u, v, z) == pytest.approx((320 + 100 * 0.05, 240 - 100 * 0.025, 2.0))
    # rotate the camera 90 deg about base z and move it; the same point in its frame is reproduced
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    T2 = np.eye(4)
    T2[:3, :3], T2[:3, 3] = R, [1.0, 2.0, 0.5]
    p_cam = np.array([0.2, 0.1, 1.5])
    p_base = R @ p_cam + T2[:3, 3]
    u, v, z = sl.project(p_base, K, T2)
    assert (u, v, z) == pytest.approx((320 + 100 * 0.2 / 1.5, 240 + 100 * 0.1 / 1.5, 1.5))
    u, v, z = sl.project([0.0, 0.0, -1.0], K, T)
    assert np.isnan(u) and np.isnan(v) and z < 0


def test_to_frame_inverts_pose():
    T = np.eye(4)
    T[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    T[:3, 3] = [1, 2, 3]
    p = np.array([0.3, -0.2, 0.7])
    assert sl.to_frame(T, T[:3, :3] @ p + T[:3, 3]) == pytest.approx(p)


def test_disc_coverage_area_and_patches():
    cov, px = sl.disc_coverage(320.0, 240.0, 50.0)
    assert px == pytest.approx(np.pi * 50 ** 2, rel=0.02)
    assert cov.shape == (16, 16)
    # the coverage integrates back to the area: sum(cov) * patch area (30 x 40 px)
    assert cov.sum() * 30 * 40 == pytest.approx(px, rel=1e-9)
    assert cov[8, 8] == 1.0 and cov[0, 0] == 0.0
    # a disc half out of the image counts only its in-image half
    _, px_edge = sl.disc_coverage(0.0, 240.0, 50.0)
    assert px_edge == pytest.approx(px / 2, rel=0.05)
    z, n = sl.disc_coverage(float("nan"), 1.0, 5.0)
    assert n == 0 and not z.any()


def test_gt_centroid_and_patch_of():
    lab = np.full((240, 320), 6, np.uint8)
    lab[100:110, 50:60] = 0
    u, v, n = sl.gt_centroid(lab, stride=2)
    assert (u, v, n) == pytest.approx((110.0, 210.0, 400))
    assert sl.gt_centroid(np.full((4, 4), 3), stride=2)[2] == 0
    assert sl.patch_of(110.0, 210.0) == (7, 2)
    assert sl.patch_of(-1.0, 3.0) is None and sl.patch_of(float("nan"), 1.0) is None


def test_contact_state_and_detect_grasp():
    fingers = {"left": (10, 11), "right": (20, 21)}
    assert sl.contact_state({10, 11, 99}, fingers, {99}) == ("left", None)
    assert sl.contact_state({20, 21}, fingers, {99}) == ("right", "right")
    assert sl.contact_state({10, 99}, fingers, {99}) == (None, None)
    both = [None, None, "right", "right", "right", None]
    held = [None, None, None, None, "right", None]
    assert sl.detect_grasp(both, held) == (2, "right")
    assert sl.detect_grasp(both, held, event="held") == (4, "right")
    # both-finger contact that never lifts (a failed grasp) is not an event; a later success is
    assert sl.detect_grasp(["left", None, "right", "right"], [None, None, None, "right"]) == (2, "right")
    assert sl.detect_grasp(["left", None], [None, None]) is None
    with pytest.raises(ValueError):
        sl.detect_grasp([None], [None], event="touch")


def test_object_radius_clip():
    assert sl.object_radius([0, 0, 0], [0.072, 0, 0], 0.005) == pytest.approx(0.031)
    assert sl.object_radius([0, 0, 0], [0.01, 0, 0], 0.005) == sl.R_OBJ_MIN
    assert sl.object_radius([0, 0, 0], [0.5, 0, 0], 0.005) == sl.R_OBJ_MAX


def test_self_identity_and_fields():
    c = np.zeros(3)
    pts = np.array([[0.01, 0, 0], [0.02, 0, 0], [0.2, 0, 0]])
    r = sl.self_identity(pts, c, 0.05)
    assert (r["self_in_n"], r["self_out_n"], r["self_identity_correct"]) == (2, 1, True)
    r = sl.self_identity(pts[[0, 2]], c, 0.05)
    assert r["self_identity_correct"] is False  # 1:1 is not a majority
    assert sl.self_identity(np.zeros((0, 3)), c, 0.05)["self_identity_correct"] is False
    cs = types.SimpleNamespace(has_target=True, target=types.SimpleNamespace(points=pts))
    f = rp.self_fields(cs, labeled=True, centre=c, radius=0.05)
    assert f["self_labeled"] and f["self_identity_correct"]
    f = rp.self_fields(cs, labeled=False, centre=c, radius=0.05)
    assert not f["self_labeled"] and f["self_identity_correct"] is None
    cs0 = types.SimpleNamespace(has_target=False, target=None)
    assert rp.self_fields(cs0, labeled=True, centre=c, radius=0.05)["self_identity_correct"] is False


# --------------------------------------------------------------------------- MuJoCo checks

SCENE_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <body name="base" pos="0.3 -0.2 0.1" euler="0 0 30">
      <geom type="box" size="0.01 0.01 0.01" rgba="0 0 0 0" contype="0" conaffinity="0"/>
      <camera name="cam" pos="0 0 1.0" xyaxes="1 0 0 0 1 0" fovy="60"/>
    </body>
    <body name="ball" pos="0.45 -0.1 0.15">
      <geom type="sphere" size="0.05" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

GRIP_XML = """
<mujoco>
  <worldbody>
    <body name="table" pos="0 0 0"><geom type="box" size="0.3 0.3 0.02"/></body>
    <body name="crate" pos="2 0 0"><geom type="box" size="0.1 0.1 0.02"/></body>
    <body name="ee_finger_l1" pos="-0.035 0 0.06"><geom type="box" size="0.005 0.02 0.03"/></body>
    <body name="ee_finger_l2" pos="0.035 0 0.06"><geom type="box" size="0.005 0.02 0.03"/></body>
    <body name="ee_finger_r1" pos="1 0 1"><geom type="box" size="0.005 0.02 0.03"/></body>
    <body name="ee_finger_r2" pos="1.2 0 1"><geom type="box" size="0.005 0.02 0.03"/></body>
    <body name="apple" pos="0 0 0.06"><freejoint/><geom type="sphere" size="0.031"/></body>
  </worldbody>
</mujoco>
"""


def test_projection_matches_mujoco_render():
    mujoco = pytest.importorskip("mujoco")
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene

    m = mujoco.MjModel.from_xml_string(SCENE_XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    h, w = 120, 160
    scene = TransportScene.attach(m, d, height=h, width=w)
    K, T_bc = scene.intrinsics("cam"), scene.T_base_cam("cam")
    try:
        ren = mujoco.Renderer(m, h, w)
    except Exception as exc:  # pragma: no cover - no GL backend
        pytest.skip(f"rendering unavailable: {exc}")
    try:
        ren.enable_segmentation_rendering()
        ren.update_scene(d, camera="cam")
        seg = np.asarray(ren.render())
    finally:
        ren.close()
    ball = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
    gid = [g for g in range(m.ngeom) if m.geom_bodyid[g] == ball]
    is_ball = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[..., 0], gid)
    ys, xs = np.nonzero(is_ball)
    assert len(xs) > 50
    p_base = sl.to_frame(scene.body_pose("base"), d.xpos[ball])
    u, v, z = sl.project(p_base, K, T_bc)
    assert abs(u - (xs + 0.5).mean()) < 1.0 and abs(v - (ys + 0.5).mean()) < 1.0
    # the projected silhouette radius f r / z matches the rendered area (sphere close to on-axis)
    r_px = K[0, 0] * 0.05 / z
    assert len(xs) == pytest.approx(np.pi * r_px ** 2, rel=0.25)


def test_contacts_and_pads_on_two_finger_scene():
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_string(GRIP_XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
           for n in ("apple", "table", "crate", "ee_finger_l1", "ee_finger_l2",
                     "ee_finger_r1", "ee_finger_r2")}
    touch = sl.touching_bodies(m, d, bid["apple"])
    fingers = {h: tuple(bid[n] for n in sl.FINGER_BODIES[h]) for h in sl.HANDS}
    support = {bid["table"], bid["crate"]}
    # resting on the table (top at z=0.02, apple centre 0.06 - radius 0.031 = 0.029 -> no table
    # contact), squeezed by both left fingers (inner faces at +-0.03 < radius 0.031)
    assert {bid["ee_finger_l1"], bid["ee_finger_l2"]} <= touch
    assert sl.contact_state(touch, fingers, support) == ("left", "left")
    # lower the apple onto the table: both fingers and support -> grasping, not held
    d.qpos[2] = 0.05
    mujoco.mj_forward(m, d)
    touch = sl.touching_bodies(m, d, bid["apple"])
    assert bid["table"] in touch
    assert sl.contact_state(touch, fingers, support)[1] is None
    g1, g2 = (sl.pad_geom(m, bid[n]) for n in sl.FINGER_BODIES["left"])
    mid = 0.5 * (d.geom_xpos[g1] + d.geom_xpos[g2])
    assert mid == pytest.approx([0.0, 0.0, 0.06])
    half = float(min(np.min(m.geom_size[g]) for g in (g1, g2)))
    assert sl.object_radius(d.geom_xpos[g1], d.geom_xpos[g2], half) == pytest.approx(0.03)
