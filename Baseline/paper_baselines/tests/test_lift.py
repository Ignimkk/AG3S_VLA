"""lift: back-projection round trip with known K, T, and the policy-mirror pixel convention."""
import numpy as np
import pytest

from Baseline.paper_baselines import lift as L


def _rot(axis, ang):
    c, s = np.cos(ang), np.sin(ang)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _camera():
    K = np.array([[300.0, 0, 127.5], [0, 300.0, 127.5], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = _rot("x", 0.3) @ _rot("y", -0.2)
    T[:3, 3] = [0.4, -0.1, 0.9]
    return K, T


def test_round_trip_known_K_T():
    K, T = _camera()
    rng = np.random.default_rng(0)
    depth = 0.6 + 0.4 * rng.random((256, 256))
    mask = np.zeros((256, 256), bool)
    mask[100:140, 30:90] = True
    pts = L.lift_mask(mask, depth, K, T)
    assert len(pts) == mask.sum()
    u, v, z = L.project_points(pts, K, T)
    vv, uu = np.nonzero(mask)
    order = np.lexsort((uu, vv))
    np.testing.assert_allclose(np.sort(u), np.sort(uu[order].astype(float)), atol=1e-9)
    np.testing.assert_allclose(np.sort(v), np.sort(vv.astype(float)), atol=1e-9)
    np.testing.assert_allclose(np.sort(z), np.sort(depth[mask]), atol=1e-12)


def test_invalid_depth_dropped_and_frame_mismatch_rejected():
    K, T = _camera()
    depth = np.ones((8, 8))
    depth[0, 0] = 0.0
    depth[0, 1] = np.nan
    depth[0, 2] = 2.5  # beyond range_max 2 m radial
    mask = np.zeros((8, 8), bool)
    mask[0, :4] = True
    assert len(L.lift_mask(mask, depth, K, T)) == 1
    with pytest.raises(ValueError):
        L.lift_mask(np.zeros((8, 9), bool), depth, K, T)


def test_robot_removal_same_for_all():
    K, T = _camera()
    depth = np.ones((8, 8))
    mask = np.ones((8, 8), bool)
    robot = np.zeros((8, 8), bool)
    robot[:, :3] = True
    assert len(L.lift_mask(mask, depth, K, T, robot_removal="none", robot_pixels=robot)) == 64
    assert len(L.lift_mask(mask, depth, K, T, robot_removal="gt_robot_seg", robot_pixels=robot)) == 40
    with pytest.raises(ValueError):
        L.lift_mask(mask, depth, K, T, robot_removal="gt_robot_seg")
    with pytest.raises(ValueError):
        L.apply_robot_removal(mask, "sphere")


def test_robot_geom_ids():
    bodyid = np.array([0, 1, 2, 2, 3, 4])
    names = ["world", "table", "robot0_link0", "gripper0_eef", "mount0_base"]
    assert L.robot_geom_ids(bodyid, names).tolist() == [2, 3, 4]


def test_raw_to_opencv_is_row_flip():
    raw = np.arange(4 * 3 * 3).reshape(4, 3, 3).astype(np.uint8)
    np.testing.assert_array_equal(L.opencv_rgb_from_raw(raw), raw[::-1])


def test_policy_mirror_convention():
    """policy256[i, j] = opencv[i, W-1-j], then 256->224 (scale 0.875): right in OpenCV = left in policy."""
    m = np.zeros((256, 256), bool)
    m[100:140, 180:220] = True  # right side of the OpenCV image
    p = L.opencv_mask_to_policy(m)
    ys, xs = np.nonzero(p)
    assert xs.max() < 112  # left half of the policy image
    np.testing.assert_allclose(xs.mean(), 0.875 * (255 - 199.5 + 0.5) - 0.5, atol=1.0)
    np.testing.assert_allclose(ys.mean(), 0.875 * (119.5 + 0.5) - 0.5, atol=1.0)
    back = L.policy_mask_to_opencv(p)
    inter = (back & m).sum()
    assert inter / m.sum() > 0.9 and inter / back.sum() > 0.9


def test_policy_to_opencv_matches_camera_adapter_point_map():
    from benchmark.adapters.libero_panda import camera

    p = np.zeros((224, 224), bool)
    p[50, 60] = True
    back = L.policy_mask_to_opencv(p)
    ys, xs = np.nonzero(back)
    u, v = camera.policy_to_opencv_pixel(60, 50)
    assert abs(xs.mean() - u) <= 1.0 and abs(ys.mean() - v) <= 1.0


def test_voxel_keys_global_grid():
    pts = np.array([[0.0001, 0.0049, -0.0001], [0.0049, 0.0001, -0.0049], [0.0051, 0, 0]])
    k = L.voxel_keys(pts)
    assert k.tolist() == [[0, 0, -1], [1, 0, 0]]
