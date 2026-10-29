"""T63 — CPU tests of the AG3S ESDF -> cuRobo v2 voxel mapping (`curobo_v2/world.py`).

The cuRobo-side formulas are restated from the kernel source so that a convention change in
either side shows up here: `VoxelGrid.create_xyzr_tensor` (voxel centres) and
`data_voxel.sample_voxel_sdf_with_grad` (continuous index `local / voxel + n / 2 - 0.5`),
`wp_collision_common.apply_collision_activation` (cost)."""

from __future__ import annotations

import numpy as np
import pytest

from AG3S.trajopt.curobo_v2 import world as W


def _quat_to_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return _quat_to_matrix(q)


def test_grid_centre_matches_curobo_voxel_centres():
    origin = np.array([0.127, -0.0054, 0.711])
    shape = (128, 64, 32)
    vs = 0.005
    pose = W.grid_centre(origin, shape, vs)
    dims = W.grid_dims(shape, vs)
    for axis in range(3):
        n = shape[axis]
        i = np.arange(n)
        # create_xyzr_tensor: (linspace(1, n, n) - round(0.5 * dims / vs)) * vs - 0.5 vs + pose
        curobo = (i + 1 - round(0.5 * dims[axis] / vs)) * vs - 0.5 * vs + pose[axis]
        ours = origin[axis] + i * vs
        assert np.allclose(curobo, ours, atol=1e-12)
        # kernel continuous index at our voxel centre i is exactly i
        local = ours - pose[axis]
        assert np.allclose(local / vs + n / 2.0 - 0.5, i, atol=1e-9)
    assert np.all(np.round(dims / vs) == np.asarray(shape))


def test_voxel_centres_are_c_order_like_the_kernel():
    origin = np.zeros(3)
    shape = (3, 4, 5)
    c = W.voxel_centres(origin, shape, 1.0)
    ix, iy, iz = 2, 1, 3
    flat = ix * shape[1] * shape[2] + iy * shape[2] + iz  # voxel_idx_to_flat
    assert np.allclose(c[flat], [ix, iy, iz])


def test_grid_pose_in_frame_moves_the_world_with_the_torso():
    rng = np.random.default_rng(3)
    T_base_frame = np.eye(4)
    T_base_frame[:3, :3] = _random_rotation(rng)
    T_base_frame[:3, 3] = rng.normal(size=3)
    centre = np.array([0.4, 0.1, 0.9])
    pose = W.grid_pose_in_frame(T_base_frame, centre)
    T_frame_grid = np.eye(4)
    T_frame_grid[:3, :3] = _quat_to_matrix(pose[3:])
    T_frame_grid[:3, 3] = pose[:3]
    p_base = rng.normal(size=(20, 3))
    p_frame = W.transform_points(np.linalg.inv(T_base_frame), p_base)
    local = W.transform_points(np.linalg.inv(T_frame_grid), p_frame)  # what the kernel computes
    assert np.allclose(local, p_base - centre, atol=1e-12)  # base-aligned grid
    assert pose[3] >= 0.0


@pytest.mark.parametrize("seed", range(5))
def test_rotation_quaternion_roundtrip(seed):
    R = _random_rotation(np.random.default_rng(seed))
    q = W.rotation_to_quat_wxyz(R)
    assert np.isclose(np.linalg.norm(q), 1.0)
    assert np.allclose(_quat_to_matrix(q), R, atol=1e-12)


def _activation_cost(sdf, r, eta):
    p = r + eta - sdf
    return np.where(p <= 0.0, 0.0, np.where(p > eta, p - 0.5 * eta, 0.5 * p * p / eta))


def test_sdf_from_activation_cost_inverts_both_regimes():
    eta = 20.0
    r = np.full(7, 0.006)
    sdf = np.array([-0.03, 0.0, 0.004, 0.006, 0.5, 1.9, 15.0])
    cost = _activation_cost(sdf, r, eta)
    back = W.sdf_from_activation_cost(cost, r, eta)
    assert np.allclose(back, sdf, atol=1e-9)
    far = W.sdf_from_activation_cost(_activation_cost(np.array([1e4]), r[:1], eta), r[:1], eta)
    assert np.isinf(far[0])


def test_sdf_inversion_is_float32_accurate_at_mm():
    eta = 20.0
    r = np.full(1000, 0.008)
    sdf = np.linspace(-0.05, 1.9, 1000)
    cost32 = _activation_cost(sdf, r, eta).astype(np.float32)
    back = W.sdf_from_activation_cost(cost32, r, eta)
    assert np.abs(back - sdf).max() < 2e-5  # 0.02 mm


def test_compose_main_and_lattice_follow_curobo_field():
    from AG3S.ag3s.fields.curobo_field import CuroboEsdfField, layer_from_arrays

    rng = np.random.default_rng(7)
    coarse = layer_from_arrays(rng.uniform(-0.05, 0.5, (8, 8, 8)).astype(np.float16),
                               np.array([-0.4, -0.4, -0.4]), 0.1)
    fine_o = np.array([-0.1, -0.1, -0.1])
    fine_d = rng.uniform(-0.02, 0.3, (16, 16, 16)).astype(np.float16)
    fine = layer_from_arrays(fine_d, fine_o, 0.0125)
    field = CuroboEsdfField((coarse, fine), outside_distance=0.5)
    pts = rng.uniform(-0.35, 0.35, (500, 3))
    want = np.asarray(field._evaluate(pts)[0])
    inside = W.inside_lattice(pts, fine_o, fine_d.shape, 0.0125)
    assert np.array_equal(inside, CuroboEsdfField._covers(fine.grid, pts))
    got = W.compose_main(np.asarray(coarse.distance(pts)), np.asarray(fine.distance(pts)), inside)
    assert np.allclose(got, want)


def test_layer_arrays_reads_record_layout():
    z = {"esdf_free_distance": np.zeros((4, 5, 6), np.float16),
         "esdf_free_origin": np.array([0.1, 0.2, 0.3]), "esdf_free_voxel_size": np.array(0.005),
         "esdf_free_shape": np.array([4, 5, 6], np.int32)}
    lay = W.layer_arrays(z, "free")
    assert lay["voxel_size"] == 0.005 and lay["distance"].dtype == np.float16
    z["esdf_free_shape"] = np.array([4, 5, 7], np.int32)
    with pytest.raises(ValueError):
        W.layer_arrays(z, "free")
