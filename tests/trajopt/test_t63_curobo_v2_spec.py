"""T63 — CPU tests of the robot -> cuRobo mapping (`AG3S/trajopt/curobo_v2/spec.py`) and the
baseline-side exporter's pure helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest

from AG3S.trajopt.curobo_v2 import spec as S
from AG3S.trajopt.curobo_v2.export_inputs import episode_servers, model_options_from_argv

ARM = [*(f"left_arm_{i}" for i in range(7)), *(f"right_arm_{i}" for i in range(7))]
TORSO = [f"torso_{i}" for i in range(6)]
MODEL_JOINTS = [*TORSO, *(f"right_arm_{i}" for i in range(7)), *(f"left_arm_{i}" for i in range(7))]


def _spec():
    spheres = []
    layout = [("ee_left", [0.0, 0.0, 0.1], 0.006), ("ee_finger_l1", [0.01, 0.0, 0.02], 0.003),
              ("ee_left", [0.0, 0.01, 0.1], 0.006), ("ee_finger_l2", [-0.01, 0.0, 0.02], 0.003),
              ("ee_finger_l1", [0.01, 0.0, 0.03], 0.004)]
    for i, (link, c, r) in enumerate(layout):
        spheres.append({"index": i, "link": link, "center": c, "radius": r})
    n = len(ARM)
    return {
        "urdf_path": "/tmp/robot/model.urdf", "curobo_base_link": "link_torso_5",
        "model_joint_names": MODEL_JOINTS, "arm_joint_names": ARM,
        "finger_joints": {
            "gripper_finger_l1": {"type": "prismatic", "axis": [-1.0, 0.0, 0.0],
                                  "parent": "ee_left", "child": "ee_finger_l1",
                                  "lower": -0.05, "upper": 0.0},
            "gripper_finger_l2": {"type": "prismatic", "axis": [1.0, 0.0, 0.0],
                                  "parent": "ee_left", "child": "ee_finger_l2",
                                  "lower": 0.0, "upper": 0.05}},
        "n_spheres": len(spheres), "spheres": spheres,
        "limits": {"joint_names": ARM, "lower": [-1.0] * n, "upper": [1.0] * n,
                   "velocity": [2.0] * n, "acceleration": [9.0] * n},
    }


def test_sphere_permutation_groups_links_in_first_appearance_order():
    spec = _spec()
    assert S.collision_link_order(spec) == ["ee_left", "ee_finger_l1", "ee_finger_l2"]
    perm = S.sphere_permutation(spec)
    # cuRobo order: ee_left (0, 2), ee_finger_l1 (1, 4), ee_finger_l2 (3)
    assert perm.tolist() == [0, 2, 1, 4, 3]
    ours = np.arange(5) * 10.0
    curobo = ours[perm]
    assert np.array_equal(curobo[np.argsort(perm)], ours)


def test_robot_cfg_dict_carries_the_served_spheres_unchanged():
    spec = _spec()
    cfg = S.robot_cfg_dict(spec)["robot_cfg"]["kinematics"]
    assert cfg["base_link"] == "link_torso_5"
    assert cfg["collision_link_names"] == ["ee_left", "ee_finger_l1", "ee_finger_l2"]
    assert cfg["lock_joints"] == {"gripper_finger_l1": 0.0, "gripper_finger_l2": 0.0}
    assert cfg["cspace"]["joint_names"] == ARM
    flat = [(link, s["center"], s["radius"]) for link in cfg["collision_link_names"]
            for s in cfg["collision_spheres"][link]]
    perm = S.sphere_permutation(spec)
    for k, i in enumerate(perm):
        src = spec["spheres"][i]
        assert flat[k] == (src["link"], src["center"], src["radius"])
    assert cfg["collision_sphere_buffer"] == 0.0


def test_robot_cfg_rejects_limits_out_of_cspace_order():
    spec = _spec()
    spec["limits"]["joint_names"] = list(reversed(ARM))
    with pytest.raises(ValueError):
        S.robot_cfg_dict(spec)


def _prismatic(origin: np.ndarray, axis, v):
    T = np.eye(4)
    T[:3, 3] = np.asarray(axis, float) * v
    return origin @ T


def test_finger_shift_equals_prismatic_joint_at_value():
    spec = _spec()
    rng = np.random.default_rng(0)
    # arbitrary joint origin (rotation + translation), as in a URDF
    a = rng.normal(size=3)
    a /= np.linalg.norm(a)
    th = 0.7
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    origin = np.eye(4)
    origin[:3, :3] = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    origin[:3, 3] = [0.01, -0.02, 0.05]
    values = {"gripper_finger_l1": -0.031, "gripper_finger_l2": 0.044}
    table = S.finger_sphere_table(spec, values)
    for joint, info in spec["finger_joints"].items():
        local = [s for s in spec["spheres"] if s["link"] == info["child"]]
        shifted = table[info["child"]]
        assert shifted.shape == (len(local), 4)
        for s, row in zip(local, shifted):
            p = np.r_[s["center"], 1.0]
            want = _prismatic(origin, info["axis"], values[joint]) @ p   # joint at v
            got = _prismatic(origin, info["axis"], 0.0) @ np.r_[row[:3], 1.0]  # locked at 0
            assert np.allclose(want, got, atol=1e-12)
            assert row[3] == s["radius"]


def test_finger_value_outside_joint_range_is_refused():
    with pytest.raises(ValueError):
        S.finger_sphere_table(_spec(), {"gripper_finger_l1": 0.01})


def test_finger_values_reads_one_step_of_the_path():
    P = np.array([[-0.04, -0.03], [0.04, 0.03]])
    assert S.finger_values(_spec(), P, 1) == {"gripper_finger_l1": -0.03,
                                              "gripper_finger_l2": 0.03}


def test_chunk_columns_roundtrip_and_q_now_order():
    spec = _spec()
    rng = np.random.default_rng(1)
    chunk = rng.normal(size=(50, 16))
    q = S.chunk_to_cspace(chunk, spec)
    assert q.shape == (50, 14)
    assert np.array_equal(q[:, :7], chunk[:, :7]) and np.array_equal(q[:, 7:], chunk[:, 8:15])
    back = S.cspace_to_chunk(q, chunk)
    assert np.array_equal(back, chunk)
    q20 = np.arange(20, dtype=float)
    qc = S.q_now_cspace(q20, spec)
    assert qc.tolist() == [MODEL_JOINTS.index(n) for n in ARM]


def test_limits_override_is_symmetric_rate_bounds():
    lim = S.limits_override(_spec())
    assert lim["position"].shape == (2, 14)
    assert np.all(lim["velocity"][0] == -lim["velocity"][1])
    assert np.all(lim["acceleration"][1] == 9.0)


def test_knot_seed_follows_the_reference_path():
    q0 = np.zeros(14)
    ref = np.outer(np.arange(1, 51) / 50.0, np.ones(14))  # straight line q0 -> 1
    knots = S.knot_seed(q0, ref, 16)
    assert knots.shape == (16, 14)
    assert np.allclose(knots[0], q0) and np.allclose(knots[-1], ref[-1])
    assert np.allclose(knots[:, 0], np.arange(16) / 15.0)


def test_model_options_from_server_argv():
    argv = ["--links", "gripper", "--sphere-spacing", "0.4", "--max-spheres-per-capsule", "32",
            "--capsule-radius-scale", "0.05", "--finger-cover", "both", "--model-xml", "/x.xml",
            "--esdf-margin", "0", "--plan-horizon", "16", "--target-field-policy",
            "exclude-authorized", "--port", "8293"]
    o = model_options_from_argv(argv)
    assert o["sphere_options"] == {"sphere_spacing": 0.4, "max_spheres_per_capsule": 32,
                                   "capsule_radius_scale": 0.05}
    assert o["finger_cover"] == {"hands": "both"} and o["plan_horizon"] == 16
    with pytest.raises(SystemExit):
        model_options_from_argv(argv + ["--palm-cover", "on"])


def test_episode_servers_follow_chain_log_blocks(tmp_path):
    (tmp_path / "chain_8292.log").write_text("\n".join([
        "server up arm=E0 pid=1 rec=/r/server_a startup_s=1",
        "start E0/E0_ep5_s53 attempt=1 obstacle=:",
        "server up arm=C1aN pid=2 rec=/r/server_b startup_s=1",
        "start C1aN/C1aN_ep7_s73 attempt=1 obstacle=:",
        "start C1aN/C1aN_ep9_s93 attempt=1 obstacle=:",
    ]))
    m = episode_servers(tmp_path)
    assert m == {7: {"server_dir": "/r/server_b", "order": 0, "seed": 73},
                 9: {"server_dir": "/r/server_b", "order": 1, "seed": 93}}
    (tmp_path / "chain_8293.log").write_text(
        "server up arm=C1aN pid=3 rec=/r/c startup_s=1\nstart C1aN/C1aN_ep1_s13 attempt=2 x\n")
    with pytest.raises(SystemExit):
        episode_servers(tmp_path)


def test_load_spec_checks_count(tmp_path):
    spec = _spec()
    spec["n_spheres"] = 99
    p = tmp_path / "s.json"
    p.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        S.load_spec(p)


def test_sample_chunk_grid_exact_and_resampled():
    t = np.arange(101) / 30.0
    pos = np.stack([np.sin(t), t ** 2], axis=1)
    exact = S.sample_chunk_grid(pos, 1 / 30.0, 1 / 15.0, 50)
    assert np.array_equal(exact, pos[2::2][:50])
    t24 = np.arange(81) / 24.0
    pos24 = np.stack([t24, 2 * t24], axis=1)  # linear: resampling is exact
    res = S.sample_chunk_grid(pos24, 1 / 24.0, 1 / 15.0, 50)
    k = np.arange(1, 51) / 15.0
    assert np.allclose(res, np.stack([k, 2 * k], axis=1))
    assert S.sample_chunk_grid(pos24[:40], 1 / 24.0, 1 / 15.0, 50) is None
