"""Framing (numpy-version-free pipe format), wire -> SceneObservation, target phrase parsing, depth perception.

Any venv with numpy + scipy; the recorded-request tests need the SL5c VLA-only run and openpi_client
(``.venv-openpi-live``)::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD .venv-openpi-live/bin/python -m pytest -q Baseline/paper_baselines/closed_loop/tests
"""
from __future__ import annotations

import json
import os
import pathlib
import re

import numpy as np
import pytest

from Baseline.paper_baselines.closed_loop import framing
from Baseline.paper_baselines.closed_loop.wire_obs import (obs_from_dict, obs_to_dict, policy_rgb_to_opencv,
                                                           scene_observation, target_phrase_from_instruction)

ROOT = pathlib.Path(__file__).resolve().parents[4]
STORAGE = pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", ROOT.parent / "AG3S_VLA-storage"))
SL5C_LI_VLA = STORAGE / "results/benchmark/20261014T180105_61f44671805b_sl5c-safelibero-spatial-dev-LI-vla_only"
STEM = "safelibero_spatial_t0_e0"


def test_framing_round_trip_bit_exact():
    rng = np.random.default_rng(0)
    obj = {"a": rng.normal(size=(3, 4)).astype(np.float32), "b": [1, 2.5, None, "x", True],
           "c": {"d": np.arange(5, dtype=np.uint16), "e": (np.float64(np.pi), np.int64(3), np.bool_(False))},
           "nan": float("nan"), "empty": np.zeros((0, 3))}
    out = framing.unpack(framing.pack(obj))
    assert out["a"].dtype == np.float32 and out["a"].tobytes() == obj["a"].tobytes()
    assert out["c"]["d"].dtype == np.uint16 and np.array_equal(out["c"]["d"], obj["c"]["d"])
    assert out["b"] == [1, 2.5, None, "x", True]
    assert out["c"]["e"] == [np.pi, 3, False]
    assert np.isnan(out["nan"]) and out["empty"].shape == (0, 3)
    with pytest.raises(TypeError):
        framing.pack({"o": np.array([object()])})


def _task_texts():
    d = json.loads((ROOT / "benchmark/configs/suites.json").read_text())
    for name, s in d["suites"].items():
        if not (name.startswith("safelibero") or name in ("libero_spatial", "libero_object", "libero_goal", "libero_10")):
            continue
        for _lv, tasks in s.get("tasks", {}).items():
            for t in tasks:
                yield name, re.sub(r"^[A-Z_]+SCENE\d+_", "", t).replace("_", " ")


def test_target_phrase_every_task_has_a_noun_phrase():
    missing = [(n, t) for n, t in _task_texts() if not target_phrase_from_instruction(t)]
    assert not missing, missing
    assert target_phrase_from_instruction(
        "pick up the black bowl between the plate and the ramekin and place it on the plate") == "black bowl"
    assert target_phrase_from_instruction("put both the alphabet soup and the tomato sauce in the basket") == "alphabet soup"
    assert target_phrase_from_instruction("turn on the stove and put the moka pot on it") == "stove"
    assert target_phrase_from_instruction("open the middle drawer of the cabinet") == "drawer"


def test_policy_rgb_to_opencv_inverts_the_client_preprocessing():
    rng = np.random.default_rng(1)
    raw = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
    raw = np.kron(raw[::8, ::8], np.ones((8, 8, 1), np.uint8))  # smooth enough to survive 256->224->256
    from PIL import Image

    pol = np.asarray(Image.fromarray(np.ascontiguousarray(raw[::-1, ::-1])).resize((224, 224), Image.BILINEAR))
    cv = policy_rgb_to_opencv(pol, (256, 256)).astype(float)
    ref = raw[::-1].astype(float)
    assert np.mean(np.abs(cv - ref)) < 12.0
    assert np.mean(np.abs(cv - raw[::-1, ::-1].astype(float))) > 2 * np.mean(np.abs(cv - ref))  # orientation matters


@pytest.fixture(scope="module")
def recorded():
    if not SL5C_LI_VLA.is_dir():
        pytest.skip("SL5c VLA-only run not available")
    pytest.importorskip("openpi_client")
    from Baseline.paper_baselines.closed_loop.replay import RecordedEpisode

    return RecordedEpisode(SL5C_LI_VLA, STEM)


def test_recorded_request_has_the_client_keys(recorded):
    req = recorded.request(0)
    for k in ("observation/image", "observation/wrist_image", "observation/state", "prompt", "ag3s/cameras",
              "ag3s/depth/head", "ag3s/K/head", "ag3s/T_base_cam/head", "ag3s/robot_state/left_wrist",
              "ag3s/seq", "ag3s/reset", "ag3s/policy_seed", "panda/robot_state", "panda/T_world_base",
              "panda/prev_step_world"):
        assert k in req, k
    assert req["ag3s/policy_seed"] == 700000 and req["ag3s/reset"] is True
    assert "ag3s/policy_seed" not in recorded.request(1)
    assert req["observation/image"].shape == (224, 224, 3) and req["ag3s/depth/head"].dtype == np.uint16


def test_scene_observation_frames_match_the_recording(recorded):
    from Baseline.paper_baselines.methods.safety_common import quat_xyzw_to_mat

    for k in (0, 7):
        req = recorded.request(k)
        obs = scene_observation(req, seq=k)
        t = int(recorded.z["chunk_t"][k])
        r = int(np.flatnonzero(recorded.obs_index == t)[0])
        T_wb = recorded.g["T_world_base"][r]
        T_we = np.eye(4)
        T_we[:3, :3], T_we[:3, 3] = quat_xyzw_to_mat(recorded.z["obs_robot0_eef_quat"][t]), recorded.z["obs_robot0_eef_pos"][t]
        ref = np.linalg.inv(T_wb) @ T_we  # = safety_common.load_recorded_call's ee_pose
        assert np.abs(obs.ee_pose - ref).max() < 1e-9
        assert np.array_equal(obs.depth["head"], recorded.g["depth_mm/head"][r].astype(np.float64) / 1000.0)
        assert np.array_equal(obs.q, recorded.g["robot_state/left_wrist"][r][:7])
        raw_cv = recorded.z["obs_agentview_image"][t][::-1].astype(float)
        assert np.mean(np.abs(obs.rgb["head"].astype(float) - raw_cv)) < 8.0
        assert obs.target_phrase == "black bowl" and not any(key.startswith("privileged/") for key in obs.extras)
        back = obs_from_dict(framing.unpack(framing.pack(obs_to_dict(obs))))
        assert np.array_equal(back.depth["head"], obs.depth["head"]) and np.array_equal(back.ee_pose, obs.ee_pose)


def test_depth_perception_removes_robot_and_table(recorded):
    from Baseline.paper_baselines.closed_loop.perception import box_masks, clusters, scene_points

    obs = scene_observation(recorded.request(7), seq=7)
    sp = scene_points(obs)
    P = sp["obstacle_points"]
    assert len(P) > 100 and sp["info"]["n_robot_removed"] > 0
    assert np.all(P[:, 2] > sp["table_z"])  # table plane removed
    from AG3S.adapters.panda import model as pm

    m = pm.build_panda_bare_model()
    q = pm.set_finger_parameters(m, obs.extras["robot_state"])
    C, R = m.sphere_centers_numeric(q)
    d = np.linalg.norm(P[:, None, :] - np.asarray(C)[None], axis=2) - np.asarray(R)[None]
    assert d.min() > 0.0195  # nothing within a robot sphere + 2 cm (voxel centre round-off 0.5 mm)
    cl = clusters(P)
    assert cl and sum(len(v) for v in cl.values()) <= len(P)
    full = box_masks({"head": (0, 0, 256, 256), "left_wrist": (0, 0, 256, 256)}, {c: obs.depth[c].shape for c in obs.depth})
    sp2 = scene_points(obs, target_masks=full)
    assert len(sp2["obstacle_points"]) == 0 and sp2["target_points"] is not None
