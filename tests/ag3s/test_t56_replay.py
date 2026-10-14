"""T56 replay capture + geometry storage (synthetic grounding results, no MuJoCo)."""

from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t56_replay as r56


def _robot(scene, target, cache):
    cache.setdefault("pad_half", {"left": 0.005, "right": 0.005})
    return {"T_world_base": np.eye(4), "pads_world": np.zeros((2, 2, 3)),
            "finger_R_world": np.tile(np.eye(3), (2, 1, 1)), "parent_world": np.zeros((2, 3)),
            "both_hand": -1, "held_hand": -1, "target_world": np.zeros(3)}


def _fake_ground(clouds, chosen):
    """ground(ag, fr, att) → (cs, debug, ms) from per-(frame, call-in-frame) definitions."""
    calls = {"n": 0}

    def ground(ag, fr, att):
        i = fr["i"]
        k = calls["n"]
        calls["n"] += 1
        pts, sup = clouds[i]
        sel = chosen[k]
        has = sel is not None
        if has and isinstance(sel, tuple):  # stand-in geometry (no indices into this cloud)
            tp, ti = sel[0], np.zeros(0, np.int64)
        elif has:
            tp, ti = pts[sel], np.asarray(sel)
        else:
            tp, ti = np.zeros((0, 3)), np.zeros(0, np.int64)
        cs = types.SimpleNamespace(has_target=has, target=types.SimpleNamespace(points=tp, point_indices=ti))
        debug = {"filtered_cloud": types.SimpleNamespace(points=pts), "support_mask": sup}
        return cs, debug, 1.0
    return ground


def _frames(n):
    for i in range(n):
        step = types.SimpleNamespace(t_step=8 * i, state=np.zeros(16), actions=np.zeros((10, 16)))
        yield {"i": i, "step": step, "scene": object(),
               "label_of": lambda p: np.zeros(len(p), np.int64)}


def test_capture_keeps_clouds_once_and_chosen_indices(monkeypatch):
    monkeypatch.setattr(r56, "robot_frame", _robot)
    rng = np.random.default_rng(0)
    clouds = {i: (rng.random((30, 3)), np.r_[np.zeros(20, bool), np.ones(10, bool)]) for i in range(2)}
    chosen = [np.arange(5), None, np.arange(3, 9), (rng.random((4, 3)),)]  # (frame, cand) in call order
    cap = r56.Capture(["a", "b"], "apple", ground=_fake_ground(clouds, chosen))
    for fr in _frames(2):
        for _ in range(2):
            cap(None, fr, {})
    assert len(cap.clouds) == 2  # one per frame, shared by both candidates
    arr = r56.arrays_of(cap, [0, 1], ["a", "b"])
    g = r56.Geometry(arr)
    assert np.allclose(g.chosen_points("a", 0), clouds[0][0][:5], atol=1e-6)
    assert g.chosen_points("b", 0).shape == (0, 3)
    assert np.allclose(g.chosen_points("a", 1), clouds[1][0][3:9], atol=1e-6)
    c = g.call("b", 1)
    assert c["member"] == r56.MEMBERSHIP_CODE["proximity"] or c["member"] == r56.MEMBERSHIP_CODE["empty"]
    assert g.chosen_points("b", 1).shape == (4, 3)  # stored stand-in points
    assert g.frame_cloud(1)["support"].sum() == 10


def test_check_order_detects_a_mismatch(monkeypatch):
    monkeypatch.setattr(r56, "robot_frame", _robot)
    clouds = {0: (np.zeros((3, 3)), np.zeros(3, bool))}
    cap = r56.Capture(["a", "b"], "apple", ground=_fake_ground(clouds, [np.arange(2), None]))
    fr = next(_frames(1))
    cap(None, fr, {})
    cap(None, fr, {})
    ok = {"a": [{"frame": 0, "arm": "a", "has_target": True}], "b": [{"frame": 0, "arm": "b", "has_target": False}]}
    cap.check_order(ok)
    with pytest.raises(RuntimeError):
        cap.check_order({"a": [{"frame": 0, "arm": "a", "has_target": False}],
                         "b": [{"frame": 0, "arm": "b", "has_target": False}]})


def test_merge_dedupes_clouds_and_checks_robot_arrays(monkeypatch, tmp_path):
    monkeypatch.setattr(r56, "robot_frame", _robot)
    rng = np.random.default_rng(1)
    clouds = {i: (rng.random((10, 3)), np.zeros(10, bool)) for i in range(2)}
    parts = []
    for names in (["a"], ["b", "c"]):
        cap = r56.Capture(names, "apple",
                          ground=_fake_ground(clouds, [np.arange(i + 1) for i in range(2 * len(names))]))
        for fr in _frames(2):
            for _ in names:
                cap(None, fr, {})
        r56.save_npz(tmp_path / f"{names[0]}.npz", r56.arrays_of(cap, [0, 1], names))
        parts.append(r56.load_npz(tmp_path / f"{names[0]}.npz"))
    g = r56.merge(parts)
    assert g.cands == ["a", "b", "c"] and len(g.z["cloud_digest"]) == 2
    assert np.allclose(g.chosen_points("c", 1), clouds[1][0][:4], atol=1e-6)
    bad = r56.Geometry({**parts[1].z, "T_world_base": parts[1].z["T_world_base"] + 1.0})
    with pytest.raises(ValueError):
        r56.merge([parts[0], bad])


def test_parent_link_swap_table():
    # T43 OC: URDF ee_finger_l1 = MJCF ee_finger_l2; the right hand is not swapped
    assert r56.PARENT_LINK_URDF == {"left": "ee_finger_l1", "right": "ee_finger_r1"}
    assert r56.PARENT_LINK_MJCF == {"left": "ee_finger_l2", "right": "ee_finger_r1"}
