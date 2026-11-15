"""TAB3 (paper Tab. 3, 2026-11-16) — the AG3S-side ablation switches, both default off = the earlier behaviour.

Row 1 `clustering.seed_source` (`--seed-source hand-nearest`): the grounding seed / peak / ranking come from the hand
proximity of each point (`target_grounding.hand_proximity`) on pre-grasp calls; the nearest object is the target.
Row 4 `clustering.identity_memory` (`--identity-memory off`): `TargetConfirm` forgets the manipulated object at every
unfrozen call (no hysteresis, no occluded carry-over, no association, no latched hold); the freeze keeps its rules.

Every check runs the real functions: `ground_target` + `TargetConfirm`, `hand_proximity` on real `HandSpheres`, and the
real `AG3S.process` with the RB-Y1 hand model (`hand_spheres` FK) for the pipeline wiring.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, ClusteringConfig
from AG3S.ag3s.stages.admissibility import HandSpheres
from AG3S.ag3s.stages.target_grounding import TargetConfirm, ground_target, hand_proximity
from AG3S.ag3s.types import AttentionPointCloud, PointCloud

OPENING = 0.07146
APPLE_C = np.array([0.50, 0.00, 0.80])
PEAR_C = APPLE_C + np.array([0.0, -0.150, 0.0])
BASE = ClusteringConfig()


def _sphere(centre, radius, n=400, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return np.asarray(centre, float) + radius * v / np.linalg.norm(v, axis=1, keepdims=True)


APPLE = _sphere(APPLE_C, 0.031, seed=0)
PEAR = _sphere(PEAR_C, 0.030, seed=2)


def _apc(*blobs):
    pts = np.concatenate([b for b, _ in blobs])
    att = np.concatenate([np.full(len(b), a, np.float32) for b, a in blobs])
    att = att + np.linspace(0.0, 1e-4, len(att), dtype=np.float32)
    return AttentionPointCloud(PointCloud(pts), att, att)


def _hand(centre, radius=0.02):
    return HandSpheres(np.asarray(centre, float).reshape(1, 3), np.array([radius]), ("ee_left",))


HAND_ON_APPLE = _hand(APPLE_C + [0.0, 0.0, 0.031 + 0.02 + 0.010])   # 10 mm above the apple
HAND_AWAY = _hand(APPLE_C + [0.0, 0.0, 0.40])


def _confirm(cfg):
    return TargetConfirm(cfg, max_opening=OPENING, overlap_distance=0.005)


def _near(target, centre, tol=0.02):
    return target is not None and float(np.linalg.norm(np.asarray(target.centroid) - centre)) < tol


# ============================================================================ config

def test_defaults_and_validation():
    c = ClusteringConfig()
    assert c.seed_source == "attention" and c.identity_memory is True
    AG3SConfig().with_overrides({"clustering": {"seed_source": "hand_nearest", "identity_memory": False}})
    for bad in ({"seed_source": "hand-nearest"}, {"seed_source": "nearest"}, {"identity_memory": 0},
                {"identity_memory": "off"}, {"seed_source": "hand_nearest", "grounding_source": "oracle"}):
        with pytest.raises(AG3SConfigError):
            AG3SConfig().with_overrides({"clustering": bad})


# ============================================================================ row 1 — the seed signal

def test_hand_proximity_is_one_at_the_nearest_point_and_zero_on_support():
    pts = np.concatenate([APPLE, PEAR])
    sup = np.zeros(len(pts), bool)
    sup[:5] = True
    s = hand_proximity(pts, HAND_ON_APPLE, sup)
    assert s.shape == (len(pts),) and s.dtype == np.float32
    assert np.all(s[:5] == 0.0) and float(s.max()) == pytest.approx(1.0)
    assert s[5:400].mean() > 5 * s[400:].mean()           # the apple is near, the pear 150 mm away
    assert hand_proximity(pts, None) is None and hand_proximity(np.zeros((0, 3)), HAND_ON_APPLE) is None
    assert hand_proximity(pts, HAND_ON_APPLE, np.ones(len(pts), bool)) is None


def test_ground_target_off_follows_attention_on_follows_the_hand():
    apc = _apc((APPLE, 0.1), (PEAR, 0.9))                 # attention on the pear
    off = ground_target(apc, BASE, seed_threshold=0.3)
    assert _near(off.target, PEAR_C)
    sig = hand_proximity(apc.points, HAND_ON_APPLE)
    on = ground_target(apc, BASE, seed_threshold=0.3, seed_signal=sig)   # the attention threshold is not applied to it
    assert _near(on.target, APPLE_C)
    assert on.target.metrics["mean_attention"] > 0.5      # the ranking value is the proximity, not the 0.1 attention
    # the hand at the pear: the pear (also where attention is)
    on2 = ground_target(apc, BASE, seed_threshold=0.3,
                        seed_signal=hand_proximity(apc.points, _hand(PEAR_C + [0, 0, 0.06])))
    assert _near(on2.target, PEAR_C)
    with pytest.raises(ValueError):
        ground_target(apc, BASE, seed_signal=sig[:-1])


def test_ground_target_on_a_flat_attention_map_still_grounds_with_the_hand():
    apc = _apc((APPLE, 0.0), (PEAR, 0.0))
    flat = AttentionPointCloud(apc.cloud, np.zeros(len(apc), np.float32))
    assert ground_target(flat, BASE).target is None       # off: NO_ATTENTION
    on = ground_target(flat, BASE, seed_signal=hand_proximity(flat.points, HAND_ON_APPLE))
    assert _near(on.target, APPLE_C)


def test_ground_target_with_confirm_adopts_the_hand_nearest_object():
    cfg = BASE
    c = _confirm(cfg)
    apc = _apc((APPLE, 0.1), (PEAR, 0.9))
    c.note_hand(HAND_ON_APPLE)
    ground_target(apc, cfg, confirm=c, seed_signal=hand_proximity(apc.points, HAND_ON_APPLE))
    assert c.last.mode == "first" and np.linalg.norm(c.manipulated.centroid - APPLE_C) < 0.02


# ---------------------------------------------------------------------------- row 1 — through AG3S.process

@pytest.fixture(scope="module")
def hand_model():
    """RB-Y1 palms + fingers with a capsule each (as `test_t32_held_object.hand_model`): `hand_spheres` works on it."""
    from AG3S.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf
    from AG3S.ag3s.robot_models.urdf_sphere_chain import UrdfCapsule

    links = ["ee_left", "ee_right"] + [f"ee_finger_{s}{i}" for s in ("l", "r") for i in (1, 2)]
    caps = [UrdfCapsule(link=l, origin=np.eye(4), radius=0.012 if l.startswith("ee_f") else 0.03,
                        length=0.04) for l in links]
    return UrdfSphereChain(parse_urdf(RBY1_URDF), extra_capsules=caps, link_filter=links)


def _ag3s(model, **clustering):
    from AG3S.ag3s.runtime.pipeline import AG3S

    cfg = AG3SConfig().with_overrides({"constraint": {"horizon": 2, "max_attached_primitives": 4},
                                       "clustering": {"gripper_max_opening": OPENING, **clustering}})
    return AG3S(cfg, robot_model=model, constraint_robot_model=model, attached_parent_links=("ee_left", "ee_right"))


def _scene_near_the_left_hand(model, q):
    """A 40 mm ball 90 mm under the left fingers and another 250 mm further along y (no attention: point-cloud input)."""
    tip = 0.5 * (model.link_pose(q, "ee_finger_l1")[:3, 3] + model.link_pose(q, "ee_finger_l2")[:3, 3])
    near_c = tip + np.array([0.0, 0.0, -0.09])
    far_c = near_c + np.array([0.0, 0.25, 0.0])
    return np.concatenate([_sphere(near_c, 0.02, seed=5), _sphere(far_c, 0.02, seed=6)]), near_c, far_c


def test_ag3s_process_off_has_no_target_and_no_record(hand_model):
    q = np.zeros(hand_model.nq)
    cloud, _, _ = _scene_near_the_left_hand(hand_model, q)
    cs = _ag3s(hand_model).process(pointcloud=cloud, robot_state=q)
    assert "seed_source" not in cs.metrics
    assert cs.manipulated is None                         # no attention -> NO_ATTENTION, as before


def test_ag3s_process_on_grounds_the_object_nearest_the_hand(hand_model):
    q = np.zeros(hand_model.nq)
    cloud, near_c, far_c = _scene_near_the_left_hand(hand_model, q)
    ag = _ag3s(hand_model, seed_source="hand_nearest")
    cs = ag.process(pointcloud=cloud, robot_state=q)
    rec = cs.metrics["seed_source"]
    assert rec["applied"] is True and rec["source"] == "hand_nearest" and rec["status"] == "ok"
    assert rec["n_target_points"] > 100 and rec["target_hand_gap_mm"] is not None
    man = ag.manipulated
    assert man is not None and np.linalg.norm(man.centroid - near_c) < 0.02
    # a grasp in progress (frozen): the attention path is used again (nothing new is seeded), recorded as skipped
    ag.set_grasp_active(True)
    cs = ag.process(pointcloud=cloud, robot_state=q)
    assert cs.metrics["seed_source"]["applied"] is False and cs.metrics["seed_source"]["skipped"] == "frozen"


# ============================================================================ row 4 — identity memory off

def _step(c, apc, cfg, *, hand=None, latched=None, frozen=None):
    if latched is not None:
        c.note_latched(latched)
    if frozen is not None:
        c.freeze(frozen)
    c.note_hand(hand)
    res = ground_target(apc, cfg, seed_threshold=0.3, confirm=c)
    return c.last, res


def _adopt(cfg):
    c = _confirm(cfg)
    _step(c, _apc((APPLE, 0.9)), cfg)
    assert c.manipulated.id == 0
    return c


def test_memory_on_is_the_t31b_rule_and_records_nothing():
    cfg = dataclasses.replace(BASE, identity_memory=True)
    c = _adopt(cfg)
    out = [_step(c, _apc((PEAR, 0.9)), cfg, hand=HAND_AWAY) for _ in range(3)]
    assert [d.mode for d, _ in out] == ["occluded", "occluded", "switch"] and c.manipulated.id == 1
    assert all("identity_memory_off" not in (r.target.metrics if r.target else {}) for _, r in out)


def test_memory_off_takes_the_call_leader_at_once_and_keeps_no_geometry():
    cfg = dataclasses.replace(BASE, identity_memory=False)
    c = _adopt(cfg)
    d, r = _step(c, _apc((PEAR, 0.9)), cfg, hand=HAND_AWAY)
    assert d.mode == "first" and c.manipulated.id == 1 and np.linalg.norm(c.manipulated.centroid - PEAR_C) < 0.02
    assert r.target.metrics["identity_memory_off"] == 1.0 and r.target.metrics["identity_memory_forgot"] == 1.0
    # the apple back with more attention: the leader again at once (no 3-frame hysteresis), a new identity
    d, _ = _step(c, _apc((APPLE, 0.9), (PEAR, 0.2)), cfg)
    assert d.mode == "first" and c.manipulated.id == 2
    # a frame with no cluster at all: nothing carried over (memory on: the object would stay, occluded)
    c.unseen()
    assert c.manipulated is None and c.memory_forgot is True


def test_memory_off_overrides_the_latched_hold():
    on = dataclasses.replace(BASE, identity_memory=True, latched_identity_hold=True)
    c = _adopt(on)
    assert [_step(c, _apc((PEAR, 0.9)), on, hand=HAND_AWAY, latched=True)[0].mode for _ in range(3)] == ["occluded"] * 3
    off = dataclasses.replace(on, identity_memory=False)
    c = _adopt(off)
    d, _ = _step(c, _apc((PEAR, 0.9)), off, hand=HAND_AWAY, latched=True)
    assert d.mode == "first" and not d.latched_hold and c.manipulated.id == 1


def test_memory_off_keeps_the_grasp_lock_while_frozen():
    cfg = dataclasses.replace(BASE, identity_memory=False)
    c = _adopt(cfg)
    modes = [_step(c, _apc((PEAR, 0.9)), cfg, hand=HAND_AWAY, frozen=True)[0].mode for _ in range(4)]
    assert modes == ["occluded"] * 4 and c.manipulated.id == 0 and c.memory_forgot is False
    d, _ = _step(c, _apc((PEAR, 0.9)), cfg, hand=HAND_AWAY, frozen=False)   # released: forgotten, the leader
    assert d.mode == "first" and c.manipulated.id == 1


def test_ag3s_carries_identity_memory_to_target_confirm(hand_model):
    assert _ag3s(hand_model)._target_confirm.identity_memory is True                       # noqa: SLF001
    assert _ag3s(hand_model, identity_memory=False)._target_confirm.identity_memory is False   # noqa: SLF001
