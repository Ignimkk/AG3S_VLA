"""SL-PE1-3 wiring (2026-11-08, dev-30) — all off by default; off = the earlier behaviour.

PE1 `clustering.close_seed_grounding`: grounding input = the object cut out under the grasp seed, adopted at once before attach.
PE2 `clustering.destination_after_held`: perception registers the destination only while an object is held.
PE3 `clustering.scene_obstacle_held_exclude`: the scene obstacle objects also lose the components at the held object now.
"""
from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, ClusteringConfig
from AG3S.ag3s.runtime.pipeline import AG3S
from AG3S.ag3s.stages.target_grounding import CLOSE_SEED_SWITCH, TargetConfirm, ground_target
from AG3S.ag3s.types import AttentionPointCloud, PointCloud

OPENING = 0.07146
BASE = ClusteringConfig()


def _sphere(centre, radius, n=400, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return np.asarray(centre, float) + radius * v / np.linalg.norm(v, axis=1, keepdims=True)


APPLE = _sphere([0.50, 0.00, 0.80], 0.031, seed=0)
PEAR = _sphere([0.50, -0.15, 0.80], 0.030, seed=2)
def _grid(a, b, step=0.008):
    """dense planar patch between corners a and b (one coordinate equal)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    axes = [np.arange(lo, hi + 1e-9, step) if hi > lo else np.array([lo]) for lo, hi in zip(a, b)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


CRATE = np.concatenate([_grid([0.30, 0.20, 0.95], [0.48, 0.50, 0.95]),      # lid
                        _grid([0.30, 0.20, 0.70], [0.48, 0.20, 0.95]),      # front wall
                        _grid([0.30, 0.20, 0.70], [0.30, 0.50, 0.95])])     # side wall: 0.18 x 0.30 x 0.25 m, not graspable


def _apc(*blobs):
    pts = np.concatenate([b for b, _ in blobs])
    att = np.concatenate([np.full(len(b), a, np.float32) for b, a in blobs])
    att = att + np.linspace(0.0, 1e-4, len(att), dtype=np.float32)
    return AttentionPointCloud(PointCloud(pts), att, att)


def _confirm(cfg):
    return TargetConfirm(cfg, max_opening=OPENING, overlap_distance=0.005)


def _step(c, apc, cfg, *, seed=False, frozen=None, pre_attach=None):
    c.note_close_seed(seed)
    if frozen is not None:
        c.freeze(frozen)
    if pre_attach is not None:
        c.note_pre_attach_grasp(pre_attach)
    ground_target(apc, cfg, seed_threshold=0.3, confirm=c)
    return c.last


# ============================================================================ config

def test_defaults_off_and_validated():
    c = ClusteringConfig()
    assert (c.close_seed_grounding, c.destination_after_held, c.scene_obstacle_held_exclude) == (False, False, False)
    AG3SConfig().with_overrides({"clustering": {"close_seed_grounding": True, "destination_after_held": True,
                                                "scene_obstacle_objects": True, "scene_obstacle_held_exclude": True}})
    for bad in ({"close_seed_grounding": 1}, {"destination_after_held": "y"},
                {"scene_obstacle_held_exclude": True},                         # needs scene_obstacle_objects
                {"close_seed_grounding": True, "grounding_source": "oracle"}):
        with pytest.raises(AG3SConfigError):
            AG3SConfig().with_overrides({"clustering": bad})


# ============================================================================ PE1 confirm

def test_off_the_seed_note_changes_nothing():
    cfg = BASE
    c = _confirm(cfg)
    _step(c, _apc((APPLE, 0.9)), cfg)
    modes = [_step(c, _apc((PEAR, 0.9)), cfg, seed=True).mode for _ in range(3)]
    assert modes == ["occluded", "occluded", "switch"]


def test_on_the_seed_object_is_adopted_at_once():
    cfg = dataclasses.replace(BASE, close_seed_grounding=True)
    c = _confirm(cfg)
    _step(c, _apc((APPLE, 0.9)), cfg)
    d = _step(c, _apc((PEAR, 0.9)), cfg, seed=True)
    assert d.mode == "switch" and d.reason == CLOSE_SEED_SWITCH and c.manipulated.id == 1
    assert c.close_seed_switched
    # no seed this call: the 3-frame rule as before
    d = _step(c, _apc((APPLE, 0.9)), cfg)
    assert d.mode == "occluded" and not c.close_seed_switched


def test_on_closing_before_attach_switches_held_does_not():
    cfg = dataclasses.replace(BASE, close_seed_grounding=True)
    c = _confirm(cfg)
    _step(c, _apc((APPLE, 0.9)), cfg)
    assert _step(c, _apc((PEAR, 0.9)), cfg, seed=True, frozen=True, pre_attach=True).reason == CLOSE_SEED_SWITCH
    c2 = _confirm(cfg)
    _step(c2, _apc((APPLE, 0.9)), cfg)
    d = _step(c2, _apc((PEAR, 0.9)), cfg, seed=True, frozen=True, pre_attach=False)   # attached
    assert d.reason != CLOSE_SEED_SWITCH and c2.manipulated.id == 0


def test_on_a_seed_named_identity_is_locked_once_closing():
    cfg = dataclasses.replace(BASE, close_seed_grounding=True)
    c = _confirm(cfg)
    _step(c, _apc((APPLE, 0.9)), cfg, seed=True)                    # adopted from the seed (first)
    d = _step(c, _apc((PEAR, 0.9)), cfg, seed=True, frozen=True, pre_attach=True)
    assert d.reason != CLOSE_SEED_SWITCH and c.manipulated.id == 0   # closing: the seed cut flipping elsewhere is ignored
    c2 = _confirm(cfg)
    _step(c2, _apc((APPLE, 0.9)), cfg, seed=True)
    assert _step(c2, _apc((PEAR, 0.9)), cfg, seed=True).reason == CLOSE_SEED_SWITCH   # before the close it still moves


# ============================================================================ PE2

def _registered(cfg, **kw):
    c = _confirm(cfg)
    for _ in range(6):   # registration needs `frames` (3) running frames
        _step(c, _apc((CRATE, 0.95), (APPLE, 0.5)), cfg, **kw)
    return c.destination is not None


def test_destination_after_held():
    assert _registered(BASE)                                                        # off: registered before any grasp
    on = dataclasses.replace(BASE, destination_after_held=True)
    assert not _registered(on)                                                      # on, nothing held: not registered
    assert not _registered(on, frozen=True, pre_attach=True)                        # closing, not attached: not yet
    assert _registered(on, frozen=True, pre_attach=False)                           # held: registered


def test_destination_held_leader_registers_a_plate_while_held():
    from AG3S.ag3s.stages.admissibility import HandSpheres
    plate = _grid([0.30, 0.20, 0.70], [0.44, 0.34, 0.70])            # flat 140 x 140 mm: graspable by its narrowest extent
    plate = np.concatenate([plate, plate + [0, 0, 0.010]])
    hand = HandSpheres(np.array([[0.50, -0.15, 1.10]]), np.array([0.04]), ("panda_hand",))   # far from the plate

    def run(cfg):
        c = _confirm(cfg)
        _step(c, _apc((APPLE, 0.9)), cfg)                             # something adopted, then held
        for _ in range(5):
            c.note_hand(hand)
            _step(c, _apc((plate, 0.95)), cfg, frozen=True, pre_attach=False)
        return c.destination is not None
    after = dataclasses.replace(BASE, destination_after_held=True)
    assert not run(after)                                             # PE2 alone: a plate never registers
    assert run(dataclasses.replace(after, destination_held_leader=True))


# ============================================================================ PE1 pipeline input

def test_close_seed_input_cuts_the_object_under_the_seed():
    bowl_on_box = np.concatenate([
        np.random.default_rng(1).uniform([0.40, -0.10, 0.10], [0.60, 0.10, 0.1001], size=(2500, 3)),   # box lid
        _sphere([0.50, 0.0, 0.135], 0.03, seed=4)])                                             # object on it
    att = np.r_[np.full(2500, 0.9, np.float32), np.full(400, 0.1, np.float32)]   # attention on the box
    apc = AttentionPointCloud(PointCloud(bowl_on_box), att, att)
    cfg = dataclasses.replace(BASE, close_seed_grounding=True)
    fake_seed = np.array([0.53, 0.0, 0.14])
    fake = types.SimpleNamespace(_close_seed=fake_seed.copy(), _target_confirm=_confirm(cfg),
                                 _close_seed_frame=None, _close_seed_plane=None,
                                 config=types.SimpleNamespace(clustering=cfg))
    fake._add_support_plane = lambda plane: AG3S._add_support_plane(fake, plane)
    cloud, excl = AG3S._close_seed_grounding_input(fake, apc, None)
    assert fake._close_seed is None and fake._close_seed_frame["applied"]
    assert fake._target_confirm._close_seed is True
    on = np.flatnonzero(np.asarray(cloud.attention) > 0)
    assert len(on) >= 300 and on.min() >= 2500 and excl.sum() == len(bowl_on_box) - len(on)
    # the lid the cut stood on joins this frame's support planes (T43 T 3(c) test), 10 mm under the band top
    sup = fake._target_confirm._support
    assert sup is not None and sup.shape == (1, 4) and np.allclose(sup[0, :3], [0, 0, 1])
    assert abs(sup[0, 3] - fake._close_seed_frame["support_plane_z_m"]) < 1e-9 and 0.09 <= sup[0, 3] <= 0.11
    # graded: the cut point nearest the seed carries the most attention
    a = np.asarray(cloud.attention)[on]
    near = np.linalg.norm(bowl_on_box[on] - fake_seed, axis=1)
    assert a[np.argmin(near)] == a.max() and a.min() >= 0.2
    # no seed: the cloud is unchanged, the plane found earlier is kept for this call's support test
    fake._close_seed = None
    fake._target_confirm._support = None
    cloud2, excl2 = AG3S._close_seed_grounding_input(fake, apc, None)
    assert cloud2 is apc and excl2 is None and fake._close_seed_frame["applied"] is False
    assert fake._target_confirm._support is not None and "kept_support_plane_z_m" in fake._close_seed_frame


# ============================================================================ PE3

def test_scene_obstacles_drop_the_held_object_now():
    table_obj = _sphere([0.40, 0.20, 0.03], 0.03, seed=7)
    lifted = _sphere([0.55, -0.10, 0.25], 0.04, seed=8)
    cloud = np.concatenate([table_obj, lifted])
    fake = types.SimpleNamespace(_obstacle_history=[])
    out, rec = AG3S._scene_obstacle_points(fake, cloud, None, None, None)
    assert len(out) and np.any(out[:, 2] > 0.2)                   # off: the lifted object is an obstacle
    fake = types.SimpleNamespace(_obstacle_history=[])
    out, rec = AG3S._scene_obstacle_points(fake, cloud, None, None, None, held_points=lifted[::5] + 0.005)
    assert len(out) and not np.any(out[:, 2] > 0.2) and rec["n_dropped_held"] == 1


# ============================================================================ flags / versions

def test_server_flags_and_versions():
    from AG3S.adapters.panda import profile as pf
    from AG3S.trajopt import serve_safe as ss

    gt24 = pf.GT24_VERSIONS[0]
    pe1, pe123, pe12345, pe123456, pe125cls, pe1234567 = pf.PE_VERSIONS[:6]
    assert list(pf.AG3S_VERSIONS[pe1]) == list(pf.AG3S_VERSIONS[gt24]) + ["--close-seed-grounding"]
    assert list(pf.AG3S_VERSIONS[pe123]) == list(pf.AG3S_VERSIONS[pe1]) + [
        "--destination-after-held", "--scene-obstacle-held-exclude"]
    keys = ("close_seed_grounding", "destination_after_held", "scene_obstacle_held_exclude")
    p = pf.parse_v11ab_args(argv=pf.version_server_argv(pe123) + list(pf.AG3S_VERSIONS[pe123]))
    sec = ss.t43t_identity_section(p)
    assert all(sec.get(k) is True for k in keys)
    assert list(pf.AG3S_VERSIONS[pe12345]) == list(pf.AG3S_VERSIONS[pe123]) + [
        "--passthrough-obstacle-class-tests", "--destination-held-leader"]
    assert list(pf.AG3S_VERSIONS[pe123456]) == list(pf.AG3S_VERSIONS[pe12345]) + ["--no-target-certify"]
    assert list(pf.AG3S_VERSIONS[pe1234567]) == list(pf.AG3S_VERSIONS[pe123456]) + ["--close-seed-attention-ratio", "5"]
    p7 = pf.parse_v11ab_args(argv=pf.version_server_argv(pe1234567) + list(pf.AG3S_VERSIONS[pe1234567]))
    assert ss.t43t_identity_section(p7).get("close_seed_attention_ratio") == 5.0
    cls = list(pf.AG3S_VERSIONS[pe125cls])
    assert "--scene-obstacle-objects" not in cls and "--passthrough-obstacle-beneath" in cls
    assert "--close-seed-grounding" in cls and "--destination-held-leader" in cls
    pf.parse_v11ab_args(argv=pf.version_server_argv(pe125cls) + cls)
    p5 = pf.parse_v11ab_args(argv=pf.version_server_argv(pe12345) + list(pf.AG3S_VERSIONS[pe12345]))
    assert ss.t43t_identity_section(p5).get("destination_held_leader") is True
    p0 = pf.parse_v11ab_args(argv=pf.version_server_argv(gt24) + list(pf.AG3S_VERSIONS[gt24]))
    assert not set(keys) & set(ss.t43t_identity_section(p0))


# ============================================================================ PE4 (TO)

def test_sqp_passes_class_tests_only_when_on():
    from AG3S.trajopt import sqp as S
    from AG3S.trajopt.config import TrajOptConfig

    seen = {}

    class Lin:
        def obstacle_object_rows(self, *a, **kw):
            seen.clear()
            seen.update(kw)
            return float("inf"), None
    for on in (False, True):
        cfg = TrajOptConfig().with_overrides({"sqp": {"passthrough_obstacle_tolerance": 0.01,
                                                      "passthrough_obstacle_class_tests": on}})
        opt = types.SimpleNamespace(config=cfg, linearizer=Lin(), lookahead_from=5)
        assert S.TrajectoryOptimizer._obstacle_row_refuses(opt, np.zeros((10, 8)), np.zeros(8), object(), {}) is False
        assert ("class_tests" in seen) is on


# ============================================================================ PE6

def test_no_target_strict_condition_and_reason():
    from AG3S.trajopt import safe_policy as SP
    from AG3S.trajopt import wire
    from AG3S.trajopt.grasp_latch import GraspPhase

    def cs(status="no_target", certified=True, manip=None, active=False):
        return types.SimpleNamespace(status=types.SimpleNamespace(value=status), geometry_certified=certified,
                                     manipulated=manip, metrics={"exclusion": {"active": active}},
                                     grounding_status=types.SimpleNamespace(value="no_admissible"),
                                     validity=types.SimpleNamespace(value="valid"))
    pol = types.SimpleNamespace(_latch=types.SimpleNamespace(phase=GraspPhase.SEARCHING), no_target_certify=True)
    f = lambda c: SP.SafePolicy._no_target_strict(pol, c)
    assert f(cs())
    assert not f(cs(status="degraded")) and not f(cs(certified=False)) and not f(cs(manip=object()))
    assert not f(cs(active=True))
    pol._latch.phase = GraspPhase.LATCHED
    assert not f(cs())
    assert wire.GATE_DEFAULT["no_target_strict"] == "execute" and "no_target_strict" in wire.REASON_KINDS


# ============================================================================ PE7

def test_attention_vets_the_seed_cut():
    bowl_on_box = np.concatenate([
        np.random.default_rng(1).uniform([0.40, -0.10, 0.10], [0.60, 0.10, 0.1001], size=(2500, 3)),
        _sphere([0.50, 0.0, 0.135], 0.03, seed=4)])
    att = np.r_[np.full(2500, 0.9, np.float32), np.full(400, 0.1, np.float32)]   # the object under the seed is unattended
    apc = AttentionPointCloud(PointCloud(bowl_on_box), att, att)
    for ratio, applied in ((None, True), (5.0, False), (0.05, True)):
        cfg = dataclasses.replace(BASE, close_seed_grounding=True, close_seed_attention_ratio=ratio)
        fake = types.SimpleNamespace(_close_seed=np.array([0.53, 0.0, 0.14]), _target_confirm=_confirm(cfg),
                                     _close_seed_frame=None, _close_seed_plane=None,
                                     config=types.SimpleNamespace(clustering=cfg))
        fake._add_support_plane = lambda plane, f=fake: AG3S._add_support_plane(f, plane)
        cloud, _ = AG3S._close_seed_grounding_input(fake, apc, None)
        assert fake._close_seed_frame["applied"] is applied, (ratio, fake._close_seed_frame)
        assert (cloud is apc) is (not applied)
    # a cut on the current identity only refreshes its geometry: not vetted
    cfg = dataclasses.replace(BASE, close_seed_grounding=True, close_seed_attention_ratio=5.0)
    fake = types.SimpleNamespace(_close_seed=np.array([0.53, 0.0, 0.14]), _target_confirm=_confirm(cfg),
                                 _close_seed_frame=None, _close_seed_plane=None,
                                 config=types.SimpleNamespace(clustering=cfg),
                                 _manipulated=types.SimpleNamespace(centroid=np.array([0.50, 0.0, 0.14])))
    fake._add_support_plane = lambda plane, f=fake: AG3S._add_support_plane(f, plane)
    AG3S._close_seed_grounding_input(fake, apc, None)
    assert fake._close_seed_frame["applied"] is True and fake._close_seed_frame["same_as_identity"] is True
    with pytest.raises(AG3SConfigError):
        AG3SConfig().with_overrides({"clustering": {"close_seed_attention_ratio": 5.0}})   # needs close_seed_grounding
