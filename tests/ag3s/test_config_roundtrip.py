"""`AG3SConfig.to_dict` / `with_overrides` keep every field they are not asked to change (T43 T2 bug:
`collision_backend` was missing from `to_dict`, so `with_overrides` reset an `esdf` server to
`primitive` and every chunk held "no scene", V5 / V6). Also `TrajOptConfig`, whose `with_overrides`
the server and the refiner use."""

from __future__ import annotations

import dataclasses

import pytest

from benchmark.ag3s.config import _SECTIONS, AG3SConfig


def _fields(cfg):
    """`{(section or "", field): value}` for every field of the config, top-level ones included."""
    out = {}
    for f in dataclasses.fields(cfg):
        value = getattr(cfg, f.name)
        if f.name in _SECTIONS:
            for g in dataclasses.fields(value):
                out[(f.name, g.name)] = getattr(value, g.name)
        else:
            out[("", f.name)] = value
    return out


BASES = [
    {},
    {"collision_backend": "esdf", "frame_id": "torso"},
    {"collision_backend": "both", "esdf": {"backend": "curobo", "fine_voxel_size": 0.005,
                                           "tsdf_voxel_size": 0.005},
     "clustering": {"subtask_gate": False, "latched_identity_hold": True},
     "pointcloud": {"range_max": 1.7}, "constraint": {"target_field_policy": "exclude_authorized"}},
]

OVERRIDES = [
    {"clustering": {"latched_identity_hold": True}},
    {"clustering": {"closing_keep_geometry": True, "supported_max_bottom": 0.06}},
    {"clustering": {"held_slip_detach": True}},
    {"profiling": {"enabled": True, "warmup_frames": 1}},
    {"esdf": {"voxel_size": 0.01}},
    {"frame_id": "base_link"},
]


def test_every_top_level_field_is_serialized():
    top = [f.name for f in dataclasses.fields(AG3SConfig) if f.name not in _SECTIONS]
    assert set(top) <= set(AG3SConfig().to_dict())
    assert "collision_backend" in top


@pytest.mark.parametrize("base", BASES)
def test_to_dict_from_dict_is_the_identity(base):
    cfg = AG3SConfig.from_dict(base)
    back = AG3SConfig.from_dict(cfg.to_dict())
    assert _fields(back) == _fields(cfg)
    assert {k: type(v) for k, v in _fields(back).items()} == \
        {k: type(v) for k, v in _fields(cfg).items()}
    assert back == cfg


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("over", OVERRIDES)
def test_with_overrides_keeps_every_field_it_does_not_touch(base, over):
    cfg = AG3SConfig.from_dict(base)
    new = cfg.with_overrides(over)
    touched = {(k, kk) for k, v in over.items() if isinstance(v, dict) for kk in v} | \
              {("", k) for k, v in over.items() if not isinstance(v, dict)}
    a, b = _fields(cfg), _fields(new)
    for key in a:
        if key in touched:
            section, name = key
            want = over[section][name] if section else over[name]
            assert b[key] == want, key
        else:
            assert b[key] == a[key], key
            assert type(b[key]) is type(a[key]), key


def test_the_t2_bug():
    cfg = AG3SConfig.from_dict({"collision_backend": "esdf"})
    assert cfg.with_overrides({"clustering": {"latched_identity_hold": True}}).collision_backend == "esdf"
    assert cfg.with_overrides({}).collision_backend == "esdf"


def test_trajopt_config_with_overrides_keeps_every_field():
    from benchmark.trajopt.config import _SECTIONS as TS, TrajOptConfig

    assert all(f.name in TS for f in dataclasses.fields(TrajOptConfig))      # no top-level scalars
    cfg = TrajOptConfig.from_dict({"collision": {"backend": "esdf", "obstacle_margin": 0.01},
                                   "limits": {"servo_model": True},
                                   "safety": {"require_certified_geometry": False}})
    assert cfg.with_overrides({}) == cfg
    new = cfg.with_overrides({"sqp": {"max_iterations": 7}})
    assert new.sqp.max_iterations == 7
    assert dataclasses.replace(new, sqp=cfg.sqp) == cfg
