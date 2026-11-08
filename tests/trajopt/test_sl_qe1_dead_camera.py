"""SL-QE1 (dev-e5, 2026-11-08): `--esdf-dead-camera-seen-ok` — G3's live-camera exception.

B5 (gt23t100) SafeLIBERO t3e0: after an empty close the wrist camera had 52k valid / 28k in-range depth pixels whose points all
fell outside the workspace bounds (n_updated 0) -> `esdf_dead_camera` -> DEGRADED -> not certified -> HOLD 29 calls (G97).
"""
import types

import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError
from AG3S.ag3s.runtime.pipeline import AG3S
from AG3S.ag3s.types import ConstraintValidity


def _stub(flag: bool):
    cfg = AG3SConfig.from_dict({"collision_backend": "esdf", "esdf": {"dead_camera_seen_ok": flag}} if flag
                               else {"collision_backend": "esdf"})
    return types.SimpleNamespace(config=cfg, constraint_robot_model=None, _esdf_cameras_seen=set())


def _field(wrist_updated: int, wrist_in_range: int = 28000):
    return types.SimpleNamespace(stats={"per_camera": {
        "head": {"n_valid_px": 58000, "n_in_range_px": 57000, "n_updated": 41000},
        "left_wrist": {"n_valid_px": 52000, "n_in_range_px": wrist_in_range, "n_updated": wrist_updated}}})


def _codes(notes):
    return [str(n).split(":", 1)[0] for n in notes]


def test_default_off_and_validated():
    assert AG3SConfig().esdf.dead_camera_seen_ok is False
    with pytest.raises(AG3SConfigError):
        AG3SConfig.from_dict({"collision_backend": "esdf", "esdf": {"dead_camera_seen_ok": 1}})


def test_off_is_unchanged_dead_camera_degrades():
    s = _stub(False)
    AG3S._esdf_coverage(s, _field(500), None)               # the wrist wrote voxels before
    notes, validity = AG3S._esdf_coverage(s, _field(0), None)
    assert "esdf_dead_camera" in _codes(notes) and validity == ConstraintValidity.DEGRADED


def test_on_seen_camera_looking_outside_stays_valid():
    s = _stub(True)
    notes, validity = AG3S._esdf_coverage(s, _field(500), None)
    assert validity == ConstraintValidity.VALID and s._esdf_cameras_seen == {"head", "left_wrist"}
    notes, validity = AG3S._esdf_coverage(s, _field(0), None)
    assert _codes(notes) == ["esdf_camera_outside_workspace"] and validity == ConstraintValidity.VALID


def test_on_never_seen_camera_still_degrades():
    s = _stub(True)                                         # extrinsics wrong from the first frame: never wrote a voxel
    notes, validity = AG3S._esdf_coverage(s, _field(0), None)
    assert "esdf_dead_camera" in _codes(notes) and validity == ConstraintValidity.DEGRADED


def test_on_no_in_range_depth_still_degrades():
    s = _stub(True)
    AG3S._esdf_coverage(s, _field(500), None)
    notes, validity = AG3S._esdf_coverage(s, _field(0, wrist_in_range=0), None)
    assert "esdf_dead_camera" in _codes(notes) and validity == ConstraintValidity.DEGRADED


def test_server_flag_plumbing():
    from AG3S.trajopt import serve_safe as ss
    from tests.trajopt.test_t88_passthrough_when_clear import BASE
    ap = ss.build_parser()
    args = ap.parse_args(BASE + ["--esdf-dead-camera-seen-ok"])
    assert ss.dynamic_field_section(args)["dead_camera_seen_ok"] is True
    assert "dead_camera_seen_ok" not in ss.dynamic_field_section(ap.parse_args(BASE))
    from AG3S.adapters.panda import server as ps
    assert "esdf_dead_camera_seen_ok" in ps.PORTED_FINAL_CONFIG_DESTS
    assert "esdf_dead_camera_seen_ok" not in ps.UNPORTED_FINAL_CONFIG_DESTS
    ps.check_unported_flags(args, ap)                       # accepted by the Panda server
