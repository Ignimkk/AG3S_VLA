"""IFG-FOLLOW — a kept target geometry follows the fruit it stands for (`clustering.kept_geometry_follow`).

  - off (default): `follow_kept` does nothing and leaves no record.
  - on, latched or closing: the kept geometry, the anchor and the held centroid move by the fixed-radius
    sphere fit of this frame's points; a touching neighbour does not pull the centre.
  - the gates: too few points, a step over `FOLLOW_MAX_STEP`, not latched / closing → nothing moves.
  - the CLI flag maps to the clustering key and needs AG3S.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, ClusteringConfig
from AG3S.ag3s.stages.geometry import fit_sphere
from AG3S.ag3s.stages.target_grounding import FOLLOW_MAX_STEP, TargetConfirm
from AG3S.ag3s.types import TargetGeometry

R = 0.033
C0 = np.array([0.50, 0.30, 0.85])


def _cap(centre, n=300, seed=0):
    """Points on the camera-facing half (−x) of a sphere of radius R at `centre`."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n * 3, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v = v[v[:, 0] < 0.0][:n]
    return np.asarray(centre, np.float64)[None, :] + R * v


def _confirm(follow=True, latched=True, pre_attach=False):
    conf = TargetConfirm(ClusteringConfig(kept_geometry_follow=follow), max_opening=0.08)
    pts = _cap(C0)
    conf._geometry = TargetGeometry(id=0, points=pts, point_indices=np.zeros(0, np.int64),
                                    centroid=pts.mean(axis=0), bounding_geometry=fit_sphere(pts),
                                    attention_score=1.0, confidence=1.0)
    conf._anchor_geom = conf._geometry
    conf._held = pts.mean(axis=0)
    conf.note_latched(latched)
    conf.note_pre_attach_grasp(pre_attach)
    return conf


def test_off_by_default_is_a_no_op():
    assert ClusteringConfig().kept_geometry_follow is False
    conf = _confirm(follow=False)
    before = conf._geometry.points.copy()
    conf.follow_kept(_cap(C0 + [0.02, 0, 0], seed=1))
    assert conf.follow_record is None
    np.testing.assert_array_equal(conf._geometry.points, before)


@pytest.mark.parametrize("latched,pre_attach", [(True, False), (False, True)])
def test_follows_a_pushed_fruit(latched, pre_attach):
    conf = _confirm(latched=latched, pre_attach=pre_attach)
    shift = np.array([0.018, -0.006, 0.0])
    held0 = conf._held.copy()
    conf.follow_kept(_cap(C0 + shift, seed=1))
    rec = conf.follow_record
    assert rec["applied"], rec
    moved = conf._geometry.centroid - _cap(C0).mean(axis=0)
    assert np.linalg.norm(moved - shift) < 0.002
    np.testing.assert_allclose(conf._anchor_geom.centroid, conf._geometry.centroid)
    np.testing.assert_allclose(conf._held - held0, moved, atol=1e-12)
    np.testing.assert_allclose(conf._geometry.bounding_geometry.center - fit_sphere(_cap(C0)).center, moved,
                               atol=1e-12)


def test_a_touching_neighbour_does_not_pull_the_centre():
    conf = _confirm()
    shift = np.array([0.010, 0.0, 0.0])
    banana = C0 + shift + np.array([0.0, 0.045, 0.0])     # surfaces 12 mm apart, inside the band
    pts = np.vstack([_cap(C0 + shift, seed=1), _cap(banana, n=200, seed=2)])
    conf.follow_kept(pts)
    assert conf.follow_record["applied"]
    moved = conf._geometry.centroid - _cap(C0).mean(axis=0)
    assert np.linalg.norm(moved - shift) < 0.004


def test_gates_leave_the_geometry_alone():
    conf = _confirm()
    before = conf._geometry.points.copy()
    conf.follow_kept(_cap(C0, n=10, seed=1))                    # too few points
    assert conf.follow_record["reason"] == "too_few_points" and not conf.follow_record["applied"]
    conf.follow_kept(_cap(C0 + [FOLLOW_MAX_STEP + 0.006, 0, 0], seed=1))   # too far in one frame
    assert not conf.follow_record["applied"]
    np.testing.assert_array_equal(conf._geometry.points, before)
    idle = _confirm(latched=False, pre_attach=False)             # neither latched nor closing
    idle.follow_kept(_cap(C0 + [0.02, 0, 0], seed=1))
    assert idle.follow_record is None


def test_usable_mask_drops_support_points():
    conf = _confirm()
    pts = _cap(C0 + [0.015, 0, 0], seed=1)
    conf.follow_kept(pts, usable=np.zeros(len(pts), bool))
    assert conf.follow_record["reason"] == "too_few_points"


def test_record_carries_the_follow_only_when_it_ran():
    on, off = _confirm(), _confirm(follow=False)
    for conf in (on, off):
        conf._id = 0
        conf.follow_kept(_cap(C0 + [0.012, 0, 0], seed=1))
    rec = on.manipulated.record()
    assert rec["follow"]["applied"] and rec["follow"]["shift_mm"] > 10.0
    assert "follow" not in off.manipulated.record()


def test_config_validation():
    with pytest.raises(AG3SConfigError):
        AG3SConfig.from_dict({"clustering": {"kept_geometry_follow": "yes"}})


BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def test_cli_flag_maps_to_the_clustering_key():
    from AG3S.trajopt import serve_safe

    parse = lambda *a: serve_safe.build_parser().parse_args([*BASE_ARGV, *a])   # noqa: E731
    assert "kept_geometry_follow" not in serve_safe.t43t_identity_section(parse())
    sec = serve_safe.t43t_identity_section(parse("--kept-geometry-follow"))
    assert sec["kept_geometry_follow"] is True
    assert AG3SConfig.from_dict({"clustering": {"kept_geometry_follow": True}}).clustering.kept_geometry_follow


def test_cli_refuses_the_flag_without_ag3s(monkeypatch, capsys):
    from AG3S.trajopt import serve_safe

    monkeypatch.setattr(sys, "argv", ["serve_safe", *BASE_ARGV, "--no-safe", "--kept-geometry-follow"])
    with pytest.raises(SystemExit) as e:
        serve_safe.main()
    assert e.value.code == 2
    assert "--kept-geometry-follow" in capsys.readouterr().err
