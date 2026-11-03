"""T81 — the old place is empty: let the target go and pick again (`clustering.target_vacancy_release`).

  1. off (default): nothing reads the views, no `vacancy` record, the decisions are the old ones;
  2. the per-point judgement on synthetic depth: empty / occluded (in front, robot pixel) / present;
  3. two consecutive empty requests release (one does not; a present request in between starts over);
  4. HELD (frozen, not closing) and after PLACED: never released; closing (latched ∧ pre-attach) is;
  5. the re-pick prefers a candidate within `target_vacancy_near` of the old centroid, and the subtask
     gate does not block it;
  6. a small move (the object still covers its old place) is not released;
  7. pipeline: off = `_vacancy_views` never called and the record keys of before; on = the record, the
     release with the single-camera depth path, and nothing else changed before the release.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, ClusteringConfig
from AG3S.ag3s.runtime.pipeline import AG3S
from AG3S.ag3s.stages.target_grounding import VACANCY_RELEASED, ClusterInfo, TargetConfirm
from AG3S.ag3s.stages.target_vacancy import VacancyView, judge, judge_camera, sample_points, vacant
from AG3S.ag3s.stages.geometry import fit_sphere
from AG3S.ag3s.types import TargetGeometry
from tests.ag3s.fixtures import (
    DEFAULT_ARM_Q,
    DEFAULT_OBJECTS,
    LABEL_BOX,
    LABEL_SPHERE,
    PlanarTwoLinkArm,
    make_scene,
)

OPENING = 0.0715
R_APPLE = 0.031
APPLE = np.array([0.50, 0.00, 0.80])
TABLE_Z = 0.75
K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
H, W = 480, 640
#: camera 0.7 m above the apple, looking straight down (camera z = base −z)
T_CAM = np.eye(4)
T_CAM[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
T_CAM[:3, 3] = [0.50, 0.00, 1.50]
PICK = {"pick": 0.9, "place": 0.08, "home": 0.02}
PLACE = {"pick": 0.05, "place": 0.9, "home": 0.05}


def _render(balls=(), boxes_z=()):
    """Depth (m) of the downward camera: the table plane, spheres `(centre, r)`, and horizontal plates
    `(x0, x1, y0, y1, z)` (a hand over the old place)."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)   # camera, z = 1
    d = np.full((H, W), T_CAM[2, 3] - TABLE_Z)
    dirs = rays @ T_CAM[:3, :3].T
    o = T_CAM[:3, 3]
    for c, r in balls:
        oc = o - np.asarray(c)
        a = (dirs ** 2).sum(-1)
        b = 2.0 * (dirs @ oc)
        cc = oc @ oc - r * r
        disc = b * b - 4 * a * cc
        hit = disc >= 0
        t = np.where(hit, (-b - np.sqrt(np.maximum(disc, 0.0))) / (2 * a), np.inf)
        d = np.where(hit & (t > 0) & (t < d), t, d)     # t is the camera z (rays have z = 1)
    for x0, x1, y0, y1, z in boxes_z:
        t = T_CAM[2, 3] - z
        p = o + dirs * t[..., None] if np.ndim(t) else o + dirs * t
        inside = (p[..., 0] >= x0) & (p[..., 0] <= x1) & (p[..., 1] >= y0) & (p[..., 1] <= y1)
        d = np.where(inside & (t < d), t, d)
    return d


def _top_cap(centre, r=R_APPLE, n=200, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n * 3, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v = v[v[:, 2] > 0.2][:n]
    return np.asarray(centre)[None, :] + r * v


def _view(depth, mask=None):
    return [VacancyView("head", depth, K, T_CAM, mask)]


OLD = _top_cap(APPLE)
D_PRESENT = _render(balls=[(APPLE, R_APPLE)])
D_EMPTY = _render()
D_HAND = _render(boxes_z=[(0.40, 0.60, -0.10, 0.10, 0.95)])


# --------------------------------------------------------------------------- 2. the judgement
def test_judgement_on_synthetic_depth():
    p = judge(OLD, _view(D_PRESENT), delta=0.02, min_points=10)
    assert p["decidable"] and p["present"] > 0.9 and p["empty"] < 0.05   # silhouette-edge pixels
    e = judge(OLD, _view(D_EMPTY), delta=0.02, min_points=10)
    assert e["decidable"] and e["empty"] > 0.9 and e["present"] < 0.1
    o = judge(OLD, _view(D_HAND), delta=0.02, min_points=10)
    assert o["decidable"] and o["occluded"] > 0.9 and o["empty"] == 0.0
    # robot pixels: occluded, never empty — and not judged (M counts non-robot points)
    r = judge(OLD, _view(D_EMPTY, np.ones((H, W), bool)), delta=0.02, min_points=10)
    assert not r["decidable"] and r["n_robot"] == len(OLD) and r["n_judged"] == 0
    half = np.zeros((H, W), bool)
    half[:, :320] = True
    rh = judge(OLD, _view(D_EMPTY, half), delta=0.02, min_points=10)
    assert rh["decidable"] and rh["occluded"] == pytest.approx(rh["n_robot"] / rh["n_valid"], abs=1e-3)
    assert rh["empty"] == pytest.approx(1.0 - rh["occluded"], abs=1e-3)
    assert vacant(e, p_empty=0.5, p_present=0.2) and not vacant(p, p_empty=0.5, p_present=0.2)
    assert not vacant(o, p_empty=0.5, p_present=0.2)


def test_out_of_image_and_invalid_depth_do_not_count_and_the_best_camera_decides():
    blind = D_EMPTY.copy()
    blind[:] = 0.0                                  # no valid depth at all
    far = np.eye(4)
    far[:3, 3] = [5.0, 5.0, 5.0]                     # looking away: nothing in front of it
    views = [VacancyView("blind", blind, K, T_CAM), VacancyView("away", D_EMPTY, K, far),
             VacancyView("head", D_EMPTY, K, T_CAM)]
    ev = judge(OLD, views, delta=0.02, min_points=10)
    assert ev["camera"] == "head" and ev["decidable"]
    assert [c["n_judged"] for c in ev["per_camera"]][:2] == [0, 0]
    assert judge(OLD, views[:2], delta=0.02, min_points=10)["why"] == "undecidable"
    assert judge(OLD, [], delta=0.02, min_points=10)["why"] == "no_views"
    c = judge_camera(OLD, views[2], 0.02)
    assert c["n_judged"] == c["n_empty"] + c["n_front"] + c["n_present"]


def test_sample_points_is_deterministic_and_bounded():
    s = sample_points(OLD, 50)
    assert len(s) == 50 and np.array_equal(s, sample_points(OLD, 50))
    assert np.array_equal(s[0], OLD[0]) and np.array_equal(s[-1], OLD[-1])
    assert len(sample_points(OLD[:10], 50)) == 10


# --------------------------------------------------------------------------- TargetConfirm
def _clusters(*items):
    clouds, infos, start = [], [], 0
    for i, (pts, score) in enumerate(items):
        pts = np.asarray(pts, np.float64)
        idx = np.arange(start, start + len(pts))
        start += len(pts)
        clouds.append(pts)
        infos.append(ClusterInfo(
            id=i, point_indices=idx, centroid=pts.mean(axis=0), mean_attention=score,
            max_attention=score, attention_mass=score, point_count=len(pts),
            spatial_compactness=1.0, distance_from_attention_peak=0.0, target_score=score))
    infos.sort(key=lambda c: -c.target_score)
    return infos, np.concatenate(clouds)


def _confirm(on=True, **kw):
    cfg = ClusteringConfig(min_points=10, target_vacancy_release=on, **kw)
    return TargetConfirm(cfg, max_opening=OPENING, overlap_distance=0.005)


#: far away from the apple (a hand / crate-free decoy), and a pushed apple 100 mm along +y
DECOY = np.array([0.50, -0.40, 0.80])
PUSHED = APPLE + np.array([0.0, 0.10, 0.0])


def _frame(conf, items, depth, *, mask=None):
    clusters, pts = _clusters(*items)
    conf.note_views(_view(depth, mask))
    return conf.select(clusters, points=pts)


def _adopt_apple(conf):
    pts = _top_cap(APPLE)
    d = _frame(conf, [(pts, 0.9)], D_PRESENT)
    assert d.mode == "first"
    # what `ground_target` does right after a `first`: the observed geometry stands from here on
    conf.observe(TargetGeometry(id=0, points=pts, point_indices=np.arange(len(pts)), centroid=pts.mean(axis=0),
                                bounding_geometry=fit_sphere(pts), attention_score=0.9, confidence=0.9))
    return d


def test_off_is_not_read_and_leaves_no_record():
    assert ClusteringConfig().target_vacancy_release is False
    conf = _confirm(on=False)
    _adopt_apple(conf)
    for _ in range(4):
        d = _frame(conf, [(_top_cap(DECOY, seed=3), 0.05)], D_EMPTY)
        assert d.mode == "occluded" and d.reason == "unobserved"
    assert conf.vacancy_record is None
    assert "vacancy" not in conf.manipulated.record()


def test_two_consecutive_empty_requests_release_and_repick_the_pushed_apple():
    conf = _confirm()
    _adopt_apple(conf)
    depth = _render(balls=[(PUSHED, R_APPLE)])
    items = [(_top_cap(DECOY, seed=3), 0.8), (_top_cap(PUSHED, seed=4), 0.3)]
    d1 = _frame(conf, items, depth)
    assert d1.mode == "occluded" and conf.vacancy_record["decision"] == "keep"
    assert conf.vacancy_record["streak"] == 1 and conf.vacancy_record["verdict"] == "empty"
    d2 = _frame(conf, items, depth)
    rec = conf.vacancy_record
    assert rec["decision"] == "release" and rec["released_id"] == 0
    # the re-pick: the near (pushed) apple, not the higher-scored decoy 400 mm away
    assert d2.mode == "first" and d2.reason == VACANCY_RELEASED and d2.manipulated_id == 1
    assert np.linalg.norm(d2.cluster.centroid - PUSHED) < 0.02
    assert rec["repick"]["near"] and rec["repick"]["distance_mm"] < 120
    assert conf.manipulated.record()["vacancy"]["decision"] == "release"


def test_a_present_request_in_between_starts_the_count_over():
    conf = _confirm()
    _adopt_apple(conf)
    gone = [(_top_cap(DECOY, seed=3), 0.05)]
    _frame(conf, gone, D_EMPTY)
    _frame(conf, gone, D_PRESENT)          # the apple is there after all (e.g. a missed cluster)
    assert conf.vacancy_record["streak"] == 0
    d = _frame(conf, gone, D_EMPTY)
    assert conf.vacancy_record["streak"] == 1 and d.mode == "occluded"


def test_occluded_and_undecidable_keep():
    conf = _confirm()
    _adopt_apple(conf)
    gone = [(_top_cap(DECOY, seed=3), 0.05)]
    for depth, mask in ((D_HAND, None), (D_EMPTY, np.ones((H, W), bool)), (D_HAND, None)):
        d = _frame(conf, gone, depth, mask=mask)
        assert d.mode == "occluded" and conf.vacancy_record["decision"] == "keep"
        assert conf.vacancy_record["streak"] == 0
    conf.note_views(None)                   # no cameras at all
    clusters, pts = _clusters(*gone)
    conf.select(clusters, points=pts)
    assert conf.vacancy_record["verdict"] == "no_views" and conf.vacancy_record["decision"] == "keep"


def test_held_never_released_closing_is():
    conf = _confirm()
    _adopt_apple(conf)
    gone = [(_top_cap(DECOY, seed=3), 0.05)]
    conf.freeze(True)                       # HELD: frozen, not closing (latch not LATCHED / CLOSING)
    for _ in range(3):
        _frame(conf, gone, D_EMPTY)
        assert conf.vacancy_record == {"applies": False, "why": "held"}
    assert conf.manipulated is not None and conf.manipulated.id == 0
    conf.note_latched(True)
    conf.note_pre_attach_grasp(True)        # CLOSING, nothing attached
    _frame(conf, gone, D_EMPTY)
    assert conf.vacancy_record["latch"] == "closing" and conf.vacancy_record["streak"] == 1
    d = _frame(conf, gone, D_EMPTY)
    assert conf.vacancy_record["decision"] == "release"
    # the frozen re-pick: the only candidate is 400 mm away → no target (never a far candidate)
    assert d.mode == "no_admissible" and d.reason == VACANCY_RELEASED and conf.manipulated is None
    assert conf.vacancy_record["repick"] == {"near": False, "n_near": 0, "n_candidates": 1}


def test_repick_is_near_only_and_stays_pending():
    """Lead decision (2026-11-01): after a release only admissible candidates within `target_vacancy_near`
    of the old centre; none → no target, never the higher-scored duplicate elsewhere."""
    conf = _confirm()
    _adopt_apple(conf)
    far_apple = [(_top_cap(DECOY, seed=3), 0.9)]           # a duplicate apple 400 mm away, high score
    _frame(conf, far_apple, D_EMPTY)
    d = _frame(conf, far_apple, D_EMPTY)
    assert conf.vacancy_record["decision"] == "release"
    assert d.mode == "no_admissible" and d.cluster is None and conf.manipulated is None
    for _ in range(4):                                       # still only the far one: still no target
        d = _frame(conf, far_apple, D_EMPTY)
        assert d.mode == "no_admissible" and d.reason == VACANCY_RELEASED and conf.manipulated is None
        assert conf.vacancy_record["reselect_pending"] == 0
    # the pushed apple comes into view near the old place → picked, not the far one
    d = _frame(conf, far_apple + [(_top_cap(PUSHED, seed=4), 0.2)], _render(balls=[(PUSHED, R_APPLE)]))
    assert d.mode == "first" and d.reason == VACANCY_RELEASED
    assert np.linalg.norm(d.cluster.centroid - PUSHED) < 0.02 and conf.vacancy_record["repick"]["near"]
    # once adopted, the ordinary rules are back (nothing pending)
    assert conf._vacancy_reselect is None


def test_not_after_placed():
    conf = _confirm()
    _adopt_apple(conf)
    conf.note_placed(True)
    for _ in range(3):
        _frame(conf, [(_top_cap(DECOY, seed=3), 0.05)], D_EMPTY)
        assert conf.vacancy_record["why"] == "placed"


def test_subtask_gate_does_not_block_the_repick():
    conf = _confirm(subtask_gate=True)
    _adopt_apple(conf)
    for _ in range(3):
        conf.note_subtask(PLACE)
    assert conf.subtask_label == "place"
    depth = _render(balls=[(PUSHED, R_APPLE)])
    items = [(_top_cap(PUSHED, seed=4), 0.3)]
    conf.note_subtask(PLACE)
    _frame(conf, items, depth)
    conf.note_subtask(PLACE)
    d = _frame(conf, items, depth)
    assert d.mode == "first" and d.reason == VACANCY_RELEASED and conf.subtask_blocked is None
    # a later first (nothing to re-pick) is gated again
    conf._forget_manipulated()
    conf.note_subtask(PLACE)
    d = _frame(conf, items, depth)
    assert d.mode == "subtask_gated"


def test_release_in_a_no_cluster_call_repicks_on_the_next_call():
    conf = _confirm()
    _adopt_apple(conf)
    for _ in range(2):
        conf.note_views(_view(D_EMPTY))
        conf.unseen()
    assert conf.vacancy_record["decision"] == "release" and conf.manipulated is None
    d = _frame(conf, [(_top_cap(DECOY, seed=3), 0.8), (_top_cap(PUSHED, seed=4), 0.3)],
               _render(balls=[(PUSHED, R_APPLE)]))
    assert d.mode == "first" and d.reason == VACANCY_RELEASED
    assert conf.vacancy_record["reselect_pending"] == 0 and conf.vacancy_record["repick"]["near"]


def test_a_small_move_is_not_released():
    conf = _confirm()
    _adopt_apple(conf)
    moved = APPLE + np.array([0.0, 0.012, 0.0])      # 12 mm: the apple still covers its old place
    depth = _render(balls=[(moved, R_APPLE)])
    for _ in range(4):
        _frame(conf, [(_top_cap(DECOY, seed=3), 0.05)], depth)
        assert conf.vacancy_record["decision"] == "keep"
        assert conf.vacancy_record["present"] > 0.2


def test_config_validation():
    with pytest.raises(AG3SConfigError):
        ClusteringConfig(target_vacancy_release="yes").validate()
    for bad in ({"target_vacancy_delta": 0.0}, {"target_vacancy_p_empty": 1.5},
                {"target_vacancy_min_points": 0}, {"target_vacancy_frames": 1.5}):
        with pytest.raises(AG3SConfigError):
            ClusteringConfig(**bad).validate()
    ClusteringConfig(target_vacancy_release=True).validate()


# --------------------------------------------------------------------------- 7. pipeline
NO_APPLE = tuple(o for o in DEFAULT_OBJECTS if o.label != LABEL_SPHERE)
SCENE_KW = dict(phase="approach", robot_state=DEFAULT_ARM_Q)


@pytest.fixture(scope="module")
def scenes():
    arm = PlanarTwoLinkArm()
    return arm, (make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q),
                 make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q, objects=NO_APPLE,
                            target_label=LABEL_BOX))


def _inputs(scene):
    return dict(attention_map=scene.attention_grid, depth=scene.depth,
                camera_intrinsics=scene.camera_intrinsics, T_base_cam=scene.T_base_cam,
                image_hw=scene.hw, **SCENE_KW)


def _run(arm, scenes_seq, **clustering):
    ag3s = AG3S(AG3SConfig.from_dict({"clustering": clustering} if clustering else {}), robot_model=arm)
    calls = []
    orig = ag3s._vacancy_views
    ag3s._vacancy_views = lambda *a, **k: calls.append(1) or orig(*a, **k)
    out = [ag3s.process(**_inputs(s)) for s in scenes_seq]
    return out, calls


def _strip(m):
    if m is None:
        return None
    return {k: v for k, v in m.items() if k != "vacancy"}


def test_pipeline_off_never_builds_views_and_on_changes_nothing_before_a_release(scenes):
    arm, (with_apple, gone) = scenes
    seq = [with_apple, with_apple, gone, gone, gone]
    off, calls_off = _run(arm, seq)
    if off[0].target is None:
        pytest.skip("grounding declined the apple scene")
    assert calls_off == []
    for cs in off:
        assert cs.metrics["manipulated"] is None or "vacancy" not in cs.metrics["manipulated"]
    # on, but a release can never come (1000 requests): every record but the vacancy key is the same
    on, calls_on = _run(arm, seq, target_vacancy_release=True, target_vacancy_frames=1000)
    assert len(calls_on) == len(seq)
    for a, b in zip(off, on):
        assert _strip(a.metrics["manipulated"]) == _strip(b.metrics["manipulated"])
        assert a.metrics["target_grounding_status"] == b.metrics["target_grounding_status"]
        assert (a.target is None) == (b.target is None)
        if a.target is not None:
            np.testing.assert_array_equal(a.target.points, b.target.points)
    vac = on[3].metrics["manipulated"]["vacancy"]
    assert vac["applies"] and vac["camera"] == "camera" and vac["verdict"] in ("empty", "kept", "undecidable")


def test_pipeline_on_releases_the_apple_whose_place_the_depth_sees_through(scenes):
    arm, (with_apple, gone) = scenes
    on, _ = _run(arm, [with_apple, gone, gone, gone], target_vacancy_release=True)
    if on[0].target is None:
        pytest.skip("grounding declined the apple scene")
    recs = [cs.metrics["manipulated"] for cs in on]
    assert recs[1]["vacancy"]["verdict"] == "empty" and recs[1]["vacancy"]["empty"] >= 0.5
    released = [r for r in recs if r is not None and (r.get("vacancy") or {}).get("decision") == "release"]
    assert len(released) == 1 and released[0] is recs[2]
    # after the release the old apple id is gone (re-picked under a new id, or nothing)
    assert recs[2]["id"] != 0


# --------------------------------------------------------------------------- serve_safe flag
BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def test_cli_flag_maps_to_the_clustering_keys():
    from AG3S.trajopt import serve_safe

    parse = lambda *a: serve_safe.build_parser().parse_args([*BASE_ARGV, *a])   # noqa: E731
    assert not any(k.startswith("target_vacancy") for k in serve_safe.t43t_identity_section(parse()))
    assert serve_safe.target_vacancy_section(parse()) == {}
    sec = serve_safe.t43t_identity_section(parse("--target-vacancy-release"))
    assert sec["target_vacancy_release"] is True and len([k for k in sec if k.startswith("target_vacancy")]) == 1
    sec = serve_safe.t43t_identity_section(parse(
        "--target-vacancy-release", "--target-vacancy-delta-mm", "15", "--target-vacancy-p-empty", "0.6",
        "--target-vacancy-p-present", "0.1", "--target-vacancy-min-points", "20", "--target-vacancy-samples", "150"))
    assert sec["target_vacancy_delta"] == pytest.approx(0.015) and sec["target_vacancy_p_empty"] == 0.6
    assert sec["target_vacancy_p_present"] == 0.1 and sec["target_vacancy_min_points"] == 20
    assert sec["target_vacancy_samples"] == 150
    cfg = AG3SConfig.from_dict({"clustering": sec}).clustering
    assert all(getattr(cfg, k) == v for k, v in sec.items())


@pytest.mark.parametrize("argv,needle", [
    (["--no-safe", "--target-vacancy-release"], "--target-vacancy-release"),
    (["--target-vacancy-delta-mm", "15"], "--target-vacancy-release"),
])
def test_cli_refusals(monkeypatch, capsys, argv, needle):
    import sys

    from AG3S.trajopt import serve_safe

    monkeypatch.setattr(sys, "argv", ["serve_safe", *BASE_ARGV, *argv])
    with pytest.raises(SystemExit) as e:
        serve_safe.main()
    assert e.value.code == 2
    assert needle in capsys.readouterr().err
