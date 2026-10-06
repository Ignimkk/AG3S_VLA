"""T43 FR — 손가락 몸통 구 수를 줄이는 선택지 (`--finger-cover-solver` · `-gap` · `-max-spheres`, 기본 = FC).

사용자 판정 (2026-10-06 11:10): "구의 개수가 너무 많다면 줄일 수 있는 방안도 TO 시간에 따라서 결정." 기본
`--finger-cover` 의 동작은 바꾸지 않는다. 선택지는 새 flag 이고, 안 주면 FC 와 비트로 같다.

이 파일이 고정하는 것:

1. **안 주면 FC 그대로** — 기본값을 명시해서 불러도 구 · 보고서가 비트로 같고, 보고서에 새 키가 없다.
   CLI 를 안 주면 새 키가 넘어가지 않는다. 새 flag 만 주거나 잘못된 값이면 `SystemExit`.
2. **`solver="ilp"`** — 미리 푼 답 (`finger_cover_presets`) 마다: 그 답을 쓰고 (`source == "preset"`),
   greedy 보다 적고, 보장은 greedy 와 같다 (대상 표본 전부 `gap` 안, 넘침 ≤ overflow + 표본 여유, 중심은
   mesh 안). 답이 mesh 에 안 맞거나 풀이가 실패하면 greedy 로 돌아간다. 작은 문제에서 최소 개수를 찾는다.
3. **`gap`** — 대상 표본이 전부 합집합에서 `gap` 안이고, 구가 준다. 넘침 보장은 그대로.
4. **`max_spheres`** — 손가락마다 그 수 이하, 못 덮은 표본을 보고서가 센다. `ilp` 와 함께면 거절.
5. 모델 배선 — `finger_cover` dict 의 새 키가 `finger_body_spheres` 까지 가고, 네 손가락이 같은 구,
   손바닥은 그대로, 최대 개도 = mesh 간격 − 2 overflow.
"""

from __future__ import annotations

import logging
import pathlib

import numpy as np
import pytest

MODEL = pathlib.Path(
    "/mnt/dev/work/pi05_TO_hybrid/rby1_description/models/rby1a/mujoco/model_transport.xml")
E3B = dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05)
FINGERS = ("ee_finger_l1", "ee_finger_l2", "ee_finger_r1", "ee_finger_r2")
#: 넘침 실측 여유 (구 표면 표본 · mesh 표면 표본 간격) — FC 테스트와 같다.
SAMPLE_TOL = 0.0003


@pytest.fixture(scope="module")
def mj():
    mujoco = pytest.importorskip("mujoco")
    if not MODEL.exists():  # pragma: no cover - asset-dependent
        pytest.skip(f"model not present: {MODEL}")
    return mujoco, mujoco.MjModel.from_xml_path(str(MODEL))


@pytest.fixture(scope="module")
def scene(mj):
    from benchmark.ag3s.experiments.sources.mujoco_source import TransportScene

    mujoco, model = mj
    return TransportScene.attach(model, mujoco.MjData(model))


def _build(scene, **kw):
    from benchmark.ag3s.experiments.reports.grounding_report import (
        GRIPPER_LINKS, build_constraint_robot_model)

    return build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=dict(E3B),
                                        **kw)


@pytest.fixture(scope="module")
def bare(scene):
    return _build(scene)


@pytest.fixture(scope="module")
def surf(mj, bare):
    from benchmark.ag3s.experiments.sources.gripper_cover import link_surface

    _, model = mj
    return link_surface(model, bare, "ee_finger_l1", spacing=0.0005)


@pytest.fixture(scope="module")
def targets(surf):
    from benchmark.ag3s.experiments.sources.gripper_cover import finger_body_targets

    return surf.points[finger_body_targets(surf)]


@pytest.fixture(scope="module")
def default(surf):
    from benchmark.ag3s.experiments.sources.gripper_cover import finger_body_spheres

    return finger_body_spheres(surf)


def _guarantees(surf, targets, C, R, overflow, gap):
    """대상 전부 `gap` 안 · 넘침 ≤ overflow + 표본 여유 · 중심은 mesh 안 · 반지름 = 깊이 + overflow 이하."""
    from benchmark.ag3s.experiments.sources.gripper_cover import _gap, _sampled_overflow, hull_depth

    g = _gap(targets, C, R)
    assert g.max() <= gap + 1e-9
    assert _sampled_overflow(C, R, surf.hulls, surf.points) <= overflow + SAMPLE_TOL
    assert np.all(hull_depth(C, surf.hulls) > 0.0)


# =============================================================================================
# 1. 안 주면 FC 그대로
# =============================================================================================


def test_explicit_defaults_are_bit_identical_to_fc(surf, default):
    from benchmark.ag3s.experiments.sources.gripper_cover import finger_body_spheres

    C, R, rep = finger_body_spheres(surf, gap=0.0, max_spheres=None, solver="greedy")
    assert np.array_equal(C, default[0]) and np.array_equal(R, default[1])
    assert rep == default[2]
    assert not {"solver", "gap_mm", "max_spheres", "source", "ilp"} & set(rep)
    assert rep["n_spheres"] == 56 and rep["n_uncovered"] == 0


def test_model_report_has_no_new_keys_when_not_given(scene):
    fc = _build(scene, finger_cover={"overflow": 0.0025})
    assert fc.finger_cover_report["options"] == {"hands": "both", "overflow": 0.0025}
    assert fc.n_spheres == 270


def test_cli_new_flags_default_to_nothing_and_build_options():
    from benchmark.trajopt import serve_safe

    base = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]
    p = serve_safe.build_parser()
    a = p.parse_args(base + ["--finger-cover"])
    assert a.finger_cover_solver is None and a.finger_cover_gap is None
    assert a.finger_cover_max_spheres is None
    assert serve_safe.finger_cover_options(a) == {"hands": "both"}
    assert serve_safe.finger_cover_options(p.parse_args(
        base + ["--finger-cover", "--finger-cover-overflow", "4", "--finger-cover-solver", "ilp"])) == \
        {"hands": "both", "overflow": 0.004, "solver": "ilp"}
    assert serve_safe.finger_cover_options(p.parse_args(
        base + ["--finger-cover", "left", "--finger-cover-gap", "1", "--finger-cover-max-spheres",
                "16"])) == {"hands": "left", "gap": 0.001, "max_spheres": 16}
    for alone in (["--finger-cover-solver", "ilp"], ["--finger-cover-gap", "1"],
                  ["--finger-cover-max-spheres", "8"]):
        with pytest.raises(SystemExit):
            serve_safe.finger_cover_options(p.parse_args(base + alone))
    for bad in (["--finger-cover-gap", "-1"], ["--finger-cover-gap", "nan"],
                ["--finger-cover-max-spheres", "0"],
                ["--finger-cover-max-spheres", "8", "--finger-cover-solver", "ilp"]):
        with pytest.raises(SystemExit):
            serve_safe.finger_cover_options(p.parse_args(base + ["--finger-cover"] + bad))
    with pytest.raises(SystemExit):
        p.parse_args(base + ["--finger-cover", "--finger-cover-solver", "exact"])


def test_announce_mentions_the_reduction_only_when_given(caplog):
    from benchmark.trajopt import serve_safe

    link = {"n_spheres": 19, "radius_mm": [7.4, 9.5], "n_uncovered": 0, "n_targets": 20774,
            "overflow_sampled_mm": 4.0}
    rep = {"n_spheres_before": 122, "n_spheres_after": 122, "links": {"ee_finger_l1": link},
           "options": {"hands": "both", "overflow": 0.004}}
    with caplog.at_level(logging.WARNING):
        serve_safe.announce_finger_cover(rep)
    assert not any("T43 FR" in r.getMessage() for r in caplog.records)
    caplog.clear()
    rep["options"]["solver"] = "ilp"
    rep["links"]["ee_finger_l1"] = {**link, "source": "preset"}
    with caplog.at_level(logging.WARNING):
        serve_safe.announce_finger_cover(rep)
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "T43 FR" in msg and "ilp" in msg and "preset" in msg


def test_unknown_values_are_rejected(surf, scene):
    from benchmark.ag3s.experiments.sources.gripper_cover import finger_body_spheres

    with pytest.raises(ValueError):
        finger_body_spheres(surf, solver="exact")
    with pytest.raises(ValueError):
        finger_body_spheres(surf, gap=-0.001)
    with pytest.raises(ValueError):
        finger_body_spheres(surf, max_spheres=0)
    with pytest.raises(ValueError):
        finger_body_spheres(surf, max_spheres=8, solver="ilp")
    with pytest.raises(ValueError):
        _build(scene, finger_cover={"overflow": 0.0025, "min_radius": 0.004})


# =============================================================================================
# 2. solver = ilp
# =============================================================================================


def _presets():
    from benchmark.ag3s.experiments.sources.finger_cover_presets import FINGER_BODY_ILP_PRESETS

    return sorted(FINGER_BODY_ILP_PRESETS)


@pytest.mark.parametrize("key", _presets())
def test_every_ilp_preset_is_used_fewer_and_keeps_the_guarantees(surf, targets, key):
    from benchmark.ag3s.experiments.sources.finger_cover_presets import FINGER_BODY_ILP_PRESETS
    from benchmark.ag3s.experiments.sources.gripper_cover import finger_body_spheres

    overflow, gap = key[0] * 1e-6, key[1] * 1e-6
    C, R, rep = finger_body_spheres(surf, overflow=overflow, gap=gap, solver="ilp")
    pre = FINGER_BODY_ILP_PRESETS[key]
    assert rep["source"] == "preset" and rep["solver"] == "ilp"
    assert rep["n_spheres"] == pre["n"] == len(pre["ijk"]) < rep["greedy_n_spheres"] == pre["greedy_n"]
    _guarantees(surf, targets, C, R, overflow, gap)
    if gap == 0.0:
        assert rep["n_uncovered"] == 0 and rep["max_gap_mm"] == 0.0


def test_ilp_falls_back_to_greedy_when_the_preset_does_not_fit_and_the_solve_fails(
        surf, default, monkeypatch, caplog):
    from benchmark.ag3s.experiments.sources import finger_cover_presets as P
    from benchmark.ag3s.experiments.sources import gripper_cover as GC

    monkeypatch.setitem(P.FINGER_BODY_ILP_PRESETS, (2500, 0),
                        {"n": 1, "greedy_n": 56, "optimal": True, "dual_bound": 1, "sec": 0,
                         "ijk": ((10, 30, 60),)})
    calls = []

    def no_answer(target, C, R, **kw):
        calls.append(C.shape[0])
        return None, {"status": 1, "optimal": False}

    monkeypatch.setattr(GC, "_ilp_cover", no_answer)
    with caplog.at_level(logging.WARNING):
        C, R, rep = GC.finger_body_spheres(surf, solver="ilp")
    assert "맞지 않습니다" in caplog.text
    assert calls and rep["source"] == "greedy_fallback"
    assert np.array_equal(C, default[0]) and np.array_equal(R, default[1])

    # 풀이가 덮지 못하는 답을 내도 쓰지 않는다
    monkeypatch.setattr(GC, "_ilp_cover", lambda target, C, R, **kw: (np.arange(3), {"optimal": True}))
    C, R, rep = GC.finger_body_spheres(surf, solver="ilp", use_presets=False)
    assert rep["source"] == "greedy_fallback" and rep["n_spheres"] == 56


def test_ilp_cover_finds_the_minimum_on_a_small_problem():
    from benchmark.ag3s.experiments.sources.gripper_cover import _greedy_exact_cover, _ilp_cover

    # 점 6 개를 한 줄에. greedy 는 가운데 큰 구 (4 점) 를 먼저 골라 3 개가 되지만, 최소는 2 개다.
    target = np.array([[x, 0.0, 0.0] for x in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)])
    C = np.array([[2.5, 0, 0], [1.0, 0, 0], [4.0, 0, 0], [0.0, 0, 0], [5.0, 0, 0]])
    R = np.array([1.6, 1.05, 1.05, 0.1, 0.1])
    g, _ = _greedy_exact_cover(target, C, R)
    sel, info = _ilp_cover(target, C, R, time_limit=30)
    assert len(g) == 3 and sorted(sel.tolist()) == [1, 2] and info["optimal"]


# =============================================================================================
# 3. gap · 4. max_spheres
# =============================================================================================


@pytest.mark.parametrize("gap,n_max", [(0.0005, 48), (0.001, 32)])
def test_gap_covers_every_sample_within_gap_with_fewer_spheres(surf, targets, gap, n_max):
    from benchmark.ag3s.experiments.sources.gripper_cover import finger_body_spheres

    C, R, rep = finger_body_spheres(surf, gap=gap)
    assert rep["n_spheres"] <= n_max < 56
    assert rep["gap_mm"] == pytest.approx(gap * 1e3)
    assert 0 < rep["n_uncovered"] and rep["max_gap_mm"] <= gap * 1e3 + 1e-6
    _guarantees(surf, targets, C, R, 0.0025, gap)


@pytest.mark.parametrize("k", [8, 16, 24])
def test_max_spheres_caps_the_count_and_reports_what_is_left(surf, targets, k):
    from benchmark.ag3s.experiments.sources.gripper_cover import _gap, finger_body_spheres

    C, R, rep = finger_body_spheres(surf, max_spheres=k)
    assert rep["n_spheres"] <= k and rep["max_spheres"] == k
    g = _gap(targets, C, R)
    assert rep["n_uncovered"] == int((g > 1e-12).sum()) > 0
    assert rep["max_gap_mm"] == pytest.approx(1e3 * g.max())


# =============================================================================================
# 5. 모델 배선
# =============================================================================================


def test_model_with_ilp_overflow_4_has_the_bare_sphere_count_and_full_cover(scene, bare, surf, targets):
    """ilp · overflow 4 mm: 손가락마다 19 구 (greedy 28) — 전체 122 구 (축선 사슬 모델과 같은 수)."""
    from benchmark.ag3s.robot_models.urdf_sphere_chain import FINGER_BODY_ROLE

    m = _build(scene, finger_cover={"overflow": 0.004, "solver": "ilp"})
    rep = m.finger_cover_report
    assert rep["options"] == {"hands": "both", "overflow": 0.004, "solver": "ilp"}
    assert m.n_spheres == bare.n_spheres == 122
    ref = m.local_spheres("ee_finger_l1")
    for link in FINGERS:
        assert rep["links"][link]["source"] == "preset" and rep["links"][link]["n_spheres"] == 19
        c, r = m.local_spheres(link)
        assert np.array_equal(c, ref[0]) and np.array_equal(r, ref[1])
    _guarantees(surf, targets, np.asarray(ref[0]), np.asarray(ref[1]), 0.004, 0.0)
    for palm in ("ee_left", "ee_right"):
        assert np.array_equal(m.local_spheres(palm)[0], bare.local_spheres(palm)[0])
        assert np.array_equal(m.local_spheres(palm)[1], bare.local_spheres(palm)[1])
    roles = dict(zip(m.sphere_link_names, m.sphere_roles))
    assert all(roles[link] == FINGER_BODY_ROLE for link in FINGERS)
    gaps = {o["parent_link"]: o["inner_gap_m"] for o in m.gripper_openings()}
    for palm in ("ee_left", "ee_right"):
        assert gaps[palm] == pytest.approx(0.106 - 2 * 0.0033 - 2 * 0.004, abs=2e-4)


def test_model_passes_gap_and_max_spheres_through(scene):
    m = _build(scene, finger_cover={"hands": "left", "gap": 0.001, "max_spheres": 24})
    rep = m.finger_cover_report
    assert rep["options"] == {"hands": "left", "overflow": 0.0025, "gap": 0.001, "max_spheres": 24}
    l1 = rep["links"]["ee_finger_l1"]
    assert l1["n_spheres"] <= 24 and l1["gap_mm"] == 1.0 and l1["max_spheres"] == 24
    assert set(rep["links"]) == {"ee_finger_l1", "ee_finger_l2"}
