"""T41 a — target 이 아닌 장애물에만 여유거리 (`collision.obstacle_margin`, `--obstacle-margin`).

T40 I 진단: 손가락 질의 구가 mesh 를 다 덮지 못하고 (꼭짓점 4.3 %) esdf_margin 0 이라 막대에 닿았다.
사용자 지시는 "질의 구는 최소로" — 그래서 구를 키우지 않고 `d − r ≥ m` 의 `m` 을 **target 이 아닌
장애물 행에만** 올린다.

이 파일이 고정하는 것:

1. **기본 off 는 한 값도 안 바뀐다** — 새 필드가 있어도 `obstacle_margin = 0` 이면 같은 배열.
   (기록 641 청크 대 HEAD 의 비트 동일은 `T41a.impl.md` 의 replay 가 쟀다; 여기서는 기록 두 청크.)
2. 켜면 `obstacle` 행만 `max(그 행의 여유, m)` — target · 지지면 · 쥔 물체 행은 그대로.
3. 테이블 위 낮은 장애물은 장애물이다 (지지면 판정은 거리 일치, 높이가 아니다).
4. 지지면 정보가 없으면 테이블도 장애물로 본다 (못 가르면 여유를 붙이는 쪽).
5. `scene_from_constraint_set` · CLI · config 배선.
"""

from __future__ import annotations

import json
import pathlib
import types

import numpy as np
import pytest

from benchmark.trajopt.config import TrajOptConfig
from benchmark.trajopt.linearize import (
    OBSTACLE_CLASSES,
    CollisionLinearizer,
    SceneSnapshot,
    scene_from_constraint_set,
    support_planes_of,
)

VOXEL = 0.005
M = 0.010                 # T41 시험값
APPROACH = 0.02           # 권한 link 의 조작 대상 마진 (0.05 × 0.4)
APPLE_R = 0.033
BAR_R = 0.012
BOX_H = 0.030
R_FINGER = 0.0033


class _Table:
    """z = 0 테이블 + 사과(구, target) + 높은 막대(구로 근사) + 낮은 상자. 해석적 거리."""

    has_labels = False

    def __init__(self):
        self.apple = np.array([0.0, 0.0, APPLE_R])
        self.bar = np.array([0.3, 0.0, 0.28])
        self.box_c = np.array([-0.3, 0.0, BOX_H / 2])
        self.box_h = np.array([0.05, 0.05, BOX_H / 2])
        self.grid = types.SimpleNamespace(voxel_size=VOXEL)

    def _others(self, p):
        bar = np.linalg.norm(p - self.bar, axis=1) - BAR_R
        q = np.maximum(np.abs(p - self.box_c) - self.box_h, 0.0)
        return np.minimum.reduce([p[:, 2], bar, np.linalg.norm(q, axis=1)])

    def distance(self, pts):
        p = np.asarray(pts, np.float64).reshape(-1, 3)
        return np.minimum(self._others(p), np.linalg.norm(p - self.apple, axis=1) - APPLE_R)

    def target_free_distance(self, pts):
        return self._others(np.asarray(pts, np.float64).reshape(-1, 3))


def _surface(centre, radius, n):
    rng = np.random.default_rng(0)
    v = rng.normal(size=(n, 3))
    return centre + radius * v / np.linalg.norm(v, axis=1, keepdims=True)


#: 질의점 (step 1 개). 0–3 은 로봇(손가락) 구, 4 는 쥔 물체 구.
ROWS = {
    "bar": 0,      # 막대 표면에서 8 mm (테이블 위 260 mm) → obstacle
    "apple": 1,    # 사과 옆 6 mm, 테이블 33 mm → target
    "table": 2,    # 테이블 위 4 mm, 멀리 아무것도 없음 → support
    "box": 3,      # 낮은 상자 윗면 위 10 mm (테이블 40 mm) → obstacle
    "held": 4,     # 쥔 물체 구, 막대 옆 8 mm → held
}


def _rig():
    field = _Table()
    centres = np.stack([
        field.bar + np.array([0.0, 0.0, -(BAR_R + 0.008)]),
        field.apple + np.array([APPLE_R + 0.006, 0.0, 0.0]),
        np.array([0.0, 0.25, 0.004]),
        np.array([-0.3, 0.0, BOX_H + 0.010]),
        field.bar + np.array([0.0, BAR_R + 0.008, 0.0]),
    ])[None]
    # 관측 점군: 사과 표면 + 사과 옆 질의점의 정확한 최근접 표면점 (KD-tree 거리가 정확하도록).
    nearest = field.apple + APPLE_R * np.array([1.0, 0.0, 0.0])
    points = np.vstack([_surface(field.apple, APPLE_R, 4000), nearest])
    lin = object.__new__(CollisionLinearizer)
    lin.robot_radii = np.full(4, R_FINGER)
    lin.query_radii = np.array([R_FINGER] * 4 + [0.03])
    lin.n_spheres = 4
    lin._attached_local = None
    return lin, centres, field, points


TABLE_PLANE = np.array([[0.0, 0.0, 1.0, 0.0]])


def _scene(field, points, *, margin=0.0, mask=None, esdf_margin=0.0, **extra):
    return SceneSnapshot(
        candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0),
        candidate_active=np.zeros(0, bool), d_safe=np.zeros((4, 0)),
        plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0), plane_active=np.zeros(0, bool),
        robot_radii=np.full(4, R_FINGER), candidate_ids=np.zeros(0, np.int64),
        esdf=field, esdf_margin=float(esdf_margin), manipulated_points=points,
        manipulated_link_margin=np.full(4, APPROACH),
        target_free_mask=None if mask is None else np.asarray(mask, bool),
        obstacle_margin=float(margin), **extra)


def _clear(lin, centres, scene):
    return lin._esdf_clearance(centres, scene)[0, :, 0]


# =============================================================================================
# 1. 기본 off
# =============================================================================================


@pytest.mark.parametrize("mask", [None, [True] * 4], ids=["relax", "exclude_authorized"])
def test_default_off_is_bit_identical(mask):
    lin, centres, field, points = _rig()
    plain = _clear(lin, centres, _scene(field, points, mask=mask))
    # 지지면 · 스위치가 실려 와도 여유가 0 이면 아무것도 읽지 않는다.
    loaded = _clear(lin, centres, _scene(field, points, mask=mask, support_planes=TABLE_PLANE,
                                         obstacle_margin_support=True))
    assert np.array_equal(plain, loaded)
    assert SceneSnapshot.__dataclass_fields__["obstacle_margin"].default == 0.0
    assert TrajOptConfig().collision.obstacle_margin == 0.0
    assert TrajOptConfig().collision.obstacle_margin_support is False


# =============================================================================================
# 2. 켜면 obstacle 행만
# =============================================================================================


@pytest.mark.parametrize("mask", [None, [True] * 4], ids=["relax", "exclude_authorized"])
def test_only_non_target_non_support_robot_rows_get_the_margin(mask):
    lin, centres, field, points = _rig()
    off = _clear(lin, centres, _scene(field, points, mask=mask, support_planes=TABLE_PLANE))
    on = _clear(lin, centres, _scene(field, points, margin=M, mask=mask,
                                     support_planes=TABLE_PLANE))
    delta = off - on
    assert delta[ROWS["bar"]] == pytest.approx(M, abs=1e-12)
    assert delta[ROWS["box"]] == pytest.approx(M, abs=1e-12)
    for name in ("apple", "table", "held"):
        assert delta[ROWS[name]] == 0.0, name
    # 값 자체 — 막대 8 mm − r − 10 mm.
    assert on[ROWS["bar"]] == pytest.approx(0.008 - R_FINGER - M, abs=1e-9)


def test_the_classes_are_what_the_rows_are_near():
    lin, centres, field, points = _rig()
    lin._esdf_clearance(centres, _scene(field, points, margin=M, support_planes=TABLE_PLANE))
    got = [OBSTACLE_CLASSES[k] for k in lin._last_obstacle_class[0]]
    assert got == ["obstacle", "target", "support", "obstacle", "held"]


def test_the_target_keeps_its_own_phase_margin_not_the_obstacle_margin():
    """`relax`: 사과 옆 행은 E1 규칙 그대로 APPROACH 마진 (20 mm) — 10 mm 로 바뀌지 않는다."""
    lin, centres, field, points = _rig()
    on = _clear(lin, centres, _scene(field, points, margin=M, support_planes=TABLE_PLANE))
    assert on[ROWS["apple"]] == pytest.approx(0.006 - R_FINGER - APPROACH, abs=1e-6)


def test_the_margin_is_a_floor_on_esdf_margin_not_a_sum():
    lin, centres, field, points = _rig()
    for esdf_margin, expect in ((0.004, M - 0.004), (0.02, 0.0)):
        off = _clear(lin, centres, _scene(field, points, esdf_margin=esdf_margin,
                                          support_planes=TABLE_PLANE))
        on = _clear(lin, centres, _scene(field, points, margin=M, esdf_margin=esdf_margin,
                                         support_planes=TABLE_PLANE))
        assert off[ROWS["bar"]] - on[ROWS["bar"]] == pytest.approx(expect, abs=1e-12)


def test_support_switch_puts_the_table_in_but_never_the_target_or_held():
    lin, centres, field, points = _rig()
    off = _clear(lin, centres, _scene(field, points, support_planes=TABLE_PLANE))
    on = _clear(lin, centres, _scene(field, points, margin=M, support_planes=TABLE_PLANE,
                                     obstacle_margin_support=True))
    delta = off - on
    assert delta[ROWS["table"]] == pytest.approx(M, abs=1e-12)
    assert delta[ROWS["apple"]] == 0.0 and delta[ROWS["held"]] == 0.0


def test_without_support_planes_the_table_is_an_obstacle():
    """지지면을 모르면 가를 수 없다 — 여유를 붙이는 쪽으로 닫힌다."""
    lin, centres, field, points = _rig()
    off = _clear(lin, centres, _scene(field, points))
    on = _clear(lin, centres, _scene(field, points, margin=M))
    assert off[ROWS["table"]] - on[ROWS["table"]] == pytest.approx(M, abs=1e-12)
    assert off[ROWS["apple"]] == on[ROWS["apple"]]


def test_a_row_beside_the_table_edge_reads_the_obstacle_not_the_plane():
    """평면은 무한하다. 그 행의 `d` 가 평면 거리와 2 voxel 넘게 다르면 지지면이 아니다."""
    lin, centres, field, points = _rig()
    shifted = np.array([[0.0, 0.0, 1.0, -0.05]])         # 평면이 실제 표면보다 50 mm 아래
    on = _clear(lin, centres, _scene(field, points, margin=M, support_planes=shifted))
    off = _clear(lin, centres, _scene(field, points, support_planes=shifted))
    assert off[ROWS["table"]] - on[ROWS["table"]] == pytest.approx(M, abs=1e-12)


def test_worst_row_identity_names_the_class_only_when_on():
    lin, centres, field, points = _rig()
    lin._esdf_label_name = lambda point, scene: None
    lin._query_name = lambda q: f"q{q}"
    lin._authorized_rows = lambda *a, **k: None
    for margin, key_expected in ((0.0, False), (M, True)):
        scene = _scene(field, points, margin=margin, support_planes=TABLE_PLANE)
        esdf = lin._esdf_clearance(centres, scene)
        out = lin._identify("esdf", (0, ROWS["bar"]), centres, scene,
                            float(esdf[0, ROWS["bar"], 0]), esdf_block=esdf)
        assert ("obstacle_margin_class" in out) is key_expected
        if key_expected:
            assert out["obstacle_margin_class"] == "obstacle"


# =============================================================================================
# 3. 배선 — scene_from_constraint_set · config · CLI
# =============================================================================================


def _constraint_set(field, points):
    from benchmark.ag3s.types import SupportSurface

    table = SupportSurface(id=1, normal=np.array([0.0, 0.0, 2.0]), offset=0.82,
                           point_indices=np.arange(3), point_count=3, rms_error=0.001,
                           safety_margin=0.01)
    return types.SimpleNamespace(
        constraints=None, esdf=field, support_surfaces=(table,), attached=None,
        manipulated=types.SimpleNamespace(points=points, sphere_centers=None, sphere_radii=None),
        manipulated_link_margin=np.full(4, APPROACH), target_field_exclude=None,
        target_field_policy="relax", destination_label=None, destination_margin=None)


def _config(**collision):
    return TrajOptConfig.from_dict({"collision": {"backend": "esdf", "esdf_margin": 0.0,
                                                  "use_support_planes": False, **collision}})


def test_scene_from_constraint_set_carries_planes_only_when_on():
    _, _, field, points = _rig()
    cs = _constraint_set(field, points)
    radii = np.full(4, R_FINGER)
    off = scene_from_constraint_set(cs, radii, _config())
    assert off.obstacle_margin == 0.0 and off.support_planes is None
    on = scene_from_constraint_set(cs, radii, _config(obstacle_margin=M))
    assert on.obstacle_margin == M and on.obstacle_margin_support is False
    # 기하 그대로: 단위 법선, offset 에 safety_margin(10 mm) 을 더하지 않는다.
    assert np.allclose(on.support_planes, [[0.0, 0.0, 1.0, 0.82]])
    assert support_planes_of(types.SimpleNamespace(support_surfaces=())).shape == (0, 4)
    both = scene_from_constraint_set(cs, radii, _config(obstacle_margin=M,
                                                        obstacle_margin_support=True))
    assert both.obstacle_margin_support is True


def test_config_rejects_a_negative_margin():
    with pytest.raises(ValueError):
        _config(obstacle_margin=-0.001)


BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def test_cli_default_adds_no_key_and_the_flag_sets_the_config():
    from benchmark.trajopt import serve_safe

    args = serve_safe.build_parser().parse_args(BASE_ARGV)
    assert serve_safe.obstacle_overrides(args) == {}
    assert serve_safe.trajopt_config_from_args(args).collision == \
        TrajOptConfig.from_dict({"collision": {"backend": "esdf", "esdf_margin": 0.05,
                                               "use_support_planes": False}}).collision
    args = serve_safe.build_parser().parse_args(BASE_ARGV + ["--obstacle-margin", "0.01"])
    cfg = serve_safe.trajopt_config_from_args(args).collision
    assert cfg.obstacle_margin == 0.01 and cfg.obstacle_margin_support is False
    args = serve_safe.build_parser().parse_args(
        BASE_ARGV + ["--obstacle-margin", "0.01", "--obstacle-margin-support"])
    assert serve_safe.trajopt_config_from_args(args).collision.obstacle_margin_support is True


@pytest.mark.parametrize("extra", [["--obstacle-margin-support"], ["--obstacle-margin", "-0.01"]])
def test_cli_rejects_meaningless_combinations(extra):
    from benchmark.trajopt import serve_safe

    args = serve_safe.build_parser().parse_args(BASE_ARGV + extra)
    with pytest.raises(SystemExit):
        serve_safe.trajopt_config_from_args(args)


# =============================================================================================
# 4. 기록 replay — T40 G E3b (있을 때만)
# =============================================================================================

_RUN = pathlib.Path("/mnt/dev/work/outputs/verify/T40/G/runs/E3b/server_8240_20261003T135313/"
                    "server_constraints/run_0000")
#: 130 = C ep1807 의 접촉 청크 (t_step 28, 손가락 ↔ 막대). 145 = B ep1968 t_step 15 (막대 없음,
#: 파지 직전 — 테이블이 최근접인 손 행 3.88 mm).
_CHUNKS = {"C1807_contact": 130, "B1968_grasp": 145}
_AUTH = ("ee_left", "ee_finger_l1", "ee_finger_l2")


def _replay(index):
    from benchmark.trajopt.experiments.fixed_input_ablation import build_field

    z = np.load(_RUN / f"chunk_{index:05d}.npz", allow_pickle=True)
    names = np.array(json.loads(str(z["summary_json"]))["sphere_link_names"])
    radii = z["sphere_radii"].astype(np.float64)
    centres = z["sphere_centres"].astype(np.float64)[None]
    cloud = z["filtered_cloud"].astype(np.float64)
    sup = cloud[z["support_mask"]]
    sup = sup[sup[:, 2] > 0.4]                         # 테이블 (바닥 z≈0 은 손에서 멀다)
    mu = sup.mean(0)
    n = np.linalg.svd(sup - mu, full_matrices=False)[2][2]
    n = n * np.sign(n[2])
    planes = np.array([[*n, float(n @ mu)]])
    target = z["target_points"].astype(np.float64)
    auth = np.isin(names, _AUTH)
    field = build_field(z, outside_distance=1.0)
    lin = object.__new__(CollisionLinearizer)
    lin.robot_radii = radii
    lin.query_radii = radii
    lin.n_spheres = radii.size
    lin._attached_local = None

    def scene(margin, **extra):
        return SceneSnapshot(
            candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0),
            candidate_active=np.zeros(0, bool), d_safe=np.zeros((radii.size, 0)),
            plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0),
            plane_active=np.zeros(0, bool), robot_radii=radii,
            candidate_ids=np.zeros(0, np.int64), esdf=field, esdf_margin=0.0,
            manipulated_points=target, manipulated_link_margin=np.where(auth, APPROACH, 0.05),
            target_free_mask=auth if "esdf_free_distance" in z.files else None,
            obstacle_margin=margin, **extra)

    return lin, centres, scene, planes, auth


@pytest.mark.skipif(not _RUN.exists(), reason="T40 G E3b 기록이 없다")
@pytest.mark.parametrize("name", sorted(_CHUNKS))
def test_recorded_chunk_off_is_identical_and_on_touches_only_obstacle_rows(name):
    lin, centres, scene, planes, auth = _replay(_CHUNKS[name])
    plain = lin._esdf_clearance(centres, scene(0.0))[0, :, 0]
    off = lin._esdf_clearance(centres, scene(0.0, support_planes=planes))[0, :, 0]
    on = lin._esdf_clearance(centres, scene(M, support_planes=planes))[0, :, 0]
    cls = lin._last_obstacle_class[0]
    assert np.array_equal(plain, off)
    changed = on != off
    assert np.all(cls[changed] == OBSTACLE_CLASSES.index("obstacle"))
    assert np.allclose(off[changed] - on[changed], M, atol=1e-12)
    assert np.all(on[cls != OBSTACLE_CLASSES.index("obstacle")]
                  == off[cls != OBSTACLE_CLASSES.index("obstacle")])
    s = centres[0] @ planes[0, :3] - planes[0, 3]
    near = auth & (off < M)
    if name == "C1807_contact":
        # 막대 옆 손 행 (테이블 위 > 100 mm) 이 여유를 받는다.
        bar_rows = near & (s > 0.10)
        assert bar_rows.sum() >= 5
        assert np.all(cls[bar_rows] == OBSTACLE_CLASSES.index("obstacle"))
    else:
        # 파지 직전: 테이블이 최근접인 손 행은 그대로다.
        table_rows = near & (cls == OBSTACLE_CLASSES.index("support"))
        assert table_rows.sum() >= 1
        assert np.all(on[table_rows] == off[table_rows])
