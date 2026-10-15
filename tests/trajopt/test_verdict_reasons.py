"""T23 — 판정을 사유별로 나눈다: `verdict_reasons` 와 위반 행 분류.

T16 에서 `--no-collision` 인데 34 청크 중 25 개가 HOLD 였다. `TrajOptStatus.safe` 하나가 파지에 필요한
접촉(손가락이 사과를 파고든다) · 가려진 target · 진짜 충돌을 모두 `violated` 로 뭉쳤기 때문이다.
이 파일이 고정하는 것:

1. 와이어 — 사유 kind 와 기본 처리, 모르는 kind 는 HOLD, 옛 서버에서는 `safe` 로 유도.
2. 행 분류 (`classify_violations`) — 권한 link ↔ 조작 대상만 `allowed_contact`. 같은 청크의 다른
   위반(팔꿈치 ↔ 테이블)은 가려지지 않는다. 손가락이 테이블에 더 가까우면 사과 옆이어도 충돌이다.
   이웃 물체(orange) 는 사과가 아니다. target 을 뺀 계층·미해결 계층은 허용 접촉의 근거가 못 된다.
3. `SafePolicy._verdict` — 사유 → `safe`. `occluded` 는 미인증이 아니다 (T20). `legacy` 로 되돌린다.

MuJoCo 도 GPU 도 없다. 행 분류는 실제 RB-Y1 구 모델(`fixtures.make_scenario`) 위에서 해석적 가짜
거리장으로 돈다 — 분류가 읽는 것(`distance` · `target_free_distance` · `label` · `answer_tier`)만 흉내낸다.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.trajopt import wire
from AG3S.trajopt.safe_policy import SafePolicy, classify_violations
from AG3S.trajopt.types import TrajOptStatus
from tests.trajopt.fixtures import empty_scene, make_scenario


# =============================================================================================
# 1. 와이어
# =============================================================================================


def test_every_kind_has_a_default_action_and_the_split_is_as_documented():
    assert set(wire.GATE_DEFAULT) == set(wire.REASON_KINDS)
    execute = {k for k, v in wire.GATE_DEFAULT.items() if v == "execute"}
    # SUBTASK-e: `subtask_no_target` — gate 가 일부러 비운 target (인증된 기하, 기록용 사유).
    # T43 Z6: `follow_to` — margin-only, non-worsening collision rows; the TO chunk runs (flag only).
    assert execute == {"allowed_contact", "budget_only", "occluded_target", "uncertified_waived",
                       "subtask_no_target", "follow_to"}
    # T27: `no_perception` (`serve_safe --no-perception`) 이 HOLD 쪽에 더해졌다 — 검사되지 않은
    # 청크이고, 실행하려면 로컬이 `--safe-gate off` 로 명시해야 한다.
    # T57-A: `task_not_ready` (`serve_safe --task-gate`, default off) — HELD but the noun probe does not
    # say the instructed object is held.
    assert {k for k, v in wire.GATE_DEFAULT.items() if v == "hold"} == {
        "collision", "uncertified", "unverified", "comms", "no_perception", "task_not_ready"}


def test_make_reason_refuses_an_unregistered_kind():
    with pytest.raises(ValueError, match="reason kind"):
        wire.make_reason("contact_allowed", "typo")
    r = wire.make_reason("allowed_contact", "finger ↔ apple", link="ee_finger_l1")
    assert r == {"kind": "allowed_contact", "action": "execute", "detail": "finger ↔ apple",
                 "evidence": {"link": "ee_finger_l1"}}


def test_one_hold_reason_holds_the_chunk():
    ok = wire.make_reason("allowed_contact")
    bad = wire.make_reason("collision")
    assert wire.gate_decision([]) == ("execute", [])
    assert wire.gate_decision([ok, wire.make_reason("budget_only")])[0] == "execute"
    action, holding = wire.gate_decision([ok, bad])
    assert action == "hold" and [r["kind"] for r in holding] == ["collision"]


def test_an_unknown_kind_or_a_malformed_entry_holds():
    """서버가 새 사유를 더했는데 로컬이 모르면 그것을 실행 쪽으로 읽지 않는다 (fail closed)."""
    assert wire.gate_decision([{"kind": "from_the_future", "detail": ""}])[0] == "hold"
    assert wire.gate_decision([{"detail": "no kind"}])[0] == "hold"
    assert wire.gate_decision(["not a dict"])[0] == "hold"


def test_a_local_table_can_be_stricter():
    strict = dict(wire.GATE_DEFAULT, allowed_contact="hold")
    assert wire.gate_decision([wire.make_reason("allowed_contact")], strict)[0] == "hold"


def test_an_old_server_gets_reasons_derived_from_safe():
    assert wire.unpack_verdict_reasons({"safe": True}) == ([], "derived")
    reasons, source = wire.unpack_verdict_reasons(
        {"safe": False, "geometry_certified": False, "ag3s_status": "degraded"})
    assert source == "derived" and [r["kind"] for r in reasons] == ["uncertified"]
    reasons, _ = wire.unpack_verdict_reasons(
        {"safe": False, "geometry_certified": True, "trajopt_status": "violated"})
    assert [r["kind"] for r in reasons] == ["collision"]
    # 유도한 사유는 언제나 옛 `safe` 와 같은 결정을 낸다.
    for resp in ({"safe": True}, {"safe": False, "geometry_certified": True},
                 {"safe": False, "geometry_certified": False}):
        action, _ = wire.gate_decision(wire.unpack_verdict_reasons(resp)[0])
        assert (action == "execute") is resp["safe"]


def test_a_malformed_reasons_key_is_read_as_hold():
    reasons, source = wire.unpack_verdict_reasons({"safe": True, wire.VERDICT_REASONS: "oops"})
    assert source == "server" and wire.gate_decision(reasons)[0] == "hold"


def test_the_key_rides_only_when_the_server_has_reasons():
    """`None` 은 키 없음 (씬 없는 청크 · 옛 서버), `[]` 은 "깨끗하다" 이고 키가 있다."""
    base = dict(ag3s_status="ok", geometry_certified=True, trajopt_status="optimal",
                max_violation_m=0.0, safe=True)
    assert wire.VERDICT_REASONS not in wire.SafetyVerdict(**base).to_dict()
    assert wire.SafetyVerdict(**base, verdict_reasons=[]).to_dict()[wire.VERDICT_REASONS] == []
    reason = wire.make_reason("allowed_contact", "x", row={"link": "l", "clearance_m": -0.006})
    out = wire.pack_response(np.zeros((4, wire.ACTION_WIDTH)),
                             wire.SafetyVerdict(**base, verdict_reasons=[reason]),
                             seq=1, timing_ms={})
    assert wire.unpack_verdict_reasons(out) == ([reason], "server")


# =============================================================================================
# 2. 행 분류 — 실제 구 모델, 해석적 가짜 거리장
# =============================================================================================

APPLE_R = 0.04
VOXEL = 0.005
TARGET = "target"


class _Field:
    """구 여러 개로 된 해석적 거리장. `CuroboEsdfField` 중 분류가 읽는 것만."""

    def __init__(self, objects, *, tier="fine"):
        #: `[(label, centre[3], radius)]`
        self.objects = [(str(n), np.asarray(c, np.float64), float(r)) for n, c, r in objects]
        self.label_names = tuple(n for n, _, _ in self.objects)
        self.grid = types.SimpleNamespace(voxel_size=VOXEL)
        self.tier = tier
        self.has_labels = True

    def _each(self, pts, skip=()):
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        cols = [np.linalg.norm(pts - c, axis=1) - r for n, c, r in self.objects if n not in skip]
        return np.stack(cols, axis=1) if cols else np.full((len(pts), 1), 10.0)

    def distance(self, pts):
        return self._each(pts).min(axis=1)

    def target_free_distance(self, pts):
        return self._each(pts, skip=(TARGET,)).min(axis=1)

    def label(self, pts):
        return self._each(pts).argmin(axis=1)

    def answer_tier(self, pts, *, target_free=False):
        n = len(np.asarray(pts).reshape(-1, 3))
        flags = np.broadcast_to(np.asarray(target_free, bool), (n,))
        out = np.full(n, self.tier, dtype="<U11")
        if self.tier == "fine":
            out[flags] = "target_free"
        return out


@pytest.fixture(scope="module")
def rig():
    """손끝 역할의 구 = `link_right_arm_5` 의 가장 낮은 구, step 3. 권한 link 는 그 link 하나."""
    s = make_scenario(horizon=6)
    lin = s.linearizer
    centres = lin.sphere_states(s.reference, s.q_now)[0]
    names = [lin._query_name(q) for q in range(lin.n_spheres)]
    step = 3
    tips = [q for q in range(lin.n_spheres) if names[q] == "link_right_arm_5"]
    tip = min(tips, key=lambda q: centres[step, q, 2])
    nxt = [q for q in tips if q != tip]
    axis = centres[step, tip] - centres[step, min(nxt, key=lambda q: centres[step, q, 2])]
    axis /= np.linalg.norm(axis)
    torso = names.index("link_torso_0")
    return types.SimpleNamespace(s=s, lin=lin, centres=centres, names=names, step=step,
                                 tip=tip, axis=axis, torso=torso,
                                 authorized=frozenset({"link_right_arm_5"}))


def _surface(centre, radius, n=3000):
    """조작 대상의 관측 점군 — 구 표면의 고른 점 (fibonacci)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.asarray(centre) + radius * np.column_stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])


def _apple_at_tip(rig, penetration=0.006, direction=None):
    d = rig.axis if direction is None else np.asarray(direction, np.float64)
    d = d / np.linalg.norm(d)
    r = float(rig.lin.query_radii[rig.tip])
    return rig.centres[rig.step, rig.tip] + d * (r + APPLE_R - penetration)


def _scene(rig, field, *, apple_centre=None, margin=0.0, target_free=False,
           manipulated=True, points=True):
    base = empty_scene(rig.lin)
    fields = {f: getattr(base, f) for f in base.__dataclass_fields__}
    per_link = np.asarray([0.0 if n in rig.authorized else margin for n in rig.names])
    fields.update(
        esdf=field, esdf_margin=margin,
        manipulated_points=(_surface(apple_centre, APPLE_R)
                            if manipulated and points and apple_centre is not None else None),
        manipulated_link_margin=per_link if manipulated else None,
        target_free_mask=(np.asarray([n in rig.authorized for n in rig.names])
                          if target_free else None))
    return type(base)(**fields)


def _classify(rig, scene, **kw):
    kw.setdefault("authorized_links", rig.authorized)
    kw.setdefault("tolerance", 1e-4)
    return classify_violations(rig.lin, rig.s.reference, rig.s.q_now, scene, **kw)


def test_a_fingertip_in_the_apple_is_allowed_contact_and_nothing_else(rig):
    apple = _apple_at_tip(rig)
    field = _Field([(TARGET, apple, APPLE_R)])
    scene = _scene(rig, field, apple_centre=apple)
    # 사전 조건: 그 청크는 정말 위반한다 (최적화기가 보는 값 그대로).
    assert rig.lin.full_violation(rig.s.reference, rig.s.q_now, scene) < -1e-4

    cls = _classify(rig, scene, manipulated={"id": 0, "state": "visible"})

    assert cls["n_violating"] >= 1
    assert cls["n_allowed"] == cls["n_violating"]
    assert cls["n_collision"] == 0 and cls["n_unresolved"] == 0
    row = cls["allowed_worst"]
    assert row["link"] == "link_right_arm_5" and row["step"] == rig.step
    assert row["clearance_m"] == pytest.approx(-0.006, abs=1e-3)
    assert row["evidence"] == "distance" and row["tier"] == "fine"
    assert cls["residual_violation_m"] == 0.0
    assert cls["manipulated_id"] == 0


def test_the_worst_row_being_allowed_does_not_hide_an_elbow_on_the_table(rig):
    """최악 행(손끝 −12 mm)이 허용이어도 팔꿈치(몸통) −5 mm 가 충돌로 남는다."""
    apple = _apple_at_tip(rig, penetration=0.012)
    r_torso = float(rig.lin.query_radii[rig.torso])
    table = rig.centres[rig.step, rig.torso] + np.array([0, 0, -1.0]) * (r_torso + 0.5 - 0.005)
    field = _Field([(TARGET, apple, APPLE_R), ("table", table, 0.5)])
    scene = _scene(rig, field, apple_centre=apple)
    worst, pair = rig.lin.worst_row(rig.s.reference, rig.s.q_now, scene)
    assert pair["link"] == "link_right_arm_5", "사전 조건: 최악 행은 손끝이다"

    cls = _classify(rig, scene)

    assert cls["n_allowed"] >= 1 and cls["n_collision"] >= 1
    assert cls["collision_worst"]["link"] == "link_torso_0"
    assert cls["collision_worst"]["why"] == "unauthorized_link"
    assert cls["residual_violation_m"] == pytest.approx(0.005, abs=1e-3)


def test_a_fingertip_nearer_the_table_than_the_apple_is_a_collision(rig):
    """지침 §10 시나리오: 손가락-사과 · 손가락-테이블 동시 근접 → 사과만 허용, 테이블은 충돌."""
    r = float(rig.lin.query_radii[rig.tip])
    tip = rig.centres[rig.step, rig.tip]
    down = np.array([0.0, 0.0, -1.0])
    side = np.cross(rig.axis, down)
    side /= np.linalg.norm(side)
    table = tip + down * (r + 0.5 - 0.004)                 # 테이블을 4 mm 파고든다
    apple = tip + side * (r + APPLE_R + 0.010)             # 사과는 10 mm 떨어져 있다
    field = _Field([(TARGET, apple, APPLE_R), ("table", table, 0.5)])
    scene = _scene(rig, field, apple_centre=apple)

    cls = _classify(rig, scene, keep_rows=True)

    # 다른 step 에서는 손끝이 사과를 파고들 수 있다 (그 행은 허용이 맞다). 판정할 것은 이 행이다.
    row = next(r for r in cls["rows"] if r["step"] == rig.step and r["query"] == rig.tip)
    assert row["class"] == "collision"
    assert row["why"] == "nearest_is_not_manipulated"
    assert row["label"] == "table"
    assert row["clearance_m"] == pytest.approx(-0.004, abs=1e-3)
    assert cls["n_collision"] >= 1


def test_a_neighbouring_fruit_is_not_the_manipulated_object(rig):
    """T14 의 orange (사과에서 105 mm) — 손끝이 orange 를 파고들면 충돌이다."""
    orange = _apple_at_tip(rig)
    apple = orange + np.cross(rig.axis, [0.0, 0.0, 1.0]) * 0.105
    field = _Field([(TARGET, apple, APPLE_R), ("orange", orange, APPLE_R)])
    scene = _scene(rig, field, apple_centre=apple)

    cls = _classify(rig, scene)

    assert cls["n_allowed"] == 0 and cls["n_collision"] >= 1
    assert cls["collision_worst"]["why"] == "nearest_is_not_manipulated"


def test_an_answer_from_the_target_free_layer_is_not_contact_with_the_target(rig):
    """target 을 뺀 계층이 답한 위반은 정의상 target 이 아닌 무엇과의 위반이다."""
    r = float(rig.lin.query_radii[rig.tip])
    tip = rig.centres[rig.step, rig.tip]
    crate = tip + rig.axis * (r + 0.2 - 0.006)
    apple = tip - rig.axis * 0.3
    field = _Field([(TARGET, apple, APPLE_R), ("crate", crate, 0.2)])
    scene = _scene(rig, field, apple_centre=apple, target_free=True)

    cls = _classify(rig, scene, keep_rows=True)

    assert cls["n_allowed"] == 0
    tips = [r for r in cls["rows"] if r["link"] == "link_right_arm_5"]
    assert tips and all(r["why"] == "answer_excludes_target" for r in tips)
    assert all(r["tier"] == "target_free" for r in tips)


@pytest.mark.parametrize("tier", ["coarse", "window_edge", "outside"])
def test_an_unresolved_tier_is_not_evidence_of_allowed_contact(rig, tier):
    """T21 `RESOLVED_TIERS`: 미세 창 밖 답은 접촉 허용의 근거가 못 된다 → `unresolved`."""
    apple = _apple_at_tip(rig)
    field = _Field([(TARGET, apple, APPLE_R)], tier=tier)
    scene = _scene(rig, field, apple_centre=apple)

    cls = _classify(rig, scene)
    assert cls["n_allowed"] == 0 and cls["n_collision"] == 0
    assert cls["n_unresolved"] >= 1
    assert cls["unresolved_worst"]["why"] == f"unresolved_tier:{tier}"

    lenient = _classify(rig, scene, allow_unresolved_tier=True)
    assert lenient["n_allowed"] == cls["n_unresolved"] and lenient["n_unresolved"] == 0


def test_without_a_manipulated_object_nothing_is_allowed(rig):
    """T20 `lost` — 권한(`manipulated_link_margin`)이 없으면 기록에 centroid 가 남아도 허용하지 않는다."""
    apple = _apple_at_tip(rig)
    field = _Field([(TARGET, apple, APPLE_R)])
    scene = _scene(rig, field, apple_centre=apple, manipulated=False)

    cls = _classify(rig, scene, manipulated={"id": 0, "state": "lost",
                                             "centroid": list(apple), "radius": APPLE_R})

    assert cls["n_allowed"] == 0
    assert cls["collision_worst"]["why"] == "no_manipulated"


def test_an_unauthorized_link_in_the_apple_is_a_collision(rig):
    apple = _apple_at_tip(rig)
    field = _Field([(TARGET, apple, APPLE_R)])
    scene = _scene(rig, field, apple_centre=apple)

    cls = _classify(rig, scene, authorized_links={"link_left_arm_5"})

    assert cls["n_allowed"] == 0
    assert cls["collision_worst"]["why"] == "unauthorized_link"


def test_without_object_points_the_label_then_the_centroid_decide(rig):
    apple = _apple_at_tip(rig)
    field = _Field([(TARGET, apple, APPLE_R)])
    scene = _scene(rig, field, apple_centre=apple, points=False)

    by_label = _classify(rig, scene)
    assert by_label["n_allowed"] >= 1
    assert by_label["allowed_worst"]["evidence"] == "label"

    field.has_labels = False
    record = {"id": 0, "state": "occluded", "centroid": list(apple), "radius": APPLE_R}
    by_centroid = _classify(rig, scene, manipulated=record)
    assert by_centroid["n_allowed"] >= 1
    assert by_centroid["allowed_worst"]["evidence"] == "centroid"
    # 공이 다른 곳이면 허용하지 않는다.
    far = dict(record, centroid=list(apple + np.array([0.3, 0.0, 0.0])))
    assert _classify(rig, scene, manipulated=far)["n_allowed"] == 0


# =============================================================================================
# 3. SafePolicy._verdict — 사유 → safe
# =============================================================================================


def _policy(**kw):
    from tests.trajopt.test_safe_policy import _safe_policy

    return _safe_policy(**kw)


class _Status:
    def __init__(self, value):
        self.value = value

    @property
    def safe(self):
        return self.value in ("optimal", "feasible")


def _result(status="violated", violation=0.0, trajectory=None, **metrics):
    base = {"max_violation_pair": None, "time_budget_hit": False, "max_iterations_hit": False,
            "sqp_iterations": 2}
    base.update(metrics)
    return types.SimpleNamespace(status=_Status(status), max_violation=float(violation),
                                 trajectory=trajectory, metrics=base, notes=[])


def _cs(status="ok", *, validity_ok=True, manipulated_state="visible", manipulated=True,
        grounding="ok"):
    return types.SimpleNamespace(
        status=types.SimpleNamespace(value=status),
        grounding_status=types.SimpleNamespace(value=grounding),
        validity=types.SimpleNamespace(value="valid" if validity_ok else "degraded"),
        geometry_certified=validity_ok,
        manipulated=object() if manipulated else None,
        metrics={"manipulated": None if manipulated_state is None else
                 {"id": 7, "state": manipulated_state, "age_frames": 3}},
        contact_context=None, candidates=[])


def _wire_up(sp, rig, scene, result, cs):
    sp.refiner.last_result = result
    sp.refiner.optimizer.linearizer = rig.lin
    sp._last_constraint_set = cs
    sp._last_snapshot = scene
    sp._last_q_now = rig.s.q_now
    sp._authorized_links = lambda _cs: rig.authorized


def test_fingertip_contact_executes_while_the_optimizer_still_says_violated(rig):
    apple = _apple_at_tip(rig)
    scene = _scene(rig, _Field([(TARGET, apple, APPLE_R)]), apple_centre=apple)
    sp = _policy()
    _wire_up(sp, rig, scene, _result("violated", 0.006, rig.s.reference), _cs())

    v = sp._verdict(None)

    assert [r["kind"] for r in v.verdict_reasons] == ["allowed_contact"]
    assert v.safe is True
    assert v.trajopt_status == "violated", "최적화기의 상태는 바꾸지 않는다"
    assert any(n.startswith("verdict: allowed_contact") for n in v.notes)
    out = v.to_dict()
    assert out["safe"] is True and out[wire.VERDICT_REASONS][0]["kind"] == "allowed_contact"


def test_legacy_verdict_policy_restores_the_old_hold(rig):
    apple = _apple_at_tip(rig)
    scene = _scene(rig, _Field([(TARGET, apple, APPLE_R)]), apple_centre=apple)
    sp = _policy(verdict_policy="legacy")
    _wire_up(sp, rig, scene, _result("violated", 0.006, rig.s.reference), _cs())

    v = sp._verdict(None)

    assert v.safe is False
    assert [r["kind"] for r in v.verdict_reasons] == ["allowed_contact"], "사유는 똑같이 싣는다"


def test_verdict_policy_is_validated():
    with pytest.raises(ValueError, match="verdict_policy"):
        _policy(verdict_policy="lenient")


def test_a_real_collision_holds(rig):
    apple = _apple_at_tip(rig, penetration=0.012)
    r_torso = float(rig.lin.query_radii[rig.torso])
    table = rig.centres[rig.step, rig.torso] + np.array([0, 0, -1.0]) * (r_torso + 0.5 - 0.005)
    scene = _scene(rig, _Field([(TARGET, apple, APPLE_R), ("table", table, 0.5)]),
                   apple_centre=apple)
    sp = _policy()
    _wire_up(sp, rig, scene, _result("violated", 0.012, rig.s.reference), _cs())

    v = sp._verdict(None)

    assert {r["kind"] for r in v.verdict_reasons} == {"allowed_contact", "collision"}
    assert v.safe is False
    collision = next(r for r in v.verdict_reasons if r["kind"] == "collision")
    assert collision["evidence"]["row"]["link"] == "link_torso_0"
    assert sp._exec_summary()["verdict"]["classification"]["n_collision"] >= 1


def test_occluded_manipulated_with_no_target_executes(rig):
    """T14 seq 22 — `no_seed` 인데 조작 대상은 `occluded` 로 살아 있다 → `occluded_target`."""
    sp = _policy()
    _wire_up(sp, rig, None, _result("violated", 0.0, rig.s.reference),
             _cs("no_target", manipulated_state="occluded", grounding="no_seed"))

    v = sp._verdict(None)

    assert [r["kind"] for r in v.verdict_reasons] == ["occluded_target"]
    assert v.safe is True and v.geometry_certified is False
    assert v.verdict_reasons[0]["evidence"]["grounding_status"] == "no_seed"
    assert v.verdict_reasons[0]["evidence"]["manipulated_id"] == 7


@pytest.mark.parametrize("cs", [
    _cs("no_target", manipulated_state="lost", manipulated=False, grounding="lost"),
    _cs("no_target", manipulated_state=None, manipulated=False, grounding="no_seed"),
    _cs("degraded", validity_ok=False, manipulated_state="occluded"),
    _cs("no_target", validity_ok=False, manipulated_state="occluded"),
], ids=["lost", "nothing", "degraded", "no_target_but_invalid"])
def test_other_uncertified_frames_hold(rig, cs):
    sp = _policy()
    _wire_up(sp, rig, None, _result("violated", 0.0, rig.s.reference), cs)

    v = sp._verdict(None)

    assert [r["kind"] for r in v.verdict_reasons] == ["uncertified"]
    assert v.safe is False


def test_a_waived_certification_follows_the_one_switch(rig):
    """기하 인증 요구 스위치는 `to_config.safety.require_certified_geometry` 하나다."""
    from AG3S.trajopt.config import TrajOptConfig

    cfg = TrajOptConfig.from_dict({
        "collision": {"backend": "esdf", "esdf_margin": 0.05, "use_support_planes": False},
        "safety": {"require_certified_geometry": False}})
    sp = _policy(to_config=cfg)
    _wire_up(sp, rig, None, _result("optimal", 0.0, rig.s.reference),
             _cs("degraded", validity_ok=False))

    v = sp._verdict(None)

    assert [r["kind"] for r in v.verdict_reasons] == ["uncertified_waived"]
    assert v.safe is True


def test_a_budget_ended_feasible_solve_executes_and_says_so(rig):
    sp = _policy()
    _wire_up(sp, rig, None, _result("feasible", 0.0, rig.s.reference, time_budget_hit=True), _cs())

    v = sp._verdict(None)

    assert [r["kind"] for r in v.verdict_reasons] == ["budget_only"]
    assert v.safe is True
    assert v.verdict_reasons[0]["evidence"]["time_budget_hit"] is True


def test_a_clean_optimal_solve_has_no_reasons_but_carries_the_key(rig):
    sp = _policy()
    _wire_up(sp, rig, None, _result("optimal", 0.0, rig.s.reference), _cs())

    v = sp._verdict(None)

    assert v.verdict_reasons == [] and v.safe is True
    assert v.to_dict()[wire.VERDICT_REASONS] == []


@pytest.mark.parametrize("status", ["solver_failed", "unconstrained"])
def test_an_unchecked_chunk_is_unverified(rig, status):
    sp = _policy()
    _wire_up(sp, rig, None, _result(status, 0.0, rig.s.reference), _cs())

    v = sp._verdict(None)

    assert [r["kind"] for r in v.verdict_reasons] == ["unverified"]
    assert v.safe is False


def test_a_violation_without_a_snapshot_is_a_collision_not_a_pass(rig):
    sp = _policy()
    _wire_up(sp, rig, None, _result("violated", 0.02, rig.s.reference), _cs())

    v = sp._verdict(None)

    assert [r["kind"] for r in v.verdict_reasons] == ["collision"]
    assert v.safe is False


def test_no_scene_keeps_the_pre_t23_response_shape():
    """씬이 없어 최적화기가 안 돈 청크는 키가 없다 — 로컬이 `uncertified` 를 유도한다."""
    sp = _policy()
    sp.refiner.last_result = None
    sp._last_constraint_set = None

    v = sp._verdict(None)

    assert v.verdict_reasons is None and v.safe is False
    assert wire.VERDICT_REASONS not in v.to_dict()
    reasons, source = wire.unpack_verdict_reasons(v.to_dict())
    assert source == "derived" and [r["kind"] for r in reasons] == ["uncertified"]
    # 서버 기록에는 남는다.
    assert sp._last_reasons[0]["kind"] == "uncertified"


def test_trajopt_status_safe_is_still_what_legacy_reads():
    """`legacy` 게이트가 읽는 상태 집합이 `TrajOptStatus.safe` 와 같다."""
    assert {s.value for s in TrajOptStatus if s.safe} == {"optimal", "feasible"}
