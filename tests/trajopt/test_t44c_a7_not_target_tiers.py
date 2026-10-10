"""T44c A7 — `safe_policy._NOT_TARGET_TIERS` 에 `target_free_coarse` (사용자 판정 2026-10-10).

`target_free_coarse` 는 target-free 질의를 target 을 뺀 **거친** 계층 (`coarse_no_target`) 이 답한 것이다.
`target_free` 와 같이 정의상 조작 대상이 아닌 무엇과의 위반이다. A7 전에는 이 tier 가 `_NOT_TARGET_TIERS` 에
없었다. 그래서 그 행이 대상 판정으로 내려갔고, 거친 계층에 target 이 남아 있으면 (A5 의 target 바닥 잔류)
`unresolved_tier:target_free_coarse` 가 됐다. `allow_unresolved_tier` 이면 **허용 접촉**이 됐다.

## 무엇을 박나

| # | 성질 |
|---|---|
| 1 | 상수: `target_free_coarse` ∈ `_NOT_TARGET_TIERS`, `curobo_field.ANSWER_TIERS` 의 `target_free*` 는 전부 포함, 접촉 tier 와 겹치지 않음 |
| 2 | 행 분류: `target_free_coarse` 가 답한 위반 = `target_free` 와 같은 경로 (`collision` · `answer_excludes_target`) — 거친 계층이 target 까지의 거리를 내도, `allow_unresolved_tier` 여도 |
| 3 | `static` 은 그대로 `static_geometry` · 미해결 tier (`coarse`) 는 그대로 `unresolved` |
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.ag3s.fields.curobo_field import ANSWER_TIERS, RESOLVED_TIERS
from benchmark.trajopt import safe_policy as SP
from tests.trajopt.test_verdict_reasons import (  # noqa: F401 — `rig` 는 fixture
    APPLE_R,
    TARGET,
    _apple_at_tip,
    _classify,
    _Field,
    _scene,
    rig,
)


class _TierField(_Field):
    """target-free 행에 `free_tier` 로 답하는 필드. `revive=True` 면 target-free 거리가 target 을 그대로
    본다 — 거친 target-free 계층에 target 바닥이 남은 경우 (A5, `T44c.verify.json`) 의 흉내."""

    def __init__(self, objects, *, free_tier, revive=False):
        super().__init__(objects)
        self.free_tier = free_tier
        self.revive = revive

    def target_free_distance(self, pts):
        return self.distance(pts) if self.revive else super().target_free_distance(pts)

    def answer_tier(self, pts, *, target_free=False):
        n = len(np.asarray(pts).reshape(-1, 3))
        flags = np.broadcast_to(np.asarray(target_free, bool), (n,))
        out = np.full(n, "fine", dtype="<U18")
        out[flags] = self.free_tier
        return out


# ---------------------------------------------------------------------------- 1 상수


def test_target_free_coarse_is_a_not_target_tier():
    assert "target_free_coarse" in SP._NOT_TARGET_TIERS
    assert set(SP._NOT_TARGET_TIERS) == {"target_free", "target_free_coarse", "static"}
    assert SP._TARGET_EXCLUDED_TIERS == ("target_free", "target_free_coarse")
    # 앞으로 target-free tier 가 더 생기면 여기서 걸린다.
    assert {t for t in ANSWER_TIERS if t.startswith("target_free")} <= set(SP._TARGET_EXCLUDED_TIERS)
    assert set(SP._NOT_TARGET_TIERS) <= set(ANSWER_TIERS)
    assert not set(SP._NOT_TARGET_TIERS) & set(SP._CONTACT_TIERS)
    assert "target_free_coarse" not in RESOLVED_TIERS


# ---------------------------------------------------------------------------- 2 행 분류


def _tip_rows(cls, rig):
    return [r for r in cls["rows"] if r["link"] == "link_right_arm_5"]


@pytest.mark.parametrize("allow_unresolved", [False, True])
@pytest.mark.parametrize("free_tier", ["target_free", "target_free_coarse"])
def test_a_target_free_coarse_row_takes_the_target_free_path(rig, free_tier, allow_unresolved):
    """손끝이 사과를 6 mm 파고들고, target-free 계층이 사과까지의 거리를 그대로 낸다 (target 잔류).
    `target_free` 와 `target_free_coarse` 는 같은 행 · 같은 class · 같은 why 를 낸다."""
    apple = _apple_at_tip(rig)
    field = _TierField([(TARGET, apple, APPLE_R)], free_tier=free_tier, revive=True)
    scene = _scene(rig, field, apple_centre=apple, target_free=True)

    cls = _classify(rig, scene, keep_rows=True, allow_unresolved_tier=allow_unresolved)

    tips = _tip_rows(cls, rig)
    assert tips
    assert all(r["tier"] == free_tier for r in tips)
    assert all(r["class"] == "collision" and r["why"] == "answer_excludes_target" for r in tips)
    assert cls["n_allowed"] == 0 and cls["n_unresolved"] == 0
    assert cls["collision_by_why"].get("answer_excludes_target", 0) == len(tips)


def test_target_free_coarse_rows_match_target_free_rows_exactly(rig):
    """같은 장면에서 tier 이름만 바꾼 두 분류: tier 를 뺀 행 · 집계가 같다."""
    apple = _apple_at_tip(rig)
    out = {}
    for t in ("target_free", "target_free_coarse"):
        field = _TierField([(TARGET, apple, APPLE_R)], free_tier=t, revive=True)
        out[t] = _classify(rig, _scene(rig, field, apple_centre=apple, target_free=True),
                           keep_rows=True)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "tier"} for r in rows]  # noqa: E731
    assert strip(out["target_free"]["rows"]) == strip(out["target_free_coarse"]["rows"])
    for k in ("n_violating", "n_allowed", "n_unresolved", "n_collision", "collision_by_why",
              "residual_violation_m"):
        assert out["target_free"][k] == out["target_free_coarse"][k], k


def test_a_crate_answered_by_the_coarse_free_tier_is_answer_excludes_target(rig):
    """target-free 거친 계층이 다른 물체 (crate) 와의 위반을 답해도 같은 사유다 (예전: `nearest_is_not_manipulated`)."""
    r = float(rig.lin.query_radii[rig.tip])
    tip = rig.centres[rig.step, rig.tip]
    crate = tip + rig.axis * (r + 0.2 - 0.006)
    apple = tip - rig.axis * 0.3
    field = _TierField([(TARGET, apple, APPLE_R), ("crate", crate, 0.2)],
                       free_tier="target_free_coarse")
    cls = _classify(rig, _scene(rig, field, apple_centre=apple, target_free=True), keep_rows=True)
    tips = _tip_rows(cls, rig)
    assert tips and all(r["why"] == "answer_excludes_target" for r in tips)


# ---------------------------------------------------------------------------- 3 다른 tier 는 그대로


@pytest.mark.parametrize("tier,cls_,why", [("static", "collision", "static_geometry"),
                                           ("coarse", "unresolved", "unresolved_tier:coarse")])
def test_other_tiers_keep_their_paths(rig, tier, cls_, why):
    apple = _apple_at_tip(rig)
    field = _TierField([(TARGET, apple, APPLE_R)], free_tier=tier, revive=True)
    cls = _classify(rig, _scene(rig, field, apple_centre=apple, target_free=True), keep_rows=True)
    tips = _tip_rows(cls, rig)
    assert tips and all(r["class"] == cls_ and r["why"] == why for r in tips)
