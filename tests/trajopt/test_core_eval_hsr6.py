"""AG3S-CORE-EVAL-HSR6 — the 6-phase HSR (T64 probe) in serving behind four default-off flags.

Task: `AG3S/ag3s/docs/handoff/AG3S-CORE-EVAL-HSR6.task.md` §1–3. Report: `AG3S-CORE-EVAL-HSR6.impl.md`.

What this file fixes (H1 — each rule's true and false case, and "off = before"):

0. the probe: shipped asset = T64 `probe_6.npz` (md5), classes read from the npz, `SubtaskProbe.proba` = the offline
   `t50_core.LogReg.proba`; `AttentionPolicy` puts `result["hsr6"]` only with the probe.
1. ① attach assist (`GraspLatch.observe_grasp(attach_assist=)`, `SafePolicy._advance_grasp`).
2. ② held gate (`hsr6.held_allowance`, `AG3S._frame_held_attached` lift withdrawn, `SafePolicy._hsr6_held_gate`).
3. ③ target exemption (`hsr6.strip_target_permission` only in retreat with nothing held).
4. ④ objective gate (NBC · A4 · the G1 slot only in grasp).
5. serve_safe flags: parse, refusals, record meta only when on; `summary_json.hsr6` only when on.
"""
from __future__ import annotations

import dataclasses
import hashlib
import shlex
import types

import numpy as np
import pytest

from AG3S.trajopt import hsr6 as H6
from AG3S.trajopt.grasp_latch import GraspPhase
from tests.trajopt.test_grasp_evidence import _fb, _finger_origin, _grasp_frames, _latched, _lock, _manipulated, \
    _policy, _signal

ALL = H6.Hsr6Flags(attach_assist=True, held_gate=True, target_exempt=True, objective=True)


def _rec(phase):
    """A `result["hsr6"]` record whose argmax is `phase`."""
    p = {c: (0.9 if c == phase else 0.02) for c in H6.PHASES6}
    return {"p": p, "argmax": phase, "probe": H6.PROBE_NAME}


# ------------------------------------------------------------------------------------------------ 0. probe
def test_shipped_probe_is_the_t64_probe_and_classes_come_from_the_npz():
    path = H6.default_probe_path()
    assert hashlib.md5(path.read_bytes()).hexdigest() == "f0bb92aaf6869276c37786d6d2351917"
    probe = H6.load_hsr6_probe()
    assert probe.classes == H6.PHASES6 and probe.dim == 512
    with np.load(path) as z:
        assert tuple(str(c) for c in z["classes"]) == H6.PHASES6          # SubtaskProbe.load reads `classes`


def test_serving_proba_equals_the_offline_logreg():
    from AG3S.ag3s.experiments.studies.t50_core import LogReg

    probe = H6.load_hsr6_probe()
    with np.load(H6.default_probe_path()) as z:
        lr = LogReg(mean=z["mean"], scale=z["scale"], coef=z["coef"], intercept=z["intercept"], C=float(z["C"]),
                    n_iter=0, converged=True)
    X = np.random.default_rng(64).normal(size=(200, 512)).astype(np.float32) * 0.5 + probe.mean.astype(np.float32)
    assert np.array_equal(probe.proba(X), lr.proba(X))
    for x in X[:20]:
        r = probe.predict(x)
        assert r["argmax"] == H6.PHASES6[int(np.argmax(lr.proba(x[None])[0]))] == H6.phase_of(r)


def test_wrong_class_order_is_refused(tmp_path):
    with np.load(H6.default_probe_path()) as z:
        d = {k: z[k] for k in z.files}
    d["classes"] = np.array(list(reversed(H6.PHASES6)))
    np.savez(tmp_path / "bad.npz", **d)
    with pytest.raises(ValueError, match="classes"):
        H6.load_hsr6_probe(tmp_path / "bad.npz")


def test_phase_of_handles_missing_and_bad_records():
    assert H6.phase_of(None) is None and H6.phase_of({}) is None
    assert H6.phase_of({"p": {c: float("nan") for c in H6.PHASES6}}) is None
    assert H6.phase_of({"argmax": "transport"}) == "transport"
    assert H6.phase_of({"argmax": "pick"}) is None
    assert H6.phase_of(_rec("retreat")) == "retreat"


def _attention_policy(**kw):
    from AG3S.trajopt.attention_policy import AttentionPolicy

    class _P:
        def infer(self, obs, **_):
            return {"actions": np.zeros((4, 16))}

    ap = AttentionPolicy(_P(), None, sampler=object(), **kw)
    feature = np.random.default_rng(1).normal(size=512).astype(np.float32)
    ap._attention_and_feature = lambda obs, **_: ({"cam": np.zeros((16, 16))}, feature if ap._want_feature() else None)
    return ap, feature


def test_attention_policy_adds_hsr6_only_with_the_probe():
    off, f = _attention_policy()
    r0 = off.infer({})
    assert "hsr6" not in r0 and "subtask" in r0 and off.hsr6_probe is None
    on, f2 = _attention_policy(hsr6_probe=H6.load_hsr6_probe())
    r1 = on.infer({})
    assert r1["subtask"] == r0["subtask"]                                   # the 3-class HSR is untouched
    assert set(r1["hsr6"]["p"]) == set(H6.PHASES6) and r1["hsr6"]["probe"] == H6.PROBE_NAME
    # the 3-class probe off, HSR6 on: the feature is still computed for HSR6
    only, _ = _attention_policy(subtask_probe=None, hsr6_probe=H6.load_hsr6_probe())
    r2 = only.infer({})
    assert "subtask" not in r2 and r2["hsr6"] == r1["hsr6"]


# ------------------------------------------------------------------------------------------------ 1. ① attach assist
def _closing_not_settled(latch, assist=None):
    """open → closing (0.90, not closed yet) → 0.715 (not settled: Δ 0.185 ≥ 0.02, but closed and blocked) → 0.715."""
    kw = {} if assist is None else {"attach_assist": assist}
    out = []
    for seq, applied, meas in ((1, [1.0] * 8, 1.0), (2, [0.0] * 8, 0.90), (3, [0.0] * 8, 0.715), (4, [0.0] * 8, 0.715)):
        out.append(latch.observe_grasp(_signal(seq, applied, meas), reach_m=0.05, manipulated_id=0,
                                       manipulated_state="visible", **kw))
    return out


def test_assist_attaches_one_request_early_in_a_held_phase():
    strong = [e.attach for e in _closing_not_settled(_latched())]
    assert strong == [False, False, False, True]
    for phase in ("lift", "transport", "placement"):
        evs = _closing_not_settled(_latched(), {"phase": phase})
        assert [e.attach for e in evs] == [False, False, True, False]
        a = evs[2].evidence["attach_assist"]
        assert a["assisted"] and a["phase"] == phase and a["strong_failed"] == ["settled"]
        assert evs[2].note.startswith("파지 확인 (HSR6 보완)")


@pytest.mark.parametrize("phase", ["approach", "grasp", "retreat", None])
def test_assist_does_nothing_outside_the_held_phases(phase):
    evs = _closing_not_settled(_latched(), {"phase": phase})
    assert [e.attach for e in evs] == [False, False, False, True]
    assert evs[2].evidence["attach_assist"]["assisted"] is False
    # the strong attach on request 4 is not an assist (nothing failed)
    assert evs[3].evidence["attach_assist"] == {"phase": phase, "assisted": False, "strong_failed": []}


def test_assist_needs_closed_and_blocked():
    """Empty-hand closing: the fingers go to the command (0.05 → not blocked) — no assist even in transport."""
    latch = _latched()
    for seq, applied, meas in ((1, [1.0] * 8, 1.0), (2, [0.0] * 8, 0.03), (3, [0.0] * 8, 0.03)):
        e = latch.observe_grasp(_signal(seq, applied, meas), reach_m=0.05, manipulated_id=0,
                                manipulated_state="visible", attach_assist={"phase": "transport"})
    assert not e.attach and e.evidence["checks"]["blocked"] is False
    assert H6.attach_assist_ok("lift", {"closed": True, "blocked": True})
    assert not H6.attach_assist_ok("lift", {"closed": True, "blocked": False})
    assert not H6.attach_assist_ok("lift", {"closed": False, "blocked": True})
    assert not H6.attach_assist_ok("grasp", {"closed": True, "blocked": True})


def test_assist_off_leaves_the_events_as_before():
    a = [dataclasses.asdict(e) for e in _closing_not_settled(_latched())]
    b = [dataclasses.asdict(e) for e in _closing_not_settled(_latched(), None)]
    assert a == b and all("attach_assist" not in e["evidence"] for e in a)


def _fbs():
    return [_fb(1, [1.0] * 8, 1.0), _fb(2, [0.0] * 8, 0.90), _fb(3, [0.0] * 8, 0.715), _fb(4, [0.0] * 8, 0.715)]


def test_assist_while_the_fingers_still_close_meets_the_t34_revoke():
    """Known consequence of the fixed rule (no `settled` in ①): closed ∧ blocked already at 0.80 while the fingers
    still move — the assist attaches there and the T34 hold check revokes it when the opening drops 0.085 on the next
    request (attach → revoke → strong attach one request later). Recorded, not changed (task §2 ① = T64 Part C)."""
    latch = _latched()
    evs = []
    for seq, applied, meas in ((1, [1.0] * 8, 1.0), (2, [0.0] * 8, 0.80), (3, [0.0] * 8, 0.715), (4, [0.0] * 8, 0.715),
                               (5, [0.0] * 8, 0.715)):
        evs.append(latch.observe_grasp(_signal(seq, applied, meas), reach_m=0.05, manipulated_id=0,
                                       manipulated_state="visible", attach_assist={"phase": "lift"}))
    assert [e.attach for e in evs][:2] == [False, True] and evs[1].evidence["attach_assist"]["assisted"]
    assert evs[2].detach and evs[2].revoked


@pytest.mark.parametrize("phase,attach_at", [("transport", 2), ("approach", 3)])
def test_safe_policy_assist_and_its_h4_record(phase, attach_at):
    sp = _policy(hsr6=H6.Hsr6Flags(attach_assist=True))
    q = np.zeros(len(sp.constraint_robot_model.joint_names))
    sp.ag3s._manipulated = _manipulated(_finger_origin(sp, q) + np.array([0.05, 0.0, 0.0]))
    _lock(sp)
    recs = []
    obs = [types.SimpleNamespace(timestamp=1.0, robot_state=q)]
    for fb in _fbs():
        sp._pending = {"scene": {}, "exec_feedback": fb}
        sp._hsr6_start(_rec(phase))
        sp._advance_grasp(obs, ["left"])
        recs.append((dict(sp._last_grasp_record), dict(sp._last_hsr6)))
    attached = [r["attach"] for r, _ in recs]
    assert attached.index(True) == attach_at
    changed = [h["changed"]["attach_assist"] for _, h in recs]
    assert changed == [i == 2 and phase == "transport" for i in range(4)]
    if phase == "transport":
        assert recs[2][1]["detail"]["attach_assist"] == {"assisted_attach": True, "rejected_by_ag3s": None}


def test_safe_policy_off_gives_no_kwarg_no_key():
    sp = _policy()
    assert sp.hsr6 is None and _policy(hsr6=H6.Hsr6Flags()).hsr6 is None
    q = np.zeros(len(sp.constraint_robot_model.joint_names))
    sp.ag3s._manipulated = _manipulated(_finger_origin(sp, q) + np.array([0.05, 0.0, 0.0]))
    _lock(sp)
    recs = list(_grasp_frames(sp, q, _fbs()))
    assert [r["attach"] for r in recs] == [False, False, False, True]
    assert all("attach_assist" not in r["evidence"] for r in recs)
    assert "hsr6" not in sp._exec_summary() and "hsr6" not in sp.metadata


# ------------------------------------------------------------------------------------------------ 2. ② held gate
@pytest.mark.parametrize("phase,held,blocked,want", [
    ("lift", True, None, True), ("transport", False, True, True), ("placement", True, False, True),
    ("transport", False, False, False),                       # no physical evidence
    ("grasp", True, True, False), ("approach", True, True, False), ("retreat", True, True, False),
    (None, True, True, False)])
def test_held_allowance_rule(phase, held, blocked, want):
    assert H6.held_allowance(phase, latch_held=held, blocked=bool(blocked)) is want


def test_safe_policy_tells_ag3s_the_allowance():
    sp = _policy(hsr6=H6.Hsr6Flags(held_gate=True))
    got = []
    sp.ag3s.set_held_allowance = got.append
    for phase, latch_phase, blocked, want in (("transport", GraspPhase.HELD, None, True),
                                              ("grasp", GraspPhase.HELD, None, False),
                                              ("lift", GraspPhase.CLOSING, True, True),
                                              ("lift", GraspPhase.CLOSING, False, False)):
        sp._pending = {}
        sp._hsr6_start(_rec(phase))
        sp._latch.phase = latch_phase
        sp._last_grasp_record = {"evidence": {"checks": {} if blocked is None else {"blocked": blocked}}}
        sp._hsr6_held_gate()
        assert got[-1] is want
        assert sp._last_hsr6["detail"]["held_gate"]["allowed"] is want


def _ag3s_with_held_body(gap_m):
    """An AG3S with a held body sphere `gap_m` above the table plane z = 0.7 at q = 0 (body cover state)."""
    from AG3S.ag3s.constraints.attached import _link_pose_numeric
    from tests.trajopt.test_safe_policy import _safe_policy

    ag3s = _safe_policy().ag3s
    model = ag3s.constraint_robot_model
    q = np.zeros(len(model.joint_names))
    link = "link_left_arm_5"
    T = _link_pose_numeric(model, q, link)
    r = 0.03
    world = T[:3, 3].copy()
    world[2] = 0.7 + r + gap_m
    local = np.linalg.inv(T) @ np.r_[world, 1.0]
    body = [(local[:3], r)]
    lifted = [(local[:3] + np.linalg.inv(T)[:3, :3] @ np.array([0, 0, 0.01]), r)]

    @dataclasses.dataclass
    class _Held:
        primitives: list

    ag3s._attached = _Held(primitives=[])
    ag3s._held_query = {"body": body, "lifted": lifted, "band": 0.008, "plane": (np.array([0, 0, 1.0]), 0.7),
                        "parent_link": link, "lift_dir": None}
    ag3s._held_record = {"body_cover": {}}
    return ag3s, q


@pytest.mark.parametrize("allowance,want_lifted,keys", [(None, True, False), (True, True, True), (False, False, True)])
def test_frame_held_attached_withdraws_the_support_lift(allowance, want_lifted, keys):
    ag3s, q = _ag3s_with_held_body(0.002)
    if allowance is not None:
        ag3s.set_held_allowance(allowance)
    held = ag3s._frame_held_attached(q, [], None)
    c = held.primitives[0].center                                   # parent-frame centre (like the query)
    assert np.allclose(c, ag3s._held_query["lifted" if want_lifted else "body"][0][0])
    frame = ag3s._held_record["body_cover"]["frame"]
    assert ("held_allowance" in frame) is keys and ("hsr6_lift_withdrawn" in frame) is keys
    if keys:
        assert frame["hsr6_lift_withdrawn"] is (allowance is False)
        assert frame["near_support"] is want_lifted


def test_frame_held_attached_far_from_the_support_is_not_a_withdrawal():
    ag3s, q = _ag3s_with_held_body(0.2)
    ag3s.set_held_allowance(False)
    ag3s._frame_held_attached(q, [], None)
    frame = ag3s._held_record["body_cover"]["frame"]
    assert frame["near_support"] is False and frame["hsr6_lift_withdrawn"] is False


def test_reset_clears_the_allowance():
    ag3s, _ = _ag3s_with_held_body(0.002)
    ag3s.set_held_allowance(False)
    ag3s.reset()
    assert ag3s._held_allowance is None


# ------------------------------------------------------------------------------------------------ 3. ③ target exemption
def _snapshot(**kw):
    from AG3S.trajopt.linearize import SceneSnapshot

    S = 4
    base = dict(candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0), candidate_active=np.zeros(0, bool),
                d_safe=np.zeros((S, 0)), plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0),
                plane_active=np.zeros(0, bool), robot_radii=np.full(S, 0.01), candidate_ids=np.zeros(0, np.int64))
    base.update(kw)
    return SceneSnapshot(**base)


def test_withdraw_only_in_retreat():
    assert H6.withdraw_target_exemption("retreat")
    for p in ("approach", "grasp", "lift", "transport", "placement", None):
        assert not H6.withdraw_target_exemption(p)


def test_strip_removes_every_target_permission():
    s = _snapshot(target_free_mask=np.array([1, 1, 0, 0], bool), manipulated_link_margin=np.full(4, 0.01),
                  manipulated_points=np.zeros((5, 3)), target_contact_rows=np.array([1, 0, 0, 0], bool),
                  target_permission_strict=True)
    out, rec = H6.strip_target_permission(s)
    assert rec["changed"] and rec["stripped"] == ["manipulated_link_margin", "manipulated_points",
                                                  "target_contact_rows", "target_free_mask"]
    for f in ("target_free_mask", "target_free_exempt", "manipulated_link_margin", "manipulated_points",
              "manipulated_spheres", "manipulated_sphere_radii", "target_contact_rows", "target_volume"):
        assert getattr(out, f) is None
    assert out.target_permission_strict is True                  # strict stays: no row may touch the target
    from AG3S.trajopt.linearize import CollisionLinearizer
    assert not CollisionLinearizer.target_contact(out, 4).any()


def test_strip_with_no_permission_returns_the_same_snapshot():
    s = _snapshot()
    out, rec = H6.strip_target_permission(s)
    assert out is s and rec == {"changed": False, "stripped": []}


# ------------------------------------------------------------------------------------------------ 4. ④ objective gate
def test_objective_only_in_grasp():
    assert H6.objective_allowed("grasp")
    for p in ("approach", "lift", "transport", "placement", "retreat", None):
        assert not H6.objective_allowed(p)


@pytest.mark.parametrize("phase,keep", [("grasp", True), ("approach", False)])
def test_g1_slot_is_withdrawn_outside_grasp(phase, keep):
    """`_geo_request` with G1 producing a goal (and the A4 anchor already gated by the same rule): grasp keeps the
    G1 goal; approach leaves the slot empty, no B2 scale / continuity-off, controller counters as before."""
    from AG3S.trajopt.config import TrajOptConfig
    from AG3S.trajopt.grasp_anchor import GraspAnchor
    from tests.trajopt.test_safe_policy import DOF, H, _safe_policy

    cfg = TrajOptConfig().with_overrides({"cost": {"geo_grasp": True, "geo_goal_mode": "correct",
                                                   "geo_track_scale": 0.25, "geo_continuity_off": True}})
    sp = _safe_policy(to_config=cfg, hsr6=H6.Hsr6Flags(objective=True))
    geo = sp._geo
    g1 = GraspAnchor(link="ee_left", goal=np.array([0.5, 0.1, 0.9]), close_row=4, w_tcp=0.0, w_goal=2000.0,
                     goal_R=np.eye(3), w_rot=5.0, goal_kind="correct")

    def fake_anchor(q_now, chunk):
        geo.last["anchor"] = {"active": True}
        geo.started, geo._anchor_active, geo.n_goal_chunks = True, True, geo.n_goal_chunks + 1
        geo._correct_goal = (g1.goal, np.eye(3))
        return g1

    geo.step = lambda **kw: {}
    geo.anchor = fake_anchor
    geo._hand = "left"
    sp._pending = {"chunk": np.zeros((H, DOF)), "timing": {}}
    sp._hsr6_start(_rec(phase))
    sp._last_grasp_record = {}
    sp._last_grasp_anchor = {}
    ctx: dict = {}
    sp._geo_request(np.zeros(sp.constraint_robot_model.nq), ["left"], ctx, None, {"filtered_cloud": np.zeros((0, 3))})
    if keep:
        assert ctx["grasp_anchor"] is g1 and ctx.get("continuity_off") and ctx.get("joint_track_scale") is not None
        assert sp._last_hsr6["changed"]["objective"] is False
    else:
        assert "grasp_anchor" not in ctx and "continuity_off" not in ctx and "joint_track_scale" not in ctx
        assert geo.started is False and geo.n_goal_chunks == 0 and geo._correct_goal is None
        assert geo.last["anchor"]["hsr6_withdrawn"] is True and geo.last["anchor"]["active"] is False
        assert sp._last_hsr6["changed"]["objective"] is True
        assert sp._last_hsr6["detail"]["objective"] == {"g1_withdrawn": True}


@pytest.mark.parametrize("phase,active", [("grasp", True), ("lift", False), (None, False)])
def test_nbc_is_withdrawn_outside_grasp(phase, active):
    from AG3S.ag3s.constraints.attached import _link_pose_numeric
    from AG3S.trajopt.config import TrajOptConfig

    cfg = TrajOptConfig().with_overrides({"collision": {"neighbour_contact_depth": 0.02}})
    sp = _policy(to_config=cfg, hsr6=H6.Hsr6Flags(objective=True))
    q = np.zeros(len(sp.constraint_robot_model.joint_names))
    palm = _link_pose_numeric(sp.constraint_robot_model, q, sp.hold_links["left"])[:3, 3]
    sp.ag3s._manipulated = _manipulated(palm + np.array([0.03, 0.0, 0.0]))
    sp._latch.phase = GraspPhase.LATCHED
    sp._pending = {}
    sp._hsr6_start(_rec(phase) if phase else None)
    sp._last_grasp_record = {"state": "latched"}
    # the arm-only test model has no finger spheres: let the gate's link prefix name the arm links instead
    sp.grasp_links = {**sp.grasp_links, "left": ("link_left_arm_5", sp.grasp_links["left"][1])}
    out = sp._neighbour_contact_gate(q, ["left"])
    assert (out is not None) is active
    assert sp._last_hsr6["changed"]["objective"] is (not active)


def test_nbc_gate_runs_before_the_finger_check_in_order():
    """The HSR6 gate is the last condition: with the existing rule off (palm far) nothing is marked as changed."""
    from AG3S.trajopt.config import TrajOptConfig

    cfg = TrajOptConfig().with_overrides({"collision": {"neighbour_contact_depth": 0.02}})
    sp = _policy(to_config=cfg, hsr6=H6.Hsr6Flags(objective=True))
    q = np.zeros(len(sp.constraint_robot_model.joint_names))
    sp.ag3s._manipulated = _manipulated(np.array([5.0, 5.0, 5.0]))
    sp._latch.phase = GraspPhase.LATCHED
    sp._pending = {}
    sp._hsr6_start(_rec("approach"))
    sp._last_grasp_record = {"state": "latched"}
    assert sp._neighbour_contact_gate(q, ["left"]) is None
    assert sp._last_hsr6["changed"]["objective"] is False


# ------------------------------------------------------------------------------------------------ 5. serve_safe
BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]
HSR6_ALL = ["--hsr6-attach-assist", "--hsr6-held-gate", "--hsr6-target-exempt", "--hsr6-objective"]


def _parse(argv):
    from AG3S.trajopt import serve_safe

    ap = serve_safe.build_parser()
    a = ap.parse_args(BASE_ARGV + list(argv))
    serve_safe.reject_hsr6_flags(ap, a)
    return a


def test_serve_flags_off_add_nothing():
    from AG3S.trajopt import serve_safe
    from tests.trajopt.test_core_eval_g1_merge import A4, A8, G1

    a = _parse(shlex.split(A8) + A4 + G1)
    assert serve_safe.hsr6_flags_from_args(a) is None and serve_safe.hsr6_record_meta(a) == {}


def test_serve_all_four_with_a8_g1():
    from AG3S.trajopt import serve_safe
    from tests.trajopt.test_core_eval_g1_merge import A4, A8, G1

    a = _parse(shlex.split(A8) + A4 + G1 + HSR6_ALL)
    assert serve_safe.hsr6_flags_from_args(a) == ALL
    meta = serve_safe.hsr6_record_meta(a)["hsr6"]
    assert meta["flags"] == ["attach_assist", "held_gate", "target_exempt", "objective"]
    assert meta["probe"] == "shipped:hsr6_t64_kv_L4"


@pytest.mark.parametrize("argv", [
    ["--hsr6-held-gate"],                                         # needs --held-body-cover
    ["--hsr6-objective"],                                         # nothing to withhold
    ["--hsr6-probe", "/x.npz"],                                   # probe without a use
    ["--hsr6-attach-assist", "--no-attention"],                   # no feature
    ["--hsr6-attach-assist", "--no-safe"]])
def test_serve_refuses_flags_that_would_do_nothing(argv):
    with pytest.raises(SystemExit):
        _parse(argv)


def test_summary_and_metadata_only_when_on():
    sp = _policy(hsr6=ALL)
    sp._pending = {}
    sp._hsr6_start(_rec("grasp"))
    s = sp._exec_summary()["hsr6"]
    assert s["phase"] == "grasp" and s["flags"] == ALL.names()
    assert s["changed"] == {"attach_assist": False, "held_gate": False, "target_exempt": False, "objective": False}
    assert sp.metadata["hsr6"] == {"flags": ALL.names()}
    sp._hsr6_start(None)                                          # no label this request
    assert sp._exec_summary()["hsr6"]["phase"] is None and sp._hsr6_phase() is None
    assert sp._hsr6_objective_ok() is False                       # ④ fails closed without a label


@pytest.mark.parametrize("phase,keep", [("grasp", True), ("transport", False), ("approach", False)])
def test_a4_anchor_is_withdrawn_outside_grasp(phase, keep):
    sp = _policy(hsr6=H6.Hsr6Flags(objective=True))
    sp._pending = {}
    sp._hsr6_start(_rec(phase))
    sp._last_grasp_anchor = {"active": True}
    a4 = object()
    got = sp._hsr6_a4_gate(a4)
    assert (got is a4) is keep
    assert sp._last_grasp_anchor.get("hsr6_withdrawn", False) is (not keep)
    assert sp._last_hsr6["changed"]["objective"] is (not keep)


@pytest.mark.parametrize("phase,held,strip", [("retreat", False, True), ("retreat", True, False),
                                              ("placement", False, False), ("approach", False, False)])
def test_target_gate_strips_only_in_retreat_with_nothing_held(phase, held, strip):
    sp = _policy(hsr6=H6.Hsr6Flags(target_exempt=True))
    sp._pending = {}
    sp._hsr6_start(_rec(phase))
    snap = _snapshot(target_free_mask=np.array([1, 0, 0, 0], bool), manipulated_points=np.zeros((3, 3)))
    cs = types.SimpleNamespace(attached=object() if held else None)
    out = sp._hsr6_target_gate(snap, cs)
    assert (out is not snap) is strip
    assert (out.target_free_mask is None) is strip
    assert sp._last_hsr6["changed"]["target_exempt"] is strip
