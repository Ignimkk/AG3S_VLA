"""IFG-S2SERVE — the training-free 3D held rule in the grasp latch (`--held-3d-rule on`, default off).

| # | property |
|---|---|
| 1 | constants = `ifg_stage2_fk.D_HOLD` / `GRIP_CLOSED`; boundaries (hand_d = 0.1313 m in, next float out; grip 0.9 out); no target → gripper only; unknown grip → not held; the per-frame rule equals `ifg_stage2_fk.rules`' held column on the stored IFG-S2FK features (storage) |
| 2 | latch: `observe_grasp(extra_checks={"held_3d": False})` blocks the CLOSING → HELD attach that the T17 evidence would make, `True` / None attach as before; once HELD, release is unchanged |
| 3 | SafePolicy: off → `observe_grasp` gets no `extra_checks` and the grasp record has no `held_3d`; on → the rule's value is ANDed and recorded; the target centre is the previous request's grounded target centroid (None → gripper only) |
| 4 | hand proxies (storage): the server FK (request joints + measured finger opening, `RobotRenderMask.pose`) equals the offline `_episode_features` proxies (full sim qpos) within 0.05 mm on held-out frames |
"""

from __future__ import annotations

import json
import os
import pathlib
import types

import numpy as np
import pytest

from AG3S.trajopt.grasp_latch import GraspPhase
from AG3S.trajopt.held_3d import HELD_3D_GRIP_CLOSED, HELD_3D_MAX_M, Held3DRule, held_3d_rule

STORE = pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", "/mnt/dev/AG3S_VLA-storage"))
S2FK = STORE / "results/development/20261019T025853+0900_f15e587290f6_93594d3dffc0/s2fk"


# ============================================================================ 1 rule


def test_constants_are_the_ifg_s2fk_ones():
    from AG3S.ag3s.experiments.studies import ifg_stage2_fk as S2

    assert HELD_3D_MAX_M == S2.D_HOLD == pytest.approx(0.1313)
    assert HELD_3D_GRIP_CLOSED == S2.GRIP_CLOSED == 0.9


def test_rule_boundaries_and_fallbacks():
    assert held_3d_rule(0.5, HELD_3D_MAX_M) == (True, True)
    assert held_3d_rule(0.5, float(np.nextafter(HELD_3D_MAX_M, 1.0))) == (True, False)
    assert held_3d_rule(0.5, 0.130) == (True, True) and held_3d_rule(0.5, 0.132) == (True, False)
    assert held_3d_rule(0.9, 0.01) == (False, False)                        # strict <
    assert held_3d_rule(float(np.nextafter(0.9, 0.0)), 0.01) == (True, True)
    for no_target in (None, float("nan")):
        assert held_3d_rule(0.5, no_target) == (True, True)
        assert held_3d_rule(0.95, no_target) == (False, False)
    assert held_3d_rule(None, 0.01) == (False, False)
    assert held_3d_rule(float("nan"), 0.01) == (False, False)
    assert held_3d_rule(0.5, 0.2, max_m=0.25) == (True, True)


@pytest.mark.skipif(not (S2FK / "features_fk.jsonl").is_file(), reason="IFG-S2FK features not in this storage")
def test_rule_equals_the_offline_held_rule_on_the_stored_features():
    from AG3S.ag3s.experiments.studies import ifg_stage2_fk as S2

    rows = [json.loads(l) for l in open(S2FK / "features_fk.jsonl")]
    rows.sort(key=lambda r: (r["episode_id"], r["frame"]))
    eps = {}
    for r in rows:
        eps.setdefault(r["episode_id"], []).append(r)
    n = 0
    for ep_rows in eps.values():
        held, _ = S2.rules(ep_rows)
        mine = [int(held_3d_rule(r["grip"], r["hand_d"])[1]) for r in ep_rows]
        assert mine == held
        n += len(mine)
    assert n > 3000


class _FK:
    def __init__(self, proxies=None, fail=False):
        self.p = proxies or {"link_left_arm_6": np.array([0.5, 0.2, 0.9]), "link_right_arm_6": np.array([0.5, -0.2, 0.9])}
        self.fail = fail
        self.roots = tuple(self.p)
        self.model_xml = "stub.xml"

    def proxies(self, robot_state, openings=None):
        if self.fail:
            raise RuntimeError("fk")
        return self.p

    def record(self):
        return {}


def test_evaluate_records_hand_d_and_falls_back():
    rule = Held3DRule(_FK())
    st = np.ones(16)
    st[7], st[15] = 0.7, 1.0
    rec = rule.evaluate(state16=st, gripper_columns=(7, 15), robot_state=np.zeros(20), openings={},
                        target_centre=np.array([0.5, 0.2, 0.9 + 0.1]))
    assert rec["grip"] == 0.7 and rec["hand_d"] == pytest.approx(0.1) and rec["held_gripper"] and rec["held_3d"]
    rec = rule.evaluate(state16=st, gripper_columns=(7, 15), robot_state=np.zeros(20), openings={},
                        target_centre=np.array([0.5, 0.0, 0.9]))
    assert rec["hand_d"] == pytest.approx(0.2) and rec["held_gripper"] and not rec["held_3d"]
    rec = rule.evaluate(state16=st, gripper_columns=(7, 15), robot_state=np.zeros(20), openings={},
                        target_centre=None)
    assert rec["hand_d"] is None and rec["held_3d"] and rec["why"].startswith("no_target")
    rec = Held3DRule(_FK(fail=True)).evaluate(state16=st, gripper_columns=(7, 15), robot_state=np.zeros(20),
                                              openings={}, target_centre=np.zeros(3))
    assert not rec["held_3d"] and rec["held_gripper"] and rec["why"].startswith("fk_failed")
    rec = rule.evaluate(state16=None, gripper_columns=(7, 15), robot_state=np.zeros(20), openings={},
                        target_centre=None)
    assert not rec["held_3d"] and rec["why"].startswith("no_grip")


# ============================================================================ 2 latch


def test_latch_ands_the_extra_check_into_the_attach():
    from tests.trajopt.test_grasp_evidence import T17, _latched, _signal

    def run(extra_at_attach):
        latch = _latched()
        out = []
        for i, (t, applied, measured) in enumerate(T17):
            kw = {} if extra_at_attach is None else {"extra_checks": {"held_3d": extra_at_attach}}
            ev = latch.observe_grasp(_signal(i + 1, applied, measured), reach_m=0.055, manipulated_id=0,
                                     manipulated_state="visible", **kw)
            out.append((t, ev))
        return latch, out

    _, base = run(None)
    assert [t for t, ev in base if ev.attach] == [128] and "held_3d" not in dict(base)[128].evidence["checks"]
    _, ok = run(True)
    assert [t for t, ev in ok if ev.attach] == [128] and dict(ok)[128].evidence["checks"]["held_3d"] is True
    latch, blocked = run(False)
    assert not any(ev.attach for _, ev in blocked) and latch.phase is GraspPhase.CLOSING
    ev = dict(blocked)[128]
    assert [k for k, v in ev.evidence["checks"].items() if not v] == ["held_3d"] and "held_3d" in ev.note


def test_extra_check_does_not_touch_release_once_held():
    from tests.trajopt.test_grasp_evidence import T17, _latched, _signal

    latch = _latched()
    for i, (_, applied, measured) in enumerate(T17):
        latch.observe_grasp(_signal(i + 1, applied, measured), reach_m=0.055, manipulated_id=0,
                            manipulated_state="visible")
    assert latch.phase is GraspPhase.HELD
    k = len(T17)
    evs = [latch.observe_grasp(_signal(k + j + 1, [1.0] * 8, 1.0), reach_m=0.055, manipulated_id=0,
                               manipulated_state="visible", extra_checks={"held_3d": False}) for j in range(2)]
    assert evs[-1].detach and latch.phase is GraspPhase.PLACED


# ============================================================================ 3 SafePolicy


class _StubRule:
    def __init__(self, value):
        self.value = value
        self.calls = []
        self.max_m, self.grip_closed = HELD_3D_MAX_M, HELD_3D_GRIP_CLOSED

    def evaluate(self, **kw):
        self.calls.append(kw)
        return {"grip": 0.7, "hand_d": 0.05, "held_gripper": True, "held_3d": self.value}


def _run_grasp(sp, monkeypatch):
    from tests.trajopt.test_grasp_evidence import _fb, _finger_origin, _grasp_frames, _lock, _manipulated

    q = np.zeros(len(sp.constraint_robot_model.joint_names))
    sp.ag3s._manipulated = _manipulated(_finger_origin(sp, q) + np.array([0.05, 0.0, 0.0]))  # noqa: SLF001
    _lock(sp)
    calls = []
    orig = sp._latch.observe_grasp                                              # noqa: SLF001

    def spy(*a, **kw):
        calls.append(kw.get("extra_checks", "absent"))
        return orig(*a, **kw)

    monkeypatch.setattr(sp._latch, "observe_grasp", spy)                         # noqa: SLF001
    recs = list(_grasp_frames(sp, q, [_fb(1, [1.0] * 8, 1.0), _fb(2, [0.0] * 8, 0.715), _fb(3, [0.0] * 8, 0.715)]))
    return calls, recs


def test_safe_policy_off_passes_no_extra_check(monkeypatch):
    from tests.trajopt.test_grasp_evidence import _policy

    sp = _policy()
    assert sp.held_3d is None and "held_3d_rule" not in sp.metadata
    calls, recs = _run_grasp(sp, monkeypatch)
    assert calls == ["absent"] * 3 and sp.ag3s.attached is not None
    assert all("held_3d" not in r for r in recs)


@pytest.mark.parametrize("value", [True, False])
def test_safe_policy_on_ands_and_records(monkeypatch, value):
    from tests.trajopt.test_grasp_evidence import _policy

    rule = _StubRule(value)
    sp = _policy(held_3d=rule)
    assert sp.metadata["held_3d_rule"]["max_m"] == HELD_3D_MAX_M
    calls, recs = _run_grasp(sp, monkeypatch)
    assert calls == [{"held_3d": value}] * 3
    assert (sp.ag3s.attached is not None) is value
    assert recs[-1]["held_3d"]["held_3d"] is value and recs[-1]["held_3d"]["hand_d"] == 0.05
    assert rule.calls[-1]["target_centre"] is None and tuple(rule.calls[-1]["gripper_columns"]) == (7, 15)


def test_target_centre_is_the_previous_requests_grounded_target():
    from tests.trajopt.test_grasp_evidence import _policy

    sp = _policy(held_3d=_StubRule(True))
    assert sp._held_3d_target_centre() is None                                  # noqa: SLF001
    sp._last_constraint_set = types.SimpleNamespace(target=None)                 # noqa: SLF001
    assert sp._held_3d_target_centre() is None                                  # noqa: SLF001
    sp._last_constraint_set = types.SimpleNamespace(                             # noqa: SLF001
        target=types.SimpleNamespace(centroid=[0.1, 0.2, 0.3]))
    assert np.array_equal(sp._held_3d_target_centre(), [0.1, 0.2, 0.3])         # noqa: SLF001


def test_serve_safe_builds_the_rule_only_when_on(monkeypatch):
    from AG3S.trajopt import held_3d, serve_safe

    monkeypatch.setattr(held_3d, "HandProxyFK", lambda xml: types.SimpleNamespace(roots=("a", "b"), model_xml=xml))
    ap = serve_safe.build_parser()
    args = ap.parse_args(["--config", "c", "--checkpoint", "k", "--model-xml", "/x.xml", "--held-3d-rule", "on",
                          "--held-3d-max-m", "0.2"])
    serve_safe.reject_ifg_s2serve_flags(ap, args)
    rule = serve_safe.held_3d_from_args(args)
    assert rule.max_m == 0.2 and rule.fk.model_xml == "/x.xml"
    assert serve_safe.ifg_s2serve_record_meta(args)["held_3d_rule"]["max_m"] == 0.2


# ============================================================================ 4 hand proxies vs offline


@pytest.mark.skipif(not (S2FK / "features_fk.jsonl").is_file(), reason="IFG-S2FK run not in this storage")
def test_server_hand_proxies_equal_the_offline_ones():
    import mujoco

    from AG3S.ag3s.experiments.studies import ifg_heldout_eval as HE
    from AG3S.ag3s.experiments.studies import ifg_stage2_fk as S2
    from AG3S.ag3s.experiments.studies.t47_replay import GRIPPER_ROOT_LINKS
    from AG3S.ag3s.robot_models import DEFAULT_RBY1_JOINTS
    from AG3S.trajopt.held_3d import HandProxyFK

    if not S2.SNAP_XML.is_file():
        pytest.skip("held-out snapshot MJCF not in this storage")
    m = mujoco.MjModel.from_xml_path(str(S2.SNAP_XML))
    d = mujoco.MjData(m)
    roots = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in GRIPPER_ROOT_LINKS}
    below = {n: [] for n in roots}
    for b in range(m.nbody):
        j = b
        while j > 0:
            j = int(m.body_parentid[j])
            for n, r in roots.items():
                if j == r:
                    below[n].append(b)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in DEFAULT_RBY1_JOINTS]
    fk = HandProxyFK(str(S2.SNAP_XML))
    refs, _ = HE.heldout_refs(S2.DUMP, S2.PREP)
    picked = [r for r in refs if r.moment == "held"][:3] + [r for r in refs if r.moment == "pre"][:2]
    assert len(picked) == 5
    for ref in picked:
        with np.load(S2.PREP / "replay" / f"{ref.frame_id}.npz") as z:
            qpos = z["qpos"]
        with np.load(ref.path / "inputs.npz") as z:
            raw = np.asarray(z["raw_state"], np.float64)
        d.qpos[:] = qpos
        mujoco.mj_kinematics(m, d)
        Rb, tb = d.xmat[bid].reshape(3, 3), d.xpos[bid]
        srv = fk.proxies(qpos[adr], {"left": raw[7], "right": raw[15]})
        for n in GRIPPER_ROOT_LINKS:
            off = Rb.T @ (d.xpos[below[n]].mean(0) - tb)
            assert np.linalg.norm(off - srv[n]) < 5e-5, (ref.frame_id, n)
