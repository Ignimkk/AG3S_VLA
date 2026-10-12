"""T46c A12 (W1) — after PLACED, under a confirmed `home`, `no_seed` is certified too (flag, default off).

T46b: every after-close uncertified HOLD in CL3 was a placed / home chunk with grounding `no_seed`
(C1a 37 · C1a+F1 129 · C1a+F1+F2 116 chunks): the attention's top seeds all lie on the support
surface, the support mask removes them, and `no_seed` was not in the SUBTASK-f predicate. W1 widens
`ag3s.types.subtask_no_target` by one case, behind `clustering.placed_no_seed_certify`:

    gate on ∧ label `home` ∧ `placed_seen` ∧ grounding `no_seed` ∧ the pipeline wrote
    `metrics["subtask"]["placed_no_seed_certify"] = True` — with SUBTASK-e/f's other conditions
    unchanged (status `no_target`, validity `valid`, nothing carved).

What is tested (T46c.task.md §4):
  (i)    on: a real placed-home `no_seed` frame is certified, executes, and says why;
  (ii)   on: a violating row still HOLDs (certification only removes `uncertified`);
  (iii)  on: no `placed_seen`, label pick / place / None, gate off, not valid, something carved,
         another grounding → as before (HOLD / the SUBTASK-f answer);
  (iv)   off: the record has no key and every decision is bit-identical to the rule before
         (SUBTASK-f) — on real frames, over the full grounding × status × validity table, and on
         recorded CL3 chunks (storage);
  (v)    config / CLI / metadata / record wiring;
  (vi)   the offline replay (`certification_replay`, `ta_offline` W0 · W1 · F1W1).

Frames come from the real `AG3S` pipeline (planar 2-link fixture) with attention only on cells that
see nothing but the table — the shape of the CL3 placed `no_seed` frames.
"""

from __future__ import annotations

import itertools
import json
import pathlib

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError
from AG3S.ag3s.types import (
    ConstraintValidity,
    GroundingStatus,
    PipelineStatus,
    subtask_no_target,
)
from AG3S.trajopt import safe_policy as sp_mod
from AG3S.trajopt import wire
from AG3S.trajopt.safe_policy import geometry_certified
from AG3S.trajopt.types import TrajOptStatus
from tests.trajopt.test_subtask_no_target import (
    HOME,
    PICK,
    PLACE,
    _Manip,
    _policy,
    _replace,
    _scene_fn_certified,
    _Status,
)

CL3 = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/"
                   "20261012T155130+0900_218229ac6464_5fb5ff542c33/verify/T46/CL3")


# ------------------------------------------------------------------------- real no_seed frames


def _support_only_inputs():
    """The fixture scene with attention only on the patches that see nothing but the table."""
    from tests.ag3s.fixtures import DEFAULT_ARM_Q, LABEL_TABLE, PlanarTwoLinkArm, make_scene

    arm = PlanarTwoLinkArm()
    sc = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    (h, w), g = sc.hw, sc.attention_grid.shape[0]
    cells = sc.labels.reshape(g, h // g, g, w // g).transpose(0, 2, 1, 3).reshape(g, g, -1)
    table_only = (cells == LABEL_TABLE).all(-1)
    att = np.where(table_only, 1.0, 0.0).astype(np.float32)
    att += 1e-3 * np.linspace(0.0, 1.0, g * g, dtype=np.float32).reshape(g, g)  # not flat
    return arm, dict(attention_map=att, depth=sc.depth, camera_intrinsics=sc.camera_intrinsics,
                     T_base_cam=sc.T_base_cam, robot_state=DEFAULT_ARM_Q, phase="approach",
                     image_hw=sc.hw)


def _ag3s(arm, gate, flag):
    from AG3S.ag3s.runtime.pipeline import AG3S

    clustering = {"subtask_gate": gate, "subtask_confirm_frames": 1}
    if flag is not None:
        clustering["placed_no_seed_certify"] = flag
    return AG3S(AG3SConfig.from_dict({"pointcloud": {"self_filter_inflation": 0.05},
                                      "clustering": clustering}), robot_model=arm)


def _frames(gate, flag):
    """`pick` (label pick, nothing placed) · `placed_home` (the W1 case) · `home_not_placed` ·
    `placed_place` — all genuine `no_seed` frames."""
    arm, inputs = _support_only_inputs()
    out = {}
    a = _ag3s(arm, gate, flag)
    a.set_subtask(PICK)
    out["pick"] = a.process(**inputs)
    a.set_placed(True)
    a.set_subtask(HOME)
    out["placed_home"] = a.process(**inputs)

    b = _ag3s(arm, gate, flag)
    b.set_subtask(HOME)
    out["home_not_placed"] = b.process(**inputs)

    c = _ag3s(arm, gate, flag)
    c.set_subtask(PICK)
    c.process(**inputs)
    c.set_placed(True)
    c.set_subtask(PLACE)
    out["placed_place"] = c.process(**inputs)
    return out


@pytest.fixture(scope="module")
def on():
    return _frames(True, True)


@pytest.fixture(scope="module")
def off():
    return _frames(True, None)          # config default


@pytest.fixture(scope="module")
def off_explicit():
    return _frames(True, False)


@pytest.fixture(scope="module")
def gate_off_flag_on():
    return _frames(False, True)


def _with_subtask(cs, **changes):
    metrics = dict(cs.metrics)
    if changes.pop("_drop", False):
        metrics.pop("subtask", None)
    else:
        sub = dict(metrics["subtask"], **changes)
        metrics["subtask"] = {k: v for k, v in sub.items() if v is not _DROP}
    return _replace(cs, metrics=metrics)


_DROP = object()


def _wire(sp, cs, trajopt_status, max_violation=0.0):
    import types

    sp.refiner.last_result = types.SimpleNamespace(
        status=_Status(trajopt_status), max_violation=float(max_violation), trajectory=None,
        notes=[], metrics={"max_violation_pair": None, "time_budget_hit": False,
                           "max_iterations_hit": False, "sqp_iterations": 2})
    sp._last_constraint_set = cs                                                # noqa: SLF001
    sp._last_snapshot = None                                                    # noqa: SLF001


def _verdict(cs, trajopt_status="violated", policy="reasons", max_violation=0.0):
    sp = _policy(verdict_policy=policy)
    _wire(sp, cs, trajopt_status, max_violation)
    return sp, sp._verdict(None)                                                # noqa: SLF001


def _assert_hold(cs, grounding="no_seed"):
    assert not subtask_no_target(cs) and geometry_certified(cs) is False
    _sp, v = _verdict(cs)
    assert [r["kind"] for r in v.verdict_reasons] == ["uncertified"] and v.safe is False
    assert v.verdict_reasons[0]["evidence"]["grounding_status"] == grounding
    assert wire.gate_decision(v.verdict_reasons)[0] == "hold"


def test_the_frames_are_what_the_tests_say(on, off, off_explicit, gate_off_flag_on):
    for frames, gate, key in ((on, True, True), (off, True, False), (off_explicit, True, False),
                              (gate_off_flag_on, False, True)):
        for name, cs in frames.items():
            assert cs.status is PipelineStatus.NO_TARGET, name
            assert cs.grounding_status is GroundingStatus.NO_SEED, name
            assert cs.validity is ConstraintValidity.VALID, name
            assert cs.target is None and cs.manipulated is None and cs.attached is None, name
            assert cs.metrics["manipulated"] is None, name
            assert cs.metrics["exclusion"]["source"] == "none", name
            assert cs.metrics["exclusion"]["active"] is False, name
            sub = cs.metrics["subtask"]
            assert sub["gate"] is gate, name
            # the record carries the flag only when on (off: the key does not exist)
            assert ("placed_no_seed_certify" in sub) is key, name
            if key:
                assert sub["placed_no_seed_certify"] is True
        sub = {n: frames[n].metrics["subtask"] for n in frames}
        assert (sub["placed_home"]["label"], sub["placed_home"]["placed_seen"]) == ("home", True)
        assert (sub["home_not_placed"]["label"],
                sub["home_not_placed"]["placed_seen"]) == ("home", False)
        assert (sub["pick"]["label"], sub["pick"]["placed_seen"]) == ("pick", False)
        assert (sub["placed_place"]["label"], sub["placed_place"]["placed_seen"]) == ("place", True)


def test_the_flag_changes_nothing_but_the_record_key(on, off):
    """Grounding, status, validity, carve record: the flag decides certification only."""
    for name in on:
        a, b = on[name], off[name]
        assert (a.status, a.grounding_status, a.validity) == (b.status, b.grounding_status,
                                                               b.validity)
        sa = {k: v for k, v in a.metrics["subtask"].items() if k != "placed_no_seed_certify"}
        assert sa == b.metrics["subtask"]
        assert a.metrics["exclusion"] == b.metrics["exclusion"]
        assert a.metrics["target_permission"] == b.metrics["target_permission"]


# ------------------------------------------------------------------------------ (i)


def test_i_placed_home_no_seed_is_certified(on):
    cs = on["placed_home"]
    assert subtask_no_target(cs) and cs.subtask_no_target
    assert geometry_certified(cs) is True
    assert cs.grounding_status is GroundingStatus.NO_SEED          # not rewritten
    assert cs.status is PipelineStatus.NO_TARGET


@pytest.mark.parametrize("policy", ["reasons", "legacy"])
def test_i_the_verdict_executes_and_says_why(on, policy):
    sp, v = _verdict(on["placed_home"], "optimal", policy=policy)
    assert v.safe is True and v.geometry_certified is True
    assert v.ag3s_status == "no_target"
    assert [r["kind"] for r in v.verdict_reasons] == ["subtask_no_target"]
    r = v.verdict_reasons[0]
    assert r["action"] == "execute"
    ev = r["evidence"]
    assert ev["grounding_status"] == "no_seed" and ev["placed_no_seed_certify"] is True
    assert ev["validity"] == "valid" and ev["ag3s_status"] == "no_target"
    assert ev["subtask_label"] == "home" and ev["placed_seen"] is True
    assert ev["exclusion_source"] == "none" and ev["exclusion_active"] is False
    assert "no attention seed is left off the support surface" in r["detail"]
    out = v.to_dict()
    reasons, source = wire.unpack_verdict_reasons(out)
    assert source == "server" and wire.gate_decision(reasons)[0] == "execute"
    assert sp._exec_summary()["verdict"]["reasons"][0]["kind"] == "subtask_no_target"  # noqa: SLF001


def test_i_the_other_reason_texts_are_unchanged(on):
    """SUBTASK-f's `no_admissible` evidence has no W1 key; the wording is the one before."""
    cs = _replace(on["placed_home"], grounding_status=GroundingStatus.NO_ADMISSIBLE)
    _sp, v = _verdict(cs, "optimal")
    r = v.verdict_reasons[0]
    assert r["evidence"]["grounding_status"] == "no_admissible"
    assert "placed_no_seed_certify" not in r["evidence"]
    assert "nothing admissible is in view after the object was placed" in r["detail"]


def test_i_scene_fn_hands_certified_to_the_optimizer(on, off, monkeypatch):
    assert _scene_fn_certified(on["placed_home"], monkeypatch) is True
    assert _scene_fn_certified(off["placed_home"], monkeypatch) is False
    assert _scene_fn_certified(on["home_not_placed"], monkeypatch) is False
    assert _scene_fn_certified(on["pick"], monkeypatch) is False


@pytest.fixture(scope="module")
def scenario():
    pytest.importorskip("casadi")
    pytest.importorskip("osqp")
    from tests.trajopt.fixtures import make_scenario

    return make_scenario(horizon=10, penetration=0.03)


def test_i_the_optimizer_status_follows_the_predicate(on, off, scenario):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    def solve(cs):
        opt = TrajectoryOptimizer(scenario.robot, scenario.layout, scenario.limits, scenario.config)
        return opt.solve(scenario.reference, scenario.q_now, scenario.scene,
                         geometry_certified=geometry_certified(cs))

    ok = solve(on["placed_home"])
    assert ok.safe and ok.metrics["geometry_certified"] is True
    held = solve(off["placed_home"])
    assert held.status is TrajOptStatus.VIOLATED
    assert any("could not certify" in n for n in held.notes)


# ------------------------------------------------------------------------------ (ii)


@pytest.mark.parametrize("policy", ["reasons", "legacy"])
def test_ii_a_violating_row_still_holds(on, policy):
    """Certification removes `uncertified` only — the collision row is still a HOLD."""
    _sp, v = _verdict(on["placed_home"], "violated", policy=policy, max_violation=0.012)
    kinds = [r["kind"] for r in v.verdict_reasons]
    assert v.safe is False
    assert "uncertified" not in kinds
    assert "collision" in kinds
    assert wire.gate_decision(v.verdict_reasons)[0] == "hold"
    if policy == "reasons":
        assert kinds[0] == "subtask_no_target"


# ------------------------------------------------------------------------------ (iii)


@pytest.mark.parametrize("placed", ["real_not_placed", False, None, "missing", "true_string", 1])
def test_iii_without_placed_seen_it_holds(on, placed):
    if placed == "real_not_placed":
        cs = on["home_not_placed"]
    elif placed == "missing":
        cs = _with_subtask(on["placed_home"], placed_seen=_DROP)
    elif placed == "true_string":
        cs = _with_subtask(on["placed_home"], placed_seen="true")
    else:
        cs = _with_subtask(on["placed_home"], placed_seen=placed)
    _assert_hold(cs)


@pytest.mark.parametrize("label", ["real_pick", "real_placed_place", "pick", "place", None, "HOME"])
def test_iii_label_not_home_holds(on, label):
    if label == "real_pick":
        cs = on["pick"]
    elif label == "real_placed_place":
        cs = on["placed_place"]
    else:
        cs = _with_subtask(on["placed_home"], label=label)
    _assert_hold(cs)


def test_iii_gate_off_holds_even_with_the_flag(gate_off_flag_on):
    for name, cs in gate_off_flag_on.items():
        _assert_hold(cs)


@pytest.mark.parametrize("flag", [_DROP, False, None, 1, "true"], ids=repr)
def test_iii_the_record_key_must_be_exactly_true(on, flag):
    _assert_hold(_with_subtask(on["placed_home"], placed_no_seed_certify=flag))


def test_iii_no_subtask_record_holds(on):
    _assert_hold(_with_subtask(on["placed_home"], _drop=True))


@pytest.mark.parametrize("validity,status", [
    (ConstraintValidity.DEGRADED, PipelineStatus.NO_TARGET),
    (ConstraintValidity.INCOMPLETE, PipelineStatus.NO_TARGET),
    (ConstraintValidity.DEGRADED, PipelineStatus.DEGRADED),
], ids=["degraded_no_target", "incomplete_no_target", "degraded"])
def test_iii_not_valid_holds(on, validity, status):
    _assert_hold(_replace(on["placed_home"], validity=validity, status=status))


@pytest.mark.parametrize("change", [
    {"manipulated": _Manip()},
    {"attached": object()},
    {"target_field_exclude": np.ones(3, bool)},
    {"manipulated_link_margin": np.zeros(3)},
    {"target": object()},
    {"metrics": "manipulated_record"},
    {"metrics": "exclusion_active"},
], ids=["manipulated", "attached", "target_free_layer", "contact_margin", "target",
        "manipulated_record", "exclusion_active"])
def test_iii_anything_carved_keeps_it_uncertified(on, change):
    base = on["placed_home"]
    if "metrics" in change:
        metrics = dict(base.metrics)
        if change["metrics"] == "manipulated_record":
            metrics["manipulated"] = {"id": 3, "state": "visible", "age_frames": 0}
        else:
            metrics["exclusion"] = dict(metrics["exclusion"], active=True)
        cs = _replace(base, metrics=metrics)
    else:
        cs = _replace(base, **change)
    assert not subtask_no_target(cs) and geometry_certified(cs) is False


@pytest.mark.parametrize("grounding", [GroundingStatus.LOST, GroundingStatus.LOW_SCORE,
                                       GroundingStatus.NO_CLUSTER, GroundingStatus.NO_GEOMETRY,
                                       GroundingStatus.NO_ATTENTION], ids=lambda g: g.value)
def test_iii_another_grounding_still_holds(on, grounding):
    _assert_hold(_replace(on["placed_home"], grounding_status=grounding), grounding=grounding.value)


# ------------------------------------------------------------------------------ (iv) off = before


def _rule_before_w1(cs):
    """The SUBTASK-f predicate: today's predicate without the W1 branch (the key removed)."""
    m = getattr(cs, "metrics", None)
    if isinstance(m, dict) and isinstance(m.get("subtask"), dict):
        sub = {k: v for k, v in m["subtask"].items() if k != "placed_no_seed_certify"}
        cs = _replace(cs, metrics=dict(m, subtask=sub))
    return subtask_no_target(cs)


def test_iv_the_widened_predicate_differs_from_subtask_f_in_exactly_one_cell(on):
    """flag × gate × label × placed_seen × grounding × status × validity: the predicate equals the
    SUBTASK-f rule everywhere except flag ∧ gate ∧ home ∧ placed ∧ no_seed ∧ no_target ∧ valid."""
    base = on["placed_home"]
    n = n_diff = 0
    for flag, gate, label, placed, g, s, v in itertools.product(
            (True, _DROP), (True, False), ("pick", "place", "home", None), (True, False),
            list(GroundingStatus), list(PipelineStatus), list(ConstraintValidity)):
        cs = _replace(_with_subtask(base, gate=gate, label=label, placed_seen=placed,
                                    placed_no_seed_certify=flag),
                      grounding_status=g, status=s, validity=v)
        widened = (flag is True and gate and label == "home" and placed
                   and g is GroundingStatus.NO_SEED and s is PipelineStatus.NO_TARGET
                   and v is ConstraintValidity.VALID)
        new = subtask_no_target(cs)
        assert new is (_rule_before_w1(cs) or widened), (flag, gate, label, placed, g, s, v)
        assert geometry_certified(cs) is (s is PipelineStatus.OK or new)
        n += 1
        n_diff += new != _rule_before_w1(cs)
    assert n_diff == 1 and n == 2 * 2 * 4 * 2 * len(GroundingStatus) * len(PipelineStatus) * len(
        ConstraintValidity)


def _snapshot(cs, trajopt_status, policy, monkeypatch, rule, max_violation=0.0):
    sp = _policy(verdict_policy=policy)
    with monkeypatch.context() as m:
        if rule is not None:
            m.setattr(sp_mod, "subtask_no_target", rule)
        _wire(sp, cs, trajopt_status, max_violation)
        v = sp._verdict(None)                                                   # noqa: SLF001
        return v.to_dict(), sp._last_reasons                                    # noqa: SLF001


@pytest.mark.parametrize("policy", ["reasons", "legacy"])
@pytest.mark.parametrize("trajopt_status,mv", [("optimal", 0.0), ("violated", 0.0),
                                               ("violated", 0.012)])
def test_iv_off_verdicts_are_bit_identical_to_the_rule_before(off, off_explicit, policy,
                                                             trajopt_status, mv, monkeypatch):
    frames = [f[n] for f in (off, off_explicit) for n in f]
    frames += [_replace(off["placed_home"], grounding_status=g) for g in GroundingStatus]
    for cs in frames:
        new = _snapshot(cs, trajopt_status, policy, monkeypatch, None, mv)
        before = _snapshot(cs, trajopt_status, policy, monkeypatch, _rule_before_w1, mv)
        assert new == before, cs.grounding_status
        assert _scene_fn_certified(cs, monkeypatch) is geometry_certified(
            _replace(cs)) is (cs.status.value == "ok" or _rule_before_w1(cs))


def test_iv_control_on_differs_from_the_rule_before(on, monkeypatch):
    """Not vacuous: on the flag-on placed-home frame the rule before is the T46b HOLD."""
    new = _snapshot(on["placed_home"], "violated", "reasons", monkeypatch, None)
    old = _snapshot(on["placed_home"], "violated", "reasons", monkeypatch, _rule_before_w1)
    assert new != old
    assert [r["kind"] for r in old[1]] == ["uncertified"]
    assert "grounding=no_seed, validity=valid" in old[1][0]["detail"]


def _cl3_runs(names):
    if not (CL3 / "analysis").is_dir():
        pytest.skip("CL3 records not in storage")
    out = []
    for n in names:
        p = CL3 / "analysis" / f"{n}.json"
        if not p.is_file():
            pytest.skip(f"{p} missing")
        out.append(json.loads(p.read_text()))
    return out


@pytest.mark.parametrize("name", ["C1a_ep54_s541", "C1aF1_ep11_s111", "C1aF1F2_ep0_s1"])
def test_iv_recorded_chunks_off_is_the_recorded_certification(name):
    """On every recorded chunk of the run: the predicate as recorded (no key) gives the recorded
    `to.metrics.geometry_certified` and the recorded `subtask_no_target` reason; the recorded
    `subtask` record has no W1 key (CL3 ran before W1)."""
    from AG3S.trajopt.experiments import certification_replay as CR

    (a,) = _cl3_runs([name])
    n = 0
    for c in a["server_chunks"]:
        s = CR._summary(c["server_file"])                                       # noqa: SLF001
        assert "placed_no_seed_certify" not in (s.get("subtask") or {})
        kinds = [r.get("kind") for r in CR.recorded_reasons(s)]
        assert CR.subtask_certified(s) is ("subtask_no_target" in kinds), c["seq"]
        assert CR.certified(s) is CR.certified(s, placed_no_seed_certify=False)
        rec = ((s.get("to") or {}).get("metrics") or {}).get("geometry_certified")
        if rec is not None:
            assert CR.certified(s) is bool(rec), c["seq"]
        assert CR.replay_reasons(s, placed_no_seed_certify=None) == CR.recorded_reasons(s)
        n += 1
    assert n == len(a["server_chunks"]) > 0


# ------------------------------------------------------------------------------ (v) wiring


def test_v_config_default_off_round_trip_and_validation():
    cfg = AG3SConfig()
    assert cfg.clustering.placed_no_seed_certify is False
    on_cfg = AG3SConfig.from_dict({"clustering": {"placed_no_seed_certify": True}})
    assert on_cfg.clustering.placed_no_seed_certify is True
    assert AG3SConfig.from_dict(on_cfg.to_dict()).clustering.placed_no_seed_certify is True
    assert cfg.with_overrides({"clustering": {"placed_no_seed_certify": True}}) \
        .clustering.placed_no_seed_certify is True
    for bad in (1, "true", None, 0.0):
        with pytest.raises(AG3SConfigError):
            AG3SConfig.from_dict({"clustering": {"placed_no_seed_certify": bad}})


def _parse(*argv):
    from AG3S.trajopt.serve_safe import build_parser

    ap = build_parser()
    return ap, ap.parse_args(["--config", "c", "--checkpoint", "k", "--model-xml", "x.xml", *argv])


def test_v_cli_flag_maps_to_the_clustering_key_only_when_given():
    from AG3S.trajopt.serve_safe import (assert_ag3s_config, placed_no_seed_section,
                                         reject_bad_flag_combinations)

    ap, args = _parse()
    assert args.placed_no_seed_certify is False and placed_no_seed_section(args) == {}
    ap, args = _parse("--placed-no-seed-certify")
    reject_bad_flag_combinations(ap, args)
    sec = placed_no_seed_section(args)
    assert sec == {"placed_no_seed_certify": True}
    cfg = AG3SConfig.from_dict({"collision_backend": "esdf"}).with_overrides({"clustering": sec})
    assert_ag3s_config(cfg, collision_backend="esdf", esdf_backend=cfg.esdf.backend,
                       clustering=sec)
    with pytest.raises(RuntimeError):
        assert_ag3s_config(AG3SConfig.from_dict({"collision_backend": "esdf"}),
                           collision_backend="esdf", esdf_backend=cfg.esdf.backend,
                           clustering=sec)


@pytest.mark.parametrize("extra", [("--no-safe",), ("--no-attention",), ("--no-perception",),
                                   ("--no-subtask-gate",)])
def test_v_cli_rejects_the_flag_where_it_cannot_act(extra, capsys):
    from AG3S.trajopt.serve_safe import reject_bad_flag_combinations

    ap, args = _parse("--placed-no-seed-certify", *extra)
    with pytest.raises(SystemExit) as e:
        reject_bad_flag_combinations(ap, args)
    assert e.value.code == 2
    assert "--placed-no-seed-certify" in capsys.readouterr().err


def _safe_policy_with(flag):
    from AG3S.ag3s.runtime.pipeline import AG3S
    from AG3S.trajopt.safe_policy import SafePolicy, grasp_parent_links
    from tests.trajopt.test_safe_policy import _robot_model, _StubPolicy

    model = _robot_model()
    clustering = {"gripper_max_opening": 0.0715}
    if flag is not None:
        clustering["placed_no_seed_certify"] = flag
    ag3s = AG3S(AG3SConfig.from_dict({"collision_backend": "esdf", "clustering": clustering}),
                robot_model=model, constraint_robot_model=model,
                attached_parent_links=grasp_parent_links())
    return SafePolicy(_StubPolicy(), ag3s=ag3s)


def test_v_client_metadata_carries_the_flag_only_when_on():
    base = _safe_policy_with(None).metadata
    assert "placed_no_seed_certify" not in base
    assert _safe_policy_with(False).metadata == base
    on = _safe_policy_with(True).metadata
    assert on.pop("placed_no_seed_certify") is True
    assert on == base


def test_v_server_record_carries_the_subtask_key(on, off):
    """`ConstraintRecordWriter` copies `metrics["subtask"]` into `summary_json` as is."""
    import inspect

    from AG3S.ag3s.experiments.sources import constraint_record

    src = inspect.getsource(constraint_record)
    assert '"subtask"' in src
    assert on["placed_home"].metrics["subtask"]["placed_no_seed_certify"] is True
    assert "placed_no_seed_certify" not in off["placed_home"].metrics["subtask"]


# ------------------------------------------------------------------------------ (vi) replay


def _summary_from(cs, reasons):
    sub = cs.metrics["subtask"]
    return {"status": cs.status.value, "grounding_status": cs.grounding_status.value,
            "validity": cs.validity.value, "has_target": cs.target is not None,
            "manipulated": cs.metrics["manipulated"], "exclusion": cs.metrics["exclusion"],
            "subtask": {k: v for k, v in sub.items() if k != "placed_no_seed_certify"},
            "target_permission": {}, "grasp": {"state": "placed"},
            "verdict": {"reasons": reasons}}


def test_vi_replay_record_matches_the_live_predicate(on, off):
    from AG3S.trajopt.experiments import certification_replay as CR

    for name in on:
        s = _summary_from(off[name], [])
        assert CR.subtask_certified(s) is subtask_no_target(off[name])
        assert CR.subtask_certified(s, placed_no_seed_certify=True) is subtask_no_target(on[name])
        assert CR.certified(s, placed_no_seed_certify=True) is geometry_certified(on[name])


def test_vi_replay_converts_only_an_uncertified_only_hold(off):
    from AG3S.trajopt.experiments import certification_replay as CR

    unc = wire.make_reason("uncertified", "recorded")
    col = wire.make_reason("collision", "recorded")
    s = _summary_from(off["placed_home"], [unc])
    assert CR.decision(CR.recorded_reasons(s)) == "hold"
    w1 = CR.replay_reasons(s, placed_no_seed_certify=True)
    assert [r["kind"] for r in w1] == ["subtask_no_target"] and CR.decision(w1) == "execute"
    # a violating row stays a HOLD
    s2 = _summary_from(off["placed_home"], [unc, col])
    w1b = CR.replay_reasons(s2, placed_no_seed_certify=True)
    assert [r["kind"] for r in w1b] == ["subtask_no_target", "collision"]
    assert CR.decision(w1b) == "hold"
    # not placed / pick: no change under W1
    for name in ("home_not_placed", "pick", "placed_place"):
        s3 = _summary_from(off[name], [unc])
        assert CR.replay_reasons(s3, placed_no_seed_certify=True) == [unc]


def test_vi_ta_offline_variants():
    from AG3S.trajopt.experiments import ta_offline as TA

    assert TA.variant_certify("base") is None and TA.variant_certify("F1") is None
    assert TA.variant_certify("W0") == {"placed_no_seed_certify": None}
    assert TA.variant_certify("W1") == {"placed_no_seed_certify": True}
    assert TA.variant_certify("F1W1") == {"placed_no_seed_certify": True}
    assert TA.variant_target_free("F1W1") == TA.variant_target_free("F1")
    assert TA.variant_target_free("W1") is None and TA.variant_server_args("W1") == ()
    for v in ("W0", "W1", "F1W1"):
        assert not {k for k in TA.VARIANTS[v] if not k.startswith("_")}   # no TO override


def test_vi_replay_reproduces_t46b_on_cl3():
    """T46b (`T46b.verify.json` numbers.W1_execute_chunks / after_close_hold_under_W1_per_chunk)."""
    from AG3S.trajopt.experiments import certification_replay as CR

    if not (CL3 / "analysis").is_dir():
        pytest.skip("CL3 records not in storage")
    res = CR.replay_runs(CL3, ("W0", "W1"))
    want = {"C1a": (37, 70, 0, 107, 37), "C1aF1": (129, 54, 1, 183, 130),
            "C1aF1F2": (116, 6, 2, 122, 118)}
    for arm, (conv, tot, unc, tot0, unc0) in want.items():
        W1 = res[arm]["variants"]["W1"]
        W0 = res[arm]["variants"]["W0"]
        assert W1["execute_conversions"] == conv, arm
        assert W1["after_close_hold"] == {"total": tot, "uncertified": unc}, arm
        assert W0["execute_conversions"] == 0 and W0["flipped"] == {}, arm
        assert W0["after_close_hold"] == {"total": tot0, "uncertified": unc0}, arm
        assert set(W1["conversions_by"]) == {
            "grounding=no_seed|phase=placed|label=home|placed_seen=True"}, arm
        # predicate fidelity on every chunk (1350 per arm)
        assert set(res[arm]["fidelity"]) <= {"pred=True|rec=True", "pred=False|rec=False"}, arm
        assert sum(res[arm]["fidelity"].values()) == 1350, arm
