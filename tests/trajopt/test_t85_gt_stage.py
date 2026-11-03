"""T85 — the MuJoCo ground-truth task stage instead of the stage classifier (`serve_safe --hsr-gt-stage` +
`pi05_infer --safe-gt-stage`, analysis only, default off).

Task: `AG3S/ag3s/docs/handoff/T85.task.md` §2–§3.1. Report: `T85.impl.md`.

What this file fixes:

0. imports come from this checkout.
1. the definition: `CausalPhase5` (one request at a time) gives, at every request, the entry `t77_labels.phase5_sequence`
   gives for the same input columns (held / palm distance / in-basket), a prefix of a run never changes with what comes
   later, and the live probe (`SceneProbe.live`, private `MjData`) measures what the offline probe measures on the same
   qpos without touching the live data.
2. the wire: no key without `gt_stage` (request as before), round trip, server metadata + the client pairing refusal.
3. `SafePolicy`: a GT stage drives `rule_phase6` · the subtask gate · the after-grasp terms exactly as `--hsr5`'s argmax of
   the same stage, the record says `probe=gt`; off = no `gt_stage_required`, no gt record, classifier path as before.
4. `serve_safe`: off adds nothing; on builds the GT config (no probe / noun source needed), refuses `--hsr5` together, no
   `--hsr6-*`, `--no-safe`.
5. off = as before **bit for bit**: every T85 function is replaced with one that explodes, and the off paths still run.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t77_labels as TL
from AG3S.trajopt import hsr5 as H5
from AG3S.trajopt import hsr6 as H6
from AG3S.trajopt import wire
from AG3S.trajopt.client import SafeRemoteClient
from tests.trajopt.test_grasp_evidence import _policy
from tests.trajopt.test_safe_client import DOF, _Scene, _Server

REPO = pathlib.Path(__file__).resolve().parents[2]
CFG_GT = H5.Hsr5Config(probe=H5.GT_PROBE_NAME, gt_stage=True)


def _rec(stage):
    """What `--hsr5` hands SafePolicy for a stage (a classifier-shaped record)."""
    p = {c: (0.9 if c == stage else 0.025) for c in H5.STAGES5}
    return {"p": p, "argmax": stage, "probe": H5.PROBE_NAME, "ms": 1.0}


def _boom(*a, **k):
    raise AssertionError("a T85 function ran with the flag off")


# ------------------------------------------------------------------------------------------------ 0. imports
def test_imports_from_this_checkout():
    from AG3S.trajopt import client, gt_stage, safe_policy, serve_safe

    for mod in (TL, H5, wire, client, gt_stage, safe_policy, serve_safe):
        assert pathlib.Path(mod.__file__).resolve().is_relative_to(REPO), mod.__file__


# ------------------------------------------------------------------------------------------------ 1. the definition
def _mes(touch, z, palm, basket):
    return {"fingers_touch": touch, "apple_z_base": z, "palm_dist_m": palm, "in_basket": basket}


#: approach, approach, grasp (palm <= 60 mm, empty-hand close included), held lift, carry, over the basket, release.
RUN = [_mes(0, 0.800, 0.30, False), _mes(0, 0.800, 0.12, False), _mes(0, 0.800, 0.05, False),
       _mes(2, 0.805, 0.01, False),                     # touches but rose 5 mm < 20 mm: not held yet -> grasp
       _mes(2, 0.830, 0.01, False), _mes(2, 0.900, 0.01, False),    # carry
       _mes(2, 0.900, 0.01, True), _mes(2, 0.880, 0.01, True),       # place
       _mes(0, 0.800, 0.20, True), _mes(0, 0.800, 0.40, False)]      # retreat (never back to approach)


def _batch(run):
    z0 = run[0]["apple_z_base"]
    held = [TL.gt_held(r["fingers_touch"], (r["apple_z_base"] - z0) * 1e3) for r in run]
    return TL.phase5_sequence(held, [r["palm_dist_m"] for r in run], [r["in_basket"] for r in run])


def test_causal_stage_is_phase5_sequence_on_the_same_columns():
    ph = TL.CausalPhase5()
    got = [ph.update(r) for r in RUN]
    assert got == _batch(RUN)
    assert got == ["approach", "approach", "grasp", "grasp", "carry", "carry", "place", "place", "retreat", "retreat"]


def test_a_prefix_never_changes_with_what_comes_later():
    for n in range(1, len(RUN) + 1):
        ph = TL.CausalPhase5()
        assert [ph.update(r) for r in RUN[:n]] == _batch(RUN)[:n]


def test_reset_starts_a_new_run_and_an_error_request_is_unlabelled():
    ph = TL.CausalPhase5()
    [ph.update(r) for r in RUN]
    ph.reset()
    assert ph.update(RUN[0]) == "approach"                 # `ever_held` and request 0's height are gone
    assert ph.update({"error": "qpos 3 != nq"}) is None
    assert ph.update(RUN[1]) == "approach"                 # the bad request did not enter the columns


def _xml():
    xml = REPO / TL.XML_REL
    if not xml.exists():
        pytest.skip("scene XML missing")
    return str(xml)


@pytest.fixture(scope="module")
def live_scene():
    import mujoco

    m = mujoco.MjModel.from_xml_path(_xml())
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def _scene_of(m, d):
    import types

    return types.SimpleNamespace(model=m, data=d)


def _put_apple(m, d, xyz):
    import mujoco

    adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "apple_free")]
    d.qpos[adr:adr + 3] = xyz
    mujoco.mj_forward(m, d)


def test_live_probe_equals_the_offline_probe_and_leaves_the_live_data_alone(live_scene):
    import mujoco

    m, d = live_scene
    live = TL.SceneProbe.live(m, d)
    off = TL.SceneProbe(_xml(), (None, "clear"))
    site = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_ee")].copy()
    _put_apple(m, d, site + [0.02, 0.0, 0.03])
    before = (d.qpos.copy(), d.qvel.copy(), d.xpos.copy(), d.qacc_warmstart.copy(), d.time)
    a = live.measure_live(d, "left")
    after = (d.qpos, d.qvel, d.xpos, d.qacc_warmstart, d.time)
    assert all(np.array_equal(x, y) for x, y in zip(before[:4], after[:4])) and before[4] == after[4]
    b = off.measure(d.qpos.copy(), "left")
    assert a == b and 0.0 < a["palm_dist_m"] < 0.06


def test_gt_stage_source_walks_approach_then_grasp_and_resets(live_scene):
    import mujoco

    from AG3S.trajopt.gt_stage import GtStageSource

    m, d = live_scene
    src = GtStageSource(_scene_of(m, d), side="left")
    site = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_ee")].copy()
    _put_apple(m, d, site + [0.0, 0.0, 0.5])
    assert src.next_stage() == "approach"
    _put_apple(m, d, site + [0.02, 0.0, 0.03])
    assert src.next_stage() == "grasp"
    assert src.last["stage"] == "grasp" and src.last["side"] == "left" and src.last["target"] == "apple"
    src.reset()
    assert src.last == {} and src._phase.z0 is None
    with pytest.raises(ValueError):
        GtStageSource(_scene_of(m, d), side="middle")
    with pytest.raises(RuntimeError):
        GtStageSource(_scene_of(m, d), side="left", target_body="no_such_body")


# ------------------------------------------------------------------------------------------------ 2. the wire
def _pack_kwargs():
    return dict(obs={"state": np.zeros(DOF)}, cameras=("zed_left",), depth={"zed_left": np.full((4, 4), 1.0)},
                intrinsics={"zed_left": np.eye(3)}, extrinsics={"zed_left": np.eye(4)},
                robot_state={"zed_left": np.zeros(DOF)}, stamps={"zed_left": 1.0}, phase="approach")


def test_pack_request_has_no_key_without_gt_stage():
    base = wire.pack_request(**_pack_kwargs())
    assert wire.pack_request(**_pack_kwargs(), gt_stage=None).keys() == base.keys()
    assert wire.PREFIX + wire.GT_STAGE not in base and wire.unpack_gt_stage(wire.strip_request(base)[1]) is None


def test_gt_stage_round_trip():
    req = wire.pack_request(**_pack_kwargs(), gt_stage="carry")
    assert set(req) - set(wire.pack_request(**_pack_kwargs())) == {wire.PREFIX + wire.GT_STAGE}
    assert wire.unpack_gt_stage(wire.strip_request(req)[1]) == "carry"


class _GtSource:
    def __init__(self, stages):
        self.stages, self.resets, self.calls = list(stages), 0, 0

    def reset(self):
        self.resets += 1

    def next_stage(self):
        self.calls += 1
        return self.stages.pop(0)


def test_client_sends_the_stage_per_request_and_resets_on_the_first(monkeypatch):
    src = _GtSource(["approach", None, "grasp"])
    server = _Server(metadata={wire.GT_STAGE_REQUIRED: True})
    client = SafeRemoteClient(policy=server, scene=_Scene(), cameras=("zed_left",), gt_stage=src)
    for i in range(3):
        client.infer({"state": np.zeros(DOF)}, reset=(i == 0))
    got = [wire.unpack_gt_stage(wire.strip_request(r)[1]) for r in server.requests]
    assert got == ["approach", None, "grasp"] and src.resets == 1 and src.calls == 3   # unmeasurable request: no key


@pytest.mark.parametrize("server_wants,client_has", [(True, False), (False, True)])
def test_pairing_mismatch_dies_on_connect(server_wants, client_has):
    meta = {wire.GT_STAGE_REQUIRED: True} if server_wants else {}
    with pytest.raises(RuntimeError, match="gt-stage pairing"):
        SafeRemoteClient(policy=_Server(metadata=meta), scene=_Scene(), cameras=("zed_left",),
                         gt_stage=_GtSource([]) if client_has else None)


def test_pairing_ok_when_both_or_neither():
    SafeRemoteClient(policy=_Server(metadata={wire.GT_STAGE_REQUIRED: True}), scene=_Scene(),
                     cameras=("zed_left",), gt_stage=_GtSource([]))
    SafeRemoteClient(policy=_Server(metadata={}), scene=_Scene(), cameras=("zed_left",))


# ------------------------------------------------------------------------------------------------ 3. SafePolicy
def test_the_record_is_classifier_shaped_and_says_gt():
    for stage in H5.STAGES5:
        rec = H5.gt_stage_record(stage)
        assert H5.stage_of(rec) == stage and rec["argmax"] == stage and rec["probe"] == "gt"
        assert rec["p"] == {c: float(c == stage) for c in H5.STAGES5}
    assert H5.gt_stage_record(None) is None and H5.gt_stage_record("hover") is None


def test_hsr5_record_and_config_are_unchanged_when_not_gt():
    cfg = H5.Hsr5Config()
    assert cfg.gt_stage is False and cfg.record() == {"probe": "shipped:" + H5.PROBE_NAME, "release_frames": 2,
                                                      "decision": "raw argmax per request"}
    assert CFG_GT.record()["gt_stage"] is True and "MuJoCo" in CFG_GT.record()["decision"]


@pytest.mark.parametrize("stage", list(H5.STAGES5) + [None])
def test_gt_stage_drives_the_rules_as_hsr5_does(stage):
    flags = H6.Hsr6Flags(attach_assist=True, held_gate=True, target_exempt=True, objective=True)
    ref = _policy(hsr6=flags, hsr5=H5.Hsr5Config())
    ref._hsr5_start(_rec(stage) if stage else None)
    gt = _policy(hsr6=flags, hsr5=CFG_GT)
    gt._hsr5_start(H5.gt_stage_record(stage))
    assert gt._hsr5_stage() == ref._hsr5_stage() == stage
    assert gt._hsr6_phase() == ref._hsr6_phase() == H5.rule_phase6(stage)
    assert gt._hsr6_objective_ok() is ref._hsr6_objective_ok()
    assert gt._subtask_gate_input() == ref._subtask_gate_input() == H5.subtask3_probabilities(stage)
    for rule in ("deflection_rate", "continuity"):
        assert gt._hsr5_after_grasp(rule) == ref._hsr5_after_grasp(rule) == H5.after_grasp_on(stage)
    assert gt._last_hsr5["probe"] == ("gt" if stage else None)


def test_infer_reads_the_client_stage_only_with_gt_on():
    """`infer` source: the GT branch is guarded by the config flag, the classifier line is the one from before."""
    import inspect

    from AG3S.trajopt import safe_policy

    src = inspect.getsource(safe_policy.SafePolicy.infer)
    assert 'getattr(self.hsr5, "gt_stage", False)' in src
    assert 'self._hsr5_start(result.get("hsr5") if isinstance(result, dict) else None)' in src
    assert src.index("gt_stage_record(wire.unpack_gt_stage(scene))") < src.index('self._hsr5_start(result.get("hsr5")')


def test_metadata_carries_gt_stage_required_only_with_gt():
    assert wire.GT_STAGE_REQUIRED not in _policy().metadata
    assert wire.GT_STAGE_REQUIRED not in _policy(hsr5=H5.Hsr5Config()).metadata
    meta = _policy(hsr5=CFG_GT).metadata
    assert meta[wire.GT_STAGE_REQUIRED] is True and meta["hsr5"] == CFG_GT.record()


# ------------------------------------------------------------------------------------------------ 4. serve_safe
BASE = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]
NOUN = ["--attention-source", "noun_target:L10H7"]
H6F = ["--hsr6-attach-assist", "--hsr6-objective", "--neighbour-contact-mm", "20"]


def _parse(argv):
    from AG3S.trajopt import serve_safe

    ap = serve_safe.build_parser()
    a = ap.parse_args(BASE + list(argv))
    serve_safe.reject_hsr6_flags(ap, a)
    serve_safe.reject_hsr5_flags(ap, a)
    return a


def test_serve_off_adds_nothing():
    from AG3S.trajopt import serve_safe

    a = _parse(NOUN + H6F)
    assert a.hsr_gt_stage is False and not serve_safe.hsr_gt_stage_on(a)
    assert serve_safe.hsr5_config_from_args(a) is None and serve_safe.hsr5_record_meta(a) == {}


def test_serve_on_builds_the_gt_config():
    from AG3S.trajopt import serve_safe

    a = _parse(["--hsr-gt-stage"] + H6F)                  # no noun source: no hidden feature is needed
    cfg = serve_safe.hsr5_config_from_args(a)
    assert cfg == CFG_GT and serve_safe.hsr_gt_stage_on(a)
    assert serve_safe.hsr5_record_meta(a) == {"hsr5": CFG_GT.record()}
    b = _parse(["--hsr-gt-stage"] + NOUN + H6F)
    assert serve_safe.hsr5_config_from_args(b) == CFG_GT


@pytest.mark.parametrize("argv", [
    ["--hsr-gt-stage", "--hsr5"] + NOUN + H6F,           # two stage sources
    ["--hsr-gt-stage"],                                   # no --hsr6-* rule to feed
    ["--hsr-gt-stage", "--no-safe"] + H6F,
    ["--hsr-gt-stage", "--no-attention"] + H6F])
def test_serve_refuses(argv):
    with pytest.raises(SystemExit):
        _parse(argv)


# ------------------------------------------------------------------------------------------------ 5. off = as before
def test_off_never_touches_a_t85_function(monkeypatch):
    """Every T85 function explodes; the flag-off client request, SafePolicy classifier path and serve parse still run."""
    from AG3S.trajopt import gt_stage, serve_safe

    monkeypatch.setattr(gt_stage.GtStageSource, "__init__", _boom)
    monkeypatch.setattr(gt_stage.GtStageSource, "next_stage", _boom)
    monkeypatch.setattr(H5, "gt_stage_record", _boom)
    monkeypatch.setattr(wire, "unpack_gt_stage", _boom)
    # client: no stage source -> not measured, not sent, request keys as before
    server = _Server()
    client = SafeRemoteClient(policy=server, scene=_Scene(), cameras=("zed_left",))
    client.infer({"state": np.zeros(DOF)}, reset=True)
    assert wire.PREFIX + wire.GT_STAGE not in server.requests[0]
    # server policy: the --hsr5 classifier record path and the no-hsr5 path
    sp = _policy(hsr5=H5.Hsr5Config())
    sp._hsr5_start(_rec("carry"))
    assert sp._hsr5_stage() == "carry" and "stage_source" not in sp._last_hsr5
    assert sp.metadata["hsr5"] == H5.Hsr5Config().record() and wire.GT_STAGE_REQUIRED not in sp.metadata
    assert wire.GT_STAGE_REQUIRED not in _policy().metadata
    # serve_safe: parse + config + meta
    a = _parse(NOUN + ["--hsr5"] + H6F)
    assert serve_safe.hsr5_config_from_args(a) == H5.Hsr5Config()


def test_pi05_infer_flag_is_default_off():
    import re

    src = (REPO / "integration" / "rby1" / "bringup" / "pi05_infer.py").read_text()
    assert re.search(r'add_argument\("--safe-gt-stage", action="store_true"', src)
    assert "**({\"gt_stage\": gt_stage_source_of(" in src and "if args.safe_gt_stage else {})" in src
    assert "--safe-gt-stage only does something together with --safe-remote" in src
