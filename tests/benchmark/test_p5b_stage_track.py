"""P5b 제약 단계 추종 (2026-11-10) — ``--stage-track ASSET``: the h_10 + GRIP stage probe drives the grasp latch. CPU only.

What this file pins:

(a) Off is unchanged: parser default None, `SafePolicy.stage_track` class default None, the RB-Y1 server refuses the flag, no earlier
    version's argv carries it, `AttentionPolicy` without ``stage_hidden`` never runs the extra pass, and the call record has no
    ``stage_track`` key.
(b) The offline definitions the asset was trained on (`experiments/segmentation/stage_track/p3_extract_hidden.py` pooling,
    `p4_feature_compare.load` GRIP order, the last executed gripper row incl. HOLD chunks) and the shipped asset's shapes / classes.
(c) The override on the real ``SafePolicy._advance_grasp`` (the T22 harness, as SL14): S3 attaches while the evidence is still CLOSING;
    S1 while CLOSING releases the closing; S3 while HELD voids an evidence release; not S3 while HELD detaches; no hidden feature ->
    no override (the evidence alone); the shadow latch follows the evidence only (its state is the feature, never forced).
(d) The P6 versions ``-pe123458-ol`` / ``-pe123458-st1``.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from tests.trajopt.test_grasp_evidence import _fb, _finger_origin, _lock, _manipulated, _policy
from AG3S.trajopt import stage_track as st

B = ["--config", "x", "--checkpoint", "y", "--esdf-margin", "0", "--model-xml", "m.xml"]


# ---------------------------------------------------------------------------------------------------- (a) off is unchanged
def test_off_by_default_parser_class_and_rby1_refusal():
    from AG3S.trajopt import serve_safe as ss
    from AG3S.trajopt.safe_policy import SafePolicy

    ap = ss.build_parser()
    assert ap.parse_args(B).stage_track is None
    assert ap.parse_args(B + ["--stage-track", "default"]).stage_track == "default"
    assert SafePolicy.stage_track is None
    with pytest.raises(SystemExit):
        ss.reject_panda_only_flags(ap, ap.parse_args(B + ["--stage-track", "default"]))
    ss.reject_panda_only_flags(ap, ap.parse_args(B))


def test_no_earlier_version_carries_the_flag():
    from AG3S.adapters.panda import profile as pf

    for v, argv in pf.AG3S_VERSIONS.items():
        if v not in pf.STAGE_TRACK_VERSIONS:
            assert "--stage-track" not in argv, v


def test_attention_policy_without_the_flag_never_runs_the_pass():
    from AG3S.trajopt.attention_policy import AttentionPolicy

    class _P:
        def infer(self, obs, **kw):
            return {"actions": np.zeros((2, 7))}

    for on in (False, True):
        ap = AttentionPolicy(_P(), None, sampler=object(), subtask_probe=None, **({"stage_hidden": True} if on else {}))
        ap._attention_and_feature = lambda obs: ({}, None)                                 # noqa: SLF001
        calls = []
        ap._add_stage_hidden = lambda obs, result: calls.append(1)                       # noqa: SLF001
        out = ap.infer({"prompt": "x"})
        assert calls == ([1] if on else []) and st.STAGE_HIDDEN_KEY not in out


def test_call_record_has_no_stage_key_when_off():
    sp, q = _sp()
    rec = _run(sp, q, [(SEQ[0], None)])[0]
    assert "stage_track" not in rec


# ---------------------------------------------------------------------------------------------------- (b) definitions and asset
def test_grip_feature_order_matches_p4():
    g = st.grip_features(gap_mm=40.0, prev_gap_mm=50.0, measured_opening=0.5, latch_state="closing", last_command=0.9)
    assert g.tolist() == [40.0, 50.0, -10.0, 0.5, 0, 0, 1, 0, 0, 0.9]
    first = st.grip_features(gap_mm=40.0, prev_gap_mm=None, measured_opening=None, latch_state=None, last_command=None)
    assert first.tolist() == [40.0, 40.0, 0.0, 1.0, 0, 0, 0, 0, 0, -1.0]
    assert len(g) == st.GRIP_DIM


def test_last_executed_command_rows():
    a = np.zeros((10, 7))
    a[:, 6] = np.arange(10)
    assert st.last_executed_command(a, 3, 5) == 2.0          # executed rows 0..2
    assert st.last_executed_command(a, 0, 5) == 4.0          # HOLD: the client ran the first n_steps rows
    assert st.last_executed_command(a, 0, None) == 9.0
    assert st.last_executed_command(None, 3, 5) is None
    assert st.gap_mm_of_state(np.r_[np.zeros(6), 0.03, -0.03]) == pytest.approx(60.0)
    assert st.gap_mm_of_state(np.zeros(4)) is None and st.gap_mm_of_state(None) is None


def test_hidden_pooling_matches_p3():
    rng = np.random.default_rng(1)
    h = rng.standard_normal((768 + 40, st.HIDDEN_DIM)).astype(np.float32)
    mask = np.r_[np.ones(768 + 17, bool), np.zeros(23, bool)]
    ref = np.concatenate([h[:256].mean(0), h[256:512].mean(0), h[768:785].mean(0)]).astype(np.float16).astype(np.float32)
    np.testing.assert_allclose(st.stage_hidden_feature(np, h, mask), ref, rtol=0, atol=2e-3)
    assert st.stage_hidden_feature(np, h, mask).shape == (st.FEATURE_DIM,)


def test_shipped_asset_loads():
    p = st.StageTrackProbe.load(st.default_asset_path())
    assert p.classes == st.STAGES and p.pca_components.shape == (128, st.FEATURE_DIM)
    rec = p.record()
    assert rec["hidden_layer"] == 10 and rec["pca_dim"] == 128 and rec["grip_dim"] == 10 and len(rec["md5"]) == 32
    out = p.predict(np.zeros(st.FEATURE_DIM), st.grip_features(gap_mm=78.0, prev_gap_mm=78.0, measured_opening=0.98,
                                                              latch_state="searching", last_command=-1.0))
    assert out["stage"] in st.STAGES and abs(sum(out["p"].values()) - 1.0) < 1e-4


# ---------------------------------------------------------------------------------------------------- (c) the override
class _Probe:
    name = "scripted"

    def __init__(self, stages):
        self.stages = list(stages)
        self.grips = []

    def predict(self, hidden, grip):
        s = self.stages.pop(0)
        self.grips.append(np.asarray(grip))
        return {"stage": s, "index": st.STAGES.index(s), "p": {x: float(x == s) for x in st.STAGES}}


STATE = {"observation/state": np.r_[np.zeros(6), 0.02, -0.02]}       # gap 40 mm
HIDDEN = {"feature": np.zeros(st.FEATURE_DIM, np.float32), "ms": 1.0}
#: the T22 sequence (SL14's): open, closing (CLOSING), same opening again (attach by evidence), hold chunk, two open requests
SEQ = [_fb(1, [1.0] * 8, 1.0), _fb(2, [0.0] * 8, 0.715), _fb(3, [0.0] * 8, 0.715),
       _fb(4, [0.72] * 8, 0.72, executed=[False] * 8), _fb(5, [1.0] * 8, 1.0), _fb(6, [1.0] * 8, 1.0)]


def _sp(distance=0.05, stages=None):
    sp = _policy()
    q = np.zeros(len(sp.constraint_robot_model.joint_names))
    sp.ag3s._manipulated = _manipulated(_finger_origin(sp, q) + np.array([distance, 0.0, 0.0]))  # noqa: SLF001
    _lock(sp)
    if stages is not None:
        sp.stage_track = _Probe(stages)
        sp._stage_track_reset()                                                    # noqa: SLF001
        for _ in range(3):
            sp._stage_shadow().update(label="obj0", score=0.9, runner_up=0.1, gripper=1.0)  # noqa: SLF001
    return sp, q


def _run(sp, q, items, hidden=HIDDEN):
    out = []
    obs = [types.SimpleNamespace(timestamp=1.0, robot_state=q)]
    for fb, _ in items:
        sp._pending = {"scene": {}, "exec_feedback": fb}                        # noqa: SLF001
        if sp.stage_track is not None:
            sp._pending.update(stage_hidden=hidden, policy_obs=STATE)            # noqa: SLF001
        sp._advance_grasp(obs, ["left"])                                         # noqa: SLF001
        sp._grasp_constraint_version = sp._grasp_state_version                   # noqa: SLF001
        sp._finish_grasp_record(types.SimpleNamespace(attached=sp.ag3s.attached))  # noqa: SLF001
        out.append(dict(sp._last_grasp_record))                                  # noqa: SLF001
    return out


def test_s3_attaches_while_the_evidence_is_still_closing():
    sp, q = _sp(stages=["approach", "carry_place"])
    recs = _run(sp, q, [(SEQ[0], None), (SEQ[1], None)])
    assert recs[0]["stage_track"]["held"] is False and recs[0]["stage_track"]["overrode"] is None
    r = recs[1]["stage_track"]
    assert r["evidence_state"] == "closing" and r["overrode"] == "attach" and r["held"] is True
    assert recs[1]["state"] == "held" and sp.ag3s.attached is not None
    assert r["shadow_state"] == "closing"                     # the shadow follows the evidence only


def test_s1_while_closing_releases_the_closing():
    moving = _fb(3, [0.0] * 8, 0.45)                         # the fingers still move: the evidence stays CLOSING
    sp, q = _sp(stages=["approach", "grasp", "approach"])
    recs = _run(sp, q, [(SEQ[0], None), (SEQ[1], None), (moving, None)])
    assert recs[1]["state"] == "closing" and "closing_released" not in recs[1]["stage_track"]
    assert recs[2]["stage_track"]["evidence_state"] == "closing"
    assert recs[2]["stage_track"].get("closing_released") is True and recs[2]["state"] == "searching"
    ref, q2 = _sp()                                           # without the flag the same requests stay CLOSING
    assert _run(ref, q2, [(SEQ[0], None), (SEQ[1], None), (moving, None)])[2]["state"] == "closing"


def test_s3_while_held_voids_an_evidence_release_and_not_s3_detaches():
    sp, q = _sp(stages=["approach"] + ["carry_place"] * 5 + ["approach"])
    recs = _run(sp, q, [(f, None) for f in SEQ] + [(_fb(7, [1.0] * 8, 1.0), None)])
    assert all(r["state"] == "held" for r in recs[1:6])       # two open requests (5, 6) do not release while S3
    assert recs[5]["stage_track"].get("suppressed_detach") is True
    assert recs[6]["stage_track"]["overrode"] == "detach" and recs[6]["state"] == "searching" and sp.ag3s.attached is None
    ref, q2 = _sp()                                           # without the flag the evidence releases on request 6
    assert _run(ref, q2, [(f, None) for f in SEQ])[5]["state"] != "held"


def test_no_hidden_feature_leaves_the_evidence_alone():
    sp, q = _sp(stages=[])
    recs = _run(sp, q, [(f, None) for f in SEQ[:3]], hidden=None)
    assert all(r["stage_track"]["held"] is None and "error" in r["stage_track"] for r in recs)
    ref, q2 = _sp()
    want = _run(ref, q2, [(f, None) for f in SEQ[:3]])
    assert [r["state"] for r in recs] == [r["state"] for r in want]


def test_grip_features_read_the_request_state_and_the_last_executed_row():
    sp, q = _sp(stages=["approach", "approach"])
    prev = np.zeros((10, 7))
    prev[:, 6] = np.linspace(-1, 1, 10)
    _run(sp, q, [(SEQ[0], None)])
    sp._stage_prev_actions = prev                                                  # noqa: SLF001
    _run(sp, q, [(SEQ[1], None)])
    g0, g1 = sp.stage_track.grips
    assert g0[0] == pytest.approx(40.0) and g0[1] == pytest.approx(40.0) and g0[9] == -1.0
    n_exec = SEQ[1]["n_exec"] if "n_exec" in SEQ[1] else 8
    assert g1[9] == pytest.approx(prev[min(n_exec, 10) - 1, 6])


# ---------------------------------------------------------------------------------------------------- (d) versions
def test_p6_versions():
    from AG3S.adapters.panda import profile as pf

    base = pf.PE_VERSIONS[6]
    assert base.endswith("-pe123458")
    ol, st1 = pf.STAGE_TRACK_VERSIONS
    assert pf.AG3S_VERSIONS[ol] == tuple(pf.AG3S_VERSIONS[base]) + ("--oracle-latch",) and pf.needs_oracle_grasp(ol)
    assert pf.AG3S_VERSIONS[st1] == tuple(pf.AG3S_VERSIONS[base]) + ("--stage-track", "default") and not pf.needs_oracle_grasp(st1)
    assert pf.is_gtto_version(ol) and pf.is_gtto_version(st1)
