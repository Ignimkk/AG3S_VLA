"""T82 — the 5-stage HSR (T77 hidden layer 4) in serving, behind `serve_safe --hsr5` (default off).

Task: `AG3S/ag3s/docs/handoff/T82.task.md` §1–§3. Report: `T82.impl.md`.

What this file fixes:

0. the asset: shipped npz = T77 `h4_img3_last.npz` (md5), classes / dim checked, `SubtaskProbe.proba` = offline LogReg.
1. the feature: `hsr5.hsr5_feature` = T77's `pi05_hidden.summary_rows` rows (0, 1, 2, 4) flattened = `t77_probe.load_X`;
   jax = numpy; on a dummy pi0.5 the serving pass (`JaxPrefixBackend.hsr5_pass`, return_hidden on for that pass only)
   gives the offline `SummaryExtractor` rows and leaves kv · attention row bit-identical to the flag-off pass.
2. the mapping table (stage → rule input) for every rule of task §2.
3. the closing release (empty close → reopen sequences, stage / HOLD / no-label resets, off = before).
4. `AttentionPolicy`: `result["hsr5"]` only with the probe; feature missing / probe failing → no key.
5. `SafePolicy`: off = no key · no kwarg · the 3-stage gate input; on = the stage drives each rule, record sources.
6. `serve_safe --hsr5`: parse, refusals, record meta only when on.
"""
from __future__ import annotations

import hashlib
import types

import numpy as np
import pytest

from AG3S.trajopt import hsr5 as H5
from AG3S.trajopt import hsr6 as H6
from AG3S.trajopt.grasp_latch import GraspPhase
from tests.trajopt.test_grasp_evidence import _finger_origin, _latched, _lock, _manipulated, _policy, _signal


def _rec(stage):
    """A `result["hsr5"]` record whose argmax is `stage`."""
    p = {c: (0.9 if c == stage else 0.025) for c in H5.STAGES5}
    return {"p": p, "argmax": stage, "probe": H5.PROBE_NAME, "ms": 1.0}


# ------------------------------------------------------------------------------------------------ 0. asset
def test_shipped_asset_is_the_t77_probe():
    path = H5.default_probe_path()
    assert hashlib.md5(path.read_bytes()).hexdigest() == H5.PROBE_MD5 == "985be91854b46ee354449025b2ae1885"
    probe = H5.load_hsr5_probe()
    assert probe.classes == H5.STAGES5 and probe.dim == H5.FEATURE_DIM == 8192 and probe.name == H5.PROBE_NAME
    with np.load(path) as z:
        assert str(z["feature"]) == "h4_img3+last" and float(z["C"]) == 0.03


def test_serving_proba_equals_the_offline_logreg():
    from AG3S.ag3s.experiments.studies.t50_core import LogReg

    probe = H5.load_hsr5_probe()
    with np.load(H5.default_probe_path()) as z:
        lr = LogReg(mean=z["mean"], scale=z["scale"], coef=z["coef"], intercept=z["intercept"], C=float(z["C"]),
                    n_iter=0, converged=True)
    X = (np.random.default_rng(82).normal(size=(64, 8192)) * probe.scale + probe.mean).astype(np.float32)
    assert np.array_equal(probe.proba(X), lr.proba(X))
    for x in X[:10]:
        r = probe.predict(x)
        assert r["argmax"] == H5.STAGES5[int(np.argmax(lr.proba(x[None])[0]))] == H5.stage_of(r)


def test_wrong_classes_or_dim_are_refused(tmp_path):
    with np.load(H5.default_probe_path()) as z:
        d = {k: z[k] for k in z.files}
    bad = dict(d, classes=np.array(list(reversed(H5.STAGES5))))
    np.savez(tmp_path / "bad.npz", **bad)
    with pytest.raises(ValueError, match="classes"):
        H5.load_hsr5_probe(tmp_path / "bad.npz")
    with pytest.raises(ValueError, match="classes"):
        H5.load_hsr5_probe(H6.default_probe_path())        # the 6-phase kv_L4 probe
    small = dict(d, mean=d["mean"][:512], scale=d["scale"][:512], coef=d["coef"][:, :512])
    np.savez(tmp_path / "small.npz", **small)
    with pytest.raises(ValueError, match="dim"):
        H5.load_hsr5_probe(tmp_path / "small.npz")


def test_stage_of_handles_missing_and_bad_records():
    assert H5.stage_of(None) is None and H5.stage_of({}) is None and H5.stage_of("carry") is None
    assert H5.stage_of({"p": {c: float("nan") for c in H5.STAGES5}}) is None
    assert H5.stage_of({"argmax": "place"}) == "place" and H5.stage_of({"argmax": "transport"}) is None
    assert H5.stage_of(_rec("retreat")) == "retreat"
    tie = {"p": {c: 0.2 for c in H5.STAGES5}}
    assert H5.stage_of(tie) == "approach"                                  # numpy argmax: the earlier stage


# ------------------------------------------------------------------------------------------------ 1. feature
def _fake_hidden(T=968, D=32, n_text=37, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(T, D)).astype(np.float32)
    mask = np.zeros(T, bool)
    mask[:768 + n_text] = True
    return h, mask


def test_feature_is_t77_summary_rows_0_1_2_4():
    from AG3S.ag3s.experiments.sources.pi05_hidden import summary_rows

    h, mask = _fake_hidden()
    f = H5.hsr5_feature(np, h, mask)
    rows, last = summary_rows(np, h[None], mask)
    assert int(last) == 804 and f.shape == (4 * 32,)
    # t77_probe.load_X: A[:, POOLS["img3+last"]].reshape(n, -1) with A = [n, 5, D]
    A = rows[None, 0]
    from AG3S.ag3s.experiments.studies.t77_probe import POOLS

    assert POOLS["img3+last"] == H5.FEATURE_ROWS
    assert np.array_equal(f, np.asarray(A[:, list(POOLS["img3+last"])]).reshape(1, -1)[0])
    # [cam_high mean | left mean | right mean | last valid token]
    np.testing.assert_allclose(f[:32], h[:256].mean(0), rtol=1e-6)
    np.testing.assert_allclose(f[64:96], h[512:768].mean(0), rtol=1e-6)
    assert np.array_equal(f[96:], h[804])


def test_feature_jax_matches_numpy():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    h, mask = _fake_hidden(n_text=1)
    fj = jax.jit(lambda a, m: H5.hsr5_feature(jnp, a, m))(h, mask)
    np.testing.assert_allclose(np.asarray(fj), H5.hsr5_feature(np, h, mask), rtol=1e-5, atol=1e-6)


@pytest.fixture(scope="module")
def dummy_backend():
    """A dummy-size pi0.5 with return_attn_probs (the serving attention copy) and the same params with
    return_hidden_states (the T77 offline extractor)."""
    pytest.importorskip("flax")
    pytest.importorskip("openpi")
    import flax.nnx as nnx
    import jax

    from AG3S.ag3s.experiments.sources import pi05_hidden as ph
    from AG3S.ag3s.experiments.sources.pi05_attention import AttentionSampler
    from AG3S.trajopt.attention_source import JaxPrefixBackend
    from openpi.models import model as _model
    from openpi.models import pi0_config

    cfg_a = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy",
                                 return_attn_probs=True)
    cfg_h = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy",
                                 return_hidden_states=True)
    model_a = cfg_a.create(jax.random.key(0))
    model_h = cfg_h.create(jax.random.key(1))
    nnx.update(model_h, nnx.state(model_a, nnx.Param))                  # the same weights
    sampler = AttentionSampler(model_a, num_steps=2)
    backend = JaxPrefixBackend(sampler)
    raw = cfg_a.fake_obs()
    obs = _model.preprocess_observation(None, raw, train=False)    # serving: preprocessed once, before the pass
    return types.SimpleNamespace(sampler=sampler, backend=backend, obs=obs, raw=raw,
                                 offline=ph.SummaryExtractor(model_h, kv_layer=2))


def test_serving_pass_gives_the_offline_feature_and_leaves_kv_and_attention(dummy_backend):
    import jax.numpy as jnp

    d = dummy_backend
    rows = np.array([770, 772], np.int32)
    kv_on, mask_on, att_on, feat = d.backend.hsr5_pass(d.obs, rows, layer=3, head=0)
    kv_off, mask_off, att_off = d.backend._jit_noun(d.sampler._state, d.obs, jnp.asarray(rows),  # noqa: SLF001
                                                    layer=3, head=0)
    for a, b in zip(jax_leaves(kv_on), jax_leaves(kv_off)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert np.array_equal(np.asarray(mask_on), np.asarray(mask_off))
    assert np.array_equal(np.asarray(att_on), np.asarray(att_off))
    # the offline T77 extractor (return_hidden_states model, summary pass) on the same observation
    off_rows, _, _, _, _ = d.offline._summary(d.offline._state, d.raw)   # noqa: SLF001 — preprocesses inside
    want = np.asarray(off_rows)[H5.HIDDEN_LAYER][list(H5.FEATURE_ROWS)].reshape(-1)
    assert np.asarray(feat).shape == want.shape == (4 * 64,)
    np.testing.assert_allclose(np.asarray(feat), want, rtol=1e-6, atol=1e-6)
    # noun not found: same kv as the flag-off kv_L4 pass, no attention row
    kv_n, mask_n, att_n, feat_n = d.backend.hsr5_pass(d.obs, None, layer=3, head=0)
    kv_p, mask_p = d.sampler._jit_prefix(d.sampler._state, d.obs)        # noqa: SLF001
    assert att_n is None
    for a, b in zip(jax_leaves(kv_n), jax_leaves(kv_p)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    np.testing.assert_allclose(np.asarray(feat_n), want, rtol=1e-6, atol=1e-6)


def jax_leaves(tree):
    import jax

    return jax.tree.leaves(tree)


def test_attention_model_flags_are_untouched_by_the_hidden_pass(dummy_backend):
    llm = dummy_backend.sampler._nnx.merge(dummy_backend.sampler._graphdef,  # noqa: SLF001
                                           dummy_backend.sampler._state).PaliGemma.llm  # noqa: SLF001
    assert llm.module.return_probs is True and llm.module.return_hidden is False


# ------------------------------------------------------------------------------------------------ 2. mapping table
@pytest.mark.parametrize("stage,sub3,phase6,after", [
    ("approach", "pick", "approach", False), ("grasp", "pick", "grasp", False),
    ("carry", "place", "transport", True), ("place", "place", "placement", True),
    ("retreat", "home", "retreat", True), (None, None, None, None)])
def test_stage_to_rule_inputs(stage, sub3, phase6, after):
    assert H5.subtask3_of(stage) == sub3 and H5.rule_phase6(stage) == phase6 and H5.after_grasp_on(stage) is after
    p = H5.subtask3_probabilities(stage)
    assert p is None if stage is None else (p == {c: float(c == sub3) for c in ("pick", "place", "home")})


@pytest.mark.parametrize("stage", list(H5.STAGES5) + [None])
def test_task_table_through_the_hsr6_rules(stage):
    """T82 §2 rows ①–④ evaluated with the same HSR6 rule code on `rule_phase6(stage)`."""
    ph = H5.rule_phase6(stage)
    checks = {"closed": True, "blocked": True}
    assert H6.attach_assist_ok(ph, checks) is (stage in ("carry", "place"))                 # ① carry · place
    assert H6.held_allowance(ph, latch_held=True, blocked=True) is (stage in ("grasp", "carry", "place"))  # ②
    assert H6.withdraw_target_exemption(ph) is (stage == "retreat")                          # ③ retreat
    assert H6.objective_allowed(ph) is (stage == "grasp")                                    # ④ grasp
    from AG3S.ag3s.stages.subtask_probe import GATED_LABELS

    gated = H5.subtask3_of(stage) in GATED_LABELS                                            # new-object gate
    assert gated is (stage in ("carry", "place", "retreat"))


# ------------------------------------------------------------------------------------------------ 3. closing release
def _empty_close_then_reopen(latch, stages, reopen=((1.0, 0.60), (1.0, 0.62), (1.0, 0.63)), kw=True):
    """open → empty close (cmd 0.45, fingers follow to 0.45 = not blocked) → the policy reopens the command (1.0) while
    the measured opening stays below 0.85 (the existing `reopened` rule needs both ≥ 0.85)."""
    seq = [((1.0,), 1.0), ((0.45,), 0.70), ((0.45,), 0.46), ((0.45,), 0.45)] + [((c,), m) for c, m in reopen]
    out = []
    for i, ((cmd,), meas) in enumerate(seq):
        extra = {"closing_release": {"stage": stages[i], "frames": 2}} if kw else {}
        out.append(latch.observe_grasp(_signal(i + 1, [cmd] * 8, meas), reach_m=0.05, manipulated_id=0,
                                       manipulated_state="visible", **extra))
    return out


def test_release_after_two_reopened_requests():
    latch = _latched()
    evs = _empty_close_then_reopen(latch, ["approach"] * 7)
    assert [e.phase for e in evs] == [GraspPhase.LATCHED] + [GraspPhase.CLOSING] * 4 + [GraspPhase.LATCHED] * 2
    cr = [e.evidence.get("closing_release") for e in evs]
    assert cr[0] is None                                                     # not closing yet
    assert [c["streak"] for c in cr[1:5]] == [0, 0, 0, 1] and cr[4]["released"] is False
    assert cr[5] == {"stage": "approach", "command": 1.0, "threshold": 0.85, "streak": 2, "frames": 2,
                     "released": True}
    assert not any(e.attach or e.detach for e in evs) and "HSR5" in evs[5].note
    assert latch._reopen_streak == 0                                         # noqa: SLF001 — reset on leaving


@pytest.mark.parametrize("stages,released", [
    (["approach", "grasp", "grasp", "grasp", "grasp", "retreat", "x"], True),     # any non-carry/place stage
    (["approach"] * 4 + ["carry", "carry", "carry"], False),                      # carry keeps closing
    (["approach"] * 4 + ["place", "approach", "place"], False),                   # place in between: never 2
    (["approach"] * 4 + [None, "approach", None], False),                         # no label: the streak restarts
    (["approach"] * 4 + ["approach", None, "approach"], False)])
def test_release_needs_two_consecutive_non_carry_place(stages, released):
    stages = [None if s == "x" else s for s in stages]
    evs = _empty_close_then_reopen(_latched(), stages)
    assert any(e.phase is GraspPhase.LATCHED and e.evidence.get("closing_release", {}).get("released")
               for e in evs) is released


def test_a_hold_chunk_restarts_the_streak():
    latch = _latched()
    seq = [([1.0] * 8, 1.0, True), ([0.45] * 8, 0.6, True), ([0.45] * 8, 0.46, True), ([1.0] * 8, 0.6, True),
           ([1.0] * 8, 0.6, False), ([1.0] * 8, 0.62, True), ([1.0] * 8, 0.63, True)]
    evs = []
    for i, (cmd, meas, ex) in enumerate(seq):
        evs.append(latch.observe_grasp(_signal(i + 1, cmd, meas, executed=None if ex else [False] * 8),
                                       reach_m=0.05, manipulated_id=0, manipulated_state="visible",
                                       closing_release={"stage": "approach", "frames": 2}))
    streaks = [(e.evidence.get("closing_release") or {}).get("streak") for e in evs]
    assert streaks == [None, 0, 0, 1, 0, 1, 2]
    assert evs[4].evidence["closing_release"]["reset"] == "no executed command"
    assert evs[6].phase is GraspPhase.LATCHED


def test_the_211_pattern_command_never_reopens_keeps_closing():
    """T71 R0 회귀 211: after the empty close the executed command stays 0.27–0.46 to the end — no release."""
    latch = _latched()
    cmds = [1.0, 0.007, 0.56, 0.49, 0.41, 0.45, 0.45, 0.46, 0.45, 0.45, 0.46, 0.45, 0.44]
    evs = [latch.observe_grasp(_signal(i + 1, [c] * 8, max(c, 0.4)), reach_m=0.2, manipulated_id=0,
                               manipulated_state="occluded", closing_release={"stage": "retreat", "frames": 2})
           for i, c in enumerate(cmds)]
    assert all(e.phase is GraspPhase.CLOSING for e in evs[1:])
    assert all(e.evidence["closing_release"]["reset"] == "command below threshold" for e in evs[1:])


def test_off_is_the_events_as_before():
    on_stages = ["approach"] * 7
    a = _empty_close_then_reopen(_latched(), on_stages, kw=False)
    b = _empty_close_then_reopen(_latched(), on_stages, kw=False)
    assert [(e.phase, e.attach, e.detach, e.note) for e in a] == [(e.phase, e.attach, e.detach, e.note) for e in b]
    assert all("closing_release" not in (e.evidence or {}) for e in a)
    assert all(e.phase is GraspPhase.CLOSING for e in a[1:])                 # off: an empty close keeps CLOSING


def test_existing_reopen_rule_still_fires_first():
    latch = _latched()
    evs = _empty_close_then_reopen(latch, ["carry"] * 7, reopen=((1.0, 0.95),))   # cmd and opening ≥ 0.85
    assert evs[4].phase is GraspPhase.LATCHED and "포기" in evs[4].note
    assert "closing_release" not in evs[4].evidence


# ------------------------------------------------------------------------------------------------ 4. AttentionPolicy
class _Backend:
    def __init__(self, feat5=None, err=None):
        self.calls = []
        self.feat5 = feat5
        self.err = err

    def run(self, inputs, **kw):
        self.calls.append(kw)
        out = {"cell": np.zeros(768, np.float32), "feature": None, "v": None, "path": "fake"}
        if kw.get("want_hsr5"):
            out["hsr5_feature"] = self.feat5
            out["hsr5_ms"] = 3.25
            if self.err:
                out["hsr5_error"] = self.err
        return out


def _ap(**kw):
    from AG3S.trajopt.attention_policy import AttentionPolicy

    class _P:
        def infer(self, obs, **_):
            return {"actions": np.zeros((4, 16))}

        def _input_transform(self, obs):
            return obs

    return AttentionPolicy(_P(), None, sampler=object(), subtask_probe=None, prefix_feature=True, **kw)


def test_attention_policy_hsr5_only_with_the_probe():
    feat = (np.random.default_rng(5).normal(size=8192)).astype(np.float32)
    off_b, on_b = _Backend(), _Backend(feat5=feat)
    off = _ap(backend=off_b)
    on = _ap(backend=on_b, hsr5_probe=H5.load_hsr5_probe())
    r0, r1 = off.infer({}), on.infer({})
    assert "hsr5" not in r0 and "want_hsr5" not in off_b.calls[0]
    assert on_b.calls[0].pop("want_hsr5") is True and on_b.calls[0] == off_b.calls[0]   # otherwise the same call
    assert set(r1) - set(r0) == {"hsr5"}
    want = H5.load_hsr5_probe().predict(feat)
    assert r1["hsr5"] == {**want, "ms": 3.25} and r1["hsr5"]["probe"] == H5.PROBE_NAME
    assert off.hsr5_probe is None and on.hsr5_probe is not None


def test_attention_policy_hsr5_extended_path_without_other_flags():
    from AG3S.trajopt.attention_policy import AttentionPolicy

    ap = AttentionPolicy(types.SimpleNamespace(infer=lambda o: {}), None, sampler=object(), subtask_probe=None,
                         backend=_Backend(), hsr5_probe=H5.load_hsr5_probe())
    assert ap._extended is True                                              # noqa: SLF001
    plain = AttentionPolicy(types.SimpleNamespace(infer=lambda o: {}), None, sampler=object(), subtask_probe=None)
    assert plain._extended is False                                          # noqa: SLF001


def test_no_feature_or_a_failing_probe_leaves_the_key_out():
    r = _ap(backend=_Backend(feat5=None, err="action-query source"), hsr5_probe=H5.load_hsr5_probe()).infer({})
    assert "hsr5" not in r and "attention" in r

    class _Bad:
        def predict(self, x):
            raise RuntimeError("boom")

    r = _ap(backend=_Backend(feat5=np.zeros(8192, np.float32)), hsr5_probe=_Bad()).infer({})
    assert "hsr5" not in r and "attention" in r


# ------------------------------------------------------------------------------------------------ 5. SafePolicy
CFG = H5.Hsr5Config()


def _sp_start(sp, stage):
    sp._pending = {}
    if sp.hsr5 is not None:
        sp._hsr5_start(None if stage is None else _rec(stage))
    if sp.hsr6 is not None:
        sp._hsr6_start({"p": {c: (0.9 if c == "approach" else 0.02) for c in H6.PHASES6}, "argmax": "approach"})


def test_off_no_key_no_kwarg_and_the_3_stage_gate_input():
    sp = _policy()
    assert sp.hsr5 is None
    q = np.zeros(len(sp.constraint_robot_model.joint_names))
    sp.ag3s._manipulated = _manipulated(_finger_origin(sp, q) + np.array([0.05, 0.0, 0.0]))
    _lock(sp)
    from tests.trajopt.test_core_eval_hsr6 import _fbs
    from tests.trajopt.test_grasp_evidence import _grasp_frames

    recs = list(_grasp_frames(sp, q, _fbs()))
    assert [r["attach"] for r in recs] == [False, False, False, True]          # as test_core_eval_hsr6 (off)
    assert all("closing_release" not in r["evidence"] for r in recs)
    assert "hsr5" not in sp._exec_summary() and "hsr5" not in sp.metadata
    from AG3S.trajopt.safe_policy import _subtask_probabilities

    for sub in ({"p": {"pick": 0.1, "place": 0.7, "home": 0.2}, "argmax": "place"}, None):
        sp._pending = {"subtask": sub, "hsr5": _rec("approach")}              # a stray hsr5 record is not read
        assert sp._subtask_gate_input() == _subtask_probabilities(sub)


@pytest.mark.parametrize("stage,want", [("approach", {"pick": 1.0, "place": 0.0, "home": 0.0}),
                                        ("grasp", {"pick": 1.0, "place": 0.0, "home": 0.0}),
                                        ("carry", {"pick": 0.0, "place": 1.0, "home": 0.0}),
                                        ("place", {"pick": 0.0, "place": 1.0, "home": 0.0}),
                                        ("retreat", {"pick": 0.0, "place": 0.0, "home": 1.0}),
                                        (None, None)])
def test_subtask_gate_reads_the_stage(stage, want):
    """With --hsr5 AG3S gets the stage's one-hot (not the 3-stage probe); None without a stage (= no label)."""
    sp = _policy(hsr5=CFG)
    _sp_start(sp, stage)
    sp._pending["subtask"] = {"p": {"pick": 0.0, "place": 0.0, "home": 1.0}}   # the 3-stage probe says home
    assert sp._subtask_gate_input() == want
    assert sp._last_hsr5["source"]["subtask_gate"] == "hsr5"


def test_subtask_gate_input_is_what_scene_fn_hands_ag3s():
    import inspect

    from AG3S.trajopt import safe_policy

    src = inspect.getsource(safe_policy.SafePolicy._scene_fn)
    assert "set_subtask(self._subtask_gate_input())" in src and "_subtask_probabilities(" not in src


@pytest.mark.parametrize("stage", list(H5.STAGES5) + [None])
def test_hsr6_rules_read_the_mapped_stage(stage):
    sp = _policy(hsr6=H6.Hsr6Flags(attach_assist=True, held_gate=True, target_exempt=True, objective=True),
                 hsr5=CFG)
    _sp_start(sp, stage)                                                     # HSR6 itself says approach
    assert sp._hsr6_phase() == H5.rule_phase6(stage)
    rec = sp._exec_summary()["hsr6"]
    assert rec["phase"] == "approach" and rec["rule_phase"] == H5.rule_phase6(stage) and rec["phase_source"] == "hsr5"
    assert sp._hsr6_objective_ok() is (stage == "grasp")
    src = sp._exec_summary()["hsr5"]["source"]
    assert {k: v for k, v in src.items() if k.startswith("hsr6_")} == {
        "hsr6_attach_assist": "hsr5", "hsr6_held_gate": "hsr5", "hsr6_target_exempt": "hsr5", "hsr6_objective": "hsr5"}


def test_hsr6_without_hsr5_is_unchanged():
    sp = _policy(hsr6=H6.Hsr6Flags(objective=True))
    _sp_start(sp, None)
    assert sp._hsr6_phase() == "approach" and "rule_phase" not in sp._last_hsr6 and "hsr5" not in sp._exec_summary()


def _gate_policy(stage, latch_phase, hsr5=True):
    from AG3S.trajopt.config import TrajOptConfig
    from AG3S.trajopt.safe_policy import SafePolicy

    p = object.__new__(SafePolicy)
    p.to_config = TrajOptConfig.from_dict({"cost": {"w_deflection_rate": 20.0, "deflection_rate_after_grasp": True,
                                                    "w_continuity": 1.0, "continuity_after_grasp": True}})
    p._latch = types.SimpleNamespace(phase=latch_phase)
    p.hsr5 = CFG if hsr5 else None
    p._pending = {}
    p._last_hsr5 = {}
    if hsr5:
        p._hsr5_start(None if stage is None else _rec(stage))
    return p


@pytest.mark.parametrize("stage,latch_phase,on", [
    ("approach", GraspPhase.HELD, False), ("grasp", GraspPhase.HELD, False),        # the stage wins over the latch
    ("carry", GraspPhase.CLOSING, True), ("place", GraspPhase.LATCHED, True), ("retreat", GraspPhase.PLACED, True),
    (None, GraspPhase.CLOSING, False), (None, GraspPhase.HELD, True)])             # no stage: the latch rule
def test_after_grasp_terms_follow_the_stage(stage, latch_phase, on):
    p = _gate_policy(stage, latch_phase)
    rate, cont = p._deflection_rate_gate(), p._continuity_after_grasp_gate()
    assert (rate is None) is on and (cont is None) is on
    if not on and stage is not None:
        assert rate == f"hsr5 {stage} (cost.deflection_rate_after_grasp)"
        assert cont == f"before_grasp: hsr5 {stage} (cost.continuity_after_grasp)"
    src = p._last_hsr5["source"]
    assert src == {"deflection_rate": "latch" if stage is None else "hsr5",
                   "continuity": "latch" if stage is None else "hsr5"}


@pytest.mark.parametrize("latch_phase", list(GraspPhase))
def test_after_grasp_terms_off_path_is_the_latch_rule(latch_phase):
    from AG3S.trajopt.safe_policy import DEFLECTION_RATE_OFF_PHASES

    p = _gate_policy(None, latch_phase, hsr5=False)
    assert (p._deflection_rate_gate() is None) is (latch_phase not in DEFLECTION_RATE_OFF_PHASES)
    assert (p._continuity_after_grasp_gate() is None) is (latch_phase not in DEFLECTION_RATE_OFF_PHASES)


def test_safe_policy_closing_release_and_its_record():
    from tests.trajopt.test_grasp_evidence import _fb

    sp = _policy(hsr5=CFG)
    q = np.zeros(len(sp.constraint_robot_model.joint_names))
    sp.ag3s._manipulated = _manipulated(_finger_origin(sp, q) + np.array([0.05, 0.0, 0.0]))
    _lock(sp)
    obs = [types.SimpleNamespace(timestamp=1.0, robot_state=q)]
    states = []
    for i, (cmd, meas) in enumerate(((1.0, 1.0), (0.45, 0.7), (0.45, 0.46), (1.0, 0.6), (1.0, 0.62))):
        sp._pending = {"scene": {}, "exec_feedback": _fb(i + 1, [cmd] * 8, meas)}
        sp._hsr5_start(_rec("approach"))
        sp._advance_grasp(obs, ["left"])
        states.append((sp._last_grasp_record["state"], dict(sp._last_hsr5["source"])))
    assert [s for s, _ in states] == ["latched", "closing", "closing", "closing", "latched"]
    assert states[0][1] == {} and states[-1][1] == {"closing_release": "hsr5"}
    assert sp._last_grasp_record["evidence"]["closing_release"]["released"] is True
    assert sp._exec_summary()["hsr5"]["stage"] == "approach"
    assert sp.metadata["hsr5"] == CFG.record()


# ------------------------------------------------------------------------------------------------ 6. serve_safe
BASE = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]
NOUN = ["--attention-source", "noun_target:L10H7"]


def _parse(argv):
    from AG3S.trajopt import serve_safe

    ap = serve_safe.build_parser()
    a = ap.parse_args(BASE + list(argv))
    serve_safe.reject_hsr5_flags(ap, a)
    return a


def test_serve_off_adds_nothing():
    from AG3S.trajopt import serve_safe

    a = _parse(NOUN)
    assert a.hsr5 is None and serve_safe.hsr5_config_from_args(a) is None and serve_safe.hsr5_record_meta(a) == {}


def test_serve_on_default_and_path():
    from AG3S.trajopt import serve_safe

    a = _parse(NOUN + ["--hsr5"])
    assert serve_safe.hsr5_config_from_args(a) == H5.Hsr5Config()
    assert serve_safe.hsr5_record_meta(a) == {"hsr5": {"probe": "shipped:" + H5.PROBE_NAME, "release_frames": 2,
                                                       "decision": "raw argmax per request"}}
    b = _parse(["--hsr5", str(H5.default_probe_path())] + NOUN)
    assert serve_safe.hsr5_config_from_args(b).probe == str(H5.default_probe_path())


@pytest.mark.parametrize("argv", [
    ["--hsr5"],                                                   # no noun source: no hidden on that path
    ["--hsr5", "--attention-source", "action:last:L8H2"],
    ["--hsr5", "/nonexistent.npz"] + NOUN,
    ["--hsr5", "--no-safe"] + NOUN,
    ["--hsr5", "--no-attention"] + NOUN])
def test_serve_refuses(argv):
    with pytest.raises(SystemExit):
        _parse(argv)


def test_announce_loads_the_probe_and_logs_one_line(caplog):
    import logging

    from AG3S.trajopt import serve_safe

    a = _parse(NOUN + ["--hsr5"])
    t57a = {}
    with caplog.at_level(logging.WARNING):
        serve_safe.announce_hsr5(a, t57a)
    assert t57a["hsr5_probe"].name == H5.PROBE_NAME
    line = [r.getMessage() for r in caplog.records if "HSR5 (T82)" in r.getMessage()]
    assert len(line) == 1 and H5.PROBE_MD5 in line[0] and "closing release" in line[0]
