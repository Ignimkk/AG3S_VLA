"""IFG-S2SERVE — the feature-memory confidence alert on the server (`--confidence-gate on`, default off).

| # | property |
|---|---|
| 1 | constants = the LOCK (`arms.confidence_gate.abstain_if_below`); `dump_rounding` = the collector's float16 storage |
| 2 | `ConfidenceGate` on a synthetic graph: first frame has no memory (no score, no alert), the score is the offline formula (`ifg_memory.unit_rows` of the chosen region · chain memory), threshold boundary, no_feature / no_region keep the memory, the δ 0.1 switch updates the memory to the best candidate, `reset` empties it |
| 3 | **held-out parity** (storage): the server path (`surface_selection` on the replay depth + FK-sphere mask + noun map, `ConfidenceGate.step` on the dumped `prefix_L8`) gives the same region as the locked `ifg_memory.r6_choice` and the same score as `IFG-S2FK` `R11b/conf.jsonl` on HO_adjacent_ep333 frames 0–11 (|Δ| ≤ 1e-12), alert only on f011 |
| 4 | pipeline: default AG3S never reaches the gate; on an alert frame the manipulated geometry handed to the field / constraints is None (exemption withheld), the grounding target is kept; attached → not applied (`skip = attached`, counted); no alert → unchanged; reset clears memory and counts |
| 5 | serving: `AttentionPolicy(prefix_feature=True)` carries the dump-rounded tensor; `SafePolicy` pops it (never on the wire) and hands it to AG3S; flags · refusals · record meta only when on |
"""

from __future__ import annotations

import json
import os
import pathlib
import types

import numpy as np
import pytest
import scipy.sparse as sp

from AG3S.ag3s.runtime import confidence_gate as CG

STORE = pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", "/mnt/dev/AG3S_VLA-storage"))
DEV = STORE / "results/development"
LOCK = DEV / "20261015T181706+0900_30955b6ca8ef_d4b3bafb79ba/LOCK_heldout.json"
CONF = DEV / "20261019T025853+0900_f15e587290f6_93594d3dffc0/s2fk/run/rounds/R11b/conf.jsonl"
HO_DUMP = DEV / "20261016T000420+0900_30955b6ca8ef_2498252a873f/heldout/dump"
HO_PREP = DEV / "20261016T030741+0900_f15e587290f6_19ea09c7a8b8/heldout_eval"
needs_heldout = pytest.mark.skipif(not (CONF.is_file() and HO_DUMP.is_dir() and (HO_PREP / "replay").is_dir()),
                                   reason="IFG-S2FK held-out run (shared storage) not mounted")


# ============================================================================ 1 constants


@pytest.mark.skipif(not LOCK.is_file(), reason="LOCK_heldout.json not in this storage")
def test_threshold_is_the_lock_value():
    arm = json.loads(LOCK.read_text())["arms"]["confidence_gate"]
    assert CG.GATE_THRESHOLD == arm["abstain_if_below"]
    assert arm["score"] == "q_mem_chain:prefix_L8 (centered)"


def test_dump_rounding_is_the_collectors_float16_storage():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 40, (768, 16)).astype(np.float32)
    out = CG.dump_rounding(r)
    assert out.dtype == np.float32 and np.array_equal(out, r.astype(np.float16).astype(np.float32))
    big = r.copy()
    big[0, 0] = 1e6                       # overflows float16 → stored float32 (the collector's fallback)
    assert np.array_equal(CG.dump_rounding(big), big)
    bad = r.copy()
    bad[1, 1] = np.nan
    assert np.isnan(CG.dump_rounding(bad)[1, 1])


# ============================================================================ 2 synthetic graph


def _geo(n_regions=3, per=25, patch_of=None):
    """Regions = chains of `per` nodes (5 mm apart), none support; node i of region r reads patch `patch_of[r]`."""
    pos, edges, rows = [], [], []
    patch_of = patch_of or [10 * (r + 1) for r in range(n_regions)]
    for r in range(n_regions):
        base = len(pos)
        for i in range(per):
            pos.append([0.3 * r + 0.005 * i, 0.0, 0.0])
            rows.append(patch_of[r])
            if i:
                edges.append([base + i - 1, base + i])
    n = len(pos)
    px = sp.csr_matrix((np.full(n, 30.0), (np.arange(n), np.asarray(rows))), shape=(n, CG.N_IMAGE_TOKENS))
    return types.SimpleNamespace(pos=np.asarray(pos, float), edges=np.asarray(edges, np.int64),
                                 support=np.zeros(n, bool), patch_px=px)


def _feat(rng, d=32):
    return rng.normal(0, 1, (CG.N_IMAGE_TOKENS, d)).astype(np.float32)


def _mask(geo, region, per=25):
    m = np.zeros(len(geo.pos), bool)
    m[region * per:(region + 1) * per] = True
    return m


def test_gate_first_frame_score_formula_and_threshold_boundary():
    from AG3S.ag3s.experiments.studies import ifg_memory as M

    rng = np.random.default_rng(1)
    geo = _geo()
    f0, f1 = _feat(rng), _feat(rng)
    g = CG.ConfidenceGate()
    r0 = g.step(geo, _mask(geo, 0), f0)
    assert r0["gate_score"] is None and r0["gate_alert"] is False and r0["gate_why"] == "no_memory"
    U0, U1 = CG.unit_patch_features(f0), CG.unit_patch_features(f1)
    w0 = np.asarray(geo.patch_px.tocsr()[_mask(geo, 0)].sum(0)).ravel()
    assert np.allclose(g.memory, M.unit_rows(w0, U0))
    r1 = g.step(geo, _mask(geo, 0), f1)
    want = float(M.unit_rows(w0, U1) @ M.unit_rows(w0, U0))
    assert r1["gate_score"] == pytest.approx(want, abs=1e-15) and r1["gate_why"] is None
    assert r1["n_candidates"] == 3
    # boundary: alert iff score < threshold (strict)
    for thr, alert in ((want, False), (np.nextafter(want, 2.0), True), (want - 1e-9, False)):
        h = CG.ConfidenceGate(threshold=thr)
        h.step(geo, _mask(geo, 0), f0)
        assert h.step(geo, _mask(geo, 0), f1)["gate_alert"] is alert
    # the same frame twice scores 1 (centred unit features)
    s = CG.ConfidenceGate()
    s.step(geo, _mask(geo, 1), f0)
    assert s.step(geo, _mask(geo, 1), f0)["gate_score"] == pytest.approx(1.0)


def test_missing_feature_or_region_keeps_the_memory_and_reset_empties_it():
    rng = np.random.default_rng(2)
    geo = _geo()
    g = CG.ConfidenceGate()
    g.step(geo, _mask(geo, 0), _feat(rng))
    mem = g.memory.copy()
    for args, why in (((geo, _mask(geo, 0), None), "no_feature"), ((geo, None, _feat(rng)), "no_region"),
                      ((geo, np.zeros(len(geo.pos), bool), _feat(rng)), "no_region"),
                      ((None, None, _feat(rng)), "no_region")):
        r = g.step(*args)
        assert r["gate_why"] == why and r["gate_score"] is None and not r["gate_alert"]
        assert np.array_equal(g.memory, mem)
    assert g.step(geo, _mask(geo, 0), np.zeros((10, 4)))["gate_why"].startswith("no_feature")
    g.reset()
    assert g.memory is None and g.step(geo, _mask(geo, 0), _feat(rng))["gate_why"] == "no_memory"


def test_no_candidate_regions_scores_nothing_but_updates_the_memory():
    rng = np.random.default_rng(3)
    geo = _geo(per=10)            # every region below MIN_NODES (20) → no candidate
    g = CG.ConfidenceGate()
    g.step(geo, _mask(geo, 0, per=10), _feat(rng))
    m0 = g.memory.copy()
    r = g.step(geo, _mask(geo, 1, per=10), _feat(rng))
    assert r["gate_why"] == "no_candidates" and r["gate_score"] is None and r["n_candidates"] == 0
    assert not np.array_equal(g.memory, m0)       # offline: memory = this frame's choice


def test_switch_at_delta_0_1_moves_the_memory_to_the_best_candidate():
    from AG3S.ag3s.experiments.studies import ifg_memory as M

    rng = np.random.default_rng(4)
    geo = _geo()
    f = _feat(rng)
    U = CG.unit_patch_features(f)
    g = CG.ConfidenceGate()
    g.step(geo, _mask(geo, 2), f)                 # memory = region 2
    r = g.step(geo, _mask(geo, 0), f)             # chose region 0; region 2 matches the memory exactly
    assert r["switched"] is True and r["best_candidate_cos"] == pytest.approx(1.0)
    w2 = np.asarray(geo.patch_px.tocsr()[_mask(geo, 2)].sum(0)).ravel()
    assert np.allclose(g.memory, M.unit_rows(w2, U))
    r = g.step(geo, _mask(geo, 2), f)             # same region as memory: no switch, score 1
    assert r["switched"] is False and r["gate_score"] == pytest.approx(1.0)


# ============================================================================ 3 held-out parity


@needs_heldout
def test_server_gate_equals_the_offline_score_on_heldout_frames():
    from AG3S.ag3s.experiments.studies import ifg_graph as G
    from AG3S.ag3s.experiments.studies import ifg_heldout_eval as HE
    from AG3S.ag3s.experiments.studies import ifg_memory as M
    from AG3S.ag3s.experiments.studies import ifg_reader as R
    from AG3S.ag3s.runtime.attention_surface import CAMERA_ORDER, SurfaceView

    load_frame = R.load_frame
    feats = M.FEATS
    HE.install()
    try:
        conf = {json.loads(l)["frame_id"]: json.loads(l) for l in open(CONF)}
        refs, _ = HE.heldout_refs(HO_DUMP, HO_PREP)
        refs = sorted([r for r in refs if r.episode_id == "HO_adjacent_ep333"], key=lambda r: r.frame)[:12]
        assert len(refs) == 12
        cfg = G.GraphConfig()
        gate = CG.ConfidenceGate()
        alerts = []
        for ref in refs:
            fin = R.load_frame(ref, features=("prefix_L8",))
            with np.load(HO_PREP / "replay" / f"{ref.frame_id}.npz") as z:
                depth, K, T, rm = z["depth"], z["K"], z["T_base_cam"], z["robot_mask"]
            nmap = fin.maps["N"].reshape(3, 16, 16)
            views = [SurfaceView(CAMERA_ORDER[i], depth[i].astype(np.float64), K[i], T[i], rm[i], nmap[i])
                     for i in range(3)]
            geo, chosen, _ = CG.surface_selection(views)
            # region: the locked offline choice on the same inputs (real GT only fills evaluator arrays)
            with np.load(HO_PREP / "labels" / f"{ref.frame_id}.npz") as z:
                cm, cl = z["class_map"], [str(c) for c in z["classes"]]
            off = M.r6_choice(G.build_geometry(depth, K, T, rm, cm, cl, cfg), cfg, fin.maps["N"])
            assert off is not None and chosen is not None and np.array_equal(off["chosen"], chosen), ref.frame_id
            rec = gate.step(geo, chosen, fin.features["prefix_L8"])
            want = conf[ref.frame_id].get("q_mem_chain:prefix_L8")
            if want is None:
                assert rec["gate_score"] is None, ref.frame_id
            else:
                assert rec["gate_score"] == pytest.approx(want, abs=1e-12), ref.frame_id
            if rec["gate_alert"]:
                alerts.append(ref.frame_id)
        assert alerts == ["HO_adjacent_ep333_f011"]
    finally:
        R.load_frame = load_frame
        M.FEATS = feats


# ============================================================================ 4 pipeline


def _pipeline(gate=None):
    from tests.ag3s.test_degradation_reason import _pipeline as base

    ag3s = base()
    if gate is not None:
        ag3s.confidence_gate = gate
    return ag3s


def _obs3():
    from AG3S.ag3s.types import CameraID
    from tests.ag3s.test_degradation_reason import _observation

    return [_observation(c, t=10.0) for c in (CameraID.HEAD, CameraID.LEFT_WRIST, CameraID.RIGHT_WRIST)]


class _StubGate:
    def __init__(self, alert):
        self.alert = alert
        self.threshold = 0.5
        self.steps = []
        self.resets = 0

    def step(self, geo, chosen, feature):
        self.steps.append(feature)
        return {"gate_score": 0.1 if self.alert else 0.9, "gate_alert": self.alert, "gate_why": None,
                "threshold": self.threshold, "ms": {"total": 0.0}}

    def reset(self):
        self.resets += 1

    def record_meta(self):
        return {}


def _spy(monkeypatch, ag3s):
    """A real manipulated identity (geometry near the synthetic wall) passes the exclusion gate; record what the
    field (`_build_esdf`'s target) and the constraints (`manipulated_geometry`) receive — the real code runs."""
    from AG3S.ag3s.runtime import confidence_gate
    from tests.trajopt.test_grasp_evidence import _manipulated

    manip = _manipulated(np.array([0.0, 0.0, 0.9]))
    seen = {"build": [], "constraints": [], "geometry": manip.geometry}
    monkeypatch.setattr(confidence_gate, "surface_selection",
                        lambda views: (None, None, {"why": "stub", "views": [v.name for v in views]}))
    monkeypatch.setattr(type(ag3s._target_confirm), "manipulated", property(lambda self: manip))  # noqa: SLF001
    monkeypatch.setattr(ag3s, "_exclusion_gate", lambda m: (m.geometry, None))
    orig_build, orig_cons = ag3s._build_esdf, ag3s._constraints

    def build(cameras, target, **kw):
        seen["build"].append(target)
        return orig_build(cameras, target, **kw)

    def cons(**kw):
        seen["constraints"].append(kw.get("manipulated_geometry"))
        return orig_cons(**kw)

    monkeypatch.setattr(ag3s, "_build_esdf", build)
    monkeypatch.setattr(ag3s, "_constraints", cons)
    return seen


def test_default_pipeline_never_reaches_the_gate(monkeypatch):
    from AG3S.ag3s.runtime import confidence_gate

    def boom(*a, **k):
        raise AssertionError("gate code reached with the gate off")

    monkeypatch.setattr(confidence_gate, "surface_selection", boom)
    ag3s = _pipeline()
    assert ag3s.confidence_gate is None
    cs = ag3s.process_multi(_obs3(), phase="approach")
    assert "confidence_gate" not in cs.metrics and "confidence_gate" not in (cs.profile or {})


@pytest.mark.parametrize("alert", [True, False])
def test_alert_withholds_the_exemption_and_keeps_the_target(monkeypatch, alert):
    gate = _StubGate(alert)
    ag3s = _pipeline(gate)
    seen = _spy(monkeypatch, ag3s)
    feat = np.ones((768, 4), np.float32)
    ag3s.set_gate_feature(feat)
    cs = ag3s.process_multi(_obs3(), phase="approach")
    rec = cs.metrics["confidence_gate"]
    assert gate.steps and gate.steps[0] is not None and np.array_equal(gate.steps[0], feat)
    assert rec["gate_alert"] is alert and rec["applied"] is alert and rec["skip"] is None
    assert rec["surface"]["why"] == "stub"
    want = None if alert else seen["geometry"]
    assert seen["build"] == [want] and seen["constraints"] == [want]
    assert cs.metrics["manipulated"] is not None          # the grounding / identity side is untouched
    assert any("confidence gate" in n for n in cs.notes) is alert
    counts = ag3s.confidence_gate_record()["episode_counts"]
    assert counts == {"frames": 1, "alerts": int(alert), "applied": int(alert), "skipped_attached": 0,
                      "skipped_no_exemption": 0}
    # the feature was consumed: the next frame has none
    ag3s.process_multi(_obs3(), phase="approach")
    assert gate.steps[-1] is None
    ag3s.reset()
    assert gate.resets == 1 and ag3s.confidence_gate_record() == {"episode_counts": {
        "frames": 0, "alerts": 0, "applied": 0, "skipped_attached": 0, "skipped_no_exemption": 0}}


def test_alert_while_attached_is_recorded_not_applied(monkeypatch):
    from AG3S.ag3s.runtime import confidence_gate

    monkeypatch.setattr(confidence_gate, "surface_selection", lambda views: (None, None, {}))
    ag3s = _pipeline(_StubGate(True))
    ag3s._attached = object()                                                    # noqa: SLF001
    notes = []
    out = ag3s._confidence_gate_step([], [], {}, None, "GEOM", notes)            # noqa: SLF001
    assert out == "GEOM" and notes == []
    rec = ag3s.confidence_gate_record()
    assert rec["applied"] is False and rec["skip"] == "attached"
    assert rec["episode_counts"]["skipped_attached"] == 1 and rec["episode_counts"]["alerts"] == 1
    ag3s._attached = None                                                        # noqa: SLF001
    out = ag3s._confidence_gate_step([], [], {}, None, None, notes)              # noqa: SLF001
    assert out is None and ag3s.confidence_gate_record()["skip"] == "no_exemption"


def test_gate_views_use_the_sphere_mask_and_the_camera_attention():
    """The views the gate scores: CAMERA_ORDER, each camera's sphere mask (not the render union), its attention."""
    from AG3S.ag3s.runtime import confidence_gate

    captured = {}

    def sel(views):
        captured["views"] = views
        return None, None, {}

    ag3s = _pipeline(CG.ConfidenceGate())
    obs = _obs3()
    for i, o in enumerate(obs):
        object.__setattr__(o, "attention_map", np.full((16, 16), float(i)))
    cams = [types.SimpleNamespace(name=o.camera_id.value, depth=o.depth, camera_intrinsics=o.camera_intrinsics,
                                  T_base_cam=o.T_base_cam, robot_mask=np.ones(o.depth.shape, bool)) for o in obs]
    sphere = {c.name: np.zeros(c.depth.shape, bool) for c in cams}
    old = confidence_gate.surface_selection
    confidence_gate.surface_selection = sel
    try:
        ag3s._confidence_gate_step(obs, cams, sphere, None, None, [])            # noqa: SLF001
    finally:
        confidence_gate.surface_selection = old
    v = captured["views"]
    assert [x.name for x in v] == ["head", "left_wrist", "right_wrist"]
    assert all(not x.robot_mask.any() for x in v)
    assert [float(x.attention[0, 0]) for x in v] == [0.0, 1.0, 2.0]


# ============================================================================ 5 serving


def test_attention_policy_carries_the_dump_rounded_feature_only_with_the_flag():
    from AG3S.trajopt.attention_source import IFG_PREFIX_L8_KEY
    from tests.trajopt.test_t57a_attention_source import _ap, _FakeBackend, _obs

    resid = (np.arange(768 * 4, dtype=np.float32).reshape(768, 4) / 7.0)

    class _B(_FakeBackend):
        def run(self, inputs, *, source, rows, want_feature, v_layer, want_resid=False):
            out = super().run(inputs, source=source, rows=rows, want_feature=want_feature, v_layer=v_layer)
            self.calls[-1]["want_resid"] = want_resid
            if want_resid:
                out["resid_l8"] = resid
            return out

    b = _B()
    out = _ap(source="noun_target:L10H7", backend=b, prefix_feature=True).infer(_obs())
    assert b.calls[-1]["want_resid"] is True
    assert np.array_equal(out[IFG_PREFIX_L8_KEY], resid.astype(np.float16).astype(np.float32))
    b2 = _B()
    out = _ap(source="noun_target:L10H7", backend=b2).infer(_obs())
    assert b2.calls[-1]["want_resid"] is False and IFG_PREFIX_L8_KEY not in out
    # the flag alone takes the extended path with the default cell (no noun tokens needed)
    from AG3S.trajopt.attention_policy import AttentionPolicy
    from tests.trajopt.test_t57a_attention_source import _StubPolicy

    b3 = _B()
    ap = AttentionPolicy(_StubPolicy(), None, sampler=object(), backend=b3, subtask_probe=None, prefix_feature=True)
    out = ap.infer(_obs())
    assert b3.calls[-1]["source"] == "action:last:L8H2" and IFG_PREFIX_L8_KEY in out


def test_safe_policy_pops_the_feature_and_hands_it_to_ag3s():
    from AG3S.trajopt import wire
    from AG3S.trajopt.attention_source import IFG_PREFIX_L8_KEY
    from tests.trajopt.test_safe_policy import DOF, _safe_policy, _StubPolicy

    feat = np.ones((768, 4), np.float32)

    class _P(_StubPolicy):
        def infer(self, obs, **kw):
            out = super().infer(obs, **kw)
            out[IFG_PREFIX_L8_KEY] = feat
            return out

    sp = _safe_policy(_P())
    sp.ag3s.confidence_gate = CG.ConfidenceGate()
    out = sp.infer({"state": np.zeros(DOF), wire.PREFIX + "cameras": []})
    assert IFG_PREFIX_L8_KEY not in out and out["policy_extra"] == 7
    assert sp.ag3s._gate_feature_pending is feat                                 # noqa: SLF001
    assert sp.metadata["confidence_gate"]["threshold"] == CG.GATE_THRESHOLD
    # gate off: the key is still never sent; metadata has no gate key
    off = _safe_policy(_P())
    out = off.infer({"state": np.zeros(DOF), wire.PREFIX + "cameras": []})
    assert IFG_PREFIX_L8_KEY not in out and "confidence_gate" not in off.metadata
    assert off.ag3s._gate_feature_pending is None                                # noqa: SLF001


BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def _args(*extra):
    from AG3S.trajopt import serve_safe

    ap = serve_safe.build_parser()
    return ap, ap.parse_args(BASE_ARGV + list(extra))


def test_flags_default_off_and_record_meta_only_when_on():
    from AG3S.trajopt import serve_safe

    ap, args = _args()
    assert (args.confidence_gate, args.confidence_gate_threshold) == ("off", None)
    assert (args.held_3d_rule, args.held_3d_max_m) == ("off", None)
    assert serve_safe.ifg_s2serve_record_meta(args) == {} and serve_safe.confidence_gate_from_args(args) is None
    serve_safe.reject_bad_flag_combinations(ap, args)
    ap, args = _args("--attention-source", "noun_target:L10H7", "--confidence-gate", "on",
                     "--confidence-gate-threshold", "0.6")
    serve_safe.reject_ifg_s2serve_flags(ap, args)
    meta = serve_safe.ifg_s2serve_record_meta(args)
    assert meta["confidence_gate"]["threshold"] == 0.6 and "held_3d_rule" not in meta
    assert serve_safe.confidence_gate_from_args(args).threshold == 0.6
    ap, args = _args("--attention-source", "noun_target:L10H7", "--confidence-gate", "on")
    assert serve_safe.confidence_gate_from_args(args).threshold == CG.GATE_THRESHOLD


@pytest.mark.parametrize("extra", [
    ["--confidence-gate-threshold", "0.5"],
    ["--held-3d-max-m", "0.1"],
    ["--confidence-gate", "on"],
    ["--confidence-gate", "on", "--attention-source", "action:last:L8H2"],
    ["--confidence-gate", "on", "--attention-source", "noun_target:L10H7", "--no-attention"],
    ["--confidence-gate", "on", "--attention-source", "noun_target:L10H7", "--no-safe"],
    ["--held-3d-rule", "on", "--no-perception"],
    ["--held-3d-rule", "on", "--held-3d-max-m", "-1"],
    ["--held-3d-rule", "on", "--held-3d-max-m", "nan"],
    ["--confidence-gate", "on", "--attention-source", "noun_target:L10H7", "--confidence-gate-threshold", "inf"],
])
def test_bad_combinations_are_refused(extra):
    from AG3S.trajopt import serve_safe

    ap, args = _args(*extra)
    with pytest.raises(SystemExit):
        serve_safe.reject_ifg_s2serve_flags(ap, args)
