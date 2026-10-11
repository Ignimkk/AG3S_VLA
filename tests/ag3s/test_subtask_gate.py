"""SUBTASK-c — the subtask label decides whether the attention target may be carved.

User spec (2026-10-02): subtask **pick** → the attention target is carved out of the ESDF (as now);
**place** / **home** → it is not. The held object's carve → attach transition stays the grasp
latch's. In code that is one rule: with `clustering.subtask_gate` on and a confirmed `place` /
`home` label, `TargetConfirm` neither adopts a first target nor counts a challenger. The current
manipulated object is untouched, and the label can only *withhold* a carve.

What is tested (task §A4):
  1. gate off → the decisions are the ungated ones, whatever labels are fed;
  2. gate on, confirmed place / home → no `first`, no `switch`, the manipulated object is kept;
  3. gate on, pick / None → the ungated behaviour;
  4. debounce: N agreeing requests to change the label, a mixed run starts over, `reset()` clears;
  5. frozen (grasp in progress): the label changes nothing;
  6. the exported probe applied with numpy reproduces its own parity number on `feat/val`;
  7. (GPU, opt-in with `AG3S_GPU_TESTS=1`) the `AttentionSampler` kv_L4 equals SUBTASK-b's.
"""

from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import dataclasses
import hashlib
import json
import os
import pathlib

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, ClusteringConfig
from AG3S.ag3s.stages.subtask_probe import (
    CLASSES,
    SubtaskDebounce,
    SubtaskProbe,
    argmax_label,
    default_probe_path,
    kv_feature,
)
from AG3S.ag3s.stages.target_grounding import (
    SUBTASK_GATED,
    ClusterInfo,
    TargetConfirm,
    ground_target,
)
from AG3S.ag3s.types import AttentionPointCloud, GroundingStatus, PointCloud

OPENING = 0.0715
CFG = ClusteringConfig(min_points=10)
APPLE_C = np.array([0.50, 0.00, 0.80])
BANANA_C = np.array([0.50, -0.40, 0.80])
CRATE_C = np.array([0.50, 0.40, 0.80])
CRATE_HALF = (0.0915, 0.2105, 0.0745)

PICK = {"pick": 0.9, "place": 0.08, "home": 0.02}
PLACE = {"pick": 0.05, "place": 0.9, "home": 0.05}
HOME = {"pick": 0.01, "place": 0.09, "home": 0.9}
LABEL_P = {"pick": PICK, "place": PLACE, "home": HOME, None: None}


def _ball(centre, radius=0.031, n=300, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return np.asarray(centre) + radius * v / np.linalg.norm(v, axis=1, keepdims=True)


def _box(centre, half, n=900, seed=1):
    rng = np.random.default_rng(seed)
    half = np.asarray(half, np.float64)
    p = rng.uniform(-1.0, 1.0, size=(n, 3))
    axis = rng.integers(0, 3, size=n)
    p[np.arange(n), axis] = np.sign(p[np.arange(n), axis])
    return np.asarray(centre) + p * half


def _apc(*blobs):
    pts = np.concatenate([b for b, _ in blobs])
    att = np.concatenate([np.full(len(b), a, np.float32)
                          + np.linspace(0, 1e-3, len(b), dtype=np.float32) for b, a in blobs])
    return AttentionPointCloud(PointCloud(pts), att, att)


def _clusters(*items):
    clouds, infos, start = [], [], 0
    for i, (pts, score) in enumerate(items):
        pts = np.asarray(pts, np.float64)
        idx = np.arange(start, start + len(pts))
        start += len(pts)
        clouds.append(pts)
        infos.append(ClusterInfo(
            id=i, point_indices=idx, centroid=pts.mean(axis=0), mean_attention=score,
            max_attention=score, attention_mass=score, point_count=len(pts),
            spatial_compactness=1.0, distance_from_attention_peak=0.0, target_score=score))
    infos.sort(key=lambda c: -c.target_score)
    return infos, np.concatenate(clouds)


def _confirm(gate=False, **kw):
    kw.setdefault("max_opening", OPENING)
    kw.setdefault("overlap_distance", 0.005)
    return TargetConfirm(dataclasses.replace(CFG, subtask_gate=gate), **kw)


def _select(confirm, *items):
    clusters, pts = _clusters(*items)
    return confirm.select(clusters, points=pts)


def _confirmed(confirm, label):
    """Feed `label` until it is confirmed (and nothing else changes)."""
    for _ in range(confirm.subtask.frames):
        confirm.note_subtask(LABEL_P[label])
    assert confirm.subtask_label == label
    return confirm


def _key(d):
    """What a decision *decides* (the subtask record fields excluded)."""
    return (d.mode, d.rank, d.manipulated_id, d.streak, d.state, d.reason, d.frozen,
            None if d.cluster is None else tuple(np.round(d.cluster.centroid, 9)))


APPLE = (_ball(APPLE_C), 0.5)
BANANA = (_ball(BANANA_C, seed=3), 0.9)
CRATE = (_box(CRATE_C, CRATE_HALF, n=3000), 0.95)

#: A sequence that exercises first, keep, hold, switch, occluded and destination registration.
SCRIPT = ([(APPLE,)] * 2 + [(APPLE, BANANA)] * 4 + [(BANANA,)] * 2 + [(CRATE, APPLE)] * 4
          + [(APPLE, BANANA)] * 3)
LABELS = ["pick", "pick", "pick", "place", "place", "place", "place", None, "home", "home",
          "home", "home", "pick", "place", "place"]


def _run_script(confirm, labels=None, freeze_from=None):
    out = []
    for k, items in enumerate(SCRIPT):
        if labels is not None:
            confirm.note_subtask(LABEL_P[labels[k]])
        if freeze_from is not None:
            confirm.freeze(k >= freeze_from)
        out.append(_key(_select(confirm, *items)))
    return out


# ----------------------------------------------------------------- 1. gate off = unchanged


def test_gate_off_decisions_do_not_depend_on_the_labels():
    plain = _run_script(_confirm())
    for labels in (LABELS, ["place"] * len(SCRIPT), ["home"] * len(SCRIPT), [None] * len(SCRIPT)):
        c = _confirm()
        assert _run_script(c, labels) == plain
        assert c.last.subtask_blocked is None
    modes = [k[0] for k in plain]
    assert "first" in modes and "switch" in modes, "the script must exercise first and switch"


def test_gate_off_records_what_it_would_have_blocked_and_on_is_the_config_default():
    # SUBTASK-g: the default is on (SUBTASK-c: off — then this asserted `is False`). The gate-off
    # shadow record below is unchanged; `_confirm()` sets the gate off explicitly.
    assert ClusteringConfig().subtask_gate is True
    assert ClusteringConfig().subtask_confirm_frames is None
    assert ClusteringConfig().subtask_probe_path is None
    c = _confirmed(_confirm(), "place")
    d = _select(c, APPLE)
    assert d.mode == "first" and c.subtask_would_block == "first" and c.subtask_blocked is None


def test_gate_off_ground_target_is_identical_with_and_without_labels():
    apc = _apc((_ball(APPLE_C, n=400), 0.9), (_ball(BANANA_C, n=400, seed=4), 0.5))
    a, b = _confirm(), _confirmed(_confirm(), "home")
    ra = ground_target(apc, CFG, seed_threshold=0.3, confirm=a)
    rb = ground_target(apc, CFG, seed_threshold=0.3, confirm=b)
    assert ra.status is rb.status is GroundingStatus.OK
    assert np.array_equal(ra.target.points, rb.target.points)
    assert ra.target.metrics == rb.target.metrics


# --------------------------------------------------------- 2. gate on, place / home blocks


@pytest.mark.parametrize("label", ["place", "home"])
def test_gate_on_place_or_home_adopts_no_first_target(label):
    c = _confirmed(_confirm(gate=True), label)
    for _ in range(5):
        d = _select(c, APPLE)
        assert d.mode == SUBTASK_GATED and d.cluster is None and d.rank == -1
        assert d.subtask_blocked == "first" and d.subtask_label == label
    assert c.manipulated is None and c.held_centroid is None


@pytest.mark.parametrize("label", ["place", "home"])
def test_gate_on_place_or_home_never_switches_and_keeps_the_manipulated(label):
    c = _confirm(gate=True)
    assert _select(c, APPLE).mode == "first"
    held_id = c.manipulated.id
    _confirmed(c, label)
    modes = [_select(c, APPLE, BANANA).mode for _ in range(8)]
    assert modes == ["hold"] * 8
    assert c.manipulated.id == held_id and c.streak == 0
    assert c.subtask_blocked == "switch"
    # the held object leaving the frame is still `occluded`, not replaced
    assert _select(c, BANANA).mode == "occluded" and c.manipulated.id == held_id


def test_gate_on_over_the_script_blocks_exactly_the_switches_under_a_confirmed_place_or_home():
    plain = _run_script(_confirm())
    c = _confirm(gate=True)
    gated, blocked = [], []
    for k, items in enumerate(SCRIPT):
        c.note_subtask(LABEL_P[LABELS[k]])
        gated.append(_key(_select(c, *items)))
        blocked.append(c.subtask_blocked)
    # identical until the place label is confirmed (3rd place request, index 5) ...
    assert gated[:5] == plain[:5] and plain[4][0] == "switch"
    # ... after which no switch happens, and the manipulated id never changes again
    assert all(k[0] != "switch" for k in gated[5:])
    assert {k[2] for k in gated[5:]} == {plain[5][2]}
    assert "switch" in [k[0] for k in plain[5:]], "ungated, the same frames do switch"
    assert "switch" in blocked[5:] and "first" not in blocked


def test_gate_on_ground_target_reports_subtask_gated_and_names_no_target():
    c = _confirmed(_confirm(gate=True), "place")
    res = ground_target(_apc((_ball(APPLE_C, n=400), 0.9)), CFG, seed_threshold=0.5, confirm=c)
    assert res.target is None and res.status is GroundingStatus.SUBTASK_GATED
    assert len(res.clusters) == 1, "the clusters are still reported"


def test_gate_does_not_touch_the_destination_registry_or_admissibility():
    plain, gated = _confirm(), _confirmed(_confirm(gate=True), "place")
    for _ in range(4):
        _select(plain, CRATE, APPLE)
        _select(gated, CRATE, APPLE)
    assert plain.destination is not None and gated.destination is not None
    assert np.array_equal(plain.destination.points, gated.destination.points)
    assert [a.record() for a in plain.last_admissibility] == \
        [a.record() for a in gated.last_admissibility]


def test_the_gate_lifts_when_pick_is_confirmed_again():
    c = _confirmed(_confirm(gate=True), "place")
    assert _select(c, APPLE).mode == SUBTASK_GATED
    c.note_subtask(PICK)
    c.note_subtask(PICK)
    assert _select(c, APPLE).mode == SUBTASK_GATED, "two pick requests are not yet a label"
    c.note_subtask(PICK)
    assert _select(c, APPLE).mode == "first"


# ------------------------------------------------------------- 3. gate on, pick / None = now


@pytest.mark.parametrize("labels", [["pick"] * len(SCRIPT), [None] * len(SCRIPT), None])
def test_gate_on_pick_or_none_is_the_ungated_behaviour(labels):
    plain = _run_script(_confirm())
    c = _confirm(gate=True)
    assert _run_script(c, labels) == plain
    assert c.subtask_blocked is None


# --------------------------------------------------------------------------- 4. debounce


def test_debounce_two_places_do_not_confirm_the_third_does():
    d = SubtaskDebounce(3)
    assert d.update(PLACE) is None and d.update(PLACE) is None
    assert d.update(PLACE) == "place" and d.gated


def test_debounce_a_mixed_run_starts_over():
    d = SubtaskDebounce(3)
    for p in (PLACE, PLACE, PICK, PLACE, PLACE):
        assert d.update(p) is None
    assert d.update(PLACE) == "place"
    # once confirmed, it takes N agreeing frames to change again
    assert d.update(HOME) == "place" and d.update(HOME) == "place"
    assert d.update(HOME) == "home"


def test_debounce_none_is_its_own_class():
    d = SubtaskDebounce(3)
    for _ in range(3):
        d.update(PLACE)
    assert d.update(None) == "place" and d.update(None) == "place"
    assert d.update(None) is None and not d.gated


def test_reset_clears_the_label():
    c = _confirmed(_confirm(gate=True), "place")
    c.reset()
    assert c.subtask_label is None and c.subtask.streak == 0
    assert _select(c, APPLE).mode == "first"


def test_confirm_frames_default_follow_target_confirm_frames_and_can_be_set():
    assert _confirm().subtask.frames == CFG.target_confirm_frames == 3
    assert TargetConfirm(dataclasses.replace(CFG, target_confirm_frames=5)).subtask.frames == 5
    assert TargetConfirm(dataclasses.replace(CFG, subtask_confirm_frames=1)).subtask.frames == 1
    assert TargetConfirm(CFG, frames=4).subtask.frames == 4


def test_argmax_label():
    assert argmax_label(None) is None and argmax_label({}) is None
    assert argmax_label(PLACE) == "place"
    assert argmax_label({"pick": float("nan"), "place": 0.5, "home": 0.5}) is None
    assert argmax_label({"pick": 0.4, "place": 0.4, "home": 0.2}) == "pick"


@pytest.mark.parametrize("bad", [{"subtask_gate": 1}, {"subtask_confirm_frames": 0},
                                 {"subtask_confirm_frames": 1.5}, {"subtask_probe_path": ""}])
def test_config_rejects_malformed_subtask_values(bad):
    with pytest.raises(AG3SConfigError):
        AG3SConfig.from_dict({"clustering": bad})


# ------------------------------------------------------------------------------ 5. frozen


@pytest.mark.parametrize("gate", [False, True])
def test_frozen_the_label_changes_nothing(gate):
    ref = _run_script(_confirm(gate=gate), freeze_from=1)
    for labels in (LABELS, ["place"] * len(SCRIPT), ["home"] * len(SCRIPT)):
        assert _run_script(_confirm(gate=gate), labels, freeze_from=1) == ref


def test_frozen_with_nothing_held_still_adopts_first_under_place():
    """While a grasp is in progress the latch decides; the gate stands aside."""
    c = _confirmed(_confirm(gate=True), "place")
    c.freeze(True)
    assert _select(c, APPLE).mode == "first" and c.subtask_blocked is None


# ------------------------------------------------------------------- pipeline entry / record


def test_pipeline_set_subtask_metrics_and_gate():
    from AG3S.ag3s.runtime.pipeline import AG3S
    from tests.ag3s.fixtures import DEFAULT_ARM_Q, PlanarTwoLinkArm, make_scene

    arm = PlanarTwoLinkArm()
    scene = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    inputs = dict(attention_map=scene.attention_grid, depth=scene.depth,
                  camera_intrinsics=scene.camera_intrinsics, T_base_cam=scene.T_base_cam,
                  robot_state=DEFAULT_ARM_Q, phase="approach", image_hw=scene.hw)

    def cfg(gate):
        return AG3SConfig.from_dict({"pointcloud": {"self_filter_inflation": 0.05},
                                     "clustering": {"subtask_gate": gate,
                                                    "subtask_confirm_frames": 1}})

    base = AG3S(cfg(False), robot_model=arm).process(**inputs)
    off = AG3S(cfg(False), robot_model=arm)
    off.set_subtask({"p": PLACE, "argmax": "place", "probe": "kv_L4_v1"})  # whole record accepted
    cs_off = off.process(**inputs)
    assert cs_off.grounding_status is GroundingStatus.OK and cs_off.target is not None
    assert np.array_equal(cs_off.target.points, base.target.points)
    m = cs_off.metrics["subtask"]
    assert m == {"p": PLACE, "argmax": "place", "label": "place", "gate": False, "blocked": None,
                 "would_block": "first", "streak": 1, "frames": 1,
                 "released": False, "released_id": None, "placed_seen": False}

    on = AG3S(cfg(True), robot_model=arm)
    on.set_subtask(PLACE)
    cs_on = on.process(**inputs)
    assert cs_on.target is None and cs_on.grounding_status is GroundingStatus.SUBTASK_GATED
    assert cs_on.metrics["subtask"]["blocked"] == "first" and on.manipulated is None
    assert any("subtask gate" in n for n in cs_on.notes)
    # consumed: the next request without a label counts as None and (frames=1) lifts the gate
    cs_next = on.process(**inputs)
    assert cs_next.metrics["subtask"]["p"] is None and cs_next.metrics["subtask"]["label"] is None
    assert cs_next.target is not None
    on.reset()
    assert on.subtask_label is None


# ------------------------------------------------------------------------ 6. probe parity

ASSET = default_probe_path()
VAL = pathlib.Path(_ag3s_workspace_path('legacy/outputs/verify/SUBTASK-b/feat/val'))


def _phase_to_class(p):
    p = np.asarray(p)
    c = np.full(p.shape, -1)
    c[(p >= 0) & (p <= 3)] = 0
    c[(p >= 4) & (p <= 8)] = 1
    c[(p >= 9) & (p <= 13)] = 2
    return c


def test_the_asset_is_well_formed():
    probe = SubtaskProbe.load()
    meta = json.loads(ASSET.with_suffix(".json").read_text())
    assert probe.classes == CLASSES and probe.dim == 512 and probe.name == "kv_L4_v1"
    assert meta["classes"] == list(CLASSES) and meta["feature"]["dim"] == 512
    assert meta["npz_sha256"] == hashlib.sha256(ASSET.read_bytes()).hexdigest()
    assert meta["parity"]["pass"] and meta["parity"]["abs_diff_vs_verify_json"] <= 0.002
    p = probe.proba(np.zeros(512))
    assert p.shape == (3,) and abs(p.sum() - 1.0) < 1e-12
    rec = probe.predict(np.zeros(512, np.float32))
    assert set(rec["p"]) == set(CLASSES) and rec["probe"] == "kv_L4_v1"


@pytest.mark.skipif(not VAL.exists(), reason="SUBTASK-b feat/val not on this machine")
def test_numpy_probe_reproduces_the_asset_parity_number():
    probe = SubtaskProbe.load()
    meta = json.loads(ASSET.with_suffix(".json").read_text())
    X, y = [], []
    for f in sorted(VAL.glob("*.npz")):
        with np.load(f, allow_pickle=False) as z:
            X.append(z["normal__kv_L4"])
            y.append(_phase_to_class(z["phase_index"]))
    X, y = np.concatenate(X), np.concatenate(y)
    keep = y >= 0
    pred = probe.proba(X[keep]).argmax(1)
    acc = float((pred == y[keep]).mean())
    assert acc == pytest.approx(meta["parity"]["val_acc_numpy"], abs=1e-12)
    assert abs(acc - meta["parity"]["val_acc_reference_verify_json"]) <= 0.002
    # a subset of the files (every 5th episode) — the per-file path, not just the bulk one
    sub = [probe.predict(x)["argmax"] for x in X[keep][::50]]
    assert sub == [CLASSES[i] for i in pred[::50]]


def test_kv_feature_reference_masks_the_padded_text():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(768 + 200, 4)).astype(np.float32)
    mask = np.ones(968, bool)
    mask[768 + 37:] = False
    f = kv_feature(v, mask)
    assert f.shape == (8,) and f.dtype == np.float32
    assert np.allclose(f[:4], v[768:768 + 37].mean(0), atol=1e-6)
    assert np.allclose(f[4:], v[:768].mean(0), atol=1e-6)


# ------------------------------------------------------- 7. GPU: the live path's kv_L4 (opt-in)

STEP = pathlib.Path(_ag3s_workspace_path('legacy/outputs/live_test/20260925_ep1807/run_0000/step_00025.npz'))
CLOSED = pathlib.Path(_ag3s_workspace_path('legacy/outputs/verify/SUBTASK-b/feat/closed/ep1807_apple.npz'))
CKPT = (_ag3s_workspace_path('legacy/pi05_TO_hybrid/checkpoints/pi05_rby1_randomized_pick_place_16d_lora/rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999'))
CONFIG = "pi05_rby1_randomized_pick_place_16d_lora"


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("AG3S_GPU_TESTS") != "1",
                    reason="GPU + checkpoint; opt in with AG3S_GPU_TESTS=1")
def test_gpu_attention_sampler_kv_l4_matches_subtask_b(capsys):
    from AG3S.trajopt.attention_policy import AttentionPolicy
    from AG3S.ag3s.experiments.sources.pi05_attention import load_policy

    attn_policy = load_policy(CONFIG, CKPT)

    class _Stub:
        """The served policy, reduced to what `AttentionPolicy` reads from it."""
        _input_transform = attn_policy._input_transform  # noqa: SLF001

        @staticmethod
        def infer(obs, **kw):
            return {"actions": np.zeros((50, 16), np.float32)}

    ap = AttentionPolicy(_Stub(), attn_policy._model)  # noqa: SLF001
    assert ap.subtask_probe is not None
    z = np.load(STEP)
    prompt = json.loads((STEP.parent / "meta.json").read_text())["prompt"]
    files = sorted(STEP.parent.glob("step_*.npz"))
    j = files.index(STEP)
    ref = np.load(CLOSED)
    assert int(ref["file_index"][j]) == j and int(ref["frame_index"][j]) == int(z["t_step"])
    want = ref["normal__kv_L4"][j].astype(np.float64)

    rows = {}
    for dtype in (np.float64, np.float32):
        obs = {"state": np.asarray(z["state"], dtype),
               "images": {c: np.transpose(z[f"image_{c}"], (2, 0, 1))
                          for c in ("cam_high", "cam_left_wrist", "cam_right_wrist")},
               "prompt": prompt}
        attention, got = ap._attention_and_feature(dict(obs))  # noqa: SLF001
        got = np.asarray(got, np.float64)
        rel = float(np.linalg.norm(got - want) / np.linalg.norm(want))
        rows[np.dtype(dtype).name] = dict(rel_l2=rel, max_abs=float(np.abs(got - want).max()),
                                          p=ap.subtask_probe.predict(got)["p"])
        result = ap.infer(dict(obs))
        assert set(result) >= {"actions", "attention", "subtask"}
        assert result["subtask"]["probe"] == "kv_L4_v1"
        assert set(attention) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
        # the attention is the same whether or not the feature is taken from the prefix pass
        alone = ap._attention(dict(obs))  # noqa: SLF001
        assert all(np.array_equal(alone[c], attention[c]) for c in attention)
    p_ref = ap.subtask_probe.predict(want)["p"]
    with capsys.disabled():
        print(f"\n[SUBTASK-c A4#7] t_step={int(z['t_step'])} ref p={p_ref}")
        for k, v in rows.items():
            print(f"[SUBTASK-c A4#7] state {k}: rel_l2={v['rel_l2']:.3e} "
                  f"max_abs={v['max_abs']:.3e} p={v['p']}")
    assert rows["float32"]["rel_l2"] < 1e-2


# --------------------------------------------- B3: home after PLACED releases the placed object


def _placed_apple(gate=True):
    """Apple adopted (pick), grasped (frozen), released and PLACED — then the label goes home."""
    c = _confirmed(_confirm(gate=gate), "pick")
    assert _select(c, APPLE).mode == "first"
    held_id = c.manipulated.id
    c.freeze(True)                       # CLOSING / HELD
    _select(c, APPLE)
    c.freeze(False)                      # detach at PLACED
    c.note_placed(True)
    return c, held_id


def test_b3_i_home_after_placed_releases_the_manipulated_object():
    c, held_id = _placed_apple()
    _confirmed(c, "home")
    d = _select(c, APPLE, BANANA)
    assert c.subtask_released == held_id and c.manipulated is None and c.held_centroid is None
    assert d.mode == SUBTASK_GATED and d.cluster is None, "home also refuses a new first"
    # released once; the next frames have nothing to release and still adopt nothing
    for _ in range(4):
        assert _select(c, APPLE, BANANA).mode == SUBTASK_GATED
        assert c.subtask_released is None and c.manipulated is None


def test_b3_i_release_also_happens_on_a_frame_without_clusters():
    c, held_id = _placed_apple()
    _confirmed(c, "home")
    assert c.unseen() is None and c.subtask_released == held_id and c.manipulated is None


@pytest.mark.parametrize("when", ["before_grasp", "held", "revoked"])
def test_b3_ii_home_without_placed_releases_nothing(when):
    c = _confirmed(_confirm(gate=True), "pick")
    _select(c, APPLE)
    held_id = c.manipulated.id
    if when in ("held", "revoked"):
        c.freeze(True)
        _select(c, APPLE)
    if when == "revoked":
        c.freeze(False)                  # T34 attach_revoked: detach, latch back to LATCHED
        c.note_placed(False)
    _confirmed(c, "home")
    for _ in range(3):
        _select(c, APPLE)
        assert c.subtask_released is None and c.manipulated.id == held_id


def test_b3_ii_placed_but_frozen_releases_nothing():
    c, held_id = _placed_apple()
    _confirmed(c, "home")
    c.freeze(True)
    _select(c, APPLE)
    assert c.subtask_released is None and c.manipulated.id == held_id


def test_b3_iii_gate_off_releases_nothing():
    c, held_id = _placed_apple(gate=False)
    _confirmed(c, "home")
    for _ in range(3):
        _select(c, APPLE)
        assert c.subtask_released is None and c.manipulated.id == held_id
    assert c.placed_seen


def test_b3_place_label_after_placed_releases_nothing():
    c, held_id = _placed_apple()
    _confirmed(c, "place")
    _select(c, APPLE)
    assert c.subtask_released is None and c.manipulated.id == held_id


def test_b3_iv_pick_after_release_reopens_first_under_a_fresh_id():
    c, held_id = _placed_apple()
    _confirmed(c, "home")
    _select(c, APPLE)
    assert c.manipulated is None
    _confirmed(c, "pick")
    d = _select(c, BANANA)
    assert d.mode == "first" and c.manipulated.id == held_id + 1
    # placed_seen stays: a later home releases again (the episode's latch has placed)
    _confirmed(c, "home")
    _select(c, BANANA)
    assert c.subtask_released == held_id + 1 and c.manipulated is None


def test_b3_v_reset_clears_placed():
    c, _ = _placed_apple()
    assert c.placed_seen
    c.reset()
    assert not c.placed_seen and c.subtask_released is None
    _confirmed(c, "pick")
    _select(c, APPLE)
    _confirmed(c, "home")
    _select(c, APPLE)
    assert c.manipulated is not None and c.subtask_released is None


def test_b3_pipeline_release_metrics_and_no_exclusion():
    from AG3S.ag3s.runtime.pipeline import AG3S
    from tests.ag3s.fixtures import DEFAULT_ARM_Q, PlanarTwoLinkArm, make_scene

    arm = PlanarTwoLinkArm()
    scene = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    inputs = dict(attention_map=scene.attention_grid, depth=scene.depth,
                  camera_intrinsics=scene.camera_intrinsics, T_base_cam=scene.T_base_cam,
                  robot_state=DEFAULT_ARM_Q, phase="approach", image_hw=scene.hw)

    def run(gate):
        ag = AG3S(AG3SConfig.from_dict({"pointcloud": {"self_filter_inflation": 0.05},
                                        "clustering": {"subtask_gate": gate,
                                                       "subtask_confirm_frames": 1}}),
                  robot_model=arm)
        ag.set_subtask(PICK)
        first = ag.process(**inputs)
        assert ag.manipulated is not None and first.target is not None
        assert first.metrics["subtask"]["placed_seen"] is False
        ag.set_placed(True)
        ag.set_subtask(HOME)
        return ag, ag.process(**inputs)

    on, cs = run(True)
    m = cs.metrics["subtask"]
    assert m["released"] is True and m["placed_seen"] is True and m["label"] == "home"
    assert on.manipulated is None and cs.metrics["manipulated"] is None
    assert cs.metrics["exclusion"]["source"] == "none" and not cs.metrics["exclusion"]["active"]
    assert cs.target is None and cs.grounding_status is GroundingStatus.SUBTASK_GATED
    assert any("released" in n for n in cs.notes)
    on.reset()
    assert on._target_confirm.placed_seen is False  # noqa: SLF001

    off, cs_off = run(False)
    assert cs_off.metrics["subtask"]["released"] is False
    assert cs_off.metrics["subtask"]["placed_seen"] is True
    assert off.manipulated is not None and cs_off.target is not None
