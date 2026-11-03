"""T78e — target-sticky memory cells (`esdf.surface_memory_target_sticky`, `OracleMemory(target_sticky=)`), default off.

| # | property |
|---|---|
| 1 | config default off · round trip · refusals (needs `attention_surface` + `surface_keep_last`); CLI flag → esdf key / record meta only when given; start-up refusals |
| 2 | off = newest wins as before: default ctor and explicit False give the same keys · points · codes · frame · `last` (no new key) over a whole sequence; this frame's code wins in every cell it touched |
| 3 | on: target → other is blocked (point · code · frame of the kept target sample), other → target lets target win, target → target updates, `n_sticky_kept` counts the kept cells |
| 4 | on: carving still drops sticky target samples (the object moved / seen through) |
| 5 | on: the ROI crop is the same as off |
| 6 | pipeline: off → the memory is built without the flag and the record has no `n_sticky_kept`; on → `target_exclusion.oracle.memory.n_sticky_kept` in the constraint record JSON |
"""

from __future__ import annotations

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError
from AG3S.ag3s.runtime.oracle_segmentation import (CODE_OBJECT_BASE, CODE_STATIC, CODE_TARGET, OracleMemory,
                                                   OracleView)

T78_ON = {"surface_split": True, "surface_target_check": "overlap", "surface_keep_last": True,
          "surface_grasp_hold": True, "surface_no_ball": True}


def _t78a():
    return pytest.importorskip("tests.ag3s.test_t78a_surface_post")


def _t46e():
    return pytest.importorskip("tests.ag3s.test_t46e_oracle_extent")


# ============================================================================ 1 config · CLI


def test_config_default_off_round_trip_and_refusals():
    assert AG3SConfig().esdf.surface_memory_target_sticky is False
    cfg = AG3SConfig.from_dict({"esdf": {"target_extent": "attention_surface", "surface_keep_last": True,
                                         "surface_memory_target_sticky": True}})
    assert cfg.esdf.surface_memory_target_sticky is True
    assert AG3SConfig.from_dict(cfg.to_dict()).esdf == cfg.esdf
    for bad in ({}, {"target_extent": "oracle_visible"}, {"target_extent": "attention_surface"}):
        with pytest.raises(AG3SConfigError, match="surface_memory_target_sticky"):
            AG3SConfig.from_dict({"esdf": {**bad, "surface_memory_target_sticky": True}})
    with pytest.raises(AG3SConfigError, match="attention_surface"):   # keep_last off-surface: the T78 rule refuses
        AG3SConfig.from_dict({"esdf": {"target_extent": "oracle_visible", "surface_keep_last": True,
                                       "surface_memory_target_sticky": True}})
    with pytest.raises(AG3SConfigError, match="surface_memory_target_sticky"):
        AG3SConfig.from_dict({"esdf": {"target_extent": "attention_surface", "surface_keep_last": True,
                                       "surface_memory_target_sticky": 1}})


def test_cli_flag_maps_to_esdf_key_and_meta_only_when_given():
    from AG3S.trajopt.serve_safe import ifg_cl1_record_meta, target_exclusion_section

    A = _t78a()
    base = A._parse(*A.SURF, *A.ALL)
    sec0, meta0 = target_exclusion_section(base), ifg_cl1_record_meta(base)
    assert base.surface_memory_target_sticky is False
    assert "surface_memory_target_sticky" not in set(sec0) | set(meta0)
    on = A._parse(*A.SURF, *A.ALL, "--surface-memory-target-sticky")
    sec = target_exclusion_section(on)
    assert {k: v for k, v in sec.items() if k not in sec0} == {"surface_memory_target_sticky": True}
    assert {k: v for k, v in ifg_cl1_record_meta(on).items() if k not in meta0} == {
        "surface_memory_target_sticky": True}
    assert AG3SConfig.from_dict({"esdf": sec}).esdf.surface_memory_target_sticky is True


def test_cli_refusals():
    A = _t78a()
    args = A._reject(*A.SURF, "--surface-keep-last", "--surface-memory-target-sticky")
    assert args.surface_memory_target_sticky
    with pytest.raises(SystemExit):   # no attention-surface
        A._reject("--surface-memory-target-sticky")
    with pytest.raises(SystemExit):   # no keep-last
        A._reject(*A.SURF, "--surface-memory-target-sticky")


# ============================================================================ toy views (one pixel per cell)

F = 100.0          # px; at z = 0.5 m one pixel = 5 mm > the 2.5 mm cell → one sample per cell
Z = 0.5012       # z / cell = 200.48: the 0.4 · 0.8 mm nearer frames below stay in the same cells
C0 = -0.3        # principal point off the pixel grid: x = (u + 0.3) · z / F sits inside a cell, not on its edge


def _view(codes, depth=Z):
    codes = np.asarray(codes, np.uint8)
    h, w = codes.shape
    K = np.array([[F, 0, C0], [0, F, C0], [0, 0, 1.0]])
    return OracleView(depth_m=np.full((h, w), float(depth)), K=K, T_base_cam=np.eye(4), labels=codes, name="cam")


def _halves(left, right, h=8, w=8):
    c = np.full((h, w), right, np.uint8)
    c[:, : w // 2] = left
    return c


def _state(mem):
    return mem.keys.copy(), mem.points.copy(), mem.codes.copy(), mem.frame.copy()


def _sequence():
    """Frames that overwrite, carve and crop: the t46e rig (fruits · hand · moved) and the toy relabel frames."""
    t46e = _t46e()
    hand = (t46e.A + np.array([0.0, 0.0, 0.08]), 0.04)
    moved = [t46e._raycast(v.T_base_cam, spheres=((t46e.A + [0.15, 0, 0], t46e.R, CODE_TARGET),
                                                  (t46e.B, t46e.R, CODE_OBJECT_BASE))) for v in t46e._views()]
    return ([t46e._views(), t46e._views(robot=hand), moved],
            [[_view(_halves(CODE_TARGET, CODE_OBJECT_BASE))], [_view(_halves(CODE_OBJECT_BASE, CODE_TARGET))],
             [_view(_halves(CODE_STATIC, CODE_STATIC))], [_view(np.zeros((8, 8), np.uint8), depth=Z + 0.1)]])


# ============================================================================ 2 off = newest wins


def test_off_is_the_default_and_newest_wins():
    for seq in _sequence():
        a, b = OracleMemory(), OracleMemory(target_sticky=False)
        for views in seq:
            sa, sb = a.update(views), b.update(views)
            assert "n_sticky_kept" not in sa and sa == sb
            for x, y in zip(_state(a), _state(b)):
                np.testing.assert_array_equal(x, y)
    # newest wins: target cells relabelled by the next frame take the new code
    mem = OracleMemory()
    mem.update([_view(np.full((8, 8), CODE_TARGET))])
    keys1 = mem.keys.copy()
    mem.update([_view(_halves(CODE_STATIC, CODE_OBJECT_BASE))])
    np.testing.assert_array_equal(mem.keys, keys1)
    assert not (mem.codes == CODE_TARGET).any() and (mem.frame == 2).all()


def test_off_matches_the_pre_t78e_merge_rule():
    """Off = the pre-T78e rule, re-derived independently: after carving and the ROI crop, the merged memory is the
    union of this frame's cells and the old cells, each cell taking this frame's first sample when it has one."""
    for seq in _sequence():
        mem = OracleMemory()
        for views in seq:
            old = {int(k): (p.copy(), int(c), int(f)) for k, p, c, f in zip(mem.keys, mem.points, mem.codes, mem.frame)}
            mem.update(views)
            for k, p, c, f in zip(mem.keys, mem.points, mem.codes, mem.frame):
                if f != mem.n_updates:     # not this frame's sample → an untouched old cell, unchanged
                    assert int(k) in old and old[int(k)][1] == int(c) and old[int(k)][2] == int(f)
                    np.testing.assert_array_equal(old[int(k)][0], p)


# ============================================================================ 3 on: the merge rule


def test_on_blocks_target_to_other_and_lets_target_win():
    off, on = OracleMemory(), OracleMemory(target_sticky=True)
    f1 = [_view(_halves(CODE_TARGET, CODE_OBJECT_BASE))]          # left target, right neighbour
    s_off, s_on = off.update(f1), on.update(f1)
    assert s_on["n_sticky_kept"] == 0 and {k: v for k, v in s_on.items() if k != "n_sticky_kept"} == s_off
    p1 = {int(k): p.copy() for k, p in zip(on.keys, on.points)}
    tgt1 = set(int(k) for k in on.keys[on.codes == CODE_TARGET])
    oth1 = set(int(k) for k in on.keys[on.codes == CODE_OBJECT_BASE])
    assert len(tgt1) == 32 and len(oth1) == 32
    # frame 2 (slightly nearer, same cells): left relabelled neighbour, right relabelled target
    f2 = [_view(_halves(CODE_OBJECT_BASE, CODE_TARGET), depth=Z - 0.0004)]
    s_off, s_on = off.update(f2), on.update(f2)
    assert "n_sticky_kept" not in s_off and s_on["n_sticky_kept"] == 32
    np.testing.assert_array_equal(on.keys, off.keys)
    code_on = dict(zip(on.keys.tolist(), on.codes.tolist()))
    code_off = dict(zip(off.keys.tolist(), off.codes.tolist()))
    frame_on = dict(zip(on.keys.tolist(), on.frame.tolist()))
    pt_on = {int(k): p for k, p in zip(on.keys, on.points)}
    for k in tgt1:                         # target → other: blocked, the kept sample is frame 1's
        assert code_off[k] == CODE_OBJECT_BASE and code_on[k] == CODE_TARGET and frame_on[k] == 1
        np.testing.assert_array_equal(pt_on[k], p1[k])
    for k in oth1:                         # other → target: target wins (as before)
        assert code_off[k] == code_on[k] == CODE_TARGET and frame_on[k] == 2
        assert not np.array_equal(pt_on[k], p1[k])
    assert s_on["n_target_samples"] == 64 and s_on["n_target_samples_this_frame"] == 32
    # frame 3: all target → target updates everywhere (nothing kept)
    s_on = on.update([_view(np.full((8, 8), CODE_TARGET), depth=Z - 0.0008)])
    assert s_on["n_sticky_kept"] == 0 and (on.frame == 3).all() and (on.codes == CODE_TARGET).all()
    # frame 4: static everywhere → all kept; other → other (no target in the cell) still updates
    s_on = on.update([_view(np.full((8, 8), CODE_STATIC), depth=Z - 0.0008)])
    assert s_on["n_sticky_kept"] == 64 and (on.codes == CODE_TARGET).all() and (on.frame == 3).all()
    mem = OracleMemory(target_sticky=True)
    mem.update([_view(_halves(CODE_TARGET, CODE_OBJECT_BASE))])
    mem.update([_view(_halves(CODE_TARGET, CODE_STATIC))])
    right = mem.codes != CODE_TARGET
    assert right.sum() == 32 and (mem.codes[right] == CODE_STATIC).all() and (mem.frame[right] == 2).all()


# ============================================================================ 4 on: carving


def test_on_carving_still_drops_sticky_target_samples():
    mem = OracleMemory(target_sticky=True)
    mem.update([_view(np.full((8, 8), CODE_TARGET))])
    mem.update([_view(np.full((8, 8), CODE_OBJECT_BASE))])
    assert mem.last["n_sticky_kept"] == 64 and (mem.codes == CODE_TARGET).all()
    st = mem.update([_view(np.zeros((8, 8), np.uint8), depth=Z + 0.1)])     # seen through: the object left
    assert st["n_carved"] == 64 and st["n_target_samples"] == 0 and len(mem) == 0 and st["n_sticky_kept"] == 0
    # seen through by a frame that labels the farther surface "other": carved first, nothing to keep
    mem = OracleMemory(target_sticky=True)
    mem.update([_view(_halves(CODE_TARGET, CODE_OBJECT_BASE))])
    st = mem.update([_view(np.full((8, 8), CODE_OBJECT_BASE), depth=Z + 0.1)])
    assert st["n_carved"] == 64 and st["n_sticky_kept"] == 0 and st["n_target_samples"] == 0
    # carve_only (A3 abstain path) too
    mem = OracleMemory(target_sticky=True)
    mem.update([_view(np.full((8, 8), CODE_TARGET))])
    assert mem.carve_only([_view(np.zeros((8, 8), np.uint8), depth=Z + 0.1)])["n_target_samples"] == 0


# ============================================================================ 5 on: ROI


def test_on_roi_crop_is_unchanged():
    codes = np.full((8, 40), CODE_STATIC, np.uint8)
    codes[:, :6] = CODE_TARGET             # 30 mm of target; the table runs 200 mm → beyond the 65 mm margin
    out = []
    for sticky in (False, True):
        mem = OracleMemory(target_sticky=sticky)
        st = mem.update([_view(codes)])
        out.append((st["n_new_in_roi"], _state(mem)))
        tgt = mem.points[mem.codes == CODE_TARGET]
        assert np.all(mem.points[:, 0] <= tgt[:, 0].max() + mem.roi_margin + 1e-12)
        assert st["n_new_in_roi"] < st["n_labelled_px"]
    assert out[0][0] == out[1][0]
    for x, y in zip(out[0][1], out[1][1]):
        np.testing.assert_array_equal(x, y)


def test_ctor_refuses_a_non_bool():
    with pytest.raises(ValueError, match="target_sticky"):
        OracleMemory(target_sticky=1)


# ============================================================================ 6 pipeline wiring · record


def _fold_and_build(monkeypatch, **esdf):
    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

    A = _t78a()
    fw, ag3s = A._wired(monkeypatch, target_exclusion="ball_owned", target_extent="attention_surface",
                        surface_keep_last=True, **esdf)
    obs, _, _ = A._rig(fw)
    A._update(ag3s, obs)
    A._update(ag3s, A._rig(fw)[0])
    field, _ = fw._build(ag3s)
    info = ConstraintRecordWriter._write_exclusion_trace(field, {})
    return ag3s, field, info


def test_pipeline_off_builds_the_memory_without_the_flag_and_records_no_new_key(monkeypatch):
    seen = []
    orig = OracleMemory.__init__

    def spy(self, *a, **k):
        seen.append(dict(k))
        orig(self, *a, **k)

    monkeypatch.setattr(OracleMemory, "__init__", spy)
    ag3s, field, info = _fold_and_build(monkeypatch)
    assert seen and all("target_sticky" not in k for k in seen)
    assert ag3s._oracle_memory.target_sticky is False
    assert "n_sticky_kept" not in field.stats["target_free"]["oracle"]["memory"]
    assert "n_sticky_kept" not in info["target_exclusion"]["oracle"]["memory"]


def test_pipeline_on_records_n_sticky_kept_in_the_constraint_record(monkeypatch):
    import json

    ag3s, field, info = _fold_and_build(monkeypatch, surface_memory_target_sticky=True)
    assert ag3s._oracle_memory.target_sticky is True
    mem_rec = info["target_exclusion"]["oracle"]["memory"]
    assert isinstance(mem_rec["n_sticky_kept"], int) and mem_rec["n_sticky_kept"] >= 0
    assert json.loads(json.dumps(info["target_exclusion"]))["oracle"]["memory"]["n_sticky_kept"] == mem_rec[
        "n_sticky_kept"]
    ag3s.reset()
    assert ag3s._oracle_memory.target_sticky is True and len(ag3s._oracle_memory) == 0
