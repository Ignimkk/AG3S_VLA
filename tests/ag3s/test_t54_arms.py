"""T54 arms: the study-side hooks change only the role they are defined to change.

Synthetic, no MuJoCo: the three-camera analytic scene of `tests/ag3s/fixtures.py` through the real
`AG3S.process_multi_debug`, and hand-built clouds through the real `ground_target`.

* pass-through hooks (L8H2 / N) are bit-identical to the unhooked serving call;
* lifting a map with `lift_on_fusion` reproduces the frame's own fused attention bit for bit;
* **A**: the cluster lists before and after the split are bit-identical to the L8H2 arm's; only the
  score that names the target changes, and it is the noun mass;
* **B**: the split ratio gate reads the L8H2 map (raw or normalised, as configured), while seeds and
  component order read the noun map.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, ClusteringConfig
from AG3S.ag3s.experiments.studies import t54_arms as A
from AG3S.ag3s.runtime.pipeline import AG3S
from AG3S.ag3s.stages.target_grounding import ground_target
from AG3S.ag3s.types import AttentionPointCloud, PointCloud
from tests.ag3s.fixtures import PlanarTwoLinkArm, make_attention_grid, make_three_camera_scene

CFG = AG3SConfig().with_overrides({
    "timing": {"expected_cameras": ["head", "left_wrist", "right_wrist"]},
    # the planar mock arm has no finger geometry: give the opening so clusters can be admissible
    "clustering": {"gripper_max_opening": 0.12},
})


@pytest.fixture(scope="module")
def arm():
    return PlanarTwoLinkArm()


@pytest.fixture(scope="module")
def scene(arm):
    return make_three_camera_scene(robot_model=arm)


def _grids_on(scene, centre, *, sigma=0.7, floor=0.0):
    """Per-camera 16x16 grids sharply peaked on `centre` (same views as the scene)."""
    out = {}
    for o in scene.observations:
        v = scene.views[o.camera_id]
        out[o.camera_id] = make_attention_grid(v.hw, v.camera_intrinsics, v.T_base_cam,
                                               np.asarray(centre, np.float64),
                                               sigma_patches=sigma, floor=floor)
    return out


@pytest.fixture(scope="module")
def maps(scene):
    view = next(iter(scene.views.values()))
    obj = {o.name: o.center for o in view.objects}
    return {
        "l8h2": {o.camera_id: o.attention_map for o in scene.observations},  # sphere + leak on box
        "noun": _grids_on(scene, obj["neighbour_box"]),
    }


def _observations(scene, cam_maps):
    return [dataclasses.replace(o, attention_map=cam_maps[o.camera_id]) for o in scene.observations]


def _run(arm_model, scene, maps, arm_spec, *, hooked=True, ag=None, cfg=CFG):
    ag = ag or AG3S(cfg, robot_model=arm_model)
    obs = _observations(scene, maps[arm_spec.primary])
    rec = A.CallRecord()
    if hooked:
        secondary = {s: maps[s] for s in arm_spec.secondary_sources()}
        with A.arm_hooks(arm_spec, secondary, rec):
            cs, dbg = ag.process_multi_debug(obs, phase="approach")
    else:
        cs, dbg = ag.process_multi_debug(obs, phase="approach")
    return cs, dbg, rec, ag


def _cluster_sets(gr):
    return sorted(tuple(np.sort(np.asarray(c.point_indices)).tolist()) for c in gr.clusters)


# ------------------------------------------------------------------------------ registry


def test_arm_table_is_the_task_table():
    a = A.ARMS
    assert list(a) == ["L8H2", "N", "A", "B", "L12H4", "A-L12H4", "N-k3", "A-k3"]
    assert (a["L8H2"].primary, a["L8H2"].split_source, a["L8H2"].choice_source) == ("l8h2",) * 3
    assert (a["N"].primary, a["N"].split_source, a["N"].choice_source) == ("noun",) * 3
    assert (a["A"].primary, a["A"].split_source, a["A"].choice_source) == ("l8h2", "l8h2", "noun")
    assert (a["B"].primary, a["B"].split_source, a["B"].choice_source) == ("noun", "l8h2", "noun")
    assert (a["A-L12H4"].primary, a["A-L12H4"].split_source, a["A-L12H4"].choice_source) == (
        "l12h4", "l12h4", "noun")
    assert (a["N-k3"].primary, a["A-k3"].choice_source) == ("noun_k3", "noun_k3")
    assert not a["L8H2"].hooked and not a["N"].hooked and a["A"].hooked and a["B"].hooked
    assert a["A"].secondary_sources() == ("noun",)
    assert a["B"].secondary_sources() == ("l8h2",)
    s = A.SOURCES
    assert (s["l8h2"]["layer"], s["l8h2"]["head"], s["l8h2"]["agg"], s["l8h2"]["denoise"]) == (8, 2, "last", 0)
    assert (s["l12h4"]["layer"], s["l12h4"]["head"], s["l12h4"]["agg"]) == (12, 4, "last")
    assert s["noun"]["cells"] == ((10, 7),) and s["noun"]["sigma"] == 0.0
    assert s["noun_k3"]["cells"] == ((8, 6), (5, 4), (4, 2)) and s["noun_k3"]["sigma"] == 1.0
    assert a["A"].extract_kinds() == ("action", "noun")
    assert a["N"].extract_kinds() == ("noun",)
    assert a["L8H2"].extract_kinds() == ("action",)


def test_action_and_noun_maps_use_the_t52_expressions():
    rng = np.random.default_rng(0)
    att = rng.random((2, 3, 3, 18, 8, 3, 16, 16)).astype(np.float16)
    m = A.action_map(att, [0, 4, 9], ["mean", "first", "last"], A.SOURCES["l8h2"])
    assert m.dtype == np.float32 and m.shape == (2, 3, 16, 16)
    assert np.array_equal(m, np.asarray(att[:, 0, 2, 8, 2], np.float64).astype(np.float32))
    block = rng.random((2, 3, 18, 8, 16, 16))
    n = A.noun_map(block, A.SOURCES["noun"])
    assert np.array_equal(n, block[:, :, 10, 7].astype(np.float32))


# ------------------------------------------------------------------------- pipeline level


def test_pass_through_hooks_are_bit_identical_to_the_serving_call(arm, scene, maps):
    for name in ("L8H2", "N"):
        spec = A.ARMS[name]
        cs0, d0, _, _ = _run(arm, scene, maps, spec, hooked=False)
        cs1, d1, rec, _ = _run(arm, scene, maps, spec, hooked=True)
        g0, g1 = d0["grounding"], d1["grounding"]
        assert [np.asarray(c.point_indices).tolist() for c in g0.clusters] == \
               [np.asarray(c.point_indices).tolist() for c in g1.clusters]
        assert [c.target_score for c in g0.clusters] == [c.target_score for c in g1.clusters]
        assert np.array_equal(g0.seed_indices, g1.seed_indices)
        assert np.array_equal(cs0.target.point_indices, cs1.target.point_indices)
        assert np.array_equal(d0["attention_cloud"].attention, d1["attention_cloud"].attention)
        assert rec.secondary == {} and rec.pre_split is not None


def test_lifting_the_primary_map_again_reproduces_the_fused_attention(arm, scene, maps):
    spec = A.ARMS["L8H2"]
    cs, dbg, rec, _ = _run(arm, scene, maps, spec)
    lifted = A.lift_on_fusion(rec.fusion, rec.observations, maps["l8h2"], rec.config)
    assert np.array_equal(lifted.attention, np.asarray(rec.fusion.attention, np.float32))
    assert np.array_equal(lifted.raw, np.asarray(rec.fusion.raw_attention, np.float32))
    assert np.array_equal(lifted.attention, np.asarray(dbg["attention_cloud"].attention, np.float32))


def test_A_candidate_lists_are_bit_identical_to_L8H2_and_only_the_choice_changes(arm, scene, maps):
    cl, dl, rl, _ = _run(arm, scene, maps, A.ARMS["L8H2"])
    ca, da, ra, _ = _run(arm, scene, maps, A.ARMS["A"])
    # before and after the split, in the order the stage built them: element-wise equal
    assert len(ra.pre_split) == len(rl.pre_split) and len(ra.post_split) == len(rl.post_split)
    assert all(np.array_equal(x, y) for x, y in zip(ra.pre_split, rl.pre_split))
    assert all(np.array_equal(x, y) for x, y in zip(ra.post_split, rl.post_split))
    assert A._digest(ra.post_split) == A._digest(rl.post_split)
    gl, ga = dl["grounding"], da["grounding"]
    assert np.array_equal(gl.seed_indices, ga.seed_indices)
    assert gl.attention_peak_index == ga.attention_peak_index
    assert _cluster_sets(gl) == _cluster_sets(ga)
    assert np.array_equal(dl["attention_cloud"].attention, da["attention_cloud"].attention)
    # the choice score is the noun mass over the cluster's points
    noun = A.lift_on_fusion(ra.fusion, ra.observations, maps["noun"], ra.config).attention
    for c in ga.clusters:
        assert c.target_score == pytest.approx(float(np.asarray(noun, np.float64)[c.point_indices].sum()),
                                               rel=0, abs=1e-9)
    # ... and the named cluster is the noun-mass leader (fresh state: rank 0 among admissible)
    assert ca.has_target
    masses = [float(np.asarray(noun, np.float64)[c.point_indices].sum()) for c in ga.clusters]
    assert np.array_equal(ca.target.point_indices, ga.clusters[int(np.argmax(masses))].point_indices)
    # the box-centred noun map makes the choice differ from L8H2's (the hook is effective)
    assert not np.array_equal(ca.target.point_indices, cl.target.point_indices)


def test_B_split_ratio_reads_the_L8H2_map_and_seeds_read_the_noun_map(arm, scene, maps):
    for source in ("raw", "normalized"):
        cfg = CFG.with_overrides({"clustering": {"attention_split_ratio_source": source}})
        _, dl, rl, _ = _run(arm, scene, maps, A.ARMS["L8H2"], cfg=cfg)
        _, dn, rn, _ = _run(arm, scene, maps, A.ARMS["N"], cfg=cfg)
        _, db, rb, _ = _run(arm, scene, maps, A.ARMS["B"], cfg=cfg)
        l8h2_cloud = dl["attention_cloud"]
        want = l8h2_cloud.raw_attention if source == "raw" else l8h2_cloud.attention
        assert rb.ratio_attention is not None
        assert np.array_equal(np.asarray(rb.ratio_attention, np.float32), np.asarray(want, np.float32))
        assert rb.ratio_source == f"l8h2_{source}"
        # primary role = noun: the same fused attention, seeds and pre-split groups as the N arm
        assert np.array_equal(db["attention_cloud"].attention, dn["attention_cloud"].attention)
        assert np.array_equal(db["grounding"].seed_indices, dn["grounding"].seed_indices)
        assert all(np.array_equal(x, y) for x, y in zip(rb.pre_split, rn.pre_split))


def test_hooks_restore_the_module_attributes_even_on_error(arm, scene, maps):
    import AG3S.ag3s.runtime.pipeline as pl
    import AG3S.ag3s.stages.target_grounding as tg

    before = (pl.fuse_observations, tg._split_groups, tg._score_cluster)
    with pytest.raises(RuntimeError):
        with A.arm_hooks(A.ARMS["A"], {"noun": maps["noun"]}, A.CallRecord()):
            raise RuntimeError("boom")
    assert (pl.fuse_observations, tg._split_groups, tg._score_cluster) == before
    with pytest.raises(ValueError):
        with A.arm_hooks(A.ARMS["A"], {}, A.CallRecord()):
            pass


# ----------------------------------------------------------------------- ground_target level


def _blob(centre, half, step=0.005):
    ax = [np.arange(c - h, c + h + 1e-9, step) for c, h in zip(centre, half)]
    g = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    return g


def _cloud(points, att, raw=None):
    pc = PointCloud(points=np.asarray(points, np.float64))
    a = np.asarray(att, np.float32)
    return AttentionPointCloud(pc, a, None if raw is None else np.asarray(raw, np.float32))


def _hooked_ground(arm_spec, cloud, cfg, secondary: dict):
    rec = A.CallRecord()
    rec.secondary.update(secondary)  # what fuse_wrap would have lifted
    with A.arm_hooks(arm_spec, {s: {} for s in arm_spec.secondary_sources()}, rec):
        gr = ground_target(cloud, cfg, seed_percentile=0.0)
    return gr, rec


def test_A_choice_by_noun_mass_over_identical_candidates():
    """Two separate blobs: L8H2 favours blob 1, the noun map blob 2 -> same clusters, A names blob 2."""
    b1, b2 = _blob((0.0, 0.0, 0.0), (0.02, 0.02, 0.02)), _blob((0.2, 0.0, 0.0), (0.02, 0.02, 0.02))
    pts = np.vstack([b1, b2])
    n1 = len(b1)
    l8h2 = np.r_[np.full(n1, 0.9), np.full(len(b2), 0.6)]
    noun = np.r_[np.full(n1, 0.2), np.full(len(b2), 0.8)]
    cfg = ClusteringConfig()
    cl = _cloud(pts, l8h2)
    g_l = ground_target(cl, cfg, seed_percentile=0.0)
    g_a, rec = _hooked_ground(A.ARMS["A"], cl, cfg, {"noun": A.Lifted(noun.astype(np.float32),
                                                                       noun.astype(np.float32))})
    assert _cluster_sets(g_l) == _cluster_sets(g_a) and len(g_a.clusters) == 2
    assert int(np.min(g_l.target.point_indices)) < n1          # L8H2 names blob 1
    assert int(np.min(g_a.target.point_indices)) >= n1         # A names blob 2 (noun mass)
    assert g_a.clusters[0].target_score == pytest.approx(float(noun[g_a.clusters[0].point_indices].sum()))


def test_B_split_follows_the_L8H2_ratio_not_the_noun_ratio():
    """One merged cluster (gap 20 mm < eps 30 mm, > split radius 15 mm). The noun map is nearly flat
    over the two parts (ratio < 2: N leaves it whole); L8H2 discriminates (ratio >= 2: B splits)."""
    b1 = _blob((0.0, 0.0, 0.0), (0.02, 0.02, 0.02))
    b2 = _blob((0.06, 0.0, 0.0), (0.02, 0.02, 0.02))   # 20 mm surface gap
    pts = np.vstack([b1, b2])
    n1 = len(b1)
    noun = np.r_[np.full(n1, 0.55), np.full(len(b2), 0.50)]
    l8h2 = np.r_[np.full(n1, 0.90), np.full(len(b2), 0.10)]
    cfg = ClusteringConfig(gripper_max_opening=0.08, attention_split_radius=0.015,
                           attention_split_ratio=2.0, attention_split_ratio_source="normalized")
    cloud_noun = _cloud(pts, noun)
    g_n = ground_target(cloud_noun, cfg, seed_percentile=0.0)
    assert len(g_n.clusters) == 1 and g_n.splits and not g_n.splits[0].accepted
    g_b, rec = _hooked_ground(A.ARMS["B"], cloud_noun, cfg,
                              {"l8h2": A.Lifted(l8h2.astype(np.float32), l8h2.astype(np.float32))})
    assert rec.ratio_source == "l8h2_normalized"
    assert np.array_equal(rec.ratio_attention, l8h2.astype(np.float32))
    assert g_b.splits and g_b.splits[0].accepted and len(g_b.clusters) == 2
    assert g_b.splits[0].ratio == pytest.approx(9.0)
    # B's pre-split group is N's (the primary map grows the clusters)
    assert len(rec.pre_split) == 1 and len(rec.pre_split[0]) == len(pts)


def test_candidate_fields_and_chosen_rank():
    rows = [{"plurality": "other_object", "admissible": True},
            {"plurality": "target", "admissible": False}]
    f = A.candidate_fields(rows)
    assert f["target_candidate_present"] is True
    assert f["target_candidate_present_admissible"] is False and f["n_candidates_admissible"] == 1
    assert A.candidate_fields([])["target_candidate_present"] is False
    unknown = [{"plurality": "target", "admissible": None}]
    assert A.candidate_fields(unknown)["target_candidate_present_admissible"] is None
