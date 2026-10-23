"""methods/evaluate_offline (BASELINE-EVAL F): offline metrics of the safety filters / sampling guidance.

Runs in both interpreters: the filter / summary / GT-clearance tests in ``.venv-baselines`` (or ``.venv-openpi-live``),
the traced-sampler test only where JAX is installed (``.venv-openpi-live``)."""
import json
import pathlib

import numpy as np
import pytest

from Baseline.paper_baselines.methods import evaluate_offline as E

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCENE = PROJECT_ROOT / "outputs/verify/SL4/raw/scene/scene_libero_spatial_t0_e0.npz"


def test_delta_stats():
    A = np.zeros((4, 7))
    B = A.copy()
    B[1, :3] = [0.3, 0.4, 0.0]
    B[2, 3] = 0.2
    s = E.delta_stats(A, B, np.array([False, True, True, False]))
    assert s["n_intervened"] == 2 and s["intervention_rate"] == 0.5 and s["call_intervened"]
    assert s["da_xyz_l2_max"] == pytest.approx(0.5) and s["da_xyz_l2_mean"] == pytest.approx(0.125)
    assert s["da_xyz_l2_mean_intervened"] == pytest.approx(0.25) and s["da_rot_l2_max"] == pytest.approx(0.2)
    assert s["gripper_unchanged"]


def test_filter_step_info_layouts():
    knows = {"steps": [{"status": "solved", "feasible": True, "emergency_stop": False, "min_h": 0.1},
                       {"status": "infeasible", "feasible": False, "emergency_stop": True, "min_h": -0.02}]}
    k = E.filter_step_info("knows", knows)
    assert k["h_min"] == -0.02 and k["h_neg_steps"] == 1 and k["qp_fail"] == 1 and k["emergency_stops"] == 1
    aegis = {"steps": [{"h": [0.3, 0.05], "status": "inactive", "solve_ms": 0.1},
                       {"h": [0.2, -0.01], "status": "fallback_nominal(infeasible)", "solve_ms": 2.0}]}
    a = E.filter_step_info("aegis", aegis)
    assert a["h_steps"] == [0.05, -0.01] and a["h_matrix"] == [[0.3, 0.05], [0.2, -0.01]] and a["qp_fail"] == 1
    ml = {"steps": [{"status": "optimal", "min_h": 0.04, "min_h_pair": "forearm_1|cookies_1", "h_by_body": {"gripper": 0.1},
                     "ms": 3.0}]}
    m = E.filter_step_info("multilink_cbf", ml)
    assert m["min_h_pair"] == ["forearm_1|cookies_1"] and m["solve_ms_steps"] == [3.0] and m["qp_fail"] == 0


def test_obstacle_ellipsoids_knows_and_aegis():
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    Q = R @ np.diag([0.01, 0.04, 0.09]) @ R.T
    out = E.obstacle_ellipsoids({"obstacles": {"cup": {"c": [1, 2, 3], "Q": Q.tolist()}}})
    np.testing.assert_allclose(sorted(out[0]["semi_axes"]), [0.1, 0.2, 0.3])
    V, s = np.asarray(out[0]["R"]), np.asarray(out[0]["semi_axes"])
    np.testing.assert_allclose(V @ np.diag(s ** 2) @ V.T, Q, atol=1e-12)
    a = E.obstacle_ellipsoids({"obstacles": [{"name": "box", "center": [0, 0, 0], "semi_axes": [1, 1, 1]}]})
    assert a[0]["name"] == "box" and a[0]["R"] == np.eye(3).tolist()


def test_featured_call():
    rows = [{"surrogate": "tracking", "da_xyz_l2_max": 0.1, "call_k": 0}, {"surrogate": "tracking", "da_xyz_l2_max": 0.5, "call_k": 3},
            {"surrogate": "constant", "da_xyz_l2_max": 9.0, "call_k": 1}]
    assert E.featured_call(rows) == 3
    assert E.featured_call([{"surrogate": "tracking", "da_xyz_l2_max": 0.0, "call_k": 2}]) == 2
    assert E.featured_call([]) is None


def _row(arm, k, intervened, pairs=()):
    gt = {"ee_obstacle_min": 0.05, "arm_obstacle_min": 0.08, "ee_obstacle_neg_rows": 0, "arm_obstacle_neg_rows": 0}
    return {"arm": arm, "episode_id": "e0", "call_k": k, "obs_index": 10 + 5 * k, "T": 10, "n_intervened": 3 if intervened else 0,
            "call_intervened": intervened, "da_xyz_l2_mean": 0.1, "da_xyz_l2_mean_intervened": 0.2 if intervened else None,
            "da_xyz_l2_max": 0.3, "da_rot_l2_mean": 0.0, "gripper_unchanged": True, "ms_total": 5.0, "solve_ms_mean": 1.0,
            "h_min": 0.02, "h_neg_steps": 0, "qp_fail": 0, "emergency_stops": 0, "min_h_pair_set": list(pairs),
            "gt_nominal": gt, "gt_modified": {**gt, "ee_obstacle_min": 0.07}}


def test_summarize_family_and_multilink_cases():
    rows = [_row("aegis", 0, False), _row("multilink_cbf", 0, True, ["forearm_1|cookies_1"]),
            _row("aegis", 1, True), _row("multilink_cbf", 1, True, ["gripper|cookies_1"])]
    s = E.summarize_family(rows, "filters")
    a = s["arms"]["multilink_cbf"]
    assert a["intervention_rate_rows"] == pytest.approx(0.3) and a["intervention_rate_calls"] == 1.0
    assert a["gt_clearance_change_ee_obstacle_mean"] == pytest.approx(0.02)
    c = s["multilink_intervenes_aegis_not"]
    assert c["n_calls"] == 1 and c["n_non_gripper_binding"] == 1 and c["calls"][0]["multilink_binding_bodies"] == ["forearm_1"]


@pytest.mark.skipif(not SCENE.exists(), reason="SL4 scene GT not present")
def test_gt_clearance_ik_and_sign():
    """IK follows the decoded path; a sphere centre inside an obstacle has negative clearance."""
    g = E.GTClearance(SCENE)
    rs = np.array([0.0, 0.2, 0.0, -2.0, 0.0, 2.2, 0.8, 0.02, -0.02])  # 7 arm + 2 finger qpos
    q0 = g.panda.set_finger_parameters(g.model, rs)
    p0 = g._fk(q0)[:3, 3]
    path = p0[None] + np.linspace(0, 1, 6)[:, None] * np.array([0.03, -0.02, -0.04])[None]
    m = g.path_metrics(rs, path, int(g.scene.frames[0]))
    assert m["ik_pos_err_max_m"] < 1e-3
    assert all(-1.0 < v <= E.CLEAR_CAP_M for v in m["ee_obstacle_by_row"])
    summ = E._gt_summary(m)  # the jsonl / summary keep the counts, drop only the per-row arrays
    assert "ee_obstacle_neg_rows" in summ and "arm_support_neg_rows" in summ and "ee_obstacle_by_row" not in summ
    # put the gripper inside the (non-target) cookies box -> clearance < 0
    row = g.scene.frame_row(int(g.scene.frames[0]))
    i = g.names.index("cookies_1")
    gid = g.scene.instances()[i].collision_geoms[0]
    _, centre = g.scene.geom_pose_base(row, gid)
    out = g.sphere_clearance(np.array([q0]), int(g.scene.frames[0]))
    assert out["obstacle"].shape == (1, len(g.links))
    D = g.signed_distance(np.vstack([centre, centre + [0, 0, 0.5]]), row, 0.2)
    assert D[0, i] < 0  # inside the cookies collision box
    assert not np.isfinite(D[1, i])  # 0.5 m above: beyond the search radius


def test_osc_path_shape_and_start():
    p = E.osc_path(np.array([0.5, 0.0, 0.3]), np.eye(3), np.zeros(3), np.tile([1.0, 0, 0, 0, 0, 0, -1], (10, 1)))
    assert p.shape == (11, 3) and np.allclose(p[0], [0.5, 0.0, 0.3]) and np.all(np.diff(p[:, 0]) > 0)


def test_traced_sampler_bit_equal_to_surrogate():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from Baseline.paper_baselines.methods.guidance_common.core import surrogate_sample

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 10, 32)).astype(np.float32)
    noise = rng.standard_normal((1, 10, 32)).astype(np.float32)

    def fn(x_next, t_next):  # a toy guidance: pull xyz towards 0 more strongly late in sampling
        return x_next.at[..., :3].add(-0.1 * (1.0 - t_next) * x_next[..., :3])

    for kind in E.SURROGATES:
        ref = np.asarray(jax.jit(lambda: surrogate_sample(x, noise, guidance_fn=fn, kind=kind))())
        out, corr = jax.jit(lambda: E.surrogate_sample_traced(x, noise, guidance_fn=fn, kind=kind))()
        assert np.array_equal(ref, np.asarray(out))
        corr = np.asarray(corr)
        assert corr.shape == (10, 10, 32) and np.all(corr[..., 3:] == 0) and np.all(corr[-1, :, :3] != 0)
    ref = np.asarray(jax.jit(lambda: surrogate_sample(x, noise))())
    out, corr = jax.jit(lambda: E.surrogate_sample_traced(x, noise))()
    assert np.array_equal(ref, np.asarray(out)) and not np.asarray(corr).any()
    del jnp


def test_parser_families():
    a = E.build_parser().parse_args(["filters", "--limit-calls", "2", "--out-dir", "outputs/x"])
    assert a.family == "filters" and a.source_run == E.SL3_DEV_RUN and a.scene_gt == E.SL3_SCENE_GT
    g = E.build_parser().parse_args(["guidance", "--into", "/tmp/r"])
    assert g.variants == list(E.VARIANTS) and g.surrogates == list(E.SURROGATES) and g.clear_jax_caches
    s = E.build_parser().parse_args(["summarize", "--into", "/tmp/r"])
    assert s.family == "summarize"


def test_summarize_merges(tmp_path):
    for fam in ("filters", "guidance"):
        (tmp_path / fam).mkdir()
        rows = [_row("knows" if fam == "filters" else "nudge|scene|tracking", 0, True)]
        if fam == "guidance":
            rows[0].pop("ms_total")
            rows[0].update(ms_build=1.0, ms_jit_and_sample=2.0, skip=False, frozen=False, finite=True)
        (tmp_path / fam / "summary.json").write_text(json.dumps(E.summarize_family(rows, fam)))
    E.run_summarize(E.build_parser().parse_args(["summarize", "--into", str(tmp_path)]))
    d = json.loads((tmp_path / "summary_all.json").read_text())
    assert {r["arm"] for r in d["table"]} == {"knows", "nudge|scene|tracking"}


@pytest.mark.skipif(not SCENE.exists(), reason="SL4 scene GT not present")
def test_fast_clearance_equals_exact_below_cap():
    """The pruned distance (centroid KD-tree / bounding-sphere bounds) equals SceneGT.exact_distance_near wherever the
    clearance is below the cap, and is inf (capped) elsewhere."""
    g = E.GTClearance(SCENE)
    t = int(g.scene.frames[len(g.scene.frames) // 2])
    row = g.scene.frame_row(t)
    rng = np.random.default_rng(0)
    centres = []
    for i in g.used:  # points scattered within ~8 cm of every measured instance
        for gid in g.scene.instances()[i].visual_geoms[:2]:
            tri = g.scene.triangles_base(row, gid).reshape(-1, 3)
            centres.append(tri[rng.integers(0, len(tri), 40)] + rng.normal(0, 0.04, (40, 3)))
    P = np.concatenate(centres)
    r = rng.uniform(0.0, 0.06, len(P))
    fast = g.clearance_distance(P, r, row)
    ref = g.signed_distance(P, row, g.cap + float(r.max()))
    need = (g.cap + r)[:, None]
    m = np.isfinite(ref) & (ref <= need)
    m[:, [i for i in range(m.shape[1]) if i not in g.used]] = False
    assert m.sum() > 50
    np.testing.assert_array_equal(fast[m], ref[m])
    used = np.zeros(m.shape[1], bool)
    used[g.used] = True
    assert not np.isfinite(fast[~m & used[None]]).any()
