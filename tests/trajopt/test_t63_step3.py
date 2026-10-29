"""T63 step 3 — the common evaluator's metrics (`curobo_v2/step3_metrics.py`), the lambda wiring
of `t63_curobo_compare` (user decision 3) and the B0 rule for chunks without `esdf_free`
(`world.compose_b0`, `OurWorlds.b0`, `ComposedSceneCollision` mode `b0`).

CPU except the last test (cuRobo + CUDA)."""

from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.trajopt.curobo_v2 import step3_metrics as M
from AG3S.trajopt.curobo_v2 import world as W

NQ = 14


def _lim(vel=1.0, acc=10.0, lo=-3.0, hi=3.0, dt=1.0 / 15.0):
    return M.step_limits({"lower": [lo] * NQ, "upper": [hi] * NQ, "velocity": [vel] * NQ,
                          "acceleration": [acc] * NQ, "dt": dt})


# =================================================================================================
# metrics
# =================================================================================================

def test_step_limits_are_per_15hz_step():
    lim = _lim(vel=1.5, acc=9.0)
    assert np.allclose(lim["max_step"], 1.5 / 15.0)
    assert np.allclose(lim["max_step_change"], 9.0 / 225.0)


def test_velocity_counts_the_anchor_and_acceleration_skips_step_zero():
    lim = _lim(vel=1.5, acc=9.0)  # max_step 0.1, max_step_change 0.04
    q_now = np.zeros(NQ)
    q = np.zeros((50, NQ))
    q[:, 3] = 0.15                      # step 0 jumps 0.15 from q_now: velocity (anchor) violation
    q[20:, 5] = 0.05                    # step 20: velocity 0.05 ok; accel 0.05 > 0.04 at 20 and 21
    out = M.kinematics(q, q_now, lim)
    assert out["n_vel_violation_steps_exec8"] == 1
    assert out["n_acc_violation_steps_exec8"] == 1      # step 1: |0 - 2*0.15 + 0.15| = 0.15 > 0.04
    assert out["n_acc_violation_steps_plan16"] == 1
    assert out["n_acc_violation_steps_all50"] == 3      # steps 1, 20, 21
    assert out["n_pos_violation_steps_all50"] == 0
    assert np.isclose(out["max_vel_excess_rad_s_exec8"], 0.05 * 15.0)
    assert np.isclose(out["max_acc_excess_rad_s2_all50"], (0.15 - 0.04) * 225.0)


def test_limit_tolerance_ignores_qp_noise():
    lim = _lim(vel=1.5)
    q = np.zeros((50, NQ))
    q[:, 0] = 0.1 + 0.5 * M.LIMIT_TOL_RAD  # one jump, at step 0 (the anchor)
    assert M.kinematics(q, np.zeros(NQ), lim)["n_vel_violation_steps_exec8"] == 0
    q[:, 0] = 0.1 + 2.0 * M.LIMIT_TOL_RAD
    assert M.kinematics(q, np.zeros(NQ), lim)["n_vel_violation_steps_exec8"] == 1


def test_position_violation_and_jerk_rms_of_a_cubic():
    lim = _lim(vel=100.0, acc=1e4, lo=-1.0, hi=1.0)
    c = 1e-6
    k = np.arange(1, 51, dtype=float)
    q = np.tile((c * k ** 3)[:, None], (1, NQ))      # q_now = 0 = c * 0^3
    out = M.kinematics(q, np.zeros(NQ), lim)
    want = 6.0 * c / lim["dt"] ** 3                  # constant 3rd difference
    for w in M.WINDOWS:
        assert np.isclose(out[f"jerk_rms_rad_s3_{w}"], want)
    assert out["n_pos_violation_steps_all50"] == int((c * k ** 3 > 1.0).sum())


def _traj_case():
    rng = np.random.default_rng(0)
    ref_q = rng.normal(scale=0.3, size=(50, NQ))
    q = ref_q.copy()
    q[-1] += 0.01
    q[-1, 2] += 0.05
    S = 7
    cl = np.full((50, S), 0.05)
    cl[3, 1] = 0.004      # exec8 min 4 mm
    cl[10, 5] = -0.002    # plan16 min -2 mm (penetration in plan window only)
    cl[40, 0] = -0.010    # all50 min -10 mm
    ref_tcp = rng.normal(scale=0.2, size=(50, 3))
    tcp = ref_tcp.copy()
    tcp[:, 0] += 0.001    # 1 mm everywhere
    tcp[12, 1] += 0.004   # plan16 max sqrt(1 + 16) mm
    tcp[-1, 2] += 0.002
    return q, ref_q, cl, tcp, ref_tcp


def test_candidate_metrics_windows_and_goal():
    q, ref_q, cl, tcp, ref_tcp = _traj_case()
    m = M.candidate_metrics(q, np.zeros(NQ), cl, tcp, ref_q, ref_tcp, _lim(vel=100, acc=1e5,
                                                                          lo=-10, hi=10))
    assert m["finite"] and m["has_trajectory"]
    assert np.isclose(m["min_clearance_mm_exec8"], 4.0)
    assert np.isclose(m["min_clearance_mm_plan16"], -2.0)
    assert np.isclose(m["min_clearance_mm_all50"], -10.0)
    assert (m["penetration_exec8"], m["penetration_plan16"], m["penetration_all50"]) == (
        False, True, True)
    assert m["worst_plan16"] == {"step": 10, "sphere": 5}
    assert np.isclose(m["tcp_disp_mm_mean_exec8"], 1.0)
    assert np.isclose(m["tcp_disp_mm_max_plan16"], np.sqrt(17.0))
    assert np.isclose(m["goal_error_joint_rad"], 0.06)
    assert np.isclose(m["goal_error_tcp_mm"], np.sqrt(1.0 + 4.0))


def test_candidate_metrics_missing_and_nonfinite_are_flagged_not_measured():
    q, ref_q, cl, tcp, ref_tcp = _traj_case()
    assert M.candidate_metrics(None, np.zeros(NQ), None, None, ref_q, ref_tcp, _lim()) == {
        "has_trajectory": False, "finite": False}
    q[5, 0] = np.nan
    assert M.candidate_metrics(q, np.zeros(NQ), cl, tcp, ref_q, ref_tcp, _lim()) == {
        "has_trajectory": True, "finite": False}


def _row(to_active, box, grasp, cands):
    return {"to_active": to_active, "box_layer": box, "grasp_active": grasp, "candidates": cands}


def _entry(min8, wall=None, cold=False, success=True, gpu=None):
    met = {"has_trajectory": True, "finite": True}
    for w in M.WINDOWS:
        met[f"min_clearance_mm_{w}"] = min8
        met[f"penetration_{w}"] = min8 < 0
        met[f"tcp_disp_mm_mean_{w}"] = met[f"tcp_disp_mm_max_{w}"] = 1.0
        met[f"jerk_rms_rad_s3_{w}"] = 2.0
        met[f"max_vel_excess_rad_s_{w}"] = met[f"max_acc_excess_rad_s2_{w}"] = 0.0
        for k in ("vel", "acc", "pos"):
            met[f"n_{k}_violation_steps_{w}"] = 1 if min8 < 0 else 0
    met["goal_error_joint_rad"] = met["goal_error_tcp_mm"] = 0.0
    solve = None if wall is None else {"wall_ms": wall, "cold": cold, "success": success,
                                       "status": "ok" if success else "fail", "device": "gpu",
                                       "gpu_solve_ms": gpu}
    return {"metrics": met, "solve": solve}


def test_aggregate_splits_counts_and_timing():
    rows = [
        _row(True, "free", False, {"a": _entry(-1.0, 50.0), "b": _entry(3.0)}),
        _row(False, "main_fine", True, {"a": _entry(2.0, 70.0, success=False), "b": _entry(-3.0)}),
        _row(True, "free", True, {"a": _entry(5.0, 4000.0, cold=True),
                                  "b": {"metrics": {"has_trajectory": True, "finite": False},
                                        "solve": None}}),
        _row(False, "free", False, {"b": _entry(1.0)}),  # `a` not run on this chunk
    ]
    agg = M.aggregate(rows, ["a", "b"])
    assert agg["all"]["n_chunks"] == 4
    assert agg["to_active"]["n_chunks"] == 2 and agg["to_inactive"]["n_chunks"] == 2
    assert agg["box_free"]["n_chunks"] == 3 and agg["box_main_fine"]["n_chunks"] == 1
    assert agg["post_grasp"]["n_chunks"] == 2 and agg["pre_grasp"]["n_chunks"] == 2
    a = agg["all"]["candidates"]["a"]
    assert (a["n_evaluated"], a["n_not_run"], a["n_solver_failure"]) == (3, 1, 1)
    assert a["n_penetration_exec8"] == 1
    assert a["metrics"]["min_clearance_mm_exec8"]["min"] == -1.0
    assert a["kinematic_counts"]["n_vel_violation_steps_exec8"] == {"total_steps": 1,
                                                                     "n_chunks_with_any": 1}
    t = a["timing"]["wall"]
    assert t["cold_ms"] == [4000.0] and t["warm"]["n"] == 2
    assert t["warm_n_over_66p7ms"] == 1 and np.isclose(t["warm_fraction_over_66p7ms"], 0.5)
    b = agg["all"]["candidates"]["b"]
    assert (b["n_evaluated"], b["n_nonfinite"], b["n_not_run"]) == (3, 1, 0)
    assert agg["to_active"]["candidates"]["b"]["n_penetration_exec8"] == 0


# =================================================================================================
# lambda wiring (user decision 3)
# =================================================================================================

@pytest.fixture(scope="module")
def T63():
    from AG3S.trajopt.experiments import t63_curobo_compare as T

    return T


BASE = {"w_track": 1.0, "w_smooth": 0.05, "w_continuity": 0.5, "w_slack": 1000.0,
        "w_deflection_rate": 20.0, "track_decay": 1.0, "servo_accel": "off",
        "servo_accel_weight": 0.01}


@pytest.mark.parametrize("lam", [1.0, 100.0, 1e4])
def test_lambda_scales_exactly_our_three_terms(T63, lam):
    w = T63.scaled_weights(BASE, lam)
    assert T63.LAMBDA_SCALED_WEIGHTS == ("w_track", "w_continuity", "w_deflection_rate")
    for k, v in BASE.items():
        want = v * lam if k in T63.LAMBDA_SCALED_WEIGHTS else v
        assert w[k] == want, k
    assert BASE["w_track"] == 1.0  # input untouched


def test_lambda_validation_and_names(T63):
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            T63.scaled_weights(BASE, bad)
    assert T63.parse_lambdas("1,100,10000") == [1.0, 100.0, 1e4]
    assert T63.candidate_name("B0", None) == "B0"
    assert T63.candidate_name("B3", 1e4) == "B3_lam10000"
    assert T63.candidate_name("B1", 1.0) == "B1_lam1"
    with pytest.raises(ValueError):
        T63.candidate_name("B0", 100.0)
    with pytest.raises(ValueError):
        T63.candidate_name("B2", None)


def test_step3_candidates_one_b0_and_each_variant_per_lambda(T63):
    c = T63.step3_candidates(["B0", "B1", "B2", "B3", "B4"], T63.STEP3_LAMBDAS)
    names = [n for n, _, _ in c]
    assert len(c) == 13 and names.count("B0") == 1 and len(set(names)) == 13
    assert ("B4_lam100", "B4", 100.0) in c
    assert all(lam is None for n, v, lam in c if v == "B0")


def test_build_terms_carries_the_scaled_weights(T63):
    torch = pytest.importorskip("torch")
    dev = types.SimpleNamespace(device="cpu", dtype=torch.float64)
    names = T63.VARIANTS["B3"][1]
    for lam in T63.STEP3_LAMBDAS:
        terms = T63.build_terms(names, T63.scaled_weights(BASE, lam), 50, dev)
        assert set(terms) == {"track", "continuity", "deflection_rate"}  # servo accel: inactive
        assert terms["track"].w_track == lam * BASE["w_track"]
        assert terms["track"].track_decay == BASE["track_decay"]
        assert terms["continuity"].w_continuity == lam * BASE["w_continuity"]
        assert terms["deflection_rate"].w == lam * BASE["w_deflection_rate"]
    with pytest.raises(SystemExit):
        T63.build_terms(names, {**BASE, "servo_accel": "strict"}, 50, dev)


def test_term_value_is_linear_in_lambda(T63):
    """The cost a term adds to cuRobo's objective scales with lambda and nothing else does."""
    torch = pytest.importorskip("torch")
    dev = types.SimpleNamespace(device="cpu", dtype=torch.float64)
    rng = np.random.default_rng(3)
    ref = rng.normal(size=(50, NQ))
    pos = torch.as_tensor(rng.normal(size=(2, 101, NQ)))
    vals = []
    for lam in (1.0, 100.0):
        t = T63.build_terms(("track",), T63.scaled_weights(BASE, lam), 50, dev)["track"]
        t.set_reference(ref)
        vals.append(t.forward_position(pos).sum().item())
    assert np.isclose(vals[1], 100.0 * vals[0])


# =================================================================================================
# B0 world without esdf_free (step 3: our main field inside the fine lattice)
# =================================================================================================

def test_compose_b0_rule():
    coarse = np.array([0.10, 0.02, 0.30, 0.05])
    box = np.array([0.08, 0.04, 0.01, 0.06])
    in_box = np.array([True, True, False, True])
    np.testing.assert_allclose(W.compose_b0(box, in_box, coarse, has_free=True),
                               W.compose_box(box, in_box, coarse))
    np.testing.assert_allclose(W.compose_b0(box, in_box, coarse, has_free=False),
                               [0.08, 0.02, 0.30, 0.05])


def _synthetic_record(rng, with_free: bool):
    z = {"esdf_coarse_distance": rng.uniform(-0.05, 0.5, (8, 8, 8)).astype(np.float16),
         "esdf_coarse_origin": np.array([-0.4, -0.4, -0.4]), "esdf_coarse_voxel_size": np.array(0.1),
         "esdf_distance": rng.uniform(-0.02, 0.3, (16, 16, 16)).astype(np.float16),
         "esdf_origin": np.array([-0.1, -0.1, -0.1]), "esdf_voxel_size": np.array(0.0125)}
    if with_free:
        z.update({"esdf_free_distance": rng.uniform(0.0, 0.3, (16, 16, 16)).astype(np.float16),
                  "esdf_free_origin": np.array([-0.1, -0.1, -0.1]),
                  "esdf_free_voxel_size": np.array(0.0125)})
    return z


def test_ourworlds_b0_without_free_is_our_main_field(T63):
    pytest.importorskip("scipy")
    from AG3S.ag3s.fields.curobo_field import CuroboEsdfField, layer_from_arrays

    rng = np.random.default_rng(11)
    z = _synthetic_record(rng, with_free=False)
    ours = T63.OurWorlds(z, 0.5)
    pts = rng.uniform(-0.35, 0.35, (800, 3))
    d, vs = ours.b0(ours.layers(pts))
    main = CuroboEsdfField((layer_from_arrays(z["esdf_coarse_distance"], z["esdf_coarse_origin"], 0.1),
                            layer_from_arrays(z["esdf_distance"], z["esdf_origin"], 0.0125)),
                           outside_distance=0.5)
    np.testing.assert_allclose(d, np.asarray(main._evaluate(pts)[0]), atol=1e-12)
    L = ours.layers(pts)
    fine_wins = L["in_fine"] & (L["fine"] < L["coarse"])
    assert np.all(vs[fine_wins] == 0.0125) and np.all(vs[~fine_wins] == 0.1)
    # with esdf_free the rule is unchanged (user decision 2): the box layer alone inside its lattice
    z2 = _synthetic_record(rng, with_free=True)
    ours2 = T63.OurWorlds(z2, 0.5)
    L2 = ours2.layers(pts)
    d2, _ = ours2.b0(L2)
    np.testing.assert_allclose(d2, W.compose_box(L2["free"], L2["in_free"], L2["coarse"]))


def test_composed_scene_b0_without_free_matches_numpy_on_cuda(T63):
    pytest.importorskip("curobo")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA")
    from AG3S.trajopt.curobo_v2.composed_world import ComposedSceneCollision
    from AG3S.trajopt.curobo_v2.robot_world import device_cfg

    rng = np.random.default_rng(12)
    n = 50
    cw = ComposedSceneCollision.build("b0", n, device_cfg(), capacity_shape=(32, 32, 32),
                                      voxel_size=0.0125)
    for with_free in (False, True, False):
        z = _synthetic_record(rng, with_free)
        info = cw.load_chunk(z, np.eye(4))
        assert info["box_layer"] == ("free" if with_free else "main_fine")
        assert float(cw.box_min_coarse.item()) == (0.0 if with_free else 1.0)
        pts = rng.uniform(-0.3, 0.3, (20 * n, 3))
        ours = T63.OurWorlds(z, 0.5)
        L = ours.layers(pts)
        d_ref, _ = ours.b0(L)
        sph = np.concatenate([pts, np.full((len(pts), 1), 0.004)], axis=1)
        sd = cw.signed_distance(sph, part="world")
        sd = np.where(np.isfinite(sd), sd, 0.5)
        # rows at least one voxel inside / outside every lattice border (cuRobo weights in-bounds
        # corners only at the border; our field clamps — a known structural difference)
        a = ours.arr["main"] if not with_free else ours.arr["free"]
        m_box = T63.lattice_margin_vox(pts, a)
        m_c = T63.lattice_margin_vox(pts, ours.arr["coarse"])
        interior = (np.abs(m_box) > 1.0) & (m_c > 1.0)
        assert interior.sum() > 200
        assert np.abs(sd - d_ref)[interior].max() < 1e-4


# =================================================================================================
# step3_eval helpers that need no snapshot (context rebuild, cuRobo outputs, summary)
# =================================================================================================

@pytest.fixture(scope="module")
def SE():
    import importlib.util
    import pathlib

    import AG3S.trajopt.curobo_v2 as pkg

    path = pathlib.Path(pkg.__file__).parent / "step3_eval.py"
    spec = importlib.util.spec_from_file_location("t63_step3_eval_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rec(path, k, t_step, refined, reference, summary):
    import json

    np.savez(path / f"chunk_{k:05d}.npz", chunk_index=np.asarray(k), t_step=np.asarray(t_step),
             refined_chunk=refined.astype(np.float32), reference_chunk=reference.astype(np.float32),
             summary_json=np.asarray(json.dumps(summary)))


def test_resolve_context_is_the_server_context(SE, tmp_path):
    rng = np.random.default_rng(21)
    prev_ref, prev_fin = rng.normal(size=(50, 16)), rng.normal(size=(50, 16))
    cols = [*range(7), *range(8, 15)]
    applied = prev_fin[:8][:, cols]
    d = applied[-1] - prev_ref.astype(np.float32).astype(np.float64)[7][cols]
    _rec(tmp_path, 9, 5, prev_fin, prev_ref, {})
    s10 = {"t_step": 6,
           "exec_continuity": {"used": True, "mode": "feedback", "previous_chunk": "refined",
                               "aligned_from_step": 8, "n_exec": 8, "plan_seq": 5},
           "exec_feedback": {"available": True, "seq": 5, "executed": [True] * 8,
                             "applied_arm": applied.tolist()},
           "ta_history": {"deflection": {"row": 7, "max_abs_rad": float(np.abs(d).max())}}}
    _rec(tmp_path, 10, 6, rng.normal(size=(50, 16)), rng.normal(size=(50, 16)), s10)
    ctx, note = SE.resolve_context(tmp_path / "chunk_00010.npz", s10, 10)
    np.testing.assert_array_equal(ctx["previous_physical_chunk"],
                                  prev_fin.astype(np.float32).astype(np.float64))
    assert ctx["previous_executed_steps"] == 8 and ctx["t_step"] == 6
    np.testing.assert_allclose(ctx["deflection_history"], d, atol=1e-6)
    assert note["continuity_used_by_server"]
    # HOLD before (n_exec 0): the cached plan is passed with 0 executed steps -> refiner drops it
    s11 = {"t_step": 7, "exec_continuity": {"used": False, "mode": "feedback",
                                            "previous_chunk": None, "n_exec": 0, "plan_seq": 6}}
    _rec(tmp_path, 11, 7, rng.normal(size=(50, 16)), rng.normal(size=(50, 16)), s11)
    ctx, _ = SE.resolve_context(tmp_path / "chunk_00011.npz", s11, 11)
    assert ctx["previous_physical_chunk"] is None and "deflection_history" not in ctx
    # switched off by the server for this chunk
    s12 = {"t_step": 8, "exec_continuity": {"used": False, "mode": "feedback", "off_by": "grasp"}}
    _rec(tmp_path, 12, 8, rng.normal(size=(50, 16)), rng.normal(size=(50, 16)), s12)
    ctx, _ = SE.resolve_context(tmp_path / "chunk_00012.npz", s12, 12)
    assert ctx["continuity_off"] == "grasp" and "previous_physical_chunk" not in ctx


def test_curobo_outputs_are_read_by_episode_and_chunk(SE, tmp_path):
    import json

    root = tmp_path / "cu"
    (root / "trajectories" / "B1_lam100").mkdir(parents=True)
    q = np.random.default_rng(2).normal(size=(2, 50, 14))
    np.savez(root / "trajectories" / "B1_lam100" / "ep7.npz", chunk_index=np.array([30, 50]),
             position=np.array([0, 20]), cspace=q, chunk=np.zeros((2, 50, 16)),
             success=np.array([True, False]), finite=np.array([True, True]))
    rows = [{"candidate": "B1_lam100", "episode": 7, "chunk_index": c, "success": s,
             "wall_ms": 50.0} for c, s in ((30, True), (50, False))]
    (root / "curobo_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    cu = SE.CuroboOutputs(root)
    assert cu.names == ["B1_lam100"]
    np.testing.assert_array_equal(cu.trajectory("B1_lam100", 7, 50), q[1])
    assert cu.trajectory("B1_lam100", 7, 40) is None and cu.trajectory("B2_lam1", 7, 30) is None
    assert cu.rows[("B1_lam100", 7, 50)]["success"] is False
    assert SE.candidate_order(cu.names, True) == ["reference", "to_recorded", "to_resolved",
                                                  "B1_lam100"]
