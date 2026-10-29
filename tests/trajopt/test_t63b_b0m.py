"""T63b B0m (user decision 2) and the physical clearance metric.

What this file pins (CPU unless marked):

1. `b0m.classify` is our TO's `CollisionLinearizer._obstacle_rows` on robot rows (target with the
   contact permission, support planes, obstacle), for strict / non-strict, held spheres / points,
   `nonauthorized_target_rows`, `obstacle_margin_support`.
2. With no contact margin and `esdf_margin` 0, `d − r − b0m.margin(class)` **is** our TO's certified
   clearance (`_esdf_clearance_core`) row for row — the B0m rule mapping.
3. `rules_from_scene` refuses what it does not implement (`target_volume`, too many planes).
4. `step3_eval.physical_scene`: the clearance becomes `d − r` with `d` the row's field answer of the
   certified scene (target-free layer on mask rows), never above the certified clearance.
5. `step3_metrics` physical keys and their aggregation (rows written before the metric count as not
   evaluated); `step3_eval.SavedResolve` / multi-dir `CuroboOutputs`.
6. Serving `--to-backend curobo-b0m`: refused without `--obstacle-margin`; the backend hands cuRobo
   the rule of the live scene in cuRobo's sphere order and records it; B0 refuses a rule.
7. (CUDA + cuRobo) `ComposedSceneCollision` mode `b0m`: classes == numpy, signed distance ==
   `d_b0 − margin`, and its cost and gradient == the `b0` world queried at radius `r + margin`.
"""

from __future__ import annotations

import dataclasses
import json
import types

import numpy as np
import pytest

from AG3S.trajopt.curobo_v2 import b0m as B0M


# =================================================================================================
# helpers
# =================================================================================================

def _scenario():
    pytest.importorskip("osqp")       # the TO fixtures (not in .venv-curobo, which runs part 7)
    from tests.trajopt.fixtures import make_scenario

    return make_scenario(horizon=10, penetration=0.03)


def _field(sc):
    from tests.trajopt.test_t43tp_fast_to import _field as f

    return f(sc)


def _scene(sc, *, strict=True, held=False, nonauth=False, support_flag=False, contact_frac=0.5,
           seed=0):
    rng = np.random.default_rng(seed)
    n = sc.linearizer.n_spheres
    kw = dict(candidate_active=np.zeros_like(sc.scene.candidate_active), esdf=_field(sc),
              esdf_margin=0.0, obstacle_margin=0.01, obstacle_margin_support=support_flag,
              support_planes=np.array([[0.0, 0.0, 1.0, -0.02], [1.0, 0.0, 0.0, 0.3]]),
              target_permission_strict=strict,
              target_contact_rows=rng.uniform(size=n) < contact_frac)
    T = np.asarray(sc.robot.link_pose(sc.q_now, "ee_finger_l1"))
    if held:
        pts = np.array([[0.0, 0.0, 0.03], [0.01, 0.0, 0.05]])
        kw.update(manipulated_spheres=pts @ T[:3, :3].T + T[:3, 3],
                  manipulated_sphere_radii=np.array([0.01, 0.012]))
    else:
        kw.update(manipulated_points=T[:3, 3] + rng.normal(0, 0.02, (40, 3)))
    if nonauth:
        kw["nonauthorized_target_rows"] = rng.uniform(size=n) < 0.3
    return dataclasses.replace(sc.scene, **kw)


def _lin(sc):
    from AG3S.trajopt.linearize import CollisionLinearizer

    return CollisionLinearizer(sc.robot, sc.layout, sc.reference.shape[1])


def _rows_near_surfaces(sc, scene, n_steps=6, seed=1):
    """Sphere centres of the reference plus perturbations, and the field distance of each."""
    lin = _lin(sc)
    centres = np.asarray(lin.sphere_states(sc.reference, sc.q_now)[0], np.float64)[:n_steps]
    centres = centres[:, :lin.n_spheres]
    flat = centres.reshape(-1, 3)
    d = np.asarray(scene.esdf.distance(flat), np.float64).reshape(centres.shape[:2])
    return lin, centres, flat, d


# =================================================================================================
# 1-3. the rule
# =================================================================================================

@pytest.mark.parametrize("strict,held,nonauth,support_flag", [
    (True, False, False, False), (False, False, False, False), (True, True, False, False),
    (True, False, True, False), (True, False, False, True)])
def test_classify_is_our_obstacle_rows(strict, held, nonauth, support_flag):
    sc = _scenario()
    scene = _scene(sc, strict=strict, held=held, nonauth=nonauth, support_flag=support_flag)
    lin, centres, flat, d = _rows_near_surfaces(sc, scene)
    rng = np.random.default_rng(3)
    # put rows on every edge: the target test, the support test, neither
    d_obj = lin._manipulated_distance(flat, scene).reshape(d.shape)
    s0 = flat @ np.array([0.0, 0.0, 1.0]) + 0.02
    pick = rng.uniform(size=d.shape)
    d = np.where(pick < 0.3, d_obj + rng.uniform(-0.007, 0.007, d.shape),
                 np.where(pick < 0.6, s0.reshape(d.shape) + rng.uniform(-0.012, 0.012, d.shape), d))
    ours = lin._obstacle_rows(d, flat, scene, d_obj)
    rules = B0M.rules_from_scene(scene, lin.n_spheres)
    got = B0M.classify(d, centres, B0M.target_distance(flat, rules["target"]).reshape(d.shape),
                       rules)
    assert np.array_equal(got, ours)
    assert len(set(np.unique(ours).tolist())) >= (2 if support_flag else 3)


def test_b0m_margin_is_our_certified_margin_without_contact_margins():
    """`d − r − margin(class)` == `_esdf_clearance_core` on a scene without contact margins (B0 has
    none) — the B0m rule mapping, row for row."""
    sc = _scenario()
    scene = _scene(sc)
    lin, centres, flat, d = _rows_near_surfaces(sc, scene)
    full = np.asarray(lin.sphere_states(sc.reference, sc.q_now)[0], np.float64)
    c_to = lin._esdf_clearance_core(full, scene)[..., 0][:, :lin.n_spheres]
    rules = B0M.rules_from_scene(scene, lin.n_spheres)
    allc = full[:, :lin.n_spheres]
    fl = allc.reshape(-1, 3)
    dd = np.asarray(scene.esdf.distance(fl), np.float64).reshape(allc.shape[:2])
    cls = B0M.classify(dd, allc, B0M.target_distance(fl, rules["target"]).reshape(dd.shape), rules)
    c_b0m = dd - lin.robot_radii[None] - B0M.margin(cls, rules["obstacle_margin"])
    np.testing.assert_array_equal(c_b0m, c_to)
    assert (cls == B0M.OBSTACLE).any() and (cls != B0M.OBSTACLE).any()


def test_rules_from_scene_reads_the_scene_and_refuses_what_it_lacks():
    sc = _scenario()
    n = sc.linearizer.n_spheres
    scene = _scene(sc, nonauth=True)
    r = B0M.rules_from_scene(scene, n)
    assert r["obstacle_margin"] == 0.01 and r["target_tol"] == float(scene.esdf.grid.voxel_size)
    assert r["support_tol"] == 2.0 * r["target_tol"] and r["planes"].shape == (2, 4)
    assert np.array_equal(r["contact"], scene.target_contact_rows & ~scene.nonauthorized_target_rows)
    assert r["target"]["kind"] == "points"
    assert B0M.rules_from_scene(_scene(sc, strict=False), n)["contact"].all()
    assert B0M.rules_from_scene(_scene(sc, support_flag=True), n)["planes"].shape == (0, 4)
    assert B0M.rules_from_scene(_scene(sc, held=True), n)["target"]["kind"] == "spheres"
    perm = np.random.default_rng(0).permutation(n)
    assert np.array_equal(B0M.to_curobo(r, perm)["contact"], r["contact"][perm])
    with pytest.raises(NotImplementedError):
        B0M.rules_from_scene(dataclasses.replace(scene, target_volume=np.zeros(5)), n)
    with pytest.raises(ValueError):
        B0M.rules_from_scene(dataclasses.replace(
            scene, support_planes=np.tile([0.0, 0.0, 1.0, 0.0], (B0M.MAX_PLANES + 1, 1))), n)
    assert B0M.margin(np.array([0, 1, 2]), 0.01).tolist() == [0.01, 0.0, 0.0]
    assert B0M.counts(np.array([0, 0, 2])) == {"obstacle": 2, "target": 0, "support": 1}


# =================================================================================================
# 4-5. the physical clearance
# =================================================================================================

@pytest.fixture(scope="module")
def SE():
    import importlib.util
    import pathlib

    import AG3S.trajopt.curobo_v2 as pkg

    path = pathlib.Path(pkg.__file__).parent / "step3_eval.py"
    spec = importlib.util.spec_from_file_location("t63b_step3_eval_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("with_mask", [False, True])
def test_physical_scene_is_distance_minus_radius_with_the_contact_policy_kept(SE, with_mask):
    sc = _scenario()
    n = sc.linearizer.n_spheres
    scene = _scene(sc)
    field = scene.esdf
    rng = np.random.default_rng(5)
    mask = rng.uniform(size=n) < 0.4
    kw = dict(manipulated_link_margin=np.where(mask, 0.02, 0.05), esdf_margin=0.0)
    if with_mask:
        # a target-free layer that differs from the main field: mask rows must read it
        field.target_free_layers = (field.layers[-1],)
        kw["target_free_mask"] = mask
    scene = dataclasses.replace(scene, **kw)
    phys = SE.physical_scene(scene)
    assert phys.obstacle_margin == 0.0 and phys.manipulated_link_margin is None
    assert phys.esdf_margin == 0.0 and phys.destination_margin == 0.0
    assert phys.target_free_mask is scene.target_free_mask
    assert np.array_equal(phys.target_contact_rows, scene.target_contact_rows)
    lin = _lin(sc)
    cert = lin.esdf_clearance(sc.reference, sc.q_now, scene)[..., 0][:, :n]
    ph = lin.esdf_clearance(sc.reference, sc.q_now, phys)[..., 0][:, :n]
    sh = types.SimpleNamespace(lin=lin)
    d = SE.row_distance(sh, sc.reference.T, sc.q_now, scene)
    np.testing.assert_array_equal(ph, d - lin.robot_radii[None])
    assert np.all(ph >= cert - 1e-15)
    assert (ph > cert + 1e-6).any()          # some margin was removed


def test_physical_metrics_and_their_aggregation():
    from AG3S.trajopt.curobo_v2 import step3_metrics as M  # pure numpy

    lim = M.step_limits({"lower": [-3.0] * 2, "upper": [3.0] * 2, "velocity": [10.0] * 2,
                         "acceleration": [100.0] * 2})
    q = np.zeros((50, 2))
    cert = np.full((50, 4), 0.004)
    cert[3, 1] = -0.006                                   # certified violation in exec8
    phys = cert + 0.010                                   # physically clear (4 mm)
    phys[12, 2] = -0.001                                  # physical penetration only in plan16
    a = M.candidate_metrics(q, np.zeros(2), cert, np.zeros((50, 3)), q, np.zeros((50, 3)), lim,
                            clearance_phys_m=phys)
    assert a["penetration_exec8"] and not a["phys_penetration_exec8"]
    assert a["phys_penetration_plan16"]
    assert a["phys_min_clearance_mm_exec8"] == pytest.approx(4.0)
    assert a["phys_min_clearance_mm_plan16"] == pytest.approx(-1.0)
    b = M.candidate_metrics(q, np.zeros(2), cert, np.zeros((50, 3)), q, np.zeros((50, 3)), lim)
    assert "phys_min_clearance_mm_exec8" not in b
    rows = [{"candidates": {"x": {"metrics": a, "solve": None}}},
            {"candidates": {"x": {"metrics": b, "solve": None}}}]
    g = M.aggregate_candidate(rows, "x")
    assert g["n_phys_evaluated"] == 1 and g["n_penetration_exec8"] == 2
    assert g["n_phys_penetration_exec8"] == 0 and g["n_phys_penetration_plan16"] == 1
    assert g["metrics"]["phys_min_clearance_mm_plan16"]["n"] == 1
    assert g["metrics"]["phys_min_clearance_mm_plan16"]["min"] == pytest.approx(-1.0)


def test_saved_resolve_and_merged_curobo_outputs(SE, tmp_path):
    ev = tmp_path / "eval"
    (ev / "trajectories").mkdir(parents=True)
    ch = np.random.default_rng(0).normal(size=(2, 50, 16))
    ch[1] = np.nan
    np.savez(ev / "trajectories" / "ep7.npz", chunk_index=np.array([30, 31]), to_resolved_chunk=ch)
    rows = [{"chunk_index": 30, "to_resolve": {"status": "optimal"},
             "candidates": {"to_resolved": {"solve": {"status": "optimal", "success": True}}}},
            {"chunk_index": 31, "to_resolve": None, "candidates": {}}]
    (ev / "rows_ep7.json").write_text(json.dumps({"rows": rows}))
    sv = SE.SavedResolve(ev)
    q, solve, info = sv.get(7, 30)
    np.testing.assert_array_equal(q, ch[0])
    assert solve["status"] == "optimal" and info["status"] == "optimal"
    assert sv.get(7, 31)[0] is None and sv.get(7, 99) == (None, None, None)
    with pytest.raises(SystemExit):
        SE.SavedResolve(tmp_path / "missing")

    def cu_dir(name, cand):
        root = tmp_path / name
        (root / "trajectories" / cand).mkdir(parents=True)
        np.savez(root / "trajectories" / cand / "ep7.npz", chunk_index=np.array([30]),
                 cspace=np.full((1, 50, 14), 1.0 if cand == "B0" else 2.0))
        (root / "curobo_rows.jsonl").write_text(json.dumps(
            {"candidate": cand, "episode": 7, "chunk_index": 30, "success": True}) + "\n")
        return root

    a, b = cu_dir("a", "B0"), cu_dir("b", "B0m")
    cu = SE.CuroboOutputs([a, b])
    assert cu.names == ["B0", "B0m"]
    assert cu.trajectory("B0m", 7, 30)[0, 0] == 2.0 and cu.trajectory("B0", 7, 30)[0, 0] == 1.0
    assert SE.CuroboOutputs(a).names == ["B0"]
    with pytest.raises(SystemExit):
        SE.CuroboOutputs([a, cu_dir("c", "B0")])


# =================================================================================================
# 6. serving
# =================================================================================================

def _args(**kw):
    base = dict(to_backend="curobo-b0m", no_safe=False, no_perception=False,
                esdf_backend="curobo", shadow=False, obstacle_margin=0.01)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_b0m_flag_needs_an_obstacle_margin():
    from AG3S.trajopt.curobo_v2 import serving as SV

    assert "curobo-b0m" in SV.TO_BACKENDS and SV.BACKEND_WORLD["curobo-b0m"] == "b0m"
    assert SV.check_to_backend_args(_args()) is None
    assert SV.check_to_backend_args(_args(obstacle_margin=0.0))
    assert SV.check_to_backend_args(_args(to_backend="curobo-b0", obstacle_margin=0.0)) is None
    for bad in (dict(no_safe=True), dict(esdf_backend="legacy"), dict(shadow=True)):
        assert "curobo-b0m" in SV.check_to_backend_args(_args(**bad))


def test_serve_safe_parser_accepts_curobo_b0m_when_present():
    """snapshot_cr0m: `--to-backend curobo-b0m` parses, the default stays `sqp`, and the flag does
    not change the TO config (skipped where serve_safe has no `--to-backend`, i.e. this tree)."""
    from AG3S.trajopt import serve_safe

    ap = serve_safe.build_parser()
    base = ["--config", "x", "--checkpoint", "y", "--model-xml", "m.xml", "--esdf-margin", "0",
            "--obstacle-margin", "0.010"]
    args = ap.parse_args(base)
    if not hasattr(args, "to_backend"):
        pytest.skip("this tree's serve_safe has no --to-backend (only the serving snapshot does)")
    assert args.to_backend == "sqp"
    on = ap.parse_args(base + ["--to-backend", "curobo-b0m"])
    assert on.to_backend == "curobo-b0m"
    assert serve_safe.trajopt_config_from_args(args) == serve_safe.trajopt_config_from_args(on)
    from AG3S.trajopt.curobo_v2 import serving as SV

    # every backend the parser offers is one the serving module knows
    choices = next(a.choices for a in ap._actions if a.dest == "to_backend")
    assert tuple(choices) == SV.TO_BACKENDS


class _FakeScene:
    def __init__(self):
        self.loaded = []

    def load_layers(self, layers, T, **kw):
        self.loaded.append(kw)
        return {"has_free": "free" in layers, "box_layer": "main_fine"}


class _FakeSolver:
    def __init__(self):
        self.scene = _FakeScene()

    def set_fingers(self, values):
        pass

    def solve(self, q14, chunk):
        return {"success": True, "chunk": np.array(chunk, np.float64), "finite": True,
                "wall_ms": 20.0, "solve_time_ms": 15.0, "cold": False, "goal_error_rad": 0.0,
                "curobo_constraints": {}}


def _spec(sc, n):
    # one link per sphere, listed in reverse: cuRobo order != ours
    links = [f"l{i}" for i in range(n)]
    return {"arm_joint_names": [str(sc.robot.joint_names[i]) for i in sc.layout.q_indices],
            "finger_joints": {}, "n_spheres": n,
            "spheres": [{"index": i, "link": links[i]} for i in range(n)],
            "collision_link_names": links[::-1], "sphere_link_names": links}


@pytest.mark.parametrize("cfg_margin", [0.01, 0.02])
def test_b0m_backend_hands_curobo_the_live_rule_in_curobo_order(tmp_path, cfg_margin):
    from AG3S.trajopt.curobo_v2 import serving as SV
    from AG3S.trajopt.curobo_v2 import spec as S
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from AG3S.trajopt.types import TrajOptStatus

    sc = _scenario()
    n = sc.linearizer.n_spheres
    scene = _scene(sc)
    spec = _spec(sc, n)
    try:
        perm = S.sphere_permutation(spec)
    except Exception as exc:  # the spec helper reads more keys on some trees
        pytest.skip(f"synthetic spec not accepted by sphere_permutation: {exc}")
    cfg = sc.config.with_overrides({"collision": {"backend": "esdf",
                                                  "obstacle_margin": cfg_margin}})
    be = SV.CuroboB0Backend(spec, sc.robot, sc.layout, log_dir=tmp_path, solver=_FakeSolver(),
                            backend="curobo-b0m")
    assert be.name == "curobo-b0m" and be.world == "b0m"
    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)
    res = be.refine(opt, sc.chunk, sc.reference, sc.q_now, scene, t_step=4)
    line = json.loads((tmp_path / "curobo_b0_chunks.jsonl").read_text().splitlines()[-1])
    if cfg_margin != 0.01:
        # the scene's margin must be the server's: a mismatch is a backend fault -> HOLD
        assert res.status is TrajOptStatus.SOLVER_FAILED and "config" in line["fallback"]
        return
    sent = be.solver.scene.loaded[-1]["b0m_rules"]
    rules = B0M.rules_from_scene(scene, n)
    assert np.array_equal(sent["contact"], rules["contact"][perm])
    assert np.array_equal(sent["planes"], rules["planes"])
    assert sent["obstacle_margin"] == 0.01
    assert line["b0m_rule"]["n_contact_rows"] == int(rules["contact"].sum())
    assert res.metrics["to_backend"] == "curobo-b0m"
    assert json.loads(res.metrics["curobo_b0"])["b0m_rule"]["obstacle_margin_m"] == 0.01


def test_b0_backend_refuses_a_rule_and_b0m_needs_one():
    from AG3S.trajopt.curobo_v2 import serving as SV

    sc = _scenario()
    spec = {"arm_joint_names": [str(sc.robot.joint_names[i]) for i in sc.layout.q_indices],
            "finger_joints": {}}
    be = SV.CuroboB0Backend(spec, sc.robot, sc.layout, solver=_FakeSolver())
    assert be.name == "curobo-b0" and be.world == "b0"
    field = _field(sc)
    with pytest.raises(ValueError, match="B0m"):
        be.solve_chunk(sc.chunk, sc.q_now, field, {}, b0m_rules={"x": 1})
    with pytest.raises(ValueError):
        SV.CuroboB0Backend(spec, sc.robot, sc.layout, solver=_FakeSolver(), backend="nope")


# =================================================================================================
# 7. CUDA + cuRobo
# =================================================================================================

def _cuda():
    torch = pytest.importorskip("torch")
    pytest.importorskip("curobo")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    return torch


def _record(rng, with_free):
    z = {"esdf_coarse_distance": rng.uniform(-0.02, 0.3, (8, 8, 8)).astype(np.float16),
         "esdf_coarse_origin": np.array([-0.4, -0.4, -0.4]), "esdf_coarse_voxel_size": np.array(0.1),
         "esdf_distance": rng.uniform(-0.01, 0.06, (16, 16, 16)).astype(np.float16),
         "esdf_origin": np.array([-0.1, -0.1, -0.1]), "esdf_voxel_size": np.array(0.0125)}
    if with_free:
        z.update({"esdf_free_distance": rng.uniform(0.0, 0.06, (16, 16, 16)).astype(np.float16),
                  "esdf_free_origin": np.array([-0.1, -0.1, -0.1]),
                  "esdf_free_voxel_size": np.array(0.0125)})
    return z


@pytest.mark.parametrize("with_free", [True, False])
def test_composed_b0m_matches_numpy_and_is_b0_at_radius_plus_margin(with_free):
    torch = _cuda()
    from AG3S.trajopt.curobo_v2 import world as W
    from AG3S.trajopt.curobo_v2.composed_world import ComposedSceneCollision
    from AG3S.trajopt.curobo_v2.robot_world import device_cfg
    from AG3S.trajopt.experiments.t63_curobo_compare import OurWorlds, lattice_margin_vox

    dev = device_cfg()
    rng = np.random.default_rng(7)
    n = 40
    z = _record(rng, with_free)
    T = np.eye(4)
    pts = rng.uniform(-0.09, 0.09, (25 * n, 3))
    ours = OurWorlds(z, 0.5)
    L = ours.layers(pts)
    d_b0, _ = ours.b0(L)
    tp = rng.uniform(-0.05, 0.05, (30, 3))
    rules = {"obstacle_margin": 0.01, "target_tol": 0.0125, "support_tol": 0.025,
             "contact": rng.uniform(size=n) < 0.5, "target": {"kind": "points", "points": tp},
             "planes": np.array([[0.0, 0.0, 1.0, -0.05], [0.0, 1.0, 0.0, 0.02]]),
             "support_exempt": True, "strict": True}
    # rows on the edges of both tests
    d_obj = B0M.target_distance(pts, rules["target"])
    cw = ComposedSceneCollision.build("b0m", n, dev, capacity_shape=(64, 64, 64), voxel_size=0.005)
    b0 = ComposedSceneCollision.build("b0", n, dev, capacity_shape=(64, 64, 64), voxel_size=0.005)
    info = cw.load_chunk(z, T, b0m_rules=rules)
    b0.load_chunk(z, T)
    assert info["b0m"]["n_support_planes"] == 2 and info["b0m"]["target"] == "points"
    with pytest.raises(ValueError):
        cw.load_chunk(z, T)                       # B0m refuses a chunk without a rule
    cw.load_chunk(z, T, b0m_rules=rules)
    cls_np = B0M.classify(d_b0.reshape(-1, n), pts.reshape(-1, n, 3), d_obj.reshape(-1, n), rules)
    sph = np.concatenate([pts, np.full((len(pts), 1), 0.004)], axis=1)
    cc = cw.b0m_classes(sph)
    a = ours.arr["main"] if not with_free else ours.arr["free"]
    interior = ((np.abs(lattice_margin_vox(pts, a)) > 1.0)
                & (lattice_margin_vox(pts, ours.arr["coarse"]) > 1.0))
    # away from the rule edges (5 mm target grid vs cKDTree; float16 grids)
    edge_t = np.abs(np.abs(d_b0 - d_obj) - rules["target_tol"])
    s = pts @ rules["planes"][:, :3].T - rules["planes"][:, 3]
    edge_s = np.abs(np.abs(s - d_b0[:, None]) - rules["support_tol"]).min(1)
    clear = interior & (edge_t > 0.003) & (edge_s > 0.001)
    assert clear.sum() > 200
    assert np.array_equal(cc["cls"][clear], cls_np.reshape(-1)[clear])
    assert len(np.unique(cls_np.reshape(-1)[clear])) == 3
    m = B0M.margin(cls_np.reshape(-1), 0.01)
    sd = cw.signed_distance(sph, part="world")
    assert np.abs(sd - (d_b0 - m))[clear].max() < 1e-4
    # cost and gradient == the b0 world at radius r + margin (the classes the optimizer used)
    qt = torch.as_tensor(sph, dtype=dev.dtype, device=dev.device).view(1, -1, n, 4)
    qm = qt.clone()
    qm[..., 3] += torch.as_tensor(cc["margin"].reshape(1, -1, n), dtype=dev.dtype,
                                  device=dev.device)
    from curobo._src.geom.collision.buffer_collision import CollisionBuffer

    w = torch.ones(1, dtype=dev.dtype, device=dev.device)
    act = torch.full((1,), 0.01, dtype=dev.dtype, device=dev.device)
    x1 = qt.clone().requires_grad_(True)
    c1 = cw.get_sphere_distance_raw(x1, CollisionBuffer.from_shape(qt.shape, dev), w, act,
                                    return_loss=True)
    c1.sum().backward()
    x2 = qm.clone().requires_grad_(True)
    c2 = b0.get_sphere_distance_raw(x2, CollisionBuffer.from_shape(qt.shape, dev), w, act,
                                    return_loss=True)
    c2.sum().backward()
    assert torch.equal(c1, c2)
    assert torch.equal(x1.grad[..., :3], x2.grad[..., :3])
    assert float(c1.sum()) > 0.0 and float(x1.grad.abs().sum()) > 0.0
    W.sdf_from_activation_cost  # noqa: B018 - the inversion `signed_distance` uses


def test_target_grid_of_held_spheres_is_their_signed_union_distance():
    torch = _cuda()  # noqa: F841
    from AG3S.trajopt.curobo_v2.composed_world import target_distance_grid

    c = np.array([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0]])
    r = np.array([0.01, 0.015])
    g = target_distance_grid(None, spheres=(c, r), device="cuda:0")
    shape = g["distance"].shape
    axes = [g["origin"][k] + np.arange(shape[k]) * g["voxel_size"] for k in range(3)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    ref = (np.linalg.norm(P[:, None] - c[None], axis=2) - r[None]).min(1)
    np.testing.assert_allclose(g["distance"].reshape(-1), ref, atol=1e-9)
