"""T63b — cuRobo B0 in the serving path (`curobo_v2/serving.py`, `serve_safe --to-backend`).

What this file pins (CPU unless marked):

1. `--to-backend` is off by default and refuses flag combinations it cannot serve.
2. `live_layers` takes the field's tiers by the recorder's naming rule (`field_layers`).
3. `certify` = the SQP path's own certification: on the trajectory an SQP solve returned, a fresh
   optimizer's `certify` gives the same status, violation, worst row, look-ahead record, cost and
   chunk; on an empty scene it is the same UNCONSTRAINED passthrough.
4. `CuroboB0Backend.refine` (fake solver): certified cuRobo chunk becomes the refined chunk (all 50
   rows); a cuRobo `success=False` is *not* a HOLD by itself; a non-finite trajectory / exception /
   field-less scene returns the reference as `solver_failed`; every chunk is logged.
5. The refiner hook (only in the serving snapshot `snapshot_cr0`): backend `None` = the SQP call,
   a backend replaces exactly that call.
6. (CUDA + cuRobo) the device float32 -> float16 rounding of a live grid equals the recorder's
   numpy rounding byte for byte, and `load_layers` (live) loads the same world as `load_chunk`
   (record).
"""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

from AG3S.trajopt.curobo_v2 import serving as SV

osqp = pytest.importorskip("osqp")


# =================================================================================================
# helpers
# =================================================================================================

def _scenario():
    from tests.trajopt.fixtures import make_scenario

    return make_scenario(horizon=10, penetration=0.03)


def _rich(sc):
    from tests.trajopt.test_t43tp_fast_to import _rich_scene

    return _rich_scene(sc, held=False)


def _la_config(sc, gate=0.5):
    """C1aN-like TO settings on the small fixture: look-ahead from step 4 (of 10) with a gate,
    deflection rate (B), esdf backend, QP eps; no servo accel (C1aN has none)."""
    return sc.config.with_overrides({
        "horizon": {"execution_length": 4, "lookahead": "obstacles", "lookahead_gate": gate},
        "collision": {"backend": "esdf", "obstacle_margin": 0.01},
        "cost": {"w_deflection_rate": 20.0},
        "qp": {"eps_abs": 1e-4, "eps_rel": 1e-4},
        "sqp": {"fast_eval": True}})


def _args(**kw):
    base = dict(to_backend="curobo-b0", no_safe=False, no_perception=False,
                esdf_backend="curobo", shadow=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


# =================================================================================================
# 1. CLI
# =================================================================================================

def test_flag_default_is_sqp_and_bad_combinations_are_refused():
    assert SV.TO_BACKENDS[0] == "sqp"
    assert SV.check_to_backend_args(_args(to_backend="sqp", no_safe=True)) is None
    assert SV.check_to_backend_args(_args()) is None
    for bad in (dict(no_safe=True), dict(no_perception=True), dict(esdf_backend="legacy"),
                dict(shadow=True), dict(to_backend="nope")):
        assert SV.check_to_backend_args(_args(**bad))


def test_serve_safe_parser_default_is_sqp_when_present():
    from AG3S.trajopt import serve_safe

    ap = serve_safe.build_parser()
    base = ["--config", "x", "--checkpoint", "y", "--model-xml", "m.xml"]
    args = ap.parse_args(base)
    if not hasattr(args, "to_backend"):
        pytest.skip("this tree's serve_safe has no --to-backend (only the serving snapshot does)")
    assert args.to_backend == "sqp"
    assert ap.parse_args(base + ["--to-backend", "curobo-b0"]).to_backend == "curobo-b0"
    # the TO configuration does not depend on the flag
    a = serve_safe.trajopt_config_from_args(ap.parse_args(base + ["--esdf-margin", "0"]))
    b = serve_safe.trajopt_config_from_args(ap.parse_args(
        base + ["--esdf-margin", "0", "--to-backend", "curobo-b0"]))
    assert a == b


# =================================================================================================
# 2. live layers
# =================================================================================================

def test_live_layers_follow_the_recorder_naming():
    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

    sc = _scenario()
    field = _rich(sc).esdf
    out = SV.live_layers(field)
    named = dict(ConstraintRecordWriter.field_layers(field))
    assert set(out) == {"coarse", "main"}                 # no target-free tier in this field
    for ours, theirs in (("coarse", "coarse"), ("main", "")):
        layer = named[theirs]
        assert np.array_equal(out[ours]["origin"], np.asarray(layer.grid.origin, np.float64))
        assert out[ours]["voxel_size"] == float(layer.grid.voxel_size)
        assert np.array_equal(np.asarray(out[ours]["distance"]), np.asarray(layer.distance_grid))
    field.target_free_layers = (field.layers[-1],)
    assert SV.live_layers(field)["free"]["voxel_size"] == float(field.layers[-1].grid.voxel_size)
    with pytest.raises(ValueError, match="two-tier"):
        SV.live_layers(types.SimpleNamespace(layers=(field.layers[0],)))


def test_finger_values_take_step_zero_of_the_param_path():
    lin = types.SimpleNamespace(_param_path=np.array([[0.01, 0.02], [0.03, 0.04]]))
    robot = types.SimpleNamespace(param_joint_names=("a", "b"),
                                  param_vector=lambda: np.array([9.0, 8.0]))
    assert SV.finger_values(lin, robot) == ({"a": 0.01, "b": 0.03}, "param_path[:, 0]")
    lin._param_path = None
    assert SV.finger_values(lin, robot) == ({"a": 9.0, "b": 8.0}, "model param_vector")


def test_failure_reasons_are_the_active_constraints():
    diag = {"curobo_constraints": {"scene_collision": {"sum": 1.0, "n_steps_positive": 3},
                                   "cspace": {"sum": 0.0, "n_steps_positive": 0},
                                   "joint_limit": {"sum": 2.0, "n_steps_positive": 1}}}
    assert SV.failure_reasons(diag) == ["joint_limit", "scene_collision"]
    assert SV.failure_reasons({}) == []


# =================================================================================================
# 3. certify == the SQP path's certification
# =================================================================================================

def _same_certification(a, b, *, gate_a):
    from AG3S.trajopt.types import TrajOptStatus

    # FEASIBLE = "the SQP loop stopped on its budget / iteration cap" — a property of the loop, not
    # of the check: the same clear trajectory is OPTIMAL when no loop ran (`iterations = 0`).
    assert b.status is (TrajOptStatus.OPTIMAL if a.status is TrajOptStatus.FEASIBLE else a.status)
    assert b.status.safe == a.status.safe
    assert b.max_violation == a.max_violation
    assert b.reference_violation == a.reference_violation
    assert b.cost == a.cost
    assert b.chunk.tobytes() == a.chunk.tobytes()
    assert b.trajectory.tobytes() == a.trajectory.tobytes()
    for key in ("clearance_m", "max_violation_pair", "limit_overshoot", "deflection_rate",
                "held_rows", "geometry_certified", "collision_enabled"):
        assert b.metrics.get(key) == a.metrics.get(key), key
    la, lb = dict(a.metrics["lookahead"]), dict(b.metrics["lookahead"])
    assert la.pop("gate") == gate_a and lb.pop("gate") == "not run (external optimizer)"
    for k in ("gate_m", "tail_reference_clearance_m", "tail_reference_worst_step"):
        la.pop(k, None)
    assert la == lb


@pytest.mark.parametrize("deflection", [None, 0.003])
def test_certify_reproduces_the_sqp_certification_of_the_same_trajectory(deflection):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    sc = _scenario()
    scene = _rich(sc)
    cfg = _la_config(sc)
    hist = None if deflection is None else np.full(sc.layout.nq_opt, deflection)
    prev = sc.reference + 0.002
    a = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg).solve(
        sc.reference, sc.q_now, scene, template=sc.chunk, previous_chunk=prev,
        deflection_history=hist)
    assert a.metrics["lookahead"]["gate"] == "open"
    fresh = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)
    b = SV.certify(fresh, a.trajectory, sc.reference, sc.q_now, scene, template=a.chunk,
                   previous_chunk=prev, deflection_history=hist)
    _same_certification(a, b, gate_a="open")
    assert b.iterations == 0 and b.metrics["sqp_iterations"] == 0
    assert b.metrics["returned"] == "initial" and b.metrics["n_candidates"] == 0


def test_certify_flags_a_colliding_trajectory_as_violated():
    """The reference penetrates (fixture): certified as is, it is VIOLATED with the SQP path's
    number for that same trajectory (`reference_violation` of the window)."""
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from AG3S.trajopt.types import TrajOptStatus

    sc = _scenario()
    scene = _rich(sc)
    cfg = _la_config(sc)
    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)
    b = SV.certify(opt, sc.reference, sc.reference, sc.q_now, scene, template=sc.chunk)
    window = -float(opt.linearizer.step_clearances(sc.reference, sc.q_now, scene)[:4].min())
    if window > cfg.safety.violation_tolerance:
        assert b.status is TrajOptStatus.VIOLATED
        assert b.max_violation == pytest.approx(window, abs=1e-12)
    else:
        assert b.status.safe
    assert b.reference_violation == pytest.approx(max(0.0, window), abs=1e-12)


def test_certify_on_an_empty_scene_is_the_sqp_passthrough():
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from AG3S.trajopt.types import TrajOptStatus
    from tests.trajopt.fixtures import empty_scene

    sc = _scenario()
    cfg = _la_config(sc)
    o1 = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)
    o2 = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)
    scene = empty_scene(o1.linearizer)
    a = o1.solve(sc.reference, sc.q_now, scene, template=sc.chunk)
    b = SV.certify(o2, sc.reference + 0.1, sc.reference, sc.q_now, scene, template=sc.chunk)
    assert a.status is b.status is TrajOptStatus.UNCONSTRAINED
    assert a.chunk.tobytes() == b.chunk.tobytes() and a.notes == b.notes


# =================================================================================================
# 4. the backend with a fake solver
# =================================================================================================

class _FakeScene:
    def __init__(self):
        self.loaded = []

    def load_layers(self, layers, T):
        self.loaded.append((sorted(layers), np.asarray(T)))
        return {"has_free": "free" in layers, "box_layer": "free" if "free" in layers
                else "main_fine"}


class _FakeSolver:
    def __init__(self, out_fn):
        self.scene = _FakeScene()
        self.fingers = []
        self.out_fn = out_fn
        self.calls = []

    def set_fingers(self, values):
        self.fingers.append(dict(values))

    def solve(self, q14, chunk):
        self.calls.append((np.array(q14), np.array(chunk)))
        return self.out_fn(q14, chunk)


def _backend(sc, out_fn, tmp_path):
    spec = {"arm_joint_names": [str(sc.robot.joint_names[i]) for i in sc.layout.q_indices],
            "finger_joints": {}}
    return SV.CuroboB0Backend(spec, sc.robot, sc.layout, log_dir=tmp_path,
                              solver=_FakeSolver(out_fn))


def _ok(shift=0.0, success=True):
    def fn(q14, chunk):
        c = np.array(chunk, np.float64)
        c[:, :7] += shift
        return {"success": success, "chunk": c, "finite": True, "wall_ms": 20.0,
                "solve_time_ms": 15.0, "cold": False, "goal_error_rad": 0.0,
                "curobo_constraints": {"scene_collision": {"sum": 0.0 if success else 1.0,
                                                           "n_steps_positive": 0 if success else 2}}}
    return fn


@pytest.mark.parametrize("success", [True, False])
def test_backend_certifies_the_curobo_chunk_and_uses_all_its_rows(tmp_path, success):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    sc = _scenario()
    scene = _rich(sc)
    cfg = _la_config(sc)
    be = _backend(sc, _ok(shift=1e-4, success=success), tmp_path)
    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)
    planned = 10
    res = be.refine(opt, sc.chunk, sc.reference[:, :planned], sc.q_now, scene, t_step=7)
    expected = np.array(sc.chunk)
    expected[:, :7] += 1e-4
    assert res.chunk.tobytes() == expected.tobytes()               # every row is cuRobo's
    ref = SV.certify(TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg),
                     sc.layout.chunk_to_trajectory(expected), sc.reference, sc.q_now, scene,
                     template=expected)
    assert res.status is ref.status and res.max_violation == ref.max_violation
    assert res.metrics["to_backend"] == "curobo-b0"
    rec = json.loads(res.metrics["curobo_b0"])
    assert rec["success"] is success and rec["ran"] is True and rec["box_layer"] == "main_fine"
    assert rec["failure_reasons"] == ([] if success else ["scene_collision"])
    line = json.loads((tmp_path / "curobo_b0_chunks.jsonl").read_text().splitlines()[-1])
    assert line["t_step"] == 7 and line["status"] == res.status.value
    # q_now handed to cuRobo = the layout's arm joints
    assert np.array_equal(be.solver.calls[0][0], sc.q_now[sc.layout.q_indices])


@pytest.mark.parametrize("mode", ["nonfinite", "raises", "no_field"])
def test_backend_failures_return_the_reference_as_solver_failed(tmp_path, mode):
    import dataclasses

    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from AG3S.trajopt.types import TrajOptStatus

    sc = _scenario()
    scene = _rich(sc)

    def fn(q14, chunk):
        if mode == "raises":
            raise RuntimeError("boom")
        return {"success": False, "chunk": None, "finite": False, "wall_ms": 1.0}

    if mode == "no_field":
        scene = dataclasses.replace(scene, esdf=None, candidate_active=sc.scene.candidate_active)
    be = _backend(sc, fn, tmp_path)
    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, _la_config(sc))
    res = be.refine(opt, sc.chunk, sc.reference, sc.q_now, scene)
    assert res.status is TrajOptStatus.SOLVER_FAILED
    assert res.chunk.tobytes() == np.asarray(sc.chunk, np.float64).tobytes()
    assert any("UNVERIFIED" in n for n in res.notes)
    assert json.loads(res.metrics["curobo_b0"])["fallback"]


def test_backend_passes_an_empty_scene_through_without_curobo(tmp_path):
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from AG3S.trajopt.types import TrajOptStatus
    from tests.trajopt.fixtures import empty_scene

    sc = _scenario()
    be = _backend(sc, _ok(), tmp_path)
    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, _la_config(sc))
    res = be.refine(opt, sc.chunk, sc.reference, sc.q_now, empty_scene(opt.linearizer))
    assert res.status is TrajOptStatus.UNCONSTRAINED and not be.solver.calls


# =================================================================================================
# 5. the refiner hook (serving snapshot only)
# =================================================================================================

def _refiner(sc, cfg, scene):
    from AG3S.trajopt.refiner import TrajOptChunkRefiner

    return TrajOptChunkRefiner(sc.robot, sc.layout, lambda ctx: (scene, sc.q_now, True), cfg)


def test_refiner_hook_off_is_the_sqp_call_and_on_replaces_only_it():
    sc = _scenario()
    scene = _rich(sc)
    cfg = _la_config(sc).with_overrides({"horizon": {"horizon": 10, "plan_horizon": None}})
    r_off = _refiner(sc, cfg, scene)
    if not hasattr(r_off, "backend"):
        pytest.skip("this tree's refiner has no backend hook (only the serving snapshot does)")
    assert r_off.backend is None
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    planned = r_off.config.horizon.planned
    ctx = {"t_step": 3, "deflection_history": np.full(sc.layout.nq_opt, 0.001)}
    out_off = r_off.refine(sc.chunk, dict(ctx))
    direct = TrajectoryOptimizer(sc.robot, sc.layout, r_off.limits, r_off.optimizer.config).solve(
        sc.layout.chunk_to_trajectory(sc.chunk)[:, :planned], sc.q_now, scene, template=sc.chunk,
        deflection_history=ctx["deflection_history"])
    assert out_off.tobytes() == direct.chunk.tobytes()

    seen = {}

    class B:
        def refine(self, optimizer, chunk, reference, q_now, scene_, **kw):
            seen.update(kw, optimizer=optimizer, chunk=chunk, reference=reference)
            return direct

        def reset(self):
            seen["reset"] = True

    r_on = _refiner(sc, cfg, scene)
    r_on.backend = B()
    out_on = r_on.refine(sc.chunk, dict(ctx))
    assert out_on.tobytes() == direct.chunk.tobytes()
    assert seen["optimizer"] is r_on.optimizer and seen["t_step"] == 3
    assert seen["reference"].shape == (sc.layout.nq_opt, planned)
    assert np.array_equal(seen["deflection_history"], ctx["deflection_history"])
    assert r_on.last_result is direct
    r_on.reset()
    assert seen["reset"] is True


# =================================================================================================
# 6. CUDA + cuRobo
# =================================================================================================

def _cuda():
    torch = pytest.importorskip("torch")
    pytest.importorskip("curobo")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    return torch


def test_device_float16_rounding_equals_the_recorder_bytes():
    torch = _cuda()
    rng = np.random.default_rng(0)
    vals = np.concatenate([rng.normal(0, 0.3, 200000), rng.uniform(-1e-6, 1e-6, 1000),
                           [0.0, -0.0, 1e-8, 6e-5, 0.5, 65504.0, 1e5]]).astype(np.float32)
    dev = torch.as_tensor(vals, device="cuda").to(torch.float16).cpu().numpy()
    assert dev.tobytes() == np.asarray(vals, np.float16).tobytes()


def test_load_layers_with_device_grids_equals_load_chunk_of_the_record():
    torch = _cuda()
    from AG3S.trajopt.curobo_v2.composed_world import ComposedSceneCollision
    from AG3S.trajopt.curobo_v2.robot_world import device_cfg

    dev = device_cfg()
    rng = np.random.default_rng(1)
    z = {"esdf_coarse_distance": rng.normal(0.2, 0.1, (40, 40, 40)).astype(np.float16),
         "esdf_coarse_origin": np.array([-0.3, -0.4, 0.1]), "esdf_coarse_voxel_size": 0.02,
         "esdf_distance": rng.normal(0.1, 0.05, (60, 64, 50)).astype(np.float16),
         "esdf_origin": np.array([0.2, -0.1, 0.5]), "esdf_voxel_size": 0.005,
         "esdf_free_distance": rng.normal(0.1, 0.05, (60, 64, 50)).astype(np.float16),
         "esdf_free_origin": np.array([0.2, -0.1, 0.5]), "esdf_free_voxel_size": 0.005}
    T = np.eye(4)
    T[:3, 3] = [0.01, -0.02, 0.9]
    for with_free in (True, False):
        zz = {k: v for k, v in z.items() if with_free or not k.startswith("esdf_free")}
        a = ComposedSceneCollision.build("b0", 270, dev)
        b = ComposedSceneCollision.build("b0", 270, dev)
        ia = a.load_chunk(zz, T)
        live = {"coarse": {"distance": torch.as_tensor(z["esdf_coarse_distance"].astype(np.float32),
                                                       device="cuda"),
                           "origin": z["esdf_coarse_origin"], "voxel_size": 0.02},
                "main": {"distance": torch.as_tensor(z["esdf_distance"].astype(np.float32),
                                                     device="cuda"),
                         "origin": z["esdf_origin"], "voxel_size": 0.005}}
        if with_free:
            live["free"] = {"distance": torch.as_tensor(z["esdf_free_distance"].astype(np.float32),
                                                        device="cuda"),
                            "origin": z["esdf_free_origin"], "voxel_size": 0.005}
        ib = b.load_layers(live, T)
        assert ia == ib
        for name in a.channels:
            va, vb = a.channels[name].data.voxels, b.channels[name].data.voxels
            for attr in ("features", "params", "dims", "inv_pose", "enable"):
                assert torch.equal(getattr(va, attr), getattr(vb, attr)), (name, attr)
        assert torch.equal(a.box_min_coarse, b.box_min_coarse)
        for k in a.boxes:
            assert all(torch.equal(x, y) for x, y in zip(a.boxes[k], b.boxes[k]))
