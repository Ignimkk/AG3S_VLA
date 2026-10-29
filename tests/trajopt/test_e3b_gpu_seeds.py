"""AG3S-PRE-CORE-TO E3-b — E2's seeds in one batched cuRobo L-BFGS (`serve_safe --to-gpu-seeds curobo`,
`sqp.gpu_seeds`, default "off").

Task: `AG3S/ag3s/docs/handoff/AG3S-PRE-CORE-TO.task.md` §6 "E3 구현 명세", row E3-b.

What this file pins:
1. Off is bit-identical: E2 on (`multi_seed` 16) with `gpu_seeds` off gives the SHA-1 measured with the pre-E3-b
   `sqp.py` (a copy with the two E3-b hunks removed, 2026-10-26), the E2 record has no E3-b key, and `gpu_seeds` is
   never imported.
2. Config / CLI: default off, bad values refused, `curobo` needs `multi_seed > 0`; `--to-gpu-seeds` adds a key only
   when on, needs `--to-multi-seed N` and the TO, and refuses a venv without CUDA / cuRobo before the checkpoint.
3. A GPU path that cannot run falls back to E2's CPU seeds — the same trajectory as off — and says why; a failed build
   is not retried on every chunk.
4. (CUDA + cuRobo only) On a stuck ESDF fixture (an obstacle head-on across the swept path, one SQP iteration: the
   standard solve stays at ~47 mm) the GPU seeds resolve the violation; the chosen candidate passes our own full
   check again (independently re-measured here); cuRobo's FK of the served spheres matches ours.
5. (CUDA + cuRobo only) Determinism: two optimizers, each warmed up, give the same trajectory and record for the same
   input; the anchor's goal term reaches the GPU problem.
6. The offline settings and the summary (`e2_multi_seed summarize` compatible).
7. `sqp.gpu_seeds_check` (`--to-gpu-seeds-check`, default "all", added 2026-10-26 by the lead after the E3 offline
   time): config / CLI; (CUDA + cuRobo only) "first-pass" returns the trajectory and choice of "all" with fewer checks,
   and when no checked result resolves it checks every one and still chooses as "all".

GPU tests: run under the shared lock, e.g.
``python3 scripts/setup/gpu_lock.py shared --est-gib 16 --tag e3b-impl -- .venv-openpi-live/bin/python -m pytest
tests/trajopt/test_e3b_gpu_seeds.py``.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import sys
import types

import numpy as np
import pytest

from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError

pytest.importorskip("casadi")
pytest.importorskip("osqp")

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]

#: SHA-1[:16] of E2-on solves (`multi_seed` 16, `gpu_seeds` off) measured 2026-10-26 with the pre-E3-b `sqp.py` (this
#: checkout's file with the two E3-b hunks removed; the current file gives the same values), `OPENBLAS_NUM_THREADS=1`.
#: Per numpy major version: the two venvs (.venv-ag3s numpy 2.4.6, .venv-openpi-live numpy 1.26.4) differ in the last
#: bits of the same solve.
PRE_SHA_BY_NUMPY = {
    "2": {"stuck_e2": "716068c0a1449648", "esdf_stuck_e2": "457dde14cdafb123", "stuck_e1c_e2": "43edae409c92cefd",
          "refiner_e2_1": "86d631b72936a168", "refiner_e2_2": "5d67019264afa313"},
    "1": {"stuck_e2": "2601e464682464bd", "esdf_stuck_e2": "685705e9f6ac0f4e", "stuck_e1c_e2": "d6c9e7094d0aa286",
          "refiner_e2_1": "0e1a67519182269f", "refiner_e2_2": "c9bab2a45550fa4e"},
}
PRE_SHA = PRE_SHA_BY_NUMPY.get(np.__version__.split(".")[0])


def _sha(a) -> str:
    return hashlib.sha1(np.ascontiguousarray(np.asarray(a, np.float64)).tobytes()).hexdigest()[:16]


def _gpu_ok() -> bool:
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    from AG3S.trajopt.gpu_seeds import available

    return available()[0]


needs_gpu = pytest.mark.skipif(not _gpu_ok(), reason="needs torch + CUDA + cuRobo (.venv-openpi-live on the GPU)")


def _strip(rec):
    """A multi-seed record without wall-clock fields (for determinism)."""
    out = {k: v for k, v in rec.items() if k not in ("total_ms", "standard_ms", "gpu")}
    out["candidates"] = [{k: v for k, v in c.items() if k != "solve_ms"} for c in rec.get("candidates", [])]
    return out


# ------------------------------------------------------------------------------------------------ fixtures
def _candidate_stuck():
    """E2's stuck fixture (candidate sphere head-on, primitive rows, one SQP iteration)."""
    from tests.trajopt.test_e2_multi_seed import _stuck

    return _stuck()


@functools.lru_cache(maxsize=None)
def _esdf_stuck(H: int = 8, R: float = 0.10):
    """An ESDF obstacle (sphere of radius R, analytic two-tier field) centred 2 mm off the most-moving sphere's path at
    the middle row of the sweep; the arm sphere model (every sphere below `link_torso_5`); `obstacle_margin` 10 mm,
    one SQP iteration. Measured: the reference penetrates 187 mm, the standard solve leaves 47 mm."""
    from AG3S.ag3s.experiments.reports.grounding_report import ARM_LINKS
    from AG3S.ag3s.fields.curobo_field import CuroboEsdfField, layer_from_arrays
    from AG3S.ag3s.robot_models import DEFAULT_RBY1_JOINTS, load_rby1
    from AG3S.trajopt.limits import build_limits
    from AG3S.trajopt.linearize import CollisionLinearizer
    from AG3S.trajopt.types import ChunkLayout
    from tests.trajopt.fixtures import Q_HOME, empty_scene, small_config, sweep_reference

    robot = load_rby1(link_filter=ARM_LINKS)
    layout = ChunkLayout.rby1(DEFAULT_RBY1_JOINTS)
    cfg = small_config(H, sqp={"max_iterations": 1, "min_iterations": 1},
                       collision={"backend": "esdf", "obstacle_margin": 0.01})
    limits = build_limits(robot, layout, dt=cfg.horizon.dt, config=cfg.limits)
    lin = CollisionLinearizer(robot, layout, H)
    ref = sweep_reference(layout, H)
    q = Q_HOME.copy()
    centres, _ = lin.sphere_states(ref, q)
    mid = H // 2
    s = int(np.argmax(np.linalg.norm(centres[mid] - centres[0], axis=1)))
    d = centres[mid + 1, s] - centres[mid - 1, s]
    d /= np.linalg.norm(d)
    perp = np.cross(d, [0.0, 0.0, 1.0])
    perp /= np.linalg.norm(perp)
    c = centres[mid, s] + perp * 0.002

    def layer(vs, n):
        o = c - vs * (n - 1) / 2.0
        X = np.stack(np.meshgrid(*[o[k] + vs * np.arange(n) for k in range(3)], indexing="ij"), -1)
        return layer_from_arrays(np.clip(np.linalg.norm(X - c, axis=-1) - R, -0.5, 0.5).astype(np.float32), o, vs)

    fine, coarse = layer(0.005, 96), layer(0.02, 64)
    field = CuroboEsdfField((coarse, fine), outside_distance=0.5, target_free_layers=(fine,))
    scene = dataclasses.replace(empty_scene(lin), esdf=field, esdf_margin=0.0, obstacle_margin=0.01)
    return types.SimpleNamespace(robot=robot, layout=layout, config=cfg, limits=limits, linearizer=lin, reference=ref,
                                 q_now=q, scene=scene, centres=centres)


def _opt(sc, n=0, gpu="off", **over):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    sqp = {"multi_seed": n, **({"gpu_seeds": gpu} if gpu != "off" else {})}
    return TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, sc.config.with_overrides({"sqp": sqp, **over}))


def _e1c_anchor(sc):
    from AG3S.trajopt.grasp_anchor import GraspAnchor, LinkPointFK

    p = LinkPointFK(sc.robot, sc.layout, 8, "ee_right")(sc.reference, sc.q_now)[0]
    return GraspAnchor("ee_right", p[5] + np.array([0.0, 0.0, 0.01]), 5, 0.0, 2000.0)


def e2_on_solves(optimizer_cls=None, refiner_module=None) -> dict[str, str]:
    """The E2-on fixture solves whose SHA `PRE_SHA` pins (also used to measure it with the pre-E3-b file)."""
    from AG3S.trajopt.refiner import TrajOptChunkRefiner
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    cls = optimizer_cls or TrajectoryOptimizer
    mk = lambda sc, cfg: cls(sc.robot, sc.layout, sc.limits, cfg)  # noqa: E731
    out = {}
    st, scene = _candidate_stuck()
    cfg = st.config.with_overrides({"sqp": {"multi_seed": 16}})
    out["stuck_e2"] = _sha(mk(st, cfg).solve(st.reference, st.q_now, scene).trajectory)
    es = _esdf_stuck()
    out["esdf_stuck_e2"] = _sha(mk(es, es.config.with_overrides({"sqp": {"multi_seed": 16}})).solve(
        es.reference, es.q_now, es.scene).trajectory)
    from tests.trajopt.test_e2_multi_seed import _head_on

    scene2 = _head_on(st, radius=0.07, step=4)
    out["stuck_e1c_e2"] = _sha(mk(st, cfg).solve(st.reference, st.q_now, scene2, grasp_anchor=_e1c_anchor(st))
                               .trajectory)
    rcfg = st.config.with_overrides({"horizon": {"plan_horizon": None}, "sqp": {"multi_seed": 16}})
    ref = TrajOptChunkRefiner(st.robot, st.layout, lambda c: (scene, st.q_now, True), rcfg)
    if optimizer_cls is not None:
        ref.optimizer = mk(st, ref.optimizer.config)
    c1 = ref.refine(st.chunk, {"t_step": 3, "previous_physical_chunk": None})
    c2 = ref.refine(st.chunk, {"t_step": 4, "previous_physical_chunk": c1})
    out["refiner_e2_1"], out["refiner_e2_2"] = _sha(c1), _sha(c2)
    return out


# ------------------------------------------------------------------------------------------------ 1. off
def test_off_is_bit_identical_to_the_pre_e3b_code(monkeypatch):
    assert TrajOptConfig().sqp.gpu_seeds == "off"
    # off never imports the GPU module (an import would raise here)
    monkeypatch.setitem(sys.modules, "AG3S.trajopt.gpu_seeds", None)
    got = e2_on_solves()
    if PRE_SHA is None:
        pytest.skip(f"no pinned SHA for numpy {np.__version__}")
    assert got == PRE_SHA
    st, scene = _candidate_stuck()
    rec = _opt(st, 16).solve(st.reference, st.q_now, scene).metrics["multi_seed"]
    assert rec["triggered"] is True and "backend" not in rec and "gpu" not in rec
    assert TrajOptConfig().to_dict()["sqp"]["gpu_seeds"] == "off"


# ------------------------------------------------------------------------------------------------ 2. config / CLI
def test_config_and_cli(monkeypatch, capsys):
    from AG3S.trajopt import serve_safe

    assert TrajOptConfig.from_dict({"sqp": {"multi_seed": 16, "gpu_seeds": "curobo"}}).sqp.gpu_seeds == "curobo"
    for bad in ({"multi_seed": 16, "gpu_seeds": "cuda"}, {"multi_seed": 16, "gpu_seeds": True},
                {"multi_seed": 0, "gpu_seeds": "curobo"}):
        with pytest.raises(TrajOptConfigError):
            TrajOptConfig.from_dict({"sqp": bad})
    # the window optimizer of a look-ahead config inherits it
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=10)
    cfg = sc.config.with_overrides({"horizon": {"lookahead": "obstacles", "lookahead_gate": 0.02},
                                    "collision": {"backend": "esdf"},
                                    "sqp": {"multi_seed": 8, "gpu_seeds": "curobo"}})
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)
    assert opt.gpu_seeds == opt.window_optimizer.gpu_seeds == "curobo"
    parse = lambda *a: serve_safe.build_parser().parse_args([*BASE_ARGV, *a])   # noqa: E731
    assert parse().to_gpu_seeds == "off"
    assert "gpu_seeds" not in serve_safe.sqp_overrides(parse("--to-multi-seed", "16"))       # off: no key
    assert serve_safe.sqp_overrides(parse("--to-multi-seed", "16", "--to-gpu-seeds", "curobo")) == {
        "multi_seed": 16, "gpu_seeds": "curobo"}
    cases = [(["--to-gpu-seeds", "curobo"], "--to-gpu-seeds"),
             (["--no-safe", "--to-gpu-seeds", "curobo", "--to-multi-seed", "16"], "--sqp-*")]
    if not _gpu_ok():
        cases.append((["--to-gpu-seeds", "curobo", "--to-multi-seed", "16"], "--to-gpu-seeds curobo:"))
    for extra, needle in cases:
        monkeypatch.setattr(sys, "argv", ["serve_safe", *BASE_ARGV, *extra])
        with pytest.raises(SystemExit) as e:
            serve_safe.main()
        assert e.value.code == 2
        assert needle in capsys.readouterr().err


# ------------------------------------------------------------------------------------------------ 3. fall-back
def test_a_gpu_path_that_cannot_run_falls_back_to_the_cpu_seeds():
    """E2's candidate fixture has no ESDF field (and in .venv-ag3s no torch): the GPU path fails, E2's CPU seeds run,
    the result is the off result, the record says why; the next trigger does not rebuild a failed batch."""
    st, scene = _candidate_stuck()
    off = _opt(st, 16).solve(st.reference, st.q_now, scene)
    opt = _opt(st, 16, "curobo")
    on = opt.solve(st.reference, st.q_now, scene)
    rec = on.metrics["multi_seed"]
    assert rec["backend"] == "sqp (cuRobo failed)" and rec["gpu"]["error"]
    assert _sha(on.trajectory) == _sha(off.trajectory)
    assert _strip(rec) == _strip({**off.metrics["multi_seed"], "backend": rec["backend"]})
    # the build failed (here: no torch, or the full RB-Y1 model of `make_scenario` has spheres outside the
    # `link_torso_5` subtree, which cannot be a cuRobo c-space) and is cached as failed, not retried per chunk
    batches = opt.__dict__["_gpu_batches"]
    assert len(batches) == 1 and all(isinstance(b, Exception) for b in batches.values())
    opt.reset()
    again = opt.solve(st.reference, st.q_now, scene).metrics["multi_seed"]
    assert again["backend"] == "sqp (cuRobo failed)" and "could not be built" in again["gpu"]["error"]


# ------------------------------------------------------------------------------------------------ 4. GPU: stuck fixture
@needs_gpu
def test_gpu_seeds_resolve_the_stuck_fixture_and_the_choice_passes_our_check():
    from AG3S.trajopt import gpu_seeds as GS
    from AG3S.trajopt.limits import limit_report

    import torch

    tf32_before = bool(torch.backends.cuda.matmul.allow_tf32)
    es = _esdf_stuck()
    tol = float(es.config.safety.violation_tolerance)
    std = _opt(es).solve(es.reference, es.q_now, es.scene)
    assert std.max_violation > 0.04                                         # the standard solve stays inside (47 mm)
    opt = _opt(es, 16, "curobo")
    warm = GS.warm_up(opt)
    assert warm and warm[0]["finite"] and warm[0]["horizon"] == 8
    res = opt.solve(es.reference, es.q_now, es.scene)
    rec = res.metrics["multi_seed"]
    assert rec["triggered"] and rec["backend"] == "curobo" and "error" not in rec["gpu"]
    cands = rec["candidates"]
    assert len(cands) == 17 and cands[0]["kind"] == "standard"
    assert cands[0]["max_violation_mm"] == pytest.approx(std.max_violation * 1e3, abs=1e-3)
    assert [c["kind"] for c in cands[1:3]] == ["policy", "stay"]
    assert all(c["sqp_iterations"] == 0 for c in cands[1:])                 # re-checked, not re-solved
    assert rec["n_resolved"] >= 9                                           # measured: 16 of 16
    g = rec["gpu"]
    assert g["cold"] is False and g["optimize_ms"] > 0 and g["check_ms"] > 0 and g["finite"]
    assert g["settings"]["gaps"] == list(GS.WORLD_GAPS) and g["world"]["box_layer"] == "free"
    assert len(g["cost_out"]) == 16 and all(o < s for o, s in zip(g["cost_out"], g["cost_seed"]))
    # the returned trajectory is a resolved moving candidate, and it passes our check again (re-measured here)
    assert rec["chosen"] > 0 and rec["chosen_kind"] != "stay" and res.max_violation <= tol and res.status.safe
    Q = np.asarray(res.trajectory, np.float64)
    assert -es.linearizer.full_violation(Q, es.q_now, es.scene) <= tol
    assert max(limit_report(Q, es.limits).values()) <= 1e-3
    q0 = es.q_now[es.layout.q_indices]
    assert np.all(np.abs(Q[:, 0] - q0) <= np.asarray(es.limits.max_step) + 1e-9)
    assert cands[rec["chosen"]]["max_violation_mm"] == pytest.approx(res.max_violation * 1e3, abs=1e-3)
    # state: as if only the chosen trajectory had been solved
    assert opt.frame_index == 1 and res.metrics["frame_index"] == 1
    np.testing.assert_array_equal(opt._previous, res.trajectory)
    assert bool(torch.backends.cuda.matmul.allow_tf32) is tf32_before      # cuRobo's global TF32 switch restored
    # cuRobo's FK of our spheres (link_torso_5 frame -> base) is ours
    from curobo.types import JointState

    b = opt._gpu_batches[(16, 8)]
    x = torch.as_tensor(es.reference.T[None], dtype=torch.float32, device="cuda").contiguous()
    st = b.eval_rollout.kin.compute_kinematics(JointState.from_position(x, joint_names=b.joint_names))
    sp = st.robot_spheres.view(1, 8, -1, 4)[0].detach().cpu().numpy().astype(np.float64)
    T = es.robot.link_pose(es.q_now, "link_torso_5")
    base = sp[..., :3] @ T[:3, :3].T + T[:3, 3]
    assert np.abs(base - es.centres[:, b.perm]).max() < 1e-5


# ------------------------------------------------------------------------------------------------ 5. GPU: determinism
@needs_gpu
def test_gpu_seeds_are_deterministic_and_carry_the_goal():
    from AG3S.trajopt import gpu_seeds as GS

    es = _esdf_stuck()
    runs = []
    for _ in range(2):
        opt = _opt(es, 16, "curobo")
        GS.warm_up(opt)
        a = opt.solve(es.reference, es.q_now, es.scene, multi_seed_index=7)
        opt.reset()
        b = opt.solve(es.reference, es.q_now, es.scene, multi_seed_index=7)
        assert _sha(a.trajectory) == _sha(b.trajectory) and _strip(a.metrics["multi_seed"]) == _strip(
            b.metrics["multi_seed"])
        runs.append(a)
    assert _sha(runs[0].trajectory) == _sha(runs[1].trajectory)
    assert _strip(runs[0].metrics["multi_seed"]) == _strip(runs[1].metrics["multi_seed"])
    assert runs[0].metrics["multi_seed"]["gpu"]["cost_out"] == runs[1].metrics["multi_seed"]["gpu"]["cost_out"]
    # the goal term: an E1-C anchor (goal rows 5-7 on this horizon) is on the GPU problem — with its weight the best
    # resolved GPU result is nearer the goal than with a zero-weight anchor (same seeds, the goal only in the record).
    # Measured: 159 mm -> 105 mm (this sweep is too fast for any plan to hold the palm on one point over three rows).
    from AG3S.trajopt.grasp_anchor import GraspAnchor, LinkPointFK

    p = LinkPointFK(es.robot, es.layout, 8, "ee_right")(es.reference, es.q_now)[0]
    best = {}
    for w in (0.0, 2000.0):
        opt = _opt(es, 16, "curobo")
        GS.warm_up(opt)
        r = opt.solve(es.reference, es.q_now, es.scene, grasp_anchor=GraspAnchor("ee_right", p[6], 5, 0.0, w))
        rec = r.metrics["multi_seed"]
        assert rec["triggered"] and rec["backend"] == "curobo" and rec["gpu"]["goal_term"] is True
        assert rec["goal_rows"] == [5, 6, 7] and all(c["goal_err_mm"] is not None for c in rec["candidates"])
        assert r.max_violation <= float(es.config.safety.violation_tolerance)
        best[w] = min(c["goal_err_mm"] for c in rec["candidates"][1:] if c["max_violation_mm"] <= 0.1)
    assert best[2000.0] < best[0.0] - 20.0, best


# ------------------------------------------------------------------------------------------------ 6. offline
def test_offline_settings_and_the_summary(tmp_path):
    from AG3S.trajopt.experiments import e2_multi_seed, e3b_gpu_seeds

    S = e3b_gpu_seeds.SETTINGS
    assert set(S) >= set(e2_multi_seed.SETTINGS) | {"E3b_16", "E3b_32", "E3b_64", "E1C_E3b_16"}
    base = TrajOptConfig()
    for n in (16, 32, 64):
        c = base.with_overrides(S[f"E3b_{n}"])
        assert c.sqp.multi_seed == n and c.sqp.gpu_seeds == "curobo"
        assert c.to_dict() == {**base.with_overrides(S[f"E2_{n}"]).to_dict(),
                               "sqp": {**base.with_overrides(S[f"E2_{n}"]).to_dict()["sqp"], "gpu_seeds": "curobo"}}
    assert base.with_overrides(S["E1C_E3b_16"]).cost == base.with_overrides(S["E1C"]).cost

    def var(viol, ms=None, backend=None, gpu_ms=None):
        e = {"max_violation_mm": viol, "solve_ms": 30.0 + (ms or 0.0), "status": "feasible"}
        if ms is not None:
            e["multi_seed"] = {"triggered": True, "chosen_kind": "detour", "total_ms": ms,
                               **({"backend": backend, "gpu": {"gpu_ms": gpu_ms, "optimize_ms": gpu_ms - 10.0,
                                                               "load_ms": 10.0, "check_ms": 20.0,
                                                               "total_ms": ms, "cold": False}}
                                  if backend else {})}
        return e

    chunks = [{"seq": 1, "variants": {"E0": var(0.0), "E2_16": var(0.0), "E3b_16": var(0.0)}},
              {"seq": 2, "variants": {"E0": var(5.0), "E2_16": var(0.0, 4000.0),
                                      "E3b_16": var(0.0, 150.0, "curobo", 110.0)}},
              {"seq": 3, "variants": {"E0": var(3.0), "E2_16": var(0.0, 3000.0),
                                      "E3b_16": var(1.0, 170.0, "curobo", 130.0)}}]
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"key": "K", "chunks": chunks}))
    t = e3b_gpu_seeds.summarize([str(path)])
    t2 = e2_multi_seed.summarize([str(path)])
    for v in ("E0", "E2_16"):
        assert t["per_setting"][v] == t2["per_setting"][v]                  # E2's table unchanged
    e3 = t["per_setting"]["E3b_16"]
    assert e3["n_base_violated"] == 2 and e3["n_resolved"] == 1 and e3["n_triggered"] == 2
    assert e3["gpu"]["backend"] == {"curobo": 2} and e3["gpu"]["gpu_ms"]["median"] == 120.0
    assert e3["gpu"]["check_ms"]["n"] == 2 and e3["multi_seed_ms_median"] == 160.0


# ------------------------------------------------------------------------------------------------ 7. first-pass check
def test_first_pass_config_and_cli(monkeypatch, capsys):
    from AG3S.trajopt import serve_safe

    assert TrajOptConfig().sqp.gpu_seeds_check == "all"
    c = TrajOptConfig.from_dict({"sqp": {"multi_seed": 16, "gpu_seeds": "curobo", "gpu_seeds_check": "first-pass"}})
    assert c.sqp.gpu_seeds_check == "first-pass"
    for bad in ({"multi_seed": 16, "gpu_seeds": "curobo", "gpu_seeds_check": "some"},
                {"multi_seed": 16, "gpu_seeds_check": "first-pass"}):
        with pytest.raises(TrajOptConfigError):
            TrajOptConfig.from_dict({"sqp": bad})
    parse = lambda *a: serve_safe.build_parser().parse_args([*BASE_ARGV, *a])   # noqa: E731
    assert parse().to_gpu_seeds_check == "all"
    on = ("--to-multi-seed", "16", "--to-gpu-seeds", "curobo")
    assert "gpu_seeds_check" not in serve_safe.sqp_overrides(parse(*on))           # all: no key
    assert serve_safe.sqp_overrides(parse(*on, "--to-gpu-seeds-check", "first-pass")) == {
        "multi_seed": 16, "gpu_seeds": "curobo", "gpu_seeds_check": "first-pass"}
    from AG3S.trajopt.experiments import e3b_gpu_seeds

    fp = TrajOptConfig().with_overrides(e3b_gpu_seeds.SETTINGS["E3b_16_fp"])
    assert fp.sqp.gpu_seeds_check == "first-pass" and fp.sqp.multi_seed == 16 and fp.sqp.gpu_seeds == "curobo"
    monkeypatch.setattr(sys, "argv", ["serve_safe", *BASE_ARGV, "--to-multi-seed", "16",
                                      "--to-gpu-seeds-check", "first-pass"])
    with pytest.raises(SystemExit) as e:
        serve_safe.main()
    assert e.value.code == 2 and "--to-gpu-seeds-check" in capsys.readouterr().err


def _solve_es(check: str, monkeypatch=None, never_resolves: bool = False):
    from AG3S.trajopt import gpu_seeds as GS

    es = _esdf_stuck()
    opt = _opt(es, 16, "curobo", **({"sqp": {"multi_seed": 16, "gpu_seeds": "curobo", "gpu_seeds_check": check}}
                                    if check != "all" else {}))
    GS.warm_up(opt)
    if never_resolves:
        original = GS.solve_seeds

        def wrapped(*a, first_pass=None, **k):
            assert first_pass is not None
            return original(*a, first_pass=(first_pass[0], lambda res: False), **k)

        monkeypatch.setattr(GS, "solve_seeds", wrapped)
    return opt.solve(es.reference, es.q_now, es.scene, multi_seed_index=7)


@needs_gpu
def test_first_pass_chooses_what_all_chooses(monkeypatch):
    full = _solve_es("all")
    fp = _solve_es("first-pass")
    a, b = full.metrics["multi_seed"], fp.metrics["multi_seed"]
    assert "first_pass" not in a["gpu"]
    assert _sha(fp.trajectory) == _sha(full.trajectory)
    assert b["chosen"] == a["chosen"] and b["chosen_kind"] == a["chosen_kind"] and b["chosen"] > 0
    assert b["gpu"]["first_pass"]["hit"] == b["chosen"]
    n = b["gpu"]["first_pass"]["n_checked"]
    assert 1 <= n < 16
    checked = [c for c in b["candidates"][1:] if c.get("checked", True)]
    assert len(checked) == n and len(b["candidates"]) == 17
    strip = lambda c: {k: v for k, v in c.items() if k != "solve_ms"}    # noqa: E731
    assert strip(b["candidates"][b["chosen"]]) == strip(a["candidates"][a["chosen"]])
    # every checked candidate measures as with "all" (the check does not depend on the order it runs in)
    for i, c in enumerate(b["candidates"]):
        if c.get("checked", True):
            assert strip(c) == strip(a["candidates"][i])
    assert fp.max_violation <= float(_esdf_stuck().config.safety.violation_tolerance) and fp.status.safe


@needs_gpu
def test_first_pass_without_a_resolving_result_checks_every_one(monkeypatch):
    full = _solve_es("all")
    fp = _solve_es("first-pass", monkeypatch, never_resolves=True)
    a, b = full.metrics["multi_seed"], fp.metrics["multi_seed"]
    assert b["gpu"]["first_pass"]["hit"] is None and b["gpu"]["first_pass"]["n_checked"] == 16
    assert all(c.get("checked", True) for c in b["candidates"])
    assert _sha(fp.trajectory) == _sha(full.trajectory) and b["chosen"] == a["chosen"]
