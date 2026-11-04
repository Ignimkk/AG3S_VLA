"""SL10c — record only: the constraint record (`--record-constraints-esdf full`) also carries the held object's
geometry and the support planes. CPU, synthetic.

(a) off / on change nothing the server does: `_held_free_points` returns the same arrays and walks the same state, the
    solver (refiner) returns bit-identical chunks whether or not the record was built in between, and the writer writes
    no new key outside `full`;
(b) on a synthetic held state the new keys have the intended shape / frame / values;
(c) the SL10 re-solve tool reads the new keys instead of refitting planes / asking for `--attached-spheres-json`,
    and goes the old way when they are absent.
"""
from __future__ import annotations

import dataclasses
import json
import types

import numpy as np
import pytest

from AG3S.ag3s.experiments.sources.constraint_record import HELD_RECORD_KEYS, ConstraintRecordWriter

Q0 = np.array([0.0, 0.17, 0.0, -2.45, 0.0, 2.6, 0.79])


# ------------------------------------------------------------------------------------------ writer
def _record(tmp_path, mode, held_record=None):
    w = ConstraintRecordWriter(tmp_path, esdf_mode=mode)
    path = w.record(t_step=3, chunk_index=1, constraint_set=None, debug={}, reference_chunk=np.zeros((4, 3)),
                    **({} if held_record is None else {"held_record": held_record}))
    w.close()
    return path


def _sample_record():
    return {"arrays": {"support_planes": np.array([[0, 0, 1.0, -0.01]]),
                       "held_free_spheres": np.array([[0.1, 0.2, 0.3, 0.04]]),
                       "held_free_first_frame": np.array([False]),
                       "attached_parent_link": np.asarray("panda_hand")},
            "summary": {"n_support_planes": 1, "held_free": {"n_points_freed": 12}}}


def test_new_keys_exist_only_in_full_mode(tmp_path):
    rec = _sample_record()
    base = {m: _record(tmp_path / f"base_{m}", m) for m in ("none", "occupancy", "full")}
    on = {m: _record(tmp_path / f"on_{m}", m, rec) for m in ("none", "occupancy", "full")}
    for mode in ("none", "occupancy"):
        with np.load(base[mode]) as a, np.load(on[mode]) as b:
            assert a.files == b.files and not set(b.files) & set(HELD_RECORD_KEYS)
            for k in a.files:
                assert np.array_equal(a[k], b[k]), k            # bit-identical, summary included
    with np.load(base["full"]) as a, np.load(on["full"]) as b:
        assert not set(a.files) & set(HELD_RECORD_KEYS)         # `full` without a held record = what it was
        assert set(b.files) - set(a.files) == set(rec["arrays"])
        for k in a.files:
            if k != "summary_json":
                assert np.array_equal(a[k], b[k]), k
        sa, sb = json.loads(str(a["summary_json"])), json.loads(str(b["summary_json"]))
        assert "held_geometry_record" not in sa
        assert sb.pop("held_geometry_record") == rec["summary"] and sa == sb
        assert str(b["attached_parent_link"]) == "panda_hand"


def test_the_writer_refuses_a_key_it_does_not_define(tmp_path):
    with pytest.raises(KeyError, match="HELD_RECORD_KEYS"):
        _record(tmp_path, "full", {"arrays": {"held_typo": np.zeros(1)}, "summary": None})


def test_background_writer_carries_the_new_keys(tmp_path):
    w = ConstraintRecordWriter(tmp_path, esdf_mode="full", background=True)
    w.record(t_step=3, chunk_index=1, constraint_set=None, debug={}, held_record=_sample_record())
    w.close()
    with np.load(w.run_dir / "chunk_00000.npz") as z:
        assert str(z["attached_parent_link"]) == "panda_hand" and z["held_free_spheres"].shape == (1, 4)


# ------------------------------------------------------------------------------------------ pipeline (a)
@pytest.fixture(scope="module")
def rby1():
    from AG3S.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf

    return UrdfSphereChain(parse_urdf(RBY1_URDF),
                           link_filter=["ee_left", "ee_right", "ee_finger_l1", "ee_finger_l2",
                                        "ee_finger_r1", "ee_finger_r2"])


def _held_pipeline(rby1, record: bool):
    mod = pytest.importorskip("tests.ag3s.test_t43t_held_free_observed")
    ag, held, q, table, _ = mod._held(rby1, 0.015)
    ag.record_held_free = record
    return ag, held, q, table, mod._plane([0, 0, 1], table)


def test_held_free_points_are_identical_with_the_record_on_or_off(rby1):
    off, _, q, _, plane = _held_pipeline(rby1, False)
    on, _, _, _, _ = _held_pipeline(rby1, True)
    dest = np.array([[0.0, 0.0, 5.0]])
    for frame in range(4):
        a = off._held_free_points(q, [plane], dest)                              # noqa: SLF001
        b = on._held_free_points(q, [plane], dest)                               # noqa: SLF001
        assert (a is None) == (b is None)
        if a is not None:
            assert a.dtype == b.dtype and np.array_equal(a, b), frame
        assert off._held_first_free_frames == on._held_first_free_frames        # noqa: SLF001
    assert off.held_free_record is None and off._held_free_stash is None        # noqa: SLF001


def test_held_free_record_shape_frame_and_first_flags(rby1):
    from AG3S.ag3s.constraints.attached import _link_pose_numeric
    from AG3S.ag3s.robot_models.held_object import held_spheres_of

    on, held, q, table, plane = _held_pipeline(rby1, True)
    local = held_spheres_of(on._attached)                                       # noqa: SLF001
    T = _link_pose_numeric(on.constraint_robot_model, np.asarray(q, np.float64), on._attached.parent_link)  # noqa: SLF001
    n_first = len(on._held_first_free)                                          # noqa: SLF001
    assert on.held_free_record is None                                          # nothing freed yet
    pts = on._held_free_points(q, [plane], None)                                # noqa: SLF001
    rec = on.held_free_record
    assert rec["spheres"].shape == (len(local) + n_first, 4)
    # frame: base. The held spheres at `robot_state` come first (T @ local), the capture spheres after them.
    for i, (c, r) in enumerate(local):
        assert np.allclose(rec["spheres"][i, :3], T[:3, :3] @ c + T[:3, 3]) and rec["spheres"][i, 3] == pytest.approx(r)
    assert not rec["first"][:len(local)].any() and rec["first"][len(local):].all()
    assert rec["first_frames_left_before"] == 2 and rec["n_held_spheres"] == len(local)
    assert rec["n_points_freed"] == len(pts) and rec["z_min_m"] == pytest.approx(table + on._guard_pad())  # noqa: SLF001
    assert rec["n_sphere_points_after_floor"] + rec["n_observed_points"] == rec["n_points_before_destination"]
    # a later frame: the first-frame spheres are gone and the flag says so
    on._held_free_points(q, [plane], None)                                      # noqa: SLF001
    on._held_free_points(q, [plane], None)                                      # noqa: SLF001
    later = on.held_free_record
    assert later["spheres"].shape == (len(local), 4) and not later["first"].any()
    assert later["first_frames_left_before"] == 0
    # the copy is read-only for the caller
    later["spheres"][:] = 0.0
    assert on.held_free_record["spheres"].any()


def test_a_frame_that_frees_nothing_does_not_show_the_previous_frames_spheres(rby1):
    on, _, q, _, plane = _held_pipeline(rby1, True)
    on._held_free_points(q, [plane], None)                                      # noqa: SLF001
    assert on.held_free_record is not None
    on._build_esdf([], None)                                                    # noqa: SLF001  (no depth: returns early)
    assert on.held_free_record is None


# ------------------------------------------------------------------------------------------ solver (a, b)
LINK = "ee_finger_l1"


def _scenario(*, attached=True):
    from tests.trajopt.fixtures import make_scenario
    from tests.trajopt.test_attached_points import _FlatField

    sc = make_scenario(horizon=10, penetration=0.03)
    rng = np.random.default_rng(0)
    pts = rng.normal(scale=0.03, size=(3, 3))
    rad = np.array([0.03, 0.02, 0.025])
    scene = dataclasses.replace(sc.scene, esdf=_FlatField(0.4), esdf_margin=0.0,
                                **({"attached_points": pts, "attached_parent_link": LINK, "attached_radii": rad}
                                   if attached else {}))
    return sc, scene, pts, rad


def _solver(sc, scene):
    from AG3S.trajopt.refiner import TrajOptChunkRefiner

    return TrajOptChunkRefiner(sc.robot, sc.layout, lambda ctx: (scene, sc.q_now, True), sc.config)


def test_building_the_record_does_not_change_what_the_solver_returns():
    """Two identical solvers; one has its linearizer asked for the record between two solves, the other never."""
    from AG3S.trajopt.held_record import build_held_record

    sc, scene, _, _ = _scenario()
    a, b = _solver(sc, scene), _solver(sc, scene)
    a1, b1 = a.refine(sc.chunk, {}), b.refine(sc.chunk, {})
    assert np.array_equal(a1, b1)
    ra = a.last_result
    build_held_record(constraint_set=None, snapshot=scene, linearizer=a.optimizer.linearizer,
                      trajectory=ra.trajectory, q_now=sc.q_now,
                      held_free={"spheres": np.zeros((1, 4)), "first": np.zeros(1, bool)},
                      finger_plan={"opening_norm": {"left": [1.0] * 10}})
    a2, b2 = a.refine(sc.chunk, {}), b.refine(sc.chunk, {})
    assert np.array_equal(a2, b2)
    assert np.array_equal(a.last_result.trajectory, b.last_result.trajectory)
    assert a.last_result.max_violation == b.last_result.max_violation
    assert a.optimizer.linearizer.n_attached == b.optimizer.linearizer.n_attached == 3
    assert np.array_equal(a.optimizer.linearizer.query_radii, b.optimizer.linearizer.query_radii)


def test_query_spheres_are_the_solvers_held_spheres_in_the_base_frame():
    from AG3S.ag3s.constraints.attached import base_frame_points
    from AG3S.ag3s.types import AttachedCollisionGeometry
    from AG3S.trajopt.held_record import build_held_record

    sc, scene, pts, rad = _scenario()
    s = _solver(sc, scene)
    s.refine(sc.chunk, {})
    traj = s.last_result.trajectory
    rec = build_held_record(constraint_set=None, snapshot=scene, linearizer=s.optimizer.linearizer, trajectory=traj,
                            q_now=sc.q_now)
    arrays = rec["arrays"]
    assert set(arrays) <= set(HELD_RECORD_KEYS)
    assert str(arrays["attached_parent_link"]) == LINK and arrays["attached_parent_link"].shape == ()
    # definition: parent frame, [c, r]
    assert arrays["held_local_spheres"].shape == (3, 4)
    assert np.allclose(arrays["held_local_spheres"][:, :3], pts) and np.allclose(arrays["held_local_spheres"][:, 3], rad)
    # query: base frame, independently computed with the numeric FK
    geo = AttachedCollisionGeometry(parent_link=LINK, T_parent_object=np.eye(4), primitives=[], points=pts)
    full = sc.layout.full_q(traj, sc.q_now)
    path = arrays["attached_query_centres"]
    assert path.shape == (traj.shape[1], 3, 3)
    for h in range(path.shape[0]):
        assert np.allclose(path[h], base_frame_points(geo, sc.robot, np.asarray(full[:, h]).reshape(-1)), atol=1e-9)
    stay = base_frame_points(geo, sc.robot, np.asarray(sc.q_now, np.float64))
    now = arrays["attached_query_spheres"]
    assert now.shape == (3, 4) and np.allclose(now[:, :3], stay, atol=1e-9) and np.allclose(now[:, 3], rad)
    assert rec["summary"]["attached_query"] == {"available": True}
    assert rec["summary"]["attached"] == {"parent_link": LINK, "n": 3, "radii_given": True}


def test_support_planes_are_always_written_and_capture_and_fingers_only_when_present():
    from AG3S.ag3s.types import SupportSurface
    from AG3S.trajopt.held_record import build_held_record

    def plane(n, off, sid):
        return SupportSurface(id=sid, normal=np.asarray(n, float), offset=float(off),
                              point_indices=np.zeros(0, np.int64), point_count=1000, rms_error=0.001)

    cs = types.SimpleNamespace(support_surfaces=[plane([0, 0, 1], -0.0106, 1), plane([0.03, 0.0, 0.9995], 0.02, 2)])
    rec = build_held_record(constraint_set=cs, snapshot=None, linearizer=None, trajectory=None, q_now=None,
                            held_capture=np.array([[0.5, 0.1, 0.03, 0.05]]),
                            finger_plan={"opening_norm": {"left": [1.0, 0.9, 0.8]}})
    a = rec["arrays"]
    assert a["support_planes"].shape == (2, 4) and np.allclose(a["support_planes"][0], [0, 0, 1, -0.0106])
    n2 = np.array([0.03, 0.0, 0.9995])
    assert np.allclose(a["support_planes"][1], [*(n2 / np.linalg.norm(n2)), 0.02])    # SupportSurface stores a unit normal
    assert a["held_capture_spheres"].shape == (1, 4)
    assert a["finger_plan_opening"].shape == (1, 3) and list(a["finger_plan_hands"]) == ["left"]
    assert not {"held_local_spheres", "attached_query_spheres", "attached_query_centres", "attached_parent_link",
                "held_free_spheres"} & set(a)                                  # nothing held: no held keys
    empty = build_held_record(constraint_set=None, snapshot=None, linearizer=None, trajectory=None, q_now=None)
    assert empty["arrays"]["support_planes"].shape == (0, 4) and set(empty["arrays"]) == {"support_planes"}


def test_a_linearizer_that_disagrees_with_the_scene_is_said_not_guessed():
    from AG3S.trajopt.held_record import build_held_record

    sc, scene, _, _ = _scenario()
    s = _solver(sc, scene)
    s.refine(sc.chunk, {})
    s.optimizer.linearizer.set_attached(None, None)
    rec = build_held_record(constraint_set=None, snapshot=scene, linearizer=s.optimizer.linearizer,
                            trajectory=s.last_result.trajectory, q_now=sc.q_now)
    assert "attached_query_spheres" not in rec["arrays"] and "held_local_spheres" in rec["arrays"]
    assert rec["summary"]["attached_query"]["available"] is False and rec["summary"]["attached_query"]["why_not"]


# ------------------------------------------------------------------------------------------ SafePolicy wiring
def _policy(tmp_path, mode):
    from tests.trajopt.test_safe_policy import _StubPolicy, _safe_policy

    sp = _safe_policy(_StubPolicy(), recorder=ConstraintRecordWriter(tmp_path, esdf_mode=mode))
    return sp


@pytest.mark.parametrize("mode,flag", [("none", False), ("occupancy", False), ("full", True)])
def test_the_pipeline_records_held_state_only_with_a_full_recorder(tmp_path, mode, flag):
    sp = _policy(tmp_path / mode, mode)
    assert sp.ag3s.record_held_free is flag


def test_the_pipeline_default_is_off():
    from tests.trajopt.test_safe_policy import _StubPolicy, _safe_policy

    assert _safe_policy(_StubPolicy()).ag3s.record_held_free is False


def _wire_held_state(sp, sc, scene):
    """A SafePolicy in a synthetic held state: the solver linearizer holds the spheres, the pipeline has its stash."""
    from AG3S.ag3s.types import SupportSurface

    opt = sp.refiner.optimizer
    opt.linearizer.set_attached(scene.attached_points, LINK)
    q_now = np.zeros(sp.layout.nq_model)
    traj = np.tile(q_now[sp.layout.q_indices].reshape(-1, 1), (1, opt.linearizer.horizon)) + 0.01
    sp.refiner.last_result = types.SimpleNamespace(trajectory=traj, max_violation=0.0, status=None)
    sp._last_snapshot = scene                                                    # noqa: SLF001
    sp._last_q_now = q_now                                                       # noqa: SLF001
    sp._last_constraint_set = types.SimpleNamespace(                             # noqa: SLF001
        esdf=None, support_surfaces=[SupportSurface(id=1, normal=np.array([0, 0, 1.0]), offset=-0.0106,
                                                    point_indices=np.zeros(0, np.int64), point_count=900,
                                                    rms_error=0.001)])
    sp._last_finger = {"plan": {"opening_norm": {"left": [1.0] * 10}}}           # noqa: SLF001
    sp.ag3s._held_free_stash = {"spheres": np.array([[0.1, 0.2, 0.3, 0.04], [0.2, 0.2, 0.3, 0.05]]),  # noqa: SLF001
                                "first": np.array([False, True]), "n_points_freed": 77}
    sp.ag3s._attached = types.SimpleNamespace()                                  # noqa: SLF001
    sp.ag3s._held_first_free = [(np.array([0.2, 0.2, 0.3]), 0.05)]               # noqa: SLF001
    return q_now, traj


def test_safe_policy_record_carries_the_new_keys_in_full_mode_only(tmp_path):
    sc, scene, pts, rad = _scenario()
    out = {}
    for mode in ("occupancy", "full"):
        sp = _policy(tmp_path / mode, mode)
        _wire_held_state(sp, sc, scene)
        sp._record(5, np.zeros((4, sc.layout.action_dim)), np.zeros((4, sc.layout.action_dim)))   # noqa: SLF001
        sp.recorder.close()
        out[mode] = sorted(sp.recorder.run_dir.glob("chunk_*.npz"))[-1]
    with np.load(out["occupancy"]) as z:
        assert not set(z.files) & set(HELD_RECORD_KEYS)
        assert "held_geometry_record" not in json.loads(str(z["summary_json"]))
    with np.load(out["full"]) as z:
        assert set(HELD_RECORD_KEYS) <= set(z.files)
        assert z["support_planes"].shape == (1, 4) and np.allclose(z["support_planes"][0], [0, 0, 1, -0.0106])
        assert z["held_free_spheres"].shape == (2, 4) and list(z["held_free_first_frame"]) == [False, True]
        assert z["held_local_spheres"].shape == (3, 4) and str(z["attached_parent_link"]) == LINK
        assert z["attached_query_spheres"].shape == (3, 4) and z["attached_query_centres"].shape[1:] == (3, 3)
        assert z["held_capture_spheres"].shape == (1, 4) and z["finger_plan_opening"].shape == (1, 10)
        summary = json.loads(str(z["summary_json"]))
        assert summary["held_geometry_record"]["held_free"]["n_points_freed"] == 77
        assert summary["held_geometry_record"]["attached_query"]["available"] is True


def test_a_failing_held_record_does_not_cost_the_chunk_its_record(tmp_path, monkeypatch):
    import AG3S.trajopt.held_record as hr

    sc, scene, _, _ = _scenario()
    sp = _policy(tmp_path, "full")
    _wire_held_state(sp, sc, scene)

    def boom(**_):
        raise RuntimeError("boom")

    monkeypatch.setattr(hr, "build_held_record", boom)
    sp._record(5, np.zeros((4, sc.layout.action_dim)), np.zeros((4, sc.layout.action_dim)))   # noqa: SLF001
    sp.recorder.close()
    assert sp._record_failures == 0                                              # noqa: SLF001
    with np.load(sorted(sp.recorder.run_dir.glob("chunk_*.npz"))[-1]) as z:
        assert json.loads(str(z["summary_json"]))["held_geometry_record"]["error"].startswith("RuntimeError")
        assert not set(z.files) & set(HELD_RECORD_KEYS)


# ------------------------------------------------------------------------------------------ the SL10 tool (c)
@pytest.fixture(scope="module")
def runtime():
    from AG3S.adapters.panda import profile
    from AG3S.trajopt import serve_safe as ss
    from AG3S.trajopt.experiments import panda_fixed_input_ablation as tool

    v = "AG3S-final-panda-v3-e0to-pc"
    args = ss.build_parser().parse_args(profile.version_server_argv(v) + profile.version_server_flags(v) + ["--robot", "panda"])
    return tool.build_runtime(args)


class _Z(dict):
    @property
    def files(self):
        return list(self)


def test_tool_helpers_read_the_new_keys_and_return_none_without_them():
    from AG3S.trajopt.experiments import panda_fixed_input_ablation as tool

    z = _Z(support_planes=np.array([[0, 0, 1.0, -0.0106], [0.03, 0, 0.9995, 0.02]]),
           held_local_spheres=np.array([[0.0, 0.0, 0.1, 0.05], [0.0, 0.0, 0.15, 0.04]]),
           attached_parent_link=np.asarray("panda_hand"), held_capture_spheres=np.array([[0.5, 0.1, 0.03, 0.05]]))
    surfaces, info = tool.support_surfaces_recorded(z)
    assert info["source"] == "record support_planes" and len(surfaces) == 2
    assert np.allclose(surfaces[1].normal, [0.03, 0, 0.9995]) and surfaces[1].offset == 0.02
    assert tool.attached_spec_from_record(z) == {"parent_link": "panda_hand",
                                                 "spheres": [[0.0, 0.0, 0.1, 0.05], [0.0, 0.0, 0.15, 0.04]]}
    assert tool.held_capture_from_record(z).shape == (1, 4)
    none = _Z()
    assert tool.support_surfaces_recorded(none)[0] is None
    assert tool.attached_spec_from_record(none) is None and tool.held_capture_from_record(none) is None
    # (0, 4) is "the server had no plane", not "not recorded": no refit is invented
    s0, i0 = tool.support_surfaces_recorded(_Z(support_planes=np.zeros((0, 4))))
    assert s0 == [] and i0["n_planes"] == 0


def _scene_from(rt, z, *, attached):
    from AG3S.trajopt.experiments import panda_fixed_input_ablation as tool
    from tests.benchmark.test_sl10_panda_fixed_input import _sphere_field

    p0 = rt.kin.fk(Q0)[0]
    field = _sphere_field(p0 + np.array([0.3, 0.0, 0.0]), 0.04)
    call = types.SimpleNamespace(row={"grasp": {"state": "held"}}, q_now=Q0, seq=1, episode=None)
    return tool.build_scene(rt, call, z, {"manipulated": {}}, field, attached=attached)


def test_build_scene_uses_the_recorded_planes_spheres_and_capture(runtime):
    rt = runtime
    z = _Z(support_planes=np.array([[0, 0, 1.0, -0.0106]]),
           held_local_spheres=np.array([[0.0, 0.0, 0.10, 0.05], [0.0, 0.01, 0.14, 0.04]]),
           attached_parent_link=np.asarray("panda_hand"), held_capture_spheres=np.array([[0.5, 0.1, 0.03, 0.05]]),
           target_points=np.zeros((3, 3), np.float32))
    scene, info = _scene_from(rt, z, attached=True)
    assert info["support_surfaces"]["source"] == "record support_planes"
    assert info["attached"] == {"source": "record held_local_spheres", "n": 2, "parent_link": "panda_hand"}
    assert info["held_capture"] == {"source": "record held_capture_spheres", "n": 1}
    assert scene.attached_parent_link == "panda_hand"
    assert np.allclose(scene.attached_points, z["held_local_spheres"][:, :3])
    assert np.allclose(scene.attached_radii, z["held_local_spheres"][:, 3])
    planes = getattr(scene, "support_planes", None)
    if planes is not None:                                   # present when the config reads planes (look-ahead / HM)
        assert np.allclose(planes, [[0, 0, 1.0, -0.0106]])
    cap = getattr(scene, "held_capture_spheres", None)
    if cap is not None:
        assert np.allclose(cap, [[0.5, 0.1, 0.03, 0.05]])
    # an explicit --attached-spheres-json wins over the record
    rt.attached_spheres = {"parent_link": "panda_hand", "spheres": [[0.0, 0.0, 0.2, 0.03]]}
    try:
        scene2, info2 = _scene_from(rt, z, attached=True)
    finally:
        rt.attached_spheres = None
    assert info2["attached"]["source"] == "--attached-spheres-json" and len(scene2.attached_points) == 1


def test_build_scene_without_the_new_keys_goes_the_old_way(runtime):
    rt = runtime
    z = _Z(target_points=np.zeros((3, 3), np.float32))
    scene, info = _scene_from(rt, z, attached=True)
    assert info["attached"] == {"source": None, "missing": "held spheres are not in the record"}
    assert scene.attached_points is None and info["held_capture"] is None
    assert info["support_surfaces"]["source"] == "support_mask + refit" and "unavailable" in info["support_surfaces"]
