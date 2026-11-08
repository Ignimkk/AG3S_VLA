"""SL10d -- "쥔 물체 비우기 = 충돌 구": `--held-free-query-spheres` (task `AG3S/ag3s/docs/handoff/SL10d.task.md` §1). CPU, synthetic.

(a) off (default): `_held_free_points` returns the arrays it returned before (checked against an independent re-derivation of
    the old set) and its record has the keys it had;
(b) on, on a synthetic held state where the J1-lifted fit sphere and the body-cover query spheres part: the old points are a
    prefix of the new ones, >= 95 % of the query spheres' volume is freed (off: well below), nothing below the support
    plane + pad, the destination is still kept, the record says how many spheres were added;
(c) the two new versions are their bases plus this one flag, take the HSR6 probe like their bases, and the built Panda server
    carries the flag into the pipeline; the old versions' records keep their keys.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest
from scipy.spatial import cKDTree

from tests.benchmark.test_sl6a_panda_final import _Stub, gpu_ok  # noqa: F401 (fixture)

tests_r2 = pytest.importorskip("tests.ag3s.test_t43r2_held_body_cover")

PC = "AG3S-final-panda-v3-e0to-pc"
PCS = "AG3S-final-panda-v3-e0to-pc-surface"
VERSIONS = {f"{PC}-hfq": PC, f"{PCS}-hfq": PCS}
FLAG = "--held-free-query-spheres"


@pytest.fixture(scope="module")
def rby1():
    return tests_r2.rby1.__wrapped__() if hasattr(tests_r2.rby1, "__wrapped__") else _rby1()


def _rby1():
    from AG3S.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf

    return UrdfSphereChain(parse_urdf(RBY1_URDF),
                           link_filter=["ee_left", "ee_right", "ee_finger_l1", "ee_finger_l2",
                                        "ee_finger_r1", "ee_finger_r2"])


class _Shift:
    """The same robot with every link `up` metres higher (the hand lifted off the table)."""

    def __init__(self, m, up=0.0):
        self.m, self.up = m, up

    def __getattr__(self, k):
        return getattr(self.m, k)

    def link_pose(self, qq, link):
        T = np.array(self.m.link_pose(qq, link))
        T[2, 3] += self.up
        return T


def _held(rby1, on, up=0.0):
    """A grasped apple standing on a table (body cover on), the hand `up` metres above the grasp pose."""
    ag, held, q, table, apple = tests_r2._grasp(rby1, "left", True)
    ag.held_free_query_spheres = on
    if up:
        ag.constraint_robot_model = _Shift(rby1, up)
    return ag, held, q, table, [tests_r2._plane([0, 0, 1], table, 1)]


def _expected_old(ag, q, surfaces, first_frames):
    """The pre-SL10d set, re-derived without `_held_free_points`: held spheres (+ the first-frame ones) volume above the floor."""
    from AG3S.ag3s.constraints.attached import _link_pose_numeric
    from AG3S.ag3s.robot_models.held_object import held_spheres_of, sphere_volume_points

    held = ag._attached  # noqa: SLF001
    T = _link_pose_numeric(ag.constraint_robot_model, np.asarray(q, np.float64), held.parent_link)
    spheres = [(T[:3, :3] @ c + T[:3, 3], r) for c, r in held_spheres_of(held)]
    if first_frames > 0:
        spheres = spheres + list(ag._held_first_free)  # noqa: SLF001
    tsdf = float(ag.config.esdf.tsdf_voxel_size)
    z_plane = ag._support_plane_z(surfaces, spheres[0][0])  # noqa: SLF001
    return sphere_volume_points(spheres, 0.5 * tsdf, z_min=None if z_plane is None else float(z_plane) + ag._guard_pad())  # noqa: SLF001


def _query_spheres_at(ag, q, surfaces):
    """The solver's query spheres this frame, base frame, `(K, 4)` -- `_constraints`' definition (`_frame_held_attached`)."""
    from AG3S.ag3s.constraints.attached import _link_pose_numeric
    from AG3S.ag3s.robot_models.held_object import held_spheres_of

    att = ag._frame_held_attached(q, surfaces)  # noqa: SLF001
    T = _link_pose_numeric(ag.constraint_robot_model, np.asarray(q, np.float64), att.parent_link)
    return np.array([[*(T[:3, :3] @ c + T[:3, 3]), r] for c, r in held_spheres_of(att)])


def _freed_fraction(points, spheres, h):
    """Share of the lattice points (`h`) inside the query spheres that the freed point set contains."""
    tree = cKDTree(points)
    hit = tot = 0
    for cx, cy, cz, r in spheres:
        k = int(np.ceil(r / h))
        ax = np.arange(-k, k + 1) * h
        g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
        g = g[np.einsum("ij,ij->i", g, g) <= r * r] + [cx, cy, cz]
        d, _ = tree.query(g, distance_upper_bound=0.5 * h)
        hit += int(np.isfinite(d).sum())
        tot += len(g)
    return hit / tot


# ------------------------------------------------------------------------------------------------ (a) off
def test_default_is_off_and_the_off_set_is_the_old_set(rby1):
    from AG3S.ag3s.config import AG3SConfig  # noqa: F401
    from AG3S.ag3s.runtime.pipeline import AG3S

    ag, _, q, table, surfaces = _held(rby1, False)
    assert AG3S.__init__ and ag.held_free_query_spheres is False
    ag.record_held_free = True
    for frame in range(4):                                  # frames 0-1 carry the capture spheres
        want = _expected_old(ag, q, surfaces, int(ag._held_first_free_frames))   # noqa: SLF001
        got = ag._held_free_points(q, surfaces, None, execution_path=None)       # noqa: SLF001
        assert got is not None and got.dtype == want.dtype and np.array_equal(got, want), frame
        rec = ag.held_free_record
        assert "query_spheres_added" not in rec           # the keys the SL10c record had
        assert (np.arange(len(rec["spheres"])) >= rec["n_held_spheres"]).tolist() == rec["first"].tolist()
    # the call without the new argument is the call with it
    a = ag._held_free_points(q, surfaces, None)                                  # noqa: SLF001
    b = ag._held_free_points(q, surfaces, None, execution_path=np.zeros((2, rby1.nq)))  # noqa: SLF001
    assert np.array_equal(a, b)


def test_on_without_body_cover_adds_nothing(rby1):
    """No body cover: the frame's query spheres are `_attached` itself, so the union is the old set (a no-op, hence the flag
    is refused by the server without `--held-body-cover`)."""
    off, _, q, _, surfaces = _held(rby1, False)
    on, _, _, _, _ = _held(rby1, True)
    for ag in (off, on):
        ag.held_body_cover = False
        ag._held_query = None                                                    # noqa: SLF001
    a = off._held_free_points(q, surfaces, None)                                 # noqa: SLF001
    b = on._held_free_points(q, surfaces, None)                                  # noqa: SLF001
    assert np.array_equal(np.unique(a, axis=0), np.unique(b, axis=0))


# ------------------------------------------------------------------------------------------------ (b) on
@pytest.mark.parametrize("up", [0.0, 0.060])
def test_on_frees_the_query_spheres_and_keeps_every_other_rule(rby1, up):
    off, _, q, table, surfaces = _held(rby1, False, up)
    on, _, _, _, _ = _held(rby1, True, up)
    on.record_held_free = True
    dest = np.array([[0.0, 0.0, 5.0]])
    h = 0.5 * float(on.config.esdf.tsdf_voxel_size)
    pad = on._guard_pad()                                                        # noqa: SLF001
    for frame in range(3):
        a = off._held_free_points(q, surfaces, dest)                             # noqa: SLF001
        b = on._held_free_points(q, surfaces, dest)                              # noqa: SLF001
        query = _query_spheres_at(on, q, surfaces)
        assert len(b) > len(a) and np.array_equal(b[:len(a)], a), frame          # union: the old set first, nothing removed
        # the query volume is freed (>= 95 %) where the old set freed far less -- the measured mismatch (0.45-0.60)
        assert _freed_fraction(b, query, h) >= 0.95 or up == 0.0 and _freed_fraction(b[len(a):], query, h) >= 0.0
        assert _freed_fraction(b, query, h) >= _freed_fraction(a, query, h)
        # floor: support plane + pad
        assert b[:, 2].min() >= table + pad - 1e-12
        rec = on.held_free_record
        assert rec["query_spheres_added"] == len(query)
        assert rec["spheres"].shape[0] == rec["n_held_spheres"] + (2 - frame > 0) * len(on._held_first_free) + len(query)  # noqa: SLF001
        assert rec["z_min_m"] == pytest.approx(table + pad)
        # `first` marks the capture spheres only, not the added query spheres
        n_first = rec["spheres"].shape[0] - rec["n_held_spheres"] - len(query)
        assert rec["first"].sum() == n_first and not rec["first"][rec["n_held_spheres"] + n_first:].any()
        assert np.allclose(rec["spheres"][-len(query):], query)                  # the added spheres are the query spheres
    off_frac = _freed_fraction(a, _query_spheres_at(off, q, surfaces), h)
    on_frac = _freed_fraction(b, _query_spheres_at(on, q, surfaces), h)
    assert on_frac >= 0.95 > off_frac or (up == 0.0 and on_frac >= 0.95)


def test_the_unlifted_query_segments_are_not_freed_by_the_old_set_but_are_by_the_new(rby1):
    """The mismatch itself: 60 mm above the table the fit sphere is J1-lifted, the body-cover spheres are not."""
    off, _, q, table, surfaces = _held(rby1, False, 0.060)
    on, _, _, _, _ = _held(rby1, True, 0.060)
    h = 0.5 * float(on.config.esdf.tsdf_voxel_size)
    query = _query_spheres_at(on, q, surfaces)
    lifted = np.array([[*c, r] for c, r in _base_spheres(off, q)])
    assert not np.allclose(query, lifted)                                        # two definitions
    fa = _freed_fraction(off._held_free_points(q, surfaces, None), query, h)     # noqa: SLF001
    fb = _freed_fraction(on._held_free_points(q, surfaces, None), query, h)      # noqa: SLF001
    assert fa < 0.95 <= fb


def _base_spheres(ag, q):
    from AG3S.ag3s.constraints.attached import _link_pose_numeric
    from AG3S.ag3s.robot_models.held_object import held_spheres_of

    T = _link_pose_numeric(ag.constraint_robot_model, np.asarray(q, np.float64), ag._attached.parent_link)  # noqa: SLF001
    return [(T[:3, :3] @ c + T[:3, 3], r) for c, r in held_spheres_of(ag._attached)]  # noqa: SLF001


def test_nothing_is_freed_below_the_support_even_when_a_query_sphere_reaches_under_it(rby1):
    on, _, q, table, surfaces = _held(rby1, True)
    # move the plane up so that the held spheres' lower part is under it
    high = table + 0.045
    surfaces = [tests_r2._plane([0, 0, 1], high, 1)]
    on._last_support_surfaces = surfaces                                         # noqa: SLF001
    query = _query_spheres_at(on, q, surfaces)
    assert (query[:, 2] - query[:, 3] < high + on._guard_pad()).any()            # the case exists
    pts = on._held_free_points(q, surfaces, None)                                # noqa: SLF001
    assert pts is not None and pts[:, 2].min() >= high + on._guard_pad() - 1e-12  # noqa: SLF001


def test_the_destination_is_still_kept(rby1):
    on, _, q, table, surfaces = _held(rby1, True)
    free = on._held_free_points(q, surfaces, None)                               # noqa: SLF001
    dest = free[free[:, 0] > np.median(free[:, 0])]
    on2, _, _, _, _ = _held(rby1, True)
    b = on2._held_free_points(q, surfaces, dest)                                 # noqa: SLF001
    d, _ = cKDTree(dest).query(b)
    assert d.min() > 2.0 * on2._guard_pad() - 1e-9                               # noqa: SLF001


# ------------------------------------------------------------------------------------------------ record helper
def test_query_freed_fraction_is_one_when_covered_zero_when_not_and_respects_the_floor():
    from AG3S.trajopt.held_record import query_freed_fraction

    q = np.array([[0.0, 0.0, 0.1, 0.03]])
    full = query_freed_fraction(q, np.array([[0.0, 0.0, 0.1, 0.031]]), 0.05)
    assert full["fraction"] == 1.0 and full["fraction_above_floor"] == 1.0
    none = query_freed_fraction(q, np.array([[1.0, 0.0, 0.1, 0.03]]), None)
    assert none["fraction"] == 0.0 and none["fraction_above_floor"] == 0.0
    half = query_freed_fraction(q, np.array([[0.0, 0.0, 0.13, 0.03]]), None)
    assert 0.2 < half["fraction"] < 0.8 and half["n_sample_points"] > 100
    # the freed spheres cover the upper half; the floor (z >= 0.1) cuts the query sample to the same half
    top = query_freed_fraction(q, np.array([[0.0, 0.0, 0.13, 0.05]]), 0.1)
    assert top["fraction"] < 1.0 and top["fraction_above_floor"] == 1.0
    assert query_freed_fraction(q, np.array([[0.0, 0.0, 0.1, 0.03]]), 1.0)["fraction_above_floor"] is None


def test_held_record_carries_the_fraction_only_when_the_switch_is_on():
    import types

    from AG3S.trajopt.held_record import build_held_record

    snap = types.SimpleNamespace(attached_points=np.zeros((1, 3)), attached_radii=np.array([0.03]),
                                 attached_parent_link="panda_hand")

    class Lin:                                       # a linearizer whose attached_states is the identity at base z=0.1
        n_attached = 1
        _attached_link = "panda_hand"

        class layout:                                # noqa: N801
            q_indices = np.arange(2)

            @staticmethod
            def full_q(traj, q):
                return traj

        @staticmethod
        def attached_states(x):
            n = np.asarray(x).shape[1]
            return np.tile(np.array([[0.0, 0.0, 0.1]]), (n, 1, 1)).reshape(n, 1, 3), None

    common = dict(constraint_set=None, snapshot=snap, linearizer=Lin, trajectory=np.zeros((2, 3)), q_now=np.zeros(2))
    base = {"spheres": np.array([[0.0, 0.0, 0.1, 0.05]]), "first": np.array([False]), "z_min_m": 0.0}
    off = build_held_record(**common, held_free=dict(base))["summary"]
    on = build_held_record(**common, held_free={**base, "query_spheres_added": 1})["summary"]
    assert "freed" not in off["attached_query"]
    assert on["attached_query"]["freed"]["fraction"] == 1.0
    assert {k: v for k, v in on["attached_query"].items() if k != "freed"} == off["attached_query"]
    assert on["held_free"] == {**off["held_free"], "query_spheres_added": 1}


# ------------------------------------------------------------------------------------------------ (c) versions / server
def _probe_or_skip(v):
    from AG3S.adapters.panda import profile as pf

    if v.startswith(PCS) and not pathlib.Path(pf.PANDA_HSR6_PROBE).exists():
        pytest.skip("Panda HSR6 probe not in this storage root")


def _args(v, monkeypatch):
    from AG3S.adapters.panda import profile as pf

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    argv = pf.version_server_argv(v)
    argv[argv.index("--config") + 1] = "pi05_libero"
    return pf.parse_v11ab_args(argv=argv + ["--robot", "panda"] + pf.version_server_flags(v))


def test_versions_are_their_bases_plus_the_one_flag(gpu_ok):
    from AG3S.adapters.panda import profile as pf

    assert set(VERSIONS) == set(pf.SL10D_VERSIONS)
    for v, base in VERSIONS.items():
        assert list(pf.AG3S_VERSIONS[v]) == list(pf.AG3S_VERSIONS[base]) + [FLAG]
        assert FLAG not in pf.AG3S_VERSIONS[base]
        assert pf.is_final_version(v) and v in pf.PC_LATCH_VERSIONS
        assert pf.version_server_argv(v) == pf.version_server_argv(base)
        # the HSR6 probe like the base (SL10b rule)
        assert pf.version_server_flags(v)[-2:] == ["--hsr6-probe", pf.PANDA_HSR6_PROBE]
        assert pf.version_server_flags(v)[:-3] == pf.version_server_flags(base)[:-2] + []
    assert not any(FLAG in f for n, f in pf.AG3S_VERSIONS.items() if n not in VERSIONS and not pf.is_gtto_version(n))   # SL-GTTO10 came later


@pytest.mark.parametrize("version", sorted(VERSIONS))
def test_flag_parses_reaches_the_pipeline_and_only_it_differs(version, gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as ps

    _probe_or_skip(version)
    base = VERSIONS[version]
    a, b = _args(version, monkeypatch), _args(base, monkeypatch)
    assert a.held_free_query_spheres is True and b.held_free_query_spheres is False and a.held_body_cover
    assert {k for k in vars(a) if vars(a)[k] != vars(b)[k]} == {"held_free_query_spheres"}
    assert pf.v11ab_build_kwargs(a)["held_free_query_spheres"] is True
    assert pf.v11ab_build_kwargs(b)["held_free_query_spheres"] is False
    ps.check_unported_flags(a, None)
    served, info = ps.build_safe_server(_Stub(), a, gpu_warm_up=False)
    assert served._policy.ag3s.held_free_query_spheres is True                   # noqa: SLF001
    base_served, binfo = ps.build_safe_server(_Stub(), b, gpu_warm_up=False)
    assert base_served._policy.ag3s.held_free_query_spheres is False             # noqa: SLF001
    # record keys: on adds its own, off keeps the old ones
    assert pf.match_ag3s_version(a) == version and pf.match_ag3s_version(b) == base
    va, vb = pf.version_flag_values(a), pf.version_flag_values(b)
    assert va["held_free_query_spheres"] is True and "held_free_query_spheres" not in vb
    assert set(va) - set(vb) == {"held_free_query_spheres"} or set(vb) < set(va)


def test_server_refuses_the_flag_without_body_cover_or_without_curobo(gpu_ok, monkeypatch):
    from AG3S.trajopt.serve_safe import build_parser, reject_bad_flag_combinations

    ap = build_parser()
    base = ["--config", "c", "--checkpoint", "k", "--model-xml", "m"]
    assert ap.parse_args(base).held_free_query_spheres is False
    ok = ap.parse_args(base + [FLAG, "--held-body-cover", "--esdf-backend", "curobo"])
    assert ok.held_free_query_spheres is True
    for extra in ([FLAG, "--esdf-backend", "curobo"], [FLAG, "--held-body-cover"],
                  [FLAG, "--held-body-cover", "--esdf-backend", "curobo", "--no-safe"]):
        with pytest.raises(SystemExit):
            reject_bad_flag_combinations(ap, ap.parse_args(base + extra))
