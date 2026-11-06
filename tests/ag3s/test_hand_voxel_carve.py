"""SL16 HVC -- the hand voxel carve (`AG3S.hand_voxel_carve`, CLI `--hand-voxel-carve`): the occupied FINE-TSDF voxels inside the hand's
volume are set free before the ESDF is made. CPU only (no GPU: the builder's TSDF access is replaced by a synthetic voxel list).

| # | property |
|---|---|
| 1 | `inside_sphere_union`: exact (not the 16 nearest centres), the floor per sphere |
| 2 | `hand_carve_hits`: the freed volume on a synthetic grid = the grown spheres; support floor · destination · target ball / points are kept; `trail`'s near rule; counts |
| 3 | `AG3S._hand_carve_spec`: hand links only, `current` = spheres + r, `trail` = history (cap 8) + 4 mm and the near rule, no state / model -> None |
| 4 | pipeline wiring (fake builder): off = no kwarg and no stats key; on = the kwarg; a frame without a spec is recorded as not applied |
| 5 | `CuroboFieldBuilder._carve_hand`: frees exactly the hit voxels, the record {mode, r_mm, n_voxels_freed} |
| 6 | CLI: defaults off; section; refusals (RB-Y1, no curobo, `-mm` without `current`, `--no-safe`) |
"""
from __future__ import annotations

import json
import types

import numpy as np
import pytest

from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder, hand_carve_hits, inside_sphere_union

VOXEL = 0.005


def _grid(lo, hi, h=VOXEL):
    """Voxel centres of a box (the occupied voxels of a synthetic TSDF)."""
    axes = [np.arange(l + 0.5 * h, u, h) for l, u in zip(lo, hi)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


# ============================================================================ 1 the union test
def test_inside_sphere_union_is_exact_beyond_sixteen_neighbours():
    rng = np.random.default_rng(0)
    c = np.vstack([rng.normal(scale=0.004, size=(40, 3)), [[0.05, 0.0, 0.0]]])   # 40 tiny spheres at the origin, one big one beside
    r = np.concatenate([np.full(40, 0.002), [0.04]])
    p = np.array([[0.02, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    # [0.02, 0, 0] is inside the big sphere (30 mm from its centre) although 16+ tiny centres are nearer to it
    assert inside_sphere_union(p, c, r).tolist() == [True, True, False]


def test_inside_sphere_union_floor_is_per_sphere():
    c = np.array([[0.0, 0.0, 0.10], [0.2, 0.0, 0.10]])
    r = np.array([0.05, 0.05])
    z = np.array([0.09, -np.inf])
    p = np.array([[0.0, 0.0, 0.085], [0.0, 0.0, 0.095], [0.2, 0.0, 0.07]])
    assert inside_sphere_union(p, c, r, z).tolist() == [False, True, True]
    assert inside_sphere_union(np.zeros((0, 3)), c, r).shape == (0,)
    assert not inside_sphere_union(p, np.zeros((0, 3)), np.zeros(0)).any()


# ============================================================================ 2 the carve rules
def _spec(centres, radii, **kw):
    c = np.asarray(centres, np.float64).reshape(-1, 3)
    return dict(mode="current", centres=c, radii=np.asarray(radii, np.float64).reshape(-1), z_min=np.full(len(c), -np.inf),
                destination=None, dest_pad=0.0, protect_ball=None, protect_points=None, protect_tol=0.0, near=None, **kw)


def test_the_freed_volume_on_a_synthetic_grid_is_the_grown_spheres():
    pts = _grid((-0.1,) * 3, (0.1,) * 3)
    for grow in (0.0, 0.010, 0.020):
        hand_r = 0.015
        hit, info = hand_carve_hits(pts, _spec([[0.0, 0.0, 0.0]], [hand_r + grow]))
        want = np.linalg.norm(pts, axis=1) <= hand_r + grow
        assert np.array_equal(hit, want)
        assert info["n_freed"] == int(want.sum()) == info["n_in_volume"]
        assert info["n_excluded_floor_or_destination"] == 0 and info["n_excluded_target"] == 0
    r = 0.035                                                   # the voxel count is the sphere's volume in voxels (within 5 %)
    n = hand_carve_hits(pts, _spec([[0.0, 0.0, 0.0]], [r]))[1]["n_freed"]
    assert abs(n / (4.0 / 3.0 * np.pi * r ** 3 / VOXEL ** 3) - 1.0) < 0.05


def test_the_support_floor_the_destination_and_the_target_are_kept():
    pts = _grid((-0.1,) * 3, (0.1,) * 3)
    base = np.linalg.norm(pts, axis=1) <= 0.04
    # support plane at z = -0.02 (+ pad): nothing below it
    spec = _spec([[0.0, 0.0, 0.0]], [0.04])
    spec["z_min"] = np.array([-0.02 + VOXEL])
    hit, info = hand_carve_hits(pts, spec)
    assert np.array_equal(hit, base & (pts[:, 2] >= -0.02 + VOXEL)) and info["n_excluded_floor_or_destination"] == int((base & ~hit).sum()) > 0
    assert not (hit & (pts[:, 2] < -0.015)).any()
    # destination: nothing within dest_pad of a destination point
    spec = _spec([[0.0, 0.0, 0.0]], [0.04])
    spec.update(destination=np.array([[0.02, 0.0, 0.0]]), dest_pad=0.010)
    hit, _ = hand_carve_hits(pts, spec)
    assert not (hit & (np.linalg.norm(pts - [0.02, 0.0, 0.0], axis=1) <= 0.010)).any() and hit.sum() < base.sum()
    # the manipulated target's volume: a ball, and the observed points + a tolerance
    spec = _spec([[0.0, 0.0, 0.0]], [0.04])
    spec["protect_ball"] = (np.array([0.03, 0.0, 0.0]), 0.015)
    hit, info = hand_carve_hits(pts, spec)
    assert not (hit & (np.linalg.norm(pts - [0.03, 0.0, 0.0], axis=1) <= 0.015)).any()
    assert info["n_excluded_target"] == int((base & (np.linalg.norm(pts - [0.03, 0.0, 0.0], axis=1) <= 0.015)).sum()) > 0
    spec = _spec([[0.0, 0.0, 0.0]], [0.04])
    spec.update(protect_points=np.array([[0.0, 0.03, 0.0]]), protect_tol=0.006)
    hit, info = hand_carve_hits(pts, spec)
    assert not (hit & (np.linalg.norm(pts - [0.0, 0.03, 0.0], axis=1) <= 0.006)).any() and info["n_excluded_target"] > 0
    assert info["n_freed"] == int(hit.sum()) and hit.sum() == base.sum() - info["n_excluded_target"]
    # a voxel the volume does not reach is never touched by any rule
    far = np.array([[0.09, 0.09, 0.09]])
    assert not hand_carve_hits(far, spec)[0].any()


def test_trail_frees_an_old_position_only_near_the_hand_now():
    old, now = np.array([0.0, 0.0, 0.0]), np.array([0.05, 0.0, 0.0])
    pts = _grid((-0.1, -0.05, -0.05), (0.15, 0.05, 0.05))
    spec = _spec([old, now], [0.02, 0.02])
    spec["near"] = {"centres": now.reshape(1, 3), "radii": np.array([0.02 + 0.020])}     # within 20 mm of the hand now
    hit, info = hand_carve_hits(pts, spec)
    in_old = np.linalg.norm(pts - old, axis=1) <= 0.02
    in_now = np.linalg.norm(pts - now, axis=1) <= 0.02
    near = np.linalg.norm(pts - now, axis=1) <= 0.04
    assert np.array_equal(hit, (in_old | in_now) & near)
    assert (in_old & ~near).any() and info["n_in_volume"] == int(((in_old | in_now) & near).sum())   # the far part of the trail is kept


# ============================================================================ 3 the pipeline's spec
class _Hand:
    """A toy self-filter model: two hand spheres, one finger, an arm sphere. `q` (3,) shifts every sphere."""

    sphere_link_names = ("panda_link7", "panda_hand", "panda_hand", "panda_leftfinger")
    BASE = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.3], [0.02, 0.0, 0.3], [0.0, 0.03, 0.28]])
    RADII = np.array([0.05, 0.03, 0.03, 0.01])

    def sphere_centers_numeric(self, q):
        return self.BASE + np.asarray(q, np.float64).reshape(1, 3), self.RADII.copy()


def _pipeline(mode="current", radius=0.020, history=None):
    from AG3S.ag3s.runtime.pipeline import AG3S

    plane = types.SimpleNamespace(normal=np.array([0.0, 0.0, 1.0]), offset=0.2)
    fake = types.SimpleNamespace(
        hand_voxel_carve=None if mode is None else {"mode": mode, "radius_m": radius}, _hand_carve_history=list(history or []),
        _frame_filter_model=_Hand(), _guard_pad=lambda: VOXEL, _support_plane_z=AG3S._support_plane_z,
        HAND_CARVE_HISTORY=AG3S.HAND_CARVE_HISTORY, HAND_CARVE_TRAIL_M=AG3S.HAND_CARVE_TRAIL_M,
        HAND_CARVE_NEAR_M=AG3S.HAND_CARVE_NEAR_M, HAND_CARVE_LINKS=AG3S.HAND_CARVE_LINKS)
    return fake, AG3S._hand_carve_spec.__get__(fake), [plane]


def test_current_spec_is_the_hand_links_grown_with_the_floor_and_the_target():
    fake, spec_fn, surfaces = _pipeline("current", 0.020)
    ball = types.SimpleNamespace(centre=np.array([0.1, 0.0, 0.3]), radius=0.04)
    pts = np.array([[0.3, 0.0, 0.3], [0.31, 0.0, 0.3]])
    spec = spec_fn(np.zeros(3), surfaces, np.array([[1.0, 1.0, 1.0]]), pts, ball)
    assert spec["mode"] == "current" and spec["r_m"] == 0.020 and "near" not in spec
    np.testing.assert_allclose(spec["centres"], _Hand.BASE[1:])                            # the arm sphere is not the hand
    np.testing.assert_allclose(spec["radii"], _Hand.RADII[1:] + 0.020)
    np.testing.assert_allclose(spec["z_min"], 0.2 + VOXEL)                                  # support plane + one fine voxel
    assert spec["dest_pad"] == 2 * VOXEL and spec["protect_tol"] == VOXEL
    np.testing.assert_allclose(spec["protect_ball"][0], [0.1, 0.0, 0.3])
    assert spec["protect_ball"][1] == pytest.approx(0.04 + VOXEL)
    np.testing.assert_allclose(spec["protect_points"], pts)
    assert len(fake._hand_carve_history) == 1                                                # this request is remembered


def test_trail_spec_unions_the_history_and_keeps_only_near_voxels():
    past = [np.array([0.01 * k, 0.0, 0.0]) for k in range(1, 12)]                            # 11 past states: only the last 8 count
    fake, spec_fn, surfaces = _pipeline("trail", history=past[-8:])
    fake._hand_carve_history = list(past)
    spec = spec_fn(np.zeros(3), surfaces, None, None, None)
    assert spec["mode"] == "trail" and spec["r_m"] == 0.004 and spec["history"] == 11 and spec["near_mm"] == 20.0
    # the caller keeps the list at 8: this call appends the current state and trims
    assert len(fake._hand_carve_history) == 8 and np.allclose(fake._hand_carve_history[-1], 0.0)
    n_hand = 3
    assert len(spec["centres"]) == n_hand * (11 + 1) and np.allclose(spec["radii"][:n_hand], _Hand.RADII[1:] + 0.004)
    np.testing.assert_allclose(spec["near"]["radii"], _Hand.RADII[1:] + 0.020)
    np.testing.assert_allclose(spec["near"]["centres"], _Hand.BASE[1:])
    assert spec["protect_ball"] is None and spec["protect_points"] is None and spec["destination"] is None


def test_no_state_or_no_model_or_no_hand_link_gives_no_spec():
    fake, spec_fn, surfaces = _pipeline("current")
    assert spec_fn(None, surfaces, None, None, None) is None
    fake._frame_filter_model = None
    assert spec_fn(np.zeros(3), surfaces, None, None, None) is None
    fake._frame_filter_model = _Hand()
    fake.HAND_CARVE_LINKS = ("not_a_link",)
    assert spec_fn(np.zeros(3), surfaces, None, None, None) is None


# ============================================================================ 4 pipeline wiring (fake builder)
def _wired(monkeypatch, carve=None):
    fw = pytest.importorskip("tests.ag3s.test_fine_window")
    ag3s = fw._ag3s(monkeypatch)
    ag3s.hand_voxel_carve = carve
    ag3s.HAND_CARVE_LINKS = ("ee_left", "ee_finger_l1", "ee_finger_l2")     # the fake model's hand (RB-Y1 names; the class constant is Panda's)
    return fw, ag3s


def test_off_is_the_call_as_before(monkeypatch):
    from AG3S.ag3s.runtime.pipeline import AG3S

    fw, off = _wired(monkeypatch)
    assert AG3S.__init__ and off.hand_voxel_carve is None and off._hand_carve_history == []
    field, _ = fw._build(off)
    kw = off._esdf_builder.calls[-1]
    assert "hand_carve" not in kw and "hand_voxel_carve" not in field.stats and off._hand_carve_history == []
    # the call is the same call with the switch present-but-off: same keys
    fw2, other = _wired(monkeypatch)
    fw2._build(other)
    assert kw.keys() == other._esdf_builder.calls[-1].keys()


def _with_record(field):
    field.stats["hand_voxel_carve"] = {"mode": "current", "n_voxels_freed": 7}
    return field


def test_on_passes_the_spec_and_records_a_frame_without_one(monkeypatch):
    fw, on = _wired(monkeypatch, {"mode": "current", "radius_m": 0.020})
    field, _ = fw._build(on)
    spec = on._esdf_builder.calls[-1]["hand_carve"]
    assert spec["mode"] == "current" and spec["r_m"] == 0.020 and len(spec["centres"]) == 3
    # the (fake) builder here writes no record: the pipeline says so instead of staying silent
    assert field.stats["hand_voxel_carve"]["applied"] is False and field.stats["hand_voxel_carve"]["why"] == "builder_wrote_no_record"
    on._esdf_builder.update = lambda cams, _u=on._esdf_builder.update, **kw: _with_record(_u(cams, **kw))
    field, _ = fw._build(on)
    assert field.stats["hand_voxel_carve"] == {"mode": "current", "n_voxels_freed": 7}          # the builder's record is left as it is
    on.reset()
    assert on._hand_carve_history == []
    fw, no_state = _wired(monkeypatch, {"mode": "trail", "radius_m": 0.020})
    field, _ = fw._build(no_state, robot_state=None)
    assert "hand_carve" not in no_state._esdf_builder.calls[-1]
    assert field.stats["hand_voxel_carve"] == {"mode": "trail", "r_mm": 4.0, "n_voxels_freed": 0, "applied": False,
                                               "why": "no_robot_state_or_model"}


# ============================================================================ 5 the builder step
class _FakeBuilderSelf:
    """What `CuroboFieldBuilder._carve_hand` uses of `self`: the TSDF voxel list and the write-back."""

    def __init__(self, centres):
        self.centres = centres
        self.freed = None
        self.boxes = None

    def occupied_voxels(self, itg, boxes=None, **kw):
        self.boxes = boxes
        lo, hi = boxes[0]
        keep = np.all((self.centres >= lo) & (self.centres <= hi), axis=1)
        n = int(keep.sum())
        return np.arange(n), np.arange(n), self.centres[keep]

    def _set_free(self, itg, pool, li, truncation):
        self.freed = (np.asarray(pool).copy(), truncation)


def test_carve_hand_frees_the_hit_voxels_and_records_them():
    pts = _grid((-0.12,) * 3, (0.12,) * 3)
    fake = _FakeBuilderSelf(pts)
    spec = _spec([[0.0, 0.0, 0.0]], [0.035])
    spec.update(r_m=0.020)
    rec = CuroboFieldBuilder._carve_hand(fake, object(), spec, 0.02)
    want = int((np.linalg.norm(pts, axis=1) <= 0.035).sum())
    assert rec["mode"] == "current" and rec["r_mm"] == 20.0 and rec["n_voxels_freed"] == want == rec["n_freed"]
    assert rec["applied"] and rec["n_spheres"] == 1 and len(fake.freed[0]) == want and fake.freed[1] == 0.02
    assert 0 < rec["n_occupied_in_box"] <= len(pts)                                  # only the blocks near the volume were read
    assert fake._hand_carve_freed.shape == (want, 3) and fake._hand_carve_freed.dtype == np.float32
    assert (np.linalg.norm(fake._hand_carve_freed, axis=1) <= 0.035 + 1e-6).all()
    # nothing in the volume: nothing freed, nothing broken
    far = _FakeBuilderSelf(pts + 5.0)
    rec = CuroboFieldBuilder._carve_hand(far, object(), spec, 0.02)
    assert rec["n_voxels_freed"] == 0 and rec["n_occupied_in_box"] == 0 and far.freed is None
    empty = _spec(np.zeros((0, 3)), [])
    empty.update(r_m=0.020)
    assert CuroboFieldBuilder._carve_hand(_FakeBuilderSelf(pts), object(), empty, 0.02)["applied"] is False


def test_builder_update_takes_the_carve_only_as_an_optional_keyword():
    import inspect

    p = inspect.signature(CuroboFieldBuilder.update).parameters["hand_carve"]
    assert p.default is None


def test_record_carries_the_summary_and_the_freed_centres_only_when_on(tmp_path):
    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

    fw = pytest.importorskip("tests.ag3s.test_fine_window")
    w = ConstraintRecordWriter(tmp_path / "rec", esdf_mode="full")

    def cs(field):
        return types.SimpleNamespace(status=None, validity=None, has_target=False, candidates=(), notes=(), profile={}, metrics={},
                                     target=None, esdf=field)

    off = fw._field(fw.APPLE)
    on = fw._field(fw.APPLE)
    on.stats["hand_voxel_carve"] = {"mode": "current", "r_mm": 20.0, "n_voxels_freed": 3, "applied": True}
    on.hand_carve_freed = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.305], [0.1, 0.2, 0.31]], np.float32)
    p_off = w.record(t_step=0, chunk_index=0, constraint_set=cs(off), debug={})
    p_on = w.record(t_step=1, chunk_index=1, constraint_set=cs(on), debug={})
    with np.load(p_off, allow_pickle=True) as z:
        assert "hand_carve_freed_centres" not in z.files and "hand_voxel_carve" not in json.loads(str(z["summary_json"]))
    with np.load(p_on, allow_pickle=True) as z:
        assert z["hand_carve_freed_centres"].shape == (3, 3) and z["hand_carve_freed_centres"].dtype == np.float32
        assert json.loads(str(z["summary_json"]))["hand_voxel_carve"] == {"mode": "current", "r_mm": 20.0, "n_voxels_freed": 3,
                                                                           "applied": True}
    w.close()


# ============================================================================ 6 CLI
BASE = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def _parse(*a):
    from AG3S.trajopt import serve_safe

    ap = serve_safe.build_parser()
    return ap, ap.parse_args([*BASE, *a])


def test_cli_defaults_off_and_the_section():
    from AG3S.trajopt import serve_safe

    _, off = _parse()
    assert off.hand_voxel_carve == "off" and off.hand_voxel_carve_mm is None and serve_safe.hand_voxel_carve_section(off) == {}
    _, cur = _parse("--hand-voxel-carve", "current")
    assert serve_safe.hand_voxel_carve_section(cur) == {"mode": "current", "radius_m": 0.020}   # the default is the fixed rule's cap
    _, cur = _parse("--hand-voxel-carve", "current", "--hand-voxel-carve-mm", "12")
    assert serve_safe.hand_voxel_carve_section(cur) == {"mode": "current", "radius_m": 0.012}
    _, tr = _parse("--hand-voxel-carve", "trail")
    assert serve_safe.hand_voxel_carve_section(tr)["mode"] == "trail"
    assert serve_safe.dynamic_field_section(cur) == {}                  # no esdf config key: the AG3S config dict is unchanged


def _panda(*a):
    ap, args = _parse("--robot", "panda", "--esdf-backend", "curobo", *a)
    return ap, args


def test_cli_accepts_panda_and_refuses_the_rest(capsys):
    from AG3S.trajopt import serve_safe

    ap, args = _panda("--hand-voxel-carve", "current", "--hand-voxel-carve-mm", "20")
    serve_safe.reject_panda_only_flags(ap, args)
    serve_safe.reject_bad_flag_combinations(ap, args)
    # RB-Y1 server refuses
    ap, args = _parse("--esdf-backend", "curobo", "--hand-voxel-carve", "trail")
    with pytest.raises(SystemExit):
        serve_safe.reject_panda_only_flags(ap, args)
    err = capsys.readouterr().err
    assert "--hand-voxel-carve" in err and "RB-Y1" in err
    ap, args = _parse("--esdf-backend", "curobo", "--hand-voxel-carve-mm", "10")
    with pytest.raises(SystemExit):
        serve_safe.reject_panda_only_flags(ap, args)
    # the mm without `current`, a non-positive mm, no curobo backend, no fine layer, no AG3S
    for extra, word in ((("--hand-voxel-carve-mm", "10"), "current"), (("--hand-voxel-carve", "trail", "--hand-voxel-carve-mm", "10"), "current"),
                        (("--hand-voxel-carve", "current", "--hand-voxel-carve-mm", "0"), "양수")):
        ap, args = _panda(*extra)
        with pytest.raises(SystemExit):
            serve_safe.reject_bad_flag_combinations(ap, args)
        assert word in capsys.readouterr().err
    ap, args = _parse("--robot", "panda", "--hand-voxel-carve", "current")                  # legacy backend
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(ap, args)
    assert "curobo" in capsys.readouterr().err
    ap, args = _panda("--hand-voxel-carve", "current", "--fine-voxel", "0")
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(ap, args)
    ap, args = _panda("--hand-voxel-carve", "current", "--no-safe")
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(ap, args)
    assert "--hand-voxel-carve" in capsys.readouterr().err


def test_build_ag3s_sets_the_switch_only_when_asked():
    import inspect

    from AG3S.trajopt import serve_safe

    p = inspect.signature(serve_safe.build_ag3s).parameters["hand_voxel_carve"]
    assert p.default is None
    src = inspect.getsource(serve_safe.ag3s_from_args)
    assert "hand_voxel_carve=hand_voxel_carve_section(args) or None" in src
