"""T96b (D27) — 손바닥 덮개 구 행만의 obstacle 여유 (`--palm-cover-margin MM`, 기본 = 안 줌 = `--obstacle-margin`).

Task: `AG3S/ag3s/docs/handoff/T96.task.md` + lead 판정 D27 (결과 전). 보고: `T96.impl.md` "T96b (D27 · D28)".
`--palm-shell-margin` (SCM, `test_scm_palm_shell_margin.py`) 과 같은 규칙을 role `palm_body` (`--palm-cover on`) 행에 건다.

1. **안 주면 그대로** — CLI 기본은 `collision` 에 키를 넣지 않고, `TrajOptConfig.to_dict` 에도 키가 없다. scene snapshot ·
   `to.metrics` · client metadata · 최악 행 · 서버 시작 줄에 키 · 줄이 없다.
2. **거절** — `--palm-cover on` 없이, `--obstacle-margin 0` 으로, 음수 · nan · inf 로 주면 서버 시작 거절; config 는 ValueError;
   `palm_body` 구가 없는 모델이면 linearizer 가 ValueError.
3. **행 규칙** — `palm_body` 열 중 분류가 `obstacle` 인 행만 여유가 `max(margin, palm_cover_margin)`; 나머지 열 · 분류는 비트로
   같다 (기록 chunk E4 1345 seq 12 · 13, 덮개 모델). 같은 값 (= `--obstacle-margin`) 을 주면 비트로 같다.
4. **껍질 경로 불변** — `palm_shell_margin` 만 준 linearizer 의 행 값이 T96b 전 식과 같다 (일반화가 껍질 경로를 바꾸지 않음).
5. Q server-arg (`server_flags_q2_delta*.txt`) 가 parse · 거절 검사 · config 를 통과한다.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import shlex
import types

import numpy as np
import pytest

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]
OM = ["--obstacle-margin", "0.010"]
PCM = ["--palm-cover", "on", "--palm-cover-margin", "5"]
RUN = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261104T050351+0900_054748e6a73c_34e19a21953d/"
                   "impl/T96")


def _parse(extra):
    from AG3S.trajopt import serve_safe

    p = serve_safe.build_parser()
    return p, p.parse_args(BASE_ARGV + list(extra))


@pytest.fixture
def no_cuda_check(monkeypatch):
    from AG3S.trajopt import gpu_seeds

    monkeypatch.setattr(gpu_seeds, "available", lambda: (True, ""))


# =============================================================================================
# 1 · 2. CLI · config · 거절
# =============================================================================================


def test_cli_default_gives_no_key():
    from AG3S.trajopt import serve_safe

    for extra in ([], ["--palm-cover", "on"], OM + ["--palm-cover", "on"]):
        _, a = _parse(extra)
        assert a.palm_cover_margin is None
        assert serve_safe.palm_cover_margin_overrides(a) == {}
    _, a = _parse(OM + PCM)
    assert serve_safe.palm_cover_margin_overrides(a) == {"palm_cover_margin": 0.005}
    assert serve_safe.palm_cover_margin_overrides(types.SimpleNamespace()) == {}      # 옛 Namespace


def test_rejected_combinations(no_cuda_check):
    from AG3S.trajopt import serve_safe

    def check(extra, ok):
        p, a = _parse(extra)
        if ok:
            serve_safe.reject_bad_flag_combinations(p, a)
        else:
            with pytest.raises(SystemExit):
                serve_safe.reject_bad_flag_combinations(p, a)

    check(OM + ["--palm-cover-margin", "5"], False)                          # 덮개 없음
    check(OM + ["--palm-cover", "off", "--palm-cover-margin", "5"], False)
    check(OM + ["--palm-shell", "on", "--palm-cover-margin", "5"], False)    # 껍질은 덮개가 아니다
    check(PCM, False)                                                        # --obstacle-margin 0 (기본)
    for bad in ("-1", "nan", "inf"):
        check(OM + ["--palm-cover", "on", "--palm-cover-margin", bad], False)
    check(OM + PCM, True)
    check(OM + ["--palm-cover", "on", "--palm-cover-margin", "0"], True)
    check(OM + ["--palm-cover", "on"], True)
    for extra in (OM + ["--palm-cover-margin", "5"], PCM):
        _, a = _parse(extra)
        with pytest.raises(SystemExit):
            serve_safe.trajopt_config_from_args(a)


def test_config_default_absent_from_to_dict_and_validated():
    from AG3S.trajopt.config import TrajOptConfig

    base = {"backend": "esdf", "obstacle_margin": 0.01}
    c = TrajOptConfig.from_dict({"collision": base})
    assert c.collision.palm_cover_margin is None
    assert "palm_cover_margin" not in c.to_dict()["collision"]
    assert TrajOptConfig.from_dict(c.to_dict()) == c
    on = TrajOptConfig.from_dict({"collision": {**base, "palm_cover_margin": 0.005}})
    assert on.to_dict()["collision"]["palm_cover_margin"] == 0.005
    assert TrajOptConfig.from_dict(on.to_dict()) == on
    assert c.with_overrides({"collision": {"palm_cover_margin": 0.005}}) == on
    d0, d1 = c.to_dict(), on.to_dict()
    assert d1["collision"].pop("palm_cover_margin") == 0.005 and d0 == d1
    for bad in (-0.001, float("nan"), float("inf"), True, "5"):
        with pytest.raises(ValueError, match="palm_cover_margin"):
            TrajOptConfig.from_dict({"collision": {**base, "palm_cover_margin": bad}})
    with pytest.raises(ValueError, match="palm_cover_margin"):
        TrajOptConfig.from_dict({"collision": {"backend": "esdf", "palm_cover_margin": 0.005}})   # obstacle_margin 0


def test_announce_metrics_metadata_snapshot_only_when_on(caplog):
    from AG3S.trajopt import serve_safe
    from AG3S.trajopt.config import TrajOptConfig
    from AG3S.trajopt.safe_policy import SafePolicy
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    off = TrajOptConfig.from_dict({"collision": {"backend": "esdf", "obstacle_margin": 0.01}})
    on = off.with_overrides({"collision": {"palm_cover_margin": 0.005}})
    both = on.with_overrides({"collision": {"palm_shell_margin": 0.004}})
    with caplog.at_level("WARNING"):
        serve_safe.announce_palm_cover_margin(off.collision)
    assert not caplog.records
    with caplog.at_level("WARNING"):
        serve_safe.announce_palm_cover_margin(on.collision)
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "palm_cover_margin: 5.0 mm" in msg and "obstacle_margin 10.0 mm" in msg

    def scm(cfg):
        o = TrajectoryOptimizer.__new__(TrajectoryOptimizer)
        o.config = cfg
        return o._scm_record()

    assert scm(off) == {}
    assert scm(on) == {"palm_cover_margin_m": 0.005, "obstacle_margin_m": 0.01}
    assert scm(both) == {"palm_shell_margin_m": 0.004, "palm_cover_margin_m": 0.005, "obstacle_margin_m": 0.01}
    assert scm(off.with_overrides({"collision": {"palm_shell_margin": 0.004}})) == {
        "palm_shell_margin_m": 0.004, "obstacle_margin_m": 0.01}                  # SCM as before

    def meta(cfg):
        sp = SafePolicy.__new__(SafePolicy)
        sp._policy = None
        sp.linearizer = types.SimpleNamespace(n_spheres=466)
        sp.to_config = cfg
        sp.shadow = False
        sp.ag3s = types.SimpleNamespace()
        sp.task_gate = "off"
        sp.held_3d = None
        return sp.metadata

    m0, m1 = meta(off), meta(on)
    assert "palm_cover_margin_m" not in m0
    assert m1.pop("palm_cover_margin_m") == 0.005 and m0 == m1


def test_snapshot_key_only_when_on():
    """`scene_from_constraint_set` (the server path) puts `palm_cover_margin` on the snapshot only when it is set."""
    T30 = pytest.importorskip("tests.trajopt.test_t30_target_free_margin")
    from AG3S.trajopt.config import TrajOptConfig
    from AG3S.trajopt.linearize import SceneSnapshot, scene_from_constraint_set

    assert SceneSnapshot.__dataclass_fields__["palm_cover_margin"].default is None
    _, _, field, points = T30._unit_rig()
    cs = T30._constraint_set(field, points, policy="exclude_authorized", mask=[True, False, False],
                             margins=[T30.APPROACH_MARGIN, T30.FULL_MARGIN, T30.FULL_MARGIN])
    radii = np.array([T30.R_FINGER, 0.02, T30.R_FINGER])
    off = TrajOptConfig.from_dict({"collision": {"backend": "esdf", "obstacle_margin": 0.01}})
    on = off.with_overrides({"collision": {"palm_cover_margin": 0.005}})
    s0, s1 = scene_from_constraint_set(cs, radii, off), scene_from_constraint_set(cs, radii, on)
    assert s0.palm_cover_margin is None and s1.palm_cover_margin == 0.005
    for f in SceneSnapshot.__dataclass_fields__:                         # every other field: the same
        if f == "palm_cover_margin":
            continue
        a, b = getattr(s0, f), getattr(s1, f)
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b), f
        elif isinstance(a, dict):
            assert a.keys() == b.keys(), f
        else:
            assert a is b or a == b, f


def test_ta_offline_variant():
    from AG3S.trajopt.config import TrajOptConfig
    from AG3S.trajopt.experiments import ta_offline as TA

    assert TA.variant_server_args("P0M_PC5") == ("--drop:--palm-shell", "--drop:--palm-shell-margin", "--palm-cover",
                                                 "on", "--palm-cover-margin", "5")
    base = TrajOptConfig.from_dict({"collision": {"backend": "esdf", "obstacle_margin": 0.01,
                                                  "palm_shell_margin": 0.005}})
    v = TA.variant_config(base, "P0M_PC5", False).collision
    assert v.palm_cover_margin == 0.005 and v.palm_shell_margin is None
    assert TA.variant_config(base, "P0M_PC", False).collision.palm_cover_margin is None


# =============================================================================================
# 2 · 4. linearizer 의 role 열 (가짜 모델)
# =============================================================================================


def _fake_lin(roles):
    from AG3S.trajopt.linearize import CollisionLinearizer

    lin = CollisionLinearizer.__new__(CollisionLinearizer)
    lin.robot_model = types.SimpleNamespace(sphere_roles=tuple(roles))
    lin.n_spheres = len(roles)
    return lin


def test_role_columns_and_refusals():
    roles = ("collision", "palm_body", "palm_body", "finger_body", "palm_shell")
    lin = _fake_lin(roles)
    np.testing.assert_array_equal(lin._palm_body_cols(7), [0, 1, 1, 0, 0, 0, 0])   # held rows: False
    np.testing.assert_array_equal(lin._palm_shell_cols(5), [0, 0, 0, 0, 1])
    np.testing.assert_array_equal(lin._palm_body_cols(2), [0, 1])
    with pytest.raises(ValueError, match="no palm_body sphere"):
        _fake_lin(("collision", "palm_shell"))._palm_body_cols(2)
    with pytest.raises(ValueError, match="no palm_shell sphere"):                     # SCM message as before
        _fake_lin(("collision", "palm_body"))._palm_shell_cols(2)
    bad = _fake_lin(("collision", "palm_body"))
    bad.n_spheres = 3
    with pytest.raises(ValueError, match="palm_cover_margin: the robot model names 2 sphere roles"):
        bad._palm_body_cols(3)


# =============================================================================================
# 3. 기록 chunk (E4 1345 B_ROI_FPR seq 12 · 13) — 덮개 모델의 행 값
# =============================================================================================

E4_INDEX = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261025T191319+0900_35b909cf096a_cc215238f581/"
                        "verify/AG3S-GPU-STAGES/M2_cpu/e4_index.json")
E4_KEY = "E4/B_ROI_FPR_ep1345_s13453"


def _have_e4():
    if not E4_INDEX.exists():
        return False
    entry = json.load(open(E4_INDEX))["runs"].get(E4_KEY)
    return bool(entry) and pathlib.Path(entry["server_segment"]["first"]).exists()


_RUNS: dict = {}


def _run(extra):
    from AG3S.trajopt.experiments import ta_offline as TA

    key = tuple(extra)
    if key not in _RUNS:
        entry = json.load(open(E4_INDEX))["runs"][E4_KEY]
        _RUNS[key] = TA.Run(E4_KEY, entry, types.SimpleNamespace(hm=False, no_plane_rows=False),
                            extra_server_args=key)
    return _RUNS[key]


def _rows(run, seq, scene=None):
    from AG3S.trajopt.experiments import ta_offline as TA

    inp, _ = run.inputs(seq)
    scene = inp["scene"] if scene is None else scene
    traj = run.layout.chunk_to_trajectory(inp["rec_chunk"])[:, :16]
    lin, cen = TA.states_of(run, traj, inp["q_now"], scene,
                            run.finger_path(inp["summary"], inp["ref_chunk"], 16))
    vals = lin._esdf_clearance(cen, scene)[..., 0][:, :run.robot.n_spheres]
    cls = np.asarray(lin._last_obstacle_class)[:, :run.robot.n_spheres]
    return vals, cls, lin, traj, inp


needs_e4 = pytest.mark.skipif(not _have_e4(), reason="E4 1345 기록이 없다")


@needs_e4
def test_cover_rows_move_only_on_obstacle_class_and_equal_value_is_bit_identical():
    pytest.importorskip("casadi")
    from AG3S.ag3s.robot_models.urdf_sphere_chain import PALM_BODY_ROLE

    pc = _run(("--palm-cover", "on"))
    assert pc.base_cfg.collision.palm_cover_margin is None
    body = np.asarray([r == PALM_BODY_ROLE for r in pc.robot.sphere_roles], bool)
    assert pc.robot.n_spheres == 466 and int(body.sum()) > 0
    n_moved = 0
    for seq in (12, 13):
        v0, c0, _, _, inp = _rows(pc, seq)
        assert inp["scene"].palm_cover_margin is None
        s5 = dataclasses.replace(inp["scene"], palm_cover_margin=0.005)
        s10 = dataclasses.replace(inp["scene"], palm_cover_margin=0.010)    # = obstacle_margin
        v1, c1, *_ = _rows(pc, seq, s5)
        v2, c2, *_ = _rows(pc, seq, s10)
        assert np.array_equal(c0, c1) and np.array_equal(c0, c2)
        assert np.array_equal(v0, v2), "the same value as obstacle_margin is bit-identical"
        assert np.array_equal(v0[:, ~body], v1[:, ~body])                    # every other sphere: bit for bit
        moved = (c1 == 0) & body[None, :]                                    # obstacle class on a palm-cover row
        assert np.allclose(v1[moved] - v0[moved], 0.005, rtol=0, atol=1e-12)
        assert np.array_equal(v0[~moved], v1[~moved])                        # target · support · held: untouched
        n_moved += int(moved.sum())
    assert n_moved > 0


@needs_e4
def test_worst_row_names_the_cover_margin_only_when_on():
    pytest.importorskip("casadi")
    pc = _run(("--palm-cover", "on"))
    for on in (False, True):
        _, _, lin, traj, inp = _rows(pc, 13)
        scene = dataclasses.replace(inp["scene"], palm_cover_margin=0.005) if on else inp["scene"]
        _, pair = lin.worst_row(traj, inp["q_now"], scene)
        assert pair is not None and "obstacle_margin_class" in pair
        if on:
            assert "palm_cover_row" in pair and ("palm_cover_margin_m" in pair) == pair["palm_cover_row"]
        else:
            assert "palm_cover_row" not in pair and "palm_cover_margin_m" not in pair


@needs_e4
def test_shell_path_is_unchanged_and_no_cover_model_is_refused():
    """4 — the shell-only margin through the generalised role check is the pre-T96b formula; a cover margin on the
    shell model (no `palm_body` sphere) is refused."""
    pytest.importorskip("casadi")
    from AG3S.ag3s.robot_models.urdf_sphere_chain import PALM_SHELL_ROLE

    ps = _run(("--palm-shell", "on"))
    shell = np.asarray([r == PALM_SHELL_ROLE for r in ps.robot.sphere_roles], bool)
    v0, c0, _, _, inp = _rows(ps, 13)
    v1, c1, *_ = _rows(ps, 13, dataclasses.replace(inp["scene"], palm_shell_margin=0.005))
    om = 0.010
    # pre-T96b: margin = where(cls == obstacle, max(margin, where(shell, 0.005, om)), margin) — with esdf_margin 0 the
    # obstacle rows' margin was om, so a shell obstacle row gains exactly om − 0.005
    moved = (c1 == 0) & shell[None, :]
    assert np.array_equal(c0, c1) and moved.any()
    assert np.allclose(v1[moved] - v0[moved], om - 0.005, rtol=0, atol=1e-12)
    assert np.array_equal(v0[~moved], v1[~moved])
    with pytest.raises(ValueError, match="no palm_body sphere"):
        _rows(ps, 13, dataclasses.replace(inp["scene"], palm_cover_margin=0.005))


# =============================================================================================
# 5. Q server-arg 파일 (T96b 새 파일)
# =============================================================================================


@pytest.mark.skipif(not (RUN / "server_flags_q2_delta0.txt").exists(), reason="Q2 server-arg 파일이 없다")
@pytest.mark.parametrize("name", ["server_flags_q2_delta0.txt", "server_flags_q2_delta5.txt"])
def test_q2_server_args_parse_and_build(name, no_cuda_check):
    from AG3S.trajopt import serve_safe

    line = (RUN / name).read_text().strip()
    old = (RUN / name.replace("_q2_", "_q_")).read_text().strip()
    assert shlex.split(line) == shlex.split(old) + ["--palm-cover-margin", "5"]
    p, a = _parse(shlex.split(line))
    serve_safe.reject_bad_flag_combinations(p, a)
    c = serve_safe.trajopt_config_from_args(a)
    assert c.collision.palm_cover_margin == 0.005 and c.collision.palm_shell_margin is None
    assert c.collision.obstacle_margin == 0.010
    assert a.surface_solid_owned and a.palm_cover == "on" and a.palm_shell == "off"
    assert a.surface_solid_dn_support is None                               # D28 default (config exclude)
