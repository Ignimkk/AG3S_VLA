"""AG3S-PRE-CORE-TO SC — 손바닥 바깥 껍질 구 (`--palm-shell {off,on,left,right}`, 기본 off).

Task: `AG3S/ag3s/docs/handoff/AG3S-PRE-CORE-TO.task.md` §6 "SC". 설계: `AG3S-PRE-CORE-TO-SC.impl.md` §2.

이 파일이 고정하는 것:

1. **off 는 그대로** — `palm_shell=None/False/{}` 와 인자 없는 호출이 같은 모델 (이름 · role · 반지름 · FK 비트 동일), CLI 기본
   `off` 는 아무것도 넘기지 않는다 (`build_ag3s` · `ta_offline.build_robot` 에 키가 없다). 기록 chunk 두 개 (E4 1345
   seq 12 · 13) 의 TrajOpt 다시 풀기가 SC 전 코드 (`snapshot_e3c`, `SC/golden_solve_sha.py`) 와 SHA 가 같다.
2. **켜면 맨 뒤에 붙는다** — 꺼진 모델이 켠 모델의 정확한 앞부분 (구 · FK · TrajOpt 거리장 행 값). 270 → 606 (손바닥마다
   +168), `left` 만 270 → 438.
3. **형상** — 대상 (파지 면 · 숨은 면이 아닌 겉면) 이 구 합집합 안이거나 표본 간격 안, 파지 공간 (통로 안 바닥면 아래) 에
   구가 한 점도 들어가지 않는다 (정확한 평면 검사), 넘침 ≤ 5 mm (+ 표본 여유). DX2 막대 접촉점 4 개 · 1345 오렌지 접촉점
   27 개가 덮인다 (꺼진 모델에서는 전부 15 mm 넘게 밖).
4. **권한** — 껍질 구도 link `ee_<side>` 이므로 `exclude_authorized` 의 target-free 마스크가 걸린다.
5. 배선 · 거절: CLI → options, `--no-safe` · `--no-perception` · `--palm-cover` · `--gripper-cover` 와 같이 쓰면 거절.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import types

import numpy as np
import pytest

from common.workspace import workspace_path as _ag3s_workspace_path

MODEL = pathlib.Path(_ag3s_workspace_path("robots/rby1/description/models/rby1a/mujoco/model_transport.xml"))
SERVER = dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05)
FC = {"hands": "both"}
BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]

#: DX2 접촉 4 건의 접촉점 (실제 자세의 `mj_geomDistance(EE_BODY_L, 막대)` 로봇 쪽 점), URDF `ee_left` link frame, mm
#: (`SC/sc_geom.py` → `SC/geom/contact_points_link.json`).
BAR_CONTACTS_MM = np.array([[21.97, -27.89, -73.23], [20.86, -27.89, -73.23],
                            [30.20, -27.89, -73.23], [17.53, -27.89, -73.23]])
#: E4 1345 B_ROI_FPR 손바닥–오렌지 접촉 t 101–127 의 접촉점 (dist < 0.5 mm, geom 1052), link frame, mm
#: (`SC/sc_geom_1345.py` → `SC/geom/orange_1345_points_link.json`).
ORANGE_CONTACTS_MM = np.array([
    [38.63, -27.89, -73.23], [39.86, -27.97, -72.95], [39.96, -28.19, -70.72], [39.33, -27.89, -73.23],
    [39.67, -27.95, -73.02], [39.58, -28.15, -71.99], [39.49, -28.18, -71.34], [39.52, -28.19, -70.78],
    [39.47, -28.19, -70.44], [39.23, -28.19, -70.13], [39.22, -28.19, -69.97], [38.57, -28.12, -72.41],
    [38.97, -28.12, -72.41], [39.11, -28.13, -72.30], [39.03, -28.15, -71.98], [39.00, -28.16, -71.76],
    [38.97, -28.17, -71.60], [39.09, -28.17, -71.56], [39.11, -28.17, -71.50], [39.27, -28.16, -71.78],
    [39.35, -28.17, -71.59], [39.58, -28.18, -71.43], [39.48, -28.19, -71.28], [39.21, -28.19, -71.09],
    [38.90, -28.19, -70.85], [38.79, -28.19, -70.61], [38.69, -28.19, -70.38]])

#: `SC/golden_solve_sha.py` (E4 1345 B_ROI_FPR, `ta_offline.Run` 입력, `TrajOptChunkRefiner.refine`), 2026-10-26 측정:
#: SC 전 코드 `snapshot_e3c` = SC 뒤 코드 (플래그 없음 · `--palm-shell off`), .venv-openpi-live 와 .venv-ag3s 에서 같은 값.
#: `--palm-shell on` 은 seq 13 이 바뀐다 (.venv-openpi-live `fa8d307e4d13cca6`, .venv-ag3s `d5a59b758cad9026` — 환경마다 달라 고정 안 함).
E4_INDEX = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261025T191319+0900_35b909cf096a_cc215238f581/"
                        "verify/AG3S-GPU-STAGES/M2_cpu/e4_index.json")
E4_KEY = "E4/B_ROI_FPR_ep1345_s13453"
GOLDEN = {"seq12": "dcc931c0ed740d85", "seq13": "e6f6986f5730bfc8"}
GOLDEN_ON = {"seq12": "dcc931c0ed740d85", "seq13": "fa8d307e4d13cca6"}


@pytest.fixture(scope="module")
def mj():
    mujoco = pytest.importorskip("mujoco")
    if not MODEL.exists():  # pragma: no cover - asset-dependent
        pytest.skip(f"model not present: {MODEL}")
    return mujoco, mujoco.MjModel.from_xml_path(str(MODEL))


@pytest.fixture(scope="module")
def scene(mj):
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene

    mujoco, model = mj
    return TransportScene.attach(model, mujoco.MjData(model))


def _build(scene, links="gripper", **kw):
    from AG3S.ag3s.experiments.reports.grounding_report import build_constraint_robot_model
    from AG3S.trajopt.serve_safe import constraint_link_filter

    return build_constraint_robot_model(scene, link_filter=constraint_link_filter(links),
                                        sphere_options=dict(SERVER), **kw)


@pytest.fixture(scope="module")
def server(scene):
    return _build(scene, finger_cover=FC)


@pytest.fixture(scope="module")
def ps(scene):
    return _build(scene, finger_cover=FC, palm_shell=True)


def _fk(m, seed=3):
    q = np.random.default_rng(seed).uniform(-1.0, 1.0, m.nq)
    return m.sphere_centers_numeric(q)


def _shell(m, link):
    from AG3S.ag3s.robot_models.urdf_sphere_chain import PALM_SHELL_ROLE

    rows = [(c, r) for (name, c, r), role in zip(m._local_spheres, m.sphere_roles)
            if name == link and role == PALM_SHELL_ROLE]
    return np.asarray([c for c, _ in rows]), np.asarray([r for _, r in rows])


def _gap(points, C, R):
    return np.min(np.linalg.norm(points[:, None, :] - C[None], axis=2) - R[None], axis=1)


# =============================================================================================
# 1. off 는 그대로
# =============================================================================================


@pytest.mark.parametrize("off", [None, False, {}])
@pytest.mark.parametrize("finger", [None, FC])
def test_off_keyword_paths_are_the_same_model(scene, server, off, finger):
    ref = server if finger else _build(scene)
    kw = {"finger_cover": finger} if finger else {}
    m = _build(scene, palm_shell=off, **kw)
    assert m.sphere_link_names == ref.sphere_link_names
    assert m.sphere_roles == ref.sphere_roles
    assert np.array_equal(m.radii, ref.radii)
    a, ra = _fk(ref)
    b, rb = _fk(m)
    assert np.array_equal(a, b) and np.array_equal(ra, rb)
    assert not hasattr(m, "palm_shell_report")


def test_cli_default_off_passes_nothing_and_on_builds_options(caplog):
    import argparse

    from AG3S.trajopt import serve_safe

    p = serve_safe.build_parser()
    args = p.parse_args(BASE_ARGV)
    assert args.palm_shell == "off" and args.palm_shell_overflow is None
    assert serve_safe.palm_shell_options(args) is None
    assert serve_safe.palm_shell_options(p.parse_args(BASE_ARGV + ["--palm-shell", "off"])) is None
    for mode, hands in (("on", "both"), ("left", "left"), ("right", "right")):
        assert serve_safe.palm_shell_options(p.parse_args(BASE_ARGV + ["--palm-shell", mode])) == {"hands": hands}
    assert serve_safe.palm_shell_options(p.parse_args(
        BASE_ARGV + ["--palm-shell", "on", "--palm-shell-overflow", "2.5"])) == {"hands": "both", "overflow": 0.0025}
    with pytest.raises(SystemExit):
        serve_safe.palm_shell_options(p.parse_args(BASE_ARGV + ["--palm-shell-overflow", "5"]))
    for bad in ("0", "-1", "nan", "inf"):
        with pytest.raises(SystemExit):
            serve_safe.palm_shell_options(p.parse_args(
                BASE_ARGV + ["--palm-shell", "on", "--palm-shell-overflow", bad]))
    with pytest.raises(SystemExit):
        p.parse_args(BASE_ARGV + ["--palm-shell", "both"])
    assert serve_safe.palm_shell_options(argparse.Namespace()) is None      # 옛 Namespace 도 off
    with caplog.at_level("WARNING"):
        serve_safe.announce_palm_shell(None)
    assert not caplog.records


@pytest.mark.parametrize("value", [None, {"hands": "both"}, {"hands": "left", "overflow": 0.004}])
def test_build_ag3s_passes_palm_shell_only_when_given(monkeypatch, value):
    if not MODEL.exists():  # pragma: no cover
        pytest.skip("model not present")
    from AG3S.ag3s.experiments.reports import grounding_report
    from AG3S.trajopt import serve_safe

    seen = {}

    class Stop(Exception):
        pass

    def fake(scene, **kw):
        seen.update(kw)
        raise Stop

    monkeypatch.setattr(grounding_report, "build_constraint_robot_model", fake)
    with pytest.raises(Stop):
        serve_safe.build_ag3s(str(MODEL), voxel=0.02, range_max=2.0, links="gripper",
                              finger_cover=FC, palm_shell=value)
    if value is None:
        assert "palm_shell" not in seen
    else:
        assert seen["palm_shell"] == value


def test_server_main_path_passes_no_key_when_off(monkeypatch):
    """`ag3s_from_args` (서버 `main` 과 offline replay 가 AG3S 를 짓는 길): off 면 `build_ag3s` 에 `palm_shell` 키가 없다."""
    from AG3S.trajopt import serve_safe

    seen = []

    class Stop(Exception):
        pass

    def fake(*a, **kw):
        seen.append(kw)
        raise Stop

    monkeypatch.setattr(serve_safe, "build_ag3s", fake)
    p = serve_safe.build_parser()
    for extra in ([], ["--palm-shell", "off"], ["--palm-shell", "on"]):
        with pytest.raises(Stop):
            serve_safe.ag3s_from_args(p.parse_args(BASE_ARGV + extra), plan_horizon_steps=16, rows_per_step=1)
    assert "palm_shell" not in seen[0] and "palm_shell" not in seen[1]
    assert seen[2]["palm_shell"] == {"hands": "both"}


def test_ta_offline_build_robot_passes_palm_shell_only_when_given(monkeypatch):
    if not MODEL.exists():  # pragma: no cover
        pytest.skip("model not present")
    from AG3S.ag3s.experiments.reports import grounding_report
    from AG3S.trajopt import serve_safe
    from AG3S.trajopt.experiments import ta_offline as TA

    seen = []

    def fake(scene, **kw):
        seen.append(kw)
        return object()

    monkeypatch.setattr(grounding_report, "build_constraint_robot_model", fake)
    base = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", str(MODEL), "--finger-cover", "both"]
    p = serve_safe.build_parser()
    TA.build_robot(p.parse_args(base))
    TA.build_robot(p.parse_args(base + ["--palm-shell", "off"]))
    TA.build_robot(p.parse_args(base + ["--palm-shell", "on"]))
    assert "palm_shell" not in seen[0] and "palm_shell" not in seen[1]
    assert seen[2]["palm_shell"] == {"hands": "both"}
    assert TA.variant_server_args("PS") == ("--palm-shell", "on")


def test_rejected_combinations():
    from AG3S.trajopt import serve_safe

    def check(extra, ok):
        p = serve_safe.build_parser()
        a = p.parse_args(BASE_ARGV + extra)
        if ok:
            serve_safe.reject_bad_flag_combinations(p, a)
        else:
            with pytest.raises(SystemExit):
                serve_safe.reject_bad_flag_combinations(p, a)

    for extra in (["--no-safe"], ["--no-perception"]):
        check(extra + ["--palm-shell", "on"], False)
        check(extra + ["--palm-shell", "off"], True)
    check(["--palm-shell", "on", "--palm-cover", "on"], False)
    assert "--gripper-cover" in _flags()
    check(["--palm-shell", "on", "--gripper-cover"], False)
    check(["--palm-shell", "on"], True)
    check(["--palm-shell", "off", "--palm-cover", "on"], True)


def _flags():
    from AG3S.trajopt import serve_safe

    return {s for a in serve_safe.build_parser()._actions for s in a.option_strings}


def test_function_rejects_palm_cover_and_gripper_cover(scene):
    with pytest.raises(ValueError):
        _build(scene, finger_cover=FC, palm_shell=True, palm_cover=True)
    with pytest.raises(ValueError):
        _build(scene, finger_cover=FC, palm_shell=True, gripper_cover=True)
    with pytest.raises(ValueError):
        _build(scene, finger_cover=FC, palm_shell={"hands": "middle"})
    with pytest.raises(ValueError):
        _build(scene, finger_cover=FC, palm_shell={"overflow": 0.0})
    with pytest.raises(ValueError):
        _build(scene, finger_cover=FC, palm_shell={"bogus": 1})


def _solve_shas(extra):
    """`SC/golden_solve_sha.py` 와 같은 계산."""
    from AG3S.trajopt.experiments import ta_offline as TA
    from AG3S.trajopt.refiner import TrajOptChunkRefiner

    entry = json.load(open(E4_INDEX))["runs"][E4_KEY]
    run = TA.Run(E4_KEY, entry, types.SimpleNamespace(hm=False, no_plane_rows=False), extra_server_args=tuple(extra))
    out = {}
    for seq in (12, 13):
        inp, _ = run.inputs(seq)
        holder = {"v": (inp["scene"], inp["q_now"], True)}
        ref = TrajOptChunkRefiner(run.robot, run.layout, lambda _c: holder["v"], run.base_cfg)
        ref.optimizer.linearizer.set_joint_parameter_path(
            run.finger_path(inp["summary"], inp["ref_chunk"], run.base_cfg.horizon.planned))
        c = ref.refine(inp["ref_chunk"], {"t_step": seq, "previous_physical_chunk": None})
        out[f"seq{seq}"] = hashlib.sha1(np.ascontiguousarray(np.asarray(c, np.float64)).tobytes()).hexdigest()[:16]
    return out, run


def _have_e4():
    if not E4_INDEX.exists():
        return False
    entry = json.load(open(E4_INDEX))["runs"].get(E4_KEY)
    return bool(entry) and pathlib.Path(entry["server_segment"]["first"]).exists()


@pytest.mark.skipif(not _have_e4(), reason="E4 1345 기록이 없다")
def test_off_recorded_chunk_resolve_is_bit_identical_to_pre_sc_code():
    pytest.importorskip("casadi")
    pytest.importorskip("osqp")
    a, _ = _solve_shas(())
    b, _ = _solve_shas(("--palm-shell", "off"))
    assert a == GOLDEN and b == GOLDEN


@pytest.mark.skipif(not _have_e4(), reason="E4 1345 기록이 없다")
def test_on_recorded_chunk_rows_keep_the_off_prefix():
    """같은 기록 chunk: 켠 모델의 TrajOpt 거리장 행 값에서 앞 270 열이 꺼진 모델과 같고, 다시 풀기는 seq 13 만 바뀐다."""
    pytest.importorskip("casadi")
    pytest.importorskip("osqp")
    on, run_on = _solve_shas(("--palm-shell", "on"))
    # seq 12: no shell row is active -> the off solve, bit for bit. seq 13: shell rows change the solve. The exact on SHA
    # (`GOLDEN_ON`, .venv-openpi-live) depends on the BLAS / numpy build (.venv-ag3s gives another), so it is not pinned.
    assert on["seq12"] == GOLDEN["seq12"] == GOLDEN_ON["seq12"]
    assert on["seq13"] != GOLDEN["seq13"]
    from AG3S.trajopt.experiments import ta_offline as TA

    entry = json.load(open(E4_INDEX))["runs"][E4_KEY]
    run_off = TA.Run(E4_KEY, entry, types.SimpleNamespace(hm=False, no_plane_rows=False))
    n0 = run_off.robot.n_spheres
    for seq in (12, 13):
        vals = []
        for run in (run_off, run_on):
            inp, _ = run.inputs(seq)
            traj = run.layout.chunk_to_trajectory(inp["rec_chunk"])[:, :16]
            lin, cen = TA.states_of(run, traj, inp["q_now"], inp["scene"],
                                    run.finger_path(inp["summary"], inp["ref_chunk"], 16))
            vals.append(lin._esdf_clearance(cen, inp["scene"])[..., 0])
        assert vals[1].shape[1] == vals[0].shape[1] + 2 * 168
        assert np.array_equal(vals[1][:, :n0], vals[0])


# =============================================================================================
# 2. 켜면 맨 뒤에 붙는다
# =============================================================================================


def test_on_appends_after_an_exact_off_prefix(server, ps):
    from AG3S.ag3s.robot_models.urdf_sphere_chain import PALM_SHELL_ROLE

    n0 = server.n_spheres
    assert n0 == 270 and ps.n_spheres == 270 + 2 * 168
    assert ps.sphere_link_names[:n0] == server.sphere_link_names
    assert ps.sphere_roles[:n0] == server.sphere_roles
    assert np.array_equal(ps.radii[:n0], server.radii)
    a, _ = _fk(server)
    b, _ = _fk(ps)
    assert np.array_equal(b[:n0], a)
    assert set(ps.sphere_roles[n0:]) == {PALM_SHELL_ROLE}
    assert set(ps.sphere_link_names[n0:]) == {"ee_left", "ee_right"}
    rep = ps.palm_shell_report
    assert rep["n_spheres_before"] == 270 and rep["n_spheres_after"] == 606
    assert rep["options"] == {"hands": "both", "overflow": 0.005, "face_cos": 0.5}
    assert not hasattr(ps, "palm_cover_report")
    assert ps.finger_cover_report == server.finger_cover_report


def test_left_only_and_radius_scale_untouched(scene, server, ps):
    m = _build(scene, finger_cover=FC, palm_shell={"hands": "left"})
    assert m.n_spheres == 270 + 168 and set(m.sphere_link_names[270:]) == {"ee_left"}
    # radius scale 은 껍질 구에 걸리지 않는다 (반지름 그대로)
    from AG3S.ag3s.experiments.reports.grounding_report import build_constraint_robot_model
    from AG3S.trajopt.serve_safe import constraint_link_filter

    m2 = build_constraint_robot_model(scene, link_filter=constraint_link_filter("gripper"),
                                      sphere_options={**SERVER, "capsule_radius_scale": 0.1},
                                      finger_cover=FC, palm_shell=True)
    assert np.array_equal(m2.radii[-2 * 168:], ps.radii[-2 * 168:])


def test_max_opening_and_grasp_limits_are_unchanged(server, ps):
    from AG3S.ag3s.experiments.sources.gripper_cover import grasp_limits

    assert ps.gripper_openings() == server.gripper_openings()
    for palm in ("ee_left", "ee_right"):
        a_in, a_ch = grasp_limits(server, palm)
        b_in, b_ch = grasp_limits(ps, palm)
        assert a_in is None and b_in is None
        assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a_ch, b_ch))


# =============================================================================================
# 3. 형상
# =============================================================================================


@pytest.fixture(scope="module")
def shell_parts(mj, server):
    from AG3S.ag3s.experiments.sources.gripper_cover import PALM_COVER_SPACING, link_surface, palm_shell_parts

    _, model = mj
    surf = link_surface(model, server, "ee_left", spacing=PALM_COVER_SPACING)
    mask, keep, info = palm_shell_parts(surf, server, "ee_left")
    return surf, mask, keep, info


def test_targets_exclude_only_the_channel_bottom_face_and_hidden(shell_parts):
    surf, mask, keep, info = shell_parts
    (A, b), = keep
    u, tip = A[0], -A[2]
    half, z_bot = -b[0], b[2]
    face = (np.abs(surf.points @ u) < half) & (surf.normals @ tip > 0.5)
    assert np.array_equal(mask, ~face & ~surf.hidden)
    assert info["n_grasp_face"] == int(face.sum()) > 0
    assert half * 1e3 == pytest.approx(47.159, abs=0.01)
    assert z_bot * 1e3 == pytest.approx(73.690, abs=0.01)
    # 막대 · 오렌지 접촉 자리 (−y 바닥 모서리) 는 대상, 같은 높이의 통로 한가운데 바닥면은 대상이 아니다
    for p_mm in (BAR_CONTACTS_MM[0], ORANGE_CONTACTS_MM[0]):
        i = int(np.argmin(np.linalg.norm(surf.points - p_mm / 1e3, axis=1)))
        assert mask[i]
    i = int(np.argmin(np.linalg.norm(surf.points - np.array([0.0, 0.0, -0.0737]), axis=1)))
    assert not mask[i]


def test_shell_spheres_never_enter_the_grasp_volume(ps, shell_parts):
    _, _, keep, _ = shell_parts
    (A, b), = keep
    for link in ("ee_left", "ee_right"):
        C, R = _shell(ps, link)
        assert R.size == 168
        # 지지 평면 거리 ≥ 반지름 ⇒ 구는 볼록 영역과 겹치지 않는다 (정확한 보장)
        assert np.all(np.max(C @ A.T + b, axis=1) >= R - 1e-12)
        # 그래서 통로 안 구는 손바닥 mesh 바닥면보다 손끝 쪽으로 나오지 않는다
        tip, half, z_bot = -A[2], -b[0], b[2]
        inside = np.abs(C @ A[0]) + R <= half
        assert np.all(C[inside] @ tip + R[inside] <= z_bot + 1e-12)


def test_targets_are_covered_and_overflow_is_bounded(ps, shell_parts):
    from AG3S.ag3s.experiments.sources.gripper_cover import _sampled_overflow, finger_body_targets

    surf, mask, _, _ = shell_parts
    C, R = _shell(ps, "ee_left")
    # 대상 = 맞붙어 묻힌 면이 아닌 겉면 (`finger_body_targets`, palm cover 와 같은 규칙) ∧ 껍질 마스크
    g = _gap(surf.points[finger_body_targets(surf) & mask], C, R)
    rep = ps.palm_shell_report["links"]["ee_left"]
    assert int((g > 1e-12).sum()) == rep["n_uncovered"] <= 30          # keep-out 에 맞닿은 바닥 모서리 표본
    assert g.max() * 1e3 <= 1.0                                         # 그것도 1 mm (표본 간격) 안
    assert rep["max_gap_mm"] == pytest.approx(g.max() * 1e3, abs=1e-9)
    assert _sampled_overflow(C, R, surf.hulls, surf.points) * 1e3 <= 5.0 + 0.2
    assert rep["keep_out_max_entry_mm"] <= 1e-9


def test_bar_and_orange_contact_points_are_covered(server, ps):
    c0, r0 = server.local_spheres("ee_left")
    C, R = _shell(ps, "ee_left")
    for pts_mm in (BAR_CONTACTS_MM, ORANGE_CONTACTS_MM):
        p = pts_mm / 1e3
        assert np.all(_gap(p, c0, r0) * 1e3 > 15.0)                     # 지금 모델: 전부 15 mm 넘게 밖
        assert np.all(_gap(p, C, R) * 1e3 <= 0.7)                       # 껍질: 안이거나 0.7 mm (표본 간격) 안


def test_right_hand_is_the_mirror_build(ps):
    """왼손 · 오른손은 link frame 에서 mesh · 통로가 같아 같은 구다 (캐시 한 번)."""
    a = _shell(ps, "ee_left")
    b = _shell(ps, "ee_right")
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


# =============================================================================================
# 4. 권한
# =============================================================================================


def test_every_shell_sphere_of_the_grasping_hand_reads_the_target_free_layer(ps):
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.constraints.clearance import ClearancePolicy
    from AG3S.ag3s.types import ContactPolicyContext

    cfg = AG3SConfig.from_dict({"collision_backend": "esdf"})
    policy = ClearancePolicy.from_config(cfg.contact, cfg.geometry, cfg.support_surface)
    authorized = set(policy.authorized_links(ContactPolicyContext.make("approach", ["left"])))
    names = list(ps.sphere_link_names)
    shell = [i for i in range(270, ps.n_spheres) if names[i] == "ee_left"]
    assert len(shell) == 168 and "ee_left" in authorized
    mask = np.asarray([n in authorized for n in names], bool)
    assert mask[shell].all()
    assert not mask[[i for i in range(270, ps.n_spheres) if names[i] == "ee_right"]].any()


def test_announce_names_the_counts(ps, caplog):
    from AG3S.trajopt import serve_safe

    with caplog.at_level("WARNING"):
        serve_safe.announce_palm_shell(ps.palm_shell_report)
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "270 → 606" in msg and "ee_left +168" in msg
