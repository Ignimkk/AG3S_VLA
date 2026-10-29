"""AG3S-PRE-CORE-TO SCM — 손바닥 껍질 구 행만의 obstacle 여유 (`--palm-shell-margin MM`, 기본 = 안 줌 = `--obstacle-margin`).

Task: `AG3S/ag3s/docs/handoff/AG3S-PRE-CORE-TO.task.md` §6 "CR · SCM · LTE". 보고: `AG3S-PRE-CORE-TO-SCM.impl.md`.

이 파일이 고정하는 것:

1. **안 주면 그대로** — CLI 기본은 `collision` 에 키를 넣지 않고, `TrajOptConfig.to_dict` 에도 키가 없다 (기록 키 불변). 기록
   chunk 두 개 (E4 1345 seq 12 · 13) 의 다시 풀기 SHA 가 SC golden (`dcc931c0ed740d85` · `e6f6986f5730bfc8`) 과 같다.
2. **일반 여유와 같은 값을 주면 그대로** — `--palm-shell on --palm-shell-margin 10` (= `--obstacle-margin 0.010`) 의 TrajOpt 행
   값 · 다시 풀기가 `--palm-shell on` 과 비트로 같다.
3. **5 mm** — 껍질 열 (role `palm_shell`) 중 분류가 `obstacle` 인 행만 +5 mm, 나머지 열 · 다른 분류 행은 비트로 같다.
   `to.metrics` 에 `palm_shell_margin_m`, client metadata · 최악 행 · 서버 시작 줄에 값이 남는다 (꺼지면 키 · 줄 없음).
4. **거절** — `--palm-shell` 없이, `--obstacle-margin 0` 으로, 음수 · nan 으로 주면 서버 시작 거절; config 는 ValueError;
   껍질 구가 없는 모델로 그 scene 을 풀면 linearizer 가 ValueError (켰다고 믿은 채 아무 일도 없는 상태를 막는다).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import shlex
import types

import numpy as np
import pytest

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]
#: E3 closed-loop 의 SAFE flag 전부 (`verify/AG3S-PRE-CORE-TO/E3/closed_occ/p/launch_occ.sh`, 2026-10-26).
E3_SAFE = (
    "--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper "
    "--target-field-policy exclude-authorized --esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 "
    "--sphere-spacing 0.4 --max-spheres-per-capsule 32 --obstacle-margin 0.010 --held-body-cover --closing-keep-geometry "
    "--latched-identity-hold --supported-max-bottom-mm 60 --held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3 "
    "--finger-cover both --plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4 "
    "--w-deflection-rate 20 --to-fast-eval --esdf-host-mirror --target-exclusion ball-owned --owned-max-dist-voxels 3 "
    "--attention-split-radius-mm 12 --attention-split-ratio-source raw --target-permission-strict "
    "--attention-source noun_target:L10H7 --attention-split-partition --split-supersedes-anchor --wrist-track-scale 0.1 "
    "--neighbour-contact-mm 20 --nontarget-evidence surface --placed-no-seed-certify --kept-geometry-follow --roi-carve "
    "--roi-carve-footprint --roi-carve-footprint-pad-px 0 --roi-carve-footprint-robot ignore --roi-carve-device cuda "
    "--blas-threads 1 --to-multi-seed 16 --to-gpu-seeds curobo --to-gpu-seeds-check first-pass")
SCM = ["--palm-shell", "on", "--palm-shell-margin", "5"]

E4_INDEX = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261025T191319+0900_35b909cf096a_cc215238f581/"
                        "verify/AG3S-GPU-STAGES/M2_cpu/e4_index.json")
E4_KEY = "E4/B_ROI_FPR_ep1345_s13453"
#: SC golden (`SC/golden_solve_sha.py`, SC 전 코드 `snapshot_e3c` 와 같음, 두 venv 같은 값).
GOLDEN = {"seq12": "dcc931c0ed740d85", "seq13": "e6f6986f5730bfc8"}


def _parse(extra):
    from AG3S.trajopt import serve_safe

    p = serve_safe.build_parser()
    return p, p.parse_args(BASE_ARGV + list(extra))


# =============================================================================================
# 1. 안 주면 그대로 · CLI · config
# =============================================================================================


def test_cli_default_gives_no_key():
    from AG3S.trajopt import serve_safe

    for extra in ([], ["--palm-shell", "on"], shlex.split(E3_SAFE), shlex.split(E3_SAFE) + ["--palm-shell", "on"]):
        _, a = _parse(extra)
        assert a.palm_shell_margin is None
        assert serve_safe.palm_shell_margin_overrides(a) == {}
    _, a = _parse(shlex.split(E3_SAFE) + SCM)
    assert serve_safe.palm_shell_margin_overrides(a) == {"palm_shell_margin": 0.005}
    assert serve_safe.palm_shell_margin_overrides(types.SimpleNamespace()) == {}     # 옛 Namespace


@pytest.fixture
def no_cuda_check(monkeypatch):
    """This CPU test has no CUDA device: `--to-gpu-seeds curobo` (E3) would be refused by `gpu_seeds.available` before
    any SCM check. The availability probe is patched to "available"; the real probe runs at server start."""
    from AG3S.trajopt import gpu_seeds

    monkeypatch.setattr(gpu_seeds, "available", lambda: (True, ""))


def test_rejected_combinations(no_cuda_check):
    from AG3S.trajopt import serve_safe

    def check(extra, ok):
        p, a = _parse(extra)
        if ok:
            serve_safe.reject_bad_flag_combinations(p, a)
        else:
            with pytest.raises(SystemExit):
                serve_safe.reject_bad_flag_combinations(p, a)

    om = ["--obstacle-margin", "0.010"]
    check(om + ["--palm-shell-margin", "5"], False)                          # 껍질 없음
    check(om + ["--palm-shell", "off", "--palm-shell-margin", "5"], False)
    check(["--palm-shell", "on", "--palm-shell-margin", "5"], False)         # --obstacle-margin 0 (기본)
    for bad in ("-1", "nan", "inf"):
        check(om + ["--palm-shell", "on", "--palm-shell-margin", bad], False)
    check(om + ["--palm-shell", "on", "--palm-shell-margin", "5"], True)
    check(om + ["--palm-shell", "left", "--palm-shell-margin", "0"], True)
    check(om + ["--palm-shell", "on"], True)
    check(shlex.split(E3_SAFE) + SCM, True)
    # the config builder refuses on its own too (offline tools call it without the parser checks)
    for extra in (om + ["--palm-shell-margin", "5"], ["--palm-shell", "on", "--palm-shell-margin", "5"]):
        _, a = _parse(extra)
        with pytest.raises(SystemExit):
            serve_safe.trajopt_config_from_args(a)


def test_config_default_absent_from_to_dict_and_validated():
    from AG3S.trajopt.config import CollisionBackendConfig as CollisionConfig, TrajOptConfig

    base = {"backend": "esdf", "obstacle_margin": 0.01}
    c = TrajOptConfig.from_dict({"collision": base})
    assert c.collision.palm_shell_margin is None
    d = c.to_dict()
    assert "palm_shell_margin" not in d["collision"]
    # AG3S-CORE-EVAL-A6: the palm neighbour-contact keys are also left out while off (same rule, `_OMIT_WHEN_DEFAULT`).
    a6 = {"neighbour_palm_contact_depth", "neighbour_palm_contact_radius", "neighbour_palm_contact_gate"}
    assert set(d["collision"]) == {f.name for f in dataclasses.fields(CollisionConfig)} - {"palm_shell_margin"} - a6
    assert TrajOptConfig.from_dict(d) == c                                   # round trip
    on = TrajOptConfig.from_dict({"collision": {**base, "palm_shell_margin": 0.005}})
    assert on.to_dict()["collision"]["palm_shell_margin"] == 0.005
    assert c.with_overrides({"collision": {"palm_shell_margin": 0.005}}) == on
    assert on.with_overrides({"cost": {}}).collision.palm_shell_margin == 0.005
    for bad in (-0.001, float("nan"), float("inf"), True, "5"):
        with pytest.raises(ValueError):
            TrajOptConfig.from_dict({"collision": {**base, "palm_shell_margin": bad}})
    with pytest.raises(ValueError):
        TrajOptConfig.from_dict({"collision": {"backend": "esdf", "palm_shell_margin": 0.005}})   # obstacle_margin 0


def test_full_e3_command_line_config_build(no_cuda_check):
    """closed-loop 준비 확인 (서버 없이): E3 명령줄 + `--palm-shell on --palm-shell-margin 5` 가 parse · 거절 검사 ·
    `sqp_overrides` · config 를 통과하고, E3 명령줄 그대로의 config 와는 `palm_shell_margin` 하나만 다르다."""
    from AG3S.trajopt import serve_safe

    p0, a0 = _parse(shlex.split(E3_SAFE))
    p1, a1 = _parse(shlex.split(E3_SAFE) + SCM)
    serve_safe.reject_bad_flag_combinations(p0, a0)
    serve_safe.reject_bad_flag_combinations(p1, a1)
    assert serve_safe.sqp_overrides(a0) == serve_safe.sqp_overrides(a1)
    c0 = serve_safe.trajopt_config_from_args(a0)
    c1 = serve_safe.trajopt_config_from_args(a1)
    assert c0.collision.palm_shell_margin is None and c1.collision.palm_shell_margin == 0.005
    assert c1.collision.obstacle_margin == 0.01 and c1.collision.esdf_margin == 0.0
    d0, d1 = c0.to_dict(), c1.to_dict()
    assert d1["collision"].pop("palm_shell_margin") == 0.005
    assert d0 == d1
    assert serve_safe.palm_shell_options(a1) == {"hands": "both"}


def test_announce_line_only_when_on(caplog):
    from AG3S.trajopt import serve_safe
    from AG3S.trajopt.config import TrajOptConfig

    off = TrajOptConfig.from_dict({"collision": {"backend": "esdf", "obstacle_margin": 0.01}})
    on = off.with_overrides({"collision": {"palm_shell_margin": 0.005}})
    with caplog.at_level("WARNING"):
        serve_safe.announce_palm_shell_margin(off.collision)
    assert not caplog.records
    with caplog.at_level("WARNING"):
        serve_safe.announce_palm_shell_margin(on.collision)
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "palm_shell_margin: 5.0 mm" in msg and "obstacle_margin 10.0 mm" in msg


def test_client_metadata_key_only_when_on():
    from AG3S.trajopt.config import TrajOptConfig
    from AG3S.trajopt.safe_policy import SafePolicy

    def meta(cfg):
        sp = SafePolicy.__new__(SafePolicy)
        sp._policy = None
        sp.linearizer = types.SimpleNamespace(n_spheres=606)
        sp.to_config = cfg
        sp.shadow = False
        sp.ag3s = types.SimpleNamespace()
        sp.task_gate = "off"
        sp.held_3d = None
        return sp.metadata

    off = TrajOptConfig.from_dict({"collision": {"backend": "esdf", "obstacle_margin": 0.01}})
    on = off.with_overrides({"collision": {"palm_shell_margin": 0.005}})
    m0, m1 = meta(off), meta(on)
    assert "palm_shell_margin_m" not in m0
    assert m1.pop("palm_shell_margin_m") == 0.005
    assert m0 == m1


def test_ta_offline_variant():
    from AG3S.trajopt.config import TrajOptConfig
    from AG3S.trajopt.experiments import ta_offline as TA

    assert TA.variant_server_args("PSM5") == ("--palm-shell", "on", "--palm-shell-margin", "5")
    base = TrajOptConfig.from_dict({"collision": {"backend": "esdf", "obstacle_margin": 0.01}})
    assert TA.variant_config(base, "PSM5", False).collision.palm_shell_margin == 0.005
    assert TA.variant_config(base, "PS", False).collision.palm_shell_margin is None


# =============================================================================================
# 2 · 3. 기록 chunk (E4 1345 B_ROI_FPR seq 12 · 13) — 행 값 · 다시 풀기 · 기록
# =============================================================================================


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


def _solve(run, seq):
    from AG3S.trajopt.refiner import TrajOptChunkRefiner

    inp, _ = run.inputs(seq)
    holder = {"v": (inp["scene"], inp["q_now"], True)}
    ref = TrajOptChunkRefiner(run.robot, run.layout, lambda _c: holder["v"], run.base_cfg)
    ref.optimizer.linearizer.set_joint_parameter_path(
        run.finger_path(inp["summary"], inp["ref_chunk"], run.base_cfg.horizon.planned))
    c = ref.refine(inp["ref_chunk"], {"t_step": seq, "previous_physical_chunk": None})
    sha = hashlib.sha1(np.ascontiguousarray(np.asarray(c, np.float64)).tobytes()).hexdigest()[:16]
    return sha, dict(ref.last_result.metrics), inp["scene"]


def _rows(run, seq):
    from AG3S.trajopt.experiments import ta_offline as TA

    inp, _ = run.inputs(seq)
    traj = run.layout.chunk_to_trajectory(inp["rec_chunk"])[:, :16]
    lin, cen = TA.states_of(run, traj, inp["q_now"], inp["scene"],
                            run.finger_path(inp["summary"], inp["ref_chunk"], 16))
    vals = lin._esdf_clearance(cen, inp["scene"])[..., 0][:, :run.robot.n_spheres]
    cls = np.asarray(lin._last_obstacle_class)[:, :run.robot.n_spheres]
    return vals, cls, inp["scene"]


needs_e4 = pytest.mark.skipif(not _have_e4(), reason="E4 1345 기록이 없다")


@needs_e4
def test_off_resolve_is_the_sc_golden_and_records_no_key():
    pytest.importorskip("casadi")
    pytest.importorskip("osqp")
    for extra in ((), ("--palm-shell", "on")):
        run = _run(extra)
        assert run.base_cfg.collision.palm_shell_margin is None
        for seq in (12, 13):
            sha, met, scene = _solve(run, seq)
            if not extra:
                assert sha == GOLDEN[f"seq{seq}"]
            assert "palm_shell_margin_m" not in met
            assert scene.palm_shell_margin is None


@needs_e4
def test_margin_equal_to_obstacle_margin_is_bit_identical_to_shell_on():
    pytest.importorskip("casadi")
    pytest.importorskip("osqp")
    ps, eq = _run(("--palm-shell", "on")), _run(("--palm-shell", "on", "--palm-shell-margin", "10"))
    assert eq.base_cfg.collision.palm_shell_margin == ps.base_cfg.collision.obstacle_margin == 0.01
    for seq in (12, 13):
        v0, c0, _ = _rows(ps, seq)
        v1, c1, s1 = _rows(eq, seq)
        assert s1.palm_shell_margin == 0.01
        assert np.array_equal(v0, v1) and np.array_equal(c0, c1)
        assert _solve(ps, seq)[0] == _solve(eq, seq)[0]


@needs_e4
def test_five_mm_moves_only_the_shell_obstacle_rows():
    pytest.importorskip("casadi")
    pytest.importorskip("osqp")
    from AG3S.ag3s.robot_models.urdf_sphere_chain import PALM_SHELL_ROLE

    ps, m5 = _run(("--palm-shell", "on")), _run(tuple(SCM))
    shell = np.asarray([r == PALM_SHELL_ROLE for r in m5.robot.sphere_roles], bool)
    assert m5.robot.n_spheres == 606 and int(shell.sum()) == 2 * 168 and shell[-336:].all()
    n_moved = 0
    for seq in (12, 13):
        v0, c0, _ = _rows(ps, seq)
        v1, c1, s1 = _rows(m5, seq)
        assert s1.palm_shell_margin == 0.005
        assert np.array_equal(c0, c1)                                        # classes do not depend on the margin
        assert np.array_equal(v0[:, ~shell], v1[:, ~shell])                  # every other sphere: bit for bit
        moved = (c1 == 0) & shell[None, :]                                   # obstacle class on a shell row
        assert np.allclose(v1[moved] - v0[moved], 0.005, rtol=0, atol=1e-12)
        assert np.array_equal(v0[~moved], v1[~moved])
        n_moved += int(moved.sum())
    assert n_moved > 0
    sha, met, _ = _solve(m5, 13)
    assert met["palm_shell_margin_m"] == 0.005 and met["obstacle_margin_m"] == 0.01


@needs_e4
def test_worst_row_names_the_shell_margin_only_when_on():
    pytest.importorskip("casadi")
    from AG3S.trajopt.experiments import ta_offline as TA

    for extra, on in ((("--palm-shell", "on"), False), (tuple(SCM), True)):
        run = _run(extra)
        inp, _ = run.inputs(13)
        traj = run.layout.chunk_to_trajectory(inp["rec_chunk"])[:, :16]
        lin, _ = TA.states_of(run, traj, inp["q_now"], inp["scene"],
                              run.finger_path(inp["summary"], inp["ref_chunk"], 16))
        _, pair = lin.worst_row(traj, inp["q_now"], inp["scene"])
        assert pair is not None and "obstacle_margin_class" in pair
        if on:
            assert "palm_shell_row" in pair
            assert ("palm_shell_margin_m" in pair) == pair["palm_shell_row"]
        else:
            assert "palm_shell_row" not in pair and "palm_shell_margin_m" not in pair


@needs_e4
def test_linearizer_refuses_a_model_without_shell_spheres():
    pytest.importorskip("casadi")
    from AG3S.trajopt.experiments import ta_offline as TA

    run = _run(())
    inp, _ = run.inputs(13)
    scene = dataclasses.replace(inp["scene"], palm_shell_margin=0.005)
    traj = run.layout.chunk_to_trajectory(inp["rec_chunk"])[:, :16]
    lin, cen = TA.states_of(run, traj, inp["q_now"], scene,
                            run.finger_path(inp["summary"], inp["ref_chunk"], 16))
    with pytest.raises(ValueError, match="no palm_shell sphere"):
        lin._esdf_clearance(cen, scene)
