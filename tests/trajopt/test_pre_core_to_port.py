"""AG3S-PRE-CORE-TO-PORT — main checkout 에 옮긴 세 기능이 한 명령줄에서 함께 parse · 검사 · 배선되는지.

보고: `AG3S/ag3s/docs/handoff/AG3S-PRE-CORE-TO-PORT.impl.md`.

옮긴 것: E3-b 1 차 검사 (`--to-gpu-seeds-check first-pass`, `snapshot_e3c`) · 손바닥 껍질 구 (`--palm-shell`,
`--palm-shell-overflow`) · 껍질 행 여유 (`--palm-shell-margin`) (`snapshot_scpt`). 이미 checkout 에 있던 것: 꼭지 덮개
(`--target-stem-cap` …, `AG3S-PRE-CORE-TO-STEMCAP.impl.md`).

이 파일이 고정하는 것:

1. **함께 parse** — E3 closed-loop 명령줄 (E2 16 seed · GPU seed · 1 차 검사) + 꼭지 덮개 + 껍질 구 + 껍질 여유가 parse 되고
   `reject_bad_flag_combinations` 를 통과하며, 각 flag 가 자기 자리 (`sqp_overrides` · `target_exclusion_section` ·
   `palm_shell_options` · `collision.palm_shell_margin` · `build_ag3s` 인자) 로만 간다 — 서로의 값을 바꾸지 않는다.
2. **꺼짐 = 키 없음** — 새 flag 를 안 주면 어느 자리에도 키가 없다 (세 기능 모두).
3. **`--help`** 가 세 기능의 flag 를 모두 보여 준다.
"""

from __future__ import annotations

import shlex

import pytest

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]
#: E3 closed-loop 의 SAFE flag 전부 (`tests/trajopt/test_scm_palm_shell_margin.py` 의 `E3_SAFE` 와 같은 문자열).
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
#: E3 명령줄에서 1 차 검사만 뺀 것 (`--to-gpu-seeds-check` 기본 all).
E3_ALL = E3_SAFE.replace(" --to-gpu-seeds-check first-pass", "")
STEM_CAP = ["--target-stem-cap", "in_ball", "--stem-cap-radius-mm", "20", "--stem-cap-top-mm", "25",
            "--stem-cap-bottom-fit-frac", "0.5"]
SHELL = ["--palm-shell", "on", "--palm-shell-overflow", "5", "--palm-shell-margin", "5"]


def _parse(extra):
    from AG3S.trajopt import serve_safe

    p = serve_safe.build_parser()
    return p, p.parse_args(BASE_ARGV + list(extra))


@pytest.fixture
def no_cuda_check(monkeypatch):
    """CPU test: `--to-gpu-seeds curobo` 는 `gpu_seeds.available` 이 CUDA 를 못 찾으면 거절된다. 그 probe 만 "있음" 으로
    바꾼다 (실제 probe 는 서버 시작 때 돈다)."""
    from AG3S.trajopt import gpu_seeds

    monkeypatch.setattr(gpu_seeds, "available", lambda: (True, ""))


def _build_kwargs(monkeypatch, args) -> dict:
    """`ag3s_from_args` 가 `build_ag3s` 에 넘기는 인자 (모델 · 거리장은 짓지 않는다)."""
    from AG3S.trajopt import serve_safe

    seen = {}
    monkeypatch.setattr(serve_safe, "build_ag3s", lambda *a, **kw: seen.update(kw) or seen)
    serve_safe.ag3s_from_args(args, plan_horizon_steps=16, rows_per_step=1)
    return seen


def test_e3_first_pass_stem_cap_and_palm_shell_parse_together(no_cuda_check, monkeypatch):
    from AG3S.trajopt import serve_safe

    p, a = _parse(shlex.split(E3_SAFE) + STEM_CAP + SHELL)
    serve_safe.reject_bad_flag_combinations(p, a)                   # 거절 없음
    # E3 (E2 seed 16 · cuRobo GPU seed · 1 차 검사) → sqp
    sqp = serve_safe.sqp_overrides(a)
    assert (sqp["multi_seed"], sqp["gpu_seeds"], sqp["gpu_seeds_check"]) == (16, "curobo", "first-pass")
    # 꼭지 덮개 → AG3S target_exclusion
    sec = serve_safe.target_exclusion_section(a)
    assert sec["target_exclusion"] == "ball_owned"
    assert sec["target_stem_cap"] == "in_ball"
    assert sec["stem_cap_radius"] == pytest.approx(0.020) and sec["stem_cap_top"] == pytest.approx(0.025)
    assert sec["stem_cap_bottom_fit_frac"] == 0.5
    # 껍질 구 → 제약 로봇 모델, 껍질 여유 → collision
    assert serve_safe.palm_shell_options(a) == {"hands": "both", "overflow": pytest.approx(0.005)}
    assert serve_safe.palm_shell_margin_overrides(a) == {"palm_shell_margin": 0.005}
    cfg = serve_safe.trajopt_config_from_args(a)
    assert cfg.sqp.gpu_seeds == "curobo" and cfg.sqp.gpu_seeds_check == "first-pass" and cfg.sqp.multi_seed == 16
    assert cfg.collision.palm_shell_margin == 0.005 and cfg.collision.obstacle_margin == 0.01
    # build_ag3s 가 셋을 모두 받는다
    kw = _build_kwargs(monkeypatch, a)
    assert kw["palm_shell"] == {"hands": "both", "overflow": pytest.approx(0.005)}
    assert kw["target_exclusion"]["target_stem_cap"] == "in_ball"


def test_each_feature_moves_only_its_own_keys(no_cuda_check, monkeypatch):
    """세 기능을 하나씩 더해도 다른 기능의 자리 (sqp · target_exclusion · palm_shell · collision) 는 그대로다."""
    from AG3S.trajopt import serve_safe

    def view(extra):
        p, a = _parse(shlex.split(E3_ALL) + list(extra))
        serve_safe.reject_bad_flag_combinations(p, a)
        cfg = serve_safe.trajopt_config_from_args(a).to_dict()
        kw = _build_kwargs(monkeypatch, a)
        return {"sqp": serve_safe.sqp_overrides(a), "cfg_sqp": cfg["sqp"], "collision": cfg["collision"],
                "target_exclusion": dict(kw["target_exclusion"]), "palm_shell": kw.get("palm_shell", "<no key>")}

    base = view([])
    assert base["palm_shell"] == "<no key>"
    assert "gpu_seeds_check" not in base["sqp"] and base["cfg_sqp"]["gpu_seeds_check"] == "all"
    assert "palm_shell_margin" not in base["collision"]
    assert "target_stem_cap" not in base["target_exclusion"]
    # off 값을 명시해도 키 없음 (세 기능 모두)
    assert view(["--to-gpu-seeds-check", "all", "--palm-shell", "off", "--target-stem-cap", "off"]) == base

    fp = view(["--to-gpu-seeds-check", "first-pass"])
    assert fp["sqp"] == {**base["sqp"], "gpu_seeds_check": "first-pass"}
    assert {k: v for k, v in fp.items() if k not in ("sqp", "cfg_sqp")} == \
        {k: v for k, v in base.items() if k not in ("sqp", "cfg_sqp")}

    cap = view(STEM_CAP)
    assert {k: v for k, v in cap.items() if k != "target_exclusion"} == \
        {k: v for k, v in base.items() if k != "target_exclusion"}
    assert {k: v for k, v in cap["target_exclusion"].items() if not k.startswith(("target_stem_cap", "stem_cap"))} \
        == base["target_exclusion"]

    shell = view(SHELL)
    assert {k: v for k, v in shell.items() if k not in ("collision", "palm_shell")} == \
        {k: v for k, v in base.items() if k not in ("collision", "palm_shell")}
    assert {k: v for k, v in shell["collision"].items() if k != "palm_shell_margin"} == base["collision"]
    assert shell["collision"]["palm_shell_margin"] == 0.005


def test_first_pass_and_shell_margin_still_refused_alone(no_cuda_check):
    """같이 쓸 수 있게 옮긴 뒤에도 각 기능의 거절 규칙은 그대로다."""
    from AG3S.trajopt import serve_safe

    no_gpu = E3_ALL.replace(" --to-gpu-seeds curobo", "")
    for extra in (shlex.split(no_gpu) + ["--to-gpu-seeds-check", "first-pass"],       # GPU seed 없이 1 차 검사
                  shlex.split(E3_SAFE) + ["--palm-shell-margin", "5"],                 # 껍질 없이 여유
                  shlex.split(E3_SAFE) + ["--stem-cap-radius-mm", "20"],               # 덮개 없이 덮개 값
                  shlex.split(E3_SAFE) + ["--palm-shell", "on", "--palm-cover", "on"]):  # 껍질 + palm cover
        p, a = _parse(extra)
        with pytest.raises(SystemExit):
            serve_safe.reject_bad_flag_combinations(p, a)


def test_help_lists_every_ported_flag():
    from AG3S.trajopt import serve_safe

    text = serve_safe.build_parser().format_help()
    for flag in ("--to-multi-seed", "--to-gpu-seeds", "--to-gpu-seeds-check", "--palm-shell", "--palm-shell-overflow",
                 "--palm-shell-margin", "--target-stem-cap", "--stem-cap-radius-mm", "--stem-cap-top-mm",
                 "--stem-cap-bottom-fit-frac"):
        assert f"{flag} " in text or f"{flag}\n" in text or f"{flag}=" in text or f"{flag}," in text, flag
