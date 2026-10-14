"""T46f A14c — T46e oracle extent (`--target-extent oracle-visible`) 와 T46f A14 (`--links gripper_wrist` ·
`--target-field-policy none`) 의 조합 규칙.

| `--target-field-policy` | `--links gripper` | `--links gripper_wrist` | 이유 |
|---|---|---|---|
| `relax` (기본) | 거절 (T46e 그대로) | 거절 | target-free 계층이 없다 — oracle 이 바꿀 것이 없다 |
| `none` (A14) | **거절** (A14c) | **거절** (A14c) | `relax` 와 같다: 예외 없음은 target-free 계층을 만들지 않는다 |
| `exclude-authorized` | 허용 | **허용** | T46f 단계 4 의 ③ |
| `exclude-all` | 허용 | **허용** | T46f 단계 4 의 ② |

oracle 이 꺼져 있으면 (`--target-extent` 없음 · `estimated`) 위 네 정책 × 두 `--links` 는 A14 의 규칙 그대로다.
`--links` 는 제약 모델 (형상) 만, `--target-extent` 는 거리장 (미세 target-free 계층) 만 바꾼다 — 서로 막지 않는다.
"""

from __future__ import annotations

import pytest

from AG3S.ag3s.config import AG3SConfig

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]
CUROBO = ["--esdf-backend", "curobo", "--fine-voxel", "0.005"]
ORACLE = ["--target-extent", "oracle-visible"]


def _check(*argv):
    from AG3S.trajopt import serve_safe

    p = serve_safe.build_parser()
    args = p.parse_args(BASE_ARGV + list(argv))
    serve_safe.reject_bad_flag_combinations(p, args)
    return args


def _rejected(capsys, *argv) -> str:
    with pytest.raises(SystemExit) as e:
        _check(*argv)
    assert e.value.code == 2                      # argparse `ap.error`
    return capsys.readouterr().err


@pytest.mark.parametrize("links", ["gripper", "gripper_wrist"])
@pytest.mark.parametrize("policy", ["relax", "none"])
def test_oracle_needs_a_target_free_policy(capsys, links, policy):
    err = _rejected(capsys, *CUROBO, *ORACLE, "--links", links, "--target-field-policy", policy)
    assert "--target-extent oracle-visible" in err and "--target-field-policy" in err
    assert policy in err                          # the message names what it got


@pytest.mark.parametrize("links", ["gripper", "gripper_wrist"])
@pytest.mark.parametrize("policy", ["exclude-authorized", "exclude-all"])
@pytest.mark.parametrize("exclusion", ["ball", "ball-owned"])
def test_oracle_with_gripper_wrist_and_target_free_policies(links, policy, exclusion):
    """T46f 단계 4: oracle × {② exclude_all · ③ exclude_authorized} × gripper_wrist 가 뜬다."""
    from AG3S.trajopt import serve_safe

    args = _check(*CUROBO, *ORACLE, "--links", links, "--target-field-policy", policy,
                  "--target-exclusion", exclusion)
    # both features reach the server: the wrist cover (constraint model) and the oracle (esdf section)
    if links == "gripper_wrist":
        assert serve_safe.wrist_cover_options(args) == {"hands": "both"}
        assert serve_safe.constraint_link_filter(args.links) is not None
    else:
        assert serve_safe.wrist_cover_options(args) is None
    esdf = serve_safe.target_exclusion_section(args)
    assert esdf["target_extent"] == "oracle_visible"
    resolved = serve_safe.resolve_target_field_policy(args.target_field_policy)
    cfg = AG3SConfig.from_dict({"esdf": esdf, "constraint": {"target_field_policy": resolved}})
    assert cfg.esdf.target_extent == "oracle_visible"
    assert cfg.constraint.excludes_target_from_field is True


@pytest.mark.parametrize("links", ["gripper", "gripper_wrist"])
def test_oracle_off_keeps_the_a14_rules(capsys, links):
    """oracle 이 없으면 네 정책은 A14 그대로: relax · none 은 curobo 없이도 뜨고, exclude-* 는 curobo 를 요구한다."""
    from AG3S.trajopt import serve_safe

    for policy in ("relax", "none"):
        args = _check("--links", links, "--target-field-policy", policy)
        assert "target_extent" not in serve_safe.target_exclusion_section(args)
    for policy in ("exclude-authorized", "exclude-all"):
        _check(*CUROBO, "--links", links, "--target-field-policy", policy)
        _rejected(capsys, "--links", links, "--target-field-policy", policy)        # legacy backend
    # `--target-extent estimated` (explicit default) is not the oracle: no extra rule with none
    args = _check("--links", links, "--target-field-policy", "none", "--target-extent", "estimated")
    assert "target_extent" not in serve_safe.target_exclusion_section(args)


def test_oracle_side_flags_still_need_the_oracle_with_none(capsys):
    """T46e: `--oracle-*` 는 oracle 에서만 — none 이 그 규칙을 바꾸지 않는다."""
    for flag in (["--oracle-extent-identity", "gt"], ["--oracle-max-dist-voxels", "4"]):
        err = _rejected(capsys, "--links", "gripper_wrist", "--target-field-policy", "none", *flag)
        assert "oracle-visible" in err


def test_main_rejects_oracle_with_none_before_loading(monkeypatch, capsys):
    """`serve_safe.main` 이 체크포인트를 올리기 전에 죽는다 (T46e 의 `_main_errors` 와 같은 경로)."""
    import sys

    from AG3S.trajopt import serve_safe

    monkeypatch.setattr(sys, "argv", ["serve_safe", "--config", "c", "--checkpoint", "k", "--model-xml",
                                      "x.xml", *CUROBO, *ORACLE, "--links", "gripper_wrist",
                                      "--target-field-policy", "none"])
    with pytest.raises(SystemExit) as e:
        serve_safe.main()
    assert e.value.code == 2
    assert "--target-field-policy" in capsys.readouterr().err
