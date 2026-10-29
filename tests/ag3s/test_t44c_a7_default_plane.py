"""T44c A7 — coarse 를 켜면 `sign_floor` 기본이 `"plane"` (사용자 판정 2026-10-10).

A6 는 `esdf.target_free_coarse_sign_floor` 를 `"guard"` (A5 그대로) 기본으로 두었다. A5 (guard) 는 target 바닥
내부를 음수로 남겨 사용자 조건 6 을 어긴다 (`T44c.verify.json`) — 쓸 이유가 비교뿐이다. A7 은 기본을
`"plane"` 으로 바꾸고 `"guard"` 를 선택지로 남긴다. `esdf.target_free_coarse` 자체의 기본은 off 그대로다.

## 무엇을 박나

| # | 성질 |
|---|---|
| 1 | config 기본 `"plane"` · `target_free_coarse` 기본 off · `"guard"` 선택지 · 왕복 · validation |
| 2 | `--target-free-coarse` 만 주면 plane — CLI 두 section 을 serve_safe 처럼 합쳐 config 로 |
| 3 | builder 의 spec 이 config 기본 plane 을 싣는다 (`sign_z_min_m` 이 params 에) · `CoarseFreeSpec` 기본도 plane |
| 4 | off bit-identical: sign_floor 는 `_coarse_free_spec` 에서만 읽히고, 그것은 `target_free_coarse` 가 켜진 갈래 안에서만 불린다 (AST) |
| 5 | (GPU) 실제 cuRobo: coarse off 에서 floor plane / guard 의 필드가 `torch.equal` · coarse on 기본 = 명시 plane |
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError

torch = pytest.importorskip("torch")

from AG3S.ag3s.fields import curobo_builder as CB  # noqa: E402
from AG3S.ag3s.fields.curobo_builder import (  # noqa: E402
    COARSE_SIGN_FLOORS,
    CoarseFreeSpec,
    CuroboFieldBuilder,
    TargetBall,
    coarse_exclusion_params,
)

VC, VF = 0.02, 0.005
TABLE = 0.861


def _ball():
    return TargetBall(centre=np.array([0.52, 0.50, TABLE + 0.045]), radius=0.05, z_min=TABLE + 0.005,
                      provenance={"support_plane_z_m": TABLE, "pad_mm": 5.0})


def _builder(**esdf):
    cfg = AG3SConfig.from_dict({"esdf": {"backend": "curobo", "voxel_size": VC,
                                         "fine_voxel_size": VF, **esdf}}).esdf
    return CuroboFieldBuilder(cfg)


# ---------------------------------------------------------------------------- 1 config


def test_default_is_plane_and_coarse_stays_off():
    e = AG3SConfig.from_dict({}).esdf
    assert e.target_free_coarse is False
    assert e.target_free_coarse_sign_floor == "plane"
    on = AG3SConfig.from_dict({"esdf": {"target_free_coarse": True}}).esdf
    assert on.target_free_coarse is True and on.target_free_coarse_sign_floor == "plane"
    assert e.TARGET_FREE_COARSE_SIGN_FLOORS == COARSE_SIGN_FLOORS == ("guard", "plane")


@pytest.mark.parametrize("floor", ["plane", "guard"])
def test_round_trip_keeps_both_choices(floor):
    cfg = AG3SConfig.from_dict({"esdf": {"target_free_coarse": True,
                                         "target_free_coarse_sign_floor": floor}})
    d = cfg.to_dict()
    assert d["esdf"]["target_free_coarse_sign_floor"] == floor
    assert AG3SConfig.from_dict(d).esdf.target_free_coarse_sign_floor == floor
    assert AG3SConfig.from_dict(AG3SConfig.from_dict({}).to_dict()).esdf \
        .target_free_coarse_sign_floor == "plane"


def test_unknown_floor_is_rejected():
    with pytest.raises(AG3SConfigError):
        AG3SConfig.from_dict({"esdf": {"target_free_coarse_sign_floor": "none"}})
    with pytest.raises(ValueError):
        CoarseFreeSpec(sign_floor="none")


# ---------------------------------------------------------------------------- 2 CLI


def _parse(*argv):
    from AG3S.trajopt.serve_safe import build_parser

    return build_parser().parse_args(["--config", "c", "--checkpoint", "k", "--model-xml", "x.xml",
                                      *argv])


def _esdf_from_cli(*argv):
    """serve_safe 가 `build_ag3s(target_exclusion=...)` 에 넘기는 두 section 을 그대로 합쳐 config 로."""
    from AG3S.trajopt.serve_safe import target_free_coarse_section, target_free_coarse_sign_section

    a = _parse(*argv)
    return AG3SConfig.from_dict({"esdf": {**target_free_coarse_section(a),
                                          **target_free_coarse_sign_section(a)}}).esdf


def test_cli_target_free_coarse_alone_gives_plane():
    from AG3S.trajopt.serve_safe import target_free_coarse_sign_section

    assert _parse().target_free_coarse_sign_floor is None
    assert target_free_coarse_sign_section(_parse("--target-free-coarse")) == {}
    e = _esdf_from_cli("--target-free-coarse")
    assert e.target_free_coarse is True and e.target_free_coarse_sign_floor == "plane"
    g = _esdf_from_cli("--target-free-coarse", "--target-free-coarse-sign-floor", "guard")
    assert g.target_free_coarse is True and g.target_free_coarse_sign_floor == "guard"
    off = _esdf_from_cli()
    assert off.target_free_coarse is False


def test_cli_help_names_plane_as_the_default():
    from AG3S.trajopt.serve_safe import build_parser

    act = next(a for a in build_parser()._actions if "--target-free-coarse-sign-floor" in a.option_strings)
    assert act.default is None and set(act.choices) == {"guard", "plane"}
    assert "= plane" in act.help


# ---------------------------------------------------------------------------- 3 builder spec


def test_builder_spec_defaults_to_plane_with_coarse_on():
    b = _builder(target_free_coarse=True)
    s = b._coarse_free_spec(_ball(), None, np.zeros((3, 3)), target_points=np.ones((4, 3)))
    assert s.sign_floor == "plane"
    assert "sign_z_min_m" in coarse_exclusion_params(_ball(), VC, s)
    assert CoarseFreeSpec().sign_floor == "plane"
    g = _builder(target_free_coarse=True, target_free_coarse_sign_floor="guard")
    sg = g._coarse_free_spec(_ball(), None, np.zeros((3, 3)))
    assert sg.sign_floor == "guard" and "sign_z_min_m" not in coarse_exclusion_params(_ball(), VC, sg)


# ---------------------------------------------------------------------------- 4 off bit-identical (구조)


def _src_tree(fn):
    return ast.parse(textwrap.dedent(inspect.getsource(fn)))


def test_sign_floor_is_read_only_by_coarse_free_spec():
    """builder 모듈에서 `target_free_coarse_sign_floor` 를 읽는 함수는 `_coarse_free_spec` 하나 (기록용 log 줄 제외)."""
    tree = ast.parse(inspect.getsource(CB))
    readers = set()
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            for node in ast.walk(fn):
                if isinstance(node, ast.Constant) and node.value == "target_free_coarse_sign_floor":
                    readers.add(fn.name)
    assert readers <= {"_coarse_free_spec", "update"}
    # update 안에서는 coarse-on 갈래의 log 줄에서만 (값에 닿지 않는다) — 아래 테스트가 갈래를 본다.


def test_coarse_free_spec_and_the_floor_live_only_under_the_coarse_on_branch():
    """`update` 에서 `_coarse_free_spec` 호출과 sign_floor 읽기는 전부 `if ... target_free_coarse ...` 안이다
    → coarse off 에서는 이 값이 어디에도 닿지 않는다 (비트 동일)."""
    tree = _src_tree(CuroboFieldBuilder.update)

    def is_coarse_on_test(node):
        return isinstance(node, ast.If) and any(
            isinstance(n, ast.Constant) and n.value == "target_free_coarse"
            for n in ast.walk(node.test))

    guarded = set()
    for node in ast.walk(tree):
        if is_coarse_on_test(node):
            for inner in ast.walk(node):
                guarded.add(id(inner))
    hits = [n for n in ast.walk(tree)
            if (isinstance(n, ast.Attribute) and n.attr == "_coarse_free_spec")
            or (isinstance(n, ast.Constant) and n.value == "target_free_coarse_sign_floor")]
    assert hits, "update 가 _coarse_free_spec 를 부르지 않는다 — 테스트를 고칠 것"
    assert all(id(n) in guarded for n in hits)


def test_pipeline_and_field_never_read_the_floor():
    import AG3S.ag3s.fields.curobo_field as F
    import AG3S.ag3s.runtime.pipeline as P
    import AG3S.trajopt.linearize as L

    for mod in (P, F, L):
        assert "target_free_coarse_sign_floor" not in inspect.getsource(mod), mod.__name__


# ============================================================================ 5 GPU (real cuRobo; skipped on CPU)


def _real_fields(overs):
    G = pytest.importorskip("tests.o4.test_gpu_esdf_field")
    from AG3S.ag3s.fields.curobo_builder import FineWindowRequest

    cams = G._scene_cameras()
    apple = np.random.default_rng(0).normal([0.51, -0.03, 0.85], 0.012, (300, 3))
    other = np.random.default_rng(3).normal([0.51, 0.03, 0.85], 0.012, (300, 3))
    ball = TargetBall(centre=[0.51, -0.03, 0.85], radius=0.045, z_min=0.825,
                      provenance={"support_plane_z_m": 0.82, "pad_mm": 5.0})
    kw = dict(target_points=apple, target_free_points=apple, target_free_ball=ball,
              fine_window=FineWindowRequest(manipulated_points=apple, placement="target_centroid"),
              nontarget_points=other)
    out = {}
    for name, over in overs.items():
        b = CuroboFieldBuilder(G._config(**over))
        for _ in range(2):
            out[name] = b.update(cams, **kw)
    return out


def _same_field(a, b):
    for la, lb in zip(a.layers, b.layers):
        assert torch.equal(la.values_device, lb.values_device)
    assert len(a.layers) == len(b.layers)
    for la, lb in zip(a.target_free_layers, b.target_free_layers):
        assert torch.equal(la.values_device, lb.values_device)
    assert len(a.target_free_layers) == len(b.target_free_layers)


def test_curobo_off_is_bit_identical_and_on_default_is_plane():
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    pytest.importorskip("curobo")
    f = _real_fields({"off": {},
                      "off_guard": {"target_free_coarse_sign_floor": "guard"},
                      "on": {"target_free_coarse": True},
                      "on_plane": {"target_free_coarse": True,
                                   "target_free_coarse_sign_floor": "plane"}})
    # off: floor 값과 무관 — 같은 필드, 거친 target-free 계층 없음.
    _same_field(f["off"], f["off_guard"])
    assert f["off"].target_free_coarse_layer is None and f["off_guard"].target_free_coarse_layer is None
    assert f["off"].stats["target_free"]["coarse"] == f["off_guard"].stats["target_free"]["coarse"] \
        == {"mode": "off"}
    pts = np.random.default_rng(5).normal([0.51, 0.0, 0.86], 0.04, (2000, 3))
    assert np.array_equal(f["off"].target_free_distance(pts), f["off_guard"].target_free_distance(pts))
    # on: `--target-free-coarse` 만 = 명시 plane.
    _same_field(f["on"], f["on_plane"])
    assert torch.equal(f["on"].target_free_coarse_layer.values_device,
                       f["on_plane"].target_free_coarse_layer.values_device)
    assert "plane_band" in f["on"].stats["target_free"]["coarse"]
    assert np.array_equal(f["on"].target_free_distance(pts), f["on_plane"].target_free_distance(pts))
