"""**기록하지 않을 때는 아무 비용도 들지 않는다** — 호출부에서 그것을 고정한다.

MuJoCo 참값을 매 frame 뽑는 비용을 기록 안 하는 실행까지 물리면 안 된다 (T6a 의 요구).
그런데 그 성질은 `pi05_infer.py` 의 **배치**에 달려 있고, 그 파일은 1700 줄이며 정책 서버와
MuJoCo 모델이 있어야 돌아서 실행으로는 검사할 수 없다. 그래서 **소스를 읽어서** 검사한다:
참값 관련 이름의 모든 사용이 기록 분기 안에 있는가.

소스 검사가 실행 검사보다 약하지만, 여기서 재려는 것이 *"값이 맞나"* 가 아니라 *"이 코드가
언제 도나"* 이므로 배치가 바로 그 답이다. 그리고 이 검사는 새 호출부가 조용히 추가되는 것을
막는다 — 조용히 추가되는 것이 정확히 이 요구가 막으려는 일이다.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

TARGET = pathlib.Path("/mnt/dev/work/pi05_TO_hybrid/rby1_bringup/pi05_infer.py")

#: 기록할 때만 돌아야 하는 이름들. `scene_truth` 는 probe, 나머지 둘은 그 모듈의 API 다.
RECORD_ONLY_NAMES = {"scene_truth", "ObjectPoseProbe", "joint_positions"}

#: 분기 조건이 기록 여부를 묻고 있다고 인정하는 표현. 둘 다 `--record-frames` 하나에서 나온다
#: (`frame_recorder` 는 그 플래그가 있을 때만 만들어진다).
GUARD_SOURCES = ("args.record_frames", "frame_recorder is not None")


@pytest.fixture(scope="module")
def tree() -> ast.AST:
    return ast.parse(TARGET.read_text())


@pytest.fixture(scope="module")
def parents(tree) -> dict:
    out = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            out[child] = node
    return out


def _ancestor_guards(node, parents) -> list[str]:
    """이 노드를 감싸는 `if` 들의 조건을 소스로. 몸통에 있을 때만 센다 (`else` 는 아니다)."""
    out = []
    child = node
    while child in parents:
        parent = parents[child]
        if isinstance(parent, ast.If) and any(child is s for s in parent.body):
            out.append(ast.unparse(parent.test))
        child = parent
    return out


def _is_none_sentinel(node, parents) -> bool:
    """`scene_truth = None` 처럼 **아무 일도 하지 않는** 초기화인가."""
    parent = parents.get(node)
    return (isinstance(node.ctx, ast.Store) and isinstance(parent, ast.Assign)
            and isinstance(parent.value, ast.Constant) and parent.value.value is None)


def test_every_scene_truth_use_sits_behind_the_record_flag(tree, parents):
    """참값 코드의 모든 사용이 기록 분기 안에 있다."""
    unguarded = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or node.id not in RECORD_ONLY_NAMES:
            continue
        if _is_none_sentinel(node, parents):
            continue
        guards = _ancestor_guards(node, parents)
        if not any(g in GUARD_SOURCES for g in guards):
            unguarded.append((node.id, node.lineno, guards))
    assert unguarded == [], (
        "기록하지 않는 실행에서도 도는 참값 코드가 있습니다 (이름, 줄, 감싸는 조건): "
        f"{unguarded}. MuJoCo 참값은 --record-frames 안쪽에서만 계산해야 합니다")


def test_scene_truth_is_never_imported_at_module_level(tree):
    """모듈 최상위 import 가 없다 — 기록하지 않는 실행은 이 모듈을 **읽지도** 않는다."""
    top_level = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [getattr(node, "module", "") or ""] + [a.name for a in node.names]
            if any("scene_truth" in n for n in names):
                top_level.append(node.lineno)
    assert top_level == []


def test_the_import_is_inside_the_record_branch(tree, parents):
    """그 대신 `if args.record_frames:` 안에 정확히 한 번 있다."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and "scene_truth" in (node.module or ""):
            found.append(_ancestor_guards(node, parents))
    assert len(found) == 1, f"scene_truth import 가 {len(found)} 군데 있습니다"
    assert "args.record_frames" in found[0]


def test_probe_is_built_once_not_per_frame(tree):
    """`ObjectPoseProbe(...)` 호출이 딱 하나다.

    프레임마다 만들면 `mj_name2id` 를 매번 물어 "이 프레임에 없었다" 와 "이 모델에 없다" 가
    구별되지 않고, 비용도 프레임마다 든다.
    """
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "ObjectPoseProbe"]
    assert len(calls) == 1
