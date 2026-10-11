#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVENT="${1:-UserPromptSubmit}"
STORAGE="${AG3S_STORAGE_ROOT:-$(dirname "$ROOT")/AG3S_VLA-storage}"
BRANCH="$(git -C "$ROOT" branch --show-current)"
HEAD_SHA="$(git -C "$ROOT" rev-parse HEAD)"
case "$BRANCH" in
  main) PURPOSE="V11AB 기준. 구조·경로 변경만 허용; 최신 AG3S 개선을 무심코 merge하지 않는다." ;;
  feature/panda-benchmark) PURPOSE="paired v2 기반 Panda benchmark 개발. 개선 branch 전체 merge로 평가 기준을 바꾸지 않는다." ;;
  feature/ag3s-development) PURPOSE="v2 이후 AG3S 개선과 migration checkpoint WIP 계속 개발." ;;
  *) PURPOSE="새 branch. doc/BRANCHES.md에서 기준과 목적을 확인한다." ;;
esac
{
  printf 'AG3S_VLA checkout: %s\nbranch: %s\nHEAD: %s\n목적: %s\n' "$ROOT" "$BRANCH" "$HEAD_SHA" "$PURPOSE"
  printf 'AG3S/ag3s·AG3S/trajopt / Baseline / benchmark / integration / robots / third_party가 현재 구조다.\n'
  printf 'CLAUDE.md, .claude/project-context.md, doc/BRANCHES.md와 migration manifest를 읽는다. doc/SHARED_STORAGE_AND_PYTHON.md를 읽는다. Input은 AG3S_STORAGE_ROOT가 선택한 별도 AG3S_VLA-storage에 있으며 이전 checkout에 의존하지 않는다.\n'
  printf 'Shared input storage: %s\nPrimary development interpreter: %s/.venv-openpi-live/bin/python\nCPU regression interpreter: %s/.venv-ag3s/bin/python\ncuRobo producer interpreter: %s/.venv-curobo/bin/python\nNew result runs: %s/results/development/<KST-date_commit_UUID>/\n' "$STORAGE" "$ROOT" "$ROOT" "$ROOT" "$STORAGE"
  cat "$ROOT/.claude/ag3s-rules-brief.md"
} | jq -Rs --arg event "$EVENT" '{suppressOutput:true,hookSpecificOutput:{hookEventName:$event,additionalContext:.}}'
