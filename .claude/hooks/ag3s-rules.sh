#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVENT="${1:-UserPromptSubmit}"
STORAGE="${AG3S_STORAGE_ROOT:-$(dirname "$ROOT")/AG3S_VLA-storage}"
BRANCH="$(git -C "$ROOT" branch --show-current)"
HEAD_SHA="$(git -C "$ROOT" rev-parse HEAD)"
case "$BRANCH" in
  main) PURPOSE="2026-10-29부터 AG3S 최종 구성 기준 (V11AB는 tag base/v11ab-monorepo). doc/active/BRANCHES.md \"main 갱신\"." ;;
  feature/panda-benchmark) PURPOSE="Panda benchmark 평가. 2026-10-29 main(AG3S 최종 구성) merge 후에도 V11AB·V11AB-pb1~pb7은 merge 전 서버 그대로이고 최종 구성 flag는 기본 꺼짐; Panda 이식은 사용자 판정 후." ;;
  feature/ag3s-development) PURPOSE="v2 이후 AG3S 개선과 migration checkpoint WIP 계속 개발." ;;
  *) PURPOSE="새 branch. doc/active/BRANCHES.md에서 기준과 목적을 확인한다." ;;
esac
case "$BRANCH" in feature/ag3s-development) RESULTS_KIND=development ;; *) RESULTS_KIND=benchmark ;; esac
{
  printf 'AG3S_VLA checkout: %s\nbranch: %s\nHEAD: %s\n목적: %s\n' "$ROOT" "$BRANCH" "$HEAD_SHA" "$PURPOSE"
  printf 'AG3S/ag3s·AG3S/trajopt / Baseline / benchmark / integration / robots / third_party가 현재 구조다.\n'
  printf 'CLAUDE.md, .claude/project-context.md, doc/active/BRANCHES.md와 migration manifest를 읽는다. doc/active/SHARED_STORAGE_AND_PYTHON.md를 읽는다. Input은 AG3S_STORAGE_ROOT가 선택한 별도 AG3S_VLA-storage에 있으며 이전 checkout에 의존하지 않는다.\n'
  printf 'Shared input storage: %s\nPrimary development interpreter: %s/.venv-openpi-live/bin/python\nCPU regression interpreter: %s/.venv-ag3s/bin/python\ncuRobo producer interpreter: %s/.venv-curobo/bin/python\nNew result runs: %s/results/%s/<KST-date_commit_UUID>/\n' "$STORAGE" "$ROOT" "$ROOT" "$ROOT" "$STORAGE" "$RESULTS_KIND"
  cat "$ROOT/.claude/ag3s-rules-brief.md"
} | jq -Rs --arg event "$EVENT" '{suppressOutput:true,hookSpecificOutput:{hookEventName:$event,additionalContext:.}}'
