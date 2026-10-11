#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVENT="${1:-UserPromptSubmit}"
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
  printf 'CLAUDE.md, .claude/project-context.md, doc/BRANCHES.md와 migration manifest를 읽는다. 기존 /mnt/dev/work는 legacy input storage이며 코드 편집 대상이 아니다.\n'
  cat "$ROOT/.claude/ag3s-rules-brief.md"
} | jq -Rs --arg event "$EVENT" '{suppressOutput:true,hookSpecificOutput:{hookEventName:$event,additionalContext:.}}'
