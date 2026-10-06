#!/usr/bin/env bash
# UserPromptSubmit hook — AG3S 검토 규칙을 매 턴 모델 컨텍스트에 주입한다.
# 규칙 본문은 옆의 ag3s-rules.md 이고, 그 파일만 고치면 된다.
set -euo pipefail
RULES="/mnt/dev/work/.claude/ag3s-rules-brief.md"   # 2026-10-06 토큰 절약: 요약본만 주입 (전문은 ag3s-rules.md)
[ -r "$RULES" ] || exit 0
# suppressOutput: 규칙 본문이 트랜스크립트에 매 턴 찍히지 않게 한다 (모델 컨텍스트에는 들어간다).
jq -Rs '{suppressOutput:true,hookSpecificOutput:{hookEventName:"UserPromptSubmit",additionalContext:.}}' < "$RULES"
