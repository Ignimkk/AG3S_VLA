#!/usr/bin/env bash
# 역할 세션을 띄운다.  사용법:  .claude/roles/launch.sh {implementer|verifier|scribe|lead}
#
# 이 컨테이너에는 `claude` 가 PATH 에 없다 — VSCode 확장이 자기 안에 번들한 바이너리를 쓴다.
# 그래서 경로를 여기서 한 번 풀고, 확장이 올라가면 자동으로 새 버전을 집게 해 둔다.
set -euo pipefail

CLI="$(command -v claude || true)"
if [ -z "$CLI" ]; then
  CLI="$(ls -d "$HOME"/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude 2>/dev/null | sort -V | tail -1 || true)"
fi
if [ -z "$CLI" ]; then
  echo "claude CLI 를 찾지 못했다. VSCode 확장이 설치돼 있는지 보라." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/setup/env.sh"

case "${1:-lead}" in
  lead)        exec "$CLI" ;;
  implementer|impl|a1)
    exec "$CLI" --agent ag3s-implementer --settings "$ROOT/.claude/roles/implementer.settings.json" "${@:2}" ;;
  verifier|verify|a2)
    exec "$CLI" --agent ag3s-verifier    --settings "$ROOT/.claude/roles/verifier.settings.json" "${@:2}" ;;
  scribe|a3)
    exec "$CLI" --agent ag3s-scribe      --settings "$ROOT/.claude/roles/scribe.settings.json" "${@:2}" ;;
  *) echo "알 수 없는 역할: $1  (lead|implementer|verifier|scribe)" >&2; exit 2 ;;
esac
