# 역할별 settings — 소유권을 권고가 아니라 차단으로 만든다

```bash
claude --agent ag3s-implementer --settings .claude/roles/implementer.settings.json
claude --agent ag3s-verifier    --settings .claude/roles/verifier.settings.json
claude --agent ag3s-scribe      --settings .claude/roles/scribe.settings.json
```

## 왜 `/${AG3S_PROJECT_ROOT}/...` 절대경로인가

`--settings <file>` 로 넘긴 파일 안에서 **슬래시 하나로 시작하는 경로(`/benchmark/**`)는
그 settings 파일이 있는 디렉토리 기준**으로 풀린다. 즉 `.claude/roles/benchmark/**` 가 되어
아무것도 막지 못한다. `//` 로 시작하는 절대경로만 의도대로 동작한다.

## 왜 전부 `Edit(...)` 인가

Claude Code 는 파일 경로 규칙을 **`Edit(...)` 와 `Read(...)` 로만** 검사한다.
`Write(...)`·`NotebookEdit(...)`·`MultiEdit(...)` 로 쓴 경로 규칙은 **받아들이지만 보지 않는다**
(시작할 때 경고만 뜬다). `Edit(...)` 하나가 파일을 고치는 모든 내장 도구를 덮는다.

## 한계 — 이것은 sandbox 가 아니다

deny 규칙은 내장 파일 도구, Claude Code 가 알아보는 Bash 파일 명령(`cat`·`tee`·`sed` …),
그리고 `> file` 같은 리다이렉션까지 덮는다. **하지만 파이썬 스크립트가 스스로 여는 파일은
덮지 않는다.** 소유권의 1차 방어선은 `.claude/agents/*.md` 의 역할 지시이고,
이 파일들은 실수로 옆 사람 파일을 고치는 것을 막는 2차 방어선이다.
