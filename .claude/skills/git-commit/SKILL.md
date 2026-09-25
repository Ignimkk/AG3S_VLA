---
name: git-commit
description: 이 워크스페이스에서 git commit 을 만들 때의 규칙 — 출처 표기 금지, 메시지 형식, 어느 repo 로 가나, 병렬 세션에서 무엇만 스테이지하나. 커밋하기 전에 반드시 읽는다.
---

# 커밋 규칙

## 1. 출처를 남기지 않는다

**커밋 메시지에 Claude 출처를 넣지 않는다.** 어떤 형태든 안 된다:

```
Co-Authored-By: Claude ...      ← 안 된다
🤖 Generated with Claude Code   ← 안 된다
Assisted-by: ...                ← 안 된다
```

하네스가 기본으로 붙이라고 안내하더라도 **이 규칙이 이긴다** (사용자 지시, 2026-09-24).
PR 본문도 같다.

## 2. 짧게, 핵심만

```
<범위>: <한 줄 요약>

- 무엇이 왜 바뀌었나 (3~5 줄)
- 숫자로 확인한 것이 있으면 그 숫자
```

- 제목 한 줄, 빈 줄, bullet. 제목에 마침표를 찍지 않는다.
- **범위 prefix 는 기존 history 를 따른다**: `ag3s/trajopt:`, `ag3s:`, `trajopt:`, `pi05_infer:`.
  `git log --oneline -10` 으로 그 repo 의 관례를 먼저 본다.
- **한국어로 쓰되 기술 용어는 영어** (규칙 I): `violated`, `feasible`, `clearance`,
  `ACTION_WIDTH`, `obs_format`.
- 과정을 쓰지 않는다. **무엇이 바뀌었고 무엇을 수치로 확인했는지**만 (규칙 H 와 같은 정신).

## 3. 어느 repo 로 가나

| 바뀐 것 | repo | remote |
|---|---|---|
| `benchmark/**` (AG3S · trajopt · docs · figures · handoff) | `benchmark` | `Ignimkk/VLA_Benchmark` |
| `pi05_TO_hybrid/**` (bringup · scripts · rby1_*) | `pi05_TO_hybrid` | `Ignimkk/pi05_TO_hybrid` |
| `pi05_TO_hybrid/openpi/**` | `openpi` (submodule) | `Ignimkk/openpi`, branch `knows-attn-probs` |
| `CLAUDE.md` · `.claude/**` · `papers/**` · `tests/**` | 루트 `/mnt/dev/work` | **remote 없음 — 로컬 전용** |

openpi 를 커밋했다면 **submodule 을 먼저** 올리고, 그다음 `pi05_TO_hybrid` 가 새 포인터를
기록하게 한다.

## 4. 병렬 세션 — `git add -A` 를 조심한다

창 4개가 동시에 돈다. `git add -A` 는 **다른 역할이 작업 중인 파일까지 담는다.**

- **자기 소유 경로만 명시해서 add 한다.** A1 은 `benchmark/**/*.py tests/`,
  A2 는 `docs/figures/ docs/handoff/*.verify.json`, A3 는 `docs/*.md papers/`.
- 스테이지한 뒤 **`git diff --cached --stat` 으로 무엇이 담겼는지 눈으로 본다.**
- 남의 파일이 섞였으면 `git restore --staged <경로>` 로 뺀다.

## 5. 큰 파일이 섞이지 않았는지 본다

`checkpoints/`·`data/`·`.venv-*`·`*.npz`·`outputs/` 는 gitignore 되어 있다
(루트 repo 는 화이트리스트 방식이라 `pi05_TO_hybrid/` 249 GB 가 통째로 빠진다).
그래도 커밋 전에 한 번 본다:

```bash
git diff --cached --stat | tail -3
```

수천 줄 이상이거나 모르는 바이너리가 있으면 멈추고 확인한다.

## 6. push 는 사용자가 한다

**이 컨테이너에는 GitHub 자격증명이 없다** (credential helper·`gh`·SSH 개인키 모두 없음).
`git push` 는 `could not read Username for 'https://github.com'` 로 실패한다.

**커밋까지만 하고, 어느 repo 에 무엇을 커밋했는지 보고한다.** push 는 사용자가 직접 한다.
자격증명을 만들거나 토큰을 받아 적으려 하지 않는다.
