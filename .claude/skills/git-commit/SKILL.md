---
name: git-commit
description: AG3S_VLA monorepo 커밋 규칙, 브랜치 확인, 소유 파일 staging, push 인수인계.
---

# AG3S_VLA 커밋 규칙

커밋 전에 실제 `git rev-parse --show-toplevel`, `git branch --show-current`,
`git remote -v`와 `doc/BRANCHES.md`를 확인한다. 이전의 다중 저장소 규칙은 폐기한다.

## 하나의 개발 저장소

| 코드/문서 | 현재 관리 위치 |
|---|---|
| AG3S·TrajOpt 및 기존 method 문서 | `AG3S/` |
| KNOWS·SEAM | `Baseline/` |
| 로봇/VLA 연결 | `integration/`, `robots/` |
| 자체 수정 OpenPI fork | `third_party/openpi/` — vendored, submodule 아님 |
| tests·CLAUDE.md·.claude·paper·운영 문서 | 같은 AG3S_VLA repository |
| 공식 benchmark·cuRobo·ALOHA | 외부 submodule, commit pin 관리 |

위 직접 개발 내용은 모두 현재 checkout의 root에서 커밋한다.
`origin`은 `git@github.com:Ignimkk/AG3S_VLA.git`이다.
기존 repository와 `legacy/*` 보존 refs에는 새 개발 commit을 쌓지 않는다.
외부 submodule을 수정할 필요가 있으면 dependency fork/업데이트 정책을 먼저 정한다.

## 커밋 내용

- 사용자 지시대로 Claude 출처·Co-Authored-By를 넣지 않는다.
- 한국어 본문에 필요한 기술 용어는 English로 쓴다.
- 제목은 `ag3s:`, `trajopt:`, `integration:`, `benchmark:`, `migration(layout):` 등 변경 범위를 표시한다.
- 자기 소유 파일만 명시해서 `git add`하고 `git diff --cached --stat`를 검토한다.
- raw recording·checkpoint·dataset·venv·cache는 Git 밖에 둔다.
  재현에 필요한 source SHA·config·artifact SHA-256은 manifest로 남긴다.
- 기존에 추적된 작은 fixture를 일괄 삭제하지 않는다.

## 브랜치 기준

`main`은 paired v2 V11AB 기준과 공통 구조 수정이다.
`feature/panda-benchmark`는 그 기준에서 adapter/benchmark 평가를 개발한다.
`feature/ag3s-development`는 v2 이후 개선 commit과 migration checkpoint를 포함한다.
개선 branch 전체를 평가 branch에 merge하면 V11AB 기준이 바뀐다.
공통 수정만 검토한 작은 commit으로 전달하고 실험 source SHA를 기록한다.

## Push

현재 사용자는 push를 직접 수행한다. agent는 commit까지 완료하고 브랜치와 SHA를 보고한다.
승인 없이 push하거나 자격증명을 생성하지 않는다. 원본 이력 보존 시 force push,
`--mirror`, squash/rebase를 사용하지 않는다.
