# AG3S_VLA branch와 merge 규칙

| Branch | 출발점 | 목적 |
|---|---|---|
| `main` | `legacy/benchmark/v2` + `legacy/pi05_TO_hybrid/v2` + 당시 pinned OpenPI | V11AB 통합 기준; 구조·경로·운영 규칙 변경만 적용 |
| `feature/panda-benchmark` | 통합 `main` | Panda adapter 및 LIBERO → SafeLIBERO → LIBERO-Safety 평가 |
| `feature/ag3s-development` | 통합 `main` + 기존 최신 이력 + migration checkpoint | v2 이후 AG3S 개선과 현재 WIP 계속 개발 |

`main`은 원본 두 tag의 SHA 자체가 아니다. 서로 다른 repository의 v2 코드를 같은
구조로 배치하고 경로를 정리한 monorepo commit이다. 원본 tag object와 commit SHA는
`legacy/*` refs에 그대로 보존한다. Workspace의 v2 대응 테스트는 benchmark/v2 commit
시간 이전의 마지막 first-parent commit을 사용했으며 manifest에 정확한 SHA를 기록한다.

두 개발 branch는 공통 `main`의 후손이다. 개선 branch에만 최신 checkpoint를 추가한다.
`feature/ag3s-development`를 benchmark branch나 main에 통째로 merge하면 V11AB 기능
기준이 바뀐다. V11AB 평가 기간에는 구조·경로 공통 수정만 작은 commit으로 전달하고,
기능 개선은 명시적인 새 실험 버전으로 평가한다. 원본 이력 이관에는 squash/rebase를
사용하지 않았다. 이후 작은 개발 수정의 cherry-pick은 가능하며 적용 SHA를 기록한다.

## Merge 전에 확인

1. `git branch --show-current`와 두 branch의 `git merge-base`를 확인한다.
2. PR/commit에 `layout`, `adapter`, `algorithm`, `evaluation` 중 변경 목적을 적는다.
3. `git diff main...HEAD -- AG3S integration robots`로 기능 변화 범위를 검토한다.
4. baseline의 tag·AG3S commit·adapter·benchmark pin·config를 run manifest에 기록한다.
5. V11AB baseline을 유지해야 하면 개선 branch 전체 merge를 하지 않는다.

## 작업 폴더

- `/mnt/dev/AG3S_VLA-development`: `feature/ag3s-development`, VS Code 창과 Claude 세션 하나.
- `/mnt/dev/AG3S_VLA-benchmark`: `feature/panda-benchmark`, 별도 VS Code 창과 Claude 세션.
- `/mnt/dev/work`: 기존 실행 환경·dataset·checkpoint·raw evidence의 legacy storage.
  새 코드 개발은 여기서 재개하지 않는다. 기존 자료는 삭제하지 않았다.

각 checkout의 venv·새 output·cache는 별도로 관리한다. `scripts/setup/env.sh`는 자신의
checkout 코드만 PYTHONPATH에 추가한다. Dataset/checkpoint 등 기존 input은
`AG3S_LEGACY_STORAGE`로 선택한다. 기존 venv를 새 폴더로 복사하거나 공유하지 않는다.

## Claude 재시작

새 폴더에서 **새 세션**을 시작한다. `/mnt/dev/work`의 이전 세션을 resume해 코드 편집을
이어가지 않는다. `CLAUDE.md`, `.claude/project-context.md`, 이 문서, 자기 branch의
`doc/migration/source-manifest.json`을 먼저 읽는다. SessionStart와 UserPromptSubmit hook이
실제 Git root·branch·HEAD와 branch 목적을 주입한다. Hook 실패 시 문서를 직접 읽고
`scripts/setup/check_workspace.py`로 실제 경로를 확인한다.

처음에는 cwd·Git root·branch/HEAD·origin·module __file__·남은 WIP를 보고한다.
코드 복사로 branch를 동기화하지 않는다. Push/fetch 후 필요한 commit을 선택한다.
