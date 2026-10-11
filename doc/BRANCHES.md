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

두 개발 branch는 최초 통합 `main`을 공통 조상으로 갖는다. 이후 공통 경로 변경은
각 branch에 cherry-pick할 수 있으므로 최신 main의 후손 관계가 항상 성립하는 것은 아니다.
개선 branch에만 최신 checkpoint를 추가한다.
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
- `/mnt/dev/AG3S_VLA-storage`: checkpoint·dataset·raw evidence의 독립 공유 input storage.
- `/mnt/dev/work`: 원본 code/Git 보존과 이동한 input의 backlink. 새 코드 개발은 여기서 재개하지 않는다.

각 checkout의 venv·새 output·cache는 별도로 관리한다. `scripts/setup/env.sh`는 자신의
checkout 코드만 PYTHONPATH에 추가한다. Dataset/checkpoint 등 기존 input은
`AG3S_STORAGE_ROOT`로 선택한다. `AG3S_LEGACY_STORAGE`는 그 아래 `legacy-inputs`의 파생 alias다. 기존 venv를 새 폴더로 복사하거나 공유하지 않는다.

## Claude 재시작

새 폴더에서 **새 세션**을 시작한다. `/mnt/dev/work`의 이전 세션을 resume해 코드 편집을
이어가지 않는다. `CLAUDE.md`, `.claude/project-context.md`, 이 문서, 자기 branch의
`doc/migration/source-manifest.json`을 먼저 읽는다. SessionStart와 UserPromptSubmit hook이
실제 Git root·branch·HEAD와 branch 목적을 주입한다. Hook 실패 시 문서를 직접 읽고
`scripts/setup/check_workspace.py`로 실제 경로를 확인한다.

처음에는 cwd·Git root·branch/HEAD·origin·module __file__·남은 WIP를 보고한다.
코드 복사로 branch를 동기화하지 않는다. Push/fetch 후 필요한 commit을 선택한다.

## Historical comparison tests and legacy imports

Some existing simulation tests intentionally compare original pi05 Git snapshots.
Run `python scripts/setup/prepare_legacy_test_fixture.py` to materialize the frozen
pi05/v2 under this checkout's ignored `.runtime/legacy-pi05`; these tests no longer
load the old `/mnt/dev/work/pi05_TO_hybrid` source tree. The fixture's Git objects
refer only to this checkout's object store. It is not another development branch.
Current integration imports/assets are separately checked at `integration/rby1`
and `robots/rby1/description`. Future adapter tests should use those current paths.

Recorded clients may still import `benchmark.ag3s` or `benchmark.trajopt`.
`benchmark/__init__.py` provides a compatibility bridge to the same canonical
module objects in this checkout; new code uses `AG3S.*` and `Baseline.*` directly.

Run `bash scripts/setup/create_cpu_env.sh` to create an independent CPU environment
and Python installation in each checkout. GPU/VLA and benchmark environments are
separate setup tasks; this migration does not claim an end-to-end rollout result.

## External artifacts

Ignored raw attention recordings and the installed paper tool binary are stored
outside Git. `experiments/manifests/legacy-artifacts.json` records their original
paths and SHA-256. Local ignored links resolve only these immutable inputs, not
another checkout's Python source. None of the newly committed checkpoint blobs
exceeds GitHub's 100 MiB limit. Original commit objects and original tags remain
unchanged; only unpublished migration checkpoints were corrected before push.

The fixture setup also materializes this branch's original benchmark source under
`.runtime/legacy-benchmark` for Git-history comparisons. Archived numerical
references expose canonical package entry points, ensuring the reference runs
its archived algorithm instead of accidentally importing the current method.
CPU simulation test dependencies are pinned in `environments/testing/requirements.txt`.


## Latest handoff

`doc/migration/CLAUDE_HANDOFF.md`에 새 세션의 첫 확인과 남은 benchmark 작업을 기록한다.
`scripts/setup/prepare_legacy_artifacts.py --relink`는 manifest의
SHA-256을 확인한 뒤 ignored input link만 복원한다. 다른 checkout의 source는 연결하지 않는다.
`pyproject.toml` editable package와 환경 requirements는 알고리즘 release 기준과 구분한다.

현재 storage와 Python 정본: `doc/SHARED_STORAGE_AND_PYTHON.md`,
`doc/migration/STORAGE_IMPLEMENTATION_REPORT.md`.
