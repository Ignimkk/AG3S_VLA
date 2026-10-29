# Storage 경로 분리 실행 결과

완료 기록: **2026-10-11 23:58 KST**. 현재 경로 및 환경 안내의 정본은 이 문서와
`doc/active/SHARED_STORAGE_AND_PYTHON.md`다. 이전 `IMPLEMENTATION_REPORT.md`는 최초 통합 당시 기록이다.
사용자가 이전 통합 commit과 원본 보존 refs를 push했다고 확인했다. 이번 후속 commit은 아직 push하지 않았다.

## 현재 구성 및 브랜치

| 작업 | 위치 / branch | 기준 |
|---|---|---|
| 개선 개발 | `/mnt/dev/AG3S_VLA-development` / `feature/ag3s-development` | v2 이후 원본 commit과 WIP checkpoint 유지 |
| 평가 개발 | `/mnt/dev/AG3S_VLA-benchmark` / `feature/panda-benchmark` | benchmark/v2 + pi05/v2의 V11AB source 유지 |
| 공통 기준 | `main` | paired v2에 이번 storage·경로·운영 변경 적용 |
| 공유 input | `/mnt/dev/AG3S_VLA-storage` | Git 외부 checkpoint·dataset·recording·기존 결과·도구 |
| 이전 workspace | `/mnt/dev/work` | 원본 코드·Git 보존, 이동한 자료 경로는 새 storage로 backlink |

`main`은 서로 다른 repository의 v2를 통합한 commit이며 원본 tag SHA 자체가 아니다.
원본 annotated tag object는 `legacy/benchmark/v2` = `183bb5e5185048453656c57893d57787f1e2ee8a`,
`legacy/pi05_TO_hybrid/v2` = `c1d304fabea9da0caca286ae8d69ba9980e9cd07`로 보존한다.
원본 tag·legacy refs·base refs는 rewrite하지 않는다. 개선 branch 전체를 main/평가 branch에 merge하지 않는다.
공통 경로 변경은 양쪽에 적용했으며 cherry-pick commit의 SHA는 각 branch에서 다를 수 있다.
OpenPI는 `third_party/openpi`에서 직접 추적한다. 공식 benchmark 3개·cuRobo·ALOHA는 5개 고정 submodule이다.

## 수행 내용

수정 전 두 clone의 검증된 Git bundle·tracked archive·설정 metadata 및 input의 hardlink namespace
백업을 먼저 만들었다. Hardlink 백업은 같은 filesystem의 rename/unlink를 보호하는 방식이며
원본 내용에 대한 독립 복제는 아니다. Dataset/checkpoint 내용은 편집하지 않았다.
약 361.3 GiB의 input은 같은 filesystem에서 35개 경로를 rename하여 독립 storage에 배치했다.
실제 위치 목록은 `experiments/manifests/shared-storage.json`에 기록한다.

새 checkout의 runtime·attention/tool link·OpenPI assets/policy_records link는 새 storage를 직접 사용한다.
`AG3S_STORAGE_ROOT`가 위치 선택 변수이고 `AG3S_LEGACY_STORAGE`는 그 아래 `legacy-inputs`의 파생 alias다.
상속된 이전 dataset/cache/interpreter 기본값도 새 경로로 전환한다. 현재 source와 editable package는
자기 checkout을 사용한다. 역사적 Git 비교는 자기 `.runtime/legacy-*` fixture와 자기 Git object만 사용한다.
Replay harness는 frozen client의 input 주소만 재설정하며 archived algorithm은 바꾸지 않는다.

기존 실험 runroot의 source link는 storage 내부 `archive/source-snapshot-20261011`에 보존한
역사적 source로 연결했다. Archive manifest에서 source 파일 8,972개 SHA-256 일치를 검증했다.
공유 storage의 외부로 나가는 symlink는 없다. 새 clone의 이전 workspace symlink도 없다.
과거 log/recording의 원래 절대경로는 provenance이므로 일괄 수정하지 않았다.
이관된 전체 파일 identity와 보존 기록은 storage의 `migration-file-identities.jsonl`, `migration-inputs.json`에 있다.

## 검증과 한계

| 항목 | 결과 |
|---|---|
| input inode/device 보존 | 108,984개 모두 일치 |
| input size/mtime까지 동일 | 108,983개 일치 |
| 기존 동작 중 loadavg log | 1개가 기존 sampler로 계속 append됨; inode 동일, 이번 작업에서 내용 편집 없음 |
| attention/tool 8개 | 양쪽 link·size·SHA-256 정상 |
| interpreter | 양쪽 독립 Python 3.11.16, own-root `.venv-ag3s` |
| source import / dependency | 양쪽 자기 source import, 5개 submodule pin 정상 |
| 선택한 경로·replay·simulation 회귀 | **각각 148 passed / 1 skipped / 5 long replay deselected** |
| 이전 code 읽기 차단 | Python audit probe 및 차단을 적용한 위 테스트 정상 |
| V11 import guard | 양쪽 server/client에서 자기 checkout source 확인 |
| Claude | 새 storage/interpreter 안내 및 session hook 정상 |

Python audit는 open·directory listing·chdir와 Python subprocess를 검사한다.
Native library 및 Python 이외 프로세스의 모든 syscall을 막는 filesystem sandbox는 아니다.
선택한 회귀 검증 결과가 이전 전체 회귀의 모든 수치 문제를 해결했다는 의미는 아니다.
이전 전체 검증에서 관찰된 numerical reproducibility 실패는 최초 보고서에 유지한다.
GPU/OpenPI/cuRobo 별도 환경 설치, Panda adapter, LIBERO 계열 episode 평가는 이번 작업 범위에서 수행하지 않았다.

아래 두 dataset alias의 target은 **이관 전부터 없었다**. 데이터 손실 또는 신규 복구로 기록하지 않는다.

- `datasets/lerobot-atomic/local/rby1_atomic_basket_14d_v2`
- `datasets/lerobot-transport/local/rby1_dataset_v1` (target: `datasets/rby1/rby1_transport_14d`)

Raw log·JUnit·경로 집계는 `doc/archive/migration/validation/storage/`에 보관한다. Raw JSON의 HEAD는
실행 시점 commit이며 후속 문서 commit으로 덮어쓰지 않는다. 실제 최종 HEAD는 `git log -1`로 확인한다.
위 검증 후 이번 임시 백업 `/mnt/dev/AG3S_VLA-storage-migration-backup-20261011T132205Z`을 **제거 완료**했다.
이전 workspace와 사용자의 다른 backup은 보존했다.

## 사용자가 수행할 작업

두 새 폴더를 별도 VS Code 창으로 열고 각각 자기 `.venv-ag3s/bin/python`을 선택한다.
새 terminal에서는 `source scripts/setup/env.sh` 후 `source .venv-ag3s/bin/activate`를 실행한다.
기존 Claude session을 resume하지 않고 새 session에서 `CLAUDE.md`, `.claude/project-context.md`,
`doc/active/BRANCHES.md`, `doc/active/SHARED_STORAGE_AND_PYTHON.md`, 이 보고서를 읽는다.

아래는 후속 commit push 명령이다. 원본 legacy branch/tag를 다시 생성하거나 force push할 필요는 없다.

```bash
cd /mnt/dev/AG3S_VLA-development
git push origin main feature/ag3s-development
cd /mnt/dev/AG3S_VLA-benchmark
git push origin feature/panda-benchmark
```

다음 구현 순서는 Panda adapter 검증 → LIBERO Spatial 소규모 평가 → SafeLIBERO Spatial I/II pilot →
전체 SafeLIBERO → LIBERO-Safety 추가 평가다.
