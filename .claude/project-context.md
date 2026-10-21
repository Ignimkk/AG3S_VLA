# AG3S_VLA 현재 구조와 작업 기준

이 파일의 checkout이 현재 project root다. 첫 응답에서 cwd·Git root·branch·HEAD·origin을 확인한다.
`CLAUDE.md`, `doc/active/BRANCHES.md`, `doc/active/migration/source-manifest.json`,
`doc/archive/migration/CLAUDE_HANDOFF.md`를 읽는다. 기록보다 실제 branch와 manifest가 우선한다.

| Branch | 목적 |
|---|---|
| main | 원본 benchmark/v2 + pi05/v2를 통합한 V11AB, 공통 경로/규칙 정리 |
| feature/panda-benchmark | paired v2에서 Panda adapter와 LIBERO 계열 benchmark 평가 |
| feature/ag3s-development | v2 이후 전체 개선 이력과 미커밋 source checkpoint 계속 개발 |

Method는 `AG3S/ag3s`, `AG3S/trajopt`, method 문서는 `AG3S/docs`와 기존 `AG3S/ag3s/docs`다.
비교 방법은 `Baseline/`, 실제 benchmark는 `benchmark/`, 로봇/VLA 연결은
`integration/`, geometry는 `robots/`, 수정 OpenPI fork는 `third_party/openpi/`다.
OpenPI는 vendored own code이며 공식 benchmark·cuRobo·ALOHA는 고정 submodule이다.

이전 `/mnt/dev/work`의 세션을 resume하지 않는다. 현재 실행은 이전 checkout에 의존하지 않는다.
Input은 별도 `AG3S_VLA-storage`에 있고 `AG3S_STORAGE_ROOT`로 선택한다.
`legacy-inputs`는 과거 relative data 경로를 위한 호환 namespace이며 source checkout이 아니다.
현재 storage·interpreter 안내는 `doc/active/SHARED_STORAGE_AND_PYTHON.md`,
최신 검증 결과는 `doc/archive/migration/STORAGE_IMPLEMENTATION_REPORT.md`를 읽는다. 새 코드는 자기 checkout에서 편집한다.
`source scripts/setup/env.sh` 후 development 주 interpreter `.venv-openpi-live`를 쓴다.
CPU 회귀는 `.venv-ag3s`, cuRobo producer는 `.venv-curobo`로 분리한다. 세 환경 모두 자기 Python 3.11.16이다.
새 output은 `$AG3S_STORAGE_ROOT/results/development/<날짜_commit_run-id>/`, cache는 자기 checkout 아래다.
OpenPI `--record`의 상대경로는 run 안에서 해석하고 시작마다 새 session 디렉터리를 만든다.
현재 GPU·checkpoint 검증은 `doc/active/migration/GPU_OUTPUT_IMPLEMENTATION_REPORT.md`,
최신 `/mnt/dev/work` 정리·삭제 조건은 `doc/active/migration/LEGACY_RETIREMENT_REPORT.md`를 따른다.
Panda benchmark 환경과 end-to-end 평가는 별도 작업이다.

개선 branch를 평가 branch에 통째로 merge하면 V11AB 기준이 바뀐다.
구조/경로 공통 수정만 선택해서 전달하고 실험마다 source·adapter·dependency SHA를 기록한다.
Push는 사용자가 수행한다. 기존 Git history를 rewrite하거나 force push하지 않는다.
