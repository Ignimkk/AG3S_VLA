# Legacy checkout retirement dependency 정리

기록: **2026-10-12 KST**, `feature/ag3s-development`, 시작 HEAD `d169945993272f123dd4c1770fa14000b899de28`.
목적: layout / archival preservation. AG3S algorithm, 모델·input, 원본 개선 이력을 유지한다.
이 보고서는 [이전 GPU/output 보고서](GPU_OUTPUT_IMPLEMENTATION_REPORT.md)의 삭제 조건에 대한 후속 기록이다.
GPU·checkpoint 검증 결과는 이전 보고서에 유지하며 이 단계에서 GPU를 추가 사용하지 않는다.

Permanent archive:
`/mnt/dev/AG3S_VLA-storage/archive/legacy-retirement-20261012T013141+0900-d1699459/`.
Raw audit run:
`/mnt/dev/AG3S_VLA-storage/results/development/20261012T013141+0900_d16994599327_f6e6fcbe1d91/`.
기존 result의 보존 자료를 복제했으며 이전 evidence와 공유 input은 그대로 유지했다.

## 보존과 변경 범위

| 항목 | 보존 방법 |
|---|---|
| 잔여 자료 | 이전 `legacy-preservation`의 파일을 `supplemental/`에 독립 복제하고 모든 SHA-256 비교 |
| 관리 문서·환경 근거 | `provenance/legacy-management-v1.4`, `provenance/gpu-environment`에 복제 |
| 원본 Git | workspace, benchmark, pi05_TO_hybrid, OpenPI, cuRobo, ALOHA, LIBERO의 complete all-refs bundle 및 verify log |
| 과거 sibling source | `archive/source-snapshot-20261011`의 동일 relative subtree로 연결; active development source로 연결하지 않음 |
| detached benchmark worktree | bundle에서 독립 clone 후 기존 HEAD로 checkout; tracked 파일 identity·clean status·fsck 확인 |

Symlink planner는 무시된 source·launcher도 포함하여 file SHA-256, size, executable 여부,
directory와 symlink target을 비교한다. `.git`, `.venv*`, bytecode와 명시한 cache만 제외한다.
Target subtree 밖을 향한 link는 명시된 stable shared input 외에는 거부한다.
변경 직전에 plan과 실제 파일을 재검증한다. Link별 atomic replace와 rollback manifest를 사용한다.

Planner는 [plan_legacy_retirement.py](../../scripts/setup/plan_legacy_retirement.py)의 read-only 도구다.
임의 archive를 자동 apply하는 기능은 제공하지 않는다. 실제 승인된 link 목록과 apply 결과는
permanent archive의 `symlink-migration.json`, 원본 비교는 raw run의 `link-plan-before.json`에 기록한다.
Git refs/HEAD/status, 보존 checksum, worktree 검증은 archive의 JSON과 독립 validation에 남긴다.

## 삭제 판단

`/mnt/dev/work` 자체는 삭제하지 않았으며 다른 프로세스를 종료하지 않았다.
사용자는 기존 VS Code 창·terminal을 닫고 새 checkout으로 전환했다고 확인했다.
UI 전환 뒤에도 extension helper가 이전 Python을 유지할 수 있으므로 실제 `/proc` 참조를 판정에 사용한다.
Process PID는 관측 시점 값이며 다른 용도로 재사용될 수 있어 종료 명령으로 재활용하지 않는다.

남은 helper가 있으면 본인이 소유한 VS Code 세션에서 interpreter를 새 checkout으로 선택하고
`Developer: Reload Window` 또는 해당 Remote 연결을 정상 종료한다. 다른 checkout을 쓰는 창은
그 checkout의 interpreter를 선택한다. 이전 helper가 남으면 해당 세션 소유자가 정상 종료한 뒤 다시 점검한다.
일괄 `pkill`이나 다른 세션 전체 종료는 수행하지 않는다.

삭제 직전에는 incoming symlink, registered worktree, `/proc` cwd/exe/open fd/maps, 보존 archive를
다시 검사한다. 현 namespace에서 읽을 수 없는 process나 외부 container/mount의 사용까지 보장하지 않는다.
실제 삭제는 별도 단계다. Current branch만 commit하고 push는 사용자가 수행한다.
