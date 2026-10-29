# AG3S-CORE-EVAL-G1 — 기하 grasp 보정 (G1) 을 최종 방법론 코드에 합치기

> writer: lead · 구현: ag3s-implementer · 작성: 2026-10-29 02:20 KST (`date -u` + 9 h).
> 상위: [AG3S-CORE-EVALUATION](AG3S-CORE-EVALUATION.task.md) §0.1 · [ADOPTED_KO.md](../../../../doc/active/reviews/AG3S_decision/ADOPTED_KO.md) §A8 "G1 통합".

## 0. 사용자 판정 (2026-10-29)

- 기하 grasp 보정 G1 ("VLA 닫기 근처 보정") 을 최종 방법론에 채택한다.
- 순서: 코드 합치기 (이 문서) → A1–A4 + A7 + G1 결합 closed-loop 18 run (lead) → 놓는 곳 허들 장면 28 run (최종 구성).
- G1 결정과 한계: [AG3S_GEO_GRASP_DECISION_20261029_KO.md](../../../../doc/active/reviews/AG3S_decision/AG3S_GEO_GRASP_DECISION_20261029_KO.md) §2.

## 1. 할 일

**출처:** worktree `/mnt/dev/AG3S_VLA-development/.claude/worktrees/geo-grasp` (branch `feature/geo-grasp`, HEAD `b3b69311` 위, commit 안 됨).
- task: 그 worktree 의 `AG3S/ag3s/docs/handoff/AG3S-GEO-GRASP.task.md`.
- G1 서버 flag: `--geo-grasp --geo-goal-mode correct --geo-track-scale 0.25 --geo-continuity-off` (기본 꺼짐).

**대상:** 메인 checkout `/mnt/dev/AG3S_VLA-development` 의 **지금 작업 트리**. 이것이 정본이다 — 평가에 쓴 snapshot_core 와 같다.

1. **G1 만 골라 합친다.**
   - worktree 의 `git diff HEAD` 에는 G1 말고도 AG3S-PRE-CORE-TO 의 더 이른 변경 (E1–E3, BLAS 등) 이 섞여 있다.
   - 이것들은 메인 작업 트리에 더 새 버전이 있으니 **메인 쪽을 그대로 둔다.**
   - 메인 작업 트리의 다른 변경 (손바닥 껍질 구, 꼭지 덮개, 팔 구, 쥐기 목표 항 window, 렌더 mask 등) 을 하나도 잃으면 안 된다.
2. **G1 에 필요한 코드만 가져온다.**
   - 예: `AG3S/ag3s/grasp/`, `AG3S/trajopt/geo_grasp.py`, `serve_safe.py` · `sqp.py` · `config.py` · `safe_policy.py` 의 geo 부분, 그 test.
   - 미채택 G2–G7 (접근 목표, 닫기 gate, 중심 통과 검사, 손목 · 팔 · 누적 장애물 검사, 거리장 조회, 타겟 쪽 유도 `approach_guide.py` 등) 은 G1 과 떼어낼 수 없을 때만 가져온다. 가져오면 기본 꺼짐을 유지하고 impl.md 에 적는다.
3. **기본 꺼짐:** geo flag 를 주지 않으면 서버 동작이 지금과 **bit 단위로 같아야** 한다.

## 2. 검증 (결과 전 고정)

| # | 검사 | 통과 조건 |
|---|---|---|
| V1 | 단위 test | 메인 checkout 에서 `.venv-ag3s` 로 `tests/trajopt` + 새 geo test 통과 (worktree 에서 돌리지 않는다 — pytest pythonpath 함정) |
| V2 | 꺼짐 동등성 | 핵심 평가 1차 서버 기록 (`$RUN/verify/AG3S-CORE-EVAL/REG_k0/runs/server_8292_*/server_constraints/run_0000/chunk_*.npz` 와 `OCC/k0/server_ag3s_8296_*/…`) 에서 요청 ≥ 20 개를 골라, 합친 코드로 같은 config (geo flag 없음) TrajOpt 를 재생한다. refined chunk 가 기록과 bit 동일. 재생 도구는 기존 것 (`AG3S/trajopt/experiments/tp_timing.py` replay 경로 등) 을 쓴다 |
| V3 | 켜짐 동등성 | worktree 코드와 합친 코드에 같은 입력 · 같은 G1 flag 를 주면 같은 결과 (worktree 의 `geo_grasp_replay.py` 류 offline 재생, 요청 ≥ 10). 단 메인 쪽 다른 기능 (껍질 구 등) 이 꺼진 구성에서 비교한다 |
| V4 | 결합 확인 (offline) | A1–A4 + A7 + G1 flag 로 서버가 뜨는지 (CPU import · arg 검사, GPU 서버 실행은 lead). 그리고 코드상 G1 항과 쥐기 목표 항 (`--grasp-anchor`) 이 같은 행에 어떻게 더해지는지 (항 이름 · 행 · 무게) 를 impl.md 에 적는다 — 동작은 바꾸지 않는다 |

## 3. 제약

- commit 하지 않는다. storage run 의 snapshot 디렉터리 · 실행 중인 서버 · 다른 세션 process 를 건드리지 않는다.
- CPU 만. 공용 pod 가 바쁘다 (worker ≤ 4) — test · 재생은 1 process, `nice -n 10`.
- 결과물: `AG3S/ag3s/docs/handoff/AG3S-CORE-EVAL-G1.impl.md`.
  - 바꾼 파일 · md5, 가져온 것 · 안 가져온 것.
  - V1–V4 결과 (숫자 · 로그 경로).
  - 결합 서버 argv 한 줄.

## 4. 사용자 판정 (2026-10-29 02:50 KST) — G1 과 A4 를 함께 켤 때

- **대체 방식 (권장안):** 기하 후보가 있는 chunk 는 G1 목표 (점군으로 확인한 grasp, 위치 + 회전), 없는 chunk 는 A4 목표 (VLA 처음 닫기 위치, 위치만).
- 구현: `CostConfig.validate` 의 G1 + A4 동시 거절만 푼다.
  - 기록에 그 chunk 를 어느 항이 채웠는지 남긴다.
  - 둘 중 하나라도 꺼지면 이전과 bit 동일하다.
- 근거 · 설명: 두 항은 같은 slot (닫는 행부터 3 행의 손바닥 목표) 을 쓴다. A4 = "VLA 의도를 지킨다", G1 = "VLA 의도를 근처의 기하적으로 맞는 grasp 로 고친다". 합산은 두 목표 사이로 당겨질 수 있어 택하지 않았다.
