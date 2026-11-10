# 다음 세션 인계 — AG3S 개선 마무리 → 최종 구성 동결 (작성 2026-11-11 03:5x KST, dev-55)

## 0. 새 세션에 붙여 넣을 프롬프트

> `doc/active/NEXT_SESSION_HANDOFF_20261111_KO.md` 를 읽고 이어서 진행해 주세요. 목표는 최종 구성 동결 (D6) 전에 AG3S 개선 작업을 끝내는 것입니다.
> 계획 · 일정은 `doc/active/experiments/AG3S_PLAN_SCHEDULE_20261110_KO.md`, 미채택 단계 추종 모듈은 `doc/active/experiments/STAGE_TRACK_ST1_KO.md` 입니다.
> §3 의 순서대로 하되, GPU 작업 전에는 계획서 §2 일정과 다른 세션 (ListAgents) 을 확인하고, 단계마다 commit 후 다음으로 가세요.

## 1. 목표와 마감

| 항목 | 내용 |
|---|---|
| 지금 할 일 | **최종 구성 동결 (D6) 전에 개선을 끝낸다** (사용자 2026-11-11). 동결 뒤 바꾸면 보고 run 을 다시 해야 한다 |
| 그 다음 | 주말 (11-14 – 11-15) AG3S 논문 실험: SafeLIBERO 4 suite × L-I · L-II, init 19–28, AG3S 최종 구성 · AG3S GT oracle · π0.5 anchor, + LIBERO · ablation · grounding · LIBERO-Safety · 실로봇 — 약 46–51 GPU h (`paper/notes/PAPER_REVISION_REPORTED_COMPARISON_20261111_KO.md` §10) |
| 마감 | 원고 제출 **11-17 20:59 KST** → 개선 · 동결은 **11-13 안** 이 현실적 [P] |
| 분할 규칙 | 개발 = E1a init 5–14 + SafeLIBERO dev ep 0–4 · LIBERO seed 7–11 · LS10 (L0 seed 7). init 15–18 = baseline port 검증. **init 19–49 = 보고 분할 — 개선 중에는 보지 않는다** |
| 비교 기준 | 지금 구성 `…-pe123458` 의 평가 규약 수치 (계획서 §3): 보정 18 → 14, LIBERO 27 → **22**, SafeLIBERO 11 → 8 · **6** · 4 (> 50 mm 1), LS 10 → **5** (위반 0), E1a 40 (5–14) → SSR **15** |
| 후보 채택 규칙 (G163 · G172, 결과 전 고정) | SafeLIBERO 11 안전 성공 ≥ 지금 · 큰 충돌 (> 50 mm) 안 늘어남 · LIBERO 27 ≥ 지금 · LS 10 ≥ 지금 · 위반 0 |

## 2. 이번 세션 (dev-55, 11-10 16:03 – 11-11 03:xx) 에 한 것

| 무엇 | 결과 | 위치 |
|---|---|---|
| 제약 단계 추종 P0–P5 (latch 의 단계 오류 측정 → hidden state 비교 → 추정기) | 오프라인: S3 놓침 197 → 69, 빈 닫힘 미감지 44 → 6, 거짓 HELD 67 → 125 (오류 합 308 → 200). HMM 은 더 나쁨 | `outputs/2026-11-10_제약단계_추종/README.md` §1–§11 (git 제외) |
| **ST1** 서버 이식 (`--stage-track`, 기본 꺼짐, test 14) | GPU 경로 검증 통과 (서버 h_10 = 학습 feature, 오차 0.0) | branch `wip/stage-track` |
| ST1 closed-loop (줄인 평가 규약 48 ep) | LIBERO 27 **21** (vs 22), SafeLIBERO 안전 성공 **5** (vs 6), > 50 mm 충돌 **3** (vs 1), LS 10 **5** (vs 5) → **미채택** | `doc/active/experiments/STAGE_TRACK_ST1_KO.md` §5 |
| ST1 실패 형태 | 요청 하나의 판정으로 attach / detach 를 바꿔 **깜빡임** (놓기 → 다시 닫기 반복 → 충돌 · 시간 초과), 일부 **이른 attach**. 대신 s9 t1 (첫 held call) · s8 t9 · SafeLIBERO t3 e3 (와인병) 은 고침 | 같은 문서 §5 |
| 발견: E1a 2 ep 에서 로봇이 **장애물 머그 · 다른 그릇을 실제로 쥠** | GT 없는 목표 식별 오류 (단계 추종으로 못 고침) | 같은 문서 §4, README §10 |
| 계획서 (전 세션 일정 · 남은 실패와 원인 · 다음 작업) | dev-4a · dev-e1 이 사용자 결정으로 갱신 (분할 A안, baseline 보고 run 보류, 주말 실험) | `doc/active/experiments/AG3S_PLAN_SCHEDULE_20261110_KO.md` |

### 2.1 ST1 closed-loop 최종 [M] (2026-11-10 23:20 – 2026-11-11 03:41)
| | pe123458 | ST1 |
|---|---|---|
| 평가 LIBERO 27 | 22 | 21 |
| SafeLIBERO 11 성공 · 안전 성공 · 충돌 · > 50 mm | 8 · 6 · 4 · 1 | 8 · 5 · 5 · 3 |
| LIBERO-Safety 10 (위반) | 5 (0) | 5 (0) — 2 ↔ 2 맞바꿈 (human_safety e0 은 ST1 이 성공) |

규칙 6 중 3 미달 → **미채택**. 원본 `outputs/2026-11-10_제약단계_추종/p6/pe123458st1/p6_compare.json`.

## 3. 다음에 할 일 (순서 · 권고)

### 3.1 개선 후보 (동결 전) — GPU 예산 순으로

| 순위 | 작업 | 노리는 실패 (계획서 §4) | 먼저 할 것 | GPU | 판단 근거 |
|---|---|---|---|---|---|
| 1 | **T1: PE11 단독 판 `…-pe12345811`** 평가 규약 전체 (+ E1a 5–14) | 충돌 기제 C (장애물이 대상과 한 덩어리 → 장애물에서 빠짐), E1a 충돌 6–7 ep | 판 · test 있음 (handoff 11-09 §4 T1) | 3.5 h | 개선 후보 중 기제 근거가 가장 강함 (offline 39/39, closed-loop 머그 277 mm · 책 6 mm 제거) |
| 2 | **정답 latch arm `…-pe123458-ol`** (줄인 규약 48 ep) | 단계 추종의 상한 | 판 있음 (`p6_queue.sh V=…-ol TAG=pe123458ol`) | 1.4 h | 이것도 지금과 차이가 없으면 단계 추종을 접는다. 차이가 크면 ST2 를 시도할 가치 |
| 3 | **ST2** (ST1 + 2 요청 연속 판정) 또는 ST3 (S1 CLOSING 해제만) | ST1 의 깜빡임 · 이른 attach | **오프라인 먼저** (`p5a_proba.npz` 에 연속 규칙 → 요청 오류 + 상태 전환 횟수, CPU 수 분) → 규칙 고정 → 서버 1 줄 변경 → 48 ep | 1.4 h | 2 번 결과가 상한을 보여줄 때만 |
| 4 | T2: τ_obs 5 mm (`…-to5`) | 기제 A (장애물 행 6–9.7 mm 침범을 허용치 안으로 통과), SafeLIBERO t2 e2 | profile 에 판 등록 + test | 3.5 h | LIBERO · LS 의 HOLD 증가 위험 |
| 5 | T3: 목적지 등록 지연 (PE10) | LIBERO s8 t9, E1a 놓기 실패 4 | 오프라인 채점 (`dest_latency.py`) | — | 설계 1 일 |
| 6 | T4: 첫 held call 틈 · 서랍 t4 | LIBERO s9 t1 · s7/s8 t4, LS human_safety (쥔 뒤 기하 미인증 HOLD) | 설계 | — | ST1 이 s9 t1 은 고쳤음 (S3 진입이 빠르면 풀림) — ST2/ST3 와 같이 보기 |
| 7 | 잘못된 물체 파지 감지 (새) | E1a t1 e8 · t2 e10 | 오프라인 (쥔 물체 위치 vs target 추정) | — | |
| 8 | PLACED 미도달 확인 | `--placed-no-seed-certify` · `home` 이 실제로 쓰이는지 | CPU (기록) | — | |

**권고 [P]:** 남은 시간 (11-11 – 11-13) 으로는 GPU 를 쓰는 후보를 2–3 개만 볼 수 있다. **1 (T1) → 2 (정답 latch) → (2 가 좋으면) 3 (ST2)** 을 먼저, 4 는 시간이 남으면. 각 후보는 결과 전에 채택 규칙을 계획서에 적고 돌린다. 동결 판단 (D6) 은 사용자.

### 3.2 동결 (D6) 뒤

1. 동결 판 이름을 계획서 · `AG3S_FINAL_CONFIG_KO.md` 에 적고 commit.
2. AG3S GT oracle 판 = **동결 구성에서 인식만 GT** 로 바꾼 판을 등록 (`--oracle-seg` 계열 flag 확인; 예전 B5 를 쓰지 않는다).
3. ablation 4 행의 flag 확인 (`PAPER_REVISION…` §10.2 표).
4. 주말 queue: 같은 기계 · 같은 init (19–28) · 같은 seed. **서버 이전 (dev-f1, AMRC → SDM) 은 이 run 들이 끝난 뒤** (기계가 바뀌면 chunk 가 달라짐).

## 4. 열린 일 · 주의

| 항목 | 상태 | 할 일 |
|---|---|---|
| branch `wip/stage-track` push | commit 됨 (e8ddf8b7, 7a140dbc, 이 문서의 commit). **이 pod 에는 GitHub 자격증명이 없다** | 사용자가 `cd /mnt/dev/AG3S_VLA && git push origin wip/stage-track` |
| ST1 을 기본 branch 에 merge | 안 함 (flag 기본 꺼짐이라 merge 해도 동작 불변) | 동결 판에 ST 계열이 들어갈 때만, dev-4a 와 queue 사이에 |
| worktree `.claude/worktrees/stage-track` | 남겨 둠. submodule · venv · `.cache/openpi` 를 주 checkout 으로 symlink 함 (git status 에 `T` · `??` 로 보임 — commit 하지 말 것) | 쓰고 나면 `git worktree remove` (symlink 먼저 지움) |
| 공유 checkout 의 계획서 사본 | `doc/active/experiments/AG3S_PLAN_SCHEDULE_20261110_KO.md` 는 공유 checkout 에서 untracked, branch 에는 commit 됨 | 다음 commit 때 공유 checkout 쪽에도 추가 (dev-4a 와 합의) |
| CPU 작업 | `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`, 세션당 2 core | 11-10 P4 가 다중 thread BLAS 로 dev-4a run 을 3–4 배 느리게 했음 |
| GPU | `scripts/setup/gpu_lock.py` 로만, 바깥 flock 금지. closed-loop launcher 는 run 마다 독점 lock 을 스스로 잡는다 | queue 가 도는 동안 그 checkout 의 HEAD 를 바꾸지 않는다 |

## 5. 세션

| 세션 | 하는 일 (11-11 03:xx) |
|---|---|
| dev-4a | baseline tranche queue F (multi-link detector · LIBERO multi-link · LS π0.5 tranche C) — baseline 보고 run 은 보류 (원문 보고값 비교로 변경) |
| dev-e1 | 원고 수정안 (`paper/notes/PAPER_REVISION_REPORTED_COMPARISON_20261111_KO.md`), 계획서 갱신 |
| dev-f1 | AMRC → SDM 서버 이전 준비 (GPU 안 씀; `/mnt/dev/_migration/README_KO.md`) |
| dev-55 (이 세션) | 제약 단계 추종 · 계획서 · 이 인계 — 다음 세션이 이어받음 |
