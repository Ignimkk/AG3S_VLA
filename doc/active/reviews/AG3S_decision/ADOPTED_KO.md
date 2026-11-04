# 채택한 것 (핵심 평가 전)

작성: 2026-10-28 21:15 KST · 기준: `feature/ag3s-development`, HEAD `b3b69311`. 작업 트리에 commit 하지 않은 변경이 있다.

- 이 문서는 주제별로 이어 쓴다. 항목마다 다음을 적는다:
  - 무엇을 채택했나 · 주요 기능 (서버 flag)
  - 채택 근거 실험
  - 한계
  - 출처
- 수치는 출처 기록의 값을 옮긴 것이다. 근거 실험은 장면 9 개 (가림 3 · 회귀 6) 를 2–5 번 돌린 규모다. 성능 주장이 아니라 핵심 평가에 넣을 구성을 정하는 개발 판단이다.
- 채택하지 않은 것: [NOT_ADOPTED_KO.md](NOT_ADOPTED_KO.md). 남은 문제와 필요한 실험: [OPEN_ISSUES_KO.md](OPEN_ISSUES_KO.md).
- 진행 기록: [AG3S-PRE-CORE-TO task](../../../../AG3S/ag3s/docs/handoff/AG3S-PRE-CORE-TO.task.md), 수치 `AG3S-PRE-CORE-TO.verify.json` `numbers.*`.

## 요약 — 핵심 평가 기준 구성

| # | 채택 | 모듈 | 서버 flag | 상태 |
|---|---|---|---|---|
| A1 | BLAS thread 1 | CCR (TrajOpt 시간) | `--blas-threads 1` (기본값) | 채택 · 검증 끝 |
| A2 | GPU 여러 초기값 + 충돌 검사 줄이기 (**A8 까지만**. A9 · P0m 에서는 뺌 — chunk 마다 표준 SQP 하나 E0, T74 Q1) | CCR | `--to-gpu-seeds curobo` + 검사 줄이기 (E3 kit). A9 · P0m 에서는 금지 | A8 에서 채택 · 검증 끝 → **A9 에서 대체** (2026-10-30, T74 Q1) |
| A3 | 손바닥 껍질 구 + 껍질 행 여유 5 mm | LTE (로봇 충돌 구) | `--palm-shell on --palm-shell-margin 5` | 채택 · 검증 끝 |
| A4 | 쥐기 목표 항 (닫는 행만) | CCR (목적함수) | `--grasp-anchor 2000 --grasp-anchor-path-weight 0` | 채택 (사용자 2026-10-28) |
| A5 | 쥐기 때 꼭지 덮개 | LTE (target 제외) | `--target-stem-cap {in_ball,full}` | 채택 (사용자 2026-10-28) · **closed-loop 검증 전** |
| A6 | 쥐기 때 가벼운 이웃 접촉 허용 | CCR · LTE (쥐기 자세) | 없음 | 채택 (사용자 2026-10-27 · 2026-10-28) · **구현 전** |
| A7 | 로봇 지우기 = 구 mask 만 (렌더 로봇 mask 끔) | ATS 입력 (ESDF 로봇 자기 제거) | `--robot-mask render` 를 넣지 않음 | **최종 채택** (사용자 2026-10-28 22:05) |

| **A8** | **논문 방법론 = A1–A4 + A7 (최종 구성)** | 전체 | 아래 A8 절의 서버 flag 전부 | **채택** (사용자 2026-10-29 01:05, 핵심 평가 1차 근거) |
| **A9** | **수정 최종 구성 (benchmark 로 보낼 구성) — 모듈별 전체 방법** | ATS · HSR · LTE · CCR 전체 | 아래 A9 절 | **P0m 채택 (사용자 2026-11-03 18:2x KST)** = P0 + `--subtask-from-latch`. 최종 확정은 T92b (seed 반복 + hull · 접촉 여유 짝 비교). benchmark 로 보냄 (main `a0b16d0a` → benchmark merge `93e2ae8d`) |

A5 · A6 은 사용자가 채택을 정했지만 아직 closed-loop 근거가 없다. 남은 일은 [OPEN_ISSUES_KO.md](OPEN_ISSUES_KO.md) O1 · O2 에 있다.

## A1. BLAS thread 1

- **기능:** 서버가 BLAS thread 를 1 로 고정한다 (`AG3S/trajopt/blas_threads.py`, `--blas-threads N`, 0 = library 기본).
- **실험:** 장면 1345 서버 smoke 를 thread 기본값과 1 로 비교했다. 같은 거리장으로 18 장면 683 chunk 를 replay 했다.
- **결과:**
  - 로봇 명령 32/32 bit 동일.
  - 서버 TrajOpt 풀이 중앙값 135.8 → 36.3 ms (−73 %).
  - replay 683/683 동일, 119.8 → 43.4 ms.
- **한계:** 없음. 명령이 바뀌지 않는 시간 설정이다.
- **출처:** `numbers.P0d`, 그림 `AG3S/ag3s/docs/figures/pre-core-to/p0-openblas-threads`. 판정 2026-10-25 ~23:30.

## A2. GPU 여러 초기값 + 충돌 검사 줄이기 (A8 까지의 serving. A9 · P0m 에서는 뺌)

> **2026-11-04 정정:** 이 절은 A8 시점 기록이다. A9 · P0m 은 `--to-multi-seed` · `--to-gpu-seeds` 를 빼고 chunk 마다 표준 SQP 하나 (E0) 로 푼다 (T74 Q1, 아래 A9 표). 옛 제목 "(= 지금 serving)" 은 A8 시점 표현이다.

- **기능:**
  - 위반이 남은 chunk 에서 cuRobo 로 초기값 16 개를 GPU 에서 한 번에 풀고, 가장 나은 것을 CPU SQP 의 시작점으로 쓴다. VLA 추종 항을 포함한다.
  - 후보 검사 범위를 줄여 시간 기준을 맞췄다.
- **실험:**
  - offline: 가림 장면 위반 chunk 68 개에서 비교했다.
    - CPU 여러 초기값: 68/68 해소, 4,197 ms.
    - GPU 초기값 + 검사 전부: 68/68 해소, 544.5 ms. 시간 기준 533 ms 미통과.
    - GPU 초기값 + 검사 줄이기: 68/68 + 회귀 2/2 해소, 302.3 ms. 통과. 궤적은 검사 전부와 동일.
  - closed-loop: 가림 18 + 회귀 18.
- **결과:**
  - 가림 금지 접촉 없는 성공 6/18 (이전과 같음). 금지 접촉 run 3 → 1.
  - 회귀 14/18 그대로, 금지 접촉 1.
- **한계:** 가림 장면 성공을 올리지 못했다. 같은 장면 5 번 흔들림에서 가림 4/15 (VLA 단독 15/15). 원인은 연속성 항이 쥐기 전에 손을 미는 것과 손바닥 접촉이다. A3 · A4 로 보완한다.
- **출처:** `numbers.E3_offline` · `E3_closed_occ` · `E3_closed_reg` · `PT`. 결과 폴더:
  - `outputs/2026-10-26_사과1345_가림장애물18_GPU여러초기값TO_closedloop/`
  - `outputs/2026-10-26_장애물없는회귀18장면_GPU여러초기값TO_closedloop/`

## A3. 손바닥 껍질 구 + 껍질 행 여유 5 mm

- **기능:** 손바닥 mesh 겉면에 접하는 구를 겉면 밖으로 5 mm 넘치게 둔다. 껍질 행만 비타겟 여유를 10 → 5 mm 로 줄인다. 거리장은 바꾸지 않는다 (로봇 모델만 바뀜).
- 상세 결정 · 이유 · 한계: [AG3S_COLLISION_MODEL_DECISION_20261028_KO.md](AG3S_COLLISION_MODEL_DECISION_20261028_KO.md) §1 · §2.
- **이번 세션에 더해진 근거 — 기둥 두 개 장면 (장면 1345, 큰 기둥 20 cm + 작은 기둥 10 cm, seed 3 개):**
  - 껍질 구가 없는 구성 (VLA 단독 · 지금 serving · 목표 항) 9 run: 모두 손바닥 몸체 ↔ 작은 기둥 접촉.
  - 껍질 구 3 run: 접촉 0. 같은 요청 · 같은 자세에서 TrajOpt 가 보는 최소 여유가 27.2 → 8.9 mm 가 되어 비켜 갔다.
  - 그러나 3 run 모두 쥐지 못했다. 기둥이 손바닥이 들어갈 자리를 막는다 → A6 이 필요한 경우다.
- **출처:** 그림 `AG3S/ag3s/docs/figures/pre-core-to/hp-two-post-scene`. 결과 폴더 `outputs/2026-10-28_기둥두개장면_손바닥접촉_4구성x3_closedloop/`.

## A4. 쥐기 목표 항 (닫는 행만)

- **기능:**
  - VLA 가 처음 닫기를 계획한 순간의 손바닥 위치 g 를 얼린다.
  - 그 뒤 chunk 의 닫는 행부터 3 행에서 손바닥을 g 로 당기는 비용 (무게 2000) 을 TrajOpt 목적함수에 더한다.
  - 닫는 행이 최적화 구간 안에 있을 때만 작동한다. 작동하지 않으면 궤적이 지금 serving 과 bit 동일하다.
- **왜 필요했나 (cost 분해):**
  - 쥐기 전 요청에서 연속성 · 꺾임 항과 기둥 충돌 벌점이 손을 사과에서 13–23 mm 밀어낸다.
  - 목적함수에 손을 되돌리는 항이 없었다.
- **실험:** 지금 serving 이 5 번 중 2 번 이하 성공한 6 장면 (가림 121–123, 회귀 850 · 954 · 211) × 2 번 = 12 짝. 같은 장면 · 같은 흔들림끼리 비교했다.
- **결과:**
  - 성공 2 → 5 (좋아짐 4, 나빠짐 1: 회귀 850 흔들림 1).
  - 작동한 가림 run 4/4 성공. 손바닥이 얼린 목표에서 14–23 mm 벗어나 있던 것이 1–3 mm 로 줄었다.
- **한계:**
  - 켜지는 조건이 좁다. 닫는 행이 구간 밖이면 효과 0 이다 (가림 122 흔들림 없음, 기둥 두 개 장면 3/3).
  - 2 번 반복이라 구성 효과는 판정하지 않았다. 핵심 평가에서 확인한다.
  - 넓힌 버전은 채택하지 않았다 ([NOT_ADOPTED_KO.md](NOT_ADOPTED_KO.md) §2 N13).
- **출처:** `numbers.E3G_grasp_goal` · `COST`. 그림 `cost-*-E3G`. 결과 폴더:
  - `outputs/2026-10-28_쥐기목표항_지금serving_공통실패6장면x2_closedloop_cost분해/`
  - `outputs/2026-10-27_TrajOpt_cost분해_지금serving_가림121-123/`

## A5. 쥐기 때 꼭지 덮개 (stem cap)

- **기능:** target 공 (ball) 위쪽 꼭지 영역의 voxel 을 target 없는 거리장에서 지운다. 손바닥이 사과 꼭지 · 잎 (시뮬레이터에서는 충돌 없는 visual) 의 TSDF 에 막히지 않게 하는 것이다.
  - `in_ball`: 공 안의 꼭지 영역만 지운다.
  - `full`: 공 위로 튀어나온 꼭지까지 지운다.
- **실험 (offline, 13 기록 · 39 요청):**
  - 쥐는 손이 새로 위반하는 행: 1,189 → 894 (`in_ball`) / 400 (`full`).
  - 지운 voxel 중 금지 class: `in_ball` 0, `full` 로봇 2.
  - 구현 규칙은 offline what-if 와 voxel 단위로 같다 (369/369, 362/362).
- **한계:** closed-loop 을 아직 돌리지 않았다. 어느 방식을 기본으로 할지 정해지지 않았다 → [OPEN_ISSUES_KO.md](OPEN_ISSUES_KO.md) O1.
- **출처:** `numbers.STEMCAP` · `LTE_conn_amb`, `AG3S/ag3s/docs/handoff/AG3S-PRE-CORE-TO-STEMCAP.impl.md`. 결과 폴더 `outputs/2026-10-27_사과꼭지_TSDF지우기_실제씬확인/`.

## A6. 쥐기 때 가벼운 이웃 접촉 허용

- **결정:** 이웃 물체 · 장애물과 닿지 않고 쥘 자세가 없으면, 쥐는 순간에 한해 가벼운 이웃 접촉을 허용한다. 사용자 판정 2026-10-27 15:50, 2026-10-28 재확인.
- **근거 실험:**
  - 충돌 고려 쥐기 자세 offline 70 run: 회귀 954 · 211 의 이웃 접촉은 모든 구성에서 쥐는 순간에 시작한다.
  - 기둥 두 개 장면: 껍질 구로 접촉을 0 으로 막으면 3/3 쥐지 못한다.
- **한계:** 구현 전이다. 허용 범위 (거리 · 부위 · 단계) 를 정해야 한다. 평가 정의 (금지 접촉) 는 바꾸지 않는다 → [OPEN_ISSUES_KO.md](OPEN_ISSUES_KO.md) O2.
- **출처:** `numbers.GIK` · `GIK_diag`, 결과 폴더 `outputs/2026-10-27_충돌고려쥐기자세_offline/` · `outputs/2026-10-28_기둥두개장면_손바닥접촉_4구성x3_closedloop/`.

## A7. 로봇 지우기 = 구 mask 만 (렌더 로봇 mask 끔)

- **결정:** depth 에서 로봇을 지울 때 로봇 구 mask 만 쓴다. 렌더 로봇 mask (`--robot-mask render`) 는 핵심 평가를 포함해 켜지 않는다. 사용자 판정 2026-10-28 21:42, 22:05 최종 확정.
- **실험:** 렌더 mask 를 켜고 끈 직접 비교 두 가지.
  - 껍질 구 위 9 장면 × 5 번: 가림 11 → 10/15, 회귀 14 → 14/30. 장면별 차이 ≥ 4 없음.
  - 지금 serving 위 단독, 6 장면 × 2 번: 12 짝 중 7 짝이 궤적 bit 동일, 성공 2 → 3. 거리장 차이는 몇 voxel, ATS target 점은 같음.
- **이유:** 거리장 · ATS · 성공을 바꾸지 못한다. 렌더에 요청마다 중앙 77 ms 가 더 들고, 실물 로봇은 손목 bracket · gripper 가 custom 이라 정확한 mesh · calibration 이 어렵다.
- **이전 문장 정리:** 팔 형상 STEP (AG3S-ARM-SHAPE) 의 "render 로 고정" 은 팔 형상을 비교하려고 묶어 둔 실험 조건이었다 (사용자의 2026-10-27 시험 허용 근거). 이 결정으로 대체됐다.
- **한계:** 팔 내접 덮개 closed-loop 은 렌더를 켠 구성끼리 비교했다. 렌더를 끈 구성에 팔 덮개를 더한 closed-loop 은 아직 없다 → [OPEN_ISSUES_KO.md](OPEN_ISSUES_KO.md) O7.
- **출처:** [NOT_ADOPTED_KO.md](NOT_ADOPTED_KO.md) N12, `numbers.SCM_SCMR_5draw` · `E3R_render_only`, `$RUN/verify/AG3S-PRE-CORE-TO/PT/timing_E3R.json`.

## A8. 논문 방법론 (최종 구성)

- **결정:** 지금 AG3S 를 논문 방법론으로 채택한다 (사용자 판정 2026-10-29 01:05 KST).
  - 구성 = A1–A4 + A7.
  - 근거 = 핵심 평가 1차: 가림 기둥 3 장면 + 장애물 없는 회귀 6 장면 × 2 번 = 18 run, 모듈별 지표.
- **결과 위치:**
  - 영상 · 그림 · README: `outputs/2026-10-28_핵심평가_모듈별_가림기둥3_회귀6_x2/`
  - 수치: `AG3S/ag3s/docs/handoff/AG3S-CORE-EVALUATION.verify.json`
  - 그림: `AG3S/ag3s/docs/figures/core-eval/`, 논문용은 `core-eval/paper/`
  - 계획 · 판정: `AG3S-CORE-EVALUATION.task.md` §0.1
- **채택 근거 요약 (모듈별 주장):**
  - ATS: 가림 장면에서 head 에 보이는 사과가 25 % 여도 target 표면 정확도 0.89, 보이는 표면 72 % 를 덮음, 다른 물체 선택 0.
  - HSR: 단계 일치율 0.97, 쥠 인정은 두 손가락 접촉 뒤 1 요청.
  - LTE: 18 run 동안 장애물 · 탁자 · 바구니 voxel 을 하나도 지우지 않음.
  - CCR: 손바닥 수정 중앙 2–12 mm, 멈춤 832 요청 중 9, 쥐기 목표 항 16 run 에서 작동, 풀이 중앙 70 ms.
  - 전체: 로봇–탁자 · 바구니 접촉 VLA 5/12 → 0/12.

### 서버 구성 (기록 `REG_k0/runs/server_*/server_proc.txt` 에서 그대로 옮김)

```text
python -m AG3S.trajopt.serve_safe --config pi05_rby1_randomized_pick_place_16d_lora --checkpoint <16D 30k ckpt> --model-xml <snapshot>/robots/rby1/description/models/rby1a/mujoco/model_transport.xml
--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized
--esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 --obstacle-margin 0.010
--held-body-cover --closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60 --held-slip-detach --held-cover-cap-mm 10
--hold-follow-to 3 --finger-cover both --plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4 --w-deflection-rate 20
--to-fast-eval --esdf-host-mirror --target-exclusion ball-owned --owned-max-dist-voxels 3 --attention-split-radius-mm 12
--attention-split-ratio-source raw --target-permission-strict --attention-source noun_target:L10H7 --attention-split-partition
--split-supersedes-anchor --wrist-track-scale 0.1 --neighbour-contact-mm 20 --nontarget-evidence surface --placed-no-seed-certify
--kept-geometry-follow --roi-carve --roi-carve-footprint --roi-carve-footprint-pad-px 0 --roi-carve-footprint-robot ignore
--roi-carve-device cuda --blas-threads 1 --to-multi-seed 16 --to-gpu-seeds curobo --to-gpu-seeds-check first-pass
--palm-shell on --palm-shell-margin 5 --grasp-anchor 2000 --grasp-anchor-path-weight 0
```

- client (`integration/rby1/bringup/pi05_infer.py`):
  - `--safe-remote --safe-timeout 120 --safe-phase approach --safe-manipulators <쥐는 손> --safe-gate reasons --safe-hold-mode measured`
  - `--obstacle-stop-distance 0 --stop-after-place 2.5`
- 넣지 않은 것:
  - `--robot-mask render` (N12)
  - `--target-stem-cap` (A5, 검증 전)
  - 가벼운 이웃 접촉 허용 쥐기 (A6, 구현 전)
  - `--arm-spheres` (O7, 판정 대기)
  - `--grasp-goal-window widen` (N13)

### 코드 상태 (benchmark checkout 으로 옮길 때 기준)

- 평가에 쓴 코드는 commit 되지 않은 작업 트리다.
  - `snapshot_core` (storage run `20261028T224856+0900_b3b69311f948_8784a1265958`) = HEAD `b3b69311` + 작업 트리.
  - HEAD 와 다른 파일 · md5 목록: 그 run 의 `snapshot_core_head.txt`.
- 2026-10-29 HEAD 는 `240e6a02` 다 (세션 B 의 T64 · T65 분석 도구만 더해짐, 서빙 코드 변화 없음).
- 옮기기 전에 사용자 요청으로 채택 코드를 먼저 commit 해야 한다 (O9).
- 주의 (CLAUDE.md): 개선 branch 전체를 benchmark branch 에 merge 하면 V11AB 평가 대상이 바뀐다. 최종 방법론은 V11AB 와 나란히 쓸 수 있는 별도 구성 (위 flag 묶음) 으로 옮기는 것을 권장한다.

### 아직 바뀔 수 있는 점 (진행 중, 2026-10-29)

- **target attention 신호** (`--attention-source`): 3-way ablation (T65 고정 입력 재생 + closed-loop 36 run 안전 확인) 결과로만 바뀐다.
  - 바꾸는 조건은 `AG3S-CORE-EVALUATION.task.md` §0.1-2.
  - 조건을 못 넘으면 명사 (`noun_target:L10H7`) 유지.
- **놓는 곳 허들 장면** (§0.2): 구성을 바꾸지 않는 평가다. VLA 가 부딪히는 조건에서 보호를 보이는 결과다.
- **알려진 한계** (논문 한계 절):
  - 바나나가 사과에 10 mm 안으로 붙은 장면의 이웃 접촉 (VLA 단독과 같은 접촉).
  - target 공이 사과보다 커서 붙은 이웃 표면 일부를 지움.
  - 요청 왕복 732 ms > 실행 창 533 ms.

### 다른 세션의 채택 항목과의 관계 (2026-10-29 확인)

이 폴더의 다른 결정 문서에도 채택 항목이 있다. A8 (평가한 최종 구성) 에 들어갔는지는 아래와 같다.

| 문서 · 항목 | 내용 | A8 에 포함? |
|---|---|---|
| [동적 거리장](AG3S_DYNAMIC_FIELD_DECISION_20261029_KO.md) DA1 | 관심 영역 안의 보이는 빈 공간 지우기 (GPU) | **포함** (`--roi-carve …` flag) |
| [기하 grasp 보정](AG3S_GEO_GRASP_DECISION_20261029_KO.md) G1 | VLA 닫기 근처 보정 (`--geo-grasp …`), 별도 worktree `feature/geo-grasp`, 단독으로만 채택 | **최종 방법론에 채택 (사용자 2026-10-29).** 코드는 아직 worktree 에만 있다 → 지금 작업 트리로 합치고, A1–A4 + A7 + G1 결합 시험을 한 뒤 A8 flag 에 넣는다 (아래 "G1 통합") |
| [target 찾기 · 쥐기 판단 (B)](AG3S_IDENTITY_PHASE_SOURCE_DECISION_20261029_KO.md) D1 | 결과 전에 고정한 attention 선택 절차 (선택 = `mean:L10H7`) | **판정 중.** 3-way (T65 고정 입력 재생 + closed-loop 36 run) 결과로 정한다 |
| (위 D1 결과) | T65 고정 입력 재생 + closed-loop 3-way (2026-10-29) | **명사 신호 (`noun_target:L10H7`) 유지 확정.** 두 대안 모두 사전 고정 규칙 미통과. `AG3S-CORE-EVALUATION.task.md` §0.1-2 |
| 같은 문서 D2 | attach reach 기준점 결함 수정 대상 | 미포함 (수정 방식 미정) |
| 같은 문서 D3 | 종료 구간 보조 평가 규칙 | 평가 규칙 (구성 아님) |

### 이웃 접촉 허용 — 지금 상태와 켜는 규칙 (사용자 지시 2026-10-29)

- **지금 켜진 것:** `--neighbour-contact-mm 20` (IFG-NBC).
  - TrajOpt 가 쥐는 손가락이 target 120 mm 안의 이웃 장애물과 20 mm 까지 겹치는 것을 허용한다.
  - A8 flag 에 들어 있다.
- **지금 꺼진 것:** A6 "쥐기 때 가벼운 이웃 접촉 허용" (쥐기 자세를 고를 때 이웃 접촉을 허용하는 기능).
  - 아직 구현되지 않았다 → [OPEN_ISSUES_KO.md](OPEN_ISSUES_KO.md) O2.
- **켜는 규칙 (사용자):** A6 이 꺼져 있어 작업 성공률이 낮아지거나 불안정해지면, 사용자 판정으로 켤 수 있게 한다.
  - lead 는 평가 (허들 장면 · benchmark 비교 포함) 마다 실패를 원인 층으로 나눈다.
  - "이웃 물체 때문에 쥘 자리가 없어 쥐지 못함 · 반복 시도" 로 분류되는 실패가 나오면, 실제 씬 · 수치와 함께 사용자에게 켤지를 묻는다.
  - 켜면 O2 의 설계 (허용 거리 · 부위 · 단계 · 대상) 를 결과 전에 고정하고 기본 꺼짐 flag 로 구현한다.
  - 평가 정의 (금지 접촉) 는 바꾸지 않는다.

### G1 통합 (사용자 판정 2026-10-29: 기하 grasp 보정을 최종 방법론에 채택)

- **flag:** `--geo-grasp --geo-goal-mode correct --geo-track-scale 0.25 --geo-continuity-off` (기본 꺼짐).
- **코드 상태:** worktree `.claude/worktrees/geo-grasp` (branch `feature/geo-grasp`, HEAD `b3b69311` 위 commit 안 됨) 에만 있다.
  - 지금 작업 트리와 같은 파일 (`serve_safe.py` · `sqp.py` · `safe_policy.py` 등 12 개) 을 서로 다르게 고쳤다. 합치기가 필요하다.
- **결합 시험이 필요한 이유:** G1 결정 문서의 한계.
  - 껍질 구 (A3) 와 함께 돌린 run 이 없다.
  - 쥐기 목표 항 (A4) 도 같은 닫는 행에서 손을 당긴다.
  - 다른 기능과 함께 쓰면 보정이 0 회 켜진 적이 있다.
- **통합 순서:**
  1. 코드 합치기 (기본 꺼짐 유지, test).
  2. A1–A4 + A7 + G1 로 같은 9 장면 × 2 closed-loop. 확인: 보정이 켜진 요청 수 > 0, A4 와 당기는 방향, 금지 접촉.
  3. 통과하면 A8 flag 에 넣는다.
- **G1 · A4 함께 켜기 (사용자 판정 2026-10-29 02:50):** 대체 방식.
  - 기하 후보가 있는 chunk 는 G1 이, 없는 chunk 는 A4 가 닫는 행 목표를 채운다 (같은 slot).
  - 설명 · 근거: `AG3S-CORE-EVAL-G1.task.md` §4.
- **G1 코드 합치기 완료 (2026-10-29 ~03:10 KST):** `AG3S/ag3s/docs/handoff/AG3S-CORE-EVAL-G1.impl.md`.
  - 메인 작업 트리에 G1 + 떼어낼 수 없는 G2–G6 분기를 넣었다 (모두 기본 꺼짐). G7 (approach_guide) 은 걷어냈다.
  - 결합 판정은 (a) 대체 방식이다.
  - 검증:
    - test 2,079 통과 (알려진 기존 실패 1).
    - flag 끔 bit 동일 106/106.
    - worktree 동등 450/450.
    - A8 + G1 서버 CPU 기동 통과.
  - 다음: A8 + G1 / A8 + G1 + HSR6 결합 closed-loop.

## A8 최종 구성 확정 (사용자 판정 2026-10-29 ~11:30 KST)

근거: 결합 시험 `outputs/2026-10-29_결합시험_A8대기하보정대HSR6_9장면x2_closedloop/` · 장면 119 진단 `outputs/2026-10-29_장면119_손바닥이웃오렌지_쥐기실패_진단/` · 세션 B T66 `outputs/2026-10-29_HSR6_쥔직후_궤적급변_원인/`.

1. **최종 구성 = A8 + 기하 grasp 보정 (G1) + HSR 6 단계 (①–④) + 이웃 접촉 허용 쥐기 (A6).** 기하 보정은 포함한다 (사용자).
2. **HSR 규칙 ② 수정 (세션 B T66 제안 수용):** 쥔 물체 들어올림 허용을 거두는 것은 HSR6 이 approach · retreat 일 때만이다. latch 가 HELD 면 HSR6 이 grasp 여도 허용을 유지한다. 이유: 쥔 직후 HSR6 가 아직 grasp 라 1–3 요청 동안 허용이 거두어지고, 쥔 사과 구가 탁자와 겹쳐 TrajOpt 가 VLA 와 6–10° 다른 궤적을 냈다 (거둔 요청 14 개 수정 중앙 6.2°, 허용 156 개 0.76°).
3. **이웃 접촉 허용 쥐기 (A6) 를 켠다.** 쥐는 순간만 허용한다 (사용자). 허용 범위는 결과 전에 `AG3S-CORE-EVAL-A6.task.md` 에 고정하고 기본 꺼짐 flag 로 구현한다. 근거: 장면 119 에서 손바닥 껍질 구 여유 5 mm 가 22 mm 떨어진 오렌지를 피하게 해, 손바닥이 VLA 보다 6–10 mm 비켜나고 사과가 손가락 사이 중심에서 21–28 mm 벗어나 쥐지 못했다 (실제 오렌지 접촉 0).
4. 다음: ② 수정 · A6 구현 → 최종 구성 확인 closed-loop → 놓는 곳 허들 장면 AG3S 14 run.
5. **최종 구성 결과 (2026-10-29 13:15 KST, 코드 `snapshot_final2` = HEAD bb85d9f8 + A6 · ② 수정):**
   - 9 장면 × 2: 금지 접촉 없는 성공 14 / 18, 금지 접촉 0. A8 1 차는 12 / 18 · 3 이다.
     - `outputs/2026-10-29_최종구성_A8+기하보정+HSR6+이웃손바닥허용_9장면x2_closedloop/`
   - 바구니 위 허들 7 장면 × 2: VLA 단독 0 / 14 → AG3S 6 / 14, 허들 접촉 12 → 2.
     - `outputs/2026-10-29_바구니위허들7장면_VLA단독대AG3S최종구성_closedloop/`
   - **주의 — 기하 보정 · 쥐기 목표 항 (A4) 은 이 구성에서 0 요청이다.**
     - HSR ④ 가 닫는 순간 목표를 HSR6 = grasp 일 때만 켜는데, 닫기 전 HSR6 는 approach 다.
     - 그래서 측정된 성공은 기하 보정 · A4 없이 나온 것이다. 논문에 이 두 항의 기여를 주장하지 않는다.
6. **최종 구성 확정 (사용자 판정 2026-10-29 ~13:40 KST): 측정한 그대로 확정한다.**
   - 서버 인자는 `AG3S-CORE-EVAL-A6.impl.md` §3 의 한 줄 그대로다. 코드는 commit `cc0d527f` (= `snapshot_final2`).
   - 기하 보정 · 쥐기 목표 항 flag 는 남기되 ④ 때문에 작동하지 않는다. 논문에서 그 기여를 주장하지 않는다.
   - 이 구성으로 main → benchmark branch 에 보내고 benchmark · baseline 비교를 한다. 방식: merge --no-ff 2 단계 (사용자).

## A9. 수정 최종 구성 — benchmark 로 보낼 구성 (모듈별 전체 방법)

> **갱신 (2026-11-03 18:3x KST, 사용자 판정): 최종 구성 = P0m = P0 + `--subtask-from-latch`.** 아래 A9.1 · A9.2 의 2026-10-31 계획 (HSR 5 단계 분류기 채택, 규칙 ①–④ 재연결, `--nontarget-evidence surface`, 표면 분할, G1) 은 P0 · P0m 에서 바뀌었다. 바뀐 줄은 각 행에 **[P0m]** 으로 적었다. 전체 flag · 명령줄 정본: `AG3S/ag3s/docs/handoff/P0_FINAL_CONFIG_FOR_BENCHMARK_20261103_KO.md`. 최종 확정은 T92b (seed 반복 + hull 채우기 · 접촉 여유 짝 비교, `T92b.task.md`) 로 한다.
> 코드 기준: development HEAD `f964439d` (코드 `63e98cc8`), main `32a45fb8`. 모든 새 flag 기본 꺼짐.

- **왜 이 절:** 지금까지 이 문서는 새로 채택한 항목만 적었다. 그래서 최종 구성에 무엇이 들어 있고 무엇이 빠졌는지 한 곳에서 보이지 않았다 (예: 논문 방법의 depth 연속 표면 추출이 최종 구성에 없다는 것을 놓쳤다 — 지금 target 제외는 공 방식). 이 절은 **AG3S 가 실제로 쓰는 방법 전부** 를 모듈별로 적는다. 방법 설명 (어떻게 동작하나 · 논문 설명과 다른 점) 은 [METHOD_STATUS_KO.md](METHOD_STATUS_KO.md).
- **기준:** A8 최종 구성 (`AG3S-CORE-EVAL-A6.impl.md` §3 서버 argv, 코드 `cc0d527f`) 에서 아래 "바뀜 · 추가" 를 적용한 것. 상태 열: 그대로 / 바뀜 (확정) / 추가 예정 (진행 중) / 빠짐.
- **보내는 시점 (사용자 2026-10-31):** ATS 두 hidden 실험 (시간 추적 T79 · target 찾기) 이 끝난 뒤 이 구성으로 benchmark checkout 에 보낸다.

### A9.1 모듈별 방법과 상태

**ATS — 목표 물체 찾기 · 추적**

| 방법 | 서버 flag | 상태 | 근거 |
|---|---|---|---|
| target 명사 attention (prompt 의 target 명사 token 행, layer 10 head 7) | `--attention-source noun_target:L10H7` | 그대로 | T57 · A8 |
| target cluster 선택 + attention 으로 붙은 덩어리 다시 가르기 (재군집 12 mm, raw attention 비율) · 쥐기 전 split 조각이 anchor 대체 | `--attention-split-radius-mm 12 --attention-split-ratio-source raw --attention-split-partition --split-supersedes-anchor` | 그대로 | T44 · IFG-SPLIT |
| 받침면 위로 뜬 후보 제외 (가장 낮은 점이 받침면 위 60 mm 넘으면) | `--supported-max-bottom-mm 60` | 그대로 | T43 |
| 잠근 target 정체 유지 · 닫는 동안 기하 고정 · 밀린 과일 따라가기 | `--latched-identity-hold --closing-keep-geometry --kept-geometry-follow` | 그대로 | T43 · IFG-FOLLOW |
| 놓은 뒤 home 에서 seed 없음 인증 | `--placed-no-seed-certify` | 그대로 | T46c |
| 로봇 지우기 = 로봇 구 mask (렌더 mask 안 씀) | (`--robot-mask` 없음) | 그대로 | A7 |
| **target 범위 = 공 + 소유 비교 (ball-owned)** | `--target-exclusion ball-owned --owned-max-dist-voxels 3 --nontarget-evidence surface` | **[P0m] 바뀜:** ball-owned 인자는 명령줄에 남지만 **무효** (표면 추출이 대신함). `--nontarget-evidence surface` (T46c 채택) 는 표면 추출로 대체 → **금지** | T44 · T46 · T78 |
| **depth · point 연속 표면 추출 (attention seed → 연속 표면) + 맞닿은 물체 attention 분할 + target 겹침 검사 + 기권 시 직전 영역 유지 + 쥔 뒤 고정 + 공 제거 + GPU** | `--target-extent attention-surface --surface-target-check overlap --surface-keep-last --surface-grasp-hold --surface-no-ball --attention-surface-device cuda` + 쥐기 직전 hull 채우기 `--surface-hull-fill` + 손 층 target 접촉 여유 `--surface-contact-allowance-mm 10` | **[P0m] 채택.** 맞닿은 물체 분할 (`--surface-split`) 은 **폐기** (사용자 2026-11-01). hull 채우기 · 접촉 여유는 T92b 에서 뺀 구성과 비교해 차이 ≤ 2 run 이면 뺀다 | T76 · T78 · T78b · T78f |
| 시간 추적: 밀리거나 가려진 target 을 VLM hidden patch 특징으로 다시 찾기 | (offline 실험, flag 없음) | **실험 중** (T79) | T75 회귀 211 |
| target 찾기: hidden patch 특징 | — | **다음 실험** | — |

**HSR — 작업 단계 · 쥠 상태**

| 방법 | 서버 flag | 상태 | 근거 |
|---|---|---|---|
| 3 단계 gate (pick · place · home) — 새 조작 대상 gate | P0: (기본 on) `kv_L4_v1` 분류기. **[P0m] `--subtask-from-latch`** | **[P0m] 바뀜:** label 을 학습 분류기 대신 grasp latch 에서 (searching · latched · closing → pick, held → place, placed → home). T92: 9 run 9 / 9, 놓기 전 막음 0. serving 의 학습 요소는 VLA 뿐 | T92 |
| 6 단계 분류기 HSR6 (layer 4 KV, 시연 라벨) + 규칙 ① 쥠 인정 보조 · ② 쥔 물체 허용 · ③ 물러남에서 target 권한 거둠 · ④ 쥐기 단계에서만 이웃 접촉 · 목표 항 | `--hsr6-attach-assist --hsr6-held-gate --hsr6-target-exempt --hsr6-objective` | **[P0m] 끔 (금지)** — 정답 단계를 넣어도 규칙이 해를 냄 (T85), 끈 P0 가 9 / 9 (T88) | T85 · T88 |
| 5 단계 분류기 (접근 · 쥐기 · 운반 · 놓기 · 물러남), VLM hidden layer 4, 물리 정답 라벨 | `--hsr5` (T82 구현) | **[P0m] 끔** — 2026-10-31 채택을 철회 (A9.2 참고). 단계 결정은 "최소 구성" (사용자 2026-11-03) | T77 · T82 · T92 |
| 손목 depth 단계 인식 (T91 · T91b) 을 gate label 출처로 | `--subtask-from-phase-machine` | **[P0m] 기본 꺼짐** — latch 대비 이점 없음 (T91c), 손목 카메라가 있는 benchmark 의 선택지 | T91c |
| 쥠 상태 latch (규칙: searching → latched → closing → held → placed), 닫힌 채 빠지면 놓음 | `--held-slip-detach` | 그대로 | T43 |

**LTE — 링크별 거리장 · target 제외**

| 방법 | 서버 flag | 상태 |
|---|---|---|
| cuRobo ESDF 3 계층 (거친 20 mm · 미세 5 mm · TSDF 5 mm), GPU 거리장을 요청마다 host 로 한 번 복사 | `--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --esdf-host-mirror` | 그대로 |
| 제약 링크 = 양손 손바닥 + 손가락 넷 (팔 · 몸통은 제약 밖) | `--links gripper --no-self-collision` | 그대로 |
| 쥐는 손 link 는 target 이 빠진 거리장을 봄 · target 예외는 권한 link 행에만 | `--target-field-policy exclude-authorized --target-permission-strict --esdf-margin 0` | 그대로 (지우는 범위는 ATS 의 target 범위를 따름 → 표면 추출로 바뀜 예정) |
| 로봇 구 모델: capsule 반지름 0.05 배 · 간격 0.4 · 최대 32 + 손가락별 몸통 구 + 손바닥 껍질 구 | `--capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 --finger-cover both --palm-shell on` | 그대로 |
| 여유: 장애물 10 mm, 손바닥 껍질 행 5 mm | `--obstacle-margin 0.010 --palm-shell-margin 5` | 그대로 (A3) |
| 쥔 물체 = 손의 일부 (몸통 전체 덮개, 반지름 상한 +10 mm) | `--held-body-cover --held-cover-cap-mm 10` | 그대로 |
| 이웃 가벼운 접촉 허용: 쥐기 전 손가락 20 mm · 쥐는 순간 손바닥 10 mm | `--neighbour-contact-mm 20 --neighbour-palm-contact-mm 10` | **[P0m] 그대로 (MuJoCo 데모 profile)**, 켜는 창 = latch · 손바닥–target 거리. **benchmark profile 은 둘 다 끄고 평가의 이웃 과일 면제도 끈다 — 접촉 허용은 target (쥘 때) · 목적지 (놓을 때) 만** (D1, 사용자 확인 2026-11-03) |
| 관심 영역 안 TSDF 를 이번 depth 로 비우기 (GPU, 가장자리 보존) | `--roi-carve --roi-carve-footprint --roi-carve-footprint-pad-px 0 --roi-carve-footprint-robot ignore --roi-carve-device cuda` | 그대로 |

**CCR — action chunk 수정 · 검증 (TrajOpt)**

| 방법 | 서버 flag | 상태 | 근거 |
|---|---|---|---|
| SQP + OSQP (정확도 1e-4), 계획 16 행 중 장애물 20 mm 안일 때만 뒤 8 행까지 풂 (look-ahead) | `--plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4` | 그대로 | T43 |
| 목적함수: VLA 추종 (손목 관절 0.1 배, 쥐기 전 target 150 mm 안) + 부드러움 | `--wrist-track-scale 0.1` (w_track 1 · w_smooth 0.05 기본) | 그대로 | IFG-TOW |
| **꺾임 변화율 (w 20) · 연속성 (w 0.5) 항: 쥐기 전에는 끄고, 쥔 뒤 (운반 · 놓기 · 돌아오기) 에만 켬** | `--w-deflection-rate 20 --deflection-rate-after-grasp --continuity-after-grasp` | **바뀜 (확정, T74 Q1)** | T74 |
| **E0 = chunk 마다 표준 SQP 풀이 하나 (여러 초기값 · GPU 초기값 batch 없음)** | `--to-multi-seed 16 --to-gpu-seeds curobo --to-gpu-seeds-check first-pass` **를 뺌** | **바뀜 (확정, T74 Q1)** | T74 |
| 쥐기 목표 항 A4 (닫는 행) | `--grasp-anchor 2000 --grasp-anchor-path-weight 0` **를 뺌** | **빠짐** (T68: 실제 작동 0, T74 에서 뺌) | T68 · T74 |
| 기하 grasp 보정 G1 | `--geo-grasp --geo-goal-mode correct --geo-track-scale 0.25 --geo-continuity-off` | **[P0m] 끔 (금지)** — 사용자 2026-11-02 "켜라고 할 때까지 끔". 쥐기 구간 강제 시 해로움 (T89) | T85 · T89 |
| 충돌만 남은 HOLD 대신 수정 chunk 실행 (인증 조건) | `--hold-follow-to 3` | 그대로 | T43 |
| 계산 옵션 (결과 bit 동일): 한 번씩 평가 · BLAS thread 1 | `--to-fast-eval --blas-threads 1` | 그대로 | T43 · A1 |
| grasp 후보 + 도달 제약 (T72b) | `--grasp-set-to …` | **채택 안 함** (T74 Q3 와 결과 같음) | T72b |

- **T74 Q1 결과 (세션 A, 10 run · 같은 seed):** 금지 접촉 없는 성공 최종 구성 4 / 10 → Q1 8 / 10, 금지 접촉 run 1 → 0, 탁자 · 바구니 접촉 0. 회귀 211 · 954 · 가림 122 · 허들 1345 가 성공으로 바뀜 (`outputs/2026-10-30_T74_E0TrajOpt_목적함수조합/`).
- **사용자 표현과 실제 동작:** "시작할 때는 꺾임 · 연속성을 끄고 place 할 때만 잠깐 켜기" — 실제 flag 는 latch 가 held 가 된 뒤 (들기 · 운반 · 놓기 · 돌아오기) 켠다. HSR 5 단계로 바꾸면 이 판정을 "운반 · 놓기" 단계에 묶을지 정해야 한다.

### A9.2 HSR — 5 단계 분류기 (VLM hidden layer 4 + 물리 정답 라벨, 사용자 판정 2026-10-31)

> **[P0m] 이 절의 "HSR 5 단계 채택" 은 철회되었다 (`--hsr5` 끔).** 단계 결정은 grasp latch 하나로 한다: 쥠 상태 (접촉 허용 창) 와 3 단계 gate label 모두 latch (T92, 사용자 2026-11-03 "단계 결정은 최소 구성"). 근거: T85 에서 정답 단계를 넣어도 단계 규칙이 해를 냈고, 규칙을 끈 P0 · P0m 이 9 / 9 였다. 아래는 2026-10-31 판정 당시의 기록으로 남긴다. 평가 정의의 "붙은 이웃 (사과 10 mm 안)" 조건은 D2 로 뺐다 (쥐는 순간 쥐는 손–이웃 과일 접촉은 거리와 상관없이 면제, 기둥 · 허들은 금지). 바닥 띠: 지지면 guard 5 mm (코드) 와 잔여 측정에서 빼는 6 mm (평가) 는 다른 값이다.

- **결정:** HSR 의 작업 단계 판단을 5 단계 (접근 · 쥐기 · 운반 · 놓기 · 물러남) 분류기 하나로 바꾸고, 입력을 VLM hidden layer 4 로 한다. ATS 두 실험 (시간 추적 · target 찾기) 을 마친 뒤 수정한 최종 구성으로 benchmark checkout 에 보낸다.
- **기존 방식 (최종 구성):**
  - 입력: VLM prefix pass 의 layer 4 V cache (head 1 개, [text 평균 | image 평균] 512 차원).
  - 분류기 두 개: 3 단계 (pick · place · home, `kv_L4_v1`, 새 조작 대상 gate) + 6 단계 (접근 · 쥐기 · 들기 · 운반 · 놓기 · 물러남, `hsr6_t64_kv_L4`, 규칙 ①–④). 쥠 상태는 따로 규칙 latch.
  - 라벨: 전문가 시연의 phase index. 시연에서는 닫음 = 쥠이라 "닫았지만 비었음" 예가 없다.
- **새 방식 (T77):**
  - 단계 수 = 제약 묶음이 갈리는 수 = 5 (들기 · 운반은 제약이 같아 합침). 3 단계 · 6 단계 두 분류기를 하나로.
  - 입력: VLM hidden layer 4 (block 4 출력 residual stream, 2048 차원) 의 카메라 3 대 image token 평균 + 마지막 유효 text token = 8192 차원. 같은 forward 에서 꺼내므로 추가 forward 없음.
  - 라벨: 전문가 시연 + **closed-loop 기록 427 run 의 MuJoCo 물리 정답** (실제로 들지 않았으면 운반 아님, 빈손 닫기 = 쥐기).
  - 분류기: 표준화 + 로지스틱 회귀 (기존과 같은 꼴). 층 · pooling · 규제 강도는 train 교차 검증만으로 고름.
- **결과 (학습에 쓰지 않은 장면 211 · 954 · 122 · 119 의 closed-loop 기록):**

| | 기존 (HSR6 → 5 단계) | layer 4 KV + 물리 정답 재학습 | **hidden layer 4 + 물리 정답 (채택)** |
|---|---|---|---|
| 빈손인데 운반 · 놓기 (위험한 오판) | 28 % | 1.5 % | 2.2 % |
| 5 단계 정확도 (test 전체 12,356 요청) | 0.47 | 0.64 | 0.72 |
| 정확도 — 성공 run (기존 기록 있는 44 run) | 0.69 | 0.90 | 0.92 |
| 쥐기 · 놓기 · 물러남 재현율 (성공 44 run) | 0.33 · 0.45 · 0.42 | 0.84 · 0.93 · 0.95 | 0.89 · 0.94 · 0.93 |
| 단계 전환 지연 중앙값 (성공 run, 요청) | 2–3 | 0 | 0–1 |
| 전문가 val 정확도 | — | 0.993 | 0.999 |

- **두 방법의 차이:**
  - 위험한 오판 (빈손 → 쥔 뒤) 을 없앤 것은 **물리 정답 라벨** 이다. 같은 layer 4 KV 로도 다시 배우면 28 % → 1.5 %.
  - hidden layer 4 는 그 위에 접근 ↔ 물러남 (쥐기 실패 뒤 멀어진 손) 과 접근 ↔ 쥐기 구분을 더 잘해 정확도가 3–8 %p 높다.
  - 미리 정한 "hidden 이 낫다" 기준 (거짓 운반률이 layer 4 KV 의 절반 이하) 은 미달이다 (0.263 대 0.333). 채택은 정확도 · 성공 run 결과를 보고 사용자가 정했다.
- **한계 · 남은 일:**
  - offline 평가만 했다. serving 구현 (VLA forward 에서 hidden layer 4 pooling, 5 단계 분류기 asset) 과 단계별 규칙 (3 단계 gate · HSR6 ①–④) 을 5 단계에 다시 잇는 일, closed-loop 확인이 남았다.
  - 분류기는 이 VLA checkpoint (RBY1 16D 30k) 의 hidden 공간과 이 시뮬레이터의 물리 정답에 묶여 있다. benchmark 에서 다른 checkpoint 를 쓰면 그 모델의 feature · 정답으로 다시 학습해야 한다.
  - 라벨 정의: closed-loop 의 쥐기 = 손바닥 60 mm 안, 놓기 = 사과가 바구니 바닥 위. 전문가 시연의 쥐기 (phase index) 와 정의가 다르다.
- **출처:** `AG3S/ag3s/docs/handoff/T77.task.md` · `T77.impl.md`, 결과 폴더 `outputs/2026-10-30_T77_HSR5단계_hidden_vs_kvL4/` · `outputs/2026-10-31_T77b_HSR5단계_성공run정확도/`, 원시 `$AG3S_STORAGE_ROOT/results/development/20261030T004052+0900_8ab3c7ef7757_1c61eaabd79c/verify/T77/` (`probe/t77_probe.json`, `success_runs_eval.json`).

### A9.3 benchmark 로 보내기 전에 남은 일

1. T78 (세션 A): 표면 추출 재생 평가 → closed-loop → 최종 구성 반영.
2. T79 → target 찾기 실험 (hidden). 채택 여부는 사용자 판정.
3. HSR 5 단계 serving 구현: VLA forward 에서 hidden layer 4 pooling, 5 단계 분류기 asset, 3 단계 gate · 규칙 ①–④ · 꺾임 변화율 / 연속성 켜는 시점을 5 단계에 다시 연결.
4. 위를 모두 켠 수정 최종 구성 closed-loop 확인 (최종 구성 · T74 Q1 과 같은 장면).
5. benchmark checkpoint 가 다르면 HSR 분류기 다시 학습 (A9.2 한계).
