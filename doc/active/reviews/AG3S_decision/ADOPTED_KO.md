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
| A2 | GPU 여러 초기값 + 충돌 검사 줄이기 (지금 serving) | CCR | `--to-gpu-seeds curobo` + 검사 줄이기 (E3 kit) | 채택 · 검증 끝 |
| A3 | 손바닥 껍질 구 + 껍질 행 여유 5 mm | LTE (로봇 충돌 구) | `--palm-shell on --palm-shell-margin 5` | 채택 · 검증 끝 |
| A4 | 쥐기 목표 항 (닫는 행만) | CCR (목적함수) | `--grasp-anchor 2000 --grasp-anchor-path-weight 0` | 채택 (사용자 2026-10-28) |
| A5 | 쥐기 때 꼭지 덮개 | LTE (target 제외) | `--target-stem-cap {in_ball,full}` | 채택 (사용자 2026-10-28) · **closed-loop 검증 전** |
| A6 | 쥐기 때 가벼운 이웃 접촉 허용 | CCR · LTE (쥐기 자세) | 없음 | 채택 (사용자 2026-10-27 · 2026-10-28) · **구현 전** |
| A7 | 로봇 지우기 = 구 mask 만 (렌더 로봇 mask 끔) | ATS 입력 (ESDF 로봇 자기 제거) | `--robot-mask render` 를 넣지 않음 | **최종 채택** (사용자 2026-10-28 22:05) |

| **A8** | **논문 방법론 = A1–A4 + A7 (최종 구성)** | 전체 | 아래 A8 절의 서버 flag 전부 | **채택** (사용자 2026-10-29 01:05, 핵심 평가 1차 근거) |

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

## A2. GPU 여러 초기값 + 충돌 검사 줄이기 (= 지금 serving)

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
