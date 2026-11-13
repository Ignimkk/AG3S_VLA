# SDM 전달: 동결 판 `pkgFrs` 의 SSR · TSR 개선 최종안 (2026-11-13)

받는 쪽: SDM 서버에서 `feature/panda-benchmark` 를 돌리는 세션. 작성: 로컬 PC 세션 (code 와 기록 문서, 그리고 SDM 에서 받은 분석 묶음만 봄. closed loop 은 돌리지 않음).

## 1. 요청

논문 표의 AG3S arm 을 돌리기 전에 아래 후보 세 개를 순서대로 거르고, 통과한 것을 합친 판 하나를 채택 검증해 주세요. 기준은 모두 결과 전에 여기 적은 대로 고정합니다. 기존 규칙 (flag 기본 꺼짐, 판은 `profile.py` 맨 끝에 추가, 결과 전 계획서 기록, queue 경계에서만 commit, GPU lock) 은 그대로입니다.

목표는 init 15–18 의 SDM 실측 112 ep 에서 SSR 48 → 59 (+10 %p) 입니다. 이 숫자가 나온다는 보장은 없습니다. 아래는 가장 근거가 큰 순서의 실험 계획입니다.

## 2. 근거 (받은 묶음에서 다시 센 것)

용어: '잃은 ep' = `pkgFrs` 는 task 실패이고 같은 episode 의 π0.5 단독은 task 성공인 것. '큰 보정' = action QP 의 `max_abs_delta_action` 이 1 이상인 call (action 범위 ±1).

| 발견 | 숫자 | 출처 |
|---|---|---|
| 잃은 ep 이 SSR 손실의 가장 큰 덩어리다 | 112 ep = 안전 성공 48 · 잃음 29 · 둘 다 task 실패 31 · 성공 + 충돌 4 | `ms_all.json`, `qp_delta_latch.csv` |
| 잃은 ep 은 큰 보정이 이어진 episode 다 | 큰 보정 5 call 이상: 잃은 29 중 26, 안전 성공 48 중 8. 144 ep 에서 5 call 이상이면 안전 성공 9 / 68, 미만이면 51 / 76 | `qp_delta_latch.csv` |
| 잃는 task 가 몰려 있다 | Spatial L-I task 1 은 4 / 4, Goal L-I task 0 은 2 / 2, Goal L-II task 1 은 2 / 2. Object L-II 는 16 ep 중 0 | `ms_all.json` |
| mug 를 빼면 거절의 절반은 진짜 장애물이다 | init 19 실패 20 ep 의 거절 529 call 중 task 물체 268, 그중 216 이 mug 4 ep. mug 를 뺀 299 call = 장애물 151 · 목적지와 구조물 76 · task 물체 52 · 그 밖 20 | `owner_tab10/` |
| 쥔 구간에서는 목적지가 쥔 물체를 막는다 | 쥔 구간 거절 301 call 중 쥔 물체 row 가 목적지 · 구조물에 51 (접시 36), task 물체에 26 | `owner_held/` |
| 계획서 §9.31 의 'target 오인식 10 ep' 은 첫 HELD 이전만 세면 7 ep 이다 | 7 = mug 4 · π0.5 단독도 실패 2 · 성공 + 충돌 1. 나머지 3 ep 은 놓은 뒤의 call 이 '쥐기 전' 으로 세어진 것 (놓은 뒤 target 이 옛 자리의 기하로 남음) | `tt_tab10/` 의 call 별 `grasp` |

큰 보정과 실패의 관계는 상관입니다. 원인인지는 후보 B 가 가릅니다.

## 3. 거르기에 쓰는 셋 (결과 전 고정)

episode 결과가 run 안의 위치에 따라 달라지므로 (commit `f1e662e8`), 기준 run 과 같은 cell · 같은 init 구성으로 돌립니다.

| 이름 | 구성 | 동결 판의 값 (TSR · CAR · SSR, > 50 mm) | 든 것 |
|---|---|---|---|
| S32 | Spatial L-I task 0–3 × init 15–18 (16 ep) + Goal L-I task 0–3 × init 17 · 18 (8 ep) + Goal L-II task 0–3 × init 17 · 18 (8 ep) | 13 · 23 · 9, 3 | 잃은 ep 13 개 (쥔 동안 큰 보정이 몰린 것 9 개 포함), 안전 성공 9 개 |
| M10 | Spatial L-I task 0 × init 5–14 (10 ep) | 9 · 8 · 7, init 10 은 안전 성공 | `-gh156` 이 장애물 mug 를 target 으로 잡아 252 mm 민 episode (init 10) |

S32 는 약 1 시간 20 분, M10 은 약 20 분입니다.

## 4. 후보

### A. `-gh156a` (이미 있음)

| 항목 | 내용 |
|---|---|
| 판 | `…-hb-qs-rs-gh156a-tro` |
| 이미 잰 것 | mug 6 ep 에서 task 성공 0 → 2, 작은 충돌 2 건 → 0 (`tab10sdm-gh156a-*` 대 `tab10sdm-frz-*`) |
| 남은 확인 | M10 |
| 통과 | init 10 에 > 50 mm 충돌이 없고, M10 의 SSR ≥ 7 |

### B. action QP 의 보정 크기 상한 (새 flag)

| 항목 | 내용 |
|---|---|
| flag | `--ee-action-qp-max-delta X` (기본 없음 = 지금과 bit 동일), 판 segment `-md5` (X = 0.5) |
| 구현 | `AG3S/trajopt/sqp.py` `_solve_ee_action` 의 상자: `lo_x = max(-1 - a_ref, -X)`, `hi_x = min(1 - a_ref, X)`. 고정 열 (회전) 처리는 그대로. 기록 `to.ee_action_qp.max_delta_cap` |
| 상한에 걸려 위반이 남을 때 | 새 rule 을 넣지 않습니다. 지금의 최종 검사와 verdict 그대로 |
| 가설 | 잃은 ep 은 연속된 큰 보정이 policy 를 궤적 밖으로 밀어서 생긴다. 상한을 두면 task 성공이 돌아온다 |
| 거르기 | S32 |
| 통과 | SSR ≥ 12 (동결 판 9), > 50 mm 충돌 ≤ 4, 동결 판의 안전 성공 9 개 중 잃는 것 ≤ 1 |
| 탈락의 뜻 | 충돌만 늘면 그 배치는 실제로 크게 비켜야 하는 것입니다. 이 방향은 닫고 논문의 한계로 적습니다 |

### C. 놓는 지점의 목적지 등록 (새 flag)

| 항목 | 내용 |
|---|---|
| flag | `--open-seed-destination` (기본 꺼짐), 판 segment `-os` |
| 구현 | `safe_policy.py` `_close_seed_request` 의 거울. latch 가 HELD 일 때 chunk 에서 gripper 열림 명령이 처음 나오는 row 의 쥔 물체 footprint 를 구하고, 그 아래 (footprint + 30 mm 띠) 의 성분을 그 call 에 목적지로 등록합니다. 등록 뒤의 처리 (목적지 margin 0, 쥔 물체 아래 row 면제) 는 지금 경로를 그대로 씁니다 |
| 근거 | close seed 가 이 구성에서 가장 효과가 컸습니다 (LIBERO 기제 확인 2 → 8 / 12). 놓는 쪽에는 같은 신호를 쓰는 code 가 없습니다. 목적지 등록은 지금 attention 1 위에 의존합니다 |
| `-hf30` 과의 차이 | `-hf30` 은 쥔 동안 항상 띠를 30 mm 로 넓혔고 탈락했습니다. 이 후보는 policy 가 손을 여는 call 에서만 작동합니다 |
| 거르기 | S32. A 가 통과했으면 mug 6 ep 도 (쥔 mug row 가 접시에 막히던 것, 계획서 §9.33) |
| 통과 | S32 의 SSR ≥ 11, > 50 mm 충돌 ≤ 4, 동결 판의 안전 성공 중 잃는 것 ≤ 1 |

## 5. 순서와 판정

| 단계 | 일 | GPU |
|---|---|---|
| 1 | A 의 M10 | 20 분 |
| 2 | B 구현 · 단위 test → S32 | 1 시간 20 분 |
| 3 | C 구현 · 단위 test → S32 | 1 시간 20 분 |
| 4 | 통과한 것을 합친 판 하나의 채택 검증: 112 ep (동결 판과 같은 run 구성) + guard (LIBERO 27 · SafeLIBERO 11 · LIBERO-Safety 10) + E1a 40 | 약 7 시간 |
| 5 | 채택된 판 (없으면 `pkgFrs`) 으로 논문 표의 AG3S arm 과 GT oracle arm | — |

채택 기준 (4 단계):

- 112 ep 에서 동결 판과 episode 짝으로 얻은 안전 성공 − 잃은 안전 성공 ≥ 4
- 어느 cell 도 안전 성공을 2 넘게 잃지 않음
- > 50 mm 충돌이 동결 판보다 1 넘게 늘지 않음
- guard 는 계획서 §9.2 그대로 (LIBERO 27 ≥ 22, LIBERO-Safety ≥ 4 · 위반 0, SafeLIBERO 11 의 > 50 mm 증가 ≤ 1)
- E1a 40 은 후퇴 확인 (SSR ≥ 19)

세 후보가 모두 탈락하면 `pkgFrs` 로 표를 돌립니다. 개선에 쓸 시간의 마감은 사용자가 정합니다.

## 6. GPU 없이 병행할 것 (CPU)

CPU 작업은 worker 2 개 이하, `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1` 로 돌립니다. 6.1 을 먼저, 6.2 를 그 뒤에 합니다.

### 6.1 잃은 29 ep 의 집계

잃은 29 ep (정의: `qp_delta_latch.csv` 에서 `src = pkgFrssdm`, `outcome` 이 F 또는 F+C, `pi05` 가 SS 또는 S+C) 에 대해 다음 표를 만들어 주세요. B 와 C 의 결과를 읽는 데 씁니다.

1. 큰 보정 call 을 쥐기 전 · 쥔 동안 · 놓은 뒤로 나눈 수. 지금 CSV 의 `latched` 는 쥐기 전과 놓은 뒤가 섞여 있습니다.
2. 큰 보정 call 의 최악 row 가 어느 link 이고 (`surface_owner_task.py`) 그 표면이 어느 물체인가.
3. 놓은 뒤 close seed 가 다시 나온 call 수와, 그때 인식된 target 이 task 물체인 call 수. 놓은 뒤의 낡은 target 기하가 다시 쥐기를 막는지 확인하는 값입니다.

### 6.2 obstacle confidence 의 offline 실험 (사용자 지시 2026-11-13: GPU 가 돌지 않는 일이므로 계획에 넣음)

**무엇을 답하나.** GT 없이 서버가 아는 값만으로 "이 표면은 진짜 장애물이다" 를 가를 수 있는가. 가를 수 있으면 장애물 row 의 여유 15 mm 를 그 confidence 에 비례하게 두는 후보 D 를 만듭니다. 이 단계는 code 를 바꾸지 않고 closed loop 도 돌리지 않습니다.

**배경.** 지금은 target · 받침 · 목적지 · 쥔 물체가 아닌 것이 모두 같은 강도의 장애물입니다. mug 를 뺀 실패 16 ep 에서 거절된 299 call 중 148 call (49 %) 은 진짜 장애물이 아닌 것이 막았고, 111 call (37 %) 은 여유 15 mm 만으로 거절됐습니다.

| 항목 | 내용 (결과 전 고정) |
|---|---|
| 자료 | `ms-pkgFrssdm-*` 112 ep + `ms-tab10sdm-*` 32 ep 의 `--record-constraints full` 기록 |
| 단위 | (call, 성분). 성분 = 장애물 물체 집합을 만드는 15 mm 연결 성분 중, 그 call 의 계획 궤적 (10 step) 에서 로봇 구 · 쥔 물체 구와 100 mm 안에 드는 것 |
| label (채점에만 쓰는 GT) | 성분의 점이 속한 물체가 그 episode 의 SafeLIBERO 장애물이면 1, 아니면 0. `surface_owner_task.py` 의 물체 대응을 그대로 씁니다. 0 은 task 물체 · 목적지와 구조물 · 다른 물체 · 로봇으로 나눠 적습니다 |
| feature (서버가 GT 없이 아는 값만) | ① 성분의 평균 attention (raw), ② 그 값의 call 간 누적 평균 (성분 대응은 중심 30 mm 안 최근접), ③ 그 call 의 성분 중 attention 순위, ④ close seed 까지 거리 (없으면 없음 표시), ⑤ 쥔 동안 gripper 열림 row 의 쥔 물체 footprint 까지 거리, ⑥ 인식된 target 중심까지 거리, ⑦ 지금까지 target 으로 인식된 call 의 비율, ⑧ 세 주축 폭과 받침면 위 높이, ⑨ latch 단계 |
| 식 | 표준화한 feature 의 logistic regression. feature 하나씩의 AUC 도 함께 보고 |
| 검증 | suite 하나를 빼고 맞춘 뒤 그 suite 로 평가 (4 회). level 하나를 빼는 것도 같이 |
| 보는 값 1 | 진짜 장애물 성분의 95 % 를 장애물로 남기는 문턱에서, 장애물이 아닌 성분 중 문턱 아래로 내려가는 비율 (= 여유가 줄어드는 비율) |
| 보는 값 2 | 기록된 거절 call 에 "문턱 아래 성분의 row 는 여유 0" 을 적용했을 때, 잃은 29 ep 에서 통과로 바뀌는 거절 call 의 비율과, 안전 성공 episode 에서 장애물이 막은 거절 call 중 통과로 바뀌는 비율 |

**진행 기준 (결과 전 고정).** 아래를 모두 채우면 후보 D 의 구현으로 갑니다. 하나라도 못 채우면 이 방향은 닫습니다.

- 보는 값 1 이 빼 둔 suite 네 개의 평균으로 50 % 이상이고, 어느 suite 에서도 35 % 아래가 아님
- 보는 값 2 에서 잃은 29 ep 의 거절 call 이 25 % 이상 통과로 바뀜
- 보는 값 2 에서 안전 성공 episode 의 장애물 거절 call 이 통과로 바뀌는 것은 5 % 이하

**먼저 확인할 것.** chunk 기록에 점별 attention 과 성분을 다시 만들 점군이 있는지. 없어서 depth 와 attention map 에서 다시 계산해야 하면 시간이 크게 늘어나므로, 그 경우 추출을 시작하지 말고 보고해 주세요.

**예상 시간.** 추출 script 3–4 시간, 추출 실행 2–4 시간, 맞추기와 표 1 시간. GPU 0. 산출물은 표, suite 별 ROC 와 feature 분포 그림, 계수 JSON 입니다.

**통과했을 때의 후보 D (지금은 구현하지 않음).** 장애물 물체 row 의 여유를 `15 mm × p` 로 둡니다 (`linearize.py` `obstacle_row_floor` 가 이미 row 별 floor 배열을 돌려줌). 구현 4–6 시간, 거르기 S32 2–3 회, 채택 검증은 B · C 와 합친 판으로 한 번입니다. GPU 를 B · C 와 나눠 써야 하므로 착수는 6.2 의 결과와 B · C 의 결과를 함께 보고 사용자가 정합니다.

## 7. 표 run 에서 같이 확인할 것

GT oracle arm (`<frozen>-gto`) 의 결과가 나오면 동결 판과 episode 짝으로 비교해 주세요. 위 분석의 예측은 "mug task 를 빼면 인식을 GT 로 바꿔도 안전 성공이 크게 늘지 않는다" 입니다. 예측과 다르게 크게 늘면 손실은 인식 쪽에 있고, 이 문서의 B · C 보다 인식을 먼저 봐야 합니다.

## 8. 하지 않기로 한 것

- row 별 obstacle confidence 로 여유를 가중하는 안의 **구현과 closed loop**: offline 실험 (6.2) 만 지금 하고, 구현은 그 결과를 본 뒤 정합니다.
- 팔 link 교착 때 물러났다 다시 접근하는 우회: B 가 '충돌만 는다' 로 끝났을 때의 다음 후보입니다.
