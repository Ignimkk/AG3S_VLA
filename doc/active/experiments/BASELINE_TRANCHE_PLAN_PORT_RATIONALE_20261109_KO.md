# Baseline 선행 실행 (1/10 분할) 계획 · port 판 근거 · 결과 (작성 2026-11-09 17:xx KST, dev-4a)

자료 유형: **MEASURED** (기록된 run 값) · **DERIVED** (기록에서 계산) · **PLANNED** (아직 안 함). 결과 절은 분할 (tranche) 이 끝날 때마다 고친다.

## 1. 사용자 결정 (2026-11-09 16:4x–16:5x)

| 결정 | 내용 |
|---|---|
| 선행 실행 | AG3S 와 무관한 baseline (π0.5 단독 · AEGIS · multi-link CBF · KNOWS) 을 **먼저** 돌려 benchmark 마다 넘어야 할 수치를 정하고, AG3S 는 그 수치를 목표로 개발한다 |
| 전용 GPU | baseline 은 전용 GPU 로 돌린다 (AG3S 개발 run 과 GPU 를 나누지 않음) |
| 실제 결과 | **지금부터 돌리는 baseline run 은 논문 표에 들어가는 결과**다 (pilot 아님) |
| 비용 | 비용이 드는 것 (AEGIS 공식 코드의 GLM-4.5V API) 은 **port 판으로 대신**한다 — 단 port 가 합리적인 근거를 밝힌다 (§4) |
| 규모 | 최종 실험 계획 (`paper/evidence/AG3S-VLA_CVPR2027_FINAL_EXPERIMENT_PLAN_20261108_KO.md` §1.2) 의 **1/10 로 먼저** 돌리고 결과를 쓴 뒤, 나머지 9/10 을 나중에 더해 같은 문서 · 표를 고친다 |

## 2. 분할 정의 (tranche)

- 보고용 분할 = **init 15–49** (판정 G171: init 5–14 는 AG3S 구성 선택에 쓴 개발 분할, 0–4 는 dev). 모든 arm · AG3S 가 같은 (suite, level, task, init, seed 7).
- 계획 규모 (보고 분할): SafeLIBERO 4 suite × task 4 × Level I · II × init 35 = 1,120 ep/arm, LIBERO 4 suite × task 10 × init 35 = 1,400 ep/arm.
- **Tranche 1 = init 15–18** (35 개 중 4 개 = 11 %, '1/10' 의 가장 가까운 정수 init 수). LIBERO init state 는 미리 무작위로 만든 것이라 연속 번호를 골라도 편향이 없다.
  이후 tranche 는 init 19–49 를 나눠 더한다 (같은 queue, `INITS=… T=t2 …`).

| tranche 1 | 장면 | arm | ep |
|---|---|---|---:|
| A. SafeLIBERO | spatial · object · goal · long × task 0–3 × Level I · II × init 15–18 (128 ep/arm) | π0.5 · AEGIS port · AEGIS port (privileged names) · multi-link CBF* · CBF* (GT seg) · KNOWS* · KNOWS* (GT seg) | 896 |
| B. LIBERO (장애물 없음) | spatial · object · goal · 10 × task 0–9 × init 15–18 (160 ep/arm) | π0.5 · AEGIS port · CBF* · KNOWS* (privileged 판은 장애물 GT 가 없어 제외) | 640 |
| C. LIBERO-Safety | π0.5 단독 1/10 = 60 ep (4 suite × L0–L2 × task 5 × 1 trial) | π0.5 | **보류** — 계획 §1.3 의 선행 판정 H1 (`pi05_libero_safety` weight + TrainConfig, weight 는 storage 에 있음) · H2 (controller) · B1 (cost 판정) 상태를 먼저 확인 |

고정 규약: checkpoint `pi05_libero`, seed 7 (episode seed = seed·100000 + task·1000 + init), `--safelibero-step-budget main_aegis`
(policy 300 / 550 + wait 20), 충돌 = 공식 1 mm 변위 (충돌해도 episode 계속), launcher `Baseline/paper_baselines/closed_loop/launch_method.py`
(GPU 독점 lock 을 스스로 잡음), 집계 = suite 동일 가중 평균, Wilson 95 % CI, arm 간 exact McNemar (계획 §1.2).

실행: `bash experiments/segmentation/bl_tranche_queue.sh` (기본 `INITS=15,16,17,18 T=t1 PARTS="A B"`), log `outputs/2026-11-09_baseline_tranche/t1/queue.log`,
run-id `blt1-<arm>-<suite>-L<level>` (LIBERO `blt1-<arm>-<suite>`).
**ESTIMATED 시간** (pilot 의 episode 당 중앙값 39–79 s + run 마다 server 기동): A 약 16 h, B 약 9 h → 약 25 h (GPU 1 개).

**같은 기계 조건:** 기계가 바뀌면 π0.5 chunk 가 달라진다 (2026-11-03 서버 이전 때 확인) → 표의 모든 arm (AG3S 포함) 은 **같은 서버**에서 돌린다.
AG3S 개발을 다른 서버에서 해도, 동결한 AG3S 의 보고 run 은 이 baseline 을 돌린 서버에서 돌리거나, 두 서버의 π0.5 단독 출력이 같음을 먼저 확인한다.
queue 가 도는 동안 이 서버의 HEAD 를 바꾸지 않는다 (git pull · commit 금지).

## 3. 이미 있는 자료 (pilot, init 5–14 — 보고에 쓰지 않음, 개발 목표로만)

2026-11-04 20:19 – 11-05 23:4x, 같은 7 arm × SafeLIBERO spatial · object · goal × Level I · II (goal Level II 는 AEGIS 2 arm 일부에서 멈춤, long 없음).
**DERIVED** (run `e1a-<arm>-<suite>-L<level>`, suite 3 개 합산, n = 80–120):

| arm | Level I TSR · CAR · SSR (%) | Level II TSR · CAR · SSR (%) |
|---|---|---|
| π0.5 단독 | 55.8 · 10.8 · 8.3 | 66.7 · 27.5 · 26.7 |
| AEGIS port (detector) | 47.5 · 34.2 · 26.7 | 56.2 · 61.6 · 37.5 |
| AEGIS port (privileged names) | 73.3 · 47.5 · 41.7 | 72.1 · 53.8 · 49.0 |
| multi-link CBF* (detector) | 48.3 · 35.8 · 24.2 | 50.0 · 57.5 · 33.8 |
| multi-link CBF* (GT seg) | 70.0 · 60.0 · 50.0 | 78.8 · 48.8 · 42.5 |
| KNOWS* (detector) | 32.5 · 6.7 · 2.5 | 25.0 · 33.8 · 5.0 |
| KNOWS* (GT seg) | 65.0 · 27.5 · 25.0 | 78.8 · 37.5 · 35.0 |
| (참고) AG3S pe123458, Spatial LI 만 | 55.0 · 57.5 · 37.5 | — |

## 4. port 판을 쓰는 근거 (AEGIS · multi-link CBF · KNOWS)

### 4.1 왜 port 인가
| arm | 공식 코드 | port 를 쓰는 이유 |
|---|---|---|
| AEGIS | 있음 (`benchmark/SafeLIBERO/main/main_aegis.py`, THU-RCSCT/vlsa-aegis) — 장애물 이름을 GLM-4.5V API 로 받음 | **사용자 결정: API 비용이 드는 경로는 port 로 대신** |
| multi-link CBF | 없음 (project page 만) | port 외에 방법이 없음 → 표에 `*` |
| KNOWS | 없음 (`Baseline/paper_baselines/methods/knows/README`: 논문만 있는 재구현) | port 외에 방법이 없음 → 표에 `*` |

### 4.2 port 가 합리적인 이유
1. **같은 평가 환경에서 safety layer 만 다르다.** 모든 port 는 AG3S 와 같은 π0.5 server 경로 (`none` 은 `serve_safe --robot panda --no-safe` 와 같음),
   같은 SafeLIBERO client (`libero_safe_client.py`, 수정 없음), 같은 step 예산 · seed · 충돌 판정 · 입력 (head + wrist depth, policy RGB) 을 쓴다.
   공식 AEGIS 는 자기 loop · 자기 evaluator 로 돌아가 horizon · wait · 집계 차이가 결과에 섞인다 → **방법 차이만 비교하려면 같은 harness 가 필요**하다.
2. **AEGIS 에서 바꾼 것은 '장애물 이름 하나를 고르는 단계' 뿐이고, 그 단계의 상한을 같이 보고한다.** 공식 코드는 episode 시작 장면 한 장으로
   GLM-4.5V 에 "장애물 하나의 이름" 을 묻는데 (`main/utils.py` `obstacle_detection`, episode 당 1 회), 프롬프트가 SafeLIBERO 장애물 이름 후보
   (책 · 모카포트 · 머그 · 보관함 · 와인병 · 우유 (+ long: binder)) 를 준다. port 는 이를 (a) Grounding DINO 일반 prompt (`aegis`, 비특권) 와
   (b) **정답 이름 (`aegis --gt` = privileged names)** 두 판으로 둔다. (b) 는 VLM 이 이름을 완벽히 고른 경우이므로 **공식 VLM 단계의 상한**이고,
   두 판이 공식 AEGIS 의 인식 성능을 위아래로 감싼다. 그 뒤의 CBF-QP 안전 단계는 port 에 그대로 있다.
3. **재현성:** 외부 API 는 model 버전 · thinking 출력이 바뀌어 재실행 결과가 달라질 수 있다. port 는 고정된 weight (`grounding-dino-tiny`) 로 결정적이다.
4. **평가 환경이 원 논문 보고값을 재현한다 (MEASURED · DERIVED, §3):** π0.5 단독 Level I = TSR 55.8 · CAR 10.8 · SSR 8.3 vs KNOWS 논문 보고 (No CBF, 4 suite 평균)
   55.0 · 12.4 · 11.5 — 같은 checkpoint 에서 같은 수준. 차이가 생기면 harness 가 아니라 safety layer 쪽이다.

### 4.3 port 의 한계 (표 · 본문에 같이 밝힌다)
| 한계 | 근거 | 대응 |
|---|---|---|
| **chunk 단위 filtering** — 원 논문은 env step 마다 CBF, port 는 policy call 마다 10 행 (client 는 0–4 행 실행) | run manifest `deviation_from_paper` | supplement Tab. S13 에 deviation 목록 |
| **AEGIS port 가 보고값보다 충돌을 덜 피함** — pilot Level I CAR 34.2 (detector) · 47.5 (privileged names) vs VLSA 보고 AEGIS CAR 68.9 (TSR 67.5) | §3 (보고값은 suite 평균 · level 합산이라 정확히 같은 조건은 아님) | 이름을 정답으로 줘도 차이가 남음 → 차이는 인식이 아니라 chunk 단위 filtering 쪽으로 추정. **Tab. 1 위쪽에 원 논문 보고값 (†) 을 같이 둔다** (계획에 이미 있음) |
| **KNOWS port 가 보고값보다 크게 낮음** — detector 판 SSR 2.5, GT seg 판 25.0 vs 보고 KNOWS SSR 59.9 | §3 | 원 논문은 VLA attention + segmenter, port detector 판은 Grounding DINO → 인식 경로가 다름. GT seg 판에서도 차이 → 9/10 전에 port 구현 점검 권장 |
| multi-link CBF 는 보고값 자체가 없음 | 공식 결과 없음 | 재현 비교 불가, `*` 표시 |

**정리:** port 는 '같은 조건에서 safety layer 를 바꿔 끼운 비교' 로는 공정하고, 특히 AEGIS 의 privileged names 판은 공식 VLM 단계의 상한이다.
그러나 port 가 원 논문 수치보다 낮게 나오는 것은 사실이므로, (1) 표에 원 논문 보고값 (†) 을 같이 두고, (2) 본문 주장은 "같은 harness 의 재구현 대비" 로
범위를 한정하며, (3) 비교 대상은 privileged (GT) 판까지 포함한 가장 강한 판으로 잡는다.

**참고 (사용자 판단용):** 공식 AEGIS 의 GLM 호출은 episode 당 1 회라 tranche 1 의 SafeLIBERO 128 ep 이면 128 회다. 공식 코드를 작은 분할에서
한 번 돌려 port 와의 차이를 직접 재면 위 한계 2 행의 '추정' 이 측정으로 바뀐다 (비용 · API key 는 사용자 결정 사항).

## 5. 결과

### 5.1 Tranche 1 (init 15–18) — PLANNED (시작 후 채움)
| arm | LIBERO SR | L-I TSR | L-I SSR | L-I CAR | L-II TSR | L-II SSR | L-II CAR |
|---|---:|---:|---:|---:|---:|---:|---:|
| π0.5 단독 | | | | | | | |
| AEGIS port (detector) | | | | | | | |
| AEGIS port (privileged names) | — | | | | | | |
| multi-link CBF* (detector) | | | | | | | |
| multi-link CBF* (GT seg) | — | | | | | | |
| KNOWS* (detector) | | | | | | | |
| KNOWS* (GT seg) | — | | | | | | |
| **AG3S (동결 뒤, 같은 서버)** | | | | | | | |

### 5.2 AG3S 목표 (tranche 결과로 갱신)
| benchmark | 목표 (계획 §10.1 판정을 baseline 실측으로 치환) |
|---|---|
| LIBERO (장애물 없음) | SR ≥ π0.5 − 1.5 pt |
| SafeLIBERO Level I · II | **SSR > 최강 비특권 baseline** (그리고 privileged 판과의 차이 보고), TSR ≥ 최강 − 1.5 pt, CAR 최고 |
| LIBERO-Safety | π0.5 대비 AAG ↑ · 충돌률 최저 · 위반 0 (C 보류 해제 뒤) |

## 6. 교체 기록
| 날짜 · 시각 | 무엇 | 이유 |
|---|---|---|
| 2026-11-09 17:xx | 문서 작성, tranche 1 정의 (init 15–18) | 사용자 결정 §1 |
