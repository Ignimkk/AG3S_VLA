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
| ~~multi-link CBF 는 보고값 자체가 없음~~ **정정 (2026-11-10 점검): 논문 Table Ib 에 static SafeLIBERO 값이 있음** — Ours TSR / CAR / SSR 60.3 / 73.0 / 51.7, frozen (optical flow 없음) 61.6 / 74.8 / 52.9, π0.5 60.0 / 16.6 / 15.1 (level 합산) | 공식 code 없음 | 우리 π0.5 합산 60.0 / 15.0 / 13.5 와 일치 → harness 정상. 표에 `*` 표시 |

**정리:** port 는 '같은 조건에서 safety layer 를 바꿔 끼운 비교' 로는 공정하고, 특히 AEGIS 의 privileged names 판은 공식 VLM 단계의 상한이다.
그러나 port 가 원 논문 수치보다 낮게 나오는 것은 사실이므로, (1) 표에 원 논문 보고값 (†) 을 같이 두고, (2) 본문 주장은 "같은 harness 의 재구현 대비" 로
범위를 한정하며, (3) 비교 대상은 privileged (GT) 판까지 포함한 가장 강한 판으로 잡는다.

**참고 (사용자 판단용):** 공식 AEGIS 의 GLM 호출은 episode 당 1 회라 tranche 1 의 SafeLIBERO 128 ep 이면 128 회다. 공식 코드를 작은 분할에서
한 번 돌려 port 와의 차이를 직접 재면 위 한계 2 행의 '추정' 이 측정으로 바뀐다 (비용 · API key 는 사용자 결정 사항).

### 4.4 KNOWS port 수정 이력 — F1 (손 ellipsoid calibration), 2026-11-09 18:57 적용 (사용자 결정)

**원칙 (사용자 판정 2026-11-09 18:5x):** baseline 에는 **원 논문이 의도한 기술만** 둔다. 논문에 없는 것은 더하지 않는다.
특히 '팔에 가려진 동안 장애물 추적을 고정' (점검 F3) 같은 가림 대처는 **그 모듈이 비교받는 핵심 능력**이므로 port 가 대신 보강하지 않는다.
원래 결과가 원문과 다르고 너무 낮아 **원인이 port 의 구현 결함으로 확인된 것만** 고친다 — 그 결과 · 원인 · 수정을 모두 남겨 논문 · 리뷰어에게 그대로 밝힌다.

| 단계 | 내용 | 근거 |
|---|---|---|
| 1. 실제 결과 (수정 전 port, MEASURED) | pilot init 5–14, spatial · object · goal: KNOWS* (detector) Level I TSR · CAR · SSR = 32.5 · 6.7 · 2.5 %, (GT seg) 65.0 · 27.5 · 25.0 %; Level II (detector) 25.0 · 33.8 · 5.0, (GT seg) 78.8 · 37.5 · 35.0 | run `e1a-knows-*`, `e1a-knows-gt-*` (2026-11-04 · 05) |
| 2. 원문과의 차이 | KNOWS 논문 Level I (4 suite 평균, DERIVED): TSR 67.5 · CAR 76.4 · SSR 59.9. 같은 harness 의 π0.5 단독은 보고값과 일치 (55.8 · 10.8 · 8.3 vs 55.0 · 12.4 · 11.5) → harness 가 아니라 port 쪽 | §4.2-4 |
| 3. 원인 (점검 `outputs/2026-11-09_KNOWS_port_점검/README.md`) | port 의 로봇 EE ellipsoid `(0.04, 0.04, 0.07)` m 가 grip site 중심이라 **Panda 손 · 손가락 충돌 mesh 꼭짓점을 하나도 감싸지 않음** (손은 grip site 뒤 3–12 cm, 옆 ±10 cm). GT seg run 의 Level I 충돌 87 건 중 **71 건의 첫 접촉이 손**, 충돌 순간 filter 자신의 분리 거리 h* > 0 이 126 / 137 ep (filter 는 비어 있다고 판단). filter 자체는 정상 (장애물 137 / 137 집합 안, 충돌 전 개입 129 / 137, QP 실패 0) | MEASURED (MuJoCo 접촉 재구성, filter 기록) |
| 4. 논문의 의도 | §3.1 EE ellipsoid 는 "offline calibrated", §5 보호하지 않는 것은 손목 위쪽 link 뿐 → 논문의 ellipsoid 는 손을 감싼다 | 논문 본문 (semi-axis 값은 Appendix 에도 없음) |
| 5. 수정 F1 | EE ellipsoid = **손 + 열린 손가락 충돌 mesh 꼭짓점의 MVEE** (Khachiyan, margin 0, grip-site frame): 중심 (−0.0075, 0, −0.0699) m, semi-axes (0.0332, 0.0884, 0.1287) m, 축 = grip-site 축에서 약 5° 기울어짐. 중심이 손목 회전에 따라 움직이는 항은 port 에 이미 있는 `cbf.filter.RobotBody` (한 부분 = 논문의 단일 ellipsoid) 로 넣음. **CBF 식 · γ · ε · normal · 장애물 인식 · 추적 · 실행 주기는 그대로** | `Baseline/paper_baselines/methods/knows/adapter.py` (`ee_model="hand_mvee"` 기본, 원래 모델 = `ee_model="grip_site_v0"`), diff `outputs/2026-11-09_KNOWS_port_점검/F1_patch/adapter.diff`, test `knows/tests/test_knows_f1_hand_ellipsoid.py` (4 개: 기본 모델 · provenance, 손 점 포함, 손 옆 장애물에서만 개입, 손목 회전 lever arm) + 기존 KNOWS test 전부 통과 |
| 6. 적용 범위 | tranche 1 의 **KNOWS 두 arm 은 모두 F1 판** (적용 시점 2026-11-09 18:57, queue 가 첫 KNOWS run 을 시작하기 전). run manifest 의 method provenance `robot.ee_model = hand_mvee` 로 구분. adapter 변경은 queue 시작 HEAD (c027b00d) 이후의 미commit 변경이라 각 KNOWS run 의 `source.patch` 에 그대로 기록됨 (queue 가 끝난 뒤 commit) | queue log `outputs/2026-11-09_baseline_tranche/t1/queue.log` |
| 7. 적용하지 않은 것 (2026-11-09 23:5x 판정으로 **대체됨 → §4.6**: F2 · F3 · F4a 넣음) | F2 (env step 마다 filter, 공용 harness 변경), **F3 (가림 동안 추적 고정 — 사용자 판정: 모듈 능력이므로 보강 안 함)**, F4 (detector recall · attention target 제외 — 논문 recipe 없음), F5 · F6 (ε · margin · γ 조정, 팔 link · 쥔 물체 보호 — 튜닝 · 방법 변경) | 점검 README §수정 |

**표 · 본문 표기:** KNOWS 행은 `KNOWS*†` — `*` = 공식 코드 없는 재구현, `†` = EE ellipsoid 를 손 · 손가락 mesh 로 calibrate (논문 §3.1 의 offline calibration 을 우리가 구현;
수정 전 port 결과는 supplement 에 같이). 원 논문 보고값도 같은 표 위쪽에 둔다.

**리뷰어 대응 문장 (초안, EN):** "KNOWS has no public code; we re-implemented it (marked *). Our first port, with a
grip-site-centred end-effector ellipsoid, reached only 2.5 % / 25.0 % SSR (detector / GT obstacles) on SafeLIBERO Level I,
far below the paper (59.9 %). Contact reconstruction showed the Panda hand struck the obstacle in 71 of 87 collisions while
the filter's own barrier was positive, i.e. the ellipsoid did not enclose the hand. Following the paper's offline
calibration (Sec. 3.1), we refit it to the hand and finger collision meshes (MVEE, no margin) and changed nothing else; all
reported KNOWS numbers use this version, and the original port's results are listed in the supplement."

**AEGIS 에 같은 종류의 수정이 생기면:** tranche 1 의 aegis · aegis-gt spatial Level I run (2026-11-09 17:59, 18:23 시작) 은 수정 전 port 로 돌았으므로 다시 돌리거나 '수정 전' 으로 표시한다 (dev-38 지적).

**참고 (test):** `closed_loop/tests/test_client_privileged.py` 실패 3 개는 F1 과 무관 — client 에 oracle 인자를 더한 17379c20 (11-05) · 2c1e4a50 (11-06) 이후 test 의 argv snapshot 이 갱신되지 않은 오래된 test (dev-38 확인).

**짝 비교 (선택, PLANNED):** F1 의 효과를 같은 episode 로 보이려면 pilot 분할 (init 5–14) 에서 F1 판 KNOWS 를 한 번 더 돌린다
(spatial · object · goal × Level I, 2 arm × 120 ep, 약 3–4 h). tranche 1 뒤 GPU 가 비면 사용자 결정으로.

### 4.5 AEGIS port 수정 이력 — 장애물 이름 문구 (공식 prefer_list), 2026-11-09 20:01 적용 (사용자 결정)

원칙은 §4.4 와 같다 (원 논문 · 공식 코드가 의도한 것만, 원인이 port 결함으로 확인된 것만 고침).

| 단계 | 내용 | 근거 |
|---|---|---|
| 1. 실제 결과 (수정 전 port, MEASURED) | pilot init 5–14, spatial · object · goal: AEGIS port (detector) Level I TSR · CAR · SSR = 47.5 · 34.2 · 26.7 %, (privileged names) 73.3 · 47.5 · 41.7 %; Level II (detector) 56.2 · 61.6 · 37.5, (names) 72.1 · 53.8 · 49.0 | run `e1a-aegis-*`, `e1a-aegis-gt-*` |
| 2. 원문과의 차이 | VLSA 보고 AEGIS: TSR 67.5 · **CAR 68.9** (suite 평균, level 합산) — 이름을 정답으로 줘도 CAR 47.5 / 53.8 | §4.3 |
| 3. 원인 (점검 `outputs/2026-11-09_AEGIS_port_점검/README.md`) | CBF · QP · EE ellipsoid (0.06 / 0.12 / 0.11, offset −0.08) · α = 10 h · 스케일 · MVEE 1 회 fit 은 **공식 코드와 같음** (기록 frame 으로 재구성한 call-0 h 차이 중앙값 0.05 mm, QP 실패 0). 차이는 **첫 call 의 장애물 위치 찾기**: names 판 224 ep 중 66 ep 이 장애물을 못 맞추거나 (38) 다른 물체를 잡음 (28) → 그 ep 의 CAR 18 % (같은 ep 의 π0.5 15 %). 맞게 잡은 158 ep 의 CAR **63.9 %** (같은 ep π0.5 19.6 %) ≈ 보고값. 못 잡는 이유 = **prompt 문구**: port 는 표시 이름 ("milk", "moka pot", "yellow book") 을, 공식 GLM 은 `prefer_list` 문구 ("red milk carton", "blue moka pot", "yellow rectangular book", utils.py:466) 를 내도록 지시받음. 같은 frame 에서 Grounding DINO 를 다시 돌려 (run 의 box 를 111 / 111 재현) 공식 문구만 넣으면 top-1 적중 **45 → 89 / 111** | MEASURED (재구성 · offline 재검출), DERIVED (분할 CAR) |
| 4. 공식 코드의 의도 | VLM 이 후보 목록에서 이름을 고르게 함 → '정답 이름' (privileged names 판) 의 올바른 대역은 그 목록의 문구 | `benchmark/SafeLIBERO/main/utils.py:464–490` |
| 5. 수정 | privileged names 판이 Grounding DINO 에 넘기는 이름을 공식 `prefer_list` 문구로 매핑 (`safety_common.AEGIS_PREFER_LIST_WORDING`: yellow book → yellow rectangular book, moka pot (+ small) → blue moka pot, red coffee mug → red mug, white storage box → 그대로, wine bottle (+ small) → black wine bottle, milk (+ small) → red milk carton). `ObstacleCloudSource(name_wording=…)` 기본 `display` (다른 방법 · arm 은 그대로), AEGIS adapter 만 기본 `aegis_prefer_list`. **CBF · QP · 형상 · 카메라 · 해상도 · crop · 실행 주기는 그대로**. 기대 효과 names 판 CAR 약 50 → 59 % (DERIVED) | diff `outputs/2026-11-09_AEGIS_port_점검/naming_patch/naming.diff`, test `aegis/tests/test_aegis_prefer_list_wording.py` (모든 SafeLIBERO 장애물 → 공식 목록, 다른 경로 불변) + AEGIS · KNOWS · 공용 test 90 통과 (`.venv-baselines`) |
| 6. 적용 범위 | tranche 1 의 privileged names 판 중 **spatial Level I (18:23 시작) 은 수정 전** → `experiments/segmentation/bl_t1_fix_queue.sh` 가 tranche 1 끝난 뒤 run-id `blt1-aegis-gt-spatial-LI-rerun` 으로 다시 돌림 (수정 전 run 은 기록으로 남김). 나머지 names 판 run 은 수정 후. detector 판 (일반 prompt "object") 은 이 경로를 쓰지 않아 영향 없음. provenance `obstacle.name_wording` 으로 구분, 미commit 변경은 각 run 의 `source.patch` 에 기록 | queue log |
| 7. 적용하지 않은 것 | backview 카메라 · 1024 px 융합 (공식 기술이나 공용 harness · π0.5 입력이 바뀜 — 사용자 결정으로 보류, 선언된 편차로 표기), 공식식 crop (`crop_mode="box"`, 효과 작음), env step 마다 filter (효과 작음), 팽창 · Q1 · α · margin 조정 · 매 call refit (튜닝) | 점검 README |

**표 · 본문 표기:** `AEGIS (port)†` — `†` = privileged names 판의 이름을 공식 prefer_list 문구로 줌. 남은 편차: chunk 단위 filtering, 머리 카메라 256 px 만 (공식: backview 융합 · 1024 px).
**리뷰어 대응 문장 (초안, EN):** "We ran AEGIS through our harness rather than its own loop. Its geometry and CBF-QP match the
official code exactly; the paid GLM-4.5V naming step is replaced by the ground-truth obstacle name (privileged upper bound of
that step) or by a generic detector prompt. Our first port passed the bare object names to Grounding DINO and failed to
localise the obstacle in 29 % of episodes (CAR 47.5 % vs 68.9 % reported); when the obstacle was localised, CAR was 63.9 %.
The official prompt makes GLM answer from a fixed list of descriptive names; we use those names, which doubles the
detector's top-1 hit rate (45 → 89 of 111 frames). All reported AEGIS numbers use this version; the first port's results
are in the supplement."

### 4.6 Baseline 재구성 원칙 R1–R5 와 F2 · F3 · F4a (사용자 결정 2026-11-09 23:5x, §4.4 의 '적용하지 않은 것' 을 대체)

**사용자 결정:** "F1–F4 를 어떻게 추가해도 상관없다. 단 실험이 논리적 구성이고 타당해야 한다" → 아래 규칙을 모든 baseline 에 같은 기준으로 적용한다.

| 규칙 | 내용 | 리뷰어 질문에 대한 답 |
|---|---|---|
| R1 넣는 기준 | 원 논문 · 공식 코드에 **적힌 것**만 구현, port 가 그와 다르면 고친다 | "논문대로 구현했다" |
| R2 빼는 기준 | 논문에 없는 tuning (ε · γ · margin) · 범위 밖 기능은 넣지 않고 한계로 밝힌다 | "baseline 을 따로 개선하지 않았다" |
| R3 공용 조건 | 실행 주기 등 harness 수준 선택은 **모든 arm 에 같이** 적용 | "조건이 공정하다" |
| R4 재현 불가 부품 | 미공개 부품 (fine-tune 한 YOLOE 등) 은 공용 부품으로 대신하고 밝힌다. perception 영향은 GT arm 으로 분리 | "perception 과 안전 기법을 분리해 봤다" |
| R5 확정 순서 | 수정은 개발 분할 (init 5–14) 에서만 검증 (수정 전 · 후 같은 episode), 설정을 **고정**한 뒤 보고 분할 (15–49) 은 한 번만 돌린다. 수정 전 결과는 supplement | "보고 결과를 보고 고친 것이 아니다" (G171 과 같은 구조) |

| 항목 | 판정 | 근거 |
|---|---|---|
| F1 손 ellipsoid | 넣음 (18:57 적용) | R1, 논문 §3.1 offline calibration |
| F3 '팔이 개입하는 동안 track 을 마지막 위치에 고정' | **넣음, GT · detector 두 판 같은 규칙** | R1, 논문 §3.2 ObjectTracker 자체 규칙 ("freezing a track at its last position when the arm intervenes"). port 가 덧붙이는 것이 아니라 KNOWS 의 가림 대처 능력을 논문대로 평가하는 것 (18:5x 판정의 취지와 같음: port 가 논문 밖의 보강을 하지 않는다) |
| F4a attention 으로 target 제외 (Eq. 2–4, π0.5 layer 12 head 3, agent view) | **넣음 (detector 판)** | R1, 논문 식 · layer · head 명시. AG3S 의 핵심 주장 (VLA attention 으로 target / 장애물 구분) 과 가장 가까운 경쟁 기법. 우리 재현 P0b 적중 0.682 (`methods/knows/docs/08-p0b-results.md`). K · δ 는 논문 미기재 → 가정값을 밝힘 |
| F4b fine-tune 한 YOLOE | 넣지 않음 | R4, recipe 미공개 → 공용 Grounding DINO "object", GT seg arm 이 상한 |
| F2 env step 마다 filter (policy 는 5 step 마다 그대로) | ~~넣음~~ → **철회 (§4.7, 01:1x)** | R1 + R3: AEGIS 공식 `main_aegis.py:299–362` 와 KNOWS 논문 모두 env step 마다 측정 EE pose 로 filter. chunk 단위 실행의 예측 오차 (5 step 뒤 p90 1.3–5 cm) 제거. baseline 에 유리한 방향이라 우리에게 보수적 |
| F5 · F6 | 넣지 않음 | R2 |
| multi-link CBF | 같은 checklist 로 점검 (2026-11-10 00:0x 시작) | 대칭성: 세 방법을 같은 기준으로 |

**순서:** tranche 1 queue 중지 (2026-11-09 23:58, 진행 중 run 1 개는 끝까지) → 변경 commit → F2 · F3 · F4a 구현 + CPU test → 개발 분할 짝 측정 (init 5–14, spatial · object · goal Level I, filter 6 arm) → 설정 고정 → tranche 1 재시작.
**tranche 1 의 기존 run 처리:** π0.5 단독 (none) run 은 영향이 없어 그대로 쓴다. filter arm run (AEGIS · CBF · KNOWS, 23:58 까지) 은 F2 전 판이라 보고하지 않고 '수정 전' 기록으로만 남긴다.

### 4.7 F2 철회 · tranche 1 재개 (사용자 결정 2026-11-10 01:1x)

**경위:** §4.6 에서 F2 (env step 마다 filter) 를 모든 filter arm 에 넣기로 하고 tranche 1 을 멈췄다 (23:58). 그 결과 이미 끝난 filter arm run
14 개 (약 5.5 h) 를 버리고 표 채우기가 2–3 일 늦어지는 비용을 dev-4a 가 충분히 따지지 않고 권했다. 사용자 지적 (01:0x) 뒤 다시 판단:
F2 의 기여는 점검에서도 2차 요인이고 (현재 충돌의 92 % 는 `h*>0`, 예측 오차와 무관), chunk 단위 실행은 **모든 arm (AG3S 포함) 이 같은 VLA chunk
인터페이스에서 동작하는 공용 조건** (R3) 이라 선언된 편차로 충분하다.

| 결정 | 내용 |
|---|---|
| F2 | **철회.** 보고 run 은 모두 `--filter-rate chunk` (launcher 기본값). F2 code (`--filter-rate step`, bfa9bdb0) 는 supplement 의 민감도 실험용으로 남긴다 |
| F1 · F3 · F4a · AEGIS 이름 문구 | 유지 (R1, 논문 · 공식 코드에 적힌 것) |
| 개발 분할 짝 측정 | 표 채운 뒤로 미룬다 (supplement 근거, GPU 가 빌 때) |
| 기존 run 재사용 | π0.5 단독 spatial LI · LII · object LI, AEGIS detector spatial LI · LII · object LI, AEGIS names spatial LII · object LI, multi-link CBF ± GT spatial LI · LII (12 run, 약 4.3 h) |
| 재실행 (`-r2`) | AEGIS names spatial LI (이름 수정 전), KNOWS ± GT spatial LI · LII (F3 · F4a 전) — 처음 run 은 '수정 전' 기록으로 남김 |
| 재사용 근거 | bfa9bdb0 의 변경은 chunk 경로에서 AEGIS · multi-link CBF 동작을 바꾸지 않음: client flag off byte-identical, `ObstacleCloudSource(target_box=True)` 기본 = 이전 동작, test 104 + 50 통과 (구현 보고 `outputs/2026-11-10_baseline_F2F3F4a/README.md`) |
| 남은 위험 | ~~multi-link CBF 점검 (진행 중) 에서 R1 수정이 나오면 그 arm 은 다시 돌린다~~ → 점검 완료 (§4.9): 수정 후보는 있으나 §4.8 규칙상 GT 판은 원문과 가까워 지금 적용하지 않음, 기존 run 유지 |

queue: `experiments/segmentation/bl_tranche_resume_queue.sh` (재실행 5 → 남은 블록 scenario-major → LIBERO, 60 run).

### 4.8 중간 점검 규칙 — 원문 지표보다 너무 낮으면 그때 개선안 적용 (사용자 결정 2026-11-10 01:3x)

**사용자 결정:** "우선 이렇게 판정하고, 원문 지표보다 너무 낮게 나오면 그때 적용. 실험 중간에 확인하면서 차이가 너무 크면 개선안을 적용한 뒤 진행."
'너무 낮음' 의 기준은 tranche 결과를 보기 **전에** 아래처럼 고정한다 (`experiments/segmentation/bl_tranche_check.py`).

| 항목 | 기준 |
|---|---|
| 점검 시점 | 재실행 5 run 끝 (spatial 블록 완성), 그 뒤 suite 블록 (Level I + II) 이 끝날 때마다 |
| harness 확인 | π0.5 단독 TSR 이 KNOWS 논문 π0.5 행과 10 pt 이내 (≥ 64 ep 일 때) — 벗어나면 port 가 아니라 harness 를 먼저 의심 |
| 표시 (FLAG) | arm 의 누적 CAR 또는 SSR 이 논문 기준값보다 **20 pt 넘게 낮고**, 그 level 에서 **≥ 64 ep** (2 suite 이상; 64 ep 에서 이항 SE ≤ 6.3 pt → 20 pt > 3 SE) |
| 기준값 | π0.5 · KNOWS: KNOWS 논문 Tab. 1 (4 suite 평균, level 별), KNOWS GT seg: 같은 표의 Naive (GT segmentation), AEGIS ± names: VLSA Tab. I (level 합산 → 우리도 합산 행으로 비교), multi-link CBF ± GT: 논문 Table Ib static SafeLIBERO 'Ours' 60.3 / 51.7 / 73.0 (TSR / SSR / CAR, level 합산 → 합산 행으로 비교) |
| FLAG 시 개선안 후보 | 점검 보고에 **이미 적힌** fidelity 수정만 (예: F2 env step 마다 filter, AEGIS backview · 1024 px, AEGIS 의 VLM 이름 단계). tuning (R2) 은 여전히 불가 |
| FLAG 시 절차 | 사용자에게 보고 → 결정 → 그 arm 의 tranche run 을 모두 같은 판으로 다시 돌림 (섞지 않음). 기록: "실제 결과 → 원문과의 차이 → 원인 → 적용한 개선 → 범위" (§4.4 형식), 결과를 보고 적용했다는 사실도 그대로 밝힘 |

**01:3x 첫 점검 (재실행 전, 참고):** AEGIS detector 판 level 합산 48 ep CAR 18.8 % (논문 68.9, −50 pt) — ep 수가 64 미만이라 아직 FLAG 아님.
같은 판의 pilot 도 낮았고 (§4.5), 원인은 공식 GLM 이름 단계를 일반 prompt "object" 로 바꾼 것 (AEGIS 점검 README). 64 ep 를 넘으면 FLAG 될 가능성이 높다.

### 4.9 multi-link CBF port 점검 결과와 판정 (2026-11-10 01:4x, §4.8 규칙 적용)

점검: `outputs/2026-11-10_multilinkCBF_port_점검/README.md` (GT seg 200 ep 를 기록 frame 으로 재실행해 served chunk 를 정확히 재현, 최대 차이 0.0).

| # | 발견 | 분류 |
|---|---|---|
| 1 | GT seg 판의 장애물 ellipsoid 가 실제 장애물보다 작음: 공식 AEGIS 의 '중심에서 먼 20 % 점 제거' 를 손목 카메라 위주 (중앙 75 %) 의 깨끗한 GT 점에 적용 → 긴 물체 끝이 잘림 (wine bottle 높이 0.254 m, semi-axis 0.077 m). wine bottle 이 충돌 89 건 중 53 건 | trim 은 논문대로 (R1), 손목 depth 사용은 논문 Fig. 2 와 다름 (R1 / R3 편차) |
| 2 | filter 자체는 작동: 장애물이 집합에 89 / 89, 충돌 전 개입 76 / 89, QP 실패 0 | — |
| 3 | 충돌 순간 닿은 robot mesh 가 장애물 ellipsoid 밖 47 / 49 (중앙 63 mm) | DERIVED |
| 4 | robot ellipsoid 가 robosuite mesh 가 아닌 cuRobo 구에 fit (hand mesh 의 70.6 % 만 포함) — 지금 충돌에는 2 / 49 만 관련 | R1, 작음 |
| 5 | chunk 예측 오차 p90 5.8–17.3 mm | F2 (철회, 공용 편차) |
| 6 | detector 판: 일반 prompt 로 hazard 최대 8 개 (논문: VLM 이 이름 붙인 1 개), call 0 에 감시 장애물 포함 97 / 200 | R4, 선언 |

**§4.8 규칙에 따른 판정:**
- **GT seg 판:** spatial 32 ep 합산 TSR / SSR / CAR 68.8 / 59.4 / 68.8 vs 논문 60.3 / 51.7 / 73.0 → CAR −4 pt, SSR +8 pt. 원문과 가까움 → **수정하지 않음**, 기존 run 유지. 점검의 수정 후보 (exact geometry ellipsoid, 머리 카메라만, mesh fit) 는 기록만 하고, 이후 suite 에서 FLAG 되면 그때 적용.
- **detector 판:** 32 ep 합산 CAR 34.4 (−38.6 pt), SSR 18.8 (−33.0 pt) — 64 ep 미만이라 아직 FLAG 아님. AEGIS detector 판 (48 ep CAR 18.8, −50 pt) 과 같은 원인 (논문의 VLM 이름 단계를 일반 prompt 로 바꿈, R4) 으로 보이며, 두 detector 판 모두 FLAG 될 가능성이 높다. 그때의 개선안 후보는 두 방법에 공통인 'VLM 이름 단계' (AEGIS: 공식 GLM-4.5V, 비용 · 계정 필요) — 사용자 결정 사항.

### 4.10 AEGIS · multi-link CBF detector 판: GLM-4.5V 이름 단계를 로컬 open VLM 으로 (2026-11-10 03:4x, dev-4a 자율 판정 — 사용자 위임 02:0x)

**사용자 위임 (2026-11-10 02:0x):** "원문과 차이가 커 개선 후 재실행해야 하면 스스로 판정하고 진행. 결과를 끝까지 기다리지 말 것." → §4.8 의 64 ep 조건을 기다리지 않고, pilot (init 5–14) + tranche 1 이 같은 방향이면 적용한다.

| 단계 | 내용 | 근거 |
|---|---|---|
| 1. 실제 결과 | AEGIS detector: pilot level 합산 CAR 약 45 %, tranche 1 48 ep CAR 18.8 % (원문 68.9). multi-link CBF detector: pilot 약 46 %, tranche 32 ep 34.4 % (원문 Table Ib 73.0) | MEASURED (`bl_tranche_check.py`) |
| 2. 원문과의 차이 | 두 arm 모두 −24 ~ −50 pt, 두 분할에서 일관 → FLAG 로 판정 | §4.8 |
| 3. 원인 | 원문은 VLM (AEGIS: GLM-4.5V, multi-link: Qwen3-VL) 이 지시문 · 영상으로 장애물 **하나**의 이름을 정하고 그 이름으로 Grounding DINO. port 는 일반 prompt "object" 의 상자 최대 8 개 (target 상자 제외) → 감시 장애물이 call 0 집합에 있는 episode 97 / 200 (object LI 0 / 40), 엉뚱한 물체가 장애물로 들어감 | AEGIS · multi-link 점검 README |
| 4. 검토한 대체안 | (c) `candidate_list`: 공식 prefer_list 각 문구를 Grounding DINO 에 넣고 최고 점수 하나 → call-0 frame 7 개 중 정답 4 (빨간 테두리 접시 등에 'red milk carton' 이 고점) → **기각**. (d) `vlm_name`: 공식 GLM prompt (utils.py:478) 를 그대로 로컬 open VLM `Qwen/Qwen2.5-VL-7B-Instruct` (greedy) 가 답함 → pilot call-0 frame 24 개 (6 run × 4) 중 이름 정답 **20**, 그 이름의 Grounding DINO top-1 상자가 장애물을 맞힘 **19** (yellow book 이 주 실패 3 / 4) | `outputs/2026-11-10_detector_candidate_list/` (`probe_candidate_list_partial.json`, `probe_vlm_name.json`) |
| 5. 적용 | AEGIS · multi-link CBF 의 detector 판 = `detector_prompt="vlm_name"` (serve_method 기본, weight `$AG3S_BASELINE_VLM_WEIGHT` = 로컬 사본). 이후 crop · ellipsoid · CBF 는 privileged names 판과 같은 경로. 비특권 (정답 이름 사용 안 함), 비용 0, 외부 API 없음. KNOWS detector 판은 그대로 (논문에 VLM 이름 단계 없음) | `safety_common.py` (vlm_name), `providers/vlm_namer.py`, `serve_method.py` |
| 6. 적용 범위 | 기존 detector run (AEGIS spatial LI · LII · object LI, multi-link spatial LI · LII) 은 '수정 전' 기록으로 남기고 `-r2` 로 재실행, 나머지 블록 · LIBERO 도 새 판. 결과를 보고 적용했다는 사실을 표 caption / supplement 에 밝힌다 | queue B (arms queue) |
| 7. 남은 편차 (선언) | VLM 이 GLM-4.5V / Qwen3-VL 이 아님 (transformers 4.53 에서 가능한 가장 가까운 open 모델), thinking mode 없음, 입력 영상 = port 의 머리 카메라 256 px (공식 1024 px) | — |

**리뷰어 대응 문장 (초안, EN):** "The official AEGIS pipeline names the obstacle with the paid GLM-4.5V API (and multi-link CBF with
Qwen3-VL). Our first non-privileged port replaced this step by a generic detector prompt, which left the monitored obstacle out of
the hazard set in half of the episodes and fell 24–50 points below the reported CAR. We therefore restored the step with the
official prompt verbatim, answered by the open Qwen2.5-VL-7B-Instruct model (greedy); it names the correct obstacle in 20 of 24
held-out frames. All reported detector-variant numbers use this version; the first port's results are in the supplement."

### 4.11 KNOWS F1 의 EE frame bug 수정 (2026-11-10 04:0x, dev-4a 자율 판정 — 사용자 위임)

| 단계 | 내용 | 근거 |
|---|---|---|
| 1. 실제 결과 | spatial LI (16 ep): F1 만 — GT seg TSR / CAR 56.2 / 25.0, detector 37.5 / 6.2; F1 + F3 + F4a (`-r2`) — GT seg 56.2 / **6.2**, detector 37.5 / **0.0** (π0.5 단독 CAR 6.2) | MEASURED |
| 2. 원문과의 차이 | KNOWS Naive (GT seg) CAR 82.2, KNOWS 76.4 → −57 ~ −76 pt. 같은 GT 장애물의 multi-link CBF 68.8 | §4.8 |
| 3. 원인 | **F1 의 frame bug**: filter 가 받는 EE 자세 (`observation/state` axis-angle) 는 `robot0_right_hand` body 자세인데, F1 hand MVEE 는 grip-site (`right_gripper`) frame 에서 calibration → 두 frame 은 Rz(90°) 차이 (`panda_gripper.xml:14`, 기록 frame 98 개 모두 90.0°). 실제로 돈 ellipsoid 는 손 · 손가락 collision vertex 의 중앙 **4.7 %** 만 포함 (보정 시 94.2 %) → F1 이 사실상 적용되지 않았음. r2 의 손 · 손가락 충돌 10 건 모두 as-run h* > 0, 보정 frame 에서는 h* ≤ 0. F3 는 주원인 아님 (F1 → r2 차이 3 / 16 ep, McNemar p = 0.25; F3 는 추적 오차 34.8 → 14.5 mm 로 줄임), Eq. 12 ε slack · chunk 단위도 주원인 아님 | `outputs/2026-11-10_KNOWS_collapse_진단/README.md` (재생이 served chunk 를 1,425 call 에서 비트 단위로 재현) |
| 4. 논문의 의도 | §3.1 "pose (c_R, R_R) is read from the proprioceptive state at each timestep and the semi-axes are calibrated offline" → calibration 한 frame 과 읽는 자세의 frame 이 같아야 함 | 논문 |
| 5. 수정 | `GRIP_SITE_TO_EE` (= Rz(−90°) 회전, robosuite XML 에서 test 로 검증) 로 MVEE 축 · 중심을 EE frame 에 옮김: `Q_local = M Q_site Mᵀ`, `ee_offset = M c_site`. MVEE 값 · CBF · F3 · F4a 는 그대로. `ee_model="hand_mvee_v1"` = 수정 전 재현용 | `methods/knows/adapter.py`, `tests/test_knows_f1_hand_ellipsoid.py` (XML quat 대조, v1 재현), methods test 110 통과 |
| 6. 적용 범위 | KNOWS ± GT 의 모든 tranche run 을 `-r3` 로 (spatial LI 를 먼저 돌려 closed-loop 효과 확인 후 계속). `-r2` 와 F1-only run 은 '수정 전' 기록 | queue C (`bl_tranche_queue_c.sh`) |
| 7. 하지 않은 것 | 개발 분할 짝 측정 (R5) 은 표 채운 뒤 supplement 근거로; ε · fixed normals · γ_h · W · margin (R2) | — |

**참고:** AEGIS 는 공식 code 의 EE ellipsoid 가 같은 right_hand frame 기준이라 (0.12 m 축이 y) 해당 없음. multi-link CBF 는 관절 FK 로 손 위치를 놓아 해당 없음.

### 4.12 multi-link CBF GT 판 → 논문의 privileged 조건 "exact geometry" (2026-11-10 05:1x, dev-4a 자율 판정)

| 단계 | 내용 | 근거 |
|---|---|---|
| 1. 실제 결과 | GT seg 판 spatial + object 64 ep (level 합산): TSR / SSR / CAR 70.3 / 39.1 / **46.9** | `bl_tranche_check.py` 04:48 |
| 2. 원문과의 차이 | 논문 Table Ib static SafeLIBERO 'Ours' 60.3 / 51.7 / 73.0 → CAR −26 pt, **FLAG** (§4.8, ≥ 64 ep) | §4.8 |
| 3. 원인 | GT seg 점군이 손목 카메라 위주 (중앙 75 %) 인데 AEGIS 공식 trim ('중심에서 먼 20 % 제거') 이 긴 물체의 끝을 잘라 hazard ellipsoid 가 실제보다 작음 (wine bottle 높이 0.254 m → 최대 semi-axis 0.077 m). 충돌 89 건 중 53 건이 wine bottle (object suite 에 집중) | `outputs/2026-11-10_multilinkCBF_port_점검/README.md` §1, M1 |
| 4. 논문의 의도 | 논문의 privileged 조건은 perception 을 거치지 않는 "exact geometry" (Table III ‡, §VI) — GT arm 은 perception 의 trim 을 통과시키는 것이 아니라 정답 형상을 줘야 함 | 논문 |
| 5. 수정 | `--gt exact` (multi-link CBF 의 `--gt` 기본): 활성 장애물의 collision geom 상자 꼭짓점의 MVEE (margin 0, base frame) 를 첫 call 에 한 번 fit, `obstacle_source="ellipsoids"`. CBF · robot 모델 · 실행 주기는 그대로. `--gt seg` 는 이전 조건으로 남김 | `closed_loop/privileged.py` `exact_ellipsoids`, `arms.py`, `serve_method.py`; test `closed_loop/tests/test_gt_exact_geometry.py`; CPU smoke 오류 0 / 60 call |
| 6. 적용 범위 | multi-link CBF GT 의 모든 tranche run 을 `-r2` (exact) 로. 이전 GT seg run 은 '수정 전' 기록 (supplement: GT seg 조건 결과) | queue D |
| 7. 하지 않은 것 | 비특권 (detector) 판의 trim 제거 (논문 trim 그대로, R2), robot ellipsoid 의 mesh 재 fit (점검 M3, 현재 충돌 2 / 49 만 관련 — FLAG 가 남으면 다음 후보), 'true motion' (optical flow 생략은 선언된 편차) | — |

**같은 시점 KNOWS (§4.11) closed-loop 확인:** spatial LI GT seg `-r3` TSR / CAR 75.0 / **43.8** (수정 전 r2 6.2), detector `-r3` 62.5 / 6.2 (detector recall 한계, R4). GT seg 는 64 ep 이 모이면 다시 판정.

### 4.13 queue D 초기 점검 — spatial LI, 수정한 세 arm (2026-11-10 06:31, dev-4a 자율 판정)

각 칸은 같은 block (spatial LI, init 15–18, 16 ep) 의 수정 전 → 수정 후 값이다 (TSR / SSR / CAR, %).

| arm | 수정 전 (c027b00d) | 수정 후 (d452c804, `-r2`) | 논문 | 판정 |
|---|---|---|---|---|
| AEGIS detector (VLM naming, §4.10) | 50.0 / 12.5 / 12.5 | 87.5 / 50.0 / **50.0** | CAR 68.9 (level 합산) | 큰 개선. 차이 −19 pt 로 16 ep 에서는 FLAG 기준 (20 pt) 안 → 진행, ≥ 64 ep 에서 재판정 |
| multi-link CBF detector (VLM naming) | 50.0 / 12.5 / 12.5 | 75.0 / 43.8 / **50.0** | 60.3 / 51.7 / 73.0 | 큰 개선. 차이 −23 pt 이지만 16 ep (95 % CI 폭 ≈ ±24 pt) → 진행, ≥ 64 ep 에서 재판정 |
| multi-link CBF GT (exact, §4.12) | 68.8 / 56.2 / 68.8 (seg) | 68.8 / 43.8 / 62.5 | 60.3 / 51.7 / 73.0 | spatial LI 는 원래 문제 block 이 아니었음 (seg 도 68.8) → 판정 불가. **결정 block 은 object LI** (seg CAR 6.2, wine bottle). queue D step 2 의 object LI `-r2` 끝에서 판정 |

**판정:** 세 arm 모두 중단할 근거가 없어 queue D 를 그대로 진행한다. 다음 점검은 multi-link GT exact object LI (예상 ~10:2x KST) 이다. 그 block 의 CAR 이 여전히 원문보다 20 pt 넘게 낮으면 §4.12 의 '다음 후보' (M3 robot ellipsoid mesh 재 fit) 를 적용하고 multi-link GT 를 `-r3` 으로 다시 돌린다.

**결정 block 결과 (2026-11-10 11:33):** multi-link GT exact object LI `-r2` TSR / SSR / CAR 50.0 / 31.2 / **62.5** (seg 6.2 → 62.5), spatial LII `-r2` 68.8 / 62.5 / 75.0. 64 ep pool (object LII 는 아직 seg) 68.8 / 42.2 / 60.9 → 논문 대비 CAR −12 pt, SSR −10 pt, FLAG 아님. **§4.12 수정이 원인을 해소했다고 판정, M3 은 적용하지 않음.** 같은 시점에 SafeLIBERO 4 suite 를 다 채운 arm: π0.5 단독 LI 50.0 / 4.7 / 6.2, LII 62.5 / 17.2 / 23.4 (논문 대비 CAR −6.2 / +0.4) · AEGIS-GT 128 ep 71.1 / 44.5 / 60.2 (CAR −8.7) — 공통 harness 가 원문 수준을 재현한다는 근거.

**pool 해석 주의:** `bl_tranche_check.py` 는 block 마다 가장 높은 `-rN` 을 고른다. 그래서 detector arm 이 아직 `-r2` 가 없는 block 에서는 수정 전 run 을 섞어 보여 준다 (06:31 의 AEGIS pooled CAR 31.2 는 수정 전 object LI · spatial LII 를 포함). queue D 의 step 3 가 그 block 을 `-r2` 로 대체한 뒤의 pool 이 판정 대상이다.

### 4.14 detector 판 64 ep 판정 (2026-11-10 15:14, dev-4a 자율 판정)

| arm | spatial LI | spatial LII | object LI | object LII | 64 ep TSR / SSR / CAR | 논문 | 판정 |
|---|---|---|---|---|---|---|---|
| AEGIS detector (VLM naming) | 50.0 | 56.2 | 43.8 | 68.8 | 76.6 / 46.9 / **54.7** | CAR 68.9 | −14 pt → 통과, 진행 |
| multi-link CBF detector (VLM naming) | 50.0 | 56.2 | **12.5** | 43.8 | 68.8 / 32.8 / **40.6** | 60.3 / 51.7 / 73.0 | −32 pt → **FLAG** |

(block 칸 = CAR %, 16 ep.) object LI 에서도 VLM 이름은 맞다 ("black wine bottle", in prefer_list). 두 arm 은 같은 perception (같은 VLM 이름, 같은 head cloud, 같은 공식 trim) 을 쓰는데, multi-link 의 object CAR 은 9 / 32 로 AEGIS 의 18 / 32 보다 낮다. 논문에서는 multi-link 가 AEGIS 이상 (73.0 vs 69.25) 이므로, 원인은 perception 보다 multi-link filter 쪽일 가능성이 있다 [I]. 후보: §4.12 의 M3 (guarded ellipsoid 가 Panda collision mesh 를 다 감싸지 않음), filter 개입 차이, task 물체 밀기 (범위 밖), noise.

**조치:** queue D 를 'aegis goal LI-r2' 끝에서 멈추고, 같은 HEAD 로 queue E (`bl_tranche_queue_e.sh`: AEGIS detector 나머지 + LIBERO AEGIS → KNOWS ±GT `-r3`) 를 시작했다. multi-link detector 의 남은 block 과 LIBERO multi-link 는 진단 (`outputs/2026-11-10_multilinkCBF_detector_FLAG/`) 이 끝나 R1 수정 여부를 정한 뒤 돌린다. 이미 끝난 multi-link detector `-r2` 4 block 은 수정이 있으면 '수정 전' 기록으로 남긴다.

**진단 결과와 판정 (2026-11-10 15:4x, `outputs/2026-11-10_multilinkCBF_detector_FLAG/README.md`):**

| 질문 | 결과 |
|---|---|
| 같은 perception 인가 | object 32 쌍 ep 모두 VLM 이름 · head box 동일 [M], offline replay 로 served chunk 재현 [D] |
| noise 인가 | 쌍 비교: AEGIS 만 충돌 0, multi-link 만 충돌 9, 둘 다 14 → McNemar p = 0.004 [D]. noise 아님. spatial 은 4 vs 4 |
| multi-link 충돌 부위 | object LI 14/14 gripper (hand 11, finger 3). LII 는 9 건 중 6 건이 쥔 task 물체가 장애물을 민 것 (논문 모델 범위 밖) [M] |
| filter 동작 | 접촉 전 14/14 개입 (중앙 10 step 전), QP 실패 · fallback 0, filter 자신의 h 는 접촉 row 에서 +1.0–4.7 cm [M] → filter 는 정상, hazard 가 작아서 '떨어져 있다' 고 판단 |
| 원인 | 공통 perception (공식 80 % trim + head 1 view) 의 hazard 가 작다 (접촉 body mesh 가 3.8–7.5 cm 밖). AEGIS 는 공식 EE ellipsoid 가 그 오차를 덮는다. 특히 instruction 에 orange juice · milk · alphabet soup 이 있으면 켜지는 공식 tall ellipsoid (0.06, 0.12, 0.20) (`main_aegis.py:161-164`), gripper mesh 바깥 margin 55 mm. multi-link 의 gripper ellipsoid 는 mesh 에 맞게 작다 (margin ~20 mm) [D] |
| M3 (mesh 재 fit) 효과 | 14 접촉 중 h* ≤ 0 로 바뀌는 것 0 건, h* 차이 ≤ 1.4 mm [D] → CAR 효과 ≈ 0 |
| 공식 two-view (head + backview) | hazard 가 두꺼워지지만 multi-link h* ≤ 0 0/14 그대로 [D] |

**판정: port 수정 없음.** multi-link filter 는 논문대로 동작하고, object suite 의 multi-link < AEGIS 차이는 (1) 논문 그대로의 공통 perception trim 과 (2) AEGIS 공식 tall-object EE heuristic 의 차이에서 온다. 다음은 하지 않는다:
- multi-link gripper 에 AEGIS 손 ellipsoid 나 cm 단위 margin 을 주는 것 (결과를 보고 정한 값 → R2 위반; 논문은 margin 값을 주지 않음).
- trim 을 빼는 것 (R2).

M3 은 효과가 측정상 0 이므로 적용하지 않고 선언된 편차로 남긴다 (guarded ellipsoid 출처 = cuRobo sphere, mesh 대비 gripper 81 % 포함 · 최대 9.5 mm 밖, 접촉 h* 차이 ≤ 1.4 mm). 논문 Table Ib 는 4 suite 합산이라 object suite 만으로 'multi-link ≥ AEGIS' 를 확인할 수 없고, goal · long 이 남아 있다. multi-link detector 의 이미 끝난 4 block 은 유효하며, 남은 block 은 같은 HEAD 에서 queue E 다음에 돌린다.

### 4.15 LIBERO-Safety tranche C 보류 해제 (2026-11-10 22:4x, 사용자 지시 "LIBERO safety 도 1/10 만큼")

근거: `outputs/2026-11-10_LIBERO-Safety_사전판정/README.md` (CPU 점검, GPU 미사용).

| 선행 판정 | 결과 | 근거 |
|---|---|---|
| H1 공식 `pi05_libero_safety` weight + TrainConfig | 해결 (2026-11-04 G8 · G9 · G15 에서 이미 완료; E1 protocol · 계획 §1.3 의 "H1 필요" 문구가 낡음) | config `third_party/openpi/src/openpi/training/config.py:785-810`, HF @e66d84bf sha256 일치, GPU smoke 1 ep `official_comparable: true` |
| H2 controller 동등성 | π0.5 단독: 해결 (client 가 fork OSC kp 750 · 출력 ±2 · 입력 clip ±1 을 쓰고 action 을 바꾸지 않음 = 공식 data · weight 와 같은 controller). **action 을 고치는 arm (AG3S · control arm · filter arm) 은 robosuite 1.4.1 단위 (0.05 m / 0.5 rad) 를 가정 → fork-native profile 이 필요** (README §3.3) | [M] CPU 측정 |
| B1 공식 cost 가 robot 접촉을 못 셈 | 해결: 계획대로 공식 cost 로 종료 + 보조 열 (geom-id robot 접촉, 제약별 평가). 보조 기준 SR 은 재실행 없이 계산 가능 | client · `bl_tranche_ls_report.py` |

**결정 (dev-4a):**
- checkpoint = 공식 `pi05_libero_safety` (논문 Table 3 π0.5 와 비교 가능한 유일한 조건).
- tranche C = 4 suite × L0–L2 × task 0–4 × **init 15 → 19 (§4.16)** × seed 7 = 60 ep, 공식 cost 종료, L2 먼저. 전체 10 trial 은 init 19–28 (§4.16 보고 분할 19–49; init 5–14 는 개발 분할이라 쓰지 않음).
- 본문 CR 은 공식 값 (R1), 보조 CR_aux 를 옆 열로 같이 보고.
- 실행: `experiments/segmentation/bl_tranche_libero_safety_queue.sh` (DRY 12/12 통과), 예상 GPU 1.4 h. 지금 queue (KNOWS → multi-link detector) 뒤에 실행.
- filter arm (AEGIS · multi-link · KNOWS) 의 LIBERO-Safety 실행은 계획 Tab. 2 밖이고 구현 2–3 일 (launcher · 단위 · GT 장애물 정의) → **하지 않음 (사용자 판정 2026-11-10 23:0x)**. LIBERO-Safety 의 비교는 계획대로 π0.5 / + generic / + external / AG3S 4 arm.


### 4.16 보고 분할 = init 19–49, baseline port 동결 (사용자 판정 2026-11-10 23:1x, A안)

| 항목 | 내용 |
|---|---|
| 문제 | R5 (§4.6) 는 "수정은 개발 분할 (init 5–14) 에서 확인, 보고 분할은 고정 뒤 한 번" 인데, §4.10–4.12 의 수정 (VLM 이름 · KNOWS frame · multi-link exact geometry) 은 tranche 1 = **init 15–18 (보고 분할의 일부) 결과를 보고** 결정했다 (사용자 위임 02:0x 아래). 결과를 보고 고친 분할을 headline 에 넣으면 R5 · G171 과 어긋난다 |
| 완화 요인 | 수정은 모두 원문 · 공식 code 근거 (값 조정 아님; margin · trim 제거 같은 값 선택은 거절, §4.14), 방향은 baseline 성능 상승 = AG3S 에 보수적 |
| 결정 (A안) | init 15–18 = **'baseline port 검증 분할'** 로 표시하고 모든 arm (π0.5 · baseline · AG3S) 의 headline 에서 뺀다. **보고 분할 = init 19–49 (31 init)**. baseline port 는 queue F 가 끝나면 **동결** (이후 수정 = 새 실험 ID). LIBERO-Safety tranche C 도 init 19 (전체 10 trial = init 19–28) |
| 기각 | B: 15–49 유지 + 공개 (방어 약함). C: 수정 전·후를 5–14 에서 재검증 (GPU 1–2 일) |
| 반영 | 공동 계획서 `AG3S_PLAN_SCHEDULE_20261110_KO.md` §2 · §5 (23:2x), `bl_tranche_queue_f.sh` (LS `INITS=19`) |

### 4.17 KNOWS-GT 64 ep FLAG → queue E 중단, queue F 앞당김 (2026-11-11 00:1x, dev-4a 자율 판정)

| block (`-r3`, 16 ep) | KNOWS-GT TSR / SSR / CAR | KNOWS detector TSR / SSR / CAR |
|---|---|---|
| spatial LI | 75.0 / 43.8 / 43.8 | 62.5 / 6.2 / 6.2 |
| spatial LII | 62.5 / 25.0 / 37.5 | 68.8 / 37.5 / 37.5 |
| object LI | 81.2 / 56.2 / 56.2 | 31.2 / 6.2 / 6.2 |
| object LII | 68.8 / 37.5 / 37.5 | 81.2 / 37.5 / 43.8 (사후 metric, 아래) |
| **64 ep** | **71.9 / 40.6 / 43.8** | **60.9 / 21.9 / 23.4** |
| 논문 (LI / LII) | 73.0 / 63.1 / 82.2 · 67.8 / 55.6 / 76.1 ("Naive GT seg") | 67.5 / 59.9 / 76.4 · 71.1 / 53.8 / 68.8 |

- KNOWS-GT CAR −35 pt (64 ep) → §4.8 FLAG. 같은 harness 의 π0.5 · AEGIS-GT · multi-link-GT 는 원문 ±9 pt 안이므로 harness 보다 KNOWS port 쪽 원인 [I].
- §4.16 (A안) 에서 port 동결은 queue F 뒤 → 지금은 init 15–18 (port 검증 분할) 로 수정 가능.
- **조치:** queue E 를 'knows-gt goal LI-r3' 끝에서 멈추고 (남은 KNOWS SafeLIBERO 7 run + LIBERO KNOWS 4 는 진단 뒤), queue F (multi-link detector 나머지 + LIBERO multi-link + LIBERO-Safety π0.5 init 19) 를 먼저 돌린다. 진단: `outputs/2026-11-11_KNOWS_GT_gap/`.
- **HEAD 변경 사고:** 23:58 에 dev-f1 의 migration commit (69d7060a, 문서 · migration script 만) 이 HEAD 를 바꿔, 'knows object LII-r3' 의 마지막 provenance 기록이 거절됨 (launcher rc=1). client 는 16/16 끝남 → 같은 `metrics.summarize` 로 `closed_loop_metrics.json` 을 사후 작성, run dir 에 `POSTHOC_METRICS_NOTE.txt`. 이후 run 은 69d7060a (실행 경로 code 동일).
- **사용자 지시 (2026-11-11 00:4x):** 진단은 지금 하되 KNOWS 판정 · 재실행은 공동 계획서의 **마지막 행**으로. 따라서 동결 순서: π0.5 · AEGIS ±GT · multi-link ±GT 는 queue F 뒤 동결 → 보고 run 가능, KNOWS ±GT 는 판정 뒤 동결.

**진단 결과 (2026-11-11 00:3x, `outputs/2026-11-11_KNOWS_GT_gap/README.md`; CPU, 기록 재생 — closed loop 아님):**

| 항목 | 결과 |
|---|---|
| KNOWS-GT 충돌 36 건의 첫 접촉 [M] | 팔 link 5–7: 12 · 쥔 / 다른 물체: 7 · 간접 (로봇 접촉 없음): 11 · 손 / 손가락 (EE): 6 |
| 원문 모델 범위 | 36 건 중 30 건 (83 %) 은 원문이 다루지 않는 부위 — 원문 §3.5 "only ... end effector ellipsoid ...; other components of the robot arm are not considered". EE filter 자체는 동작 (EE 접촉 π0.5 37/53 → 6/36). 같은 EE-only 인 AEGIS-GT 도 같은 양상 (같은 4 block CAR 56.2) |
| port 고유 원인 | **GT 판의 정의가 원문 행과 다름.** 원문 Tab. 1 "Naive (GT segmentation)" = "a single fixed obstacle ellipsoid is placed using ground-truth segmentation at t = 0 and never updated" (KNOWS §4.1, 확인함). 우리 GT 판은 매 call head + wrist GT mask 로 중심을 다시 재고 F3 tracking 을 씀 → 정적 장애물의 추정 중심이 중앙 35 mm (p90 72 mm) 흔들림, 4 건은 실제 겹침인데 h > 0 [D] |
| 나머지 EE 충돌 | chunk 단위 지연 3 (공통 편차) · 장애물 MVEE 의 가시면 한계 1 |
| detector 판 | 충돌 49 건 중 38 건에서 장애물이 제약 집합에 없음; call 0 에 장애물 빠짐 45/64 → Grounding DINO recall (R4 로 선언한 YOLOE 대체) 이 지배. F4a 오제외와 분리 불가 (target track 중심 미기록) |

**권고 (판정은 공동 계획서 마지막 행, 사용자 지시 00:4x):**
- **F5 (R1):** KNOWS-GT 판을 원문 Naive 정의대로 — t = 0 에 GT segmentation 으로 놓은 장애물 ellipsoid 를 고정 (`serve_method.py` gt seg 분기에서 knows 이면 `track_centroids=False`; adapter 는 이미 지원, F3 자동 생략). 예상 CAR 43.8 → 50–57 (원문 대비 약 −25 pt 는 남음: 팔 · 물체 접촉은 원문 모델 밖, 값 조정 금지 R2). 적용하면 GT 판 tranche 1 전체를 `-r4` 로 재실행.
- detector 판: 수정 없음 (R4 recall 한계). attention target 중심 기록을 추가해 F4a 오제외를 셀 수 있게 하는 것은 기록 개선 (결과 불변).
- 원문 충돌 정의 "contacted any non-target obstacle" 와 SafeLIBERO 1 mm 규칙의 차이 가능성 [I] 은 검증 불가 → R3 대로 SafeLIBERO 규칙 유지, 공개.


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
| 2026-11-09 18:57 | KNOWS port F1 (손 · 손가락 MVEE EE ellipsoid) 적용, tranche 1 KNOWS arm 은 F1 판; F3 등 나머지 미적용 | §4.4, 사용자 결정 |
| 2026-11-09 20:01 | AEGIS port: privileged names 판 이름을 공식 prefer_list 문구로 (backview 등 미적용), tranche 1 spatial LI names 판 재실행 queue | §4.5, 사용자 결정 |
| 2026-11-09 23:58 | tranche 1 queue 중지 (진행 중 aegis-gt object LI 는 끝까지), 재구성 원칙 R1–R5 와 F2 · F3 · F4a 채택, multi-link CBF 점검 시작 | §4.6, 사용자 결정 |
| 2026-11-10 01:2x | F2 철회 (사용자 결정), 기존 run 12 개 재사용, KNOWS 4 · AEGIS names spatial LI 재실행, tranche 1 재개 (`bl_tranche_resume_queue.sh`) | §4.7 |
| 2026-11-10 01:3x | 중간 점검 규칙 (20 pt · ≥ 64 ep) 고정, `bl_tranche_check.py` | §4.8, 사용자 결정 |
| 2026-11-10 01:4x | multi-link CBF 점검 완료, §4.8 규칙상 GT 판 원문 근접 → 미수정, 논문 Table Ib 값 반영 (§4.1 정정) | §4.9 |
| 2026-11-10 03:4x | AEGIS · multi-link CBF detector 판 → VLM 이름 (`vlm_name`, Qwen2.5-VL-7B, 공식 prompt), 기존 detector run 5 개 `-r2` 재실행 예정; KNOWS 원인 분석 중 | §4.10, 사용자 위임 |
| 2026-11-10 04:0x | KNOWS F1 EE frame bug 수정 (Rz 90°), KNOWS ± GT 전부 `-r3` 재실행; queue C (KNOWS spatial LI 먼저 → 확정 arm 나머지 → AEGIS · CBF detector VLM 판 → KNOWS 나머지) | §4.10, §4.11, 사용자 위임 |
| 2026-11-10 05:1x | multi-link CBF GT FLAG → `--gt exact` (논문 privileged 조건), GT run 전부 `-r2`; queue D (변경 arm spatial LI 먼저) | §4.12, 사용자 위임 |
