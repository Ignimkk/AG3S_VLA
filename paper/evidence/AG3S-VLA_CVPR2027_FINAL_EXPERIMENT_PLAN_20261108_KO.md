# AG3S-VLA CVPR 2027 최종 실험 계획 — 원고의 빨간 글씨를 측정값으로 바꾸기

- 작성: 2026-11-08 KST. 상태: **`PLANNED`** (실행 승인 아님). 상위 문서: [AG3S-VLA_CVPR2027_Experimental_Design.md](AG3S-VLA_CVPR2027_Experimental_Design.md)(E1–E7, 표기 규칙), [E1 baseline protocol](../experiments/configs/E1_SAFELIBERO_LIBEROSAFETY_BASELINE_PROTOCOL_20261104_KO.md).
- 대상 원고: `paper/latex/exp_section/04_experiments.tex`(4.1–4.4), `05_conclusion.tex`. 두 파일의 **빨간 글씨(`\proj{…}`) 165곳**은 모두 투영값이다. 이 문서는 그 하나하나를 어떤 실험의 어떤 산출물로 대체할지 정한다.
- 원칙: 투영값은 목표이지 결과가 아니다. 결과가 투영과 다르면 **원고를 결과에 맞춘다.** 표본·분할·임계값은 실행 전에 이 문서에서 고정하고, 사후에 바꾸면 변경 이력을 남긴다. 문헌 보고값(†)과 자체 측정값(R)은 끝까지 섞지 않는다.

## 0. 한눈에 보기

| 단계 | 실험 | 원고 위치 | 산출물 | 선행 조건 |
|---|---|---|---|---|
| S0 | 설정 동결 (head · threshold · split) | 4.1 Implementation details | `LOCK_eval.json` | AG3S 최종 구현 동결 |
| E1a | SafeLIBERO + LIBERO 주 비교 | Tab. 1, 4.2 첫 문단 | `e1a_*/episodes.jsonl`, 집계 표 | S0, baseline port 검증 |
| E1b | LIBERO-Safety L0–L2 · HRI | Tab. 2, 4.2 둘째 문단 | `e1b_*/…` | H1 · H2 · B1 판정 |
| E4-F | 실패 원인 분류 | 4.4 "Where it still fails" | 실패 라벨 표, replay 영상 | E1a 완료 |
| E5 | Ablation 7 변형 | Tab. 3, 4.3 | `e5_*/…` | S0 |
| E2/E3 | Grounding · identity | Tab. 4, Fig. 5, 4.4 첫 문단 | `e2_*/summary.json` | GT 생성기, held-out 장면 |
| E6 | Held-out 일반화 · run time | 4.4 마지막 문단 | split 기록, timing.json | E1a 구성 |
| E7 | 실로봇 | Fig. 4, 4.2 셋째 문단 | trial 기록, 센서 로그, 영상 | 하드웨어 · 안전 절차 |
| SUP | Supplementary · 리뷰어 대비 자료 | 부록 | §6, §7 | 위 전부 |

권장 순서: S0 → E1a pilot(초기 상태 5–14, 320 episode/arm) → E5 (같은 320) → E1a 전체(5–49) → E2/E3 → E4-F → E6 → E1b → E7 → SUP. pilot에서 투영과 방향이 다르면 원고와 목표를 먼저 고친다.

## 1. 원고의 빨간 항목 전수표

### 1.1 설정 (4.1)

| 원고 문장 | 결정할 것 | 결정 방법 | 기록 |
|---|---|---|---|
| RB-Y1 dual-arm, RealSense D435 3대(head · 양 손목) | 실제 플랫폼 · 카메라 수 · 모델 | 하드웨어 확인 | E7 §3.8 |
| 손목 F/T 센서 5 N + 외부 카메라 2대, annotator 2명 blind | 접촉 판정 규칙 | pilot 10 trial로 임계값 · 일치율 확인 | E7 |
| layer 8, head 6 (noun attention) · prefix layer 8 (identity) | 최종 checkpoint에서 선택 | dev initial state에서만: 물체 위 attention peak 비율 · IoU로 선택, 후보 전체 표를 supplement에 | S0 |
| scenario당 initial state 0–4 dev, 5–49 eval (1,440 episode/method) | 분할 | 현재 E1 protocol 안 B. 대안 A(5–14, 320)는 pilot 전용 | S0 |
| NVIDIA H200 1장 | 측정 하드웨어 | timing 측정 시 1 job 독점 | E6 |

### 1.2 Tab. 1 — LIBERO + SafeLIBERO (E1a)

행 9개 × 열 7개(LIBERO SR; Level I/II × TSR · SSR · CAR). 모든 셀이 투영값. ETS · SBU · CNF · suite별 값은 supplement.

| arm | 원고 표기 | 구현 상태 (2026-11-04 기준) | 비고 |
|---|---|---|---|
| π0.5 alone | `none` | 실행 가능 | 투영 TSR 57.8 / CAR 18.6은 VLSA Tab. I(같은 checkpoint) 보고값에 맞춘 것 |
| + generic geometry | `--target-field-policy none` 계열 | Panda profile 추가 필요 | AG3S와 **같은 TO** |
| + external segmentation | Grounded-SAM-2 mask → code image | client mask 전송 · 서버 `oracle-visible` 허용 **구현 필요** | prompt = 지시문 noun |
| AEGIS (official code) / (official, GT) | `benchmark/SafeLIBERO/main/main_aegis.py` (THU-RCSCT/vlsa-aegis submodule, 수정 없음) | 공식 코드 있음 → **주 행은 공식 코드 실행** (GLM-4.5V API 키 · prompt · 버전 기록, env-step CBF-QP, 우리 horizon · evaluator와 동일 조건 확인). port(`aegis`, GLM 대체 · chunk 단위)는 supplement Tab. S13의 대조용으로만 | 사용자 판정 2026-11-08: 재구현 대신 공식 코드 |
| multi-link CBF* | `multilink_cbf` | 공식 코드 없음(project page만, survey [1]) → port(∗) | detector 입력 version |
| KNOWS* | `knows` | 공식 코드 없음(`methods/knows/README`: paper-only reimplementation) → port(∗) | segmenter = YOLOE 또는 GDINO 명시; 보고값(KNOWS Tab. 1)과 재현값의 차이를 Tab. S13에 |
| AG3S-VLA | 최종 구성 flag | 동결 후 | non-GT 경로(gt24att 계열)여야 함. GT perception(B5)은 **oracle 행으로만** |

- **Tab. 1 상단 보고값 블록(†)은 같은 π0.5-LIBERO checkpoint의 저자 보고값으로 교체했다(사용자 판정 2026-11-08):** VLSA Tab. I(π0.5 CAR 17.1 / TSR 57.8; AEGIS 68.9 / 67.5; suite 평균, level 합산, SSR 없음)과 KNOWS Tab. 1(No CBF / Naive / KNOWS, suite × level SR · CR · SSR → 우리가 4 suite 평균을 **DERIVED**로 계산: Level I No CBF 55.0 / 11.5 / CAR 12.4, Naive 73.0 / 63.1 / 82.2, KNOWS 67.5 / 59.9 / 76.4; Level II 57.8 / 20.0 / 23.0, 67.8 / 55.6 / 76.1, 71.1 / 53.8 / 68.8). KNOWS의 CR 정의(비target 장애물 접촉)와 공식 1 mm 규칙이 다름을 caption에 명시. SafeDojo(OFT 기반) 블록은 supplement Tab. S2로.
- **보고값이 바꾼 것:** 두 논문 모두 safety layer가 π0.5 단독보다 **TSR을 올린다**(57.8 → 67.5; 55.0 → 67.5). 충돌이 target을 넘어뜨려 task 실패가 되기 때문이다. 따라서 AG3S가 SafeLIBERO에서 TSR을 잃으면 baseline보다 약 10 pt 아래가 되어 SSR 최고를 주장할 수 없다. 투영을 AG3S TSR 66.0(π0.5 +8.2, 최강 layer −1.5) · SSR 61.5 · CAR 87.5로 고쳤고, '미세하게 낮다'는 LIBERO 무장애물 열(95.4 vs 96.8)에만 남겼다.
- LIBERO SR 열: 장애물 없는 원 suite, 같은 initial state 분할. "safety layer가 없을 때의 비용"을 재는 열이므로 π0.5 · AG3S · 모든 safety layer를 다 돌린다.
- 표본: 전체 = 4 suite × 4 task × 2 level × 45 init = **1,440 episode/arm** (+ LIBERO 4 suite × 10 task × 45 = 1,800). 9 arm이면 SafeLIBERO만 12,960 episode. **ESTIMATED** 시간(E1 protocol 실측 57 s/ep VLA-only, filter 64–95 s, TO arm은 독점 lock 1 job): VLA-only ≈ 23 h, filter arm ≈ 26–38 h, TO arm은 request당 latency에 비례(투영 0.3 s면 ≈ 30 h, 현재 prototype 1–3 s면 60–120 h). 따라서 **전체 분할은 headline 4 arm(π0.5, AG3S, KNOWS*, AEGIS* GT)만**, 나머지 5 arm은 pilot 분할(5–14, 320 ep)로 먼저 채우고 시간이 남으면 확장. 표에는 arm마다 n을 적는다.
- 고정: checkpoint `pi05_libero`, config, seed 7, `--safelibero-step-budget main_aegis`(policy 300/550 + wait 20), 충돌 = 공식 1 mm 변위, 충돌해도 episode 계속. 같은 (suite, level, task, init, seed)를 모든 arm에.
- 집계: suite 동일 가중 평균(원문 방식 재확인). SSR은 episode 단위 conjunction. Wilson 95 % CI; arm 간 차이는 scenario cluster bootstrap 2,000회(seed 20261104) + exact McNemar.
- 원고 문장용 파생값: TSR−SSR, CAR−SSR 분해; "무충돌 실패 중 hold timeout 비율 79 %" → episode 종료 사유(hold timeout / 잘못된 grasp / 놓침)를 `episodes.jsonl`에 기록해야 함(현재 없음 → client에 `end_reason` 추가, 기본 꺼짐).
- "AEGIS* 잔여 충돌의 61 %가 이름 없는 clutter · 손목 · 전완 · 운반 물체" → 충돌 row에 (robot link, 접촉 물체, 물체가 named obstacle인지) 기록 필요.

### 1.3 Tab. 2 — LIBERO-Safety (E1b)

- 선행 판정 3개(E1 protocol §3): H1 `pi05_libero_safety` weight(12.4 GB) + openpi TrainConfig, H2 controller kp 750 · ±2 action 처리, B1 공식 `info['cost']`가 robot 접촉을 못 세는 문제(공식 그대로 쓰고 보조 contact 판정을 별도 열로).
- 표본: 물리 4 suite × L0–L2 × 5 task × 10 trial = 600/arm, arm 4개(π0.5, generic, external, AG3S). 본문은 AAG · TSA · FSHOA L2, supplement에 전체 + HRI + 실행 시간 + LDLJ.
- 원문 ±는 3 training seed 분산 → 우리는 단일 weight라 표기 불가. 본문 캡션에 이미 적혀 있음.
- "π0.5 row가 보고값보다 낮다" 문장은 **결과가 그럴 때만** 남긴다.

### 1.4 Fig. 4 + 4.2 셋째 문단 — 실로봇 (E7)

- 4 task(clear / similar neighbor / carry-path obstacle / unseen objects) × 4 method(π0.5, generic, external, AG3S) × 20 trial = 320 trial. task별 초기 배치 20개를 사전 등록하고 method 순서를 무작위화. unseen objects task는 development에 쓴 물체와 겹치지 않음을 물체 목록으로 증명.
- 판정: task success(목표 위치 · 놓음), safe success, unsafe contact(F/T 임계 + 외부 영상 annotator 2명, 불일치 시 제3자). AG3S의 자체 추정은 판정에 쓰지 않는다.
- 기록: 모든 trial 영상(3인칭 + wrist), F/T 로그, AG3S 선택 surface overlay, 실패 사유. 안전 절차(속도 제한, e-stop, 실행 예산) 문서화.
- 20 trial/task는 초기값. pilot 10 trial 후 효과 크기와 CI를 보고 최종 횟수를 정한다(설계 문서 §10).

### 1.5 Tab. 3 — Ablation (E5)

7 변형, 모두 SafeLIBERO Level I pilot 분할(5–14, 320 ep)에서 full과 paired. 각 변형의 구현 flag:

| 변형 | 바꾸는 것 | flag / 구현 |
|---|---|---|
| Seeds: attention → nearest object | ATS seed를 "손에 가장 가까운 3D 후보"로 | geometry-rule target 선택기 **구현 필요** |
| Extent: graph → radius growth | depth-continuous graph 대신 radius + clustering | 기존 coarse extent 경로 |
| Extent: graph → external mask | Grounded-SAM-2 mask lift | E1a external 행과 같은 경로 |
| Identity: memory off | per-request 선택 | memory flag off |
| Policy: target always obstacle | carve-out 없음 | `--target-field-policy none` |
| Policy: target freed, all links/stages | global exemption | 기존 G 계열 flag |
| Policy: carried object unattached | attach off | attach flag off |

- ΔSSR과 95 % CI는 paired cluster bootstrap. "Object suite에서 distractor가 더 가까운 initial state가 약 1/3" 같은 설명 수치는 initial state별 거리 계산 script로 산출해 sidecar에 남긴다.

### 1.6 Tab. 4 + Fig. 5 + 4.4 첫 문단 — Grounding · identity (E2/E3)

- 장면: LIBERO 물체로 구성한 진단 장면 4종(single target / similar objects / touching clutter / partial occlusion) + occlusion 시퀀스(부분 → 완전 가림 → 재등장). task · 물체 · 배치로 dev/held-out 분리, 분할 hash 기록.
- GT: 시뮬레이터 instance label + depth + calibration(image time), 관측 표면만, 5 mm voxel support, 보이지 않는 뒷면 제외. 빈 예측 · 미관측 GT 처리 규칙 고정.
- arm: geometry rule, YOLOE(지시문 noun), Grounded-SAM-2(지시문 noun / 정답 이름), oracle, AG3S w/o memory, AG3S. 외부 모델의 checkpoint · prompt · confidence 기록(기존 2026-11-01 factorial 결과에서 prompt 민감도가 컸음).
- 지표: soft 3D IoU · precision · recall · target-ID accuracy · abstain; identity accuracy · ID switch 수 · re-ID율. frame → episode → task 집계.
- Fig. 5 자료: 3행 × 4열에 쓸 held-out 장면 3개의 attention map, SAM-2 mask lift, AG3S surface, GT를 같은 camera로 렌더.
- "같은 장면에서 지시만 바꾸면 target · 제약이 바뀐 비율"은 scene pair 표로 supplement.

### 1.7 4.4 "Where it still fails" — 실패 분류 (E4-F)

E1a AG3S arm의 **모든** 충돌 episode와 무충돌 실패 episode를 라벨링한다(원고의 46 / 31 / 23 %, 79 / 14 %는 투영).

원고(4.4 B안)는 두 축으로만 보고한다: **perception-side**(표현이 장면과 어긋남) 대 **action-side**(생성 뒤 정제의 한계). supplement Tab. S9에는 아래 세부 분류를 전부 싣는다.

| 축 | 분류 | 판정 근거 (자동 + 수동) |
|---|---|---|
| P1 | 미관측 표면 (뒷면, depth hole) | 충돌 지점이 TSDF에 관측되지 않은 voxel인지 |
| P2 | target identity가 장면과 어긋남 (가림 · 닿은 이웃 · 잘못된 첫 reference · 목적지 오등록) | memory의 target id vs GT target id; DestinationRegistry 첫 등록 body vs GT 목적지(dev-38 보고 2026-11-08: LIBERO t9 · SafeLIBERO t3에서 cabinet이 접시 대신 등록) |
| P3 | hand mesh가 collision sphere 밖 | 충돌 시점 sphere–mesh 거리(T58 진단 script 재사용) |
| A1 | hold timeout (corridor < margin, 새 접근 pose 없음) | episode 종료 사유 = hold, 마지막 hold 위치의 corridor 폭 |
| A2 | 수정된 chunk가 grasp · place pose를 흐트러뜨림 (grasp 이탈, 놓기 위치 어긋남) | grasp 시도 후 미파지 / 놓은 위치 오차; nominal 대비 deflection 크기 |
| A3 | TO infeasible → nominal 실행 후 충돌 | solver status 기록 |

원고 문장의 비율(perception 71 % of collisions = 46/17/8; action = 나머지 충돌 + 무충돌 실패의 93 % = hold 79 / deflected 14)은 투영값이다.

산출: episode별 라벨 표, 분류별 대표 replay 영상 2개씩(supplement).

### 1.8 4.4 마지막 문단 — 일반화 · run time (E6)

- Held-out: E1a/E2에서 development · 임계값 선택에 쓰지 않은 물체 · 배치 · occlusion 조건을 **사전에** 목록화하고, 그 조건에서의 SSR 하락폭을 AG3S · KNOWS* · external에 대해 산출. split provenance 저장.
- Run time: observation → returned chunk 경계, 1 job 독점, 첫 request(JIT) 제외, 2,000 request 이상. 단계별(attention read-out, surface graph, ESDF, TO) p50/p95, 실행 창(deployed chunk 기준, 현재 투영 400 ms) 초과 비율. baseline(AEGIS*, external)도 같은 경계로. **현재 prototype은 모든 arm이 예산을 넘었으므로(CLAIMS C007) 투영 290/410 ms는 최적화(그래프 GPU화 · backbone 공유) 뒤에만 가능** — 안 되면 원고 문장을 "cost statement"로 바꾼다.

### 1.9 5장 Conclusion

"충돌 절반", "task success 2 pt 내 유지", "hold에서 비롯"은 전부 E1a · E4-F 결과로 대체. future work 인용(SpatialVLA · PointVLA · ActiveVLA · constrained FM · OmniGuide · GuidedVLA)은 서지 1차 출처 재확인.

## 2. 공통 규약 (모든 실험)

1. checkpoint · evaluator · controller · action 변환 · horizon · seed · initial state 분할을 run manifest에 기록. Panda 7-D run용 manifest 형식을 새로 정한다(현재 schema는 RB-Y1 16-D 전용).
2. 모든 arm은 같은 (scenario, init, seed) → paired. 실패 · timeout · QP 실패 · crash를 버리지 않음(crash는 `UNKNOWN`, 같은 seed 1회 재실행, 기록).
3. dev initial state에서 고른 모든 임계값 · head · flag를 `LOCK_eval.json`에 hash와 함께 동결. eval 분할의 결과를 본 뒤 설정을 바꾸면 **새 실험 ID**.
4. 통계: 비율 Wilson 95 % CI, paired 차이 cluster bootstrap 2,000회(고정 seed) + exact McNemar, 다중 비교 보정 없음(전부 나열). 비열등 · 동등 주장 없음.
5. GPU: 공용 lock 규칙(TO 포함 run 독점, VLA 단독 · 추출은 공유 2 job), timing run은 독점 1 job. 긴 작업 전후 다른 세션에 알림.
6. 기록: 결과마다 `AG3S-VLA_EXPERIMENT_RESULTS_KO.md` §10 양식(실행 ID, 상태, 조건, 원본 경로 · hash, 집계 script). 원고 수치는 `NUMBER_MAP`에 (표 셀 → result ID → JSON pointer)로 연결한 뒤에만 검정으로 바꾼다.

## 3. 실험별 protocol 요약

### 3.1 S0 — 설정 동결
- 최종 AG3S non-GT 구성 flag 확정(B5의 GT perception 입력을 모두 ATS · memory 출력으로 대체했는지 확인).
- dev(0–4)에서 attention head 후보표(layer × head × 물체 위 peak 비율 · dev IoU), identity layer, 임계값 선정 → `LOCK_eval.json`.
- 산출: supplement Tab. S1(hyperparameter 선정표).

### 3.2 E1a — SafeLIBERO + LIBERO
- §1.2. 명령 초안은 E1 protocol §2.6. 순서: pilot(5–14) 9 arm → headline 4 arm 전체(5–49) → 여유 시 나머지.
- 추가 기록 flag: `end_reason`, 충돌 row의 (link, body, named?) 정보, CSC용 접촉 step 수(가능하면).

### 3.3 E1b — LIBERO-Safety
- §1.3. H1 · H2 · B1 해결 전에는 `official_comparable: False` 진단만. 해결 후 600 ep × 4 arm.

### 3.4 E5 — Ablation
- §1.5. pilot 분할 320 ep × 7 변형 + full. 변형 구현 2개(geometry-rule seed, external mask 경로)는 E1a의 external 행과 공유.

### 3.5 E2/E3 — Grounding · identity
- §1.6. GT 생성기 검증(수동 20 frame 대조) 후 실행. 외부 모델은 CPU/GPU · 버전 기록.

### 3.6 E4-F — 실패 분류
- §1.7. 자동 분류 script + 수동 확인 2명, 일치율 기록.

### 3.7 E6 — 일반화 · run time
- §1.8. 최적화 전후 timing을 둘 다 남긴다(원고는 최종값만).

### 3.8 E7 — 실로봇
- §1.4. 하드웨어 · calibration · 안전 절차 문서 → pilot 10 trial → 최종 횟수 확정 → 320 trial.

## 4. 우선순위와 시간 (ESTIMATED)

| 순서 | 항목 | 표본 | 추정 시간 | 비고 |
|---|---|---|---|---|
| 1 | S0 | — | 1–2일 | 구현 동결 필요 |
| 2 | E1a pilot 9 arm | 320 ep × 9 | 3–5일 (GPU 1–2 job) | 방향 판정 → 원고 수정 |
| 3 | E5 | 320 × 7 | 2–4일 | TO 포함, 독점 |
| 4 | E1a headline 4 arm 전체 | 1,440 × 4 (+LIBERO) | 5–8일 | AG3S latency에 좌우 |
| 5 | E2/E3 | 장면 수에 따라 | 2–3일 | GT 생성기 선행 |
| 6 | E4-F | E1a 결과 | 1–2일 | |
| 7 | E6 | split + timing | 1–2일 | 최적화 뒤 |
| 8 | E1b | 600 × 4 | 2–3일 + 선행 판정 | H1 다운로드 12.4 GB |
| 9 | E7 | 320 trial | 3–5일 | 하드웨어 |
| 10 | SUP | — | 2–3일 | 아래 §6–7 |

시간은 E1 protocol의 실측(57 s/ep)과 현재 latency에서 외삽한 추정이며 실측 후 갱신한다.

## 5. 산출물 경로

- run 원본: `AG3S_VLA-storage/results/benchmark/<날짜_commit_run-id>/{manifest.json, episodes.jsonl, timing.json, video/}`.
- 집계: `paper/experiments/summaries/<EXP>_summary.json` + script + 입력 hash.
- 사람이 볼 영상 · 그림: `outputs/<YYYY-MM-DD>_<실험>/` (`python -m common.experiment_view`), 논문 figure는 `paper/figures/cvpr_v2/`.
- 등록: `RESULTS.md`(R0xx), `CLAIMS.md`(C0xx 상태), `NUMBER_MAP_CVPR_V2.md`(표 셀 → pointer).

## 6. Supplementary에 넣을 자료 (본문 실험과 함께 수집)

| ID | 내용 | 출처 실험 |
|---|---|---|
| Tab. S1 | hyperparameter 선정표: attention layer × head 후보, identity layer, 모든 임계값, dev 분할 근거 | S0 |
| Tab. S2 | SafeLIBERO suite × level × task × arm 전체 결과(TSR · SSR · CAR · ETS · SBU · CNF, n, Wilson CI) | E1a |
| Tab. S3 | 모든 arm 쌍의 paired 차이 + bootstrap CI + McNemar p (다중 비교 보정 없음 명시) | E1a |
| Tab. S4 | LIBERO(장애물 없음) suite별 SR | E1a |
| Tab. S5 | LIBERO-Safety 전체: 4 suite × L0–L2, HRI, CR · 실행 시간 · LDLJ; `official_comparable` 여부 | E1b |
| Tab. S6 | ablation suite별 분해 + "distractor가 더 가까운 initial state 비율" | E5 |
| Tab. S7 | grounding 전체: 장면 유형별 · 외부 모델 prompt/confidence별(지시문 vs 정답 이름), abstain | E2 |
| Tab. S8 | identity: occlusion 구간 정의, ID switch · re-ID 분모, 시퀀스별 | E3 |
| Tab. S9 | 실패 분류 전수표(episode ID, 분류, 근거 값) | E4-F |
| Tab. S10 | held-out 조건 목록 · 분할 hash · 조건별 SSR | E6 |
| Tab. S11 | run time, **정책 추론 포함 같은 경계(observation → 실행 가능한 chunk)**: π0.5 단독 / + AEGIS(공식, CBF-QP per step) / + external segmentation / + AG3S, p50·p95; AG3S 단계별(attention read-out · surface graph · ESDF 2장 · refinement); 최적화 전후 두 열 | E6 |
| Tab. S12 | 실로봇 trial별 결과, F/T 임계, annotator 일치율, 물체 목록(unseen 증명) | E7 |
| Tab. S13 | AEGIS: 공식 코드 우리 환경 실행값 vs 원문 보고값 vs port(chunk 단위) 대조; KNOWS*: 재현값 vs 보고값; SafeDojo Tab. 1–2 OFT 블록(본문에서 이동) | E1a |
| Tab. S14 | 민감도: dev에서 다음 순위 attention head 3개의 held-out IoU 변화; sphere margin ±5 mm의 SSR · CAR 변화(pilot 분할 320 ep); depth noise σ ∈ {0, 5, 10 mm} | E2, E5 |
| Tab. S15 | 지시문 변경 scene pair: 같은 장면 · 로봇 상태에서 지시만 바꿨을 때 target · surface · 제약이 새 물체로 옮겨간 쌍 수 n/N | E2 |
| Fig. S1–S3 | 분류별 실패 replay 2개씩; grounding 추가 예시; 실로봇 접촉 순간 | E4-F, E2, E7 |
| Fig. 1 (본문) | **v3 (2026-11-09): 기존 Fig. 1 + Fig. 2 병합.** 세 축(identity · extent · permission)마다 Prior(Tab. 1 baseline 실제 출력) vs AG3S, 위쪽 SafeLIBERO Level I 장면 줄은 선택. 구성안 `paper/figures/fig01_teaser/TEASER_V2_SAFELIBERO_PLAN_KO.md` | E1a eval run, E2 grounding run |
| Video | 실로봇 4 task × 2 method 대표 trial, SafeLIBERO Level I/II 대표 episode(π0.5 vs AG3S 나란히), wrist-view overlay | E1a, E7 |
| 문서 | 재구현 baseline의 원 논문과의 차이(GLM 단계 대체, chunk 단위 filter 등), 접촉 정책 `(stage, link, object, contact type)` 표, 측정 경계 그림 | 전체 |

## 7. 리뷰어가 요청할 가능성이 높은 자료 (미리 수집)

| 예상 요청 | 미리 준비할 것 | 비고 |
|---|---|---|
| "AEGIS 수치가 원문(67.5)과 다르다. 왜?" | 주 행이 **공식 코드** 실행이므로 구현 의심은 제거됨. Tab. S13: 공식 코드 우리 환경 실행값 vs 원문 보고값 vs port, 같은 scenario · seed; 차이 원인(GLM prompt · 버전, horizon, initial state 분할) | 가장 먼저 올 질문 |
| "KNOWS* · multi-link*는 재구현인데 공정한가?" | 공식 코드 없음을 README · project page 기록으로 증명; KNOWS 보고값(Tab. 1 DERIVED 평균)과 재현값을 Tab. 1 안에서 나란히 비교(76.4 vs 76.0 CAR 등) | |
| "KNOWS와 무엇이 다른가? attention 쓰는 건 같다" | Tab. S7의 extent 비교(KNOWS segmenter vs 우리 graph), Tab. 3 "graph → external mask" 행, Fig. 5 | |
| "TSR이 π0.5보다 낮다. 안전 레이어가 성능을 깎는 것 아닌가" | TSR−SSR / CAR−SSR 분해, hold timeout 비율, LIBERO 무장애물 SR, 95 % CI | 본문 4.2에 이미 틀 있음 |
| "유의한가? 표본이 충분한가?" | Tab. S3 CI · McNemar, 사전 고정 표본 근거(§8) | |
| "attention head 하나에 의존하면 다른 checkpoint에서 되나?" | Tab. S1 후보표 + Tab. S14 head 민감도(dev 다음 순위 3개로 Tab. 4 재실행; 본문 4.4에 한 문장) | 추가 실험 소형, **본문 반영됨** |
| "sphere margin에 민감한가?" | Tab. S14 margin ±5 mm의 SSR · CAR (pilot 320 ep × 2) | 추가 실험 소형, 본문 한 문장 |
| "depth 노이즈 · calibration 오차에 민감한가?" | E2에서 depth noise σ ∈ {0, 5, 10 mm}, extrinsic 오차 ±5 mm/±1° 민감도 표 | 추가 실험 소형 |
| "privileged 정보를 쓰지 않았다는 증거" | AG3S non-GT 구성 flag 목록, B5(GT) oracle 행과의 분리, LOCK 파일 | |
| "dev/test 누수" | 분할 hash, 임계값 선택 로그 날짜 < eval 실행 날짜 | |
| "실시간인가?" | Tab. S11, 실행 창 정의, 초과 비율; 안 되면 cost statement | |
| "실로봇 안전 판정이 주관적" | F/T 로그 + annotator 2명 일치율 + 영상 전부 공개 | |
| "1,600 episode가 왜 필요한가 / 왜 1,440인가" | §8 표본 근거, dev 5개 제외 이유 | |
| "code · checkpoint 공개" | 익명 repo 준비(코드, LOCK, 집계 script, manifest), 공개 범위 결정 | 제출 전 |
| "다른 base policy(OpenVLA-OFT)에서도 되나?" | 범위 밖임을 Limitations에 명시; 가능하면 Spatial suite만 OFT로 소형 실험 | 선택 |
| "Level II 일반화 주장의 근거" | 학습 없음(제약은 관측에서) + Tab. S10 held-out 조건 | |
| "오라클과의 차이가 왜 남나" | E4-F 분류(미관측 표면 비율), Tab. 4 oracle 행 | |

## 8. 표본 수의 근거

- SafeLIBERO 공식 규모가 32 scenario × 50 initial state = 1,600/method(VLSA)이며 SafeDojo · KNOWS도 같은 규모(task당 50 episode)를 쓴다. LIBERO 표준 평가도 task당 50 trial(4 suite 2,000 rollout). 1,600은 "많이"가 아니라 benchmark 규약이다.
- 통계적으로: scenario 하나(50 ep)의 비율 95 % CI 폭은 약 ±14 pt라 scenario 단위로는 아무것도 못 가르고, 1,440 pooled면 약 ±2.5 pt, paired 차이는 그보다 좁다. 본문이 말하는 차이(TSR 0.9–2.3 pt, SSR 5 pt, CAR 13 pt)를 가르려면 pooled 규모가 필요하다.
- dev 5개(0–4)를 뺀 1,440을 보고하고, 공식 0–49 전체 1,600도 supplement에 같이 둔다(dev 포함 수치와 분리 표기).
- 실로봇 20 trial/task는 ±20 pt CI라 task별 차이는 못 가르고 pooled(80 trial/method)만 가른다. 본문에서 task별 수치는 기술로만 쓴다.

## 9. 완료 체크리스트

- [ ] S0 LOCK 파일 · Tab. S1
- [ ] E1a pilot 9 arm → 투영 대비 방향 판정 → 원고 수정
- [ ] E5 7 변형
- [ ] E1a headline 4 arm 전체 분할
- [ ] E2/E3 GT 생성기 검증 → 실행 → Fig. 5 자료
- [ ] E4-F 전수 라벨링 → Tab. S9, Fig. S1
- [ ] E6 split 목록 · timing(최적화 전후)
- [ ] E1b 선행 판정 3개 → 600 × 4
- [ ] E7 안전 절차 → pilot → 320 trial → 영상 · 로그
- [ ] AEGIS 공식 코드 대조(Tab. S13)
- [ ] 민감도 2종(head, depth noise)
- [ ] NUMBER_MAP_CVPR_V2: 원고 빨간 글씨 165곳 → result ID 전부 연결 후 검정으로 변경
- [ ] 익명 code/data 공개본

## 10. 채워 넣을 표 · 실험 내용 · 예상 결과 (원고 Tab. 1–4, Fig. 4–5, 본문 수치)

아래 표의 수치는 **원고 v3의 투영값(목표)** 이다. 각 칸은 `결과 파일 → 집계 script → NUMBER_MAP` 으로 연결된 뒤에만 검정 글씨가 된다. "예상 결과"는 가설이고, "판정"은 결과가 가설과 다를 때 원고를 어떻게 고칠지 미리 정한 규칙이다.

### 10.1 Tab. 1 — LIBERO + SafeLIBERO (E1a)

**실험 내용.** 같은 frozen π0.5-LIBERO · 같은 공식 evaluator · 같은 (suite, level, task, init 5–49, seed 7)에서 9개 arm을 돌린다. LIBERO 열은 장애물 없는 원 suite(4 suite × 10 task × init 5–49). SafeLIBERO는 4 suite × 4 task × Level I/II. 집계는 suite 동일 가중 평균. 채울 셀 = 9 arm × 7 열 = 63개(+ supplement의 ETS · SBU · CNF · suite별).

| arm | 입력 | LIBERO SR | L-I TSR | L-I SSR | L-I CAR | L-II TSR | L-II SSR | L-II CAR | 결과 파일 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| (†) π0.5 alone, VLSA 보고 | – | – | 57.8 | – | 17.1 | (level 합산) | | | 원문 Tab. I |
| (†) AEGIS, VLSA 보고 | VLM + detector | – | 67.5 | – | 68.9 | (level 합산) | | | 원문 Tab. I |
| (†) π0.5 alone (No CBF), KNOWS 보고 · DERIVED 평균 | – | – | 55.0 | 11.5 | 12.4 | 57.8 | 20.0 | 23.0 | 원문 Tab. 1 |
| (†) Naive (GT seg.), KNOWS 보고 · DERIVED | privileged | – | 73.0 | 63.1 | 82.2 | 67.8 | 55.6 | 76.1 | 원문 Tab. 1 |
| (†) KNOWS, KNOWS 보고 · DERIVED | attention + segmenter | – | 67.5 | 59.9 | 76.4 | 71.1 | 53.8 | 68.8 | 원문 Tab. 1 |
| π0.5 alone | – | 96.8 | 57.8 | 13.5 | 18.6 | 59.4 | 19.0 | 25.2 | `e1a-none-*` |
| + generic geometry | depth | 93.9 | 42.3 | 39.1 | 88.4 | 44.0 | 41.2 | 90.1 | `e1a-generic-*` |
| + external segmentation | depth + mask | 95.0 | 60.0 | 46.5 | 71.0 | 61.5 | 48.0 | 74.0 | `e1a-extseg-*` |
| AEGIS (official code) | VLM + detector | 96.5 | 66.5 | 49.8 | 69.0 | 67.0 | 51.0 | 72.0 | `e1a-aegis-official-*` |
| AEGIS (official code, GT) | privileged | 96.5 | 67.5 | 52.5 | 73.5 | 68.0 | 53.5 | 75.5 | `e1a-aegis-official-gt-*` |
| Multi-link CBF* | VLM + detector | 96.1 | 65.0 | 50.0 | 72.5 | 66.0 | 51.5 | 75.0 | `e1a-mlcbf-*` |
| KNOWS* | attention + segmenter | 95.8 | 66.5 | 55.0 | 76.0 | 69.0 | 53.0 | 70.0 | `e1a-knows-*` |
| **AG3S-VLA** | depth + VLA internals | 95.4 | **66.0** | **61.5** | 87.5 | **68.0** | **63.0** | 89.0 | `e1a-ag3s-*` |

**예상 결과(가설).** (a) 우리 π0.5 단독 재현은 보고값 사이(TSR 55–58, CAR 12–17)에 들어온다 → harness 신뢰도. (b) 모든 safety layer가 π0.5보다 TSR을 올린다(보고값과 같은 방향); AG3S는 CAR·SSR 최고, TSR은 최강 layer의 −1.5 pt 이내, LIBERO 무장애물에서만 π0.5보다 −1.4 pt. (c) 최강 비특권 baseline(KNOWS*) 대비 충돌률(1−CAR) 약 절반(24.0 → 12.5). (d) generic geometry는 CAR 최고·TSR 최저. (e) KNOWS는 보고값에서도 Level I → II에서 SSR이 떨어지고(59.9 → 53.8) AG3S는 올라 Level II 격차가 커짐.

**판정.** (a)에서 재현 π0.5가 두 보고값 범위를 5 pt 이상 벗어나면 evaluator · horizon · checkpoint를 먼저 점검(결과 해석 전). (b)에서 AG3S TSR이 최강 layer보다 3 pt 넘게 낮으면 "within 1.5 points"를 실제 값으로 바꾸고 hold timeout 비율을 함께 보고; SSR이 최고가 아니면 scoped SOTA 문장 전체 삭제. (c)에서 CAR 우위가 5 pt 미만이면 "half"를 삭제. AEGIS 공식 코드 TSR이 보고값(67.5)과 5 pt 넘게 다르면 GLM prompt · horizon 차이를 Tab. S13에 기록하고 본문은 우리 환경값으로 쓴다.

### 10.2 Tab. 2 — LIBERO-Safety L2 (E1b)

**실험 내용.** 공식 `pi05_libero_safety` weight(H1) · controller 동등성(H2) · cost 판정(B1) 확인 뒤, 물리 4 suite × L0–L2 × 5 task × 10 trial. 본문은 AAG · TSA · FSHOA L2 + pooled CR. 채울 셀 = 4 arm × 4 열.

| arm | AAG | TSA | FSHOA | CR | 결과 파일 |
|---|---:|---:|---:|---:|---|
| π0.5 alone | 32.7 | 54.0 | 49.3 | 26.0 | `e1b-none-*` |
| + generic geometry | 19.3 | 57.3 | 56.0 | 8.7 | `e1b-generic-*` |
| + external segmentation | 29.3 | 56.7 | 53.3 | 14.0 | `e1b-extseg-*` |
| **AG3S-VLA** | **37.3** | **62.0** | **60.7** | **6.7** | `e1b-ag3s-*` |

**예상 결과.** AAG(허용 접촉이 필요한 파지)에서 generic은 크게 하락, AG3S는 상승; TSA·FSHOA(순수 회피)에서 AG3S ≥ generic; CR은 AG3S 최저. **판정.** AAG에서 AG3S < π0.5이면 "allowed-versus-forbidden split" 문장을 삭제하고 원인(파지 허용 시점)을 4.4 실패 분류에 추가. π0.5 row가 보고값(35.3/56.7/51.3)보다 높게 나오면 "single training seed" 문장 삭제.

### 10.3 Fig. 4 + 본문 — 실로봇 (E7)

**실험 내용.** 4 task × 4 method × 20 trial. 채울 값: task별 · method별 task success / safe success / unsafe contact(%).

| method | task success | unsafe contact | unseen-object task: success / unsafe |
|---|---:|---:|---|
| π0.5 alone | 71.3 | 45.0 | 65.0 / 42.5 |
| + generic geometry | 55.0 | 11.3 | (supplement) |
| + external segmentation | (supplement) | (supplement) | (supplement) |
| **AG3S-VLA** | 68.8 | **8.8** | 62.5 / 10.0 |

**예상 결과.** unsafe contact가 45 → 9 %로 감소, task success는 −2.5 pt. unseen 물체에서도 같은 패턴. **판정.** 80 trial pooled의 Wilson CI가 겹치면 "feasibility study"로만 쓰고 수치 비교 문장을 제거. unsafe contact 판정의 annotator 일치율 < 0.8이면 F/T 센서 기준만 보고.

### 10.4 Tab. 3 — Ablation, SafeLIBERO Level I (E5)

**실험 내용.** full과 7 변형을 pilot 분할(init 5–14, 320 ep)에서 paired. 채울 셀 = 8 행 × 4 열(ΔSSR은 bootstrap CI 포함).

| 구성 | TSR | SSR | CAR | ΔSSR [95 % CI] | 예상 메커니즘 |
|---|---:|---:|---:|---|---|
| **full** | 66.0 | **61.5** | 87.5 | – | |
| attention → nearest object | 62.0 | 54.8 | 81.5 | −6.7 [−9.1, −4.2] | Object suite에서 distractor가 더 가까운 init에서 오선택 |
| graph → radius growth | 65.0 | 54.3 | 77.8 | −7.2 [−9.8, −4.5] | 닿은 장애물이 freed region에 포함 → CAR 하락 |
| graph → external mask | 60.0 | 51.2 | 72.0 | −10.3 [−13.0, −7.4] | 같은 원인 + prompt 의존 |
| memory off | 64.5 | 57.4 | 83.5 | −4.1 [−6.2, −1.9] | 파지 순간 가림에서 이웃으로 점프 |
| target always obstacle | 36.0 | 34.5 | **93.0** | −27.0 [−30.5, −23.2] | 파지 금지 → TSR 반토막 |
| target freed, all links/stages | **66.5** | 55.5 | 77.0 | −6.0 [−8.4, −3.5] | 손목·운반 물체가 target 주변으로 진입 |
| carried object unattached | 65.5 | 54.6 | 78.0 | −6.9 [−9.3, −4.4] | 운반 단계 물체–장애물 충돌 |

**예상 결과.** 효과 크기 순서: target always obstacle ≫ external mask > radius growth ≈ carried unattached ≈ nearest object ≈ all-links > memory off. **판정.** CI가 0을 포함하는 변형은 본문에서 "needed"라 쓰지 않고 "not separable on this set"으로 바꾼다(현재 link-specific 항목이 가장 위험). 순서가 바뀌면 4.3 세 문단의 순서와 "largest effect" 문장을 결과에 맞춘다.

### 10.5 Tab. 4 + Fig. 5 — Grounding · identity, held-out 진단 장면 (E2/E3)

**실험 내용.** 진단 장면 4종 + occlusion 시퀀스, 시뮬레이터 GT, 5 mm soft IoU, frame → episode → task 집계. 채울 셀 = 7 행 × 5 열.

| method | IoU | Prec. | Rec. | ID acc. | ID switch/seq | 예상 실패 원인 |
|---|---:|---:|---:|---:|---:|---|
| geometry, nearest-to-hand | 0.44 | 0.61 | 0.58 | 0.63 | – | 지시문 무시 → 가까운 distractor |
| YOLOE (지시문 noun) | 0.35 | 0.52 | 0.49 | 0.70 | 5.0 | 검출 누락(기존 dev 0.06–0.12) |
| Grounded-SAM-2 (지시문 noun) | 0.48 | 0.55 | 0.74 | 0.81 | 4.2 | 닿은 이웃 병합, 로봇 가림 |
| Grounded-SAM-2 (정답 이름) | 0.61 | 0.66 | 0.80 | 0.90 | 3.1 | prompt 의존 (−0.13 IoU) |
| oracle target | 0.80 | 0.93 | 0.85 | 1.00 | 0.0 | 관측 한계만 |
| AG3S w/o memory | 0.69 | 0.88 | 0.76 | 0.90 | 2.8 | 가림에서 identity 점프 |
| **AG3S-VLA** | **0.72** | **0.90** | 0.78 | **0.96** | **0.4** | |

- 지시문 변경 scene pair(Tab. S15): 같은 장면 · 로봇 상태에서 지시만 바꿔 target · surface · 제약이 새 물체로 옮겨가는 비율. RQ1(instruction-conditioned grounding)의 가장 직접적인 증거이며 4.4에 한 문장으로 들어감. 목표 ≥ 90 %; 실패 쌍은 attention이 두 물체에 걸치는 경우로 예상.
- 민감도(Tab. S14): dev 다음 순위 head 3개 → held-out IoU 변화 ≤ 0.02, margin ±5 mm → SSR ±1.1 pt가 목표. 변화가 크면 4.4의 "not fragile" 문장을 삭제하고 선택 절차(dev 고정)만 남긴다.

**예상 결과.** precision이 차이를 만든다(0.90 vs 0.55); recall은 비슷; memory는 IoU가 아니라 switch를 줄인다; oracle과의 차이 ≈ 0.08. **판정.** AG3S IoU < SAM-2(정답 이름)이면 "beats an external segmenter"를 "without an external segmenter"로 바꾸고 비교를 지시문 prompt 조건에 한정. 기존 dev(Object suite, 2026-11-01)에서 noun+FK는 0.36–0.41, SAM-2 0.21이었으므로 **절대값은 장면 구성에 크게 좌우된다** — 목표 0.72는 held-out 장면 설계가 끝난 뒤 재조정.

### 10.6 4.4 본문 수치 — 실패 분류 · 일반화 · run time (E4-F, E6)

| 원고 수치 | 뜻 | 산출 방법 | 판정 |
|---|---|---|---|
| perception-side 71 % of collisions (미관측 46 / identity 17 / sphere 밖 8) | AG3S Level I 충돌 원인 (P1–P3) | E1a AG3S 충돌 episode 전수 라벨(§1.7) | 분포가 달라도 두 축 문장 유지, 수치만 교체 |
| action-side: 나머지 충돌 + 무충돌 실패의 93 % (hold 79 / deflected pose 14) | CAR−SSR의 원인 (A1–A3) | `end_reason`, deflection, solver status | hold 비율 < 50 %면 5장 action 축 future work의 근거를 약화해 고침 |
| AEGIS* 잔여 충돌 61 %가 이름 없는 clutter · 다른 hand link · 운반 물체 (forearm 아님 — 3.4는 palm·finger sphere만 제약) | CBF 범위 한계 | 충돌 row의 (link, body, named?) | < 40 %면 "two mechanisms" 중 첫째를 축소 |
| held-out SSR 하락 AG3S −1.8 / KNOWS* −7.5 / external −9.1 | 일반화 | E6 split별 SSR | AG3S 하락이 baseline과 같으면 "generalization" 문단 삭제 |
| run time, 정책 포함 경계: AG3S 490 / 620 ms vs AEGIS 205 vs external 560 | **비용 보고**(실시간 주장 아님) | E6 timing, 독점 1 job, 정책 추론 포함 | 어떤 값이 나와도 문장 구조 유지("one to two orders of magnitude more than an ellipsoid filter"); AG3S가 external보다 느리면 그 비교 구를 삭제 |
| 단계 비용: attention 6 · graph 95 · ESDF 80 · refinement 110 ms | 비용 분해 | timing.json 단계별 | 수치만 교체 |

### 10.7 4.1 설정 문장

| 원고 문장 | 채울 값 | 출처 |
|---|---|---|
| layer 8, head 6 / prefix layer 8 | S0 선정표 | `LOCK_eval.json`, Tab. S1 |
| init 0–4 dev / 5–49 eval, 1,440 ep | 분할 | split hash |
| RB-Y1, D435 3대 / F/T 5 N, 외부 카메라 2대, annotator 2명 | 하드웨어 · 판정 | E7 setup 문서 |
| NVIDIA H200 1장 | timing 하드웨어 | E6 manifest |

### 10.8 5장 문장

"halves the collisions", "highest safe-success rate", "preserving task success within about two points", "remainder traced to holds" → 각각 10.1(c), 10.1(b), 10.6 hold 비율에 연동. 셋 중 하나라도 판정에서 탈락하면 5장 첫 문단을 해당 결과로 다시 쓴다.

## 11. 투영 대 실측 대조 기록 (결과가 나올 때마다 채움)

§10의 목표값과 실제 측정값을 한 줄씩 대조한다. "차이"는 실측 − 투영. "판정 적용"은 §10의 규칙 중 무엇을 적용했는지, "원고 수정"은 바뀐 문장 위치. 실측 열은 result ID(R0xx)와 실행 ID를 함께 적는다. 투영값은 지우지 않는다 — 우리가 무엇을 추구했는지의 기록이다.

| # | 항목 (원고 위치) | 투영 | 실측 (R-ID / 실행 ID) | 차이 | 판정 적용 | 원고 수정 | 기록일 |
|---|---|---:|---|---:|---|---|---|
| 1 | π0.5 alone L-I TSR / CAR (Tab. 1; 보고 VLSA 57.8 / 17.1, KNOWS DERIVED 55.0 / 12.4) | 57.8 / 18.6 | | | 10.1 (a) | | |
| 2 | AG3S L-I TSR / SSR / CAR (Tab. 1) | 66.0 / 61.5 / 87.5 | | | 10.1 (b)(c) | | |
| 3 | AG3S L-II TSR / SSR / CAR (Tab. 1) | 68.0 / 63.0 / 89.0 | | | 10.1 (e) | | |
| 4 | AG3S LIBERO SR vs π0.5 (Tab. 1) | 95.4 vs 96.8 | | | 10.1 (b) | | |
| 5 | KNOWS* L-I TSR / SSR / CAR (Tab. 1; 보고 DERIVED 67.5 / 59.9 / 76.4) | 66.5 / 55.0 / 76.0 | | | 10.1 (c) | | |
| 6 | AEGIS official (GT) L-I TSR / CAR (Tab. 1; 보고 67.5 / 68.9 level 합산) | 67.5 / 73.5 | | | 10.1 AEGIS 규칙 | | |
| 7 | AEGIS 공식 코드 우리 환경 vs 원문 보고 (Tab. S13) | 일치 ±5 pt | | | | | |
| 8 | generic geometry L-I TSR / CAR (Tab. 1) | 42.3 / 88.4 | | | 10.1 (d) | | |
| 9 | external segmentation L-I TSR / SSR / CAR (Tab. 1) | 60.0 / 46.5 / 71.0 | | | | | |
| 10 | 충돌률 비: AG3S / 최강 비특권 baseline | 12.5 / 24.0 | | | 10.1 (c) | | |
| 11 | LIBERO-Safety AG3S AAG / TSA / FSHOA / CR (Tab. 2) | 37.3 / 62.0 / 60.7 / 6.7 | | | 10.2 | | |
| 12 | LIBERO-Safety π0.5 (Tab. 2) | 32.7 / 54.0 / 49.3 / 26.0 | | | 10.2 | | |
| 13 | 실로봇 AG3S success / unsafe (Fig. 4) | 68.8 / 8.8 | | | 10.3 | | |
| 14 | 실로봇 π0.5 success / unsafe (Fig. 4) | 71.3 / 45.0 | | | 10.3 | | |
| 15 | 실로봇 unseen task AG3S / π0.5 | 62.5·10.0 / 65.0·42.5 | | | 10.3 | | |
| 16 | ablation ΔSSR: nearest object (Tab. 3) | −6.7 [−9.1, −4.2] | | | 10.4 | | |
| 17 | ablation ΔSSR: radius growth | −7.2 [−9.8, −4.5] | | | 10.4 | | |
| 18 | ablation ΔSSR: external mask | −10.3 [−13.0, −7.4] | | | 10.4 | | |
| 19 | ablation ΔSSR: memory off | −4.1 [−6.2, −1.9] | | | 10.4 | | |
| 20 | ablation ΔSSR: target always obstacle | −27.0 [−30.5, −23.2] | | | 10.4 | | |
| 21 | ablation ΔSSR: all links/stages | −6.0 [−8.4, −3.5] | | | 10.4 CI 규칙 | | |
| 22 | ablation ΔSSR: carried unattached | −6.9 [−9.3, −4.4] | | | 10.4 | | |
| 23 | grounding AG3S IoU / Prec / Rec (Tab. 4) | 0.72 / 0.90 / 0.78 | | | 10.5 | | |
| 24 | grounding Grounded-SAM-2 지시문 / 정답 이름 IoU | 0.48 / 0.61 | | | 10.5 | | |
| 25 | grounding oracle IoU | 0.80 | | | 10.5 | | |
| 26 | identity switch/seq AG3S / w/o memory / SAM-2 | 0.4 / 2.8 / 4.2 | | | 10.5 | | |
| 27 | identity accuracy AG3S / w/o memory | 0.96 / 0.90 | | | 10.5 | | |
| 28 | 충돌 원인: perception-side 비율 (미관측 / identity / sphere 밖) (4.4) | 71 % (46 / 17 / 8) | | | 10.6 | | |
| 29 | action-side: 무충돌 실패 중 hold timeout / deflected pose (4.4) | 93 % (79 / 14) | | | 10.6 | | |
| 30 | AEGIS* 잔여 충돌 중 비명명 clutter · 다른 hand link · 운반 물체 | 61 % | | | 10.6 | | |
| 31 | held-out SSR 하락 AG3S / KNOWS* / external | −1.8 / −7.5 / −9.1 | | | 10.6 | | |
| 32 | run time, 정책 포함 경계: AG3S p50 / p95 (4.4) | 490 / 620 ms | | | 10.6 | | |
| 33 | run time 단계: attention / graph / ESDF×2 / refinement | 6 / 95 / 80 / 110 ms | | | 10.6 | | |
| 34 | run time 같은 경계: π0.5 + AEGIS(official) / + external seg. | 205 / 560 ms | | | 10.6 | | |
| 35 | attention head / identity layer (4.1) | L8 H6 / prefix L8 | | | 10.7 | | |
| 36 | head 민감도: 다음 순위 3 head의 held-out IoU 변화 (4.4) | ≤ 0.02 | | | 10.6 | | |
| 37 | margin ±5 mm의 SSR 변화 (4.4) | ±1.1 pt | | | 10.6 | | |
| 38 | 지시문 변경 scene pair에서 target · 제약 전환 (4.4) | n/N (목표 ≥ 90 %) | | | 10.5 | | |

## 12. "외부 segmenter를 쓰지 않는 이유"와 "실시간성" — 논거별 근거 실험

원고에서 이 논거는 Introduction · Related Work(동기)와 4.4(근거)에 나뉘어 들어간다. 아래는 각 논거를 어떤 실험이 뒷받침하는지, 그리고 **어디까지 말해도 되는지**다.

| # | 논거 | 쓸 수 있는 문장 (근거 실험 뒤) | 근거 실험 · 수치 | 현재 확신도 |
|---|---|---|---|---|
| A | **가려짐 · 닿은 물체에서의 identity** — per-frame segmenter는 로봇이 target을 가리면 마스크가 로봇을 따라가고, 이웃이 닿으면 둘을 합친다. attention + 3D memory는 identity를 시간에 걸쳐 유지하고 extent를 depth 불연속으로 끊는다 | "An external mask spans touching objects and follows the occluding hand; the policy's attention with depth continuity and a temporal memory keeps the target apart and the same" | E2/E3: Tab. 4 precision(0.90 vs 0.55), ID switch(0.4 vs 4.2), Fig. 5 | 높음 — 2026-11-01 dev에서 같은 방향 관측(SAM-2 0.21 vs noun+FK 0.36–0.41) |
| B | **prompt 의존성** — segmenter는 물체의 정식 이름이 필요하고 지시문 noun만 주면 성능이 떨어진다. attention은 정책이 실제로 실행하는 지시문에 조건화되어 있다 | "The segmenter is only as good as its prompt (−0.13 IoU from canonical name to instruction noun); the attention is conditioned on the instruction the policy executes" | E2: Tab. 4 두 prompt 행 | 높음 — dev에서 prompt 민감도 큼 |
| C | **정책과 제약의 일관성** — segmenter가 고른 target과 정책이 향하는 target이 다를 수 있다. 정책 내부에서 읽으면 "정책이 잡으려는 것"을 그대로 푼다 | "The constraint frees the object the policy is about to grasp, not the object a second model believes is meant" | E2 target-ID accuracy + 지시 변경 scene pair 표 | 중간 — 논리는 분명, 수치는 ID accuracy로만 |
| D | **추가 모델 비용** — 외부 segmenter는 loop 안의 두 번째 네트워크다: VRAM, latency, 버전·prompt 관리. attention read-out은 정책 forward pass에서 공짜로 나온다 | "No second perception model in the loop: the attention read-out costs 6 ms on the forward pass that produces the actions, whereas the segmenter adds X ms and Y GB" | E6: timing 단계별 + VRAM 측정(추가: `nvidia-smi` peak per arm) | 높음 — 측정만 하면 됨. **단 '실시간'이라고는 쓰지 않는다**(아래 F) |
| E | **동적 상황 비용** — 장면이 바뀔 때 segmenter는 매 frame 다시 돌려야 하고, 그 비용은 frame rate에 비례한다. attention은 어차피 매 request 나온다 | (확신 없음) "Per-frame re-segmentation scales with the control rate; the attention is produced at every request regardless" | 동적 장애물 실험이 없으면 **비용 scaling 문장만**, 동적 장면 성능 주장은 없음. KNOWS Level III(moving obstacles)를 돌리면 근거 가능 | 낮음 — 사용자도 불확실. 동적 실험 없이는 Limitations에 "static scenes only" 명시 |
| F | **전체 pipeline 비용** — 정밀 표면과 link별 field는 ellipsoid 필터보다 1–2 자릿수 비싸다. 실시간은 주장하지 않고 **정책 추론을 포함한 같은 경계**에서 비용을 보고한다(AEGIS CBF-QP 0.356 ms/step, KNOWS "lower overhead", 학습 기반은 배포 비용 0) | "We report this as a cost, not as real-time capability" (4.4 · 5장 반영 완료, 2026-11-08) | E6: Tab. S11 | 확정 — 사용자 판정: 실시간은 빼고 비용으로 |
| G | **open-set** — 검출기 vocabulary에 없는 물체도 지시문 noun의 attention은 따라간다 | "The attention follows the instruction's noun whether or not a detector has a class for it" | E6 held-out 물체에서 external 하락폭(−9.1)과 AG3S(−1.8) | 중간 — held-out 물체가 실제로 검출기 vocabulary 밖인지 확인 필요 |

### 12.1 "왜 외부 segmenter를 쓰지 않는가?"에 대한 답변 틀 (비용은 근거가 아님)

비용(D)은 KNOWS도 같은 논거(attention은 공짜)를 쓰고 우리 총비용은 geometry가 지배하므로 **차별점이 아니다**. 답은 측정 가능한 네 가지로 한다.

1. **extent — 닿은 물체와 가림.** 이미지 마스크는 닿은 이웃을 한 덩어리로 자르고 로봇이 가리면 로봇을 따라간다. 우리는 depth 불연속에서 멈추고 rendered robot mask로 로봇을 뺀다. → Tab. 4 precision(0.90 vs 0.55), Tab. 3 "graph → external mask" −10.3, Tab. 1 external 행의 CAR.
2. **prompt — 무엇을 segment할지 누가 말해 주나.** segmenter는 물체의 정식 이름이 필요하고 지시문 noun만 주면 −0.13 IoU. attention은 정책이 실제로 실행하는 지시문에 이미 조건화되어 있다. → Tab. 4 두 prompt 행, Tab. S15 지시문 변경 scene pair.
3. **일관성 — 정책이 잡으려는 것을 푼다.** segmenter가 고른 target과 정책이 향하는 target이 다르면 엉뚱한 표면이 풀린다. 정책 내부에서 읽으면 제약과 행동이 같은 물체를 가리킨다. → Tab. 4 target-ID accuracy, 실패 분류 P2.
4. **open-set과 시간.** 검출기 vocabulary 밖 물체도 noun attention은 따라가고(held-out −1.8 vs −9.1), per-frame segmenter에는 identity가 없지만 우리는 memory로 시간에 걸쳐 유지한다(switch 0.4 vs 4.2).

비용은 한 문장만, 동등점으로: "like KNOWS, the attention is read from the forward pass that produces the actions, so no second perception model runs in the loop; the cost of our layer is the geometry (Tab. S11)."

권장 배치: Intro에 A · C · D를 동기로(한 문장씩), Related Work에 B, 4.4에 A · B · D · G의 근거, E는 Limitations("static scenes; per-frame cost argument only"), F는 측정 결과에 따라 4.4 run time 또는 Limitations. "real-time"이라는 단어는 F가 측정으로 확인된 뒤에만 쓴다.

## 13. Fig. 1 v3 (기존 Fig. 1 + Fig. 2 병합)

구성안 · 장면 선택 기준 · 현재 자료 inventory · 제작 절차 · caption 초안은 [`paper/figures/fig01_teaser/TEASER_V2_SAFELIBERO_PLAN_KO.md`](../figures/fig01_teaser/TEASER_V2_SAFELIBERO_PLAN_KO.md). 지금 자료(11-04 같은 장면 baseline 영상, 11-08 AG3S 6-panel 진단 렌더)로는 layout 시안만 가능하고, 최종 그림은 E1a eval 분할의 최종 non-GT 구성 run(칸 2 · 3, 위쪽 줄)과 E2 grounding held-out run(칸 1)에서 칸마다 후보 3개를 뽑아 만든다. 2026-11-09 v3: CVPR 2026 4편의 Fig. 1이 모두 "기존 vs 우리" 개념 비교이고 수치를 넣지 않으므로, 기존 teaser와 세 축 개념 그림을 한 장으로 합치고 파이프라인을 Fig. 2로 올린다.
