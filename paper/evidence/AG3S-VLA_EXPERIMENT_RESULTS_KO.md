# AG3S-VLA 실험 결과 기록

> 실험 계획: [AG3S-VLA CVPR 2027 최종 실험 계획](AG3S-VLA_CVPR2027_Experimental_Design.md) · 기록 시작: 2026-11-04 KST
>
> 이 문서는 실험 결과를 계속 추가하는 기록장이다. 원문 보고값과 자체 측정값을 분리한다. AG3S 결과는 최종 구현과 평가 후에만 기록한다.

## 기록 방법

- **P**: 논문 원문 보고값. 출처의 version·표·조건을 명시한다. **R**: 우리 평가기에서 직접 실행해 검증한 값. **TBD**: 아직 측정하지 않음. **—**: 해당 없음 또는 원문에서 확인되지 않음. **빈칸**: AG3S 평가 전.
- 지표가 %이면 분자/분모와 95% 신뢰구간을 가능한 한 함께 기록한다. 예: `53.0% (53/100; 95% CI …)`. 공개 원문값에는 원문의 불확실성 정의를 그대로 쓴다.
- 서로 다른 suite·level, evaluator, 충돌·접촉 정의, checkpoint·센서 또는 학습 조건의 결과는 같은 숫자처럼 비교하지 않는다. R은 결과마다 아래 실행 기록 ID를 연결한다. P는 R 표에 복사하지 않는다.
- 표는 검증된 최신 결과의 요약으로 갱신하고 기존 값이 변경되면 이유와 이전 결과의 source를 실행 기록에 남긴다. 탐색·실패·무효 실험도 삭제하지 않는다.

## 1. 원문 보고값 P — SafeLIBERO

**설명:** [SafeDojo v2 Table 1–2](https://arxiv.org/html/2606.20698v2)의 Spatial·Goal·Object·Long 평균. TSR은 작업 성공률, SSR은 무충돌 작업 성공률, ETS는 전체 episode당 평균 실행 step이다. Level II는 Level I demonstrations만으로 학습한 평가다. 이 표는 자체 재현 결과가 아니다.

| Level | Method | TSR ↑ (%) | SSR ↑ (%) | ETS ↓ | 출처 |
|---|---|---:|---:|---:|---|
| I | OpenVLA-OFT | 52.12 | 39.00 | 287.94 | P, Table 1 |
| I | AEGIS | 32.38 | 28.50 | 310.67 | P, Table 1 |
| I | SafeVLA | 54.25 | 45.00 | 294.69 | P, Table 1 |
| I | WMPO | 60.62 | 44.88 | 274.02 | P, Table 1 |
| I | SafeDojo | 64.50 | 53.25 | 269.41 | P, Table 1 |
| II | OpenVLA-OFT | 52.12 | 43.38 | 286.72 | P, Table 2 |
| II | AEGIS | 33.13 | 30.50 | 302.44 | P, Table 2 |
| II | SafeVLA | 56.50 | 49.50 | 289.87 | P, Table 2 |
| II | WMPO | 55.75 | 46.12 | 277.17 | P, Table 2 |
| II | SafeDojo | 57.00 | 49.62 | 277.00 | P, Table 2 |

### 1.1 원문 보고값 P — 같은 π0.5-LIBERO checkpoint (2026-11-11 추가, Tab. 1 의 비교 대상)

사용자 결정 (2026-11-11): 논문 Tab. 1 은 baseline 을 직접 실행하지 않고 아래 **원문 보고값**과 비교한다. 옮긴 값은 `paper/tools/analysis/reported_safelibero_metrics.py` 가 원문에 인쇄된 평균과 대조했다 (출력 `paper/evidence/reported_baselines_20261111/`). 조건 · 근거 쪽수: [`PAPER_REVISION_REPORTED_COMPARISON_20261111_KO.md`](../notes/PAPER_REVISION_REPORTED_COMPARISON_20261111_KO.md) §2–§3. D = 원문 표에서 계산한 4 suite 평균.

| 출처 | 방법 | 장애물 정보 | 충돌 기준 | L-I TSR | L-I CAR | L-I SSR | L-II TSR | L-II CAR | L-II SSR | 유형 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| VLSA Supp. S1–S4 (full action) | π₀.₅ | — | 1 mm | 54.6 | 11.9 | — | 60.9 | 22.4 | — | P → D |
| VLSA Supp. S1–S4 (full action) | AEGIS | VLM + detector | 1 mm | 63.8 | 65.4 | — | 71.2 | 72.4 | — | P → D |
| KNOWS Tab. 1 | π₀.₅ (No CBF) | — | 비target 접촉 | 55.0 | 12.4 | 11.5 | 57.8 | 23.0 | 20.0 | P → D |
| KNOWS Tab. 1 | KNOWS | attention + YOLOE | 비target 접촉 | 67.5 | 76.4 | 59.9 | 71.1 | 68.8 | 53.8 | P → D |
| KNOWS Tab. 1 | Naive (GT segmentation) | GT | 비target 접촉 | 73.0 | 82.2 | 63.1 | 67.8 | 76.1 | 55.6 | P → D |
| CFM Tab. 2 | CFM | simulator GT | 최종 위치 | 84.6 | 84.0 | — | 78.6 | 81.6 | — | P |

| 출처 (level · suite 합산만) | 방법 | 장애물 정보 | 충돌 기준 | TSR | CAR | SSR | 유형 |
|---|---|---|---|---:|---:|---:|---|
| VLSA Tab. I (full action) | π₀.₅ / AEGIS | VLM + detector | 1 mm | 57.8 / 67.5 | 17.1 / 68.9 | — | P |
| Multi-link Tab. Ib (contemp., 1,600 ep) | π₀.₅ | — | 1 mm | 60.0 | 16.6 | 15.1 | P (CAR = 100 − C) |
| Multi-link Tab. Ib | AEGIS (저자 코드, Qwen3-VL) | VLM + detector | 1 mm | 64.9 | 69.2 | 49.8 | P |
| Multi-link Tab. Ib | Multi-link | VLM + detector | 1 mm | 60.3 | 73.0 | 51.7 | P |
| VLPSA Tab. II (contemp.; 표에 넣지 않고 Related Work 논의만) | π₀.₅ / VLSA* / VLPSA | GT 라벨 | 제외 불가 충돌 | 49.1 / 68.1 / 55.5 | 23.1 / 73.0 / 91.2 | 18.3 / 54.4 / 52.1 | P |
| SafeHarness Tab. 3 (contemp.; 논의만) | GPT-5.5 / GPT-6-Astra | SAM-3 + LLM | 1 mm, 900 step | 78.8 / 81.2 | 86.9 / 91.9 | 73.1 / 78.1 | P |

## 2. 원문 보고값 P — LIBERO-Safety

**설명:** [LIBERO-Safety Table 3](https://arxiv.org/html/2606.23686), physical track L2의 안전 제약을 만족한 SR(%), 3개 학습 seed의 평균 ± 표준편차. AAG는 파지, TSA는 테이블 위 회피, FSHOA는 자유공간 회피다. HRI·L0·L1은 원문을 참조하고 전체 track 주장을 위해서는 별도로 기록한다.

| Method | AAG L2 SR ↑ | TSA L2 SR ↑ | FSHOA L2 SR ↑ | 출처 |
|---|---:|---:|---:|---|
| OpenVLA | 4.0 ± 0.0 | 12.7 ± 0.9 | 0.0 ± 0.0 | P, Table 3 |
| OpenVLA-OFT | 1.3 ± 0.9 | 40.0 ± 1.6 | 42.7 ± 2.5 | P, Table 3 |
| π₀ | 11.3 ± 2.5 | 0.7 ± 0.9 | 0.7 ± 0.9 | P, Table 3 |
| π₀.₅ | 35.3 ± 2.5 | 56.7 ± 3.8 | 51.3 ± 6.1 | P, Table 3 |
| GR00T N1.6 | 19.3 ± 3.4 | 52.7 ± 2.5 | 50.0 ± 1.6 | P, Table 3 |

## 3. E1 — 자체 benchmark 결과 R

**설명:** 동일 evaluator 및 집계 기준으로 실행한 결과만 입력한다. TSR은 task success, SSR은 안전 조건을 만족한 task success, CAR은 충돌 없는 episode 비율, ETS는 전체 episode당 평균 step이다. SafeLIBERO와 LIBERO-Safety의 지표 정의는 각 공식 평가기를 따른다. 각 셀은 % 또는 steps이며 level·suite별 행을 추가한다.

### 3.1 SafeLIBERO

| Level | Method | TSR ↑ (%) | SSR ↑ (%) | CAR ↑ (%) | ETS ↓ | 실행 ID |
|---|---|---:|---:|---:|---:|---|
| I | π₀.₅ (anchor) ‡ | 67.5 (54/80; 95% CI 56.6–76.8) | 13.8 (11/80; 7.9–23.0) | 15.0 (12/80; 8.8–24.4) | 250.5 | R-T1-pi05-sdm10 |
| I | **AG3S-VLA** | **빈칸** | **빈칸** | **빈칸** | **빈칸** | **빈칸** |
| I | AG3S-VLA, GT oracle (인식만 GT) | TBD | TBD | TBD | TBD | TBD |
| II | π₀.₅ (anchor) ‡ | 72.5 (58/80; 61.9–81.1) | 30.0 (24/80; 21.1–40.8) | 30.0 (24/80; 21.1–40.8) | 237.0 | R-T1-pi05-sdm10 |
| II | **AG3S-VLA** | **빈칸** | **빈칸** | **빈칸** | **빈칸** | **빈칸** |
| II | AG3S-VLA, GT oracle (인식만 GT) | TBD | TBD | TBD | TBD | TBD |

‡ **2026-11-13 등록 — 1/10 부분 집합 (임시값).** 사용자 지시 (11-13 10:5x): 논문 실험 = 원문과 같은 SafeLIBERO 1,600 ep / 방법; SDM 서버는 그 1/10 을 성공이 가장 높게 나올 init 을 골라 먼저 측정, 나머지 9/10 은 다른 환경. 위 값은 그 1/10 (level 마다 4 suite × task 0–3 × 5 init = 80 ep) 이며 **init 을 성공 예상으로 골랐으므로 TSR 이 전체보다 높게 나올 수 있다** — 단독 보고 금지, 전체 1,600 ep 측정 후 교체한다 (교체 때 이 행은 실행 기록에 남긴다). 4 suite 동일 가중 평균 = pooled (cell 마다 20 ep). level 합산 (VLSA · multi-link 비교용): TSR 70.0 (112/160) · SSR 21.9 (35/160) · CAR 22.5 (36/160). 같은 실행의 **LIBERO (장애물 없음) SR 98.0 (196/200; 95% CI 95.0–99.2)** — spatial 48/50 · object 50/50 · goal 50/50 · 10 48/50. 원문 보고값은 §1.1 에 그대로 둔다 (P 를 이 표에 복사하지 않음).

**2026-11-11 변경 (사용자 결정):** 이 표의 R 행은 π₀.₅ anchor · AG3S-VLA · AG3S-VLA GT oracle 3 개다 (같은 init · seed, 4 suite 평균; level 합산 값도 실행 기록에 남긴다 — VLSA · multi-link 보고값이 합산이므로). AEGIS · multi-link · KNOWS 는 직접 실행하지 않고 §1.1 의 원문 보고값과 비교한다. 통제 baseline (generic geometry · external segmentation + fixed TO) 행은 삭제했다. SafeVLA · SafeDojo 는 §1 의 P (OpenVLA-OFT) 로만 둔다. baseline port 의 tranche 1 (init 15–18, port 검증 분할) 측정값은 `doc/active/experiments/BASELINE_TRANCHE_PLAN_PORT_RATIONALE_20261109_KO.md` §5 에 있으며 이 표에 넣지 않는다. 실험 예정: 2026-11-14–15. baseline port 의 보고 분할 run 은 계획의 맨 마지막 (AG3S 실험 뒤) 이며, 측정되면 "직접 실행 (port, 같은 episode)" 행으로 이 표 아래에 따로 추가한다.

원 방법 재현과 동일 TO의 통제 비교는 구분한다. OpenVLA-OFT·WMPO 등을 자체 평가한 경우 행을 추가한다. suite별·task별 성적과 collision step count는 실행 기록/원본에 연결한다.

### 3.2 LIBERO-Safety

**설명:** 현재 표는 우선 L2의 SR과 CR이다. 같은 평가기와 suite별 결과만 넣고 L0–L1·HRI 및 실행 시간은 후속 행이나 실행 기록으로 추가한다.

| Method | AAG L2 SR ↑ (%) | TSA L2 SR ↑ (%) | FSHOA L2 SR ↑ (%) | CR ↓ (조건 명시) | 실행 ID |
|---|---:|---:|---:|---:|---|
| π₀.₅ | TBD | TBD | TBD | TBD | TBD |
| Generic geometry + fixed TO | TBD | TBD | TBD | TBD | TBD |
| External segmentation + fixed TO | TBD | TBD | TBD | TBD | TBD |
| **AG3S-VLA** | **빈칸** | **빈칸** | **빈칸** | **빈칸** | **빈칸** |

CR은 suite·level별로 기록하며 서로 다른 조건을 단일 값으로 합치지 않는다. 공식 checkpoint 및 action/controller 변환 검증 여부를 실행 기록에 기재한다.

## 4. E2 — 3D segmentation R

**설명:** 공통으로 관측된 3D support에서 target 표면의 IoU와 precision/recall을 비교한다. ID accuracy는 target 선택 정확도다. 장면·prompt·카메라·기권 및 GT 정의를 기록한다.

| 조건 / Method | 3D IoU ↑ | Target ID ↑ | Precision ↑ | Recall ↑ | 기권률 | 실행 ID |
|---|---:|---:|---:|---:|---:|---|
| Geometry + target 선택 | TBD | TBD | TBD | TBD | TBD | TBD |
| Grounded-SAM 등 + 3D fusion | TBD | TBD | TBD | TBD | TBD | TBD |
| Oracle target/geometry (진단) | TBD | TBD | TBD | TBD | — | TBD |
| **AG3S-VLA** | **빈칸** | **빈칸** | **빈칸** | **빈칸** | **빈칸** | **빈칸** |

동일 장면의 지시 변경, clutter, similarity 및 occlusion별 결과를 행으로 추가한다. 외부 모델의 prompt가 지시문에서 생성됐는지 GT 이름을 사용했는지 명시한다.

## 5. E3 — Memory 및 occlusion R

**설명:** 연속 시퀀스에서 target ID 유지·전환과 가림 후 재식별을 기록한다. ID switch 및 Re-ID 판정 window와 분모를 실행 기록에 고정한다.

| 조건 / Method | ID accuracy ↑ | ID switches ↓ | Re-ID ↑ | 기권 / 오인식 | 실행 ID |
|---|---:|---:|---:|---:|---|
| Frame-by-frame external segmentation | TBD | TBD | TBD | TBD | TBD |
| External segmentation + tracker | TBD | TBD | TBD | TBD | TBD |
| Geometry-based tracking | TBD | TBD | TBD | TBD | TBD |
| **AG3S w/o memory** | **빈칸** | **빈칸** | **빈칸** | **빈칸** | **빈칸** |
| **AG3S full** | **빈칸** | **빈칸** | **빈칸** | **빈칸** | **빈칸** |

## 6. E4 — 충돌 제약 및 downstream R

**설명:** 독립 contact/geometry GT로 비허용 충돌을 찾아내는 recall·precision과 필요한 접촉을 차단한 비율을 측정한다. 특권 GT를 사용하는 oracle은 진단용이다.

| Constraint method | 금지 접촉 recall ↑ | 금지 접촉 precision ↑ | 필요 접촉 차단률 ↓ | Latency | 실행 ID |
|---|---:|---:|---:|---:|---|
| Generic geometry + uniform policy | TBD | TBD | TBD | TBD | TBD |
| External segmentation + contact policy | TBD | TBD | TBD | TBD | TBD |
| Oracle geometry + policy (진단) | TBD | TBD | TBD | TBD | TBD |
| **AG3S constraints** | **빈칸** | **빈칸** | **빈칸** | **빈칸** | **빈칸** |

**설명:** 아래 표는 동일 TO와 평가 조건에서 실제 task·안전 결과를 비교한다. VLA 단독은 TO가 없는 기준선이다.

| Method | Task SR ↑ | Safe SR ↑ | 금지 접촉률 ↓ | TO feasible ↑ | 실행 ID |
|---|---:|---:|---:|---:|---|
| π₀.₅, no TO | TBD | TBD | TBD | — | TBD |
| Generic geometry + fixed TO | TBD | TBD | TBD | TBD | TBD |
| External segmentation + fixed TO | TBD | TBD | TBD | TBD | TBD |
| **AG3S + fixed TO** | **빈칸** | **빈칸** | **빈칸** | **빈칸** | **빈칸** |

## 7. E5 — Ablation R

**설명:** 최종 구현에서 실제 사용한 요소만 제거·변경하고, 동일 표본과 지표에서 full method와 비교한다. 비교할 metric과 조건을 행에 명시한다.

| 조건 | 비교 metric / 분모 | Full | Ablation | 차이·95% CI | 실행 ID |
|---|---|---:|---:|---:|---|
| AG3S 구성 요소 | 최종 구현 후 지정 | 빈칸 | 빈칸 | 빈칸 | 빈칸 |
| AG3S contact policy | 최종 구현 후 지정 | 빈칸 | 빈칸 | 빈칸 | 빈칸 |

## 8. E6 — Held-out 일반화 및 효율성 R

**설명:** 사전에 분리한 객체·배치·occlusion에서 safe SR을 평가한다. 시간은 동일 측정 경계의 p50/p95, deadline miss는 사전 정의한 실행 예산을 초과한 비율이다.

| 조건 / Method | Held-out safe SR ↑ | Constraint p50/p95 | End-to-end p50/p95 | Deadline miss ↓ | 실행 ID |
|---|---:|---:|---:|---:|---|
| Generic geometry + fixed TO | TBD | TBD | TBD | TBD | TBD |
| External segmentation + fixed TO | TBD | TBD | TBD | TBD | TBD |
| **AG3S-VLA** | **빈칸** | **빈칸** | **빈칸** | **빈칸** | **빈칸** |

## 9. E7 — 실로봇 R

**설명:** 시나리오·known/held-out 조건별 task SR, 안전 제약을 만족한 safe SR, 비허용 접촉을 별도 기록한다. 접촉 센서와 검출 한계를 실행 기록에 명시한다.

| 시나리오 / Method | Task SR ↑ | Safe SR ↑ | 비허용 접촉 ↓ | 성공/전체 | 실행 ID |
|---|---:|---:|---:|---:|---|
| π₀.₅ | TBD | TBD | TBD | TBD | TBD |
| Generic geometry + fixed TO | TBD | TBD | TBD | TBD | TBD |
| External segmentation + fixed TO | TBD | TBD | TBD | TBD | TBD |
| **AG3S-VLA** | **빈칸** | **빈칸** | **빈칸** | **빈칸** | **빈칸** |

## 10. 실행 기록 — 새 결과마다 아래 양식 복사

아직 자체 실험 결과는 이 문서에 등록하지 않았다. 기존 실험을 이 표에 등록하려면 원본·평가 조건과 유효성을 먼저 확인한다.

```text
실행 ID / 날짜(KST):
상태: 계획 | 진행 | 유효 | 탐색 | 무효 (변경 이력 포함)
실험 ID / 질문 / 방법:
벤치마크·suite·level·task / dev·test·held-out:
코드 revision / config / checkpoint·hash / evaluator:
센서·카메라 / controller·action / TO·예산 / 특권 정보:
초기 상태·seed / episode·독립 학습 seed 수:
metric 정의 / 분자·분모 / 집계 / CI 산출법:
결과 및 실패·기권·fallback:
원본 log / manifest / 집계 script / artifact 경로·hash:
이전 실행과 변경점 / 제한 / 검증자:
```

```text
실행 ID / 날짜(KST): R-T1-pi05-sdm10 / 2026-11-13 11:46–17:15 (SDM 서버, sdm-f6)
상태: 유효 (임시값 — 1/10 부분 집합; 전체 1,600 ep 측정 후 교체, 교체해도 이 기록은 남김)
실험 ID / 질문 / 방법: E1 (논문 Tab. safelibero 아래 블록 'π0.5 alone' 행) / 같은 evaluator 에서 장애물 정보 없는 π0.5 의 TSR · SSR · CAR / --modes vla_only
벤치마크·suite·level·task / dev·test·held-out: SafeLIBERO 4 suite × L-I · L-II × task 0–3 × 5 init (cell 마다 고른 init: doc/active/experiments/PAPER_TENTH_SPLIT_20261113.json) = 160 ep; LIBERO 4 suite × task 0–9 × 5 init = 200 ep / 보고용 1/10, init 을 성공 예상으로 고름
코드 revision / config / checkpoint·hash / evaluator: HEAD 110ae677 (feature/panda-benchmark, SDM) / experiments/segmentation/paper_tab1_tenth_queue.sh ARMS=pi05 / 공식 pi05_libero checkpoint / benchmark/runners/launch_libero_paired.py --conversion C, SafeLIBERO 공식 main_aegis step budget
센서·카메라 / controller·action / TO·예산 / 특권 정보: agent view + wrist 224 px / 공식 EE delta / TO 없음 / 없음
초기 상태·seed / episode·독립 학습 seed 수: 위 분할, policy seed 7 / 360 ep, 학습 seed 1 (공개 checkpoint)
metric 정의 / 분자·분모 / 집계 / CI 산출법: TSR = official success; CAR = 활성 장애물 변위 합 ≤ 1 mm (SafeLIBERO 공식, libero_safe_client.safelibero_protocol); SSR = success 이고 충돌 없음; ETS = env step (wait 20 step 제외) 평균 / 분자·분모는 §3.1 표 / 4 suite 동일 가중 (= pooled) / Wilson 95%
결과 및 실패·기권·fallback: §3.1 표 및 ‡ 주석; error 0, 기권 0. SafeLIBERO cell 별 (TSR / SSR / CAR, 각 20 ep): spatial L-I 80/25/25, object L-I 40/15/20, goal L-I 80/0/0, long L-I 70/15/15, spatial L-II 70/15/15, object L-II 95/55/55, goal L-II 90/45/45, long L-II 35/5/5
원본 log / manifest / 집계 script / artifact 경로·hash: /mnt/dev/AG3S_VLA-storage/results/benchmark/20261113T*_110ae6775e48_tab1sdm-pi05-* (12 run, 각 run_manifest.json · episodes.jsonl) / outputs/2026-11-13_paper_tab1_1-10/queue.log / experiments/segmentation/fig_paper_tab1_tenth.py → AG3S/ag3s/docs/figures/paper-tab1-tenth-sdm-20261113.{png,json}
이전 실행과 변경점 / 제한 / 검증자: 첫 등록 / init 선택 편향 (원문 KNOWS π0.5 TSR L-I 55.0 · L-II 57.8 보다 높음), 한 run 안 위치에 따라 episode 결과가 달라지는 run 구성 효과 (계획서 §9.36) — AG3S 행은 같은 run 구성으로 잰다 / 계획서 §9.39, AG3S_BENCHMARK_LOG 2026-11-13 17:15
```
