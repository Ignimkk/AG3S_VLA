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
| I | π₀.₅ | TBD | TBD | TBD | TBD | TBD |
| I | Generic geometry + fixed TO | TBD | TBD | TBD | TBD | TBD |
| I | External segmentation + fixed TO | TBD | TBD | TBD | TBD | TBD |
| I | AEGIS | TBD | TBD | TBD | TBD | TBD |
| I | SafeVLA | TBD | TBD | TBD | TBD | TBD |
| I | SafeDojo | TBD | TBD | TBD | TBD | TBD |
| I | **AG3S-VLA** | **빈칸** | **빈칸** | **빈칸** | **빈칸** | **빈칸** |
| II | π₀.₅ | TBD | TBD | TBD | TBD | TBD |
| II | Generic geometry + fixed TO | TBD | TBD | TBD | TBD | TBD |
| II | External segmentation + fixed TO | TBD | TBD | TBD | TBD | TBD |
| II | AEGIS | TBD | TBD | TBD | TBD | TBD |
| II | SafeVLA | TBD | TBD | TBD | TBD | TBD |
| II | SafeDojo | TBD | TBD | TBD | TBD | TBD |
| II | **AG3S-VLA** | **빈칸** | **빈칸** | **빈칸** | **빈칸** | **빈칸** |

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
