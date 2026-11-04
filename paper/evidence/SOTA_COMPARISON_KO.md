# AG3S-VLA SOTA 비교 및 주장 근거

기준일: 2026-11-04 (KST). 목적은 AG3S-VLA의 논문 contribution 중 성능 주장을 위한 비교 기준과 실험 계획을 정리하는 것이다. 아래 문헌 수치는 각 원문의 보고값이며 AG3S-VLA의 실험 결과가 아니다.

## 결론과 현재 주장 상태

**AG3S-VLA의 SOTA 달성은 아직 확인되지 않았다.** [기존 주장 등록부](CLAIMS.md)의 C001·C003·C004는 가설이며, 기존 hurdle 실험은 서로 다른 코드 상태의 비교 또는 소규모 탐색 결과다. 해당 결과를 아래 공개 벤치마크의 동일 조건 결과로 취급할 수 없다. 최신 결과가 등록되는 대로 원본 manifest와 함께 다시 판정한다.

SOTA는 벤치마크, suite·난도, 지표, 데이터 및 평가 조건을 명시한 범위 안에서 주장한다. 기존 baseline보다 높거나 성공률과 안전성의 균형이 개선된 것만으로 전체 SOTA가 되는 것은 아니다. 복수 지표에서 일방적으로 우세하지 않을 경우에는 측정된 trade-off를 구체적으로 기술한다. 문헌 검색은 완전한 리더보드를 보장하지 않으므로 제출 직전에 갱신한다.

## 1. SafeLIBERO: 기존 보고 결과

[SafeDojo 논문 v2, Table 1–2](https://arxiv.org/html/2606.20698v2)의 네 suite(Spatial, Goal, Object, Long) 평균이다. Level II는 Level I demonstration만으로 학습한 zero-shot 평가다. 표의 AEGIS 수치는 **SafeDojo 논문에서 평가한 구성**이며 AEGIS 원 논문의 다른 action space 및 평가 설정 결과와 그대로 합치지 않는다.

| Level | 방법 | TSR ↑ (%) | SSR ↑ (%) | ETS ↓ (steps) |
|---|---|---:|---:|---:|
| I | OpenVLA-OFT | 52.12 | 39.00 | 287.94 |
| I | AEGIS | 32.38 | 28.50 | 310.67 |
| I | SafeVLA | 54.25 | 45.00 | 294.69 |
| I | WMPO | 60.62 | 44.88 | 274.02 |
| I | SafeDojo | 64.50 | 53.25 | 269.41 |
| II | OpenVLA-OFT | 52.12 | 43.38 | 286.72 |
| II | AEGIS | 33.13 | 30.50 | 302.44 |
| II | SafeVLA | 56.50 | 49.50 | 289.87 |
| II | SafeDojo | 57.00 | 49.62 | 277.00 |

TSR은 작업 완료 비율, SSR은 작업 완료와 장애물 무충돌을 모두 만족한 비율, ETS는 전체 episode의 평균 실행 step이다. CAR은 충돌 없이 종료한 episode 비율이고 CSC는 전체 episode당 접촉 simulation step 평균이다. TSR과 CAR만으로 SSR을 산출할 수 없으며, ETS는 timeout과 조기 종료에도 영향을 받는다. 각각의 분모와 충돌 대상은 evaluator 정의를 따른다. SafeDojo의 Level I SSR은 해당 표에서 SafeVLA보다 8.25 **percentage point** 높다. Level II 차이는 0.12 point로 작다.

### 후속 연구와 비교 가능성

- [AEGIS 원 논문 프로젝트](https://vlsa-aegis.github.io/)의 원래 지표 및 action-space 설정도 확인한다. [SOTA2 리더보드](https://www.sota2.com/research/sota/robot-manipulation-on-safelibero-level-i)는 제3자 탐색 보조 자료이며 확인 시 페이지의 실제 순위 내용이 제공되지 않아 최신 순위를 검증하지 못했다. 논문과 코드가 우선이다.

## 2. LIBERO-Safety: 물리 안전 기준

[공식 프로젝트](https://libero-safety.github.io/), [공식 코드](https://github.com/LIBERO-SAFETY/LIBERO-Safety), [논문 Table 3](https://arxiv.org/html/2606.23686)에 따른다. SafeLIBERO와는 별도 벤치마크이므로 level 이름이나 SR을 서로 혼합하지 않는다. Table 3의 SR은 안전 제약을 만족하며 작업을 완료한 비율이며, 아래 값은 3개 학습 seed의 평균 ± 표준편차다. 해당 논문의 비교 모델 중 π₀.₅는 전체 suite에 걸친 높은 성능을 보이지만 각 개별 셀의 최고값을 모두 차지하지는 않는다.

| 방법 | AAG L2 SR ↑ (%) | TSA L2 SR ↑ (%) | FSHOA L2 SR ↑ (%) |
|---|---:|---:|---:|
| OpenVLA | 4.0 ± 0.0 | 12.7 ± 0.9 | 0.0 ± 0.0 |
| OpenVLA-OFT | 1.3 ± 0.9 | 40.0 ± 1.6 | 42.7 ± 2.5 |
| π₀ | 11.3 ± 2.5 | 0.7 ± 0.9 | 0.7 ± 0.9 |
| π₀.₅ | 35.3 ± 2.5 | 56.7 ± 3.8 | 51.3 ± 6.1 |
| GR00T N1.6 | 19.3 ± 3.4 | 52.7 ± 2.5 | 50.0 ± 1.6 |

AAG는 주변 물체를 고려한 파지, TSA는 테이블 위 공간 회피, FSHOA는 자유 공간에서 손·물체 회피를 평가한다. 위는 AG3S와 관련성이 높은 L2 예시이며, 전체 benchmark SOTA 주장에는 HRI를 포함한 원문 전체 physical suite와 L0–L2를 동일하게 평가해야 한다. 논문은 충돌률(CR), 실행 시간, trajectory smoothness도 보조 지표로 제시한다. Semantic Safety는 정적 관측과 지시에 대한 위험 요청 거부율을 평가하는 별도 track이므로 물리 충돌 회피 결과로 그 성능을 주장하지 않는다. 공식 프로젝트·코드에서 독립 통합 SOTA 순위는 확인하지 못했으므로 Table 3은 **원 논문의 baseline 기준**으로만 사용한다.

## 3. AG3S-VLA 비교 실험 설계

| 비교 대상 | 목적 및 주의점 |
|---|---|
| π₀.₅ baseline | 같은 checkpoint와 입력에서 AG3S의 전체 효과 측정. benchmark별 원 논문의 fine-tuning 조건과 일치 여부 기록 |
| π₀.₅ + 기존 safety filter / AEGIS | 같은 backbone에서 추론 시 안전 필터 비교. AEGIS 원 설정과 재현 설정을 구분 |
| SafeVLA, SafeDojo | 학습 기반 안전 방법과 비교. 학습 데이터·연산량과 backbone 차이를 명시하고 가능하면 공식 checkpoint 재평가 |
| π₀.₅ + AG3S only / TO only / AG3S + TO | 각 구성 요소와 결합 효과 분리. 전체 구성은 AG3S-VLA (Ours) |

비교는 우선 **동일 evaluator에서 직접 실행한 결과**로 구성한다. 재현이 불가능한 방법의 논문 보고값은 별도 표에 원문 protocol 및 차이를 명시하고 공정한 직접 비교나 SOTA 순위 판정의 근거로 자동 사용하지 않는다. SafeDojo 등 학습 기반 방법과 추론 기반 방법은 허용된 학습 데이터와 비용이 다를 수 있으므로 이를 숨기지 않는다.

1. **사전 고정:** benchmark 버전, suite·level, task 목록과 집계 가중치, 초기 상태·seed·episode 수, timeout, simulator·robot·control frequency, action space·chunk, 관측 및 카메라를 기록한다. 목표 물체의 허용 접촉, robot/link 및 운반 물체의 금지 접촉, 충돌 판정과 조기 종료 규칙을 고정한다.
2. **동등한 자원과 입력:** checkpoint, fine-tuning 데이터, 안전 정보의 출처(시뮬레이터 GT 또는 perception), obstacle geometry, runtime budget과 hardware를 기록한다. 동일 자원 비교와 추가 정보를 허용한 비교를 구분한다.
3. **주요 결과:** SafeLIBERO에서는 TSR·SSR·CAR·CSC·ETS, LIBERO-Safety에서는 공식 SR·CR과 시간·smoothness를 원 정의로 보고한다. 각 suite·level의 성공/전체 episode 수, 실패 원인 및 전체 집계 방식을 함께 제시한다. 필요하면 전신·파지 물체 접촉과 허용 접촉을 별도 진단 지표로 기록한다.
4. **불확실성과 공정성:** 가능한 경우 동일 초기 상태·seed로 paired 평가하고 episode 단위 paired bootstrap 신뢰구간과 독립 학습 seed 간 변동을 분리해 보고한다. 중복 episode를 독립 seed로 세지 않는다. 하이퍼파라미터는 test와 분리된 개발 세트에서 고정한다.
5. **실행 비용:** end-to-end latency의 측정 경계와 median/p95, 실패·timeout 및 안전 정지 빈도를 보고한다. 이전 [주장 등록부](CLAIMS.md)의 runtime 결과는 모든 구성에서 해당 chunk budget을 넘었으므로 별도 개선 및 검증 전 실시간이라고 주장하지 않는다.
6. **출처 등록:** 코드 revision, 설정, checkpoint, evaluator, raw episode log, 집계 script 및 manifest/hash를 보존하고 [결과 등록부](RESULTS.md)와 연결한다. 제출 직전 최신 논문 및 공개 benchmark 결과를 재검색한다.

## 4. 결과 판정과 논문 문장

- **범위가 지정된 SOTA:** 동일 protocol·지표에서 최신 강력한 공개 방법들과 비교해 최고 결과가 확인되면, 예를 들어 “SafeLIBERO Level I의 [지정 suite 집계]에서 SSR [X]%를 달성하여 비교한 최고 기존 결과 [Y]%를 [Δ] percentage point 상회했다”라고 쓴다. 재현 또는 원문 값의 출처, 조건과 불확실성을 함께 제시한다. 수치는 현재 미측정이다.
- **안전–성공 trade-off 개선:** 동일 기준에서 안전 지표 개선과 작업 성공률 변화가 함께 관측될 때 두 수치를 모두 제시한다. 모든 방법보다 우세하지 않다면 “SOTA” 대신 해당 비교 대상에 대한 구체적 개선으로 기술한다. 비열등성이나 성능 유지 주장은 사전에 정한 허용 한계와 불확실성 평가가 필요하다.
- **부분적 우수성:** 일부 suite·level이나 특정 metric만 앞서면 해당 범위만 명시한다. 개별 suite의 최고값을 전체 벤치마크 최고값으로 확대하지 않는다.
- **현재 사용 가능한 표현:** “AG3S-VLA가 안전 제약 하에서 작업 성공과 충돌 회피를 함께 개선하는지 SafeLIBERO 및 LIBERO-Safety에서 평가한다.” 이는 연구 목표이며 성능 달성 주장이나 실험 완료 보고가 아니다.
