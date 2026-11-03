# AG3S baseline 원문 결과와 비교 조건 (2026-10-31)

## 범위와 읽는 법

사용자가 제공한 PDF 12편 중 closed-loop 비교 논문 7편의 실험 방법과 보고값을 원문에서 추출했다. Grounding 논문 5편은 아래에 목적과 비교 단위를 정리했다. 별도 π0.5 논문은 제공 목록에 없지만 선택 참고 자료이고, VLSA 등 원문에 기본 policy 조건이 기록돼 있다. 원문의 성능과 우리 재구현 성능은 별개다. 이 문서의 값은 **논문 저자가 보고한 값**이며, 우리 코드의 재현 결과가 아니다. 페이지는 PDF 파일 기준, 지표는 원문 정의를 사용한다.

## 1. VLSA / AEGIS — SafeLIBERO 표준 안전 제약

[원문 PDF](../../../papers/VLSA_%20Vision-Language-Action%20Models%20with%20Plug-and-Play%20Safety%20Constraint%20Layer%20%281%29.pdf), PDF p.5–6, §IV-A 및 Table I. 질문: π0.5의 행동을 CBF 안전 제약으로 수정하면 장애물을 피하면서 과제를 완수하는가? SafeLIBERO 4개 suite × 4 task × 2 level × 50 episode = 1,600회, π0.5-LIBERO. Level I은 target 부근, Level II는 이동 경로의 장애물. horizon은 Spatial/Goal/Object 300, Long 550 step이다. CAR은 충돌 없는 episode 비율, TSR은 작업 성공률, ETS는 실행 step이다. 회전까지 고려하는 full action과 이동만 고려하는 translational 설정은 분리한다.

| 원문 설정 / 방법 | CAR ↑ | TSR ↑ | ETS ↓ |
|---|---:|---:|---:|
| 이동만, π0.5 | 18.7% | 50.9% | 278.2 |
| 이동만, AEGIS | 77.9% | 68.1% | 262.3 |
| 전체 action, π0.5 | 17.1% | 57.8% | 260.8 |
| 전체 action, AEGIS | 68.9% | 67.5% | 257.6 |

안전 성공률(성공 ∧ 충돌 없음)은 이 Table I에서 제시하지 않으므로 CAR과 TSR만으로 계산하지 않는다. 우리 AEGIS는 원문의 VLM 단계 등과 다르다. 재실행 시 action 범위·카메라·장애물 인식·제어 주기와 step budget을 맞추거나 차이를 명시해야 한다.

## 2. KNOWS — attention target 선택과 동적 장애물

[원문 PDF](../../../papers/Your%20Model%20Already%20Knows_%20Attention-Guided%20Safety%20Filter%20for%20Vision-Language-Action%20Models%20%284%29.pdf), PDF p.5–6, §4 및 Table 1. 질문: 모델 내부 attention으로 조작 target을 골라 필터 대상에서 제외하면서 충돌을 줄일 수 있는가? π0.5, SafeLIBERO 4 suite, 각 task당 50 episode, static Level I·II와 저자 추가 dynamic Level III. 세 지표는 SR=작업 성공 episode, CR=비-target 장애물 충돌 episode, SSR=성공하고 충돌 없는 episode다. YOLOE를 물체 segmentation에 fine-tune했다. Naive는 원문 AEGIS 전체 파이프라인이 아니라 초기 시점 GT segmentation으로 얻은 고정 장애물 타원체다.

| Suite / level | No CBF SR / CR / SSR | Naive SR / CR / SSR | KNOWS SR / CR / SSR |
|---|---|---|---|
| Spatial I | 67.5 / 86.5 / 13.5 | 69.0 / 26.0 / 61.0 | 63.0 / 32.5 / 53.0 |
| Spatial II | 49.0 / 88.0 / 11.0 | 80.5 / 34.5 / 63.5 | 87.5 / 27.0 / 70.0 |
| Object I | 43.0 / 83.0 / 14.0 | 73.0 / 4.0 / 73.0 | 71.0 / 6.5 / 70.5 |
| Object II | 69.0 / 72.0 / 25.5 | 87.5 / 11.0 / 80.0 | 79.5 / 17.5 / 69.0 |
| Goal I | 51.0 / 95.0 / 5.0 | 89.5 / 9.0 / 82.0 | 87.0 / 10.0 / 81.0 |
| Goal II | 66.0 / 64.5 / 29.0 | 55.0 / 36.0 / 36.5 | 82.5 / 41.0 / 52.0 |
| Long I | 58.5 / 86.0 / 13.5 | 60.5 / 32.0 / 36.5 | 49.0 / 45.5 / 35.0 |
| Long II | 47.0 / 83.5 / 14.5 | 48.0 / 14.0 / 42.5 | 35.0 / 39.5 / 24.0 |

모든 숫자는 %. Level III는 별도 동적 조건이므로 위 표에 합치지 않았다(원문 Table 1). 정적 조건에서 KNOWS가 Naive보다 모든 셀에서 좋은 것은 아니다. 우리 closed-loop KNOWS adapter는 현재 attention target exclusion 대신 detector target box를 사용하므로 원 방법 충실도 보완이 필수다.

## 3. Multi-link safety filter — 보호 link 범위

[원문 PDF](../../../papers/multi%20link%20Link%20Safety%20Filtering%20for%20VLA%20Policies%20Around%20Moving%20Hazards.pdf), PDF p.4–5, §IV 및 Table I(b). 질문: 같은 인식에서 end-effector만이 아닌 여러 link를 보호하면 접촉을 더 줄이는가? π0.5-LIBERO, SafeLIBERO 정적 32개 scene, configuration당 1,600 episode. 표의 C는 충돌로 기록한 episode, S는 과제 성공, SS는 안전 성공이다.

| 방법 | C ↓ | S ↑ | SS ↑ |
|---|---:|---:|---:|
| π0.5 단독 | 83.44% | 60.00% | 15.06% |
| AEGIS, 저자 이식 | 30.75% | 64.88% | 49.75% |
| Multi-link, 논문 전체 시스템 | 27.00% | 60.31% | 51.69% |
| Multi-link, 위치 초기 고정 | 25.19% | 61.62% | 52.88% |
| EE만 보호, 위치 초기 고정 | 31.31% | 65.62% | 51.56% |

원문은 물체 위치의 축별 절대 변화량 합 >1 mm를 충돌 지표로 사용하며, 직접 robot–hazard 접촉과 다를 수 있다고 명시한다. 동일 인식·controller를 둔 아래 두 coverage 행이 link 범위 비교에 더 적절하다. 우리 재구현은 동적 장애물 optical flow 등을 생략했다.

## 4. Constrained flow matching — 행동 생성 중 제약

[원문 PDF](../../../papers/Neuro-Symbolic%20Safety%20Guidance%20for%20Vision-Language-Action%20Models%20via%20Constrained%20Flow%20Matching.pdf), PDF p.8–9, §5 및 Table 1. 질문: 생성 후 필터 대신 flow matching 중 제약을 주면 성공과 충돌 회피를 함께 유지하는가? SafeLIBERO 4 suite, I·II, π0.5-LIBERO. **장애물 위치는 simulator에서 취득하는 privileged 조건**이다. 원문 Table 1의 AEGIS·π0.5 행은 VLSA의 translational 결과와 일치하므로 이를 독립 재실행으로 단정하지 않는다.

| 원문 방법 | CAR ↑ | TSR ↑ | ETS ↓ |
|---|---:|---:|---:|
| π0.5 | 18.69% | 50.88% | 278.24 |
| AEGIS | 77.85% | 68.13% | 262.30 |
| CFM | 82.81% | 81.62% | 299.97 |

원문의 baseline과 CFM은 장애물 인식 조건이 다를 수 있다. 위 CFM 값은 인식까지 포함한 비특권 결과로 소개하지 않는다. 우리 CFM은 재구현이며 새 action box 설정으로 다중 episode를 다시 검증해야 한다.

## 5. NUDGE — 생성 중 거리장 보정

[원문 PDF](../../../papers/Zero-Shot%20Reactive%20Obstacle%20Avoidance%20for%20Generative%20Robot%20Policies.pdf), PDF p.5–6, §IV-C 및 Table III. 질문: 장애물이 추가된 과제에서 정책의 작업 능력을 유지하면서 접촉을 줄일 수 있는가? π0.5 LIBERO-Object 10 task, task당 초기 상태 10개, 조건당 방법별 100 episode. 하나 또는 세 개의 구형 장애물을 추가했다. NUDGE는 simulator mask로 robot·target·basket·table을 제외하고 41개 Panda 구와 payload 구를 사용한다.

| 장애물 수 / 원문 방법 | 작업 성공 SR ↑ | 접촉률 CR ↓ |
|---|---:|---:|
| 1개 / π0.5 | 30% | 13.15% |
| 1개 / OmniGuide | 30% | 12.52% |
| 1개 / NUDGE | 44% | 2.30% |
| 3개 / π0.5 | 4% | 38.26% |
| 3개 / OmniGuide | 7% | 33.59% |
| 3개 / NUDGE | 13% | 14.01% |

이 표의 CR은 **전체 기록 control step 중 접촉 step 비율**이다. KNOWS 등의 episode 충돌률과 다르다. 원문의 NUDGE와 OmniGuide도 서로 다른 환경 표현을 사용한다. 우리 NUDGE 부분 재구현은 link·payload 등 차이가 있으므로 원문과 동등한 구현이 아니다.

## 6. OmniGuide — 일반 정책을 inference-time guidance

[원문 PDF](../../../papers/OmniGuide_%20Universal%20Guidance%20Fields%20for%20Enhancing%20Generalist%20Robot%20Policies%20%282%29.pdf), PDF p.5–6, §IV-A 및 Fig. 5. 질문: collision·semantic guidance가 기존 policy의 안전 및 과제 성공을 개선하는가? RoboCasa에서 GR00T N1.6 기반, 조건별 50회. safety rate는 robot body와 정적 가구가 한 번도 충돌하지 않은 episode 비율이다. 그림으로 결과가 제시돼 있으나 여기서는 막대그래프의 숫자를 눈대중으로 옮기지 않았다. NUDGE 원문 Table III의 OmniGuide 이식 결과는 위 §5에 별도 명시했다. 우리 구현은 π0.5 EE-delta 이식이므로 원 논문의 RoboCasa 결과와 직접 비교할 수 없다.

## 7. Lan-o3dp — 목표/장애물 분리와 guidance

[원문 PDF](../../../papers/Language-Guided%20Object-Centric%20Diffusion%20Policy%20for%20Generalizable%20and%20Collision-Aware%20Robotic%20Manipulation.pdf), PDF p.4–6, §IV 및 Table II. 질문: object-centric policy의 작업 성공 및 inference-time guidance의 장애물 회피가 가능한가? RLBench 21 task의 학습된 policy 성공은 최고 5개 평가의 평균으로 보고돼, 단일 test에서의 성공률로 취급하지 않는다. 실제 로봇 회피 실험은 장애물별 5회다. Table II의 무 guidance는 7개 조건 모두 0%, guidance는 pouring의 bottles/box/kettle/phone/laptop에 각각 100/60/80/100/60%, brush bottles 40%, drawer bottles 20%로 보고된다. 실제 로봇·소표본이며 우리 π0.5 guidance-only 이식 수치와 별도다.

## 8. Perception·grounding 원문과 후속 평가

이 5편의 원래 benchmark 결과는 로봇 안전 성공률이 아니므로 위의 closed-loop 성능표에 넣지 않는다. 각 provider를 **같은 RGB-D frame, 같은 depth lift, 같은 GT**에 적용해 identity 정확도, 이웃을 target으로 잘못 포함한 정도, target 누락 및 3D voxel 오류를 측정하는 것이 AG3S 비교에 직접적이다.

| 원문 | 원문이 주로 시험하는 것 | AG3S 비교에서 역할 |
|---|---|---|
| [Grounding DINO](../../../papers/Grounding%20DINO%20Marrying%20DINO%20with%20Grounded%20Pre-Training%20for%20Open-Set%20Object%20Detection.pdf) | 개방형 객체 검출, COCO·LVIS·RefCOCO 등 | text로 target box를 얻는 외부 detector |
| [SAM 2](../../../papers/SAM%202%20Segment%20Anything%20in%20Images%20and%20Videos.pdf) | 이미지·동영상 prompt 기반 segmentation | Grounding DINO와 결합한 per-frame mask 및 tracking |
| [YOLOE](../../../papers/YOLOE%20Real-Time%20Seeing%20Anything.pdf) | 개방형 검출·segmentation | 외부 mask provider |
| [Localization heads](../../../papers/Your%20Large%20Vision-Language%20Model%20Only%20Needs%20A%20Few%20Attention%20Heads%20For%20Visual%20Grounding.pdf) | VLM attention head의 visual grounding | VLA attention 선택의 대조군 |
| [TokenCut](../../../papers/Self-Supervised%20Transformers%20for%20Unsupervised%20Object%20Discovery%20using%20Normalized%20Cut.pdf) | transformer feature graph 기반 object discovery | feature graph segmentation 대조군 |

## 9. 비교 전 반드시 고정할 차이와 제안

1. **보고값 대 재실행:** 원문 표와 동일한 benchmark라도 checkpoint·action 범위·초기 상태·시간 제한·인식 권한·link 보호 범위를 먼저 비교한다. 수치 출처를 paper-reported / official code rerun / adapted / reimplementation으로 표기한다.
2. **분모·충돌 정의:** episode-level collision, step-level contact, 1 mm 물체 변위, 로봇 접촉은 같은 값이 아니다. 성공과 충돌의 결합 값인 SS는 별도 episode 기록이 없으면 CAR과 TSR에서 복원하지 않는다.
3. **공정한 factorial 비교:** KNOWS 이식이 현재 detector box를 사용하므로 원문의 attention+YOLOE를 충실히 복원하거나 명확한 이식 행으로 제한한다. F1/F2와 F5/F6이 같은 target mask·장애물 정보를 실제로 공유하는지 검증한다. privileged GT target과 GT obstacle은 별도 입력이다.
4. **비교력:** 논문 표의 평균값만으로는 paired 차이, confidence interval, 보호하지 않는 link 접촉을 알 수 없다. 우리의 episode별 outcome 및 failure 분류를 남기고 같은 init·seed로 비교한다. 추가로 접촉이 없어도 정지해 실패한 사례와 성공했지만 충돌한 사례를 분리한다.
5. **현 실행 상태:** 기존 파일럿은 [baseline 현황](AG3S_BASELINE_STATUS_20261031_KO.md)에 별도 기록돼 있다. 현재 설정의 전체 재측정이나 위 원문 수치 재현 검증은 아직 수행하지 않았다.
