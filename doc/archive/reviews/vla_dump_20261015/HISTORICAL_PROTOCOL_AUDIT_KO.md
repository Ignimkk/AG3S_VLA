# 내부 feature 실험 프로토콜과 T43–T61의 중복·차이 감사

작성: 2026-10-15. 대상: [실험 프로토콜](../../../active/reviews/vla_dump_20261015/EXPERIMENT_PROTOCOL_KO.md), [개발 계획](../../../active/plans/develope_PLAN.md), T43–T61의 task/impl/verify 및 주요 구현. **기존 기록·코드를 대조한 감사이며 새 rollout이나 성능 재측정을 수행한 결과가 아니다.** 수치는 해당 verify의 최신 하위 결과에서 인용했다. [기계 판독 근거](HISTORICAL_AUDIT_EVIDENCE.json)에 파일 지문과 추출한 결과를 보존했다.

## 1. 결론과 이전 제안의 정정

**동일한 연구 문제를 이미 지속적으로 개선해 왔고, 새 프로토콜의 상당 부분은 기존 방법을 재사용해야 한다.** “attention+depth”, “명사와 action attention 결합”, “물리 rule+HSR”, “손과 함께 움직이는 점 검사”, “episode별 held-out 평가” 자체는 새 제안이 아니다.

특히 이전 문서에서 손 좌표계의 동반 운동을 추가 연구 방향으로 제시하면서 **T56의 실제 구현과 test 결과를 충분히 반영하지 못했다.** 핵심 운동 일관성 검사는 이미 있다. 새로 검증할 차이는 이를 attention 후보의 사후 self-score에서 **온라인 extent/attachment 추정의 관측 증거**로 바꾸는 부분이다. 또한 temporal evaluator 전체가 없다는 표현은 부정확했다. T55/T56/T60의 평가기를 재사용해야 한다.

반면 다음의 **결합 알고리즘과 그 추가 이득**은 조사한 기존 runtime/study에서 확인하지 못했다.

> Attention seed → SigLIP/projector/prefix feature prototype cosine → depth surface를 구분한 sparse graph → feature·위치·법선 affinity와 unary를 결합한 membership → 동일 identity의 extent 및 false-exemption 평가.

이는 저장소 조사 범위 안에서의 미확인이다. 외부 문헌상 novelty나 프로젝트 전체의 부재를 증명하는 주장이 아니다. 현재 `grounding_probe.py`에는 한 프레임의 feature prototype 진단만 있고, 이 전체 경로의 다중 episode 개선은 미검증이다.

## 2. 방법 단위 대조

| 프로토콜 구성 | 기존 근거 | 동일한 부분 | 아직 다른 부분 / 해야 할 일 |
|---|---|---|---|
| S0 provenance·재현·정합 | T48b, T53, T54b, T55 | 재렌더 pixel parity, baseline 재현, 기록/카메라/시간 정렬 | 새 full/compact collector의 tensor·action parity만 추가 |
| S1 action/noun source 비교 | T47, T49, T51/b, T54/c, T61 | head/query/aggregation 선택과 action·noun 비교 | 새 seed×feature prototype 비교; 기존 source 선택을 다시 처음부터 하지 않음 |
| S1 action+text 융합 | T54 A/B | 두 attention 신호를 같은 cloud에서 역할별 결합 | A×N 점별 곱이나 feature cosine과는 연산이 다름 |
| S2 depth→3D·영역 확장 | 기존 grounding, T44–46, T52 | 역투영, self-filter, multiview fusion, seed growth, cluster·ball | depth surface 경계 유지·feature-aware extent를 같은 baseline과 비교 |
| S3 경계/제외 영역 개선 | T44 ball_owned, T46 F1/F2, 개발 계획 DX-01 | target 잔여 및 이웃 물체 과잉 제거 문제 해결 | 기존 F1은 구 표면 규칙, F2는 ambiguous 처리; feature graph와 동일하지 않음 |
| S4 temporal identity·hand frame | TargetConfirm/anchor, T54b, T46g, T56 | identity 유지, attach/freeze replay, 손 좌표계 기하 | 관측 표면 membership의 시간적 갱신; 잘못된 extent 고착 방지의 추가 이득 |
| P0/P1 물리 strong rule+HSR | T48/b, T50, T55 | 물리 predicate, HSR 및 noun probe, causal persistence, 실패 개입 | P2–P4의 3D residual을 추가한 증분 효과만 새 비교 |
| P4 co-motion | T56 S2 | 손 좌표로 변환한 점의 최근접 거리·이동량 threshold | world-static 대 hand-static 가설 경쟁, online inference, 독립 attachment 판정 |
| S5 counterfactual | T48/b, T50, T55 | prompt/state/image 변경, empty/slip 재시뮬레이션 | region별 key/value patch 및 같은 noise의 action/FK 효과 비교는 동일 구현 미확인 |
| S6 downstream | T43–46, T57, T58; T60 평가 창 | contact/constraint/closed-loop 비교, 종료 정의 | 새 feature extent를 연결한 matched closed-loop는 미확인 |

### T54와 새 feature 융합은 무엇이 다른가

[`t54_arms.py`](../../../../AG3S/ag3s/experiments/studies/t54_arms.py)의 정의:

- N: noun map을 기존 lifting/seeding/growth/clustering에 입력한다.
- A: L8H2가 만든 후보 cluster를 noun attention mass로 선택한다.
- B: noun map으로 seed/growth를 하고 L8H2를 split ratio gate에, noun mass를 cluster 선택에 사용한다. Split component 순서까지 전부 L8H2로 바꾸는 것은 아니다.

새 S1은 `μ = normalize(Σ a_i normalize(f_i))`, `s_j = normalize(f_j)·μ`로 **특징 벡터 공간에서** 비슷한 patch를 찾는다. T54의 attention mass 결합과 목적은 가깝지만 입력·연산이 다르다. 따라서 T54 N/B를 기존 비교군으로 유지해야 “새 feature의 추가 가치”를 말할 수 있다. A×N 곱도 이 prototype과 별도 arm이다.

### 기존 geometry와 새 graph는 무엇이 다른가

[`target_grounding.py::grow_region`](../../../../AG3S/ag3s/stages/target_grounding.py)는 seed에서 KD-tree radius 이웃을 breadth-first로 확장한다. 이 함수 입력은 points/seed/radius이며 feature 벡터나 feature affinity는 받지 않는다. 이후 clustering, attention split, identity 유지가 이어진다.

[T46 구현](../../../../AG3S/ag3s/docs/handoff/T46.impl.md)의 F1은 관측점으로 맞춘 구의 표면·거리와 이웃 보호 조건으로 non-target 근거를 정화한다. F2는 ambiguous seed를 제외하는 선택이다. 이들 역시 boundary 개선 시도이므로 반드시 비교해야 하지만, `w_ij(feature, position, normal)`와 `(m-s)^TΛ(m-s)+λm^TLm`을 이용한 membership 최적화와 같은 알고리즘은 아니다.

“depth를 쓴다”는 공통점만으로 새 방법이라거나 동일 방법이라고 분류하지 않는다. **입력 표현, 목적 함수/규칙, 출력, 사용 시점, 평가 대상**을 모두 대조한다.

## 3. T56: 동반 운동은 이미 구현·test됨

근거: [T56 구현의 S2 정의](../../../../AG3S/ag3s/docs/handoff/T56.impl.md), [`t56_selfscore.py::hand_frame / s2_extension`](../../../../AG3S/ag3s/experiments/studies/t56_selfscore.py), [test 결과](../../../../AG3S/ag3s/docs/handoff/T56.verify.json)의 `test.numbers.selection`.

기존 S2는 non-support point를 손 주변 ball에서 고르고, `p_hand = R_hand^T(p_world-o_hand)`로 변환한 뒤 기준 frame의 점들과 최근접 거리를 비교한다. 손의 최소 이동량·최소 점 수·co-motion tolerance가 있고, 손 근처에 있다는 이유만으로 positive로 삼지 않는다. `held` 또는 `previous` 기준도 비교했다.

**사용 목적은 attention cell/grounding arm 선택용 self-label이다.** 이미 latch가 HELD인 구간에만 적용하며 latch가 풀리면 중단한다. 따라서 다음은 구분해야 한다.

- 재사용 가능: 좌표 변환, 주변점 추출, 최근접 residual, motion/no-evidence 진단.
- 그대로 주장 불가: “이 residual로 latch와 독립적으로 파지를 탐지했다”, “전체 target segmentation이 개선됐다”, “pick/place/home을 추정했다”.
- 새로운 검증: attachment onset 이전에도 관측 residual을 계산하고, world-static/hand-static 가설을 비교한다. 정지·가림이면 uncertain으로 둔다. GT는 평가에만 사용한다. 이후 T55 strong-rule에 이 증거를 추가했을 때의 개선을 측정한다.

결과도 조건부다. Test의 `current/cells`에서 S1→S2는 동률 후보 수 11→5, 동률 집합 평균 regret 0.05517→0.01230이었다. 그러나 `hold/cells`에서는 동률 후보 수 27→28, 해당 regret 0.04085→0.04156이었다. 이는 **S2의 효과가 모든 조건에서 일관되지는 않았음**을 보여준다. 선택된 하나의 regret와 동률 집합 regret는 다른 지표다. 이 숫자를 segmentation IoU나 성공률로 읽으면 안 된다.

## 4. T55: 작업 관련 판별도 이미 상당 부분 수행됨

근거: [`t55_study.py`](../../../../AG3S/ag3s/experiments/studies/t55_study.py), [`t55_predicate.py`](../../../../AG3S/ag3s/experiments/studies/t55_predicate.py), [T55 결과](../../../../AG3S/ag3s/docs/handoff/T55.verify.json)의 `test`.

실제 비교는 `strong=HELD`, `hsr=HELD AND persist(HSR)`, `noun=HELD AND persist(noun probe)`다. 같은 latch trace를 유지한다. Noun probe는 target token V 또는 final hidden을 입력으로 한 supervised logistic regression이고, episode-group CV로 feature/규제를 고른 뒤 고정했다. 새 weight 학습을 하지 않는 feature prototype과는 구분한다.

라벨도 구분한다. `P_grasp`는 무엇인가를 보유, `P_task`는 **지시한 물체를 보유**다. pick/place/home 3-class와 동일하지 않다. Instruction swap에서는 같은 물리 상태라도 P_task가 달라지고 P_grasp는 유지된다. 주 조건은 바뀐 prompt/장면에서 grounding을 다시 계산한 `reground`다.

Test 16 episode의 여러 개입 trace를 합친 P_task 지표:

| 지표 | strong | strong+HSR | strong+noun probe |
|---|---:|---:|---:|
| false-pass rate, negative 6616개 trace-frame | 4.7461% | 3.5520% | 0.2872% |
| false-reject rate, positive 131개 trace-frame | 15.2672% | 16.0305% | 16.7939% |
| strong 대비 false-pass 감소의 episode-block 95% CI | — | −2.216~−0.431 pp | −6.445~−2.582 pp |

이는 개입 조건을 합친 gate/predicate 평가다. 자연 운용의 발생률·phase accuracy·closed-loop 성공률로 일반화하지 않는다. 자연 조건만 보면 false-pass 수는 strong 32, HSR 9, noun 12여서, pooled 결과의 noun 우위를 모든 조건의 우위로 해석할 수도 없다.

다음 실험에서는 새 P0/P1 평가기를 만들기보다 이 predicate·persistence·event/censoring·bootstrap을 재사용한다. 새로운 3D motion 증거는 기존 strong/HSR/noun과 **같은 입력·라벨·평가 창**에서 추가 비교한다. Binary task-ready 개선과 3-class phase 개선은 별도 표로 보고한다.

## 5. T61: head 선택을 다시 시작할 필요가 없음

근거: [T61 task](../../../../AG3S/ag3s/docs/handoff/T61.task.md), [고정 선택](../../../../AG3S/ag3s/docs/handoff/T61.selection.json), [`t61_select.py`](../../../../AG3S/ag3s/experiments/studies/t61_select.py), [결과](../../../../AG3S/ag3s/docs/handoff/T61.verify.json)의 `heldout`.

T51/b의 선택기와 T60 P 평가 창을 재사용하여 query별 후보 수를 고정하고, fold 안에서 ranking·선택을 수행했다. GT identity 기준 ranking과 one-SE 규칙으로 **`action:mean:L10H7`, denoise step 0, k=1**을 고정했다. 이것은 **prefix의 `noun_target:L10H7`과 다르다.** 같은 layer/head 번호만 보고 같은 map으로 취급하지 않는다.

별도의 16 episode held-out 결과, episode 평균 identity 지표:

| 평가 창 | 기존 action:last:L8H2 | 선택 action:mean:L10H7 | paired 차이, episode bootstrap 95% CI |
|---|---:|---:|---|
| P, placement endpoint까지 | 62.94% | 69.98% | +7.04 pp [−1.27, +18.36] |
| H, home endpoint까지 | 58.77% | 63.00% | +4.23 pp [−5.51, +16.44] |
| full | 55.17% | 70.08% | +14.92 pp [+3.08, +27.92] |

P/H에 도달하지 못한 episode는 정의된 전체 창을 유지한다. **주 P 창의 episode CI는 0을 포함한다.** Full 창에서의 개선이나 frame moving-block CI만 골라 명확한 주 평가 우위라고 쓰지 않는다. 이 지표는 offline grounding이고 source를 바꾼 closed-loop 성공률이 아니다.

[T57-A](../../../../AG3S/ag3s/docs/handoff/T57-A.task.md)에 source 선택 옵션이 구현되어 있다. 조사 시점의 [`attention_policy.py`](../../../../AG3S/trajopt/attention_policy.py)는 기본 last/L8H2를 유지한다. T61 후보의 서빙 parity·closed-loop 확인과 기본 승격은 별도다. 또한 T61 offline replay는 voxel 0.020 등 자체 설정이 있으므로, 새 5mm surface graph와 과거 숫자를 직접 빼서 효과라고 해서는 안 된다.

새 프로토콜에는 기존 L8H2와 별도로 T61 선택을 **고정 reference control**로 추가했다. T61 test는 결과를 이미 검토했으므로, 이를 보고 새 extent 방법을 튜닝한 후 다시 “미관찰 test”라고 부르지 않는다.

## 6. T43–T61별 진행 범위와 재사용 판단

아래는 번호별 핵심과 관련 하위 실험의 요약이다. 모든 접미 실험의 raw trajectory를 재계산한 전수 검증은 아니다.

| 실험 | 확인한 수행 내용 | 새 프로토콜과의 관계 |
|---|---|---|
| [T43](../../../../AG3S/ag3s/docs/handoff/T43.task.md) | 충돌 구 coverage, held-object geometry, 연속 경로/서보 검사, HOLD 및 TO 개선 | 같은 전체 제어 파이프라인; feature segmentation은 아님. 개선 효과를 기하/제어 변경과 분리할 근거 |
| [T44](../../../../AG3S/ag3s/docs/handoff/T44.task.md) | ball_owned exclusion, permission 일관성, 후속 split/contact 비교 | target extent→constraint 연결은 이미 구현·실험. S6 재사용 |
| [T45](../../../../AG3S/ag3s/docs/handoff/T45.task.md) | palm/finger coverage audit와 후속 palm cover | segmentation 오류와 robot collision-model 오류를 분리 |
| [T46](../../../../AG3S/ag3s/docs/handoff/T46.impl.md) | F1/F2 target 잔여 보정, 후속 wrist geometry; [T46g](../../../../AG3S/ag3s/docs/handoff/T46g.impl.md)는 가림 때 손에 따라 centroid 이동 | S2/S3와 문제 동일·규칙 다름. T46g는 rigid transport이며 co-motion 관측 검증 자체는 아님 |
| [T47](../../../../AG3S/ag3s/docs/handoff/T47.verify.json) | attention mass/entropy head 선택, GT·grounding 검증; T47b aggregation 후속 | S1 source 비교 기반. 새 feature prototype 비교와 구분 |
| [T48/b](../../../../AG3S/ag3s/docs/handoff/T48b.verify.json) | prompt/state/image counterfactual, empty/slip/phantom 및 E0 parity | P 및 S5의 입력 개입 재사용. hidden K/V patch와는 다름 |
| [T49](../../../../AG3S/ag3s/docs/handoff/T49.verify.json) | target/destination noun attention 추출·평가, 1040 frame 추출 완료 | noun query 방법은 이미 존재; 새로 발명한 방법으로 쓰지 않음 |
| [T50](../../../../AG3S/ag3s/docs/handoff/T50.verify.json) | probe 학습·feature 부분·base/LoRA 비교로 instruction 둔감 원인 분리 | feature로 phase/task 정보를 읽는 시도 존재; dense extent 추정과 다름 |
| [T51/b](../../../../AG3S/ag3s/docs/handoff/T51b.verify.json) | action/noun 공통 선택, calibration/CV/bootstrap/held-out | head 선택·불확실성 평가 재사용; 최신 선택은 T61 참고 |
| [T52](../../../../AG3S/ag3s/docs/handoff/T52.verify.json) | map→seed→cluster→ball 실패 원인 분해 | S1/S2 진단과 동일 방향. 당시 grow_region 중간 집합 미저장 한계 보완 |
| [T53](../../../../AG3S/ag3s/docs/handoff/T53.verify.json) | 새 16 episode rollout 및 attention/noun/KV/GT batch 추출 | collector/source/GT 재사용. 현재 frame022도 여기서 유래 |
| [T54/b/c](../../../../AG3S/ag3s/docs/handoff/T54c.verify.json) | action/noun/역할별 결합, latch·attach 재생 보정, 새 16 episode 후속 | 융합 자체 이미 수행. 기존 N/B와 새 feature extent를 비교 |
| [T55](../../../../AG3S/ag3s/docs/handoff/T55.verify.json) | strong/HSR/noun+시간 지속성, dev 고정 및 test 완료 | P baseline을 재사용. task-ready와 3-class phase 구분 |
| [T56](../../../../AG3S/ag3s/docs/handoff/T56.verify.json) | HELD self-label 및 손 동반 운동 S2, dev/test 완료 | co-motion primitive 동일. 목적은 후보 선택, full segmentation/독립 phase는 아님 |
| [T57](../../../../AG3S/ag3s/docs/handoff/T57-A.impl.md) | attention source 및 task gate serving 옵션 구현·검사 기록 | 구현 존재. 조사한 기록만으로 전체 새 source closed-loop 완료/승격을 주장하지 않음 |
| [T58](../../../../AG3S/ag3s/docs/handoff/T58.verify.json) | 장애물 pilot: post 있는 ON/OFF 6/6 접촉, 일부 target-free 과잉 제거 및 손 형상 빈틈 진단 | extent·query geometry 병목의 실행 증거. feature graph 구현 증거는 아님 |
| [T59](../../../../AG3S/ag3s/docs/handoff/T59.verify.json) | phase별/비분할 grounding attribution, 완료·return 창 분석 | P/identity lifecycle 및 평가 범위 진단 재사용 |
| [T60](../../../../AG3S/ag3s/docs/handoff/T60.task.md) | P/H 종료 술어 및 이전 후보 재집계; 후속 stall 분석 | 평가 창을 임의로 다시 정하지 않음. 새로운 segmentation 제안은 아님 |
| [T61](../../../../AG3S/ag3s/docs/handoff/T61.verify.json) | P 창 head 재선택·선택 고정·별도 16 episode held-out | 최신 고정 source control로 사용. 신규 feature extent의 효과와 분리 |

## 7. develope_PLAN.md와의 일치 및 실행 우선순위

[§0.10](../../../active/plans/develope_PLAN.md)은 이미 **identity와 extent 분리**, ball 전후 손실, 선택적 voxel/surface 제외, multiview/time consistency, 불확실 영역 처리를 명시한다. [DX-01/02](../../../active/plans/develope_PLAN.md)는 같은 identity의 extent 비교 및 같은 상태의 HSR/strong-rule 비교를 요구한다. 새 프로토콜의 목표·통제·평가 방향은 이 계획과 매우 가깝다. 다만 계획의 존재가 모든 후보의 구현 완료를 뜻하지는 않는다.

계획은 경계 보존 방법 **하나**를 우선 개발하고 HSR 추가 가치는 보조로 두라고 한다. 따라서 다음 실행 순서를 권한다.

1. T53/T54/T55/T56/T60/T61의 loader·label·split·평가기를 재사용하고 새 collector parity만 확인한다.
2. 기존 source L8H2, noun L10H7, T61 action mean L10H7 및 T54 N/B를 고정 비교군으로 둔다. 같은 cloud/필터/extent builder에서 source 효과를 먼저 분리한다.
3. **주 실험 하나:** 같은 source/identity에서 기존 region/ball 또는 ball_owned 처리, geometry-only, feature+geometry extent를 비교한다. 선택된 source가 좋아진 효과를 extent 개선으로 계산하지 않는다. Oracle identity는 별도 진단이다.
4. 필요할 때만 T56 motion primitive를 extent 시간 일관성에 연결한다. 처음부터 graph·OT·HMM·모든 feature layer를 동시에 확장하지 않는다.
5. Phase가 실제 병목일 때 T55 baseline에 3D residual을 추가한다. T56 HELD conditioning을 그대로 두고 독립 hold onset detector라고 부르지 않는다.
6. T60 P/H 및 전체 창을 같은 정의로 보고, 고정 trace와 matched closed-loop를 분리한다. 기존 결과를 이미 확인한 episode와 겹치지 않는 새 group을 최종 test로 잠근다.

## 8. 추가 dump에 대한 영향

- **유지:** SigLIP encoded / canonical image_tokens, 선택 prefix features, depth·RGB·K/T·전처리 mapping·timestamp·FK/손 상태. 기존 실험과 다른 feature-aware extent를 검증할 핵심이다.
- **기존 연구 재사용용 유지:** noun token 위치와 KV(특히 T55 v5), d0 action head/query 정보, noun rows, latch/attach/detach와 실제 실행 구간. CORE의 전체 d0 action probs와 prefix KV는 T61 mean source와 T55 v5를 포함한다.
- **대부분 frame에서 생략 가능하다는 판단 유지:** 전층 MLP gate/up/act, full attention logits, SigLIP 전층 full attention, canonical tensor의 중복 사본. T43–T61 재현에 이것들 모두가 필요하지는 않다. 별도 mechanism audit sample에는 보존한다.
- CORE의 후속 denoise selected-map이 L8H2만이면 T61 후보의 후속 step 분석은 불가능하다. 그런 분석을 사전 선택할 때만 mean:L10H7 map을 추가하고 budget을 다시 계산한다. d0 T61 재현만을 위해 모든 denoise tensor를 저장할 필요는 없다.

## 9. 기록 해석상 주의와 이번 수정

T55/T56/T61 verify에는 과거 top-level `verdict`, `status`, `not_measured`와 나중에 추가된 `test`/`heldout`이 공존한다. 예를 들어 T56 상단은 dev 완료·STOP이지만 하단 test는 완료다. 이번 판단은 날짜·원본 경로·하위 결과를 함께 확인했다. 과거 실험의 STOP/승인 지시는 현재 감사 작업의 중단 지시로 해석하지 않았다.

기존 handoff/verify 원본과 serving 설정은 수정하지 않았다. 새 감사 보고서와 근거 JSON을 추가하고, 실험 프로토콜·설정·덤프 정책에 기존 연구 재사용과 남은 차이를 반영했다. **새 방법의 성능 개선이나 연구 novelty를 확정한 것은 아니다.**
