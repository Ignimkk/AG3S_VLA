# 새 agent 시작 안내: 내부 feature + RGB-D 타겟 영역 추정

작성: 2026-10-15. 이 문서는 읽기 순서와 첫 실행 범위를 안내한다. 알고리즘·평가의 상세 실행 정본은 [EXPERIMENT_PROTOCOL_KO.md](../../../active/reviews/vla_dump_20261015/EXPERIMENT_PROTOCOL_KO.md) v1.2이다. 이번 안내 작성 시 새 성능 실험은 실행하지 않았다.

## 현재 작업 분담 — 사용자 확인

**새 프레임 dump는 다른 agent가 이미 수행 중이다.** 이 문서를 받는 분석 agent는 중복 수집 작업이나 별도 collector 개발을 시작하지 않는다. 수집 결과의 경로·manifest·schema를 확보하면 기존 프로토콜과 대조하고 reader/정합 검사에 연결한다. 그 전에는 기존 frame022 재현과 graph·평가기 구현을 진행할 수 있다. 필요한 필드가 누락되면 구체적인 항목과 사용 목적을 정리해 사용자에게 전달한다. 추가 수집이 이미 필요하거나 승인됐다고 가정하지 않는다.

## 1. 이미 얻은 긍정적 결과

T53 ep1839 frame022에서 **명사 attention으로 seed를 고르고 내부 visual feature의 cosine으로 영역을 확장하면, 명사 attention 단독보다 타겟 회수율과 IoU가 높았다.** 학습을 추가하지 않은 단일 프레임 진단이다.

평가: 3개 camera의 768 patch 중 score 상위 20개를 선택하고 patch 전체를 positive로 취급했다. GT target의 patch 내 실제 면적을 적분했다. 3D surface segmentation 성능과 동일하지 않다.

| 방법 | Precision | Recall | IoU |
|---|---:|---:|---:|
| action L8H2 map | 0.0792 | 0.1527 | 0.0550 |
| noun L10H7 map | 0.3255 | 0.6280 | 0.2729 |
| noun seed + SigLIP encoded prototype | 0.4832 | 0.9322 | 0.4669 |
| noun seed + projector prototype | 0.4837 | 0.9332 | 0.4676 |

따라서 첫 연구 가설은 **좋은 identity seed에서 내부 feature가 물체 영역을 확장하고, 3D surface 분리가 그 확장에서 섞인 비타겟 표면을 줄일 수 있다**는 것이다. 앞부분의 가능성은 이 프레임에서 실제 관측했고, 뒷부분의 성능 향상은 아직 실험해야 한다.

Precision 약 48%여서 혼입이 여전히 크다. Action×noun 곱은 noun 단독보다 나빴고, 같은 feature도 action seed에서는 효과가 작았다. Seed 선택과 extent 개선을 분리해야 한다. 이 사례를 보고 고른 방법을 이 사례에서 다시 평가한 값을 held-out 성능으로 쓰지 않는다. 작업 단계 추정의 개선은 이 단일 프레임 실험에서 검증하지 않았다.

## 2. 읽기 순서

1. [DEPTH_GROUNDING_REVIEW_KO.md](DEPTH_GROUNDING_REVIEW_KO.md): §2–3부터 읽는다. 실제 수치·혼합 patch·depth 진단과 아직 미검증인 제안을 구분한다. [grounding_metrics.json](grounding_metrics.json), [grounding_top20.png](grounding_top20.png)가 근거다.
2. [HISTORICAL_PROTOCOL_AUDIT_KO.md](HISTORICAL_PROTOCOL_AUDIT_KO.md): T43–T61의 중복·차이와 재사용 경로. T54 attention 융합, T55 predicate 평가, T56 co-motion, T61 고정 head 선택을 재구현하지 않는다.
3. [EXPERIMENT_PROTOCOL_KO.md](../../../active/reviews/vla_dump_20261015/EXPERIMENT_PROTOCOL_KO.md): v1.2 실행 정본. 우선 §0–7, §11–14를 읽는다. [experiment_spec.json](../../../active/reviews/vla_dump_20261015/experiment_spec.json)은 실행 전 채워야 하는 설정이며 완성된 runner가 아니다.
4. [3D Surface Graph 개선 가이드](../../../active/reviews/AG3S_3D_Surface_Graph_and_Parameter_Tuning_Guide.md): §15 필수. 최초 실패 후 당위성 확인·개선·대조 실험·전체 dev 재평가를 수행한다. 본문의 binary Graph Cut 예시와 정본의 연속 graph를 구분한다.
5. [DUMP_POLICY_KO.md](DUMP_POLICY_KO.md): 추가 수집 시 필수. CORE/SURVEY/FULL_AUDIT, canonical tensor, RGB-D/좌표/시간/실행 메타데이터 계약을 따른다. 기존 collector가 새 profile을 이미 지원한다고 가정하지 않는다.
6. [develope_PLAN.md](../../../active/plans/develope_PLAN.md): §0.10 및 DX-01/02. Identity와 extent를 분리하고 경계 보존 방법 하나를 우선 검증한다.

원래 [Internal Feature Grounding Prompt](../AG3S_VLA_Internal_Feature_Grounding_Prompt.md)와 [REPORT_KO.md](REPORT_KO.md)는 배경 자료다. 후속 감사의 정정을 적용하고 v1.2를 실행 기준으로 쓴다.

## 3. 첫 작업: Attention + Feature graph + 3D

첫 목표는 **같은 seed와 관측 surface에서 geometry-only보다 feature+geometry가 타겟 경계를 더 정확하게 추정하는지** 확인하는 것이다. 전체 phase 체계를 새로 만드는 작업부터 시작하지 않는다.

1. **재현·정합(S0):** frame022 기존 진단을 재현하고 camera/patch mapping, depth 단위, K/T, timestamp, feature provenance를 검증한다. 범용 loader와 compact reader를 연결할 때 기존 producer를 재사용한다.
2. **semantic score(S1):** attention top-5 seed에서 L2-normalized feature의 가중 prototype을 만든다. 각 patch와 cosine을 계산한다. Smoke 구현은 noun+SigLIP/projector부터 가능하지만, dev 방법 선택 전에는 정본의 3 seed×5 feature 비교와 고정 source control을 완료한다. T61 action mean L10H7과 prefix noun L10H7을 혼동하지 않는다.
3. **관측 surface(S2):** depth를 역투영하고 FK robot filter·관측 support 처리를 적용한다. Depth discontinuity를 먼저 나눈 뒤 같은 surface 안에서 voxel을 만든다. 하나의 patch에 target/손/table이 섞이면 분리된 노드에 불확실한 evidence를 연결한다. Patch feature를 복제한 만큼 독립 semantic 증거가 늘었다고 계산하지 않는다.
4. **joint graph(S3):** 노드는 관측 surface/voxel, edge는 유효한 local surface 이웃이다. 위치·법선·feature 유사도로 edge weight를 만들고 prototype score를 unary evidence로 연결한다. `0≤m≤1`에서 `(m-s)^TΛ(m-s)+λm^TLm`을 최소화한다. Feature 항을 뺀 동일 graph가 geometry-only 대조군이다. Unary가 없는 component는 unknown으로 둔다. 상세 수치·threshold·dev 선택 규칙은 정본을 따른다.
5. **대조 평가:** 기존 serving/T54 N·B, feature-only, geometry-only, feature+geometry를 같은 데이터에서 비교한다. Source 선택 효과와 extent 효과를 분리하고, visible 3D IoU/recall·neighbor/robot/support 혼입·false-exemption·latency를 보고한다. GT는 evaluator에만 전달한다.
6. **실패 개선과 확장:** 최초 FG 결과가 나쁘면 가이드 §15에 따라 seed/node/edge/unary 원인을 진단하고 근거 있는 최소 수정·재실험을 수행한다. 공간 경계의 이득을 확인한 뒤 시간 일관성을 확장할 수 있으며, 가림/시간 불일치가 실패 원인으로 확인된 경우에도 제한적으로 추가한다. T56 co-motion primitive는 재사용하되 HELD 조건부 self-label을 독립 grasp detector로 부르지 않는다. Phase는 T55 평가를 재사용하는 별도 후속이다.

원래 frame과 이미 분석한 기존 episode는 regression/dev 용도로 표시한다. 새로운 group의 held-out을 방법 고정 후 평가한다. 모든 구간을 full dump할 필요는 없다. 필요한 feature/depth가 없으면 우선 진행 중인 수집의 산출물에 포함되는지 확인한다. 그래도 누락되면 재추론·검증된 replay 가능 여부와 필요한 보완 항목을 보고한다.

## 4. agent에게 전달할 작업 문구

> START_HERE_KO.md의 순서로 근거·감사·v1.2 프로토콜을 읽고 기존 코드를 재사용하라. 첫 목표는 S0–S3: 동일 seed/관측 surface에서 geometry-only 대비 feature+geometry extent의 추가 가치를 검증하는 것이다. frame022 재현 후 batch 입력·surface graph·비교 평가를 구현하고 실제 실행하라. T54/T55/T56/T61을 새 방법으로 재구현하지 말고 재사용하라. 새 dump는 다른 agent가 수집 중이므로 중복 수집하지 말고, 그 산출물을 받아 schema·정합·split을 확인하라. 누락 필드만 구체적으로 보고하라. 결과에는 episode별 지표, 실패/결측, 비용, 재현 명령, 실행/미실행 범위를 남겨라. 최초 Attention+Feature+3D 실패로 끝내지 말고, 3D Surface Graph 개선 가이드 §15의 당위성 확인→최소 수정→대조 실험→전체 dev 재평가를 수행하라. 각 cycle의 최초 실패와 개선/기각 근거를 보존하라. 새 단계 분류기와 closed-loop 확장은 공간 비교·개선의 결론 뒤에 판단하라.

첫 완료 산출물: 실행 가능한 범용 비교 runner, 입력·split·설정 manifest, 동일 기준의 baseline/G/FG 결과, 실패 overlay, episode별 지표와 실행 비용. 최초 실패 시 개선 registry·round별 가설/대조/결정 및 음성 결과 종료 근거도 포함한다. Pilot만으로 일반적 성능 향상이나 novelty를 확정하지 않는다.
