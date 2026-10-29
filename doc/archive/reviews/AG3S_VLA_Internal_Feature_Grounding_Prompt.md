# AG3S-VLA: Frozen VLA 내부 특징 기반 Target–Obstacle Segmentation 연구

> **2026-10-15 실행 명세:** 후속 agent는 [상세 실험 방법론](../../active/reviews/vla_dump_20261015/EXPERIMENT_PROTOCOL_KO.md)과 [덤프 저장 정책](vla_dump_20261015/DUMP_POLICY_KO.md)을 실행 정본으로 사용한다. 아래는 초기 연구 제안이며, 실행 순서·비교군이 다르면 후속 명세를 따른다. 특히 feature-only 성공 여부와 관계없이 geometry-only/feature+geometry를 비교한다.

## 1. 연구 목적

현재 AG3S는 π0.5의 Action-to-Image Attention(L8H2, 16×16)을 3D Point Cloud로 Lifting하여 Target과 Obstacle을 분리한다.

하지만 Attention은 객체 분할을 위해 학습된 정보가 아니므로, Target과 Obstacle이 인접하거나 일부 가려지면 두 객체가 혼합되는 문제가 발생한다.

본 연구의 목표는 **별도의 YOLOE, Grounding DINO, SAM 등의 Segmentation 모델이나 Mask 지도 학습 없이, Frozen π0.5의 내부 특징만으로 Target–Obstacle 분리 성능을 개선하는 것**이다.

핵심 가설:

- Attention은 VLA의 행동 목표와 관련된 Target Seed를 제공한다.
- SigLIP 및 PaliGemma의 Image Features는 객체 영역 확장에 활용할 수 있다.
- Depth 및 3D Geometry를 결합하면 Target–Obstacle 혼합을 줄일 수 있다.
- 최종적으로 외부 Segmenter 없이 안전한 3D Contact Constraint를 생성할 수 있다.

단, 이 가설들이 반드시 성립한다고 가정하지 말고 실험으로 검증한다.

## 2. 코드 및 기존 실험 분석

구현 전 현재 Repository와 실험 기록을 확인한다.

- π0.5 Attention 추출 경로 및 L8H2 선택 근거
- 16×16 Attention → 224×224 이미지 → 3D Point Cloud 매핑
- Head/Left Wrist/Right Wrist Camera 처리 방식
- Target/Obstacle 분리 및 ESDF 전달 구조
- 기존 T47~T58의 Attention 실패 및 Occlusion 사례
- T49/T53의 Noun Attention 및 Hidden Feature 분석 결과

기존 구현과 중복되는 작업은 하지 않는다.

추가로 다음 특징의 실제 추출 가능 여부와 텐서 형상을 확인한다.

| Feature | 목적 |
|---|---|
| SigLIP Image Features (projector 이전) | 패치별 Appearance Similarity |
| Projector 이후 Image Features | VLA 입력 공간 특징 비교 |
| PaliGemma Layer별 Image Hidden States | 언어·시각 융합 특징 비교 |
| Image–Image Attention / Q, K, V | 패치 간 연관성 분석 |
| Action/Noun Attention | Target Seed 생성 |
| Action Chunk 및 Action Expert 출력 | 행동 관련성 검증 |

기존 Serving 성능에 미치는 영향을 측정하고, 새로운 특징 추출은 기본 OFF로 유지한다.

## 3. 알고리즘 구현 및 단계별 실험

### Experiment A — Attention + SigLIP Feature Similarity

**가장 먼저 구현할 실험이다.**

1. 기존 L8H2 Action Attention에서 Target Seed를 추출한다.
2. SigLIP의 패치별 Feature를 추출한다.
3. Seed에 해당하는 Feature의 가중평균으로 Target Prototype을 생성한다.
4. Target Prototype과 나머지 패치의 Cosine Similarity를 계산한다.
5. Feature Similarity Map을 생성하고 Target 영역 확장에 사용한다.
6. 기존 Attention Threshold 방식과 비교한다.

핵심 수식:

`Target Prototype = WeightedMean(Attention Seed Features)`

`Similarity(i) = Cosine(Target Prototype, Image Feature(i))`

검증 항목:

- Target IoU / Recall
- Target–Obstacle Contamination
- 잘못된 Target 영역 확장
- 타깃과 장애물이 인접할 때의 분리 성능
- 추론 지연

### Experiment B — Layer별 Feature 및 Image–Image Attention

Experiment A의 결과를 기준으로 다음 특징을 비교한다.

- SigLIP Feature
- Projector 이후 Feature
- PaliGemma L4/L8/L12 Image Hidden States
- Image–Image Self-Attention

특히 TokenCut, CutLER의 Feature Affinity 및 Graph Partition 아이디어를 참고한다.

단, Self-Attention이 동일 객체의 연결성을 보장하지 않는다는 점에 유의한다.

목표는 객체 구분에 유리한 Feature와 Layer를 실험적으로 선정하는 것이다.

### Experiment C — Attention-guided 3D Feature Graph

Experiment A/B에서 유효성이 확인된 특징을 3D Geometry와 결합한다.

그래프의 노드는 Image Patch 또는 3D Point이며, Edge Weight는 다음 정보를 반영한다.

1. Appearance Affinity: Feature Cosine Similarity
2. Geometric Affinity: 3D Distance
3. Depth Discontinuity: 깊이 경계
4. Spatial Connectivity: 이미지 및 3D 공간의 연결성

Attention Seed를 기반으로 Seeded Graph Cut, Label Propagation 등의 방법을 검토한다.

최종 출력:

- Target Region
- Obstacle Region
- Unknown Region

**Unknown 및 신뢰도가 부족한 영역은 충돌 회피 대상으로 유지한다.**

Target 영역이 확장되었다는 이유만으로 장애물을 ESDF에서 제거해서는 안 된다.

### Experiment D — Multi-view Fusion 및 Occlusion

Head/Left Wrist/Right Wrist Camera의 Attention과 Image Features를 결합한다.

- 각 카메라에서 독립적인 Target Candidate 생성
- Depth 및 Camera Extrinsic을 통한 3D Association
- 관측 가시성, Depth 유효성, Feature Consistency 기반 융합
- 기존 Target ID의 시간적 일관성 검증

Head Camera에서 Target이 가려져도 Wrist Camera에서 관측할 수 있다면 이를 활용하도록 한다.

단, 가려진 카메라의 낮은 Attention으로 정상 관측을 억제하지 않는다.

### Experiment E — Action Influence 검증 (Offline)

Attention이 높은 패치가 실제 Action Generation에도 영향을 주는지 확인한다.

- Image Patch/Feature Perturbation
- 동일한 Initial Noise로 Action Chunk 재추론
- Perturbation 전후 Action Difference 측정
- Attention Score와 Action Influence 간 관계 분석

Action에 영향을 주는 영역이 반드시 Target은 아니라는 점을 고려하여, 장애물과 로봇 자체에 대한 영향도 분리해서 분석한다.

이 실험은 우선 Offline으로 수행한다.

## 4. Baseline 및 평가

동일한 π0.5, RGB-D 입력, ESDF, Trajectory Optimizer 조건에서 비교한다.

| Baseline | 구성 |
|---|---|
| B0 | 기존 AG3S Attention-only |
| B1 | Attention + SigLIP |
| B2 | Attention + 최적 Layer Features |
| B3 | Attention + Feature + 3D Graph |
| B4 | B3 + Multi-view Fusion |
| External | YOLOE / Grounded SAM / SAM2 (사용 가능한 경우) |

주요 평가 지표:

- Target IoU / Precision / Recall
- Target–Obstacle Contamination
- False Target Exclusion Rate
- Occlusion 상황 Target ID 유지율
- Collision Rate
- Task Success Rate
- 추가 추론 Latency 및 GPU Memory

Segmentation Ground Truth는 가능하면 MuJoCo의 Object ID 및 Geometry 정보를 이용한다. 외부 Segmenter의 출력을 검증 없이 정답으로 사용하지 않는다.

기존 성공 사례뿐 아니라 T58과 같이 장애물이 타깃에 인접하거나 가리는 실패 사례를 반드시 포함한다.

## 5. 구현 및 안전성 원칙

- π0.5 모델 가중치는 수정하지 않는다.
- 별도 Segmentation Model 또는 Mask Supervision은 사용하지 않는다.
- 기존 AG3S 파이프라인과 호환되도록 구현한다.
- 모든 신규 기능은 Flag로 제어하며 기본 OFF로 유지한다.
- 추출한 Feature의 차원, Layer, 정규화 방식 및 이미지 좌표 정렬을 검증한다.
- 기존 Target/Obstacle 구분, ESDF, Link-specific Contact Constraints의 안전 규칙을 유지한다.
- Target Seed가 잘못된 경우 잘못된 영역 확장이 발생하는지 반드시 평가한다.
- 효과가 검증되지 않은 알고리즘을 최종 Serving 기본값으로 적용하지 않는다.

## 6. 구현 순서 및 결과 보고

우선 **Experiment A를 구현하고 검증**한다.

결과가 유의미하다면 Experiment B → C → D 순서로 확장한다. 성능 개선이 없으면 무리하게 다음 단계를 구현하지 말고 실패 원인과 수정 가설을 제시한다.

각 단계에서 다음 항목을 보고한다.

1. 구현 내용 및 변경 파일
2. 실험 환경과 Baseline
3. 정량적 비교 결과
4. 대표적인 성공/실패 사례
5. 기존 AG3S 대비 개선 및 악화된 부분
6. 추가 계산 비용
7. 다음 단계 진행 여부 및 근거

실험 결과와 측정값을 생성하거나 추정해서 작성하지 않는다.

## 7. 최종 연구 목표

본 연구가 입증하려는 Contribution은 다음과 같다.

**Policy-Conditioned, Segmentation-Free 3D Target Grounding from Frozen VLA Internal Representations.**

기존 Segmentation 모델은 객체 영역을 예측하도록 학습된 별도 모델이다. 반면 AG3S는 VLA의 행동 생성 과정에서 이미 계산된 Attention과 Visual Features를 활용한다.

이때 Novelty는 Attention 사용 자체가 아니라 다음의 결합에 있다.

- VLA의 Action-conditioned Target Selection
- Frozen Visual Features를 이용한 Target Region Expansion
- 3D Geometry 기반 Target–Obstacle 분리
- Task-dependent Collision Constraints와의 직접적인 연계

궁극적으로 **외부 Segmenter 없이도 VLA의 행동 의도에 맞는 3D Target을 안정적으로 식별하고, 충돌 회피에 활용할 수 있는지** 입증하는 것이 목표다.

구현을 시작하기 전에 기존 Repository와 실험 기록을 분석하고, Experiment A의 구체적인 구현 계획을 확정하라. 이후 구현·검증을 수행하고 실제 결과를 보고하라.
