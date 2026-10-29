# AG3S: VLA-Guided 3D Target Segmentation

> Claude Code용 방법론·실험 지침 | 2026-10-20 | **연구 가설이며 아직 검증되지 않음**

## 1. 목표와 전제

장면 내 **모든 객체를 개별 분할하지 않는다.** Frozen π0.5 VLA의 작업 조건부 **Attention·spatial Image Feature**와 RGB-D 기하학으로 **현재 조작 Target의 관측 3D 표면 vs 나머지 관측 표면**을 이진 분리한다.

- **Target:** 현재 작업 대상의 관측 표면
- **Non-target:** 그 외 관측 환경 표면 → 충돌 제약 유지 (개별 객체 ID 불필요)
- **Unknown:** 가림/깊이 누락 등 미관측 공간 → 별도 불확실성 정책
- **Robot:** URDF/FK 기반 self-mask로 환경 Point에서 제외

Target에 속한다고 **전 링크 충돌 면제**가 되는 것은 아니다. 허용된 작업 단계·로봇 링크에 한해 접촉하고, Grasp 후에는 물체를 *attached collision object*로 취급한다.

### 공통 처리

RGB-D 정렬 → Point Cloud와 Robot Self-mask → VLA Attention/Feature를 3D Point 또는 Supervoxel에 대응 → Target Seed 추정 → 아래 A–D 중 하나로 분할.

현재 π0.5의 16×16 Attention → 224×224 RGB → 640×480 Depth 매핑을 참고하되, **crop/resize/calibration 및 token 좌표를 검증**한다. 분할용 Feature는 공간 위치가 있는 Image Token에서 추출해야 하며, **전역 512-D 요약 벡터는 그대로 사용 불가**하다.

## 2. 네 가지 방법

| 방법 | VLA 내부값의 역할 | 분할 경계를 정하는 정보 | 목적 |
|---|---|---|---|
| **A. Attention Seed + 3D Clustering** | Seed 선택 | 3D 거리 | 단순 기하 기준선 |
| **B. Attention Seed + RGB-D Graph Cut** | 동일 Seed 선택 | 색상·깊이·표면 법선·공간 연결성 | 정교한 Geometry-only 기준선 |
| **C. Feature-Guided Graph** | Seed + Feature Unary | B의 기하 비용 + VLA Feature 유사도 | Feature의 추가 기여 |
| **D. Joint Graph Optimization** | Seed + Attention Unary + Feature Unary/Pairwise | VLA 내부값 + 3D Geometry | 최종 확장안 |

### A — Attention Seed + 3D Clustering

Attention Seed를 3D에 투영하고 Euclidean Clustering을 수행한다. **Seed 포함 Cluster만 Target**으로 지정한다. 접촉 객체 병합과 Point 누락에 따른 과분할에 취약하다.

- **실제 분할 연구:** [FEC: Fast Euclidean Clustering for Point Cloud Segmentation — Cao et al., *Drones* 2022](https://doi.org/10.3390/drones6110325). Point Cloud Segmentation에 Euclidean Clustering을 적용·개선. *VLA Seed 자체를 제안한 논문은 아님.*

### B — Attention Seed + RGB-D Graph Cut

Point/Supervoxel을 Node로, 3D 인접 관계를 Edge로 구성한다. Seed는 초기 Target 제약으로만 사용하고 **RGB·Depth·Normal·Boundary** 기반 Unary/Pairwise 비용을 최적화해 Target/Non-target을 분리한다. Attention 응답 자체는 경계 비용에 넣지 않는다.

- **실제 분할 연구:** [Learning of Perceptual Grouping for Object Segmentation on RGB-D Data — Richtsfeld et al., *JVCIR* 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC3902499/): RGB-D Surface Patch 관계 + Graph Cut으로 객체 분할.
- **실제 분할 연구:** [Graph-Based Segmentation for RGB-D Data Using 3-D Geometry Enhanced Superpixels — Yang et al., *IEEE T-Cyb* 2015](https://doi.org/10.1109/TCYB.2014.2340032): 3D 기하 Superpixel + 그래프 병합으로 RGB-D 분할 개선.

### C — Attention Seed + VLA Feature Graph

**B와 같은 Seed·Geometry**를 사용하되, Target Seed의 대표 Feature와 각 Point의 공간 Feature 간 **Cosine Similarity를 Target Unary**에 추가한다. 이것이 Target 경계 결정에 실제로 도움이 되는지 검증한다.

- **실제 분할 연구:** [TokenCut: Self-Supervised Transformers for Unsupervised Object Discovery Using Normalized Cut — Wang et al., *CVPR* 2022](https://arxiv.org/abs/2202.11539): Transformer Token Feature 유사도 그래프로 객체를 분할. *2D, 비-VLA.*
- **실제 분할 연구:** [3D Graph Neural Networks for RGBD Semantic Segmentation — Qi et al., *ICCV* 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Qi_3D_Graph_Neural_ICCV_2017_paper.html): 2D Feature와 3D KNN Graph로 RGB-D 분할. *학습형 GNN이므로 제안한 Frozen-Feature Graph Cut과 동일하지 않음.*

### D — Attention + Feature + Geometry Joint Graph Optimization

Attention을 **Target 가능성(Unary)**, VLA Feature를 **Target Seed 유사도(Unary)·인접 Feature 유사도(Pairwise)**, RGB-D를 **표면 연속성(Pairwise)**으로 사용한다.

**목적함수 개념:** `E = U_seed + λa·U_attention + λf·U_feature + λg·Pairwise_geometry + λp·Pairwise_feature`

불확실한 영역은 잘못된 Target 접촉 허용으로 이어지지 않도록 보수적으로 처리한다. Temporal Consistency는 단일 프레임 분할이 검증된 후 별도로 추가한다.

- **실제 분할 연구:** [Talking to DINO — Barsellotti et al., *ICCV* 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Barsellotti_Talking_to_DINO_Bridging_Self-Supervised_Vision_Backbones_with_Language_for_ICCV_2025_paper.html): Attention 및 Patch Feature·언어 정렬을 실제 분할에 활용. *2D·비-VLA·비-Graph Cut.*
- **실제 분할 연구:** [DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation — Yin et al., *CVPR* 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Yin_DFormerv2_Geometry_Self-Attention_for_RGBD_Semantic_Segmentation_CVPR_2025_paper.html): Depth 기하 정보를 Attention에 반영하여 실제 RGB-D 분할 개선. *학습형 Backbone이며 제안한 Graph 방식과 다름.*

**문헌 해석:** 위 7편은 **모두 실제 Segmentation을 수행하거나 개선한 논문**이다. 다만 네 방법의 **VLA Seed/Feature + RGB-D Graph + Contact Permission** 전체를 그대로 구현한 선행연구가 아니다. 특히 D는 두 관련 분할 연구의 서로 다른 요소를 참고한 **검증 대상 설계안**이다.

## 3. 실험 및 Ablation

### E0. Feature Probe (최우선)

- 같은 Target 표면 / Target–접촉 장애물 / Target–유사 객체 사이의 **Cosine Similarity 분포 및 Same-vs-Different AUC** 비교.
- 카메라·가림·시점별 검사. **Feature 구분력이 없다면 C/D 구현 전 추출 위치와 Feature 종류부터 재검토.**

### E1. Segmentation 비교

- **A / B / C / D1(B+Attention Unary) / D2(Full D)** 비교.
- 동일 Scene·RGB-D·VLA Seed로 통제. `D2 vs Shuffled Feature`, `Predicted vs Oracle Seed`(진단)도 비교.
- 평가 조건: **이격 / 접촉·밀집 / 부분 가림 / 유사 객체 / 카메라·그리퍼 가림 / Pick→Transport**.
- 튜닝 데이터와 Held-out 테스트 분리. Temporal Consistency는 별도 Ablation.

### E2. 안전한 조작 평가

- 동일한 VLA Action Proposal에 각 Target Mask를 적용해 **잘못된 Target 면제와 Collision Constraint 변화**를 먼저 Offline 평가.
- 이후 MuJoCo Closed-loop에서 `VLA-only / Geometry B / Full D2`의 안전한 작업 성공률을 비교.
- 분할 방식 이외의 Planner·접촉 허용 규칙·시드 등은 가능한 한 동일하게 유지.

## 4. 평가지표

| 구분 | 핵심 지표 | 이유 |
|---|---|---|
| Feature | **Cosine 분포 / AUC** | VLA Feature에 객체 구분 정보가 있는가? |
| Segmentation | **Target 3D IoU**, Precision/Recall, Boundary F-score | 관측 Target 표면을 정확히 분리하는가? |
| **안전 핵심** | **Non-target Leakage**, **Near-target Leakage** | 장애물 Point가 Target Mask에 잘못 포함되는가? |
| Tracking | Identity Switch, Target Loss Rate | 타겟을 시간에 따라 유지하는가? |
| Execution | **Safe Task Success**, Forbidden Contact Rate, 불필요한 Target 차단률 | 안전성과 작업 성공을 동시에 달성하는가? |
| 효율 | 추가 처리 지연/P95, GPU 메모리 | 실시간 사용이 가능한가? |

- `Non-target Leakage = (실제 Non-target 중 Target으로 잘못 분류한 Point 수) / (평가 Non-target Point 수)`; `Near-target`은 Target 인접 장애물에 한정.
- `Safe Task Success = Task Success AND 비허용 접촉 없음`.
- **독립 Scene/Episode 단위** 통계·신뢰구간을 사용한다. 단순 IoU뿐 아니라 위험한 **false contact exemption**을 반드시 보고한다.

## 5. 구현·논문 판단 기준

1. **B vs A:** Geometry Graph의 실제 이득 확인.
2. **C vs B:** VLA Feature가 기하학만으로 불명확한 Target 경계를 개선하는지 확인 (**핵심 가설**).
3. **D2 vs C/D1:** Attention과 Feature 결합의 독립적 이득 확인.
4. **최종:** Leakage 감소가 **Safe Task Success 증가**로 연결되는지 확인.
5. C/D가 B보다 낫지 않거나 위험한 오분류가 늘면 **“VLA Feature가 Segmentation을 개선한다”는 기여를 주장하지 않는다.** Target Selection 또는 Contact Permission 기여로 범위를 수정한다.

**논문 주장(실험 성공 시):** 전체 객체 Instance를 복원하지 않고, Frozen VLA의 작업 조건부 내부 표현과 3D 기하 단서를 결합하여 **작업 Target만 비타겟 관측 표면에서 선택적으로 분리**하며, **링크·단계별 접촉 제약**에 연결한다.
