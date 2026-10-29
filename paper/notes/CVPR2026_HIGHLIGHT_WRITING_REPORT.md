# CVPR 2026 Highlight 4편의 글쓰기 흐름과 문장 표현 — AG3S VLA 원고에 적용하기

작성: 2026-10-07 KST · 대상: `paper/cvpr2026_ref_paper/`의 Highlight 논문 4편(본문 전체, 참고문헌 제외)
목적: 내용이 아니라 **논문을 끌고 가는 흐름과 문장 표현**을 파악한다. 앞선 보고서
([CVPR2026_REFERENCE_STRUCTURE_REPORT](CVPR2026_REFERENCE_STRUCTURE_REPORT.md), VLA 5편)를 대체하지 않고 보완한다.

| 약칭 | 논문 | 유형 |
|---|---|---|
| BiPreManip | Learning Affordance-Based Bimanual Preparatory Manipulation through Anticipatory Collaboration | 새 문제 정의 + 학습 방법 |
| GeoPredict | Leveraging Predictive Kinematics and 3D Gaussian Geometry for Precise VLA Manipulation | VLA에 모듈 추가 |
| AutoMoMa | Scalable Trajectory Generation for Whole-Body Mobile Manipulation | 시스템·데이터(기존 표현 + GPU 최적화의 결합) |
| Visual-RRT | Finding Paths toward Visual-Goals via Differentiable Rendering | 기존 두 기법의 결합(RRT + differentiable rendering) |

---

## 0. 한 문장 요약

> Highlight 논문은 **새 기술이 아니라 선명한 문제**로 시작한다. "기존 방법은 X를 가정한다 → 그런데 실제로는 Y다 →
> 둘을 그냥 합치면 Z 때문에 안 된다 → 그래서 우리는 이렇게 했다"의 흐름을, **구체적인 예시 하나**로 처음부터 끝까지 끌고 간다.

Visual-RRT와 AutoMoMa는 기존 기술을 결합한 논문이다. 결합 자체를 기여로 내세우지 않고,
**"결합이 왜 non-trivial한가"**를 설명하는 데 지면을 쓴다. 주제가 특별하지 않은 우리에게 가장 중요한 교훈이다.

---

## 1. 네 편의 흐름 비교

### 1.1 Abstract

| 논문 | 문장 순서 | 특징 |
|---|---|---|
| Visual-RRT | 분야의 강점(널리 쓰임, 이유) → **However, 기존 방법은 X를 요구한다, while 실제 응용은 Y만 준다** → we propose X that enables … by unifying A from … with B from … → We further introduce (i)(ii) → 실험 → **bridging the gap between … and …** | 가정과 현실의 불일치를 한 문장에 대비 |
| GeoPredict | **VLA는 강하지만 reactive·2D-centric이라 정밀 3D 작업에서 불안정** → we propose → 두 모듈 각각 한 문장 → **실용 속성 하나(추론 시 추가 비용 거의 없음)** → 실험 | 한계를 형용사 두 개로 압축 |
| AutoMoMa | 왜 어려운가(상태 공간 조합 폭발) → 기존 획득 방법들은 노동 집약적이거나 계산이 너무 비쌈 → **The core bottleneck is …** → Here we introduce → **구체 수치(5,000 episodes/GPU-hour, 80배)** → 실험이 드러낸 통찰 → 영향 | "core bottleneck" 한 구로 문제를 못 박음 |
| BiPreManip | **일상 사례 두 개(평평한 태블릿, 책상 위 펜 뚜껑)** → 이런 작업이 요구하는 것 → 새 문제 이름 정의 → 접근 → 실험 | 예시로 시작해 독자가 문제를 즉시 그림 |

공통점: 4–7문장. 첫 두 문장 안에 **한계가 한 구로 고정**된다("reactive and 2D-centric", "require explicit goal configurations",
"the core bottleneck", "difficult to directly grasp").

### 1.2 Introduction의 논리 전개

**Visual-RRT (가장 교과서적)**
1. 분야 확립: RRT는 robust하고 이론적 보장이 있어 표준이 되었다. 개선 연구 나열 한 문장.
2. **"Despite their advances, … assume that …"** — 기존 방법의 *숨은 가정*을 꺼낸다.
3. **"In contrast, several emerging applications …"** — 현실은 그 가정을 만족하지 않는다.
4. **"To bridge this gap, we extend … However, this extension is non-trivial since …"** — 결합이 왜 어려운지 구체적으로.
5. **"Our key idea is to …"** — 핵심 아이디어 한 문장.
6. 각 추가 기법을 *기존 방법의 구체적 결함*에 대응시켜 소개("as existing RRT states (e.g., path cost) are not designed to capture …").
7. 실험 한 문장 → 기여 3개.

**GeoPredict**
1. VLA의 성공 → **"However, they operate primarily in 2D …"**
2. **"To move beyond purely reactive policies, recent work has begun to …"** — 기존 해법 흐름 소개
3. **"Although these methods provide useful …, they often …"** — 그 해법의 한계 두 개
4. **"we argue that two … capabilities are particularly valuable. First, … Second, … Crucially, …"** — *필요조건*을 먼저 선언
5. **"In this paper, we introduce … First, at the kinematic level … Second, at the geometric level …"** — 필요조건과 1:1로 모듈
6. 기여 (1)(2)(3)을 한 문단 안에 inline

**AutoMoMa**
1. 왜 중요한가 → 왜 어려운가(탐색 공간 폭발, 해가 sparse)
2. **획득 방법을 하나씩, 한 문장에 장점+단점**: "Teleoperation yields …, but …", "Simulation-based RL automates …, but …",
   "Planning-based methods ensure …, yet …", "End-to-end learning … but …"
3. **"Collectively, these constraints have forced prior datasets to compromise on scale, diversity, or kinematic fidelity"** — 종합 진단
4. "We address this with …" → 수치
5. **실험이 드러낸 통찰을 Intro에 미리**: "confirming that data scarcity—not algorithmic limitations—has been the binding constraint"
6. 기여 (i)(ii)(iii) + **"Together, these contributions establish … as the first framework to bridge …"**

**BiPreManip**
1. 구체 사례 → **"While such behaviors appear effortless for humans, they pose significant challenges for robots, requiring …"**
2. 기존 연구를 **유형으로 분류**(symmetric / sequential-independent / complementary) → "However, most prior work assumes …"
3. "In contrast, our work focuses on …" — 새 문제 이름
4. 방법을 "first envisions … then leverages …"로 요약
5. **Fig. 2를 따라가는 예시 walkthrough**: "For instance, as illustrated in Figure 2, when a capped pen lies on its side, our framework …"
6. 벤치마크 구성 → 기여 3개(첫째가 "We introduce the problem of …")

### 1.3 Related Work

- 2–3 절, 각 절은 **한 단락 서술 + "In this paper/work, we …"로 마무리**(Visual-RRT, BiPreManip, GeoPredict 모두).
- GeoPredict는 절마다 **"To address this gap, our GeoPredict …"**로 닫는다.
- AutoMoMa는 절 안에 굵은 소제목(Model-based planning / Learning-based planning / Teleoperation …)을 두고,
  **기존 데이터셋 비교표(Table 1)**를 Related Work에 넣는다(우리 Table 1과 같은 위치·역할).
- Visual-RRT는 Related Work 끝에서 범위를 정확히 한정한 first-claim을 쓴다:
  "To the best of our knowledge, this is the first method that integrates **A directly into B**".

### 1.4 Method

- **문제를 먼저 정식화**한다. Visual-RRT 3.1(visual-goal motion planning의 정의), AutoMoMa 3장 Preliminaries
  (상태 정의, 제약식 (2)–(6)과 각 제약이 무엇을 보장하는지 한 문장씩), GeoPredict 3.1(Problem Definition + Base VLA).
- 각 모듈 절의 전형 문장 구조 — **결함 → 해법 → 통찰 → 정의 → 효과**:
  - "While X enables …, naively applying them to all nodes leads to inefficient planning." (결함)
  - "To address this, we introduce …" (해법)
  - "Our key insight is that … serves as an implicit …" (통찰)
  - 수식 정의
  - "This allows … while preserving …" / "Importantly, this … preserves …" (효과와 보존되는 성질)
- GeoPredict는 구성요소마다 **"이것이 왜 도움이 되는가"**를 인과로 설명한다
  ("because joint motion exhibits inertia …, encoding past dynamics enables …").
- AutoMoMa는 **phase에 따라 충돌 모델을 바꾸는 방법**을 한 문단으로 설명한다(접근 단계와 조작 단계).
- BiPreManip는 3.1 Overview에서 전체 파이프라인을 **절 번호를 괄호로 달아** 한 문단에 서술한다.

### 1.5 Experiments

- **두 개의 상보적 과제로 나누어 평가**(Visual-RRT: 경로 계획 / 자세 복원 — 후자는 어려움의 한 부분을 *분리*해서 측정).
- 각 실험 절에 **Experimental setup / Baselines / Evaluation metrics / Results**를 굵은 머리말로 고정.
- **결과 문단은 숫자보다 메커니즘**: 각 baseline이 *왜* 실패하는지 설명한다.
  - BiPreManip: "For ACT, although … its predictions are primarily conditioned on RGB … As a result, the predicted actions may appear
    reasonable in the 2D image space but …"
  - Visual-RRT: "This gap arises from error propagation: inaccurate goal estimation in the first stage causes …"
- GeoPredict의 ablation은 **"building the model from the ground up"** — 구성요소를 하나씩 더하며 누적 수치를 서술하고,
  마지막 증가분을 **"validates our hypothesis: …"**로 연결한다.
- AutoMoMa의 결과 절은 **굵은 주제어 + 발견 + 이유**(Configuration space complexity / Local generalization / …).
- **정성 그림이 주장을 직접 보여준다**: Visual-RRT Fig. 8(탐색/활용 샘플 색 구분), GeoPredict Fig. 4(빨간 상자로 개선 부위 강조).

### 1.6 Conclusion과 Limitation

- Conclusion은 1문단: 무엇을 했나 → 핵심 기법 이름 반복 → 실험 → **분야에 주는 의미 한 문장**
  ("We bridge sampling-based planning and vision-centric robotics …").
- Limitation은 **짧고 구체적**:
  - Visual-RRT: 실패 그림(Fig. 9)과 함께 "occluded or symmetric robot parts can produce visually similar renderings".
  - AutoMoMa: "Limitations and Conclusion" 합본 — 알려진 기하에 의존, 구 근사의 부정확성, 그리고 future work.
  - GeoPredict: 한계를 인정하되 완화 요인을 함께 말한다("While acquiring … presents scaling challenges, the increasing availability … mitigates …").

---

## 2. 앞선 5편(VLA 일반 논문)과 Highlight 4편의 차이

| 측면 | VLA 5편 | Highlight 4편 |
|---|---|---|
| 문제 제시 | 한계를 나열하고 모듈로 대응 | **숨은 가정**을 꺼내고 현실과 대비 |
| 결합의 서술 | "we propose X with A and B" | **"this extension is non-trivial since …"** — 결합의 어려움을 먼저 설명 |
| 예시 | 거의 없음 | 한 예시를 Abstract–Intro–Method–Figure에 걸쳐 반복 |
| 모듈 동기 | "we introduce A" | **"naively doing X leads to Y; to address this …"** |
| 결과 서술 | 수치와 순위 | **baseline이 실패하는 이유** |
| 수사 | "state-of-the-art", "significantly" 빈번 | "consistently", "substantially", 제한된 first-claim |
| Limitation | 대체로 없음 | 짧게, 구체적으로, 때로 실패 그림과 함께 |

**우리에게 중요한 결론**: 우리 논문은 Visual-RRT·AutoMoMa처럼 **기존 부품(VLA, ESDF, SQP)의 결합**이다.
Highlight 논문들은 결합을 숨기지 않고, **결합이 왜 어려운지**와 **각 부품이 결합 때문에 어떻게 바뀌어야 했는지**를 기여로 만든다.

---

## 3. AG3S VLA 원고에 적용

### 3.1 Introduction을 Visual-RRT 흐름으로 다시 짜기

| 단계 | 우리 문장(초안 방향) |
|---|---|
| 분야 확립 | VLA는 범용 조작의 표준이 되었다 … 정책은 충돌을 계산하지 않고, 성공해도 안전하지 않다(현재 1문단 유지) |
| 숨은 가정 | "Distance-field safety layers **assume that the obstacles are known**: every observed surface is something to avoid." |
| 현실과의 대비 | "In manipulation, however, the robot **must** touch one of those surfaces, and which one changes as the task proceeds." |
| 결합의 어려움 | "Placing a distance field behind a VLA is therefore **non-trivial**: (i) the field cannot tell the target from the obstacles, (ii) the right treatment of the target depends on the sub-task, and (iii) removing the target from the field hides it from every link." |
| 기존 해법의 비용 | Limitation I/II (현재 유지) — "Existing layers resolve (i) and (ii) by adding models around the policy …" |
| 핵심 아이디어 | "**Our key idea is that the policy already answers both questions**: its attention indicates which object it is about to manipulate, and its hidden state encodes the stage of the task." |
| 모듈 | 각 모듈을 (i)–(iii)의 어려움에 1:1로 대응시킨다: ATS→(i), HSR→(ii), LTE→(iii), CCR→실행 |
| 예시 walkthrough | BiPreManip 방식: "For instance, when the policy reaches for an apple next to a pear (Fig. 1), ATS segments the apple from the attention, LTE lets only the fingers touch it while the forearm keeps clear, and once HSR reads *place*, the apple travels as part of the hand …" |
| 기여 | 3개 유지, 마지막 문장: "Together, these components **bridge** VLA policies and geometric motion generation …" (AutoMoMa 형식) |

### 3.2 각 모듈 절을 "naive → 문제 → 해법" 구조로

| 모듈 | naive 방법 | 그 결과 | 우리 해법 |
|---|---|---|---|
| ATS | attention 최대값 픽셀을 target으로 | 물체의 일부만 덮이고 프레임마다 흔들림 | attention은 seed만, 물체 범위는 depth 클러스터, 3프레임 확인 |
| ATS | attention이 낮은 점을 장애물에서 제거 | 보이지 않는 장애물이 사라짐 | attention은 어떤 점도 지우지 않는다 |
| HSR | 생성 텍스트로 sub-task 읽기 | action-only fine-tuning 정책은 텍스트를 내지 못함 | KV cache의 linear read-out |
| LTE | target을 field에서 삭제 | 모든 링크가 target을 못 봄 | 손바닥·손가락만 target-free ESDF를 조회 |
| LTE | 잡은 물체를 장애물로 유지 | 손이 자기 물체와 충돌 | attached object로 편입, 놓으면 해제 |
| CCR | 한 명령씩 보정 | chunk 나머지가 미검사 | 실행 구간 전체를 SQP로, 모든 행 재검사 |

문장 틀: "While X …, naively Y would Z. To address this, …. Importantly, this preserves …".

### 3.3 Experiments를 메커니즘 중심으로

- **두 개의 상보적 평가**(Visual-RRT): (1) 공개 벤치마크에서의 충돌·성공, (2) target·sub-task 처리를 *분리*해서 보는 진단
  (ATS의 target 일치율, HSR의 label 정확도와 gate on/off의 놓은 뒤 처리).
- **baseline이 실패하는 이유를 쓴다**:
  - field without exemption → "stops at the grasp, because the target repels the fingers"
  - global exclusion → "the forearm and the attached object pass through the target region"
  - VLA only → 충돌 장면
- **ablation은 GeoPredict처럼 누적형**(우리 V-chain이 이미 그 형태): 마지막 증가분을 "validates our hypothesis that …"로 연결.
- **정성 그림 한 장이 주장을 보여준다**: attention heatmap → 3D cluster → target-free ESDF slice → 정제 전/후 궤적, 개선 부위 강조.

### 3.4 Limitation은 짧고 구체적으로, 실패 사례와 함께

Visual-RRT·AutoMoMa처럼 "Limitations and Conclusion"을 합치고, 3–4문장 + 실패 그림 1개(예: attention이 장애물 조각에 걸린 프레임).
GeoPredict처럼 완화 요인을 함께 쓴다(예: "gripper-link coverage is the evaluated configuration; the same mechanism applies to any query sphere").

### 3.5 반드시 인용해야 할 새 논문 — AutoMoMa

AutoMoMa(4.2 Collision processing)는 **단계별로 물체의 충돌 처리를 바꾼다**: 접근 단계에서는 물체와 겹치는 환경 voxel을 지우고
고해상도 mesh로 바꾸며, 조작 단계에서는 물체를 로봇 kinematic chain의 링크로 편입한다.
이것은 우리 LTE·attached object와 **개념이 겹친다**. 다만 AutoMoMa는 *알려진 장면 기하*로 오프라인 데이터를 생성하며,
perception·VLA·실시간 정제가 없다. Related Work 2문단(collision avoidance / trajectory optimisation)에 넣고 차이를 한 문장으로 적는다.
`references.bib`에 추가가 필요하다(서지는 PDF에서 확인 후).

---

## 4. 문장 표현 모음 (형식만, 내용은 우리 것으로)

### 문제·가정
- "Despite their advances, existing … assume that …"
- "In contrast, … provide only …, where … is not explicitly known."
- "The core bottleneck is …" / "… remains an open challenge."
- "While such behaviors appear effortless for humans, they pose significant challenges for robots, requiring …"
- "Collectively, these constraints have forced … to compromise on …"

### 결합의 어려움·핵심 아이디어
- "To bridge this gap, we extend … to … However, this extension is non-trivial since …"
- "Our key idea is to … while maintaining …"
- "We argue that two … are particularly valuable. First, … Second, … Crucially, …"

### 방법
- "While X enables …, naively applying it to … leads to …. To address this, we introduce …"
- "Our key insight is that … serves as an implicit …"
- "Importantly, this … preserves …" / "This allows … without …"
- "Concretely, …" / "Specifically, …"

### 실험
- "This gap arises from …: …"
- "For X, although …, its … As a result, …"
- "Starting from the baseline, adding … improves … Critically, activating … achieves …"
- "This final gain validates our hypothesis: …"
- "Interestingly, … shows slight degradation, suggesting that …"

### 결론·한계
- "We bridge … and …, offering a direction for …"
- "While … presents …, … mitigates these concerns."
- "Limitation. While … effectively handles …, … remains challenging."

**표시 없이 쓰지 않을 것**: 출처·자료 유형·검증 상태가 없는 수치, 검정 없는 "significantly", 범위 없는 "the first".

---

## 5. 제안하는 다음 수정

1. Introduction을 3.1 표의 흐름으로 재작성(숨은 가정 → non-trivial 결합 → key idea → 모듈↔어려움 1:1 → 예시 walkthrough).
2. Method 각 절 첫 문단을 3.2 표의 "naive → 문제 → 해법" 구조로 재작성.
3. Related Work에 AutoMoMa 추가(서지 확인 후).
4. Experiments 결과 자리(TBD)에 "baseline 실패 이유" 문장 틀을 미리 배치.
5. Conclusion을 "Limitations and Conclusion" 합본으로, 실패 그림 계획 추가.
6. 제목 후보 검토(현재 "AG3S VLA: Target-Aware Distance Fields from VLA Attention for Refining Action Chunks"). Highlight 제목은
   "무엇을 / 무엇으로" 형식이 짧다(예: "Finding Paths toward Visual-Goals via Differentiable Rendering"). 사용자 판단.
