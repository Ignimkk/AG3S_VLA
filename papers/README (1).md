# AG3S 비교 실험: 원문 요청 목록

PDF는 이 폴더에 넣는다. 파일명은 자유이며 논문 제목을 알아볼 수 있으면 된다. 가능하면 본문과 supplementary/appendix를 함께 넣고, 버전이 있으면 파일명에 표시한다. 아래 목록은 2026-10-30 실험 계획의 비교군이며, 직접 실행 가능 여부나 원 논문 수치의 우리 조건과의 호환성은 원문을 읽은 뒤 확인한다.

## 1. Closed-loop 안전 비교 — 우선 7편

| 방법 | 논문 | 원문 | 확인 목적 |
|---|---|---|---|
| KNOWS | Your Model Already Knows: Attention-Guided Safety Filter for Vision-Language-Action Models | https://arxiv.org/abs/2606.09749 | attention 기반 target 선택, 성공·충돌·안전 성공, 구현 충실도 |
| VLSA / AEGIS | VLSA: Vision-Language-Action Models with Plug-and-Play Safety Constraint Layer | https://arxiv.org/abs/2512.11891 | SafeLIBERO 조건과 공식 지표, AEGIS 및 원 논문 결과 |
| Multi-link safety filter | Multi-Link Safety Filtering for VLA Policies Around Moving Hazards | https://arxiv.org/abs/2609.40007 | 다중 link 보호 범위, 충돌 판정, 부분 재구현 충실도 |
| OmniGuide | OmniGuide: Universal Guidance Fields for Enhancing Generalist Robot Policies | https://arxiv.org/abs/2603.10052 | guidance 방식, 원 평가 조건, EE-delta 이식 차이 |
| CFM | Neuro-Symbolic Safety Guidance for Vision-Language-Action Models via Constrained Flow Matching | https://arxiv.org/abs/2607.01378 | SafeLIBERO·GT 입력·제약 및 재구현 충실도 |
| NUDGE | Zero-Shot Reactive Obstacle Avoidance for Generative Robot Policies | https://arxiv.org/abs/2609.35231 | 반응형 회피와 sampling guidance, 재구현의 가정 |
| Lan-o3dp | Language-Guided Object-Centric Diffusion Policy for Generalizable and Collision-Aware Manipulation | https://github.com/qqianfeng/lan-o3dp | 본래 policy와 guidance-only 이식의 차이, 실험 조건 |

Lan-o3dp는 [arXiv 버전](https://arxiv.org/abs/2407.00451)의 제목이 출판 논문 제목과 다르게 표시될 수 있으므로 버전을 확인한다. π0.5 단독은 별도의 안전 방법이 아니라 공통 기본 policy다.

## 2. Grounding·perception 비교 — 5편

| 방법 | 논문 | 원문 | 확인 목적 |
|---|---|---|---|
| Grounding DINO | Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection | https://arxiv.org/abs/2303.05499 | Grounded-SAM 계열의 text 기반 detector |
| SAM 2 | SAM 2: Segment Anything in Images and Videos | https://arxiv.org/abs/2408.00714 | per-frame segmentation과 causal tracking 비교 |
| YOLOE | YOLOE: Real-Time Seeing Anything | https://arxiv.org/abs/2503.07465 | 외부 open-vocabulary segmentation 비교 |
| Localization heads | Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding | https://arxiv.org/abs/2503.06287 | attention head 기반 target localization |
| TokenCut | Self-Supervised Transformers for Unsupervised Object Discovery Using Normalized Cut | https://arxiv.org/abs/2202.11539 | feature graph 기반 object discovery |

Grounded-SAM-2는 이 계획에서 Grounding DINO와 SAM 2를 조합한 provider다. 필요 시 SAM v1 비교용으로 [Segment Anything](https://arxiv.org/abs/2304.02643)도 추가한다. GT oracle 및 동일 시스템에서 만든 F2/F5/F6은 별도 논문이 아니라 구성 비교다.

## 3. 공통 policy — 참고 1편

[π0.5: a Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054): frozen π0.5 policy와 checkpoint, 원 평가 조건 확인.

## 수령 후 확인할 항목

각 논문의 방법·코드·checkpoint, 평가 suite 및 level, 초기 상태와 표본 수, action/controller, privileged 입력, 보호 link, 성공·충돌·안전 성공 정의, 시간 측정 경계, 원문 수치의 페이지/표/버전을 기록한다. 원 논문 수치와 우리 재실행 수치는 구분한다. 원문 조건이 다르면 수치의 직접 우열을 주장하지 않는다. WBAG는 기존 사용자 결정에 따라 제외한다.
