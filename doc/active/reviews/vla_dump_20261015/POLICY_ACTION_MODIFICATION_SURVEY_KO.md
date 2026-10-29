# 학습된 정책의 action · 궤적 · 자세를 추론 시점에 바꾸는 방법 — 문헌 조사

작성: 2026-10-21 KST.

- 사용자 지시: "정책 기반으로 action을 수행하는 로봇의 궤적이나 자세를 변경하는 논문들이 있었는지 확인해봐"
- 조사: literature-analyst sub-agent. 웹 조사만 했고, 우리 저장소 파일은 읽지 않았다
- 확인 범위
  - arXiv abs/HTML 페이지, 학회 proceedings, project page
  - 검색 결과는 2026-10(arXiv 2610.xxxxx)까지 나왔다
  - 2026년 논문 대부분은 PDF 본문을 읽지 않았다
- 수치는 각 논문이 보고한 값이다. 우리가 재현하지 않았다. 확인하지 못한 항목은 3절에 모았다

## 결론

1. **Park et al. 2026 (arXiv 2606.09749, "Your Model Already Knows")가 우리 핵심 주장과 가장 많이 겹친다.**
   - 겹치는 점
     - π0.5 attention head(L12 H3)로 target을 고른다
     - 고른 target을 obstacle에서 빼고 training-free CBF-QP filter를 돌린다
   - 다른 점(abs·HTML 기준)
     - 보호 범위가 end-effector ellipsoid 하나다. 우리는 cuRobo ESDF를 쓴다
     - step마다 Cartesian delta를 보정한다. 우리는 50-step joint chunk를 SQP TrajOpt로 보정한다
     - single-arm, sim only다
     - grasp 중 접촉이나 잡을 지점 밀림(drift)은 다루지 않는다
   - 그래서 우리 novelty는 "attention으로 target을 구분한다"가 아니라 그 다음 단계에서 찾아야 한다
2. **OmniGuide (arXiv 2603.10052)는 우리 구조에 대한 직접적인 반론이다.** VLA chunk를 cuRobo로 post-hoc 보정하는 baseline보다 denoising 중 guidance가 낫다고 주장한다(수치는 미확인). Multi-Link Safety Filtering (2609.40007)도 "frozen policy는 chunk 안의 보정을 보지 못한다"고 쓴다. 우리가 본 잡을 지점 밀림과 같은 현상이다.
3. **(a) 손목 관절 가중치(IFG-TOW)에 직접 대응하는 선행연구는 찾지 못했다.** 부분적으로만 겹친다.
   - AEGIS(2512.11891)의 rotation 포함 설정. 오히려 회피율이 translation-only 설정보다 낮았다(68.9 vs 77.9%)
   - OmniGuide의 orientation 항
   - 고전적 null-space 학습(Armesto et al.)

   검색 범위에 한계가 있으므로 "최초"라고 주장하지 않는다.
4. **(b) 이웃 물체와의 제한된 접촉(IFG-NBC)에 가장 가까운 연구**
   - IMPACT(ICRA 2026, project page 기준): VLM이 물체마다 접촉 허용 정도를 매긴다. learned policy는 없고 planner가 end-effector로 민다
   - Jain et al. IJRR 2013: 낮은 force의 접촉은 무해하다고 보고 force를 제한한다
   - 고전적 push-grasping(Dogar & Srinivasa)

## 1. 방법 표

TF는 training-free(정책 weight 고정, 추가 학습 없음), SLB는 SafeLIBERO다.

| # | 방법 (venue, year) | 무엇을 바꾸나 | 어떻게 | TF | 정책 | 평가 · 수치 |
|---|---|---|---|---|---|---|
| 1 | AEGIS / VLSA (arXiv 2512.11891, 2025) | EE action | VLM이 고른 obstacle 1개와 EE를 ellipsoid로 근사한다. VLA action과의 편차를 최소화하는 CBF-QP를 푼다. translation-only와 rotation 포함 설정 | Yes | π0.5 | SLB 32 scenario. translational CAR 18.7→77.9%, TSR 50.9→68.1%. full CAR 17.1→68.9%, TSR 57.8→67.5% |
| 2 | Your Model Already Knows (Park et al., arXiv 2606.09749, 2026, under review) | EE Cartesian delta | attention head mass로 target을 고른다. 1위와 2위의 gap이 δ보다 클 때만 target을 obstacle에서 뺀다. 이후 separating-hyperplane CBF-QP를 풀고, infeasible이면 정지한다 | Yes | π0.5 | sim only. Object-III SSR 40.0→70.5%, Goal-III CR 90.5→30.5%. Spatial-I에서는 CR 26.0→32.5%로 나빠짐. 1 step 약 49 ms |
| 3 | Neuro-Symbolic Safety Guidance via Constrained Flow Matching (arXiv 2607.01378, 2026) | denoising 중 chunk | 중간 예측 trajectory를 minimum-norm 제약 최적화로 보정 | 미확인 | flow VLA | SLB collision avoidance 82.8%(+6.3), TSR 81.6%(+19.8) |
| 4 | VLPSA (arXiv 2609.22462, 2026) | 명령(whole-body와 쥔 물체 포함) | 2-resolution Poisson Safety Function으로 만든 CBF-QP | Yes | π0.5 | SLB collision avoidance 23.1→91.2%. real FR3 |
| 5 | Multi-Link Safety Filtering for VLA (arXiv 2609.40007, 2026) | 명령 | gripper·wrist·forearm에 ellipsoid 5개를 두고 barrier QP 적용 | Yes | π0.5 | sim collision 65.62→27.27%, SSR 29.35→50.43%. SO-101 실물 접촉 3/16(filter 없음 11/16) |
| 6 | OmniGuide (arXiv 2603.10052, 2026) | denoising 중 chunk | FK로 계산한 Cartesian energy의 gradient를 velocity field에 더한다. collision repeller는 −log SDF이고 SDF에서 task 영역을 제거한다. orientation attractor 포함 | Yes | π0.5, GR00T N1.6 | RoboCasa success 24.2→92.4%, collision avoidance 7.0→93.5%. cuRobo post-hoc 대비 우월 주장 |
| 7 | VLS (arXiv 2602.03973, 2026) | denoising | VLM이 만든 differentiable reward로 guidance | Yes | diffusion/flow | CALVIN +31%, LIBERO-PRO +13% |
| 8 | SafeBimanual (CoRL 2025) | bimanual trajectory | denoising 중 cost guidance. VLM이 keypoint cost를 schedule | Yes | diffusion | RoboTwin SR +11.1(CoRL판), real SR +32.5% |
| 9 | DynaGuide (NeurIPS 2025) | denoising | 외부 dynamics model로 guidance | Yes | diffusion | CALVIN steering 약 70% |
| 10 | ITPS (ICRA 2025) | sampling | human input으로 sampling을 bias | Yes | diffusion | sim · real |
| 11 | Real-Time Chunking (NeurIPS 2025) | 다음 chunk | 실행이 확정된 prefix를 inpainting으로 고정 | Yes | diffusion/flow VLA | Kinetix, real bimanual |
| 12 | SafeDiffuser (ICLR 2025) | diffusion plan | CBF invariance를 denoising에 삽입 | 미확인 | Diffuser류 | maze · locomotion · manipulation |
| 13 | SafeFlowMatcher (ICLR 2026) | 실행 path | prediction 뒤에 CBF-QP 보정 | 미확인 | flow planner | |
| 14 | DPCC (L4DC 2025) | denoising | projection과 constraint tightening | Yes | trajectory diffusion | sim manipulator |
| 15 | Motion Planning Diffusion (IROS 2023 / T-RO 2025) | trajectory | collision · smoothness cost guidance | — | diffusion planner | real arm |
| 16 | V-GPS (CoRL 2024) | action 선택 | Q-value로 후보를 re-rank | 정책 고정, value 학습 | Octo, OpenVLA 등 | 12 task |
| 17 | FM-Steer (CVPR 2026) | action 선택 · 정제 | value-guided cascaded denoising | 정책 재학습 없음 | flow VLA | sim · real |
| 18 | Policy Decorator (ICLR 2025) | action | residual RL | No | BeT, DP | ManiSkill, Adroit |
| 19 | IMPACT (ICRA 2026, project page 기준) | 계획 경로 | VLM이 물체별 contact tolerance를 매기고 contact-aware A*로 계획 | Yes | VLM과 planner | sim 3200회 이상, real 약 200회 |
| 20 | Push-grasping (IROS 2010) / Grasp Planning Through Clutter (RSS 2012) | grasp motion | 주변 물체를 의도적으로 밀며 잡음 | 고전 planner | — | HERB |
| 21 | Reaching in Clutter with Whole-arm Tactile Sensing (IJRR 2013) | 접촉 force | 낮은 force 접촉을 허용하고 force를 제한 | — | — | real · sim |
| 22 | ReKep (CoRL 2024) / OK-Robot (RSS 2024) / SkipVLA (arXiv 2609.20648) | EE pose · 단계 분할 | VLM keypoint 제약 / VLM과 grasp primitive / free-space는 planner, 접촉 구간은 VLA | 대체로 Yes | VLM, VLA | SkipVLA: LIBERO 최대 2.5배 빠름, SR 동일 |

## 2. 우리 작업과의 관계

| 우리 쪽 | 가장 가까운 선행연구 | 시사점 |
|---|---|---|
| attention으로 target을 고르고 obstacle에서 뺌 | Park et al. 2606.09749 | 반드시 인용 · 비교. 차별점은 ESDF, joint chunk TrajOpt, bimanual, HSR, grasp 중 접촉 처리 |
| post-hoc TrajOpt로 VLA chunk 보정 | OmniGuide, Multi-Link Safety Filtering | post-hoc 보정의 약점(정책이 보정을 보지 못함)이 이미 지적됨. denoising 내부 보정(B · C 방식, `AG3S/docs/AG3S_PHASE_DENOISING_PLAN.md` §4)과의 비교가 필요 |
| (a) 손목 관절 가중치 (IFG-TOW) | 직접 대응 없음. AEGIS rotation 설정, OmniGuide orientation 항 | AEGIS에서는 rotation을 더하자 회피율이 내려갔다. ablation으로 제시 |
| (b) 이웃 물체와의 제한된 손가락 접촉 (IFG-NBC) | IMPACT, Jain et al. 2013, push-grasping | "VLA attention으로 판단한 물체별 제한 접촉"으로 범위를 좁혀 주장 |

## 3. 확인하지 못한 항목 (UNVERIFIED)

- 2026 arXiv 논문들(#2–7, SkipVLA)의 peer-review 상태. #2는 "under review"로만 표기되어 있다
- OmniGuide의 cuRobo baseline 수치와 real 실험 수치
- #3의 training-free 여부. 수치가 percentage point인지 relative인지
- VLPSA의 TSR과 baseline 이름
- RTC real 실험에 쓴 VLA 종류
- SafeDiffuser · SafeFlowMatcher의 학습 필요 여부
- SafeFlow(2504.08661)의 venue
- FM-Steer의 저자 · 수치 · verifier 학습 여부
- IMPACT의 venue(project page에만 근거)
- AEGIS가 target을 obstacle 후보에서 빼는지 여부

## 출처

- AEGIS https://arxiv.org/abs/2512.11891
- Park et al. https://arxiv.org/html/2606.09749
- Neuro-Symbolic Safety Guidance https://arxiv.org/abs/2607.01378
- VLPSA https://arxiv.org/abs/2609.22462
- Multi-Link Safety Filtering https://arxiv.org/abs/2609.40007
- OmniGuide https://arxiv.org/html/2603.10052
- VLS https://arxiv.org/abs/2602.03973
- SafeBimanual https://proceedings.mlr.press/v305/deng25c.html
- DynaGuide https://arxiv.org/abs/2506.13922
- ITPS https://arxiv.org/abs/2411.16627
- RTC https://arxiv.org/abs/2506.07339
- SafeDiffuser https://proceedings.iclr.cc/paper_files/paper/2025/hash/f95606d8e870020085990d9650b4f2a1-Abstract-Conference.html
- SafeFlowMatcher https://arxiv.org/abs/2509.24243
- DPCC https://proceedings.mlr.press/v283/romer25a.html
- Motion Planning Diffusion https://arxiv.org/abs/2308.01557
- V-GPS https://proceedings.mlr.press/v270/nakamoto25a.html
- FM-Steer https://cvpr.thecvf.com/virtual/2026/poster/40101
- Policy Decorator https://proceedings.iclr.cc/paper_files/paper/2025/hash/45c361d4117d598d4bb6568b407e9ac9-Abstract-Conference.html
- IMPACT https://impact-planning.github.io/
- Grasp Planning Through Clutter https://www.roboticsproceedings.org/rss08/p08.pdf
- Jain et al. https://arxiv.org/abs/1304.6146
- ReKep https://proceedings.mlr.press/v270/huang25g.html
- SkipVLA https://arxiv.org/abs/2609.20648
