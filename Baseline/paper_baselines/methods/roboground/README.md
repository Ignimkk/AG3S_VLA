# roboground — RoboGround [11] (fetch + README 만)

| 항목 | 내용 |
|---|---|
| 원 논문 | Huang et al., *RoboGround: Robotic Manipulation with Grounded Vision-Language Priors* (CVPR 2025), arXiv:2504.21530 |
| 출처 repo | `ZzZZCHS/RoboGround@25a862634398bc0407bc7b36b5d8b7c61d7a502b`. 하위 학습 repo: `ZzZZCHS/groundingLMM@3958ffdb…` (GLaMM LoRA), `ZzZZCHS/GR1@16b63e57…` (policy). 하위 repo 는 fetch 하지 않았다 |
| clone 위치 | `Baseline/paper_baselines/external/RoboGround` (untracked. `external/fetch_sources.py --only RoboGround`) |
| license | 상위 repo **MIT** (LICENSE 확인). GR1 과 groundingLMM 은 GitHub license 가 표기되지 않았다 |
| 상태 | **실행하지 않는다.** fetch 와 이 README 까지만 했다 |

## 방법과 입력

- 2단계 구조다.
  - (1) GLaMM 기반 grounded VLM 이 instruction 과 RGB 에서 **target 물체 mask** 와 **placement 영역 mask** 를 낸다.
  - (2) GR-1 policy 가 RGB 와 이 mask 를 **입력으로 받아** action 을 낸다.
- 학습 data: RoboCasa demo 에 distractor 를 넣고 GPT 로 instruction 을 새로 만든 자체 dataset 이다 (HF `ZzZZCHS/RoboGround_Data`, 512×512 obs + GT mask).
- 평가: RoboCasa 기반 자체 benchmark 다 (pick-and-place, Easy / Appearance / Spatial / Common-sense). LIBERO 와 SafeLIBERO 결과는 없다.

## checkpoint

**없다.** README 와 project page 어디에도 weight 링크가 없다. data 와 학습 code 만 공개됐다.

## 우리 protocol 에 넣으려면

1. grounding 만 떼어서 쓰는 길 (현실적):
   - GLaMM grounding 을 LIBERO frame 에서 target mask provider 로 쓴다 (A 의 `TargetMaskProvider` interface).
   - weight 가 없으므로 `groundingLMM` 으로 LoRA 를 RoboGround data 에서 **학습** 해야 한다.
   - GPU 학습 비용이 크다.
2. policy 째로 쓰는 길:
   - GR-1 을 LIBERO 에서 mask 입력으로 다시 학습해야 한다.
   - frozen π0.5 를 바꾸지 않는 우리 setting 과 구조가 다르다 (grounding 이 policy 입력이다).
3. 어느 쪽이든 robosuite, robocasa, robomimic fork (repo 안의 사본) 와 환경을 따로 만들어야 한다. 이 checkout 의 venv 에는 넣지 않는다.
