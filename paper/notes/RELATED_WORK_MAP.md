# Related Work Map — Verified Sources and Reading Queue

Updated: 2026-10-06 KST

## 선정 상태 — 사용자 후보 목록 대기

선행연구 논문은 아직 최종 선정하지 않았다. 사용자가 추후 간추린 목록을 제공한다.
아래 3편은 기존 로컬 자료를 탐색적으로 분석한 참고 후보이며 최종 선정 목록이 아니다.
서지·내용의 검증 상태와 원고 포함 여부는 별개다. 현재 bibliography와 Related Work의
인용은 임시 초안이며, 사용자 목록을 받은 뒤 비교 대상·분석 우선순위·최종 인용 범위를 정리한다.
그 전에는 문헌 후보를 자동으로 확대하거나 최종 baseline·novelty 기준으로 확정하지 않는다.

R1에서 기존 PDF 3편의 핵심 method/evaluation/limitations와 primary 서지를 확인했다.
검토 범위·페이지·차이는 [REFERENCE_ANALYSIS_R1](REFERENCE_ANALYSIS_R1.md),
file identity와 version은 [REFERENCE_PROVENANCE_R1](REFERENCE_PROVENANCE_R1.json)을 따른다.
`VERIFIED_IN_SCOPE`는 아래 내용 확인이며 전체 literature/novelty 검증 완료가 아니다.

| ID / citation key | Local PDF | 공개본과 인용 범위 | 상태 |
|---|---|---|---|
| RW001 / `yang2025lohovla` | [LoHoVLA.pdf](../../papers/LoHoVLA.pdf) | [2506.00411v1](https://arxiv.org/abs/2506.00411v1), 2025 preprint; local stamp 일치 | `VERIFIED_IN_SCOPE` |
| RW002 / `sundaralingam2026curobov2` | [cuRoboV2 PDF](../../papers/cuRoboV2_%20Dynamics-Aware%20Motion%20Generation%20with%20Depth-Fused%20Distance%20Fields%20for%20High-DoF%20Robots.pdf) | [2603.05493v2](https://arxiv.org/abs/2603.05493v2), 2026 technical report; local stamp 일치 | `VERIFIED_IN_SCOPE` |
| RW003 / `sundaralingam2023curobo` | [curobo_report_v1.pdf](../../papers/curobo_report_v1.pdf) | [2310.17274](https://arxiv.org/abs/2310.17274), 2023 report; local version unstamped/byte match 미확인 | `VERIFIED_IN_SCOPE`, local version `UNKNOWN` |

[references.bib](../latex/references.bib) 3개와 [Related Work](../latex/sections/02_related_work.tex)의
실제 citation을 연결했다. Minimum-jerk report를 ICRA 초기 논문과 혼동하지 않는다.
확인하지 않은 conference acceptance를 기재하지 않는다.

## 비교 축과 다음 조사

| 분야 | R1의 확인 내용 | 추가 검증 |
|---|---|---|
| VLA policy / action | LoHoVLA sub-task+discrete pick/place action, reward 기반 replanning | 우리 16-D chunk와 같은 조건의 action/refinement 선행연구 |
| Attention / 3D grounding | 이 3편만으로 N001의 priority를 판단할 수 없음 | Attention causal evidence, task-relevance backprojection, multi-view grounding |
| Distance field | cuRobo/nvblox ESDF; V2 depth fusion·dense ESDF·task-specific resolution | Unknown-space·thin/occluded geometry·GT coverage·우리 source version |
| Contact / safety refinement | V2 MPC/live depth와 learned-policy safety layer 후속 제안 | Phase-specific allowed contact·payload geometry·safety shield·fallback |
| Evaluation | Task/planning/recall/retargeting metrics 구분 | 동일 split·GT·timing budget·task–safety trade-off protocol |

ACoT-VLA, ActiveVLA, AVA-VLA, OptimusVLA, SaPaVe의 PDF는
`paper/references/cvpr2026_vla/`에서 확인하지 못했다. 원문이 제시한 미선정 예시이며,
상태는 `UNSELECTED_EXAMPLES`다. 필수 분석 목록으로 확정하지 않는다.
명칭·출판 정보·결과는 확인 전 citation에 넣지 않는다.

2027 contemporaneous-work cutoff는 [SUBMISSION_POLICY](../guidelines/CVPR_SUBMISSION_POLICY.md)를 따른다.
위 3개 공개 record의 최초 공개일은 cutoff 전이다. 영향을 준 work는 공개 날짜와 무관하게
cite/discuss하고, 최신 출판본·더 가까운 prior art는 후속 조사에서 다시 확인한다.
