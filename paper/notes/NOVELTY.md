# Novelty — Candidates and Verification Tasks

Updated: 2026-10-06 KST

현재 모든 novelty 항목은 **`UNVERIFIED`**다. 내부 구현의 존재나 여러 module의 연결만으로
선행연구 대비 novelty가 성립한다고 쓰지 않는다.

선행연구의 최종 선정은 사용자 후보 목록을 기다린다. R1의 overlap 기록은 기존 자료의
탐색적 관찰이며 최종 비교 대상이나 novelty 판정 기준을 확정한 것이 아니다.

| 후보 | 구별하려는 내용 | 필요한 비교·증거 |
|---|---|---|
| N001 | VLA internal task relevance를 explicit multi-view 3D representation으로 연결 | Representation·attention·3D grounding 선행연구와 동일 조건 비교 |
| N002 | Manipulation phase에 따른 target-aware contact 처리 | Target/payload/obstacle 구분 선행연구, GT와 ablation |
| N003 | Task representation과 constraint-based action refinement의 연결 | 기존 safety shield·trajectory refinement·geometry method 비교 |
| N004 | Task와 safety를 함께 평가하는 protocol과 failure analysis | Dataset/split·독립 contact GT·metric coverage·reproducibility 확인 |

"first", "novel", "unprecedented", "guaranteed safety" 등의 표현은 검토 전 사용하지 않는다.
Novelty 검토에는 각 reference의 실제 내용, earliest version과 peer-reviewed version,
concept overlap·difference·experiment relevance·code availability를 기록한다.

[RELATED_WORK_MAP](RELATED_WORK_MAP.md)의 자료 검토와 P001–P005의 실험을 연결한다.
원문의 main message는 연구 방향으로 유지하며 입증 여부는 [CLAIMS](../evidence/CLAIMS.md)를 따른다.

## R1의 overlap과 후보 축소

근거: [REFERENCE_ANALYSIS_R1](REFERENCE_ANALYSIS_R1.md). 후보 상태는 모두 `UNVERIFIED`다.

| 후보 | 확인한 overlap | 앞으로 검증할 구체적인 차이 |
|---|---|---|
| N001 | LoHoVLA task-conditioned sub-task/action 생성; V2 depth fusion·task-specific ESDF resolution과 multi-camera 후속 제안 | 내부 relevance의 실제 의미·causal ablation·multi-view 3D 연결과 최근 grounding 방법의 차이 |
| N002 | V2 robot segmentation은 이미 구현됨. Task-target contact 허용과는 다른 규칙 | Phase별 target/payload/obstacle 규칙, 제한 조건, independent contact GT와 mask 오류의 영향 |
| N003 | cuRobo/nvblox ESDF motion generation과 V2 live-depth MPC; V2 p.38 learned-policy safety layer 제안 | Geometry-only refinement 대비 relevance/contact conditioning의 효과와 execution budget·fallback |
| N004 | V2 geometry feasibility와 downstream policy tracking/variance 평가 | 동일 task에서 안전·task outcome·coverage를 독립 측정하는 protocol의 차이와 재현 가능성 |

ESDF 구성·GPU planning·robot sphere distance·VLA와 최적화의 단순 연결을 독립 novelty로 쓰지 않는다.
논문의 future work는 구현 완료와 구분하되 priority/overlap 검토에 포함한다.
이 3편에서 확인하지 못한 절차를 전체 prior art의 부재로 해석하지 않는다.
Safety shield·attention/3D grounding·task-aware contact의 더 가까운 문헌과 구현 audit가 남아 있다.
