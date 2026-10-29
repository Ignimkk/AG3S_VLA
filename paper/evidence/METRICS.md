# Metrics and Statistical Protocol

Updated: 2026-10-05 19:45 KST

## Metric registry

| ID | 정의 | 현재 상태 |
|---|---|---|
| M001 | 기존 T39 task success: planning-row apple lift 판정과 마지막 planning-row crate 포함 판정의 conjunction | R001·R002에 등록 |
| M001_PAIRED | 동일 `(episode, policy_seed)`의 두 task outcomes에 대한 paired 2×2 | R003에 등록 |
| M002 | Trial에 금지 contact가 한 번이라도 있었는지; allowed-contact policy와 관측 범위를 명시 | `PLANNED`; 실제 값 없음 |
| M003 | Safe success: 같은 trial에서 task success이고 금지 contact가 없으며 요구 safety coverage가 완전한 경우 | `PLANNED`; marginal rate를 곱해 만들지 않음 |
| M004 | 실행 궤적과 독립적인 참 geometry의 최소 signed clearance, 단위 m/mm와 대상 link·payload 범위 명시 | `PLANNED`; ESDF query와 구분 |
| M005 | Grounding target identity·3D error·phase별 agreement, obstacle retention | `PLANNED`; task target GT·camera frame·visibility 필요 |
| M006 | 조건별 task/safety와 train/test 분리 provenance | `PLANNED`; unseen label 미확정 |
| M007 | 동일 측정 경계의 latency·deadline miss·memory; cold/warm 및 자원 간섭 구분 | `PLANNED`; runtime 결과 미등록 |
| M008 | HOLD/abort/timeout 빈도·원인과 task outcome | `PLANNED`; collision 감소의 proxy로 쓰지 않음 |
| M009 | Arm과 gripper의 의미·unit을 구분한 action modification | `PLANNED`; 16개 channel의 raw 차이를 단위 확인 없이 합치지 않음 |

## M001의 source 정의

[기존 analysis의 definitions](../../outputs/verify/T39/s23/analysis.json)에 따르면 grasp는
planning-row apple 높이가 초기보다 50 mm 이상 올라간 경우다. Place는 마지막 planning row의
apple 중심이 crate local bounds에 들어가는 경우이며 success는 둘의 conjunction이다.
이 정의는 stable placement, 전체 시뮬레이션 contact 또는 하드웨어 성공의 확인이 아니다.
구체적인 bounds와 좌표 변환은 해당 source를 따른다.

새 task-success 규칙을 도입하면 새로운 metric/version으로 기록한다. 기존 결과의 정의를
바꾼 뒤 과거 값과 그대로 비교하지 않는다.

## Safety·coverage 규약

Target 접촉도 phase·허용 link·접촉 종류를 정의한다. Target을 obstacle field에서 지우는 것과
허용되지 않은 target/payload contact까지 평가에서 지우는 것은 다르다.
Robot link, self-collision, held-object geometry, obstacle, support surface, 관측되지 않은 영역,
실행 sim sub-step의 coverage를 명시한다. Planning/control row 재구성만 있으면 실제
sub-step contact를 완전히 판정한 것으로 쓰지 않는다.

Missing contact/clearance는 `UNKNOWN`이고 0이 아니다. Truncated trial이나 stop을
자동 safe success로 세지 않는다. Signed distance의 clamp/false-zero와 target mask에 의해
왜곡된 지표를 독립 ground truth 없이 채점하지 않는다.

## Statistics — 사전 고정할 작업 계획

- Trial unit은 episode·policy seed·condition이다. 같은 episode의 seed와 frame은 계층적으로 관리한다.
- Binary 비교에는 matched pair를 유지한다. 연속 metric도 paired change와 episode별 분포를 남긴다.
- Confidence interval, cluster 단위, random seed, exclusion 규칙, multiple comparison 정책을 고정한다.
- "성능 유지"를 검증하려면 허용 가능한 non-inferiority margin과 판정 방법을 실행 전에 정한다.
  현재 margin·sample-size 결정은 `TBD`다. 유의하지 않은 차이를 equivalence로 바꾸지 않는다.
- Aggregation한 값은 source와 계산 script·입력 hash·unit·denominator를 저장하고 별도 result로 등록한다.
- p-value, CI, 평균·percent 등을 이 Step에서 새 성능 결과로 계산하지 않았다.

독립 safety 판정과 protocol 고정은 P001–P005의 실행 전 요건이다. 이 문서는 실험 실행 승인이 아니다.
