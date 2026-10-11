---
name: cvpr-experiment
description: Plan CVPR experiments and audit existing protocols or candidate records without launching research runs.
---

# CVPR Experiment Protocol

[공통 workflow](../../../paper/notes/CLAUDE_WORKFLOW.md)의 연구 실행 경계를 읽는다.
요청: `$ARGUMENTS`. `plan Pxxx`, `audit EXPxxx 또는 candidate 경로`를 지원한다.
기본은 [EXPERIMENTS](../../../paper/evidence/EXPERIMENTS.md)의 미정 protocol 정리다.

1. 해당 Claim, [BASELINES](../../../paper/evidence/BASELINES.md),
   [METRICS](../../../paper/evidence/METRICS.md), manifest schema를 읽는다.
2. 비교 질문과 baseline/method snapshot을 고정한다. 동일 checkpoint·scene·seed·budget,
   train/tuning/test provenance와 실제 state/action 16-D를 명시한다. E3b의 project SQP를
   cuRobo MotionGen baseline으로 바꾸어 설명하지 않는다.
3. Allowed contact, robot/payload 범위, independent contact GT, substep coverage,
   task-success 판정, 실패·timeout·retry·exclusion 규칙을 사전에 정의한다.
4. Sample size·paired design·episode cluster·aggregation·uncertainty를 설계한다.
   성능 유지 주장은 사전 margin과 non-inferiority 분석이 필요하다. 미정 값은 `TBD`다.
5. Runtime에서는 hardware·clock·warm/cold start·동기화·end-to-end 측정 경계와
   deadline miss를 정의한다. 관측되지 않은 실시간 Hz나 collision metric을 추정하지 않는다.
6. `paper/experiments/configs/`에 protocol을 작성하고 계획 registry에 연결한다.
   기존 자료 audit는 completion·source hash·metric·selection bias와 로그 연결을 보고한다.

산출물: 질문, 통제 조건, metric, 분석 계획, raw/provenance 수집 요건, 실행 담당자에게
전달할 명령 초안 및 미정 항목. 명령은 초안임을 표시한다.
이 skill은 GPU/model/MuJoCo/실제 robot 실험을 시작하지 않는다. 실제 실행은 별도의
연구 수행 요청과 기존 AG3S 구현·검증 workflow를 따른다. 계획 파일은 active whitelist가 아니다.
