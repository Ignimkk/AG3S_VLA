---
name: cvpr-evidence
description: Trace CVPR claims and numerical statements to evidence records, data types, provenance, metrics, and verification status.
---

# CVPR Evidence Audit

[공통 workflow](../../../paper/notes/CLAUDE_WORKFLOW.md)의 Evidence tracking을 읽는다.
요청: `$ARGUMENTS`. `full`, `claim Cxxx`, `experiment EXPxxx`, `result Rxxx`,
`register candidate-path`를 지원한다. 기본은 `full` audit다.

1. [CLAIMS](../../../paper/evidence/CLAIMS.md),
   [EXPERIMENTS](../../../paper/evidence/EXPERIMENTS.md),
   [RESULTS](../../../paper/evidence/RESULTS.md), [METRICS](../../../paper/evidence/METRICS.md),
   [Manifest 운영](../../../paper/experiments/manifests/README.md)을 읽는다.
2. Repository root에서 `python3 paper/tools/analysis/validate_manifests.py`를 실행한다.
   실행 시각·exit status·검사 범위를 기록한다. 실패한 source의 결과 사용을 중단하고
   실패 원인을 보고한다. 실행하지 못하면 `NOT_CHECKED`로 남긴다.
3. 요청 범위의 claim→experiment→result→verify pointer·원자료·snapshot을 추적한다.
   실제 16-D, denominator, pairing, metric 정의, condition과 scope를 대조한다.
4. Safety/contact와 task outcome, source snapshot과 current main, 단순 차이와
   non-inferiority를 구분한다. 미측정 값을 0으로 처리하거나 실패 run을 임의로 제외하지 않는다.
5. 원고·figure/table이 있으면 수치·caption까지 역추적한다. 없으면 artifact audit는
   `NOT_CHECKED`다. Registry audit 성공을 원고 전체 audit 성공으로 보고하지 않는다.

등록 요청이 있으면 candidate의 provenance와 가능한 자료를 audit한 뒤 schema에 맞춰
manifest·관련 registry·source audit를 갱신한다. 측정·파생·추정·계획·placeholder와
검증 상태를 기록하며, 등록 상태는 논문 사용 가능성을 제한하지 않는다.
Audit만 요청받았을 때 status나 결과값을 변경하지 않는다.

보고 형식: 대상 ID/문장, source 경로·pointer, 판정(`VERIFIED_IN_SCOPE`, `UNSUPPORTED`,
`SOURCE_DRIFT`, `NOT_CHECKED`), 필요한 조치. Validator는 novelty·physical correctness나
train/test 독립성을 자동 인증하지 않으므로 해당 항목은 별도 판단한다.
