# Verified Results

Updated: 2026-10-05 19:45 KST

## 등록 범위

현재 결과는 **EXP001의 descriptive task outcomes만**이다. 아래 값은 새 실험이나
simulation 재실행 결과가 아니라 기존 verifier source의 수치를 연결한 것이다.
원자료 hash: `74a172b64af191f71b94557952678bb9bafc2604a43fcded14d9712832b8a78d`.

| Result | Metric·조건 | 값 | 정확한 source key |
|---|---|---|---|
| R001 | M001 / E0: T39 VLA 단독 | 30 / 48 | `/numbers/E0/n_success`, `/numbers/E0/n_runs` |
| R002 | M001 / E3b: T39 통합 snapshot | 29 / 48 | `/numbers/E3b/n_success`, `/numbers/E3b/n_runs` |
| R003 | M001_PAIRED: 둘 다 성공 / E0만 / E3b만 / 둘 다 실패 | 22 / 8 / 7 / 11 | `/numbers/paired_2x2/success` |

Source: [T39.verify.json](../../benchmark/ag3s/docs/handoff/T39.verify.json),
[T39 research log](../../benchmark/ag3s/docs/AG3S_T0T6_LOG.md),
[등록 manifest](../experiments/manifests/EXP001_T39_task_outcomes.json).
R003의 원 source object에는 exact McNemar p = 1.0도 기록되어 있다.
이 값으로 equivalence, non-inferiority 또는 "조작 능력 유지"를 확정하지 않는다.

## 해석 범위

- Model은 기록된 16-D checkpoint, condition은 해당 시점의 E0/E3b snapshot이다.
- M001은 기존 planning-row grasp/place 규칙이다. 물리적 collision 판정과 다르다.
- Test split 표시는 metadata에서 확인했으나 training/tuning과의 독립성은 아직 검토하지 않았다.
- 다른 episode의 seed를 별도 task로 간주하거나 시간별 frame을 독립 trial로 세지 않는다.
- 현 main은 이후 변경됐다. 이 결과를 최신 default gate/config의 성능으로 쓰지 않는다.
- Collision rate, safe success, full-body/attached-object clearance, grounding accuracy,
  real-robot 성과는 이 registry에 없다. 미측정은 `NOT_MEASURED` 또는 `null`이며 0이 아니다.
- Raw replay·ESDF proxy·HOLD count를 실제 실행 contact의 대용물로 섞지 않는다.

## 결과 추가 시 필수 항목

Result ID, experiment/condition/metric ID, source hash와 JSON pointer, 값·unit·denominator,
sample unit·aggregation·uncertainty, raw source, exclusion/retry 규칙, limitations,
claim/artifact 연결을 기록한다. 다른 scope의 결과는 새 항목으로 등록한다.

[METRICS](METRICS.md), [EXPERIMENTS](EXPERIMENTS.md), [CLAIMS](CLAIMS.md)를 함께 갱신한다.
