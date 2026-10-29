# Baselines and Comparison Protocol

Updated: 2026-10-05 19:45 KST

## 기록된 baseline

| ID | 조건 | 범위·status |
|---|---|---|
| B001 | T39 E0: 같은 16-D VLA checkpoint, `--no-safe` | EXP001에 등록된 descriptive task outcome |
| B002 | T39 E3b: AG3S + cuRobo ESDF + project SQP/OSQP, gate/HOLD | EXP001의 비교 조건; latest main 또는 모든 robot link의 safety 보장 아님 |

[T39 실행 설정](../../outputs/verify/T39/p/start_server.sh),
[client 설정](../../outputs/verify/T39/p/run_client.sh),
[manifest](../experiments/manifests/EXP001_T39_task_outcomes.json)에 snapshot과 flags를 기록했다.
B002는 `--links gripper`, `--safe-phase approach`, `--safe-manipulators left`,
`--no-self-collision` 등 해당 평가 범위를 포함한다.

## 계획한 비교와 ablation

| 후보 | 역할 | 확정 전 필요한 것 |
|---|---|---|
| Same-policy vanilla VLA | Main 비교 | 동일 checkpoint, scene, seed, horizon, allowed contact |
| 동일 refinement + 대조 target 정보 | Representation/grounding 기여 분리 | Attention 외 변경 요인을 고정하고 비교 validity 확인 |
| Target separation/contact policy 제거 | Target-aware 구성 ablation | task 접촉과 obstacle contact 판정 유지 |
| Safety field/refinement/gate 별도 제거 | 구성 요소 기여 분리 | joint 변경을 단일 요소 ablation으로 부르지 않기 |
| GT geometry 또는 GT target | Oracle upper-bound/진단 | GT 제공 범위 명시; deployable baseline과 구분 |
| 선행연구 method | 외부 scientific baseline | 논문·code·license·robot/task 호환성과 실행 조건 확인 |

합성 attention은 control/diagnostic이며 실제 VLA attention 결과에 섞지 않는다.
기존 `legacy`/`cuRobo` regression baseline은 코드 회귀 검사다. 논문 method competitor나
공식 cuRobo MotionGen 비교로 자동 전환하지 않는다.

## 공정 비교 규약

Checkpoint, instruction, episode/split, scene, policy seed, deterministic warmup,
sensor/calibration, execution horizon, timeout/stop 기준, metric implementation,
자원 조건과 software revision을 같게 하거나 차이를 명시한다.

Failure·HOLD·timeout·missing output·retry를 버리지 않는다. 평가 불가능한 safety 값은
unknown으로 남기고 coverage를 보고한다. 실험 후 성공 episode만 골라 만든 obstacle 시연은
전체 benchmark의 main result나 generalization 성능으로 해석하지 않는다.

외부 baseline과 reference 검토는 [RELATED_WORK_MAP](../notes/RELATED_WORK_MAP.md)를 따른다.
