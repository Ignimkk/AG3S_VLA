# V7g — 논문의 주 구성 (sub-task gate ON)

작성: 2026-10-06 KST. 사용자 지시: "gate on으로 실험했습니다 … 사용한 것을 기반으로 논문작성". 모든 수치는 등록 전(UNREGISTERED)이며
본문에는 쓰지 않는다. 근거: `benchmark/ag3s/docs/handoff/T43V7g.verify.json`, `T43GD.verify.json`, `T43Z3.impl.md`, `T43Z6.impl.md`,
`outputs/verify/T43/V7g/p/{start_server,run_client}.sh`, `SESSION_STATE.md` (2026-10-06 08:10–10:35 UTC).

## 구성 (server flags)

V7g = V5 − `--no-subtask-gate` + `--held-slip-detach` + `--held-cover-cap-mm 10` + `--hold-follow-to 3`
- E3B 공통: `--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized
  --esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32`
- V5 추가: `--obstacle-margin 0.010 --held-body-cover --closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60`
- client: `--safe-gate reasons --safe-hold-mode measured --safe-phase approach --safe-manipulators {left|right}`
- snapshot `/mnt/dev/work-t43v7` = benchmark `f2f335b`.

## 새 구성요소의 동작 (방법 서술용)

| flag | 동작 |
|---|---|
| sub-task gate ON | label이 place/home이면 첫 target 채택과 switch challenger를 막는다. PLACED 뒤 home이면 manipulated object를 해제 → exemption 종료 |
| `--held-slip-detach` | HELD에서 (attach 때 개도 − 현재 개도 ≥ 0.05) ∧ (현재 개도 − 마지막 실행된 명령 < 0.05)가 2 요청 연속 → detach, PLACED. 실행 피드백만 사용 |
| `--held-cover-cap-mm 10` | held query 구: 반지름 ≤ 관측 반폭 + 10 mm, 중심은 관측 몸통 중심에서 10 mm 안으로 당김, 중복 제거. 길쭉한 물체(주축 > 폭×1.6)는 제외 |
| `--hold-follow-to 3` | HOLD일 청크가 (collision 행만) ∧ (모든 행이 ESDF−r ≥ 0) ∧ (최소 clearance가 현재보다 나쁘지 않음)이면 refined chunk 실행. V7g 19 run에서 실행 0, 거절 16 |

## 측정 (미등록, 저자 참고용)

- 19 run (H 14 + B 5), 재시도 0. S/G/P 16/19 = V5와 동일(p 1.0). 접촉 포함 허들 성공 9/14(T43CT).
- HOLD 청크 V5 217 → V5D 58 → V7g 16. held 행 HOLD 166 → 34 → 1.
- gate: 1,146 청크 중 label pick 374 · place 296 · home 438 · None 38; 첫 채택 차단 387, switch 차단 54.
- GD (V7 gate off vs V7g on, 19 쌍): 성공 동일 16/16; PLACED 뒤 target switch 25 → 0, carve 440 → 60, `subtask_no_target` 0 → 423.
- 왕복 시간은 여전히 chunk 주기(533 ms)를 넘는다(V5 기준 median 1,145.7 ms).

## 논문 서술에서의 주의

- 성공률에서 gate on/off 차이는 없다. gate의 효과는 **놓은 뒤의 target 처리**(잘못된 재채택·carve 제거)로 보여야 한다.
- follow-TO는 켜져 있으나 한 번도 실행되지 않았다 → 방법의 핵심으로 쓰지 않는다.
- 진행 중: V8(`--finger-cover`), DX(사과–막대 접촉 진단). 결과가 나오면 주 구성이 바뀔 수 있다.
