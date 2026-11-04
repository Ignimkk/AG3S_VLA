# SL10e — 구현 (들어 올림 때도 아래 받침면 허용)

> writer: ag3s-implementer · 읽는 쪽: verifier, scribe, lead · worktree `wip/sl10e-plift` · 2026-11-05

## 무엇을 했나 (평이한 요약 먼저)
HSR6 ⑤(내려놓기 접촉 허용: 쥔 물체 행이 그 아래 받침면에 막히지 않음)는 단계 라벨이 `placement`일 때만 켜졌다.
새 flag `--hsr6-placement-contact-lift`(기본 꺼짐)를 켜면 ⑤의 단계 조건이 `placement ∨ lift`가 된다. latch 조건(HELD, 또는 CLOSING ∧ closed ∧ blocked)과
풀리는 행의 기하 조건(`_placement_contact_margin`: 쥔 물체 행만, 아래 45° 원뿔, 쥔 물체 xy footprint)은 건드리지 않았고 `transport`에도 켜지 않는다.
판 두 개(`-pc-plift`, `-pc-surface-plift`)를 등록했다.

## 단계 라벨 문자열 확인
- `AG3S/trajopt/hsr6.py` `PHASES6 = ("approach", "grasp", "lift", "transport", "placement", "retreat")`, `PLACEMENT = "placement"`. `_hsr6_phase()`는 `phase_of(...)`의 raw argmax 문자열을 그대로 돌려준다.
- Panda probe(`hsr6_panda_kv_L4.npz`)는 retreat만 없는 5 class라 `lift` · `transport` · `placement` 모두 같은 문자열이다. SL10d 기록(`held_phase_placement.json`)의 키도 `lift` · `transport` · `placement`.
- `--hsr5`(사용 안 함)의 `RULE_PHASE6`은 carry→`transport`, place→`placement`이고 `lift`로 가는 stage는 없다. 즉 `--hsr5`와 함께 쓰면 lift 확장은 아무 일도 안 한다 (Panda 판은 `--hsr5`를 쓰지 않는다).

## 바뀐 파일
| 파일 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/trajopt/hsr6.py` | `PLACEMENT_LIFT_PHASES`, `PLACEMENT_LIFT_RULE`, `Hsr6Flags.placement_contact_lift`(마지막 field, `on`에 포함), `placement_contact_ok(..., lift=False)` | 단계 조건만 넓힘. `lift=False`면 이전 식과 같다 |
| `AG3S/trajopt/safe_policy.py` `_hsr6_placement_contact` | flag가 켜졌을 때만 `lift=True`를 넘기고 `rec["phase_rule"]="placement\|lift"` 기록 | 꺼짐이면 호출 인자 · 기록 키가 이전과 같다 |
| `AG3S/trajopt/serve_safe.py` | `--hsr6-placement-contact-lift`, `hsr6_flags_from_args`, `reject_hsr6_flags`(⑤ 없이 거절), `reject_panda_only_flags`(RB-Y1 거절) | 기본 꺼짐, 거절 규칙 |
| `AG3S/adapters/panda/profile.py` | `PLACEMENT_LIFT_FLAGS`, `SL10E_VERSIONS`(두 판), `PC_LATCH_VERSIONS`에 추가, `version_flag_values`에 새 dest는 켜졌을 때만 합류 | SL10d 판과 같은 그룹 규칙 |
| `tests/benchmark/test_sl10e_placement_lift.py` | 38 test | 아래 |

## 단위 검증
```bash
cd <worktree>; export AG3S_STORAGE_ROOT=/mnt/dev/AG3S_VLA-storage AG3S_PROJECT_ROOT=$PWD
MUJOCO_GL=osmesa PYTHONPATH=$PWD /mnt/dev/AG3S_VLA-benchmark/.venv-ag3s/bin/python -m pytest <files> -q -p no:cacheprovider
```
- `test_sl10e_placement_lift.py`: 38 passed (꺼짐 bit 동일: phase 8 × latch 6 × checks 7 = 336 조합을 독립 복사본과 비교; 켬 표; lift 행은 placement 행과 같은 latch 조건; SafePolicy 기록 키; 판 = 기준 + flag 하나; 서버 build가 `Hsr6Flags.placement_contact_lift`를 SafePolicy에 전달; ⑤ 없이 거절; RB-Y1 거절(`main` 포함)).
- 회귀: `test_sl10*.py`, `test_sl9_ablation_versions.py`, `test_sl6h_panda_placement_contact.py`, `tests/trajopt/test_sl6h_placement_contact.py`, `test_sl7_panda_p0m.py`, `test_merge_dev_main_20261103_panda.py` — 273 passed, 2 failed (환경 원인, 아래).
  - `test_sl6h_panda_placement_contact.py::test_client_row_records_the_hsr6_record`: worktree에 `.venv-libero`가 없어 py3.8 client subprocess가 FileNotFoundError (코드와 무관).
  - `test_sl10_panda_fixed_input.py::test_tool_does_not_modify_server_adapter_or_recorder`: 커밋 전 작업 트리의 변경을 보는 guard라 commit 전에만 실패한다 (commit 뒤 재실행 결과는 최종 보고).
- `AG3S.trajopt.hsr6.__file__`이 worktree를 가리키는 것을 확인했다.

## verifier 가 알아야 할 것
- 새 flag: `--hsr6-placement-contact-lift` (기본 꺼짐). 판 `AG3S-final-panda-v3-e0to-pc-plift`, `AG3S-final-panda-v3-e0to-pc-surface-plift`.
- 켜면 기록: `hsr6.detail.placement_contact.phase_rule = "placement|lift"`, `hsr6.flags`에 `placement_contact_lift`. 꺼짐이면 두 곳 모두 이전 키 그대로.
- 판의 flag 값 기록(`version_flag_values`)은 새 dest를 켠 판에서 SL10 · SL10d 판처럼 이전 그룹의 dest(기본값)도 같이 들어온다. 기준 판의 기록 키는 바뀌지 않는다.
- 기제 확인에 쓸 필드: `hsr6.detail.placement_contact.{on, phase, applied, rows_released, held_rows_released}`. lift call에서 `phase=="lift"` ∧ `held_rows_released>0` 비율이 판정의 "≥ 0.5"에 해당한다.
- 재생산 필요한 산출물 없음. 옛 기록과 호환 깨짐 없음.
- transport에서 아래를 지나는 장애물은 허용되지 않는다 (의도). 단, lift 단계 라벨이 실제 들어 올리기가 끝난 뒤에도 남을 수 있어 쥔 물체가 옆으로 움직이는 동안의 "아래 원뿔" 조건은 기하 조건이 그대로 막는다.

## 내가 기대하는 결과
(구현자 기대는 측정 전 verifier에게 보이지 않는 절이지만 이 단계는 구현 사실만 있어 비워둔다.)
