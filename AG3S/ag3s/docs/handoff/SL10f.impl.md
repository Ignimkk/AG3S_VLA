# SL10f impl — 미인증 HOLD 풀기 진단 판 등록 (2026-11-05)

task: `SL10f.task.md` (main checkout). 새 코드 경로 없음, 판 등록만.

## 바꾼 것

| 파일 | 내용 |
|---|---|
| `AG3S/adapters/panda/profile.py` | `UNCERTIFIED_WAIVE_FLAGS = ("--allow-uncertified",)`, `SL10F_VERSIONS`, 판 `AG3S-final-panda-v3-e0to-pc-uncwaive` = pc판 flag + 이 flag 하나. `PC_LATCH_VERSIONS` 에 추가(HSR6 probe 를 pc판처럼 받음). `version_flag_values` 에 SL10f 그룹: 새 dest `allow_uncertified` 는 켜졌을 때만 기록 키에 합류 (꺼진 기존 판의 기록 키 불변; 합류하지 않으면 `match_ag3s_version` 이 새 판을 pc판으로 읽는다) |
| `tests/benchmark/test_sl10f_uncwaive_version.py` | 6 test |

## 서버 수락 여부

- 거절 없음. `allow_uncertified` 는 `UNPORTED_FINAL_CONFIG_DESTS` 에 없고, `check_unported_flags` · `reject_hsr6_flags` · `reject_panda_only_flags` 통과, `build_safe_server` (CPU, stub) 가 빌드됨.
- 해석된 TrajOpt config: 새 판 `safety.require_certified_geometry=False` (SafePolicy `to_config` 에서도), pc판 `True`.
- 안전 규칙 · 기본값 · 다른 판은 건드리지 않았다.

## test

- 새 test 6/6. 판 = 기준 + flag 하나, 기존 50판 `AG3S_VERSIONS` 해시 `9cb986d5…` 동일 (62b4a888 에서 측정), 새 판 외 어떤 판 기록에도 `allow_uncertified` 키 없음, `lp.server_argv` · 안전 launcher `make_server_argv` · `version_server_flags` 가 판을 받음.
- 같이 돌린 sl10e · sl9 · sl6h: 71 passed, 1 failed (`test_client_row_records_the_hsr6_record`). 실패 원인은 worktree 에 `.venv-libero` 가 없어서(FileNotFoundError) 이고, main checkout 에서는 통과한다. 이 변경과 무관.
- `eval_matrix.build_jobs` 는 같은 이유로 worktree 에서 못 돌려서, 그것이 쓰는 `version_server_flags` 호출로 갈음했다.
