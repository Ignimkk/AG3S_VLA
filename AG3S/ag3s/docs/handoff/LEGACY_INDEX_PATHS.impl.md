# LEGACY_INDEX_PATHS — 구현 (migration fix)

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead · 2026-10-12 14:17 KST · base `bd778e5e`

## 목적
`/mnt/dev/work` 에서 옮긴 뒤 실패하던 recorded-replay test 42개가, 공유 storage 의 기록 input 만 읽고
(`/mnt/dev/work` 접근 없이) 다시 돌게 한다. 기록 JSON 은 고치지 않는다 (read-only input). algorithm code 는 손대지 않았다.

## 원인
T43 S/U 의 index JSON (`results/workspace/verify/T43/S/raw/chunk_index_t40t41.json`, `chunk_index_t42.json`,
`held_runs_all.json`, `T43/U/raw/chunk_index_t43v_t42_2.json`) 은 key 와 nested value
(`server`, `first_chunk_file`, `chunks[*].npz`) 에 옛 절대경로 `/mnt/dev/work/outputs/verify/...` 를 담고 있다.
test 는 key 를 `workspace_path('legacy/outputs/verify')` = `/mnt/dev/AG3S_VLA-storage/legacy-inputs/outputs/verify/...`
로 계산하므로 `KeyError` (42 failed, `results/development/20261012T132631+0900_8efc1157a495_414d71da8006/impl/T46/pytest_all.log`).

## 바뀐 파일
| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `common/workspace.py:40` | `OLD_CHECKOUT_PREFIX = "/mnt/dev/work/"` (기존 `bind_historical_client_inputs` 안의 지역 `old_prefix` 를 module 상수로) | mapping 을 한 곳에 둔다 |
| `common/workspace.py:43` | `rebase_legacy_path(str)` — `/mnt/dev/work/<rel>` → `legacy_input_path(<rel>)`, 그 밖은 그대로 | 기존 checkpoint mapping 과 같은 규칙 재사용 |
| `common/workspace.py:50` | `rebase_legacy_paths(obj)` — 읽은 JSON 의 dict key·string value·list 를 재귀로 rebase (사본 반환, 입력 불변) | index 의 key 와 nested path 둘 다 |
| `common/workspace.py:69` | `bind_historical_client_inputs` 가 `rebase_legacy_path` 를 쓰도록 (동작 동일) | 중복 제거 |
| `tests/trajopt/test_t43r2_held_body_cover_replay.py:100,116` | index 와 `held_runs_all.json` 을 load 직후 `rebase_legacy_paths` | `truth` 도 옛 key 라 `.get` 이 조용히 `None` 이 되는 것을 막음 |
| `scripts/testing/t43t/{ground_replay:9, qnow:15, probe:8, closing_replay:21, scan:6, support_scan:7}.py` | module-level `IX = rebase_legacy_paths({...})` | `test_t43t_replay.py` 가 `ground_replay`·`qnow` (→ `familyA_step2`) 를 거쳐 읽음. 같은 index 를 읽는 나머지 넷도 같은 한 줄로 맞춤 |
| `tests/test_workspace_rebase.py` (신규) | helper unit test 3개 (key/nested/list rebase, 다른 문자열 불변·입력 불변, checkpoint binding 동일) | common code 변경의 coverage |

T46 / T47 의 uncommitted file 은 건드리지 않았다 (test 실행 시 tree 에는 그대로 있었다).

## 단위 검증
run: `results/development/20261012T141738+0900_bd778e5e3453_df29d87addf1/impl/legacy-index-paths/`
```bash
# GPU lock (/tmp/ag3s-vla-gpu-validation.lock) + nvidia-smi compute process 없음 확인 후,
# XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa PYTHONPATH=$PWD
.venv-openpi-live/bin/python scripts/setup/run_without_legacy.py -m pytest \
  tests/trajopt/test_t43r2_held_body_cover_replay.py tests/trajopt/test_t43t_replay.py tests/test_workspace_rebase.py -q
```
| 대상 | 이전 | 이후 |
|---|---|---|
| 두 test file (42 collected) | 42 failed (`KeyError`) | **42 passed** (`/mnt/dev/work` read denial active) |
| `tests/test_workspace_rebase.py` | — | 3 passed (`.venv-openpi-live`, `.venv-ag3s` 모두) |
| `test_legacy_retirement.py`·`test_run_output.py`·`test_policy_record_output.py` (`.venv-ag3s`) | — | 29 passed, 1 skipped |

log: `pytest_two_files_no_legacy.log` (45 passed, 100 s), `pytest_unit_{openpi,ag3s}.log`, `grep_old_prefix_tests.txt`.
`tests/`·`scripts/testing/` 의 다른 `/mnt/dev/work` 참조는 `test_policy_record_output.py:64` (config 문자열 literal, 기록 index 아님) 뿐이다.

## verifier 가 알아야 할 것
- 새로 생긴 플래그·기본값 변경: 없음 (`common.workspace` 에 public helper 2개 + 상수 1개 추가)
- 재생산이 필요한 산출물: 없음. 기록 JSON 수정 없음
- 옛 기록과 호환이 깨지는가: 없음. 전체 회귀는 verifier 몫

## 내가 기대하는 결과
전체 pytest 에서 이 42개가 빠져 이전 `42 failed, 2746 passed` 의 failed 가 0 이 된다 (T46 변경과 무관할 때).
