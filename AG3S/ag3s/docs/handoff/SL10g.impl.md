# SL10g — 구현

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead

## 무엇을 했나 (평이한 요약 먼저)
LIBERO-Safety paired launcher 가 `args.record_constraints = "none"` 을 코드에 박아 두어 server 의 constraint 기록(`--record-constraints`)을 켤 수 없었다. 이를 LIBERO launcher(`launch_libero_paired.py`)와 같은 `--record-constraints {none,occupancy,full}` 옵션으로 열었다. 기본은 `none` 이고, `none` 일 때 dry-run 출력 · server argv · settings 키가 base(541d0bc8)와 bit 동일하다. 기록 argv 생성 · settings · chunk 집계는 모두 shared launcher(`lp.run_mode`, `lp.record_constraints_argv`, `lp.record_constraints_summary`)의 것을 그대로 쓴다. 기록 로직은 복제하지 않았다. 서버 판 · flag 는 바꾸지 않았다.

## 바뀐 파일
| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/runners/launch_libero_safety_paired.py` (`build_parser`) | `--record-constraints` (choices = `lp.RECORD_CONSTRAINTS_CHOICES`, 기본 `none`) 추가 | 기록 켜기 |
| 같은 파일 `main` | 하드코딩 `args.record_constraints = "none"` 삭제; `vla_only` 모드와 함께면 `ValueError` (LIBERO launcher 와 같은 문구) | LIBERO launcher 와 같은 거부 |
| 같은 파일 `main` dry-run | server argv 에 `lp.record_constraints_argv(m, lp.RUN_DIR_TOKEN, …)` 를 붙여 출력 (`none` 이면 `[]`) | dry-run 이 실제 argv 와 맞음 |
| 같은 파일 docstring | 옵션 · 거부 한 줄 설명 | |
| `tests/benchmark/test_sl10g_ls_record_constraints.py` | 신규 (11 test) | 아래 |

경로 확인: LS 는 `lp.run_mode` 를 그대로 호출한다 (`install_hooks` 는 client/protocol/runtime/server_argv 만 바꿈). 따라서 `run_mode` 가 server argv 끝에 `--record-constraints <run_dir>/server_constraints --record-constraints-esdf <값>` 을 붙이고, `settings.record_constraints` 와 `launcher_result.record_constraints`(chunk 수 · bytes · writer_runs)를 쓴다. 서버는 `AG3S_RUN_DIR=<run_dir>` 로 뜨고 chunk 는 `<run_dir>/server_constraints/run_*/chunk_*.npz`.

`run_parallel_gpu.sh` 는 변경 없음: 기본 launcher 가 `launch_libero_paired.py` 이고 LS launcher 는 직접 실행한다(독점 flock 은 `run_mode` 안). `--record-constraints full` 은 직접 호출 인자로 그대로 간다.

## 단위 검증
```bash
cd .runtime/worktrees/sl10g-lsrec && AG3S_PROJECT_ROOT=$PWD PYTHONPATH=$PWD AG3S_STORAGE_ROOT=/mnt/dev/AG3S_VLA-storage MUJOCO_GL=osmesa \
  /mnt/dev/AG3S_VLA-benchmark/.venv-ag3s/bin/python -m pytest tests/benchmark/test_sl10g_ls_record_constraints.py \
  tests/benchmark/test_libero_safety_client.py tests/benchmark/test_launcher_record_constraints.py tests/benchmark/test_sl10f_uncwaive_version.py -q -p no:cacheprovider
```
결과는 아래 "한계" 참고. 새 test: parser(choices/default/잘못된 값), `none` dry-run 이 base wrapper(git show 541d0bc8)와 byte 동일(모드 3종), `none` 일 때 flag · settings 키 없음, `occupancy`/`full` dry-run = base + LIBERO helper argv, `run_mode` 에서 resolved argv · token argv · `settings.record_constraints` · `launcher_result.record_constraints.n_chunk_files`, `vla_only` 거부 2종, LIBERO helper 의 거부(비 server 모드 · 알 수 없는 값).

## verifier 가 알아야 할 것
- 새로 생긴 플래그: `--record-constraints {none,occupancy,full}` (기본 `none` = 이전과 동일). 다른 기본값 변화 없음.
- 재생산이 필요한 산출물: 없음. `full` 실행 시 `<run_dir>/server_constraints/` 에 chunk npz 가 생긴다 (용량은 SL5j 와 같음).
- 옛 기록과 호환이 깨지는가: 없음 (`none` 이면 기록 키 자체가 없음).
- **한계:** worktree 에는 `.venv-ag3s`, vendored `benchmark/LIBERO-Safety`, `benchmark/LIBERO` 내용이 없다. 그래서 (1) 새 test 의 dry-run 은 `l0.libero_control_freq` 를 fake(20)로 바꾼다 (값은 검증 대상 아님), (2) worktree 에서는 `test_libero_safety_client.py::test_adapter_uses_fork_controller_constants`, `::test_task_map_matches_constants` 와 `test_launcher_record_constraints.py::test_parallel_script_passes_the_flag_through` 가 vendored 파일 · `.venv-ag3s` 부재로 실패한다. 세 개 모두 이 변경과 무관하며 main checkout 에서는 29 passed, 3 skipped.
- 첫 실행(SL10g) 시 `J0`: 이전 pc판 run(`76283ac9`)과 episode 별 성공 · 실행 action bit 동일 확인은 verifier/lead 몫 (기록 쓰기는 서버 계산을 바꾸지 않는다고 SL5j 에서 확인됨, 여기서 재측정하지 않음).

## 내가 기대하는 결과
dry-run · 단위 test 수준의 변경이라 측정 결과에 대한 기대는 없다.
