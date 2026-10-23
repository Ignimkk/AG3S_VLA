# SL5j-launcher — 구현

> writer: ag3s-implementer · 읽는 쪽: lead, verifier, scribe · 작성 2026-10-23 KST · 기준 HEAD `32a4ca8f` (commit 안 함)

## 무엇을 했나 (평이한 요약 먼저)

benchmark launcher에 `--record-constraints {none,occupancy,full}` 인자를 넣었다. 기본값 `none`이면 server argv,
manifest settings, dry-run 출력이 SL5j 전(`32a4ca8f`)과 byte 단위로 같다. 이것은 옛 launcher를 git에서 그대로
불러와 새 launcher와 나란히 돌려 비교하는 test로 고정했다. `none`이 아니면 VLA+V11AB server argv 끝에
`--record-constraints <run_dir>/server_constraints --record-constraints-esdf <값>`을 붙이고, `vla_only` mode와 함께 주면
run을 만들기 전에 거부한다.

**막는 문제가 하나 있다.** Panda server는 이 flag를 parse만 하고 기록기를 만들지 않는다(아래 "verifier가 알아야
할 것" 1). 그래서 launcher만 고친 지금 상태로 SL5j-G를 돌리면 `server_constraints/`에 청크가 **하나도 남지 않는다**.
SL5j-G 전에 adapter(`AG3S/adapters/panda/server.py`)에서 기록기를 연결하는 작업이 따로 필요하다. 이번 범위에서는
adapter를 고치지 않았다.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/runners/launch_libero_paired.py:18-24` | docstring에 사용법과 manifest 기록 방식 | |
| `benchmark/runners/launch_libero_paired.py:51-53` | `RECORD_CONSTRAINTS_CHOICES`, `RECORD_CONSTRAINTS_DIRNAME = "server_constraints"`, `RUN_DIR_TOKEN = "<run_dir>"` | |
| `benchmark/runners/launch_libero_paired.py:70-79` | `record_constraints_argv(mode, run_dir, record)`: `none`이면 `[]`, 그 밖에는 flag 4개. `vla_only`이거나 모르는 값이면 `ValueError` | 기본값일 때 argv가 바뀌지 않게 하려고 `[]`를 이어 붙이는 방식으로 했다 |
| `benchmark/runners/launch_libero_paired.py:82-88` | `record_constraints_summary`: server가 멈춘 뒤 `run_*/chunk_*.npz` 개수, byte, `meta.json` 유무를 센다 | 기록이 비어도 run은 끝까지 돈다. 결과를 `launcher_result`에 남겨 빈 기록을 바로 알아보게 했다 |
| `benchmark/runners/launch_libero_paired.py:213-228` | `settings.server_argv`에 `<run_dir>` token이 든 flag를 넣고, `settings.record_constraints = {esdf, dir, argv_note}` 추가(`none`이 아닐 때만) | `create_run`이 manifest를 쓰는 시점에는 run 디렉토리 경로(timestamp 포함)가 아직 없다 |
| `benchmark/runners/launch_libero_paired.py:236` | `create_run` 뒤 실제 경로로 flag를 붙인다. `role=server` execution entry와 `Popen`에는 이 실제 argv가 들어간다 | |
| `benchmark/runners/launch_libero_paired.py:273-280` | server를 멈춘 뒤 `launcher_result.record_constraints`를 기록하고, 청크가 0개면 `!!!` 경고를 낸다 | |
| `benchmark/runners/launch_libero_paired.py:352` | argparse 인자(choices, 기본 `none`) | |
| `benchmark/runners/launch_libero_paired.py:374-377` | `vla_only`가 `--modes`에 있으면 `ValueError`. 기본 `--modes`가 `vla_only,vla_v11ab`이므로 `--modes vla_v11ab`를 꼭 줘야 한다. checkpoint 검사와 run 할당보다 먼저 거부하도록 `modes` 계산을 checkpoint 검사 앞으로 옮겼다 | task 1항 |
| `benchmark/runners/launch_libero_paired.py:380-382` | dry-run 출력에 flag를 보인다(`<run_dir>` token) | |
| `tests/benchmark/test_launcher_record_constraints.py` (신규, 20 test) | 아래 | |

`run_parallel_gpu.sh`는 바꾸지 않았다. job line을 `shlex.split` 해서 그대로 넘기므로
(`run_parallel_gpu.sh` `start_job`), `--record-constraints occupancy`가 launcher argv에 그대로 들어가는지 test로 확인했다.

## 사용법

```bash
source scripts/setup/env.sh
.venv-ag3s/bin/python benchmark/runners/launch_libero_paired.py --modes vla_v11ab \
  --task-suite-name safelibero_spatial --safety-level I --data-split "<split>" \
  --task-ids 0,1,2,3 --episode-ids 0,1,2,3,4 --run-id sl5j-... --conversion C --ag3s-version V11AB-pb7 \
  --record-constraints full
# run_parallel_gpu.sh jobs 파일에서도 같은 줄에 --modes vla_v11ab --record-constraints full 을 붙이면 된다
```

manifest에서 찾는 곳:

- `settings.record_constraints`: `{"esdf": "full", "dir": "<run_dir>/server_constraints", "argv_note": ...}`
- `settings.server_argv`: 끝 4개가 `--record-constraints <run_dir>/server_constraints --record-constraints-esdf full`(token 그대로)
- `executions[role=server].argv`: 실제 경로로 바뀐 argv
- `executions[role=launcher_result].record_constraints`: `{esdf, dir, writer_runs, n_chunk_files, bytes, meta_json}`

## 단위 검증

```bash
source scripts/setup/env.sh
MUJOCO_GL=osmesa PYTHONPATH=$PWD .venv-ag3s/bin/python -m pytest -q tests/benchmark/test_launcher_record_constraints.py
#   20 passed
MUJOCO_GL=osmesa PYTHONPATH=$PWD .venv-ag3s/bin/python -m pytest -q tests/benchmark/test_parallel_gpu_launcher.py \
  tests/benchmark/test_safelibero_client.py tests/benchmark/test_panda_sl5a.py tests/benchmark/test_launcher_record_constraints.py
#   52 passed
MUJOCO_GL=osmesa PYTHONPATH=$PWD .venv-ag3s/bin/python -m pytest -q tests/benchmark
#   148 passed, 17 skipped, 4 failed + 2 errors — 전부 test_panda_sl3.py, ModuleNotFoundError: openpi_client.
#   HEAD 32a4ca8f(이 변경을 stash 한 상태)에서도 같은 4 failed + 2 errors가 난다. 이 변경과 무관하다.
```

| test | 무엇을 고정하나 |
|---|---|
| `test_default_server_argv_and_settings_byte_identical_to_pre_sl5j` × {vla_only, vla_v11ab} × {V11AB, V11AB-pb5, V11AB-pb7} | `git show 32a4ca8f:benchmark/runners/launch_libero_paired.py`를 별도 module로 불러와 같은 mock(port, clock, subprocess, lock)으로 `run_mode`를 돌린다. `Popen` argv, `create_run(settings=…)`, execution entry 3개(server, client, launcher_result)의 `json.dumps` byte가 같다 |
| `test_default_dry_run_output_byte_identical_to_pre_sl5j` | `main --dry-run` stdout byte가 같다 |
| mutation 확인(한 번 실행하고 원복) | 기본값을 `occupancy`로 바꾸면 위 identity test 7개가 전부 실패한다. 즉 test가 실제로 차이를 잡는다 |
| `test_record_flags_are_appended_under_the_run_dir[occupancy/full]` | 실제 argv = 기존 argv + `--record-constraints {run_dir}/server_constraints --record-constraints-esdf <값>`(맨 끝). settings에는 token이 든 argv가 들어간다. `launcher_result`에서 청크 3개, 30 byte, `run_0000`, `meta.json`을 센다 |
| `test_record_flags_parse_in_serve_safe` | 붙인 argv가 `serve_safe.build_parser` + `reject_bad_flag_combinations`를 통과하고, V11AB-pb7 `v11ab_build_kwargs`가 그대로다 |
| `test_no_chunk_file_is_reported` | 청크가 0개면 `n_chunk_files: 0`이 남고 `!!!` 경고가 출력된다 |
| `test_vla_only_mode_is_refused_before_any_run` × 4가지 `--modes` | 기본값, `vla_only`, 두 순서 모두 `ValueError`. checkpoint 검사와 `create_run`보다 먼저 거부한다 |
| `test_vla_only_argv_is_refused_at_the_helper`, `test_bad_choice_is_rejected_by_the_parser` | helper와 argparse에서 막는다 |
| `test_parallel_script_passes_the_flag_through` | `run_parallel_gpu.sh`(fake launcher, nvidia-smi, cgroup)가 job line을 바꾸지 않고 넘긴다 |

## 기록 용량 추정 → `full`을 권한다

근거 1, 실제 기록(Panda run에는 기록이 없어서 RB-Y1 cuRobo `full` 기록을 썼다):
`development/20261021T205721+0900_f15e587290f6_ceef1742ca26/verify/T63b/cr0m/runs/server_8292_20261021T155515/server_constraints/run_0000`,
750 청크.

| 항목(청크 하나, zip 뒤) | MB |
|---|---|
| 청크 평균(최소 – 최대) | 4.63 (4.07 – 5.43) |
| `esdf_distance`(주 계층 5 mm, 128³ float16, 원본 4.19 MB) | 1.09 – 1.31 |
| `esdf_free_distance`(target 없는 계층) | 1.08 – 1.30 |
| `esdf_coarse_distance`(20 mm) | 0.84 – 0.95 |
| 거리장 3계층 합 | ≈ 3.0 – 3.5 |
| 점구름 3개 + attention 2개(38,324 점) | ≈ 1.5 |
| clearance, sphere, chunk, summary | ≈ 0.05 |

근거 2, CPU dummy `ConstraintRecordWriter`(동기, Panda 모양: 128³ 3계층, H 10 × S 220 clearance, 점 9,400 / 33,896개, 해석적
LIBERO 장면 거리장). 스크립트는 `/tmp/claude-0/-mnt-dev-AG3S-VLA-benchmark/924e7536-bb1c-4b05-ae29-7bedc4b65559/scratchpad/dummy_record_size.py`,
수치는 같은 폴더의 `dummy/sizes.json`.

| esdf mode | 점 9,400 | 점 33,896 | 동기 쓰기 ms/청크 |
|---|---|---|---|
| none | 0.42 MB | 1.41 MB | 13 – 47 |
| occupancy | 0.42 MB | 1.41 MB | 13 – 47 |
| full | 2.44 MB | 3.43 MB | 135 – 165 |

합성 거리장은 실제보다 잘 압축된다(3계층 ≈ 2.0 MB, 실제는 3.0 – 3.5 MB). 그래서 `full`은 실제 기록 쪽 수치를
쓴다. 점 개수 범위는 SL5i calls log에서 잡았다. Panda는 `points: 108912 raw -> 33896 voxel`이고 RB-Y1은
`497278 -> 138285`인데, RB-Y1 기록에는 38,324점이 남았다. 같은 비율이면 Panda는 약 9.4k점, 최대로 잡으면 33.9k점이다.

**SL5j-G 규모.** SL5i-G 파지 인식 + closing 면제판(V11AB-pb7) run의 call 수는 LI dev 20 episode 1,012, LIBERO 10 init0
325로 합계 **1,337**이다.

| mode | 청크당 | × 1,337 call | 비고 |
|---|---|---|---|
| occupancy | 0.4 – 1.5 MB | **0.6 – 2.0 GB** | **cuRobo backend에서는 점유 배열이 없다.** 격자 메타, 점구름, 제약 행만 남는다 |
| full | 3.5 – 5.4 MB | **4.7 – 7.3 GB**(RB-Y1 평균 4.63 MB로 6.2 GB) | 거리장 3계층(주, coarse, target 없는 계층)이 남는다 |

storage(`/mnt/dev`)에는 8.3 TB가 비어 있다. 두 mode 모두 용량은 제약이 되지 않는다.

`full`을 권하는 이유는 진단 가능성이다.

- Panda server는 `--esdf-backend curobo`로 뜬다. cuRobo backend에서 `occupancy` mode는 `esdf_occupancy`를 남기지 않는다.
  `constraint_record.py:483-496`은 `esdf_occupancy_available: False`를 쓰고, `serve_safe.py:2353`이 시작할 때 이를 경고한다.
  그러면 SL5j-G J1 ③(기록된 ESDF 계층에서 가장 가까운 점유 voxel)을 잴 수 없다.
- H-ghost(쥔 bowl 옛 자리의 잔상)는 TSDF가 여러 frame에 걸쳐 기억한 것이다. 이번 frame의 점구름에는 보이지 않는다.
  이것을 보려면 거리장 자체, 특히 `esdf_free_distance`와 주 계층의 차이가 있어야 한다.
- 쓰기 비용: 동기 기준으로 청크당 약 150 ms 더 든다. serve_safe 기록기는 `background=True`(T38 B5)라서 응답 경로
  밖에서 쓴다. 128³ 계층 D2H도 worker thread에서 한다(`_DeferredGrid`).

## verifier가 알아야 할 것

1. **막는 문제: Panda server는 `--record-constraints`를 무시한다.** `serve_safe.main`은 `--robot panda`이면
   `serve_safe.py:2244`에서 `AG3S.adapters.panda.server.main`으로 return한다. `ConstraintRecordWriter`를 만드는 코드
   (`serve_safe.py:2287`)는 RB-Y1 분기에만 있다. Panda의 `build_safe_server`는 `SafePolicy(..., recorder=None, ...)`를
   고정으로 넘긴다(`AG3S/adapters/panda/server.py:305`). `profile.py:52`와 `:523`에도 "recording flags are parsed but not used"라고
   적혀 있다. 그래서 launcher의 이 option만으로는 디렉토리조차 생기지 않는다. 이번 launcher는 run이 끝난 뒤
   `n_chunk_files: 0`과 `!!!` 경고로 이것을 드러낼 뿐이다. GPU run 한 번을 쓴 뒤에야 알게 된다.
   → SL5j-G 전에 adapter 작업이 필요하다. `build_safe_server`에서 `args.record_constraints`가 있으면 serve_safe와 같은
   meta로 `ConstraintRecordWriter(..., background=True)`를 만들어 `SafePolicy(recorder=...)`에 넘기고, 종료 시 flush를 보장해야 한다.
   launcher는 SIGINT로 멈추므로 atexit flush는 돌지만 `close()`가 없으면 `meta.json`이 남지 않는다. RB-Y1 T63b 기록에도
   `meta.json`이 없었다. 이것은 이번 범위(adapter 금지) 밖이다.
2. 새 flag: `launch_libero_paired.py --record-constraints {none,occupancy,full}`, 기본 `none`. `none`이 아니면 `--modes vla_v11ab`가 필수다.
3. manifest: `settings.server_argv`는 `<run_dir>` token을 쓰고, 실제 경로는 `executions[role=server].argv`에 있다.
   `none`이면 settings key 집합과 argv가 SL5j 전과 같다.
4. J0(기록이 실행을 바꾸는가)과 관련된 사실: SL5i-G V11AB-pb7 LI calls log에서 `time_budget_ms 50.0`, `time_budget_hit`는
   false 1,561회이고 true는 0회다. `max_iterations_hit`는 true 1,189회, false 372회다. TO 반복 수는 wall-clock이 아니라 반복
   상한이 정했다. adapter 연결 뒤 `SafePolicy.record` 경로가 생기면 그 경로가 응답 내용에 영향을 주는지는 J0이 잰다.
5. 재생산이 필요한 산출물: 없음. 옛 기록과의 호환: 깨지지 않는다(기본값이면 그대로).

## 내가 기대하는 결과

없음. 측정이 아니라 orchestration이다. 다만 위 1이 풀리기 전에 SL5j-G를 돌리면 기록이 비는 것은 확실하다.
