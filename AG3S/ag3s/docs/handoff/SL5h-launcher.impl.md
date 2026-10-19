# SL5h-launcher — 구현

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead · 작성 2026-10-20 03:25 KST (2026-10-19 18:25 UTC)
> base: `feature/panda-benchmark` `8e7ce1f8` (commit 안 함 — lead가 한다)

## 무엇을 했나 (평이한 요약 먼저)

이제 같은 checkout 안에서 launcher 두 개를 하나의 공용 GPU lock 아래 동시에 돌릴 수 있다.
새 script `run_parallel_gpu.sh`가 lock을 한 번 잡고, GPU가 비어 있는지와 pod 메모리를 한 번 확인한 뒤,
`AG3S_GPU_LOCK_HELD=1`로 job을 최대 2개씩 실행한다.
그 env가 있으면 launcher는 lock을 직접 잡지 않고 busy-GPU 대기도 하지 않는다. 대신 "caller가 lock을 실제로 쥐고 있고, 그 lock이 나에게 물려준 fd로 잡혀 있는가"를 확인하고, 아니면 거부한다.
env가 없으면 launcher의 lock 처리는 이전과 같다. 바뀐 것은 들여쓰기와 `if` 두 줄뿐이다.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/runners/launch_libero_paired.py:9-11` | docstring에 held mode 설명 | 사용법 기록 |
| `benchmark/runners/launch_libero_paired.py:39` | `GPU_LOCK_HELD_ENV = "AG3S_GPU_LOCK_HELD"` | env 이름을 한 곳에 둔다 |
| `benchmark/runners/launch_libero_paired.py:107-160` | `GpuLockNotHeld`, `gpu_lock_held_by_caller()`, `_same_file_fds()`, `check_caller_holds_gpu_lock()` | caller가 lock을 쥐고 있는지 확인하는 함수. 확인만 하고 lock은 잡지 않는다 |
| `benchmark/runners/launch_libero_paired.py:200-216` | `run_mode`: held이면 `result.update(check_caller_holds_gpu_lock())`, 아니면 원래 block(blocking `flock` + busy-GPU 대기)을 그대로 실행 | task 1 |
| `benchmark/runners/launch_libero_paired.py:232` | `finally`: `if lock is not None:` 안에서 release(held mode에서는 release하지 않는다) | launcher가 잡지 않은 lock은 풀지 않는다 |
| `benchmark/runners/launch_libero_paired.py:329-334` | `main`: held이고 확인에 실패하면 `refused: …`를 stderr에 출력하고 `return 2`. `versions()`·`create_run`보다 먼저 실행된다 | 거부할 때 run 디렉터리가 남지 않게 한다 |
| `benchmark/runners/run_parallel_gpu.sh` (새 파일, 140줄) | lock → GPU 확인 → 메모리 확인 → job 최대 2개를 병렬 실행 → 로그 | task 2 |
| `tests/benchmark/test_parallel_gpu_launcher.py` (새 파일, 테스트 함수 8개 / 11 case) | mock과 fake로 단위 검증. GPU는 쓰지 않는다 | task 4 |

server argv, client argv, `libero_safe_client.py`, `serve_safe.py`, adapter, profile은 바꾸지 않았다(`git diff --stat`에는 `launch_libero_paired.py` 한 파일만 나온다).

### lock 확인 방법 (`check_caller_holds_gpu_lock`)

1. lock 파일을 새로 열어 `LOCK_EX|LOCK_NB`를 시도한다. **성공하면** lock이 비어 있다는 뜻이다. 그 자리에서 `LOCK_UN`하고 거부한다(`... is not locked`).
2. **task.md보다 강화한 점:** 1만으로는 "다른 checkout이 lock을 쥐고 있는 경우"를 걸러내지 못한다. 그러면 checkout 사이 동시 실행이 일어날 수 있다. 그래서 `/proc/self/fd`에서 같은 inode를 가리키는 **물려받은 fd**를 찾고, 그 fd에 `LOCK_EX|LOCK_NB`를 시도한다. 이 시도는 lock을 쥔 open file description에서만 성공하고(이미 쥔 lock이라 no-op), 다른 description에서는 실패한다. 이런 fd가 없으면 거부한다(`... not through a descriptor inherited by this launcher`). 물려받은 fd에는 절대 `LOCK_UN`하지 않는다.
   - CLAUDE.md가 허용하는 세 형태는 모두 fd를 물려준다: `exec 9>>lock; flock 9`, `( flock -w N 9 …; CMD ) 9>lock`, `flock -w N lock CMD`(`-o` 없이).
   - Python `subprocess`(`close_fds=True`)로 launcher를 띄우면 fd가 넘어가지 않아 거부된다. 의도한 동작이다.
3. 성공하면 `launcher_result`에 다음을 기록한다: `gpu_lock_mode: "held_by_caller"`, `gpu_lock_caller_pid: os.getppid()`, `gpu_lock_fd: <fd>`(script에서는 9). held mode에서는 `gpu_lock_wait_s` · `gpu_lock_acquired_utc` · `gpu_lock_released_utc`를 쓰지 않는다.

`AG3S_GPU_LOCK_HELD` 값: 없음 또는 `0`이면 기존 경로, `1`이면 held mode, 그 밖의 값이면 `ValueError`.

### 사용법

```bash
source scripts/setup/env.sh
cat > /tmp/claude-…/jobs.txt <<'EOF'
# launch_libero_paired.py 인자 한 줄 = job 하나 (shell quoting, shlex로 파싱, eval 안 함)
--task-ids 0 --episode-ids 0,1,2,3,4 --run-id sl5h-t0 --conversion B --ag3s-version V11AB-pb5
--task-ids 3 --episode-ids 0,1,2,3,4 --run-id sl5h-t3 --conversion B --ag3s-version V11AB-pb5
EOF
benchmark/runners/run_parallel_gpu.sh --jobs /tmp/claude-…/jobs.txt \
    [--max-parallel 2] [--est-gib-per-run 12] [--limit-gib 64] [--lock-wait 600] [--stagger-s 30] [--log PATH]
```

- 순서: job 줄을 모두 먼저 파싱한다(실패하면 lock을 잡기 전에 거부) → `exec 9>>lock; flock -w N 9` → `nvidia-smi --query-compute-apps`가 비었는지 확인 → `working set + est × min(max_parallel, job 수) ≤ limit` 확인 → job을 실행한다. 한 job이 끝나면 다음 job을 넣는다(`wait -n -p`). 두 번째 이후 job은 다른 job이 돌고 있으면 `--stagger-s`초 기다린 뒤 시작한다(checkpoint loading 시점을 엇갈리게 한다) → 끝나면 `flock -u 9`.
- `--max-parallel`은 1 또는 2만 받는다(pod 규칙: server ≤ 2).
- 로그: 기본 경로는 `outputs/parallel_gpu/<UTC>_<pid>.log`(gitignored). 줄마다 UTC와 KST가 함께 찍힌다. 각 job의 `start`/`end` 줄에는 pid, rc, 그 시점 working set, job 인자가 들어간다. launcher stdout은 `<log>.job<i>.out`에 저장된다.
- exit code: 0이면 전부 rc=0, 1이면 실패한 job이 있음, 2이면 거부(lock, busy GPU, 메모리, 인자), 130이면 SIGINT/TERM.
- SIGINT/TERM을 받으면 실행 중인 launcher에 SIGINT를 전달한다. launcher의 `finally`가 자기 server를 내린다. 그 뒤 job마다 rc를 기록하고 종료한다. script는 `set -m`을 쓴다. 이것이 없으면 non-interactive bash의 background job은 SIGINT를 무시해서, launcher가 SIGINT를 받지 못한다.
- lock 수명: launcher도 fd 9를 물려받는다. 그래서 script가 SIGKILL로 죽어도 launcher가 살아 있는 동안은 lock이 유지된다. launcher가 띄우는 server와 client는 `Popen` 기본값 `close_fds=True` 때문에 fd 9를 받지 않는다.
- test 전용 override: `AG3S_PARALLEL_TEST_{LOCK,LAUNCHER,NVIDIA_SMI,CGROUP_DIR}`. 실제 사용에서는 설정하지 않는다.

## 단위 검증

```bash
source scripts/setup/env.sh
MUJOCO_GL=osmesa PYTHONPATH=$AG3S_PROJECT_ROOT CUDA_VISIBLE_DEVICES= .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider \
  tests/benchmark/test_parallel_gpu_launcher.py \
  "tests/benchmark/test_safelibero_client.py::test_launcher_records_the_safelibero_protocol" \
  "tests/benchmark/test_safelibero_client.py::test_ag3s_version_flags_reach_the_server_argv"
```

**13 passed in 7.05 s** (새 test 11개 + 기존 launcher test 2개). GPU는 쓰지 않았다. 모든 lock은 `tmp_path` 아래 파일이다. 공용 `/tmp/ag3s-vla-gpu-validation.lock`은 건드리지 않았다.

| test | 확인하는 것 |
|---|---|
| `test_env_unset_takes_own_lock_and_checks_gpu` | env가 없을 때 flock 호출이 정확히 `[LOCK_EX, LOCK_UN]`(blocking)인지, busy-GPU 확인이 실행되는지, result key 순서 `role, mode, port, gpu_lock_wait_s, gpu_lock_acquired_utc`와 `gpu_lock_released_utc`가 유지되는지 |
| `test_env_value_must_be_1_or_unset` | 없음/`0`이면 False, `1`이면 True, 그 밖의 값이면 ValueError |
| `test_held_env_refuses_free_lock_and_leaves_it_free` | lock이 비어 있으면 거부하고, 거부한 뒤에도 lock이 비어 있는지 |
| `test_held_env_refuses_lock_held_by_unrelated_process` | 다른 process가 lock을 쥐고 있고 fd가 이 process에 없으면 거부하는지 |
| `test_held_env_main_refuses_before_allocating_a_run` | `main`이 rc 2와 `refused`를 내고, `create_run`·`versions`가 호출되지 않는지 |
| `test_held_env_with_caller_lock_skips_flock_and_gpu_wait` | 모든 flock 호출에 `LOCK_NB`가 붙어 있고 `LOCK_UN`이 없는지, busy-GPU 확인이 0회인지, `held_by_caller`·`os.getppid()`·fd가 기록되는지, 끝난 뒤에도 caller가 lock을 쥐고 있는지 |
| `test_script_runs_at_most_two_jobs_under_one_held_lock` | 실제 bash script, fake launcher, job 3개(하나는 rc 3): batch rc 1, start·end 각 3줄, quoting 보존, job 3개 모두 `held_by_caller`(fd 상속 경로 end-to-end), 동시 실행 최대값이 정확히 2, 끝난 뒤 lock이 풀리는지 |
| `test_script_refusals_start_no_job[busy_gpu / memory / lock_taken / max_parallel]` | 각 거부 사유에서 rc 2, `REFUSED`가 기록되고 job이 하나도 시작되지 않는지 |

수동 확인(scratchpad, fake launcher):
- 실제 cgroup 값 기준 preflight 줄: `working set 33.70 GiB + 12 x 2 = 57.70 GiB <= 64 GiB`.
- job 안에서 `{'gpu_lock_mode': 'held_by_caller', 'gpu_lock_caller_pid': <script pid>, 'gpu_lock_fd': 9}`.
- SIGINT 전달: 두 job 모두 `KeyboardInterrupt`를 받아 rc 5로 정리됐다. script rc는 130이고, 종료 뒤 lock이 비었다.

## 기본 경로가 그대로라는 근거 (diff 범위)

`git diff -w`(공백 무시) 기준으로 `run_mode`에서 바뀐 것은 아래뿐이다. 원래 lock과 대기 block 본문은 들여쓰기만 바뀌었다.

```diff
-    lock = open(GPU_LOCK, "a")
+    held = gpu_lock_held_by_caller()
+    lock = None if held else open(GPU_LOCK, "a")
     try:
+        if held:
+            # caller (run_parallel_gpu.sh) holds the lock and checked the GPU once before starting its jobs
+            result.update(check_caller_holds_gpu_lock())
+        else:
             t_wait = time.monotonic()      # ← 이하 원래 block 그대로
@@ finally
+        if lock is not None:
             result["gpu_lock_released_utc"] = ...   # ← 원래 3줄 그대로
```

`main`에는 `if gpu_lock_held_by_caller():` block 하나만 추가됐다. env가 없으면 이 block은 건너뛴다.
나머지 hunk는 docstring, 상수 하나, 새 함수뿐이다. env가 없으면 실행 순서, flock 호출 순서, result key와 순서가 이전과 같다. 첫 번째 test가 이를 고정한다.

## verifier 가 알아야 할 것

- 새로 생긴 flag와 기본값 변경: env `AG3S_GPU_LOCK_HELD`(launcher), script option(위 사용법 참고). launcher CLI flag와 기본값은 바뀌지 않았다.
- 재생산이 필요한 산출물: 없음. 실행 경로(server, client argv)가 그대로다.
- 옛 기록과 호환: 깨지지 않는다. held mode run의 `launcher_result`에는 `gpu_lock_wait_s`·`gpu_lock_acquired_utc`·`gpu_lock_released_utc` 대신 `gpu_lock_mode`·`gpu_lock_caller_pid`·`gpu_lock_fd`가 들어간다. lock을 잡은 시각과 푼 시각은 batch log에 있다.
- 알려진 한계:
  - port: 두 launcher가 `l0.choose_port`(20000~29999 무작위 + bind 확인)로 port를 고른다. 그런데 server가 실제로 bind하는 시점은 checkpoint loading 뒤다. 그 사이에 같은 port를 고를 확률이 job 쌍마다 약 1/10⁴ 남는다. 그런 일이 생기면 두 번째 server가 bind에 실패해 그 job이 rc≠0으로 끝난다. 결과가 섞이지는 않는다. `--stagger-s`가 이 확률을 더 줄인다. port 선택 코드는 범위 밖이라 고치지 않았다.
  - 메모리 추정 12 GiB/run은 task.md의 기본값이다. 실측값이 아니다.
  - 병렬 실행이 결과를 bit 단위로 바꾸지 않는다는 것은 이 구현이 보장하지 않는다. task.md대로 첫 병렬 run에서 verifier가 비교해야 한다.

## 내가 기대하는 결과

첫 병렬 run(2 job)의 episode npz와 calls npz가 같은 인자로 직렬 실행한 run과 bit 단위로 같을 것으로 본다. 근거는 task.md의 `time_budget_hit = 0`과 반복 수 상한 종료다. batch log에서 두 job의 시작과 끝 구간이 겹치고, 각 `launcher_result`에 `held_by_caller`가 기록되어 있을 것으로 본다.
