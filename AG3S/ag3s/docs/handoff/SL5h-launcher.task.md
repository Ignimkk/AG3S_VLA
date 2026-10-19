# SL5h-launcher — 같은 checkout 안 병렬 GPU run을 위한 launcher option (orchestration만)

> writer: lead · 읽는 쪽: implementer · 작성 2026-10-19 KST

## 왜

사용자 지시(2026-10-19): GPU(H200 NVL 140 GB)를 적절히 잘 활용한다.

benchmark V11AB 계열 TO는 반복 수 상한으로만 끝난다. 근거는 두 가지다.

- SL5c~SL5g의 약 1만 call에서 `time_budget_hit = 0`이고, `max_iterations = 3`에서 멈춘다.
- 같은 code로 다시 돌린 결과가 bit 단위로 같다. smoke run(`f4f97337`) t0 · t3과 SL5g run(`a8ad92ef`)의 같은 episode가 episode npz와 calls npz 전부에서 같다.

그래서 이 checkout 안에서는 server 2개를 동시에 돌려도 결과가 바뀌지 않을 것으로 본다. 첫 병렬 run에서 bit 단위 비교로 확인한다.

checkout 사이 동시 실행은 계속 배제한다. development 두 세션과 합의했다.

지금은 `benchmark/runners/launch_libero_paired.py`의 `run_mode`가 막고 있다.

- mode마다 `/tmp/ag3s-vla-gpu-validation.lock`을 직접 `fcntl.flock`으로 잡는다.
- 다른 GPU compute process가 있으면 기다린다.

그래서 launcher 두 개를 동시에 띄우면 하나가 기다린다.

## 무엇을 바꾸나 (orchestration만)

1. `launch_libero_paired.py`: env `AG3S_GPU_LOCK_HELD=1`이면 다음처럼 동작한다.
   - **lock 처리:** launcher 자신은 lock을 잡지 않는다. 대신 caller가 lock을 실제로 잡고 있는지 확인한다. 확인 방법은 `fcntl.flock(LOCK_EX | LOCK_NB)` 시도가 실패하는지 보는 것이다. lock이 비어 있으면 거부한다(exit, 이유 출력).
   - **busy-GPU 대기:** 이 모드에서는 하지 않는다. caller가 시작 전에 한 번 확인한다.
   - **기록:** `launcher_result`에 `gpu_lock_mode: "held_by_caller"`와 caller pid(`os.getppid()`)를 기록한다.
   - **기본 동작:** env가 없으면 지금과 byte 단위로 같은 동작이다.
2. 새 script `benchmark/runners/run_parallel_gpu.sh` (또는 `.py`). 동작 순서:
   - 바깥에서 lock을 잡는다: `exec 9>>lock; flock -w <초> 9`
   - `nvidia-smi --query-compute-apps`가 비어 있는지 확인한다. 아니면 거부한다.
   - pod working set(`memory.current − inactive_file`) + 추정(인자, 기본 12 GiB/run × 동시 수)이 64 GiB 이하인지 확인한다. 아니면 거부한다.
   - `AG3S_GPU_LOCK_HELD=1`로 launcher job 최대 2개를 동시에 실행한다. job은 launcher 인자 줄 목록으로 받는다.
   - 하나가 끝나면 다음 job을 넣는다. 동시 실행 수는 ≤ 2다.
   - 모두 끝나면 lock을 푼다.
   - 각 job의 시작 · 끝 · rc · working set을 한 log에 기록한다.
   - **CLAUDE.md 주의:** `flock -w N 9 CMD 9>lock` 형태는 쓰지 않는다. fd 번호 뒤에 command를 붙이면 `9`를 파일 이름으로 해석해 공용 lock을 잡지 못한다.
3. 바꾸지 않는 것: server argv, client argv, `libero_safe_client.py`, `serve_safe.py`, adapter, profile. **실행 경로는 그대로다.**
4. 단위 test(`tests/benchmark/`나 기존 위치):
   - env가 없으면 기존 경로를 탄다.
   - env가 있는데 lock이 비어 있으면 거부한다.
   - env가 있고 lock이 잡혀 있으면 flock을 호출하지 않는다.
   - GPU 실행 없이 mock으로 확인한다.

## 넘겨줄 것

- `AG3S/ag3s/docs/handoff/SL5h-launcher.impl.md`. 포함할 것: 바뀐 file, 사용법 예시, test 결과, 기본 경로가 그대로라는 근거(diff 범위).
- commit은 lead가 한다. 너는 commit하지 않는다.

## 규칙

- 지금 진행 중인 benchmark run은 없다. 다른 verifier가 CPU로 `outputs/verify/SL5h/`를 쓰고 있으니 그 경로는 건드리지 않는다.
- GPU를 쓰지 않는다. CPU worker ≤ 1, pod working set이 빠듯하다(약 60 GiB / 64). 무거운 import나 test 전체 실행은 피하고, 필요한 test만 돌린다.
