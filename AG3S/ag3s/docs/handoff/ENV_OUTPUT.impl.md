# ENV_OUTPUT — 구현

> writer: ag3s-implementer · 2026-10-12 00:35:54 KST · source task: `ENV_OUTPUT.task.md`

## 무엇을 했나

새 실행은 공유 storage의 `results/development/<KST date_commit_UUID>`에 기록하도록 공통 run allocator를 추가했다. Server·client·training process는 `AG3S_RUN_DIR`를 상속하여 같은 실행 아래에 기록한다. 명시적인 writable output이 run 밖을 가리키거나 기존 output subtree의 symlink가 run 밖으로 나가면 실행 전에 거부한다.

Bringup CLI의 video·trajectory·policy input·AG3S record·constraint·frame·trace 경로와 atomic/grid evaluation 기본 경로를 run 안으로 이동했다. Training launcher의 guard/train log·checkpoint 경로와 safe server의 constraint/log를 같은 방식으로 격리했다. AG3S algorithm, simulator dynamics, checkpoint/dataset input, legacy evidence는 수정하지 않았다.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `common/run_output.py:20` | storage boundary·symlink validation | 이전 input 및 run 밖의 output write 차단 |
| `common/run_output.py:57` | KST timestamp·commit·UUID run, manifest, environment reuse | process/restart 간 격리 및 provenance |
| `common/run_output.py:98` | relative/absolute writable path validation | cwd의 historical `policy_records` 등으로 잘못 기록하는 문제 제거 |
| `common/run_output.py:116` | unique persistent logging, run-local `WANDB_DIR` | log restart overwrite 및 inherited old log destination 방지 |
| `integration/rby1/bringup/pi05_infer.py:940` | `--run-dir`, writable flag routing, logging | rollout artifact들을 선택 run에 격리 |
| `integration/rby1/bringup/pi05_ex_infer.py:1109` | `--run-dir`, atomic/grid evaluation 및 video routing | relative evaluation default를 공유 input 밖으로 이동 |
| `AG3S/trajopt/serve_safe.py:2425` | `--run-dir`, `--record-constraints`, persistent logging | server record/log 격리 |
| `scripts/rby1/launch_train_after_norm.sh:8` | run allocation 및 tmux inheritance | guard/train log·checkpoint를 run 아래 저장 |
| `scripts/rby1/launch_atomic_train_after_norm.sh:8` | 같은 training output isolation | atomic training의 기존 output default 제거 |
| `tests/test_run_output.py:1` | 12개 CPU storage-contract test | 입력 보존, restart/multiprocess, output/symlink escape 검증 |

## 단위 검증

```bash
MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/AG3S_VLA-development \
  .venv-ag3s/bin/python -m pytest tests/test_run_output.py \
  tests/trajopt/test_safe_policy.py tests/trajopt/test_no_collision_ablation.py \
  tests/trajopt/test_no_limits.py tests/trajopt/test_subtask_wiring.py -q
```

**88 passed**, 94 existing OSQP deprecation warnings, 9.79 s. Routing contract tests는 12개다. 초기 CPU interpreter에 `tzdata`가 없어 `ZoneInfo` 사용이 실패했고, 고정 KST UTC+09 offset을 stdlib만으로 사용하도록 수정한 뒤 전부 통과했다.

```bash
MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/AG3S_VLA-development \
  .venv-ag3s/bin/python -m py_compile common/run_output.py \
  integration/rby1/bringup/pi05_infer.py integration/rby1/bringup/pi05_ex_infer.py \
  AG3S/trajopt/serve_safe.py
bash -n scripts/rby1/launch_train_after_norm.sh scripts/rby1/launch_atomic_train_after_norm.sh
git diff --check
```

Python/shell syntax 및 whitespace 정상. GPU 실행·다른 checkout 실행·기존 process 종료는 수행하지 않았다.

Final review에서 training tmux child가 이전 tmux server의 `PYTHONPATH` 및 cache environment를 상속할 수 있음을 확인했다. 두 launcher의 child command에서 자기 checkout의 `scripts/setup/env.sh`를 다시 source한 뒤 명시적인 run/storage/training override를 export하도록 보완했다. 이 shell-only 보완은 `bash -n`과 `git diff --check`로 확인했으며 training을 실행하거나 CPU 전체 suite를 반복하지 않았다.

## verifier 가 알아야 할 것

- 새 flag: bringup CLIs 및 `AG3S.trajopt.serve_safe`에 `--run-dir`. 생략 시 `AG3S_RUN_DIR` 또는 새 run 사용.
- Shell: `export AG3S_RUN_DIR="$(.venv-openpi-live/bin/python -m common.run_output)"` 후 server/client를 실행하면 같은 run 공유. `unset AG3S_RUN_DIR` 후 다음 실행은 새 run.
- 명시적인 relative writable argument는 cwd 대신 run 기준이다. Absolute output은 선택 run 안에서만 허용한다.
- Atomic/grid resume는 기존 run을 `--run-dir`로 다시 선택한다. 과거 recording은 input으로 보존하며 그 경로에서 resume/write하지 않는다.
- Logging helper는 Python `logging` record를 persist하며 console handler를 유지한다. Bringup의 기존 `print()` stdout은 console에 남는다. Full console transcript가 필요하면 run/logs 아래에서 shell `tee`로 저장한다. Training launcher는 이미 stdout/stderr를 run-local `tee`로 저장한다.
- OpenPI `PolicyRecorder`의 unique session 및 `step_0` 보호, training configuration default는 lead 소유 파일이며 이 handoff의 unit test 범위와 구분한다.
- Symlink 검사는 현재 filesystem 상태의 run/root/output subtree를 검사한다. 공격적인 concurrent symlink 교체를 막는 syscall sandbox는 아니다.
- 재생산해야 하는 GPU/checkpoint artifact 없음. 기존 history 및 recording format을 변경하지 않았다.

## 내가 기대하는 결과

Verifier가 새 run 아래에서만 writable artifact가 생성되는 것과 existing recording의 내용 불변을 독립적으로 확인한다. `AG3S_RUN_DIR`를 상속한 child process는 동일 manifest를 유지하고, 상속 없이 시작한 다음 run은 새로운 UUID 경로를 사용한다. 현재 algorithm 관련 기존 CPU regression 결과가 유지된다.
