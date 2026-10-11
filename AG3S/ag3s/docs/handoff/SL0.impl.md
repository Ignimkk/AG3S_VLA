# SL0 — 구현

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead · 작성 2026-10-12 04:40 KST
> branch `feature/panda-benchmark`, HEAD `2ee68eed` (commit 안 함 — lead 가 commit)

## 무엇을 했나 (평이한 요약 먼저)

이 checkout 안에 공식 LIBERO client 와 같은 **Python 3.8.20 `.venv-libero`** 를 만들고(OpenPI 예제 pin 그대로),
공식 `examples/libero/main.py` 와 같은 동작의 client 와, run 생성 → 빈 port 선택 → storage checkpoint 로 OpenPI server
기동 → client 실행 → 자기 server 만 종료하는 launcher 를 만들었다. Policy 없이 할 수 있는 확인(CPU unit test 16개,
`.venv-libero` 에서 libero_spatial task 0 env 생성 + dummy 10 step smoke, 상수 action 을 돌려주는 stub server 로 client 기록
경로 1회)까지 했다. GPU server 와 실제 pi0.5 episode 는 실행하지 않았다.

구현 중 공식 예제 그대로는 이 host 에서 돌지 않는 원인 세 가지를 찾아 client/launcher 쪽에서 막았다:
(1) 이 fork 의 `openpi-client` 가 `ping_timeout` 을 websockets 에 넘기는데, Python 3.8 에서 쓸 수 있는 마지막 websockets 13.1 에는
그 인자가 없어 연결 시 `TypeError` 가 난다. (2) pod 의 전역 `LD_LIBRARY_PATH` 에 `/usr/local/lib/python3.10/dist-packages/torch/lib`
가 있어 venv 의 torch 1.11 이 다른 `libtorch_python.so` 를 읽고 import 가 실패한다. (3) LIBERO 의 최상위 `libero/` 에
`__init__.py` 가 없어 일반 editable 설치로는 `import libero` 가 되지 않는다 (공식 README 는 이 때문에 `PYTHONPATH` 도 export 한다).

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `scripts/setup/create_libero_env.sh` (새 파일) | 자기 `.python` 에 3.8.20, 자기 `.cache/uv`, `--link-mode copy`. `uv pip sync <예제 requirements.txt> <LIBERO requirements.txt> --extra-index-url …/cu113 --index-strategy=unsafe-best-match` (공식 README "Without Docker" 와 같음) → `-e openpi-client` → `-e benchmark/LIBERO` (`editable_mode=compat`, `:39`) | task 1. compat 은 공식 README 의 `PYTHONPATH=<LIBERO>` 와 같은 디렉터리를 `.pth` 로 넣어 venv 자체로 import 가 되게 함 |
| `create_libero_env.sh:48` | `$LIBERO_CONFIG_PATH/config.yaml` 을 이 checkout 의 `benchmark/LIBERO/libero/libero/{bddl_files,init_files,assets}` 로 생성. `datasets` 는 `$AG3S_STORAGE_ROOT/datasets/libero` (존재하지 않음, 만들지 않음) | 첫 import 의 `input()` 질문 방지 |
| `create_libero_env.sh:68-85` | freeze 기록, `uv pip sync --dry-run` 재현 확인(no-op 아니면 exit 3), `pip check` 출력 그대로 표시 | task 1 dry-run 재현 · 충돌 숨기지 않기 |
| `environments/libero/requirements.txt`, `libero-requirements.txt`, `validated-freeze.txt` (새 파일) | 공식 두 requirements 의 byte 사본 + 설치 결과 freeze(67개, editable 제외) | 재현 기록. 사본 일치는 test 로 확인 |
| `benchmark/runners/libero_pi05_client.py` (새 파일, py3.8) | 공식 loop (`run_episode` `:202`), `_get_libero_env`·`_quat2axisangle` 은 공식과 AST 동일. `--run-dir` 규칙 `:64`, LIBERO config guard `:359`, py3.8 websockets compat `:98`, `episodes.jsonl` exclusive open `:378`, npz/mp4 exclusive publish `:81` | task 2 |
| `benchmark/runners/launch_libero_pi05_only.py` (새 파일, py3.11) | port 선택 `:74` (8123 금지, bind 확인), checkpoint identity `:114` (aggregate 재계산 `:104`, norm_stats SHA, 파일 크기, `.partial` 거부, `--verify-checksums` 로 전체 rehash), server/client argv `:196`, client env `:226` (PYTHONPATH 제거, foreign site/dist-packages `LD_LIBRARY_PATH` 제거 `:209`, thread cap 4), settings `:263`, 실행 `:327`, 자기 PID 만 종료 `:314`, `--dry-run` `:402` | task 3 |
| `tests/benchmark/__init__.py`, `tests/benchmark/test_libero_pi05_sl0.py` (새 파일) | 16 test | task 4 |

손대지 않은 것: `third_party/openpi/**` (serve_policy · openpi-client 포함), `benchmark/LIBERO` (git status clean, `libero.egg-info` 는 LIBERO 의 .gitignore 대상), `AG3S/**`, storage 기존 input.

### 공식 예제(`third_party/openpi/examples/libero/main.py`)와의 차이 목록

| # | 차이 | 동작 영향 |
|---|---|---|
| 1 | `--task-ids`/`--episode-ids` 로 부분집합 실행 (공식: 모든 task × `range(num_trials_per_task)`) | task 0 episode 0 은 공식 sweep 의 첫 episode 와 같은 순서(`np.random.seed(7)` → suite → env 생성·`env.seed(7)` → reset → `set_init_state(init[0])`). 부분집합의 뒤쪽 episode 는 같은 env 에서 앞선 episode 를 거치지 않으므로 전체 sweep 의 같은 번호 episode 와 이력이 다를 수 있음 |
| 2 | Exception: 공식처럼 잡고 episode 를 끝내되 `status=execution_error` + traceback 기록, client exit code 1 | 공식은 예외 시 이전 episode 의 `done` 이 남아 있으면 그 값으로 성공/실패를 쓴다. 여기서는 episode 시작 시 `done=False` |
| 3 | 영상 경로/이름 `videos/<suite>_t<task>_e<ep>_<success|failure>.mp4` (공식 `rollout_<task 문장>_<suffix>.mp4`, 같은 task 의 다음 episode 가 덮어씀). 임시 파일 → `os.link` 로 exclusive publish | 프레임·fps(10) 동일 |
| 4 | 추가 기록: `episodes.jsonl`, step 별 npz (raw obs 는 `_quat2axisangle` 의 in-place clip 전에 복사), `client.infer` 왕복 시간(`perf_counter`), 응답의 `server_timing`/`policy_timing` | 정책 입력·실행 action 불변 |
| 5 | 무거운 import(libero, openpi_client, imageio)를 함수 안으로 이동 | 없음 (CPU test 가 3.11 에서 module 을 import 하기 위함) |
| 6 | CLI: tyro `--args.x` → argparse `--x`; `--port` 필수(공식 기본 8000) ; launcher 는 `--host 127.0.0.1` 전달 (client 기본값은 공식과 같은 `0.0.0.0`) | 없음 |
| 7 | websockets 13.1(py3.8)에서는 `ping_timeout` 없이 연결하는 subclass 사용 (`make_policy_client`), jsonl 의 `client_connector` 에 기록 | upstream OpenPI client 와 같은 연결 (13.x sync client 는 keepalive ping 을 보내지 않음) |
| 8 | LIBERO import 전 `LIBERO_CONFIG_PATH/config.yaml` 없으면 거부 | 없음 (config 없으면 LIBERO 가 `input()` 으로 멈추고 `~/.libero` 를 만든다 — 실제로 한 번 겪어 내가 만든 빈 `~/.libero` 를 지웠다) |
| 9 | 환경: `MUJOCO_GL=osmesa` (공식 Docker 기본 `egl`, README 대안 `glx`), PYTHONPATH 제거, foreign torch `LD_LIBRARY_PATH` 제거, `OMP/MKL/OPENBLAS/NUMBA/LP_NUM_THREADS=4` (이미 설정돼 있으면 유지) | 렌더러가 공식 GPU EGL 이 아니라 llvmpipe. 픽셀 단위 차이 가능성은 측정하지 않았다. thread cap 유무로 stub episode 2회의 npz 14개 배열이 모두 bit-identical |
| 10 | 설치: uv 0.12.13 (공식 Docker 는 uv 0.5.1), LIBERO editable compat 모드 (공식 README 는 editable + PYTHONPATH) | 버전 pin 은 예제 그대로 |

## 단위 검증

```bash
cd /mnt/dev/AG3S_VLA-benchmark && source scripts/setup/env.sh
.venv-ag3s/bin/python -m pytest -q -p no:cacheprovider tests/benchmark
# → 16 passed in 0.79s
.venv-ag3s/bin/python -m pytest -q -p no:cacheprovider tests/benchmark tests/test_run_artifacts.py tests/test_policy_recorder_destination.py -rs
# → 20 passed, 1 skipped (test_policy_recorder_destination: .venv-ag3s 에 flax 없음 — 기존과 같은 skip)
bash scripts/setup/create_libero_env.sh --check
# → "Would make no changes" / "[create_libero_env] freeze reproduction: no changes"
```

| test | 무엇을 |
|---|---|
| `test_protocol_constants_match_official` | 예제 source AST: dummy action, 256, host, 224, 5, libero_spatial, 10, seed 7, suite별 max_steps, fps 10 |
| `test_quat2axisangle_source_and_values_match_official` | 함수 AST 동일(docstring 제외) + 예제 함수를 exec 해 68개 quat(단위 4원수, w=1, w>1, w<-1)에서 값·in-place clip 동일 |
| `test_get_libero_env_matches_official_apart_from_deferred_imports` | import 줄 제외 AST 동일 |
| `test_run_dir_rejection_rules` 외 1 | results root 자체 · manifest 없음 · 밖 · symlink 탈출 · storage input · 없는 경로 · `..` 거부, `--run-dir` 필수 |
| `test_libero_config_required`, `test_exclusive_publish_never_overwrites` | config guard, 기존 파일 보존 |
| `test_run_episode_status_and_counts` ×3 | 가짜 env/policy 로 success / max_steps / execution_error 분류, step·호출 수, 실행 action = chunk 의 앞 5개, 224 uint8 image, 8D state |
| `test_port_selection_excludes_forbidden_and_busy` | 8123 거부, 사용 중 port 거부·건너뜀 |
| `test_aggregate_uses_path_order_of_producer`, `test_checkpoint_identity_rules` | aggregate 규칙, 크기/내용(rehash)/aggregate 변조, `.partial` 존재, CHECKSUMS 없음 |
| `test_dry_run_commands_and_settings` | server argv `--port P --record policy:checkpoint --policy.config pi05_libero --policy.dir <storage ckpt>`, `--env`·`gs://` 없음, client argv 가 client parser 에 그대로 통과, settings 필드, run 미생성 |
| `test_client_environment_…`, `test_recorded_requirements_are_the_official_copies` | env 정리, requirements 사본 byte 일치 |

**`.venv-libero` smoke (policy 없음)**: `env -u PYTHONPATH LD_LIBRARY_PATH=<정리값> MUJOCO_GL=osmesa .venv-libero/bin/python smoke.py`
→ exit 0, 12.7 s. Python 3.8.20 (`.venv-libero` → `.python/cpython-3.8.20`), robosuite/mujoco/numpy 모두 `.venv-libero`, openpi_client 는 이 checkout source.
libero_spatial 10 task, task 0 = "pick up the black bowl between the plate and the ramekin and place it on the plate", init state 50개(92D).
`set_init_state(init[0])` 후 dummy 10 step: agentview/wrist 256×256×3 uint8 (평균 120.2 / 124.5 — 빈 화면 아님), `robot0_joint_pos` 7D 존재,
8D state `[-0.2112, -0.0111, 1.1742, 3.1409, -0.0028, -0.0877, 0.0387, -0.0387]`, reward 0, done False(np.bool_), 224 전처리 uint8. `PYOPENGL_PLATFORM=osmesa`.

**client 기록 경로 확인 (stub, GPU·policy 없음)**: 매 호출 상수 dummy chunk(10×7)를 돌려주는 websocket stub 과 scratchpad 의 임시 storage 로
client 를 launcher 가 만든 argv/env 그대로 실행 → exit 0, 72 s, `status=max_steps`, step 230(대기 10 + 220), 호출 44, frame 220,
npz 14 배열(obs 231개, action 230×7, chunks 44×10×7), mp4 1개, 같은 run 에 재실행 시 `episodes.jsonl` 이 있어 거부(내용 불변). 결과는 storage 에 남기지 않았다.

**pip check (숨기지 않음)**: `Found 23 incompatibilities` — 모두 공식 방식(`uv pip sync` 는 의존성을 풀지 않음)으로 LIBERO requirements 의
transitive 의존성이 설치되지 않은 것: gym→gym-notices; hydra-core→omegaconf, antlr4-python3-runtime; robomimic→h5py, tensorboard, tensorboardx, egl-probe;
transformers→filelock, huggingface-hub, regex, tokenizers; wandb→click, gitpython, promise, shortuuid, sentry-sdk, docker-pycreds, protobuf, pathtools, setproctitle;
bddl→pytest, networkx, jupytext. 평가 경로(benchmark → envs → OffScreenRenderEnv → step)는 이것들 없이 import·실행됐다(smoke, stub).

## verifier 가 알아야 할 것

- **실행 명령 (task 0 episode 0)** — GPU 가 비어 있는지 먼저 확인:
  ```bash
  cd /mnt/dev/AG3S_VLA-benchmark && source scripts/setup/env.sh
  .venv-ag3s/bin/python benchmark/runners/launch_libero_pi05_only.py \
      --task-suite-name libero_spatial --task-ids 0 --episode-ids 0 --seed 7 --run-id sl0-pi05only-t0e0
  ```
  stdout 첫 줄이 run 경로. `--dry-run` 을 붙이면 아무것도 만들지 않고 두 argv 와 settings 만 출력. `--verify-checksums` 를 붙이면 시작 전 12 GB 전체 rehash.
  `--port` 를 주지 않으면 20000-29999 에서 빈 port 를 고른다. Server ready 대기 상한 `--server-timeout 1800` s.
- run 아래 생기는 것: `run_manifest.json` (settings + executions: launcher 가 `role=server`/`client`/`launcher_result`, serve_policy 자신도 1행),
  `logs/server_console.log`, `logs/serve_policy_<P>.log`, `logs/libero_client.log`, `policy_records/step_*.npy`, `episodes.jsonl`,
  `episodes/libero_spatial_t0_e0.npz`, `videos/libero_spatial_t0_e0_<success|failure>.mp4`, `source.patch`. client cwd 가 run 이므로 MuJoCo 가 `MUJOCO_LOG.TXT` 를 쓰면 run 안에 생긴다.
- server load 시간 = `launcher_result.server_ready_seconds` (Popen → `/healthz` 200). 호출별 latency 는 `episodes.jsonl` 의 `policy_latency_ms` (첫 값이 JIT 포함), server 측 값은 `server_timing`.
- manifest `settings.commands.client` 의 `--run-dir` 은 run 생성 전이라 `<run>` 이다. 실제 argv 는 `executions[role=client].argv`.
- **CHECKSUMS.json 의 `aggregate_rule` 문구와 실제 계산이 다르다**: 문구는 "sorted '<sha256>  <path>\n' lines" 이지만 producer 는 record 를 **path 순**으로 정렬한다.
  line 문자열 정렬로 재계산하면 `baffa4ca…` 가 나와 불일치, path 순이면 기록값 `1cb098d8…` 과 일치. launcher 는 path 순 (`launch_libero_pi05_only.py:104`).
  현재 checkpoint: 16 파일, 12,439,085,481 bytes, 크기 모두 일치, norm_stats SHA-256 `b3a44bb2…60bd84` = 기록값 (전체 rehash 는 하지 않았다).
- 새로 생긴 플래그·기본값 변경: 기존 코드 변경 없음. 새 script 의 플래그만 (위 참조).
- 재생산이 필요한 산출물: 없음.
- 옛 기록과 호환이 깨지는가: 없음 (새 파일만).
- 환경 위치: `.python/cpython-3.8.20-linux-x86_64-gnu`, `.venv-libero`, `.cache/uv`, `.runtime/libero/config.yaml` — 모두 이 checkout (gitignore 대상).
  `/mnt/dev/work` 경로는 쓰지 않았다. 재현: `bash scripts/setup/create_libero_env.sh` (재실행 시 sync 가 editable 을 지웠다 다시 넣고 같은 상태로 수렴), 확인만은 `--check`.

## 내가 기대하는 결과

<!-- verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다 -->
- 실제 episode: `status` 는 `success` 또는 `max_steps` (execution_error 아님). 공개 수치(Spatial 98.8%)로 보면 success 가능성이 높지만 1 episode 로는 아무 말도 못 한다.
  렌더러가 osmesa(llvmpipe)인 점이 공식 EGL 대비 어떤 영향을 주는지는 모른다.
- 호출 수는 성공 시 대략 (실행 step/5) 올림, 실패 시 44. 첫 호출은 JIT 로 수십 s 단위일 수 있음. client 쪽 sim+렌더만으로 episode 당 약 70 s (stub 측정).
- server load(ready)는 수 분 이내.
