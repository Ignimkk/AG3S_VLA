# AG3S Benchmark — V11AB Panda / LIBERO 계열 평가 기록

> **이 파일의 용도.** `feature/panda-benchmark` 에서 수행하는 **V11AB(paired v2) Panda benchmark 평가**의
> 구현과 검증 기록이다. 단계는 SL-0 부터 이어 쓴다. 2026-10-12 사용자 지시로 Panda/LIBERO 계열 기록은
> 이 파일이 정본이다.
>
> RB-Y1 16D 모델의 T0→T6 기록은 [`AG3S_T0T6_LOG.md`](AG3S_T0T6_LOG.md) 가 그대로 맡는다. 두 파일은
> 대상 로봇·환경·checkpoint 가 달라 서로의 숫자를 대신하지 않는다.
>
> **계획 정본** — [`doc/develope_PLAN_SUMMARY.md`](../../../doc/develope_PLAN_SUMMARY.md)
> (§0 한눈에 · §2.1 SL-0~SL-8 · §7 Phase 순서와 통과 기준).
>
> **수치 규칙.** 이 파일의 모든 수치는 `AG3S/ag3s/docs/handoff/<STEP>.verify.json` 의 `numbers` 를 출처로 한다.
> 출처가 없는 숫자는 쓰지 않고, 필요하면 verifier 에게 측정을 요청한다. 식별자(경로 · SHA · commit)는
> verify.json 또는 impl.md 에 있는 것만 옮긴다.

---

## 진행 현황

실행 경로는 plan §0 의 다섯 단계다: **Panda adapter → LIBERO Spatial 소규모 → SafeLIBERO Spatial I/II pilot →
전체 SafeLIBERO → LIBERO-Safety**.

| Phase (plan §7) | 범위 | 상태 | 비고 |
|---|---|---|---|
| **P1** Panda adapter | SL-0 pi0.5-only 1 episode, 독립 환경 · checkpoint · run 기록 | **완료 (2026-10-12, 사용자 확정)** | 아래 [SL0](#sl0--pi05-only-libero-spatial-1-episode-2026-10-12) |
| | SL-1 agent-view · wrist RGB/depth, intrinsics/extrinsics → AG3S 관측 형식 | **다음** | |
| | SL-2 ~ SL-8 (Panda geometry, attention read-out, ATS/LTE, EE delta ↔ CCR, phase, contact evaluator, runner) | 미착수 | P1 통과 기준에는 pi0.5-only 뿐 아니라 V11AB 최소 1 episode 도 포함 |
| **P2** LIBERO Spatial 소규모 평가 | 고정 task/seed paired 비교 | 미착수 | SL0 의 1 episode 로는 성능을 말할 수 없다 |
| **P3** SafeLIBERO Spatial Level I/II pilot | contact 판정 검증 + paired 결과 | 미착수 | |
| **P4** 전체 SafeLIBERO | official I/II 전체 | 미착수 | |
| **P5** LIBERO-Safety 추가 평가 | 독립 protocol · adapter | 미착수 | |

---

## 용어

기록에 쓰는 말을 여기 모은다. **새 용어를 쓸 때는 여기 추가한다.**

| 용어 | 뜻 |
|---|---|
| **V11AB** | 평가 대상으로 고정한 method. 원본 benchmark v2 와 pi05_TO_hybrid v2 를 통합한 paired-v2 기반 설정(plan §0.1). 개선 branch 의 후속 algorithm 은 섞지 않는다 |
| **pi0.5-only** | VLA(pi0.5)만 쓰고 AG3S safety layer 를 붙이지 않은 조건. V11AB 와 비교할 **기준 경로** |
| **frozen pi0.5 LIBERO checkpoint** | 이번 평가에서 다시 학습하지 않는 `pi05_libero` weight. RB-Y1 checkpoint 를 Panda 에 쓰지 않는다 |
| **official success** | 공식 LIBERO 예제와 같은 판정: `env.step()` 이 돌려주는 `done` 이 True 가 되면 성공. AG3S 가 따로 정의한 metric 과 구분한다 |
| **status** | episode 종료 사유. `success`(done) · `max_steps`(상한까지 done 없음) · `execution_error`(예외) 셋 중 하나 |
| **execution_error** | client 가 예외를 잡고 episode 를 끝낸 경우. 성공·실패로 조용히 섞지 않고 별도 status 와 traceback 으로 기록한다. 이후 단계의 오류 · UNKNOWN 집계가 이 구분에 기대므로 처음부터 분리했다 |
| **action chunk / replan** | policy 가 한 번의 호출로 돌려주는 연속 action 묶음. 이번 protocol 에서는 chunk 전체가 아니라 앞부분 `replan_steps` 개만 실행하고 다시 policy 를 호출한다 |
| **wait steps / dummy action** | episode 시작 직후 물체가 안정되도록 policy 없이 넣는 10 step. 고정 action `[0]*6 + [-1]` (gripper open) |
| **init state** | LIBERO 가 task 마다 제공하는 초기 물리 상태 목록. episode 번호가 그 인덱스를 고른다 |
| **OSC_POSE** | LIBERO 의 EE delta controller. action 앞 6 차원이 EE 위치·회전 delta, 7번째가 gripper 다 |
| **run dir / run manifest** | 실행마다 `results/benchmark/<시각_commit_run-id>/` 에 새로 만드는 고유 디렉터리와, 그 안의 `run_manifest.json`(설정 · 환경 · 실행 argv · checkpoint 식별). 기존 기록을 덮어쓰지 않는다 |
| **norm stats** | policy 입력/출력 정규화 통계(`norm_stats.json`). checkpoint 와 짝이 맞아야 하므로 SHA-256 을 함께 고정한다 |
| **aggregate SHA** | `CHECKSUMS.json` 의 파일별 SHA-256 들을 하나로 묶은 값. checkpoint 전체의 동일성 확인에 쓴다 |
| **JIT first call** | JAX 가 첫 호출에서 model 을 compile 하는 시간. steady-state latency 가 아니므로 나머지 호출과 분리해 보고한다 |
| **policy_records** | OpenPI server 가 호출마다 저장하는 입출력 기록(`policy_records/step_*.npy`) |
| **osmesa / EGL** | MuJoCo offscreen 렌더러. 공식 Docker 기본은 GPU EGL, 이번 실행은 CPU software 렌더링(osmesa) |
| **pod working set** | pod 의 `memory.current` 에서 inactive file cache 를 뺀 값. pod 메모리 압박의 실제 지표 |
| **paired 비교** | 같은 task · init state · seed · checkpoint · controller · horizon 에서 조건만 바꿔 비교하는 방식 |

---

## 고정 기준

SL0 에서 확정했고 이후 단계가 바꾸지 않는다. 바뀌면 그 STEP 이 사유를 기록한다.

| 항목 | 값 |
|---|---|
| branch / 기준 commit | `feature/panda-benchmark` / SL0 구현 `6ba81024`, 검증 `696abd52` |
| checkpoint | `/mnt/dev/AG3S_VLA-storage/checkpoints/panda/pi05_libero` (read-only storage input, 공개 GCS 에서 받음, OpenPI config `pi05_libero`) |
| checkpoint aggregate SHA-256 | `1cb098d84bde409fb4b3931ed676195b80b0755c6ee60177e3d6c96508f1d891` |
| norm stats SHA-256 | `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84` (`assets/physical-intelligence/libero/norm_stats.json`) |
| LIBERO client 환경 | `.venv-libero`, **Python 3.8.20**, OpenPI 예제 `requirements.txt` pin 그대로 (robosuite 1.4.1, mujoco 3.2.3, numpy 1.22.4, torch 1.11.0+cu113). 재현 기록: `environments/libero/` |
| policy server 환경 | `.venv-openpi-live`, Python 3.11.16, GPU |
| launcher 환경 | `.venv-ag3s`, Python 3.11.16 (CPU) |
| LIBERO source | submodule `benchmark/LIBERO` @ `f78abd68ee28` (수정 금지) |
| client 기준 | `third_party/openpi/examples/libero/main.py` |
| protocol 상수 | suite `libero_spatial`, seed 7, `num_steps_wait` 10, `replan_steps` 5, `resize_size` 224, render 256, `max_steps` 220 (Spatial), dummy action `[0]*6+[-1]`, video 10 fps |
| 판정 | official success = `env.step()` 의 `done` |
| 렌더러 | `MUJOCO_GL=osmesa` |

사용자 판정 (2026-10-12): `pi05_libero` 는 공개 GCS 에서 storage 의 새 input 으로 받는다. LIBERO client 는 공식
protocol 과 같은 별도 Python 3.8 `.venv-libero` 를 이 checkout 안에 만든다. 전역 `python3.11` shim 은 건드리지 않는다.

---

## SL0 — pi0.5-only LIBERO Spatial 1 episode (2026-10-12)

> 2026-10-12 KST. 구현 commit `6ba81024`(KST 04:40) · 검증 commit `696abd52`(KST 05:00). 사용자 확정.
> 원본: [`handoff/SL0.task.md`](handoff/SL0.task.md) · [`handoff/SL0.impl.md`](handoff/SL0.impl.md) ·
> [`handoff/SL0.verify.json`](handoff/SL0.verify.json). raw 출력은 `outputs/verify/SL0/`.

### 이 STEP 이 답하는 질문

이 checkout 만으로(이전 workspace 없이) frozen pi0.5 LIBERO checkpoint 가 공식 LIBERO Spatial protocol 의
1 episode 를 공식 판정까지 완주하고, 그 run 이 입력을 건드리지 않고 고유 디렉터리에 재현 가능하게 기록되는가.

```
LIBERO sim (.venv-libero, py3.8) --websocket--> OpenPI policy server (.venv-openpi-live, GPU)
   obs: agentview/wrist RGB(180 deg rotation), eef pos + axis-angle + gripper qpos, prompt
   <-- action chunk (EE delta 6 + gripper 1), replan_steps 만 실행 후 다시 호출
   official success = env.step() 의 done
```

### 무엇을 만들었나

공식 LIBERO client 와 같은 동작의 client 와, run 생성 → 빈 port 선택 → storage checkpoint 로 server 기동 →
client 실행 → **자기가 띄운 server 만** 종료하는 launcher 를 만들었다. AG3S 코드와 `third_party/openpi`,
`benchmark/LIBERO` 는 수정하지 않았다.

| 파일 | 역할 |
|---|---|
| `scripts/setup/create_libero_env.sh` | 자기 `.python` 에 Python 3.8.20 을 설치하고 `.venv-libero` 생성. 공식 두 requirements 를 `uv pip sync` 로 설치, freeze 기록과 재현 확인(no-op 여부) |
| `environments/libero/{requirements,libero-requirements,validated-freeze}.txt` | 공식 requirements 의 byte 사본과 설치 결과 freeze |
| `benchmark/runners/libero_pi05_client.py` | Python 3.8 client. 공식 loop 와 같은 동작에 `episodes.jsonl` · step 별 npz · mp4 를 exclusive create 로 기록 |
| `benchmark/runners/launch_libero_pi05_only.py` | launcher. port 선택(8123 금지, bind 확인), checkpoint identity 확인, server/client 실행과 manifest 기록, `--dry-run` |
| `tests/benchmark/test_libero_pi05_sl0.py` | CPU unit test (공식 예제와 protocol 상수 · `_quat2axisangle` 일치, `--run-dir` 거부 규칙, port 선택, launcher command 구성 등) |

`.venv-libero` 는 Python 3.8 용으로 구현 중 세 가지 장애를 client/launcher 쪽에서 막았다.
(1) 이 fork 의 `openpi-client` 가 보내는 `ping_timeout` 을 Python 3.8 에서 쓸 수 있는 마지막 websockets 가 받지 못해 연결 시 `TypeError` 가 난다.
(2) pod 의 전역 `LD_LIBRARY_PATH` 가 다른 Python 의 `torch/lib` 를 가리켜 venv 의 torch 가 import 되지 않는다.
(3) LIBERO 최상위 `libero/` 에 `__init__.py` 가 없어 일반 editable 설치로는 import 되지 않는다.

### 실행

| 항목 | 값 |
|---|---|
| 명령 | `.venv-ag3s/bin/python benchmark/runners/launch_libero_pi05_only.py --task-suite-name libero_spatial --task-ids 0 --episode-ids 0 --seed 7 --run-id sl0-pi05only-t0e0` |
| run dir | `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261012T045129_6ba81024d243_sl0-pi05only-t0e0` |
| task | libero_spatial task 0, episode 0 (init state 0): "pick up the black bowl between the plate and the ramekin and place it on the plate" |
| server port | 24038 |
| client / server returncode | 0 / -2 (launcher 가 보낸 종료 signal. `server_console.log` 의 `KeyboardInterrupt` traceback 은 이 종료다) |

### 결과 — Figure

세 figure 는 verifier 가 만들었다 (`outputs/verify/SL0/scripts/make_figures.py`, sidecar `.json` 동봉).

**실제 scene** — agentview · wrist 원본 frame 의 시작 / 첫 policy 호출 / 중간 / 끝 montage (raw 256×256, 공식 180° 회전 전).

![SL0 scene](figures/sl0-scene.png)

**graph** — EEF xyz · gripper qpos · 실행 action · gripper 명령의 step 별 시계열(회색 = wait 10 step, 점선 = done step)과
호출별 latency. 첫 호출(JIT compile 포함)은 steady-state 와 자릿수가 달라 막대로 따로 그렸다.

![SL0 trend](figures/sl0-trend.png)

**table** — protocol · checkpoint · 환경 · 결과 · 입력 보호 · 회귀를 한 장에 모은 표.

![SL0 table](figures/sl0-table.png)

### 결과 — 수치

> **episode 1개의 사실이다. 성공률이 아니다.** task 0 · init state 0 · seed 7 에서 한 번 성공했다는 것 이상은
> 말하지 않는다. 이 episode 로 pi0.5 의 성능이나 공개 수치와의 일치 여부를 판정하지 않는다
> (성능 확인은 P2 소규모 평가의 일이다).

**Episode**

| 항목 | 값 |
|---|---|
| official_success | `true` |
| status | `success` (error 없음) |
| env step (wait 포함 / 제외) | 113 / 103 (wait 10, `max_steps` 220) |
| done 이 선 step index | 112 |
| policy 호출 수 | 21 |
| action chunk 배열 shape | 21 × 10 × 7 |
| mp4 | `libero_spatial_t0_e0_success.mp4`, 103 frames, 10 fps, 224×224 |
| 실행 action 과 chunk 불일치 step | 0 |
| wait 구간 action = dummy action | `true` |

**Latency** (client 는 `client.infer` 왕복, server 는 서버 측 `infer_ms`, policy 는 모델 추론 구간)

| 항목 | 값 |
|---|---|
| server ready (Popen → `/healthz` 200) | 58.03 s |
| orbax restore (server log) | 30.44 s |
| client episode 총 시간 | 90.42 s |

| 구간 | 첫 호출 (JIT 포함) | 호출 2~21 (n=20) p50 | p95 | max |
|---|---|---|---|---|
| client 왕복 | 36,497.7 ms | 88.6 ms | 220.6 ms | 305.6 ms |
| server infer | 36,493.9 ms | 85.8 ms | 182.9 ms | 303.6 ms |
| policy infer | 35,366.2 ms | 46.8 ms | 121.6 ms | 126.1 ms |

첫 호출 36.5 s 는 JIT compile 이 지배한다. 이 값을 steady-state latency 로 쓰지 않는다. 이후 호출이 V11AB 의 runtime
deadline 안에 드는지는 AG3S 가 붙은 뒤에 따로 잰다 — SL0 는 측정과 기록만 한다.

**기록 완결성** (plan §6.1 필수 항목 중 이 단계 해당분)

| 산출물 | 상태 |
|---|---|
| `episodes.jsonl` 필수 field | 누락 0 |
| episode npz 배열 | 누락 0 |
| mp4 | 있음 |
| `policy_records/` | 21 파일, 21 개 모두 npz 의 action chunk 와 일치 |
| server log | `logs/libero_client.log` · `logs/serve_policy_24038.log` · `logs/server_console.log` 있음 |
| `run_manifest.json` 필수 20 항목 | 16 present, **4 missing** (아래 열린 항목) |
| `source.patch` | 0 bytes, dirty = false |

**입력 보호** (실행 전후 storage 비교)

| 항목 | 값 |
|---|---|
| checkpoint 디렉터리 stat diff | 0 줄 |
| storage 기존 input (results 제외, 21,026 entries) stat diff | 0 줄 |
| `results/benchmark` 새 dir | 1개 (이번 run), 삭제된 dir 0 |
| `/mnt/dev/work` 경로 hit | 3 venv origin 0, run 의 text file 0 |

**checkpoint identity** — 독립 rehash script 로 16 파일 12,439,085,481 bytes 를 163.9 s 에 전부 다시 해시했다.
aggregate(파일 path 순 정렬) 재계산 값이 기록값 `1cb098d8…` 과 일치하고 per-file mismatch 는 0 이다.
(launcher 의 `--verify-checksums` 경로 자체는 이 STEP 에서 실행하지 않았다. 아래 not measured.)

**환경 독립성**

| venv | interpreter | 위치 |
|---|---|---|
| `.venv-libero` | Python 3.8.20 | `/mnt/dev/AG3S_VLA-benchmark/.python/cpython-3.8.20-linux-x86_64-gnu` |
| `.venv-openpi-live` | Python 3.11.16 | `/mnt/dev/AG3S_VLA-benchmark/.python/cpython-3.11.16-linux-x86_64-gnu` |
| `.venv-ag3s` | Python 3.11.16 | 같은 3.11.16 |

**회귀**

| 검사 | 결과 |
|---|---|
| live 보고서 범위 CPU test (`.venv-openpi-live`) | 73 passed, 0 skipped, 77 warnings |
| `tests/benchmark` (`.venv-ag3s`) | 16 passed, 0 skipped |
| `tests/benchmark` + `test_run_artifacts` + `test_policy_recorder_destination` | 20 passed, 1 skipped (`.venv-ag3s` 에 flax 가 없어 recorder test skip. impl.md 가 기존과 같은 skip 이라고 기록) |
| 16D legacy ESDF 기준선 | violated start 14/15 · has_target 15/15 · frame0 clearance -29.031 mm — 기준값과 일치 |
| 16D cuRobo ESDF 기준선 | violated start 10/15 · has_target 15/15 · frame0 clearance -9.172 mm — 기준값과 일치 |

SL0 가 새 파일만 추가했고 기존 코드를 바꾸지 않았으므로, 위 16D 기준선이 그대로인 것을 확인했다.

**자원** (실행 중 sampling 70 회)

| 항목 | 값 |
|---|---|
| pod working set 최대 | 61.403 GiB |
| GPU 전체 사용 최대 | 44,948 MiB (외부 2 process 36,136 MiB 포함) |
| 이 run 의 server GPU 최대 | 8,812 MiB |

### 공식 예제와의 차이

공식 `third_party/openpi/examples/libero/main.py` 와 **정책 입력 · 실행 action 은 같고**, 아래는 기록·운영 쪽 차이다
(출처 `SL0.impl.md`).

| # | 차이 | 영향 |
|---|---|---|
| 1 | `--task-ids` / `--episode-ids` 로 부분집합 실행 | task 0 episode 0 은 공식 sweep 의 첫 episode 와 같은 순서. 뒤쪽 episode 는 앞선 episode 를 거치지 않아 전체 sweep 의 같은 번호와 env 이력이 다를 수 있다 |
| 2 | 예외 시 `status=execution_error` + traceback 기록, client exit code 1. episode 시작 시 `done=False` | 공식은 예외 시 이전 episode 의 `done` 이 남을 수 있다 |
| 3 | 영상 이름 `videos/<suite>_t<task>_e<ep>_<success|failure>.mp4`, exclusive publish | 공식은 같은 task 의 다음 episode 가 덮어쓴다. frame · fps 동일 |
| 4 | 추가 기록: `episodes.jsonl`, step 별 npz, `client.infer` 왕복 시간, `server_timing` / `policy_timing` | 정책 입력 · 실행 action 불변 |
| 5 | 무거운 import 를 함수 안으로 이동 | 없음 (CPU test 가 Python 3.11 에서 module 을 import 하기 위함) |
| 6 | CLI 를 argparse 로, `--port` 필수, launcher 는 `--host 127.0.0.1` 전달 | 없음 |
| 7 | websockets 13.1 (Python 3.8) 에서 `ping_timeout` 없이 연결하는 subclass | upstream OpenPI client 와 같은 연결 |
| 8 | LIBERO import 전 `LIBERO_CONFIG_PATH/config.yaml` 이 없으면 거부 | 없음 (config 가 없으면 LIBERO 가 `input()` 으로 멈춘다) |
| 9 | 환경: `MUJOCO_GL=osmesa`, PYTHONPATH 제거, foreign torch `LD_LIBRARY_PATH` 제거, thread 수 제한 | **렌더러가 공식 GPU EGL 이 아니라 CPU llvmpipe.** pixel 단위 차이 가능성은 측정하지 않았다 |
| 10 | 설치 도구 버전, LIBERO editable compat 모드 | pin 은 예제 그대로 |

`pip check` 는 공식 방식(`uv pip sync` 는 의존성을 풀지 않는다)으로 LIBERO requirements 의 transitive 의존성이
설치되지 않아 불일치를 보고한다. 목록은 `SL0.impl.md`. 평가 경로(benchmark → env → `OffScreenRenderEnv` → step)는
이것들 없이 import · 실행됐다.

### 측정하지 않은 것

- renderer 차이(osmesa vs 공식 EGL)의 pixel 영향
- **성공률** (episode 1개)
- launcher 의 `--verify-checksums` 경로 자체 (독립 rehash script 로 대신 재계산)
- 다른 task / episode

### 열린 항목

| # | 내용 | 영향 / 다음 |
|---|---|---|
| 1 | **manifest 필수 항목 4개 누락**: `protected_authorized_links`, `phase`, `level`, `runtime_schedule` | pi0.5-only 에는 해당하지 않는 항목이지만 "해당 없음" 이라는 **선언도 기록돼 있지 않다**. V11AB 와 같은 형식으로 비교하려면 N/A 를 명시해 채워야 한다. 처리 시점은 lead 판정 대기 |
| 2 | manifest 의 `serve_policy` 자기 execution 항목에 `settings.env = "aloha_sim"` 이 기록됨 | serve_policy `Args` 기본값이 그대로 기록된 것이다. 실제 policy 는 `policy:checkpoint` 경로(`pi05_libero`, storage checkpoint)로 로딩됐다. 그러나 manifest 만 읽으면 환경이 aloha_sim 처럼 보여 오독 위험이 있다. 같은 항목의 `record_dir` 은 null |
| 3 | `.venv-libero` 의 torch(1.11.0+cu113)가 standalone import 후 process 종료 시 `free(): invalid pointer` 로 abort | episode 는 영향받지 않았다(client returncode 0, 기록 완료). pod 의 `LD_LIBRARY_PATH` 를 그대로 쓰면 torch import 자체가 `undefined symbol` 로 실패해, launcher 가 client env 에서 foreign site/dist-packages 항목을 제거한다. 종료 시 abort 의 원인은 확인하지 않았다 |
| 4 | **73-test 의 명령이 이전에 기록돼 있지 않았다.** | 이전 live 보고서와 live-validation run 에 정확한 argv 가 없어, verifier 가 collected 수 4+2+23+6+38 = 73 과 warning 출처(t43ta 16+61 = 77)로 파일 구성을 **재구성**해 실행했다. 재구성 명령은 verify.json `pytest_regression73` 에 있다. 이 범위의 정본 명령으로 삼을지는 lead 판정 대기 |
| 5 | **pod working set 이 실행 중 최대 61.403 GiB.** lead 가 전달한 pod budget 은 64 GiB 인데(이 값은 verify.json 에 없다), 여유가 작다 | 측정 전 checkpoint page cache 를 `posix_fadvise DONTNEED` 로 비웠다(파일 내용 · mtime 불변, stat diff 0). 공유 host 의 load average 는 실행 직전 68~92 였다. server 와 client 가 한 pod 에서 돌 때의 memory 여유를 V11AB(추가 geometry 처리) 단계에서 다시 본다 |
| 6 | `CHECKSUMS.json` 의 `aggregate_rule` 문구와 실제 계산이 다르다 | 문구는 "sorted `<sha256>  <path>\n` lines" 인데 producer 는 record 를 **path 순** 으로 정렬한다. line 문자열 정렬로 재계산하면 `baffa4ca…` 가 나와 불일치, path 순이면 기록값과 일치한다. launcher 는 path 순을 쓴다. 문구 수정은 이 STEP 의 범위 밖 |
| 7 | manifest 의 `storage.rollouts`(`<run>/runs`) 경로가 만들어지지 않았다 | 이 run 의 산출물은 그 경로를 쓰지 않으므로 기록만 한다 |

### 판정

verifier 판정(`verify.json` 의 verdict): 16D 기준선 일치(legacy · cuRobo), `status=success` · `official_success=true`,
checkpoint aggregate 일치, input stat diff 0 · 새 results dir 1, 73 passed · `tests/benchmark` 16 passed.
**SL0 의 통과 기준은 "완주와 기록"이며(완주 = status 가 success 또는 max_steps), 충족했다.** 사용자가 2026-10-12 에 확정했다.
다음은 SL-1(agent-view · wrist RGB/depth 와 camera 변환 → AG3S 관측 형식)이다.

### 되돌아올 지점 (SL0)

| 전환 신호 | 돌아와서 할 일 |
|---|---|
| Python 3.8 pinned stack 이 이 host(CUDA driver 580, OSMesa)에서 설치 · 렌더가 안 된다 | 3.11 단일 env 대안은 공식 조건과 달라지므로 **사용자에게 다시 묻는다.** (이번에는 설치 · 렌더 모두 되어 해당 없음) |
| 1 episode 가 `execution_error` | adapter 단계로 넘어가지 않고 원인(관측 형식 · norm stats · port/통신)을 먼저 닫는다. (이번에는 `success` 라 해당 없음) |
| 이후 episode 가 failure | SL0 통과와 별개다(완주 · 기록이 기준). 성능 확인은 P2 소규모 평가에서 한다 |
| V11AB 와 manifest 형식을 맞출 때 열린 항목 1 · 2 가 비교를 막는다 | launcher 의 settings 에 N/A 선언과 policy source 를 명시하도록 고친다 (구현은 implementer) |
| renderer 가 결과 차이의 원인으로 의심된다 | osmesa 와 EGL 의 pixel 영향을 verifier 에게 측정 요청한다 (이번에는 측정하지 않음) |
