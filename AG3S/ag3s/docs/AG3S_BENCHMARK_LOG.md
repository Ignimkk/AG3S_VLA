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
| | SL-1 agent-view · wrist RGB/depth, intrinsics/extrinsics → AG3S 관측 형식 | **완료 (2026-10-12, 사용자 확정)** | 아래 [SL1](#sl1--libero-panda-camera-adapter-rgbdepth--k--t_base_cam--v11ab-wire-관측-2026-10-12). 판정 2건(K principal point 127.5, `robot_state` image 순간)은 SL2 에서 구현 예정 |
| | SL-2 Panda geometry / self-filter | **다음** | SL1 판정 2건 반영 포함 |
| | SL-3 ~ SL-8 (attention read-out, ATS/LTE, EE delta ↔ CCR, phase, contact evaluator, runner) | 미착수 | P1 통과 기준에는 pi0.5-only 뿐 아니라 V11AB 최소 1 episode 도 포함. SL-3 은 attention → depth 사상에서 policy image 의 좌우 반전을 되돌려야 한다(SL1) |
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
| **camera adapter** | LIBERO/robosuite 관측(OpenGL row order RGB · normalized depth · camera pose)을 V11AB 가 읽는 형식(OpenCV row order · metric z-depth · K · `T_base_cam`)으로 바꾸는 Python 3.8 module (`benchmark/adapters/libero_panda/`) |
| **wire replica** | AG3S 를 import 할 수 없는 Python 3.8 쪽에서 `AG3S.trajopt.wire` 의 camera 직렬화(key · dtype)를 그대로 다시 구현한 것. 3.11 쪽 `unpack_camera_observations` 로 왕복 test 해 동치를 보장한다 |
| **K / `T_base_cam`** | K = pinhole intrinsics(3×3, fx fy cx cy). `T_base_cam` = camera 좌표(OpenCV 축)를 robot base frame 으로 옮기는 4×4. base frame 은 MuJoCo body `robot0_base` |
| **OpenCV / OpenGL row order** | OpenGL render 는 image 의 첫 행이 아래쪽(bottom-up)이고, OpenCV 는 첫 행이 위쪽(top-down, +y 가 아래, +z 가 앞). adapter 는 raw 를 `[::-1]` 로 OpenCV 순서로 바꾼다 |
| **z-depth** | camera 광축(+z) 방향 거리. ray 를 따라간 거리(range)가 아니다. robosuite 는 `near / (1 - d (1 - near/far))` 로 normalized depth 를 metric z-depth 로 바꾼다 |
| **back-project** | pixel (u, v)과 depth 로 3D 점을 만드는 연산. K 의 역 + `T_base_cam` |
| **principal point (cx, cy)** | 광축이 image 와 만나는 pixel 좌표. pixel 중심이 정수 index 인 좌표계에서 W = 256 이면 정중앙은 127.5 이고, robosuite 의 `W/2 = 128` 은 0.5 pixel 어긋난다 |
| **pixel centre / MSAA sample position** | RGB · segmentation 은 pixel 중심에서 sample 된다. 이 renderer(osmesa llvmpipe, MSAA `offsamples` 4)에서 depth 는 pixel 안 `(u+0.375, v+0.875)` 에서 sample 된다. renderer 에 따라 달라질 수 있다 |
| **physics substep / image instant** | MuJoCo 가 `env.step` 하나 안에서 여러 번 도는 시간 단위(0.002 s). robosuite 는 camera render 를 substep loop 안에서 하므로 `env.step` 이 돌려준 joint 값은 image 보다 한 substep 뒤다. image 가 찍힌 순간을 image instant 라 부른다 |
| **segmentation** | pixel 마다 그 pixel 에 보이는 MuJoCo geom id 를 담은 image. 이 STEP 에서는 depth 정합의 정답(어느 물체인가)으로만 쓴다 |
| **action replay** | 기존 run 의 기록 action 을 policy 없이 같은 task · init state · seed 로 다시 실행해 새 run 에 추가 정보(geometry)를 기록하는 방식. 원 기록과 byte 단위로 같은지 확인할 수 있다 |
| **left-right mirror (policy image)** | pi0.5 가 받는 image 는 OpenCV image 의 열만 뒤집은 것(`policy256 == opencv[:, ::-1]`)이다. 180° 회전이 아니다 |
| **co-visible** | 두 camera 가 모두 보고 있는 영역. cross-camera 일관성(#4)은 이 영역에서만 의미가 있다 |

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
> run dir 이름의 시각(예: `20261012T045129`)은 **KST 다.** `common/run_artifacts.py` 의 `create_run` 이 UTC+9 timezone 으로 시각을 만들기 때문이다(구조상 확정, 추정 아님).
> 원본: [`handoff/SL0.task.md`](handoff/SL0.task.md) · [`handoff/SL0.impl.md`](handoff/SL0.impl.md) ·
> [`handoff/SL0.verify.json`](handoff/SL0.verify.json). raw 출력은 `outputs/verify/SL0/`.

### 이 STEP 이 답하는 질문

이 checkout 만으로(이전 workspace 없이) frozen pi0.5 LIBERO checkpoint 가 공식 LIBERO Spatial protocol 의
1 episode 를 공식 판정까지 완주하고, 그 run 이 입력을 건드리지 않고 고유 디렉터리에 재현 가능하게 기록되는가.

```
LIBERO sim (.venv-libero, py3.8) --websocket--> OpenPI policy server (.venv-openpi-live, GPU)
   obs: agentview/wrist RGB(~~180 deg rotation~~ → SL1 측정: left-right mirror, 180° 회전 아님), eef pos + axis-angle + gripper qpos, prompt
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

**실제 scene** — agentview · wrist 원본 frame 의 시작 / 첫 policy 호출 / 중간 / 끝 montage (raw 256×256, 공식 ~~180° 회전~~ 좌우 반전(SL1 #3) 전).

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
| 1 | ~~**manifest 필수 항목 4개 누락**: `protected_authorized_links`, `phase`, `level`, `runtime_schedule`~~ | ~~pi0.5-only 에는 해당하지 않는 항목이지만 "해당 없음" 이라는 **선언도 기록돼 있지 않다**. V11AB 와 같은 형식으로 비교하려면 N/A 를 명시해 채워야 한다. 처리 시점은 lead 판정 대기~~ **SL1 에서 처리**: manifest 에 `n/a` + 이유와 `runtime_schedule`(control 20 Hz, replan 5 step 등)을 명시하도록 launcher 를 고쳤다. 확인은 unit test(`run_cpu_regression.sh` B 44 passed) 수준이며, 실제 policy server 를 띄운 launcher 실행에서 manifest 를 다시 확인한 것은 아니다. |
| 2 | ~~manifest 의 `serve_policy` 자기 execution 항목에 `settings.env = "aloha_sim"` 이 기록됨~~ | ~~serve_policy `Args` 기본값이 그대로 기록된 것이다. 실제 policy 는 `policy:checkpoint` 경로(`pi05_libero`, storage checkpoint)로 로딩됐다. 그러나 manifest 만 읽으면 환경이 aloha_sim 처럼 보여 오독 위험이 있다. 같은 항목의 `record_dir` 은 null~~ **SL1 에서 처리**: checkpoint 경로로 서빙하면 `env = None`, `policy = {source, config, dir}` 로 기록한다(`serve_policy.py` 최소 변경). unit test 확인 수준이며 live 서버 실행의 manifest 는 재확인 전이다. 이전 run 의 `aloha_sim` 기록은 그대로 남아 있다. |
| 3 | `.venv-libero` 의 torch(1.11.0+cu113)가 standalone import 후 process 종료 시 `free(): invalid pointer` 로 abort | episode 는 영향받지 않았다(client returncode 0, 기록 완료). pod 의 `LD_LIBRARY_PATH` 를 그대로 쓰면 torch import 자체가 `undefined symbol` 로 실패해, launcher 가 client env 에서 foreign site/dist-packages 항목을 제거한다. 종료 시 abort 의 원인은 확인하지 않았다 |
| 4 | ~~**73-test 의 명령이 이전에 기록돼 있지 않았다.**~~ | ~~이전 live 보고서와 live-validation run 에 정확한 argv 가 없어, verifier 가 collected 수 4+2+23+6+38 = 73 과 warning 출처(t43ta 16+61 = 77)로 파일 구성을 **재구성**해 실행했다. 재구성 명령은 verify.json `pytest_regression73` 에 있다. 이 범위의 정본 명령으로 삼을지는 lead 판정 대기~~ **SL1 에서 처리**: 정본 명령은 `bash scripts/testing/run_cpu_regression.sh` 이다(A 가 SL0 의 5 파일 73 개). SL1 검증에서 A 73 passed, exit 0. |
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
| ~~V11AB 와 manifest 형식을 맞출 때 열린 항목 1 · 2 가 비교를 막는다~~ | ~~launcher 의 settings 에 N/A 선언과 policy source 를 명시하도록 고친다 (구현은 implementer)~~ SL1 에서 처리 |
| renderer 가 결과 차이의 원인으로 의심된다 | osmesa 와 EGL 의 pixel 영향을 verifier 에게 측정 요청한다 (이번에는 측정하지 않음) |

---

## SL1 — LIBERO Panda camera adapter: RGB/depth · K · T_base_cam → V11AB wire 관측 (2026-10-12)

> 2026-10-12 KST. 구현 commit `0e13395e`(KST 12:33) · 검증 commit `6c7ce76f`(KST 13:14). 사용자 확정 2026-10-12.
> 원본: [`handoff/SL1.task.md`](handoff/SL1.task.md) · [`handoff/SL1.impl.md`](handoff/SL1.impl.md) ·
> [`handoff/SL1.verify.json`](handoff/SL1.verify.json). raw 출력은 `outputs/verify/SL1/`.
> 이 절의 수치는 모두 `SL1.verify.json` 의 `numbers` 가 출처다. 단위는 mm 이고, 따로 적지 않으면 114 observation(task 0 · episode 0 의 replay 전체) 위의 값이다.

### 이 STEP 이 답하는 질문

LIBERO Panda 의 두 camera 관측을, **policy 입력을 바꾸지 않고**, V11AB 가 읽는 OpenCV / robot-base 규약의
metric depth · K · `T_base_cam` 으로 **물체 · table 에 mm 수준으로** 옮길 수 있는가?

```
robosuite render (OpenGL row order, depth = normalized z-buffer [0,1])
  -> [SL1 adapter, Python 3.8]  OpenCV row order, metric z-depth(m), K(fovy), T_base_cam = inv(T_world_base) * T_world_cam (OpenCV axes)
  -> V11AB wire key: ag3s/cameras, ag3s/depth/<cam>(uint16 mm), ag3s/K/<cam>, ag3s/T_base_cam/<cam>,
                     ag3s/robot_state/<cam>, ag3s/stamp/<cam>, ag3s/stamp_mode = sim_frozen
  -> AG3S.trajopt.wire.unpack_camera_observations -> CameraObservation (OpenCV, robot base frame)
camera 이름 table: agentview -> head, robot0_eye_in_hand -> left_wrist
```

`left_wrist` 로 매핑한 이유는 pi05_libero transform 이 wrist image 를 `left_wrist_0_rgb` slot 에 넣기 때문이다. SL-3 attention token 순서와 맞추기 위한 선택이며, SL-3 audit 결과가 다르면 adapter table 만 바꾼다.

### 무엇을 만들었나

| 구성 | 내용 |
|---|---|
| camera adapter (`benchmark/adapters/libero_panda/camera.py`, Python 3.8) | camera 별 metric z-depth(OpenCV row order) · K(fovy 식) · `T_base_cam`(MuJoCo body `robot0_base` 기준, OpenCV 축) · 9D `robot_state`(joint 7 + gripper 2) · stamp. robosuite `camera_utils` 의 세 공식을 다시 썼다(그 module 이 `h5py` 를 import 하는데 `.venv-libero` 는 공식 pin 이라 설치하면 freeze 재현이 깨진다). 원 공식과의 일치는 AST 로 꺼내 exec 한 값으로 비교한다 |
| policy image <-> OpenCV pixel 사상 | client 의 `[::-1, ::-1]` + `resize_with_pad` 256→224 를 명시적 함수(`opencv_to_policy_pixel` / `policy_to_opencv_pixel`)와 역함수로 만들었다 |
| `capture_state` observable | robosuite 는 camera render 를 `env.step` 끝이 아니라 physics substep loop 안에서 찍는다. 그래서 `env.step` 이 돌아온 뒤의 `env.sim` 은 image 보다 늦다. image 와 같은 substep 의 `T_world_cam` 등을 obs 에 싣는 읽기 전용 observable 을 geometry capture 가 켜졌을 때만 설치한다 |
| wire replica (`wire_pack.py`, Python 3.8) | `AG3S.trajopt.wire.pack_request` 의 camera 부분과 key · dtype 이 같은 dict/npz. AG3S 를 import 하지 않는 복제이므로 cross-env test 로 동치성을 보장한다 |
| client `--capture-geometry` (기본 off) | on 이면 `camera_depths=True` 로 env 를 만들고 step 마다 `episodes/<stem>_geom.npz` 기록. `--capture-segmentation` · `--capture-float-depth` 도 기본 off. off 이면 client argv · 출력이 SL0 와 같다 |
| action replay (`libero_replay.py` + `launch_libero_replay.py`) | 기존 run 의 기록 action 으로 policy 없이 같은 task · init state · seed 를 재생하고 새 run 에 geometry 를 기록한다. client 의 `run_tasks` 를 그대로 써서 seed · env 생성 · 대기 · replan · done 처리가 같다. 기록 `action` 과 chunk 가 bit 단위로 안 맞으면 시작을 거부하고, 기록이 모자라면 action 을 만들어내지 않고 `execution_error` 로 끝낸다 |
| manifest 해당 없음 항목 + `runtime_schedule` | SL0 launcher 의 manifest settings 에 protected/authorized links · phase · level 등 pi0.5-only 에 해당 없는 항목을 `n/a` + 이유로 명시. `runtime_schedule` 은 LIBERO source 에서 AST 로 읽은 control 20 Hz, replan 5 step, policy 호출 주기 5 step = 0.25 s sim 등을 적는다 |
| `serve_policy` env 기록 수정 | checkpoint 경로로 서빙하면 manifest 의 `env` 를 `None`, `policy = {source, config, dir}` 로 기록한다(이전에는 `Args` 기본값 `aloha_sim` 이 그대로 남았다). load · serve 동작은 변경 없음 |
| 고정 CPU 회귀 `scripts/testing/run_cpu_regression.sh` | A: SL0 verifier 가 재구성한 5 파일(`.venv-openpi-live` + `run_without_legacy.py`), B: `tests/benchmark`(같은 venv), C: `tests/benchmark`(`.venv-ag3s`). 하나라도 실패면 exit 1 |
| test | `tests/benchmark/test_libero_sl1.py` + `.venv-libero` 용 `libero_py38_checks.py`(pytest 가 subprocess 로 호출) |

`AG3S/**` algorithm · config, `robots/**`, `benchmark/LIBERO` submodule, storage 기존 input 은 건드리지 않았다.

**계획과 달라진 점** (출처 `SL1.impl.md`)

| # | 차이 | 영향 |
|---|---|---|
| 1 | robosuite `camera_utils` 를 import 하지 않고 공식을 다시 씀 | 값은 AST 비교로 확인 |
| 2 | depth 를 float64 로 계산 후 float32 저장 | robosuite 의 float32 계산과의 차이는 wire 양자화보다 작다고 impl.md 가 기록한다 (이 STEP 의 verify 수치는 아님). wire 왕복 오차는 #5 에서 측정 |
| 3 | geometry on 이면 env 에 `ag3s_capture_state` observable 추가 | 아래 #1 이 RGB 불변을 확인한다 |
| 4 | segmentation 은 `env.step` 이 돌아온 시점에 따로 render | image 순간보다 1 substep 뒤. 아래 "capture instant" 에서 크기를 잰다 |
| 5 | K 는 robosuite / AG3S RB-Y1 과 같은 `cx = W/2`, `cy = H/2` | pixel 중심이 정수 index 인 AG3S `backproject` 규약에서는 광축이 `(W-1)/2 = 127.5` 라 0.5 pixel 어긋난다. 이 STEP 에서는 고치지 않고 크기를 쟀다. 아래 판정 |

### 실행

| 항목 | 값 |
|---|---|
| source (replay 대상) | SL0 run `20261012T045129_6ba81024d243_sl0-pi05only-t0e0` (task 0, episode 0, 읽기만) |
| run a (geometry on, depth float 포함) | `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261012T123505_0e13395ee6a6_sl1-replay-geom-t0e0` |
| run b (geometry + segmentation + depth float) | `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261012T123554_0e13395ee6a6_sl1-replay-seg-t0e0` |
| run c (flag 없음 = SL0 경로 보존 확인) | `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261012T123646_0e13395ee6a6_sl1-replay-off-t0e0` |
| 명령 | `.venv-ag3s/bin/python benchmark/runners/launch_libero_replay.py --source-run <SL0 run id> --task-ids 0 --episode-ids 0 [--capture-geometry --capture-float-depth [--capture-segmentation]] --run-id sl1-replay-{geom,seg,off}-t0e0` |
| 환경 | GPU · policy server 불필요. launcher `.venv-ag3s` -> replay `.venv-libero`. 렌더러 `llvmpipe (LLVM 15.0.7, 256 bits)`, Mesa 23.2.1 (OSMesa), `offsamples` 4 |
| 자원 | pod working set 최대 45.179 GiB, 새 results dir 3개 |

### 결과 — Figure

세 figure 는 verifier 가 만들었다 (`outputs/verify/SL1/scripts/make_figures.py`, sidecar `.json` 동봉).

**실제 scene** — run b 의 observation 55. 위 줄: head RGB(segmentation 윤곽 + 투영한 body origin + wrist depth 를 back-project 해 head 위에 얹은 점), wrist RGB(반대 방향 overlay), policy 가 실제로 받은 224×224 image(SL0 `policy_records` step_9), head depth(OpenCV row). 아래: 두 camera 를 `robot0_base` frame 으로 back-project 한 3D point cloud 와 camera frustum, table top z.

![SL1 scene](figures/sl1-scene.png)

**graph** — 왼쪽 위부터 #1 replay 차이(전부 0), #2 물체별 median 거리 막대와 전체 CDF(세 K 규약), #4 wrist -> head nearest neighbour CDF, depth sample 위치 격자(`|z_ray - depth| < 0.01 mm` 인 pixel 비율), capture instant 별 robot 표면 거리 CDF.

![SL1 graph](figures/sl1-graph.png)

**table** — 변환 규약 · K · T 와 #1~#7 결과를 한 장에 모은 표.

![SL1 table](figures/sl1-table.png)

### 결과 — #1 replay 재현성 (depth 활성화가 policy 입력을 바꾸지 않는가)

| 비교 대상 (SL0 원 기록 vs) | episode npz SHA-256 | video SHA-256 | 비교 key 수 / 최대 절대 차이 | done step | jsonl |
|---|---|---|---|---|---|
| run a (depth on) | 동일 | 동일 | 14 / 0.0 | 112 | success, env step 113 (대기 제외 103), policy 호출 21 |
| run b (depth + segmentation) | 동일 | 동일 | 14 / 0.0 | 112 | 동일 |
| run c (flag 없음) | 동일 | 동일 | 14 / 0.0 | 112 | 동일 |

- episode npz SHA-256 은 네 run 모두 `e8a15133b8e2ae4aa9c069532d6eca9eba96ed453b070f1b3523a8c48aaaa088`, mp4 SHA-256 은 `9ce6114fb5d99dc57549b28c54a6e047d9b967f4b15b381584c88899e7036f8a` 다. 즉 **RGB · eef/joint/gripper state · action · video 가 byte 단위로 같다.**
- run a 와 run b 의 geometry npz 는 공통 32 key 가 모두 같고, run b 에만 `segmentation_geom/head`, `segmentation_geom/left_wrist` 가 더 있다.
- SL0 의 policy image(21 호출 x image · wrist)는 기록된 raw 에 client 의 변환을 다시 적용한 값과 21/21 로 일치한다.

따라서 `camera_depths=True` 와 `ag3s_capture_state` observable 은 render 와 물리 상태를 바꾸지 않는다. task 문서의 "별도 render pass 로 전환" 조건에는 해당하지 않는다. #7 의 SL0 경로 보존(run c 가 SL0 와 동일)도 이 표로 확인됐다.

### 결과 — #2 depth <-> geom 표면 정합

segmentation 으로 고른 물체 pixel 을 back-project 한 base-frame 점과, 같은 순간(image instant)의 MuJoCo geom 표면 사이의 최근접 거리다. segmentation 과 pose 는 모두 image instant 로 맞췄다(아래 capture instant 의 V1). K 규약 세 가지를 같은 depth 에 적용했다.

| 이름 | principal point (cx, cy; pixel 중심이 정수 index 인 좌표) | 뜻 |
|---|---|---|
| `K_recorded` | (128.0, 128.0) | SL1 adapter 가 기록한 값 (robosuite 와 같은 `W/2`) |
| `K_pixel_centre` | (127.5, 127.5) | pixel 중심 규약. RGB · segmentation 이 sample 되는 위치와 일치 |
| `K_depth_sample` | (127.625, 127.125) | 이 renderer 가 depth 를 sample 하는 위치 (아래 (c)) |

**head (agentview)** — median / p95 (mm)

| 대상 | n (pixel) | `K_recorded` | `K_pixel_centre` | `K_depth_sample` |
|---|---|---|---|---|
| table | 3,153,421 | 2.00 / 2.85 | 0.90 / 1.28 | 0.18 / 0.41 |
| bowl_1 (target) | 75,545 | 1.20 / 2.56 | 0.54 / 1.17 | 0.16 / 0.44 |
| bowl_2 | 54,013 | 1.14 / 2.70 | 0.52 / 1.25 | 0.16 / 0.46 |
| plate | 148,629 | 2.03 / 2.57 | 0.86 / 1.19 | 0.19 / 0.39 |
| ramekin | 18,973 | 1.62 / 2.79 | 0.77 / 1.31 | 0.15 / 0.40 |
| cookies | 58,839 | 1.93 / 2.30 | 0.80 / 1.17 | 0.17 / 0.36 |
| cabinet | 943,236 | 1.56 / 2.03 | 0.56 / 0.97 | 0.13 / 0.28 |
| stove | 179,550 | 2.39 / 2.83 | 1.02 / 1.35 | 0.16 / 0.34 |
| robot | 854,825 | 1.26 / 2.79 | 0.58 / 1.34 | 0.17 / 0.42 |

**left_wrist (robot0_eye_in_hand)** — median / p95 (mm)

| 대상 | n (pixel) | `K_recorded` | `K_pixel_centre` | `K_depth_sample` |
|---|---|---|---|---|
| table | 3,854,155 | 0.25 / 0.51 | 0.25 / 0.49 | 0.25 / 0.47 |
| bowl_1 (target) | 642,523 | 0.33 / 0.99 | 0.23 / 0.56 | 0.18 / 0.46 |
| bowl_2 | 1,212 | 0.39 / 1.18 | 0.33 / 0.67 | 0.09 / 0.42 |
| plate | 924,562 | 0.28 / 0.74 | 0.25 / 0.55 | 0.25 / 0.49 |
| ramekin | 43,746 | 0.49 / 1.82 | 0.34 / 0.87 | 0.17 / 0.47 |
| cookies | 255,586 | 0.26 / 0.58 | 0.24 / 0.49 | 0.22 / 0.47 |
| cabinet | 130,177 | 1.12 / 1.58 | 0.28 / 0.89 | 0.12 / 0.33 |
| stove | 150,097 | 0.28 / 1.22 | 0.26 / 0.52 | 0.23 / 0.47 |
| robot | 769,260 | 0.36 / 0.56 | 0.19 / 0.39 | 0.15 / 0.31 |

wrist depth 가 0.05 m 보다 가까운 pixel(전체의 8.2 %)은 AG3S `backproject` 에서 빠지므로 이 표와 #4 에 들어가지 않는다(아래 #6).

**table plane 잔차** (114 frame 의 median)

| camera | K | plane fit RMS | tilt | z - table top 평균 | z - table top RMS |
|---|---|---|---|---|---|
| head | `K_recorded` | 0.226 | 0.104 deg | +2.001 | 2.048 |
| head | `K_pixel_centre` | 0.226 | 0.048 deg | +0.836 | 0.883 |
| head | `K_depth_sample` | 0.226 | 0.006 deg | -0.038 | 0.230 |
| left_wrist | `K_recorded` | 0.286 | 0.002 deg | +0.025 | 0.295 |
| left_wrist | `K_pixel_centre` | 0.286 | 0.002 deg | +0.011 | 0.288 |
| left_wrist | `K_depth_sample` | 0.286 | 0.001 deg | -0.000 | 0.286 |

읽는 법.

- **depth row 순서와 base frame 정의는 맞다.** table plane 이 평평하게(fit RMS 0.2 mm 대) 복원되고 z 가 table top 높이와 mm 안에서 맞는다.
- head 의 오차는 거의 전부 **principal point 규약 하나**에서 온다. table median 이 2.00 -> 0.90 -> 0.18 mm 로 줄고, plane fit RMS 는 그대로 0.226 mm 라 plane 의 모양(기하)은 이미 맞고 평행이동·기울기만 남는 형태다. wrist 는 camera 가 table 에 가까워 K 규약의 영향이 작다(table 0.25 mm 로 동일).
- **표면까지의 거리가 5 mm(V11AB fine layer voxel)를 넘는 pixel 비율**은 `K_recorded` 기준 head table 0.40 %, bowl_1 1.17 %, plate 0.87 %, ramekin 2.10 %, robot 1.06 %; wrist table 0.35 %, bowl_1 0.30 %, plate 0.46 %, ramekin 0.98 %. 20 mm(coarse layer voxel)를 넘는 비율은 head table 0.35 %, ramekin 1.94 %, robot 0.90 %, wrist table 0.19 %. 이 꼬리는 K 규약을 바꿔도 거의 같다(head ramekin 2.10 % / 1.94 % 가 세 K 모두 같다). 이 꼬리의 원인은 분리하지 않았다.
- 1-pixel 침식(경계 pixel 제외)으로 median 이 거의 안 변한다(head table 2.00 -> 1.99 mm). 경계 pixel 이 median 을 끌고 있지 않다.
- 독립 방법(`mj_multiRay` 로 쏜 ray 와 depth 의 차이, 일치한 pixel 만)도 같은 순서다: head table median 2.72 -> 1.18 -> 0.25 mm, wrist table 1.42 -> 0.68 -> 0.29 mm (`K_recorded` -> `K_pixel_centre` -> `K_depth_sample`).

**결론: 세 K 규약 모두 median · p95 가 5 mm(fine voxel) 아래이므로 task 의 "오차가 voxel 이상이면 SL-2 로 가지 않는다" 조건에는 걸리지 않는다.** 규약 선택의 근거는 아래 판정에 있다.

### 결과 — #3 policy image <-> OpenCV pixel 대응

**policy 입력이 OpenCV image 의 무엇인가.**

| 측정 | 결과 |
|---|---|
| `policy256 == opencv[:, ::-1]` (열 반전 = left-right mirror, 행 순서 그대로) | 228 / 228 image |
| `policy256 == opencv rot180` (180° 회전) | 0 / 228 |
| 점 하나를 policy 변환에 통과시킨 dot test (81 점, sigma 2 px) | 좌표 오차 최대 0.0246 px, `policy_to_opencv_pixel` 왕복 오차 0 |
| 방향 단서 (head, frame 0) | base +y 로 5 cm 가면 OpenCV du = +16.6, dv = 0.0 px; base +z 로 5 cm 가면 du = +0.69, dv = -11.50 px |
| 방향 단서 (wrist, frame 0) | base +y 5 cm: du = -35.40, dv = +1.17 px; base +z 5 cm: du = +4.62, dv = -11.06 px |
| 물체 삼각형 방향 (OpenCV pinhole 예측과 일치 / policy 예측과 일치) | head 4814 / 5073 vs 259 / 5073 (114 frame); wrist 1565 / 1566 vs 1 / 1566 |
| policy 좌표 규칙 | `policy224 (dx, dy) = 0.875 * (-du, +dv)` (`policy256[i, j] = opencv[i, 255 - j]`, 256 -> 224 scale 0.875, pad 0) |

예: head frame 0 에서 target bowl_1 의 mask centroid 는 OpenCV (194.39, 131.63), policy 224 image 에서 (52.97, 115.11) 이다. plate 는 OpenCV (201.15, 166.87) -> policy (47.06, 145.95). OpenCV 에서 오른쪽에 있던 물체가 policy 에서는 왼쪽에 보이고 세로 위치는 같은 비율로 유지된다.

**투영이 물체에 떨어지는가.** 물체의 body origin(MuJoCo body 기준점)을 K · `T_base_cam` 으로 투영한 점이 OpenCV segmentation 에서 그 물체 mask 안에 있는지(hit)다. 이 hit 는 body origin 한 점의 검사다.

| 대상 | head hit / visible (`K_recorded`) | wrist hit / visible (`K_recorded`) |
|---|---|---|
| bowl_1 (target) | 114 / 114 | 87 / 114 |
| bowl_2 | 96 / 114 | 0 / 0 (시야 밖) |
| plate | 113 / 114 | 102 / 113 |
| ramekin | 44 / 73 | 25 / 30 |
| cookies | 114 / 114 | 89 / 89 |
| cabinet | 114 / 114 | 19 / 19 |
| stove | 114 / 114 | 0 / 19 |
| **합계** | `K_recorded` 709 / 757, `K_pixel_centre` 712 / 757, `K_depth_sample` 713 / 757 | 322 / 384, 324 / 384, 325 / 384 |

hit 가 아닌 경우(예: wrist 의 stove 0/19)는 body origin 점이 다른 물체나 robot 에 가려지거나 물체 바깥에 있어서일 수 있으나, 원인은 분리하지 않았다. 투영이 mask 에서 어긋나는 크기를 보려면 아래 mask bbox 중심 오차가 더 직접적이다.

**mesh bbox 중심 - mask bbox 중심** (median du, dv / |오차| median / p95, pixel)

| camera | n | `K_recorded` | `K_pixel_centre` | `K_depth_sample` |
|---|---|---|---|---|
| head | 626 | (+0.32, +0.73) / 1.12 / 7.93 | (-0.18, +0.23) / 0.43 / 8.33 | (-0.05, -0.14) / 0.66 / 8.16 |
| left_wrist | 156 (`K_depth_sample` 은 155) | (+0.48, +0.58) / 0.84 / 20.85 | (-0.02, +0.08) / 0.33 / 17.96 | (+0.10, -0.30) / 0.50 / 17.75 |

p95 는 occlusion 을 보정하지 않은 값이라 가려진 물체를 포함한다. median 은 `K_pixel_centre` 가 가장 작다(head 0.43 px, wrist 0.33 px).

### 결과 — #4 두 camera 일관성

table + plate 의 static 표면에서 wrist 점 -> head 점 nearest neighbour 거리다(n = 4,778,717, 두 표면 합). `all_covisible` 은 verify.json 이 따로 제시한 부분집합으로, 두 camera 가 모두 보는 점으로 좁힌 값이다(n 은 K 별로 4,420,513 / 4,425,143 / 4,429,904).

| 대상 | 지표 | `K_recorded` | `K_pixel_centre` | `K_depth_sample` |
|---|---|---|---|---|
| all | median / p95 (mm) | 2.46 / 10.69 | 1.66 / 9.41 | 1.43 / 8.69 |
| all | 5 mm 초과 / 20 mm 초과 | 6.80 % / 3.33 % | 6.34 % / 3.22 % | 6.14 % / 3.14 % |
| all_covisible | median / p95 (mm) | 2.41 / 3.41 | 1.61 / 2.55 | 1.38 / 2.34 |
| all_covisible | 5 mm 초과 / 20 mm 초과 | 0.23 % / 0.15 % | 0.23 % / 0.16 % | 0.25 % / 0.15 % |

- 같은 table 위 점의 간격 자체가 head 2.82 mm, wrist 1.40 mm(median) 이므로, nearest 거리의 median 이 1.4~2.5 mm 인 것은 두 cloud 의 sampling 간격 수준이다.
- all 과 all_covisible 의 큰 차이(p95 10.69 -> 3.41 mm, 5 mm 초과 6.80 % -> 0.23 %)는 상대 camera 가 보지 않는 영역의 점 때문이다 (해석. 두 집합의 정의 외에 원인 분석은 하지 않았다).

### 결과 — #5 wire 왕복 (Python 3.8 pack -> 3.11 `unpack_camera_observations`)

| 항목 | head | left_wrist |
|---|---|---|
| unpack 성공 frame | 114 / 114 | 114 / 114 |
| 반환 type | `AG3S.ag3s.types.CameraObservation` (camera id `HEAD`, `LEFT_WRIST`) | 동일 |
| depth 양자화 오차 최대 | 0.499992 mm | 0.5 mm (0.5000000000000004, 부동소수 표현) |
| clip 된 pixel / 비유한 값 | 0 / 0 | 0 / 0 |
| K · T · robot_state · stamp 최대 절대 차이 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| wire depth dtype | uint16 | uint16 |

실패 0 (114 file, 228 camera frame). `stamp_mode` 는 `sim_frozen`. 양자화 오차는 0.5 mm 한계(uint16 mm 반올림)를 넘지 않는다.

### 결과 — capture instant (image 가 찍힌 순간)

task 에 없던 측정이다. robosuite 는 render 를 physics substep 안에서 하고, 그 substep 의 상태 적분은 render 뒤에 끝난다. 그래서 `env.step` 이 돌려주는 joint 값과 그 시점의 `sim_state` 는 image 보다 **한 substep(0.002 s) 뒤**다. verifier 는 image 순간을 `P = forward(integratePos(sim_state.qpos, qvel, -dt))` 로 정의하고 re-render 로 검증했다.

| 검증 | 결과 |
|---|---|
| P 에서 re-render 한 RGB · depth 가 기록과 일치 | head 114 / 114, left_wrist 114 / 114 (RGB, depth 모두 bit-identical) |
| 기록한 `T_world_cam` · `T_world_base` 와 P 의 차이 | 0.0 (head, wrist, base) |
| `sim_state`(1 substep 뒤)에서 re-render 한 image 와 기록의 차이 | frame 0 은 같고(아무것도 움직이기 전), frame 10 / 11 / 40 / 80 / 113 에서는 다르다. 예: frame 40 wrist RGB 21,446 pixel, depth 최대 47.8 mm; head RGB 3,486 pixel |
| `sim_state_at_return - sim_state` 시각 | 0 s 또는 0.002 s (`{0.0, 0.002}`) |
| `robot_state` 가 obs 의 `robot0_joint_pos` / `gripper_qpos` 와 차이 | 0.0 (bit 일치) |
| `robot_state` 와 image 순간 qpos 의 차이, 최대 | 0.0017 rad (두 camera 동일). joint 별 [0.0003, 0.0017, 0.0003, 0.0017, 0.0007, 0.0017, 0.0009, 0.0002, 0.0002] rad (joint 7 + gripper 2) |
| 기록한 segmentation 이 어느 순간의 것인가 | `sim_state` 에서 re-render 한 값과 114 / 114 일치 (image 순간보다 1 substep 뒤) |
| 기록 segmentation vs image 순간 segmentation | frame 별 다른 pixel 수 head median 81.5 · max 179 (같은 frame 4 개), wrist median 82.0 · max 316 (같은 frame 1 개) (256 x 256 = 65,536 pixel 중) |

segmentation / pose 의 순간이 표면 거리에 미치는 영향(`K_recorded`, median / p95, mm). V1 = segmentation 과 pose 모두 image 순간, V2 = 기록 segmentation + image 순간 pose, V3 = 기록 segmentation + `sim_state` pose, V4 = 기록 segmentation + `env.step` 반환 순간 pose.

| 대상 | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| head robot | 1.26 / 2.79 | 1.26 / 2.80 | 1.32 / 2.82 | 1.37 / 2.92 |
| head bowl_1 | 1.20 / 2.56 | 1.20 / 2.57 | 1.20 / 2.64 | 1.20 / 2.72 |
| wrist robot | 0.36 / 0.56 | 0.36 / 0.56 | 0.27 / 0.65 | 0.37 / 0.86 |
| wrist bowl_1 | 0.33 / 0.99 | 0.33 / 1.00 | 0.33 / 1.02 | 0.38 / 1.13 |

순간이 어긋나도 median 은 0.1 mm 안팎으로 변하고, wrist robot p95 가 0.56 -> 0.86 mm 로 가장 크다. 이 STEP 의 #2 수치는 모두 V1 이다.

### 결과 — #6 V11AB 범위 정보 (측정만, SL-4 ESDF box 판정 자료)

| 항목 | 값 |
|---|---|
| robot base 원점 world z | 0.912 m (`T_world_base` 평행이동 (-0.66, 0, 0.912), 에피소드 내 최대 변화 0) |
| table top z (world / base frame) | 0.9 m / **-0.012 m** (base 원점보다 12 mm 낮다) |
| table mesh bbox (base frame) | z -0.062 ~ -0.012 m, x 0.16 ~ 1.16, y -0.6 ~ 0.6 |
| 물체 높이 (base frame z 범위, frame 113: 놓인 뒤) | bowl_1 [-0.0043, 0.061] (plate 위), bowl_2 [-0.0136, 0.039], plate [-0.0091, 0.0061], ramekin [-0.0127, 0.0307], cookies [-0.0127, 0.0074] m |
| 고정 가구 (base frame z 범위) | cabinet [-0.0024, 0.2146], stove [-0.027, 0.0469] m |
| frame 0 의 물체 z 범위 | bowl_1 / bowl_2 / plate / ramekin 의 z 최소 0.058 m, cookies 0.048 m (frame 113 과 다르다. 원인은 분리하지 않았다) |
| 점 cloud 의 base frame z, `K_depth_sample`, 5 frame 간격 (head) | p50 -0.0119, min -0.7908, max 0.5008 m; z < 0 인 비율 **68.84 %**, z > 1.6 m 인 비율 0 |
| 같은 지표 (wrist) | p50 -0.0116, min -0.0125, max 0.3147 m; z < 0 비율 **69.28 %**, z > 1.6 m 0 |
| depth 범위 밖 pixel, head | 0.05 m 미만 0 %, **3.0 m 초과 2.53 %**, depth 최소 0.63 m, 최대 3.069 m |
| depth 범위 밖 pixel, wrist | **0.05 m 미만 8.2 %**, 3.0 m 초과 0 %, depth 최소 0.042 m, 최대 0.397 m |
| depth 가 0 인 pixel | 두 camera 모두 0 % |

V11AB 의 depth 유효 범위는 0.05 ~ 3.0 m, 기본 ESDF box 는 base frame z ∈ [0, 1.6] 이다(`AG3S/ag3s/config.py` · `fields/esdf.py`, task 문서의 audit). 이 값들은 아래 (d), (e) 의 근거다.

### 결과 — #7 SL0 경로 보존 · 회귀

| 검사 | 결과 |
|---|---|
| flag 없는 replay(run c) = SL0 | episode npz · video SHA-256 동일 (#1 표) |
| `run_cpu_regression.sh` A (SL0 5 파일, `.venv-openpi-live`) | **73 passed** (SL0 의 73 과 같다) |
| 같은 script B (`tests/benchmark`, `.venv-openpi-live`) | 44 passed |
| 같은 script C (`tests/benchmark`, `.venv-ag3s`) | 43 passed, 1 skipped (serve_policy test, `.venv-ag3s` 에 flax 없음) |
| script exit status | 0 |
| 16D legacy ESDF 기준선 | violated start 14/15 · has_target 15/15 · frame0 clearance -29.031 mm — 기준값과 일치 |
| 16D cuRobo ESDF 기준선 | 이 STEP 에서는 돌리지 않았다 (GPU 미사용 지시) |

### 이 STEP 에서 알게 된 것 (평이한 설명)

**(a) 공식 policy 전처리는 OpenCV image 를 180° 회전시키는 것이 아니라 좌우로 뒤집는다.** SL0 기록은 이 변환을 "180° 회전"으로 적었다(`~~`로 표시한 SL0 의 해당 줄). robosuite 의 raw image 는 OpenGL row order(아래가 첫 행)이고 client 는 `[::-1, ::-1]` 로 행과 열을 모두 뒤집는다. 행 뒤집기는 OpenGL -> OpenCV 변환과 상쇄되므로, 결과는 OpenCV image 에서 열만 뒤집은 것(left-right mirror)과 같다(`policy256 == opencv[:, ::-1]` 228/228, rot180 0/228). **policy 입력은 바꾸지 않았다.** 바뀌는 것은 읽는 쪽이다: SL-3 에서 attention token 위치를 depth pixel 로 옮길 때 `policy224 (dx, dy) = 0.875 * (-du, +dv)` 로 좌우를 되돌려야 한다. 그렇지 않으면 attention 이 물체의 좌우가 바뀐 곳에 놓인다.

**(b) `robot_state` 가 image 보다 한 physics substep 늦었다.** robosuite 는 image 를 substep loop 안에서 찍는데, `env.step` 이 돌려주는 joint 값은 그 substep 의 적분이 끝난 뒤의 값이다. 카메라 pose(`T_world_cam`)는 image 순간과 bit 단위로 일치하도록 만들었지만 `robot_state` 는 obs 의 joint 값과 같아서 image 보다 0.002 s 뒤다(최대 0.0017 rad). 이 크기가 표면 거리에 주는 영향은 작지만(위 V1~V4), self-filter 가 robot pose 와 image 를 같은 순간으로 보도록 SL-2 전에 맞춘다.

**(c) depth 는 pixel 안의 다른 위치에서 sample 된다.** 이 환경의 renderer(osmesa llvmpipe, MSAA `offsamples` 4)에서 depth 는 pixel 중심 `(u+0.5, v+0.5)` 가 아니라 `(u+0.375, v+0.875)` 에서 sample 된다. 격자 실험에서 head 의 `|z_ray - depth| < 0.01 mm` 비율은 `(0.375, 0.875)` 에서 최대(0.806 / 0.821 / 0.829, frame 0 / 56 / 113)이고 wrist 는 1.0 이다. pixel 중심 `(0.5, 0.5)` 에서는 `|z_ray - depth|` median 이 head 에서 1.34 / 1.26 / 1.23 mm 였다. 반면 RGB 와 segmentation 은 pixel 중심에서 sample 된다(ray 와 segmentation 의 일치율이 offset 0.5 에서 최대: head 0.969, wrist 0.853~0.861; offset 0.0 에서는 head 0.954). K 의 principal point 를 어디에 두느냐가 head table 에서 2.00 / 0.90 / 0.18 mm 의 차이를 만드는 이유다. 공식 EGL/GPU renderer 에서 depth sample 위치가 같은지는 측정하지 않았다.

**(d) table top 이 base 원점보다 12 mm 아래라, 기본 ESDF box(z >= 0)의 바깥이다.** table top 은 base frame z = -0.012 m 이고 plate · bowl_2 · ramekin · cookies 의 바닥(frame 113)은 -0.009 ~ -0.014 m 다. head 점 cloud 의 68.84 %, wrist 의 69.28 % 가 z < 0 이다. 기본 box 를 그대로 쓰면 table 과 놓인 물체의 아래쪽이 voxel grid 에 들어가지 않는다. box 를 낮출지, table 을 support surface 로 따로 다룰지는 **SL-4 에서 결정한다.**

**(e) wrist 의 가까운 pixel 이 depth 최소(0.05 m) 아래다.** wrist depth 최소는 0.042 m 이고 0.05 m 미만 pixel 이 8.2 % 다. V11AB 의 `backproject` 는 이 pixel 을 버린다(task 문서의 `config.py` 0.05–3.0 m 범위). head 는 3.0 m 를 넘는 pixel 이 2.53 % (최대 3.069 m)이며, 이것도 버려진다. 어떤 물체가 이 영역에 있는지(gripper 가 쥔 물체 등)는 SL-2 · SL-3 에서 본다.

### 판정 (2026-10-12, 사용자)

| # | 판정 |
|---|---|
| 1 | **SL1 확정.** |
| 2 | **K 의 principal point 를 pixel 중심 규약 `(W-1)/2 = 127.5` 로 바꾼다.** adapter 의 calibration 만 바꾸는 것이며 V11AB algorithm 은 바꾸지 않는다. 바뀐 규약은 manifest 에 기록한다. |
| 3 | **MSAA sample 위치의 K(`K_depth_sample`, 127.625 / 127.125)는 채택하지 않는다.** renderer 에 따라 달라지는 값이기 때문이다. |
| 4 | **`robot_state` 를 image 순간 값으로 고친다.** SL-2 전에 반영한다. |

위 2, 4 는 **SL2 에서 구현한다. 아직 구현도 측정도 되지 않았다.** 이 절의 `K_pixel_centre` 열은 현재 기록한 depth 에 오프라인으로 K 만 바꿔 적용해 본 값이며, SL2 구현 후의 adapter 출력을 다시 재야 한다. 그 값이 위 기대(head table median 약 0.9 mm, table plane 평균 +0.84 mm)와 같을지는 그때 확인한다.

### 측정하지 않은 것

- 공식 EGL/GPU renderer 에서의 depth sample 위치 (측정은 osmesa llvmpipe, Mesa 23.2.1, `offsamples` 4 에서만)
- 다른 task · episode, 다른 해상도 (이 STEP 은 libero_spatial task 0 episode 0 의 replay 하나)
- 16D cuRobo 기준선 (GPU 미사용)
- `mj_multiRay` 가 gripper hand visual 과 plate 일부 pixel 을 통과해(ray geom != segmentation) ray 기반 수치는 일치한 pixel 만 쓴 값이다
- mask bbox 중심 오차의 occlusion 보정 (p95 는 occlusion 포함)
- #2 표면 거리의 5 mm 초과 꼬리(0.3~2.1 %)의 원인
- live policy server 와 `--capture-geometry` 를 함께 돌린 실행 (client 와 replay 가 같은 `run_tasks` 를 쓰므로 replay 로 실행됐다. server 와의 live 조합은 측정하지 않았다)
- 새로 추가한 manifest `n/a` 항목 · `runtime_schedule` · `serve_policy` 기록을 **실제 policy server 가 있는 launcher 실행**에서 확인한 것 (unit test 로만 확인: B 44 passed)

### 열린 항목

| # | 내용 | 영향 / 다음 |
|---|---|---|
| 1 | SL2 에서 K 를 127.5 로, `robot_state` 를 image 순간으로 고친 뒤 #2 · capture instant 를 재측정 | 판정 2 · 4 |
| 2 | ESDF box z 범위 · table 처리 | SL-4 에서 결정 ((d)) |
| 3 | SL-3 attention -> depth 사상에서 좌우 되돌리기 | `opencv_to_policy_pixel` / `policy_to_opencv_pixel` 사용 ((a)) |
| 4 | wrist 0.05 m 미만 pixel 과 head 3.0 m 초과 pixel 의 처리 | SL-2 · SL-3 ((e)) |
| 5 | segmentation 은 image 순간보다 1 substep 뒤에 render 된다 | image 순간 id 가 필요하면 `sim_state` 로 별도 env 에서 재생성(구현 차이 #4). V11AB 가 segmentation 을 쓰는 지점이 정해지면 판단 |
| 6 | 렌더러가 EGL 이면 depth sample 위치가 다를 수 있다 | 판정 3(MSAA K 비채택)의 근거. EGL 환경을 쓰게 되면 재측정 |

### 되돌아올 지점 (SL1)

| 전환 신호 | 돌아와서 할 일 |
|---|---|
| #1 에서 RGB 가 달라졌다면 | geometry 를 policy 와 별도 render pass 로 얻는 방식으로 전환하고 사용자에게 보고 (이번에는 bit-identical 이라 해당 없음) |
| #2 오차가 voxel 이상이었다면 | near/far · fovy · base pose 정의를 재점검, 원인 확정 전 SL-2 로 가지 않음 (이번에는 p95 가 2.85 mm 이하라 해당 없음) |
| SL-3 token 순서 audit 이 `left_wrist` slot 과 다르다 | adapter 의 camera 이름 table 만 변경 |
| 공식 EGL renderer 를 쓰게 된다 | depth sample 위치와 K 규약 재측정 (판정 3) |
