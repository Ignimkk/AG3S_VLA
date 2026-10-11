# SL0 — pi0.5-only LIBERO Spatial 1 episode: 독립 환경 · 공식 판정 · run 기록

> writer: lead (A0) · 읽는 쪽: implementer, verifier · 작성 2026-10-12 KST
> branch `feature/panda-benchmark`, 시작 HEAD `2ee68eed`. 계획 정본: `doc/develope_PLAN_SUMMARY.md` §2.1 SL-0, §7 P1.

## 왜 지금 이것인가

Panda benchmark 경로(Panda adapter → LIBERO Spatial → SafeLIBERO Spatial I/II → 전체 SafeLIBERO → LIBERO-Safety)의
첫 gate다. AG3S(V11AB)를 붙이기 **전에**, frozen pi0.5 LIBERO checkpoint가 이 checkout의 독립 환경에서
공식 LIBERO protocol 그대로 1 episode를 완주하고, 그 결과가 고유 run 디렉터리에 기록되는지 확인한다.
이것이 이후 SL-1(camera/depth), SL-2(Panda geometry), V11AB 비교의 **pi0.5-only 기준 경로**가 된다.

```
LIBERO sim (.venv-libero, py3.8) ──websocket──▶ OpenPI policy server (.venv-openpi-live, GPU)
   obs: agentview/wrist RGB(180° 회전), eef pos+axis-angle+gripper qpos, prompt
   ◀── action chunk (EE delta 6 + gripper 1), 5 step 실행 후 replan
   official success = env.step()의 done
```

사용자 판정 (2026-10-12): pi05_libero는 공개 GCS에서 storage의 새 input으로 받는다. LIBERO client는
공식 protocol과 같은 **별도 Python 3.8 `.venv-libero`** 를 이 checkout에 만든다. 전역 `python3.11` shim은 건드리지 않는다.

## 고정 입력

| 항목 | 값 |
|---|---|
| checkpoint | `$AG3S_STORAGE_ROOT/checkpoints/panda/pi05_libero/` (read-only, lead가 받음). `CHECKSUMS.json`에 파일별 SHA-256과 aggregate |
| norm stats | 위 checkpoint의 `assets/physical-intelligence/libero/norm_stats.json` |
| policy config | OpenPI `pi05_libero` (`third_party/openpi/src/openpi/training/config.py:761`) |
| LIBERO source | submodule `benchmark/LIBERO` @ `f78abd68ee28` (수정 금지) |
| client 기준 | `third_party/openpi/examples/libero/main.py` + `requirements.txt` (pinned: robosuite 1.4.1, mujoco 3.2.3, numpy 1.22.4, torch 1.11.0+cu113) |
| protocol | suite `libero_spatial`, seed 7, `num_steps_wait` 10, `replan_steps` 5, `resize_size` 224, render 256, `max_steps` 220, dummy action `[0]*6+[-1]` |
| 이번 episode | task_id 0, episode_idx 0 (해당 task의 init state 0) |

## 구현에 요청하는 것 (→ implementer)

1. **`.venv-libero` 생성 script** `scripts/setup/create_libero_env.sh`
   - `create_live_env.sh`와 같은 방식: 자기 `UV_PYTHON_INSTALL_DIR=$ROOT/.python`, 자기 `.cache/uv`, `--link-mode copy`.
     Python 3.8.x를 자기 `.python`에 설치한다. 이전 `/mnt/dev/work` venv·source를 쓰지 않는다.
   - OpenPI 예제 `requirements.txt`의 pin 그대로 설치(extra index는 예제와 동일). `benchmark/LIBERO` editable,
     `third_party/openpi/packages/openpi-client` editable.
   - LIBERO 첫 import의 interactive config 질문이 나오지 않도록 `$LIBERO_CONFIG_PATH/config.yaml`(env.sh 기본: `.runtime/libero`)을
     이 checkout의 `benchmark/LIBERO/libero/libero/{bddl_files,init_files,assets}` 등으로 생성한다. Dataset 경로가 필요하면 storage 아래 비어 있는 위치를 가리키되 생성/쓰기는 하지 않는다.
   - 설치 후 resolved freeze를 `environments/libero/` 에 남기고(`requirements.txt` 사본 + `validated-freeze.txt`), dry-run 재현 확인.
   - pip check 충돌이 있으면 숨기지 말고 impl.md에 그대로 적는다.
2. **client** `benchmark/runners/libero_pi05_client.py` — **Python 3.8에서 실행**되므로 3.8 문법만 쓴다.
   - 관측 구성·image 전처리·`_quat2axisangle`·replan·dummy wait·success 판정은 공식 예제와 **동일 동작**. 차이가 생기면 diff 목록을 impl.md에 적는다.
   - 인자: `--run-dir`(필수, 이미 존재하는 run root), `--host`, `--port`, `--task-suite-name`, `--task-ids`, `--episode-ids`, `--seed`.
     `--run-dir` 이 `$AG3S_STORAGE_ROOT/results/benchmark/` 아래가 아니거나 `run_manifest.json`이 없으면 거부.
   - 저장(모두 exclusive create, 기존 파일 덮어쓰기 금지), run root 아래:
     - `episodes.jsonl` — episode당 1행: suite, task_id, task description, episode_idx, init-state index, seed, `official_success`(공식 예제와 같은 `done` 판정), `status` ∈ {success, max_steps, execution_error}, error 메시지, 실행 step 수(wait 제외/포함 모두), policy 호출 수, 호출별 latency 목록, 시작/종료 시각.
       Exception은 공식 예제처럼 삼키되 `status=execution_error`로 **구분 기록**한다(성공·실패로 조용히 섞지 않음).
     - `episodes/<suite>_t<task>_e<ep>.npz` — step별: 실행 action, 받은 전체 action chunk, `robot0_eef_pos/quat`, `robot0_joint_pos`, `robot0_gripper_qpos`, 전처리 전 agentview/wrist 원본 frame(256, 회전 전), reward, done.
     - `videos/<suite>_t<task>_e<ep>_<success|failure>.mp4` — 공식 예제와 같은 전처리 frame, 10 fps.
3. **launcher** `benchmark/runners/launch_libero_pi05_only.py` — `.venv-ag3s`에서 실행.
   - `common.run_artifacts.create_run`으로 새 run 생성. settings에 condition `pi0.5-only`, AG3S 비활성(serve_policy에 safety layer 미사용 경로임을 명시),
     protocol 상수 전부, task/episode 목록, checkpoint 경로 + `CHECKSUMS.json` aggregate SHA-256 + norm_stats SHA-256,
     `.venv-libero`/`.venv-openpi-live` interpreter와 핵심 package version, privileged input 없음, MUJOCO_GL.
   - **빈 port를 새로 고른다**(bind 확인; 8123 금지; 고른 port를 manifest에 기록). Server는
     `.venv-openpi-live`로 `third_party/openpi/scripts/serve_policy.py --port <P> --record policy:checkpoint --policy.config pi05_libero --policy.dir <ckpt>`,
     `AG3S_RUN_DIR=<run>`. Server ready를 확인한 뒤 client 실행, 끝나면 **자기가 띄운 server PID만** 종료. `record_execution`으로 실제 argv 기록.
   - 기존 `--env LIBERO` default 경로(gs://)를 쓰지 않는다 — 반드시 storage checkpoint.
   - `--dry-run`: run을 만들지 않고 두 command와 settings만 출력.
4. **CPU unit test** `tests/benchmark/`: 공식 예제와 protocol 상수 일치(예제 source를 AST로 읽어 비교), `_quat2axisangle` 일치,
   `--run-dir` 거부 규칙, port 선택(8123 제외·사용 중 제외), launcher dry-run command 구성. `.venv-ag3s`로 실행.
   추가로 `.venv-libero`에서 libero_spatial task 0 env 생성 + dummy 10 step **smoke**(policy 없음)까지는 implementer가 확인한다.

**손대지 않을 것:** `AG3S/**`, `integration/**`, `robots/**`의 algorithm 코드, V11AB 설정, `benchmark/LIBERO` 등 submodule source,
`third_party/openpi` model/policy 코드(serve_policy 수정이 꼭 필요하면 이유를 impl.md에 쓰고 최소 변경), storage의 기존 input.
GPU episode 실행과 figure는 하지 않는다(verifier 담당).

## 측정에 요청하는 것 (→ verifier)

| # | 무엇을 재나 | 어떤 숫자로 | 통과 기준 |
|---|---|---|---|
| 1 | 환경 독립성 | 3 venv의 interpreter/`__file__` origin, `/mnt/dev/work` 경로 0건 | 모두 이 checkout |
| 2 | checkpoint identity | `CHECKSUMS.json` 재계산 aggregate 일치, norm_stats SHA | 일치 |
| 3 | 1 episode 완주 | official_success, status, 실행 step 수, policy 호출 수 | `status`가 success 또는 max_steps (execution_error 아님) |
| 4 | latency | server load 시간, 첫 호출(JIT) 별도, 나머지 호출 p50/p95/max (ms) | 측정·기록만 |
| 5 | 기록 완결성 | episodes.jsonl/npz/mp4/policy_records/server log/manifest 필드 | plan §6.1 필수 항목 중 해당분 존재. 누락 목록 기록 |
| 6 | input 보호 | 실행 전후 storage 기존 input의 변경 0건(checkpoint 디렉터리 파일 mtime/size, results 아래 새 dir은 이번 run 1개) | 0건 |
| 7 | 회귀 | 기존 CPU 테스트(`tests/` 중 live 보고서의 73개 범위) | 기존과 동일 pass |

- 1 episode 결과로 성공률을 말하지 않는다. 숫자는 episode 1개의 사실이다.
- figure 3종: (1) 실제 scene — agentview·wrist 원본 frame의 시작/중간/끝 montage, (2) graph — EEF xyz·gripper·action 시계열 + policy latency,
  (3) table — protocol·checkpoint·환경·결과 표. `make-figure` skill, sidecar `.json`.
- raw: `outputs/verify/SL0/`, run root 경로를 verify.json에 기록.

## 이 STEP 이 답하는 질문

이 checkout만으로(이전 workspace 없이) frozen pi0.5 LIBERO checkpoint가 공식 LIBERO Spatial protocol의 1 episode를
공식 판정까지 완주하고, 그 run이 input을 건드리지 않고 고유 디렉터리에 재현 가능하게 기록되는가?

## 되돌아올 지점

- Python 3.8 pinned stack이 이 host(CUDA driver 580, OSMesa)에서 설치/렌더가 안 되면 → 3.11 단일 env 대안은 공식 조건과 달라지므로 사용자에게 다시 묻는다.
- 1 episode가 execution_error면 → adapter 단계로 넘어가지 않고 원인(관측 형식·norm stats·port/통신)을 먼저 닫는다.
- 결과가 failure여도 SL0 통과 여부와는 별개다(완주·기록이 기준). 성능 확인은 P2 소규모 평가에서 한다.
