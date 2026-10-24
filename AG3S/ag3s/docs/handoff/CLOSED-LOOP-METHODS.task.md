# CLOSED-LOOP-METHODS — 비교 방법을 LIBERO / SafeLIBERO closed loop 에 연결하고 benchmark 고유 지표 측정

> writer: lead · 읽는 쪽: implementer I → verifier V3 · 작성 2026-10-24 KST
> 사용자 지시 2026-10-24: "네 진행하세요" — (a) methods closed-loop 연결과 (b) ATS 진단(별도 task `ATS-CLOUD-DIAG`)을 진행한다.
> 선행: `3c677553` (BASELINE-EVAL).
> 지표 정의는 `Baseline/paper_baselines/docs/metrics.md` 의 `closed-loop` 행이다: SafeLIBERO TSR / CAR / ETS, KNOWS SR / CR / SSR, LIBERO SR 등.

## 왜

지금까지 methods 9 arm 은 open loop 대리 지표만 있다. 원 논문이 보고하는 지표(TSR · CAR · ETS 등)는 rollout 이 있어야 잴 수 있다. 논문 Tab. 4(SafeLIBERO)에는 같은 checkpoint 로 다시 돌린 행이 필요하다.

## 설계 원칙 (lead)

1. **기존 code 는 고치지 않는 것을 우선한다.**
   - 기존 구조는 V11AB 의 `AG3S.trajopt.serve_safe --robot panda` (server) + `benchmark/runners/libero_safe_client.py` (client, `.venv-libero`) + `launch_libero_paired.py` (launcher) 다.
   - methods 는 **같은 wire protocol 을 쓰는 새 server** `Baseline/paper_baselines/closed_loop/serve_method.py` 와 **새 launcher** `closed_loop/launch_method.py` 로 붙인다.
   - launcher 는 `launch_libero_paired` / `launch_libero_pi05_only` 의 helper 를 import 해서 재사용한다(create_run, GPU lock, suite registry, client argv, paired summary). client 는 기존 것을 그대로 쓴다.
2. **server 구조**
   - pi0.5 policy 는 `serve_safe --no-safe` 경로와 같은 방식으로 load 한다(같은 checkpoint · norm stats · seed 처리).
   - `SafetyFilter` arm: policy chunk → filter → client. 실행 순간 기준 action 규약(EE delta, K-of-H)은 기존 server 와 같게 한다.
   - `SamplingGuidance` arm: `pi0.sample_actions(guidance_fn=...)` 으로 sampling 안에서 적용한다. openpi 는 수정하지 않는다.
3. **venv 를 섞지 않는다.**
   - filter 의 의존성(cvxpy · osqp · Grounding DINO)은 `.venv-baselines` 에 있다. `.venv-openpi-live` 에 설치하지 않는다.
   - filter 를 같은 process 에서 import 할 수 없으면, `.venv-baselines` 의 sidecar process(local socket 또는 pipe)로 분리한다.
   - guidance 는 `.venv-openpi-live` 안에서 JAX 로 돈다.
4. **관측**
   - filter 와 guidance 에 필요한 depth · K · T · q · ee 는 지금 wire 관측으로 오는지 확인한다(V11AB 가 이미 받는다).
   - privileged 입력(GT segmentation, SafeLIBERO obstacle 이름)이 wire 에 없으면, 우선 non-privileged 변형(depth + detector, scene depth obstacle)으로 연결한다.
   - privileged 변형에 client 수정이 필요하면 **flag(기본 off, off 일 때 argv · 출력 bit 동일)로 최소 변경**한다. 수정 전에 같은 checkout session `ag3s-vla-benchmark-1a` 에 SendMessage 로 알리고 진행 중인 run 이 없는지 확인한다.
5. **기록**
   - run 은 `create_run` 으로 `results/benchmark/<날짜_commit_run-id>/` 에 만든다.
   - manifest 에 method · 변형 · 원 방법과의 차이 · privileged 여부를 남긴다.
   - 매 call 의 개입 여부 · ‖Δa‖ · h 를 기록한다(offline 지표와 같은 key).
6. **지표 계산** — `closed_loop/metrics.py`
   - SafeLIBERO TSR / CAR / ETS 는 공식 정의를 따른다(`metrics.md` 의 출처: 공식 code 의 collision 판정 1 mm, ETS 정의).
   - 기존 benchmark evaluator 의 contact 판정(SL5c 의 safe success)도 함께 낸다. 두 정의는 섞지 않는다.
   - KNOWS 의 SR / CR / SSR 은 원 정의대로 낸다.

## Implementer I

- 위 설계를 구현한다:
  - `closed_loop/serve_method.py`, `launch_method.py`, `metrics.py`, (필요하면) filter sidecar
  - tests: wire round-trip, guidance off 일 때 VLA-only 와 action 이 bit 동일, filter identity 경로
- **수용 기준:** `--method none` 이 VLA-only 기존 server 와 같은 chunk 를 내야 한다. 같은 obs · seed 로 bit 동일인지 test 한다(GPU 없이 가능한 범위에서는 CPU JAX, 아니면 V3 에게 넘긴다).
- CPU smoke 는 fake policy 또는 저장된 obs replay 로 한다. 기존 `bm-setup-b-cpu-client-smoke-fakepolicy` 를 참고한다.
- `CLOSED-LOOP-METHODS-I.impl.md` 에 V3 명령을 남긴다.
- GPU 금지. commit 금지.
- 기존 파일을 수정해야 하면 impl.md 에 이유와 bit 동일성 증거를 남긴다.

## Verifier V3 (GPU, lock)

1. **smoke:** arm 마다 SafeLIBERO spatial Level I dev 1 episode.
   - arm: `none`(= VLA-only), aegis(non-privileged), knows, multilink_cbf, constrained_flow_matching, nudge, omniguide, lan_o3dp_guidance_adapted
   - `--method none` 이 기존 VLA-only run(SL5c 같은 episode)과 같은 action 을 내는지 확인한다.
2. **pilot:** SafeLIBERO spatial Level I dev 20 episode(SL5c 와 같은 episode · seed), 전 arm.
   - `run_parallel_gpu.sh` 방식으로 최대 2 job, 같은 checkout 안에서만 돌린다.
   - 시간이 4 h 를 넘을 것으로 보이면 smoke 뒤 lead 에게 추정 시간을 알리고 기다린다.
3. **지표:** arm 별 TSR / CAR / ETS(공식), safe success(SL5c 정의), KNOWS SR / CR / SSR, 개입률 · ‖Δa‖, latency.
   - 이미 있는 V11AB 결과(SL5c / SL5i)는 같은 protocol 일 때만 참조 행으로 둔다.
4. **figure:** `figures/baselines/closed_loop/<method>/`
   - 실제 장면: rollout frame strip, 장애물 접촉 순간
   - 핵심 기능: call 별 h · 개입 시계열, nominal vs 실행 EE path
   - 표: arm 별 지표
5. `CLOSED-LOOP-METHODS-V3.verify.json` 을 만든다. 해석은 쓰지 않는다.

- **GPU 규칙:** nvidia-smi 확인 → `flock -w <초> LOCK CMD`. 시작과 끝에 `ag3s-vla-benchmark-1a`, `ag3s-vla-development-ed` 에 알린다.

## 되돌아올 지점

- privileged 변형, Level II · 다른 suite, LIBERO-Safety 는 pilot 결과를 보고 사용자가 정한다.
- 지표 정의가 공식 code 와 다르게 구현된 것이 보이면 pilot 수치를 쓰지 않고 다시 잰다.
