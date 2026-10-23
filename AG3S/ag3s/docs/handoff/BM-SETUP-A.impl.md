# BM-SETUP-A — 구현

> writer: ag3s-implementer · 읽는 쪽: verifier, scribe, lead · 2026-10-23 12:50 KST
> checkout `/mnt/dev/AG3S_VLA-benchmark`, branch `feature/panda-benchmark`, 기준 HEAD `2c58cee1` (commit 안 함, working tree 변경)
> **GPU 사용 없음.** CPU worker ≤ 2. 작업 중 pod working set 최대 약 38 GiB.

## 무엇을 했나 (평이한 요약 먼저)

LIBERO 4 suite(spatial · object · goal · 10)와 SafeLIBERO 4 suite × Level I/II, 모두 56 task를 policy 없이 CPU에서 열어 봤다.
56/56 task에서 env 생성 · reset · wait · 관측 변환이 SL1–SL2 규약 그대로였다. SafeLIBERO 32 task에서는 robot을 세워 두면
충돌이 0건이었다. EE를 obstacle 쪽으로 몰면 32/32 task에서 충돌 판정이 섰고, 모두 MuJoCo의 robot–obstacle 접촉이 함께 기록됐다.
공용 runner는 이미 모든 suite 이름을 받았다. 그래서 바꾼 것은 세 가지다: 시작 전 task · episode 범위 검사, suite protocol 표,
SafeLIBERO step budget opt-in flag. spatial의 기존 argv · settings · dry-run 출력 · client 연결은 변경 전 commit과 byte 단위로 같다.
새 method를 version으로 등록하고 jobs 파일 생성 → 실행 → 집계로 이어지는 recipe(`eval_matrix.py` + `EVAL_RECIPE.md`)를 만들었다.
GPU smoke jobs 파일은 만들기만 했고 실행하지 않았다.

**사용자 결정이 필요한 발견 1건:** 공식 `main_aegis.py`는 wait 20 step 뒤 `t = 0`으로 되돌린다(234줄).
따라서 SafeLIBERO의 300 / 550은 **wait를 뺀 policy step 수**다. 현재 기본 protocol(SL5c, 사용자 2026-10-14 판정
"wait 포함 t < 300")은 policy step 280 / 530만 준다. 기본값은 바꾸지 않았고, `--safelibero-step-budget main_aegis`로 공식값을 고를 수 있다.

---

## 1. Suite별 확인 표 (CPU, policy 없음)

조건: seed 7, `libero_pi05_client.make_task_env(task, 7, capture_geometry=True)`(client와 같은 env, depth on + SL1 capture observable),
init 0, `start_episode` → wait(LIBERO 10 / SafeLIBERO 20 dummy step) → 관측 변환. raw: `outputs/verify/BM-SETUP-A/probe/*.json`,
요약: `outputs/verify/BM-SETUP-A/probe_summary.json`.

| suite (level) | task 생성 / reset | init state | 관측 규약 ¹ | base pose (world) | 받침면 z (base) ² | fixture reset 이동 ³ | 충돌 판정 (hold / drive / 접촉) ⁴ | protocol 출처 |
|---|---|---|---|---|---|---|---|---|
| libero_spatial | 10/10 | 50 ×10 | ✓ | (-0.66, 0, 0.912) | main_table −0.011 | cabinet · stove ≤ 13.4 mm | — | openpi `examples/libero/main.py:60-61, 37, 104` (220 / wait 10) |
| libero_object | 10/10 | 50 ×10 | ✓ | (-0.60, 0, **0.0**) | **floor +0.001** | 없음 | — | 〃 `:62-63` (280) |
| libero_goal | 10/10 | 50 ×10 | ✓ | (-0.66, 0, 0.912) | main_table −0.011 | cabinet · stove · wine rack ≤ 13.4 mm | — | 〃 `:64-65` (300) |
| libero_10 | 10/10 | 50 ×10 | ✓ | 3종: kitchen (-0.66, 0, 0.912) · living room (-0.51, 0, **0.42**) · study (-0.75, 0, 0.912) | kitchen −0.011 · living room +0.006 · study −0.031 | stove · cabinet · rack · microwave · caddy ≤ 12.7 mm | — | 〃 `:66-67` (520) |
| libero_90 | 90 task × 50 init 개수만 | — | — | — | — | — | — | **범위 밖** (`:68-69`, 400) |
| safelibero_spatial I | 4/4 | 50 ×4 | ✓ | (-0.66, 0, 0.912) | −0.011 | cabinet · stove ≤ 13.1 mm | 0/4 · 4/4 · 4/4 | `main_aegis.py:78-79, 53` |
| safelibero_spatial II | 4/4 | 50 ×4 | ✓ | 〃 | −0.011 | 〃 | 0/4 · 4/4 · 4/4 | 〃 |
| safelibero_object I | 4/4 | 50 ×4 | ✓ | (-0.60, 0, 0.0) | floor +0.001 | 없음 | 0/4 · 4/4 · 4/4 | `:80-81` |
| safelibero_object II | 4/4 | 50 ×4 | ✓ | 〃 | 〃 | 없음 | 0/4 · 4/4 · 4/4 | 〃 |
| safelibero_goal I | 4/4 | 50 ×4 | ✓ | (-0.66, 0, 0.912) | −0.011 | cabinet · stove · rack ≤ 7.7 mm | 0/4 · 4/4 · 4/4 | `:82-83` |
| safelibero_goal II | 4/4 | 50 ×4 | ✓ | 〃 | −0.011 | 〃 | 0/4 · 4/4 · 4/4 | 〃 |
| safelibero_long I | 4/4 | 50 ×4 | ✓ | (-0.51, 0, 0.42) | living room +0.006 | 없음 | 0/4 · 4/4 · 4/4 | `:84-85` |
| safelibero_long II | 4/4 | 50 ×4 | ✓ | 〃 | 〃 | 없음 | 0/4 · 4/4 · 4/4 | 〃 |

1. 관측 규약(SL1–SL2), 모든 task · 두 camera에서 성립: 어댑터 RGB = raw `[::-1]`, policy image `raw[::-1, ::-1]` = OpenCV `[:, ::-1]`
   (좌우 반전), depth 유한 100 %, K pixel-centre(cx = cy = 127.5, fx = fy, fovy 45°), `T_base_cam` rigid, `robot_state` 9-D
   (image 순간 joint 7 + finger 2; step 반환 시점 `obs` 값과의 차이 ≤ 1.7e-4, image 순간과 반환 시점이 한 substep 다르기 때문, SL2), policy state 8-D, controller `OperationalSpaceController`, 20 Hz.
2. 받침면 z = head depth를 받침면 segmentation pixel에 back-project한 base-frame z 중앙값(m). Panda ESDF 상자 바닥은 base z −0.10
   (`profile.PANDA_ESDF_BOUNDS_LOWER_Z`)이다. 모든 suite에서 받침면이 상자 안에 있다.
3. 두 번째 `start_episode` 뒤 joint 없는 world 직속 body의 `body_pos` 변화(SL4b). 변화가 있으면 reset 이력이 장면을 바꾼다.
4. SafeLIBERO: active obstacle(공식 규칙)을 기준으로 잰다. hold = wait 뒤 20 dummy step 동안의 충돌 판정 수(최대 변위 8.6e-8 m, 기준 1 mm).
   drive = EE를 obstacle 쪽으로 OSC P-control로 보냈을 때 판정이 선 task 수(2–17 step).
   접촉 = 판정 step 이전 physics substep에서 robot(`robot0_*`/`gripper0_*`)–obstacle MuJoCo 접촉이 있었던 task 수.
   active obstacle: spatial moka pot(LI t3만 wine bottle) · object LI wine bottle / LII moka pot · goal moka pot(LII t1 red mug) ·
   long moka pot small / red mug. 나머지 obstacle과 다른 suite용 물체는 작업 공간 밖에 놓이며 두 camera 모두 0 pixel이다.

### Step protocol 대조

| 항목 | LIBERO (openpi example) | SafeLIBERO `main_aegis.py` | 우리 client (기본) | 일치 |
|---|---|---|---|---|
| wait | 10 (`:37`) | 20 (`:53`) | 10 / 20 | ✓ |
| loop | `t < max_steps + wait`, wait를 t에 셈 (`:104`) | wait 루프(`:166`) 뒤 **`t = 0`(`:234`)**, `while t < max_steps`(`:254`) | `t < max_steps + wait` | LIBERO ✓ · SafeLIBERO **✗** |
| policy step 상한 | 220 / 280 / 300 / 520 | 300 / 300 / 300 / 550 | LIBERO 같음 · SafeLIBERO 280 / 280 / 280 / 530 | SafeLIBERO −20 |
| 성공 | `done` | `done` (`:392`), SS = done ∧ ¬collide (`:420`) | 같음 | ✓ |
| 충돌 | — | README 규칙, wait 직후 기준 (`:237-252, 367-373`) | `CollisionMonitor` 같음 | ✓ (`collide_t`는 wait 포함 index: 공식 `collide_time` = `collide_t − 20`) |
| ETS | — | 끝난 episode의 `t`(`:404`): 성공이면 step − 1, 실패면 300 / 550 | 집계에서 계산 | 기본 budget에서는 실패가 280 / 530으로 끝나 논문 값과 척도가 다르다 |

근거: `test_suite_registry.py::test_main_aegis_resets_t_after_the_wait`(`t = 0`이 같은 indent의 무조건 문장임을 확인),
Supplementary Table S1–S4에서 실패 행의 ETS가 300.0 / 550.0이다.

## 2. Adapter 가정 표 — spatial에만 맞는 것이 있는가

| 가정 | 쓰는 곳 | spatial 밖에서 | 결과 / 조치 |
|---|---|---|---|
| camera 이름 `agentview`·`robot0_eye_in_hand`, K = fovy, OpenGL row 순서 | online (`camera.py`) | 56/56 task 같음 | 유지 |
| robot base = `robot0_base`, `T_base_cam`은 매 frame 계산 | online | base world pose가 scene마다 다름(z 0.0 / 0.42 / 0.912). head camera의 base 기준 위치도 다름 | 매 frame 계산하므로 문제없음 |
| ESDF 상자 base z ∈ [−0.10, 1.60] (Panda override) | online (profile) | 받침면 z −0.031 … +0.006, 작업 물체 전부 상자 안 | 유지. 상자 밖 물체는 SafeLIBERO의 치워 둔 물체뿐이고 camera에 0 pixel |
| BDDL fixture → MuJoCo body 이름 규칙 `X_main`/`X_<part>` + alias `main_table→table` | offline (`task_target`, scene GT · perception replay · SL3/4 도구) | `kitchen_table`(body `table`) · `study_table` · `floor`(이름 그대로인 body)가 풀리지 않았다 | **고침**: `FIXTURE_BODY_ALIASES`에 3개 추가. spatial의 `main_table` 항목은 그대로 |
| target 규칙 = 첫 goal predicate의 첫 인자(On/In → 목적지) | offline (ATS 채점 · GT) | libero_goal t0: target이 region(`wooden_cabinet_1_middle_region`, body 아님) · goal t7 · libero_10 t2: target이 fixture(`flat_stove_1`, TurnOn) · libero_10 t0/1/4/6/7/8 · safelibero_long 4/4: 옮길 물체가 2개 | **고치지 않음, 표시만 함.** online ATS는 attention으로 target을 정하므로 영향이 없다. GT 채점을 이 suite로 넓힐 때 규칙을 정해야 한다 |
| `task_target`의 BDDL root = LIBERO | offline | SafeLIBERO task는 이 함수로 못 찾는다(fork 경로) | probe는 경로를 직접 넘겨 확인했다. offline 도구를 SafeLIBERO로 넓힐 때 root 인자가 필요 |
| 한 episode에 target 1개(ATS 후 latch · attach는 한 물체) | online V11AB / pb | 2-물체 task: libero_10 6/10, safelibero_long 4/4 | CPU로 확인할 수 없다. 첫 물체를 놓은 뒤 두 번째 target을 다시 잡는지는 GPU smoke에서 본다 |
| reset 이력(fixture 재배치, SL4b) | 재생 · export · pairing | fixture가 있는 scene(spatial · goal · kitchen · study, SafeLIBERO spatial · goal)에서 매 reset마다 ≤ 13.4 mm 이동. floor · living room scene은 0 | pairing 규칙 8(아래 recipe)에 넣었다 |
| gripper `empty_below` (`robots/panda/description/gripper_calibration.json`) | online latch | spatial bowl 10 task로 잰 값. 더 두꺼운 물체(병 · 상자)는 개도가 더 커서 empty로 오인할 방향이 아님 | CPU로 확인 불가. smoke에서 latch 기록 확인 |
| segmentation id = geom id, 물체 = `<name>_main` 등 | offline | 모든 BDDL 물체 · fixture가 body로 풀림(alias 추가 후 unresolved 0) | 유지 |
| SafeLIBERO obstacle 이름 `*obstacle*_joint0`, active 선택 규칙 | 충돌 판정 | 4 suite 모두 같은 규칙으로 1개가 고정되며 BDDL objects에 있음 | 유지 |
| 옛 run 이름 `libero10-init0` | 기록 | 실제는 LIBERO **Spatial** task 0–9 × init 0이다. `libero_10` suite가 아니다 | recipe와 보고서에 주의로 적음 |

## 3. 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/runners/launch_libero_paired.py:26-31` | docstring: 모든 suite · 범위 검사 · step budget | 사용법 |
| `…/launch_libero_paired.py:60-61` | `SAFELIBERO_STEP_BUDGETS`, 기본 `total_incl_wait` | opt-in budget |
| `…/launch_libero_paired.py:99-113` | `step_budget(args)`; `client_argv`는 budget이 기본값이 **아닐 때만** `--safelibero-step-budget`를 붙임 | 기본 argv 불변 |
| `…/launch_libero_paired.py:131-138` | `protocol_settings`: 기본값이 아닐 때만 `max_steps` · `loop` · `safelibero_step_budget` 갱신 | 기본 settings 불변 |
| `…/launch_libero_paired.py:229-236, 261` | `run_tag(mode, args)`로 run id 꼬리 계산을 분리(내용 같음) | `eval_matrix`가 run 이름을 미리 앎 |
| `…/launch_libero_paired.py:382-384` | `--safelibero-step-budget {total_incl_wait, main_aegis}` | 위 |
| `…/launch_libero_paired.py:404-409` | budget을 SafeLIBERO에만 허용. `suite_registry.check_selection`: task / episode 범위, registry split label과 suite · split 일치를 **checkpoint 확인과 GPU lock 전에** 검사 | 잘못된 id로 server를 띄우지 않게 |
| `benchmark/runners/libero_safe_client.py:53-76` | `SAFELIBERO_STEP_BUDGETS`, `safelibero_policy_steps`, `safelibero_step_budget_record` | opt-in budget |
| `…/libero_safe_client.py:320-322, 343-347, 357` | client flag. 기본값이 아니면 jsonl extra에 `safelibero_step_budget`; `max_steps = safelibero_policy_steps(...)`(기본 = 기존 식 `total − 20`) | 위 |
| `benchmark/adapters/libero_panda/task_target.py:43-48` | `FIXTURE_BODY_ALIASES` + `kitchen_table`, `study_table`, `floor` | probe에서 풀리지 않은 받침면 3종 |
| **새** `benchmark/configs/suites.json` | suite protocol 표(steps · wait · split · 지표 정의 · task 이름 · 공개 기준 · 출처 줄) | 과제 3 |
| **새** `benchmark/runners/suite_registry.py` | 표 loader, split label, `check_selection` (py3.8 호환, stdlib) | launcher · recipe 공용 |
| **새** `benchmark/runners/eval_matrix.py` | `plan` / `smoke` / `aggregate`, VLA-only 재사용 판정(`run_basis`, `episode_reusable`, `apply_reuse`), 시간 상한 | 과제 4 · 5 |
| **새** `benchmark/evaluators/suite_probe.py` | CPU probe(py3.8): 관측 규약 · 받침면 · 이름 규칙 · target 규칙 · ESDF 상자 · fixture reset · SafeLIBERO hold / drive + substep 접촉 | 과제 1 · 2 |
| **새** `benchmark/docs/EVAL_RECIPE.md` | 새 method 등록 → plan → 실행 → 집계 한 장 | 과제 4 |
| **새** `tests/benchmark/test_suite_registry.py` (7) | 표 = client 상수 = openpi example / main_aegis 원문, `t = 0` 위치, task 이름 = 원본 파일, split label, alias, probe 상자 = profile 상자 | |
| **새** `tests/benchmark/test_bm_setup_a_identity.py` (25) | `2c58cee1` 대비 동일성(아래) + 거절 경로 + main_aegis opt-in | |
| **새** `tests/benchmark/test_eval_matrix.py` (7) | suite 파싱, jobs 줄의 launcher dry-run 왕복, smoke 계획 · 시간식, Wilson · McNemar · ETS, 합성 run 집계, 재사용 규칙 1–8 | |
| **새** `tests/benchmark/test_suite_probe.py` (2) | py3.8 probe: SafeLIBERO Object II t0(floor scene, 충돌 판정 + 접촉), LIBERO 10 t5(study table) | |

### spatial 동일성 근거 (기준 = `2c58cee1`, 변경 전 HEAD)

`tests/benchmark/test_bm_setup_a_identity.py`가 변경 전 launcher(`git show 2c58cee1:…`)와 지금 launcher를 같은 인자로 돌려 비교한다.
대상은 4 case: LIBERO Spatial 기본, LIBERO Spatial seg + V11AB-pb7 + seed 11, SafeLIBERO Spatial I(pb5), SafeLIBERO Spatial II.

| 비교 | 결과 |
|---|---|
| parsed args(새 key 제외) · server argv(두 mode) · client argv · run tag | 같음 |
| `protocol_settings` · `runtime_schedule` JSON byte | 같음 |
| `run_mode`의 `create_run` kwargs(manifest settings 전체) · 시작한 server argv · execution 기록 3개 | byte 같음 (case × mode 8개) |
| `--dry-run` stdout | byte 같음 (case 4개) |
| client(py3.8) `main()` 연결: `max_steps` · jsonl extra · parsed args · collision wait · policy seed, 변경 전 `libero_safe_client.py`를 `git show`로 실행해 비교 | LIBERO Spatial · SafeLIBERO Spatial II 모두 JSON 같음 |

기존 `test_launcher_record_constraints.py`(SL5j 이전 `32a4ca8f` 대비 동일성)와 `test_safelibero_client.py`도 그대로 통과한다.

## 4. Recipe 사용법 (정본 `benchmark/docs/EVAL_RECIPE.md`)

```bash
source scripts/setup/env.sh
# (1) AG3S/adapters/panda/profile.py AG3S_VERSIONS 에 "V11AB-pbN": (<flags>) 추가
# (2) jobs
.venv-ag3s/bin/python benchmark/runners/eval_matrix.py plan --version V11AB-pbN --conversion C \
    --suites all --split dev --run-prefix bm-pbN --out-dir outputs/eval/bm-pbN-dev \
    [--vla-only reuse --vla-only-runs <run dir> ... [--accept-diff <path> ...]] [--safelibero-step-budget main_aegis]
# (3) GPU (lead 조율 뒤)
benchmark/runners/run_parallel_gpu.sh --jobs outputs/eval/bm-pbN-dev/jobs.txt
# (4) 집계 -> plan.json 옆 aggregate.json / aggregate.md
.venv-ag3s/bin/python benchmark/runners/eval_matrix.py aggregate --plan outputs/eval/bm-pbN-dev/plan.json
```

- 집계 항목: suite × level마다 official success(Wilson 95 %), SafeLIBERO `collided` · safe success · **CAR / TSR / ETS**
  (정의는 `main_aegis.py` 줄 번호와 Supplementary 표 범례 그대로, `suites.json` `safelibero.metrics`).
  VLA-only 대비 paired 2×2, gained / lost episode, 비율 차이 bootstrap 95 % 구간, McNemar exact p, family 합계.
- VLA-only 재사용: episode 단위 8개 조건(SL5g pairing basis + reset 이력). `pairing_basis.json`에 근거를 남긴다.
  예시(실제 SL5f VLA-only LI dev run으로 시험): `outputs/verify/BM-SETUP-A/example_reuse_LI_dev/pairing_basis.json`.
  `38722284` 이후 VLA-only 경로에서 `AG3S/adapters/panda/server.py`, `AG3S/trajopt/serve_safe.py`, `launch_libero_paired.py`가
  바뀌었으므로 **재사용 거절** 판정이 나왔다. 이 commit이 들어가면 `libero_safe_client.py`도 목록에 추가된다(기본 동작 불변, 위 identity test).
- 옛 이름 주의: `libero10-init0` run = LIBERO Spatial task 0–9 × init 0.

## 5. GPU smoke 목록 (만들기만 함, 실행 안 함)

| 파일 | 내용 | job | 시간 상한 (직렬 / 2 병렬) |
|---|---|---|---|
| `outputs/verify/BM-SETUP-A/smoke/jobs.txt` (+ `plan.json`) | suite × level 12개 각각 task 0 · episode 0(dev), `--modes vla_only,vla_v11ab`, V11AB 기준선 · 역변환 B, 기본 step budget | 12 | 74 분 / **39 분** |
| `outputs/verify/BM-SETUP-A/smoke_main_aegis/jobs.txt` (선택) | SafeLIBERO 8개만, `--safelibero-step-budget main_aegis` | 8 | 54 분 / 27 분 |

job별 상한(분, VLA-only + V11AB, server 시작 포함): libero_spatial 4.4 · object 5.0 · goal 5.2 · libero_10 7.5 ·
safelibero spatial / object / goal 각 5.7 · long 8.9. 계산식은 `estimate_job`이다: (server 시작 + step 상한 × step당 시간)을 두 mode에 대해 더한다.
step당 시간은 SL5d–SL5i spatial run의 중앙값(LIBERO 0.263 / 0.361 s, SafeLIBERO 0.312 / 0.440 s, `timing_basis.json`)이다.
다른 suite의 step당 시간은 아직 재지 않았다. 12 job 모두 launcher `--dry-run`을 통과했다(protocol의 suite · level · max_steps · wait 확인).
실행: `source scripts/setup/env.sh && benchmark/runners/run_parallel_gpu.sh --jobs outputs/verify/BM-SETUP-A/smoke/jobs.txt`.

## 6. dev / test 분리 — **확정(사용자 2026-10-23)**

| suite | dev | test | 상태 |
|---|---|---|---|
| libero_spatial | init 0–4 × 10 task | 5–49 | 확정 2026-10-12 (plan §6.0) |
| libero_object · libero_goal · libero_10 | suite마다 init 0–4 × 10 task (50) | 5–49 (450) | **확정(사용자 2026-10-23)** |
| safelibero_spatial I · II | level마다 episode 0–4 × 4 task (20) | 5–49 (180) | 확정 2026-10-14 |
| safelibero_object · goal · long I · II | suite · level마다 episode 0–4 × 4 task (20) | 5–49 (180) | **확정(사용자 2026-10-23)** |

`suites.json` `splits.status`와 `EVAL_RECIPE.md`에 같은 내용으로 적었다. `--data-split` label은 `suite_registry.split_label`이 만든다.

## 단위 검증

```bash
source scripts/setup/env.sh
.venv-ag3s/bin/python -m pytest -q tests/benchmark -p no:cacheprovider
.venv-ag3s/bin/python -m pytest -q tests/benchmark/test_suite_registry.py tests/benchmark/test_bm_setup_a_identity.py \
    tests/benchmark/test_eval_matrix.py tests/benchmark/test_suite_probe.py tests/benchmark/test_launcher_record_constraints.py \
    tests/benchmark/test_safelibero_client.py tests/benchmark/test_parallel_gpu_launcher.py -p no:cacheprovider
```

- `tests/benchmark` 전체: **235 passed, 2 skipped** (321 s).
- 위 묶음 재실행: 78 passed, 1 failed. 실패한 것은 `test_parallel_gpu_launcher.py::test_script_runs_at_most_two_jobs_under_one_held_lock`이다.
  가짜 launcher의 시작 · 끝 시각 겹침(`peak == 2`)을 보는 시간 의존 test다. 단독으로 4/4 통과했고 전체 실행에서도 통과했다.
  이 STEP과 관계없는 기존 test의 시간 의존 flaky로 본다.
- probe 실행(raw): LIBERO 4 suite × 10 task + SafeLIBERO 8 × 4 task, xargs `-P 2`. suite마다 72–112 s.

```bash
O=outputs/verify/BM-SETUP-A
.venv-ag3s/bin/python $O/scripts/py38run.py benchmark/evaluators/suite_probe.py --suite safelibero_goal --safety-level II \
    --out $O/probe/safelibero_goal_LII.json --frames-dir $O/frames
.venv-ag3s/bin/python $O/scripts/py38run.py benchmark/evaluators/suite_probe.py --suite libero_90 --count-only --out $O/probe/libero_90_counts.json
.venv-ag3s/bin/python $O/scripts/summarize_probe.py        # -> probe_summary.json
.venv-ag3s/bin/python $O/scripts/timing_basis.py $O/timing_basis.json
.venv-ag3s/bin/python benchmark/runners/eval_matrix.py smoke --out-dir $O/smoke
```

## verifier 가 알아야 할 것

- 새 flag · 기본값: launcher / client `--safelibero-step-budget {total_incl_wait (기본), main_aegis}`. 기본값에서는 argv · settings · jsonl이 변경 전과 같다.
  launcher가 이제 task / episode 범위와 registry split label을 시작 전에 거절한다. 전에는 client가 server를 띄운 뒤 실패했다.
- 재생산이 필요한 산출물: 없음(GPU run 없음).
- 옛 기록과 호환: 깨지지 않음. 다만 commit 뒤에는 `eval_matrix` 재사용 규칙 5(code diff)가 옛 VLA-only run에 대해 `libero_safe_client.py`와
  `launch_libero_paired.py`의 diff를 보고한다. 받아들이려면 `--accept-diff`와 이 문서의 identity test 근거가 필요하다.
- raw: `outputs/verify/BM-SETUP-A/` — `probe/` (12 suite-level JSON + libero_90 개수), `probe_summary.json`, `frames/*.npz`
  (task마다 wait 뒤 head · wrist OpenCV RGB, head depth, head · wrist segmentation, model 이름 표, figure용), `logs/`, `scripts/`,
  `smoke/` · `smoke_main_aegis/` (jobs.txt · plan.json), `example_reuse_LI_dev/`, `timing_basis.json`.
- figure는 만들지 않았다(규칙 A의 figure는 verifier 몫). `frames/`가 scene figure의 입력이다.
- probe의 SafeLIBERO drive는 `env.sim.step`을 drive 동안만 감싸 substep 접촉을 기록하고, 끝나면 원래 method로 되돌린다(`del sim.step`).
- LIBERO-Safety(BM-SETUP-B) 공용 runner hook 요청: 작성 시점에 `BM-SETUP-B.impl.md`의 hook 절이 아직 비어 있었다(`__HOOKS__`). 반영한 것 없음.

## 사용자 결정이 필요한 것

1. **SafeLIBERO step budget.** 공식 `main_aegis.py`는 policy 300 / 550 step이고 우리 기본은 280 / 530이다.
   (a) 기본 유지: SL5c–SL5i 결과와 그대로 비교할 수 있다. 공식 TSR / ETS와는 척도가 다르다.
   (b) P4 전체 SafeLIBERO부터 `main_aegis` 사용: 공식 수치와 비교할 수 있다. VLA-only를 포함해 다시 돌려야 한다.
   (c) 둘 다: dev는 기존, test / 공식 비교는 main_aegis.
   2026-10-14 판정("wait 포함 t < 300")이 원문 code와 다르다는 점을 함께 판단해야 한다.
2. **2-물체 task(libero_10 6/10, safelibero_long 4/4)와 region · fixture target(libero_goal t0 · t7, libero_10 t2)**:
   GT 채점(ATS 정답)을 이 task로 넓힐 때 target 규칙을 정해야 한다(첫 물체만 / 모든 물체 / 제외). online 평가 자체에는 영향이 없다.
3. smoke 구성: V11AB 기준선 + 역변환 B로 만들었다(SL5c 기준선과 같은 구성). 최신 개선판(파지 인식 + closing 면제판, 역변환 C)으로도 smoke를 돌릴지.
4. 옛 VLA-only run 재사용: `38722284` 이후 VLA-only 경로 3 file이 바뀌었다. 재사용할지(`--accept-diff`와 근거) 아니면 VLA-only를 다시 돌릴지.

## 내가 기대하는 결과

<!-- verifier는 측정이 끝나기 전에 이 절을 읽지 않는다 -->
- smoke 12 job은 모두 client rc 0으로 끝나고, run 이름이 `plan.json`의 `run_suffix`와 맞으며, `aggregate`가 missing 없이 12 group을 낸다.
- SafeLIBERO smoke episode의 `safelibero.obstacle`은 probe의 active obstacle과 같다(같은 task · init 0).
- 실제 episode 시간은 시간 상한보다 짧다(성공 episode가 일찍 끝난다).
