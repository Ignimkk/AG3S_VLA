# AG3S-PRE-CORE-TO E3-a — 구현 (GPU QP backend: OSQP CUDA)

> writer: ag3s-implementer · 읽는 쪽: lead, verifier, scribe
> 작성: 2026-10-26 03:35 UTC (`date -u`; 이 pod 의 KST = UTC + 9 h → 12:35 KST). 기준 코드: `feature/ag3s-development` HEAD `b3b69311` + E1 · E2 작업 트리 + 같은 시각 E3-b 작업 (다른 implementer) + 이 변경 (commit 안 함).
> 명세: [`AG3S-PRE-CORE-TO.task.md`](AG3S-PRE-CORE-TO.task.md) §6 "E3 구현 명세" 행 E3-a. 방식은 [`AG3S-PRE-CORE-TO-E2.impl.md`](AG3S-PRE-CORE-TO-E2.impl.md) 를 따랐다.

## 무엇을 했나 (평이한 요약 먼저)

기본 꺼짐 flag `--qp-backend {osqp,osqp-cuda}` (config `qp.solver`, 새 값 `"osqp-cuda"`) 를 넣었다. 켜면 TrajOpt 의 모든 QP — 표준 풀이, look-ahead 창 풀이, E2 multi-seed 의 seed 풀이 — 가 OSQP 1.1.3 의 CUDA algebra 로 풀린다. CUDA 풀이가 CPU 경로라면 실패로 볼 결과를 내거나 CUDA library 가 예외를 내면, 그 QP 하나를 CPU OSQP 로 다시 풀고 그 횟수를 `to.metrics.qp_backend` 에 센다.
꺼져 있으면 궤적이 HEAD 와 bit 단위로 같다 (E2 golden SHA 5 개, `.venv-ag3s`; `.venv-openpi-live` 에서는 `git archive HEAD` 사본과 같은 SHA). 기록 키도 같다 (`qp_backend` 키 없음, `to_dict()` 키 집합 그대로 — 기존 `qp.solver` 키의 값만 바뀐다).
offline 비교용 settings module `AG3S.trajopt.experiments.e3a_gpu_qp` (`E3a_0`, `E3a_16`, E2 settings 포함) 와 `summarize` (= `e2_multi_seed summarize` + CUDA 횟수) 를 만들었다.

**설치에서 생긴 일 (lead 지시로 고침):** `osqp-cu12` 를 venv 에 설치하면 osqp 1.1.3 이 **algebra 를 지정하지 않은 모든 `osqp.OSQP()` 의 기본을 CUDA 로 바꾼다** (우선순위 cuda > mkl > builtin, `CUDA_VISIBLE_DEVICES=''` 여도). 같은 venv 를 쓰는 다른 세션의 snapshot 코드가 조용히 GPU 로 갈 수 있어서, venv 에서 지우고 checkout 아래 별도 디렉터리에 설치한 뒤 이 backend 를 고를 때만 `sys.path` 에 붙인다. 아래 "설치 기록" 에 venv 에 있던 UTC 구간을 적었다.

## 설치 기록

| 시각 (`date -u`) | 무엇 | 기록 (`$RUN/verify/AG3S-PRE-CORE-TO/E3/a/install/`) |
|---|---|---|
| 02:32:04Z 직전 | `.venv-openpi-live` `pip freeze` | `pip_freeze_before.txt` |
| **02:32:04Z** (+ 몇 초) | `pip install --no-deps osqp-cu12==1.1.3` → venv 에 `osqp_cuda.cpython-311-x86_64-linux-gnu.so` | `install_time_utc.txt`, `pip_install.log`, `pip_freeze_after.txt` (차이: `osqp-cu12==1.1.3` 한 줄) |
| **02:34:11Z** | `pip uninstall -y osqp-cu12` (lead 지시) | `uninstall_time_utc.txt`, `pip_uninstall.log`, `pip_freeze_after_uninstall.txt` (= before 와 같음) |
| 02:34:1xZ | 새 process 에서 `osqp.OSQP().algebra` → `builtin`, `algebras_available()` → `['builtin']` 확인 | — |
| 02:34:3xZ | `pip install --no-deps --target /mnt/dev/AG3S_VLA-development/.cache/osqp-cu12 osqp-cu12==1.1.3` (`.cache/` 는 git-ignored) | `pip_install_target.log`, `pip_freeze_after_target_install.txt` (= before), `osqp_cuda_so.sha256` (`384eec93…5473`) |

- **venv 에 CUDA algebra 가 있던 구간: 2026-10-26 02:32:04Z – 02:34:11Z (KST 11:32:04 – 11:34:11).** 이 구간에 `.venv-openpi-live` 로 **새로 시작한** process 가 bare `osqp.OSQP()` 를 불렀다면 CUDA algebra 였다. 그 전부터 돌던 process (서버 · pi05_infer, 02:32 이전 시작) 는 영향 없음 (import 시점에 결정). 설치 직후 `ps` 로 본 `.venv-openpi-live` process 는 모두 10-18 · 10-25 시작이었다. 그 구간에 이 세션이 띄운 것은 probe 1 개 (CUDA 를 명시) 뿐이다.
- NFS 의 파일 mtime (`pip_install.log` 02:31:13) 은 pod 시계와 ~50 s 어긋난다. 위 표는 pod 의 `date -u` 다.
- `osqp_cuda.so` 는 `/usr/local/cuda/targets/x86_64-linux/lib` 의 `libcublas.so.12` · `libcusparse.so.12` · `libcudart.so.12` · `libnvJitLink.so.12` 에 link 된다 (`ldd`). torch 가 먼저 올린 `nvidia-*-cu12` wheel 의 같은 soname 이 있으면 그것을 쓴다 (같은 12.x ABI).
- `.venv-ag3s` · `.venv-curobo` 에는 설치하지 않았다. `.venv-ag3s` (osqp 1.1.3) 도 `.cache/osqp-cu12` 의 extension 을 import 할 수 있다 (CPU test 가 그 경로를 쓴다, device 없으면 fallback).

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/trajopt/qp.py:69–96` | E3-a 설명 (설치 위치, 설정 정렬과 그 근거 숫자, fallback) | |
| `AG3S/trajopt/qp.py:98–111` | `OSQP_CUDA_DIR_ENV` (`AG3S_OSQP_CUDA_DIR`), 기본 `<checkout>/.cache/osqp-cu12`, `_CUDA_ALIGNED` (CPU build 기본값으로 맞출 ADMM 설정 16 개), process 전체 횟수 `_CUDA_COUNTS` · `_CUDA_FAILED_STATUS` | |
| `AG3S/trajopt/qp.py:113–160` | `osqp_cuda_dir`, `_load_osqp_cuda` (필요할 때만 `sys.path.append`, 그 뒤 `osqp.interface.OSQP_ALGEBRA_BACKEND` 가 비어 있으면 `"builtin"` 으로 고정), `osqp_cuda_available`, `_cpu_defaults` (builtin build 에서 기본값을 읽음) | venv 오염 없이 CUDA 를 쓰고, 같은 process 의 bare `osqp.OSQP()` (예: `Baseline/knows_vla/cbf/filter.py`) 는 CPU 에 남게 |
| `AG3S/trajopt/qp.py:162–176` | `cuda_mark` · `cuda_record_since` | 횟수 차이 = chunk 기록 |
| `AG3S/trajopt/qp.py:209–219` | `reset` 이 fallback solver 도 reset | |
| `AG3S/trajopt/qp.py:232–233` | `solve`: `"osqp-cuda"` → `_solve_osqp_cuda` | `"osqp"` 분기는 글자 그대로 |
| `AG3S/trajopt/qp.py:242–262` | `_solve_osqp(…, algebra="builtin")`: `osqp.OSQP(algebra=algebra)` 로 **항상 이름을 준다**; CUDA 면 setup 에 `_cpu_defaults()` 를 먼저 넣고 setup 횟수를 센다. 나머지 (update · warm start · eps 갱신 · 상태 판정 · `accept_max_iter`) 는 두 algebra 가 같은 코드 | 기본 경로의 유일한 변화 = `algebra="builtin"` 명시 (lead 지시 3) |
| `AG3S/trajopt/qp.py:322–352` | `_solve_osqp_cuda`: CUDA 시도 → 성공이면 반환; 실패 (CPU 경로의 성공 판정과 같은 기준) 또는 예외 (`ImportError` 제외 — 설정 오류로 그대로 올림) 면 `_cpu_fallback` (`solver="osqp"` 인 `QpSolver`, 처음 실패 때 만듦) 에 같은 `_last_x/_last_y` 를 주고 다시 풂. 반환 status `"<cpu status> (cpu fallback; cuda: <cuda status>)"`, iterations = CUDA + CPU | 명세 "실패하면 CPU OSQP 로 다시 푼다(기록)" |
| `AG3S/trajopt/qp.py` 끝 `__all__` | 새 이름 | |
| `AG3S/trajopt/config.py:759, 801–803` | `QpConfig.SOLVERS` 에 `"osqp-cuda"` | 새 config 키 없음 |
| `AG3S/trajopt/config.py:814–817` | `solver="osqp-cuda"` + `polish=True` → `TrajOptConfigError` | CUDA algebra 는 간접 (PCG) 풀이뿐이고 OSQP polish 는 직접 풀이가 필요 (probe: `status_polish -1`) |
| `AG3S/trajopt/serve_safe.py:1674–1678` | `--qp-backend {osqp,osqp-cuda}` (기본 `osqp`, help 한국어) | |
| `AG3S/trajopt/serve_safe.py:2816` | `--no-safe` 와 함께 `osqp-cuda` → `ap.error` (기존 `to_flags` 목록에 한 줄) | TO 없는 서버에 TO flag |
| `AG3S/trajopt/serve_safe.py:2836–2841` | `osqp-cuda` 인데 extension 을 import 할 수 없으면 `ap.error` (체크포인트 올리기 전) | 조용히 CPU 로 돌지 않게 |
| `AG3S/trajopt/serve_safe.py:3064–3073` | `qp_overrides`: `osqp-cuda` 면 `{"solver": "osqp-cuda"}` (+ `--qp-eps` 와 합침), `osqp` 면 키 없음 | |
| `AG3S/trajopt/serve_safe.py:3223–3229` | 시작 log 한 줄 (WARNING, 켰을 때만) | |
| `AG3S/trajopt/refiner.py:164–170, 201–204` | `self.config.qp.solver == "osqp-cuda"` 일 때만 solve 전후 `cuda_mark` → `result.metrics["qp_backend"] = cuda_record_since(mark)` | 한 chunk 의 모든 QP (표준 · 창 · seed · TOA3 대체 풀이 · fallback) 를 한 곳에서 센다. sqp.py (E2 · E3-b 가 고치는 중) 는 건드리지 않았다. 명세의 "config.py / serve_safe.py 만" 밖이지만 기록 자리가 여기밖에 없어 최소로 넣었다 |
| `AG3S/trajopt/experiments/tp_timing.py:455–457` | CUDA setting 이면 chunk 출력에 `qp_backend` | 같은 이유, 꺼진 setting 출력 그대로 |
| `AG3S/trajopt/experiments/e3a_gpu_qp.py` (새 파일, 95 줄) | `SETTINGS` = `e2_multi_seed.SETTINGS` + `E3a_0` · `E3a_16` + smoke 전용 `SMOKE_IT1_E3a_0/16`, `summarize [--base NAME]` | offline 비교 |
| `tests/trajopt/test_e3a_gpu_qp.py` (새 파일) | 11 test | 아래 |

`multi_seed.py`, E2 선택 코드, `sqp.py`, `Baseline/**`, 기존 test 의 bare `osqp.OSQP()` 는 건드리지 않았다 (lead 지시 3).

## flag 의미

| flag | config | 뜻 |
|---|---|---|
| (없음) / `--qp-backend osqp` | `qp.solver = "osqp"` (기본) | 지금과 같다. `qp_overrides` 에 키 없음, `to.metrics.qp_backend` 없음 |
| `--qp-backend osqp-cuda` | `qp.solver = "osqp-cuda"` | 이 optimizer · 창 optimizer · seed solver (`_ms_solver = QpSolver(self.config.qp)`) 의 모든 QP 가 CUDA. `to.metrics.qp_solver = "osqp-cuda"`, `to.metrics.qp_backend` 가 생긴다 |

`to.metrics.qp_backend` (서버 기록에서는 E2 처럼 `_scalar` 문자열, offline 은 dict):

```
{backend: "osqp-cuda", qp_solves, cuda_solved, cuda_failed, cuda_errors, cuda_setups,
 cpu_fallback_solved, cpu_fallback_failed, failed_statuses: {status: n}}
```

- `qp_solves` = 이 chunk 에서 CUDA backend 로 들어온 QP 수. `cuda_failed` 는 예외 (`cuda_errors`) 를 포함한다. `cpu_fallback_solved + cpu_fallback_failed = cuda_failed`.
- `cuda_setups` = 새 CUDA OSQP setup 수 (처음 · sparsity 가 바뀔 때). 횟수는 process 전체 counter 의 차이라, 같은 process 에서 TrajOpt 가 두 thread 로 동시에 돌면 섞인다 (서버 · tp_timing 은 순차).

## CPU 와 CUDA 의 차이

| 항목 | CPU (`osqp`) | CUDA (`osqp-cuda`) |
|---|---|---|
| 설정으로 준 값 (`eps_abs` · `eps_rel` · `max_iter` · `polish` · `adaptive_rho` · `warm_starting` · `verbose`) | config | **같은 값** (`eps_after_first` 의 `update_settings` 도 같음) |
| build 마다 기본이 다른 ADMM 설정 | builtin 기본: `check_dualgap 1`, `check_termination 25`, `adaptive_rho_tolerance 5`, `rho_is_vec 1` | CUDA build 기본은 `0`, `5`, `2`, `0` → **CPU build 기본값을 setup 에 넣는다** (`_CUDA_ALIGNED` 16 개; 나머지 10 개는 원래 같은 값) |
| 정밀도 | float64 | **float32** (설정값도 float32 로 저장: 1e-3 → 0.0010000000475). 해는 float64 로 바꿔 반환 |
| 선형계 | QDLDL (직접) | **PCG (간접)**, `cg_*` 기본 그대로 |
| polish | 설정대로 | 불가 → config 가 거절 |
| warm start | setup 이 같으면 update 후 `warm_start(last x, last y)` (또는 `x0`) | 같은 코드. float64 배열을 넘기고 extension 이 변환 |
| 첫 setup | ~2 ms | CUDA context · cuBLAS 초기화로 process 의 첫 setup 이 ~1.9 s (probe, 공유 lock, 판정 아님), 그 뒤 setup ~8 ms |
| 실패 | trust region 축소 (SQP) | 같은 QP 를 CPU 로 다시 풀고 그 결과를 쓴다. 그래도 실패면 CPU 와 같은 실패 경로 |
| fallback 의 warm start | — | CUDA 시도와 같은 `_last_x/_last_y` 를 주지만, fallback solver 가 **새로 setup 될 때** (처음 · sparsity 변경) 는 OSQP 규칙대로 cold start (기존 CPU 경로도 새 setup 은 cold) |

**왜 설정을 맞췄나 (probe, RB-Y1 fixture QP 1014 변수 · 2210 행, 공유 lock):** 정확한 최적 (CPU eps 1e-8 + polish) 의 목적값 −29.659. CPU eps 1e-3: −29.468, 최적에서 max |Δx| 0.080. CUDA 를 그대로 두면 `solved` 라고 하면서 목적값 **−2226** (max |Δx| 0.148, dual 잔차 6.9) — 상대 dual 기준 (`eps_rel · ‖q‖∞`, ‖q‖∞ = 1e4) 이 느슨한데 duality gap 검사가 꺼져 있어서다. CPU 기본값으로 맞추면 −29.760 / 0.039 (scalar rho) · **−29.715 / 0.035 (vector rho, 채택)** — CPU 보다 최적에 가깝다. 대신 ADMM 반복이 350 → 1775 로 는다 (float32 + PCG).

## 단위 검증

```bash
cd /mnt/dev/AG3S_VLA-development
# 새 test, CPU (GPU test 3 개 skip)
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=osmesa PYTHONPATH=. \
  .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider tests/trajopt/test_e3a_gpu_qp.py
# 새 test, GPU (공유 lock)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=osmesa PYTHONPATH=. \
  python3 scripts/setup/gpu_lock.py shared --est-gib 8 --tag e3a-impl -- \
  .venv-openpi-live/bin/python -m pytest -q -p no:cacheprovider tests/trajopt/test_e3a_gpu_qp.py
# trajopt 전체
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=osmesa PYTHONPATH=. \
  .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider tests/trajopt \
  --deselect tests/trajopt/test_t43t_replay.py::test_off_is_bit_identical_to_the_pre_fix_commit
```

- `.venv-ag3s` (CPU): **8 passed, 3 skipped** (GPU 3 개; `__file__` = 이 checkout 확인).
- `.venv-openpi-live` (GPU, 공유 lock): **10 passed, 1 skipped** (`test_no_device_falls_back_without_crashing` 은 device 가 보이면 skip — 반대 venv 에서 돈다).
- trajopt 전체 (`.venv-ag3s`, 02:47–03:10Z, 22 min 57 s): **1961 passed, 31 skipped, 1 deselected, 1 failed.** 실패 1 개 `test_to_only_policy.py::test_the_no_perception_path_builds_no_ag3s_and_no_attention_copy` 는 `inspect.getsource(serve_safe.build_to_only_policy)` 가 **다른 함수의 본문** (`trajopt_config_from_args`) 을 돌려준 것 — E3-b implementer 가 실행 중 (02:50:14Z) `serve_safe.py` 를 고쳐 줄 번호가 밀린 경쟁이다. 그 파일만 다시 돌리면 **35 passed**. 따라서 실제 결과는 1962 passed / 31 skipped / 0 failed 로 본다. (E2 의 1974 passed · 30 skipped 는 `tests/ag3s/test_t46_target_free_residual.py` 를 포함한 수다; 그 파일 + `test_qp.py` + `test_e2_multi_seed.py` 를 따로 돌림: 44 passed, 2 skipped.) log `$RUN/verify/AG3S-PRE-CORE-TO/E3/a/tests/pytest_trajopt_venv-ag3s.log`.
- `git archive HEAD` 사본을 `.venv-openpi-live` 로 돌린 기본 fixture SHA 5 개 = 이 checkout 의 SHA 5 개 (`7a29e7a7…`, `c1b90fc8…`, `47cff433…`, `f251d424…`, `038dfc50…`). `.venv-ag3s` 의 값은 E2 golden 과 같다 (`2a2b43de…` 등) — venv 마다 BLAS · numpy 가 달라 값이 다르다.

| test | 내용 |
|---|---|
| `test_default_backend_is_bit_identical_and_named_builtin` | 기본 fixture 5 개 (표준, 막힌 fixture 연속 2, refiner 2 chunk) 의 SHA 가 HEAD 식 호출 (bare `osqp.OSQP()`) 과 같고, `.venv-ag3s` 에서는 E2 golden 과 같다. 모든 생성이 `algebra="builtin"`. `qp_backend` 키 없음. CUDA extension 을 같은 process 에 올린 뒤에도 bare `osqp.OSQP().algebra == "builtin"` 이고 SHA 같음 |
| `test_cuda_solves_a_small_qp_to_the_cpu_solution` (GPU) | 20 변수 강볼록 QP, eps 1e-5: GPU 가 `solved` (fallback 없음), max |Δx| ≤ 1e-3, 목적값 상대 차 ≤ 1e-4, 값만 바뀐 재풀이는 setup 을 다시 하지 않음 |
| `test_cuda_on_the_trajopt_fixture_qp_and_the_aligned_settings` (GPU) | CUDA solver 의 `_CUDA_ALIGNED` 값 = CPU build 기본값 (float32 rel 1e-6), eps · max_iter · adaptive_rho · warm_starting · polish 그대로; fixture QP 목적값 5 % 안, 제약 잔차 ≤ 5e-3 |
| `test_cuda_optimizer_solve_matches_the_cpu_outcome` (GPU) | 같은 fixture 의 SQP 풀이: 상태 같음, 둘 다 침범 ≤ 0.1 mm, 궤적 차 < 0.05 rad, `qp_solver` 기록 `osqp-cuda` |
| `test_failed_cuda_solve_is_resolved_on_the_cpu_and_recorded` | CUDA 가 `primal infeasible` 을 내면: 반환 = CPU 해 (bit 같음), status 표시, 횟수 기록 (전체 dict 비교). CUDA 가 예외를 내면: 다음 CPU 해 (warm start 같음) 와 bit 같음, `cuda_errors`, CUDA backend 다시 setup, `reset` 이 fallback 도 비움 |
| `test_no_device_falls_back_without_crashing` (device 없을 때) | 실제 `OSQPException` ("No GPU detected") → fallback, 기록 |
| `test_missing_extension_is_a_configuration_error` | 새 process: extension 경로가 없으면 `osqp_cuda_available` False + 설치 명령, 풀이는 `ImportError` (CPU 로 조용히 돌지 않음), bare `osqp.OSQP()` 는 builtin |
| `test_standard_and_seed_solves_all_go_through_the_cuda_path` | (CUDA 호출을 builtin 으로 바꾼 가짜로) 막힌 fixture + `multi_seed 16`: 표준 1 + seed 16 = 17 QP 모두 CUDA 경로, 결과 chunk 는 CPU 와 bit 같음, seed solver · 주 solver 모두 `osqp-cuda`, 기록 키는 `qp_backend` 하나만 는다 |
| `test_config_and_cli` | `SOLVERS`, polish 거절, 모르는 값 거절, `to_dict` 키 집합 같음, `--qp-backend` 기본 · `osqp` → 키 없음 (config 전체 같음), `osqp-cuda` → `{"solver": "osqp-cuda"}`, `--qp-eps` 와 합침, 잘못된 선택 거절, `--no-safe` 와 함께 · extension 없음 → `ap.error` |
| `test_offline_settings_are_registered_and_valid` | `E3a_0` = E0 + CUDA, `E3a_16` = `E2_16` + CUDA (다른 절은 같음), E2 settings 불변 |
| `test_summarize_is_e2_plus_the_backend_counts` | `per_setting` · `per_run` 은 `e2_multi_seed summarize` 와 같고, `qp_backend` 합계 (전체 · E0 침범 chunk) |

## offline smoke (1 episode, 판정 아님, 시간 주장 없음)

```bash
RUN=/mnt/dev/AG3S_VLA-storage/results/development/20261025T191319+0900_35b909cf096a_cc215238f581
O=$RUN/verify/AG3S-PRE-CORE-TO/E3/a/offline_smoke
bash $O/run_smoke.sh > $O/AG3S_case3_L3_s13453.log 2>&1      # 공유 lock, --variants E0 E2_16 E3a_0 E3a_16 --field cuda
.venv-ag3s/bin/python -m AG3S.trajopt.experiments.e3a_gpu_qp summarize $O/AG3S_case3_L3_s13453.json > $O/summary.json
```

입력 `OCC/AG3S_case3_L3_s13453` (S-OCC, 46 chunk), `--mode server --replay-wrist-track --field cuda`. 산출 (`$O`): `AG3S_case3_L3_s13453.json` · `.log` (46 chunk, 2165 s), `summary.json`, `links.txt`, `run_smoke.sh`.

| setting | E0 침범 chunk | 해소 (≤ 0.1 mm) | 고른 seed | 침범 중앙값 · 최대 (mm) | CUDA QP 수 | CUDA 실패 → CPU fallback (성공 / 실패) |
|---|---|---|---|---|---|---|
| `E0` | 35 | 0 | — | 11.46 · 13.31 | — | — |
| `E2_16` | 35 | **35** | detour 35 | 0 · 0 | — | — |
| `E3a_0` | 35 | 0 | — | 11.46 · 13.31 | 138 | 2 (2 / 0), 모두 `maximum iterations reached` |
| `E3a_16` | 35 | **35** | detour 35 | 0 · 0 | 1817 | **468 (359 / 109)**, 모두 `maximum iterations reached` |

- 해소 결과는 E2 와 같다 (35 / 35, 모두 detour). 이 episode 에는 probe gate 가 목표를 만든 chunk 가 없다 (`n_with_goal` 0) — 목표 오차 기준은 이 smoke 로 보지 못했다.
- 궤적 SHA: `E0` = `E3a_0` 46 중 34 chunk (QP 후보가 하나도 수락되지 않아 같은 초기 iterate 가 나간 chunk 로 본다; 확인 안 함), `E2_16` = `E3a_16` 2 chunk. float32 · PCG 라 같지 않은 것이 정상이다.
- **seed 풀이에서 CUDA QP 의 26 % (468 / 1817) 가 4000 반복 상한에 닿았다.** 표준 풀이 (`E3a_0`) 는 1.4 % (2 / 138). seed 의 첫 QP 는 warm start (이 solver 의 이전 해) 가 seed 의 시작점과 멀어 float32 PCG-ADMM 이 4000 반복 안에 수렴하지 못한 것으로 본다 (확인 안 함). fallback 의 CPU 도 그중 109 개를 못 풀었다 (CPU `E2_16` 이 같은 QP 를 풀었는지는 기록이 없어 모른다).
- **시간 (공유 lock, 판정 아님):** wall 중앙값 · p95 — `E0` 64 · 136 ms, `E3a_0` 442 · 4195 ms, `E2_16` 3432 · 4259 ms, `E3a_16` **54 662 · 60 677 ms**. 발동 chunk 의 multi-seed ms 중앙값 `E2_16` 3543 · `E3a_16` 54 617. 같은 시각 pytest 전체 (CPU 1 process) 가 돌았다. CUDA 쪽이 한 자릿수 이상 느리다는 방향은 lock 종류로 뒤집히기 어렵다고 보지만, 숫자는 lead 의 독점 lock 측정이 정본이다.
- **link:** 02:47:28Z 에 만들었고 tp_timing 은 03:23:34Z 에 끝났다. EXIT 의 "다른 tp_timing 이 읽는가" 검사 (`pgrep -f "tp_timing.*occ_index.json"`) 가 **자기 부모 shell** (명령 문자열에 그 글자가 있었다) 에 걸려 link 를 남겼다. `ps` 로 python tp_timing process 가 없음을 확인하고 03:31:46Z 에 손으로 지웠다 (끝난 뒤 8 분 남아 있었음; 그 사이 이 run 을 쓰는 서버가 시작했는지는 모름). 아래 full 명령은 python process 만 보는 pattern (`^\S*python\S* -m AG3S.trajopt.experiments.tp_timing .*occ_index`) 을 쓴다. 지금 run 안의 link 는 `snapshot_e1/e2/m1/third_party/openpi/third_party/{aloha,libero}` 6 개뿐 (이전부터 있던 것).

## full 비교 명령 (lead 가 실행)

`E0` · `E2_16` (CPU QP) 과 `E3a_0` · `E3a_16` (CUDA QP) 을 같은 pass 에 넣으면 `summarize` 가 같은 파일의 E0 와 chunk 별로 비교한다. 시간을 재려면 **독점 lock** 으로 (아래 `exclusive`). CPU 와 GPU setting 을 한 pass 에 섞으면 tp_timing 이 setting 순서를 chunk 마다 돌리므로 공정하지만, 판정용 시간이 필요하면 `E2_16` 과 `E3a_16` 만 넣는 것도 된다.

**S-OCC (OCC round 4, 18 run, E0 침범 68 chunk) — link 를 만들고, 끝나면 지운다:**

```bash
RUN=/mnt/dev/AG3S_VLA-storage/results/development/20261025T191319+0900_35b909cf096a_cc215238f581
IDX=$RUN/verify/AG3S-PRE-CORE-TO/E1/occ_index/occ_index.json
O=$RUN/verify/AG3S-PRE-CORE-TO/E3/a/offline_occ
mkdir -p $O
cd /mnt/dev/AG3S_VLA-development
bash $RUN/verify/AG3S-PRE-CORE-TO/E1/occ_index/make_links.sh
trap 'pgrep -f "^\S*python\S* -m AG3S.trajopt.experiments.tp_timing .*occ_index" >/dev/null || bash $RUN/verify/AG3S-PRE-CORE-TO/E1/occ_index/make_links.sh --remove' EXIT
for key in $(python3 -c "import json;print(' '.join(json.load(open('$IDX'))['runs']))"); do
  n=$(basename $key)
  MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/AG3S_VLA-development OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= \
    python3 scripts/setup/gpu_lock.py exclusive --est-gib 8 --tag e3a-occ-$n -- \
    /mnt/dev/AG3S_VLA-development/.venv-openpi-live/bin/python -m AG3S.trajopt.experiments.tp_timing --index $IDX --key $key \
    --settings-module AG3S.trajopt.experiments.e3a_gpu_qp \
    --variants E0 E2_16 E3a_0 E3a_16 \
    --field cuda --mode server --replay-wrist-track --out $O/$n.json > $O/$n.log 2>&1 || echo "FAIL $key" >> $O/fail.txt
done
pgrep -f "^\S*python\S* -m AG3S.trajopt.experiments.tp_timing .*occ_index" >/dev/null || bash $RUN/verify/AG3S-PRE-CORE-TO/E1/occ_index/make_links.sh --remove; trap - EXIT
find $RUN -type l | head    # 비어 있어야 한다 (서버가 밖을 가리키는 link 가 있는 run 을 거절한다)
/mnt/dev/AG3S_VLA-development/.venv-ag3s/bin/python -m AG3S.trajopt.experiments.e3a_gpu_qp summarize $O/*.json > $O/summary.json
touch $O/DONE
```

**S-REG (E4 18 장면, E0 침범 2 chunk):**

```bash
RUN=/mnt/dev/AG3S_VLA-storage/results/development/20261025T191319+0900_35b909cf096a_cc215238f581
IDX=$RUN/verify/AG3S-GPU-STAGES/M2_cpu/e4_index.json
O=$RUN/verify/AG3S-PRE-CORE-TO/E3/a/offline_e4
mkdir -p $O
cd /mnt/dev/AG3S_VLA-development
for key in $(python3 -c "import json;print(' '.join(json.load(open('$IDX'))['runs']))"); do
  n=$(basename $key)
  MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/AG3S_VLA-development OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= \
    python3 scripts/setup/gpu_lock.py exclusive --est-gib 8 --tag e3a-e4-$n -- \
    /mnt/dev/AG3S_VLA-development/.venv-openpi-live/bin/python -m AG3S.trajopt.experiments.tp_timing --index $IDX --key $key \
    --settings-module AG3S.trajopt.experiments.e3a_gpu_qp \
    --variants E0 E2_16 E3a_0 E3a_16 \
    --field cuda --mode server --replay-wrist-track --out $O/$n.json > $O/$n.log 2>&1 || echo "FAIL $key" >> $O/fail.txt
done
/mnt/dev/AG3S_VLA-development/.venv-ag3s/bin/python -m AG3S.trajopt.experiments.e3a_gpu_qp summarize $O/*.json > $O/summary.json
touch $O/DONE
```

- `summary.json`: `e2_multi_seed summarize` 의 모든 키 (`per_setting[v]`: `n_base_violated`, `n_resolved`, **`share_resolved_goal_le_10mm`**, `chosen_kind` · `chosen_kind_when_resolved`, `solve_ms_*`, `n_triggered`, **`multi_seed_ms_median/p95/max`** …, `per_run`, `all_chunks`) + **`qp_backend[v]`** (`all` · `base_violated`: chunk 수, `qp_solves`, `cuda_solved`, `cuda_failed`, `cuda_errors`, `cuda_setups`, `cpu_fallback_solved/failed`, `failed_statuses`).
- 판정에 쓸 시간: `E3a_16` 의 `multi_seed_ms_median` (발동 chunk 의 표준 + seed 전체 ms) 와 `solve_ms_median` (E0 침범 chunk) — 기준 533 ms. fallback 이 많으면 그 CPU 시간이 시간에 들어 있다 (`qp_backend.base_violated.cpu_fallback_*` 와 같이 본다).
- `--field cuda` 는 거리장도 GPU 에 둔다. `--field cpu` 로 바꾸면 E2 와 같은 장 경로에서 QP 만 GPU 다.

## verifier 가 알아야 할 것

- 새 flag: `--qp-backend {osqp,osqp-cuda}` (기본 `osqp`). config 는 기존 키 `qp.solver` 에 새 값 `"osqp-cuda"` — `to_dict()` 키 집합 그대로.
- 꺼짐이면 `to.metrics` 키 · 응답 · 기록이 같다. 켜면 `to.metrics.qp_solver = "osqp-cuda"` 와 `to.metrics.qp_backend` 가 생기고, fallback 한 QP 의 status 문자열에 `(cpu fallback; cuda: …)` 가 붙는다 (SQP 는 실패 QP 의 status 만 notes 에 쓰므로, fallback 이 성공하면 notes 에는 안 보인다 — 횟수는 `qp_backend` 로 본다).
- **CUDA extension 위치:** `/mnt/dev/AG3S_VLA-development/.cache/osqp-cu12` (git 밖). 다른 checkout · worktree 에서 쓰려면 `AG3S_OSQP_CUDA_DIR` 을 이 경로로 준다 (기본은 `<그 checkout>/.cache/osqp-cu12`).
- **같은 process 의 다른 OSQP 사용자:** 이 backend 를 쓰면 `osqp_cuda` 가 import 되므로, `OSQP_ALGEBRA_BACKEND` 를 `builtin` 으로 고정해 bare `osqp.OSQP()` (`Baseline/knows_vla/cbf/filter.py:423` 등) 가 CPU 에 남게 했다. 환경변수 `OSQP_ALGEBRA_BACKEND` 를 누가 이미 줬다면 그 값을 존중한다.
- 재생산이 필요한 산출물: 없음. 옛 기록과 호환: 깨지지 않음.

## 확인하지 못한 것 · 확실하지 않은 것

0. **seed 풀이의 CUDA 반복 상한 (smoke 26 %).** fallback 이 성공해도 그 QP 는 CUDA 4000 반복 + CPU 를 모두 쓴다. `max_iter` 를 낮추고 `accept_max_iter` 를 켜는 것, seed 의 첫 QP 를 seed 시작점에서 warm start (`qp.warm_start_iterate`) 하는 것이 줄일 후보지만 명세 밖이라 만들지 않았다 (기존 flag `--qp-max-iter` · `--qp-accept-max-iter` · `--qp-warm-iterate` 의 config 키로 offline setting 에 바로 넣을 수 있다).
1. **시간.** 공유 lock 의 probe 에서 CUDA QP 하나가 CPU 보다 훨씬 느렸다 (fixture QP: CPU 10 ms / 350 반복, CUDA (정렬) 1.5 s / 1775 반복; 정렬하지 않으면 67 ms 지만 답이 틀림). 1000 변수 급 QP 는 GPU 에서 kernel launch 가 지배한다. 판정용 시간은 lead 가 독점 lock 으로 잰다. 정렬이 시간을 많이 쓰므로 `check_termination` (25 → 5) 처럼 답에 영향이 적은 것만 CUDA 기본으로 돌리는 변형이 필요할 수 있다 — 명세 ("같은 ADMM 설정") 밖이라 만들지 않았다.
2. **fallback 의 기준.** 명세 문장 "non-solved status" 를 CPU 경로의 성공 판정과 같게 읽었다: `solved` · `solved inaccurate` · (`accept_max_iter` 일 때) `maximum iterations reached (accepted)` 는 CUDA 결과를 쓴다. `solved inaccurate` 도 CPU 로 다시 풀어야 한다면 `_solve_osqp_cuda` 의 `if cuda.success` 한 줄을 `cuda.status == "solved"` 로 바꾸면 된다.
3. **fallback 이 시간 예산을 넘길 수 있다.** CUDA 실패 (대개 `maximum iterations reached`, 4000 반복) 뒤 CPU 로 다시 풀므로 그 QP 시간 = CUDA 4000 반복 + CPU. SQP 의 `time_budget_ms` 는 반복 사이에서만 본다.
4. 첫 CUDA setup (~1.9 s, CUDA context) 이 첫 TO chunk 에 들어간다. 서버에서 warm-up 은 넣지 않았다. offline 에서는 각 setting 의 첫 발동 chunk 의 시간에 들어간다 (중앙값에는 영향 작음, 최대값에는 들어감).
5. sparsity 가 바뀔 때마다 CUDA OSQP 를 새로 setup 한다 (CPU 와 같은 규칙, ~8 ms). seed solver 와 주 solver 가 각각 GPU 메모리를 잡는다 (수 MB).
6. float32: `_INF = 1e20` 은 float32 로 표현되고 CUDA 의 `OSQP_INFTY` (1e17) 보다 커서 무한으로 처리된다. 비용 계수 (‖q‖∞ 1e4 = `w_slack`) 와 관절 각 (1e-3 rad) 의 비가 float32 정밀도 (~1e-7) 안이라 문제 없다고 보지만, 실기록 QP 전체로 확인하지는 않았다 — smoke 와 offline 의 `cuda_failed` 로 본다.
7. `.venv-ag3s` 에서도 CUDA extension 이 import 된다 (device 없으면 `OSQPException` → fallback). CPU 회귀 venv 에서 `--qp-backend osqp-cuda` 를 쓰는 사람은 없겠지만, 그러면 모든 QP 가 fallback 으로 CPU 에서 풀린다 (`cuda_errors` 로 보인다).
8. 그림은 만들지 않았다 (verifier 몫).

## 내가 기대하는 결과

> verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.

- `E3a_16` 의 해소 비율은 `E2_16` 과 거의 같을 것이다 (같은 seed, QP 해만 float32 / PCG 로 조금 다름). 시간은 실행 창 533 ms 를 넘을 것으로 본다 — QP 하나가 이 크기에서 GPU 쪽이 더 느리고, seed 를 순차로 푸는 구조는 그대로라서다. E3-a 가 시간 기준을 넘으려면 seed 들을 한 batch 로 푸는 것 (E3-b 의 방식) 이 필요할 것이다.
