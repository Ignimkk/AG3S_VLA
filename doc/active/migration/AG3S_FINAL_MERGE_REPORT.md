# AG3S 최종 구성 merge — feature/panda-benchmark (2026-10-29 KST)

## 무엇을 했나

- `origin/main` `3ff415d5` (AG3S 최종 구성, 사용자 판정 2026-10-29) 를 `feature/panda-benchmark` 에 `merge --no-ff` 했다.
- merge 직전 상태는 tag `pre-ag3s-final-merge` (`5ad4f791`) 다. commit 할 WIP 는 없었다 (untracked `main.pdf` = 논문 컴파일본, commit 하지 않음).
- 정본:
  - 최종 구성 spec: `doc/active/reviews/AG3S_decision/ADOPTED_KO.md` §A8 "A8 최종 구성 확정" 6.
  - 서버 인자: `AG3S/ag3s/docs/handoff/AG3S-CORE-EVAL-A6.impl.md` §3.
  - branch 설명: `doc/active/BRANCHES.md` "main 갱신", 이 merge 는 같은 문서의 "feature/panda-benchmark 에 main 반영".
- push 는 하지 않았다 (사용자 몫).

## 충돌 해결 원칙과 결과

시험 merge 에서 29 개 충돌이 났다 (코드 5, OpenPI 2, 환경 2, 규칙 · 문서 · migration 기록 20).

| 영역 | 해결 |
|---|---|
| `AG3S/trajopt` (`serve_safe` · `safe_policy` · `linearize` · `grasp_latch`) · `AG3S/ag3s/runtime/pipeline.py` | 양쪽을 다 살렸다. 우리 쪽 SL5a 연결 지점 (`--robot`, `layout` · `gripper_columns` · `finger_map` · `hands`, attention 주입) 과 V11AB-pb1~pb7 flag, main 쪽 최종 구성 flag 를 모두 둔다. 모두 기본 꺼짐. |
| `build_ag3s` 호출 | main 은 `serve_safe.main` 의 호출을 `ag3s_from_args()` 로 옮겼다. pb1 `reach_at_capture` 를 그 안으로 옮겼다. |
| pb1 대 T46g | 둘 다 SL5b reach 실패를 고친 다른 방법이다 (pb1 `--reach-at-capture` = benchmark, T46g `--latch-reach-hand-relative` = development). 둘을 같이 주면 T46g 가 pb1 의 reach 를 덮어쓰므로 `reject_bad_flag_combinations` 가 거절한다. |
| OpenPI `policy.py` · `serve_policy.py` | record 방식 하나 = benchmark 방식 (아래). |
| `scripts/setup/env.sh` · `environments/openpi-live/requirements.txt` | benchmark 환경 분리를 유지했다 (`AG3S_RESULTS_ROOT=results/benchmark`, 검증된 live pin). main 의 cache 경로 export 는 받았다. main 이 가져온 development venv 파일은 `environments/README.md` 에 참고용으로 구분해 적었다. |
| 규칙 · 로그 안내 (`CLAUDE.md`, `.claude/*`) | benchmark 쪽 기준. 이 checkout 의 정본 로그는 `AG3S_BENCHMARK_LOG.md` · `AG3S_BASELINE_LOG.md`, RB-Y1 로그 (`AG3S_T47_LOG.md` · `AG3S_T0T6_LOG.md`) 는 development 가 쓰고 여기서는 링크만 한다. |
| 문서 위치 | main 의 새 구조 (`doc/active`, `doc/archive`) 를 따랐다. benchmark 의 live 환경 보고서와 `validation/live/` 는 현재 문서라서 `doc/active/migration/` 에 두었다. `source-manifest.json` 은 새 위치에 benchmark 내용 그대로 두고 `ag3s_final_merge` 항목을 더했다. 이 경로를 읽는 benchmark 코드 (`run_artifacts` · `eval_matrix` · test) 를 고쳤다. |

### OpenPI record 방식 — benchmark 방식으로 정한 이유

- 정한 것: `serve_policy` 가 benchmark run (`common/run_artifacts.ensure_run`) 안의 새 폴더를 정한다. `PolicyRecorder` 는 이미 있는 폴더를 거절한다. main 의 thread lock 과 배타적 파일 생성 (`xb`) 은 받았다.
- 버린 것: main 의 `allocate_run_dir` + `session_*` 하위 폴더 방식.
- 이유:
  1. benchmark run manifest 는 commit 을 고정한다 (`record_execution` 이 다른 commit 의 run 을 거절). record 가 그 run 밖 (`results/development`) 으로 가면 이 연결이 끊긴다.
  2. 이미 있는 폴더를 거절하면 "record 폴더 하나 = 서버 시작 하나" 가 지켜진다. `session_*` 를 조용히 만들면 한 run 에 두 서버의 기록이 섞여도 알 수 없다.
  3. 동시 요청에서 step 번호가 겹치지 않게 하는 main 의 lock 은 이 두 원칙과 충돌하지 않으므로 받았다.
- test: `tests/test_policy_recorder_destination.py` (benchmark), `tests/test_policy_record_output.py` (main test 를 benchmark 방식으로 바꿈). `.venv-openpi-live` 에서 8 passed.

## V11AB 를 merge 전 서버로 지키기 위해 바꾼 것

merge 전 · 후 tree 에서 V11AB · pb1~pb7 argv 의 parse 결과 · TrajOpt config · `build_ag3s` kwargs · Panda AG3S config 를 직접 비교했다.

| 발견 | 조치 |
|---|---|
| 기존 키의 값 변화 | 없음 (경로 문자열만 다름) |
| main 이 ROI carve 를 cuRobo field 의 기본으로 켬 (`roi_carve_serving_default`). V11AB argv 를 그대로 주면 `build_kwargs.target_exclusion` 에 ROI carve 가 켜져 들어간다 | `profile.V11AB_PRE_MERGE_DEFAULTS` 에 `--no-roi-carve` |
| main 이 OpenBLAS 를 기본 1 thread 로 제한 (`--blas-threads 1`) | 같은 곳에 `--blas-threads 0` (library 기본 = merge 전) |
| Panda AG3S config 는 V11AB `build_ag3s` 조립의 복사본이라 main 이 더한 kwargs (cover · target exclusion · contact · confidence gate) 를 **조용히 버린다** | `profile.UNPORTED_BUILD_KEYWORDS` 가 켜진 값이면 거절 |
| Panda 서버는 자기 SafePolicy 를 짓는다. 최종 구성 flag 106 개 중 일부만 (`trajopt_config_from_args` 를 거치는 것) 적용되고 나머지는 무시된다 | `server.UNPORTED_FINAL_CONFIG_DESTS` 가 기본값이 아닌 값이면 시작 전에 거절 |
| profile 이 `serve_safe.main` 안의 `build_ag3s` 호출을 AST 로 찾는데, main 에서는 `ag3s_from_args` 로 옮겨져 Panda 서버가 시작하지 못함 | `_main_build_call` 이 두 배치를 모두 확인 (추측하지 않음), `**{...}` kwargs 처리 |
| main 이 `scripts/benchmark/v11ab/` (V11AB 서버 flag 의 정본 launcher) 를 지움 | 되살림 |
| main 의 `serve_safe` 는 시작할 때 `common/run_output.prepare_run_outputs` 로 `results/development` 밖의 `AG3S_RUN_DIR` 를 거절 → benchmark launcher 가 띄운 서버가 시작하지 못함 | `run_output.RESULTS_KIND = "benchmark"` (env.sh 의 `AG3S_RESULTS_ROOT` 도 같음) |
| RB-Y1 golden (`tests/trajopt/_sl5a_golden.py`) | SafePolicy 경로: 차이 12 개가 모두 상수 정의 위치 문자열의 줄 번호 (127 → 130). 줄 번호만 정규화. serve_safe 표면: 기존 키 값 변화 0, 추가 키 282 개, `ag3s_from_args` 로 옮겨진 호출식 3 개. 추가 키는 merge 시점 값 (`tests/trajopt/data/ag3s_final_merge_surface.json`) 과 같을 때만 비교에서 뺀다 (`PB_DEFAULT_OFF` 와 같은 방식) |

## 검증

### (1) test

| 묶음 | interpreter | 결과 |
|---|---|---|
| `tests/trajopt` 전체 | `.venv-ag3s` | 2183 passed, 36 skipped, 1 failed = `test_t43t_replay.py::test_off_is_bit_identical_to_the_pre_fix_commit` (development 의 G1 · HSR6 · A6 impl 에 같은 기록이 있는 기존 실패, `.venv-ag3s` numpy 주 버전의 golden 차이) — 33 분 |
| 위 실패 test 파일 | `.venv-openpi-live` | 9 passed |
| `tests/benchmark` + `tests/test_run_output.py` + `tests/test_run_artifacts.py` | `.venv-ag3s` | 252 passed, 2 skipped, 1 failed = `test_parallel_gpu_launcher.py::test_script_runs_at_most_two_jobs_under_one_held_lock` (가짜 job 1.5 s 가 겹치는지 보는 timing test, trajopt 전체 test 와 동시에 돌린 때). 부하 없이 단독 실행: 11 passed. merge 로 바뀌지 않은 파일이다 |
| OpenPI recorder test 2 파일 | `.venv-openpi-live` | 8 passed |

merge 로 고친 test: `tests/benchmark/test_panda_sl4.py` (served build call 위치 · 새 keyword 꺼짐 확인), `tests/test_run_output.py` (`RESULTS_KIND`), `tests/test_policy_record_output.py` (benchmark record 방식), `tests/trajopt/_sl5a_golden.py` (위 정규화), `tests/benchmark/test_eval_matrix.py` (manifest 경로).

### (2) RB-Y1 최종 구성 기록 재풀이

- 기록: storage `results/development/20261028T224856+0900_b3b69311f948_8784a1265958/verify/AG3S-CORE-EVAL/FIN_k0` (최종 구성 서버, 기록된 argv = A6 impl §3).
- 방법: development A6 A2 의 `a2_replay.py` 와 같은 재생 경로 (`tp_timing.run_timing` server mode, 기록된 서버 argv). CPU 라 GPU seed 만 끈다 (CPU 여러 초기값 16 은 켬). 같은 도구를 main `3ff415d5` tree 와 merge tree 에서 돌려 refined 를 비교했다.
- 요청: 서버 chunk 와 client planning row 를 대조해 episode 구간을 정했다. 장면 211 (chunk 0–74, 75 요청), 장면 119 (chunk 209–236, 28 요청; A6 창이 켜지는 쥐기 장면).

| 장면 | 요청 | refined bit 동일 (main = merge) | 요청별 기록 같음 | 다른 기록 키 | 쥐기 목표 anchor 켜짐 | 여러 초기값 발동 |
|---|---|---|---|---|---|---|
| 211 | 75 | **75** | 74 | 요청 16 의 `multi_seed.*_ms` (시간) 만 | 12 | 1 |
| 119 | 28 | **28** | 28 | — | 7 | 0 |

- 한계: 서버 원기록과의 비교는 판정에 쓰지 않았다. 원기록은 GPU seed (`--to-gpu-seeds curobo`) 로 풀렸고 이 재풀이는 끈 것이다 (development A2 와 같은 조건). 두 tree 모두 원기록과 bit 동일한 요청은 장면 211 에서 8 / 75, 장면 119 에서 1 / 28 로 같다.
- merge tree 와 main tree 의 AG3S 코드 차이는 benchmark 추가분 7 파일 (`serve_safe` · `safe_policy` · `linearize` · `grasp_latch` · `pipeline` · `config` · `sqp`, 모두 기본 꺼짐 기능) 뿐이다. GPU seed · cuRobo · field 코드는 main 과 같다.
- 도구 · 원시 결과: storage run (아래 "원시 기록").

### (3) Panda V11AB smoke

<!-- MERGE-SMOKE-RESULTS -->

## 원시 기록

<!-- MERGE-RAW -->

## 다음 (최종 구성을 Panda 에서 실험하기 전에 사용자 판정 필요)

1. HSR6 probe 는 RB-Y1 checkpoint 전용이다. Panda VLA 에서 재학습할지, 끌지. 끄면 HSR ④ 도 꺼져서 기하 grasp 보정 (G1) 과 쥐기 목표 항 (A4) 이 처음으로 작동하게 된다 (RB-Y1 최종 구성에서는 0 요청).
2. attention head `L10H7` 을 Panda VLA (`pi05_libero`) 에서 다시 고를지.
3. A6 손바닥 위치 (`left_ee` site, hold link 원점에서 100 mm 앞) · 손바닥 껍질 구를 Panda 손 (`panda_hand` · finger) 에 어떻게 대응시킬지.
