# T55_windows — 구현 (T60 종료 구간 P · H 보조 채점: T55 실행, T56 준비)

> writer: ag3s-implementer (A1) · 2026-10-14 19:40 KST · 읽는 쪽: verifier, scribe, lead
> 근거: 사용자 판정 2026-10-14 (`T55.task.md` · `T56.task.md` 끝 절, e4 세션 경유): **주 판정은 기존 전체 구간**이다. P · H 는 보조 분석으로 나란히 보고한다. mask 정의는 분석 전에 고정하고 모든 arm 에 같게 적용한다. 주 창은 바꾸지 않는다. T55 는 해제 · 복귀 이후 동작도 평가하므로 P 에서 자른 결과가 전체 평가를 대신하지 않는다.
> HEAD `284366d2` (새 파일은 아직 commit 안 함). **새 파일만 만들었다.** 기존 `t54_*` · `t55_*` · `t56_*` · `t60_*` 는 한 줄도 바꾸지 않았다 (md5 는 아래 표, 작업 전후 같음).

## 무엇을 했나 (평이한 요약 먼저)

T60 이 T53 16 episode 에 만든 종료 mask (P = 놓기 완료, H = 준비자세 복귀 완료) 를 T55 test 채점에 걸어 보는 보조 도구를 만들었다. 채점 함수는 `t55_study.score_test` 를 그대로 부르고, trace 마다 "채점할 frame" (`Trace.window`) 만 mask 와 AND 한다.
- **P · H 를 쓰기 전에** 전체 구간을 다시 채점한다. 기존 `test_results.json` 과 **완전히 같아야** 다음으로 넘어간다. 실제로 같았다 (latch mode 3 개 모두, bootstrap CI · trace 별 count · natural_events 까지).
- 그 뒤 P · H 에서 조건 × arm × predicate 표와 **합산 (모든 조건) P_task 판정 행** 을 냈다. 합산 행은 trace count 를 episode block 별로 먼저 더하고 나눈다 (lead 의 합산 정의). CI 는 episode block bootstrap (B 10000, seed 5507) 이고, prereg 허용폭을 기계적으로 확인한다.
- T56 용 같은 도구 (`t56_windows.py`) 도 만들었다. T56 test 출력이 아직 없으므로 **돌리지 않았고**, synthetic test 만 했다.

## mask 적용 규칙 (분석 전에 고정, 코드 docstring 과 `MASK_RULE` 에 같은 문장)

| 항목 | 규칙 |
|---|---|
| mask 원천 | T60 step1 `masks/T53__C1a_<key>_s<seed>.npz` (protocol `T60-v1`). frame f 는 `t_step[f] ≤ t_end` 이면 유지한다. P (또는 H) 에 도달하지 않은 run 은 75 frame 을 모두 유지한다. 파일의 keep 배열을 `t60_termination.keep_frames(chunk_start_steps, t_P/t_H)` 로 다시 만들어 같은지 확인한다 (`t55_windows.py:106`) |
| 기록 대 재시뮬 | mask 는 **기록된** T53 rollout 에서 나왔다. T55 E0 재시뮬은 16 개 test episode 모두 기록과 bit 단위로 같다 (`e0_exact`). 그래서 trace 마다 재시뮬 `t_step` 이 mask 의 `chunk_start_steps` 와 같은지 확인한다 (`:147`). 채점한 100 trace (natural 16 · instruction 48 · empty 16 · slip 20) 모두 같았다 |
| 개입 조건 | **episode 하나에 mask 하나**를 정하고, 그 episode 의 모든 trace 에 건다: natural (E0), instruction ×3 (E0 image, 명사 교체), empty, slip ghost · park. 개입 재시뮬은 같은 기록 동작을 open loop 로 재생한다. 그래서 로봇의 작업 시간축 (놓기 · 복귀) 은 기록과 같다. 개입된 물리로 P · H 를 다시 구하지 **않는다.** park 된 target 은 놓이지 않으므로 그렇게 하면 empty 조건만 full horizon 이 남는다. 그러면 같은 episode 안에서 조건마다 창이 달라진다 |
| 채점 frame | 창 W 에서 trace 의 채점 frame = 그 trace 의 조건 창 (전체, empty · slip 은 `window_start` 이후) **AND** episode 의 W mask (`apply_window`, `:160`). 나머지 (지속 K, feature 지속 p, latch, prereg arm) 는 전체 trace 에서 `score_test` 와 똑같이 계산한다 |
| arm 간 동일성 | trace 하나에 `window` 하나가 있고, 그것을 strong · hsr · noun 이 함께 읽는다. 두 predicate (P_task, P_grasp) 와 latch mode 3 개 (reground · record · oracle) 에도 같은 mask 를 쓴다 |
| 성질 | 창은 prefix (`t_step ≤ t_end`) 이고 모든 filter 는 causal 이다. 그래서 창을 건 채점은 trace 를 t_end 에서 자른 채점과 같다 (test 로 고정) |

T53 끝점 (T60 mask, 16 episode): P 도달 9 (t_P 247–312), H 도달 8. 제거 frame 은 P 362 / 1200, H 280 / 1200 이다 (T60 §3 표와 같음). 개입 창 시작 (`window_start` 13–47) 은 모든 episode 에서 t_P 보다 앞이다. 그래서 채점 frame 이 0 이 된 trace 는 없다.

## 바뀐 파일 (모두 새 파일)

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/ag3s/experiments/studies/t55_windows.py` (새, 543 줄) | `load_mask` (`:106`, keep 재유도 검사), `check_alignment` (`:147`), `apply_window` (`:160`), `score_window` (`:185`, `t55_study.score_test` 그대로), `natural_events` (`:199`), `pooled_decision` (`:216`, 합산 P_task 판정 행 + bootstrap + 허용폭), `parity` (`:279`, full 이 `test_results.json` 과 완전 일치해야 P · H 를 씀, 실패 시 exit 2), `side_by_side` · `decision_rows` · `headline` (`:300`–`:373`), `run` (`:441`, 입력 · 고정 코드 md5 를 앞뒤로 기록) | T55 보조 채점 |
| `AG3S/ag3s/experiments/studies/t56_windows.py` (새, 397 줄) | `window_positions` (`:81`), `record_positions` (`:92`), `analyze_windows` (`:112`, `t56_study.episode_tables` · `evaluate_group` 그대로, record 의 `phase_pos` 자리에 창 위치를 넣은 복사본 사용), `criteria` (`:140`, `stage_test` 의 기계적 기준과 같은 식), `parity` (`:222`, full 이 `t56_summary.json` 의 `results[c][g]["all"]` · `criteria_mechanical` 과 NaN 을 구분해 완전 일치), `stage_args` (`:286`, test 단계가 쓴 인자를 summary 에서 그대로 다시 읽음), `run` (`:318`) | T56 보조 채점 (준비만, 실행 안 함) |
| `tests/ag3s/test_t55_windows.py` (새, 8) | mask 재유도 · 파일 하나 규칙, 창이 모든 trace · arm 에 같음, t_step 불일치 거부, **창 = 잘라낸 trace** (score_test 전체 결과 비교), 합산 판정 행 산술 · 허용폭, parity 도구, **실데이터**: ep1804 · ep1840 의 full 채점 trace 별 row 가 `test_results.json` 과 같음 | |
| `tests/ag3s/test_t56_windows.py` (새, 7, synthetic) | 위치가 t_step mask 를 따름 (frame 부분집합 포함), geometry frame 불일치 거부, full = test 단계 평가, 창이 선택을 바꾸는 사례, `both` 범위 = 잘라낸 record (1e-12 이내), 모든 후보 · 변형에 같은 frame, event 없는 episode, 기준 · parity 도구 | |

고정 파일 md5 (작업 전후 같음): `t55_study` 03e26427… · `t55_predicate` 52520b94… · `t55_resim` d5f77fbf… · `t55_features` b5aca3e5… · `t55_perception` 231b0808… · `t56_study` b51dab54… · `t56_select` e25cfc8a… · `t56_selfscore` 93e608fa… · `t56_replay` f7c62936… · `t56_held` 5a1e57ca… · `t56_cells` 7c4e4360… · `t54_stats` ec8e4fa1… · `t54_replay` 21a6dd92… · `t54_arms` fabb3c47… · `t54_study` 513c5b39… · `t60_termination` 94edf05c…. 새 파일: `t55_windows` e1b35d78…, `t56_windows` f56cdeb9….

## T56 창의 두 범위 (`t56_windows`)

| 범위 | self 점수 | GT 품질 | 뜻 |
|---|---|---|---|
| `both` (보조의 주) | 창 안 frame 만 | 창 안 frame 만 | 완료 이후 frame 이 고르지도 채점되지도 않음 (`episode_tables(phase_self=True)`) |
| `gt` | 모든 frame (test 단계 선택 그대로) | 창 안 frame 만 | 전체 창에서 고른 것이 작업 구간에서 얼마나 손해인가 |

S1 · S2 label 을 만드는 HELD event 는 놓기 전이라 두 창 안에 있다. bootstrap draw 행렬 (B · seed 는 prereg) 은 모든 창에 같은 것을 쓴다.

## 단위 검증

```bash
MUJOCO_GL=osmesa OMP_NUM_THREADS=1 PYTHONPATH=$PWD .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider \
  tests/ag3s/test_t55_windows.py tests/ag3s/test_t56_windows.py
MUJOCO_GL=osmesa OMP_NUM_THREADS=1 PYTHONPATH=$PWD .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider tests/ag3s
```

- 새 test: **15 passed** (T55 8, T56 7), 8.6 s. T55 의 실데이터 test 는 skip 되지 않았다.
- 전체 `tests/ag3s`: **1664 passed, 17 skipped**, 16 warnings, 374 s, rc 0 (working set 33 GiB + 8 ≤ 60 확인 뒤 1 회, log `…/verify/T55_windows/logs/pytest_full.log`).

## 실행 (T55, 이번에 실제로 돌림)

```bash
cd /mnt/dev/AG3S_VLA-development && source scripts/setup/env.sh
export AG3S_RUN_DIR="$(.venv-openpi-live/bin/python -m common.run_output)"
#   -> /mnt/dev/AG3S_VLA-storage/results/development/20261014T193322+0900_284366d2bbec_e3469147913d
MUJOCO_GL=osmesa OMP_NUM_THREADS=1 PYTHONPATH=$PWD .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.t55_windows
#   defaults: --t55-run …/20261013T171325+0900_cf38c9dbca7b_2ebfd8a5a43c
#             --t60-step1 …/20261013T223348+0900_cf38c9dbca7b_8aef81831126/verify/T60/step1
```

CPU 1 thread, 59 s, 최대 RSS 0.74 GiB. 입력 (`T55.prereg.json` · `test_results.json`) 과 고정 코드의 md5 는 앞뒤로 같다 (`inputs_unchanged: true`).

**출력** (`$AG3S_RUN_DIR/verify/T55_windows/`):

| 파일 | 내용 |
|---|---|
| `t55_windows.json` | 규칙 · 입력 md5 · parity · 끝점 · 창별 채점 frame 수 · 창별 natural_events · `pooled_decision[window][mode][arm]` · `headline[mode]` |
| `scores_full.json` · `scores_P.json` · `scores_H.json` | 창마다 `score_test` 전체 출력 (mode 3 개, per-trace row 포함) |
| `side_by_side.csv` | mode × 조건 × predicate × arm × 창 (full · P · H 가 붙어 있음). count · rate · 4 상태별 false pass · 지연 · 미검출, 결합 arm 은 paired 차이와 episode CI |
| `decision.csv` | mode × arm × 창의 합산 P_task 판정 행 (Δ false pass · Δ false reject · Δ 지연 · Δ 미검출, CI, 허용폭 통과 여부) |
| `logs/run.log` · `logs/pytest_full.log` | 실행 · 전체 test log |

**parity 결과**: full 창이 `test_results.json` 의 `results.reground` · `results.record` · `results.oracle` (bootstrap CI · moving block · McNemar · per_episode 포함) 과 `natural_events` 를 **완전히** 재현했다 (per_episode 는 mode 마다 400 행) (`parity.exact: true`, 첫 차이 없음).

## T56 실행 방법 (T56 test 가 끝난 뒤, verifier)

```bash
export AG3S_RUN_DIR="$(.venv-openpi-live/bin/python -m common.run_output)"   # 새 run
MUJOCO_GL=osmesa OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=$PWD \
  .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.t56_windows \
  [--t56-run …/20261013T223613+0900_cf38c9dbca7b_60a2394b8ccc]
#   reads verify/T56/test/analysis/t56_summary.json (+ its args: --t54-run, --t53-run, kinds, conditions),
#   the prereg (sha256 must match the summary), the replay / held outputs — all read only
#   -> $AG3S_RUN_DIR/verify/T56_windows/{t56_windows.json, windows.csv}
```

- summary 가 없으면 바로 멈춘다 (지금 상태에서 확인함: rc 1, "no T56 test summary").
- full 이 summary 와 다르면 P · H 를 쓰지 않고 exit 2.
- 메모리 · 시간: test 단계의 `analyze` 는 전체 + phase 4 × 2 = 9 회 평가한다. 이 도구는 전체 + 창 2 × 범위 2 = 5 회이고 replay 를 다시 하지 않는다. 그래서 test 단계의 채점 (`st score`, 추정 6 GiB) 보다 크지 않을 것이다 (실측 아님).

## verifier 가 알아야 할 것

- 새로 생긴 플래그 · 기본값 변경: 없음 (새 CLI 두 개만: `--t55-run` · `--t60-step1` · `--out-run`, `--t56-run` · `--t60-step1` · `--out-run`).
- 재생산이 필요한 산출물: 없음. 기존 run 은 읽기만 한다. 출력은 새 run 의 `verify/T55_windows/` · `verify/T56_windows/` 다.
- 옛 기록과 호환이 깨지는가: 없음.
- 위 T55 수치는 구현자가 낸 것이다. verifier 는 같은 명령을 새 run 에서 다시 돌려 확인한다 (59 s).
- figure 는 만들지 않았다 (verifier 의 일).

## 주의할 점 (해석이 아니라 읽는 법)

1. **P · H 창은 대부분 음성 frame 을 지운다.** 합산 P_task 의 양성 frame (131), false reject, 지연, 미검출은 세 창에서 같다. P_task 가 참인 frame 은 모두 t_P 이전이기 때문이다. false pass 의 count 도 거의 같다 (strong 314 → P 308 → H 314). Δ false-pass **rate** 가 커지는 것은 대부분 분모 (음성 frame 6616 → 4082 / 4656) 가 줄어서다. 비교는 count 와 함께 읽는다.
2. instruction 조건의 wrong-object false pass 는 세 창에서 같다 (wrong-object frame 333 개가 모두 t_P 이전).
3. H 창은 P 에 도달하고 H 에 도달하지 못한 episode (ep1866) 의 75 frame 을 모두 남긴다 (T60 규칙). 그래서 H 가 P 보다 frame 이 많다.
4. mask 는 기록된 시간축에서 정한다. empty · slip 재시뮬에서 개입 뒤 물리가 바뀌어도 창은 E0 의 것이다 (위 규칙 표).
5. 이 결과는 보조 분석이다. 주 판정은 전체 창이고, T55 는 해제 · 복귀 뒤 동작도 평가하므로 P 결과가 그것을 대신하지 않는다.

## 내가 기대하는 결과
(**verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.**)

T55 는 위 실행으로 이미 수치가 나왔다. 아래는 그 값이다 (reground, 합산 P_task, 결합 − strong, [episode 95 % CI]).

| arm | 창 | Δ false pass rate | Δ false reject rate | Δ 지연 (frame) | Δ 미검출 | false pass count strong → arm / 음성 | 허용폭 |
|---|---|---|---|---|---|---|---|
| hsr | full | −1.19 %p [−2.22, −0.43] | +0.76 %p [0, +2.38] | +0.10 [0, +0.33] | 0 | 314 → 235 / 6616 | 충족 |
| hsr | P | −1.84 %p [−3.60, −0.65] | +0.76 %p [0, +2.38] | +0.10 [0, +0.33] | 0 | 308 → 233 / 4082 | 충족 |
| hsr | H | −1.70 %p [−3.02, −0.61] | +0.76 %p [0, +2.38] | +0.10 [0, +0.33] | 0 | 314 → 235 / 4656 | 충족 |
| noun | full | −4.46 %p [−6.44, −2.58] | +1.53 %p [0, +4.76] | 0.00 [0, 0] | 0 | 314 → 19 / 6616 | 충족 |
| noun | P | −7.10 %p [−12.38, −3.48] | +1.53 %p [0, +4.76] | 0.00 [0, 0] | 0 | 308 → 18 / 4082 | 충족 |
| noun | H | −6.34 %p [−9.87, −3.32] | +1.53 %p [0, +4.76] | 0.00 [0, 0] | 0 | 314 → 19 / 4656 | 충족 |

instruction 조건 wrong-object false pass (reground): strong 216 · hsr 212 · noun 7 (full · P · H 같음, 음성 3600 / 2514 / 2760). record · oracle mode 도 같은 방향이다 (`decision.csv`).

T56 은 실행하지 않았으므로 기대값이 없다.
