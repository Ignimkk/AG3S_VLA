# AG3S-PRE-CORE-TO E2 — 구현 (multi-seed TrajOpt)

> writer: ag3s-implementer · 읽는 쪽: lead, verifier, scribe
> 작성: 2026-10-26 05:20 KST (`date -u` 20:20 UTC + 9 h). 기준 코드: `feature/ag3s-development` HEAD `b3b69311` + E1 작업 트리 + 이 변경 (commit 안 함).
> 명세: [`AG3S-PRE-CORE-TO.task.md`](AG3S-PRE-CORE-TO.task.md) §6 "E2 구현 명세", §2 (IFG-TOA · T43 Q `_stay_start` · 211 국소 최소 교착). 방식은 [`AG3S-PRE-CORE-TO-E1.impl.md`](AG3S-PRE-CORE-TO-E1.impl.md) 를 따랐다.

## 무엇을 했나 (평이한 요약 먼저)

기본 꺼짐 flag `--to-multi-seed N` (config `sqp.multi_seed`) 을 넣었다. 켜면, chunk 의 표준 풀이(지금 SQP, E1-C 가 켜져 있으면 E1-C 포함)가 침범을 `safety.violation_tolerance`(0.1 mm) 안으로 풀지 못한 chunk 에서만, 같은 SQP 를 N 개의 다른 시작점에서 다시 푼다. 시작점은 정책 계획 · 제자리 · 손바닥 우회(가장 깊이 침범한 구의 거리장 gradient 방향과 그 수직 두 방향으로 5 / 10 cm) · 관절 noise(σ 0.02 / 0.05 rad) 다. 표준 풀이를 포함한 N + 1 개 중 최대 침범 → 목표 오차 → 관절 경로 길이 → 가속도 제곱합 순으로 하나를 고른다. 같으면 표준 풀이가 이긴다.
꺼져 있으면 궤적이 HEAD 와 bit 단위로 같다 (fixture 6 경우, `git archive HEAD` 사본에서 잰 SHA). 켜져 있어도 발동하지 않은 chunk 는 궤적이 같고 기록 키 `multi_seed` 하나만 는다 (ep211 기록 replay 75/75 chunk SHA 같음, 6 setting).
offline 비교용 settings module `AG3S.trajopt.experiments.e2_multi_seed` (`E2_16/32/64`, `E1C_E2_16/32/64`, E1 settings 포함) 와 `summarize` 를 만들었다.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/trajopt/multi_seed.py` (새 파일, 223 줄) | 상수 (`DETOUR_MAGNITUDES` 5 · 10 cm, `NOISE_SIGMAS` 0.02 · 0.05 rad, `RNG_SALT`, `LIMIT_SLACK` 1e-3, `PALM_LINKS` `ee_left`/`ee_right`), `seed_specs(n)` (:90), `chunk_rng` (:115), `perpendicular_frame` (:121), `bump` (:137), `damped_pinv` (:152), `detour_seed` (:158), `noise_seed` (:173), `random_direction` (:183), `path_length` (:190), `accel_sq` (:196), `selection_key` (:204), `select` (:212) | seed 구성과 고르는 규칙 (geometry 와 무관한 부분만, 단독 test 가능) |
| `AG3S/trajopt/config.py:704–715` | `SqpConfig.multi_seed: int = 0` + 검증 (int ≥ 0, bool 거절) | flag 의 config 자리. 창 optimizer 가 `with_overrides` 로 물려받는다 |
| `AG3S/trajopt/sqp.py:127` | 창 optimizer 에 `_ms_is_window = True` | 창 optimizer 는 주인이 chunk 번호를 넘길 때만 multi-seed |
| `AG3S/trajopt/sqp.py:138` | `self.multi_seed` | |
| `AG3S/trajopt/sqp.py:219–256` | 새 `solve`: 꺼지면 `_solve_standard` 그대로 호출. 켜지면 맨 위 호출에서만 표준 풀이 → `_multi_seed`. `multi_seed_index` keyword (없으면 자기 풀이 횟수) | 발동 조건 · 중첩 호출 구분 (`_ms_depth`, `_ms_child_index`) |
| `AG3S/trajopt/sqp.py:258` | 예전 `solve` 본문 = `_solve_standard` (이름만 바뀜) | |
| `AG3S/trajopt/sqp.py:301–307` | IFG-TOA3 안전 우선 대체 풀이를 부를 때 `_ms_child_index` 를 잠시 `None` | 대체 풀이 안에서 창 optimizer 가 seed 를 한 번 더 돌지 않게 (꺼지면 값만 바뀌고 쓰이지 않음) |
| `AG3S/trajopt/sqp.py:378–405` | `_multi_seed`: 발동 판정 (`gate closed` 면 창 쪽 기록 그대로, `nothing to avoid`, `within tolerance`) | |
| `AG3S/trajopt/sqp.py:407–504` | `_run_seeds`: seed 마다 `_solve(initial_iterate=…)`, 별도 QP solver 객체 `_ms_solver`, 후보 기록 · 선택 · 상태 복원 (`_previous` = 고른 궤적, `frame_index` = 표준 풀이 뒤 값), `solve_time_ms` = 표준 + seed 전체 | |
| `AG3S/trajopt/sqp.py:506–571` | `_seed_geometry`: 표준 결과의 `max_violation_pair` → 멀어지는 방향 (esdf: `_esdf_directions` = 값과 같은 층의 gradient, candidate: 점 − 중심, plane: 법선), 손바닥 고르기 | |
| `AG3S/trajopt/sqp.py:573–610` | `_build_seeds` (position box 로 clip) | |
| `AG3S/trajopt/sqp.py:612–645` | `_goal_probe` · `_goal_error_mm` · `_seed_anchor_record` | 목표 오차 (선택 2 순위), 고른 seed 의 `grasp_anchor` 기록 |
| `AG3S/trajopt/sqp.py:660, 683–687, 720–728, 751–757, 819–822` | `_solve(initial_iterate=None)`: 주면 그것이 첫 iterate (warm-start 혼합 · stay start 없음), servo observer 는 표준 풀이의 상태를 다시 씀 (`_ms_servo_state`, observer 는 같은 seq 를 두 번 받으면 상태가 바뀐다) | 기본 `None` 이면 예전과 같은 경로 |
| `AG3S/trajopt/sqp.py:1085–1087` | `_solve_window` 가 창 optimizer 에 `multi_seed_index` 를 넘김 (켜졌고 맨 위 풀이일 때만) | 창 풀이가 표준 풀이인 chunk 에서 seed 가 창에서 한 번 돈다 |
| `AG3S/trajopt/refiner.py:186–189` | `sqp.multi_seed > 0` 일 때만 `multi_seed_index = context["t_step"]` | seed RNG 를 요청 번호로 (online · offline 같은 chunk 같은 seed). 꺼지면 호출이 글자 그대로 같다 |
| `AG3S/trajopt/serve_safe.py:1822–1827` | `--to-multi-seed N` (기본 0, help 한국어) | |
| `AG3S/trajopt/serve_safe.py:2813–2814` | N < 0 → `ap.error` | `--no-safe` 와 함께면 기존 `--sqp-*` 검사가 막는다 |
| `AG3S/trajopt/serve_safe.py:2864–2865` | `sqp_overrides`: N > 0 일 때만 `{"multi_seed": N}` | 0 이면 키 없음 |
| `AG3S/trajopt/serve_safe.py:3166–3170` | 시작 log 한 줄 (WARNING, 켰을 때만) | |
| `AG3S/trajopt/experiments/tp_timing.py:37–39, 317–320, 445–447` | multi-seed setting 이 있으면 probe gate 를 만들고 (`goal_eval`), setting 출력에 `multi_seed` (optimizer 기록 그대로) | 꺼진 setting 만이면 출력 그대로 |
| `AG3S/trajopt/experiments/e2_multi_seed.py` (새 파일) | `SETTINGS` = E1 settings + `E2_16/32/64` + `E1C_E2_16/32/64` + smoke 전용 `SMOKE_IT1_*`, `summarize [--base NAME]` | offline 비교 |
| `tests/trajopt/test_e2_multi_seed.py` (새 파일) | 13 test | 아래 |

`e1_goal.py`, `blas_threads.py`, `--blas-threads`, 부분별 시간 코드, kit / task 문서는 건드리지 않았다.

## flag 의미

| flag | config | 뜻 |
|---|---|---|
| (없음) / `--to-multi-seed 0` | `sqp.multi_seed = 0` | 지금과 같다. `sqp_overrides` 에 키도 없다 |
| `--to-multi-seed N` (N > 0) | `sqp.multi_seed = N` | 표준 풀이의 `max_violation > safety.violation_tolerance` 인 chunk 에서만 N seed 를 더 푼다. 그 밖의 chunk 는 표준 풀이 그대로 + 기록 `{"triggered": false, "reason": …}` |

- **발동하지 않는 경우** (기록 `reason`): `standard solve within tolerance` · `nothing to avoid` (scene 없음) · `look-ahead gate closed: …` (창 optimizer 에 번호가 안 간 드문 경우만; 보통은 창 optimizer 자신의 기록이 실린다).
- **look-ahead (V11AB 계열, `plan_horizon 16`, gate 20 mm):** gate 가 열리면 seed 는 계획 전체(16 행, 인증은 창 8 행)에서 풀린다. gate 가 닫히면 표준 풀이가 창 optimizer(8 행)의 풀이이므로 seed 도 거기서 한 번 풀리고 꼬리는 정책 그대로다 (E1 과 같은 구조).
- **E1-C 와 함께:** 표준 풀이 = E1-C 앵커 풀이 + TOA3 대체 풀이 (그대로). seed 는 같은 앵커 항으로 풀고 (대체 풀이는 seed 에는 없음), 목표 오차가 선택 2 순위가 된다. 고른 seed 의 `grasp_anchor` 기록은 그 궤적으로 다시 만들고 `"multi_seed"` 표시를 붙인다.
- **E0 위의 E2 (`E2_*`):** 앵커가 없으므로 목표 오차 순위는 비고 (모두 0) 침범 → 경로 길이 → 가속도로 고른다. 목표 오차는 offline `goal_eval` 로 따로 잰다.

## seed 구성 (`multi_seed.seed_specs`, 결정적)

| 순서 | 종류 | 첫 iterate |
|---|---|---|
| 0 | `policy` | 정책 계획 그대로 (표준 풀이는 warm-start 혼합 또는 stay 에서 시작하므로 다르다) |
| 1 | `stay` | `Q[:, k] = q_now` (T43 Q `_stay_start` 의 계획) |
| 2–7 | `detour` | `away` · `e1` · `e2` × 5 · 10 cm |
| 8–9 | `noise` | σ 0.02 · 0.05 rad, 시작 행 고정 |
| 10– | 채움 | 방향 하나에 detour 5 · 10 cm 두 개, 그 뒤 noise σ 0.02 · 0.05 두 개(새 draw)를 되풀이. 방향 순서 `-e1`, `away+e2`, `away+e1`, `away−e1`, `−e2`, 그 뒤 `away` 쪽 반구의 random 방향 |

- N = 16: policy · stay · detour 6 · noise 2 · detour `-e1` 2 · noise 2 · detour `away+e2` 2. N = 32 · 64 는 앞 16 이 같고 뒤를 같은 규칙으로 채운다.
- **방향:** 표준 결과의 최악 행(`max_violation_pair`: block · step · query · point)에서. esdf 행은 `CollisionLinearizer._esdf_directions` (값을 낸 층과 같은 층의 gradient, target-free 행이면 `target_free_gradient`), candidate 행은 점 − 구 중심, plane 행은 법선. `e1` = 수평이고 `away` 에 수직, `e2` = `away × e1` 을 위(z ≥ 0)로 뒤집은 것. `away` 가 거의 수직이면 `e1` 은 world x 축을 투영한 것.
- **손바닥:** 앵커가 있으면 그 link 를 먼저, 다음 `ee_left` · `ee_right`. 최악 구의 Jacobian 열과 손바닥 Jacobian 열이 가장 많이 겹치는 것 (같으면 앞의 것). 겹침 0 이면 손바닥 없음 → detour 자리는 noise.
- **bump:** 행마다 sin² (0 행과 H−1 행에서 0, 최악 step 에서 1; 최악 step 은 [1, H−2] 로 자름). 손바닥 변위 `m · b_k · d` 를 행 k 의 손바닥 Jacobian(정책 계획에서)의 damped pseudo-inverse `J'(JJ' + 0.03² I)⁻¹` 로 관절에 되돌린다. 그 손바닥 chain 의 관절만 움직인다.
- **noise:** 같은 손바닥 chain 관절에만 (손바닥이 없으면 모든 관절), 0 행은 0.
- **RNG:** seed 자리 i 의 noise 는 `default_rng([0xE2, chunk 번호, i])`, random 방향 j 는 `default_rng([0xE2, chunk 번호, 100000 + j])`. chunk 번호 = 요청의 `t_step` (refiner 가 넘김), 없으면 optimizer 의 맨 위 풀이 횟수. 같은 chunk 번호 · 같은 입력이면 같은 seed · 같은 결과 (test 5).
- 모든 seed 는 position box 로 clip. 속도 · 가속도는 맞추지 않는다 (QP 의 soft 행이 맞춘다).

## 고르는 규칙 (`multi_seed.selection_key` / `select`)

후보 = 표준 풀이(0 번) + seed N 개. 작은 쪽이 이기는 사전식 key, 같으면 앞 번호:

1. 최대 침범: tolerance 이하면 0, 아니면 0.1 mm 단위 정수
2. 목표 오차 (mm, 0.1 mm 단위): 목표 행에서 손바닥과 고정 목표의 Euclid 거리 최대. 앵커에 이 horizon 의 목표 행이 있을 때만, 없으면 모두 0
3. 관절 경로 길이 `Σ_k ‖Q_k − Q_{k−1}‖` (`Q_{−1} = q_now`)
4. 가속도 제곱합 `Σ_k ‖Q_{k+1} − 2Q_k + Q_{k−1}‖²` (같은 열)

- **자격:** seed 결과의 관절 한계 초과 (`_limit_overshoot`: position · velocity · acceleration · 첫 행 anchor) 가 max(표준 결과의 초과, 1e-3) 보다 크면 후보에서 뺀다. QP 의 속도 · 가속도 행이 soft 라 정상 결과도 ~1e-5 를 남기는데, 먼 seed 를 한 반복 안에 못 끌어온 결과는 1e-2 이상이다 (fixture 측정: 0.004–0.17).
- 표준 풀이는 언제나 자격이 있으므로 고른 결과의 1 순위 key 는 표준 풀이보다 나쁠 수 없다.

## 기록 `to.metrics.multi_seed` (서버 기록에는 `_scalar` 로 문자열, offline 은 dict)

```
{triggered, n_seeds, index, standard_violation_mm, tolerance_mm,
 (발동 안 함) reason,
 (발동) geometry {block, link, step, palm, peak_step, n_joints, away, e1, e2 [, direction_error]},
        goal_rows,
        candidates [{kind, detail?, max_violation_mm, goal_err_mm, status, eligible, path_length_rad, accel_sq,
                     limit_overshoot, cost, reference_deviation, solve_ms, sqp_iterations}],
        chosen, chosen_kind, chosen_violation_mm, chosen_goal_err_mm, n_resolved, standard_ms, total_ms}
```

seed 가 고르면 `notes` 에 한 줄 (`multi-seed (E2): seed i (kind) of N replaced the standard solve (a mm -> b mm)`).

## 단위 검증

```bash
cd /mnt/dev/AG3S_VLA-development
# 새 test + E1 · TOA · BLAS test
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=osmesa PYTHONPATH=. \
  .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider tests/trajopt/test_e2_multi_seed.py \
  tests/trajopt/test_e1_grasp_goal.py tests/trajopt/test_ifg_toa_grasp_anchor.py tests/trajopt/test_blas_threads.py
# trajopt 전체 + QpProblem 을 쓰는 ag3s test
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=osmesa PYTHONPATH=. \
  .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider tests/trajopt tests/ag3s/test_t46_target_free_residual.py \
  --deselect tests/trajopt/test_t43t_replay.py::test_off_is_bit_identical_to_the_pre_fix_commit
```

- 첫 명령: **38 passed** (E2 13 + E1 15 + TOA 8 + BLAS 2; 모듈 `__file__` = 이 checkout 확인).
- 둘째 명령: **1974 passed, 30 skipped, 1 deselected, 0 failed** (19 min 58 s, 최종 코드).
- 첫 전체 실행(중간 코드)에서 refiner 를 spy optimizer(`config` 속성 없음)로 바꾸는 기존 test 3 개가 `AttributeError` 로 실패했다 → refiner 가 자기 `self.config.sqp` 를 읽도록 고쳤다 (`refiner.py:188`), 3 개 다시 통과.
- `test_t43t_replay.py::test_off_is_bit_identical_to_the_pre_fix_commit` 는 HEAD 에서도 실패하는 기존 실패라 뺐다 (E1 impl 에서 확인).

| test | 내용 |
|---|---|
| `test_off_is_bit_identical_to_head` | 꺼짐 (기본 · `multi_seed_index` 를 줘도) 의 궤적 SHA 가 HEAD 사본 값과 같다: 표준 fixture, 막힌 fixture 연속 2 chunk, E1-C 앵커, refiner 2 chunk (`t_step` 있음). `multi_seed` 키 없음 |
| `test_not_triggered_when_the_standard_solve_resolves` | 켜도 표준 풀이가 풀면 같은 궤적 · 다음 chunk 도 같음, 기록 `triggered False`, seed solver 를 만들지도 않음 |
| `test_stuck_standard_solve_is_resolved_by_a_detour_seed` | 막힌 fixture: 장애물을 쓸고 가는 경로 위에 정면으로 (기준 침범 159 mm), SQP 1 반복 → 표준 풀이 18.8 mm 남음. detour seed 가 0 으로 풀고, 반환은 풀린 후보, 기록 · notes · `solve_time_ms` = 표준 + seed, `_previous` = 반환 궤적, `frame_index` 1, 주 QP solver 의 마지막 해가 꺼짐 run 과 같음 |
| `test_e1c_on_top_runs_the_seeds_once_with_the_anchor` | E1-C (앵커 58.7 mm → 대체 풀이 11.6 mm 남음) 위: seed 가 한 번만 돌고 (`_run_seeds` 1 회), 모든 후보에 목표 오차, 반환 침범 0, `grasp_anchor` 기록이 고른 seed 의 목표 오차와 같음 |
| `test_selection_key_order_and_ties` | 침범 > 목표 > 길이 > 가속도, tolerance 이하는 0, 0.1 mm 단위, 같으면 표준, 자격 없음(None)은 못 이김 |
| `test_recorded_choice_is_the_rule_on_the_recorded_candidates_and_a_goal_moves_it` | 기록된 후보에 규칙을 다시 적용하면 기록된 선택과 같다. 목표 없으면 `stay` 가 고름 (경로 길이 0). 가중치 0 앵커로 목표만 주면 (비용 같음, 침범 값 같음) 풀린 후보 중 목표 오차 최소가 고름 (`stay` 아님) |
| `test_same_input_same_result_and_the_index_drives_only_the_random_seeds` | 같은 입력 · 번호 → 같은 궤적 · 기록 (시간 필드 제외). 번호를 바꾸면 noise · random 방향 seed 만 바뀐다 (64 seed). 번호가 없으면 0, 1, … |
| `test_seed_specs_and_construction` | 순서 · 개수 (0, 1, 5, 16, 32, 64), 16 ⊂ 32 ⊂ 64, frame 직교 · e1 수평 · e2 위, bump 끝 0 · 꼭대기 1, detour 5 cm → 꼭대기 행 손바닥 4–6 cm (방향 밖 < 1 cm), 0 · H−1 행 · 다른 관절 불변, noise 0 행 고정, stay = q_now (box clip), 방향 없으면 noise |
| `test_config_and_cli` | 기본 0, 음수 · bool · 실수 거절, `--to-multi-seed 0` · 없음 → 키 없음, 16 → `{"multi_seed": 16}`, 음수 · `--no-safe` 와 함께 `ap.error` |
| `test_refiner_hands_the_request_index_only_when_on` | 꺼지면 `multi_seed_index` 를 안 넘김, 켜면 `t_step` 그대로 |
| `test_lookahead_window_runs_its_own_multi_seed_once` | look-ahead: gate 닫힘 → 창 optimizer 에서 1 회, 열림 → 바깥에서 1 회 (계획 10 행 유지). E1 앵커가 아무것도 적용 못하는 chunk (`_solve_plain_recorded` 중첩) 에서도 창 optimizer 가 번호를 받아 1 회 — 이 경우는 처음 구현에서 빠졌던 것을 smoke 로 찾아 고쳤다 (고치기 전 코드로 이 test 실패 확인) |
| `test_offline_settings_are_registered_and_valid` · `test_summarize_counts_the_e0_violated_chunks` | settings 와 요약 |

## offline smoke (1 episode, 판정 아님)

```bash
RUN=/mnt/dev/AG3S_VLA-storage/results/development/20261025T191319+0900_35b909cf096a_cc215238f581
O=$RUN/verify/AG3S-PRE-CORE-TO/E2/offline_smoke
cd /mnt/dev/AG3S_VLA-development
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/AG3S_VLA-development OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-openpi-live/bin/python -m AG3S.trajopt.experiments.tp_timing \
  --index $RUN/verify/AG3S-GPU-STAGES/M2_cpu/e4_index.json --key E4/B_ROI_FPR_ep211_s2113 \
  --settings-module AG3S.trajopt.experiments.e2_multi_seed \
  --variants E0 E1C E2_16 E2_32 E2_64 E1C_E2_16 E1C_E2_32 E1C_E2_64 \
  --field cpu --mode server --replay-wrist-track --progress --out $O/B_ROI_FPR_ep211_s2113.json
.venv-ag3s/bin/python -m AG3S.trajopt.experiments.e2_multi_seed summarize $O/B_ROI_FPR_ep211_s2113.json > $O/summary.json
# 발동 경로 확인 (smoke 전용 settings: SQP 1 반복)
... tp_timing ... --variants SMOKE_IT1_E0 SMOKE_IT1_E2_16 SMOKE_IT1_E1C_E2_16 ... --out $O/B_ROI_FPR_ep211_s2113_smoke_it1.json
.venv-ag3s/bin/python -m AG3S.trajopt.experiments.e2_multi_seed summarize --base SMOKE_IT1_E0 $O/B_ROI_FPR_ep211_s2113_smoke_it1.json > $O/summary_smoke_it1.json
```

산출 (`$O`, link 없음): `B_ROI_FPR_ep211_s2113.json` · `.log` (75 chunk, 49.2 s), `summary.json`, `B_ROI_FPR_ep211_s2113_smoke_it1.json` · `.log` (75 chunk, 32.7 s), `summary_smoke_it1.json`.

- **본 smoke:** ep211 (S-REG) 에서 E0 가 0.1 mm 를 넘긴 chunk 가 없다 (최대 0.07 mm) → 6 개 multi-seed setting 모두 **한 번도 발동하지 않음**. 궤적 SHA 가 E0 ↔ `E2_16/32/64`, E1C ↔ `E1C_E2_16/32/64` 75/75 같음 (켜도 발동 안 한 chunk 는 그대로라는 것을 기록 replay 로 확인). gate 닫힌 E1C chunk 5 개도 창 optimizer 의 기록 (`within tolerance`) 이 실렸다. `summary.json` 의 `per_setting` 은 비어 있다 (대상 chunk 0).
- **발동 smoke (`SMOKE_IT1_*`):** SQP 1 반복이면 seq 15 하나가 0.435 mm 를 남긴다. `SMOKE_IT1_E2_16`: 실제 ESDF 에서 방향을 읽고 (block esdf, link `ee_finger_l1`, 손바닥 `ee_left`, gate open), 16 seed 중 9 개가 침범 0, `policy` seed 를 고름 → 0 mm. `SMOKE_IT1_E1C_E2_16` 은 표준(E1-C) 풀이가 이미 0 이라 발동 안 함.
- **시간 (발동 chunk 1 개, 판정 아님, pod load ≈ 38, 같은 시각 pytest process 1 개):** 표준 30.4 ms + seed 16 개 1060.9 ms (seed 하나 34–144 ms, 중앙값 ≈ 55 ms) = `solve_ms` 1091 ms. 실행 창 533 ms 를 넘는다. 발동하지 않은 chunk 의 시간은 E0 와 같다 (중앙값 27.3 vs 27.2 ms).

## full 비교 명령 (lead 가 실행)

8 setting 을 같은 pass 에 넣는다 (`summarize` 가 같은 파일의 E0 · E1C 와 chunk 별로 비교). 시간은 E0 위반 chunk 수 × seed 수 × ~55 ms 가 더해진다: S-OCC (E0 위반 68 chunk / 967) 는 seed 몫만 약 68 × (16+32+64) × 2 × 55 ms ≈ 14 분 + 기본 ≈ 4 분, S-REG (E0 위반 2 chunk) 는 E1 의 2 배 setting ≈ episode 당 90 s × 18 ≈ 27 분 (CPU 1 process 기준, 추정).

**S-REG (E4 18 장면):**

```bash
RUN=/mnt/dev/AG3S_VLA-storage/results/development/20261025T191319+0900_35b909cf096a_cc215238f581
IDX=$RUN/verify/AG3S-GPU-STAGES/M2_cpu/e4_index.json
O=$RUN/verify/AG3S-PRE-CORE-TO/E2/offline_e4
mkdir -p $O
cd /mnt/dev/AG3S_VLA-development
for key in $(python3 -c "import json;print(' '.join(json.load(open('$IDX'))['runs']))"); do
  n=$(basename $key)
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/AG3S_VLA-development OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /mnt/dev/AG3S_VLA-development/.venv-openpi-live/bin/python -m AG3S.trajopt.experiments.tp_timing --index $IDX --key $key \
    --settings-module AG3S.trajopt.experiments.e2_multi_seed \
    --variants E0 E1C E2_16 E2_32 E2_64 E1C_E2_16 E1C_E2_32 E1C_E2_64 \
    --field cpu --mode server --replay-wrist-track --out $O/$n.json > $O/$n.log 2>&1 || echo "FAIL $key" >> $O/fail.txt
done
/mnt/dev/AG3S_VLA-development/.venv-ag3s/bin/python -m AG3S.trajopt.experiments.e2_multi_seed summarize $O/*.json > $O/summary.json
touch $O/DONE
```

**S-OCC (OCC round 4, 18 run) — link 를 만들고, 끝나면 반드시 지운다:**

```bash
RUN=/mnt/dev/AG3S_VLA-storage/results/development/20261025T191319+0900_35b909cf096a_cc215238f581
IDX=$RUN/verify/AG3S-PRE-CORE-TO/E1/occ_index/occ_index.json
O=$RUN/verify/AG3S-PRE-CORE-TO/E2/offline_occ
mkdir -p $O
cd /mnt/dev/AG3S_VLA-development
bash $RUN/verify/AG3S-PRE-CORE-TO/E1/occ_index/make_links.sh
trap 'bash $RUN/verify/AG3S-PRE-CORE-TO/E1/occ_index/make_links.sh --remove' EXIT
for key in $(python3 -c "import json;print(' '.join(json.load(open('$IDX'))['runs']))"); do
  n=$(basename $key)
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/AG3S_VLA-development OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /mnt/dev/AG3S_VLA-development/.venv-openpi-live/bin/python -m AG3S.trajopt.experiments.tp_timing --index $IDX --key $key \
    --settings-module AG3S.trajopt.experiments.e2_multi_seed \
    --variants E0 E1C E2_16 E2_32 E2_64 E1C_E2_16 E1C_E2_32 E1C_E2_64 \
    --field cpu --mode server --replay-wrist-track --out $O/$n.json > $O/$n.log 2>&1 || echo "FAIL $key" >> $O/fail.txt
done
bash $RUN/verify/AG3S-PRE-CORE-TO/E1/occ_index/make_links.sh --remove; trap - EXIT
find $RUN -type l | head    # 비어 있어야 한다 (서버가 밖을 가리키는 link 가 있는 run 을 거절한다)
/mnt/dev/AG3S_VLA-development/.venv-ag3s/bin/python -m AG3S.trajopt.experiments.e2_multi_seed summarize $O/*.json > $O/summary.json
touch $O/DONE
```

- `summary.json`: `per_setting[v]` — `n_base_violated` (E0 > 0.1 mm chunk 수), `n_resolved`, **`n_resolved_goal_le_10mm` · `share_resolved_goal_le_10mm`** (§7 E2 기준의 비율: 해소 ∧ 목표 오차 axis max ≤ 10 mm, 목표 없는 chunk 는 해소만), Euclid 판, `n_with_goal`, 침범 중앙값 · 최대, `solve_ms` 중앙값 · p95 · 최대 (그 chunk 들), `n_triggered`, `multi_seed_ms` 중앙값 · p95 · 최대, **`chosen_kind` · `chosen_kind_when_resolved`** (아래 1 번 때문에 꼭 본다), `status`. `per_run`, `all_chunks` (= `e1_goal summarize`).
- §7 E2 판정: `E2_*` 또는 `E1C_E2_*` 하나 이상의 `share_resolved_goal_le_10mm` 가 `E1C` 의 것보다 높은가.
- CPU 병렬이 필요하면 key 목록을 둘로 나눠 두 process 로 (worker ≤ 4 규칙).

## verifier 가 알아야 할 것

- 새 flag: `--to-multi-seed N` (기본 0). 새 config 키 `sqp.multi_seed` (기본 0) — `to_dict()` 에 키가 하나 는다 (E1 의 `cost.grasp_goal_box` 와 같은 방식). 서버 기록 meta 의 `sqp` (`sqp_meta`) 는 그대로.
- 꺼짐이면 `to.metrics` 키 · 응답 · 기록이 같다. 켜면 모든 TO chunk 에 `to.metrics.multi_seed` 가 생기고, 발동한 chunk 의 `to.solve_ms` 에 seed 시간이 들어간다.
- `tp_timing` 출력: multi-seed setting 에 `multi_seed` 키, 그리고 multi-seed setting 이 있으면 probe gate 의 `grasp_goal` (chunk) · `goal_eval` · `grasp_anchor` (setting) 키가 생긴다 (E1 setting 과 같음).
- 재생산이 필요한 산출물: 없음. 옛 기록과 호환: 깨지지 않음.

## 확인하지 못한 것 · 확실하지 않은 것

1. **명세의 선택 순서는 목표가 없을 때 `stay` 를 고르기 쉽다.** 침범을 푼 후보끼리는 (목표가 없으면) 관절 경로 길이가 결정하고, 제자리 계획은 경로 길이가 거의 0 이다. 막힌 fixture 에서 그렇게 됐다 (`chosen_kind stay`). closed-loop 에서 `stay` 가 이기면 로봇은 그 chunk 에 움직이지 않는다 — HOLD 와 같은 정지이고 (실행은 되므로 판정은 feasible), 211 같은 교착을 풀지 못할 수 있다. offline 비율("침범 해소")은 이것도 해소로 센다. 그래서 요약에 `chosen_kind_when_resolved` 를 넣었다. 실기록 smoke 의 한 발동 chunk 에서는 `policy` 가 이겼다 (1 반복 뒤 stay 결과의 경로 길이 0.106 rad > policy 0.075 rad). 3 순위를 "정책 계획에서의 편차" (`reference_deviation`, 후보 기록에 있음) 로 바꿀지는 lead · 사용자 판단이다. 후보 기록에 `cost` · `reference_deviation` 을 같이 실어 두었으므로 다른 규칙을 기록에서 다시 적용해 볼 수 있다 (다만 server mode 의 뒤 chunk 는 따라 바뀌지 않는다).
2. **E0 위 E2 는 목표를 모른다.** 서버는 앵커 flag 가 없으면 gate 를 돌리지 않으므로 `E2_*` 의 선택에 목표 오차가 없다. 가중치 0 앵커 (`GraspAnchor(link, goal, c, 0, 0)`) 를 주면 비용은 그대로 두고 목표만 선택에 넣을 수 있다 (test 로 확인). 그런 setting 은 만들지 않았다 (명세 밖).
3. **시간:** 발동 chunk 하나에 seed 16 개 ≈ 1.06 s (smoke 1 chunk). 32 · 64 는 비례해서 늘 것이다. seed 는 순차로 푼다 (병렬 없음). 첫 발동 chunk 에서 seed 용 OSQP 객체가 setup 된다. 실행 창 533 ms 안에 들지 않으므로 closed-loop 시간 판정은 이 숫자와 함께 봐야 한다 (명세: "시간이 실시간에 맞지 않으면 그 숫자가 E3의 동기다").
4. **seed 는 대체 풀이(TOA3) 없이 푼다.** E1-C 위에서 seed 는 앵커 항을 지닌 채 풀리고, 앵커 때문에 남는 침범을 앵커 없이 다시 푸는 단계는 없다 (표준 풀이 후보에는 있다). 앵커 없는 seed 도 넣을지는 열어 두었다.
5. **seed 의 속도 · 가속도:** position box 로만 clip 한다. 10 cm detour · σ 0.05 noise 는 첫 iterate 에서 속도 · 가속도 한계를 넘기 쉽고, SQP 반복 상한 안에 못 끌어오면 자격에서 빠진다 (막힌 fixture, 1 반복: 16 · 32 · 64 seed 중 2 · 3 · 6 개). 반복 상한이 3 인 서버 설정에서 이 비율은 offline 에서 본다.
6. 이미 있는 동작 (고치지 않음): IFG-TOA3 대체 풀이의 재귀 `solve` 는 `joint_track_scale` (IFG-TOW) 을 넘기지 않는다 (`kw` 에서 빠짐) — 대체 풀이는 손목 가중치 없이 풀린다. multi-seed 의 seed 는 원래 scale 을 다시 넣어 푼다.
7. servo observer (`limits.servo_model`) 를 켠 구성에서 seed 는 표준 풀이가 갱신한 observer 상태를 다시 쓴다 (observer 는 한 chunk 에 한 번만 갱신). 그 구성의 실제 run 으로는 확인하지 않았다 (fixture 는 servo 꺼짐; V11AB 는 look-ahead 라 servo 와 함께 못 쓴다).
8. 그림은 만들지 않았다 (verifier 몫).

## 내가 기대하는 결과

> verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.

- S-REG 는 E0 위반 chunk 가 2 개뿐이라 판정에 거의 기여하지 않을 것이다. S-OCC 의 68 chunk 는 E1 offline 기록상 모두 케이스 3 L3 (막대 아래, 들어 올리는 경로) 이고, detour / stay 로 침범은 많이 풀릴 것으로 보지만 `stay` 가 고르는 비율이 높으면 closed-loop 이득은 offline 비율보다 작을 것이다. seed 수 16 → 64 의 이득은 작고 시간만 4 배일 것으로 본다.
