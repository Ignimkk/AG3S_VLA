# AG3S-PRE-CORE-TO-GGW — 구현

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead · 2026-10-28 01:35 KST
> 명세: `AG3S-PRE-CORE-TO.task.md` §6 "쥐기 목표 항 켜는 조건 넓히기 + 연속성 · 꺾임 항은 운반 · 놓기 단계에만" (사용자 판정 2026-10-28 01:07 KST)
> commit 안 함 (작업 트리에 이전 작업의 미commit 변경이 많다 — 아래 표의 hunk 만 이 STEP 의 것)

## 무엇을 했나 (평이한 요약 먼저)

TrajOpt 에 기본 꺼짐 option 두 개를 넣었다. (1) `--grasp-goal-window widen`: 쥐기 목표 항이 켜진 chunk (gate 는 그대로) 에서,
닫는 행이 최적화 구간 밖이거나 없으면 구간의 **마지막 3 행**, chunk 가 닫힌 채 시작하면 **처음 3 행** 에 목표를 둔다
— E3G 기록에서 목표가 얼려진 뒤에도 목표 행이 없어 항이 작동하지 않던 두 경우다. (2) `--continuity-after-grasp`: 연속성 항
(직전 chunk 겹침) 을 쥐기 전 (latch searching · latched · closing) chunk 에서 빼고 held 부터 쓴다 — `--deflection-rate-after-grasp`
(IFG-TOB) 와 같은 단계 판정. 둘 다 꺼짐 (기본 `close` · 없음) 이면 해 · 기록 키 · config dump · 시작 로그가 지금과 같다 (test 로 보였다).

## 규칙

목표 행 (`GraspAnchor.goal_mask(H)`, G = `cost.grasp_goal_rows` = 3, H = **실제로 푸는 구간**, c = 정책 chunk 의 첫 닫는 행):

| c | `close` (기본, 지금) | `widen` |
|---|---|---|
| None (닫기 계획 없음) | 없음 | H−G … H−1 |
| c ≥ H (구간 밖) | 없음 | H−G … H−1 |
| c = 0 (닫힌 채 시작) | 없음 | 0 … G−1 |
| 0 < c < H | c … min(c+G, H)−1 | 같음 |
| 목표 없음 · held · gate 꺼짐 | 없음 | 없음 (gate 무변경) |

- **두 horizon:** refiner 는 look-ahead 가 열리면 바깥 optimizer (H = 16) 로, 닫히면 window optimizer (H = 8, `_solve_window` 가
  `reference[:, :8]` 와 같은 anchor 를 넘김) 로 푼다. 각 optimizer 의 `_anchor_setup` 이 자기 H 로 `anchor_targets` → `masks(H)` →
  `goal_mask(H)` 를 부르므로 넓힌 행은 언제나 실제로 푸는 구간 기준이다. 같은 c = 12 가 H = 16 이면 12–14 (안쪽, 무변경), H = 8 이면 5–7.
- **CPU SQP:** QP 항 (`anchor_qp_terms`) · merit (`_objective` 의 `anchor_cost`) · E1-B 상자 (`_anchor_setup` 의 `goal_mask`) 모두 같은 mask.
- **E2 / E3-b GPU seed:** E2 의 `_goal_probe` (seed 선택 기준 goal error · first-pass 순서) 가 `goal_mask(H)` 를, E3-b 의
  `gpu_seeds.anchor_terms_for` 가 `anchor_targets` (= 같은 mask) 를 읽는다 — GPU seed 의 목표 항도 넓힌 행에 걸린다 (test 로 보였다).
  close 에서 c = 0 · ≥ H 이면 probe 가 None 이라 GPU 목표 항이 없던 것이, widen 에서는 생긴다. `gpu_seeds.py` 코드는 바꾸지 않았다.
- **연속성 항:** `SafePolicy._scene_fn` 이 최종 latch 상태로 `context["continuity_off"]` 를 정한다 (T37 과 같은 자리 · 같은 키).
  refiner `_continuity_reference` 가 이 키를 보면 previous chunk 를 `None` 으로 넘긴다 → `problem.build_problem` · `objective` 의
  continuity 항, GPU seed 의 continuity 항 (`prev_mask = 0`), look-ahead warm start 정렬 (`_seed_is_aligned(None)` = False) 이 모두 빠진다.
  QP 에 직전 chunk 겹침을 강제하는 다른 행은 없다 (`problem.py` 의 `previous_chunk` 사용은 이 항 둘뿐).
  `cost.grasp_continuity_off` (T37: closing · held 에서 끔) 와 함께 켜면 config 단계에서 거절한다 (뜻이 반대).

## 바뀐 파일 (이 STEP 이 더한 hunk 만, 현재 줄 번호)

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/trajopt/config.py:289-300` | `CostConfig.grasp_goal_window = "close"` · `continuity_after_grasp = False` (설명 주석) | config 기본값 (꺼짐) |
| `AG3S/trajopt/config.py:343-352` | validate: window ∈ {close, widen} · bool 검사 · `grasp_continuity_off` 와 배타 | 뜻 없는 조합 거절 |
| `AG3S/trajopt/config.py:1197` · `1205-1213` | `to_dict` 가 두 field 를 기본값일 때 뺀다 (`_OMIT_WHEN_DEFAULT` · `_omit_default`, 기존 `_OMIT_WHEN_NONE` 과 같은 이유) | 꺼짐이면 기록의 `trajopt_config` dump 키가 그대로 |
| `AG3S/trajopt/grasp_anchor.py:32-36` | module docstring 에 GGW 규칙 | |
| `AG3S/trajopt/grasp_anchor.py:69-70` | `GraspAnchor.goal_window = "close"` | anchor 가 규칙을 싣는다 — 모든 소비자가 같은 mask |
| `AG3S/trajopt/grasp_anchor.py:84-93` | `goal_mask`: widen 갈래 (close 갈래는 글자 그대로) | 위 표 |
| `AG3S/trajopt/grasp_anchor.py:98-110` | `goal_window_rule(H)` | 기록용 규칙 문자열 |
| `AG3S/trajopt/grasp_anchor.py:272-274` | `anchor_record`: widen 이면 `goal_window` · `goal_window_rule` · `horizon` | close 면 키 없음 |
| `AG3S/trajopt/sqp.py:297-302` | `_solve_standard`: widen 이면 `applied_horizon` · `goal_rows_applied` (· `window`) | 실제로 푼 행 |
| `AG3S/trajopt/sqp.py:326-327` | 안전 우선 plain 재풀이가 이기면 `goal_rows_applied = []` | 목표 항이 결과에 안 들어갔다 |
| `AG3S/trajopt/sqp.py:373-382` | `_applied_goal_rows` (look-ahead 닫힘: window optimizer 의 기록이 실제 적용 행) | 바깥 기록은 H = 16 의 mask 로 다시 쓰여 window 의 행을 덮었다 — widen 에서만 바로잡는다 |
| `AG3S/trajopt/sqp.py:397-400` | `_solve_plain_recorded`: widen 이면 같은 키 | (widen 에서는 목표가 없을 때만 이 길) |
| `AG3S/trajopt/sqp.py:720-721` | `_seed_anchor_record`: widen 이면 같은 키 | E2 · E3-b 가 seed 결과를 고른 chunk |
| `AG3S/trajopt/safe_policy.py:41-43` | `__all__` 에 `continuity_after_grasp_off_reason` | |
| `AG3S/trajopt/safe_policy.py:998-999` | `_scene_fn`: `off = self._continuity_off_reason()` (예전 `self._grasp_continuity_gate()`) | |
| `AG3S/trajopt/safe_policy.py:1595-1612` | `_continuity_off_reason` (T37 gate → 없으면 GGW gate) · `_continuity_after_grasp_gate` (`w_continuity > 0` 일 때만) | TOB `_deflection_rate_gate` 와 같은 꼴 |
| `AG3S/trajopt/safe_policy.py:1882-1896` | `_grasp_anchor_gate`: widen 이면 anchor 에 `goal_window`, 기록에 `goal_window` · `goal_rows_by_horizon` · `goal_window_rule_by_horizon` (plan · window horizon) | gate 판정 무변경 |
| `AG3S/trajopt/safe_policy.py:2630-2645` | `continuity_after_grasp_off_reason(phase, enabled)` (순수 함수, 문자열 상태도 받음) | 재생 도구용 (T37 함수와 같은 꼴) |
| `AG3S/trajopt/serve_safe.py:285-288` | `cost_overrides`: `grasp_goal_window` (close 면 키 없음) · `continuity_after_grasp` (꺼지면 키 없음) | |
| `AG3S/trajopt/serve_safe.py:366-371` | `announce_cost_weights` 가 `_fmt_cost` 로 출력 (숫자 · bool 은 예전 `:g` 그대로, 문자열은 그대로) | `"widen"` 에 `:g` 가 ValueError |
| `AG3S/trajopt/serve_safe.py:1848-1858` | argparse `--grasp-goal-window {close,widen}` (default close) · `--continuity-after-grasp` | |
| `AG3S/trajopt/serve_safe.py:2842-2849` | 거절: widen 인데 `--grasp-anchor` · `--grasp-goal-box` 둘 다 없음; `--continuity-after-grasp` + `--w-continuity 0`. `--no-safe` 와는 기존 `--w-*` 검사가 거절 | |
| `AG3S/trajopt/serve_safe.py:3422-3430` | `announce_ta` 시작 로그 두 줄 (`TO grasp-goal window (AG3S-PRE-CORE-TO-GGW)` · `TO continuity after grasp (…)`) | 다른 TO option 과 같은 꼴 |
| `tests/trajopt/test_pre_core_to_ggw.py` (새 파일) | 16 test | 아래 |

serve_safe · sqp · safe_policy · config 의 다른 diff hunk 는 이 STEP 이전의 미commit 작업이고 건드리지 않았다. `gpu_seeds.py` · `refiner.py` · `problem.py` 무변경.

## 기록 (요청 하나의 예)

**chunk summary `grasp.anchor`** (SafePolicy gate, widen, latch closing, chunk 가 닫힌 채 시작, plan 16 · look-ahead window 8):

```json
{"active": true, "hand": "left", "link": "ee_left", "close_row": 0, "goal_m": [0.0, 0.22, 0.6832], "goal_frame": 7,
 "palm_to_goal_mm": 0.0, "goal_window": "widen",
 "goal_rows_by_horizon": {"8": [0, 1, 2], "16": [0, 1, 2]},
 "goal_window_rule_by_horizon": {"8": "starts closed (close row 0): first rows", "16": "starts closed (close row 0): first rows"}}
```

**`to.metrics.grasp_anchor`** (SQP, widen, c = 12, plan 10 · window 4 test fixture, look-ahead gate 닫힘 → window 만 풂) — 바깥 기록의
`goal_rows` 는 H = 10 기준 [7, 8, 9], 실제 적용은 `applied_horizon` 4 · `goal_rows_applied` [1, 2, 3]; window 의 안전 우선 plain 재풀이가
이기면 `goal_rows_applied: []` · `fallback_used: true`. 키: `variant, link, w_tcp, w_goal, close_row, n_track_rows, n_goal_rows, goal_rows,
goal_m, palm_dev_from_reference_max_mm, goal_window ("widen"), goal_window_rule ("close row beyond this horizon: last rows"), horizon,
goal_err_*, applied_horizon, goal_rows_applied, window {…window optimizer 의 기록…}, fallback_used[, fallback]`.
look-ahead 가 열린 chunk 는 `window` 없이 `applied_horizon = 16`, `goal_rows_applied = goal_rows`.

**`to.metrics.continuity_off_by`** (`--continuity-after-grasp`, latch latched): `"before_grasp: grasp latch latched (cost.continuity_after_grasp)"`,
`continuity_active: false`, `continuity_reason: "disabled for this chunk: before_grasp: …"`. held 부터: `continuity_off_by: null`,
`continuity_active: true`. T37 사유 (`"grasp latch closing (cost.grasp_continuity_off)"`) 와 접두어로 구별된다.

목표 항이 켜진 요청 판정: `grasp.anchor.active` (gate) ∧ `to.metrics.grasp_anchor.goal_rows_applied` 가 비지 않음.

## 단위 검증

```bash
cd /mnt/dev/AG3S_VLA-development
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=osmesa PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 nice -n 19 \
  .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider tests/trajopt/test_pre_core_to_ggw.py   # 새 파일
# 관련 기존 파일 (anchor · E1 · TOA · TOB · sqp · safe_policy · port · E2 · E3b · T37 · refiner · TOW · SC · SCM · problem)
... tests/trajopt/test_ifg_toa_grasp_anchor.py tests/trajopt/test_e1_grasp_goal.py tests/trajopt/test_ifg_tob_deflection_rate_after_grasp.py \
    tests/trajopt/test_sqp.py tests/trajopt/test_safe_policy.py tests/trajopt/test_pre_core_to_port.py tests/trajopt/test_e3b_gpu_seeds.py \
    tests/trajopt/test_e2_multi_seed.py tests/trajopt/test_t37_grasp_continuity.py tests/trajopt/test_t37_grasp_continuity_replay.py \
    tests/trajopt/test_refiner.py tests/trajopt/test_ifg_tow_wrist_track.py tests/trajopt/test_scm_palm_shell_margin.py \
    tests/trajopt/test_sc_palm_shell.py tests/trajopt/test_problem.py
# 전체 tests/trajopt
```

결과 (CPU, worker 1, `.venv-ag3s`, 모듈 `__file__` 이 이 checkout 인 것 확인):

| 범위 | 결과 |
|---|---|
| 새 파일 `test_pre_core_to_ggw.py` | 16 passed |
| 관련 기존 16 파일 + 새 파일 | 205 passed, 4 skipped (GPU 필요 test) |
| 전체 `tests/trajopt` | 2025 passed, 35 skipped (torch · jax · CUDA · cuRobo 없음), **1 failed = 알려진 기존 실패** `test_t43t_replay.py::test_off_is_bit_identical_to_the_pre_fix_commit` (.venv-ag3s numpy golden, 이 STEP 과 무관) — 22 분 |

새 test 가 고정하는 것: config 기본 · 검사 · 배타 · 꺼짐이면 `to_dict` 키 없음; `goal_mask` (H = 8 · 16, c ∈ {None, 0, 1, H−2, H−1, H, H+5}, close = 옛 규칙 그대로);
기록 키 (close 에 새 키 없음); SQP — E1 golden SHA (`HEAD_SHA["toa_goal"]`, HEAD 35b909cf) 가 `goal_window="close"` 명시와 widen 의 c 안쪽에서 bit 동일,
widen 의 c 없음이 마지막 행을 목표로 당김; look-ahead 닫힘에서 적용 행 = window 의 행 (plain 이 이기면 []), 같은 chunk 를 close 로 풀면 "close row beyond this horizon";
E2 `_goal_probe` · E3-b `anchor_terms_for` · seed 기록이 같은 넓힌 행; anchor gate 판정 · 고정 목표가 close 와 widen 에서 같음;
연속성 gate (searching · latched · closing 에서 사유, held · placed 에서 None, flag 꺼짐이면 모든 단계에서 T37 그대로) 와 refiner 결과
(held 에서 flag 유무 궤적 bit 동일 · 항 켜짐, 쥐기 전 flag 켜짐이면 항 꺼짐 · 궤적 달라짐, flag 꺼짐이면 held 와 같은 궤적);
CLI (mapping · 꺼짐 = 키 없음 · 거절 4 가지 · E3 + 껍질 구 + 꼭지 덮개 + 쥐기 목표 항 + 꺾임 항과 함께 parse · 두 flag 외 config 동일) · `--help` · 시작 로그.

## verifier 가 알아야 할 것

- **새 flag (기본 꺼짐):** `--grasp-goal-window {close,widen}` (기본 close, config `cost.grasp_goal_window`), `--continuity-after-grasp`
  (config `cost.continuity_after_grasp`). widen 은 `--grasp-anchor` 또는 `--grasp-goal-box` 와 함께만. serving 기본값 · kit 변경 없음.
- **재생산이 필요한 산출물:** 없음.
- **옛 기록과 호환:** 깨지지 않는다. 꺼짐이면 기록 키 그대로. 켜짐 기록에만 새 키 (`goal_window*`, `goal_rows_by_horizon`, `applied_horizon`,
  `goal_rows_applied`, `window`) 와 새 `continuity_off_by` 값 (`before_grasp: …`).
- **offline 재생 도구:** `e1_goal.RecordedGraspGate` · `step3_eval` 은 widen 을 모른다 (close 로 재생). 필요하면 별도 작업.
  `continuity_after_grasp_off_reason` 은 기록 상태 문자열로 같은 판정을 낸다.
- **c 가 구간 안쪽이지만 끝 근처 (c = H−1)** 는 명세 (c) 대로 "그대로" — 1 행만 받는다 (widen 이 늘리지 않음).
- **GPU 경로:** smoke 안 함 (명세). GPU seed 의 목표 항은 `anchor_terms_for` 가 넓힌 행을 읽는 것까지 CPU 로 확인했다.

## 내가 기대하는 결과

<verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다>
가림 장면에서 목표 항이 켜진 요청 수가 E3G 보다 늘고 (특히 122 · 123 흔들림 없음), 쥐기 전 요청 1 의 첫 밀림이 연속성 · 꺾임 항을
끈 구성에서 줄어든다. 운반 단계 가속 한계 초과는 held 부터 두 항이 켜지므로 E3G 와 비슷할 것이다.
