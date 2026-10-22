# SL5i — 구현 (파지 인식 수정 · closing 면제)

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead · 작성 2026-10-23 02:40 KST (2026-10-22 17:40 UTC)
> base: `feature/panda-benchmark` `2bb5041b` (commit 하지 않았다 — lead 가 한다). GPU 는 쓰지 않았다.

## 무엇을 했나 (평이한 요약 먼저)

flag 세 개를 추가했다. 셋 다 기본값은 꺼짐이고, 꺼져 있으면 동작이 이전과 bit 단위로 같다.

- **`--grasp-reach-sticky`** (pb6): 한 닫힘 시도 안에서 reach 가 한 번 통과하면, 시도가 끝날 때까지 통과로 둔다. 같은 판단을 AG3S `attach` 의 reach gate 도 받는다.
- **`--grasp-drop-from-open`** (pb6): `dropped` 를 재는 기준을 "마지막으로 열려 있던 개도"로 바꾼다.
- **`--closing-hand-exempt`** (pb7): CLOSING 중에 손가락이 무언가에 막혀 멈추면(`closed`·`not_empty`·`blocked` 모두 참), pb4 의 held 면제를 그 call 에도 적용한다.

profile 두 개를 추가했다: `V11AB-pb6` = pb5 + 앞의 두 flag, `V11AB-pb7` = pb6 + closing 면제.

기록된 SL5g pb5-C calls log 로 latch 를 offline 에서 다시 돌렸다. flag 를 끈 replay 는 30 episode 모든 call 의 latch state 와 check 를 기록과 똑같이 재현했다. flag 를 켜면 6 episode 에서 기록과 갈라진다. 6 곳 모두 closing→held 이고, MuJoCo 에서 두 손가락이 target 에 닿아 있으며 bowl 이 들린 순간이다. 새로 생긴 거짓 held 는 0 이다.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/trajopt/grasp_latch.py:94-101` | module docstring 에 pb6 절 | 규칙 기록 |
| `AG3S/trajopt/grasp_latch.py:236-252` | `LatchConfig.reach_sticky` · `drop_from_open` (기본 False), 근거 숫자 주석 | ①(a)(b) switch |
| `AG3S/trajopt/grasp_latch.py:530, 546-547` | `observe_grasp(reach_context=None)` kwarg | sticky 통과 기록에 capture frame 을 남긴다. 판단에는 쓰지 않는다 |
| `AG3S/trajopt/grasp_latch.py:585-593` | LATCHED→CLOSING 에서 `drop_from_open` 이면 `opening_before = _last_open["opening"]` (없으면 예전 값). `_drop_basis` 기록. `_reach_pass = None` | (b) 기준 개도, 새 시도에서 sticky 를 초기화 |
| `AG3S/trajopt/grasp_latch.py:607-608` | flag on 일 때만 `evidence["drop_from_open"]` (`source` `last_open`/`transition`, `transition_opening`, `last_open`) | (b) record |
| `AG3S/trajopt/grasp_latch.py:620-622` | flag on 일 때만 `_sticky_reach` → `evidence["reach_sticky"]` | (a) |
| `AG3S/trajopt/grasp_latch.py:635-645` | attach note: sticky 를 썼으면 `reach sticky <통과값> mm (feedback seq …; 지금 … mm)`. flag off 에서는 이전과 같은 문자열 | sticky 로 붙을 때 지금 reach 가 None 이어도 note 가 깨지지 않게 |
| `AG3S/trajopt/grasp_latch.py:675, 698, 704` | `_last_opening` 을 갱신하는 세 경로 모두에서 `_track_open()` | (b) 추적은 SEARCHING 부터 |
| `AG3S/trajopt/grasp_latch.py:708-739` | `_track_open` (명령 ≥ `gripper_closed_below` **이고** 개도 ≥ 문턱인 요청을 기억), `_sticky_reach` (같은 `manipulated_id` 일 때만 sticky, `now`/`now_mm`/`passed`/`same_object`/`used`) | (a)(b) 본체 |
| `AG3S/trajopt/grasp_latch.py:891-900` | `_reset_grasp`: `_reach_pass`·`_drop_basis` 를 비운다. `_last_open` 은 `keep_opening=False`(=`reset()`) 일 때만 비운다 | 시도 경계 = reopen · revoke(`_reset_grasp(keep_opening=True)`) · LATCHED→CLOSING · `reset()` |
| `AG3S/ag3s/runtime/pipeline.py:615, 659-665` | `AG3S.attach(reach_sticky=None)` kwarg + docstring | (a) attach gate |
| `AG3S/ag3s/runtime/pipeline.py:709-717, 755-757` | `passed` 기록이 있으면 거리를 재기는 하되 거절하지 않는다. prebuilt / TargetGeometry 두 분기 모두 | latch 는 held 인데 attach 가 거부되는 어긋남을 막는다 |
| `AG3S/ag3s/runtime/pipeline.py:722-723, 839-840, 848-855` | `held_record["reach_gate"]` (`dist_mm`, `limit_mm`, `over_limit`, `decided_by` `latch_pass`/`measured`, `latch_pass`). kwarg 를 줬을 때만 생긴다 | record |
| `AG3S/trajopt/linearize.py:246-248` | `SceneSnapshot.grasp_contact` 문서에 `closing_exempt` | ② |
| `AG3S/trajopt/linearize.py:1030-1036` | `"closing_exempt" in gc` 일 때만 `closing_exempt_rows` 기록. 참이고 `latch == "closing"` 이면 held 면제와 같은 규칙(authorized ∧ class target 행 → margin −1) | ② 본체. key 가 없으면 pb4/pb5 와 같다 |
| `AG3S/trajopt/safe_policy.py:194-197` | `SafePolicy.closing_hand_exempt = False` | ② switch |
| `AG3S/trajopt/safe_policy.py:755` | `to.grasp_contact` record 조건에 `closing_hand_exempt` 추가 | record |
| `AG3S/trajopt/safe_policy.py:1169-1203` | `_grasp_contact`: flag on 일 때만 `"closing_exempt": closing_exempt_now()`. `CLOSING_EXEMPT_CHECKS`, `closing_exempt_now()` (CLOSING ∧ 이번 call 의 `last_grasp.evidence.checks` 에서 셋이 모두 True), `set_latch_options(**changes)` (latch config 교체, `_Confirm` 둘에도 반영) | ②, CLI 로 latch config 를 바꾸는 입구 |
| `AG3S/trajopt/safe_policy.py:1266, 1281-1285, 1313-1314` | `reach_sticky` 일 때만 `reach_context={geometry_frame, reach_state}` 를 넘긴다 | (a) record |
| `AG3S/trajopt/safe_policy.py:1322-1328` | attach 할 때, flag on 이고 `evidence.reach_sticky.same_object` 이면 `ag3s.attach(..., reach_sticky=…)`. off 이면 kwarg 자체를 넘기지 않는다 | (a) gate 일치 |
| `AG3S/trajopt/serve_safe.py:1514-1527` | `--grasp-reach-sticky` · `--grasp-drop-from-open` · `--closing-hand-exempt` | CLI |
| `AG3S/trajopt/serve_safe.py:1785-1795` | `apply_grasp_contact_flags`: 켜진 flag 만 적용하고 보고(`closing_hand_exempt`, `latch_reach_sticky`, `latch_drop_from_open`). 꺼져 있으면 반환 dict 가 pb5 와 key 단위로 같다 | Panda server 와 RB-Y1 serve_safe 가 모두 이 함수를 지난다 |
| `AG3S/adapters/panda/profile.py:221-234` | `V11AB-pb6` · `V11AB-pb7` + 근거 주석 | profile |
| `tests/trajopt/test_sl5i_grasp_recognition.py` (새 파일, 19 test) | 아래 단위 검증 | |
| `tests/trajopt/_sl5a_golden.py:389-391` | `PB_DEFAULT_OFF` 에 새 argparse key 세 개(꺼진 값)를 추가 | pb1–pb5 와 같은 방식. **꺼져 있을 때만** 비교에서 뺀다. 켜지거나 기본값이 바뀌면 여전히 차이로 잡힌다 |

`git diff --stat`: 7 file, +217 −16 (code 6 + `_sl5a_golden.py`). 새 file 은 test 1 개와 `outputs/verify/SL5i-pre/**` 다.

## 단위 검증

```bash
source scripts/setup/env.sh
CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=$AG3S_PROJECT_ROOT OMP_NUM_THREADS=2 .venv-ag3s/bin/python -m pytest -q \
  tests/trajopt/test_sl5i_grasp_recognition.py tests/trajopt/test_pb45_grasp_contact.py \
  tests/benchmark/test_panda_sl5a.py tests/benchmark/test_safelibero_client.py      # 47 passed
.venv-ag3s/bin/python -m pytest -q tests/trajopt tests/benchmark                     # 1605 passed, 47 failed, 2 errors (아래)
.venv-ag3s/bin/python -m pytest -q tests/ag3s tests/test_run_artifacts.py tests/test_policy_recorder_destination.py  # 1033 passed
.venv-ag3s/bin/python -m tests.trajopt._sl5a_golden --check tests/trajopt/data/sl5a_rby1_golden.npz           # identical, 0 differences
.venv-ag3s/bin/python -m tests.trajopt._sl5a_golden --check-serve-safe tests/trajopt/data/sl5a_serve_safe_golden.json  # identical, 0 differences
```

새 test 19 개의 범위:

- **기본값:** flag 가 꺼져 있으면 evidence 에 새 key 가 없고, `_grasp_contact` 는 None 이다.
- **sticky reach:**
  - t2e4 모양(91.1 mm 통과 → 123 mm): off 에서는 attach 하지 않고, on 에서는 다음 정지 call 에 attach 하며 두 값을 모두 기록한다.
  - 통과한 적이 없거나 다른 `manipulated_id` 면 sticky 가 적용되지 않는다.
  - reopen 으로 시도가 끝나거나 `reset()` 하면 초기화된다.
  - `revert_attach`(HELD→CLOSING)는 같은 시도로 보고 sticky 를 유지한다.
- **drop from open:**
  - t5e0 모양(SEARCHING 에서 1.0 → 이미 닫힌 0.0719 에서 latched): off 에서는 `dropped` 만 실패하고, on 에서는 `opening_before` 1.0 으로 attach 한다.
  - 열린 요청이 없으면 `source=transition` 으로 예전 값을 쓴다.
  - "명령은 열림인데 개도는 0.80" 인 요청은 열린 요청으로 세지 않는다.
- **attach gate:**
  - 130 mm target 에 `passed` 없이 `reach_sticky` 를 주면 거절한다(예전 그대로).
  - `passed` 가 있으면 붙이고 `reach_gate.decided_by=latch_pass` 를 남긴다.
  - kwarg 가 없으면 `reach_gate` key 가 생기지 않는다.
- **SafePolicy 통합:** 닫힘 시도 중 manipulated 가 130 mm capture 로 바뀌는 경우, off 는 closing 에 남는다. on 은 latch HELD, attach 거절 없음, `reach_gate=latch_pass`, sticky context `geometry_frame=4` 다.
- **closing 면제:**
  - 조건 표: CLOSING 이고 셋이 모두 참일 때만 적용된다. 하나라도 False 이거나 LATCHED/HELD/SEARCHING 이면 적용되지 않는다.
  - 분석 rig 에서 authorized 손 ↔ manipulated 행(apple)만 빠지고 bar/table/box/held 행은 그대로다. `closing_exempt_rows == 1`.
  - authorized 가 아니거나 flag False 이거나 latch 가 held/latched 면 바뀌는 것이 없다(`closing_exempt_rows == 0`).
  - key 가 없으면 record 도 pb5 와 같다.
- **profile/CLI:**
  - `pb6 == pb5 + [sticky, drop]`, `pb7 == pb6 + [closing]`.
  - pb5 argv 의 반환 dict 는 key 4 개 그대로이고 latch config 는 `LatchConfig()` 그대로다.
  - pb6/pb7 은 config 와 `_Confirm._cfg` 를 함께 교체한다.

**변경 전부터 있던 실패 (내 변경과 무관):**

- `test_panda_sl3.py`: 4 failed + 2 errors. `.venv-ag3s` 에 `openpi_client` 가 없어서다. 변경 전 run(`checks/pytest_before_subset.log`)에서도 같았다.
- `test_t43r2_held_body_cover_replay.py` 33 개와 `test_t43t_replay.py` 9 개: legacy harness 의 chunk index 가 옛 path 를 key 로 써서 나는 `KeyError` 다. 변경하지 않은 HEAD 사본(`git archive HEAD` → scratchpad)에서도 같은 42 개가 실패했다.

## 기존 동작이 그대로라는 근거

| 항목 | 변경 전 | 변경 후 |
|---|---|---|
| 회귀 기준선 legacy (`esdf_rollout … --constraint-links arms`, 단독, CPU) | 위반으로 시작 14/15 · has_target 15/15 · frame0 −29.031048280806342 mm · clearance_before sha1 `40798fb0…` | **같음** (sha1 까지) — `outputs/verify/SL5i-pre/checks/baseline_compare.txt` |
| SL5h-G 기록값 (`outputs/verify/SL5h-G/baseline/base_legacy.json`) | 위와 같음 (sha1 동일) | – |
| RB-Y1 golden `sl5a_rby1_golden.npz` (SafePolicy 경로) | identical | **identical, 0 differences** (allowlist 와 상관없는 첫 check) |
| RB-Y1 golden `sl5a_serve_safe_golden.json` | identical | 새 argparse key 3 개가 "only in current" 로 잡혔다. 꺼진 값을 `PB_DEFAULT_OFF` 에 넣은 뒤 **identical**(pb1–pb5 와 같은 방식) |
| offline latch replay `off` (pb5 latch 설정 그대로) | – | SL5g pb5-C 30 episode 의 모든 call 에서 state · checks 가 기록과 같다(**0 diff**). 다시 계산한 `closing_steps` 도 기록과 같다(**0 mismatch**) |
| 코드 경로 | – | flag off 이면 새 kwarg 를 넘기지 않는다(`reach_context`, `reach_sticky`). 새 evidence/record key 를 만들지 않는다. `apply_grasp_contact_flags` 반환 dict 도 같다 |

cuRobo 기준선은 GPU 가 필요해서 돌리지 않았다.

## offline latch replay (SL5g pb5-C, CPU)

- script: `outputs/verify/SL5i-pre/scripts/latch_replay.py`, `replay_table.py`
- 결과 file: `latch_replay.json` · `replay_table.csv` · `replay_summary.json`
- 실행 시간과 메모리: 1.2 s, RSS 42 MB, worker 1

### 입력

- call k 의 feedback 은 call k−1 의 것이다(`evidence.feedback_seq`).
- `last_command` 는 client 와 같은 식으로 다시 만들었다: `float32((1 − clip(actions_executed[k−1][n−1, 6])) / 2)`, n = `evidence.n_steps`.
- 나머지(개도, `closing_steps`, reach, `manipulated_id`/state, HELD 의 `placed`/`hold_checks`)는 기록의 evidence 를 그대로 넣었다.
- 기록에 `held_slip` 이 있으므로 `slip_detach` 를 켰다.

### replay 가 하지 않는 것

- grounding lock(SEARCHING→LATCHED)은 attention 으로 다시 계산하지 않았다. 기록의 `grasp.state` 가 `latched` 가 된 call 에서 강제로 잠갔다. 이 전이는 flag 와 상관이 없다.
- 기록된 attach 거절은 off 에서 같은 call 에서 그대로 반복한다. on 에서는 reach 이외의 사유일 때만 반복한다.

### 해석 범위

**첫 held 뒤로는 closed loop 가 갈라진다.** attach 가 제약을 바꾸기 때문이다. 그래서 정확한 비교는 첫 divergence 까지만 의미가 있다. divergence 는 flag-on latch 의 출력(state/attach/detach)이 기록과 처음 달라진 call 이다. 그 call 까지는 실제 run 과 입력이 같다. 기록과 같은 seq 에서 held 가 되면 divergence 가 아니다.

### 표

- "양손가락": SL5h MuJoCo probe 에서 그 call 의 obs 에 왼쪽·오른쪽 손가락이 모두 target 에 닿아 있는가(O/X).
- "lifted": 양손가락 O 이고 rise ≥ 30 mm 인가.
- "lifted·closing call": 첫 held 이전 CLOSING call 가운데 lifted 인 것의 수다.

| set | ep | 원래 held seq (obs, 양손가락, rise mm) | pb6 held seq (obs, 양손가락, rise mm) | 첫 divergence (원인) | 첫 lifted call (obs) | lifted·closing call off→pb6 | pb6 에도 남는 실패 check | 새 거짓 held |
|---|---|---|---|---|---|---|---|---|
| LI | t0e0 | – | 16 (100, O, 90.5) | **16** closing→held (reach sticky; 양손가락 O, lifted O) | 14 (90) | 32→2 | {settled: 2} | 아니오 |
| LI | t0e1 | 18 (110, X, 9.3) | 18 (110, X, 9.3) | **25** closing→held (reach sticky, revoke 뒤 재시도; 양손가락 O, lifted O) | 24 (140) | 0→0 | – | 아니오 |
| LI | t0e2 | – | – | – | – | 0→0 | – | 아니오 |
| LI | t0e3 | 15 (95, O, 62.3) | 15 (95, O, 62.3) | – | 15 (95) | 0→0 | – | 아니오 |
| LI | t0e4 | – | – | – | – | 0→0 | – | 아니오 |
| LI | t1e0 | – | – | – | – | 0→0 | – | 아니오 |
| LI | t1e1 | 10 (70, O, 70.6) | 10 (70, O, 70.6) | – | 10 (70) | 0→0 | – | 아니오 |
| LI | t1e2 | – | – | – | 13 (85) | 43→43 | {settled: 1, blocked: 43} | 아니오 |
| LI | t1e3 | 9 (65, X, 33.2) | 9 (65, X, 33.2) | – | 10 (70) | 0→0 | – | 아니오 |
| LI | t1e4 | 12 (80, O, 144.9) | 12 (80, O, 144.9) | – | 10 (70) | 2→2 | {settled: 2, reach: 1} | 아니오 |
| LI | t2e0 | – | – | – | 9 (65) | 39→39 | {reach: 39, settled: 3, not_empty: 1, blocked: 2} | 아니오 |
| LI | t2e1 | 12 (80, X, 68.7) | 12 (80, X, 68.7) | **52** closing→held (drop from open, revoke 뒤 재시도; 양손가락 O, lifted O) | 10 (70) | 2→2 | {blocked: 1, settled: 1} | 아니오 |
| LI | t2e2 | 10 (70, O, 1.3) | 10 (70, O, 1.3) | – | 12 (80) | 0→0 | – | 아니오 |
| LI | t2e3 | 9 (65, O, 2.3) | 9 (65, O, 2.3) | – | 38 (210) | 0→0 | – | 아니오 |
| LI | t2e4 | – | 24 (140, O, 160.0) | **24** closing→held (reach sticky; 양손가락 O, lifted O) | 22 (130) | 34→2 | {blocked: 2} | 아니오 |
| LI | t3e0 | – | – | – | – | 0→0 | – | 아니오 |
| LI | t3e1 | – | – | – | – | 0→0 | – | 아니오 |
| LI | t3e2 | – | – | – | – | 0→0 | – | 아니오 |
| LI | t3e3 | – | – | – | – | 0→0 | – | 아니오 |
| LI | t3e4 | – | – | – | – | 0→0 | – | 아니오 |
| L10 | t0e0 | 15 (85, X, 0.0) | 15 (85, X, 0.0) | – | 18 (100) | 0→0 | – | 아니오 |
| L10 | t1e0 | 14 (80, O, 40.6) | 14 (80, O, 40.6) | – | 14 (80) | 0→0 | – | 아니오 |
| L10 | t2e0 | 18 (100, O, 62.0) | 18 (100, O, 62.0) | – | 18 (100) | 0→0 | – | 아니오 |
| L10 | t3e0 | 12 (70, O, 51.0) | 12 (70, O, 51.0) | – | 12 (70) | 0→0 | – | 아니오 |
| L10 | t4e0 | – | – | – | – | 0→0 | – | 아니오 |
| L10 | t5e0 | – | 10 (60, O, 41.9) | **10** closing→held (drop from open; 양손가락 O, lifted O) | 10 (60) | 34→0 | – | 아니오 |
| L10 | t6e0 | 15 (85, O, 60.3) | 15 (85, O, 60.3) | – | 15 (85) | 0→0 | – | 아니오 |
| L10 | t7e0 | 10 (60, O, 6.4) | 10 (60, O, 6.4) | **15** closing→held (drop from open, revoke 뒤 재시도; 양손가락 O, lifted O) | 13 (75) | 0→0 | – | 아니오 |
| L10 | t8e0 | 15 (85, O, 69.5) | 15 (85, O, 69.5) | – | 15 (85) | 0→0 | – | 아니오 |
| L10 | t9e0 | – | – | – | – | 0→0 | – | 아니오 |

### 합계

| | LI (20) | LIBERO 10 (10) |
|---|---|---|
| off replay fidelity (state·checks 다른 call / 입력 불일치) | 0 / 0 | 0 / 0 |
| held 에 도달한 episode: off · sticky · drop · pb6 | 8 · 10 · 8 · 10 | 7 · 7 · 8 · 8 |
| 첫 held 이전 lifted·closing call: off → pb6 | 152 → 90 | 34 → 0 |
| pb6 divergence (모두 closing→held, 양손가락 O, lifted O) | 4 (sticky 3, drop 1) | 2 (drop 2) |
| **새 거짓 held** (flag 때문에 held 가 됐는데 양손가락 X) | **0** | **0** |
| 기록의 첫 held 가 양손가락 X 였던 것 (pb6 에서도 같은 seq, 새로 생긴 것 아님) | t0e1 s18 (rise 9.3) · t1e3 s9 (33.2) · t2e1 s12 (68.7) | t0e0 s15 (0.0) |

**SL5g 숫자와 맞춰 보기:** SL5h 의 pb5 LI lifted·closing call 은 189 개다. 그중 152 개는 첫 held 이전이다. 나머지 37 개는 첫 held 가 회수(revoke)된 뒤의 재시도에서 나왔다(t0e1 32, t2e1 5).

### pb6 에도 남는 것 (이번 변경 범위 밖, verifier/lead 참고)

- **LI t1e2:** `blocked` 가 43 call 실패한다.
  - 개도 0.027–0.037 이 명령 ≈ 0 보다 0.05(`blocked_gap`) 이상 위에서 멈추지 않는다.
  - Panda 손가락이 bowl 테두리를 거의 끝까지 닫고 쥔 경우다.
  - reach 는 sticky 로 풀렸다. reach 실패가 42 에서 0 이 됐다.
- **LI t2e0:** `reach` 가 39 call 실패한다.
  - latch 가 잠근 manipulated(id 1)가 손에서 400–600 mm 떨어져 있다(seq 11 부터 visible).
  - 손은 SEARCHING 중에 이미 bowl 을 쥐었다(seq 9, 개도 0.038). 그래서 이것은 다시 잰 값 때문이 아니라 대상 식별(identity) 문제다. sticky 로 통과할 수 없다.
  - `dropped` 38 은 drop from open 으로 풀렸다.

### pb7 (closing 면제) 이 적용될 call

pb6 replay 의 첫 held 이전 CLOSING call 가운데 `closed`·`not_empty`·`blocked` 가 모두 참인 call 을 셌다. 이 call 들에서 closing 면제가 적용된다.

- LI: 103 call. 양손가락 O 56, X 47.
- LIBERO 10: 23 call. 양손가락 O 7, X 16.

X 인 call 의 분포:

- LI: 47 중 30 이 held 에 도달하지 못한 episode 에 있다(t0e2 9, t3e2 10, t1e0 5, t3e3 4, t3e0 1, t3e1 1). 나머지 17 은 held 직전 call 이다(episode 당 1–5).
- LIBERO 10: 16 중 7 이 held 에 도달하지 못한 t9e0 에 있다. 나머지 9 는 held 직전 call 이다(episode 당 1–2).

면제가 빼는 행은 손의 authorized link 중 nearest surface 가 manipulated object 인 행(class target)뿐이다. obstacle · table · 다른 물체 행은 그대로 남는다. 다만 이 call 들에서는 쥔 물체가 아닌 manipulated object 에 대한 손의 제약도 빠진다. closed-loop 에서 확인할 항목이다.

## verifier 가 알아야 할 것

### 새로 생긴 flag · 기본값 변경

flag 는 셋이다. 기본값은 모두 꺼짐이다.

- `--grasp-reach-sticky` → `LatchConfig.reach_sticky`
- `--grasp-drop-from-open` → `LatchConfig.drop_from_open`
- `--closing-hand-exempt` → `SafePolicy.closing_hand_exempt`

profile 은 둘이다.

- `V11AB-pb6` = pb5 + `--grasp-reach-sticky --grasp-drop-from-open`
- `V11AB-pb7` = pb6 + `--closing-hand-exempt`

기존 기본값은 하나도 바꾸지 않았다.

### 새 record key (flag on 일 때만 생긴다)

- `grasp.evidence.reach_sticky`: `now`, `now_mm`, `passed{reach_mm, feedback_seq, manipulated_id, context{geometry_frame, reach_state}}`, `same_object`, `used`
- `grasp.evidence.drop_from_open`: `source`, `transition_opening`, `last_open{opening, command, feedback_seq}`
- `AG3S held_record.reach_gate`
- `to.grasp_contact.closing_exempt_rows`

### 재생산이 필요한 산출물

없다. 기존 npz/fixture 는 다시 만들지 않았다. golden fixture 도 그대로다.

### 옛 기록과의 호환

깨지지 않는다. flag off 이면 record 가 key 단위로 같다.

### 설계상 선택 (task 문구를 이렇게 해석했다)

1. **sticky 가 끝나는 시점:** sticky 는 reopen(LATCHED 로 돌아감) · revoke(`attach_revoked`) · LATCHED→CLOSING · `reset()` 에서 끝난다. `revert_attach`(AG3S 거절 → CLOSING)는 같은 시도로 본다.
2. **sticky 의 범위:** 통과 기록과 `manipulated_id` 가 같을 때만 적용한다. 다른 물체로 바뀌면 지금 값을 쓴다(fail-closed).
3. **sticky 를 AG3S 에 넘기는 조건:** latch evidence 의 `same_object` 가 참일 때만 넘긴다. AG3S 는 거리를 계속 잰다. 한도를 넘었을 때만 `decided_by=latch_pass` 로 latch 판단을 따른다.
4. **"열려 있음" 의 정의:** 실행된 마지막 명령과 측정 개도가 모두 ≥ `gripper_closed_below`(0.85) 인 요청이다. HOLD 청크(n_exec 0)는 명령이 없으므로 열림으로 세지 않는다. 추적은 `reset()` 에서만 비운다.
5. **revoke 뒤 재시도의 drop 기준:** revoke 로 LATCHED 에 돌아가 손을 열지 않고 다시 닫으면 `opening_before` 는 첫 시도 전의 열린 개도다. task 정의("닫힘 명령이 시작되기 전 마지막으로 열려 있던 개도") 그대로다.
   - 그래서 그 재시도에서는 `dropped` 가 사실상 참이다.
   - replay 의 LI t2e1 s52 와 L10 t7e0 s15 가 이 경우다. 둘 다 양손가락 O, lifted O 다.
   - attach 뒤에는 T34 revoke check 가 다시 지킨다.
6. **closing 면제 조건의 근거:** 그 call 의 `observe_grasp` 결과(`latch.last_grasp.evidence.checks`)로 판단한다. `revert_attach` 로 CLOSING 에 돌아온 call 은 checks 가 모두 참인 evidence 를 들고 있으므로 조건이 참이다.
7. **pb7 과 pb5 zone 의 겹침:** pb7 에서 pb5 grasp zone 과 겹치는 행은 면제(−1 m)가 이긴다. 그 행은 `zone_rows` 에도 함께 세진다(`min(margin, −allowance)` 가 −1 을 그대로 둔다).

### 측정 대기

- closed-loop GPU 비교(pb5 / pb6 / pb7, 변환 C)는 하지 않았다. verifier 몫이다.
- figure 도 만들지 않았다. 규칙 A 의 figure 는 verifier 가 만든다. 재료는 `replay_table.csv` 와 `latch_replay.json` 이다.

## 내가 기대하는 결과

(verifier 는 측정을 마친 뒤에 읽는다.)

- pb6 은 LIBERO t5e0, LI t0e0·t2e4 에서 bowl 을 든 직후 HELD 로 넘어갈 것으로 본다. 그러면 manipulated centroid 가 손을 따라가고, 위쪽 밀기(over-lift)가 줄 것으로 본다.
- LI t1e2(`blocked`)와 t2e0(identity)은 pb6 으로 바뀌지 않을 것이다.
- pb7 은 closing 중 쥔 구간의 TO 위쪽 행을 줄일 것으로 본다. 다만 양손가락 X 인 closing-exempt call(LI 47, LIBERO 16)에서 손 ↔ manipulated object 접촉이 늘지 않는지 함께 봐야 한다.
