# 동결 판 `pkgFrs` 의 code 점검 — flag 가 켜는 부분이 의도대로 짜였는가 (2026-11-13)

> **이 문서의 위치.** 로컬 PC 의 `doc/active/handoff_SL_AUD_20261113/` 에서 쓴 것을 branch `audit/sl-aud` 의 `doc/active/handoff_SL_AUD_20261113/` 로 옮긴 사본이다.
>
> **질문.** code 는 잘 돌아가지만 성능 (SSR · TSR) 이 구현 문제에서 오는지 의심된다. 동결 판의 flag 가 켜는 code 가 의도대로 짜였는지 확인한다 (사용자 지시 2026-11-13, 범위는 'flag 가 켜는 것만').
>
> **근거와 한계.** branch `feature/panda-benchmark`, HEAD `110ae677` 의 code 를 읽었다. 숫자는 SDM 묶음 (SDM 에서 받은 분석 묶음 (`3_raw_i19`, init 19 의 raw episode 3 개)) 의 raw 기록 3 episode, 209 call 에서 셌다. 이 PC 에는 GPU · torch · osqp 가 없어 closed loop 과 실제 QP solver 는 돌리지 못했다. 그래서 **각 결함이 SSR · TSR 을 몇 episode 움직이는지는 재지 못했다.** 3 episode 는 표본이 작다. §7 의 집계를 SDM 의 144 episode 에서 해야 크기가 나온다.

**용어.**

| 용어 | 뜻 |
|---|---|
| TSR · CAR · SSR | task 성공률 · 추가된 장애물이 밀리지 않은 비율 · 둘 다 만족한 비율 |
| chunk · call · step | VLA 가 한 번에 내는 action 10 행 · 서버 요청 한 번 (5 step 마다) · 제어 주기 하나 (0.05 s) |
| row | 로봇 충돌 구 하나와 최근접 표면 사이의 거리 제약 한 줄 |
| gate · pass-through | chunk 를 고칠지 정하는 검사 · 고치지 않고 그대로 실행하는 것 |
| action QP | gate 를 통과하지 못한 chunk 를 policy action 공간에서 `min ‖Δa‖²` 로 고치는 최적화 |
| 보정 크기 | 한 call 의 `max |Δa|`. action 범위는 ±1 이므로 1 이상이면 한 축을 절반 넘게 뒤집는 크기다 |
| backoff · 여유 (margin) | QP 가 선형화 오차를 덮으려고 더 띄우는 5 mm · 장애물 물체 row 에 더 요구하는 15 mm |
| slack | QP 가 못 지킨 row 의 남은 위반량. 벌점을 주고 허용한다 |
| latch | gripper 개도와 명령으로 정하는 쥠 단계 (SEARCHING → LATCHED → CLOSING → HELD → PLACED) |
| IK | VLA 의 EE 경로를 관절 궤적으로 바꾸는 역기구학. 이 서버는 damped least squares 를 row 마다 60 회까지 돈다 |
| 확인 등급 | **확인** = code 를 줄 단위로 읽었고 기록 또는 계산으로 대조함. **code 확인** = code 는 읽었으나 실제 run 에서의 빈도 · 크기는 모름. **추정** = 읽기만 했고 대조하지 않음 |

---

## 1. 결론

1. **의도와 다르게 짜인 곳이 있다.** flag 가 켜는 code 에서 11 곳, 문서에는 '사용' 인데 작동하지 않거나 어긋난 flag 3 개를 확인했다 (§3 – §5).
2. **가장 큰 것은 action QP 다.** 세 가지가 겹친다.
   - QP 가 chunk 의 step 0 부터 모든 row 를 지키려 하는데, action 은 앞 step 을 거의 움직이지 못한다.
   - gate 가 100 mm 까지 통과시키는 row 를 QP 는 +5 mm 로 올린다.
   - 관절 한계로 거절된 chunk 도 관절 한계 제약이 없는 이 QP 로 간다.
   - 기록된 큰 보정 (보정 크기 1 이상) 은 이 세 가지와 맞는다 (§3).
3. **놓은 뒤의 상태가 없다.** opening rule 은 `PLACED` 로 가지 않는다. 놓은 뒤 서버는 '쥐기 전' 으로 돌아가고, target 은 옛 자리의 기하로 남는다. 이 위에 놓인 flag 세 개가 함께 어긋난다 (§4).
4. **flag 밖에서 하나가 더 나왔다.** chunk → 관절 변환의 IK 가 팔이 거의 펴진 자세에서 수렴하지 않는다. 실패한 두 episode 에서 60 call 중 37, 110 call 중 36 이 그랬고 성공한 episode 는 0 이다. gate 와 action QP 는 이 관절 궤적을 그대로 쓴다 (§6).
5. **wiring 은 정상이다.** 동결 flag 87 개가 모두 Panda 경로의 config 에 도달한다. flag 가 만든 state 는 하나 (`_obstacle_margin_now`) 만 빼고 episode reset 에서 지워진다.
6. **권하는 순서.** action QP 의 세 가지를 고친 후보를 먼저 잰다. code 는 이 PC 에서 flag (기본 꺼짐) 로 넣고 unit test 까지 돌렸다 (§9). 그다음이 놓은 뒤의 상태다 (§8).

---

## 2. 무엇을 어떻게 봤나

| 항목 | 내용 |
|---|---|
| 대상 | `AG3S/adapters/panda/profile.py:1332` `FROZEN_20261112` 의 실효 flag (기본값이 아닌 option 87 개) |
| wiring | `profile.version_server_argv` + `version_server_flags` 로 argv 를 만들고 `serve_safe.build_parser()` · `trajopt_config_from_args` · `panda_ag3s_config` 로 실효 값을 뽑아 기본값과 비교 |
| code 읽기 | 세 영역으로 나눠 병렬로 읽고 (TO gate · action QP / 인식 · ESDF / latch · adapter), 보고된 결함마다 인용된 줄을 다시 읽어 대조 |
| 기록 대조 | init 19 의 raw 3 episode: Goal L-I task 0 (실패, 60 call), Long L-II task 3 (실패, 110 call), Object L-II task 2 (안전 성공, 39 call). `*_calls.jsonl` · `chunk_summary.jsonl` · `*_geom.npz` |
| 로컬 계산 | URDF 만으로 `PandaKinematics` 를 만들어 IK 재현, response model 산술 |
| 의도의 기준 | `doc/active/experiments/AG3S_FINAL_CONFIG_KO.md`, `serve_safe.py` 의 flag 도움말, 각 함수의 docstring |

script 는 `doc/active/handoff_SL_AUD_20261113/scripts_20261113/` 에 있다. 묶음 경로가 script 안에 적혀 있으므로 다른 곳에서 돌리려면 `B = ...` 줄을 고친다.

| script | 하는 일 |
|---|---|
| `audit_wiring.py` | 동결 flag 를 Panda port 목록과 대조하고 실효 TO config 를 출력 |
| `audit_ag3s_config.py` | 실효 AG3S config 를 기본값과 비교 |
| `audit_pred_vs_actual.py` | 서버가 예측한 5 step 뒤 EE 위치와 simulator 의 실제 위치 비교 |
| `audit_ik_repro.py` | 기록된 관절 상태에서 서버 IK 를 재현 |
| `audit_ik_reference.py` | 같은 target 을 적응 damping 400 회로 다시 풀어 서버 IK 와 비교 |

실행: `PYTHONUTF8=1 PYTHONPATH=. python doc/active/handoff_SL_AUD_20261113/scripts_20261113/<script> [episode 폴더 …]` (numpy 만 필요).

---

## 3. action QP 의 결함 (`ee` · `qs` · `om15p10s40` · `pg4`)

### 3.1 표

| # | flag | 무엇이 다른가 | 등급 | code |
|---|---|---|---|---|
| 1 | `--ee-action-qp` · `--ee-action-qp-scaled` | QP 가 step 0 부터 모든 row 를 지키려 한다. action 은 앞 step 을 거의 못 움직이므로 작은 부족에도 보정이 상자 끝까지 간다 | 확인 | `sqp.py:1591-1596`, `action.py:596-601` |
| 2 | `--ee-action-qp` | gate 가 통과시키는 row 를 QP 는 +5 mm 로 올린다 | 확인 | `sqp.py:1392`, `1575-1579` |
| 3 | `--ee-action-qp` + `--passthrough-limit-tolerance 0.2` | 관절 한계로 거절된 chunk 가 관절 한계 제약이 없는 QP 로 간다 | 확인 | `sqp.py:1398-1400`, `1499-1649` |
| 4 | `--obstacle-row-priority 10` | `-scaled` 형태에서는 거의 작동하지 않는다 | 확인 (산술) | `sqp.py:1604-1610` |
| 5 | `--ee-action-qp-progress 0.4` | 놓은 뒤에도 '쥐기 전' 으로 보고 옛 집는 자리 쪽으로 끈다 | 확인 | `sqp.py:1659-1675` |
| 6 | `--ee-action-qp` | 재선형화의 조기 종료 조건이 부호가 반대다 | 확인, 영향 작음 | `sqp.py:1634-1636` |

### 3.2 1 번 — 앞 step 의 row

**code.** `ee_action_map` 은 action 행 `j` 의 변화가 step `k ≥ j` 의 EE 위치를 `β · 0.05 · (1 − α^(k−j+1)) / (1 − α)` 만큼 움직인다고 본다 (`action.py:596-601`, α 0.717 · β 0.0706). step 0 의 계수는 3.5 mm 다.

| | step 0 | step 1 | step 2 | step 3 | step 4 |
|---|---:|---:|---:|---:|---:|
| action 1 단위를 계속 줄 때 EE 이동 (mm) | 3.5 | 9.6 | 17.5 | 26.7 | 36.8 |
| 손에 붙은 구를 5 mm 옮기는 데 필요한 보정 크기 | 1.42 | 0.62 | 0.35 | 0.23 | 0.17 |
| 20 mm 옮기는 데 필요한 보정 크기 | 5.66 | 2.46 | 1.42 | 0.94 | 0.68 |

- QP 는 쓰인 row 전부에 `value + G·Δa + s ≥ backoff` 를 건다 (`sqp.py:1591`). step 을 가리지 않는다.
- step 0 에서 floor 에 딱 걸린 row 는 backoff 5 mm 만으로 보정 1.42 를 요구한다. 상자 `|a_ref + Δa| ≤ 1` 안에서 풀리지 않는다.
- 장애물 물체 row 는 floor 에 여유 15 mm 와 backoff 5 mm 가 더해진다. step 1 에서 여유 안으로 10 mm 들어온 row 도 풀리지 않는다.
- class docstring (`sqp.py:100-103`) 과 `--ee-action-qp-scaled` 의 도움말은 "action gradient 가 0 인 row (step 0 = q_now) 는 QP 에서 뺀다" 고 적는다. Panda 에서 step 0 은 `q_now` 가 아니라 action 행 0 을 실행한 뒤의 자세다. gradient 가 3.5e-3 이어서 검사 `gn > 1e-9` 에 걸리지 않고 row 가 남는다.

**풀리지 않을 때.** `-scaled` 형태는 row 를 action gradient 의 크기로 나눈다. slack 이 action 단위가 되고 벌점은 1 차 100 · 2 차 1e4 다 (`sqp.py:126-127`). 한 row 가 풀리지 않으면 slack 을 줄이는 쪽이 `‖Δa‖²` 보다 항상 싸다. 그래서 gradient 가 0 이 아닌 action 성분이 모두 상자 끝으로 간다. 보정이 row 의 법선 방향이 아니게 된다.

- 축소 모형 (같은 비용 · 상자, step 0 row 하나): `a_ref = [0.3, −0.2, −0.5]`, 법선 +x, 부족 5 mm → `Δa = [0.7, 1.2, 1.5]`. 부족 2 mm 에서는 법선 방향을 유지한다.
- 이 부분은 OSQP 로 돌리지 못했다. 기록에는 30 성분 중 27 개가 ±1 에 붙은 call 이 있다 (Goal L-I task 0 call 27, slack 70 mm).

**기록.** IK 가 수렴한 action QP call 72 개 (3 episode) 를 gate 가 찍은 가장 나쁜 row 의 step 으로 나눴다.

| 가장 나쁜 row 의 step | call | 위반 중앙값 (mm) | 보정 크기 중앙값 | 보정 1 이상 | slack 1 mm 초과 |
|---|---:|---:|---:|---:|---:|
| 0 | 2 | 8.9 | 1.37 | 2 | 2 |
| 1–2 | 3 | 7.5 | 1.11 | 2 | 2 |
| 3–4 | 9 | 19.8 | 1.65 | 9 | 8 |
| 5–9 | 58 | 15.9 | 0.51 | 11 | 4 |

step 0–4 는 14 call 중 13 call 이 보정 1 이상이고, step 5–9 는 58 call 중 11 call 이다. 가장 나쁜 row 가 QP 를 묶는 row 와 같다는 보장은 없으므로 방향을 보여 주는 표다.

### 3.3 2 번 — gate 와 QP 의 기준이 다르다

| row | gate 가 거절하는 실제 거리 | QP 가 지키려는 실제 거리 |
|---|---|---|
| 장애물 물체 row, 팔 link | 25 mm 미만 (인내 한도 뒤 10 mm) | 30 mm (인내 한도 뒤 15 mm) |
| 장애물 물체 row, 손바닥 껍질 구 | 20 mm 미만 (5 mm) | 25 mm (10 mm) |
| 장애물 물체 row, 쥔 물체 구 | 15 mm 미만 (0 mm) | 20 mm (5 mm) |
| 그 밖의 row (target · 받침면 · 목적지 · 쥔 물체의 상) | −100 mm 미만 (장애물 분류면 −90 mm) | +5 mm (장애물 분류면 +15 mm) |

'실제 거리' 는 구 표면에서 장 표면까지다. 값은 code 에서 읽었다 (`linearize.py:1085-1095`, `1133`, `sqp.py:1464`, `1531`).

- 마지막 행이 문제다. chunk 가 어떤 이유로든 QP 에 들어가면, gate 가 받아들이던 접촉 row 도 +5 mm 로 올려야 한다. 3.2 와 겹쳐 step 0 부터다.
- 이를 겨눈 option 이 이미 있다: `--ee-action-qp-gate-rows` (SL-SOTA4, `sqp.py:1551-1556`). 동결 판의 argv 에는 없다.
- **이 문제와 그 option 은 2026-11-11 에 이미 기록됐다** (계획서 §9.8; 이 문서의 첫 판은 이것을 빠뜨렸다). 당시 관측: QP call 894 개 중 469 개가 반복 한도까지 갔고 그때 보정 크기 중앙값이 1.27 – 1.55. `-gr` 판은 E1a task 0 의 10 ep 에서 TSR 5 · CAR 6 · SSR 4 로, 없는 판 (9 · 8 · 7) 보다 나빠 쓰지 않기로 했다. 그 run 은 `-qs` (행 정규화 QP) · `-hr150` · `-hb` 이전의 base 였다.
- `-gr` 이 나빴던 이유의 후보를 §9 의 offline demo 가 보여 준다: 그 option 은 접촉 row 를 −100 mm 까지 풀어, 장애물 row 를 지키려는 보정이 손을 탁자 쪽으로 미는 것을 막지 못한다 (demo 에서 행 0 의 z 보정이 −1). 그래서 §9 의 SL-AUD1 은 '풀지 않고, policy chunk 보다 나빠지지 않게만' 둔다.
- grasp zone release · held-hand-exempt · held-beneath 로 풀린 row 는 여유가 −1 m 가 되어 QP 에 들어오지 않는다. 영향이 없다.

**기록.** 여유만 조금 모자라 거절된 call 의 보정 크기 (IK 가 수렴한 call):

| episode · call | 여유 부족 (mm) | 가장 나쁜 row 의 실제 여유 (mm) | 보정 크기 |
|---|---:|---:|---:|
| Goal L-I task 0 · 4 | 0.9 | 14.1 | 1.04 |
| Goal L-I task 0 · 10 | 4.6 | 10.4 | 1.20 |
| Long L-II task 3 · 38 | 3.3 | 11.7 | 1.69 |
| Long L-II task 3 · 44 | 7.5 | 7.5 | 1.71 |

### 3.4 3 번 — 관절 한계 거절

- gate 의 마지막 검사가 관절 한계다 (`sqp.py:1398-1400`). 넘으면 사유 `limit` 로 거절한다.
- 거절된 chunk 는 `_solve_ee_action` 으로 간다. 이 QP 의 제약은 충돌 row 와 action 상자뿐이다 (`sqp.py:1613-1617`). 관절 한계를 고칠 수 없고, 3.3 의 기준으로 충돌 row 만 올린다.
- 기록: Goal L-I task 0 call 43 · 44. 장애물 물체 row 위반 0, slack 0 인데 보정 크기 0.90 · 1.61 이다. 두 call 모두 §6 의 IK 실패 구간이다.

### 3.5 4 번 — priority

- `sqp.py:1609-1610` 은 slack 의 1 차 항만 10 배로 만든다 (100 → 1000). 2 차 항 1e4 는 모든 row 가 같다.
- 서로 부딪히는 두 row 사이에서 priority 가 옮기는 양은 `(1000 − 100) / (2 · 1e4) = 0.045` action 단위, 거리로 0.2 – 0.5 mm 다.
- 의도 (`sqp.py:1538-1541`) 는 "깊은 장애물 row 하나가 얕은 접촉 row 여럿과 맞바뀌지 않게" 다. `-scaled` 형태에서는 그 맞바꿈이 그대로 일어난다. 3.3 때문에 접촉 row 가 많이 들어오므로 더 그렇다. CAR 쪽의 손실이다.

### 3.6 5 번 · 6 번

- **5 번.** `_progress_direction` 은 latch 가 `held` 면 위, `closing` 이면 없음, 그 밖은 모두 target 중심 쪽이다. 놓은 뒤 latch 는 `LATCHED` 이고 target 은 옛 자리다 (§4.1). Goal L-I task 0 에서 놓은 뒤 24 call 이 이 mode 였다.
- **5 번의 크기.** 비용이 `½‖Δa‖² − g·u·Δa` 형태라 제약이 없으면 10 행 모두에 `Δa = 0.4 · u` 가 들어간다. docstring 과 도움말의 식 (`min ‖Δa‖² − g·Σu·Δa`) 으로는 0.2 다. 위반이 20 mm 이상이면 실행 5 행 동안 약 15 mm 를 민다.
- **6 번.** `full_violation` 은 '양수 = 여유' 를 돌려준다 (`linearize.py:1901-1904`). 다른 호출은 모두 부호를 뒤집는데 (`sqp.py:1092`, `1901`, `1974`) 여기만 그대로 `≤ violation_tolerance` 와 비교한다. slack 이 0 이면 둘째 반복에서 거의 항상 끝난다. τ 가 100 mm 라 결과가 달라지는 경우는 드물다.

---

## 4. latch · 인식의 결함

### 4.1 `PLACED` 가 없다 (`--grasp-opening-rule`)

| 항목 | 내용 |
|---|---|
| code | `grasp_latch.py:664-665` 에서 opening rule 이 먼저 return 한다. 그 detach 는 `LATCHED` 로 간다 (`:873`). `PLACED` 를 넣는 줄 (`:572`, `:805`, `:813`) 에는 닿지 않는다 |
| 의도 | 최종 구성 문서 §1 3 단계는 "… → HELD → PLACED", §3.2 는 `--placed-no-seed-certify` 를 "사용 (`--subtask-from-latch` 가 home 을 만듦)" 으로 적는다. detach 가 `LATCHED` 로 가는 것 자체는 opening rule 의 설계다. 어긋나는 것은 그 위에 놓인 flag 들이다 |
| 기록 | Goal L-I task 0: call 31 에 detach, call 33 부터 label `pick`, target 은 call 59 까지 `occluded` 로 옛 중심 `[0.53, 0.02, 0.05]` |
| 등급 | 확인 |

놓은 뒤 일어나는 일:

- `--subtask-from-latch`: `home` 이 나오지 않는다. label 이 `place` → `pick` 으로 바로 돌아간다.
- `--placed-no-seed-certify`: 발동 조건이 없다. 서버의 inert 경고 (`server.py:1142`) 는 `--subtask-from-latch` 가 꺼졌을 때만 뜨므로 log 에도 남지 않는다.
- `--latched-identity-hold`: '안 보임' 만으로는 identity 를 바꾸지 않는다 (`target_grounding.py:1635-1663`). `target_lost_frames` 가 `None` 이라 '잃음' 도 되지 않는다. target 이 옛 자리에 고정된다.
- `--grasp-zone-release`: 옛 자리 100 mm 안의 손 row 가 풀린다.
- `--ee-action-qp-progress`: §3.6 의 5 번.
- target-free 층과 ROI carve 상자가 옛 자리에 놓인다. 방금 놓은 물체는 보통 표면이고, 목적지 15 mm 안 규칙으로만 장애물 집합에서 빠진다.

### 4.2 표

| # | flag | 무엇이 다른가 | 등급 | code |
|---|---|---|---|---|
| 7 | `--scene-obstacle-target-cut` | 쥔 동안과 놓은 뒤에도 옛 target 중심에서 '다시 자른 물체' 를 장애물 집합에서 뺀다. 물체가 떠난 자리의 받침 물체나 이웃 장애물이 빠질 수 있다 | code 확인 | `pipeline.py:1208-1218`, 호출 `3291-3300` |
| 8 | `--scene-obstacle-objects` | 최근 3 call 합집합의 앞 두 call 분은 지금의 제외 규칙으로 다시 거르지 않는다 | code 확인 | `pipeline.py:1261-1264` |
| 9 | `--destination-held-leader` | 점수 1 위 cluster 하나만 본다. 1 위가 손 60 mm 안의 것이면 연속 3 frame 조건이 0 으로 돌아간다 | code 확인 | `target_grounding.py:1822-1825`, `admissibility.py:314-332` |
| 10 | `--close-seed-grounding` × `--attention-split-partition` | 자른 점에 거리로 매긴 attention 을 partition 이 읽어 물체 하나를 둘로 가를 수 있다 | code 확인, 합성 점군에서 재현 | `pipeline.py:683-684`, `attention_split.py:333-336` |
| 11 | `--passthrough-obstacle-class-tests` | 쥐기 전 손 row 는 거리를 target 을 뺀 층에서, 방향은 target 이 든 층에서 읽는다. 문서의 'target 부피' 검사는 돌지 않는다 | code 확인, 영향 미확인 | `linearize.py:2089`, `2121-2124`, `2100`, `3002` |

- **7 번.** 도움말의 의도는 "그릇에 붙은 장애물이 같이 빠지지 않게" 다. test (`test_sl_pe11_target_cut.py`) 는 쥐기 전만 본다. 빠진 물체의 row 는 장애물 물체 row 가 아니게 되어 100 mm 허용을 받는다. CAR 쪽의 손실이다.
- **8 번.** 목적지가 call N 에 등록되면 접시의 N−1 · N−2 점이 N+2 까지 집합에 남는다. 쥔 물체 인정과 target 전환도 같다. `--obstacle-surface-tol-mm 40` 이라 그 점 40 mm 안의 row 가 장애물 물체 row 가 된다. 기록의 '목적지 등록 2 – 16 call 지연' 에 2 call 을 더한다.
- **9 번.** 도움말은 "쥔 동안 손 (60 mm) 밖 attention 1 위 cluster" 다. code 는 전체 1 위만 본다. 1 위가 손 근처의 쥔 물체 조각이고 '잡을 수 있음' 이면 streak 가 0 이 된다. 한 번 잘못 등록되면 바뀌지 않는다 (`admissibility.py:300-312`).
- **10 번.** 거리 가중은 "split 뒤 손 아래 조각이 이기게" 하려던 것이다 (교체 기록 2026-11-08 22:19). 합성 점군의 결과: 60 × 60 × 200 mm 상자의 한쪽 끝 seed → 갈림 (121 + 119 점), 반만 보이는 140 mm 그릇의 테두리 seed → 갈림 (83 + 79 점), 다 보이는 그릇 → 안 갈림. 가중이 없던 때 (모두 1.0) 는 어느 것도 갈리지 않는다. 실제 점군에서의 빈도는 모른다.
- **11 번.** `--target-permission-strict` 가 넣는 `target_membership = "layer_diff"` 는 `in_target_volume` 안에서만 읽힌다. 그 함수는 `scene.target_volume` 이 있을 때만 불리고, 그것은 `--target-volume-exempt` 가 있어야 생긴다. 동결 판에는 없다.

### 4.3 읽기만 한 것 (추정)

| flag | 내용 | code |
|---|---|---|
| `--held-body-cover` | 문서는 body cover 가 self-filter 에 들어간다고 적지만, self-filter 는 맞춘 구 (반지름 ≤ 39 mm) 를 받는다. 테두리를 쥔 140 mm 그릇의 먼 벽이 점군에 남는다 | `pipeline.py:4188-4200` |
| `--held-point-model-mm 20` | 쥔 물체의 질의 구를 받침면까지 채울 때 받침면이 탁자 평면이다. 100 mm 이하 높이의 물체 위에서 집으면 모델이 실제 바닥 아래로 늘어난다 | `held_object.py:909-912` |
| `--supported-max-bottom-mm 60` | 기억해 둔 국소 받침면이 수평으로 무한하다. 같은 높이의 어느 cluster 든 '받쳐짐' 을 통과한다 | `pipeline.py:639-643`, `target_grounding.py:1875-1880` |
| `--grasp-opening-rule` 의 reach | reach 한계 120 mm 는 RB-Y1 의 값이다. 닫는 동안 한 번도 통과하지 못하면 `CLOSING` 에 머문다. `--grasp-opening-rule-reach-sticky` 는 한 번 통과한 뒤만 덮는다 | `grasp_latch.py:880`, `886-890` |
| `--grasp-opening-rule` 의 개도 | 개도는 손가락 하나의 관절값이다. attach 는 0.03 이상, 빈손 detach 는 0.02 미만이라 치우쳐 쥔 물체에서 어긋날 수 있다 | `grasp_latch.py:865`, `879` |

---

## 5. 문서와 다르게 작동하는 flag

| flag | 문서 | 실제 | 등급 |
|---|---|---|---|
| `--placed-no-seed-certify` | §3.2 '사용' | 발동하지 않는다 (§4.1) | 확인 |
| `--hold-follow-to 3` | §3.9 '사용' | 발동하지 않는다. 위반이 100 mm 를 넘을 때만 불리는데 (`safe_policy.py:3546`), 여유가 10 mm 이하이므로 그때는 `d − r < −90 mm` 이고 항상 '관통' 으로 거절된다 (`:3878`). 100 mm 초과는 모두 HOLD 다 | 확인 (산술) |
| `--obstacle-row-patience 6` | "6 call 연속 머물면" | 7 번째 call 에서 풀린다 (`sqp.py:1457` 의 `>`; 기록도 `blocked_calls` 7 에서 `relaxed`). 다른 사유 (기하 미인증 · 일반 위반) 로 거절된 call 에서는 counter 가 멈춘다. `_obstacle_margin_now` 는 `reset()` 에서 지워지지 않는다 (`sqp.py:274-283`) | 확인 |
| `--obstacle-row-priority 10` | "slack 가중 10" | `-scaled` 형태에서는 0.2 – 0.5 mm 의 차이 (§3.5) | 확인 (산술) |
| 최종 구성 문서 §1 5.2 · §3.9 | τ_obs 10 mm | 실효 값은 0 mm (§3.11 과 argv 가 맞다) | 확인 |

---

## 6. flag 밖에서 나온 것

사용자 지시의 범위 밖이다. 다만 §3 의 flag 가 이 값들을 직접 쓰므로 적는다.

### 6.1 IK 가 팔이 펴진 자세에서 수렴하지 않는다

**기록.** `conversion.ik_iters_max` 가 한도 60 인 call 과, 그때의 위치 오차 `ik_pos_err_m_max`.

| episode | 결과 | call | 한도에 걸린 call | 위치 오차 최대 |
|---|---|---:|---:|---:|
| Goal L-I task 0 | 실패 | 60 | 37 | 402 mm |
| Long L-II task 3 | 실패 | 110 | 36 | 56 mm |
| Object L-II task 2 | 안전 성공 | 39 | 0 | 0.01 mm |

**code.** `PandaKinematics.ik` (`action.py:198-215`) 는 damping 1e-3, step clip 0.2 rad, 60 회, 관절 한계 clip 이다. 수렴하지 않아도 예외 없이 결과를 돌려준다. `ChunkToJointPolicy.infer` (`server.py:237-243`) 는 수렴 여부를 보지 않고 그 관절 chunk 를 TO 의 reference 로 넘긴다. action QP 의 map (`ee_action_map`) 도 그 row 의 Jacobian 으로 만든다.

**원인.** Panda 의 팔꿈치 특이점은 관절 4 가 약 −0.47 rad 일 때다 (link offset 82.5 mm 때문에 관절 한계 −0.07 이 아니다; `atan(0.0825/0.316) + atan(0.0825/0.384)` 로 계산). Goal L-I task 0 의 접시는 base 에서 x 0.73 m 이고, 그 위의 손 자세는 도달 한계 근처다. call 26 에서 관절 4 는 −0.507, Jacobian 의 최소 특이값은 0.006 이다.

**재현** (`audit_ik_repro.py`, `audit_ik_reference.py`). 기록된 관절 상태에서 한 방향 action 을 10 행 준 chunk:

| 상태 | 방향 | 서버 IK 의 위치 오차 (행 0 … 9, mm) | 400 회 기준 solver |
|---|---|---|---|
| call 23 (관절 4 = −1.43) | +x | 0 … 0, 행 9 에서 9.6 | — |
| call 26 (−0.51) | +x | 3.2 … 90.6 | 3.2 … 88.9 (실제로 닿지 않음) |
| call 27 (−0.17) | −x (당기기) | 0, 18, 69, 167, 260, … 224 | 0, 3.3, 10.7, … 83.3 |
| call 45 (−0.07) | −x | 20, 63, 178, 347, … 322 | 3.3, 9.0, … 87.4 |

- 미는 방향 (+x) 의 오차는 target 이 실제로 닿지 않아서다. 기준 solver 도 같다.
- 당기는 방향의 오차는 solver 가 발산한 것이다. 회전 오차가 1 rad 까지 간다.

**실제 궤적과의 비교** (`audit_pred_vs_actual.py`). 서버가 실행 action 으로 예측한 5 step 뒤 EE 위치와 simulator 의 실제 위치:

| | 예측한 이동 (mm) | 실제 이동 (mm) | 차이 (mm) |
|---|---:|---:|---:|
| 보통의 call | — | — | 3 – 12 |
| Goal L-I task 0 call 27 | 215.7 | 62.6 | 155.9 |
| call 28 | 128.1 | 52.6 | 81.9 |
| call 31 | 175.0 | 81.4 | 214.4 |

**이 episode 에서의 결과.**

- gate 거절 41 call 중 22 call 이 'link5 대 장애물' 이고 모두 IK 가 한도에 걸린 call 이다. 팔의 관절 배치가 틀린 채 link5 의 위치를 검사한 것이다.
- 보정 크기 1 이상인 21 call 중 17 call 이 IK 실패 구간이다.
- 그릇은 접시 중심에서 약 50 mm 벗어난 곳에 놓였다 (그릇 `[0.765, 0.030]`, 접시 `[0.721, 0.006]`).
- Long L-II task 3 은 다르다. 보정 1 이상인 34 call 중 20 call 은 IK 가 수렴한 call 이다 (mug 가 조작 대상 후보에서 빠지는 문제, 기존 기록).

보통의 call 에서도 5 step 예측 오차가 3 – 12 mm 다. 지키려는 여유 (15 mm) 와 같은 크기다.

### 6.2 그 밖

| 항목 | 내용 | code |
|---|---|---|
| `-lsnative` 판 (LIBERO-Safety) | `ee_action_map` 이 response model 을 받지 않고 기본 값 (β 0.0706) 으로 map 을 만든다. forward 변환은 fork model (β 0.2026) 을 쓴다. QP 가 보정의 효과를 2.4 – 2.9 배 작게 본다. LIBERO · SafeLIBERO run 에는 해당 없음 | `action.py:582`, `server.py:242` |
| run 안의 위치 의존 | client 의 `env.reset()` 이 전역 numpy RNG 로 fixture 위치를 다시 뽑는다고 주석에 적혀 있다. env 는 task 마다 한 번 만든다. 서버 state 가 아니라 이쪽이 후보다. LIBERO 쪽 code 는 읽지 않았다 | `libero_pi05_client.py:199-206` |
| gripper 명령의 중간 값 | latch 의 닫힘 문턱은 정규화 명령 0.85 (action −0.7) 이고 simulator 는 부호로 움직인다. 그 사이 값이 나오면 어긋난다. 기록 206 call 에서 0 건이라 문제로 올리지 않는다 | `grasp_latch.py:844-845` |

---

## 7. 의도대로인 것 · 확인하지 못한 것

### 7.1 읽었고 의도와 맞는 것

| 영역 | flag |
|---|---|
| wiring | 87 개 전부 (`audit_wiring.py`, `audit_ag3s_config.py`) |
| gate | `--passthrough-when-clear` 의 검사 순서, `--violation-tolerance-mm 100`, `--passthrough-obstacle-tolerance-mm 0`, `--passthrough-obstacle-full-horizon`, `--lookahead obstacles` · `--lookahead-gate-mm 20`, `--obstacle-margin 0.010`, `--palm-shell-margin 5`, `--occluded-certify`, `--no-target-certify`, `--to-fast-eval` 의 cache |
| row 면제 | `--grasp-zone-release`, `--grasp-zone-mm 10`, `--held-beneath-contact`, `--held-hand-exempt`, `--obstacle-hand-beneath` |
| action QP | map 의 행 순서, 상자, `--ee-action-qp-translation-only` 의 열 고정, `--ee-action-qp-progress` 의 부호 · frame, `clip(a_ref + Δa)` 전송 |
| 인식 | `--close-seed-grounding` 의 seed 계산 · 즉시 채택 · closing 잠금, `--destination-after-held`, `--scene-obstacle-held-exclude`, `--scene-obstacle-held-motion` 의 frame, `--closing-keep-geometry`, `--kept-geometry-follow`, `--split-supersedes-anchor`, `--target-exclusion ball-owned` · `--owned-max-dist-voxels 3`, `--nontarget-evidence surface`, `--roi-carve` 계열, `--reach-at-capture` |
| latch | `--grasp-opening-rule-moved`, `--grasp-opening-rule-reach-sticky`, `--grasp-reach-sticky`, `--latch-phase` 의 mapping |
| 문서대로 inert | `--grasp-drop-from-open`, `--held-slip-detach`, `--closing-hand-exempt` |
| reset | flag 가 만든 state 는 `_obstacle_margin_now` 만 남는다 |

### 7.2 확인하지 못한 것

- §4.2 의 7 – 11 번과 §4.3 이 실제 run 에서 얼마나 자주 걸리는지.
- §3.2 의 '상자 끝으로 밀림' 을 OSQP 로 재현하는 것.
- 읽지 않은 code: `AG3S/trajopt/qp.py`, `AG3S/ag3s/fields/curobo_field.py` 의 질의 쪽, `--esdf-margin` · `--no-self-collision` · `--destination-margin-mm` 의 TO 쪽, `cap_held_spheres`.
- 그림. 이 PC 에서는 run 을 새로 돌릴 수 없어 장면 그림을 만들지 않았다.

---

## 8. 다음에 할 일

### 8.1 SDM 에서 셀 것 (CPU, 기록만 읽음)

`ms-pkgFrssdm-*` 112 ep 와 `ms-tab10sdm-*` 32 ep 의 `*_calls.jsonl` 에서 episode 마다 아래를 세고, 결과 (안전 성공 · 잃음 · 둘 다 실패) 와 나란히 놓는다.

| 셀 값 | key | 답하는 질문 |
|---|---|---|
| IK 가 한도에 걸린 call 수 | `conversion.ik_iters_max == 60` | §6.1 이 3 episode 밖에서도 실패와 함께 나오는가 |
| 그 call 의 위치 오차 | `conversion.ik_pos_err_m_max` | 닿지 않는 target (수십 mm) 과 발산 (100 mm 이상) 의 비율 |
| 사유 `limit` 의 call 수와 보정 크기 | `to.passthrough_clear.reason`, `to.ee_action_qp.max_abs_delta_action` | §3.4 의 빈도 |
| 가장 나쁜 row 의 step 별 보정 크기 | `to.passthrough_clear.obstacle_rows.worst.step` | §3.2 의 표를 144 ep 로 |
| 여유만으로 거절된 call 의 보정 크기 | `…worst.clearance_m ≥ 0` | §3.3 의 크기 |
| detach 뒤의 거절 call 수와 progress mode | `grasp.opening_rule.decision == "detach"` 이후, `to.ee_action_qp.progress.mode` | §4.1 의 크기 |

### 8.2 고치는 순서 (권고)

| 순서 | 일 | 이유 |
|---|---|---|
| 1 | 접촉 row 를 policy chunk 보다 나빠지지 않게만 둔다 (SL-AUD1 `-rf`) | §3.3. 기존 `-gr` 은 11-11 에 탈락했다 (§3.3). 구현됨 (§9) |
| 2 | action QP 에서 action 이 움직이지 못하는 row 를 뺀다 (SL-AUD3 `-mr8` · `-mr12`) | §3.2. 구현됨 (§9) |
| 3 | 사유 `limit` 인 chunk 를 action QP 로 보내지 않는다 (SL-AUD2 `-lp`) | §3.4. 구현됨 (§9) |
| 4 | 놓은 뒤의 상태를 만든다 (opening rule 의 detach 뒤 target 기하를 버리고 progress 를 끈다) | §4.1, §3.6 의 5 번 |
| 5 | IK 가 수렴하지 않은 call 을 구분한다 (기록에 표시하고, gate 가 그 관절 row 를 믿지 않게) | §6.1. 범위 밖이므로 넣을지는 사용자 판단 |
| 뒤로 | `--obstacle-row-priority` 의 2 차 항, patience 의 세 가지, §4.2 의 7 – 11 번 | 크기를 모른다. 8.1 의 집계와 별도 기록이 먼저다 |

1 – 3 은 `doc/active/handoff_SL_AUD_20261113/2026-11-13_SDM_handoff_final_plan_KO.md` 의 후보 B (보정 크기 상한) 와 같은 증상을 겨눈다. 상한은 큰 보정을 자르기만 하고, 1 – 3 은 큰 보정이 나오는 이유를 없앤다. 후보 B 를 재기 전에 1 을 먼저 재면 둘을 가를 수 있다.

---

## 9. 구현한 것 — 후보 flag 여덟 개: action QP 다섯, 놓는 구간 셋 (2026-11-13 밤 추가)

사용자 지시 (2026-11-13 "진행하세요. 설치해도 됩니다") 로 이 PC 에서 code 를 고치고 unit test 까지 돌렸다. **closed loop 은 돌리지 않았다.** 아래 flag 는 모두 기본 꺼짐이고, 꺼져 있으면 동결 판의 결과 · 기록 key 가 그대로다.

step 이름 `SL-AUD1` – `8` 은 이 문서에서 임시로 붙인 것이다. SDM 의 번호 규칙에 맞춰 바꿔도 된다. 시험 순서 · 기준 · 일정은 `2026-11-13_SL-AUD_test_plan_KO.md` 가 정본이다.

### 9.1 무엇을 넣었나

| step | flag (sqp key) | 판 segment | 하는 일 | 겨누는 결함 |
|---|---|---|---|---|
| SL-AUD1 | `--ee-action-qp-reference-floor` (`ee_action_reference_floor`) | `-rf` | 장애물 물체 row 가 아닌 ESDF row 를 `min(backoff, policy chunk 에서의 clearance)` 에 둔다. QP 가 그 row 를 policy chunk 보다 나쁘게 만들지 못하고, 고치지도 않는다 | §3.3 |
| SL-AUD2 | `--ee-action-qp-limit-passthrough` (`ee_action_limit_passthrough`) | `-lp` | 통과 판정이 관절 한계 하나로만 거절했고 (사유 `limit`) 다음이 action QP 인 chunk 는 그대로 통과시킨다 | §3.4 |
| SL-AUD3 | `--ee-action-qp-min-reach-mm X` (`ee_action_min_reach`) | `-mr8` · `-mr12` | 고칠 수 있는 action 성분에 대한 gradient 크기가 X mm (action 1 단위당) 보다 작은 row 를 QP 에서 뺀다. 최종 검사는 그 row 도 본다 | §3.2 |
| SL-AUD4 | `--ee-action-qp-audit-fixes` (`ee_action_audit_fixes`) | `-af` | (a) 조기 종료가 `full_violation` 의 부호를 맞게 읽는다 (b) QP 가 쓰는 장애물 row 여유는 이번 call 의 통과 판정이 정한 값이다 | §3.6 의 6 번, §5 의 patience |
| SL-AUD5 | `--ee-action-qp-max-delta X` (`ee_action_max_delta`) | `-md5` | action 한 성분의 보정을 X 로 자른다 (전달 문서의 후보 B). `|a_ref + Δa| ≤ 1` 상자는 그대로 | 큰 보정 (증상) |
| SL-AUD6 | `--scene-obstacle-history-refilter` (`clustering.scene_obstacle_history_refilter`) | `-ohr` | 장애물 집합의 앞 두 요청 분에서, 이번 요청이 빼는 점 15 mm 안의 점도 뺀다 | §4.2 의 8 번 |
| SL-AUD7 | `--destination-leader-outside-hand` (`clustering.destination_leader_outside_hand`) | `-dlo` | 목적지 후보를 손 60 mm 밖 cluster 중 점수 1 위로 본다 | §4.2 의 9 번 |
| SL-AUD8 | `--ee-action-qp-progress-release-off` (`SafePolicy.progress_release_off`) | `-pro` | opening rule 이 놓은 뒤 다음 닫기 시도 전까지 진행 선호의 'target 쪽' 을 끈다. target 기하는 버리지 않는다 | §3.6 의 5 번, §4.1 |

- **SL-AUD1 과 `-gr` 의 차이.** `-gr` (SL-SOTA4) 은 접촉 row 를 −100 mm 까지 푼다. SL-AUD1 은 풀지 않는다. policy chunk 가 이미 −30 mm 인 row 는 −30 mm 에, +2 mm 인 row 는 +2 mm 에, +20 mm 인 row 는 예전처럼 +5 mm 에 묶인다. 두 flag 는 같이 켤 수 없다.
- **SL-AUD3 의 문턱.** hand 에 붙은 구의 gradient 크기는 step 0 – 4 에서 3.5 · 7.0 · 10.6 · 14.0 · 17.3 mm 다. 8 은 step 0 – 1, 12 는 step 0 – 2 의 hand row 를 뺀다. 팔 link row 는 EE 병진으로 덜 움직이므로 더 뒤 step 까지 빠진다. `--ee-action-qp-translation-only` 로 고정된 회전 성분은 세지 않는다.
- **SL-AUD2 의 근거.** 사유 `limit` 는 앞의 검사 (기하 인증 · 일반 위반 100 mm · 장애물 물체 row) 를 모두 통과했다는 뜻이다. 관절 한계는 verdict 에 들어가지 않는다 (`_finish` 는 note 만 남긴다).

### 9.2 바뀐 파일 과 등록한 판

| 파일 | 내용 |
|---|---|
| `AG3S/trajopt/sqp.py` | `solve` (SL-AUD2 · 4b), `_solve_ee_action` (SL-AUD1 · 3 · 4a), `_linearize_with_obstacle_floor` (SL-AUD1), `_finish_passthrough` 의 note 인자 |
| `AG3S/trajopt/config.py` | `SqpConfig` 의 key 네 개 · 검증 · 기본값이면 dump 에서 생략 |
| `AG3S/trajopt/serve_safe.py` | flag 네 개 · 조합 검사 · `passthrough_overrides` |
| `AG3S/adapters/panda/server.py` | `PORTED_FINAL_CONFIG_DESTS` 에 여덟 개 |
| `AG3S/ag3s/runtime/pipeline.py`, `AG3S/ag3s/stages/target_grounding.py` · `admissibility.py`, `AG3S/ag3s/config.py` | SL-AUD6 · 7 |
| `AG3S/trajopt/grasp_latch.py` (놓았음 표시), `AG3S/trajopt/safe_policy.py` | SL-AUD8 |
| `AG3S/adapters/panda/profile.py` | `SOTA_AUDIT_VERSIONS` 26 개 = base 2 (동결 판, 그 `-gh156a` 판) × segment 13 (`GT15_VERSIONS = … + SOTA_VERSIONS` 줄 바로 뒤. `SOTA_VERSIONS` 에는 넣지 않음) |
| `tests/trajopt/test_sl_aud_action_qp.py` | 새 test 17 개 (action QP 의 동작 · flag · config) |
| `tests/ag3s/test_sl_aud_place_phase.py` | 새 test 12 개 (놓는 구간의 동작 · flag) |
| `tests/benchmark/test_sl_aud_versions.py` | 새 test 1 개 (판 등록) |
| `tests/trajopt/test_sl_sota11_opening_rule_reach_sticky.py`, `tests/trajopt/test_sl_gtto11_scaled_qp.py`, `tests/benchmark/test_sl_sota18_hand_sized.py` | 판 목록을 고정한 줄에 `SOTA_AUDIT_VERSIONS` 를 더함 (각 한 줄) |

branch `audit/sl-aud` 의 commit 둘로 나눴다: `282e22d9` (code + 동작 test) 와 `ac62c2bc` (판 등록 + 그 test). `origin/feature/panda-benchmark` 의 `e5f0254d` 위에 올렸다. 가져오는 방법은 시험 계획서 §3 에 있다.

판 이름은 `<base 에서 -tro 를 뗀 것><segment>-tro` 다.

| segment | 더한 flag |
|---|---|
| `-gr` | `--ee-action-qp-gate-rows` (SL-SOTA4 를 동결 base 에서 다시 재는 용도) |
| `-rf` · `-lp` · `-mr8` · `-mr12` · `-af` · `-md5` | 위 표의 flag 하나씩 |
| `-rf-lp-mr8-af` | SL-AUD1 – 4 를 모두 (`-mr8`) |
| `-ohr` · `-dlo` · `-pro` | SL-AUD6 · 7 · 8 하나씩 |
| `-ohr-dlo-pro` | SL-AUD6 – 8 을 모두 |
| `-rf-lp-mr8-af-ohr-dlo-pro` | 전부 (채택 gate 용) |

### 9.3 이 PC 에서 돌린 test

환경: Python 3.10, numpy 2.2.6 · scipy 1.15.3 · osqp 1.1.3 · casadi 3.8.2 · mujoco 3.17.0 (`AG3S_dev/.venv-audit`, repo 밖). `third_party/curobo` submodule 을 받아 `franka.yml` 을 읽었다.

| 묶음 | 결과 |
|---|---|
| 새 test 두 파일 | 18 통과 |
| action QP · 통과 판정 · 판 목록 관련 기존 test | §9.6 |

새 test 가 고정하는 것:

- 꺼짐: 결과 · row · 기록 key 가 그대로다 (`reference_floor` · `min_reach_m` · `limit_passthrough` 가 없다).
- SL-AUD1: 받아들여진 60 mm model 위반을 QP 가 되돌리지 않는다 (보정 0.06 → 0.01). 그 row 를 더 나쁘게 만드는 보정은 slack 을 낸다 (`-gr` 은 내지 않는다). 실제 linearizer 의 ESDF block 과 shape 이 맞는다.
- SL-AUD2: 실제 optimizer 에서 사유 `limit` 인 chunk 가 그대로 나가고 action QP 가 돌지 않는다. action map 이 없거나 사유가 다르면 예전과 같다.
- SL-AUD3: 약한 row (action 1 단위당 3 mm) 가 action 을 상자 끝으로 미는 것 (보정 1.0, 포화 1) 이 사라진다 (보정 0).
- SL-AUD4: 조기 종료 횟수, 낡은 여유 값.
- config · flag 조합 · 판 등록 · 동결 판의 config 가 그대로인 것.

### 9.4 offline demo — 실제 Panda action map 으로 본 보정 크기

`doc/active/handoff_SL_AUD_20261113/scripts_20261113/audit_qp_demo.py`. 합성 장면이다: grip site 에 구 하나, 손이 따라 움직이는 벽 (장애물 물체 row) 과 손 아래 탁자 (장애물 물체 row 아님). policy chunk 는 벽과 나란히 +y 로 0.6. action map 은 실제 `ee_action_map` (response model + Jacobian), 설정은 동결 판 (여유 15 mm · backoff 5 mm · priority 10 · `-qs` · `-tro`).

**벽이 여유 안쪽 1 mm (실제 여유 14 mm), 탁자는 3 mm 아래.**

| 판 | 보정 크기 | ±1 에 붙은 성분 | 남은 slack (mm) | 뺀 row | 행 4 에서 EE 이동 (mm) | 행 0 의 보정 (x · y · z) |
|---|---:|---:|---:|---:|---:|---|
| 동결 판 | 1.00 | 3 | 2.5 | 0 | 14.7 | −1.00 · +0.40 · +1.00 |
| `-gr` | 1.00 | 3 | 2.5 | 0 | 14.9 | −1.00 · +0.40 · **−1.00** |
| `-rf` | 1.00 | 2 | 2.5 | 0 | 10.8 | −1.00 · +0.40 · 0.00 |
| `-mr8` | 0.42 | 0 | 0.0 | 4 | 9.3 | −0.42 · 0.00 · +0.15 |
| `-rf-mr8` | 0.43 | 0 | 0.0 | 4 | 8.8 | −0.43 · 0.00 · 0.00 |
| `-rf-mr12` | 0.28 | 0 | 0.0 | 6 | 7.2 | −0.28 · 0.00 · 0.00 |

**벽은 같고, 손 아래에 30 mm 의 model 위반 (통과 판정이 받아들이는 것).**

| 판 | 보정 크기 | ±1 에 붙은 성분 | 남은 slack (mm) | 행 4 에서 EE 이동 (mm) | 행 0 – 2 의 z 보정 |
|---|---:|---:|---:|---:|---|
| 동결 판 | 1.00 | 9 | 31.5 | 37.9 | +1 · +1 · +1 |
| `-gr` | 1.00 | 3 | 2.5 | 14.9 | −1 · 0 · 0 |
| `-mr8` | 1.00 | 8 | 17.7 | 37.9 | +1 · +1 · +1 |
| `-rf-mr8` | 0.43 | 0 | 0.0 | 8.8 | 0 · 0 · 0 |

읽는 법:

- **동결 판은 여유 1 mm 부족에 action 세 성분을 상자 끝으로 보낸다.** 벽의 법선은 x 인데 y 와 z 도 끝까지 간다 (§3.2 의 '상자 끝으로 밀림' 이 OSQP 에서 재현됨).
- **`-gr` 은 z 를 −1 로 보낸다.** 탁자 row 를 −100 mm 까지 풀었기 때문에, 장애물 row 의 slack 을 조금이라도 줄이는 방향이면 손을 탁자 쪽으로 민다. 11-11 의 탈락과 맞는 방향이다.
- **두 flag 는 같이 있어야 한다.** `-rf` 만으로는 step 0 의 벽 row 가 x · y 를 포화시키고, `-mr8` 만으로는 받아들여진 위반을 되돌리려 z 가 포화된다. `-rf-mr8` 에서 보정이 벽의 법선 방향 하나로 줄고 slack 이 0 이 된다.
- 합성 장면이다. 실제 장면은 row 가 수백 개이고 팔 link row 가 섞인다. 크기는 closed loop 에서만 나온다.

### 9.5 SDM 에서 할 일

넣는 방법, 1 차 시험의 셋 · 기준, 돌려받을 값, 일정은 `2026-11-13_SL-AUD_test_plan_KO.md` 에 있다 (이 절의 첫 판에 있던 내용을 그 문서로 옮기고 고쳤다).

**위험.**

- SL-AUD3 은 가까운 step 의 row 를 QP 에서 뺀다. 0.1 – 0.15 초 안의 접촉은 QP 가 피하려 하지 않는다 (어차피 action 으로 3.5 – 9.6 mm 밖에 못 움직인다). CAR 이 내려갈 수 있으므로 > 50 mm 충돌 수를 기준에 넣었다.
- SL-AUD1 은 QP 가 받아들여진 model 위반을 고치지 않게 한다. 장애물 집합에서 빠진 실제 장애물 (§4.2 의 7 번) 은 QP 가 밀어내지 않는다. `-gr` 과 같은 위험이고, `-gr` 보다 작다 (더 나빠지게는 못 한다).
- SL-AUD2 는 3 episode 에서 2 call 뿐이었다. 효과가 작을 것이다.
- SL-AUD5 (상한) 는 offline 계산에서 포화를 다음 행으로 옮길 뿐이었다 (EE 이동 15.5 대 14.7 mm). 단독으로는 기대하지 않는다.

### 9.6 기존 test 의 결과

시험 계획서 (`2026-11-13_SL-AUD_test_plan_KO.md`) §9 에 있다. 요약: 변경으로 새로 실패하는 test 는 0 이고, closed loop 과 `tests/trajopt` · `tests/benchmark` 전체는 이 PC 에서 돌리지 못했다.
