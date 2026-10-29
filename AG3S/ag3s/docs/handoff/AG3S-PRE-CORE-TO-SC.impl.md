# AG3S-PRE-CORE-TO-SC — 손 끝 몸체 형상 보정 (손바닥 바깥 껍질 구) 구현 · offline 검증

> writer: ag3s-implementer · 읽는 쪽: lead, verifier, scribe
> 작성 시작 2026-10-26 17:03 KST (`date -u` 08:03 UTC + 9 h). **§3 의 offline 측정 정의는 측정 전에 이 시각에 고정했다.** 결과는 §5 (측정 뒤 추가).
> 코드는 사본 `$RUN/snapshot_sc` 에서만 고쳤다 (main checkout 코드는 건드리지 않음). patch: `$RUN/verify/AG3S-PRE-CORE-TO/SC/sc.patch` (887 줄, sha256 앞 16 `c00347b068469827`, `snapshot_e3c` 대비 코드 6 파일 + test 1 파일).
> `RUN = /mnt/dev/AG3S_VLA-storage/results/development/20261025T191319+0900_35b909cf096a_cc215238f581`

## 0. 요약 (평이하게)

- **무엇을 했나:** TrajOpt 의 로봇 구 모델에 손바닥 **바깥 껍질 구** (`--palm-shell`, 기본 off) 를 넣었다. 손바닥 겉면 중 손가락 사이 바닥면 (사과가 들어앉는 면) 만 빼고 덮고, 그 바닥면 아래로는 한 점도 나오지 않는다. 꺼짐이면 모델 · 기록 키 · 궤적이 이전과 비트로 같다 (test).
- **막대 접촉 (ⓐ):** 4 건 모두 접촉점이 새 구 표면 0.3 mm 안에 들고 (지금 모델: 17–28 mm 밖), TrajOpt 가 계획 자세에서 본 여유가 +7…+15 mm → **−5…−13 mm** 가 된다. 접촉 2–16 step 전부터 침범으로 보인다 → 이 접촉은 TrajOpt 가 미리 볼 수 있었다.
- **그러나 쥐기 (ⓑ · ⓒ):** 접촉 자리는 손바닥 −y 쪽 바닥 모서리이고, **1345 오렌지 접촉과 같은 자리**다 (§1). 그리고 쥐는 순간에 그 모서리 · 옆면은 사과 (target-free 층에 남은 사과 조각) · 이웃 과일에 10 mm 여유 안으로 붙는다. 그래서 껍질은 성공한 쥐기 20 개 중 12 개에서 닫기 전 1 s 동안 새 위반 행을 만들고 (최대 −12.9 mm), 다시 풀면 팔을 0.5° 넘게 바꾸는 요청이 17 / 59 (palm cover 18 / 59) 다. 1345 에서는 palm cover 와 같은 오렌지 행이 같은 깊이로 생기고 수정량도 같은 크기다.
- **판정 (§3 에 미리 고정한 문장):** ⓒ "palm cover 와 같은 실패 위험 — 형상만으로는 피하지 못함". **형상 보정만으로는 막대 접촉을 보면서 쥐기를 지킬 수 없다.** 다음 판단 (여유 규칙 · target-free 층 · 허용 규칙 중 무엇을 바꿀지, 또는 알려진 한계로 둘지) 은 사용자 몫이다 — §6 에 숫자로 정리했다.

## 1. 먼저 알게 된 것 — 막대 접촉과 1345 오렌지 접촉은 손의 같은 자리다 (설계 전 측정)

접촉점을 왼손 손바닥 URDF link `ee_left` frame 으로 옮겨 보았다 (`SC/sc_geom.py`, `SC/sc_geom_1345.py`; 원시 `SC/geom/*.json`).
link frame 축: `u` = x (손가락 닫히는 축), 손끝 방향 `tip` = −z, 손가락 통로 반폭 47.16 mm (열린 손가락 안쪽 면), 손바닥 mesh 바닥면 z = −73.69 mm.

| 접촉 | 접촉점 (x, y, z) mm, link frame | 그 자리 mesh 법선 | 통로 안? | 지금 T43 R 정의의 "파지 면"? |
|---|---|---|---|---|
| E3 case3 L3 s13453, t 88, 막대 | (21.97, −27.89, −73.23) | (0, −0.96, −0.28) | 예 | 예 |
| E0 case3 L3 s13453, t 366, 막대 | (20.86, −27.89, −73.23) | (0, −0.96, −0.28) | 예 | 예 |
| E0 case3 L3 s23453, t 324, 막대 | (30.20, −27.89, −73.23) | (0, −0.96, −0.28) | 예 | 예 |
| E0 case3 L3 s33453, t 103, 막대 | (17.53, −27.89, −73.23) | (0, −0.96, −0.28) | 예 | 예 |
| **E4 1345 오렌지 (palm cover 가 쥐기를 잃은 장면), t 101–127** | **(38.6–40.0, −27.9…−28.2, −73.2…−70.0)** | (0, −0.96, −0.28) · (0, −1, 0) | 예 | 예 |
| (참고) T44d 손바닥–이웃 과일 접촉 35 점 (오른손, `test_t45a10_palm_cover.py`) | x −16…−42, y −27.8…−28.2, z −73.2 · −52 | | | |

- 다섯 접촉 모두 **손가락 통로 안, −y 쪽 바닥 모서리** (바닥면과 −y 옆면이 만나는 경사면, 바닥면에서 0.5 mm 위) 에 있다. 막대와 오렌지의 차이는 x 로 10–20 mm 뿐이다.
- 이 자리는 손끝 쪽을 보는 면이 아니라 **옆 (−y) 을 보는 모서리**다 (법선 · tip = 0.28). 그런데 T43 R (`gripper_cover.link_surface`) 은 통로 안에서 지금 손바닥 구 끝 (z_lim 58.4 mm) 보다 손끝 쪽인 점을 모두 파지 면으로 친다 → 덮지 않는다.
- **결론: 막대 접촉 자리를 덮는 어떤 형상 보정도 1345 의 오렌지 접촉 자리를 같이 덮는다.** 형상만으로 둘을 가를 수 없다 (x 로 가르는 것은 결과를 보고 맞춘 설계라 쓰지 않는다). 그래서 ⓒ 는 "피했다" 가 아니라 "palm cover 와 비교해 무엇이 같고 무엇이 줄었나" 로 잰다 (§3 ⓒ).

## 2. 설계 — 손바닥 바깥 껍질 구 (`--palm-shell`, 기본 off)

palm cover 가 한 일을 둘로 나눴다: (a) 바깥을 보는 면 (옆면 · 윗면 · 모서리) 을 덮는다 — 막대 · 이웃 물체가 닿는 곳, (b) 손가락 사이 바닥면 (손끝 쪽을 보는 면) 을 덮고 그 아래로 5 mm 나온다 — 사과가 들어앉는 곳. **껍질은 (a) 만 한다.** (b) 는 지금 그대로 (손바닥 축선 구 23 개가 남는다).

| 항목 | 내용 |
|---|---|
| 덮을 대상 | 손바닥 collision 겉면 표본 (1 mm) 중 **파지 면이 아닌 것**, 그리고 강체 이웃 (손목 · FT 센서) 에 묻힌 면이 아닌 것. 파지 면 = 통로 안 (`|u·p| < 47.16 mm`) 이고 법선이 손끝 쪽 (`n·tip > 0.5`) 인 바닥면. 위 표의 접촉 자리 (`n·tip = 0.28`) 는 대상이다 |
| 구 계산 | palm cover 와 같다 (`finger_body_spheres`: 후보 = mesh 안 2 mm 격자점, 반지름 = 깊이 + overflow 5 mm, greedy 로 대상 전부 담음) |
| 더한 제약 | **파지 공간 keep-out**: `{|u·p| ≤ 47.16 mm, tip·p ≥ 73.69 mm}` (통로 안에서 바닥면보다 손끝 쪽) 에 구가 한 점도 들어가지 않는다 (지지 평면 거리 — 표본이 아닌 정확한 보장, 실측 침범 0.0 mm). 옆 (±y) · 위로는 overflow 까지 나온다 |
| 모델에 넣는 방식 | 손바닥 축선 구를 **대신하지 않고 맨 뒤에 붙인다** (role `palm_shell`, 반지름 그대로). 꺼진 모델 = 켠 모델의 정확한 앞부분 (구 이름 · 반지름 · 중심 비트 동일). link 이름 `ee_<side>` 그대로 → 접촉 권한 · target-free 층 조회가 palm cover 와 같이 걸린다 (사과는 이 구들에게 장애물이 아니다) |
| 구 수 (서버 구성 `--links gripper --finger-cover both`) | 270 → **606** (손바닥마다 +168, r 2.0–29.0 mm, 대상 24,862 표본 중 24 개 미달 — 최대 틈 0.64 mm, 넘침 실측 5.06 mm). `left` 만이면 270 → 438. 참고: palm cover 466 |
| 같이 쓰지 않는 것 | `--palm-cover` (같은 손바닥을 통째로 바꾼다) · `--gripper-cover` (껍질 위에 덮개를 또 만든다) → 서버 시작 시 거절, 함수는 ValueError |

palm cover 실패 분석에서 고른 이유: palm cover 의 최악 행은 손바닥 구 vs **hand layer 의 오렌지**였다 (AG3S_TSDF_DYNAMIC_LOG §15.7 (iii)). hand layer 에서 사과는 빠져 있어 사과 자체는 막지 않았다. 그러므로 쥐기를 막는 것은 "손바닥 구가 사과에 닿는다" 가 아니라 "target 아닌 표면 (이웃 · 탁자 · target-free 층에 남은 사과 조각) 이 손바닥 구 + 10 mm 여유 안에 든다" 다. 껍질은 (i) 바닥면 아래로 나오지 않아 탁자 · 남은 사과 조각과의 행을 지금처럼 두고, (ii) 옆으로는 palm cover 와 같이 나온다 → 1345 오렌지 행은 남을 것이다 (§1). ⓒ 가 이것을 숫자로 본다.

## 3. offline 측정 정의 (측정 전 고정, 2026-10-26 17:03 KST) — CPU 만, worker ≤ 2

모델: old = 기록 서버 명령줄 그대로 (`ta_offline.server_args` → `build_robot`), new = 같은 명령줄 + `--palm-shell on`. 거리장 = 그 step 을 실행한 요청의 기록 (`fixed_input_ablation.build_field`). 손가락 경로 = 서버와 같은 `FingerJointMap.path`.

**ⓐ 접촉 4 건 (E3 case3 L3 s13453, E0 case3 L3 s13453 · s23453 · s33453), step t ∈ [t_c − 16, t_c].** DX2 의 `dx2_contact.py` 와 같은 재구성.
- a1 접촉점 덮임: DX2 와 같은 접촉점 (실제 자세에서 `mj_geomDistance(EE_BODY_L, 막대)` 의 로봇 쪽 점) 에서 실제 자세 구 표면까지 `min |p − c| − r` (≤ 0 = 구 안), old · new, t_c 값.
- a2 계획 여유: 계획 자세 (`refined_chunk` 그 행) 에서 구 중심의 기록 거리장 값 − r 의 최소 (TrajOpt 가 읽는 층: 권한 link = target-free 층; DX2 `F_plan_free` 와 같은 정의), old · new, t_c 값과 창 전체. 판정 문턱 둘을 함께 적는다: `< 0` (TrajOpt 가 침범으로 본다), `< 10 mm` (obstacle margin — TrajOpt 의 위반 행이 된다). new 가 처음 `< 10 mm` 가 되는 step (t_c 몇 step 전인가).
- a3 참고: 같은 계획 자세에서 new 구와 실제 막대 geom 의 정확한 거리 − r (DX2 `G_plan` 과 같은 정의).
- 덧붙여: 접촉점과 target (사과) 중심의 거리 (IFG-NBC 허용 반경 120 mm 와 비교 — 이웃 허용을 손바닥으로 넓히면 막대도 허용되는지 판단용).

**ⓑ 기존 성공 쥐기 — 새 구가 쥐기를 막는가.** 대상 (결과 전 고정): E3 S-REG 의 금지 접촉 없는 성공 14 run (`E3/closed_reg/agg/e3_closed_loop_reg.json` 의 `success_without_forbidden`) + E3 가림 성공 6 run (case1 L1 · L3 × 3 seed, `E3/closed_occ/e3_runs.json`). 창 = 실행 step 이 `[t_close − 15, t_close]` (1 s, 15 Hz) 와 겹치는 요청 (t_close: 회귀 = analysis `grasp_t`, 가림 = `e3_runs.json` `t_close`).
- b1 (주): 그 요청의 계획 궤적 (`refined_chunk` plan horizon 16 행) 을 `ta_offline.Run.inputs` 가 재구성한 장면 (target-free 마스크 · obstacle margin · 목적지 · support plane 행 포함) 에서 old / new 모델로 TrajOpt 의 행 값 (`CollisionLinearizer._esdf_clearance` = d − r − margin, plane 행 = n·c − offset − r) 을 잰다. **새 위반 행 = new 모델에만 있는 구 (껍질 구) 중 값 < 0 인 행.** 보고: 새 위반 행이 1 개 이상인 요청 수 / 창 요청 수, 가장 깊은 값 (mm), 층 (거리장 / 탁자 평면) 별 수, 같은 요청에서 old 모델에 이미 위반 행이 있었는지.
- b2 (보조, 쥐기를 굽혔을까): 같은 창 요청을 현재 serving 구성 기록 (회귀 = E4 index 의 같은 14 장면, 가림 = E0 round 4 의 같은 6 run — E3 기록은 GPU seed 구성이라 CPU 에서 다시 풀 수 없다) 에서 `ta_offline` 로 다시 풀어 (variant `base` · `PS`, `--ref-variant base`) `PS` 대 `base` 의 행 0–7 왼팔 최대 차 (deg) 를 잰다. 기준값으로 palm cover (`PC`) 도 같이 푼다.

> **고침 (2026-10-26 17:37 KST, ⓑ 결과를 보기 전 · 창 정의만):** 회귀 run 의 t_close 를 analysis `grasp_t` 로 정했으나, 첫 계산에서 창 요청의 grasp 상태가 `held` (이미 쥔 뒤) 로 나와 `grasp_t` 가 닫는 시각이 아니라 쥐기가 성립한 시각임을 알았다 (닫기보다 13–33 step 늦다). 과제 정의 "닫기 1 s 전 ~ 닫기" 에 맞게 **가림 장면과 같은 agg4 정의** (OCC `p3/agg4.py:57`: 쥐는 손 측정 gripper 열림이 처음으로 초기값의 0.95 배 아래가 된 step, 손은 analysis `used_arm`) 로 바꿨다. 회귀 14 중 8 run 이 **오른손**으로 쥔다 (`--palm-shell on` 은 두 손 모두). 첫 계산 (grasp_t 창) 의 원시는 지우지 않고 `SC/b1/graspt_window_attempt/` · `SC/b2/graspt_window_attempt/` 에 두었다 (b2 는 4 run 에서 멈춤). 가림 6 run 의 창은 그대로다.

**ⓒ palm cover 실패 사례 (1345 · seed 13453, E4 B_ROI_FPR 기록 — palm cover closed-loop 의 기준 run 과 같은 장면 · seed).** 창 = 요청 11–19 (palm cover 실패 분석 표와 같은 요청).
- c1: b1 과 같은 행 값을 old · `PC` (palm cover) · `PS` (껍질) 모델로 잰다. 요청마다 손바닥 구 위반 행 수, 가장 깊은 값, 그 행의 가장 가까운 거리장 표면점 (`p − d∇d`) 이 GT 의 오렌지 · 사과 · 탁자 중 어디에 가까운가 (8 mm 이내 귀속, palm cover 분석과 같은 문턱).
- c2: 같은 요청을 `ta_offline` 로 `base` · `PC` · `PS` 로 다시 풀어 행 0–7 왼팔 최대 차 (deg, vs base). palm cover closed-loop 에서 처음 접근 수정량은 0.060 rad (3.4°), 기준 0.006 rad.
- 판정 문장 (결과 전 고정): `PS` 의 c1 위반 행이 오렌지에 귀속되고 c2 수정량이 `PC` 와 같은 크기 (±30 %) 이면 "palm cover 와 같은 실패 위험 — 형상만으로는 피하지 못함", `PS` 수정량이 `PC` 의 1/3 이하이면 "위험이 줄어듦 (closed-loop 확인 필요)", 그 사이는 "일부 줄어듦".

## 4. 바뀐 파일 · 단위 검증 · verifier 가 알아야 할 것

### 4.1 바뀐 파일 (줄 = `snapshot_sc` 기준; `sc.patch` 는 `snapshot_e3c` 대비, main checkout 에 `git apply --check` 통과 — serve_safe 는 E3-b first-pass 가 checkout 에 아직 없어 hunk 3 개가 5–14 줄 밀려 붙는다)

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/ag3s/robot_models/urdf_sphere_chain.py:240-249` | `PALM_SHELL_ROLE = "palm_shell"` 추가, `EXACT_SPHERE_ROLES` 에 포함 | 껍질 구는 내접 · 깊이 + overflow 로 이미 정한 반지름이라 `capsule_radius_scale` · 간격 inflation 을 받으면 안 된다 |
| `AG3S/ag3s/experiments/sources/gripper_cover.py:50, 323-326` | `PALM_SHELL_ROLE` import, `grasp_limits` 가 껍질 구를 T43 R 덮개처럼 뺀다 | 손가락 통로 경계가 껍질 때문에 바뀌지 않게 (off 에는 영향 없음) |
| `AG3S/ag3s/experiments/sources/gripper_cover.py:1024-1118` | `PALM_SHELL_OVERFLOW` · `PALM_SHELL_FACE_COS`, `palm_shell_parts` (대상 마스크 + 파지 공간 keep-out), `palm_shell_spheres` (`finger_body_spheres(target_mask=, keep_out=)`), `palm_shell_capsules` (`_body_capsules`, 캐시 키에 통로 포함) | 설계 §2. 새 알고리즘 없이 palm cover · 손목 cover 의 기존 계산을 재사용 |
| `AG3S/ag3s/experiments/reports/grounding_report.py:154-155, 235, 290-301, 421-455` | `PALM_SHELL_HANDS`, `build_constraint_robot_model(palm_shell=None)`: palm_cover · gripper_cover 와 같이 주면 ValueError, 켜면 손바닥 link 에 껍질 구를 `gap` 맨 뒤에 붙여 모델을 다시 짓고 `model.palm_shell_report` | 꺼짐 = 인자 없던 호출과 글자 그대로 같음; 켜짐 = 꺼진 모델이 정확한 앞부분 |
| `AG3S/trajopt/serve_safe.py:685-731` | `PALM_SHELL_MODES`, `announce_palm_shell`, `palm_shell_options` | palm cover 와 같은 규약 (off → None) |
| `AG3S/trajopt/serve_safe.py:1145-1146, 1184, 1256-1258, 1343` | `ag3s_from_args` → `build_ag3s(palm_shell=)` (off 면 키 없음) → `build_constraint_robot_model` · 시작 WARNING | 배선 |
| `AG3S/trajopt/serve_safe.py:2113-2120, 2865-2870, 3578-3581` | CLI `--palm-shell {off,on,left,right}` · `--palm-shell-overflow MM`; `--no-safe` · `--no-perception` · `--palm-cover on` · `--gripper-cover` 와 같이 쓰면 시작 거절; client metadata `palm_shell` (켰을 때만 키) | 켜짐을 믿었는데 효과가 없는 상태를 막는다 |
| `AG3S/trajopt/experiments/ta_offline.py:154-155, 192-193, 265-267` | variant `PS` (`--palm-shell on`), `build_robot` 이 명령줄의 `--palm-shell` 을 읽는다 | offline 비교 (b2 · c2) |
| `AG3S/trajopt/curobo_v2/export_inputs.py:60` | `--palm-shell` 을 재구성하지 못하는 flag 목록에 추가 | cuRobo 입력 export 가 다른 모델로 조용히 export 하지 않게 |
| `tests/trajopt/test_sc_palm_shell.py` (새, 26 test) | §4.2 | |

### 4.2 단위 검증

```bash
# cwd = $RUN/snapshot_sc, PYTHONPATH = SC/env.sh (snapshot_sc 먼저; `AG3S.trajopt.sqp.__file__` 이 snapshot_sc 를 가리킴 확인)
.venv-openpi-live/bin/python -m pytest -q tests/trajopt/test_sc_palm_shell.py        # 26 passed
.venv-ag3s/bin/python        -m pytest -q tests/trajopt/test_sc_palm_shell.py        # 26 passed (CPU 회귀 venv, numpy 2.4.6)
.venv-ag3s/bin/python        -m pytest -q tests/trajopt/test_t45a10_palm_cover.py    # (위와 같은 실행) 28 passed
.venv-openpi-live/bin/python -m pytest -q tests/trajopt/test_t45a10_palm_cover.py test_t43r* test_t43fc* test_t43fr* test_t46f* test_t63_curobo_v2_spec.py   # 199 passed
# tests/trajopt + tests/ag3s 전체: 파일마다 별도 프로세스 (SC/tests/run_files.sh, worker 2)
#   208 파일 · 3,870 passed · 40 skipped · 4 failed — 4 failed 는 SC 전 코드 snapshot_e3c 에서도 같은 venv 로 똑같이 실패 (SC/tests/pre_change_e3c_same_failures.txt)
```

- 4 failed (내 변경과 무관, 사전 존재): `test_t44_target_exclusion_owned::test_the_record_writes_the_trace_keys`, `test_e1_grasp_goal::test_off_and_toa_are_bit_identical_to_head`, `test_e2_multi_seed::test_off_is_bit_identical_to_head` · `test_not_triggered_when_the_standard_solve_resolves` (E1 · E2 의 golden SHA 는 `.venv-ag3s` 에서 잰 값이라 `.venv-openpi-live` 에서는 다르다). 원시 `SC/tests/per_file/*.txt`, 합계 `SC/tests/per_file_summary.json`.
- 처음에는 한 프로세스로 전체를 돌렸는데 40 % 지점에서 RSS 43 GB 로 pod cgroup OOM-kill 됐다 (`dmesg`: `Killed process 763693 (python) anon-rss:43676764kB`). 그래서 파일마다 프로세스를 나눴다. 한 프로세스 전체 실행의 메모리 누적은 이번 변경 전에도 있는지 확인하지 않았다.

test 가 고정하는 것: (1) off — `palm_shell=None/False/{}` × finger cover 유무가 인자 없는 모델과 이름 · role · 반지름 · FK 비트 동일, CLI 기본 off · `--palm-shell off` 면 `build_ag3s` · `ag3s_from_args` · `ta_offline.build_robot` 에 키 없음, **기록 chunk 두 개 (E4 1345 seq 12 · 13) 의 TrajOpt 다시 풀기 SHA 가 SC 전 코드 `snapshot_e3c` 와 같다** (`dcc931c0ed740d85` · `e6f6986f5730bfc8`, 두 venv 모두; `SC/golden_solve_sha.py`). (2) on — 앞 270 구 (이름 · 반지름 · role · FK) 와 기록 chunk 의 TrajOpt 거리장 행 값 앞 270 열이 off 와 비트 동일, 270 → 606 (`left` 438), 다시 풀기는 껍질 행이 없는 seq 12 는 off 와 같고 seq 13 만 바뀐다. (3) 형상 — 대상 = 통로 안 손끝 쪽 바닥면 · 숨은 면만 뺀 것, 껍질 구가 파지 공간에 들어가지 않음 (평면 거리 ≥ 반지름, 정확), 대상 미달 표본 24 개 · 최대 0.64 mm, 넘침 실측 ≤ 5.2 mm, DX2 막대 접촉점 4 개 · 1345 오렌지 접촉점 27 개가 0.7 mm 안 (지금 모델에서는 전부 15 mm 넘게 밖), 왼손 · 오른손 같은 구, max opening · 손가락 통로 경계 불변. (4) 권한 — 쥐는 손의 껍질 구 168 개 전부 target-free 층 마스크, 반대 손 0. (5) 거절 — 함수 ValueError 5 경우, 서버 시작 거절 4 경우.

**사전 환경 문제 (내 변경과 무관):** `tests/trajopt/test_t41b_rest_start.py` · `test_t41c_hold_measured.py` · `test_t43q_replay.py` 는 snapshot 에 `legacy/benchmark` fixture 가 없어 collection 단계에서 `FileNotFoundError` (`common/workspace.py:27`) 로 멈춘다 — 전체 실행에서 `--ignore` 로 뺐다.

### 4.3 verifier 가 알아야 할 것

- 새 flag: `--palm-shell {off,on,left,right}` (기본 off), `--palm-shell-overflow MM` (기본 5.0). 기본값 변경 없음. client metadata · summary 의 키는 켰을 때만 (`palm_shell`), 꺼짐이면 기록 키가 이전과 같다.
- 켜면 손바닥 구가 손바닥마다 +168 (전체 606; palm cover 466 보다 많다). TrajOpt 거리장 질의 행이 그만큼 는다 — **시간은 재지 않았다** (독점 lock 측정 필요).
- 껍질 구는 모델 **맨 뒤**에 붙는다 (palm cover 는 손바닥 자리를 대신한다). npz 의 `sphere_link_names` 로 열을 고를 것.
- 재생산할 산출물 없음. 옛 기록과 호환 깨짐 없음.
- `ta_offline` 의 장면 재구성에는 IFG-NBC (손가락 이웃 접촉 허용) 가 없다 — 손바닥 · 껍질 구에는 원래 걸리지 않으므로 이 비교에는 영향이 없다.

## 5. offline 결과 (원시: `$RUN/verify/AG3S-PRE-CORE-TO/SC/`)

### 5.1 ⓐ 접촉 4 건 — 새 모델은 접촉을 덮고, TrajOpt 가 미리 본다

원시 `SC/a/*.json` (step 별 17 행), 요약 `SC/a/summary_a.json`, 스크립트 `SC/sc_a_contacts.py` · `SC/a/run_a.sh`. old 값은 DX2 와 같다 (F_plan 15.04 · 13.57 · 7.42 · 12.91 mm).

| 접촉 run | t_c (grasp 상태) | 실제 mesh 거리 | a1 접촉점 → 구 표면 old → **new** (mm, ≤ 0 = 안) | a2 계획 여유 F_plan old → **new** (mm) | new 가장 얕은 행 | a3 계획 자세 정확 거리 old → new 껍질 (mm) | 창 17 step 중 new 껍질 F_plan < 10 mm 인 step · 처음 < 10 mm / < 0 이 된 때 | 접촉점–사과 중심 |
|---|---|---|---|---|---|---|---|---|
| E3 case3 L3 s13453 | 88 (latched) | −0.45 | 23.71 → **0.30** | 15.04 → **−5.16** | `ee_left` 껍질 | 16.65 → 0.47 | 5 · t_c−4 / t_c−2 | 115 mm |
| E0 case3 L3 s13453 | 366 (latched) | −1.05 | 24.69 → **0.14** | 13.57 → **−10.98** | `ee_left` 껍질 | 16.68 → −12.79 | 14 · t_c−16 / t_c−16 | 120 mm |
| E0 case3 L3 s23453 | 324 (latched) | −0.59 | 16.92 → **0.23** | 7.42 → **−8.00** | `ee_left` 껍질 | 7.14 → −5.27 | 17 · t_c−16 / t_c−16 | 56 mm |
| E0 case3 L3 s33453 | 103 (latched) | −1.50 | 27.62 → **−0.35** | 12.91 → **−12.53** | `ee_left` 껍질 | 17.55 → −13.17 | 7 · t_c−16 / t_c−2 | 118 mm |

- 접촉점은 4 건 모두 껍질 구 표면 0.30 mm 안 (표본 1 mm 간격 수준; 창 안 접촉 근처 step 최대 0.38 mm).
- TrajOpt 가 계획 자세에서 읽는 여유가 4 건 모두 음수 (−5…−13 mm) 가 된다 → obstacle margin 10 mm 기준으로도 0 기준으로도 위반 행. 처음 보이는 때는 접촉 2–16 step 전 (E3 s13453 은 4 step 전부터 < 10 mm, 2 step 전부터 < 0).
- 접촉점–사과 중심 56–120 mm: IFG-NBC 의 허용 반경 120 mm 안이다 (§6 의 F2 판단 근거).

### 5.2 ⓑ 기존 성공 쥐기 — 새 구가 쥐기를 막는가

**b1 (주, §3 정의):** 원시 `SC/b1/b1_rows_gt.json` (= `b1_rows.json` 과 같은 행 값 + GT 귀속), 요약 `SC/b1/summary_b1_final.json`, 스크립트 `SC/sc_rows.py` · `SC/sc_common.py` · `SC/summarize_final.py`, job `SC/b1/jobs_gt.json`. 창 = 닫기 1 s 전 ~ 닫기 (§3 고침 뒤 정의), 요청 59 개. E3 가림 서버 폴더에는 `server_proc.txt` 가 없어 구 모델 · 충돌 설정은 같은 장면 E0 serving 명령줄 (`E1/occ_index/<run>/server_proc.txt`) 로 지었다 (E3 = E2 + GPU seed 라 충돌 · 구 flag 는 같다).

| 구분 | run | 창 요청 | 새 위반 행이 있는 요청 | 새 위반 행 수 | 가장 깊은 값 | old 모델도 위반 행이 있던 요청 |
|---|---|---|---|---|---|---|
| 회귀 (E3 S-REG 금지 접촉 없는 성공) | 14 | 41 | **16** (8 run) | 1,221 | −12.9 mm | 10 (새 위반 요청 16 중 12 는 old 위반 없음) |
| 가림 (E3 case1 L1 · L3 성공) | 6 | 18 | **6** (4 run) | 442 | −11.25 mm | 5 (새 위반 요청 6 중 2 는 old 위반 없음) |
| 합 | 20 | 59 | **22 (12 run)** | 1,663 | −12.9 mm | 15 (새 위반 요청 22 중 14 는 old 에 위반 행이 없던 요청) |

- 탁자 평면 행: 이 구성의 장면에는 support plane 행이 없다 (탁자는 거리장 안) → 평면 위반 0.
- run 별 (요청 수 · 새 위반 요청 · 최저 mm): ep56 3·0·+6.5, ep119 3·3·−12.6, ep121 3·1·−3.1, ep126 3·3·−4.0, ep250 3·1·−11.2, ep309 3·3·−3.7, ep320 3·0·+42.0, ep325 3·1·−3.9, ep572 3·0·+9.2, ep753 3·0·+0.5, ep901 2·0·+37.7, ep1596 3·0·+10.7, ep1668 3·3·−12.9, ep1837 3·1·−3.6, case1 L1 s13453 3·2·−11.25, s23453 3·0·+2.4, s33453 3·1·−10.4, case1 L3 s13453 3·0·+5.7, s23453 3·1·−8.8, s33453 3·2·−5.3.
- **(측정 뒤 덧붙인 진단 — 고정 지표 아님)** 새 위반 행의 가장 가까운 거리장 표면점을 GT 로 귀속: 8 mm 안 — orange 328 · pear 71 · apple 41 · 없음 1,223; 25 mm 안 — **apple 868 · pear 445 · orange 350**. 즉 행의 절반은 target-free 층에 남은 **사과 (쥐는 대상) 주변 표면**, 나머지는 이웃 과일이다. 손바닥 두께 (y 66 mm) 가 사과 지름보다 작아, 쥐는 순간 사과 옆면이 손바닥 ±y 옆면 아래쪽 바깥으로 나온다 — 껍질이 덮는 바로 그 자리다.

**b2 (보조, 다시 풀기):** 원시 `SC/b2/*.json`, 요약 `SC/b2/summary_b2.json`, 스크립트 `SC/sc_ta.py` · `SC/run_b2c2.sh`. 현재 serving 기록 (E4 회귀 14 + E0 가림 6, 같은 장면 · 같은 창 정의), variant `base` · `PS` · `PC`.

| | 창 요청 | 팔 변화 > 0.1° | > 0.5° | 최대 | 중앙값 | 손가락–target 거리가 base 보다 5 mm 넘게 멀어진 요청 (최대) |
|---|---|---|---|---|---|---|
| **PS (껍질)** | 59 | 33 | **17** | 4.19° | 0.13° | 17 (21.6 mm) |
| PC (palm cover, 기준) | 59 | 37 | 18 | 4.30° | 0.19° | 17 (25.6 mm) |

- run 별 최대 (PS · PC, °): ep121 2.62 · 4.14, ep250 4.19 · 4.30, ep309 3.72 · 1.35, case1 L1 s33453 3.21 · 4.16, case1 L3 s23453 1.49 · 3.95, s33453 1.57 · 3.96 — 껍질이 palm cover 보다 작은 run 이 있지만 쥐기 근처에서 팔을 바꾸는 요청 수는 거의 같다.

### 5.3 ⓒ palm cover 실패 사례 (1345 · E4 기록, 요청 11–19)

원시 `SC/c1/c1_rows.json` · `SC/c2/c2_1345.json`, 요약 `SC/c1/summary_c1_final.json` · `SC/c2/summary_c2.json`.

| 요청 (t0, 상태) | old 손바닥 최저 (mm) | PC 새 위반 행 · 최저 mm · 귀속 (25 mm) | PS 새 위반 행 · 최저 mm · 귀속 (25 mm) | c2 다시 풀기 팔 변화 PS · PC (°) |
|---|---|---|---|---|
| 11 (88, latched) | +6.1 | 81 · −19.0 · orange 79 | 143 · −18.1 · orange 140 | 0.33 · 0.00 |
| 12 (96, latched) | +4.7 | 168 · −20.7 · orange 152 | 275 · −19.2 · orange 267 | 1.31 · 1.25 |
| 13 (104, latched) | +7.7 | 170 · −17.0 · orange 164 | 232 · −16.3 · orange 226 | 1.61 · 1.21 |
| 14 (112, latched) | +10.8 | 75 · −15.2 · orange 74 | 119 · −14.9 · orange 119 | 0.98 · 0.96 |
| 15 (120, latched) | +10.7 | 125 · −15.3 · orange 120 | 210 · −14.9 · orange 210 | 1.12 · 1.14 |
| 16 (128, closing) | +12.7 | 43 · −13.6 · orange 43 | 78 · −13.0 · orange 78 | 1.05 · 1.10 |
| 17 (136, closing) | +15.7 | 37 · −4.4 · orange 37 | 24 · −2.9 · orange 24 | 0.72 · 1.12 |
| 18–19 (held) | +23.5 · +41.3 | 0 | 0 | ≤ 0.24 |

- 껍질의 위반 행은 palm cover 와 **같은 자리 · 같은 깊이** (최저 −19.2 대 −20.7 mm) 이고 거의 전부 오렌지에 귀속된다 (요청 11–17 의 PS 1,081 행 중 25 mm 기준 오렌지 1,064 · 사과 17; 8 mm 기준으로도 오렌지 다수). 가장 깊은 행의 구는 손바닥 frame (42, −17, −58) mm — −y 옆면 아래쪽.
- c2: 창 최대 팔 변화 PS 1.61° 대 PC 1.25° (비 1.29), > 0.5° 인 요청 6 대 6, 손가락이 사과에서 멀어지는 양 최대 23.0 대 22.0 mm.
- **§3 의 고정 판정 문장: "palm cover 와 같은 실패 위험 — 형상만으로는 피하지 못함"** (위반 행이 오렌지에 귀속 ∧ 수정량이 PC 의 ±30 % 안).

## 6. 해석과 다음 판단에 쓸 숫자 (사용자 판정 대상)

- **사실:** 막대 접촉 4 건 · 1345 오렌지 접촉 · T44d 손바닥–과일 접촉 35 점이 모두 손바닥 −y 쪽 바닥 모서리 (link frame y ≈ −28, z −52…−73 mm) 에 있다. 이 자리는 쥐는 순간 사과 옆면과 이웃 과일이 가장 가까이 오는 자리이기도 하다 (b1 귀속). 그래서 이 자리를 덮는 형상은 (껍질이든 palm cover 든) 막대를 보는 만큼 쥐기 근처에서 팔을 바꾼다.
- **해석 [I]:** 원인 층은 형상만이 아니라 "정확해진 손바닥 형상 + obstacle margin 10 mm + target-free 층의 사과 잔여" 의 결합이다. 형상 (AG3S robot model) 은 이번 껍질로 고칠 수 있음을 보였다 (ⓐ). 쥐기 쪽 행의 절반은 target-free 층에 남은 사과 주변 표면 (LTE) 이다.
- **측정 뒤 덧붙인 what-if (고정 지표 아님, 같은 행 값에서 계산; 구현하지 않음):** 껍질 행의 여유만 10 → 5 / 0 mm 로 줄인다면 —
  - ⓑ b1 새 위반 요청 22 → **12 / 8** (run 12 → 7 / 5).
  - ⓒ 1345 요청 11–16 의 PS 위반 행 1,057 → 613 / 277 — 0 mm 에서도 남는다 (VLA 경로가 실제로 오렌지에 닿는다, mesh −0.57 mm).
  - ⓐ 막대 4 건은 0 mm 여유에서도 접촉 step 에 위반 (F_plan −5…−13 mm).
  - 여유 규칙은 TrajOpt (CCR) 층의 결정이라 task 의 구성 요소 범위를 벗어난다 — lead · 사용자 판정 없이 구현하지 않았다.
- **IFG-NBC 식 이웃 허용을 손바닥으로 넓히는 것 (F2) 은 막대와 오렌지를 가르지 못한다:** 막대 접촉점–사과 중심 56–120 mm, 막대 쪽 껍질 위반 구 중심–사과 34–155 mm 로 허용 반경 120 mm 안에 든다 (1345 의 오렌지 행도 전부 120 mm 안).
- **사용자에게 물을 선택지 (권하지 않는다):** (1) 알려진 한계로 둔다 (palm cover 판정과 같은 결론), (2) 껍질 + 껍질 행 여유 축소를 TrajOpt 에 새 flag 로 넣고 회귀 18 · 가림 18 closed-loop 로 확인 (위 what-if 로는 쥐기 근처 위반이 남는다), (3) target-free 층의 사과 잔여를 먼저 줄인다 (LTE, b1 행의 절반), (4) 다른 방법.

## 7. 프로세스 · 캐시

- 내 프로세스는 모두 끝냈다 (CPU 만, 동시 worker ≤ 2, GPU · 서버 · closed-loop 없음). storage run 안에 바깥을 가리키는 link 를 만들지 않았다 (`find $RUN/snapshot_sc -type l` = snapshot_e3c 와 같은 상대 link 2 개뿐). `E1/occ_index/make_links.sh` 를 쓰지 않았다 (server_proc.txt 를 직접 읽음). NFS 가 멈춘 옛 폴더는 열지 않았다.
- `python3 /mnt/dev/AG3S_VLA-development/scripts/setup/reclaim_cache.py $RUN $RUN/snapshot_sc` → `reclaim: ws 22.0->22.0 GiB | cache 1.0->0.0 GiB, 94562 files, 2 paths`

## 내가 기대하는 결과 (verifier 가 다시 잴 때)

- `--palm-shell off` (또는 flag 없음) 서버 · replay 는 이전과 비트 동일 (모델 · 기록 키 · refined chunk).
- `--palm-shell on` closed-loop 을 돌린다면: case3 L3 의 손–막대 접촉은 줄거나 사라지고, 회귀 · 가림 성공 쥐기 일부와 1345 는 palm cover 처럼 쥐기를 잃을 것 (§5.2 · §5.3).
