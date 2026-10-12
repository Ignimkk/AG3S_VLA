# SL5a — V11AB online server의 Panda 주입(최소 core 변경) + EE delta ↔ joint 변환, offline 검증

> writer: lead (A0) · 읽는 쪽: implementer, verifier · 작성 2026-10-13 KST
> branch `feature/panda-benchmark`. 계획: `doc/develope_PLAN_SUMMARY.md` §2.1 SL-5, §0.1(V11AB 동작 유지).
> 사용자 판정 2026-10-13: SL-5는 RB-Y1 기본 동작을 그대로 둔 **최소 core 변경**(주입 지점 추가)으로 진행.
> **착수 조건:** SL4b 측정이 끝난 뒤(같은 working tree에서 core를 바꾸므로). SL5b(live closed-loop 1 episode)는 SL5a 확정 후.

## 왜 지금 이것인가

SL4까지 perception(ATS → target-free layer → ESDF)은 core 무수정으로 Panda에 연결됐다. V11AB의 나머지 절반인 **CCR(chunk refinement,
TrajOpt SQP)** 은 online server(`serve_safe` → `SafePolicy`)에서만 돈다. 그 경로는 RB-Y1 16-D 절대 joint chunk를 가정하고, pi05_libero는
**EE delta(OSC_POSE) 7-D × 10 step** chunk를 낸다. SL5a는 (1) server가 Panda profile을 받도록 주입 지점을 열고, (2) EE delta chunk를
joint reference로 바꿔 V11AB가 정제한 뒤 다시 OSC action으로 되돌리는 변환을 만들고, (3) 이것을 recorded dev episode로 offline 검증한다.

```
pi05_libero chunk A (10×7: Δp·0.05 m, Δθ·0.5 rad, grip ±1)
  → [adapter] 이상 추종 가정으로 EE pose 적분 + IK(seed q_now) → Q_ref (10×7) + gripper column
  → SafePolicy/CCR (V11AB objective·constraint·gate 그대로; horizon만 Panda chunk에 맞춤) → Q*
  → [adapter] 행별: Q*_k == Q_ref_k 이면 **원래 A_k를 그대로**, 바뀐 행만 실행 시점 EE pose 기준 OSC action으로 역변환
```

## 확인된 사실 (audit 2026-10-13, read-only)

| 항목 | 근거 |
|---|---|
| TO core(refiner/sqp/problem/linearize/limits)는 `layout.nq_opt`·`q_indices`만 사용 → 7-joint 단일 팔 가능 | `AG3S/trajopt/refiner.py`, `sqp.py`, `problem.py`, `types.py:78-96` |
| RB-Y1 고정: `ChunkLayout.rby1`(`safe_policy.py:256`, 이름 조회로 crash), `wire.gripper_columns(nq//2)`(`safe_policy.py:262`, Panda에서 joint4를 gripper로 오인), `FingerJointMap`(`:269`, 두 손 가정), hand index(909-929, 966-971, 1124, 1230, 1310-1320) | audit |
| V11AB flag: plan-horizon 16, lookahead obstacles, gate 20 mm, qp-eps 1e-4, w-deflection-rate 20, esdf-margin 0, obstacle-margin 0.010, links gripper, held-body-cover, held-slip-detach, hold-follow-to 3, finger-cover both; HorizonConfig 기본 execution_length 8, control_hz 15 | `scripts/benchmark/v11ab/env.sh`, `start_server.sh`, `AG3S/trajopt/config.py:41-43` |
| optimizer horizon ≤ chunk 행 수; pi05_libero chunk = 10행, client replan 5 | `refiner.py:82-88,119`; `libero_pi05_client.py` |
| joint limit: 기본 RB-Y1 MJCF → Panda는 `--limits-source urdf` + acceleration 기본값 필요 | `config.py:299,316,328`, `urdf_sphere_chain.py:1259-1273` |
| grasp latch/attach는 executed gripper command + 측정 개도 + reach로 판정(subtask label 아님), 개도 1=open 정규화 | `grasp_latch.py`, `safe_policy.py:909-990` |
| OSC_POSE: Δp = 0.05·a (m), Δθ = 0.5·a (rad, axis-angle, **world frame 좌곱**), goal 기준 = 실행 시점 `gripper0_grip_site` world pose, a ∈ [−1,1] clip; gripper −1 open / +1 close; control 20 Hz | robosuite 1.4.1 `controllers/osc.py:259-266`, `base_controller.py:104-123` |
| attention extractor는 RB-Y1 alias 고정, unmirror 없음 | `serve_safe.py:1037-1059, 2346` |

## Lead 결정 (근거 포함 — 사용자 보고 대상)

1. **Horizon 적응:** pi05_libero chunk가 10행이라 V11AB의 plan 16 / exec 8을 그대로 쓸 수 없다. **plan 10 / exec(window K) 5 / control_hz 20**으로
   둔다(V11AB의 plan = 2·K 비율 유지, client replan 5와 일치). 강제된 적응이며 manifest에 V11AB 원값과 함께 기록. 다른 V11AB flag는 그대로.
2. **무개입 동일성:** V11AB가 행을 바꾸지 않으면(‖Q*_k − Q_ref_k‖∞ ≤ 1e-9 rad) 그 행은 policy의 **원래 action을 bit 단위로** 보낸다.
   → 개입이 없을 때 pi0.5+V11AB = pi0.5-only. 변환 오차가 성능 차이로 섞이지 않게 하는 공정성 장치.
3. **바뀐 행의 역변환:** client가 실행 직전에 현재 EE pose로 OSC action을 계산(`a_pos = clip((FK_p(Q*_k) − ee_now)/0.05)`, 회전도 같은 방식).
   saturation 발생 여부를 행별로 기록.
4. **Gripper:** 정제 대상 아님(V11AB passthrough). Panda finger qpos ↔ 정규화 개도(1=open) 변환과 executed command 정규화를 adapter에 둔다.

## 구현에 요청하는 것 (→ implementer)

**Core 변경 원칙:** `AG3S/trajopt/**`·`AG3S/ag3s/**`는 **선택 인자/flag 추가만** 허용. 기본값은 기존과 동일해 RB-Y1 동작이 바뀌지 않아야 한다.
변경 전에 RB-Y1 SafePolicy 경로의 **golden 출력**을 저장하고(가능한 offline 경로, 예: `trajopt/experiments/safe_replay.py` 또는 동등한 고정 입력 fixture),
변경 후 bit 동일함을 test로 보인다. 모든 core diff는 impl.md에 줄 단위로.

1. `serve_safe`: `--robot {rby1,panda}`(기본 rby1). panda면 `AG3S/adapters/panda/profile.build_panda_ag3s` 사용, horizon/limits override, attention extractor에
   alias map·camera list·unmirror 주입, recorder metadata에 profile 기록.
2. `SafePolicy.__init__`: 선택 인자 `layout`, `gripper_columns`, `finger_map`(또는 convention), `hands`(gripper index map); 기본값 = 현재 RB-Y1 동작.
   Panda용 `ChunkLayout`(7 joint + gripper) 생성기는 adapter 쪽에.
3. Robot state 규약: Panda client는 camera별 9-D(q7+finger2)를 보내고 server/adapter가 7 + finger로 split(SL4 `split_robot_state` 재사용).
4. **Action adapter** (`AG3S/adapters/panda/action.py`): EE delta chunk → Q_ref (적분 + IK; damped least squares, seed q_now/이전 해, 수렴·잔차 기록),
   정제 결과 → 행별 원 action 유지 또는 OSC 역변환(client 측 함수와 공용 수식), gripper 정규화. 수식은 robosuite 1.4.1 source와 test로 대조.
5. **LIBERO safe client** (`benchmark/runners/libero_safe_client.py`, py3.8): V11AB client와 같은 request(`ag3s/*` keys, exec_feedback, seq, reset, stamp_mode=sim_frozen,
   phase approach, manipulators left)를 보내고, 응답의 행별 action을 실행. SL1 camera adapter로 geometry 첨부. `--record`/run 규칙은 SL0와 동일. (live 실행은 SL5b)
6. **Offline 검증 도구** (`benchmark/runners/ccr_replay.py`): recorded dev episode(SL3/SL4b run)의 policy 입력·chunk + geometry를 server와 같은 SafePolicy 경로에 넣어
   Q_ref, Q*, gate 열림, 바뀐 행, 역변환 action을 기록(새 run). MuJoCo open-loop 확인용으로 바뀐 행이 없는 chunk의 재생 = 원 기록 bit 동일 확인.
7. Test(CPU): RB-Y1 golden 동일, Panda layout/gripper column, EE 적분·IK 왕복(FK(IK(x)) 잔차), OSC 수식 대조, 무개입 bit 동일, wire 왕복.

**손대지 않을 것:** V11AB objective·constraint·gate 값(horizon 외), RB-Y1 기본 경로, submodule, storage input. EE 공간 직접 최적화는 하지 않는다(별도 variant).

## 측정에 요청하는 것 (→ verifier)

| # | 무엇을 | 숫자 | 통과 기준 |
|---|---|---|---|
| 1 | RB-Y1 동일성 | golden bit 비교, legacy·cuRobo 16D baseline, CPU regression | 모두 동일 |
| 2 | 변환 정확도 | IK 잔차(위치 mm, 회전 deg), EE 적분 vs 실제 실행 EE(dev 기록) 차이 분포 | 숫자만 |
| 3 | 무개입 동일성 | gate가 닫힌 chunk 비율, 그 chunk의 출력 action = 원 action(bit) 비율 | 100 % |
| 4 | 개입 통계 | gate 열린 chunk 비율, 바뀐 행 수, ‖Q*−Q_ref‖ 분포, 역변환 saturation 비율, SQP 반복·시간 p50/p95 | 숫자만 |
| 5 | 비용 | server 1 call(perception+CCR) p50/p95 vs policy 주기 250 ms | 측정 |

- figure 3종(`sl5a-*`): scene(개입 chunk의 Q_ref vs Q* EE 궤적을 실제 scene/ESDF 위에), graph(#2–#5), table(core diff 목록, horizon 적응, 결정 1–4).
- GPU는 실제로 lock을 잡는 형태, 시작·종료 main 통보.

## 이 STEP 이 답하는 질문

V11AB CCR을 RB-Y1 동작 변경 없이 Panda에 연결하고, pi05_libero의 EE delta chunk를 정확히 joint로 옮겨 정제한 뒤, 개입이 없을 때는 원 action을
그대로 보내는가?

## 되돌아올 지점

- golden 불일치 → core 변경 되돌림, 원인 확정 전 진행 금지.
- IK/적분 오차가 voxel(5 mm) 이상으로 크면 → 이상 추종 가정 재검토(OSC 응답 모델 포함 적분) — 사용자 보고.
- 역변환 saturation이 잦으면 → 행 분할 등은 method 변경이므로 variant로 사용자 판정.
