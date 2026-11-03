# 세션 ce 가 스스로 내린 판정 — 2026-11-03 밤 (사용자 검토용)

> writer: lead (세션 ce / ag3s-vla-development-65) · 시작 2026-11-03 00:40 KST
> 사용자 지시 (2026-11-02 23:5x KST): "판단은 스스로 수행하세요. 스스로 판단한 것은 따로 문서로 정리해 놓으세요. 아침 8 시에 확인하고 틀린 판정에 피드백하겠다."
> 판정마다: **무엇을 정했나 · 왜 (기록 근거) · 다른 선택지 · 틀렸으면 무엇을 다시 돌리나.** 사용자가 직접 정한 것은 "사용자 판정" 으로 따로 둔다.

## 0. 사용자가 직접 정한 것 (참고, 이 문서의 판정 아님)

- G1 (기하 grasp 보정, `--geo-grasp*`) 은 이후 모든 단계에서 끈다. 켜라고 할 때까지.
- 쥐는 순간 쥐는 손이 붙은 이웃에 닿는 것은 실패로 세지 않는다 (따로 센다).

## 0.1 사용자 판정 (2026-11-03 18:2x KST) — 이 문서의 판정이 아니라 사용자가 정한 것

1. **P0m 을 최종 구성으로 채택** (T92, `--subtask-from-latch` 명시). serving 의 학습 요소는 VLA 뿐. T91c 는 기본 꺼짐, 손목 카메라가 있는 benchmark 의 선택지.
2. **확정 전에 한 번 더 잰다 (T92b):** (가) P0m 대 (나) P0m − `--surface-hull-fill` − `--surface-contact-allowance-mm 10`, 9 run + 허들 12 run, k1 추가 ((가) k1, (나) k0 · k1). 확정 조건: 채택 구성 9 run 합계 (k0 + k1) ≥ 16 / 18, 금지 접촉 ≤ 1, 탁자 · 바구니 0. (가) 대 (나) 차이 ≤ 2 run 이면 (나) 가 최종. 이 실험으로 "32 run 전 확정 안 함" 을 닫는다.
3. **D1 확인:** benchmark profile 은 `--neighbour-contact-mm` · `--neighbour-palm-contact-mm` 와 평가의 이웃 과일 면제를 모두 끄고, 접촉 허용은 target (쥘 때) · 목적지 (놓을 때) 만. MuJoCo 데모 profile 과 다른 점으로 명시. LIBERO (장애물 없음) pilot 에서 끈 비용을 먼저 잰다.
4. **미채택 4 건 이견 없음:** P1 (충돌 없으면 통과) 은 1 번 (P0m 채택) 뒤 재시험, T93 은 T93b 로, T90 은 seed 가 아니라 "후보 중 prompt 물체 확인자" 역할로 나중에 재설계, T91c 는 benchmark 선택지.
5. **확정 문서 수정:** P0_FINAL_CONFIG · ADOPTED A9 — 옛 3 단계 gate 는 P0 에서 켜져 있었고 P0m 에서 latch 로 대체, A9.2 HSR 5 단계 채택 철회 (`--hsr5` 끔), ball-owned 인자 무효 · `--nontarget-evidence surface` 금지, 바닥 띠 guard 5 mm 와 잔여 측정 제외 6 mm 는 다른 값, 평가 정의의 "붙은 (10 mm)" 조건은 D2 로 뺌, 코드 기준 HEAD f964439d (코드 63e98cc8) · main 32a45fb8 · 새 flag 기본 꺼짐. → 반영함 (18:4x KST).

## 0.2 사용자 판정 (2026-11-04 00:xx KST)

1. **T92b → hull 채우기 · 접촉 여유 10 mm 포함을 최종으로 확정.** 단, 껐다 켰다 할 수 있게 둔다 — 이미 각각 독립 기본 꺼짐 flag (`--surface-hull-fill`, `--surface-contact-allowance-mm M`), benchmark 는 `p0m` · `p0m-nohull` 판. "32 run 전 확정 안 함" 문장은 T92b 로 닫는다.
2. **Panda P0m 의 TrajOpt 정의:** 제안대로 — benchmark 가 같은 LIBERO 9 episode 로 (c) 연속성 · 꺾임 변화율 항만 끈 판, (b) pc판의 Panda 보정을 얹은 판 (target 범위 · 단계 판정은 P0m 그대로, HSR6 끔), (d) 충돌 없으면 VLA 그대로 통과 (`--passthrough-when-clear`) 를 붙인 판을 돌려 포화 원인을 가른다. 성공이 비슷하면 보정이 적은 쪽. SafeLIBERO closed loop 전에 "지운 voxel 0 call 41 %" 원인을 기록으로 확인.

## 1. 판정 목록

| # | 판정 | 왜 | 다른 선택지 | 틀렸으면 다시 돌릴 것 |
|---|---|---|---|---|
| D1 | **벤치마크 (SafeLIBERO · LIBERO-safety) 에서 접촉 허용은 target 자체 (쥘 때) 와 목적지 (놓을 때) 로만 좁힌다.** MuJoCo 데모 평가는 "붙은 이웃 접촉 허용" 그대로 둔다. | SafeLIBERO Level I 은 target 옆에 장애물을 두고 그 접촉을 충돌로 센다 (`paper/notes/SAFELIBERO_PLAN.md` §2). 이웃까지 허용하면 Level I 에서 충돌로 잡힌다. | 벤치마크에서도 이웃 허용 (Level I 충돌 증가 감수) | 벤치마크는 아직 안 돌렸다. task 문서만 고친다 (재실험 없음) |
| D2 | **"쥐는 순간 이웃 접촉" 창 = 정답 쥠 (두 손가락 접촉 + 20 mm 상승) 전 ∧ 쥐는 손 손바닥–사과 중심 ≤ 60 mm, 쥐는 손 (손바닥 · 두 손가락 subtree). 이 창 안에서 쥐는 손이 이웃에 닿으면 이웃–사과 거리와 상관없이 실패로 세지 않는다** (dev-07 임시 값의 "사과 표면 10 mm 안 이웃" 조건을 뺌, 00:5x KST 에 dev-07 에 알림). 집계 코드 = dev-07 `verify/T87/agg/grasp_contact.py` 복사 + `ATT_MM = inf`. | 사용자 표현 "물체를 집을 때는 그냥 접촉 허용". dev-07 검증에서 회귀 954 의 같은 바나나 문지름이 간격 10 / 11 mm 로 갈렸다 (12 step 제외 · 5 step 금지) — 임의 경계. 60 mm 는 A6 · 정답 단계 "쥐기" 와 같은 값 | 10 mm 조건 유지 (dev-07 임시 값) | 집계만 다시 (CPU, 재실험 없음) |
| D3 | **다음 기준 구성 (T88 P0) = HSR 규칙 4 개 끔 · 분류기 없음 [정정: 옛 3 단계 gate 분류기 `kv_L4_v1` 는 P0 에서 켜져 있었고 P0m 에서 latch 로 대체 — 1.0 절 · D19] (쥠 상태는 grasp latch 의 측정 개도 · feedback 만) · G1 끔 · 표면 추출 (분할 없음) + hull 채우기 + 접촉 여유 10 mm · E0 TrajOpt.** target 유지 기억은 넣지 않는다. | ① 사용자: "HSR 끄고 물리 신호나 정답으로 단계" · 기억 "HSR6 폐기". ② T85 에서 HSR 규칙이 정답 단계로 G1 을 풀어 쥐기 2 건을 망쳤다. ③ 접촉 여유는 T78f 구성과 같게 해 S9cl (4 / 10) · T85 (3 / 10) 와 한 변수씩 비교. 유지 기억 효과는 dev-07 T87 이 따로 잰다 | `--hsr5` (5 단계 분류기) 를 넣은 기준 (dev-07 T87 B0) | 그 구성으로 P0 다시 (closed-loop 10 run, 약 40 분) |
| D4 | **새 기능 T88: VLA 계획에 충돌 위반이 없으면 TrajOpt 가 VLA chunk 를 그대로 통과 (`--passthrough-when-clear`, 기본 꺼짐).** | T85 실패 run 에서 2° 이상 바꾼 요청 중 15 개가 VLA 계획 위반 0. 10 run 이 모두 VLA 단독 성공 장면이므로 이런 수정은 해만 낸다. LIBERO (장애물 없음) 에서 성공률을 pi0.5 와 같게 유지하려면 필요. | 비용항 (손목 추종 약화 · 연속성) 을 하나씩 끄기 | flag 를 끄면 지금과 같다 (bit 동일 test). P1 결과만 버리면 된다 |
| D5 | **P0 는 `snapshot_t85` 코드로 먼저 돌린다** (T88 구현 전). P1 은 `snapshot_t88`. | GPU 가 밤새 dev-07 T87 (120 run) 과 나눠 써서, 구현을 기다리면 P0 를 못 돌릴 수 있다. T88 코드는 꺼짐 = bit 동일이라 P0 결과는 같아야 한다. | P0 · P1 모두 `snapshot_t88` | 두 snapshot 의 차이 파일 목록을 아침 보고에 적는다. 차이가 T88 블록 밖이면 P0 를 snapshot_t88 로 다시 |
| D6 | **평가는 새 정의 (쥐는 순간 이웃 접촉 제외) 로 판정하고 옛 정의 결과도 같이 낸다.** 10 run 차이 ≤ 2 는 "구분 못 함". | 사용자 판정 + dev-07 T87 §3 과 같은 규칙. 같은 구성에서도 run 결과가 바뀌는 것을 T85 에서 봤다 (허들 119 · 850 · 가림 121) | 성공 수만 보고 | 집계만 다시 |
| D7 | **T85 의 "HSR 끄고" 해석이 틀렸음을 인정하고 P0 를 그 해석의 올바른 시험으로 둔다.** T85 는 분류기만 끄고 HSR 규칙 4 개는 켠 채 정답 단계를 넣었다. | 사용자 지적 (2026-11-02) "HSR 끄고, 단계를 물리적 그리퍼나 정답을 활용" | — | — |
| D8 | **실패 원인 층을 성공률 대신 요청 단위 기록으로 적는다** (T85 진단 도구 `t85_bend_cause.py`: 큰 수정 · HOLD 요청마다 VLA 계획 위반 · 걸린 충돌 행의 실제 물체 · G1 작동). | 사용자: "성공률만 제공하고 뭐가 안된다고만 하면 해결할 수 없다" | — | — |

| D10 | **T88 통과 경로는 기존 `_passthrough` (status UNCONSTRAINED) 가 아니라 `_finish(reference, iterations=0)` (status OPTIMAL, 충돌 측정 키 포함) 를 쓴 구현을 받아들인다** (implementer 판단, `T88.impl.md`). | UNCONSTRAINED 이면 safe_policy 가 `unverified` 로 HOLD 할 수 있고 clearance 기록이 없다. OPTIMAL 경로는 측정값을 그대로 남긴다 | 기존 `_passthrough` 재사용 | P1 결과만 다시 (40 분) |
| D11 | **허들 hurdle run 의 "쥐는 순간 이웃 접촉" 판정은 outcome 도구 (hurdle_outcome) 의 kinematic 재생 frame 에 창 판정을 붙여 쓴다** (`verify/T88/agg/t88_outcomes.py`, 원본 `t88_outcomes.orig.py`). 회귀 run 은 dev-07 `grasp_contact.py` (물리 재생 json). | 접촉 출처를 원래 판정과 같게 둔다. S9cl · T85 허들 211 의 접촉은 dev-07 도구와 같은 사건으로 확인 | dev-07 도구의 kinematic 경로로 통일 | 집계만 다시 |

| D12 | **P1 (충돌 없으면 통과) 은 채택하지 않는다. flag 는 기본 꺼짐으로 코드에 남긴다. 다음 기준 구성은 P0.** | P1 은 사전 기준 미통과 (금지 접촉 3 · 탁자 1). 다만 금지 접촉 3 건 중 2 건은 통과가 아니라 **손바닥 구 모델이 몸체를 다 못 덮는** 기존 문제 (계획기가 손바닥 몸체–장애물 거리를 실제보다 멀게 봄) 이고, 1 건은 미확정. 통과 규칙 자체는 설계대로 작동했다 (위반 0 인데 바꾼 요청 11 → 6, 남은 6 은 한계 초과 · 인증 없음으로 일부러 거절) | P1 채택 (해 감소 효과 기대) | P1 을 손바닥 구 보완 뒤 다시 (약 30 분) |
| D13 | **다음 STEP 후보 1 순위 = 손바닥 (EE_BODY) 구 모델 보완** (서버 시작 로그가 이미 "ee_left · ee_right 구 6.1–6.2 mm, 덮으려면 33.3 mm, 27.2 mm 부족" 이라고 경고). 2 순위 = 손 층에 남는 로봇 자신 voxel (P1 HOLD 9 중 4) · 사과 내부 voxel. | P1 · P0 실패 · HOLD 의 걸린 점 정답 분류 | 허들 우회 (BENCHMARK 전략 P3) 를 먼저 | — (아직 착수 안 함, 사용자 검토 뒤). 기존 flag `--capsule-radius-scale-link ee_left=… ee_right=…` 로 손바닥 몸체 구만 키울 수 있으나, URDF 크기 (반지름 33 mm) 로 키우면 쥐는 순간 사과 · 탁자와 겹쳐 쥐기를 막을 위험이 있어 밤사이 시험 없이 돌리지 않았다. 손바닥 껍질 (`--palm-shell`) 과 겹치는 범위부터 확인한 뒤 정한다 |
| D14 | **(13:3x KST) 작업 단계 "최소 구성" = 3 단계 gate 의 label 을 grasp latch phase 에서 만든다 (T92 `--subtask-from-latch`: searching · latched · closing → pick, held → place, placed → home).** `--no-subtask-gate` (gate 끔) 이 아니라 label source 만 바꾼다. | 사용자 "작업 단계 결정은 우선 최소 구성". P0 에서 학습 분류기가 맡는 일은 이 gate 하나뿐이고 (21 / 396 요청에서 새 target 선택을 막음), gate 를 통째로 끄면 놓은 뒤 다른 과일을 새 target 으로 잡아 지울 수 있다. latch 는 Panda 에서도 같은 신호 (개도 · 실행 feedback) | `--no-subtask-gate` · dev-07 T91 상태 기계 (RBY1 offline 사전 기준 미통과) | P0m closed-loop 9 run (약 30 분) |
| D15 | **아이디어 A (T93) 는 "VLA chunk 에 충돌 위반이 있을 때만, 같은 noise 로 다시 샘플링하며 거리장 비용 gradient 로 유도" 하는 2 회 샘플링으로 넣는다.** 위반이 없는 요청은 bit 동일, 유도 뒤에도 TrajOpt · HOLD 판정은 그대로. | 거리장은 이번 요청의 attention 으로 짓기 때문에 첫 샘플링 전에는 없다. 2 회 샘플링이면 이번 요청의 정확한 거리장을 쓰고, 위반 없는 요청 (대부분) 은 손대지 않는다 (P1 에서 본 "해" 를 만들지 않음) | 직전 요청의 거리장으로 첫 샘플링부터 유도 (1 회, 거리장 1 요청 늦음) | T93 closed-loop 만 다시 |
| D16 | **T93 · T94 · T95 의 효과는 놓는 곳 허들 7 장면 × 2 에서 허들 211 을 뺀 12 run 으로 판정하고, 기존 9 run 으로 해 없음을 확인한다.** | P0 는 기존 9 run 에서 9 / 9 (새 정의) 라 우회 기법이 이길 자리가 없다. 허들 14 run 은 VLA 단독 0 / 14, 2026-10-29 최종 구성 6 / 14 — 경로를 막는 장애물 우회가 필요한 장면 | 기존 9 run 만 | 장면 묶음을 바꿔 다시 |
| D17 | **T92 · T93 구현을 implementer 2 개로 동시에 돌린다** (서로 독립 · 같은 파일은 STEP 블록으로 나눔). | 두 구현은 결과가 서로를 정하지 않는다. 같은 파일 (safe_policy · serve_safe) 은 편집 직전 다시 읽기로 충돌을 막는다 | 순차 구현 | — |
| D18 | **(15:3x KST) T93 구현의 두 가지 지시 이탈을 받아들인다:** ① 첫 VLA 호출에 `noise=` 를 넘기지 않고 정책 key 에서 같은 noise 를 미리 읽는다 (`fm_guide.peek_noise`). ② 재샘플을 `sqp.solve` 안이 아니라 `TrajOptChunkRefiner.refine` 에서 reference 를 바꾸는 형태로 넣는다 (`sqp.py` 무변경). | ① `noise=` 를 넘기면 jit 그래프가 바뀌어 꺼짐 = bit 동일이 깨질 수 있다. 실제 model 에서 유도 세기 0 재샘플이 원래 chunk 와 0.0 rad 차이로 확인됨. ② solve 앞에서 바꾸면 lookahead gate · `_solve_window` 가 모두 같은 유도 reference 를 본다 | 지시대로 (`noise=` · sqp 안) | T93 closed-loop 만 다시 |
| D19 | **T92 (작업 단계 최소 구성) 은 9 run 사전 기준을 통과했다고 판정하고, P0m (= P0 + `--subtask-from-latch`) 을 T93 의 기준 구성으로 쓴다.** | 새 정의 9 / 9 (P0 9 / 9) · 옛 정의 7 / 9 (P0 7 / 9) · 금지 접촉 0 · 탁자 0 · gate 가 막은 21 요청 모두 놓은 뒤 (놓기 전 0). 기준 1–3 모두 충족 | P0 를 기준으로 | — |
| D20 | **T93 기본값 = 유도 세기 300 · 마지막 5 개 flow 단계 · 단계당 0.05 rad** (구현자가 smoke 로 한 번 고정). closed-loop 전에 바꾸지 않는다. | smoke (허들 954, 위반 요청 4) 에서 이 설정이 4 요청 모두 위반 0 mm, 관절 변화 0.055–0.080 rad. 10 단계 전부 유도는 34 mm 요청에서 5.4 mm 에 멈춤. 근거가 위반 요청 4 개뿐이라 일반성은 closed-loop 이 확인한다 | 세기 100 · 10 단계 | T93 closed-loop 만 다시 |
| D21 | **(15:3x KST) 네트워크 끊김으로 종료 신호를 받은 T92 허들 run 1 개 (1345 흔들림 1, 첫 요청에서 중단) 는 기록을 scratchpad 로 옮기고 같은 코드 · 같은 서버 인자로 그 run 만 다시 돌린다.** 이후 긴 실행은 세션과 분리 (setsid) 해 띄운다. | 첫 요청에서 끊겨 결과가 없다. 같은 snapshot · seed 라 다시 돌려도 다른 run 과 비교 가능 | 허들 기준을 11 run 으로 | — |
| D22 | **(17:4x KST) T93 (VLA flow 유도) 은 사전 기준 ① · ② 미통과로 채택하지 않는다 (`--fm-guide` 기본 꺼짐).** T90 (hidden probe target 신호) 은 사전 조건 (다른 물체 선택 요청이 늘지 않음) 미충족으로 다음 단계 (어려운 장면) 로 가지 않는다. | 사전 고정 기준의 적용. T93: 9 run 6/9 · 허들 6/12 (P0m 9/9 · 7/12). T90: 사과 아닌 다수 요청 92 → 96 | — | — |
| D23 | **(17:5x KST) T91c (손목 depth 단계를 gate label 출처로) 는 사전 규칙 §2.3 대로 "latch 보다 낫다" 가 아니다 → gate 출처는 T92 latch 유지, T91c 기본 꺼짐.** | 재생에서 gate 결정이 바뀐 요청 1 (787 중). 그 run (허들 119 흔들림 1) closed-loop 결과 latch · T91c 모두 허들 접촉 (성공 0 대 0) | — | — |

## 1.0 정정 (2026-11-03 08:xx KST)

- **P0 · P1 의 "HSR 끔" 은 정확하지 않다.** 단계 규칙 4 개 (`--hsr6-*`) · 5 단계 분류기 · G1 은 껐지만, **옛 3 단계 분류기 gate (`kv_L4_v1`) 는 서버 기본 on 이라 켜져 있었다** (P0 396 요청 모두 gate on, 21 요청에서 새 target 선택을 막음). 이 gate 를 끄려면 `--no-subtask-gate` 가 필요하다.
- 사용자 지시 (2026-11-03): 허들 211 은 이후 모든 시험 · 실험에서 제외. 그래서 9 run 기준 P0 = 새 정의 9 / 9, 옛 정의 7 / 9.

## 1.1 밤사이 결과 요약 (05:4x KST 갱신)

| 구성 (같은 10 run · 같은 seed) | 성공 옛 정의 | 성공 새 정의 | 금지 접촉 (새) | 결과 폴더 |
|---|---|---|---|---|
| 공 방식 (T74 Q1) | 8 | 8 | 0 | `outputs/2026-10-31_T78_표면추출_closedloop_10run` |
| 표면 + 분할, HSR6 분류기 (T78 S3) | 5 | 5 | 1 | 같음 |
| 표면 + 접촉 여유, HSR6 분류기 (S9cl) | 4 | 4 | 1 | `outputs/2026-11-02_T78f_접촉여유_closedloop_10run` |
| 표면 + 유지 기억, HSR 끔 (dev-07 T84SR) | 4 | 6 | 1 | dev-07 |
| 표면 + 접촉 여유, HSR6 규칙 + 정답 단계 (T85) | 3 | 5 | 2 | `outputs/2026-11-02_T85_정답단계공급_closedloop_10run` |
| **P0: 표면 + 접촉 여유, HSR 끔 · G1 끔** | **7** | **9** | **0** | `outputs/2026-11-03_T88_P0_HSR끔_G1끔_단순구성_closedloop` |
| P1: P0 + 충돌 없으면 통과 | 6 | 7 | 3 | `outputs/2026-11-03_T88_P1_충돌없으면통과_closedloop` |

- 그림: `AG3S/ag3s/docs/figures/t88-p0-config-compare.png` (P1 열 없음).
- 이 표는 seed 1 개 · 10 run 이다. 같은 구성에서도 run 결과가 바뀐다. 32 run 또는 seed 반복 전에는 확정하지 않는다.

## 1.2 (03:0x KST 기록)

- **P0 (HSR 끔 · G1 끔) = 7 / 10 (옛 정의), 9 / 10 (새 정의), 금지 0, 탁자 · 바구니 0** — T88 §2 사전 기준 충족 (새 정의). 같은 10 run 에서 분류기 단계 4 / 10, 정답 단계 3 / 10 (새 정의 5 / 10). 결과 `outputs/2026-11-03_T88_P0_HSR끔_G1끔_단순구성_closedloop/README.md`.
- 남은 실패 허들 211 = 사과 안쪽 voxel 이 손 층에 남아 HOLD (LTE) → TrajOpt 꺾임 (TO). 붙은 이웃 장면.
- **주의:** 이 결과는 seed 1 개 · 10 run 이다. 같은 구성에서도 run 결과가 바뀌는 것을 T85 에서 봤다. 32 run 또는 seed 반복 전에는 "HSR 을 끄면 좋아진다" 를 확정하지 않는다.
- T88 implementer 가 전체 `tests/trajopt` 를 한 process 로 돌리다 pytest 가 약 30 GB 까지 커져 pod memcg OOM 이 났다 (본인 보고, 01:xx). 다른 세션 process 영향 여부는 확인하지 못했다.

- 03:25 P1 lock 획득 → 서버가 시작에서 거절 ("Output must be inside the run"): P0 집계가 run 디렉터리 안에 만든 영상 보기용 symlink (`agg/outcome_p0/view_occ/placement4` → 다른 run) 때문. symlink 폴더를 scratchpad 로 옮기고 (삭제 아님) 03:35 KST 에 P1 을 다시 줄 세움. 집계 스크립트도 끝나면 view 폴더를 run 밖으로 옮기게 고침.

## 2. 다른 세션과 어긋나는 점 (사용자 확인 필요)

- dev-07 T87 은 "채택한 HSR 은 5 단계 (`--hsr5`)" 를 기준으로 6 구성 × 20 run 을 돈다. 이 세션은 사용자 지시 ("HSR 끄고") 로 HSR 규칙을 끈 기준 (P0) 을 쓴다. 둘 중 어느 쪽이 최종 기준인지 사용자가 정해 주면 다른 쪽을 그 기준으로 다시 돌린다.

## 3. 진행 기록 (시간 · KST)

- 00:24 run 디렉터리 `20261103T002419+0900_8ab3c7ef7757_82e2790bab11` (T88).
- 00:3x T88 task 문서, implementer 1 개 (pass-through 구현).
- 00:4x P0 kit (`verify/T88/kit`, launcher guard: HSR · G1 · 분할 flag 금지) 점검 통과, 독점 lock 줄에 섬 (T87 실행 중, pod working set 61.8 GiB).
- 01:01 P0 closed-loop 시작 (서버 검사 통과).
- 01:1x T88 구현 끝 (`T88.impl.md`, 단위 시험 27 통과). `snapshot_t88` 생성 · 점검 통과. **D5 확인:** `snapshot_t85` 대비 바뀐 코드 = T88 (`sqp.py` · `config.py` · `serve_safe.py`, 꺼짐 = bit 동일 test) + dev-07 T86 의 오프라인 파일 (`pi05_hidden.py` 의 patch pool · `t86_*.py`, serving 은 `--hsr5` 일 때만 `pi05_hidden` 을 씀 = P0 · P1 은 안 씀). 그래서 P0 를 `snapshot_t85` 로 돌린 것은 P1 과의 비교에 영향이 없다고 판단 (P0 재실행 안 함).
- 01:3x **주의:** P0 closed-loop 중에 이 세션의 다시 집계 프로세스가 CPU 약 13 코어 (numpy thread) 를 쓰고 있었다 (pod load 99 / 256 코어, 다른 세션 작업 포함). P0 첫 장면이 약 15 분 걸렸다 (T85 는 3–4 분). 일시 정지를 시도했으나 권한이 거부되어 그대로 두었다. TrajOpt 는 시간 예산 (50 ms) 이 있으므로 집계에서 시간 예산에 걸린 요청 비율을 T85 와 비교해 영향을 확인한다. 영향이 크면 P0 · P1 을 다시 돌린다.
  - **D9 (01:45 확인):** P0 첫 81 chunk 의 TrajOpt 풀이 시간 중앙 501 ms (T85 같은 수 101 ms) 로 5 배 느리지만, 시간 예산에 걸린 chunk 0 (반복 상한 3 회로 끝남) 이고 client 는 추론 동안 시뮬레이터를 멈춘다 (`stamp_mode sim_frozen`). 그래서 느려진 것은 벽시계뿐이고 결과는 같다고 판단해 **다시 돌리지 않는다.** 틀렸다면 P0 · P1 을 한가한 시간에 다시 (각 약 40 분).
- 01:2x P1 독점 lock 줄에 섬. 같은 시각 다시 집계 (공 방식 · 분할 켠 표면 · 접촉 여유 · HSR 끔 · 정답 단계, 새 평가 정의) 진행 중 (CPU 1 worker, run 당 약 5 분).
