# 인계 — 세션 dev-07 (lead, 지금 이름 ag3s-vla-development-da) → 세션 d2 (e1/eb 대화)

> 작성: 2026-11-03 16:35 KST · 사용자 지시 "d2 에게 니가 인계 내용을 전달하세요" (development checkout 을 세션 하나로 모음).
> 이 문서 뒤로 dev-07 은 새 작업을 시작하지 않는다. 판정 기록 정본: `AG3S/ag3s/docs/handoff/DECISIONS_20261103_dev07_KO.md` (E1–E18).

## 0. 한눈에

| STEP | 무엇 | 상태 | 남은 일 |
|---|---|---|---|
| T86 | hidden patch probe 로 ATS target 찾기 (offline) | 끝 — 결과 `outputs/2026-11-03_T86_ATS_target찾기_hidden_probe/` | — |
| T87 | 표면 추출 위 기능 4 개 따로 재기 (HSR 5 단계, 구성당 18 run) | 끝 — 네 기능 모두 기본 꺼짐 유지 (E14). `outputs/2026-11-03_T87_표면추출위_기능따로재기_HSR5단계/` | — |
| T89 | G1 기하 grasp 보정을 쥐기 구간에서 강제로 켬 (9 run) | 끝 — 해로움, G1 끔 유지 (E16). `outputs/2026-11-03_T89_기하보정_쥐기구간강제_closedloop/`, `T89.verify.json` | — |
| **T90** | T86 probe 를 serving ATS target 신호로 (`--attention-source hidden_probe`), 9 run closed-loop | **closed-loop 끝, 집계 중간** | §1 |
| T91 | 학습 없는 물리 상태 기계로 5 단계 인식 (offline) | 끝 — 사전 기준 미통과 (E12) | — |
| T91b | 손목 depth "손가락 사이 표면" 으로 운반 판정 (offline, RBY1 + SafeLIBERO) | 끝 — RBY1 통과 (E15), SafeLIBERO 사전 기준 미통과 · 원인 확인 (E18) | §3 (benchmark 평가) |
| **T91c** | T91b 를 serving 3 단계 gate label 출처로 (`--subtask-from-phase-machine`) | **구현 중단 (재연결로 implementer 끊김)** | §2 |

## 1. T90 — 집계 이어받기
- run: `/mnt/dev/AG3S_VLA-storage/results/development/20261103T125521+0900_ed0145a692ad_47f74feee80d/verify/T90/` (snapshot_t90 = git archive ed0145a6, `kit/flags_P.txt`). 비교 B0 = T87 기준 구성 k0 같은 9 run (`…905ff913aad2/verify/T87`).
- 사전 고정 (`T90.task.md`): 주 지표 = 금지 접촉 없는 성공 (쥐는 순간 이웃 과일 접촉 면제 = 보정 성공) · 실패 첫 사건 · 띠 제외 잔여 · ATS target 정밀도 (target 점 중 정답 사과 10 mm 안 비율). sigmoid 확률 그대로 (E13).
- 지금까지 (`agg/t90_table.md`, verifier 숫자, 9 run 허들 211 제외):

| 구성 | plain 성공 | 보정 성공 | 금지 접촉 run | 첫 사건 b · e2 | 띠 제외 잔여 중앙 | 쥐기 창 HOLD 합 |
|---|---|---|---|---|---|---|
| P (hidden probe) | 5 | 5 | 1 | 2 · 2 | 0.156 | 37 |
| B0 (명사 attention) | 6 | 7 | 1 | 1 · 1 | 0.143 | 2 |

  run 별로 P 가 잃은 것: 회귀 954 (이웃 접촉 → 첫 사건 b), 허들 954 (못 놓음 e2). 둘 다 B0 성공.
- **아직 없는 것:** ATS 정밀도 열 (지금 표는 None — `t90_aggregate.py` 가 `KeyError: 'precision'` 로 멈췄다, precision 계산 끝나기 전 실행). 계산 process 2 개가 아직 돈다 (dev-07 scratchpad `envrun.sh` 아래, 메모리 대기 · nice 10):
  - pid 4136356 = `agg/t90_ats_precision.py agg/spec/spec_P_k0.json … agg/ats/ats_P_k0.json` (log `agg/logs/ats_P_k0.out`)
  - pid 4123697 = 같은 도구로 B0 (`…T87/agg/spec/spec_B0_k0.json`, log `agg/logs/ats_B0_k0.out`). 앞선 B0 시도 2 개는 worker 한도로 멈춤 (`*.stopped_for_worker_limit.out`, `*.killed_for_worker_limit.out`).
  - 끝나면 `agg/t90_aggregate.py` 다시 → `t90_table.{md,json}`, 그 뒤 figure (`AG3S/ag3s/docs/figures/t90-*`, 실제 씬 · 그래프 · 표) · `AG3S/ag3s/docs/handoff/T90.verify.json` · `outputs/2026-11-03_T90_ATS_hiddenprobe_serving_closedloop/` (영상 `python -m common.experiment_view`, B0 와 같은 장면 나란히). `run_t89_t90_agg.sh` 가 순서를 갖고 있다.
- 판정 문장 규칙 (task 고정): 성공 수 차 ≤ 2 run 은 "구분 못 함". P 5 vs B0 7 (보정) 은 차 2 → 구분 못 함. 다만 HOLD 37 vs 2 와 첫 사건 b (쥐지 못함) 가 늘어난 것은 원인 진단 대상 (ATS 정밀도로 target 점이 사과 밖으로 갔는지 확인 → ATS 층인지 HSR/CCR 층인지).
- 사용자 맥락: 사용자는 T86 offline 9 %p 상승을 "엄청난 효과" 로 보고 추가 시험을 지시했다 → 결과가 구분 못 함이어도 정밀도 · HOLD 원인을 같이 보고해야 한다.

## 2. T91c — 구현 이어받기 (`T91c.task.md`)
- 왜: T92 latch gate 가 9 / 9 를 냈지만 offline 위험한 오판은 latch 2.5 % · T91b 0.49 % (T77 test, 장면 평균). 그래서 **먼저 T92 P0m 기록 고정 입력 재생으로 두 label 이 달라 gate 결정이 바뀌는 요청이 있는지** 보고, 있을 때만 closed-loop (E17).
- 작업 트리에 이미 들어간 것 (implementer 가 07:12 UTC 까지 쓴 것, py_compile 통과, **test 미실행**):
  - `AG3S/trajopt/phase_gate.py` (새, `PhaseGate` — 서버가 쓰는 phase machine 감싸기)
  - `safe_policy.py` `# --- T91c` 블록 6 곳 (178 · 236 · 617 · 1396 `_subtask_gate_input` · 1407 · 2795 기록), `serve_safe.py` 2 곳 (2583 flag · 3169 거절 / kwargs)
  - `AG3S/ag3s/experiments/studies/t91c_gate_replay.py` (새, docstring 에 입력 출처 정리: 측정 개도 = client `state[7|15]`, 손 site = qpos FK, ATS target / 목적지 = 직전 요청 chunk 기록, VLA 의도 = `reference_chunk` gripper 열, 손목 depth = client step `depth_wrist_cam_<l|r>` stride 4)
- 없는 것: `tests/trajopt/test_t91c_phase_gate.py`, `T91c.impl.md`, 재생 실행.
- 확인할 것: (1) 서버가 요청마다 손목 depth 를 실제로 받는지 (replay 는 client step 의 depth 를 읽음 — serving 경로에서 같은 key 가 서버 요청에 들어가는지), (2) 꺼짐 bit 동일, (3) `--subtask-from-latch` · `--hsr5` · `--hsr-gt-stage` · `--no-subtask-gate` 와 동시 사용 거절, (4) T92 · T93 블록 무변경.
- 권장: 새 implementer 를 `T91c.task.md` + 이 절로 띄워 test · impl.md 를 마치게 하고, 재생은 lead 가 직접.

## 3. T91b — SafeLIBERO 점검 결과와 benchmark 평가
- `outputs/2026-11-03_T91b_손목카메라_손가락사이_운반판정_RBY1_SafeLIBERO/` (README 점검 절, `2_SafeLIBERO_오판원인_다른bowl을쥠.png`).
- 정정: 예전 보고의 "6 episode 에서 72–146 mm 들림" 은 부호 실수 (생성 직후 낙하). 위험한 오판 24 call 중 21 call (spatial t1_e0 · t1_e2) 은 VLA 가 같은 모양의 다른 bowl (bowl_2) 을 쥔 구간 — 손–bowl_2 49–59 mm, 손–목표 bowl 278–328 mm. T91b 의 "운반" 은 물리적으로 맞다. 두 episode 를 빼면 0.50 % 지만 결과 뒤 결정이라 판정은 "미통과 (4.3 %)" 그대로 (E18).
- 다음: benchmark checkout 에서 같은 평가를 작업 4 개 이상으로, "무언가를 쥐었나" 와 "쥔 물체가 목표인가" 를 따로 센다. SafeLIBERO 정확도 56 % (RBY1 83 %), 놓기 · 물러나기 구분이 약함. benchmark 쪽은 세션 benchmark-44 / -84 와 조율.
- 원시: RBY1 `…20261103T155417+0900_ed0145a692ad_9c4075e3612b/verify/T91b/t91b_eval.json`, SafeLIBERO `…20261103T153040+0900_ed0145a692ad_ab86b7d536e0/verify/T91b/`.

## 4. 커밋 목록 (d2 가 이미 받은 목록 + 추가)
- dev-07 소유 (미커밋): `AG3S/trajopt/phase_machine.py`, `AG3S/trajopt/finger_gap.py`, `AG3S/trajopt/phase_gate.py`, T91c 블록 (safe_policy · serve_safe), `AG3S/ag3s/experiments/studies/{t91_phase_eval,t91b_safelibero,t91b_safelibero_export,t91c_gate_replay}.py`, `tests/trajopt/{test_t91_phase_machine,test_t91b_finger_gap}.py` (+ T91c test 생기면), handoff `T89/T90/T91/T91b/T91c.task.md`, `T91.impl.md`, `T91b.impl.md` (있으면), `T89.verify.json`, `DECISIONS_20261103_dev07_KO.md`, 이 문서, figure `t89-*` (+ `t90-*` 생기면).
- T91c 블록은 test 전이면 커밋에서 빼거나 test 를 먼저 마친다.
- 커밋 · benchmark merge 는 사용자 지시 때만 (main 02fe59e6 → feature/panda-benchmark merge 는 사용자 판정 대기, benchmark-44/-84 대기 중).

## 5. 지켜야 할 사용자 규칙 (이 대화에서 확정)
- HSR = 5 단계 (`--hsr5`), 3 단계 gate 는 서버 기본 on → kit 에 required / forbidden 확인 (메모리 `hsr6-discarded-check-flags`). T92 계열은 `--subtask-from-latch` 로 gate 출처를 바꿈.
- G1 (`--geo-grasp*`) 끔 (T89 는 예외였음). 쥐는 순간 쥐는 손의 이웃 과일 접촉은 실패 아님. 잔여 target voxel 은 바닥 띠 제외만. 허들 211 (HURDLE_ep211_s2113) 은 모든 집계에서 제외 (회귀 211 유지).
- 스스로 판단하고 판정은 DECISIONS 파일에 기록, 사용자가 검토. 보고는 한국어, outputs 폴더 먼저.
- 공용 pod: working set ≤ 64 GiB, worker ≤ 4 (위 T90 precision 2 개가 worker 2 를 쓰고 있음), 서버 ≤ 2, closed-loop 은 서버 1 개씩.

## 6. dev-07 이 남긴 process
- pid 4118788 · 4118850 (`envrun.sh`) → 4136356 · 4123697 (T90 ATS precision, §1). 끝나면 log 끝에 `rc=… <시각>` 이 찍힌다. 다른 dev-07 process 없음 (T91c implementer · T90 verifier 는 재연결 때 끊김).
