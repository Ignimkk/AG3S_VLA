# benchmark 이식 지시 — 2026-11-03 판정 4 가지 (T92 · T93 · T90 · T91c) 를 LIBERO · SafeLIBERO · LIBERO-safety 에서 시험

> writer: development lead (세션 d2 = e1/eb 대화) · 작성: 2026-11-03 18:0x KST.
> 사용자 지시 (2026-11-03): "오늘 판정 정리한 4 가지도 benchmark checkout 으로 넘겨서 benchmark 3 가지 시뮬레이션에서 테스트하도록 지시해 주세요."
> 받는 쪽: benchmark checkout (`/mnt/dev/AG3S_VLA-benchmark`, `feature/panda-benchmark`) 세션. 앞선 이식 문서 `P0_FINAL_CONFIG_FOR_BENCHMARK_20261103_KO.md` (P0 구성 전체) 위에 더하는 것이다.

## 0. 코드 전달 경로

- development `feature/ag3s-development` 커밋 `63e98cc8` (코드) · `9ee95bb7` (문서) → **main `32a45fb8`** (이전 main `02fe59e6` = P0 구성 후보도 포함).
- benchmark 는 아직 `02fe59e6` 를 merge 하지 않았다 (시험 merge · 회귀 비교 중). **이번 지시로 main `32a45fb8` 을 merge 대상으로 바꾼다.** 시험 merge → 회귀 test 비교 (merge 전 HEAD 와 같은 실패인지) → merge 절차는 benchmark 쪽 기존 사용자 판정대로 따른다.
- 새 기능은 모두 **기본 꺼짐 flag** 다. 꺼짐 = 이전과 bit 동일 (development test). merge 만으로 benchmark 동작은 바뀌지 않는다.

## 1. 4 가지 판정과 무엇을 시험하나

| STEP | 기능 · flag | RB-Y1 판정 (2026-11-03) | benchmark 에서 시험할 질문 |
|---|---|---|---|
| **T92** | 3 단계 gate label 을 학습 분류기 대신 grasp latch 상태에서 (`--subtask-from-latch`). searching · latched · closing → pick, held → place, placed → home | **통과, 최종 구성 후보.** 9 run 9/9 (분류기 gate 와 같음), 놓기 전 gate 막음 0. 허들 12 run 7/12 | 학습 분류기 (`kv_L4_v1`, RB-Y1 전용) 가 없는 Panda 에서 단계 출처를 latch 로 바꿔도 성공 · 접촉이 유지되는가. **benchmark 의 단계 출처 기본 후보** |
| **T93** | VLA (pi0.5) flow-matching 생성을 거리장 충돌 비용으로 유도 (`--fm-guide`, 세기 300 · 마지막 5 단계 · 0.05 rad) | **미통과, 기본 꺼짐.** 9 run 6/9 (기준 9/9), 허들 12 6/12 (기준 7/12). 위반 감소 80 %, HOLD 45 → 12. 해는 다가감 · 닫는 중 유도에서, 효과는 운반 중에 남 | 장애물이 경로를 막는 SafeLIBERO · LIBERO-safety 장면에서 유도가 회피 · 성공을 늘리는가, LIBERO (장애물 없음) 에서 해가 없는가 |
| **T90** | ATS target 신호 = VLA hidden patch probe (`--attention-source hidden_probe`) | **다음 단계 안 감.** 성공 5/9 대 7/9 (구분 못 함), target 정밀도 같음 (0.91), 사과 아닌 물체 선택 요청 92 → 96 | target 을 찾는 신호로 hidden probe 가 Panda · LIBERO 물체에서 명사 attention (L8H6) 보다 정확한가 |
| **T91c** | gate label 출처 = 손목 depth 손가락 사이 표면 단계 인식 (`--subtask-from-phase-machine`, T91 · T91b) | **latch 대비 이점 없음, 기본 꺼짐.** 재생 787 요청 중 gate 결정 변화 1, 그 run closed-loop 0 대 0 | 손목 카메라가 있는 LIBERO 계열에서 latch 보다 단계 오판이 줄고 gate 결정이 나아지는가 |

결과 · 근거: development checkout `outputs/2026-11-03_T92_…`, `_T93_…`, `_T90_…`, `_T91c_…` 의 README, handoff `T90–T93.task.md` §결과, `DECISIONS_20261103_KO.md` D14–D23.

## 2. Panda 로 옮길 때 꼭 확인할 것 (development 에서 확인된 차이)

1. **T93 은 Panda 에서 그대로는 유도가 돌지 않는다.** `fm_guide` 는 VLA 출력 → TrajOpt 관절 chunk 변환이 차원별 affine 이어야 기울기를 되돌린다 (요청마다 검사, 아니면 유도 안 하고 기록에 이유). RB-Y1 은 관절 delta 라 affine 이지만, **Panda 는 pi05_libero 가 EE (OSC) delta 를 내고 `action.delta_chunk_to_joint_chunk` (IK) 로 관절 chunk 를 만든다.** 그래서 Panda 용으로는 관절 공간 기울기를 EE delta 공간으로 되돌리는 사상 (IK Jacobian 전치 등) 을 adapter 쪽에 구현해야 한다. 구현 전 시험이면 "유도 0 요청" 으로 기록될 것이다 — 그것을 결과로 오인하지 않는다.
2. **T90 hidden probe 는 RB-Y1 checkpoint 의 hidden state 로 학습했다.** pi05_libero 에는 그대로 쓸 수 없다. T86 절차 (시뮬레이터 정답 target mask 로 patch probe 학습) 를 LIBERO 계열 기록으로 다시 해야 한다. 같은 날 사용자 판정 "Panda 표면 추출은 Panda 용 head · layer 사용" 과 같은 맥락이다.
3. **T91c 의 손가락 사이 ROI 와 문턱 (gap_tau 0.3, d 0.03–0.09 m) 은 RB-Y1 손목 카메라 · 그리퍼 기준이다.** Panda 손목 카메라 (eye-in-hand) 의 depth · 그리퍼 폭에 맞춰 ROI 를 다시 정한다. T91b SafeLIBERO offline 은 위험한 오판 4.3 % (기준 2 %) 로 미통과였고, 오판 21/24 는 VLA 가 다른 bowl 을 쥔 2 episode 였다 (`outputs/2026-11-03_T91b_…`).
4. **T92 는 로봇 독립이다** (측정 개도 · 실행 feedback 만). 다만 benchmark 의 지금 최종 구성은 HSR6 규칙 · `--hsr5` 계열을 쓰므로, `--subtask-from-latch` 는 `--hsr5` · `--hsr-gt-stage` · `--no-subtask-gate` 와 같이 줄 수 없다 (argparse 거절). 비교 구성을 짤 때 이 조합을 정리한다.
5. **접촉 허용 (D1, 사용자 확인 2026-11-03 18:2x KST):** benchmark profile 은 `--neighbour-contact-mm` · `--neighbour-palm-contact-mm` 와 평가의 이웃 과일 면제를 **모두 끈다.** 접촉 허용은 target (쥘 때) · 목적지 (놓을 때) 만 — MuJoCo 데모 profile 과 다른 점으로 명시한다. 이 두 허용이 RB-Y1 쥐기를 7 → 9 / 10 으로 올렸으므로 **LIBERO (장애물 없음) pilot 에서 끈 비용을 먼저 잰다.**
6. **최종 구성 갱신 (사용자 2026-11-03 18:2x KST): P0m 채택** — T92 (`--subtask-from-latch`) 가 benchmark 단계 출처의 기본이다 (`kv_L4_v1` 없음). hull 채우기 · 접촉 여유 10 mm 는 development T92b 결과에 따라 빠질 수 있다 (차이 ≤ 2 run 이면 뺌) — 결과가 나오면 알린다.

## 3. 시험 설계 (benchmark 세션이 정하고 결과 전에 고정)

- 세 시뮬레이션: **LIBERO** (장애물 없음 — 해 없음 확인), **SafeLIBERO** (Level I · II), **LIBERO-safety**.
- 비교는 짝지어: benchmark 의 현재 기준 구성 대 그 구성 + 시험 기능 하나. 같은 task · episode · seed.
- 주 지표: 성공 (각 벤치마크 정의) 과 금지 접촉 (충돌) 을 함께. HOLD · 정지 증가는 보호로 세지 않는다.
- 순서 제안: T92 (기본 후보, 가장 싸다) → T93 (Panda 사상 구현 뒤) → T91c (ROI 조정 뒤) → T90 (probe 재학습 뒤). 각 STEP 의 사전 기준 · 장면 · 규모는 benchmark 쪽 task 문서에 적고 결과 전에 고정한다.
- 결과는 benchmark checkout `outputs/<날짜>_<실험>/` 에 (영상 · 표 · README), 숫자는 verify json 으로.

## 4. development 쪽에서 아직 정해지지 않은 것 (사용자 판단 대기)

- T93b (유도를 쥔 뒤 구간에만) · B (T94 chunk 단위 CBF) 순서, 손바닥 충돌 구 모델 보완. 정해지면 같은 경로로 다시 알린다.
