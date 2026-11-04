# SL8b — 처음 고른 물체의 고정 풀기: 창(window) 투표 교체 + 누적 attention 해제 (기본 꺼짐 flag)

> writer: lead (benchmark checkout), 2026-11-04 15:2x KST. **결과를 보기 전에 고정.**
> 사용자 판정 (2026-11-04 15:1x KST):
> 1. "처음 고른 물체를 끝까지 고정하는 것을 풀어야 한다. 몇 프레임 이상 혹은 누적된 어텐션이 결국 다른 것을 본다면 타겟을 교체해야 한다."
> 2. development 세션이 방법론 몇 가지를 제안한다. 그것을 적용한 뒤 테스트를 진행한다.
>
> 앞 단계: SL8(`SL8.task.md`, 결과 `outputs/2026-11-04_SL8_ATS_target장애물분할/`).

## 1. 근거 (SL8 기록에서, 행동 attention 조건 A)

첫 call에 target이 아닌 물체로 고정된 episode는 10/20이다. 그 뒤 target으로 바뀐 episode는 0이다.

이 10 episode의 첫 닫힘 전 call 102개에서는 다음과 같았다.
- seed 다수가 target: 44 call
- target 다수 cluster가 존재: 13 call
- 그 cluster가 점수 1위: 11 call

현재 교체 규칙(`TargetConfirm.select`, `target_confirm_frames = 3`)은 "같은 challenger가 **연속** 3 frame 1위, 점수 ≥ 0.1"이다. 조건을 못 채우는 frame이 하나만 끼어도 count가 0이 된다.

| 경우 | 해당 episode | 무엇을 바꾸면 되나 |
|---|---|---|
| (a) target cluster가 1위인 frame이 있으나 다른 물체와 번갈아 나옴 | t1_e2 · t1_e4 · t2_e0 | 연속 대신 창 안 투표 수로 센다 |
| (b) seed는 target을 보는데 target이 fixture · obstacle과 한 cluster로 묶여 자기 cluster가 없음 | task 3 전부, t0_e1 | 누적 attention으로 고정을 푼다. 다만 옮겨 갈 target cluster가 없으므로, 묶임(SL8 문제 1)이 풀리기 전에는 "틀린 물체를 지우지 않음"까지가 기대 효과다 |

## 2. 변경 (implementer, 모두 기본 꺼짐)

### 2.1 창 투표 교체 — `--identity-switch-window W --identity-switch-votes M`

- config: `clustering.target_switch_window`(None = 끔), `clustering.target_switch_votes`(기본 3).
- 켜면 "연속 streak" 대신 최근 W번의 `select` 호출 중 **같은 challenger가 1위로 표를 얻은 호출 수**를 센다.
  - 같은 challenger: `target_identity_tolerance` 안의 위치.
  - 표를 얻는 조건은 지금 `counts`와 같다: 점수 ≥ `target_switch_min_score`, frozen 아님, 손 가림 아님, latched identity hold · subtask gate 아님.
- 그 수가 M 이상이면 교체한다. mode는 `switch`, reason은 `window_vote`.
- 표를 못 얻은 호출은 count를 0으로 만들지 않고, 창에서 밀려 나갈 뿐이다.
- 이번 시험값: **W = 5, M = 3**.

### 2.2 누적 attention 해제 — `--identity-attention-release K --identity-attention-off-frac F --identity-attention-radius-mm R`

- config: `clustering.identity_attention_release`(None = 끔), `identity_attention_off_frac`(기본 0.7), `identity_attention_radius`(기본 0.015 m).
- **쥐기 전에만 작동한다.** 무언가를 유지 중이고, frozen(closing)도 아니고, grasp latched도 아닌 `select` 호출에서만 센다.
- 각 호출에서 이번 frame seed 점의 attention 가중 비율을 잰다. 대상은 유지 중인 물체의 마지막 관측 형상(`last_geometry` 점)에서 R 안에 있는 seed다.
  - seed 정보는 `ground_target`이 `select` 전에 confirm에 넘긴다(새 `note_seeds`, 기본 꺼짐이면 호출만 하고 쓰지 않음).
- 최근 K번 호출의 "유지 물체 밖" 비율 평균이 F 이상이면 **고정을 푼다**. 이때 교체 후보는 점수 1위가 아니라 attention이 가리키는 cluster다:
  - 이번 frame seed의 attention 질량을 가장 많이 가진 admissible cluster가 그 질량의 50% 이상을 가지면 그 cluster로 교체한다. mode는 `switch`, reason은 `attention_release`.
  - 그런 cluster가 없으면 아무것도 유지하지 않는다. target 없음, reason은 `attention_release_none`. 그 뒤 새로 고를 때(`first`)도 같은 "seed 질량 50% 이상" 조건을 쓴다. 이 조건은 다음 교체 · 고정이 될 때까지 유지한다.
- 이번 시험값: **K = 4, F = 0.7, R = 15 mm**.

### 2.3 서버 · Panda 판

- `serve_safe` argparse에 위 flag를 넣는다. 기본은 꺼짐이고, 끈 판의 response · config는 그대로여야 한다.
- `AG3S/adapters/panda/profile.py`에 판을 등록한다. 셋 모두 `PC_LATCH_VERSIONS`에 넣는다.

| 판 | 정의 |
|---|---|
| `AG3S-final-panda-v3-e0to-pc-idwin` | pc판 + 2.1 |
| `AG3S-final-panda-v3-e0to-pc-idattn` | pc판 + 2.2 |
| `AG3S-final-panda-v3-e0to-pc-idboth` | pc판 + 2.1 + 2.2 |

### 2.4 테스트

- `TargetConfirm` 단위 테스트를 만든다:
  - 번갈아 1위인 challenger가 창 투표로 교체된다.
  - 끈 상태에서는 연속 규칙 그대로다.
  - seed가 유지 물체 밖에 K번 있으면 해제된다.
  - 쥐는 중에는 해제되지 않는다.
  - seed 질량 50% 미만이면 target이 없다.
- 회귀: AG3S core 테스트와 Panda 회귀 테스트(`tests/benchmark/`)가 통과해야 한다. 끈 판은 bit 동일이어야 한다.

## 3. 시험 1 — offline 재생 (SL8과 같은 20 episode, 같은 채점)

- 재생: `outputs/2026-11-04_SL8_ATS_target장애물분할/scripts/sl8_ats_replay.py --version <판>`
- 조건: 행동 attention(A)으로 세 판을 돌린다. oracle(C)은 `-idboth`만 돌린다.
- 비교 기준: SL8의 pc판 A · C.

### 판정 규칙 (결과 전 고정)

1. **회복:** 첫 고정이 틀린 episode 중, 첫 닫힘 전에 target으로 바뀐 episode 수를 센다. SL8 A에서는 0/10이었다.
2. **잘못된 교체:** 첫 고정이 맞은 episode 중, 첫 닫힘 전에 target이 아닌 물체로 바뀐 episode 수. **1 이하여야 한다.**
3. **첫 닫힘 전 call의 선택 정답률:** SL8 A의 0.33보다 낮으면 안 된다.
4. **obstacle 잘못 지움:** 0.5%를 넘으면 안 된다. SL8 A는 0.13%였다.
5. 2–4를 지키는 판 중 1(회복)이 가장 큰 판을 closed-loop 후보로 고른다. 같으면 변경이 적은 판, 곧 `-idwin`을 고른다.
   - 셋 다 1이 0이면 보고하고 멈춘다.

## 4. 시험 2 — closed loop (development 제안 적용 후)

- 사용자 판정 2에 따라, development 세션이 제안한 방법론을 같은 방식(기본 꺼짐 flag + 판 이름)으로 적용한 뒤 돌린다. 3.5에서 고른 판과 합친다.
- 장면은 SL7과 같다:
  - LIBERO 9 (libero_spatial task 0–4 · 6–9 × init 0)
  - SafeLIBERO 11 (VLA 단독이 공식 성공한 dev episode)
- 비교 대상: pc판(SL6-H · SL7g), 후보 `-pc-surface-fb`(SL7g · SL7h).
- 그 판정 규칙은 development 제안이 들어온 뒤, 실행 전에 이 문서에 덧붙여 고정한다.

## 5. 출력

- 원시 기록: storage `results/benchmark/<날짜_commit>_sl8b-*`
- 보기: `outputs/2026-11-04_SL8b_고정풀기_offline/`
