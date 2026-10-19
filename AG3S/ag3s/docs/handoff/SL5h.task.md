# SL5h — over-lift · action 포화의 출처 진단 (CPU 부분)

> writer: lead · 읽는 쪽: verifier · 작성 2026-10-19 KST
> code: `8e7ce1f8`. SL5g docs commit이고, 실행 code는 `a8ad92ef`와 같다. 이 단계는 **CPU만** 쓴다.
> GPU counterfactual(#G)은 development checkout과 일정을 합의한 뒤 lead가 따로 지시한다.

## 배경 (SL5g, `handoff/SL5g.verify.json`)

VLA-only가 성공했는데 pb가 실패한 episode는 31건이다. 가장 큰 유형은 **C_lifted_overlift 11건**이다.

| 지표 | pb4 | pb5 | 같은 episode의 VLA-only |
|---|---|---|---|
| LI lift 구간에서 dz ≥ 0.999 step | 452/999 | 398/916 | 0 |

lead가 대표 사례 LI t2e4 pb5를 직접 확인한 내용:

- **action 포화:** 실행 action이 obs 170 이후 x · y · z · rotation 여러 축에서 동시에 ±1로 포화된다.
- **상승:** eef z가 1.496 m까지 오른다. VLA-only는 최고 1.185 m다.
- **latch가 held로 넘어가지 않는다:** 끝까지 `closing`이다. 그 이유는 다음과 같다.
  - call 30의 `grasp.evidence`에 `reach_mm 123.46 > reach_limit_mm 120`이 기록돼 있다.
  - `checks.reach = false`이고, `manipulated_state = occluded`다.
  - 그래서 note가 "파지 미확인: reach"다.
- **역변환 C 포화:** 같은 call의 `conversion`은 `saturated_rows [0, 1]`, `max_abs_correction 1.39`다.

## 기록된 data

- 각 run의 `episodes/*_calls.npz`
  - `actions_policy`: (calls, 10, 7). VLA 원본 chunk
  - `joint_reference` · `joint_refined`: (calls, 10, 8). TO 전 / 후 joint
  - `actions_response`: 역변환을 거쳐 client에 돌려준 action
  - `actions_executed`: (calls, 5, 7)
  - `predicted_ee_base`
- `*_calls.jsonl`
  - `conversion`: `corrected_rows`, `saturated_rows`, `max_abs_correction`, `realisation`
  - `to`: status, MI gate, contact_phase, grasp_contact
  - `grasp`: state, transition, evidence와 checks, reach_mm / reach_limit_mm, manipulated_state
  - `feedback_server`
- episode npz(`action`, `obs_robot0_eef_pos` …)와 MuJoCo 상태 (`*_geom.npz`, `sim_state_at_return`)

## 무엇을 재나 (→ verifier, Opus)

대상 run:

- **수정본:** SL5g pb4-C · pb5-C (LI, LIBERO 10)
- **비교:** SL5e pb3-C, SL5d pb2-C, SL5c V11AB-A(변환 A), VLA-only (SL5f)

| # | 측정 | 내용 |
|---|---|---|
| D1 | 포화 분해 | call / row마다 다음 단계 중 어디에서 처음 \|a\| ≥ 0.999가 되는지 센다. ① `actions_policy`(VLA) → ② TO 수정(`joint_refined − joint_reference`, rad, EE 변화 mm) → ③ 역변환 C의 보정(`max_abs_correction`, `saturated_rows`) → ④ `actions_response` → ⑤ `actions_executed`. MI gate가 적용된 call(refined = reference)과 아닌 call을 나눠서 센다. "TO가 바꾸지 않았는데도 역변환만으로 포화"되는 경우를 따로 센다 |
| D2 | 역변환 C 동역학 | 포화가 이어지는 구간에서 다음을 call 순서로 낸다: 목표 joint 대비 실제 joint의 lag(누적 오차), correction 크기, realised_fraction. 오차가 커지는데 보정이 같은 방향으로 계속 쌓이는지(windup 모양) 본다. 정의와 threshold를 적는다 |
| D3 | latch · reach | 실제로 들고 있는 obs에서 latch state 분포를 낸다. 실제 lift 기준은 SL5g #3 정의(두 손가락 접촉 + rise ≥ 30 mm)다. 같은 call의 `evidence.checks`에서 실패한 check, `reach_mm`과 `reach_limit_mm`의 분포, `manipulated_state` 분포도 낸다. reach_mm이 limit를 넘는 시점과 실제 lift 시점의 순서를 본다 |
| D4 | closing 중 쥔 물체의 제약 | latch가 `closing`이고 실제로 lift 중인 call에서 다음을 센다: 최적화된 행 중 쥔 bowl에 대한 행의 비율, 그 행들이 민 방향(EE 변화의 z 성분 부호). TO가 쥔 bowl을 obstacle로 보고 hand를 위로 미는지 확인한다 |
| D5 | version · 변환별 서명 | 같은 서명을 version별로 센다: over-lift(최고 rise > 250 mm), lift 구간 포화 step 비율, `checks.reach = false`인 held-상태 call. 대상은 pb2-C · pb3-C · pb4-C · pb5-C, V11AB-A(SL5c), VLA-only다. pb2 이후에만 나타나는지(pb1 = reach-at-capture, 변환 C 도입 시점)를 본다 |
| D6 | pb5 성공 episode와의 대조 | LIBERO t0e0 · t1e0 · t2e0 · t3e0 · t6e0의 pb5 성공 episode에서 같은 D1 · D3 숫자를 잰다. 왜 거기서는 포화나 latch 정체가 없었는지 본다 |

- **판정 문장을 쓰지 않는다.** 가설마다 지지 · 반박하는 숫자만 둔다. 가설:
  - H1: 포화는 역변환 C의 windup에서 생긴다
  - H2: 포화는 TO 수정에서 생긴다
  - H3: latch가 reach check에서 막혀 held로 못 가고, 쥔 bowl이 제약으로 남아 hand를 밀어 올린다
  - H4: VLA reference 자체가 OOD 상태에서 포화한다
- figure 3종: `AG3S/ag3s/docs/figures/sl5h-*` + `.json`
  1. 실제 scene: LI t2e4 pb5의 lift 직후 · 포화 시작 · 최고점 frame. 쥔 bowl, 제약 행, reach 거리를 표시한다
  2. graph: call별 ①~⑤ 크기, reach_mm vs limit, latch timeline
  3. table: D1 분해 집계, D5 version별 서명
- 산출물:
  - raw: `outputs/verify/SL5h/`
  - verify.json: `AG3S/ag3s/docs/handoff/SL5h.verify.json`

## #G GPU counterfactual — 이 단계에서 실행하지 않는다

development와 합의한 뒤 lead가 실행한다. 후보는 다음과 같다.

- pb5-B · pb5-A의 LI dev + LIBERO 10. H1 확인용이다.
- 같은 code의 pb5-C 일부를 병렬로 다시 돌려 SL5g와 bit 단위로 비교한다. 병렬 실행이 결과를 바꾸지 않는지 확인하는 것이다.
- verifier는 D1~D6 결과로 보아 #G 중 어느 것이 가설 구분에 필요한지 숫자로 제안한다.

## 실행 규칙

- code 수정 금지, commit 금지. 측정 script만 `outputs/verify/SL5h/scripts/`에 둔다.
- `.venv-ag3s/bin/python`을 쓰고, `CUDA_VISIBLE_DEVICES=`로 CPU만 쓴다.
- **CPU worker ≤ 2. 메모리 주의:** pod working set이 이미 약 59 GiB다(한도 64). npz는 하나씩 열고 바로 닫는다. 큰 배열은 필요한 key만 읽는다. RSS ≤ 3 GB를 지킨다. 시작 전과 중간에 `memory.current − inactive_file`을 확인해, 62 GiB를 넘으면 멈추고 보고한다.
- 0 byte log나 D state가 보이면 NFS stall일 수 있다. 같은 파일을 반복해서 stat하지 말고 보고한다.

## 이 STEP이 답하는 질문

pb4 · pb5의 over-lift와 action 포화는 VLA, TO, 역변환 C, latch(reach check) 중 어디에서 시작되는가? 그것은 pb2 이후 변경과 함께 나타났는가?
