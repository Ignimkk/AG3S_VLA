# HSR · phase별 attention 적용 — 지도교수 보고 원고 (2026-10-13)

> 청중: 지도교수. 본문 8장 (6-2 포함) + 부록 5장. 각 슬라이드 = 제목 · 한 줄 메시지 · 그림 · 슬라이드 문구 · 대본(speaker notes).
> 그림 경로는 repo root (/mnt/dev/AG3S_VLA-development) 기준. 수치는 모두 verify.json 출처 (T59 · SUBTASK-b/d/e/f · T48 · T50).

---

## 표지
- 제목: VLA attention의 단계별 적용 — HSR (Hidden-state Sub-task Recognition)과 phase-aware target 관리
- 부제: AG3S 진행 보고 · 2026-10-13

---

## Slide 1. 문제 — attention이 가리키는 물체는 작업 단계에 따라 바뀐다

**한 줄 메시지:** 같은 attention이라도 "지금 무엇을 하는 중인가"를 모르면 들고 가는 중 · 놓은 뒤에 엉뚱한 물체를 target으로 잡는다.

**그림:** `AG3S/ag3s/docs/figures/ppt-1013/t59-ppt-timeline-ep1809-W2.png` (16D 실제 rollout 1개, frame 0–39)

**슬라이드 문구**
- 매 frame attention만으로 target을 고르면 (phase 미분할)
  - 들고 가는 중(held): 10 frame 중 8 frame에서 table 위의 다른 과일 선택 (또는 target 없음)
  - 놓고 해제한 뒤: crate 안에 이미 놓인 banana를 다시 target으로 선택
- 현재 시스템 (phase 분할): 같은 구간에서 쥔 banana 유지, 놓은 뒤 target 해제

**대본**
교수님, 먼저 문제를 실제 기록으로 보여 드리겠습니다. 이 그림은 16D 모델이 banana를 crate에 넣는 rollout 하나를 frame 단위로 펼친 것입니다. 맨 위 띠는 시뮬레이터 기준의 실제 상태로, 쥐기 전, 쥐고 있음, 놓음 순서입니다.
아래에서 두 번째 줄은 phase를 구분하지 않고 매 frame attention이 가장 강한 물체를 target으로 고른 결과입니다. 쥐고 들어 올리는 순간부터 target을 놓쳐서, held 10 frame 중 8 frame에서 table 위의 다른 과일을 고르거나 아무것도 고르지 못합니다. 놓은 뒤에도 crate 안에 이미 놓인 banana를 다시 target으로 잡습니다.
맨 아래 줄은 현재 시스템입니다. 쥔 동안에는 banana를 유지하고, 놓은 뒤에는 target을 비웁니다. attention 자체는 같고, 그것을 언제 믿고 언제 믿지 않을지를 phase가 정해 줘야 한다는 것이 이번 설계의 출발점입니다.

---

## Slide 2. 설계 — phase 신호 두 개와 phase별 attention 사용

**한 줄 메시지:** attention은 "무엇을 잡을지"에만 쓰고, "쥐었는지"는 grasp latch가, "작업이 어디까지 왔는지"는 HSR이 정한다.

**그림:** 블록도 (도형으로 직접 그린다) + 보조 `paper/figures/paperbanana/figure2/reference_montage_v2.png` 의 오른쪽 아래 C3 패널 (target → attached → released) 을 잘라 사용

블록도 내용:
```
3 camera + instruction → frozen π0.5 (prefix forward 1회)
   ├─ attention (L8H2) ──────→ target 후보 (3D grounding)
   └─ KV cache (L4 V) → HSR → pick / place / home
gripper 명령 · 개도 · 실행 feedback → grasp latch (SEARCHING→LATCHED→CLOSING→HELD→PLACED)
   ⇒ TargetConfirm: target 채택 · 유지 · 해제 → target-free ESDF / attached sphere
```

**슬라이드 문구 (표)**

| 단계 | latch | attention 사용 | 그 물체의 field 처리 |
|---|---|---|---|
| 집으러 감 (pick) | SEARCHING · LATCHED | attention 1위 cluster가 3 frame 연속이면 target 채택 · 교체 | target만 지운 target-free layer, 손가락만 질의 |
| 손을 닫는 중 | CLOSING | freeze: attention이 바뀌어도 target 고정 | 유지 |
| 들고 감 (place) | HELD | target 결정에 쓰지 않음, HSR이 새 target 채택을 막음 | 쥔 물체 = attached sphere (robot의 일부) |
| 놓은 뒤 (home) | PLACED | HSR home이면 놓인 물체 해제 | 다시 obstacle |

- 원칙: HSR label은 carving을 막을 수만 있고 추가할 수는 없다.

**대본**
설계는 신호를 세 가지로 나눴습니다. attention은 무엇을 잡을지에만 씁니다. 실제로 쥐었는지는 gripper 개도와 실행 feedback으로 판단하는 grasp latch가 정합니다. 작업이 집기, 옮기기, 복귀 중 어디에 있는지는 HSR이 정합니다.
표처럼 attention이 target을 정하는 것은 집으러 가는 단계뿐입니다. 손을 닫는 동안에는 target을 고정하고, 들고 가는 동안에는 쥔 물체를 robot의 일부로 붙여 충돌 검사를 합니다. 놓은 뒤 HSR이 home이라고 판단하면 그 물체를 다시 장애물로 돌려놓습니다.
핵심 원칙은 HSR이 지울 대상을 추가할 수는 없고 막을 수만 있다는 것입니다. HSR이 틀려도 장애물이 사라지는 방향으로는 틀리지 않도록 일부러 권한을 좁혔습니다.

---

## Slide 3. HSR 구현 — VLA를 재학습하지 않고 hidden state에서 subtask를 읽는다

**한 줄 메시지:** π0.5의 subtask text 생성은 우리 checkpoint에서 쓸 수 없어서, frozen KV cache 위에 linear probe 하나를 얹었다.

**그림:** 왼쪽 `AG3S/ag3s/docs/figures/subtask/subtask-inference-paths.png`, 오른쪽 `AG3S/ag3s/docs/figures/subtask-b/subtask-b-scene.png` (위쪽 줄만 잘라 사용 가능)

**슬라이드 문구**
- 시도 1: zero-shot text decode → 사용 불가
  - 들고 가는 36 frame 중 30 frame에서 영어 대신 FAST action token 출력
  - 우리 16D는 subtask text로 학습된 적이 없음
- 채택: HSR = KV cache (layer 4, V) [text 평균 | image 평균] 512차원 → logistic regression
  - 재학습 없음, attention 계산 경로에 추가 forward 없음 (같은 prefix pass의 KV cache 재사용), probe 추론 0.092 ms
- 각주: feature 9종 비교에서 val 정확도 0.974–0.981로 차이가 작았고, layer 4는 layer 17과 동률 (부록 A5)
- 출력이 N회 연속 같으면 확정 label (debounce)

**대본**
HSR은 Hidden-state Sub-task Recognition의 약자입니다. 원래 π0.5 논문은 subtask 문장을 먼저 생성하고 그 문장을 조건으로 action을 만듭니다. 그런데 우리가 fine-tuning한 16D checkpoint는 subtask text로 학습된 적이 없어서, 실제로 decode해 보니 들고 가는 구간 36 frame 중 30 frame에서 영어가 아니라 action token이 나왔습니다.
그래서 재학습 없이 모델 내부 표현에서 읽는 방법을 택했습니다. attention을 계산할 때 이미 만들어지는 KV cache에서 512차원 feature를 꺼내고, 그 위에 logistic regression 하나만 학습했습니다. attention 경로에 forward를 더하지 않고, probe 추론은 0.1 ms 미만입니다. layer는 9개 후보를 비교했는데 차이가 작아서, 특정 layer가 뚜렷이 우월하다고 주장하지는 않습니다. 오른쪽 그림처럼 실제 장면에서 pick, place, home 확률이 거의 1로 나옵니다.

---

## Slide 4. HSR 검증 — 잘 읽히지만, 무엇을 읽는지에는 한계가 있다

**한 줄 메시지:** phase는 높은 정확도로 읽히지만 instruction · 실제 쥠 상태에는 둔감하다 → 권한을 좁게 준 것이 맞았다.

**그림:** 오른쪽 `AG3S/ag3s/docs/figures/t48/t48b-score-accuracy-bars.png`

**슬라이드 문구 (표)**

| 항목 | 값 |
|---|---|
| HSR 정확도: val / closed loop | 0.981 / 0.959 |
| gripper 값을 가렸을 때 | 0.980 (state가 아니라 image에서 읽음) |
| 팔 관절만 쓴 control | 0.697 |
| instruction의 target 명사를 바꿨을 때 (T48a) | 반응 없음: place → pick 0.040 (변형 전 오류율과 같음) |
| 쥔 물체를 image에서 지웠을 때 (T48b, pilot) | 정답 pick 1 / 13 → 쥠 상태보다 팔 자세 순서를 읽을 가능성 |
| 원인이 fine-tuning인가 (T50) | 아님 (base 모델도 같은 기준 미달) |

- 주의: dataset phase는 scripted라 frame 번호만으로 1.000 → closed loop 0.959가 더 정직한 수치

**대본**
HSR 자체의 검증입니다. 정확도는 validation 0.98, 실제 closed loop 기록에서 0.96이고, gripper 값을 가려도 그대로라 image에서 읽고 있다는 것까지 확인했습니다.
하지만 입력을 일부러 바꿔 보는 counterfactual 실험에서 한계가 나왔습니다. instruction의 물체 이름을 바꿔도 반응하지 않았고, 쥔 물체를 image에서 지워도 대부분 여전히 place라고 답했습니다. "물체를 쥐었다"보다는 "팔이 이 자세에 있다"를 읽고 있을 가능성이 있습니다.
그래서 HSR에 attach나 target 선택 권한을 주지 않은 설계가 맞았다고 봅니다. 쥐었는지는 물리 신호인 grasp latch가 판단하고, HSR은 작업 단계를 알려 주는 보조 신호로만 씁니다.

---

## Slide 5. Closed-loop 검증 — HSR gate를 켜면 놓은 뒤의 잘못된 target이 사라진다

**한 줄 메시지:** 같은 episode·seed 48쌍에서 놓은 뒤 target 교체 21 run → 0 run. 성공률은 변화 없음.

**그림:** `AG3S/ag3s/docs/figures/subtask-d/subtask-d-scene-1800_s18001.png` (실제 씬 + ESDF 단면, 위 gate off / 아래 gate on)

**슬라이드 문구 (표)** — 16D, 24 episode × 2 seed = 48쌍

| 항목 | gate off (latch만) | gate on (latch + HSR) |
|---|---|---|
| 놓은 뒤 target 교체가 있었던 run | 21 | 0 |
| 놓은 뒤 carving이 켜진 chunk | 965 / 1,051 | 57 / 1,041 |
| 해제 후 정지(HOLD) chunk: 첫 구현 → 최종 | — | 984 → 28 |
| 마지막 팔 자세의 준비 자세 거리 p95 (rad) | 0.424 | 0.164 |
| success / grasp / place | 29 / 35 / 29 | 28 / 34 / 28 |

**대본**
실제 closed loop에서 HSR gate를 끈 경우와 켠 경우를 같은 episode, 같은 seed 48쌍으로 비교했습니다. 위쪽 gate off에서는 사과를 놓은 뒤에도 crate 밖 탁자 쪽 영역이 계속 지워져 있습니다. target이 다른 과일로 넘어갔기 때문입니다. 이런 교체가 25 run 중 21 run에서 일어났고, gate를 켜면 0이 됩니다.
처음 구현에서는 target을 비우자 시스템이 인증되지 않은 상태로 보고 robot을 멈추는 문제가 있었습니다. 두 번 수정해서 해제 후 정지 chunk를 984에서 28로 줄였고, robot이 준비 자세로 돌아오게 했습니다.
다만 성공률은 29에서 28로 차이가 없습니다. 이 결과는 성공률 향상이 아니라 "놓은 뒤 잘못된 carving 제거"로 말씀드리는 것이 정확합니다.

---

## Slide 6. 16D 정량 — phase를 나누지 않으면 무엇을 고르는가 (T59)

**한 줄 메시지:** 16D에서 phase 미분할 grounding은 들고 가는 동안 target을 0.59밖에 유지하지 못하고, 놓은 뒤에는 이미 놓인 물체를 다시 고른다.

**그림:** 왼쪽 `AG3S/ag3s/docs/figures/ppt-1013/t59-ppt-phase-bars.png`, 오른쪽 `AG3S/ag3s/docs/figures/ppt-1013/t59-ppt-scene-ep1809.png`

**슬라이드 문구 (표)** — T53 16D 16 episode, offline, 같은 frame에서 비교 (target 비율, episode 평균 [95% CI])

| 단계 | phase 미분할 | phase 분할 (serving) |
|---|---|---|
| 쥐기 전 (477 frame) | 0.82 [0.69, 0.93] | 0.89 [0.75, 1.00] |
| 들고 가는 중 (106 frame) | 0.59 [0.39, 0.76], 다른 물체 0.35 | 0.86 [0.63, 1.00] |

- 놓고 해제된 119 frame (HSR home)에서 phase 미분할이 고른 것: 이미 놓인 물체 81 · 다른 물체 7 · table 5 · crate 1 · 없음 25
- 들고 가는 동안 attention 자체는 16D에서도 crate에 몰림 (14D와 같음). 다만 crate는 크기 규칙(admissibility)에서 빠져 선택되지 않고 (held에서 crate 선택 0 / 106), 남은 약한 attention 중 table 위 다른 과일이 선택됨 → 다음 슬라이드

**대본**
앞의 예시가 한 episode였다면, 이 슬라이드는 16D 모델 rollout 16개 전체입니다. 새로 rollout하지 않고 기존 기록에서 frame마다 두 방식을 같은 시점에 비교했습니다.
쥐기 전에는 두 방식이 비슷합니다. 차이는 들고 가는 동안 생깁니다. phase를 나누지 않으면 target 유지 비율이 0.59로 떨어지고 35%는 옆의 다른 과일을 고릅니다. 현재 시스템은 같은 frame에서 0.86을 유지합니다.
놓은 뒤가 더 분명합니다. 현재 시스템이 target을 해제한 119 frame에서, phase를 나누지 않았다면 81 frame에서 이미 놓은 물체를 다시 target으로 잡았을 것입니다. 오른쪽 가운데 장면이 바로 그 순간입니다.
왜 이런 일이 생기는지는 다음 슬라이드에서 말씀드리겠습니다.

---

## Slide 6-2. 원인 분석 — 들고 가는 동안 policy attention은 crate를 본다

**한 줄 메시지:** 실패의 시작은 attention 단계다. 들고 가는 동안 L8H2 attention은 target이 아니라 crate에 몰리고, crate는 크기 규칙에서 빠지므로 남은 약한 attention끼리 경쟁하다 다른 과일이 선택된다.

**그림:** `AG3S/ag3s/docs/figures/t59/t59-diag-attention-stage.png` ((a) 2D attention 질량, (b) 3D seed 구성, (c) 처음 실패한 단계)

**슬라이드 문구 (표)** — T53 16D 16 episode, offline

| 지표 | 쥐기 전 | 들고 가는 중 |
|---|---|---|
| 2D attention 중 target 몫 | 0.106 | 0.021 |
| 2D attention 중 crate 몫 | 0.057 | 0.473 |
| 3D seed 중 crate 비율 | 0.42 | 0.92 |

- 들고 가는 중 phase 미분할이 틀린 37 frame의 처음 실패 단계: target이 cloud에서 사라짐 0 · target에 seed 없음 17 · 다른 cluster에 병합 2 · 후보는 있으나 다른 과일이 attention 순위에서 이김 18
- 가림 · self-filter 때문이 아님 (106 frame 모두 target 점이 관측됨)
- 놓은 뒤: phase 미분할은 frame마다 처음부터 고르므로 "이미 놓았다"는 정보가 없음. crate 안 과일은 잡을 수 있는 크기라 후보로 남고, serving이 그것을 다시 고르지 않은 것은 HSR gate 때문 (81 / 81 frame이 gate로 막힘)
- 확인 안 된 것: policy가 왜 들고 가는 동안 crate를 보는지 (가설: 목적지를 본다), 2D에서 거의 없는 다른 과일 attention이 3D seed를 받는 경로 (가설: 16×16 patch lifting 번짐)

**대본**
원인을 단계별로 분해했습니다. 들고 가는 동안 L8H2 attention의 절반 가까이가 crate에 있고, target에는 2%밖에 없습니다. 3D seed로 옮기면 92%가 crate입니다. 예전 14D에서 보았던 "들고 가는 동안 목적지를 본다"는 현상이 16D에서도 같다는 뜻입니다.
다만 지금은 crate가 크기 규칙에서 빠지기 때문에 target으로 선택되지는 않습니다. 대신 남은 약한 attention끼리 경쟁해서, table 위의 다른 과일이 이기거나 target에 seed가 하나도 남지 않습니다. 손에 가려서 target이 사라진 경우는 한 번도 없었습니다.
놓은 뒤의 재선택은 구조적인 이유입니다. 매 frame 새로 고르는 방식에는 "이미 놓았다"는 정보가 없고, crate 안의 과일은 잡을 수 있는 크기라 후보로 남습니다. 현재 시스템에서는 HSR gate가 81 frame 모두 이를 막았습니다.
policy가 왜 들고 가는 동안 crate를 보는지는 아직 측정하지 못했습니다. 목적지를 본다는 해석은 가설로 남겨 두겠습니다.

---

## Slide 7. 정리 · 한계 · 논의드릴 점

**정리**
- attention은 pick 단계에서만 target을 정하고, 이후 단계는 grasp latch와 HSR이 물체의 역할을 바꾼다
- HSR: 재학습 없이, 추가 비용 0.1 ms 미만으로 phase를 0.96 정확도로 읽음
- 효과: 놓은 뒤 target 교체 21 → 0 run, 들고 가는 동안 target 유지 0.59 → 0.86 (offline)

**한계 (측정하지 않았거나 아직 나쁜 것)**
- 성공률 향상 없음 (29 대 28)
- HSR은 instruction · 쥠 상태에 둔감 → 보조 구성 요소로 둠
- strong rule(grasp latch만)과 phase source 맞비교는 진행 중 (T55)
- 16D 정량(T59)은 offline. phase 미분할 closed loop는 없음

**논의드릴 점**
1. HSR을 논문에서 "보조 구성 요소"로 두는 범위가 적절한지
2. phase 미분할 arm을 closed loop로 확인할 필요가 있는지
3. 크기로 구분되지 않는 목적지(bowl · plate)에서 gate를 검증할지

**대본**
정리하면, attention은 무엇을 잡을지만 정하고, 쥔 뒤와 놓은 뒤에는 물리 신호와 HSR이 물체의 역할을 바꾸는 구조입니다. 그 결과 놓은 뒤의 잘못된 target 교체가 사라졌고, 16D 기록에서 들고 가는 동안 target 유지율이 0.59에서 0.86으로 높았습니다.
한계도 분명합니다. 성공률은 오르지 않았고, HSR은 instruction이나 실제 쥠 상태를 잘 구분하지 못합니다. 그래서 HSR은 핵심 기여가 아니라 보조 구성 요소로 두고 있고, grasp latch만 쓰는 방식과의 직접 비교는 현재 T55에서 진행 중입니다.
세 가지에 대해 교수님 의견을 듣고 싶습니다. HSR을 논문에서 어느 범위로 주장할지, phase 미분할을 closed loop로 확인할지, 크기로 구분되지 않는 목적지에서 검증이 필요한지입니다.

---

## 부록 A1. 이전 모델(14D)에서의 문제
- 그림: `AG3S/ag3s/docs/figures/ppt-1013/attention-latch-crop-f0-20.png`
- 문구: 들고 가는 frame 7–20 동안 grounding이 crate 선택 → latch 도입 계기 (14D 시기 기록)
- 대본: 이전 14D 모델에서는 들고 가는 동안 attention이 목적지인 crate를 가리켰고, 그대로 target으로 쓰면 crate가 지워졌습니다. 이것이 grasp latch를 도입한 계기였습니다.

## 부록 A2. 손을 닫는 중의 문제 (16D, T20)
- 그림: `AG3S/ag3s/docs/figures/t20/t20-t14-topview.png`
- 문구: 사과가 손에 가려진 frame에서 1위가 crate · pear로 바뀜 (점수 0.015–0.047) → freeze 근거. 당시 self-filter가 사과 점을 지운 것도 함께 작용
- 대본: 손을 닫는 동안에는 사과가 손가락에 가려 attention 1위가 crate나 배로 바뀌었습니다. 점수는 매우 낮았지만 이전 규칙은 target을 바꿨고, 그래서 손을 닫는 동안 target을 고정하는 freeze를 넣었습니다.

## 부록 A3. 측정 정의의 변경 (투명성)
- 문구: serving 쪽 class의 원래 정의는 "같은 frame의 GT 위치". 쥔 물체는 attached로 처리되어 기록된 centroid가 마지막 관측 위치에 머물러 held target이 0.14로 왜곡됨 → 결과를 본 뒤 "centroid를 관측한 frame의 GT"(0.86)로 주 정의 변경 (사용자 판정), 두 값 모두 기록
- 대본: 측정 정의를 결과를 본 뒤 한 번 바꿨기 때문에 말씀드립니다. 원래 정의로는 쥔 물체의 위치가 갱신되지 않아 값이 왜곡되어서, 관측 시점 기준으로 바꿨고 두 값을 모두 남겼습니다.

## 부록 A4. 놓은 뒤 구간(W2)의 쏠림
- 그림: `AG3S/ag3s/docs/figures/ppt-1013/t59-ppt-phase-bars-W2.png`
- 문구: placed 153 frame 중 88 frame은 준비 자세로 복귀하지 못한 2 episode에서 나옴 → 주 수치는 place 직후 종료(W1) 기준
- 대본: 놓은 뒤 구간은 복귀하지 못한 두 episode가 절반 이상을 차지해서, 본문의 주 수치는 place 직후에서 자른 기준으로 드렸습니다.

## 부록 A5. HSR feature 선택 근거
- 문구 (표): 9개 feature를 같은 logistic regression으로 비교 (선택 기준 = val 정확도 최대, 동률은 목록 순서)

| feature | val | closed loop (246 frame) |
|---|---|---|
| prefix_last | 0.9765 | 0.9309 |
| prefix_img | 0.9804 | 0.9553 |
| kv_L0 | 0.9800 | 0.9593 |
| kv_L4 (채택) | 0.9808 | 0.9593 |
| kv_L8 | 0.9781 | 0.9553 |
| kv_L12 | 0.9796 | 0.9634 |
| kv_L17 | 0.9808 | 0.9593 |
| ae_final | 0.9800 | 0.9634 |
| attn_L8 | 0.9742 | 0.9553 |

- linear probe를 쓴 근거: action expert의 입력은 prefix KV · noise · time뿐인데 action이 phase마다 다르므로 phase 정보는 KV 안에 있어야 한다는 논리 (문헌 인용 없음). K, 나머지 layer, 다른 pooling, nonlinear probe는 비교하지 않음
- 대본: layer 4는 layer 17과 val 정확도가 같았고 목록 순서로 정해졌습니다. closed loop에서는 layer 12나 action expert 마지막 hidden이 한 frame 더 맞혔습니다. 즉 특정 layer가 우월해서가 아니라, 대부분의 feature에서 phase가 선형으로 읽힌다는 것이 정확한 해석입니다.

## 예상 질문 (마지막 슬라이드 또는 notes)
- Q. HSR 없이 grasp latch만으로 충분하지 않은가? → 손을 닫는 중 · 들고 가는 중은 latch로 해결. 하지만 latch는 "작업이 끝났다"를 몰라 놓은 뒤 target 교체(21 run)를 막지 못함 → HSR 도입 이유. 직접 비교는 T55 진행 중.
- Q. 성공률이 안 오르는데 왜 필요한가? → 이 과제에서는 놓은 뒤 잘못 지워진 물체에 닿는 상황이 드묾. 효과는 놓은 뒤 장애물 정보를 올바르게 유지하는 것. 연속 작업 · 다른 목적지 영향은 미측정.
- Q. 0.59 대 0.86은 유의한가? → episode bootstrap 95% CI [0.39, 0.76] · [0.63, 1.00] 일부 겹침. held가 있는 episode 10개라 방향은 일관되나 강한 통계 주장은 하지 않음.
