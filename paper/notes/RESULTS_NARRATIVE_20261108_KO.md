# 결과 표를 어떻게 설명할 것인가 — Tab. 1–4 · Fig. 4의 서술 논리

- 작성: 2026-11-08 KST. 대상: `paper/latex/exp_section/04_experiments.tex` v4의 표와 본문 수치. **모든 "우리 run" 수치는 투영값(빨강)** 이며, 이 문서는 수치가 아니라 **읽는 순서와 각 비교가 타당한 이유**를 고정한다. 결과가 바뀌면 숫자만 바꾸고 논리는 유지한다(바뀌어야 하는 경우는 §8).
- 지표 관계(모든 설명의 바탕): 분모는 전부 전체 episode. **SSR ≤ min(TSR, CAR)**. **TSR − SSR = 충돌하면서 성공한 비율**(정책 단독의 문제). **CAR − SSR = 안 부딪혔지만 실패한 비율**(안전 레이어의 비용). 이 두 차이를 읽으면 표의 모든 행이 설명된다.

## 1. Tab. 1 — LIBERO + SafeLIBERO: 여섯 단계로 읽는다

| 단계 | 보는 곳 | 말하는 것 | 왜 타당한가 | 리뷰어 반문 → 답 |
|---|---|---|---|---|
| ① 재현 검증 | 상단 † π0.5(VLSA 57.8 / 17.1; KNOWS 55.0 / 12.4) vs 우리 π0.5 행(57.8 / 18.6) | 우리 harness가 저자들의 환경과 같은 결과를 낸다 | 같은 checkpoint를 두 논문이 독립적으로 보고했고 우리 값이 그 사이에 있다. 이후 모든 비교의 신뢰 근거 | "evaluator가 다르지 않나" → 공식 evaluator · horizon · 1 mm 규칙 사용, Tab. S2에 suite별 |
| ② 문제 정의 | π0.5 행의 TSR − SSR = 44.3 pt | 정책 단독은 성공의 3/4을 장애물을 밀고 지나가며 얻는다 | 정책은 장애물을 표현하지 않으므로 시연된 동작을 그대로 실행한다. 이것이 1장의 문제 정의와 같은 수치 | "LIBERO 정책이 원래 약한 것 아닌가" → LIBERO 열 96.8: 장애물이 없으면 거의 다 성공 |
| ③ 제약 출처 사다리 | generic → external → AG3S (같은 TO) | 무엇이 성능을 만드는지 귀속 | 세 행은 optimizer · 접촉 정책 · episode가 같고 **제약의 출처만** 다르다. generic은 target까지 금지해 TSR 42 (파지 차단), external은 이웃을 같이 풀어 CAR 71 (leakage), AG3S만 둘 다 잡는다 | "왜 segmenter baseline이 약한가, 튜닝 부족 아닌가" → prompt 두 조건 · Tab. 4 precision으로 원인이 "닿은 이웃 병합"임을 보임 |
| ④ 공개 layer 비교 | AEGIS(공식 코드) · multi-link* · KNOWS* vs AG3S | 충돌률 절반(24.0 → 12.5), SSR 최고(61.5 vs 55.0), TSR 동급(66.0 vs 최고 67.5) | 같은 episode에서 paired. CBF 계열은 이름 붙은 장애물 하나를 ellipsoid로 피하므로 이름 없는 clutter · 운반 물체 충돌이 남는다(잔여 충돌의 61 %). KNOWS는 extent를 segmenter에서 가져와 ③의 leakage를 물려받는다 | "AEGIS 재구현이 약한 것" → 공식 코드 실행, 보고값(67.5 / 68.9)과 나란히. "KNOWS*는 재구현" → 보고 76.4 vs 재현 76.0 CAR이 표 안에서 일치 |
| ⑤ 일반화 | Level II 열; KNOWS 보고 59.9 → 53.8 | 장애물 배치가 바뀌면 segmenter · 학습 신호 기반은 떨어지고 우리는 오른다 | 우리 제약은 매 요청 관측에서 읽으므로 배치에 맞춰 튜닝된 것이 없다. KNOWS 저자 보고값에서도 같은 하락이 보인다 | "Level II가 더 쉬운 것 아닌가" → π0.5 단독도 Level II에서 TSR 59.4로 비슷; 차이는 SSR에서만 |
| ⑥ 비용 | LIBERO 열 95.4 vs 96.8; CAR − SSR = 26.0 | 안전 레이어의 비용은 hold다 | 장애물이 없으면 1.4 pt, 있으면 무충돌 실패 26 pt 중 79 %가 hold timeout. 이것이 5장 action 축 future work의 근거 | "trade-off를 숨긴 것 아닌가" → 분해를 본문에 그대로 씀 |

**"안전 레이어가 TSR을 올린다"는 역설의 설명.** 모든 layer(보고값 포함)가 π0.5보다 TSR이 높다. 장애물을 밀면 target이 넘어지거나 배치가 막혀 task도 실패하기 때문이다. 이 문장을 본문에 한 번 써 두면 "왜 우리 TSR이 π0.5보다 높지?"라는 의아함이 사라진다. 반대로 LIBERO 무장애물 열에서는 밀 장애물이 없으므로 hold 비용만 남아 1.4 pt 낮다 — 두 열의 방향이 다른 것이 오히려 논리의 증거다.

**Naive(GT segmentation) 행의 역할.** KNOWS 논문이 보고한 "정답 segmentation + ellipsoid"는 특권 정보이므로 우리는 이를 넘는다고 쓰지 않는다. "정답 segmentation을 받은 필터와 2 pt 차이"로 쓴다. 특권 행을 넘었다고 쓰는 순간 "그럼 정답을 쓰면 더 좋지 않나"라는 반문이 성립해 버린다.

## 2. Tab. 2 — LIBERO-Safety L2: 다른 분류 체계에서 같은 메커니즘

| 열 | 뜻 | 우리 논리 |
|---|---|---|
| AAG | 주변 물체를 건드리지 않고 **target을 잡는** 능력 | 허용 접촉(target)과 금지 접촉(주변)을 가르는 것이 우리 contact policy 그 자체 → AG3S 상승(32.7 → 37.3), generic은 파지를 금지해 급락(19.3). **Tab. 1 사다리 ③의 "파지 차단" 현상을 다른 벤치마크가 독립적으로 재현**한다는 것이 핵심 문장 |
| TSA · FSHOA | 탁상 clutter · 사람 손–물체 회피 | 순수 장애물 회피 → AG3S ≥ generic ≥ external > π0.5. 손–물체처럼 **이름 붙이기 어려운 장애물**도 관측 표면이므로 보호된다(D·G 논거) |
| CR | 충돌률 | 26.0 → 6.7. generic(8.7)보다도 낮은 이유는 hold가 장애물 근처에서 timeout으로 끝나는 비율이 낮아서 |

상단 † 블록은 3 training seed의 ±가 있는 저자 보고값이고 우리 π0.5는 단일 공개 weight라 그보다 낮다(35.3 → 32.7). 이 차이를 표 안에서 먼저 인정해야 "우리 환경이 유리한 것 아닌가"가 안 나온다. SR 정의(안전 위반 없이 완료)가 SafeLIBERO의 SSR에 해당한다는 점을 한 번 짚는다.

## 3. Fig. 4 — 실로봇: feasibility, 개선 주장이 아니다

- 읽는 순서: π0.5 단독의 unsafe contact 45 %(운반 물체 끌기, 파지 중 이웃 누르기) → AG3S 8.8 %, task 68.8 vs 71.3 → unseen 물체에서도 같은 패턴(62.5 / 10.0).
- 타당성의 근거는 수치가 아니라 **판정 방식**이다: F/T 센서 + 외부 영상 + 방법을 모르는 annotator 2명. AG3S 자체 추정으로 판정하지 않는다.
- 쓰지 말 것: 20 trial/task는 ±20 pt CI라 task별 비교 문장은 안 된다. 80 trial pooled만 비교하고 task별은 그림으로만.

## 4. Tab. 3 — Ablation: 효과 크기의 순서가 이야기다

읽는 순서는 **정책 → extent → identity**. (i) target을 항상 장애물로 두면 CAR 최고(93.0)지만 TSR −30 pt → "파지 허용이 없으면 안전 레이어는 작업을 못 한다." (ii) 모든 link · 모든 단계에 풀면 TSR은 돌아오지만 CAR 77.0 → "손목과 운반 물체가 target 주변으로 들어간다." (iii) 운반 물체를 붙이지 않으면 −6.9 → 전부 운반 단계 충돌. 여기까지가 **contact policy 세 항이 모두 필요**하다는 증명. (iv) extent: graph → external mask −10.3, → radius growth −7.2, 둘 다 CAR로 잃는다 → "freed region이 닿은 장애물을 삼킨다", 그리고 external mask가 가장 크다는 것이 **segmenter로는 못 고치는 문제**라는 근거. (v) identity: nearest object −6.7(Object suite 집중), memory off −4.1(파지 순간 가림 집중) → attention과 memory가 각각 다른 상황을 담당.

왜 타당한가: 변형마다 **하나만** 바꾸고 optimizer는 고정, 같은 320 episode paired, CI 제시. CI가 0을 포함하는 변형은 "필요하다"고 쓰지 않는다(현재 link-specific 항이 가장 위험). 효과 크기 순서(정책 ≫ external mask > radius ≈ carried ≈ nearest ≈ all-links > memory)가 결과에서 바뀌면 4.3 세 문단의 순서를 결과에 맞춘다.

## 5. Tab. 4 — Grounding: precision이 이야기다

- 핵심 문장: "recall은 같고 precision이 다르다(0.90 vs 0.55)." segmenter는 target에 **도달**하지만 닿은 이웃까지 **삼킨다**. 우리는 depth 불연속에서 멈춘다. 이것이 Tab. 1 ③·④의 leakage를 **원인 수준에서** 설명하는 표다.
- prompt 두 행(지시문 noun 0.48 / 정답 이름 0.61): segmenter는 prompt 품질에 종속이고, attention은 정책이 실제로 실행하는 지시문에 조건화 → B 논거.
- oracle 0.80과의 차이 0.08: "선택이 아니라 관측의 한계"(뒷면 · depth hole) → 4.4 perception 축, 5장 future work로 연결.
- identity: memory가 IoU는 거의 안 바꾸고(0.69 → 0.72) switch를 2.8 → 0.4로 줄인다 → "extent는 frame 단위 결정, identity는 시간 단위 결정"이라는 방법론 3.3의 설계 주장을 수치로 뒷받침.
- 주의: 절대값은 장면 구성에 크게 좌우된다(11-01 dev에서는 0.36–0.41 vs SAM-2 0.21). 비교는 **같은 held-out 장면 · 같은 GT · 같은 support**에서만 말한다.

## 6. 4.4 본문 수치 — 두 축과 비용

- 실패 두 축: 충돌의 71 %는 perception(미관측 46 · identity 17 · sphere 밖 8), 무충돌 실패의 93 %는 action(hold 79 · pose 흐트러짐 14). 이 분해가 5장 두 future work(표현–기하 동기화 / 생성 단계 제약)의 유일한 수치 근거다. 비율이 바뀌어도 "두 축" 문장은 유지되고, 한 축이 10 % 미만이면 그 축의 future work 비중을 줄인다.
- 일반화: held-out에서 AG3S −1.8 vs KNOWS* −7.5 vs external −9.1. 논리는 "학습된 요소가 frozen policy뿐이라 노출되는 shift는 perceptual"이다.
- run time: 측정 결과가 baseline보다 나쁘면 문장을 빼고 "loop 안에 두 번째 perception 모델이 없다"(6 ms read-out)만 남긴다(사용자 판정 2026-11-08).

## 7. 한 문단 요약 (초록 · 결론용 영문 틀; 수치는 결과로 교체)

> On SafeLIBERO with the same frozen π0.5, AG3S-VLA halves the collisions of the strongest safety layer (12.5 % vs. 24.0 % of episodes) and gives the highest safe-success rate (61.5 %), with task success on par with the best layer and 1.4 points below the policy alone on obstacle-free LIBERO. Ablations attribute the gain to the depth-bounded target surface and the stage- and link-conditioned contact policy, and the grounding study shows why: against a touching neighbor an external mask reaches the target but swallows the neighbor (precision 0.55 vs. 0.90), while the policy's own attention bounded by depth continuity does not.

## 8. 결과가 투영과 다를 때 논리가 바뀌는 경우

| 결과 | 바뀌는 논리 | 원고 수정 |
|---|---|---|
| AG3S TSR이 최강 layer보다 3 pt 넘게 낮음 | "TSR 동급" 삭제, trade-off를 전면에 | 4.2 ④, 5장 첫 문단; scoped SOTA는 SSR·CAR에만 |
| AG3S SSR이 최고가 아님 | SOTA 문장 전부 삭제; 기여 3을 "충돌 절반"으로 | 1장 기여 3, 초록, 4.2, 5장 |
| CAR 우위 5 pt 미만 | "half" 삭제; 기여를 segmenter-free grounding(Tab. 4)으로 이동 | 4.2, 5장 |
| external segmentation이 AG3S와 동급 | 기여 1·2의 근거 약화 → Tab. 4 precision · prompt 민감도 · 일반화(−9.1)로 "같은 성능이라도 모델 하나 적다"로 재구성 | 4.2 ③, 4.4, 1장 |
| Level II에서 AG3S도 하락 | ⑤ 삭제; 일반화 주장은 held-out · 실로봇 unseen에만 | 4.2, 4.4 |
| ablation 어떤 항의 CI가 0 포함 | 그 항을 "needed"에서 "not separable on this set"으로 | 4.3 |

## 9. 분량 축소 권고 (사용자 질문 2026-11-08에 대한 답, §10 참조)

> **상태 (2026-11-09).** 아래 1–6은 2026-11-08 22:40에 반영했다(Tab. 4는 YOLOE · oracle 두 행만 삭제, w/o memory 행은 identity 열 때문에 유지). 이후 지시문 변경 · 민감도 · 비용 문장이 들어가 4–5장 실측은 **≈ 3.75쪽**이다. 논문 전체 8쪽 예산은 `SEC1-3_REVISION_REQUESTS_20261108_KO.md` §H가 정본이다.

본문 4–5장이 약 3.9쪽으로 상한에 닿았다. 논리를 해치지 않는 축소 순서는 다음과 같다(추정 절약량은 현재 빌드 기준).

| 우선 | 대상 | 조치 | 절약 | 논리 영향 |
|---|---|---|---|---|
| 1 | **Fig. 5 (grounding 정성 그림)** | 삭제. 정책 내부 grounding과 segmenter의 차이는 새 Fig. 1(v3)의 identity · extent 칸이 보여주고, 지시 변경 행은 supplement Fig. S2로 | ≈ 0.30쪽 | 없음 — 4.4 본문이 Fig. 1을 참조("panel d" → "identity"로 바꿀 것) |
| 2 | 4.1 Implementation details + Metrics | 5문장으로 압축(head · split · optimizer · paired · GPU), 나머지 supplement(AVA-VLA식 "details in Appendix") | ≈ 0.15쪽 | 없음 |
| 3 | 4.2 SafeLIBERO 문단 | Naive 문장 · Level II 설명 후반 · "two mechanisms" 중 ablation과 중복되는 부분 축약 | ≈ 0.12쪽 | 없음(4.3이 같은 내용을 격리) |
| 4 | Tab. 4 | YOLOE · oracle · w/o memory 행 삭제(oracle → 본문 한 구, w/o memory → Tab. 3이 담당), 4행으로 | ≈ 0.05쪽 | 없음 |
| 5 | 4.4 Generalization + run time | 3문장으로 | ≈ 0.06쪽 | 없음 |
| 6 | 4.1 Benchmarks | LIBERO-Safety suite 나열 · 실로봇 task 나열 축약 | ≈ 0.06쪽 | 없음 |

합계 ≈ 0.7쪽 → 4–5장 ≈ 3.2쪽. 1–3장에 대한 의견은 §10.

## 10. 1–3장 축소에 대한 의견 (판정은 사용자)

- **3.5 Constrained action refinement(TO)** (실측 0.25쪽 → 1/3로 압축 시 ≈ 0.17쪽 절약): 삭제는 권하지 않는다. "fixed optimizer"가 무엇인지 모르면 Tab. 1의 사다리(같은 TO) · Tab. 3(optimizer 고정) · 5장 action 축(생성 뒤 정제의 한계)이 모두 공중에 뜬다. 대신 **현재의 1/3로 압축**: 식 (8) 하나와 "deflection을 최소화하며 Eq. (2)를 soft constraint로 부과, 실패 시 HOLD" 한 문단만 남기고, 가중치 · deflection-rate gating · continuity 항 · 선형화 세부는 supplement.
- **Algorithm 1**: supplement로. 새 Fig. 2(현재 Fig. 3)가 같은 순서를 보여준다. ≈ 0.22쪽(실측).
- **Fig. 1 · Fig. 2 · Fig. 3 (2026-11-09 개정)**: 처음에는 "참고 논문은 overview 그림 1개"라며 Fig. 2 삭제를 권했으나 근거가 부정확했다. CVPR 2026 4편을 다시 확인하니 네 편 모두 Fig. 1 = "기존 vs 우리" 개념 비교(수치 없음), Fig. 2 = 방법 구조 하나다. 그래서 **기존 Fig. 1과 Fig. 2를 새 Fig. 1 한 장으로 병합**하고(세 칸이 뼈대), 현재 Fig. 3을 Fig. 2로 올린다. ≈ 0.4–0.45쪽. 구성안 `paper/figures/fig01_teaser/TEASER_V2_SAFELIBERO_PLAN_KO.md`.
- **Introduction**: 현재 약 1.5쪽. 참고 논문들은 1쪽 안팎. 둘째 · 셋째 문단(VLA 일반 설명)을 절반으로.
- **Related Work**: 현재 분량 적정.
