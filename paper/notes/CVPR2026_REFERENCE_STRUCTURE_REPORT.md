# CVPR 2026 VLA 논문 5편의 구성 분석과 AG3S 논문 작성 지침

작성: 2026-10-06 KST · 대상: `paper/cvpr2026_ref_paper/`의 PDF 5편(본문 전체, 참고문헌 제외)
목적: 내용을 가져오는 것이 아니라 **논문을 어떻게 구성하고 어떤 문장 구조로 쓰는지**를 파악해 우리 원고에 적용한다.

| 약칭 | 논문 |
|---|---|
| ACoT | ACoT-VLA: Action Chain-of-Thought for Vision-Language-Action Models |
| AVA | AVA-VLA: Improving VLA models with Active Visual Attention |
| ActiveVLA | ActiveVLA: Injecting Active Perception into VLA Models for Precise 3D Robotic Manipulation |
| Optimus | Global Prior Meets Local Consistency: Dual-Memory Augmented VLA (OptimusVLA) |
| SaPaVe | SaPaVe: Towards Active Perception and Manipulation in VLA Models for Robotics |

---

## 1. 다섯 편의 공통 골격

### 1.1 분량과 위치 (8 page 본문 기준)

| 구간 | 위치(page) | 분량 | 비고 |
|---|---|---|---|
| Fig. 1 (teaser) | 1 | 1/3–1/2 page | 5편 모두 첫 페이지에 "기존 방법 vs 우리" 비교 그림 |
| Abstract | 1 | 약 190–230 단어 | |
| Introduction | 1–2 | 약 750–900 단어 | 3편은 p1 시작, ActiveVLA·SaPaVe는 큰 teaser 때문에 p2 시작 |
| Related Work | 2–3 | 약 300–600 단어, **2–3 문단** | 0.5–0.75 page. 짧다 |
| Method | 3–5 | 약 2–2.5 page | Fig. 2 = 전체 구조도 (a)(b)(c) |
| Experiments | 5–8 | 약 3 page | 표 4–6개, 그림 2–3개 |
| Conclusion | 8 | 85–190 단어, 1 문단 | Limitation 절은 본문에 없음(5편 모두) |

**시사점**: 결과에 약 3 page를 쓴다. 우리처럼 Related Work와 Method에 6 page를 쓰는 구성은 없다. 실험이 논문의 절반 가까이를 차지한다.

### 1.2 Abstract의 문장 순서 (5편 공통)

1. **배경 한 문장**: "VLA models have emerged as / shown remarkable progress …"
2. **한계 한두 문장**: "However, most existing approaches …" — 한계를 *구체적인 하나의 원인*으로 말한다.
   (ACoT: 중간 추론이 언어·영상 공간에 있어 action에 간접적 / AVA: 매 timestep 독립 처리 = MDP 가정 /
   ActiveVLA: 고정 카메라로 시점 선택 불가 / Optimus: 노이즈 prior와 action 분포의 차이 + 이력 부재 /
   SaPaVe: 능동 시점 이동과 조작을 함께 배우지 못함)
3. **제안 한 문장**: "To address …, we propose X, a … framework that …"
4. **구성요소 2–3개를 이름과 함께**: "Specifically, we introduce two complementary components: A and B. The former … while the latter …"
5. **결과**: "Extensive experiments in simulation and real-world … demonstrate …" — Optimus·SaPaVe는 수치를 넣는다
   ("98.6% on LIBERO", "up to 31.25% higher success rates").
6. 코드/프로젝트 링크(익명 제출본에서는 제외).

### 1.3 Introduction의 문단 구성

| 문단 | 내용 | 대표 문장 구조 |
|---|---|---|
| 1 | VLA의 성공과 일반적 구조 | "Recent advances in … have demonstrated impressive progress …" |
| 2 | **이름 붙인 한계(gap)** 한 가지 | "Despite …, a critical challenge persists: …" / "Limitation I: …, Limitation II: …"(Optimus) |
| 3 | 한계가 왜 생기고 왜 기존 해법이 부족한가 | "While a straightforward solution is …, it …" |
| 4 | **핵심 통찰 → 제안** | "We observe that … / Our key insight is that … / We contend that …" → "To this end, we propose X" |
| 5 | 구성요소 설명(모듈 이름 + 한 문장씩) | "We first propose A, which … Secondly, we devise B, which …" |
| 6 | 결과 요약(수치) | "On LIBERO, X attains …, surpassing …" |
| 7 | Contributions 3개 | 아래 1.4 |

- 5편 모두 **Fig. 1을 Introduction 안에서 참조**하며 "기존(a) vs 우리(b)"를 그림으로 먼저 보여 준다.
- ActiveVLA는 인용구로 시작한다("Perception is not a passive process…"). 흔하지 않은 장치이며, 쓰려면 한 번만 쓴다.
- Optimus는 한계를 **Limitation I / II로 번호를 붙여** 굵게 제시하고, 각 모듈이 각 한계에 1:1로 대응한다. 읽는 사람이 구조를 바로 잡는다.

### 1.4 Contributions 작성 방식

- 개수는 **3개**(ACoT만 4개).
- 공통 패턴: **(1) 개념·프레임워크** — "To the best of our knowledge, the first …"
  / **(2) 구성요소(모듈 또는 데이터셋·벤치마크)** / **(3) 실험** — "Extensive experiments … state-of-the-art …".
- 각 항목은 1–2 문장이며, 방법의 세부 단계는 넣지 않는다(세부는 Method).
- 굵은 머리말을 붙인다(ActiveVLA: "Active Perception for VLA Models: …").

### 1.5 Related Work

- 2–3개의 주제 문단만 쓴다(예: "Vision-Language-Action Models", "World-Model-based Policies").
- **모든 문단이 "In contrast, our …" 또는 "However, … To address this, we …"로 끝난다.** 논문 나열이 아니라
  "기존 접근 → 한계 → 우리의 차별점"의 순서다.
- 개별 논문을 깊게 설명하지 않는다. 대표 논문 2–3개만 한 문장씩 언급하고 나머지는 인용 묶음으로 처리한다.
- Optimus는 "Preliminaries and Related Work"로 합쳐서, 배경 수식(flow matching)과 선행연구를 한 절에서 끝낸다.

### 1.6 Method

- 첫 문장은 **절 안내**: "In this section, we first … (Sec. 3.1). Next, … (Sec. 3.2) … Finally, … (Sec. 3.4)."
- **3.1 Problem Formulation / Preliminaries**: 정책을 수식 한 줄로 정의한다. `a_{t:t+H-1} = π_θ(o_t, l)`.
  그 다음 "우리는 여기에 무엇을 더하는가"를 같은 기호로 쓴다(ACoT: guidance g ∈ {g_lang, g_vis, g_action};
  AVA: MDP `A_t ~ P(A_t|x_t)` → POMDP `A_t ~ P(A_t|x_t, r_{t-1})`).
- 이후 절은 **Introduction에서 이름 붙인 모듈과 1:1**로 대응한다.
- 각 모듈 절의 순서: **동기 한 문장 → 정의·수식 → 설계 이유**. 수식은 모듈당 2–4개.
- **Fig. 2 = 전체 구조도**이며, 캡션에서 (a)(b)(c) 패널마다 한 문장씩 모듈을 설명한다. 캡션만 읽어도 방법이 이해된다.
- 학습 절차(loss, 단계별 학습)를 마지막 절에 둔다.

### 1.7 Experiments

- 시작에 **연구 질문**을 명시한다.
  Optimus: "Q1. Does … outperform …? Q2. … real-world? Q3. How do GPM and LCM individually contribute?"
  SaPaVe: (1)–(5) 질문을 쓰고 각 질문을 Sec. 4.x에 대응시킨다.
- 4.1 Experimental Setup: Benchmarks / Implementation Details / Baselines / Evaluation Metrics를 굵은 머리말로 쓴다.
- 4.2 주 비교 표: 표준 벤치마크(LIBERO, CALVIN, RLBench, LIBERO-Plus, RoboTwin)와 **10–25개 baseline**.
- 4.3 **Ablation**: 모듈을 하나씩 더하는 체크마크 표(#1 EAR, #2 IAR, #3 both / w/o Stage 1, w/o D.A.H. …)와
  하이퍼파라미터 표 또는 그래프.
- 4.4 정성 분석(attention·heatmap 시각화), 효율 분석(latency, NFE), 실기 실험.
- **결과 문단의 문장 구조**:
  "As shown in Table X, our approach achieves …, surpassing … by Y%. We attribute this to … . This highlights the
  critical role of …"
  SaPaVe는 **결과 문단마다 굵은 주장 문장으로 시작**한다("Dynamic viewpoints are crucial in active manipulation.").
  리뷰어가 문단의 결론을 먼저 읽는다.

### 1.8 Conclusion

- 1 문단. 문제 → 제안(모듈 이름 반복) → 실험 결과 → 한 문장의 전망
  ("We believe that learning to 'think' in the language of actions is a critical step towards …").
- 별도 Limitations 절은 5편 모두 본문에 없다(보충자료로 보내거나 생략한 것으로 보인다).

---

## 2. 우리 원고와의 차이

| 항목 | 참고 논문 | 현재 우리 원고 | 조치 |
|---|---|---|---|
| Fig. 1 | p1 teaser, 기존 vs 우리 | 없음(구조도가 Fig. 1) | **teaser를 새로 만든다**(아래 3.2) |
| 한계의 이름 | 한 단어/구로 명명 | "dilemma"로 서술 | 한 구로 고정(아래 3.1) |
| Problem formulation 수식 | 3.1에 정책 수식 | 없음 | **3.1 추가** |
| 모듈 이름 | 2–3개, 약어 | 이름 없음 | 결정 필요(아래 3.4) |
| Related Work | 2–3 문단, 0.5–0.75 page | 4절 + Table 1, 약 1 page | 문단 축소. Table 1은 Intro 또는 Experiments로 이동 검토 |
| Method | 2–2.5 page | 약 3 page + Fig. 2 schematic + Algorithm | Algorithm·schematic을 보충자료로, 상세 수치는 표로 |
| Experiments | 약 3 page, 연구 질문, 표준 벤치마크, 다수 baseline | Setup만, 자체 시뮬레이션 | 연구 질문 형식으로 재구성, 벤치마크 결정 필요 |
| Limitations | 본문에 없음 | 본문 0.3 page | 한 문단으로 줄이거나 Conclusion에 합침 — 단 우리 시스템은 안전 주장이므로 범위 명시는 남긴다 |

**가장 큰 차이는 실험이다.** 참고 논문은 모두 학습 방법이고 공개 벤치마크(LIBERO 계열)에서 많은 baseline과 비교한다.
우리는 자체 RB-Y1 MuJoCo 장면에서 VLA-only와 비교한다. 리뷰어는 "SafeLIBERO나 LIBERO-Safety에서 VLSA·KNOWS·WBAG와
비교했는가"를 물을 가능성이 높다. 이것은 원고가 아니라 실험 계획의 결정이며, 제출까지의 시간을 고려해 정해야 한다.

---

## 3. 우리 논문에 적용할 구체안

### 3.1 한계에 이름 붙이기 (Introduction 2문단)

용어 규칙(범용 용어 우선, 새 용어 최소화)과 충돌하지 않도록, 새 단어를 만들지 않고 **명사구 하나**로 고정한다.

- 제안: **"the target–obstacle ambiguity of distance fields"** — 거리장은 가장 가까운 표면까지의 거리만 저장하므로,
  잡아야 할 물체와 피해야 할 물체를 구분하지 못한다.
- Optimus처럼 두 한계로 나누는 방법도 가능하다.
  **Limitation I**: 대상과 장애물을 구분하려면 외부 perception model(detector, segmenter, VLM)이 필요하다.
  **Limitation II**: 하위 작업마다 대상의 처리가 달라야 하지만, 하위 작업은 외부 LLM이나 추가 학습으로만 얻는다.
  → 우리의 두 차별점(perception model 없음, 외부 LLM·추가 학습 없음)과 1:1로 대응해 가장 명확하다. **권장.**

### 3.2 Fig. 1 teaser (p1)

참고 논문 5편의 teaser 형식("기존 방법 / 우리 방법" 좌우 비교)을 따른다. 실제 시뮬레이션 장면을 쓴다(규칙 A).

- (a) 기존: RGB → **VLM·detector·segmenter** → target/obstacle mask → collision filter
  (말풍선: "external models, names, failures when masks are wrong")
- (b) 우리: 같은 장면에서 **VLA의 attention heatmap → 3D target** + **KV cache → sub-task(pick/place/home)**
  → ESDF에서 손가락만 target을 허용 → 정제된 chunk
- 하단 띠: 같은 episode에서 VLA-only가 장애물에 닿는 프레임과 우리가 피하는 프레임(실측 결과가 나온 뒤 채움)

현재 Fig. 1 구조도는 Fig. 2(Method 개요)로 옮긴다.

### 3.3 Abstract 문장 틀 (현재 186 단어 → 190–220 단어 유지)

1. VLA는 충돌을 고려하지 않으며 성공한 episode도 안전하지 않을 수 있다. (현재 유지)
2. **However**, 거리장으로 이를 보완하려면 target과 obstacle을 구분해야 하고, 기존 방법은 이를 외부 perception model과
   외부 언어 모델에 의존한다. ← 한계를 "외부 모델 의존" 하나로 선명하게
3. **To address this, we propose AG3S VLA**, a … layer that uses the policy's own internal signals.
4. **Specifically**, two components: attention-based grounding / cache-based sub-task read-out (+ allowed collision + SQP).
5. **Experiments** … (결과 수치 1개 이상)

### 3.4 모듈 이름 — 결정 필요

참고 논문 5편 모두 모듈에 이름(약어)을 붙이고 Intro·Method·Ablation 표에서 같은 이름을 반복한다
(EAR/IAR, AVA, A-VS/A-3Z, GPM/LCM, D.A.H./C.A./U.S.K.I.). Ablation 표가 읽기 쉬워지는 장점이 크다.
우리 규칙("새 용어 최소화")과 충돌하므로, **범용 단어만으로 된 설명형 이름**을 제안한다(약어는 표에서만).

| 구성요소 | 설명형 이름(제안) | 표용 약어 |
|---|---|---|
| attention → 3D target | Attention-Seeded Target Grounding | ATG |
| KV cache → sub-task | Cache-based Sub-task Read-out | CSR |
| 손가락만 target 허용 + attached object | Link-specific Allowed Collision | LAC |
| chunk 정제 + 정지 gate | Chunk Refinement with Stop Gate | — |

### 3.5 Introduction 문단 계획 (약 750–850 단어)

1. VLA의 성공 + 충돌을 고려하지 않음(pi0.5 원문) + 성공해도 안전하지 않음(LIBERO-Safety, SafeVLA-Bench). [현재 유지]
2. 거리장이 대안이지만 **Limitation I / II** (3.1).
3. 기존 해법과 그 비용: VLSA·Lan-o3dp·KNOWS는 perception model, Lan-o3dp·pi0.5는 외부 LLM 또는 co-training.
4. **Key insight**: "We observe that the policy already carries both signals …" → "To this end, we propose AG3S VLA".
5. 구성요소 한 문장씩(ATG, CSR, LAC + refinement). "The former … while the latter …" 구조.
6. 결과 요약(수치) — 실험 후.
7. Contributions 3개:
   - **Conceptually**: VLA 내부 신호만으로 target과 sub-task를 얻는 안전 layer. "To our knowledge, the first collision-avoidance
     layer for VLAs that requires neither a perception model nor an external language model."(범위를 이렇게 한정)
   - **Technically**: ATG, CSR, LAC와 chunk refinement.
   - **Empirically**: (실험 후) 충돌 감소·성공률·ablation.
8. (선택) 장별 안내 한 줄씩 — 참고 논문들은 Intro 끝 대신 **Method 첫 문장**에서 절 안내를 한다. 지면이 부족하면 Method로 옮긴다.

### 3.6 Related Work (0.6–0.75 page, 3문단)

참고 논문 방식대로 문단마다 "기존 접근 → 한계 → In contrast, ours"로 끝낸다.
1. **VLA policies and action chunks** — 현재 유지, 3–4 문장으로 축소.
2. **Collision avoidance for VLAs** — VLSA, Lan-o3dp, KNOWS(+ WBAG, MoManipVLA를 인용 묶음으로) → "all rely on external perception or
   simulator identities. In contrast, AG3S VLA derives the target from the policy's attention and depth."
3. **Sub-task reasoning** — Lan-o3dp(GPT-4), pi0.5(co-training), LoHoVLA → "In contrast, we read the sub-task from the frozen cache …"
4. (trajectory optimisation은 2번 문단 끝이나 Method 3.x 배경으로 흡수 — cuRobo/cuRoboV2 한두 문장)

### 3.7 Method (2–2.5 page)

- 첫 문장에 절 안내.
- **3.1 Problem Formulation**: `a_{t:t+H-1} = π_θ(o_t, l)` → 우리가 푸는 문제
  `a' = argmin ‖a' − a‖ s.t. h_i(a') ≥ 0 for all query spheres i, with the target allowed only for contact links in sub-task s`.
  target T와 sub-task s를 VLA 내부 신호에서 얻는다는 점을 기호로 보여 준다(T = g(attention, depth), s = r(KV cache)).
- 3.2 ATG, 3.3 CSR, 3.4 LAC, 3.5 Chunk refinement and gate — Intro의 이름과 1:1.
- Fig. 2 = 구조도(현재 Fig. 1), 캡션에 (a)–(d) 패널 설명.
- Algorithm 1과 현재 Fig. 2 schematic은 보충자료로 보내는 것을 검토(지면 확보).

### 3.8 Experiments (약 3 page) — 연구 질문 형식

- **Q1**: AG3S VLA는 VLA-only보다 충돌을 줄이면서 성공률을 유지하는가? (주 비교 표)
- **Q2**: perception model 없는 grounding은 perception model 기반 grounding(simulator GT, YOLOE)과 비교해 어떠한가?
- **Q3**: 하위 작업 판단이 target 처리에 필요한가? (CSR gate on/off) — **V5는 gate off였으므로 반드시 실험 필요**
- **Q4**: 각 구성요소의 기여는? (체크마크 ablation 표 — 기존 V1→V4→V5 진행이 이 형식에 그대로 맞는다)
- **Q5**: 계산 비용은? (단계별 latency)
- 정성 그림: attention heatmap → 3D cluster → ESDF slice(target 허용 전/후) → 정제 전/후 궤적.
- 결과 문단은 SaPaVe처럼 **굵은 주장 문장**으로 시작한다. 단, 주장은 등록된 수치가 지지하는 범위로만 쓴다.

---

## 4. 재사용할 문장 구조 (내용이 아닌 형식)

| 용도 | 문장 구조 |
|---|---|
| 한계 제시 | "Despite …, a critical challenge persists: …" / "However, most existing approaches overlook …" |
| 한계 원인 | "As a result, …" / "Consequently, these approaches …" |
| 기존 해법의 한계 | "While a straightforward solution is to …, it …" |
| 통찰 | "We observe that …" / "Our key insight is that …" / "We contend that …" |
| 제안 | "To this end, we propose X, a … that …" / "Building upon this insight, we introduce …" |
| 구성요소 대비 | "The former …, while the latter …" / "Note that these two mechanisms are complementary." |
| 절 안내 | "In this section, we first … (Sec. 3.1). Next, … Finally, …" |
| 결과 | "As shown in Table X, … achieves …, surpassing … by …" |
| 해석 | "We attribute this to …" / "A plausible explanation is that …" / "This highlights the critical role of …" |
| Ablation | "Removing X causes a drop of …, which indicates …" / "Each component alone improves over …, and their combination achieves …" |
| 결론 전망 | "We believe that … is a critical step towards …" |

**주의**: 참고 논문들은 "state-of-the-art", "the first", "significantly"를 자주 쓴다. 우리는 등록된 결과가 지지하는 범위에서만 쓰고,
"the first"는 범위를 한정한 한 문장으로 제한한다(현재 원고의 "To our knowledge, no collision-avoidance layer for VLAs …"처럼).
유의성 검정이 없으면 "significantly"를 쓰지 않는다.

---

## 5. 다음 수정 순서 (제안)

1. Introduction을 3.5 계획(Limitation I/II, key insight, 이름 붙인 구성요소)으로 재작성.
2. Related Work를 3문단(0.6–0.75 page)으로 축소, Table 1 위치 결정.
3. Method에 3.1 Problem Formulation 추가, 절 이름을 구성요소 이름과 일치.
4. Experiments를 Q1–Q5 형식의 골격으로 재작성(결과 자리는 TBD).
5. Fig. 1 teaser 계획서 작성 → 실험 후 실제 장면으로 제작.

결정 필요: (a) 모듈 이름 사용 여부(3.4), (b) Limitation I/II 형식 채택 여부(3.1), (c) 공개 벤치마크(SafeLIBERO 등) 추가 여부(2절).
