# CVPR 2027 원고 수정 전략 — 2026-10-18 KST

> **사용자 판정 (2026-10-18, 이 문서의 §6 · §7 · §10 을 대체한다)**
> 1. **SafeLIBERO · 실물 로봇은 포함한다** (다른 세션에서 수행 중). 원고에 결과 슬롯 (Table · 문단) 을 미리 만들고 `\tbd{}` 로 표시한다.
> 2. **본 논문의 시스템 = 개선 중인 새 방법론** (noun attention seed + depth-continuous surface graph + rendered robot mask + centered-feature identity memory · alert + 기하 phase/grasp state + supported-surface LTE + CCR, 즉 현재 원고의 configuration B). 실험 수치가 이전 방법보다 다소 낮아도 이 방법을 본 방법으로 쓴다. 기존 serving 구성 (L8H2 · radius + clustering · ball-owned) 은 **ablation · 비교 행**으로 내린다. §4.4 의 "B 를 ablation 으로" 는 철회한다.
> 3. **CVPR best paper 를 목표로 작성 시작.** W1 = Abstract · Intro · Method 재작성, Experiments Q1–Q4 골격, Fig 1–3 v1.
>
> 4. (2026-10-18 추가) **원고 수치는 보류** — 아직 불안정하므로 `\pend{}` 주황 표시를 유지한다. 등록 등급 (REGISTERED_SOURCE_LINKED) 인정 여부도 보류.
> 5. **제목 유지.**
> 6. **실험은 다른 세션이 수행한다.** 이 세션은 논문 작성 작업만 한다 (텍스트 · 그림 · 표 · supplementary · evidence 등록). 그림 제작에 필요한 CPU replay · render 는 일부 허용. §7 의 GPU 실험 목록은 이 세션의 작업이 아니다.
> 7. SafeLIBERO 결과는 아직 없다. 완성되면 사용자가 알린다 → §4.6 · §4.7 은 슬롯 유지.
>
> 결과: §6 의 A 안 / B 안 선택은 **B 안 (새 방법론 끝까지)** 으로 확정. closed-loop Table 1 의 "ours" 행은 SURF 계열이며, 허들 14 run 의 현재 값은 VLA 단독 0/14 → ours 7/14 (coarse-extent ablation 11/14). IFG-CL3 (선별 없는 40 × 2) 와 같은 40 장면의 VLA 단독 run 이 Table 1 의 주 행이 된다.

작성 근거: `paper/latex/main.tex` (abstract · intro · related work · method 활성, experiments 보류, 7 page / 참고문헌 6 page 시작),
`paper/latex/sections/04_experiments.tex` (보류된 IFG dev 평가), `paper/evidence/*.md`, `paper/figures/paperbanana/*`,
`AG3S/ag3s/docs/AG3S_T47_LOG.md` 진행 현황, `doc/active/reviews/vla_dump_20261015/WORKLOG_S0S3_KO.md` (IFG held-out · closed-loop),
`AG3S/ag3s/docs/handoff/*.verify.json` (T39 · T43V11AB · T44e-CL2 · T46-CL3 · T53 · T58 · CL4).
일정: paper registration **2026-11-10 AoE**, submission **2026-11-16 AoE**, supplementary 11-23 (`guidelines/CVPR_2027_DEADLINES.md`). 오늘부터 29일.

> 이 문서는 계획이다. 아래의 모든 수치는 해당 verify.json · summary.json 에서 옮겨 적은 것이며, 원고에 넣기 전에는
> `paper/evidence/RESULTS.md` 등록과 `/cvpr-evidence` 추적을 거친다.

---

## 0. 한 줄 결론

현재 원고는 **잘 쓴 method 설명서**이지 CVPR 논문이 아니다. 성과를 앞세운 teaser 가 없고, Method 의 절반이 "하지 않는다 · 보장하지 않는다" 류의 방어 문장이며, Experiments 가 본문에 없다.
4주 안에 할 일은 새 실험을 벌이는 것이 아니라 **이미 있는 두 기둥을 앞으로 끌어내는 것**이다.

| 기둥 | 결과 | 역할 |
|---|---|---|
| **Closed-loop (허들 장면)** | VLA 단독 **0 / 14** 성공 (13 / 14 가 장애물 접촉으로 종료) → AG3S V11AB **12 / 14** (접촉 포함 성공 12 / 14). 운반 중 사과–허들 접촉 run 5 → **0** | Fig 1 teaser · Table 1 · "왜 필요한가" |
| **Grounding (사전 잠금 held-out 34 episode)** | observed-surface IoU **0.773 → 0.876** (+0.103 [+0.083, +0.123], **34 / 0 episode 개선**), 이웃 표면 오포함 mass **−74 %** (B 안 −92 %), precision 0.894 → 0.949 | Fig 4 · Table 2 · CVPR 독자가 평가할 vision 기여 |

여기에 **"feature 는 경계를 못 긋고 identity 만 안다"** 는 분석 (Table 3) 을 insight 로 얹으면 CVPR 이 좋아하는 구조 (문제 → 기전 → 성과 → 분석) 가 완성된다.

---

## 1. Evidence inventory — 빛과 그림자

### 1.1 빛 (본문 앞쪽, 그림과 표로 크게)

| # | 결과 | 수치 | 출처 |
|---|---|---|---|
| L1 | 허들 closed-loop, VLA 단독 대 AG3S (paired 14 run) | E0 success 0 · grasp 11 · place 0, obstacle stop 13 / 14; V11AB success 12, CT 12 / 14; V1 vs E0 discordant 6 / 0 (p 0.031) | `T43W.verify.json`, `T43V11AB.verify.json` `H_totals_by_rule` |
| L2 | attached object 효과 (허들 운반 중 사과–허들 접촉) | V8 5 run → V11A 2 → V11AB **0** | `T43V11AB.verify.json` `contact_table_by_body_phase` |
| L3 | held-out grounding (잠금 후 평가, 34 ep · 2,523 frame) | G:N 0.773 → A 0.876 → B 0.891; A−G:N +0.103 [+0.083, +0.123] 34/0; render mask 효과 +0.044 [+0.038, +0.050] 34/0; 잡기 전 0.970, 손에 든 동안 0.640 → 0.848 | `heldout_locked/summary.json`, `figures/ifg/ifg-heldout-locked.json` |
| L4 | 이웃 표면 오포함 (neighbor FP mass) | 176,070 → 46,329 (A, −73.7 %) → 13,582 (B, −92.3 %) | 같은 summary |
| L5 | 동일 robot-mask 조건의 dev 비교 | 0.837 → 0.893 (+0.057 [0.006, 0.103]), 27 / 5 episode | EXP002 R004 |
| L6 | consistency alert | held-out AUROC 0.925, 오선택 100 중 85 포착, 경보 시 오포함 표면 88 % 제거 · target 6.7 % 손실; dev AUROC 0.998 (vs 0.900) | WORKLOG 10-16 22:50, EXP002 R006 |
| L7 | feature fusion 부정 결과 (insight) | edge +0.000, unary −0.066, both −0.067, shuffled 0.485, oracle seed 0.918; 경계 edge 의 73.5 % 가 같은 patch | EXP002 R005, `04_experiments.tex` |
| L8 | phase · held-state | 0.878 → 0.939 (+0.067 [0.015, 0.118]); held 0.828 → 0.933 | EXP002 R007 |
| L9 | 실제 장면 자산 | T53 ep1842 (pear + orange), T43 ep1982 허들 (E0 실패 / V11AB 성공) replay 렌더 전부 보유 | `paper/figures/paperbanana/*/assets` |

### 1.2 그림자 (Limitations 한 문단 + supplementary, 수치는 그대로 적되 맥락을 붙인다)

| # | 사실 | 처리 |
|---|---|---|
| S1 | clear 장면에서 성공률 개선 없음: T39 30/48 vs 29/48 (p 1.0), CL2 14/18, T53 test 9/16 | "장애물 없는 장면에서 손실 없이" 로 쓰되 **동등성 주장 금지**. paired 2×2 표를 supp 에 |
| S2 | surface extent (SURF) closed-loop 미확정: C1a 성공 장면 34 seed 에서 28/34 vs 34/34; 허들 7/14 vs 11/14 (p 0.22) | IFG-CL3 (선별 없는 40 × 2) 결과로 결정 (§6). 결과가 나쁘면 "정밀 extent 는 offline 검증, 실행 통합은 진행 중 + 원인 분석" 으로 Limitations |
| S3 | latency: client 왕복 median 1.1 s (C1a) · 2.9 s (SURF), 533 ms 예산 초과 | Table 4 에 단계별 ms 를 그대로. 완화 요인 병기 (별도 model copy 의 extraction pass, 미최적화 prototype, prefix pass 공유 시 제거 가능) |
| S4 | target 옆 기둥 (T58): E0 · AG3S 모두 3/3 접촉, 실제 손 점이 constraint sphere 밖 12.7–38.8 mm | Limitations 의 failure figure 로 (CVPR Highlight 논문들이 쓰는 방식). CL4 wrist · palm 형상은 성공률을 낮췄으므로 "coverage 확장은 future work" |
| S5 | LTE 대 global exclusion: gripper scope 에서 퇴화 (T46d), CL4 9 vs 7 (p 0.69) | 본문에서 "우위" 주장 제거. 설계 근거 (다른 link 가 target 을 장애물로 유지) 는 Fig 2 C2 로만 설명, 수치는 supp |
| S6 | B 안 switch 효과 미확정 (+0.015 [−0.012, +0.039]) | B 를 "adopted component, validation pending" 으로 두지 말고 **A 를 본 방법, B 를 ablation 행**으로 내린다 (아래 §4.4) |
| S7 | simulation only, RB-Y1 1 종, 과일 4 종, replay depth, 선택된 key frame | Experimental setup 에 한 번 명확히, Limitations 에 한 문장 |
| S8 | 외부 segmenter baseline · 공개 benchmark (SafeLIBERO) 없음 | §7 참조. segmenter 는 offline 비교로 4주 안에 가능, SafeLIBERO 는 이번 제출에서 제외 |

---

## 2. Thesis · 제목

**Thesis (한 문장):** *A frozen VLA already says what to touch; depth says where that object ends; its features say whether it is still the same object. AG3S keeps these three roles separate and turns them into link- and state-specific contact constraints.*

현재 원고의 identity / extent / permission 구분은 유지한다. 바꿀 것은 **"구분" 을 결론이 아니라 기전으로 쓰고, 성과로 닫는 것**이다.

| 후보 제목 | 평가 |
|---|---|
| (현재) AG3S: Grounding Target Surfaces for Selective Contact in Vision-Language-Action Policies | 정확하지만 밋밋. "selective contact" 가 무슨 성과인지 안 보임 |
| **What May Be Touched: Grounding Target Surfaces from a Frozen VLA for Contact-Aware Action Refinement** | 질문형 hook + 입력 (frozen VLA) + 출력 (action refinement). 추천 |
| AG3S: Attention-Grounded 3D Surfaces Turn a Frozen VLA into a Contact-Aware Policy | 짧고 강하나 "turn into" 가 과장으로 읽힐 수 있음 |

---

## 3. 8 page 배분 (2026 fallback 기준, 참고문헌 별도)

| 구간 | 현재 | 목표 | 비고 |
|---|---|---|---|
| Title + Fig 1 + Abstract | Fig 1 = TikZ 도식 | Fig 1 = 실제 장면 teaser, 높이 ≤ 2.9 in | 1 page 안에 abstract 와 intro 첫 문단까지 |
| Introduction | ~1.1 (6 문단 + 기여 3) | 1.0 | 근거 문단을 성과 요약 문단으로 교체 |
| Related Work | ~0.6 | 0.5 | 4 문단 유지, 각 문단 끝 "In contrast, AG3S …" |
| Method | ~2.7 | **2.2** | "Scope of this method draft" 삭제, 파라미터 문단 → supp, 방어 문장 1/3 로 |
| Experiments | 0 (보류) | **3.0** | Q1–Q4, 표 4, 그림 3 |
| Conclusion + Limitations | 0.2 | 0.4 | 합본 "Limitations and Conclusion" |
| 합계 | ~5 + 0 | 8.0 | |

---

## 4. 절별 수정 지침

### 4.1 Abstract (150–200 단어, 현재 220 단어 · 성과 없음)

순서: (1) 배경 한 문장 (VLA 는 접촉을 계산하지 않는다) → (2) 한계를 **한 구절로 명명** ("a distance field cannot tell the target from its neighbours, and the right treatment of the target changes as the task proceeds") → (3) 제안: frozen VLA 의 attention · hidden state · depth 로 identity / extent / permission 을 따로 정하고 link · state 별 contact constraint 로 바꾼다 → (4) 구성요소 3 개 이름 → (5) **수치 2–3 개** (held-out IoU +0.10, 이웃 오포함 −74 %, 허들 장면 성공 0/14 → 12/14) → (6) 범위 한 문장 (simulation, frozen VLA, no external segmenter).

### 4.2 Introduction (Visual-RRT 흐름, `notes/CVPR2026_HIGHLIGHT_WRITING_REPORT.md` §3.1)

| 문단 | 내용 | 현재 원고에서 |
|---|---|---|
| 1 | 사과 옆 기둥 사례 → identity ≠ extent ≠ permission | 유지, 2 문장 줄임 |
| 2 | 숨은 가정: "safety layer 는 관측된 모든 표면을 장애물로 둔다" → 현실: 하나는 반드시 만져야 하고 그 대상이 바뀐다 → **결합이 non-trivial 한 이유 (i)(ii)(iii)** | 2 문단을 이 구조로 재배열 |
| 3 | 문제 정의 (유지) | 유지 |
| 4 | **Key idea**: 정책이 이미 두 질문에 답한다 + 신호별 역할 분리 (feature 경계 fusion 의 실패 관찰을 한 문장으로) | 4 문단 압축 |
| 5 | 모듈 ↔ 어려움 1:1 (ATS→i, HSR→ii, LTE→iii, CCR→실행) + Fig 1 walkthrough | 5 문단 재작성 |
| 6 | **성과 요약** (held-out +0.103 · 34/0, 이웃 −74 %, 허들 0/14→12/14, alert AUROC 0.925) | 현재의 "inconclusive" 문단 교체. 미확정 사항은 §4.6 으로 |
| 7 | 기여 3 개 (굵은 머리말) + "Together, these bridge …" | 유지, 문장 다듬기 |

### 4.3 Related Work

유지. 추가: AutoMoMa 의 phase 별 collision 처리 (attached geometry 중복 인정), SafeLIBERO 계열 (VLSA · KNOWS · WBAG) 을 "benchmark 비교는 future work" 로 한 문장. 각 문단 끝을 "In contrast, AG3S …" 로 닫는다. novelty 를 "최초" 가 아니라 **"coarse policy evidence → observed 3D extent → restricted contact exception" 의 변환**에 둔다 (현재 문장 유지).

### 4.4 Method (2.2 page)

- **삭제:** "Scope of this method draft" 문단, "Instantiation and fixed parameters" 문단 (→ supp Table S1), 각 절 끝의 보증 부인 문장 (예: "does not certify …", "partial observations do not establish …") 은 Limitations 한 곳으로 모은다.
- **구조:** 각 subsection 첫 문단을 "naive → 문제 → 우리 설계 → 보존되는 성질" 로. 예) ATS: "Lifting the attention peak alone covers part of the object and flickers; growing a radius absorbs the neighbour. We seed a depth-continuous surface graph instead, so the boundary is decided by observed continuity and the seed only chooses which surface."
- **수식:** 현재 14 개 → 11 개. (3)(4) lift+affinity 합치기, (7)(8) descriptor+switch 합치기, (9) confidence 는 문장으로.
- **B 안 처리:** 본 방법 = A (surface graph + render mask + 선택 규칙 + alert). identity switching 은 "temporal reassociation (optional)" 한 문단으로 내리고 Table 3 의 ablation 행에서만 수치를 보인다. 이렇게 하면 S6 이 본문에서 사라진다.
- **HSR:** 학습된 probe 와 physical grasp state 의 분리는 유지 (좋은 문장). floor-referenced height 는 supp.
- **LTE · CCR:** 유지하되 "우위" 가 아니라 "다른 link 는 target 을 장애물로 유지한다" 는 **보존 성질**로 설명. Eq. (linkquery) 와 attached object 문단은 Fig 2 C2 · C3 와 1:1.
- **용어:** 2026-10-06 규칙 유지 (범용 용어, 새 용어 최소). 모듈 약어 ATS / HSR / LTE / CCR 는 Fig 2 · 3 과 본문에서 같은 자리에 같은 색으로.

### 4.5 Experiments (3.0 page) — 연구 질문 4 개

| Q | 질문 | 표 · 그림 | 핵심 문장 (claim-first) |
|---|---|---|---|
| Q1 | AG3S 는 VLA 단독이 실패하는 접촉 장면을 성공시키는가 | **Table 1** closed-loop (장면 유형 × arm: success, collision-free, safe success, HOLD); **Fig 6** 궤적 · attached sphere | "On the hurdle scenes the frozen policy alone completes 0 of 14 episodes and stops on contact in 13; with AG3S it completes 12 without a forbidden contact." |
| Q2 | 내부 attention 으로 target 표면을 segmenter 없이 얼마나 정확히 복원하는가 | **Table 2** held-out (G:N / radius+cluster / A / B / oracle / 외부 segmenter); **Fig 4** episode paired, phase 별, 실제 장면 | "Attention-seeded surfaces improve every one of 34 held-out episodes and cut neighbour leakage by 74 %." |
| Q3 | 어떤 신호가 어떤 결정을 맡아야 하는가 (분석) | **Table 3** feature fusion ablation (+ shuffled), alert AUROC, phase; **Fig 5** centering 전후 · patch 경계 진단 | "Internal features do not refine a metric boundary, but they do tell whether the selected region is still the instructed object." |
| Q4 | 비용 | **Table 4** 단계별 ms (median · p95), 533 ms 대비 | 숫자만, 완화 요인은 Limitations |

- 결과 문단은 **baseline 이 왜 실패하는지**로 쓴다 ("the radius grows into the touching orange because 3D proximity cannot see the depth discontinuity"; "VLA-only drives the carried apple straight into the bar because nothing in the policy represents the bar").
- setup 에 pre-registered seeds (sha256), 장면 선정 규칙, 접촉 판정 (MuJoCo contact rows, qpos replay mismatch 0) 을 **장점으로** 쓴다. reviewer 가 가장 자주 묻는 "cherry-picking" 을 선제 차단.
- 현재 `04_experiments.tex` 의 dev 표 (Table 1 matched-mask, Table 2 features, Table 3 phase) 는 supp 로 내리고 본문은 held-out 수치로 교체한다. dev 수치는 "development comparison" 으로 각주 1 개.

### 4.6 Limitations and Conclusion (합본 0.4 page)

3–5 문장 + failure figure 1 개 (T58 기둥 장면: 손 점이 sphere 밖). 항목: simulation · 단일 로봇, 1 Hz 급 latency 와 그 원인, target 옆 기둥에서의 coverage 한계, surface extent 의 실행 통합 상태 (IFG-CL3 결과에 따라 문장 확정), 외부 segmenter · 공개 benchmark 비교는 후속. 각 항목에 완화 요인 한 구절.

---

## 5. Figure · Table 계획과 규격

CVPR `cvpr.sty`: `\textwidth` 6.875 in, `\columnwidth` 3.25 in, letter. 모든 그림은 **vector PDF** (matplotlib 합성, 사진 패널만 raster 300 dpi 이상), 그림 안 글자 **≥ 7 pt** (인쇄 크기 기준), 색은 paperbanana 공용 color code (target orange `#E8711A`, neighbour blue `#2E75B6`, ours gold `#D9A406`, attached purple `#7030A0`, hold red `#C00000`).

| 그림 | 참조 이미지 | 현재 비율 → 목표 | 위치 | 내용 |
|---|---|---|---|---|
| **Fig 1** teaser | `figure1/variants/variant_A_performance.png` (2720×1733, 1.57:1, 전폭 시 4.4 in — **너무 높다**) | **≥ 2.4:1, 높이 ≤ 2.9 in** | title block (abstract 위) | 아래 5.1 |
| **Fig 2** overview | `figure2/reference_montage_v2.png` (1.52:1, 4.5 in) | **≥ 2.0:1, ≤ 3.4 in** | p.2 상단 (double column) | 아래 5.2 |
| **Fig 3** pipeline | `figure3/reference_montage.png` (2.0:1, 3.4 in) | 2.0–2.2:1 유지 | Method 첫 페이지 상단 | 아래 5.3 |
| Fig 4 grounding | 새로 | single/double | Q2 | held-out episode paired dot plot (34/0), phase 별 IoU, 실제 장면 2 쌍 (radius vs surface; volume mask vs render mask in hand) |
| Fig 5 analysis | `ifg_results/exp/exp-R10-centering.png` 재구성 | single | Q3 | feature fusion bar (+shuffled), centering 전후 cosine, patch 경계 진단 삽화 |
| Fig 6 closed-loop | T43 ep1982 E0 vs V11AB | double | Q1 | tool-centre 궤적 측면도 + 사과–허들 거리 timeline + attached sphere, HOLD 표시 |
| Fig 7 failure | T58 ep1830 | single | Limitations | 기둥 접촉 순간, sphere vs mesh |
| Table 1–4 | — | — | — | §4.5 |

### 5.1 Fig 1 — variant A 를 기준으로 고칠 것

유지: 2 × 2 (VLA only / VLA + AG3S × pick / place) + 아래 "How" 4 패널 + 오른쪽 performance. 수정:

1. **높이 압축.** 2 × 2 를 좌측 55 % 폭에, performance 를 우측 20 %, How 를 하단 얇은 띠 (패널 높이 ≤ 0.7 in) 로. 전체 2.4–2.6:1.
2. **performance 수치를 등록 결과로 교체** (현재 placeholder 45/85 등은 제거). 추천 2 개: *Task success on hurdle scenes* 0/14 → 12/14, *No forbidden contact before place* 1/14 → 12/14. 막대 아래 작은 글씨 "n = 14 paired episodes, simulation". 세 번째를 넣는다면 *clear scenes* 30/48 vs 29/48 을 "preserved" 로 — 단, Table 1 과 같은 정의여야 한다.
3. **column 제목 정직화.** 두 장면 모두 T43 허들 episode 1982 다. "Pick: obstacle next to the target" 은 reviewer 가 adjacent-post 장면으로 읽는다 (T58 에서는 AG3S 도 실패). "Pick beside the bar" / "Carry over the bar" 로. IFG-CL3 의 adjacent stratum 이 좋으면 그때 교체.
4. How 패널: 1 attention (T53 ep1842 wrist), 2 3D target vs neighbour, 3 HSR + grasp state (curve 는 실제 기록), 4 link-specific field (gold gripper + blue voxels). 각 패널 제목 ≤ 4 단어.
5. 좌상단에 instruction 문장 italic ("Put the apple in the basket.").
6. caption 3 문장: 문제 / AG3S 가 하는 일 / 수치의 조건 (paired, sim, n).
7. 생성 경로: PaperBanana 결과물은 layout 초안으로만. 최종은 `paper/tools/plotting/paperbanana_fig1_variants.py` 를 확장해 실제 PNG 자산 + matplotlib 로 PDF 출력, sidecar `.json` 에 모든 수치의 source key.

### 5.2 Fig 2 — reference_montage_v2 를 기준으로

유지: 상단 framework strip + 하단 C1 What / C2 Who / C3 When (prior vs ours). 수정:

1. **높이 압축:** 각 card 안에서 prior 와 ours 를 **좌우**로 (지금은 상하). 상단 strip 높이 0.9 in, card 2.3 in.
2. 모듈 이름을 본문과 일치: C1 = ATS (identity + extent), C2 = LTE (link-specific), C3 = HSR + grasp state (lifetime). 수식 box 는 본문 기호 (δ_{i,t}, d_t, d_t^{\setminus R}) 로.
3. C1 prior 사진 (GT mask 를 segmenter 대역으로 쓴 것) 은 caption 에 "illustration" 명시. 가능하면 실제 Grounded-SAM 출력으로 교체 (§7 의 segmenter baseline 을 돌리면 자연히 생긴다).
4. C2 prior "global exclusion: neighbour erased" 패널은 실제 ball 범위 렌더 (`f2b`) 그대로. 좋다.
5. C3 의 세 장면 (target / attached / released) 유지. "attached: avoids the bar" 는 L2 의 근거 (5 → 0 run) 와 연결.
6. 제목 줄 "AG3S: one frozen VLA decides what may be touched, by which link, and when" 은 그림 안 제목으로 두지 말고 caption 첫 문장으로.

### 5.3 Fig 3 — reference_montage 를 기준으로

유지: 5 stage + 하단 one-episode timeline. 수정:

1. Stage 2–3 ("Multi-view 3D grounding", "Boundary-preserving extent") 을 **§6 의 결정에 따라** surface graph (A 안) 또는 serving ball-owned 로 통일한다. 본문 Method 와 다른 그림을 싣지 않는다.
2. 하단 timeline 은 Fig 6 (closed-loop) 로 옮겨 Fig 3 높이를 2.4:1 로 줄이는 안을 우선 검토. 남기면 2.0:1.
3. 각 stage 에 입력 · 출력 명사 한 줄 (예: "attention 3 × 16 × 16 → seeds", "surface set T_t → exclusion R_t").
4. CCR box 의 수식은 Eq. (clearance) 와 같은 기호.

---

## 6. 핵심 결정 — "본 논문의 시스템" 은 무엇인가

두 extent 방법이 있다. (a) serving 의 radius + clustering + ball-owned exclusion (closed-loop 근거 L1 · L2 보유, grounding 정확도 낮음), (b) IFG surface graph = SURF (held-out grounding 근거 L3 · L4 보유, closed-loop 미확정 S2, latency S3).

| 안 | 본문 시스템 | 장점 | 위험 |
|---|---|---|---|
| **A** | closed-loop = (a), grounding 절 = (b) 를 ATS 의 개선 extent 로 제시, closed-loop 통합 결과는 Table 1 의 추가 행 | 모든 headline 이 이미 있음. 4주 안에 확실 | reviewer: "제안한 grounding 이 왜 closed-loop 에 없나" |
| **B** | closed-loop = (b) 끝까지 | 일관성 | IFG-CL3 가 나쁘면 headline 붕괴, latency 2.9 s |

**추천: A 로 쓰기 시작하고, IFG-CL3 (선별 없는 40 × 2, launcher 대기 중, GPU 약 5 h) 결과로 Table 1 의 SURF 행 문장을 확정한다.** 사전 결정 규칙 (결과를 보기 전에 고정):

- SURF − C1aN paired 차이의 95 % CI 하한이 −10 %p 보다 크고, 이웃 과일 접촉 run · 비허용 접촉 run 이 늘지 않으면 → SURF 행을 본문 Table 1 에 넣고 "comparable success with fewer neighbour contacts / HOLD" 중 **관측된 것만** 쓴다. Fig 3 를 surface graph 로 통일.
- 그 외 → SURF 행은 본문 Table 1 에 그대로 싣되 문장은 "the precise extent does not yet transfer to closed-loop success; §Limitations analyses why" 로. Fig 3 stage 2–3 는 serving 경로로 그리고 surface graph 는 Fig 4 에서만.
- 어느 쪽이든 **Abstract · Intro 의 수치는 L1–L4 만** 쓴다 (IFG-CL3 에 의존하지 않게).

---

## 7. 4주 실험 우선순위 (논문에 꼭 필요한 것만)

| 순위 | 실험 | 비용 | 메우는 구멍 |
|---|---|---|---|
| 1 | **IFG-CL3** (대기 중) | GPU 5 h | S2, Table 1 SURF 행 |
| 2 | **E0 (VLA 단독) 을 IFG-CL3 의 같은 40 장면 × seed 에** | GPU ~3 h | Table 1 의 VLA-only 열을 clear / adjacent / place 전 stratum 에서 pre-registered 로. 현재 E0 비교는 T39 clear 와 T43 허들뿐 |
| 3 | **E0 허들 14 run 을 현재 client 설정으로 재실행** | GPU ~1 h | L1 을 같은 날 · 같은 코드의 paired 로 (현재 C1a 11/14 는 IFG-HURDLE, E0 0/14 는 T43 시점) |
| 4 | **외부 segmenter offline baseline** (Grounded-SAM 또는 YOLOE mask → 같은 depth lift → held-out 34 ep IoU · neighbour FP) | GPU 수 h + 구현 1–2 일 | S8, Table 2 의 "why not a segmenter" 행, Fig 2 C1 prior 실물 |
| 5 | 기록 기반 ablation 집계 (새 rollout 없음): extent 변형 (CL4 X1–X4), HSR vs rule (T55), attached on/off (T43 V8→V11AB), LTE vs global (CL4 X4 vs G4 → supp) | CPU | Table 3 하단 · supp |
| 6 | runtime 표 (IFG-CL1 `timing.json`, T53 `timing_ms`) | 없음 | Table 4 |
| 7 | (여유 시) 가림 1/3 · 1/2 exploratory | CPU | supp |

**이번 제출에서 제외:** SafeLIBERO (adapter · 체크포인트 · 1,600 episode — 시작도 안 했고 4주 안에 불가), 실물 로봇. Related Work 와 Limitations 에 한 문장씩.

---

## 8. 작업 방식

- **원고 writer 1 명 (scribe 역할)**, 그림은 `paper/tools/plotting/` 의 스크립트로 verifier 역할이 만들고 sidecar 에 source key 를 남긴다. PaperBanana 는 layout 초안 전용 (최종본에 생성 이미지 금지 — `paperbanana/README.md` 규칙 그대로).
- 수치가 tex 에 들어가는 추적 경로: verify.json / summary.json 또는 계산·가정 근거 → `paper/evidence/RESULTS.md` 상태 기록 → `/cvpr-evidence` → tex. 등록 전 수치도 자료 유형과 상태를 표시해 사용할 수 있다.
- 내부 검토: W2 말 `/cvpr-review` (cvpr-reviewer agent, 읽기 전용) 1 회, W3 말 1 회. 제출 전 `/cvpr-compliance full`.
- 주간 이정표

| 주 | 기간 | 산출 |
|---|---|---|
| W1 | 10-19 – 10-25 | IFG-CL3 · E0 40 · E0 허들 실행 (순위 1–3); §6 결정; Abstract · Intro · Method 재작성; Fig 1–3 v1 (실제 장면, 수치는 `\tbd`) |
| W2 | 10-26 – 11-01 | Experiments 절 + Table 1–4; Fig 4–7; segmenter baseline; 1차 내부 review |
| W3 | 11-02 – 11-08 | 8 page 완성본, supplementary (dev gallery · head-selection protocol T51/T61 · HSR counterfactual T48 · 전체 표 · 영상), 2차 review 반영 |
| W4 | 11-09 – 11-16 | **11-10 registration**; compliance · anonymity · 참고문헌 검증; 11-16 제출; 11-23 supp |

---

## 9. 정직성의 경계 (짧게)

설득력 있게 쓰는 것과 사실을 바꾸는 것은 다르다. 이 계획은 전자만 한다: 강한 결과를 앞에, 약한 결과를 Limitations · supp 에, baseline 의 실패 이유를 설명하고, 모든 수치의 자료 유형·출처·검증 상태를 표시한다. **출처나 상태를 숨긴 수치, 표기 없는 placeholder 막대, 반대 결과의 삭제, 동등성 선언**은 하지 않는다. 그런 논문은 rebuttal 에서 무너지고, CVPR Highlight 급 논문들은 한계를 보이면서도 통과한다 (`notes/CVPR2026_HIGHLIGHT_WRITING_REPORT.md` §1.6).

---

## 10. 사용자 판정이 필요한 항목

1. §6 의 A 안 (closed-loop = serving, surface graph = grounding 절) 으로 쓰기 시작하는 데 동의하는가. 사전 결정 규칙 (CI 하한 −10 %p) 도 함께.
2. 제목 후보 중 하나 (§2).
3. B 안 switch 를 본 방법에서 ablation 행으로 내리는 것 (§4.4).
4. 순위 2–4 의 추가 GPU 실험 (E0 40 run, E0 허들 14, segmenter baseline) 승인.
5. Fig 1 performance 막대의 지표 2 개 (§5.1 추천안) 와 column 제목 정직화.
6. Fig 3 하단 timeline 을 Fig 6 로 옮기는 안.
7. SafeLIBERO · 실물 로봇을 이번 제출에서 제외하는 것.
