# 1–3장 수정 요청 목록 (사용자가 직접 반영)

- 작성: 2026-11-08 KST. 기준 원고: `paper/latex/AG3S_ Grounding Target Surfaces ... .pdf` (2026-11-08 17:47 빌드), 줄 번호는 그 PDF의 review-mode 번호. 4–5장은 `paper/latex/exp_section/`의 v4가 기준.
- 분류: **[일관성]** 4–5장과 맞지 않아 반드시 고칠 것 · **[분량]** 8쪽 안에 넣기 위한 축소 권고 · **[그림]** Fig. 1 v3(기존 Fig. 1 + Fig. 2 병합) 관련. 판정이 필요한 항목은 ◆.

## A. Abstract

| # | 위치 | 현재 | 요청 | 분류 |
|---|---|---|---|---|
| A1 | 019–027 | "On 34 pre-registered held-out episodes … IoU from 0.77 to 0.88 … 92 % … stopped by contact with a bar … 13 of 14 hurdle episodes … completes 7" | 삭제. 이전 RB-Y1 hurdle 실험 수치. Tab. 1·2 결과 문장으로 교체(결과 전까지 빨간 placeholder). 틀: "On SafeLIBERO with the same frozen π0.5, AG3S-VLA halves the collisions of the strongest safety layer and gives the highest safe-success rate, with task success on par with the best layer; …" (`paper/notes/RESULTS_NARRATIVE_20261108_KO.md` §7) | 일관성 |
| A2 | 028 | "[TBD: SafeLIBERO and real-robot sentence]" | A1과 합쳐 한 문장 | 일관성 |
| A3 | 024 | "without an external segmenter or any update to the VLA" | 유지 (4·5장의 핵심 포지셔닝과 일치) | — |

## B. Introduction

| # | 위치 | 현재 | 요청 | 분류 |
|---|---|---|---|---|
| B1 | 127–137 | "AG3S-VLA outperforms existing VLA-based methods in manipulation success and safety performance, achieving state-of-the-art results. Real-world evaluations further show strong generalization" | "manipulation success" 우위 삭제. 범위 지정: "reduces collisions by about half against the strongest safety layer on the same frozen policy and gives the highest safe-success rate, with task success on par with that layer". "state-of-the-art"는 "safe-success rate among safety layers on the same checkpoint"로 한정. "strong generalization" → "transfers to a real robot and to held-out objects" | 일관성 |
| B2 | 156–164 기여 3 | "State-of-the-Art Performance and Real-World Generalization … outperforms … in manipulation success and safety performance … strong generalization and high success rates" | 제목 "Safe-Success Gains and Real-World Deployment". 본문: "highest SSR and CAR among safety layers on the same frozen π0.5 on SafeLIBERO and LIBERO-Safety, with task success preserved; deployment on a real robot including objects unseen during development." | 일관성 |
| B3 | 139–147 기여 1 | "the first VLA framework to perform task-conditioned 3D object surface segmentation directly from policy-internal representations without … external … segmentation networks" | 유지하되 KNOWS와의 차이를 한 구로: "whereas prior attention-based filters select among externally segmented objects, AG3S-VLA derives the surface extent itself" | 일관성 |
| B4 | 109–112 | "multi-view RGB-D observations are lifted" | "calibrated RGB-D observations" (SafeLIBERO는 agent + wrist 2 view) | 일관성 |
| B5 | 112–116 기여 2 | "maintains target identity and location under partial occlusion through a spatiotemporal 3D object memory" | 유지. 단 4.4 실패 분석이 "identity가 어긋나는 경우 17 %"를 보고하므로 "maintains"를 "keeps … across occlusion and distractors"로 완화 권고 | 일관성 |
| B6 ◆ | 030–068 (1–2문단) | VLA 일반 설명 두 문단(≈ 0.45쪽) | 절반으로(≈ 0.2쪽). 참고 CVPR 2026 논문의 Intro는 1쪽 안팎, 현재 Fig. 2를 뺀 본문 ≈ 1.4쪽 | 분량 |
| B7 | 087–102 | "Recent studies have explored attention-based target identification [12], obstacle estimation using external perception models [7] …" | 유지. 4.2의 baseline 실패 메커니즘(이름 붙은 장애물만 · ellipsoid · 외부 segmenter)과 같은 서술이라 일관 | — |
| B8 | 069–086 | 사과 옆 장애물 사례, "contact with the target must be permitted during grasping, while … restricted during other stages" | 유지. Tab. 3의 contact policy ablation이 이 문장을 그대로 검증 | — |

## C. Figures 1–3

| # | 위치 | 현재 | 요청 | 분류 |
|---|---|---|---|---|
| C1 | Fig. 1 (title block) + Fig. 2 (p.2 상단) | Fig. 1 RB-Y1 hurdle teaser(수치 0.773/0.891/−92 %), Fig. 2 세 축 개념 비교 + 위쪽 흐름 줄 | **두 그림을 새 Fig. 1 한 장으로 병합** (2026-11-09 결정, CVPR 2026 4편의 Fig. 1 = 개념 비교 관례). Fig. 2의 세 칸(identity · extent · permission, Prior vs AG3S)이 뼈대, Fig. 1의 장면은 위쪽 한 줄로 축소(선택), 수치 막대 · 내부 4단계 · Fig. 2 흐름 줄은 삭제. Prior 이미지는 Tab. 1 baseline의 실제 출력으로 교체. 구성안 `paper/figures/fig01_teaser/TEASER_V2_SAFELIBERO_PLAN_KO.md` | 그림 · 분량(≈ 0.4–0.5쪽) |
| C2 | Fig. 3 (파이프라인) | 현재 Fig. 3 | **새 Fig. 2로 번호 변경**. 내용 유지. Intro 10행 `fig:overview`와 `fig:teaser` 두 참조는 새 Fig. 1 하나로 통일, Method 58행 참조는 "Fig. 1, identity"로 (구성안 §8) | 일관성 |
| C3 | Fig. 2 caption 118 → 새 Fig. 1 caption | "permission panels are recorded frames of the coarse-extent configuration" | 최종 구성 frame으로 교체하거나 문장 삭제 | 일관성 |
| C4 | Fig. 3 caption 280–286 | "Target segmentation (Sec. 3.2) … Task stage recognition (Sec. 3.4) … Action refinement (Sec. 3.5)" | 유지. 3.5를 압축해도 caption의 역할 설명은 그대로 | — |

## D. Related Work

| # | 위치 | 현재 | 요청 | 분류 |
|---|---|---|---|---|
| D1 | 185–209 | AEGIS "vision-language-guided obstacle identification … control barrier functions"; KNOWS "identifies the target from attention … external segmentation model … ellipsoid" | 유지. 4.2·4.4의 baseline 논거와 일치 | — |
| D2 | 224–231 | "unlike KNOWS, which uses attention only to select among externally segmented objects, our work …" | 유지. B3과 같은 구분 | — |
| D3 | 206–209 | "Frozen VLA representations have also been shown to encode task progress and failure signals [1, 6]" | 유지. 5장 perception 축이 같은 인용(`zhang2026frozenvlas`, `bhardwaj2026progress`)을 쓰므로 bib 키 일치 확인 | 일관성 |

## E. Method

| # | 위치 | 현재 | 요청 | 분류 |
|---|---|---|---|---|
| E1 ◆ | 3.5 (387–402) | Constrained action refinement: 식 (8), 가중치 W_t, deflection-rate gating, continuity 항, HOLD | **삭제 금지, 1/3로 압축**. 4장이 "fixed optimizer"에 세 번 기댐(Tab. 1 사다리 · Tab. 3 optimizer 고정 · 5장 action 축). 남길 것: 식 (8) + "deflection 최소화, Eq. (2)를 soft constraint로, 실패 시 HOLD = intervention" 한 문단. 옮길 것(supplement): W_t 정의, deflection-rate 항의 grasp 후 gating, continuity 항, 선형화 · solver 세부. ≈ 0.17쪽 (3.5절 실측 0.25쪽의 2/3) | 분량 |
| E2 ◆ | Algorithm 1 (p.5) | 18줄 의사코드 | supplement로. 새 Fig. 2(현재 Fig. 3 파이프라인)가 같은 순서를 보여줌. ≈ 0.22쪽 (실측: 5쪽 왼쪽 열의 44 %) | 분량 |
| E3 | 3.4, 376–386 | "clearance … imposed on the spheres of both palms and fingers; the forearms, the torso, and self-collision are not constrained" | **유지**. 4.2를 이 문장에 맞춰 "spheres of both hands and of the attached object"로 썼음. 바꾸면 4.2도 바꿔야 함 | 일관성 |
| E4 | 3.4, 350–358 | Task stage recognition: 학습 없음(gripper 피드백 + palm–target 거리) | 유지. 4.1 "the stage is read from gripper feedback and the palm–target distance (Sec. 3.4)"가 참조 | 일관성 |
| E5 | 3.3, 336–344 | "missing evidence can only keep or withdraw an accepted surface, never extend it" | 유지. 4.4 "identity drifted … 17 %"와 5장 "memory can keep a wrong first reference"가 이 설계의 한계로 서술됨 | 일관성 |
| E6 | 3.2, 284–290 | "The depth pixels of all cameras are lifted" | 유지(2 view도 "all cameras"). B4와 함께 "multi-view" 단어만 점검 | 일관성 |
| E7 | 3.1, 254–264 | "Prior safety layers take the target region from an external segmenter or a primitive such as an ellipsoid [7, 12], and the latter also frees the neighbor that touches the target" | 유지. Tab. 1 ③·④와 Tab. 4의 leakage 논거 그대로 | — |

## F. 용어 일치 점검 (1–3장 ↔ 4–5장)

| 4–5장 용어 | 1–3장에서 같은 뜻으로 쓰는지 확인 |
|---|---|
| contact links / contact policy | 3.4 "contact links in L, the palm and fingers" ✓; "permission"(Fig. 2) → 삭제 또는 "contact policy"로 통일 |
| attached object / carried object | 3.4 "attached spheres" ✓; 4장은 "carried object"도 씀 → 하나로 통일 권고("attached object") |
| stage (approach / grasp / …) | 3.4 ✓; 1장 "manipulation stages" ✓ |
| hold (intervention) | 3.5 ✓ |
| target surface / observed surface | 3.2 ✓ |
| generic geometry / external segmentation (baseline 이름) | 1장 "distance-field safety layer", "external segmenter" ✓ |

## G. 반영 순서 제안

1. 일관성 항목(A1–A2, B1–B4, C3, D3) — 결과 전이라도 지금 고칠 수 있다.
2. 분량 항목 — §H의 예산표 순서대로. 1단계(C1 병합, E2, E1, B6)를 먼저 반영하고 다시 빌드해 실측한다.
3. C1 병합 그림의 최종본 — E1a eval run · E2 grounding run 뒤 (`paper/figures/fig01_teaser/TEASER_V2_SAFELIBERO_PLAN_KO.md`).

## H. 8쪽 분량 예산 (2026-11-09)

**현재 분량 (실측).** 1–3장은 기존 원고 PDF(2026-11-08 17:47 빌드)의 절 제목 위치로 쟀다. 1–3장은 그 뒤 바뀌지 않았다. 4–5장은 `latex/exp_section` standalone 빌드(2026-11-08 23:40)로 쟀다. 두 열 원고라 위치를 "쪽 + 열 높이 / 2"로 환산했다.

| 범위 | 위치 | 분량 |
|---|---|---|
| 제목 · 초록 · Fig. 1 | 1쪽 시작 → 1쪽 오른쪽 열 y 0.85 | ≈ 0.93쪽 |
| 1 Introduction (Fig. 2 포함) | → 3쪽 오른쪽 열 y 0.59 | ≈ 1.87쪽 |
| 2 Related Work | → 4쪽 왼쪽 열 y 0.82 | ≈ 0.62쪽 |
| 3 Method (Fig. 3 · Algorithm 1 포함) | → 7쪽 왼쪽 열 y 0.26 | ≈ 2.72쪽 |
| 4 Experiments + 5 Conclusion | standalone, References가 4쪽 오른쪽 열 y 0.36 | ≈ 3.75쪽 |
| **본문 합계** | | **≈ 9.9쪽** |

CVPR 2027 본문 한도는 figure · table 포함 8쪽이다(`paper/guidelines/CVPR_FORMAT_RULES.md`, `CONFIRMED_2027`). **줄여야 할 양은 ≈ 1.9쪽**이고, 여유를 두면 2.0쪽이다. 합본에서 그림 위치가 달라지면 ±0.1쪽 바뀔 수 있으니 1단계 뒤 다시 잰다.

**축소 예산 (추정, 큰 것부터).** "논리 비용"은 그 항목을 빼면 약해지는 주장이다.

| 단계 | 항목 | 조치 | 절약 | 논리 비용 |
|---|---|---|---|---|
| 1 | C1 | 기존 Fig. 1 + Fig. 2 → 새 Fig. 1 한 장 | ≈ 0.40 (위쪽 줄 빼면 ≈ 0.45) | 없음. CVPR 2026 관례와 일치 |
| 1 | E2 | Algorithm 1 → supplement | ≈ 0.22 | 없음. 새 Fig. 2가 같은 순서 |
| 1 | E1 | 3.5절을 1/3로 (식 하나 + HOLD 한 문단) | ≈ 0.17 | 없음. 4장이 기대는 "fixed optimizer" 정의는 남음 |
| 1 | B6 | Intro 1–2문단(VLA 일반 설명) 절반 | ≈ 0.20 | 없음 |
| | | **1단계 소계** | **≈ 1.0** | → ≈ 8.9쪽 |
| 2 | Intro | 나머지 문단을 다듬어 Intro를 ≈ 1.0쪽으로 (참고 논문 수준) | ≈ 0.20 | 낮음 |
| 2 | Method | 3.2–3.4 문장 압축(식 유도 · 구현 세부를 supplement로) | ≈ 0.20 | 낮음. 식과 정의는 유지 |
| 2 | Related Work | 0.62 → ≈ 0.45쪽 (세 문단을 각 2–3문장 줄임) | ≈ 0.15 | 낮음 |
| 2 | 4–5장 | 4.1 Benchmarks 한 문장, 4.4 "Where it still fails" 두 문장, 5장 Limitations 압축 | ≈ 0.25 | 낮음. 세부는 supplement Tab. S9 |
| | | **2단계 소계** | **≈ 0.8** | → ≈ 8.1쪽 |
| 3 | Tab. 2 | LIBERO-Safety 표를 supplement로, 본문은 2문장 + 수치 | ≈ 0.30 | **있음.** 두 번째 벤치마크가 본문에서 표 없이 문장으로만 남음. AAG(파지 허용) 근거가 약해짐 |
| 3 | Fig. 3 (새 Fig. 2) | 파이프라인 높이 3.05 → 2.6 in | ≈ 0.05 | 없음. 글자 7 pt 유지 필요 |
| | | **3단계 소계** | **≈ 0.35** | → ≈ 7.75쪽 |

**판단.** 1–2단계만으로는 ≈ 8.1쪽이라 한도를 약간 넘을 가능성이 높다. 3단계 중 Fig. 2 높이 축소는 비용이 없으니 함께 하고, Tab. 2 이동은 1–2단계 반영 뒤 실측해 0.1쪽 이상 넘칠 때만 쓴다. 그 경우에도 Tab. 1 상단 보고값 블록(재현 검증), 4.3 ablation 표, 실패 두 축 분해는 줄이지 않는다. 이 셋이 4장 논리의 뼈대이기 때문이다(`RESULTS_NARRATIVE_20261108_KO.md` §1 · §4 · §6).
