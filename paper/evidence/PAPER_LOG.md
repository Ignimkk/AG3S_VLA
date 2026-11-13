# Paper Preparation and Evidence Log

Updated: 2026-10-06 00:01 KST

이 로그는 논문 workspace·작성·Evidence 처리 기록이다. Research implementation와
실행 결과의 정본은 [AG3S_T0T6_LOG.md](../../benchmark/ag3s/docs/AG3S_T0T6_LOG.md)이며
논문 준비만으로 그 파일을 수정하지 않는다.

## Step 1–2 — 기존 구성 기록 연결

구성 시각·수정 이유·규정 확인 기록은 [SETUP_PLAN](../SETUP_PLAN.md)과
[SETUP_CHANGES](../notes/SETUP_CHANGES.md)에 있다. 이를 research result로 취급하지 않는다.

## Step 3 — Evidence 구조와 source 등록

시각: 2026-10-05 19:45 KST

| 작업 | 산출물 | 확인 범위 |
|---|---|---|
| Claim·experiment·result·baseline·metric·artifact·log registry | `evidence/` 7개 문서 | Hypothesis와 descriptive observation 구분 |
| 연구 story·novelty·contribution·related work·question·TODO | `notes/` 6개 문서 | 아직 입증되지 않은 문장은 candidate로 유지 |
| EXP001 등록 | Manifest·source audit | 기존 T39 96 run / 48 pair / 24 episode의 16-D array와 provenance |
| Manifest 검증 | Schema·candidate 예제·validator | Source identity, 실제 array shape, paired run, verify pointer |

R001–R003의 등록 수치는 [RESULTS](RESULTS.md)를 따른다. 위 run/pair/episode 개수는
[source audit](../experiments/summaries/EXP001_source_audit.json)의 해당 count key에 연결된다.
새 collision·runtime·generalization·hardware 실험은 수행하지 않았다.

### 수정한 해석

원문의 attention observation과 desired safety/performance outcome은 hypothesis로 표시했다.
Manifest 존재만으로 evidence 사용을 허용하지 않고 status·scope·dimension·provenance를
확인하도록 했다. Real-robot과 각 ablation은 계획이며 결과가 아니다.

### 검증

Registered source audit와 semantic validator, 14-D·위장 등록·잘못된 pointer·변조된 값의
거부 확인, Markdown local link와 cross ID, 200자 `CLAUDE.md` 제한을 확인한다.
검사 결과는 [SETUP_CHANGES](../notes/SETUP_CHANGES.md)의 Step 3 기록에 남긴다.

Step 3 종료 시점의 다음 구성 작업은 Step 4: CVPR skill과 역할 정의였다.

## Step 4 — Claude 작성·검토 환경 구성

시각: 2026-10-05 20:43 KST

- CVPR skill 7개와 읽기 전용 검토 역할 3개를 생성했다.
- 공통 [CLAUDE_WORKFLOW](../notes/CLAUDE_WORKFLOW.md)에 작성·Evidence·인용·figure·review·
  compliance의 입력/출력과 역할 소유권을 연결했다.
- Skill/agent 정적 검증과 기존 EXP001의 현재 source 재검증을 통과했다.
  R001–R003의 수치·scope·claim status는 바꾸지 않았다.
- 등록된 task outcome을 safety/performance preservation로 확장하지 않고,
  미생성 원고·PDF의 검토는 NOT_CHECKED/PRE_DRAFT_REVIEW로 보고하도록 했다.
- 모델 호출·새 실험·그림 렌더링·원고 검토·공식 제출은 수행하지 않았다.

구체적인 변경 이유와 검증은 [SETUP_CHANGES](../notes/SETUP_CHANGES.md)의 Step 4를 따른다.
Step 4 종료 시점의 다음 구성 작업은 Step 5: 공식 template과 manuscript 기반이었다.

## Step 5 — Template와 manuscript·artifact 기반

시각: 2026-10-05 21:08 KST

- Official 2026 tagged kit 원본과 full commit/hash를 보존하고 provisional working style을 구성했다.
- 영문 7개 section과 별도 supplementary PDF를 빌드했다.
- F02/A01은 conceptual이며 구현 audit나 safety 입증이 아니다.
- T00은 기존 EXP001·R001/R002의 task counts만 사용한다. T01–T04와 나머지 measured figure는
  planned 상태이며 누락값을 0으로 채우지 않았다. R001–R003의 값·scope·claim status는 유지했다.
- Style/source hash와 current manifest, T00의 exact verifier pointer·분모·render source를 검증했다.
- PDF의 실제 Times 계열 font, letter page, 별도 문서, metadata·원자료 경로 노출과 배치를 확인했다.
  [STEP5_BUILD_AUDIT](../notes/STEP5_BUILD_AUDIT.json)는 scaffold 검증이며 최종 compliance audit가 아니다.
- 새 실험·실측 figure·통계 결과·citation은 생성하지 않았다. 기존 research source·로그·raw 결과를 변경하지 않았다.

원고·artifact 입력은 [MANUSCRIPT_MAP](../notes/MANUSCRIPT_MAP.md), 변경 이유는
[SETUP_CHANGES](../notes/SETUP_CHANGES.md)의 Step 5를 따른다. 당시 다음 구성 작업은 Step 6였다.

## Step 6 — 제출 checklist와 전체 workspace 검증

시각: 2026-10-06 00:01 KST

- Submission 관련 checklist 5개와 최종 구성 audit를 생성했다.
- 실제 정책·원고 준비 상태를 수동 확인하고 핵심 evidence·citation·구현/protocol·OpenReview
  미완료 항목을 FINAL_AUDIT에 남겼다. 실제 제출 준비 판정은 NOT_READY다.
- Current manifest·artifact source·공식 kit/working style hash와 Claude config·문서·PDF를 검증했다.
- Main/supplementary를 캐시만 사용해 별도 Step 6 PDF로 빌드했다. 이전 PDF snapshot을 보존했다.
- R001–R003의 값·범위, claim의 hypothesis/descriptive status를 바꾸지 않았다.
- 새 실험·측정·citation·Claude agent/skill 모델 호출·OpenReview mutation은 수행하지 않았다.

검사: [WORKSPACE_VALIDATION](../submission/WORKSPACE_VALIDATION.json).
현재 workspace 구성은 Step 1–6 완료이며 실제 연구·논문 완성 작업은 TODO의 미완료 항목을 따른다.

## Research Step R1 — 로컬 문헌 3편과 인용 초안

시각: 2026-10-06 00:48 KST

- 사용자 복원 Root CLAUDE.md를 그대로 보존하고 논문 진입점은 paper/CLAUDE.md로 지정했다.
- LoHoVLA·cuRoboV2·cuRobo report의 핵심 section과 primary 서지를 확인하고
  [R1 분석](../notes/REFERENCE_ANALYSIS_R1.md)·[provenance](../notes/REFERENCE_PROVENANCE_R1.json)에 기록했다.
- 검증한 preprint/report 3개를 bibliography에 추가하고 English Related Work에 모두 연결했다.
- 기존 거리장/geometry refinement와의 overlap, learned-policy safety-layer 후속 제안을 명시했다.
  모든 novelty 후보는 UNVERIFIED이며 기존 claim/result status와 값은 유지했다.
- 최소 공개본과 실제 local version을 구분했다. cuRobo local file은 version stamp가 없어 UNKNOWN이다.
- Cached main/supplementary build와 실제 citation-key 집합 일치를 확인했다.
  최종 검증은 [WORKSPACE_VALIDATION](../submission/WORKSPACE_VALIDATION.json)을 따른다.
- 연구 실험·source 수정·raw data 수정·Claude agent 모델 호출·OpenReview mutation은 수행하지 않았다.

다음 한 Step은 R2: 현재 구현 snapshot과 method 범위의 읽기 전용 audit다.
광범위한 grounding/contact/safety 문헌 검토와 최종 novelty 판정은 별도 후속 작업으로 남긴다.

## 선행연구 선정 상태 정정

시각: 2026-10-06 00:56 KST

사용자 설명에 따라 선행연구 선정 상태를 CANDIDATE_USER_SELECTION_PENDING으로 기록했다. R1은 기존 자료의 탐색적 분석이다. 최종 비교 대상과 인용 범위는 추후 사용자 후보 목록을 받은 뒤 정리한다. 이번 수정은 문서의 상태 설명이며 새 연구 Step이나 문헌 분석을 실행한 것이 아니다.

## 작성 담당과 준비 범위 확정

시각: 2026-10-06 01:04 KST

역할 범위를 정정했다. 준비 담당 assistant의 필수 환경 준비는 완료(잔여 0개)이고, 실제 원고 작성·후속 분석은 Claude Code 및 연구 담당자가 맡는다. notes/CLAUDE_HANDOFF.md에 작성 시작 요청·입력 위치·미확인 사항을 정리했다. 기존 R1은 탐색적 추가 작업 기록으로 보존하며 최종 선정이나 novelty 확정으로 승격하지 않았다. 이번 작업은 인계 안내와 준비 범위 정리만 수행했으며 연구 분석·본문 작성·실험·Claude 모델 호출은 진행하지 않았다.

## Draft 1 — Abstract · Introduction · Related Work · Method · Experimental Setup

시각: 2026-10-06 KST (사용자 지시: 실험 전까지만 작성, 결과는 완성 후 추가, 본문 8 page 목표)

- 정리: `latex/build/step6/`, `supplementary/build/step6/`, 오래된 `latex/build/main.pdf` 계열과 `tools/analysis/__pycache__/`를 삭제했다.
  현재 검증이 가리키는 `research-r1/` 빌드는 보존했다. 이후 새 빌드가 `latex/build/main.pdf`를 다시 만든다.
- 입력: `/papers`의 PDF 19편을 sub-agent 둘이 읽고(PyMuPDF 텍스트 추출, figure 미확인), 구현 사실은
  read-only sub-agent가 코드·로그에서 file:line과 함께 정리했다. 원자료는
  [METHOD_FACTS_DRAFT1](../notes/METHOD_FACTS_DRAFT1.md), [REFERENCE_ANALYSIS_R2](../notes/REFERENCE_ANALYSIS_R2.md).
- 작성: 00–04 section, 05 conclusion(pending), 06 limitations, F02(pipeline TikZ)·F06(exemption schematic)·A01.
  `references.bib`는 19개 entry. venue는 PDF에 인쇄된 것만 쓰고, MoManipVLA(CVPR 2025)·YOLOE(ICCV 2025)는
  출판사 페이지로 확인했다. 나머지는 arXiv preprint로 인용한다.
- 결과: 본문 5.5 page + references. 등록 결과(EXP001)는 본문에 쓰지 않았다. 빨간 `\tbd{}`가 남은 결정·결과 자리다.
- 주요 발견: KNOWS(arXiv 2606.09749)가 pi0.5 attention으로 Target을 고르고 obstacle 집합에서 제외하며,
  MoManipVLA가 실기 ESDF 구성에서 target mask를 제외한다. 따라서 기여를 '메커니즘'이 아니라
  querier-dependent exemption·payload attach·chunk refinement로 좁혔다. 선행연구 최종 선정은 여전히 사용자 몫이다.
- 한계: 새 실험·새 측정 없음. 논문 수치는 모두 코드·설정값이며 성과 수치는 없다.

## Draft 2 — 사용자 기여 3개 반영, V5 구성, 인용 보강

시각: 2026-10-06 KST (사용자 피드백: 기여 3개, 제목, cuRobo=Mapper만, 평가는 T43 V5 기준)

- 제목을 사용자 안으로 교체: 'AG3S VLA: Target-Aware Distance Fields from VLA Attention for Refining Action Chunks'.
- 기여를 사용자 3개로 재구성하되 표현을 사실에 맞췄다. (1) attention만이 아니라 attention이 seed, depth geometry가 object를 결정 — 'segmenter-free'.
  (2) VLA가 sub-task를 '출력'하는 것이 아니라 frozen KV cache의 linear probe가 닫힌 3-class 단계 라벨을 읽음 —
  [SUBTASK_FACTS](../notes/SUBTASK_FACTS.md). (3) 통합 refinement.
- Method를 V5 구성에 맞게 수정: supported-cluster 규칙, latched 동안 evidence 기반 switch, closing 동안 pre-closing geometry 유지,
  held-body whole-body spheres, obstacle row 10 mm margin, measured-pose HOLD. [V5_FACTS](../notes/V5_FACTS.md).
- 주의: V5는 `--no-subtask-gate`로 돌았다. 즉 최선 구성에서는 phase probe gate가 꺼져 있다. 본문은 gate를 'optional'로 쓰고 TBD로 표시했다.
- 인용: 19 → 36 entry. 새 entry는 arXiv abstract 페이지에서 저자·제목·날짜를 다시 읽었다. [REFERENCE_ANALYSIS_R3](../notes/REFERENCE_ANALYSIS_R3.md).
- 새 실험·새 측정 없음. V5 수치(hurdle 12/14 등)는 당시 draft에 반영되지 않았다.

## Draft 3 — Introduction · Related Work 재구성 (사용자 지시)

시각: 2026-10-06 KST
- Introduction: Background → Problem(세 질문: Target 식별, 질의자별 면제, grasp 후 payload) → Key idea → Contributions(3개, 한 줄씩) → Organisation(2장~6장 한 줄씩). run-in heading은 구조를 보이기 위한 것으로 최종본에서 뺄 수 있다.
- Related Work: 2.1 VLA policies and action chunks / 2.2 Finding the Target: perception and attention(perception first · attention(KNOWS) · hidden states) / 2.3 Trajectory optimisation and safety filters(geometry-based motion generation · VLA safety filters · guidance/chunk-level correction). 각 절 끝에 'Our setting'으로 차이를 적었다.
- 인용 추가 없음(기존 36개 안에서 재배치). 빌드 통과, 8 page(본문 약 6.5 page + references).

## Draft 4 — 용어 규칙 추가, 연구 배경 보강

시각: 2026-10-06 KST
- 규칙 추가(사용자 지시): 범용 기술 용어를 쓰고 새 용어는 최소화 — `.claude/rules/cvpr-workspace.md` '원고 용어 규칙'.
- 용어 교체: exemption by querier → link-specific allowed collision, Target-free layer → target-free ESDF,
  authorised links → contact links, payload → attached (collision) object, admissibility → candidate filtering,
  fail-closed → fail-safe, phase probe → sub-task probe, 고유명사 'Target' → target object.
- 연구 배경에 두 항목 추가: (1) VLA는 충돌 회피를 계산하지 않는다 — pi0.5 원문(PD 추종, trajectory planning·collision detection 없음),
  LIBERO-Safety Table 5(pi0.5 CR 10.0% vs privileged planner 0.0%). (2) 성공해도 안전하지 않다 — SafeVLA-Bench 초록
  (고성공 tabletop 정책도 unsafe episode 13–15%, 성공 rollout의 36–56%가 안전 조건 위반). 세 문장 모두 로컬 PDF 텍스트에서 확인.

## Draft 5 — 두 가지 차별점 강조와 수행 방법 나열 (사용자 지시)

시각: 2026-10-06 KST
- Introduction 'Key idea': (1) perception model 없이 target/obstacle 분리, (2) 외부 LLM·정책 재학습 없이 sub-task 판단을 강조하고,
  sub-task마다 target 처리가 다른 이유(pick/place(lift·transport·lower)/home)를 적었다. "To our knowledge" 한 문장으로 범위를 한정했다.
- 정정: probe는 정책을 재학습하지 않지만 linear classifier(1,539 parameters)를 시연 phase label로 fitting한다 → "non-training"이 아니라
  "the only fitted component"로 썼다.
- pi0.5 원문 확인: 원래 pi0.5는 sub-task text를 학습해 예측한다(high-level subtask prediction) → 대비 문장 근거.
- Contributions 1·2에 수행 단계를 (a)–(f), (a)–(e)로 나열. Method 3.2를 단계 (1)–(6)로, 3.3을 단계 (1)–(5)로 재구성.
- 새 표: Table 1 (T05, 선행 충돌회피 layer 비교: target/obstacle 출처, sub-task 출처, geometry, 보호 링크, 보정 방식 — 모든 셀은
  REFERENCE_ANALYSIS_R2/R3 근거), Table 2 (sub-task별 target 처리; 시연 phase 0-3/4-8/9-13 매핑은 SUBTASK_FACTS 근거).
- 분량: 총 9 page, 본문 약 7.3 page. 결과 1.5 page를 넣으려면 약 0.8 page를 줄여야 한다.

## Draft 6 — sub-task '추가 학습 없음'의 의미 정정 (사용자 설명)

시각: 2026-10-06 KST
- 사용자 의미: VLA는 embodiment를 위해 우리 환경에서 한 번 fine-tune하고, sub-task를 위해 그 이상 학습하지 않는다.
- 반영: Abstract·Intro·Method 3.3·Table 1 문구를 "no training beyond the embodiment fine-tuning"으로 바꿨다. linear read-out(1,539 params)은
  VLA를 바꾸지 않고 fine-tuning 시연에 이미 있는 phase annotation을 label로 쓴다고 명시했다(사실 유지).
- 분량은 신경 쓰지 않고 8 page 목표로 작성하기로 함(사용자 지시). 현재 본문 약 7.3 page.

## Draft 7 — Abstract·Introduction·Related Work 압축 (사용자 피드백)

시각: 2026-10-06 KST
- Abstract: 186 단어(결과 문장 추가 시 약 200–210). '200~300 글자'는 단어 기준으로 해석함 — 사용자 확인 필요.
- Introduction: 기여 3개를 한 문장씩 핵심만. Key idea에 두 차별점(perception model 없음, 외부 LLM·추가 학습 없음)을 강조. Organisation은 장별 한 줄.
- Related Work: 2.1 VLA·action chunk(유지) / 2.2 Perception-based collision avoidance(VLSA, Lan-o3dp, KNOWS) /
  2.3 Sub-task reasoning(Lan-o3dp GPT-4, pi0.5 co-training, LoHoVLA) / 2.4 Trajectory optimisation(cuRobo·cuRoboV2, MoManipVLA, WBAG).
  각 절: 기존 접근 → 한계 → 우리가 메우는 공백. OmniGuide·attention·hidden-state 문단 삭제. Table 1(선행 비교) 유지.
- 분량: 총 8 page(본문 약 6.5 page).

## CVPR 2026 참고 논문 5편 구성 분석

시각: 2026-10-06 KST
- `cvpr2026_ref_paper/`의 ACoT-VLA, AVA-VLA, ActiveVLA, OptimusVLA, SaPaVe를 본문 전체(참고문헌 제외) 읽고 구성·문장 구조를 분석했다.
- 보고서: [CVPR2026_REFERENCE_STRUCTURE_REPORT](../notes/CVPR2026_REFERENCE_STRUCTURE_REPORT.md). 내용 인용 목적이 아니라 형식 참고용.
- 핵심: p1 teaser(기존 vs 우리), 이름 붙인 한계, 3개 contribution, Related Work 2–3문단(문단 끝 "In contrast"), Method 3.1 problem formulation,
  실험 약 3 page(연구 질문·체크마크 ablation), 본문 Limitations 없음.

## Draft 8 — 보고서 적용 (Limitation I/II, 모듈 이름, 공개 벤치마크 가정, V7g 기준)

시각: 2026-10-06 KST
- 사용자 판정: Limitation I/II 채택 · 역할이 드러나는 모듈 이름 사용("Target grounding" 대신) · 공개 벤치마크는 추후 추가하므로 사용한 것으로 작성 ·
  gate on 실험 기준으로 작성.
- gate on 실험 확인: V7g (V5 − `--no-subtask-gate` + `--held-slip-detach` + `--held-cover-cap-mm 10` + `--hold-follow-to 3`, 19 run) 와
  GD (V7 off vs V7g on 19 쌍). 근거 [V7G_FACTS](../notes/V7G_FACTS.md). 수치는 당시 draft에 반영되지 않음.
- 모듈 이름: ATS (Attention-based Target Segmentation), HSR (Hidden-state Sub-task Recognition), LTE (Link-specific Target Exemption),
  CCR (Constrained Chunk Refinement). 방법 이름 매크로 \sys = "AG3S VLA".
- 구조: Fig. 1 teaser(기존 vs 우리, schematic) · Intro(배경 → Limitation I/II → 관찰·제안 → 모듈 → 기여 3개 → 장 안내) ·
  Related Work 3문단(문단 끝 "In contrast") · Method(절 안내 → 3.1 Problem formulation 수식 → ATS → HSR → LTE → CCR) ·
  Experiments(Q1–Q5, SafeLIBERO + dual-arm 장면, 주 비교표·ablation 표 골격, 값 TBD) · Conclusion 1문단 + Limitations 1문단.
- Method를 V7g에 맞춤: sub-task gate ON, slip detach, attached 구 상한(관측 반폭 + 10 mm), 측정 자세 정지, follow-TO 조건(실행 0회였음 — 핵심으로 쓰지 않음).
- 빌드 통과, 8 page(본문 약 6.7 page).

## CVPR 2026 Highlight 4편 글쓰기 분석

시각: 2026-10-07 KST
- `cvpr2026_ref_paper/`의 BiPreManip, GeoPredict, AutoMoMa(Scalable Trajectory Generation for Whole-Body Mobile Manipulation), Visual-RRT를
  본문 전체 읽고 흐름·문장 표현을 분석했다. 보고서: [CVPR2026_HIGHLIGHT_WRITING_REPORT](../notes/CVPR2026_HIGHLIGHT_WRITING_REPORT.md).
- 핵심: 숨은 가정 → 현실 대비 → "non-trivial" 결합 → key idea → 결함별 해법 → baseline 실패 메커니즘 → 짧은 limitation.
- 발견: AutoMoMa가 단계별 물체 충돌 처리(접근 단계 voxel 정리, 조작 단계 물체를 링크로 편입)를 서술 — LTE·attached object와 개념 겹침,
  Related Work 인용 필요(서지 확인 전).

## Draft 9 — Highlight 보고서 적용

시각: 2026-10-07 KST (사용자: "적용해보세요")
- Introduction: 숨은 가정("every observed surface is something to avoid") → 조작이 가정을 깸 → 결합이 non-trivial한 세 이유 (i)(ii)(iii)
  → Limitation I/II → "Our key idea is that the policy already answers both questions" → ATS·HSR·LTE가 (i)(ii)(iii)에 1:1 → 사과/배 예시
  walkthrough → 기여 3개 + "Together, … bridge …".
- Method: ATS·HSR·LTE(target-free ESDF, attached object)·CCR 첫 문단을 "naive 방법 → 실패 → 해법"으로. 추측성 문장 두 곳은 완화.
- Related Work: AutoMoMa 추가(CVPR 2026, CVF virtual 페이지로 venue 확인; PDF에는 venue 미인쇄). Table 1에 AutoMoMa 행.
- Experiments: 결과 자리에 baseline 실패 메커니즘·누적 ablation·Q3(성공률 동일 예상, 놓은 뒤 처리로 보고) 문장 틀을 TBD로 배치.
- Conclusion and Limitations 합본, limitation에 ATS 실패 사례 그림 자리(TBD).

## 연구 개발·실험·검증·논문 수정 계획 작성

시각: 2026-10-07 18:54 KST
- 사용자 요청에 따라 repository root에 [develope_PLAN.md](../../develope_PLAN.md)를 작성했다.
- 논문의 중심을 frozen VLA 내부 신호와 depth에서 task-dependent, link-specific collision constraints를 구성하는 문제로 정리했다.
- 준비·구현·fixed-input 진단·closed-loop pilot·claim 결정·held-out 본 평가·독립 검증·논문 작성·최종 점검의 9개 단계와 완료 기준을 명시했다.
- 10개 실험에 통제 조건, independent contact/task metric, 판정 기준, 필요한 Figure/Table을 연결했다. 8개 Figure·8개 Table의 패널/열, source, 배치와 provenance 요건을 작성했다.
- 최우선 진단은 동일 ATS/LTE/CCR에서 strong rule과 HSR을 비교하는 것이다. HSR의 독립 효과가 확인되지 않을 때 기여 비중을 낮추는 기준도 포함했다.
- 기존 V7/V7g/V8 자료는 candidate로 구분했다. 실제 추가 probe 학습과 별도 attention pass를 공개하고, baseline/split/geometry coverage/실행 길이의 차이를 통제하도록 계획했다.
- 문서 정적 검증: 단계·실험·Figure/Table ID와 연결, 실험별 metric/판정/artifact 항목, 로컬 Markdown 링크 11개, 코드 블록 delimiter 확인 통과.
- 이번 작업은 계획서 작성과 기록이다. 연구 코드·원자료·실험 registry·원고 변경, GPU/MuJoCo 실행, 성능 재계산 또는 evidence 등록은 수행하지 않았다.

## 논문 완성 우선으로 개발 계획 조정

시각: 2026-10-07 19:04 KST
- 사용자는 시스템 전체의 연구 완성도보다 남은 시간 안에 완성도 높은 논문을 만드는 것을 우선한다고 설명했다.
- `develope_PLAN.md`의 0절에 PAPER-FIRST 실행 계획을 추가했다. 현재 순서는 PA-1~PA-5이며 기존 S0~S8/DX-00~09는 선택용 상세 명세로 보존했다.
- Main 결과와 target-derived contact constraint의 핵심 비교를 먼저 확보하고, HSR·CCR의 독립 novelty 입증은 결과와 필요한 주장에 따라 선택하도록 했다.
- 기본 산출물은 Figure 1–3, main/ablation table, 간결한 runtime/설정 자료다. 기존 evidence 선별과 논문 작성을 초기에 시작하고, 대규모 새 연구를 중단할 시점도 정했다.
- 원고의 핵심 성과를 선명하게 제시하고 상세 구현은 supplementary로 배치하되, 수치·비교조건·학습/추론 범위·결론을 바꾸는 제약은 정확하게 설명한다.
- 이번 작업은 계획 및 기록 수정이며 실험·원고 변경은 수행하지 않았다.

## CVPR 적합성·Appendix·프로젝트 페이지 계획 반영

시각: 2026-10-07 19:18 KST
- 사용자 요청에 따라 CVPR 2027 CFP, 2026 Author/Reviewer Guidelines, 공식 author kit를 확인했다. 2027 CFP와 주제는 공개되어 있으며 세부 부록 규정은 이번 확인에서 확보하지 못했다.
- [Appendix·프로젝트 페이지 계획](../notes/CVPR_APPENDIX_PROJECT_PAGE_PLAN.md)을 작성하고 [develope_PLAN.md](../../develope_PLAN.md) 0.6–0.8절에 반영했다. Visual signal→3D target→link-specific contact constraints를 CVPR 독자에게 보여주는 구성을 제안했다.
- Appendix A–F 목차, 해당 Figure/Table, GitHub Pages 페이지 구조, 실험별 영상·그림 원자료와 PA-1~PA-5 작업을 연결했다. 기존 본문 3개 Figure와 핵심 실험을 재사용하며 별도 대규모 웹/연구 프로젝트를 추가하지 않는다.
- 공개 project website와 심사 내용을 확장하는 외부 링크의 규정상 차이를 기록했다. 부록 정책·2027 상태 파일에는 부분 재확인만 추가했다.
- NVIDIA FoundationStereo와 Google Research·UW Nerfies의 공식 프로젝트 페이지를 시각적 설명·비교 데모 구성의 참고로 검토했다.
- 이번 작업은 계획 문서와 규정 확인 기록 수정이다. Appendix 본문·원고·연구 코드 변경, 실험, 사이트 구축·배포는 수행하지 않았다.

## 실험 구성 확정 (사용자 판정)

시각: 2026-10-07 KST
- SafeLIBERO(주 결과: pi0.5 + AG3S 대 VLSA·KNOWS·WBAG 2–3개, 인용 시 각 논문 지표 정의), MuJoCo(자체 과제: baseline 3–4개, 코드 없으면
  재구현 표시; 우선순위 KNOWS → VLSA → WBAG), 실물(AG3S만, 실행 가능성 시연).
- Ablation은 부품 대 부품(ATS·HSR·LTE·attached object의 대안 비교).
- 반영 문서: [SAFELIBERO_PLAN](../notes/SAFELIBERO_PLAN.md) 1절, `develope_PLAN.md` 0.9절. 원고·코드 변경 없음.

## Draft 10 — Introduction·Related Work·Method 개정

기록 날짜: 2026-10-08 KST (중단 후 재개, 클라이언트 날짜 기준)
- 사용자의 집필 요청에 따라 기존 `latex/main.tex` 구조에서 세 절을 개정했다. 중심 기여는 VLA attention/depth의 target geometry를 단계별·링크별 접촉 제약에 연결하는 것이다.
- Working title과 Abstract·Conclusion·Limitations를 같은 방향으로 정리했다. HSR는 별도로 학습한 probe, attention은 별도 extraction pass임을 명확히 했으며 VLA collision-avoidance 재학습이 없다는 범위로 표현했다.
- Method의 query 수식·phase/grasp 역할·continuity/slack·target-free region을 관련 구현과 대조했다. 세부 최종 설정과 성능은 구체적인 TBD로 남겼다.
- Related Work는 기존 verified reference와 주요 primary source를 대조해 네 문단으로 작성했다. 최초성이나 미검증 성능 우위는 추가하지 않았다.
- Figure 1·3은 실제 자료로 채울 storyboard, Figure 2는 정확한 두-pass 데이터 흐름으로 변경했다. 기존 prior-work 대형 표와 상세 algorithm은 현재 본문에서 제외했다.
- 재개 시 추가된 사용자 확정 실험 구성(SafeLIBERO 주 평가, 자체 MuJoCo 비교·ablation, 실물 feasibility)을 확인하고 Experiments 골격과 표에 반영했다.
- 공식 kit/style·manifest·artifact hash 검증, Tectonic 오프라인 빌드 통과. 참고문헌 포함 8페이지 전체를 렌더링해 시각 점검했다. 인용 19개 key와 참조가 해결되며 overfull/underfull·미해결 인용 경고가 없다.
- 기록: [DRAFT10_REVISION](../notes/DRAFT10_REVISION.md), [DRAFT10_BUILD_AUDIT](../notes/DRAFT10_BUILD_AUDIT.json). 연구 코드 변경·추가 실험·성능 등록·사이트 배포는 수행하지 않았다.

## 2026-10-13 KST — F01–F03 CVPR-Figure 약식 draft

- 사용자 요청: CVPR-Figure(github.com/ChizkiyahuOhayon/CVPR-Figure, commit `c220e5d`) engine으로 Figure 1(demo teaser)·2(conceptual overview)·3(method pipeline)을 "모두 개선 성공" 가정의 약식으로 제작.
- 산출: `figures/fig0{1,2,3}_*/cvprfig/` 아래 spec(`.yaml`)·`out/`(svg/pdf/png/pptx)·`figure.json` sidecar·`figure.tex`. 기존 storyboard `figure.tex/json`은 수정하지 않았다.
- Raster crop: `tools/plotting/cvprfig_assets.py` → `figures/cvprfig_assets/`(10개, source sha256과 crop box는 `assets.json`). 새 수치·출력은 만들지 않았다.
- 검증: 세 spec 모두 engine auditor FAIL 0. 남은 WARN은 의도한 placeholder(F01 4개, F02 1개, F03 1개)뿐. Host에 Times metric font가 없어 STIXGeneral alias로 렌더(PDF embed 확인).
- 한계: F01 demo frame은 `ep1807_apple.mp4` stand-in(controller 설정 미등록)이며 비교 패널·결과 문구는 측정 전 의도다. F03의 boundary-preserving extent는 planned(점선). `is_measured_result=false`.

## 2026-10-13 KST — PaperBanana용 F1–F3 자료 패키지

- 사용자 요청: 실제 코드·원고·실험 결과 기준으로 PaperBanana 입력 자료(brief_en.txt·caption.md·evidence.md·assets·reference_montage.png) 준비.
- 위치: `figures/paperbanana/figure{1,2,3}/` 및 `figures/paperbanana/README.md`(사용 방법·색 규약·상태 요약). 기존 코드·원고 파일은 수정하지 않았다.
- 근거: 읽기 전용 audit 3건(구현 코드, 원고, 실험 결과·이미지). 수치는 verify.json key를 직접 대조했다(T53 `numbers.task_success` 9/16·`success_ct` 7/16, T46-CL3 `t1_success.task_success` 14/13/13 of 18, T46 held-out `lost_neighbour_seeds` 0/0/2463, T44 log 7056행 833→344).
- Asset: `tools/plotting/paperbanana_assets.py`가 T53 C1a ep1842 기록(3인칭 영상, client step npz, server chunk dump, GT replay)에서 14개 PNG와 sidecar를 생성했다. F2 ownership 패널은 serving 함수 `_ball_owned_split`를 기록된 점에 호출했다(seed는 표면 1 voxel 이내 grid로 근사, sidecar에 명시).
- 확인한 범위 제한: `ball-owned`·`exclude-authorized`는 code default off·평가 arm on; query는 gripper link만(`--links gripper`); phase는 주입값(approach); 경계 보존·LTE의 closed-loop 이득은 미확립; 이전 stand-in `ep1807_apple.mp4`는 사용하지 않았다.
- 원고 불일치(미수정, 보고만): Method는 ball만 설명하고 ball-owned를 다루지 않음; contact 예외 용어가 혼재.

## 2026-10-13 KST — PaperBanana reference montage 재제작 (사용자 피드백: 품질 개선)

- 이전 montage(연구용 진단 차트 조합)를 폐기하고 `figures/paperbanana/figure*/assets/diagnostic/`으로 옮겼다.
- 새 자료는 최신 closed-loop 기록 T53 C1a ep1842(2026-10-13)의 기록 `qpos`를 MuJoCo 3.11로 1600×1200 replay 렌더링해 만들었다(`tools/plotting/paperbanana_render.py`, CPU OSMesa). 여기에 server dump overlay(target cluster, attention cloud, query sphere, ball, ESDF voxel)와 기록 chunk의 FK(TCP 경로, attached sphere 운반)를 얹었다. 재시뮬레이션은 하지 않았다.
- F2 (c)는 채택된 T46 F1 surface evidence를 serving 함수(`nontarget_surface_evidence` + `_ball_owned_split`)로 이 기록 chunk에 offline 적용한 결과다(target 122 제거, 이웃 32 유지, ambiguous 2). T53 run 자체는 proximity evidence였다(49/32/75). 이 as-run 버전도 asset으로 남겼다.
- `tools/plotting/paperbanana_compose.py`가 논문 figure 형태의 reference montage 3장과 F3 차트(TCP 경로, episode timeline)를 생성한다. brief_en.txt(2,653 / 2,450 / 3,372자), evidence.md, caption.md, README를 새 asset에 맞게 갱신했다.

## 2026-10-13 KST — Figure 1 두 안 (사용자 요청: 성공 가정 draft)

- 위치: `figures/paperbanana/figure1/variants/`. A = 성능형(`variant_A_performance.png`), B = 한계→필요성형(`variant_B_limitation.png`). 각 안에 PaperBanana brief(2,665 / 2,481자)와 README가 있다.
- Pick/place 장면: T40 G 허들 run(E0 = VLA 단독, E3b = AG3S)의 기록 66-D qpos(`frames.jsonl`)에 run 자체의 허들(`PickPlaceObstacleManager`)을 배치해 MuJoCo로 replay 렌더링했다. 손끝 경로는 FK, 충돌 표시는 `mj_geomDistance` 최근접점이다(`tools/plotting/paperbanana_fig1_scenes.py`).
- 메커니즘 패널은 T53 C1a ep1842 렌더를 재사용했다. 성능 막대는 layout용 placeholder 값이며 차트에 그렇게 표시했다(`tools/plotting/paperbanana_fig1_variants.py`).
- 2026-10-13 수정(사용자 지적: pick 환경 불일치): Figure 1 pick/place 장면 네 개를 모두 T43 ep1982 `t43p2_overbar` 한 환경으로 교체했다. pick 짝은 seed 19822, place 짝은 seed 19823이며 VLA 단독 E0와 AG3S V11AB를 비교한다. gripper–bar 거리 −0.1 / 0.6 mm는 T43 W 분석값과 일치한다.

## 2026-10-13 KST — Figure 2를 overview로 재구성 (사용자 요청)

- 구성: (a) End-to-end VLA, VLA + geometric safety filter, AG3S의 pipeline 비교 카드(블록 흐름 + 같은 장면의 결과). (b) 같은 장면의 ① attention → ② 3D target grounding → ③ task·link-conditioned field → ④ sub-task·grasp 상태 → ⑤ CCR.
- 장면은 Figure 1과 같은 T43 V11AB ep1982 overbar다. ③은 같은 chunk의 기록된 target-free ESDF(`esdf_free_distance`)를 사용했다. `tools/plotting/paperbanana_fig2_overview.py`를 쓴다. 이전 개념도는 `*_v1_conceptual.*`로 보존했다.

## 2026-10-13 KST — Figure 2 v2 (contribution 중심 재구성, v1 유지)

- 진단: v1은 비교 대상이 약했고(가장 가까운 선행인 segmenter·global exclusion이 빠짐), 차이가 블록 수로만 보였으며, 기여 단계 표시와 link별 query 규칙이 없었다.
- v2(`figure2/reference_montage_v2.png`, `brief_v2_en.txt` 3.6k자, `caption_v2.md`): 상단 framework는 frozen VLA의 세 출력이 link-specific contact representation 식 하나로 모여 CCR로 가는 구조다. 하단은 C1 What / C2 Who / C3 When을 각각 prior와 AG3S로 대비한다.
- 출처: C1 prior는 T53 ep1842 GT target label로 이상적 segmenter를 대신한다. C3는 T43 V11AB ep1982 chunk 5/22/34. `tools/plotting/paperbanana_fig2_v2.py`.


## 2026-10-15 — IFG 개발 결과와 novelty PDF를 반영한 원고 개정

- 사용자 승인 범위: 지정한 두 보고서/PDF 확인 후 실제 논문 수정.
- 제목·초록·기여·related work·method·experiments·conclusion·limitations를 현 구현과 결과에 맞췄다. 실측 표 3개, 벡터 도식 2개, 실측 그래프와 replay 예시를 연결했다. 기존 이미지 원본은 보존했다.
- EXP002: 180 completed frames / 32 dev episodes, 실제 16-D state/action header, source code hash 확인. R9 표/paired bootstrap을 raw rows에서 재집계했고 R11b 동일 147프레임 AUROC=.997636 대 .899527을 재계산했다. Precision 결측 및 phase 집계 단위 차이를 본문에 명시했다.
- `python3 paper/tools/latex/build_ifg.py` PASS: 활성 IFG evidence/그림 hash/공식 working style 확인, PDF 9페이지(본문 8페이지 이내, 참고문헌 8페이지에서 시작), undefined citation/reference 및 overfull box 없음. Times 계열 본문 font embedded 확인. 첫 페이지·도식·표·결과 페이지 시각 검토.
- 기존 `build.py --check-only`는 이번 원고에서 사용하지 않는 EXP001의 이전 `benchmark/ag3s/docs/handoff/T39.verify.json` 경로 부재로 실패한다. 이를 통과했다고 보고하지 않았고, 기존 validator를 완화하지 않았다. 현재 IFG 형식용 별도 audit/build를 사용한다.
- 남은 TeX 경고: 캐시된 lineno.sty/algorithm.sty의 기존 UTF-8 경고. 모델·serving·수집기·원자료 변경 없음. Full submission audit 또는 독립 held-out 검증을 수행한 것은 아니다.
- 개정/문헌 판정: `paper/notes/IFG_REVISION_20261015_KO.md`; 이번 작업만의 before/after: `paper/notes/IFG_REVISION_20261015.diff`; build record: `paper/notes/IFG_REVISION_20261015_BUILD.json`.

## 2026-10-16 — 기존 전체 AG3S에 개선을 통합한 1–3장 개정

- 사용자 최신 지시를 반영해 IFG 개선만 중심에 둔 구성을 전체 ATS–HSR–LTE–CCR 방법론으로 재작성했다. 초록/Introduction/Related Work/Method와 개념도 2개를 정렬했다.
- 새 surface graph, centered feature memory(B), consistency 및 3D state cue를 기존 모듈의 개선으로 배치했다. Grasp/carry/release, link-specific fields, attached geometry, chunk refinement, 실행 prefix 검사를 유지했다.
- 새 모듈의 offline 구현과 post-switch confidence·phase gating·permission lifecycle·prefix check 통합 설계를 구분했다. 미검증 성능, 최초성 또는 안전 보장은 주장하지 않았다.
- 실험·결과 source와 그림을 보존하고 이번 main PDF에서는 제외했다. 모델·serving·수집기·원자료 변경 없음.
- `python3 paper/tools/latex/build_method.py` PASS. PDF 총 7페이지, 참고문헌은 6페이지 시작. 공식 style hash 일치, undefined reference/citation 및 overfull box 없음. 글꼴 포함과 첫 페이지·구조도·방법 수식 페이지 시각 검토. 기존 cached package의 UTF-8 경고만 남음.
- 현재 범위의 빌드 검사이며 실험 evidence 재검증·독립 성능 평가·최종 제출 검증이 아니다. 이전 `build_ifg.py`는 결과 수치를 요구하는 이전 실험 원고용이므로 현 1–3장에는 적용하지 않는다.
- 개정 기록: `paper/notes/AG3S_METHOD_REVISION_20261016_KO.md`; 빌드 기록: `paper/notes/AG3S_METHOD_REVISION_20261016_BUILD.json`.

## 2026-10-18 — Introduction의 동기·문제정의·통찰·근거 연결

- 사용자 요청에 따라 사례를 압축하고 핵심 구분을 첫 문단으로 옮겼다. 배경·기존 한계 다음에 입력/출력/조건을 포함한 문제정의를 독립 문단으로 명시했다.
- 개발 관찰→정보별 역할 분리→전체 AG3S 방법→근거와 기여로 연결했다. Related/Method/실험 source와 연구 코드는 변경하지 않았다.
- EXP002 audit 재실행 PASS. Offline 180프레임/32 dev episode의 범위만 Introduction에 사용했다. Switching 효과 미확정 및 실행 성능과의 구분을 명시했다. 최근 SURF 실패 원인에 대한 새로운 인과 주장은 하지 않았다.
- Method draft build PASS: PDF 7페이지, 참고문헌 6페이지 시작, 공식 style hash 유지, unresolved reference/citation 및 overfull box 없음. 2–3페이지 시각 검토.
- 기록: `paper/notes/INTRO_REVISION_20261018_KO.md`, `INTRO_REVISION_20261018.diff`, `INTRO_REVISION_20261018_BUILD.json`.

## 2026-10-18 — CVPR v1 evidence registry 등록 (EXP003–EXP006, R008–R026)

시각: 2026-10-18 KST (audit 실행 system clock 09:09 KST). 범위: 원고가 인용할 수치를 registry에 등록만 했다. 연구 코드·`AG3S/`·`doc/`·storage·`paper/latex/`·`paper/figures/`는 수정하지 않았고, 실험·rollout·model 실행은 없었다.

- **등록 수준 = REGISTERED_SOURCE_LINKED.** 새 script `paper/tools/analysis/audit_cvpr_v1_evidence.py`가 source JSON의 기록된 JSON pointer에서 모든 등록값을 다시 읽어 대조하고(float |diff| ≤ 5e-4, count·string exact) 14개 source 파일(storage 6, handoff 7, WORKLOG 1)의 sha256을 `paper/experiments/summaries/CVPR_V1_evidence_audit.json`에 고정했다. **16-D manifest validation(NPY header·per-run 재집계·rollout 재실행)은 하지 않았다** — EXP001·EXP002의 `REGISTERED`와 다른 수준이며 EXPERIMENTS.md에 명시했다. 결과: 434 checks, **431 PASS, 0 FAIL, 3 `source=worklog`**(text-presence만). 모든 result ID PASS.
- **EXP003 (R008–R012)** held-out locked grounding, 34 성공 rollout episode / 2,523 frame (clear 13 / adjacent 12 / place 9). arm별 IoU·precision·recall, neighbour/support/robot FP·TP sum, paired 차이(A−G:N +.10304614664251237 [.08323494981427215, .12313285145943233] 34/0; A−A-fk +.04403272045947481 34/0; B−A +.014779314128842732 [−.0116789100514432, .03944222857471973] 14/6), grasp-state IoU(pre .9698507944447411, held .8475770155578735 vs FK mask .64040778567183, placed .845807102001963), confidence alert(`/gate_success`: threshold .5904207590672415, 85/100, false alarm 153, FP 제거 .8788…, TP 손실 .0670…, AUROC .9251551724137931). **모두 `heldout_locked/summary.json`에 있어 WORKLOG 의존이 없다.** B−A의 CI는 0을 포함해 switching 효과는 미확정으로 적었다.
- **EXP004 (R013–R018)** hurdle T43 H 14 run. E0(VLA 단독)는 `T43W.verify.json` `summary_per_arm["E0/H"]`(key에 `/` 포함 → pointer `E0~1H`) success 0 / grasp 11 / place 0 / obstacle stop 13, per_run 14행 재집계 일치, **2026-10-05 T43 W code state**. V8/V11A/V11AB `H_totals_by_rule` success_ct 8/8/12, carry-phase 사과–hurdle 접촉 run 5/2/0, **2026-10-07 T43 V11 code state**. C1a/C1aN/SURF(IFG-HURDLE, **2026-10-17 현재 구현**) S_old 11/11/8, S_contact 11/11/7, grasp 13/13/9, HOLD 86/82/11, 비권한 link–target 0/0/1; pairs C1aN vs SURF success_ct both 6 / C1aN-only 5 / SURF-only 1 / p .21875; client round trip median/p95 1256/2034, 1177/2152, 3331/4979 ms. 같은 (episode, seed)이지만 code state가 셋이라 `MIXED_CODE_STATES`로 표시했다.
- **EXP005 (R019–R021)** held-out closed-loop C1aN vs SURF, 34 seed 쌍(collector에서 **C1a가 성공한** episode에서 선별, `/C1a_reference/n_success` 34/34). 34/34 vs 28/34, pairs 28/6/0, McNemar p .03125, SURF 실패 no_grasp 5 · not_placed 1, HOLD 50 vs 19, completion median 16.0 (n 34) vs 14.666… (n 28), neighbour 접촉 2 vs 2; latency client 1137.48/1923.85 vs 2857.69/3623.06 ms, server total 796.46 vs 2586.57, attention_surface 1663.45 (build 827.567 · solve 285.104 · codes 383.465), render mask 158.58. 상태 **EXPLORATORY_SELECTION_CONDITIONED**, 등록 해석 문장은 WORKLOG 2026-10-18 14:30 절을 그대로 영어로 옮겼다("…does not estimate the performance difference over the full set of scenes not selected by success").
- **EXP006 (R022–R026)** Limitations context: T58 기둥 접촉 6/6 (E0 3/3, ON 3/3; hand 점 sphere 밖 12.72–38.8 mm, script가 `cover_rows`에서 min/max 계산), CL4 success_ct X1 10 / X2 5 / X3 7 / X4 9 / G4 7 of 18, X4 vs G4 p .6875, FT–target 접촉 0, T46d gripper scope 0/349,920, CL2 C0 14/18 (success_ct 12/18), T53 9/16 (7/16, HOLD 37). T39(R001–R003)와 EXP002 dev 수치(R004–R007)는 cross-reference만 했다.
- **Claim:** C012–C020 추가(SUPPORTED_DESCRIPTIVE / EXPLORATORY_DESCRIPTIVE / EXPLORATORY_SELECTION_CONDITIONED / NOT_ESTABLISHED). C001–C005 hypothesis·C007 NOT_ESTABLISHED 유지(등록된 모든 arm이 533.3 ms budget을 넘는다).
- **원고 위치별 번호 지도:** `paper/evidence/NUMBER_MAP_CVPR_V1.md` (Abstract · Intro · Table 1 closed-loop · Table 2 held-out grounding · Table 3 analysis · Table 4 runtime · Limitations).
- **JSON과 WORKLOG의 불일치 2건 (JSON을 정본으로 등록, audit에 `source=worklog`로 기록):** (1) WORKLOG 2026-10-17 23:40 "C1aN만 성공 6, SURF만 성공 2"는 `IFG-HURDLE/agg.json` success_ct의 5 / 1과 다르다(p .21875는 5/1과 일치). (2) WORKLOG 03:30 "완료 시간 median (둘 다 성공 28쌍)"의 C1aN 값은 agg.json에서 n 34 기준이며 28쌍 한정값은 JSON에 없다.
- **JSON에 없어 등록하지 않은 것:** E0 대 V11AB·대 SURF의 paired 2×2(서로 다른 code state, source에 pair 표 없음); IFG-CL3 선별 없는 재평가(실행 중); attention extraction·VLA inference 단독 시간(`not_measured`).
- **남은 검증:** 이 등록은 source-linked 수준이다. 원고 확정 전 `/cvpr-evidence full`에서 manifest 수준(16-D array·per-run provenance) 검증 추가 여부를 판단한다. 원고(`paper/latex/`)와 figure는 이번에 바꾸지 않았다.

## 2026-10-23 KST — CVPR 2027 정책 · author kit 반영, WBAG 제거

- CVPR 2027 Author Guidelines 와 author kit v1 (LaTeX) 공개 확인. kit 원본을 `paper/template/cvpr2027_official/` 에 보존 (archive sha256 은 `paper/template/PROVENANCE.json`), working `cvpr.sty` 를 2027 bytes 로 교체. 2026 대비 차이는 review copy 의 Paper ID 제거뿐.
- 필수 "LLM Usage Statement" 를 `latex/sections/07_llm_usage.tex` 로 main text 뒤 · references 앞에 추가 (저자 확인 필요).
- 사용자 지시로 WBAG (`zhen2026wbag`) 를 bib · Related Work · SafeLIBERO 표에서 제거.
- `paper/guidelines/` 의 2027 항목을 `CONFIRMED_2027` 로 갱신, SOURCES S16 · S17 추가.
- 8 page 감축안: `paper/notes/CVPR2027_COMPLIANCE_AND_PAGES_20261023_KO.md` (미적용, Fig 2 처리 판정 대기).
- 사용자 판정 (2026-10-23): Fig 1 · Fig 2 · Fig 3 모두 유지. 8 page 감축은 추후, 그림 삭제 없이 본문 압축 · supplementary 이동으로.

## 2026-10-23 KST — Introduction · Related Work 검토와 참고문헌 보강

- 문장 검토: Intro 2문단의 근거 없는 주장 ("exempting a volume around the detection") 을 확인 가능한 서술 ("approximate each object by an enclosing volume", VLSA · KNOWS 의 ellipsoid 표현) 로 교정; Key idea 의 attention · hidden-state 주장에 인용 추가; "We present" 문단의 콜론 나열 한 문장을 네 문장으로 분리; 1문단과 반복되던 "the apple itself must clear the post" 를 다른 표현으로; 기여 2 와 Key idea 의 중복 축소. RW 2문단의 접속 문장 분리, 3문단을 고전 (potential field · trajectory optimization · CBF · depth-fused field) → language-to-3D constraint → VLA safety layer → attached / allowed collision 순으로 재구성.
- 철자: 원고 전체 (본문 · caption · supplementary · Fig 1–3 그림 글자) 를 American 으로 통일 (neighbour → neighbor, behaviour → behavior, modelled → modeled).
- 참고문헌 21편 추가 (유사 논문 KNOWS · VLSA 의 참고문헌 구성 참고, 고전 포함): RT-1, RT-2, OpenVLA, Octo, Grounding DINO, SAM, DINO, normalized cuts, Felzenszwalb graph segmentation, random walker, local-global consistency, Khatib potential field, CHOMP, TrajOpt, CBF (Ames 2019), Voxblox, MoveIt, VoxPoser, ReKep, SafeVLA, LIBERO. 서지는 모두 1차 기록 (Crossref DOI, PMLR v229 · v270 목록, NeurIPS 목록, JOSER 사본) 과 대조. 기존 nvblox · Curless 1996 · YOLOE 를 본문에서 인용하도록 연결.
- Contemporaneous 확인: agarwal2026multilink (v1 2026-09-30), guo2026nudge (v1 2026-09-28) 는 2026-09-15 이후 → 인용 · 논의로 충분.
- 빌드: BUILD_PASS, overfull 0, 미정의 참조 0, BibTeX 경고 0.

## 2026-11-04 19:50–20:05 KST — E1 baseline R 수집 protocol 초안 (`/cvpr-experiment`)

- 입력: `AG3S-VLA_CVPR2027_Experimental_Design.md`, `SOTA_COMPARISON_KO.md`, `AG3S-VLA_EXPERIMENT_RESULTS_KO.md` (commit `89b42459`), `papers/` PDF.
- 산출: `paper/experiments/configs/E1_SAFELIBERO_LIBEROSAFETY_BASELINE_PROTOCOL_20261104_KO.md` (`PLANNED`), `EXPERIMENTS.md` 계획 표에 P007 · P008 추가. **실험은 실행하지 않았다**(CPU `--dry-run`만).
- 확인한 것: 공식 `main_aegis.py`의 horizon은 policy 300 / 550 step + wait 20 (harness `--safelibero-step-budget main_aegis`와 같음; 기존 dev run의 기본값 `total_incl_wait`는 280 / 530). 기존 baseline 기록(BL-7 · BL-11 · BL-13 · SL7 D13)은 dev split 또는 비공식 조건이라 E1 R 칸에 넣지 않는다. SafeDojo · SafeVLA · WMPO는 π0.5 harness에서 실행 불가 → P only. LIBERO-Safety 원문 비교는 `pi05_libero_safety`(이 기계에 없음) · controller · cost 문제 해결 전 불가.
- 문헌 표 점검: SafeDojo Table 1–2와 LIBERO-Safety Table 3 L2 값은 PDF와 일치. `SOTA_COMPARISON_KO.md` §1에 Level II WMPO 행 누락(수정하지 않고 보고만 함). VLSA Table I의 OFT full-action 평균 CAR 5.7 %는 표 내부 불일치.
- 등록 형식 문제: manifest schema는 16-D 전용이라 Panda 7-D run을 등록할 수 없다(별도 형식 TBD).

## 2026-11-08 18:05 KST — 4. Experiments 전면 재작성 (standalone 빌드)

- 사용자 요청: `paper/evidence`의 실험 계획(E1–E7, RQ1–RQ4, P/R/TBD 규칙)과 CVPR 2026 논문 4편(`papers/cvpr2026/`: ACoT-VLA · AVA-VLA · ActiveVLA · ForceVLA2)의 실험 절 흐름을 반영해 4장을 전부 교체. 기존 `sections/04_experiments.tex`(IFG 개발 결과 중심)는 수정하지 않고 보존했다.
- 산출: `paper/latex/exp_section/04_experiments.tex` (4.1 Setup · 4.2 Main Results on Safety Benchmarks · 4.3 Task-Conditioned 3D Grounding · 4.4 Constraint Validity and Ablations · 4.5 Generalization, Efficiency, and Real-Robot Deployment; 표 6개, 그림 자리 2개), `build_experiments.tex`(공식 `cvpr.sty` review mode, section 4 · figure 4부터 번호), `references_experiments.bib`(SafeDojo · WMPO · OpenVLA-OFT · SAM 2, 각각 SafeDojo PDF 서지 목록과 SAM 2 PDF 머리에서 옮김), `build_experiments.pdf`.
- 수치 규칙: 문헌 보고값(P)만 표에 넣었다 — SafeLIBERO Level I/II (SafeDojo v2 Tab. 1–2, OpenVLA-OFT 계열), LIBERO-Safety L2 SR (원문 Tab. 3). 자체 재현(R)과 AG3S 칸은 모두 `\tbdc`/`\tbd{}` placeholder. **등록된 개발 결과(R008–R026)는 설계 문서 §1 규칙에 따라 표에 넣지 않았고**, 어느 칸의 후보인지 tex 상단 주석에 적었다. 벤치마크 정의(Level I/II, TSR·SSR·CAR·ETS, 1 mm 변위 규칙, AAG·TSA·FSHOA·HRI, L0–L2)는 VLSA · SafeDojo · LIBERO-Safety 원문 PDF와 E1 protocol 문서로 확인했다.
- 빌드: pdflatex + bibtex (TeX Live apt 설치, `TEXINPUTS=..:`), 6쪽(본문 5쪽 중간까지, 참고문헌 5–6쪽), undefined citation/reference 0, overfull 1건(2 pt 미만, Tab. 2). 전 페이지 PNG 시각 검토.
- 미수행: main.tex 병합, 기존 preamble에 `\tbdc` 추가, 실험 실행·수치 등록. 본문 길이(약 4.5쪽)는 수치 확정 후 압축 대상.

## 2026-11-08 18:40 KST — 4. Experiments v2: 분량 축소 + 목표 결과(빨간 글씨) 삽입

- 사용자 피드백: v1 분량 과다. CVPR 2026 참고 논문의 순서(4 → 4.1 Setup → 4.2 Main Results → 4.3 Ablation → 4.4 Analysis → Conclusion)로 재구성하고, 아직 측정하지 않은 결과는 사용자가 제시한 방향(LIBERO 단독 성공률은 π0.5보다 미세하게 낮음, 충돌률은 크게 감소, 다른 baseline 대비 약 10 point 충돌 감소, 일반화)에 맞춘 **목표값과 그 이유를 빨간 글씨로** 작성.
- 표기: `\proj{…}` = 투영된(미검증) 수치·설명(빨강), `\tbd{…}` = 미확정 설정(빨강), 검정 = 벤치마크 정의·protocol·문헌 보고값(†). 투영값은 VLSA Tab. I(같은 π0.5-LIBERO: CAR 17.1 % / TSR 57.8 %, AEGIS 68.9 % / 67.5 %)와 SafeDojo Tab. 1–2에 맞춰 잡았다. 어느 것도 측정값이 아니다.
- 산출: `paper/latex/exp_section/04_experiments.tex`(v2, 표 4개 · 그림 자리 2개), `build_experiments.pdf` 5쪽(본문 약 3.3쪽, 참고문헌 4–5쪽), overfull 0 · undefined 0. v1은 덮어썼다(기록은 이 로그와 이전 항목).

## 2026-11-08 19:10 KST — 4. Experiments v3 + 5. Conclusion 초안, 4–5장 3.7쪽

- 사용자 지시: 4장의 4개 세부 절(4.1–4.4)은 유지하고 내용만 수정; 각 TBD에 이상적인 예시 내용을 채움; trade-off는 숨기지 않고 보고; 4–5장을 3.5–4쪽 안에.
- 내용 변경: Q1을 "충돌 감소량과 task success 비용, 그 trade-off의 비교"로; 4.2에 TSR−SSR(충돌하며 성공) / CAR−SSR(무충돌 실패) 분해 추가; 4.4에 "Where it still fails" 문단(충돌·무충돌 실패의 원인 분류 → 5장 future work의 근거) 추가; 모든 `\tbd` 자리를 구체적 예시로 채우되 `\proj`(빨강)로 표시(로봇·카메라, 접촉 판정 센서, attention layer/head, dev/eval initial state 분할, GPU, 그림 구성).
- 5장 Conclusion 초안(`exp_section/05_conclusion.tex`): 요약 + "baseline toward collision-free execution" 포지셔닝 + 한계/future work 3가지(관측 한계·3D 표현 VLA 연구 인용, identity 신뢰도, 제약을 action 생성 단계로). 인용 추가: ActiveVLA(로컬 PDF), SpatialVLA·PointVLA(현 원고 PDF 서지에서 복사, 제출 전 1차 출처 재확인 필요).
- 빌드: overfull 0 · undefined 0. 4장 시작부터 5장 끝까지 약 3.7쪽(References가 4쪽 오른쪽 열 중간에서 시작).
- 디스크 변경 확인: `exp_section/` 파일은 이 세션의 sed/python 편집 외에 내용 변경 없음; 18:50 commit 3a14bf16(다른 세션)이 v2 파일들을 그대로 commit함.

## 2026-11-08 19:40 KST — Q1 표현 수정, 최종 실험 계획 문서 작성

- 4장: Q1을 "충돌을 줄이면서 task success를 얼마나 유지하는가"로, 4.2 · 5장의 "cost" 표현을 "preserved within … points"로 바꿈(사용자 지적). 빌드 재확인(overfull 0, 4–5장 약 3.7쪽 유지).
- `AG3S-VLA_CVPR2027_FINAL_EXPERIMENT_PLAN_20261108_KO.md` 작성(`PLANNED`): 원고의 `\proj` 165곳을 실험 ID(S0, E1a, E1b, E2/E3, E4-F, E5, E6, E7, SUP)와 산출물로 매핑; 공통 규약; 실험별 protocol; 우선순위 · ESTIMATED 시간; supplementary 자료 13개 표 + 영상; 리뷰어 예상 요청 15개와 사전 준비물; 표본 수 근거; 체크리스트. 실험은 실행하지 않았다.
- 19:55 KST 추가: 최종 실험 계획 §10 — 원고 Tab. 1–4 · Fig. 4 · 본문 수치를 표 그대로 옮겨 칸별 목표값 · 결과 파일 · 예상 메커니즘 · 결과가 다를 때의 원고 수정 규칙(판정)을 적음.
- 20:10 KST 추가: 최종 실험 계획 §11 투영 대 실측 대조표(35 항목, 실측·차이·판정·원고 수정 열 비움) · §12 외부 segmenter 비사용 · 실시간성 논거 7개(A–G)와 근거 실험 · 확신도 · 배치 권장.

## 2026-11-08 20:45 KST — 4–5장 최종 반영(B안), 1–3장 수정 요청 정리, Fig. 1 teaser v2 구성안

- 4.4 "Where it still fails"를 세 문장(perception-side / action-side 두 축, 세부는 supplement Tab. S9)으로 축소. 4.2의 "every link / forearm"을 3.4(palm · finger sphere만 제약)에 맞게 "both hands' collision spheres and the attached object"로 수정.
- 5장을 두 축의 future work로 재작성: perception(내부 표현과 장면 기하의 동기화; frozen VLA probing 연구 · 3D 표현 VLA 인용), action(생성 뒤 정제의 한계 → 제약을 반영한 chunk 생성, 6D grasp/place pose; "pursuing in ongoing work"). 빌드 overfull 0 · undefined 0, 4–5장 약 3.75쪽(References 4쪽 오른쪽 열 y 0.49).
- `paper/figures/fig01_teaser/TEASER_V2_SAFELIBERO_PLAN_KO.md`: SafeLIBERO Level I triplet(π0.5 / AEGIS* / AG3S) + AG3S 내부 3단 레이아웃, 장면 선택 기준, 자료 inventory(11-04 같은 장면 영상 · 11-08 6-panel 진단 렌더는 시안용), 제작 절차, caption 초안. 최종 실험 계획 §1.7 · §10.6 · §11 · §6 · §13 갱신.
- 1–3장 수정 요청은 사용자에게 위치와 함께 전달(직접 반영 예정).

## 2026-11-08 21:30 KST — Tab. 1 보고값 블록을 같은 checkpoint(VLSA · KNOWS)로 교체, 공식 코드 우선 규칙, 투영값 정합

- 사용자 판정: (a) Tab. 1 상단을 같은 π0.5-LIBERO checkpoint의 저자 보고값(VLSA Tab. I; KNOWS Tab. 1의 4 suite 평균 DERIVED)으로 교체, SafeDojo(OFT) 블록은 supplement로; 공식 코드가 있으면 재구현 대신 공식 코드(AEGIS = `main_aegis.py` submodule), 없는 것(KNOWS, multi-link)만 ∗.
- 보고값이 드러낸 논리 문제: VLSA · KNOWS 모두 safety layer가 π0.5보다 **TSR을 올린다**(57.8 → 67.5, 55.0 → 67.5). AG3S가 SafeLIBERO에서 TSR을 잃으면 SSR 최고 주장이 불가능하므로 투영을 TSR 66.0 / SSR 61.5 / CAR 87.5(Level I), 68.0 / 63.0 / 89.0(Level II)으로 고치고, "π0.5보다 미세하게 낮다"는 LIBERO 무장애물 열(95.4 vs 96.8)에만 남김. 4.2 문단 · Tab. 3 · 5장 · 계획 문서 §1.2 · §7 · §10.1 · §10.4 · §11 동기화.
- 빌드 PASS(overfull 0 · undefined 0). 4–5장 약 3.9쪽(References 4쪽 오른쪽 열 y 0.76) — 상한 4쪽에 근접, 그림 실물 삽입 시 Tab. 4 YOLOE 행·4.4 generalization 문장 축소 예정.
- 22:00 KST 추가: 4.2의 Naive(GT segmentation) 비교 문장 수정(투영 SSR 61.5 < 보고 63.1이므로 "넘는다" → "2 pt 차이, 특권 정보"). `paper/notes/RESULTS_NARRATIVE_20261108_KO.md` 작성: Tab. 1 여섯 단계 읽기 · Tab. 2–4 · Fig. 4 · 4.4 수치의 서술 논리와 리뷰어 반문 답, 결과가 투영과 다를 때 바뀌는 논리(§8), 분량 축소 권고(§9: Fig. 5 삭제 등 ≈ 0.7쪽), 1–3장 축소 의견(§10: 3.5 압축 · Alg. 1 supplement · Fig. 2 제거 권고).

## 2026-11-08 22:40 KST — 4장 축소 반영(≈ 0.6쪽), 1–3장 수정 요청 문서

- 4장 축소(사용자 판정, 논리 유지): Fig. 5 삭제(4.4가 Fig. 1 teaser v2 패널 (d) 참조), 4.1 Metrics + Implementation을 한 문단으로, 4.1 Benchmarks 나열 축약, 4.2 SafeLIBERO 문단 축약, Tab. 4에서 YOLOE · oracle 행 삭제(supplement), 4.4 generalization + run time 3문장. 빌드 PASS(overfull 0 · undefined 0). 4–5장 ≈ 3.6쪽(References 4쪽 오른쪽 열 y 0.25; 이전 3.9).
- `paper/notes/SEC1-3_REVISION_REQUESTS_20261108_KO.md`: Abstract · Intro · Fig. 1–3 · Related Work · Method 항목별(위치 · 현재 문장 · 요청 · 분류: 일관성/분량/그림), 용어 일치 점검표, 반영 순서. 이전에 구두로 전달한 요청(초록 수치, 기여 3 범위, multi-view, Fig. 2 caption, 3.4 유지)을 모두 포함. ◆ 판정 항목: Intro 1–2문단 절반, Fig. 2 삭제, 3.5 1/3 압축, Alg. 1 supplement(합계 ≈ 1.4쪽).
- 23:10 KST 추가: 4.4에 지시문 변경 scene pair 문장(RQ1 직접 증거)과 head · margin 민감도 한 문장(supplement Tab. S14) 복원 · 추가; "carried object" → "attached object"로 용어 통일(3.4와 일치). 계획 문서 §6(Tab. S14 · S15) · §7 · §10.5 · §11(36–38) 갱신. 빌드 PASS(overfull 0 · undefined 0), 4–5장 ≈ 3.7쪽(References 4쪽 오른쪽 열 y 0.36).
- 23:40 KST 추가: 실시간성 처리 확정(사용자 판정) — 4.4 run time을 "정책 추론 포함 같은 경계"의 비용 보고로 재작성(AG3S 490/620 ms vs π0.5+AEGIS 205 vs external 560, 단계 분해; 모두 투영), "one to two orders of magnitude more than an ellipsoid filter … a cost, not real-time capability"; 5장 Limitations에 비용 한계 문장 추가. 계획 문서 Tab. S11 형식 · §10.6 · §11(32–34) · §12 F · §12.1("왜 외부 segmenter를 안 쓰나" 답변 틀: extent · prompt · 일관성 · open-set/시간; 비용은 동등점) 갱신. 빌드 PASS.

## 2026-11-09 KST — Fig. 1 · Fig. 2 병합 결정과 구성안 v3

- CVPR 2026 4편(ACoT-VLA, AVA-VLA, ActiveVLA, ForceVLA2)의 그림 caption과 배치를 `papers/cvpr2026/*.pdf`에서 직접 확인. 네 편 모두 Fig. 1 = "기존 방식 vs 우리 방식" 개념 비교(수치 없음), Fig. 2 = 방법 구조 하나. 직전 답변(KNOWS · SafeDojo를 근거로 "CVPR 2026에는 개념 비교 그림이 없다")은 잘못된 근거였으므로 정정.
- 결정(사용자 동의): 기존 Fig. 1(teaser)과 Fig. 2(세 축 개념 비교)를 새 Fig. 1 한 장으로 병합. Fig. 2의 세 칸이 뼈대, Fig. 1 장면은 위쪽 한 줄(선택), 수치 막대 · 내부 4단계 · Fig. 2 흐름 줄 삭제. 현재 Fig. 3 → 새 Fig. 2. 추정 절약 ≈ 0.4–0.5쪽(시안 빌드 후 실측).
- 문서: `paper/figures/fig01_teaser/TEASER_V2_SAFELIBERO_PLAN_KO.md`를 v3로 개정(근거표, 세 축 ↔ 기여 · 절 · 표 대응, 레이아웃, 칸별 Prior baseline과 장면 기준, 자료 목록, 제작 절차, caption 초안, 원고 참조 변경 목록). `fig01_teaser/PLAN.md` · `fig02_overview/PLAN.md`에 포인터, SEC1-3 수정 요청 C1 · C2, 실험 계획 §6 · §13 갱신.
- 원고 tex는 아직 바꾸지 않음: Intro 10행 · Method 58행 그림 참조, 04_experiments 159행 "(Fig.~1, panel~d)", build_experiments `\setcounter{figure}{3}` → `{2}`는 병합 그림 반영 시 함께 수정(구성안 §8).
- 2026-11-09 추가: 분량 문서 갱신. 기존 원고 PDF의 절 제목 위치로 1–3장을 실측(≈ 6.13쪽: 제목 · 초록 · Fig. 1 0.93, Intro 1.87, Related 0.62, Method 2.72)하고 4–5장 standalone 3.75쪽과 합쳐 본문 ≈ 9.9쪽(사용자 보고 10쪽과 일치), 8쪽까지 ≈ 1.9쪽 축소 필요. `notes/SEC1-3_REVISION_REQUESTS_20261108_KO.md` §H에 3단계 예산표(1단계 ≈ 1.0: Fig. 1+2 병합 · Algorithm 1 · 3.5 · Intro 1–2문단; 2단계 ≈ 0.8; 3단계 ≈ 0.35, Tab. 2 이동은 마지막 수단) 추가. 실측으로 E1 0.25→0.17, E2 0.3→0.22 정정. `RESULTS_NARRATIVE` §9에 반영 완료 표시, §10의 "Fig. 2 삭제" 권고를 병합으로 정정.

## 2026-11-09 KST — Fig. 1 v3 자산: segmenter 결과 그림 발견, 최종 run에서 Ours · 위쪽 줄 렌더

- 외부 segmenter · ellipsoid 결과는 이전 컨테이너(11-01–11-03) run의 그림으로 `AG3S/ag3s/docs/figures/baselines/`에 남아 있음(원본 마스크 배열은 이 컴퓨터에 없음). `paper/tools/plotting/fig1_v3_assets.py --only crop`으로 10개 그림의 패널을 `paper/figures/fig01_teaser/v3/assets/crop_*.png`로 분리.
- 같은 스크립트 `--only render`로 최종 AG3S(pe123458) E1a run과 VLA 단독 run의 SafeLIBERO Level I task 2 episode 9(seed 7, initial state 9, 와인병)에서 위쪽 줄 frame과 칸 1–3 Ours 이미지 생성. 3D 점은 기록된 K · T_base_cam으로 투영, chunk 매핑은 `gtto_feature_scene.map_chunks`(solve_ms 일치 확인).
- 확인한 사실: chunk의 16×16 `attention_head` 지도는 점별 attention 값과 어느 방향으로도 상관 없음(|r| < 0.03) → 칸 1은 점별 값으로 그림. call 0의 seed 90개 중 29개가 와인병에 걸렸으나 target 표면에는 병 근처 점 0개; call 1부터 seed는 그릇에만.
- 배치 시안 `paper/figures/fig01_teaser/v3/mock_fig1_v3.png`. 구성안 §5.1에 자산 inventory 추가.
- 2026-11-09 추가: Fig. 1 v3 최종 제작 프롬프트 `paper/figures/fig01_teaser/FIG1_V3_FINAL_PROMPT_KO.md` 작성(목적, 병합 정의, 레이아웃, 칸별 1순위 · 대안 이미지 경로, 제작 · LaTeX 반영 절차, 정직성 제약, 완료 기준).

## 2026-11-09 12:08 KST — Fig. 1 v3 제작 · 원고 반영 (구 Fig. 1 + Fig. 2 병합)

- `FIG1_V3_FINAL_PROMPT_KO.md`대로 `paper/tools/plotting/cvpr_v3_fig1_compose.py`를 새로 만들어 `paper/figures/fig01_teaser/v3/fig01_v3.{pdf,png,json}`(위쪽 줄 포함, 6.875 × 2.889 in, 패널 9)과 `fig01_v3_notop.*`(6.875 × 1.699 in, 패널 6)을 생성. 두 변형 모두 `layout_audit` 위반 0, 최소 글자 7.0 pt, 그림 안 수치 0. sidecar에 패널별 원본 경로 · sha256 · crop · 출처 run/episode/chunk · raster 편집 · 인쇄 글자 목록 기록.
- 자료 유형: 위쪽 줄 · Ours 패널 = 기록된 run의 렌더(VLA 단독 `20261108T233915_4ef922024674_…vla_only`, AG3S `20261109T080310_a811331c11d8_…pe123458-C`, SafeLIBERO spatial Level I t2 e9, seed 7, init 9, 두 run commit 다름). Prior 패널 = 이전 컨테이너 baseline 문서 그림의 crop(LIBERO, 장애물 없음, 다른 장면).
- 원고: wrapper `figures/fig01_teaser/figure_v3.tex` · `figure_v3_notop.tex` 신설, `latex/main.tex`가 `figure_v3`를 입력하고 구 Fig. 2 줄 삭제, Intro 10행 참조 `fig:teaser`로 통합 · 예시 문장을 그릇 · 병으로 수정, Method 58행 · 실험 159행의 그림 참조 삭제(해당 장면이 새 그림에 없음), `build_experiments.tex` figure counter 3 → 2, `\bibliography`에 `exp_section/references_experiments` 추가.
- 빌드(pdflatex, 사용하지 않는 `algorithm`/`algpseudocode`는 scratch의 빈 stub): 12 → 11쪽, References 시작 10 → 9쪽, undefined reference · citation 0, 그림 번호 1 v3 · 2 pipeline · 3 held-out · 4 failure. 위쪽 줄 제외 변형도 11쪽 · References 9쪽.
- 상세 · 프롬프트와 달라진 점: `figures/fig01_teaser/TEASER_V2_SAFELIBERO_PLAN_KO.md` §5.2.

## 2026-11-09 13:26 KST — Fig. 1 v4: 같은 장면 비교 (figurelabs 시안 배치 + 실제 기록 frame)

- 사용자 판단: v3는 teaser로 부족(Prior가 다른 장면, Ours segmentation 거침). 사용자가 figurelabs로 만든 배치 시안(AI 생성 장면, 논문 사용 불가)의 구성을 실제 기록으로 다시 만듦.
- 같은 episode 확인(SafeLIBERO spatial Level I t2 e9, init 9, seed 7, 와인병): VLA 단독 · AEGIS · multi-link CBF · KNOWS 모두 step 86에 병 충돌(KNOWS는 과제 실패), AG3S pe123458 안전 성공. AEGIS detector 상자는 episode 내내 같고(cabinet · 접시) 병에는 상자가 없음(`…aegis…/method_calls/calls.jsonl`).
- `paper/tools/plotting/fig1_v4_assets.py`: 기록된 head depth · K · T_base_cam으로 픽셀을 3D로 올려 기록된 점 · voxel · sphere에 대응시키는 방식으로 Ours 4장을 다시 렌더(점찍기 → 픽셀 mask, "segmentation 보정"은 표시 방식만 바꾼 것), Prior 3장(AEGIS 상자, KNOWS end-effector ellipsoid)과 episode 끝 frame 생성. KNOWS 물체 ellipsoid는 회전이 기록되지 않아(`serve_method.py:316`) 그리지 않음.
- `paper/tools/plotting/cvpr_v4_fig1_compose.py` → `paper/figures/fig01_teaser/v4/fig01_v4.{pdf,png,json}`: 6.875 × 2.880 in, 패널 12, layout audit 위반 0, 최소 글자 7.0 pt, 수치 0. wrapper `figures/fig01_teaser/figure_v4.tex`(caption 초안). `main.tex`는 아직 v3 입력(전환은 사용자 결정).
- 주의: init 9는 구성 선택에 쓴 분할(init 5–14) 안. AEGIS run(commit 89b424595b57)과 AG3S run(a811331c11d8)은 commit 다름, 초기 상태 · seed 같음.

## 2026-11-09 13:50 KST — Fig. 1 v5: 지시 → 두 lane(Prior vs AG3S) storyboard

- 사용자 요청: v4가 뭉툭하고 덜 예쁨 → 사람-로봇 소통, 로봇 비전이 타겟 기하를 잡아 충돌 제약으로 바꿔 성공하는 흐름, 외부 segmentation 미사용의 간접 표현.
- `paper/tools/plotting/cvpr_v5_fig1_compose.py` → `paper/figures/fig01_teaser/v5/fig01_v5.{pdf,png,json}` (6.875 × 2.860 in, layout audit 위반 0, 최소 7.0 pt, 수치 0). 사람 아이콘 + 지시문 → Prior lane(frozen VLA + 점선 "external detector" 모듈 → AEGIS 상자 → "target surface not modelled" → KNOWS 손 ellipsoid → AEGIS 끝 frame) / AG3S lane(같은 frozen VLA → "its own attention" → attention·seed → 3D 점구름(target 표면 주황, 나머지 파랑) → contact-link 거리장 단면 → AG3S 끝 frame).
- 새 패널 두 개는 AG3S chunk_00165(call 6, obs 50)의 기록 배열을 직접 그림: `filtered_cloud`, `support_mask`, `target_free_removed/nontarget_kept_ijk`, `esdf_distance` vs `esdf_free_distance`(그릇 voxel: main 0, free 중앙 8.7 mm → 비워짐; 이웃 voxel: 둘 다 0 → 장애물 유지; 확인함). 단면 z = 0.062 m, 카메라 정렬(오른쪽 = +y, 아래 = +x).
- 둥근 모서리는 `mutation_aspect`로 인치 단위 원형 모서리, badge는 패널 폭 안으로 축약. wrapper `figures/fig01_teaser/figure_v5.tex`(caption 초안). `main.tex`는 여전히 v3 입력.

## 2026-11-09 14:26 KST — Fig. 2 (method pipeline): 사용자 템플릿에 실제 기록 삽입

- 입력: 사용자 템플릿 `paper/figures/figure02_method_pipeline.png`(AI 생성 배치, 1920 × 816, 가짜 큐브 · 공 장면). 원본은 수정하지 않음.
- `paper/tools/plotting/fig2_template_fill.py` → `paper/figures/fig02_method/fig02_method_pipeline.{png,pdf,json}`(2배 3840 × 1632, 6.875 in 기준 558 dpi). 사진 칸 10개를 SafeLIBERO t2 e9 AG3S run(pe123458)의 기록으로 채움: 입력 head + wrist RGB(obs 20), ① call 2 점구름 · target 점 · seed(3D), ② target 점(obs 30, 50)과 held 중 attached sphere 중심(obs 70–130)의 시간 추적, ③ call 6 거리장 단면 C1(esdf_distance) / C2(esdf_free_distance), ⑤ obs 110 손 sphere + 실행된 end-effector 경로(obs 85–139), S1–S5 obs 35/60/95/125/139.
- 기록과 맞지 않는 템플릿 글자 수정: "p40.5" → π0.5, "RGB + 2 wrists" → "RGB-D + wrist", S4 "release" → "lower onto plate", S5 "Retreat / move away" → "Success / bowl on plate, episode ends"(gripper가 끝까지 닫혀 있고 성공 시 episode 종료). 범례의 target / 이웃 구를 논문 색(주황 / 파랑)으로 변경.
- 남은 문제(사용자 판단): 템플릿 글자가 인쇄 크기 ≈ 4–6 pt(7 pt 하한 미달), 범례 "Frozen policy, depth, robot model: none" 의미 불명, "s.t. clearance ≥ −x₁" 표기, robot state 아이콘이 Panda가 아님, ④ stage chip(approach … retreat)과 기록된 HSR 라벨(pick/place, grasp searching/latched/closing/held) 불일치.

## 2026-11-09 14:51 KST — Fig. 2 모듈 ①②③: 장면 overlay 변형(B) 추가, A/B 비교

- `fig2_template_fill.py scene` → `paper/figures/fig02_method/fig02_method_pipeline_scene.{png,pdf,json}`. ①②③ 칸만 같은 기록 배열을 head 카메라 위에 그림(기록 depth로 픽셀을 3D로 올려 대응): ① obs 30 target 점 → 주황 mask, seed 금색, 나머지 관측 표면 옅은 파랑, ② obs 130 frame 위에 처음 grounding된 target 점과 단계별 중심(obs 30–130) 마커 + 점선, ③ obs 50 frame 좌 C1(그릇 포함 관측 표면 빨강) / 우 C2(그릇 주황, 나머지 빨강, 손 sphere 금색).
- 비교 그림 `paper/figures/fig02_method/fig02_modules123_compare.png`(위 A = 3D / 단면, 아래 B = 장면 overlay). 기본 변형(A)은 그대로 유지.

## 2026-11-09 14:58 KST — Fig. 2 B①: target 밖 seed 표시 제외 (사용자 요청)

- B① 장면판에서 target 점에서 2 cm보다 먼 seed를 그리지 않음(표시 선택). call 2 기록 seed 101개 중 95개 표시, 6개(병 4 · 서랍장 2, target까지 5.7–8.7 cm) 생략. 생략 수와 거리는 `fig02_method_pipeline_scene.json` `slots.p1`에 기록. caption에 "모든 seed가 그릇 위"라고 쓰지 않는다.

## 2026-11-09 17:44 KST — Fig. 2 ②: 관측 vs 기억 구분 패널 새로 생성 (사용자 요청, 기존 파일 유지)

- 이유: 기존 ②는 held 구간(obs 70–130)의 attached sphere를 obs 30, 50의 target 점과 같은 방식으로 그려, memory가 그릇을 계속 추적한 것처럼 보였음. 기록은 다름 — call 0–9(obs 20–65)는 state visible · association accepted(shift 0.35–1.26 mm), call 10–23(obs 70–135)은 grounding no_seed / no_cluster · state occluded(age 1–14) · exclusion source attached이고, `target_exclusion.geometry_version`이 observed_frame 9 · 222점(같은 sha1)으로 고정. 즉 obs 65에서 마지막으로 본 형상을 손에 붙여(attached_query_spheres 80개, r 17.3 mm) 들고 다니며, 다시 관측하지 않음.
- 새 패널: 관측 = 주황 실선(채움), 기억(관측 아님) = 주황 점선 외곽.
  - B 장면판(`r_p2_mem_scene`): 2 × 2 frame. 위 obs 30, 50(target 점 8 mm 이내 depth 픽셀), 아래 obs 90, 130(attached sphere 투영 외곽 점선). 네 frame 모두 768 px 확대 frame에서 같은 330 px 창. 행 앞 tag(채움 = 관측, 점선 = 기억), 행 안 화살표 = 시간.
  - A 3D판(`r_p2_mem`): obs 30, 50 target 점 채움 + obs 90, 130 attached sphere 중심 속 빈 원 + 실행된 end-effector 경로(obs 30–130) 점선. 들린 그릇이 잘리지 않게 범위 확대.
- 출력(새 파일): `paper/figures/fig02_method/fig02_method_pipeline_mem.{png,pdf,json}`, `fig02_method_pipeline_scene_mem.{png,pdf,json}`, 비교 `fig02_module2_compare_mem.png`(before / revised × A / B). 실행 `fig2_template_fill.py abstract-mem scene-mem`. 기존 `fig02_method_pipeline{,_scene}.*`, `fig02_modules123_compare.png`는 md5 동일 확인.
- 남은 점: 템플릿 캡션 "track candidates over time"은 새 패널 의미(가려져도 마지막 형상 유지)와 어긋남. sphere 반경 17 mm 때문에 점선 외곽이 그릇보다 큼(기록 그대로).

## 2026-11-10 KST — Fig. 2 ④ task stage recognition: hidden state + gripper feedback 도식 (사용자 요청, `*_mem` 파일에만)

- `draw_m4()`가 템플릿 ④ 카드 내부(템플릿 px x 1383–1612, y 150–421)를 배경색으로 덮고 다시 그림. `fig02_method_pipeline_scene_mem.*`과 `fig02_method_pipeline_mem.*`에 적용. 기존 `fig02_method_pipeline{,_scene}.*`는 md5 동일.
- 내용(원고 03_method.tex HSR 절과 `AG3S/ag3s/stages/subtask_probe.py` 기준): Frozen VLA π0.5의 VLM backbone(PaliGemma) layer 4 → hidden state f_t(prefix K/V: text 평균 ++ image 평균, 512-d) → linear probe. gripper feedback(측정 opening → grasp state z_t)이 classifier에 합류 → pick / place / home → current stage s_t. 템플릿의 5개 stage chip(approach … retreat)은 원고 label 3개로 바꿈.
- 도식일 뿐임: 이 SafeLIBERO run의 call 기록에는 probe 출력이 없음(grasp latch searching / latched / held만 기록). 그래서 어떤 stage도 "현재"로 강조하지 않음. caption에 이 episode의 probe 출력처럼 쓰지 않는다.
- 남은 점: ④ 안의 작은 글자(10–11 px / 1920 px)는 인쇄 ≈ 2.6–2.9 pt. 템플릿 하단 "gripper feedback z_t (open / hold / placed)"는 원고 z_t ∈ {free, held, uncertain}과 다름.

## 2026-11-10 KST — Fig. 2: "palm–target distance" 제거 (사용자 요청, `*_mem` 파일에만)

- 템플릿의 "palm–target distance" chip과 그 연결선(gripper feedback chip 아래 → ④ 아래 화살표)을 측정한 배경색(254)으로 덮음(`PALM_PATH`, 템플릿 px 6개 box, sidecar `text_edits.removed_palm_target_path`). gripper feedback chip과 ③으로 가는 화살표는 유지. 이로써 ④로 들어오는 바깥 화살표는 없음(gripper feedback은 ④ 안에 그려져 있음). 기존 `fig02_method_pipeline{,_scene}.*`는 md5 동일.

## 2026-11-10 00:1x KST — HSR + gripper feedback 결합: Method · Supplementary draft (dev-38, 사용자 요청, `/cvpr-write`)

- 새 파일: `paper/notes/HSR_GRIPPER_FUSION_DRAFT_20261110.md` (`DRAFT`). `latex/sections/*.tex` · `supplementary/supplementary.tex` 는 고치지 않았다.
- 근거: offline 기록 재생 G174 (`outputs/2026-11-10_HSR6_gripper_재생/replay.json`, script `experiments/segmentation/hsr6_gripper_replay.py`). 40 SafeLIBERO dev episode, 정답은 sim 위치 근사, closed loop 아님.
- 내용: (1) Method 문단 1A — `PLANNED`, closed loop 채택 · 검증 전에는 넣지 않는다. (2) `04_experiments.tex` SafeLIBERO 문장 수정안 1B. (3) supplementary 새 절과 표 3 개 (LaTeX). (4) 그림 · 영상 목록. (5) 제안 ID C021 · R027 · EXP-H1 (등록하지 않음).
- 기록과 어긋남 발견: `04_experiments.tex` 의 "sub-task probe is replaced by the gripper rule because LIBERO demonstrations carry no phase labels". Panda probe 는 sim label 로 학습했고 (SL6-B), 지금의 gripper 규칙은 사용자 판정 P0m 에 따른 선택이다. 수정은 하지 않고 draft 에 제안만 했다.
- 용어: 원고 규칙에 따라 "sub-task probe" 로 썼다.
- manifest validator: 미실행. `python3 tools/analysis/validate_manifests.py` 가 `ModuleNotFoundError: jsonschema` 로 실패했고, checkout 의 venv 에도 jsonschema 가 없다. R027 은 미검증이다.

## 2026-11-10 KST — Fig. 2 ② caption 변경 (사용자 요청, `*_mem` 파일에만)

- "track candidates over time" → "keep the target's identity over time". 템플릿 px (786, 302, 990, 320)을 inpaint 후 Nimbus Sans Narrow 21 px로 다시 씀(sidecar `text_edits.module2_caption`). 새 ② 패널(관측 = 실선, 가려진 동안 = 마지막 형상 유지)과 의미를 맞춤. 기존 `fig02_method_pipeline{,_scene}.*`는 md5 동일.

## 2026-11-10 KST — Fig. 1 v6 teaser: 축별 비교 + 다른 장면 gallery (사용자 요청)

- 배경: 사용자가 figurelabs mock 2장(`fig01_teaser/Robot-Safety-Teaser-Figure-image-{1,4}.png`)의 스타일(사람–로봇 소통, Prior vs ours)은 좋지만 첫 페이지 teaser로는 내용이 빈약하다고 판단. mock은 AI가 그린 장면이라 쓰지 않고, 같은 구성을 기록 frame으로 다시 만듦. image-4의 "dialogue / clarify" 요소는 AG3S 기능이 아니라 넣지 않음(로봇 아이콘은 frozen VLA 표시용).
- `paper/tools/plotting/cvpr_v6_fig1_compose.py` → `paper/figures/fig01_teaser/v6/fig01_v6.{pdf,png,json}`, 6.875 × 3.40 in, 최소 글자 7 pt, layout audit 위반 0.
  - 위: 사람 아이콘 + 명령 말풍선 → frozen VLA(로봇 아이콘), 색 key.
  - 가운데(t2 e9, init 9, seed 7, 둘 다 obs 110): Prior hero(AEGIS, detector box, 병 넘어짐) | Identity / Extent / Permission 축별 ✗/✓ thumbnail 짝 | AG3S hero(장애물 파랑, 들고 있는 그릇 보라, 손가락 금색, 병 그대로). Prior thumbnail = AEGIS box(병 box 없음, 서랍장+그릇을 한 box), KNOWS 손 ellipsoid. AG3S thumbnail = v4 asset r_identity / r_extent / r_permission.
  - 아래 gallery: 같은 init·seed에서 VLA 단독 충돌 + AG3S 무충돌 성공인 episode 4개 — (t0 e9 wine bottle), (t2 e6 moka pot), (t2 e8 milk), (t3 e5 book). 왼쪽 = VLA 단독 collide_t + 40 frame, 오른쪽 = AG3S 마지막 frame. 선택 조건은 스크립트에서 assert(collided / not collided / official_success, init·seed 일치)로 확인하고 sidecar에 run · npz sha256 · frame 기록.
  - 선택 근거 표(이번 세션 계산, spatial LI t0–t3 e5–14, 40 episode): VLA 단독 collided ∧ AG3S safe success인 episode는 (0,9), (2,5), (2,6), (2,8), (2,9), (2,11), (2,12), (2,14), (3,5). gallery는 장애물 · task가 겹치지 않게 4개 고름. 예시이며 비율이 아님(caption에 명시).
- wrapper `paper/figures/fig01_teaser/figure_v6.tex`(caption 포함). `main.tex`는 아직 v3를 씀(바꾸지 않음).

## 2026-11-10 KST — Fig. 1 v6: 원고 템플릿 크기에 맞춤 (사용자 요청)

- 높이 3.40 → 3.01 in(hero 1.56 → 1.32, gallery 0.70 → 0.60, 위 band 0.36 → 0.33). 폭은 `\textwidth` 6.875 in 그대로. 최소 7 pt, layout audit 위반 0.
- caption 6줄 → 4줄(`figure_v6.tex`): 축 설명을 한 문장씩으로 줄이고, gallery가 선택한 예시이며 비율이 아님을 유지.
- 확인 빌드(scratchpad 복사본 `main_v6.tex`, `main.tex` 변경 없음, pdflatex + bibtex, undefined 0): 첫 페이지에 abstract 전체 + Introduction 시작이 들어감(v3 원고는 abstract 11줄만 들어감). 전체 11쪽으로 v3와 같음. 본문이 9쪽까지 가는 것은 v3에서도 같음(teaser와 무관).

## 2026-11-11 KST — 원문 보고값 비교 · 관련연구 수정안 (사용자 요청)

- 사용자 결정: SafeLIBERO 비교는 각 원문의 보고 지표로 한다. baseline 재구현 · 재실행 없음 (마감 11-17 20:59 KST).
- 새 문서: `paper/notes/PAPER_REVISION_REPORTED_COMPARISON_20261111_KO.md` (수정 계획, 원고 tex 는 고치지 않음). 대상 PDF `paper/AG3S_ … (5).pdf` 의 줄 번호 기준.
- 원문 확인 (PDF 직접): VLSA (+ supplement S1–S4), KNOWS, CFM, multi-link, VLPSA, SafeDojo, SafeHarness, DetAug, NUDGE · OmniGuide · Lan-o3dp (SafeLIBERO 아님). 공식 충돌 기준 = `benchmark/SafeLIBERO/main/main_aegis.py:370` (1 mm, L1, 매 step).
- 변환 script `paper/tools/analysis/reported_safelibero_metrics.py` → `paper/evidence/reported_baselines_20261111/` (json, md). 옮긴 표 전부를 원문 인쇄 평균과 assert 대조 — 통과. 새 [D]: AEGIS level 별 (VLSA supplement, full action) L-I TSR / CAR 63.8 / 65.4, L-II 71.2 / 72.4.
- 그림 `paper/tools/plotting/reported_safelibero_scatter.py` → `reported_tsr_car.png` (계획용, 원고용 아님).
- 발견: CFM 의 π0.5 · AEGIS 행 = VLSA 평행이동 supplement 값 (재실행 아님으로 보임). CFM · VLPSA 는 GT 장애물 정보 사용. Multi-link 가 공식 1 mm · GT 없음 · 1,600 ep 로 우리 조건과 가장 같음. 원고 l.421–426 의 분할 기술 (5 dev / 45 report) 이 실제 결정 (5–14 / 15–18 / 19–49) 과 다름.
- `papers/` 에 SafeHarness (2609.20822v2) · DetAug (2609.18395v1) PDF 추가.

## 2026-11-11 KST — 수정안 §8 결정 반영 · §10 실험 구성 추가 (사용자 요청)

- 사용자 결정: D3 AG3S GT oracle 행 넣음 · D4 통제 baseline 2 개 삭제 · D5 VLPSA 는 Related Work 논의만. D1 · D2 는 미정 (재설명 요청).
- `PAPER_REVISION_REPORTED_COMPARISON_20261111_KO.md` §10 추가: CVPR 2026 논문 (ACoT · AVA · ActiveVLA · ForceVLA2) 실험 흐름 확인 — 주 비교 표는 원문 보고값 나열이 관행 (ACoT Tab. 2 "*" = 직접 재현 행만 표시). Q1–Q4 + 분석으로 재구성, 짝 비교 vs 보고값 비교 구분, 타당성 체크리스트 7 항, ablation 후보 4 행 (flag 확인 필요), GPU 예산 ≈ 46–51 h.
- 확인: E2 (grounding) · E5 (ablation) · E7 (실로봇) · E1b (LIBERO-Safety) 는 계획 문서상 PLANNED (미실행). 원고 Tab. 2 · Tab. 3 · Fig. 3 · §4.4 수치는 측정 근거 없음 → 새 결정 D6 (동결) · D7 (미실행 실험 처리) 추가.

## 2026-11-11 KST — 수정안: 사용자 방침 반영 (모든 실험 실행 · 문헌 배치 중심)

- 사용자 방침: 계획한 실험은 모두 실행하며 "아직 안 했다" 는 이유로 원고에서 빼지 않는다. 수정안의 삭제 권고 (실로봇 · Tab. 3 · LIBERO-Safety) 를 철회, "측정값으로 교체" 로 바꿈 (§0 · §7 P0-1 · §8 D7 · §10.3).
- 새 절 §A: 문헌 배치표 (논문 → Intro / Related Work / Tab. 1 / 기타, 역할), 원고에 넣을 capability 비교표 초안 ("정보를 어디서 가져오는가", 원문 사실만 — 넣기 전 칸별 재대조 필요), 실험별 비교 대상 문헌.

## 2026-11-11 03:xx KST — 계획 변경을 계획 · 결과 문서에 반영 (사용자 승인)

- `doc/active/experiments/AG3S_PLAN_SCHEDULE_20261110_KO.md`: §1 변경 요약, §2 baseline 보고 run 행 → 보류 + 주말 AG3S 실험 행, §5 9a, §8 기록.
- `AG3S-VLA_CVPR2027_FINAL_EXPERIMENT_PLAN_20261108_KO.md`: §0.0 계획 변경 표 (이전 내용보다 우선) + §1.2 · §3.2 · §10.1 머리 주석.
- `AG3S-VLA_EXPERIMENT_RESULTS_KO.md`: §1.1 같은 checkpoint 원문 보고값 (P, level 별 · 합산), §3.1 R 표를 π0.5 anchor · AG3S · AG3S GT oracle 3 행으로.
- 수정안 문서에서 "측정값으로 교체" 문구 삭제 (사용자 지시; 실험은 주말).
- 2026-11-11 03:4x: 사용자 지시 — baseline port 보고 run 은 보류가 아니라 계획의 맨 마지막 (AG3S 실험 뒤). 계획서 §1 · §2 · §8, 최종 실험 계획 §0.0, 결과 기록 §3.1, 수정안 D2 반영.
- 2026-11-11 04:xx: 최종 확인 — 최종 실험 계획에 §0.0.1 (변경된 분할 · 표본 · 순서, 선행 조건 · 미확인 항목) 추가, §4 · §9 에 대체 주석. 발견: E5 분할 (5–14) · 표본 불일치, GT oracle 판 미등록, LIBERO-Safety AG3S arm H2 미해결, 실로봇 arm 구성 (generic · external) 미정, 주말 48 h 에는 E1a + LIBERO + ablation 만 들어감.
- 2026-11-11 04:xx: ablation 정정 — §0.0.1 · 수정안 §10 에서 변형을 4 개로 줄였던 것을 철회, Tab. 2 의 8 행 (계획 §1.5 7 변형 + Stage) 모두 유지, 분할만 보고 분할로. GPU 추정 56–61 h.

## 2026-11-13 17:5x KST — Tab. `tab:safelibero` 의 π0.5 단독 행을 SDM 1/10 측정값으로 (사용자 지시, sdm-f6)

- 사용자: "pi0.5 돌린 것을 논문의 실험 table 에 반영 … 나중에 풀버전을 측정하면 교체".
- `paper/latex/exp_section/04_experiments.tex` 하단 블록 '$\pi_{0.5}$ alone' 행: 투영값 (\proj, 빨강) → 측정값 (\pend, 주황 = 측정 · registry 미등록), 행 이름에 $^\ddagger$, caption 에 1/10 부분 집합 · init 을 성공 예상으로 고름 · 전체 1,600 ep 로 교체 예정을 적음.
- 값 (%, suite 동일 가중 = 각 cell 20 ep 이라 pooled 와 같음): LIBERO SR 98.0 (196/200) | L-I TSR 67.5 (54/80) · SSR 13.8 (11/80 = 13.75) · CAR 15.0 (12/80) | L-II TSR 72.5 (58/80) · SSR 30.0 (24/80) · CAR 30.0 (24/80).
- 자료 유형: [M] closed loop, `tab1sdm-pi05-*` 12 run (`/mnt/dev/AG3S_VLA-storage/results/benchmark/20261113T*_110ae6775e48_tab1sdm-pi05-*`), HEAD 110ae677, seed 7, main_aegis step budget, 공식 SafeLIBERO 충돌 정의 (장애물 변위 > 1 mm). 분할 `doc/active/experiments/PAPER_TENTH_SPLIT_20261113.json`. cell 별 표: 계획서 §9.39, 그림 `AG3S/ag3s/docs/figures/paper-tab1-tenth-sdm-20261113.png` (+ `.json`).
- 주의: init 을 성공 예상으로 고른 부분 집합이라 TSR 이 원문 보고값 (KNOWS 원문 π0.5 L-I 55.0 · L-II 57.8) 보다 높게 나올 수 있다 — 단독 보고 금지, 전체 측정으로 교체. 같은 절의 본문 단락 (\proj, '57.8 % · 18.6 % · 44.3 points') 은 투영 문장으로 그대로 두었다 (표와 숫자가 다름 — 전체 측정 후 함께 고침). pdflatex 가 이 서버에 없어 컴파일은 확인하지 못했다.
