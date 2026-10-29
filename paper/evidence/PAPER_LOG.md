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
