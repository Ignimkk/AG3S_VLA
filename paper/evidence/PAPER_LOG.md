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
- 새 실험·새 측정 없음. V5 수치(hurdle 12/14 등)는 미등록 자료라 본문에 쓰지 않았다.

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
  GD (V7 off vs V7g on 19 쌍). 근거 [V7G_FACTS](../notes/V7G_FACTS.md). 수치는 미등록이라 본문에 쓰지 않음.
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
