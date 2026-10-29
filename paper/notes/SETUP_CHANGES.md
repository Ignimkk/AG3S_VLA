# 논문 Workspace 수정 내역

## Step 1

수행 시각: 2026-10-05 18:59 KST

| 대상 | 수정 내용 | 이유 |
|---|---|---|
| 준비 원문 §1·§22·§23 | Claude 설정을 Root `.claude/`로 통일 | 기존 설정과 중복되는 `paper/.claude/` 방지 |
| 준비 원문 §6·§23 | 상세 positioning은 `notes/PAPER_STORY.md`, 상세 규칙은 scoped rule로 배치 | 사용자 요청인 `CLAUDE.md` 200자 제한 유지 |
| 준비 원문 §7·§22 | `/paper/...`를 `/mnt/dev/work/paper/...`로 수정 | 실제 workspace와 다른 절대 경로 바로잡기 |
| 준비 원문 첫머리 | 공식 규정과 일정이 미검증인 준비 초안임을 표시 | 오래된 정책이나 날짜를 현재 확정 규정으로 취급하지 않기 |
| Root `CLAUDE.md` | 짧은 진입 안내로 변경, 원문 전체는 `.claude/project-context.md`에 보존 | 기존 프로젝트 목적·환경·역할 정보 보존 |
| `paper/CLAUDE.md` | 짧은 논문 진입 안내 생성 | 논문 폴더에서 시작할 때 작업 기준 안내 |
| `.claude/rules/cvpr-workspace.md` | `paper/**` 범위의 논문 작업 규칙 생성 | 연구 코드 읽기 전용, provenance, 단계별 작업 원칙 명시 |
| `.claude/ag3s-rules.md` | 연구 구현 규칙과 논문 작업 규칙의 적용 범위 명시 | 연구 실행 로그와 논문 준비 기록의 충돌 방지 |
| `.gitignore` | `!/paper/` 예외 추가 | 기존 whitelist에서 제외된 논문 문서를 버전 관리 대상에 포함 |
| `paper/` | 기본 폴더와 README·계획 생성 | 후속 Step의 작업 위치 확정 |

## 검증

- Root `CLAUDE.md`: 181자, 공백·줄바꿈 포함.
- `paper/CLAUDE.md`: 156자, 공백·줄바꿈 포함.
- 기존 Root `CLAUDE.md`와 보존한 `.claude/project-context.md`의 bytes 일치 확인.
- Markdown 내부 참조 경로 및 기본 폴더 확인.
- Git whitespace 검사와 `paper/` ignore 예외 확인.

CVPR 날짜·정책의 사실관계, 최신 Author Kit, 선행연구 목록은 Step 2 이후에 검증한다.
기존 연구 코드·실험 결과·research log는 변경하지 않았다.

## Step 2 — 공식 자료 검증과 규정 문서 구성

수행 시각: 2026-10-05 19:21 KST

| 대상 | 수정·보완 | 이유 |
|---|---|---|
| `guidelines/` | 9개 문서 생성, `SOURCES.md`에 13개 공식 출처 기록 | 정책별 상태와 provenance 명시 |
| 원문 §4·§7 | 2027 Guidelines/Kit 미공개 단정을 `UNKNOWN`과 관찰 결과로 수정 | URL 404와 repository 표시는 전체 공개 여부의 증명이 아님 |
| 원문 §5 / DEADLINES | 날짜 일치 확인, KST 계산 가정과 시각 미확인 상태 추가 | AoE 날짜와 실제 timestamp를 구분 |
| 원문 §10 | 저자 추가·삭제 마감과 reviewer service 보완 | 2027 공식 profile 안내 및 CFP 반영 |
| 원문 §12 | 개선 method 결과·corrected main PDF의 supplementary 제한 추가 | 2026 reviewer FAQ의 provisional 기준 반영 |
| 원문 §13 | 모든 URL 금지가 아니라 content 확장·review 우회 금지임을 보완 | 2026 FAQ 예외를 정확히 유지 |
| 원문 §14 | conference/review LLM 계획과 author-side 정책 구분 | CFP의 범위를 과도하게 해석하지 않기 |
| 원문 §15 | rebuttal의 1페이지에 reference·figure가 포함됨을 명시 | Main-paper reference 예외를 rebuttal에 적용하지 않기 |
| README·계획·scoped rule | Step 2 완료 상태와 최신 정책 문서 읽기 순서 반영 | 다음 Step에서 준비 원문의 예시를 확정 규정으로 읽지 않기 |

### 검증

- 9개 문서의 존재, local reference와 source anchor 확인.
- 원문 날짜와 CFP·Dates 공통 항목 일치 확인; 4개 KST 참고값 계산 확인.
- 2027 확정/2026 fallback/미확인 항목 및 `PROJECT_POLICY` 구분 확인.
- Root와 `paper/CLAUDE.md`는 기존 내용 그대로이며 각각 181자·156자 유지.
- Git whitespace와 새 문서의 trailing whitespace 확인.

2027 세부 정책과 kit은 `UNKNOWN`인 항목이 남아 있으며 제출 전에 재확인해야 한다.
Evidence 구조·manifest·research story 작성은 Step 3에서 진행한다.

## Step 3 — Evidence 구조와 연구 계획

수행 시각: 2026-10-05 19:48 KST

| 대상 | 수정·구성 | 이유 |
|---|---|---|
| `evidence/` | Claim·experiment·result·baseline·metric·artifact·log 7개 문서 | 주장부터 원자료까지 추적하고 미측정 항목 구분 |
| `notes/` | Story·novelty·contribution·related work·reviewer question·TODO 6개 문서 | 후보·가설·검증된 결과를 혼동하지 않기 |
| `experiments/manifests/` | EXP001과 운영 README | 자료의 dimension·provenance·scope·검증 상태 기록 |
| `experiments/configs/` | JSON Schema 2020-12와 비활성 candidate 예제 | 14-D와 provenance 없는 등록을 구조적으로 구분 |
| `experiments/summaries/` | Source audit | 96 run의 실제 배열 header·metadata·완료 자료와 hash 기록 |
| `tools/analysis/validate_manifests.py` | 읽기 전용 semantic validator | Source drift·짝지은 run·실제 dimension·verify key와 값 확인 |
| 원문 §17 | whitelist를 상태 registry로 변경 | status·dimension·provenance·scope 표시 |
| 원문 §18·§26 | Attention observation과 desired outcome을 연구 가설로 명시 | 아직 없는 safety 개선·성능 유지 결과를 주장하지 않기 |
| README·계획·scoped rule | Step 3 완료와 Evidence 읽기 순서 반영 | 후속 작성·검증 workflow에 사용 |

### 검증 결과

- T39의 실제 16-D 기록 96 run / 48 pair / 24 episode source audit 통과.
- 등록한 R001–R003의 verify pointer·값·denominator 확인; paired table와 analysis 일치.
- 정상 manifest와 candidate 예제 처리 확인.
- 14-D 선언, 14-D shape, provenance 없는 위장 등록, candidate의 evidence 사용,
  `numbers` 밖 pointer, source와 다른 값, source hash 불일치의 7개 negative case 거부 확인.
- Evidence/notes의 local reference와 ID 연결, JSON schema, Python 구문, whitespace 확인.
- Root `CLAUDE.md` 181자, `paper/CLAUDE.md` 156자 유지.

T40/T41/T42 후속 자료와 PDF 내용은 아직 논문 evidence/citation으로 검증하지 않았다.
Novelty·main safety claim·generalization·runtime·hardware 성과는 미확정이다.
공식 template 도입·그림 생성·skill 작성은 후속 Step에 남아 있다.

## Step 4 — Claude 작성·검증 workflow

수행 시각: 2026-10-05 20:43 KST

| 대상 | 수정·구성 | 이유 |
|---|---|---|
| `.claude/skills/` | CVPR skill 7개 생성 | 작성·Evidence·protocol·figure·인용·review·compliance의 작업 구분 |
| `.claude/agents/` | Literature/evidence/reviewer의 3개 읽기 전용 역할 | 독립적인 보고와 parent writer의 파일 소유권 분리 |
| `notes/CLAUDE_WORKFLOW.md` | 호출 예시·Evidence tracking·역할 입력/출력·audit 상태 기록 | 정의 생성과 연구/검토 수행을 혼동하지 않기 |
| Skill/agent 운영 | Tool whitelist, `model: inherit`, parent validator 결과 전달 | 연구 코드/결과 수정·실험 실행·중첩 호출 방지와 미검증 항목 명시 |
| 정책 문서 2개 | Step 3의 Evidence·Step 4의 skill을 생성 예정에서 실제 구성 상태로 수정 | 오래된 workspace 안내 바로잡기; 공식 정책 확인 시각은 유지 |
| 원문 §22·§23·첫머리 | 실제 7개 skill·3개 역할, audit의 NOT_CHECKED 및 현재 정책 기준 추가 | 개념 예시와 실행 가능한 정의의 차이 명시 |
| README·계획·TODO·PAPER_LOG·scoped rule | Step 4 완료·workflow 읽기 순서·다음 Step 5 반영 | 다음 단계의 작업 기준 연결 |

### 검증 결과

- skill-creator의 quick validator: CVPR skill 7개 모두 PASS.
- Agent YAML의 지원 field·unique name·tools·model 검증 PASS.
- 3개 검토 역할에 Write/Edit/Bash·중첩 agent 호출 도구를 부여하지 않았다.
- 생성한 11개 workflow 파일의 Markdown local link 55개 검증 PASS.
- EXP001 semantic validator 재실행 PASS: 등록 범위·R001–R003 유지.
- 변경한 문서의 local link·whitespace·Git diff 검증 PASS.
- 기존 settings/hook·연구 skill 4개·AG3S 역할 3개의 bytes 보존 확인.
- Root `CLAUDE.md` 181자, `paper/CLAUDE.md` 156자 유지.
- 공식 Claude Code skill/subagent 문서의 기본 형식과 로컬 CLI 2.1.282를 확인했다.

실제 Claude 모델 호출·agent delegation은 검증하지 않았다. 원고·최종 PDF가 없으므로
review/compliance 실행 성공이나 submission readiness를 보고하지 않는다.
기존 연구 코드·로그·raw 결과를 변경하지 않았고 새 실험·figure·LaTeX build는 실행하지 않았다.

## Step 5 — Template와 원고·artifact 기반

수행 시각: 2026-10-05 21:08 KST

| 대상 | 수정·구성 | 이유 |
|---|---|---|
| `template/` | 공식 2026 tag 원본 14개 파일·PROVENANCE·README | 2027 미확인 상태를 유지하면서 reproducible fallback pinning |
| `latex/`, `supplementary/` | 원본 style/ bst, 별도 main·supplementary, 7개 English section·metadata·pending bibliography | 빌드 가능한 draft 기반과 공개/내부 provenance 분리 |
| `figures/`, `tables/`, `algorithms/` | PLAN·native LaTeX·sidecar; F02/A01 conceptual, T00 descriptive, 나머지 placeholder | 미측정 성과를 시각 자료로 위장하지 않기 |
| `tools/latex/build.py` | Source/kit/style/manifest/artifact 검증 후 별도 PDF build | Source drift·잘못된 T00 count를 작성 단계에서 확인 |
| Main/supplementary entry | T1 font encoding, review hyperref option | 초기 XeTeX TU font 대체와 supplementary review warning 해결 |
| `notes/MANUSCRIPT_MAP.md`·build audit | Section·claim·artifact map과 실제 build/PDF 검증 범위 | 계획·draft·검증된 evidence·최종 준비 판정 구분 |
| `.gitignore` | 논문 build/cache/intermediate 제외 | 연구 source와 원본/working 파일 보존, generated 파일 분리 |
| 원문 §1·§8·§19–§21 | Tagged kit의 실제 file 구성, pinning·year·bibliography·supplementary 보완, safe 성과 예시를 pending으로 수정 | 현재 제공한 구조와 미입증 안전성의 과장 방지 |
| Guidelines·Evidence·README·계획·TODO·PAPER_LOG | 부분 재확인과 실제 artifact 상태·Step 5 완료 반영 | 오래된 예정 안내 바로잡기 |

### 검증 결과

- 공식 원본 14개 파일과 두 working style/bst 세트 SHA-256 동일성 확인.
- Existing manifest validator와 T00 source hash·exact pointer·값·분모 검증 PASS.
- Main 3페이지·supplementary 1페이지 별도 PDF build PASS; optional placeholder 7개 cached build PASS.
- 초기 font 대체 수정 후 실제 Times 계열 font·letter page·review 표시 확인.
- Main/supplementary 전체 페이지를 시각 검토하고 figure·algorithm·table의 column fit를 확인했다.
- PDF 텍스트·metadata·font embedding·미측정 표시·원자료 경로/기관/저자 노출을 제한된 범위에서 검사했다.
- 전체 local link·JSON·Python 구문·whitespace와 기존 `.claude/` 보존 검증 PASS.
- Root `CLAUDE.md` 181자, `paper/CLAUDE.md` 156자 유지.

잔여 compiler encoding 메시지와 inspection 범위는 [STEP5_BUILD_AUDIT](STEP5_BUILD_AUDIT.json)를 따른다.
임시 compiler/PDF 도구는 `/tmp/`에서 사용했으며 기존 연구 venv·시스템 package를 수정하지 않았다.
실험·새 measured figure·bibliography 내용 검증·원고 review·submission audit는 수행하지 않았다.
다음 Step 6에서 submission checklist와 전체 workspace 검증을 진행한다.

## Step 6 — 제출 checklist와 전체 구성 검증

수행 시각: 2026-10-06 00:01 KST

| 대상 | 수정·구성 | 이유 |
|---|---|---|
| `submission/` | Checklist 5개·FINAL_AUDIT·사용 안내·camera-ready 단계 안내 | 실제 검사 근거·미완료 항목과 단계별 책임 명시 |
| `tools/analysis/validate_workspace.py` | Config·link/anchor·JSON/Python·Evidence/style/artifact·PDF 검사 | 재현 가능한 workspace 검증과 검사 범위 기록 |
| `submission/WORKSPACE_VALIDATION.json` | 실제 검사 결과·PDF/source/log hash | Workspace 통과와 과학적 타당성·제출 readiness 구분 |
| 제출 완료 조건 | UNKNOWN/NOT_CHECKED·BLOCKING 구분, 빈 bibliography·미측정 metric·paper ID 유지 | 환경 구성만으로 제출 checklist를 자동 완료하지 않기 |
| 원문 §24·첫머리 | 8-page provisional·cited-reference 범위·저자 마감·실제 skill 실행 요건 보완 | 준비 예시와 현재 정책/완료 상태 구분 |
| Guidelines·README·계획·TODO·PAPER_LOG·workflow | Step 6 부분 재확인·전체 6개 Step 완료·후속 연구 범위 반영 | 구성 완료와 논문 준비 상태를 함께 보고 |

### 검증 결과

- Skill 7개·읽기 전용 role 3개의 YAML/name/tools와 scoped rule 확인.
- 기존 `.claude/` 및 두 CLAUDE.md의 33개 파일 bytes 보존 확인; 181자·156자 유지.
- Markdown local link/anchor·JSON·Python 구문·whitespace·Evidence/artifact ID 연결 검사 PASS.
- Official original 14개 파일·working style/bst·current EXP001·T00 pointer/count 검증 PASS.
- Main 3페이지·supplementary 1페이지 cached rebuild와 PDF font/page/text/log 검사 PASS.
- Step 5의 source/PDF hash snapshot과 기존 raw research 자료를 보존했다.
- 수동 draft 준비 audit에서 가설·미측정 결과·빈 bibliography·method/protocol/OpenReview의
  미완료 항목을 확인하고 FINAL_AUDIT에 기록했다.
- 실제 Claude 모델 호출·세 full audit 명령·scientific experiment·OpenReview 등록/제출은 수행하지 않았다.

최초 report 생성 시 자신의 링크를 검사하는 순서 문제는 명시적인 IN_PROGRESS report 생성으로 수정했다.
전체 workspace 구성은 COMPLETE이며 실제 제출 준비는 NOT_READY다.
후속 연구·논문 작업은 [TODO](TODO.md)와 [FINAL_AUDIT](../submission/FINAL_AUDIT.md)를 따른다.

## Research Step R1 — Root 보존 범위 수정과 로컬 선행연구 분석

수행 시각: 2026-10-06 00:48 KST

사용자의 최신 지시를 반영했다. 이전 Step에서 Root CLAUDE.md까지 축약한 것은
200자 제한의 범위를 잘못 적용한 변경이었다. 사용자가 복원한 Root 파일은 이번 작업에서
수정하지 않았다. 아래 변경은 paper 진입 문서를 사용하도록 안내와 검증 조건을 바로잡은 것이다.
이전 Step의 181자 기록은 당시 상태이며 현재 보존 정책으로 해석하지 않는다.

| 대상 | 수정 내용·이유 |
|---|---|
| Rule·계획·원문·README·workflow | Root 복원본 수정 금지, paper/CLAUDE.md 진입, 200자 제한은 paper만 적용 |
| Workspace validator | Root 길이 검사를 제거하고 현재 SHA-256을 보고; PDF provenance에 bib hash 추가; 외부 URI는 목록 기록 후 정책별 검토 |
| R1 분석·provenance·related-work map | PDF 3편의 핵심 페이지·방법·평가·한계·version과 primary 서지 검증 |
| references.bib·Related Work·preamble | 검증한 preprint/report 3개를 실제 인용하고 bibliography 활성화 |
| NOVELTY·TODO·audit/checklist·원고 map | 이미 구현/제안된 overlap 기록; 전체 novelty는 UNVERIFIED; 다음 R2 scope 명시 |

수정한 과학적 해석: depth fusion·dense ESDF·live-depth MPC와 learned-policy safety layer의
넓은 아이디어는 cuRoboV2에 이미 구현 또는 제안되어 있다. 이를 독립 novelty로 확정하지 않는다.
cuRobo 파일명의 v1을 arXiv version으로 사용하지 않으며, minimum-jerk report와 ICRA 초기 논문의
서지 정보를 섞지 않는다. Library method와 우리 SQP/OSQP pipeline도 구분한다.

### 검증과 보존

- Root SHA-256: `cc3425396629709b3bc5afea5c4b309a994efe77f7d33829b2e7fab5393e311c` — 작업 시작 기준과 일치.
- paper/CLAUDE.md: 156자, 기존 bytes 유지. [보존 baseline](R1_PRESERVATION_BASELINE.json)과 비교한다.
- 실제 BibTeX entries/citations/rendered bibitems 3개 집합 일치; BibTeX log에 오류·경고 없음.
- Main 3페이지·supplementary 1페이지를 cached compiler로 별도 research-r1 폴더에 빌드했다.
- 기존 Step 5/6 PDF는 보존; [Step 6 검증 snapshot](WORKSPACE_VALIDATION_STEP6.json)도 별도로 보존했다.
- 최종 구조·links·source/manifest·font/layout/PDF 검사 결과는
  [현재 WORKSPACE_VALIDATION](../submission/WORKSPACE_VALIDATION.json)을 따른다.
- Compiler의 기존 upstream package 주석 encoding 경고는 남아 있으며 PDF glyph/log 검사와 구분한다.
- Source PDF·연구 코드·raw 결과·기존 claim/result는 변경하지 않았다. 실험·agent 모델 호출·제출은 수행하지 않았다.

R1의 제한된 문헌 분석은 완료했다. 전체 prior-art/novelty·method/source·실험 protocol 검증은 남아 있다.

## 선행연구 선정 상태 정정

시각: 2026-10-06 00:56 KST

사용자는 선행연구를 아직 최종 선정하지 않았으며 추후 후보 목록을 제공한다. R1의 3편을 탐색적 참고 후보로 명시하고 서지 검증과 선정 상태를 분리했다. Related-work map·분석·provenance·NOVELTY·TODO·README에 선정 대기를 반영했다. 원문의 논문명도 미선정 예시로 표시했다. 현재 원고 인용은 임시 초안으로 유지하며 후보 목록 수령 전 자동 문헌 확대·최종 baseline 선정을 진행하지 않는다.

## 작성 담당과 준비 범위 확정

시각: 2026-10-06 01:04 KST

사용자가 실제 논문 작성은 Claude Code가 맡고 준비 담당 assistant는 규칙·사전 작업만 수행한다고 범위를 명확히 했다. 이전 R2를 준비 담당자의 다음 작업으로 잡은 안내를 수정했다. Step 1–6 환경 준비는 완료이며 남은 필수 준비 항목은 0개다. CLAUDE_HANDOFF를 생성하고 README·계획·TODO·workflow·원문·scoped rule·최종 audit에 역할과 인계 범위를 반영했다. 최종 원고·연구·제출의 미완료 항목은 유지하며 환경 준비 항목과 구분했다. Root와 paper의 CLAUDE.md, 기존 원고·bibliography·PDF source는 수정하지 않았다. paper/CLAUDE.md는 156자 그대로이며 기존 README 진입 경로로 인계 문서를 안내한다.
