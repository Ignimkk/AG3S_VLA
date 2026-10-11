# CVPR 2027 논문 준비 계획

기준 문서: [paper_structure_setup.md](paper_structure_setup.md)

## 진행 원칙

- 실제 원고 작성은 Claude Code가 맡고, 준비 담당 assistant는 규칙·환경·사전 작업만 수행한다.
- 필수 환경 준비는 Step 1–6 완료이며 남은 필수 준비 항목은 0개다.
  작성 시작과 역할 분담은 [CLAUDE_HANDOFF](notes/CLAUDE_HANDOFF.md)를 따른다.

- 한 번의 작업에서 1 Step만 수행하고 변경 사항과 검증 결과를 보고한다.
- `/mnt/dev/work/paper/`는 논문 작업 공간, `/mnt/dev/work/.claude/`는 Claude 설정 위치다.
- 기존 연구 코드를 복사하거나 수정하지 않는다.
- CVPR 정책, template 버전, citation, experiment 결과를 확인 없이 확정하지 않는다.
- Root `/mnt/dev/work/CLAUDE.md`는 사용자 복원본을 수정하지 않는다.
- 논문 진입점 `paper/CLAUDE.md`만 공백·줄바꿈을 포함해 200자 이내로 유지한다.

사용자의 최신 지시에 따라 아래 Step 1–6 기록의 Root 축약·181자 검증은 과거 이력이다.
현재 규칙은 Root 보존과 `paper/CLAUDE.md` 사용이며, 과거 지시로 Root를 다시 축약하지 않는다.

## 단계별 계획

| Step | 작업 | 주요 산출물 | 상태 |
|---|---|---|---|
| 1 | 원문 전체 검토, 경로·기존 지침 정리, 기본 작업 공간 구성 | `README.md`, `CLAUDE.md`, 이 계획, scoped rule, 수정 기록, 기본 폴더 | 완료 |
| 2 | 공식 CVPR 2027 자료와 2026 fallback 검증 | `guidelines/`의 9개 문서(`SOURCES.md` 포함), 검증된 일정과 정책 분류 | 완료 |
| 3 | Evidence 구조와 연구 계획 구성 | `evidence/`의 7개 문서, manifest schema, `notes/`의 story·novelty·contribution·계획 | 완료 |
| 4 | Claude 작성·검증 workflow 구성 | `.claude/skills/`의 7개 CVPR skill, `.claude/agents/`의 3개 역할 정의, 공통 workflow | 완료 |
| 5 | 공식 template 및 manuscript·figure·table·algorithm 기반 구성 | Pinned 원본, 7개 section, 별도 supplementary, artifact 계획·sidecar·build helper | 완료 |
| 6 | Submission checklist와 전체 구성 검증 | 제출 checklist 5개·최종 audit·workspace validator·cached build 검증 | 완료 |

Step 2에서 최신 공식 Author Kit을 확인한 뒤 Step 5의 template 버전을 정한다.
Step 3에서는 16-D 여부와 provenance를 확인한 experiment만 whitelist에 등록한다.
연구 계획·가설은 검증된 결과와 구분한다. Step 4의 역할 정의 생성 자체로 experiment나
review를 실행하지 않는다. 이 계획은 논문 작성 환경 구축 계획이며, 연구 수행과 실제
submission 완료를 의미하지 않는다.

## 추가 작업 이력과 Claude Code에 넘길 업무

| Step | 범위 | 상태 |
|---|---|---|
| R1 | 기존 PDF 3편의 핵심 내용·primary 서지·novelty overlap·초기 Related Work | 탐색적 추가 작업 이력; 최종 문헌 선정은 사용자 목록 대기 |
| R2 | 현 구현 snapshot·method 범위·변경 이력의 읽기 전용 audit | Claude Code·연구 담당자의 후속 업무; 준비 담당자의 다음 Step이 아님 |

R1 근거는 [문헌 분석](notes/REFERENCE_ANALYSIS_R1.md), 작성·연구 후속 항목은 [TODO](notes/TODO.md)를 따른다.
이 항목들은 준비 담당자가 자동으로 계속 수행할 계획이 아니다. 기존 추가 작업 기록은 보존한다.

## Step 1 완료 기록

- 수행 시각: 2026-10-05 18:59 KST
- 준비 문서 전체 1,275줄을 읽고 26개 절을 6개 작업 단계로 나눴다.
- 기존 Root `CLAUDE.md`를 `.claude/project-context.md`에 원문 그대로 보존했다.
- Root 및 `paper/CLAUDE.md`를 각각 200자 이내로 작성했다.
- 기존 `.claude/`를 공통 설정 위치로 사용하고, 논문 전용 rule을 추가했다.
- 기존 AG3S hook 규칙에 적용 범위를 명시해 논문 작업 기록과 연구 실행 로그를 구분했다.
- 원문의 `.claude` 위치, 상세 지침 배치, 잘못된 절대 경로를 바로잡았다.
- `.gitignore`에 `paper/` 예외를 추가해 논문 문서를 버전 관리할 수 있게 했다.
- 기본 폴더를 생성했다. 후속 Step의 문서·skill·LaTeX 파일은 아직 만들지 않았다.

수정 이유와 검증 결과는 [SETUP_CHANGES.md](notes/SETUP_CHANGES.md)에 기록한다.
현재 **Step 1–6의 workspace 구성을 모두 완료**했다. 실제 연구·원고 완성·제출 준비는
[TODO](notes/TODO.md)와 [FINAL_AUDIT](submission/FINAL_AUDIT.md)의 미완료 항목을 따른다.

## Step 2 완료 기록

- 수행 시각: 2026-10-05 19:21 KST
- 공식 CFP·Dates·profile 안내와 2026 guidelines·공식 Author Kit을 확인했다.
- `guidelines/`에 계획된 9개 문서를 생성하고, 13개 공식 출처의 확인 범위와 시각을 기록했다.
- 원문의 주요 날짜는 공식 CFP와 일치한다. KST 참고값은 AoE 날짜 종료 가정으로 별도 계산했다.
- 2027 Guidelines URL의 404와 확인한 kit의 2026 표시는 관찰 사실로 기록하고,
  2027 상세 규정·kit 공개 여부는 `UNKNOWN`으로 구분했다.
- Author-list 마감, reviewer service, 외부 링크 예외, LLM 안내 범위,
  supplementary 개선 method 제한과 rebuttal 전체 page count를 보완했다.
- 문서·출처 참조, 정책 분류와 날짜 계산, 기존 `CLAUDE.md` 200자 제한을 검증했다.
- Step 2의 조사·문서화는 완료했다. 미공개/미확인 정책의 재확인과 kit 도입은 후속 작업이다.
  실험·skill 생성·template 다운로드·LaTeX build는 수행하지 않았다.

## Step 3 완료 기록

- 수행 시각: 2026-10-05 19:48 KST
- Evidence 7개 문서와 연구 계획 notes 6개 문서를 생성했다.
- Claim·experiment·result·baseline·metric·figure/table의 ID와 연결을 구성하고
  연구 가설, descriptive observation, 미측정 항목을 구분했다.
- T39의 96 run / 48 pair / 24 episode를 읽기 전용으로 audit했다. 모든 run의 실제
  state/action 16-D 배열 header와 metadata·완료 자료·source identity를 확인했다.
- EXP001 manifest에는 해당 snapshot의 task outcome R001–R003만 등록했다.
  Safety·성능 유지·runtime·generalization·novelty·hardware 결과는 등록하지 않았다.
- JSON manifest schema·candidate 예제·source audit·읽기 전용 validator를 생성했다.
- 정상 등록과 candidate 예제의 처리를 확인했고, 14-D·위장 등록·잘못된 pointer·변조된 값·
  source hash 불일치 등 7개 invalid case가 거부되는지 확인했다.
- 원문의 attention observation과 desired outcome을 가설·연구 목표로 명시했다.
- 후속 T40/T41/T42 자료와 선행연구 PDF는 미등록/미검토 상태를 기록했다.
  기존 research source·로그·raw 결과는 수정하지 않았고 새 scientific experiment는 실행하지 않았다.

Step 3은 Evidence 관리 기반을 구성한 것이다. 주요 연구 가설의 입증이나 논문 완성을 뜻하지 않는다.
Step 3 종료 시점의 다음 구성 작업은 Step 4였다.

## Step 4 완료 기록

- 수행 시각: 2026-10-05 20:43 KST
- `cvpr-write`, `cvpr-evidence`, `cvpr-experiment`, `cvpr-figure`,
  `cvpr-reference-analysis`, `cvpr-review`, `cvpr-compliance`의 7개 skill을 생성했다.
- `literature-analyst`, `evidence-auditor`, `cvpr-reviewer`의 3개 읽기 전용 역할을 생성했다.
- [CLAUDE_WORKFLOW](notes/CLAUDE_WORKFLOW.md)에 호출 예시·Evidence gate·역할 소유권·
  최종 audit의 완료 조건을 구성했다. Parent writer만 논문 파일 변경을 반영한다.
- 공식 Claude Code 문서의 project skill/agent 형식을 확인했다. 로컬 CLI는 2.1.282다.
- Skill validator 7개, agent YAML·unique name·도구 제한, 내부 참조와 whitespace를 검증했다.
- 기존 EXP001을 재검증했고 기존 설정·연구용 skill/agent와 두 CLAUDE.md는 보존했다.
- 오래된 Step 3/4 예정 안내를 실제 구성 상태로 수정했다.
- 실제 Claude 모델 호출·agent delegation·연구 실험·원고 review·submission audit는 실행하지 않았다.

Step 4는 Claude 작성 환경 구성 완료를 뜻한다. 당시 다음 구성 작업은 Step 5였다.

## Step 5 완료 기록

- 수행 시각: 2026-10-05 21:08 KST
- 2027 CFP·Guidelines URL과 공식 kit repository·main·tags를 재확인했다.
  해당 Guidelines URL은 404이며 확인한 kit은 2026이다. 2027 공개 여부는 UNKNOWN으로 유지했다.
- Tag `CVPR2026-v1(latex)`, commit `12909ae437f6dbc7435069cfdb4ca44c18e6a02f`의
  원본 14개 파일을 보존하고 archive/file SHA-256·다운로드 시각을 기록했다.
- Working main/supplementary style·bst는 original bytes를 유지했다. Target year만 2027로
  분리했고 paper ID는 미등록 placeholder다. 공식 2027 kit으로 표시하지 않는다.
- 7개 English section, bibliography pending 구조와 별도 supplementary를 구성했다.
- F02 overview와 A01 algorithm은 conceptual draft, T00은 등록 task counts만 연결한
  descriptive table, 나머지는 planned placeholder·PLAN·sidecar로 구분했다.
- Tectonic 0.17.0을 `/tmp/`에서 사용해 main 3페이지·supplementary 1페이지를 빌드했다.
  초기 XeTeX font 대체는 T1 encoding으로 수정했고 Times 계열 실제 font를 확인했다.
- Default main 밖의 7개 placeholder도 별도 cached build에서 렌더링했다.
- Source 검증 build helper, PDF page/metadata/text/font 및 시각 검토를 수행했다.
  자세한 범위·tool provenance·잔여 경고는 [STEP5_BUILD_AUDIT](notes/STEP5_BUILD_AUDIT.json)를 따른다.
- 새 scientific experiment·선행연구 bibliography audit·원고 review·submission audit는 수행하지 않았다.

Step 5는 빌드 가능한 draft 기반 구성 완료이며 논문 완성/제출 준비 완료가 아니다.
Step 5 종료 시점의 다음 작업은 Step 6: Submission checklist와 전체 구성 검증이었다.

## Step 6 완료 기록

- 수행 시각: 2026-10-06 00:01 KST
- Submission·anonymity·page limit·references·supplementary checklist 5개와
  FINAL_AUDIT·사용 안내·camera-ready 단계 안내를 생성했다.
- Workspace 구성 완료, draft 상태, 실제 제출 readiness를 구분했다. 미완료 scientific
  evidence·bibliography·method/protocol·OpenReview·full audit를 자동 완료 표시하지 않았다.
- 2027 CFP·Dates·profile·Guidelines URL·kit repository와 2026 fallback의 관련 안내를 재확인했다.
  확인한 2027 Guidelines URL은 404이며 상세 정책·kit 상태는 UNKNOWN을 유지했다.
- Workspace validator를 작성해 Markdown/anchor·JSON/Python·Claude config·Evidence/artifact
  연결·original/working style·PDF font/page/제한된 identity scan을 검증했다.
- Main/supplementary를 캐시로 다시 빌드해 별도 `build/step6/`에 보존했다. 기존 Step 5 PDF와
  source audit snapshot은 변경하지 않았다. Main 3페이지, supplementary 1페이지다.
- 기존 `.claude/`와 두 CLAUDE.md의 33개 파일 hash를 보존했다. 각각 181자·156자를 유지했다.
- 실제 모든 section을 읽어 가설·descriptive outcome·미측정 항목의 구분을 수동 확인했다.
  이는 Claude skill 호출이나 최종 제출물 전체 review의 완료 기록이 아니다.
- 검사 결과는 [WORKSPACE_VALIDATION](submission/WORKSPACE_VALIDATION.json),
  수정 이유는 [SETUP_CHANGES](notes/SETUP_CHANGES.md)에 기록했다.

요청된 6개 구성 Step을 완료했다. 실험·선행연구 내용 검증·actual OpenReview 등록/제출은
후속 연구/논문 작업이며 이번 workspace 구성 완료에 포함되지 않는다.
