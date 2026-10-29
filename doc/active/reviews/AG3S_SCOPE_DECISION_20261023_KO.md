# AG3S 연구 범위 결정 근거와 읽은 문서

작성: 2026-10-23 15:04 KST · 상태: **기존 문서·코드 검토에 따른 설계 결정**.
기준: `feature/ag3s-development`, HEAD `8da37119afd3b8ae1aaaa09315066c34c880857d`, 기존 dirty working tree 포함.
새 rollout·학습·성능 재측정·문헌 우선권 조사는 수행하지 않았다. 아래 기존 수치는 출처의 집계값을 읽은 것이며 재현 검증을 새로 통과했다는 뜻이 아니다.

결정 정본: [한 페이지 연구 범위](../plans/AG3S_RESEARCH_SCOPE_20261023_KO.md).
상세 구성·비교·종료 기준: [핵심 평가 task](../../../AG3S/ag3s/docs/handoff/AG3S-CORE-EVALUATION.task.md).

## 1. 판단

**현재 필요한 일은 211을 성공시키는 추가 규칙이 아니라, 내부 의미 단서를 접촉 기하로 바꾸는 연결이 실제로 유용한지 한정된 평가로 판정하는 것이다.**

이 방향은 새로운 연구 목표로 바꾸는 것이 아니다. 기존 개발 계획 §0.10도 identity·extent·contact를 나누고 경계 보존 방법 하나에 집중하라고 했다. 다만 실행이 실패 장면별 수정으로 길어졌고, 원고의 통합 설명·실제 serving 구성·성공 수의 의미가 서로 달라졌다. 이번에는 그 차이를 분리하고 개발 종료 조건을 고정했다.

## 2. 결정에 영향을 준 근거

| 확인한 사실 | 근거 | 이번 결정 |
|---|---|---|
| 기존 계획은 ATS→기하→접촉 연결을 중심으로 두고 HSR·CCR은 우선 지원 요소로 두었다 | 개발 계획 §0.1·0.10·1.2 | 네 모듈을 모두 독립 novelty로 완성하려고 하지 않음 |
| 같은 render mask의 dev에서 표면 방식 IoU 0.837→0.893. Held-out의 graph+mask는 0.773→0.876이나 성공 rollout 34개에 조건부 | 기술 요약 §3.1, RESULTS R008–R011 | 표면 방식은 평가할 근거가 있음. 실행 개선이나 전체 장면 일반화로 확대하지 않음 |
| Identity memory 추가 IoU 차이 +0.0148, 95% CI [−0.0117,+0.0394] | CLAIMS C014 / RESULTS R010 | 핵심 후보에 반드시 포함할 필요 없음. 10-15의 잠정 B 채택을 이번 평가에서는 최소 구성으로 재검토 |
| Feature edge/unary는 기존 geometry parent를 안정적으로 개선하지 않았다 | 방법론 변화 §2·3, 기술 요약 §2.2·3.2·3.3 | “내부값을 많이 쓴다”는 이유로 graph에 feature를 추가하지 않음 |
| 선별 없는 40 scene 비교의 contact-free success는 공 방식 17, 표면 방식 19; 차이 +0.05, CI [−0.125,+0.225] | `IFG-CL3.verify.json#/numbers/paired_primary` | 전체 closed-loop의 이득·동등성은 아직 미확인 |
| 같은 40 scene 정의에서 이웃 접촉은 primary에 포함되지 않음. 접촉 run은 각각 6·4 | 같은 JSON의 `primary_definition`, `per_arm/*/all/n_neighbour_contact` | 새 평가에는 이웃 과일 접촉을 포함. 과거 primary를 새 metric으로 이름만 바꾸지 않음 |
| 사과 24 scene: VLA 18, 기존 구성 16; 경보+3D 쥠을 더한 구성 15. 두 기능은 동시 변경 | `IFG-APPLE.verify.json`, `IFG-APPLE-S2.verify.json` | 경보·추가 쥠 규칙을 필수로 두지 않음. 두 기능 각각이 해롭다고 인과적으로 단정하지 않음 |
| 경보의 serving 점수 분포와 offline 점수 분포가 다름 | `IFG-APPLE-S2-CAUSE.verify.json`, WORKLOG 10-20 03:10 이후 | offline 경보 성능을 online 제어 이득으로 옮기지 않음 |
| 최근 9/10은 noun 기반 기존 분할에 손목 가중 0.1·이웃 접촉 20 mm 허용 등을 추가한 결과 | `IFG-APPLE-NBC.verify.json#/numbers/success`, 실제 `p/env.sh` | 가장 높은 성공 구성과 핵심 방법 후보를 구분. 원칙에 맞지 않는 접촉 완화는 본 평가에서 제외 |
| 최근 `NBCTS/NBCTF` argv에는 `--target-extent attention-surface`가 없음 | 실제 `p/env.sh`의 `arm_flags` | “표면 graph가 9/10 또는 7/10”이라고 쓰지 않음. `--nontarget-evidence surface`는 다른 설정 |
| 현재 kit은 `--links gripper`; 기존 일부 window에는 비권한 링크의 target 근접 query가 없었음 | 실제 `p/start_server.sh`, RESULTS R024 | 두 보정 구성에 공통 `gripper_wrist`를 선택. LTE 비교가 실제 성립하는지 먼저 확인 |
| 표면 serving에는 미세 layer만 교체·estimated fallback·TSDF sphere∪render mask 경로가 존재 | `config.py:1180–1231`, `runtime/attention_surface.py` 서두 | 원고의 이상화된 전체 surface-only 설명과 실제 구현을 구분 |
| 고정 형상 replay의 닫는 중 중심 오차 36.01→2.05 mm, closed-loop 성공 9→7/10, 최초 분기 미세 | WORKLOG 10-23 10:34·12:32·13:19, follow task | 기하 추종 성과와 전체 성공 효과를 구분. 작은 변화가 원인이라는 사실은 설정 효과 부재의 증거가 아님 |
| 미상 제약 중 stale 1,144행·artefact 1,134행 / 2,573행 | `IFG-UNKNOWN.verify.json#/numbers/U1_U2_fail6_pre_grasp/row_counts` | 전체 5,423행 중 (1,144+1,134)/5,423≈42.0%. “전체 제약의 89%”를 연구 우선순위 근거로 쓰지 않음 |
| 211에서 실제 두 손가락 접촉 후 held 미인정과 슬립이 관찰됨. Open-loop 명령 치환 결과는 closed-loop가 아님 | `IFG-211-GRASP.verify.json`, WORKLOG 10-23 진단 | 발생 위치와 인과를 구별. 실패는 평가 대상이며 전부 수정할 의무는 없음 |
| 등록된 surface 구성의 client 왕복 median 약 2.9초; 당시 533 ms budget 초과 | RESULTS R021 / CLAIMS C019 | 이번에 real-time을 목표·성과로 주장하지 않음. 실제 실행 비용은 계속 보고 |

위의 서로 다른 snapshot 결과를 합쳐 새 후보의 예상 성공률을 만들지 않았다. 이번에 고른 후보는 같은 mask·검사 링크·controller에서 새로 평가해야 한다.

## 3. 왜 이 구성을 선택했는가

표면 graph는 이미 구현돼 있고 의미 단서와 공간 경계를 연결하는 명확한 역할이 있다. 반면 identity 재연결·confidence 경보·추가 쥠 규칙은 독립 기여나 serving 일치가 아직 충분히 확인되지 않았다. 이를 모두 포함해야 “VLA 내부값을 쓴 방법”이 되는 것은 아니다. Noun attention도 VLA 내부 정보다.

그래서 **표면 graph를 검증할 핵심 후보**로 고르고, 공 모양 제외를 같은 조건의 강한 내부 baseline으로 둔다. 공 모양 제외도 이미 noun attention을 사용하므로 “기하-only”라고 부르지 않는다. 의미 정보의 필요성은 별도의 same-input 진단으로 확인한다. 외부 segmenter 대비 우수성은 이번 세 구성으로 증명할 수 없다.

이미 있는 phase probe를 즉시 제거하고 새 상태기를 만드는 것은 새 방법 변경이다. 기존 phase/latch를 공통으로 유지하고 오류를 측정한다. “고정한다”는 것은 신뢰성이 증명됐다는 뜻이 아니다. 공통 상태 오류가 두 방법을 모두 가릴 때는 구현 준비 실패 또는 판정 불가로 종료할 수 있다.

기존 20 mm 이웃 접촉 허용은 controller의 관용을 늘리는 설계다. 실제 침범 20 mm를 관측했다는 뜻은 아니지만, 불필요한 이웃 접촉을 줄이는 연구의 필수 요소로 둘 근거는 없다. 이를 제외해 성공률이 내려가면 그 손실을 그대로 보고한다.

## 4. 기존 계획과 충돌할 때 적용할 순서

이번 사용자는 연구 범위 고정·구성 선정·종료 기준의 결정을 요청했다. 따라서 이 결정과 한 페이지 범위가 **이번 평가 회차의 우선순위**다. 기존 완료 실험의 설정·결과·당시 사용자 판정은 바꾸지 않는다.

| 이전 지시/계획 | 이번 적용 |
|---|---|
| 실패 분석→개선 반복 | 이번 회차는 계약 오류 수정 1회·pilot 1회·test 1회로 제한 |
| 약 10 scene의 빠른 반복 | development pilot은 10 scene. 독립 확정 평가의 표본 설계는 별도이며 반복한 10 scene을 test라고 부르지 않음 |
| VLA 성공 scene으로 모듈 시험 | 진단 자료에 적용. 전체 성능 평가는 성공 결과로 선별하지 않음 |
| Identity 강화안 B 채택 | 과거 채택/결과 보존. 이번 최소 후보에서는 feature switching·경보를 제외 |
| 4단계 성공 시나리오 찾기 | 시연/figure 용도로 유지. 성공률 평가 분모와 분리 |
| SafeLIBERO 주 평가·실물 시연 | 전체 논문 확장의 기존 계획. 이번 RB-Y1 핵심 검증 완료가 이를 완료한 것은 아님 |
| 옛 로그의 모든 필수 실험 | 선택한 주장에 필요한 것만. 모든 HSR·optimizer 비교의 완료를 선행 조건으로 두지 않음 |

## 5. 읽은 문서 목록

아래는 **결정에 사용한 내용과 읽은 범위**다. 긴 로그·JSON은 관련 절/키를 읽었으며 저장소 전체나 raw rollout을 전수 검증한 목록이 아니다. 본 세션 앞선 검토에서 읽은 자료도 포함한다.

### 5.1 논문 문서

| 문서 | 읽은 범위 / 판단에 사용한 내용 |
|---|---|
| [METHOD_GUIDE_KO.md](../../../paper/METHOD_GUIDE_KO.md) | §1–5, §7–12 중심: identity/extent/permission, contact field, 실제 동작의 설명 |
| [CONTACT_PRESERVING_GUIDANCE_GUIDE_KO.md](../../../paper/CONTACT_PRESERVING_GUIDANCE_GUIDE_KO.md) | §1–8.1: 아직 검증하지 않은 후속 가설, grasp 조건과 생성 중 보정의 구분 |
| [CLAIMS.md](../../../paper/evidence/CLAIMS.md) | 주장 registry와 10-15·10-18 추가 등록: 가설/기술 통계/미입증 범위 |
| [RESULTS.md](../../../paper/evidence/RESULTS.md) | R001–R026 중심: 수치·분모·snapshot·source pointer·비교의 한계 |
| [EXPERIMENTS.md](../../../paper/evidence/EXPERIMENTS.md) | 계획/등록 실험 목록 및 비교 역할 확인 |
| [BASELINES.md](../../../paper/evidence/BASELINES.md) | baseline·ablation·oracle·공정 비교 규약 |
| [PAPER_STORY.md](../../../paper/notes/PAPER_STORY.md) | 전체: working hypothesis와 현재 근거의 구분 |
| [03_method.tex](../../../paper/latex/sections/03_method.tex) | 앞부분의 task-conditioned contact representation·surface graph 설명 |
| [04_experiments.tex](../../../paper/latex/sections/04_experiments.tex) | Setup, 비교 구성, Q1–Q4: 원고의 arm 명칭·contact 정의·표·runtime |

### 5.2 개발 방향·연구 기록

| 문서 | 읽은 범위 / 판단에 사용한 내용 |
|---|---|
| [develope_PLAN.md](../plans/develope_PLAN.md) | §0–1, §4, §7 중심: 논문 중심 우선순위, identity/extent 분리, 평가·중단 기준 |
| [develope_PLAN_SUMMARY.md](../plans/develope_PLAN_SUMMARY.md) | 머리말·핵심 기여 요약. 날짜/학회 규정은 이번에 검증하지 않음 |
| [NEXT_SESSION_START_KO.md](../NEXT_SESSION_START_KO.md) | 전체: 40 scene 평가·성공 조건부 집계·다음 단계의 당시 상태 |
| [AG3S_TWO_PAPER_REVIEW_20261012_KO.md](AG3S_TWO_PAPER_REVIEW_20261012_KO.md) | §1–4 중심: 내부 신호의 존재보다 downstream utility, HSR 교란. 외부 논문 원문 재검증은 아님 |
| [METHODOLOGY_EVOLUTION_KO.md](vla_dump_20261015/METHODOLOGY_EVOLUTION_KO.md) | 전체: feature 경계 가설 실패·잠정 B 채택·기여 분리 |
| [SUMMARY_METHODS_RESULTS_KO.md](vla_dump_20261015/SUMMARY_METHODS_RESULTS_KO.md) | §0–3, §4–6 중심: 알고리즘·dev 결과·한계·재현 경로 |
| [EXPERIMENT_PROTOCOL_KO.md](vla_dump_20261015/EXPERIMENT_PROTOCOL_KO.md) | §0–4: split·입력 계약·반복 개선 규약·GT 격리 |
| [FINAL_EXPERIMENT_AND_FIGURES_PLAN_KO.md](vla_dump_20261015/FINAL_EXPERIMENT_AND_FIGURES_PLAN_KO.md) | 시나리오 선정·장애물 기하 기준·시연과 성능 평가의 구분 |
| [WORKLOG_S0S3_KO.md](vla_dump_20261015/WORKLOG_S0S3_KO.md) | 10-19 사과·10-20 경보·10-21 TrajOpt 변형·10-22 제약 원인·10-23 211/형상 추종 절 |
| [AG3S_T47_LOG.md](../../../AG3S/ag3s/docs/AG3S_T47_LOG.md) | 현황·결정·미측정 목록 검색, T55 §4–6, T63/T63b 상태 및 종료 근거 확인 |
| [AG3S_PHASE_DENOISING_PLAN.md](../../../AG3S/docs/AG3S_PHASE_DENOISING_PLAN.md) | §1–2, §4.6–9 중심: ATS→LTE 계약, 평가 범위, 생성 중 개입은 후속이라는 구분 |

### 5.3 실행 task와 검증 원자료

경로 기준: [`AG3S/ag3s/docs/handoff/`](../../../AG3S/ag3s/docs/handoff/).

| 파일 | 읽은 내용 |
|---|---|
| `IFG-APPLE-run.task.md` | 24 scene 선정·VLA baseline·seed·평가 정의 |
| `IFG-S2SERVE.task.md` | feature 경보·추가 3D 쥠·기존 HSR 유지의 정확한 구성 |
| `IFG-SPLIT.task.md` | 분할과 anchor 교체·독립 scene 확인 조건 |
| `IFG-APPLE-TSURF.task.md` | `nontarget-evidence surface`와 place 뒤 인증의 의미 |
| `IFG-APPLE-FOLLOW.task.md` | sphere 추종·기존 비교 기준·기하 지표·미측정 |
| `IFG-211-GRASP.task.md` | 실제 궤적 기준 원인 후보와 사전 정의 |
| `IFG-CL3.verify.json` | `numbers/paired_primary`, `paired_success`, `per_arm`, `primary_definition` |
| `IFG-APPLE.verify.json` | `numbers/paired_primary`, `per_arm` |
| `IFG-APPLE-S2.verify.json` | `numbers/per_arm`, `alarm` |
| `IFG-APPLE-S2-CAUSE.verify.json` | 서버/오프라인 경보 분포의 GT phase별 집계 |
| `IFG-APPLE-NBC.verify.json` | `numbers/success`, `paired` |
| `IFG-APPLE-NBC-TARGETSURFACE.verify.json` | 집계 구조·success/hold/contact 지표 경로 확인 |
| `IFG-211-GRASP.verify.json` | 정의, 원인 판정, replay 정합, open-loop 추가 시험 |
| `IFG-UNKNOWN.verify.json` | `numbers/U1_U2_fail6_pre_grasp`, 분류 정의·verdict·not_measured |
| `IFG-PRECHECK-A.verify.json` | 미상 제약 비율과 분모·attribution 정의 |
| `IFG-AUDIT-LTE.verify.json` | `numbers/exemption_summary`의 target 잔류·이웃 손실·stale 구분 |
| `IFG-SPLIT.verify.json`, `IFG-AUDIT-ATS.verify.json`, `IFG-AUDIT-HSR.verify.json` | 집계 구조와 진단 범위 확인; 새로운 전체 수치 분석은 하지 않음 |

### 5.4 구현·운영 대조

- [config.py](../../../AG3S/ag3s/config.py): extent/robot-mask/fallback 및 kept geometry 설정.
- [attention_surface.py](../../../AG3S/ag3s/runtime/attention_surface.py): 서두의 graph·seed·code image·abstain 계약 및 선택 함수 위치.
- [serve_safe.py](../../../AG3S/trajopt/serve_safe.py): link group, target policy, render 연결, 경보·손목 가중치·이웃 접촉·복구 flag.
- Storage의 형상 추종 kit `p/env.sh`, `p/start_server.sh`: 실제 공통 flag와 각 구성 delta. 경로는 평가 task §2.3에 기록.
- [영상 폴더 README](../../../outputs/2026-10-23_사과10장면_고정형상따라감_비교/README.md): 같은 장면의 구성·결과·원시 기록 경로. 영상 전체를 다시 시청한 것은 아님.
- [CLAUDE.md](../../../CLAUDE.md), [.claude/ag3s-rules.md](../../../.claude/ag3s-rules.md), [.claude/ag3s-rules-brief.md](../../../.claude/ag3s-rules-brief.md), [BRANCHES.md](../BRANCHES.md): 연구 작업·기록·브랜치 경계.
- [paper/CLAUDE.md](../../../paper/CLAUDE.md), [cvpr-workspace.md](../../../.claude/rules/cvpr-workspace.md), [paper/README.md](../../../paper/README.md), [paper/SETUP_PLAN.md](../../../paper/SETUP_PLAN.md): 작업 경계·evidence 상태·준비 상태 관련 부분. 이번 작업은 원고 수정이나 학회 규정 확인이 아님.

## 6. 이번 작업에서 완료한 것과 남은 것

완료: 문서/구현 대조, 연구 범위 한 페이지, 핵심 후보·비교군·목표·예산·종료표 작성, 기존 진입 문서에 새 기준 링크 추가.

문서 검증: 새 문서 3개의 로컬 파일 링크가 모두 존재하고 code fence가 닫혔음을 확인했다. 설계 argv의 flag 40개 이름이 CLI source에 존재함을 확인했다. 이는 flag 조합의 실행 검증이 아니다. 기존 안내 문서 변경에 `git diff --check`를 수행했다.

미실행: 후보 구성의 CPU/GPU 동작 검증, 새 평가기 구현, split 실물 목록·hash 고정, 형상 추종 verify 보완, 신규 rollout, 논문 수치 갱신. 그러므로 **방법론의 선택은 완료했지만 성능 가설의 판정은 아직 하지 않았다.**
