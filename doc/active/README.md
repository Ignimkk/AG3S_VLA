# doc/active — 현재 진행에 필요한 문서

정리: 2026-10-18 KST (사용자 지시). 반영이 끝났거나 지난 문서는 [../archive/](../archive/README.md)에 있다.

**새 세션은 [NEXT_SESSION_START_KO.md](NEXT_SESSION_START_KO.md)부터 읽는다.**

**2026-10-23 연구 범위 결정:** [한 페이지 범위](plans/AG3S_RESEARCH_SCOPE_20261023_KO.md) · [검토 근거/읽은 문서](reviews/AG3S_SCOPE_DECISION_20261023_KO.md) · [핵심 평가 구성과 종료 기준](../../AG3S/ag3s/docs/handoff/AG3S-CORE-EVALUATION.task.md). 이번 평가 회차의 우선 기준이며, 아래 과거 계획과 실험 결과는 이력으로 보존한다.

| 문서 | 무엇 |
|---|---|
| [BRANCHES.md](BRANCHES.md) | branch별 목적과 기준(main / feature/panda-benchmark / feature/ag3s-development). 세션 시작 때 읽는다 |
| [SHARED_STORAGE_AND_PYTHON.md](SHARED_STORAGE_AND_PYTHON.md) | storage 경로와 interpreter 설정의 정본 |
| [migration/source-manifest.json](migration/source-manifest.json) | migration source 기준(branch · commit) |
| [migration/GPU_OUTPUT_IMPLEMENTATION_REPORT.md](migration/GPU_OUTPUT_IMPLEMENTATION_REPORT.md) | GPU 환경 설치 · 검증 · 삭제 조건 |
| [migration/LEGACY_RETIREMENT_REPORT.md](migration/LEGACY_RETIREMENT_REPORT.md) | legacy 정리의 최신 판정 |
| [plans/develope_PLAN.md](plans/develope_PLAN.md) · [develope_PLAN_SUMMARY.md](plans/develope_PLAN_SUMMARY.md) | 개발 방향의 기준(§0.10 PAPER-FIRST) |
| [reviews/AG3S_decision/](reviews/AG3S_decision/README.md) | **채택 · 미채택 결정 모음**(여러 세션의 실험 기준). 첫 항목: 로봇 충돌 구 모델(손 채택 · 팔 내접 덮개, 팔 기본값 판정 대기) |
| [reviews/AG3S_TWO_PAPER_REVIEW_20261012_KO.md](reviews/AG3S_TWO_PAPER_REVIEW_20261012_KO.md) | T47 이후 연구 방향(두 논문 비교) |
| [reviews/AG3S_3D_Surface_Graph_and_Parameter_Tuning_Guide.md](reviews/AG3S_3D_Surface_Graph_and_Parameter_Tuning_Guide.md) | 3D 표면 graph · 파라미터 튜닝 가이드(IFG 기준) |
| [reviews/vla_dump_20261015/WORKLOG_S0S3_KO.md](reviews/vla_dump_20261015/WORKLOG_S0S3_KO.md) | **IFG 작업 기록(시간순, 정본)** |
| [reviews/vla_dump_20261015/FINAL_EXPERIMENT_AND_FIGURES_PLAN_KO.md](reviews/vla_dump_20261015/FINAL_EXPERIMENT_AND_FIGURES_PLAN_KO.md) | **다음 단계 계획**: 최종 실험 사슬과 논문 figure |
| [reviews/vla_dump_20261015/FINAL_REPORT_KO.md](reviews/vla_dump_20261015/FINAL_REPORT_KO.md) | IFG 결과의 쉬운 설명판(dev) |
| [reviews/vla_dump_20261015/METHODOLOGY_EVOLUTION_KO.md](reviews/vla_dump_20261015/METHODOLOGY_EVOLUTION_KO.md) | 방법론의 변화와 채택안 B |
| [reviews/vla_dump_20261015/SUMMARY_METHODS_RESULTS_KO.md](reviews/vla_dump_20261015/SUMMARY_METHODS_RESULTS_KO.md) | IFG 방법 · 결과 기술 요약 |
| [reviews/vla_dump_20261015/EXPERIMENT_PROTOCOL_KO.md](reviews/vla_dump_20261015/EXPERIMENT_PROTOCOL_KO.md) · `experiment_spec.json` · `improvement_round_template.json` | IFG 실험 protocol(v1.2)과 개선 round 규약 |
| [reviews/vla_dump_20261015/Novelty_contribution.pdf](reviews/vla_dump_20261015/Novelty_contribution.pdf) | 사용자 novelty / contribution 정리 |

storage run 폴더 색인: `/mnt/dev/AG3S_VLA-storage/results/development/RUN_INDEX_IFG.md`
