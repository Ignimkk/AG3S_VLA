---
name: cvpr-write
description: Draft or revise this project's CVPR manuscript sections from scoped claims, registered evidence, and verified references.
---

# CVPR Manuscript Writing

[공통 workflow](../../../paper/notes/CLAUDE_WORKFLOW.md)의 작업 경계와 Evidence gate를 먼저 읽는다.
요청: `$ARGUMENTS`. `outline`, `section 이름`, `revise 경로`를 지원한다.
인자가 없으면 현재 원고와 TODO를 확인해 작성할 부분을 정한다.

1. [PAPER_STORY](../../../paper/notes/PAPER_STORY.md),
   [CONTRIBUTIONS](../../../paper/notes/CONTRIBUTIONS.md), 해당 Claim·Result·Metric을 읽는다.
2. 문장별 claim ID, 정량 문장의 result ID·source pointer, citation의 검증 상태를 연결한다.
   등록 결과를 쓰기 전에 manifest validator를 실행하고 성공한 범위만 사용한다.
3. Method의 구현 사실은 source snapshot·config와 대조한다. 계획된 단계나 가설을
   구현·성과로 쓰지 않는다. EXP001을 safety improvement나 task preservation의 근거로 쓰지 않는다.
4. 영문으로 작성한다. 결과가 없는 abstract/contribution/conclusion은 연구 질문과
   `DRAFT — EVIDENCE PENDING` 상태로 작성한다. 미확인 수치는 비워 두고 필요한 근거를 기록한다.
5. Template 도입 전에는 `paper/notes/`에 outline/draft를 둔다. 이후에는 실제 존재하는
   `paper/latex/sections/`의 해당 파일을 수정한다. 임의의 template을 만들지 않는다.
6. Claim→Experiment→Result→문장 연결과 citation·scope를 다시 확인한다.
   변경한 파일, 사용한 ID, 미확정 문장과 검증 결과를 보고하고 PAPER_LOG에 기록한다.

원고 전체에 대한 새 성과 주장이나 최종 제출 준비 판정은 공통 workflow의 세 audit를 따른다.
평범한 문장 교정에 전체 review를 반복할 필요는 없다.
