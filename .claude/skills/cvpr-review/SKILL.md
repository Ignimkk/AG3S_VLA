---
name: cvpr-review
description: Critique the authors' own CVPR draft for unsupported claims, method clarity, fair comparisons, and missing evidence.
---

# CVPR Author Draft Review

[공통 workflow](../../../paper/notes/CLAUDE_WORKFLOW.md)의 내부 검토와 정책 구분을 읽는다.
요청: `$ARGUMENTS`. `full`, `section 이름`, `claim Cxxx`를 지원한다. 기본은 `full`이다.
대상은 저자 자신의 원고다. 학회에서 배정받은 타인 원고의 공식 peer review용이 아니다.

1. 실제 원고의 존재와 검토 범위를 확인한다. 원고가 없으면 registry/story 기반
   `PRE_DRAFT_REVIEW`로 보고하고 원고 review 완료로 표시하지 않는다.
2. [PAPER_STORY](../../../paper/notes/PAPER_STORY.md),
   [NOVELTY](../../../paper/notes/NOVELTY.md),
   [REVIEWER_QUESTIONS](../../../paper/notes/REVIEWER_QUESTIONS.md), 관련 Evidence와 인용 분석을 읽는다.
3. Representation→3D grounding→contact policy→refinement가 어떤 scientific contribution인지,
   구현 사실과 개념 도식이 일치하는지 확인한다. Attention reliability는 실측·대조군·ablation으로 판단한다.
4. 비교의 checkpoint·scene·seed·budget·split, task/safety 분리, geometry/contact coverage,
   selection bias·실패 사례, paired/cluster 분석, runtime 경계, generalization을 점검한다.
5. 원고의 정확한 문장/section·ID·source를 근거로 major/minor finding을 작성한다.
   이미 읽은 자료에서 확인 가능한 내용은 다시 필요한 실험으로 요구하지 않는다.
   확인하지 못한 자료는 `NOT_CHECKED`로 표시한다.
6. 각 finding에 주장에 미치는 영향과 구체적 수정·필요 실험을 연결하고
   우선순위·미해결 질문을 보고한다. Audit 자체는 원고나 registry를 수정하지 않는다.

숫자 acceptance probability·가상의 reviewer consensus를 만들지 않는다.
반증과 근거 부족을 구분한다. Methodology critique의 완료와 evidence/compliance 통과는 별개다.
