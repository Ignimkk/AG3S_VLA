
# CVPR 2027 VLA Paper Workspace

> Step 1에서 경로·설정 위치를 정리하고 Step 2에서 CVPR 공식 자료를 확인했다.
> 정책의 현재 적용 상태와 출처는 `guidelines/CVPR_2027_STATUS.md` 및
> `guidelines/SOURCES.md`를 따른다. 아래 예시에서 2026 fallback을 2027 확정 규정으로
> 해석하지 않는다. 계획은 `SETUP_PLAN.md`, 수정 내역은 `notes/SETUP_CHANGES.md`에 있다.
> Step 3에서 Evidence 구조와 연구 계획을 구성했다. 등록된 근거는 T39 snapshot의
> 제한된 task outcomes뿐이며 `evidence/CLAIMS.md`·`RESULTS.md`의 status와 scope를 따른다.
> Safety·성능 유지·novelty·generalization·runtime·hardware 성과는 아직 확정하지 않았다.
> Step 4에서 CVPR skill 7개와 읽기 전용 검토 역할 3개를 구성했다. 실제 정의와 호출 예시는
> `notes/CLAUDE_WORKFLOW.md`를 따른다. 아래 예시는 실제 audit 실행·완료 기록이 아니다.
> Step 5에서 pin한 공식 2026 fallback, 7개 영문 section과 별도 supplementary를 도입했다.
> F02/A01은 conceptual, T00은 제한된 descriptive 결과, 나머지는 미측정 계획이다.
> 현재 파일·상태는 `notes/MANUSCRIPT_MAP.md`와 `evidence/FIGURES_TABLES.md`를 따른다.
> Step 6까지 workspace 구성을 완료했다. 실제 제출 상태는 `submission/FINAL_AUDIT.md`의
> NOT_READY이며 핵심 실험·인용·protocol·실제 OpenReview 준비는 미완료다.

이 문서의 환경 준비 Step 1–6은 완료했다. 실제 원고는 Claude Code가 작성하고,
준비 담당 assistant는 규칙·환경·사전 작업만 담당한다. 후속 연구 작업을 자동으로
준비 담당자의 다음 Step으로 확장하지 않는다. 인계는 `notes/CLAUDE_HANDOFF.md`를 따른다.

Root:

```text
/mnt/dev/work/paper
```

본 폴더는 CVPR 2027 VLA 논문의

```text
Research
→ Evidence
→ Experiment
→ Figure/Table
→ Manuscript
→ Review
→ Submission
```

전체 과정을 관리한다.

기존 연구 코드는 `/mnt/dev/work` 아래의 repository를 사용하고, `paper/` 안으로 복사하지 않는다.

---

# 1. 최종 권장 폴더 구조

```text
/mnt/dev/work/paper
│
├── CLAUDE.md
├── README.md
│
├── guidelines/
│   ├── CVPR_2027_STATUS.md
│   ├── CVPR_2027_DEADLINES.md
│   ├── CVPR_FORMAT_RULES.md
│   ├── CVPR_SUBMISSION_POLICY.md
│   ├── CVPR_ANONYMITY_POLICY.md
│   ├── CVPR_LLM_POLICY.md
│   ├── CVPR_SUPPLEMENTARY_POLICY.md
│   ├── CVPR_REBUTTAL_POLICY.md
│   └── SOURCES.md
│
│
├── template/
│   └── cvpr2026_official/
│       ├── main.tex
│       ├── preamble.tex
│       ├── rebuttal.tex
│       ├── cvpr.sty
│       ├── ieeenat_fullname.bst
│       ├── main.bib
│       └── sec/  # pin한 2026 tag에는 fig/가 없음
│
│
├── latex/
│   ├── main.tex
│   ├── preamble.tex
│   ├── metadata.tex
│   ├── references.bib
│   ├── cvpr.sty
│   ├── ieeenat_fullname.bst
│   │
│   └── sections/
│       ├── 00_abstract.tex
│       ├── 01_introduction.tex
│       ├── 02_related_work.tex
│       ├── 03_method.tex
│       ├── 04_experiments.tex
│       ├── 05_conclusion.tex
│       └── 06_limitations.tex
│
│
├── evidence/
│   ├── CLAIMS.md
│   ├── EXPERIMENTS.md
│   ├── RESULTS.md
│   ├── BASELINES.md
│   ├── METRICS.md
│   ├── FIGURES_TABLES.md
│   └── PAPER_LOG.md
│
│
├── experiments/
│   ├── manifests/
│   ├── configs/
│   ├── summaries/
│   └── statistics/
│
│
├── figures/
│   ├── fig01_teaser/
│   ├── fig02_overview/
│   ├── fig03_attention_to_3d/
│   ├── fig04_qualitative/
│   ├── fig05_analysis/
│   └── supplementary/
│
│
├── tables/
│   ├── table00_descriptive/  # EXP001 task counts; P001 main safety와 구분
│   ├── table01_main/
│   ├── table02_generalization/
│   ├── table03_ablation/
│   ├── table04_efficiency/
│   └── supplementary/
│
│
├── algorithms/
│   └── algorithm01_safe_vla.tex
│
│
├── references/
│   ├── cvpr2026_vla/
│   │   ├── ACoT-VLA.pdf
│   │   ├── ActiveVLA.pdf
│   │   ├── AVA-VLA.pdf
│   │   ├── OptimusVLA.pdf
│   │   └── SaPaVe.pdf
│   │
│   ├── vla/
│   ├── attention/
│   ├── safety/
│   ├── 3d_perception/
│   └── motion_planning/
│
│
├── notes/
│   ├── PAPER_STORY.md
│   ├── NOVELTY.md
│   ├── CONTRIBUTIONS.md
│   ├── RELATED_WORK_MAP.md
│   ├── REVIEWER_QUESTIONS.md
│   └── TODO.md
│
│
├── tools/
│   ├── analysis/
│   ├── plotting/
│   └── latex/
│
│
├── supplementary/
│   ├── supplementary.tex
│   ├── experiments/
│   ├── figures/
│   └── tables/
│
│
└── submission/
    ├── SUBMISSION_CHECKLIST.md
    ├── ANONYMITY_CHECKLIST.md
    ├── PAGE_LIMIT_CHECKLIST.md
    ├── REFERENCES_CHECKLIST.md
    ├── SUPPLEMENTARY_CHECKLIST.md
    ├── FINAL_AUDIT.md
    └── camera_ready/
```

---

Claude 설정은 기존 repository의 `/mnt/dev/work/.claude/`에 모아 관리한다.
`paper/.claude/`를 중복 생성하지 않는다.

```text
/mnt/dev/work/.claude/
├── project-context.md
├── rules/
│   └── cvpr-workspace.md
├── skills/
│   ├── cvpr-write/SKILL.md
│   ├── cvpr-evidence/SKILL.md
│   ├── cvpr-experiment/SKILL.md
│   ├── cvpr-figure/SKILL.md
│   ├── cvpr-reference-analysis/SKILL.md
│   ├── cvpr-review/SKILL.md
│   └── cvpr-compliance/SKILL.md
└── agents/
    ├── literature-analyst.md
    ├── evidence-auditor.md
    └── cvpr-reviewer.md
```

위 구조는 최종 목표다. 생성 시점은 `SETUP_PLAN.md`를 따른다.
기존 AG3S 설정과 사용자가 복원한 Root `CLAUDE.md`는 수정하지 않는다.
논문 진입 안내는 `paper/CLAUDE.md`에만 200자 이내로 담고,
상세 지침은 `.claude/rules/`와 `paper/` 문서에 둔다.

---

# 2. 기존 연구 코드와 Paper Workspace 관계

논문 폴더 안에 AG3S나 π0.5 코드를 복사하지 않는다.

Claude는 다음 경로를 **read-only research source**로 사용한다.

```text
/mnt/dev/work/benchmark/ag3s
/mnt/dev/work/benchmark/trajopt
/mnt/dev/work/src
/mnt/dev/work/pi05_TO_hybrid
/mnt/dev/work/outputs
/mnt/dev/work/tests
```

논문 작성 목적의 코드 분석은 허용하지만 기존 구현을 수정하지 않는다.

논문용 통계/plotting/LaTeX 생성 코드만 다음에 생성할 수 있다.

```text
/mnt/dev/work/paper/tools/
```

---

# 3. CVPR 규정 전용 폴더 추가

가장 중요한 변경이다.

```text
paper/guidelines/
```

를 추가한다.

Claude는 논문을 작성하거나 수정하기 전에 이 폴더의 규정을 확인해야 한다.

---

# 4. `CVPR_2027_STATUS.md`

```markdown
# CVPR 2027 Status

Target:
CVPR 2027

Conference:
June 20–25, 2027
Seattle, Washington, USA

Official CVPR 2027 CFP:
AVAILABLE

Official CVPR 2027 Author Guidelines:
UNKNOWN — checked URL returns 404; re-check publication path

Official CVPR 2027 Author Kit:
UNKNOWN — checked official repository identifies CVPR 2026

Therefore:

1. Use CVPR 2027 rules whenever officially available.
2. Use CVPR 2026 Author Guidelines for rules not yet specified for 2027.
3. Mark every inherited 2026 rule as provisional.
4. Re-check the official CVPR 2027 website before submission.
5. Never assume that a 2026 policy automatically remains unchanged.
```

CVPR 2027 대회는 2027년 6월 20–25일 Seattle에서 열립니다.

---

# 5. `CVPR_2027_DEADLINES.md`

```markdown
# CVPR 2027 Important Dates

Paper Registration:
November 10, 2026 AOE

Paper Submission:
November 16, 2026 AOE

Supplementary Materials:
November 23, 2026 AOE

Reviews Released:
January 25, 2027 AOE

Rebuttal:
January 25 – February 1, 2027 AOE

Final Decisions:
February 25, 2027 AOE

Conference:
June 20–25, 2027
Seattle, Washington
```

이 일정은 이미 CVPR 2027 공식 CFP에 발표되어 있으며, paper registration과 submission deadline은 **연장하지 않는다고 명시**되어 있습니다.

---

# 6. CVPR 2027 Topic Positioning

다음 상세 positioning은 `notes/PAPER_STORY.md`에 정리한다.
`paper/CLAUDE.md`에는 문서 안내만 두어 200자 제한을 유지한다.

```markdown
## CVPR 2027 Scope

CVPR 2027 explicitly includes:

- Multimodal Learning
- Multimodal Generative and Foundation Models
- Reasoning & Reinforcement Learning
- Embodied Vision and Robotics
- Agentic AI
- Physical AI: World Models, VLA, Planning, Simulation
- 3D Reconstruction and Geometry
- Scene Analysis and Understanding

Our paper should primarily position itself within:

Physical AI: VLA
+
Embodied Vision and Robotics
+
3D Scene Understanding
+
Multimodal Representation Learning
```

CVPR 2027에서 **VLA가 공식 CFP topic으로 명시**된 것은 이번 연구의 positioning에 매우 중요합니다.

---

# 7. `CVPR_FORMAT_RULES.md`

확인한 2027 Author Guidelines URL이 404이므로, 미확인 항목에는 CVPR 2026 규정을 provisional하게 적용한다.

```markdown
# CVPR Paper Formatting Rules

STATUS:
PROVISIONAL — CVPR 2026 rules

Update this file when CVPR 2027 Author Guidelines are released.

## Main Paper

Maximum:
8 pages

The 8 pages include:

- main text
- figures
- tables
- equations

References do NOT count toward the 8-page limit.

References may continue onto additional pages.

## Required format

Use the official CVPR LaTeX template.

Review manuscript:

\documentclass[10pt,twocolumn,letterpaper]{article}

\usepackage[review]{cvpr}

Do not manually modify:

- margins
- font size
- column spacing
- page geometry
- CVPR style definitions

Overlength papers may be desk rejected.

## Language

Paper must be written in English.

## Recommended repository policy

Do not modify cvpr.sty.

Keep the original official template under:

/mnt/dev/work/paper/template/

and work from a copied version under:

/mnt/dev/work/paper/latex/
```

CVPR 2026은 **본문+Figure+Table을 포함해 최대 8페이지이며 references는 제한에서 제외**합니다. 공식 template을 사용하지 않거나 8페이지를 넘거나 적절히 anonymize하지 않은 논문은 review 없이 reject될 수 있습니다.

현재 공식 `cvpr-org/author-kit`도 여전히 **CVPR 2026 버전**으로 표시되어 있습니다. Review mode는 `\usepackage[review]{cvpr}`를 사용합니다.

---

# 8. Template 관리 방식

공식 template을 바로 수정하지 않는다.

```text
template/cvpr2026_official/
```

에는 원본 그대로 보존한다. Step 5는 tag `CVPR2026-v1(latex)`와 full commit
`12909ae437f6dbc7435069cfdb4ca44c18e6a02f`를 사용한다. Archive/file checksum·시각은
`template/PROVENANCE.json`에 기록했다. Current main branch 대신 고정된 tagged revision을 사용한다.

실제 논문은

```text
latex/
```

에서 작성한다.

즉:

```text
Official CVPR Template
        │
        │ copy
        ▼
paper/latex/
        │
        ▼
Our manuscript
```

`cvpr.sty`·`.bst`는 원본 bytes를 유지하고 원고·preamble·metadata만 별도로 구성한다.
Working confYear=2027은 local target 변경이며 공식 2027 template이라는 뜻이 아니다.
원본 예시 bibliography를 연구 citation으로 복사하지 않는다. Supplementary는 별도 entry/PDF로 만든다.
빌드와 source 검증은 `tools/latex/build.py`, 사용 방법은 `latex/README.md`를 따른다.

2027 공식 Author Kit이 나오면

```text
template/
├── cvpr2026_official/
└── cvpr2027_official/
```

로 추가한 뒤 `latex/`를 migration 한다.

---

# 9. `CVPR_ANONYMITY_POLICY.md`

현재는 CVPR 2026 규정을 provisional하게 적용한다.

```markdown
# Double-Blind / Anonymity Rules

STATUS:
PROVISIONAL — CVPR 2026 rules

CVPR uses double-blind review.

The review manuscript must not disclose:

- author names
- institution
- laboratory
- acknowledgments revealing identity
- grant IDs revealing identity
- institution-identifiable demo material
- non-anonymized supplementary material

Do not write:

"At KETI we developed..."

"Our previous system..."

"Our laboratory robot..."

when this makes the authors identifiable.

Self-citations must be written neutrally.

Bad:
"In our previous work [X]..."

Preferred:
"Previous work [X]..."

Do not include public project URLs that identify the authors.

All supplementary material and videos must also preserve anonymity.
```

CVPR 2026은 double-blind review를 사용하며 acknowledgments, grant ID, supplementary video 등에 저자나 기관을 추론할 수 있는 정보를 포함하지 않도록 요구합니다.

---

# 10. `CVPR_SUBMISSION_POLICY.md`

```markdown
# Submission Policy

## CVPR 2027 confirmed rules

Submission platform:
OpenReview

All co-authors must have a complete and up-to-date OpenReview profile.
No author additions or removals are accepted after abstract registration.
Incomplete author registration can lead to desk rejection.

When invited, authors commit to reviewer/AC/SAC service.
Exemptions require prior disclosure and approval by program chairs.

Paper registration:
November 10, 2026 AOE

Paper submission:
November 16, 2026 AOE

## Dual submission

Do not submit substantially similar work to another
conference or workshop during the CVPR 2027 review period:

November 16, 2026
through
February 22, 2027.

Before submission run:

/cvpr-compliance dual-submission
```

CVPR 2027 CFP에서 이 review-period dual submission restriction이 이미 명시되어 있습니다.

CVPR 2026에서는 substantially similar content를 **20% 이상 overlap**으로 정의한 세부 규칙이 있었지만, CVPR 2027의 상세 Author Guidelines가 아직 없으므로 이 수치는 현재 `2026 참고사항`으로만 보관하고 2027 규정으로 단정해서는 안 됩니다.

---

# 11. Contemporaneous Work 규칙

2027에서 새로 특히 주의해야 한다.

`CVPR_SUBMISSION_POLICY.md`에 추가한다.

```markdown
## Contemporaneous Work — CVPR 2027

Cutoff:

September 15, 2026

Work appearing online after this date will generally
be treated as contemporaneous work.

However:

- relevant work should still be cited
- comparisons should be performed when feasible
- work that influenced our submission is prior work
  regardless of publication date

Claude must not use the contemporaneous-work rule
to avoid citing papers that actually influenced this work.
```

이는 CVPR 2027 공식 CFP에 명시되어 있습니다.

---

# 12. `CVPR_SUPPLEMENTARY_POLICY.md`

현재는 2026 규정을 provisional하게 사용한다.

```markdown
# Supplementary Material

STATUS:
PROVISIONAL — CVPR 2026 rules

Supplementary material is optional.

Potential content:

- additional experiments
- additional ablations
- implementation details
- additional figures
- additional tables
- mathematical derivations
- qualitative examples
- videos
- reproducibility details
- anonymized code

Important:

The main contribution must remain understandable
without supplementary material.

PROVISIONAL_2026 — reviewer FAQ:
Do not include improved-method results after additional tuning/training,
or an updated/corrected main submission PDF, in supplementary.

Never hide a critical experiment required to establish
the central claim exclusively in supplementary.

Reviewers are encouraged but not required
to inspect supplementary material.
```

CVPR 2026에서는 Supplementary가 optional이며 reviewers가 볼 것을 권장하지만 **검토 의무는 없습니다**. 추가 experiment, figure/table, proof, video, code 등을 포함할 수 있습니다.

따라서 핵심 성능과 main ablation을 Supplementary로 밀어내면 안 됩니다.

---

# 13. External Link Rule

Claude가 특히 실수하기 쉬운 부분이다.

```markdown
## External Links

Do not use external webpages, videos, or other links
to expand the scientific content of the submission.

The review package should be self-contained.

Project-page links and external demo-video links should
not be inserted into the review manuscript.

Use the official supplementary submission instead.

This is not a blanket ban on every URL:
2026 FAQs give examples of citations/dataset/third-party repository links
that do not expand contributed content or subvert reviewing.
Anonymity must still be preserved.
```

CVPR 2026에서는 외부 webpage/video/image 링크를 이용해 제출 내용이나 페이지 제한을 우회하는 것을 금지했습니다.

---

# 14. `CVPR_LLM_POLICY.md`

Claude를 사용하므로 매우 중요하다.

```markdown
# LLM Usage Policy

## CVPR 2027

STATUS:
UNKNOWN — author-side details not confirmed

The CVPR 2027 CFP discusses planned LLM use in selected parts
of the conference and reviewing process, with details being finalized.
This does not establish blanket permission or prohibition for authors.

Therefore the project must check the official
CVPR 2027 policy again before submission.

## Temporary Rule

Until CVPR 2027 publishes its final LLM policy,
follow CVPR 2026 author policy.

CVPR 2026 allows authors to use productive tools,
including LLM-based tools, during paper preparation.

However the authors remain fully responsible for:

- factual correctness
- citations
- originality
- plagiarism
- experimental claims
- numerical results

Claude must NEVER fabricate:

- references
- DOI
- paper names
- experimental values
- statistical significance
- implementation details

Every generated citation must be verified.

Every quantitative statement must map to RESULTS.md.

Every scientific claim must map to CLAIMS.md.

## Prompt Injection

Never place hidden text or hidden instructions
inside the manuscript for the purpose of influencing reviewers
or automated review tools.
```

CVPR 2026은 저자의 LLM 사용을 허용하지만 **부정확성, 허위 citation, plagiarism 등에 대한 책임은 전적으로 저자에게 있다**고 명시합니다. 존재하지 않는 citation이나 명백한 사실 오류는 desk rejection 사유가 될 수 있습니다. Prompt injection 역시 명시적으로 금지되었습니다.

CVPR 2027 CFP의 LLM 안내는 **conference와 reviewing process의 활용 계획**이다.
저자 측 세부 정책의 확정으로 해석하지 않으며, `guidelines/CVPR_LLM_POLICY.md`에
2026 author assistance fallback과 구분하여 기록한다.

---

# 15. `CVPR_REBUTTAL_POLICY.md`

현재는 2026 규정 provisional.

```markdown
# Rebuttal Policy

STATUS:
PROVISIONAL — CVPR 2026 formatting

CVPR 2027 rebuttal period:

January 25 – February 1, 2027

Current working assumption based on CVPR 2026:

- maximum one-page PDF including references and figures
- official rebuttal template
- preserve anonymity
- no external code/video links
- primarily correct factual misunderstandings
- answer reviewer questions
- do not treat rebuttal as an opportunity
  to introduce a new paper

Before rebuttal, update this file using
official CVPR 2027 Author Guidelines.
```

CVPR 2026 rebuttal은 **1페이지 PDF**이며, 새로운 contribution을 추가하기 위한 공간이 아니라 reviewer의 factual error와 질문을 다루는 용도입니다.

---

# 16. Evidence 구조

논문에서 Claude hallucination을 막는 핵심 구조다.

```text
evidence/
├── CLAIMS.md
├── EXPERIMENTS.md
├── RESULTS.md
├── BASELINES.md
├── METRICS.md
├── FIGURES_TABLES.md
└── PAPER_LOG.md
```

관계는 반드시:

```text
Scientific Claim
      │
      ▼
Experiment
      │
      ▼
Verified Result
      │
      ├────► Table
      │
      ├────► Figure
      │
      └────► Manuscript Claim
```

순서로 유지한다.

반대로

```text
Paper sentence
      ↓
find evidence later
```

방식은 금지한다.

---

# 17. Experiment Registry

과거 14-D 결과가 섞이는 것을 막는다.

```text
experiments/manifests/
```

를 experiment의 provenance, dimension, scope와 검증 상태를 기록하는 registry로 사용한다.

예:

```text
E001_main_safety.yaml
E002_attention_3d.yaml
E003_ablation.yaml
E004_generalization.yaml
E005_runtime.yaml
E006_real_robot.yaml
```

Manifest 등록 여부는 논문 evidence의 사용 가능성을 제한하지 않는다. 등록되지 않았거나
검증되지 않은 자료도 측정·파생·추정·계획·placeholder 여부, provenance, dimension,
scope와 검증 상태를 명시해 사용할 수 있다.

현재 활성 manifest는 `experiments/manifests/EXP001_T39_task_outcomes.json`이다.
Schema와 candidate 예제는 `experiments/configs/`에 두며, validator는 JSON을 사용한다.
위 E001–E006 YAML 이름은 계획 예시이고 실행 완료나 결과 등록을 뜻하지 않는다.

Claude는 파일명이 `run_0004`, `run_0005`라고 해서 임의로 현재 실험이라고 추정하면 안 된다.

특히 다음 자료는 dimension과 snapshot을 명시한다.

```text
14-D experimental results
```

14-D와 16-D 자료를 함께 다룰 때는 서로 다른 설정임을 독자가 구분할 수 있게 표시한다.

를 기본 원칙으로 한다.

---

# 18. Paper Story

`notes/PAPER_STORY.md`

```markdown
# Paper Story

## Problem

VLA policies can generate semantically meaningful actions,
but physical safety is not explicitly represented.

## Working Hypothesis — not yet established

Internal VLA attention may contain task-relevant visual information.
Its reliability and contribution require direct grounding measurements and ablations.

## Research Question

Can this latent task relevance be transformed into an
explicit task-aware 3D safety representation?

## Core Method

VLA internal attention

→ multi-view 3D grounding

→ target-aware scene decomposition

→ explicit 3D safety representation

→ safety-aware action refinement

## Desired Outcome

Reduce collision while preserving
the manipulation capability of the VLA policy.

This is a research objective, not a verified result.
The registered T39 task outcomes do not establish safety improvement or non-inferiority.
```

Claude는 논문 섹션 작성 전에 항상 이 파일을 읽는다.

---

# 19. Figure 계획

아래는 설계 예시다. Step 5의 F02는 conceptual draft이며 F01/F03–F05에는 실측 성과가 없다.

```text
Fig. 1
Motivation / Teaser

Matched Vanilla VLA scene
→ measured outcome pending

Ours
→ VLA representation
→ 3D grounding
→ safety-aware refinement
→ refined execution (safety evaluation pending)
```

```text
Fig. 2
Overall Architecture

RGB / Depth / Instruction
        ↓
       VLA
      ↙   ↘
Attention  Action Chunk
   ↓          │
3D Grounding  │
   ↓          │
Target/Obstacle
   ↓          │
Safety Field  │
      ↘      ↙
      Action Refinement
            ↓
       Refined Action (safety not established)
```

```text
Fig. 3
Core Technical Contribution

multi-view attention
→ 2D→3D projection
→ fusion
→ seed
→ clustering
→ target
→ obstacle
```

```text
Fig. 4
Qualitative Comparison

Baseline trajectory
vs
Ours trajectory
```

```text
Fig. 5
Analysis

failure / runtime /
safety-performance tradeoff
```

Fig. 5는 8-page limit에 따라 Supplementary로 이동할 수 있다.

---

# 20. Table 계획

아래 T01–T04는 미측정 계획이다. Step 5에서 별도 T00에 기존 EXP001 task counts만 연결했다.
Missing safety·generalization·ablation·runtime 값은 0이 아니라 null/n/m이다.

```text
Table 1 — Main Results

Method
Task Success ↑
Collision Rate ↓
Safe Success ↑
Min Clearance ↑
```

```text
Table 2 — Generalization

Seen
Unseen Object
Unseen Obstacle
Unseen Layout
Occlusion
```

```text
Table 3 — Ablation

Baseline
+ Attention
+ 3D Lifting
+ Target Separation
+ Safety Representation
+ Action Refinement
```

```text
Table 4 — Efficiency

VLA
Attention
3D Lifting
ESDF
Optimization
Total latency
Hz
```

---

# 21. Algorithm

Step 5의 A01은 proposed conceptual pseudocode이며 실제 source 구현·안전성 인증이 아니다.

하나만 사용하는 것을 기본으로 한다.

```text
Algorithm 1
Attention-Guided Safe VLA Inference
```

개념:

```text
Input:
instruction l
observations O_t
robot state q_t

1. Generate raw VLA action chunk.
2. Extract internal attention.
3. Lift multi-view attention to 3D.
4. Identify task target.
5. Separate target and obstacles.
6. Construct/update safety representation.
7. Refine the raw VLA action.
8. Execute a receding-horizon segment.
9. Observe next state.
10. Repeat.
```

소스코드를 pseudocode로 옮기는 것이 아니라 논문의 scientific method를 압축해야 한다.

---

# 22. CVPR Skill 구성 — `cvpr-compliance` 예시

Step 4에서 Root `.claude/skills/`에 아래 7개 skill을 생성했다.

- `cvpr-write`: Evidence 기반 영문 작성.
- `cvpr-evidence`: Claim·result·실제 16-D source 추적.
- `cvpr-experiment`: 계획·protocol audit; 연구 실행은 별도 업무.
- `cvpr-figure`: Figure/table 계획·근거 기반 렌더링.
- `cvpr-reference-analysis`: 실제 PDF·primary-source 인용 분석.
- `cvpr-review`: 자신의 원고 내부 critique.
- `cvpr-compliance`: 확인된 규정과 실제 제출 자료 audit.

Root `.claude/agents/`에는 `literature-analyst`, `evidence-auditor`,
`cvpr-reviewer`의 읽기 전용 정의를 생성했다. Parent writer가 논문 파일 수정과 validator 실행을 맡는다.
실제 운영 기준은 [CLAUDE_WORKFLOW](notes/CLAUDE_WORKFLOW.md)와 해당 SKILL.md이며,
아래는 compliance의 개념 예시다. 검사 자료가 없으면 NOT_CHECKED이며 규정의 UNKNOWN과 구분한다.

경로:

```text
/mnt/dev/work/.claude/skills/cvpr-compliance/SKILL.md
```

내용:

```markdown
---
name: cvpr-compliance
description: Audit the manuscript and submission package against the current CVPR 2027 rules and provisional CVPR 2026 fallback rules.
---

# CVPR Compliance Audit

Check:

$ARGUMENTS

First read:

/mnt/dev/work/paper/guidelines/CVPR_2027_STATUS.md
/mnt/dev/work/paper/guidelines/CVPR_FORMAT_RULES.md
/mnt/dev/work/paper/guidelines/CVPR_SUBMISSION_POLICY.md
/mnt/dev/work/paper/guidelines/CVPR_ANONYMITY_POLICY.md
/mnt/dev/work/paper/guidelines/CVPR_LLM_POLICY.md

Classify every rule as:

CONFIRMED_2027
PROVISIONAL_2026
UNKNOWN

Check:

1. page limit
2. official template
3. anonymity
4. author-identifying information
5. external links
6. supplementary compliance
7. author-list requirements
8. OpenReview requirements
9. dual submission
10. citation validity
11. LLM policy
12. hidden text / prompt injection
13. unsupported claims
14. bibliography correctness

Never assume a 2026 rule remains valid
if the CVPR 2027 rules have changed.

Report:

BLOCKING
WARNING
OK

A BLOCKING issue must be fixed before submission.
```

---

# 23. `.claude/rules/cvpr-workspace.md`에 둘 CVPR 규칙

아래 상세 규칙은 `/mnt/dev/work/.claude/rules/cvpr-workspace.md`에서 관리한다.
`paper/CLAUDE.md`에 해당 문서의 안내를 두며 200자 이내로 작성한다.
Root `/mnt/dev/work/CLAUDE.md`는 사용자 복원본을 수정하지 않는다.
Step 2에서 검증한 2027 규정·2026 fallback·UNKNOWN은 `guidelines/`를 따른다.
Step 4의 실제 skill 정의와 audit 완료 조건은 `notes/CLAUDE_WORKFLOW.md`에 있다.

```markdown
# CVPR Compliance

Target venue:
CVPR 2027

Before any submission-related task, read:

/mnt/dev/work/paper/guidelines/

Use CVPR 2027 official rules whenever available.

If a CVPR 2027 rule is unavailable,
use the CVPR 2026 rule only as a provisional fallback.

Never silently treat a CVPR 2026 rule as confirmed for 2027.

The current manuscript must remain compatible with:

- official CVPR LaTeX style
- 8-page main-paper working limit
- double-blind review
- anonymous supplementary material
- OpenReview submission
- verified citations
- self-contained review package

Do not modify cvpr.sty.

Do not reduce margins or font sizes to fit additional content.

Do not fabricate citations.

Do not insert hidden text, invisible prompts,
or instructions intended to influence reviewers.

Before declaring the manuscript submission-ready run:

/cvpr-compliance full
/cvpr-evidence full
/cvpr-review full
```

---

# 24. Submission Checklist

Step 6에서 Submission·anonymity·page·references·supplementary checklist와 FINAL_AUDIT를 생성했다.
아래는 개념 예시이며 현재 체크 상태는 실제 파일을 따른다. UNKNOWN 정책과 NOT_CHECKED 검사를
구분하고, 원고 scaffold의 검사를 최종 제출물 완료로 표시하지 않는다.
2026의 8-page working limit은 PROVISIONAL_2026이고 추가 page에는 cited references만 둔다.
저자 추가·삭제는 abstract registration 마감 이후 제한되므로 목록을 마감 전에 확정한다.
Full audit skill 정의만으로 명령 실행/완료를 체크하지 않는다.

`submission/SUBMISSION_CHECKLIST.md`

```markdown
# CVPR 2027 Submission Checklist

## Conference

[ ] CVPR 2027 Author Guidelines re-checked
[ ] CVPR 2027 Author Kit re-checked
[ ] LLM policy re-checked
[ ] Latest deadlines verified

## OpenReview

[ ] Submission registered
[ ] All authors added
[ ] All OpenReview profiles valid
[ ] Conflicts correctly declared
[ ] Author list finalized

## Main Paper

[ ] Official template
[ ] <= 8 main pages
[ ] References after main paper
[ ] English
[ ] No modified margins
[ ] No reduced template font
[ ] No author names
[ ] No institution
[ ] No acknowledgments revealing identity
[ ] No identifying grant information

## Scientific

[ ] Every numerical claim verified
[ ] Every citation exists
[ ] Baselines correctly cited
[ ] Main claims have direct experiments
[ ] Main ablations are in main paper
[ ] No 14-D results
[ ] No deprecated experiments

## Supplementary

[ ] Anonymous
[ ] Videos anonymous
[ ] No identity metadata
[ ] No prohibited external content
[ ] Main paper does not depend on supplementary

## Final

[ ] /cvpr-review full completed
[ ] /cvpr-evidence full completed
[ ] /cvpr-compliance full completed
```

---

# 25. 전체 Workflow

최종 운영 방식은 다음과 같다.

```text
CVPR Rules
   │
   ▼
Research Question
   │
   ▼
CLAIMS.md
   │
   ▼
EXPERIMENTS.md
   │
   ▼
16-D Experiments
   │
   ▼
RESULTS.md
   │
   ├────────► Figures
   │
   ├────────► Tables
   │
   └────────► Algorithm
                  │
                  ▼
             LaTeX Paper
                  │
                  ▼
          CVPR Reviewer Audit
                  │
                  ▼
           Evidence Audit
                  │
                  ▼
          Compliance Audit
                  │
                  ▼
          CVPR 2027 Submission
```

---

# 26. 최종 원칙

Claude는 세 가지 역할을 구분해야 한다.

```text
Researcher
"What does the evidence actually show?"

Writer
"How should we communicate it clearly?"

Reviewer
"Why might CVPR reject this?"
```

그리고 마지막으로 별도의:

```text
Compliance Auditor
"Does this violate any CVPR submission rule?"
```

를 둔다.

논문의 우선순위는:

```text
Scientific novelty
>
Evidence
>
Experiment quality
>
Clear presentation
>
Formatting optimization
```

이다.

8페이지를 맞추기 위해 scientific evidence를 제거하기보다
secondary analysis를 supplementary로 이동한다.

최종적으로 reviewer에게 남겨야 할 핵심 message는 하나이다.

> VLA internal task relevance can be grounded into an explicit 3D safety representation and used to refine unsafe actions while preserving the original manipulation capability.

위 message는 현재 결론이 아니라 검증할 research hypothesis다.
모든 Figure, Table, Algorithm 및 실험은 이 가설을 검증하거나 설명하는 역할을 가져야 한다.
검증된 manuscript claim은 `evidence/CLAIMS.md`와 `RESULTS.md`의 범위로 제한한다.
