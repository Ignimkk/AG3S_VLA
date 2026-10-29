# CVPR Paper Formatting Rules

## 2026-10-23 — `CONFIRMED_2027`

- 본문 **8 page** (figure · table 포함), CVPR style. 초과 허용은 cited references 와 **LLM Usage Statement** 뿐 [S16](SOURCES.md#s16).
- 공식 template 미사용은 리뷰 없이 reject, margin · font 를 상당히 바꾸면 desk reject [S16](SOURCES.md#s16).
- 2027 kit v1 [S17](SOURCES.md#s17): `cvpr.sty` 는 2026 대비 review copy 의 Paper ID 표기만 제거; margin · font · 2단 · 줄 번호 동일. `ieeenat_fullname.bst` · `preamble.tex` 동일.
- Working copy: `paper/latex/cvpr.sty`, `paper/supplementary/cvpr.sty` = 2027 원본 bytes. `\paperID` 는 더 이상 쓰지 않는다.


Target venue: CVPR 2027

Last verified: 2026-10-05 19:21 KST

## 현재 적용

2027의 상세 format은 `UNKNOWN`이다. 다음은 **`PROVISIONAL_2026`** 작업 기준이다.

| 항목 | 임시 기준 | 근거 |
|---|---|---|
| Main paper | 본문·figure·table 포함 최대 8페이지; 추가 페이지는 cited references만 | [S06](SOURCES.md#s06), [S09](SOURCES.md#s09) |
| Language | English | S09 |
| Document | 10pt, two-column, letterpaper; review mode | [S08](SOURCES.md#s08) |
| Layout | 공식 margin·font·column layout 유지 | [S10](SOURCES.md#s10) |
| Review 표시 | paper ID·line ruler·page number 확인 | S09, S10 |

[2026 Author Guidelines](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines), [공식 kit introduction](https://raw.githubusercontent.com/cvpr-org/author-kit/main/sec/1_intro.tex), [공식 kit formatting](https://raw.githubusercontent.com/cvpr-org/author-kit/main/sec/2_formatting.tex).

현재 확인한 2026 `main.tex`의 시작 설정:

```latex
\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage[review]{cvpr}
```

[공식 main.tex](https://raw.githubusercontent.com/cvpr-org/author-kit/main/main.tex).

## Template 관리 — `PROJECT_POLICY`

- 공식 원본은 `paper/template/`에 보존하고, working copy는 `paper/latex/`에 둔다.
- `cvpr.sty`를 수정하거나 margin·font size를 줄여 page limit을 맞추지 않는다.
- Step 5에서 kit을 재확인하고 tag `CVPR2026-v1(latex)`의 full commit과 archive/file checksum을 기록했다.
  원본·working style provenance는 [template README](../template/README.md)를 따른다.
- 현재 working fallback은 pin한 2026 kit이다. 2027 kit을 확인하면 원본을 별도 보존하고 migration한다.
- Supplementary는 main PDF와 분리한다. Rebuttal에는 별도 [규칙](CVPR_REBUTTAL_POLICY.md)을 적용한다.

2026 원본의 `confYear=2026`을 그대로 제출하지 않는다. 2027 값으로 바꾸는 것만으로
2027 공식 template을 사용했다고 보고하지 않는다.

출처 기록: [S06](SOURCES.md#s06), [S08](SOURCES.md#s08), [S09](SOURCES.md#s09), [S10](SOURCES.md#s10).

## Step 5 template 적용 기록

Template 재확인: 2026-10-05 21:01 KST. Full policy audit 시각은 위 Last verified를 따른다.
Pinned commit: `12909ae437f6dbc7435069cfdb4ca44c18e6a02f`.
Working metadata는 year=2027이며 upstream 원본은 2026을 유지한다. Main/supplementary의
`cvpr.sty`·`.bst`는 original과 동일한 bytes다. 별도 파일의 draft macro·algorithm package는
style·margin·font·spacing을 바꾸지 않는다. 세부 2027 format은 여전히 UNKNOWN이다.

2026 fallback의 kit/sample과 현재 scaffold의 build 성공은 2027 submission compliance 인증이 아니다.
출처: [S14](SOURCES.md#s14), [Build 안내](../latex/README.md).
