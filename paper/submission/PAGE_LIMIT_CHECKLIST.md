# Page Limit and Format Checklist

Updated: 2026-10-05 23:52 KST

2027 상세 format: `UNKNOWN`. `PROVISIONAL_2026`은 main 본문·figure·table 최대 8페이지이며
추가 페이지에는 cited references만 둔다. Rebuttal은 별도의 전체 1페이지 기준이다.
[FORMAT](../guidelines/CVPR_FORMAT_RULES.md), [REBUTTAL](../guidelines/CVPR_REBUTTAL_POLICY.md),
[2026 Author Guidelines](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines).

## 현재 scaffold

- [x] Pin한 원본과 두 working `cvpr.sty`·`.bst`의 bytes가 동일하다.
- [x] 10pt/two-column/letterpaper·review mode·line ruler·page number를 확인했다.
- [x] 실제 Times 계열 font embedding과 font 대체·undefined reference·overflow 여부를 검사했다.
- [x] Main 3페이지, supplementary 1페이지를 별도 PDF로 빌드했다.
- [x] Font encoding 호환을 위해 T1을 사용하며 margin/font-size/spacing을 축소하지 않았다.

Current build는 [WORKSPACE_VALIDATION](WORKSPACE_VALIDATION.json)을 따른다. Scaffold의 짧은 page count는
완성된 manuscript의 main-body/references 경계나 2027 최종 format 통과를 의미하지 않는다.
현재 References는 pending 문구이며 실제 cited-reference page가 아니다.

## 최종 manuscript / rebuttal

- [ ] 최신 2027 kit·정책으로 migration 필요 여부를 확인한다.
- [ ] Actual paper ID와 conference metadata를 설정한다.
- [ ] 실제 PDF에서 본문 끝과 cited references의 시작을 확인하고 8-page provisional 기준을 검사한다.
- [ ] 추가 reference page에 method·analysis·figure/table·uncited bibliography가 없는지 확인한다.
- [ ] Official style·font·margin·column·caption을 확인하고 불필요한 layout override를 제거한다.
- [ ] 모든 figure/table의 font size·clipping·legibility를 최종 page size에서 확인한다.
- [ ] Main에 supplementary가 append되지 않았는지 확인한다.
- [ ] Rebuttal 작성 시 최신 정책·template과 reference/figure 포함 총 page count를 별도 검사한다.

핵심 결과를 supplementary로 옮기거나 font/margin을 줄여 제한을 맞추지 않는다.
