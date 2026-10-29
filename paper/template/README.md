# Official Author Kit provenance

현재 기준: `CONFIRMED_2027`, CVPR 2027 author kit **v1 (LaTeX)**, 2026-10-23 KST 내려받음.

- 공식 안내: [CVPR 2027 Author Guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)
- 원본 archive: [author-kit-CVPR2027-v1-latex.zip](https://media.eventhosts.cc/Conferences/CVPR2027/author-kit-CVPR2027-v1-latex.zip), 로컬 사본 `archive_zip/`, sha256은 [PROVENANCE.json](PROVENANCE.json).
- [cvpr2027_official](cvpr2027_official/): archive의 12개 파일을 원본 bytes 그대로 보존했다.
- Working `paper/latex/cvpr.sty` · `paper/supplementary/cvpr.sty` · `.bst` 는 2027 original과 같은 bytes다.
- 2026 kit 대비 차이: `cvpr.sty` 에서 review copy 의 Paper ID 표기 제거 (README: "removed the Paper ID to reduce the risk of leaking IDs"), `main.tex` · `rebuttal.tex` 의 `\paperID` 제거와 `\confYear` 2027. margin · font · 2단 · 줄 번호는 같다.
- 이전 fallback (2026 kit, `PROVISIONAL_2026`) 은 [cvpr2026_official](cvpr2026_official/) 과 [PROVENANCE_2026.json](PROVENANCE_2026.json) 에 이력으로 보존한다.

검증: `python3 paper/tools/latex/build.py --check-only`, 원고 빌드 `python3 paper/tools/latex/build_cvpr_v1.py`.
