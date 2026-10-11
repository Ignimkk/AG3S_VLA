# CVPR LaTeX draft scaffold

[main.tex](main.tex)는 review mode, 10pt/two-column/letterpaper이며 7개 section을 연결한다.
현재 내용은 연구 가설·평가 계획과 제한된 기존 task outcome이다. 제출용 완성 원고가 아니다.
원본은 [template](../template/README.md), section/artifact 연결은
[MANUSCRIPT_MAP](../notes/MANUSCRIPT_MAP.md)을 따른다.

## Build

Repository root에서 실행한다. Build helper는 원본·working style hash, 현재 manifest와 T00 source
pointer·값·render source를 검증한 뒤 main/supplementary를 각각 빌드한다.

```bash
python3 paper/tools/latex/build.py --check-only
python3 paper/tools/latex/build.py --engine latexmk
python3 paper/tools/latex/build.py --engine tectonic
```

Compiler가 PATH에 없다면 `--compiler /실제/compiler/path`로 지정한다.
`--target main`, `--target supplementary`, 기본 `--target all`을 지원한다.
`latexmk`는 pdflatex/BibTeX 환경, Tectonic은 자체 package cache와 compiler가 필요하다.
사용 가능한 compiler가 없으면 helper는 실패 상태로 종료하며 build 성공을 만들지 않는다.
이 Step은 `/tmp/cvpr-latex-tools/tectonic` 0.17.0을 임시 사용했다. 임시 파일이 사라진 뒤에는
별도로 사용 가능한 compiler를 지정한다. 기존 연구 venv나 시스템 package는 변경하지 않았다.

출력: `paper/latex/build/main.pdf`, `paper/supplementary/build/supplementary.pdf`.
Build intermediate/log/PDF는 [paper .gitignore](../.gitignore)의 대상이며 원본/working 소스는 보존한다.
Main에는 supplementary를 input하지 않는다. 두 문서가 실제로 분리되는지 확인한다.

## Draft 수정

- Working [metadata.tex](metadata.tex)의 year=2027, paperID=`*****`는 준비용이다.
- [references.bib](references.bib)는 R1에서 검증한 3개 entry를 담고 Related Work에서 모두 인용한다.
  `hasverifiedreferences`를 true로 설정했다. [R1 분석](../notes/REFERENCE_ANALYSIS_R1.md)의
  제한된 확인 범위를 따르며 전체 survey와 novelty 판정은 미완료다.
- T00 source/render 연결을 변경할 때 source 검증과 sidecar hash를 함께 갱신한다.
- F02/A01 conceptual 내용, T01의 `n/m`, 모든 DRAFT/EVIDENCE PENDING 문구를 삭제하는 것만으로
  결과가 입증되거나 제출 준비가 완료되지 않는다.
- Style·margin·font·spacing을 조정해 page limit에 맞추지 않는다.
- 최종 manuscript 이전에 source implementation, citations, evidence와 최신 2027 kit을 검증한다.

## 검증 범위

[STEP5_BUILD_AUDIT](../notes/STEP5_BUILD_AUDIT.json)에 main/supplementary PDF와 optional artifact
빌드, source hash·font·페이지·metadata·시각 검토 범위를 기록한다.
첫 빌드에서 XeTeX의 TU font 대체를 발견해 entry에 T1 `fontenc`를 추가했다.
이는 official Times font의 encoding 호환 처리이며 font family/size나 style 변경이 아니다.
남은 compiler 경고는 cached third-party package의 encoding 메시지로 기록한다.
PDF의 font 대체·undefined reference·넘친 box 여부는 별도로 확인했다.
[fontenc primary documentation](https://ctan.org/pkg/fontenc).
