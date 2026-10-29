# CVPR LaTeX draft scaffold

## 2026-10-16 현재 원고: 전체 AG3S의 1–3장

사용자 지시에 따라 기존 ATS–HSR–LTE–CCR 전체 방법론에 surface graph, feature memory, 3D state 개선을 통합했다. 활성 [main.tex](main.tex)는 **초록, Introduction, Related Work, Method, 개념도 2개, 참고문헌**을 포함한다. 실험·결과·결론·별도 한계 절은 source를 보존하고 main에서 제외했다.

현재 원고 빌드:

```bash
python3 paper/tools/latex/build_method.py
```

출력은 `paper/latex/build/main.pdf`다. [개정 내용과 구현/설계 경계](../notes/AG3S_METHOD_REVISION_20261016_KO.md), [현재 빌드 기록](../notes/AG3S_METHOD_REVISION_20261016_BUILD.json)을 먼저 읽는다. 이 검사는 활성 원고와 공식 style, 도식 provenance, 인용·layout만 확인하며 실험 evidence 또는 제출 준비를 인증하지 않는다.

이전 IFG 개발 결과 원고와 `build_ifg.py`는 [2026-10-15 기록](../notes/IFG_REVISION_20261015_KO.md)에 해당한다. 현재 main은 결과 수치를 포함하지 않으므로 해당 실험 원고용 빌드 검사를 사용하지 않는다. 아래 legacy `build.py`도 EXP001/T39의 이전 경로를 검사하므로 현재 1–3장용 명령과 구분한다.

공식 template 원본은 [template](../template/README.md)에 보존되어 있다. Review mode, 10pt/two-column/letterpaper와 기존 font/margins는 유지했다. 완성형 방법의 일부 통합 검증과 실험·결과 작성이 남아 있으므로 제출용 완성 원고는 아니다.

## Build

### VS Code / LaTeX Workshop

저장소의 `.vscode/settings.json`은 기본 recipe를 `Tectonic (cached)`로 지정한다.
컴파일러는 `paper/tools/bin/tectonic`이며, 현재 영구 저장소의 설치 파일을 가리킨다.
`latexmk` 설치나 shell PATH 변경 없이 사용할 수 있다. VS Code에서 `main.tex`를 열고
`LaTeX Workshop: Build LaTeX project`를 실행한다. 기존 설정이 계속 사용되면
`Developer: Reload Window` 후 다시 빌드한다.

PDF와 SyncTeX는 `paper/latex/build/main.pdf`, `paper/latex/build/main.synctex.gz`에 생성된다.
하위 section을 편집할 때는 파일 상단의 `% !TEX root = ../main.tex`를 통해 main을 빌드한다.
이 recipe는 원고 컴파일용이며 아래 helper의 실험 evidence 검증과는 별도다.

### 검증을 포함한 command-line build

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
초기 구성에서는 `/tmp/cvpr-latex-tools/tectonic` 0.17.0을 임시 사용했다.
현재 워크스페이스 컴파일러를 명시하려면 다음과 같이 실행한다.

```bash
python3 paper/tools/latex/build.py --target main --engine tectonic \
  --compiler paper/tools/bin/tectonic --offline
```

기존 연구 venv나 시스템 package는 변경하지 않았다.

출력: `paper/latex/build/main.pdf`, `paper/supplementary/build/supplementary.pdf`.
Build intermediate/log/PDF는 [paper .gitignore](../.gitignore)의 대상이며 원본/working 소스는 보존한다.
Main에는 supplementary를 input하지 않는다. 두 문서가 실제로 분리되는지 확인한다.

## Draft 수정

- Working [metadata.tex](metadata.tex)의 year=2027, paperID=`*****`는 준비용이다.
- [references.bib](references.bib)는 후속 개정에서 확장된 참고문헌을 담는다. 초기 [R1 분석](../notes/REFERENCE_ANALYSIS_R1.md)과 각 revision note의 출처 확인 범위를 구분한다. 포괄적인 최초성 검증을 완료한 것은 아니다.
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
