# Evidence and workspace validation

Python 3.10+와 기존 `jsonschema`, `PyYAML`을 사용한다. PDF 검사에만 `PyMuPDF`가 추가로 필요하다.
검사 도구는 research code·GPU/model/MuJoCo 실험을 실행하지 않는다.

```bash
python3 paper/tools/analysis/validate_manifests.py
python3 paper/tools/latex/build.py --check-only
python3 paper/tools/analysis/validate_workspace.py
python3 paper/tools/analysis/validate_workspace.py --main-pdf paper/latex/build/main.pdf --supplementary-pdf paper/supplementary/build/supplementary.pdf --report paper/submission/WORKSPACE_VALIDATION.json
```

`validate_workspace.py`는 구조·skill/agent YAML·문서 link/anchor·JSON/Python·Evidence/artifact·
kit/style source를 검사한다. PDF 두 경로를 제공하면 font/page/제한된 text/metadata·log도 검사한다.
PDF 검사를 생략하면 PASS_STATIC_CHECKS와 NOT_CHECKED를 보고하며 PDF 검증을 주장하지 않는다.
보고서를 쓰려면 `--report paper/...`를 지정한다. 지정하지 않으면 파일을 수정하지 않고 stdout으로 보고한다.
`--baseline`에는 별도로 보존한 path→SHA-256 JSON을 전달해 기존 파일의 변경 여부를 대조할 수 있다.
200자 제한은 `paper/CLAUDE.md`에만 적용한다. Root `CLAUDE.md`는 수정하지 않으며
현재 hash를 보고한다. 보존 확인은 작업 시작 시 저장한 baseline과 비교한다.
Bibliography의 외부 URI는 금지로 단정하지 않고 PDF report에 목록을 기록한다.
실제 외부 링크 허용·anonymity 판정은 [제출 정책](../../guidelines/CVPR_SUBMISSION_POLICY.md)의
해당 범위로 별도 검토한다. PDF provenance에는 bibliography source hash도 포함한다.

Step 6은 `/tmp/cvpr-pdf-tools`의 임시 PyMuPDF를 `PYTHONPATH`로 사용했다. 임시 도구가
사라지면 사용 가능한 Python 환경의 의존성으로 검사한다. 연구 venv·시스템 package는 변경하지 않았다.
PDF는 repository 안의 경로를 지정하며 compiler/TeX 로그를 같은 폴더에 보존한다.
빌드는 [LaTeX README](../../latex/README.md)의 compiler 설정을 따른다.

워크스페이스 검사 성공은 scientific validity·전체 anonymity·citation의 정확성이나
실제 OpenReview submission을 인증하지 않는다. [최종 audit](../../submission/FINAL_AUDIT.md)의
미해결 항목과 [source 범위](../../guidelines/SOURCES.md)를 함께 읽는다.
