# Official Author Kit provenance

현재 fallback: `PROVISIONAL_2026`, tag `CVPR2026-v1(latex)`,
commit `12909ae437f6dbc7435069cfdb4ca44c18e6a02f`.

- [공식 pinned repository](https://github.com/cvpr-org/author-kit/tree/12909ae437f6dbc7435069cfdb4ca44c18e6a02f)
- [PROVENANCE.json](PROVENANCE.json): archive URL/hash, tag/full commit, 시각, original 파일별 SHA-256.
- [cvpr2026_official](cvpr2026_official/): 공식 archive의 14개 파일을 원본 bytes 그대로 보존했다.
- Working `cvpr.sty`·`.bst`는 original과 동일한 bytes다. Working manuscript/preamble만 별도로 작성한다.

2027 Guidelines URL의 404와 repository/tag의 2026 표시는 확인한 관찰이다.
2027 kit 전체의 부재를 단정하지 않으며 확인 상태는 UNKNOWN으로 유지한다.
Working confYear=2027은 local target 변경이며 공식 2027 kit으로 보고하지 않는다.
새 공식 2027 kit 확인 시 별도 원본·provenance를 보존하고 working copy를 migration한다.
검증: `python3 paper/tools/latex/build.py --check-only`.
