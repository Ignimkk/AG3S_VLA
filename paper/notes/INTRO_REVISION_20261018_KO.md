# Introduction 구성 개정 — 2026-10-18 KST

사용자 요청: 앞서 합의한 네 가지 편집 개선을 연구동기 → 배경·한계 → 문제정의 → 통찰·방법 → 근거·기여의 다섯 단계에 연결한다.

## 반영 내용

| 단계 | 실제 문단 | 변경 |
|---|---|---|
| 동기 | 1 | 사과를 잡는 사례를 압축하고 identity/extent/permission 구분을 첫 문단에 제시 |
| 배경·한계 | 2 | VLA, geometric refinement, grounding 및 attachment의 배경을 연결. 과도한 제외와 부족한 target coverage라는 양쪽 문제를 설명 |
| 문제정의 | 3 | frozen VLA 내부 정보·calibrated depth·robot feedback을 입력으로, 링크·상태별 collision representation을 구성하는 문제를 명시. Contact links/support는 주어진 규칙 |
| 통찰·방법 | 4–5 | Feature boundary fusion의 개발 관찰에서 정보별 역할 분리로 연결. 구현 세부 목록을 줄이고 전체 AG3S의 grounding→constraint→execution 흐름 유지 |
| 근거·기여 | 6 및 마지막 목록 | 확인된 offline grounding/consistency 근거와 미확정 switching 효과를 구분. 이웃 보호와 필요한 접촉 유지가 함께 검증되어야 함을 명시 |

네 편집 개선은 각각 사례 압축(1), 핵심 통찰 전진 배치(1·4), 설계 이유 중심 방법 설명(4·5), 실증 근거 추가(6)에 반영했다. 다섯 단계는 다섯 문단을 강제하는 형식이 아니며, 통찰과 방법을 나누어 본문은 여섯 문단이다.

## 근거와 제한

- `python3 paper/tools/analysis/audit_ifg_evidence.py`를 이번 작업에서 실행하여 PASS 확인. 등록된 EXP002와 현재 원자료/source/집계의 일치를 재검증했다.
- 개발 데이터의 matched-mask extent 개선과 feature fusion 진단, consistency 후보 신호만 요약했다. 효과 크기·표·본격적인 실험 절은 이번 범위에 추가하지 않았다.
- Identity switching의 추가 효과는 inconclusive로 명시했다. 근거는 선택된 180프레임/32개 dev episode 및 replay depth이며, held-out 또는 closed-loop 성능으로 확장하지 않았다.
- 최근 사용자 보고의 SURF 실패 원인이나 robot mask 영향을 확정하지 않았다. 불완전한 target coverage가 접촉을 막을 수 있다는 일반적 문제 설명은 해당 실패의 원인을 단정한 문장이 아니다.
- 검증된 최종 실행 결과가 준비되면 근거 문단을 교체·확장하도록 TeX 주석을 남겼다. 성공 결과를 전제하지 않는다.

## 변경 및 검증 범위

- 원고 수정: `paper/latex/sections/01_introduction.tex`만 변경. Abstract/Related Work/Method/실험 source/연구 코드는 유지.
- `python3 paper/tools/latex/build_method.py` PASS. 총 7페이지, 참고문헌 6페이지 시작, unresolved citation/reference와 overfull box 없음, 공식 style hash 유지.
- Introduction 후반과 Related Work로 이어지는 2–3페이지를 시각 확인했다. 기존 cached package UTF-8 경고는 남아 있다.
- 현재 PDF: `paper/latex/build/main.pdf`.
- 이번 빌드 결과는 `INTRO_REVISION_20261018_BUILD.json`에 저장했다. 빌드 도구가 고정 경로에 출력하는 20261016 기록은 이전 내용으로 복원해 역사 기록을 보존했다.
- Before/after는 `INTRO_REVISION_20261018.diff`에 기록했다. 최종 제출 검증이나 새 실험 실행이 아니다.
