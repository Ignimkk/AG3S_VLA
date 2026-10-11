# 16-D Experiment Manifest Registry

Updated: 2026-10-05 19:45 KST

## 상태와 사용 범위

| 상태 | 의미 | 논문 결과로 사용 |
|---|---|---|
| `PLANNED` | 예상 protocol | 불가 |
| `CANDIDATE` | Source audit 전 자료 | 불가 |
| `REGISTERED` | 실제 16-D 자료·provenance·검증 key와 scope를 연결한 등록 | 해당 scope만 가능 |
| `DEPRECATED` | 대체·오염·문제 발견 자료 | 불가; 기록 삭제 대신 이유 보존 |

파일 존재만으로 whitelist 등록이 성립하지 않는다. `paper_eligible=true`와
`REGISTERED`, 실제 source 검증 통과가 함께 필요하다. 이 status는 모든 과학적 주장이나
submission readiness를 승인하는 표시가 아니다.

[Schema](../configs/manifest.schema.json)는 JSON Schema 2020-12를 사용하며
state/action dimension을 16으로 제한한다. [Candidate 예제](../configs/manifest.example.json)는
`configs/`에 두어 활성 registry와 분리했다. YAML을 사용하려면 같은 data model로 변환한 뒤
검증해야 한다. 현재 validator는 JSON만 읽는다.

## 현재 등록

- [EXP001](EXP001_T39_task_outcomes.json): T39의 descriptive task outcome만 등록.
- [Source audit](../summaries/EXP001_source_audit.json): 실제 배열 header·metadata·완료 자료·hash 확인.
- 후속 후보는 [EXPERIMENTS](../../evidence/EXPERIMENTS.md)에 있고 아직 registry에 없다.

## 검증 실행

Workspace root에서:

```bash
python3 paper/tools/analysis/validate_manifests.py
```

특정 manifest 또는 예제:

```bash
python3 paper/tools/analysis/validate_manifests.py paper/experiments/configs/manifest.example.json
```

Python 3.10 이상과 `jsonschema`가 필요하다. 현재 환경에서 실행 확인했다.
Validator는 schema, source hash, 연구 log의 선택된 section hash, 실제 NPY array header,
metadata·completion·pairing, `verify.numbers`의 JSON pointer와 값·denominator를 확인한다.
GPU, model inference, MuJoCo, 기존 연구 script를 실행하지 않는다.

Research log는 선택된 T39 section을 hash하므로 다른 절의 append만으로 등록을 무효화하지 않는다.
선택된 절이나 verify/raw 자료가 바뀌면 re-audit가 필요하다.

## Traceability

CLAIMS → planned protocol → manifest → verify/raw source → RESULTS → figure/table → manuscript.
결과를 다른 snapshot·metric·task·scope로 옮기거나 14-D 자료를 등록하지 않는다.
Large raw data·model code는 복사하지 않고 source path와 hash를 보존한다.
