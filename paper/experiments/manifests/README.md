# 16-D Experiment Manifest Registry

## IFG 추가 등록 (2026-10-15)

[EXP002](ifg/EXP002_IFG_development.json)는 `IFG_OFFLINE_1` 형식의 per-request dump 등록이다. 검증 명령: `python3 paper/tools/analysis/audit_ifg_evidence.py`. 기존 EXP001의 closed-loop trajectory schema와 분리하고 각 validator가 수행한 검사를 상태로 기록한다. REGISTERED의 범위는 offline development grounding/probing이다. [개정 기록](../../notes/IFG_REVISION_20261015_KO.md)을 참고한다.

Updated: 2026-10-05 19:45 KST

## 상태와 사용 범위

| 상태 | 의미 | 논문 표기 |
|---|---|---|
| `PLANNED` | 예상 protocol | 계획값으로 표시 |
| `CANDIDATE` | Source audit 전 자료 | candidate로 표시 |
| `REGISTERED` | registry에 기록된 자료; 실측·검증 완료를 뜻하지 않음 | 자료 유형·검증 상태와 scope 표시 |
| `DEPRECATED` | 대체·오염·문제 발견 자료 | deprecated 이유와 함께 표시 |

Manifest는 자료의 상태와 provenance를 기록하는 registry이며 논문 포함 여부를 결정하는
whitelist가 아니다. 등록·검증 여부와 관계없이 자료를 사용할 수 있고, 그 상태를 명시한다.
등록 수치는 `MEASURED`, `DERIVED`, `ESTIMATED`, `PLANNED`, `PLACEHOLDER`, `UNKNOWN`으로
구분할 수 있으며 source pointer와 audit은 가능한 경우에만 연결한다.

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
