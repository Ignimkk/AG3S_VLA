# Claude 전환 handoff

기존 session을 멈추고 아래 폴더를 각각 별도 VS Code 창으로 연 뒤 **새 session**을 시작한다.

| 폴더 | 작업 branch |
|---|---|
| `/mnt/dev/AG3S_VLA-development` | `feature/ag3s-development` |
| `/mnt/dev/AG3S_VLA-benchmark` | `feature/panda-benchmark` |

두 clone의 origin: `https://github.com/Ignimkk/AG3S_VLA.git`. 사용자가 최초 통합 commit과
legacy refs를 push했다고 확인했다. 이번 storage 경로 분리의 후속 commit은 사용자가 push한다.
현재 main은 원본 v2 tag SHA 자체가 아니라 paired v2 source와 공통 경로 수정을 통합한 commit이다.
원본 tag와 모든 원본 branch는 `legacy/*`로 보존한다. 원본 workspace에는 v2 tag가 없었다.
개선 branch는 latest benchmark/workspace checkpoint와 원본 후속 commit을 포함한다.
현재 기존 workspace의 dirty source는 checkpoint commit에 반영되어 있으나 원본 index와
working tree는 그대로 두었다. 원본 dirty 상태를 새 clone에 다시 덮어씌우지 않는다.

새 Claude에게 보낼 첫 메시지:

```text
AG3S_VLA 통합과 폴더 전환이 완료되었습니다. 현재 열린 checkout에서만 개발합니다.
CLAUDE.md, .claude/project-context.md, doc/BRANCHES.md,
doc/migration/source-manifest.json, doc/SHARED_STORAGE_AND_PYTHON.md,
doc/migration/STORAGE_IMPLEMENTATION_REPORT.md를 먼저 읽으세요.
Input은 /mnt/dev/AG3S_VLA-storage이며 자기 .venv-ag3s/bin/python을 선택하세요.
첫 응답에는 cwd, Git root, branch/HEAD, origin, 사용할 Python과 module __file__을 보고하세요.
개선 branch와 V11AB 평가 branch의 source 기준을 혼합하지 마세요.
기존 /mnt/dev/work의 Python source나 venv로 되돌아가지 마세요.
먼저 읽기 전용으로 전환과 남은 작업을 확인한 뒤 해당 branch의 개발을 이어가세요.
```

SessionStart/UserPromptSubmit hook은 실제 root·branch·HEAD·목적을 주입한다.
역할별 CLI는 `.claude/roles/launch.sh`가 자기 root로 권한 설정을 materialize한다.
기존 auto memory가 이전 구조를 주장하면 tracked 문서와 actual Git 정보를 따른다.
기존 세션 memory를 검증 없이 새 세션으로 복사하지 않는다. 외부 agent에게 자동 메시지를 보내지 않았다.

다음 구현 순서: Panda robot/camera/action adapter 검증 → LIBERO Spatial 소규모 평가 →
SafeLIBERO Spatial Level I/II pilot → 전체 SafeLIBERO → LIBERO-Safety 추가 평가.
아직 Panda adapter 기능과 benchmark rollout을 구현/실행하지 않았다.
GPU/VLA dependency 및 checkpoint, Panda geometry/camera calibration, 평가 protocol을 먼저 확정한다.

## V11AB benchmark 실험 계획 — 2026-10-12 개정

최신 실행 계획은 [develope_PLAN_SUMMARY.md](../develope_PLAN_SUMMARY.md)를 읽는다.
평가 대상은 paired-v2 **V11AB**로 고정한다. V7g/V8 선정 및 strong-rule/EE optimizer의 묵시적 대체 지침은 적용하지 않는다.
Panda adapter → LIBERO Spatial → SafeLIBERO Spatial I/II pilot → 전체 SafeLIBERO → LIBERO-Safety 순서로 진행한다.
Variant·공식 metric·통과 gate·고유 run 저장 규칙은 새 계획을 따른다. 이번 문서 개정은 rollout 수행 결과가 아니다.
