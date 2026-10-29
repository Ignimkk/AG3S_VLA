# LEGACY_RETIREMENT — 구현

> writer: ag3s-implementer · 2026-10-12 KST · 읽는 쪽: verifier, scribe, lead

## 무엇을 했나

Historical sibling symlink를 frozen archive의 동일 relative subtree로 전환하기 전에 content identity와 안전 조건을 검증하는 read-only planner를 구현했다. Git ignore로 누락된 source도 SHA-256 비교에 포함하며, Git metadata·bytecode·명시적인 runtime environment/cache만 제외한다. 실제 symlink 변경, 외부 process 종료, `/mnt/dev/work` 삭제 및 GPU 작업은 수행하지 않았다.

## 바뀐 파일

| 파일 | 무엇이 | 왜 |
|---|---|---|
| `scripts/setup/plan_legacy_retirement.py` | JSON inventory → per-subtree hash/path/kind/executable manifest 및 202개 replacement 계획 | old source identity를 확인하지 않은 historical relink 방지 |
| `tests/test_legacy_retirement.py` | temp fixture 기반 safety regression 17개 | ignored source 누락, content 변화, runtime crawling, symlink escape, stale inventory 방지 |

## Planner 사용

```bash
MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/AG3S_VLA-development \
.venv-ag3s/bin/python scripts/setup/plan_legacy_retirement.py \
  --inventory <sibling-work-paths.json> \
  --archive /mnt/dev/AG3S_VLA-storage/archive/source-snapshot-20261011 \
  --output <plan.json>
```

`--legacy-root` 기본값은 `/mnt/dev/work`다. `--apply`는 없다. Root가 mutation 직전에 `validate_plan_current(plan)`을 호출하여 다시 content hash와 incoming link target을 확인할 수 있다. Planner는 concurrency lock, Git metadata preservation 또는 atomic replacement를 대신하지 않는다. Root가 관련 Git HEAD/index와 tracked/untracked modification 상태를 별도로 보존·검증해야 한다.

Exclusion policy는 `.git`, `__pycache__`, `.cache`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `.uv-cache`, `.tox`, `.nox`, `.venv*`, `*.pyc`, `*.pyo`이다. Inventory의 symlink는 traversal하지 않으며 target escape, broken target, archive→legacy checkout backlink, symlink parent redirect를 blocked 처리한다. Cross-subtree source link도 별도 identity 검증이 없으면 blocked 처리한다. Shared input은 explicit stable root allowlist인 storage `assets`, `datasets`, `checkpoints`, `recordings`, `legacy-inputs`만 허용한다.

## 단위 검증

```bash
MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/AG3S_VLA-development \
.venv-ag3s/bin/python -m pytest -q tests/test_legacy_retirement.py
```

**17 passed.** Meaningful temp fixtures verify missing ignored source, changed bytes, source/archive symlink escape, legacy backlink, shared-input allowlist, stale source/incoming target, duplicate inventory and Git/cache/venv traversal exclusion.

## verifier 가 알아야 할 것

- Initial raw inventory: `results/development/20261012T002906+0900_4292882_46c9b2ae/legacy-audit/sibling-work-paths.json` under storage.
- Initial read-only planner output: `/tmp/legacy-retirement-plan-before.json`; root should retain it in current run permanent evidence.
- Initial source/archive scan: **202 incoming links, 11 unique subtrees, 202 ready, 0 blocked**. All included path/kind/content/execute identities matched; ignored files were not excluded by `.gitignore`.
- This implementer result is not final independent post-mutation verification. Verifier owns subsequent inventory, unchanged old source/Git checks and before/after figures.
- AG3S algorithm, improvement history and runtime import APIs unchanged.

## 내가 기대하는 결과

All 202 historical sibling links can point to their frozen archive counterpart without altering historical source content. Remaining active `/mnt/dev/work` sessions require user-owned transition and final audit; this planner does not authorize deletion or process termination.


## 추가 발견 및 shared-backlink planner — 2026-10-12 KST

Independent verifier가 literal legacy target을 기준으로 전체 sibling link를 조사하여 **176개 추가 backlink**를 발견했다. Initial inventory의 202개는 archive source link이며, 추가 176개는 이미 storage로 연결된 old alias를 경유하는 **22개 sibling × 8개 namespace**다. 따라서 retirement 대상 전체는 **378개**다.

`build_backlink_plan(rows, *, allowed_targets, legacy=...)`와 `validate_backlink_plan_current(plan)`를 추가했다. Frozen source hash 비교와 별개로, 이 planner는 old alias가 정확히 하나의 symlink로 opt-in storage path에 직접 연결되는지 확인한다. Incoming 및 old alias의 raw target, resolved target, lstat device/inode/mode/mtime와 shared target의 device/inode/mode를 보존하고, mutation 직전 다시 비교한다. Shared target 전체 content를 traversal/hash하지 않아 약 360 GiB input 재읽기를 피한다.

Opt-in namespace는 `checkpoints`, `lerobot_transport`, `lerobot_atomic_20260825`, `cache`, `backups`, `data`, `outputs`, `logs`만 허용한다. `observed_backlink_targets(storage_root)`는 현재 관측된 exact storage target을 반환하지만 호출자가 `allowed_targets`로 명시적으로 전달해야 한다. Archive source의 immutable input allowlist를 broad results/cache/backups로 확장하지 않았다. 이 허용은 기존 historical backlink 경유 제거에만 적용되며 new output 정책을 바꾸지 않는다.

실제 raw `shared-backlink-before.json`로 작성한 `/tmp/legacy-shared-backlink-plan.json`의 결과는 **176 ready / 0 blocked**다. Root가 실제 atomic replacement를 수행하며, implementer는 external mutation을 수행하지 않았다. 추가 7개 safety test는 opt-in 누락, chained symlink, nonsymlink/broken old target, incoming raw 변경, old alias inode 변경, shared target inode 변경, storage traversal/hash 금지를 검사한다.
