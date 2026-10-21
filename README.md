# AG3S_VLA

AG3S method: `AG3S/ag3s`, trajectory optimization: `AG3S/trajopt`.
KNOWS/SEAM: `Baseline`; LIBERO/SafeLIBERO/LIBERO-Safety: `benchmark`.
RB-Y1: `integration/rby1`, robot assets: `robots/rby1/description`.
Modified OpenPI is vendored with its original history at `third_party/openpi`.

Read [branch and merge rules](doc/active/BRANCHES.md), [migration manifest](doc/active/migration/source-manifest.json)
and [environment setup](environments/README.md) before development.
Two independent clones use the same `AG3S_VLA` origin. No remote was pushed during migration.

Run `source scripts/setup/env.sh` and `python scripts/setup/check_workspace.py` from your checkout.
Official benchmark submodules are pinned; SafeLIBERO's Python project is `benchmark/SafeLIBERO/safelibero`.
Panda adapter implementation and actual benchmark rollouts are separate subsequent work.
