# Independent environments

Run `source scripts/setup/env.sh` in each checkout. Create a separate virtual environment,
then install the CPU requirements in `environments/ag3s/requirements.txt`. Install the local
manipulation package from `integration/rby1/manipulation` when needed. Do not share editable
installs or a writable virtual environment across the two branches.

OpenPI has its own `third_party/openpi/pyproject.toml` and `uv.lock`. cuRobo has separate
requirements at `environments/curobo/requirements.txt`; preserve its CUDA/Torch constraints.
Full GPU/VLA environment installation is not implied by path migration.

LIBERO variants all export a `libero` Python package. Use separate benchmark environments
or install exactly one variant in the selected environment; do not editable-install all
three simultaneously. SafeLIBERO's project is `benchmark/SafeLIBERO/safelibero`.
Model weights and additional upstream assets are not downloaded as part of repository migration.

Shared inputs default to the sibling `AG3S_VLA-storage` directory via `AG3S_STORAGE_ROOT`.
`AG3S_LEGACY_STORAGE` is a derived compatibility alias to its `legacy-inputs` namespace;
it no longer selects an old source checkout. New caches, LIBERO configuration and outputs
default to the current checkout. See `doc/SHARED_STORAGE_AND_PYTHON.md` for interpreter selection. Historical evidence paths
inside archived documentation and verification JSON were preserved, not falsified as new runs.

CPU bootstrap: `bash scripts/setup/create_cpu_env.sh` installs Python 3.11.16 and
the recorded CPU pins into `.venv-ag3s` of this checkout. Package files are copied,
not linked across the two environments. Source `scripts/setup/env.sh` before tests.


## Editable package

`pyproject.toml`은 직접 관리 Python package만 editable 설치한다.
버전 `0.0.0`은 package 메타데이터용이며 AG3S release나 실험 버전이 아니다.
수치 dependency는 각 environment requirements/기존 OpenPI lockfile을 사용한다.
Robot assets·submodule·대형 input을 wheel에 포함하지 않으므로 이 프로젝트는
`uv pip install --python .venv-ag3s/bin/python --no-deps -e .`로 개발 checkout에서 사용한다.
ROS manipulation/OpenPI 실행 경로는 `source scripts/setup/env.sh`가 설정한다.

## Verified benchmark live environments (2026-10-12 KST)

`bash scripts/setup/create_live_env.sh` creates this checkout's `.venv-openpi-live`;
pass `.venv-curobo` for a separate producer. Both use own-root Python 3.11.16 and
copied wheel installations without system-site-packages. Historical inventory constraints
and validated transitive constraints are committed separately. The one documented
metadata exception is gym-aloha's MuJoCo <3 requirement; live MuJoCo is 3.11.0.
See `doc/migration/LIVE_ENVIRONMENT_IMPLEMENTATION_REPORT.md` for checks and limits.
The development checkout's environment was not modified.
