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

Shared legacy inputs default to `/mnt/dev/work` via `AG3S_LEGACY_STORAGE`. New caches,
LIBERO configuration and outputs default to the current checkout. Historical evidence paths
inside archived documentation and verification JSON were preserved, not falsified as new runs.
