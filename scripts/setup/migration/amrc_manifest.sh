#!/usr/bin/env bash
# AMRC -> SDM migration, step 0 (run on AMRC; read-only except writing the manifest).
# Records what SDM must reproduce, so bootstrap_sdm.sh can diff against it:
#   git HEAD / local branches / worktrees / submodules / dirty+untracked files, `uv pip freeze` per venv,
#   hand-installed apt packages, and top-level sizes.
# usage: bash scripts/setup/migration/amrc_manifest.sh     -> /mnt/dev/_migration/manifest/<stamp>/ (+ latest symlink)
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=/mnt/dev/_migration/manifest/$(date +%Y%m%dT%H%M%S)
mkdir -p "$OUT"
cd "$ROOT"
git rev-parse --abbrev-ref HEAD > "$OUT/git_branch.txt"
git rev-parse HEAD > "$OUT/git_head.txt"
git for-each-ref --format='%(refname:short) %(objectname) %(upstream:short) %(upstream:track)' refs/heads > "$OUT/git_local_branches.txt"
git worktree list --porcelain > "$OUT/git_worktrees.txt"
git submodule status --recursive > "$OUT/git_submodules.txt"
git status --porcelain=v1 > "$OUT/git_status.txt"
for v in "$ROOT"/.venv-*; do
  [ -x "$v/bin/python" ] || continue
  uv pip freeze --python "$v/bin/python" 2>/dev/null | sed -E 's#file:///mnt/dev/AG3S_VLA#<ROOT>#' > "$OUT/freeze_$(basename "$v").txt"
done
grep -h 'Commandline: apt' /var/log/apt/history.log 2>/dev/null > "$OUT/apt_history.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/gpu.txt" 2>/dev/null
cat /etc/os-release > "$OUT/os_release.txt"
env | grep -E '^(CUDA_VERSION|NV_LIBCUBLAS_VERSION|NVIDIA_PYTORCH_VERSION|PYTORCH_VERSION)=' > "$OUT/image_env.txt"
ln -sfn "$(basename "$OUT")" /mnt/dev/_migration/manifest/latest
echo "manifest -> $OUT"; ls "$OUT"
