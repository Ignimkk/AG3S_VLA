#!/usr/bin/env bash
# AMRC -> SDM migration: run ONCE inside the NEW SDM container after pull_from_amrc.sh (git, kit, tree) finished.
# Restores what lived only in the AMRC container (not on the volume) and checks the result against the AMRC manifest:
#   apt packages, /root config (Claude memory/sessions, codex, rclone gdrive token, uv), Qwen2.5-VL cache path,
#   GPU / venv interpreters / `uv pip freeze` diff / MuJoCo osmesa render.
# Runbook: doc/active/migration/SDM_MIGRATION_RUNBOOK_20261110_KO.md
# usage: bash scripts/setup/migration/bootstrap_sdm.sh        (WITH_TEX=0 skips the ~1 GB texlive install)
set -uo pipefail
M=/mnt/dev/_migration
ROOT=/mnt/dev/AG3S_VLA
[ "$(cd "$(dirname "$0")/../../.." && pwd)" = "$ROOT" ] || { echo "ERROR: checkout must live at $ROOT (venvs embed this path)"; exit 1; }
[ -d "$M/root_home" ] || { echo "ERROR: $M missing - run pull_from_amrc.sh kit first"; exit 1; }
FAIL=0; warn() { echo "!! $*"; FAIL=$((FAIL+1)); }

echo "== 1/5 apt packages (the AMRC container had these installed by hand; see manifest apt_history.txt)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git git-lfs curl ca-certificates build-essential cmake pkg-config \
  libosmesa6 libgl1 libglib2.0-0 libegl1 libglfw3 ffmpeg rclone rsync openssh-server poppler-utils || warn "apt base set"
if [ "${WITH_TEX:-1}" = 1 ]; then
  apt-get install -y -qq texlive-latex-base texlive-latex-recommended texlive-latex-extra \
    texlive-fonts-recommended texlive-bibtex-extra || warn "apt texlive"
fi

echo "== 2/5 restore /root config from the newest archive"
TAR=$(ls -1t "$M"/root_home/root_home_*.tar.gz | head -1); echo "archive: $TAR"
STAGE=$(mktemp -d /root/.restore.XXXX)
tar xzf "$TAR" -C "$STAGE"
for d in .claude .codex .config .local .libero .nv .copilot; do
  [ -e "$STAGE/$d" ] && { mkdir -p "/root/$d"; cp -a "$STAGE/$d/." "/root/$d/"; }
done
[ -e "$STAGE/.claude.json" ] && cp -a "$STAGE/.claude.json" /root/.claude.json
for f in .bashrc .profile .zshrc; do
  [ -e "$STAGE/$f" ] || continue
  [ -e "/root/$f" ] && ! cmp -s "$STAGE/$f" "/root/$f" && cp -a "/root/$f" "/root/$f.sdm-orig"
  cp -a "$STAGE/$f" "/root/$f"
done
mkdir -p /root/.ssh && chmod 700 /root/.ssh   # keep the platform's authorized_keys
[ -d "$STAGE/.ssh" ] && find "$STAGE/.ssh" -maxdepth 1 -type f ! -name authorized_keys -exec cp -an {} /root/.ssh/ \;
rm -rf "$STAGE"

echo "== 3/5 caches"
mkdir -p /root/.cache
# bl_tranche_queue_{c,d,e}.sh default AG3S_BASELINE_VLM_WEIGHT=/root/.cache/ag3s_local/Qwen2.5-VL-7B-Instruct
[ -e /root/.cache/ag3s_local ] || ln -s "$M/cache/ag3s_local" /root/.cache/ag3s_local
[ -d /root/.cache/ag3s_local/Qwen2.5-VL-7B-Instruct ] || warn "Qwen2.5-VL cache missing"

echo "== 4/5 GPU and interpreters"
export PATH=/root/.local/bin:$PATH
command -v uv >/dev/null && uv --version || warn "uv missing (/root/.local/bin/uv)"
nvidia-smi --query-gpu=name,memory.total,mig.mode.current --format=csv || warn "nvidia-smi"
for v in "$ROOT"/.venv-*; do
  printf '%-22s ' "$(basename "$v")"; "$v/bin/python" -c 'import sys; print(sys.version.split()[0])' || warn "$(basename "$v") interpreter broken"
done
"$ROOT/.venv-openpi-live/bin/python" -c 'import jax; d=jax.devices(); print("jax:", d); assert d[0].platform=="gpu"' || warn "jax GPU"
"$ROOT/.venv-baselines/bin/python" -c 'import torch; print("torch cuda:", torch.cuda.get_device_name(0))' || warn "torch GPU"
# env.sh defaults MUJOCO_GL=osmesa; EGL already failed on AMRC (2026-11-10), so only osmesa is required.
MUJOCO_GL=osmesa "$ROOT/.venv-libero/bin/python" -c 'import mujoco; m=mujoco.MjModel.from_xml_string("<mujoco/>"); r=mujoco.Renderer(m,32,32); r.render(); print("mujoco osmesa render OK")' \
  2>/dev/null || warn "mujoco osmesa render (libosmesa6?)"

echo "== 5/5 package sets vs AMRC manifest ($M/manifest/latest)"
MAN=$M/manifest/latest
if [ -d "$MAN" ]; then
  for f in "$MAN"/freeze_.venv-*.txt; do
    v=${f##*/freeze_}; v=${v%.txt}
    [ -x "$ROOT/$v/bin/python" ] || { warn "$v missing"; continue; }
    if diff -q <(uv pip freeze --python "$ROOT/$v/bin/python" 2>/dev/null | sed -E 's#file:///mnt/dev/AG3S_VLA#<ROOT>#') "$f" >/dev/null; then
      echo "$v: identical to AMRC"
    else warn "$v differs from AMRC:"; diff <(uv pip freeze --python "$ROOT/$v/bin/python" 2>/dev/null | sed -E 's#file:///mnt/dev/AG3S_VLA#<ROOT>#') "$f" | head -10; fi
  done
  [ "$(git -C "$ROOT" rev-parse HEAD)" = "$(cat "$MAN/git_head.txt")" ] && echo "git HEAD = AMRC manifest" \
    || echo "note: git HEAD differs from manifest (fine if AMRC committed after the manifest)"
else warn "no manifest"; fi
ls /root/.claude/projects/-mnt-dev/memory/MEMORY.md >/dev/null 2>&1 && echo "Claude memory restored" || warn "Claude memory"

echo; [ $FAIL = 0 ] && echo "BOOTSTRAP OK" || echo "BOOTSTRAP: $FAIL warning(s) above"
cat <<'EOF'
Next (idle GPU):  cd /mnt/dev/AG3S_VLA && source scripts/setup/env.sh
                  .venv-ag3s/bin/python scripts/setup/validate_live_env.py
then one short LIBERO smoke episode, compared with the same task/seed run on AMRC.
EOF
