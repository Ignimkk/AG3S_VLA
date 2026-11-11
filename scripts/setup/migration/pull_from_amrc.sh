#!/usr/bin/env bash
# AMRC -> SDM migration: run INSIDE the NEW SDM container (NAS volume mounted at exactly /mnt/dev).
# Pulls over SSH from the AMRC container. AMRC itself cannot reach SDM ("No route to host", 2026-11-10).
#   AMRC_HOST / AMRC_PORT = SSH address of the AMRC container (AMRC portal: [컨테이너 목록] > container > [설명] > SSH)
# Runbook: doc/active/migration/SDM_MIGRATION_RUNBOOK_20261110_KO.md
#
# usage: AMRC_HOST=172.21.121.x AMRC_PORT=3xxxx bash pull_from_amrc.sh <mode>
#   probe    TCP reachability only
#   git      clone from GitHub (if absent), then fetch ALL local branches of the AMRC checkout over SSH
#            (unpushed commits included), check out AMRC's branch at AMRC's HEAD, init submodules, recreate worktrees
#   kit      /mnt/dev/_migration   (root config tarball, Qwen cache, manifest)                 ~16 GB
#   tree     working trees over the clone: gitignored + untracked + uncommitted files, venvs, .python, .runtime,
#            .cache (minus uv), outputs/ ... (.git excluded)                                     ~54 GB
#   storage  /mnt/dev/AG3S_VLA-storage, FILTERED: visuals+records+checkpoints+LS assets (+extra list)
#   all      git -> kit -> tree -> storage
# Every rsync mode is resumable: rerun the same mode. Nothing is ever deleted on SDM.
set -uo pipefail
: "${AMRC_HOST:?set AMRC_HOST}"; : "${AMRC_PORT:?set AMRC_PORT}"
MODE=${1:-probe}
ROOT=/mnt/dev/AG3S_VLA
GITHUB=https://github.com/Ignimkk/AG3S_VLA.git
SSH="ssh -p $AMRC_PORT -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
R="root@$AMRC_HOST"
die() { echo "ERROR: $*" >&2; exit 1; }

probe() {
  if timeout 5 bash -c "echo > /dev/tcp/$AMRC_HOST/$AMRC_PORT" 2>/dev/null; then echo "TCP OK: SDM reaches AMRC $AMRC_HOST:$AMRC_PORT"
  else echo "TCP FAIL: SDM cannot reach AMRC -> runbook section 'SSH 전송이 안 될 때'"; exit 2; fi
}

ensure_ssh() {
  command -v rsync >/dev/null && command -v git >/dev/null || { apt-get update -qq; apt-get install -y -qq rsync git openssh-client; }
  mkdir -p /root/.ssh && chmod 700 /root/.ssh
  [ -f /root/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N '' -f /root/.ssh/id_ed25519 -q
  if ! $SSH -o BatchMode=yes "$R" true 2>/dev/null; then
    echo "Installing this SDM key on AMRC (enter the AMRC container password once):"
    ssh-copy-id -p "$AMRC_PORT" -i /root/.ssh/id_ed25519.pub "$R" || die "key install failed"
  fi
}

# The rsync sender runs ON AMRC (CPU + NFS IO there): lowest priority, optional bandwidth cap (BWLIMIT=KB/s, e.g. 50000).
# Tell the AMRC GPU-queue sessions before a big pull: closed-loop runs are wall-clock sensitive.
RS=(rsync -aH --numeric-ids --partial --info=progress2,stats1 --rsync-path="nice -n 19 ionice -c3 rsync")
[ -n "${BWLIMIT:-}" ] && RS+=(--bwlimit="$BWLIMIT")
pull() { "${RS[@]}" -e "$SSH" "$@"; }

do_git() {
  if [ ! -d "$ROOT/.git" ]; then
    git clone "$GITHUB" "$ROOT" || die "clone failed"
  fi
  cd "$ROOT"
  git remote get-url amrc >/dev/null 2>&1 || git remote add amrc "ssh://$R:$AMRC_PORT/mnt/dev/AG3S_VLA"
  GIT_SSH_COMMAND="$SSH" git fetch amrc '+refs/heads/*:refs/remotes/amrc/*' || die "fetch from AMRC failed"
  local br head
  br=$($SSH "$R" git -C /mnt/dev/AG3S_VLA rev-parse --abbrev-ref HEAD)
  head=$($SSH "$R" git -C /mnt/dev/AG3S_VLA rev-parse HEAD)
  for b in $(git for-each-ref --format='%(refname:lstrip=3)' refs/remotes/amrc); do
    [ "$b" = "$(git branch --show-current)" ] || git branch -f "$b" "amrc/$b" >/dev/null 2>&1 || echo "branch $b in use by a worktree: left as is"
  done
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "tracked files already modified (tree was pulled): skipping checkout/reset so nothing is discarded"
  else
    git checkout -q "$br" && git reset -q --hard "amrc/$br"
  fi
  [ "$(git rev-parse HEAD)" = "$head" ] || die "HEAD mismatch: SDM $(git rev-parse HEAD) vs AMRC $head"
  git submodule update --init --recursive || die "submodule update failed"
  # recreate AMRC's extra worktrees (e.g. .claude/worktrees/stage-track) on the same branch
  $SSH "$R" git -C /mnt/dev/AG3S_VLA worktree list --porcelain | awk '/^worktree /{p=$2} /^branch /{sub("refs/heads/","",$2); print p, $2}' |
  while read -r path b; do
    [ "$path" = /mnt/dev/AG3S_VLA ] && continue
    [ -e "$path" ] && { echo "worktree exists: $path"; continue; }
    git worktree add -q "$path" "$b" && echo "worktree: $path [$b]"
  done
  echo "git OK: $br @ $head"
}

do_tree() {
  [ -d "$ROOT/.git" ] || die "run mode 'git' first"
  local ex=(--exclude='.git' --exclude='/.cache/uv/' --exclude='__pycache__/' --exclude='*.pyc')
  # main tree; extra worktrees are separate trees, pulled one by one below
  pull "${ex[@]}" --exclude='/.claude/worktrees/' "$R:$ROOT/" "$ROOT/" || die "tree rsync failed"
  $SSH "$R" git -C /mnt/dev/AG3S_VLA worktree list --porcelain | awk '/^worktree /{print $2}' | while read -r path; do
    [ "$path" = "$ROOT" ] && continue
    [ -d "$path" ] && pull "${ex[@]}" "$R:$path/" "$path/"
  done
  cd "$ROOT" && echo "--- git status vs AMRC manifest (should list the same files)" && git status --short | head -40
}

# storage: user decision 2026-11-11 -- move visuals + records only, never the bulk npz/npy (~174 GB of results).
# Always: checkpoints/, extracted LIBERO-Safety assets (may hold npy/meshes, so taken whole).
# Extra npz/npy that sessions asked for: /mnt/dev/_migration/storage_extra_include.txt on AMRC (read fresh each run).
do_storage() {
  local extra=/tmp/storage_extra_include.txt
  $SSH "$R" cat /mnt/dev/_migration/storage_extra_include.txt 2>/dev/null | grep -E '^[+-] ' > "$extra" || true
  local f=(--prune-empty-dirs --filter="merge $extra"
    --include='/checkpoints/***'
    --include='/benchmarks/libero-safety/assets_94ccca77/***'
    --include='/benchmarks/libero-safety/assets_zip_94ccca77/CHECKSUMS.json'
    --include='*/')
  for e in mp4 webm gif png jpg jpeg svg pdf html json jsonl log md csv tsv txt patch yaml yml tex; do f+=(--include="*.$e"); done
  f+=(--exclude='*')
  echo "storage filter: $(wc -l < "$extra") extra +/- rule(s) from AMRC"
  pull "${f[@]}" "$R:/mnt/dev/AG3S_VLA-storage/" /mnt/dev/AG3S_VLA-storage/
}

case "$MODE" in
  probe)   probe ;;
  git)     probe; ensure_ssh; do_git ;;
  kit)     probe; ensure_ssh; pull "$R:/mnt/dev/_migration/" /mnt/dev/_migration/ ;;
  tree)    probe; ensure_ssh; do_tree ;;
  storage) probe; ensure_ssh; do_storage ;;
  all)     probe; ensure_ssh; do_git && pull "$R:/mnt/dev/_migration/" /mnt/dev/_migration/ && do_tree &&
           do_storage ;;
  *)       die "unknown mode: $MODE" ;;
esac
