#!/usr/bin/env bash
# AMRC -> (Ubuntu PC disk) -> SDM migration relay. Run ON THE UBUNTU PC as a normal user.
# Why: no machine reaches both servers. AMRC is reachable only from the office wifi (Ubuntu PC), SDM only over VPN,
# and the two containers cannot reach each other (No route to host both ways, 2026-11-10/11).
# User decision 2026-11-11 17:xx: download everything to the Ubuntu PC before AMRC is removed (11-12 08:00),
# then later (from the same PC, on VPN) upload to SDM. Runbook: doc/active/migration/SDM_MIGRATION_RUNBOOK_20261110_KO.md §0-1
#
# usage (PC):
#   AMRC=root@172.21.121.110 AMRC_PORT=30587 bash pc_relay.sh check                 # ssh + free disk space
#   AMRC=root@172.21.121.110 AMRC_PORT=30587 bash pc_relay.sh download [kit|tree|storage|all]   # resumable, rerun = delta
#   AMRC=... AMRC_PORT=... bash pc_relay.sh verify                                   # what would still change (dry-run)
#   SDM=root@<SDM_IP> SDM_PORT=<port> EXISTING=archive bash pc_relay.sh upload     # later, on VPN
#   LOCAL=<dir> overrides the PC store (default ~/ag3s_migration); layout mirrors /mnt/dev.
set -uo pipefail
LOCAL=${LOCAL:-$HOME/ag3s_migration}
CMD=${1:-check}; PART=${2:-all}
die() { echo "ERROR: $*" >&2; exit 1; }
command -v rsync >/dev/null || die "install rsync: sudo apt-get install -y rsync"

# -a without owner/group/devices/specials: the PC user is not root. -H keeps hard links. --partial resumes big files.
RS=(rsync -rlptH --no-devices --no-specials --partial --info=progress2,stats1)

amrc_ssh() { : "${AMRC:?set AMRC=root@<amrc ip>}"; : "${AMRC_PORT:?set AMRC_PORT}"
  echo "ssh -p $AMRC_PORT -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new"; }

# AMRC side runs the rsync sender at lowest priority (GPU queues there are wall-clock sensitive).
from_amrc() { "${RS[@]}" -e "$(amrc_ssh)" --rsync-path="nice -n 19 ionice -c3 rsync" "$@"; }

storage_filter() {   # same rules as pull_from_amrc.sh do_storage (user decision 2026-11-11: visuals + records, npz/npy only on request)
  local extra=$LOCAL/.storage_extra_rules.txt
  $(amrc_ssh) "$AMRC" cat /mnt/dev/_migration/storage_extra_include.txt 2>/dev/null | grep -E '^[+-] ' > "$extra" || true
  SF=(--prune-empty-dirs --filter="merge $extra"
      --include='/checkpoints/***'
      --include='/benchmarks/libero-safety/assets_94ccca77/***'
      --include='/benchmarks/libero-safety/assets_zip_94ccca77/CHECKSUMS.json'
      --include='*/')
  for e in mp4 webm gif png jpg jpeg svg pdf html json jsonl log md csv tsv txt patch yaml yml tex; do SF+=(--include="*.$e"); done
  SF+=(--exclude='*')
  echo "storage filter: $(wc -l < "$extra") extra +/- rule(s) from AMRC"
}

TREE_EX=(--exclude='/.cache/uv/' --exclude='__pycache__/' --exclude='*.pyc')   # .git, worktrees, venvs, outputs/ all kept

case "$CMD" in
  check)
    $(amrc_ssh) "$AMRC" 'hostname; df -h /mnt/dev | tail -1' || die "cannot ssh to AMRC"
    mkdir -p "$LOCAL"; echo "PC store: $LOCAL"; df -h "$LOCAL" | tail -1
    echo "needed: about 200-230 GB (checkout ~54, _migration ~16, filtered storage incl. requested npz ~120-150)" ;;
  download)
    mkdir -p "$LOCAL"
    case "$PART" in kit|all) from_amrc "$AMRC:/mnt/dev/_migration/" "$LOCAL/_migration/" || die "kit" ;; esac
    case "$PART" in tree|all) from_amrc "${TREE_EX[@]}" "$AMRC:/mnt/dev/AG3S_VLA/" "$LOCAL/AG3S_VLA/" || die "tree" ;; esac
    case "$PART" in storage|all) storage_filter; from_amrc "${SF[@]}" "$AMRC:/mnt/dev/AG3S_VLA-storage/" "$LOCAL/AG3S_VLA-storage/" || die "storage" ;; esac
    date '+%F %T' > "$LOCAL/.last_download_$PART"; echo "download $PART done -> $LOCAL" ;;
  verify)   # dry-run: lists files that differ between AMRC and the PC copy (should be ~empty after the final pass)
    mkdir -p "$LOCAL"; storage_filter
    for pair in "_migration/::" "AG3S_VLA/::tree" "AG3S_VLA-storage/::storage"; do
      d=${pair%%::*}; k=${pair##*::}; extra=(); [ "$k" = tree ] && extra=("${TREE_EX[@]}"); [ "$k" = storage ] && extra=("${SF[@]}")
      n=$("${RS[@]}" -n -i -e "$(amrc_ssh)" "${extra[@]}" "$AMRC:/mnt/dev/$d" "$LOCAL/$d" 2>/dev/null | grep -c '^[<>ch.][f]' || true)
      echo "$d: $n file(s) still differ"
    done ;;
  upload)
    : "${SDM:?set SDM=root@<sdm ip>}"; : "${SDM_PORT:?set SDM_PORT}"
    S="ssh -p $SDM_PORT -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new"
    $S "$SDM" 'command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync)' || die "cannot ssh to SDM"
    # Guard (dev-4a 2026-11-11): SDM may already hold a fresh checkout built from SDM_FRESH_SETUP_20261111_KO.md.
    # Uploading the AMRC checkout on top would mix .git objects/index and venv site-packages. Choose explicitly:
    #   EXISTING=archive is the decided choice (2026-11-12, runbook §0-2): code/docs move via GitHub, this copy is reference only.
    #   EXISTING=replace  move SDM's checkout to /mnt/dev/AG3S_VLA.fresh-<stamp>, then upload AMRC's (AMRC venvs used)
    #   EXISTING=archive  keep SDM's checkout; put AMRC's at /mnt/dev/AG3S_VLA_amrc (reference only: its venvs
    #                     point at /mnt/dev/AG3S_VLA and are not usable there)
    dest=AG3S_VLA
    if $S "$SDM" test -e /mnt/dev/AG3S_VLA; then
      case "${EXISTING:-}" in
        replace) st=$(date +%Y%m%dT%H%M); $S "$SDM" mv /mnt/dev/AG3S_VLA "/mnt/dev/AG3S_VLA.fresh-$st" || die "mv"; echo "SDM checkout moved to AG3S_VLA.fresh-$st" ;;
        archive) dest=AG3S_VLA_amrc ;;
        *) die "SDM already has /mnt/dev/AG3S_VLA (fresh setup). Rerun with EXISTING=archive (decided 2026-11-12, runbook §0-2; code moves via GitHub only)" ;;
      esac
    fi
    "${RS[@]}" -e "$S" "$LOCAL/_migration/" "$SDM:/mnt/dev/_migration/" || die "upload _migration"
    "${RS[@]}" -e "$S" "$LOCAL/AG3S_VLA/" "$SDM:/mnt/dev/$dest/" || die "upload checkout"
    # storage: no --delete, so files the fresh setup already fetched (same checkpoints) are kept/overwritten, never removed
    "${RS[@]}" -e "$S" "$LOCAL/AG3S_VLA-storage/" "$SDM:/mnt/dev/AG3S_VLA-storage/" || die "upload storage"
    echo "upload done. On SDM: bash /mnt/dev/AG3S_VLA/scripts/setup/migration/bootstrap_sdm.sh" ;;
  *) die "unknown command: $CMD" ;;
esac
