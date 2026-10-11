#!/bin/bash
# md5 of every file the SUBTASK-d processes can load from the snapshot and of the symlinked trees it reaches into
# (same file set as T39 p/code_md5.sh, root ${AG3S_PROJECT_ROOT}-sd; plus the subtask probe asset).
cd ${AG3S_PROJECT_ROOT}-sd && {
  find -L benchmark -path AG3S/ag3s/asset -prune -o -type f -name '*.py' -print
  find AG3S/ag3s/asset/subtask_probe -type f
  find integration/rby1/bringup -type f -name '*.py'
  find -L integration/rby1/manipulation/src -type f -name '*.py'
  find -L robots/rby1/description/models/rby1a/mujoco -maxdepth 1 -type f -name '*.xml'
  find -L third_party/openpi/src third_party/openpi/packages/openpi-client/src -type f -name '*.py'
} | LC_ALL=C sort | xargs md5sum
