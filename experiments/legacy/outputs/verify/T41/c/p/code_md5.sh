#!/bin/bash
# md5 of every file the G processes can load from the snapshot ${AG3S_PROJECT_ROOT}-t41c and of the symlinked trees it reaches into.
# (T39 version + mujoco scenes/*.xml, where the hurdle lives.)  usage: code_md5.sh > file   (compare against p/code_md5_g0.txt)
cd ${AG3S_PROJECT_ROOT}-t41c && {
  find -L benchmark -path AG3S/ag3s/asset -prune -o -type f -name '*.py' -print
  find integration/rby1/bringup -type f -name '*.py'
  find -L integration/rby1/manipulation/src -type f \( -name '*.py' -o -name '*.json' \)
  find -L robots/rby1/description/models/rby1a/mujoco -maxdepth 1 -type f -name '*.xml'
  find -L robots/rby1/description/models/rby1a/mujoco/scenes -type f -name '*.xml'
  find -L third_party/openpi/src third_party/openpi/packages/openpi-client/src -type f -name '*.py'
} | LC_ALL=C sort | xargs md5sum
