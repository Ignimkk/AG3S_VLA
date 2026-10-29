#!/bin/bash
# SUBTASK-f: md5 of every file the processes can load from the snapshot ${AG3S_PROJECT_ROOT}-sg (same file set as SUBTASK-e p/code_md5.sh,
# plus rby1_description/.../mujoco/scenes/*.xml, because T40 edits pick_place_obstacles.xml in the main checkout; in work-sg
# rby1_bringup, rby1_manipulation, rby1_description are git archive 22d8dae, not symlinks).
cd ${AG3S_PROJECT_ROOT}-sg && {
  find -L benchmark -path AG3S/ag3s/asset -prune -o -type f -name '*.py' -print
  find AG3S/ag3s/asset/subtask_probe -type f
  find integration/rby1/bringup -type f -name '*.py'
  find -L integration/rby1/manipulation/src -type f -name '*.py'
  find -L robots/rby1/description/models/rby1a/mujoco -maxdepth 1 -type f -name '*.xml'
  find -L robots/rby1/description/models/rby1a/mujoco/scenes -type f -name '*.xml'
  find -L third_party/openpi/src third_party/openpi/packages/openpi-client/src -type f -name '*.py'
} | LC_ALL=C sort | xargs md5sum
