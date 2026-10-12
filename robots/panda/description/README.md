# Panda (LIBERO) robot description for AG3S

Benchmark-branch adapter assets (SL2, 2026-10-12). Read by `AG3S/adapters/panda/model.py`.

| file | what | how it was made |
|---|---|---|
| `panda_libero.urdf` | kinematics of the Panda LIBERO simulates (robosuite 1.4 `MountedPanda` + `PandaGripper`); no meshes | cuRobo `franka_panda.urdf` (submodule `third_party/curobo` 78fd485f, read only) with the hand / finger offsets corrected to robosuite's; table in the file header |
| `libero_panda_self_filter_cover_confined_hand_mount.json` | **default** self-filter covering capsules (role `self_filter_cover`), link frames of the URDF: 1036 capsules -> 4210 spheres (self-filter model 4271 with the 61 `franka.yml` spheres). Hand (`gripper0_right_gripper`) and mount bodies fitted with V11AB's confined rule | generated, see below; provenance (cover name, geometry digest, capsule digest, sources, fitter constants) inside. SL3 0.1, user decision 2026-10-12 |
| `libero_panda_self_filter_cover_unconfined.json` | comparison variant: V11AB's function on every body, 51 capsules -> 184 spheres (model 245) | the SL2 default `libero_panda_self_filter_cover.json`, renamed unchanged (SHA-256 `895b7dbf…`) |

Constraint spheres are not copied: they are read from cuRobo
`third_party/curobo/curobo/content/configs/robot/franka.yml` (`collision_spheres`, 61 spheres on 11
links) at build time.

## Regenerating the cover

```bash
cd /mnt/dev/AG3S_VLA-benchmark && source scripts/setup/env.sh
# 1. export the robot geometry from a real LIBERO env (.venv-libero, Python 3.8; ~10 s)
.venv-libero/bin/python benchmark/adapters/libero_panda/robot_geometry.py export --source libero --out <scratch>/geo_libero.npz
# 2. fit with V11AB's self_filter_covering_capsules (.venv-ag3s; default variant ~30 s, unconfined ~5 s)
OMP_NUM_THREADS=4 .venv-ag3s/bin/python -m AG3S.adapters.panda.cover --geometry <scratch>/geo_libero.npz \
    --variant confined_hand_mount --out robots/panda/description/libero_panda_self_filter_cover_confined_hand_mount.json --force
OMP_NUM_THREADS=4 .venv-ag3s/bin/python -m AG3S.adapters.panda.cover --geometry <scratch>/geo_libero.npz \
    --variant unconfined --out <scratch>/unconfined.json   # = the committed unconfined capsules (its JSON predates cover_name)
```

The export step runs from the default shell: `robot_geometry.py` re-executes itself with the launcher's
client environment (foreign `LD_LIBRARY_PATH` torch entries and `PYTHONPATH` dropped; SL3 0.2).

`tests/benchmark/test_panda_sl2.py` re-exports the geometry from the robosuite robot model, checks
that its geometry digest equals the one recorded in each cover JSON, refits both variants, and compares
capsule by capsule.

## Joint map (adapter `robot_state` -> URDF)

| `robot_state` index | MuJoCo joint | URDF joint | sign |
|---:|---|---|---:|
| 0–6 | `robot0_joint1..7` | `panda_joint1..7` | +1 |
| 7 | `gripper0_finger_joint1` (range [0, 0.04]) | `panda_finger_joint1` | +1 |
| 8 | `gripper0_finger_joint2` (range [−0.04, 0]) | `panda_finger_joint2` (axis −y, [0, 0.04]) | −1 |

Frames: `panda_link0` = MuJoCo `robot0_base` = `robot0_link0`; `panda_hand` = `robot0_right_hand`;
`panda_left/rightfinger` = `gripper0_left/rightfinger`; `panda_grip_site` = `gripper0_grip_site`;
`panda_eye_in_hand` = camera `robot0_eye_in_hand` with OpenCV axes.
