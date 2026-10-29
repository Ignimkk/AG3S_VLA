# Figure 1 evidence — demo teaser

Prepared 2026-10-13 KST on `feature/ag3s-development` @ `cf38c9db`. Code line numbers refer to that tree.

## Data source (one recorded run; nothing re-run or simulated for this figure)

| Item | Value |
|---|---|
| Run | T53, arm **C1a** = current serving configuration (AG3S on). Flags: `AG3S/ag3s/docs/handoff/T46-CL3.verify.json` → `design.arms.C1a` (`--target-exclusion ball-owned --owned-max-dist-voxels 3 --target-field-policy exclude-authorized --links gripper` …) |
| Episode | `ep1842` (test split), prompt "Put the pear in the basket.", right arm, neighbour orange next to the pear |
| Run dir | `/mnt/dev/AG3S_VLA-storage/results/development/20261013T100210+0900_4a540f1a18d7_cb037c425d2f/verify/T53/runs/C1a/R/C1a_ep1842_s18421/` |
| Server dumps | `runs/C1a/server_829*/server_constraints/run_0000/chunk_*.npz`. Paired to the episode by `t53_records.pair_segment` (refined chunk == client planning rows) |
| Outcome of this episode | `T53.verify.json` → `numbers.per_episode[*].success_ct` = True for ep1842 (task done and no disallowed contact under the T53 definition) |
| GT grasp timeline | `gt/ep1842.grasp.npz`: `grasp_state` pre (chunks 0–19) → held (20–31) → placed (32–74); `touching` = fingers + pear at chunks 18–31 |

## Panels → code → status

| Panel | Code (entry) | Status |
|---|---|---|
| Keyframes | replay of recorded `qpos` (client step records) | Recorded state, re-rendered |
| RGB + VLA attention | `AG3S/trajopt/attention_policy.py:37-41` (`_LAYER=8, _HEAD=2`, denoise step 0, last action query), second model copy | **Implemented, serving default.** The head was not re-selected after T47–T51 (`AG3S_T47_LOG.md` "L8H2 유지") |
| 3D target grounding | `AG3S/ag3s/runtime/multiview.py:407 fuse_observations`; `stages/target_grounding.py:1611 ground_target`; `stages/admissibility.py:416 assess` | **Implemented, serving** |
| Target boundary vs neighbour | `fields/curobo_builder.py:416 _ball_owned_split` (T44 I1) | **Implemented.** Off by code default; **on in the evaluated arms** (CL2 C1a, CL3, T53) |
| Grasp contact (finger spheres) | sphere set and `target_permission` in the chunk dump; `AG3S/trajopt/grasp_latch.py:381 GraspLatch` | **Implemented.** Contact links = `ee_right`, `ee_finger_r1`, `ee_finger_r2` (135 of 270 query spheres in this chunk) |

## Verified numbers that may accompany this figure (not printed in it)

All from `AG3S/ag3s/docs/handoff/*.verify.json`. Their registration status is recorded separately in `paper/evidence/RESULTS.md` and does not determine manuscript use.

| Source key | Value | Meaning |
|---|---|---|
| `T53.verify.json` `numbers.task_success` | 9 (of 16 episodes) | Task success of the current serving arm on the T53 test set |
| `T53.verify.json` `numbers.success_ct` | 7 (of 16) | Task success with no disallowed contact |
| `T53.verify.json` `numbers.hold_chunks_total` | 37 | HOLD chunks over the 16 runs |

There is **no matched VLA-only run** for the current configuration. The only paired E0/E3b data (T39) predates target exclusion and uses a clear scene. Do not add a "VLA-only fails" panel to Figure 1 from existing data.

## Assets

All renders are **MuJoCo 3.11 replays of the recorded `qpos`** from the client step records (record step j = control step 8j), re-rendered at 1600x1200 from chosen viewpoints with the CPU renderer. Nothing is re-simulated. Overlays come from the server chunk dumps, or from FK on the recorded chunks. Generator: `paper/tools/plotting/paperbanana_render.py` (renders) and `paperbanana_compose.py` (montage, F3 charts). Each PNG has a `.json` sidecar with source paths, SHA-256 hashes and the values used. The earlier chart-style assets are kept in `assets/diagnostic/`.

| File | Content | Chunk |
|---|---|---|
| `assets/f1_key_approach.png`, `f1_key_grasp.png`, `f1_key_carry.png`, `f1_key_release.png` | replay keyframes | 10, 20, 26, 34 (GT grasp_state pre / held / held / placed) |
| `assets/f1_wrist_attention.png` | active (right) wrist camera re-rendered at 1280x960 + the served 16x16 attention, bicubic-upsampled | 10 |
| `assets/f1_grounding_scene.png` | ATS target cluster (orange) and neighbour surface (blue) as spheres in the scene; fruit meshes faded | 17 |
| `assets/f1_contact_scene.png` | contact links tinted gold on the target points at the closing chunk | 18 (latch `closing`; GT `touching` fingers + pear) |
| `reference_montage.png` | layout draft for PaperBanana | — |

## Limits

- The neighbour colour comes from the simulator's fruit position (45 mm radius) and is for display only; the method never receives GT.
- Keyframes and close-ups are replays of recorded states from new viewpoints; they are not the client's own camera stream.
- Attention is min-max scaled per view for display. Serving uses raw mass with percentile normalisation after fusion.
- The earlier stand-in video `results/rby1/rollouts/ep1807_apple.mp4` (configuration unknown) is **not** used here.
