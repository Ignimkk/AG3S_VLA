# Figure 3 evidence — full method pipeline

Prepared 2026-10-13 KST on `feature/ag3s-development` @ `cf38c9db`, from a read-only code audit. Code default vs evaluated arm: the arm is T46-CL3 / T53 **C1a** (`T46-CL3.verify.json` → `design`).

## Modules, I/O and status

| Box | Code (entry) | Inputs → outputs | Status / default |
|---|---|---|---|
| VLA serving pass | `AG3S/trajopt/safe_policy.py` `SafePolicy.infer` → policy | 3 RGB 224², prompt, 16-D state → chunk 50×16 (absolute joints) | Implemented. VLA frozen (π0.5 LoRA, embodiment fine-tune) |
| Extraction pass | `AG3S/trajopt/attention_policy.py:54 AttentionPolicy`, `:37-41`; sampler `AG3S/ag3s/experiments/sources/pi05_attention.py:52-56,154-175` | Same inputs → attention {cam: 16×16 raw mass}, from layer 8, head 2, denoise step 0, last action query | Implemented, separate model copy (`return_attn_probs`). On unless `--no-attention` |
| Sub-task probe (HSR) | feature `pi05_attention.py:109 _kv_feature`; `AG3S/ag3s/stages/subtask_probe.py:79 SubtaskProbe`, `:153 SubtaskDebounce`; asset `AG3S/ag3s/asset/subtask_probe/kv_L4_v1.npz` | Layer-4 KV V, head 0, [text mean ‖ image mean] 512-D → p(pick, place, home), 3-frame debounce | Implemented; gate on by default (`config.py:353`). Blocks adopting or switching a target in place/home; never drives attach/detach. Treated as auxiliary in the plan |
| Back-projection, self-filter, fusion | `runtime/multiview.py:407 fuse_observations`, `:313 fuse`; `stages/attention_lifting.py:289 lift` | depth ×3, K, `T_base_cam = FK(q_capture)·T_link_cam`, attention → fused cloud (1 cm voxels, max raw attention across views, percentile 5–99 normalisation) | Implemented |
| Seeds, growing, candidate filtering, confirmation (ATS) | `stages/target_grounding.py:269 extract_seeds`, `:1611 ground_target`, `:1566 _score_cluster`, `:518 TargetConfirm`; `stages/attention_split.py:196`; `stages/admissibility.py:416 assess`, `:239 DestinationRegistry` | Cloud → target cluster: top 5% seeds, DBSCAN eps 3 cm, score 0.7·attention + 0.3·geometry; graspable width ≤ max opening (71.46 mm from the gripper model); not destination; 3-frame hysteresis | Implemented, serving |
| Boundary-preserving extent | `pipeline.py:3151 _target_ball`; `fields/curobo_builder.py:416 _ball_owned_split`; F1 `:469` | Target points + non-target evidence → seeds removed from the fine target-free layer | `ball-owned` implemented: off by code default, **on in the evaluated arms**. F1 surface evidence adopted (T46c) but off in T53 C1a |
| Scene field | `pipeline.py:2571 _build_esdf`; `fields/curobo_builder.py:987`, `:836 place_fine_window`; `fields/curobo_field.py:142 CuroboEsdfField` | Self-filtered depth → cuRobo TSDF → coarse 20 mm + fine 5 mm ESDF (`min` of tiers); fine window around the swept hand | Implemented with `--esdf-backend curobo` (code default `legacy`) |
| Link-specific allowed collision (LTE) | `config.py:854`; `to_adapter.py:227-241`; `linearize.py:927-950`; target-free query `curobo_field.py:356` | Per query sphere: active-hand contact links → target-free layer; others → scene field | Policy `exclude_authorized` in the evaluated arm (code default `relax`). Queries cover **gripper links only** (`--links gripper`) |
| Grasp state machine | `AG3S/trajopt/grasp_latch.py:47-53, 109, 120 LatchConfig, 381 GraspLatch`; driver `safe_policy.py:1038,1119` | Executed gripper commands, measured opening, target identity → searching / latched / closing / held / placed; attach/detach | Implemented (evidence mode) |
| Attached object | `pipeline.py:682 attach`, `:1352 detach`; `robot_models/held_object.py` | Held object → surface-fit sphere (+ cover) on the hand; queries the scene field (20 mm destination margin); target-free layer off while held | Implemented |
| CCR | `AG3S/trajopt/sqp.py:70 TrajectoryOptimizer`, `:198 solve`; `linearize.py:850 _esdf_clearance_core`; `qp.py` (OSQP) | Reference chunk, q_now, scene snapshot → refined chunk. Constraint `d − r − m + s ≥ 0`, joint, velocity and acceleration boxes, trust region | Implemented. Evaluated arm: 16 planned rows (8 executed + 8 look-ahead) |
| Post-check gate | `safe_policy.py:1463 _verdict`, `:1522 _reasons`; `wire.py:366-384` | Verdict reasons → execute (`allowed_contact`, `budget_only`, `occluded_target`, …) or HOLD (`collision`, `uncertified`, …) | Implemented |
| Manipulation phase | client `--safe-phase` (`integration/rby1/bringup/pi05_infer.py:819`), server `safe_policy.py:860` | Injected, **not inferred**; constant `approach` in the evaluated runs | Do not draw a perception-driven phase classifier |

## Assets

All renders are **MuJoCo 3.11 replays of the recorded `qpos`** from the client step records (record step j = control step 8j), re-rendered at 1600x1200 from chosen viewpoints with the CPU renderer. Nothing is re-simulated. Overlays come from the server chunk dumps, or from FK on the recorded chunks. Generator: `paper/tools/plotting/paperbanana_render.py` (renders) and `paperbanana_compose.py` (montage, F3 charts). Each PNG has a `.json` sidecar with source paths, SHA-256 hashes and the values used. The earlier chart-style assets are kept in `assets/diagnostic/`.

| File | Content | Chunk |
|---|---|---|
| `assets/f3_in_{head,left_wrist,right_wrist}.png` | the three policy cameras re-rendered at 960x720 | 10 |
| `assets/f3_att_{head,left_wrist,right_wrist}.png` | the same views + served 16x16 attention | 10 |
| `assets/f3_lifted_attention.png` | fused cloud as spheres coloured by normalised attention (grey to red), top-5% seeds in teal | 17 |
| `assets/f3_target_cluster.png` | selected target cluster (orange) among observed surfaces | 17 |
| `assets/f3_scene_field.png` | recorded cuRobo main ESDF (5 mm): voxels with abs(d) <= 4 mm near the target | 17 |
| `assets/f3_attached_object.png` | attached sphere recorded at attach (chunk 19), carried rigidly with the hand link by FK and shown during carry | 19 to 26 |
| `assets/f3_ccr_paths.png`, `assets/f3_ccr_tcp.png` | FK of site `right_ee` along the recorded reference vs refined chunk (16 planned rows). Chunk 19 = largest executed-row tool-centre shift in this run (25.2 mm) | 19 |
| `assets/f3_timeline.png` | per-chunk grasp-latch state, sub-task probe p(pick/place/home), HOLD chunks [18] | all 75 |
| `reference_montage.png` | pipeline layout draft | — |

The Stage 3 thumbnail reuses `figure2/assets/f2c_ownership_render.png` (F1 evidence applied offline; see Figure 2 evidence).

## Limits

- The dump stores the attached sphere at its attach-time pose in the base frame. The carry render moves it with the hand link (`EE_BODY_R`) by FK of the recorded states, as the server does with its parent link `ee_right`.
- Timing (`summary_json.profile_ms`, T54 budget) is not registered; do not print a latency.
- Draw no wrist or arm link queries and no learned phase classifier.
