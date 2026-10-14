# SL5b → feature/ag3s-development: grasp latch `reach` against an occluded target's last observation

Date: 2026-10-14. From `feature/panda-benchmark` (checkout `/mnt/dev/AG3S_VLA-benchmark`). The user decided on
2026-10-14 that this goes to the development branch. The benchmark branch does **not** change the latch, and the
reach limit stays at 120 mm.

## What happens

V11AB grasp latch (`AG3S/trajopt/grasp_latch.py`, `_grasp_checks`) attaches only if every check passes. One
check is `reach`: the distance from the parent link origin to the manipulated object's centroid must be
≤ `LatchConfig.reach` (120 mm).

On Panda / LIBERO the policy lifts the object right after closing. Once lifted, the object is no longer
segmented (`ag3s=no_target`, grounding `no_cluster`), and its state becomes `occluded`. While occluded, the
latch keeps the **last observed geometry**, i.e. the centroid of the object as it sat on the table. As the hand
rises, the distance from the finger origin to that stale centroid grows past 120 mm. Every other check passes
(`settled`, `closed`, `dropped`, `not_empty`, `blocked`, `manipulated`), yet attach is refused. Until attach, the
held object stays in the ESDF as an obstacle, and the verdict gives collision HOLDs
(`panda_hand ↔ esdf`, `answer_excludes_target`).

## Evidence (libero_spatial task 0, init 0, policy_seed 700000)

| run (under `AG3S_VLA-storage/results/benchmark/`) | code | conversion | result | latch |
|---|---|---|---|---|
| `20261014T143323_945b608c6a0c_sl5b-smoke4-t0e0-vla_v11ab-A` | `945b608c` | A | fail (220 steps, 25 HOLD) | never attached: reach 177.8–178.0 mm from seq 27 on (HOLD deadlock) |
| `20261014T143614_945b608c6a0c_sl5b-smoke4-t0e0-vla_v11ab-B` | `945b608c` | B | fail (220 steps) | reach failed for seq 19–26 (127.9–184.2 mm, occluded since seq 19, geometry frozen at pipeline frame 17); attached at seq 27 (111.7 mm) |
| `20261014T163754_a06b2ac23d09_sl5b-smoke5-t0e0-vla_v11ab-A` | `a06b2ac2` (feedback fix) | A | fail (220 steps, 12 HOLD) | never attached: occluded from obs 80, reach up to 357.1 mm; opening 0.043–0.059 also fails `blocked` (0.05) on some calls |
| `20261014T163552_a06b2ac23d09_sl5b-smoke5-t0e0-vla_v11ab-B` | `a06b2ac2` | B | **success** (114 steps) | attached at seq 18 (obs 100, reach 103.5 mm) — in the same call the bowl became occluded; afterwards reach 133.9–149.8 mm while held |

- `smoke4` used an `exec_feedback.measured_gripper` that was one chunk stale. That was a benchmark-adapter bug,
  fixed in `a06b2ac2`. With the current capture, `settled` passed one call earlier, and attach happened just
  before the reach grew. **The underlying dependence on the stale centroid is unchanged**: whether attach
  happens depends on whether `settled` passes before the lift.
- Per call, the logs hold `grasp.evidence` (`reach_mm`, `manipulated_state`, every check) and `grasp.alignment`
  (`pipeline_frame`, `geometry_frame`, `geometry_age_frames`) in `episodes/libero_spatial_t0_e0_calls.jsonl`.
  Latch transitions are in `logs/server_console.log`.
- Figures:
  - `AG3S/ag3s/docs/figures/sl5b-smoke4-t0e0-ab.png` (before the feedback fix);
  - `AG3S/ag3s/docs/figures/sl5b-smoke5-t0e0-ab-feedback.png` (after the fix);
  - each with a `.json` sidecar holding `reach_mm_per_call`: (obs, reach, HOLD, manipulated_state).
- Numbers: `AG3S/ag3s/docs/handoff/SL5b-smoke.verify.json`, `SL5b-feedback.verify.json`.

## Reproduce (benchmark checkout)

```bash
cd /mnt/dev/AG3S_VLA-benchmark && git checkout a06b2ac2   # or later on feature/panda-benchmark
source scripts/setup/env.sh
export OPENPI_DATA_HOME=$PWD/.cache/openpi XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer=
.venv-ag3s/bin/python benchmark/runners/launch_libero_paired.py --task-ids 0 --episode-ids 0 \
    --run-id <id> --modes vla_v11ab --conversion A --step SL5b-smoke     # takes the shared GPU lock itself
```

## Not decided here

- How the latch should treat an occluded manipulated object (for example the held pose, or the hand-relative
  centroid at the last visible frame). That is a method change on the development branch.
- `blocked_gap` 0.05 vs Panda's grasp openings (0.043–0.059 here). It is kept at the RB-Y1 value by the user's
  2026-10-14 decision.
