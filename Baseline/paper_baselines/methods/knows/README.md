# KNOWS [17] — `Baseline/paper_baselines/methods/knows`

| item | value |
|---|---|
| paper | *Your Model Already Knows: Attention-Guided Safety Filter for Vision-Language-Action Models*, arXiv:2606.09749v1 (main.pdf [17]) |
| source | **no official code**. This folder is our paper-only reimplementation, moved unchanged from `Baseline/knows_vla` by `git mv` (PAPER-METHODS, 2026-10-23; history preserved) |
| license | project code (no third-party code) |
| interface | `adapter.py` → `KnowsSafetyFilter` (`get_method("knows")`), the common `methods/interface.py` `SafetyFilter` |
| state | CBF-QP (`cbf/`) + tests unchanged and passing after the move (49 passed in `.venv-ag3s`, same test set as before). Adapter runs on recorded SL3 frames (CPU smoke). Closed-loop rollout: next STEP |

## What the adapter does (and what differs from the paper)

`KnowsSafetyFilter.filter(obs, actions)` runs `cbf.filter.SafetyFilter` (paper Eq. 12, `CbfParams()` = paper
defaults; D1-D7 variants via `cbf_params={...}`) on every action of the chunk:

| step | paper | here |
|---|---|---|
| robot geometry | one EE ellipsoid, "calibrated offline" | Franka-hand semi-axes (0.04, 0.04, 0.07) in the EE frame (`offline_filter.DEFAULT_GRIPPER_SEMI_AXES`) |
| obstacles | ellipsoids from "manipulable object" segmentation, MVEE, shapes frozen at t=0, centroids tracked | same rule; segmentation = simulator GT (`obstacle_source="gt_seg"`, privileged) or target-free clouds from upstream (`"given"`); support surfaces (table) never obstacles |
| track identity (§3.2) | one track per object; swaps re-associated by HSV histogram; occluded track frozen at its last position | `associate="centroid"` (default for `detector` / `given`, whose names are re-ranked per call): Hungarian on centroid distance, gate 0.10 m, unmatched track held at its last centroid (CLOSED-LOOP-FIX). `"name"` (default for `gt_seg`): match by instance name, unmatched -> t=0 centroid (previous behaviour) |
| obstacle set / crop (BASELINE-BENCH-READY) | all manipulable objects except the target | `gt_seg`: `gt_obstacle_set="all_non_target"` (default) or `"obstacle_names"` (SafeLIBERO obstacle only, sensitivity variant). `detector`: `crop_mode="depth_trim"` (default; box-centre object only, table / background / robot base removed, clouds associated in 3D) or `"box"` (V5 whole-box crop) — `safety_common.py` docstring |
| target exclusion (Eq. 4) | attention top-1 object | BDDL target + destination (privileged, `exclude="task_objects"`); `"phase_target"` = `offline_filter --target gt` rule |
| units | "scaled to physical units" | `a[:3] x 0.05 m`, `a[3:6] x 0.5 rad` (robosuite OSC `output_max`) and back |
| chunk | per control step | T = 1 per env step reproduces it; T > 1 predicts the EE pose with the measured 0.0115 m / action unit (`safety_common.propagate_ee`) |
| emergency stop | infeasible QP → zero motion | same (reachable only with `slack_weight`, docs/16) |

Known behaviour inherited from the paper's relaxed normals: the eps box can satisfy a row by tilting the separating
plane, so the true barrier can still shrink faster than `(1 - gamma) h` (`docs/16-filter-variants.md`; the adapter
test checks the discrete CBF only for `normals="fixed"`). Eq. (12) has no bound on `delta_c`, so the QP is never
infeasible and `qp_fail` / emergency stops stay 0 with the paper defaults (docs/16 D3).

What the closed-loop `h` is (CLOSED-LOOP-FIX, `info["steps"][k]`): `min_h` = Eq. (6) `h(n)` at the **stored**
separating normal = the QP's own row value before the step's `delta_n`, at the EE pose before row `k` along the
**filtered** chunk (row 0 = measured pose). Because the stored normal lags the optimum, `h(n) <= h* = max_n h(n)`;
`h_star` / `h_star_next` (after the QP step) / `h_star_next_nominal` report the true separation, `cbf_ok` the
docs/16 condition on it, `relax_m` / `need_m` what tilting the normal can supply vs what Eq. (7) demands of an
overlapping pair. `diagnostics=False` skips them (the served chunk is the same either way, tested).

## Run

```bash
source Baseline/paper_baselines/setup/paper_baselines_env.sh      # .venv-baselines has osqp/cvxpy (PAPER-METHODS C)
CUDA_VISIBLE_DEVICES="" .venv-baselines/bin/python -m pytest Baseline/paper_baselines/methods/knows/tests -q
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 .venv-baselines/bin/python -m Baseline.paper_baselines.methods.smoke_safety_filters --arms knows
```

`Baseline.knows_vla` remains importable as a thin alias package (`Baseline/knows_vla/__init__.py`) because
`AG3S/ag3s/stages/geometry.py` (MVEE reuse), `benchmark/__init__.py` and `scripts/setup/check_workspace.py` still
name it and may not be edited here. `doc/migration/validation/*.json` keep the old path as provenance.

---

The rest of this file is the original `knows_vla` README (P0 probes; historical commands use the pre-migration
paths `benchmark.knows_vla` / `src/openpi`).

## knows_vla — KNOWS reverse-engineering probes (original README)

Tooling for the reverse engineering of *Your Model Already Knows: Attention-Guided Safety Filter
for Vision-Language-Action Models* (arXiv:2606.09749v1). Analysis lives in
[papers/KNOWS/](../../papers/KNOWS/); this directory holds only runnable code.

Nothing here implements the paper's safety filter yet. These are the P0 probes that decide whether
implementing it is justified — see [06-repro-plan.md](../../papers/KNOWS/06-repro-plan.md) §8.

```text
knows_vla/
├── probe_p0a.py      P0a: attention-extraction plumbing + shape validation (no scene needed)
├── libero_env.py     LIBERO env with GT instance segmentation + BDDL goal parsing
├── collect_p0b.py    P0b stage 1: policy-driven rollout -> npz (runs in the LIBERO venv)
├── probe_p0b.py      P0b stage 2: layer x head attention sweep vs GT target (openpi venv)
└── libero_config/    project-local LIBERO paths; keeps ~/.libero untouched
```

## Two venvs, on purpose

| venv | contents | used by |
|---|---|---|
| `src/openpi/.venv` | JAX, openpi, pi05_libero | `probe_p0a.py`, `probe_p0b.py`, policy server |
| `src/openpi/examples/libero/.venv` | numpy 1.22.4, robosuite 1.4.1, mujoco 3.2.3, LIBERO | `collect_p0b.py` |

They cannot be merged: LIBERO pins numpy 1.22 and opencv 4.6, which the JAX stack does not accept.
This mirrors upstream openpi, which runs the simulator and the policy in separate processes.

## One-time LIBERO setup

Already done in this workspace; recorded so it can be repeated.

```bash
cd src/openpi
uv venv --python 3.10 examples/libero/.venv
V=examples/libero/.venv/bin/python
uv pip install --python $V "numpy==1.22.4" "robosuite==1.4.1" "bddl==1.0.1" "gym==0.25.2" \
    easydict cloudpickle future "opencv-python==4.6.0.66" "imageio[ffmpeg]" tqdm "hydra-core==1.2.0"
uv pip install --python $V torch --torch-backend=cpu   # libero.benchmark imports torch
uv pip install --python $V "matplotlib==3.5.3"         # pin: newer pulls numpy 2.x and breaks cv2
uv pip install --python $V "mujoco==3.2.3"             # pin: 3.11 renamed MjData.qM -> M
uv pip install --python $V -e packages/openpi-client
```

Gotchas that cost time:

- **matplotlib must be pinned.** Installing it unpinned upgrades numpy to 2.x, after which
  `cv2` fails with `numpy.core.multiarray failed to import`.
- **mujoco must be 3.2.3** (what openpi's own `examples/libero/requirements.txt` pins). robosuite
  1.4.1 calls `mujoco.mj_fullM(..., data.qM)`; mujoco 3.11 renamed that field to `M`.
- **`~/.libero/config.yaml` is global and shared.** `openvla_ws` also uses LIBERO on this machine,
  so we never write it. `libero_config/` here is pointed at via `LIBERO_CONFIG_PATH` instead.
- LIBERO's init-state files are pickled torch tensors; torch >= 2.6 refuses them under
  `weights_only=True`. `libero_env.py` patches `torch.load` for this process only.

## Running

### P0a — plumbing (a few minutes, CPU)

```bash
JAX_PLATFORMS=cpu src/openpi/.venv/bin/python -m benchmark.knows_vla.probe_p0a
```

Validates the tensor layout predicted in [02-architecture.md](../../papers/KNOWS/02-architecture.md)
§3.4 and that `return_attn_probs` leaves the model byte-identical. Results:
[07-p0a-results.md](../../papers/KNOWS/07-p0a-results.md).

### Where each piece runs

Inference goes on the GPU server, the simulator stays local — the split this project already uses.
`collect_p0b.py` is a websocket client, so it only needs `--host`.

| step | host | needs MuJoCo | cost |
|---|---|---|---|
| `serve_policy.py` | **GPU server** | no | GPU-bound |
| `collect_p0b.py` | **local** | yes | light |
| `probe_p0b.py` | **GPU server** | **no** | the expensive one |

Stage 2 touches no simulator at all: it reads the recorded npz and runs policy forwards. That is
also where nearly all the time goes (~8 s per rollout step on CPU), so it belongs on the server.

Nothing here opens a window — `OffScreenRenderEnv` renders headlessly through EGL — so collection
*can* also run server-side if EGL is available there, but the split above needs no such check.

### P0b stage 1 — collect rollouts (local)

Server:

```bash
cd src/openpi && .venv/bin/python scripts/serve_policy.py --env LIBERO
```

Local (LIBERO venv):

```bash
LIBERO_CONFIG_PATH=$PWD/benchmark/knows_vla/libero_config \
PYTHONPATH=$PWD/src/openpi/third_party/libero:$PWD MUJOCO_GL=egl \
src/openpi/examples/libero/.venv/bin/python -m benchmark.knows_vla.collect_p0b \
    --host <server> --port 8000 --task-ids 0 1 2 3 --max-steps 220
```

Writes `data/knows_p0b/<suite>_task<i>_ep<j>.npz` with the 224x224 images the policy actually saw,
the 256x256 GT instance segmentation, depth, state, actions, and the BDDL-derived target labels.

`depth_full` is ~256 KB/step and **P0b does not read it** — it is there for the later ellipsoid
fitting. Drop it before copying episodes to the server if transfer size matters.

### P0b stage 2 — score the attention signal (server)

```bash
src/openpi/.venv/bin/python -m benchmark.knows_vla.probe_p0b \
    --episodes data/knows_p0b/*.npz --window-k 5
```

If running both on one box, **stop the policy server first** — two ~6 GB models plus XLA
compilation will not fit in 62 GB and the probe dies with
`LLVM compilation error: Cannot allocate memory`.

Results: [08-p0b-results.md](../../papers/KNOWS/08-p0b-results.md).

## The openpi change these depend on

`return_attn_probs` (default off) on `Pi0Config`, threaded to `gemma.Module`. 36 lines across three
files; SEAM's byte-for-byte baseline parity tests still pass. Rationale and the reason the flag has
to be a static dataclass field rather than a call argument:
[07-p0a-results.md](../../papers/KNOWS/07-p0a-results.md) §2.
