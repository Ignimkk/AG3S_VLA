# PaperBanana package for F1–F3

Prepared 2026-10-13 KST on `feature/ag3s-development` @ `cf38c9db`. Existing code and paper files were not modified.

## What is here

| Folder | Role | Assets (replay renders of the recorded run) |
|---|---|---|
| `figure1/` | Demo teaser (Abstract) | 4 keyframes, wrist RGB + attention, 3D target grounding in the scene, contact-link close-up |
| `figure2/` | Conceptual overview (Introduction) | (a) all surfaces, (b) ball, (c) ownership split, as 1 cm voxels in one frame; (d) link-specific query spheres |
| `figure3/` | Full method pipeline (Method) | camera inputs + attention, lifted attention, target cluster, scene ESDF voxels, attached object, tool-centre path chart, episode timeline |

Each folder holds:
- `brief_en.txt`: text for PaperBanana's method box, under its 4,000-character limit. It describes the montage layout.
- `caption.md`: a CVPR-style caption (draft). An HTML comment states the scope.
- `evidence.md`: code paths and functions, implementation status, verify.json keys, assets and limits.
- `assets/*.png` + `.json`: one sidecar per asset, with source hashes and the values used. The older chart-style assets are kept in `assets/diagnostic/`.
- `reference_montage.png`: a polished layout draft of the whole figure. It is the single reference image PaperBanana accepts.

**Source:** every image comes from **one recorded AG3S closed-loop run**, the latest at the time of writing. It is T53, the current serving arm C1a, on 2026-10-13, episode 1842 ("Put the pear in the basket.", with an orange touching the pear).

**How the images were made:** the robot and object states (`qpos`) recorded at each chunk were **re-rendered with MuJoCo** at 1600x1200 from chosen viewpoints. This replays the recorded states; nothing was re-simulated. On top of the renders are server-dump overlays (target cluster, fused cloud and attention, query spheres, ball, ESDF voxels) and FK of the recorded reference and refined chunks.

## How to use with PaperBanana

1. Open the AI scientific illustration page. Choose the model (GPT Image 2 is the page's recommended option) and the "text to SVG" mode.
2. Paste `figureN/brief_en.txt` into the method text box.
3. If the account allows a reference image (a paid option), upload `figureN/reference_montage.png`. It guides layout and content, especially the F1 demo strip.
4. Export SVG. Replace any generated photo-like region with the real renders from `assets/` in Inkscape or Illustrator: F1 keyframes and close-ups, the F2 voxel panels, and the F3 thumbnails. Do not keep a generated image of the robot or the scene.
5. Before submission, check the result against `evidence.md`. A generated figure must not add modules, objects, numbers or comparisons that the evidence does not support.

## Shared colour code

| Meaning | Colour |
|---|---|
| target object | orange `#E8711A` |
| neighbour / obstacle | blue `#2E75B6` |
| contact links (active hand) and our contributions | gold `#C99A06` (fills `#FFF2CC`) |
| other links | grey `#7F7F7F` |
| attached object | purple `#7030A0` |
| hold / failure | red `#C00000` |
| frozen VLA | pale navy `#DEEBF7` |
| distance fields | pale blue `#DAE8FC` |

## Regenerate

```bash
MUJOCO_GL=osmesa .venv-ag3s/bin/python paper/tools/plotting/paperbanana_render.py    # renders + sidecars
MUJOCO_GL=osmesa .venv-ag3s/bin/python paper/tools/plotting/paperbanana_compose.py   # F3 charts + montages
.venv-ag3s/bin/python paper/tools/plotting/paperbanana_assets.py                      # optional: diagnostic charts
```

The renderer is CPU-only (OSMesa) and does not use the GPU. The scripts read only from storage and the checkout, and write only into this folder. The diagnostic script writes into `assets/`, so move its output to `assets/diagnostic/` if you rerun it.

## Status summary (details in each evidence.md)

- **Implemented, serving default:**
  - attention extraction (L8H2, separate pass)
  - multi-view fusion and ATS grounding
  - sub-task probe gate
  - grasp state machine and attached object
  - CCR with post-check gate
- **Implemented, on in the evaluated arms but off by code default:**
  - `ball-owned` boundary-preserving exclusion
  - `exclude-authorized` link-specific queries
  - cuRobo ESDF backend
- **Not established:** the closed-loop benefit of boundary preservation (CL3 task success 14 / 13 / 13 of 18) and of link-specific vs global exclusion (degenerate in the gripper-only scope, T46d).
- **Planned:** wrist-link queries (T46f), SafeLIBERO main evaluation.
- **Result status:** registration and verification state are recorded separately and do not determine manuscript use. No figure prints a number.
