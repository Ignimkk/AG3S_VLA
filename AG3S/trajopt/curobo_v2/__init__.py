"""T63 — cuRobo v2 `TrajectoryOptimizer` on our recorded inputs (served RB-Y1 sphere model,
AG3S target-excluded ESDF).

| module | venv | what |
|---|---|---|
| `export_inputs` | `.venv-ag3s` + **baseline snapshot** on PYTHONPATH (run as a script) | served sphere model, TO limits, per-chunk `q_now` / finger paths / baseline FK |
| `spec` · `world` | any (numpy) | robot -> cuRobo config mapping, ESDF -> cuRobo voxel mapping, activation inversion |
| `robot_world` | `.venv-curobo` + CUDA | cuRobo FK in our sphere order, voxel world with per-layer signed distance |
| `costs` | `.venv-curobo` | extra per-step cost terms without editing third_party/curobo (B1-B4) |
| `solver` | `.venv-curobo` + CUDA | B0 (basic cuRobo, fixed dt on our 15 Hz chunk grid) |

Importing this package imports nothing heavy.
"""
