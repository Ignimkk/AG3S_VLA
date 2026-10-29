"""T63 — cuRobo v2 `TrajectoryOptimizer` vs our TO (CCR) on fixed recorded inputs.

Steps implemented here: **0 (parity gate)**, **1 (B0 = basic cuRobo)** and **2** — the composed
B0 world of user decision 2 (`--stage parity_composed`, `composed_world.py`) and the cumulative
variants B0-B4 (`--stage smoke`; terms in `curobo_v2.costs`, weights from the baseline server logs,
`baseline_weights`). Step 3 (`--stage step3`, user decision 3): B0 and B1-B4 x lambda
(`--lambda 1,100,10000`; lambda multiplies our three terms only, `scaled_weights`), trajectories
only — every candidate is measured by the common evaluator `curobo_v2/step3_eval.py` (snapshot side).
From step 3 the B0 world of a chunk without `esdf_free` is our main field `min(coarse, fine)` inside
the fine lattice (`world.compose_b0`; `--stage parity_composed --only-no-free` re-checks it) and
`--b4-mask` defaults to the record's mask.

Inputs: `curobo_v2/export_inputs.py` output (baseline snapshot side, `.venv-ag3s`) — robot spec,
per-episode `q_now`, finger paths, baseline FK centres, chunk file paths.

    source scripts/setup/env.sh
    cd $AG3S_PROJECT_ROOT
    ( flock -w 600 9 || exit 13
      MUJOCO_GL=osmesa PYTHONPATH=$AG3S_PROJECT_ROOT .venv-curobo/bin/python \\
        -m AG3S.trajopt.experiments.t63_curobo_compare \\
        --inputs $AG3S_RUN_DIR/impl/T63/inputs --out $AG3S_RUN_DIR/impl/T63 \\
        --stage parity --episodes 211 126 ) 9>/tmp/ag3s-vla-gpu-validation.lock

## Parity (step 0) — what is compared, and the gate fixed before measuring

| quantity | cuRobo side | reference | tolerance |
|---|---|---|---|
| sphere centre | cuRobo FK (`link_torso_5` frame, fingers shifted per step) -> base frame with the baseline `T_base_torso5` | (a) baseline FK of the recorded refined chunk, all plan-window steps; (b) record `sphere_centres` (step 0) | 1 mm (the `fixed_input_ablation` fidelity tolerance) |
| signed distance, main field | per layer from cuRobo's own collision kernel (activation inverted), composed with our two-tier rule (coarse, and min(coarse, fine) inside the fine lattice), minus radius | record `clearance` (16 x 270) = recorder's `distance - r - esdf_margin` on the main field | `voxel_size / 2` of the layer that answered (fine 2.5 mm, coarse 10 mm) |
| signed distance, target-excluded layer (`esdf_free`) | cuRobo kernel | our `EsdfField` on the same array (the field code is byte-identical to the snapshot) | 2.5 mm |

Rows are split into `interior` (query inside the answering lattice by more than one voxel) and
`border` (within one voxel of the lattice edge or outside it): at the border cuRobo weights only
in-bounds corners while our field clamps to the edge, a known structural difference. **Gate (fixed
before the first run): PASS iff centre max error <= 1 mm (both references) and no interior row of
the main field exceeds its tolerance.** Border rows and the free layer are reported, not gated.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
import time
from typing import Any, Optional

import numpy as np

TOOL = "AG3S.trajopt.experiments.t63_curobo_compare"
CENTRE_TOL_MM = 1.0


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None if np.isnan(obj) else ("inf" if obj > 0 else "-inf")
    return obj


def dist_stats(x_mm: np.ndarray) -> dict[str, Any]:
    x = np.abs(np.asarray(x_mm, np.float64).reshape(-1))
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "median_mm": float(np.median(x)),
            "p95_mm": float(np.percentile(x, 95)), "max_mm": float(x.max())}


def lattice_margin_vox(points: np.ndarray, layer: dict[str, Any]) -> np.ndarray:
    """Signed distance (in voxels) of each point to the lattice box first..last voxel centre;
    negative = outside."""
    o = np.asarray(layer["origin"], np.float64)
    last = o + (np.asarray(layer["distance"].shape) - 1.0) * layer["voxel_size"]
    gap = np.minimum(points - o, last - points).min(axis=1)
    return gap / float(layer["voxel_size"])


# =================================================================================================
# loading
# =================================================================================================

def load_inputs(inputs: pathlib.Path, episodes: Optional[list[int]]):
    from AG3S.trajopt.curobo_v2 import spec as S

    spec = S.load_spec(inputs / "robot_spec.json")
    manifest = json.loads((inputs / "export_manifest.json").read_text())
    eps = {int(e["episode"]): e for e in manifest["episodes"]}
    chosen = episodes or sorted(eps)
    missing = [e for e in chosen if e not in eps]
    if missing:
        raise SystemExit(f"episodes {missing} were not exported")
    data = {e: dict(np.load(eps[e]["file"], allow_pickle=False)) for e in chosen}
    return spec, manifest, data


def our_field_layers(z, outside: float):
    """Our evaluator on the record arrays (`fixed_input_ablation.build_field`)."""
    from AG3S.ag3s.fields.curobo_field import CuroboEsdfField, layer_from_arrays

    def layer(prefix):
        return layer_from_arrays(z[f"{prefix}_distance"], z[f"{prefix}_origin"],
                                 float(z[f"{prefix}_voxel_size"]))

    free = layer("esdf_free") if "esdf_free_distance" in z else None
    main = CuroboEsdfField((layer("esdf_coarse"), layer("esdf")), outside_distance=outside,
                           target_free_layers=(free,) if free is not None else ())
    free_only = (CuroboEsdfField((free,), outside_distance=outside) if free is not None
                 else None)
    return main, free_only


# =================================================================================================
# step 0 — parity
# =================================================================================================

def parity_chunk(robot, vworld, spec, ep: dict[str, np.ndarray], i: int,
                 outside: float, esdf_margin: float) -> dict[str, Any]:
    from AG3S.trajopt.curobo_v2 import spec as S
    from AG3S.trajopt.curobo_v2 import world as W

    path = str(ep["chunk_file"][i])
    with np.load(path, allow_pickle=False) as zf:
        z = {k: zf[k] for k in zf.files if k.startswith("esdf") or k in (
            "refined_chunk", "sphere_centres", "clearance", "sphere_radii")}
    T = np.asarray(ep["T_base_torso5"][i], np.float64)
    P = np.asarray(ep["finger_path_plan"][i], np.float64)
    H = P.shape[1]
    q = S.chunk_to_cspace(z["refined_chunk"], spec)[:H]

    # FK, grouped by identical finger values (in place sphere update between groups)
    sph_t5 = np.empty((H, spec["n_spheres"], 4))
    cols = {tuple(np.round(P[:, k], 12)): [] for k in range(H)}
    for k in range(H):
        cols[tuple(np.round(P[:, k], 12))].append(k)
    for key, steps in cols.items():
        robot.set_fingers(S.finger_values(spec, P, steps[0]))
        sph_t5[steps] = robot.spheres(q[steps])
    centres = W.transform_points(T, sph_t5[..., :3].reshape(-1, 3)).reshape(H, -1, 3)
    radii = sph_t5[0, :, 3]
    base = np.asarray(ep["centres_refined_plan"][i], np.float64)
    out: dict[str, Any] = {
        "file": path, "chunk_index": int(ep["chunk_index"][i]), "seq": int(ep["seq"][i]),
        "n_finger_groups": len(cols),
        "radius_max_abs_diff_m": float(np.abs(radii - z["sphere_radii"]).max()),
        "centre_vs_baseline_fk_max_mm": float(np.abs(centres - base).max() * 1e3),
        "centre_vs_record_step0_max_mm": float(np.abs(centres[0] - z["sphere_centres"]).max()
                                               * 1e3),
    }

    # signed distance per layer through cuRobo's kernel
    vworld.load(z, T)
    flat_t5 = sph_t5.reshape(-1, 4)
    has_free = "esdf_free_distance" in z
    names = ("coarse", "main", "free") if has_free else ("coarse", "main")
    sd = {name: vworld.signed_distance(flat_t5, name).reshape(H, -1) for name in names}
    pts = centres.reshape(-1, 3)
    lay = {name: W.layer_arrays(z, name) for name in names}
    m_fine = lattice_margin_vox(pts, lay["main"]).reshape(H, -1)
    m_coarse = lattice_margin_vox(pts, lay["coarse"]).reshape(H, -1)
    inside_fine = m_fine >= 0.0
    d_coarse = np.where(m_coarse >= 0.0, sd["coarse"], outside)
    d_main = W.compose_main(d_coarse, sd["main"], inside_fine)
    fine_wins = inside_fine & (sd["main"] < d_coarse)
    clearance_cu = d_main - radii[None, :] - esdf_margin
    rec = np.asarray(z["clearance"], np.float64)[:H]
    diff_mm = (clearance_cu - rec) * 1e3
    tol_mm = np.where(fine_wins, lay["main"]["voxel_size"], lay["coarse"]["voxel_size"]) * 500.0
    margin = np.where(fine_wins, m_fine, m_coarse)
    interior = margin > 1.0
    over = np.abs(diff_mm) > tol_mm

    # our evaluator on the same centres (shows the record is reproducible on the reference side)
    main_ours, free_ours = our_field_layers(z, outside)
    d_ours = np.asarray(main_ours._evaluate(pts)[0], np.float64).reshape(H, -1)
    ours_mm = (d_ours - radii[None, :] - esdf_margin - rec) * 1e3
    if has_free:
        m_free = lattice_margin_vox(pts, lay["free"]).reshape(H, -1)
        free_in = m_free > 1.0
        free_diff_mm = (sd["free"] - np.asarray(free_ours._evaluate(pts)[0]).reshape(H, -1)) * 1e3
    else:
        m_free = np.full((H, rec.shape[1]), -1.0)
        free_in = np.zeros_like(m_free, bool)
        free_diff_mm = np.zeros_like(m_free)

    out.update({
        "rows": int(rec.size),
        "rows_fine_answer": int(fine_wins.sum()),
        "rows_interior": int(interior.sum()),
        "main_abs_diff_mm_all": dist_stats(diff_mm),
        "main_abs_diff_mm_interior": dist_stats(diff_mm[interior]),
        "main_abs_diff_mm_border": dist_stats(diff_mm[~interior]),
        "main_over_tol_interior": int((over & interior).sum()),
        "main_over_tol_border": int((over & ~interior).sum()),
        "main_over_tol_near_50mm": int((over & (np.abs(rec) < 0.05)).sum()),
        "rows_near_50mm": int((np.abs(rec) < 0.05).sum()),
        "ours_vs_record_abs_diff_mm": dist_stats(ours_mm),
        "has_free_layer": bool(has_free),
        "free_abs_diff_mm_interior": dist_stats(free_diff_mm[free_in]),
        "free_over_2p5mm_interior": int((np.abs(free_diff_mm[free_in]) > 2.5).sum()),
        "free_rows_interior": int(free_in.sum()),
        "free_rows_outside_lattice": int((m_free < 0.0).sum()),
        "min_record_clearance_mm": float(rec.min() * 1e3),
        "min_curobo_clearance_mm": float(clearance_cu.min() * 1e3),
        "_diff_mm": diff_mm, "_tol_mm": tol_mm, "_interior": interior,
        "_free_diff_mm": free_diff_mm[free_in],
    })
    return out


def run_parity(args, spec, data) -> dict[str, Any]:
    from AG3S.trajopt.curobo_v2.robot_world import RobotModel, VoxelWorld

    robot = RobotModel(spec)
    vworld = VoxelWorld()
    outside = float(spec["field"]["max_distance"]) if spec["field"]["unknown_policy"] == "free" \
        else -float(spec["field"]["max_distance"])
    margin = float(spec["field"]["esdf_margin"] or 0.0)
    per_chunk, diffs, tols, interiors, free_d = [], [], [], [], []
    for e, ep in data.items():
        n = len(ep["chunk_file"])
        idx = range(n) if args.chunks == "all" else [int(c) for c in args.chunks.split(",")]
        for i in idx:
            r = parity_chunk(robot, vworld, spec, ep, i, outside, margin)
            diffs.append(r.pop("_diff_mm").reshape(-1))
            tols.append(r.pop("_tol_mm").reshape(-1))
            interiors.append(r.pop("_interior").reshape(-1))
            free_d.append(r.pop("_free_diff_mm").reshape(-1))
            r["episode"] = e
            per_chunk.append(r)
        print(f"[t63-parity] ep{e}: {len(idx)} chunks", flush=True)
    d = np.concatenate(diffs)
    t = np.concatenate(tols)
    it = np.concatenate(interiors)
    over = np.abs(d) > t
    centre_b = max(r["centre_vs_baseline_fk_max_mm"] for r in per_chunk)
    centre_r = max(r["centre_vs_record_step0_max_mm"] for r in per_chunk)
    summary = {
        "n_chunks": len(per_chunk), "n_rows": int(d.size),
        "centre_vs_baseline_fk_max_mm": centre_b, "centre_vs_record_step0_max_mm": centre_r,
        "radius_max_abs_diff_m": max(r["radius_max_abs_diff_m"] for r in per_chunk),
        "main_abs_diff_mm_all": dist_stats(d), "main_abs_diff_mm_interior": dist_stats(d[it]),
        "main_abs_diff_mm_border": dist_stats(d[~it]),
        "main_fraction_over_tol_all": float(over.mean()),
        "main_fraction_over_tol_interior": float(over[it].mean()) if it.any() else None,
        "main_n_over_tol_interior": int((over & it).sum()),
        "main_n_over_tol_border": int((over & ~it).sum()),
        "free_abs_diff_mm_interior": dist_stats(np.concatenate(free_d)),
        "free_n_over_2p5mm_interior": int(sum(r["free_over_2p5mm_interior"] for r in per_chunk)),
        "ours_vs_record_max_mm": max(r["ours_vs_record_abs_diff_mm"].get("max_mm", 0.0)
                                     for r in per_chunk),
        "free_rows_outside_lattice": int(sum(r["free_rows_outside_lattice"] for r in per_chunk
                                             if r["has_free_layer"])),
        "free_rows_total": int(sum(r["rows"] for r in per_chunk if r["has_free_layer"])),
        "n_chunks_without_free_layer": int(sum(not r["has_free_layer"] for r in per_chunk)),
        "tolerance": "main: voxel_size/2 of the answering layer (fine 2.5 mm, coarse 10 mm); "
                     f"centres: {CENTRE_TOL_MM} mm; free layer (reported only): 2.5 mm",
    }
    summary["gate_pass"] = bool(centre_b <= CENTRE_TOL_MM and centre_r <= CENTRE_TOL_MM
                                and summary["main_n_over_tol_interior"] == 0)
    summary["gate_rule"] = ("PASS iff centre max error <= 1 mm (baseline FK and record) and no "
                            "interior main-field row exceeds its tolerance (fixed before the first "
                            "run; border rows and the free layer reported, not gated)")
    return {"summary": summary, "chunks": per_chunk}


# =================================================================================================
# step 1 — B0
# =================================================================================================

def run_b0(args, spec, data) -> dict[str, Any]:
    from AG3S.trajopt.curobo_v2 import spec as S
    from AG3S.trajopt.curobo_v2 import world as W
    from AG3S.trajopt.curobo_v2.robot_world import RobotModel, VoxelWorld
    from AG3S.trajopt.curobo_v2.solver import CuroboChunkSolver

    t0 = time.perf_counter()
    solver = CuroboChunkSolver(spec, num_seeds=args.num_seeds, timing=args.timing)
    init_s = time.perf_counter() - t0
    robot = RobotModel(spec)
    vworld = VoxelWorld()
    outside = float(spec["field"]["max_distance"])
    rows = []
    for e, ep in data.items():
        n = len(ep["chunk_file"])
        idx = range(n) if args.b0_chunks == "all" else [int(c) for c in args.b0_chunks.split(",")]
        for i in idx:
            path = str(ep["chunk_file"][i])
            with np.load(path, allow_pickle=False) as zf:
                z = {k: zf[k] for k in zf.files if k.startswith("esdf") or k in (
                    "reference_chunk", "refined_chunk", "sphere_radii")}
            T = np.asarray(ep["T_base_torso5"][i], np.float64)
            fingers = S.finger_values(spec, ep["finger_path_full"][i], 0)
            q_now = S.q_now_cspace(ep["q_now"][i], spec)
            # B0 world: the target-excluded layer; a chunk recorded without one (no target to
            # exclude) gets the main fine layer — what our TO's field answers there
            world_layer = "free" if "esdf_free_distance" in z else "main"
            solver.set_world(z, T, world_layer)
            solver.set_fingers(fingers)
            res = solver.solve(q_now, z["reference_chunk"])
            row = {"episode": e, "chunk_index": int(ep["chunk_index"][i]),
                   "seq": int(ep["seq"][i]), "file": path, "world_layer": world_layer,
                   **{k: v for k, v in res.items() if k not in ("chunk", "cspace",
                                                                  "interpolated_start")}}
            if res["cspace"] is not None:
                # min clearance of the B0 command: (a) B0's own world (free layer, cuRobo kernel);
                # (b) our main composite field (no contact policy) — smoke numbers, not step 3
                robot.set_fingers(fingers)
                sph = robot.spheres(res["cspace"])
                vworld.load(z, T)
                sd_free = vworld.signed_distance(sph.reshape(-1, 4), world_layer).reshape(
                    sph.shape[:2])
                pts = W.transform_points(T, sph[..., :3].reshape(-1, 3))
                main_ours, _ = our_field_layers(z, outside)
                d_main = np.asarray(main_ours._evaluate(pts)[0]).reshape(sph.shape[:2])
                r = sph[0, :, 3]
                ref_q = S.chunk_to_cspace(z["reference_chunk"], spec)
                row.update({
                    "min_clearance_b0_world_mm_exec8": float((sd_free[:8] - r).min() * 1e3),
                    "min_clearance_b0_world_mm_all50": float((sd_free - r).min() * 1e3),
                    "min_clearance_main_field_mm_exec8": float((d_main[:8] - r).min() * 1e3),
                    "min_clearance_main_field_mm_all50": float((d_main - r).min() * 1e3),
                    "max_joint_dev_vs_reference_rad": float(np.abs(res["cspace"] - ref_q).max()),
                })
                if args.save_trajectories:
                    out_dir = pathlib.Path(args.out) / "b0_trajectories"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(out_dir / f"ep{e}_chunk{int(ep['chunk_index'][i]):05d}.npz",
                                        b0_chunk=res["chunk"], reference_chunk=z["reference_chunk"],
                                        refined_chunk=z["refined_chunk"], q_now=ep["q_now"][i])
            rows.append(row)
            print(f"[t63-b0] ep{e} chunk {row['chunk_index']}: success={row['success']} "
                  f"wall {row['wall_ms']:.1f} ms solve {row['solve_time_ms']:.1f} ms "
                  f"min clr b0world/main(8) "
                  f"{row.get('min_clearance_b0_world_mm_exec8', float('nan')):.1f}/"
                  f"{row.get('min_clearance_main_field_mm_exec8', float('nan')):.1f} mm", flush=True)
    warm = [r["wall_ms"] for r in rows if not r["cold"]]
    return {"settings": solver.settings_record(), "init_s": init_s,
            "summary": {"n": len(rows), "n_success": int(sum(r["success"] for r in rows)),
                        "n_nonfinite": int(sum(not r["finite"] for r in rows)),
                        "cold_wall_ms": [r["wall_ms"] for r in rows if r["cold"]],
                        "warm_wall_ms_median": float(np.median(warm)) if warm else None,
                        "warm_wall_ms_max": float(np.max(warm)) if warm else None,
                        "warm_over_66p7ms": int(sum(w > 1000.0 / 15.0 for w in warm))},
            "chunks": rows}


# =================================================================================================
# step 2 — composed worlds (user decision 2), B1-B4 smoke
# =================================================================================================

#: B-variants: (world mode, extra terms). Cumulative, as the task table.
VARIANTS = {
    "B0": ("b0", ()),
    # T63b user decision 2: B0 + our TO's obstacle margin on the non-target rows (`b0m.py`)
    "B0m": ("b0m", ()),
    "B1": ("b0", ("track",)),
    "B2": ("b0", ("track", "continuity")),
    "B3": ("b0", ("track", "continuity", "deflection_rate", "servo_accel")),
    "B4": ("lte", ("track", "continuity", "deflection_rate", "servo_accel")),
}
FINGER_LINKS = {"left": ("ee_finger_l1", "ee_finger_l2"), "right": ("ee_finger_r1", "ee_finger_r2")}


def baseline_weights(manifest: dict[str, Any]) -> dict[str, Any]:
    """The TO weights of the baseline C1aN servers, **from their start logs** (not tuned).

    | weight | source |
    |---|---|
    | w_track, w_smooth, w_continuity, w_slack | `serve_safe.log` `TO objective:` line |
    | w_deflection_rate | `serve_safe.log` `TO deflection rate (T43 TA B): w_deflection_rate=` line |
    | track_decay | no serve_safe flag exists -> `CostConfig.track_decay` default 1.0 (snapshot `AG3S/trajopt/config.py:234`) |
    | servo_accel | no `--servo-accel` in the server argv -> `LimitsConfig.servo_accel` default `off` (snapshot `config.py:398`); weight default 0.01 (`config.py:405`) |
    """
    import re

    found: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for sd in sorted({e["server_dir"] for e in manifest["episodes"]}):
        log = pathlib.Path(sd) / "serve_safe.log"
        for n, line in enumerate(log.read_text(errors="replace").splitlines(), start=1):
            m = re.search(r"TO objective: w_track=(\S+) w_smooth=(\S+) w_continuity=(\S+) "
                          r"w_slack=(\S+)", line)
            if m:
                vals = dict(zip(("w_track", "w_smooth", "w_continuity", "w_slack"),
                                map(float, m.groups())))
                for k, v in vals.items():
                    if k in found and found[k] != v:
                        raise SystemExit(f"servers disagree on {k}: {found[k]} vs {v}")
                    found[k] = v
                    sources[k] = f"{log}:{n}"
            m = re.search(r"w_deflection_rate=(\S+)", line)
            if m and "TO deflection rate" in line:
                v = float(m.group(1))
                if "w_deflection_rate" in found and found["w_deflection_rate"] != v:
                    raise SystemExit("servers disagree on w_deflection_rate")
                found["w_deflection_rate"] = v
                sources["w_deflection_rate"] = f"{log}:{n}"
    argvs = [a for a in manifest["server_argv"].values()]
    if any("--servo-accel" in a for a in argvs) or any("--track-decay" in a for a in argvs):
        raise SystemExit("server argv sets servo accel / track decay — read them, do not default")
    found.setdefault("w_deflection_rate", 0.0)
    found["track_decay"] = 1.0
    sources["track_decay"] = "snapshot AG3S/trajopt/config.py:234 default (no serve_safe flag)"
    found["servo_accel"] = "off"
    found["servo_accel_weight"] = 0.01
    sources["servo_accel"] = ("snapshot AG3S/trajopt/config.py:398 default 'off' (no --servo-accel "
                              "in the server argv); weight config.py:405")
    missing = {"w_track", "w_smooth", "w_continuity", "w_slack"} - set(found)
    if missing:
        raise SystemExit(f"weights {missing} not found in the server logs")
    return {"weights": found, "sources": sources}


def build_terms(names, weights: dict[str, Any], rows: int, dev) -> dict[str, Any]:
    from AG3S.trajopt.curobo_v2 import costs as C

    kw = dict(rows=rows, device=dev.device, dtype=dev.dtype)
    w = weights
    out: dict[str, Any] = {}
    for name in names:
        if name == "track":
            out[name] = C.TrackTerm(w["w_track"], w["track_decay"], **kw)
        elif name == "continuity":
            out[name] = C.ContinuityTerm(w["w_continuity"], **kw)
        elif name == "deflection_rate":
            out[name] = C.DeflectionRateTerm(w["w_deflection_rate"], **kw)
        elif name == "servo_accel":
            if w["servo_accel"] != "off":
                raise SystemExit("servo accel is on in the baseline — the per-joint bound a_C "
                                 "(sqp._build_servo_accel) must be exported first")
            # inactive under the baseline (`servo_accel = off`): implemented and unit-tested,
            # not added to the solver (it would be identically 0)
        else:
            raise ValueError(name)
    return out


# =================================================================================================
# step 3 — lambda (user decision 3) and the cuRobo candidates
# =================================================================================================

#: user decision 3 (2026-10-21): lambda in {1, 100, 1e4}, fixed before results
STEP3_LAMBDAS: tuple[float, ...] = (1.0, 100.0, 1e4)
#: the weights lambda multiplies — our objective terms. **Not** scaled: B4's target-margin channel
#: (a world channel, not a cost weight), cuRobo's own c-space / collision weights (cuRobo defaults),
#: and the servo-accel term (inactive in the baseline; `build_terms` refuses it if it were on).
LAMBDA_SCALED_WEIGHTS: tuple[str, ...] = ("w_track", "w_continuity", "w_deflection_rate")


def scaled_weights(weights: dict[str, Any], lam: float) -> dict[str, Any]:
    """`weights` with `LAMBDA_SCALED_WEIGHTS` multiplied by `lam` (> 0); everything else as is."""
    lam = float(lam)
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError(f"lambda must be a positive finite number, got {lam}")
    out = dict(weights)
    for k in LAMBDA_SCALED_WEIGHTS:
        out[k] = float(weights[k]) * lam
    return out


def parse_lambdas(text: str) -> list[float]:
    vals = [float(v) for v in str(text).split(",") if v.strip()]
    if not vals:
        raise ValueError("--lambda needs at least one value")
    for v in vals:
        scaled_weights({k: 1.0 for k in LAMBDA_SCALED_WEIGHTS}, v)  # validates
    return vals


#: variants without our objective terms (no lambda)
NO_LAMBDA_VARIANTS: tuple[str, ...] = ("B0", "B0m")


def candidate_name(variant: str, lam: Optional[float]) -> str:
    """Output name of a cuRobo candidate: `B0` / `B0m` (no lambda) or `<variant>_lam<lambda>`."""
    if variant in NO_LAMBDA_VARIANTS:
        if lam is not None:
            raise ValueError(f"{variant} has no TO terms, so no lambda")
        return variant
    if lam is None:
        raise ValueError(f"{variant} needs a lambda")
    return f"{variant}_lam{float(lam):g}"


def step3_candidates(variants, lambdas) -> list[tuple[str, str, Optional[float]]]:
    """`[(name, variant, lambda)]`: B0 / B0m once, every other variant once per lambda."""
    out = []
    for v in variants:
        if v not in VARIANTS:
            raise ValueError(f"unknown variant {v}")
        if v in NO_LAMBDA_VARIANTS:
            out.append((candidate_name(v, None), v, None))
        else:
            out.extend((candidate_name(v, lam), v, float(lam)) for lam in lambdas)
    return out


def fk_spheres(robot, spec, q: np.ndarray, P: np.ndarray) -> np.ndarray:
    """`(T, 14)` + finger path `(4, >=T)` -> `(T, S, 4)` spheres in the torso_5 frame (our order),
    the finger value of each step applied (grouped by identical values)."""
    from AG3S.trajopt.curobo_v2 import spec as S

    T = q.shape[0]
    out = np.empty((T, spec["n_spheres"], 4))
    groups: dict[tuple, list[int]] = {}
    for k in range(T):
        groups.setdefault(tuple(np.round(P[:, k], 12)), []).append(k)
    for steps in groups.values():
        robot.set_fingers(S.finger_values(spec, P, steps[0]))
        out[steps] = robot.spheres(q[steps])
    return out


class OurWorlds:
    """Our field code on one record (`AG3S.ag3s.fields.curobo_field`, byte-identical to the
    snapshot): the per-layer answers and the compositions of `world.py`."""

    def __init__(self, z, outside: float):
        from AG3S.ag3s.fields.curobo_field import CuroboEsdfField, layer_from_arrays
        from AG3S.trajopt.curobo_v2 import world as W

        def layer(prefix):
            return layer_from_arrays(z[f"{prefix}_distance"], z[f"{prefix}_origin"],
                                     float(z[f"{prefix}_voxel_size"]))

        self.W = W
        self.coarse = CuroboEsdfField((layer("esdf_coarse"),), outside_distance=outside)
        self.fine = layer("esdf")
        self.has_free = "esdf_free_distance" in z
        self.free = layer("esdf_free") if self.has_free else None
        self.arr = {n: W.layer_arrays(z, n) for n in (("coarse", "main", "free") if self.has_free
                                                       else ("coarse", "main"))}
        self.main_field = CuroboEsdfField((layer("esdf_coarse"), layer("esdf")),
                                          outside_distance=outside,
                                          target_free_layers=(self.free,) if self.has_free else ())

    def layers(self, pts):
        W = self.W
        d_coarse = np.asarray(self.coarse._evaluate(pts)[0], np.float64)
        in_fine = W.inside_lattice(pts, self.arr["main"]["origin"],
                                   self.arr["main"]["distance"].shape, self.arr["main"]["voxel_size"])
        d_fine = np.full(len(pts), np.inf)
        d_fine[in_fine] = self.fine.distance(pts[in_fine])
        if self.has_free:
            a = self.arr["free"]
            in_free = W.inside_lattice(pts, a["origin"], a["distance"].shape, a["voxel_size"])
            d_free = np.full(len(pts), np.inf)
            d_free[in_free] = self.free.distance(pts[in_free])
        else:
            in_free, d_free = in_fine, d_fine
        return {"coarse": d_coarse, "fine": d_fine, "in_fine": in_fine, "free": d_free,
                "in_free": in_free}

    def b0(self, L):
        """user decision 2: the target-excluded box inside its lattice, else coarse. Without
        `esdf_free` (step 3, user decision 3 lead default): our main field `min(coarse, fine)`
        inside the fine lattice (`world.compose_b0`). Returns `(d, voxel size of the answer)`."""
        d = self.W.compose_b0(L["free"], L["in_free"], L["coarse"], has_free=self.has_free)
        if self.has_free:
            box_answers = L["in_free"]
            vs_box = self.arr["free"]["voxel_size"]
        else:
            box_answers = L["in_fine"] & (L["fine"] < L["coarse"])
            vs_box = self.arr["main"]["voxel_size"]
        vs = np.where(box_answers, vs_box, self.arr["coarse"]["voxel_size"])
        return d, vs

    def lte(self, L, excl_rows):
        d = self.W.compose_lte(L["coarse"], L["fine"], L["in_fine"], L["free"], L["in_free"],
                               excl_rows)
        fine_wins = L["in_fine"] & (L["fine"] < L["coarse"])
        vs_main = np.where(fine_wins, self.arr["main"]["voxel_size"],
                           self.arr["coarse"]["voxel_size"])
        _, vs_box = self.b0(L)
        return d, np.where(excl_rows, vs_box, vs_main)


def _cu_order(arr_ours: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """our sphere order -> cuRobo's (`perm[i_curobo] = i_ours`), last axis."""
    return np.asarray(arr_ours)[..., perm]


def contact_rows(contact: dict[int, Any], ep: int, i: int, source: str, record_mask):
    """`(excl (S,), margin (S,) NaN = none)` in our order for B4."""
    c = contact[ep]
    margin = np.asarray(c["margin"][i], np.float64)
    excl = (np.asarray(record_mask, bool) if source == "record"
            else np.asarray(c["mask"][i], bool))
    return excl, margin


def exempt_target_margin(excl: np.ndarray, margin: np.ndarray) -> np.ndarray:
    """Rows that ask the target-free layer get **no** target margin (T30 F1: `linearize.py`
    `_esdf_clearance_core`, `is_object &= ~exempt`; `exclude_authorized` -> exempt = the mask).
    Returns the margin with NaN on those rows."""
    return np.where(np.asarray(excl, bool), np.nan, np.asarray(margin, np.float64))


def load_b0m_rules(path: Optional[str], episodes) -> dict[int, Any]:
    """`step3_eval.py --export-b0m-rules` output: `{episode: {chunk_index: row index, arrays}}`."""
    if not path:
        return {}
    out = {}
    for e in episodes:
        a = dict(np.load(pathlib.Path(path) / f"b0m_rules_{e}.npz"))
        a["_index"] = {int(c): j for j, c in enumerate(a["chunk_index"])}
        out[e] = a
    return out


def b0m_rules_of(rules_ep: dict[str, Any], chunk_index: int, target_points) -> dict[str, Any]:
    """One chunk's `b0m.rules_from_scene` (our order) back from the export; the target points are
    the record's (`target_points`, what `build_scene` gives the scene) — their count is checked."""
    j = rules_ep["_index"].get(int(chunk_index))
    if j is None:
        raise SystemExit(f"chunk {chunk_index} is not in the B0m rule export")
    has_target = bool(rules_ep["has_target"][j])
    target: dict[str, Any] = {"kind": None}
    if has_target:
        tp = np.asarray(target_points, np.float64).reshape(-1, 3)
        if len(tp) != int(rules_ep["n_target_points"][j]):
            raise SystemExit(f"chunk {chunk_index}: {len(tp)} target points != exported "
                             f"{int(rules_ep['n_target_points'][j])}")
        target = {"kind": "points", "points": tp}
    k = int(rules_ep["n_planes"][j])
    return {"obstacle_margin": float(rules_ep["obstacle_margin"][j]),
            "target_tol": float(rules_ep["target_tol"][j]),
            "support_tol": float(rules_ep["support_tol"][j]),
            "contact": np.asarray(rules_ep["contact"][j], bool), "target": target,
            "planes": np.asarray(rules_ep["planes"][j], np.float64)[:k],
            "support_exempt": bool(rules_ep["support_exempt"][j]),
            "strict": bool(rules_ep["strict"][j])}


def load_contact(path: Optional[str], episodes) -> dict[int, Any]:
    if not path:
        return {}
    return {e: dict(np.load(pathlib.Path(path) / f"contact_{e}.npz")) for e in episodes}


def run_parity_composed(args, spec, data, contact) -> dict[str, Any]:
    """Parity of the composed worlds: `ComposedSceneCollision` (cuRobo kernels, our composition
    rule) vs `OurWorlds` (our field code, the same rule), on the recorded refined chunk's plan
    window. Tolerance = `voxel_size/2` of the answering layer, fixed before the run (as step 0)."""
    import torch
    from scipy.spatial import cKDTree

    from AG3S.trajopt.curobo_v2 import spec as S
    from AG3S.trajopt.curobo_v2 import world as W
    from AG3S.trajopt.curobo_v2.composed_world import ComposedSceneCollision
    from AG3S.trajopt.curobo_v2.robot_world import RobotModel, device_cfg

    dev = device_cfg()
    robot = RobotModel(spec, dev)
    perm = S.sphere_permutation(spec)
    inv = np.argsort(perm)
    worlds = {w: ComposedSceneCollision.build(w, spec["n_spheres"], dev) for w in args.worlds}
    outside = float(spec["field"]["max_distance"])
    acc = {w: {"diff": [], "tol": [], "inside_coarse": [], "box_answer": []} for w in worlds}
    tgt_diff: list[np.ndarray] = []
    extra = {"rows_free_vs_to_formula_over_tol": 0, "rows_free_window_edge_checked": 0,
             "rows_nofree_fine_vs_main_over_tol": 0, "rows_nofree_in_box": 0,
             "n_chunks_no_free": 0, "n_chunks_target_rows": 0}
    per_chunk = []
    n_skipped_has_free = 0
    for e, ep in data.items():
        n = len(ep["chunk_file"])
        idx = range(n) if args.chunks == "all" else [int(c) for c in args.chunks.split(",")]
        for i in idx:
            path = str(ep["chunk_file"][i])
            with np.load(path, allow_pickle=False) as zf:
                if getattr(args, "only_no_free", False) and "esdf_free_distance" in zf.files:
                    # step 3: only the chunks whose b0 rule changed (no target-excluded layer)
                    n_skipped_has_free += 1
                    continue
                z = {k: zf[k] for k in zf.files if k.startswith("esdf") or k in (
                    "refined_chunk", "target_points", "sphere_radii")}
            T = np.asarray(ep["T_base_torso5"][i], np.float64)
            P = np.asarray(ep["finger_path_plan"][i], np.float64)
            H = P.shape[1]
            q = S.chunk_to_cspace(z["refined_chunk"], spec)[:H]
            sph = fk_spheres(robot, spec, q, P)  # (H, S, 4) torso5, ours
            r = sph[0, :, 3]
            pts = W.transform_points(T, sph[..., :3].reshape(-1, 3))
            ours = OurWorlds(z, outside)
            L = ours.layers(pts)
            Lr = {k: v.reshape(H, -1) for k, v in L.items()}
            rec = {"episode": e, "chunk_index": int(ep["chunk_index"][i]), "has_free": ours.has_free}
            if not ours.has_free:
                extra["n_chunks_no_free"] += 1
                # step 2 rule (main fine alone inside the box) vs our main composition (min with
                # coarse, the b0 rule since step 3): how often the two answer differently there
                inb = L["in_fine"]
                extra["rows_nofree_in_box"] += int(inb.sum())
                dmain = W.compose_main(L["coarse"], L["fine"], L["in_fine"])
                extra["rows_nofree_fine_vs_main_over_tol"] += int(
                    (np.abs(dmain - L["fine"])[inb] > 0.0025).sum())
            else:
                # our TO's own target-free rule inside the window is min(f, max(d, b)); how often
                # it differs from f alone (window edge) on the excluded rows
                tf = np.asarray(ours.main_field.target_free_distance(pts), np.float64)
                inb = L["in_free"]
                extra["rows_free_window_edge_checked"] += int(inb.sum())
                extra["rows_free_vs_to_formula_over_tol"] += int(
                    (np.abs(tf - L["free"])[inb] > 0.0025).sum())
            sph_cu = sph[:, perm].reshape(-1, 4)
            for wname, cw in worlds.items():
                excl = margin = None
                if wname == "lte":
                    excl, margin = contact_rows(contact, e, i, args.b4_mask,
                                                ep["target_field_exclude"][i])
                    if not ours.has_free:
                        excl = np.zeros_like(excl)
                    margin = exempt_target_margin(excl, margin)
                    tp = z.get("target_points")
                    cw.load_chunk(z, T, excl_curobo=_cu_order(excl, perm),
                                  target_margin_curobo=_cu_order(margin, perm),
                                  target_points=None if tp is None or not len(tp) else tp)
                else:
                    cw.load_chunk(z, T)
                sd = cw.signed_distance(sph_cu, part="world").reshape(H, -1)[:, inv]
                sd = np.where(np.isfinite(sd), sd, outside)
                if wname == "b0":
                    d_ref, vs = ours.b0(L)
                else:
                    d_ref, vs = ours.lte(L, np.tile(excl, H))
                d_ref = d_ref.reshape(H, -1)
                diff = (sd - d_ref) * 1e3
                tol = vs.reshape(H, -1) * 500.0
                acc[wname]["diff"].append(diff.reshape(-1))
                acc[wname]["tol"].append(tol.reshape(-1))
                in_c = W.inside_lattice(pts, ours.arr["coarse"]["origin"],
                                        ours.arr["coarse"]["distance"].shape,
                                        ours.arr["coarse"]["voxel_size"])
                acc[wname]["inside_coarse"].append(in_c)
                acc[wname]["box_answer"].append(L["in_free"] if wname == "b0" else
                                                np.where(np.tile(excl, H), L["in_free"],
                                                         L["in_fine"]))
                rec[f"{wname}_max_abs_diff_mm"] = float(np.abs(diff).max())
                rec[f"{wname}_n_over_tol"] = int((np.abs(diff) > tol).sum())
                if wname == "lte" and np.isfinite(margin).any() and z.get("target_points") is not None \
                        and len(z["target_points"]):
                    extra["n_chunks_target_rows"] += 1
                    st = cw.signed_distance(sph_cu, part="target").reshape(H, -1)[:, inv]
                    d_obj = cKDTree(np.asarray(z["target_points"], np.float64)).query(pts)[0] \
                        .reshape(H, -1)
                    on = np.isfinite(margin)[None, :] & (d_obj < 0.08) & np.isfinite(st)
                    tgt_diff.append(np.stack([((st - d_obj) * 1e3)[on], d_obj[on] * 1e3], 1))
            per_chunk.append(rec)
        print(f"[t63-parity-composed] ep{e}: {len(idx)} chunks", flush=True)
        torch.cuda.empty_cache()
    summary: dict[str, Any] = {"n_chunks": len(per_chunk), "extra": extra,
                               "only_no_free": bool(getattr(args, "only_no_free", False)),
                               "n_chunks_skipped_with_free_layer": n_skipped_has_free}
    for wname, a in acc.items():
        d = np.concatenate(a["diff"])
        t = np.concatenate(a["tol"])
        ic = np.concatenate(a["inside_coarse"])
        bx = np.concatenate(a["box_answer"])
        over = np.abs(d) > t
        summary[wname] = {
            "n_rows": int(d.size), "abs_diff_mm_all": dist_stats(d),
            "abs_diff_mm_box_answer": dist_stats(d[bx]),
            "abs_diff_mm_coarse_answer": dist_stats(d[~bx & ic]),
            "n_over_tol": int(over.sum()), "n_over_tol_box_answer": int((over & bx).sum()),
            "n_over_tol_coarse_answer": int((over & ~bx & ic).sum()),
            "n_rows_outside_coarse_lattice": int((~ic).sum()),
            "n_over_tol_outside_coarse_lattice": int((over & ~ic).sum()),
            "fraction_over_tol": float(over.mean()),
            "gate_pass": bool(over.sum() == 0),
        }
    if tgt_diff:
        tdo = np.concatenate(tgt_diff)
        td, dob = tdo[:, 0], tdo[:, 1]
        summary["lte_target_channel"] = {
            "rows": int(td.size), "abs_diff_mm": dist_stats(td),
            "n_over_2p5mm": int((np.abs(td) > 2.5).sum()),
            "abs_diff_mm_dobj_lt_10mm": dist_stats(td[dob < 10.0]),
            "abs_diff_mm_dobj_10_to_80mm": dist_stats(td[dob >= 10.0]),
            "n_over_2p5mm_dobj_10_to_80mm": int((np.abs(td[dob >= 10.0]) > 2.5).sum()),
            "definition": "target channel (cuRobo kernel on the 5 mm target-distance grid) vs "
                          "cKDTree distance to the target points, rows with a target margin "
                          "and d_obj < 80 mm"}
    summary["gate_rule"] = ("PASS iff no row exceeds voxel_size/2 of the layer that answers it "
                            "(box layer 2.5 mm, coarse 10 mm); fixed before the run")
    summary["tolerance_mm"] = {"box/fine/free": 2.5, "coarse": 10.0}
    return {"summary": summary, "chunks": per_chunk}


def run_parity_b0m(args, spec, data, b0m_rules) -> dict[str, Any]:
    """T63b B0m parity on the recorded `refined_chunk`'s plan window (per row = sphere x step):

    (A) **rule parity** — cuRobo `ComposedSceneCollision` mode `b0m` (the optimizer's own queries:
        `signed_distance` = `d_b0 − margin`, `b0m_classes`) vs the numpy reference (`OurWorlds.b0`
        + `b0m.classify` with `cKDTree` target distance). Tolerance `voxel/2` of the answering layer
        (as `parity_composed`) on rows whose classes agree; class disagreements are counted with
        how far the row is from the rule's tolerance edge.
    (B) **our TO's certified clearance** — `c_TO + r` (the evaluator export, `step3_eval.py
        --export-b0m-rules`) vs the cuRobo B0m world distance, on the rows where our TO and B0
        read the same distance (`|d_TO − d_b0| ≤ voxel/2`) and B0m's margin is our TO's margin;
        the other rows are counted by reason (contact margin, world differs, class differs)."""
    import torch
    from scipy.spatial import cKDTree

    from AG3S.trajopt.curobo_v2 import b0m as B0M
    from AG3S.trajopt.curobo_v2 import spec as S
    from AG3S.trajopt.curobo_v2 import world as W
    from AG3S.trajopt.curobo_v2.composed_world import ComposedSceneCollision
    from AG3S.trajopt.curobo_v2.robot_world import RobotModel, device_cfg

    dev = device_cfg()
    robot = RobotModel(spec, dev)
    perm = S.sphere_permutation(spec)
    inv = np.argsort(perm)
    cw = ComposedSceneCollision.build("b0m", spec["n_spheres"], dev)
    outside = float(spec["field"]["max_distance"])
    import AG3S.trajopt.curobo_v2.composed_world as CWM

    if getattr(args, "b0m_target_grid", None):
        vox, pad = (float(v) for v in args.b0m_target_grid.split(","))
        CWM.B0M_TARGET_GRID_VOXEL_M, CWM.B0M_TARGET_GRID_PAD_M = vox, pad
    #: rows whose B0m margin can change cuRobo's cost: d_b0 - r < obstacle_margin + activation
    #: (0.01) + 10 mm slack
    act_zone = 0.03
    A = {"diff": [], "tol": [], "agree": [], "cls_np": [], "cls_cu": [], "edge_t": [], "edge_s": [],
         "obj_err": [], "obj_np": [], "obj_err_all": [], "grid_voxel": set(), "acting": []}
    B = {"diff_ok": [], "n_rows": 0, "n_same_d": 0, "n_same_margin": 0,
         "reasons": {}, "class_pairs": {}}
    per_chunk = []
    for e, ep in data.items():
        rex = b0m_rules[e]
        n = len(ep["chunk_file"])
        idx = range(n) if args.chunks == "all" else [int(c) for c in args.chunks.split(",")]
        for i in idx:
            path = str(ep["chunk_file"][i])
            with np.load(path, allow_pickle=False) as zf:
                z = {k: zf[k] for k in zf.files if k.startswith("esdf") or k in (
                    "refined_chunk", "target_points", "sphere_radii")}
            k = int(ep["chunk_index"][i])
            rules = b0m_rules_of(rex, k, z.get("target_points"))
            j = rex["_index"][k]
            T = np.asarray(ep["T_base_torso5"][i], np.float64)
            P = np.asarray(ep["finger_path_plan"][i], np.float64)
            H = P.shape[1]
            q = S.chunk_to_cspace(z["refined_chunk"], spec)[:H]
            sph = fk_spheres(robot, spec, q, P)  # (H, S, 4) torso5, ours
            r = sph[0, :, 3]
            pts = W.transform_points(T, sph[..., :3].reshape(-1, 3))
            ours = OurWorlds(z, outside)
            L = ours.layers(pts)
            d_b0, vs = ours.b0(L)
            d_b0 = d_b0.reshape(H, -1)
            tol = vs.reshape(H, -1) * 0.5
            d_obj = B0M.target_distance(pts, rules["target"])
            d_obj = None if d_obj is None else d_obj.reshape(H, -1)
            cls_np = B0M.classify(d_b0, pts.reshape(H, -1, 3), d_obj, rules)
            m_np = B0M.margin(cls_np, rules["obstacle_margin"])
            winfo = cw.load_chunk(z, T, b0m_rules=B0M.to_curobo(rules, perm))
            A["grid_voxel"].add(winfo["b0m"]["target_grid_voxel_m"])
            sph_cu = sph[:, perm].reshape(-1, 4)
            sd = cw.signed_distance(sph_cu, part="world").reshape(H, -1)[:, inv]
            sd = np.where(np.isfinite(sd), sd, outside - m_np)
            cc = cw.b0m_classes(sph_cu)
            cls_cu = cc["cls"].reshape(H, -1)[:, inv]
            agree = cls_cu == cls_np
            diff = sd - (d_b0 - m_np)
            A["diff"].append(diff[agree]); A["tol"].append(tol[agree])
            A["agree"].append(agree.reshape(-1))
            A["cls_np"].append(cls_np.reshape(-1)); A["cls_cu"].append(cls_cu.reshape(-1))
            A["acting"].append(((d_b0 - r[None]) < act_zone).reshape(-1))
            if d_obj is not None:
                # target channel (grid) vs cKDTree, rows within 30 mm of the target (where the
                # margin can act: d < r + obstacle_margin + activation)
                o_cu = cc["d_obj"].reshape(H, -1)[:, inv]
                near = d_obj < 0.03
                A["obj_err_all"].append(np.where(np.isfinite(o_cu), o_cu - d_obj, np.inf)[near])
            if (~agree).any():
                # how far from the rule's edges the disagreeing rows are (target |d-d_obj| - tol,
                # support min_k |d - s_k| - tol_s)
                if d_obj is not None:
                    A["edge_t"].append((np.abs(d_b0 - d_obj) - rules["target_tol"])[~agree])
                    A["obj_err"].append((o_cu - d_obj)[~agree])
                    A["obj_np"].append(d_obj[~agree])
                pl = rules["planes"]
                if len(pl):
                    s_k = pts.reshape(H, -1, 3) @ pl[:, :3].T - pl[:, 3]
                    A["edge_s"].append((np.abs(s_k - d_b0[..., None]).min(-1)
                                        - rules["support_tol"])[~agree])
            # (B) vs our TO's certified clearance (evaluator export, same chunk, plan window)
            c_to = np.asarray(rex["to_clear"][j], np.float64)[:H]
            d_to = np.asarray(rex["to_d"][j], np.float64)[:H]
            m_to = np.asarray(rex["to_margin"][j], np.float64)[:H]
            cls_to = np.asarray(rex["to_cls"][j])[:H]
            same_d = np.abs(d_to - d_b0) <= tol
            m_b0m = B0M.margin(cls_cu, rules["obstacle_margin"])
            same_m = np.abs(m_to - m_b0m) <= 1e-6
            B["n_rows"] += int(same_d.size)
            B["n_same_d"] += int(same_d.sum())
            B["n_same_margin"] += int((same_d & same_m).sum())
            acting = (d_b0 - r[None]) < act_zone
            B["n_acting"] = B.get("n_acting", 0) + int(acting.sum())
            B["n_acting_same_d_margin_differs"] = (B.get("n_acting_same_d_margin_differs", 0)
                                                   + int((acting & same_d & ~same_m).sum()))
            ok = same_d & same_m
            B["diff_ok"].append(((c_to + r[None]) - sd)[ok])
            # a TO margin B0m does not carry: a contact margin (target rows with a margin, or any
            # margin other than 0 / obstacle_margin, e.g. 50 mm on non-authorized rows)
            contact_m = (~np.isclose(m_to, 0.0, atol=1e-6)
                         & (~np.isclose(m_to, rules["obstacle_margin"], atol=1e-6)
                            | (cls_to == B0M.TARGET)))
            for lab, sel in (("world differs (|d_TO - d_b0| > voxel/2)", ~same_d),
                             ("contact margin (TO target-row / non-authorized margin, not in B0m)",
                              same_d & ~same_m & contact_m),
                             ("class differs (target test: target grid vs cKDTree)",
                              same_d & ~same_m & ~contact_m)):
                B["reasons"][lab] = B["reasons"].get(lab, 0) + int(sel.sum())
            sel = same_d & ~same_m
            for a, b in zip(cls_to[sel].reshape(-1), cls_cu[sel].reshape(-1)):
                key = f"TO {B0M.CLASS_NAMES[a] if 0 <= a < 3 else a} / B0m {B0M.CLASS_NAMES[b]}"
                B["class_pairs"][key] = B["class_pairs"].get(key, 0) + 1
            per_chunk.append({"episode": e, "chunk_index": k, "has_free": ours.has_free,
                              "n_class_disagree": int((~agree).sum()),
                              "max_abs_diff_mm_agree": float(np.abs(diff[agree]).max() * 1e3)
                              if agree.any() else None,
                              "classes_curobo": B0M.counts(cls_cu),
                              "n_rows_same_as_to": int(ok.sum())})
        print(f"[t63-parity-b0m] ep{e}: {len(idx)} chunks", flush=True)
        torch.cuda.empty_cache()
    d = np.concatenate(A["diff"]) * 1e3
    t = np.concatenate(A["tol"]) * 1e3
    ag = np.concatenate(A["agree"])
    acting = np.concatenate(A["acting"])
    cn, cc_ = np.concatenate(A["cls_np"]), np.concatenate(A["cls_cu"])
    dB = np.concatenate(B["diff_ok"]) * 1e3 if B["diff_ok"] else np.zeros(0)
    summary = {
        "n_chunks": len(per_chunk),
        "A_rule_parity": {
            "n_rows": int(ag.size), "n_class_agree": int(ag.sum()),
            "acting_zone": "d_b0 - r < 30 mm (obstacle_margin 10 + activation 10 + 10 slack): the "
                           "rows whose B0m margin can change cuRobo's cost",
            "n_rows_acting": int(acting.sum()), "n_class_agree_acting": int((ag & acting).sum()),
            "confusion_acting": {f"np {B0M.CLASS_NAMES[a]} / cu {B0M.CLASS_NAMES[b]}":
                                 int(((cn == a) & (cc_ == b) & acting).sum())
                                 for a in range(3) for b in range(3)
                                 if ((cn == a) & (cc_ == b) & acting).any()},
            "class_counts_numpy": B0M.counts(cn), "class_counts_curobo": B0M.counts(cc_),
            "confusion": {f"np {B0M.CLASS_NAMES[a]} / cu {B0M.CLASS_NAMES[b]}":
                          int(((cn == a) & (cc_ == b)).sum()) for a in range(3) for b in range(3)
                          if ((cn == a) & (cc_ == b)).any()},
            "abs_diff_mm_class_agree": dist_stats(d), "n_over_tol_class_agree": int((np.abs(d) > t).sum()),
            "disagree_target_edge_mm": dist_stats(np.concatenate(A["edge_t"]) * 1e3)
            if A["edge_t"] else None,
            "disagree_support_edge_mm": dist_stats(np.concatenate(A["edge_s"]) * 1e3)
            if A["edge_s"] else None,
            "disagree_target_dobj_err_mm": dist_stats(np.concatenate(A["obj_err"]) * 1e3)
            if A["obj_err"] else None,
            "disagree_target_dobj_mm": dist_stats(np.concatenate(A["obj_np"]) * 1e3)
            if A["obj_np"] else None,
            "target_channel_err_mm_rows_dobj_lt_30mm": dist_stats(
                np.concatenate(A["obj_err_all"]) * 1e3) if A["obj_err_all"] else None,
            "target_grid_voxel_m": sorted(v for v in A["grid_voxel"] if v is not None),
            "gate_rule": "class-agreeing rows within voxel/2 of the answering layer (fixed before "
                         "the run); class disagreements reported with their distance to the "
                         "rule's tolerance edge (target channel: distance grid vs cKDTree)"},
        "B_vs_our_certified_clearance": {
            "definition": "(c_TO + r) - d_B0m(cuRobo) on rows with |d_TO - d_b0| <= voxel/2 and "
                          "B0m margin == our TO's margin; c_TO = evaluator clearance (server "
                          "config, record contact policy) of the recorded refined plan window",
            "n_rows": B["n_rows"], "n_same_distance": B["n_same_d"],
            "n_same_distance_and_margin": B["n_same_margin"],
            "n_rows_acting": B.get("n_acting", 0),
            "n_rows_acting_same_distance_margin_differs": B.get("n_acting_same_d_margin_differs", 0),
            "abs_diff_mm": dist_stats(dB),
            "n_over_2p5mm": int((np.abs(dB) > 2.5).sum()) if dB.size else 0,
            "excluded_rows_by_reason": B["reasons"], "margin_mismatch_class_pairs": B["class_pairs"]},
    }
    return {"summary": summary, "chunks": per_chunk}


def step3_chunk_inputs(spec, ep, e: int, i: int, contact, b4_mask: str,
                       b0m_rules: Optional[dict] = None) -> dict[str, Any]:
    """The fixed per-chunk inputs every cuRobo candidate gets (as `run_smoke`, without our field)."""
    from AG3S.trajopt.curobo_v2 import spec as S
    from AG3S.trajopt.curobo_v2.record_context import chunk_record_context

    names = list(spec["sphere_link_names"])
    path = str(ep["chunk_file"][i])
    with np.load(path, allow_pickle=False) as zf:
        z = {k: zf[k] for k in zf.files if k.startswith("esdf") or k in (
            "reference_chunk", "refined_chunk", "target_points")}
    excl, margin = (contact_rows(contact, e, i, b4_mask, ep["target_field_exclude"][i])
                    if contact else (np.zeros(len(names), bool), np.full(len(names), np.nan)))
    if "esdf_free_distance" not in z:
        excl = np.zeros_like(excl)
    margin = exempt_target_margin(excl, margin)
    Pfull = np.asarray(ep["finger_path_full"][i], np.float64)
    rules = (None if not b0m_rules else
             b0m_rules_of(b0m_rules[e], int(ep["chunk_index"][i]), z.get("target_points")))
    return {"episode": e, "i": i, "chunk_index": int(ep["chunk_index"][i]), "b0m_rules": rules,
            "seq": int(ep["seq"][i]), "file": path, "z": z, "ctx": chunk_record_context(path),
            "T": np.asarray(ep["T_base_torso5"][i], np.float64),
            "q_now": S.q_now_cspace(ep["q_now"][i], spec),
            "ref50": S.chunk_to_cspace(z["reference_chunk"], spec),
            "excl": excl, "margin": margin, "fingers": S.finger_values(spec, Pfull, 0),
            "has_free": "esdf_free_distance" in z}


def _chunk_stream(spec, data, positions, contact, b4_mask, prefetch: bool = False,
                  b0m_rules: Optional[dict] = None):
    """`step3_chunk_inputs` of every selected chunk, in order. `prefetch` reads the next record in a
    background thread while the current one is solved — off by default: in the 2026-10-21 smoke
    the warm wall time of B0 was 71 ms with it against 53 ms in step 2 on the same chunks (host
    contention during the timed solve), so timing runs load each record before its solve."""
    from concurrent.futures import ThreadPoolExecutor

    keys = []
    for e, ep in data.items():
        n = len(ep["chunk_file"])
        idx = (list(range(n)) if positions == "all"
               else [int(p) for p in positions.split(",") if int(p) < n])
        keys.extend((e, i) for i in idx)
    if not prefetch:
        for e, i in keys:
            yield step3_chunk_inputs(spec, data[e], e, i, contact, b4_mask, b0m_rules)
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = (pool.submit(step3_chunk_inputs, spec, data[keys[0][0]], keys[0][0], keys[0][1],
                           contact, b4_mask, b0m_rules) if keys else None)
        for n_k in range(len(keys)):
            c = fut.result()
            if n_k + 1 < len(keys):
                e2, i2 = keys[n_k + 1]
                fut = pool.submit(step3_chunk_inputs, spec, data[e2], e2, i2, contact, b4_mask,
                                  b0m_rules)
            yield c


def run_step3_curobo(args, spec, data, contact, wts, b0m_rules: Optional[dict] = None
                     ) -> dict[str, Any]:
    """T63 step 3, cuRobo side: B0 and B1-B4 x lambda on the chosen chunks. **No metric here** —
    the common evaluator (`curobo_v2/step3_eval.py`, snapshot side) measures every candidate.

    One solver per (variant, lambda): weights are constants of the captured CUDA graph. **One
    solver alive at a time** (built, run over every chosen chunk, deleted — the step 2 pattern):
    with all 13 solvers alive and the chunk loop outermost, the second chunk crashed inside cuRobo
    (`CUDA error: operation not supported on global/shared address space`, 2026-10-21 smoke), so
    the records are streamed once per candidate instead (`_chunk_stream`). A solver's first solve
    (CUDA-graph capture) is timed as **cold** and discarded; the same chunk is then solved again
    warm, so every kept result is a warm solve.

    Writes (`--out`): `trajectories/<candidate>/ep<E>.npz` (`chunk_index`, `position`, `cspace
    (n, 50, 14)` NaN where no result, `chunk (n, 50, 16)`, `success`, `finite`) when a candidate
    finishes, `curobo_rows.jsonl` (one line per chunk x candidate, appended as solved) and the
    returned dict (`curobo_step3.json`)."""
    import torch

    from AG3S.trajopt.curobo_v2 import spec as S
    from AG3S.trajopt.curobo_v2.robot_world import device_cfg
    from AG3S.trajopt.curobo_v2.solver import CuroboChunkSolver

    dev = device_cfg()
    rows_n = 50 if args.term_window == "chunk" else int(args.plan_rows)
    perm = S.sphere_permutation(spec)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "curobo_rows.jsonl"
    if rows_path.exists():
        raise SystemExit(f"{rows_path} exists — use a new --out (earlier outputs are not overwritten)")

    def solve_one(sv, c) -> dict[str, Any]:
        terms, solver = sv["terms"], sv["solver"]
        if "track" in terms:
            terms["track"].set_reference(c["ref50"])
        if "continuity" in terms:
            tail = c["ctx"]["continuity_tail"]
            terms["continuity"].set_previous(None if tail is None else tail[:rows_n])
        if "deflection_rate" in terms:
            terms["deflection_rate"].set_reference(c["ref50"])
            terms["deflection_rate"].set_history(c["ctx"]["deflection_history"])
        lte = {}
        if sv["world"] == "b0m":
            from AG3S.trajopt.curobo_v2 import b0m as B0M

            lte = dict(b0m_rules=B0M.to_curobo(c["b0m_rules"], perm))
        if sv["world"] == "lte":
            tp = c["z"].get("target_points")
            lte = dict(excl_curobo=_cu_order(c["excl"], perm),
                       target_margin_curobo=_cu_order(c["margin"], perm),
                       target_points=None if tp is None or not len(tp) else tp)
        winfo = solver.set_world(c["z"], c["T"], **lte)
        solver.set_fingers(c["fingers"])
        res = solver.solve(c["q_now"], c["z"]["reference_chunk"])
        res["world_info"] = winfo
        return res

    cands = step3_candidates(args.variants, args.lambdas)
    t_start = time.perf_counter()
    cand_out: dict[str, Any] = {}
    n_chunks = 0
    gpu_peak = []
    with rows_path.open("w") as fh:
        for name, v, lam in cands:
            world, term_names = VARIANTS[v]
            w = wts["weights"] if lam is None else scaled_weights(wts["weights"], lam)
            terms = build_terms(term_names, w, rows_n, dev)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            solver = CuroboChunkSolver(spec, num_seeds=args.num_seeds, timing=args.timing,
                                       world=world, extra_terms=list(terms.values()),
                                       term_lambda=lam)
            sv = {"variant": v, "lambda": lam, "world": world, "terms": terms, "solver": solver}
            init_s = time.perf_counter() - t0
            cold = None
            buf: dict[int, dict[str, list]] = {}
            n_c = 0
            tc = time.perf_counter()
            for c in _chunk_stream(spec, data, args.b0_chunks, contact, args.b4_mask,
                                   prefetch=args.prefetch,
                                   b0m_rules=b0m_rules if world == "b0m" else None):
                e = c["episode"]
                if cold is None:
                    r0 = solve_one(sv, c)  # CUDA-graph capture: timed as cold, discarded
                    cold = {"episode": e, "chunk_index": c["chunk_index"], "wall_ms": r0["wall_ms"],
                            "gpu_solve_ms": r0["solve_time_ms"], "success": r0["success"]}
                res = solve_one(sv, c)
                breakdown = None
                if args.objective_breakdown and res["finite"]:
                    breakdown = solver.optimizer_breakdown(c["q_now"], c["z"]["reference_chunk"])
                failing = {k: x["n_steps_positive"] for k, x in
                           (res.get("curobo_constraints") or {}).items()
                           if x["n_steps_positive"] > 0}
                row = {"episode": e, "chunk_index": c["chunk_index"], "seq": c["seq"],
                       "position": c["i"], "candidate": name, "variant": v, "lambda": lam,
                       "world": res["world_info"], "success": res["success"],
                       "finite": res["finite"], "failing_constraints": failing,
                       "wall_ms": res["wall_ms"], "gpu_solve_ms": res["solve_time_ms"],
                       "total_time_ms": res["total_time_ms"], "cold": False,
                       "device": "gpu (cuda; wall = host perf_counter around solve_cspace with "
                                 "cuda.synchronize)",
                       "goal_error_rad": res["goal_error_rad"],
                       "start_error_rad": res["start_error_rad"],
                       "continuity_overlap": (0 if c["ctx"]["continuity_tail"] is None
                                              else min(len(c["ctx"]["continuity_tail"]), rows_n)),
                       "deflection_boundary": c["ctx"]["deflection_history"] is not None,
                       "optimizer_objective_at_solution": (None if breakdown is None
                                                           else breakdown["solution"])}
                fh.write(json.dumps(_jsonable(row)) + "\n")
                q = res["cspace"]
                b = buf.setdefault(e, {"chunk_index": [], "position": [], "cspace": [],
                                       "chunk": [], "success": [], "finite": []})
                b["chunk_index"].append(c["chunk_index"])
                b["position"].append(c["i"])
                b["cspace"].append(np.full((50, 14), np.nan) if q is None else q)
                b["chunk"].append(np.full((50, 16), np.nan) if res["chunk"] is None
                                  else res["chunk"])
                b["success"].append(bool(res["success"]))
                b["finite"].append(bool(res["finite"]))
                n_c += 1
                if n_c % 25 == 0:
                    fh.flush()
                    print(f"[t63-step3] {name}: {n_c} chunks ({time.perf_counter() - tc:.0f} s)",
                          flush=True)
            fh.flush()
            tdir = out_dir / "trajectories" / name
            tdir.mkdir(parents=True, exist_ok=True)
            for e, b in buf.items():
                np.savez_compressed(tdir / f"ep{e}.npz", **{k: np.asarray(x) for k, x in b.items()})
            peak = (torch.cuda.max_memory_allocated() / 2 ** 30
                    if torch.cuda.is_available() else None)
            gpu_peak.append(peak)
            cand_out[name] = {"variant": v, "lambda": lam, "world": world, "terms": list(terms),
                              "weights_used": {k: w[k] for k in LAMBDA_SCALED_WEIGHTS},
                              "init_s": init_s, "cold_solve": cold, "n_chunks": n_c,
                              "elapsed_s": time.perf_counter() - tc,
                              "gpu_max_memory_allocated_gib": peak,
                              "settings": solver.settings_record()}
            n_chunks = max(n_chunks, n_c)
            print(f"[t63-step3] {name} done: {n_c} chunks in {cand_out[name]['elapsed_s']:.0f} s "
                  f"(init {init_s:.1f} s, cold {None if cold is None else round(cold['wall_ms'])} ms)",
                  flush=True)
            del solver, sv, terms
            torch.cuda.empty_cache()
    elapsed = time.perf_counter() - t_start
    return {
        "weights": wts, "lambdas": list(args.lambdas),
        "lambda_scaled_weights": list(LAMBDA_SCALED_WEIGHTS),
        "lambda_not_scaled": ["B4 target-margin channel (world)", "cuRobo c-space / collision "
                              "weights (cuRobo defaults)", "servo-accel term (inactive)"],
        "term_rows": rows_n, "term_window": args.term_window, "b4_mask": args.b4_mask,
        "solver_order": "candidate-outer, one solver alive at a time",
        "prefetch": bool(args.prefetch),
        "candidates": cand_out, "n_chunks": n_chunks, "elapsed_s": elapsed,
        "elapsed_per_chunk_s": elapsed / max(n_chunks, 1),
        "elapsed_per_chunk_candidate_s": elapsed / max(n_chunks * len(cands), 1),
        "gpu_max_memory_allocated_gib": max([g for g in gpu_peak if g is not None], default=None),
        "rows_file": str(rows_path), "trajectories_dir": str(out_dir / "trajectories"),
    }


def term_values_ref(q50: np.ndarray, ref50: np.ndarray, ctx: dict[str, Any], weights, rows: int
                    ) -> dict[str, float]:
    """Our TO's term values (numpy reference, TO units) of a 50-row trajectory, on `rows` rows."""
    from AG3S.trajopt.curobo_v2 import objective_ref as O

    Q, R = q50[:rows].T, ref50[:rows].T
    tail = ctx.get("continuity_tail")
    P = None if tail is None else np.asarray(tail)[:rows].T
    w = weights
    return {
        "track": O.tracking_value(Q, R, w["w_track"], w["track_decay"]),
        "continuity": O.continuity_value(Q, P, w["w_continuity"]),
        "deflection_rate": O.deflection_rate_value(Q, R, w["w_deflection_rate"],
                                                   ctx.get("deflection_history")),
        "smooth_not_in_variants": O.smooth_value(Q, w["w_smooth"]),
    }


def run_smoke(args, spec, data, contact, wts) -> dict[str, Any]:
    import torch

    from AG3S.trajopt.curobo_v2 import spec as S
    from AG3S.trajopt.curobo_v2 import world as W
    from AG3S.trajopt.curobo_v2.record_context import chunk_record_context
    from AG3S.trajopt.curobo_v2.robot_world import RobotModel, device_cfg
    from AG3S.trajopt.curobo_v2.solver import CuroboChunkSolver

    from scipy.spatial import cKDTree

    dev = device_cfg()
    weights = wts["weights"]
    rows = 50 if args.term_window == "chunk" else int(args.plan_rows)
    robot = RobotModel(spec, dev)
    perm = S.sphere_permutation(spec)
    names = list(spec["sphere_link_names"])
    outside = float(spec["field"]["max_distance"])
    out_dir = pathlib.Path(args.out)

    # chunks and their fixed per-chunk inputs (shared by every variant)
    chunks = []
    for e, ep in data.items():
        n = len(ep["chunk_file"])
        idx = range(n) if args.b0_chunks == "all" else [int(c) for c in args.b0_chunks.split(",")]
        hand = str(ep["hand"])
        fidx = np.asarray([k for k, nm in enumerate(names) if nm in FINGER_LINKS[hand]])
        for i in idx:
            path = str(ep["chunk_file"][i])
            with np.load(path, allow_pickle=False) as zf:
                z = {k: zf[k] for k in zf.files if k.startswith("esdf") or k in (
                    "reference_chunk", "refined_chunk", "target_points")}
            ctx = chunk_record_context(path)
            T = np.asarray(ep["T_base_torso5"][i], np.float64)
            Pfull = np.asarray(ep["finger_path_full"][i], np.float64)
            ref50 = S.chunk_to_cspace(z["reference_chunk"], spec)
            sph_ref = fk_spheres(robot, spec, ref50, Pfull)
            tcp_ref = W.transform_points(T, sph_ref[:, fidx, :3].mean(1))
            excl, margin = (contact_rows(contact, e, i, args.b4_mask, ep["target_field_exclude"][i])
                            if contact else (np.zeros(len(names), bool),
                                             np.full(len(names), np.nan)))
            if "esdf_free_distance" not in z:
                excl = np.zeros_like(excl)
            margin = exempt_target_margin(excl, margin)
            chunks.append({"episode": e, "i": i, "chunk_index": int(ep["chunk_index"][i]),
                           "file": path, "z": z, "ctx": ctx, "T": T, "Pfull": Pfull,
                           "q_now": S.q_now_cspace(ep["q_now"][i], spec), "ref50": ref50,
                           "tcp_ref": tcp_ref, "fidx": fidx, "excl": excl, "margin": margin,
                           "fingers": S.finger_values(spec, Pfull, 0),
                           "ours": OurWorlds(z, outside)})

    def clearances(c, q50):
        sph = fk_spheres(robot, spec, q50, c["Pfull"])
        r = sph[0, :, 3]
        pts = W.transform_points(c["T"], sph[..., :3].reshape(-1, 3))
        L = c["ours"].layers(pts)
        d_b0, _ = c["ours"].b0(L)
        d_lte, _ = c["ours"].lte(L, np.tile(c["excl"], q50.shape[0]))
        cl_b0 = d_b0.reshape(q50.shape[0], -1) - r
        cl_lte = d_lte.reshape(q50.shape[0], -1) - r
        tp = c["z"].get("target_points")
        if np.isfinite(c["margin"]).any() and tp is not None and len(tp):
            d_obj = cKDTree(np.asarray(tp, np.float64)).query(pts)[0].reshape(q50.shape[0], -1)
            m = np.where(np.isfinite(c["margin"]), c["margin"], 0.0)
            cl_t = np.where(np.isfinite(c["margin"])[None, :], d_obj - r - m, np.inf)
            cl_lte = np.minimum(cl_lte, cl_t)
        tcp = W.transform_points(c["T"], sph[:, c["fidx"], :3].mean(1))
        return cl_b0, cl_lte, tcp

    results: dict[str, Any] = {}
    seed_costs: dict[str, Any] = {}
    for vname in args.variants:
        world, term_names = VARIANTS[vname]
        terms = build_terms(term_names, weights, rows, dev)
        t0 = time.perf_counter()
        solver = CuroboChunkSolver(spec, num_seeds=args.num_seeds, timing=args.timing,
                                   world=world, extra_terms=list(terms.values()))
        init_s = time.perf_counter() - t0
        rows_out = []
        warmup = None
        order = ([chunks[0]] if args.warmup and chunks else []) + chunks
        for n_iter, c in enumerate(order):
            is_warmup = bool(args.warmup) and n_iter == 0
            if "track" in terms:
                terms["track"].set_reference(c["ref50"])
            if "continuity" in terms:
                tail = c["ctx"]["continuity_tail"]
                terms["continuity"].set_previous(None if tail is None else tail[:rows])
            if "deflection_rate" in terms:
                terms["deflection_rate"].set_reference(c["ref50"])
                terms["deflection_rate"].set_history(c["ctx"]["deflection_history"])
            lte = {}
            if world == "lte":
                tp = c["z"].get("target_points")
                lte = dict(excl_curobo=_cu_order(c["excl"], perm),
                           target_margin_curobo=_cu_order(c["margin"], perm),
                           target_points=None if tp is None or not len(tp) else tp)
            winfo = solver.set_world(c["z"], c["T"], **lte)
            solver.set_fingers(c["fingers"])
            res = solver.solve(c["q_now"], c["z"]["reference_chunk"])
            if is_warmup:
                # first solve of a solver instance = CUDA-graph capture; its result can differ
                # from a warm re-solve of the same chunk (measured), so it is discarded
                warmup = {"episode": c["episode"], "chunk_index": c["chunk_index"],
                          "wall_ms": res["wall_ms"], "success": res["success"]}
                continue
            key = f"{c['episode']}_{c['chunk_index']}"
            breakdown = solver.optimizer_breakdown(c["q_now"], c["z"]["reference_chunk"])
            seed_costs.setdefault(vname, {})[key] = breakdown["seed"]
            failing = {k: v["n_steps_positive"] for k, v in res.get("curobo_constraints", {}).items()
                       if v["n_steps_positive"] > 0}
            row = {"episode": c["episode"], "chunk_index": c["chunk_index"], "world": winfo,
                   "success": res["success"], "failing_constraints": failing,
                   "wall_ms": res["wall_ms"], "solve_time_ms": res["solve_time_ms"],
                   "cold": res["cold"], "finite": res["finite"],
                   "goal_error_rad": res["goal_error_rad"],
                   "start_error_rad": res["start_error_rad"],
                   "continuity_overlap": (0 if c["ctx"]["continuity_tail"] is None
                                          else min(len(c["ctx"]["continuity_tail"]), rows)),
                   "deflection_boundary": c["ctx"]["deflection_history"] is not None,
                   "metrics_constraints_at_solution": res.get("curobo_costs_at_solution"),
                   "optimizer_objective_at_solution": breakdown["solution"]}
            if res["cspace"] is not None and res["finite"]:
                q50 = res["cspace"]
                cl_b0, cl_lte, tcp = clearances(c, q50)
                dt = np.linalg.norm(tcp - c["tcp_ref"], axis=1) * 1e3
                row.update({
                    "min_clearance_b0world_mm_exec8": float(cl_b0[:8].min() * 1e3),
                    "min_clearance_b0world_mm_all50": float(cl_b0.min() * 1e3),
                    "min_clearance_lteworld_mm_exec8": float(cl_lte[:8].min() * 1e3),
                    "min_clearance_lteworld_mm_all50": float(cl_lte.min() * 1e3),
                    "tcp_vs_reference_mm_mean_exec8": float(dt[:8].mean()),
                    "tcp_vs_reference_mm_max_all50": float(dt.max()),
                    "max_joint_dev_vs_reference_rad": float(np.abs(q50 - c["ref50"]).max()),
                    "to_terms_at_solution": term_values_ref(q50, c["ref50"], c["ctx"], weights, rows),
                })
                c.setdefault("q", {})[vname] = q50
                tdir = out_dir / "trajectories" / vname
                tdir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(tdir / f"ep{c['episode']}_chunk{c['chunk_index']:05d}.npz",
                                    chunk=res["chunk"], cspace=q50,
                                    reference_chunk=c["z"]["reference_chunk"],
                                    refined_chunk=c["z"]["refined_chunk"])
            rows_out.append(row)
            print(f"[t63-smoke] {vname} ep{c['episode']} chunk {c['chunk_index']}: "
                  f"success={row['success']} wall {row['wall_ms']:.1f} ms "
                  f"clr b0w(8/50) {row.get('min_clearance_b0world_mm_exec8', float('nan')):.1f}/"
                  f"{row.get('min_clearance_b0world_mm_all50', float('nan')):.1f} "
                  f"tcp {row.get('tcp_vs_reference_mm_mean_exec8', float('nan')):.1f} mm "
                  f"fail {failing}", flush=True)
        warm = [r["wall_ms"] for r in rows_out if not r["cold"]]
        obj = {}
        for r in rows_out:
            for k, v in (r.get("optimizer_objective_at_solution") or {}).items():
                obj.setdefault(k, []).append(v)
        fail_reasons: dict[str, int] = {}
        for r in rows_out:
            if not r["success"]:
                for k in r["failing_constraints"]:
                    fail_reasons[k] = fail_reasons.get(k, 0) + 1
        ok = [r for r in rows_out if "min_clearance_b0world_mm_exec8" in r]
        results[vname] = {
            "world": world, "terms": list(terms), "settings": solver.settings_record(),
            "init_s": init_s, "warmup_solve": warmup,
            "summary": {
                "optimizer_objective_at_solution_median": {k: float(np.median(v))
                                                          for k, v in obj.items()},
                "n": len(rows_out), "n_success": int(sum(r["success"] for r in rows_out)),
                "failure_reasons_chunks": fail_reasons,
                "n_nonfinite": int(sum(not r["finite"] for r in rows_out)),
                "cold_wall_ms": [r["wall_ms"] for r in rows_out if r["cold"]],
                "warm_wall_ms_median": float(np.median(warm)) if warm else None,
                "warm_wall_ms_max": float(np.max(warm)) if warm else None,
                "warm_over_66p7ms": int(sum(w > 1000.0 / 15.0 for w in warm)),
                "min_clearance_b0world_mm_exec8_min": min(r["min_clearance_b0world_mm_exec8"]
                                                          for r in ok),
                "min_clearance_b0world_mm_all50_min": min(r["min_clearance_b0world_mm_all50"]
                                                          for r in ok),
                "n_chunks_b0world_exec8_below0": int(sum(r["min_clearance_b0world_mm_exec8"] < 0
                                                         for r in ok)),
                "n_chunks_b0world_all50_below0": int(sum(r["min_clearance_b0world_mm_all50"] < 0
                                                         for r in ok)),
                "min_clearance_lteworld_mm_exec8_min": min(r["min_clearance_lteworld_mm_exec8"]
                                                           for r in ok),
                "n_chunks_lteworld_exec8_below0": int(sum(r["min_clearance_lteworld_mm_exec8"] < 0
                                                          for r in ok)),
                "tcp_vs_reference_mm_mean_exec8_median": float(np.median(
                    [r["tcp_vs_reference_mm_mean_exec8"] for r in ok])),
                "max_joint_dev_vs_reference_rad_median": float(np.median(
                    [r["max_joint_dev_vs_reference_rad"] for r in ok])),
            },
            "chunks": rows_out}
        del solver
        torch.cuda.empty_cache()

    # how far each variant moved from B0 (same chunk) — the effect of the added terms
    base = args.variants[0]
    for vname in args.variants[1:]:
        diffs = [float(np.abs(c["q"][vname] - c["q"][base]).max()) for c in chunks
                 if base in c.get("q", {}) and vname in c.get("q", {})]
        results[vname]["summary"][f"max_joint_diff_vs_{base}_rad_median"] = (
            float(np.median(diffs)) if diffs else None)
        results[vname]["summary"][f"max_joint_diff_vs_{base}_rad_max"] = (
            float(np.max(diffs)) if diffs else None)
    return {"weights": wts, "term_rows": rows, "term_window": args.term_window,
            "optimizer_objective_at_seed": seed_costs, "variants": results}


# =================================================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True, help="export_inputs.py output dir")
    ap.add_argument("--out", required=True, help="output dir ($AG3S_RUN_DIR/impl/T63)")
    ap.add_argument("--stage", choices=("parity", "b0", "both", "parity_composed", "smoke",
                                        "step3", "parity_b0m"), default="both")
    ap.add_argument("--b0m-target-grid", default=None, metavar="VOXEL,PAD",
                    help="parity_b0m only: override the B0m target grid (default 0.0025,0.06)")
    ap.add_argument("--b0m-rules", default=None,
                    help="T63b: step3_eval.py --export-b0m-rules output dir (B0m / parity_b0m)")
    ap.add_argument("--lambda", dest="lambdas", default=",".join(f"{v:g}" for v in STEP3_LAMBDAS),
                    help="step3: multipliers of our terms (track, continuity, deflection-rate); "
                         "one solver per (variant, lambda); B0 has none")
    ap.add_argument("--only-no-free", action="store_true",
                    help="parity_composed: only the chunks recorded without esdf_free (the b0 rule "
                         "changed for them in step 3)")
    ap.add_argument("--no-objective-breakdown", dest="objective_breakdown", action="store_false",
                    help="step3: skip the per-term optimizer objective at the solution")
    ap.add_argument("--prefetch", action="store_true",
                    help="step3: read the next record in a background thread during a solve "
                         "(faster; inflates the measured solve time — not for timing runs)")
    ap.add_argument("--worlds", default="b0,lte", help="parity_composed: world modes")
    ap.add_argument("--variants", default="B0,B1,B2,B3,B4", help="smoke: variants (cumulative)")
    ap.add_argument("--term-window", choices=("chunk", "plan"), default="chunk",
                    help="rows the TO terms cover: chunk = all 50 rows cuRobo optimizes; plan = "
                         "the TO's plan window (--plan-rows)")
    ap.add_argument("--plan-rows", type=int, default=16)
    ap.add_argument("--no-warmup", dest="warmup", action="store_false",
                    help="smoke: do not discard a first (graph-capture) solve per variant")
    ap.add_argument("--contact", default=None,
                    help="export_inputs --contact-policy-from output dir (B4 / lte parity)")
    ap.add_argument("--b4-mask", choices=("recomputed", "record"), default="record",
                    help="target_field_exclude rows for B4: the record's (default since step 3, "
                         "user decision 3) or recomputed by fixed_input_ablation.contact_policy")
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--chunks", default="all", help="parity: `all` or comma list of chunk positions")
    ap.add_argument("--b0-chunks", default="0,20,40,60",
                    help="B0 smoke: chunk positions within each episode (`all` for every chunk)")
    ap.add_argument("--num-seeds", type=int, default=None, help="default: cuRobo's (4)")
    ap.add_argument("--timing", choices=("exact", "resample16"), default="exact",
                    help="exact = 21 knots, 15 Hz grid on the state grid; resample16 = cuRobo's "
                         "16 knots, states linearly resampled")
    ap.add_argument("--save-trajectories", action="store_true")
    ap.add_argument("--force", action="store_true", help="run B0 even if parity failed")
    args = ap.parse_args(argv)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec, manifest, data = load_inputs(pathlib.Path(args.inputs), args.episodes)
    meta = {"tool": TOOL, "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "argv": sys.argv, "inputs": str(pathlib.Path(args.inputs).resolve()),
            "episodes": list(data), "snapshot": manifest.get("snapshot")}
    try:
        import curobo
        meta["curobo_version"] = getattr(curobo, "__version__", None)
    except Exception:  # pragma: no cover
        pass

    parity_ok = True
    if args.stage in ("parity", "both"):
        res = run_parity(args, spec, data)
        parity_ok = res["summary"]["gate_pass"]
        (out / "parity.json").write_text(json.dumps(_jsonable({**meta, **res}), indent=1))
        print("[t63-parity] " + json.dumps(_jsonable(res["summary"])), flush=True)
    if args.stage in ("b0", "both"):
        if not parity_ok and not args.force:
            print("[t63] parity gate FAILED — B0 not run (task: stop after step 0)", flush=True)
            return 2
        res = run_b0(args, spec, data)
        (out / f"b0_smoke_{args.timing}.json").write_text(json.dumps(_jsonable({**meta, **res}), indent=1))
        print("[t63-b0] " + json.dumps(_jsonable(res["summary"])), flush=True)
    if args.stage == "parity_composed":
        args.worlds = [w for w in args.worlds.split(",") if w]
        if "lte" in args.worlds and not args.contact:
            raise SystemExit("--contact is required for the lte world")
        contact = load_contact(args.contact, list(data))
        res = run_parity_composed(args, spec, data, contact)
        (out / "parity_composed.json").write_text(json.dumps(_jsonable({**meta, **res}), indent=1))
        print("[t63-parity-composed] " + json.dumps(_jsonable(res["summary"])), flush=True)
        parity_ok = all(res["summary"][w]["gate_pass"] for w in args.worlds)
    if args.stage == "step3":
        args.variants = [v for v in args.variants.split(",") if v]
        args.lambdas = parse_lambdas(args.lambdas)
        if "B4" in args.variants and not args.contact:
            raise SystemExit("--contact is required for B4")
        contact = load_contact(args.contact, list(data))
        if "B0m" in args.variants and not args.b0m_rules:
            raise SystemExit("--b0m-rules is required for B0m (step3_eval.py --export-b0m-rules)")
        b0m_rules = load_b0m_rules(args.b0m_rules, list(data)) if "B0m" in args.variants else None
        wts = baseline_weights(manifest)
        res = run_step3_curobo(args, spec, data, contact, wts, b0m_rules)
        (out / "curobo_step3.json").write_text(json.dumps(_jsonable({**meta, **res}), indent=1))
        print(f"[t63-step3] {res['n_chunks']} chunks x {len(res['candidates'])} candidates in "
              f"{res['elapsed_s']:.0f} s", flush=True)
    if args.stage == "parity_b0m":
        if not args.b0m_rules:
            raise SystemExit("--b0m-rules is required for parity_b0m")
        res = run_parity_b0m(args, spec, data, load_b0m_rules(args.b0m_rules, list(data)))
        (out / "parity_b0m.json").write_text(json.dumps(_jsonable({**meta, **res}), indent=1))
        print("[t63-parity-b0m] " + json.dumps(_jsonable(res["summary"])), flush=True)
    if args.stage == "smoke":
        args.variants = [v for v in args.variants.split(",") if v]
        if "B4" in args.variants and not args.contact:
            raise SystemExit("--contact is required for B4")
        contact = load_contact(args.contact, list(data))
        wts = baseline_weights(manifest)
        res = run_smoke(args, spec, data, contact, wts)
        name = f"smoke_{args.timing}_{args.term_window}.json"
        (out / name).write_text(json.dumps(_jsonable({**meta, **res}), indent=1))
        print("[t63-smoke] " + json.dumps(_jsonable({v: r["summary"] for v, r in
                                                     res["variants"].items()})), flush=True)
    return 0 if parity_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
