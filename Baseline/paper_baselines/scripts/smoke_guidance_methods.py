#!/usr/bin/env python3
"""CPU smoke of the sampling-guidance methods on recorded LIBERO policy calls (PAPER-METHODS, Implementer D).

For each recorded call (SL3 dev run, default episode 0 calls 1 and 2):

1. ``load_recorded_call`` -> ``SceneObservation`` (wire depth, K, T_base_cam; GT masks as *privileged* target /
   robot / ignore masks) + the recorded raw action chunk ``(10, 7)``;
2. the chunk is normalized with the pi05_libero norm stats and used as the surrogate policy's target
   (``core.surrogate_sample``: the pi0 Euler loop with the straight-flow velocity towards that chunk -- the frozen
   network is not loaded on CPU);
3. every method's ``guidance_fn`` runs inside that jitted loop; the output chunk is compared with the unguided one.

Two obstacle variants per call: ``scene`` (depth-lifted obstacles: every pixel minus target / robot / table /
destination) and ``injected`` (a privileged 3 cm sphere placed 1 cm beside the recorded EE path, the synthetic
repeller check on a real chunk).

Writes ``outputs/paper_baselines_smoke/methods/guidance/summary.json`` + ``guidance_smoke_<episode>_c<call>.png``.

Run::

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \\
        .venv-openpi-live/bin/python Baseline/paper_baselines/scripts/smoke_guidance_methods.py
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from Baseline.paper_baselines.methods.guidance_common.core import (chunk_to_model_space, load_norm_stats,
                                                                   model_space_to_chunk, surrogate_sample)
from Baseline.paper_baselines.methods.guidance_common.evaluation import SCENE_KW, injected_obs, path_metrics
from Baseline.paper_baselines.methods.guidance_common.recorded import SL3_DEV_RUN, load_recorded_call
from Baseline.paper_baselines.methods.guidance_common.scene import build_scene
from Baseline.paper_baselines.methods.interface import get_method, jsonable

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
SURROGATES = ("tracking", "constant")
METHODS = ("omniguide", "constrained_flow_matching", "lan_o3dp_guidance_adapted", "nudge")
DEFAULT_NORM = "checkpoints/panda/pi05_libero/assets/physical-intelligence/libero/norm_stats.json"


def storage_root() -> pathlib.Path:
    return pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", PROJECT_ROOT.parent / "AG3S_VLA-storage"))


def run_call(episode, call, norm, seed, out_dir, make_fig=True) -> dict:
    import jax

    obs, chunk, meta = load_recorded_call(SL3_DEV_RUN, episode, call)
    H = chunk.shape[0]
    x = chunk_to_model_space(chunk, norm)
    noise = np.random.default_rng(seed).standard_normal(x.shape).astype(np.float32)
    base = np.asarray(jax.jit(lambda: surrogate_sample(x, noise))())
    raw_base = model_space_to_chunk(base, norm)
    rec = {"meta": meta, "H": H, "surrogate_reproduction_err_raw": float(np.abs(raw_base - chunk).max()),
           "variants": {}}

    scene_obs = build_scene(obs, **SCENE_KW)
    obs_inj, inj_c, k = injected_obs(obs, scene_obs, chunk)
    variants = {"scene": obs, "injected": obs_inj}
    figdata = {}
    for vname, o in variants.items():
        vrec = {}
        for name in METHODS:
            m = get_method(name, scene_kwargs=SCENE_KW)
            t0 = time.perf_counter()
            fn = m.make_guidance_fn(o, norm, noise=noise)
            scene = m.last_scene
            t1 = time.perf_counter()
            res = {}
            for kind in SURROGATES:
                out = np.asarray(jax.jit(lambda: surrogate_sample(x, noise, guidance_fn=fn, kind=kind))())
                raw_g = model_space_to_chunk(out, norm)
                d = np.abs(raw_g - raw_base)[:, :3]
                mg = path_metrics(raw_g, scene, H)
                res[kind] = {"finite": bool(np.all(np.isfinite(out))), "intervened_rows": (d.max(1) > 1e-4).tolist(),
                             "max_abs_delta_raw_xyz": float(d.max()), "guided_chunk_raw": raw_g.tolist(),
                             "ee_min_sdf_guided_m": mg["ee_min_sdf_m"], "hand_min_sdf_guided_m": mg["hand_min_sdf_m"],
                             "path_guided": mg["path_base"]}
            t2 = time.perf_counter()
            out2 = np.asarray(jax.jit(lambda: surrogate_sample(x, noise, guidance_fn=fn))())  # recompiles (closure)
            mb = path_metrics(raw_base, scene, H)
            vrec[name] = {"scene_build_s": t1 - t0, "jit_and_run_s_both_surrogates": t2 - t1,
                          "deterministic": bool(np.array_equal(model_space_to_chunk(out2, norm), res["tracking"]["guided_chunk_raw"])),
                          "ee_min_sdf_base_m": mb["ee_min_sdf_m"], "hand_min_sdf_base_m": mb["hand_min_sdf_m"],
                          "surrogates": res, "provenance": m.provenance(),
                          "ctx_flags": {k2: v for k2, v in (m.last_ctx or {}).items() if isinstance(v, (bool, int, float, str))}}
            figdata[(vname, name)] = (np.asarray(mb["path_base"]), np.asarray(res["tracking"]["path_guided"]),
                                      np.asarray(res["constant"]["path_guided"]), scene)
            f = lambda v: "None" if v is None else f"{v:+.4f}"  # noqa: E731
            print(f"[{meta['episode']} c{call}] {vname:8s} {name:27s} "
                  f"ee_sdf {f(mb['ee_min_sdf_m'])} -> trk {f(res['tracking']['ee_min_sdf_guided_m'])} / "
                  f"const {f(res['constant']['ee_min_sdf_guided_m'])} | hand_sdf {f(mb['hand_min_sdf_m'])} -> "
                  f"trk {f(res['tracking']['hand_min_sdf_guided_m'])} / const {f(res['constant']['hand_min_sdf_guided_m'])} | "
                  f"max|da| trk {res['tracking']['max_abs_delta_raw_xyz']:.3f} const {res['constant']['max_abs_delta_raw_xyz']:.3f}",
                  flush=True)
        rec["variants"][vname] = vrec
    rec["scene_info"] = jsonable(scene_obs.info)
    rec["injected_sphere"] = {"centre": inj_c.tolist(), "radius": 0.03, "near_row": k}
    if make_fig:
        rec["figure"] = str(plot(figdata, meta, out_dir / f"guidance_smoke_{meta['episode']}_c{call}.png"))
    return rec


def calibrate_nudge_kappa(norm, episodes=range(10), calls=range(1, 21, 3)) -> dict:
    """The paper's kappa: median non-zero l_inf gradient (normalized coords) over a calibration set of calls."""
    import jax.numpy as jnp

    vals = {"scene": [], "injected": []}
    for ep in episodes:
        for call in calls:
            try:
                obs, chunk, _ = load_recorded_call(SL3_DEV_RUN, ep, call)
            except (IndexError, KeyError):
                break
            x = jnp.asarray(chunk_to_model_space(chunk, norm)[0])
            for kind in ("scene", "injected"):
                m = get_method("nudge", scene_kwargs=SCENE_KW)
                o = obs if kind == "scene" else injected_obs(obs, build_scene(obs, **SCENE_KW), chunk)[0]
                m.make_guidance_fn(o, norm)
                v = float(np.abs(np.asarray(m.gradient(x, m.last_ctx, m.path_for(chunk.shape[0], m.last_scene)))).max())
                if v > 0:
                    vals[kind].append(v)
    allv = vals["scene"] + vals["injected"]
    return {"kappa_median": float(np.median(allv)) if allv else None, "n": len(allv),
            "per_kind_median": {k: (float(np.median(v)) if v else None) for k, v in vals.items()}, "values": vals}


def plot(figdata, meta, path: pathlib.Path) -> pathlib.Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = sorted({v for v, _ in figdata}, reverse=True)
    fig, axes = plt.subplots(len(variants), len(METHODS), figsize=(4.2 * len(METHODS), 4.0 * len(variants)), squeeze=False)
    for i, v in enumerate(variants):
        for j, name in enumerate(METHODS):
            ax = axes[i, j]
            pb, pg, pc, scene = figdata[(v, name)]
            o = scene.obstacles
            ctr = pb.mean(0)
            if len(o.points):
                sel = np.linalg.norm(o.points[:, :2] - ctr[None, :2], axis=1) < 0.25
                ax.scatter(o.points[sel, 0], o.points[sel, 1], s=4, c="0.6", label="obstacle pts")
            for c, r in zip(o.sphere_c, o.sphere_r):
                ax.add_patch(plt.Circle(c[:2], r, color="tab:red", alpha=0.35, label="obstacle sphere"))
            if scene.target_points is not None and len(scene.target_points):
                tp = scene.target_points
                ax.scatter(tp[::7, 0], tp[::7, 1], s=3, c="tab:green", alpha=0.5, label="target pts")
            p0 = scene.ee_pos
            ax.plot(*np.vstack([p0, pb])[:, :2].T, "-o", ms=3, c="0.2", label="unguided")
            ax.plot(*np.vstack([p0, pg])[:, :2].T, "-o", ms=3, c="tab:blue", label="guided (tracking surrogate)")
            ax.plot(*np.vstack([p0, pc])[:, :2].T, "--", ms=2, c="tab:orange", label="guided (constant surrogate)")
            ax.set_xlim(ctr[0] - 0.15, ctr[0] + 0.15)
            ax.set_ylim(ctr[1] - 0.15, ctr[1] + 0.15)
            ax.set_aspect("equal")
            ax.set_title(f"{name}\n[{v}]", fontsize=9)
            ax.set_xlabel("x base (m)")
            ax.set_ylabel("y base (m)")
            if i == 0 and j == 0:
                h, l = ax.get_legend_handles_labels()
                uniq = dict(zip(l, h))
                ax.legend(uniq.values(), uniq.keys(), fontsize=7, loc="best")
    fig.suptitle(f"Sampling guidance smoke (surrogate flow, CPU): {meta['episode']} call {meta['call']} "
                 f"(obs {meta['obs_index']}), EE path top view, OSC response decode", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--calls", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--norm-stats", default=None)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs/paper_baselines_smoke/methods/guidance"))
    ap.add_argument("--no-fig", action="store_true")
    ap.add_argument("--calibrate-kappa", action="store_true", help="recompute NUDGE kappa (episodes 0-9, calls 1,4,..,19)")
    a = ap.parse_args(argv)
    norm_path = pathlib.Path(a.norm_stats) if a.norm_stats else storage_root() / DEFAULT_NORM
    norm = load_norm_stats(str(norm_path))
    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.calibrate_kappa:
        k = calibrate_nudge_kappa(norm)
        (out_dir / "nudge_kappa_calibration.json").write_text(json.dumps(k, indent=1))
        print(json.dumps({kk: v for kk, v in k.items() if kk != "values"}))
        return 0
    recs = [run_call(a.episode, c, norm, a.seed, out_dir, make_fig=not a.no_fig) for c in a.calls]
    summary = {"what": "PAPER-METHODS D guidance smoke (surrogate flow, CPU)", "norm_stats": str(norm_path),
               "methods": list(METHODS), "scene_kwargs": SCENE_KW, "calls": recs,
               "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=1))
    print("wrote", out_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
