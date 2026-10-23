"""Recorded-frame CPU smoke of the safety filters (PAPER-METHODS C): KNOWS, VLSA/AEGIS, multi-link CBF.

For each (policy call, method arm): rebuild the call's :class:`SceneObservation` from a ``--capture-geometry
--capture-segmentation`` LIBERO run, feed the **recorded action chunk** through ``filter``, check the contract
(:func:`validate_filter_result`) and write

  <out>/<stem>_c<call>_<arm>.json   provenance, intervened, |delta a|, per-step barrier / status, obstacles, timing
  <out>/<stem>_c<call>_<arm>.npz    actions_in, actions_out, intervened
  <out>/<stem>_c<call>.png          predicted EE path nominal vs filtered (top / side view) + |delta a| + min h per arm
  <out>/summary.json                one row per (call, arm)

Arms: ``knows``, ``aegis``, ``multilink_cbf`` (obstacles from GT segmentation, privileged) and
``aegis:detector_names`` / ``aegis:detector_all`` (Grounding DINO tiny on CPU; names = privileged LIBERO metadata,
all = generic prompt minus the target box). No pass/fail judgement is made here.

    source Baseline/paper_baselines/setup/paper_baselines_env.sh
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 .venv-baselines/bin/python -m Baseline.paper_baselines.methods.smoke_safety_filters \\
        --run 20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0 --stem libero_spatial_t0_e0 --calls 0 12
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from Baseline.paper_baselines.methods.interface import get_method, jsonable, validate_filter_result
from Baseline.paper_baselines.methods.safety_common import POS_GAIN_M, load_recorded_call, rollout_positions

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_OUT = PROJECT_ROOT / "outputs/paper_baselines_smoke/methods"
ARMS = {
    "knows": ("knows", {}),
    "aegis": ("aegis", {}),
    "multilink_cbf": ("multilink_cbf", {}),
    "aegis:detector_names": ("aegis", {"obstacle_source": "detector", "detector_prompt": "obstacle_names"}),
    "aegis:detector_all": ("aegis", {"obstacle_source": "detector", "detector_prompt": "all_objects"}),
}
C_NOMINAL, C_FILTERED, C_OBST, C_INK = "#52514e", "#2a78d6", "#eb6834", "#0b0b0b"


def min_h_steps(info: dict) -> list[float]:
    out = []
    for s in info.get("steps", []):
        if s.get("min_h") is not None:
            out.append(float(s["min_h"]))
        elif s.get("h"):
            out.append(float(min(s["h"])))
        else:
            out.append(float("nan"))
    return out


def obstacle_outlines(info: dict) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(name, centre (3,), Q (3, 3) squared-radius shape matrix) of each obstacle ellipsoid."""
    obs = info.get("obstacles", [])
    out = []
    if isinstance(obs, dict):  # KNOWS: {name: {"c", "Q"}}
        for k, v in obs.items():
            out.append((k, np.asarray(v["c"]), np.asarray(v["Q"])))
    else:
        for v in obs:
            R = np.asarray(v.get("R", np.eye(3)))
            out.append((v["name"], np.asarray(v["center"]), R @ np.diag(np.asarray(v["semi_axes"]) ** 2) @ R.T))
    return out


def projected_ellipse(ctr, Q, i, j, **kw):
    """Exact shadow of the ellipsoid on the (i, j) plane: shape matrix ``P Q P^T``."""
    from matplotlib.patches import Ellipse

    q2 = np.asarray(Q)[np.ix_([i, j], [i, j])]
    w, V = np.linalg.eigh(q2)
    ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    return Ellipse((ctr[i], ctr[j]), 2 * np.sqrt(max(w[1], 1e-10)), 2 * np.sqrt(max(w[0], 1e-10)), angle=ang, **kw)


def plot_call(path: pathlib.Path, title: str, obs, A: np.ndarray, results: dict, obstacles_from: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = list(results)
    fig, axes = plt.subplots(len(arms), 4, figsize=(17, 3.6 * len(arms)), squeeze=False)
    for r, arm in enumerate(arms):
        res = results[arm]
        p_nom = rollout_positions(obs.ee_pose, A, POS_GAIN_M)
        p_fil = rollout_positions(obs.ee_pose, res.actions, POS_GAIN_M)
        ax0 = axes[r, 0]
        ax0.imshow(obs.rgb["head"])
        ax0.set_title(f"{arm}: head camera (OpenCV frame)", fontsize=9)
        ax0.axis("off")
        for c, (i, j, lab) in enumerate([(0, 1, "top view (x-y, base)"), (0, 2, "side view (x-z, base)")]):
            ax = axes[r, 1 + c]
            for name, ctr, Q in obstacle_outlines(obstacles_from[arm]):
                ax.add_patch(projected_ellipse(ctr, Q, i, j, fill=False, ec=C_OBST, lw=1.2))
                ax.annotate(name[:14], (ctr[i], ctr[j]), fontsize=6, color="#52514e", ha="center")
            ax.plot(p_nom[:, i], p_nom[:, j], "--", color=C_NOMINAL, lw=2, label="nominal (recorded chunk)")
            ax.plot(p_fil[:, i], p_fil[:, j], "-", color=C_FILTERED, lw=2, label="filtered")
            ax.plot(p_nom[0, i], p_nom[0, j], "o", color=C_INK, ms=6)
            ax.set_aspect("equal", "datalim")
            ax.set_title(lab, fontsize=9)
            ax.grid(alpha=0.2)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="best")
        ax3 = axes[r, 3]
        d = np.linalg.norm(res.actions[:, :3] - A[:, :3], axis=1)
        k = np.arange(len(A))
        ax3.bar(k, d, color=C_FILTERED, width=0.6, label="|delta a_xyz| (action units)")
        ax3.set_xlabel("action index in chunk")
        ax3.set_ylabel("|delta a_xyz|")
        ax3.set_title(f"intervened {int(res.intervened.sum())}/{len(A)}", fontsize=9)
        ax3.grid(alpha=0.2, axis="y")
        mh = min_h_steps(res.info)
        for kk, h in enumerate(mh):
            ax3.annotate(f"h={h:.3f}", (kk, d[kk]), fontsize=6, rotation=90, va="bottom", ha="center", color="#52514e")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", default="20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0")
    p.add_argument("--stem", default="libero_spatial_t0_e0")
    p.add_argument("--calls", type=int, nargs="+", default=[0, 12])
    p.add_argument("--arms", nargs="+", default=["knows", "aegis", "multilink_cbf", "aegis:detector_names"],
                   choices=sorted(ARMS))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--no-figure", action="store_true")
    args = p.parse_args(argv)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    detector = None
    rows = []
    for call in args.calls:
        obs, A, meta = load_recorded_call(args.run, args.stem, call)
        results, infos = {}, {}
        for arm in args.arms:
            name, kw = ARMS[arm]
            if kw.get("obstacle_source") == "detector":
                if detector is None:
                    from Baseline.paper_baselines.providers.grounding_dino import GroundingDinoDetector

                    detector = GroundingDinoDetector("grounding-dino-tiny", device="cpu")
                kw = {**kw, "detector": detector}
            f = get_method(name, **kw)
            f.reset()
            t0 = time.perf_counter()
            res = f.filter(obs, A)
            ms = (time.perf_counter() - t0) * 1e3
            validate_filter_result(res, A)
            results[arm], infos[arm] = res, res.info
            da = np.abs(res.actions - A)
            row = {**meta, "arm": arm, "method": name, "ms_total": round(ms, 1), "T": int(len(A)),
                   "n_intervened": int(res.intervened.sum()), "intervened": res.intervened.astype(int).tolist(),
                   "max_abs_delta_xyz": float(da[:, :3].max()), "max_abs_delta_rot": float(da[:, 3:6].max()),
                   "gripper_unchanged": bool(np.array_equal(res.actions[:, 6], A[:, 6])),
                   "min_h_first": (min_h_steps(res.info) or [None])[0], "min_h_min": float(np.nanmin(min_h_steps(res.info)))
                   if min_h_steps(res.info) else None,
                   "n_obstacles": len(obstacle_outlines(res.info)), "contract": "ok"}
            rows.append(row)
            stem = f"{args.stem}_c{call}_{arm.replace(':', '-')}"
            (out / f"{stem}.json").write_text(json.dumps(jsonable({"row": row, "provenance": f.provenance(),
                                                                    "info": res.info}), indent=1))
            np.savez_compressed(out / f"{stem}.npz", actions_in=A, actions_out=res.actions, intervened=res.intervened)
            print(f"[{args.stem} c{call} t={meta['obs_index']}] {arm:22s} {ms:8.0f} ms  intervened "
                  f"{row['n_intervened']}/{row['T']}  max|da_xyz| {row['max_abs_delta_xyz']:.3f}  "
                  f"min_h {row['min_h_min'] if row['min_h_min'] is None else round(row['min_h_min'], 4)}  "
                  f"obstacles {row['n_obstacles']}", flush=True)
        if not args.no_figure:
            plot_call(out / f"{args.stem}_c{call}.png",
                      f"{args.stem} call {call} (obs_index {meta['obs_index']}): recorded pi0.5 chunk through each safety "
                      f"filter; EE path predicted with {POS_GAIN_M} m / action unit", obs, A, results, infos)
    (out / "summary.json").write_text(json.dumps(jsonable({"run": args.run, "stem": args.stem, "rows": rows}), indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
