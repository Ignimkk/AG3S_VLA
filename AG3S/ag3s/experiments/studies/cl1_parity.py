"""IFG-CL1 — dev-split parity of the serving attention-surface extent and render mask with the offline LOCK arm.

Two checks, CPU only (MuJoCo osmesa render for `--render`), writing one JSON row per frame:

* **extent** — `runtime.attention_surface.select_target_surface` on server-shaped inputs (per-camera 16×16 noun L10H7
  maps from the dump `derived.npz`, replay depth / K / T, the R6 render mask, *dummy* GT) vs the locked chain
  (`ifg_graph.build_geometry` with the real GT + `ifg_round_ns.eligible(…, True)` + `g_and_top1`) — node IoU (node ids
  matched by position), seed patches, node count vs the stored `rounds/R6/rows` `top1-only-ns+rm:N` `pred_voxels`,
  and the share of re-lifted code-1 pixels that fall in a selected node's voxel. Per-stage ms.
* **render** (`--render`) — `runtime.robot_render_mask.RobotRenderMask` from the replay `qpos` reduced to what a
  request carries (20-joint robot_state + finger openings) vs the stored offline `rounds/R6/robot_mask`: differing
  pixels per camera, render ms.

  MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv-ag3s/bin/python -m \
      AG3S.ag3s.experiments.studies.cl1_parity --run $RUN --out $AG3S_RUN_DIR/cl1_parity [--render] [--every N]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np


def extent_row(run: pathlib.Path, ref) -> dict:
    from AG3S.ag3s.experiments.studies import ifg_graph as G
    from AG3S.ag3s.experiments.studies import ifg_reader as R
    from AG3S.ag3s.experiments.studies import ifg_round_ns as RNS
    from AG3S.ag3s.runtime.attention_surface import CAMERA_ORDER, SurfaceView, select_target_surface
    from AG3S.ag3s.runtime.oracle_segmentation import backproject_labels

    fid = ref.frame_id
    cfg = G.GraphConfig()
    fin = R.load_frame(ref, features=())
    with np.load(run / "s0/replay" / f"{fid}.npz") as z:
        depth, K, T = z["depth"], z["K"], z["T_base_cam"]
    with np.load(run / "rounds/R6/robot_mask" / f"{fid}.npz") as z:
        rm = z["robot_mask"]
    with np.load(run / "s0/labels" / f"{fid}.npz") as z:
        cm, cl = z["class_map"], [str(c) for c in z["classes"]]
    geo = G.build_geometry(depth, K, T, rm, cm, cl, cfg)
    res = RNS.g_and_top1(geo, cfg, fin.maps["N"], RNS.eligible(geo, cfg, True))
    off, seeds = set(), None
    if res is not None:
        sub, six, m, mt, info = res
        lut = {tuple(p): i for i, p in enumerate(geo.pos)}
        off = {lut[tuple(p)] for p in sub.pos[np.isfinite(mt) & (mt >= 0.5)]}
        seeds = [int(s) for s in six]
    row = next(json.loads(line) for line in open(run / "rounds/R6/rows" / f"{fid}.jsonl")
               if '"top1-only-ns+rm:N"' in line)
    nmap = np.asarray(fin.maps["N"])
    views = [SurfaceView(c, depth[i], K[i], T[i], rm[i], nmap[256 * i:256 * (i + 1)].reshape(16, 16))
             for i, c in enumerate(CAMERA_ORDER)]
    t = time.perf_counter()
    r = select_target_surface(views)
    ms = (time.perf_counter() - t) * 1e3
    new = set(r.selected_nodes.tolist())
    inside = None
    if r.codes is not None:
        P = np.vstack([backproject_labels(depth[i], K[i], T[i], (r.codes[c] == 1).astype(np.uint8))[0]
                       for i, c in enumerate(CAMERA_ORDER)])
        allp = np.concatenate([G._cam_pixels(depth[i], K[i], T[i], rm[i], i, cfg.range_max)[0] for i in range(3)])
        kmin = np.floor(allp / cfg.voxel).astype(np.int64).min(0)
        kp = {tuple(x) for x in (np.floor(P / cfg.voxel).astype(np.int64) - kmin).tolist()}
        ks = {tuple(x) for x in (np.floor(geo.pos[sorted(new)] / cfg.voxel).astype(np.int64) - kmin).tolist()}
        inside = len(kp & ks) / max(len(kp), 1)
    union = off | new
    return {"frame_id": fid, "moment": ref.moment, "n_offline": len(off), "n_server": len(new),
            "node_iou": 1.0 if not union else len(off & new) / len(union),
            "row_pred_voxels": row.get("pred_voxels"), "seeds_offline": seeds,
            "seeds_server": r.record.get("seed_patches"), "why": r.record.get("why"),
            "code1_voxels_in_selected": inside, "ms": round(ms, 3), "ms_parts": r.record["ms"]}


def render_row(run: pathlib.Path, ref, renderer, model_xml: str) -> dict:
    import mujoco

    from AG3S.ag3s.robot_models import DEFAULT_RBY1_JOINTS
    from AG3S.ag3s.robot_models.gripper_state import openings_from_mujoco
    from AG3S.ag3s.runtime.attention_surface import CAMERA_ORDER

    fid = ref.frame_id
    m = renderer._ref_model if hasattr(renderer, "_ref_model") else mujoco.MjModel.from_xml_path(model_xml)
    renderer._ref_model = m
    d = mujoco.MjData(m)
    with np.load(run / "s0/replay" / f"{fid}.npz") as z:
        depth, K, T, qpos = z["depth"], z["K"], z["T_base_cam"], z["qpos"]
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in DEFAULT_RBY1_JOINTS]
    views = [{"name": n, "depth_m": depth[i], "K": K[i], "T_base_cam": T[i], "robot_state": qpos[adr]}
             for i, n in enumerate(CAMERA_ORDER)]
    masks, rec = renderer.masks(views, openings_from_mujoco(m, d, renderer.convention))
    with np.load(run / "rounds/R6/robot_mask" / f"{fid}.npz") as z:
        r6 = z["robot_mask"]
    return {"frame_id": fid, "moment": ref.moment,
            "diff_px": [int((masks[i] != r6[i]).sum()) for i in range(3)],
            "r6_px": [int(r6[i].sum()) for i in range(3)], "render_ms": rec["ms"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--model-xml", default=None)
    a = ap.parse_args()
    from AG3S.ag3s.experiments.studies import ifg_reader as R

    a.out.mkdir(parents=True, exist_ok=True)
    refs = [r for r in R.list_frames() if (a.run / "s0/replay" / f"{r.frame_id}.npz").exists()][::a.every]
    with open(a.out / "extent.jsonl", "w") as f:
        for ref in refs:
            row = extent_row(a.run, ref)
            f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    if a.render:
        from AG3S.ag3s.runtime.robot_render_mask import RobotRenderMask

        xml = a.model_xml or str(pathlib.Path(__file__).resolve().parents[4]
                                 / "robots/rby1/description/models/rby1a/mujoco/model_transport.xml")
        r = RobotRenderMask(xml)
        with open(a.out / "render.jsonl", "w") as f:
            for ref in refs:
                row = render_row(a.run, ref, r, xml)
                f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
        r.close()


if __name__ == "__main__":
    main()
