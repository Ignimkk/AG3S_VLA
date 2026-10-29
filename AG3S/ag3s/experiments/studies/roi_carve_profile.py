"""AG3S-ROI-CARVE-GPU — where does the region-of-interest carve spend its time? (offline field replay, not closed-loop)

The recorded observations of a closed-loop run (policy record: per request the three depth images, K, T_base_cam and
qpos) are fed, request by request, into the server's own AG3S builder (`dynamic_field_replay.run_episode`'s field
stage) with the recorded server argv. The carve's region of interest is the one the server recorded for the same
request (`esdf_stats.roi_carve.boxes`), so the voxels judged are the replay TSDF's occupied voxels in the server's boxes.

Every part of `CuroboFieldBuilder._roi_carve` is timed with a CUDA synchronize on both sides (wall ms):
  occupied.meta      block metadata to the host + the block-vs-box test
  occupied.select    the occupied test and `nonzero` on the GPU
  occupied.d2h       the voxel indices (and nothing else) to the host
  occupied.host      voxel centres, the in-box filter (numpy)
  seen_through       the whole judgement (all cameras), of which
    footprint.filter the `minimum_filter` calls
  set_free           the write back
`--compare-gpu` also runs the GPU judgement (`esdf.roi_carve_gpu`) on the same TSDF state before the CPU one is applied
and records how many voxels the two decide differently.

  MUJOCO_GL=osmesa .venv-openpi-live/bin/python -m AG3S.ag3s.experiments.studies.roi_carve_profile \
      --kit <run>/verify/AG3S-TASK-DYNAMIC-MAP-E4 --configuration B_ROI_FPR --argv-file <argv.txt> --out <dir>
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib
import shlex
import time

import numpy as np

from AG3S.ag3s.experiments.studies import dynamic_field_replay as dfr


class Timers:
    def __init__(self):
        self.t: dict = {}
        self.active = False

    def add(self, key, ms):
        if self.active:
            self.t[key] = self.t.get(key, 0.0) + ms

    def pop(self):
        out, self.t = self.t, {}
        return out


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(timers, key, fn):
    def wrap(*a, **k):
        _sync()
        t0 = time.perf_counter()
        r = fn(*a, **k)
        _sync()
        timers.add(key, (time.perf_counter() - t0) * 1e3)
        return r
    return wrap


def occupied_voxels_timed(timers, itg, *, band_voxels=None, surface_only=False, boxes=None, with_sdf=False):
    """`CuroboFieldBuilder.occupied_voxels` with its four parts timed (same arithmetic; checked against it)."""
    import torch
    from AG3S.ag3s.fields import curobo_builder as cb

    band_voxels = cb.CUROBO_SEED_BAND_VOXELS if band_voxels is None else band_voxels
    assert not with_sdf and not surface_only
    empty = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3)))
    _sync()
    t0 = time.perf_counter()
    ts = itg.tsdf
    data = ts.data if hasattr(ts, "data") else ts
    bd = data.block_data
    n_alloc = int(data.num_allocated.detach().cpu().reshape(-1)[0])
    if n_alloc <= 0 or bd.dim() != 3 or bd.shape[1] <= 1:
        return empty
    bs = int(round(bd.shape[1] ** (1.0 / 3.0)))
    D, H, W = (int(v) for v in itg.grid_shape)
    vs = float(itg.voxel_size)
    origin = itg.origin.detach().cpu().numpy().reshape(-1)[:3].astype(np.float64)
    coords = data.block_coords.detach().cpu().numpy().reshape(-1, 3)[:n_alloc].astype(np.int64)
    slots = data.block_to_hash_slot.detach().cpu().numpy()[:n_alloc]
    live = np.flatnonzero(slots >= 0)
    grid = np.array([W, H, D], np.int64)
    offsets = ((grid + bs - 1) // bs) // 2
    if boxes is not None and live.size:
        blo = origin + ((coords[live] + offsets) * bs - 0.5 * grid).astype(np.float64) * vs
        bhi = blo + bs * vs
        keep = np.zeros(len(live), bool)
        for lo, hi in boxes:
            keep |= np.all((bhi >= np.asarray(lo)) & (blo <= np.asarray(hi)), axis=1)
        live = live[keep]
    t1 = time.perf_counter()
    timers.add("occupied.meta", (t1 - t0) * 1e3)
    if not live.size:
        return empty
    t_live = torch.as_tensor(live, device=bd.device, dtype=torch.long)
    w = bd[t_live, :, 1].float()
    s = bd[t_live, :, 0].float()
    band = float(band_voxels) * vs * w
    occ = (w > 0) & (s <= band)
    nz = torch.nonzero(occ, as_tuple=True)
    _sync()
    t2 = time.perf_counter()
    timers.add("occupied.select", (t2 - t1) * 1e3)
    bi, li = (t.detach().cpu().numpy() for t in nz)
    t3 = time.perf_counter()
    timers.add("occupied.d2h", (t3 - t2) * 1e3)
    if not len(bi):
        return empty
    loc = np.stack([li % bs, (li // bs) % bs, li // (bs * bs)], axis=1)
    g = (coords[live[bi]] + offsets) * bs + loc
    centres = origin + (g.astype(np.float64) + 0.5 - 0.5 * grid) * vs
    out = (live[bi], li, centres)
    if boxes is not None:
        inside = cb.in_boxes(centres, boxes)
        out = tuple(a[inside] for a in out)
    timers.add("occupied.host", (time.perf_counter() - t3) * 1e3)
    timers.add("occupied.n", float(len(out[0])))
    return out


def recorded_roi(chunk_file):
    with np.load(chunk_file, allow_pickle=True) as z:
        s = json.loads(str(z["summary_json"]))
    roi = (s.get("esdf_stats") or {}).get("roi_carve")
    roi = ast.literal_eval(roi) if isinstance(roi, str) else roi
    return roi, (s.get("profile_ms") or {})


def run_episode(argv, pdir, chunks, gt, timers, compare_gpu: bool, check_every: int = 10):
    import scipy.ndimage
    from AG3S.ag3s.fields import curobo_builder as cb
    from AG3S.trajopt import serve_safe

    args = serve_safe.build_parser().parse_args(argv)
    ag3s = serve_safe.ag3s_from_args(args, plan_horizon_steps=None, rows_per_step=None)
    cfg = ag3s.config
    if ag3s._esdf_builder is None:
        ag3s._esdf_builder = cb.CuroboFieldBuilder(cfg.esdf)
    builder = ag3s._esdf_builder
    B = cb.CuroboFieldBuilder
    orig = {"roi_boxes": cb.roi_boxes, "seen_through": cb.seen_through, "fp": cb._seen_through_footprint,
            "minf": scipy.ndimage.minimum_filter, "occ": B.__dict__["occupied_voxels"], "set_free": B.__dict__["_set_free"],
            "roi": B.__dict__["_roi_carve"], "gpu_cameras": cb.gpu_cameras}
    state = {"boxes": None, "frame": -1, "n_check": 0, "gpu_diff": []}
    orig_occ = orig["occ"].__func__

    def occ(itg, **k):
        if not timers.active:
            return orig_occ(itg, **k)
        out = occupied_voxels_timed(timers, itg, **k)
        if state["frame"] % check_every == 0:    # the timed copy must be the function it times
            ref = orig_occ(itg, **k)
            assert all(np.array_equal(a, b) for a, b in zip(out, ref)), "timed occupied_voxels differs"
            state["n_check"] += 1
        return out

    def roi_carve(self, itg, cameras, boxes, truncation, gpu_cams=None):
        if compare_gpu:   # the GPU judgement on the same state, before the CPU one writes (not timed)
            was, timers.active = timers.active, False
            cpu_pool, cpu_li, centres = orig_occ(itg, boxes=boxes)
            vs = float(itg.voxel_size)
            margin_cfg = getattr(self.config, "free_space_carve_margin", None)
            m = float(margin_cfg) if margin_cfg is not None else float(truncation) + vs
            hit_cpu, _ = orig["seen_through"](centres, cameras, m, depth_min=float(self.config.depth_min),
                                              depth_max=float(self.config.depth_max), footprint_voxel=vs,
                                              footprint_pad_px=int(self.config.roi_carve_footprint_pad_px),
                                              footprint_robot=str(self.config.roi_carve_footprint_robot))
            ts = itg.tsdf
            dev = (ts.data if hasattr(ts, "data") else ts).block_data.device
            gc = orig["gpu_cameras"](cameras, dev, float(self.config.depth_min), float(self.config.depth_max))
            g_pool, g_li, g_hit, _, _ = self._roi_judge_gpu(itg, gc, boxes, truncation)
            g_pool, g_li, g_hit = (t.cpu().numpy() for t in (g_pool, g_li, g_hit))
            a = set(zip(cpu_pool[hit_cpu].tolist(), cpu_li[hit_cpu].tolist()))
            b = set(zip(g_pool[g_hit].tolist(), g_li[g_hit].tolist()))
            state["gpu_diff"].append({"frame": state["frame"], "voxel_size": vs, "n_judged_cpu": int(len(cpu_pool)),
                                      "n_judged_gpu": int(len(g_pool)), "n_hit_cpu": len(a), "n_hit_gpu": len(b),
                                      "cpu_only": len(a - b), "gpu_only": len(b - a)})
            timers.active = was
        _sync()
        t0 = time.perf_counter()
        r = orig["roi"](self, itg, cameras, boxes, truncation, gpu_cams)
        _sync()
        timers.add("roi_carve_total", (time.perf_counter() - t0) * 1e3)
        return r

    cb.roi_boxes = lambda *a, **k: state["boxes"]
    cb.gpu_cameras = _timed(timers, "gpu_cameras", orig["gpu_cameras"])
    cb.seen_through = _timed(timers, "seen_through", orig["seen_through"])
    cb._seen_through_footprint = _timed(timers, "footprint.per_camera", orig["fp"])
    scipy.ndimage.minimum_filter = _timed(timers, "footprint.filter", orig["minf"])
    B.occupied_voxels = staticmethod(occ)
    B._set_free = staticmethod(_timed(timers, "set_free", orig["set_free"].__func__))
    B._roi_carve = roi_carve
    rows = []
    try:
        for j, chunk_file, _state in chunks:
            sp = pdir / f"step_{j:05d}.npz"
            if not sp.exists():
                continue
            roi, prof = recorded_roi(chunk_file)
            if not roi or not roi.get("boxes"):
                continue
            state["boxes"] = [(np.asarray(lo, np.float64), np.asarray(hi, np.float64)) for lo, hi in roi["boxes"]]
            state["frame"] = j
            step = np.load(sp)
            qpos = np.asarray(step["qpos"], np.float64)
            rs = dfr.robot_state_from_qpos(gt.m, qpos)
            obs = dfr.observations(step, rs, j)
            ag3s._frame_guard_radius = 0.0
            ag3s._frame_guard_centre = None
            ag3s._frame_filter_config = ag3s._config_with_guard_radius(0.0)
            ag3s._mask_guard_px, ag3s._frame_depth_masks = [], []
            ag3s._prepare_frame_filter()
            cams = ag3s._depth_cameras_from(obs, None, None, None, rs)
            timers.active = True
            _sync()
            t0 = time.perf_counter()
            builder.update(cams, target_points=None, exclude_target=False, frame_index=j, roi_points=[np.zeros((1, 3))])
            _sync()
            t_upd = (time.perf_counter() - t0) * 1e3
            timers.active = False
            t = timers.pop()
            rows.append({"request": j, "update_ms": round(t_upd, 3),
                         **{k: round(v, 3) for k, v in t.items()},
                         "recorded": {"roi_ms": roi.get("ms"), "esdf_ms": prof.get("esdf"),
                                      "n_roi_occupied": {k: (roi.get(k) or {}).get("n_roi_occupied")
                                                         for k in ("coarse_tsdf", "fine_tsdf")}}})
    finally:
        cb.roi_boxes, cb.seen_through, cb._seen_through_footprint = orig["roi_boxes"], orig["seen_through"], orig["fp"]
        cb.gpu_cameras = orig["gpu_cameras"]
        scipy.ndimage.minimum_filter = orig["minf"]
        B.occupied_voxels, B._set_free, B._roi_carve = orig["occ"], orig["set_free"], orig["roi"]
    return rows, state["n_check"], state["gpu_diff"]


def summarise(rows):
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and k != "request"})
    out = {}
    for k in keys:
        v = np.array([r.get(k, 0.0) for r in rows], np.float64)
        out[k] = {"n": int(len(v)), "median": round(float(np.median(v)), 3), "p95": round(float(np.percentile(v, 95)), 3),
                  "mean": round(float(v.mean()), 3)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kit", required=True)
    ap.add_argument("--configuration", required=True)
    ap.add_argument("--argv-file", required=True, help="the recorded server argv (one line)")
    ap.add_argument("--extra", default="", help="flags appended to the argv (e.g. --roi-carve-gpu)")
    ap.add_argument("--episodes", nargs="*", type=int, default=None)
    ap.add_argument("--compare-gpu", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kit = pathlib.Path(a.kit)
    argv = shlex.split(pathlib.Path(a.argv_file).read_text())
    argv = [t for i, t in enumerate(argv) if t not in ("--run-dir", "--record-constraints")
            and (i == 0 or argv[i - 1] not in ("--run-dir", "--record-constraints"))] + shlex.split(a.extra)
    model_xml = argv[argv.index("--model-xml") + 1]
    gt = dfr.GroundTruth(model_xml)
    eps = dfr.episodes_of(kit, a.configuration)
    sel = sorted(eps) if a.episodes is None else [e for e in a.episodes if e in eps]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    timers = Timers()
    all_rows, per_ep, diffs, n_check = [], {}, [], 0
    for ep in sel:
        pdir, chunks = eps[ep]
        t_ep = time.perf_counter()
        rows, nc, gd = run_episode(argv, pdir, chunks, gt, timers, a.compare_gpu)
        n_check += nc
        for r in rows:
            r["episode"] = ep
        for d in gd:
            d["episode"] = ep
        all_rows += rows
        diffs += gd
        per_ep[str(ep)] = summarise(rows) if rows else {}
        print(ep, "requests", len(rows), "roi_carve_total median",
              per_ep[str(ep)].get("roi_carve_total", {}).get("median"), "wall_s", round(time.perf_counter() - t_ep, 1),
              flush=True)
    (out / "rows.json").write_text(json.dumps(all_rows))
    if diffs:
        (out / "gpu_compare.json").write_text(json.dumps(diffs))
    summ = {"argv_extra": a.extra, "episodes": sel, "n_requests": len(all_rows), "n_timed_copy_checks": n_check,
            "summary": summarise(all_rows),
            "gpu_compare": None if not diffs else {
                "n_calls": len(diffs), "n_judged_cpu": int(sum(d["n_judged_cpu"] for d in diffs)),
                "n_judged_gpu": int(sum(d["n_judged_gpu"] for d in diffs)),
                "n_hit_cpu": int(sum(d["n_hit_cpu"] for d in diffs)), "n_hit_gpu": int(sum(d["n_hit_gpu"] for d in diffs)),
                "cpu_only": int(sum(d["cpu_only"] for d in diffs)), "gpu_only": int(sum(d["gpu_only"] for d in diffs))}}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
