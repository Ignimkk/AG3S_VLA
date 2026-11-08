#!/usr/bin/env python
"""What the AG3S perception / TO module "sees" at selected calls of a recorded LIBERO episode.

One PNG per selected call (2 x 3 panels) + `index.json` next to them:

  (1) agentview RGB at the call's observation      (2) wrist RGB at the same observation
  (3) top view (x-y, base frame): fused cloud by role (target red / support grey / other blue)
      + robot spheres coloured by their min clearance over the executed steps of this chunk
  (4) fine ESDF, horizontal slice at gripper height (diverging, zero contour) + target-free-removed voxels
  (5) the same slice of the target-free tier (the target carved out)
  (6) side view (x-z): cloud + spheres + support plane(s) + the slice height

Usage:
  .venv-ag3s/bin/python experiments/segmentation/gtto_feature_scene.py \
      --run '*gtto-gt14obs-libero9s10-*' --stem libero_spatial_t4_e0 --auto 4 --out <dir>
  ... --obs 60,75,100            (obs indices; the call whose `obs_index` equals each value)

CPU only, one process, one chunk npz in memory at a time. Reads only; writes only under --out.

---------------------------------------------------------------------------------------------------
Record conventions VERIFIED for this script (2026-11-08, run 20261107T200934_..._gtto-gt14obs-libero9s10-...):

* Writer: `AG3S/ag3s/experiments/sources/constraint_record.py::ConstraintRecordWriter`, called from
  `AG3S/trajopt/safe_policy.py::_record`.
* Frame: all points / centres are in the robot base frame (MuJoCo body `robot0_base`), metres
  (`frame_record.py` manifest "frames_of_reference"; `<stem>_geom.npz` conventions_json "base_frame").
* `esdf_origin` (and `esdf_coarse_origin`, `esdf_free_origin`, `target_free_trace_origin`) is the CENTRE of voxel
  (0,0,0), NOT the min corner (`frame_record.py`: "centre of voxel (0,0,0), not the grid corner";
  `constraint_record.py::_write_exclusion_trace`: "voxel centre = origin + ijk * voxel_size").
* Index order: `esdf_distance[i, j, k]` with (i, j, k) = (x, y, z) (`curobo_builder.oracle_candidates` builds
  ijk with meshgrid indexing="ij" over x, y, z and indexes the grid as grid[i, j, k]). Checked empirically on
  chunk 91: the fine ESDF at round((target_points - origin)/vs) has median 0.0 (surface seeds) with (x,y,z) order
  vs 0.040 m with the axes reversed; `target_free_removed_ijk` voxels all have fine-ESDF value 0 (seeds).
* ESDF sign: negative = inside a surface, positive = free (manifest). Values are float16 metres.
* Tiers (`ConstraintRecordWriter.field_layers`): `esdf_*` = finest layer (5 mm, 128^3, moving window),
  `esdf_coarse_*` = coarse layer (20 mm), `esdf_free_*` = target-free layer on the SAME grid as the fine one.
  The target-free tier and the `target_free_*_ijk` trace are absent on calls with no target (status no_target,
  e.g. while the object is held: then the held object is `attached_query_spheres` (K,4) / `attached_query_centres`
  (H,K,3), base frame).
* `clearance (H, 170)`: `cs.esdf.distance(centres along the REFINED plan) - sphere_radius - esdf_margin`, per plan
  step x robot sphere (`safe_policy._record`). `sphere_centres (170,3)` = step 0 of that plan. Held spheres are not
  among the 170 columns.
* Chunk <-> call mapping: chunks of one server run are numbered globally (`chunk_index`), `t_step` restarts at 0 per
  episode and equals the call's `seq` in `<stem>_calls.jsonl` (NOT the env `obs_index`). Segments (t_step resets)
  follow `episodes.jsonl` order; checked by `summary.to.solve_ms == calls[seq].to.solve_ms` and
  `summary.exec_feedback.t_step_start == calls[seq-1].obs_index` (done again for every rendered call below).
* Executed steps of call `seq`: env steps whose episode `chunk_index == seq` (non-wait); cross-checked against
  `calls[seq+1].feedback_server.n_exec`.
* Images: `obs_*_image` are robosuite 'opengl' (row 0 = bottom). Default display `[::-1, ::-1]` = the policy view
  (pi05 client transform; left-right mirrored w.r.t. the true camera); `--image-view opencv` shows `[::-1]`
  (true camera orientation).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Rectangle  # noqa: E402

BENCH = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
HAND_LINKS = ("panda_hand", "panda_leftfinger", "panda_rightfinger")
ESDF_CLIP_M = 0.10          # colour range of the ESDF slices (+-)
C_TARGET, C_SUPPORT, C_OTHER = "#d62728", "#9a9a9a", "#1f5fbf"


# --------------------------------------------------------------------------------------------- I/O
def resolve_run(spec: str) -> pathlib.Path:
    p = pathlib.Path(spec)
    if p.is_absolute() and p.is_dir():
        return p
    hits = sorted(d for d in BENCH.glob(spec) if d.is_dir())
    if not hits:
        sys.exit(f"no run matches {spec!r} under {BENCH}")
    if len(hits) > 1:
        sys.exit("several runs match; pass a narrower --run:\n  " + "\n  ".join(h.name for h in hits))
    return hits[0]


def load_calls(run: pathlib.Path, stem: str) -> list[dict]:
    path = run / "episodes" / f"{stem}_calls.jsonl"
    calls = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for i, c in enumerate(calls):
        if int(c["seq"]) != i:
            raise ValueError(f"{path}: line {i} has seq {c['seq']}")
    return calls


def chunk_scalar(path: str, key: str):
    with np.load(path) as z:
        return z[key].item() if key in z.files else None


def chunk_summary(path: str) -> dict:
    with np.load(path) as z:
        return json.loads(str(z["summary_json"]))


def episode_stems(run: pathlib.Path) -> list[str]:
    out = []
    for line in (run / "episodes.jsonl").read_text().splitlines():
        if line.strip():
            o = json.loads(line)
            out.append(pathlib.Path(o["npz"]).stem if o.get("npz") else
                       f"{o['suite']}_t{o['task_id']}_e{o['episode_idx']}")
    return out


def segments(run: pathlib.Path) -> list[list[str]]:
    """Chunk files of every server run dir, split where `t_step` restarts (one segment = one episode)."""
    segs: list[list[str]] = []
    for rdir in sorted((run / "server_constraints").glob("run_*")):
        prev = None
        for f in sorted(glob.glob(str(rdir / "chunk_*.npz"))):
            t = int(chunk_scalar(f, "t_step"))
            if prev is None or t <= prev:
                segs.append([])
            segs[-1].append(f)
            prev = t
    return segs


def _solve_ms_match(chunk: str, call: dict) -> bool:
    a = (chunk_summary(chunk).get("to") or {}).get("solve_ms")
    b = (call.get("to") or {}).get("solve_ms")
    return a is not None and b is not None and abs(float(a) - float(b)) < 2e-3


def map_chunks(run: pathlib.Path, stem: str, calls: list[dict]) -> tuple[list[str], str]:
    segs = segments(run)
    stems = episode_stems(run)
    how = ""
    if len(segs) == len(stems) and stem in stems:
        seg = segs[stems.index(stem)]
        if len(seg) == len(calls) and _solve_ms_match(seg[0], calls[0]):
            return seg, "episodes.jsonl order + seq0 solve_ms match"
        how = "episodes.jsonl order did not verify; "
    for seg in segs:
        if len(seg) == len(calls) and _solve_ms_match(seg[0], calls[0]) and _solve_ms_match(seg[-1], calls[-1]):
            return seg, how + "searched: length + first/last solve_ms match"
    sys.exit(f"could not map server chunks to {stem} ({len(segs)} segments, {len(calls)} calls)")


# --------------------------------------------------------------------------------------------- selection
def pick_calls(calls: list[dict], obs: str | None, auto: int | None) -> list[int]:
    obs_idx = [int(c["obs_index"]) for c in calls]
    if obs:
        out = []
        for v in (int(x) for x in obs.split(",") if x.strip()):
            i = int(np.argmin([abs(o - v) for o in obs_idx]))
            if obs_idx[i] != v:
                print(f"[warn] no call at obs {v}; using nearest call seq {i} (obs {obs_idx[i]})")
            out.append(i)
        return sorted(set(out))
    n = max(1, int(auto or 4))
    want = {int(round(x)) for x in np.linspace(0, len(calls) - 1, n)}
    held = [i for i, c in enumerate(calls) if (c.get("grasp") or {}).get("state") == "held"]
    if held and held[0] not in want:
        # replace the evenly spaced pick nearest to the first held call (keeps N)
        nearest = min(want - {0, len(calls) - 1} or want, key=lambda i: abs(i - held[0]))
        want.discard(nearest)
        want.add(held[0])
    return sorted(want)


# --------------------------------------------------------------------------------------------- helpers
def to_img(raw: np.ndarray, view: str) -> np.ndarray:
    return raw[::-1, ::-1] if view == "policy" else raw[::-1]


def grid_meta(z, prefix: str):
    if prefix + "distance" not in z.files:
        return None
    return {"o": np.asarray(z[prefix + "origin"], np.float64).reshape(3),
            "vs": float(z[prefix + "voxel_size"]),
            "shape": tuple(int(v) for v in np.asarray(z[prefix + "shape"]).reshape(3))}


def slice_k(meta: dict, zh: float) -> tuple[int, bool]:
    k = int(np.rint((zh - meta["o"][2]) / meta["vs"]))   # origin = centre of voxel 0
    kc = int(np.clip(k, 0, meta["shape"][2] - 1))
    return kc, kc != k


def extent_xy(meta: dict):
    o, vs, (nx, ny, _) = meta["o"], meta["vs"], meta["shape"]
    return [o[0] - vs / 2, o[0] + (nx - 0.5) * vs, o[1] - vs / 2, o[1] + (ny - 0.5) * vs]


def clearance_colour(c_m: float) -> str:
    if not np.isfinite(c_m):
        return "#7f7f7f"
    if c_m < 0.0:
        return "#e00000"
    if c_m < 0.010:
        return "#ff9900"
    return "#2ca02c"


def draw_spheres(ax, centres2d, radii, colours, order, dashed=None, lw=0.8, alpha=0.18):
    patches_e = [Circle(tuple(centres2d[i]), radii[i]) for i in order]
    cols = [colours[i] for i in order]
    ls = [((0, (3, 2)) if dashed is not None and dashed[i] else "solid") for i in order]
    pc = PatchCollection(patches_e, facecolors=cols, edgecolors=cols, linewidths=lw, linestyles=ls, alpha=None)
    pc.set_facecolor([matplotlib.colors.to_rgba(c, alpha) for c in cols])
    pc.set_edgecolor([matplotlib.colors.to_rgba(c, 0.95) for c in cols])
    ax.add_collection(pc)


def esdf_slice_panel(ax, fig, grid, meta, k, title, removed_k=None, ref_grid=None, spheres=None, held=None,
                     tp=None, crop=None):
    sl = np.asarray(grid[:, :, k], np.float32).T          # rows = y, cols = x
    ext = extent_xy(meta)
    im = ax.imshow(sl, origin="lower", extent=ext, cmap="RdBu",
                   norm=TwoSlopeNorm(vmin=-ESDF_CLIP_M * 0.25, vcenter=0.0, vmax=ESDF_CLIP_M),
                   interpolation="nearest")
    o, vs, (nx, ny, _) = meta["o"], meta["vs"], meta["shape"]
    xs = o[0] + np.arange(nx) * vs
    ys = o[1] + np.arange(ny) * vs
    if np.nanmin(sl) < 0 < np.nanmax(sl):
        ax.contour(xs, ys, sl, levels=[0.0], colors="k", linewidths=1.0)
    if ref_grid is not None:
        rs = np.asarray(ref_grid[:, :, k], np.float32).T
        if np.nanmin(rs) < 0 < np.nanmax(rs):
            ax.contour(xs, ys, rs, levels=[0.0], colors="#555555", linewidths=0.8, linestyles="--")
    if removed_k is not None and len(removed_k):
        ax.scatter(o[0] + removed_k[:, 0] * vs, o[1] + removed_k[:, 1] * vs, s=6, marker="s",
                   facecolors="none", edgecolors="#c800c8", linewidths=0.6, zorder=4)
    zk = o[2] + k * vs
    if tp is not None and len(tp):
        m = np.abs(tp[:, 2] - zk) <= 2 * vs
        if m.any():
            ax.scatter(tp[m, 0], tp[m, 1], s=4, c=C_TARGET, zorder=5)
    if spheres is not None:
        c, r, col = spheres
        dz = np.abs(c[:, 2] - zk)
        hit = np.where(dz < r)[0]
        for i in hit:
            ax.add_patch(Circle((c[i, 0], c[i, 1]), float(np.sqrt(r[i] ** 2 - dz[i] ** 2)), fill=False,
                                ec=col[i], lw=0.9, zorder=6))
    if held is not None:
        for s in held:
            if abs(s[2] - zk) < s[3]:
                ax.add_patch(Circle((s[0], s[1]), float(np.sqrt(s[3] ** 2 - (s[2] - zk) ** 2)), fill=False,
                                    ec="#7b2cbf", lw=1.4, ls="--", zorder=6))
    if crop is not None:
        ax.set_xlim(crop[0], crop[1])
        ax.set_ylim(crop[2], crop[3])
    else:
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x [m] (base)")
    ax.set_ylabel("y [m] (base)")
    ax.set_title(title, fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, extend="both")
    cb.set_label("signed distance [m]", fontsize=8)
    cb.ax.tick_params(labelsize=7)


# --------------------------------------------------------------------------------------------- one call
def render_call(run, stem, calls, seq, chunk_path, ep, n_exec_ep, out_dir, view, run_tag, slice_mode):
    call = calls[seq]
    obs = int(call["obs_index"])
    checks = {}
    with np.load(chunk_path) as z:
        S = json.loads(str(z["summary_json"]))
        checks["t_step_eq_seq"] = int(z["t_step"]) == seq
        checks["solve_ms_match"] = abs(float((S.get("to") or {}).get("solve_ms", np.nan))
                                       - float((call.get("to") or {}).get("solve_ms", np.nan))) < 2e-3
        fb = S.get("exec_feedback") or {}
        if seq > 0 and fb.get("available"):
            checks["feedback_t_step_start_eq_prev_obs"] = int(fb.get("t_step_start", -1)) == int(
                calls[seq - 1]["obs_index"])
        cloud = np.asarray(z["filtered_cloud"], np.float32) if "filtered_cloud" in z.files else np.zeros((0, 3))
        sup = (np.asarray(z["support_mask"], bool) if "support_mask" in z.files and
               len(z["support_mask"]) == len(cloud) else np.zeros(len(cloud), bool))
        tp = np.asarray(z["target_points"], np.float32) if "target_points" in z.files else np.zeros((0, 3))
        sc = np.asarray(z["sphere_centres"], np.float64) if "sphere_centres" in z.files else np.zeros((0, 3))
        sr = np.asarray(z["sphere_radii"], np.float64) if "sphere_radii" in z.files else np.zeros(0)
        clr = np.asarray(z["clearance"], np.float64) if "clearance" in z.files else None
        held = np.asarray(z["attached_query_spheres"], np.float64) if "attached_query_spheres" in z.files else None
        planes = np.asarray(z["support_planes"], np.float64) if "support_planes" in z.files else np.zeros((0, 4))
        fine = grid_meta(z, "esdf_")
        free = grid_meta(z, "esdf_free_")
        coarse = grid_meta(z, "esdf_coarse_")
        D = np.asarray(z["esdf_distance"], np.float32) if fine else None
        F = np.asarray(z["esdf_free_distance"], np.float32) if free else None
        removed = (np.asarray(z["target_free_removed_ijk"], np.int64).reshape(-1, 3)
                   if "target_free_removed_ijk" in z.files else np.zeros((0, 3), np.int64))
        tr_o = np.asarray(z["target_free_trace_origin"], np.float64) if "target_free_trace_origin" in z.files else None
        C = None
        if free is None and coarse is not None:
            C = np.asarray(z["esdf_coarse_distance"], np.float32)
    if tr_o is not None and fine is not None:
        checks["trace_on_fine_grid"] = bool(np.allclose(tr_o, fine["o"]))
    names = list(S.get("sphere_link_names") or [])
    # executed steps of this chunk
    n_exec = int(n_exec_ep[seq]) if seq < len(n_exec_ep) else 0
    nxt = (calls[seq + 1].get("feedback_server") or {}) if seq + 1 < len(calls) else {}
    if nxt.get("n_exec") is not None:
        checks["n_exec_matches_next_feedback"] = int(nxt["n_exec"]) == n_exec
    H = clr.shape[0] if clr is not None else 0
    n_use = n_exec if 0 < n_exec <= H else H
    cmin = clr[:n_use].min(axis=0) if clr is not None and n_use else np.full(len(sr), np.nan)
    colours = [clearance_colour(c) for c in cmin]
    order = np.argsort(-np.nan_to_num(cmin, nan=1e9))      # worst drawn last (on top)
    hand = np.array([i for i, n in enumerate(names) if n in HAND_LINKS], int)
    nm = np.asarray(names)
    tips = [float(sc[nm == f, 2].min()) for f in HAND_LINKS[1:] if len(sc) and (nm == f).any()]
    if isinstance(slice_mode, float):
        zh, z_src = slice_mode, "user --slice value"
    elif slice_mode == "target" and len(tp):
        zh, z_src = float(np.median(tp[:, 2])), f"median z of {len(tp)} target points"
    elif slice_mode in ("fingertip", "target", "auto") and tips:
        zh, z_src = float(np.mean(tips)), "fingertip = mean of each finger's lowest sphere centre"
    elif len(hand):
        zh, z_src = float(np.median(sc[hand, 2])), f"median of {len(hand)} hand/finger sphere centres"
    else:
        zh, z_src = (float(sc[:, 2].max()) if len(sc) else 0.0), "highest sphere (no hand spheres named)"
    if slice_mode == "auto" and fine is not None and len(removed):
        # fingertip, unless that fine slice holds no target-free-removed voxel: then the nearest z layer holding
        # >= 25 % of the busiest layer's removed voxels (so panels 4/5 show the carve). Stated in the titles.
        k0, _ = slice_k(fine, zh)
        cnt = np.bincount(removed[:, 2], minlength=fine["shape"][2])
        if cnt[k0] == 0:
            cand = np.where(cnt >= 0.25 * cnt.max())[0]
            kb = int(cand[np.argmin(np.abs(cand - k0))])
            z_src = (f"auto: fingertip z={zh:.3f} m has no removed voxel -> nearest carved layer "
                     f"(dz {(kb - k0) * fine['vs'] * 1e3:+.0f} mm)")
            z_tip = zh
            zh = float(fine["o"][2] + kb * fine["vs"])
    z_tip = locals().get("z_tip")
    tpm = (S.get("target_permission") or {}).get("target_field_exclude")
    excl = np.asarray(tpm, bool) if tpm is not None and len(tpm) == len(sr) else np.zeros(len(sr), bool)

    grasp = (call.get("grasp") or {}).get("state")
    pt = ((call.get("to") or {}).get("passthrough_clear") or {})
    pt_txt = f"{pt.get('reason')} ({'applied' if pt.get('applied') else 'not applied'})" if pt else "n/a"
    ag3s_status = call.get("ag3s_status") or ((call.get("ag3s") or {}).get("status"))

    fig, axs = plt.subplots(2, 3, figsize=(19.5, 12.0))
    fig.suptitle(
        f"{run_tag}\n{stem}  |  obs {obs}  (call seq {seq}, chunk {pathlib.Path(chunk_path).stem})  |  "
        f"grasp: {grasp}  |  pass-through: {pt_txt}  |  ag3s: {ag3s_status}, TO: {call.get('trajopt_status')}",
        fontsize=11.5)

    # (1)(2) RGB
    for ax, key, lab in ((axs[0, 0], "obs_agentview_image", "agentview"),
                         (axs[0, 1], "obs_robot0_eye_in_hand_image", "wrist (eye-in-hand)")):
        ax.imshow(to_img(ep[key][obs], view))
        vlab = "policy view = raw[::-1, ::-1], mirrored L-R" if view == "policy" else "camera view = raw[::-1]"
        ax.set_title(f"({'1' if 'agent' in key else '2'}) {lab} RGB @ obs {obs}  [{vlab}]", fontsize=10)
        ax.axis("off")

    # cloud roles
    other = ~sup
    roi_pts = np.vstack([p for p in (cloud, sc, tp) if len(p)]) if (len(cloud) or len(sc)) else np.zeros((1, 3))
    lo = np.percentile(roi_pts, 0.5, axis=0) - 0.05
    hi = np.percentile(roi_pts, 99.5, axis=0) + 0.05
    if len(sc):
        lo = np.minimum(lo, (sc - sr[:, None]).min(0) - 0.02)
        hi = np.maximum(hi, (sc + sr[:, None]).max(0) + 0.02)

    def cloud_view(ax, a, b):
        ax.scatter(cloud[sup, a], cloud[sup, b], s=1.2, c=C_SUPPORT, alpha=0.5, lw=0, rasterized=True)
        ax.scatter(cloud[other, a], cloud[other, b], s=1.5, c=C_OTHER, alpha=0.6, lw=0, rasterized=True)
        if len(tp):
            ax.scatter(tp[:, a], tp[:, b], s=4, c=C_TARGET, alpha=0.9, lw=0, zorder=3, rasterized=True)
        if len(sc):
            draw_spheres(ax, sc[:, [a, b]], sr, colours, order, dashed=excl)
        if held is not None and len(held):
            for s in held:
                ax.add_patch(Circle((s[a], s[b]), s[3], fill=False, ec="#7b2cbf", lw=1.6, ls="--", zorder=5))
        ax.set_xlim(lo[a], hi[a])
        ax.set_ylim(lo[b], hi[b])
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)

    n_neg = int(np.sum(cmin < 0)) if np.isfinite(cmin).any() else 0
    n_near = int(np.sum((cmin >= 0) & (cmin < 0.010))) if np.isfinite(cmin).any() else 0
    cmin_all = float(np.nanmin(cmin)) if np.isfinite(cmin).any() else float("nan")
    legend = [Line2D([], [], ls="", marker="o", ms=5, mfc=C_TARGET, mec="none", label=f"target points ({len(tp)})"),
              Line2D([], [], ls="", marker="o", ms=5, mfc=C_SUPPORT, mec="none", label=f"support ({int(sup.sum())})"),
              Line2D([], [], ls="", marker="o", ms=5, mfc=C_OTHER, mec="none", label=f"other ({int(other.sum())})"),
              Patch(fc="#e0000040", ec="#e00000", label=f"sphere min clr < 0 ({n_neg})"),
              Patch(fc="#ff990040", ec="#ff9900", label=f"0-10 mm ({n_near})"),
              Patch(fc="#2ca02c40", ec="#2ca02c", label=">= 10 mm")]
    if excl.any():
        legend.append(Patch(fc="none", ec="k", ls="--", label=f"dashed edge: TO queries these {int(excl.sum())} rows\n"
                            "on the target-free tier; recorded clr\nis vs the field WITH the target"))
    if held is not None and len(held):
        legend.append(Line2D([], [], color="#7b2cbf", ls="--", label=f"held object (attached, {len(held)})"))

    # (3) top view
    ax = axs[0, 2]
    cloud_view(ax, 0, 1)
    if fine:
        e = extent_xy(fine)
        ax.add_patch(Rectangle((e[0], e[2]), e[1] - e[0], e[3] - e[2], fill=False, ec="k", ls=":", lw=1.0))
        legend.append(Line2D([], [], color="k", ls=":", label="fine ESDF window"))
    dst = S.get("destination") or {}
    if dst.get("aabb_min") and dst.get("aabb_max"):
        a0, a1 = dst["aabb_min"], dst["aabb_max"]
        ax.add_patch(Rectangle((a0[0], a0[1]), a1[0] - a0[0], a1[1] - a0[1], fill=False, ec="#8c564b", lw=1.2,
                               ls="-."))
        legend.append(Line2D([], [], color="#8c564b", ls="-.", label=f"destination ({dst.get('source')})"))
    ax.legend(handles=legend, fontsize=7, loc="lower left", framealpha=0.85)
    ax.set_xlabel("x [m] (base)")
    ax.set_ylabel("y [m] (base)")
    ax.set_title(f"(3) top view: fused cloud ({len(cloud)} pts) + {len(sc)} robot spheres @ plan step 0\n"
                 f"sphere colour = min clearance over executed steps 0..{n_use - 1} "
                 f"(min {cmin_all * 1e3:.1f} mm)", fontsize=10)

    # crop for the slices: fine window, or around the hand if no fine grid
    crop = None
    hand_xy = sc[hand][:, :2] if len(hand) else None
    sph = (sc, sr, colours) if len(sc) else None
    # (4) fine slice
    ax = axs[1, 0]
    rec = {"slice_z_m": zh, "slice_z_source": z_src}
    if fine is not None:
        k, clipped = slice_k(fine, zh)
        rk = removed[removed[:, 2] == k] if len(removed) else removed
        rec.update({"fine_k": k, "fine_k_clipped": clipped, "fine_slice_z_m": float(fine["o"][2] + k * fine["vs"]),
                    "n_removed_total": int(len(removed)), "n_removed_in_slice": int(len(rk))})
        esdf_slice_panel(ax, fig, D, fine, k,
                         f"(4) fine ESDF ({fine['vs'] * 1e3:.0f} mm) slice k={k}, z={fine['o'][2] + k * fine['vs']:.3f} m"
                         f"{' (CLIPPED to grid)' if clipped else ''}\n[{z_src}]\nblack = zero level; magenta squares = target-free removed"
                         f"\nvoxels in slice ({len(rk)} of {len(removed)}); circles = sphere sections; red = target pts",
                         removed_k=rk, spheres=sph, held=held, tp=tp)
    else:
        ax.text(0.5, 0.5, "no fine ESDF recorded", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    # (5) target-free slice (or coarse fallback)
    ax = axs[1, 1]
    if free is not None and F is not None:
        k5, cl5 = slice_k(free, zh)
        same = fine is not None and np.allclose(free["o"], fine["o"]) and free["shape"] == fine["shape"]
        d_in = None
        if same and len(removed):
            rk = removed[removed[:, 2] == k5]
            if len(rk):
                d_in = (float(np.median(D[rk[:, 0], rk[:, 1], rk[:, 2]])),
                        float(np.median(F[rk[:, 0], rk[:, 1], rk[:, 2]])))
        rec.update({"free_k": k5, "removed_median_fine_vs_free_m": d_in})
        esdf_slice_panel(ax, fig, F, free, k5,
                         f"(5) target-free tier slice k={k5}, z={free['o'][2] + k5 * free['vs']:.3f} m\n"
                         f"black = zero level (target carved out)\ndashed grey = fine-tier zero level (with target)",
                         ref_grid=D if same else None, spheres=sph, held=held)
    elif C is not None:
        k5, cl5 = slice_k(coarse, zh)
        crop = extent_xy(fine) if fine is not None else None
        rec.update({"free_k": None, "coarse_k": k5})
        esdf_slice_panel(ax, fig, C, coarse, k5,
                         f"(5) NO target-free tier on this call (status {S.get('status')})\n-> coarse tier "
                         f"({coarse['vs'] * 1e3:.0f} mm) slice k={k5}, z={coarse['o'][2] + k5 * coarse['vs']:.3f} m"
                         f"{' (CLIPPED)' if cl5 else ''}\ncropped to the fine window; dashed purple = held object",
                         spheres=sph, held=held, crop=crop)
    else:
        ax.text(0.5, 0.5, "no target-free / coarse tier recorded", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    # (6) side view
    ax = axs[1, 2]
    cloud_view(ax, 0, 2)
    xx = np.array([lo[0], hi[0]])
    yc = float(np.median(cloud[:, 1])) if len(cloud) else 0.0
    for p in planes:
        if abs(p[2]) > 0.5:
            ax.plot(xx, (p[3] - p[0] * xx - p[1] * yc) / p[2], color="#444444", lw=1.0, ls="-")
    ax.axhline(zh, color="#c800c8", lw=1.0, ls="--")
    if z_tip is not None:
        ax.axhline(z_tip, color="#c800c8", lw=0.8, ls=":")
    ax.set_xlabel("x [m] (base)")
    ax.set_ylabel("z [m] (base)")
    ax.set_title(f"(6) side view (x-z): cloud + spheres; grey line = support plane(s) ({len(planes)}),\n"
                 + (f"magenta dashed = slice z={zh:.3f} m (nearest carved layer)\ndotted = fingertip z={z_tip:.3f} m"
                  if z_tip is not None else f"magenta dashed = slice height z={zh:.3f} m\n({z_src})"),
                 fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / f"{stem}_obs{obs:04d}_seq{seq:02d}.png"
    fig.savefig(out, dpi=90)
    plt.close(fig)

    rec.update({
        "png": str(out), "seq": seq, "obs_index": obs, "chunk": str(chunk_path), "grasp_state": grasp,
        "passthrough": pt, "ag3s_status": ag3s_status, "trajopt_status": call.get("trajopt_status"),
        "summary_status": S.get("status"), "n_cloud": int(len(cloud)), "n_support": int(sup.sum()),
        "n_target_points": int(len(tp)), "n_spheres": int(len(sc)), "n_exec_steps": n_exec,
        "clearance_steps_used": n_use, "min_clearance_exec_mm": cmin_all * 1e3 if np.isfinite(cmin_all) else None,
        "n_spheres_below_0": n_neg, "n_spheres_0_10mm": n_near,
        "worst_sphere_link": names[int(np.nanargmin(cmin))] if names and np.isfinite(cmin).any() else None,
        "has_target_free_tier": free is not None, "held_spheres": None if held is None else held.tolist(),
        "checks": checks,
    })
    return rec


# --------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n---")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run dir name or glob under results/benchmark (or absolute path)")
    ap.add_argument("--stem", required=True, help="episode stem, e.g. libero_spatial_t4_e0")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--obs", help="comma-separated env obs indices (calls' obs_index)")
    g.add_argument("--auto", type=int, help="pick N calls spread over the episode (incl. the first held call)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-view", choices=("policy", "opencv"), default="policy",
                    help="policy = raw[::-1, ::-1] (what pi05 sees; mirrored), opencv = raw[::-1] (true camera)")
    ap.add_argument("--slice", default="auto",
                    help="ESDF slice height: auto (default: fingertip, or the nearest carved layer when the fingertip "
                         "slice has no target-free-removed voxel) | fingertip | hand | target | <z [m], base frame>")
    a = ap.parse_args()
    try:
        slice_mode = float(a.slice)
    except ValueError:
        slice_mode = a.slice
        if slice_mode not in ("auto", "fingertip", "hand", "target"):
            ap.error("--slice must be auto, fingertip, hand, target or a number")

    run = resolve_run(a.run)
    calls = load_calls(run, a.stem)
    seg, how = map_chunks(run, a.stem, calls)
    picks = pick_calls(calls, a.obs, a.auto if a.auto or not a.obs else None)
    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = run.name.split("_", 2)[-1] if run.name.count("_") >= 2 else run.name
    ep = np.load(run / "episodes" / f"{a.stem}.npz", mmap_mode=None)
    ci = np.asarray(ep["chunk_index"])
    wait = np.asarray(ep["is_wait"], bool) if "is_wait" in ep.files else np.zeros(len(ci), bool)
    n_exec_ep = np.bincount(ci[(ci >= 0) & ~wait], minlength=len(calls))
    if "chunk_t" in ep.files:
        ct = np.asarray(ep["chunk_t"]).tolist()
        if ct != [int(c["obs_index"]) for c in calls]:
            print("[warn] episode chunk_t != calls obs_index")
    print(f"run {run.name}\n{len(calls)} calls, chunks {pathlib.Path(seg[0]).stem}..{pathlib.Path(seg[-1]).stem} ({how})")
    records = []
    for seq in picks:
        r = render_call(run, a.stem, calls, seq, seg[seq], ep, n_exec_ep, out_dir, a.image_view, run_tag, slice_mode)
        bad = [k for k, v in r["checks"].items() if v is False]
        print(f"  seq {seq:2d} obs {r['obs_index']:3d} grasp {r['grasp_state']:<9} min clr "
              f"{r['min_clearance_exec_mm']:.1f} mm -> {pathlib.Path(r['png']).name}"
              + (f"  [CHECK FAILED: {bad}]" if bad else ""))
        records.append(r)
    index = {"run": run.name, "run_dir": str(run), "stem": a.stem, "mapping": how, "image_view": a.image_view,
             "chunks": [seg[0], seg[-1]], "selection": {"obs": a.obs, "auto": a.auto},
             "colour_rule_clearance": {"red": "< 0 mm", "orange": "0-10 mm", "green": ">= 10 mm"},
             "calls": records}
    (out_dir / f"{a.stem}_index.json").write_text(json.dumps(index, indent=1, default=float))
    print(f"index -> {out_dir / (a.stem + '_index.json')}")


if __name__ == "__main__":
    main()
