"""Fig. 1 v3 assets: real-run renders for the 'ours' rows and the top strip, plus panel crops of recorded baseline figures.

render: one SafeLIBERO Level I episode of the final AG3S run and the VLA-only run on the same initial state and seed.
  top_vla_*.png / top_ag3s_*.png  agent-view frames at the VLA's first official collision step and later steps
  c1_ours_attention.png           noun attention of the policy (head camera, 16x16) over the agent view + seed points
  c2_ours_extent.png              target points (orange) and neighbour voxels kept in the field (blue) projected
  c3_ours_links.png               robot collision spheres: contact links (gold) / other links (grey) / attached (purple)
crop: every image tile of the listed baseline documentation figures, written as crop_<figure>_rXcY.png.
All images are OpenCV camera view (raw[::-1]); attention is mirrored back from the policy view. Sidecar: assets.json.
"""
import argparse, glob, hashlib, json, pathlib, sys
import numpy as np
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/segmentation"))
import gtto_feature_scene as fs  # noqa: E402  (read-only reuse: load_calls, map_chunks)

BENCH = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
DOCS = ROOT / "AG3S/ag3s/docs/figures/baselines"
CROPS = ["2026-11-03_safelibero-seg-ats/fig4_erase_keep_maps.png", "2026-11-03_safelibero-seg-ats/fig3_scene_overlays.png",
         "grounded_sam2_track/v2-sl3-track-strip.png", "sl4b/v2-sl4b-scene-overlay.png",
         "2026-11-01_multibench-segmentation/occlusion_tracking_example.png",
         "2026-11-01_multibench-segmentation/sam2_prompt_comparison_example.png",
         "2026-11-01_segmentation-baseline/fixed_segmentation_examples.png",
         "aegis/v1-aegis-scene-detection-ellipsoid.png", "knows/v1-knows-scene-ellipsoids.png",
         "multilink_cbf/v1-multilink-scene-link-ellipsoids.png"]
UP = 3  # 256 px -> 768 px for drawing
ORANGE, BLUE, GOLD, GREY, PURPLE = (255, 140, 0), (40, 110, 230), (240, 190, 20), (150, 150, 150), (150, 60, 200)


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def run_dir(pattern):
    hits = sorted(glob.glob(str(BENCH / pattern)))
    assert len(hits) == 1, (pattern, hits)
    return pathlib.Path(hits[0])


def episode(run, task, ep):
    for line in (run / "episodes.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d["task_id"] == task and d["episode_idx"] == ep:
            return d
    raise KeyError((run, task, ep))


def cam_view(raw):
    return np.ascontiguousarray(raw[::-1])


def big(img):
    return Image.fromarray(img).resize((256 * UP, 256 * UP), Image.LANCZOS)


def project(P, K, T_base_cam):
    P = np.asarray(P, np.float64).reshape(-1, 3)
    Tc = np.linalg.inv(T_base_cam)
    pc = P @ Tc[:3, :3].T + Tc[:3, 3]
    z = pc[:, 2]
    uv = (pc[:, :2] / np.maximum(z[:, None], 1e-6)) @ np.diag([K[0, 0], K[1, 1]]) + K[:2, 2]
    return uv, z


def geom_at(geom, obs):
    rows = np.where(np.asarray(geom["obs_index"]) == obs)[0]
    assert len(rows), f"obs {obs} not in geometry record"
    i = int(rows[0])
    return np.asarray(geom["K/head"][i]), np.asarray(geom["T_base_cam/head"][i])


def dots(draw, uv, z, colour, r=2.2, outline=None):
    for (u, v), zz in zip(uv, z):
        if zz <= 0 or not (0 <= u < 256 and 0 <= v < 256):
            continue
        x, y = (u + 0.5) * UP, (v + 0.5) * UP
        draw.ellipse([x - r * UP / 2, y - r * UP / 2, x + r * UP / 2, y + r * UP / 2], fill=colour + (200,),
                     outline=outline)


def render(args, out, rec):
    ag_run = run_dir(f"*qe-pe123458-e1a-spatial-LI-t{args.task}-*")
    vla_run = run_dir(f"*qe-vla-e1a-spatial-LI-t{args.task}-*")
    stem = f"safelibero_spatial_t{args.task}_e{args.ep}"
    d_ag, d_vla = episode(ag_run, args.task, args.ep), episode(vla_run, args.task, args.ep)
    assert d_ag["seed"] == d_vla["seed"] and d_ag["init_state_index"] == d_vla["init_state_index"]
    ct = d_vla["safelibero"]["collide_t"]
    print(f"{stem}: seed {d_ag['seed']} init {d_ag['init_state_index']}; VLA collided at {ct}, "
          f"AG3S collided={d_ag['safelibero']['collided']} success={d_ag['official_success']}")
    A_ag = np.load(ag_run / d_ag["npz"])["obs_agentview_image"]
    A_vla = np.load(vla_run / d_vla["npz"])["obs_agentview_image"]
    steps = sorted({max(ct - 10, 0), ct, ct + 25, ct + 60})
    for name, A in (("vla", A_vla), ("ag3s", A_ag)):
        for s in steps:
            s2 = min(s, len(A) - 1)
            p = out / f"top_{name}_step{s2:03d}.png"
            big(cam_view(A[s2])).save(p)
            rec[p.name] = {"source": str((vla_run if name == "vla" else ag_run) / d_ag["npz"]), "frame": s2,
                           "episode": stem, "note": f"VLA first official collision at step {ct}"}

    calls = fs.load_calls(ag_run, stem)
    seg, how = fs.map_chunks(ag_run, stem, calls)
    geom = np.load(ag_run / "episodes" / f"{stem}_geom.npz", allow_pickle=True)
    grasp = [((c.get("grasp") or {}).get("state") or "") for c in calls]
    print("grasp states per call:", grasp)

    def chunk(seq):
        return np.load(seg[seq], allow_pickle=True)

    # identity: first call with seeds and an open hand
    # first call after the target is latched (in t2 e9, 29 of 90 seeds of call 0 lie on the bottle; none later)
    s1 = next((i for i in range(len(calls)) if grasp[i] == "latched" and len(chunk(i)["seed_indices"]) > 0),
              next(i for i in range(len(calls)) if grasp[i] != "held" and len(chunk(i)["seed_indices"]) > 0))
    z = chunk(s1); obs = int(calls[s1]["obs_index"]); K, Tbc = geom_at(geom, obs)
    # The stored 16x16 attention_head map does not correlate with the per-point values in any orientation
    # (|r| < 0.03), so the panel shows the per-point attention the method uses: attention_values on attention_cloud.
    P = np.asarray(z["attention_cloud"]); val = np.asarray(z["attention_values"], np.float64)
    uv, zz = project(P, K, Tbc)
    vis = (zz > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < 256) & (uv[:, 1] >= 0) & (uv[:, 1] < 256)
    hi = np.quantile(val[vis], 0.995) if vis.any() else 1.0
    w = np.clip(val / max(hi, 1e-12), 0, 1)
    heat = np.zeros((256 * UP, 256 * UP), np.float32)
    for (u, v), ww, ok in zip(uv, w, vis):
        if ok and ww > 0.02:
            x, y = int((u + 0.5) * UP), int((v + 0.5) * UP)
            heat[max(y - 4, 0):y + 5, max(x - 4, 0):x + 5] = np.maximum(heat[max(y - 4, 0):y + 5, max(x - 4, 0):x + 5], ww)
    heat = np.asarray(Image.fromarray(np.uint8(heat * 255)).filter(__import__("PIL.ImageFilter", fromlist=["x"])
                                                                 .GaussianBlur(5)), np.float32) / 255.0
    heat = heat / max(heat.max(), 1e-9)
    base = np.asarray(big(cam_view(A_ag[obs])), np.float32)
    cmap = np.stack([np.ones_like(heat), 0.15 + 0.7 * heat, 0.1 * (1 - heat)], -1) * 255   # yellow-red
    a = (0.75 * heat ** 0.8)[..., None]
    img = Image.fromarray(np.uint8(base * (1 - a) + cmap * a)).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
    seeds = P[np.asarray(z["seed_indices"])]
    uv, zz = project(seeds, K, Tbc); dots(dr, uv, zz, GOLD, r=2.2, outline=(90, 60, 0))
    p = out / "c1_ours_attention.png"; Image.alpha_composite(img, ov).convert("RGB").save(p)
    rec[p.name] = {"source_chunk": seg[s1], "call_seq": s1, "obs_index": obs, "grasp": grasp[s1],
                   "n_points_visible": int(vis.sum()), "n_seeds": int(len(seeds)),
                   "content": "noun attention lifted to the observed points (attention_values on attention_cloud), "
                              "projected to the head camera and blurred; gold = seed points. The stored 16x16 "
                              "attention_head map is not used (no correlation with the per-point values)."}

    # extent: call before holding with the most target-free voxels removed
    pre = [i for i in range(len(calls)) if grasp[i] != "held"]
    s2 = max(pre, key=lambda i: len(chunk(i)["target_free_removed_ijk"]))
    z = chunk(s2); obs = int(calls[s2]["obs_index"]); K, Tbc = geom_at(geom, obs)
    o, vs = np.asarray(z["target_free_trace_origin"]), float(z["esdf_voxel_size"])
    kept = o + (np.asarray(z["target_free_nontarget_kept_ijk"], np.float64) + 0.5) * vs
    removed = o + (np.asarray(z["target_free_removed_ijk"], np.float64) + 0.5) * vs
    img = big(cam_view(A_ag[obs])).convert("RGBA"); ov = Image.new("RGBA", img.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
    uv, zz = project(kept, K, Tbc); dots(dr, uv, zz, BLUE, r=2.4)
    uv, zz = project(removed, K, Tbc); dots(dr, uv, zz, ORANGE, r=2.4)
    p = out / "c2_ours_extent.png"; Image.alpha_composite(img, ov).convert("RGB").save(p)
    rec[p.name] = {"source_chunk": seg[s2], "call_seq": s2, "obs_index": obs, "grasp": grasp[s2],
                   "n_removed": int(len(removed)), "n_neighbour_kept": int(len(kept)),
                   "content": "orange = voxels removed from the target-free field (target surface); blue = non-target "
                              "voxels kept protected next to it"}

    # permission: a held call (transport) if any, else the last call
    held = [i for i in range(len(calls)) if grasp[i] == "held"]
    s3 = held[len(held) // 3] if held else len(calls) - 1
    z = chunk(s3); obs = int(calls[s3]["obs_index"]); K, Tbc = geom_at(geom, obs)
    S = json.loads(str(z["summary_json"]))
    names = list(S.get("sphere_link_names") or [])
    excl = np.asarray(((S.get("target_permission") or {}).get("target_field_exclude")) or [False] * len(names), bool)
    sc, sr = np.asarray(z["sphere_centres"]), np.asarray(z["sphere_radii"])
    img = big(cam_view(A_ag[obs])).convert("RGBA"); ov = Image.new("RGBA", img.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
    uv, zz = project(sc, K, Tbc)
    order = np.argsort(-zz)
    for i in order:
        if zz[i] <= 0:
            continue
        rpx = K[0, 0] * sr[i] / zz[i] * UP
        col = GOLD if (names and names[i] in fs.HAND_LINKS) else GREY
        x, y = (uv[i, 0] + 0.5) * UP, (uv[i, 1] + 0.5) * UP
        dr.ellipse([x - rpx, y - rpx, x + rpx, y + rpx], fill=col + (70,), outline=col + (230,), width=2)
    n_att = 0
    if "attached_query_spheres" in z.files:
        aq = np.asarray(z["attached_query_spheres"], np.float64).reshape(-1, 4) if np.asarray(
            z["attached_query_spheres"]).size else np.zeros((0, 4))
        n_att = len(aq)
        if n_att:
            uva, za = project(aq[:, :3], K, Tbc)
            for (u, v), zz2, r in zip(uva, za, aq[:, 3]):
                rpx = K[0, 0] * r / max(zz2, 1e-6) * UP; x, y = (u + 0.5) * UP, (v + 0.5) * UP
                dr.ellipse([x - rpx, y - rpx, x + rpx, y + rpx], fill=PURPLE + (80,), outline=PURPLE + (230,), width=2)
    p = out / "c3_ours_links.png"; Image.alpha_composite(img, ov).convert("RGB").save(p)
    rec[p.name] = {"source_chunk": seg[s3], "call_seq": s3, "obs_index": obs, "grasp": grasp[s3],
                   "n_spheres": int(len(sc)), "n_attached": n_att, "target_field_exclude_any": bool(excl.any()),
                   "content": "gold = hand/finger spheres (contact links), grey = other link spheres, purple = attached"}
    rec["_episode"] = {"ag3s_run": str(ag_run), "vla_run": str(vla_run), "stem": stem, "chunk_mapping": how,
                       "seed": d_ag["seed"], "init_state_index": d_ag["init_state_index"],
                       "obstacle": d_ag["safelibero"]["obstacle"], "vla_collide_t": ct,
                       "ag3s_collided": d_ag["safelibero"]["collided"], "ag3s_success": d_ag["official_success"]}


def tiles(arr, min_side=110, white=245):
    nonwhite = (arr < white).any(-1)
    def runs(v):
        out, start = [], None
        for i, b in enumerate(v):
            if b and start is None: start = i
            if not b and start is not None: out.append((start, i)); start = None
        if start is not None: out.append((start, len(v)))
        return out
    res = []
    for r0, r1 in runs(nonwhite.mean(1) > 0.25):
        if r1 - r0 < min_side: continue
        band = nonwhite[r0:r1]
        for c0, c1 in runs(band.mean(0) > 0.6):
            if c1 - c0 >= min_side: res.append((r0, r1, c0, c1))
    return res


def crop(out, rec):
    for f in CROPS:
        src = DOCS / f; arr = np.asarray(Image.open(src).convert("RGB"))
        boxes = tiles(arr)
        rows = sorted({b[0] for b in boxes})
        for (r0, r1, c0, c1) in boxes:
            ri = rows.index(r0); ci = sorted(b[2] for b in boxes if b[0] == r0).index(c0)
            name = f"crop_{src.parent.name}__{src.stem}_r{ri}c{ci}.png"
            Image.fromarray(arr[r0:r1, c0:c1]).save(out / name)
            rec[name] = {"source": str(src.relative_to(ROOT)), "sha256": sha(src), "box_rows_cols": [r0, r1, c0, c1]}
        print(f"{f}: {len(boxes)} tiles")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=2); ap.add_argument("--ep", type=int, default=9)
    ap.add_argument("--out", default=str(ROOT / "paper/figures/fig01_teaser/v3/assets"))
    ap.add_argument("--only", choices=("render", "crop", "all"), default="all")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    side = out / "assets.json"
    rec = json.loads(side.read_text()) if side.exists() else {}
    if a.only in ("crop", "all"): crop(out, rec)
    if a.only in ("render", "all"): render(a, out, rec)
    rec["_script"] = {"path": str(pathlib.Path(__file__).relative_to(ROOT)), "sha256": sha(__file__)}
    side.write_text(json.dumps(rec, indent=1, default=str))
    print("->", out)


if __name__ == "__main__":
    main()
