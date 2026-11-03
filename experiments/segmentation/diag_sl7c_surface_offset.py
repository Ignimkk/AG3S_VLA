#!/usr/bin/env python3
"""SL7c ② — why the P0m surface region lands ~150 mm off the ATS target point on the Panda (task `SL7c.task.md` ②).

For each chosen (episode, call) of the SL7b offline replay (`surface.why == not_ats_target`), replay the same episode
with the same AG3S object (`eval_real_ats_safelibero.py` loop: stored noun rows L8H6, gripper-physics grasp signal)
up to that call, capture the views and ATS points that `_attention_surface_update_t78` receives, and re-run the
selection on the CPU with `keep_debug=True` (geometry, seed patches, solve, pixel→node map). Then measure against the
GT segmentation of the same instant:

  (a) attention: are the top attention patches / the seed patches on the GT target mask?
  (b) dummy view: does the padded right-wrist view give any node or seed?
  (c) graph spread: seed-patch nodes on the target vs the selected region's pixels on the target
  (d) resolution: patch footprint and depth-pixel footprint (mm) at the target, # patches the target covers

Writes `<out>/<stem>_t<t>.json` (numbers) and `<out>/<stem>_t<t>.png` (overlay per camera + top-down metric view).
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'benchmark/runners')); sys.path.insert(0, str(Path(__file__).resolve().parent))
import perception_replay as PR
import safelibero_scoring as S
from eval_real_ats_offline import build_ag3s
from eval_real_ats_safelibero import physical_state

GRID, PATCH = 16, 16          # 16×16 tokens per camera; 256 px render / 16 = 16 px per patch (policy 224 / 16 = 14)
TARGET = S.ci('target')


def project(P, K, T_base_cam, shape):
    """base-frame points (N, 3) → pixel (u, v) of the OpenCV camera, in-image and in front only."""
    if P is None or not len(P):
        return np.zeros((0, 2))
    Tc = np.linalg.inv(T_base_cam)
    pc = P @ Tc[:3, :3].T + Tc[:3, 3]
    ok = pc[:, 2] > 1e-3
    uv = (pc[ok, :2] / pc[ok, 2:3]) @ K[:2, :2].T + K[:2, 2]
    h, w = shape
    ok2 = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv[ok2]


def patch_of(u, v):
    return (np.asarray(v) // PATCH).astype(int) * GRID + (np.asarray(u) // PATCH).astype(int)


def analyse(views, ats_pts, rec_live, labels, rgb, stem, t):
    from AG3S.ag3s.runtime.attention_surface import ARM_SPHERES, select_target_surface, graph_config
    from AG3S.ag3s.runtime import attention_surface_post as post_mod

    res = select_target_surface(views, arm=ARM_SPHERES, keep_debug=True, device='cpu')
    rec = res.record
    geo, pix = res.geometry, res.pixel_nodes
    sel = np.zeros(len(geo.pos), bool) if geo is None else np.zeros(len(geo.pos), bool)
    sel[res.selected_nodes] = True
    out = {'stem': stem, 't': int(t), 'cpu_record': {k: v for k, v in rec.items() if isinstance(v, (int, float, str, bool, list))},
           'live_record': rec_live, 'cameras': {}}
    # target check with the CPU selection (same rule as live)
    if res.codes is not None:
        region = post_mod.region_from_codes(views, res.codes, voxel=float(graph_config().voxel))
        ok, tc = post_mod.target_check(region, ats_pts)
        out['cpu_target_check'] = {k: v for k, v in tc.items() if isinstance(v, (int, float, str, bool))}
    seeds = [int(s) for s in rec.get('seed_patches', [])]
    smap = np.concatenate([np.asarray(v.attention, np.float64).reshape(-1) for v in views])
    top5_all = [int(i) for i in np.argsort(-smap)[:5]]
    out['seed_patches'] = seeds
    out['top5_attention_patches_any'] = top5_all
    out['seed_cameras'] = sorted({views[s // 256].name for s in seeds})
    per = {}
    for c, (v, (u, vv, node)) in enumerate(zip(views, pix)):
        name = v.name
        info = {'n_valid_px': int((node >= 0).sum())}
        if name not in labels:            # dummy right wrist (b)
            info['dummy'] = True
            info['n_seed_patches'] = int(sum(1 for s in seeds if s // 256 == c))
            info['attention_sum'] = float(np.asarray(v.attention).sum())
            per[name] = info
            continue
        lab = labels[name]
        tmask = lab == TARGET
        ok = node >= 0
        tp_px = tmask[vv, u]
        # (a) attention vs GT target mask
        A = np.asarray(v.attention, np.float64).reshape(GRID, GRID)
        tpatch = np.zeros(GRID * GRID)
        np.add.at(tpatch, patch_of(*np.nonzero(tmask)[::-1]), 1)
        rank = np.argsort(-A.reshape(-1))
        info['target_px'] = int(tmask.sum())
        info['target_patches_any'] = int((tpatch > 0).sum())
        info['target_patches_ge25pct'] = int((tpatch >= 0.25 * PATCH * PATCH).sum())
        info['top5_cam_patches'] = [int(r) for r in rank[:5]]
        info['top5_cam_on_target'] = int(sum(tpatch[r] > 0 for r in rank[:5]))
        info['top1_cam_target_px'] = int(tpatch[rank[0]])
        info['attention_mass_on_target_patches'] = float(A.reshape(-1)[tpatch > 0].sum() / max(A.sum(), 1e-12))
        cam_seeds = [s - 256 * c for s in seeds if s // 256 == c]
        info['seed_patches_cam'] = cam_seeds
        info['seed_patches_on_target'] = int(sum(tpatch[s] > 0 for s in cam_seeds))
        # (c) seed-patch nodes on target vs selected region pixels on target
        ppx = patch_of(u, vv)
        in_seed = np.isin(ppx, cam_seeds) & ok
        info['seed_patch_valid_px'] = int(in_seed.sum())
        info['seed_patch_px_on_target_frac'] = float(tp_px[in_seed].mean()) if in_seed.any() else None
        reg = ok & sel[np.where(ok, node, 0)]
        info['region_px'] = int(reg.sum())
        info['region_px_on_target_frac'] = float(tp_px[reg].mean()) if reg.any() else None
        cls = lab[vv[reg], u[reg]]
        info['region_px_class'] = {S.CLASSES[k]: int((cls == k).sum()) for k in np.unique(cls)} if reg.any() else {}
        cls_s = lab[vv[in_seed], u[in_seed]]
        info['seed_px_class'] = {S.CLASSES[k]: int((cls_s == k).sum()) for k in np.unique(cls_s)} if in_seed.any() else {}
        # (d) resolution at the target
        d = np.asarray(v.depth_m)
        z = d[tmask & (d > 0)]
        fx = float(np.asarray(v.K)[0, 0])
        if len(z):
            zm = float(np.median(z))
            info['target_depth_m'] = zm
            info['depth_px_mm'] = 1e3 * zm / fx
            info['patch_mm'] = 1e3 * zm * PATCH / fx
        info['fx'] = fx
        # ATS points in this view
        uv_ats = project(ats_pts, np.asarray(v.K), np.asarray(v.T_base_cam), d.shape)
        if len(uv_ats):
            ui, vi = uv_ats[:, 0].astype(int), uv_ats[:, 1].astype(int)
            info['ats_proj_n'] = int(len(uv_ats))
            info['ats_proj_on_target_frac'] = float(tmask[vi, ui].mean())
        per[name] = info
        info['_draw'] = dict(A=A, tmask=tmask, reg=(u[reg], vv[reg]), seeds=cam_seeds, uv_ats=uv_ats, top5=list(rank[:5]))
    out['cameras'] = per
    # 3D: region, ATS, GT target points (back-projected GT target pixels of both cameras)
    gt3 = []
    for v in views:
        if v.name in labels:
            uu, vv_ = np.nonzero((labels[v.name] == TARGET).T)
            d = np.asarray(v.depth_m)[vv_, uu]
            k = d > 0
            Kc = np.asarray(v.K)
            pc = np.stack([(uu[k] - Kc[0, 2]) / Kc[0, 0] * d[k], (vv_[k] - Kc[1, 2]) / Kc[1, 1] * d[k], d[k]], 1)
            T = np.asarray(v.T_base_cam)
            gt3.append(pc @ T[:3, :3].T + T[:3, 3])
    gt3 = np.concatenate(gt3) if gt3 else np.zeros((0, 3))
    rp = geo.pos[sel] if geo is not None else np.zeros((0, 3))
    seed_nodes = np.zeros(0, np.int64)
    for c, (u, vv, node) in enumerate(pix):
        cs = [s - 256 * c for s in seeds if s // 256 == c]
        m = np.isin(patch_of(u, vv), cs) & (node >= 0)
        seed_nodes = np.concatenate([seed_nodes, node[m]])
    sp = geo.pos[np.unique(seed_nodes)] if len(seed_nodes) else np.zeros((0, 3))
    cen = lambda P: None if P is None or not len(P) else [round(float(x), 4) for x in P.mean(0)]
    out['centroids_m'] = {'gt_target': cen(gt3), 'ats': cen(ats_pts), 'region': cen(rp), 'seed_nodes': cen(sp)}
    dist = lambda a, b: None if a is None or b is None else round(1e3 * float(np.linalg.norm(np.subtract(a, b))), 1)
    c3 = out['centroids_m']
    out['dist_mm'] = {'region_to_gt': dist(c3['region'], c3['gt_target']), 'ats_to_gt': dist(c3['ats'], c3['gt_target']),
                      'seed_to_gt': dist(c3['seed_nodes'], c3['gt_target']), 'region_to_ats': dist(c3['region'], c3['ats'])}
    draw3 = dict(gt3=gt3, rp=rp, sp=sp, ats=ats_pts if ats_pts is not None else np.zeros((0, 3)))
    return out, draw3


def figure(out, draw3, rgb, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm

    cams = [c for c in ('head', 'left_wrist') if c in out['cameras'] and '_draw' in out['cameras'][c]]
    fig, axes = plt.subplots(1, len(cams) + 1, figsize=(5.2 * (len(cams) + 1), 5.2))
    for ax, c in zip(axes, cams):
        dr = out['cameras'][c]['_draw']
        img = rgb[c]
        ax.imshow(img)
        A = dr['A']
        A_up = np.kron(A / max(A.max(), 1e-12), np.ones((PATCH, PATCH)))
        im = ax.imshow(A_up, cmap='magma', alpha=0.45, vmin=0, vmax=1, interpolation='nearest')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label('L8H6 attention (÷ camera max)')
        if dr['tmask'].any():
            ax.contour(dr['tmask'].astype(float), levels=[0.5], colors='lime', linewidths=1.6)
        u, v = dr['reg']
        ax.scatter(u, v, s=1, c='cyan', alpha=0.35, linewidths=0)
        for s in dr['seeds']:
            r, q = divmod(s, GRID)
            ax.add_patch(plt.Rectangle((q * PATCH, r * PATCH), PATCH, PATCH, fill=False, ec='yellow', lw=1.8))
        for s in dr['top5']:
            r, q = divmod(int(s), GRID)
            ax.add_patch(plt.Rectangle((q * PATCH + 2, r * PATCH + 2), PATCH - 4, PATCH - 4, fill=False, ec='white', lw=0.8, ls='--'))
        if len(dr['uv_ats']):
            ax.scatter(dr['uv_ats'][:, 0], dr['uv_ats'][:, 1], s=6, c='red', marker='x', linewidths=0.6)
        ci = out['cameras'][c]
        ax.set_title(f"{c}: seed on target {ci['seed_patches_on_target']}/{len(ci['seed_patches_cam'])}, "
                     f"region on target {0 if ci['region_px_on_target_frac'] is None else ci['region_px_on_target_frac']:.0%}",
                     fontsize=9)
        ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0); ax.axis('off')
    ax = axes[-1]
    g, rp, sp, at = draw3['gt3'], draw3['rp'], draw3['sp'], draw3['ats']
    ax.scatter(sp[:, 0], sp[:, 1], s=3, c='orange', label='seed-patch nodes')
    ax.scatter(rp[:, 0], rp[:, 1], s=2, c='deepskyblue', label='selected surface region')
    ax.scatter(g[:, 0], g[:, 1], s=4, c='limegreen', marker='s', label='GT target (depth px)')
    if len(at):
        ax.scatter(at[:, 0], at[:, 1], s=4, c='red', marker='x', label='ATS target points')
    ax.set_aspect('equal'); ax.set_xlabel('base x [m]'); ax.set_ylabel('base y [m]')
    ax.set_title(f"top-down: region↔GT {out['dist_mm']['region_to_gt']} mm, ATS↔GT {out['dist_mm']['ats_to_gt']} mm", fontsize=9)
    ax.legend(fontsize=7, loc='best', markerscale=3)
    fig.suptitle(f"{out['stem']} call t={out['t']} — yellow box = seed patch, white dashed = top-5 attention, "
                 f"green line = GT target, cyan = region, red x = ATS", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--offline-run', type=Path, required=True)
    ap.add_argument('--picks', nargs='+', required=True, help='stem:t')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--version', default='AG3S-final-panda-p0m')
    a = ap.parse_args()
    if os.environ.get('AG3S_PROJECT_ROOT') != str(ROOT):
        raise RuntimeError('source scripts/setup/env.sh first')
    man = json.loads((a.offline_run/'manifest.json').read_text())
    live = {}
    for line in (a.offline_run/'frames.jsonl').open():
        r = json.loads(line); live[(r['stem'], r['t'])] = r
    gt_dirs = {p.name[:-len('_gt.npz')]: p.parent for d in man['dataset'] for p in Path(d).glob('*_gt.npz')}
    noun_run = Path(man['noun_run'])
    head = (8, 6)
    ag3s, _, _ = build_ag3s(a.version, head)
    a.out.mkdir(parents=True, exist_ok=True)
    orig = ag3s._attention_surface_update_t78
    cap = {}

    def spy(observations, cameras, sphere_masks, render_masks, ats_points):
        cap['args'] = (observations, cameras, sphere_masks, render_masks, ats_points)
        return orig(observations, cameras, sphere_masks, render_masks, ats_points)

    ag3s._attention_surface_update_t78 = spy
    summary = []
    for pick in a.picks:
        stem, t_pick = pick.split(':'); t_pick = int(t_pick)
        ep = S.Episode(gt_dirs[stem], stem)
        with np.load(S.RUN/'episodes'/(stem+'_geom.npz'), allow_pickle=False) as z:
            geom = {k: z[k] for k in z.files}
        with np.load(S.RUN/'episodes'/(stem+'_calls.npz'), allow_pickle=False) as z:
            executed = z['actions_executed'][:, :, 6]
        with np.load(S.RUN/'episodes'/(stem+'.npz'), allow_pickle=False) as z:
            rgb_all = {'head': z['obs_agentview_image'], 'left_wrist': z['obs_robot0_eye_in_hand_image']}
        with np.load(noun_run/'noun_rows'/(stem+'_noun.npz'), allow_pickle=False) as z:
            noun = {k: z[k] for k in ('noun_image', 'obs_index', 'layers', 'noun_ok')}
        li = int(np.flatnonzero(noun['layers'] == head[0])[0])
        times = [int(t) for t in ep.call_frames]
        rows_of = {int(o): i for i, o in enumerate(geom['obs_index'])}
        openings = np.array([abs(float(geom['robot_state/head'][rows_of[t]][7]))/0.04 for t in times])
        closed, _, _ = physical_state(openings, executed)
        ag3s.reset()
        for k, t in enumerate(times):
            if t > t_pick:
                break
            v = noun['noun_image'][k, li, head[1]]
            grids = np.stack([v[j*256:(j+1)*256].reshape(16, 16) for j in range(2)])
            observations, fingers, rs9, _ = PR.build_observations(geom, rows_of[t], grids, 'unmirror', ag3s.robot_model)
            PR.apply_frame_state(ag3s, fingers, rs9)
            ag3s.set_grasp_active(bool(closed[k])); ag3s.set_grasp_latched(bool(closed[k]))
            cap.clear()
            cs, debug = ag3s.process_multi_debug(observations, phase=PR_phase(), active_manipulators=['left'], static_geometry=None)
        surf = dict(ag3s._surface_frame or {})
        t78 = surf.get('t78') or {}
        rec_live = {'why': surf.get('why'), 'n_selected_nodes': surf.get('n_selected_nodes'),
                    'seed_patches': surf.get('seed_patches'), 'target_check': t78.get('target_check')}
        jl = live[(stem, t_pick)]['surface']
        assert rec_live['why'] == jl.get('why') and rec_live['n_selected_nodes'] == jl.get('n_selected_nodes'), (rec_live, jl)
        obs_, cams_, sm_, rm_, ats_pts = cap['args']
        views, mask_src, _ = ag3s._surface_views(obs_, cams_, sm_, rm_)
        fr = ep.frame(t_pick)
        gr = rows_of[t_pick]
        # stored RGB is in the simulator (OpenGL) orientation; depth / labels are OpenCV -> flip both axes? check by GT
        rgb = {}
        for c in ('head', 'left_wrist'):
            img = rgb_all[c][gr]
            cand = {'as_is': img, 'flipud': img[::-1], 'rot180': img[::-1, ::-1]}
            rgb[c] = cand[RGB_ORIENT]
        ats = None if ats_pts is None else np.asarray(ats_pts, np.float64).reshape(-1, 3)
        out, draw3 = analyse(views, ats, rec_live, fr['labels'], rgb, stem, t_pick)
        out['mask_source'] = mask_src
        out['task'] = int(stem.split('_t')[1].split('_')[0]); out['prompt'] = ep.prompt
        figure(out, draw3, rgb, a.out/f'{stem}_t{t_pick}.png')
        for c in out['cameras'].values():
            c.pop('_draw', None)
        (a.out/f'{stem}_t{t_pick}.json').write_text(json.dumps(out, indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))
        summary.append(out)
        print('done', pick, out['dist_mm'], flush=True)
    (a.out/'summary.json').write_text(json.dumps(summary, indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))


RGB_ORIENT = os.environ.get('SL7C_RGB_ORIENT', 'flipud')


def PR_phase():
    from AG3S.adapters.panda import profile as pf
    return pf.manifest_entry()['client_phase']


if __name__ == '__main__':
    main()
