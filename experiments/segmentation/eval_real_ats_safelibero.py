#!/usr/bin/env python3
"""Real Panda final-configuration ATS replayed on the recorded SafeLIBERO VLA-only rollouts, scored class-aware.

Grasp / latch state comes from **gripper physics only** (no HSR probe, no GT): per call k, from the previous chunk's last
executed gripper command and the measured finger opening at the call's image instant,

    closed_cmd = last executed command of chunk k-1 is "close" (> 0)
    settled    = |opening_k - opening_{k-1}| < 0.02
    holding    = closed_cmd and settled and opening_k > empty_below (Panda calibration 0.0253) and opening_k < 0.9
    active     = closed_cmd                                  -> set_grasp_active, set_grasp_latched
    (released when the last executed command opens again)

`--grasp none` keeps the earlier no-grasp values (the Object replay condition). The noun-row attention `L8H6` comes from
`safelibero_noun_rows.py`. The AG3S object is built from the final-configuration server flags; nothing from GT enters the call.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'benchmark/runners')); sys.path.insert(0, str(Path(__file__).resolve().parent))
import perception_replay as PR
import safelibero_scoring as S
from eval_real_ats_offline import build_ag3s, sha

EMPTY_BELOW = json.loads((ROOT/'robots/panda/description/gripper_calibration.json').read_text())['panda']['empty_below']
OPEN_ABOVE = 0.9
SETTLE = 0.02


def physical_state(opening, executed_gripper):
    """Per-call (closed_cmd, settled, holding) from physical quantities only. executed_gripper: (n_calls, 5) executed commands."""
    n = len(opening)
    closed, settled, holding = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    for k in range(n):
        closed[k] = k > 0 and executed_gripper[k-1][-1] > 0
        settled[k] = k > 0 and abs(opening[k]-opening[k-1]) < SETTLE
        holding[k] = closed[k] and settled[k] and EMPTY_BELOW < opening[k] < OPEN_ABOVE
    return closed, settled, holding


def surface_summary(ag3s):
    """SL7b — the attention-surface record of this call (`AG3S._surface_frame`), scalars only: was a region applied, why
    not (`grasp_hold` / `not_ats_target` / the selection's own reason), node count, target-check numbers. None when the
    version does not use the surface extent."""
    surf = getattr(ag3s, "_surface_frame", None)
    if not isinstance(surf, dict):
        return None
    t78 = surf.get("t78") or {}
    keep = lambda d: {k: v for k, v in (d or {}).items() if isinstance(v, (bool, int, float, str)) or v is None}
    return {**keep(surf), "region": keep(t78.get("region")), "target_check": keep(t78.get("target_check")),
            "hold": keep(t78.get("hold")), "hull": keep(t78.get("hull"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--noun-run', type=Path, required=True)
    ap.add_argument('--dataset', type=Path, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--grasp', choices=('none', 'physical'), required=True)
    ap.add_argument('--version', default='AG3S-final-panda')
    ap.add_argument('--head', default='8,6')
    ap.add_argument('--limit-episodes', type=int, default=0)
    ap.add_argument('--limit-calls', type=int, default=0)
    ap.add_argument('--protocol', type=Path, required=True)
    a = ap.parse_args()
    if os.environ.get('AG3S_PROJECT_ROOT') != str(ROOT):
        raise RuntimeError('source scripts/setup/env.sh first')
    head = tuple(int(x) for x in a.head.split(','))
    gt_dirs = {p.name[:-len('_gt.npz')]: p.parent for d in a.dataset for p in d.glob('*_gt.npz')}
    stems = sorted(gt_dirs, key=lambda s: (int(s.split('_t')[1].split('_')[0]), int(s.rsplit('_e', 1)[1])))
    if a.limit_episodes:
        stems = stems[:a.limit_episodes]
    ag3s, info, flags = build_ag3s(a.version, head)
    from AG3S.adapters.panda import profile as pf
    phase_name = pf.manifest_entry()['client_phase']
    a.out.mkdir(parents=True, exist_ok=False); (a.out/'clouds').mkdir()
    (a.out/'manifest.json').write_text(json.dumps({
        'kind': 'offline real-ATS replay on SafeLIBERO (perception only)', 'version': a.version, 'server_flags': flags,
        'attention': 'noun_target:L%dH%d from stored noun rows' % head, 'grasp_mode': a.grasp,
        'physical_rule': {'closed_cmd': 'last executed gripper command of the previous chunk > 0', 'settled_eps': SETTLE,
                          'empty_below': EMPTY_BELOW, 'open_above': OPEN_ABOVE, 'active': 'closed_cmd (set_grasp_active and set_grasp_latched)'},
        'source_run': str(S.RUN), 'noun_run': str(a.noun_run), 'dataset': [str(d) for d in a.dataset],
        'script_sha256': sha(__file__), 'protocol_sha256': sha(a.protocol), 'episodes': stems}, indent=1, default=str)+'\n')
    t_start = time.monotonic()
    with (a.out/'frames.jsonl').open('x') as out:
        for n, stem in enumerate(stems, 1):
            ep = S.Episode(gt_dirs[stem], stem)
            with np.load(S.RUN/'episodes'/(stem+'_geom.npz'), allow_pickle=False) as z:
                geom = {k: z[k] for k in z.files}
            with np.load(S.RUN/'episodes'/(stem+'_calls.npz'), allow_pickle=False) as z:
                executed = z['actions_executed'][:, :, 6]; call_obs = z['call_obs_index']
            with np.load(a.noun_run/'noun_rows'/(stem+'_noun.npz'), allow_pickle=False) as z:
                noun = {k: z[k] for k in ('noun_image', 'obs_index', 'layers', 'noun_ok')}
            layer_idx = np.flatnonzero(noun['layers'] == head[0])
            if len(layer_idx) != 1 or not np.array_equal(noun['obs_index'], call_obs) or not np.array_equal(call_obs, ep.call_frames):
                raise ValueError('join mismatch (noun layer / call frames) for '+stem)
            times = [int(t) for t in ep.call_frames]
            if a.limit_calls:
                times = times[:a.limit_calls]
            rows_of = {int(o): i for i, o in enumerate(geom['obs_index'])}
            openings = np.array([abs(float(geom['robot_state/head'][rows_of[t]][7]))/0.04 for t in times])
            closed, settled, holding = physical_state(openings, executed)
            ag3s.reset()
            kept, kept_ext = [], []
            for k, t in enumerate(times):
                if not noun['noun_ok'][k]:
                    raise ValueError('noun token missing '+stem)
                v = noun['noun_image'][k, layer_idx[0], head[1]]
                grids = np.stack([v[j*256:(j+1)*256].reshape(16, 16) for j in range(2)])
                observations, fingers, rs9, _ = PR.build_observations(geom, rows_of[t], grids, 'unmirror', ag3s.robot_model)
                PR.apply_frame_state(ag3s, fingers, rs9)
                if a.grasp == 'physical':
                    ag3s.set_grasp_active(bool(closed[k])); ag3s.set_grasp_latched(bool(closed[k]))
                t0 = time.perf_counter()
                cs, debug = ag3s.process_multi_debug(observations, phase=phase_name, active_manipulators=['left'], static_geometry=None)
                wall = time.perf_counter()-t0
                target = debug['grounding'].target
                pts = None if target is None else np.asarray(target.points, np.float64)
                trace = getattr(cs.esdf, 'exclusion_trace', None) or {}
                ext = None
                if len(trace.get('removed_ijk', ())):
                    ext = np.asarray(trace['origin'], np.float64)+np.asarray(trace['removed_ijk'], np.float64)*float(trace['voxel_size'])
                fr = ep.frame(t); gb = S.grid_of(fr)
                manip = cs.metrics.get('manipulated')
                row = {'stem': stem, 'task': int(stem.split('_t')[1].split('_')[0]), 'episode': int(stem.rsplit('_e', 1)[1]), 't': t,
                       'status': debug['grounding'].status.value, 'has_target': target is not None,
                       'n_target_points': 0 if pts is None else int(len(pts)), 'n_removed_voxels': 0 if ext is None else int(len(ext)),
                       'centroid': None if target is None else PR._jsonable(target.centroid),
                       'manipulated_state': None if not manip else str(manip.get('state')),
                       'physical': {'opening': float(openings[k]), 'closed_cmd': bool(closed[k]), 'settled': bool(settled[k]), 'holding': bool(holding[k])},
                       'grasp_signal_used': a.grasp == 'physical' and bool(closed[k]), 'wall_s': wall,
                       'target_cluster': S.score_points(fr, pts, gb)['3d'], 'erased_extent': S.score_points(fr, ext, gb)['3d'],
                       'meta': S.frame_meta(fr, gb[1]), 'surface': surface_summary(ag3s)}
                out.write(json.dumps(row, allow_nan=False, default=str)+'\n'); out.flush()
                kept.append(np.zeros((0, 3), np.float32) if pts is None else pts.astype(np.float32))
                kept_ext.append(np.zeros((0, 3), np.float32) if ext is None else ext.astype(np.float32))
            np.savez_compressed(a.out/'clouds'/(stem+'.npz'), call_frames=np.asarray(times), offsets=np.cumsum([0]+[len(x) for x in kept]),
                                target_points=np.concatenate(kept), ext_offsets=np.cumsum([0]+[len(x) for x in kept_ext]),
                                ext_points=np.concatenate(kept_ext), opening=openings, closed_cmd=closed, settled=settled, holding=holding)
            print('%d/%d %s elapsed=%.0fs' % (n, len(stems), stem, time.monotonic()-t_start), flush=True)


if __name__ == '__main__':
    main()
