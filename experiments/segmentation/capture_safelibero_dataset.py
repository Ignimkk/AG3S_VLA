#!/usr/bin/env python3
"""Image-instant GT (geom-id segmentation, both cameras) for recorded SafeLIBERO VLA-only rollouts.

Every recorded observation t of an episode is re-rendered from the stored `sim_state_image`; the rendered RGB of both
cameras must equal the stored raw observation bit for bit, otherwise the run stops. Episodes of a task are replayed in
order on one env (`start_episode` per episode), as the source run did, because LIBERO's reset re-samples fixture
placement from the global RNG. Geom-id maps are stored in OpenCV orientation (rows flipped like `opencv_rgb_from_raw`),
-1 = no geom. Classes are resolved later from the stored instance table; nothing here is shown to a model.
"""
import hashlib, json, pathlib, sys, time, traceback
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'benchmark/runners'))
from benchmark.adapters.libero_panda import client_env
if client_env.needs_reexec():                      # py3.8 client side: no checkout PYTHONPATH, sanitized LD_LIBRARY_PATH
    client_env.reexec_with_client_environment()
import libero_pi05_client as base
import libero_safe_client as safe

RUN = pathlib.Path('/mnt/dev/AG3S_VLA-storage/results/benchmark/20261017T151126_38722284e2fa_sl5f-safelibero-LI-dev-LI-vla_only')
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
CAMS = (('head', 'agentview', 'obs_agentview_image'), ('left_wrist', 'robot0_eye_in_hand', 'obs_robot0_eye_in_hand_image'))


def sha(p):
    h = hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    if OUT is None:
        raise SystemExit('usage: capture_safelibero_dataset.py OUT_DIR [task ids...]')
    tasks = [int(x) for x in sys.argv[2:]] or [0, 1, 2, 3]
    safe.activate_safelibero()
    from libero.libero import benchmark
    from benchmark.adapters.libero_panda import task_target as tt
    OUT.mkdir(parents=True, exist_ok=False)
    suite = benchmark.get_benchmark_dict()['safelibero_spatial'](safety_level='I')
    rows = [json.loads(l) for l in (RUN/'episodes.jsonl').read_text().splitlines() if l.strip()]
    (OUT/'manifest.json').write_text(json.dumps({'source_run': str(RUN), 'source_manifest_sha256': sha(RUN/'run_manifest.json'),
        'script_sha256': sha(__file__), 'seed': 7, 'tasks': tasks,
        'gt': 'rendered at sim_state_image of every recorded observation; RGB of both cameras verified bit-identical',
        'orientation': 'OpenCV (rows flipped)'}, indent=2)+'\n')
    report = []
    for tid in tasks:
        env = None
        try:
            task = suite.get_task(tid)
            env, _ = base.make_task_env(task, 7, False)
            inits = suite.get_task_init_states(tid)
            bddl = safe.SAFELIBERO_ROOT/'libero/libero/bddl_files'/task.problem_folder/task.bddl_file
            target, destination = tt.goal_roles(tt.parse_bddl(bddl))
            for row in sorted((r for r in rows if r['task_id'] == tid), key=lambda r: r['episode_idx']):
                ei = int(row['episode_idx']); stem = 'safelibero_spatial_t{}_e{}'.format(tid, ei); t0 = time.time()
                base.start_episode(env, inits, ei)
                model = env.sim.model
                names = [model.body_id2name(i) or '' for i in range(model.nbody)]
                with np.load(RUN/'episodes'/(stem+'.npz'), allow_pickle=False) as z:
                    raw = {c: z[k] for c, _, k in CAMS}; chunk_t = z['chunk_t']
                with np.load(RUN/'episodes'/(stem+'_geom.npz'), allow_pickle=False) as g:
                    states = g['sim_state_image']; indices = g['obs_index']
                    if names != [str(n) for n in g['model_body_names']] or not np.array_equal(model.geom_bodyid, g['model_geom_bodyid']):
                        raise ValueError('body/geom table mismatch')
                if not np.array_equal(indices, np.arange(len(states))) or len(states) != len(raw['head']):
                    raise ValueError('nonconsecutive observation indices')
                obstacles = []
                for joint in model.joint_names:
                    if 'obstacle' in joint:
                        n = joint.replace('_joint0', '')
                        obstacles.append({'name': n, 'geom_ids': [gi for gi in range(model.ngeom) if (model.body_id2name(model.geom_bodyid[gi]) or '').startswith(n)],
                                          'body_id': int(model.body_name2id(n + '_main' if (n + '_main') in model.body_names else n))})
                seg = {c: np.empty((len(states), 256, 256), np.int16) for c, _, _ in CAMS}
                active = None
                for i, state in enumerate(states):
                    sim = env.sim; sim.set_state_from_flattened(state.copy()); sim.forward()
                    for c, cam, _ in CAMS:
                        rgb = sim.render(camera_name=cam, height=256, width=256)
                        if not np.array_equal(rgb, raw[c][i]):
                            raise ValueError('RGB mismatch {} frame {}'.format(c, i))
                        s = sim.render(camera_name=cam, height=256, width=256, segmentation=True)
                        gid = np.where(s[..., 0] == 5, s[..., 1], -1).astype(np.int16)
                        seg[c][i] = gid[::-1]
                    if i == int(chunk_t[0]):
                        active = [o['name'] for o in obstacles
                                  if env.sim.data.body_xpos[o['body_id']][2] > 0 and abs(env.sim.data.body_xpos[o['body_id']][0]) < 0.5
                                  and abs(env.sim.data.body_xpos[o['body_id']][1]) < 0.5]
                inst = {'target': target, 'destination': destination, 'active_obstacles': active,
                        'target_geoms': tt.instance_geom_ids(target, model.geom_bodyid, names),
                        'destination_geoms': tt.instance_geom_ids(destination, model.geom_bodyid, names),
                        'obstacle_geoms': {o['name']: o['geom_ids'] for o in obstacles}}
                np.savez_compressed(OUT/(stem+'_gt.npz'), obs_index=indices, call_frames=chunk_t, seg_head=seg['head'],
                                    seg_left_wrist=seg['left_wrist'], instances=json.dumps(inst), prompt=str(task.language))
                report.append({'stem': stem, 'status': 'ok', 'frames': len(states), 'calls': int(len(chunk_t)), 'active_obstacles': active,
                               'target': target, 'destination': destination, 'seconds': time.time()-t0})
                print(stem, 'ok', len(states), active, flush=True)
        except Exception as exc:
            report.append({'task': tid, 'status': 'error', 'error': repr(exc), 'traceback': traceback.format_exc()})
            print('task', tid, 'ERROR', repr(exc), flush=True)
        finally:
            if env is not None:
                env.close()
        (OUT/'report.json').write_text(json.dumps(report, indent=1)+'\n')
    if any(r['status'] != 'ok' for r in report):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
