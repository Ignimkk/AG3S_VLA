#!/usr/bin/env python3
"""Target-noun attention rows for recorded SafeLIBERO VLA-only rollouts (no policy_records exist for these runs).

The policy request of call k is rebuilt exactly as the client sent it: images = client preprocessing of the raw
observation at chunk_t[k] (rotate 180 deg, resize_with_pad 224), state = policy_state[k], prompt = task description.
The noun-row reader is the one of `benchmark/runners/attention_audit.py --noun-rows` (same prefix pass and slice as the
served `noun_target:L<l>H<h>` source). Writes <out>/noun_rows/<stem>_noun.npz with the keys the Object run uses.
Run with .venv-openpi-live on the GPU inside the shared lock.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark.runners import attention_audit as AA
from benchmark.runners import launch_libero_pi05_only as launcher

RUN = Path('/mnt/dev/AG3S_VLA-storage/results/benchmark/20261017T151126_38722284e2fa_sl5f-safelibero-LI-dev-LI-vla_only')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--task-ids', type=int, nargs='+', default=[0, 1, 2, 3])
    ap.add_argument('--limit-calls', type=int, default=0)
    ap.add_argument('--limit-episodes', type=int, default=0)
    a = ap.parse_args()
    if os.environ.get('AG3S_PROJECT_ROOT') != str(ROOT):
        raise RuntimeError('source scripts/setup/env.sh first')
    rows = [json.loads(l) for l in (RUN/'episodes.jsonl').read_text().splitlines() if l.strip()]
    rows = sorted((r for r in rows if r['task_id'] in a.task_ids), key=lambda r: (r['task_id'], r['episode_idx']))
    if a.limit_episodes:
        rows = rows[:a.limit_episodes]
    from AG3S.ag3s.experiments.sources.pi05_attention import load_policy
    ckpt = launcher.default_checkpoint()
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'noun_rows').mkdir()
    policy = load_policy(AA.POLICY_CONFIG, str(ckpt))
    reader = AA.NounRowReader(policy, config=AA.POLICY_CONFIG, layers=AA._layer_range('2-17'))
    index = []
    t0 = time.time()
    for r in rows:
        stem = 'safelibero_spatial_t{}_e{}'.format(r['task_id'], r['episode_idx'])
        with np.load(RUN/'episodes'/(stem+'.npz'), allow_pickle=False) as z:
            ep = {k: z[k] for k in ('chunk_t', 'policy_state', 'obs_agentview_image', 'obs_robot0_eye_in_hand_image')}
        n = len(ep['chunk_t']) if not a.limit_calls else a.limit_calls
        keys = ('noun_image', 'noun_mass', 'rows', 'token_index', 'noun_ok', 'call_k', 'chunk_t', 'obs_index', 'seconds')
        acc = {k: [] for k in keys}
        for k in range(n):
            t = int(ep['chunk_t'][k])
            img, wrist = AA.client_images(ep['obs_agentview_image'][t], ep['obs_robot0_eye_in_hand_image'][t], 224)
            req = {'observation/image': img, 'observation/wrist_image': wrist,
                   'observation/state': np.asarray(ep['policy_state'][k], np.float64), 'prompt': str(r['task_description'])}
            res = reader.read(req)
            ok = bool(res['ok'])
            acc['noun_ok'].append(ok)
            acc['noun_image'].append(res['image'] if ok else None)
            acc['noun_mass'].append(np.stack([res['mass'][m] for m in ('image_mass', 'text_mass', 'bos', 'self')]) if ok else None)
            acc['rows'].append(res['rows'] if ok else np.full(3, -1, np.int32))
            acc['token_index'].append(-1 if res['lookup']['token_index'] is None else int(res['lookup']['token_index']))
            acc['call_k'].append(k); acc['chunk_t'].append(t); acc['obs_index'].append(t); acc['seconds'].append(res['seconds'])
            if k == 0:
                first = res['lookup']
        shape = {key: next((v.shape for v in acc[key] if v is not None), None) for key in ('noun_image', 'noun_mass')}
        if shape['noun_image'] is None:
            raise ValueError('{}: no target noun found in {!r}: {}'.format(stem, r['task_description'], first))
        for key in ('noun_image', 'noun_mass'):
            acc[key] = [np.zeros(shape[key], np.float32) if v is None else v for v in acc[key]]
        arrays = {k: np.asarray(v) for k, v in acc.items()}
        arrays['noun_image'] = arrays['noun_image'].astype(np.float32)
        arrays['layers'] = np.asarray(reader.layers, np.int64)
        np.savez_compressed(a.out/'noun_rows'/(stem+'_noun.npz'), **arrays)
        index.append({'stem': stem, 'calls': n, 'noun_ok': int(sum(acc['noun_ok'])), 'lookup_first': first,
                      'prompt': r['task_description'], 'source_npz_sha256': sha(RUN/'episodes'/(stem+'.npz'))})
        print(stem, n, 'ok', sum(acc['noun_ok']), 'elapsed %.0fs' % (time.time()-t0), flush=True)
    (a.out/'noun_index.json').write_text(json.dumps({'source_run': str(RUN), 'checkpoint': str(ckpt), 'script_sha256': sha(__file__),
        'request': 'rebuilt from the stored raw observations (client preprocessing), policy_state, task description',
        'episodes': index}, indent=1, default=str)+'\n')


if __name__ == '__main__':
    main()
