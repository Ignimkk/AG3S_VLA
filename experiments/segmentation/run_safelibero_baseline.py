#!/usr/bin/env python3
"""Run one external segmenter on the recorded SafeLIBERO VLA-only rollouts (both cameras, every policy-call frame).

Input to the model: RGB (OpenCV orientation, 256x256) and a prompt derived from the task instruction only
(`phrase` = target noun phrase of "pick up the X and place ...", `instruction` = the whole sentence). No GT, depth,
robot state or object name is given. `track` providers receive every recorded frame 0..last call in time order
(causal) but masks are stored for the policy-call frames only. Masks are written packed per episode; scoring is a
separate script. Run with .venv-baselines, GPU inside the shared lock.
"""
import argparse, hashlib, json, re, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Baseline.paper_baselines.interface import TargetQuery, validate_results

RUN = Path(__import__('os').environ.get('AG3S_SL_SOURCE_RUN', '/mnt/dev/AG3S_VLA-storage/results/benchmark/20261017T151126_38722284e2fa_sl5f-safelibero-LI-dev-LI-vla_only'))  # nb1104: env override, default unchanged
CAMS = (('head', 'obs_agentview_image'), ('left_wrist', 'obs_robot0_eye_in_hand_image'))
PHRASE = re.compile(r'^\s*(?:pick up|pick|grab|pass me) the (?P<target>.+?)(?: and (?:place|put|bring|cut)\b.*|\s*[.!]?\s*$)', re.I)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def build(name, device, conf):
    if name == 'grounded_sam':
        from Baseline.paper_baselines.providers.grounded_sam import GroundedSAMProvider
        return GroundedSAMProvider(device=device)
    if name in ('grounded_sam2_frame', 'grounded_sam2_track'):
        from Baseline.paper_baselines.providers.grounded_sam2 import GroundedSAM2Provider
        return GroundedSAM2Provider(mode='track' if name.endswith('track') else 'per_frame', device=device)
    if name == 'yoloe':
        from Baseline.paper_baselines.providers.yoloe import YOLOEProvider
        return YOLOEProvider(device=device, conf=conf) if conf else YOLOEProvider(device=device)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--provider', required=True, choices=('grounded_sam', 'grounded_sam2_frame', 'grounded_sam2_track', 'yoloe'))
    ap.add_argument('--mode', required=True, choices=('phrase', 'instruction'))
    ap.add_argument('--dataset', type=Path, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--yoloe-conf', type=float, default=0.0)
    ap.add_argument('--limit-episodes', type=int, default=0)
    ap.add_argument('--limit-calls', type=int, default=0)
    a = ap.parse_args()
    gts = sorted(p for d in a.dataset for p in d.glob('*_gt.npz'))
    if a.limit_episodes:
        gts = gts[:a.limit_episodes]
    model = build(a.provider, a.device, a.yoloe_conf)
    track = a.provider.endswith('track')
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'masks').mkdir()
    (a.out/'manifest.json').write_text(json.dumps({
        'provider': model.provenance(), 'mode': a.mode, 'dataset': [str(d) for d in a.dataset], 'source_run': str(RUN),
        'script_sha256': sha(__file__), 'input': 'RGB (OpenCV orientation) + prompt from the instruction only',
        'track_input': 'all recorded frames 0..last call in time order, both cameras' if track else None,
        'yoloe_conf': a.yoloe_conf or None}, indent=1, default=str)+'\n')
    errors, rows, t0 = [], [], time.time()
    for path in gts:
        stem = path.name[:-len('_gt.npz')]
        with np.load(path, allow_pickle=False) as z:
            calls = [int(x) for x in z['call_frames']]
            prompt = str(z['prompt'])
        m = PHRASE.fullmatch(prompt)
        if not m:
            errors.append({'stem': stem, 'error': 'phrase parser failed: '+prompt})
            continue
        if a.limit_calls:
            calls = calls[:a.limit_calls]
        with np.load(RUN/'episodes'/(stem+'.npz'), allow_pickle=False) as z:
            raw = {c: z[k] for c, k in CAMS}
        query = TargetQuery(instruction=prompt, target_phrase=m.group('target').strip(), prompt_mode=a.mode, episode_id=stem)
        model.reset()
        frames = range(0, calls[-1]+1) if track else calls
        keep = set(calls)
        saved = {c: [] for c, _ in CAMS}
        info = []
        for t in frames:
            rgb = {c: np.ascontiguousarray(raw[c][t][::-1]) for c, _ in CAMS}
            try:
                res = model.predict(rgb, query)
                validate_results(res, rgb)
            except Exception as exc:
                errors.append({'stem': stem, 't': t, 'error': repr(exc), 'traceback': traceback.format_exc()})
                if t in keep:
                    for c, _ in CAMS:
                        saved[c].append(np.zeros((256, 256), bool))
                    info.append({'t': t, 'error': repr(exc)})
                continue
            if t in keep:
                for c, _ in CAMS:
                    saved[c].append(res[c].mask)
                info.append({'t': t, **{c: {'found': bool(res[c].found), 'score': None if res[c].score is None else float(res[c].score),
                                              'n_cand': 0 if res[c].boxes is None else int(len(res[c].boxes)),
                                              'source': res[c].extras.get('source', 'detect'),
                                              'box': None if res[c].boxes is None or res[c].extras.get('selected') is None
                                              or len(res[c].boxes) <= res[c].extras['selected'] else np.asarray(res[c].boxes[res[c].extras['selected']]).round(1).tolist()}
                                          for c, _ in CAMS}})
        np.savez_compressed(a.out/'masks'/(stem+'.npz'), call_frames=np.asarray(calls),
                            **{c: np.packbits(np.stack(saved[c]), axis=-1) for c, _ in CAMS})
        rows.append({'stem': stem, 'prompt': query.prompt, 'calls': len(calls), 'frames_run': len(list(frames)), 'info': info})
        (a.out/'frames.json').write_text(json.dumps(rows, default=str)+'\n')
        print(a.provider, a.mode, stem, len(calls), 'elapsed %.0fs' % (time.time()-t0), flush=True)
    (a.out/'summary_input.json').write_text(json.dumps({'episodes': len(rows), 'errors': errors, 'seconds': time.time()-t0}, indent=1)+'\n')
    if errors:
        print('ERRORS', len(errors), flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
