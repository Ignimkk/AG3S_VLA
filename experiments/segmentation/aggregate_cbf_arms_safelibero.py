#!/usr/bin/env python3
"""Aggregate the simple-CBF-QP closed loops on SafeLIBERO spatial Level I dev (20 episodes) against the VLA-only reference.

Primary metrics (fixed before results, the official SafeLIBERO definitions of Baseline.paper_baselines.closed_loop.metrics):
collision episodes (sum|dpos| of an active obstacle > 1 mm after the wait), official success, safe success (success and no collision).
"""
import argparse, json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Baseline.paper_baselines.closed_loop import metrics as M
from AG3S.ag3s.experiments.common import figstyle

RESULTS = Path('/mnt/dev/AG3S_VLA-storage/results/benchmark')
REF = RESULTS/'20261017T151126_38722284e2fa_sl5f-safelibero-LI-dev-LI-vla_only'
LABEL = {'multilink_cbf': 'Multi-link CBF-QP', 'knows': 'KNOWS', 'aegis': 'AEGIS-style'}


def arm_name(run):
    s = run.name.split('cbf20261103-')[1].split('-LI-')[0]        # e.g. multilink_cbf-gt / knows
    base = s[:-3] if s.endswith('-gt') else s
    return '%s (%s)' % (LABEL.get(base, base), 'GT obstacle' if s.endswith('-gt') else 'detector obstacle'), base, s.endswith('-gt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='cbf20261103')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    rows = []
    ref = M.summarize(REF)
    rows.append(('VLA only (reference, same 20 episodes)', ref, None))
    for run in sorted(RESULTS.glob('*%s-*' % a.prefix)):
        if not (run/'episodes.jsonl').exists():
            continue
        try:
            if len(M.load_rows(run)) < 20:
                print('skip (incomplete)', run.name); continue
            s = M.summarize(run, REF)
        except Exception as e:
            print('skip', run.name, repr(e)); continue
        rows.append((arm_name(run)[0], s, str(run)))
    a.out.mkdir(parents=True, exist_ok=True)
    table = []
    for name, s, run in rows:
        o = s['safelibero_official']; mc = s.get('method_calls') or {}; p = s.get('paired_vs_ref')
        table.append({'arm': name, 'run': run, 'n': o['n'], 'official_success': o['n_success'], 'collision_episodes': o['n_collision'],
                      'safe_success': o['n_safe_success'], 'collision_free_rate': o['CAR'],
                      'intervention_rate_rows': mc.get('intervention_rate_rows'), 'h_neg_steps': mc.get('h_neg_steps'),
                      'method_ms_p50': mc.get('method_ms_p50'),
                      'collided_only_ref': None if not p else len(p['collided']['ref_only']),
                      'collided_only_here': None if not p else len(p['collided']['here_only']),
                      'success_gained': None if not p else len(p['official_success']['here_only']),
                      'success_lost': None if not p else len(p['official_success']['ref_only'])})
    (a.out/'cbf_arms.json').write_text(json.dumps(table, indent=1)+'\n')
    f = lambda x, p=2: '-' if x is None else ('%.'+str(p)+'f') % x
    L = ['| arm | 공식 성공 | 충돌 episode | 안전 성공 | 충돌 회피율 | 개입한 행 비율 | 기준 대비 충돌 해소 / 새 충돌 | 기준 대비 성공 얻음 / 잃음 | 서버 ms (p50) |', '|---|---:|---:|---:|---:|---:|---|---|---:|']
    for t in table:
        L.append('| %s | %d/20 | %d/20 | %d/20 | %s | %s | %s / %s | %s / %s | %s |' % (
            t['arm'], t['official_success'], t['collision_episodes'], t['safe_success'], f(t['collision_free_rate']),
            f(t['intervention_rate_rows']), t['collided_only_ref'] if t['collided_only_ref'] is not None else '-',
            t['collided_only_here'] if t['collided_only_here'] is not None else '-',
            t['success_gained'] if t['success_gained'] is not None else '-', t['success_lost'] if t['success_lost'] is not None else '-', f(t['method_ms_p50'], 0)))
    (a.out/'cbf_arms.md').write_text('\n'.join(L)+'\n')
    print('\n'.join(L))
    figstyle.use_korean()
    fig, ax = plt.subplots(figsize=(11, 0.9*len(table)+2))
    names = [t['arm'] for t in table][::-1]
    ys = range(len(names))
    for k, (key, col, lab) in enumerate([('collision_episodes', '#d95f02', '충돌 episode'), ('official_success', '#1b9e77', '공식 성공'), ('safe_success', '#2a6df4', '안전 성공')]):
        ax.barh([y+(k-1)*0.27 for y in ys], [t[key] for t in table][::-1], height=0.25, color=col, label=lab)
        for y, t in zip(ys, table[::-1]):
            ax.text(t[key]+0.2, y+(k-1)*0.27, str(t[key]), va='center', fontsize=8)
    ax.set_yticks(list(ys)); ax.set_yticklabels(names, fontsize=9); ax.set_xlim(0, 22)
    ax.set_xlabel('episode 수 (20 = SafeLIBERO spatial Level I dev, task 0–3 × init 0–4)'); ax.legend(loc='lower right', fontsize=8)
    ax.set_title('단순 CBF-QP 필터 closed loop: 충돌 · 성공 (VLA 단독 기준은 맨 아래)')
    fig.tight_layout(); fig.savefig(a.out/'fig_cbf_arms.png', dpi=140); plt.close(fig)


if __name__ == '__main__':
    main()
